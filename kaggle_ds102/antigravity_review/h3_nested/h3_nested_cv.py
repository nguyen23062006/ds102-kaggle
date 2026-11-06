"""H3 nested: class-weight selection evaluated with genuinely nested grouped CV.

Hosted Google Colab only. No pretrained assets, no external data, dev/test are
never used (not fitted, not scored) by this script, dev is never merged into train.

Design (frozen before any run; see README_H3_NESTED.md and h3_selection.py):
  Outer: the SAME near-duplicate cluster folds as V53/V56/V57/V60
         (groups.npz from /content/ds102_v53/run, StratifiedGroupKFold(3,
         shuffle=True, random_state=2053)).
  Inner: inside each outer-train partition only, StratifiedGroupKFold(3,
         shuffle=True, random_state=4053+outer_fold) on the same cluster ids.
         For each inner split, the V11 recipe is refit from scratch on
         inner-train rows: FeaturesV2 TF-IDF + SVD128 + LinearSVC/NB-SVM,
         CatBoost text (500), latent CatBoost (800). Predictions on
         inner-validation rows form inner OOF for that outer fold.
  Select: weights chosen ONLY from inner OOF (h3_selection.select_weights).
  Score:  applied exactly once to the outer-validation probabilities of the
         unchanged V11 recipe (reused, manifest-checked V53 caches). Baseline
         = same probabilities with fixed V11 weights (1, 1.2, 1.2, .9).

Outer-validation labels are read only after the weights for that fold have
been selected and written to disk.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path


def guard():
    if platform.system() != 'Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Hosted Google Colab only; local fitting disabled.')


if __name__ == '__main__':
    guard()  # Fail before importing any ML package on a local machine.

HERE = Path(__file__).resolve().parent
PARENT_SOURCE = Path('/content/ds102_v53/source')
PARENT_RUN = Path('/content/ds102_v53/run')
sys.path.insert(0, str(PARENT_SOURCE))
sys.path.insert(0, str(HERE))

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from catboost_v3 import make_cat, cat_features
from train_submit import load_data, metrics
from improved_features import prepare
from review_common import max_jaccard, class1_strata, macro
from codex_checked_common import paired_cluster_bootstrap, group_summary
from experiment_v53 import Representations, fit_svm, tree, checked
import h3_selection as sel

CONFIG = dict(
    experiment='H3 nested class-weight selection',
    recipe='V11: .35*SVM/NB-SVM + .35*CatText500 + .30*latent SVD128 CatBoost800',
    outer=dict(protocol='cluster', groups='V53 groups.npz (Jaccard>=.7 components)',
               splitter='StratifiedGroupKFold', n_splits=3, shuffle=True, random_state=2053,
               probabilities='reuse V53 cluster fold caches svm/cat/latent (p0), manifest-checked'),
    inner=dict(splitter='StratifiedGroupKFold', n_splits=3, shuffle=True,
               random_state='4053 + outer_fold', groups='same cluster ids restricted to outer-train',
               refit='all transforms and models refit on inner-train rows only'),
    model_seed=102, svd_dimensions=128, svm_C=.5, svm_temperature=3,
    cat_text_iterations=500, latent_iterations=800,
    baseline_weights=list(sel.BASELINE_WEIGHTS),
    grid=dict(w0=[1.0], w1=list(sel.W1_GRID), w2=list(sel.W2_GRID), w3=list(sel.W3_GRID),
              n_candidates=315, objective='macro-F1 on inner OOF',
              ties='min L1 distance to baseline weights, then lexicographic (w1,w2,w3)'),
    gates=dict(cluster_min_delta=.003, min_positive_folds=2, max_fold_drop=.005,
               min_novel_fake_f1_delta=0., bootstrap_q025_positive=True,
               bootstrap=dict(kind='paired cluster', resamples=2000, seed=2053)),
    template_threshold=.6,
    expected_baseline_pooled_macro=0.7240841964761913,
    pretrained=False, external_data=False, dev_merged_into_train=False,
    dev_or_test_used=False, automatic_kaggle_submission=False,
)
W_BASE = np.asarray(sel.BASELINE_WEIGHTS)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_savez(path, **arrays):
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def fit_components(fit_frame, predict_frame, smoke):
    """Refit the V11 recipe on fit_frame only; predict label-blind predict_frame."""
    guard()
    assert 'spam_label' not in predict_frame.columns
    y = fit_frame.spam_label.to_numpy()
    assert set(np.unique(y)) == {0, 1, 2, 3}
    n = len(predict_frame)
    enc = Representations(smoke=smoke)
    x, m = enc.fit_base(fit_frame)                 # vectorizers + SVD: fit rows only
    v, vm = enc.base.transform(predict_frame)
    svm = checked(fit_svm(x, m, [(v, vm)], y)[0], n)
    model = make_cat('SqrtBalanced', iterations=20 if smoke else 500)
    model.set_params(thread_count=2)
    model.fit(cat_features(fit_frame), y)
    cat = checked(model.predict_proba(cat_features(predict_frame)), n)
    del model
    gc.collect()
    model = tree(20 if smoke else 800)
    model.fit(enc.latent(fit_frame, x), y)
    assert np.array_equal(model.classes_, [0, 1, 2, 3])
    lat = checked(model.predict_proba(enc.latent(predict_frame, v)), n)
    del model, enc, x, m, v, vm
    gc.collect()
    return dict(svm=svm, cat=cat, latent=lat)


def recipe(parts, n):
    return checked(.35 * parts['svm'] + .35 * parts['cat'] + .3 * parts['latent'], n)


def components_cached(path, mid, tr, va, fit):
    """Load a fold checkpoint (verified) or compute it with fit() and save."""
    if path.exists():
        z = np.load(path, allow_pickle=False)
        assert z['manifest_id'].item() == mid
        assert np.array_equal(z['tr'], tr) and np.array_equal(z['va'], va)
        return {k: checked(z[k], len(va)) for k in ['svm', 'cat', 'latent']}
    parts = fit()
    atomic_savez(path, manifest_id=mid, tr=tr, va=va, **parts)
    return parts


def main(args):
    guard()
    start = time.time()
    inputs, out = Path(args.inputs), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ---- provenance: identical parent run, inputs and library versions ----
    pm = json.loads((PARENT_RUN / 'run_manifest.json').read_text())
    parent_mid = hashlib.sha256(json.dumps(pm, sort_keys=True).encode()).hexdigest()
    for name, sha in pm['sources'].items():
        assert digest(PARENT_SOURCE / name) == sha, name
    expected = json.loads((inputs / 'inputs_manifest.json').read_text())['files']
    assert expected == pm['inputs']
    for name, sha in expected.items():
        assert digest(inputs / name) == sha, name
    assert {k: importlib.metadata.version(k) for k in pm['versions']} == pm['versions']
    manifest = dict(config=CONFIG, smoke=args.smoke, allow_outer_refit=args.allow_outer_refit,
                    source_sha256={p.name: digest(p) for p in [Path(__file__), HERE / 'h3_selection.py']},
                    parent_manifest_sha256=digest(PARENT_RUN / 'run_manifest.json'),
                    parent_groups_sha256=digest(PARENT_RUN / 'groups.npz'),
                    inputs=expected, versions=pm['versions'])
    mid = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    mp = out / 'run_manifest.json'
    if mp.exists():
        assert json.loads(mp.read_text()) == manifest, 'Changed source/config: use a new output directory.'
    mp.write_text(json.dumps(manifest, indent=2))

    report = dict(stage='initializing', manifest_id=mid, config=CONFIG, smoke=args.smoke,
                  execution='hosted Colab CPU', submission_created=False,
                  best_user_reported_kaggle=.77436, folds={},
                  caveat=('V11 weights were chosen historically on these training rows, which favours the '
                          'baseline. Grouped folds are a stress test of adaptively developed recipes, not '
                          'untouched data. Bootstrap is conditional, not a post-selection guarantee.'))
    name = 'metrics_h3_nested_smoke.json' if args.smoke else 'metrics_h3_nested.json'

    def status(stage, **kw):
        report.update(stage=stage, elapsed_seconds=time.time() - start, **kw)
        (out / name).write_text(json.dumps(report, indent=2))
        print(stage, json.dumps({k: v for k, v in kw.items() if k != 'fold_report'}), flush=True)

    try:
        data = load_data(inputs / 'data')
        train = prepare(data['train'])           # dev/test are not prepared or used
        y_all = train.spam_label.to_numpy()
        groups_all = np.load(PARENT_RUN / 'groups.npz')['groups']
        assert len(groups_all) == len(y_all)
        if args.smoke:
            rows = np.sort(np.concatenate([np.flatnonzero(y_all == c)[:150] for c in range(4)]))
            train = train.iloc[rows].reset_index(drop=True)
            y, groups = y_all[rows], groups_all[rows]
        else:
            y, groups = y_all, groups_all
        report['group_info'] = group_summary(groups, y)

        n = len(y)
        outer_base = np.zeros((n, 4))
        outer_fold = np.full(n, -1)
        template = np.zeros(n, bool)
        base_pred = np.full(n, -1)
        cand_pred = np.full(n, -1)
        deltas = []
        outer = StratifiedGroupKFold(3, shuffle=True, random_state=2053)
        for f, (tr, va) in enumerate(outer.split(np.zeros(n), y, groups)):
            assert not set(groups[tr]) & set(groups[va])
            assert set(y[tr]) == set(y[va]) == {0, 1, 2, 3}
            outer_fold[va] = f
            folder = out / f'outer{f}'
            folder.mkdir(parents=True, exist_ok=True)

            # ---- 1. inner OOF from outer-train rows only ----
            inner_prob = np.zeros((len(tr), 4))
            inner_seen = np.zeros(len(tr), bool)
            inner = StratifiedGroupKFold(3, shuffle=True, random_state=4053 + f)
            for k, (a, b) in enumerate(inner.split(np.zeros(len(tr)), y[tr], groups[tr])):
                itr, iva = tr[a], tr[b]
                assert not set(groups[itr]) & set(groups[iva])
                assert not (set(itr) | set(iva)) & set(va), 'inner rows must exclude outer validation'
                assert set(y[itr]) == set(y[iva]) == {0, 1, 2, 3}
                status('fit_inner', outer_fold=f + 1, inner_fold=k + 1, rows_fit=len(itr), rows_predict=len(iva))
                parts = components_cached(
                    folder / f'inner{k}.npz', mid, itr, iva,
                    lambda: fit_components(train.iloc[itr], train.iloc[iva].drop(columns='spam_label'), args.smoke))
                inner_prob[b] = recipe(parts, len(iva))
                inner_seen[b] = True
            assert inner_seen.all()

            # ---- 2. select weights on inner OOF; persist BEFORE touching outer labels ----
            weights, info = sel.select_weights(inner_prob, y[tr])
            assert abs(info['baseline_inner_macro'] - macro(y[tr], sel.weighted_labels(inner_prob, W_BASE))) < 1e-12
            (folder / 'selected_weights.json').write_text(json.dumps(dict(manifest_id=mid, **info), indent=2))
            atomic_savez(folder / 'inner_oof.npz', manifest_id=mid, tr=tr, prob=inner_prob)

            # ---- 3. outer-validation probabilities of the unchanged V11 recipe ----
            shared = PARENT_RUN / 'cluster' / f'fold{f}'
            reuse = (not args.smoke) and all((shared / (c + '.npz')).exists() for c in ['svm', 'cat', 'latent'])
            if reuse:
                z = np.load(shared / 'indices.npz')
                assert np.array_equal(z['tr'], tr) and np.array_equal(z['va'], va)
                parts = {}
                for c in ['svm', 'cat', 'latent']:
                    z = np.load(shared / (c + '.npz'), allow_pickle=False)
                    assert z['manifest_id'].item() == parent_mid
                    parts[c] = checked(z['p0'], len(va))
                source = 'reused V53 cache'
            else:
                if not (args.smoke or args.allow_outer_refit):
                    raise RuntimeError(f'V53 outer cache missing for fold {f}; rerun with --allow-outer-refit.')
                status('fit_outer', outer_fold=f + 1, rows_fit=len(tr), rows_predict=len(va))
                parts = components_cached(
                    folder / 'outer.npz', mid, tr, va,
                    lambda: fit_components(train.iloc[tr], train.iloc[va].drop(columns='spam_label'), args.smoke))
                source = 'refit on outer-train'
            outer_base[va] = recipe(parts, len(va))

            # ---- 4. evaluate exactly once on outer validation ----
            base_pred[va] = sel.weighted_labels(outer_base[va], W_BASE)
            cand_pred[va] = sel.weighted_labels(outer_base[va], weights)
            template[va] = max_jaccard(train.comment.iloc[va].tolist(),
                                       train.comment.iloc[tr[y[tr] == 1]].tolist()) >= CONFIG['template_threshold']
            mb, mc = macro(y[va], base_pred[va]), macro(y[va], cand_pred[va])
            deltas.append(mc - mb)
            report['folds'][str(f)] = dict(outer_probabilities=source, inner=info,
                                           outer_baseline_macro=mb, outer_candidate_macro=mc, delta=mc - mb,
                                           outer_label_changes=int((base_pred[va] != cand_pred[va]).sum()))
            status('outer_fold_complete', outer_fold=f + 1, fold_deltas=deltas)

        assert (outer_fold >= 0).all()
        b, c = base_pred, cand_pred
        pooled_base = macro(y, b)
        if not args.smoke and all(v['outer_probabilities'] == 'reused V53 cache' for v in report['folds'].values()):
            assert abs(pooled_base - CONFIG['expected_baseline_pooled_macro']) < 1e-9, pooled_base
        delta = macro(y, c) - pooled_base
        novel = class1_strata(y, c, template)['novel']['f1'] - class1_strata(y, b, template)['novel']['f1']
        boot = paired_cluster_bootstrap(y, b, c, groups, n=200 if args.smoke else 2000, seed=2053)
        g = CONFIG['gates']
        gates = dict(macro=bool(delta >= g['cluster_min_delta']),
                     folds=bool(sum(d > 0 for d in deltas) >= g['min_positive_folds']),
                     worst_fold=bool(min(deltas) >= -g['max_fold_drop']),
                     novel_fake=bool(novel >= g['min_novel_fake_f1_delta']),
                     bootstrap=bool(boot['q025'] > 0))
        report.update(baseline=metrics(y, b), candidate=metrics(y, c), delta=delta, fold_deltas=deltas,
                      novel_fake_f1_delta=novel, bootstrap=boot, gates=gates,
                      total_outer_label_changes=int((b != c).sum()))
        atomic_savez(out / ('oof_smoke.npz' if args.smoke else 'outer_oof.npz'), manifest_id=mid, y=y,
                     folds=outer_fold, prob=outer_base, baseline=b, candidate=c, template=template)
        if args.smoke:
            status('smoke_passed', eligible_for_selection=False)
            return
        if all(gates.values()):
            # Weights for a future full-train model: same rule on the outer OOF, whose rows were
            # each predicted by a model not trained on them. Not applied to dev/test here.
            final_w, final_info = sel.select_weights(outer_base, y)
            status('passed_all_gates_weights_only', final_weights=final_w.tolist(), final_selection=final_info)
        else:
            status('rejected_on_nested_cluster')
    except BaseException as exc:
        (out / 'error.txt').write_text(traceback.format_exc())
        status('error', error=str(exc))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', required=True, help='/content/ds102_v53/inputs (manifest-checked)')
    p.add_argument('--output', required=True, help='new directory, e.g. /content/ds102_h3_nested/run')
    p.add_argument('--smoke', action='store_true', help='600-row API check, 20 trees, SVD16; not eligible')
    p.add_argument('--allow-outer-refit', action='store_true',
                   help='refit outer folds if V53 caches are missing (adds ~3 outer fits)')
    main(p.parse_args())
