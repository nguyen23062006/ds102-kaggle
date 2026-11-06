"""E1 + E2 + E3 in one hosted-Colab run (no pretrained, train-only fitting).

E1  near-duplicate-cluster-grouped CV: v5 recipe vs v6 recipe, same folds.
E2  --dedup-weight: within-class sample weight 1/sqrt(cluster size) for all
    three components (SVM/NB-SVM, CatBoost text, latent CatBoost).
E3  every fold model also predicts dev/test; averaging the K fold models gives
    inference on the same score scale the weights/temperature were tuned on.

Usage on Colab (source dir must contain the v1-v6 modules + claude_review/*):
  python cluster_cv.py --seed 102                 # E1 (+E3 outputs)
  python cluster_cv.py --seed 102 --dedup-weight  # E2 (+E3 outputs)
Checkpoints per fold; re-running resumes. Writes only under --out.
Nothing here overwrites output_v1..v6.
"""
import argparse, gc, json, os, platform, time, traceback
from pathlib import Path
import sys as _sys
_sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1])]
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC
from review_common import (W_V5, W_V6, near_dup_groups, max_jaccard, class_f1, macro,
                           class1_strata, group_bootstrap_delta)


def hosted_guard():
    if platform.system() != 'Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Hosted Google Colab required; local training disabled.')


def dedup_weights(y, groups):
    """1/sqrt(cluster size), renormalised so every class keeps its original total weight."""
    size = np.bincount(groups)[groups].astype(float)
    w = 1 / np.sqrt(size)
    for c in range(4):
        m = y == c; w[m] *= m.sum() / w[m].sum()
    return w


def svm_component(train, frames, sw):
    """Same recipe as train_submit_v2.fit_predict (RECIPE), plus optional sample weights."""
    from train_submit_v2 import RECIPE
    from train_submit import make_model, SEED
    from improved_features import FeaturesV2, nb_ratios
    y = train.spam_label.to_numpy(); f = FeaturesV2(); xt, mt = f.fit_transform(train)
    tf = [f.transform(d) for d in frames]; res = [np.zeros((len(d), 4)) for d in frames]
    for cfg in RECIPE:
        if cfg['kind'] == 'svm':
            m = make_model(cfg, y).fit(sparse.hstack([xt, mt], format='csr'), y, sample_weight=sw)
            for j, (xv, mv) in enumerate(tf):
                res[j] += cfg['ensemble_weight'] * m.decision_function(sparse.hstack([xv, mv], format='csr'))
        else:
            # NB ratios stay unweighted (identical to v2) to isolate the effect of sample weights on the fit
            for c, r in enumerate(nb_ratios(xt, y)):
                m = LinearSVC(C=cfg['C'], class_weight='balanced', dual=True, max_iter=30000, random_state=SEED)
                m.fit(sparse.hstack([xt.multiply(r), mt], format='csr'), (y == c).astype(int), sample_weight=sw)
                for j, (xv, mv) in enumerate(tf):
                    res[j][:, c] += cfg['ensemble_weight'] * m.decision_function(
                        sparse.hstack([xv.multiply(r), mv], format='csr'))
    return res


def cat_component(train, frames, sw):
    from catboost_v3 import make_cat, cat_features
    m = make_cat('SqrtBalanced', iterations=500); m.set_params(thread_count=2)
    m.fit(cat_features(train), train.spam_label.to_numpy(), sample_weight=sw, verbose=False)
    return [m.predict_proba(cat_features(d)) for d in frames]


def latent_component(train, frames, sw):
    from latent_v6 import LatentFeatures, make_model
    f = LatentFeatures(); xt = f.transform(train, fit=True)
    m = make_model(); m.fit(xt, train.spam_label.to_numpy(), sample_weight=sw, verbose=False)
    return [m.predict_proba(f.transform(d), ntree_end=800) for d in frames]


def recipes(svm, cat, lat):
    base = .5 * softmax(3 * svm, axis=1) + .5 * cat
    return {'v5': (base * W_V5).argmax(1), 'v6': ((.7 * base + .3 * lat) * W_V6).argmax(1),
            'v5_noweight': base.argmax(1)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='/content/ds102_data')
    ap.add_argument('--out', default='/content/claude_review_out')
    ap.add_argument('--seed', type=int, default=102)
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--thr', type=float, default=.6)
    ap.add_argument('--dedup-weight', action='store_true')
    a = ap.parse_args(argv)
    hosted_guard()
    np.random.seed(a.seed)
    from train_submit import load_data
    from improved_features import prepare
    tag = f"{'e2' if a.dedup_weight else 'e1'}_seed{a.seed}_k{a.folds}"
    out = Path(a.out) / tag; out.mkdir(parents=True, exist_ok=True)
    def status(stage, **kv):
        (out / 'status.json').write_text(json.dumps(dict(stage=stage, time=time.time(), **kv), indent=2))
    try:
        data = load_data(Path(a.data_dir))
        train, dev, test = [prepare(data[k]) for k in ['train', 'dev', 'test']]
        y = train.spam_label.to_numpy(); ydev = dev.spam_label.to_numpy()
        frames_ext = [dev.drop(columns='spam_label'), test]
        status('groups')
        groups = near_dup_groups(train.comment.tolist(), thr=a.thr)
        gsize = np.bincount(groups)
        group_info = {'n_groups': int(len(gsize)), 'largest': int(gsize.max()),
                      'rows_in_groups_ge2': int((gsize[groups] >= 2).sum()),
                      'class1_rows_in_groups_ge2': int(((gsize[groups] >= 2) & (y == 1)).sum())}
        if gsize.max() > .02 * len(y):
            raise RuntimeError(f'Largest near-dup cluster too big ({gsize.max()}); raise --thr.')
        sw_all = dedup_weights(y, groups) if a.dedup_weight else None
        split = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
        folds = np.full(len(y), -1)
        oof = {k: np.zeros((len(y), 4)) for k in ['svm', 'cat', 'lat']}
        ext = {k: [np.zeros((len(d), 4)) for d in frames_ext] for k in ['svm', 'cat', 'lat']}
        for f, (tr, va) in enumerate(split.split(train, y, groups=groups)):
            folds[va] = f; ck = out / f'fold{f}.npz'
            if ck.exists():
                z = np.load(ck)
                assert np.array_equal(z['va'], va), 'checkpoint folds differ'
                for k in oof:
                    oof[k][va] = z[f'{k}_va']; ext[k][0] += z[f'{k}_dev'] / a.folds; ext[k][1] += z[f'{k}_test'] / a.folds
                continue
            assert set(y[tr]) == {0, 1, 2, 3}
            fit = train.iloc[tr]; frames = [train.iloc[va].drop(columns='spam_label')] + frames_ext
            sw = None if sw_all is None else sw_all[tr]
            saved = {'va': va}
            for k, fn in [('svm', svm_component), ('cat', cat_component), ('lat', latent_component)]:
                status(f'fold{f}_{k}', group_info=group_info); hosted_guard()
                pv, pd_, pt = fn(fit, frames, sw); gc.collect()
                oof[k][va] = pv; ext[k][0] += pd_ / a.folds; ext[k][1] += pt / a.folds
                saved.update({f'{k}_va': pv, f'{k}_dev': pd_, f'{k}_test': pt})
            np.savez_compressed(ck, **saved)
        # template flag for strata: class-1 look-alike in the fitting part of each fold / in full train for dev
        text = train.comment.tolist(); tmpl = np.zeros(len(y), bool)
        for f in range(a.folds):
            va = np.where(folds == f)[0]; r1 = np.where((folds != f) & (y == 1))[0]
            tmpl[va] = max_jaccard([text[i] for i in va], [text[i] for i in r1]) >= .6
        tmpl_dev = max_jaccard(dev.comment.tolist(), [text[i] for i in np.where(y == 1)[0]]) >= .6
        po = recipes(oof['svm'], oof['cat'], oof['lat'])
        pdv = recipes(ext['svm'][0], ext['cat'][0], ext['lat'][0])
        ptest = recipes(ext['svm'][1], ext['cat'][1], ext['lat'][1])
        rep = {'tag': tag, 'seed': a.seed, 'folds': a.folds, 'thr': a.thr, 'dedup_weight': a.dedup_weight,
               'group_info': group_info, 'class1_per_fold': [int(((folds == f) & (y == 1)).sum()) for f in range(a.folds)],
               'class1_template_share_oof': float(tmpl[y == 1].mean()),
               'class1_template_share_dev': float(tmpl_dev[ydev == 1].mean()),
               'svm_margin_std': {'oof': float(oof['svm'].std()), 'dev_foldavg': float(ext['svm'][0].std())},
               'pretrained': False, 'fit_data': 'train folds only; dev/test only predicted',
               'dev_caveat': 'dev reused many times in this project; bootstrap SD ~0.032; secondary signal only'}
        for name in po:
            rep[name] = {'oof_macro': macro(y, po[name]),
                         'oof_fold_macro': [macro(y[folds == f], po[name][folds == f]) for f in range(a.folds)],
                         'oof_class_f1': class_f1(y, po[name]).round(4).tolist(),
                         'oof_class1_strata': class1_strata(y, po[name], tmpl),
                         'dev_foldavg_macro': macro(ydev, pdv[name]),
                         'dev_foldavg_class_f1': class_f1(ydev, pdv[name]).round(4).tolist(),
                         'dev_class1_strata': class1_strata(ydev, pdv[name], tmpl_dev),
                         'test_pred_counts': np.bincount(ptest[name], minlength=4).tolist()}
        rep['v6_minus_v5'] = {
            'fold_deltas': [b - c for b, c in zip(rep['v6']['oof_fold_macro'], rep['v5']['oof_fold_macro'])],
            'cluster_bootstrap': group_bootstrap_delta(y, po['v5'], po['v6'], groups, seed=a.seed),
            'non_template_cluster_bootstrap': group_bootstrap_delta(y[~tmpl], po['v5'][~tmpl], po['v6'][~tmpl],
                                                                    groups[~tmpl], seed=a.seed),
            'dev_foldavg_delta': rep['v6']['dev_foldavg_macro'] - rep['v5']['dev_foldavg_macro']}
        np.savez_compressed(out / 'oof_and_foldavg.npz', folds=folds, groups=groups, tmpl=tmpl, tmpl_dev=tmpl_dev,
                            **{f'oof_{k}': v for k, v in oof.items()},
                            **{f'dev_{k}': v[0] for k, v in ext.items()}, **{f'test_{k}': v[1] for k, v in ext.items()})
        # E3 candidate CSVs (fold-average inference). Written only here; never auto-submitted.
        for name in ['v5', 'v6']:
            sub = data['samplesubmission'].copy()
            sub['spam_label'] = sub.ID.astype(str).map(pd.Series(ptest[name], index=test.id.astype(str)))
            assert sub.spam_label.notna().all() and sub.spam_label.isin([0, 1, 2, 3]).all()
            sub.spam_label = sub.spam_label.astype(int); sub.to_csv(out / f'submission_{name}_foldavg.csv', index=False)
        (out / 'report.json').write_text(json.dumps(rep, indent=2, ensure_ascii=False))
        status('complete'); print(json.dumps(rep, indent=2, ensure_ascii=False))
        return rep
    except BaseException as exc:
        status('error', error=str(exc), traceback=traceback.format_exc()); raise


if __name__ == '__main__':
    main()
