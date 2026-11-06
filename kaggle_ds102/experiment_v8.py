"""Fixed pairwise CatBoost specialists; all fitting requires hosted Colab.

No pretrained weights, external data, dev fitting, or parameter search.
Run from the extracted ds102_v7_bundle directory on Colab.
"""
import gc, hashlib, json, os, platform, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'claude_review'))
import numpy as np
from scipy import sparse
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from train_submit import load_data, metrics
from train_submit_v2 import fit_predict, RECIPE
from improved_features import prepare, FeaturesV2
from catboost_v3 import make_cat, cat_features
from codex_checked_common import paired_cluster_bootstrap
from review_common import near_dup_groups, max_jaccard, macro, class1_strata, W_V5

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output_v8_experiment'
CONFIG = dict(specialists=[1,2], trees=500, depth=5, learning_rate=.1,
              specialist_weight=.2, seed=102, folds=3, cluster_threshold=.7,
              class_weight='SqrtBalanced', min_delta=.003,
              min_positive_folds=2, max_novel_f1_drop=.01, max_dev_drop=.005)


def guard():
    if platform.system() != 'Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Training is allowed on hosted Colab only.')


def branch(fit, frames):
    guard()
    y = fit.spam_label.to_numpy()
    # Pure per-row feature extraction; supervised text processing happens inside fit.
    features = cat_features(fit)
    targets = [cat_features(d) for d in frames]
    result = [np.zeros((len(d), 4)) for d in frames]
    for c in CONFIG['specialists']:
        mask = (y == 0) | (y == c)
        model = make_cat('SqrtBalanced', iterations=CONFIG['trees'])
        model.set_params(loss_function='Logloss', thread_count=2)
        print('specialist', c, 'fit_rows', int(mask.sum()), flush=True)
        model.fit(features.loc[mask], (y[mask] == c).astype(int), verbose=False)
        assert np.array_equal(model.classes_, [0, 1])
        for p, target in zip(result, targets):
            p[:, c] = model.predict_proba(target)[:, 1]
        print('specialist', c, 'complete', flush=True)
    return result


def base(fit, frames):
    guard()
    s = fit_predict(fit, frames, RECIPE)
    model = make_cat('SqrtBalanced', iterations=500)
    model.set_params(thread_count=2)
    model.fit(cat_features(fit), fit.spam_label.to_numpy(), verbose=False)
    return [.5 * softmax(3 * a, axis=1) + .5 * model.predict_proba(cat_features(d))
            for a, d in zip(s, frames)]


def pred(p):
    assert p.ndim == 2 and p.shape[1] == 4 and np.isfinite(p).all()
    assert np.allclose(p.sum(1), 1, atol=1e-5) and (p >= 0).all()
    return (p * W_V5).argmax(1)


def blend(a, b):
    assert a.shape == b.shape and np.isfinite(b).all()
    assert ((b >= 0) & (b <= 1)).all()
    # Each specialist redistributes only its {0,c} probability mass.
    result = a.copy()
    for c in CONFIG['specialists']:
        mass = a[:, 0] + a[:, c]
        delta = mass * b[:, c] - a[:, c]
        result[:, c] += CONFIG['specialist_weight'] * delta
        result[:, 0] -= CONFIG['specialist_weight'] * delta
    # Equivalent to .6*base + .2*specialist01 + .2*specialist02.
    assert (result >= -1e-12).all() and np.allclose(result.sum(1), 1)
    return result


def main():
    guard()
    OUT.mkdir(exist_ok=True)
    data = load_data(ROOT / 'data')
    train, dev, test = [prepare(data[k]) for k in ['train', 'dev', 'test']]
    y = train.spam_label.to_numpy()
    ref = np.load(ROOT / 'colab_reference.npz', allow_pickle=False)
    assert np.array_equal(ref['y'], y)
    source_paths = list(ROOT.glob('*.py')) + list((ROOT/'claude_review').glob('*.py'))
    import importlib.metadata
    manifest = dict(config=CONFIG, versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','scikit-learn','catboost']}, sha256={str(p.relative_to(ROOT)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in
        source_paths + list((ROOT/'data').glob('*.csv')) + [ROOT/'colab_reference.npz']})
    mp = OUT/'manifest.json'
    if mp.exists():
        assert json.loads(mp.read_text()) == manifest, 'Changed input/source; use a fresh output directory.'
    mp.write_text(json.dumps(manifest, indent=2))
    report = dict(config=CONFIG, protocols={}, caveat='Exploratory after prior iterations; Kaggle improvement unknown.')
    def save(stage):
        report['stage'] = stage
        (OUT/'report.json').write_text(json.dumps(report, indent=2))
        print(stage, flush=True)
    save('build_groups')
    groups = near_dup_groups(train.comment.tolist(), thr=.7)
    assert np.bincount(groups).max()/len(y) <= .02
    for protocol, split_groups in [('exact', train.comment.to_numpy()), ('cluster', groups)]:
        bp = np.zeros((len(y), 4)); lp = bp.copy()
        folds = np.full(len(y), -1); template = np.zeros(len(y), bool)
        splits = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=102)
        for fold, (tr, va) in enumerate(splits.split(train, y, split_groups)):
            folds[va] = fold
            if protocol == 'exact':
                assert np.all(ref['folds'][va] == fold), 'Reference fold mismatch'
            assert not set(split_groups[tr]) & set(split_groups[va])
            ck = OUT/f'{protocol}_fold{fold}.npz'
            save(f'{protocol}_fold{fold+1}')
            if ck.exists():
                z = np.load(ck, allow_pickle=False)
                assert np.array_equal(z['tr'], tr) and np.array_equal(z['va'], va)
                bp[va], lp[va] = z['base'], z['branch']
            else:
                frames = [train.iloc[va].drop(columns='spam_label')]
                lp[va] = branch(train.iloc[tr], frames)[0]
                bp[va] = ref['oof'][va] if protocol == 'exact' else base(train.iloc[tr], frames)[0]
                tmp = ck.with_suffix('.tmp.npz')
                np.savez_compressed(tmp, tr=tr, va=va, base=bp[va], branch=lp[va])
                tmp.replace(ck)
            template[va] = max_jaccard(train.comment.iloc[va].tolist(),
                train.comment.iloc[tr[y[tr] == 1]].tolist()) >= .6
            gc.collect()
        old, new = pred(bp), pred(blend(bp, lp))
        deltas = [macro(y[folds==f], new[folds==f])-macro(y[folds==f], old[folds==f]) for f in range(3)]
        novel_delta = class1_strata(y, new, template)['novel']['f1'] - class1_strata(y, old, template)['novel']['f1']
        boot = paired_cluster_bootstrap(y, old, new, groups, n=2000)
        gates = dict(macro=macro(y,new)-macro(y,old)>=.003,
                     folds=sum(d>0 for d in deltas)>=2, novel=novel_delta>=-.01,
                     paired_interval=boot['q025']>0)
        report['protocols'][protocol] = dict(baseline=metrics(y,old), candidate=metrics(y,new),
            fold_deltas=deltas, novel_fake_f1_delta=novel_delta, bootstrap=boot, gates=gates)
        save(f'{protocol}_complete')
        if not all(gates.values()):
            save('rejected_on_'+protocol)
            return
    save('fit_full_train_branch')
    dp, tp = branch(train, [dev.drop(columns='spam_label'), test])
    old, new = pred(ref['dev']), pred(blend(ref['dev'], dp))
    report['dev'] = dict(baseline=metrics(dev.spam_label,old), candidate=metrics(dev.spam_label,new))
    if macro(dev.spam_label,new) < macro(dev.spam_label,old)-.005:
        save('rejected_on_dev'); return
    predicted = pred(blend(ref['test'],tp))
    sub = data['samplesubmission'].copy()
    mapping = dict(zip(test.id.astype(str), predicted))
    sub['spam_label'] = sub.ID.astype(str).map(mapping).astype(int)
    assert len(sub)==len(test) and sub.ID.is_unique and sub.spam_label.isin(range(4)).all()
    sub.to_csv(OUT/'submission_v8_candidate.csv', index=False)
    report['rows'] = len(sub)
    save('candidate_ready_not_submitted')


if __name__ == '__main__':
    main()
