"""Fixed NMF topic augmentation, evaluated on near-duplicate grouped folds first.

All fitting requires hosted Colab. No pretrained assets or external data.
Earlier rows have all influenced development: new grouped folds are a stress
test, not an independent untouched test. No threshold relaxation or CSV export
on a failed gate. Each fitted component has a fingerprinted checkpoint.
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

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy import sparse
from scipy.special import softmax
from sklearn.decomposition import TruncatedSVD, MiniBatchNMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from catboost_v3 import cat_features, make_cat
from improved_features import FeaturesV2, dense_features, nb_ratios, prepare
from train_submit import load_data, make_model, metrics
from review_common import near_dup_groups, max_jaccard, class1_strata, macro
from codex_checked_common import paired_cluster_bootstrap, group_summary

CONFIG = dict(
    model_seed=102, cluster_split_seed=2053, exact_split_seed=102,
    folds=3, group_threshold=.7, template_threshold=.6,
    svd_dimensions=128, topic_dimensions=64, topic_max_features=60000,
    topic_ngrams=[1, 2], topic_min_df=3, topic_batch_size=512,
    topic_epochs=100, topic_transform_iterations=100,
    trees=800, depth=5, learning_rate=.06, l2_leaf_reg=8,
    candidate_weights=[.7, .15, .15], class_weights=[1., 1.2, 1.2, .9],
    gates=dict(cluster_min_delta=.003, exact_min_delta=.002,
               min_positive_folds=2, max_fold_drop=.005,
               min_novel_fake_f1_delta=0., bootstrap_q025_positive=True,
               min_dev_delta=.001, min_dev_without_exact_overlap_delta=.001),
    fit_data='training partitions only; dev/test never fit',
    pretrained=False, external_data=False,
)
BEST_SHA = '05bffeff984bd4927c8dd3e9bd3613aef6a7abce303e6e1dcb7cfa4a12012059'
W = np.array(CONFIG['class_weights'])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(p, n):
    p = np.asarray(p, dtype=np.float64)
    assert p.shape == (n, 4) and np.isfinite(p).all()
    assert (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-5)
    return p


def labels(p):
    return (p * W).argmax(1)


def tree(trees):
    guard()
    return CatBoostClassifier(
        iterations=trees, depth=5, learning_rate=.06, l2_leaf_reg=8,
        loss_function='MultiClass', auto_class_weights='SqrtBalanced',
        cat_features=['category', 'rating'], max_ctr_complexity=1,
        random_seed=102, thread_count=2, allow_writing_files=False, verbose=False)


class Representations:
    """The historical SVD128 view and a separate nonnegative word-topic view."""
    def __init__(self, smoke=False):
        self.base = FeaturesV2()
        self.svd = TruncatedSVD(n_components=16 if smoke else 128,
                                n_iter=5, random_state=102)
        self.word = TfidfVectorizer(
            ngram_range=(1, 2), min_df=3, max_features=60000,
            sublinear_tf=True, lowercase=False,
            token_pattern=r'(?u)\b\w+\b', dtype=np.float32)
        self.nmf = MiniBatchNMF(
            n_components=8 if smoke else 64, init='nndsvda', batch_size=512,
            beta_loss='frobenius', max_iter=5 if smoke else 100,
            transform_max_iter=100, random_state=102, tol=1e-4,
            max_no_improvement=10, alpha_W=0., alpha_H=0., l1_ratio=0.)

    def fit_base(self, train):
        guard()
        x, m = self.base.fit_transform(train)
        self.svd.fit(normalize(x))
        return x, m

    def latent(self, frame, x=None):
        if x is None:
            x, _ = self.base.transform(frame)
        review = normalize(x)
        title = normalize(sparse.hstack([
            self.base.word.transform(frame.product_name),
            self.base.char.transform(frame.product_name)], format='csr'))
        z = self.svd.transform(review)
        t = self.svd.transform(title)
        cosine = np.asarray(review.multiply(title).sum(1)).ravel()
        a = np.hstack([z, t, z*t, np.abs(z-t), cosine[:, None], dense_features(frame)])
        result = pd.DataFrame(a.astype(np.float32), columns=[f'f_{i}' for i in range(a.shape[1])])
        result['category'] = frame.category.to_numpy(dtype=str)
        result['rating'] = frame.rating.to_numpy(dtype=str)
        return result

    def fit_topics(self, train):
        guard()
        # Neither labels nor held-out catalogue text enter vocabulary or NMF fit.
        x = self.word.fit_transform(train.comment)
        self.nmf.fit(x)

    def augmented(self, frame, latent):
        r = self.word.transform(frame.comment)
        t = self.word.transform(frame.product_name)
        # Use transform for BOTH fitting and prediction rows: same score scale.
        z = normalize(self.nmf.transform(r))
        q = normalize(self.nmf.transform(t))
        lexical_cos = np.asarray(r.multiply(t).sum(1)).ravel()
        a = np.hstack([z, q, z*q, np.abs(z-q),
                       (z*q).sum(1)[:, None], lexical_cos[:, None],
                       np.asarray((r != 0).sum(1)), np.asarray((t != 0).sum(1))])
        assert np.isfinite(a).all() and (a >= 0).all()
        return pd.concat([latent.reset_index(drop=True), pd.DataFrame(
            a.astype(np.float32), columns=[f'topic_{i}' for i in range(a.shape[1])])], axis=1)


def fit_svm(x, m, targets, y):
    guard()
    out = [np.zeros((xx.shape[0], 4)) for xx, _ in targets]
    model = make_model(dict(weight='balanced', C=.5), y)
    model.fit(sparse.hstack([x, m], format='csr'), y)
    assert np.max(model.n_iter_) < model.max_iter
    for p, (xx, mm) in zip(out, targets):
        p += .75 * model.decision_function(sparse.hstack([xx, mm], format='csr'))
    for c, ratio in enumerate(nb_ratios(x, y)):
        model = LinearSVC(C=.5, class_weight='balanced', dual=True,
                          max_iter=30000, random_state=102)
        model.fit(sparse.hstack([x.multiply(ratio), m], format='csr'), (y == c).astype(int))
        assert np.max(model.n_iter_) < model.max_iter
        for p, (xx, mm) in zip(out, targets):
            p[:, c] += .25 * model.decision_function(
                sparse.hstack([xx.multiply(ratio), mm], format='csr'))
    return [softmax(3*p, axis=1) for p in out]


def partition(train, frames, folder, mid, status, baseline_needed, smoke=False):
    """Resume per component; fit strictly on the supplied training partition."""
    guard()
    folder.mkdir(parents=True, exist_ok=True)
    y = train.spam_label.to_numpy()
    bundles = folder/'representations.joblib'
    features = folder/'latent_features.joblib'
    if bundles.exists() and features.exists():
        b = joblib.load(bundles)
        assert b['manifest_id'] == mid
        enc = b['encoder']
        b = joblib.load(features)
        assert b['manifest_id'] == mid
        latent = b['frames']
    else:
        status('fit_partition_svd', partition=str(folder))
        enc = Representations(smoke=smoke)
        xt, _ = enc.fit_base(train)
        latent = [enc.latent(train, xt)] + [enc.latent(d) for d in frames]
        joblib.dump(dict(manifest_id=mid, encoder=enc), bundles)
        joblib.dump(dict(manifest_id=mid, frames=latent), features)
    result = {}
    for component in (['svm', 'cat', 'latent', 'topics'] if baseline_needed else ['topics']):
        ck = folder/f'{component}.npz'
        if ck.exists():
            q = np.load(ck, allow_pickle=False)
            assert q['manifest_id'].item() == mid
            values = [checked(q[f'p{i}'], len(d)) for i, d in enumerate(frames)]
        else:
            status('fit_partition_component', partition=str(folder), component=component)
            if component == 'svm':
                # Encoder was fit on the SAME fitting partition only.
                x, m = enc.base.transform(train)
                values = fit_svm(x, m, [enc.base.transform(d) for d in frames], y)
            elif component == 'cat':
                model = make_cat('SqrtBalanced', iterations=20 if smoke else 500)
                model.set_params(thread_count=2)
                model.fit(cat_features(train), y)
                values = [model.predict_proba(cat_features(d)) for d in frames]
            else:
                model = tree(20 if smoke else 800)
                if component == 'topics':
                    enc.fit_topics(train)
                    xx = [enc.augmented(d, a) for d, a in zip([train]+frames, latent)]
                else:
                    xx = latent
                model.fit(xx[0], y)
                assert np.array_equal(model.classes_, [0, 1, 2, 3])
                values = [model.predict_proba(a) for a in xx[1:]]
                if component == 'topics':
                    joblib.dump(dict(manifest_id=mid, encoder=enc), folder/'topic_encoder.joblib')
                    model.save_model(str(folder/'topic_model.cbm'))
                del xx
            values = [checked(p, len(d)) for p, d in zip(values, frames)]
            tmp = ck.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, manifest_id=mid, **{f'p{i}':p for i,p in enumerate(values)})
            tmp.replace(ck)
            gc.collect()
        result[component] = values
    return result


def main(args):
    guard()
    start = time.time()
    inputs = Path(args.inputs)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    expected = json.loads((inputs/'inputs_manifest.json').read_text())['files']
    for name, sha in expected.items():
        assert digest(inputs/name) == sha, name
    versions = {n: importlib.metadata.version(n) for n in ['numpy','scipy','pandas','scikit-learn','catboost']}
    historical = json.loads((inputs/'baseline/latent_oof_manifest.json').read_text())
    assert versions == historical['versions']
    sources = {p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    manifest = dict(config=CONFIG, inputs=expected, sources=sources, versions=versions, smoke=args.smoke)
    mid = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    mp = out/'run_manifest.json'
    if mp.exists():
        assert json.loads(mp.read_text()) == manifest, 'Use a new output directory for changed source/config.'
    mp.write_text(json.dumps(manifest, indent=2))
    report = dict(manifest_id=mid, config=CONFIG, stage='initializing', protocols={},
                  execution='hosted Colab CPU', submission_created=False,
                  best_user_reported_kaggle=.77436,
                  caveat='Grouped folds are a new stress test of adaptively chosen recipes, not untouched data. Bootstrap is conditional; Kaggle improvement is unknown.')
    def status(stage, **kw):
        report.update(stage=stage, elapsed_seconds=time.time()-start, **kw)
        (out/'metrics_v53.json').write_text(json.dumps(report, indent=2))
        print(stage, json.dumps(kw), flush=True)
    try:
        data = load_data(inputs/'data')
        train = prepare(data['train'])
        y = train.spam_label.to_numpy()
        if args.smoke:
            take = np.concatenate([np.flatnonzero(y == c)[:100] for c in range(4)])
            tr = np.concatenate([take[c*100:c*100+80] for c in range(4)])
            va = np.concatenate([take[c*100+80:c*100+100] for c in range(4)])
            result = partition(train.iloc[tr], [train.iloc[va].drop(columns='spam_label')],
                               out/'smoke', mid, status, True, smoke=True)
            for values in result.values():
                checked(values[0], len(va))
            status('smoke_passed', rows_fit=len(tr), rows_predict=len(va),
                   components=list(result), no_eligible_model_or_submission=True)
            return
        status('build_near_duplicate_groups')
        groups = near_dup_groups(train.comment.tolist(), thr=.7)
        gi = group_summary(groups, y)
        assert gi['largest_fraction'] <= .02, 'Clusters too large; stop rather than change grouping after observing scores.'
        np.savez_compressed(out/'groups.npz', groups=groups)
        ref = np.load(inputs/'baseline/reference_v3.npz', allow_pickle=False)
        old = np.load(inputs/'baseline/latent_oof.npz', allow_pickle=False)
        assert np.array_equal(ref['y'], y) and np.array_equal(old['y'], y)
        assert np.array_equal(ref['folds'], old['folds'])
        for protocol, group, seed in [('cluster', groups, 2053), ('exact', train.comment, 102)]:
            bp = np.zeros((len(y),4)); cp = bp.copy(); folds = np.full(len(y),-1)
            tmpl = np.zeros(len(y), bool); deltas = []
            splitter = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
            for f, (tr, va) in enumerate(splitter.split(train, y, groups=group)):
                assert not set(np.asarray(group)[tr]) & set(np.asarray(group)[va])
                assert set(y[tr]) == set(y[va]) == {0,1,2,3}
                folds[va] = f
                folder = out/protocol/f'fold{f}'
                folder.mkdir(parents=True, exist_ok=True)
                ids = folder/'indices.npz'
                if ids.exists():
                    q = np.load(ids); assert np.array_equal(q['tr'],tr) and np.array_equal(q['va'],va)
                else:
                    np.savez_compressed(ids,tr=tr,va=va)
                result = partition(train.iloc[tr], [train.iloc[va].drop(columns='spam_label')],
                                   folder, mid, status, protocol == 'cluster')
                if protocol == 'cluster':
                    base = .35*result['svm'][0]+.35*result['cat'][0]
                    latent = result['latent'][0]
                else:
                    assert np.all(ref['folds'][va] == f)
                    base = .7*ref['oof'][va]
                    latent = old['candidate'][va]
                bp[va] = checked(base+.3*latent, len(va))
                cp[va] = checked(base+.15*latent+.15*result['topics'][0], len(va))
                tmpl[va] = max_jaccard(train.comment.iloc[va].tolist(),
                    train.comment.iloc[tr[y[tr] == 1]].tolist()) >= .6
                deltas.append(macro(y[va],labels(cp[va]))-macro(y[va],labels(bp[va])))
                status('fold_complete', protocol=protocol, fold=f+1, fold_deltas=deltas, group_info=gi)
                if min(deltas) < -.005 or sum(d>0 for d in deltas)+(2-f) < 2:
                    status('pruned_on_predeclared_fold_gates', rejected_protocol=protocol)
                    return
            a,b = labels(bp),labels(cp)
            delta = macro(y,b)-macro(y,a)
            novel_delta = class1_strata(y,b,tmpl)['novel']['f1']-class1_strata(y,a,tmpl)['novel']['f1']
            boot = paired_cluster_bootstrap(y,a,b,groups,n=2000,seed=2053)
            gate = dict(macro=bool(delta >= CONFIG['gates'][protocol+'_min_delta']),
                        folds=bool(sum(d>0 for d in deltas)>=2),
                        worst_fold=bool(min(deltas)>=-.005),
                        novel_fake=bool(novel_delta>=0), bootstrap=bool(boot['q025']>0))
            report['protocols'][protocol] = dict(baseline=metrics(y,a),candidate=metrics(y,b),
                delta=delta,fold_deltas=deltas,novel_fake_f1_delta=novel_delta,bootstrap=boot,gates=gate)
            np.savez_compressed(out/f'{protocol}_oof.npz',y=y,folds=folds,baseline=bp,candidate=cp,template=tmpl)
            status('protocol_complete', protocol=protocol, delta=delta, gates=gate)
            if not all(gate.values()):
                status('rejected_on_'+protocol)
                return
        # External labels are first accessed for evaluation after BOTH CV gates.
        dev, test = [prepare(data[k]) for k in ['dev','test']]
        full = np.load(inputs/'baseline/latent_full.npz',allow_pickle=False)
        fm = json.loads((inputs/'baseline/latent_full_manifest.json').read_text())
        assert full['manifest_id'].item()==hashlib.sha256(json.dumps(fm,sort_keys=True).encode()).hexdigest()
        bd = checked(.7*ref['dev']+.3*full['dev'],len(dev))
        bt = checked(.7*ref['test']+.3*full['test'],len(test))
        sample = data['samplesubmission'].copy()
        sample['spam_label'] = sample.ID.astype(str).map(dict(zip(test.id.astype(str),labels(bt)))).astype(int)
        assert hashlib.sha256(sample.to_csv(index=False).encode()).hexdigest()==BEST_SHA
        result = partition(train,[dev.drop(columns='spam_label')],out/'full',mid,status,False)
        dp = labels(checked(.7*ref['dev']+.15*full['dev']+.15*result['topics'][0],len(dev)))
        yd = dev.spam_label.to_numpy(); unseen = ~dev.comment.isin(set(train.comment)).to_numpy()
        delta = macro(yd,dp)-macro(yd,labels(bd))
        ud = macro(yd[unseen],dp[unseen])-macro(yd[unseen],labels(bd)[unseen])
        status('dev_evaluated',dev_delta=delta,dev_without_exact_overlap_delta=ud,
               baseline_dev=metrics(yd,labels(bd)),candidate_dev=metrics(yd,dp))
        if delta<.001 or ud<.001:
            status('rejected_on_dev')
            return
        enc = joblib.load(out/'full/topic_encoder.joblib')
        assert enc['manifest_id']==mid
        enc = enc['encoder']; model = tree(800)
        model.load_model(str(out/'full/topic_model.cbm'))
        topic_test = checked(model.predict_proba(enc.augmented(test,enc.latent(test))),len(test))
        tp = labels(checked(.7*ref['test']+.15*full['test']+.15*topic_test,len(test)))
        sample['spam_label'] = sample.ID.astype(str).map(dict(zip(test.id.astype(str),tp))).astype(int)
        assert len(sample)==3974 and sample.ID.equals(data['samplesubmission'].ID) and sample.ID.is_unique
        assert sample.spam_label.isin(range(4)).all()
        path = out/'submission_v53_candidate.csv'; sample.to_csv(path,index=False)
        pd.testing.assert_frame_equal(pd.read_csv(path),sample,check_dtype=False)
        status('candidate_ready_not_submitted',submission_created=True,rows_verified=len(sample),
               sha256=digest(path),test_changes_vs_best=int((tp!=labels(bt)).sum()))
    except BaseException as exc:
        (out/'error.txt').write_text(traceback.format_exc())
        status('error',error=str(exc))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--smoke',action='store_true')
    main(parser.parse_args())
