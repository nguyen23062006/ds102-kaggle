"""Hosted-Colab-only seed ensemble of v6; no pretrained models or dev fitting.

One fixed candidate: average latent models with seeds 102,17,2026.
Keep v6's .7/.3 mixture, weights, features and all other hyperparameters.
"""
import gc, hashlib, importlib.metadata, json, os, platform, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent/'claude_review'))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from train_submit import load_data, metrics
from improved_features import prepare
from latent_v6 import LatentFeatures, make_model
from review_common import near_dup_groups, max_jaccard, class1_strata, macro
from codex_checked_common import paired_cluster_bootstrap

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output_v10_experiment'
CONFIG=dict(seeds=[102,17,2026],svd_seed='same as model seed',split_seed=102,
    folds=3,alpha=.3,weights=[1.,1.2,1.2,.9],trees=800,depth=5,learning_rate=.06,
    min_oof_delta=.001,min_positive_folds=2,min_bootstrap_q05=0.,
    max_novel_fake_f1_drop=.01,max_dev_drop=.005)


def guard():
    if platform.system()!='Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('All fitting must run on hosted Colab.')


def checked(p,n):
    p=np.asarray(p)
    assert p.shape==(n,4) and np.isfinite(p).all()
    assert (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-5)
    return p


def predict(base,latent):
    return ((.7*base+.3*latent)*np.array(CONFIG['weights'])).argmax(1)


def fit_predict(train,frames,seed):
    guard()
    f=LatentFeatures();f.svd.set_params(random_state=seed)
    x=f.transform(train,fit=True)
    model=make_model();model.set_params(random_seed=seed,thread_count=2)
    model.fit(x,train.spam_label.to_numpy(),verbose=False)
    assert np.array_equal(model.classes_,[0,1,2,3])
    result=[checked(model.predict_proba(f.transform(df)),len(df)) for df in frames]
    del f,x,model;gc.collect()
    return result


def main():
    guard();OUT.mkdir(exist_ok=True)
    data=load_data(ROOT/'data')
    train,dev,test=[prepare(data[k]) for k in ['train','dev','test']]
    y=train.spam_label.to_numpy()
    ref=np.load(ROOT/'colab_reference.npz',allow_pickle=False)
    assert np.array_equal(ref['y'],y)
    for name,n in [('oof',len(y)),('dev',len(dev)),('test',len(test))]:checked(ref[name],n)
    paths=list(ROOT.glob('*.py'))+list((ROOT/'claude_review').glob('*.py'))+list((ROOT/'data').glob('*.csv'))+[ROOT/'colab_reference.npz']
    manifest=dict(config=CONFIG,versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','scikit-learn','catboost']},
        sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    mid=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    mp=OUT/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest,'Manifest changed; use a fresh output directory.'
    mp.write_text(json.dumps(manifest,indent=2))
    report=dict(config=CONFIG,baseline_user_reported_kaggle=.77269,
        execution='hosted Google Colab CPU',pretrained=False,fit_data='train only',
        caveat='Fixed candidate after many prior experiments; bootstrap is conditional and does not remove selection bias. Kaggle improvement unknown.')
    def status(stage,**kv):
        report.update(stage=stage,**kv)
        (OUT/'report.json').write_text(json.dumps(report,indent=2))
        print(stage,kv,flush=True)
    start=time.time()
    oof=np.zeros((3,len(y),4));folds=np.full(len(y),-1)
    split=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=102)
    for fold,(tr,va) in enumerate(split.split(train,y,train.comment)):
        assert np.all(ref['folds'][va]==fold),'Reference split mismatch'
        assert not set(train.comment.iloc[tr])&set(train.comment.iloc[va])
        folds[va]=fold
        for j,seed in enumerate(CONFIG['seeds']):
            status('cv_fit',fold=fold+1,seed=seed)
            ck=OUT/f'fold{fold}_seed{seed}.npz'
            if ck.exists():
                z=np.load(ck,allow_pickle=False)
                assert z['manifest_id'].item()==mid and np.array_equal(z['tr'],tr) and np.array_equal(z['va'],va)
                p=checked(z['prediction'],len(va))
            else:
                p=fit_predict(train.iloc[tr],[train.iloc[va].drop(columns='spam_label')],seed)[0]
                tmp=ck.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,manifest_id=mid,tr=tr,va=va,prediction=p)
                tmp.replace(ck)
            oof[j,va]=p
        status('fold_complete',fold=fold+1)
    old=predict(ref['oof'],oof[0]);new=predict(ref['oof'],oof.mean(0))
    groups=near_dup_groups(train.comment.tolist(),thr=.7)
    template=np.zeros(len(y),bool)
    for fold in range(3):
        va=np.flatnonzero(folds==fold);tr=np.flatnonzero((folds!=fold)&(y==1))
        template[va]=max_jaccard(train.comment.iloc[va].tolist(),train.comment.iloc[tr].tolist())>=.6
    delta=macro(y,new)-macro(y,old)
    fd=[macro(y[folds==f],new[folds==f])-macro(y[folds==f],old[folds==f]) for f in range(3)]
    novel_delta=class1_strata(y,new,template)['novel']['f1']-class1_strata(y,old,template)['novel']['f1']
    boot=paired_cluster_bootstrap(y,old,new,groups,n=2000)
    gates=dict(macro=delta>=.001,folds=sum(v>0 for v in fd)>=2,
               novel=novel_delta>=-.01,bootstrap=boot['q05']>0)
    status('cv_complete',baseline_oof=metrics(y,old),candidate_oof=metrics(y,new),
        delta=delta,fold_deltas=fd,novel_fake_f1_delta=novel_delta,bootstrap=boot,gates=gates)
    np.savez_compressed(OUT/'oof.npz',latent=oof,folds=folds,y=y)
    if not all(gates.values()):
        status('rejected_on_cv',elapsed_seconds=time.time()-start);return
    # Keep dev completely outside fitting; labels used only after full fits.
    ext=[]
    frames=[dev.drop(columns='spam_label'),test]
    for seed in CONFIG['seeds']:
        status('full_train_fit',seed=seed)
        ck=OUT/f'full_seed{seed}.npz'
        if ck.exists():
            z=np.load(ck,allow_pickle=False);assert z['manifest_id'].item()==mid
            result=[checked(z['dev'],len(dev)),checked(z['test'],len(test))]
        else:
            result=fit_predict(train,frames,seed)
            tmp=ck.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,manifest_id=mid,dev=result[0],test=result[1]);tmp.replace(ck)
        ext.append(result)
    dev_latent=np.stack([p[0] for p in ext])
    test_latent=np.stack([p[1] for p in ext])
    dp_old=predict(ref['dev'],dev_latent[0]);dp=predict(ref['dev'],dev_latent.mean(0))
    report['dev']=dict(baseline=metrics(dev.spam_label,dp_old),candidate=metrics(dev.spam_label,dp))
    if macro(dev.spam_label,dp)<macro(dev.spam_label,dp_old)-.005:
        status('rejected_on_dev',elapsed_seconds=time.time()-start);return
    labels=predict(ref['test'],test_latent.mean(0))
    sub=data['samplesubmission'].copy()
    sub['spam_label']=sub.ID.astype(str).map(dict(zip(test.id.astype(str),labels))).astype(int)
    assert len(sub)==len(test) and sub.ID.is_unique and sub.spam_label.isin(range(4)).all()
    path=OUT/'submission_v10_candidate.csv';sub.to_csv(path,index=False)
    pd.testing.assert_frame_equal(pd.read_csv(path),sub,check_dtype=False)
    status('candidate_ready_not_submitted',rows=len(sub),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),elapsed_seconds=time.time()-start)


if __name__=='__main__':main()
