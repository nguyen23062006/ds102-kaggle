"""One fixed trial: replace v6's 64D SVD by 128D. Hosted Colab only."""
import gc, hashlib, importlib.metadata, json, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'claude_review'))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from experiment_v10 import guard, checked, predict
from latent_v6 import LatentFeatures, make_model
from train_submit import load_data, metrics
from improved_features import prepare
from review_common import near_dup_groups, max_jaccard, class1_strata, macro
from codex_checked_common import paired_cluster_bootstrap

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output_v11_experiment'
CONFIG=dict(dimensions=128,seed=102,folds=3,trees=800,depth=5,lr=.06,
    alpha=.3,weights=[1.,1.2,1.2,.9],min_oof_delta=.002,
    min_positive_folds=2,max_novel_fake_f1_drop=.01,min_bootstrap_q05=0.,max_dev_drop=.005)


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fit_latent(train,frames,dimensions,status):
    guard();status('fit_features',dimensions=dimensions)
    f=LatentFeatures();f.svd.set_params(n_components=dimensions,random_state=102)
    xt=f.transform(train,fit=True)
    status('fit_model',dimensions=dimensions)
    model=make_model();model.set_params(thread_count=2,random_seed=102)
    model.fit(xt,train.spam_label.to_numpy(),verbose=200)
    assert np.array_equal(model.classes_,[0,1,2,3])
    result=[checked(model.predict_proba(f.transform(d)),len(d)) for d in frames]
    del f,xt,model;gc.collect();return result


def main():
    guard();OUT.mkdir(exist_ok=True);start=time.time()
    previous=ROOT/'output_v10_experiment'
    old_manifest=json.loads((previous/'manifest.json').read_text())
    versions={k:importlib.metadata.version(k) for k in old_manifest['versions']}
    assert versions==old_manifest['versions'],'Library versions differ from cached baseline'
    for name,sha in old_manifest['sha256'].items():assert digest(ROOT/name)==sha,name
    old_mid=hashlib.sha256(json.dumps(old_manifest,sort_keys=True).encode()).hexdigest()
    manifest=dict(config=CONFIG,versions=versions,baseline_manifest=old_manifest,
        baseline_checkpoint_sha256={f'fold{f}_seed102.npz':digest(previous/f'fold{f}_seed102.npz') for f in range(3)},
        source_sha256=digest(__file__))
    mid=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    mp=OUT/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest,'Manifest changed'
    mp.write_text(json.dumps(manifest,indent=2))
    report=dict(config=CONFIG,baseline_user_reported_kaggle=.77269,
        execution='hosted Colab CPU',pretrained=False,fit_data='train only',
        caveat='Compared with current 64D refit from V10, not bit-identical historical v6. Repeated OOF/development use creates selection bias; bootstrap is conditional. Kaggle improvement unknown.')
    def status(stage,**kv):
        report.update(stage=stage,**kv);(OUT/'report.json').write_text(json.dumps(report,indent=2))
        print(stage,kv,flush=True)
    data=load_data(ROOT/'data');train,dev,test=[prepare(data[k]) for k in ['train','dev','test']]
    y=train.spam_label.to_numpy();ref=np.load(ROOT/'colab_reference.npz',allow_pickle=False)
    assert np.array_equal(y,ref['y'])
    base=np.zeros((len(y),4));new=np.zeros_like(base);folds=np.full(len(y),-1)
    split=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=102)
    for fold,(tr,va) in enumerate(split.split(train,y,train.comment)):
        status('fold_start',fold=fold+1)
        assert np.all(ref['folds'][va]==fold)
        assert not set(train.comment.iloc[tr])&set(train.comment.iloc[va])
        folds[va]=fold
        z=np.load(previous/f'fold{fold}_seed102.npz',allow_pickle=False)
        assert z['manifest_id'].item()==old_mid and np.array_equal(z['tr'],tr) and np.array_equal(z['va'],va)
        base[va]=checked(z['prediction'],len(va))
        ck=OUT/f'fold{fold}_128.npz'
        if ck.exists():
            z=np.load(ck,allow_pickle=False)
            assert z['manifest_id'].item()==mid and np.array_equal(z['tr'],tr) and np.array_equal(z['va'],va)
            p=checked(z['prediction'],len(va))
        else:
            p=fit_latent(train.iloc[tr],[train.iloc[va].drop(columns='spam_label')],128,status)[0]
            tmp=ck.with_suffix('.tmp.npz');np.savez_compressed(tmp,manifest_id=mid,tr=tr,va=va,prediction=p);tmp.replace(ck)
        new[va]=p
        a=predict(ref['oof'][va],base[va]);b=predict(ref['oof'][va],p)
        status('fold_complete',fold=fold+1,fold_delta=macro(y[va],b)-macro(y[va],a))
    a=predict(ref['oof'],base);b=predict(ref['oof'],new)
    fd=[macro(y[folds==f],b[folds==f])-macro(y[folds==f],a[folds==f]) for f in range(3)]
    status('cv_diagnostics')
    groups=near_dup_groups(train.comment.tolist(),thr=.7);template=np.zeros(len(y),bool)
    for fold in range(3):
        va=np.flatnonzero(folds==fold);tr=np.flatnonzero((folds!=fold)&(y==1))
        template[va]=max_jaccard(train.comment.iloc[va].tolist(),train.comment.iloc[tr].tolist())>=.6
    delta=macro(y,b)-macro(y,a)
    nd=class1_strata(y,b,template)['novel']['f1']-class1_strata(y,a,template)['novel']['f1']
    boot=paired_cluster_bootstrap(y,a,b,groups,n=2000)
    gates=dict(macro=delta>=.002,folds=sum(d>0 for d in fd)>=2,novel=nd>=-.01,bootstrap=boot['q05']>0)
    status('cv_complete',baseline_oof=metrics(y,a),candidate_oof=metrics(y,b),delta=delta,
        fold_deltas=fd,novel_fake_f1_delta=nd,bootstrap=boot,gates=gates)
    np.savez_compressed(OUT/'oof.npz',baseline=base,candidate=new,y=y,folds=folds)
    if not all(gates.values()):status('rejected_on_cv',elapsed_seconds=time.time()-start);return
    frames=[dev.drop(columns='spam_label'),test];full=[]
    for dimensions in [64,128]:
        ck=OUT/f'full_{dimensions}.npz'
        if ck.exists():
            z=np.load(ck,allow_pickle=False);assert z['manifest_id'].item()==mid
            result=[checked(z['dev'],len(dev)),checked(z['test'],len(test))]
        else:
            result=fit_latent(train,frames,dimensions,status)
            tmp=ck.with_suffix('.tmp.npz');np.savez_compressed(tmp,manifest_id=mid,dev=result[0],test=result[1]);tmp.replace(ck)
        full.append(result)
    old=predict(ref['dev'],full[0][0]);dp=predict(ref['dev'],full[1][0])
    report['dev']=dict(baseline=metrics(dev.spam_label,old),candidate=metrics(dev.spam_label,dp))
    if macro(dev.spam_label,dp)<macro(dev.spam_label,old)-.005:
        status('rejected_on_dev',elapsed_seconds=time.time()-start);return
    tp=predict(ref['test'],full[1][1]);sub=data['samplesubmission'].copy()
    sub['spam_label']=sub.ID.astype(str).map(dict(zip(test.id.astype(str),tp))).astype(int)
    assert len(sub)==len(test) and sub.ID.is_unique and sub.spam_label.isin(range(4)).all()
    path=OUT/'submission_v11_candidate.csv';sub.to_csv(path,index=False)
    pd.testing.assert_frame_equal(pd.read_csv(path),sub,check_dtype=False)
    status('candidate_ready_not_submitted',rows=len(sub),sha256=digest(path),elapsed_seconds=time.time()-start)


if __name__=='__main__':main()
