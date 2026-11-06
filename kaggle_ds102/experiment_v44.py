"""Fixed XGBoost classifier on train-only classwise lexical and SVD128 features.

Baseline caches are predictions from the previously trained V11 pipeline,
not pretrained external weights. New model/encoder fitting uses train only.
"""
import argparse,gc,hashlib,importlib.metadata,json,os,platform,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from cloud_checks import guard,checked
from train_submit import load_data,metrics
from improved_features import prepare
from latent_v6 import LatentFeatures
from review_common import near_dup_groups,max_jaccard,class1_strata,macro
from codex_checked_common import paired_cluster_bootstrap

WEIGHTS=np.array([1.,1.2,1.2,.9])
BEST_SHA='05bffeff984bd4927c8dd3e9bd3613aef6a7abce303e6e1dcb7cfa4a12012059'
GATES=dict(min_oof_delta=.002,min_positive_folds=2,max_fold_drop=.003,
    min_novel_fake_delta=0.,min_dev_delta=.001,bootstrap_q05_positive=True)

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prediction(p):return (p*WEIGHTS).argmax(1)
def combine(ref,old,new,alpha):return checked(.7*ref+(.3-alpha)*old+alpha*new,len(ref))

def submission(data,test,p):
    sub=data['samplesubmission'].copy();ids=sub.ID.copy()
    sub['spam_label']=sub.ID.astype(str).map(dict(zip(test.id.astype(str),p)))
    assert not sub.spam_label.isna().any()
    sub['spam_label']=sub.spam_label.astype(int)
    assert list(sub)==['ID','spam_label'] and len(sub)==3974
    assert sub.ID.equals(ids) and sub.ID.is_unique and sub.spam_label.isin(range(4)).all()
    return sub

def baseline_features(train,frames,folder,feature_id,status):
    """Cache only features fitted on this partition, never development labels."""
    path=folder/'base_features.npz'
    if path.exists():
        q=np.load(path,allow_pickle=False);assert q['feature_id'].item()==feature_id
        q.close();return path,feature_id
    status('fit_review_svd128')
    f=LatentFeatures();f.svd.set_params(n_components=128,random_state=102)
    all_x=[f.transform(train,fit=True)]+[f.transform(d) for d in frames]
    columns=[f'f_{i}' for i in range(575)];payload=dict(feature_id=feature_id,numeric_columns=np.array(columns))
    for prefix,x,df in zip(['train']+[f'frame{i}' for i in range(len(frames))],all_x,[train]+frames):
        assert len(x)==len(df) and all(c in x for c in columns)
        assert np.array_equal(x[['category','rating']].to_numpy(dtype=str),df[['category','rating']].astype(str).to_numpy())
        payload[prefix+'_numeric']=x[columns].to_numpy(dtype=np.float32)
        payload[prefix+'_categories']=x[['category','rating']].to_numpy(dtype=str)
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**payload);tmp.replace(path)
    del f,all_x,payload;gc.collect();return path,feature_id

def main(args):
    guard();start=time.time();inputs=Path(args.inputs);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    assert args.trial=='lexical'
    from xgb_lexical_v44 import fit_predict,CONFIG as backend_config
    alpha=.15
    expected=json.loads((inputs/'inputs_manifest.json').read_text())['files']
    for name,h in expected.items():assert digest(inputs/name)==h,'Input fingerprint mismatch: '+name
    versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','scikit-learn','catboost']}
    historical=json.loads((inputs/'baseline/latent_oof_manifest.json').read_text())
    assert versions==historical['versions'],'Use the package versions in notebook setup before running.'
    assert importlib.metadata.version('xgboost')=='3.4.1'
    versions['xgboost']=importlib.metadata.version('xgboost')
    sources={p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    manifest=dict(format=1,trial=args.trial,backend=backend_config,alpha=alpha,gates=GATES,
        data_and_baseline_sha256=expected,source_sha256=sources,versions=versions,seed=102)
    mid=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    mp=out/'run_manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest,'This output belongs to another run; choose a new output folder.'
    mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    report=dict(trial=args.trial,pretrained=False,external_data=False,execution='hosted Colab CPU',
        fit_data='train only; no dev/test feature, selector, model or calibration fitting',classifier='XGBoost 500 depth3 hist CPU, square-root class weights',
        baseline_user_reported_kaggle=.77436,previous_v27_user_reported_kaggle=.77229,
        kaggle_candidate_score=None,submission_created=False,manifest_id=mid,
        caveat='Repeated model selection on these OOF folds/dev. Bootstrap is conditional and does not remove selection bias or template overlap. No guaranteed leaderboard improvement.')
    def status(stage,**kwargs):
        report.update(stage=stage,elapsed_seconds=time.time()-start,**kwargs)
        (out/'metrics_v44.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(stage,json.dumps(kwargs,ensure_ascii=False),flush=True)
    try:
        (out/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),
            platform=platform.system(),hosted_colab=True,cpu=os.cpu_count(),versions=versions),indent=2))
        status('validate_inputs')
        data=load_data(inputs/'data');train,dev,test=[prepare(data[k]) for k in ['train','dev','test']]
        assert [len(x) for x in [train,dev,test]]==[14306,1590,3974]
        ref=np.load(inputs/'baseline/reference_v3.npz',allow_pickle=False)
        lat=np.load(inputs/'baseline/latent_oof.npz',allow_pickle=False)
        full=np.load(inputs/'baseline/latent_full.npz',allow_pickle=False)
        fm=json.loads((inputs/'baseline/latent_full_manifest.json').read_text())
        assert full['manifest_id'].item()==hashlib.sha256(json.dumps(fm,sort_keys=True).encode()).hexdigest()
        y=train.spam_label.to_numpy();folds=lat['folds'];assert np.array_equal(y,lat['y'])
        assert np.array_equal(y,ref['y']) and np.array_equal(folds,ref['folds'])
        base=checked(.7*ref['oof']+.3*lat['candidate'],len(y));old=prediction(base)
        bd=checked(.7*ref['dev']+.3*full['dev'],len(dev));bt=checked(.7*ref['test']+.3*full['test'],len(test))
        best=submission(data,test,prediction(bt));best_bytes=best.to_csv(index=False).encode()
        assert hashlib.sha256(best_bytes).hexdigest()==BEST_SHA
        assert digest(inputs/'baseline/submission_best_v11.csv')==BEST_SHA
        assert abs(macro(y,old)-.7749451430616494)<1e-12
        (out/'submission_baseline_v11.csv').write_bytes(best_bytes)
        assert np.array_equal(prediction(combine(ref['oof'],lat['candidate'],lat['candidate'],alpha)),old)
        status('baseline_verified',baseline_oof=metrics(y,old),baseline_dev=metrics(dev.spam_label,prediction(bd)),
            baseline_csv_sha256=BEST_SHA)
        splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=102)
        oof=np.zeros_like(base);fold_deltas=[]
        for fold,(tr,va) in enumerate(splitter.split(train,y,train.comment)):
            assert np.all(folds[va]==fold) and not set(train.comment.iloc[tr])&set(train.comment.iloc[va])
            folder=out/f'fold{fold}';folder.mkdir(exist_ok=True)
            ck=folder/'prediction.npz';status('fold_start',active_fold=fold+1,fold_deltas=fold_deltas)
            if ck.exists():
                q=np.load(ck,allow_pickle=False)
                assert q['manifest_id'].item()==mid and np.array_equal(q['tr'],tr) and np.array_equal(q['va'],va)
                p=checked(q['prediction'],len(va))
            else:
                fit=train.iloc[tr];frames=[train.iloc[va].drop(columns='spam_label')]
                # Reuse only the exact fitting-fold features produced by V43.
                prior_root=Path('/content/ds102_v43/runs/lexical_01')
                prior=prior_root/f'fold{fold}/base_features.npz'
                prior_manifest=json.loads((prior_root/'run_manifest.json').read_text())
                assert prior_manifest['data_and_baseline_sha256']==expected
                prior_mid=hashlib.sha256(json.dumps(prior_manifest,sort_keys=True).encode()).hexdigest()
                prior_id=hashlib.sha256(json.dumps(dict(manifest_id=prior_mid,tr=tr.tolist(),va=va.tolist()),sort_keys=True).encode()).hexdigest()
                if prior.exists():
                    q=np.load(prior,allow_pickle=False);assert q['feature_id'].item()==prior_id
                    path,feature_id=prior,prior_id;q.close();status('reuse_verified_v43_svd_features',active_fold=fold+1)
                else:
                    feature_id=hashlib.sha256(json.dumps(dict(manifest_id=mid,tr=tr.tolist(),va=va.tolist()),sort_keys=True).encode()).hexdigest()
                    path,feature_id=baseline_features(fit,frames,folder,feature_id,status)
                p=fit_predict(fit,frames,folder,feature_id,status,path,feature_id)[0]
                tmp=ck.with_suffix('.tmp.npz');np.savez_compressed(tmp,manifest_id=mid,tr=tr,va=va,prediction=p);tmp.replace(ck)
            oof[va]=p;new=prediction(combine(ref['oof'][va],lat['candidate'][va],p,alpha))
            fold_deltas.append(macro(y[va],new)-macro(y[va],old[va]))
            status('fold_complete',active_fold=fold+1,fold_deltas=fold_deltas)
        new=prediction(combine(ref['oof'],lat['candidate'],oof,alpha))
        status('cv_diagnostics',active_fold=None)
        template=np.zeros(len(y),bool)
        for fold in range(3):
            va=np.flatnonzero(folds==fold);tr=np.flatnonzero((folds!=fold)&(y==1))
            template[va]=max_jaccard(train.comment.iloc[va].tolist(),train.comment.iloc[tr].tolist())>=.6
        novel_delta=class1_strata(y,new,template)['novel']['f1']-class1_strata(y,old,template)['novel']['f1']
        delta=macro(y,new)-macro(y,old)
        boot=paired_cluster_bootstrap(y,old,new,near_dup_groups(train.comment.tolist(),thr=.7),n=2000)
        gates=dict(oof=bool(delta>=GATES['min_oof_delta']),
            folds=bool(sum(d>0 for d in fold_deltas)>=GATES['min_positive_folds']),
            worst_fold=bool(min(fold_deltas)>=-GATES['max_fold_drop']),
            novel_fake=bool(novel_delta>=GATES['min_novel_fake_delta']),
            bootstrap=bool(boot['q05']>0) if GATES['bootstrap_q05_positive'] else True)
        status('cv_complete',candidate_oof=metrics(y,new),delta=delta,fold_deltas=fold_deltas,
            novel_fake_f1_delta=novel_delta,bootstrap=boot,gates=gates)
        if not all(gates.values()):status('rejected_on_cv');return report
        folder=out/'full';folder.mkdir(exist_ok=True);ck=folder/'prediction.npz'
        status('fit_full_train')
        if ck.exists():
            q=np.load(ck,allow_pickle=False);assert q['manifest_id'].item()==mid
            ps=[checked(q['dev'],len(dev)),checked(q['test'],len(test))]
        else:
            frames=[dev.drop(columns='spam_label'),test]
            feature_id=hashlib.sha256((mid+':full').encode()).hexdigest()
            path,feature_id=baseline_features(train,frames,folder,feature_id,status)
            ps=fit_predict(train,frames,folder,feature_id,status,path,feature_id)
            tmp=ck.with_suffix('.tmp.npz');np.savez_compressed(tmp,manifest_id=mid,dev=ps[0],test=ps[1]);tmp.replace(ck)
        dp=prediction(combine(ref['dev'],full['dev'],ps[0],alpha))
        dev_delta=macro(dev.spam_label,dp)-macro(dev.spam_label,prediction(bd))
        unseen=~dev.comment.isin(set(train.comment)).to_numpy()
        status('dev_evaluated',candidate_dev=metrics(dev.spam_label,dp),dev_delta=dev_delta,
            candidate_dev_without_exact_overlap=metrics(dev.spam_label.to_numpy()[unseen],dp[unseen]),
            baseline_dev_without_exact_overlap=metrics(dev.spam_label.to_numpy()[unseen],prediction(bd)[unseen]))
        if dev_delta<GATES['min_dev_delta']:status('rejected_on_dev');return report
        tp=prediction(combine(ref['test'],full['test'],ps[1],alpha));sub=submission(data,test,tp)
        path=out/'submission_v44_candidate.csv';sub.to_csv(path,index=False)
        pd.testing.assert_frame_equal(pd.read_csv(path),sub,check_dtype=False)
        status('candidate_ready_not_submitted',submission_created=True,rows_verified=len(sub),
            candidate_sha256=digest(path),test_changes_vs_best=int((tp!=prediction(bt)).sum()))
        return report
    except BaseException as exc:
        (out/'error.txt').write_text(traceback.format_exc())
        status('error',error=str(exc));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--trial',choices=['lexical'],default='lexical')
    main(parser.parse_args())
