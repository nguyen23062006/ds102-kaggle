"""Fixed agreement committee of catalogue and supervised-column latent models.

Rule declared after V47 and first V48 fold observations: adaptive selection bias
remains. No meta-model is fit, no leaderboard/test labels are used. Original
V47/V48 rejections stay unchanged. Only0/2 may change, and only if both agree.
"""
import hashlib,importlib.metadata,json,sys,time,traceback
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from cloud_checks import guard,checked
from train_submit import load_data,metrics
from improved_features import prepare
from review_common import near_dup_groups,macro
from codex_checked_common import paired_cluster_bootstrap
from handoff_v43 import baseline_features,submission
from catalogue_svd_v40 import fit_predict as fit_catalogue
from supervised_svd_v48 import fit_predict as fit_supervised

ROOT=Path('/content/ds102_v49');OUT=ROOT/'run';INPUTS=Path('/content/ds102_v43/inputs')
W=np.array([1.,1.2,1.2,.9])
CONFIG=dict(rule='Change0/2 only when catalogue and supervised-latent predictions agree on0/2',
    catalogue_alpha=.3,supervised_alpha=.15,frozen_classes=[1,3],seed=102,
    min_oof_gain=.002,min_positive_folds=2,max_fold_drop=.003,min_dev_gain=.001,
    require_bootstrap_q05_positive=True,baseline_user_reported_kaggle=.77436)

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def label(p):return (p*W).argmax(1)
def decide(ref,old,cat,sup):
    base=label(checked(.7*ref+.3*old,len(ref)))
    a=label(checked(.7*ref+.3*cat,len(ref)))
    b=label(checked(.7*ref+.15*old+.15*sup,len(ref)))
    mask=np.isin(base,[0,2])&(a==b)&np.isin(a,[0,2])
    pred=base.copy();pred[mask]=a[mask]
    assert np.array_equal(pred==1,base==1) and np.array_equal(pred==3,base==3)
    return pred

def main():
    guard();OUT.mkdir(parents=True,exist_ok=True);start=time.time()
    expected=json.loads((INPUTS/'inputs_manifest.json').read_text())['files']
    for n,h in expected.items():assert digest(INPUTS/n)==h
    historical=json.loads((INPUTS/'baseline/latent_oof_manifest.json').read_text())
    versions={k:importlib.metadata.version(k) for k in historical['versions']}
    assert versions==historical['versions']
    parents={47:Path('/content/ds102_v47/runs/catalogue_01'),48:Path('/content/ds102_v48/runs/supervised_svd_01')}
    parent_ids={}
    for v,folder in parents.items():
        manifest=json.loads((folder/'run_manifest.json').read_text())
        assert manifest['data_and_baseline_sha256']==expected
        parent_ids[v]=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    manifest=dict(config=CONFIG,versions=versions,inputs=expected,
        parent_manifest_sha256={str(v):digest(p/'run_manifest.json') for v,p in parents.items()},
        sources={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
    mid=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    mp=OUT/'run_manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    report=dict(config=CONFIG,pretrained=False,external_data=False,execution='hosted Colab CPU',
        fit_data='train only; fixed vote rule, no trained meta-parameters',submission_created=False,
        selection_caveat='Rule proposed after prior model/partial-fold outcomes; repeated OOF/dev selection remains. Paired bootstrap is conditional, not post-selection proof or guaranteed Kaggle gain.')
    def status(stage,**kwargs):
        report.update(stage=stage,elapsed_seconds=time.time()-start,**kwargs)
        (OUT/'metrics_v49.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(stage,json.dumps(kwargs,ensure_ascii=False),flush=True)
    try:
        data=load_data(INPUTS/'data');train,dev,test=[prepare(data[k]) for k in ['train','dev','test']]
        ref=np.load(INPUTS/'baseline/reference_v3.npz');lat=np.load(INPUTS/'baseline/latent_oof.npz');full=np.load(INPUTS/'baseline/latent_full.npz')
        y=train.spam_label.to_numpy();folds=lat['folds']
        assert np.array_equal(y,lat['y']) and np.array_equal(y,ref['y']) and np.array_equal(folds,ref['folds'])
        full_manifest=json.loads((INPUTS/'baseline/latent_full_manifest.json').read_text())
        assert full['manifest_id'].item()==hashlib.sha256(json.dumps(full_manifest,sort_keys=True).encode()).hexdigest()
        baseline=label(.7*ref['oof']+.3*lat['candidate']);pred=baseline.copy();fd=[]
        splitter=StratifiedGroupKFold(3,shuffle=True,random_state=102)
        for f,(tr,va) in enumerate(splitter.split(train,y,train.comment)):
            assert np.all(folds[va]==f) and not set(train.comment.iloc[tr])&set(train.comment.iloc[va])
            ps={};status('fold_start',active_fold=f+1,fold_deltas=fd)
            for v in [47,48]:
                cache=parents[v]/f'fold{f}/prediction.npz'
                if cache.exists():
                    q=np.load(cache,allow_pickle=False)
                    assert q['manifest_id'].item()==parent_ids[v] and np.array_equal(q['tr'],tr) and np.array_equal(q['va'],va)
                    ps[v]=checked(q['prediction'],len(va))
                else:
                    assert v==48,'Catalogue parent must have completed all folds'
                    folder=OUT/f'fold{f}_supervised_completion';folder.mkdir(exist_ok=True)
                    cache=folder/'prediction.npz'
                    if cache.exists():
                        q=np.load(cache);assert q['manifest_id'].item()==mid and np.array_equal(q['tr'],tr) and np.array_equal(q['va'],va)
                        ps[v]=checked(q['prediction'],len(va))
                    else:
                        feature_id=hashlib.sha256((mid+':fold'+str(f)).encode()).hexdigest()
                        status('complete_missing_supervised_component',active_fold=f+1)
                        ps[v]=fit_supervised(train.iloc[tr],[train.iloc[va].drop(columns='spam_label')],folder,feature_id,status)[0]
                        np.savez_compressed(cache,manifest_id=mid,tr=tr,va=va,prediction=ps[v])
            pred[va]=decide(ref['oof'][va],lat['candidate'][va],ps[47],ps[48])
            fd.append(macro(y[va],pred[va])-macro(y[va],baseline[va]))
            status('fold_complete',active_fold=f+1,fold_deltas=fd)
            if f<2 and (sum(x>0 for x in fd)+(2-f)<2 or min(fd)<-.003):
                status('pruned_on_predeclared_fold_gates',full_oof_available=False);return
        assert np.array_equal(pred==1,baseline==1) and np.array_equal(pred==3,baseline==3)
        delta=macro(y,pred)-macro(y,baseline)
        status('cv_diagnostics',active_fold=None,delta=delta,fold_deltas=fd)
        boot=paired_cluster_bootstrap(y,baseline,pred,near_dup_groups(train.comment.tolist(),thr=.7),n=2000)
        gates=dict(oof=bool(delta>=.002),folds=bool(sum(x>0 for x in fd)>=2),worst_fold=bool(min(fd)>=-.003),
            novel_fake=True,bootstrap=bool(boot['q05']>0))
        status('cv_complete',baseline_oof=metrics(y,baseline),candidate_oof=metrics(y,pred),
            delta=delta,fold_deltas=fd,novel_fake_f1_delta=0.,class1_and3_pred_mask_exactly_preserved=True,
            bootstrap=boot,gates=gates)
        np.savez_compressed(OUT/'oof.npz',y=y,folds=folds,prediction=pred)
        if not all(gates.values()):status('rejected_on_cv');return
        frames=[dev.drop(columns='spam_label'),test];external={}
        for v,fn in [(47,fit_catalogue),(48,fit_supervised)]:
            folder=OUT/f'full_component{v}';folder.mkdir(exist_ok=True);cache=folder/'predictions.npz'
            if cache.exists():
                q=np.load(cache);assert q['manifest_id'].item()==mid
                external[v]=[checked(q['dev'],len(dev)),checked(q['test'],len(test))]
            else:
                fid=hashlib.sha256((mid+':full'+str(v)).encode()).hexdigest()
                status('fit_full_train',component=v)
                if v==47:
                    path,fid=baseline_features(train,frames,folder,fid,status)
                    external[v]=fn(train,frames,folder,fid,status,path,fid)
                else:external[v]=fn(train,frames,folder,fid,status)
                np.savez_compressed(cache,manifest_id=mid,dev=external[v][0],test=external[v][1])
        old_dev=label(.7*ref['dev']+.3*full['dev'])
        dp=decide(ref['dev'],full['dev'],external[47][0],external[48][0])
        dev_delta=macro(dev.spam_label,dp)-macro(dev.spam_label,old_dev)
        status('dev_evaluated',baseline_dev=metrics(dev.spam_label,old_dev),candidate_dev=metrics(dev.spam_label,dp),dev_delta=dev_delta)
        if dev_delta<.001:status('rejected_on_dev');return
        old_test=label(.7*ref['test']+.3*full['test'])
        best=submission(data,test,old_test)
        assert hashlib.sha256(best.to_csv(index=False).encode()).hexdigest()=='05bffeff984bd4927c8dd3e9bd3613aef6a7abce303e6e1dcb7cfa4a12012059'
        tp=decide(ref['test'],full['test'],external[47][1],external[48][1]);sub=submission(data,test,tp)
        path=OUT/'submission_v49_candidate.csv';sub.to_csv(path,index=False)
        status('candidate_ready_not_submitted',submission_created=True,rows_verified=len(sub),sha256=digest(path),test_changes_vs_best=int((tp!=old_test).sum()))
    except BaseException as exc:
        (OUT/'error.txt').write_text(traceback.format_exc());status('error',error=str(exc));raise

if __name__=='__main__':main()
