"""Frozen evaluation protocol; the model fitter is supplied explicitly by experiment_v59.py."""
import argparse, gc, hashlib, importlib.metadata, json, os, platform, sys, time, traceback
from pathlib import Path

def guard():
    if platform.system()!='Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Hosted Google Colab only; local fitting disabled.')

if __name__=='__main__': guard()

sys.path.insert(0,'/content/ds102_v53/source')
import joblib
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import normalize
from catboost_v3 import make_cat,cat_features
from train_submit import load_data,metrics
from improved_features import prepare
from review_common import near_dup_groups,max_jaccard,class1_strata,macro
from codex_checked_common import paired_cluster_bootstrap
from experiment_v53 import Representations,fit_svm,tree,checked,labels,BEST_SHA

CONFIG={}  # Set explicitly by experiment_v59.py before main().

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()



def baseline_partition(train,frame,folder,mid,status):
    """Fallback only when V53 did not reach this fold. Never fit its NMF branch."""
    folder.mkdir(parents=True,exist_ok=True);cache=folder/'baseline.npz'
    if cache.exists():
        z=np.load(cache);assert z['manifest_id'].item()==mid
        return checked(z['prediction'],len(frame)),checked(z['latent_prediction'],len(frame))
    guard();status('fit_missing_baseline_features',partition=str(folder))
    enc=Representations();x,m=enc.fit_base(train);v,vm=enc.base.transform(frame)
    svm=fit_svm(x,m,[(v,vm)],train.spam_label.to_numpy())[0]
    status('fit_missing_baseline_cat',partition=str(folder))
    model=make_cat('SqrtBalanced',iterations=500);model.set_params(thread_count=2)
    model.fit(cat_features(train),train.spam_label.to_numpy())
    cat=model.predict_proba(cat_features(frame));del model;gc.collect()
    status('fit_missing_baseline_latent',partition=str(folder))
    model=tree(800);model.fit(enc.latent(train,x),train.spam_label.to_numpy())
    lat=model.predict_proba(enc.latent(frame,v))
    result=checked(.35*svm+.35*cat+.3*lat,len(frame))
    np.savez_compressed(cache,manifest_id=mid,prediction=result,latent_prediction=lat)
    return result,checked(lat,len(frame))

def main(a,model_fitter):
    guard();start=time.time();inputs=Path(a.inputs);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    parent=Path('/content/ds102_v53/run');src=Path('/content/ds102_v53/source')
    pm=json.loads((parent/'run_manifest.json').read_text())
    parent_mid=hashlib.sha256(json.dumps(pm,sort_keys=True).encode()).hexdigest()
    for n,h in pm['sources'].items():assert digest(src/n)==h
    expected=json.loads((inputs/'inputs_manifest.json').read_text())['files']
    assert expected==pm['inputs']
    for n,h in expected.items():assert digest(inputs/n)==h
    assert {n:importlib.metadata.version(n) for n in pm['versions']}==pm['versions']
    manifest=dict(config=CONFIG,parent_manifest_sha256=digest(parent/'run_manifest.json'),
        parent_groups_sha256=digest(parent/'groups.npz'),
        parent_sources=pm['sources'],inputs=expected,versions=pm['versions'],
        source_sha256=digest(__file__),smoke=a.smoke)
    mid=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    mp=out/'run_manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    mp.write_text(json.dumps(manifest,indent=2))
    report=dict(stage='initializing',manifest_id=mid,config=CONFIG,protocols={},
        execution='hosted Colab CPU',submission_created=False,best_user_reported_kaggle=.77436,
        caveat='Adaptive project history remains; new grouped folds are a stress test, not untouched data. Conditional bootstrap is not post-selection proof or a promise of Kaggle gain.')
    def status(stage,**kw):
        report.update(stage=stage,elapsed_seconds=time.time()-start,**kw)
        (out/'metrics_v59.json').write_text(json.dumps(report,indent=2));print(stage,json.dumps(kw),flush=True)
    try:
        data=load_data(inputs/'data');train=prepare(data['train']);y=train.spam_label.to_numpy()
        groups=np.load(parent/'groups.npz')['groups'];assert len(groups)==len(y)
        if a.smoke:
            # Deliberately unique groups for a small API/prediction check only.
            tr=np.concatenate([np.flatnonzero(y==c)[:60] for c in range(4)])
            va=np.concatenate([np.flatnonzero(y==c)[60:70] for c in range(4)])
            model=model_fitter(train.iloc[tr],np.arange(len(tr)))
            frame=train.iloc[va].drop(columns='spam_label');p0=checked(model.predict_proba(frame),len(va))
            p1=checked(model.predict_proba(frame.assign(spam_label=np.arange(len(va))%4)),len(va))
            assert np.allclose(p0,p1,atol=1e-12),'Prediction reads target labels'
            split=len(frame)//2;pb=np.vstack([model.predict_proba(frame.iloc[:split]),model.predict_proba(frame.iloc[split:])])
            assert np.allclose(p0,pb,atol=1e-12),'Prediction depends on inference batch size'
            status('smoke_passed',rows_fit=len(tr),rows_predict=len(va),eligible_for_selection=False);return
        ref=np.load(inputs/'baseline/reference_v3.npz');old=np.load(inputs/'baseline/latent_oof.npz')
        assert np.array_equal(ref['y'],y) and np.array_equal(old['y'],y)
        for protocol,group,seed in [('cluster',groups,2053),('exact',train.comment.to_numpy(),102)]:
            bp=np.zeros((len(y),4));cp=bp.copy();folds=np.full(len(y),-1);template=np.zeros(len(y),bool);deltas=[]
            splits=StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(train,y,group)
            for f,(tr,va) in enumerate(splits):
                assert not set(group[tr])&set(group[va]);assert set(y[tr])==set(y[va])=={0,1,2,3}
                folds[va]=f;folder=out/protocol/f'fold{f}';folder.mkdir(parents=True,exist_ok=True)
                if protocol=='cluster':
                    shared=parent/'cluster'/f'fold{f}'
                    if all((shared/(n+'.npz')).exists() for n in ['svm','cat','latent']):
                        z=np.load(shared/'indices.npz');assert np.array_equal(z['tr'],tr) and np.array_equal(z['va'],va)
                        ps={}
                        for n in ['svm','cat','latent']:
                            z=np.load(shared/(n+'.npz'));assert z['manifest_id'].item()==parent_mid
                            ps[n]=checked(z['p0'],len(va))
                        bp[va]=.35*ps['svm']+.35*ps['cat']+.3*ps['latent'];latent=ps['latent']
                    else:
                        bp[va],latent=baseline_partition(train.iloc[tr],train.iloc[va].drop(columns='spam_label'),folder,mid,status)
                else:
                    assert np.all(ref['folds'][va]==f) and np.all(old['folds'][va]==f)
                    bp[va]=checked(.7*ref['oof'][va]+.3*old['candidate'][va],len(va));latent=old['candidate'][va]
                ck=folder/'head.npz'
                if ck.exists():
                    z=np.load(ck);assert z['manifest_id'].item()==mid and np.array_equal(z['tr'],tr) and np.array_equal(z['va'],va)
                    p=checked(z['prediction'],len(va))
                else:
                    status('fit_context_head',protocol=protocol,fold=f+1,fold_deltas=deltas)
                    model=model_fitter(train.iloc[tr],groups[tr])
                    p=checked(model.predict_proba(train.iloc[va].drop(columns='spam_label')),len(va))
                    np.savez_compressed(ck,manifest_id=mid,tr=tr,va=va,prediction=p)
                    del model;gc.collect()
                cp[va]=checked(.85*bp[va]+.15*p,len(va))
                template[va]=max_jaccard(train.comment.iloc[va].tolist(),train.comment.iloc[tr[y[tr]==1]].tolist())>=.6
                deltas.append(macro(y[va],labels(cp[va]))-macro(y[va],labels(bp[va])))
                status('fold_complete',protocol=protocol,fold=f+1,fold_deltas=deltas)
                if min(deltas)<-.005 or sum(d>0 for d in deltas)+(2-f)<2:
                    if f==2:np.savez_compressed(out/(protocol+'_oof_rejected.npz'),y=y,folds=folds,baseline=bp,candidate=cp,template=template)
                    status('pruned_on_predeclared_fold_gates',rejected_protocol=protocol);return
            b,c=labels(bp),labels(cp);delta=macro(y,c)-macro(y,b)
            nd=class1_strata(y,c,template)['novel']['f1']-class1_strata(y,b,template)['novel']['f1']
            boot=paired_cluster_bootstrap(y,b,c,groups,n=2000,seed=2053)
            gates=dict(macro=bool(delta>=CONFIG['gates'][protocol+'_min_delta']),folds=bool(sum(d>0 for d in deltas)>=2),
                worst_fold=bool(min(deltas)>=-.005),novel_fake=bool(nd>=0),bootstrap=bool(boot['q025']>0))
            report['protocols'][protocol]=dict(baseline=metrics(y,b),candidate=metrics(y,c),delta=delta,
                fold_deltas=deltas,novel_fake_f1_delta=nd,bootstrap=boot,gates=gates)
            np.savez_compressed(out/(protocol+'_oof.npz'),y=y,folds=folds,baseline=bp,candidate=cp)
            status('protocol_complete',protocol=protocol,delta=delta,gates=gates)
            if not all(gates.values()):status('rejected_on_'+protocol);return
        dev,test=[prepare(data[n]) for n in ['dev','test']];full=np.load(inputs/'baseline/latent_full.npz')
        fm=json.loads((inputs/'baseline/latent_full_manifest.json').read_text())
        assert full['manifest_id'].item()==hashlib.sha256(json.dumps(fm,sort_keys=True).encode()).hexdigest()
        bd=checked(.7*ref['dev']+.3*full['dev'],len(dev));bt=checked(.7*ref['test']+.3*full['test'],len(test))
        sub=data['samplesubmission'].copy();sub['spam_label']=sub.ID.astype(str).map(dict(zip(test.id.astype(str),labels(bt)))).astype(int)
        assert hashlib.sha256(sub.to_csv(index=False).encode()).hexdigest()==BEST_SHA
        status('fit_full_context_head');full_model=out/'full_head.joblib'
        if full_model.exists():
            saved=joblib.load(full_model);assert saved['manifest_id']==mid;model=saved['model']
        else:
            model=model_fitter(train,groups);joblib.dump(dict(manifest_id=mid,model=model),full_model)
        pdv=checked(model.predict_proba(dev.drop(columns='spam_label')),len(dev));dp=labels(.85*bd+.15*pdv)
        yd=dev.spam_label.to_numpy();unseen=~dev.comment.isin(set(train.comment)).to_numpy()
        delta=macro(yd,dp)-macro(yd,labels(bd));ud=macro(yd[unseen],dp[unseen])-macro(yd[unseen],labels(bd)[unseen])
        status('dev_evaluated',dev_delta=delta,dev_without_exact_overlap_delta=ud,
            baseline_dev=metrics(yd,labels(bd)),candidate_dev=metrics(yd,dp))
        if delta<.001 or ud<.001:status('rejected_on_dev');return
        pt=checked(model.predict_proba(test),len(test));tp=labels(.85*bt+.15*pt)
        sub['spam_label']=sub.ID.astype(str).map(dict(zip(test.id.astype(str),tp))).astype(int)
        assert len(sub)==3974 and sub.ID.equals(data['samplesubmission'].ID) and sub.ID.is_unique and sub.spam_label.isin(range(4)).all()
        path=out/'submission_v59_candidate.csv';sub.to_csv(path,index=False)
        np.savez_compressed(out/'external_predictions.npz',dev=pdv,test=pt)
        status('candidate_ready_not_submitted',submission_created=True,rows_verified=len(sub),sha256=digest(path),test_changes_vs_best=int((tp!=labels(bt)).sum()))
    except BaseException as e:
        (out/'error.txt').write_text(traceback.format_exc());status('error',error=str(e));raise

if __name__=='__main__':
    raise SystemExit('Run experiment_v59.py, not this protocol driver.')
