"""Strongly regularized residual combination of three train-only model views."""
import hashlib,importlib.metadata,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'claude_review'))
import numpy as np
import pandas as pd
from scipy.special import softmax,logsumexp
from scipy.optimize import minimize
from experiment_v10 import guard,checked
from train_submit import load_data,metrics
from improved_features import prepare
from review_common import near_dup_groups,class1_strata,macro
from codex_checked_common import paired_cluster_bootstrap

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'output_v36_experiment'
CONFIG=dict(ridge=2.,parameter_bound=.5,template_positive_weight=.5,
    baseline_class_weights=[1,1.2,1.2,.9],min_oof_gain=.002,min_novel_fake_f1_gain=.01,
    min_dev_gain=.003,max_fold_drop=.003,seed=102,
    baseline_user_reported_kaggle=.77436,baseline_sha256='05bffeff984bd4927c8dd3e9bd3613aef6a7abce303e6e1dcb7cfa4a12012059')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def design(base,views):
    w=np.array(CONFIG['baseline_class_weights'])
    q=np.clip(base*w,1e-8,None);q/=q.sum(1,keepdims=True)
    features=[]
    for p in views:
        v=np.clip(p*w,1e-8,None);v/=v.sum(1,keepdims=True)
        features.append(np.clip(np.log(v)-np.log(q),-4,4)/4)
    x=np.stack(features,axis=2)
    entropy=-(q*np.log(q)).sum(1)/np.log(4)
    x=np.concatenate([x,np.repeat(entropy[:,None,None],4,axis=1),np.ones((len(q),4,1))],axis=2)
    assert x.shape==(len(q),4,5) and np.isfinite(x).all()
    return np.log(q),x

def fit(logbase,x,y,novel):
    guard();counts=np.bincount(y,minlength=4);assert (counts>0).all()
    sw=1/counts[y].astype(float)
    sw[(y==1)&~novel]*=.5
    for c in range(4):sw[y==c]/=sw[y==c].sum()
    sw/=sw.sum()
    target=np.eye(4)[y];ridge=CONFIG['ridge']
    def objective(flat):
        theta=flat.reshape(4,5);logits=logbase+np.einsum('ncf,cf->nc',x,theta)
        loss=float(np.sum(sw*(logsumexp(logits,axis=1)-logits[np.arange(len(y)),y]))+.5*ridge*np.sum(theta*theta))
        residual=(softmax(logits,axis=1)-target)*sw[:,None]
        grad=np.einsum('nc,ncf->cf',residual,x)+ridge*theta
        return loss,grad.ravel()
    r=minimize(objective,np.zeros(20),jac=True,method='L-BFGS-B',
        bounds=[(-.5,.5)]*20,options={'maxiter':200,'gtol':1e-8,'ftol':1e-12})
    assert r.success and np.isfinite(r.x).all(),str(r.message)
    return r.x.reshape(4,5),dict(iterations=int(r.nit),objective=float(r.fun),parameter_norm=float(np.linalg.norm(r.x)))
def predict(logbase,x,theta):return (logbase+np.einsum('ncf,cf->nc',x,theta)).argmax(1)

def main():
    guard();OUT.mkdir(exist_ok=True);start=time.time()
    pm=json.loads((ROOT/'output_v11_experiment/manifest.json').read_text())
    assert digest(ROOT/'experiment_v11.py')==pm['source_sha256']
    assert {k:importlib.metadata.version(k) for k in pm['versions']}==pm['versions']
    for n,h in pm['baseline_manifest']['sha256'].items():assert digest(ROOT/n)==h,n
    fm=json.loads((ROOT/'output_v11_experimental/manifest.json').read_text())
    full=np.load(ROOT/'output_v11_experimental/full_128.npz',allow_pickle=False)
    assert full['manifest_id'].item()==hashlib.sha256(json.dumps(fm,sort_keys=True).encode()).hexdigest()
    manifest=dict(config=CONFIG,source_sha256=digest(__file__),versions=pm['versions'],
        inputs={n:digest(ROOT/n) for n in ['output_v11_experiment/oof.npz','output_v11_experiment/manifest.json',
        'output_v11_experimental/full_128.npz','output_v11_experimental/manifest.json','colab_components_v5.npz',
        'colab_reference.npz','output_v33_experiment/calibration.npz']})
    mp=OUT/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    mp.write_text(json.dumps(manifest,indent=2))
    report=dict(config=CONFIG,pretrained=False,execution='hosted Colab',
        fit_data='meta calibration on train OOF only; dev/test excluded',
        caveat='Existing-OOF meta cross-validation is conditional and NOT fully nested: upstream fitted models overlap. Adaptive repeated OOF/dev selection remains. Macro-weighted loss is a surrogate for macroF1. No confirmed Kaggle gain.')
    def status(stage,**kv):
        report.update(stage=stage,**kv);(OUT/'report.json').write_text(json.dumps(report,indent=2));print(stage,kv,flush=True)
    data=load_data(ROOT/'data');train,dev,test=[prepare(data[k]) for k in ['train','dev','test']]
    z=np.load(ROOT/'output_v11_experiment/oof.npz',allow_pickle=False);ref=np.load(ROOT/'colab_reference.npz',allow_pickle=False)
    comp=np.load(ROOT/'colab_components_v5.npz',allow_pickle=False);n=np.load(ROOT/'output_v33_experiment/calibration.npz',allow_pickle=False)
    y=train.spam_label.to_numpy();folds=z['folds']
    assert all(np.array_equal(v['y'],y) and np.array_equal(v['folds'],folds) for v in [z,ref,comp,n])
    novel=n['novel'];template=~novel
    base=checked(.7*ref['oof']+.3*z['candidate'],len(y))
    lb,x=design(base,[softmax(3*comp['svm_oof'],axis=1),comp['cat_oof'],z['candidate']])
    a=lb.argmax(1);b=np.empty(len(y),int);fitinfo=[]
    status('fit_residual_crossfit')
    for f in range(3):
        fitmask=folds!=f;theta,info=fit(lb[fitmask],x[fitmask],y[fitmask],novel[fitmask])
        b[~fitmask]=predict(lb[~fitmask],x[~fitmask],theta);fitinfo.append(info)
    final_theta,finalinfo=fit(lb,x,y,novel)
    delta=macro(y,b)-macro(y,a);fd=[macro(y[folds==f],b[folds==f])-macro(y[folds==f],a[folds==f]) for f in range(3)]
    nd=class1_strata(y,b,template)['novel']['f1']-class1_strata(y,a,template)['novel']['f1']
    status('diagnostics')
    groups=near_dup_groups(train.comment.tolist(),thr=.7);boot=paired_cluster_bootstrap(y,a,b,groups,n=2000)
    gates=dict(macro=bool(delta>=.002),folds=bool(sum(v>0 for v in fd)>=2),
        worst=bool(min(fd)>=-.003),novel=bool(nd>=.01),conditional_bootstrap=bool(boot['q05']>0))
    status('cv_complete',baseline=metrics(y,a),conditional_crossfit_candidate=metrics(y,b),delta=delta,
        fold_deltas=fd,novel_fake_f1_delta=nd,fit_info=fitinfo,final_fit_info=finalinfo,
        final_theta=final_theta.tolist(),full_fit_score_not_validation=macro(y,predict(lb,x,final_theta)),bootstrap=boot,gates=gates)
    np.savez_compressed(OUT/'residual.npz',theta=final_theta,y=y,folds=folds,pred=b)
    if not all(gates.values()):status('rejected_on_cv',elapsed_seconds=time.time()-start);return
    output=[]
    for part,df in [('dev',dev),('test',test)]:
        basep=checked(.7*ref[part]+.3*full[part],len(df))
        l,v=design(basep,[softmax(3*comp['svm_'+part],axis=1),comp['cat_'+part],full[part]])
        output.append((l.argmax(1),predict(l,v,final_theta)))
    oldd,dp=output[0];oldt,tp=output[1]
    sub=data['samplesubmission'].copy();sub['spam_label']=sub.ID.astype(str).map(dict(zip(test.id.astype(str),oldt))).astype(int)
    assert hashlib.sha256(sub.to_csv(index=False).encode()).hexdigest()==CONFIG['baseline_sha256']
    report['dev']=dict(baseline=metrics(dev.spam_label,oldd),candidate=metrics(dev.spam_label,dp))
    dd=report['dev']['candidate']['macro_f1']-report['dev']['baseline']['macro_f1']
    if dd<.003:status('rejected_on_dev',dev_delta=dd,elapsed_seconds=time.time()-start);return
    sub=data['samplesubmission'].copy();sub['spam_label']=sub.ID.astype(str).map(dict(zip(test.id.astype(str),tp))).astype(int)
    assert len(sub)==3974 and sub.ID.equals(data['samplesubmission'].ID) and sub.ID.is_unique and sub.spam_label.isin(range(4)).all()
    path=OUT/'submission_v36.csv';sub.to_csv(path,index=False);pd.testing.assert_frame_equal(pd.read_csv(path),sub,check_dtype=False)
    status('candidate_ready_not_submitted',dev_delta=dd,rows=len(sub),sha256=digest(path),test_changes_vs_best=int((tp!=oldt).sum()),elapsed_seconds=time.time()-start)

if __name__=='__main__':main()
