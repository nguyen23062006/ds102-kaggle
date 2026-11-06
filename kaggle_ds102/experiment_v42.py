"""Conditional OOF residual stacker for class0/class2 only; no pretrained.

Uses all five existing score views. Existing-OOF meta CV is conditional and
NOT fully nested, because upstream fits overlap. No development/test fitting.
"""
import hashlib,importlib.metadata,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'claude_review'))
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,softmax
from experiment_v10 import guard,checked
from train_submit import load_data,metrics
from improved_features import prepare
from review_common import macro,near_dup_groups
from codex_checked_common import paired_cluster_bootstrap

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'output_v42_experiment'
CONFIG=dict(ridge=.5,parameter_bound=.75,positive_class2_weight=2.,
    baseline_class_weights=[1.,1.2,1.2,.9],log_ratio_clip=6.,
    min_oof_gain=.002,min_positive_folds=2,max_fold_drop=.003,
    min_dev_gain=.001,seed=102,baseline_user_reported_kaggle=.77436,
    frozen_decisions=[1,3],views=['svm','cat700','latent128','bpe_svm','cat2000'])

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def baseline_labels(base):return (base*np.asarray(CONFIG['baseline_class_weights'])).argmax(1)
def odds(p):return np.log(np.maximum(p[:,2],1e-8))-np.log(np.maximum(p[:,0],1e-8))
def features(base,views):
    offset=odds(base)+np.log(1.2)
    x=np.stack([np.clip(odds(v)-odds(base),-6,6) for v in views],axis=1)
    return offset,x

def fit(offset,x,y):
    assert np.array_equal(np.unique(y),[0,2])
    mean=x.mean(0);scale=np.maximum(x.std(0),.1)
    a=np.column_stack([(x-mean)/scale,np.ones(len(x))])
    target=(y==2).astype(float);weight=np.where(y==2,2.,1.);denom=weight.sum()
    def objective(theta):
        score=offset+a@theta
        loss=np.sum(weight*(np.logaddexp(0,score)-target*score))/denom
        loss+=.5*CONFIG['ridge']*np.dot(theta,theta)
        grad=a.T@(weight*(expit(score)-target))/denom+CONFIG['ridge']*theta
        return loss,grad
    result=minimize(objective,np.zeros(a.shape[1]),jac=True,method='L-BFGS-B',
        bounds=[(-.75,.75)]*a.shape[1],options=dict(maxiter=200,ftol=1e-12))
    assert result.success and np.isfinite(result.x).all(),result.message
    return dict(theta=result.x,mean=mean,scale=scale),dict(iterations=int(result.nit),
        objective=float(result.fun),parameter_norm=float(np.linalg.norm(result.x)))

def predict(base,offset,x,params):
    pred=baseline_labels(base);mask=np.isin(pred,[0,2]);original=pred.copy()
    a=np.column_stack([(x-params['mean'])/params['scale'],np.ones(len(x))])
    score=offset+a@params['theta']
    pred[mask]=np.where(score[mask]>0,2,0)
    assert np.array_equal(pred[~mask],original[~mask])
    assert np.array_equal(pred==1,original==1) and np.array_equal(pred==3,original==3)
    return pred

def main():
    guard();OUT.mkdir(exist_ok=True);start=time.time()
    pm=json.loads((ROOT/'output_v11_experiment/manifest.json').read_text())
    assert {k:importlib.metadata.version(k) for k in pm['versions']}==pm['versions']
    for name,h in pm['baseline_manifest']['sha256'].items():assert digest(ROOT/name)==h,name
    paths=['output_v11_experiment/oof.npz','colab_reference.npz','colab_components_v5.npz',
        'output_v29_experiment/oof.npz']+[f'output_v26_experiment/fold{f}_text2000.npz' for f in range(3)]
    manifest=dict(config=CONFIG,source_sha256=digest(__file__),versions=pm['versions'],
        inputs={name:digest(ROOT/name) for name in paths})
    mp=OUT/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    mp.write_text(json.dumps(manifest,indent=2))
    report=dict(config=CONFIG,pretrained=False,execution='hosted Colab',
        fit_data='train OOF calibration only, no dev/test',
        caveat='Adaptive hypothesis from aggregate OOF error complementarity. Conditional meta CV is not fully nested and bootstrap does not remove repeated selection/upstream overlap. No confirmed Kaggle gain.')
    def status(stage,**kwargs):
        report.update(stage=stage,**kwargs);(OUT/'report.json').write_text(json.dumps(report,indent=2));print(stage,kwargs,flush=True)
    z=np.load(ROOT/'output_v11_experiment/oof.npz');ref=np.load(ROOT/'colab_reference.npz');c=np.load(ROOT/'colab_components_v5.npz');alt=np.load(ROOT/'output_v29_experiment/oof.npz')
    y=z['y'];folds=z['folds']
    for a in [ref,c,alt]:assert np.array_equal(a['y'],y) and np.array_equal(a['folds'],folds)
    assert digest(ROOT/'colab_components_v5.npz')=='a97e1e591d06cd8634d9709db3303cd106d9118efece504fa2e2f69a266e08cb'
    newcat=np.zeros_like(ref['oof']);seen=np.zeros(len(y),bool)
    for f in range(3):
        q=np.load(ROOT/f'output_v26_experiment/fold{f}_text2000.npz');va=q['va']
        assert np.all(folds[va]==f) and not seen[va].any();newcat[va]=q['prediction'];seen[va]=True
    assert seen.all()
    base=checked(.7*ref['oof']+.3*z['candidate'],len(y))
    views=[softmax(3*c['svm_oof'],axis=1),c['cat_oof'],z['candidate'],alt['bpe_svm'],checked(newcat,len(y))]
    offset,x=features(base,views);a=baseline_labels(base);b=a.copy();infos=[]
    zero=dict(theta=np.zeros(6),mean=np.zeros(5),scale=np.ones(5))
    assert np.array_equal(predict(base,offset,x,zero),a),'Zero residual must reproduce baseline'
    fit_rows=np.isin(y,[0,2])&np.isin(a,[0,2])
    for f in range(3):
        tr=(folds!=f)&fit_rows;va=folds==f
        params,info=fit(offset[tr],x[tr],y[tr]);b[va]=predict(base[va],offset[va],x[va],params)
        infos.append(dict(fold=f,**info,theta=params['theta'].tolist()))
    assert np.array_equal(a==1,b==1) and np.array_equal(a==3,b==3)
    delta=macro(y,b)-macro(y,a);fd=[macro(y[folds==f],b[folds==f])-macro(y[folds==f],a[folds==f]) for f in range(3)]
    params,info=fit(offset[fit_rows],x[fit_rows],y[fit_rows])
    train=prepare(load_data(ROOT/'data')['train']);assert np.array_equal(train.spam_label.to_numpy(),y)
    status('bootstrap_diagnostics',delta=delta,fold_deltas=fd)
    groups=near_dup_groups(train.comment.tolist(),thr=.7)
    boot=paired_cluster_bootstrap(y,a,b,groups,n=2000)
    gates=dict(macro=bool(delta>=.002),folds=bool(sum(d>0 for d in fd)>=2),
        worst=bool(min(fd)>=-.003),conditional_bootstrap=bool(boot['q05']>0))
    np.savez_compressed(OUT/'meta_params.npz',**params,y=y,folds=folds,crossfit_pred=b)
    status('cv_eligible_pending_dev' if all(gates.values()) else 'rejected_on_cv',
        baseline=metrics(y,a),conditional_crossfit_candidate=metrics(y,b),delta=delta,fold_deltas=fd,
        class1_and3_predictions_exactly_preserved=True,fit_info=infos,full_fit_info=info,
        bootstrap=boot,gates=gates,elapsed_seconds=time.time()-start,submission_created=False)

if __name__=='__main__':main()
