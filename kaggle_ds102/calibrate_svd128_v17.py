"""Re-fit the existing coarse decision-weight recipe for SVD128 probabilities.

Calibration labels are train OOF only. Cross-fitted calibration is NOT fully
nested model CV: shared upstream fitting and repeated selection remain caveats.
"""
import hashlib,importlib.metadata,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'claude_review'))
import numpy as np
import pandas as pd
from experiment_v10 import guard,checked
from selection_v5 import fit_weights
from train_submit import load_data,metrics
from improved_features import prepare
from review_common import near_dup_groups,max_jaccard,class1_strata,macro
from codex_checked_common import paired_cluster_bootstrap

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'output_v17_experiment'
CONFIG=dict(baseline_weights=[1.,1.2,1.2,.9],grid='existing selection_v5.fit_weights unchanged',
    calibration='fit other two OOF folds, apply held-out fold',baseline_user_reported_kaggle=.77436,
    min_crossfit_delta=.001,min_positive_folds=2,max_fold_drop=.004,max_novel_fake_f1_drop=.01,
    max_dev_drop=.005,baseline_sha256='05bffeff984bd4927c8dd3e9bd3613aef6a7abce303e6e1dcb7cfa4a12012059')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    guard();OUT.mkdir(exist_ok=True);start=time.time()
    previous=ROOT/'output_v11_experiment';pm=json.loads((previous/'manifest.json').read_text())
    assert digest(ROOT/'experiment_v11.py')==pm['source_sha256']
    assert {k:importlib.metadata.version(k) for k in pm['versions']}==pm['versions']
    for n,h in pm['baseline_manifest']['sha256'].items():assert digest(ROOT/n)==h,n
    full_dir=ROOT/'output_v11_experimental';fm=json.loads((full_dir/'manifest.json').read_text())
    full_mid=hashlib.sha256(json.dumps(fm,sort_keys=True).encode()).hexdigest()
    full=np.load(full_dir/'full_128.npz',allow_pickle=False);assert full['manifest_id'].item()==full_mid
    manifest=dict(config=CONFIG,sources={n:digest(ROOT/n) for n in ['calibrate_svd128_v17.py','selection_v5.py']},
        baseline_manifest=digest(previous/'manifest.json'),oof=digest(previous/'oof.npz'),
        full=digest(full_dir/'full_128.npz'),full_manifest=digest(full_dir/'manifest.json'),versions=pm['versions'])
    mp=OUT/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    mp.write_text(json.dumps(manifest,indent=2))
    report=dict(config=CONFIG,pretrained=False,execution='hosted Colab CPU',
        fit_data='calibration on train OOF only; no dev/test fitting',
        caveat='Other-fold OOF calibrators do not constitute fully nested model validation. Many earlier trials reused OOF/dev; selection bias remains. Bootstrap conditional; Kaggle improvement unknown.')
    def status(stage,**kv):
        report.update(stage=stage,**kv);(OUT/'report.json').write_text(json.dumps(report,indent=2));print(stage,kv,flush=True)
    data=load_data(ROOT/'data');train,dev,test=[prepare(data[k]) for k in ['train','dev','test']]
    z=np.load(previous/'oof.npz',allow_pickle=False);ref=np.load(ROOT/'colab_reference.npz',allow_pickle=False)
    y=train.spam_label.to_numpy();folds=z['folds']
    assert np.array_equal(z['y'],y) and np.array_equal(ref['y'],y) and np.array_equal(ref['folds'],folds)
    p=checked(.7*ref['oof']+.3*z['candidate'],len(y));oldw=np.array(CONFIG['baseline_weights'])
    a=(p*oldw).argmax(1);b=np.empty(len(y),dtype=int);fold_weights=[]
    status('crossfit_calibration')
    for fold in range(3):
        fit=folds!=fold;w=fit_weights(p[fit],y[fit]);b[~fit]=(p[~fit]*w).argmax(1);fold_weights.append(w.tolist())
    w=fit_weights(p,y)
    fd=[macro(y[folds==f],b[folds==f])-macro(y[folds==f],a[folds==f]) for f in range(3)]
    groups=near_dup_groups(train.comment.tolist(),thr=.7);template=np.zeros(len(y),bool)
    for f in range(3):
        va=np.flatnonzero(folds==f);tr=np.flatnonzero((folds!=f)&(y==1))
        template[va]=max_jaccard(train.comment.iloc[va].tolist(),train.comment.iloc[tr].tolist())>=.6
    delta=macro(y,b)-macro(y,a);nd=class1_strata(y,b,template)['novel']['f1']-class1_strata(y,a,template)['novel']['f1']
    boot=paired_cluster_bootstrap(y,a,b,groups,n=2000)
    gates=dict(macro=delta>=.001,folds=sum(d>0 for d in fd)>=2,worst_fold=min(fd)>=-.004,novel=nd>=-.01)
    status('cv_complete',baseline_oof=metrics(y,a),crossfit_oof=metrics(y,b),delta=delta,fold_deltas=fd,
        fold_weights=fold_weights,final_weights=w.tolist(),novel_fake_f1_delta=nd,bootstrap=boot,gates=gates,
        full_weight_fit_score_not_validation=macro(y,(p*w).argmax(1)))
    if not all(gates.values()):status('rejected_on_crossfit',elapsed_seconds=time.time()-start);return
    if np.allclose(w,oldw):status('no_final_weight_change',elapsed_seconds=time.time()-start);return
    bd=checked(.7*ref['dev']+.3*full['dev'],len(dev));bt=checked(.7*ref['test']+.3*full['test'],len(test))
    old=(bd*oldw).argmax(1);dp=(bd*w).argmax(1);baseline_test=(bt*oldw).argmax(1)
    sub=data['samplesubmission'].copy();sub['spam_label']=sub.ID.astype(str).map(dict(zip(test.id.astype(str),baseline_test))).astype(int)
    assert hashlib.sha256(sub.to_csv(index=False).encode()).hexdigest()==CONFIG['baseline_sha256']
    report['baseline_matches_best_submission']=True
    report['dev']=dict(baseline=metrics(dev.spam_label,old),candidate=metrics(dev.spam_label,dp))
    report['dev_delta']=report['dev']['candidate']['macro_f1']-report['dev']['baseline']['macro_f1']
    if report['dev_delta']<-.005:status('rejected_on_dev',elapsed_seconds=time.time()-start);return
    tp=(bt*w).argmax(1);sub=data['samplesubmission'].copy();ids=sub.ID.copy()
    sub['spam_label']=sub.ID.astype(str).map(dict(zip(test.id.astype(str),tp))).astype(int)
    assert len(sub)==len(test) and sub.ID.is_unique and sub.ID.equals(ids) and sub.spam_label.isin(range(4)).all()
    path=OUT/'submission_v17.csv';sub.to_csv(path,index=False)
    pd.testing.assert_frame_equal(pd.read_csv(path),sub,check_dtype=False)
    status('candidate_ready_not_submitted',rows=len(sub),sha256=digest(path),test_changes_vs_best=int((tp!=baseline_test).sum()),elapsed_seconds=time.time()-start)


if __name__=='__main__':main()
