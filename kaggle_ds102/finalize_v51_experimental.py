"""Exploratory full-fit follow-up of fixed V50, with stronger dev conditions.

V50's original +.002 OOF gate remains FAILED. No original report/gate is edited.
This follow-up is not a strict-CV promotion. Extra dev tests were declared
before any full-fit/dev predictions of the committee were observed.
"""
import hashlib,importlib.metadata,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from cloud_checks import guard,checked
from train_submit import load_data,metrics
from improved_features import prepare
from handoff_v43 import baseline_features,submission
from experiment_v50 import decide,label
from catalogue_svd_v40 import fit_predict as fit_catalogue
from supervised_svd_v48 import fit_predict as fit_supervised
from review_common import macro,near_dup_groups
from codex_checked_common import paired_cluster_bootstrap

ROOT=Path('/content/ds102_v51');OUT=ROOT/'run';INPUTS=Path('/content/ds102_v43/inputs')
CONFIG=dict(qualification='Exploratory follow-up, original V50 strict CV gate rejected',
    original_min_oof_gain=.002,observed_oof_gain=.001463981731077979,
    min_dev_gain=.003,min_dev_no_exact_overlap_gain=.003,require_dev_bootstrap_q05_positive=True,
    original_class1_predictions_frozen=True,baseline_user_reported_kaggle=.77436)

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    guard();OUT.mkdir(parents=True,exist_ok=True);start=time.time()
    parent=Path('/content/ds102_v50/run');cv=json.loads((parent/'metrics_v50.json').read_text())
    pm=json.loads((parent/'run_manifest.json').read_text())
    assert cv['stage']=='rejected_on_cv' and not cv['gates']['oof']
    assert all(cv['gates'][k] for k in ['folds','worst_fold','novel_fake','bootstrap'])
    assert cv['delta']>=.001 and cv['bootstrap']['q025']>0
    assert cv['novel_fake_f1_delta']==0 and cv['class1_pred_mask_exactly_preserved']
    for name in ['catalogue_svd_v40.py','supervised_svd_v48.py','experiment_v50.py','handoff_v43.py']:
        assert digest(Path(__file__).parent/name)==pm['sources'][name],name
    expected=json.loads((INPUTS/'inputs_manifest.json').read_text())['files']
    assert expected==pm['inputs']
    for n,h in expected.items():assert digest(INPUTS/n)==h,n
    versions={k:importlib.metadata.version(k) for k in pm['versions']};assert versions==pm['versions']
    manifest=dict(config=CONFIG,versions=versions,inputs=expected,
        parent_cv_sha256=digest(parent/'metrics_v50.json'),parent_manifest_sha256=digest(parent/'run_manifest.json'),
        sources={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
    mid=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    mp=OUT/'run_manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest,'Use a fresh output directory for changed source'
    mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    report=dict(config=CONFIG,cv=cv,pretrained=False,external_data=False,
        execution='hosted Colab CPU',fit_data='train only; dev/test excluded from every fitting step',
        original_cv_gate_passed=False,experimental=True,submission_created=False,
        caveat='OOF and dev have been reused for many experiments. V50 practical effect-size floor failed. This exploratory follow-up keeps the fixed rule and requires a larger dev gain and positive conditional dev bootstrap. Neither test removes selection bias or guarantees Kaggle improvement.')
    def status(stage,**kwargs):
        report.update(stage=stage,elapsed_seconds=time.time()-start,**kwargs)
        (OUT/'metrics_v51_experimental.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(stage,json.dumps(kwargs,ensure_ascii=False),flush=True)
    try:
        data=load_data(INPUTS/'data');train,dev,test=[prepare(data[k]) for k in ['train','dev','test']]
        ref=np.load(INPUTS/'baseline/reference_v3.npz');full=np.load(INPUTS/'baseline/latent_full.npz')
        assert np.array_equal(ref['y'],train.spam_label.to_numpy())
        fm=json.loads((INPUTS/'baseline/latent_full_manifest.json').read_text())
        assert full['manifest_id'].item()==hashlib.sha256(json.dumps(fm,sort_keys=True).encode()).hexdigest()
        baseline_test=label(.7*ref['test']+.3*full['test'])
        best=submission(data,test,baseline_test)
        assert hashlib.sha256(best.to_csv(index=False).encode()).hexdigest()=='05bffeff984bd4927c8dd3e9bd3613aef6a7abce303e6e1dcb7cfa4a12012059'
        report['baseline_matches_best_submission']=True
        frames=[dev.drop(columns='spam_label'),test];external={}
        for v,fn in [(47,fit_catalogue),(48,fit_supervised)]:
            folder=OUT/f'full_component{v}';folder.mkdir(exist_ok=True);cache=folder/'predictions.npz'
            status('full_train_component_start',component=v)
            if cache.exists():
                q=np.load(cache);assert q['manifest_id'].item()==mid
                external[v]=[checked(q['dev'],len(dev)),checked(q['test'],len(test))]
            else:
                fid=hashlib.sha256((mid+':full'+str(v)).encode()).hexdigest()
                if v==47:
                    path,fid=baseline_features(train,frames,folder,fid,status)
                    external[v]=fn(train,frames,folder,fid,status,path,fid)
                else:external[v]=fn(train,frames,folder,fid,status)
                temp=cache.with_suffix('.tmp.npz');np.savez_compressed(temp,manifest_id=mid,dev=external[v][0],test=external[v][1]);temp.replace(cache)
        old=label(.7*ref['dev']+.3*full['dev'])
        dp=decide(ref['dev'],full['dev'],external[47][0],external[48][0])
        assert np.array_equal(old==1,dp==1)
        y=dev.spam_label.to_numpy();unseen=~dev.comment.isin(set(train.comment)).to_numpy()
        gain=macro(y,dp)-macro(y,old);gain_unseen=macro(y[unseen],dp[unseen])-macro(y[unseen],old[unseen])
        status('dev_diagnostics',dev_delta=gain,dev_no_exact_overlap_delta=gain_unseen)
        boot=paired_cluster_bootstrap(y,old,dp,near_dup_groups(dev.comment.tolist(),thr=.7),n=5000)
        gates=dict(dev_gain=bool(gain>=.003),dev_no_exact_overlap_gain=bool(gain_unseen>=.003),
            conditional_dev_bootstrap=bool(boot['q05']>0),original_fake_decisions_preserved=bool(np.array_equal(old==1,dp==1)))
        status('dev_evaluated',baseline_dev=metrics(y,old),candidate_dev=metrics(y,dp),
            baseline_dev_without_exact_overlap=metrics(y[unseen],old[unseen]),
            candidate_dev_without_exact_overlap=metrics(y[unseen],dp[unseen]),dev_delta=gain,
            dev_no_exact_overlap_delta=gain_unseen,dev_bootstrap=boot,additional_dev_gates=gates)
        if not all(gates.values()):status('exploratory_followup_rejected_on_dev');return
        tp=decide(ref['test'],full['test'],external[47][1],external[48][1]);changes=int((tp!=baseline_test).sum())
        assert np.array_equal(tp==1,baseline_test==1)
        if changes==0:status('no_test_changes_no_candidate_export');return
        sub=submission(data,test,tp);path=OUT/'submission_v51_experimental.csv';sub.to_csv(path,index=False)
        pd.testing.assert_frame_equal(pd.read_csv(path),sub,check_dtype=False)
        status('experimental_candidate_ready_not_submitted',submission_created=True,rows_verified=len(sub),
            candidate_sha256=digest(path),test_changes_vs_best=changes)
    except BaseException as exc:
        (OUT/'error.txt').write_text(traceback.format_exc());status('error',error=str(exc));raise

if __name__=='__main__':main()
