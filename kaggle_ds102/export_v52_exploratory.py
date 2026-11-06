"""Export an explicitly exploratory V50-rule CSV from actual V51 full fits.

V50 strict-CV and V51 extra-dev rejections remain unchanged. Export is a
reviewable experiment requested by the user, NOT a promotion under those gates.
Point estimates improved on OOF/dev; dev uncertainty still includes declines.
No new parameters are selected or fit here unless replay needs missing full fits.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from cloud_checks import guard,checked
from train_submit import load_data,metrics
from improved_features import prepare
from handoff_v43 import submission
from experiment_v50 import decide,label

INPUTS=Path('/content/ds102_v43/inputs');FULL=Path('/content/ds102_v51/run')
OUT=Path('/content/ds102_v52_exploratory')

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    guard();start=time.time();OUT.mkdir(exist_ok=True)
    required=[FULL/'metrics_v51_experimental.json',FULL/'full_component47/predictions.npz',FULL/'full_component48/predictions.npz']
    if not all(p.exists() for p in required):
        # Clean replay fits the exact frozen final recipe; training/dev policies
        # and the original rejection evidence are not rewritten by this export.
        from finalize_v51_experimental import main as fit_final
        fit_final()
    assert all(p.exists() for p in required)
    rp=FULL/'metrics_v51_experimental.json';before=digest(rp);r=json.loads(rp.read_text());cv=r['cv']
    assert r['experimental'] and not r['original_cv_gate_passed']
    assert cv['stage']=='rejected_on_cv' and cv['delta']>0 and cv['bootstrap']['q025']>0
    assert cv['class1_pred_mask_exactly_preserved'] and cv['novel_fake_f1_delta']==0
    assert r['dev_delta']>0 and r['dev_no_exact_overlap_delta']>0
    manifest51=json.loads((FULL/'run_manifest.json').read_text())
    mid51=hashlib.sha256(json.dumps(manifest51,sort_keys=True).encode()).hexdigest()
    expected=json.loads((INPUTS/'inputs_manifest.json').read_text())['files']
    assert expected==manifest51['inputs']
    for n,h in expected.items():assert digest(INPUTS/n)==h,n
    assert digest(Path(__file__).parent/'experiment_v50.py')==manifest51['sources']['experiment_v50.py']
    data=load_data(INPUTS/'data');train,dev,test=[prepare(data[k]) for k in ['train','dev','test']]
    ref=np.load(INPUTS/'baseline/reference_v3.npz');full=np.load(INPUTS/'baseline/latent_full.npz')
    assert np.array_equal(ref['y'],train.spam_label.to_numpy())
    fm=json.loads((INPUTS/'baseline/latent_full_manifest.json').read_text())
    assert full['manifest_id'].item()==hashlib.sha256(json.dumps(fm,sort_keys=True).encode()).hexdigest()
    ps={}
    for v in [47,48]:
        q=np.load(FULL/f'full_component{v}/predictions.npz',allow_pickle=False)
        assert q['manifest_id'].item()==mid51
        ps[v]=[checked(q['dev'],len(dev)),checked(q['test'],len(test))]
    old_dev=label(.7*ref['dev']+.3*full['dev']);old_test=label(.7*ref['test']+.3*full['test'])
    dp=decide(ref['dev'],full['dev'],ps[47][0],ps[48][0]);tp=decide(ref['test'],full['test'],ps[47][1],ps[48][1])
    assert np.array_equal(dp==1,old_dev==1) and np.array_equal(tp==1,old_test==1)
    dev_metrics=metrics(dev.spam_label.to_numpy(),dp)
    assert abs(dev_metrics['macro_f1']-r['candidate_dev']['macro_f1'])<1e-12
    best=submission(data,test,old_test)
    assert hashlib.sha256(best.to_csv(index=False).encode()).hexdigest()=='05bffeff984bd4927c8dd3e9bd3613aef6a7abce303e6e1dcb7cfa4a12012059'
    changes=int((tp!=old_test).sum());assert changes>0,'Same labels as best file; no new candidate'
    sub=submission(data,test,tp);path=OUT/'submission_v52_exploratory.csv';sub.to_csv(path,index=False)
    pd.testing.assert_frame_equal(pd.read_csv(path),sub,check_dtype=False)
    assert digest(rp)==before,'Original rejection report must remain unchanged'
    report=dict(stage='exploratory_candidate_ready_not_submitted',experimental=True,
        recommendation='Exploratory test only; NOT a strict-CV promotion or confirmed Kaggle improvement',
        original_cv_gate_passed=False,additional_dev_gates_passed=all(r['additional_dev_gates'].values()),
        original_rejections_preserved=True,original_followup_report_sha256=before,
        baseline_user_reported_kaggle=.77436,candidate_kaggle_score=None,
        cv=cv,baseline_dev=r['baseline_dev'],candidate_dev=dev_metrics,dev_delta=r['dev_delta'],
        dev_no_exact_overlap_delta=r['dev_no_exact_overlap_delta'],dev_bootstrap=r['dev_bootstrap'],
        original_additional_dev_gates=r['additional_dev_gates'],
        rows_verified=len(sub),test_changes_vs_best=changes,dev_changes_vs_best=int((dp!=old_dev).sum()),
        original_fake_review_prediction_mask_preserved=True,
        prediction_counts={str(c):int((tp==c).sum()) for c in range(4)},
        candidate_sha256=digest(path),pretrained=False,external_data=False,
        fit_data='train only; dev/test excluded from all fitting',execution='hosted Colab CPU',
        recipe='Fixed agreement of catalogueSVD32+SVD128 CatBoost and supervised-column-SVD128 CatBoost; base remainsV11. Only change non-class1 predictions if both fixed model views agree on0/2/3.',
        caveat='The original .002 OOF floor and extra .003 dev/bootstrap conditions were not met. OOF/dev have been repeatedly reused. Both point estimates increased, but dev interval includes decreases. Do not claim this file will improve Kaggle. No test labels or external model weights used.',
        elapsed_export_seconds=time.time()-start)
    (OUT/'metrics_v52_exploratory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    manifest=dict(source_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        inputs=expected,full_manifest_sha256=digest(FULL/'run_manifest.json'),
        full_prediction_sha256={str(v):digest(FULL/f'full_component{v}/predictions.npz') for v in [47,48]},
        original_cv_report_sha256=digest(Path('/content/ds102_v50/run/metrics_v50.json')),
        original_dev_rejection_report_sha256=before,submission_sha256=digest(path))
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({k:report[k] for k in ['stage','rows_verified','test_changes_vs_best','dev_delta','original_cv_gate_passed','additional_dev_gates_passed','candidate_sha256']}),flush=True)

if __name__=='__main__':main()
