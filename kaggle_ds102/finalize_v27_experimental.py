"""Exploratory full fit of V27 after a small positive but sub-threshold CV gain.

The original V27 rejection is preserved. This does not promote the candidate.
"""
import hashlib,importlib.metadata,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_v10 import guard,checked
from experiment_v27 import replacement,labels,CONFIG
from text_capacity_v26 import fit_predict
from train_submit import load_data,metrics
from improved_features import prepare

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'output_v27_experimental'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    guard();OUT.mkdir(exist_ok=True);start=time.time()
    cvdir=ROOT/'output_v27_experiment';cv=json.loads((cvdir/'report.json').read_text())
    cm=json.loads((cvdir/'manifest.json').read_text())
    assert cv['stage']=='rejected_on_cv' and cv['delta']>0
    assert not cv['gates']['macro'] and all(cv['gates'][k] for k in ['folds','worst_fold','novel'])
    assert {k:importlib.metadata.version(k) for k in cm['versions']}==cm['versions']
    for n,h in cm['sources'].items():assert digest(ROOT/n)==h,n
    assert digest(ROOT/'colab_components_v5.npz')==cm['components_sha256']
    fm=json.loads((ROOT/'output_v11_experimental/manifest.json').read_text())
    full=np.load(ROOT/'output_v11_experimental/full_128.npz',allow_pickle=False)
    assert digest(ROOT/'output_v11_experimental/full_128.npz')==cm['baseline_full']
    assert full['manifest_id'].item()==hashlib.sha256(json.dumps(fm,sort_keys=True).encode()).hexdigest()
    manifest=dict(cv_report_sha256=digest(cvdir/'report.json'),cv_manifest_sha256=digest(cvdir/'manifest.json'),
        sources={n:digest(ROOT/n) for n in ['finalize_v27_experimental.py','experiment_v27.py','text_capacity_v26.py']},
        cv_manifest=cm,rule='Exploratory only: positive OOF, at least2 positive folds, worst>=-.004, novel fake F1 drop<=.01; require non-decreasing dev before CSV.')
    mid=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    mp=OUT/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    mp.write_text(json.dumps(manifest,indent=2))
    report=dict(qualification='Experimental only; V27 original strict CV macro threshold failed. Original rejection preserved.',
        cv=cv,pretrained=False,fit_data='train only; dev/test excluded',execution='hosted Colab',
        cv_gain_below_threshold=True,dev_rule='No dev macro-F1 decrease allowed.',
        baseline_user_reported_kaggle=.77436,
        caveat='Adaptive OOF hypothesis and repeatedly inspected dev. Conditional interval includes zero. No confirmed Kaggle improvement.')
    def status(stage,**kv):
        report.update(stage=stage,**kv);(OUT/'metrics_v27_experimental.json').write_text(json.dumps(report,indent=2))
        print(stage,kv,flush=True)
    data=load_data(ROOT/'data');train,dev,test=[prepare(data[k]) for k in ['train','dev','test']]
    ref=np.load(ROOT/'colab_reference.npz',allow_pickle=False);comp=np.load(ROOT/'colab_components_v5.npz',allow_pickle=False)
    assert np.array_equal(ref['y'],train.spam_label.to_numpy()) and np.array_equal(comp['y'],ref['y'])
    bd=checked(.7*ref['dev']+.3*full['dev'],len(dev));bt=checked(.7*ref['test']+.3*full['test'],len(test))
    sub=data['samplesubmission'].copy()
    oldt=labels(bt);sub['spam_label']=sub.ID.astype(str).map(dict(zip(test.id.astype(str),oldt))).astype(int)
    assert hashlib.sha256(sub.to_csv(index=False).encode()).hexdigest()==CONFIG['baseline_csv_sha256']
    status('fit_full_train_text2000')
    ck=OUT/'full_text2000.npz'
    if ck.exists():
        q=np.load(ck,allow_pickle=False);assert q['manifest_id'].item()==mid
        ps=[checked(q['dev'],len(dev)),checked(q['test'],len(test))]
    else:
        feature_id=hashlib.sha256(json.dumps(dict(config=CONFIG,partition='full',data=cm['baseline_full_manifest'],frames=['dev','test']),sort_keys=True).encode()).hexdigest()
        ps=fit_predict(train,[dev.drop(columns='spam_label'),test],OUT/'full_artifacts',feature_id,status)
        tmp=ck.with_suffix('.tmp.npz');np.savez_compressed(tmp,manifest_id=mid,dev=ps[0],test=ps[1]);tmp.replace(ck)
    dp=labels(replacement(bd,ps[0],comp['cat_dev']))
    oldd=labels(bd);report['dev']=dict(baseline=metrics(dev.spam_label,oldd),candidate=metrics(dev.spam_label,dp))
    dd=report['dev']['candidate']['macro_f1']-report['dev']['baseline']['macro_f1']
    report['dev_delta']=dd;report['dev_changes']=int((dp!=oldd).sum())
    if dd<0:status('rejected_on_dev',elapsed_seconds=time.time()-start);return
    tp=labels(replacement(bt,ps[1],comp['cat_test']))
    sub=data['samplesubmission'].copy();sub['spam_label']=sub.ID.astype(str).map(dict(zip(test.id.astype(str),tp))).astype(int)
    assert len(sub)==3974 and sub.ID.is_unique and sub.ID.equals(data['samplesubmission'].ID) and sub.spam_label.isin(range(4)).all()
    path=OUT/'submission_v27_experimental.csv';sub.to_csv(path,index=False)
    pd.testing.assert_frame_equal(pd.read_csv(path),sub,check_dtype=False)
    status('experimental_candidate_ready_not_submitted',rows=len(sub),sha256=digest(path),test_changes_vs_best=int((tp!=oldt).sum()),elapsed_seconds=time.time()-start)

if __name__=='__main__':main()
