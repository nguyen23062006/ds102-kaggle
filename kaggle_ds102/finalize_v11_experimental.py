"""Explicit exploratory full fit of the fixed V11 recipe, not CV promotion.

V11 failed its original promotion gates. This separate follow-up retains that
result, evaluates dev, and creates a clearly labelled experimental submission.
No claim of a leaderboard gain; no automatic Kaggle submission.
"""
import hashlib,importlib.metadata,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_v10 import guard,checked,predict
from experiment_v11 import fit_latent
from improved_features import prepare
from train_submit import load_data,metrics

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'output_v11_experimental'
CONFIG=dict(dimensions=128,seed=102,trees=800,alpha=.3,weights=[1.,1.2,1.2,.9],
    policy='exploratory holdout evaluation and file generation; not promotion under original V11 CV gates')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    guard();OUT.mkdir(exist_ok=True);start=time.time()
    previous=ROOT/'output_v11_experiment'
    prior=json.loads((previous/'report.json').read_text());pm=json.loads((previous/'manifest.json').read_text())
    assert not all(prior['gates'].values()),'This helper is for the documented exploratory follow-up.'
    assert prior['delta']>0 and sum(d>0 for d in prior['fold_deltas'])>=2
    assert digest(ROOT/'experiment_v11.py')==pm['source_sha256']
    assert {k:importlib.metadata.version(k) for k in pm['versions']}==pm['versions']
    for n,h in pm['baseline_manifest']['sha256'].items():assert digest(ROOT/n)==h,n
    manifest=dict(config=CONFIG,source_sha256=digest(__file__),prior_manifest=digest(previous/'manifest.json'),
        prior_report_sha256=digest(previous/'report.json'),versions=pm['versions'])
    mid=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    mp=OUT/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    mp.write_text(json.dumps(manifest,indent=2))
    report=dict(config=CONFIG,baseline_user_reported_kaggle=.77269,pretrained=False,
        execution='hosted Colab CPU',fit_data='train only; dev/test excluded from fitting',
        previous_cv=dict(baseline=prior['baseline_oof']['macro_f1'],candidate=prior['candidate_oof']['macro_f1'],
            delta=prior['delta'],fold_deltas=prior['fold_deltas'],gates=prior['gates'],bootstrap=prior['bootstrap']),
        qualification='EXPERIMENTAL, prior CV promotion gates failed; Kaggle score unknown',
        caveat='Adaptive selection after repeated OOF and dev use. Current baseline refit differs from historical v6. A valid CSV is not evidence of better leaderboard performance.')
    def status(stage,**kv):
        report.update(stage=stage,**kv);(OUT/'report.json').write_text(json.dumps(report,indent=2));print(stage,kv,flush=True)
    data=load_data(ROOT/'data');train,dev,test=[prepare(data[k]) for k in ['train','dev','test']]
    ref=np.load(ROOT/'colab_reference.npz',allow_pickle=False)
    assert np.array_equal(ref['y'],train.spam_label.to_numpy())
    frames=[dev.drop(columns='spam_label'),test];predictions=[]
    for dimensions in [64,128]:
        ck=OUT/f'full_{dimensions}.npz'
        if ck.exists():
            z=np.load(ck,allow_pickle=False);assert z['manifest_id'].item()==mid
            ps=[checked(z['dev'],len(dev)),checked(z['test'],len(test))]
        else:
            ps=fit_latent(train,frames,dimensions,status)
            tmp=ck.with_suffix('.tmp.npz');np.savez_compressed(tmp,manifest_id=mid,dev=ps[0],test=ps[1]);tmp.replace(ck)
        predictions.append(ps)
    old=predict(ref['dev'],predictions[0][0]);dp=predict(ref['dev'],predictions[1][0])
    bt=predict(ref['test'],predictions[0][1]);tp=predict(ref['test'],predictions[1][1])
    report['dev']=dict(baseline=metrics(dev.spam_label,old),candidate=metrics(dev.spam_label,dp))
    report['dev_delta']=report['dev']['candidate']['macro_f1']-report['dev']['baseline']['macro_f1']
    unseen=~dev.comment.isin(set(train.comment)).to_numpy()
    report['dev_without_exact_overlap']=dict(rows=int(unseen.sum()),baseline=metrics(dev.spam_label.to_numpy()[unseen],old[unseen]),
        candidate=metrics(dev.spam_label.to_numpy()[unseen],dp[unseen]))
    report['dev_guardrail_passed']=bool(report['dev_delta']>=-.005)
    report['test_changes_vs_current_baseline']=int((tp!=bt).sum())
    sub=data['samplesubmission'].copy();ids=sub.ID.copy()
    sub['spam_label']=sub.ID.astype(str).map(dict(zip(test.id.astype(str),tp))).astype(int)
    assert len(sub)==len(test) and sub.ID.is_unique and sub.ID.equals(ids) and sub.spam_label.isin(range(4)).all()
    path=OUT/'submission_v11_experimental.csv';sub.to_csv(path,index=False)
    saved=pd.read_csv(path);pd.testing.assert_frame_equal(saved,sub,check_dtype=False)
    status('experimental_file_ready_not_submitted',rows=len(sub),columns=list(sub.columns),
        sha256=digest(path),elapsed_seconds=time.time()-start)


if __name__=='__main__':main()
