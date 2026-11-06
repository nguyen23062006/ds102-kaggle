"""Fixed 50/50 latent dimensional ensemble; cached OOF screening, Colab only."""
import hashlib,importlib.metadata,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'claude_review'))
import numpy as np
import pandas as pd
from experiment_v10 import guard,checked,predict
from experiment_v11 import fit_latent
from train_submit import load_data,metrics
from improved_features import prepare
from review_common import near_dup_groups,max_jaccard,class1_strata,macro
from codex_checked_common import paired_cluster_bootstrap

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'output_v12_experiment'
CONFIG=dict(dimensions=[64,128],latent_average=[.5,.5],alpha=.3,
    weights=[1.,1.2,1.2,.9],seed=102,min_oof_delta=.001,
    min_positive_folds=2,min_bootstrap_q05=0.,max_novel_fake_f1_drop=.01,max_dev_drop=.005)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    guard();OUT.mkdir(exist_ok=True);start=time.time()
    previous=ROOT/'output_v11_experiment';pm=json.loads((previous/'manifest.json').read_text())
    assert digest(ROOT/'experiment_v11.py')==pm['source_sha256']
    assert {k:importlib.metadata.version(k) for k in pm['versions']}==pm['versions']
    for n,h in pm['baseline_manifest']['sha256'].items():assert digest(ROOT/n)==h,n
    manifest=dict(config=CONFIG,source_sha256=digest(__file__),v11_manifest=digest(previous/'manifest.json'),
        oof_sha256=digest(previous/'oof.npz'),versions=pm['versions'])
    mid=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    mp=OUT/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    mp.write_text(json.dumps(manifest,indent=2))
    report=dict(config=CONFIG,baseline_user_reported_kaggle=.77269,pretrained=False,
        execution='hosted Colab CPU',fit_data='train only',
        caveat='Fixed dimensional ensemble proposed after viewing V11. Adaptive repeated OOF use has selection bias; bootstrap does not remove it. Baseline is current v6 refit, not historical submission. Kaggle improvement unknown.')
    def status(stage,**kv):
        report.update(stage=stage,**kv);(OUT/'report.json').write_text(json.dumps(report,indent=2));print(stage,kv,flush=True)
    data=load_data(ROOT/'data');train,dev,test=[prepare(data[k]) for k in ['train','dev','test']]
    z=np.load(previous/'oof.npz',allow_pickle=False);ref=np.load(ROOT/'colab_reference.npz',allow_pickle=False)
    y=train.spam_label.to_numpy();folds=z['folds']
    assert np.array_equal(z['y'],y) and np.array_equal(ref['y'],y) and np.array_equal(ref['folds'],folds)
    p64=checked(z['baseline'],len(y));p128=checked(z['candidate'],len(y))
    a=predict(ref['oof'],p64);b=predict(ref['oof'],.5*p64+.5*p128)
    status('cached_oof_diagnostics')
    groups=near_dup_groups(train.comment.tolist(),thr=.7);template=np.zeros(len(y),bool)
    for f in range(3):
        va=np.flatnonzero(folds==f);tr=np.flatnonzero((folds!=f)&(y==1))
        template[va]=max_jaccard(train.comment.iloc[va].tolist(),train.comment.iloc[tr].tolist())>=.6
    delta=macro(y,b)-macro(y,a)
    fd=[macro(y[folds==f],b[folds==f])-macro(y[folds==f],a[folds==f]) for f in range(3)]
    nd=class1_strata(y,b,template)['novel']['f1']-class1_strata(y,a,template)['novel']['f1']
    boot=paired_cluster_bootstrap(y,a,b,groups,n=2000)
    gates=dict(macro=delta>=.001,folds=sum(d>0 for d in fd)>=2,novel=nd>=-.01,bootstrap=boot['q05']>0)
    status('cv_complete',baseline_oof=metrics(y,a),candidate_oof=metrics(y,b),delta=delta,
        fold_deltas=fd,novel_fake_f1_delta=nd,bootstrap=boot,gates=gates)
    if not all(gates.values()):status('rejected_on_cv',elapsed_seconds=time.time()-start);return
    frames=[dev.drop(columns='spam_label'),test];full=[]
    for dimensions in [64,128]:
        ck=OUT/f'full_{dimensions}.npz'
        if ck.exists():
            q=np.load(ck,allow_pickle=False);assert q['manifest_id'].item()==mid
            result=[checked(q['dev'],len(dev)),checked(q['test'],len(test))]
        else:
            result=fit_latent(train,frames,dimensions,status)
            tmp=ck.with_suffix('.tmp.npz');np.savez_compressed(tmp,manifest_id=mid,dev=result[0],test=result[1]);tmp.replace(ck)
        full.append(result)
    old=predict(ref['dev'],full[0][0]);dp=predict(ref['dev'],.5*full[0][0]+.5*full[1][0])
    report['dev']=dict(baseline=metrics(dev.spam_label,old),candidate=metrics(dev.spam_label,dp))
    if macro(dev.spam_label,dp)<macro(dev.spam_label,old)-.005:
        status('rejected_on_dev',elapsed_seconds=time.time()-start);return
    tp=predict(ref['test'],.5*full[0][1]+.5*full[1][1]);sub=data['samplesubmission'].copy()
    sub['spam_label']=sub.ID.astype(str).map(dict(zip(test.id.astype(str),tp))).astype(int)
    assert len(sub)==len(test) and sub.ID.is_unique and sub.spam_label.isin(range(4)).all()
    path=OUT/'submission_v12_candidate.csv';sub.to_csv(path,index=False)
    pd.testing.assert_frame_equal(pd.read_csv(path),sub,check_dtype=False)
    status('candidate_ready_not_submitted',rows=len(sub),sha256=digest(path),elapsed_seconds=time.time()-start)


if __name__=='__main__':main()
