"""Novelty-aware class1 decision calibration from train OOF only.

Cross-fitting a calibrator on existing OOF is not fully nested model CV.
"""
import hashlib,importlib.metadata,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'claude_review'))
import numpy as np
import pandas as pd
from experiment_v10 import guard,checked
from train_submit import load_data,metrics
from improved_features import prepare
from review_common import near_dup_groups,max_jaccard,class1_strata,macro
from codex_checked_common import paired_cluster_bootstrap

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'output_v33_experiment'
CONFIG=dict(novel_class1_weights=[.8,1.,1.2,1.4,1.6],template_class1_weight=1.2,
    baseline_weights=[1.,1.2,1.2,.9],template_jaccard=.6,
    min_oof_gain=.0015,min_novel_fake_f1_gain=.01,min_dev_gain=.003,
    max_fold_drop=.003,min_positive_folds=2,baseline_user_reported_kaggle=.77436,
    baseline_sha256='05bffeff984bd4927c8dd3e9bd3613aef6a7abce303e6e1dcb7cfa4a12012059')

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def predict(p,novel,w):
    scores=p*np.array(CONFIG['baseline_weights'])
    scores[novel,1]=p[novel,1]*w
    return scores.argmax(1)
def choose(p,y,novel):
    assert (y[novel]==1).any()
    ranked=[]
    for w in CONFIG['novel_class1_weights']:
        pred=predict(p,novel,w)
        ranked.append((macro(y[novel],pred[novel]),-abs(w-1.2),-w,w))
    return max(ranked)[-1]

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
        oof_sha256=digest(ROOT/'output_v11_experiment/oof.npz'),full_sha256=digest(ROOT/'output_v11_experimental/full_128.npz'),
        baseline_manifest=digest(ROOT/'output_v11_experiment/manifest.json'),full_manifest=digest(ROOT/'output_v11_experimental/manifest.json'))
    mp=OUT/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    mp.write_text(json.dumps(manifest,indent=2))
    report=dict(config=CONFIG,pretrained=False,execution='hosted Colab',fit_data='decision calibration on train OOF only; dev/test excluded',
        caveat='Adaptive novelty-calibration hypothesis. Existing OOF cross-calibration is not fully nested CV because upstream fits overlap. Repeated OOF/dev selection bias; Kaggle improvement unknown.')
    def status(stage,**kv):
        report.update(stage=stage,**kv);(OUT/'report.json').write_text(json.dumps(report,indent=2));print(stage,kv,flush=True)
    data=load_data(ROOT/'data');train,dev,test=[prepare(data[k]) for k in ['train','dev','test']]
    z=np.load(ROOT/'output_v11_experiment/oof.npz',allow_pickle=False);ref=np.load(ROOT/'colab_reference.npz',allow_pickle=False)
    y=train.spam_label.to_numpy();folds=z['folds']
    assert np.array_equal(z['y'],y) and np.array_equal(ref['y'],y) and np.array_equal(folds,ref['folds'])
    p=checked(.7*ref['oof']+.3*z['candidate'],len(y))
    status('compute_fold_safe_novelty');template=np.zeros(len(y),bool)
    for f in range(3):
        va=np.flatnonzero(folds==f);tr=np.flatnonzero((folds!=f)&(y==1))
        template[va]=max_jaccard(train.comment.iloc[va].tolist(),train.comment.iloc[tr].tolist())>=.6
    novel=~template;a=predict(p,novel,1.2);b=np.empty(len(y),int);ws=[]
    for f in range(3):
        fit=folds!=f;w=choose(p[fit],y[fit],novel[fit]);ws.append(w)
        b[~fit]=predict(p[~fit],novel[~fit],w)
    final_w=choose(p,y,novel);fixed=predict(p,novel,final_w)
    fd=[macro(y[folds==f],b[folds==f])-macro(y[folds==f],a[folds==f]) for f in range(3)]
    delta=macro(y,b)-macro(y,a);noveld=class1_strata(y,b,template)['novel']['f1']-class1_strata(y,a,template)['novel']['f1']
    groups=near_dup_groups(train.comment.tolist(),thr=.7);boot=paired_cluster_bootstrap(y,a,b,groups,n=2000)
    gates=dict(macro=delta>=CONFIG['min_oof_gain'],folds=sum(v>0 for v in fd)>=2,
        worst=min(fd)>=-CONFIG['max_fold_drop'],novel=noveld>=CONFIG['min_novel_fake_f1_gain'],
        final_supported=bool(sum(np.isclose(w,final_w) for w in ws)>=2),final_changed=bool(not np.isclose(final_w,1.2)))
    status('cv_complete',baseline=metrics(y,a),crossfit_candidate=metrics(y,b),delta=delta,
        fold_deltas=fd,fold_novel_weights=ws,final_novel_weight=final_w,
        novel_fake_f1_delta=noveld,novel_fake_before=class1_strata(y,a,template),novel_fake_after=class1_strata(y,b,template),
        full_weight_fit_score_not_validation=macro(y,fixed),bootstrap=boot,gates=gates)
    np.savez_compressed(OUT/'calibration.npz',y=y,folds=folds,novel=novel,crossfit_pred=b,final_novel_weight=final_w)
    if not all(gates.values()):status('rejected_on_cv',elapsed_seconds=time.time()-start);return
    frames=[dev,test];fullnovel=[]
    for df in frames:fullnovel.append(max_jaccard(df.comment.tolist(),train.comment[y==1].tolist())<.6)
    bd=checked(.7*ref['dev']+.3*full['dev'],len(dev));bt=checked(.7*ref['test']+.3*full['test'],len(test))
    oldd=predict(bd,fullnovel[0],1.2);oldt=predict(bt,fullnovel[1],1.2)
    sub=data['samplesubmission'].copy();sub['spam_label']=sub.ID.astype(str).map(dict(zip(test.id.astype(str),oldt))).astype(int)
    assert hashlib.sha256(sub.to_csv(index=False).encode()).hexdigest()==CONFIG['baseline_sha256']
    dp=predict(bd,fullnovel[0],final_w)
    report['dev']=dict(baseline=metrics(dev.spam_label,oldd),candidate=metrics(dev.spam_label,dp))
    dd=report['dev']['candidate']['macro_f1']-report['dev']['baseline']['macro_f1']
    if dd<CONFIG['min_dev_gain']:status('rejected_on_dev',dev_delta=dd,elapsed_seconds=time.time()-start);return
    tp=predict(bt,fullnovel[1],final_w);sub=data['samplesubmission'].copy()
    sub['spam_label']=sub.ID.astype(str).map(dict(zip(test.id.astype(str),tp))).astype(int)
    assert len(sub)==3974 and sub.ID.equals(data['samplesubmission'].ID) and sub.ID.is_unique and sub.spam_label.isin(range(4)).all()
    path=OUT/'submission_v33.csv';sub.to_csv(path,index=False);pd.testing.assert_frame_equal(pd.read_csv(path),sub,check_dtype=False)
    status('candidate_ready_not_submitted',dev_delta=dd,rows=len(sub),sha256=digest(path),test_changes_vs_best=int((tp!=oldt).sum()),elapsed_seconds=time.time()-start)

if __name__=='__main__':main()
