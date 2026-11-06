"""Exploratory class2-only use of the cached BoW2000 text branch."""
import hashlib,importlib.metadata,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'claude_review'))
import numpy as np
from scipy.special import softmax
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from experiment_v10 import guard,checked
from text_capacity_v26 import CONFIG as PROJECTION_CONFIG,fit_predict
from train_submit import load_data,metrics
from improved_features import prepare
from review_common import near_dup_groups,max_jaccard,class1_strata,macro
from codex_checked_common import paired_cluster_bootstrap

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'output_v27_experiment'
CONFIG=dict(projection=PROJECTION_CONFIG,representation='Class2-only text capacity update; compensate class0 mass, preserve class1/3 scores',class2_update_coefficient=.175,weights=[1.,1.2,1.2,.9],
    min_oof_delta=.001,min_positive_folds=2,max_fold_drop=.004,max_novel_fake_f1_drop=.01,
    max_dev_drop=.005,bootstrap_policy='report uncertainty, not a binary significance gate',
    baseline_user_reported_kaggle=.77436,
    baseline_csv_sha256='05bffeff984bd4927c8dd3e9bd3613aef6a7abce303e6e1dcb7cfa4a12012059')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def labels(p):return (p*np.array(CONFIG['weights'])).argmax(1)
def replacement(base,new_cat,old_cat):
    out=base.copy();delta=np.clip(.175*(new_cat[:,2]-old_cat[:,2]),-base[:,2],base[:,0])
    out[:,2]+=delta;out[:,0]-=delta
    return checked(out,len(base))


def main():
    guard();OUT.mkdir(exist_ok=True);start=time.time()
    previous=ROOT/'output_v11_experiment';pm=json.loads((previous/'manifest.json').read_text())
    assert digest(ROOT/'experiment_v11.py')==pm['source_sha256']
    assert {k:importlib.metadata.version(k) for k in pm['versions']}==pm['versions']
    for n,h in pm['baseline_manifest']['sha256'].items():assert digest(ROOT/n)==h,n
    full_dir=ROOT/'output_v11_experimental';fm=json.loads((full_dir/'manifest.json').read_text())
    full_mid=hashlib.sha256(json.dumps(fm,sort_keys=True).encode()).hexdigest()
    full=np.load(full_dir/'full_128.npz',allow_pickle=False);assert full['manifest_id'].item()==full_mid
    manifest=dict(config=CONFIG,sources={n:digest(ROOT/n) for n in ['experiment_v27.py','text_capacity_v26.py']},
        baseline_oof=digest(previous/'oof.npz'),baseline_manifest=digest(previous/'manifest.json'),
        baseline_full=digest(full_dir/'full_128.npz'),baseline_full_manifest=digest(full_dir/'manifest.json'),
        versions={k:importlib.metadata.version(k) for k in pm['versions']})
    manifest['components_sha256']=digest(ROOT/'colab_components_v5.npz')
    assert manifest['components_sha256']=='a97e1e591d06cd8634d9709db3303cd106d9118efece504fa2e2f69a266e08cb'
    source_manifest=json.loads((ROOT/'output_v26_experiment/manifest.json').read_text())
    for n,h in source_manifest['sources'].items():assert digest(ROOT/n)==h,n
    source_mid=hashlib.sha256(json.dumps(source_manifest,sort_keys=True).encode()).hexdigest()
    manifest['v26_manifest_sha256']=digest(ROOT/'output_v26_experiment/manifest.json')
    manifest['v26_predictions']={f'fold{f}':digest(ROOT/'output_v26_experiment'/f'fold{f}_text2000.npz') for f in range(3)}
    mid=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    mp=OUT/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest,'Manifest changed'
    mp.write_text(json.dumps(manifest,indent=2))
    report=dict(config=CONFIG,pretrained=False,initialization='all vectorizers/SVD fit from fitting reviews, no pretrained parameters',
        execution='hosted Colab',fit_data='train only',
        projection_method='Reuse BoW2000 fold models; update class2 by .175*(new_cat2-old_cat2), compensate class0 with clipping; class1/3 raw probabilities unchanged',
        caveat='Adaptive hypothesis after inspecting V26 per-class OOF errors; selection bias applies. Class2-only delta fixed at the prior .175 coefficient; no grid. Preserving class1/3 scores need not preserve argmax labels. Bootstrap is conditional; Kaggle improvement unknown.')
    def status(stage,**kv):
        report.update(stage=stage,**kv);(OUT/'report.json').write_text(json.dumps(report,indent=2));print(stage,kv,flush=True)
    data=load_data(ROOT/'data');train,dev,test=[prepare(data[k]) for k in ['train','dev','test']]
    z=np.load(previous/'oof.npz',allow_pickle=False);ref=np.load(ROOT/'colab_reference.npz',allow_pickle=False)
    components=np.load(ROOT/'colab_components_v5.npz',allow_pickle=False)
    y=train.spam_label.to_numpy();folds=z['folds']
    assert np.array_equal(components['y'],y) and np.array_equal(components['folds'],folds)
    for part in ['oof','dev','test']:
        recomposed=.5*softmax(3*components['svm_'+part],axis=1)+.5*components['cat_'+part]
        assert np.allclose(recomposed,ref[part],atol=1e-7,rtol=1e-6)
    assert np.array_equal(z['y'],y) and np.array_equal(ref['y'],y) and np.array_equal(ref['folds'],folds)
    base=checked(.7*ref['oof']+.3*z['candidate'],len(y));oof=np.zeros_like(base)
    split=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=102)
    completed_deltas=[]
    for fold,(tr,va) in enumerate(split.split(train,y,train.comment)):
        status('fold_start',fold=fold+1,partition='cross_validation')
        assert np.all(folds[va]==fold) and not set(train.comment.iloc[tr])&set(train.comment.iloc[va])
        source_ck=ROOT/'output_v26_experiment'/f'fold{fold}_text2000.npz'
        q=np.load(source_ck,allow_pickle=False)
        assert q['manifest_id'].item()==source_mid and np.array_equal(q['tr'],tr) and np.array_equal(q['va'],va)
        p=checked(q['prediction'],len(va))
        oof[va]=p;a=labels(base[va]);b=labels(replacement(base[va],p,components['cat_oof'][va]))
        fold_delta=macro(y[va],b)-macro(y[va],a)
        status('fold_complete',fold=fold+1,fold_delta=fold_delta,text2000_alone_macro=macro(y[va],p.argmax(1)))
        completed_deltas.append(fold_delta)
    a=labels(base);b=labels(replacement(base,oof,components['cat_oof']))
    fd=[macro(y[folds==f],b[folds==f])-macro(y[folds==f],a[folds==f]) for f in range(3)]
    status('cv_diagnostics');groups=near_dup_groups(train.comment.tolist(),thr=.7);template=np.zeros(len(y),bool)
    for fold in range(3):
        va=np.flatnonzero(folds==fold);tr=np.flatnonzero((folds!=fold)&(y==1))
        template[va]=max_jaccard(train.comment.iloc[va].tolist(),train.comment.iloc[tr].tolist())>=.6
    delta=macro(y,b)-macro(y,a)
    nd=class1_strata(y,b,template)['novel']['f1']-class1_strata(y,a,template)['novel']['f1']
    boot=paired_cluster_bootstrap(y,a,b,groups,n=2000)
    gates=dict(macro=delta>=.001,folds=sum(d>0 for d in fd)>=2,worst_fold=min(fd)>=-.004,novel=nd>=-.01)
    status('cv_complete',baseline_oof=metrics(y,a),candidate_oof=metrics(y,b),text2000_alone=metrics(y,oof.argmax(1)),
        delta=delta,fold_deltas=fd,novel_fake_f1_delta=nd,bootstrap=boot,gates=gates,
        bootstrap_q05_positive=boot['q05']>0)
    np.savez_compressed(OUT/'oof.npz',text2000=oof,y=y,folds=folds)
    if not all(gates.values()):status('rejected_on_cv',elapsed_seconds=time.time()-start);return
    status('full_train',partition='full_train')
    ck=OUT/'full_text2000.npz'
    if ck.exists():
        q=np.load(ck,allow_pickle=False);assert q['manifest_id'].item()==mid
        ps=[checked(q['dev'],len(dev)),checked(q['test'],len(test))]
    else:
        feature_id=hashlib.sha256(json.dumps(dict(data=pm['baseline_manifest']['sha256'],bow_tokens=2000,seed=102,partition='full',frames=['dev','test']),sort_keys=True).encode()).hexdigest()
        ps=fit_predict(train,[dev.drop(columns='spam_label'),test],OUT/'full_artifacts',feature_id,status)
        tmp=ck.with_suffix('.tmp.npz');np.savez_compressed(tmp,manifest_id=mid,dev=ps[0],test=ps[1]);tmp.replace(ck)
    bd=checked(.7*ref['dev']+.3*full['dev'],len(dev));bt=checked(.7*ref['test']+.3*full['test'],len(test))
    old=labels(bd);dp=labels(replacement(bd,ps[0],components['cat_dev']));baseline_test=labels(bt)
    sub=data['samplesubmission'].copy();mapping=dict(zip(test.id.astype(str),baseline_test))
    sub['spam_label']=sub.ID.astype(str).map(mapping).astype(int)
    assert hashlib.sha256(sub.to_csv(index=False).encode()).hexdigest()==CONFIG['baseline_csv_sha256']
    report['baseline_matches_best_submission']=True
    report['dev']=dict(baseline=metrics(dev.spam_label,old),candidate=metrics(dev.spam_label,dp))
    report['dev_delta']=report['dev']['candidate']['macro_f1']-report['dev']['baseline']['macro_f1']
    if report['dev_delta']<-.005:status('rejected_on_dev',elapsed_seconds=time.time()-start);return
    tp=labels(replacement(bt,ps[1],components['cat_test']));sub=data['samplesubmission'].copy();ids=sub.ID.copy()
    sub['spam_label']=sub.ID.astype(str).map(dict(zip(test.id.astype(str),tp))).astype(int)
    assert len(sub)==len(test) and sub.ID.is_unique and sub.ID.equals(ids) and sub.spam_label.isin(range(4)).all()
    path=OUT/'submission_v27.csv';sub.to_csv(path,index=False)
    pd.testing.assert_frame_equal(pd.read_csv(path),sub,check_dtype=False)
    status('candidate_ready_not_submitted',rows=len(sub),sha256=digest(path),test_changes_vs_best=int((tp!=baseline_test).sum()),elapsed_seconds=time.time()-start)


if __name__=='__main__':main()
