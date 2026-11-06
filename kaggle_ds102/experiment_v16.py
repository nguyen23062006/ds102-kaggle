"""Binary fake-review copy expert; row-local alignment features, no pretrained."""
import hashlib,importlib.metadata,json,re,sys,time,unicodedata
from difflib import SequenceMatcher
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'claude_review'))
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from experiment_v10 import guard,checked
from improved_features import prepare,tokens,dense_features
from train_submit import load_data,metrics
from review_common import near_dup_groups,max_jaccard,class1_strata,macro
from codex_checked_common import paired_cluster_bootstrap

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'output_v16_experiment'
CONFIG=dict(expert='fake versus all other classes',expert_weight=.15,trees=700,
    depth=5,lr=.06,l2=8,seed=102,review_char_cap=6000,context_char_cap=18000,
    weights=[1.,1.2,1.2,.9],min_oof_delta=.001,min_positive_folds=2,
    max_fold_drop=.004,min_novel_fake_f1_delta=0.,max_dev_drop=.005,
    baseline_user_reported_kaggle=.77436,
    baseline_csv_sha256='05bffeff984bd4927c8dd3e9bd3613aef6a7abce303e6e1dcb7cfa4a12012059')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(s):return ' '.join(tokens(re.sub(r'<[^>]*>',' ',s)))
def ascii_text(s):return ''.join(c for c in unicodedata.normalize('NFKD',s) if not unicodedata.combining(c)).replace('đ','d')
def labels(p):return (p*np.array(CONFIG['weights'])).argmax(1)


def copy_features(frame):
    # No fitting, corpus statistics or labels: each row uses its own input only.
    numeric=dense_features(frame);extra=[]
    for row in frame.itertuples(index=False):
        r=clean(row.comment)[:6000];rw=r.split();ra=ascii_text(r);vals=[]
        for context in [row.product_name,row.description]:
            c=clean(context)[:18000];ca=ascii_text(c)
            if not r or not c:
                vals.extend([np.log1p(len(c)),0.,0.,0.,0.,0.,0.,0.,0.,0.]);continue
            score=fuzz.partial_ratio(r,c)/100.
            longest=SequenceMatcher(None,rw,c.split(),autojunk=True).find_longest_match().size
            vals.extend([np.log1p(len(c)),fuzz.ratio(r,c)/100.,score,
                fuzz.partial_ratio(ra,ca)/100.,longest/max(len(rw),1),np.log1p(longest),
                float(r in c),float(ra in ca),len(r)/max(len(c),1),score*np.log1p(len(r))])
        extra.append(vals)
    a=np.hstack([numeric,np.asarray(extra,dtype=np.float32)])
    assert np.isfinite(a).all()
    x=pd.DataFrame(a.astype(np.float32),columns=[f'feature_{i}' for i in range(a.shape[1])])
    x['category']=frame.category.astype(str).to_numpy();x['rating']=frame.rating.astype(str).to_numpy()
    return x


def model():
    guard();return CatBoostClassifier(iterations=700,depth=5,learning_rate=.06,l2_leaf_reg=8,
        loss_function='Logloss',auto_class_weights='SqrtBalanced',cat_features=['category','rating'],
        max_ctr_complexity=1,random_seed=102,thread_count=2,allow_writing_files=False,verbose=False)


def combine(base,fake):
    fake=np.asarray(fake);assert fake.shape==(len(base),) and np.isfinite(fake).all() and ((fake>=0)&(fake<=1)).all()
    other=[0,2,3];den=base[:,other].sum(1,keepdims=True)
    shares=np.divide(base[:,other],den,out=np.full_like(base[:,other],1/3),where=den>1e-12)
    q=base.copy();q[:,1]=fake;q[:,other]=shares*(1-fake[:,None])
    return checked(.85*base+.15*q,len(base))


def main():
    guard();OUT.mkdir(exist_ok=True);start=time.time()
    previous=ROOT/'output_v11_experiment';pm=json.loads((previous/'manifest.json').read_text())
    assert digest(ROOT/'experiment_v11.py')==pm['source_sha256']
    assert {k:importlib.metadata.version(k) for k in pm['versions']}==pm['versions']
    for n,h in pm['baseline_manifest']['sha256'].items():assert digest(ROOT/n)==h,n
    full_dir=ROOT/'output_v11_experimental';fm=json.loads((full_dir/'manifest.json').read_text())
    full_mid=hashlib.sha256(json.dumps(fm,sort_keys=True).encode()).hexdigest()
    full=np.load(full_dir/'full_128.npz',allow_pickle=False);assert full['manifest_id'].item()==full_mid
    manifest=dict(config=CONFIG,source_sha256=digest(__file__),baseline_oof=digest(previous/'oof.npz'),
        baseline_manifest=digest(previous/'manifest.json'),baseline_full=digest(full_dir/'full_128.npz'),
        baseline_full_manifest=digest(full_dir/'manifest.json'),
        versions={k:importlib.metadata.version(k) for k in [*pm['versions'],'rapidfuzz']})
    mid=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    mp=OUT/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    mp.write_text(json.dumps(manifest,indent=2))
    report=dict(config=CONFIG,pretrained=False,execution='hosted Colab CPU',fit_data='train only',
        caveat='Adaptive hypothesis from past OOF errors; repeated OOF/dev use creates selection bias. Alignment scores are features, not rules assigning labels. Kaggle improvement unknown.')
    def status(stage,**kv):
        report.update(stage=stage,**kv);(OUT/'report.json').write_text(json.dumps(report,indent=2));print(stage,kv,flush=True)
    data=load_data(ROOT/'data');train,dev,test=[prepare(data[k]) for k in ['train','dev','test']]
    z=np.load(previous/'oof.npz',allow_pickle=False);ref=np.load(ROOT/'colab_reference.npz',allow_pickle=False)
    y=train.spam_label.to_numpy();folds=z['folds']
    assert np.array_equal(z['y'],y) and np.array_equal(ref['y'],y) and np.array_equal(ref['folds'],folds)
    base=checked(.7*ref['oof']+.3*z['candidate'],len(y));oof=np.zeros(len(y))
    status('row_local_features');x=copy_features(train)
    split=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=102)
    for fold,(tr,va) in enumerate(split.split(train,y,train.comment)):
        status('fold_fit',fold=fold+1)
        assert np.all(folds[va]==fold) and not set(train.comment.iloc[tr])&set(train.comment.iloc[va])
        ck=OUT/f'fold{fold}_copy.npz'
        if ck.exists():
            q=np.load(ck,allow_pickle=False)
            assert q['manifest_id'].item()==mid and np.array_equal(q['tr'],tr) and np.array_equal(q['va'],va)
            p=q['fake']
        else:
            m=model();m.fit(x.iloc[tr],(y[tr]==1).astype(int),verbose=False)
            assert np.array_equal(m.classes_,[0,1])
            p=m.predict_proba(x.iloc[va])[:,1]
            tmp=ck.with_suffix('.tmp.npz');np.savez_compressed(tmp,manifest_id=mid,tr=tr,va=va,fake=p);tmp.replace(ck)
        oof[va]=p;a=labels(base[va]);b=labels(combine(base[va],p))
        delta=macro(y[va],b)-macro(y[va],a);status('fold_complete',fold=fold+1,fold_delta=delta)
        if delta<-.004:
            status('pruned_on_fixed_worst_fold_gate',completed_folds=fold+1,elapsed_seconds=time.time()-start);return
    a=labels(base);b=labels(combine(base,oof))
    fd=[macro(y[folds==f],b[folds==f])-macro(y[folds==f],a[folds==f]) for f in range(3)]
    status('cv_diagnostics');groups=near_dup_groups(train.comment.tolist(),thr=.7);template=np.zeros(len(y),bool)
    for fold in range(3):
        va=np.flatnonzero(folds==fold);tr=np.flatnonzero((folds!=fold)&(y==1))
        template[va]=max_jaccard(train.comment.iloc[va].tolist(),train.comment.iloc[tr].tolist())>=.6
    delta=macro(y,b)-macro(y,a)
    nd=class1_strata(y,b,template)['novel']['f1']-class1_strata(y,a,template)['novel']['f1']
    boot=paired_cluster_bootstrap(y,a,b,groups,n=2000)
    gates=dict(macro=delta>=.001,folds=sum(d>0 for d in fd)>=2,worst_fold=min(fd)>=-.004,novel=nd>=0)
    status('cv_complete',baseline_oof=metrics(y,a),candidate_oof=metrics(y,b),delta=delta,
        fold_deltas=fd,novel_fake_f1_delta=nd,bootstrap=boot,gates=gates)
    np.savez_compressed(OUT/'oof.npz',fake=oof,y=y,folds=folds)
    if not all(gates.values()):status('rejected_on_cv',elapsed_seconds=time.time()-start);return
    status('full_fit');ck=OUT/'full_copy.npz'
    if ck.exists():
        q=np.load(ck,allow_pickle=False);assert q['manifest_id'].item()==mid;pdev,ptest=q['dev'],q['test']
    else:
        m=model();m.fit(x,(y==1).astype(int),verbose=False)
        pdev=m.predict_proba(copy_features(dev))[:,1];ptest=m.predict_proba(copy_features(test))[:,1]
        tmp=ck.with_suffix('.tmp.npz');np.savez_compressed(tmp,manifest_id=mid,dev=pdev,test=ptest);tmp.replace(ck)
        m.save_model(str(OUT/'copy_expert.cbm'))
    bd=checked(.7*ref['dev']+.3*full['dev'],len(dev));bt=checked(.7*ref['test']+.3*full['test'],len(test))
    old=labels(bd);dp=labels(combine(bd,pdev));baseline_test=labels(bt)
    sub=data['samplesubmission'].copy();sub['spam_label']=sub.ID.astype(str).map(dict(zip(test.id.astype(str),baseline_test))).astype(int)
    assert hashlib.sha256(sub.to_csv(index=False).encode()).hexdigest()==CONFIG['baseline_csv_sha256']
    report['baseline_matches_best_submission']=True
    report['dev']=dict(baseline=metrics(dev.spam_label,old),candidate=metrics(dev.spam_label,dp))
    report['dev_delta']=report['dev']['candidate']['macro_f1']-report['dev']['baseline']['macro_f1']
    if report['dev_delta']<-.005:status('rejected_on_dev',elapsed_seconds=time.time()-start);return
    tp=labels(combine(bt,ptest));sub=data['samplesubmission'].copy();ids=sub.ID.copy()
    sub['spam_label']=sub.ID.astype(str).map(dict(zip(test.id.astype(str),tp))).astype(int)
    assert len(sub)==len(test) and sub.ID.is_unique and sub.ID.equals(ids) and sub.spam_label.isin(range(4)).all()
    path=OUT/'submission_v16.csv';sub.to_csv(path,index=False)
    pd.testing.assert_frame_equal(pd.read_csv(path),sub,check_dtype=False)
    status('candidate_ready_not_submitted',rows=len(sub),sha256=digest(path),test_changes_vs_best=int((tp!=baseline_test).sum()),elapsed_seconds=time.time()-start)


if __name__=='__main__':main()
