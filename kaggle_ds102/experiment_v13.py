"""Fixed context-tagged TF-IDF/SVM branch, mixed 15% into v6. Colab only."""
import gc,hashlib,importlib.metadata,json,sys,time,warnings
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'claude_review'))
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.exceptions import ConvergenceWarning
from experiment_v10 import guard,checked
from experiment_v11 import fit_latent
from train_submit import load_data,metrics
from improved_features import prepare,FeaturesV2,tokens
from review_common import near_dup_groups,max_jaccard,class1_strata,macro
from codex_checked_common import paired_cluster_bootstrap

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'output_v13_experiment'
CONFIG=dict(C=.5,context_scale=1.,temperature=3.,branch_weight=.15,
    context_max_features=120000,context_min_df=2,seed=102,
    weights=[1.,1.2,1.2,.9],min_oof_delta=.001,min_positive_folds=2,
    max_novel_fake_f1_drop=.01,min_bootstrap_q05=0.,max_dev_drop=.005)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def context_text(frame):
    result=[]
    for row in frame.itertuples(index=False):
        words=tokens(row.comment);parts=[]
        for channel,context in [('title',row.product_name),('description',row.description),
                                ('category',row.category+' '+row.categories)]:
            known=set(tokens(context))
            parts.extend(channel+('hit_' if w in known else 'new_')+w for w in words)
            parts.append('channel_separator')
        parts.extend('rating'+str(row.rating)+'_'+w for w in words)
        result.append(' '.join(parts) or 'empty_context')
    return result


def branch(train,frames,status):
    guard();status('context_features')
    f=FeaturesV2();x,m=f.fit_transform(train)
    ctx=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=120000,
        sublinear_tf=True,lowercase=False,dtype=np.float32,token_pattern=r'(?u)\b\w+\b')
    cx=ctx.fit_transform(context_text(train))
    xt=sparse.hstack([x,m,cx],format='csr')
    model=LinearSVC(C=.5,class_weight='balanced',dual=True,max_iter=30000,
                    tol=1e-4,random_state=102)
    status('context_svm_fit',features=xt.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(xt,train.spam_label.to_numpy())
    assert np.array_equal(model.classes_,[0,1,2,3]) and np.max(model.n_iter_)<model.max_iter
    result=[]
    for d in frames:
        x,m=f.transform(d);cx=ctx.transform(context_text(d))
        p=softmax(3*model.decision_function(sparse.hstack([x,m,cx],format='csr')),axis=1)
        result.append(checked(p,len(d)))
    del f,ctx,xt,model;gc.collect();return result


def labels(prob):return (prob*np.array(CONFIG['weights'])).argmax(1)


def main():
    guard();OUT.mkdir(exist_ok=True);start=time.time()
    previous=ROOT/'output_v11_experiment';pm=json.loads((previous/'manifest.json').read_text())
    assert digest(ROOT/'experiment_v11.py')==pm['source_sha256']
    assert {k:importlib.metadata.version(k) for k in pm['versions']}==pm['versions']
    for n,h in pm['baseline_manifest']['sha256'].items():assert digest(ROOT/n)==h,n
    manifest=dict(config=CONFIG,source_sha256=digest(__file__),v11_manifest=digest(previous/'manifest.json'),
        baseline_oof_sha256=digest(previous/'oof.npz'),versions=pm['versions'])
    mid=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    mp=OUT/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    mp.write_text(json.dumps(manifest,indent=2))
    report=dict(config=CONFIG,baseline_user_reported_kaggle=.77269,pretrained=False,
        execution='hosted Colab CPU',fit_data='train only',
        caveat='Context hypothesis proposed after error analysis of reused OOF. Adaptive selection bias remains; bootstrap is conditional. Baseline is current v6 refit; Kaggle improvement unknown.')
    def status(stage,**kv):
        report.update(stage=stage,**kv);(OUT/'report.json').write_text(json.dumps(report,indent=2));print(stage,kv,flush=True)
    data=load_data(ROOT/'data');train,dev,test=[prepare(data[k]) for k in ['train','dev','test']]
    z=np.load(previous/'oof.npz',allow_pickle=False);ref=np.load(ROOT/'colab_reference.npz',allow_pickle=False)
    y=train.spam_label.to_numpy();folds=z['folds']
    assert np.array_equal(z['y'],y) and np.array_equal(ref['y'],y) and np.array_equal(ref['folds'],folds)
    base=checked(.7*ref['oof']+.3*z['baseline'],len(y));oof=np.zeros_like(base)
    split=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=102)
    for fold,(tr,va) in enumerate(split.split(train,y,train.comment)):
        status('fold_start',fold=fold+1)
        assert np.all(folds[va]==fold) and not set(train.comment.iloc[tr])&set(train.comment.iloc[va])
        ck=OUT/f'fold{fold}_context.npz'
        if ck.exists():
            q=np.load(ck,allow_pickle=False)
            assert q['manifest_id'].item()==mid and np.array_equal(q['tr'],tr) and np.array_equal(q['va'],va)
            p=checked(q['prediction'],len(va))
        else:
            p=branch(train.iloc[tr],[train.iloc[va].drop(columns='spam_label')],status)[0]
            tmp=ck.with_suffix('.tmp.npz');np.savez_compressed(tmp,manifest_id=mid,tr=tr,va=va,prediction=p);tmp.replace(ck)
        oof[va]=p
        a=labels(base[va]);b=labels(.85*base[va]+.15*p)
        status('fold_complete',fold=fold+1,fold_delta=macro(y[va],b)-macro(y[va],a))
    a=labels(base);b=labels(.85*base+.15*oof)
    fd=[macro(y[folds==f],b[folds==f])-macro(y[folds==f],a[folds==f]) for f in range(3)]
    status('cv_diagnostics');groups=near_dup_groups(train.comment.tolist(),thr=.7);template=np.zeros(len(y),bool)
    for fold in range(3):
        va=np.flatnonzero(folds==fold);tr=np.flatnonzero((folds!=fold)&(y==1))
        template[va]=max_jaccard(train.comment.iloc[va].tolist(),train.comment.iloc[tr].tolist())>=.6
    delta=macro(y,b)-macro(y,a)
    nd=class1_strata(y,b,template)['novel']['f1']-class1_strata(y,a,template)['novel']['f1']
    boot=paired_cluster_bootstrap(y,a,b,groups,n=2000)
    gates=dict(macro=delta>=.001,folds=sum(d>0 for d in fd)>=2,novel=nd>=-.01,bootstrap=boot['q05']>0)
    status('cv_complete',baseline_oof=metrics(y,a),candidate_oof=metrics(y,b),delta=delta,
        fold_deltas=fd,novel_fake_f1_delta=nd,bootstrap=boot,gates=gates)
    np.savez_compressed(OUT/'oof.npz',context=oof,y=y,folds=folds)
    if not all(gates.values()):status('rejected_on_cv',elapsed_seconds=time.time()-start);return
    frames=[dev.drop(columns='spam_label'),test]
    ext={}
    for name,fn in [('context',lambda:branch(train,frames,status)),
                    ('latent64',lambda:fit_latent(train,frames,64,status))]:
        status('full_train',component=name);ck=OUT/f'full_{name}.npz'
        if ck.exists():
            q=np.load(ck,allow_pickle=False);assert q['manifest_id'].item()==mid
            p=[checked(q['dev'],len(dev)),checked(q['test'],len(test))]
        else:
            p=fn();tmp=ck.with_suffix('.tmp.npz');np.savez_compressed(tmp,manifest_id=mid,dev=p[0],test=p[1]);tmp.replace(ck)
        ext[name]=p
    bd=.7*ref['dev']+.3*ext['latent64'][0];bt=.7*ref['test']+.3*ext['latent64'][1]
    old=labels(bd);dp=labels(.85*bd+.15*ext['context'][0])
    report['dev']=dict(baseline=metrics(dev.spam_label,old),candidate=metrics(dev.spam_label,dp))
    if macro(dev.spam_label,dp)<macro(dev.spam_label,old)-.005:
        status('rejected_on_dev',elapsed_seconds=time.time()-start);return
    tp=labels(.85*bt+.15*ext['context'][1]);sub=data['samplesubmission'].copy()
    sub['spam_label']=sub.ID.astype(str).map(dict(zip(test.id.astype(str),tp))).astype(int)
    assert len(sub)==len(test) and sub.ID.is_unique and sub.spam_label.isin(range(4)).all()
    path=OUT/'submission_v13_candidate.csv';sub.to_csv(path,index=False)
    pd.testing.assert_frame_equal(pd.read_csv(path),sub,check_dtype=False)
    status('candidate_ready_not_submitted',rows=len(sub),sha256=digest(path),elapsed_seconds=time.time()-start)


if __name__=='__main__':main()
