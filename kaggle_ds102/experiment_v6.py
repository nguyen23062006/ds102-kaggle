"""Hosted Colab: template neighbours and probability-calibrated kernel SVM.

No pretrained models or outside data. All feature fitting is inside the same
grouped train folds as previous versions. Dev labels are not read for selection.
"""
import gc,json,time,traceback,os,platform
from pathlib import Path
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics.pairwise import rbf_kernel
from improved_features import prepare,FeaturesV2
from train_submit import load_data,metrics,SEED
from select_v2 import macro
from selection_v5 import cross_weight


def hosted_guard():
    if platform.system()!='Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Hosted Google Colab required; local training disabled.')


def kernel(text_a,meta_a,text_b,meta_b):
    k=(text_a@text_b.T).toarray()
    k+=1;np.square(k,out=k)
    k+=.5*rbf_kernel(meta_a,meta_b,gamma=.3)
    k/=4.5
    return np.ascontiguousarray(k,dtype=np.float64)


def neighbours(query,train,y,batch_size=256):
    """Eight nearest TF-IDF reviews; all references belong to fitting data."""
    counts=np.bincount(y,minlength=4)
    class_weight=np.sqrt(len(y)/(4*counts))
    probs=np.zeros((query.shape[0],4),dtype=np.float32)
    maximum=np.zeros(query.shape[0],dtype=np.float32)
    k=min(8,len(y))
    for start in range(0,query.shape[0],batch_size):
        sim=(query[start:start+batch_size]@train.T).toarray()
        idx=np.argpartition(sim,-k,axis=1)[:,-k:]
        values=np.take_along_axis(sim,idx,axis=1)
        weights=np.maximum(values-.2,0)**4
        labels=y[idx]
        votes=np.stack([(weights*(labels==c)).sum(axis=1)*class_weight[c] for c in range(4)],axis=1)
        votes+=1e-8
        probs[start:start+len(sim)]=votes/votes.sum(axis=1,keepdims=True)
        maximum[start:start+len(sim)]=values.max(axis=1)
    return probs,maximum


def make_kernel_model(y):
    counts=np.bincount(y,minlength=4)
    cw={c:float(np.sqrt(len(y)/(4*counts[c]))) for c in range(4)}
    # libsvm fits its sigmoid calibrators using only the fit partition.
    # Outer grouped CV keeps every held-out review group outside that process.
    return SVC(C=1.,kernel='precomputed',class_weight=cw,probability=True,
               random_state=SEED,cache_size=512)


def neighbour_class_balance(probabilities,counts,power=.5):
    """Change neighbour voting's class-count correction without refitting IDF."""
    factor=(np.asarray(counts,dtype=float)/max(counts))**(.5-power)
    p=probabilities*factor
    return p/p.sum(axis=1,keepdims=True)


def candidates(base,hard,prob,near,similarity):
    yield 'v3_reference',base,{'kind':'reference','alpha':0.}
    for alpha in [.05,.1,.2]:
        yield f'hard_kernel_{alpha}',(1-alpha)*base+alpha*np.eye(4)[hard],{'kind':'hard','alpha':alpha}
    for alpha in [.15,.3,.5]:
        yield f'probability_kernel_{alpha}',(1-alpha)*base+alpha*prob,{'kind':'probability','alpha':alpha}
    for threshold in [.7,.85]:
        for alpha in [.3,.5]:
            gate=(similarity>=threshold).astype(float)[:,None]*alpha
            yield f'neighbours_{threshold}_{alpha}',(1-gate)*base+gate*near,{'kind':'neighbours','alpha':alpha,'threshold':threshold}


def select(base,hard,prob,near,similarity,y,folds):
    rows=[]
    configurations=list(candidates(base,hard,prob,near,similarity))
    # The original inverse-square-root class correction is compared with
    # milder/no correction, using only fit-fold label counts.
    for power in [0.,.25]:
        adjusted=np.empty_like(near)
        for fold in np.unique(folds):
            mask=folds==fold;counts=np.bincount(y[~mask],minlength=4)
            adjusted[mask]=neighbour_class_balance(near[mask],counts,power)
        gate=.3*(similarity>=.85).astype(float)[:,None]
        configurations.append((f'neighbours_0.85_0.3_power_{power}',(1-gate)*base+gate*adjusted,
            {'kind':'neighbours','alpha':.3,'threshold':.85,'class_power':power}))
    for name,p,recipe in configurations:
        row=cross_weight(p,y,folds)
        raw_folds=[macro(y[folds==f],p[folds==f].argmax(1)) for f in np.unique(folds)]
        row['fold_macro_f1']=[v+d for v,d in zip(raw_folds,row['fold_deltas'])] if row['use_weights'] else raw_folds
        row.update(name=name,recipe=recipe)
        rows.append(row)
    rows.sort(key=lambda r:r['selection_score'],reverse=True)
    baseline=next(r for r in rows if r['recipe']['kind']=='reference')
    for row in rows:
        row['fold_delta_vs_v5']=(np.array(row['fold_macro_f1'])-np.array(baseline['fold_macro_f1'])).tolist()
    return rows


def choose_stable(rows):
    """Prefer consistent fold gains when aggregate CV differs by < .001."""
    baseline=next(r for r in rows if r['recipe']['kind']=='reference')
    best=rows[0]
    if best['selection_score']<baseline['selection_score']+.002:return baseline
    stable=[r for r in rows if r['recipe']['kind']!='reference'
            and r['selection_score']>=best['selection_score']-.001
            and min(r['fold_delta_vs_v5'])>=0]
    return max(stable,key=lambda r:r['selection_score']) if stable else best


def main(data_dir='/content/ds102_data',output_dir='/content/ds102_v6_experiment',
         reference_path='/content/colab_reference.npz'):
    hosted_guard()
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    def status(stage,**values):
        (out/'status.json').write_text(json.dumps(dict(stage=stage,time=time.time(),**values),indent=2))
    try:
        df=prepare(load_data(Path(data_dir))['train']);y=df.spam_label.to_numpy()
        ref=np.load(reference_path);assert np.array_equal(y,ref['y'])
        hard=np.full(len(y),-1,dtype=int);prob=np.zeros((len(y),4),dtype=np.float32)
        near=np.zeros_like(prob);similarity=np.zeros(len(y),dtype=np.float32)
        folds=np.full(len(y),-1);results=[]
        split=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED)
        for fold,(tr,va) in enumerate(split.split(df,y,groups=df.comment)):
            folds[va]=fold;status('features',fold=fold+1,results=results)
            f=FeaturesV2();xt,mt=f.fit_transform(df.iloc[tr])
            xv,mv=f.transform(df.iloc[va].drop(columns='spam_label'))
            xt=normalize(xt);xv=normalize(xv)
            mt.data=np.clip(mt.data,-.75,.75);mv.data=np.clip(mv.data,-.75,.75)
            status('neighbours',fold=fold+1,results=results)
            near[va],similarity[va]=neighbours(xv,xt,y[tr])
            status('kernel_matrix',fold=fold+1,results=results)
            kt=kernel(xt,mt,xt,mt);kv=kernel(xv,mv,xt,mt)
            status('kernel_probability_fit',fold=fold+1,results=results)
            m=make_kernel_model(y[tr]);m.fit(kt,y[tr])
            assert np.array_equal(m.classes_,np.arange(4))
            hard[va]=m.predict(kv);prob[va]=m.predict_proba(kv)
            results.append({'kernel_hard':macro(y[va],hard[va]),
                            'kernel_prob':macro(y[va],prob[va].argmax(1)),
                            'neighbours':macro(y[va],near[va].argmax(1))})
            del m,kt,kv,xt,xv,mt,mv,f;gc.collect()
            np.savez_compressed(out/'partial.npz',hard=hard,prob=prob,near=near,
                                similarity=similarity,y=y,folds=folds)
        assert np.array_equal(folds,ref['folds'])
        np.savez_compressed(out/'oof.npz',hard=hard,prob=prob,near=near,
                            similarity=similarity,y=y,folds=folds)
        rows=select(ref['oof'],hard,prob,near,similarity,y,folds)
        (out/'selection.json').write_text(json.dumps(rows,indent=2))
        report={'fold_scores':results,'best':rows[0],'chosen':choose_stable(rows),
            'selection_rule':'Require >=.002 CV gain; within .001 of best, prefer a candidate improving every fold over v5.',
            'reference_v5':next(r for r in rows if r['recipe']['kind']=='reference'),
            'pretrained':False,'fit_data':'train only; dev not read for selection',
            'execution':'hosted Colab CPU',
            'caveat':'Repeated OOF model selection; weight calibration on existing OOF is not fully nested CV.'}
        (out/'cv_results.json').write_text(json.dumps(report,indent=2));status('complete',**report)
        return report
    except BaseException as exc:
        status('error',error=str(exc),traceback=traceback.format_exc());raise
