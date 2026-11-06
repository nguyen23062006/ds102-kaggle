"""E1/E2/E3 with training-only dedup weights, probability averaging and manifests.

Original Claude files and v1-v6 remain unchanged. All fits require hosted Colab.
"""
import argparse,gc,hashlib,json,os,platform,time,traceback,sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1])]
import numpy as np
import pandas as pd
import sklearn,catboost
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC
from train_submit import load_data,make_model,SEED
from improved_features import prepare,FeaturesV2,nb_ratios
from train_submit_v2 import RECIPE
from cluster_cv import cat_component,latent_component
from codex_checked_common import (W_V5,W_V6,near_dup_groups,max_jaccard,class_f1,
    macro,class1_strata,paired_cluster_bootstrap,group_summary,dedup_train_weights,
    recipe_probabilities,labels)


def hosted_guard():
    if platform.system()!='Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Hosted Google Colab required; local training disabled.')


def check_converged(model):
    if np.max(model.n_iter_)>=model.max_iter:
        raise RuntimeError('SVM did not converge; reject this checkpoint before prediction.')


def svm_component(train,frames,sw):
    hosted_guard()
    y=train.spam_label.to_numpy();f=FeaturesV2();xt,mt=f.fit_transform(train)
    transformed=[f.transform(d) for d in frames]
    result=[np.zeros((len(d),4)) for d in frames]
    for cfg in RECIPE:
        if cfg['kind']=='svm':
            model=make_model(cfg,y).fit(sparse.hstack([xt,mt],format='csr'),y,sample_weight=sw)
            check_converged(model)
            for j,(xv,mv) in enumerate(transformed):
                result[j]+=cfg['ensemble_weight']*model.decision_function(sparse.hstack([xv,mv],format='csr'))
        elif cfg['kind']=='nb':
            for c,r in enumerate(nb_ratios(xt,y)):
                model=LinearSVC(C=cfg['C'],class_weight='balanced',dual=True,
                    max_iter=30000,random_state=SEED)
                model.fit(sparse.hstack([xt.multiply(r),mt],format='csr'),(y==c).astype(int),sample_weight=sw)
                check_converged(model)
                for j,(xv,mv) in enumerate(transformed):
                    result[j][:,c]+=cfg['ensemble_weight']*model.decision_function(sparse.hstack([xv.multiply(r),mv],format='csr'))
        else:raise ValueError('Unknown SVM recipe')
    return result


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_matrix(values,n,probability):
    values=np.asarray(values)
    if values.shape!=(n,4) or not np.isfinite(values).all():raise ValueError('Bad prediction matrix')
    if probability and (np.any(values<0) or not np.allclose(values.sum(1),1,atol=1e-5)):
        raise ValueError('Invalid probability rows')
    return values


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='/content/ds102_data')
    p.add_argument('--out',default='/content/claude_review_out/checked')
    p.add_argument('--seed',type=int,default=102,help='Split/bootstrap seed; model seed remains 102')
    p.add_argument('--folds',type=int,default=3)
    p.add_argument('--thr',type=float,default=.6)
    p.add_argument('--dedup-weight',action='store_true')
    p.add_argument('--bootstrap',type=int,default=2000)
    a=p.parse_args(argv);hosted_guard()
    if a.folds<2 or not 0<a.thr<=1 or a.bootstrap<200:p.error('Invalid folds, threshold or bootstrap count')
    source=Path(__file__).resolve().parents[1]
    source_names=['train_submit.py','train_submit_v2.py','improved_features.py','catboost_v3.py',
        'latent_v6.py','experiment_v6.py','selection_v5.py','select_v2.py',
        'claude_review/cluster_cv.py','claude_review/review_common.py',
        'claude_review/codex_checked_common.py','claude_review/cluster_cv_checked.py']
    manifest={'protocol':'cluster_cv_checked_v1','split_seed':a.seed,'model_seed':SEED,
        'folds':a.folds,'group_threshold':a.thr,'group_min_tokens':4,'stratum_threshold':.6,
        'dedup_weight':a.dedup_weight,'dedup_normalization':'within each fitting partition and class',
        'ensemble':'softmax per fold, then average unweighted recipe probabilities, then class weights',
        'bootstrap_resamples':a.bootstrap,'svm_recipe':RECIPE,'W_V5':W_V5.tolist(),'W_V6':W_V6.tolist(),
        'versions':{'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'catboost':catboost.__version__},
        'data_sha256':{n:digest(Path(a.data_dir)/f'ds102_{n}.csv') for n in ['train','dev','test','samplesubmission']},
        'source_sha256':{n:digest(source/n) for n in source_names}}
    manifest_id=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    tag=f"{'e2' if a.dedup_weight else 'e1'}_seed{a.seed}_k{a.folds}_thr{a.thr:g}_{manifest_id[:12]}"
    out=Path(a.out)/tag;out.mkdir(parents=True,exist_ok=True)
    mpath=out/'manifest.json'
    if mpath.exists() and json.loads(mpath.read_text())!=manifest:raise RuntimeError('Manifest mismatch; do not resume')
    mpath.write_text(json.dumps(manifest,indent=2))
    def status(stage,**kv):
        (out/'status.json').write_text(json.dumps(dict(stage=stage,time=time.time(),manifest_id=manifest_id,**kv),indent=2))
    try:
        data=load_data(Path(a.data_dir))
        train,dev,test=[prepare(data[k]) for k in ['train','dev','test']]
        y=train.spam_label.to_numpy();yd=dev.spam_label.to_numpy()
        status('groups');groups=near_dup_groups(train.comment.tolist(),thr=a.thr)
        gi=group_summary(groups,y)
        if gi['largest_fraction']>.02:raise RuntimeError('Largest cluster exceeds 2%; inspect chaining and predefine stricter threshold before fitting.')
        split=list(StratifiedGroupKFold(n_splits=a.folds,shuffle=True,random_state=a.seed).split(train,y,groups=groups))
        for tr,va in split:
            assert set(y[tr])==set(y[va])=={0,1,2,3},'A train/validation fold lacks a class'
            assert not set(groups[tr])&set(groups[va])
        folds=np.full(len(y),-1);oof={k:np.zeros((len(y),4)) for k in ['svm','cat','lat']}
        ext=[dev.drop(columns='spam_label'),test]
        ext_components={k:[np.zeros((len(d),4)) for d in ext] for k in oof}
        ext_prob={k:[np.zeros((len(d),4)) for d in ext] for k in ['v5','v6']}
        times=[];margin_dev_per_fold=[]
        for fold,(tr,va) in enumerate(split):
            folds[va]=fold;fit=train.iloc[tr];frames=[train.iloc[va].drop(columns='spam_label')]+ext
            # Deliberately calculated AFTER splitting, using training labels only.
            sw=dedup_train_weights(y[tr],groups[tr]) if a.dedup_weight else None
            fold_outputs={}
            for k,fn in [('svm',svm_component),('cat',cat_component),('lat',latent_component)]:
                ck=out/f'fold{fold}_{k}.npz';start=time.time()
                if ck.exists():
                    z=np.load(ck,allow_pickle=False)
                    assert z['manifest_id'].item()==manifest_id and np.array_equal(z['tr'],tr) and np.array_equal(z['va'],va)
                    predictions=[z['validation'],z['dev'],z['test']]
                    seconds=float(z['seconds']);resumed=True
                else:
                    status('fit',fold=fold+1,component=k,group_info=gi);hosted_guard()
                    predictions=fn(fit,frames,sw);gc.collect()
                    predictions=[checked_matrix(v,len(d),k!='svm') for v,d in zip(predictions,frames)]
                    seconds=time.time()-start;resumed=False
                    tmp=ck.with_suffix('.tmp.npz')
                    np.savez_compressed(tmp,manifest_id=manifest_id,tr=tr,va=va,
                        validation=predictions[0],dev=predictions[1],test=predictions[2],seconds=seconds)
                    tmp.replace(ck)
                predictions=[checked_matrix(v,len(d),k!='svm') for v,d in zip(predictions,frames)]
                fold_outputs[k]=predictions;oof[k][va]=predictions[0]
                for j in range(2):ext_components[k][j]+=predictions[j+1]/a.folds
                times.append({'fold':fold+1,'component':k,'seconds':seconds,'resumed':resumed})
                status('component_complete',fold=fold+1,component=k,seconds=seconds)
            for j in range(2):
                pr=recipe_probabilities(*(fold_outputs[k][j+1] for k in ['svm','cat','lat']))
                for name in ext_prob:ext_prob[name][j]+=pr[name]/a.folds
            margin_dev_per_fold.append(float(fold_outputs['svm'][1].std()))
        text=train.comment.tolist();tmpl=np.zeros(len(y),bool)
        for fold in range(a.folds):
            va=np.flatnonzero(folds==fold);r1=np.flatnonzero((folds!=fold)&(y==1))
            tmpl[va]=max_jaccard([text[i] for i in va],[text[i] for i in r1])>=.6
        td=max_jaccard(dev.comment.tolist(),train.loc[train.spam_label==1,'comment'].tolist())>=.6
        po=recipe_probabilities(oof['svm'],oof['cat'],oof['lat'])
        predictions={name:labels(prob,name) for name,prob in po.items()}
        legacy=recipe_probabilities(*(ext_components[k][0] for k in ['svm','cat','lat']))
        rep={'tag':tag,'manifest_id':manifest_id,'group_info':gi,'model_seed':SEED,'split_seed':a.seed,
            'group_threshold':a.thr,'stratum_threshold':.6,'dedup_weight':a.dedup_weight,
            'class_counts_per_fold':[np.bincount(y[folds==f],minlength=4).tolist() for f in range(a.folds)],
            'class1_template_share_oof':float(tmpl[y==1].mean()),'class1_template_share_dev':float(td[yd==1].mean()),
            'svm_margin_std':{'oof':float(oof['svm'].std()),'dev_per_fold':margin_dev_per_fold,
                              'dev_mean_margin':float(ext_components['svm'][0].std())},
            'component_timings':times,'pretrained':False,'fit_data':'train partitions only',
            'caveats':['New grouped OOF evaluates frozen recipes selected in earlier experiments; not a completely independent test.',
                'Bootstrap is paired and conditional on fitted predictions; does not include prior selection/training uncertainty.',
                'Dev has been repeatedly inspected; secondary signal only.',
                'Mean fold probabilities and softmax of mean margins differ; primary E3 uses mean probabilities.',
                'Identical training-size does not guarantee ensemble confidence or margin variance equals pooled OOF.']}
        for name in po:
            pred=predictions[name];dv=labels(ext_prob[name][0],name);te=labels(ext_prob[name][1],name)
            rep[name]={'oof_macro':macro(y,pred),'oof_fold_macro':[macro(y[folds==f],pred[folds==f]) for f in range(a.folds)],
                'oof_class_f1':class_f1(y,pred).tolist(),'oof_class1_strata':class1_strata(y,pred,tmpl),
                'dev_foldavg_macro':macro(yd,dv),'dev_foldavg_class_f1':class_f1(yd,dv).tolist(),
                'dev_class1_strata':class1_strata(yd,dv,td),
                'dev_labels_differ_from_softmax_mean_margin':int((dv!=labels(legacy[name],name)).sum()),
                'test_prediction_counts':np.bincount(te,minlength=4).tolist()}
            sub=data['samplesubmission'].copy()
            sub['spam_label']=sub.ID.astype(str).map(pd.Series(te,index=test.id.astype(str)))
            assert sub.spam_label.notna().all() and sub.spam_label.isin([0,1,2,3]).all()
            sub.spam_label=sub.spam_label.astype(int)
            path=out/f'submission_{name}_foldavg.csv';sub.to_csv(path,index=False)
            pd.testing.assert_frame_equal(pd.read_csv(path),sub,check_dtype=False)
            assert len(sub)==len(test) and sub.ID.equals(data['samplesubmission'].ID)
        status('bootstrap')
        rep['v6_minus_v5']={'fold_deltas':(np.array(rep['v6']['oof_fold_macro'])-np.array(rep['v5']['oof_fold_macro'])).tolist(),
            'cluster_bootstrap':paired_cluster_bootstrap(y,predictions['v5'],predictions['v6'],groups,n=a.bootstrap,seed=a.seed),
            'non_template_cluster_bootstrap':paired_cluster_bootstrap(y[~tmpl],predictions['v5'][~tmpl],predictions['v6'][~tmpl],groups[~tmpl],n=a.bootstrap,seed=a.seed),
            'dev_foldavg_delta':rep['v6']['dev_foldavg_macro']-rep['v5']['dev_foldavg_macro']}
        np.savez_compressed(out/'oof_and_foldavg.npz',folds=folds,groups=groups,tmpl=tmpl,tmpl_dev=td,y=y,
            **{f'oof_{k}':v for k,v in oof.items()},
            **{f'dev_{k}_prob':v[0] for k,v in ext_prob.items()},**{f'test_{k}_prob':v[1] for k,v in ext_prob.items()})
        (out/'report.json').write_text(json.dumps(rep,indent=2));status('complete');print(json.dumps(rep,indent=2));return rep
    except BaseException as exc:
        status('error',error=str(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
