"""Train-only DS102 V27 exploratory class2 text inference; hosted Colab required; no pretrained."""
import gc,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.preprocessing import normalize
from improved_features import prepare,FeaturesV2
from experiment_v6 import hosted_guard,kernel,neighbours,make_kernel_model,neighbour_class_balance
from train_submit import load_data,metrics


def baseline_predict(train,frames):
    from train_submit_v2 import fit_predict,RECIPE
    from catboost_v3 import make_cat,cat_features
    margins=fit_predict(train,frames,RECIPE);gc.collect()
    m=make_cat('SqrtBalanced',iterations=500);m.set_params(thread_count=2)
    m.fit(cat_features(train),train.spam_label.to_numpy(),verbose=250)
    old_cat=[m.predict_proba(cat_features(d)) for d in frames]
    del m;gc.collect()
    m=make_cat('SqrtBalanced',iterations=500)
    m.set_params(thread_count=2,feature_calcers=['BoW:top_tokens_count=2000','NaiveBayes','BM25'])
    m.fit(cat_features(train),train.spam_label.to_numpy(),verbose=100)
    return [(.5*softmax(3*s,axis=1)+.5*p,p,m.predict_proba(cat_features(d)))
            for s,p,d in zip(margins,old_cat,frames)]


def component_predict(train,frames,recipe,status=None):
    hosted_guard()
    if recipe['kind']=='latent':
        from latent_v6 import LatentFeatures,make_model as make_latent_model
        if status:status('fit_latent_features')
        f=LatentFeatures();f.svd.set_params(n_components=128,random_state=102);xt=f.transform(train,fit=True)
        if status:status('fit_latent_catboost')
        m=make_latent_model();m.fit(xt,train.spam_label.to_numpy(),verbose=False)
        return [(m.predict_proba(f.transform(df),ntree_end=recipe['trees']),None) for df in frames]
    f=FeaturesV2();xt,mt=f.fit_transform(train)
    xt=normalize(xt);mt.data=np.clip(mt.data,-.75,.75)
    y=train.spam_label.to_numpy();kind=recipe['kind']
    if kind=='neighbours':
        result=[]
        for df in frames:
            x,_=f.transform(df);p,s=neighbours(normalize(x),xt,y)
            if recipe.get('class_power',.5)!=.5:
                p=neighbour_class_balance(p,np.bincount(y,minlength=4),recipe['class_power'])
            result.append((p,s))
        return result
    if status:status('full_kernel')
    gram=kernel(xt,mt,xt,mt)
    m=make_kernel_model(y)
    if kind=='hard':m.set_params(probability=False)
    if status:status('full_kernel_fit')
    m.fit(gram,y);del gram;gc.collect()
    result=[]
    for frame_idx,df in enumerate(frames):
        if status:status('predict',frame=frame_idx)
        pieces=[]
        for start in range(0,len(df),256):
            xv,mv=f.transform(df.iloc[start:start+256]);xv=normalize(xv)
            mv.data=np.clip(mv.data,-.75,.75)
            gram=kernel(xv,mv,xt,mt)
            pieces.append(np.eye(4)[m.predict(gram)] if kind=='hard' else m.predict_proba(gram))
        result.append((np.vstack(pieces),None))
    return result


def main(selection,data_dir='/content/ds102_data',output_dir='/content/ds102_v6',reference_path=None):
    hosted_guard()
    assert reference_path is None,'This runner fits every component fresh; no reference cache.'
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    def status(stage,**values):
        (out/'status.json').write_text(json.dumps(dict(stage=stage,time=time.time(),**values),indent=2))
    try:
        data=load_data(Path(data_dir))
        train,dev,test=[prepare(data[k]) for k in ['train','dev','test']]
        frames=[dev.drop(columns='spam_label'),test]
        recipe=selection['recipe']
        if recipe['kind']=='reference':raise ValueError('v6 experiment did not choose a new component; retain v5.')
        status('component_features',recipe=recipe)
        extra=component_predict(train,frames,recipe,status)
        gc.collect();status('baseline_predictions')
        if reference_path:
            z=np.load(reference_path)
            assert np.array_equal(z['y'],train.spam_label.to_numpy())
            base=[z['dev'],z['test']]
            baseline_source='previously user-authorized cached v3 predictions'
        else:
            base=baseline_predict(train,frames)
            baseline_source='fresh train-only fit on hosted Colab'
        scores=[];alpha=float(recipe['alpha'])
        for (b,old_cat,new_cat),(p,s) in zip(base,extra):
            weight=alpha
            if recipe['kind']=='neighbours':weight=alpha*(s>=recipe['threshold']).astype(float)[:,None]
            combined=(1-weight)*b+weight*p
            delta=np.clip(.175*(new_cat[:,2]-old_cat[:,2]),-combined[:,2],combined[:,0])
            combined[:,2]+=delta;combined[:,0]-=delta
            assert (combined>=0).all() and np.allclose(combined.sum(1),1,atol=1e-5)
            scores.append(combined)
        weights=np.asarray(selection['weights'])
        dp,tp=[(p*weights).argmax(1) for p in scores]
        sub=data['samplesubmission'].copy()
        lookup=pd.Series(tp,index=test.id.astype(str))
        sub['spam_label']=sub.ID.astype(str).map(lookup)
        if sub.spam_label.isna().any() or not sub.spam_label.isin([0,1,2,3]).all():raise ValueError('Invalid IDs/labels')
        sub.spam_label=sub.spam_label.astype(int)
        path=out/'submission_v27_experimental.csv';sub.to_csv(path,index=False)
        saved=pd.read_csv(path);pd.testing.assert_frame_equal(saved,sub,check_dtype=False)
        assert len(saved)==len(test) and saved.ID.equals(data['samplesubmission'].ID)
        np.savez_compressed(out/'predictions.npz',dev=scores[0],test=scores[1],weights=weights)
        report={'selection':selection,'dev':metrics(dev.spam_label.to_numpy(),dp),
            'v5_reference':{'cross_weight_oof_macro_f1':.7681613908279921,'dev_macro_f1':.7348200162618499,'user_reported_kaggle':.76740},
            'rows_verified':len(saved),'pretrained':False,'fit_data':'train only; dev/test excluded from feature/model fitting',
            'selection_data':'grouped train OOF for candidate hyperparameters; dev used to compare finalists',
            'new_training_execution':'hosted Google Colab CPU','baseline_source':baseline_source,
            'libraries':{'scikit-learn':'1.7.2; BSD-3-Clause','CatBoost':'1.2.10; Apache-2.0'},
            'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(data_dir).glob('ds102_*.csv')},
            'caveat':'Repeated OOF selection; weighting cross-validation is on existing OOF, not fully nested. Dev reused in development; Kaggle score unknown.'}
        (out/'metrics_v27_experimental.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        status('complete',dev_macro_f1=report['dev']['macro_f1'],rows=len(saved))
        return report
    except BaseException as exc:
        status('error',error=str(exc),traceback=traceback.format_exc());raise
