"""Fit a configuration selected on train OOF; evaluate dev only afterwards."""
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from experiment_v5 import hosted_guard,linear_fit_predict,make_rich_cat,rich_cat_features
from improved_features import prepare
from train_submit import load_data,metrics


def baseline_predict(train,frames,return_components=False):
    from train_submit_v2 import fit_predict,RECIPE
    from catboost_v3 import make_cat,cat_features
    margins=fit_predict(train,frames,RECIPE)
    gc.collect()
    m=make_cat('SqrtBalanced',iterations=500);m.set_params(thread_count=2)
    m.fit(cat_features(train),train.spam_label.to_numpy(),verbose=250)
    cat=[m.predict_proba(cat_features(d)) for d in frames]
    if return_components:return margins,cat
    return [.5*softmax(3*s,axis=1)+.5*p for s,p in zip(margins,cat)]


def main(selection,family,data_dir='/content/ds102_data',
         output_dir='/content/ds102_v5',reference_path=None,components_path=None):
    hosted_guard()
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    def status(stage,**values):
        (out/'status.json').write_text(json.dumps(dict(stage=stage,**values),indent=2))
    try:
        data=load_data(Path(data_dir))
        train,dev,test=[prepare(data[k]) for k in ['train','dev','test']]
        frames=[dev.drop(columns='spam_label'),test]
        status('baseline_predictions')
        if family in ['temperature','stacking']:
            if components_path:
                z=np.load(components_path)
                assert np.array_equal(z['y'],train.spam_label.to_numpy())
                margins=[z['svm_dev'],z['svm_test']];cat_values=[z['cat_dev'],z['cat_test']]
                baseline_source='user-authorized cached SVM and CatBoost predictions'
                if family=='stacking': meta_svm,meta_cat=z['svm_oof'],z['cat_oof']
            else:
                if family=='stacking':
                    from stacking_v5 import make_base_oof
                    meta_svm,meta_cat=make_base_oof(train)
                margins,cat_values=baseline_predict(train,frames,return_components=True)
                baseline_source='freshly fitted from train on hosted Colab'
            base=[.5*softmax(3*s,axis=1)+.5*p for s,p in zip(margins,cat_values)]
        elif reference_path:
            z=np.load(reference_path)
            assert np.array_equal(z['y'],train.spam_label.to_numpy())
            assert z['dev'].shape==(len(dev),4) and z['test'].shape==(len(test),4)
            # This cache was checksum-verified against the previously permitted
            # v3 scores; the four input CSV hashes are recorded in the report.
            base=[z['dev'],z['test']]
            baseline_source='user-authorized cached v3 predictions'
        else:
            base=baseline_predict(train,frames)
            baseline_source='freshly fitted from train on hosted Colab'
        alpha=float(selection['new_model_weight'])
        status('new_model',family=family,config=selection['config'])
        if alpha==0:
            new=base
        elif family=='linear':
            new=linear_fit_predict(train,frames,selection['config'])
        elif family=='cat':
            m=make_rich_cat();m.fit(rich_cat_features(train),train.spam_label.to_numpy(),verbose=300)
            new=[m.predict_proba(rich_cat_features(d),ntree_end=selection['config']['trees']) for d in frames]
        elif family=='fasttext':
            from fasttext_scratch import fit_predict
            new=fit_predict(train,frames,selection['config'],out/'fasttext_fit')
        elif family=='temperature':
            cfg=selection['config'];mix=cfg['cat_weight'];temp=cfg['temperature']
            new=[(1-mix)*softmax(temp*s,axis=1)+mix*p for s,p in zip(margins,cat_values)]
        elif family=='stacking':
            from stacking_v5 import fit_meta,features
            meta=fit_meta(features(meta_svm,meta_cat),train.spam_label.to_numpy(),selection['config'])
            new=[meta.predict_proba(features(s,p)) for s,p in zip(margins,cat_values)]
        else:
            raise ValueError('Unknown model family')
        probs=[(1-alpha)*a+alpha*b for a,b in zip(base,new)]
        w=np.asarray(selection['weights'])
        dp,tp=[(p*w).argmax(1) for p in probs]
        sub=data['samplesubmission'].copy()
        lookup=pd.Series(tp,index=test.id.astype(str))
        sub['spam_label']=sub.ID.astype(str).map(lookup)
        if sub.spam_label.isna().any() or not sub.spam_label.isin([0,1,2,3]).all():
            raise ValueError('Invalid prediction IDs or labels')
        sub.spam_label=sub.spam_label.astype(int)
        path=out/'submission_v5.csv';sub.to_csv(path,index=False)
        saved=pd.read_csv(path)
        pd.testing.assert_frame_equal(saved,sub,check_dtype=False)
        assert len(saved)==len(test) and saved.ID.equals(data['samplesubmission'].ID)
        np.savez_compressed(out/'predictions.npz',dev=probs[0],test=probs[1],weights=w)
        report={'selection':selection,'new_family':family,
            'dev':metrics(dev.spam_label.to_numpy(),dp),
            'v3_dev':metrics(dev.spam_label.to_numpy(),base[0].argmax(1)),
            'fit_data':'train only; no dev or test labels for selection/fitting',
            'pretrained':False,'execution':'hosted Google Colab CPU',
            'libraries':{'scikit-learn':'1.7.2 (BSD-3-Clause)','CatBoost':'1.2.10 (Apache-2.0)'},
            'baseline_source':baseline_source,'rows_verified':len(saved),
            'prediction_counts':{int(k):int(v) for k,v in saved.spam_label.value_counts().items()},
            'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(data_dir).glob('ds102_*.csv')},
            'caveat':'OOF used repeatedly for model selection. Calibration cross-validation is on existing OOF, not a fully nested refit. Dev also reused in development. Kaggle score is unknown.'}
        if family=='fasttext' and alpha>0:
            report['libraries']['fastText']='official v0.9.2 C++ CLI (MIT); random initialization'
        (out/'metrics_v5.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        status('complete',dev_macro_f1=report['dev']['macro_f1'],rows=len(saved))
        return report
    except BaseException as exc:
        import traceback
        status('error',error=str(exc),traceback=traceback.format_exc())
        raise
