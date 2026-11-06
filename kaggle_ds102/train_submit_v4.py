"""Hosted Colab v4: train all models from scratch; pretrained models forbidden."""
import gc,json,os,platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from train_submit import load_data,metrics,SEED
from train_submit_v2 import fit_predict,RECIPE
from improved_features import prepare,FeaturesV2
from catboost_v3 import make_cat,cat_features
from kernel_experiment_colab import kernel


def kernel_predict(train,frames):
    f=FeaturesV2();tx,mx=f.fit_transform(train);tx=normalize(tx)
    mx.data=np.clip(mx.data,-.75,.75)
    y=train.spam_label.to_numpy();counts=np.bincount(y,minlength=4)
    weights={c:float(np.sqrt(len(y)/(4*counts[c]))) for c in range(4)}
    gram=kernel(tx,mx,tx,mx)
    model=SVC(kernel='precomputed',C=1.,class_weight=weights,cache_size=512,random_state=SEED)
    model.fit(gram,y);del gram;gc.collect()
    predictions=[]
    for frame in frames:
        result=[]
        for start in range(0,len(frame),512):
            a,b=f.transform(frame.iloc[start:start+512]);a=normalize(a)
            b.data=np.clip(b.data,-.75,.75)
            result.extend(model.predict(kernel(a,b,tx,mx)).tolist())
        predictions.append(np.asarray(result,dtype=int))
    return predictions


def main_v4(data_dir='/content/ds102_data',output_dir='/content/ds102_v4'):
    if platform.system()!='Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Hosted Google Colab required; local training is disabled.')
    data=load_data(Path(data_dir));train,dev,test=[prepare(data[n]) for n in ['train','dev','test']]
    frames=[dev.drop(columns='spam_label'),test]
    print('Fit nonlinear SVM on Colab CPU...',flush=True)
    kd,kt=kernel_predict(train,frames);gc.collect()
    print('Fit SVM and NB-SVM on Colab CPU...',flush=True)
    bd,bt=fit_predict(train,frames,RECIPE);gc.collect()
    print('Fit CatBoost on Colab CPU...',flush=True)
    cat=make_cat('SqrtBalanced',iterations=500)
    cat.fit(cat_features(train),train.spam_label.to_numpy(),verbose=250)
    pd3=.5*softmax(3*bd,axis=1)+.5*cat.predict_proba(cat_features(dev))
    pt3=.5*softmax(3*bt,axis=1)+.5*cat.predict_proba(cat_features(test))
    d=(.9*pd3+.1*np.eye(4)[kd]).argmax(1)
    t=(.9*pt3+.1*np.eye(4)[kt]).argmax(1)
    sub=data['samplesubmission'].copy()
    sub['spam_label']=sub.ID.astype(str).map(dict(zip(test.id.astype(str),t.tolist())))
    assert not sub.isna().any().any() and sub.spam_label.isin([0,1,2,3]).all()
    sub['spam_label']=sub.spam_label.astype(int)
    out=Path(output_dir);out.mkdir(exist_ok=True)
    path=out/'submission_v4.csv';sub.to_csv(path,index=False)
    pd.testing.assert_frame_equal(pd.read_csv(path),sub,check_dtype=False)
    report={'dev':metrics(dev.spam_label.to_numpy(),d),'v3_dev':metrics(dev.spam_label.to_numpy(),pd3.argmax(1)),
            'rows_verified':len(sub),'kernel_weight':.1,'pretrained':False,'fit_data':'train only',
            'execution':'hosted Colab CPU','cv_reference':{'v3':.7651232845255038,'v4':.7686788914302343},
            'caveat':'CV reference was measured on the supplied competition data; not rerun by this fixed-config script. Dev reused in development; Kaggle score unknown.'}
    (out/'metrics_v4.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('Saved:',path,'; dev F1-macro:',report['dev']['macro_f1'],flush=True)
    return report


if __name__=='__main__':main_v4()
