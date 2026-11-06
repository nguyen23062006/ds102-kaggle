"""Training-label-aware column weighting before SVD; all parameters fit in train."""
import gc,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import chi2
from sklearn.preprocessing import normalize
from improved_features import FeaturesV2,dense_features
from latent_v6 import make_model
from cloud_checks import guard,checked

CONFIG=dict(dimensions=128,svd_iterations=5,column_weight='sqrt chi2 / positive median',
    weight_clip=[.5,4.],chi2_input='binary term presence',trees=800,depth=5,
    learning_rate=.06,l2=8,seed=102,new_latent_coefficient=.15)

class SupervisedLatentFeatures:
    def __init__(self):
        self.base=FeaturesV2();self.svd=TruncatedSVD(n_components=128,n_iter=5,random_state=102)
    def fit(self,train):
        guard();x=sparse.hstack([self.base.word.fit_transform(train.comment),
            self.base.char.fit_transform(train.comment)],format='csr')
        presence=x.copy();presence.data[:]=1.
        scores=chi2(presence,train.spam_label.to_numpy())[0]
        assert np.isfinite(scores).all() and (scores>=0).all()
        med=float(np.median(scores[scores>0]));assert med>0
        self.weights=np.clip(np.sqrt(scores/med),.5,4.).astype(np.float32)
        self.svd.fit(normalize(x.multiply(self.weights).tocsr()))
        self.weight_summary=dict(median_chi2=med,minimum=float(self.weights.min()),
            maximum=float(self.weights.max()),fraction_upper_clip=float((self.weights==4).mean()))
        return self
    def vectorize(self,text):
        x=sparse.hstack([self.base.word.transform(text),self.base.char.transform(text)],format='csr')
        return normalize(x.multiply(self.weights).tocsr())
    def transform(self,frame):
        r=self.vectorize(frame.comment);t=self.vectorize(frame.product_name)
        z=self.svd.transform(r);p=self.svd.transform(t)
        cosine=np.asarray(r.multiply(t).sum(axis=1)).ravel()
        a=np.hstack([z,p,z*p,np.abs(z-p),cosine[:,None],dense_features(frame)]).astype(np.float32)
        assert a.shape==(len(frame),575) and np.isfinite(a).all()
        x=pd.DataFrame(a,columns=[f'f_{i}' for i in range(575)])
        x['category']=frame.category.astype(str).to_numpy();x['rating']=frame.rating.astype(str).to_numpy()
        return x

def fit_predict(train,frames,folder,feature_id,status,cache_path=None,cache_id=None):
    guard();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    status('fit_train_only_supervised_column_svd128')
    features=SupervisedLatentFeatures().fit(train)
    xt=features.transform(train);external=[features.transform(df) for df in frames]
    status('fit_catboost_supervised_svd128',feature_columns=xt.shape[1],**features.weight_summary)
    m=make_model();m.fit(xt,train.spam_label.to_numpy(),verbose=200)
    assert np.array_equal(m.classes_,[0,1,2,3])
    ps=[checked(m.predict_proba(x),len(df)) for x,df in zip(external,frames)]
    m.save_model(str(folder/'catboost.cbm'));joblib.dump(features,folder/'features.joblib',compress=3)
    (folder/'metadata.json').write_text(json.dumps(dict(config=CONFIG,feature_id=feature_id,
        weights=features.weight_summary),indent=2))
    del m,features,xt,external;gc.collect();return ps
