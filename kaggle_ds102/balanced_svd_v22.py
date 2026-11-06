"""Train-label-balanced SVD basis with unweighted transforms for every row.

The row scale is used only when fitting the basis. Training/validation/test
coordinates all use the same unweighted X @ components.T mapping.
"""
import gc,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from improved_features import FeaturesV2,dense_features
from latent_v6 import make_model
from experiment_v10 import guard,checked

CONFIG=dict(dimensions=128,svd_iterations=5,row_scale_exponent=.25,seed=102,
    trees=800,depth=5,lr=.06,l2=8,classifier_weight='SqrtBalanced')


class BalancedFeatures:
    def __init__(self,dimensions=128):
        self.base=FeaturesV2();self.svd=TruncatedSVD(n_components=dimensions,n_iter=5,random_state=102)

    def review(self,df,fit=False):
        w=self.base.word.fit_transform(df.comment) if fit else self.base.word.transform(df.comment)
        c=self.base.char.fit_transform(df.comment) if fit else self.base.char.transform(df.comment)
        return normalize(sparse.hstack([w,c],format='csr'))

    def build(self,df,review):
        z=self.svd.transform(review)
        title=normalize(sparse.hstack([self.base.word.transform(df.product_name),self.base.char.transform(df.product_name)],format='csr'))
        t=self.svd.transform(title);cosine=np.asarray(review.multiply(title).sum(1)).ravel()
        a=np.hstack([z,t,z*t,np.abs(z-t),cosine[:,None],dense_features(df)]).astype(np.float32)
        assert np.isfinite(a).all()
        x=pd.DataFrame(a,columns=[f'f_{i}' for i in range(a.shape[1])])
        x['category']=df.category.astype(str).to_numpy();x['rating']=df.rating.astype(str).to_numpy()
        return x

    def fit_transform(self,train):
        guard();y=train.spam_label.to_numpy();counts=np.bincount(y,minlength=4);assert (counts>0).all()
        review=self.review(train,fit=True)
        scale=np.power(len(y)/(4*counts[y]),.25).astype(np.float32)
        weighted=review.multiply(scale[:,None]).tocsr()
        self.svd.fit(weighted)
        self.fit_class_counts=counts.tolist()
        # Never return weighted-row coordinates, which would require labels at inference.
        return self.build(train,review)

    def transform(self,frame):return self.build(frame,self.review(frame))


def fit_predict(train,frames,folder,feature_id,status):
    guard();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    ck=folder/'balanced_features.npz'
    if ck.exists():
        z=np.load(ck,allow_pickle=False);assert z['feature_id'].item()==feature_id
        cols=z['numeric_columns'].tolist()
        xt=pd.DataFrame(z['train_numeric'],columns=cols);external=[]
        for j,col in enumerate(['category','rating']):xt[col]=z['train_categories'][:,j]
        for i in range(len(frames)):
            x=pd.DataFrame(z[f'frame{i}_numeric'],columns=cols)
            for j,col in enumerate(['category','rating']):x[col]=z[f'frame{i}_categories'][:,j]
            external.append(x)
    else:
        status('fit_balanced_svd128')
        f=BalancedFeatures();xt=f.fit_transform(train)
        external=[f.transform(frame) for frame in frames]
        cols=[c for c in xt if c not in ['category','rating']]
        arrays=dict(feature_id=feature_id,numeric_columns=np.array(cols),
            train_numeric=xt[cols].to_numpy(dtype=np.float32),train_categories=xt[['category','rating']].to_numpy(dtype=str))
        for i,x in enumerate(external):
            arrays[f'frame{i}_numeric']=x[cols].to_numpy(dtype=np.float32)
            arrays[f'frame{i}_categories']=x[['category','rating']].to_numpy(dtype=str)
        tmp=ck.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(ck)
        (folder/'projection_metadata.json').write_text(json.dumps(dict(config=CONFIG,fit_class_counts=f.fit_class_counts,
            row_scale='(N/(4*n_class))**0.25; fit basis only',prediction_transform='unweighted input projected onto fitted basis'),indent=2))
        del f,arrays;gc.collect()
    assert len(xt)==len(train) and [len(x) for x in external]==[len(d) for d in frames]
    status('fit_catboost_balanced_projection')
    m=make_model();m.fit(xt,train.spam_label.to_numpy(),verbose=200)
    assert np.array_equal(m.classes_,[0,1,2,3])
    result=[checked(m.predict_proba(x),len(x)) for x in external]
    m.save_model(str(folder/'catboost.cbm'))
    del xt,external,m;gc.collect();return result
