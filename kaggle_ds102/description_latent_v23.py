"""Review/title plus description relations in a train-only TF-IDF/SVD space."""
import gc,json,re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize
from latent_v6 import LatentFeatures,make_model
from improved_features import dense_features
from experiment_v10 import guard,checked

CONFIG=dict(dimensions=128,description_dimensions=32,description_char_cap=4000,
    trees=800,depth=5,lr=.06,l2=8,seed=102,classifier_weight='SqrtBalanced',
    vocabulary_fit='training reviews only; descriptions are transform-only')


class DescriptionFeatures(LatentFeatures):
    def __init__(self,dimensions=128,description_dimensions=32):
        super().__init__();self.svd.set_params(n_components=dimensions)
        self.description_dimensions=min(dimensions,description_dimensions)

    def description_features(self,df,review,z):
        unique=[];lookup={};inverse=[]
        for value in df.description:
            text=' '.join(re.sub(r'<[^>]*>',' ',value).split())[:4000]
            if text not in lookup:lookup[text]=len(unique);unique.append(text)
            inverse.append(lookup[text])
        inverse=np.asarray(inverse,dtype=np.int64)
        # Deduplication only avoids repeated transformation; no vocabulary/IDF fitting.
        ux=normalize(sparse.hstack([self.base.word.transform(unique),self.base.char.transform(unique)],format='csr'))
        ud=self.svd.transform(ux);d=ud[inverse]
        cosine=np.empty(len(df),dtype=np.float32)
        for start in range(0,len(df),256):
            end=min(start+256,len(df))
            cosine[start:end]=np.asarray(review[start:end].multiply(ux[inverse[start:end]]).sum(1)).ravel()
        dn=np.linalg.norm(d,axis=1);zn=np.linalg.norm(z,axis=1)
        latent_cos=(z*d).sum(1)/np.maximum(dn*zn,1e-9)
        distance=np.linalg.norm(z-d,axis=1);k=self.description_dimensions
        return np.hstack([d[:,:k],z[:,:k]*d[:,:k],np.abs(z[:,:k]-d[:,:k]),
            cosine[:,None],latent_cos[:,None],dn[:,None],distance[:,None]])

    def fit_transform(self,train):return self.transform(train,fit=True)

    def transform(self,df,fit=False):
        if fit:
            guard()
            review=normalize(sparse.hstack([self.base.word.fit_transform(df.comment),self.base.char.fit_transform(df.comment)],format='csr'))
            z=self.svd.fit_transform(review)
        else:
            review=normalize(sparse.hstack([self.base.word.transform(df.comment),self.base.char.transform(df.comment)],format='csr'))
            z=self.svd.transform(review)
        title=normalize(sparse.hstack([self.base.word.transform(df.product_name),self.base.char.transform(df.product_name)],format='csr'))
        t=self.svd.transform(title);cosine=np.asarray(review.multiply(title).sum(1)).ravel()
        extra=self.description_features(df,review,z)
        a=np.hstack([z,t,z*t,np.abs(z-t),cosine[:,None],dense_features(df),extra]).astype(np.float32)
        assert np.isfinite(a).all()
        x=pd.DataFrame(a,columns=[f'f_{i}' for i in range(a.shape[1])])
        x['category']=df.category.astype(str).to_numpy();x['rating']=df.rating.astype(str).to_numpy()
        return x


def fit_predict(train,frames,folder,feature_id,status):
    guard();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    ck=folder/'description_features.npz'
    if ck.exists():
        z=np.load(ck,allow_pickle=False);assert z['feature_id'].item()==feature_id
        cols=z['numeric_columns'].tolist();xt=pd.DataFrame(z['train_numeric'],columns=cols);external=[]
        for j,col in enumerate(['category','rating']):xt[col]=z['train_categories'][:,j]
        for i in range(len(frames)):
            x=pd.DataFrame(z[f'frame{i}_numeric'],columns=cols)
            for j,col in enumerate(['category','rating']):x[col]=z[f'frame{i}_categories'][:,j]
            external.append(x)
    else:
        status('fit_description_svd_features')
        f=DescriptionFeatures();xt=f.fit_transform(train);external=[f.transform(d) for d in frames]
        cols=[c for c in xt if c not in ['category','rating']]
        arrays=dict(feature_id=feature_id,numeric_columns=np.array(cols),train_numeric=xt[cols].to_numpy(dtype=np.float32),train_categories=xt[['category','rating']].to_numpy(dtype=str))
        for i,x in enumerate(external):
            arrays[f'frame{i}_numeric']=x[cols].to_numpy(dtype=np.float32)
            arrays[f'frame{i}_categories']=x[['category','rating']].to_numpy(dtype=str)
        tmp=ck.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(ck)
        (folder/'feature_metadata.json').write_text(json.dumps(CONFIG,indent=2))
        del f,arrays;gc.collect()
    assert len(xt)==len(train) and [len(x) for x in external]==[len(d) for d in frames]
    status('fit_catboost_description_context',feature_columns=xt.shape[1])
    m=make_model();m.fit(xt,train.spam_label.to_numpy(),verbose=200)
    assert np.array_equal(m.classes_,[0,1,2,3])
    result=[checked(m.predict_proba(x),len(x)) for x in external]
    m.save_model(str(folder/'catboost.cbm'))
    del xt,external,m;gc.collect();return result
