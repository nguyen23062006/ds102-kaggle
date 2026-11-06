"""Train-fold catalogue vocabulary/SVD, independent of review-only SVD128.

Only supplied fitting product metadata form the corpus. No pretrained parameters,
external data, development/test fitting, or labels enter this representation.
"""
import gc,html,json,re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from latent_v6 import LatentFeatures,make_model
from experiment_v10 import guard,checked

CONFIG=dict(dimensions=32,description_char_cap=4096,word_max_features=50000,
    char_max_features=80000,word_ngram=[1,2],char_ngram=[3,5],max_df=.95,
    trees=800,depth=5,lr=.06,l2=8,seed=102)

def catalogue_documents(df):
    def clean(value):
        value=html.unescape(str(value))
        value=re.sub(r'<[^>]+>',' ',value)
        return re.sub(r'\s+',' ',value).strip()[:CONFIG['description_char_cap']]
    return [str(t)+' '+clean(d) for t,d in zip(df.product_name,df.description)]

class CatalogueEncoder:
    def __init__(self):
        self.word=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.95,
            max_features=50000,sublinear_tf=True,dtype=np.float32)
        self.char=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=3,
            max_df=.95,max_features=80000,sublinear_tf=True,dtype=np.float32)
    def vectorize(self,texts):
        return normalize(sparse.hstack([self.word.transform(texts),
            self.char.transform(texts)],format='csr'))
    def fit(self,train):
        guard();docs=list(dict.fromkeys(catalogue_documents(train)))
        assert len(docs)>3,'Need at least four distinct fitting catalogue documents'
        x=normalize(sparse.hstack([self.word.fit_transform(docs),
            self.char.fit_transform(docs)],format='csr'))
        self.k=min(CONFIG['dimensions'],x.shape[0]-1,x.shape[1]-1)
        assert self.k>0
        self.svd=TruncatedSVD(n_components=self.k,n_iter=5,random_state=102)
        self.svd.fit(x);self.fit_documents=len(docs);self.vocabulary_size=x.shape[1]
        return self
    def transform(self,df):
        r=self.vectorize(df.comment.tolist());p=self.vectorize(catalogue_documents(df))
        z=self.svd.transform(r);t=self.svd.transform(p)
        zn=np.linalg.norm(z,axis=1);tn=np.linalg.norm(t,axis=1)
        lexical=np.asarray(r.multiply(p).sum(axis=1)).ravel()
        cosine=(z*t).sum(1)/np.maximum(zn*tn,1e-9)
        z=np.pad(z,((0,0),(0,32-self.k)));t=np.pad(t,((0,0),(0,32-self.k)))
        out=np.hstack([z,t,z*t,np.abs(z-t),lexical[:,None],
            cosine[:,None],zn[:,None],tn[:,None]]).astype(np.float32)
        assert out.shape==(len(df),132) and np.isfinite(out).all()
        return out

def fit_predict(train,frames,folder,feature_id,status,cache_path=None,cache_id=None):
    guard();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if cache_path is not None:
        q=np.load(cache_path,allow_pickle=False);assert q['feature_id'].item()==cache_id
        cols=q['numeric_columns'][:575].tolist();assert cols==[f'f_{i}' for i in range(575)]
        all_frames=[]
        for prefix in ['train']+[f'frame{i}' for i in range(len(frames))]:
            x=pd.DataFrame(q[prefix+'_numeric'][:,:575],columns=cols)
            for j,c in enumerate(['category','rating']):x[c]=q[prefix+'_categories'][:,j]
            all_frames.append(x)
    else:
        status('fit_full_svd128_features')
        f=LatentFeatures();f.svd.set_params(n_components=128,random_state=102)
        all_frames=[f.transform(train,fit=True)]+[f.transform(d) for d in frames]
        del f;gc.collect()
    status('fit_catalogue_only_svd32')
    encoder=CatalogueEncoder().fit(train)
    joblib.dump(encoder,folder/'catalogue_encoder.joblib',compress=3)
    augmented=[]
    for x,df in zip(all_frames,[train]+frames):
        assert len(x)==len(df)
        assert np.array_equal(x[['category','rating']].to_numpy(dtype=str),
            df[['category','rating']].astype(str).to_numpy())
        extra=encoder.transform(df)
        joined=pd.concat([x.reset_index(drop=True),pd.DataFrame(extra,
            columns=[f'catalogue_{i}' for i in range(extra.shape[1])])],axis=1)
        assert joined.shape[1]==709
        augmented.append(joined)
    status('fit_catboost_catalogue_svd',feature_columns=709,
        fitting_catalogue_documents=encoder.fit_documents,
        catalogue_vocabulary=encoder.vocabulary_size)
    m=make_model();m.fit(augmented[0],train.spam_label.to_numpy(),verbose=200)
    assert np.array_equal(m.classes_,[0,1,2,3])
    result=[checked(m.predict_proba(x),len(x)) for x in augmented[1:]]
    m.save_model(str(folder/'catboost.cbm'))
    (folder/'metadata.json').write_text(json.dumps(dict(config=CONFIG,feature_id=feature_id,
        fit_documents=encoder.fit_documents,vocabulary_size=encoder.vocabulary_size,
        fitted_dimensions=encoder.k),indent=2))
    del m,encoder,all_frames,augmented;gc.collect();return result
