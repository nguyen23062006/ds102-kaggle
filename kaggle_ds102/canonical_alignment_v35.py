"""Ridge-regularized canonical review/title alignment fitted on train class0."""
import gc,json
from pathlib import Path
import numpy as np
import pandas as pd
from latent_v6 import LatentFeatures,make_model
from experiment_v10 import guard,checked

CONFIG=dict(svd_dimensions=128,canonical_dimensions=32,ridge_relative=.1,
    alignment_fit_label=0,trees=800,depth=5,lr=.06,l2=8,seed=102)

class CanonicalAlignment:
    def __init__(self,dimensions=32,ridge=.1):
        self.dimensions=dimensions;self.ridge=ridge

    def fit(self,z,t,mask):
        guard();x=np.asarray(z[mask],dtype=np.float64);y=np.asarray(t[mask],dtype=np.float64)
        assert len(x)>self.dimensions and x.shape==y.shape
        self.mean_x=x.mean(0);self.mean_y=y.mean(0);x=x-self.mean_x;y=y-self.mean_y
        n=len(x)-1;cx=x.T@x/n;cy=y.T@y/n;cxy=x.T@y/n
        def invroot(c):
            regularizer=self.ridge*max(float(np.trace(c))/len(c),1e-12)
            e,v=np.linalg.eigh(c);return (v*(1/np.sqrt(np.maximum(e,0)+regularizer)))@v.T
        ax=invroot(cx);ay=invroot(cy);u,s,vt=np.linalg.svd(ax@cxy@ay,full_matrices=False)
        k=self.dimensions;self.wx=ax@u[:,:k];self.wy=ay@vt.T[:,:k];self.singular_values=s[:k]
        assert s.max()<=1.000001
        signs=np.sign(self.wx[np.argmax(np.abs(self.wx),axis=0),np.arange(k)]);signs[signs==0]=1
        self.wx*=signs;self.wy*=signs
        self.fit_rows=len(x);return self

    def transform(self,z,t):
        u=(np.asarray(z,dtype=np.float64)-self.mean_x)@self.wx
        v=(np.asarray(t,dtype=np.float64)-self.mean_y)@self.wy
        un=np.linalg.norm(u,axis=1);vn=np.linalg.norm(v,axis=1)
        cos=(u*v).sum(1)/np.maximum(un*vn,1e-9);dist=np.linalg.norm(u-v,axis=1)
        out=np.hstack([u,v,u*v,np.abs(u-v),cos[:,None],dist[:,None],np.abs(un-vn)[:,None]]).astype(np.float32)
        assert np.isfinite(out).all();return out

def fit_predict(train,frames,folder,feature_id,status,cache_path=None,cache_id=None):
    guard();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if cache_path is not None:
        q=np.load(cache_path,allow_pickle=False);assert q['feature_id'].item()==cache_id
        cols=q['numeric_columns'][:575].tolist();assert cols==[f'f_{i}' for i in range(575)]
        xt=pd.DataFrame(q['train_numeric'][:,:575],columns=cols);external=[]
        for j,c in enumerate(['category','rating']):xt[c]=q['train_categories'][:,j]
        for i in range(len(frames)):
            x=pd.DataFrame(q[f'frame{i}_numeric'][:,:575],columns=cols)
            for j,c in enumerate(['category','rating']):x[c]=q[f'frame{i}_categories'][:,j]
            external.append(x)
    else:
        status('fit_full_svd128_features')
        f=LatentFeatures();f.svd.set_params(n_components=128,random_state=102)
        xt=f.transform(train,fit=True);external=[f.transform(d) for d in frames]
        cols=[c for c in xt if c not in ['category','rating']];assert len(cols)==575
        del f;gc.collect()
    for x,df in zip([xt]+external,[train]+frames):
        assert len(x)==len(df)
        assert np.array_equal(x[['category','rating']].to_numpy(dtype=str),df[['category','rating']].astype(str).to_numpy())
    a=xt[cols].to_numpy(dtype=np.float32);mask=train.spam_label.to_numpy()==0
    status('fit_class0_canonical_alignment',alignment_rows=int(mask.sum()))
    align=CanonicalAlignment().fit(a[:,:128],a[:,128:256],mask)
    np.savez_compressed(folder/'canonical_projection.npz',feature_id=feature_id,mean_x=align.mean_x,mean_y=align.mean_y,wx=align.wx,wy=align.wy,correlations=align.singular_values)
    all_frames=[xt]+external;augmented=[]
    for x in all_frames:
        numeric=x[cols].to_numpy(dtype=np.float32)
        extra=align.transform(numeric[:,:128],numeric[:,128:256])
        joined=pd.concat([x.reset_index(drop=True),pd.DataFrame(extra,columns=[f'cca_{i}' for i in range(extra.shape[1])])],axis=1)
        assert joined.shape[1]==708
        augmented.append(joined)
    xt=augmented[0];external=augmented[1:]
    status('fit_catboost_canonical_alignment',feature_columns=xt.shape[1])
    m=make_model();m.fit(xt,train.spam_label.to_numpy(),verbose=200)
    assert np.array_equal(m.classes_,[0,1,2,3])
    result=[checked(m.predict_proba(x),len(x)) for x in external]
    m.save_model(str(folder/'catboost.cbm'))
    (folder/'metadata.json').write_text(json.dumps(dict(config=CONFIG,feature_id=feature_id,
        alignment_rows=int(mask.sum()),canonical_correlations=align.singular_values.tolist()),indent=2))
    del xt,external,m,augmented,all_frames,align;gc.collect();return result
