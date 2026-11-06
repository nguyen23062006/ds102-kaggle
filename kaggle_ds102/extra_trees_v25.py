"""ExtraTrees on the unchanged train-only SVD128 representation."""
import gc,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder
from latent_v6 import LatentFeatures
from experiment_v10 import guard,checked

CONFIG=dict(n_estimators=600,max_features='sqrt',min_samples_leaf=2,
    bootstrap=False,random_state=102,n_jobs=2,class_weight='sqrt inverse frequency',
    dimensions=128,svd_iterations=5)

def fit_predict(train,frames,folder,feature_id,status,cache_path=None,cache_id=None):
    guard();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    ck=folder/'svd128_features.npz'
    if cache_path is not None:
        ck=Path(cache_path);assert ck.exists()
    if ck.exists():
        z=np.load(ck,allow_pickle=False)
        assert z['feature_id'].item()==(cache_id if cache_path is not None else feature_id)
        assert len(z['numeric_columns'])>=575
        cols=z['numeric_columns'][:575].tolist();assert cols==[f'f_{i}' for i in range(575)]
        xt=pd.DataFrame(z['train_numeric'][:,:575],columns=cols);external=[]
        for j,col in enumerate(['category','rating']):xt[col]=z['train_categories'][:,j]
        for i in range(len(frames)):
            x=pd.DataFrame(z[f'frame{i}_numeric'][:,:575],columns=cols)
            for j,col in enumerate(['category','rating']):x[col]=z[f'frame{i}_categories'][:,j]
            external.append(x)
    else:
        status('fit_full_train_svd128')
        f=LatentFeatures();f.svd.set_params(n_components=128,random_state=102)
        xt=f.transform(train,fit=True);external=[f.transform(d) for d in frames]
        cols=[c for c in xt if c not in ['category','rating']];assert len(cols)==575
        arrays=dict(feature_id=feature_id,numeric_columns=np.array(cols),
            train_numeric=xt[cols].to_numpy(dtype=np.float32),train_categories=xt[['category','rating']].to_numpy(dtype=str))
        for i,x in enumerate(external):
            arrays[f'frame{i}_numeric']=x[cols].to_numpy(dtype=np.float32)
            arrays[f'frame{i}_categories']=x[['category','rating']].to_numpy(dtype=str)
        tmp=ck.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(ck)
        del f,arrays;gc.collect()
    assert len(xt)==len(train) and [len(x) for x in external]==[len(d) for d in frames]
    for x,df in zip([xt]+external,[train]+frames):
        assert np.array_equal(x[['category','rating']].to_numpy(dtype=str),df[['category','rating']].astype(str).to_numpy())
        assert np.isfinite(x[cols].to_numpy()).all()
    enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
    cat=enc.fit_transform(xt[['category','rating']])
    xx=np.hstack([xt[cols].to_numpy(dtype=np.float32),cat])
    vv=[np.hstack([x[cols].to_numpy(dtype=np.float32),enc.transform(x[['category','rating']])]) for x in external]
    counts=np.bincount(train.spam_label.to_numpy(),minlength=4);assert (counts>0).all()
    weights={i:float(np.sqrt(len(train)/(4*n))) for i,n in enumerate(counts)}
    status('fit_extra_trees',feature_columns=xx.shape[1])
    m=ExtraTreesClassifier(n_estimators=600,max_features='sqrt',min_samples_leaf=2,
        bootstrap=False,random_state=102,n_jobs=2,class_weight=weights)
    m.fit(xx,train.spam_label.to_numpy());assert np.array_equal(m.classes_,[0,1,2,3])
    result=[checked(m.predict_proba(x),len(x)) for x in vv]
    (folder/'metadata.json').write_text(json.dumps(dict(config=CONFIG,class_counts=counts.tolist(),
        class_weights=weights,extra_trees_feature_importance=m.feature_importances_.tolist()),indent=2))
    del xt,external,xx,vv,m;gc.collect();return result
