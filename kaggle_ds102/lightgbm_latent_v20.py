"""Regularized, balanced leaf-wise LightGBM over train-fitted SVD128 review/product relations."""
import gc,json
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from latent_v6 import LatentFeatures
from experiment_v10 import guard,checked

PARAMS=dict(objective='multiclass',n_estimators=400,learning_rate=.04,num_leaves=15,
    max_depth=6,min_child_samples=60,min_split_gain=.05,reg_alpha=.5,reg_lambda=25.,
    subsample=.8,subsample_freq=1,colsample_bytree=.8,random_state=102,n_jobs=2,
    deterministic=True,force_col_wise=True,bagging_seed=102,feature_fraction_seed=102,
    data_random_seed=102,verbosity=-1)


def frames_from_cache(z):
    columns=z['numeric_columns'].tolist();cats=['category','rating']
    train=pd.DataFrame(z['train_numeric'],columns=columns)
    result=[]
    count=int(z['frame_count'])
    for i in range(count):result.append(pd.DataFrame(z[f'frame{i}_numeric'],columns=columns))
    for j,col in enumerate(cats):
        known=sorted(set(z['train_categories'][:,j].tolist()))
        dtype=pd.CategoricalDtype(categories=known,ordered=False)
        train[col]=pd.Categorical(z['train_categories'][:,j],dtype=dtype)
        for i,frame in enumerate(result):frame[col]=pd.Categorical(z[f'frame{i}_categories'][:,j],dtype=dtype)
    return train,result


def fit_predict(train,frames,folder,feature_id,status):
    guard();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    ck=folder/'svd128_features.npz'
    if ck.exists():
        z=np.load(ck,allow_pickle=False);assert z['feature_id'].item()==feature_id
        xt,external=frames_from_cache(z)
    else:
        status('fit_svd128_features')
        f=LatentFeatures();f.svd.set_params(n_components=128,random_state=102)
        xt=f.transform(train,fit=True);external=[f.transform(d) for d in frames]
        columns=[c for c in xt if c not in ['category','rating']]
        arrays=dict(feature_id=feature_id,frame_count=len(frames),numeric_columns=np.array(columns),
            train_numeric=xt[columns].to_numpy(dtype=np.float32),
            train_categories=xt[['category','rating']].to_numpy(dtype=str))
        for i,x in enumerate(external):
            arrays[f'frame{i}_numeric']=x[columns].to_numpy(dtype=np.float32)
            arrays[f'frame{i}_categories']=x[['category','rating']].to_numpy(dtype=str)
        tmp=ck.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(ck)
        del f,xt,external,arrays;gc.collect()
        z=np.load(ck,allow_pickle=False);xt,external=frames_from_cache(z)
    assert len(xt)==len(train) and [len(x) for x in external]==[len(d) for d in frames]
    numeric=xt.select_dtypes(include='number').to_numpy()
    assert np.isfinite(numeric).all()
    y=train.spam_label.to_numpy();counts=np.bincount(y,minlength=4);assert (counts>0).all()
    weights={c:float(len(y)/(4*counts[c])) for c in range(4)}
    m=LGBMClassifier(**PARAMS,class_weight=weights)
    status('fit_lightgbm',feature_columns=xt.shape[1])
    m.fit(xt,y,categorical_feature=['category','rating'])
    assert np.array_equal(m.classes_,[0,1,2,3])
    result=[checked(m.predict_proba(x),len(x)) for x in external]
    m.booster_.save_model(str(folder/'lightgbm.txt'))
    (folder/'category_levels.json').write_text(json.dumps({c:xt[c].cat.categories.tolist() for c in ['category','rating']},ensure_ascii=False))
    del z,xt,external,numeric,m;gc.collect()
    return result
