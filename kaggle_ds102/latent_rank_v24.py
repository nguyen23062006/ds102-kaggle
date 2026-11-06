"""Fixed 192D follow-up of the 128D model; train-only TF-IDF/SVD."""
import gc,json
from pathlib import Path
import numpy as np
import pandas as pd
from latent_v6 import LatentFeatures,make_model
from experiment_v10 import guard,checked

CONFIG=dict(dimensions=192,svd_iterations=5,seed=102,trees=800,depth=5,lr=.06,l2=8,
    classifier_weight='SqrtBalanced',vocabulary_fit='training reviews only')

def fit_predict(train,frames,folder,feature_id,status):
    guard();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    ck=folder/'rank192_features.npz'
    if ck.exists():
        z=np.load(ck,allow_pickle=False);assert z['feature_id'].item()==feature_id
        assert z['frame_count'].item()==len(frames)
        cols=z['numeric_columns'].tolist()
        xt=pd.DataFrame(z['train_numeric'],columns=cols);external=[]
        for j,col in enumerate(['category','rating']):xt[col]=z['train_categories'][:,j]
        for i in range(len(frames)):
            x=pd.DataFrame(z[f'frame{i}_numeric'],columns=cols)
            for j,col in enumerate(['category','rating']):x[col]=z[f'frame{i}_categories'][:,j]
            external.append(x)
    else:
        status('fit_train_only_svd192')
        f=LatentFeatures();f.svd.set_params(n_components=192,random_state=102)
        xt=f.transform(train,fit=True);external=[f.transform(d) for d in frames]
        cols=[c for c in xt if c not in ['category','rating']]
        assert len(cols)==4*192+63
        arrays=dict(feature_id=feature_id,frame_count=len(frames),numeric_columns=np.array(cols),
            train_numeric=xt[cols].to_numpy(dtype=np.float32),
            train_categories=xt[['category','rating']].to_numpy(dtype=str))
        for i,x in enumerate(external):
            arrays[f'frame{i}_numeric']=x[cols].to_numpy(dtype=np.float32)
            arrays[f'frame{i}_categories']=x[['category','rating']].to_numpy(dtype=str)
        tmp=ck.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(ck)
        (folder/'feature_metadata.json').write_text(json.dumps(CONFIG,indent=2))
        del f,arrays;gc.collect()
    assert len(xt)==len(train) and [len(x) for x in external]==[len(d) for d in frames]
    for x in [xt]+external:
        assert np.isfinite(x.drop(columns=['category','rating']).to_numpy()).all()
    status('fit_catboost_rank192',feature_columns=xt.shape[1])
    m=make_model();m.fit(xt,train.spam_label.to_numpy(),verbose=200)
    assert np.array_equal(m.classes_,[0,1,2,3])
    result=[checked(m.predict_proba(x),len(x)) for x in external]
    m.save_model(str(folder/'catboost.cbm'))
    del xt,external,m;gc.collect();return result
