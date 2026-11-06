"""Train-only lexical/SVD features with asymmetric XGBoost trees, CPU only."""
import gc,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from lexical_chi2_v41 import LexicalEncoder
from cloud_checks import guard,checked

CONFIG=dict(n_estimators=500,max_depth=3,learning_rate=.05,min_child_weight=3,
    subsample=.85,colsample_bytree=.85,reg_lambda=10.,reg_alpha=.1,
    tree_method='hist',device='cpu',objective='multi:softprob',num_class=4,
    eval_metric='mlogloss',random_state=102,n_jobs=2,verbosity=0)

def fit_predict(train,frames,folder,feature_id,status,cache_path=None,cache_id=None):
    guard();assert cache_path is not None
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    q=np.load(cache_path,allow_pickle=False);assert q['feature_id'].item()==cache_id
    cols=q['numeric_columns'][:575].tolist();assert cols==[f'f_{i}' for i in range(575)]
    status('fit_train_only_lexical_selector_and_category_encoder')
    selector=LexicalEncoder().fit(train)
    enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
    cats=enc.fit_transform(train[['category','rating']].astype(str))
    matrices=[]
    for i,(prefix,df) in enumerate(zip(['train']+[f'frame{j}' for j in range(len(frames))],[train]+frames)):
        assert np.array_equal(q[prefix+'_categories'].astype(str),df[['category','rating']].astype(str).to_numpy())
        numeric=q[prefix+'_numeric'][:,:575].astype(np.float32)
        onehot=cats if i==0 else enc.transform(df[['category','rating']].astype(str))
        a=np.hstack([numeric,selector.transform(df),onehot]).astype(np.float32)
        assert len(a)==len(df) and np.isfinite(a).all();matrices.append(a)
    y=train.spam_label.to_numpy();counts=np.bincount(y,minlength=4)
    assert (counts>0).all();weight=np.sqrt(counts.max()/counts[y]).astype(np.float32)
    # Match CatBoost SqrtBalanced normalization (largest class has weight one).
    m=XGBClassifier(**CONFIG)
    status('fit_xgboost_lexical',features=matrices[0].shape[1],trees=CONFIG['n_estimators'])
    m.fit(matrices[0],y,sample_weight=weight,eval_set=[(matrices[0],y)],verbose=100)
    assert np.array_equal(m.classes_,[0,1,2,3])
    ps=[checked(m.predict_proba(x),len(df)) for x,df in zip(matrices[1:],frames)]
    m.save_model(folder/'xgboost.ubj')
    joblib.dump(dict(selector=selector,categories=enc),folder/'encoders.joblib',compress=3)
    (folder/'metadata.json').write_text(json.dumps(dict(config=CONFIG,feature_id=feature_id,
        selected_columns=selector.selected_columns,fit_class_counts=counts.tolist(),
        min_weight=float(weight.min()),max_weight=float(weight.max())),indent=2))
    del m,selector,enc,matrices;gc.collect();return ps
