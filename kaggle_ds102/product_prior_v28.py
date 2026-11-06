"""Smoothed product/category label priors, fit on fitting rows only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_v10 import guard,checked

CONFIG=dict(smoothing=50.,power=.25,key=['category','product_name'],seed=102)
def keys(df):
    return df.category.astype(str)+'\u241f'+df.product_name.astype(str)

def fit_predict(train,frames,folder,feature_id,status):
    guard();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    status('fit_product_priors',fit_rows=len(train))
    y=train.spam_label.to_numpy();global_p=np.bincount(y,minlength=4)/len(y);assert (global_p>0).all()
    count=pd.crosstab(keys(train),y).reindex(columns=range(4),fill_value=0).astype(float)
    n=count.sum(1).to_numpy()
    smooth=pd.DataFrame((count.to_numpy()+50*global_p)/(n[:,None]+50),index=count.index,columns=range(4))
    result=[];seen=[]
    for df in frames:
        p=smooth.reindex(keys(df)).to_numpy()
        absent=~np.isfinite(p).all(1);p[absent]=global_p
        result.append(checked(p,len(df)));seen.append(float((~absent).mean()))
    (folder/'prior_metadata.json').write_text(json.dumps(dict(config=CONFIG,feature_id=feature_id,
        global_prior=global_p.tolist(),products=len(count),frame_seen_fractions=seen),indent=2))
    return result
