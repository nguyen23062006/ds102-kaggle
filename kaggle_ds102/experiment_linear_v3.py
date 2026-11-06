import json
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold
from train_submit import load_data, make_model, SEED
from improved_features import prepare, FeaturesV2
from select_v2 import macro

out=Path('kaggle_ds102/experiments_v3'); out.mkdir(exist_ok=True)
df=prepare(load_data(Path('/Users/ngien2306/Downloads/ds-102-r-11-cnvn'))['train'])
y=df.spam_label.to_numpy()
configs=[{'C':c,'meta_scale':s,'weight':'balanced'} for c,s in [(.1,1),(.25,1),(.25,2),(.5,2)]]
scores=np.zeros((len(configs),len(df),4))
folds=np.full(len(df),-1)
for fold,(tr,va) in enumerate(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED).split(df,y,groups=df.comment)):
    print('FOLD',fold+1,flush=True); folds[va]=fold
    f=FeaturesV2(); xt,mt=f.fit_transform(df.iloc[tr]); xv,mv=f.transform(df.iloc[va])
    for i,cfg in enumerate(configs):
        xx=sparse.hstack([xt,mt*cfg['meta_scale']],format='csr')
        vv=sparse.hstack([xv,mv*cfg['meta_scale']],format='csr')
        m=make_model(cfg,y[tr]).fit(xx,y[tr])
        scores[i,va]=m.decision_function(vv)
        print(cfg,'F1',macro(y[va],scores[i,va].argmax(1)),flush=True)
np.savez_compressed(out/'linear_oof.npz',scores=scores,y=y,folds=folds)
(out/'linear_configs.json').write_text(json.dumps(configs))
for cfg,s in zip(configs,scores):print('OOF',cfg,macro(y,s.argmax(1)),flush=True)
