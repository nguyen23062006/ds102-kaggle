import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from catboost_v3 import cat_features, make_cat
from train_submit import load_data, SEED
from improved_features import prepare
from select_v2 import macro

OUT=Path('kaggle_ds102/experiments_v3')
OUT.mkdir(exist_ok=True)
data=load_data(Path('/Users/ngien2306/Downloads/ds-102-r-11-cnvn'))
df=prepare(data['train']); y=df.spam_label.to_numpy()
X=cat_features(df)
splits=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED).split(X,y,groups=df.comment))
weights=['SqrtBalanced','Balanced']
scores=np.zeros((len(weights),len(df),4)); folds=np.full(len(df),-1)
for fold,(tr,va) in enumerate(splits):
    print('FOLD',fold+1,flush=True); folds[va]=fold
    for i,weight in enumerate(weights):
        model=make_cat(weight)
        model.fit(X.iloc[tr],y[tr],verbose=250)
        scores[i,va]=model.predict_proba(X.iloc[va])
        print(weight,'macro F1',macro(y[va],scores[i,va].argmax(1)),flush=True)
    np.savez_compressed(OUT/'partial.npz',scores=scores,y=y,folds=folds)
np.savez_compressed(OUT/'oof.npz',scores=scores,y=y,folds=folds)
(OUT/'configs.json').write_text(json.dumps(weights))
for weight,ss in zip(weights,scores):
    print('OOF',weight,macro(y,ss.argmax(1)),flush=True)
