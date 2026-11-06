"""Regularized stacking of training OOF SVM margins and CatBoost probabilities.

The meta validation reuses base OOF predictions; it is not fully nested CV.
Development remains separate and is inspected only after configuration choice.
"""
import gc,json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from experiment_v5 import hosted_guard
from train_submit import SEED
from select_v2 import macro


CONFIGS=[{'C':c,'weight':w} for c in [.1,1.] for w in ['none','sqrt']]


def features(margins,cat):
    return np.hstack([margins,np.log(np.clip(cat,1e-6,1.))])


def fit_meta(x,y,config):
    hosted_guard()
    weights=None
    if config['weight']=='sqrt':
        counts=np.bincount(y,minlength=4)
        weights={c:float(np.sqrt(len(y)/(4*counts[c]))) for c in range(4)}
    m=make_pipeline(StandardScaler(),LogisticRegression(C=config['C'],
        class_weight=weights,solver='lbfgs',max_iter=1000,random_state=SEED))
    m.fit(x,y)
    if np.max(m[-1].n_iter_)>=m[-1].max_iter:raise RuntimeError('Meta model did not converge')
    return m


def make_base_oof(train):
    """For a fully reproducible fresh fit, regenerate level-one training OOF."""
    hosted_guard()
    from finalize_v5 import baseline_predict
    y=train.spam_label.to_numpy();svm=np.zeros((len(y),4));cat=np.zeros_like(svm)
    split=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED)
    for fold,(tr,va) in enumerate(split.split(train,y,groups=train.comment)):
        print('Regenerate base OOF:',fold+1,flush=True)
        s,p=baseline_predict(train.iloc[tr],[train.iloc[va].drop(columns='spam_label')],return_components=True)
        svm[va]=s[0];cat[va]=p[0];gc.collect()
    return svm,cat


def main(component_path='/content/colab_components_v5.npz',output_dir='/content/ds102_v5_stacking'):
    hosted_guard()
    z=np.load(component_path);y=z['y'];folds=z['folds']
    x=features(z['svm_oof'],z['cat_oof'])
    base=.5*softmax(3*z['svm_oof'],axis=1)+.5*z['cat_oof']
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    oof=np.zeros((len(CONFIGS),len(y),4),dtype=np.float32);rows=[]
    for i,cfg in enumerate(CONFIGS):
        for fold in np.unique(folds):
            fit=folds!=fold;m=fit_meta(x[fit],y[fit],cfg)
            oof[i,~fit]=m.predict_proba(x[~fit])
        for alpha in [.5,1.]:
            p=(1-alpha)*base+alpha*oof[i]
            fold_scores=[macro(y[folds==f],p[folds==f].argmax(1)) for f in np.unique(folds)]
            rows.append({'family':'stacking','config':cfg,'model_index':i,
                'new_model_weight':alpha,'weights':[1.,1.,1.,1.],
                'raw_macro_f1':macro(y,p.argmax(1)),
                'selection_score':macro(y,p.argmax(1)),
                'fold_macro_f1':fold_scores,
                'caveat':'Meta CV on reused base OOF, not fully nested; dev must confirm.'})
    rows.sort(key=lambda r:r['selection_score'],reverse=True)
    np.savez_compressed(out/'oof.npz',oof=oof,y=y,folds=folds)
    (out/'selection.json').write_text(json.dumps(rows,indent=2))
    (out/'status.json').write_text(json.dumps({'stage':'complete','best':rows[0]},indent=2))
    return rows
