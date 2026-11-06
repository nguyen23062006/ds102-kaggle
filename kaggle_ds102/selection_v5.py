"""Coarse probability reweighting, validated on separate OOF folds."""
from itertools import product
import numpy as np
from select_v2 import macro


def fit_weights(probabilities, y):
    choices = []
    # A small fixed grid: minority-class recall matters for macro F1.
    # Identity is included. Never access development or test labels here.
    for a,b,c in product([.8,1.,1.2,1.4], [.8,1.,1.2,1.4], [.9,1.,1.1]):
        w=np.array([1.,a,b,c])
        score=macro(y,(probabilities*w).argmax(1))
        penalty=.002*float(np.square(np.log(w)).sum())
        choices.append((score-penalty,-float(np.square(np.log(w)).sum()),w))
    return max(choices,key=lambda row:row[:2])[2]


def cross_weight(probabilities,y,folds):
    pred=np.empty(len(y),dtype=int); weights=[]; fold_deltas=[]
    for fold in np.unique(folds):
        fit=folds!=fold
        w=fit_weights(probabilities[fit],y[fit]); weights.append(w.tolist())
        pred[~fit]=(probabilities[~fit]*w).argmax(1)
        fold_deltas.append(macro(y[~fit],pred[~fit])-
            macro(y[~fit],probabilities[~fit].argmax(1)))
    raw=macro(y,probabilities.argmax(1)); cross=macro(y,pred)
    use=bool(cross>=raw+.002 and min(fold_deltas)>=-.002)
    return {'raw_macro_f1':raw,'cross_weight_macro_f1':cross,
        'fold_deltas':fold_deltas,'fold_weights':weights,'use_weights':use,
        'weights':fit_weights(probabilities,y).tolist() if use else [1.,1.,1.,1.],
        'selection_score':cross if use else raw}


def compare(reference,new_oof,y,folds,configs):
    rows=[]
    row=cross_weight(reference,y,folds)
    row.update(model_index=-1,config='v3_reference',new_model_weight=0.)
    rows.append(row)
    for i,values in enumerate(new_oof):
        for alpha in [.15,.3,.5,1.]:
            p=(1-alpha)*reference+alpha*values
            row=cross_weight(p,y,folds)
            row.update(model_index=i,config=configs[i],new_model_weight=alpha)
            rows.append(row)
    return sorted(rows,key=lambda row:row['selection_score'],reverse=True)
