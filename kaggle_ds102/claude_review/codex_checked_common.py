"""Checked statistical helpers; no model or vectorizer fitting."""
import numpy as np
from scipy.special import softmax
from review_common import (W_V5,W_V6,near_dup_groups,max_jaccard,
                           class_f1,macro,class1_strata)


def dedup_train_weights(y,groups):
    """Accept only a training partition; never normalize using validation labels."""
    y=np.asarray(y);_,inv=np.unique(groups,return_inverse=True)
    sizes=np.bincount(inv)[inv].astype(float)
    w=1/np.sqrt(sizes)
    for c in np.unique(y):
        m=y==c;w[m]*=m.sum()/w[m].sum()
    return w


def recipe_probabilities(svm,cat,lat):
    """Apply softmax separately to each model's margins, before ensembling folds."""
    base=.5*softmax(3*svm,axis=1)+.5*cat
    return {'v5':base,'v6':.7*base+.3*lat}


def labels(probabilities,name):
    return (probabilities*(W_V5 if name=='v5' else W_V6)).argmax(1)


def paired_cluster_bootstrap(y,pred_a,pred_b,groups,n=2000,seed=102):
    """Paired conditional bootstrap of fixed predictions, NOT a post-selection test."""
    y=np.asarray(y);a=np.asarray(pred_a);b=np.asarray(pred_b)
    if len(y)==0:raise ValueError('Cannot bootstrap an empty subset')
    if not (len(y)==len(a)==len(b)==len(groups)):raise ValueError('Length mismatch')
    _,inv=np.unique(groups,return_inverse=True);g=int(inv.max()+1)
    rng=np.random.default_rng(seed);d=np.empty(n);missing=0
    for i in range(n):
        w=np.bincount(rng.integers(0,g,g),minlength=g)[inv].astype(float)
        missing+=int((np.bincount(y,weights=w,minlength=4)==0).any())
        d[i]=macro(y,b,w)-macro(y,a,w)
    q=np.quantile(d,[.025,.05,.95,.975])
    return {'observed_delta':macro(y,b)-macro(y,a),'mean':float(d.mean()),
            'sd':float(d.std(ddof=1)),'q025':float(q[0]),'q05':float(q[1]),
            'q95':float(q[2]),'q975':float(q[3]),'fraction_delta_gt_0':float((d>0).mean()),
            'groups':g,'rows':len(y),'resamples':n,'seed':seed,
            'fraction_resamples_missing_a_class':missing/n,
            'intervals':'q025-q975: central 95%; q05-q95: central 90%',
            'caveat':'Conditional on these selected/fitted predictions. Does not remove train/validation template overlap, selection bias, or training-sample uncertainty. Positive fraction is not a posterior probability or a calibrated p-value.'}


def group_summary(groups,y=None):
    _,inv=np.unique(groups,return_inverse=True);sizes=np.bincount(inv)
    out={'groups':len(sizes),'largest':int(sizes.max()),
         'largest_fraction':float(sizes.max()/len(inv)),
         'rows_in_groups_ge2':int((sizes[inv]>=2).sum()),
         'group_size_quantiles':np.quantile(sizes,[.5,.9,.99,1.]).tolist()}
    if y is not None:
        mass=np.bincount(4*inv+np.asarray(y),minlength=4*len(sizes)).reshape(-1,4)
        mixed=(mass>0).sum(1)>1
        out['mixed_label_groups']=int(mixed.sum())
        out['rows_in_mixed_label_groups']=int(sizes[mixed].sum())
        out['class1_rows_in_groups_ge2']=int(((sizes[inv]>=2)&(np.asarray(y)==1)).sum())
    return out
