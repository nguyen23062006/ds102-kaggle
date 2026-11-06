"""Fixed hinge-loss SVM/NB-SVM using unchanged original features."""
import gc,json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.special import softmax
from sklearn.svm import LinearSVC
from improved_features import FeaturesV2,nb_ratios
from train_submit_v2 import RECIPE
from experiment_v10 import guard,checked

CONFIG=dict(loss='hinge',C=.5,class_weight='balanced',dual=True,max_iter=30000,
    temperature=3.,svm_weight=.75,nb_weight=.25,new_svm_coefficient=.175,seed=102)

def fit_predict(train,frames,folder,feature_id,status):
    guard();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    y=train.spam_label.to_numpy();f=FeaturesV2()
    status('fit_original_tfidf')
    xt,mt=f.fit_transform(train);vv=[f.transform(d) for d in frames]
    result=[np.zeros((len(d),4)) for d in frames];ratios=nb_ratios(xt,y)
    status('fit_hinge_svm',text_features=xt.shape[1])
    m=LinearSVC(C=.5,loss='hinge',class_weight='balanced',dual=True,max_iter=30000,tol=1e-4,random_state=102)
    m.fit(sparse.hstack([xt,mt],format='csr'),y)
    assert np.array_equal(m.classes_,[0,1,2,3]) and np.max(m.n_iter_)<m.max_iter
    iterations=[int(m.n_iter_)]
    for j,(xv,mv) in enumerate(vv):result[j]+=.75*m.decision_function(sparse.hstack([xv,mv],format='csr'))
    del m;gc.collect()
    for cls,r in enumerate(ratios):
        status('fit_hinge_nb_svm',class_index=cls)
        m=LinearSVC(C=.5,loss='hinge',class_weight='balanced',dual=True,max_iter=30000,tol=1e-4,random_state=102)
        m.fit(sparse.hstack([xt.multiply(r),mt],format='csr'),(y==cls).astype(int))
        assert np.max(m.n_iter_)<m.max_iter;iterations.append(int(m.n_iter_))
        for j,(xv,mv) in enumerate(vv):result[j][:,cls]+=.25*m.decision_function(sparse.hstack([xv.multiply(r),mv],format='csr'))
        del m;gc.collect()
    (folder/'metadata.json').write_text(json.dumps(dict(config=CONFIG,feature_id=feature_id,iterations=iterations),indent=2))
    return [checked(softmax(3*p,axis=1),len(d)) for p,d in zip(result,frames)]
