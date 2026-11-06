"""Nested-calibrated 0-vs-2 SVM, with every feature fit inside inner folds."""
import gc,json,warnings
from pathlib import Path
import joblib
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from improved_features import FeaturesV2
from cloud_checks import guard,checked

CONFIG=dict(C=.5,class_weight='balanced',inner_folds=3,calibration='sigmoid',
    ensemble=False,max_iter=30000,seed=102,conditional_expert_weight=.5,
    fixed_predictions=[1,3],fit_labels=[0,2])

class ReviewFeatures(TransformerMixin,BaseEstimator):
    def fit(self,X,y=None):
        guard();self.features_=FeaturesV2();self.features_.fit_transform(X)
        return self
    def transform(self,X):
        a,b=self.features_.transform(X)
        return sparse.hstack([a,b],format='csr')

def fit_predict(train,frames,folder,feature_id,status,cache_path=None,cache_id=None):
    guard();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    fit=train[train.spam_label.isin([0,2])].reset_index(drop=True)
    y=fit.spam_label.to_numpy();X=fit.drop(columns='spam_label')
    cv=list(StratifiedGroupKFold(3,shuffle=True,random_state=102).split(X,y,X.comment))
    for tr,va in cv:
        assert set(y[tr])=={0,2} and set(y[va])=={0,2}
        assert not set(X.comment.iloc[tr])&set(X.comment.iloc[va])
    estimator=Pipeline([('features',ReviewFeatures()),('svm',LinearSVC(C=.5,
        class_weight='balanced',dual=True,max_iter=30000,random_state=102))])
    model=CalibratedClassifierCV(estimator=estimator,method='sigmoid',cv=cv,
        ensemble=False,n_jobs=1)
    status('fit_inner_group_calibrated_binary_svm',fit_rows=len(fit),inner_folds=3)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(X,y)
    assert np.array_equal(model.classes_,[0,2])
    ps=[]
    for df in frames:
        p=model.predict_proba(df)
        assert p.shape==(len(df),2) and np.isfinite(p).all()
        q=np.zeros((len(df),4));q[:,0]=p[:,0];q[:,2]=p[:,1]
        ps.append(checked(q,len(df)))
    joblib.dump(model,folder/'binary_svm_calibrated.joblib',compress=3)
    (folder/'metadata.json').write_text(json.dumps(dict(config=CONFIG,feature_id=feature_id,
        fit_rows=len(fit),fit_class_counts={int(c):int((y==c).sum()) for c in [0,2]},
        feature_fitting='Each inner calibration fitting partition has its own vectorizers/scaler; final estimator fits all outertrain0/2. No heldout outer/dev/test features are fit.'),indent=2))
    del model,fit,estimator;gc.collect();return ps
