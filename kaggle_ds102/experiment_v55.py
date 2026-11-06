"""Fixed RBF-SVM on training-only latent features, calibrated inside outer train.

No pretrained assets. The upstream SVD is fitted on the outer fitting rows;
inner calibration holds out labels and refits scaler/SVC, but does not refit
upstream SVD for each calibration split. Outer grouped evaluation is untouched
by all feature, classifier and calibration fitting. No fully nested upstream
calibration claim is made.
"""
import argparse,hashlib,json,os,platform,sys
from pathlib import Path

def guard():
    if platform.system()!='Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Hosted Google Colab only; local fitting disabled.')

if __name__=='__main__':guard()
sys.path.insert(0,'/content/ds102_v53/source')
import joblib
import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.svm import SVC
from experiment_v53 import Representations
import protocol_v55 as driver

CONFIG=dict(model_family='RBF-SVM on SVD128/numeric/product-relation features',
    C=4.,gamma='scale',numeric_clip=5.,categorical_scale=.2,
    sample_weight='sqrt inverse class frequency, both SVC and calibration',
    calibration='three inner group folds; sigmoid; ensemble=True; n_jobs=1',
    representation_calibration_caveat='SVD fits outer training rows, including inner calibration text without labels; scaler/SVC refit on inner training rows. No outer evaluation row fits any stage.',
    seed=102,cluster_split_seed=2053,cluster_threshold=.7,folds=3,
    candidate_alpha=.15,class_weights=[1.,1.2,1.2,.9],
    gates=dict(cluster_min_delta=.003,exact_min_delta=.002,min_positive_folds=2,
        max_fold_drop=.005,min_novel_fake_f1_delta=0.,bootstrap_q025_positive=True,
        min_dev_delta=.001,min_dev_without_exact_overlap_delta=.001),
    pretrained=False,external_data=False,fit_data='outer training partitions only')
STATE={}

class LatentSVC(ClassifierMixin,BaseEstimator):
    def fit(self,X,y,sample_weight=None):
        guard()
        self.columns_=[c for c in X if c not in ['category','rating']]
        self.scaler_=StandardScaler()
        self.category_=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float64)
        x=np.clip(self.scaler_.fit_transform(X[self.columns_]),-5,5)
        cat=self.category_.fit_transform(X[['category','rating']])
        self.model_=SVC(C=4.,gamma='scale',kernel='rbf',class_weight=None,
            probability=False,decision_function_shape='ovr',cache_size=512,
            max_iter=100000,random_state=102)
        self.model_.fit(np.hstack([x,.2*cat]),y,sample_weight=sample_weight)
        assert self.model_.fit_status_==0,'SVM did not converge'
        self.classes_=self.model_.classes_
        return self
    def transform(self,X):
        x=np.clip(self.scaler_.transform(X[self.columns_]),-5,5)
        return np.hstack([x,.2*self.category_.transform(X[['category','rating']])])
    def decision_function(self,X):return self.model_.decision_function(self.transform(X))
    def predict(self,X):return self.model_.predict(self.transform(X))

class RawFramesPredictor:
    def __init__(self,encoder,classifier):
        self.encoder=encoder;self.classifier=classifier;self.classes_=classifier.classes_
    def predict_proba(self,frame):
        return self.classifier.predict_proba(self.encoder.latent(frame))

def fitted_head(train,groups,seed=102):
    guard();y=train.spam_label.to_numpy();encoder=None;features=None
    parent=Path('/content/ds102_v53/run')
    pm=json.loads((parent/'run_manifest.json').read_text())
    mid=hashlib.sha256(json.dumps(pm,sort_keys=True).encode()).hexdigest()
    indices=train.index.to_numpy()
    for f in range(3):
        folder=parent/'cluster'/f'fold{f}';idx=folder/'indices.npz'
        if idx.exists() and np.array_equal(np.load(idx)['tr'],indices):
            enc=joblib.load(folder/'representations.joblib')
            feat=joblib.load(folder/'latent_features.joblib')
            assert enc['manifest_id']==feat['manifest_id']==mid
            encoder=enc['encoder'];features=feat['frames'][0]
            break
    if encoder is None:
        key=hashlib.sha256(json.dumps(dict(indices=indices.tolist(),config=driver.CONFIG),sort_keys=True).encode()).hexdigest()
        cache=STATE['out']/'feature_cache'/key;cache.mkdir(parents=True,exist_ok=True)
        path=cache/'representations.joblib'
        if path.exists():
            saved=joblib.load(path);assert saved['key']==key
            encoder=saved['encoder'];features=saved['features']
        else:
            encoder=Representations();x,_=encoder.fit_base(train)
            features=encoder.latent(train,x)
            joblib.dump(dict(key=key,encoder=encoder,features=features),path)
    assert len(features)==len(train) and features.shape[1]==577
    splits=list(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(train,y,groups))
    for tr,va in splits:
        assert not set(groups[tr])&set(groups[va])
        assert set(y[tr])==set(y[va])=={0,1,2,3}
    sw=np.sqrt(len(y)/(4*np.bincount(y,minlength=4)))[y]
    model=CalibratedClassifierCV(LatentSVC(),method='sigmoid',cv=splits,n_jobs=1,ensemble=True)
    model.fit(features,y,sample_weight=sw)
    assert np.array_equal(model.classes_,[0,1,2,3])
    return RawFramesPredictor(encoder,model)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    STATE['out']=Path(a.output)
    driver.CONFIG={**CONFIG,'head_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    driver.main(a,fitted_head)
