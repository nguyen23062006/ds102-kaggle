"""Controlled Antigravity H1: append its unchanged 7 intent features to V11 SVD.

Same fitting rows, representation, CatBoost parameters and held-out groups.
No eval_set or early stopping on the evaluation fold. All four labels remain
free to change. Human interpretations of hidden test labels are not training data.
"""
import argparse, hashlib, json, os, platform, sys
from pathlib import Path

def guard():
    if platform.system()!='Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Hosted Google Colab only; local fitting disabled.')

if __name__=='__main__':guard()
sys.path.insert(0,'/content/ds102_v53/source')
import joblib
import numpy as np
import pandas as pd
from experiment_v53 import Representations, tree
from h1_intent_features import IntentFeatureExtractor
import h1_intent_features
import protocol_v60 as driver

STATE={}
CONFIG=dict(model_family='Antigravity H1: SVD128 + unchanged seven intent features + CatBoost',
    pretrained=False,external_data=False,fit_data='training partitions only',
    trees=800,depth=5,learning_rate=.06,l2_leaf_reg=8,cat_weight='SqrtBalanced',
    early_stopping=False,holdout_eval_set=False,frozen_class_predictions=False,
    combination='.35*SVM + .35*CatText + .15*old_SVD128 + .15*H1_SVD128',
    class_weights=[1.,1.2,1.2,.9],
    gates=dict(cluster_min_delta=.003,exact_min_delta=.002,min_positive_folds=2,
        max_fold_drop=.005,min_novel_fake_f1_delta=0.,bootstrap_q025_positive=True,
        min_dev_delta=.001,min_dev_without_exact_overlap_delta=.001),
    author_h1_diagnostic=dict(min_class2_recall_gain=.05,max_class0_f1_drop=.01),
    caveat='Adaptive hypotheses; author inspected some test text but hidden labels are unknown. No guessed labels used.')

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def append_intent(frame,latent):
    values=IntentFeatureExtractor().transform(frame)
    assert values.shape==(len(frame),7) and np.isfinite(values).all()
    return pd.concat([latent.reset_index(drop=True),pd.DataFrame(values,
        columns=[f'intent_{i}' for i in range(7)])],axis=1)

class Predictor:
    def __init__(self,encoder,model):self.encoder=encoder;self.model=model;self.classes_=model.classes_
    def predict_proba(self,frame):
        return self.model.predict_proba(append_intent(frame,self.encoder.latent(frame)))

def fitted_head(train,groups,seed=102):
    guard()
    key=hashlib.sha256(json.dumps(dict(indices=train.index.tolist(),config=driver.CONFIG,
        smoke=STATE['smoke']),sort_keys=True).encode()).hexdigest()
    folder=STATE['out']/'partitions'/key;folder.mkdir(parents=True,exist_ok=True)
    encoder=None;latent=None
    parent=Path('/content/ds102_v53/run')
    pm=json.loads((parent/'run_manifest.json').read_text())
    mid=hashlib.sha256(json.dumps(pm,sort_keys=True).encode()).hexdigest()
    if not STATE['smoke']:
        for part in sorted((parent/'cluster').glob('fold*')):
            z=np.load(part/'indices.npz')
            if not np.array_equal(z['tr'],train.index.to_numpy()):continue
            enc=joblib.load(part/'representations.joblib')
            data=joblib.load(part/'latent_features.joblib')
            assert enc['manifest_id']==data['manifest_id']==mid
            encoder=enc['encoder'];latent=data['frames'][0]
            assert len(latent)==len(train) and latent.shape[1]==577
            (folder/'representation_reference.json').write_text(json.dumps(dict(
                parent=str(part),parent_manifest_id=mid,indices_sha256=digest(part/'indices.npz')),indent=2))
            print(json.dumps(dict(stage='reused_matching_svd',partition=str(part),rows=len(train))),flush=True)
            break
    if encoder is None:
        ck=folder/'representation.joblib'
        if ck.exists():
            cached=joblib.load(ck);assert cached['key']==key
            encoder=cached['encoder'];latent=cached['latent']
        else:
            encoder=Representations(smoke=STATE['smoke']);x,_=encoder.fit_base(train)
            latent=encoder.latent(train,x)
            joblib.dump(dict(key=key,encoder=encoder,latent=latent),ck)
    model=tree(20 if STATE['smoke'] else 800)
    path=folder/'h1_classifier.cbm'
    if path.exists():model.load_model(str(path))
    else:
        matrix=append_intent(train,latent)
        print(json.dumps(dict(stage='fit_intent_catboost',rows=len(train),features=matrix.shape[1])),flush=True)
        model.fit(matrix,train.spam_label.to_numpy(),verbose=200)
        tmp=folder/'h1_classifier.tmp.cbm';model.save_model(str(tmp));tmp.replace(path)
    assert np.array_equal(model.classes_,[0,1,2,3])
    return Predictor(encoder,model)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True)
    p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    STATE.update(out=Path(a.output),smoke=a.smoke)
    driver.CONFIG={**CONFIG,'head_source_sha256':digest(__file__),
        'intent_source_sha256':digest(h1_intent_features.__file__)}
    driver.main(a,fitted_head)
