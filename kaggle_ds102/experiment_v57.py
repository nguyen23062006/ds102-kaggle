"""Conditional fastText head for classes 0/2/3; baseline owns fake decisions.

Proposed after V56's aggregate class metrics. V56 remains rejected. This is
a new supervised fit on three classes, not an export of post-hoc relabelled
V56 output. Adaptive selection history and conditional-CI limitations remain.
"""
import argparse,hashlib,json,os,platform,sys
from pathlib import Path

def guard():
    if platform.system()!='Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Hosted Google Colab only; local fitting disabled.')

if __name__=='__main__':guard()
sys.path.insert(0,'/content/ds102_v56/source')
import joblib
import numpy as np
from catboost import CatBoostClassifier
import experiment_v56 as features
import protocol_v57 as driver

ROOT=Path('/content/ds102_v57')
CONFIG={**features.CONFIG,'model_family':'Conditional classes0/2/3 fastText32 CatBoost; fake mask frozen',
    'supervised_classes':[0,2,3],'decision_rule':'Baseline class1 predictions fixed; otherwise mix conditional baseline/common head0.85/0.15',
    'combination':'routing; preserve baseline fake predictions, conditional 0/2/3 mixture',
    'selection_note':'Proposed after V56 aggregate outcomes; no independent-data or post-selection CI guarantee'}
STATE={}

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Predictor:
    def __init__(self,encoder,model):
        self.encoder=encoder;self.model=model;self.classes_=model.classes_
    def predict_proba(self,frame):
        # Four slots are the routing interface. The fitted model has 3 classes.
        p=self.model.predict_proba(self.encoder.transform(frame))
        out=np.zeros((len(frame),4));out[:,self.model.classes_.astype(int)]=p
        assert np.all(out[:,1]==0)
        return out

def fitted_head(train,groups,seed=102):
    guard();features.STATE['smoke']=STATE['smoke']
    key=hashlib.sha256(json.dumps(dict(indices=train.index.tolist(),config=driver.CONFIG,
        smoke=STATE['smoke']),sort_keys=True).encode()).hexdigest()
    folder=STATE['out']/'partitions'/key;folder.mkdir(parents=True,exist_ok=True)
    encoder=None;matrix=None
    parent=Path('/content/ds102_v56')/('smoke' if STATE['smoke'] else 'run')/'partitions'
    for candidate in parent.glob('*'):
        mp=candidate/'partition_manifest.json';marker=candidate/'embedding_complete.json'
        fp=candidate/'training_features.joblib'
        if not mp.exists() or not marker.exists() or not fp.exists():continue
        pm=json.loads(mp.read_text())
        if pm['indices']!=train.index.tolist():continue
        assert pm['smoke']==STATE['smoke']
        assert pm['config']['head_source_sha256']==digest(features.__file__)
        assert pm['config']['fasttext_build']['binary_sha256']==digest(features.BINARY)
        encoder=features.TextEncoder(candidate).fit(train)
        cached=joblib.load(fp);assert cached['key']==pm['key'];matrix=cached['features']
        (folder/'embedding_reference.json').write_text(json.dumps(dict(parent_partition=str(candidate),
            corpus_sha256=encoder.corpus_sha,model_sha256=digest(encoder.model_path),
            parent_partition_manifest_sha256=digest(mp)),indent=2))
        print(json.dumps(dict(stage='reuse_matching_training_embedding',rows=len(train),parent=str(candidate))),flush=True)
        break
    if encoder is None:
        encoder=features.TextEncoder(folder/'embedding').fit(train);matrix=encoder.transform(train)
    assert len(matrix)==len(train) and matrix.shape[1]==294
    target=train.spam_label.to_numpy();mask=target!=1
    assert set(target[mask])=={0,2,3}
    model=CatBoostClassifier(iterations=20 if STATE['smoke'] else 800,depth=5,
        learning_rate=.06,l2_leaf_reg=8,loss_function='MultiClass',auto_class_weights='SqrtBalanced',
        cat_features=['category','rating'],max_ctr_complexity=1,random_seed=102,
        thread_count=2,allow_writing_files=False,verbose=200)
    path=folder/'common_classifier.cbm'
    if path.exists():model.load_model(str(path))
    else:
        print(json.dumps(dict(stage='fit_common_classes',fitting_rows=int(mask.sum()),excluded_class1_rows=int((~mask).sum()))),flush=True)
        model.fit(matrix.loc[mask],target[mask],verbose=200)
        tmp=folder/'common_classifier.tmp.cbm';model.save_model(str(tmp));tmp.replace(path)
    assert np.array_equal(model.classes_,[0,2,3])
    return Predictor(encoder,model)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    STATE.update(out=Path(a.output),smoke=a.smoke)
    build=json.loads((features.ROOT/'fasttext_build_manifest.json').read_text())
    assert digest(features.BINARY)==build['binary_sha256']
    driver.CONFIG={**CONFIG,'head_source_sha256':digest(__file__),'embedding_source_sha256':digest(features.__file__),
        'fasttext_build':{k:build[k] for k in ['version','source_url','source_archive_sha256','binary_sha256','compiler','build_flags','license']}}
    driver.main(a,fitted_head)
