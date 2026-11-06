"""BM25 feature weighting for the sparse SVM/NB-SVM branch, no pretrained.

Positive Robertson IDF, k1=1.2, b=.75 followed by per-view L2 normalization.
Document length is count mass in the fitted ngram vocabulary. All statistics
and vocabulary are learned strictly inside each fitting partition.
"""
import argparse, hashlib, json, os, platform, sys
from pathlib import Path

def guard():
    if platform.system() != 'Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Hosted Google Colab only; local fitting disabled.')

if __name__ == '__main__': guard()
sys.path.insert(0, '/content/ds102_v53/source')
import joblib
import numpy as np
from scipy import sparse
from scipy.special import softmax
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from improved_features import FeaturesV2, nb_ratios
from train_submit import make_model
import protocol_v59 as driver

STATE = {}
CONFIG = dict(model_family='BM25 word1-3/char2-5 LinearSVC + NB-SVM',
    pretrained=False, external_data=False, fit_data='training partitions only',
    k1=1.2, b=.75, idf='log(1+(N-df+.5)/(df+.5))',
    document_length='ngram count mass in fitted vocabulary', normalization='L2 each view',
    word_max_features=140000, char_max_features=180000,
    svm_C=.5, svm_weight='balanced', margin_mix=[.75,.25], temperature_multiplier=3.,
    combination='.85 * complete V11 baseline + .15 * BM25 SVM branch',
    class_weights=[1.,1.2,1.2,.9],
    gates=dict(cluster_min_delta=.003, exact_min_delta=.002, min_positive_folds=2,
        max_fold_drop=.005, min_novel_fake_f1_delta=0., bootstrap_q025_positive=True,
        min_dev_delta=.001, min_dev_without_exact_overlap_delta=.001),
    selection_note='New feature weighting, not a retune of V58 or past gates')

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class BM25:
    def __init__(self, **kwargs):
        self.vectorizer = CountVectorizer(lowercase=False, dtype=np.float32, **kwargs)
    def fit_transform(self, texts):
        guard()
        x = self.vectorizer.fit_transform(texts).tocsr()
        df = np.bincount(x.indices, minlength=x.shape[1])
        self.idf = np.log1p((len(texts)-df+.5)/(df+.5)).astype(np.float32)
        self.avg = max(float(x.sum(1).mean()), 1e-12)
        return self.weight(x)
    def weight(self, x):
        x = x.astype(np.float32, copy=True).tocsr()
        dl = np.asarray(x.sum(1)).ravel()
        denom = np.repeat(1.2*(.25+.75*dl/self.avg), np.diff(x.indptr))
        x.data = (x.data*2.2/(x.data+denom))*self.idf[x.indices]
        assert np.isfinite(x.data).all() and (x.data >= 0).all()
        return normalize(x, copy=False)
    def transform(self, texts): return self.weight(self.vectorizer.transform(texts))

class Predictor:
    classes_ = np.arange(4)
    def __init__(self, encoder, model, binary):
        self.encoder=encoder; self.model=model; self.binary=binary
    def predict_proba(self, frame):
        x,m = self.encoder.transform(frame)
        scores = .75*self.model.decision_function(sparse.hstack([x,m],format='csr'))
        for c,(ratio,model) in enumerate(self.binary):
            scores[:,c] += .25*model.decision_function(sparse.hstack([x.multiply(ratio),m],format='csr'))
        return softmax(3*scores,axis=1)

def fitted_head(train, groups, seed=102):
    guard()
    key=hashlib.sha256(json.dumps(dict(indices=train.index.tolist(),config=driver.CONFIG,
        smoke=STATE['smoke']),sort_keys=True).encode()).hexdigest()
    folder=STATE['out']/'partitions'/key;folder.mkdir(parents=True,exist_ok=True)
    path=folder/'predictor.joblib'
    if path.exists():
        saved=joblib.load(path);assert saved['key']==key
        return saved['model']
    encoder=FeaturesV2()
    encoder.word=BM25(ngram_range=(1,3),min_df=2,max_features=140000,token_pattern=r'(?u)\b\w+\b')
    encoder.char=BM25(analyzer='char',ngram_range=(2,5),min_df=3,max_features=180000)
    x,m=encoder.fit_transform(train);y=train.spam_label.to_numpy()
    print(json.dumps(dict(stage='bm25_features',rows=len(train),text_features=x.shape[1],meta_features=m.shape[1])),flush=True)
    model=make_model(dict(weight='balanced',C=.5),y)
    model.fit(sparse.hstack([x,m],format='csr'),y)
    assert np.max(model.n_iter_) < model.max_iter
    binary=[]
    for c,ratio in enumerate(nb_ratios(x,y)):
        one=LinearSVC(C=.5,class_weight='balanced',dual=True,max_iter=30000,random_state=102)
        one.fit(sparse.hstack([x.multiply(ratio),m],format='csr'),(y==c).astype(int))
        assert np.max(one.n_iter_) < one.max_iter
        binary.append((ratio,one))
    result=Predictor(encoder,model,binary)
    tmp=folder/'predictor.tmp.joblib';joblib.dump(dict(key=key,model=result),tmp);tmp.replace(path)
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True)
    p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    STATE.update(out=Path(a.output),smoke=a.smoke)
    driver.CONFIG={**CONFIG,'head_source_sha256':digest(__file__)}
    driver.main(a,fitted_head)
