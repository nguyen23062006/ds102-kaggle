"""Train fastText skipgram from random initialization on fitting text only.

Corpus: training reviews plus unique training title/description documents.
No pretrained vectors, input model, outside text, dev/test fitting or label
strings in the corpus. The CatBoost head learns supervised fitting labels.
"""
import argparse,hashlib,json,os,platform,re,subprocess,sys,time
from pathlib import Path

def guard():
    if platform.system()!='Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Hosted Google Colab only; local fitting disabled.')

if __name__=='__main__':guard()
sys.path.insert(0,'/content/ds102_v53/source')
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import normalize
from improved_features import dense_features,tokens
import protocol_v56 as driver

ROOT=Path('/content/ds102_v56')
BINARY=ROOT/'upstream/fastText-0.9.2/fasttext'
CONFIG=dict(model_family='Own-training-text skipgram32 review/catalogue relations + CatBoost',
    fasttext_version='0.9.2',initialization='fresh random; no input model or pretrainedVectors',
    corpus='fitting reviews plus unique fitting title/description pairs; no labels in text',
    dim=32,epoch=5,lr=.05,ws=5,neg=5,minCount=2,minn=3,maxn=6,bucket=50000,
    wordNgrams=1,thread=1,seed=102,description_char_cap=4096,review_token_cap=512,
    trees=800,depth=5,learning_rate=.06,l2_leaf_reg=8,cat_weight='SqrtBalanced',
    candidate_alpha=.15,combination='0.7*v3 + 0.15*original_SVD128 + 0.15*new_head',
    cluster_split_seed=2053,cluster_threshold=.7,folds=3,class_weights=[1.,1.2,1.2,.9],
    gates=dict(cluster_min_delta=.003,exact_min_delta=.002,min_positive_folds=2,
        max_fold_drop=.005,min_novel_fake_f1_delta=0.,bootstrap_q025_positive=True,
        min_dev_delta=.001,min_dev_without_exact_overlap_delta=.001),
    pretrained=False,external_data=False,fit_data='training partitions only')
STATE={}

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def lexical(text,cap=None):
    words=[w for w in tokens(str(text)) if not w.startswith('__label__')]
    if cap is not None:words=words[:cap]
    return ' '.join(words) or 'emptyreview'

def texts(frame):
    review=[lexical(t,512) for t in frame.comment]
    title=[lexical(t) for t in frame.product_name]
    description=[lexical(re.sub(r'<[^>]+>',' ',str(t))[:4096]) for t in frame.description]
    return review,title,description

class TextEncoder:
    def __init__(self,folder):self.folder=Path(folder);self.vector_calls=0

    def fit(self,train):
        guard();self.folder.mkdir(parents=True,exist_ok=True)
        reviews,titles,descriptions=texts(train)
        catalogue=list(dict.fromkeys(zip(titles,descriptions)))
        corpus_lines=reviews+[t+' '+d for t,d in catalogue]
        corpus='\n'.join(corpus_lines)+'\n'
        self.corpus_sha=hashlib.sha256(corpus.encode()).hexdigest()
        path=self.folder/'corpus_train_only.txt';marker=self.folder/'embedding_complete.json'
        self.model_path=self.folder/'skipgram.bin'
        if marker.exists():
            old=json.loads(marker.read_text())
            assert old['corpus_sha256']==self.corpus_sha
            assert old['binary_sha256']==digest(BINARY)
            assert old['model_sha256']==digest(self.model_path)
            return self
        path.write_text(corpus)
        epochs=1 if STATE['smoke'] else 5
        command=[str(BINARY),'skipgram','-input',str(path),'-output',str(self.folder/'skipgram'),
                 '-dim','32','-epoch',str(epochs),'-lr','0.05','-ws','5','-neg','5',
                 '-minCount','2','-minn','3','-maxn','6','-bucket','50000',
                 '-wordNgrams','1','-thread','1','-seed','102','-verbose','2']
        assert '-pretrainedVectors' not in command
        stats=dict(stage='fit_skipgram',training_rows=len(train),unique_catalogue_documents=len(catalogue),
            corpus_lines=len(corpus_lines),corpus_tokens=sum(len(t.split()) for t in corpus_lines),
            corpus_sha256=self.corpus_sha,epochs=epochs)
        print(json.dumps(stats),flush=True)
        start=time.time()
        with (self.folder/'skipgram_training.log').open('w') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        assert self.model_path.exists()
        stats.update(stage='skipgram_complete',seconds=time.time()-start,command=command,
            binary_sha256=digest(BINARY),model_sha256=digest(self.model_path),
            initialization='random; trained on this fitting partition only')
        marker.write_text(json.dumps(stats,indent=2))
        print(json.dumps({k:v for k,v in stats.items() if k not in ['command']}),flush=True)
        return self

    def vectors(self,values):
        # Only aggregate vectors leave the CLI; no review text is printed.
        result=subprocess.run([str(BINARY),'print-sentence-vectors',str(self.model_path)],
            input='\n'.join(values)+'\n',capture_output=True,text=True,check=True)
        lines=result.stdout.splitlines()
        assert len(lines)==len(values),'fastText changed row count'
        a=np.asarray([np.fromstring(line,sep=' ',dtype=np.float32) for line in lines])
        assert a.shape==(len(values),32) and np.isfinite(a).all()
        return a

    def transform(self,frame):
        a,b,c=[self.vectors(t) for t in texts(frame)]
        norms=np.column_stack([np.linalg.norm(z,axis=1) for z in [a,b,c]])
        a,b,c=[normalize(z) for z in [a,b,c]]
        scalar=np.column_stack([(a*b).sum(1),(a*c).sum(1),(b*c).sum(1)])
        values=np.hstack([a,b,c,a*b,np.abs(a-b),a*c,np.abs(a-c),scalar,norms,dense_features(frame)])
        assert values.shape==(len(frame),292) and np.isfinite(values).all()
        result=pd.DataFrame(values.astype(np.float32),columns=[f'context_{i}' for i in range(values.shape[1])])
        result['category']=frame.category.to_numpy(dtype=str)
        result['rating']=frame.rating.to_numpy(dtype=str)
        return result

class Predictor:
    def __init__(self,encoder,model):self.encoder=encoder;self.model=model;self.classes_=model.classes_
    def predict_proba(self,frame):return self.model.predict_proba(self.encoder.transform(frame))

def fitted_head(train,groups,seed=102):
    guard()
    key=hashlib.sha256(json.dumps(dict(indices=train.index.tolist(),config=driver.CONFIG,
        smoke=STATE['smoke']),sort_keys=True).encode()).hexdigest()
    folder=STATE['out']/'partitions'/key;folder.mkdir(parents=True,exist_ok=True)
    mp=folder/'partition_manifest.json';pm=dict(key=key,indices=train.index.tolist(),config=driver.CONFIG,smoke=STATE['smoke'])
    if mp.exists():assert json.loads(mp.read_text())==pm
    mp.write_text(json.dumps(pm,indent=2))
    encoder=TextEncoder(folder).fit(train)
    model_path=folder/'classifier.cbm';feature_path=folder/'training_features.joblib'
    if feature_path.exists():
        cache=joblib.load(feature_path);assert cache['key']==key;features=cache['features']
    else:
        features=encoder.transform(train);joblib.dump(dict(key=key,features=features),feature_path)
    model=CatBoostClassifier(iterations=20 if STATE['smoke'] else 800,depth=5,
        learning_rate=.06,l2_leaf_reg=8,loss_function='MultiClass',auto_class_weights='SqrtBalanced',
        cat_features=['category','rating'],max_ctr_complexity=1,random_seed=102,
        thread_count=2,allow_writing_files=False,verbose=200)
    if model_path.exists():model.load_model(str(model_path))
    else:
        print(json.dumps(dict(stage='fit_context_catboost',rows=len(train),features=features.shape[1],trees=20 if STATE['smoke'] else 800)),flush=True)
        model.fit(features,train.spam_label.to_numpy(),verbose=200)
        assert np.array_equal(model.classes_,[0,1,2,3])
        tmp=folder/'classifier.tmp.cbm';model.save_model(str(tmp));tmp.replace(model_path)
    return Predictor(encoder,model)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    STATE.update(out=Path(args.output),smoke=args.smoke)
    build=json.loads((ROOT/'fasttext_build_manifest.json').read_text())
    assert digest(BINARY)==build['binary_sha256']
    driver.CONFIG={**CONFIG,'head_source_sha256':digest(__file__),
        'fasttext_build':{k:build[k] for k in ['version','source_url','source_archive_sha256','binary_sha256','compiler','build_flags','license']}}
    driver.main(args,fitted_head)
