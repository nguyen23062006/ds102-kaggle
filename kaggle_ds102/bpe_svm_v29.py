"""Train-only SentencePiece BPE + original character/metadata SVM and NB-SVM."""
import gc,io,json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.special import softmax
import sentencepiece as spm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from improved_features import FeaturesV2,dense_features,nb_ratios
from train_submit import make_model
from train_submit_v2 import RECIPE
from experiment_v10 import guard,checked

CONFIG=dict(vocab_size=8000,model_type='bpe',split_by_whitespace=False,
    character_coverage=1.,byte_fallback=True,char_cap=8192,svm_temperature=3.,
    new_svm_coefficient=.175,recipe=RECIPE,seed=102,sentencepiece_version='0.2.1')

class BPEFeatures:
    def __init__(self,folder):
        self.folder=Path(folder);self.base=FeaturesV2();self.sp=None
        self.base.word=TfidfVectorizer(tokenizer=str.split,token_pattern=None,
            ngram_range=(1,3),min_df=2,max_features=140000,
            sublinear_tf=True,lowercase=False,dtype=np.float32)

    def encode(self,texts):
        return [' '.join(self.sp.encode(t[:8192],out_type=str)) for t in texts]

    def fit_transform(self,df):
        guard();self.folder.mkdir(parents=True,exist_ok=True);buf=io.BytesIO()
        spm.SentencePieceTrainer.train(sentence_iterator=(t[:8192] for t in df.comment),
            model_writer=buf,model_type='bpe',vocab_size=8000,character_coverage=1.,
            byte_fallback=True,normalization_rule_name='identity',
            split_by_whitespace=False,hard_vocab_limit=False,num_threads=2,
            shuffle_input_sentence=False,input_sentence_size=0,max_sentence_length=65536,
            bos_id=-1,eos_id=-1,pad_id=-1,unk_id=0,minloglevel=1)
        proto=buf.getvalue();(self.folder/'tokenizer.model').write_bytes(proto)
        self.sp=spm.SentencePieceProcessor(model_proto=proto)
        encoded=self.encode(df.comment)
        text=sparse.hstack([self.base.word.fit_transform(encoded),self.base.char.fit_transform(df.comment)],format='csr')
        meta=sparse.hstack([self.base.product.fit_transform(df.product_name)*.2,
            self.base.category.fit_transform(df[['category','rating']].astype(str))*.2,
            sparse.csr_matrix(self.base.scale.fit_transform(dense_features(df)))*.15],format='csr')
        return text,meta

    def transform(self,df):
        encoded=self.encode(df.comment)
        text=sparse.hstack([self.base.word.transform(encoded),self.base.char.transform(df.comment)],format='csr')
        meta=sparse.hstack([self.base.product.transform(df.product_name)*.2,
            self.base.category.transform(df[['category','rating']].astype(str))*.2,
            sparse.csr_matrix(self.base.scale.transform(dense_features(df)))*.15],format='csr')
        return text,meta

def fit_predict(train,frames,folder,feature_id,status):
    guard();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    y=train.spam_label.to_numpy();f=BPEFeatures(folder)
    status('fit_bpe_and_tfidf')
    xt,mt=f.fit_transform(train);vv=[f.transform(d) for d in frames]
    result=[np.zeros((len(d),4)) for d in frames]
    ratios=nb_ratios(xt,y)
    status('fit_bpe_svm',text_features=xt.shape[1],metadata_features=mt.shape[1])
    m=make_model(RECIPE[0],y);m.fit(sparse.hstack([xt,mt],format='csr'),y)
    assert np.array_equal(m.classes_,[0,1,2,3]) and np.max(m.n_iter_)<m.max_iter
    for j,(xv,mv) in enumerate(vv):result[j]+=.75*m.decision_function(sparse.hstack([xv,mv],format='csr'))
    del m;gc.collect()
    for cls,r in enumerate(ratios):
        status('fit_bpe_nb_svm',class_index=cls)
        m=LinearSVC(C=.5,class_weight='balanced',dual=True,max_iter=30000,random_state=102)
        m.fit(sparse.hstack([xt.multiply(r),mt],format='csr'),(y==cls).astype(int))
        assert np.max(m.n_iter_)<m.max_iter
        for j,(xv,mv) in enumerate(vv):result[j][:,cls]+=.25*m.decision_function(sparse.hstack([xv.multiply(r),mv],format='csr'))
        del m;gc.collect()
    (folder/'metadata.json').write_text(json.dumps(dict(config=CONFIG,feature_id=feature_id,
        actual_vocabulary=f.sp.get_piece_size(),text_features=xt.shape[1],fit_rows=len(train)),indent=2))
    return [checked(softmax(3*p,axis=1),len(d)) for p,d in zip(result,frames)]
