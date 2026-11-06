"""Train-only classwise lexical selection plus original SVD128 CatBoost.

Unlike frequency-limited text dictionaries, allocate a fixed selection quota
to each class. Feature selection uses fitting labels only; transform ignores
labels. Every parameter is learned from the supplied training partition.
"""
import gc,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from latent_v6 import LatentFeatures,make_model
from experiment_v10 import guard,checked

CONFIG=dict(per_class_per_view=128,word_max_features=100000,char_max_features=120000,
    min_df=3,word_ngram=[1,3],char_ngram=[2,5],binary=True,
    trees=800,depth=5,lr=.06,l2=8,seed=102,new_latent_coefficient=.15)

class LexicalEncoder:
    def __init__(self):
        self.word=CountVectorizer(ngram_range=(1,3),min_df=3,
            max_features=100000,binary=True,dtype=np.float32)
        self.char=CountVectorizer(analyzer='char',ngram_range=(2,5),min_df=3,
            max_features=120000,binary=True,dtype=np.float32)
    def fit(self,train):
        guard();y=train.spam_label.to_numpy();assert np.array_equal(np.unique(y),[0,1,2,3])
        self.indices=[];self.vocabulary_sizes=[]
        for vectorizer in [self.word,self.char]:
            x=vectorizer.fit_transform(train.comment)
            selected=set()
            for label in range(4):
                scores=chi2(x,(y==label).astype(int))[0]
                scores=np.nan_to_num(scores,nan=-np.inf)
                selected.update(np.argsort(-scores,kind='stable')[:128].tolist())
            self.indices.append(np.array(sorted(selected),dtype=np.int64))
            self.vocabulary_sizes.append(x.shape[1])
        self.selected_columns=sum(len(i) for i in self.indices)
        return self
    def transform(self,df):
        out=np.hstack([v.transform(df.comment)[:,indices].toarray()
            for v,indices in zip([self.word,self.char],self.indices)]).astype(np.float32)
        assert out.shape==(len(df),self.selected_columns)
        assert np.isfinite(out).all() and np.isin(out,[0.,1.]).all()
        return out

def fit_predict(train,frames,folder,feature_id,status,cache_path=None,cache_id=None):
    guard();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if cache_path is not None:
        q=np.load(cache_path,allow_pickle=False);assert q['feature_id'].item()==cache_id
        cols=q['numeric_columns'][:575].tolist();assert cols==[f'f_{i}' for i in range(575)]
        all_frames=[]
        for prefix in ['train']+[f'frame{i}' for i in range(len(frames))]:
            x=pd.DataFrame(q[prefix+'_numeric'][:,:575],columns=cols)
            for j,c in enumerate(['category','rating']):x[c]=q[prefix+'_categories'][:,j]
            all_frames.append(x)
    else:
        status('fit_full_svd128_features')
        f=LatentFeatures();f.svd.set_params(n_components=128,random_state=102)
        all_frames=[f.transform(train,fit=True)]+[f.transform(d) for d in frames]
        del f;gc.collect()
    status('fit_train_only_classwise_lexical_selection')
    encoder=LexicalEncoder().fit(train)
    joblib.dump(encoder,folder/'lexical_selector.joblib',compress=3)
    augmented=[]
    for x,df in zip(all_frames,[train]+frames):
        assert len(x)==len(df)
        assert np.array_equal(x[['category','rating']].to_numpy(dtype=str),
            df[['category','rating']].astype(str).to_numpy())
        extra=encoder.transform(df)
        joined=pd.concat([x.reset_index(drop=True),pd.DataFrame(extra,
            columns=[f'chi2_{i}' for i in range(extra.shape[1])])],axis=1)
        assert joined.shape[1]==577+encoder.selected_columns
        augmented.append(joined)
    status('fit_catboost_selected_lexical',feature_columns=augmented[0].shape[1],
        selected_lexical_columns=encoder.selected_columns,
        original_vocabulary_sizes=encoder.vocabulary_sizes)
    m=make_model();m.fit(augmented[0],train.spam_label.to_numpy(),verbose=200)
    assert np.array_equal(m.classes_,[0,1,2,3])
    result=[checked(m.predict_proba(x),len(x)) for x in augmented[1:]]
    m.save_model(str(folder/'catboost.cbm'))
    (folder/'metadata.json').write_text(json.dumps(dict(config=CONFIG,feature_id=feature_id,
        selected_columns=encoder.selected_columns,vocabulary_sizes=encoder.vocabulary_sizes),indent=2))
    del m,encoder,all_frames,augmented;gc.collect();return result
