"""CatBoost text with an additional character trigram dictionary, no pretrained."""
import gc,json
from pathlib import Path
import numpy as np
from catboost_v3 import make_cat,cat_features
from cloud_checks import guard,checked

DICTIONARIES=[
 dict(dictionary_id='Word',gram_order='1',occurrence_lower_bound='3',max_dictionary_size='40000'),
 dict(dictionary_id='Bigram',gram_order='2',occurrence_lower_bound='3',max_dictionary_size='40000'),
 dict(dictionary_id='Char3',token_level_type='Letter',gram_order='3',occurrence_lower_bound='3',max_dictionary_size='40000')]
CONFIG=dict(iterations=500,depth=5,learning_rate=.1,l2=5,seed=102,
    dictionaries=DICTIONARIES,feature_calcers=['BoW:top_tokens_count=700','NaiveBayes','BM25'],
    new_cat_coefficient=.175,old_cat_coefficient=.175,threads=2)

def fit_predict(train,frames,folder,feature_id,status,cache_path=None,cache_id=None):
    guard();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    m=make_cat('SqrtBalanced',iterations=CONFIG['iterations'])
    m.set_params(thread_count=2,dictionaries=DICTIONARIES)
    status('fit_catboost_word_and_character_text',trees=CONFIG['iterations'])
    m.fit(cat_features(train),train.spam_label.to_numpy(),verbose=100)
    assert np.array_equal(m.classes_,[0,1,2,3])
    ps=[checked(m.predict_proba(cat_features(df)),len(df)) for df in frames]
    m.save_model(str(folder/'character_text.cbm'))
    (folder/'metadata.json').write_text(json.dumps(dict(config=CONFIG,feature_id=feature_id,
        actual_parameters=m.get_all_params()),indent=2))
    del m;gc.collect();return ps
