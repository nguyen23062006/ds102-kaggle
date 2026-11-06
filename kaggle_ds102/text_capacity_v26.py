"""CatBoost text with BoW top_tokens_count=2000, all other v3 settings fixed."""
import gc,json
from pathlib import Path
import numpy as np
from catboost_v3 import make_cat,cat_features
from experiment_v10 import guard,checked

CONFIG=dict(trees=500,depth=5,lr=.1,l2=5,seed=102,thread_count=2,
    feature_calcers=['BoW:top_tokens_count=2000','NaiveBayes','BM25'],
    class_weight='SqrtBalanced',new_cat_coefficient=.175,old_cat_coefficient=.175)

def fit_predict(train,frames,folder,feature_id,status):
    guard();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    status('fit_catboost_text2000')
    m=make_cat('SqrtBalanced',iterations=500)
    m.set_params(thread_count=2,feature_calcers=CONFIG['feature_calcers'])
    m.fit(cat_features(train),train.spam_label.to_numpy(),verbose=100)
    assert np.array_equal(m.classes_,[0,1,2,3])
    result=[checked(m.predict_proba(cat_features(d)),len(d)) for d in frames]
    m.save_model(str(folder/'catboost_text2000.cbm'))
    (folder/'metadata.json').write_text(json.dumps(dict(config=CONFIG,feature_id=feature_id,
        actual_params=m.get_all_params()),indent=2))
    del m;gc.collect();return result
