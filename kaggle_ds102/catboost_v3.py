"""CatBoost text + thông tin review/sản phẩm, không dùng mô hình pretrained."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from improved_features import dense_features, tokens
from train_submit import SEED


def cat_features(df):
    a = dense_features(df)
    x = pd.DataFrame(a, columns=[f'stat_{i}' for i in range(a.shape[1])])
    x['comment'] = [' '.join(tokens(t)) or 'emptyreview' for t in df.comment]
    x['category'] = df.category.astype(str).to_numpy()
    x['rating'] = df.rating.astype(str).to_numpy()
    return x


def make_cat(weight='SqrtBalanced', iterations=500):
    return CatBoostClassifier(
        iterations=iterations, depth=5, learning_rate=.1, l2_leaf_reg=5,
        loss_function='MultiClass', auto_class_weights=weight,
        thread_count=4, random_seed=SEED, allow_writing_files=False,
        text_features=['comment'], cat_features=['category', 'rating'],
        tokenizers=[{'tokenizer_id':'Space', 'separator_type':'ByDelimiter', 'delimiter':' '}],
        dictionaries=[{'dictionary_id':'Word', 'gram_order':'1', 'occurrence_lower_bound':'3', 'max_dictionary_size':'40000'},
                      {'dictionary_id':'Bigram', 'gram_order':'2', 'occurrence_lower_bound':'3', 'max_dictionary_size':'40000'}],
        feature_calcers=['BoW:top_tokens_count=700', 'NaiveBayes', 'BM25'],
        verbose=False,
    )
