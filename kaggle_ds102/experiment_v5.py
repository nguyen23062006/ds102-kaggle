"""Hosted Colab only: compare new sparse text views and richer CatBoost.

No pretrained weights, embeddings, outside data, or development labels for
selection. All vocabularies and scales fit inside grouped training folds.
"""
import gc
import json
import os
import platform
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import LinearSVC
from improved_features import prepare, dense_features, tokens
from train_submit import load_data, SEED, metrics
from select_v2 import macro


def hosted_guard():
    if platform.system() != 'Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Training requires hosted Google Colab; local fitting disabled.')


class TextViews:
    def __init__(self):
        self.word = TfidfVectorizer(ngram_range=(1, 3), min_df=2,
            max_features=160000, sublinear_tf=True, token_pattern=r'(?u)\b\w+\b',
            dtype=np.float32)
        self.char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6),
            min_df=3, max_features=220000, sublinear_tf=True, dtype=np.float32)
        self.ascii = TfidfVectorizer(analyzer='char_wb', strip_accents='unicode',
            ngram_range=(3, 5), min_df=4, max_features=100000,
            sublinear_tf=True, dtype=np.float32)
        self.product = TfidfVectorizer(ngram_range=(1, 2), min_df=3,
            max_features=25000, sublinear_tf=True, dtype=np.float32)
        self.category = OneHotEncoder(handle_unknown='ignore', dtype=np.float32)
        self.scaler = StandardScaler()

    def build(self, df, fit=False):
        def transform(model, values):
            return model.fit_transform(values) if fit else model.transform(values)
        text = sparse.hstack([transform(self.word, df.comment),
            transform(self.char, df.comment), .5*transform(self.ascii, df.comment)], format='csr')
        nums = dense_features(df)
        nums = self.scaler.fit_transform(nums) if fit else self.scaler.transform(nums)
        # Limit rare metadata outliers, retaining previously validated features.
        nums = np.clip(nums, -5, 5).astype(np.float32)
        meta = sparse.hstack([.3*transform(self.product, df.product_name),
            .3*transform(self.category, df[['category', 'rating']].astype(str)),
            .15*sparse.csr_matrix(nums)], format='csr')
        return sparse.hstack([text, meta], format='csr')


LINEAR_CONFIGS = [dict(C=.5, weight='balanced'), dict(C=1., weight='sqrt'),
                  dict(C=2., weight='sqrt')]


def linear_fit_predict(fit, frames, config):
    hosted_guard()
    f = TextViews(); x = f.build(fit, fit=True)
    yy = fit.spam_label.to_numpy()
    weight = config['weight']
    if weight == 'sqrt':
        counts = np.bincount(yy, minlength=4)
        weight = {c:float(np.sqrt(len(yy)/(4*counts[c]))) for c in range(4)}
    model = LinearSVC(C=config['C'], class_weight=weight, dual=True,
        max_iter=20000, random_state=SEED)
    model.fit(x, yy)
    if np.max(model.n_iter_) >= model.max_iter:
        raise RuntimeError('Linear SVM did not converge; do not select this run.')
    return [softmax(3*model.decision_function(f.build(d)), axis=1) for d in frames]


def rich_cat_features(df):
    from catboost_v3 import cat_features
    x = cat_features(df)
    # Product/category identities are ordinary supplied input features, with
    # CatBoost's ordered target statistics, never external labels.
    x['product_name'] = df.product_name.to_numpy()
    x['categories'] = df.categories.astype(str).to_numpy()
    n = np.maximum(df.num_review.astype(float).to_numpy(), 1)
    x['sold_per_review'] = np.log1p(df.num_sold.astype(float).to_numpy()/n)
    return x


def make_rich_cat():
    from catboost_v3 import make_cat
    m = make_cat('SqrtBalanced', iterations=900)
    m.set_params(depth=6, learning_rate=.07, l2_leaf_reg=8, thread_count=2,
        cat_features=['category', 'rating', 'product_name', 'categories'],
        feature_calcers=['BoW:top_tokens_count=2000', 'NaiveBayes', 'BM25'])
    return m


def main(data_dir='/content/ds102_data', output_dir='/content/ds102_v5_experiment',
         family='linear'):
    hosted_guard()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    def status(**values):
        (out/'status.json').write_text(json.dumps(values, ensure_ascii=False, indent=2))
    try:
        df = prepare(load_data(Path(data_dir))['train'])
        y = df.spam_label.to_numpy(); folds = np.full(len(y), -1)
        configs = LINEAR_CONFIGS if family == 'linear' else [{'model':'rich_cat', 'trees':500}, {'model':'rich_cat', 'trees':900}]
        oof = np.zeros((len(configs),len(y),4), dtype=np.float32)
        results = [[] for _ in configs]
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
        for fold,(tr,va) in enumerate(splitter.split(df,y,groups=df.comment)):
            folds[va] = fold
            train, valid = df.iloc[tr], df.iloc[va].drop(columns='spam_label')
            if family == 'linear':
                status(stage='features',fold=fold+1,results=results)
                f = TextViews(); xt=f.build(train,fit=True); xv=f.build(valid)
                for i,c in enumerate(configs):
                    status(stage='fit',fold=fold+1,config=c,results=results)
                    weight=c['weight']
                    if weight=='sqrt':
                        counts=np.bincount(y[tr],minlength=4)
                        weight={k:float(np.sqrt(len(tr)/(4*counts[k]))) for k in range(4)}
                    m=LinearSVC(C=c['C'],class_weight=weight,dual=True,
                        max_iter=20000,random_state=SEED).fit(xt,y[tr])
                    if np.max(m.n_iter_) >= m.max_iter:
                        raise RuntimeError('Linear SVM failed to converge')
                    oof[i,va]=softmax(3*m.decision_function(xv),axis=1)
                    results[i].append(macro(y[va],oof[i,va].argmax(1)))
                del f,xt,xv,m
            else:
                status(stage='fit',fold=fold+1,config='rich_cat_900',results=results)
                m=make_rich_cat(); m.fit(rich_cat_features(train),y[tr],verbose=300)
                for i,c in enumerate(configs):
                    oof[i,va]=m.predict_proba(rich_cat_features(valid),ntree_end=c['trees'])
                    results[i].append(macro(y[va],oof[i,va].argmax(1)))
                del m
            gc.collect()
            np.savez_compressed(out/'partial.npz',oof=oof,y=y,folds=folds)
            status(stage='fold_complete',fold=fold+1,results=results)
        report={'family':family,'configs':configs,'fold_scores':results,
            'oof_macro_f1':[macro(y,p.argmax(1)) for p in oof],
            'fit_data':'train only','pretrained':False,'execution':'hosted Colab CPU'}
        np.savez_compressed(out/'oof.npz',oof=oof,y=y,folds=folds)
        (out/'cv_results.json').write_text(json.dumps(report,indent=2))
        status(stage='complete',**report)
        return report
    except BaseException as exc:
        status(stage='error',error=str(exc),traceback=traceback.format_exc())
        raise
