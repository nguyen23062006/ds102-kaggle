"""Train-only latent TF-IDF review/product relations + numeric CatBoost.

TruncatedSVD is fitted from scratch inside each training fold. No pretrained
embeddings, language models or outside corpora are used.
"""
import gc,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedGroupKFold
from catboost import CatBoostClassifier
from improved_features import FeaturesV2,dense_features,prepare
from experiment_v6 import hosted_guard
from selection_v5 import cross_weight
from select_v2 import macro
from train_submit import load_data,SEED


class LatentFeatures:
    def __init__(self):
        self.base=FeaturesV2()
        self.svd=TruncatedSVD(n_components=64,n_iter=5,random_state=SEED)

    def transform(self,df,fit=False):
        if fit:
            review=sparse.hstack([self.base.word.fit_transform(df.comment),
                                  self.base.char.fit_transform(df.comment)],format='csr')
            review=normalize(review)
            z=self.svd.fit_transform(review)
        else:
            review=sparse.hstack([self.base.word.transform(df.comment),
                                  self.base.char.transform(df.comment)],format='csr')
            review=normalize(review)
            z=self.svd.transform(review)
        title=normalize(sparse.hstack([self.base.word.transform(df.product_name),
                                      self.base.char.transform(df.product_name)],format='csr'))
        t=self.svd.transform(title)
        cosine=np.asarray(review.multiply(title).sum(axis=1)).ravel()
        x=np.hstack([z,t,z*t,np.abs(z-t),cosine[:,None],dense_features(df)])
        x=pd.DataFrame(x.astype(np.float32),columns=[f'f_{i}' for i in range(x.shape[1])])
        x['category']=df.category.astype(str).to_numpy()
        x['rating']=df.rating.astype(str).to_numpy()
        return x


def make_model():
    return CatBoostClassifier(iterations=800,depth=5,learning_rate=.06,l2_leaf_reg=8,
        loss_function='MultiClass',auto_class_weights='SqrtBalanced',
        cat_features=['category','rating'],max_ctr_complexity=1,
        random_seed=SEED,thread_count=2,allow_writing_files=False,verbose=False)


def main(data_dir='/content/ds102_data',output_dir='/content/ds102_v6_latent',reference_path='/content/colab_reference.npz'):
    hosted_guard()
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    def status(stage,**values):
        (out/'status.json').write_text(json.dumps(dict(stage=stage,time=time.time(),**values),indent=2))
    try:
        df=prepare(load_data(Path(data_dir))['train']);y=df.spam_label.to_numpy()
        ref=np.load(reference_path);assert np.array_equal(y,ref['y'])
        trees=[400,800];oof=np.zeros((2,len(y),4),dtype=np.float32)
        folds=np.full(len(y),-1);results=[[],[]]
        split=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED)
        for f,(tr,va) in enumerate(split.split(df,y,groups=df.comment)):
            folds[va]=f;status('fit_svd',fold=f+1,results=results)
            features=LatentFeatures();xt=features.transform(df.iloc[tr],fit=True)
            xv=features.transform(df.iloc[va].drop(columns='spam_label'))
            status('fit_catboost',fold=f+1,results=results)
            m=make_model();m.fit(xt,y[tr],verbose=False)
            for i,n in enumerate(trees):
                oof[i,va]=m.predict_proba(xv,ntree_end=n)
                results[i].append(macro(y[va],oof[i,va].argmax(1)))
            del features,m,xt,xv;gc.collect()
            np.savez_compressed(out/'partial.npz',oof=oof,y=y,folds=folds)
        assert np.array_equal(folds,ref['folds'])
        rows=[]
        for i,n in enumerate(trees):
            for alpha in [.15,.3,.5]:
                p=(1-alpha)*ref['oof']+alpha*oof[i]
                row=cross_weight(p,y,folds)
                raw=[macro(y[folds==f],p[folds==f].argmax(1)) for f in np.unique(folds)]
                row['fold_macro_f1']=[a+b for a,b in zip(raw,row['fold_deltas'])] if row['use_weights'] else raw
                row.update(name=f'latent_cat_{n}_{alpha}',recipe={'kind':'latent','trees':n,'alpha':alpha})
                rows.append(row)
        rows.sort(key=lambda r:r['selection_score'],reverse=True)
        np.savez_compressed(out/'oof.npz',oof=oof,y=y,folds=folds)
        (out/'selection.json').write_text(json.dumps(rows,indent=2))
        report={'best':rows[0],'fold_scores':results,'pretrained':False,'fit_data':'train only',
                'representation':'TF-IDF and 64D SVD fitted on each training fold; review/title relations',
                'caveat':'Repeated OOF selection; weighting checks not fully nested CV.'}
        (out/'cv_results.json').write_text(json.dumps(report,indent=2));status('complete',**report)
        return report
    except BaseException as exc:
        status('error',error=str(exc),traceback=traceback.format_exc());raise
