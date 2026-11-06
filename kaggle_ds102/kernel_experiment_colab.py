"""Nonlinear SVM experiment. Hosted Colab only, no pretrained models.

Train/development data remain separate. Run main from a notebook thread to
allow progress checks through MCP without keeping one tool call open.
"""
import gc
import json
import os
import platform
import traceback
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from train_submit import load_data, SEED, metrics
from improved_features import prepare, FeaturesV2
from select_v2 import macro


def kernel(text_a, meta_a, text_b, meta_b):
    k = (text_a @ text_b.T).toarray()
    k += 1
    np.square(k, out=k)
    k += .5 * rbf_kernel(meta_a, meta_b, gamma=.3)
    k /= 4.5
    return np.ascontiguousarray(k, dtype=np.float64)


def features(fit, validation):
    f = FeaturesV2()
    tx, mx = f.fit_transform(fit)
    tv, mv = f.transform(validation)
    # Limit rare metadata outliers; no validation data used to fit scaling.
    mx.data = np.clip(mx.data, -.75, .75)
    mv.data = np.clip(mv.data, -.75, .75)
    return normalize(tx), mx, normalize(tv), mv


def main(data_dir='/content/ds102_data', output_dir='/content/kernel_experiment', configs_override=None):
    if platform.system() != 'Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Run on hosted Google Colab only; local training is disabled.')
    output = Path(output_dir)
    output.mkdir(exist_ok=True)
    def status(**values):
        (output/'status.json').write_text(json.dumps(values, ensure_ascii=False, indent=2))
    try:
        data = load_data(Path(data_dir))
        train = prepare(data['train'])
        y = train.spam_label.to_numpy()
        configs = configs_override or [{'C':1.,'weight':'balanced'}, {'C':4.,'weight':'balanced'},
                                      {'C':1.,'weight':'sqrt_balanced'}]
        oof = np.full((len(configs),len(train)),-1,dtype=int)
        fold_results = [[] for _ in configs]
        for fold,(tr,va) in enumerate(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED).split(train,y,groups=train.comment)):
            status(stage='features',fold=fold+1,results=fold_results)
            xt,mt,xv,mv = features(train.iloc[tr],train.iloc[va].drop(columns='spam_label'))
            status(stage='kernel',fold=fold+1,results=fold_results)
            kt = kernel(xt,mt,xt,mt)
            kv = kernel(xv,mv,xt,mt)
            for i,cfg in enumerate(configs):
                status(stage='fit',fold=fold+1,config=cfg,results=fold_results)
                weight = cfg['weight']
                if weight=='sqrt_balanced':
                    counts=np.bincount(y[tr],minlength=4)
                    weight={c:float(np.sqrt(len(tr)/(4*counts[c]))) for c in range(4)}
                model=SVC(kernel='precomputed',C=cfg['C'],class_weight=weight,
                          cache_size=512,random_state=SEED)
                model.fit(kt,y[tr])
                oof[i,va]=model.predict(kv)
                fold_results[i].append(macro(y[va],oof[i,va]))
                del model
            del kt,kv,xt,mt,xv,mv
            gc.collect()
            np.savez_compressed(output/'partial.npz',oof=oof,y=y)
        scores=[macro(y,p) for p in oof]
        chosen=int(np.argmax(scores))
        report={'configs':configs,'fold_scores':fold_results,'oof_macro_f1':scores,
                'chosen_index':chosen,'fit_data':'train only','pretrained':False}
        (output/'cv_results.json').write_text(json.dumps(report,indent=2))
        np.savez_compressed(output/'oof.npz',oof=oof,y=y)
        status(stage='complete',**report)
    except BaseException as exc:
        status(stage='error',error=str(exc),traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    main()
