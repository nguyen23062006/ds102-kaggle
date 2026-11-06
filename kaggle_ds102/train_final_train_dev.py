"""Conditional final train+dev refit of the fixed best SVD128 recipe.

Do not execute with dev labels until competition/course rules are confirmed.
There is no held-out dev metric after merging dev. No automatic submission.
"""
import argparse,gc,hashlib,importlib.metadata,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from experiment_v10 import guard,checked
from experiment_v11 import fit_latent
from train_submit import load_data
from train_submit_v2 import fit_predict,RECIPE
from catboost_v3 import make_cat,cat_features
from improved_features import prepare


def main(data_dir,output_dir,rule_confirmation):
    guard()
    if not rule_confirmation or not rule_confirmation.strip():
        raise ValueError('Record explicit competition/course confirmation that dev labels may be used for final training.')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);start=time.time()
    source=Path(__file__).resolve().parent
    names=['train_final_train_dev.py','experiment_v10.py','experiment_v11.py','latent_v6.py',
        'experiment_v6.py','selection_v5.py','train_submit.py','train_submit_v2.py','catboost_v3.py',
        'improved_features.py','select_v2.py']
    def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest=dict(recipe=dict(latent_dimensions=128,trees=800,alpha=.3,weights=[1.,1.2,1.2,.9]),
        rule_confirmation=rule_confirmation,
        training_data='original train plus original dev; test excluded',
        source_sha256={n:digest(source/n) for n in names},
        data_sha256={p.name:digest(p) for p in Path(data_dir).glob('ds102_*.csv')},
        versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','scikit-learn','catboost']})
    mid=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    mp=out/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest,'Manifest changed'
    mp.write_text(json.dumps(manifest,indent=2))
    report=dict(pretrained=False,execution='hosted Colab CPU',rule_confirmation=rule_confirmation,
        qualification='Final refit uses dev as training. Earlier dev scores are historical; no held-out dev metric for this refit. Kaggle score unknown.',
        selection='Fixed existing SVD128 recipe; no hyperparameter or weight search after merging dev.')
    def status(stage,**kv):
        report.update(stage=stage,**kv);(out/'report.json').write_text(json.dumps(report,indent=2));print(stage,kv,flush=True)
    data=load_data(Path(data_dir));original_train=prepare(data['train']);dev=prepare(data['dev']);test=prepare(data['test'])
    train=pd.concat([original_train,dev],ignore_index=True)
    assert len(train)==len(original_train)+len(dev)
    status('training_ready',train_rows=len(original_train),dev_rows=len(dev),combined_rows=len(train),
        class_counts={str(k):int(v) for k,v in train.spam_label.value_counts().sort_index().items()},
        heldout_dev_metric=None,test_used_for_fitting=False)
    parts={}
    for name in ['svm','cat_text','latent128']:
        ck=out/f'{name}_test.npz'
        if ck.exists():
            z=np.load(ck,allow_pickle=False);assert z['manifest_id'].item()==mid;p=z['prediction']
        else:
            status('fit',component=name)
            if name=='svm':p=fit_predict(train,[test],RECIPE)[0]
            elif name=='cat_text':
                m=make_cat('SqrtBalanced',iterations=500);m.set_params(thread_count=2)
                m.fit(cat_features(train),train.spam_label.to_numpy(),verbose=100)
                p=m.predict_proba(cat_features(test));del m
            else:p=fit_latent(train,[test],128,status)[0]
            tmp=ck.with_suffix('.tmp.npz');np.savez_compressed(tmp,manifest_id=mid,prediction=p);tmp.replace(ck)
        if name!='svm':checked(p,len(test))
        else:assert p.shape==(len(test),4) and np.isfinite(p).all()
        parts[name]=p;gc.collect()
    base=.5*softmax(3*parts['svm'],axis=1)+.5*parts['cat_text']
    scores=checked(.7*base+.3*parts['latent128'],len(test))
    pred=(scores*np.array([1.,1.2,1.2,.9])).argmax(1)
    sub=data['samplesubmission'].copy();ids=sub.ID.copy()
    sub['spam_label']=sub.ID.astype(str).map(dict(zip(test.id.astype(str),pred))).astype(int)
    assert len(sub)==len(test) and sub.ID.is_unique and sub.ID.equals(ids) and sub.spam_label.isin(range(4)).all()
    path=out/'submission_train_dev.csv';sub.to_csv(path,index=False)
    pd.testing.assert_frame_equal(pd.read_csv(path),sub,check_dtype=False)
    status('candidate_ready_not_submitted',rows=len(sub),sha256=digest(path),elapsed_seconds=time.time()-start)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--rule-confirmation',required=True,help='Actual course/competition confirmation, not an assumed permission.')
    a=p.parse_args();main(a.data_dir,a.output_dir,a.rule_confirmation)
