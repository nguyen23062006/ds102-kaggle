"""Recompute reference CV on Colab; never uploads local cached predictions."""
import gc,json,os,platform,traceback
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.model_selection import StratifiedGroupKFold
from train_submit import load_data,SEED,metrics
from improved_features import prepare
from train_submit_v2 import fit_predict,RECIPE
from catboost_v3 import make_cat,cat_features
from select_v2 import macro


def v3_scores(fit, frames):
    scores=fit_predict(fit,frames,RECIPE)
    model=make_cat('SqrtBalanced',iterations=500)
    model.fit(cat_features(fit),fit.spam_label.to_numpy(),verbose=False)
    return [.5*softmax(3*s,axis=1)+.5*model.predict_proba(cat_features(d)) for s,d in zip(scores,frames)]


def main():
    if platform.system()!='Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Hosted Colab only; no local fitting.')
    out=Path('/content/blend_experiment');out.mkdir(exist_ok=True)
    def status(**d):(out/'status.json').write_text(json.dumps(d,ensure_ascii=False,indent=2))
    try:
        data=load_data(Path('/content/ds102_data'))
        train,dev,test=[prepare(data[n]) for n in ['train','dev','test']]
        y=train.spam_label.to_numpy();p=np.zeros((len(y),4));fold_ids=np.full(len(y),-1)
        dirs=[Path('/content/kernel_experiment'),Path('/content/kernel_lowc')]
        best_dir=max(dirs,key=lambda d:max(json.loads((d/'cv_results.json').read_text())['oof_macro_f1']))
        kr=json.loads((best_dir/'cv_results.json').read_text());kz=np.load(best_dir/'oof.npz')
        assert np.array_equal(kz['y'],y)
        kp=kz['oof'][kr['chosen_index']]
        for f,(tr,va) in enumerate(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED).split(train,y,groups=train.comment)):
            status(stage='v3_reference_CV',fold=f+1)
            p[va]=v3_scores(train.iloc[tr],[train.iloc[va].drop(columns='spam_label')])[0]
            fold_ids[va]=f;gc.collect()
        results=[]
        for a in [0.,.05,.1,.2]:
            pred=((1-a)*p+a*np.eye(4)[kp]).argmax(1)
            results.append({'kernel_weight':a,'oof_macro_f1':macro(y,pred),
                            'fold_macro_f1':[macro(y[fold_ids==f],pred[fold_ids==f]) for f in range(3)]})
        np.savez_compressed(out/'oof.npz',v3_scores=p,y=y,folds=fold_ids,kernel_pred=kp)
        (out/'cv_results.json').write_text(json.dumps(results,indent=2))
        best=max(results,key=lambda d:d['oof_macro_f1']);base=results[0]['oof_macro_f1']
        if best['kernel_weight']==0 or best['oof_macro_f1'] < base+.002:
            status(stage='not_improved',results=results,kernel_cv=kr)
            return
        status(stage='fit_final_v3',results=results,chosen=best)
        dev_p,test_p=v3_scores(train,[dev.drop(columns='spam_label'),test]);gc.collect()
        status(stage='fit_final_kernel',results=results,chosen=best)
        import kernel_finalize_colab
        kernel_finalize_colab.main(cv_dir=str(best_dir),output_dir='/content/blend_kernel',allow_cv_lower_for_blend=True)
        kz=np.load('/content/blend_kernel/predictions.npz')
        a=best['kernel_weight']
        dev_pred=((1-a)*dev_p+a*np.eye(4)[kz['dev']]).argmax(1)
        test_pred=((1-a)*test_p+a*np.eye(4)[kz['test']]).argmax(1)
        sub=data['samplesubmission'].copy()
        by_id=dict(zip(test.id.astype(str),test_pred.tolist()))
        sub['spam_label']=sub.ID.astype(str).map(by_id)
        assert not sub.isna().any().any() and sub.spam_label.isin([0,1,2,3]).all()
        sub['spam_label']=sub.spam_label.astype(int)
        path=out/'submission_v4.csv';sub.to_csv(path,index=False)
        import pandas as pd
        pd.testing.assert_frame_equal(pd.read_csv(path),sub,check_dtype=False)
        unseen=~dev.comment.isin(set(train.comment)).to_numpy()
        report={'selection':results,'chosen':best,'kernel_cv':kr,
                'v3_dev':metrics(dev.spam_label.to_numpy(),dev_p.argmax(1)),
                'dev':metrics(dev.spam_label.to_numpy(),dev_pred),
                'dev_without_overlap':metrics(dev.spam_label.to_numpy()[unseen],dev_pred[unseen]),
                'rows_verified':len(sub),'pretrained':False,'fit_data':'train only','execution':'Google Colab CPU',
                'caveat':'CV used for model selection; dev reused in iterative development; new Kaggle score unknown'}
        (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        status(stage='complete',report=report)
    except BaseException as e:
        status(stage='error',error=str(e),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
