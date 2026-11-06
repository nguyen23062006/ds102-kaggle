"""Fit the train-CV-selected nonlinear model on hosted Colab only."""
import gc
import hashlib
import json
import os
import platform
from pathlib import Path
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from train_submit import load_data, metrics, SEED
from improved_features import prepare, FeaturesV2
from kernel_experiment_colab import kernel


def main(data_dir='/content/ds102_data',cv_dir='/content/kernel_experiment',output_dir='/content/kernel_final',allow_cv_lower_for_blend=False):
    if platform.system()!='Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Hosted Colab required; local training disabled.')
    out=Path(output_dir);out.mkdir(exist_ok=True)
    def status(**values):
        (out/'status.json').write_text(json.dumps(values,ensure_ascii=False,indent=2))
    try:
        report=json.loads((Path(cv_dir)/'cv_results.json').read_text())
        best=report['chosen_index']; cfg=report['configs'][best]
        if report['oof_macro_f1'][best] <= .7651232845255038 and not allow_cv_lower_for_blend:
            status(stage='not_improved',cv=report,reference_v3_oof=.7651232845255038)
            return
        data=load_data(Path(data_dir))
        train,dev,test=[prepare(data[n]) for n in ['train','dev','test']]
        y=train.spam_label.to_numpy()
        f=FeaturesV2()
        status(stage='features',chosen_config=cfg)
        tx,mx=f.fit_transform(train);tx=normalize(tx)
        mx.data=np.clip(mx.data,-.75,.75)
        weight=cfg['weight']
        if weight=='sqrt_balanced':
            counts=np.bincount(y,minlength=4)
            weight={c:float(np.sqrt(len(y)/(4*counts[c]))) for c in range(4)}
        status(stage='full_kernel',chosen_config=cfg)
        k=kernel(tx,mx,tx,mx)
        status(stage='fit',chosen_config=cfg)
        model=SVC(C=cfg['C'],kernel='precomputed',class_weight=weight,cache_size=512,random_state=SEED)
        model.fit(k,y)
        del k;gc.collect()
        def predict(frame):
            result=[]
            for start in range(0,len(frame),512):
                a,b=f.transform(frame.iloc[start:start+512])
                a=normalize(a);b.data=np.clip(b.data,-.75,.75)
                kk=kernel(a,b,tx,mx)
                result.extend(model.predict(kk).tolist())
            return np.asarray(result,dtype=int)
        status(stage='predict_dev',chosen_config=cfg)
        dev_pred=predict(dev)
        status(stage='predict_test',chosen_config=cfg)
        test_pred=predict(test)
        submission=data['samplesubmission'].copy()
        by_id=dict(zip(test.id.astype(str),test_pred.tolist()))
        submission['spam_label']=submission.ID.astype(str).map(by_id)
        assert not submission.isna().any().any()
        assert submission.spam_label.isin([0,1,2,3]).all()
        submission['spam_label']=submission.spam_label.astype(int)
        path=out/'submission_kernel.csv';submission.to_csv(path,index=False)
        import pandas as pd
        saved=pd.read_csv(path)
        np.savez_compressed(out/'predictions.npz',dev=dev_pred,test=test_pred)
        pd.testing.assert_frame_equal(saved,submission,check_dtype=False)
        assert len(saved)==len(test) and saved.ID.equals(data['samplesubmission'].ID)
        unseen=~dev.comment.isin(set(train.comment)).to_numpy()
        result={'cv':report,'chosen_config':cfg,'dev':metrics(dev.spam_label.to_numpy(),dev_pred),
                'dev_without_comment_overlap':metrics(dev.spam_label.to_numpy()[unseen],dev_pred[unseen]),
                'rows_verified':len(saved),'pretrained':False,'execution':'hosted Google Colab CPU',
                'fit_data':'train only; dev/test not used to fit features/model or choose parameters',
                'data_sha256':{n:hashlib.sha256((Path(data_dir)/f'ds102_{n}.csv').read_bytes()).hexdigest() for n in ['train','dev','test']}}
        (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        status(stage='complete',result=result)
    except BaseException as exc:
        import traceback
        status(stage='error',error=str(exc),traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    main()
