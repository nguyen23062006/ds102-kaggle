"""DS102 v3: trung bình điểm v2 và CatBoost; tham số đã chốt bằng CV train."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import catboost
from scipy.special import softmax
from train_submit import load_data, find_data_dir, metrics, LABELS, SEED
from improved_features import prepare
from train_submit_v2 import fit_predict, RECIPE
from catboost_v3 import cat_features, make_cat


CAT_FRACTION = .5
TEMPERATURE_MULTIPLIER = 3.
REFERENCE_TRAIN_SHA = '15202b19b2dc2c6c89afeb209024bb0dc77ddf927de44381a96c24a961d823a5'


def main_v3(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--output-dir', default='output_v3')
    args = parser.parse_args(argv)
    data_dir = find_data_dir(args.data_dir)
    data = load_data(data_dir)
    columns = ['comment','product_name','description','category','categories','rating','num_sold','num_review']
    for name in ['train','dev','test']:
        if not set(columns).issubset(data[name].columns):
            raise ValueError(f'{name}: thiếu cột dữ liệu cho v3')
        for col in ['rating','num_sold','num_review']:
            data[name][col] = pd.to_numeric(data[name][col], errors='raise')
            if not np.isfinite(data[name][col]).all() or (data[name][col] < 0).any():
                raise ValueError(f'{name}.{col}: giá trị không hợp lệ')
    train,dev,test = [prepare(data[n]) for n in ['train','dev','test']]
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    y = train.spam_label.to_numpy()
    print('Huấn luyện SVM + NB-SVM trên train...',flush=True)
    base_dev,base_test = fit_predict(train,[dev.drop(columns='spam_label'),test],RECIPE)
    print('Huấn luyện CatBoost trên train...',flush=True)
    model = make_cat('SqrtBalanced',iterations=500)
    model.fit(cat_features(train),y,verbose=250)
    if not np.array_equal(model.classes_,np.asarray(LABELS)):
        raise ValueError('Sai thứ tự lớp CatBoost')
    cat_dev = model.predict_proba(cat_features(dev))
    cat_test = model.predict_proba(cat_features(test))
    dev_scores = (1-CAT_FRACTION)*softmax(TEMPERATURE_MULTIPLIER*base_dev,axis=1)+CAT_FRACTION*cat_dev
    test_scores = (1-CAT_FRACTION)*softmax(TEMPERATURE_MULTIPLIER*base_test,axis=1)+CAT_FRACTION*cat_test
    dev_pred,test_pred = dev_scores.argmax(1),test_scores.argmax(1)
    y_dev = dev.spam_label.to_numpy()
    dev_metrics = metrics(y_dev,dev_pred)
    baseline_metrics = metrics(y_dev,base_dev.argmax(1))
    unseen = ~dev.comment.isin(set(train.comment)).to_numpy()
    print(f"Dev v2: {baseline_metrics['macro_f1']:.5f}; dev v3: {dev_metrics['macro_f1']:.5f}",flush=True)
    by_id = pd.Series(test_pred,index=test.id.astype(str))
    submission = data['samplesubmission'].copy()
    submission['spam_label'] = submission.ID.astype(str).map(by_id)
    if submission.spam_label.isna().any() or not submission.spam_label.isin(LABELS).all():
        raise ValueError('Submission có nhãn không hợp lệ')
    submission['spam_label'] = submission.spam_label.astype(int)
    csv_path = output/'submission_v3.csv'
    submission.to_csv(csv_path,index=False)
    saved = pd.read_csv(csv_path)
    pd.testing.assert_frame_equal(saved,submission,check_dtype=False)
    if len(saved)!=len(test) or not saved.ID.equals(data['samplesubmission'].ID):
        raise ValueError('Sai số dòng/thứ tự ID')
    pd.DataFrame({'comment':data['dev'].comment,'true_label':y_dev,'v2_label':base_dev.argmax(1),
                  'v3_label':dev_pred,'comment_seen_in_train':~unseen}).to_csv(output/'dev_predictions_v3.csv',index=False)
    np.savez_compressed(output/'scores_v3.npz',v2_dev=base_dev,v2_test=base_test,
                        cat_dev=cat_dev,cat_test=cat_test,dev_scores=dev_scores,test_scores=test_scores)
    hashes={name:hashlib.sha256((data_dir/f'ds102_{name}.csv').read_bytes()).hexdigest() for name in ['train','dev','test']}
    reference = {'source':'Previously completed 3-fold grouped CV; not rerun by this training script',
                 'v2_oof_macro_f1':.7589770118322317,'v3_oof_macro_f1':.7651232845255038}
    if hashes['train']!=REFERENCE_TRAIN_SHA:
        reference=None
    report={
        'seed':SEED,'python':platform.python_version(),'sklearn':sklearn.__version__,'catboost':catboost.__version__,
        'v2_recipe':RECIPE,'catboost_params':model.get_params(),'cat_fraction':CAT_FRACTION,
        'svm_score_multiplier':TEMPERATURE_MULTIPLIER,'data_sha256':hashes,
        'fit_data':'train only; dev/test not used to fit models or features; no pretrained models',
        'reference_cv':reference,'dev':dev_metrics,'v2_dev_same_run':baseline_metrics,
        'dev_without_comment_overlap':metrics(y_dev[unseen],dev_pred[unseen]) if unseen.any() else None,
        'selection_caveat':'CV used for model selection; dev reused during iterative development; no new Kaggle score available',
        'submission_rows_verified':len(saved),'prediction_counts':{int(k):int(v) for k,v in saved.spam_label.value_counts().items()},
    }
    (output/'metrics_v3.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Đã kiểm tra và lưu {len(saved)} dòng: {csv_path}',flush=True)
    return report


if __name__=='__main__':
    main_v3()
