"""DS102 v2: đặc trưng quan hệ review/sản phẩm, SVM và calibration F1-macro."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC
from train_submit import load_data, find_data_dir, make_model, make_features, metrics, LABELS, SEED
from improved_features import prepare, FeaturesV2, nb_ratios
from select_v2 import macro, calibrate


# Cấu hình được chốt từ cross-validation trên train, trước đánh giá dev v2.
RECIPE = [
    {'kind': 'svm', 'C': .5, 'weight': 'balanced', 'ensemble_weight': .75},
    {'kind': 'nb', 'C': .5, 'weight': 'balanced', 'ensemble_weight': .25},
]


def fit_predict(fit_df, predict_frames, recipe):
    """Không nhận nhãn của các tập dự đoán."""
    y = fit_df.spam_label.to_numpy()
    features = FeaturesV2()
    Xt, Mt = features.fit_transform(fit_df)
    transformed = [features.transform(df) for df in predict_frames]
    result = [np.zeros((len(df), 4)) for df in predict_frames]
    ratios = nb_ratios(Xt, y) if any(c['kind'] == 'nb' for c in recipe) else None
    for config in recipe:
        if config['kind'] == 'svm':
            model = make_model(config, y).fit(sparse.hstack([Xt, Mt], format='csr'), y)
            if np.max(model.n_iter_) >= model.max_iter:
                raise RuntimeError('SVM chưa hội tụ; tăng max_iter trước khi xuất kết quả.')
            for j, (Xv, Mv) in enumerate(transformed):
                result[j] += config['ensemble_weight'] * model.decision_function(sparse.hstack([Xv, Mv], format='csr'))
        elif config['kind'] == 'nb':
            for cls, r in enumerate(ratios):
                model = LinearSVC(C=config['C'], class_weight='balanced', dual=True,
                                  max_iter=30000, random_state=SEED)
                model.fit(sparse.hstack([Xt.multiply(r), Mt], format='csr'), (y == cls).astype(int))
                if np.max(model.n_iter_) >= model.max_iter:
                    raise RuntimeError('NB-SVM chưa hội tụ; tăng max_iter trước khi xuất kết quả.')
                for j, (Xv, Mv) in enumerate(transformed):
                    result[j][:, cls] += config['ensemble_weight'] * model.decision_function(
                        sparse.hstack([Xv.multiply(r), Mv], format='csr'))
        else:
            raise ValueError(f"Model không được hỗ trợ: {config['kind']}")
    return result


def main_v2(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--output-dir', default='output_v2')
    parser.add_argument('--folds', type=int, default=3)
    args = parser.parse_args(argv)
    if args.folds < 2:
        parser.error('--folds phải >= 2')
    data_dir = find_data_dir(args.data_dir)
    data = load_data(data_dir)
    expected = {'comment', 'product_name', 'description', 'category', 'categories', 'rating', 'num_sold', 'num_review'}
    for name in ['train', 'dev', 'test']:
        missing = expected - set(data[name].columns)
        if missing:
            raise ValueError(f'{name} thiếu cột {missing}')
        for col in ['rating', 'num_sold', 'num_review']:
            values = pd.to_numeric(data[name][col], errors='raise')
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f'{name}.{col}: giá trị số không hợp lệ')
            data[name][col] = values
    train, dev, test = [prepare(data[n]) for n in ['train', 'dev', 'test']]
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    y = train.spam_label.to_numpy()
    oof = np.zeros((len(train), 4))
    fold_ids = np.full(len(train), -1)
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(splitter.split(train, y, groups=train.comment)):
        if set(y[tr]) != set(LABELS):
            raise ValueError('Fold không đủ 4 lớp; giảm số folds.')
        print(f'CV fold {fold+1}/{args.folds}', flush=True)
        oof[va] = fit_predict(train.iloc[tr], [train.iloc[va].drop(columns='spam_label')], RECIPE)[0]
        fold_ids[va] = fold
        print(f'  Raw F1-macro={macro(y[va], oof[va].argmax(axis=1)):.5f}', flush=True)
    cross_pred = np.zeros(len(train), dtype=int)
    for f in range(args.folds):
        fit = fold_ids != f
        bias_fold = calibrate(oof[fit], y[fit])
        cross_pred[~fit] = (oof[~fit] + bias_fold).argmax(axis=1)
    raw_score = macro(y, oof.argmax(axis=1))
    cross_score = macro(y, cross_pred)
    use_bias = cross_score > raw_score + .002
    bias = calibrate(oof, y) if use_bias else np.zeros(4)
    print(f'OOF raw={raw_score:.5f}; cross-calibrated={cross_score:.5f}; bias={bias.tolist()}', flush=True)
    print('Fit toàn bộ train và dự đoán dev/test...', flush=True)
    dev_scores, test_scores = fit_predict(train, [dev.drop(columns='spam_label'), test], RECIPE)
    dev_pred = (dev_scores + bias).argmax(axis=1)
    test_pred = (test_scores + bias).argmax(axis=1)
    dev_y = dev.spam_label.to_numpy()
    unique_dev = ~dev.comment.isin(set(train.comment)).to_numpy()
    dev_report = metrics(dev_y, dev_pred)
    print(f"Dev F1-macro={dev_report['macro_f1']:.5f}", flush=True)
    print(classification_report(dev_y, dev_pred, labels=LABELS, digits=4, zero_division=0), flush=True)
    by_id = pd.Series(test_pred, index=test.id.astype(str))
    submission = data['samplesubmission'].copy()
    submission['spam_label'] = submission.ID.astype(str).map(by_id)
    if submission.spam_label.isna().any() or not submission.spam_label.isin(LABELS).all():
        raise ValueError('Nhãn submission không hợp lệ')
    submission['spam_label'] = submission.spam_label.astype(int)
    path = output/'submission_v2.csv'
    submission.to_csv(path, index=False)
    saved = pd.read_csv(path)
    pd.testing.assert_frame_equal(saved, submission, check_dtype=False)
    if len(saved) != len(test) or not saved.ID.equals(data['samplesubmission'].ID):
        raise ValueError('Sai số dòng hoặc thứ tự ID')
    pd.DataFrame({'comment': data['dev'].comment, 'true_label': dev_y,
                  'predicted_label': dev_pred, 'comment_seen_in_train': ~unique_dev}).to_csv(output/'dev_predictions_v2.csv', index=False)
    np.savez_compressed(output/'scores_v2.npz', oof=oof, y=y, folds=fold_ids,
                        dev_scores=dev_scores, test_scores=test_scores, bias=bias)
    report = {
        'recipe': RECIPE, 'seed': SEED, 'folds': args.folds, 'python': platform.python_version(),
        'sklearn': sklearn.__version__, 'data_sha256': {name: hashlib.sha256((data_dir/file).read_bytes()).hexdigest()
            for name, file in [('train','ds102_train.csv'), ('dev','ds102_dev.csv'), ('test','ds102_test.csv')]},
        'fit_data': 'train only; no dev or test used to fit features, models, or calibration',
        'oof_raw_macro_f1': raw_score, 'oof_cross_calibration_macro_f1': cross_score,
        'calibration_bias': bias.tolist(), 'dev': dev_report,
        'dev_without_comment_overlap': metrics(dev_y[unique_dev], dev_pred[unique_dev]) if unique_dev.any() else None,
        'selection_caveat': 'CV estimates were used for model selection; dev was previously used to inspect baseline errors.',
        'submission_rows_verified': len(saved),
        'prediction_counts': {int(k): int(v) for k, v in saved.spam_label.value_counts().items()},
    }
    (output/'metrics_v2.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Đã kiểm tra và lưu {len(saved)} dòng: {path}', flush=True)
    return report


if __name__ == '__main__':
    main_v2()
