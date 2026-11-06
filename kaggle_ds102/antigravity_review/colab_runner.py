"""Hosted Google Colab Runner for Antigravity DS102 Review Experiments.

Enforces project constraints:
- Only executes full model training on hosted Google Colab (`guard()` enforced).
- No external data or pretrained weights.
- No merging dev into train.
- Outputs isolated artifacts in `antigravity_review_output/`.
- Does not submit to Kaggle.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np

# Ensure guard passes only on Linux / Colab
def guard():
    if platform.system() != 'Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError(
            'HOSTED GOOGLE COLAB ONLY: Local training on Mac is strictly disabled. '
            'Execute this runner inside Google Colab with hosted CPU/GPU runtime.'
        )


def compute_macro_f1(y_true, y_pred, n_classes=4):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = (2 * p * r) / max(p + r, 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s)), [float(x) for x in f1s]


def load_datasets(data_dir):
    import pandas as pd
    p = Path(data_dir)
    train = pd.read_csv(p / 'ds102_train.csv')
    dev = pd.read_csv(p / 'ds102_dev.csv')
    test = pd.read_csv(p / 'ds102_test.csv')
    sample = pd.read_csv(p / 'ds102_samplesubmission.csv')
    return train, dev, test, sample


def run_experiment_h1(train, dev, test, out_dir):
    """Run Hypothesis 1: SVD128 + Intent Features with CatBoost."""
    guard()
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.model_selection import StratifiedGroupKFold
    from catboost import CatBoostClassifier

    from h1_intent_features import IntentFeatureExtractor

    print("Running Hypothesis 1 (Intent & Deferral Features)...")
    out_dir.mkdir(parents=True, exist_ok=True)

    y = train['spam_label'].to_numpy()
    comments = train['comment'].fillna('').astype(str).tolist()

    # 1. Intent Features
    intent_extractor = IntentFeatureExtractor()
    intent_feats_train = intent_extractor.transform(train)
    intent_feats_dev = intent_extractor.transform(dev)
    intent_feats_test = intent_extractor.transform(test)

    # 2. Text TF-IDF + SVD128
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
    svd = TruncatedSVD(n_components=128, random_state=102)

    # 3-fold StratifiedGroupKFold on train.comment
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=102)
    groups = train['comment'].to_numpy()

    oof_preds = np.zeros(len(y), dtype=int)
    oof_probs = np.zeros((len(y), 4), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(train, y, groups)):
        print(f"--- Fitting Fold {fold + 1}/3 ---")
        X_tr_tfidf = tfidf.fit_transform([comments[i] for i in tr_idx])
        X_va_tfidf = tfidf.transform([comments[i] for i in va_idx])

        X_tr_svd = svd.fit_transform(X_tr_tfidf)
        X_va_svd = svd.transform(X_va_tfidf)

        # Concatenate SVD128 + Intent Features
        X_tr_combined = np.hstack([X_tr_svd, intent_feats_train[tr_idx]])
        X_va_combined = np.hstack([X_va_svd, intent_feats_train[va_idx]])

        clf = CatBoostClassifier(
            iterations=800,
            depth=5,
            learning_rate=0.06,
            l2_leaf_reg=8,
            loss_function='MultiClass',
            auto_class_weights='SqrtBalanced',
            random_seed=102,
            thread_count=2,
            verbose=200
        )
        clf.fit(X_tr_combined, y[tr_idx], eval_set=(X_va_combined, y[va_idx]), verbose=200)
        probs_va = clf.predict_proba(X_va_combined)
        oof_probs[va_idx] = probs_va
        oof_preds[va_idx] = np.argmax(probs_va, axis=1)

    macro_oof, class_f1s = compute_macro_f1(y, oof_preds)
    print(f"H1 OOF Macro F1: {macro_oof:.5f}")
    print(f"H1 Class F1s: {class_f1s}")

    metrics_res = {
        'hypothesis': 'H1_Intent_Features',
        'oof_macro_f1': macro_oof,
        'oof_class_f1s': class_f1s,
        'timestamp': time.time(),
    }
    with open(out_dir / 'metrics_h1.json', 'w') as f:
        json.dump(metrics_res, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Antigravity Review Colab Runner")
    parser.add_argument('--data-dir', required=True, help="Path to raw CSV directory")
    parser.add_argument('--output-dir', default='./antigravity_review_output', help="Output directory")
    parser.add_argument('--hypothesis', choices=['h1', 'h2', 'h3', 'all'], default='h1')
    args = parser.parse_args()

    # Guard against local execution
    guard()

    train, dev, test, sample = load_datasets(args.data_dir)
    out_dir = Path(args.output_dir)

    if args.hypothesis in ['h1', 'all']:
        run_experiment_h1(train, dev, test, out_dir)


if __name__ == '__main__':
    main()
