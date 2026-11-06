"""Independent Diagnostic Tool for Kaggle DS102 Project.

Scope: READ-ONLY analysis on cached OOF arrays, test predictions, and CSV files.
Runs locally on Mac using standard library + numpy.
NO MODEL FITTING OR TRAINING IS PERFORMED IN THIS SCRIPT.
"""

import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
DATA_DIR = Path('/Users/ngien2306/Downloads/ds-102-r-11-cnvn')


def calculate_macro_f1(y_true, y_pred, n_classes=4):
    """Compute Macro F1 and per-class metrics without scikit-learn."""
    per_class = {}
    f1_list = []
    for c in range(n_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = (2 * p * r) / max(p + r, 1e-12)
        f1_list.append(f1)
        per_class[str(c)] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': round(p, 5),
            'recall': round(r, 5),
            'f1': round(f1, 5),
            'support': int(np.sum(y_true == c)),
        }
    return float(np.mean(f1_list)), per_class


def run_sensitivity_simulation():
    """Simulate the sensitivity of Macro F1 on a public test subset of ~1,000 samples."""
    # Approximate class distribution in public test (assumed 1,000 samples)
    # Class 0: ~740, Class 1: ~15, Class 2: ~75, Class 3: ~170
    sim_results = {}
    
    # Baseline scenario:
    # Class 1: 15 true, model predicts 10 TP, 3 FP, 5 FN -> F1 = 20 / (20 + 3 + 5) = 20/28 = 0.714
    # Class 2: 75 true, model predicts 45 TP, 15 FP, 30 FN -> F1 = 90 / (90 + 15 + 30) = 90/135 = 0.667
    # If 2 samples of Class 1 flip to False Negative (e.g. model gets more conservative):
    # Class 1: 8 TP, 3 FP, 7 FN -> F1 = 16 / (16 + 3 + 7) = 16/26 = 0.615 (Drop of 0.099 in F1_1!)
    # Drop in Macro F1 = 0.099 / 4 = -0.0248!
    
    sim_results['class1_sensitivity_note'] = (
        "With only ~15 Class 1 items on Public LB, losing just 1 True Positive "
        "drops Class 1 F1 by ~0.05, causing an immediate ~0.0125 drop in Macro F1."
    )
    sim_results['class2_sensitivity_note'] = (
        "With ~75 Class 2 items on Public LB, converting 2 True Positives into Class 0 "
        "drops Class 2 F1 by ~0.02, causing a ~0.005 drop in Macro F1."
    )
    return sim_results


def analyze_test_flips():
    """Analyze the exact text of predictions flipped between V11, V27, and V52."""
    test_csv = DATA_DIR / 'ds102_test.csv'
    if not test_csv.exists():
        return {'status': 'data_not_found'}

    with open(test_csv, encoding='utf-8') as f:
        test_data = {r['id']: r for r in csv.DictReader(f)}

    def read_sub(p):
        path = WORKSPACE / p
        if not path.exists():
            return None
        with open(path, encoding='utf-8') as f:
            return {r['ID']: int(r['spam_label']) for r in csv.DictReader(f)}

    v11 = read_sub('output_v11_experimental/submission_v11_experimental.csv')
    v27 = read_sub('output_v27_experimental/submission_v27_experimental.csv')
    v52 = read_sub('output_v52_exploratory/submission_v52_exploratory.csv')

    diffs = {}
    if v11 and v27:
        diff_v27_ids = [k for k in v11 if v11[k] != v27[k]]
        diffs['v11_vs_v27'] = {
            'count': len(diff_v27_ids),
            'samples': [
                {
                    'id': k,
                    'v11': v11[k],
                    'v27': v27[k],
                    'rating': test_data[k]['rating'],
                    'product': test_data[k]['product_name'][:40],
                    'comment': test_data[k]['comment'][:120],
                }
                for k in diff_v27_ids
            ],
        }

    if v11 and v52:
        diff_v52_ids = [k for k in v11 if v11[k] != v52[k]]
        diffs['v11_vs_v52'] = {
            'count': len(diff_v52_ids),
            'samples': [
                {
                    'id': k,
                    'v11': v11[k],
                    'v52': v52[k],
                    'rating': test_data[k]['rating'],
                    'product': test_data[k]['product_name'][:40],
                    'comment': test_data[k]['comment'][:120],
                }
                for k in diff_v52_ids
            ],
        }
    return diffs


def analyze_cached_oof():
    """Analyze the cached reference OOF arrays."""
    ref_path = WORKSPACE / 'colab_reference.npz'
    if not ref_path.exists():
        return {'status': 'ref_not_found'}

    z = np.load(ref_path)
    oof = z['oof']
    y = z['y']
    folds = z['folds']

    # Raw argmax vs Weighted argmax
    raw_macro, raw_per_class = calculate_macro_f1(y, np.argmax(oof, axis=1))
    weights = np.array([1.0, 1.2, 1.2, 0.9])
    weighted_macro, weighted_per_class = calculate_macro_f1(y, np.argmax(oof * weights, axis=1))

    # Confusion Matrix
    pred_w = np.argmax(oof * weights, axis=1)
    cm = np.zeros((4, 4), dtype=int)
    for true_c in range(4):
        for pred_c in range(4):
            cm[true_c, pred_c] = int(np.sum((y == true_c) & (pred_w == pred_c)))

    # Fold by fold metrics
    fold_metrics = []
    for f in range(3):
        mask = (folds == f)
        m, pc = calculate_macro_f1(y[mask], pred_w[mask])
        fold_metrics.append({'fold': f, 'macro_f1': round(m, 5), 'per_class': pc})

    return {
        'raw_macro_f1': round(raw_macro, 5),
        'weighted_macro_f1': round(weighted_macro, 5),
        'raw_per_class': raw_per_class,
        'weighted_per_class': weighted_per_class,
        'confusion_matrix': cm.tolist(),
        'fold_metrics': fold_metrics,
    }


def main():
    print('Running independent diagnostics on Kaggle DS102...')
    results = {
        'sensitivity_simulation': run_sensitivity_simulation(),
        'cached_oof_analysis': analyze_cached_oof(),
        'test_flip_analysis': analyze_test_flips(),
    }
    out_path = Path(__file__).resolve().parent / 'diagnostics_summary.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'Diagnostics successfully saved to {out_path}')


if __name__ == '__main__':
    main()
