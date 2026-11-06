"""Hypothesis 3: Cost-Sensitive Probability Calibration & Direct Macro F1 Threshold Search.

Addresses the decision-rule bottleneck: resolves the 'Class 0 Black Hole'
where 49.8% of Class 1 and 41.9% of Class 2 are absorbed by majority Class 0.
Uses nested cross-validation to prevent optimistic winner's curse.

Can evaluate cached probability arrays locally on Mac, or fit on Colab.
"""

import numpy as np


def compute_macro_f1(y_true, y_pred, n_classes=4):
    """Compute Macro F1 with pure NumPy."""
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


class NestedThresholdOptimizer:
    """Finds optimal class decision multipliers using strictly nested cross-validation."""

    def __init__(self, n_classes=4, seed=102):
        self.n_classes = n_classes
        self.seed = seed
        self.best_weights_ = None

    def _grid_search(self, probs, y_true):
        """Coordinate / grid search over candidate class multipliers."""
        best_score = -1.0
        best_w = np.ones(self.n_classes, dtype=np.float32)

        # Class 0 baseline is fixed at 1.0
        # Search over class 1 (minority fake), class 2 (logistics/unboxing), class 3 (spam)
        w1_grid = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        w2_grid = [0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        w3_grid = [0.7, 0.8, 0.9, 1.0, 1.1]

        for w1 in w1_grid:
            for w2 in w2_grid:
                for w3 in w3_grid:
                    w = np.array([1.0, w1, w2, w3], dtype=np.float32)
                    preds = np.argmax(probs * w, axis=1)
                    score, _ = compute_macro_f1(y_true, preds, self.n_classes)
                    if score > best_score:
                        best_score = score
                        best_w = w.copy()

        return best_w, best_score

    def fit_nested(self, probs, y_true, folds):
        """Run strictly nested cross-validation across the given outer folds."""
        n_samples = len(y_true)
        unique_folds = np.unique(folds)
        oof_preds = np.zeros(n_samples, dtype=int)
        fold_weights = {}

        for val_fold in unique_folds:
            train_mask = (folds != val_fold)
            val_mask = (folds == val_fold)

            # Optimize weights strictly on training folds
            best_w, train_score = self._grid_search(probs[train_mask], y_true[train_mask])
            fold_weights[int(val_fold)] = best_w.tolist()

            # Predict on holdout validation fold
            val_preds = np.argmax(probs[val_mask] * best_w, axis=1)
            oof_preds[val_mask] = val_preds

        # Global fit across all folds for final test inference
        final_w, overall_train_score = self._grid_search(probs, y_true)
        self.best_weights_ = final_w

        nested_macro, nested_f1s = compute_macro_f1(y_true, oof_preds, self.n_classes)
        return {
            'nested_oof_macro_f1': nested_macro,
            'nested_oof_per_class': nested_f1s,
            'fold_weights': fold_weights,
            'final_recommended_weights': final_w.tolist(),
        }

    def predict(self, probs, weights=None):
        if weights is None:
            weights = self.best_weights_
        return np.argmax(probs * weights, axis=1)


if __name__ == '__main__':
    from pathlib import Path
    ref_path = Path(__file__).resolve().parent.parent / 'colab_reference.npz'
    if ref_path.exists():
        print("Running Hypothesis 3 on cached reference OOF probabilities...")
        z = np.load(ref_path)
        probs = z['oof']
        y = z['y']
        folds = z['folds']

        # 1. Baseline fixed weights [1.0, 1.2, 1.2, 0.9]
        fixed_w = np.array([1.0, 1.2, 1.2, 0.9])
        fixed_preds = np.argmax(probs * fixed_w, axis=1)
        fixed_macro, fixed_f1s = compute_macro_f1(y, fixed_preds)
        print(f"Fixed Weights Baseline Macro F1: {fixed_macro:.5f}")
        print(f"Per-class F1: {[round(x, 4) for x in fixed_f1s]}")

        # 2. Nested Threshold Optimizer
        opt = NestedThresholdOptimizer(n_classes=4, seed=102)
        nested_results = opt.fit_nested(probs, y, folds)
        print("\nNested Threshold Optimization Results:")
        print(f"Nested OOF Macro F1: {nested_results['nested_oof_macro_f1']:.5f}")
        print(f"Nested Per-class F1: {[round(x, 4) for x in nested_results['nested_oof_per_class']]}")
        print("Fold weights chosen:", nested_results['fold_weights'])
        print("Final recommended weights:", nested_results['final_recommended_weights'])
    else:
        print("Cached reference NPZ not found, running mock dry-run...")
        rng = np.random.RandomState(42)
        mock_p = rng.dirichlet(np.ones(4), size=100)
        mock_y = rng.randint(0, 4, size=100)
        mock_f = np.repeat([0, 1], 50)
        opt = NestedThresholdOptimizer(n_classes=4)
        res = opt.fit_nested(mock_p, mock_y, mock_f)
        print("Mock test passed:", res['nested_oof_macro_f1'])
