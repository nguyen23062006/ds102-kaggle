"""H3 nested: frozen class-weight search space and selection rule (pure NumPy).

No model fitting happens here. The rule is fixed BEFORE any nested run:
- w0 is fixed at 1.0 (argmax is scale invariant, so one class is the anchor).
- Exhaustive grid over (w1, w2, w3) below: 7 * 9 * 5 = 315 candidates.
  The historical V11 weights (1.0, 1.2, 1.2, 0.9) are inside the grid.
- Objective: macro-F1 of argmax(p * w) on INNER out-of-fold probabilities.
- Ties (score within 1e-12 of the best): smallest L1 distance to the V11
  weights, then lexicographic order of (w1, w2, w3). Fully deterministic.
No smoothing, no secondary rule, no use of outer-validation rows.
"""
import itertools
import numpy as np

BASELINE_WEIGHTS = (1.0, 1.2, 1.2, 0.9)
W1_GRID = (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
W2_GRID = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6)
W3_GRID = (0.7, 0.8, 0.9, 1.0, 1.1)
TIE_TOLERANCE = 1e-12


def candidate_grid():
    """Return the frozen candidate weight vectors, shape (315, 4)."""
    rows = [(1.0, a, b, c) for a, b, c in itertools.product(W1_GRID, W2_GRID, W3_GRID)]
    grid = np.array(rows, dtype=np.float64)
    assert grid.shape == (315, 4)
    assert any(np.allclose(r, BASELINE_WEIGHTS) for r in grid)
    return grid


def macro_f1(y, pred, n_classes=4):
    """Unweighted macro-F1 from a confusion matrix; same formula as review_common.macro."""
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    cm = np.bincount(n_classes * y + pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * np.diag(cm), den, out=np.zeros(n_classes), where=den > 0)
    return float(f1.mean())


def weighted_labels(prob, weights):
    prob = np.asarray(prob, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    assert prob.ndim == 2 and prob.shape[1] == 4 and weights.shape == (4,)
    return (prob * weights).argmax(1)


def select_weights(prob, y):
    """Select weights on (inner) OOF probabilities. Returns (weights, info dict)."""
    prob = np.asarray(prob, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    assert len(prob) == len(y) and set(np.unique(y)) == {0, 1, 2, 3}
    assert np.isfinite(prob).all() and (prob >= 0).all()
    grid = candidate_grid()
    scores = np.array([macro_f1(y, weighted_labels(prob, w)) for w in grid])
    best = scores.max()
    tied = np.flatnonzero(scores >= best - TIE_TOLERANCE)
    base = np.asarray(BASELINE_WEIGHTS)
    order = sorted(tied, key=lambda i: (float(np.abs(grid[i] - base).sum()), tuple(grid[i][1:])))
    chosen = grid[order[0]]
    base_idx = int(np.flatnonzero(np.all(np.isclose(grid, base), axis=1))[0])
    info = dict(
        selected_weights=chosen.tolist(),
        selected_inner_macro=float(best),
        baseline_inner_macro=float(scores[base_idx]),
        inner_gain_over_baseline=float(best - scores[base_idx]),
        n_tied_at_best=int(len(tied)),
        n_candidates=int(len(grid)),
        selected_is_baseline=bool(np.allclose(chosen, base)),
    )
    return chosen, info
