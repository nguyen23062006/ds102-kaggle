"""Unit tests for h3_selection (NumPy only, no fitting). Intended to be run by Codex
on Colab before the nested run:  python test_h3_selection.py
Not executed on the Mac (only syntax-checked there)."""
import numpy as np
import h3_selection as sel


def test_grid():
    g = sel.candidate_grid()
    assert g.shape == (315, 4) and np.all(g[:, 0] == 1.0)
    assert len({tuple(r) for r in g}) == 315


def test_macro_matches_reference_formula():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 500); p = rng.integers(0, 4, 500)
    f1 = []
    for c in range(4):
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    assert abs(sel.macro_f1(y, p) - np.mean(f1)) < 1e-12


def test_baseline_kept_when_flat():
    # Identical probability rows: every candidate gives the same predictions -> tie -> baseline.
    y = np.array([0, 1, 2, 3] * 10)
    prob = np.tile([.97, .01, .01, .01], (40, 1))
    w, info = sel.select_weights(prob, y)
    assert info['selected_is_baseline'] and np.allclose(w, sel.BASELINE_WEIGHTS)


def test_selection_uses_only_given_rows_and_is_deterministic():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 400)
    prob = rng.dirichlet(np.ones(4), 400)
    prob[np.arange(400), y] += .3
    prob /= prob.sum(1, keepdims=True)
    w1, i1 = sel.select_weights(prob, y)
    w2, i2 = sel.select_weights(prob.copy(), y.copy())
    assert np.array_equal(w1, w2) and i1 == i2
    assert i1['selected_inner_macro'] >= i1['baseline_inner_macro']


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('PASS', name)
