"""Read-only diagnostics on cached predictions + raw CSV metadata.

NO model/vectorizer/SVD/scaler fitting happens here: only counting, argmax and
bootstrap resampling of already-saved predictions. Safe to run on the Mac
(pure stdlib + numpy, a few seconds). Writes claude_review/diagnostics.json.
"""
import csv, html, json, re, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / 'Downloads/ds-102-r-11-cnvn'
RNG = np.random.default_rng(102)
csv.field_size_limit(sys.maxsize)


def norm(v):
    t = html.unescape(unicodedata.normalize('NFC', str(v)))
    t = t.replace('\\r\\n', '\n').replace('\\n', '\n').replace('\\r', '\n')
    return re.sub(r'\s+', ' ', t).strip().lower()


def read(name):
    with open(DATA / f'ds102_{name}.csv', encoding='utf-8', newline='') as fh:
        return list(csv.DictReader(fh))


def f1s(y, p, k=4):
    out = []
    for c in range(k):
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        out.append(0. if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return np.array(out)


def macro(y, p):
    return float(f1s(y, p).mean())


def softmax(s):
    s = s - s.max(1, keepdims=True); e = np.exp(s); return e / e.sum(1, keepdims=True)


def entropy(p):
    return float(np.mean(-(p * np.log(np.clip(p, 1e-12, 1))).sum(1)))


def main():
    rep = {}
    tr, dv, te = read('train'), read('dev'), read('test')
    ytr = np.array([int(r['spam_label']) for r in tr]); ydv = np.array([int(r['spam_label']) for r in dv])

    # ---- 1. product-level overlap / label concentration (grouping is by comment only)
    prod = lambda rows: [norm(r['product_name']) for r in rows]
    ptr, pdv, pte = prod(tr), prod(dv), prod(te)
    st = set(ptr)
    rep['product'] = {
        'n_unique_train': len(st),
        'dev_rows_with_product_seen_in_train': float(np.mean([p in st for p in pdv])),
        'test_rows_with_product_seen_in_train': float(np.mean([p in st for p in pte])),
        'test_rows_with_product_seen_in_train_or_dev': float(np.mean([p in st | set(pdv) for p in pte])),
    }
    by = defaultdict(list)
    for p, y in zip(ptr, ytr): by[p].append(y)
    # how predictive is "majority label of the other reviews of the same product" (leave-one-out)?
    loo = []
    for p, y in zip(ptr, ytr):
        c = Counter(by[p]); c[y] -= 1
        loo.append(max(range(4), key=lambda k: (c[k], k == 0)) if sum(c.values()) else 0)
    rep['product']['train_loo_product_majority_macro_f1'] = macro(ytr, np.array(loo))
    sizes = np.array([len(by[p]) for p in ptr])
    rep['product']['train_rows_in_products_with_>=5_reviews'] = float(np.mean(sizes >= 5))
    for c in (1, 2, 3):
        rows = ytr == c
        rep['product'][f'class{c}_share_of_label_mass_in_top10pct_products'] = float(
            sum(sorted(Counter(np.array(ptr)[rows]).values(), reverse=True)[:max(1, len(st) // 10)]) / rows.sum())

    # ---- 2. CV-vs-inference scale shift on cached v5 components
    z = np.load(ROOT / 'colab_components_v5.npz'); r = np.load(ROOT / 'colab_reference.npz')
    folds = z['folds']
    def blend(svm, cat): return .5 * softmax(3 * svm) + .5 * cat
    oof, dev = blend(z['svm_oof'], z['cat_oof']), blend(z['svm_dev'], z['cat_dev'])
    rep['scale_shift'] = {
        'svm_margin_std_oof': float(z['svm_oof'].std()), 'svm_margin_std_dev': float(z['svm_dev'].std()),
        'svm_margin_std_test': float(z['svm_test'].std()),
        'svm_top2_gap_oof': float(np.mean(np.diff(np.sort(z['svm_oof'], 1)[:, -2:], axis=1))),
        'svm_top2_gap_dev': float(np.mean(np.diff(np.sort(z['svm_dev'], 1)[:, -2:], axis=1))),
        'svm_top2_gap_test': float(np.mean(np.diff(np.sort(z['svm_test'], 1)[:, -2:], axis=1))),
        'cat_entropy_oof': entropy(z['cat_oof']), 'cat_entropy_dev': entropy(z['cat_dev']),
        'cat_entropy_test': entropy(z['cat_test']),
        'blend_entropy_oof': entropy(oof), 'blend_entropy_dev': entropy(dev),
        'reference_matches_blend_oof': float(np.abs(oof - r['oof']).max()),
        'pred_class_rate_oof': np.bincount(oof.argmax(1), minlength=4).tolist(),
        'pred_class_rate_dev': np.bincount(dev.argmax(1), minlength=4).tolist(),
        'pred_class_rate_test': np.bincount(blend(z['svm_test'], z['cat_test']).argmax(1), minlength=4).tolist(),
    }

    # ---- 3. noise floor of dev and of OOF-fold scores
    w5 = np.array([1, 1, 1.2, .9])
    p5 = (r['dev'] * w5).argmax(1)
    base = macro(ydv, p5)
    boots, cls1 = [], []
    for _ in range(2000):
        i = RNG.integers(0, len(ydv), len(ydv))
        boots.append(macro(ydv[i], p5[i]))
    rep['dev_noise'] = {
        'v5_dev_from_cache': base,
        'bootstrap_sd': float(np.std(boots)),
        'bootstrap_95ci': [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
        'macro_change_per_single_class1_error': 'approx 0.25*2/(2TP+FP+FN) ~ 0.0045-0.007 with 17 positives',
        'dev_class_counts': np.bincount(ydv, minlength=4).tolist(),
    }
    # weight grid sensitivity: how much does dev move across the grid that OOF chose from?
    from itertools import product as P
    grid = {}
    for a, b, c in P([.8, 1., 1.2, 1.4], [.8, 1., 1.2, 1.4], [.9, 1., 1.1]):
        w = np.array([1, a, b, c])
        grid[str(w.tolist())] = (macro(ytr, (r['oof'] * w).argmax(1)), macro(ydv, (r['dev'] * w).argmax(1)))
    oofs = np.array([v[0] for v in grid.values()]); devs = np.array([v[1] for v in grid.values()])
    rep['weight_grid'] = {
        'oof_range': [float(oofs.min()), float(oofs.max())], 'dev_range': [float(devs.min()), float(devs.max())],
        'spearman_oof_vs_dev': float(np.corrcoef(np.argsort(np.argsort(oofs)), np.argsort(np.argsort(devs)))[0, 1]),
    }
    # per-fold class-1 instability in OOF
    rep['oof_fold_class_f1_v3ref'] = [f1s(ytr[folds == f], r['oof'][folds == f].argmax(1)).round(4).tolist()
                                      for f in np.unique(folds)]
    rep['oof_fold_class1_counts'] = [int(np.sum(ytr[folds == f] == 1)) for f in np.unique(folds)]

    # ---- 4. duplicates
    ctr = set(norm(x['comment']) for x in tr)
    rep['dup'] = {'dev_comment_seen_in_train': int(sum(norm(x['comment']) in ctr for x in dv)),
                  'test_comment_seen_in_train': int(sum(norm(x['comment']) in ctr for x in te))}
    out = Path(__file__).with_name('diagnostics.json')
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
