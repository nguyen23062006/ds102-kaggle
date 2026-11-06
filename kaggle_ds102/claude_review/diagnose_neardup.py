"""Read-only: is OOF class-1 (fake review) easier than dev class-1 because of
near-duplicate templates / product campaigns leaking across comment-grouped folds?

Pure-Python token-set Jaccard and counting only; no fitting. A few seconds.
"""
import json, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from diagnose_cached import read, norm, ROOT

TOK = re.compile(r'(?u)\b\w+\b')


def toks(s):
    return frozenset(TOK.findall(norm(s)))


def jac(a, b):
    return len(a & b) / max(len(a | b), 1)


def main():
    tr, dv, te = read('train'), read('dev'), read('test')
    ytr = np.array([int(r['spam_label']) for r in tr]); ydv = np.array([int(r['spam_label']) for r in dv])
    folds = np.load(ROOT / 'colab_reference.npz')['folds']
    T = [toks(r['comment']) for r in tr]; D = [toks(r['comment']) for r in dv]; E = [toks(r['comment']) for r in te]
    P = [norm(r['product_name']) for r in tr]; PD = [norm(r['product_name']) for r in dv]
    rep = {}
    for c in (1, 2, 3):
        idx = np.where(ytr == c)[0]
        # nearest same-class neighbour in OTHER folds (what OOF effectively gets to see)
        oof_nn = [max((jac(T[i], T[j]) for j in idx if folds[j] != folds[i]), default=0) for i in idx]
        d_idx = np.where(ydv == c)[0]
        dev_nn = [max((jac(D[i], T[j]) for j in idx), default=0) for i in d_idx]
        prod_c = defaultdict(set)
        for j in idx: prod_c[P[j]].add(folds[j])
        oof_prod = np.mean([len(prod_c[P[i]] - {folds[i]}) > 0 for i in idx])
        dev_prod = np.mean([PD[i] in prod_c for i in d_idx])
        rep[f'class{c}'] = {
            'n_train': int(len(idx)), 'n_dev': int(len(d_idx)),
            'oof_nn_jaccard_median': float(np.median(oof_nn)), 'dev_nn_jaccard_median': float(np.median(dev_nn)),
            'oof_share_nn_ge_0.6': float(np.mean(np.array(oof_nn) >= .6)),
            'dev_share_nn_ge_0.6': float(np.mean(np.array(dev_nn) >= .6)),
            'oof_share_product_has_same_class_in_other_fold': float(oof_prod),
            'dev_share_product_has_same_class_in_train': float(dev_prod),
        }
    # class-1 in test: how many test rows are near-dups (>=0.6) of train class-1 rows?
    idx1 = np.where(ytr == 1)[0]
    rep['test_rows_nn_ge_0.6_to_train_class1'] = int(sum(max(jac(e, T[j]) for j in idx1) >= .6 for e in E))
    rep['dev_rows_nn_ge_0.6_to_train_class1'] = int(sum(max(jac(d, T[j]) for j in idx1) >= .6 for d in D))
    out = Path(__file__).with_name('diagnostics_neardup.json')
    out.write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2))


if __name__ == '__main__':
    main()
