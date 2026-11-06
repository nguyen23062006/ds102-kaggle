"""Shared helpers for the Claude review experiments (no fitting in this file).

Near-duplicate grouping uses exact token-set Jaccard on a binary token matrix
whose vocabulary is just "tokens seen in the given texts" (deterministic
counting, not a learned model). Still: call these from Colab scripts.
"""
import re
import numpy as np
from scipy import sparse

TOK = re.compile(r'(?u)\b\w+\b')
W_V5 = np.array([1., 1., 1.2, .9])
W_V6 = np.array([1., 1.2, 1.2, .9])


def token_matrix(texts, vocab=None):
    """Binary doc x token matrix. If vocab is given, unseen tokens still count in set sizes."""
    grow = vocab is None
    vocab = {} if grow else vocab
    rows, cols, sizes = [], [], []
    for i, t in enumerate(texts):
        s = set(TOK.findall(t)); sizes.append(len(s))
        for w in s:
            j = vocab.setdefault(w, len(vocab)) if grow else vocab.get(w)
            if j is not None:
                rows.append(i); cols.append(j)
    b = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(texts), len(vocab)))
    return b, np.asarray(sizes, np.float32), vocab


def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i


def near_dup_groups(texts, thr=.6, min_tokens=4, chunk=1024):
    """Connected components of (exact duplicate) OR (Jaccard >= thr and both have >= min_tokens tokens)."""
    texts = list(texts); n = len(texts)
    parent = np.arange(n)
    def union(a, b):
        ra, rb = _find(parent, a), _find(parent, b)
        if ra != rb: parent[max(ra, rb)] = min(ra, rb)
    first = {}
    for i, t in enumerate(texts):
        if t in first: union(first[t], i)
        else: first[t] = i
    b, s, _ = token_matrix(texts); bt = b.T.tocsc(); ok = s >= min_tokens
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        inter = (b[start:end] @ bt).toarray()
        jac = inter / np.maximum(s[start:end, None] + s[None, :] - inter, 1)
        ii, jj = np.nonzero(jac >= thr); ii = ii + start
        keep = (jj > ii) & ok[ii] & ok[jj]
        for a, c in zip(ii[keep], jj[keep]): union(int(a), int(c))
    roots = np.array([_find(parent, i) for i in range(n)])
    _, gid = np.unique(roots, return_inverse=True)
    return gid


def max_jaccard(query_texts, ref_texts, chunk=1024):
    """For each query text: max token-set Jaccard to any reference text (0 if no refs)."""
    if len(ref_texts) == 0: return np.zeros(len(query_texts), np.float32)
    rb, rs, vocab = token_matrix(ref_texts); qb, qs, _ = token_matrix(query_texts, vocab)
    rbt = rb.T.tocsc(); out = np.zeros(len(query_texts), np.float32)
    for start in range(0, len(query_texts), chunk):
        end = min(start + chunk, len(query_texts))
        inter = (qb[start:end] @ rbt).toarray()
        out[start:end] = (inter / np.maximum(qs[start:end, None] + rs[None, :] - inter, 1)).max(1)
    return out


def class_f1(y, pred, w=None):
    cm = np.bincount(4 * y + pred, weights=w, minlength=16).reshape(4, 4)
    den = cm.sum(0) + cm.sum(1)
    return np.divide(2 * np.diag(cm), den, out=np.zeros(4), where=den > 0)


def macro(y, pred, w=None):
    return float(class_f1(y, pred, w).mean())


def group_bootstrap_delta(y, pred_a, pred_b, groups, n=1000, seed=102):
    """Cluster bootstrap of macro(pred_b) - macro(pred_a). Returns mean, 5%, 95%, P(delta>0)."""
    rng = np.random.default_rng(seed)
    _, inv = np.unique(groups, return_inverse=True); g = inv.max() + 1
    d = np.empty(n)
    for k in range(n):
        w = np.bincount(rng.integers(0, g, g), minlength=g)[inv].astype(float)
        d[k] = macro(y, pred_b, w) - macro(y, pred_a, w)
    return {'mean': float(d.mean()), 'q05': float(np.quantile(d, .05)),
            'q95': float(np.quantile(d, .95)), 'p_gt_0': float(np.mean(d > 0))}


def class1_strata(y, pred, template_mask):
    """Class-1 F1 on rows that look like a class-1 template in the fitting data vs. novel rows."""
    out = {}
    for name, m in [('template', template_mask), ('novel', ~template_mask)]:
        tp = int(np.sum((y == 1) & (pred == 1) & m)); fn = int(np.sum((y == 1) & (pred != 1) & m))
        fp = int(np.sum((y != 1) & (pred == 1) & m))
        out[name] = {'pos': tp + fn, 'tp': tp, 'fp': fp, 'fn': fn,
                     'f1': 2 * tp / max(2 * tp + fp + fn, 1), 'recall': tp / max(tp + fn, 1)}
    return out
