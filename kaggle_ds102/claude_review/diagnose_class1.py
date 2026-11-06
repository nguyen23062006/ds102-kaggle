"""Read-only: v5 class-1 recall/precision split by "has a near-duplicate
class-1 template in the fitting data" (Jaccard >= 0.6). No fitting."""
import json
from pathlib import Path
import numpy as np
from diagnose_cached import read, ROOT
from diagnose_neardup import toks, jac

W5 = np.array([1, 1, 1.2, .9])


def main():
    tr, dv = read('train'), read('dev')
    ytr = np.array([int(r['spam_label']) for r in tr]); ydv = np.array([int(r['spam_label']) for r in dv])
    r = np.load(ROOT / 'colab_reference.npz'); folds = r['folds']
    T = [toks(x['comment']) for x in tr]; D = [toks(x['comment']) for x in dv]
    idx1 = np.where(ytr == 1)[0]
    po = (r['oof'] * W5).argmax(1); pd_ = (r['dev'] * W5).argmax(1)
    # for every row: does a class-1 train row in the *fitting* data look like it?
    oof_t = np.array([max((jac(T[i], T[j]) for j in idx1 if folds[j] != folds[i]), default=0) >= .6 for i in range(len(T))])
    dev_t = np.array([max(jac(d, T[j]) for j in idx1) >= .6 for d in D])
    def stats(y, p, m):
        out = {}
        for name, sel in [('template', m), ('novel', ~m)]:
            tp = int(np.sum((y == 1) & (p == 1) & sel)); fn = int(np.sum((y == 1) & (p != 1) & sel))
            fp = int(np.sum((y != 1) & (p == 1) & sel))
            out[name] = dict(pos=tp + fn, tp=tp, fn=fn, fp=fp, recall=tp / max(tp + fn, 1),
                             f1=2 * tp / max(2 * tp + fp + fn, 1), rows=int(sel.sum()))
        return out
    rep = {'oof_v5': stats(ytr, po, oof_t), 'dev_v5': stats(ydv, pd_, dev_t)}
    Path(__file__).with_name('diagnostics_class1.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == '__main__':
    main()
