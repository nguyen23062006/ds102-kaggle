"""E1 step 0 (zero training): re-score EXISTING OOF by class-1 stratum.

Inputs (already on Colab from v6 run):
  /content/colab_reference.npz           v3/v5 base OOF (oof, y, folds)
  /content/ds102_v6_latent/oof.npz       latent CatBoost OOF (oof[0]=400 trees, oof[1]=800)
No model is fitted; only argmax/counting/bootstrap. Prints JSON and writes
claude_review_out/step0_strata.json.
"""
import argparse, json
from pathlib import Path
import sys as _sys
_sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1])]
import numpy as np
from train_submit import load_data
from improved_features import prepare
from review_common import (W_V5, W_V6, max_jaccard, class_f1, macro, class1_strata,
                           group_bootstrap_delta, near_dup_groups)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='/content/ds102_data')
    ap.add_argument('--reference', default='/content/colab_reference.npz')
    ap.add_argument('--latent', default='/content/ds102_v6_latent/oof.npz')
    ap.add_argument('--out', default='/content/claude_review_out')
    a = ap.parse_args(argv)
    df = prepare(load_data(Path(a.data_dir))['train']); y = df.spam_label.to_numpy()
    ref = np.load(a.reference); lat = np.load(a.latent)
    assert np.array_equal(ref['y'], y) and np.array_equal(lat['folds'], ref['folds'])
    folds = ref['folds']; text = df.comment.tolist()
    # template flag: does a class-1 row from the OTHER folds look like this row?
    tmpl = np.zeros(len(y), bool)
    for f in np.unique(folds):
        va = np.where(folds == f)[0]; ref1 = np.where((folds != f) & (y == 1))[0]
        tmpl[va] = max_jaccard([text[i] for i in va], [text[i] for i in ref1]) >= .6
    v5 = (ref['oof'] * W_V5).argmax(1)
    v6 = ((.7 * ref['oof'] + .3 * lat['oof'][1]) * W_V6).argmax(1)
    groups = near_dup_groups(text)
    rep = {'template_rows': int(tmpl.sum()), 'class1_template_share': float(tmpl[y == 1].mean())}
    for name, p in [('v5', v5), ('v6', v6)]:
        rep[name] = {'macro': macro(y, p), 'class_f1': class_f1(y, p).round(4).tolist(),
                     'class1_strata': class1_strata(y, p, tmpl),
                     # macro F1 if class-1 template positives were as rare as on dev (~24%):
                     'macro_excl_class1_template_rows': macro(y[~tmpl], p[~tmpl])}
    rep['delta_v6_minus_v5_all'] = group_bootstrap_delta(y, v5, v6, groups)
    rep['delta_v6_minus_v5_non_template'] = group_bootstrap_delta(y[~tmpl], v5[~tmpl], v6[~tmpl], groups[~tmpl])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'step0_strata.json').write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2))


if __name__ == '__main__':
    main()
