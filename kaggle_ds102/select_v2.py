"""Chọn mô hình/ensemble và bias bằng OOF train, không xem nhãn dev."""
import json
from pathlib import Path
import numpy as np


def macro(y, pred):
    cm = np.bincount(4 * y + pred, minlength=16).reshape(4, 4)
    den = cm.sum(axis=0) + cm.sum(axis=1)
    return float(np.mean(np.divide(2 * np.diag(cm), den, out=np.zeros(4), where=den > 0)))


def calibrate(scores, y):
    # 3 offsets tương đối so với nhãn 0; grid nhỏ để hạn chế overfit.
    bias = np.zeros(4)
    for _ in range(3):
        previous = bias.copy()
        for cls in [1, 2, 3]:
            choices = []
            for value in np.linspace(-.4, .4, 17):
                proposal = bias.copy()
                proposal[cls] = value
                objective = macro(y, (scores + proposal).argmax(axis=1)) - .005 * np.square(proposal).sum()
                choices.append((objective, -abs(value), float(value)))
            bias[cls] = max(choices)[2]
        if np.allclose(previous, bias):
            break
    return bias


def select(scores, y, folds, configs):
    # NB-SVM C=1/2 có cảnh báo chưa hội tụ ở thử nghiệm, không chọn chúng.
    candidates = [(cfg['name'], {i: 1.}) for i, cfg in enumerate(configs)
                  if cfg['kind'] != 'nb' or cfg['C'] == .5]
    # Blends định trước, tránh tìm hàng trăm trọng số trên lớp hiếm.
    byname = {c['name']: i for i, c in enumerate(configs)}
    for svm_w in [.25, .5, .75]:
        candidates.append((f'blend_svm_nb_{svm_w}',
                           {byname['svm_0.5_balanced']: svm_w, byname['nb_0.5']: 1-svm_w}))
    rows = []
    for name, weights in candidates:
        s = sum(scores[i] * w for i, w in weights.items())
        cross_pred = np.zeros(len(y), dtype=int)
        for f in np.unique(folds):
            fit = folds != f
            b = calibrate(s[fit], y[fit])
            cross_pred[~fit] = (s[~fit] + b).argmax(axis=1)
        raw, calibrated = macro(y, s.argmax(axis=1)), macro(y, cross_pred)
        # Chọn bias chỉ nếu calibration cải thiện khi kiểm tra chéo.
        use_bias = calibrated > raw + .002
        row = {'name': name, 'weights': weights, 'raw_oof_macro_f1': raw,
               'cross_calibration_macro_f1': calibrated, 'use_bias': use_bias,
               'selection_score': calibrated if use_bias else raw,
               'bias': calibrate(s, y).tolist() if use_bias else [0., 0., 0., 0.]}
        rows.append(row)
    rows.sort(key=lambda r: r['selection_score'], reverse=True)
    return rows


if __name__ == '__main__':
    folder = Path('kaggle_ds102/experiments_v2')
    z = np.load(folder/'oof.npz')
    cfg = json.loads((folder/'configs.json').read_text())
    rows = select(z['scores'], z['y'], z['folds'], cfg)
    (folder/'selection.json').write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(r['name'], 'raw', round(r['raw_oof_macro_f1'], 5),
              'cross calibrated', round(r['cross_calibration_macro_f1'], 5), 'bias', r['bias'])
