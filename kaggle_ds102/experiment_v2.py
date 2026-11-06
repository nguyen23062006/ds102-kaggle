"""Thí nghiệm chỉ dùng train để chọn cấu hình; cache điểm OOF cho bước chọn."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC
from train_submit import load_data, make_features, make_model, SEED
from improved_features import prepare, FeaturesV2, nb_ratios

OUT = Path('kaggle_ds102/experiments_v2')
OUT.mkdir(exist_ok=True)
data = load_data(Path('/Users/ngien2306/Downloads/ds-102-r-11-cnvn'))
train = prepare(data['train'])
y = train.spam_label.to_numpy()
configs = [
    {'name': 'baseline', 'kind': 'baseline', 'C': 1., 'weight': 'balanced'},
    *[{'name': f'svm_{c}_{w}', 'kind': 'svm', 'C': c, 'weight': w} for c in [.5, 1., 2.] for w in ['balanced', 'sqrt_balanced']],
    *[{'name': f'nb_{c}', 'kind': 'nb', 'C': c, 'weight': 'balanced'} for c in [.5, 1., 2.]],
]
scores = np.zeros((len(configs), len(train), 4))
fold_ids = np.full(len(train), -1)
splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
for fold, (tr, va) in enumerate(splitter.split(train, y, groups=train.comment)):
    print('FOLD', fold + 1, flush=True)
    fold_ids[va] = fold
    features = FeaturesV2()
    Xt, Mt = features.fit_transform(train.iloc[tr])
    Xv, Mv = features.transform(train.iloc[va])
    X, V = sparse.hstack([Xt, Mt], format='csr'), sparse.hstack([Xv, Mv], format='csr')
    ratios = nb_ratios(Xt, y[tr])
    for i, cfg in enumerate(configs):
        if cfg['kind'] == 'baseline':
            f = make_features()
            bx = f.fit_transform(train.iloc[tr].comment)
            bv = f.transform(train.iloc[va].comment)
            model = make_model(cfg, y[tr]).fit(bx, y[tr])
            pred_scores = model.decision_function(bv)
            del f, bx, bv
        elif cfg['kind'] == 'svm':
            model = make_model(cfg, y[tr]).fit(X, y[tr])
            pred_scores = model.decision_function(V)
        else:
            pred_scores = []
            for c, r in enumerate(ratios):
                xx = sparse.hstack([Xt.multiply(r), Mt], format='csr')
                vv = sparse.hstack([Xv.multiply(r), Mv], format='csr')
                model = LinearSVC(C=cfg['C'], class_weight='balanced', dual=True,
                                  max_iter=10000, random_state=SEED).fit(xx, (y[tr] == c).astype(int))
                pred_scores.append(model.decision_function(vv))
            pred_scores = np.asarray(pred_scores).T
        scores[i, va] = pred_scores
        print(cfg['name'], round(f1_score(y[va], pred_scores.argmax(axis=1), average='macro'), 5), flush=True)
    np.savez_compressed(OUT/'oof_partial.npz', scores=scores, y=y, folds=fold_ids)
np.savez_compressed(OUT/'oof.npz', scores=scores, y=y, folds=fold_ids)
(OUT/'configs.json').write_text(json.dumps(configs, indent=2))
print('ALL FOLDS COMPLETE', flush=True)
for cfg, s in zip(configs, scores):
    print(cfg['name'], f1_score(y, s.argmax(axis=1), average='macro'), flush=True)
