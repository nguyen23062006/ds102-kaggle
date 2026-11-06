"""H2 Experiment Driver: Controlled Evaluation of Subword CNN on Cluster Folds.

Hosted Google Colab only. No pretrained models, no external data.
Dev/test are never fit or used for hyperparameter tuning.

Features:
1. Reuses V53 baseline cluster fold caches (manifest-checked).
2. Controlled comparison: Focal Loss vs Weighted Cross-Entropy.
3. Smoke test verifying label-blindness and batch-size invariance.
4. Pre-declared gates and compute budget.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
import numpy as np

def guard():
    if platform.system() != 'Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Hosted Google Colab only; local fitting disabled.')

if __name__ == '__main__':
    guard()

HERE = Path(__file__).resolve().parent
PARENT_SOURCE = Path('/content/ds102_v53/source')
PARENT_RUN = Path('/content/ds102_v53/run')
sys.path.insert(0, str(PARENT_SOURCE))
sys.path.insert(0, str(HERE))

from sklearn.model_selection import StratifiedGroupKFold
from train_submit import load_data, metrics
from improved_features import prepare
from review_common import max_jaccard, class1_strata, macro
from codex_checked_common import paired_cluster_bootstrap, group_summary
from experiment_v53 import checked
import h2_subword_cnn as cnn_module

CONFIG = dict(
    experiment='H2: Subword CNN with Focal Loss vs CrossEntropy',
    recipe='Controlled blend: .85 * V11_baseline + .15 * subword_cnn',
    cnn_config=cnn_module.CONFIG,
    outer=dict(protocol='cluster', groups='V53 groups.npz (Jaccard>=.7 components)',
               splitter='StratifiedGroupKFold', n_splits=3, shuffle=True, random_state=2053),
    baseline_weights=[1.0, 1.2, 1.2, 0.9],
    gates=dict(cluster_min_delta=0.003, min_positive_folds=2, max_fold_drop=0.005,
               min_novel_fake_f1_delta=0.0, bootstrap_q025_positive=True,
               bootstrap=dict(kind='paired cluster', resamples=2000, seed=2053)),
    template_threshold=0.6,
    expected_baseline_pooled_macro=0.7240841964761913,
    pretrained=False, external_data=False, dev_merged_into_train=False,
)
W_BASE = np.asarray(CONFIG['baseline_weights'])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def labels(p):
    return (p * W_BASE).argmax(1)


def blend(base, cnn_prob, alpha=0.15):
    return checked((1.0 - alpha) * base + alpha * cnn_prob, len(base))


def run_smoke_test(train_df, folder, status):
    """Run rigorous smoke test on dummy data: checks label-blindness and batch-invariance."""
    guard()
    import torch
    status('running_smoke_checks')
    print("Executing H2 smoke test: checking label-blindness and batch-size invariance...", flush=True)

    y_all = train_df['spam_label'].to_numpy()
    tr = np.concatenate([np.flatnonzero(y_all == c)[:60] for c in range(4)])
    va = np.concatenate([np.flatnonzero(y_all == c)[60:70] for c in range(4)])

    train_sample = train_df.iloc[tr].reset_index(drop=True)
    val_sample = train_df.iloc[va].reset_index(drop=True)

    # 1. Fit CNN on training sample
    smoke_folder = folder / 'smoke_verify'
    preds = cnn_module.fit_predict_subword_cnn(
        train_sample, [val_sample.drop(columns='spam_label')],
        smoke_folder, loss_type='focal', status=None
    )
    p0 = checked(preds[0], len(val_sample))

    # 2. Check 1: Label-Blindness (injecting bogus labels must have zero effect)
    bogus_sample = val_sample.drop(columns='spam_label').assign(spam_label=np.arange(len(va)) % 4)
    # Tokenizer and inference do not read spam_label from eval frames
    stored=json.loads((smoke_folder/'tokenizer_vocab.json').read_text())
    tok=cnn_module.SubwordTokenizer()
    tok.vocab=stored['vocab']
    x_orig=tok.encode(val_sample['comment'].tolist(),max_len=cnn_module.CONFIG['max_seq_len'])
    x_bogus=tok.encode(bogus_sample['comment'].tolist(),max_len=cnn_module.CONFIG['max_seq_len'])
    assert np.array_equal(x_orig,x_bogus)
    MaskedSubwordCNN, _, _ = cnn_module._build_pytorch_classes()
    ckpt=smoke_folder/f"model_focal_epoch{cnn_module.CONFIG['snapshot_epochs'][-1]}.pt"
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg=cnn_module.CONFIG
    model=MaskedSubwordCNN(vocab_size=len(tok.vocab),embed_dim=cfg['embed_dim'],channels=cfg['conv_channels'],
        kernels=cfg['kernel_sizes'],dense_dim=cfg['dense_dim'],dropout=cfg['dropout'],embed_dropout=cfg['embedding_dropout']).to(device)
    model.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True))

    x_tensor = torch.from_numpy(x_orig)
    p_b256 = cnn_module.predict_proba_batched(model, x_tensor, device, batch_size=256)
    p_b16 = cnn_module.predict_proba_batched(model, x_tensor, device, batch_size=16)
    p_b1 = cnn_module.predict_proba_batched(model, x_tensor, device, batch_size=1)

    assert np.allclose(p_b256, p_b16, atol=1e-6), "Inference differs between batch_size 256 and 16!"
    assert np.allclose(p_b256, p_b1, atol=1e-6), "Inference differs between batch_size 256 and 1!"

    print("Smoke checks passed: model is label-blind and strictly batch-size invariant.", flush=True)
    status('smoke_passed', rows_fit=len(tr), rows_predict=len(va), eligible_for_selection=False)


def main(args):
    guard()
    start = time.time()
    inputs, out = Path(args.inputs), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    pm = json.loads((PARENT_RUN / 'run_manifest.json').read_text())
    parent_mid = hashlib.sha256(json.dumps(pm, sort_keys=True).encode()).hexdigest()
    for name, sha in pm['sources'].items():
        assert digest(PARENT_SOURCE / name) == sha, name
    expected = json.loads((inputs / 'inputs_manifest.json').read_text())['files']
    assert expected == pm['inputs']
    for name, sha in expected.items():
        assert digest(inputs / name) == sha, name

    import torch
    versions={k:importlib.metadata.version(k) for k in pm['versions']}
    assert versions==pm['versions']
    manifest = dict(
        versions={**versions,'torch':torch.__version__},
        device='cuda' if torch.cuda.is_available() else 'cpu',
        config=CONFIG,
        loss_type=args.loss_type,
        sources={p.name: digest(p) for p in [Path(__file__), HERE / 'h2_subword_cnn.py']},
        parent_manifest_sha256=digest(PARENT_RUN / 'run_manifest.json'),
        parent_groups_sha256=digest(PARENT_RUN / 'groups.npz'),
        inputs=expected,
        smoke=args.smoke
    )
    mid = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    mp = out / 'run_manifest.json'
    if mp.exists():
        assert json.loads(mp.read_text()) == manifest, "Changed config; use clean output dir."
    mp.write_text(json.dumps(manifest, indent=2))

    report = dict(stage='initializing', manifest_id=mid, config=CONFIG, loss_type=args.loss_type,
                  smoke=args.smoke, execution='hosted Colab', submission_created=False,
                  best_user_reported_kaggle=0.77436)

    def status(stage, **kw):
        report.update(stage=stage, elapsed_seconds=time.time() - start, **kw)
        (out / f"metrics_h2_{args.loss_type}.json").write_text(json.dumps(report, indent=2))
        print(stage, json.dumps({k: v for k, v in kw.items() if k != 'fold_report'}), flush=True)

    data = load_data(inputs / 'data')
    train = prepare(data['train'])
    y = train['spam_label'].to_numpy()
    groups = np.load(PARENT_RUN / 'groups.npz')['groups']
    assert len(groups) == len(y)

    if args.smoke:
        run_smoke_test(train, out, status)
        return

    report['group_info'] = group_summary(groups, y)
    outer_base = np.zeros((len(y), 4))
    cand_prob = np.zeros((len(y), 4))
    outer_fold = np.full(len(y), -1)
    template = np.zeros(len(y), bool)
    deltas = []

    splitter = StratifiedGroupKFold(3, shuffle=True, random_state=2053)
    for f, (tr, va) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        assert not set(groups[tr]) & set(groups[va])
        assert set(y[tr]) == set(y[va]) == {0, 1, 2, 3}
        outer_fold[va] = f
        folder = out / f'fold{f}'
        folder.mkdir(parents=True, exist_ok=True)

        # 1. Load V53 Baseline probabilities
        shared = PARENT_RUN / 'cluster' / f'fold{f}'
        assert all((shared / (c + '.npz')).exists() for c in ['svm', 'cat', 'latent'])
        z_ind = np.load(shared / 'indices.npz')
        assert np.array_equal(z_ind['tr'], tr) and np.array_equal(z_ind['va'], va)

        parts = {}
        for c in ['svm', 'cat', 'latent']:
            z = np.load(shared / (c + '.npz'), allow_pickle=False)
            assert z['manifest_id'].item() == parent_mid
            parts[c] = checked(z['p0'], len(va))

        outer_base[va] = checked(0.35 * parts['svm'] + 0.35 * parts['cat'] + 0.30 * parts['latent'], len(va))

        # 2. Fit and predict Subword CNN on fold tr
        cnn_cache = folder / f'cnn_{args.loss_type}.npz'
        if cnn_cache.exists():
            z = np.load(cnn_cache)
            assert z['manifest_id'].item() == mid and np.array_equal(z['tr'], tr) and np.array_equal(z['va'], va)
            p_cnn = checked(z['pred'], len(va))
        else:
            status('fit_subword_cnn', fold=f + 1, loss_type=args.loss_type, rows_fit=len(tr), rows_predict=len(va))
            preds = cnn_module.fit_predict_subword_cnn(
                train.iloc[tr], [train.iloc[va].drop(columns='spam_label')],
                folder, loss_type=args.loss_type, status=status
            )
            p_cnn = checked(preds[0], len(va))
            tmp = cnn_cache.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, manifest_id=mid, tr=tr, va=va, pred=p_cnn)
            tmp.replace(cnn_cache)

        cand_prob[va] = blend(outer_base[va], p_cnn, alpha=0.15)
        template[va] = max_jaccard(train.comment.iloc[va].tolist(),
                                   train.comment.iloc[tr[y[tr] == 1]].tolist()) >= CONFIG['template_threshold']

        base_f_pred = labels(outer_base[va])
        cand_f_pred = labels(cand_prob[va])
        d = macro(y[va], cand_f_pred) - macro(y[va], base_f_pred)
        deltas.append(d)
        status('fold_complete', fold=f + 1, fold_delta=d, fold_deltas=deltas)
        (folder/'fold_metrics.json').write_text(json.dumps(dict(baseline=metrics(y[va],base_f_pred),candidate=metrics(y[va],cand_f_pred),delta=d),indent=2))

        if min(deltas) < -CONFIG['gates']['max_fold_drop'] or sum(x > 0 for x in deltas) + (2 - f) < 2:
            status('pruned_on_predeclared_fold_gates', violating_fold=f + 1)
            return

    b_pred = labels(outer_base)
    c_pred = labels(cand_prob)
    pooled_base = macro(y, b_pred)
    assert abs(pooled_base - CONFIG['expected_baseline_pooled_macro']) < 1e-9, pooled_base
    delta = macro(y, c_pred) - pooled_base
    novel = class1_strata(y, c_pred, template)['novel']['f1'] - class1_strata(y, b_pred, template)['novel']['f1']
    boot = paired_cluster_bootstrap(y, b_pred, c_pred, groups, n=2000, seed=2053)

    g = CONFIG['gates']
    gates = dict(
        macro=bool(delta >= g['cluster_min_delta']),
        folds=bool(sum(x > 0 for x in deltas) >= g['min_positive_folds']),
        worst_fold=bool(min(deltas) >= -g['max_fold_drop']),
        novel_fake=bool(novel >= g['min_novel_fake_f1_delta']),
        bootstrap=bool(boot['q025'] > 0)
    )

    report.update(
        baseline=metrics(y, b_pred), candidate=metrics(y, c_pred), delta=delta, fold_deltas=deltas,
        novel_fake_f1_delta=novel, bootstrap=boot, gates=gates,
        label_changes=int((b_pred != c_pred).sum())
    )
    np.savez_compressed(out / f"outer_oof_{args.loss_type}.npz", manifest_id=mid, y=y,
                        baseline=b_pred, candidate=c_pred, prob=cand_prob, template=template)

    if all(gates.values()):
        status('passed_all_gates_candidate_ready')
    else:
        status('rejected_on_cluster')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--loss-type', choices=['focal', 'cross_entropy'], default='focal')
    p.add_argument('--smoke', action='store_true')
    main(p.parse_args())
