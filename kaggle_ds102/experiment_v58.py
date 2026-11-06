"""Supervised fastText trained from random initialization on each fitting fold.

Unlike V56/57, labels train the token representations and classifier jointly.
Fixed experiment; no search against dev/test or leaderboard labels.
"""
import argparse, hashlib, json, os, platform, subprocess, sys
from pathlib import Path

def guard():
    if platform.system() != 'Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Hosted Google Colab only; local fitting disabled.')

if __name__ == '__main__': guard()
sys.path.insert(0, '/content/ds102_v53/source')
import numpy as np
from improved_features import tokens
import protocol_v58 as driver

BINARY = Path('/content/ds102_v56/upstream/fastText-0.9.2/fasttext')
STATE = {}
CONFIG = dict(model_family='Supervised fastText64 with word bigrams and character3-6',
    pretrained=False, external_data=False, initialization='fresh random',
    fit_data='training partitions only', dim=64, epoch=25, lr=.2,
    wordNgrams=2, minn=3, maxn=6, bucket=100000, minCount=2, loss='softmax',
    seed=102, thread=1, text='review only; 512 token cap',
    resampling='per-class target round(sqrt(n_max * n_class)); deterministic bootstrap minority only',
    combination='.85 * complete V11 baseline + .15 * supervised fastText',
    class_weights=[1.,1.2,1.2,.9],
    gates=dict(cluster_min_delta=.003, exact_min_delta=.002,
        min_positive_folds=2, max_fold_drop=.005, min_novel_fake_f1_delta=0.,
        bootstrap_q025_positive=True, min_dev_delta=.001,
        min_dev_without_exact_overlap_delta=.001),
    selection_note='Adaptive research after V57 rejection; bootstrap is conditional, not post-selection proof')

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def texts(frame):
    return [' '.join([w for w in tokens(str(t)) if not w.startswith('__label__')][:512])
            or 'emptyreview' for t in frame.comment]

class Predictor:
    classes_ = np.arange(4)
    def __init__(self, path): self.path = str(path)
    def predict_proba(self, frame):
        result = subprocess.run([str(BINARY), 'predict-prob', self.path, '-', '4', '0'],
            input='\n'.join(texts(frame))+'\n', capture_output=True, text=True, check=True)
        lines = result.stdout.splitlines()
        assert len(lines) == len(frame)
        p = np.zeros((len(frame), 4))
        for i, line in enumerate(lines):
            fields = line.split(); assert len(fields) == 8
            seen = set()
            for name, value in zip(fields[::2], fields[1::2]):
                assert name.startswith('__label__')
                k = int(name[len('__label__'):]); assert 0 <= k < 4 and k not in seen
                seen.add(k); p[i,k] = float(value)
        assert np.isfinite(p).all() and (p >= 0).all() and (p.sum(1) > 0).all()
        return p / p.sum(1, keepdims=True)

def fitted_head(train, groups, seed=102):
    guard()
    key = hashlib.sha256(json.dumps(dict(indices=train.index.tolist(), config=driver.CONFIG,
        smoke=STATE['smoke']), sort_keys=True).encode()).hexdigest()
    folder = STATE['out']/'partitions'/key; folder.mkdir(parents=True, exist_ok=True)
    y = train.spam_label.to_numpy(); counts = np.bincount(y, minlength=4)
    assert (counts > 0).all()
    rng = np.random.default_rng(102)
    chosen = []
    for c in range(4):
        idx = np.flatnonzero(y == c)
        target = int(round(np.sqrt(counts.max()*counts[c])))
        chosen.extend(idx.tolist())
        chosen.extend(rng.choice(idx, max(0,target-len(idx)), replace=True).tolist())
    chosen = rng.permutation(chosen); content = texts(train)
    corpus = '\n'.join(f'__label__{y[i]} {content[i]}' for i in chosen)+'\n'
    corpus_sha = hashlib.sha256(corpus.encode()).hexdigest()
    corpus_path = folder/'train.txt'; model_path = folder/'supervised.bin'
    marker = folder/'complete.json'
    if marker.exists():
        cached = json.loads(marker.read_text())
        assert cached['key'] == key and cached['corpus_sha256'] == corpus_sha
        assert cached['model_sha256'] == digest(model_path)
    else:
        corpus_path.write_text(corpus)
        cmd = [str(BINARY), 'supervised', '-input', str(corpus_path),
            '-output', str(folder/'supervised'), '-dim', '64', '-epoch', '2' if STATE['smoke'] else '25',
            '-lr', '.2', '-wordNgrams', '2', '-minn', '3', '-maxn', '6',
            '-bucket', '100000', '-minCount', '2', '-loss', 'softmax',
            '-thread', '1', '-seed', '102', '-verbose', '2']
        assert '-pretrainedVectors' not in cmd
        print(json.dumps(dict(stage='fit_supervised_fasttext', rows=len(train),
            resampled_rows=len(chosen), class_counts=counts.tolist())), flush=True)
        with (folder/'training.log').open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        marker.write_text(json.dumps(dict(key=key, indices=train.index.tolist(), config=driver.CONFIG,
            corpus_sha256=corpus_sha, model_sha256=digest(model_path), command=cmd,
            rows=len(train), resampled_rows=len(chosen), class_counts=counts.tolist()), indent=2))
    return Predictor(model_path)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--inputs', required=True)
    p.add_argument('--output', required=True); p.add_argument('--smoke', action='store_true'); a=p.parse_args()
    STATE.update(out=Path(a.output), smoke=a.smoke)
    build = json.loads(Path('/content/ds102_v56/fasttext_build_manifest.json').read_text())
    assert digest(BINARY) == build['binary_sha256']
    driver.CONFIG = {**CONFIG, 'head_source_sha256':digest(__file__), 'fasttext_build':build}
    driver.main(a, fitted_head)
