"""Package existing competition data and source; performs no ML fitting."""
import ast, hashlib, json, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path('/Users/ngien2306/Downloads/ds-102-r-11-cnvn')
names = ['experiment_v7.py', 'train_submit.py', 'train_submit_v2.py',
         'improved_features.py', 'catboost_v3.py', 'select_v2.py',
         'claude_review/review_common.py', 'claude_review/codex_checked_common.py',
         'colab_reference.npz']
payload = {n: (ROOT/n).read_bytes() for n in names}
for n, b in payload.items():
    if n.endswith('.py'): ast.parse(b, filename=n)
for kind in ['train', 'dev', 'test', 'samplesubmission']:
    name = f'ds102_{kind}.csv'
    payload['data/'+name] = (DATA/name).read_bytes()
manifest = {n: hashlib.sha256(b).hexdigest() for n,b in payload.items()}
payload['bundle_manifest.json'] = json.dumps(manifest,indent=2).encode()
bundle = ROOT/'ds102_v7_bundle.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
    for n,b in payload.items(): z.writestr(n,b)

setup = '''import os, platform, zipfile, hashlib, json
from pathlib import Path
assert platform.system() == 'Linux' and os.environ.get('COLAB_RELEASE_TAG'), 'Hosted Colab only'
bundle = Path('/content/ds102_v7_bundle.zip')
if not bundle.exists():
    from google.colab import files
    uploaded = files.upload()
    assert bundle.exists(), 'Upload ds102_v7_bundle.zip'
    del uploaded
root = Path('/content/ds102_v7_bundle')
root.mkdir(exist_ok=True)
with zipfile.ZipFile(bundle) as z:
    for name in z.namelist():
        assert (root/name).resolve().is_relative_to(root.resolve()), 'Unsafe ZIP path'
    z.extractall(root)
manifest = json.loads((root/'bundle_manifest.json').read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
print('Bundle checked:', len(manifest), 'files. Hosted Colab CPU:', os.cpu_count())
'''
run = '''import subprocess, sys, os
env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
subprocess.run([sys.executable, '/content/ds102_v7_bundle/experiment_v7.py'], env=env, check=True)
'''
cells = [dict(cell_type='markdown',metadata={},source=[
    '# DS102 v7 — thử nghiệm, chưa xác nhận tăng điểm\n',
    'Upload ds102_v7_bundle.zip. Chỉ chạy hosted Colab; không pretrained.\n',
    'Nhánh Logistic Regression C=4, sqrt-balanced, blend 20%; không tìm hệ số.\n',
    'So sánh v5 trên exact-group CV rồi near-duplicate-group CV; gate macro +0.003, ít nhất 2/3 fold tăng, bootstrap 95% dương, novel fake F1 không giảm quá 0.01. Dev chỉ đánh giá sau CV, không fit.\n',
    'Chỉ xuất CSV candidate khi đạt gate. Không tự nộp Kaggle; điểm leaderboard chưa biết.\n'])]
for code in [setup, '%pip -q install catboost==1.2.10 scikit-learn==1.7.2', run]:
    cells.append(dict(cell_type='code',metadata={},source=code.splitlines(keepends=True),outputs=[],execution_count=None))
nb = dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),cells=cells)
(ROOT/'DS102_Colab_v7_experiment.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2))
(ROOT/'v7_colab_setup.py').write_text(setup)
print(json.dumps(dict(bundle=str(bundle),bytes=bundle.stat().st_size,files=len(manifest),syntax='passed'),ensure_ascii=False))
