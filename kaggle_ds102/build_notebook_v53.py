"""Package the frozen V53 source for manual hosted-Colab reproduction."""
import ast
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'handoff_v53'
SOURCE = OUT/'DS102_v53_source.zip'
SOURCE_SHA = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
INPUT_SHA = '19c942cbf6e615591dec5cf1e248f3130ee65793edcdd21f1a38df17e92d2a5a'
cells = []


def markdown(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(True)))


def code(text):
    ast.parse(text)
    cells.append(dict(cell_type='code', metadata={}, source=text.splitlines(True),
                      execution_count=None, outputs=[]))


markdown('''# DS102 V53 — kiểm chứng theo nhóm review gần trùng

Mô hình thử nghiệm; chưa biết điểm Kaggle. Mốc tốt nhất do người dùng báo: **V11 0.77436**.

- Chỉ huấn luyện trên **hosted Google Colab CPU**, không chọn local runtime.
- Không pretrained, không mô hình/ngữ liệu bên ngoài, không gộp dev vào train.
- Thêm 64 chủ đề NMF học từ review của phần train, giữ SVD128 và dùng CatBoost.
- Kiểm chứng trước bằng 3 fold gộp review gần trùng (Jaccard 0.7, seed 2053), rồi mới chấm trên fold cũ.
- Giữ nguyên mọi ngưỡng đã định trước; chỉ xuất CSV khi cả hai CV và dev đều qua kiểm tra.
- Fold mới vẫn chứa dữ liệu đã từng dùng trong phát triển; không phải tập kiểm tra độc lập chưa từng xem.

Chạy từng ô theo thứ tự. Chuẩn bị `DS102_v43_inputs.zip` đã được cung cấp trước đó.
Không tự động nộp Kaggle. Notebook này chứa cùng mã nguồn với `DS102_v53_source.zip`.
''')
code('''import os, platform, sys, subprocess
from pathlib import Path
assert platform.system()=='Linux' and os.environ.get('COLAB_RELEASE_TAG'), 'Hosted Colab only'
ROOT=Path('/content/ds102_v53'); ROOT.mkdir(exist_ok=True)
PY=ROOT/'venv/bin/python'; SRC=ROOT/'source'; INPUTS=ROOT/'inputs'; OUT=ROOT/'run'
subprocess.check_call([sys.executable,'-m','venv','--without-pip','--system-site-packages',str(ROOT/'venv')])
subprocess.check_call([sys.executable,'-m','pip','--python',str(PY),'install','--quiet',
    '--disable-pip-version-check','numpy==2.4.6','scipy==1.16.3','pandas==2.2.3',
    'scikit-learn==1.7.2','catboost==1.2.10','joblib'])
print('Môi trường Colab sẵn sàng; không cần restart kernel.')
''')
code(f'''import base64, hashlib, json, zipfile
SRC.mkdir(exist_ok=True)
payload=base64.b64decode({base64.b64encode(SOURCE.read_bytes()).decode()!r})
assert hashlib.sha256(payload).hexdigest()=={SOURCE_SHA!r}
source_zip=ROOT/'DS102_v53_source.zip'; source_zip.write_bytes(payload)
with zipfile.ZipFile(source_zip) as z:
    for n in z.namelist(): assert (SRC/n).resolve().is_relative_to(SRC.resolve())
    z.extractall(SRC)
print('Đã ghi mã nguồn cố định:',len(list(SRC.glob('*.py'))),'module')
''')
code(f'''from google.colab import files
expected={INPUT_SHA!r}
packs=[p for p in Path('/content').glob('DS102_v43_inputs*.zip') if hashlib.sha256(p.read_bytes()).hexdigest()==expected]
if not packs:
    print('Chọn DS102_v43_inputs.zip từ máy của bạn.')
    uploaded=files.upload()
    packs=[p for p in Path('/content').glob('DS102_v43_inputs*.zip') if hashlib.sha256(p.read_bytes()).hexdigest()==expected]
    del uploaded
assert packs, 'Thiếu gói đầu vào đúng phiên bản; không đổi mã băm để bỏ qua kiểm tra.'
INPUTS.mkdir(exist_ok=True)
with zipfile.ZipFile(packs[0]) as z:
    for n in z.namelist(): assert (INPUTS/n).resolve().is_relative_to(INPUTS.resolve())
    z.extractall(INPUTS)
expected_files=json.loads((INPUTS/'inputs_manifest.json').read_text())['files']
for n,sha in expected_files.items(): assert hashlib.sha256((INPUTS/n).read_bytes()).hexdigest()==sha,n
print('Đã kiểm tra',len(expected_files),'file đầu vào.')
''')
code('''env=os.environ.copy()
env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1')
subprocess.check_call([str(PY),'-u',str(SRC/'experiment_v53.py'),'--inputs',str(INPUTS),
                       '--output',str(ROOT/'smoke'),'--smoke'],env=env,cwd=SRC)
smoke=json.loads((ROOT/'smoke/metrics_v53.json').read_text())
assert smoke['stage']=='smoke_passed'
print('Thử nhỏ thành công; kết quả này chưa dùng để chọn mô hình.')
''')
code('''# Không khởi động thêm khi cùng lượt thử đang chạy trong runtime này.
running=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
assert not any(str(SRC/'experiment_v53.py') in s and '--smoke' not in s and '--output '+str(OUT) in s for s in running), 'V53 đang chạy; chờ lượt hiện tại.'
command=[str(PY),'-u',str(SRC/'experiment_v53.py'),'--inputs',str(INPUTS),'--output',str(OUT)]
with (ROOT/'manual_run.log').open('a') as log:
    proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          text=True,bufsize=1,env=env,cwd=SRC)
    for line in proc.stdout:
        print(line,end='',flush=True); log.write(line); log.flush()
    rc=proc.wait()
assert rc==0, 'Lượt chạy lỗi; tải feedback để kiểm tra. Có checkpoint để tiếp tục.'
report=json.loads((OUT/'metrics_v53.json').read_text())
print('Trạng thái cuối:',report['stage'])
print('Có ứng viên để nộp:',report.get('submission_created',False))
''')
code('''# Có thể chạy ô này cả khi ô thử nghiệm dừng/lỗi để lấy báo cáo.
feedback=ROOT/'feedback_v53.zip'
with zipfile.ZipFile(feedback,'w',zipfile.ZIP_DEFLATED) as z:
    for n in ['metrics_v53.json','run_manifest.json','error.txt']:
        p=OUT/n
        if p.exists(): z.write(p,n)
    candidate=OUT/'submission_v53_candidate.csv'
    if (OUT/'metrics_v53.json').exists():
        report=json.loads((OUT/'metrics_v53.json').read_text())
        if report.get('stage')=='candidate_ready_not_submitted' and candidate.exists():
            z.write(candidate,candidate.name)
files.download(str(feedback))
if (OUT/'metrics_v53.json').exists() and json.loads((OUT/'metrics_v53.json').read_text()).get('stage')=='candidate_ready_not_submitted':
    files.download(str(OUT/'submission_v53_candidate.csv'))
else:
    print('Chưa có ứng viên vượt đủ ngưỡng; giữ V11 làm mốc.')
''')
notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
              language_info=dict(name='python'),colab=dict(name='DS102_Colab_v53.ipynb',provenance=[])),nbformat=4,nbformat_minor=5)
path=OUT/'DS102_Colab_v53.ipynb'
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
print(path)
