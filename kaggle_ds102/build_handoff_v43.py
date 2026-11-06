"""Build the user-run notebook and an explicit data/prediction inputs archive.

Only standard-library source, archive and integrity processing runs locally.
No ML libraries are imported, and nothing is trained or scored on the Mac.
"""
import ast,hashlib,json,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RAW=Path('/Users/ngien2306/Downloads/ds-102-r-11-cnvn')
OUT=ROOT/'handoff_v43';OUT.mkdir(exist_ok=True)
DATA_HASH={
 'ds102_train.csv':'15202b19b2dc2c6c89afeb209024bb0dc77ddf927de44381a96c24a961d823a5',
 'ds102_dev.csv':'d895cc298b1e5b56bbb95c9f9b15610e5207a00cfd12cc79f32061e27b7a8ccf',
 'ds102_test.csv':'82c21afd6f8f4610912bc12ed19c3b6b1a586fb0dfa5d2296f6548ae4fc6b398',
 'ds102_samplesubmission.csv':'4cf7832636057152395b8b20302c11a691fd2ab35d50057ef43443f3d388424d'}

def digest(data):return hashlib.sha256(data).hexdigest()
def code(s):return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(True))
def markdown(s):return dict(cell_type='markdown',metadata={},source=s.splitlines(True))

def main():
    archive=OUT/'DS102_v43_inputs.zip';files={}
    for n,h in DATA_HASH.items():
        b=(RAW/n).read_bytes();assert digest(b)==h,n;files['data/'+n]=b
    files['baseline/reference_v3.npz']=(ROOT/'colab_reference.npz').read_bytes()
    mapping={
      'output_v11_experiment/oof.npz':'baseline/latent_oof.npz',
      'output_v11_experiment/manifest.json':'baseline/latent_oof_manifest.json',
      'output_v11_experimental/full_128.npz':'baseline/latent_full.npz',
      'output_v11_experimental/manifest.json':'baseline/latent_full_manifest.json'}
    backup_path=ROOT/'ds102_v27_prediction_checkpoints.zip'
    assert digest(backup_path.read_bytes())=='f5f8b223f27b86871b76ed6b2808b9527f24663a9181143f4db0076f86ceac07'
    with zipfile.ZipFile(backup_path) as backup:
        # The existing archive was verified when the user authorized its backup.
        for old,new in mapping.items():files[new]=backup.read(old)
    files['baseline/submission_best_v11.csv']=(ROOT/'output_v11_experimental/submission_v11_experimental.csv').read_bytes()
    baseline_metadata=json.loads(files['baseline/latent_oof_manifest.json'])
    assert digest(files['baseline/reference_v3.npz'])==baseline_metadata['baseline_manifest']['sha256']['colab_reference.npz']
    assert digest(files['baseline/submission_best_v11.csv'])=='05bffeff984bd4927c8dd3e9bd3613aef6a7abce303e6e1dcb7cfa4a12012059'
    versions=json.loads(files['baseline/latent_oof_manifest.json'])['versions']
    input_manifest=dict(format=1,files={n:digest(b) for n,b in files.items()},
        provenance='Four supplied competition CSVs and saved predictions from own train-only V11 models; no external model weights.')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,b in files.items():z.writestr(n,b)
        z.writestr('inputs_manifest.json',json.dumps(input_manifest,ensure_ascii=False,indent=2).encode())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for n,h in input_manifest['files'].items():assert digest(z.read(n))==h
    old_nb=json.loads((ROOT/'DS102_Colab_SVD128_experimental.ipynb').read_text())
    embed=next(c for c in old_nb['cells'] if c['cell_type']=='code' and 'modules=json.loads(' in ''.join(c['source']))
    tree=ast.parse(''.join(embed['source']))
    call=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='modules' for t in n.targets))
    modules=json.loads(ast.literal_eval(call.args[0]));modules.pop('run_submission.py')
    modules['cloud_checks.py']='''import os,platform
import numpy as np
def guard():
    if platform.system()!='Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Use hosted Google Colab, not a local runtime.')
def checked(p,n):
    p=np.asarray(p)
    assert p.shape==(n,4) and np.isfinite(p).all()
    assert (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-5)
    return p
'''
    for n in ['catalogue_svd_v40.py','lexical_chi2_v41.py']:
        modules[n]=(ROOT/n).read_text().replace('from experiment_v10 import guard,checked','from cloud_checks import guard,checked')
    for n in ['review_common.py','codex_checked_common.py']:
        modules[n]=(ROOT/'claude_review'/n).read_text()
    modules['handoff_v43.py']=(ROOT/'handoff_v43.py').read_text()
    for n,s in modules.items():ast.parse(s,filename=n)
    pack_sha=digest(archive.read_bytes())
    setup='''# Chỉ chạy trên máy chủ Colab. Không chọn local runtime.
import os,platform,sys,subprocess
from pathlib import Path
assert platform.system()=='Linux' and os.environ.get('COLAB_RELEASE_TAG'), 'Hãy chạy trên hosted Google Colab.'
ROOT43=Path('/content/ds102_v43');ROOT43.mkdir(exist_ok=True)
PY43=ROOT43/'venv/bin/python'
subprocess.check_call([sys.executable,'-m','venv','--without-pip','--system-site-packages',str(ROOT43/'venv')])
packages=PACKAGES
subprocess.check_call([sys.executable,'-m','pip','--python',str(PY43),'install','--quiet','--disable-pip-version-check',*packages])
print('Đã chuẩn bị môi trường Colab riêng; không cần khởi động lại kernel.')
'''.replace('PACKAGES',repr([k+'=='+v for k,v in versions.items()]))
    upload='''import hashlib,json,zipfile
from google.colab import files
PACK43=ROOT43/'DS102_v43_inputs.zip'
EXPECTED_PACK_SHA='__PACK_DIGEST__'
if not PACK43.exists():
    print('Chọn file DS102_v43_inputs.zip đã tải về máy.')
    uploaded43=files.upload()
    chosen43=next((b for n,b in uploaded43.items() if hashlib.sha256(b).hexdigest()==EXPECTED_PACK_SHA),None)
    assert chosen43 is not None,'Cần chọn đúng gói DS102_v43_inputs.zip đi kèm notebook này.'
    PACK43.write_bytes(chosen43)
    del uploaded43,chosen43
assert hashlib.sha256(PACK43.read_bytes()).hexdigest()==EXPECTED_PACK_SHA,'Gói đầu vào khác phiên bản notebook này.'
INPUTS43=ROOT43/'inputs';INPUTS43.mkdir(exist_ok=True)
with zipfile.ZipFile(PACK43) as z:
    for info in z.infolist():
        target43=(INPUTS43/info.filename).resolve()
        assert target43.is_relative_to(INPUTS43.resolve()),'Đường dẫn ZIP không hợp lệ'
    assert sum(i.file_size for i in z.infolist())<200_000_000
    z.extractall(INPUTS43)
manifest43=json.loads((INPUTS43/'inputs_manifest.json').read_text())
for n,h in manifest43['files'].items():
    assert hashlib.sha256((INPUTS43/n).read_bytes()).hexdigest()==h,n
print('Đã kiểm tra dữ liệu và các dự đoán nền từ những lượt train trước.')
'''.replace('__PACK_DIGEST__',pack_sha)
    embedding='import json\nSRC43=ROOT43/"source";SRC43.mkdir(exist_ok=True)\nmodules43=json.loads('+repr(json.dumps(modules,ensure_ascii=False))+')\nfor name,source in modules43.items():\n    (SRC43/name).write_text(source)\nprint("Đã ghi",len(modules43),"module mã nguồn.")\n'
    configuration='''# Giữ nguyên "lexical" cho lượt thử đầu tiên.
# Sau khi gửi báo cáo, có thể đổi sang "catalogue" theo hướng dẫn tiếp theo.
TRIAL43='lexical'
RUN_TAG43='lexical_01'
assert TRIAL43 in ['lexical','catalogue']
assert RUN_TAG43 and all(c.isalnum() or c in '_-' for c in RUN_TAG43)
OUT43=ROOT43/'runs'/RUN_TAG43;OUT43.mkdir(parents=True,exist_ok=True)
print('Lượt chạy:',TRIAL43,'| Thư mục:',OUT43)
print('Có 3 lượt kiểm tra. Chạy lại cùng cấu hình trong phiên này sẽ dùng checkpoint đã hoàn tất.')
'''
    run='''import subprocess,os,zipfile
command43=[str(PY43),'-u',str(SRC43/'handoff_v43.py'),'--inputs',str(INPUTS43),'--output',str(OUT43),'--trial',TRIAL43]
process43=subprocess.Popen(command43,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
    cwd=SRC43,env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
try:
    with (OUT43/'training.log').open('a') as log43:
        for line43 in process43.stdout:
            print(line43,end='',flush=True);log43.write(line43);log43.flush()
    exit43=process43.wait()
    print('\nKết thúc, mã trạng thái:',exit43)
except KeyboardInterrupt:
    process43.terminate()
    try:process43.wait(timeout=20)
    except subprocess.TimeoutExpired:process43.kill();process43.wait()
    print('Đã dừng lượt chạy. Checkpoint các fold hoàn tất được giữ trong runtime này.')
finally:
    FEEDBACK43=OUT43/'feedback_v43.zip'
    with zipfile.ZipFile(FEEDBACK43,'w',zipfile.ZIP_DEFLATED) as feedback43:
        for name43 in ['metrics_v43.json','run_manifest.json','environment.json','training.log','error.txt']:
            path43=OUT43/name43
            if path43.exists():feedback43.write(path43,name43)
    print('Báo cáo để gửi lại:',FEEDBACK43)
'''
    # The embedded Python must contain a literal backslash-n in this print call.
    run=run.replace("print('\nKết thúc, mã trạng thái:'", "print('\\nKết thúc, mã trạng thái:'")
    download='''import json,zipfile
from google.colab import files
# Tạo lại feedback kể cả khi bạn vừa dừng ô train.
FEEDBACK43=OUT43/'feedback_v43.zip'
with zipfile.ZipFile(FEEDBACK43,'w',zipfile.ZIP_DEFLATED) as feedback43:
    for name43 in ['metrics_v43.json','run_manifest.json','environment.json','training.log','error.txt']:
        path43=OUT43/name43
        if path43.exists():feedback43.write(path43,name43)
report43=OUT43/'metrics_v43.json'
if report43.exists():
    summary43=json.loads(report43.read_text())
    print('Trạng thái:',summary43.get('stage'))
    print('Chênh lệch F1 OOF:',summary43.get('delta'))
    print('Chênh lệch F1 dev:',summary43.get('dev_delta'))
    print('Đủ điều kiện xuất ứng viên:',summary43.get('submission_created',False))
files.download(str(FEEDBACK43))
candidate43=OUT43/'submission_v43_candidate.csv'
if candidate43.exists() and report43.exists() and summary43.get('submission_created',False):
    print('Ứng viên thử nghiệm; chưa biết điểm Kaggle. Chỉ nộp CSV, không nộp ZIP báo cáo.')
    files.download(str(candidate43))
else:
    print('Chưa có ứng viên đạt điều kiện. Gửi feedback_v43.zip để chỉnh lượt tiếp theo; không cần nộp lại baseline.')
'''
    cells=[markdown('''# DS102 — bạn chạy Colab, gửi báo cáo để chỉnh tiếp

**Chạy thử mới, chưa có điểm Kaggle. Không pretrained, không dùng dữ liệu ngoài.**

Tải notebook này và `DS102_v43_inputs.zip`. Mở notebook trên Google Colab, dùng hosted runtime CPU, chọn **Runtime → Run all / Thời gian chạy → Chạy tất cả**, rồi chọn gói ZIP khi được hỏi. Không chạy trên Mac/local runtime. Nếu phiên thí nghiệm cũ vẫn chạy, dừng phiên đó trước khi bắt đầu notebook mới.

Gói đầu vào chứa bốn CSV bạn đã cung cấp và dự đoán đã lưu từ các mô hình tự train trước đây. Đây là cache kết quả của dự án, không phải trọng số mô hình pretrained bên ngoài. Baseline phải tái tạo đúng file từng đạt **0.77436** trước khi thử nhánh mới.

Lượt mặc định thử đặc trưng từ/cụm ký tự được chọn riêng cho từng lớp, bổ sung cho SVD128/CatBoost. Chạy đủ 3 fold. Nếu tăng OOF ít nhất .002, ít nhất 2 fold tăng, fold xấu nhất không giảm quá .003, fake mới không giảm và bootstrap q05 dương, mới fit toàn train rồi kiểm tra dev tăng ít nhất .001. Không đổi gate sau khi xem kết quả. Dev luôn giữ riêng.

**Chạy xong gửi lại `feedback_v43.zip`.** Nếu xuất `submission_v43_candidate.csv` và bạn thử trên Kaggle, gửi thêm điểm chính xác. ZIP báo cáo không chứa review gốc, nhãn từng dòng, checkpoint dự đoán hoặc trọng số mô hình. Không tự submit.

CPU có thể cần vài chục phút hoặc lâu hơn; theo dõi các dòng `fold_start`, số cây và `fold_complete`. Nếu bị ngắt, chạy lại cùng cấu hình trong runtime còn dữ liệu để dùng lại fold đã hoàn tất. Checkpoint trên `/content` mất khi runtime bị xoá.

Notebook được tạo và kiểm tra cấu trúc/cú pháp; chưa chạy lại toàn bộ notebook sạch. Các bộ mã hoá mới đã qua micro-preflight trên Colab. OOF/dev đã được dùng nhiều lần; một ứng viên qua kiểm tra vẫn có thể giảm điểm leaderboard.

Mã nguồn pipeline được nhúng đầy đủ. Thư viện: scikit-learn (BSD-3-Clause), CatBoost (Apache-2.0); không tự công bố dự án hoặc dữ liệu.
'''),markdown('## 1. Chuẩn bị môi trường Colab'),code(setup),markdown('## 2. Tải gói đầu vào'),code(upload),markdown('## 3. Ghi mã nguồn'),code(embedding),markdown('## 4. Cấu hình lượt thử'),code(configuration),markdown('## 5. Chạy và theo dõi'),code(run),markdown('## 6. Tải báo cáo gửi lại'),code(download)]
    for i,c in enumerate(cells):
        c['id']=f'ds102-v43-{i}'
        if c['cell_type']=='code':ast.parse(''.join(c['source']),filename=f'cell{i}')
    nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python'),colab=dict(name='DS102_Colab_v43_handoff.ipynb',provenance=[])))
    notebook=OUT/'DS102_Colab_v43_handoff.ipynb';notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=1))
    source_zip=OUT/'DS102_v43_source.zip'
    with zipfile.ZipFile(source_zip,'w',zipfile.ZIP_DEFLATED) as z:
        for n,s in modules.items():z.writestr(n,s)
    release=dict(notebook=notebook.name,inputs=archive.name,inputs_bytes=archive.stat().st_size,
        sha256={p.name:digest(p.read_bytes()) for p in [notebook,archive,source_zip]},
        source_modules=len(modules),validation='Python AST for all cells/modules and ZIP/hash integrity; no local ML execution; fresh full notebook not executed')
    (OUT/'release_manifest.json').write_text(json.dumps(release,ensure_ascii=False,indent=2))
    print(json.dumps(release,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
