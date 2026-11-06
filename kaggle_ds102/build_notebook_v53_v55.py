"""Package the three frozen hypotheses for a manual Colab run."""
import ast,base64,hashlib,json,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'handoff_v53_v55';OUT.mkdir(exist_ok=True)
book=json.loads((ROOT/'handoff_v53_v54/DS102_Colab_v53_v54.ipynb').read_text())
modules={n:(ROOT/n).read_text() for n in ['experiment_v55.py','protocol_v55.py']}
for n,s in modules.items():ast.parse(s,filename=n)
book['cells'][0]['source'] += ['\n\nGiả thuyết thứ ba, V55: SVM-RBF với đặc trưng SVD128 và quan hệ review/sản phẩm. ',
    'V55 chạy nếu hai giả thuyết trước đều bị loại. Không pretrained và giữ nguyên các ngưỡng chọn. ',
    'SVD chỉ fit phần train ngoài; scaler/SVM fit lại trong mỗi fold hiệu chỉnh nội bộ. ',
    'SVD không được fit lại ở từng fold hiệu chỉnh nội bộ, nên không tuyên bố calibration được lồng hoàn toàn ở mọi bước. ',
    'Mọi bước đều không fit dòng đánh giá ngoài.\n']

def cell(s):
    ast.parse(s)
    return dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=None,outputs=[])

deploy=cell(f'''import json,hashlib
ROOT55=Path('/content/ds102_v55');SRC55=ROOT55/'source';SRC55.mkdir(parents=True,exist_ok=True)
modules55=json.loads({json.dumps(modules)!r})
hashes55={json.dumps({n:hashlib.sha256(s.encode()).hexdigest() for n,s in modules.items()})}
for name,source in modules55.items():
    assert hashlib.sha256(source.encode()).hexdigest()==hashes55[name]
    (SRC55/name).write_text(source)
print('Đã ghi mã nguồn cố định của V55.')
''')
book['cells'].insert(4,deploy)
followup=cell('''ready=[]
for version,folder in [(53,ROOT/'run'),(54,ROOT54/'run')]:
    p=folder/f'metrics_v{version}.json'
    if p.exists() and json.loads(p.read_text())['stage']=='candidate_ready_not_submitted':ready.append(version)
if ready:
    print('Đã có ứng viên vượt đủ ngưỡng:',ready,'; không chạy thêm V55 trong lượt này.')
else:
    for version,folder in [(53,ROOT/'run'),(54,ROOT54/'run')]:
        p=folder/f'metrics_v{version}.json'
        assert p.exists() and json.loads(p.read_text())['stage'].startswith(('rejected_on_','pruned_on_')), 'Xử lý lỗi trước khi chạy V55.'
    running=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
    assert not any(str(SRC55/'experiment_v55.py') in s for s in running),'V55 đang chạy.'
    for folder,extra in [('smoke',['--smoke']),('run',[])]:
        cmd=[str(PY),'-u',str(SRC55/'experiment_v55.py'),'--inputs',str(INPUTS),'--output',str(ROOT55/folder),*extra]
        with (ROOT55/(folder+'.log')).open('a') as log:
            proc55=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env,cwd=SRC55)
            for line in proc55.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            rc55=proc55.wait()
        assert rc55==0,'V55 bị lỗi; tải feedback để kiểm tra.'
        if folder=='smoke':assert json.loads((ROOT55/'smoke/metrics_v55.json').read_text())['stage']=='smoke_passed'
    print('Trạng thái V55:',json.loads((ROOT55/'run/metrics_v55.json').read_text())['stage'])
''')
book['cells'].insert(-1,followup)
book['cells'][-1]=cell('''from google.colab import files
feedback=ROOT/'feedback_v53_v55.zip';candidates=[]
with zipfile.ZipFile(feedback,'w',zipfile.ZIP_DEFLATED) as z:
    for version,folder in [(53,ROOT/'run'),(54,ROOT54/'run'),(55,ROOT55/'run')]:
        for name in [f'metrics_v{version}.json','run_manifest.json','error.txt']:
            p=folder/name
            if p.exists():z.write(p,f'v{version}/{name}')
        p=folder/f'metrics_v{version}.json'
        if p.exists() and json.loads(p.read_text()).get('stage')=='candidate_ready_not_submitted':
            candidate=folder/f'submission_v{version}_candidate.csv'
            if candidate.exists():z.write(candidate,f'v{version}/{candidate.name}');candidates.append(candidate)
files.download(str(feedback))
for candidate in candidates:files.download(str(candidate))
if not candidates:print('Chưa có ứng viên vượt đủ ngưỡng; giữ V11 làm mốc.')
''')
book['metadata']['colab']['name']='DS102_Colab_v53_v55.ipynb'
for c in book['cells']:
    if c['cell_type']=='code':ast.parse(''.join(c['source']))
(OUT/'DS102_Colab_v53_v55.ipynb').write_text(json.dumps(book,ensure_ascii=False,indent=1)+'\n')
with zipfile.ZipFile(OUT/'DS102_v53_v55_source.zip','w',zipfile.ZIP_DEFLATED) as z:
    with zipfile.ZipFile(ROOT/'handoff_v53_v54/DS102_v53_v54_source.zip') as base:
        for name in base.namelist():z.writestr(name,base.read(name))
    for name,source in modules.items():z.writestr('v55_source/'+name,source)
with zipfile.ZipFile(OUT/'DS102_v53_v55_source.zip') as z:assert z.testzip() is None
manifest=dict(stage='V53_V54_rejected_V55_result_awaiting_MCP',smoke_passed_versions=[53,54,55],
    clean_notebook_end_to_end_executed=False,pretrained=False,
    files={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['DS102_Colab_v53_v55.ipynb','DS102_v53_v55_source.zip']})
(OUT/'release_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(OUT/'DS102_Colab_v53_v55.ipynb')
