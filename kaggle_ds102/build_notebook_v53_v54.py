"""Reproducible two-hypothesis Colab handoff, no training during packaging."""
import ast,base64,hashlib,json,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'handoff_v53_v54';OUT.mkdir(exist_ok=True)
book=json.loads((ROOT/'handoff_v53/DS102_Colab_v53.ipynb').read_text())
source54=(ROOT/'experiment_v54.py').read_bytes()
sha54=hashlib.sha256(source54).hexdigest()
book['cells'][0]['source'] += ['\n\nGiả thuyết thứ hai, V54: ComplementNB trên TF-IDF từ/ký tự, hiệu chỉnh sigmoid bằng 3 fold nhóm nội bộ. ',
    'V54 tận dụng đúng baseline nhóm của V53 và chạy sau khi V53 bị loại; không chạy khi V53 đã tạo ứng viên đủ ngưỡng. ',
    'Cả hai giữ nguyên ngưỡng CV/dev đã chốt. Không có bảo đảm tăng điểm Kaggle.\n']

def cell(s):
    ast.parse(s)
    return dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=None,outputs=[])

deploy=cell(f'''import base64,hashlib
ROOT54=Path('/content/ds102_v54');SRC54=ROOT54/'source';SRC54.mkdir(parents=True,exist_ok=True)
source54=base64.b64decode({base64.b64encode(source54).decode()!r})
assert hashlib.sha256(source54).hexdigest()=={sha54!r}
(SRC54/'experiment_v54.py').write_bytes(source54)
print('Đã ghi nguồn V54, giữ nguyên thư mục nguồn V53.')
''')
book['cells'].insert(3,deploy)
followup=cell('''report53=json.loads((OUT/'metrics_v53.json').read_text())
if report53['stage']=='candidate_ready_not_submitted':
    print('V53 đã vượt các ngưỡng; không chạy thêm V54 trong lượt này.')
else:
    assert report53['stage'].startswith(('rejected_on_','pruned_on_')), 'Xử lý lỗi V53 trước khi chạy V54.'
    running=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
    assert not any(str(SRC54/'experiment_v54.py') in s for s in running), 'V54 đang chạy.'
    for folder,extra in [('smoke',['--smoke']),('run',[])]:
        cmd=[str(PY),'-u',str(SRC54/'experiment_v54.py'),'--inputs',str(INPUTS),'--output',str(ROOT54/folder),*extra]
        with (ROOT54/(folder+'.log')).open('a') as log:
            proc54=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env,cwd=SRC54)
            for line in proc54.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            rc54=proc54.wait()
        assert rc54==0,'V54 bị lỗi; tải feedback để kiểm tra.'
        if folder=='smoke':assert json.loads((ROOT54/'smoke/metrics_v54.json').read_text())['stage']=='smoke_passed'
    print('Trạng thái V54:',json.loads((ROOT54/'run/metrics_v54.json').read_text())['stage'])
''')
book['cells'].insert(-1,followup)
book['cells'][-1]=cell('''from google.colab import files
feedback=ROOT/'feedback_v53_v54.zip'
candidates=[]
with zipfile.ZipFile(feedback,'w',zipfile.ZIP_DEFLATED) as z:
    for version,folder in [(53,ROOT/'run'),(54,ROOT54/'run')]:
        for name in [f'metrics_v{version}.json','run_manifest.json','error.txt']:
            p=folder/name
            if p.exists():z.write(p,f'v{version}/{name}')
        mp=folder/f'metrics_v{version}.json'
        if mp.exists() and json.loads(mp.read_text()).get('stage')=='candidate_ready_not_submitted':
            p=folder/f'submission_v{version}_candidate.csv'
            if p.exists():z.write(p,f'v{version}/{p.name}');candidates.append(p)
files.download(str(feedback))
for candidate in candidates:files.download(str(candidate))
if not candidates:print('Chưa có ứng viên vượt đủ ngưỡng; giữ V11 làm mốc.')
''')
book['metadata']['colab']['name']='DS102_Colab_v53_v54.ipynb'
for c in book['cells']:
    if c['cell_type']=='code':ast.parse(''.join(c['source']))
(OUT/'DS102_Colab_v53_v54.ipynb').write_text(json.dumps(book,ensure_ascii=False,indent=1)+'\n')
with zipfile.ZipFile(OUT/'DS102_v53_v54_source.zip','w',zipfile.ZIP_DEFLATED) as z:
    with zipfile.ZipFile(ROOT/'handoff_v53/DS102_v53_source.zip') as base:
        for n in base.namelist():
            ast.parse(base.read(n).decode(),filename=n);z.writestr('v53_source/'+n,base.read(n))
    ast.parse(source54.decode());z.writestr('v54_source/experiment_v54.py',source54)
with zipfile.ZipFile(OUT/'DS102_v53_v54_source.zip') as z:assert z.testzip() is None
manifest=dict(stage='prepared_while_research_runs',smoke_v53_passed=True,smoke_v54_passed=False,
    clean_notebook_end_to_end_executed=False,pretrained=False,
    files={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['DS102_Colab_v53_v54.ipynb','DS102_v53_v54_source.zip']})
(OUT/'release_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(OUT/'DS102_Colab_v53_v54.ipynb')
