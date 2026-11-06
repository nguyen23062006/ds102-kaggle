"""Build replay notebook only after the actual V51 exploratory CSV is ready.

Source/JSON/ZIP integrity processing only; never import or fit ML locally.
"""
import ast,hashlib,json,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output_v51_experimental'

def sha(data):return hashlib.sha256(data).hexdigest()

def main():
    report=json.loads((OUT/'metrics_v51_experimental.json').read_text())
    assert report['stage']=='experimental_candidate_ready_not_submitted'
    assert report['experimental'] and not report['original_cv_gate_passed']
    assert all(report['additional_dev_gates'].values())
    csv=OUT/'submission_v51_experimental.csv';assert sha(csv.read_bytes())==report['candidate_sha256']
    cvp=ROOT/'output_v50_experiment/metrics_v50.json';mp=ROOT/'output_v50_experiment/run_manifest.json'
    cv=json.loads(cvp.read_text());parent=json.loads(mp.read_text())
    assert not cv['gates']['oof'] and cv['bootstrap']['q025']>0
    with zipfile.ZipFile(ROOT/'handoff_v43/DS102_v43_source.zip') as z:
        modules={n:z.read(n).decode() for n in z.namelist()}
    for n in ['experiment_v48.py','supervised_svd_v48.py','experiment_v49.py','experiment_v50.py','finalize_v51_experimental.py']:
        modules[n]=(ROOT/n).read_text()
    for n,h in parent['sources'].items():assert sha(modules[n].encode())==h,n
    for n,s in modules.items():ast.parse(s,filename=n)
    nb=json.loads((ROOT/'handoff_v43/DS102_Colab_v43_handoff.ipynb').read_text())
    upload=''.join(nb['cells'][4]['source'])
    marker="if not PACK43.exists():\n    print('Chọn file DS102_v43_inputs.zip đã tải về máy.')"
    assert marker in upload
    sidebar="""# Hỗ trợ ZIP đã tải bằng bảng Tệp bên trái khi tiện ích trong ô bị chặn.
import shutil
if not PACK43.exists():
    for existing51 in Path('/content').glob('DS102_v43_inputs*.zip'):
        if hashlib.sha256(existing51.read_bytes()).hexdigest()==EXPECTED_PACK_SHA:
            shutil.copy2(existing51,PACK43)
            break
"""
    upload=upload.replace(marker,sidebar+marker)
    nb['cells'][4]['source']=upload.splitlines(True)
    nb['cells'][0]['source']=[
        '# DS102 V51 — ứng viên thử nghiệm, không pretrained\n',
        'Mốc Kaggle đã biết: **0.77436**. Chưa biết điểm Kaggle của V51.\n',
        'V50 tăng OOF0.001464,2/3fold dương,bootstrap95%[0.000162;0.002795],giữ nguyên dự đoán fake review. Tuy nhiên ngưỡng tăng OOF0.002 ban đầu **không đạt**; kết quả bị loại đó vẫn được giữ.\n',
        'V51 là bước thử nghiệm bổ sung: chỉ xuất CSV nếu dev và dev bỏ trùng train đều tăng ít nhất0.003,bootstrapdevq05>0. Không coi các điều kiện này là bằng chứng hết thiên lệch chọn mô hình hoặc bảo đảm tăng Kaggle.\n',
        'Notebook fit lại hai nhánh catalogue/SVD có trọng số từ đúng train trên hostedColabCPU. Dev/test không được dùng để fit. Không pretrained hoặc dữ liệu ngoài.\n',
        'Dùng lại **DS102_v43_inputs.zip** đã có (19MB), chứa4CSV và dự đoán nền V11 tự train trước đây. Không cần ghép nối MCP để tự chạy notebook. Mã nguồn đầy đủ được nhúng.\n',
        'Bạn có thể tải ZIP bằng bảng Tệp bên trái trước rồi chọn Runtime→Run all. Notebook nhận file ZIP đó; nếu chưa có thì sẽ hỏi tải lên. Giữtab mở trong lúc máy chủ Colab huấn luyện. Notebook không tự submit Kaggle.\n',
        'Chạy xong gửi feedback_v51.zip; nếu cóCSV mới và bạn nộp thử, gửi thêm điểm Kaggle chính xác. Notebook được kiểm tra source/cú pháp; chưa chạy lại một phiên notebook sạch từ đầu đến cuối.\n']
    embedding="import json\nfrom pathlib import Path\nROOT51=Path('/content/ds102_v51');ROOT51.mkdir(exist_ok=True)\nSRC43=ROOT51/'source';SRC43.mkdir(exist_ok=True)\nmodules51=json.loads("+repr(json.dumps(modules,ensure_ascii=False))+')\n'
    embedding+="for name,source in modules51.items():\n    (SRC43/name).write_text(source)\n"
    embedding+="parent51=Path('/content/ds102_v50/run');parent51.mkdir(parents=True,exist_ok=True)\n"
    embedding+="(parent51/'metrics_v50.json').write_text("+repr(cvp.read_text())+")\n"
    embedding+="(parent51/'run_manifest.json').write_text("+repr(mp.read_text())+")\n"
    embedding+="print('Đã ghi mã nguồn và bản sao bằng chứng CV; không có nhãn test.')\n"
    nb['cells'][6]['source']=embedding.splitlines(True)
    nb['cells'][8]['source']=["OUT43=Path('/content/ds102_v51/run');OUT43.mkdir(parents=True,exist_ok=True)\n",
        "print('V51 thử nghiệm: fit lại hai nhánh từ train, rồi áp dụng điều kiện dev đã chốt.')\n"]
    run=''.join(nb['cells'][10]['source'])
    old="command43=[str(PY43),'-u',str(SRC43/'handoff_v43.py'),'--inputs',str(INPUTS43),'--output',str(OUT43),'--trial',TRIAL43]"
    assert old in run
    run=run.replace(old,"command43=[str(PY43),'-u',str(SRC43/'finalize_v51_experimental.py')]")
    run=run.replace('feedback_v43.zip','feedback_v51.zip').replace('metrics_v43.json','metrics_v51_experimental.json')
    nb['cells'][10]['source']=run.splitlines(True)
    download=''.join(nb['cells'][12]['source']).replace('feedback_v43.zip','feedback_v51.zip').replace('metrics_v43.json','metrics_v51_experimental.json').replace('submission_v43_candidate.csv','submission_v51_experimental.csv')
    nb['cells'][12]['source']=download.splitlines(True)
    for i,c in enumerate(nb['cells']):
        c['id']=f'ds102-v51-experimental-{i}'
        if c['cell_type']=='code':
            c['outputs']=[];c['execution_count']=None
            ast.parse(''.join(c['source']),filename=f'cell{i}')
    nb['metadata']['colab']['name']='DS102_Colab_v51_experimental.ipynb'
    path=OUT/'DS102_Colab_v51_experimental.ipynb';path.write_text(json.dumps(nb,ensure_ascii=False,indent=1))
    archive=OUT/'DS102_v51_source.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for n,s in modules.items():z.writestr(n,s)
        z.writestr('cv_evidence/metrics_v50.json',cvp.read_bytes())
        z.writestr('cv_evidence/run_manifest.json',mp.read_bytes())
        z.writestr('metrics_v51_experimental.json',(OUT/'metrics_v51_experimental.json').read_bytes())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    release=dict(stage='experimental_not_confirmed_on_kaggle',original_cv_gate_passed=False,
        baseline_user_reported_kaggle=.77436,source_modules=len(modules),
        required_inputs='handoff_v43/DS102_v43_inputs.zip',
        sha256={p.name:sha(p.read_bytes()) for p in [path,archive,csv,OUT/'metrics_v51_experimental.json']},
        validation='Actual component fits and scoring on hostedColab; replay notebook/source AST and ZIP integrity checked; clean replay notebook not rerun end-to-end.')
    (OUT/'release_manifest.json').write_text(json.dumps(release,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(release,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
