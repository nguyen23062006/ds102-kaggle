"""Build replay notebook for the explicitly exploratory V52 CSV; original gate failures disclosed.

Source/JSON/ZIP integrity processing only; never import or fit ML locally.
"""
import ast,hashlib,json,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output_v52_exploratory'

def sha(data):return hashlib.sha256(data).hexdigest()

def main():
    report=json.loads((OUT/'metrics_v52_exploratory.json').read_text())
    assert report['stage']=='exploratory_candidate_ready_not_submitted'
    assert report['experimental'] and not report['original_cv_gate_passed']
    assert report['original_rejections_preserved'] and not report['additional_dev_gates_passed']
    assert report['dev_delta']>0
    csv=OUT/'submission_v52_exploratory.csv';assert sha(csv.read_bytes())==report['candidate_sha256']
    cvp=ROOT/'output_v50_experiment/metrics_v50.json';mp=ROOT/'output_v50_experiment/run_manifest.json'
    cv=json.loads(cvp.read_text());parent=json.loads(mp.read_text())
    assert not cv['gates']['oof'] and cv['bootstrap']['q025']>0
    with zipfile.ZipFile(ROOT/'handoff_v43/DS102_v43_source.zip') as z:
        modules={n:z.read(n).decode() for n in z.namelist()}
    for n in ['experiment_v48.py','supervised_svd_v48.py','experiment_v49.py','experiment_v50.py','finalize_v51_experimental.py','export_v52_exploratory.py']:
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
        '# DS102 V52 — CSV thử nghiệm, không pretrained\n',
        'Đã tạo CSV 3974dòng,11nhãn khác file tốt nhất từng đạtKaggle0.77436. **Chưa biết điểm Kaggle của V52.**\n',
        'Cấu hình cố định: chỉ đổi nhãn ngoài lớp fake review khi cả hai nhánh mô tả sản phẩm và SVD có trọng số đều đồng ý. Giữ nguyên toàn bộ quyết định lớp1 của bản nền.\n',
        'OOF0.774945→0.776409;dev0.726349→0.728946. **Ngưỡng CV0.002 và bộ điều kiện dev bổ sung không đạt**; các báo cáo bị loại vẫn được giữ. Đây là bản xuất để kiểm chứng thử nghiệm, không phải mô hình đã vượt mọi gate.\n',
        'Bootstrap OOF95%[0.000162;0.002795];dev95%[-0.001891;0.008294]. Dev vẫn có thể giảm theo khoảng bất định; OOF/dev đã dùng nhiều lần và bootstrap không loại bỏ thiên lệch chọn mô hình. Không bảo đảm tăng điểm leaderboard.\n',
        'Notebook fit lại hai nhánh từ đúng train trên hostedColabCPU,không pretrained hoặc dữ liệu ngoài. Dev/test không được dùng fit. Dùng **DS102_v43_inputs.zip** đã có (19MB), chứa4CSV và dự đoán nền V11 tự train trước đây.\n',
        'TảiZIP bằng bảng Tệp bên trái rồi chọn Runtime→Run all. Notebook nhậnZIP đã tải; nếu chưa có sẽ hỏi tải lên. Giữtab mở. Không tự submitKaggle.\n',
        'Gửi feedback_v52.zip và điểm Kaggle nếu bạn nộp thửCSV. Nguồn đầy đủ được nhúng; kiểm tra source/cú pháp đã chạy. Chưa chạy lại notebook sạch trọn vẹn; notebook fit mô hình cuối và dùng lại bằng chứngCV đã lưu,không chạy lại toàn bộ nghiên cứuCV.\n']
    embedding="import json\nfrom pathlib import Path\nROOT51=Path('/content/ds102_v51');ROOT51.mkdir(exist_ok=True)\nSRC43=ROOT51/'source';SRC43.mkdir(exist_ok=True)\nmodules51=json.loads("+repr(json.dumps(modules,ensure_ascii=False))+')\n'
    embedding+="for name,source in modules51.items():\n    (SRC43/name).write_text(source)\n"
    embedding+="parent51=Path('/content/ds102_v50/run');parent51.mkdir(parents=True,exist_ok=True)\n"
    embedding+="(parent51/'metrics_v50.json').write_text("+repr(cvp.read_text())+")\n"
    embedding+="(parent51/'run_manifest.json').write_text("+repr(mp.read_text())+")\n"
    embedding+="print('Đã ghi mã nguồn và bản sao bằng chứng CV; không có nhãn test.')\n"
    nb['cells'][6]['source']=embedding.splitlines(True)
    nb['cells'][8]['source']=["OUT43=Path('/content/ds102_v52_exploratory');OUT43.mkdir(parents=True,exist_ok=True)\n",
        "print('V52: fit lại hai nhánh từ train; báo cáo rõ các gate gốc không đạt và xuất CSV thử nghiệm khi các mức điểm quan sát đều tăng.')\n"]
    run=''.join(nb['cells'][10]['source'])
    old="command43=[str(PY43),'-u',str(SRC43/'handoff_v43.py'),'--inputs',str(INPUTS43),'--output',str(OUT43),'--trial',TRIAL43]"
    assert old in run
    run=run.replace(old,"command43=[str(PY43),'-u',str(SRC43/'export_v52_exploratory.py')]")
    run=run.replace('feedback_v43.zip','feedback_v52.zip').replace('metrics_v43.json','metrics_v52_exploratory.json')
    nb['cells'][10]['source']=run.splitlines(True)
    nb['cells'][12]['source']="import json,zipfile\nfrom google.colab import files\nFEEDBACK43=OUT43/'feedback_v52.zip'\nwith zipfile.ZipFile(FEEDBACK43,'w',zipfile.ZIP_DEFLATED) as z:\n    for name in ['metrics_v52_exploratory.json','run_manifest.json','training.log','error.txt']:\n        p=OUT43/name\n        if p.exists():z.write(p,name)\nreport43=OUT43/'metrics_v52_exploratory.json'\nsummary43=json.loads(report43.read_text()) if report43.exists() else {}\nprint('Trạng thái:',summary43.get('stage'))\nprint('Chênh lệch dev:',summary43.get('dev_delta'))\nprint('Gate CV gốc đạt:',summary43.get('original_cv_gate_passed'))\nprint('Gate dev bổ sung đạt:',summary43.get('additional_dev_gates_passed'))\nfiles.download(str(FEEDBACK43))\ncandidate43=OUT43/'submission_v52_exploratory.csv'\nif candidate43.exists() and summary43.get('stage')=='exploratory_candidate_ready_not_submitted':\n    print('CSV thử nghiệm: chưa qua các gate ngặt ban đầu và chưa biết điểm Kaggle. Chỉ nộp CSV nếu bạn muốn kiểm chứng.')\n    files.download(str(candidate43))\nelse:\n    print('Chưa có CSV. Gửi feedback_v52.zip để kiểm tra log.')\n".splitlines(True)
    for i,c in enumerate(nb['cells']):
        c['id']=f'ds102-v52-exploratory-{i}'
        if c['cell_type']=='code':
            c['outputs']=[];c['execution_count']=None
            ast.parse(''.join(c['source']),filename=f'cell{i}')
    nb['metadata']['colab']['name']='DS102_Colab_v52_exploratory.ipynb'
    path=OUT/'DS102_Colab_v52_exploratory.ipynb';path.write_text(json.dumps(nb,ensure_ascii=False,indent=1))
    archive=OUT/'DS102_v52_source.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for n,s in modules.items():z.writestr(n,s)
        z.writestr('cv_evidence/metrics_v50.json',cvp.read_bytes())
        z.writestr('cv_evidence/run_manifest.json',mp.read_bytes())
        z.writestr('metrics_v52_exploratory.json',(OUT/'metrics_v52_exploratory.json').read_bytes())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    release=dict(stage='exploratory_not_confirmed_on_kaggle',original_cv_gate_passed=False,
        baseline_user_reported_kaggle=.77436,source_modules=len(modules),
        required_inputs='handoff_v43/DS102_v43_inputs.zip',
        sha256={p.name:sha(p.read_bytes()) for p in [path,archive,csv,OUT/'metrics_v52_exploratory.json']},
        validation='Actual component fits and scoring on hostedColab; replay notebook/source AST and ZIP integrity checked; clean replay notebook not rerun end-to-end.')
    (OUT/'release_manifest.json').write_text(json.dumps(release,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(release,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
