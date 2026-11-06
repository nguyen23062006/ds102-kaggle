"""Build a self-contained, fresh-training Colab notebook from saved selection."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
MODULES=['train_submit.py','improved_features.py','select_v2.py','train_submit_v2.py',
         'catboost_v3.py','experiment_v5.py','selection_v5.py','temperature_v5.py',
         'fasttext_scratch.py','stacking_v5.py','finalize_v5.py']


def build(selection_path=None):
    selection_path=Path(selection_path or ROOT/'output_v5'/'chosen_config.json')
    chosen=json.loads(selection_path.read_text())
    modules={name:(ROOT/name).read_text() for name in MODULES}
    for name,source in modules.items(): ast.parse(source,filename=name)
    cells=[]
    def markdown(source):
        cells.append(dict(cell_type='markdown',metadata={},source=source.splitlines(True)))
    def code(source):
        ast.parse(source)
        cells.append(dict(cell_type='code',metadata={},source=source.splitlines(True),
                          execution_count=None,outputs=[]))
    markdown('''# DS102 v5 — không dùng pretrained

Chạy trên **hosted Google Colab**, CPU đủ dùng. Tất cả mô hình học từ train;
không dùng pretrained embedding, pretrained model hoặc dữ liệu ngoài cuộc thi.
Thư viện: scikit-learn (BSD-3-Clause), CatBoost (Apache-2.0), và fastText (MIT)
nếu cấu hình được chọn sử dụng fastText. fastText được biên dịch từ mã nguồn
chính thức v0.9.2; không tải pretrained model hoặc pretrained vector.

Upload `colab_data.zip` hoặc bốn CSV. Notebook chứa toàn bộ mã pipeline và
huấn luyện lại từ đầu, không phụ thuộc cache dự đoán của lần thử trước.
Cấu hình đã chọn bằng OOF trên train; notebook này không chạy lại tìm kiếm CV.
Điểm Kaggle chỉ biết sau khi nộp CSV, không thể bảo đảm tăng điểm.
''')
    code('''import os,platform,sys,subprocess
assert platform.system()=='Linux' and os.environ.get('COLAB_RELEASE_TAG'), 'Use hosted Colab, not local runtime'
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
subprocess.check_call([sys.executable,'-m','pip','install','-q','scikit-learn==1.7.2','catboost==1.2.10'])
from threadpoolctl import threadpool_limits
threadpool_limits(2)
''')
    code('''from google.colab import files
from pathlib import Path
import zipfile
uploaded=files.upload()
data_dir=Path('/content/ds102_data');data_dir.mkdir(exist_ok=True)
required={'ds102_train.csv','ds102_dev.csv','ds102_test.csv','ds102_samplesubmission.csv'}
for name,content in uploaded.items():
    if name.endswith('.zip'):
        with zipfile.ZipFile(name) as z:
            for member in z.namelist():
                if Path(member).name in required:
                    (data_dir/Path(member).name).write_bytes(z.read(member))
    elif Path(name).name in required:
        (data_dir/Path(name).name).write_bytes(content)
assert all((data_dir/n).exists() for n in required),'Missing one of four CSVs'
''')
    code('import json,sys\nfrom pathlib import Path\nsource=Path("/content/ds102_v5_source");source.mkdir(exist_ok=True)\n'
         +'modules=json.loads('+repr(json.dumps(modules,ensure_ascii=False))+')\n'
         +'for name,content in modules.items():\n    (source/name).write_text(content)\n'
         +'sys.path.insert(0,str(source))\n')
    code('from finalize_v5 import main\nimport json\n'
         +'chosen=json.loads('+repr(json.dumps(chosen,ensure_ascii=False))+')\n'
         +'report=main(chosen["selection"],chosen["family"],data_dir=str(data_dir),reference_path=None)\n'
         +'print("Dev F1-macro:",report["dev"]["macro_f1"])\n')
    code('''from google.colab import files
files.download('/content/ds102_v5/submission_v5.csv')
files.download('/content/ds102_v5/metrics_v5.json')
''')
    notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,
        metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                  'language_info':{'name':'python'},'colab':{'name':'DS102_Colab_v5_no_pretrained.ipynb'}})
    for i,cell in enumerate(cells): cell['id']=f'ds102-v5-{i}'
    path=ROOT/'DS102_Colab_v5_no_pretrained.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1))
    print(path)
    return path


if __name__=='__main__':build()
