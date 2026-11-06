"""Build a standalone hosted-Colab fresh-training v6 notebook."""
import ast,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MODULES=['train_submit.py','improved_features.py','select_v2.py','selection_v5.py',
         'train_submit_v2.py','catboost_v3.py','experiment_v6.py','latent_v6.py','train_submit_v6.py']


def build():
    chosen=json.loads((ROOT/'output_v6'/'chosen_config.json').read_text())
    modules={n:(ROOT/n).read_text() for n in MODULES}
    for n,s in modules.items():ast.parse(s,filename=n)
    cells=[]
    def code(s):
        ast.parse(s);cells.append({'cell_type':'code','metadata':{},'source':s.splitlines(True),'execution_count':None,'outputs':[]})
    cells.append({'cell_type':'markdown','metadata':{},'source':['# DS102 v6 — không dùng pretrained\n',
        '\nChạy hosted Google Colab CPU; không dùng local runtime. Upload ZIP hoặc bốn CSV.\n',
        'Tất cả mô hình/từ điển/IDF/scaler học từ train, không gộp dev hoặc test.\n',
        'Notebook chứa toàn bộ source và train lại từ đầu, không phụ thuộc cache cũ.\n',
        'Cấu hình đã chọn từ grouped OOF trên train; notebook không chạy lại tìm kiếm CV.\n',
        'Thư viện nguồn mở: scikit-learn 1.7.2 (BSD-3-Clause), CatBoost 1.2.10 (Apache-2.0).\n',
        'Điểm Kaggle chỉ xác định sau khi nộp CSV.\n']})
    code("""import os,platform,sys,subprocess
assert platform.system()=='Linux' and os.environ.get('COLAB_RELEASE_TAG'),'Hosted Colab required'
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
subprocess.check_call([sys.executable,'-m','pip','install','-q','scikit-learn==1.7.2','catboost==1.2.10'])
from threadpoolctl import threadpool_limits
threadpool_limits(2)
""")
    code("""from google.colab import files
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
assert all((data_dir/n).exists() for n in required),'Missing CSV'
""")
    code('import json,sys\nfrom pathlib import Path\nsource=Path("/content/ds102_v6_source");source.mkdir(exist_ok=True)\n'
         +'modules=json.loads('+repr(json.dumps(modules,ensure_ascii=False))+')\n'
         +'for name,content in modules.items():\n    (source/name).write_text(content)\n'
         +'sys.path.insert(0,str(source))\n')
    code('from train_submit_v6 import main\nimport json\nselection=json.loads('+repr(json.dumps(chosen,ensure_ascii=False))+')\n'
         +'report=main(selection,data_dir=str(data_dir),reference_path=None)\n'
         +'print("Dev F1-macro:",report["dev"]["macro_f1"])\n')
    code("""from google.colab import files
files.download('/content/ds102_v6/submission_v6.csv')
files.download('/content/ds102_v6/metrics_v6.json')
""")
    for i,c in enumerate(cells):c['id']=f'ds102-v6-{i}'
    n={'nbformat':4,'nbformat_minor':5,'cells':cells,
       'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                   'language_info':{'name':'python'},'colab':{'name':'DS102_Colab_v6_no_pretrained.ipynb'}}}
    path=ROOT/'DS102_Colab_v6_no_pretrained.ipynb'
    path.write_text(json.dumps(n,ensure_ascii=False,indent=1));print(path)
    return path


if __name__=='__main__':build()
