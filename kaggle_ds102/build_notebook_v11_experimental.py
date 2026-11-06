"""Build source-complete fresh-training notebook; no local ML execution."""
import ast,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MODULES=['train_submit.py','improved_features.py','select_v2.py','selection_v5.py',
    'train_submit_v2.py','catboost_v3.py','experiment_v6.py','latent_v6.py','train_submit_v6.py']


def build():
    modules={n:(ROOT/n).read_text() for n in MODULES}
    assert 'n_components=64' in modules['latent_v6.py']
    modules['latent_v6.py']=modules['latent_v6.py'].replace('n_components=64','n_components=128')
    modules['train_submit_v6.py']=modules['train_submit_v6.py'].replace('submission_v6.csv','submission_v11_experimental.csv').replace('metrics_v6.json','metrics_v11_experimental.json')
    runner='''import json,os,platform
from pathlib import Path
assert platform.system()=='Linux' and os.environ.get('COLAB_RELEASE_TAG'),'Hosted Colab required'
from train_submit_v6 import main
selection={'recipe':{'kind':'latent','trees':800,'alpha':.3},'weights':[1.,1.2,1.2,.9],
           'name':'svd128_experimental','qualification':'Prior V11 CV promotion gates failed; experimental submission, Kaggle score unknown.'}
out=Path('/content/ds102_v11_experimental')
report=main(selection,data_dir='/content/ds102_v11_data',output_dir=str(out),reference_path=None)
report['qualification']=selection['qualification']
report['inference_note']='Fresh full pipeline fit; earlier exploratory CSV used cached v3 baseline. Numerical differences are possible.'
(out/'metrics_v11_experimental.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('EXPERIMENTAL CSV:',out/'submission_v11_experimental.csv')
print('Dev F1:',report['dev']['macro_f1'])
'''
    modules['run_submission.py']=runner
    for name,source in modules.items():ast.parse(source,filename=name)
    cells=[dict(cell_type='markdown',metadata={},source=[
        '# DS102 SVD128 — bản thử nghiệm, chưa xác nhận tăng leaderboard\n',
        'V6 đã đạt Kaggle 0.77269. V11 tăng OOF khoảng0.0018 nhưng KHÔNG vượt gate bootstrap và ngưỡng macro đã chốt.\n',
        'Notebook này tạo ứng viên thăm dò, không khẳng định tốt hơn v6. Không tự nộp Kaggle.\n',
        'Huấn luyện lại toàn bộ pipeline từ train trên hosted Colab, không cache prediction, không pretrained, không gộp dev.\n',
        'Dùng ZIP dữ liệu cũ hoặc bốn CSV cuộc thi. Mọi file khác trong ZIP bị bỏ qua.\n',
        'Thư viện nguồn mở: scikit-learn BSD-3-Clause, CatBoost Apache-2.0. Source đầy đủ được nhúng bên dưới.\n'])]
    def code(s):
        ast.parse(s);cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=None,outputs=[]))
    code('''import os,platform,sys,subprocess
assert platform.system()=='Linux' and os.environ.get('COLAB_RELEASE_TAG'),'Hosted Colab required'
subprocess.check_call([sys.executable,'-m','pip','install','-q','numpy==2.4.6','scipy==1.16.3','pandas==2.2.3','scikit-learn==1.7.2','catboost==1.2.10'])
''')
    code('''from google.colab import files
from pathlib import Path
import zipfile
uploaded=files.upload()
data_dir=Path('/content/ds102_v11_data');data_dir.mkdir(exist_ok=True)
required={'ds102_train.csv','ds102_dev.csv','ds102_test.csv','ds102_samplesubmission.csv'}
for name,content in uploaded.items():
    if name.lower().endswith('.zip'):
        with zipfile.ZipFile(name) as z:
            for member in z.namelist():
                if Path(member).name in required:
                    (data_dir/Path(member).name).write_bytes(z.read(member))
    elif Path(name).name in required:
        (data_dir/Path(name).name).write_bytes(content)
assert all((data_dir/n).exists() for n in required),'Missing competition CSV'
del uploaded
''')
    code('import json\nfrom pathlib import Path\nsource=Path("/content/ds102_v11_experimental_source");source.mkdir(exist_ok=True)\n'
        +'modules=json.loads('+repr(json.dumps(modules,ensure_ascii=False))+')\n'
        +'for name,content in modules.items():\n    (source/name).write_text(content)\n')
    code('''import os,sys,subprocess
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
subprocess.run([sys.executable,'-u',str(source/'run_submission.py')],cwd=str(source),env=env,check=True)
''')
    code('''from google.colab import files
files.download('/content/ds102_v11_experimental/submission_v11_experimental.csv')
files.download('/content/ds102_v11_experimental/metrics_v11_experimental.json')
''')
    for i,c in enumerate(cells):c['id']=f'ds102-svd128-experimental-{i}'
    nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')))
    path=ROOT/'DS102_Colab_SVD128_experimental.ipynb';path.write_text(json.dumps(nb,ensure_ascii=False,indent=1));print(path)


if __name__=='__main__':build()
