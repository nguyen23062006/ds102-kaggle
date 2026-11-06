"""Build a fresh-training V26 notebook after an actual candidate has passed screening.

This builder only edits source/JSON; it does not import or execute ML modules.
"""
import ast,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def build():
    evidence=json.loads((ROOT/'output_v26_experiment/report.json').read_text())
    assert evidence['stage']=='candidate_ready_not_submitted','V26 has not qualified'
    nb=json.loads((ROOT/'DS102_Colab_SVD128_experimental.ipynb').read_text())
    embedding=next(c for c in nb['cells'] if c['cell_type']=='code' and 'modules=json.loads(' in ''.join(c['source']))
    tree=ast.parse(''.join(embedding['source']))
    call=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='modules' for t in n.targets))
    modules=json.loads(ast.literal_eval(call.args[0]))
    driver=modules.pop('train_submit_v6.py')
    old="    return [.5*softmax(3*s,axis=1)+.5*m.predict_proba(cat_features(d)) for s,d in zip(margins,frames)]"
    assert driver.count(old)==1
    new="""    old_cat=[m.predict_proba(cat_features(d)) for d in frames]
    del m;gc.collect()
    m=make_cat('SqrtBalanced',iterations=500)
    m.set_params(thread_count=2,feature_calcers=['BoW:top_tokens_count=2000','NaiveBayes','BM25'])
    m.fit(cat_features(train),train.spam_label.to_numpy(),verbose=100)
    return [.5*softmax(3*s,axis=1)+.25*p+.25*m.predict_proba(cat_features(d))
            for s,p,d in zip(margins,old_cat,frames)]"""
    driver=driver.replace(old,new).replace('submission_v11_experimental.csv','submission_v26.csv').replace('metrics_v11_experimental.json','metrics_v26.json')
    driver=driver.replace('Train-only DS102 v6 inference','Train-only DS102 V26 text-capacity inference')
    modules['train_submit_v26.py']=driver
    cv={k:evidence[k] for k in ['baseline_oof','candidate_oof','fold_deltas','novel_fake_f1_delta','gates','bootstrap','dev_delta','test_changes_vs_best']}
    runner="""import json,os,platform
from pathlib import Path
assert platform.system()=='Linux' and os.environ.get('COLAB_RELEASE_TAG'),'Hosted Colab required'
from train_submit_v26 import main
selection={'recipe':{'kind':'latent','trees':800,'alpha':.3},'weights':[1.,1.2,1.2,.9],
           'name':'V26_SVD128_text2000_half_branch'}
out=Path('/content/ds102_v26')
report=main(selection,data_dir='/content/ds102_v26_data',output_dir=str(out),reference_path=None)
report.pop('v5_reference',None)
report['baseline_user_reported_kaggle']=0.77436
report['cv_evidence']=CV_EVIDENCE
report['qualification']='Passed the recorded exploratory train-OOF/dev gates; Kaggle score unknown.'
report['recipe']='0.35*SVM_probability+0.175*CatText700+0.175*CatText2000+0.30*latent128'
report['inference_note']='Fresh fit of all components from provided train CSV; the exploratory CSV used authorized baseline caches. Numerical differences are possible. Dev excluded from training.'
(out/'metrics_v26.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('CSV:',out/'submission_v26.csv')
print('Dev F1:',report['dev']['macro_f1'])
""".replace('CV_EVIDENCE',repr(cv))
    modules['run_submission.py']=runner
    for name,src in modules.items():ast.parse(src,filename=name)
    nb['cells'][0]['source']=[
        '# DS102 V26 — SVD128 và CatBoost text2000\n',
        'Mốc Kaggle trước đó:0.77436. Ứng viên V26 đã qua các tiêu chí OOF/dev đã ghi nhận; chưa biết điểm Kaggle mới.\n',
        'Notebook fit lại toàn bộ pipeline từ train trên hosted Colab; không dùng cache dự đoán, pretrained hoặc dữ liệu bên ngoài. Dev giữ riêng.\n',
        'OOF/dev đã được xem nhiều lần nên còn thiên lệch chọn mô hình. Không bảo đảm tăng leaderboard và không tự submit.\n',
        'Source đầy đủ được nhúng; scikit-learn BSD-3-Clause, CatBoost Apache-2.0. Mọi fit chỉ chạy sau kiểm tra hosted Colab.\n']
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            text=''.join(cell['source']).replace('ds102_v11_experimental_source','ds102_v26_source').replace('ds102_v11_experimental','ds102_v26').replace('ds102_v11_data','ds102_v26_data').replace('submission_v11_experimental.csv','submission_v26.csv').replace('metrics_v11_experimental.json','metrics_v26.json')
            cell['source']=text.splitlines(True)
            cell['outputs']=[];cell['execution_count']=None
    embedding['source']=('import json\nfrom pathlib import Path\nsource=Path("/content/ds102_v26_source");source.mkdir(exist_ok=True)\n'
        +'modules=json.loads('+repr(json.dumps(modules,ensure_ascii=False))+')\n'
        +'for name,content in modules.items():\n    (source/name).write_text(content)\n').splitlines(True)
    for i,cell in enumerate(nb['cells']):
        cell['id']=f'ds102-v26-{i}'
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    (ROOT/'train_submit_v26.py').write_text(driver)
    (ROOT/'DS102_Colab_v26_no_pretrained.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1))
    print('V26 notebook/source built; all Python parsed. Notebook not executed by builder.')

if __name__=='__main__':build()
