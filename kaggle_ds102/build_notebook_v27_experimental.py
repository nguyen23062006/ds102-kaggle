"""Build a source-complete fresh-training notebook for the V27 exploratory recipe.

Only source/JSON processing runs here, never training on the local machine.
"""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def build():
    evidence=json.loads((ROOT/'output_v27_experimental/metrics_v27_experimental.json').read_text())
    assert evidence['stage']=='experimental_candidate_ready_not_submitted','No qualified exploratory CSV'
    cv=evidence['cv']
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
    return [(.5*softmax(3*s,axis=1)+.5*p,p,m.predict_proba(cat_features(d)))
            for s,p,d in zip(margins,old_cat,frames)]"""
    driver=driver.replace(old,new)
    old='        f=LatentFeatures();xt=f.transform(train,fit=True)'
    assert driver.count(old)==1
    driver=driver.replace(old,'        f=LatentFeatures();f.svd.set_params(n_components=128,random_state=102);xt=f.transform(train,fit=True)')
    driver=driver.replace("    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)","    assert reference_path is None,'This runner fits every component fresh; no reference cache.'\n    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)")
    old="        for b,(p,s) in zip(base,extra):"
    assert driver.count(old)==1
    driver=driver.replace(old,"        for (b,old_cat,new_cat),(p,s) in zip(base,extra):")
    old="            scores.append((1-weight)*b+weight*p)"
    assert driver.count(old)==1
    driver=driver.replace(old,"""            combined=(1-weight)*b+weight*p
            delta=np.clip(.175*(new_cat[:,2]-old_cat[:,2]),-combined[:,2],combined[:,0])
            combined[:,2]+=delta;combined[:,0]-=delta
            assert (combined>=0).all() and np.allclose(combined.sum(1),1,atol=1e-5)
            scores.append(combined)""")
    driver=driver.replace('submission_v11_experimental.csv','submission_v27_experimental.csv').replace('metrics_v11_experimental.json','metrics_v27_experimental.json')
    driver=driver.replace('Train-only DS102 v6 inference','Train-only DS102 V27 exploratory class2 text inference')
    modules['train_submit_v27_experimental.py']=driver
    cvcopy={k:cv[k] for k in ['baseline_oof','candidate_oof','delta','fold_deltas','novel_fake_f1_delta','gates','bootstrap']}
    runner="""import json,os,platform
from pathlib import Path
assert platform.system()=='Linux' and os.environ.get('COLAB_RELEASE_TAG'),'Hosted Colab required'
from train_submit_v27_experimental import main
selection={'recipe':{'kind':'latent','trees':800,'alpha':.3},'weights':[1.,1.2,1.2,.9],
           'name':'V27_experimental_class2_text2000_update'}
out=Path('/content/ds102_v27_experimental')
report=main(selection,data_dir='/content/ds102_v27_data',output_dir=str(out),reference_path=None)
report.pop('v5_reference',None)
report['baseline_user_reported_kaggle']=0.77436
report['cv_evidence']=CV_EVIDENCE
report['qualification']='Experimental: OOF improved but the original +0.001 macro threshold failed; original rejection preserved. Kaggle score unknown.'
report['recipe']='Start with SVD128 baseline; delta=clip(0.175*(new_cat2-old_cat2),-baseline2,baseline0); add delta to class2, subtract from class0.'
report['inference_note']='Fresh fit of all components from train CSV. Earlier exploratory CSV used authorized baseline caches. Numerical differences are possible. Dev excluded from training.'
(out/'metrics_v27_experimental.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('EXPERIMENTAL CSV:',out/'submission_v27_experimental.csv')
print('Dev F1:',report['dev']['macro_f1'])
""".replace('CV_EVIDENCE',repr(cvcopy))
    modules['run_submission.py']=runner
    assert 'n_components=128' in modules['latent_v6.py'],'Wrong SVD rank in source notebook'
    for name,source in modules.items():ast.parse(source,filename=name)
    nb['cells'][0]['source']=[
        '# DS102 V27 — ứng viên thử nghiệm, không pretrained\n',
        'Mốc Kaggle trước đó:0.77436. OOF V27 tăng0.000925, dưới ngưỡng chọn chính thức0.001; không gọi là mô hình đã vượt mọi gate.\n',
        'CSV thử nghiệm chỉ được xuất sau full-fit khi dev không giảm. Chưa xác nhận mức tăng Kaggle; không tự submit.\n',
        'Notebook fit lại toàn bộ pipeline từ train trên hosted Colab, không cần cache dự đoán; dev giữ riêng, chỉ dùng bốn CSV được cung cấp.\n',
        'OOF/dev được xem nhiều lần nên còn thiên lệch lựa chọn. Source đầy đủ được nhúng; scikit-learn BSD-3-Clause, CatBoost Apache-2.0.\n']
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            text=''.join(cell['source']).replace('ds102_v11_experimental_source','ds102_v27_experimental_source').replace('ds102_v11_experimental','ds102_v27_experimental').replace('ds102_v11_data','ds102_v27_data').replace('submission_v11_experimental.csv','submission_v27_experimental.csv').replace('metrics_v11_experimental.json','metrics_v27_experimental.json')
            cell['source']=text.splitlines(True);cell['outputs']=[];cell['execution_count']=None
    embedding['source']=('import json\nfrom pathlib import Path\nsource=Path("/content/ds102_v27_experimental_source");source.mkdir(exist_ok=True)\n'
        +'modules=json.loads('+repr(json.dumps(modules,ensure_ascii=False))+')\n'
        +'for name,content in modules.items():\n    (source/name).write_text(content)\n').splitlines(True)
    upload=next(c for c in nb['cells'] if c['cell_type']=='code' and 'uploaded=files.upload()' in ''.join(c['source']))
    expected={n:hashlib.sha256((Path('/Users/ngien2306/Downloads/ds-102-r-11-cnvn')/n).read_bytes()).hexdigest()
        for n in ['ds102_train.csv','ds102_dev.csv','ds102_test.csv','ds102_samplesubmission.csv']}
    upload['source']+=('\nimport hashlib\nexpected_data_sha256='+repr(expected)+'\nfor name,h in expected_data_sha256.items():\n    assert hashlib.sha256((data_dir/name).read_bytes()).hexdigest()==h, "Competition CSV differs: "+name\n').splitlines(True)
    for i,cell in enumerate(nb['cells']):
        cell['id']=f'ds102-v27-experimental-{i}'
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    (ROOT/'train_submit_v27_experimental.py').write_text(driver)
    (ROOT/'DS102_Colab_v27_experimental.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1))
    print('Fresh V27 notebook/source built and parsed; not trained by builder.')

if __name__=='__main__':build()
