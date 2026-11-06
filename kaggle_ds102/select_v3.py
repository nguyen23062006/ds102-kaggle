import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from select_v2 import macro, calibrate


def choose_v3():
    root=Path('kaggle_ds102')
    old=np.load(root/'output_v2/scores_v2.npz')
    cat=np.load(root/'experiments_v3/oof.npz')
    linear=np.load(root/'experiments_v3/linear_oof.npz')
    assert np.array_equal(old['y'],cat['y']) and np.array_equal(old['folds'],cat['folds'])
    assert np.array_equal(old['y'],linear['y']) and np.array_equal(old['folds'],linear['folds'])
    y,folds=old['y'],old['folds']
    # Nhiệt độ cố định; không chọn dựa trên dev hoặc Kaggle.
    base_p=softmax(3*old['oof'],axis=1)
    candidates=[('v2',base_p,{'svm_recipe':'v2','cat_weight':0.})]
    for i,weight in enumerate(['SqrtBalanced','Balanced']):
        for frac in [.25,.5,.75,1.]:
            p=(1-frac)*base_p+frac*cat['scores'][i]
            candidates.append((f'v2_cat_{weight}_{frac}',p,
                               {'svm_recipe':'v2','cat_model_weight':weight,'cat_weight':frac}))
    # Meta scale=2 gặp cảnh báo chưa hội tụ: không sử dụng.
    for i,c in enumerate([.1,.25]):
        p=softmax(3*linear['scores'][i],axis=1)
        candidates.append((f'svm_C_{c}',p,{'svm_recipe':f'C_{c}','cat_weight':0.}))
    result=[]
    for name,p,recipe in candidates:
        raw=macro(y,p.argmax(1))
        logs=np.log(np.clip(p,1e-12,1.))
        cp=np.zeros(len(y),int)
        for fold in np.unique(folds):
            tr=folds!=fold
            b=calibrate(logs[tr],y[tr])
            cp[~tr]=(logs[~tr]+b).argmax(1)
        cross=macro(y,cp)
        use=cross>raw+.003
        final_pred=cp if use else p.argmax(1)
        result.append({'name':name,**recipe,'raw_oof_macro_f1':raw,
                       'cross_calibration_macro_f1':cross,'selection_score':cross if use else raw,
                       'bias':calibrate(logs,y).tolist() if use else [0.,0.,0.,0.],
                       'fold_macro_f1':[macro(y[folds==f],final_pred[folds==f]) for f in np.unique(folds)]})
    result.sort(key=lambda r:r['selection_score'],reverse=True)
    (root/'experiments_v3/selection.json').write_text(json.dumps(result,indent=2))
    for r in result: print(r['name'], 'raw',round(r['raw_oof_macro_f1'],5),
                           'cross',round(r['cross_calibration_macro_f1'],5),'bias',r['bias'],flush=True)
    return result


if __name__=='__main__':choose_v3()
