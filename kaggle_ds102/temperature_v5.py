"""Select coarse SVM temperature / CatBoost mixing using saved train OOF."""
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from experiment_v5 import hosted_guard
from selection_v5 import cross_weight
from select_v2 import macro


def main(component_path='/content/colab_components_v5.npz',
         output_dir='/content/ds102_v5_temperature'):
    hosted_guard()
    z=np.load(component_path);y=z['y'];folds=z['folds']
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    # Fixed coarse candidates. Original v3 (temperature=3, cat_weight=.5)
    # remains in the grid. No development labels are read by this function.
    for temperature in [1.5,3.,6.]:
        for cat_weight in [.25,.5,.75]:
            p=(1-cat_weight)*softmax(temperature*z['svm_oof'],axis=1)+cat_weight*z['cat_oof']
            row=cross_weight(p,y,folds)
            row.update(config={'temperature':temperature,'cat_weight':cat_weight},
                       family='temperature',new_model_weight=1.,model_index=0)
            row['raw_fold_f1']=[macro(y[folds==f],p[folds==f].argmax(1)) for f in np.unique(folds)]
            rows.append(row)
    rows.sort(key=lambda r:r['selection_score'],reverse=True)
    (out/'selection.json').write_text(json.dumps(rows,indent=2))
    (out/'status.json').write_text(json.dumps({'stage':'complete','best':rows[0]},indent=2))
    return rows
