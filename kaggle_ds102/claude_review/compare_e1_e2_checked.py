"""No fitting: paired E2-vs-E1 comparison on identical grouped validation rows."""
import argparse,json,os,platform,sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1])]
import numpy as np
from codex_checked_common import (recipe_probabilities,labels,macro,class_f1,
    class1_strata,paired_cluster_bootstrap)


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--e1-dir',required=True);p.add_argument('--e2-dir',required=True)
    p.add_argument('--out',default='/content/claude_review_out/checked/comparison')
    p.add_argument('--bootstrap',type=int,default=2000)
    a=p.parse_args(argv)
    if platform.system()!='Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Run comparison on hosted Colab; no fitting.')
    d1=Path(a.e1_dir);d2=Path(a.e2_dir)
    r1=json.loads((d1/'report.json').read_text());r2=json.loads((d2/'report.json').read_text())
    m1=json.loads((d1/'manifest.json').read_text());m2=json.loads((d2/'manifest.json').read_text())
    assert m1['dedup_weight'] is False and m2['dedup_weight'] is True
    m1.pop('dedup_weight');m2.pop('dedup_weight')
    assert m1==m2,'E2 must only change dedup weights; source/data/split/model settings differ.'
    z1=np.load(d1/'oof_and_foldavg.npz');z2=np.load(d2/'oof_and_foldavg.npz')
    for k in ['y','folds','groups','tmpl','tmpl_dev']:
        assert np.array_equal(z1[k],z2[k]),f'Misaligned {k}'
    y=z1['y'];groups=z1['groups'];folds=z1['folds'];tmpl=z1['tmpl']
    q1=recipe_probabilities(*(z1[f'oof_{k}'] for k in ['svm','cat','lat']))
    q2=recipe_probabilities(*(z2[f'oof_{k}'] for k in ['svm','cat','lat']))
    p1={k:labels(v,k) for k,v in q1.items()};p2={k:labels(v,k) for k,v in q2.items()}
    report={'same_data_source_split_models_verified':True,'E1':str(d1),'E2':str(d2),
        'v5_refit_dev_reference':.7348200162618499,'candidates':{}}
    for name in ['v5','v6']:
        old=r1[name];new=r2[name]
        delta=new['oof_macro']-old['oof_macro']
        novel=new['oof_class1_strata']['novel']['f1']-old['oof_class1_strata']['novel']['f1']
        template=new['oof_class1_strata']['template']['f1']-old['oof_class1_strata']['template']['f1']
        gates={'macro_gain_at_least_003':delta>=.003,'novel_class1_f1_gain_at_least_03':novel>=.03,
            'template_class1_f1_not_down_more_than_05':template>=-.05,
            'dev_not_below_v5_refit_by_more_than_015':new['dev_foldavg_macro']>=.7348200162618499-.015}
        report['candidates'][name]={'E1_macro':old['oof_macro'],'E2_macro':new['oof_macro'],
            'E2_minus_E1':delta,'novel_class1_f1_delta':novel,'template_class1_f1_delta':template,
            'fold_deltas':(np.array(new['oof_fold_macro'])-np.array(old['oof_fold_macro'])).tolist(),
            'dev_foldavg_macro':new['dev_foldavg_macro'],'retention_gates':gates,'passes_all_gates':all(gates.values()),
            'paired_cluster_bootstrap':paired_cluster_bootstrap(y,p1[name],p2[name],groups,n=a.bootstrap),
            'non_template_paired_bootstrap':paired_cluster_bootstrap(y[~tmpl],p1[name][~tmpl],p2[name][~tmpl],groups[~tmpl],n=a.bootstrap),
            'vs_primary_E1_v5':{'delta':macro(y,p2[name])-macro(y,p1['v5']),
                'paired_cluster_bootstrap':paired_cluster_bootstrap(y,p1['v5'],p2[name],groups,n=a.bootstrap)}}
    report['caveat']='Predefined experimental gates are not proof of a Kaggle improvement. Paired bootstrap is conditional on selected/fitted predictions, not all prior model selection. Grouped CV is a novelty stress test; hidden-test template composition is unknown.'
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return report


if __name__=='__main__':main()
