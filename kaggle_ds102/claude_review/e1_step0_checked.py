"""No fitting: reproduce strata, paired uncertainty, and audit cluster sizes.

This only diagnoses OLD exact-group OOF. It cannot turn that OOF into
near-cluster-held-out predictions; E1 must retrain for that estimand.
"""
import argparse,json,os,platform,time,traceback
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1])]
import numpy as np
from train_submit import load_data
from improved_features import prepare
from codex_checked_common import (W_V5,W_V6,max_jaccard,near_dup_groups,macro,
    class_f1,class1_strata,paired_cluster_bootstrap,group_summary)


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='/content/ds102_data')
    p.add_argument('--reference',default='/content/colab_reference.npz')
    p.add_argument('--latent',default='/content/ds102_v6_latent/oof.npz')
    p.add_argument('--v5-predictions',default='/content/ds102_v5/predictions.npz')
    p.add_argument('--v6-predictions',default='/content/ds102_v6/predictions.npz')
    p.add_argument('--out',default='/content/claude_review_out/step0_checked')
    p.add_argument('--bootstrap',type=int,default=2000)
    a=p.parse_args(argv)
    if platform.system()!='Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Run the diagnostic on hosted Colab; no fitting performed.')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    def status(stage,**extra):
        (out/'status.json').write_text(json.dumps(dict(stage=stage,time=time.time(),**extra),indent=2))
    try:
        data=load_data(Path(a.data_dir));tr=prepare(data['train']);dv=prepare(data['dev'])
        y=tr.spam_label.to_numpy();yd=dv.spam_label.to_numpy()
        ref=np.load(a.reference);lat=np.load(a.latent)
        assert np.array_equal(y,ref['y']) and np.array_equal(y,lat['y'])
        assert np.array_equal(ref['folds'],lat['folds'])
        assert lat['oof'].shape==(2,len(y),4)
        assert np.isfinite(lat['oof']).all() and np.isfinite(ref['oof']).all()
        folds=ref['folds'];text=tr.comment.tolist();tmpl=np.zeros(len(y),bool)
        status('template_strata')
        for fold in np.unique(folds):
            va=np.flatnonzero(folds==fold);r1=np.flatnonzero((folds!=fold)&(y==1))
            tmpl[va]=max_jaccard([text[i] for i in va],[text[i] for i in r1])>=.6
        td=max_jaccard(dv.comment.tolist(),tr.loc[tr.spam_label==1,'comment'].tolist())>=.6
        pred={'v5':(ref['oof']*W_V5).argmax(1),
              'v6':((.7*ref['oof']+.3*lat['oof'][1])*W_V6).argmax(1)}
        status('cluster_audit')
        cluster_reports={};g0=None
        for thr in [.6,.7,.8]:
            status('cluster_audit',threshold=thr)
            groups=near_dup_groups(text,thr=thr)
            cluster_reports[str(thr)]=group_summary(groups,y)
            if thr==.6:g0=groups.copy()
        report={'scope':'Descriptive rescore of frozen, repeatedly selected OLD OOF; not new cluster CV',
                'template_definition':'max token-set Jaccard >=.6 to fit-fold class-1 reviews; diagnostic only',
                'class1_template_counts':{'oof':[int(tmpl[y==1].sum()),int((y==1).sum())],
                                          'dev':[int(td[yd==1].sum()),int((yd==1).sum())]},
                'cluster_audit':cluster_reports,'models':{}}
        for name,prediction in pred.items():
            report['models'][name]={'frozen_weight_macro':macro(y,prediction),
                'class_f1':class_f1(y,prediction).tolist(),'class1_strata':class1_strata(y,prediction,tmpl),
                'macro_excluding_all_template_flagged_rows':macro(y[~tmpl],prediction[~tmpl])}
        status('paired_oof_bootstrap')
        report['oof_delta']=paired_cluster_bootstrap(y,pred['v5'],pred['v6'],g0,n=a.bootstrap)
        report['non_template_oof_delta']=paired_cluster_bootstrap(y[~tmpl],pred['v5'][~tmpl],pred['v6'][~tmpl],g0[~tmpl],n=a.bootstrap)
        status('paired_dev_bootstrap')
        p5=np.load(a.v5_predictions);p6=np.load(a.v6_predictions)
        d5=(p5['dev']*p5['weights']).argmax(1);d6=(p6['dev']*p6['weights']).argmax(1)
        assert len(d5)==len(yd)==len(d6)
        gd=near_dup_groups(dv.comment.tolist(),thr=.6)
        report['dev']={'v5':macro(yd,d5),'v6':macro(yd,d6),
            'changed_predictions':int((d5!=d6).sum()),
            'paired_row_bootstrap':paired_cluster_bootstrap(yd,d5,d6,np.arange(len(yd)),n=a.bootstrap),
            'paired_cluster_bootstrap':paired_cluster_bootstrap(yd,d5,d6,gd,n=a.bootstrap),
            'class1_v5':class1_strata(yd,d5,td),'class1_v6':class1_strata(yd,d6,td)}
        report['warnings']=['Template strata are defined with fitting-part labels for diagnostics, not available as test labels.',
            'Removing all template-flagged rows is not standardization to a 24% class-1 template share.',
            'Single-model dev SD does not quantify the paired v6-v5 difference.',
            'Equal estimated recall 32/103 and 4/13 is compatible with the composition hypothesis, not proof of equal generalization.',
            'Frozen-weight OOF metrics differ from previously reported cross-weight metrics; neither this bootstrap nor reweighting repairs the old splitting or repeated selection.']
        (out/'report.json').write_text(json.dumps(report,indent=2));status('complete')
        print(json.dumps(report,indent=2));return report
    except BaseException as exc:
        status('error',error=str(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
