"""Secondary frozen-configuration check of Antigravity H1; not a release bypass.

V60 narrowly missed the predeclared effect-size gate. Preserve that result.
This script measures dev performance of exactly the same frozen candidate,
without fitting dev, choosing iterations on dev, exporting CSV or auto-submitting.
"""
import hashlib, importlib.metadata, json, os, platform, sys, time, traceback
from pathlib import Path

if platform.system()!='Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
    raise RuntimeError('Hosted Colab only.')
sys.path[:0]=['/content/ds102_v60/source','/content/ds102_v53/source']
import joblib
import numpy as np
import experiment_v60 as h1
import protocol_v60 as protocol
from train_submit import load_data,metrics
from improved_features import prepare
from experiment_v53 import labels,checked,BEST_SHA

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

root=Path('/content/ds102_v60');out=root/'dev_verification';out.mkdir(exist_ok=True)
start=time.time()
report=dict(stage='initializing',diagnostic_only=True,submission_created=False,
    parent_cluster_gate_result='rejected_on_cluster; remains unchanged',
    selection_note='Follow-up chosen after positive clustered results narrowly below .003. Not an independent test or retroactive gate pass.',
    frozen_combination='.35*SVM+.35*CatText+.15*old_SVD128+.15*H1_SVD128')
def status(stage,**kwargs):
    report.update(stage=stage,elapsed_seconds=time.time()-start,**kwargs)
    (out/'metrics_v60_dev.json').write_text(json.dumps(report,indent=2))
    print(stage,json.dumps(kwargs),flush=True)
try:
    parent=json.loads((root/'run/metrics_v60.json').read_text())
    pm=json.loads((root/'run/run_manifest.json').read_text())
    assert parent['stage']=='rejected_on_cluster'
    assert digest(h1.__file__)==parent['config']['head_source_sha256']
    assert digest(h1.h1_intent_features.__file__)==parent['config']['intent_source_sha256']
    assert digest(protocol.__file__)==pm['source_sha256']
    inputs=Path('/content/ds102_v53/inputs')
    for n,s in pm['inputs'].items():assert digest(inputs/n)==s
    for n,s in pm['parent_sources'].items():assert digest(Path('/content/ds102_v53/source')/n)==s
    assert {n:importlib.metadata.version(n) for n in pm['versions']}==pm['versions']
    manifest=dict(parent_manifest_sha256=digest(root/'run/run_manifest.json'),
        source_sha256=digest(__file__),config=parent['config'],inputs=pm['inputs'])
    mid=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    mp=out/'run_manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    mp.write_text(json.dumps(manifest,indent=2))
    data=load_data(inputs/'data');train=prepare(data['train']);dev=prepare(data['dev'])
    ref=np.load(inputs/'baseline/reference_v3.npz');full=np.load(inputs/'baseline/latent_full.npz')
    fm=json.loads((inputs/'baseline/latent_full_manifest.json').read_text())
    assert full['manifest_id'].item()==hashlib.sha256(json.dumps(fm,sort_keys=True).encode()).hexdigest()
    bd=checked(.7*ref['dev']+.3*full['dev'],len(dev))
    bt=checked(.7*ref['test']+.3*full['test'],len(data['test']))
    sub=data['samplesubmission'].copy()
    sub['spam_label']=sub.ID.astype(str).map(dict(zip(data['test'].id.astype(str),labels(bt)))).astype(int)
    assert hashlib.sha256(sub.to_csv(index=False).encode()).hexdigest()==BEST_SHA
    protocol.CONFIG=parent['config'];h1.STATE.update(out=out,smoke=False)
    path=out/'full_head.joblib'
    if path.exists():
        saved=joblib.load(path);assert saved['manifest_id']==mid;model=saved['model']
    else:
        status('fit_full_train',rows=len(train))
        model=h1.fitted_head(train,np.load('/content/ds102_v53/run/groups.npz')['groups'])
        joblib.dump(dict(manifest_id=mid,model=model),path)
    p=checked(model.predict_proba(dev.drop(columns='spam_label')),len(dev))
    cp=checked(bd+.15*(p-full['dev']),len(dev))
    y=dev.spam_label.to_numpy();b=labels(bd);c=labels(cp)
    unseen=~dev.comment.isin(set(train.comment)).to_numpy()
    bm=metrics(y,b);cm=metrics(y,c)
    bu=metrics(y[unseen],b[unseen]);cu=metrics(y[unseen],c[unseen])
    np.savez_compressed(out/'dev_predictions.npz',manifest_id=mid,y=y,baseline=bd,candidate=cp,head=p,unseen=unseen)
    status('dev_verification_complete',baseline=bm,candidate=cm,
        delta=cm['macro_f1']-bm['macro_f1'],
        without_exact_overlap_delta=cu['macro_f1']-bu['macro_f1'],
        changed_predictions=int((b!=c).sum()),corrected_errors=int(((b!=y)&(c==y)).sum()),
        introduced_errors=int(((b==y)&(c!=y)).sum()),
        dev_checks=dict(macro_gain=bool(cm['macro_f1']-bm['macro_f1']>=.001),
            without_exact_overlap_gain=bool(cu['macro_f1']-bu['macro_f1']>=.001)))
except BaseException as e:
    (out/'error.txt').write_text(traceback.format_exc());status('error',error=str(e));raise
