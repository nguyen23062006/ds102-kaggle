"""One-task Colab runner: run V54 only after V53 rejects the fixed candidate."""
import hashlib,json,os,platform,subprocess,sys,time
from pathlib import Path

assert platform.system()=='Linux' and os.environ.get('COLAB_RELEASE_TAG'), 'Hosted Colab only'
root=Path('/content/ds102_v54');root.mkdir(exist_ok=True)
parent=Path('/content/ds102_v53');python=parent/'venv/bin/python'
source=root/'source/experiment_v54.py'
expected=sys.argv[1]

def status(stage,**kwargs):
    p=root/'queue_status.json'
    p.write_text(json.dumps(dict(stage=stage,time=time.time(),**kwargs),indent=2))
    print(stage,json.dumps(kwargs),flush=True)

status('waiting_for_v53_completion')
deadline=time.monotonic()+5400
while not (parent/'run_exit.txt').exists():
    if time.monotonic()>deadline:
        status('parent_not_finished_before_wait_limit');sys.exit(0)
    time.sleep(10)
rc=int((parent/'run_exit.txt').read_text())
report=json.loads((parent/'run/metrics_v53.json').read_text())
if rc!=0 or not report['stage'].startswith(('rejected_on_','pruned_on_')):
    status('not_started_after_parent',parent_exit=rc,parent_stage=report['stage']);sys.exit(0)
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected,'V54 source changed after queueing'
env=os.environ.copy()
env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1')
for name,extra in [('smoke',['--smoke']),('run',[])]:
    status('running_'+name)
    cmd=[str(python),'-u',str(source),'--inputs',str(parent/'inputs'),'--output',str(root/name),*extra]
    with (root/(name+'.log')).open('a') as log:
        process=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=source.parent)
    (root/(name+'_exit.txt')).write_text(str(process.returncode))
    if process.returncode:
        status('error_in_'+name,returncode=process.returncode);sys.exit(process.returncode)
    if name=='smoke':
        assert json.loads((root/'smoke/metrics_v54.json').read_text())['stage']=='smoke_passed'
report=json.loads((root/'run/metrics_v54.json').read_text())
status('complete',result_stage=report['stage'],submission_created=report['submission_created'])
