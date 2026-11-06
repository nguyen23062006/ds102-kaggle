"""Optional fastText classifier trained from random initialization on Colab.

Uses the official MIT-licensed v0.9.2 C++ CLI. No pretrained vectors, downloaded
model, external corpus, or Python fastText bindings are used.
"""
import gc
import hashlib
import json
import math
import subprocess
import traceback
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from experiment_v5 import hosted_guard
from improved_features import prepare,tokens
from train_submit import load_data,SEED
from select_v2 import macro

CONFIGS=[{'balance_power':.5},{'balance_power':.75}]


def install():
    hosted_guard()
    root=Path('/content/fastText-0.9.2'); binary=root/'fasttext'
    if not binary.exists():
        archive=Path('/content/fastText-0.9.2.zip')
        urllib.request.urlretrieve('https://github.com/facebookresearch/fastText/archive/refs/tags/v0.9.2.zip',archive)
        with zipfile.ZipFile(archive) as z: z.extractall('/content')
        # GCC in recent Colab images requires an explicit cstdint include for
        # the unmodified 0.9.2 source. This changes compilation, not the model.
        subprocess.run(['make','-j2','CXX=c++ -include cstdint'],cwd=root,
                       check=True,capture_output=True,text=True)
        (root/'archive_sha256.txt').write_text(hashlib.sha256(archive.read_bytes()).hexdigest())
    return binary


def document(row):
    words=[t for t in tokens(row.comment) if not t.startswith('__label__')]
    title=set(tokens(row.product_name)); description=set(tokens(row.description))
    s=set(words); n=max(len(s),1)
    context=[f'rating_{row.rating}',f'lengthbin_{int(math.log2(max(len(words),1)))}',
             f'titleoverlap_{int(8*len(s&title)/n)}',
             f'descoverlap_{int(8*len(s&description)/n)}']
    context += ['product_'+w for w in tokens(row.product_name)[:30]]
    context += ['category_'+w for w in tokens(str(row.category))[:10]]
    return ' '.join(words[:300]+context)


def fit_predict(train,frames,config,folder):
    hosted_guard()
    binary=install();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    y=train.spam_label.to_numpy();counts=np.bincount(y,minlength=4)
    rng=np.random.default_rng(SEED)
    lines=[]
    for row in train.itertuples(index=False):
        factor=(counts.max()/counts[int(row.spam_label)])**config['balance_power']
        copies=int(factor)+int(rng.random()<factor-int(factor))
        lines += [f'__label__{int(row.spam_label)} '+document(row)]*copies
    rng.shuffle(lines)
    corpus=folder/'train.txt';corpus.write_text('\n'.join(lines)+'\n')
    prefix=folder/'model'
    command=[str(binary),'supervised','-input',str(corpus),'-output',str(prefix),
             '-lr','0.5','-epoch','25','-dim','64','-wordNgrams','2',
             '-minn','2','-maxn','5','-bucket','100000','-minCount','2',
             '-loss','softmax','-thread','1','-verbose','0']
    subprocess.run(command,check=True,capture_output=True,text=True)
    result=[]
    for i,df in enumerate(frames):
        path=folder/f'predict_{i}.txt'
        path.write_text('\n'.join(document(row) for row in df.itertuples(index=False))+'\n')
        output=subprocess.run([str(binary),'predict-prob',str(prefix)+'.bin',str(path),'4'],
                              check=True,capture_output=True,text=True).stdout.splitlines()
        if len(output)!=len(df): raise RuntimeError('fastText prediction count mismatch')
        p=np.zeros((len(df),4),dtype=np.float32)
        for k,line in enumerate(output):
            values=line.split()
            for label,probability in zip(values[::2],values[1::2]):
                p[k,int(label.removeprefix('__label__'))]=float(probability)
        p/=p.sum(axis=1,keepdims=True)
        if not np.isfinite(p).all(): raise RuntimeError('Invalid fastText probabilities')
        result.append(p)
    return result


def main(data_dir='/content/ds102_data',output_dir='/content/ds102_v5_fasttext'):
    hosted_guard()
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    def status(**values):
        (out/'status.json').write_text(json.dumps(values,indent=2))
    try:
        status(stage='install_official_source');install()
        df=prepare(load_data(Path(data_dir))['train']);y=df.spam_label.to_numpy()
        folds=np.full(len(y),-1);oof=np.zeros((len(CONFIGS),len(y),4),dtype=np.float32)
        results=[[] for _ in CONFIGS]
        split=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED)
        for f,(tr,va) in enumerate(split.split(df,y,groups=df.comment)):
            folds[va]=f
            for i,c in enumerate(CONFIGS):
                status(stage='fit',fold=f+1,config=c,results=results)
                oof[i,va]=fit_predict(df.iloc[tr],[df.iloc[va].drop(columns='spam_label')],c,out/f'fold{f}_config{i}')[0]
                results[i].append(macro(y[va],oof[i,va].argmax(1)))
                gc.collect()
            np.savez_compressed(out/'partial.npz',oof=oof,y=y,folds=folds)
        report={'family':'fasttext','configs':CONFIGS,'fold_scores':results,
                'oof_macro_f1':[macro(y,p.argmax(1)) for p in oof],
                'pretrained':False,'license':'MIT','fit_data':'train only',
                'initialization':'random; no pretrainedVectors argument'}
        np.savez_compressed(out/'oof.npz',oof=oof,y=y,folds=folds)
        (out/'cv_results.json').write_text(json.dumps(report,indent=2));status(stage='complete',**report)
        return report
    except BaseException as exc:
        status(stage='error',error=str(exc),traceback=traceback.format_exc())
        raise
