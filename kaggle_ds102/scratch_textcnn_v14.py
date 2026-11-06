"""Small random-initialized TextCNN, trained only on the supplied fit partition.

Inspired by CNN-rand (Kim 2014, https://aclanthology.org/D14-1181/), with
additional review/title relations and numeric features. No downloaded weights.
"""
import gc,json,os,random,time
from collections import Counter
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import StandardScaler
from improved_features import tokens,dense_features
from experiment_v10 import guard,checked

CONFIG=dict(vocabulary=30000,min_count=2,embedding=64,channels=48,kernels=[2,3,4],
    review_length=160,title_length=40,hidden=128,dropout=.3,embedding_dropout=.15,
    epochs=12,snapshot_epochs=[10,11,12],batch_size=64,lr=.0015,weight_decay=.01,
    class_weights='inverse sqrt class frequency from fitting labels',seed=102,
    initialization='all weights random; no pretrained embeddings or models')


class Features:
    def fit(self,frame):
        guard();counter=Counter()
        for col in ['comment','product_name']:
            for text in frame[col]:counter.update(tokens(text))
        words=sorted((w for w,n in counter.items() if n>=2),key=lambda w:(-counter[w],w))[:30000]
        self.vocab={w:i+2 for i,w in enumerate(words)}
        self.scaler=StandardScaler().fit(dense_features(frame))
        return self

    def encode(self,series,length,tail=False):
        result=np.zeros((len(series),length),dtype=np.int64)
        for i,text in enumerate(series):
            ws=tokens(text)
            if len(ws)>length:ws=ws[:length-32]+ws[-32:] if tail else ws[:length]
            result[i,:len(ws)]=[self.vocab.get(w,1) for w in ws]
        return torch.from_numpy(result)

    def transform(self,frame):
        r=self.encode(frame.comment,160,tail=True);t=self.encode(frame.product_name,40)
        nums=np.clip(self.scaler.transform(dense_features(frame)),-5,5).astype(np.float32)
        assert np.isfinite(nums).all()
        return r,t,torch.from_numpy(nums)


class TextCNN(nn.Module):
    def __init__(self,vocabulary,numeric):
        super().__init__()
        self.embedding=nn.Embedding(vocabulary,64,padding_idx=0)
        nn.init.uniform_(self.embedding.weight,-.1,.1)
        with torch.no_grad():self.embedding.weight[0].zero_()
        self.embedding_dropout=nn.Dropout(.15)
        self.convs=nn.ModuleList([nn.Conv1d(64,48,k) for k in [2,3,4]])
        self.pool_norm=nn.LayerNorm(144);self.mean_norm=nn.LayerNorm(64)
        self.numeric=nn.Sequential(nn.Linear(numeric,32),nn.ReLU(),nn.LayerNorm(32))
        self.output=nn.Sequential(nn.Linear(144+4*64+32,128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,4))

    def masked_mean(self,e,ids):
        mask=(ids!=0).unsqueeze(-1)
        return self.mean_norm((e*mask).sum(1)/mask.sum(1).clamp_min(1))

    def forward(self,r,t,nums):
        re=self.embedding_dropout(self.embedding(r));te=self.embedding_dropout(self.embedding(t))
        length=(r!=0).sum(1);parts=[]
        for kernel,conv in zip([2,3,4],self.convs):
            act=torch.relu(conv(re.transpose(1,2)))
            valid=(length-kernel+1).clamp_min(1)
            mask=torch.arange(act.shape[2],device=r.device)[None,:]<valid[:,None]
            parts.append(act.masked_fill(~mask[:,None,:],-1e4).amax(2))
        pool=self.pool_norm(torch.cat(parts,1))
        rm=self.masked_mean(re,r);tm=self.masked_mean(te,t)
        return self.output(torch.cat([pool,rm,tm,rm*tm,(rm-tm).abs(),self.numeric(nums)],1))


def probabilities(model,inputs,device):
    model.eval();parts=[]
    with torch.inference_mode():
        for batch in DataLoader(TensorDataset(*inputs),batch_size=256,shuffle=False,num_workers=0):
            p=torch.softmax(model(*(x.to(device) for x in batch)),dim=1)
            parts.append(p.cpu().numpy())
    return checked(np.concatenate(parts),len(inputs[0]))


def fit_predict(train,frames,folder,status):
    guard();torch.set_num_threads(2)
    random.seed(102);np.random.seed(102);torch.manual_seed(102)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(102)
    torch.use_deterministic_algorithms(True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    status('cnn_features',device=str(device))
    features=Features().fit(train);xt=features.transform(train)
    external=[features.transform(d) for d in frames]
    y=train.spam_label.to_numpy();counts=np.bincount(y,minlength=4)
    assert (counts>0).all()
    model=TextCNN(len(features.vocab)+2,xt[2].shape[1]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.01)
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(np.sqrt(len(y)/(4*counts)),dtype=torch.float32,device=device))
    generator=torch.Generator().manual_seed(102)
    loader=DataLoader(TensorDataset(*xt,torch.from_numpy(y.astype(np.int64))),
        batch_size=64,shuffle=True,generator=generator,num_workers=0)
    result=[np.zeros((len(d),4),dtype=np.float64) for d in frames]
    (folder/'features.json').write_text(json.dumps(dict(vocabulary=features.vocab,
        scaler_mean=features.scaler.mean_.tolist(),scaler_scale=features.scaler.scale_.tolist(),config=CONFIG),ensure_ascii=False))
    start=time.time()
    for epoch in range(1,13):
        model.train();total=0.;samples=0
        for r,t,num,target in loader:
            r,t,num,target=[v.to(device) for v in [r,t,num,target]]
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(r,t,num),target)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=float(loss.detach())*len(target);samples+=len(target)
        status('cnn_epoch',epoch=epoch,training_loss=total/samples,elapsed_seconds=time.time()-start)
        if epoch in [10,11,12]:
            for i,inputs in enumerate(external):result[i]+=probabilities(model,inputs,device)/3
            path=folder/f'epoch{epoch}.pt';tmp=path.with_suffix('.tmp.pt')
            torch.save(model.state_dict(),tmp);tmp.replace(path)
    result=[checked(p,len(d)) for p,d in zip(result,frames)]
    del features,xt,external,model,optimizer,loader;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return result
