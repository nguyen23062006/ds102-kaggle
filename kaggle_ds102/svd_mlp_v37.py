"""Small randomly initialized neural classifier on train-only SVD128 features."""
import gc,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from latent_v6 import LatentFeatures
from experiment_v10 import guard,checked

CONFIG=dict(hidden=[128,64],dropout=.35,epochs=40,snapshots=[20,30,40],
    learning_rate=.0003,weight_decay=.01,batch_size=256,seed=102,
    class_weight='sqrt inverse frequency',latent_replacement_fraction=.15)

class Network(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.skip=nn.Linear(d,4)
        nn.init.zeros_(self.skip.weight);nn.init.zeros_(self.skip.bias)
        self.net=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Dropout(.35),
            nn.Linear(128,64),nn.ReLU(),nn.Dropout(.35),nn.Linear(64,4))
    def forward(self,x):return self.skip(x)+self.net(x)

def fit_predict(train,frames,folder,feature_id,status,cache_path=None,cache_id=None):
    guard();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    try:torch.set_num_interop_threads(2)
    except RuntimeError:pass
    torch.manual_seed(102);torch.use_deterministic_algorithms(True);rng=np.random.default_rng(102)
    if cache_path is not None:
        q=np.load(cache_path,allow_pickle=False);assert q['feature_id'].item()==cache_id
        cols=q['numeric_columns'][:575].tolist();assert cols==[f'f_{i}' for i in range(575)]
        xt=pd.DataFrame(q['train_numeric'][:,:575],columns=cols);external=[]
        for j,c in enumerate(['category','rating']):xt[c]=q['train_categories'][:,j]
        for i in range(len(frames)):
            x=pd.DataFrame(q[f'frame{i}_numeric'][:,:575],columns=cols)
            for j,c in enumerate(['category','rating']):x[c]=q[f'frame{i}_categories'][:,j]
            external.append(x)
    else:
        status('fit_full_svd128_features')
        f=LatentFeatures();f.svd.set_params(n_components=128,random_state=102)
        xt=f.transform(train,fit=True);external=[f.transform(d) for d in frames]
        cols=[c for c in xt if c not in ['category','rating']];assert len(cols)==575
        del f;gc.collect()
    for x,df in zip([xt]+external,[train]+frames):
        assert len(x)==len(df)
        assert np.array_equal(x[['category','rating']].to_numpy(dtype=str),df[['category','rating']].astype(str).to_numpy())
    scaler=StandardScaler();a=np.clip(scaler.fit_transform(xt[cols]),-6,6).astype(np.float32)
    enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
    a=np.hstack([a,enc.fit_transform(xt[['category','rating']])])
    vv=[np.hstack([np.clip(scaler.transform(x[cols]),-6,6).astype(np.float32),enc.transform(x[['category','rating']])]) for x in external]
    assert np.isfinite(a).all() and all(np.isfinite(v).all() for v in vv)
    x=torch.from_numpy(np.ascontiguousarray(a));y_np=train.spam_label.to_numpy();y=torch.tensor(y_np,dtype=torch.long)
    counts=np.bincount(y_np,minlength=4);weights=torch.tensor(np.sqrt(len(y_np)/(4*counts)),dtype=torch.float32)
    m=Network(a.shape[1]);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01)
    lossfn=nn.CrossEntropyLoss(weight=weights);result=[np.zeros((len(v),4),dtype=np.float64) for v in vv]
    for epoch in range(1,41):
        m.train();order=rng.permutation(len(y_np));total_loss=0.
        for start in range(0,len(order),256):
            idx=order[start:start+256];opt.zero_grad(set_to_none=True)
            loss=lossfn(m(x[idx]),y[idx]);assert torch.isfinite(loss)
            loss.backward();opt.step();total_loss+=float(loss.detach())*len(idx)
        if epoch%10==0:status('fit_svd_mlp',epoch=epoch,training_loss=total_loss/len(y_np))
        if epoch in [20,30,40]:
            m.eval()
            with torch.inference_mode():
                for i,v in enumerate(vv):
                    pieces=[torch.softmax(m(torch.from_numpy(np.ascontiguousarray(v[j:j+2048]))),dim=1).numpy() for j in range(0,len(v),2048)]
                    result[i]+=np.vstack(pieces)/3
            torch.save(m.state_dict(),folder/f'epoch{epoch}.pt')
    (folder/'metadata.json').write_text(json.dumps(dict(config=CONFIG,feature_id=feature_id,
        input_dimensions=a.shape[1],scale_mean=scaler.mean_.tolist(),scale_std=scaler.scale_.tolist(),
        categories=[c.tolist() for c in enc.categories_]),indent=2))
    del m,x,y,opt,vv,xt,external;gc.collect()
    return [checked(p,len(d)) for p,d in zip(result,frames)]
