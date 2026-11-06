import os,platform
assert platform.system()=='Linux' and os.environ.get('COLAB_RELEASE_TAG'), 'Colab only'
import numpy as np
import torch
import h2_subword_cnn as h
torch.set_num_threads(2)
torch.manual_seed(102)
tok=h.SubwordTokenizer(min_freq=1).fit(['abcdefghij'])
inv={v:k for k,v in tok.vocab.items()}
assert {len(inv[i]) for i in tok.encode(['abcdefghij'],max_len=6)[0]}=={3,4,5}
vocab=tok.vocab.copy();tok.encode(['completely unseen text']);assert tok.vocab==vocab
CNN,Focal,CE=h._build_pytorch_classes()
model=CNN(vocab_size=30).eval()
for length in [0,1,2,3,4,5,9]:
    x=torch.zeros((1,12),dtype=torch.long);x[:,:length]=2
    xx=torch.cat([x,torch.zeros((1,8),dtype=torch.long)],dim=1)
    with torch.inference_mode():
        a=model(x);b=model(xx)
    assert torch.isfinite(a).all() and torch.allclose(a,b,atol=1e-6),length
alpha=torch.tensor([1.,2.,3.,4.]);logits=torch.randn(12,4);y=torch.arange(12)%4
assert torch.allclose(Focal(alpha,gamma=0)(logits,y),CE(alpha)(logits,y))
print('PASS tokenizer interleaving; inference vocabulary unchanged; padding invariance including short/empty inputs; focal gamma0 equals weighted CE')
