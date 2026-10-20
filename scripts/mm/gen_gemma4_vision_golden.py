#!/usr/bin/env python3
"""Independent CPU equations from HF a08ace4b modeling_gemma4.py, not C++ output.
Reduced BF16 MHA model; no transformers package or original weights required.
"""
import json
import math
import pathlib
import struct
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
OUT = pathlib.Path(__file__).resolve().parents[2]/'tests/fixtures/gemma4_e4b_vision'
OUT.mkdir(parents=True, exist_ok=True)
D = torch.bfloat16
W = {}
def add(name, shape, seed=1, norm=False):
    n=math.prod(shape)
    x=((torch.arange(n)*7+seed*11)%29-14).float()/64
    if norm: x=1+x
    W[name]=x.reshape(shape).to(D)
    return W[name]
p='model.vision_tower.'
add(p+'patch_embedder.input_proj.weight',(4,768),3)
add(p+'patch_embedder.position_embedding_table',(2,32,4),4)
b=p+'encoder.layers.0.'
for i,name in enumerate(['q','k','v','o']):add(b+f'self_attn.{name}_proj.linear.weight',(4,4),i+3)
for i,name in enumerate(['gate','up']):add(b+f'mlp.{name}_proj.linear.weight',(6,4),i+9)
add(b+'mlp.down_proj.linear.weight',(4,6),14)
for i,name in enumerate(['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm','self_attn.q_norm','self_attn.k_norm']):add(b+name+'.weight',(4,),i+3,True)
clips={}
for i,name in enumerate(['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']):
    bounds=[-0.25-i/16,0.125+i/16,-0.1875-i/32,0.09375+i/32]
    clips[name]=bounds
    for key,value in zip(['input_min','input_max','output_min','output_max'],bounds):W[b+name+'.'+key]=torch.tensor(value,dtype=D)
add('model.embed_vision.embedding_projection.weight',(2,4),7)
add('model.language_model.embed_tokens.weight',(3,2),3)
add('model.language_model.norm.weight',(2,),norm=True)
t='model.language_model.layers.0.'
for name in ['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm','self_attn.q_norm','self_attn.k_norm']:add(t+name+'.weight',(2,),norm=True)
for name in ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']:add(t+name+'.weight',(2,2))
header={};data=bytearray()
for name,x in W.items():
    raw=x.contiguous().view(torch.uint16).flatten().tolist();start=len(data)
    data.extend(struct.pack('<'+'H'*len(raw),*raw))
    header[name]={'dtype':'BF16','shape':list(x.shape),'data_offsets':[start,len(data)]}
h=json.dumps(header,separators=(',',':')).encode();h+=b' '*((-len(h))%8)
(OUT/'model.safetensors').write_bytes(struct.pack('<Q',len(h))+h+data)
def flat(x):return x.float().flatten().tolist()
def norm(x,name=None):
    y=x.float()*torch.pow(x.float().square().mean(-1,keepdim=True)+1e-6,-0.5)
    if name:y=y*W[b+name+'.weight'].float()
    return y.to(D)
def linear(x,name):
    lo,hi,olo,ohi=clips[name]
    return F.linear(x.clamp(lo,hi),W[b+name+'.linear.weight']).clamp(olo,ohi)
def run(pixels,pos):
    h=F.linear((2*(pixels-0.5)).to(D),W[p+'patch_embedder.input_proj.weight'])
    table=W[p+'patch_embedder.position_embedding_table']
    h=h+(table[0,pos[:,0]]+table[1,pos[:,1]])
    patch=h.clone();n=norm(h,'input_layernorm')
    q=norm(linear(n,'self_attn.q_proj'),'self_attn.q_norm')
    k=norm(linear(n,'self_attn.k_proj'),'self_attn.k_norm')
    v=norm(linear(n,'self_attn.v_proj'))
    def rope(x):
        # hd=4: two axes each with one frequency, theta exponent zero.
        pieces=[]
        for axis in range(2):
            z=x[:,axis*2:axis*2+2];angle=pos[:,axis].float().unsqueeze(-1)
            co=angle.cos().to(D);si=angle.sin().to(D)
            pieces.append(z*co+torch.stack([-z[:,1],z[:,0]],-1)*si)
        return torch.cat(pieces,-1)
    q,k=rope(q),rope(k)
    # Source eager attention: BF16 scores, float32 softmax, BF16 probabilities.
    scores=q@k.T
    probs=scores.float().softmax(-1).to(D)
    a=probs@v
    attention=a.clone()
    h=h+norm(linear(a,'self_attn.o_proj'),'post_attention_layernorm')
    n=norm(h,'pre_feedforward_layernorm')
    a=F.gelu(linear(n,'mlp.gate_proj'),approximate='tanh')*linear(n,'mlp.up_proj')
    mlp=norm(linear(a,'mlp.down_proj'),'post_feedforward_layernorm')
    h=h+mlp
    width=int(pos[:,0].max())+1
    idx=pos[:,0]//3+(width//3)*(pos[:,1]//3)
    pool=(F.one_hot(idx).float().T/9)@h.float()
    pool=(pool.to(D).float()*2).to(D)
    out=F.linear(norm(pool),W['model.embed_vision.embedding_projection.weight'])
    return dict(pixels=flat(pixels),positions=pos.flatten().tolist(),patch=flat(patch),attention=flat(attention),mlp=flat(mlp),encoder=flat(h),pooled=flat(pool),projected=flat(out))
config=dict(hidden_size=4,num_attention_heads=1,num_key_value_heads=1,head_dim=4,num_hidden_layers=1,intermediate_size=6,patch_size=16,pooling_kernel_size=3,position_embedding_size=32,rms_norm_eps=1e-6,use_clipped_linears=True,hidden_activation='gelu_pytorch_tanh')
i=torch.arange(18*768);pixels=((i*13+i//11)%256).float().reshape(18,768)/255
pos=torch.tensor([[x,y] for y in range(3) for x in range(6)])
small=run(pixels,pos)
i=torch.arange(384*384*3);rgb=((i*17+i//7)%256).reshape(384,384,3)
pixels=(rgb.float()*(1.0/255)).reshape(24,16,24,16,3).permute(0,2,1,3,4).reshape(576,768)
pos=torch.tensor([[x,y] for y in range(24) for x in range(24)])
image=run(pixels,pos)
(OUT/'oracle.json').write_text(json.dumps(dict(config=config,small=small,image_projected=image['projected']),separators=(',',':'))+'\n')
print(torch.__version__,torch.version.git_version,'CPU only; equations, not full HF model')
print('fixture bytes',sum(f.stat().st_size for f in OUT.iterdir()))
