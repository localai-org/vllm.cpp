import torch, pathlib, hashlib, json, math
torch.set_num_threads(2)
root=pathlib.Path(__file__).resolve().parents[2]
assert torch.__version__.split('+')[0] == '2.11.0'
assert torch.version.git_version == '70d99e998b4955e0049d13a98d77ae1b14db1f45'
rows=[]
fixtures=root/'tests/fixtures/gemma4_e4b_resize'
fixtures.mkdir(exist_ok=True)
for n,h,w in [(70,17,23),(140,901,713),(280,31,117),(560,1201,1403),(1120,19,29),(70,1,65536),(70,65536,1),(70,384,401),(70,401,384)]:
 s=math.sqrt(n*2304/(h*w)); oh=int(s*h//48)*48; ow=int(s*w//48)*48
 if oh==0: oh=48;ow=min(w//h*48,n*48)
 elif ow==0: ow=48;oh=min(h//w*48,n*48)
 y=torch.arange(h).reshape(h,1,1);x=torch.arange(w).reshape(1,w,1);c=torch.arange(3).reshape(1,1,3)
 a=((y*7+x*13+c*83)%256).to(torch.uint8)
 # hard edges force cubic overshoot in addition to ramp discontinuities
 a[:, :w//2, 0]=0;a[:,w//2:,0]=255
 b=torch.nn.functional.interpolate(a.permute(2,0,1).unsqueeze(0),size=(oh,ow),mode='bicubic',align_corners=False,antialias=True).squeeze(0).permute(1,2,0).contiguous()
 raw=bytes(b.flatten().tolist())
 (fixtures/f'{n}-{h}-{w}.rgb').write_bytes(raw)
 v=14695981039346656037
 for z in raw:v=((v^z)*1099511628211)&((1<<64)-1)
 rows.append(f'{{{n},{h},{w},{oh},{ow},{v}ULL}},')
 print(n,h,w,oh,ow,hashlib.sha256(raw).hexdigest(),v)
(root/'tests/vllm/multimodal/gemma4_resize_golden.inc').write_text('// Generated independently by Torch 2.11.0 CPU uint8 ATen bicubic AA; see resize evidence.\n'+'\n'.join(rows)+'\n')
