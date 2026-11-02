import pathlib,subprocess,json,shlex,sys,os
r=pathlib.Path('/tmp/lumen-fp16-wave1');b=pathlib.Path('/tmp/lumen-build-legacy');d=pathlib.Path('/tmp/lumen-fp16-wave1-build');d.mkdir(exist_ok=True);db=json.load(open(b/'compile_commands.json'))
mods=['src/vt/ops.cpp','src/vt/op_provider.cpp','src/vt/cuda/cuda_ops.cu','src/vt/cuda/cuda_layernorm.cu','src/vt/cuda/cuda_moe.cu','src/vt/cuda/cuda_glue.cu','src/vt/cuda/cuda_matmul.cu','src/vt/cuda/cuda_quant_dot.cu']
def run(a,label):
 with (d/(label+'.log')).open('w') as f:p=subprocess.run(a,cwd=b,stdout=f,stderr=subprocess.STDOUT)
 print(label,p.returncode,flush=True)
 if p.returncode:print((d/(label+'.log')).read_text()[-5000:]);raise SystemExit(p.returncode)
def comp(rel,label=None,arch=None,source=None):
 label=label or pathlib.Path(rel).name;is_test=rel.startswith('tests/');base=('src/vt/cuda/cuda_ops.cu' if rel.endswith('.cu') else 'tests/vt/test_cuda_ops.cpp') if is_test else rel;e=next(z for z in db if z['file'].endswith('/'+base));a=shlex.split(e['command']);a=[str(r/rel) if x==e['file'] else x.replace('/tmp/lumen-pascal',str(r)) for x in a];a[1:1]=['-I'+str((r/rel).parent),'-I'+str(r/'src')]
 if rel.endswith('.cu'):
  a=[x for x in a if not (x.startswith('--generate-code=') or x.startswith('-gencode='))];a.insert(1,'--generate-code=arch=compute_'+str(arch or 61)+',code=sm_'+str(arch or 61))
 
 if source is not None:
  variant=d/(label+('.cu' if rel.endswith('.cu') else '.cpp'));variant.write_text(source);a=[str(variant) if x==str(r/rel) else x for x in a]
 a[a.index('-o')+1]=str(d/(label+'.o'));run(a,label+'-compile');return str(d/(label+'.o'))
def link(label,objs):
 c=subprocess.check_output(['ninja','-C',str(b),'-t','commands','tests/test_cuda_ops'],text=True).strip().splitlines()[-1];a=[x for x in shlex.split(c) if x not in [':','&&'] and not x.startswith('-Wl,--dependency-file=')];a=[str(d/'test.o') if x.endswith('test_cuda_ops.cpp.o') else str(d/'base.a') if x in ('libvllm.a',str(b/'libvllm.a')) else x.replace(str(b/'libvllm.a'),str(d/'base.a')) for x in a];a[a.index('-o')+1]=str(d/label);a[1:1]=objs;run(a,label+'-link')
mode=sys.argv[1]
if mode=='red':
 comp('tests/vt/test_cuda_fp16_scalar.cpp','test');(d/'base.a').symlink_to(b/'libvllm.a') if not (d/'base.a').exists() else None;link('red',[])
elif mode=='compile':
 for rel in mods:comp(rel)
elif mode=='link':
 if (d/'base.a').exists():(d/'base.a').unlink()
 # A thin archive references unchanged object files rather than copying 600 MB.
 entries=[]
 for e in db:
  if e['file'].startswith('/tmp/lumen-pascal/src/') and not any(e['file'].endswith('/'+m) for m in mods):
   a=shlex.split(e['command']);o=pathlib.Path(e['directory'])/a[a.index('-o')+1]
   if o.exists():entries.append(str(o))
 missing = ['qwen3_5.cpp.o', 'qwen3_dflash.cpp.o', 'qwen3_dflash2.cpp.o', 'qwen3_dspark.cpp.o']
 for member in missing:
  target=d/member
  if not target.exists():
   with target.open('wb') as f:subprocess.run(['ar','p',str(b/'libvllm.a'),member],stdout=f,check=True)
  entries.append(str(target))
 run(['ar','crsT',str(d/'base.a')]+entries,'thin-archive');link('green',[str(d/(pathlib.Path(m).name+'.o')) for m in mods]+[str(d/'reference.o')])
elif mode=='test':
 run(['flock','/home/user/gpu.lock',str(d/'green')]+sys.argv[2:],'green-run')
