import json,pathlib,shlex,subprocess,shutil,sys
root=pathlib.Path('/tmp/lumen-cuda-smem-repair'); build=pathlib.Path('/tmp/lumen-build-legacy'); d=pathlib.Path('/tmp/smem-repair'); d.mkdir(exist_ok=True)
db=json.load(open(build/'compile_commands.json'))
def run(a,log):
 with open(d/(log+'.log'),'w') as f:
  p=subprocess.run(a,cwd=build,stdout=f,stderr=subprocess.STDOUT)
 print(log,p.returncode,flush=True)
 if p.returncode: print((d/(log+'.log')).read_text()[-2000:],flush=True)
 return p.returncode
def comp(rel,label,text=None,arch=None):
 e=next((x for x in db if x['file'].endswith('/'+rel)),None)
 if e is None:e=next(x for x in db if x['file'].endswith('/src/vllm/v1/attention/backend.cpp'))
 src=root/rel
 if text is not None:src=d/(label+src.suffix);src.write_text(text)
 a=shlex.split(e['command']);a=[str(src) if x==e['file'] else x.replace('/tmp/lumen-pascal',str(root)) for x in a]
 a.insert(1,'-I'+str((root/rel).parent));a[a.index('-o')+1]=str(d/(label+'.o'))
 for opt in ['-MF','-MT']:
  if opt in a:a[a.index(opt)+1]=str(d/(label+'.d'))
 if arch:
  a=[x for x in a if not x.startswith('-gencode=')]
  for i in range(len(a)-1,-1,-1):
   if a[i]=='-gencode':del a[i:i+2]
  a[1:1]=['-gencode',f'arch=compute_{arch},code=sm_{arch}']
 if run(a,label+'-compile'):raise RuntimeError('compile failed '+label)
 return str(d/(label+'.o'))
def link(test,label,replace={},extra=[]):
 lib=d/(label+'.a');shutil.copyfile(d/'frozen.a',lib)
 removes=['cuda.cpp.o','cuda_legacy.cpp.o']+list(replace)
 subprocess.run(['ar','d',str(lib)]+removes,check=True)
 objs=[str(d/'platform.o')]+extra
 if label!='public-noreg':objs.append(str(d/'descriptor.o'))
 objs+=list(replace.values())
 c=subprocess.check_output(['ninja','-C',str(build),'-t','commands','tests/test_cuda_legacy_capi'],text=True).strip().splitlines()[-1]
 a=[x for x in shlex.split(c) if x not in [':','&&']]
 a=[str(d/(test+'.o')) if x.endswith('test_cuda_legacy_capi.cpp.o') else x.replace(str(build/'libvllm.a'),str(lib)) if str(build/'libvllm.a') in x else str(lib) if x=='libvllm.a' else x for x in a]
 a[a.index('-o')+1]=str(d/label)
 if '-Wl,--dependency-file=tests/CMakeFiles/test_cuda_legacy_capi.dir/link.d' in a:a.remove('-Wl,--dependency-file=tests/CMakeFiles/test_cuda_legacy_capi.dir/link.d')
 a[1:1]=objs
 if test in ['test_cuda_ops','test_attn_validate_configuration']:a.insert(1,str(build/'tests/libvllm_test_main.a'))
 if run(a,label+'-link'):raise RuntimeError('link failed '+label)
 return str(d/label)
def gpu(exe,label,args=[]):return run(['flock','/home/user/gpu.lock',exe]+args,label)

comp('tests/vt/test_cuda_legacy_public.cpp','test_cuda_legacy_public')
exe=link('test_cuda_legacy_public',sys.argv[1],{'cuda_paged_attn.cu.o':sys.argv[2]} if len(sys.argv)>2 else {})
sys.exit(gpu(exe,sys.argv[1]+'-run'))
