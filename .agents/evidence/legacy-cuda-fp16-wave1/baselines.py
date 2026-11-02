exec(open('/tmp/lumen-fp16-build.py').read().split('mode=sys.argv[1]')[0])
for rel,label in [('tests/vt/test_cuda_ops.cpp','cuda-ops-test'),('tests/vt/test_cuda_quant_dot.cpp','quant-dot-test'),('tests/vt/test_op_provider.cpp','provider-test')]:
 comp(rel,label)
for target in ['test_cuda_ops','test_cuda_quant_dot','test_cuda_legacy_matmul','test_cuda_legacy_public','test_cuda_legacy_intrinsics','test_cuda_legacy_refusals','test_op_provider']:
 c=subprocess.check_output(['ninja','-C',str(b),'-t','commands','tests/'+target],text=True).strip().splitlines()[-1]
 a=[x for x in shlex.split(c) if x not in [':','&&'] and not x.startswith('-Wl,--dependency-file=')]
 if target!='test_cuda_legacy_intrinsics':
  a=[str(d/'base.a') if x in ('libvllm.a',str(b/'libvllm.a')) else x.replace(str(b/'libvllm.a'),str(d/'base.a')) for x in a]
  a=[str(d/'cuda-ops-test.o') if x.endswith('test_cuda_ops.cpp.o') else str(d/'provider-test.o') if x.endswith('test_op_provider.cpp.o') else str(d/'quant-dot-test.o') if x.endswith('test_cuda_quant_dot.cpp.o') else x for x in a]
  a[1:1]=[str(d/(pathlib.Path(m).name+'.o')) for m in mods]
 a[a.index('-o')+1]=str(d/target)
 run(a,target+'-link')
 args=['--test-case-exclude=*published MoE geometry*'] if target=='test_cuda_quant_dot' else []
 run(['flock','/home/user/gpu.lock',str(d/target)]+args,target+'-run')
