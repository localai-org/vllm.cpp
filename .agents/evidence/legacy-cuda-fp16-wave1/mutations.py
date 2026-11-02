exec(open('/tmp/lumen-fp16-build.py').read().split('mode=sys.argv[1]')[0])
from concurrent.futures import ThreadPoolExecutor
import hashlib
cases=[]
def case(label,rel,old,new,test):
 s=(r/rel).read_text();assert old in s,label;cases.append((label,rel,s.replace(old,new,1),test))
case('mut-provider','src/vt/op_provider.cpp','if (selected != nullptr && std::strcmp(slot.providers[i].name, selected) == 0)','if (selected != nullptr)','*admission belongs*')
case('mut-registration','src/vt/cuda/cuda_glue.cu','RegisterOpScalarType(OpId::kCastF32, DeviceType::kCUDA, kNativeProviderName, DType::kF16);','/* typed registration deleted */','*reverse cast*')
case('mut-cast','src/vt/cuda/cuda_glue.cu','return __half2float(p[i]);','return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(p)[i]);','*reverse cast*')
case('mut-residual','src/vt/cuda/cuda_ops.cu','template <> __device__ inline float ResRound<__half>(float v) { return __half2float(__float2half_rn(v)); }','template <> __device__ inline float ResRound<__half>(float v) { return v; }','*rounds the residual before variance*')
case('mut-store','src/vt/cuda/cuda_ops.cu','p[i] = __float2half_rn(v);','reinterpret_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16(v);','*ordinary and quantized embedding*')
case('mut-merged-silu','src/vt/cuda/cuda_ops.cu','static_cast<float>(static_cast<Tin>(gate / (1.0f + expf(-gate))))','gate / (1.0f + expf(-gate))','*SiLU rounds before multiplication exactly*')
case('mut-split-silu','src/vt/cuda/cuda_moe.cu','static_cast<float>(static_cast<Tg>(g / (1.0f + expf(-g))))','g / (1.0f + expf(-g))','*SiLU rounds before multiplication exactly*')
with ThreadPoolExecutor(max_workers=2) as pool:
 list(pool.map(lambda c:comp(c[1],c[0],source=c[2]),cases))
for label,rel,source,test in cases:
 objs=[str(d/((label if m==rel else pathlib.Path(m).name)+'.o')) for m in mods]+[str(d/'reference.o')]
 link(label,objs)
 with (d/(label+'-run.log')).open('w') as f:
  p=subprocess.run(['flock','/home/user/gpu.lock',str(d/label),'--test-case='+test],stdout=f,stderr=subprocess.STDOUT)
  f.write('\nMUTATION_EXIT '+str(p.returncode)+'\n')
 assert p.returncode==1,(label,p.returncode)
 print(label,'detected',flush=True)
 (d/label).unlink()
assert subprocess.check_output(['git','status','--porcelain'],cwd=r,text=True)==''
