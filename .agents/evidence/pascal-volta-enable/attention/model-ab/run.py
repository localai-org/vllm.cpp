import pathlib, subprocess, time, json, threading, ctypes, os, re, hashlib, datetime, statistics
import numpy as np
out=pathlib.Path('/tmp/lumen-real-gates/attention-ab')
binary=pathlib.Path('/tmp/pascal-attention/actual-frozen')
root=pathlib.Path('/tmp/lumen-real-gates/tiny')
model='/tmp/lumen-models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf'
nv=ctypes.CDLL('libnvidia-ml.so.1');assert nv.nvmlInit_v2()==0
handle=ctypes.c_void_p();assert nv.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(handle))==0
class Mem(ctypes.Structure):_fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
def memory():
 m=Mem();assert nv.nvmlDeviceGetMemoryInfo(handle,ctypes.byref(m))==0;return m.used
manifest={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'binary':str(binary),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'source_head':'22d433baf','model':model,'model_sha256':'9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0','context':256,'block_size':16,'num_blocks':32,'sampling':{'temperature':0,'ignore_eos':True,'seed':0},'mutex':'/home/user/gpu.lock held for complete campaign','order':'p16g16 ON/OFF,p16g64 OFF/ON,p128g16 ON/OFF,p128g64 OFF/ON','warm_repetitions':3,'cold_repetition':0,'memory_sample_ms':10,'inherited_attention_env':{k:v for k,v in os.environ.items() if k.startswith(('VT_ATTN','VT_FA2'))}}
manifest['gpu_before']=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
manifest['processes_before']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
allrows=[]
for axis,(p,t) in enumerate([(16,16),(16,64),(128,16),(128,64)]):
 oracle=np.load(root/f'p{p}-t{t}'/'greedy_ids.npy')
 assert oracle.shape==(3,t) and np.array_equal(oracle[0],oracle[1]) and np.array_equal(oracle[0],oracle[2])
 for enabled in ([True,False] if axis%2==0 else [False,True]):
  label=f'p{p}-g{t}-'+('on' if enabled else 'off')
  env=dict(os.environ);env['VT_ATTN_DECODE_OPT']='1' if enabled else '0'
  cmd=[str(binary),model,str(root/f'p{p}.i32'),str(t),'4']
  before=memory();peak=[before];stop=threading.Event()
  def monitor():
   while not stop.wait(.01):peak[0]=max(peak[0],memory())
  th=threading.Thread(target=monitor);th.start();start=time.monotonic()
  with (out/(label+'.log')).open('w') as f:result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
  wall=time.monotonic()-start;stop.set();th.join()
  text=(out/(label+'.log')).read_text()
  matches=re.findall(r'repetition=(\d+) prompt_tokens=(\d+) generated_tokens=(\d+) seconds=([\d.]+) ids=([\d,]+)',text)
  row={'axis':f'p{p}-g{t}','enabled':enabled,'label':label,'command':cmd,'environment':{'VT_ATTN_DECODE_OPT':env['VT_ATTN_DECODE_OPT']},'returncode':result.returncode,'wall_seconds':wall,'baseline_gpu_memory_bytes':before,'peak_gpu_memory_bytes':peak[0],'incremental_peak_bytes':peak[0]-before,'oracle_file':str(root/f'p{p}-t{t}'/'greedy_ids.npy'),'oracle_sha256':hashlib.sha256((root/f'p{p}-t{t}'/'greedy_ids.npy').read_bytes()).hexdigest(),'repetitions':[]}
  assert result.returncode==0,(label,result.returncode,text[-1000:]);assert len(matches)==4,(label,matches)
  for rep,pp,tt,seconds,ids in matches:
   actual=np.array([int(x) for x in ids.split(',')],dtype=np.int32)
   assert int(pp)==p and int(tt)==t and np.array_equal(actual,oracle[0]),(label,rep,'tokens differ')
   row['repetitions'].append({'index':int(rep),'seconds':float(seconds),'tokens':len(actual),'oracle_exact':True,'cold':int(rep)==0,'tokens_per_second':t/float(seconds)})
  warm=[x['seconds'] for x in row['repetitions'] if not x['cold']]
  row['warm_median_seconds']=statistics.median(warm);row['warm_min_seconds']=min(warm);row['warm_max_seconds']=max(warm);row['warm_mean_seconds']=statistics.mean(warm);row['warm_median_tokens_per_second']=t/row['warm_median_seconds']
  (out/(label+'.json')).write_text(json.dumps(row,indent=2)+'\n');allrows.append(row)
  print(json.dumps({'label':label,'returncode':result.returncode,'warm_median_seconds':row['warm_median_seconds'],'memory_peak':peak[0],'all_four_exact':True}),flush=True)
summary=[]
for p,t in [(16,16),(16,64),(128,16),(128,64)]:
 on=next(x for x in allrows if x['axis']==f'p{p}-g{t}' and x['enabled']);off=next(x for x in allrows if x['axis']==f'p{p}-g{t}' and not x['enabled'])
 summary.append({'axis':on['axis'],'on_warm_median_seconds':on['warm_median_seconds'],'off_warm_median_seconds':off['warm_median_seconds'],'on_tokens_per_second':on['warm_median_tokens_per_second'],'off_tokens_per_second':off['warm_median_tokens_per_second'],'throughput_on_over_off':off['warm_median_seconds']/on['warm_median_seconds'],'latency_reduction_fraction':1-on['warm_median_seconds']/off['warm_median_seconds'],'on_incremental_peak_bytes':on['incremental_peak_bytes'],'off_incremental_peak_bytes':off['incremental_peak_bytes'],'token_exact_repetitions':8})
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
manifest['gpu_after']=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
manifest['processes_after']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
manifest['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
