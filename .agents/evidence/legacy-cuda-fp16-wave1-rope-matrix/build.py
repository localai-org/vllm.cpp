exec(open('/tmp/fp16-independent-review/driver.py').read())
r=pathlib.Path('/tmp/lumen-fp16-wave1-matrix');original=pathlib.Path('/tmp/fp16-independent-review');d=pathlib.Path('/tmp/fp16-matrix-repair');d.mkdir(exist_ok=True)
(d/'base.a').symlink_to(original/'base.a') if not (d/'base.a').exists() else None
objects=[str(original/(pathlib.Path(m).name+'.o')) for m in mods]+[str(original/'reference.o')]
comp('tests/vt/test_cuda_fp16_scalar.cpp','test')
link('scalar',objects)
s=(r/'src/vt/ops.cpp').read_text()
needle='  reinterpret_cast<RopeFromCacheFn>(\n      GetOp(OpId::kRopeFromCache, q.device.type))'
assert s.count(needle)==1
for dt in ['BF16','F32','F16']:
 mutant=s.replace(needle,'  if (q_states.dtype == DType::k'+dt+') throw std::runtime_error("mutation removed '+dt+' cached production dispatch");\n'+needle)
 comp('src/vt/ops.cpp','mut-'+dt,source=mutant)
 link('mut-'+dt,[str(d/('mut-'+dt+'.o')) if m=='src/vt/ops.cpp' else str(original/(pathlib.Path(m).name+'.o')) for m in mods]+[str(original/'reference.o')])
# Old test + BF16 mutant establishes that the original regression misses this branch.
old=subprocess.check_output(['git','show','6c785c4a8:tests/vt/test_cuda_fp16_scalar.cpp'],cwd=r,text=True)
comp('tests/vt/test_cuda_fp16_scalar.cpp','old-test',source=old)
(d/'new-test.o').write_bytes((d/'test.o').read_bytes());(d/'test.o').write_bytes((d/'old-test.o').read_bytes())
link('old-mut-BF16',[str(d/'mut-BF16.o') if m=='src/vt/ops.cpp' else str(original/(pathlib.Path(m).name+'.o')) for m in mods]+[str(original/'reference.o')])
(d/'test.o').write_bytes((d/'new-test.o').read_bytes())
