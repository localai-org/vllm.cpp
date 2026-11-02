exec(open('/tmp/metadata-lib.py').read())
comp('tests/vllm/v1/attention/test_attn_validate_configuration.cpp','test_attn_validate_configuration')
rel='src/vllm/v1/attention/cuda_legacy.cpp';s=(root/rel).read_text()
cases=[('green',s),('head',s.replace('return size > 0 && size <= 256 && size % 8 == 0;', 'return size > 0;')),('block',s.replace('override { return {16}; }','override { return {1}; }')),('connector',s.replace('bool supports_kv_connector() const override { return false; }','bool supports_kv_connector() const override { return true; }')),('bf16',s.replace('return {"auto", "bfloat16"};','return {"auto"};')),('cacheblock',s.replace('if (size % 16 != 0) throw','if (false) throw')),('dtype',s.replace('return {vt::DType::kBF16, vt::DType::kF32};','return {vt::DType::kBF16};')),('fp8',s.replace('return {"auto", "bfloat16"};','return {"auto", "bfloat16", "fp8"};')),('name',s.replace('return "CUDA_LEGACY";', 'return "WRONG_NAME";')),('shape',s.replace('return {blocks, 2, size, heads, dim};','return {blocks, size, 2, heads, dim};')),('cap',s.replace('(cap.major == 7 && cap.minor == 0)','(cap.major == 7 && (cap.minor == 0 || cap.minor == 5))')),('encoder',s.replace('return type == "decoder";', '(void)type; return true;'))]
for method in ['is_mla','is_sparse','supports_sink','supports_mm_prefix','supports_per_head_quant_scales','supports_sliding_window','supports_non_causal','supports_batch_invariance','supports_pcp']:
 cases.append((method,s.replace(' public:',f' public:\n  bool {method}() const override {{ return true; }}',1)))
for label,text in cases:
 comp(rel,'descriptor',text)
 exe=link('test_attn_validate_configuration',label)
 result=run([exe],label+'-run')
 if label=='green':assert result==0,(label,result)
 else:assert result!=0,(label,'mutation survived')
# Restore the immutable production descriptor object and rerun unchanged full suite.
comp(rel,'descriptor');exe=link('test_attn_validate_configuration','restored');assert run([exe],'restored-run')==0
