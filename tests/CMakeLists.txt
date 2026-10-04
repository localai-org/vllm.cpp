add_library(vllm_test_main STATIC doctest_main.cpp)
target_include_directories(vllm_test_main SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/third_party)

if(UNIX)
  add_test(NAME test_triton_aot_drift_contract
    COMMAND bash "${CMAKE_SOURCE_DIR}/tests/scripts/test_triton_aot_drift.sh")
endif()

# CPU-only benchmark-harness contract. The tools deliberately use only the
# Python standard library; image/model/GPU preflights remain dgx-only.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME test_serve_low_tools
  COMMAND ${Python3_EXECUTABLE} -m unittest discover
    -s "${CMAKE_SOURCE_DIR}/tests/tools" -p "test_*.py")
set_tests_properties(test_serve_low_tools PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}")

function(vllm_cpp_add_test name)
  add_executable(${name} ${ARGN})
  # tests/ on the include path so every suite can reach the shared test helpers
  # under tests/support/ as "support/<name>.h". No header name under tests/
  # collides with one under include/ or src/, so nothing else resolves here.
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  if(TARGET vllm::sanitize_test_shared)
    target_link_libraries(${name} PRIVATE vllm::sanitize_test_shared vllm_test_main)
  else()
    target_link_libraries(${name} PRIVATE vllm::vllm vllm_test_main)
  endif()
  vllm_cpp_set_warnings(${name})
  add_test(NAME ${name} COMMAND ${name})
  # A gate that cannot run must not report success. doctest exits 0 after a
  # TEST_CASE returns early, printing "assertions: 0 | 0 passed | 0 failed" and
  # "Status: SUCCESS!" — indistinguishable, in a log or a `&&` chain, from a gate
  # that loaded a model and matched an oracle (issue #463; multimodal-speed.md
  # §17.7 hit exactly this). A test whose preconditions are absent exits 77
  # instead and CTest reports it **Skipped**. Inert for any test that never
  # returns 77, which today is all but test_voxtral_e2e.
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

# The gate on the golden-comparison helper itself (issue #449): a NaN operand
# must be a FAILURE, never the 0.0 that both historical spellings returned.
vllm_cpp_add_test(test_max_abs_diff support/test_max_abs_diff.cpp)
# ...and the wrapper must RAISE, not merely return +infinity, so the
# "these must differ" callers cannot read a NaN as a difference. That case is
# skipped in the normal run because a raised failure would count as one; here it
# is forced on and REQUIRED to exit non-zero. Without the guard this entry fails.
add_test(NAME test_max_abs_diff_nan_raises
  COMMAND test_max_abs_diff --no-skip --test-case=*RAISES*)
set_tests_properties(test_max_abs_diff_nan_raises PROPERTIES WILL_FAIL TRUE)

vllm_cpp_add_test(test_version vllm/test_version.cpp)
set(_vllm_cpp_expected_build_version "${PROJECT_VERSION}")
if(DEFINED VLLM_CPP_BUILD_VERSION)
  set(_vllm_cpp_expected_build_version "${VLLM_CPP_BUILD_VERSION}")
endif()
target_compile_definitions(test_version PRIVATE
  VLLM_CPP_EXPECTED_BUILD_VERSION="${_vllm_cpp_expected_build_version}")
vllm_cpp_add_test(test_sampling_params vllm/test_sampling_params.cpp)
vllm_cpp_add_test(test_punica_cpu vllm/lora/test_punica_cpu.cpp)
vllm_cpp_add_test(test_lora_layers vllm/lora/test_lora_layers.cpp)
vllm_cpp_add_test(test_lora_mapping vllm/lora/test_lora_mapping.cpp)
vllm_cpp_add_test(test_peft_helper vllm/lora/test_peft_helper.cpp)
vllm_cpp_add_test(test_lora_checkpoints vllm/lora/test_lora_checkpoints.cpp)
vllm_cpp_add_test(test_plugin_system
  vllm/plugins/test_plugin_system.cpp
  vllm/plugins/toy_model_plugin.cpp)
vllm_cpp_add_test(test_hf_config vllm/test_hf_config.cpp)
vllm_cpp_add_test(test_generation_config vllm/config/test_generation_config.cpp)
# ENG-HF-MODEL-DOWNLOAD W2 (#1280): the HuggingFace cache layout. No socket,
# so it is not behind VLLM_CPP_SERVER; the fake-hub half is, below.
vllm_cpp_add_test(test_hf_cache vllm/transformers_utils/test_hf_cache.cpp)
vllm_cpp_add_test(test_safetensors vllm/test_safetensors.cpp)
vllm_cpp_add_test(test_load_direct_upload vllm/test_load_direct_upload.cpp)
vllm_cpp_add_test(test_gguf vllm/test_gguf.cpp)
vllm_cpp_add_test(test_nvfp4_dequant vllm/test_nvfp4_dequant.cpp)
vllm_cpp_add_test(test_awq_gptq_dequant vllm/test_awq_gptq_dequant.cpp)
vllm_cpp_add_test(test_mxfp4_dequant vllm/test_mxfp4_dequant.cpp)
vllm_cpp_add_test(test_ct_nvfp4_emulation vllm/test_ct_nvfp4_emulation.cpp)
# MODEL-TEXT-nemotron-h W1 (#517): the ModelOpt MIXED_PRECISION per-module
# quant-algo resolver. Two arms, deliberately two BINARIES: the curated fixture
# arm is always-on, and the exhaustive real-config arm exits 77 (Skipped) when
# the 30B checkpoint is not staged. Folding them into one target would let the
# skip swallow the always-on gate as well.
#
# The resolver header is INTERNAL (src/, not include/) until a loader consumes
# it, so both targets take ${CMAKE_SOURCE_DIR}/src on the include path — the
# same arrangement test_linear_method and test_model_registry already use.
vllm_cpp_add_test(test_modelopt_mixed_precision
  vllm/model_executor/layers/quantization/test_modelopt_mixed_precision.cpp)
target_include_directories(test_modelopt_mixed_precision PRIVATE
  ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(test_modelopt_mixed_precision PRIVATE
  MODELOPT_MIXED_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures/modelopt_mixed_precision")
vllm_cpp_add_test(test_modelopt_mixed_precision_checkpoint
  vllm/model_executor/layers/quantization/test_modelopt_mixed_precision_checkpoint.cpp)
# tests/parity as well as src/: the exhaustive arm resolves its checkpoint
# through the SINGLE pinned parity::Nemotron35LightningSnapshot() rather than
# joining $CHECKPOINT_ROOT by hand, so that one resolver owns the revision pin
# for both env spellings (#517 LOW-3) -- the same arrangement test_qwen36_weights
# uses for the 35B pin.
target_include_directories(test_modelopt_mixed_precision_checkpoint PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/tests/parity)
vllm_cpp_add_test(test_gguf_dequant vllm/test_gguf_dequant.cpp)
vllm_cpp_add_test(test_gguf_nvfp4 vllm/test_gguf_nvfp4.cpp)
# gguf_nvfp4_goldens.inc lives next to the test source.
target_include_directories(test_gguf_nvfp4 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm)
vllm_cpp_add_test(test_qwen36_weights vllm/test_qwen36_weights.cpp)
# GATE-PIN-UNPINNED-SNAPSHOTS (#471): resolves the 35B checkpoint through the
# pinned parity::HfSnapshot, so it needs tests/parity on the include path.
target_include_directories(test_qwen36_weights PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity)
vllm_cpp_add_test(test_gguf_qwen36_loader vllm/test_gguf_qwen36_loader.cpp)
vllm_cpp_add_test(test_gguf_keep_quant vllm/test_gguf_keep_quant.cpp)
vllm_cpp_add_test(test_model_loader_gguf vllm/test_model_loader_gguf.cpp)
# The dots3-note cases compare FromModelDir's thrown bytes with
# `vllm::Dots3NoteGgufRefusal()`, declared in the MODEL-PRIVATE header
# src/vllm/model_executor/models/dots3_note.h (#2882 W9a). Same arrangement
# test_dots3_note_scaffold uses.
target_include_directories(test_model_loader_gguf PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_model_registry vllm/models/test_model_registry.cpp)
# MODEL-TEXT-mimo-v2 W1 gate (ISSUE-LOCAL-01M3F82S8ZYCTTDSKPFF5PGAH7):
# config parses, KV-cache spec builds, model is discoverable.
vllm_cpp_add_test(test_mimov2_w1 vllm/models/test_mimov2_w1.cpp)
# MODEL-TEXT-kolibri-1 W1 gate: config parse + registry + KV-cache spec +
# FP8-block loader on a synthetic checkpoint + the REAL-index manifest
# (kolibri1_manifest.inc, captured by scripts/gen-kolibri1-manifest.py).
vllm_cpp_add_test(test_kolibri1 vllm/models/test_kolibri1.cpp)
# kolibri1_weights.h is a MODEL-PRIVATE header under src/, not include/vllm/.
target_include_directories(test_kolibri1 PRIVATE ${CMAKE_SOURCE_DIR}/src)
# MODEL-TEXT-kolibri-1 W2 gate: the CPU forward against the in-test scalar
# reference (the kolibri1.py transcription), the ragged batch, the
# logits_indices gather, and the full-depth 50-layer structural run.
vllm_cpp_add_test(test_kolibri1_w2 vllm/models/test_kolibri1_w2.cpp)
target_include_directories(test_kolibri1_w2 PRIVATE ${CMAKE_SOURCE_DIR}/src)
# MODEL-TEXT-kolibri-1 W3 gate (the row's token gate): the CPU forward over
# the REAL fp8 checkpoint reproduces the transformers golden run
# (kolibri1_goldens.json, generated by scripts/gen-kolibri1-goldens.py).
vllm_cpp_add_test(test_kolibri1_w3 vllm/models/test_kolibri1_w3.cpp)
target_include_directories(test_kolibri1_w3 PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(test_kolibri1_w3 PRIVATE
  KOLIBRI1_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/kolibri1_goldens.json")
# mimo_v2_weights.h is a MODEL-PRIVATE header under src/, not include/vllm/.
vllm_cpp_add_test(test_mimov2_w2 vllm/models/test_mimov2_w2.cpp)
target_include_directories(test_mimov2_w2 PRIVATE ${CMAKE_SOURCE_DIR}/src)
# MODEL-TEXT-mimo-v2 W5 gate (ISSUE-LOCAL-01M3F82S8ZYCTTDSKPFF5PGAH7):
# end-to-end forward through the production registry entry point.
vllm_cpp_add_test(test_mimov2_forward vllm/models/test_mimov2_forward.cpp)
target_include_directories(test_mimov2_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_vocoder1d vllm/models/test_vocoder1d.cpp)
# For src/vt/cpu/cpu_conv1d_block.h: the multi-block case asserts the geometry
# it claims rather than assuming it (#1664). Same reach as test_host_parallel.
target_include_directories(test_vocoder1d PRIVATE ${CMAKE_SOURCE_DIR}/src)
# The bit-identity gate for the row-wise parallelisation of the host-reference
# kernels (#672). It carries VERBATIM copies of the pre-parallel serial loops as
# its oracle, so it reaches src/ for `host_parallel.h` (the size guard whose
# threshold every case must clear, or it would test the inline path twice) and
# for `Threadpool::SwapForTesting`, the same hook the vt determinism suites use.
# CPU-only.
vllm_cpp_add_test(test_host_parallel vllm/models/test_host_parallel.cpp)
target_include_directories(test_host_parallel PRIVATE ${CMAKE_SOURCE_DIR}/src)
# The accounting gate for the MiniMax-Music3 stage profiler (#672). It reaches
# src/ for `music3_profile.h`, which is an INTERNAL instrument and deliberately
# not on the public surface. CPU-only, no checkpoint.
vllm_cpp_add_test(test_music3_profile vllm/models/test_music3_profile.cpp)
target_include_directories(test_music3_profile PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_speech_engine vllm/multimodal/test_speech_engine.cpp)
vllm_cpp_add_test(test_indextts2_family vllm/models/test_indextts2_family.cpp)
vllm_cpp_add_test(test_campplus vllm/models/test_campplus.cpp)
vllm_cpp_add_test(test_indextts2_conditioning vllm/models/test_indextts2_conditioning.cpp)
vllm_cpp_add_test(test_emovec vllm/models/test_emovec.cpp)
vllm_cpp_add_test(test_w2vbert vllm/models/test_w2vbert.cpp)
vllm_cpp_add_test(test_fvq vllm/models/test_fvq.cpp)
vllm_cpp_add_test(test_codec_encoder vllm/models/test_codec_encoder.cpp)
vllm_cpp_add_test(test_w2v_fbank vllm/models/test_w2v_fbank.cpp)
vllm_cpp_add_test(test_wavenet vllm/models/test_wavenet.cpp)
vllm_cpp_add_test(test_dit_tail vllm/models/test_dit_tail.cpp)
vllm_cpp_add_test(test_dit_skip vllm/models/test_dit_skip.cpp)
vllm_cpp_add_test(test_dit_front vllm/models/test_dit_front.cpp)
vllm_cpp_add_test(test_dit_stack vllm/models/test_dit_stack.cpp)
vllm_cpp_add_test(test_bigvgan vllm/models/test_bigvgan.cpp)
vllm_cpp_add_test(test_indextts2_s2mel_loader vllm/models/test_indextts2_s2mel_loader.cpp)
vllm_cpp_add_test(test_indextts2_render vllm/models/test_indextts2_render.cpp)
vllm_cpp_add_test(test_indextts2_e2e vllm/models/test_indextts2_e2e.cpp)
vllm_cpp_add_test(test_indextts2_talker_loader vllm/models/test_indextts2_talker_loader.cpp)
vllm_cpp_add_test(test_vocos vllm/models/test_vocos.cpp)
vllm_cpp_add_test(test_lenreg vllm/models/test_lenreg.cpp)
vllm_cpp_add_test(test_cfm vllm/models/test_cfm.cpp)
vllm_cpp_add_test(test_adaln vllm/models/test_adaln.cpp)
vllm_cpp_add_test(test_dit vllm/models/test_dit.cpp)
vllm_cpp_add_test(test_talker vllm/models/test_talker.cpp)
vllm_cpp_add_test(test_tiktoken_bpe vllm/models/test_tiktoken_bpe.cpp)
vllm_cpp_add_test(test_indextts2_pipeline vllm/models/test_indextts2_pipeline.cpp)
target_include_directories(test_dit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_adaln PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_cfm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_lenreg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_vocos PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_fvq PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_codec_encoder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_w2v_fbank PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_wavenet PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_dit_tail PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_dit_front PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_dit_stack PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
# VT-CONV1D-MODEL-BLOCK (#1684): the time-block case reads `Conv1dTimeBlock` so
# it asserts the geometry it claims. Same reach as test_vocoder1d.
target_include_directories(test_bigvgan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models
  ${CMAKE_SOURCE_DIR}/src)
target_include_directories(test_w2vbert PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_campplus PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
vllm_cpp_add_test(test_gpt2 vllm/models/test_gpt2.cpp)
target_include_directories(test_gpt2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
vllm_cpp_add_test(test_deberta_v2 vllm/models/test_deberta_v2.cpp)
target_include_directories(test_deberta_v2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
vllm_cpp_add_test(test_gliner2 vllm/models/test_gliner2.cpp)
target_include_directories(test_gliner2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
vllm_cpp_add_test(test_gliner2_e2e vllm/models/test_gliner2_e2e.cpp)
target_compile_definitions(test_gliner2_e2e PRIVATE
  GLINER2_E2E_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/gliner2_e2e")
vllm_cpp_add_test(test_gliner2_ner vllm/models/test_gliner2_ner.cpp)
vllm_cpp_add_test(test_cua_s1 vllm/models/test_cua_s1.cpp)
target_include_directories(test_cua_s1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
vllm_cpp_add_test(test_modernbert vllm/models/test_modernbert.cpp)
target_include_directories(test_modernbert PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
vllm_cpp_add_test(test_laya vllm/models/test_laya.cpp)
target_include_directories(test_laya PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
vllm_cpp_add_test(test_laya_sequence vllm/models/test_laya_sequence.cpp)
target_include_directories(test_laya_sequence PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
vllm_cpp_add_test(test_minimax_h3 vllm/models/test_minimax_h3.cpp)
# VT-CONV1D-MODEL-BLOCK (#1684): the time-block case reads `Conv1dTimeBlock` so
# it asserts the geometry it claims. Same reach as test_vocoder1d.
target_include_directories(test_minimax_h3 PRIVATE ${CMAKE_SOURCE_DIR}/src)
# ROAD-V1-DIT-LORA: H3-specific contract name rewriting (model.diffusion_model.
# prefix) and load-time fusion through the shared dit_lora seam.
vllm_cpp_add_test(test_h3_lora vllm/models/test_h3_lora.cpp)
# ROAD-V1-LORA-RUNTIME: prompt tag parsing and name resolution for runtime LoRA.
vllm_cpp_add_test(test_dit_runtime_lora vllm/models/test_dit_runtime_lora.cpp)
# MiniMax-Music3 W1 (#672): the six-component diffusers-arm loader, gated against
# the REAL checkpoint's headers (minimax_music3_manifest.inc, 1012 tensors, no
# weight bytes) so CI never needs the 28.5 GB asset.
vllm_cpp_add_test(test_minimax_music3_loader vllm/models/test_minimax_music3_loader.cpp)
target_include_directories(test_minimax_music3_loader PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
# MiniMax-Music3 W2+W3 (#672): the AUTOREGRESSIVE half — prompt assembly, the
# semantic CFG logit pipeline, the learned 8-layer condition mix and the RVQ
# depth decoder — against goldens produced by EXECUTING upstream's own classes at
# reduced dimensions in float32 (minimax_music3_ar_goldens.inc, no weight bytes).
# The full-scale bf16 companion is tests/parity/test_minimax_music3_ar_real.cpp.
vllm_cpp_add_test(test_minimax_music3_ar vllm/models/test_minimax_music3_ar.cpp)
target_include_directories(test_minimax_music3_ar PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models
  # ...and src/ for `music3_profile.h`, the INTERNAL instrument that carries the
  # `ar.depth_device` / `ar.depth_host` buckets this suite drives on both sides
  # (#1839). Deliberately not a public header.
  ${CMAKE_SOURCE_DIR}/src)
# The FULL-SCALE companion: the REAL bf16 checkpoint driven on the REAL inputs,
# against the committed oracle capture (frame_hiddens / condition_chunk0 /
# rvq_codes). Checkpoint-gated on MUSIC3_CHECKPOINT or CHECKPOINT_ROOT; every
# case emits a loud SKIP when the 28.5 GB asset is absent, so CI still compiles,
# links and runs it.
vllm_cpp_add_test(test_minimax_music3_ar_real parity/test_minimax_music3_ar_real.cpp)
target_compile_definitions(test_minimax_music3_ar_real PRIVATE
  MUSIC3_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens/minimax_music3_oracle")
target_include_directories(test_minimax_music3_ar_real PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity)
# MiniMax-Music3 W2's remainder (#672): the 8.6B `Qwen3ForCausalLM` forward
# itself, driven through the ADDITIVE `Qwen3DenseModel::ForwardEmbeds` entry and
# TEACHER-FORCED on the capture's own rvq_codes, against frame_hiddens[:, :4096].
# Checkpoint-gated like its two siblings; every case emits a loud SKIP when the
# 28.5 GB asset is absent, so CI still compiles, links and runs it. SEPARATE from
# test_minimax_music3_ar_real because it stages 18.5 GB that gate does not need.
vllm_cpp_add_test(test_minimax_music3_llm_real parity/test_minimax_music3_llm_real.cpp)
target_compile_definitions(test_minimax_music3_llm_real PRIVATE
  MUSIC3_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens/minimax_music3_oracle")
target_include_directories(test_minimax_music3_llm_real PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity)
# MiniMax-Music3 W4+W5 (#672): the ACOUSTIC half — the flow-matching DiT, the
# FlowMatchEulerDiscreteScheduler with invert_sigmas, the CFG mix, the denoise
# loop's window bookkeeping and the DAC Flow-VAE vocoder — against goldens
# produced by EXECUTING upstream's own classes at reduced dimensions in float32
# (minimax_music3_acoustic_goldens.inc, no weight bytes). The full-scale fp32
# companion is tests/parity/test_minimax_music3_acoustic_real.cpp.
vllm_cpp_add_test(test_minimax_music3_acoustic vllm/models/test_minimax_music3_acoustic.cpp)
# Reaches src/ for `music3_profile.h` — an INTERNAL instrument, deliberately not
# on the public surface — because the intra-DiT spans (#1542) live inside
# `DitForwardDevice` and this is the only suite that drives that forward.
target_include_directories(test_minimax_music3_acoustic PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models
  ${CMAKE_SOURCE_DIR}/src)
# The FULL-SCALE companion: the REAL fp32 2.4B DiT and 0.054B vocoder driven on
# the REAL inputs, against the committed oracle capture (denoise_{first,last}_*,
# vocoder_input_chunk0, waveform). Checkpoint-gated exactly as the AR one is;
# every case emits a loud SKIP when the 28.5 GB asset is absent, so CI still
# compiles, links and runs it.
vllm_cpp_add_test(test_minimax_music3_acoustic_real
  parity/test_minimax_music3_acoustic_real.cpp)
target_compile_definitions(test_minimax_music3_acoustic_real PRIVATE
  MUSIC3_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens/minimax_music3_oracle")
target_include_directories(test_minimax_music3_acoustic_real PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity)
# W6 (#672): the SPEECH-FAMILY seam, the request contract and the chunk plan.
# Pure or synthetic-tree only, so it needs neither the checkpoint nor a golden.
# W7 of #672 -- the QUANTIZED ARMS. Nothing here loads a quantized weight:
# no quantized MiniMax-Music3 checkpoint is staged, and none is implemented.
# What it gates is that every quantized format a user can obtain today is
# DIAGNOSED BY NAME and refused, instead of surfacing as the generic
# "missing seven diffusers directories" or as a shape mismatch on an
# unrelated tensor. Needs no checkpoint and no network.
vllm_cpp_add_test(test_minimax_music3_quant vllm/models/test_minimax_music3_quant.cpp)

vllm_cpp_add_test(test_minimax_music3_speech vllm/models/test_minimax_music3_speech.cpp)
# W6's FULL-SCALE companion: the pipeline TAIL driven on the shipped weights,
# against the capture's own waveform, and the WAV the delivery path emits.
# Checkpoint-gated with a loud SKIP, exactly as the W4/W5 real gate is.
# W7 of #672 -- the ONE quantized arm that is implemented, against the REAL
# pinned artifact (audio-cpp/MiniMax-Music3-GGUF @c36aaeed,
# rvq_depth_decoder_q4_k.gguf, 406 MB). Skips loudly without it. Proves the
# QUANTIZED PATH IS TAKEN (resident ggml types + the Q4_K lattice), not merely
# that the numbers land inside a tolerance.
vllm_cpp_add_test(test_minimax_music3_quant_real parity/test_minimax_music3_quant_real.cpp)
target_compile_definitions(test_minimax_music3_quant_real PRIVATE
  MUSIC3_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens/minimax_music3_oracle")

# test_minimax_music3_e2e_real is NOT registered here. It links `ApiServer`,
# which `CMakeLists.txt` compiles only inside `if(VLLM_CPP_SERVER)`, so its
# registration lives inside that same guard below with the other
# server-linking suites (#1883).

# THE ENGINE'S OWN CALL to the DiT arm selector (#1131, row MUSIC3-DIT-ARM-REACH).
# Wave 1 of that row gated the RULE from a CPU runner; what stayed unreached was
# the two-line CALL in `Music3SpeechEngine::Synthesize`, which needs an
# accelerator and the 28.5 GB checkpoint and therefore cannot run on anything CI
# owns. This entry is that gate. It drives `include/vllm.h` -- the production
# entry point -- with `device = 1` and asserts WHICH ARM RAN off the engine's own
# profile buckets, never off the audio, because the two arms agree by design.
#
# LABELLED, because a gate whose preconditions no runner has must be selectable
# and excludable by name rather than by whoever remembers it exists:
#   ctest -L gpu    selects it   (inside an `rc` lease; see docs/USAGE.md)
#   ctest -LE gpu   excludes it
# Without a device or a checkpoint it EXITS 77 and CTest reports **Skipped** --
# `vllm_cpp_add_test` registers SKIP_RETURN_CODE 77 -- so it can never be read as
# a pass. RUN_SERIAL because it stages ~10 GB of fp32 DiT onto the accelerator.
vllm_cpp_add_test(test_minimax_music3_device_arm_real
  parity/test_minimax_music3_device_arm_real.cpp)
# Reaches src/ for `music3_profile.h`, the INTERNAL instrument the engine's own
# `profile::Report` prints. Deliberately not a public header: it is how this gate
# reads which arm ran, and it is not part of the ABI.
target_include_directories(test_minimax_music3_device_arm_real PRIVATE
  ${CMAKE_SOURCE_DIR}/src)
set_tests_properties(test_minimax_music3_device_arm_real PROPERTIES
  RUN_SERIAL ON
  LABELS "gpu;checkpoint;music3")

# THE ENGINE'S OWN CALL to the DEPTH arm selector (#1839, row
# MUSIC3-DEPTH-ARM-REACH). The twin of the entry above, and a separate entry for
# a reason: #1131 named BOTH device-arm twins, the DiT one closed with it, and
# the depth call site at `minimax_music3_speech.cpp:638` stayed unreached --
# deleting it left `test_minimax_music3_ar` 37/37 and `test_minimax_music3_speech`
# 9/9 green. Two call sites on one `--speech-device 1` switch is how one of them
# drifts, so each carries its own gate rather than one gate asserting both.
#
# Same labels, same 77-on-missing-precondition contract, same lease. `ctest -L gpu`
# now selects TWO tests, and that count is pinned by name in
# `scripts/check-test-registration.py` because `ctest -L` returns 0 over an empty
# selection. RUN_SERIAL because it stages the depth decoder and the DiT onto the
# accelerator and loads 28.5 GB to do it.
vllm_cpp_add_test(test_minimax_music3_depth_arm_real
  parity/test_minimax_music3_depth_arm_real.cpp)
# Reaches src/ for `music3_profile.h`, the INTERNAL instrument the engine's own
# `profile::Report` prints. Deliberately not a public header: it is how this gate
# reads which arm ran, and it is not part of the ABI.
target_include_directories(test_minimax_music3_depth_arm_real PRIVATE
  ${CMAKE_SOURCE_DIR}/src)
set_tests_properties(test_minimax_music3_depth_arm_real PROPERTIES
  RUN_SERIAL ON
  LABELS "gpu;checkpoint;music3")
# ARCH-ONE-SURFACE ROW 2 fold gate: the video seam == the replicated pre-fold
# pipeline == the goldens the PRE-fold minimax-h3-gen binary rendered at the
# branch base (frames + WAV byte-identical, mux argv pinned).
vllm_cpp_add_test(test_minimax_h3_video_fold vllm/models/test_minimax_h3_video_fold.cpp)
target_compile_definitions(test_minimax_h3_video_fold PRIVATE
  MINIMAX_H3_VIDEO_FOLD_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/minimax_h3_video_fold")
# LTX-2.5 L1: the GENERALIZED VideoEngine seam + its family registry, held to
# the SAME pre-fold goldens through the abstract path (and to the never-guess
# refusals). Needs the fold fixture builder, which lives beside the H3 test.
vllm_cpp_add_test(test_video_engine vllm/multimodal/test_video_engine.cpp)
target_include_directories(test_video_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_compile_definitions(test_video_engine PRIVATE
  MINIMAX_H3_VIDEO_FOLD_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/minimax_h3_video_fold")
# The render phase INSTRUMENT, on its own, over synthetic timelines (row
# LTX25-PHASE-INSTRUMENT, #1668/#1569/#1571). Its own executable rather than a
# block inside `test_ltx2_video`, because two of its cases need a table of
# thousands of records that no render produces, and because an instrument
# question should not cost a model fixture build. The reachability half — that a
# render through the ABI actually emits what these cases pin — stays in
# `test_ltx2_video`, which is the only place that can prove it.
vllm_cpp_add_test(test_render_phase_log vllm/multimodal/test_render_phase_log.cpp)
# LTX-2.5 L7: the family behind the VideoEngine seam and the driving loop, run
# over a reduced-dimension checkpoint set written in the SHIPPED file format.
vllm_cpp_add_test(test_ltx2_video vllm/multimodal/test_ltx2_video.cpp)
# The reader-anchor claim this suite gates is a claim ABOUT a source file, so it
# reads that file. Recorded `file:line` anchors go stale within the same PR here
# (a merge of origin/main moved all nine of these by 25 lines mid-review), which
# is why the anchors are derived from the source and compared with the recorded
# ones rather than trusted.
#
# The second path is the conditioning header, and it is a DIFFERENT file from the
# one the refusal it checks lives in. The generated-keyframe refusal states its
# claims about this tree as a list of names, and the suite re-derives each one
# from these declarations. Reading the claim out of the file that makes it is the
# tautology issue #911 records, so the two files are deliberately not the same.
#
# The third path is the DFR canvas header (row LTX25-DFR-PIPELINE, #986), read
# for the same reason and by the same rule. The temporal-rounds refusal rules the
# canvas layout OUT as a blocker on the grounds that it is ported here, which is
# a claim about THIS tree; the suite re-derives it from these declarations rather
# than from the message that makes it.
target_compile_definitions(test_ltx2_video PRIVATE
  LTX2_VIDEO_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/vllm/multimodal/ltx2_video.cpp"
  LTX2_CONDITIONING_HEADER_PATH="${CMAKE_SOURCE_DIR}/include/vllm/model_executor/models/ltx2_conditioning.h"
  LTX2_DFR_HEADER_PATH="${CMAKE_SOURCE_DIR}/include/vllm/model_executor/models/ltx2_dfr.h")
# LTX-2.5 image conditioning (row LTX25-IMAGE-COND, issue #644): the CHAIN from
# PPM bytes to conditioned, noised tokens, gated against upstream ltx_core AND
# ltx_pipelines executed at reduced dimensions. ltx2_image_cond_goldens.inc
# lives next to the test source.
vllm_cpp_add_test(test_ltx2_image_cond vllm/multimodal/test_ltx2_image_cond.cpp)
target_include_directories(test_ltx2_image_cond
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/multimodal)
# #659/#660: both diffusion lanes' device resolution held against a PARTIAL
# backend that DECLINES the architecture. Its own executable for the same reason
# test_device_selection is one — it registers a fake platform and backend in the
# global registries (XPU slot, and over the CUDA slot so CurrentPlatform()
# resolves to it on a CUDA build too), and process isolation keeps that out of
# every other suite.
vllm_cpp_add_test(test_diffusion_device_seam vllm/multimodal/test_diffusion_device_seam.cpp)
# LTX25-GUIDED-VIDEO link B (#1426): the x0 model's DEVICE branch, run end to end
# through `LoadVideoEngine` on a SYNTHETIC unified-memory accelerator. Its own
# executable for the same reason the seam suite above is one — it registers a
# fake platform in the XPU and CUDA slots, which is process-global — and so that
# `test_ltx2_video`'s own 105-case baseline stays exactly where it is. The
# fixture header it shares with `test_ltx2_video` resolves beside the source.
vllm_cpp_add_test(test_ltx2_video_device_forward
                  vllm/multimodal/test_ltx2_video_device_forward.cpp)
# LTX-2.5 DiT parity gate (row MODEL-DIFFUSION-LTX25, spec phase L2). The goldens
# live next to the test source, like every other generated .inc.
vllm_cpp_add_test(test_ltx2 vllm/models/test_ltx2.cpp)
# LTX-2.5 L8: the DEVICE-RESIDENT forward, held to the SAME ltx2_goldens.inc the
# L2 CPU gate above is held to. Its own target so test_ltx2's recorded 29/1615
# baseline stays exactly where it is; it runs on the CPU backend everywhere and
# adds the CUDA cases when a CUDA backend is registered.
vllm_cpp_add_test(test_ltx2_device vllm/models/test_ltx2_device.cpp)
# LTX-2.5 L4: the VAE bricks, gated against upstream ltx_core executed at reduced
# dimensions. ltx2_vae_goldens.inc lives next to the test source.
vllm_cpp_add_test(test_ltx2_vae vllm/models/test_ltx2_vae.cpp)
target_include_directories(test_ltx2_vae PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
# LTX25-DECODE-THREADS (issue #1009): the two threading cases reach
# vt::cpu::Threadpool::SwapForTesting and the pool's work-stealing cursor, which
# live under src/ like every other CPU-threading A/B in tests/vt/.
target_include_directories(test_ltx2_vae PRIVATE ${CMAKE_SOURCE_DIR}/src)
# LTX25-TILED-DECODE (issue #644): the tiling algebra, the AUTO layout and the
# STREAMING tiled decode, gated against upstream ltx_core executed at reduced
# dimensions. Its own target so the decode arms — which run the reduced decoder 18
# times per arm — do not lengthen test_ltx2_vae's recorded case count.
# ltx2_tiling_goldens.inc lives next to the test source.
vllm_cpp_add_test(test_ltx2_tiling vllm/models/test_ltx2_tiling.cpp)
target_include_directories(test_ltx2_tiling PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
# LTX-2.5 L3: the text conditioning path, gated against upstream ltx_core executed
# at reduced dimensions. ltx2_text_goldens.inc lives next to the test source.
vllm_cpp_add_test(test_ltx2_text_encoder vllm/models/test_ltx2_text_encoder.cpp)
target_include_directories(test_ltx2_text_encoder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
# LTX-2.5 L5: the pipeline layer — sigma schedule, noiser, diffusion steps,
# guidance, patchifiers, the recipe table, the latent spatial upsampler, the
# duration head and the embeddings connector. ltx2_pipeline_goldens.inc lives
# next to the test source.
# LTX-2.5 DFR canvas layout (ROW LTX25-DFR-PIPELINE, issue #986). Gated against
# EXECUTED upstream `dfr_layout` and the three pure-index `dfr_pipeline` helpers;
# ltx2_dfr_goldens.inc lives next to the test source. Every value is an INDEX, so
# the suite compares exact vectors rather than bounding a magnitude — there is no
# NaN, no shape disagreement and no scale for a looser gate to notice.
vllm_cpp_add_test(test_ltx2_dfr vllm/models/test_ltx2_dfr.cpp)
target_include_directories(test_ltx2_dfr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)

vllm_cpp_add_test(test_ltx2_pipeline vllm/models/test_ltx2_pipeline.cpp)
target_include_directories(test_ltx2_pipeline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
# The out-of-scope ledger separates a REACHABLE refusal from a declared marker, and
# docs/FEATURES.md is where that distinction reaches a user. It had re-merged the two
# into one "Declared, not requestable" row, so the public surface contradicted the
# ledger it projects; the case below reads the doc and holds it to the split.
# The ledger's RETIREMENT note carries an absence claim about upstream, and it
# shipped false twice in the same header. The case that guards it reads the header
# itself rather than a copy of the sentence kept in the test.
# The ledger's reachable/marker split is a claim about the PRODUCT TREE — that
# something calls the function holding a reachable refusal, and that nothing calls
# the one holding a marker. It was wrong about `kBetaScheduler` for the whole of
# this row because it was maintained by hand, so the case now walks src/, include/
# and examples/ under this root and derives it, with two positive controls in the
# same walk (an absence claim about our own tree is still an absence claim).
target_compile_definitions(test_ltx2_pipeline PRIVATE
  VLLM_CPP_FEATURES_DOC_PATH="${CMAKE_SOURCE_DIR}/docs/FEATURES.md"
  LTX2_PIPELINE_HEADER_PATH="${CMAKE_SOURCE_DIR}/include/vllm/model_executor/models/ltx2_pipeline.h"
  VLLM_CPP_SOURCE_ROOT="${CMAKE_SOURCE_DIR}")
# LTX-2.5 L6: the quantized loaders, gated three ways — the SHIPPED checkpoints'
# own manifests, a few hundred of their own bytes against torch, and synthetic
# files for whole-model materialization. All three .inc files live beside the
# test source.
# LTX25-RETAKE (#924): the ported pieces at value level — the temporal region
# mask under both coordinate conventions, the truncate-or-pad conform and the
# CLI-stage validations. Its own target rather than cases inside
# test_ltx2_pipeline, which several concurrent rows of this campaign edit. The
# REACHABILITY proof for the same row lives in test_ltx2_video.
vllm_cpp_add_test(test_ltx2_iclora_reference vllm/models/test_ltx2_iclora_reference.cpp)
vllm_cpp_add_test(test_ltx2_retake vllm/models/test_ltx2_retake.cpp)
vllm_cpp_add_test(test_ltx2_lora vllm/models/test_ltx2_lora.cpp)
vllm_cpp_add_test(test_ltx2_loader vllm/models/test_ltx2_loader.cpp)
target_include_directories(test_ltx2_loader PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
vllm_cpp_add_test(test_video_api vllm/entrypoints/openai/test_video_api.cpp)
vllm_cpp_add_test(test_speech_api vllm/entrypoints/openai/test_speech_api.cpp)
target_include_directories(test_model_registry PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_platform vllm/platforms/test_platform.cpp)
vllm_cpp_add_test(test_cuda_arch_manifest
  vllm/platforms/test_cuda_arch_manifest.cpp)
vllm_cpp_add_test(test_device_selection
  vllm/entrypoints/test_device_selection.cpp)
# ENG-EXPERT-STREAM #1123: the load-time GGUF device-fit refusal, in two targets
# for two different questions. The arithmetic runs everywhere; the REACHABILITY
# gate needs a fake WEIGHT-STAGING platform in the CUDA lookup slot, because
# `needs_weight_staging()` is true on exactly one platform in this tree and the
# branch is otherwise unreachable on a host with no CUDA device. Its own
# executable for the same reason test_device_selection is one.
vllm_cpp_add_test(test_gguf_device_fit
  vllm/model_executor/test_gguf_device_fit.cpp)
vllm_cpp_add_test(test_gguf_device_fit_reach
  vllm/entrypoints/test_gguf_device_fit_reach.cpp)
# LOAD-GGUF-MMPROJ (#821): the `clip` mmproj reader, and — separately — the
# proof that the production loader reaches it. The second target enters through
# `LoadedEngine::FromModelDir` with `--mmproj` set, so deleting the call site in
# `model_loader.cpp` reds it; the first would stay green, which is exactly the
# difference between gating a class and gating a capability.
vllm_cpp_add_test(test_clip_mmproj_gguf
  vllm/models/test_clip_mmproj_gguf.cpp)
vllm_cpp_add_test(test_gguf_mmproj_reach
  vllm/entrypoints/test_gguf_mmproj_reach.cpp)
# QUANT-QWEN38-27B-GGUF-ARM (#821 W2): the committed 866- and 334-name
# manifests of `unsloth/Qwen3.8-27B-GGUF` @ fe1e2a23d9, and the tensor
# accounting against them. Split the same way for the same reason — the first
# target calls the two enumerations, the second enters through
# `LoadedEngine::FromModelDir`, so deleting either refusal call site in
# `model_loader.cpp` reds the second and leaves the first green.
vllm_cpp_add_test(test_qwen38_27b_gguf_manifest
  vllm/models/test_qwen38_27b_gguf_manifest.cpp)
vllm_cpp_add_test(test_gguf_accounting_reach
  vllm/entrypoints/test_gguf_accounting_reach.cpp)
# FIX-FP8-BLOCKWISE-REFUSAL #1166: block-wise (128x128) FP8 is refused BY NAME
# at load. Every case enters through `ModelRegistry::Load` rather than through
# the predicate, so deleting that call site reds this target. That is the
# reachability proof and not a class test.
vllm_cpp_add_test(test_fp8_block_quant
  vllm/model_executor/layers/test_fp8_block_quant.cpp)
# MODEL-FP8-BLOCK-WEIGHT #1189 M3: `Fp8BlockWeight`, the `weight_scale_inv`
# loader rung, and the config reader. Every case enters through
# `ModelRegistry::Load` or `LoadQwen3_5Dense` -- the loader that call reaches --
# and never through `LoadFp8BlockRaw`, so deleting the rung reds this target.
vllm_cpp_add_test(test_fp8_block_weight_load
  vllm/model_executor/models/test_fp8_block_weight_load.cpp)

# MODEL-FP8-BLOCK-LINEAR (#1189 M4): the block-wise FP8 linear method and the
# dense forward that reaches it.
vllm_cpp_add_test(test_fp8_block_linear
  vllm/model_executor/models/test_fp8_block_linear.cpp)

# MODEL-FP8-BLOCK-MERGED (#1189 M6): the merged gate_up and QKV projections of
# the block-wise FP8 arm. Enters at `ModelRegistry::Forward`, so deleting either
# production merge call site reds this target.
vllm_cpp_add_test(test_fp8_block_merged
  vllm/model_executor/models/test_fp8_block_merged.cpp)

# GATE-FP8-NUMERIC-BOUND -- the numerical lower bound on per-projection outputs
# and the per-block scale-variance probe, #1189's gate design layers 2 and 3.
vllm_cpp_add_test(test_fp8_block_numeric_bound
  vllm/model_executor/models/test_fp8_block_numeric_bound.cpp)
vllm_cpp_add_test(test_linear_method vllm/model_executor/layers/test_linear_method.cpp)
target_include_directories(test_linear_method PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_qwen3_break_point vllm/models/test_qwen3_break_point.cpp)
vllm_cpp_add_test(test_qwen3_decode_graph_seam vllm/models/test_qwen3_decode_graph_seam.cpp)
vllm_cpp_add_test(test_qwen3_moe_decode_graph_seam vllm/models/test_qwen3_moe_decode_graph_seam.cpp)
vllm_cpp_add_test(test_moe_async_device_ids vllm/models/test_moe_async_device_ids.cpp)
vllm_cpp_add_test(test_rocm_moe_bf16 vllm/models/test_rocm_moe_bf16.cpp)
vllm_cpp_add_test(test_voxtral_decode_graph_seam vllm/models/test_voxtral_decode_graph_seam.cpp)
vllm_cpp_add_test(test_deepseek_v2_decode_graph_seam vllm/models/test_deepseek_v2_decode_graph_seam.cpp)
vllm_cpp_add_test(test_qwen3_5_decode_graph_seam vllm/models/test_qwen3_5_decode_graph_seam.cpp)
# W6 (#1374): the dispatch counters live under src/, beside the runner (:561).
target_include_directories(test_qwen3_5_decode_graph_seam PRIVATE ${CMAKE_SOURCE_DIR}/src)
# SPEC-DFLASH2 W10 repair (#1865): the model-side FA-2 dtype/lane class — the
# spec-as-decode verify selects bf16 through the SPEC lane's own toggles. The
# seam lives in qwen3_5_internal.h under src/.
vllm_cpp_add_test(test_qwen3_5_fa2_class vllm/models/test_qwen3_5_fa2_class.cpp)
target_include_directories(test_qwen3_5_fa2_class PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_qwen3_dflash_decode_graph_seam vllm/models/test_qwen3_dflash_decode_graph_seam.cpp)
# G1 for the row: bit-exactness against eager over THREE replays, on a real
# device. Inert without a CUDA backend (every case skips), because a CPU
# "replay" recomputes nothing.
vllm_cpp_add_test(test_decode_graph_seam_g1_cuda vllm/models/test_decode_graph_seam_g1_cuda.cpp)
vllm_cpp_add_test(test_decode_graph_sizes vllm/models/test_decode_graph_sizes.cpp)
vllm_cpp_add_test(test_qwen35_paged_forward vllm/models/test_qwen35_paged_forward.cpp)
# GDN-MOE-BF16-OUT (#1168): the MoE forward's GDN output-activation dtype probe
# lives in the not-installed qwen3_5_internal.h, beside the other detail:: seams.
target_include_directories(test_qwen35_paged_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_test(NAME test_qwen35_zero_rope_unfused COMMAND test_qwen35_paged_forward
  "--test-case=qwen35 zero rotary*")
set_tests_properties(test_qwen35_zero_rope_unfused PROPERTIES
  ENVIRONMENT "VT_FUSE_ATTN_PREAMBLE=0")
# GDN-MOE-BF16-OUT (#1168), fresh-review repair. `GdnOutDType()` caches its
# getenv in a function-local static, so one process observes exactly ONE value of
# VT_GDN_OUT_BF16 and a single run cannot tell a resolver that READS the lever
# from one that was hardwired to its default answer — the mutation
# `return DType::kBF16;` left this binary 5/5 13/13. Re-run the SAME binary with
# the lever forced to 0, the shape `_glue_fuse_off` and
# `test_dense_gateup_fused_marlin_off_*` above already use. This is also the
# documented f32 rollback and the denominator of the row's same-binary A/B, so
# without this entry the arm an operator is told to run is ungated.
add_test(NAME test_qwen35_paged_forward_gdn_out_f32
  COMMAND test_qwen35_paged_forward)
set_tests_properties(test_qwen35_paged_forward_gdn_out_f32 PROPERTIES
  ENVIRONMENT "VT_GDN_OUT_BF16=0")
# GDN-MOE-PACKED-BA (#1169): the MoE safetensors loader builds the ONE merged
# `in_proj_ba` owner (rows [b; a], nk) through `LoadQwen3_5MoeLayer`, the
# production per-layer entry, over an in-memory resolver. Hermetic, CPU, <1s.
vllm_cpp_add_test(test_qwen35_moe_gdn_ba_owner vllm/models/test_qwen35_moe_gdn_ba_owner.cpp)
# MoE vision tower (#891): the tower is loaded (and its ABSENCE refused by name)
# and the forked VL forward actually CONSUMES the merger rows.
vllm_cpp_add_test(test_qwen3_5_moe_vision vllm/models/test_qwen3_5_moe_vision.cpp)
# Dense-arm vision tower (MODEL-QWEN35-DENSE-VL-EXL3) and the checkpoint-read
# tower geometry (ISSUE-LOCAL-01M3RT4GVEY4QBE5AYBT8RDM89). This file landed in
# 3edb37da9 without a target, so none of its cases ever ran.
vllm_cpp_add_test(test_qwen3_5_dense_vision vllm/models/test_qwen3_5_dense_vision.cpp)
# SPEC-MTP I5a: GDN layer spec routing (GdnBlockPaged spec branch) — needs the
# internal header under src/ for the GdnBlockPagedForTest entry point.
vllm_cpp_add_test(test_qwen3_5_gdn_spec_routing vllm/models/test_qwen3_5_gdn_spec_routing.cpp)
vllm_cpp_add_test(test_qwen3_5_gguf_mtp vllm/models/test_qwen3_5_gguf_mtp.cpp)
vllm_cpp_add_test(test_qwen3_dflash_gguf vllm/models/test_qwen3_dflash_gguf.cpp)
target_include_directories(test_qwen3_5_gdn_spec_routing PRIVATE ${CMAKE_SOURCE_DIR}/src)

# MODEL-MM-QWEN4-EXP (ISSUE-LOCAL-01M2ENTH6YA5FWEDY6CFHF4NAM): the deferred GDN
# V-head permutation. Same reason for the src/ include — it drives the block
# through `GdnBlockPagedForTest`.
vllm_cpp_add_test(test_gdn_v_head_permute vllm/models/test_gdn_v_head_permute.cpp)
target_include_directories(test_gdn_v_head_permute PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_qwen27_dense_forward vllm/models/test_qwen27_dense_forward.cpp)
# MODEL-GDN-OUTPUT-GATE-TYPE (#489): `GlueFuseEnabled()` defaults ON and is read
# ONCE per process into a function-local static, so one run of these two suites
# can only ever execute the glue-FUSED gated-RMSNorm tail. The UNFUSED tail
# carries its own `vt::RmsNormGatedArgs{eps, sigmoid_gate}` construction in each
# of the three GDN blocks, and inverting only that construction left the entire
# declared focused gate GREEN (test_hf_config 210|210, test_qwen27_dense_forward
# 583|583, test_qwen3_5_gdn_spec_routing 52|52) — live code, but ungated.
# Re-run the SAME binaries with the lever forced to 0 so the `VT_GLUE_FUSE=0`
# fallback is gated too, mirroring the precedent the fusion row itself set
# (parity-ledger.md:112 — "fused ON default AND `VT_GLUE_FUSE=0` fallback 16/16")
# and the per-lever re-run below.
foreach(_glue_fuse_test test_qwen27_dense_forward test_qwen3_5_gdn_spec_routing)
  add_test(NAME ${_glue_fuse_test}_glue_fuse_off COMMAND ${_glue_fuse_test})
  set_tests_properties(${_glue_fuse_test}_glue_fuse_off PROPERTIES
    ENVIRONMENT "VT_GLUE_FUSE=0" SKIP_RETURN_CODE 77)
endforeach()
# MODEL-GDN-OUTPUT-GATE-TYPE (#489), the FP8 half. Each GDN tail carries TWO fp8
# gate-carrying constructions: the `vt::kRmsNormGatedQuantFp8` FusedRecipe copy
# (taken at the `VT_FUSED_CHAIN_ADOPT` default) and the direct
# `vt::RmsNormGatedQuantFp8` hand-call (taken only at =0). `FusedChainAdoptEnabled()`
# is another read-once static, so a single run reaches exactly one of the two and
# the other is live-but-ungated — the same defect the `_glue_fuse_off` entries
# above were added for. Re-run the SAME binary with the lever forced to 0.
# CUDA-only in effect: the fp8 tail needs `Platform::supports_fp8()`.
add_test(NAME test_qwen3_5_gdn_spec_routing_fused_chain_off
  COMMAND test_qwen3_5_gdn_spec_routing)
set_tests_properties(test_qwen3_5_gdn_spec_routing_fused_chain_off PROPERTIES
  ENVIRONMENT "VT_FUSED_CHAIN_ADOPT=0" SKIP_RETURN_CODE 77)
# PERF-27B-DENSE-MARLIN-GATEUP (#365): the pre-kernel DECISIONS that route the
# 27B's dense W4A16 MLP through the fused Marlin gate_up pair. CPU-only by
# construction — the fused GEMM itself is CUDA + VT_MARLIN_NVFP4.
vllm_cpp_add_test(test_dense_gateup_fused_marlin
  vllm/models/test_dense_gateup_fused_marlin.cpp)
# Each of the three levers the composed guard ANDs is read ONCE per process into
# a function-local static, so a single run can only ever observe one value of
# each and a deleted / hardwired / misspelled term stays green (fresh-review
# findings F1/F2). Re-run the SAME binary with each lever forced to 0: the
# "EVERY lever term is load-bearing" case then asserts the guard reads OFF, which
# is the assertion a dropped term fails.
foreach(_gateup_lever VT_DENSE_MARLIN_GATEUP VT_NVFP4_MARLIN VT_MOE_FUSED_W13)
  string(TOLOWER "${_gateup_lever}" _gateup_lever_lc)
  add_test(NAME test_dense_gateup_fused_marlin_off_${_gateup_lever_lc}
    COMMAND test_dense_gateup_fused_marlin)
  set_tests_properties(test_dense_gateup_fused_marlin_off_${_gateup_lever_lc}
    PROPERTIES ENVIRONMENT "${_gateup_lever}=0")
endforeach()
vllm_cpp_add_test(test_qwen35_plain_weights vllm/models/test_qwen35_plain_weights.cpp)
# UNALIGNED SAFETENSORS PAYLOADS (#772, fourth recurrence after #301/#627/#674).
# The four loaders that formed a `const uint16_t*` over an mmap'd tensor, driven
# over checkpoints whose payload parity is FORCED odd by one space of JSON header
# padding. Two of the four launder every access through `memcpy`, so UBSan cannot
# see them and only these value checks can — which is why this is a suite of its
# own and not another sanitizer sweep.
vllm_cpp_add_test(test_loader_unaligned_offsets
  vllm/models/test_loader_unaligned_offsets.cpp)
# The two Voxtral loader steps are declared in an INTERNAL header (src/, not
# include/): they are not public ABI and must not become so for a test's sake.
target_include_directories(test_loader_unaligned_offsets PRIVATE
  ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_qwen3_8_text_only vllm/models/test_qwen3_8_text_only.cpp)
# The PUBLISHED `Qwen/Qwen3.8-2.4T-A95B` config.json, committed VERBATIM, so the
# config-resolution case reads the real document (nested `rope_parameters`, 92
# `layer_types`, transformers-4.57.3 `dtype`) instead of a paraphrase of it.
# Its `model.safetensors.index.json` (1609 tensors, 213 shards, 4.89 TB) is
# committed the same way, so the load-plan dry run resolves every name against
# the real document rather than a summary of it (#740).
target_compile_definitions(test_qwen3_8_text_only PRIVATE
  QWEN3_8_CONFIG_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/qwen3_8_2_4t_a95b/config.json"
  QWEN3_8_INDEX_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/qwen3_8_2_4t_a95b/model.safetensors.index.json")
# POOL-DEVICE-KEY (#516, .agents/specs/pool-device-key.md): the shared scratch
# pool's free list must be keyed by DEVICE as well as size class, or a block
# allocated through one backend is handed to a DBuf running on another —
# host-side SIGSEGV in one direction, silent all-NaN output in the other.
# Hardware-free: two fake backends stand in for two devices, so this runs
# everywhere and in ~milliseconds rather than only where a GPU is present.
vllm_cpp_add_test(test_device_pool vllm/models/test_device_pool.cpp)
# ENG-POOL-BEST-FIT (#1922): the concurrent case, run a SECOND time with the
# bypass lane OFF, because otherwise ThreadSanitizer never sees it.
#
# `sanitize-cpu` sets `VT_POOL_BYPASS=1` for the whole suite (ci.yml), and it is
# right to: the production pool deliberately retains scratch blocks, and ASan's
# leak detector cannot tell that cache from a leak. But under bypass `Get` is a
# bare `Alloc` and `Put` a bare `Free` — the pool takes no lock, holds no free
# list, and has no shared state to race on. So the one case in this suite whose
# subject IS the locking would have skipped itself in the only lane that could
# have judged it, and the file would have been nominally sanitizer-covered while
# giving the sanitizer nothing to observe.
#
# This entry is safe in the ASan lane for the reason the blanket bypass is not:
# it runs a single case, over a DIRECTLY-CONSTRUCTED pool that it drains before
# it returns, so nothing is retained at exit.
add_test(NAME test_device_pool_concurrent
  COMMAND test_device_pool --test-case=*concurrent*)
set_tests_properties(test_device_pool_concurrent PROPERTIES
  ENVIRONMENT "VT_POOL_BYPASS=0"
  SKIP_RETURN_CODE 77)
vllm_cpp_add_test(test_moe_resident_lifetime vllm/models/test_moe_resident_lifetime.cpp)
# dots3-note (#699) keeps its weights header beside the model in src/, not in
# include/, so this suite needs src/ on its include path to see the MoE block.
target_include_directories(test_moe_resident_lifetime PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_qwen3_load vllm/models/test_qwen3_load.cpp)
vllm_cpp_add_test(test_qwen3_forward vllm/models/test_qwen3_forward.cpp)
target_include_directories(test_qwen3_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_qwen3_dflash_forward vllm/models/test_qwen3_dflash_forward.cpp)
target_include_directories(test_qwen3_dflash_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)
# SPEC-DFLASH2 W11 (#1890): the draft block attention's route onto the SHARED
# paged seam -- the admission table, and the byte-for-byte equivalence with
# vt::DFlashPagedBlockAttention that makes the route safe to default ON. The
# classifier lives in the not-installed qwen3_dflash_internal.h, beside the
# other detail:: seams, so this suite needs src/ on its include path.
vllm_cpp_add_test(test_qwen3_dflash_block_route vllm/models/test_qwen3_dflash_block_route.cpp)
target_include_directories(test_qwen3_dflash_block_route PRIVATE ${CMAKE_SOURCE_DIR}/src)
# SPEC-DFLASH2 W2 (#1314): the DFlash2 draft -- the published configs' nested
# spellings (spec `## Owed` O3/O4), the conv weights, and the draft forward that
# runs the grouped convolution.
vllm_cpp_add_test(test_qwen3_dflash2_draft vllm/models/test_qwen3_dflash2_draft.cpp)
target_include_directories(test_qwen3_dflash2_draft PRIVATE ${CMAKE_SOURCE_DIR}/src)
# SPEC-DFLASH2 W5 (#1314): the GGUF DFlash2 drafter arm. The container the other
# suite cannot reach -- a GGUF declares no `architectures` and writes the same
# `general.architecture = dflash` a DFlash1 drafter writes -- plus the LOWER
# BOUND its k-quant arms need, because this lane dequantizes to bf16 by design
# and a token gate therefore cannot see a quantized tensor that never decoded.
vllm_cpp_add_test(test_qwen3_dflash2_gguf vllm/models/test_qwen3_dflash2_gguf.cpp)
target_include_directories(test_qwen3_dflash2_gguf PRIVATE ${CMAKE_SOURCE_DIR}/src)
# SPEC-DSPARK W2: the DSpark Markov transition head + draft->target vocab map.
vllm_cpp_add_test(test_qwen3_dspark_markov vllm/models/test_qwen3_dspark_markov.cpp)
target_include_directories(test_qwen3_dspark_markov PRIVATE ${CMAKE_SOURCE_DIR}/src)
# SPEC-DSPARK W3: speculators-format DSpark config translation.
vllm_cpp_add_test(test_qwen3_dspark_config vllm/models/test_qwen3_dspark_config.cpp)
target_include_directories(test_qwen3_dspark_config PRIVATE ${CMAKE_SOURCE_DIR}/src)
# SPEC-DSPARK W3: the DSpark draft weight loader (Markov head + d2t).
vllm_cpp_add_test(test_qwen3_dspark_weights vllm/models/test_qwen3_dspark_weights.cpp)
target_include_directories(test_qwen3_dspark_weights PRIVATE ${CMAKE_SOURCE_DIR}/src)
# SPEC-DSPARK W4: sequential Markov sampling + the anchor block layout.
vllm_cpp_add_test(test_cudagraph_dispatch vllm/v1/worker/gpu/test_cudagraph_dispatch.cpp)
target_include_directories(test_cudagraph_dispatch PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_dspark_sample vllm/v1/spec_decode/test_dspark_sample.cpp)
target_include_directories(test_dspark_sample PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_llama_forward vllm/models/test_llama_forward.cpp)
target_include_directories(test_llama_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)
# FUSION-DENSE-MIGRATE: executed CPU coverage for the four DENSE model TUs whose
# gate-up MLP now routes through layers::UnquantizedMlpGateUpMethod (their only
# other gates are checkpoint-gated + dgx-only).
vllm_cpp_add_test(test_dense_gate_up_seam_forward
  vllm/models/test_dense_gate_up_seam_forward.cpp)
target_include_directories(test_dense_gate_up_seam_forward PRIVATE
  ${CMAKE_SOURCE_DIR}/src)
# ARCH-ONE-SURFACE fold ROW 6: embeddings through the registry/runner path on
# the committed tiny LlamaModel fixture (see scripts/mm/llama_embed_fixture_gen.py).
vllm_cpp_add_test(test_llama_embedding_fold
  vllm/models/test_llama_embedding_fold.cpp)
target_compile_definitions(test_llama_embedding_fold PRIVATE
  LLAMA_EMBED_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/llama_embed_e2e")
vllm_cpp_add_test(test_kev vllm/models/test_kev.cpp)
target_include_directories(test_kev PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
vllm_cpp_add_test(test_clm vllm/models/test_clm.cpp)
target_include_directories(test_clm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
vllm_cpp_add_test(test_gliner25_decide vllm/models/test_gliner25_decide.cpp)
target_include_directories(test_gliner25_decide PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
vllm_cpp_add_test(test_xor vllm/models/test_xor.cpp)
target_include_directories(test_xor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
vllm_cpp_add_test(test_tev1 vllm/models/test_tev1.cpp)
target_include_directories(test_tev1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
# MODEL-NIMBLE: reference prompt/answer goldens (scripts/gen-nimble-goldens.py)
# and NimbleDecide over a synthetic Qwen3.5 dense model.
vllm_cpp_add_test(test_nimble vllm/models/test_nimble.cpp)
target_include_directories(test_nimble PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
# MODEL-TEV1 Phase 6: Tev1 on /v1/systemone and vllm_decide. Goldens from the
# Ollama and togethercomputer/tev1 references (scripts/gen-tev1-goldens.py),
# then Tev1Decide through a real synthetic engine and its C ABI handle.
vllm_cpp_add_test(test_tev1_systemone vllm/models/test_tev1_systemone.cpp)
target_include_directories(test_tev1_systemone PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_mistral_forward vllm/models/test_mistral_forward.cpp)
target_include_directories(test_mistral_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_gemma3_load vllm/models/test_gemma3_load.cpp)
vllm_cpp_add_test(test_gemma3_forward vllm/models/test_gemma3_forward.cpp)
target_include_directories(test_gemma3_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_gemma2_load vllm/models/test_gemma2_load.cpp)
vllm_cpp_add_test(test_gemma2_forward vllm/models/test_gemma2_forward.cpp)
target_include_directories(test_gemma2_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_gemma_load vllm/models/test_gemma_load.cpp)
vllm_cpp_add_test(test_gemma_forward vllm/models/test_gemma_forward.cpp)
target_include_directories(test_gemma_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_gemma4_honesty vllm/models/test_gemma4_honesty.cpp)
# #1205: the Gemma-4 device-expert upload must refuse where `vt::MatmulBTAlphaBeta`
# has no arm, entered through `vllm::RunGemma4Moe`. Its own binary: the layer's
# env knobs are frozen into function-local statics on the first call, so the arm
# it measures has to be decided before main() and cannot share a process with a
# case that wants a different one.
vllm_cpp_add_test(test_gemma4_moe_device_arm_guard
                  vllm/models/test_gemma4_moe_device_arm_guard.cpp)
# Its own binary, not a case in the file above: `RunGemma4Moe` freezes
# `VT_GEMMA4_FP8_NATIVE` in a function-local `static const` on the first call in
# the process, and these two files must disagree about it (#2623).
vllm_cpp_add_test(test_gemma4_moe_fp8_native_arm_guard
                  vllm/models/test_gemma4_moe_fp8_native_arm_guard.cpp)
vllm_cpp_add_test(test_gemma4_indexed_max_t vllm/models/test_gemma4_indexed_max_t.cpp)
target_compile_definitions(test_gemma4_indexed_max_t PRIVATE
  VLLM_CPP_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
vllm_cpp_add_test(test_qwen3_moe_load vllm/models/test_qwen3_moe_load.cpp)
vllm_cpp_add_test(test_qwen3_moe_forward vllm/models/test_qwen3_moe_forward.cpp)
target_include_directories(test_qwen3_moe_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_deepseek_v2_load vllm/models/test_deepseek_v2_load.cpp)
vllm_cpp_add_test(test_deepseek_v2_forward vllm/models/test_deepseek_v2_forward.cpp)
target_include_directories(test_deepseek_v2_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)

# DeepSeek-V4-Flash (DeepseekV4ForCausalLM) W1/W2 scaffolding gate: the arch
# resolves through the registry + the shipped config descends through
# ParseDeepseekV4Params. No checkpoint / GPU — the forward + strict gate are
# named W3-W8 residuals (.agents/specs/deepseek-v4-flash.md).
vllm_cpp_add_test(test_deepseek_v4_scaffold vllm/models/test_deepseek_v4_scaffold.cpp)
target_include_directories(test_deepseek_v4_scaffold PRIVATE ${CMAKE_SOURCE_DIR}/src)

# KV-DSV4-MULTICACHE W8 slice 4 + slice 6 (#2455): the fp8_ds_mla BRIDGE — the
# published byte page is actually written and read, and `--kv-cache-dtype auto`
# stops refusing the layout the model's own factory declared.
vllm_cpp_add_test(test_deepseek_v4_fp8_ds_mla_bridge
  vllm/models/test_deepseek_v4_fp8_ds_mla_bridge.cpp)
target_include_directories(test_deepseek_v4_fp8_ds_mla_bridge
                           PRIVATE ${CMAKE_SOURCE_DIR}/src)
# DeepSeek-V4.1-Flash (DeepseekV41ForCausalLM) W1 scaffolding gate: the arch
# RESOLVES through the registry, the REAL published config.json descends and
# validates through ParseDeepseekV41Params (nested text_config/vision_config ->
# flat, the vision_* lift and the expert_dtype lift out of quantization_config),
# and every unimplemented arm -- load, forward, KV-cache spec and the GGUF
# `deepseek41` container -- refuses BY NAME. No checkpoint / GPU: W1 claims no
# load and no forward (.agents/specs/deepseek-v4-1-flash.md `## Work breakdown`).
vllm_cpp_add_test(test_deepseek_v4_1_scaffold vllm/models/test_deepseek_v4_1_scaffold.cpp)
target_compile_definitions(test_deepseek_v4_1_scaffold PRIVATE
  DEEPSEEK_V4_1_CKPT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/deepseek_v4_1")
target_include_directories(test_deepseek_v4_1_scaffold PRIVATE ${CMAKE_SOURCE_DIR}/src)
# `gguf_builder.h` is the shared in-memory GGUF fixture builder under tests/vllm.
target_include_directories(test_deepseek_v4_1_scaffold PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm)

# DeepSeek-V4.1-Flash W3c (indexer arm) + W3d (`kv_source_layer`-keyed KV
# topology) host-reference gate. Ports the three upstream `indexer_k_store`
# cases and the `fused_q_kv_rmsnorm` case at `e77daef89e`, and gates the
# topology -- which has NO upstream test at all -- against the real published
# config.json. No checkpoint / GPU.
vllm_cpp_add_test(test_deepseek_v4_1_indexer vllm/models/test_deepseek_v4_1_indexer.cpp)
target_compile_definitions(test_deepseek_v4_1_indexer PRIVATE
  DEEPSEEK_V4_1_CKPT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/deepseek_v4_1")
target_include_directories(test_deepseek_v4_1_indexer PRIVATE ${CMAKE_SOURCE_DIR}/src)

# DeepSeek-V4-Pro shape-generality gate (#504): the Pro config descends through
# the SAME ParseDeepseekV4Params with zero new keys, and compress_ratios alone
# reproduces the compressor/indexer layer counts of BOTH real checkpoints
# (Flash 41/21, Pro 61/30). No checkpoint / GPU — Pro's 805 GiB does not fit one
# GB10, so a load gate is memory-infeasible (.agents/specs/deepseek-v4-pro.md).
vllm_cpp_add_test(test_deepseek_v4_pro_variant vllm/models/test_deepseek_v4_pro_variant.cpp)
target_include_directories(test_deepseek_v4_pro_variant PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Laguna-S-2.1 W1/W2 scaffolding gate: LagunaForCausalLM RESOLVES through the
# registry + ParseLagunaParams descends the shipped config (nested dual-rope +
# variable Q-head + ungrouped sigmoid-noaux MoE). No 73 GB checkpoint / GPU — the
# forward + loader materialization + strict dual-oracle gate are named W3/W4
# residuals (.agents/specs/laguna-s21-w1w2-2026-07-30.md).
vllm_cpp_add_test(test_laguna_scaffold vllm/models/test_laguna_scaffold.cpp)

vllm_cpp_add_test(test_laguna_fused_gate_up
  vllm/models/test_laguna_fused_gate_up.cpp)
target_include_directories(test_laguna_scaffold PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_laguna_nvfp4_loader vllm/models/test_laguna_nvfp4_loader.cpp)
target_include_directories(test_laguna_nvfp4_loader PRIVATE ${CMAKE_SOURCE_DIR}/src)
# The `laguna` GGUF loader itself, which had no test at all — which is why the
# W6a gather-residency change (#1989) could break it invisibly. Needs
# tests/vllm on the include path for gguf_builder.h.
vllm_cpp_add_test(test_laguna_gguf_load vllm/models/test_laguna_gguf_load.cpp)
target_include_directories(test_laguna_gguf_load PRIVATE ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_CURRENT_SOURCE_DIR}/vllm)

# The qwen3 GGUF loader (`Qwen3HfConfigFromGguf`, `LoadQwen3FromGguf`). Synthetic
# GGUF via gguf_builder.h; no model file needed. Needs tests/vllm for gguf_builder.h.
vllm_cpp_add_test(test_qwen3_gguf_weights vllm/models/test_qwen3_gguf_weights.cpp)
target_include_directories(test_qwen3_gguf_weights PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm)

# DeepSeek-V4-Flash W3: the NEW attention primitives (DSA Lightning-Indexer
# top-k selection + 512-wide MLA attention sinks + grouped output-LoRA) unit-
# gated vs hand-derived cases + double-precision references. Full-model gate is
# multi-Spark-blocked (156.7 GiB); MHC/MoE are W5/W6 (.agents/specs/deepseek-v4-flash.md).
vllm_cpp_add_test(test_deepseek_v4_dsa vllm/models/test_deepseek_v4_dsa.cpp)

# DeepSeek-V4-Flash W4: the DSA COMPRESSOR forward (softmax-weighted window pool
# + save-time APE add + RMSNorm) + the fp8_ds_mla KV-cache state layout (448 fp8
# UE8M0 + 64 bf16 rope, 576B stride) unit-gated vs hand-derived cases +
# double-precision references. Full-model gate is multi-Spark-blocked (156.7 GiB);
# MHC/MoE/device-kernel are W5/W6/W7 (.agents/specs/deepseek-v4-flash.md).
vllm_cpp_add_test(test_deepseek_v4_compressor vllm/models/test_deepseek_v4_compressor.cpp)
# DeepSeek-V4-Flash W5 — Manifold/Markov Hyper-Connections (MHC): the Sinkhorn-
# normalized hc_mult-stream residual manifold + mHC pre/post mixes + hc_head
# collapse + folded RMSNorms. Ported from the vLLM eager reference
# (mhc/torch.py + triton.py, 4 upstream impls agree) + gated against a
# from-first-principles double-precision derivation. Full-model gate multi-Spark-
# blocked; MoE/device-kernel are W6/W7 (.agents/specs/deepseek-v4-flash.md).
vllm_cpp_add_test(test_deepseek_v4_mhc vllm/models/test_deepseek_v4_mhc.cpp)
# DeepSeek-V4.1-Flash W3a — Engram: the n-gram hash over a compressed token
# vocabulary, the prime bucket layout, the sharded fp8 / ue8m0 table lookup and
# the sigmoid gate that injects the lookup into the hyper-connection manifold.
# Ports tests/kernels/test_engram.py @ vLLM e77daef89e (778 lines), including
# its two scalar oracles and their rtol=0/atol=0 tolerances; adds the ue8m0
# byte-0 case its own fixtures cannot reach, and the prime-layout case upstream
# leaves ungated (.agents/specs/deepseek-v4-1-flash.md, wave W3a).
vllm_cpp_add_test(test_deepseek_v4_1_engram vllm/models/test_deepseek_v4_1_engram.cpp)
# DeepSeek-V4-Flash W6 — the sqrtsoftplus + hash-routed MoE: the V4 router score
# function (sqrt∘softplus), the noaux_tc bias-for-selection top-k, the hash
# `tid2eid` route that bypasses top-k, and the clamped SwiGLU expert activation.
# Ported from the vLLM eager reference (fused_topk_bias_router.py + activation.py,
# cross-checked SGLang) + gated against hand-derived literals and double-precision
# references. Full-model gate multi-Spark-blocked (156.7 GiB); device-kernel +
# forward assembly are W7 (.agents/specs/deepseek-v4-flash.md).
vllm_cpp_add_test(test_deepseek_v4_moe vllm/models/test_deepseek_v4_moe.cpp)
# DeepSeek-V4.1-Flash W3b — the MXFP8 32x32 UE8M0 linear family host reference.
# Ports tests/quantization/test_fp8.py:67-238 @ vLLM e77daef89e (a BIT-EXACT
# oracle for the scale expansion) and derives the quantizer / emulation-linear
# cases from first principles in double precision, because every upstream mxfp8
# linear test is CUDA/ROCm-gated. No e2e gate exists: no published V4.1 artifact
# fits any device this project reaches (.agents/specs/deepseek-v4-1-flash.md).
vllm_cpp_add_test(test_deepseek_v4_1_mxfp8 vllm/models/test_deepseek_v4_1_mxfp8.cpp)
# DeepSeek-V4-Flash W7 — the DeepseekV4Model::Forward ASSEMBLY: composes the four
# landed host primitives (W3 DSA/MLA seams, W4 compressor + fp8_ds_mla KV, W5 MHC
# + Sinkhorn, W6 sqrtsoftplus/hash MoE) into an end-to-end logits producer at a
# tiny synthetic config. STRUCTURAL/composition gate (interleave runs, [T,hc,H]
# stream, hash-by-tid routing, DSA select+compress, finiteness) + RED-first
# miswires — NOT a real-checkpoint token gate (that is W8, multi-Spark, 156.7 GiB).
vllm_cpp_add_test(test_deepseek_v4_forward vllm/models/test_deepseek_v4_forward.cpp)
# DeepSeek-V4-Flash W2b — the `deepseek4` GGUF keep-quant TOWER materialization:
# tiny synthetic GGUF (real blk.N.* names + Q8_0 keep-quant) -> accounting +
# keep-quant residency + load->Forward end-to-end, RED-first (missing/leftover).
vllm_cpp_add_test(test_deepseek_v4_gguf_load vllm/models/test_deepseek_v4_gguf_load.cpp)
target_include_directories(test_deepseek_v4_gguf_load PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm)
# DeepSeek-V4-Flash MTP W1 — the native nextn self-speculative draft head: tiny
# synthetic nextn forward (nvidia/mtp.py:128-258) + RED-first miswires + the
# LOSSLESS gate (DS4 draft + DS4 target verified by the shared RejectionSampler ==
# pure target greedy => MTP-on == MTP-off). See .agents/specs/deepseek-v4-mtp.md.
vllm_cpp_add_test(test_deepseek_v4_mtp vllm/models/test_deepseek_v4_mtp.cpp)
vllm_cpp_add_test(test_deepseek_v4_mtp_inventory vllm/models/test_deepseek_v4_mtp_inventory.cpp)
vllm_cpp_add_test(test_deepseek_v4_dspark_entry vllm/models/test_deepseek_v4_dspark_entry.cpp)
# MODEL-DSV4-EXL3 W1b — the rank-sliced EXL3 loader arm: a hermetic four-rank
# fixture through the PRODUCTION loader entry, coalesced TP4->TP1 and checked
# byte-for-byte against the rank inputs, plus refuse-by-name for every arm the
# wave does not implement. See .agents/specs/model-dsv4-exl3.md.
vllm_cpp_add_test(test_deepseek_v4_exl3_loader vllm/models/test_deepseek_v4_exl3_loader.cpp)
target_include_directories(test_deepseek_v4_exl3_loader PRIVATE ${CMAKE_SOURCE_DIR}/src)

# MODEL-MM-deepseek-v4 W3B (#2411) — `exp_probs_b_vl`, the per-layer MoE routing
# bias DeepSeek-V4-Flash-Vision adds for image tokens. Drives BOTH weight arms
# (the `deepseek4` GGUF and the safetensors loader) over hermetic fixtures that
# carry the tensor on every layer, and pins the text checkpoint's inertness when
# it is absent. See .agents/specs/deepseek-v4-flash-vision.md.
vllm_cpp_add_test(test_deepseek_v4_mm_loader vllm/models/test_deepseek_v4_mm_loader.cpp)
# MODEL-MM-deepseek-v4 (#2411): the pinned OFFICIAL artifact's own metadata,
# committed verbatim. `config.json` is the released file; the two manifests are
# derived from it and from shard 1's safetensors HEADER by
# `scripts/check-deepseek-v4-vision-manifests.py`, which reads two HTTP ranges
# and never a weight payload. They are what ties this tree's derived tensor map
# to the 156.287 GiB checkpoint nobody here can stage, so the loader gate can
# ask a real question about an artifact it cannot open.
target_compile_definitions(test_deepseek_v4_mm_loader PRIVATE
  DEEPSEEK_V4_VISION_CONFIG="${CMAKE_SOURCE_DIR}/tests/parity/goldens/deepseek_v4_vision/config.json"
  DEEPSEEK_V4_VISION_INDEX_MANIFEST="${CMAKE_SOURCE_DIR}/tests/parity/goldens/deepseek_v4_vision/index_manifest.json"
  DEEPSEEK_V4_VISION_HEADER_MANIFEST="${CMAKE_SOURCE_DIR}/tests/parity/goldens/deepseek_v4_vision/shard1_header_manifest.json")
target_include_directories(test_deepseek_v4_mm_loader PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/vllm ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_deepseek_v4_exl3_device_residency
                  vllm/models/test_deepseek_v4_exl3_device_residency.cpp)
target_include_directories(test_deepseek_v4_exl3_device_residency
                           PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_deepseek_v4_exl3_forward vllm/models/test_deepseek_v4_exl3_forward.cpp)
# MODEL-DSV4-EXL3 W2d: the SAME suite under the rollback arm. The fused MoE flag
# is read once per process, so the only way to gate both arms is to run the
# binary twice; without this entry the arm an operator is told to fall back to is
# ungated, and the fused arm's launch-count claim rests on one number with
# nothing to contrast it against.
add_test(NAME test_deepseek_v4_exl3_forward_loop_arm
  COMMAND test_deepseek_v4_exl3_forward)
set_tests_properties(test_deepseek_v4_exl3_forward_loop_arm PROPERTIES
  ENVIRONMENT "VT_DSV4_EXL3_FUSED_MOE=0" SKIP_RETURN_CODE 77)
target_include_directories(test_deepseek_v4_mtp PRIVATE ${CMAKE_SOURCE_DIR}/src)
# DeepSeek-V4-Flash W7-DEVICE — the four NEW V4 op families' CUDA kernels
# (src/vt/cuda/cuda_deepseek_v4.cu, dispatched through the OpProvider seam) each
# unit-gated vs its landed HOST reference at small shape + the ForwardDevice
# composition gate (device forward == host forward, near-tie). The device cases
# SKIP on a CPU-only build; on the DGX GB10 they are the RUNTIME evidence the
# kernels match the references. NOT a real-checkpoint token gate (W8, multi-Spark).
vllm_cpp_add_test(test_cuda_deepseek_v4 vllm/models/test_cuda_deepseek_v4.cpp)
target_include_directories(test_cuda_deepseek_v4 PRIVATE ${CMAKE_SOURCE_DIR}/src)
# O34: the ROCm/HIP arm of the DeepSeek-V4 MHC device-kernel family. A 1:1 mirror
# of the CUDA MHC cases above, run against the ROCm provider registered in
# rocm_deepseek_v4.hip. The device cases SKIP on a build without a ROCm backend
# (exit 77 — "assertions: 0 ... SUCCESS!" is not a pass, tests/CMakeLists.txt:30-38).
vllm_cpp_add_test(test_rocm_deepseek_v4_mhc vt/test_rocm_deepseek_v4_mhc.cpp)
target_include_directories(test_rocm_deepseek_v4_mhc PRIVATE ${CMAKE_SOURCE_DIR}/src)
set_tests_properties(test_rocm_deepseek_v4_mhc PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
# KIMI-K3 W2/W5 SCAFFOLDING (KimiK3ForConditionalGeneration) — the DERIVE-AND-SHIP
# structural gate: registry-resolve + nested-config descent + text-backbone
# enumeration + REFUSE-by-name forward + MXFP4-refuse loader. No checkpoint / GPU;
# K3 is beyond the pinned oracle (.agents/specs/kimi-k3.md).
vllm_cpp_add_test(test_kimi_k3_scaffold vllm/models/test_kimi_k3_scaffold.cpp)
vllm_cpp_add_test(test_muse_glimmer_scaffold vllm/models/test_muse_glimmer_scaffold.cpp)
vllm_cpp_add_test(test_muse_glimmer_vision vllm/models/test_muse_glimmer_vision.cpp)
target_include_directories(test_muse_glimmer_vision PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
# The FIRST reference run of the perception encoder on the RELEASED tensors
# (.agents/specs/muse-glimmer-parity.md): env-gated, the tensors are ~3.7 GB.
vllm_cpp_add_test(test_muse_glimmer_vision_real vllm/models/test_muse_glimmer_vision_real.cpp)
target_include_directories(test_muse_glimmer_vision_real PRIVATE ${CMAKE_SOURCE_DIR}/src)
# Muse Glimmer W1 TEXT TOWER gate: the whole text forward vs an INDEPENDENT fp32
# reference transcribed from vllm#51655 head 075d645af, plus a property test per
# mechanism (weightless embed_norm, QK-norm, query pre-scale, the iRoPE NoPE/sliding
# split, split pre/post eps, the offset-free final norm, the attention output gate,
# the output multiplier and the soft-cap). CPU-only, no checkpoint. No speed axis
# exists for this model — the pinned oracle cannot load it (specs/muse-glimmer.md §0).
# Nemotron Nano VL / Omni image path (row
# MODEL-MM-nano-nemotron-vl-nemotron-h-nano-vl-v2): the dynamic-resolution
# processor, the RADIO tower and the mlp1 projector against a torch
# transcription of vLLM e126687a9a (scripts/mm/nano_nemotron_vl_ref.py), plus
# the released configs. The real-checkpoint arm is env-gated.
vllm_cpp_add_test(test_nano_nemotron_vl_vision vllm/models/test_nano_nemotron_vl_vision.cpp)
target_include_directories(test_nano_nemotron_vl_vision PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_compile_definitions(test_nano_nemotron_vl_vision PRIVATE
  NANO_NEMOTRON_VL_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures")
# The same row's REACHABILITY gate: a tiny Omni checkpoint loads through
# ModelRegistry under the released name and an image decodes through the
# runner (encode_mm -> embed_mm -> the paged NemotronH forward), token-equal to
# an independent host reference and different from the text-only decode.
vllm_cpp_add_test(test_nano_nemotron_vl_registry vllm/models/test_nano_nemotron_vl_registry.cpp)
target_include_directories(test_nano_nemotron_vl_registry PRIVATE
  ${CMAKE_SOURCE_DIR}/src)
# Cohere2MoeForCausalLM (North) gates (.agents/specs/cohere2-moe.md): the released
# North-Mini-Code-1.0 config.json (committed verbatim) and its refusals; three tiny
# checkpoints read back through the production loader against the torch
# transcription of the pinned cohere2_moe.py (scripts/cohere2-moe-ref.py), f32 arm
# tight and bf16 arm on the envelope; an env-gated real-tensor per-layer arm
# (VLLM_COHERE2_MOE_REAL_DIR) and an env-gated structural arm over the released
# index (VLLM_COHERE2_MOE_INDEX). CPU-only, no checkpoint needed in CI.
vllm_cpp_add_test(test_cohere2_moe vllm/models/test_cohere2_moe.cpp)
target_include_directories(test_cohere2_moe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_compile_definitions(test_cohere2_moe PRIVATE
  COHERE2_MOE_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/north_mini_code_1_0")
# ...and its reachability gate: ModelRegistry::Load under "Cohere2MoeForCausalLM"
# and a greedy decode through GPUModelRunner, token-equal to the transcription.
vllm_cpp_add_test(test_cohere2_moe_registry vllm/models/test_cohere2_moe_registry.cpp)
target_include_directories(test_cohere2_moe_registry PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
vllm_cpp_add_test(test_muse_glimmer_text vllm/models/test_muse_glimmer_text.cpp)
# The DFlash drafter's RELEASED config.json, committed verbatim. It is the only
# real artifact that OMITS qk_scale_factor / post_norm_eps / output_multiplier /
# final_logit_softcapping, so it is what pins their defaults to the architecture's
# constants rather than to neutral values (issue #412). Committed, not read off
# the NAS, so the gate is asset-free.
target_compile_definitions(test_muse_glimmer_text PRIVATE
  MUSE_GLIMMER_DRAFT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/muse_glimmer_30b_assistant")
# ...and the SAME gate on the HAND-CALL arm. `FusedChainAdoptEnabled()` defaults ON
# and is read once per process into a function-local static, so the three
# `else vt::RmsNorm(...)` fallbacks in muse_glimmer.cpp (input layernorm,
# pre-feedforward layernorm, final norm) are UNREACHABLE in a default-env run —
# mutating one of them left every gate green — and a same-process env flip cannot
# reach them either. A SECOND ctest registration of the same binary is the only way
# to gate that arm, and it is the arm that runs wherever the fusion recipes are
# unavailable, so both must be green. The `... the fusion arm the environment
# selected` case inside the binary is what proves this registration really took the
# other branch instead of silently repeating the first.
add_test(NAME test_muse_glimmer_text_fallback COMMAND test_muse_glimmer_text)
set_tests_properties(test_muse_glimmer_text_fallback PROPERTIES
  ENVIRONMENT "VT_FUSED_CHAIN_ADOPT=0")

# Muse Glimmer W4 WIRING gate: the REAL-CHECKPOINT structural contract (every name
# EnumerateMuseGlimmerTensors produces exists in the released 1436-tensor
# Muse-Glimmer-30B index at the right shape, and nothing in the checkpoint is
# unaccounted) plus the perception-encoder wiring (vision load with the q|k|v merge
# order and every bias, the soft-token projection, the masked scatter onto the
# image/video rows, and the mm forward branch). The structural half runs off a
# COMMITTED header-only fixture so CI never needs the NAS; a second case re-reads
# the live checkpoint when VLLM_MUSE_CKPT points at it. CPU-only, no GPU, no tensor
# bytes. NOT an image e2e correctness gate — the pinned oracle cannot load this
# model at all, so no speed or token-exactness claim exists (specs/muse-glimmer.md §0).
vllm_cpp_add_test(test_muse_glimmer_wiring vllm/models/test_muse_glimmer_wiring.cpp)

# ENG-MM-INPUT-PIPELINE L3 (#607): the TOWER SKIP. Shares the tiny synthetic
# Muse Glimmer checkpoint with the wiring gate above (muse_glimmer_tiny_fixture.h)
# so both ask their question of the same bytes.
vllm_cpp_add_test(test_tower_skip vllm/models/test_tower_skip.cpp)
# ENG-MM-INPUT-PIPELINE (#1359): the tower's host STORAGE DTYPE, counted in
# bytes through both production loaders. Shares the same tiny Muse Glimmer
# checkpoint for the same reason the skip gate does — the question is what the
# loader did with THOSE bytes. A token gate cannot see a dtype that is too wide,
# so this is the only shape of gate that can (specs/vision-tower-dtype-polarity.md).
vllm_cpp_add_test(test_vision_tower_dtype vllm/models/test_vision_tower_dtype.cpp)
target_include_directories(test_vision_tower_dtype PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_compile_definitions(test_muse_glimmer_wiring PRIVATE
  MUSE_GLIMMER_CKPT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/muse_glimmer_30b")
# Muse Glimmer REAL-WEIGHT gate. Loads the released 30B bf16 checkpoint (or a
# reduced-depth model assembled from its REAL tensors) through the real loader,
# forwards it on CPU, and compares against a standalone torch transcription of
# vllm#51655 head 075d645af produced by scripts/mm/muse_glimmer_text_ref.py.
# Every case SKIPs unless its env var names a fixture directory, so CI never
# depends on the 59.55 GB NAS asset. This is NOT a token-exact gate against the
# model's own runtime — no such runtime exists on this box (released
# transformers does not register `muse_glimmer` and the pin has no Muse code) —
# and it makes no speed claim whatsoever (specs/muse-glimmer.md §0).
vllm_cpp_add_test(test_muse_glimmer_real_weights
  vllm/models/test_muse_glimmer_real_weights.cpp)
target_compile_definitions(test_muse_glimmer_real_weights PRIVATE
  VLLM_MUSE_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models")

# Muse Glimmer GGUF K-QUANT gate (.agents/porting-a-model.md §2). The quantized
# arm the port was missing: config-from-GGUF, the canonical<->GGUF name map, the
# three convert-time transforms this loader inverts (the +1 sandwich-norm
# pre-offset, the query pre-scale folded into attn_q_norm, the iRoPE mask on
# sliding_window_pattern), and STRUCTURAL ACCOUNTING in both directions against
# COMMITTED header-only manifests of the real 731/809/58-tensor files, so CI
# needs no asset. A live case runs when VLLM_MUSE_GGUF names the 16.76 GB
# k-quant. No forward, no e2e, and NO SPEED AXIS: the pinned oracle cannot load
# `muse_glimmer` in any weight format (specs/muse-glimmer.md §0).
vllm_cpp_add_test(test_muse_glimmer_gguf vllm/models/test_muse_glimmer_gguf.cpp)
target_include_directories(test_muse_glimmer_gguf PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)

# MODEL-MM-QWEN4-EXP W6a: the LOAD PLAN for Qwen3.8-Flash-Next's only runnable
# artifact (`unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ1_S`, 67.56 GiB in 3 shards).
# Every ggml type in the real 1224-tensor table resolves, the table is accounted
# in both directions against a COMMITTED header-only manifest, the `qwen4exp`
# config builder reads the shipped metadata and refuses by its own name, and the
# residency of the 51.2 G-parameter n-gram table is ASSERTED with the arithmetic
# that makes it load-bearing. No forward, no token claim, no speed axis: the
# model class this config names is not registered yet (#1987/#1988 own it, and
# .agents/specs/qwen4-exp-flash-next.md `## Owed` tracks the wiring).
vllm_cpp_add_test(test_qwen4_exp_gguf_load_plan
  vllm/models/test_qwen4_exp_gguf_load_plan.cpp)
target_include_directories(test_qwen4_exp_gguf_load_plan PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)

# MODEL-MM-QWEN4-EXP W5a (#2031): the `qwen4exp` GGUF WEIGHT LOADER, entered
# through the production `load_weights` hook. Two instruments: the committed
# 1224-tensor manifest of the shipped artifact gates the name map and the shape
# rules with no asset in CI, and a synthetic tiny GGUF gates the VALUES -- the
# `+1` norm fold and its one `ssm_norm` exception, the V-head reorder,
# `log(-ssm_a)`, and keep-quant residency including the n-gram gather table.
# No forward and no KV spec: both still refuse by name, and
# .agents/specs/qwen4-exp-flash-next.md `## Owed` tracks them.
vllm_cpp_add_test(test_qwen4_exp_gguf_weights
  vllm/models/test_qwen4_exp_gguf_weights.cpp)
target_include_directories(test_qwen4_exp_gguf_weights PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
# qwen4_exp*.h are MODEL-PRIVATE headers under src/, not include/vllm/.
target_include_directories(test_qwen4_exp_gguf_weights PRIVATE ${CMAKE_SOURCE_DIR}/src)
# The released Qwen3.8-Flash-Next config.json, committed by W1: the name-map and
# shape cases resolve their expectations through it so both sides of every
# comparison come from the model's author.
target_compile_definitions(test_qwen4_exp_gguf_weights PRIVATE
  QWEN4_EXP_CKPT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/qwen4_exp")

# Kimi-Linear-48B-A3B (KimiLinearForCausalLM) W1 scaffolding gate: registry-resolve
# + config parse (20 KDA + 7 NoPE-MLA schedule, MLA/MoE/KDA dims, reject cases) +
# checkpoint name-map faithfulness (block_sparse_moe, verified vs the HF index) +
# loader round-trip on a synthetic 2-layer checkpoint (throws BY NAME on a missing
# or mis-shaped tensor). CPU-only; the forward REFUSES-by-name (W3-W6).
# .agents/specs/kimi-linear.md.
vllm_cpp_add_test(test_kimi_linear_scaffold vllm/models/test_kimi_linear_scaffold.cpp)

# dots3-note W1+W2 STRUCTURAL gate (#699, .agents/specs/dots3-note.md §4 / §7):
# registry-resolve + config descent off the REAL released config.json, with one
# RED-first assertion per §4 CONFIG TRAP (n_group/topk_group=1,
# indexer_rope_interleave=True, num_nextn_predict_layers=1, the two lora
# rescales, the sliding rope's own theta and the GPT-J layout both ropes share),
# the on-disk name map accounted BOTH WAYS over the WHOLE released shard index
# (W2: all 38006 tensors -- 35381 language, 2195 vision, 430 audio -- from a
# committed headers-only projection of the index joined to all 133 safetensors
# headers, plus W1's 1614-tensor slice kept as a cross-check), the two tower
# files as NAMED W6/W7 deferrals asserted by count, the padded 1088-wide MLA
# row, and the load/GGUF/forward refusals
# — the last driven through the REAL model the factory returns, never a
# fabricated LoadedModel (#730/#784). Spec §6.4 records that NO oracle for this
# model runs on hardware this project owns, so these unit assertions are the
# only instrument the row has. CPU-only, no checkpoint, no speed claim.
vllm_cpp_add_test(test_dots3_note_scaffold vllm/models/test_dots3_note_scaffold.cpp)
target_compile_definitions(test_dots3_note_scaffold PRIVATE
  DOTS3_NOTE_CKPT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/dots3_note_prev")
# dots3_note.h is a MODEL-PRIVATE header under src/, not include/vllm/: W1 ships
# nothing on the public ABI and include/vllm/ is a USER_USAGE_PREFIXES surface
# (#515). Same arrangement test_nemotron_h_scaffold uses.
target_include_directories(test_dots3_note_scaffold PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Qwen4-Exp W1 -- config resolution, validation, registration and refuse-by-name
# (#1981, .agents/specs/qwen4-exp-flash-next.md). Fixture is the published
# `Qwen/Qwen3.8-Flash-Next` config.json verbatim.
vllm_cpp_add_test(test_qwen4_exp_scaffold vllm/models/test_qwen4_exp_scaffold.cpp)
target_compile_definitions(test_qwen4_exp_scaffold PRIVATE
  QWEN4_EXP_CKPT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/qwen4_exp")
# qwen4_exp.h is a MODEL-PRIVATE header under src/, not include/vllm/: W1 ships
# nothing on the public ABI. Same arrangement test_dots3_note_scaffold uses.
target_include_directories(test_qwen4_exp_scaffold PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Qwen4-Exp W5c (#2031) -- the KV-cache spec, driven through the PRODUCTION
# `make_kv_cache` registry hook rather than by constructing spec structs. Three
# groups: the QSA layers' paged K+V, ONE uniform recurrent group carrying four
# states on every linear-attention layer, and the QSA indexer side cache as an
# `MLAAttentionSpec` at compress_ratio 4. Fixture is the published
# `Qwen/Qwen3.8-Flash-Next` config.json verbatim.
# MODEL-MM-QWEN4-EXP W5f (#2031, #2336): the LAYER LOOP against the lane-pinned
# transformers 5.16.0 oracle, end to end. `qwen4_exp_forward.h` and the weight
# structs are MODEL-PRIVATE headers under src/, hence the src/ grant, and the
# golden is regenerated by scripts/gen-qwen4-exp-forward-goldens.py.
vllm_cpp_add_test(test_qwen4_exp_layer_loop vllm/models/test_qwen4_exp_layer_loop.cpp)
target_include_directories(test_qwen4_exp_layer_loop PRIVATE ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_qwen4_exp_runner vllm/models/test_qwen4_exp_runner.cpp)
target_include_directories(test_qwen4_exp_runner PRIVATE ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_qwen4_exp_matmul_bt_dtype
  vllm/models/test_qwen4_exp_matmul_bt_dtype.cpp)
target_include_directories(test_qwen4_exp_matmul_bt_dtype PRIVATE ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_qwen4_exp_kv_cache vllm/models/test_qwen4_exp_kv_cache.cpp)
target_compile_definitions(test_qwen4_exp_kv_cache PRIVATE
  QWEN4_EXP_CKPT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/qwen4_exp")

# MODEL-MM-QWEN4-EXP (#2449): the hyper-connection INJECT weight's residency.
# Its own target rather than another case in the layer loop, because the claim
# is a DEVICE one and the case that carries it needs a CUDA queue: on the CPU
# tier it reports UNMEASURED and the CPU case beside it is a regression guard
# that cannot see the defect. `qwen4_exp_weights.h` is a MODEL-PRIVATE header
# under src/, the same grant the layer-loop target takes.
vllm_cpp_add_test(test_qwen4_exp_inject_residency
  vllm/models/test_qwen4_exp_inject_residency.cpp)
target_include_directories(test_qwen4_exp_inject_residency PRIVATE ${CMAKE_SOURCE_DIR}/src)

# GLM-5.3-Flash (`Glm5NextForConditionalGeneration`) W1 -- the config layer and
# the two production entry points that reach it: `ModelRegistry::Resolve` and
# the `general.architecture` GGUF dispatch. No numerics: W1 has none. The
# fixture is the REAL published `config.json` byte-for-byte, so this gates the
# port against what the checkpoint says rather than what its author believed.
# `glm5_next.h` is a MODEL-PRIVATE header under src/, the same arrangement
# qwen4_exp uses; the GGUF half is public because the dispatch table includes it.
vllm_cpp_add_test(test_glm5_next_scaffold vllm/models/test_glm5_next_scaffold.cpp)
target_compile_definitions(test_glm5_next_scaffold PRIVATE
  GLM5_NEXT_CKPT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/glm5_next")
target_include_directories(test_glm5_next_scaffold PRIVATE ${CMAKE_SOURCE_DIR}/src)
# `gguf_builder.h` is the shared in-memory GGUF fixture builder under tests/vllm.
target_include_directories(test_glm5_next_scaffold PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm)

# GLM-5.3-Flash W4 -- the mHC residual topology (#2098). Gates the reuse of
# DeepSeek-V4's Sinkhorn/pre/post at this row's constants AND the one piece that
# is NOT V4's: `Glm5NextTextHyperHead` is an unweighted mean, so a port that
# reuses `HcHeadCollapse` runs and emits fluent text through a wrong final
# projection. The goldens are the RUN output of transformers v5.16.1, captured by
# `fixtures/gen_glm5_next_mhc_goldens.py`. `glm5_next_mhc.h` is MODEL-PRIVATE
# under src/, the same arrangement `glm5_next.h` uses.
# W5c (#2242): the weight tower and `load_weights`. The suite reads the
# COMMITTED 1412-tensor header manifest of `unsloth/GLM-5.3-Flash-GGUF
# UD-Q2_K_XL` (`vllm/models/glm5_next_gguf_manifest.inc`, generated by
# `scripts/gen-glm5-next-gguf-manifest.py`) plus one synthetic miniature, so CI
# gates the name map against the real 101.2535 GiB checkpoint with no asset.
# `glm5_next_loader.h` is MODEL-PRIVATE under src/, the same arrangement
# `glm5_next.h` uses.
vllm_cpp_add_test(test_glm5_next_gguf_load
  vllm/models/test_glm5_next_gguf_load.cpp)
target_include_directories(test_glm5_next_gguf_load PRIVATE ${CMAKE_SOURCE_DIR}/src)
# the manifest .inc lives beside the suite
target_include_directories(test_glm5_next_gguf_load PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)

vllm_cpp_add_test(test_glm5_next_mhc vllm/models/test_glm5_next_mhc.cpp)
target_include_directories(test_glm5_next_mhc PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(test_glm5_next_mhc PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures)

# GLM-5.3-Flash W3 -- the DSA indexer's k-pool compression (#2213). Gates the
# stage `deepseek_v4_dsa.cpp` does NOT have: the top-k runs over LEARNED pooled
# candidates, not raw tokens, and the ragged tail is appended raw and unscored.
# The fixture runs at seq_len 21 against index_topk 8 ON PURPOSE -- at or below
# index_topk a raw top-k selects everything and the pooling is unobservable --
# and asserts SET equality of the selected indices, because top-k error is
# bimodal and a tolerance on the values passes a wrong selection. Goldens are the
# RUN output of transformers v5.16.1, captured by
# `fixtures/gen_glm5_next_dsa_goldens.py`. `glm5_next_dsa.h` is MODEL-PRIVATE
# under src/, the same arrangement `glm5_next_mhc.h` uses.
vllm_cpp_add_test(test_glm5_next_dsa vllm/models/test_glm5_next_dsa.cpp)
target_compile_definitions(test_glm5_next_dsa PRIVATE
  GLM5_NEXT_CKPT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/glm5_next")
target_include_directories(test_glm5_next_dsa PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(test_glm5_next_dsa PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures)

# GLM-5.3-Flash W9c-0 -- the k-pool indexer's two DEVICE ops (#2415). The CUDA
# arm of what `test_glm5_next_dsa` gates on the host, run at the SAME fixture
# geometry and against the SAME transformers v5.16.1 goldens, so the device
# answer is compared with the oracle directly and not only with our own C++.
# `np = ceil(21/4) = 6` against `kNumPools = 5` makes the `keep` compaction
# observable; a kernel that skipped it puts the ragged tail three columns late.
# The device cases SKIP without a CUDA backend, so this is a real gate only on a
# leased device -- an `assertions: 0` line here is a skip wearing a pass.
vllm_cpp_add_test(test_glm5_next_kpool_device vllm/models/test_glm5_next_kpool_device.cpp)
target_include_directories(test_glm5_next_kpool_device PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(test_glm5_next_kpool_device PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures)

# GLM-5.3-Flash W5 -- the 288 routed + 1 shared expert MoE, its grouped
# `noaux_tc` router and the clamped-SwiGLU epilogue (#2223). The router is gated
# at the PUBLISHED 288/top-8 and the composed block at 8 experts, because the
# selection is where the expert count matters and the stacked expert tower is
# not. It asserts SET equality of the selected experts and prints the separation
# margin: top-k error is BIMODAL, so a tolerance on the weights passes a wrong
# selection whose values happen to be close. Goldens are the RUN output of
# transformers v5.16.1, captured by `fixtures/gen_glm5_next_moe_goldens.py`.
# `glm5_next_moe.h` is MODEL-PRIVATE under src/, the same arrangement
# `glm5_next_dsa.h` uses.
vllm_cpp_add_test(test_glm5_next_moe vllm/models/test_glm5_next_moe.cpp)
target_compile_definitions(test_glm5_next_moe PRIVATE
  GLM5_NEXT_CKPT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/glm5_next")
target_include_directories(test_glm5_next_moe PRIVATE ${CMAKE_SOURCE_DIR}/src)

# GLM-5.3-Flash W5b-2 -- the decoder layer, the mHC stream threading, the
# assembled `Glm5NextTextModel::Forward` and the KV binding (#2241). Runs a FIVE
# layer model whose schedule carries all four control-flow combinations
# `modeling_glm5_next.py:1261-1272` selects between plus a `shared` DSA layer,
# at the PUBLISHED `hc_mult` of 4, so the `[T, hc_mult, hidden]` manifold is
# asserted per stream and against an EARLY-COLLAPSE decoy captured from the same
# oracle run -- a port that threads `[T, hidden]` is finite, right-shaped and
# fluent, and nothing else in this tree catches it. Goldens are the RUN output
# of transformers v5.16.1, captured by
# `fixtures/gen_glm5_next_layer_goldens.py`. `glm5_next_layer.h` is
# MODEL-PRIVATE under src/, the same arrangement `glm5_next_attn.h` uses; the
# published `config.json` is read so the cache widths the layer fills are gated
# against the ones the production `make_kv_cache` hook publishes.
vllm_cpp_add_test(test_glm5_next_layer vllm/models/test_glm5_next_layer.cpp)
target_compile_definitions(test_glm5_next_layer PRIVATE
  GLM5_NEXT_CKPT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/glm5_next")
target_include_directories(test_glm5_next_layer PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(test_glm5_next_layer PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures)
target_include_directories(test_glm5_next_moe PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures)

# GLM-5.3-Flash W5b-1 -- `Glm5NextTextAttention` and its CROSS-LAYER top-k
# sharing (#2241). The highest-value case is the `shared` layer: a port that
# recomputes its own indexer where upstream reuses the previous layer's
# selection RUNS and emits plausible tokens, so the fixture carries BOTH the
# correct output and what a recomputing port produces and asserts ours is the
# first. It also carries a SQUARE `kv_b_proj` case, where the untransposed
# reading of the converter's half-transposed `k_b` is shape-valid and merely
# wrong, and a left-padded row whose every key is masked -- upstream fills that
# with `finfo.min` and not `-inf`, so its output is FINITE. Goldens are the RUN
# output of transformers v5.16.1, captured by
# `fixtures/gen_glm5_next_attn_goldens.py`, which asserts the sha256 of the
# INSTALLED reference module. `glm5_next_attn.h` is MODEL-PRIVATE under src/,
# the same arrangement `glm5_next_dsa.h` uses.
vllm_cpp_add_test(test_glm5_next_attn vllm/models/test_glm5_next_attn.cpp)
target_compile_definitions(test_glm5_next_attn PRIVATE
  GLM5_NEXT_CKPT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/glm5_next")
target_include_directories(test_glm5_next_attn PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(test_glm5_next_attn PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures)

# GLM-5.3-Flash W5b-1 -- the `OwnedTensor` -> host f32 bridge and the residency
# decision it implements (#2241, spec `## Owed` O22). Pins the arithmetic that
# forces the per-layer choice -- 0.4654 GiB for one bridged DSA layer against a
# 426.72 GiB materialized tower and a ~119.63 GiB box -- and pins the 1 GiB
# per-tensor ceiling BETWEEN the largest legitimate tensor (`o_proj`, 0.25 GiB)
# and the smallest expert bank (`up_exps`, 9.0 GiB), so it is neither a mute
# switch nor a gate that fires on ordinary work. Runs over the synthetic
# `glm5next` miniature through the PRODUCTION `load_weights` hook.
vllm_cpp_add_test(test_glm5_next_bridge vllm/models/test_glm5_next_bridge.cpp)
target_include_directories(test_glm5_next_bridge PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(test_glm5_next_bridge PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)

# GLM-5.3-Flash W5b-2b -- THE ENGINE BINDING and the reachability proof this row
# has owed since W2 (#2241, spec `## Owed` O15/O16/O17/O23/O25/O26). Every case
# enters through `ModelRegistry::Forward` on the synthetic `glm5next` miniature,
# so deleting the `Glm5NextHostForward` call in the registry's forward hook reds
# the suite. The independently assembled RESIDENT tower is the oracle for the
# streamed path's composition, and the `lm_head` chunk size is a parameter
# because at any geometry small enough to run in a test the default gives ONE
# chunk and the loop would never be entered twice.
vllm_cpp_add_test(test_glm5_next_forward vllm/models/test_glm5_next_forward.cpp)
vllm_cpp_add_test(test_glm5_next_placement vllm/models/test_glm5_next_placement.cpp)
target_include_directories(test_glm5_next_placement PRIVATE ${CMAKE_SOURCE_DIR}/src)
set_tests_properties(test_glm5_next_placement PROPERTIES
  ENVIRONMENT "VT_GLM5_NEXT_DEVICE_EXPERTS=1")
add_test(NAME test_glm5_next_placement_default
  COMMAND ${CMAKE_COMMAND} -E env --unset=VT_GLM5_NEXT_DEVICE_EXPERTS
  $<TARGET_FILE:test_glm5_next_placement>
  "--test-case=glm5_next placement: loaded CPU experts never upload to ROCm")
add_test(NAME test_glm5_next_placement_off COMMAND test_glm5_next_placement
  "--test-case=glm5_next placement: loaded CPU experts never upload to ROCm")
set_tests_properties(test_glm5_next_placement_off PROPERTIES
  ENVIRONMENT "VT_GLM5_NEXT_DEVICE_EXPERTS=0")
add_test(NAME test_glm5_next_placement_nonexact COMMAND test_glm5_next_placement
  "--test-case=glm5_next placement: loaded CPU experts never upload to ROCm")
set_tests_properties(test_glm5_next_placement_nonexact PROPERTIES
  ENVIRONMENT "VT_GLM5_NEXT_DEVICE_EXPERTS=11")
target_include_directories(test_glm5_next_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(test_glm5_next_forward PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)

# dots3-note W3 -- the FULL-attention layer (#699, #1846,
# .agents/specs/dots3-note.md §7 W3). Checks `_forward_note_mla`'s four deltas
# over plain DeepSeek MLA -- the two lora rescales, `k_rope_only_layernorm`, the
# headwise gate and the DSA indexer's rope geometry -- against an INDEPENDENT
# double-precision reference transcribed from the upstream python
# (`vllm/models/dots3_note/nvidia/model.py`, `vllm/model_executor/models/
# deepseek_v2.py`, `vllm/ir/ops/layernorm.py`, the rotary-embedding pair) rather
# than from our C++. Spec §6.4 option B: this model has NO oracle on any
# hardware this project owns, so the reference IS the correctness argument and
# a shared-helper comparison would prove nothing. CPU-only, no checkpoint, no
# GPU, no speed claim.
vllm_cpp_add_test(test_dots3_note_attn vllm/models/test_dots3_note_attn.cpp)
target_compile_definitions(test_dots3_note_attn PRIVATE
  DOTS3_NOTE_CKPT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/dots3_note_prev")
# Same model-private-header arrangement as the scaffold gate above.
target_include_directories(test_dots3_note_attn PRIVATE ${CMAKE_SOURCE_DIR}/src)

# dots3-note W6a (#2512, spec §4.11): the DENSE VISION TOWER against an
# INDEPENDENT double-precision reference written from vLLM `nvidia/vision.py` +
# `nvidia/vision_attention.py` at `9035151d6`, plus the config traps, the
# refuse-by-name table, the 235-tensor enumeration and the image processor.
#
# A CONSISTENCY gate and not a correctness one, and the file says so in its own
# header: spec §6.4 option B records that NO oracle for this model runs on any
# hardware this project owns, so nothing here is compared against vLLM and no
# performance number is claimable. CPU-only, no checkpoint download, no GPU.
vllm_cpp_add_test(test_dots3_note_vision vllm/models/test_dots3_note_vision.cpp)
target_compile_definitions(test_dots3_note_vision PRIVATE
  DOTS3_NOTE_CKPT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/dots3_note_prev"
  # W9d (#2881): the reference-independence case re-reads THIS file and counts
  # the qualified names in each reference namespace, through the shared
  # instrument in `dots3_note_ref_independence.h`. Same arrangement the audio
  # target uses; the instrument takes the path as an argument precisely so the
  # two targets share one scan and one pp-number rule.
  DOTS3_VISION_TEST_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/test_dots3_note_vision.cpp")

# dots3-note W7a (#2703): the AUDIO tower and its front end. Two fixture
# directories, because the two halves are gated against different things — the
# tower against the committed dots3-note config, and the SHARED slaney
# filterbank against `voxtral_mel_filters_f32.bin`, which is a third party's
# implementation of the same `mel_filter_bank` call and the only real oracle
# this row gets.
vllm_cpp_add_test(test_dots3_note_audio vllm/models/test_dots3_note_audio.cpp)
target_compile_definitions(test_dots3_note_audio PRIVATE
  DOTS3_NOTE_CKPT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/dots3_note_prev"
  VOXTRAL_AUDIO_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/multimodal/fixtures/voxtral_audio"
  # The reference-independence case re-reads THIS file and counts the qualified
  # names in each reference namespace, so the enumeration is a measurement
  # rather than a transcription asserted against itself.
  DOTS3_AUDIO_TEST_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/test_dots3_note_audio.cpp")
# dots3_note.h and dots3_note_audio.h are MODEL-PRIVATE headers under src/, not
# include/vllm/ — same arrangement test_dots3_note_vision uses.
target_include_directories(test_dots3_note_audio PRIVATE
  ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
# Same model-private-header arrangement as the two gates above: dots3_note.h and
# dots3_note_vision.h live under src/, because W1 ships nothing on the public ABI
# and include/vllm/ is a USER_USAGE_PREFIXES surface (#515).
target_include_directories(test_dots3_note_vision PRIVATE
  ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)

# Nemotron-H (NemotronHForCausalLM) W3 STRUCTURAL gate (#517,
# .agents/specs/nemotron-h-model.md §4 W3): registry-resolve + config descent off
# the REAL released config.json (52-entry layers_block_type schedule, the legacy
# hybrid_override_pattern/mamba_* aliases, the refuse cases) + the on-disk name map
# accounted BOTH WAYS against all 18487 tensors of the released index + the
# heterogeneous KV topology (6-layer full-attention group + 23-layer Mamba2 group,
# shapes from mamba2_state_shape). The structural half runs off a COMMITTED
# headers-only fixture so CI never needs the 20.1 GiB NAS asset; a final case
# re-verifies that fixture against the LIVE checkpoint when VT_NEMOTRON35_SNAPSHOT
# names it, and SKIPs loudly otherwise. CPU-only, no forward (W4), no speed claim.
vllm_cpp_add_test(test_nemotron_h_scaffold vllm/models/test_nemotron_h_scaffold.cpp)
target_compile_definitions(test_nemotron_h_scaffold PRIVATE
  NEMOTRON_H_CKPT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/nemotron_h_35_lightning")
# GATE-PIN-UNPINNED-SNAPSHOTS (#471): the live re-verification resolves the
# checkpoint through the pinned parity::HfSnapshot, so it needs tests/parity on
# the include path.
# nemotron_h.h is a MODEL-PRIVATE header under src/, not include/vllm/: W3 ships
# nothing on the public ABI and include/vllm/ is a USER_USAGE_PREFIXES surface
# (#515). Same call as the W1 resolver header.
target_include_directories(test_nemotron_h_scaffold PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# Nemotron-H W4 FORWARD gate (#517, .agents/specs/nemotron-h-model.md §4 W4): the
# hybrid layer loop over all three block kinds, every block compared ELEMENTWISE
# against a reference written independently in `double` from the upstream
# formulas — not against a second call of the code under test, and not through a
# helper both arms share. Both dtype arms (f32 and the released bf16) on every
# case; a LONG-prompt arm as well as a short one, because only 6 of 52 layers are
# attention; the SSD chunk-size invariance; the SSM state dtype carried across
# two legs, which is the only way it is observable at all; the routed scale on
# the OUTPUT rather than the router weights or logits; the f32 router; and the
# no-RoPE finding. CPU-only, no checkpoint, no speed claim.
vllm_cpp_add_test(test_nemotron_h_forward vllm/models/test_nemotron_h_forward.cpp)
# nemotron_h.h / nemotron_h_forward.h are MODEL-PRIVATE headers under src/, not
# include/vllm/: W4 ships nothing on the public ABI and include/vllm/ is a
# USER_USAGE_PREFIXES surface (#515). Same call as the W1 resolver header.
target_include_directories(test_nemotron_h_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Nemotron-H WEIGHT LOADER gate (#517, .agents/specs/nemotron-h-model.md §5b and
# §6b "Still owed after W4"): materializes the 18487 released tensors from the
# REAL 20.1 GiB checkpoint and runs the registered forward on them. Gated
# STRUCTURALLY, not by tokens — every tensor materialized or deferred BY NAME,
# the per-scheme composition (5935 NVFP4 W4A16 g16 triples + 46 FP8 W8A8 static
# triples + the bf16/f32 remainder) asserted against the checkpoint's own memory
# format, and the loaded dtypes asserted to be the SHIPPED ones rather than
# wider. The checkpoint resolves through the CONTENT-pinned
# parity::Nemotron35LightningSnapshot (#569) and the gate SKIPS loudly when it is
# absent -- a NAMED, COUNTED skip with a closing accounting case, because a skip
# that prints `Passed 0.00 sec` with zero assertions is indistinguishable from a
# real pass -- so CI never needs the NAS asset. Not the W6 token gate and no
# speed claim: it consumes the committed oracle.json for ONE forward per prompt
# and the argmax of the last position against that prompt's FIRST token; the
# full 32-token greedy decode stays W6's.
vllm_cpp_add_test(test_nemotron_h_loader vllm/models/test_nemotron_h_loader.cpp)
target_compile_definitions(test_nemotron_h_loader PRIVATE
  NEMOTRON_H_GOLDENS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/parity/goldens/nemotron_35_lightning_greedy")
# nemotron_h.h / nemotron_h_loader.h are MODEL-PRIVATE headers under src/, not
# include/vllm/ (#515); tests/parity carries the content-pinned snapshot
# resolver.
target_include_directories(test_nemotron_h_loader PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# Nemotron-H MIXED_PRECISION MEMORY FORMS, OFFLINE (#517,
# .agents/specs/nemotron-h-model.md §6d). The loader's design claim — every
# weight held in the format it SHIPS in, widened only TRANSIENTLY at the GEMM
# call site — lived entirely behind the 20.1 GiB checkpoint gate above, so on a
# runner with no CHECKPOINT_ROOT it was not executed at all: deleting
# NemotronHOwned::View's refusal left the whole suite green. This gate needs NO
# checkpoint and therefore runs on EVERY arm. It asserts that View REFUSES a
# non-dense weight by name, and that DenseBf16 reproduces a dequant derived
# INDEPENDENTLY from the upstream ModelOpt formula — the nibble order, the
# per-16 group scale, the multiplied weight_scale_2, and the FP8 input_scale
# that is carried and NOT applied. Those are exactly §6d's M4/M5 defects, which
# leave every structural count correct. CPU-only, no golden, no speed claim.
vllm_cpp_add_test(test_nemotron_h_quantized_forms
  vllm/models/test_nemotron_h_quantized_forms.cpp)
# nemotron_h.h / nemotron_h_forward.h are MODEL-PRIVATE headers under src/, not
# include/vllm/ (#515).
target_include_directories(test_nemotron_h_quantized_forms PRIVATE
  ${CMAKE_SOURCE_DIR}/src)

# A2-Q2a (#810): the DEVICE MoE arm on a SYNTHETIC NVFP4 fixture — the cheap RED
# in front of the real-checkpoint gate. It exists because `BuildTiny` in
# test_nemotron_h_forward.cpp is all-`kDense` (:892 -> PackMoe :873) and so
# cannot reach the NVFP4 arena at all; without it the 21 GB checkpoint would be
# the ONLY instrument and every RED would cost a GB10 window. Its geometry
# (hidden 128, moe_intermediate 64) is the smallest that resolves on Marlin's
# thread configs, which `TinyParams` (24 / 10) does not. Skips LOUDLY with a
# reason on a GPU-less box.
vllm_cpp_add_test(test_nemotron_h_moe_device
  vllm/models/test_nemotron_h_moe_device.cpp)
target_include_directories(test_nemotron_h_moe_device PRIVATE
  ${CMAKE_SOURCE_DIR}/src)

# A2-P (#810, .agents/specs/nemotron-h-a2p-paged-forward.md): the PAGED forward,
# gated through the PRODUCTION entry point. Every case builds a real
# NemotronHLoadedModel from a synthetic checkpoint, hands it to a real
# GPUModelRunner, and drives execute_model / sample_tokens — so the paged
# forward is reached at runner.cpp:1465 through ModelRegistry::Forward, never
# through a fabricated ModelForwardInput. That is the difference AGENTS.md
# §"Nothing lands dead" is about: before this change NemotronHDeviceForward had
# ONE non-declaration call site in the whole tree and it was a test.
#
# The multi-step arm is the only one that can see a DROPPED CARRY (with one leg
# and fresh state the recurrent half is unobservable — nemotron_h_forward.h
# says so outright), the per-block numeric arm is the only one that can see a
# dropped mechanism whose argmax is unchanged, and the page dtypes are asserted
# directly because a too-WIDE page is numerically correct and invisible to both.
# CPU-only, no checkpoint, no speed claim: the A3 end-to-end token gate against
# the released checkpoint is the on-box binding gate.
vllm_cpp_add_test(test_nemotron_h_paged_forward
  vllm/models/test_nemotron_h_paged_forward.cpp)
target_include_directories(test_nemotron_h_paged_forward PRIVATE
  ${CMAKE_SOURCE_DIR}/src)

# Kimi-Linear CPU REFERENCE FORWARD gates (W2-W6): the per-op reference forwards
# (KDA layer vs the kimi_kda host-ref composition; NoPE-MLA vs a materialized-MHA
# reference; sigmoid noaux_tc router + block vs a hand-computed case) and the whole
# 2-layer forward (loader materializes host weights, finite/coherent, greedy-decodes
# N tokens with the context advancing). The DEVICE forward stays refuse-by-name
# (born-on-runner W6/W7). .agents/specs/kimi-linear.md §5.
vllm_cpp_add_test(test_kimi_linear_forward vllm/models/test_kimi_linear_forward.cpp)
vllm_cpp_add_test(test_kimi_linear_paged vllm/models/test_kimi_linear_paged.cpp)
vllm_cpp_add_test(test_kimi_linear_fold_gate vllm/models/test_kimi_linear_fold_gate.cpp)
# #2840: the committed K=3 oracle capture is what LICENSES the strict bar, and
# the checkpoint-free case that reads it needs to find the goldens tree.
target_compile_definitions(test_kimi_linear_fold_gate PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")

# KDA (Kimi Delta Attention) net-new gated-linear-attention primitives — host
# reference + unit gate (KERNEL-KDA-DELTA). Shared unblocker for Kimi-Linear-48B
# and Kimi-K3 (W4). Hand-derived literal cases + double-precision references over
# the per-channel [H,D] low-rank decay, the sigmoid-gated output norm, the 3 q/k/v
# short convs and q/k L2-norm. Full-model gate is the Kimi-Linear-48B proxy
# (DGX-blocked); mirrors the DSA lane (.agents/specs/kda-kernel-delta.md).
vllm_cpp_add_test(test_kimi_kda vllm/models/test_kimi_kda.cpp)

# GLM-5.3-Flash (MODEL-MM-GLM53-FLASH W2) — the KDA arm's numerics.
vllm_cpp_add_test(test_glm5_next_kda vllm/models/test_glm5_next_kda.cpp)

# MLA CAMPAIGN W8 (DeepSeek-V2-Lite, DeepseekV2ForCausalLM) — the SACRED MLA
# greedy correctness gate: drives a compact prompt battery through the paged
# engine and checks STRICT token-exact agreement with the vLLM 0.25.0 oracle
# (which W0 measured DETERMINISTIC on this model at batch=1, K=5, 8/8 prompts).
# Also gates the SCHEDULER/RUNNER batch ORDER MLA requires (decodes first,
# with-context prefills leading the prefill tail) with staggered concurrent
# requests + a prefix-cache-driven with-context prefill.
# Checkpoint-gated, dgx-only; skips on CPU/CI when the snapshot/goldens are absent.
vllm_cpp_add_test(test_deepseek_v2_paged_engine vllm/models/test_deepseek_v2_paged_engine.cpp)
target_compile_definitions(test_deepseek_v2_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_deepseek_v2_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# GLM/DSA G1 (GLM-4.7-Flash, Glm4MoeLiteForCausalLM) — reuses the DeepSeek-V2 MLA
# stack over the SAME DeepseekV2Weights; the FIRST e2e exercise of the q_lora
# query branch AND the noaux_tc grouped router. Loader gate (host, MTP-tail skip)
# + the SACRED paged-engine greedy gate (checkpoint-gated, dgx-only).
vllm_cpp_add_test(test_glm4_moe_lite_load vllm/models/test_glm4_moe_lite_load.cpp)
# The MoE ROUTER DTYPE gate (#2928): GLM's fp32 router is a property of
# `Glm4MoE`, not a config key, and the difference is measured through the CPU
# forward at GLM's top-4-of-64 `noaux_tc` shape. Host-only, no checkpoint.
vllm_cpp_add_test(test_glm4_moe_lite_router_dtype vllm/models/test_glm4_moe_lite_router_dtype.cpp)
vllm_cpp_add_test(test_glm4_moe_lite_paged_engine vllm/models/test_glm4_moe_lite_paged_engine.cpp)
target_compile_definitions(test_glm4_moe_lite_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_glm4_moe_lite_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_qwen27_paged_forward vllm/models/test_qwen27_paged_forward.cpp)
target_include_directories(test_qwen27_paged_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)
# #2534. The GGUF keep-quant lm_head must emit logits with f32 RESOLUTION, which
# is the property the Q4_K_M token gate is decided by and which no token gate we
# own can see. Public headers only, so no extra include directory.
vllm_cpp_add_test(test_qwen35_gguf_logits_f32 vllm/models/test_qwen35_gguf_logits_f32.cpp)
# BACKEND-TENSTORRENT-KEEPQUANT W3 (#2959): the q4km vehicle's dequantized-
# artifact dump — the teacher-forcing oracle input the row spec prescribes
# (our bit-exact loader writes a safetensors dir; the pinned transformers
# cannot parse the `qwen35` GGUF arch). Env-gated: skips loudly unless
# VLLM_CPP_QWEN35_Q4KM_GGUF and VLLM_CPP_QWEN35_Q4KM_DUMP_DIR are both set.
vllm_cpp_add_test(test_qwen35_gguf_q4km_dequant_dump vllm/models/test_qwen35_gguf_q4km_dequant_dump.cpp)
# #2534. `detail::ActDType` caches its getenv in a function-local static, so one
# process observes exactly ONE value of VT_ACT_F32 and a single run cannot tell a
# resolver that READS the lever from one hardwired to its default answer. Re-run
# the SAME binary with the lever ON, the shape `test_qwen35_paged_forward_gdn_out_f32`
# and `_glue_fuse_off` above already use.
#
# SCOPED, deliberately, and this entry is NOT the whole binary. `VT_ACT_F32=1` is
# REFUSED at the resolver (the f32 conversion is incomplete; see qwen3_5.cpp and
# the row spec), so the nine cases that CONSTRUCT a qwen35 model legitimately
# cannot run under the lever -- they would hit the refusal as an unexpected
# `THREW exception` and red this entry for doing exactly what it was told to do.
# What this entry has to prove is narrower: that the resolver READS the lever.
# The two `qwen27 activation*` cases are the ones that assert that, and neither
# builds a model.
#
# The filter is `qwen27 activation*` and NOT a full name because doctest's
# `--test-case` splits its argument on COMMAS: a name containing one silently
# matches nothing, the binary exits 0 having run zero cases, and the entry passes
# while testing nothing. Both target names are comma-free today and the wildcard
# keeps that true if either is reworded.
#
# The two properties below are what stop a mute pass. FAIL_REGULAR_EXPRESSION
# takes precedence in ctest, so a run that matched no case -- `assertions: 0` --
# FAILS instead of reporting success, which is the `zero assertions is a skip
# wearing a pass` trap this repository has already been bitten by. The PASS regex
# then requires doctest's own verdict rather than trusting the exit status.
add_test(NAME test_qwen27_paged_forward_act_f32
  COMMAND test_qwen27_paged_forward "--test-case=qwen27 activation*")
set_tests_properties(test_qwen27_paged_forward_act_f32 PROPERTIES
  ENVIRONMENT "VT_ACT_F32=1"
  FAIL_REGULAR_EXPRESSION "assertions:[ ]*0[ |]"
  PASS_REGULAR_EXPRESSION "Status: SUCCESS!")
# #2590. THE REACHABILITY GATE for the activation dump, and it is a gate rather
# than an observation because the hermetic writer suite
# (`test_act_dump`) drives the writer DIRECTLY and therefore proves the class
# works, never that anything reaches it -- the exact shape `AGENTS.md` "Nothing
# lands dead" names.
#
# This entry runs the SAME paged-forward binary with the dump pointed at a real
# directory, so the dump is exercised through `DenseForwardLayers`, the
# production dense forward. It passes only on a line the instrument prints when
# it has actually written a RESIDUAL-STREAM blob: `stream_blobs=` followed by a
# non-zero count. Deleting the `ActDumpStream` call site therefore reds THIS
# entry while leaving the plain binary green, which is what makes it measure a
# capability.
#
# `blobs_this_step` would NOT do, and a mutation proved it rather than an
# argument: the GDN stage probes fire off the same environment variable, so the
# total stayed non-zero with both stream call sites deleted and a gate keyed on
# it reported success over a forward that dumped no hidden state at all.
#
# The directory is created at CONFIGURE time on purpose. The writer refuses a
# missing directory by design (#2590), so a `mkdir` folded into the test command
# would be testing the shell rather than the instrument.
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/act_dump_2590)
add_test(NAME test_qwen27_paged_forward_act_dump
  COMMAND test_qwen27_paged_forward)
set_tests_properties(test_qwen27_paged_forward_act_dump PROPERTIES
  ENVIRONMENT "VT_DUMP_ACT=${CMAKE_CURRENT_BINARY_DIR}/act_dump_2590"
  FAIL_REGULAR_EXPRESSION "assertions:[ ]*0[ |];Status: FAILURE"
  PASS_REGULAR_EXPRESSION "stream_blobs=[1-9]")
vllm_cpp_add_test(test_unicode_data vllm/test_unicode_data.cpp)
vllm_cpp_add_test(test_pretokenizer vllm/test_pretokenizer.cpp)
# Metaspace split=true + Metaspace decoder (Parakeet tokenizer shape) —
# ARCH-ONE-SURFACE ROW 1 W1(a); gates against the pre-refactor DecodeIds
# reference on the committed parakeet_e2e fixture tokenizers.
vllm_cpp_add_test(test_tokenizer_metaspace_split
  vllm/test_tokenizer_metaspace_split.cpp)
target_compile_definitions(test_tokenizer_metaspace_split PRIVATE
  PARAKEET_E2E_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/parakeet_e2e")
vllm_cpp_add_test(test_bpe vllm/test_bpe.cpp)
# The load-time MergeTokenOutOfVocabulary refusal asserts that every
# committed HF golden still loads, so test_bpe reaches the goldens dir.
target_compile_definitions(test_bpe PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
# SPEC-BPE-QUADRATIC-MERGE (#1365) — the long-pretoken equivalence corpus and
# the absolute cost bound. The corpus ids are HF `tokenizers` 0.22.2 output over
# the two committed goldens (an oracle, not a snapshot of ours); the cost case
# is the row's ONLY timing assertion and it is RED against the O(n^2) merge.
vllm_cpp_add_test(test_bpe_equivalence vllm/test_bpe_equivalence.cpp)
target_compile_definitions(test_bpe_equivalence PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
vllm_cpp_add_test(test_detokenizer vllm/test_detokenizer.cpp)
vllm_cpp_add_test(test_request vllm/test_request.cpp)
vllm_cpp_add_test(test_kv_cache_utils vllm/v1/test_kv_cache_utils.cpp)
# issue #371: the recurrent-state budget (spec widens GDN state by k+1).
vllm_cpp_add_test(test_kv_state_budget vllm/v1/core/test_kv_state_budget.cpp)
# issue #1983: the recurrent-state pool sized by the KV budget, not by
# --max-num-seqs. Driven from the REAL registry builder, never a hand-built group.
vllm_cpp_add_test(test_hybrid_kv_budget vllm/v1/core/test_hybrid_kv_budget.cpp)
# MakeQwen3_5KVCacheSpec lives in src/, and driving the REAL registry builder
# rather than a hand-built group is the whole point of the fixture (#1963).
target_include_directories(test_hybrid_kv_budget PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_prefix_match_unit vllm/v1/test_prefix_match_unit.cpp)
vllm_cpp_add_test(test_kv_cache_interface vllm/v1/test_kv_cache_interface.cpp)
vllm_cpp_add_test(test_block_pool vllm/v1/test_block_pool.cpp)
vllm_cpp_add_test(test_kv_events vllm/v1/test_kv_events.cpp)
vllm_cpp_add_test(test_single_type_kv_cache_manager vllm/v1/test_single_type_kv_cache_manager.cpp)
vllm_cpp_add_test(test_kv_cache_coordinator vllm/v1/test_kv_cache_coordinator.cpp)
vllm_cpp_add_test(test_kv_cache_manager vllm/v1/test_kv_cache_manager.cpp)
vllm_cpp_add_test(test_none_hash_determinism vllm/v1/test_none_hash_determinism.cpp)
# This suite explicitly exercises configured and unconfigured hash seeding in
# child processes. Insulate its unconfigured control arm from a developer shell
# that happens to export Python's seed (the Nix dev shell exports 0).
set_tests_properties(test_none_hash_determinism PROPERTIES
  ENVIRONMENT_MODIFICATION "PYTHONHASHSEED=unset:")
vllm_cpp_add_test(test_prefix_cache_stats vllm/v1/test_prefix_cache_stats.cpp)
vllm_cpp_add_test(test_prometheus_metrics vllm/v1/test_prometheus_metrics.cpp)
vllm_cpp_add_test(test_kv_offload_cpu vllm/v1/test_kv_offload_cpu.cpp)
vllm_cpp_add_test(test_kv_offload_fs vllm/v1/test_kv_offload_fs.cpp)
vllm_cpp_add_test(test_kv_offload_tiering vllm/v1/test_kv_offload_tiering.cpp)
vllm_cpp_add_test(test_kv_offload_connector vllm/v1/test_kv_offload_connector.cpp)
# LMCache MODE-1 (lm://) wire codec — KV-EXTERNAL-CACHE W1. Byte/bit-exact
# against fixtures captured from the real Python codec (struct + the `blake3`
# PyPI package + numpy); see tests/fixtures/lmcache/lmcache_fixtures.json.
vllm_cpp_add_test(test_lmcache_codec vllm/v1/kv_offload/lmcache/test_lmcache_codec.cpp)
target_compile_definitions(test_lmcache_codec PRIVATE
  TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")
# DeepSeek-V4 Vision W1: pinned prompt encoding and image preprocessing.
vllm_cpp_add_test(test_deepseek_v4_encoding
                  vllm/multimodal/test_deepseek_v4_encoding.cpp)
vllm_cpp_add_test(test_deepseek_v4_image_processor
                  vllm/multimodal/test_deepseek_v4_image_processor.cpp)
# DeepSeek-V4 Vision W2: pinned reduced-shape ViT and aligner stage parity.
vllm_cpp_add_test(test_deepseek_v4_vision
                  vllm/models/test_deepseek_v4_vision.cpp)
target_compile_definitions(test_deepseek_v4_vision PRIVATE
  DEEPSEEK_V4_VISION_GOLDEN="${CMAKE_SOURCE_DIR}/tests/parity/goldens/deepseek_v4_vision/goldens.json")
# DeepSeek-V4 Vision W3A: the `deepseek4v` arm of the `clip` mmproj reader. It
# gates the four layout mismatches between what `mmproj-BF16.gguf` stores and
# what the W2 tower consumes, and every one of them is a silent wrong answer
# rather than a crash. It proves nothing about reachability: W4 owns the
# production call site.
vllm_cpp_add_test(test_deepseek_v4_mmproj
                  vllm/models/test_deepseek_v4_mmproj.cpp)
target_include_directories(test_deepseek_v4_mmproj PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)

vllm_cpp_add_test(test_deepseek_v4_mm_reach
                  vllm/models/test_deepseek_v4_mm_reach.cpp)
target_include_directories(test_deepseek_v4_mm_reach PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models
                           ${CMAKE_CURRENT_SOURCE_DIR}/vllm)
# MODEL-MM-deepseek-v4 (#2411): can a checkpoint whose published KV cache groups
# DISAGREE about their block size be served at all? It enters at
# `LoadedEngine::FromModelDir` on the DEFAULT configuration, so it needs the
# language GGUF fixture beside the model gates, exactly as the suite above does.
vllm_cpp_add_test(test_deepseek_v4_multigroup_kv
                  vllm/entrypoints/test_deepseek_v4_multigroup_kv.cpp)
target_include_directories(test_deepseek_v4_multigroup_kv PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models
                           ${CMAKE_CURRENT_SOURCE_DIR}/vllm)
# Multimodal M1 processor-parity gate: C++ Qwen3-VL image pipeline
# (pixel_values/grid/mm-hash/placeholder-expansion) BIT-identical to the vLLM
# 0.25.0 oracle fixture (tests/vllm/multimodal/fixtures/qwen3vl).
vllm_cpp_add_test(test_qwen3vl_processor vllm/multimodal/test_qwen3vl_processor.cpp)
target_compile_definitions(test_qwen3vl_processor PRIVATE
  MM_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl")
# Multimodal M2a vision-tower faithfulness gate: the C++ Qwen3_VisionTransformer
# forward vs the dumped vLLM 0.25.0 tower reference (patch-embed/block0/merger/
# DeepStack taps + full [196,10240] output). GPU + VLLM_QWEN3VL_WEIGHTS required,
# else skipped (scripts/mm/m2a_tower_ref_dump.py + m2a_tower_weight_dump.py).
vllm_cpp_add_test(test_qwen3vl_tower vllm/multimodal/test_qwen3vl_tower.cpp)
target_compile_definitions(test_qwen3vl_tower PRIVATE
  MM_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl"
  TOWER_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl_tower")
# Gemma-4 G2-impl: NaFlex SigLIP2 vision-tower per-stage faithfulness gate
# (patch-embed -> encoder -> pooled -> projected vs transformers-eager refs).
# Weights via VLLM_GEMMA4_VISION_WEIGHTS (dumped, uncommitted); refs default to
# the committed golden vision_refs dir (VLLM_GEMMA4_VISION_REFS to override).
vllm_cpp_add_test(test_gemma4_vision_tower vllm/multimodal/test_gemma4_vision_tower.cpp)
target_include_directories(test_gemma4_vision_tower PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(test_gemma4_vision_tower PRIVATE
  GEMMA4_VISION_REF_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens/gemma4_e4b_image/vision_refs")
# CLAIM-GEMMA4-MM-E2E: the STRICT image->text token-exact gate THROUGH the ENGINE
# registered mm-forward (ModelRegistry::Forward). dgx-only (CUDA + cached E4B).
vllm_cpp_add_test(test_gemma4_registry_e2e vllm/multimodal/test_gemma4_registry_e2e.cpp)
target_include_directories(test_gemma4_registry_e2e PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(test_gemma4_registry_e2e PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
# Gemma-4 G3: USM-Conformer audio-tower per-stage faithfulness gate (subsample ->
# rel-pos-enc -> N conformer layers -> output_proj -> projected vs the dumped
# transformers-eager refs). Host f32, no GPU. Weights via VLLM_GEMMA4_AUDIO_WEIGHTS
# (dumped, uncommitted); refs from the committed golden audio_refs dir.
vllm_cpp_add_test(test_gemma4_audio_tower vllm/multimodal/test_gemma4_audio_tower.cpp)
target_include_directories(test_gemma4_audio_tower PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(test_gemma4_audio_tower PRIVATE
  GEMMA4_AUDIO_REF_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens/gemma4_e4b_audio/audio_refs")
# Multimodal M2b/M2c text-backbone unit gates: MRoPE get_rope_index, 3-section
# rope application (vt::RopeFromCache), DeepStack scatter, and embed-merge —
# BIT/near-exact vs the dumped vLLM 0.25.0 reference (CPU-only, no weights).
vllm_cpp_add_test(test_qwen3vl_text vllm/multimodal/test_qwen3vl_text.cpp)
target_compile_definitions(test_qwen3vl_text PRIVATE
  TEXT_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl_text")
# Multimodal M2c STRICT e2e image->text token-exact gate on Qwen3-VL-4B: the full
# pipeline (processor -> tower -> merge -> forked MRoPE/DeepStack greedy decode)
# vs the committed vLLM 0.25.0 golden. GPU + Qwen3-VL-4B checkpoint required, else
# skipped (VLLM_QWEN3VL_CKPT or the default HF cache path).
vllm_cpp_add_test(test_qwen3vl_e2e vllm/multimodal/test_qwen3vl_e2e.cpp)
target_compile_definitions(test_qwen3vl_e2e PRIVATE
  MM_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl"
  TEXT_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl_text")
# MM-ENGINE-FORWARD — the SAME STRICT image->text token-exact gate driven THROUGH
# the ENGINE registered forward (ModelRegistry::Forward → ForwardQwen3VL...), the
# engine-mm-forward closing gate. GPU + Qwen3-VL-4B checkpoint required, else skipped.
vllm_cpp_add_test(test_qwen3vl_registry_e2e vllm/multimodal/test_qwen3vl_registry_e2e.cpp)
target_compile_definitions(test_qwen3vl_registry_e2e PRIVATE
  MM_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl"
  TEXT_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl_text")
# M3-b — STRICT image->text token-exact on the Qwen3.6-27B GDN-hybrid gate model
# vs the committed vLLM 0.25.0 golden. GPU + the vision-inclusive bf16 checkpoint
# Qwen/Qwen3.6-27B required, else skipped (VLLM_QWEN36_CKPT or the HF cache path).
vllm_cpp_add_test(test_qwen3_5_vl_e2e vllm/multimodal/test_qwen3_5_vl_e2e.cpp)
target_compile_definitions(test_qwen3_5_vl_e2e PRIVATE
  MM_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl"
  GOLD_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3_5_27b")
# MoE vision tower (#891) — the HARDWARE gate on Qwen/Qwen3.6-35B-A3B bf16: the
# 333 model.visual.* tensors load and the tower runs; the full ~67 GiB e2e is
# opt-in behind VLLM_MOE_VISION_E2E=1. Skipped without the checkpoint.
vllm_cpp_add_test(test_qwen3_5_moe_vision_hw vllm/multimodal/test_qwen3_5_moe_vl_hw.cpp)
target_compile_definitions(test_qwen3_5_moe_vision_hw PRIVATE
  MM_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl")
# Vision-tower SPEED A-B (CLAIM-MULTIMODAL-SPEED-TOWER): baseline host-weights
# forward (per-call marshal+compute) vs the resident-weights fast forward (compute
# only), on the Qwen3.6-27B vision tower. Timing only; correctness is proven by
# test_qwen3_5_vl_e2e (the host overload now delegates to the resident forward).
vllm_cpp_add_test(bench_qwen3_5_vl_tower vllm/multimodal/bench_qwen3_5_vl_tower.cpp)
target_compile_definitions(bench_qwen3_5_vl_tower PRIVATE
  MM_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl")
# Multimodal M3c VIDEO-processor UNIT gate: C++ Qwen3-VL video pipeline
# (pixel_values_videos/grid/timestamps/timestamp-interleaved expansion) BIT-identical
# to the vLLM 0.25.0 oracle fixture (tests/vllm/multimodal/fixtures/qwen3vl_video).
vllm_cpp_add_test(test_qwen3vl_video_processor vllm/multimodal/test_qwen3vl_video_processor.cpp)
target_compile_definitions(test_qwen3vl_video_processor PRIVATE
  MM_VIDEO_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl_video")
# Multimodal M3c STRICT e2e VIDEO->text token-exact gate on Qwen3-VL-4B: the full
# pipeline (video processor -> tower per-frame windowed attn -> merge -> forked
# video MRoPE/DeepStack greedy decode) vs the committed vLLM 0.25.0 golden. GPU +
# Qwen3-VL-4B checkpoint required, else skipped (VLLM_QWEN3VL_CKPT or HF cache).
vllm_cpp_add_test(test_qwen3vl_video_e2e vllm/multimodal/test_qwen3vl_video_e2e.cpp)
target_compile_definitions(test_qwen3vl_video_e2e PRIVATE
  MM_VIDEO_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl_video")
# Audio-track A1 audio-processor parity gate: the C++ Whisper-class audio INPUT
# pipeline (WAV decode -> log-mel input_features -> placeholder expansion -> mm-hash)
# vs the vLLM 0.25.0 / transformers 5.13.1 oracle fixture for openai/whisper-small
# (tests/vllm/multimodal/fixtures/whisper_audio; scripts/mm/a0_audio_ref.py).
# Log-mel rel-L2 within a stated tolerance (FFT vs DFT summation order); ids + mm-hash
# bit/byte-exact. RED-first: perturbed mel/hop/normalization blow the tolerance.
vllm_cpp_add_test(test_audio_processor vllm/multimodal/test_audio_processor.cpp)
target_compile_definitions(test_audio_processor PRIVATE
  MM_AUDIO_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/whisper_audio")
# P4 (MODEL-AUDIO-PARAKEET-ENCODER) — the Parakeet log-mel FRONT END. Gates the
# mel filter bank against a dump of transformers `audio_utils.mel_filter_bank`
# (the exact function vLLM's ParakeetExtractor calls) AND against librosa's
# float32 bank (what HF's own ParakeetFeatureExtractor uses), then the full
# `input_features` against a dump of ParakeetFeatureExtractor.__call__ on a fixed
# waveform. Fixture COMMITTED (228 KiB), so it runs in CI with no download
# (scripts/mm/p4_parakeet_extractor_ref.py). CPU-only.
vllm_cpp_add_test(test_parakeet_audio_processor
  vllm/multimodal/test_parakeet_audio_processor.cpp)
target_compile_definitions(test_parakeet_audio_processor PRIVATE
  MM_PARAKEET_AUDIO_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/parakeet_audio")
# P4 — the ParakeetEncoder/ParakeetForCTC MODULE gate: the forward vs an
# INDEPENDENT in-test scalar reference written from the transformers definition
# (and performing `_rel_shift` LITERALLY, so agreement is evidence for
# vt::AttentionRelPos's closed form). No fixture, no checkpoint, CPU-only.
vllm_cpp_add_test(test_parakeet_encoder vllm/models/test_parakeet_encoder.cpp)
# P4 — the e2e gate: the whole encoder + CTC head + greedy collapse vs a dumped
# HuggingFace `ParakeetForCTC` ORACLE (transformers 5.3.0 — the module vLLM
# itself runs). Stage-by-stage rel-L2 plus EXACT greedy and collapsed token ids.
# The oracle is a small SEEDED, RANDOMLY-INITIALISED model, not a pretrained
# checkpoint, so the gate is on logits and ids and never on a transcript; the
# pretrained-checkpoint arm is the separate VLLM_PARAKEET_CKPT case, skipped
# when unset. Fixture COMMITTED (464 KiB, scripts/mm/p4_parakeet_oracle_dump.py).
vllm_cpp_add_test(test_parakeet_ctc_engine vllm/models/test_parakeet_ctc_engine.cpp)
target_compile_definitions(test_parakeet_ctc_engine PRIVATE
  PARAKEET_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/parakeet")

# ARCH-ONE-SURFACE ROW 1 fold gate: the library transcription seam
# (ParakeetTranscriber) byte-identical to the PRE-refactor example pipeline
# AND to the committed goldens captured from the pre-refactor binary; plus the
# registry refuse-by-task contract. Fixture committed
# (scripts/mm/parakeet_e2e_fixture_gen.py).
vllm_cpp_add_test(test_parakeet_transcription_fold
  vllm/models/test_parakeet_transcription_fold.cpp)
target_compile_definitions(test_parakeet_transcription_fold PRIVATE
  PARAKEET_E2E_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/parakeet_e2e")
# P5: the RNN-T / TDT TRANSDUCER gate: the LSTM prediction network, the joint
# (with the TDT duration head) and the greedy transducer loop vs a dumped
# HuggingFace `ParakeetForRNNT` / `ParakeetForTDT` ORACLE from transformers
# `main`: which DOES implement them, contrary to what the P4 spike recorded off
# the locally installed 5.3.0. Decoder and joint tensors by rel-L2; the emitted
# sequence and the per-step durations EXACT. Also gates the LSTM cell against an
# independent in-test scalar reference. The oracle is a small SEEDED,
# RANDOMLY-INITIALISED model, so nothing is claimed about a transcript; the
# pretrained arm (VLLM_PARAKEET_TRANSDUCER_CKPT + _REF) gates ids against a real
# `generate()` run and is skipped when unset. Fixture COMMITTED (1.1 MiB,
# scripts/mm/p5_parakeet_transducer_oracle_dump.py). CPU-only.
vllm_cpp_add_test(test_parakeet_transducer vllm/models/test_parakeet_transducer.cpp)
target_compile_definitions(test_parakeet_transducer PRIVATE
  PARAKEET_TRANSDUCER_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/parakeet_transducer")
# M3d — NEAR-TIE-ROBUST e2e VIDEO->text token gate on the Qwen3.6-27B GDN-hybrid
# gate model: the full pipeline (video processor -> tower per-frame windowed attn,
# 27B vision config -> merge -> forked GDN-hybrid video MRoPE greedy decode) vs the
# committed vLLM 0.25.0 golden. GPU + Qwen/Qwen3.6-27B checkpoint required, else
# skipped (VLLM_QWEN36_CKPT or the HF cache path).
vllm_cpp_add_test(test_qwen3_5_vl_video_e2e vllm/multimodal/test_qwen3_5_vl_video_e2e.cpp)
target_compile_definitions(test_qwen3_5_vl_video_e2e PRIVATE
  MM_VIDEO_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3_5_27b_video")
# Audio-track A2 Whisper encoder-tower faithfulness gate: the C++
# WhisperAudioEncoderForward (2x Conv1d im2col+MatmulBT + sinusoidal pos + 12
# pre-norm bidirectional encoder blocks + final layer_norm) vs the dumped
# transformers-5.13.1 WhisperEncoder reference (per-stage post_conv/post_pos/
# block0/final_ln within the bf16-depth envelope). Needs a CUDA backend + the
# dumped encoder.* weights (VLLM_WHISPER_ENC_WEIGHTS); else skipped, not failed
# (scripts/mm/a2_audio_encoder_{ref,weight}_dump.py; embed_positions committed).
vllm_cpp_add_test(test_whisper_audio vllm/multimodal/test_whisper_audio.cpp)
target_compile_definitions(test_whisper_audio PRIVATE
  MM_AUDIO_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/whisper_audio")
# A3 — e2e AUDIO->TEXT on Voxtral-Mini-3B. GPU-gated: runs only with CUDA + the
# downloaded consolidated.safetensors (VLLM_VOXTRAL_SAFETENSORS); else skipped, not
# failed. Fixtures (WAV, log-mel/mel/sinusoid goldens, prompt+golden tokens) are
# committed (scripts/mm/a3_voxtral_oracle_capture.py).
vllm_cpp_add_test(test_voxtral_e2e vllm/multimodal/test_voxtral_e2e.cpp)
target_compile_definitions(test_voxtral_e2e PRIVATE
  MM_VOXTRAL_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/voxtral_audio")
# Multimodal M1 encoder-cache manager (inert-when-off engine seam).
vllm_cpp_add_test(test_encoder_cache_manager vllm/v1/core/test_encoder_cache_manager.cpp)
# LMCache MODE-1 (lm://) blocking TCP client — KV-EXTERNAL-CACHE W2. Round-trip
# vs an in-process C++ mock server (always), and vs a REAL lmcache.v1.server
# when VT_LMCACHE_LIVE_HOST/PORT are set (scripts/lmcache/run_live_roundtrip.sh).
vllm_cpp_add_test(test_lmcache_client vllm/v1/kv_offload/lmcache/test_lmcache_client.cpp)
# LMCache lm:// KVConnector subclass over the W5 seam — KV-EXTERNAL-CACHE W3.
# Connector-level round-trip (store->lookup->shortcut->load-identical) through
# the REAL scheduler vs an in-process mock server (always), foreign-key refusal,
# default-off inertness; the live gate vs a real server under VT_LMCACHE_LIVE_*.
vllm_cpp_add_test(test_lmcache_connector vllm/v1/kv_offload/lmcache/test_lmcache_connector.cpp)
# LMCache lm:// KEY AGREEMENT vs a REAL Python vLLM+LMCache peer —
# KV-EXTERNAL-CACHE W4 crux gate. Byte-identical CacheEngineKey strings at
# chunk_size 256 with vLLM's own sha256_cbor hash; fixtures dumped from the real
# lmcache ChunkedTokenDatabase by scripts/lmcache/gen_key_agreement_fixtures.py.
vllm_cpp_add_test(test_lmcache_key_agreement vllm/v1/kv_offload/lmcache/test_lmcache_key_agreement.cpp)
target_compile_definitions(test_lmcache_key_agreement PRIVATE
  TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")
vllm_cpp_add_test(test_scheduler_config vllm/test_scheduler_config.cpp)
# vllm#49343 eagle-draft max_position_embeddings clamp (config/speculative.h).
vllm_cpp_add_test(test_speculative_draft_max_position_embeddings
  vllm/config/test_speculative_draft_max_position_embeddings.cpp)
# SPEC-DSPARK W1: the DSpark config slice (method acceptance, block-size floor,
# lookahead slots, draft identity).
vllm_cpp_add_test(test_speculative_dspark vllm/config/test_speculative_dspark.cpp)
# SPEC-DSPARK-QWEN3-ROUTING (#1193): REACHABILITY of that classification — the
# loader's own ResolveSpecConfig reads the draft's config.json and refuses the
# DeepSeek-V4 DSpark lane by name, instead of trusting the CLI method string.
vllm_cpp_add_test(test_dspark_draft_routing
  vllm/entrypoints/test_dspark_draft_routing.cpp)
# SPEC-DFLASH2 W1 (#1314): REACHABILITY of the DFlash2 refusal — the loader
# reads the draft's own config.json and refuses `DFlash2DraftModel` by name,
# instead of drafting it through the DFlash1 lane with the grouped convolution
# and the candidate selector silently absent.
vllm_cpp_add_test(test_dflash2_draft_routing
  vllm/entrypoints/test_dflash2_draft_routing.cpp)
# #1160: `--speculative-config` key ADMISSION — an unknown or misspelled key is
# refused by name, and an upstream key this engine does not implement is refused
# with the missing part named, instead of being silently dropped.
vllm_cpp_add_test(test_speculative_unknown_keys
  vllm/config/test_speculative_unknown_keys.cpp)
# SPEC-DRAFTER-CHAIN W1 (#1522): the `vllm_cpp.drafter_chain` extension of
# --speculative-config -- its parsing, its validation and its refusals. The
# nesting under `vllm_cpp` is the same convention `--offload-config` already
# uses for this engine's non-vLLM keys, and it is what bounds the collision
# surface with vLLM's own `SpeculativeConfig` to exactly one name.
vllm_cpp_add_test(test_speculative_drafter_chain
  vllm/config/test_speculative_drafter_chain.cpp)
# SPEC-MTP-K-GT-1 (#81): the MTP speculation-DEPTH config contract — the k=1
# default, the upstream divisibility rule, and that a configured k above 1 is
# carried rather than silently degraded.
vllm_cpp_add_test(test_speculative_mtp_depth vllm/config/test_speculative_mtp_depth.cpp)
# ENG-MM-INPUT-PIPELINE L1 (#607): get_limit_per_prompt precedence —
# language_model_only ⇒ 0 BEFORE the map, else the map, else the 999 default.
vllm_cpp_add_test(test_multimodal_config vllm/config/test_multimodal_config.cpp)
vllm_cpp_add_test(test_offload_config vllm/config/test_offload_config.cpp)
# ENG-RESIDENCY-CONFIG (#1110): the `vllm_cpp` extension of --offload-config.
vllm_cpp_add_test(test_weight_residency_config
  vllm/config/test_weight_residency_config.cpp)

vllm_cpp_add_test(test_deepseek_v4_paged_equiv
  vllm/models/test_deepseek_v4_paged_equiv.cpp)
vllm_cpp_add_test(test_multi_kv_refusal
  vllm/model_executor/test_multi_kv_refusal.cpp)
vllm_cpp_add_test(test_device_token_ids_refusal
  vllm/model_executor/test_device_token_ids_refusal.cpp)
# ENG-MM-EMBED-DEVICE-IDS (#2730): the MULTIMODAL half of the same channel --
# `MmEmbedInputs` carries the device identifiers, both registered `embed_mm`
# hooks splice them over the buffer they gather from, and a hook that does not
# declare the capability is refused instead of embedding token id 0.
vllm_cpp_add_test(test_mm_embed_device_token_ids
  vllm/models/test_mm_embed_device_token_ids.cpp)
# The dots3-note case loads the tiny on-disk checkpoint through the REAL loader,
# for the same reason its own gates do: `Dots3NoteLoadedModel` lives in the
# anonymous namespace of its registry translation unit and cannot be borrowed.
target_compile_definitions(test_mm_embed_device_token_ids PRIVATE
  DOTS3_NOTE_CKPT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/dots3_note_prev")
vllm_cpp_add_test(test_device_placement
  vllm/model_executor/test_device_placement.cpp)

# ...and the expert-stream knob's own binary (#1122). `ResolveExpertStreamRequested`
# caches its answer in a function-local static, so a process can observe what it
# resolved exactly ONCE. That one observation is spent here, on the field the
# resolver reads — rewiring it to the adjacent `.mmap` left every other suite green.
vllm_cpp_add_test(test_expert_stream_latch
  vllm/config/test_expert_stream_latch.cpp)
# ...and its REACHABILITY: the loader and the C ABI install it. `vllm.h` is the
# public surface, so this suite needs src/ on the include path exactly as
# test_capi does (it reaches the internal MakeEngineHandle hook there; here it is
# the internal weight_residency header beside the public one).
vllm_cpp_add_test(test_weight_residency_reach
  vllm/entrypoints/test_weight_residency_reach.cpp)
target_include_directories(test_weight_residency_reach PRIVATE ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_placement_reach
  vllm/entrypoints/test_placement_reach.cpp)
target_include_directories(test_placement_reach PRIVATE ${CMAKE_SOURCE_DIR}/src)

# ISSUE-LOCAL-01M3SDXGYYTS9FDXKZDAE24N0R: the loader's tokenizer/ subdirectory
# fallback (MODEL-LAYA), entered through LoadedEngine::FromModelDir.
vllm_cpp_add_test(test_tokenizer_subdir
  vllm/entrypoints/test_tokenizer_subdir.cpp)

vllm_cpp_add_test(test_placement_dump_dtype
  vllm/model_executor/test_placement_dump_dtype.cpp)
target_include_directories(test_placement_dump_dtype PRIVATE ${CMAKE_SOURCE_DIR}/src)

# ENG-HYBRID-PLACEMENT W3i (#2714): the placement round trip, executed rather
# than reasoned about. ITS OWN BINARY, and for the same reason its siblings above
# are -- it registers a loopback backend and platform in the process-global kXPU
# slot to give the seam an engine that is not the CPU, and that registration must
# not leak into a suite whose subject is a different device.
vllm_cpp_add_test(test_placed_moe_roundtrip
  vllm/model_executor/test_placed_moe_roundtrip.cpp)
target_include_directories(test_placed_moe_roundtrip PRIVATE ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_act_dump
  vllm/model_executor/test_act_dump.cpp)
target_include_directories(test_act_dump PRIVATE ${CMAKE_SOURCE_DIR}/src)

# BACKEND-ROCM-BF16-MOE (#3116): the LM-head output boundary, replayed over the
# primary's own captured head input. The fixture is a directory of raw captures,
# so the two replay cases skip themselves (`doctest::skip`, reported skipped in
# doctest's own summary) and the binary still exits 77 (Skipped) without
# `VT_MOE_HEAD_FIXTURE` rather than reporting a green gate that never ran. The
# fixture-free call-site case in the same binary runs either way, and a failure
# in it keeps its own non-zero exit instead of being folded into the skip.
vllm_cpp_add_test(test_qwen3_moe_lm_head_bf16
  vllm/model_executor/test_qwen3_moe_lm_head_bf16.cpp)

vllm_cpp_add_test(test_moe_fit_resolver
  vllm/model_executor/test_moe_fit_resolver.cpp)
target_include_directories(test_moe_fit_resolver PRIVATE ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_attention_window_adoption
  vllm/model_executor/layers/attention/test_attention_window_adoption.cpp)

# ENG-ATTENTION-WINDOW W3 (#2388): the kill switch itself, which had no test in
# either of the two environment spellings it replaced.
vllm_cpp_add_test(test_disable_sliding_window
  vllm/model_executor/layers/attention/test_disable_sliding_window.cpp)
target_include_directories(test_attention_window_adoption PRIVATE ${CMAKE_SOURCE_DIR}/src)
# `vllm-cli`'s `--offload-config` (#1135). It runs the BINARY, because that is the
# only thing that proves the FLAG arrives: `test_weight_residency_reach` already
# drives `vllm_engine_load` with the same JSON string, so a test that called the
# ABI here would re-prove the ABI. `$<TARGET_FILE:vllm-cli>` resolves at generate
# time, after every directory is processed; the build ORDER dependency is declared
# in the top-level CMakeLists, where the `vllm-cli` target already exists.
# Registered UNCONDITIONALLY, and with examples off it leaves `VLLM_CLI_BINARY`
# undefined: the suite then exits 77 and CTest reports it Skipped, which is what
# `SKIP_RETURN_CODE` above is for. Registering it only under the option would hide
# the gate's absence instead of reporting it.
vllm_cpp_add_test(test_cli_offload_config
  vllm/entrypoints/test_cli_offload_config.cpp)
if(TARGET vllm-cli OR VLLM_CPP_BUILD_EXAMPLES)
  target_compile_definitions(test_cli_offload_config
    PRIVATE VLLM_CLI_BINARY="$<TARGET_FILE:vllm-cli>")
endif()
vllm_cpp_add_test(test_cli_kv_cache_dtype
  vllm/entrypoints/test_cli_kv_cache_dtype.cpp)
if(TARGET vllm-cli OR VLLM_CPP_BUILD_EXAMPLES)
  target_compile_definitions(test_cli_kv_cache_dtype
    PRIVATE VLLM_CLI_BINARY="$<TARGET_FILE:vllm-cli>")
endif()
vllm_cpp_add_test(test_weight_offloader vllm/model_executor/test_weight_offloader.cpp)
vllm_cpp_add_test(test_weight_offload_policy vllm/model_executor/test_weight_offload_policy.cpp)
vllm_cpp_add_test(test_expert_slot_cache vllm/model_executor/test_expert_slot_cache.cpp)
vllm_cpp_add_test(test_gguf_expert_span vllm/model_executor/test_gguf_expert_span.cpp)
vllm_cpp_add_test(test_expert_streamer vllm/model_executor/test_expert_streamer.cpp)
vllm_cpp_add_test(test_host_expert_slot_store vllm/model_executor/test_host_expert_slot_store.cpp)
# ENG-EXPERT-STREAM-DEVICE W1 (#1124), gate G1: the DEVICE slot store and the
# fill contract that makes it fillable. It needs no separate binary -- it
# registers nothing process-global, reads no environment latch, and drives its
# own `vt::Backend` instances by hand -- but it is its own file because the
# oracle it compares against is `HostExpertSlotStore`, so a suite that IS the
# host store cannot also be the comparison.
vllm_cpp_add_test(test_device_expert_slot_store
  vllm/model_executor/test_device_expert_slot_store.cpp)
# ENG-EXPERT-STREAM F3 (#912): the REACHABILITY gate, in its own binary because
# VT_MOE_EXPERT_STREAM is read ONCE into a function-local static, so one process
# cannot run both the streaming and non-streaming arms from the environment.
vllm_cpp_add_test(test_expert_stream_wiring vllm/model_executor/test_expert_stream_wiring.cpp)
# The lane's observability seam lives in the internal header beside the model,
# the same arrangement test_model_registry uses.
target_include_directories(test_expert_stream_wiring PRIVATE ${CMAKE_SOURCE_DIR}/src)
# ENG-EXPERT-STREAM F7 (#912): its own binary too, and specifically one that does
# NOT set VT_MOE_EXPERT_STREAM_SLOT_BYTES -- that override is what would hide the
# defect, and the store is built from the FIRST model a process runs.
vllm_cpp_add_test(test_expert_stream_mixed_slot vllm/model_executor/test_expert_stream_mixed_slot.cpp)
target_include_directories(test_expert_stream_mixed_slot PRIVATE ${CMAKE_SOURCE_DIR}/src)
# ENG-EXPERT-STREAM-DEVICE W0c (#1124): the PLATFORM branch of the expert-slice
# seam. Its own binary for two independent reasons -- it latches
# VT_MOE_EXPERT_STREAM like its siblings, and it registers a fake backend and a
# fake platform in the process-global kXPU slot, which must not leak into any
# other suite.
vllm_cpp_add_test(test_expert_stream_device_slot vllm/model_executor/test_expert_stream_device_slot.cpp)
target_include_directories(test_expert_stream_device_slot PRIVATE ${CMAKE_SOURCE_DIR}/src)
# ENG-EXPERT-STREAM-DEVICE W0f (#1299): which branch `ResidentWeight` takes on a
# staging platform whose kernels can dereference host storage. Its own binary for
# the same reason as its sibling above -- it registers a fake backend and a fake
# platform in the process-global kXPU slot, which must not leak into any other
# suite -- and separately because the two disagree about the backend's
# UnifiedMemory(), which is a process-global registration and not a per-case flag.
vllm_cpp_add_test(test_resident_weight_host_addressable vllm/model_executor/test_resident_weight_host_addressable.cpp)
target_include_directories(test_resident_weight_host_addressable PRIVATE ${CMAKE_SOURCE_DIR}/src)
# ENG-EXPERT-STREAM-DEVICE (#2711): does `ResidentWeightF32` still hold its copy
# source when the copy runs? Its own binary for the same reason as the sibling
# above -- it registers a fake backend and a fake platform in the process-global
# kXPU slot -- and separately because its backend DEFERS every `Copy` until a
# `Synchronize`, which is the opposite of the sibling's eager `memcpy` and would
# change what every case in that file measures.
vllm_cpp_add_test(test_resident_weight_f32_copy_retires vllm/model_executor/test_resident_weight_f32_copy_retires.cpp)
target_include_directories(test_resident_weight_f32_copy_retires PRIVATE ${CMAKE_SOURCE_DIR}/src)
# ENG-EXPERT-STREAM (#1091): the step clock at the MoE entry points ForwardLayers
# does NOT cover, and the final statistics line. Its own binary because it is the
# only one that can observe `steps == 0` -- the store is a process-lifetime
# singleton, so a case asking that question must run before anything ends a step.
vllm_cpp_add_test(test_expert_stream_steps vllm/model_executor/test_expert_stream_steps.cpp)
target_include_directories(test_expert_stream_steps PRIVATE ${CMAKE_SOURCE_DIR}/src)
# MODEL-TEXT-GLM-MOE-DSA W3 (#2214): G3 on a model with MORE THAN 4 experts and
# MORE THAN 4 layers, which is the bound `expert-streaming.md` `## Owed` records
# against every fixture this row inherited. Its own binary because the wider
# model needs a larger VT_MOE_EXPERT_STREAM_SLOTS than test_expert_stream_wiring
# sets, and that knob is read once into a function-local static behind a
# process-lifetime singleton.
vllm_cpp_add_test(test_expert_stream_wide_logits
  vllm/model_executor/test_expert_stream_wide_logits.cpp)
target_include_directories(test_expert_stream_wide_logits PRIVATE ${CMAKE_SOURCE_DIR}/src)
# MODEL-TEXT-GLM-MOE-DSA W3 (#2214): the LOAD-TIME slot-capacity refusal. Pure
# arithmetic over the shared seam, so it needs neither a forward nor the
# environment latch its siblings above are separated by.
vllm_cpp_add_test(test_expert_stream_capacity
  vllm/model_executor/test_expert_stream_capacity.cpp)
# ENG-MM-INPUT-PIPELINE L1 (#607): the refusal those limits carry — the min()
# fold against the model's own ceiling, upstream's exact message, and both call
# sites with the enable_mm_embeds escape.
vllm_cpp_add_test(test_processing_limits vllm/multimodal/test_processing_limits.cpp)
vllm_cpp_add_test(test_request_queue vllm/v1/test_request_queue.cpp)
vllm_cpp_add_test(test_sched_output vllm/v1/test_sched_output.cpp)
vllm_cpp_add_test(test_scheduler vllm/v1/test_scheduler.cpp)
vllm_cpp_add_test(test_scheduler_lpm vllm/v1/test_scheduler_lpm.cpp)
vllm_cpp_add_test(test_async_scheduler vllm/v1/test_async_scheduler.cpp)
vllm_cpp_add_test(test_scheduler_wave vllm/v1/test_scheduler_wave.cpp)
vllm_cpp_add_test(test_block_table vllm/v1/worker/test_block_table.cpp)
vllm_cpp_add_test(test_input_batch vllm/v1/worker/test_input_batch.cpp)
vllm_cpp_add_test(test_sampling_metadata vllm/v1/sample/test_metadata.cpp)
vllm_cpp_add_test(test_logits_processors vllm/v1/sample/test_logits_processors.cpp)
vllm_cpp_add_test(test_sampler vllm/v1/sample/test_sampler.cpp)
vllm_cpp_add_test(test_host_buffer_staging vllm/v1/sample/test_host_buffer_staging.cpp)
vllm_cpp_add_test(test_logits_processor_host_addressable
  vllm/v1/sample/test_logits_processor_host_addressable.cpp)
vllm_cpp_add_test(test_prepare_inputs vllm/v1/worker/test_prepare_inputs.cpp)
vllm_cpp_add_test(test_combine_tokens vllm/v1/worker/test_combine_tokens.cpp)
vllm_cpp_add_test(test_combine_row_predicate vllm/v1/worker/test_combine_row_predicate.cpp)
# SPEC-DFLASH2 A2-2 (#2802): the spec-decode VERIFY route predicate, and the
# per-STEP reduction both sampling entry points and the input-combine refusal
# turn on. Same shape and same reason as the row predicate above.
vllm_cpp_add_test(test_verify_route_predicate vllm/v1/worker/test_verify_route_predicate.cpp)
# SPEC-DFLASH2 A2-3 (#2911): the async placeholder fill's per-request rule, read
# off the same per-req_state draft buffer the combine's scatter reads. Same shape
# and same reason as the two predicates above -- one rule, one expression.
vllm_cpp_add_test(test_draft_fill vllm/v1/worker/test_draft_fill.cpp)
# SPEC-DFLASH2 A2-4 (#3004): the async decode arm's propose reads this step's
# committed ids where they live, on every write-back branch, instead of refusing
# by name when the branch that ran left them unreadable on the host.
vllm_cpp_add_test(test_committed_ids vllm/v1/worker/test_committed_ids.cpp)
vllm_cpp_add_test(test_async_output vllm/v1/worker/test_async_output.cpp)
vllm_cpp_add_test(test_async_runner_flag vllm/v1/worker/test_async_runner_flag.cpp)
# ENG-POOL-BEST-FIT (#1922): the shared scratch pool must reach a steady state
# across sequential requests driven through the production LoadedEngine entry.
vllm_cpp_add_test(test_engine_scratch_steady_state
  vllm/v1/worker/test_engine_scratch_steady_state.cpp)
# ENG-POOL-BEST-FIT (#1922): this gate RUNS in the sanitizer lanes, with the
# pool switched back on. It is not skipped there, and that distinction cost a
# measurement to establish.
#
# `sanitize-cpu` sets `VT_POOL_BYPASS=1` for the whole suite (ci.yml), under
# which `Get` is a bare `Alloc`, `Put` a bare `Free`, and `stats()` reads all
# zeroes -- so every case in this file asserted `CHECK( 0 == 1 )` against a pool
# that was not there, and BOTH sanitizer lanes were red on run 32889401375.
#
# The first repair was to skip. That was wrong: a skip guard turns a red into a
# green that measured nothing, and the pool's steady-state behaviour would have
# been silently unexercised under ASan and TSan forever. The right repair is to
# supply the missing precondition, which is what this line does.
#
# What made it possible: the case now DRAINS the pool after its last assertion.
# Without that, running it pooled under `detect_leaks=1` reports
# `3362944 byte(s) leaked in 103 allocation(s)` -- correctly, because the pool
# never returns a block to the driver by design. Measured: pooled + drained is
# `EXIT=0`, 19/19 `SUCCESS`, `drained 624995 B`.
#
# The cost, stated rather than left implicit: for THIS ONE BINARY, ASan can no
# longer see a use-after-free of a released `DBuf`, because a pooled block stays
# mapped. That is the trade `device_pool.h` documents under its BYPASS lane, it
# applies to no other binary in the suite, and the binary in question exists to
# measure the pool.
set_tests_properties(test_engine_scratch_steady_state PROPERTIES
  ENVIRONMENT "VT_POOL_BYPASS=0")
vllm_cpp_add_test(test_runner vllm/v1/worker/test_runner.cpp)
# test_runner uses detail::ValidateGdnStateIndices from the internal header.
target_include_directories(test_runner PRIVATE ${CMAKE_SOURCE_DIR}/src)
# #2534. VT_DUMP_LOGITS is read ONCE at static init, so one process observes one
# value and the default run can only ever exercise the INERT arm. Re-run the SAME
# binary with the instrument pointed at a per-run scratch directory, the shape
# `test_qwen35_paged_forward_gdn_out_f32` and `_glue_fuse_off` already use.
#
# SCOPED to the one case, because the instrument writes files keyed by request id
# and the other cases in this binary reuse those ids -- an unscoped run would
# append several requests' rows into one file and make the size assertion
# meaningless. `--test-case` is safe here only because the name is COMMA-FREE:
# doctest splits that filter on commas, and a name containing one would match
# nothing, exit 0, and pass while testing zero cases.
#
# The two properties are what stop that mute pass. FAIL_REGULAR_EXPRESSION takes
# precedence in ctest, so a run that matched no case (`assertions: 0`) FAILS
# rather than reporting success; PASS_REGULAR_EXPRESSION then requires doctest's
# own verdict instead of trusting the exit status.
add_test(NAME test_runner_logit_dump
  COMMAND ${CMAKE_COMMAND}
    -E env VT_DUMP_LOGITS=${CMAKE_CURRENT_BINARY_DIR}/logit_dump_scratch
    $<TARGET_FILE:test_runner> "--test-case=runner: the logit dump*")
set_tests_properties(test_runner_logit_dump PROPERTIES
  FAIL_REGULAR_EXPRESSION "assertions:[ ]*0[ |]"
  PASS_REGULAR_EXPRESSION "Status: SUCCESS!")
# The scratch directory must EXIST and be EMPTY before that entry runs, or a
# stale .f32 from a previous run would be appended to and the size assertion
# would pass on the wrong bytes.
add_test(NAME test_runner_logit_dump_scratch_reset
  COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/logit_dump_scratch)
set_tests_properties(test_runner_logit_dump_scratch_reset PROPERTIES
  FIXTURES_SETUP logit_dump_scratch)
# ...and RECREATE it. The instrument deliberately does not mkdir -- it refuses by
# name instead -- so without this the entry would red on a missing directory
# rather than on anything it is meant to measure.
add_test(NAME test_runner_logit_dump_scratch_make
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/logit_dump_scratch)
set_tests_properties(test_runner_logit_dump_scratch_make PROPERTIES
  FIXTURES_SETUP logit_dump_scratch
  DEPENDS test_runner_logit_dump_scratch_reset)
set_tests_properties(test_runner_logit_dump PROPERTIES
  FIXTURES_REQUIRED logit_dump_scratch)
vllm_cpp_add_test(test_common_attn_metadata vllm/v1/attention/test_common_attn_metadata.cpp)
vllm_cpp_add_test(test_attn_backend_registry vllm/v1/attention/test_attn_backend_registry.cpp)
vllm_cpp_add_test(test_attn_validate_configuration
  vllm/v1/attention/test_attn_validate_configuration.cpp)
# SPEC-DFLASH2 W10 (#1857): the spec-as-decode reorder-threshold policy mirror.
vllm_cpp_add_test(test_spec_as_decode vllm/v1/attention/test_spec_as_decode.cpp)
vllm_cpp_add_test(test_attention_window vllm/model_executor/layers/attention/test_attention.cpp)
vllm_cpp_add_test(test_chunked_local_attention
  vllm/v1/attention/test_chunked_local_attention.cpp)
vllm_cpp_add_test(test_rotary_embedding
  vllm/model_executor/layers/rotary_embedding/test_rotary_embedding.cpp)
vllm_cpp_add_test(test_linear_scaling_rope
  vllm/model_executor/layers/rotary_embedding/test_linear_scaling_rope.cpp)
vllm_cpp_add_test(test_pooler
  vllm/model_executor/layers/pooler/test_pooler.cpp)
vllm_cpp_add_test(test_pooler_heads
  vllm/model_executor/layers/pooler/test_pooler_heads.cpp)
vllm_cpp_add_test(test_pooling_runner
  vllm/v1/worker/gpu/pool/test_pooling_runner.cpp)
vllm_cpp_add_test(test_gdn_metadata_builder vllm/v1/attention/test_gdn_metadata_builder.cpp)
vllm_cpp_add_test(test_mtp_speculator vllm/v1/spec_decode/test_mtp_speculator.cpp)
# SPEC-MTP-K-GT-1 (#81): MTP speculation DEPTH through the production loader —
# the configured k reaches the propose path, or is refused by name.
vllm_cpp_add_test(test_mtp_depth vllm/v1/spec_decode/test_mtp_depth.cpp)
# ENG-CUDAGRAPH-BREAK W6 (#1374): the graph-eligibility predicate and its
# counters live under src/, beside the runner that computes them, exactly as
# test_cudagraph_dispatch reads them (:561).
target_include_directories(test_mtp_depth PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_rejection_sampler vllm/v1/spec_decode/test_rejection_sampler.cpp)
vllm_cpp_add_test(test_prepare_prefill_inputs vllm/v1/spec_decode/test_prepare_prefill_inputs.cpp)
# SPEC-MTP-K-GT-1 (#81): the draft-DECODE half of the autoregressive propose —
# the step-1 entry state, the between-step advance, and both max_model_len clamps.
vllm_cpp_add_test(test_prepare_decode_inputs vllm/v1/spec_decode/test_prepare_decode_inputs.cpp)
vllm_cpp_add_test(test_dflash_kvprep vllm/v1/spec_decode/test_dflash_kvprep.cpp)
target_include_directories(test_dflash_kvprep PRIVATE ${CMAKE_SOURCE_DIR}/src)
# SPEC-DFLASH2 W1 (#1314): the DFlash causality rule — the port of vLLM's
# tests/v1/spec_decode/test_dflash_causality.py as edited by vllm#52816, plus the
# assertion that a DFlash1 checkpoint's resolution is unchanged.
vllm_cpp_add_test(test_dflash_causality vllm/v1/spec_decode/test_dflash_causality.cpp)
vllm_cpp_add_test(test_dflash_propose vllm/v1/spec_decode/test_dflash_propose.cpp)
# SPEC-DFLASH2 W4 (#1314): the INVERSE guard, once the walk lands and there is
# no missing mechanism left to refuse -- a DFlash2 block that reaches the DFlash1
# per-slot argmax throws by name, and the walk's own refusals are gated beside
# it. W1/W2/W3 read "the candidate-selector refusal" here, which was the boundary
# the architecture was left at while its selector was unported.
vllm_cpp_add_test(test_dflash2_argmax_guard
  vllm/v1/spec_decode/test_dflash2_argmax_guard.cpp)
# SPEC-DFLASH2 W4 (#1314): reachability (spec G5) -- a real LoadedEngine through
# GPUModelRunner::propose_drafts_block, with the drafts read off the production
# trace.
vllm_cpp_add_test(test_dflash2_runner_reach
  vllm/v1/spec_decode/test_dflash2_runner_reach.cpp)
# W10 (#1857): the spec-as-decode classification case reads the graph-dispatch
# counters, whose header lives in the src tree (same need as test_mtp_depth).
target_include_directories(test_dflash2_runner_reach PRIVATE ${CMAKE_SOURCE_DIR}/src)
# MODEL-DFLASH2-EXL3 (#2495 item 7): the same engine entry, with an EXL3 draft.
# Its OWN binary because the draft store is a function-local static per arm and
# the target it needs is wider than every other case in the shared fixture.
vllm_cpp_add_test(test_dflash2_exl3_reach
  vllm/v1/spec_decode/test_dflash2_exl3_reach.cpp)
target_include_directories(test_dflash2_exl3_reach PRIVATE ${CMAKE_SOURCE_DIR}/src)
# SPEC-DFLASH2 W9 (#1849): the VT_SPEC_TRACE=2 device-segment trace of the
# DFlash2 draft phase. Its OWN binary because the trace level is latched by a
# function-local static on the first propose, so one process observes one level
# and the reach binary above latches "1".
vllm_cpp_add_test(test_dflash2_draft_phase_trace
  vllm/v1/spec_decode/test_dflash2_draft_phase_trace.cpp)
# SPEC-DFLASH2 (#1946): the device-byte count for the shared embedding table.
# Its OWN binary, and NOT merged into the reach binary below, because it
# registers a fake platform + backend in the kXPU slot to get a countable device
# allocation on a CPU tier -- and the platform registry is process-global, so a
# binary that also builds a LoadedEngine would be resolving its device against a
# registration that exists only for this measurement.
vllm_cpp_add_test(test_dflash2_embed_dedup
  vllm/v1/spec_decode/test_dflash2_embed_dedup.cpp)
target_include_directories(test_dflash2_embed_dedup PRIVATE ${CMAKE_SOURCE_DIR}/src)
# SPEC-DFLASH2 (#1946): the production reachability half -- a real LoadedEngine,
# no fake platform, reading the rebind off the engine's own stderr.
vllm_cpp_add_test(test_dflash2_embed_dedup_reach
  vllm/v1/spec_decode/test_dflash2_embed_dedup_reach.cpp)
target_include_directories(test_dflash2_embed_dedup_reach PRIVATE ${CMAKE_SOURCE_DIR}/src)
# SPEC-DFLASH2 (#1919): the draft context store's capacity, through the
# production async front — a >4K-token prompt must not kill EngineCore, and a
# request that outgrows the store falls back to the non-speculative path. Its
# OWN binary for the same reason the two above have theirs: `VT_SPEC_TRACE` is
# latched once per process, and this one needs level 1 with a `max_model_len`
# above 4096, which no other binary drives.
vllm_cpp_add_test(test_dflash2_ctx_capacity
  vllm/v1/spec_decode/test_dflash2_ctx_capacity.cpp)
# FIX-BLOCK-TABLE-ROW-WIDTH (ISSUE-LOCAL-01M36YFXAHPXMCFAGWQABT6KCE): each KV
# cache group's block-table row is sized from its spec, through the production
# DFlash2 engine. Its own binary because the fixture header is per-binary.
vllm_cpp_add_test(test_block_table_row_width
  vllm/v1/worker/test_block_table_row_width.cpp)
# SPEC-DFLASH2 (#2008): the draft context must follow the REQUEST, not the batch
# ROW. Its OWN binary for the reason every fixture binary here has one --
# `VT_SPEC_TRACE` is latched once per process by a function-local static on the
# first propose, and this one needs level 1 with `max_num_seqs` above 1, which no
# other binary drives.
vllm_cpp_add_test(test_dflash2_concurrency
  vllm/v1/spec_decode/test_dflash2_concurrency.cpp)
# SPEC-DFLASH2 (#2042) — prefix caching and the DFlash2 draft context: a request
# admitted on a prefix-cache hit starts mid-sequence with an empty draft store,
# and must fall back to the target instead of taking EngineCore down. Its OWN
# binary for the same latch reason as the others, and because it is the only one
# that drives the engine with `enable_prefix_caching` on.
vllm_cpp_add_test(test_dflash2_prefix_cache
  vllm/v1/spec_decode/test_dflash2_prefix_cache.cpp)
target_include_directories(test_dflash_propose PRIVATE ${CMAKE_SOURCE_DIR}/src)
# SPEC-NGRAM (ROAD-V1-D3) — the draft-free n-gram matcher unit gate (ports
# vllm/tests/v1/spec_decode/test_ngram.py, host-side, runs everywhere).
vllm_cpp_add_test(test_ngram_proposer vllm/v1/spec_decode/test_ngram_proposer.cpp)
target_include_directories(test_ngram_proposer PRIVATE ${CMAKE_SOURCE_DIR}/src)
# SPEC-DRAFT-MODEL (CLAIM-SPEC-DRAFT-MEDUSA) — the generic separate-draft-model
# greedy propose brick; host-side, reuses the landed RejectionSampler verify.
vllm_cpp_add_test(test_draft_model_proposer
  vllm/v1/spec_decode/test_draft_model_proposer.cpp)
target_include_directories(test_draft_model_proposer PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_structured_output vllm/v1/structured_output/test_structured_output.cpp)
vllm_cpp_add_test(test_apply_grammar_bitmask vllm/v1/structured_output/test_apply_grammar_bitmask.cpp)
vllm_cpp_add_test(test_backend_native vllm/v1/structured_output/test_backend_native.cpp)
vllm_cpp_add_test(test_backend_xgrammar vllm/v1/structured_output/test_backend_xgrammar.cpp)
vllm_cpp_add_test(test_jump_forward vllm/v1/structured_output/test_jump_forward.cpp)
vllm_cpp_add_test(test_json_schema_to_gbnf vllm/v1/structured_output/test_json_schema_to_gbnf.cpp)
vllm_cpp_add_test(test_response_format_e2e vllm/v1/structured_output/test_response_format_e2e.cpp)
vllm_cpp_add_test(test_engine_types vllm/test_engine_types.cpp)
vllm_cpp_add_test(test_engine_core vllm/v1/test_engine_core.cpp)
vllm_cpp_add_test(test_engine_core_proc vllm/v1/test_engine_core_proc.cpp)
vllm_cpp_add_test(test_async_admission_timing vllm/v1/test_async_admission_timing.cpp)
vllm_cpp_add_test(test_input_processor vllm/v1/test_input_processor.cpp)
target_compile_definitions(test_input_processor PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
vllm_cpp_add_test(test_output_processor vllm/v1/test_output_processor.cpp)
vllm_cpp_add_test(test_async_llm vllm/v1/test_async_llm.cpp)
vllm_cpp_add_test(test_llm_engine vllm/v1/test_llm_engine.cpp)
vllm_cpp_add_test(test_loaded_engine_dense vllm/entrypoints/test_loaded_engine_dense.cpp)
# FIX-KV-GROUP-LAYER-COUNT (#1963, #1966): the KV-budget cases drive
# MakeQwen3_5KVCacheSpec, the shipping registry helper, whose header is private
# to src/. A hand-built KVCacheGroupSpec is a shape no registry emits and cannot
# see the defect, which is exactly why the pre-existing unit coverage missed it.
target_include_directories(test_loaded_engine_dense PRIVATE ${CMAKE_SOURCE_DIR}/src)
# FIX-KV-POOL-MIN-FIT (ISSUE-LOCAL-01M36QVG0KGKEP4MSMKT18MMZ4): the startup
# check that the KV pool holds one max_model_len request counts EVERY KV cache
# group and the null block. Entered through the production LoadedEngine
# constructor with the DFlash2 fixture, because the defect needs the
# three-group (fa, gdn, fa_draft) speculative shape to be seen.
vllm_cpp_add_test(test_kv_pool_min_fit vllm/entrypoints/test_kv_pool_min_fit.cpp)
# KV-FP8 W3 (#1593): half-sized KV blocks, --kv-cache-dtype threading and the
# checkpoint k/v-scale path, entered through the LoadedEngine loader.
vllm_cpp_add_test(test_kv_cache_fp8_wiring vllm/entrypoints/test_kv_cache_fp8_wiring.cpp)
vllm_cpp_add_test(test_dspark_block_size_guard vllm/entrypoints/test_dspark_block_size_guard.cpp)
# ENG-HF-MODEL-DOWNLOAD W2 (#1280): the HuggingFace cache walk entered through
# the production loader, with a repository identifier rather than a path. The
# transformers_utils suite calls the function directly and cannot see a deleted
# call site; this one can. Named `dflash` so `ctest -R "dflash|dspark"` runs it.
vllm_cpp_add_test(test_dflash_draft_hf_cache vllm/entrypoints/test_dflash_draft_hf_cache.cpp)
# SPEC-DRAFTER-CHAIN W1 (#1522): the REACHABILITY and ORDERING half. W1 stores a
# field nothing resolves yet, so the one way it could land dead is a parsed chain
# no production caller reads. `ResolveSpecConfig` is that reader and it refuses by
# name; `FromModelDir` runs the refusal ahead of the path resolution, which is what
# G5's "before any weight I/O" means. Needs src/ on the include path for the same
# reason `test_weight_residency_reach` does: it drives the public `vllm.h` ABI.
vllm_cpp_add_test(test_drafter_chain_reach
  vllm/entrypoints/test_drafter_chain_reach.cpp)
target_include_directories(test_drafter_chain_reach PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_beam_search vllm/entrypoints/test_beam_search.cpp)
# ROAD-V1-MM serving W1: OpenAI multimodal content-part parse + decode + route
# to the existing single-sequence mm processors (CPU, no model weights). Reuses
# the M1 image (qwen3vl) + A1 audio (whisper_audio) processor-parity fixtures.
vllm_cpp_add_test(test_chat_mm vllm/entrypoints/openai/test_chat_mm.cpp)
target_compile_definitions(test_chat_mm PRIVATE
  MM_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl"
  MM_AUDIO_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/whisper_audio"
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
vllm_cpp_add_test(test_openai_protocol vllm/entrypoints/openai/test_protocol.cpp)
vllm_cpp_add_test(test_openai_serving vllm/entrypoints/openai/test_serving.cpp)
vllm_cpp_add_test(test_sse_keepalive vllm/entrypoints/openai/test_sse_keepalive.cpp)
vllm_cpp_add_test(test_openai_run_batch
  vllm/entrypoints/openai/test_run_batch.cpp)
vllm_cpp_add_test(test_openai_serving_chat_stream
  vllm/entrypoints/openai/test_serving_chat_stream.cpp)
vllm_cpp_add_test(test_openai_logprobs vllm/entrypoints/openai/test_logprobs.cpp)
vllm_cpp_add_test(test_openai_utf8 vllm/entrypoints/openai/test_utf8.cpp)
vllm_cpp_add_test(test_streaming_parser_engine
  vllm/parser/engine/test_streaming_parser_engine.cpp)
vllm_cpp_add_test(test_parser_engine_assembly
  vllm/parser/engine/test_parser_engine_assembly.cpp)
vllm_cpp_add_test(test_openai_tool_parsers
  vllm/entrypoints/openai/tool_parsers/test_tool_parsers.cpp)
vllm_cpp_add_test(test_tool_parser_detect
  vllm/entrypoints/openai/tool_parsers/test_detect.cpp)
vllm_cpp_add_test(test_protocol_tool_turns
  vllm/entrypoints/openai/test_protocol_tool_turns.cpp)
vllm_cpp_add_test(test_reasoning_parser_detect
  vllm/entrypoints/openai/reasoning_parsers/test_detect.cpp)
vllm_cpp_add_test(test_reasoning_think_auto
  vllm/entrypoints/openai/reasoning_parsers/test_think_auto.cpp)
vllm_cpp_add_test(test_deepseek
  vllm/entrypoints/openai/tool_parsers/test_deepseek.cpp)
vllm_cpp_add_test(test_deepseek_v32
  vllm/entrypoints/openai/tool_parsers/test_deepseek_v32.cpp)
vllm_cpp_add_test(test_deepseek_v4
  vllm/entrypoints/openai/tool_parsers/test_deepseek_v4.cpp)
vllm_cpp_add_test(test_longcat
  vllm/entrypoints/openai/tool_parsers/test_longcat.cpp)
vllm_cpp_add_test(test_kimi_k2
  vllm/entrypoints/openai/tool_parsers/test_kimi_k2.cpp)
vllm_cpp_add_test(test_tool_parser_inkling
  vllm/entrypoints/openai/tool_parsers/test_inkling.cpp)
vllm_cpp_add_test(test_granite_tool_parser
  vllm/entrypoints/openai/tool_parsers/test_granite_tool_parser.cpp)
vllm_cpp_add_test(test_granite4_tool_parser
  vllm/entrypoints/openai/tool_parsers/test_granite4_tool_parser.cpp)
vllm_cpp_add_test(test_granite_20b_fc_tool_parser
  vllm/entrypoints/openai/tool_parsers/test_granite_20b_fc_tool_parser.cpp)
vllm_cpp_add_test(test_tool_parsers_llama
  vllm/entrypoints/openai/tool_parsers/test_llama.cpp)
vllm_cpp_add_test(test_tool_parsers_pythonic
  vllm/entrypoints/openai/tool_parsers/test_pythonic.cpp)
vllm_cpp_add_test(test_tool_parsers_llama4_pythonic
  vllm/entrypoints/openai/tool_parsers/test_llama4_pythonic.cpp)
vllm_cpp_add_test(test_tool_choice_grammar
  vllm/entrypoints/openai/tool_parsers/test_tool_choice_grammar.cpp)
vllm_cpp_add_test(test_structural_tags
  vllm/entrypoints/openai/tool_parsers/test_structural_tags.cpp)
vllm_cpp_add_test(test_tool_parser_mistral
  vllm/entrypoints/openai/tool_parsers/test_mistral.cpp)
vllm_cpp_add_test(test_reasoning_base_thinking
  vllm/entrypoints/openai/reasoning_parsers/test_base_thinking.cpp)
vllm_cpp_add_test(test_reasoning_deepseek_r1
  vllm/entrypoints/openai/reasoning_parsers/test_deepseek_r1.cpp)
vllm_cpp_add_test(test_reasoning_mistral
  vllm/entrypoints/openai/reasoning_parsers/test_mistral.cpp)
vllm_cpp_add_test(test_reasoning_minimax_m2
  vllm/entrypoints/openai/reasoning_parsers/test_minimax_m2.cpp)
vllm_cpp_add_test(test_reasoning_step3
  vllm/entrypoints/openai/reasoning_parsers/test_step3.cpp)
vllm_cpp_add_test(test_reasoning_olmo3
  vllm/entrypoints/openai/reasoning_parsers/test_olmo3.cpp)
vllm_cpp_add_test(test_reasoning_deepseek_v3
  vllm/entrypoints/openai/reasoning_parsers/test_deepseek_v3.cpp)
vllm_cpp_add_test(test_reasoning_qwen3
  vllm/entrypoints/openai/reasoning_parsers/test_qwen3.cpp)
vllm_cpp_add_test(test_tool_parser_xlam
  vllm/entrypoints/openai/tool_parsers/test_xlam_tool_parser.cpp)
vllm_cpp_add_test(test_tool_parser_phi4mini
  vllm/entrypoints/openai/tool_parsers/test_phi4mini_tool_parser.cpp)
vllm_cpp_add_test(test_tool_parser_internlm
  vllm/entrypoints/openai/tool_parsers/test_internlm2_tool_parser.cpp)
vllm_cpp_add_test(test_tool_parser_jamba
  vllm/entrypoints/openai/tool_parsers/test_jamba_tool_parser.cpp)
vllm_cpp_add_test(test_tool_parser_step3
  vllm/entrypoints/openai/tool_parsers/test_step3.cpp)
vllm_cpp_add_test(test_tool_parser_step3p5
  vllm/entrypoints/openai/tool_parsers/test_step3p5.cpp)
vllm_cpp_add_test(test_tool_parser_qwen3_coder
  vllm/entrypoints/openai/tool_parsers/test_qwen3_coder.cpp)
vllm_cpp_add_test(test_tool_parser_minicpm5
  vllm/entrypoints/openai/tool_parsers/test_minicpm5.cpp)
vllm_cpp_add_test(test_tool_parser_hy_v3
  vllm/entrypoints/openai/tool_parsers/test_hy_v3.cpp)
vllm_cpp_add_test(test_tool_parser_hunyuan_a13b
  vllm/entrypoints/openai/tool_parsers/test_hunyuan_a13b_tool_parser.cpp)
vllm_cpp_add_test(test_tool_parser_apertus
  vllm/entrypoints/openai/tool_parsers/test_apertus_tool_parser.cpp)
vllm_cpp_add_test(test_tool_parser_ernie45
  vllm/entrypoints/openai/tool_parsers/test_ernie45_tool_parser.cpp)
vllm_cpp_add_test(test_tool_parser_gigachat3
  vllm/entrypoints/openai/tool_parsers/test_gigachat3_tool_parser.cpp)
vllm_cpp_add_test(test_tool_parser_olmo3
  vllm/entrypoints/openai/tool_parsers/test_olmo3.cpp)
vllm_cpp_add_test(test_tool_parser_lfm2
  vllm/entrypoints/openai/tool_parsers/test_lfm2.cpp)
vllm_cpp_add_test(test_tool_parser_poolside_v1
  vllm/entrypoints/openai/tool_parsers/test_poolside_v1.cpp)
vllm_cpp_add_test(test_tool_parser_functiongemma
  vllm/entrypoints/openai/tool_parsers/test_functiongemma.cpp)
vllm_cpp_add_test(test_glm47
  vllm/entrypoints/openai/tool_parsers/test_glm47.cpp)
vllm_cpp_add_test(test_minimax_m2_tool
  vllm/entrypoints/openai/tool_parsers/test_minimax_m2_tool.cpp)
vllm_cpp_add_test(test_tool_parser_gemma4
  vllm/entrypoints/openai/tool_parsers/test_gemma4.cpp)
vllm_cpp_add_test(test_tool_parser_seed_oss
  vllm/entrypoints/openai/tool_parsers/test_seed_oss.cpp)
vllm_cpp_add_test(test_tool_parser_muse_glimmer
  vllm/entrypoints/openai/tool_parsers/test_muse_glimmer_tool_parser.cpp)
vllm_cpp_add_test(test_reasoning_muse_glimmer
  vllm/entrypoints/openai/reasoning_parsers/test_muse_glimmer_reasoning_parser.cpp)
vllm_cpp_add_test(test_chat_template vllm/entrypoints/test_chat_template.cpp)
# The real Qwen3.5 chat template (extracted from the GGUF) lives under
# tests/fixtures/; hand its absolute path to the test so it resolves regardless
# of the working directory ctest runs it from.
target_compile_definitions(test_chat_template PRIVATE
  VLLM_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")
# The HTTP api_server smoke test needs the vendored cpp-httplib (gated).
if(VLLM_CPP_SERVER)
  # ENG-HF-MODEL-DOWNLOAD W2 (#1280): the hub protocol against an in-process fake
  # hub. It starts a REAL cpp-httplib server on an ephemeral port, so it shares
  # api_server.cpp's gate and the RUN_SERIAL treatment two rows below.
  vllm_cpp_add_test(test_hf_hub vllm/transformers_utils/test_hf_hub.cpp)
  # ENG-HF-MODEL-DOWNLOAD W3/W4 (#1280): the byte transport and the `--model`
  # grammar, both against an in-process fake hub on an ephemeral port. Same
  # gate and same RUN_SERIAL treatment as the suite above.
  vllm_cpp_add_test(test_downloader vllm/transformers_utils/test_downloader.cpp)
  vllm_cpp_add_test(test_model_resolver
    vllm/transformers_utils/test_model_resolver.cpp)
  # ENG-HF-MODEL-DOWNLOAD W5 (#1280): the transport-layer-security suite. It is
  # the ONE place in this row that speaks https, and the one place that measures
  # whether the server half and the fetcher half were compiled against the same
  # cpp-httplib layout. Its TLS arm starts a real `httplib::SSLServer` on
  # loopback, so it shares the RUN_SERIAL treatment below.
  vllm_cpp_add_test(test_tls_transport
    vllm/transformers_utils/test_tls_transport.cpp)
  # THE EXPECTATION, computed on a path that does NOT read the macro under test.
  # `VLLM_CPP_TEST_EXPECT_TLS` says "the user asked for the download feature and
  # this host has the OpenSSL development files", which is a fact about the
  # CONFIGURATION. The suite then asserts that the BUILD delivered transport
  # layer security. Deriving the expectation from `CPPHTTPLIB_OPENSSL_SUPPORT`
  # instead would make the case a tautology that a silently disabled feature
  # passes, which is the exact failure `scripts/validate-container-image.py`
  # exists to catch in the container.
  if(VLLM_CPP_HF_DOWNLOAD_REQUESTED AND VLLM_CPP_OPENSSL)
    find_package(OpenSSL QUIET)
    if(OpenSSL_FOUND)
      target_compile_definitions(test_tls_transport PRIVATE
        VLLM_CPP_TEST_EXPECT_TLS=1)
    endif()
  endif()
  vllm_cpp_add_test(test_openai_api_server vllm/entrypoints/openai/test_api_server.cpp)
  # ENG-MM-INPUT-PIPELINE P2 (#2379): the SERVED image request, end to end
  # through the real ApiServer -> AsyncLLM -> Scheduler -> GPUModelRunner ->
  # ModelRegistry::Forward chain on a CPU queue. Its own target rather than a
  # case inside test_openai_api_server, because it needs a different model (a
  # registry LoadedModel carrying a vision tower) and a different tokenizer
  # fixture (one whose added tokens are the vision markers).
  vllm_cpp_add_test(test_openai_api_server_mm_forward
    vllm/entrypoints/openai/test_api_server_mm_forward.cpp)
  # dots3-note W6a (#2512): the SERVED image request for the SECOND multimodal
  # architecture, end to end through the same production chain. Its own target
  # for the same reason the Qwen3-VL one has one: a different model (a registry
  # LoadedModel carrying a dots3-note DENSE vision tower, loaded from a
  # synthetic on-disk checkpoint) and a different tokenizer fixture (one whose
  # added tokens are dots3-note's own image markers).
  vllm_cpp_add_test(test_openai_api_server_dots3_mm_forward
    vllm/entrypoints/openai/test_api_server_dots3_mm_forward.cpp)
  # MODEL-MM-deepseek-v4 W5 (#2411): the REQUEST path for the third multimodal
  # architecture. Its own target for the reason the two above have one: a
  # different tokenizer fixture (one whose added tokens are the pinned DeepSeek
  # chat template's markers and the `<|deepseek_image|>` placeholder) and a
  # different processor.
  # Nemotron Nano VL / Omni chat seam (row
  # MODEL-MM-nano-nemotron-vl-nemotron-h-nano-vl-v2).
  vllm_cpp_add_test(test_nano_nemotron_vl_mm_chat
    vllm/entrypoints/openai/test_nano_nemotron_vl_mm_chat.cpp)
  target_compile_definitions(test_nano_nemotron_vl_mm_chat PRIVATE
    NANO_NEMOTRON_VL_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures")
  vllm_cpp_add_test(test_deepseek_v4_mm_chat
    vllm/entrypoints/openai/test_deepseek_v4_mm_chat.cpp)
  # The language and projector GGUF fixtures live beside the model gates, and
  # they are shared with the W3B/W4 suites so the artifact is described once.
  target_include_directories(test_deepseek_v4_mm_chat PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vllm ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
  target_compile_definitions(test_openai_api_server_dots3_mm_forward PRIVATE
    DOTS3_NOTE_CKPT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/dots3_note_prev")
  # The tiny-checkpoint fixture header lives beside the model gates, and
  # dots3_note.h is a MODEL-PRIVATE header under src/ (#515).
  target_include_directories(test_openai_api_server_dots3_mm_forward PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
  # #1982: WHEN the first chat SSE frame reaches the client. Its own target so
  # the gated-runner fixture cannot perturb the keepalive suite's timings, and
  # inside this guard because it drives the production ApiServer dispatch,
  # which only exists when VLLM_CPP_SERVER is ON.
  vllm_cpp_add_test(test_chat_stream_first_frame
    vllm/entrypoints/openai/test_chat_stream_first_frame.cpp)
  # /v1/audio/transcriptions dispatch + socket smoke run against the REAL
  # library transcription seam on the committed parakeet_e2e fixture
  # (ARCH-ONE-SURFACE ROW 1).
  target_compile_definitions(test_openai_api_server PRIVATE
    PARAKEET_E2E_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/parakeet_e2e"
    # /v1/embeddings dispatch + socket smoke run against the REAL engine path
    # (LoadedEngine -> PoolingRunner) on the committed llama_embed_e2e fixture
    # (ARCH-ONE-SURFACE ROW 6).
    LLAMA_EMBED_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/llama_embed_e2e"
    # #1681: the REAL published Qwen3.8 chat template, rendered through the
    # production /v1/chat/completions dispatch. Absolute so it resolves from
    # whatever working directory ctest runs the suite in.
    VLLM_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")
  target_include_directories(test_openai_api_server PRIVATE ${CMAKE_SOURCE_DIR}/src)
  # M3.6: the OpenAI server CONFORMANCE suite — the full API contract exercised
  # end to end over the REAL cpp-httplib server on an ephemeral port.
  vllm_cpp_add_test(test_openai_conformance vllm/entrypoints/openai/test_conformance.cpp)
  # Both spin up a REAL cpp-httplib server (background accept thread) and drive it
  # with a client on a bounded connect/read timeout. They pass reliably run alone,
  # but under a saturated `ctest -jN` (the multi-minute paged-engine/bench tests
  # peg every core) the server's accept thread starves and the client sees
  # connection failures / an abort. RUN_SERIAL reproduces the standalone
  # condition — the test runs with no other ctest test concurrent — so the
  # parallel suite is deterministic. (A shared RESOURCE_LOCK would only serialize
  # these two against each other, not against the CPU-hog tests that actually
  # cause the starvation.)
  set_tests_properties(test_openai_api_server test_openai_conformance test_hf_hub
    test_downloader test_model_resolver test_tls_transport
    PROPERTIES RUN_SERIAL ON)
  # MODEL-MUSIC3 (#672): the end-to-end suite. It CONSTRUCTS an `ApiServer`
  # and drives its speech routes over a real socket, so it links the
  # translation unit `CMakeLists.txt` compiles only inside this guard. It was
  # registered outside it until #1883, which made every
  # `-DVLLM_CPP_SERVER=OFF` configuration configure cleanly and then fail to
  # link; `scripts/check-test-registration.py` now refuses that shape.
  vllm_cpp_add_test(test_minimax_music3_e2e_real parity/test_minimax_music3_e2e_real.cpp)
  target_compile_definitions(test_minimax_music3_e2e_real PRIVATE
    MUSIC3_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens/minimax_music3_oracle"
    # Where the e2e case writes the WAV it generated, so a human can listen to it.
    # The BUILD tree, never the source tree: it is an artifact of a run and not a
    # golden, and nothing under tests/parity/goldens/ is created or replaced.
    BUILD_ARTIFACT_DIR="${CMAKE_BINARY_DIR}/music3")
  target_include_directories(test_minimax_music3_e2e_real PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/parity)
  # This suite binds a REAL cpp-httplib server on an ephemeral port (the
  # speech-only route-table cases, #672), which is the same reason
  # test_openai_api_server and test_openai_conformance carry RUN_SERIAL above:
  # under a saturated `ctest -jN` the server's accept thread starves and the
  # client sees a connection failure rather than the defect it was looking for.
  set_tests_properties(test_minimax_music3_e2e_real PROPERTIES RUN_SERIAL ON)

  # SERVE-RECIPE-ARGS (#606): the accepted-and-inert serve-argument seam. It
  # drives the REAL VllmServerMain (owned by the library behind the same
  # VLLM_CPP_SERVER gate) by re-execing this binary, so it needs src/ on the
  # include path for server_main.h, exactly as test_openai_api_server does.
  vllm_cpp_add_test(test_serve_recipe_args
    vllm/entrypoints/openai/test_serve_recipe_args.cpp)
  target_include_directories(test_serve_recipe_args PRIVATE ${CMAKE_SOURCE_DIR}/src)
  # ENG-HF-MODEL-DOWNLOAD W4 (#1280): THE REACHABILITY GATE. It enters at the
  # REAL VllmServerMain with `--model org/repo` and an in-process fake hub, so
  # it needs src/ on the include path for server_main.h, exactly as the suite
  # above does, and the committed llama_embed_e2e checkpoint as the payload the
  # hub serves. It binds a port and forks a server, so it is RUN_SERIAL for the
  # same starvation reason as test_openai_api_server.
  vllm_cpp_add_test(test_serve_hf_model
    vllm/entrypoints/openai/test_serve_hf_model.cpp)
  target_include_directories(test_serve_hf_model PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_compile_definitions(test_serve_hf_model PRIVATE
    VLLM_LLAMA_EMBED_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/llama_embed_e2e")
  set_tests_properties(test_serve_hf_model PROPERTIES RUN_SERIAL ON)
  # ENG-MM-INPUT-PIPELINE wave L2 (#607): --language-model-only /
  # --limit-mm-per-prompt. Same shape and the same reason as the row above — it
  # re-execs the REAL VllmServerMain to observe the flags reaching the config and
  # a malformed limit aborting before the load.
  vllm_cpp_add_test(test_serve_mm_limits
    vllm/entrypoints/openai/test_serve_mm_limits.cpp)
  target_include_directories(test_serve_mm_limits PRIVATE ${CMAKE_SOURCE_DIR}/src)
  # ENG-RESIDENCY-CONFIG (#1110): --offload-config's vllm_cpp key reaching the
  # loader through the REAL VllmServerMain. Same include path for server_main.h.
  vllm_cpp_add_test(test_serve_residency_config
    vllm/entrypoints/openai/test_serve_residency_config.cpp)
  target_include_directories(test_serve_residency_config PRIVATE ${CMAKE_SOURCE_DIR}/src)
  # KV-FP8 W3 (#1593): --kv-cache-dtype reaching EngineParams through the REAL
  # VllmServerMain. Same shape and the same reason as the row above.
  vllm_cpp_add_test(test_serve_kv_cache_dtype
    vllm/entrypoints/openai/test_serve_kv_cache_dtype.cpp)
  target_include_directories(test_serve_kv_cache_dtype PRIVATE ${CMAKE_SOURCE_DIR}/src)
  # MODEL-MM-deepseek-v4 W5 (#2411): --mmproj and the resolved model config
  # reaching the multimodal chat install through the REAL VllmServerMain. Same
  # shape and the same reason as the two rows above; it also needs the GGUF
  # fixtures beside the model gates, because the two lines it holds are only
  # observable on a checkpoint whose architecture declares multimodal support.
  vllm_cpp_add_test(test_serve_deepseek_v4_mm
    vllm/entrypoints/openai/test_serve_deepseek_v4_mm.cpp)
  target_include_directories(test_serve_deepseek_v4_mm PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/vllm
    ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
  # MODEL-MM-nano-nemotron-vl-nemotron-h-nano-vl-v2: the engine's max_model_len
  # reaching the Nemotron Nano VL chat install through the REAL VllmServerMain
  # AND the C ABI's vllm_chat. Same re-exec shape as the row above; the tiny
  # checkpoint is written as a model directory at run time.
  vllm_cpp_add_test(test_serve_nano_nemotron_vl_mm
    vllm/entrypoints/openai/test_serve_nano_nemotron_vl_mm.cpp)
  target_include_directories(test_serve_nano_nemotron_vl_mm PRIVATE ${CMAKE_SOURCE_DIR}/src)
endif()
vllm_cpp_add_test(test_outputs vllm/test_outputs.cpp)

# M2.1: the benchmark harness smoke — drives the synthetic engine through the
# full admission + step() measurement loop (shares examples/bench/bench_core.h
# with the vllm-bench CLI).
vllm_cpp_add_test(test_bench examples/test_bench.cpp)
target_include_directories(test_bench PRIVATE ${CMAKE_SOURCE_DIR}/examples/bench)
# KV-FP8 W7 (#2619): `--kv-cache-dtype` on `vllm-bench`. It runs the BINARY,
# because that is the only thing that proves the FLAG arrives: `test_bench`
# above already drives `RunBench` from a hand-built `BenchConfig`, so a test
# that set the field here would re-prove the field. Same `$<TARGET_FILE:...>` +
# `SKIP_RETURN_CODE 77` shape as `test_cli_kv_cache_dtype`, and registered
# UNCONDITIONALLY for the same reason: with examples off the suite exits 77 and
# CTest reports it Skipped, which REPORTS the gate's absence instead of hiding
# it behind an option.
vllm_cpp_add_test(test_bench_kv_cache_dtype
  examples/test_bench_kv_cache_dtype.cpp)
if(TARGET vllm-bench OR VLLM_CPP_BUILD_EXAMPLES)
  target_compile_definitions(test_bench_kv_cache_dtype
    PRIVATE VLLM_BENCH_BINARY="$<TARGET_FILE:vllm-bench>")
endif()
# #2759: `--ignore-eos` / `--skip-chat-template` / `--chat-template` on
# `vllm-bench`. Same shape and the same reason as the row above -- it runs the
# BINARY, because `test_bench` drives `RunBench` from a hand-built `BenchConfig`
# and stayed green on every day these flags did not exist. Registered
# UNCONDITIONALLY so that with examples off the suite exits 77 and CTest reports
# the gate's absence instead of hiding it behind an option.
vllm_cpp_add_test(test_bench_eos_chat_template
  examples/test_bench_eos_chat_template.cpp)
if(TARGET vllm-bench OR VLLM_CPP_BUILD_EXAMPLES)
  target_compile_definitions(test_bench_eos_chat_template
    PRIVATE VLLM_BENCH_BINARY="$<TARGET_FILE:vllm-bench>")
endif()

# C ABI (M3.5 Task 1): the C++ test drives the public C API over a synthetic
# engine via the internal MakeEngineHandle hook (reached under src/).
vllm_cpp_add_test(test_capi capi/test_capi.cpp)
vllm_cpp_add_test(test_capi_qwen35_full_attn_state capi/test_qwen35_full_attn_state.cpp)
target_include_directories(test_capi PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(test_capi PRIVATE
  PARAKEET_E2E_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/parakeet_e2e"
  MINIMAX_H3_VIDEO_FOLD_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/minimax_h3_video_fold"
  LLAMA_EMBED_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/llama_embed_e2e"
  GLINER2_E2E_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/gliner2_e2e"
  # #1681: the real published Qwen3.8 chat template, rendered through the C
  # ABI's own vllm_chat path. Absolute so it resolves from whatever working
  # directory ctest runs the suite in.
  VLLM_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")
vllm_cpp_add_test(test_chat_prompt capi/test_chat_prompt.cpp)
target_include_directories(test_chat_prompt PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Pure-C compile check: include/vllm.h must compile as C11 with no C++ leaks.
enable_language(C)
add_library(vllm_capi_c_check STATIC capi/c_header_compile.c)
target_link_libraries(vllm_capi_c_check PRIVATE vllm::vllm)
target_compile_options(vllm_capi_c_check PRIVATE
  $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -Werror -std=c11>)

# ── Packaging DoD (M3.5 Task 3): dlopen the built libvllm.so + verify exports ──
# These are gated on the shared-lib target existing. The dlopen smoke test links
# ONLY the doctest main + ${CMAKE_DL_LIBS} (NOT libvllm) — it resolves the whole
# C ABI at runtime via dlsym, proving a header-less FFI consumer works.
if(TARGET vllm_shared)
  add_executable(test_dlopen capi/test_dlopen.cpp)
  target_link_libraries(test_dlopen PRIVATE vllm_test_main ${CMAKE_DL_LIBS})
  # Only the public header for the ABI *types*; the .so path is injected so the
  # test dlopen()s the exact artifact this build produced.
  target_include_directories(test_dlopen PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_compile_definitions(test_dlopen PRIVATE
    VLLM_SHARED_LIB_PATH="$<TARGET_FILE:vllm_shared>")
  vllm_cpp_set_warnings(test_dlopen)
  add_test(NAME test_dlopen COMMAND test_dlopen)

  # Export-set verification: fail if libvllm.so leaks any non-`vllm_*` dynamic
  # symbol (i.e. C++ internals). Runs nm via cmake -P.
  if(UNIX AND NOT APPLE)
    add_test(NAME capi_shared_exports_only_abi
      COMMAND ${CMAKE_COMMAND}
        "-DVLLM_SHARED_LIB=$<TARGET_FILE:vllm_shared>"
        -P "${CMAKE_SOURCE_DIR}/cmake/VerifyExports.cmake")
  endif()
endif()
vllm_cpp_add_test(test_cuda_backend vt/test_cuda_backend.cpp)
vllm_cpp_add_test(test_dropin_abi vt/test_dropin_abi.cpp)
target_include_directories(test_dropin_abi PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_cuda_ops vt/test_cuda_ops.cpp)
vllm_cpp_add_test(test_dtype vt/test_dtype.cpp)
vllm_cpp_add_test(test_backend vt/test_backend.cpp)
add_executable(test_vulkan_loader
  vt/test_vulkan_loader.cpp
  ${CMAKE_SOURCE_DIR}/src/vt/vulkan/vulkan_loader.cpp)
target_include_directories(test_vulkan_loader PRIVATE
  ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_vulkan_loader PRIVATE vllm_test_main ${CMAKE_DL_LIBS})
vllm_cpp_set_warnings(test_vulkan_loader)
add_test(NAME test_vulkan_loader COMMAND test_vulkan_loader)
# CPU-vs-device op-equality harness (BACKEND-METAL-MLX W0). Unconditional: it
# discovers which non-CPU backends this build registered and compares only those,
# so it is meaningful on a CUDA box, on the Metal M4, and inert (CPU-only bit-
# exactness checks) on a plain CPU build.
vllm_cpp_add_test(test_backend_cross_device vt/test_backend_cross_device.cpp)
# The head_dim-128 decode arm (#382) ships default OFF, and its flag is read into
# a `static const bool` -- once per process. The default registration above
# therefore only ever gates the PagedAttnOnline fallback; the arm that runs the new
# kernel needs its own INVOCATION with the flag set. Same shape as the
# test_dense_gateup_fused_marlin_off_* lever pair above. The filter avoids commas:
# doctest reads them as filter separators and the case name contains three.
add_test(NAME test_backend_cross_device_vt_attn_decode_d128
  COMMAND test_backend_cross_device "--test-case=*head_dim 128*")
set_tests_properties(test_backend_cross_device_vt_attn_decode_d128 PROPERTIES
  ENVIRONMENT "VT_ATTN_DECODE_D128=1" SKIP_RETURN_CODE 77)
# The ROCm GCN-arch capability parse (BACKEND-ROCM W0). UNCONDITIONAL by design:
# include/vt/rocm/rocm_arch.h is deliberately free of HIP headers so the one part
# of that skeleton carrying a DECISION is gated on a CPU-only CI runner with no
# AMD GPU. The rest of the ROCm skeleton is under VLLM_CPP_HIP below.
vllm_cpp_add_test(test_rocm_arch vt/test_rocm_arch.cpp)
# The pure half of the ROCm pinned H2D bounce ring (chunk plan, cross-call slot
# reuse, the five-term predicate). UNCONDITIONAL for exactly the reason
# test_rocm_arch is: include/vt/rocm/rocm_pinned_h2d.h carries no HIP header, and
# a ring that silently never engages produces identical bytes -- only the host
# residency differs -- so the decision has to be gated on a runner with no AMD
# GPU. The reachability half is in test_backend_cross_device.
vllm_cpp_add_test(test_rocm_pinned_h2d vt/test_rocm_pinned_h2d.cpp)
vllm_cpp_add_test(test_rocm_f16_contract vt/test_rocm_f16_contract.cpp)
# The grow-only per-stream device scratch pool (#2712). UNCONDITIONAL for the
# same reason test_rocm_arch is: src/vt/grow_only_stream_scratch.h is free of
# HIP and CUDA headers, so the piece of the ROCm scratch allocators that holds
# the SYNCHRONISATION is gated on a CPU-only runner with no accelerator.
vllm_cpp_add_test(test_grow_only_stream_scratch vt/test_grow_only_stream_scratch.cpp)
target_include_directories(test_grow_only_stream_scratch PRIVATE ${CMAKE_SOURCE_DIR}/src)
# The fused ROCm norm-quant token is pure host bookkeeping. Keep its identity,
# one-shot, and concurrency contract gated when no AMD compiler is installed.
vllm_cpp_add_test(test_rocm_norm_quant_bridge vt/test_rocm_norm_quant_bridge.cpp)
target_include_directories(test_rocm_norm_quant_bridge PRIVATE ${CMAKE_SOURCE_DIR}/src)
# The ROCm quant-dot device cases skip without HIP. Its scratch policy is
# compiled with host fakes so capture ownership remains gated in CPU-only CI.
vllm_cpp_add_test(test_rocm_quant_dot vt/test_rocm_quant_dot.cpp)
target_include_directories(test_rocm_quant_dot PRIVATE ${CMAKE_SOURCE_DIR}/src)
# #785 P1 host package (fixture/oracle/trace classifier). CPU-only.
vllm_cpp_add_test(test_ops_paged_attn_sharedk_wmma_p1
  vt/test_ops_paged_attn_sharedk_wmma_p1.cpp)
vllm_cpp_add_test(test_triton_aot_arch_dispatch
  vt/test_triton_aot_arch_dispatch.cpp)
target_include_directories(test_triton_aot_arch_dispatch PRIVATE
  ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_cpu_isa_x86 vt/test_cpu_isa_x86.cpp)
target_include_directories(test_cpu_isa_x86 PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_cpu_isa_arm vt/test_cpu_isa_arm.cpp)
target_include_directories(test_cpu_isa_arm PRIVATE ${CMAKE_SOURCE_DIR}/src)
if(VLLM_CPP_HIP)
# Plain C++ (no HIP header): every assertion goes through the vt:: seam. Each
# case no-ops when the build has HIP but the box has no AMD GPU.
vllm_cpp_add_test(test_rocm_backend vt/test_rocm_backend.cpp)
# HIP runtime queries prove allocation provenance and stream capture for split.
vllm_cpp_add_test(test_rocm_attn_gate_split vt/test_rocm_attn_gate_split.cpp)
target_compile_definitions(test_rocm_attn_gate_split PRIVATE __HIP_PLATFORM_AMD__)
# Resource-specific cases run in separate processes so a missing second device
# or insufficient memory reports its own skip without hiding the ordinary gate.
add_test(NAME test_rocm_attn_gate_split_device_binding COMMAND test_rocm_attn_gate_split
  --no-skip "--test-case=ROCm split binds*")
add_test(NAME test_rocm_attn_gate_split_large_index COMMAND test_rocm_attn_gate_split
  --no-skip "--test-case=ROCm split indexes*")
set_tests_properties(test_rocm_attn_gate_split_device_binding
  test_rocm_attn_gate_split_large_index PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
vllm_cpp_add_test(test_rocm_f16_weights vt/test_rocm_f16_weights.cpp)
add_test(NAME test_rocm_f16_graph_dedup COMMAND test_rocm_f16_weights
  "--test-case=ROCm F16 graph handles*")
set_tests_properties(test_rocm_f16_graph_dedup PROPERTIES
  ENVIRONMENT "VT_CUDA_GRAPH_DEDUP=1")
add_executable(test_rocm_f16_device vt/test_rocm_f16_device.hip)
target_link_libraries(test_rocm_f16_device PRIVATE vllm::vllm)
foreach(_f16_allocation_target test_rocm_f16_weights test_rocm_f16_device)
  target_sources(${_f16_allocation_target} PRIVATE vt/rocm_f16_alloc_observer.hip)
  target_link_options(${_f16_allocation_target} PRIVATE
    "LINKER:--wrap=hipMalloc" "LINKER:--wrap=hipFree")
endforeach()
vllm_cpp_set_warnings(test_rocm_f16_device)
add_test(NAME test_rocm_f16_device COMMAND test_rocm_f16_device)
set_tests_properties(test_rocm_f16_device PROPERTIES SKIP_RETURN_CODE 77)
add_executable(test_rocm_f16_model vllm/test_rocm_f16_model.cpp)
target_link_libraries(test_rocm_f16_model PRIVATE vllm::vllm)
vllm_cpp_set_warnings(test_rocm_f16_model)
add_test(NAME test_rocm_f16_model COMMAND test_rocm_f16_model)
set_tests_properties(test_rocm_f16_model PROPERTIES SKIP_RETURN_CODE 77)
# Duplicate provider registrations keep their first entry. Wrap the native
# kernel symbols at link time so observation still executes the real kernels.
foreach(_f16_observed_target test_rocm_f16_model test_gguf_keep_quant)
  target_link_options(${_f16_observed_target} PRIVATE
    "LINKER:--wrap=_ZN2vt4rocm16MatmulKernelRocmERNS_5QueueERNS_6TensorERKS3_S6_"
    "LINKER:--wrap=_ZN2vt4rocm18MatmulBTKernelRocmERNS_5QueueERNS_6TensorERKS3_S6_"
    "LINKER:--wrap=_ZN2vt4rocm19EmbeddingKernelRocmERNS_5QueueERNS_6TensorERKS3_S6_"
    "LINKER:--wrap=_ZN2vt4rocm17CastF32KernelRocmERNS_5QueueERNS_6TensorERKS3_")
endforeach()
vllm_cpp_add_test(test_rocm_fp8_kv_cache vt/test_rocm_fp8_kv_cache.cpp)
# BACKEND-ROCM silu-gate dtype rounding repair (#2889 silu item, #1954): the
# ROCm SiluAndMul/MoeSiluMul kernels narrow silu(gate) to the gate dtype before
# the multiply, matching the CPU oracle RoundThrough and upstream vLLM silu_kernel.
# Self-skipping without a ROCm device, mirroring test_rocm_backend.cpp guard.
vllm_cpp_add_test(test_ops_rocm_silu_rounding vt/test_ops_rocm_silu_rounding.cpp)
# #785 P1 GPU product-seam witness. Executable only — NOT add_test.
# Ordinary CTest must not see this target. Runner fail-closes on 77/nonzero.
add_executable(test_ops_paged_attn_sharedk_wmma_p1_gpu
  vt/test_ops_paged_attn_sharedk_wmma_p1_gpu.cpp)
target_include_directories(test_ops_paged_attn_sharedk_wmma_p1_gpu PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR})
if(TARGET vllm::sanitize_test_shared)
  target_link_libraries(test_ops_paged_attn_sharedk_wmma_p1_gpu PRIVATE
    vllm::sanitize_test_shared vllm_test_main)
else()
  target_link_libraries(test_ops_paged_attn_sharedk_wmma_p1_gpu PRIVATE
    vllm::vllm vllm_test_main)
endif()
vllm_cpp_set_warnings(test_ops_paged_attn_sharedk_wmma_p1_gpu)
# GFX1100-TG200-T2b: ROCm decode-graph capture through ModelRegistry::Forward.
# Self-skips without a ROCm device (vt::rocm::DeviceAvailable()) and on non-
# gfx1100 arches (per-arch opt-in keeps the path eager). On gfx1100 it enters
# the production forward and exercises the captured decode-graph path.
vllm_cpp_add_test(test_rocm_decode_graph_forward
  vllm/models/test_rocm_decode_graph_forward.cpp)
target_include_directories(test_rocm_decode_graph_forward PRIVATE
  ${CMAKE_SOURCE_DIR}/src)
endif()
if(VLLM_CPP_METAL)
  vllm_cpp_add_test(test_metal_backend vt/test_metal_backend.cpp)
  if(VLLM_CPP_MLX)
    # THE #1584 EXACTNESS CASE IN ITS OWN PROCESS (issue #1692 O2). The case
    # asserts `declines == 1` on the FIRST decline of the process, so it only
    # measures anything while `MlxFallback`'s function-local static is cold. In
    # the full-suite run it is first by line order and therefore cold, but that
    # is a property of where it sits in the file; a dedicated process makes it a
    # property of the gate. This is O3 of the same row
    # (https://github.com/mudler/vllm.cpp/issues/1812) applied to the Metal arm,
    # in the shape `test_max_abs_diff_nan_raises` above already uses.
    #
    # `--test-case=` splits its argument on COMMAS, so the case name deliberately
    # contains none. A filter that matches nothing prints "0 cases ran" and exits
    # 0, so the macos-metal-mlx CI job additionally asserts the `test cases:` line
    # rather than trusting this entry's exit status alone.
    # QUOTED, and that is not cosmetic: unquoted, CMake splits the argument on
    # its spaces into `--test-case=*EXACTLY`, `one` and `decline*`, doctest
    # matches no case, prints `0 cases ran` and `Status: SUCCESS!`, and ctest
    # reports the entry green forever. The macos-metal-mlx job asserts the
    # `test cases: 1 | 1 passed` line out of a `ctest -V` run for exactly this.
    add_test(NAME test_metal_backend_mlx_decline_exact
      COMMAND test_metal_backend "--test-case=*EXACTLY one decline*")
    set_tests_properties(test_metal_backend_mlx_decline_exact
      PROPERTIES SKIP_RETURN_CODE 77)
  endif()
endif()
if(VLLM_CPP_VULKAN)
  vllm_cpp_add_test(test_vulkan_backend vt/test_vulkan_backend.cpp)
  # Reaches the backend-internal vulkan_context.h / vulkan_spirv.h (both under
  # src/, both plain C++ with no Vulkan headers in their interface) to gate the
  # COMMITTED SPIR-V table and the probed float-controls. Same escape hatch
  # test_model_registry already uses above.
  target_include_directories(test_vulkan_backend PRIVATE ${CMAKE_SOURCE_DIR}/src)
endif()
if(VLLM_CPP_TENSTORRENT)
  # Plain C++ (no ttnn/tt-metal header): every assertion goes through the
  # public vt:: seam, same rationale as test_vulkan_backend.cpp's own comment.
  # Each case SKIPS (not fails) when the build has the backend compiled in but
  # the box has no Blackhole card.
  vllm_cpp_add_test(test_tenstorrent_backend vt/test_tenstorrent_backend.cpp)
endif()
vllm_cpp_add_test(test_tensor vt/test_tensor.cpp)
# BACKEND-ACCEL-PROVIDER: the acceleration-provider seam (include/vt/op_provider.h)
# — deterministic selection, capability predicates, decline-and-fall-back and the
# selection stats. Device-independent (registers onto the unused kXPU slots).
vllm_cpp_add_test(test_op_provider vt/test_op_provider.cpp)
# S5 portable reference tier (accelerator-seam-audit.md): the unified-memory
# safety gate, correctness-with-zero-native-kernels via the CPU fallback, and the
# native-wins / observability contract. Hardware-free — a fake backend on the
# unused kXPU slot stands in for a unified accelerator, so it runs on every build.
vllm_cpp_add_test(test_reference_tier vt/test_reference_tier.cpp)
# Scale-out W1 (scale-out-distributed.md §W1): the vt::Communicator collective
# abstraction proven on the CPU in-process multi-rank transport — 2/4-rank
# AllReduce-sum + AllGather correctness, Send/Recv rendezvous, and the
# world_size==1 byte-identical no-op that keeps the single-GPU path untouched.
# Hardware-free (N host threads share one barrier); needs threads.
vllm_cpp_add_test(test_communicator vt/test_communicator.cpp)
find_package(Threads REQUIRED)
target_link_libraries(test_communicator PRIVATE Threads::Threads)
# Scale-out W2 (scale-out-distributed.md §W2): the tensor-parallel CORRECTNESS
# gate with NO GPU — sharded-matmul + RowParallel all-reduce == the unsharded
# tp=1 forward, through the ACTUAL W2 helpers (TpShard/TpAllReduceSum) over the
# W1 CPU communicator; RED-first (a dropped all-reduce is the wrong partial).
vllm_cpp_add_test(test_tp_forward vt/test_tp_forward.cpp)
target_link_libraries(test_tp_forward PRIVATE Threads::Threads)
# BACKEND-DISTRIBUTED-TP TP-W1 (tensor-parallelism-spike.md §S4): the rank-layout
# group math == vLLM's arange(world).reshape(...).view(-1, tp) (an independent
# reshape oracle in-test), + tp=1 inertness + the thread-local current-rank
# accessors. Pure integer math; uses std::thread only for the thread_local
# isolation check.
vllm_cpp_add_test(test_parallel_layout vllm/distributed/test_parallel_layout.cpp)
target_link_libraries(test_parallel_layout PRIVATE Threads::Threads)
# Scale-out W2 multi-device backend registry: Device{type,index} addressability
# and the byte-neutral device-0 proof (a second backend at index 1 leaves index 0
# and the type-level API identical). Hardware-free (a fake backend on kXPU).
vllm_cpp_add_test(test_backend_multidevice vt/test_backend_multidevice.cpp)
vllm_cpp_add_test(test_arena vt/test_arena.cpp)
vllm_cpp_add_test(test_ops_matmul vt/test_ops_matmul.cpp)
# #839 host lifetime + source invariants (no GPU).
vllm_cpp_add_test(test_ops_gemma4_prefill_peer vt/test_ops_gemma4_prefill_peer.cpp)
target_compile_definitions(test_ops_gemma4_prefill_peer PRIVATE
  VLLM_CPP_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
# D1 (fold plan Tier D1): the merged-QKV == separate-q/k/v A/B — one MatmulBT over
# the merged [Nq+Nk+Nv,H] owner + a contiguous QkvSplit is BYTE-IDENTICAL to three
# separate MatmulBT (bit-exact GEMM math, wider N; RED-first). CPU-only.
vllm_cpp_add_test(test_ops_qkv_merge vt/test_ops_qkv_merge.cpp)
# Fold plan Tier C2: the vision-tower merged-QKV *with bias* A/B — ONE MatmulBT +
# a fused merged-[3d] BIAS epilogue + a contiguous QkvSplit is BYTE-IDENTICAL to
# three separate {MatmulBT + bias} (the NEW merged-bias piece vs D1; RED-first
# wrong-bias-slice fails). Exercises models::FusedMergedQkvBiasSplit. CPU-only.
vllm_cpp_add_test(test_ops_qkv_merged_bias vt/test_ops_qkv_merged_bias.cpp)
# QUANT-GGUF-CIQ-GEMM G1: block dtypes, the quant traits table and its
# cross-check against the GGUF reader's GgmlTraits, and the kMatmulBTQuant
# skeleton's generic-composite fallback. CPU-only.
vllm_cpp_add_test(test_ops_quant_traits vt/test_ops_quant_traits.cpp)
# MODEL-MM-QWEN4-EXP W6a: the DEQUANTIZING GATHER — `vt::Embedding` over a
# block-quantized table, one row decoded per gathered id (port of llama.cpp's
# `ggml_compute_forward_get_rows_q`). This is what lets the 51.2 G-parameter
# `per_layer_token_embd` n-gram table of Qwen3.8-Flash-Next stay IQ4_NL-resident
# at 28.8 GB instead of expanding to 102.4 GB of bf16. Gated BIT-EXACTLY against
# the pinned llama.cpp `b10451` decoding REAL bytes of that shipped tensor.
# CPU-only: `EmbeddingKernelCuda` still refuses a block table (#1989 `## Owed`).
vllm_cpp_add_test(test_ops_embedding_quant vt/test_ops_embedding_quant.cpp)
# QUANT-GGUF-CIQ-GEMM G2/G3: the activation quantizers (quantize_row_q8_0/q8_K)
# and the six tier-0 generic vec_dot kernels plus their kMatmulBTQuant wiring.
# Gated against an INDEPENDENT f64 dequantize-then-dot reference, upstream's
# test-quantize-fns/test-backend-ops thresholds, block-boundary/ragged-K shapes
# and run-to-run + thread-count bit-exactness. Reaches src/ for the threadpool
# header (SwapForTesting) used by the thread-count determinism case. CPU-only.
vllm_cpp_add_test(test_ops_quant_dot vt/test_ops_quant_dot.cpp)
target_include_directories(test_ops_quant_dot PRIVATE ${CMAKE_SOURCE_DIR}/src)
# MODEL-DSV4-EXL3 W1a: the EXL3 (exllamav3) trellis reference dequant. Gated by
# tiles the TEST packs itself off the ENCODE side (tests/test_quant_fn.py:83-128
# + pack.cu:29-57), so the fixture never shares code with the decode under test,
# plus a dense double-precision blockwise-Hadamard reference and a real
# rank-sliced DeepSeek-V4 expert linear (skips when the shard is not mounted).
vllm_cpp_add_test(test_exl3_dequant vt/test_exl3_dequant.cpp)
vllm_cpp_add_test(test_exl3_gemm vt/test_exl3_gemm.cpp)
# ISSUE-LOCAL-01M2BYPW7YTC2B2MY023ETTKQ2: the M > 144 reconstruct dispatch through
# dense_attn::Exl3MatmulD must fall back to Exl3Gemm where no reconstruct kernel exists.
vllm_cpp_add_test(test_exl3_matmul_dispatch vt/test_exl3_matmul_dispatch.cpp)
# MODEL-DSV4-EXL3 W2c: the m<=8 GEMV selection envelope, gated host-side because
# the sentence this row inherited about it describes a guard exl3_gemv.cu:53 has
# commented out. W2d: the fused MoE mgemm, its host-side token sort, and the
# activation choice that mirrors vLLM rather than the kernel oracle.
vllm_cpp_add_test(test_exl3_gemv vt/test_exl3_gemv.cpp)
vllm_cpp_add_test(test_exl3_moe vt/test_exl3_moe.cpp)
# BACKEND-ROCM-EXL3 (#2433): the ROCm arms of the exactly two ops an EXL3
# checkpoint ran on the CPU reference tier. Built in EVERY configuration and not
# only under VLLM_CPP_HIP -- its cases skip when no ROCm backend is registered,
# and a suite that only a HIP build compiles is a suite CI never reads.
vllm_cpp_add_test(test_exl3_rocm vt/test_exl3_rocm.cpp)
# BACKEND-VULKAN-EXL3 (#2530): the Vulkan arms of the same two ops. Built in
# EVERY configuration for the same reason as the ROCm sibling above -- its cases
# skip when no Vulkan backend is registered. UNLIKE that sibling this suite needs
# NO GPU and NO lease to say something: llvmpipe is a conformant Vulkan 1.1
# device, so the `build-test-vulkan` CI leg runs the byte gate for real.
vllm_cpp_add_test(test_exl3_vulkan vt/test_exl3_vulkan.cpp)

# QUANT-EXL3 W1a (#2181). Placed after the MODEL-DSV4-EXL3 kernel suites above
# rather than among them: these two gate the SCHEME on the shared linear seam,
# not the kernels.
vllm_cpp_add_test(test_cast_f16 vt/test_cast_f16.cpp)
vllm_cpp_add_test(test_exl3_linear_method vllm/model_executor/layers/test_exl3_linear_method.cpp)
vllm_cpp_add_test(test_exl3_native_loader vllm/model_executor/layers/test_exl3_native_loader.cpp)
vllm_cpp_add_test(test_exl3_checkpoint vllm/model_executor/layers/test_exl3_checkpoint.cpp)
# The decode against REAL exllamav3 data (#2181): the gate whose absence let a
# wrong codebook ship.
vllm_cpp_add_test(test_exl3_real_decode vt/test_exl3_real_decode.cpp)
# The EXL3 arm REACHED from the dense forward (#2181): the capability gate, not
# a class gate.
vllm_cpp_add_test(test_llama_exl3_forward vllm/models/test_llama_exl3_forward.cpp)

# MODEL-QWEN35-EXL3 (#2495 items 3 and 5): the EXL3 arm of the DENSE Qwen3.5
# loader and forward, gated through LoadQwen3_5Dense and ModelRegistry::Forward.
vllm_cpp_add_test(test_qwen35_exl3 vllm/models/test_qwen35_exl3.cpp)
# MODEL-DFLASH2-EXL3 (#2495 item 7): the EXL3 arm of the DFlash2 DRAFT loader
# and forward, gated through LoadQwen3DFlash over a real safetensors file and
# then through Qwen3DFlashModel::ForwardBlockLogits.
vllm_cpp_add_test(test_qwen3_dflash2_exl3 vllm/models/test_qwen3_dflash2_exl3.cpp)
# MODEL-DFLASH2-NVFP4 (#2758): the ModelOpt NVFP4 arm of the same DRAFT loader,
# gated through MakeQwen3DFlashDraftConfig + LoadQwen3DFlash over a real
# safetensors file — the exact pair LoadDflashDraft calls. A load gate and not a
# forward twin: vt::MatmulNvfp4 is registered for CUDA only, so there is no host
# path to decode a packed weight against.
vllm_cpp_add_test(test_qwen3_dflash2_nvfp4 vllm/models/test_qwen3_dflash2_nvfp4.cpp)
# QUANT-GGUF-CIQ-GEMM G7: repack-at-load for the q8_0 quant GEMM. Proves the
# i8mm interleave transform matches make_block_q8_0x4 and round-trips (portable),
# and that the repacked gemm/gemv is BYTE-IDENTICAL to the plain quant GEMM
# across decode/leftover/prefill shapes and thread counts (i8mm boxes; skips
# coherently elsewhere). Reaches src/ for the block/repack/threadpool headers.
vllm_cpp_add_test(test_ops_quant_repack vt/test_ops_quant_repack.cpp)
target_include_directories(test_ops_quant_repack PRIVATE ${CMAKE_SOURCE_DIR}/src)
# QUANT-GGUF-CIQ-GEMM-CUDA: the CUDA keep-quant GEMM (kCUDA provider for
# kMatmulBTQuant). Gates the GPU dequant-in-kernel dot against the landed CPU
# keep-quant oracle + an f64 dequant reference on the Q8_K family (DeepSeek-V4's
# IQ2_XXS/IQ3_XXS/Q2_K plus the other four k-quants). Skips coherently with no
# CUDA device. Reaches src/ for vt/cpu block/quant headers used by the reference.
vllm_cpp_add_test(test_cuda_quant_dot vt/test_cuda_quant_dot.cpp)
target_include_directories(test_cuda_quant_dot PRIVATE ${CMAKE_SOURCE_DIR}/src)
# KGATHER: the CUDA DEQUANTIZING GATHER. `vt::Embedding` over a block-quantized
# table on the device, gated BIT-EXACTLY against the CPU arm (itself gated
# against the pinned llama.cpp on real shipped bytes) across every encoding
# `vt::cpu::BlockToFloat` decodes, in f32 and bf16 out, i32 and i64 ids, plus the
# out-of-range report. Also pins the CUDA decoder list EQUAL to the CPU one,
# which is the property `DeviceQuantGatherSupported` (a device predicate with no
# dtype) rests on. Skips coherently with no CUDA device; the list-equality case
# is a build property and runs without one. Reaches src/ for the vt/cuda seam
# header and the vt/cpu quant headers.
vllm_cpp_add_test(test_cuda_embedding_quant vt/test_cuda_embedding_quant.cpp)
target_include_directories(test_cuda_embedding_quant PRIVATE ${CMAKE_SOURCE_DIR}/src)
# BACKEND-ROCM-QUANT-GATHER: the production gate starts at the public loader.
vllm_cpp_add_test(test_capi_rocm_embedding_quant capi/test_rocm_embedding_quant.cpp)
# Quantized WMMA reaches both formats through the public completion API.
vllm_cpp_add_test(test_capi_rocm_quant_wmma capi/test_rocm_quant_wmma.cpp)
vllm_cpp_add_test(test_rocm_embedding_quant vt/test_rocm_embedding_quant.cpp)
target_include_directories(test_rocm_embedding_quant PRIVATE ${CMAKE_SOURCE_DIR}/src)
if(VLLM_CPP_HIP)
  set_source_files_properties(vt/test_rocm_embedding_quant.cpp PROPERTIES LANGUAGE HIP)
  if(VLLM_CPP_HIP_ARCHITECTURES)
    set_property(TARGET test_rocm_embedding_quant PROPERTY HIP_ARCHITECTURES "${VLLM_CPP_HIP_ARCHITECTURES}")
  endif()
endif()
add_executable(rocm_quant_gather_capture ../tools/rocm_quant_gather/capture.cpp)
target_link_libraries(rocm_quant_gather_capture PRIVATE vllm)
target_include_directories(rocm_quant_gather_capture PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
add_executable(rocm_quant_wmma_capture ../tools/rocm_quant_wmma/capture.cpp)
target_link_libraries(rocm_quant_wmma_capture PRIVATE vllm)
# CPU-ELEM-GEMM: the specialized + SIMD elementwise (bf16/f16/f32) GEMM. The
# gate is BYTE-IDENTITY against an independent scalar reference across every
# dtype combination, both weight orientations, ragged K/N, row-strided
# activations, thread counts, and the exhaustive 16-bit f16/bf16 widening
# domain. Reaches src/ for the tier header and the threadpool test hook.
vllm_cpp_add_test(test_ops_matmul_elem vt/test_ops_matmul_elem.cpp)
target_include_directories(test_ops_matmul_elem PRIVATE ${CMAKE_SOURCE_DIR}/src)
# LTX25-CONNECTOR-REPAIR (#2434): the M-BLOCKING half of the same contract.
# `MatmulOneChunk` now pads its 16-row activation tile up to a whole number of
# `mr` blocks, so the rows that used to fall through to the one-row kernel are
# computed by the M-blocked one instead. That decides WHICH ROWS SHARE ONE
# ACCUMULATOR SET, and which rows a chunk covers is the threadpool's partition,
# so the gate runs the whole dtype matrix at ragged M and repeats it at SEVEN
# worker counts. The file above gates one shape in one dtype at four counts.
vllm_cpp_add_test(test_ops_matmul_elem_mblock vt/test_ops_matmul_elem_mblock.cpp)
target_include_directories(test_ops_matmul_elem_mblock PRIVATE ${CMAKE_SOURCE_DIR}/src)
# LTX25-TEXT-LINEAR-MEM (#1286): the seam's ALLOCATION bound, on both
# `MatmulOneChunk` instantiations. Every other GEMM gate here asserts values or
# byte equality, so a kernel that allocated a whole intermediate per worker
# would pass all of them. It counts bytes REQUESTED through replaced global
# `operator new` overloads, not peak RSS -- the peak-RSS draft read zero at
# every geometry while passing every bound, because glibc retains the arena
# (spec `## Outcome` W4).
#
# Registered only on Linux, and NOT because of /proc or the threadpool: the
# counter reads nothing under /proc and the pool is portable. It is that the
# replaced allocation functions serve the WHOLE PROGRAM, so the gate depends on
# every `delete` in the binary reaching the matching replaced `delete`, and on
# its fixed slack term being calibrated against the library doing the
# allocating. Both hold for glibc/libstdc++ and are unestablished elsewhere; a
# standard library linked as its own runtime keeps its own `operator new`, and
# the mismatch is a heap corruption rather than a red assertion. Compiling it
# elsewhere with a skip would print `Status: SUCCESS!` over `assertions: 0`,
# which reads as a pass. Extending the lanes is owed, not a one-line change.
# Reaches src/ for the threadpool test hook, which is what lets one process
# sweep the worker count.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  vllm_cpp_add_test(test_ops_matmul_mem vt/test_ops_matmul_mem.cpp)
  target_include_directories(test_ops_matmul_mem PRIVATE ${CMAKE_SOURCE_DIR}/src)
  # The C allocator is counted through the LINKER, not through a symbol
  # definition. `--wrap` redirects the calls made by the objects in this link --
  # which includes libvllm.a, and so cpu_ops.cpp -- while leaving `malloc` itself
  # defined by whoever normally defines it. Defining `malloc` in the test would
  # instead be a second strong definition beside AddressSanitizer's own
  # interceptor, which is what would break the sanitize-cpu lane. Scoped to this
  # one target, so nothing else in the suite is affected.
  #
  # These flags are LOAD-BEARING, not an optimization: the test references
  # `__real_malloc` directly, so dropping one fails the LINK rather than quietly
  # costing the gate its malloc coverage. Verified by mutation (M-W).
  target_link_options(test_ops_matmul_mem PRIVATE
    -Wl,--wrap=malloc
    -Wl,--wrap=calloc
    -Wl,--wrap=realloc
    -Wl,--wrap=aligned_alloc
    -Wl,--wrap=posix_memalign)
endif()
# CPU-tier contract for the env-gated cuBLASLt GEMM algo-selection diagnostic
# helpers. Reaches the internal header under src/ (not public API); the cuBLASLt
# emit it gates is CUDA-only and lives in src/vt/cuda/cuda_matmul.cu.
vllm_cpp_add_test(test_gemm_algo_log vt/test_gemm_algo_log.cpp)
target_include_directories(test_gemm_algo_log PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-tier contract for the cuBLASLt FP8 plan-cache pure plumbing (the
# VT_FP8_PLAN_CACHE flag + the Fp8PlanKey equality/hash completeness). Reaches the
# internal header under src/ (not public API); the cached cuBLASLt {desc, layouts,
# algo} it keys is CUDA-only and lives in src/vt/cuda/cuda_matmul.cu, with the
# byte-exact cached-vs-fresh GEMM proof in vt/test_ops_fp8_cutlass.cpp.
vllm_cpp_add_test(test_fp8_plan_cache vt/test_fp8_plan_cache.cpp)
target_include_directories(test_fp8_plan_cache PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-tier contract for the cuBLASLt heuristic-result cache's pure plumbing
# (VT_GEMM_PLAN_CACHE flag, DEFAULT ON, + the GemmPlanKey equality/hash
# completeness; issue #1732: the per-call heuristic query returns status 14
# inside CUDA graph capture, so the cache is the fix and its default is ON).
# Reaches the internal header under src/ (not public API); the cached
# cublasLtMatmulHeuristicResult_t it keys is CUDA-only and lives in
# src/vt/cuda/cuda_matmul.cu; the end-to-end proof is the row's board gate.
vllm_cpp_add_test(test_gemm_plan_cache vt/test_gemm_plan_cache.cpp)
target_include_directories(test_gemm_plan_cache PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-tier contract for the register-resident packed-decode tiling A/B flag
# (VT_GDN_PACKED_REG_TILE). Reaches the internal header under src/ (not public
# API); the kernel it gates is CUDA-only and lives in src/vt/cuda/cuda_gdn.cu.
vllm_cpp_add_test(test_gdn_packed_reg_tile vt/test_gdn_packed_reg_tile.cpp)
target_include_directories(test_gdn_packed_reg_tile PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-tier contract for the vendored Triton AOT packed-decode fast-path flag
# (VT_GDN_PACKED_DECODE_TRITON, default ON). Reaches the internal header under
# src/ (not public API); the vendored cubin it gates is CUDA-only and lives in
# src/vt/cuda/cuda_gdn.cu (TryTritonPackedDecode -> gdn_decode_h{48,32}).
vllm_cpp_add_test(test_gdn_packed_decode_triton vt/test_gdn_packed_decode_triton.cpp)
target_include_directories(test_gdn_packed_decode_triton PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-tier contract for the opt-in fused-GDN decode BV16 value tile. Reaches
# the CUDA-free selector/geometry header under src/; the executing kernel lives
# in cuda_gdn.cu and is covered by same-process byte-exact CUDA comparisons.
vllm_cpp_add_test(test_gdn_decode_fused vt/test_gdn_decode_fused.cpp)
target_include_directories(test_gdn_decode_fused PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-tier REDUCTION-ORDER contract for the single-warp MoE router top-k
# (VT_MOE_ROUTER_WARP, default ON; spec moe-router-topk-single-warp.md, #378).
# The claim is that the lane map `expert = lane + 32*slot` plus a per-lane
# halving tree performs the bit-identical float additions as the incumbent
# 256-thread block reduction in src/vt/cuda/cuda_moe.cu -- pure IEEE-754, so it
# is provable with no GPU and no nvcc. Reaches the CUDA-free map/tree/selector
# header under src/; the executing kernel is covered on-device by the
# parallel-vs-serial byte-exactness case in test_ops_moe_grouped.
vllm_cpp_add_test(test_moe_router_warp_map vt/test_moe_router_warp_map.cpp)
target_include_directories(test_moe_router_warp_map PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-tier contract for the two NUMERICS-NEUTRAL FP4-quant vectorized-load
# fast-path flags (VT_FP4_QUANT_FAST + VT_SILU_FP4_FAST, both default OFF /
# '1'-opt-in). Reaches the internal header under src/ (not public API); the
# bit-identical vectorized kernels they gate are CUDA-only and live in
# src/vt/cuda/cuda_matmul_nvfp4.cu (ScaledFp4QuantFastKernel /
# SiluAndMulFp4QuantFastKernel).
vllm_cpp_add_test(test_fp4_quant_fast vt/test_fp4_quant_fast.cpp)
target_include_directories(test_fp4_quant_fast PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-tier contract for the decode-RMSNorm kernel-efficiency A/B flag
# (VT_RMSNORM_DECODE_FAST). Reaches the internal header under src/ (not public
# API); the vectorized kernel it gates is CUDA-only and lives in
# src/vt/cuda/cuda_ops.cu (RmsNormRowFastKernel).
vllm_cpp_add_test(test_rmsnorm_decode_fast vt/test_rmsnorm_decode_fast.cpp)
target_include_directories(test_rmsnorm_decode_fast PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-tier contract for the GDN gated-RMSNorm decode kernel-efficiency A/B flag
# (VT_RMSNORM_GATED_FAST). Reaches the internal header under src/ (not public
# API); the bit-identical fast kernel it gates is CUDA-only and lives in
# src/vt/cuda/cuda_gdn.cu (RmsNormGatedRowFastKernel).
vllm_cpp_add_test(test_rmsnorm_gated_fast vt/test_rmsnorm_gated_fast.cpp)
target_include_directories(test_rmsnorm_gated_fast PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-tier contract for the GDN decode causal_conv1d state-update kernel-efficiency
# A/B flag (VT_CONV_UPDATE_FAST). Reaches the internal header under src/ (not public
# API); the bit-identical fast kernel it gates is CUDA-only and lives in
# src/vt/cuda/cuda_gdn.cu (CausalConv1dUpdateFastKernel).
vllm_cpp_add_test(test_conv_update_fast vt/test_conv_update_fast.cpp)
target_include_directories(test_conv_update_fast PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-tier contract for the GDN PREFILL causal_conv1d forward register-window kernel
# (VT_CONV_REG) and split fused post-conv kernel (VT_GDN_POSTCONV_SPLIT) A/B flags.
# Reaches the internal header under src/ (not public API); the bit-identical fast
# kernels they gate are CUDA-only and live in src/vt/cuda/cuda_gdn.cu
# (CausalConv1dFwdRegKernel / GdnPostConvSplitKernel).
vllm_cpp_add_test(test_gdn_prefill_conv vt/test_gdn_prefill_conv.cpp)
target_include_directories(test_gdn_prefill_conv PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-tier contract for the CUDA-graph-safe grow-only device scratch bookkeeping
# (src/vt/cuda/graph_safe_scratch.h): a grown-past scratch block is RETIRED (kept
# resident), never freed, so a device pointer baked into a captured pure-decode CUDA
# graph stays valid — the fix for the 35B c2+ online-serving illegal-memory-access.
# The device proof is the DGX serving harness; this pins the never-free guarantee.
vllm_cpp_add_test(test_breakable_graph vt/test_breakable_graph.cpp)
vllm_cpp_add_test(test_persistent_step_input vt/test_persistent_step_input.cpp)
vllm_cpp_add_test(test_graph_safe_scratch vt/test_graph_safe_scratch.cpp)
target_include_directories(test_graph_safe_scratch PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_graph_dedup vt/test_graph_dedup.cpp)
target_include_directories(test_graph_dedup PRIVATE ${CMAKE_SOURCE_DIR}/src)
# ENG-CUDAGRAPH-DEDUP / #1184: the two device-free halves of the CUDA/HIP binding --
# the error-latch guard every entry point installs, and the structural-signature walk.
# Both lived inside a header that includes <cuda_runtime.h> and were therefore reached
# by NO test on any tier; the latch defect hid in exactly that gap.
vllm_cpp_add_test(test_graph_dedup_runtime vt/test_graph_dedup_runtime.cpp)
target_include_directories(test_graph_dedup_runtime PRIVATE ${CMAKE_SOURCE_DIR}/src)
# ENG-CUDAGRAPH-DEDUP / #1226: the coarse-key knob itself. vt::GraphDedupCoarseKeyEnabled
# caches its answer in a function-local static -- deliberately, because the drivers
# capture lazily and a knob that moved between two captures would give one model a mixed
# set of handles -- so ONE PROCESS CAN OBSERVE ONE VALUE. Proving that "1" enables the
# coarse key and "10" does not therefore takes one process per value, which is what these
# entries are. Without them the accessor's only coverage was a single unset-variable
# CHECK_FALSE, which two mutations survived: accepting any non-null value, and reading a
# completely different variable name.
vllm_cpp_add_test(test_graph_dedup_coarse_env vt/test_graph_dedup_coarse_env.cpp)
target_include_directories(test_graph_dedup_coarse_env PRIVATE ${CMAKE_SOURCE_DIR}/src)
function(vllm_cpp_add_coarse_key_case suffix value expect)
  add_test(NAME test_graph_dedup_coarse_env_${suffix}
    COMMAND test_graph_dedup_coarse_env)
  set_tests_properties(test_graph_dedup_coarse_env_${suffix} PROPERTIES
    ENVIRONMENT "VT_CUDA_GRAPH_DEDUP_COARSE_KEY=${value};VT_COARSE_KEY_EXPECT=${expect}"
    SKIP_RETURN_CODE 77)
endfunction()
# The one value that turns it on. This entry is what proves the accessor reads THIS
# variable: a mutation that reads any other name leaves it unset and fails here.
vllm_cpp_add_coarse_key_case(on "1" 1)
# ...and the near misses, each of which enables the key under a `value != nullptr` test.
# "10" is a hand-edited "1" plus a stray keystroke, and it turning a default-off
# correctness-sensitive path on by accident is the failure the polarity exists to stop.
vllm_cpp_add_coarse_key_case(off_zero "0" 0)
vllm_cpp_add_coarse_key_case(off_ten "10" 0)
vllm_cpp_add_coarse_key_case(off_oh_one "01" 0)
vllm_cpp_add_coarse_key_case(off_one_x "1x" 0)
vllm_cpp_add_coarse_key_case(off_true "true" 0)
vllm_cpp_add_coarse_key_case(off_empty "" 0)
# The knob is DEFINED but the OTHER knob is the one set. The two are separate variables
# and the coarse key must not piggyback on VT_CUDA_GRAPH_DEDUP: a run that asked only for
# dedup must get the exact key the 2026-08-18 device gate ran, or the A/B is not one.
add_test(NAME test_graph_dedup_coarse_env_off_sibling_knob
  COMMAND test_graph_dedup_coarse_env)
set_tests_properties(test_graph_dedup_coarse_env_off_sibling_knob PROPERTIES
  ENVIRONMENT "VT_CUDA_GRAPH_DEDUP=1;VT_COARSE_KEY_EXPECT=0"
  SKIP_RETURN_CODE 77)
# CPU threadpool (QUANT-GGUF-CPU-THREADPOOL): barrier/park-wake/chunk-steal
# stress + the n_threads 1/3/20 determinism battery. Reaches the internal
# header under src/ (the pool is not public API).
vllm_cpp_add_test(test_cpu_threadpool vt/test_cpu_threadpool.cpp)
target_include_directories(test_cpu_threadpool PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_ops_nvfp4_matmul vt/test_ops_nvfp4_matmul.cpp)
vllm_cpp_add_test(test_ops_nvfp4_fp4 vt/test_ops_nvfp4_fp4.cpp)
target_include_directories(test_ops_nvfp4_fp4 PRIVATE ${CMAKE_SOURCE_DIR}/src)
if(NOT WIN32)
  vllm_cpp_add_test(test_nvfp4_persistent_cache vt/test_nvfp4_persistent_cache.cpp)
  target_include_directories(test_nvfp4_persistent_cache PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_compile_definitions(test_nvfp4_persistent_cache PRIVATE
    TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")
endif()
if(VLLM_CPP_CUTLASS)
  foreach(_NVFP4_TACTIC RANGE 0 31)
    add_test(NAME test_ops_nvfp4_fp4_tactic_${_NVFP4_TACTIC}
      COMMAND test_ops_nvfp4_fp4
        "--test-case=matmul_nvfp4_cutlass every forced SM12 tactic matches reference and captures")
    set_tests_properties(test_ops_nvfp4_fp4_tactic_${_NVFP4_TACTIC} PROPERTIES
      ENVIRONMENT
        "VT_FP4_FORCE_TACTIC=${_NVFP4_TACTIC};VT_FP4_FULL_TACTICS=1;VT_FP4_AUTOTUNE=1")
  endforeach()
  add_test(NAME test_ops_nvfp4_persistent_runtime
    COMMAND test_ops_nvfp4_fp4
      "--test-case=NVFP4 persistent runtime loads frozen plans before warmup")
  set_tests_properties(test_ops_nvfp4_persistent_runtime PROPERTIES
    ENVIRONMENT
      "VT_FP4_TEST_PERSISTENT_RUNTIME=1;VT_FP4_PERSISTENT_CACHE=1;VT_FP4_AUTOTUNE_CACHE_READONLY=1;VT_FP4_FLASHINFER_CACHE_PATH=${CMAKE_SOURCE_DIR}/tests/fixtures/nvfp4_flashinfer_v025_gb10/autotune_configs.json;VT_FP4_AUTOTUNE_CACHE_PATH=${CMAKE_CURRENT_BINARY_DIR}/nvfp4-runtime-readonly-unused.json;VT_FP4_AUTOTUNE_DELAY_US=5000;VT_FP4_PLAN_CACHE=1;VT_FP4_AUTOTUNE=1;VT_FP4_FULL_TACTICS=1")
  add_test(NAME test_ops_nvfp4_persistent_save_lifecycle
    COMMAND test_ops_nvfp4_fp4
      "--test-case=NVFP4 persistent runtime publishes only completed warmup")
  set_tests_properties(test_ops_nvfp4_persistent_save_lifecycle PROPERTIES
    ENVIRONMENT
      "VT_FP4_TEST_PERSISTENT_SAVE=1;VT_FP4_PERSISTENT_CACHE=1;VT_FP4_AUTOTUNE_CACHE_READONLY=0;VT_FP4_AUTOTUNE_CACHE_PATH=${CMAKE_CURRENT_BINARY_DIR}/nvfp4-runtime-save-lifecycle.json;VT_FP4_AUTOTUNE_DELAY_US=0;VT_FP4_PLAN_CACHE=1;VT_FP4_AUTOTUNE=1;VT_FP4_FULL_TACTICS=1")
endif()
vllm_cpp_add_test(test_ops_fp8_cutlass vt/test_ops_fp8_cutlass.cpp)
# VT-FP8-W8A8-CPU-ARM (#468): the CPU arm of the static fp8 W8A8 path. Runs on a
# box with NO GPU by construction — that is the point of the row — and carries a
# CUDA-gated arm (G2) for CPU-vs-CUDA byte agreement wherever a device exists.
vllm_cpp_add_test(test_ops_fp8_cpu vt/test_ops_fp8_cpu.cpp)
# VT-QUANT-FP8-GROUP (#1189 M1): the dynamic per-token per-group fp8 activation
# quant. CPU-gateable by construction; its CPU-vs-CUDA byte-identity arm (G6) is
# CUDA-gated and reports PENDING rather than skipping where no device exists.
vllm_cpp_add_test(test_ops_quant_fp8_group_cpu vt/test_ops_quant_fp8_group_cpu.cpp)
# VT-MATMUL-FP8-BLOCK-REF (#1189 M2): the 128x128 block-scaled fp8 GEMM, CPU
# reference arm — the numerical oracle every later block-fp8 kernel is measured
# against. Its G4 holds the constraint the row exists for: the scales apply in
# the MAINLOOP, once per K-block, which an epilogue-folded alpha cannot express.
vllm_cpp_add_test(test_ops_matmul_fp8_block_cpu vt/test_ops_matmul_fp8_block_cpu.cpp)
# VT-MATMUL-FP8-BLOCK-CUDA (#1189 M5): the HOST-SIDE contract of the same op's
# CUDA arm — the tile-config heuristic, the refusal predicate, the two scale
# layouts cutlass DEDUCES from the problem shape, and the dispatch counter. Runs
# on every machine including one with no GPU and no CUDA toolkit, which is the
# whole point: those four are the parts of an sm_12xa-only kernel that can be
# wrong in a way a successful compile cannot see. Reaches the internal header
# under src/ (not public API), exactly like its test_fp8_plan_cache neighbour.
vllm_cpp_add_test(test_fp8_block_scaled_dispatch vt/test_fp8_block_scaled_dispatch.cpp)
target_include_directories(test_fp8_block_scaled_dispatch PRIVATE ${CMAKE_SOURCE_DIR}/src)
# PERF-FP8-SMALL-M-DISPATCH (#1866): the PER-TENSOR fp8 GEMM's sm120 tile ladder,
# host tier. Same arrangement and the same reason as the blockwise neighbour
# above -- a wrong tile is a SLOW answer, so no correctness suite in this tree
# can see it and it has to be gated by value on a host with no GPU.
vllm_cpp_add_test(test_fp8_per_tensor_dispatch vt/test_fp8_per_tensor_dispatch.cpp)
target_include_directories(test_fp8_per_tensor_dispatch PRIVATE ${CMAKE_SOURCE_DIR}/src)
# VT-MATMUL-FP8-BLOCK-CUDA (#1189 M5), the DEVICE tier: the CUDA arm against the
# CPU reference arm of the SAME op, carrying upstream's own
# test_w8a8_block_fp8_cutlass_matmul whole (M=32, N=576, K=7168, rel_diff <
# 0.001). IT HAS NEVER RUN: no GPU was leased for that row and it skips without a
# device. The row's spec lists the on-hardware leg under `## Owed`; a skip here
# is not a pass and must not be reported as one.
vllm_cpp_add_test(test_ops_matmul_fp8_block_cuda vt/test_ops_matmul_fp8_block_cuda.cpp)
target_include_directories(test_ops_matmul_fp8_block_cuda PRIVATE ${CMAKE_SOURCE_DIR}/src)
# fp8 plan-cache arms, re-pinned by FIX-FP8-PLAN-CAPTURE-1843: the cache is
# DEFAULT ON (on CUDA 13.3 the uncached lane queries the cuBLASLt heuristic
# inside CUDA-graph capture and dies — #1843), so the cache-ON arm of the
# byte-exact case is the DEFAULT arm (no env: the first MatmulFp8CublasLt call
# on a shape builds the plan fresh, later calls hit the cache, and the
# first-vs-later byte compare is fresh-plan == cached-plan). The rollback arm
# pins the VT_FP8_PLAN_CACHE=0 escape hatch (every call rebuilds fresh — the
# same-binary A/B arm), proving it stays byte-stable across repeats.
add_test(NAME test_ops_fp8_cutlass_plan_cache_default_on
  COMMAND test_ops_fp8_cutlass
    "--test-case=fp8 cuBLASLt cached-plan GEMM is BYTE-identical to the fresh-plan GEMM")
add_test(NAME test_ops_fp8_cutlass_plan_cache_rollback_off
  COMMAND test_ops_fp8_cutlass
    "--test-case=fp8 cuBLASLt cached-plan GEMM is BYTE-identical to the fresh-plan GEMM")
set_tests_properties(test_ops_fp8_cutlass_plan_cache_rollback_off PROPERTIES
  ENVIRONMENT "VT_FP8_PLAN_CACHE=0")
# Opt-in arm: PERF-FP8-ALPHA-FOLD's per-column alpha applied INSIDE the cuBLASLt
# epilogue (VT_FP8_ALPHA_VEC_EPILOGUE=1) versus the shipped two-launch form. This
# is the BINDING byte-exactness comparison. The plain test_ops_fp8_cutlass run has
# the toggle unset, where both sides execute the same fallback code and the case
# only proves that fallback is unchanged.
add_test(NAME test_ops_fp8_alpha_vec_epilogue_on
  COMMAND test_ops_fp8_cutlass
    "--test-case=fp8 cuBLASLt per-column alpha is BYTE-identical to the two-launch form")
set_tests_properties(test_ops_fp8_alpha_vec_epilogue_on PROPERTIES
  ENVIRONMENT "VT_FP8_ALPHA_VEC_EPILOGUE=1")
vllm_cpp_add_test(test_ops_moe_grouped vt/test_ops_moe_grouped.cpp)
# Reaches the CUDA-free VT_MOE_ROUTER_WARP selector header under src/, through
# the shared test-only pin tests/vt/moe_router_warp_env.h: the parallel-vs-serial
# router case PINS that toggle both ways and ASSERTS the pinned state, so it can
# no longer degenerate to block-vs-serial when the ambient environment says 0
# (spec moe-router-topk-single-warp.md §9, #378). Same include pattern as
# test_moe_router_warp_map above, which executes that pin without a GPU.
target_include_directories(test_ops_moe_grouped PRIVATE ${CMAKE_SOURCE_DIR}/src)
# Wave TIEBREAK (#2586): whether the CUDA top-k REALISES the lowest-index
# tie-break it declares, at E=512 k=10 -- the geometry qwen4_exp routes, which
# MoeRouterWarpValuesPerThread excludes from the warp kernel and which no other
# case executes. Reaches the same src/ selector header through the shared pin
# tests/vt/moe_router_warp_env.h, so it needs the same src/ include path.
vllm_cpp_add_test(test_moe_router_tie_stability vt/test_moe_router_tie_stability.cpp)
target_include_directories(test_moe_router_tie_stability PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_ops_moe_grouped_bf16 vt/test_ops_moe_grouped_bf16.cpp)
vllm_cpp_add_test(test_rocm_moe_grouped_bf16 vt/test_rocm_moe_grouped_bf16.cpp)
vllm_cpp_add_test(test_rocm_moe_upstream vt/test_rocm_moe_upstream.cpp)
vllm_cpp_add_test(test_moe_bf16_native_contract vt/test_moe_bf16_native_contract.cpp)
# BACKEND-ROCM-BF16-MOE (#3094): the row's CPU gate on the production token
# gate's reference-set predicate. The predicate is pure host code in
# tests/support/rocm_moe_reference_set.h, so this case needs no device and no
# oracle file, while tests/vllm/models/test_rocm_moe_bf16.cpp applies the same
# predicate to the captured records.
vllm_cpp_add_test(test_rocm_moe_reference_set support/test_rocm_moe_reference_set.cpp)
if(VLLM_CPP_HIP)
  foreach(moe_test IN ITEMS test_rocm_moe_grouped_bf16 test_rocm_moe_upstream
                            test_moe_bf16_native_contract)
    target_compile_definitions(${moe_test} PRIVATE __HIP_PLATFORM_AMD__=1)
  endforeach()
endif()
vllm_cpp_add_test(test_ops_moe_grouped_bf16_gate_up_silu
                  vt/test_ops_moe_grouped_bf16_gate_up_silu.cpp)
vllm_cpp_add_test(test_ops_moe_nongated_relu2 vt/test_ops_moe_nongated_relu2.cpp)
vllm_cpp_add_test(test_ops_rmsnorm vt/test_ops_rmsnorm.cpp)
vllm_cpp_add_test(test_ops_residual_rmsnorm vt/test_ops_residual_rmsnorm.cpp)
if(VLLM_CPP_HIP)
  target_compile_definitions(test_ops_residual_rmsnorm PRIVATE __HIP_PLATFORM_AMD__=1)
  target_compile_definitions(test_rocm_moe_bf16 PRIVATE __HIP_PLATFORM_AMD__=1)
  # The head-boundary replay names the ROCm device itself (hipGetDevice /
  # hipSetDevice around each queue), exactly as the residual siblings do.
  target_compile_definitions(test_qwen3_moe_lm_head_bf16 PRIVATE __HIP_PLATFORM_AMD__=1)
endif()
vllm_cpp_add_test(test_ops_rmsnorm_weight_dtype vt/test_ops_rmsnorm_weight_dtype.cpp)
vllm_cpp_add_test(test_ops_rmsnorm_quant_fp8_weight_dtype
                  vt/test_ops_rmsnorm_quant_fp8_weight_dtype.cpp)
vllm_cpp_add_test(test_ops_fused_chain vt/test_ops_fused_chain.cpp)
vllm_cpp_add_test(test_ops_layernorm vt/test_ops_layernorm.cpp)
vllm_cpp_add_test(test_fused_chain_additivity vt/test_fused_chain_additivity.cpp)
vllm_cpp_add_test(test_ops_activation vt/test_ops_activation.cpp)
vllm_cpp_add_test(test_ops_rope vt/test_ops_rope.cpp)
vllm_cpp_add_test(test_ops_rope_cache vt/test_ops_rope_cache.cpp)
vllm_cpp_add_test(test_ops_fused_norm_rope vt/test_ops_fused_norm_rope.cpp)
vllm_cpp_add_test(test_compiled_gemma vt/test_compiled_gemma.cpp)
vllm_cpp_add_test(test_ops_gdn vt/test_ops_gdn.cpp)
target_include_directories(test_ops_gdn PRIVATE ${CMAKE_SOURCE_DIR}/src)
if(VLLM_CPP_TRITON)
  # A separate process guarantees this race test reaches the generated module
  # loaders before another gate-shape GDN case can warm them.
  add_test(NAME test_ops_gdn_aot_concurrent_first_load
    COMMAND test_ops_gdn
      "--test-case=CUDA gdn Triton AOT concurrent first load is safe across two queues")
endif()
vllm_cpp_add_test(test_ops_kda_recurrence vt/test_ops_kda_recurrence.cpp)
target_include_directories(test_ops_kda_recurrence PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_ops_kda_chunk_prefill vt/test_ops_kda_chunk_prefill.cpp)
target_include_directories(test_ops_kda_chunk_prefill PRIVATE ${CMAKE_SOURCE_DIR}/src)
# Mamba2 / SSD host references (.agents/specs/mamba2-ssd.md W1, #496). Three
# sibling ops of the GDN/KDA lane above, gated against a sequential
# double-precision reference and across chunk sizes.
vllm_cpp_add_test(test_ops_mamba2_ssd vt/test_ops_mamba2_ssd.cpp)
target_include_directories(test_ops_mamba2_ssd PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_ops_mamba2_state_update vt/test_ops_mamba2_state_update.cpp)
target_include_directories(test_ops_mamba2_state_update PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_ops_mamba2_gated_norm vt/test_ops_mamba2_gated_norm.cpp)
target_include_directories(test_ops_mamba2_gated_norm PRIVATE ${CMAKE_SOURCE_DIR}/src)
# vt::RmsNormGroup — the UNGATED per-group RMS norm (MODEL-MM-QWEN4-EXP W5d-1,
# #2249 item 1). Sibling of the two above: same grouped reduction, no gate. Gated
# against the lane-pinned transformers goldens the qwen4_exp gated-residual arm
# already answers to, so both arms are held to ONE oracle.
vllm_cpp_add_test(test_ops_rms_norm_group vt/test_ops_rms_norm_group.cpp)
target_include_directories(test_ops_rms_norm_group PRIVATE ${CMAKE_SOURCE_DIR}/src)
# The CUDA arm of the same op (MODEL-MM-QWEN4-EXP W6-CUDA-B). A SEPARATE suite
# rather than cases in the file above, because that one runs on every host and
# this one skips LOUDLY without a CUDA backend; mixing the two would make a
# CPU-only run print a skip for a case that has nothing to do with a device.
# Held to the SAME goldens AND to the CPU arm BITWISE — the op has no
# transcendental, so anything but byte equality is a defect. Compiled
# unconditionally, like every other CUDA suite here.
vllm_cpp_add_test(test_ops_rms_norm_group_cuda vt/test_ops_rms_norm_group_cuda.cpp)
target_include_directories(test_ops_rms_norm_group_cuda PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_ops_moe vt/test_ops_moe.cpp)
vllm_cpp_add_test(test_ops_glue vt/test_ops_glue.cpp)
vllm_cpp_add_test(test_ops_attn_preamble vt/test_ops_attn_preamble.cpp)
vllm_cpp_add_test(test_ops_attention vt/test_ops_attention.cpp)
# vt::AttentionDenseFa2 (multimodal-speed §17, #432): fast-path agreement with the
# byte-exact scalar kernel, full-key-range coverage, fall-through totality, and the
# VT_FA2_DENSE A/B knob. CUDA-only content; skips cleanly on a CPU box.
vllm_cpp_add_test(test_ops_attention_dense_fa2 vt/test_ops_attention_dense_fa2.cpp)
# vt::AttentionCross, DIRECTLY, on every backend that registers it. Its own
# target rather than more cases in test_ops_attention or test_ltx2_device, both
# of which carry recorded baselines. It exists because the CUDA kernel's whole
# flash-tiling machinery (cross-tile rescale, multi-block launch, npl>1 head
# split, GQA broadcast, ChooseTileCols' halving) is unreachable from every
# LTX-2.5 call geometry, which is tiles=1/npl=1/nblk=1/Hq==Hkv throughout.
vllm_cpp_add_test(test_ops_attention_cross vt/test_ops_attention_cross.cpp)
vllm_cpp_add_test(test_ops_attention_elem_dispatch vt/test_ops_attention_elem_dispatch.cpp)
vllm_cpp_add_test(test_ops_rmsnorm_elem_dispatch vt/test_ops_rmsnorm_elem_dispatch.cpp)
target_include_directories(test_ops_rmsnorm_elem_dispatch PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_ops_dflash_block_attn vt/test_ops_dflash_block_attn.cpp)
vllm_cpp_add_test(test_dflash_attn_grid vt/test_dflash_attn_grid.cpp)
vllm_cpp_add_test(test_dflash_attn_mask vt/test_dflash_attn_mask.cpp)
vllm_cpp_add_test(test_ops_dflash_paged_block_attn vt/test_ops_dflash_paged_block_attn.cpp)
# SPEC-DFLASH2 W2 (#1314): the DFlash2 grouped dynamic depthwise convolution,
# gated against upstream's own sequential reference at block 5 (modulo arm), 8
# and 16 (the two published checkpoints) plus CUDA==CPU bit-identity.
vllm_cpp_add_test(test_ops_dflash2_grouped_conv vt/test_ops_dflash2_grouped_conv.cpp)
vllm_cpp_add_test(test_ops_topk_values_indices vt/test_ops_topk_values_indices.cpp)
vllm_cpp_add_test(test_ops_dflash2_selector_edges vt/test_ops_dflash2_selector_edges.cpp)
# SPEC-DFLASH2 W4 (#1314): the candidate selector's PATH WALK -- one program per
# request with the slot-to-slot dependency as a loop inside it, gated against
# upstream's own `_selector_walk_kernel` control flow plus CUDA==CPU bit-exactness.
vllm_cpp_add_test(test_ops_dflash2_path_walk vt/test_ops_dflash2_path_walk.cpp)
vllm_cpp_add_test(test_ops_reshape_cache vt/test_ops_reshape_cache.cpp)
# KV-FP8 W1: fp8 KV-cache store (ReshapeAndCacheFp8) + paged-attention read
# dequant + the ParseCacheDType config wiring.
vllm_cpp_add_test(test_ops_fp8_kv_cache vt/test_ops_fp8_kv_cache.cpp)
# KV-FP8 W2 (#1593): the CUDA fp8-e4m3 KV store + the fp8 dequant on the CUDA
# paged-attention read, gated for parity against the W1 CPU reference. The
# device cases skip with a MESSAGE when no CUDA backend is present; the
# provider-routing case runs on every leg.
vllm_cpp_add_test(test_cuda_fp8_kv_cache vt/test_cuda_fp8_kv_cache.cpp)
# MLA campaign W3: the compressed-latent cache write + the grouped-topk router.
vllm_cpp_add_test(test_ops_mla_cache vt/test_ops_mla_cache.cpp)
# KV-DSV4-MULTICACHE W8 slices 2 and 3 (#2455): the fp8_ds_mla STORE and READ
# over DeepSeek-V4's REGION-SPLIT rank-2 byte page, which no rank-3 cache op can
# express. Byte-exact against a poison-filled block, all host-side.
vllm_cpp_add_test(test_ops_ds_mla_cache vt/test_ops_ds_mla_cache.cpp)
# MLA campaign W4: the two-stage split-KV MQA decode over the compressed latent.
vllm_cpp_add_test(test_ops_mla_attn vt/test_ops_mla_attn.cpp)
# dots3-note W4b-3c (#699): the DSA "Lightning Indexer" selection pair, gated
# against the W3 host reference `vllm::deepseek_v4::DsaIndexerLogits` /
# `DsaTopkSelect` that this op family lifts onto `vt::Tensor`.
vllm_cpp_add_test(test_ops_dsa_indexer vt/test_ops_dsa_indexer.cpp)
# MLA campaign W5: the FLASH_ATTN MLA prefill (QK 192 / V 128) ...
vllm_cpp_add_test(test_ops_mla_prefill vt/test_ops_mla_prefill.cpp)
# ... and the workspace-bounded chunked-context loop (gather + LSE merge).
vllm_cpp_add_test(test_ops_mla_chunked_context vt/test_ops_mla_chunked_context.cpp)
# MLA campaign W6: the two primitives weight ABSORPTION is built out of —
# vt::BatchedMatmul (torch.bmm) and vt::ConcatMlaNopeRope (concat_mla_q /
# _concat_k_nope_k_pe).
vllm_cpp_add_test(test_ops_mla_absorb vt/test_ops_mla_absorb.cpp)
# MLA campaign W6: the MLA attention BLOCK — projections, both q_lora branches,
# decoupled RoPE, the load-time kv_b_proj -> W_UK/W_UV split, and the
# prefill-MHA / decode-MQA dispatch, incl. the absorbed-vs-unabsorbed proof.
vllm_cpp_add_test(test_mla_attention_block
  vllm/model_executor/layers/attention/test_mla_attention_block.cpp)
vllm_cpp_add_test(test_ops_moe_router_grouped vt/test_ops_moe_router_grouped.cpp)
# Conformer / FastConformer audio-encoder kernels (spike
# .agents/specs/parakeet-conformer-encoder.md rows P1/P2/P3). Each gate is
# BYTE-IDENTITY against an INDEPENDENT in-test scalar reference (never against
# the kernel), across dtype x shape x thread count, matching the
# test_ops_matmul_elem.cpp discipline. The relpos-attention reference performs
# the upstream `_rel_shift` LITERALLY, so byte-identity is what proves the
# kernel's closed-form index. They reach src/ for the threadpool test hook
# (Threadpool::SwapForTesting) used by the determinism cases. CPU-only.
vllm_cpp_add_test(test_ops_conv2d vt/test_ops_conv2d.cpp)
target_include_directories(test_ops_conv2d PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_ops_conv1d_depthwise vt/test_ops_conv1d_depthwise.cpp)
target_include_directories(test_ops_conv1d_depthwise PRIVATE ${CMAKE_SOURCE_DIR}/src)
# BigVGAN / DAC vocoder 1-D convolutions (#672): vt::Conv1d + vt::ConvTranspose1d
# against a verbatim copy of the pre-op vocoder1d host loop, plus a CPU-vs-CUDA
# BYTE-equality arm that skips loudly on a CPU box. The four consumers' own
# end-to-end bit-identity gate is tests/vllm/models/test_host_parallel.cpp.
vllm_cpp_add_test(test_ops_conv1d_general vt/test_ops_conv1d_general.cpp)
target_include_directories(test_ops_conv1d_general PRIVATE ${CMAKE_SOURCE_DIR}/src)
# General 3-D convolution (LTX25-DEVICE-RESIDENCY W5, #1007): vt::Conv3d against
# an independent in-test scalar reference, plus the case that proves its
# per-input-channel partial is a DIFFERENT number from kConv2d's flat
# accumulator. The LTX-2.5 decode's own byte-identity gate is
# tests/vllm/models/test_ltx2_vae.cpp.
vllm_cpp_add_test(test_ops_conv3d vt/test_ops_conv3d.cpp)
target_include_directories(test_ops_conv3d PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_ops_attn_relpos vt/test_ops_attn_relpos.cpp)
target_include_directories(test_ops_attn_relpos PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_ops_paged_attn vt/test_ops_paged_attn.cpp)
# SPEC-DFLASH2 W10 (#1857): the host lane-classification seam of the CUDA
# paged-attention dispatch (include/vt/paged_attn_route.h).
vllm_cpp_add_test(test_paged_attn_route vt/test_paged_attn_route.cpp)
# PERF-CPU-ATTN-DTYPE: the hoisted dtype dispatch vs the pre-change per-element
# switch, memcmp over every dtype the switch handled (spec §3, issue #391).
vllm_cpp_add_test(test_ops_paged_attn_dtype vt/test_ops_paged_attn_dtype.cpp)
vllm_cpp_add_test(test_ops_sample vt/test_ops_sample.cpp)
if(UNIX)
  add_test(NAME test_rocm_sample_scratch
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/tests/scripts/test_rocm_sample_scratch.py")
  set_tests_properties(test_rocm_sample_scratch PROPERTIES
    ENVIRONMENT "CXX=${CMAKE_CXX_COMPILER}")
endif()
vllm_cpp_add_test(test_gemma4_rocm_fp8_seams vt/test_gemma4_rocm_fp8_seams.cpp)
vllm_cpp_add_test(test_ops_getblas_dualslot vt/test_ops_getblas_dualslot.cpp)
target_compile_definitions(test_ops_getblas_dualslot PRIVATE
  VLLM_CPP_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(VLLM_CPP_HIP)
  # Separate HIP target: calls production GetBlas. Missing devices → exit 77.
  # .cpp would compile as host CXX and miss __HIP_PLATFORM_AMD__.
  set_source_files_properties(vt/test_ops_getblas_product.cpp PROPERTIES LANGUAGE HIP)
  vllm_cpp_add_test(test_ops_getblas_product vt/test_ops_getblas_product.cpp)
endif()
vllm_cpp_add_test(test_ops_penalties vt/test_ops_penalties.cpp)

vllm_cpp_add_test(test_npy parity/test_npy.cpp)
target_compile_definitions(test_npy PRIVATE
  VLLM_CPP_GOLDENS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/parity/goldens")

vllm_cpp_add_test(test_tokenizer_parity vllm/test_tokenizer_parity.cpp)
target_compile_definitions(test_tokenizer_parity PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")

# MLA CAMPAIGN W8 — the DeepSeek family's tokenizer parity gate. DeepSeek's
# pre_tokenizer is a seven-stage HF `Sequence` PIPELINE over enumerated
# codepoint ranges (not an alternation regex like every family before it), so
# it is MEASURED against HF `tokenizers` over a stage-stress corpus rather than
# argued. This is what unblocked the DeepSeek-V2-Lite paged-engine gate.
vllm_cpp_add_test(test_tokenizer_parity_deepseek vllm/test_tokenizer_parity_deepseek.cpp)
target_compile_definitions(test_tokenizer_parity_deepseek PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")

# LOAD-SENTENCEPIECE — the SentencePiece (Metaspace + byte-fallback) BPE
# tokenizer parity gate. Mistral-7B-v0.3's tokenizer.json is the first of this
# family (▁ metaspace + `<0xNN>` byte-fallback vocab + BOS-prepending
# TemplateProcessing). MEASURED byte-exact against HF `tokenizers` (= vLLM
# 0.25.0's backend) over a Metaspace/byte-fallback/special-token corpus. This
# is what unblocks the Mistral paged-engine SACRED gate.
vllm_cpp_add_test(test_tokenizer_parity_mistral vllm/test_tokenizer_parity_mistral.cpp)
target_compile_definitions(test_tokenizer_parity_mistral PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")

# TOK-LLAMA4-PRE (issue #347) — the GPT-4o / o200k tokenizer parity gate. This
# family is what a GGUF tags `tokenizer.ggml.pre = "llama4"` (Muse Glimmer 30B).
# It shares kLlama3's `\p{N}{1,3}` digit grouping but splits CamelCase and keeps
# the contraction as a word suffix, so an alias would mistokenize silently —
# only comparing IDS against HF `tokenizers` catches that, which is what this
# gate does.
vllm_cpp_add_test(test_tokenizer_parity_gpt4o vllm/test_tokenizer_parity_gpt4o.cpp)
target_compile_definitions(test_tokenizer_parity_gpt4o PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")

# TOK-DEEPSEEK-V3-PRE (issue #1924) — the DeepSeek-V3 / V4-Flash tokenizer
# parity gate. A FOUR-stage `Sequence` (three `Split(Isolated)` then
# `ByteLevel(use_regex=false)`) that neither the generic single-Split walk nor
# the seven-stage DeepSeek-V2 recognizer could read, so `vllm-server` refused
# every DeepSeek-V4-Flash safetensors checkpoint before a weight byte. The
# golden `tokenizer.json` is a byte-for-byte copy of the checkpoint's own, so
# the gate cannot pass on a fixture that reproduces a misreading of the shape.
vllm_cpp_add_test(test_tokenizer_parity_deepseek_v3 vllm/test_tokenizer_parity_deepseek_v3.cpp)
target_compile_definitions(test_tokenizer_parity_deepseek_v3 PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")

vllm_cpp_add_test(test_op_parity parity/test_op_parity.cpp)
target_compile_definitions(test_op_parity PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_op_parity PRIVATE ${CMAKE_SOURCE_DIR}/tests/parity)

# Paged-engine 35B greedy acceptance gate (checkpoint-gated, dgx-only; skips on
# CPU/CI when the real snapshot is absent). Counterpart to RunQwen36Logits.
vllm_cpp_add_test(test_qwen36_paged_engine parity/test_qwen36_paged_engine.cpp)
target_compile_definitions(test_qwen36_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen36_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# ROW-SERVE-ASYNC-LLM — the ASYNC-serving 35B greedy token-exact gate (the
# depth-2 AsyncLLM counterpart of the SYNC SACRED gate above). Reproduces the
# async batch-1 token-0 degeneration P0 on the mirror-OFF default and proves the
# device-resident mirror fixes it (checkpoint-gated, dgx-only; skips on CPU/CI).
vllm_cpp_add_test(test_qwen36_async_serving parity/test_qwen36_async_serving.cpp)
target_compile_definitions(test_qwen36_async_serving PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen36_async_serving PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# ADDITIVE-MODEL W4 — the SACRED token-exact gate: Qwen3-0.6B (dense) greedy
# decode vs the vLLM 0.25.0 oracle, through the full paged engine
# (checkpoint-gated, dgx-only; skips on CPU/CI when the snapshot is absent).
vllm_cpp_add_test(test_qwen3_paged_engine parity/test_qwen3_paged_engine.cpp)
target_compile_definitions(test_qwen3_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen3_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# ROCm M4 — the GDN sibling of the Qwen3-dense gate: Qwen3.5-0.8B greedy decode
# vs the pinned vLLM-ROCm oracle (the ONLY capture of this model — no dgx
# capture exists; the base golden pair is ROCm-captured on gfx1100).
# Checkpoint-gated: skips loudly when the HF snapshot is absent.
vllm_cpp_add_test(test_qwen35_paged_engine parity/test_qwen35_paged_engine.cpp)
target_compile_definitions(test_qwen35_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen35_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)
add_test(NAME test_qwen35_paged_engine_prerequisites
  COMMAND ${CMAKE_COMMAND}
    "-DTEST_EXE=$<TARGET_FILE:test_qwen35_paged_engine>"
    "-DPROBE_ROOT=${CMAKE_CURRENT_BINARY_DIR}/qwen35_paged_engine_prerequisites"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/parity/test_qwen35_paged_engine_prerequisites.cmake")

# ROW-SERVE-ASYNC-DENSE-MIRROR — the classic-dense async-serving greedy gate:
# Qwen3-0.6B/4B async batch-1 + concurrency reproduce the race-free SYNC engine
# continuation token-for-token. RED on VT_ASYNC_DEVICE_MIRROR=0 (dense EmbedInto
# races the combine), GREEN on the default fix (checkpoint-gated, dgx-only; skips
# on CPU/CI when the snapshot is absent).
vllm_cpp_add_test(test_qwen3_dense_async_serving parity/test_qwen3_dense_async_serving.cpp)
target_compile_definitions(test_qwen3_dense_async_serving PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen3_dense_async_serving PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# ROAD-V1-D4-APC W3 — the cache-ON automatic-prefix-caching e2e gate on Qwen3-4B
# (dense, full-attention, APC-eligible): APC-ON == APC-OFF token-exact (RED-first
# cache-bug catch) + non-zero hit counters + vLLM K-run membership + prefill/TTFT
# drop (checkpoint-gated, dgx-only; skips on CPU/CI when the snapshot is absent).
vllm_cpp_add_test(test_qwen3_apc_e2e parity/test_qwen3_apc_e2e.cpp)
target_compile_definitions(test_qwen3_apc_e2e PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen3_apc_e2e PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# GLM-4-9B-0414 (Glm4ForCausalLM) dense SACRED gate — first GLM-family model (G2).
vllm_cpp_add_test(test_glm4_paged_engine parity/test_glm4_paged_engine.cpp)
target_compile_definitions(test_glm4_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_glm4_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# OLMo-2-0425-1B (Olmo2ForCausalLM) dense SACRED gate — first OLMo-family model (W4).
vllm_cpp_add_test(test_olmo2_paged_engine parity/test_olmo2_paged_engine.cpp)
target_compile_definitions(test_olmo2_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_olmo2_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# Gemma-4 E4B (Gemma4ForConditionalGeneration) text STRICT gate — first Gemma-4
# model; RED->GREEN proof of the G1b runner per-layer-KV head_dim change.
vllm_cpp_add_test(test_gemma4_paged_engine parity/test_gemma4_paged_engine.cpp)
target_compile_definitions(test_gemma4_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_gemma4_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_granite_paged_engine parity/test_granite_paged_engine.cpp)
target_compile_definitions(test_granite_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_granite_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_minicpm_paged_engine parity/test_minicpm_paged_engine.cpp)
target_compile_definitions(test_minicpm_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_minicpm_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_minicpm3_paged_engine parity/test_minicpm3_paged_engine.cpp)
target_compile_definitions(test_minicpm3_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_minicpm3_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_phi3_paged_engine parity/test_phi3_paged_engine.cpp)
target_compile_definitions(test_phi3_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_phi3_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# Muse Glimmer 30B Q4_K_M GGUF gate vs the registered llama.cpp oracle (b10451)
# on the same file (.agents/specs/muse-glimmer-parity.md). Checkpoint-gated on
# VLLM_MUSE_GGUF_PARITY.
vllm_cpp_add_test(test_muse_glimmer_gguf_paged_engine parity/test_muse_glimmer_gguf_paged_engine.cpp)
target_compile_definitions(test_muse_glimmer_gguf_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_muse_glimmer_gguf_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_olmo3_paged_engine parity/test_olmo3_paged_engine.cpp)
target_compile_definitions(test_olmo3_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_olmo3_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_stablelm_paged_engine parity/test_stablelm_paged_engine.cpp)
target_compile_definitions(test_stablelm_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_stablelm_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_internlm2_paged_engine parity/test_internlm2_paged_engine.cpp)
target_compile_definitions(test_internlm2_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_internlm2_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_internlm3_paged_engine parity/test_internlm3_paged_engine.cpp)
target_compile_definitions(test_internlm3_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_internlm3_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_yi_paged_engine parity/test_yi_paged_engine.cpp)
target_compile_definitions(test_yi_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_yi_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_commandr_paged_engine parity/test_commandr_paged_engine.cpp)
target_compile_definitions(test_commandr_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_commandr_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_phi_paged_engine parity/test_phi_paged_engine.cpp)
target_compile_definitions(test_phi_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_phi_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# GGUF 35B greedy acceptance gate (checkpoint-gated, dgx-only; skips when the
# APEX k-quant GGUFs are absent). Loads the REAL .gguf single-file path
# (metadata + embedded vocab + k-quant dequant) through the paged engine and
# asserts greedy token-for-token vs the pinned same-file llama.cpp oracle.
vllm_cpp_add_test(test_qwen36_gguf_engine parity/test_qwen36_gguf_engine.cpp)
target_compile_definitions(test_qwen36_gguf_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen36_gguf_engine PRIVATE ${CMAKE_SOURCE_DIR}/tests/parity)

# `QUANT-GGUF-NVFP4` column C: native fp4 compute out of a GGUF container.
# Asset-gated (VLLM_NVFP4_GGUF, optionally VLLM_NVFP4_ST_DIR); compiles and links
# on CPU/CI but only RUNS where the real 27B NVFP4 files exist.
vllm_cpp_add_test(test_qwen27_gguf_nvfp4_compute
  parity/test_qwen27_gguf_nvfp4_compute.cpp)

# `QUANT-GGUF-NVFP4` column C, MoE arm: the 35B A3B stacked-expert residency.
# Asset-gated (VLLM_NVFP4_MOE_GGUF, optionally VLLM_NVFP4_MOE_ST_DIR); compiles
# and links on CPU/CI but only RUNS where the real 35B NVFP4 files exist.
vllm_cpp_add_test(test_qwen36_gguf_nvfp4_compute
  parity/test_qwen36_gguf_nvfp4_compute.cpp)

# Dense 27B W4A4 greedy acceptance gate (checkpoint-gated, dgx-only; skips on
# CPU/CI when the real snapshot is absent). Counterpart to the 35B gate; asserts
# the paged dense engine reproduces the pip-vLLM oracle greedy continuation
# token-for-token via the fp4-resident W4A4 GEMM. See qwen27b-w4a4-notes.md §5.
vllm_cpp_add_test(test_qwen27_paged_engine parity/test_qwen27_paged_engine.cpp)
target_compile_definitions(test_qwen27_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen27_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# `GATE-27B-FP8-TOWER-GOLDEN` (issue #466): the SECOND dense 27B arm, pinned to
# nvidia/Qwen3.6-27B-NVFP4@0893e160 (`modelopt_mixed` — W4A16 NVFP4 MLP + FP8
# W8A8 tower + NVFP4 head). The gate above pins unsloth@890bdef7, whose GDN
# input projections are BF16, so every fp8 branch is dead code there and that
# gate can never fail for an fp8 defect. This arm asserts the fp8 GDN
# input-projection dispatch contract UNCONDITIONALLY — a zero count is a
# failure, not a skipped block. Checkpoint-gated, dgx-only; on CPU/CI it refuses
# loudly instead of reporting a vacuous pass.
vllm_cpp_add_test(test_qwen27n_fp8_tower_paged_engine
  parity/test_qwen27n_fp8_tower_paged_engine.cpp)
target_compile_definitions(test_qwen27n_fp8_tower_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen27n_fp8_tower_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# `PERF-27B-LMHEAD-FP4` (issue #213): the ModelOpt NVFP4 lm_head stays PACKED.
# Synthetic (no checkpoint, no GPU) BECAUSE test_qwen27_paged_engine runs the
# unsloth @890bdef7 snapshot, whose head is BF16 — that gate cannot see this path.
vllm_cpp_add_test(test_qwen27_dense_lmhead_fp4
  parity/test_qwen27_dense_lmhead_fp4.cpp)

# SPEC-DFLASH D2 — the DFlash draft-forward parity gate vs the dumped vLLM DFlash
# draft reference (fc combine + context-free block forward per-stage hidden +
# proposed ids). GPU + z-lab-draft + NVFP4-target + fixture gated (skips on CPU/CI).
vllm_cpp_add_test(test_qwen3_dflash_draft_parity parity/test_qwen3_dflash_draft_parity.cpp)
target_compile_definitions(test_qwen3_dflash_draft_parity PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen3_dflash_draft_parity PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# SPEC-DFLASH D3 (DF-DRAFT-KV-PREP) — the decisive GPU numeric-parity gate vs the
# REAL loaded vLLM draft: prepare_dflash_inputs INTEGER bit-exact (vs vLLM's actual
# Triton kernel), context-KV precompute per-layer rel-L2 within envelope, block
# proposal STRICT-or-near-tie. GPU + z-lab-draft + NVFP4-target + fixture gated.
vllm_cpp_add_test(test_qwen3_dflash_kvprep_parity parity/test_qwen3_dflash_kvprep_parity.cpp)
target_compile_definitions(test_qwen3_dflash_kvprep_parity PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen3_dflash_kvprep_parity PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# SPEC-MTP I5d — the 27B k=1 MTP speculative-decode three-way identity gate
# (our spec-ON == our spec-OFF == vLLM greedy) + measured nonzero acceptance.
vllm_cpp_add_test(test_qwen35_gguf_spec_decode parity/test_qwen35_gguf_spec_decode.cpp)
vllm_cpp_add_test(test_qwen27_spec_decode parity/test_qwen27_spec_decode.cpp)
target_compile_definitions(test_qwen27_spec_decode PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen27_spec_decode PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# SPEC-DFLASH D5 — the 27B DFlash block-diffusion e2e correctness gate: our
# DFlash-ON == the committed vLLM-DFlash-ON golden (STRICT mode-matched) + measured
# nonzero acceptance ~ vLLM's (the dead-drafter trap). dgx-only (needs the NVFP4
# target + the z-lab draft + the CUDA spec kernels); loud SKIP on the CPU dev box.
vllm_cpp_add_test(test_qwen27_dflash_spec_decode
  parity/test_qwen27_dflash_spec_decode.cpp)
target_compile_definitions(test_qwen27_dflash_spec_decode PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen27_dflash_spec_decode PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# SPEC-DFLASH2 W6 — the DFlash2 e2e gate (#1314): G2 draft-token identity and
# G3 SAME-TRAJECTORY acceptance against the beyond-pin oracle
# (vllm-project/vllm#52816 @ 66e5414c6, TRITON_ATTN). dgx-only; loud SKIP
# everywhere else, naming the two checkpoints and the golden it needs.
vllm_cpp_add_test(test_qwen38_dflash2_spec_decode
  parity/test_qwen38_dflash2_spec_decode.cpp)
target_compile_definitions(test_qwen38_dflash2_spec_decode PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen38_dflash2_spec_decode PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# SPEC-MTP M-mtp-2 — the 35B MoE-MTP k=1 speculative-decode three-way identity
# gate (Qwen3_5MoeMTP: our spec-ON == our spec-OFF == vLLM greedy) + measured
# nonzero acceptance. Mirror of the 27B gate at 35B (MoE MTP draft head).
vllm_cpp_add_test(test_qwen36_spec_decode parity/test_qwen36_spec_decode.cpp)
target_compile_definitions(test_qwen36_spec_decode PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen36_spec_decode PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# GATE-PIN-UNPINNED-SNAPSHOTS (#471). The resolver gate: CPU-only, needs no
# checkpoint, synthesises its own two-revision cache. It is the direct proof that
# a pinned accessor selects rather than substitutes, and unlike every gate it
# protects, it runs everywhere.
vllm_cpp_add_test(test_hf_snapshot_pinning parity/test_hf_snapshot_pinning.cpp)
target_include_directories(test_hf_snapshot_pinning PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity)

# SPEC-NGRAM (ROAD-V1-D3) — the 27B draft-free ngram e2e gate: our ngram-ON ==
# the committed vLLM-ngram-ON golden (STRICT, exactness-preserving greedy) +
# measured nonzero acceptance ~ vLLM's. dgx-only; loud SKIP on the CPU dev box.
vllm_cpp_add_test(test_qwen27_ngram_spec_decode
  parity/test_qwen27_ngram_spec_decode.cpp)
target_compile_definitions(test_qwen27_ngram_spec_decode PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen27_ngram_spec_decode PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_qwen27_spec_decode_concurrent
  parity/test_qwen27_spec_decode_concurrent.cpp)
target_include_directories(test_qwen27_spec_decode_concurrent PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# ADDITIVE-MODEL W4 (Qwen3-Coder-30B-A3B, Qwen3MoeForCausalLM) — the SACRED
# full-attention BF16 MoE greedy correctness gate: drives a compact prompt
# battery through the paged engine and checks per-(prompt,pos) set membership vs
# the vLLM 0.25.0 oracle's K-run greedy set (STRICT where vLLM is deterministic).
# Checkpoint-gated, dgx-only; skips on CPU/CI when the snapshot/goldens are absent.
vllm_cpp_add_test(test_moe_two_engines vllm/models/test_moe_two_engines.cpp)
target_include_directories(test_moe_two_engines PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_qwen3coder_paged_engine vllm/models/test_qwen3coder_paged_engine.cpp)
target_compile_definitions(test_qwen3coder_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen3coder_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# ADDITIVE-MODEL W2/W4 (OPT-125m, OPTForCausalLM) — the CROSS-FAMILY additivity
# canary: a host-only loader gate plus the SACRED greedy correctness gate, which
# is STRICT token-exact because the vLLM 0.25.0 oracle measured DETERMINISTIC on
# this model (K=5, 0 multi-valued cells). Checkpoint-gated, dgx-only.
vllm_cpp_add_test(test_opt_load vllm/models/test_opt_load.cpp)
vllm_cpp_add_test(test_opt_paged_engine vllm/models/test_opt_paged_engine.cpp)
target_compile_definitions(test_opt_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_opt_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# KV-EXTERNAL-CACHE (LMCache): the connector-ON full-model OUTPUT-INVARIANCE +
# throughput gate. Real OPT-125m generation loop vs a live lmcache.v1.server;
# checkpoint- and server-gated (loud SKIP otherwise). Driven by
# scripts/lmcache/run_output_invariance.sh.
vllm_cpp_add_test(test_lmcache_output_invariance
  vllm/models/test_lmcache_output_invariance.cpp)

# ADDITIVE-MODEL W2/W4 (Llama-3.2-1B, LlamaForCausalLM) — the CROSS-FAMILY dense
# additivity bring-up: a host-only loader gate plus the SACRED near-tie-robust
# greedy correctness gate vs the vLLM 0.25.0 oracle (llama3 rope-scaling, no
# qk-norm, GQA 32/8, head_dim 64, tied lm_head). Checkpoint-gated, dgx-only.
vllm_cpp_add_test(test_llama_load vllm/models/test_llama_load.cpp)
target_include_directories(test_llama_load PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_llama_paged_engine parity/test_llama_paged_engine.cpp)
target_compile_definitions(test_llama_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_llama_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# ADDITIVE-MODEL W2/W4 (Mistral-7B-v0.3, MistralForCausalLM) — the FIFTH-family
# near-additive dense bring-up: a host-only loader gate plus the SACRED
# near-tie-robust greedy correctness gate vs the vLLM 0.25.0 oracle (PLAIN rope
# theta 1e6 / no scaling, no qk-norm, GQA 32/8, head_dim 128, UNTIED lm_head,
# sliding_window null). Checkpoint-gated, dgx-only.
vllm_cpp_add_test(test_mistral_load vllm/models/test_mistral_load.cpp)
target_include_directories(test_mistral_load PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_mistral_paged_engine parity/test_mistral_paged_engine.cpp)
target_compile_definitions(test_mistral_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_mistral_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# ADDITIVE-QUANT W2/W4 (Qwen3-32B-NVFP4A16, Qwen3ForCausalLM + compressed-tensors
# NVFP4A16 / W4A16) — the QUANT-SCHEME additivity row: the SAME already-token-exact
# dense forward with only the Linear storage/compute scheme changed. A host-only
# loader gate plus the SACRED greedy correctness gate, which is STRICT token-exact
# because the vLLM 0.25.0 oracle measured DETERMINISTIC on this checkpoint (K=5,
# 0 multi-valued cells). Checkpoint-gated, dgx-only.
vllm_cpp_add_test(test_qwen3_32b_nvfp4a16_load
  vllm/models/test_qwen3_32b_nvfp4a16_load.cpp)

vllm_cpp_add_test(test_qwen3_5_lm_head_dtypes
  vllm/models/test_qwen3_5_lm_head_dtypes.cpp)

# LOAD-MODELOPT-NVFP4-BORROW (#1647, .agents/specs/load-modelopt-nvfp4-borrow.md):
# WHERE a dense-loaded weight's bytes live, asked through the production loader
# `LoadQwen3_5Dense` over a REAL safetensors shard on disk — an in-memory fake
# resolver carries no `StTensor::mapping`, so `BorrowStTensorBytes` fails closed
# on it and the defect would read as already fixed. Hermetic and CPU-only: the
# shard is written to a temporary file per case, and the one platform shape a CPU
# tier cannot otherwise reach (unified memory that nevertheless STAGES) comes
# from a fake platform in the unused kXPU slot.
vllm_cpp_add_test(test_qwen3_5_dense_load_residency
  vllm/models/test_qwen3_5_dense_load_residency.cpp)

# QUANT-QWEN38-27B-NVFP4-ARM W4 (#821, .agents/specs/qwen38-27b-quant-arms.md):
# the compressed-tensors `mixed-precision` accounting + scheme-resolution gate
# for `unsloth/Qwen3.8-27B-NVFP4`. Hermetic — it reads the two COMMITTED
# manifests and the COMMITTED `config.json`, never a weight byte; the live
# re-verification arm is env-gated on VLLM_CPP_QWEN38_27B_NVFP4_DIR and SKIPS
# loudly without it. `src` is on the include path because the resolver under
# test is a model-private header under src/, not an include/vllm/ surface.
vllm_cpp_add_test(test_qwen38_27b_nvfp4_arm
  vllm/models/test_qwen38_27b_nvfp4_arm.cpp)
target_compile_definitions(test_qwen38_27b_nvfp4_arm PRIVATE
  QWEN38_27B_NVFP4_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/qwen38_27b_nvfp4")
target_include_directories(test_qwen38_27b_nvfp4_arm PRIVATE
  ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)

# QUANT-QWEN38-27B-NVFP4-ARM W5 (#821, #1574,
# .agents/specs/qwen38-27b-quant-arms.md): the ModelOpt `MIXED_PRECISION`
# accounting + config/tensor cross-check for
# `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`. A SECOND artifact of the same model,
# and a second file rather than a widening of the gate above: that one pins the
# 1968-name compressed-tensors set and this one pins a 2001-name ModelOpt set,
# and a gate that accepted both by loosening would measure neither. Hermetic —
# it reads the four COMMITTED manifests and the two COMMITTED config documents,
# never a weight byte; the live re-verification arm is env-gated on
# VLLM_CPP_QWEN38_27B_MODELOPT_MTP_DIR and SKIPS loudly without it. `src` is on
# the include path because the resolver under test is a model-private header
# under src/, not an include/vllm/ surface.
vllm_cpp_add_test(test_qwen38_27b_modelopt_mtp_arm
  vllm/models/test_qwen38_27b_modelopt_mtp_arm.cpp)
target_compile_definitions(test_qwen38_27b_modelopt_mtp_arm PRIVATE
  QWEN38_27B_MODELOPT_MTP_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/qwen38_27b_modelopt_mtp")
target_include_directories(test_qwen38_27b_modelopt_mtp_arm PRIVATE
  ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)

vllm_cpp_add_test(test_qwen3_32b_nvfp4a16_paged_engine
  vllm/models/test_qwen3_32b_nvfp4a16_paged_engine.cpp)
target_compile_definitions(test_qwen3_32b_nvfp4a16_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen3_32b_nvfp4a16_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# The registry type-confusion CLASS gate (#847, .agents/specs/registry-downcast-sweep.md).
# One case per SHAPE of the swept `static_cast<XLoadedModel&>(model)` site — const
# and non-const binding, `prepare` and `forward`, an alias family that shares one
# factory across two registered architectures, and the one inline-call site that a
# uniform search-and-replace over the other 33 would step past. Every case enters
# through the production seam (`reg.factory->forward` / `->prepare`, what
# `ModelRegistry::Forward` calls) and refuses before any weight is touched, so it is
# CPU-only and needs no checkpoint.
vllm_cpp_add_test(test_registry_downcast_refusal
  vllm/models/test_registry_downcast_refusal.cpp)

# Qwen4-Exp W4 -- Qwen Sparse Attention (#1991, .agents/specs/qwen4-exp-flash-next.md).
# Gates the indexer, the side-cache sizing and the GATHER consumer against goldens
# captured by RUNNING transformers 5.16.0, this row's accepted ALGORITHM lane pin
# (vLLM registers no `qwen4_exp` at 6a5e8f5979, so the primary oracle has nothing
# to run). Two oracles need no weights: sub-budget bit-identity with dense
# attention, and index-set equality with the reference above the budget. One case
# is NOT a correctness gate at all -- `keys_visited` is what separates the gather
# from a mask, and a mask-only QSA would pass every value comparison here.
# CPU-only, no checkpoint, no GPU, no speed claim.
vllm_cpp_add_test(test_qwen4_exp_qsa vllm/models/test_qwen4_exp_qsa.cpp)
# qwen4_exp_qsa.h is a MODEL-PRIVATE header under src/, not include/vllm/: W4 ships
# nothing on the public ABI and include/vllm/ is a USER_USAGE_PREFIXES surface
# (#515). Same arrangement test_dots3_note_attn uses.
target_include_directories(test_qwen4_exp_qsa PRIVATE ${CMAKE_SOURCE_DIR}/src)

# MODEL-MM-QWEN4-EXP W3 (#1988) — the 4-branch GATED-RESIDUAL hyper-connection
# stream and its grouped RMSNorm, against goldens dumped by EXECUTING the
# lane-pinned oracle source (transformers v5.16.0 `modeling_qwen4_exp.py`, the
# two classes lifted verbatim by line range) plus an independent
# double-precision reference. Host reference; no arm of this model runs yet, so
# no token and no speed claim. `qwen4_exp_hc.h` is a MODEL-PRIVATE header under
# src/ (the `dots3_note.h` precedent), hence the explicit src/ grant.
vllm_cpp_add_test(test_qwen4_exp_hc vllm/models/test_qwen4_exp_hc.cpp)
target_include_directories(test_qwen4_exp_hc PRIVATE ${CMAKE_SOURCE_DIR}/src)

# MODEL-MM-QWEN4-EXP W5b-4 (#2167) — the DEVICE arm of Qwen Sparse Attention:
# vt::Qwen4ExpQsaCompress and vt::Qwen4ExpQsaGatherAttention, plus the indexer
# COMPOSED from the vt::DsaIndexerLogits / vt::DsaTopkSelect pair this tree
# already had, all against the SAME lane-pinned transformers goldens the W4 host
# reference answers to. Includes a released-config case at 3002 tokens of
# context, past the 2048 indexer budget below which every candidate is selected.
# CPU only: no CUDA arm of either op exists. Reaches the model-private
# qwen4_exp_qsa.h, hence the src/ grant.
vllm_cpp_add_test(test_qwen4_exp_qsa_device vllm/models/test_qwen4_exp_qsa_device.cpp)
target_include_directories(test_qwen4_exp_qsa_device PRIVATE ${CMAKE_SOURCE_DIR}/src)

# MODEL-MM-QWEN4-EXP W5b-5 (#2211) — `Qwen4ExpTextAttention` as ONE production
# block, and the first place under src/ that COMPOSES the QSA indexer. Gated
# against the lane-pinned oracle's own `Qwen4ExpTextAttention.forward` and, for
# the two settings no selection can see, against the oracle's own pre-top-k
# `scores` tensor BY VALUE. Carries the NaN-poison probe at the BLOCK layer and
# a released-config case past the 2048 indexer budget. CPU only. Reaches the
# model-private qwen4_exp_qsa_block.h / qwen4_exp_weights.h, hence the src/ grant.
vllm_cpp_add_test(test_qwen4_exp_qsa_block vllm/models/test_qwen4_exp_qsa_block.cpp)
target_include_directories(test_qwen4_exp_qsa_block PRIVATE ${CMAKE_SOURCE_DIR}/src)

# MODEL-MM-QWEN4-EXP W5b-2 (#2031) — the DEVICE arm of the same stream:
# vt::Qwen4ExpGatedResidual / vt::Qwen4ExpGatedResidualWriteBack against the SAME
# lane-pinned goldens the host reference is gated on, so the two arms answer to
# one oracle rather than to each other, plus a MODEL-WIDTH (hidden_size 2560,
# hc_count 4, hc_lowrank 320) agreement check the golden shapes cannot make.
# CPU arms only, and the two ops now DIFFER on device coverage: the write-back
# has a CUDA arm (W6-CUDA) gated byte-identically in test_qwen4_exp_cuda below,
# the gated residual has none because its grouped norm owes a reduction-width
# decision first. Reaches the model-private `qwen4_exp_hc.h`, hence the src/ grant.
vllm_cpp_add_test(test_qwen4_exp_hc_device vllm/models/test_qwen4_exp_hc_device.cpp)
target_include_directories(test_qwen4_exp_hc_device PRIVATE ${CMAKE_SOURCE_DIR}/src)

# MODEL-MM-QWEN4-EXP W5b (#2031) -- the assembled layer loop, and the
# loader/op compositions no single-sided suite can see. Needs `src/` for the
# model-private qwen4_exp headers. It takes NO checkpoint fixture dir: the suite
# builds its `qwen4exp` file in memory through
# `tests/support/qwen4_exp_gguf_fixture.h` and parses no released config, so a
# `QWEN4_EXP_CKPT_FIXTURE_DIR` grant here would compile a path nothing reads.
vllm_cpp_add_test(test_qwen4_exp_forward vllm/models/test_qwen4_exp_forward.cpp)
target_include_directories(test_qwen4_exp_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)

# MODEL-MM-QWEN4-EXP W5d-4 (#2249 item 4) — the MoE weight adapter, gated against
# a from-scratch double-precision reimplementation of the lane-pinned oracle's
# own `Qwen4ExpTextSparseMoeBlock.forward` that reads the SOURCE stacked towers,
# not the `MoeBlockWeights` the adapter produced. Carries the RED this wave was
# written against — the rank-3 towers dropped into the seam's rank-2 `_kq` fields
# — so #2249's "the shapes match" cannot be re-derived. CPU only: a weight
# adapter has no CUDA arm to compare, and the one GPU-only fact it turns on (the
# (bf16,f32) combo `MatmulBTKernelCuda` refuses) is asserted on the DTYPE
# instead. Reaches the model-private qwen4_exp_moe.h / qwen4_exp_weights.h, hence
# the src/ grant.
vllm_cpp_add_test(test_qwen4_exp_moe vllm/models/test_qwen4_exp_moe.cpp)
target_include_directories(test_qwen4_exp_moe PRIVATE ${CMAKE_SOURCE_DIR}/src)
# MOEDIV (#2552): the selected-expert-id tap. `MoeSelFpCalls()` is another
# read-once static, so the default run above can only ever observe the switch
# UNSET -- where the tap case asserts the tap is INERT. Re-run the SAME binary
# with it set: that is the arm which gates the printed ids, the bf16-ulp margin
# and the set-equality hash against `MoeReference`.
#
# DELIBERATELY UNFILTERED. A `-tc=` pattern that matches nothing exits 0 having
# run nothing, and doctest reports that as `assertions: 0 ... SUCCESS!` -- a
# skip wearing a pass, which this row has already paid for. Running the whole
# binary cannot fail that way, and it buys a second property for free: every
# value case in the suite re-runs with the tap ON, so a tap that perturbed the
# block's own output would red them.
add_test(NAME test_qwen4_exp_moe_sel_fp COMMAND test_qwen4_exp_moe)
set_tests_properties(test_qwen4_exp_moe_sel_fp PROPERTIES
  ENVIRONMENT "VT_MOE_SEL_FP=4096")

# QUANT-CUDA-GATES W8: the device-resident keep-quant MoE arm. NOT CUDA-only --
# the arm asks the op table rather than naming a device, so it is default-ON on
# every backend the three ops are registered on, and the suite has a CPU, a CUDA
# and a ROCm case. The CPU cases run unconditionally and gate arm SELECTION on a
# box with no GPU at all; each device case prints a LOUD [SKIP] where its backend
# is absent, because a device arm that silently vanishes from the gate is the
# failure this suite exists to stop. It reaches only the public seam header
# (qwen3_5_moe_block.h) plus the one probe symbol `vllm::MoeKqDeviceCalls()`, so
# it needs no src/ grant.
vllm_cpp_add_test(test_qwen35_moe_kq_device vllm/models/test_qwen35_moe_kq_device.cpp)
# The same binary on the REFERENCE arm, so the fallback the row promises is a
# gated route and not a claim. The suite forces the variable per case, so this
# registration proves the process-level default does not change the verdict.
add_test(NAME test_qwen35_moe_kq_device_ref COMMAND test_qwen35_moe_kq_device)
set_tests_properties(test_qwen35_moe_kq_device_ref PROPERTIES
  ENVIRONMENT "VT_MOE_KQ_FAST=0")
# The same binary with the VT_MOE_SEL_FP TAP ARMED, which is the only way to gate
# the `MoeSelFpCalls() == 0` term of the arm's predicate. That term is
# load-bearing: `MoeSelFp` is called only inside the reference path, so without
# it an operator who arms the tap gets the fast arm and a silently empty tap.
# It cannot be a case in the suite -- `MoeSelFpCalls()` caches its read in a
# process-static on first use, so arming it from inside the binary either comes
# first and disarms every other case or comes later and is ignored. With the tap
# armed every arm-selection expectation in the suite flips to "must not run", so
# deleting the term reddens THIS registration while the two above stay green.
# DELIBERATELY UNFILTERED, for the reason test_qwen4_exp_moe_sel_fp gives above.
add_test(NAME test_qwen35_moe_kq_device_sel_fp COMMAND test_qwen35_moe_kq_device)
set_tests_properties(test_qwen35_moe_kq_device_sel_fp PROPERTIES
  ENVIRONMENT "VT_MOE_SEL_FP=4096")
# THE ROCm LANE, BECAUSE THE DOCUMENTED ROCm GATE IS A NAME FILTER. `docs/ROCM.md:24`
# runs `ctest --test-dir build-hip -R 'rocm|cross_device'`, and none of the three
# registrations above carries either token -- so this suite's ROCm cases never ran in
# the lane a ROCm operator is told to run, on an arm that is DEFAULT-ON on ROCm with
# `strix:gpu0` (gfx1151) a live fleet device. Registering the same binary under a name
# that carries `rocm`, filtered to its ROCm cases, puts them in that lane.
#
# A doctest filter that matches NOTHING exits 0, so a renamed case would turn this
# registration into a green that measured nothing -- the trap
# `scripts/run-doctest-selected.sh` exists for. ctest cannot run that wrapper on every
# platform this file is configured on, so the zero-selection case is made RED by the
# summary line instead: doctest prints `test cases: 0 | ...` when the filter selects
# none. It is a FAIL_REGULAR_EXPRESSION and NOT a PASS_REGULAR_EXPRESSION, because
# CMake ignores the exit status of a test that has the latter, which would swap one
# silent green for another.
add_test(NAME test_rocm_qwen35_moe_kq_device COMMAND test_qwen35_moe_kq_device "-tc=*(ROCm)*")
set_tests_properties(test_rocm_qwen35_moe_kq_device PROPERTIES
  FAIL_REGULAR_EXPRESSION "test cases: +0 [|]")

# The five suites that reach src/vllm/support/platform_compat.h, granted per
# target rather than globally: the file already carries 123 explicit
# ${CMAKE_SOURCE_DIR}/src grants, and a blanket one in vllm_cpp_add_test would
# convert that deliberate opt-in into a repo-wide default (#503).
foreach(_pc_target
        test_safetensors
        test_minimax_h3
        test_minimax_h3_video_fold
        test_kv_offload_connector
        test_kv_offload_tiering)
  if(TARGET ${_pc_target})
    target_include_directories(${_pc_target} PRIVATE ${CMAKE_SOURCE_DIR}/src)
  endif()
endforeach()

# Qwen4-Exp (`Qwen4ExpForConditionalGeneration`) W2 -- the hashed n-gram
# embedding and the PLE dilated depthwise conv (#1987, campaign #1978,
# .agents/specs/qwen4-exp-flash-next.md). These are the ONLY two components of
# the model with NO vLLM op, confirmed by a negative search rather than an
# unfound one, so transformers v5.16.0 -- this row's accepted lane pin -- is the
# sole oracle and the goldens beside this file were produced by EXECUTING
# upstream's own bytes. The spec admits no token gate for this row until an arm
# runs (nothing published fits any fleet device), so G0 component goldens are
# the whole instrument. CPU-only, no checkpoint, no GPU, no speed claim.
vllm_cpp_add_test(test_qwen4_exp_ple vllm/models/test_qwen4_exp_ple.cpp)
# qwen4_exp_ple.h is a MODEL-PRIVATE header under src/, same arrangement as the
# dots3-note gates above: W2 ships nothing on the public ABI.
target_include_directories(test_qwen4_exp_ple PRIVATE ${CMAKE_SOURCE_DIR}/src)

# MODEL-MM-QWEN4-EXP W5b-3 (#2156) — the DEVICE arm of the same convolution:
# vt::Qwen4ExpPleConv against the SAME lane-pinned goldens the W2 host reference
# is gated on, so the two arms answer to one oracle rather than to each other,
# plus a MODEL-WIDTH (10240 channels) agreement check the golden shapes cannot
# make. CPU arm only here; the CUDA arm this op gained in W6-CUDA is gated in
# test_qwen4_exp_cuda below, against these same goldens. Reaches the
# model-private `qwen4_exp_ple.h`, hence the src/ grant.
vllm_cpp_add_test(test_qwen4_exp_ple_device vllm/models/test_qwen4_exp_ple_device.cpp)
target_include_directories(test_qwen4_exp_ple_device PRIVATE ${CMAKE_SOURCE_DIR}/src)

# MODEL-MM-QWEN4-EXP W5e-1 (#2336) — vt::Qwen4ExpPleGate, the PLE signed-sqrt
# gate and its broadcast sigmoid scale, against section J of the SAME lane-pinned
# goldens: modeling_qwen4_exp.py:1181-1182 + the :1184 flatten, exec'd verbatim.
# Includes the vt::BatchedMatmul composition of the :1180 dot, so the claim that
# the dot needs no new op is measured rather than asserted. CPU arm only here;
# the CUDA arm this op gained in W6-CUDA is gated in test_qwen4_exp_cuda below.
# No src/ grant — this test reaches nothing but the public vt:: surface.
vllm_cpp_add_test(test_qwen4_exp_ple_gate vllm/models/test_qwen4_exp_ple_gate.cpp)

# MODEL-MM-QWEN4-EXP W6-CUDA — the DEVICE-ARM gate for the three qwen4_exp ops
# that now have CUDA kernels (`vt::Qwen4ExpPleConv`, `vt::Qwen4ExpPleGate`,
# `vt::Qwen4ExpGatedResidualWriteBack`). Holds each to the SAME lane-pinned
# transformers goldens the CPU arms answer to, AND to the CPU arm itself on
# identical inputs — byte-identically for the write-back, which is what makes
# that op's oracle gate transitive. Skips LOUDLY on a box with no CUDA backend,
# except the registration case, which is guarded on the BUILD and not the
# device. Compiled unconditionally, like every other CUDA suite here.
# No src/ grant — it reaches nothing but the public vt:: surface.
vllm_cpp_add_test(test_qwen4_exp_cuda vllm/models/test_qwen4_exp_cuda.cpp)

# MODEL-MM-QWEN4-EXP W6-CUDA-B — the DEVICE-ARM gate for the three qwen4_exp ops
# that own a REDUCTION and that W6-CUDA left owed: `vt::Qwen4ExpGatedResidual`
# (the mixer), `vt::Qwen4ExpQsaCompress` and `vt::Qwen4ExpQsaGatherAttention`.
# `vt::RmsNormGroup`'s CUDA arm is the fourth and is gated beside its CPU suite
# in test_ops_rms_norm_group_cuda above, because it is a shared `vt::` seam and
# not a qwen4_exp op. Each op is held to the SAME lane-pinned transformers
# goldens its CPU arm answers to AND to the CPU arm itself: BYTE-identically for
# the compressor, within one f32 ulp for the gather (whose only divergence source
# is `expf`), and within the oracle's own band for the mixer, whose three
# projections go through the shared `vt::MatmulBT` seam and so re-associate a
# reduction the CPU arm walks. The gather's read SET is gated two ways --
# `keys_visited` for the count and a NaN-poisoned cache for the membership --
# because a mask over a dense cache passes every value comparison. Skips LOUDLY
# on a box with no CUDA backend. No src/ grant: it reaches nothing but the public
# vt:: surface and its own fixture directory.
vllm_cpp_add_test(test_qwen4_exp_cuda_reductions
                  vllm/models/test_qwen4_exp_cuda_reductions.cpp)

# MODEL-MM-QWEN4-EXP, issue ISSUE-LOCAL-01M2CJXQMV9R9JGRSKZMW4W21F, spec
# `.agents/specs/qwen4-exp-rocm-hc-norm.md` -- the ROCm twin of the `CUDA W7:`
# battery above, for the grouped hyper-connection norm inside
# `vt::Qwen4ExpGatedResidual`. It exists because until it did, the ONLY committed
# test reaching that ROCm arm was `test_backend_cross_device.cpp:5580` at
# `hidden_size = 7`, a width at which one thread does the whole group and no
# reduction SHAPE is expressible. The two files share one fixture and one derived
# bound through `vllm/models/qwen4_exp_hc_synth.h` rather than carrying a copy
# each. Skips LOUDLY on a box with no ROCm backend, and names the OP as well as
# the backend, because a built-but-unregistered arm and an absent backend are
# different failures. No src/ grant: it reaches only the public vt:: surface.
vllm_cpp_add_test(test_qwen4_exp_rocm_reductions
                  vllm/models/test_qwen4_exp_rocm_reductions.cpp)

# MODEL-MM-QWEN4-EXP W5e-2 (#2336) — `RunQwen4ExpPleBlock`, the PLE layer as ONE
# production composition and the LAST of the three block seams. Gated end to end
# against the SAME lane-pinned goldens the W2 host reference is held to
# (`kPleExpectedOutput` and `kPleMaskedExpectedOutput`, upstream's own
# `Qwen4ExpTextPLELayer.forward` executed under torch), single-shot and
# incremental, so the composition and the reference answer to ONE oracle. Carries
# the three FIXTURE-separation measurements that make this wave's mutation
# battery interpretable, the conv_mask's paired EOS obligation, and the
# analytic bound on the f32 score accumulator inside the clamp band at the
# RELEASED width. CPU only: every op it is the first production caller of is
# registered on kCPU alone. Reaches the model-private qwen4_exp_ple_block.h /
# qwen4_exp_ple.h / qwen4_exp_weights.h, hence the src/ grant.
vllm_cpp_add_test(test_qwen4_exp_ple_block vllm/models/test_qwen4_exp_ple_block.cpp)
target_include_directories(test_qwen4_exp_ple_block PRIVATE ${CMAKE_SOURCE_DIR}/src)

# MODEL-MM-QWEN4-EXP W5d-2 (#2249 item 5) — behaviour-neutrality gate for giving
# `BuildMropeCosSinHost` external linkage. Pins the interleaved and chunked
# mRoPE cos|sin tables BITWISE against the values the file-static definition
# produced at base SHA 94de63ff5, and is itself the seam case: a FOREIGN
# translation unit reaching the definition only through the new public header.
# No src/ grant — the whole point is that the header is public.
vllm_cpp_add_test(test_qwen3_5_mrope vllm/models/test_qwen3_5_mrope.cpp)

# GLM-5.3 (`GlmMoeDsaForCausalLM`) W2 -- the config, the registration, the
# `glm-dsa` GGUF arm and the refuse-by-name forward (#2214, spec
# `.agents/specs/glm-dsa-latest-deepseek.md` §3.7). The suite reads the
# COMMITTED `zai-org/GLM-5.3` config.json (`vllm/models/glm_moe_dsa_config_glm53.inc`,
# revision `935644c05e76fc198714f4cca449fd8b970ff6d7`) so the three-way indexer
# schedule agreement is gated with no asset, and it enters through
# `LoadedEngine::FromModelDir` so the registration and the dispatch-table row
# are gated as REACHED rather than merely present.
vllm_cpp_add_test(test_glm_moe_dsa_config vllm/models/test_glm_moe_dsa_config.cpp)
target_include_directories(test_glm_moe_dsa_config PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)

# W4: the heterogeneous indexer schedule, the `skip_topk` selection reuse and the
# fp32 router gate GEMM. It reads the SAME committed `config.json` fixture the
# W2 suite does, so the 21/57 split it asserts is the checkpoint's own and not a
# literal, and it drives `mla::ForwardMlaAttentionBlock` on a sparse step —
# which nothing else in `tests/` does outside the dots3-note model forward.
vllm_cpp_add_test(test_glm_moe_dsa_schedule vllm/models/test_glm_moe_dsa_schedule.cpp)
target_include_directories(test_glm_moe_dsa_schedule PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)

# W7 / G2: the structural loader gate. Two cases run with no artifact (the claim
# set against the checkpoint's own committed config.json); the third reads the
# six real UD-IQ1_S shard HEADERS and is skipped unless VT_GLM_DSA_GGUF names
# shard 1. Spec `.agents/specs/glm-dsa-latest-deepseek.md` §3.6 G2.
vllm_cpp_add_test(test_glm_moe_dsa_gguf_census
  vllm/models/test_glm_moe_dsa_gguf_census.cpp)
target_include_directories(test_glm_moe_dsa_gguf_census PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)

# W7: the load itself, on a COMPLETE synthetic `glm-dsa` GGUF with the same
# structure as the 201.83 GiB artifact -- heterogeneous indexer schedule, a
# leading dense block, stacked keep-quant towers and an MTP tail to drop. Enters
# through `LoadedEngine::FromModelDir`, which is the reachability proof.
vllm_cpp_add_test(test_glm_moe_dsa_gguf_load
  vllm/models/test_glm_moe_dsa_gguf_load.cpp)
target_include_directories(test_glm_moe_dsa_gguf_load PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)

# W9: THE FORWARD. Drives a first token through `LoadedEngine::FromModelDir` --
# the entry point a user arrives through, so deleting the forward call in
# `ForwardGlmMoeDsaForCausalLM` reds it -- then gates the pieces on the same
# synthetic model: finite logits with the distribution printed, the `kShared`
# selection reuse, the routed experts reaching `expert_stream::ExpertSlice`
# (GLM-5.3 is that lane's SECOND client, spec O15), and the two refusals this
# build still owes. Spec `.agents/specs/glm-dsa-latest-deepseek.md` §3.7 W9.
vllm_cpp_add_test(test_glm_moe_dsa_forward
  vllm/models/test_glm_moe_dsa_forward.cpp)
target_include_directories(test_glm_moe_dsa_forward PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
# THE LANE IS ON FOR THIS TARGET, and that is load-bearing rather than a
# convenience. Expert streaming is default OFF; with it off `ExpertSlice` builds
# no store, both arms of the streamed-vs-resident case read the tower in place,
# and the comparison would pass on a forward that never touched the seam. The
# suite asserts `StreamRequested()` and the lane's own fill counters, so a run
# without this environment reds rather than quietly measuring nothing.
#
# The suite ALSO sets the variable from its own namespace-scope static (#2450),
# because this property only exists when ctest launches the binary and an
# implementer debugging one case runs the binary directly. Both are kept: the
# static is what makes a direct run honest, and this line is what states the
# lane's environment where a reader of the build looks for it.
set_tests_properties(test_glm_moe_dsa_forward PROPERTIES
  ENVIRONMENT "VT_MOE_EXPERT_STREAM=1")

# QUANT-QWEN38-27B-NVFP4-ARM W7 (#2760). A ModelOpt checkpoint that DECLARES
# `quant_algo: "NVFP4"` -- 4-bit weights AND 4-bit ACTIVATIONS -- and that this
# build runs W4A16 has to say so at load. Every case enters through
# `vllm::LoadQwen3_5Dense` with stderr captured, never through the notice
# function alone, because a test that calls the function directly stays green
# when the production call site is deleted. Cases 1-6 are hermetic over a
# one-layer synthetic checkpoint in the artifact's own module names, as is case
# 8; case 7 is env-gated on VLLM_CPP_QWEN38_27B_RADIXARK_DIR and reads only
# `config.json` and the three shards' safetensors headers.
vllm_cpp_add_test(test_qwen38_27b_radixark_w4a4_notice
  vllm/models/test_qwen38_27b_radixark_w4a4_notice.cpp)
target_include_directories(test_qwen38_27b_radixark_w4a4_notice PRIVATE
  ${CMAKE_SOURCE_DIR}/src)
