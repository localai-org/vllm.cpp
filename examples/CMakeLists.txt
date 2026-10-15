add_executable(dump_container dump_container/main.cpp)
target_link_libraries(dump_container PRIVATE vllm::vllm)
vllm_cpp_set_warnings(dump_container)

add_executable(tokenize tokenize/main.cpp)
target_link_libraries(tokenize PRIVATE vllm::vllm)
vllm_cpp_set_warnings(tokenize)

add_executable(dequant_nvfp4 dequant_nvfp4/main.cpp)
target_link_libraries(dequant_nvfp4 PRIVATE vllm::vllm)
vllm_cpp_set_warnings(dequant_nvfp4)

# deepseek-v4-gen: the single-Spark DeepSeek-V4-Flash keep-quant GGUF greedy
# generation driver (the W8-run vehicle — stateless full-recompute greedy loop).
add_executable(deepseek-v4-gen deepseek_v4_gen/main.cpp)
target_link_libraries(deepseek-v4-gen PRIVATE vllm::vllm)
vllm_cpp_set_warnings(deepseek-v4-gen)

# laguna-gen: the single-GB10 Laguna-S-2.1 keep-quant GGUF greedy generation
# driver (W5 RUNNABLE vehicle — 3-shard keep-quant load + stateless recompute).
add_executable(laguna-gen laguna_gen/main.cpp)
target_link_libraries(laguna-gen PRIVATE vllm::vllm)
vllm_cpp_set_warnings(laguna-gen)

# kimi-linear-gen: the Kimi-Linear-48B-A3B golden token battery, a THIN CLIENT
# of the public C ABI (ONE SURFACE / ROW 7 §20.3 B4): vllm_engine_load + the
# v13 vllm_complete_tokens over the pre-tokenized golden prompts. Links the
# packaged shared library like vllm-cli — no internal headers.
add_executable(kimi-linear-gen kimi_linear_gen/main.cpp)
target_link_libraries(kimi-linear-gen PRIVATE vllm::shared)
vllm_cpp_set_warnings(kimi-linear-gen)

# nemotron-h-gen: the Nemotron-3.5-Lightning-30B (NemotronHForCausalLM) A3
# end-to-end token battery against the PINNED vLLM ORACLE golden, and a THIN
# CLIENT of the public C ABI in exactly the kimi-linear-gen shape above:
# vllm_engine_load + the v13 vllm_complete_tokens over the golden's
# pre-tokenized prompts, linking the packaged shared library so it reaches no
# internal header and needs no example-abi-allowlist row. #810,
# .agents/specs/nemotron-h-abi-e2e.md §6.1.
add_executable(nemotron-h-gen nemotron_h_gen/main.cpp)
target_link_libraries(nemotron-h-gen PRIVATE vllm::shared)
vllm_cpp_set_warnings(nemotron-h-gen)

# vllm-bench (M2.1): the throughput/latency benchmark harness — the gate-#1
# measurement tool. Links the full C++ engine (vllm::vllm) so it can build both a
# synthetic CPU engine (no --model) and a real loaded engine (--model on dgx).
add_executable(vllm-bench bench/main.cpp)
target_link_libraries(vllm-bench PRIVATE vllm::vllm)
target_include_directories(vllm-bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bench)
vllm_cpp_set_warnings(vllm-bench)

# vllm-cli: a llama.cpp-style client driven ENTIRELY through the C ABI. It links
# the packaged SHARED libvllm (vllm_shared) using only the public include/vllm.h,
# proving library-first consumption (M3.5 Task 3).
add_executable(vllm-cli cli/main.cpp)
target_link_libraries(vllm-cli PRIVATE vllm::shared)
vllm_cpp_set_warnings(vllm-cli)

if(VLLM_CPP_XPU AND VLLM_CPP_IMAGE_CODECS)
  # Direct embedding-client proof: exported C ABI only, no private engine path.
  add_executable(native-vision-chat native_vision_chat/main.cpp)
  target_link_libraries(native-vision-chat PRIVATE vllm::shared)
  target_include_directories(native-vision-chat SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
  vllm_cpp_set_warnings(native-vision-chat)
endif()

# gliner-cli: a THIN CLIENT of the public C ABI (MODEL-GLINER25). Loads a
# GLiNER2.5 checkpoint and runs zero-shot NER via vllm_gliner_ner (ABI v27),
# proving library-first consumption of the NER surface.
add_executable(gliner-cli gliner_cli/main.cpp)
target_link_libraries(gliner-cli PRIVATE vllm::shared)
vllm_cpp_set_warnings(gliner-cli)

# quant-gemm-bench (QUANT-GGUF-CIQ-GEMM G6 evidence tool, NOT a gate): op-level
# GFLOP/s for kMatmulBTQuant, portable vs Arm i8mm mmla tier.
add_executable(quant-gemm-bench quant_gemm_bench/main.cpp)
target_link_libraries(quant-gemm-bench PRIVATE vllm::vllm)
target_include_directories(quant-gemm-bench PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_set_warnings(quant-gemm-bench)

add_executable(vllm-cpu-kernel-bench cpu_kernel_bench/main.cpp)
target_link_libraries(vllm-cpu-kernel-bench PRIVATE vllm::vllm)
target_include_directories(vllm-cpu-kernel-bench PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_set_warnings(vllm-cpu-kernel-bench)
if(VLLM_CPP_BUILD_TESTS)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME test_cpu_kernel_bench_cli
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/tests/scripts/test_cpu_kernel_bench.py
      $<TARGET_FILE:vllm-cpu-kernel-bench>)
endif()

# The OpenAI HTTP server example is gated on the vendored cpp-httplib transport.
if(VLLM_CPP_SERVER)
  # vllm-video-studio: a STANDALONE browser console for MiniMax-H3, deliberately
  # SEPARATE from the API server. examples/server is the OpenAI-compatible API
  # surface and a UI does not belong in it; this owns its own endpoints and
  # drives the public C ABI (vllm_video_*) like any FFI consumer would. It shares
  # the server's gate only because it uses the same vendored httplib transport.
  add_executable(video-studio video_studio/main.cpp)
  target_link_libraries(video-studio PRIVATE vllm::shared)
  target_include_directories(video-studio SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
  set_target_properties(video-studio PROPERTIES OUTPUT_NAME vllm-video-studio)
  vllm_cpp_set_warnings(video-studio)

  add_executable(server server/main.cpp)
  # ARCH-ONE-SURFACE: server/main.cpp is a THIN CLIENT of the public C ABI
  # (vllm_server_main, v17) -- it includes vllm.h and nothing else.  The release
  # executable links the static core below, but reaches it only through that same
  # public surface.
  # Release bundles carry one static-core server.  The executable remains a
  # thin C-ABI client; only the link form changes so an extracted archive does
  # not depend on a separately installed libvllm.
  target_link_libraries(server PRIVATE vllm::vllm)
  if(VLLM_CPP_LITERAL_STATIC)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR VLLM_CPP_CUDA OR
       VLLM_CPP_METAL OR VLLM_CPP_MLX OR VLLM_CPP_VULKAN OR VLLM_CPP_HIP OR
       VLLM_CPP_XPU)
      message(FATAL_ERROR
        "VLLM_CPP_LITERAL_STATIC is restricted to the experimental Linux CPU-only lane")
    endif()
    target_link_options(server PRIVATE -static -static-libgcc -static-libstdc++)
  endif()
  set_target_properties(server PROPERTIES OUTPUT_NAME vllm-server)
  if(VLLM_CPP_MLX)
    set_target_properties(server PROPERTIES
      BUILD_RPATH "${MLX_ROOT}/lib"
      INSTALL_RPATH "@loader_path/../lib")
    find_file(_vllm_mlx_metallib mlx.metallib
      PATHS "${MLX_ROOT}/lib" "${MLX_ROOT}" NO_DEFAULT_PATH)
    if(NOT _vllm_mlx_metallib)
      message(FATAL_ERROR
        "MLX release packaging requires mlx.metallib under MLX_ROOT")
    endif()
    install(FILES "${MLX_LIBRARY}" "${_vllm_mlx_metallib}"
      DESTINATION ${CMAKE_INSTALL_LIBDIR}
      COMPONENT vllm-server)
  endif()
  vllm_cpp_set_warnings(server)
  install(TARGETS server
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT vllm-server)

  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  set(_vllm_server_backends)
  if(VLLM_CPP_CUDA)
    list(APPEND _vllm_server_backends cuda)
  endif()
  if(VLLM_CPP_METAL)
    if(VLLM_CPP_MLX)
      list(APPEND _vllm_server_backends metal-mlx)
    else()
      list(APPEND _vllm_server_backends metal)
    endif()
  endif()
  if(VLLM_CPP_VULKAN)
    list(APPEND _vllm_server_backends vulkan)
  endif()
  if(VLLM_CPP_HIP)
    list(APPEND _vllm_server_backends rocm)
  endif()
  if(VLLM_CPP_XPU)
    list(APPEND _vllm_server_backends xpu)
  endif()
  if(NOT _vllm_server_backends)
    list(APPEND _vllm_server_backends cpu)
  endif()
  string(JOIN "-" _vllm_server_backend ${_vllm_server_backends})
  string(TOLOWER "${CMAKE_SYSTEM_NAME}" _vllm_server_os)
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _vllm_server_arch)
  set(_vllm_server_artifact_backend "${_vllm_server_backend}")
  if(VLLM_CPP_CUDA)
    set(_vllm_server_artifact_backend cuda)
  elseif(VLLM_CPP_LITERAL_STATIC)
    set(_vllm_server_artifact_backend cpu-static)
  endif()
  if(_vllm_server_os STREQUAL "linux")
    if(VLLM_CPP_LITERAL_STATIC)
      set(_vllm_server_abi musl)
    else()
      set(_vllm_server_abi glibc)
    endif()
    set(_vllm_server_artifact_id
      "linux-${_vllm_server_arch}-${_vllm_server_abi}-${_vllm_server_artifact_backend}")
  elseif(_vllm_server_os STREQUAL "darwin")
    set(_vllm_server_artifact_id
      "macos-${_vllm_server_arch}-${_vllm_server_artifact_backend}")
  else()
    set(_vllm_server_artifact_id
      "${_vllm_server_os}-${_vllm_server_arch}-${_vllm_server_artifact_backend}")
  endif()
  set(_vllm_server_release_dir "${CMAKE_BINARY_DIR}/release")
  set(_vllm_server_stage_dir "${_vllm_server_release_dir}/stage")
  set(_vllm_server_archive
    "${_vllm_server_release_dir}/vllm.cpp-${PROJECT_VERSION}-${_vllm_server_artifact_id}.tar.gz")

  add_custom_target(vllm-server-stage
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/package-server.py
      --build-dir ${CMAKE_BINARY_DIR}
      --stage-dir ${_vllm_server_stage_dir}
      --config $<CONFIG>
    DEPENDS server ${CMAKE_SOURCE_DIR}/scripts/package-server.py
    COMMENT "Installing the vllm-server component into an empty staging prefix"
    VERBATIM)

  add_custom_target(vllm-server-archive
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/package-server.py
      --build-dir ${CMAKE_BINARY_DIR}
      --stage-dir ${_vllm_server_stage_dir}
      --archive ${_vllm_server_archive}
      --archive-format tar.gz
      --config $<CONFIG>
    DEPENDS server ${CMAKE_SOURCE_DIR}/scripts/package-server.py
    BYPRODUCTS ${_vllm_server_archive}
    COMMENT "Creating deterministic installed vllm-server archive"
    VERBATIM)

  if(VLLM_CPP_BUILD_TESTS)
    add_test(NAME test_server_help COMMAND $<TARGET_FILE:server> --help)
    set_tests_properties(test_server_help PROPERTIES
      PASS_REGULAR_EXPRESSION "--max-num-seqs N.*--max-num-batched-tokens N.*--cuda-profile-graph-replays N.*--cuda-profile-graph-batch N.*--benchmark-shutdown-fifo F.*--enable-force-include-usage.*--\\[no-\\]enable-prefix-caching")

    # --video-family (row MODEL-DIFFUSION-LTX25 / #435). Black-box, because the
    # flag table lives in an anonymous namespace inside server_main.cpp and there
    # is no unit seam onto it: the binary IS the surface under test. Both cases
    # exit during ParseArgs, so neither loads a model, a checkpoint or a byte of
    # weights, and both run in milliseconds.
    #
    # 1. AN UNREGISTERED NAME IS REFUSED, and the refusal NAMES what is
    #    registered. Before the flag existed this printed "unknown argument
    #    '--video-family'" instead, which is the RED.
    add_test(NAME test_server_video_family_unregistered
      COMMAND $<TARGET_FILE:server> --model /nonexistent-model-dir
              --video-family not-a-registered-family)
    set_tests_properties(test_server_video_family_unregistered PROPERTIES
      PASS_REGULAR_EXPRESSION "--video-family 'not-a-registered-family' is not a registered video family. Registered families:.*ltx-2\\.5.*minimax-h3")

    # 2. A REGISTERED name is ACCEPTED — and specifically, the check consults the
    #    REGISTRY rather than a list written out here, so the family LTX-2.5
    #    registers from its own file passes. Proven by getting PAST the family
    #    check to the next validation in ParseArgs (the tool-parser dialect),
    #    with the family refusal absent from the output. A hardcoded list that
    #    omitted ltx-2.5 REDs on the FAIL expression, and a check accidentally
    #    dropped altogether REDs on it too (the unknown-argument spelling).
    add_test(NAME test_server_video_family_registered
      COMMAND $<TARGET_FILE:server> --model /nonexistent-model-dir
              --video-family ltx-2.5
              --tool-call-parser definitely-not-a-registered-parser)
    set_tests_properties(test_server_video_family_registered PROPERTIES
      PASS_REGULAR_EXPRESSION "unknown tool-call parser \"definitely-not-a-registered-parser\""
      FAIL_REGULAR_EXPRESSION "not a registered video family;unknown argument")

    # 3. --video-extra takes KEY=VALUE, and a bare KEY is REFUSED rather than
    #    read as an empty value. An extra that silently became "" is how a
    #    mistyped knob renders the default and looks like the feature not
    #    working — the same failure the families' own unknown-extra refusals
    #    exist to stop, and it would be defeated one layer earlier here.
    add_test(NAME test_server_video_extra_needs_a_value
      COMMAND $<TARGET_FILE:server> --model /nonexistent-model-dir
              --video-extra dit_config_path)
    set_tests_properties(test_server_video_extra_needs_a_value PROPERTIES
      PASS_REGULAR_EXPRESSION "--video-extra takes KEY=VALUE, got 'dit_config_path'")

    # A SPEECH-ONLY server (#672): --speech-model with NO --model. Upstream's own
    # recipe is `sgl-omni serve --model MiniMaxAI/MiniMax-Music3` and nothing
    # else, and requiring --model here meant serving a 28.5 GB music model also
    # staged an unrelated text model no request would touch. Black-box for the
    # same reason the --video-family cases above are: the flag table lives in an
    # anonymous namespace and the binary IS the surface.
    #
    # 1. NEITHER flag is still an error, and the message now names the second
    #    way to satisfy it. The pre-change spelling had no --speech-model
    #    clause, so this expression is the RED.
    add_test(NAME test_server_neither_model_flag
      COMMAND $<TARGET_FILE:server>)
    set_tests_properties(test_server_neither_model_flag PROPERTIES
      PASS_REGULAR_EXPRESSION "--model <dir> is required \\(or --speech-model <dir> for a speech/music-only server\\)")

    # 2. --speech-model ALONE gets PAST the argument check and into the speech
    #    LOAD, which is the whole claim: before the change it exited at
    #    "--model <dir> is required" without ever reaching a checkpoint. Proven
    #    by the load refusal for a directory nothing claims, which is evidence
    #    that the speech branch ran, and by asserting the --model demand is
    #    ABSENT from the output.
    add_test(NAME test_server_speech_only_reaches_the_speech_load
      COMMAND $<TARGET_FILE:server> --speech-model /nonexistent-speech-dir)
    set_tests_properties(test_server_speech_only_reaches_the_speech_load PROPERTIES
      PASS_REGULAR_EXPRESSION "--speech-model /nonexistent-speech-dir"
      FAIL_REGULAR_EXPRESSION "--model <dir> is required")

    # 3. --speech-family STILL requires --speech-model. The new accepted
    #    combination must not have widened this one into "a family with no
    #    checkpoint is fine now".
    add_test(NAME test_server_speech_family_still_needs_a_checkpoint
      COMMAND $<TARGET_FILE:server> --model /nonexistent-model-dir
              --speech-family minimax-music3)
    set_tests_properties(test_server_speech_family_still_needs_a_checkpoint PROPERTIES
      PASS_REGULAR_EXPRESSION "--speech-family names a family but --speech-model names no checkpoint")

    # --speculative-config key ADMISSION, from the production entry point (#1160).
    # `tests/vllm/config/test_speculative_unknown_keys.cpp` calls the parser
    # directly, so it localizes a failure and proves NOTHING about reach: it stays
    # green with the server's call site deleted. These three enter through the
    # server binary on its default configuration, which is where a user meets this
    # flag, and go RED when `server_main.cpp` stops calling
    # `ParseSpeculativeConfigJson`. All three exit before a byte of weights is
    # read, because the parse runs before `LoadedEngine::FromModelDir`.
    #
    # 1. The issue's reproduction. Before the fix this key was DROPPED, so the
    #    server started and drafted greedy, and the run reached the model-load
    #    error below instead of this refusal.
    add_test(NAME test_server_speculative_rejects_probabilistic_draft
      COMMAND $<TARGET_FILE:server> --model /nonexistent-model-dir
              --speculative-config
              "{\"method\":\"dspark\",\"model\":\"d\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"probabilistic\"}")
    set_tests_properties(test_server_speculative_rejects_probabilistic_draft PROPERTIES
      PASS_REGULAR_EXPRESSION "speculative-config: draft_sample_method \"probabilistic\" is not implemented.*SPEC-ACCEPT-VARIANTS"
      FAIL_REGULAR_EXPRESSION "model path is not a directory")

    # 2. A MISSPELLED key. `num_speculatve_tokens` used to fall through to the
    #    resolved default, so the user silently got a k they did not ask for.
    add_test(NAME test_server_speculative_rejects_a_misspelled_key
      COMMAND $<TARGET_FILE:server> --model /nonexistent-model-dir
              --speculative-config "{\"method\":\"mtp\",\"num_speculatve_tokens\":7}")
    set_tests_properties(test_server_speculative_rejects_a_misspelled_key PROPERTIES
      PASS_REGULAR_EXPRESSION "speculative-config: unknown key \"num_speculatve_tokens\""
      FAIL_REGULAR_EXPRESSION "model path is not a directory")

    # 3. The CONTROL, and the case that makes the two above mean something. A
    #    valid config gets PAST the admission check and on to the model load, so
    #    an admission check that refused everything would go red here rather than
    #    passing the two refusal cases for the wrong reason.
    add_test(NAME test_server_speculative_accepts_a_valid_config
      COMMAND $<TARGET_FILE:server> --model /nonexistent-model-dir
              --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":1}")
    set_tests_properties(test_server_speculative_accepts_a_valid_config PROPERTIES
      PASS_REGULAR_EXPRESSION "model path is not a directory: /nonexistent-model-dir"
      FAIL_REGULAR_EXPRESSION "speculative-config")
  endif()
endif()

# minimax-h3-mux: muxes MiniMax-H3 PPM frames + WAV into an MP4 by invoking
# ffmpeg — a THIN client of the public C ABI (include/vllm.h + vllm::shared
# ONLY, like vllm-cli), per the ONE SURFACE directive. This is the RATIFIED
# home of that process spawn (developer decision 2026-08-03): the library
# composes the argv (vllm_video_mux_argv), examples/ runs it.
add_executable(minimax-h3-mux minimax_h3_mux/main.cpp)
target_link_libraries(minimax-h3-mux PRIVATE vllm::shared)
vllm_cpp_set_warnings(minimax-h3-mux)

# minimax-h3-gen: MiniMax-H3 video generation end to end — a THIN client of
# the public C ABI (ARCH-ONE-SURFACE ROW 2). The assembly pipeline it used to
# own privately lives in the library behind vllm_video_engine_load +
# vllm_video_generate; the fold gate holds its frames + WAV byte-identical to
# the pre-fold binary. Lives here because it invokes ffmpeg (the ratified
# boundary).
add_executable(minimax-h3-gen minimax_h3_gen/main.cpp)
target_link_libraries(minimax-h3-gen PRIVATE vllm::shared)
vllm_cpp_set_warnings(minimax-h3-gen)

# ltx2-gen: an LTX-2.5 render end to end, the SECOND family behind the generic
# video seam and a THIN client of the same C ABI. It exists rather than a flag on
# minimax-h3-gen because LTX-2.5 cannot load without the v18 family-specific
# extras (the audio-stream prompt embeds, the DiT config the shipped FP8
# checkpoint does not carry, the latent upsampler) and that example drives
# neither `family` nor the extras arrays. Row MODEL-DIFFUSION-LTX25 / #435.
add_executable(ltx2-gen ltx2_gen/main.cpp)
target_link_libraries(ltx2-gen PRIVATE vllm::shared)
vllm_cpp_set_warnings(ltx2-gen)

# vocoder-conv-ab: the CPU-vs-device A/B for the BigVGAN / DAC vocoder
# convolution chain (#672), SAME BINARY with `VLLM_CPP_VOCODER_DEVICE` as the
# only variable. A benchmarks/ source rather than an examples/ one, and so out of
# the ABI-client rule's scope by the same reading `vulkan_gemm_ab.cpp` lands on
# (surface-coverage-2026-08-07.md, "Out of the gated examples/ tree"): it links
# `vllm::vllm` and calls the vocoder core directly, because the thing being timed
# is the seam and not a model. NOT gated on CUDA — the CPU arm is half the A/B.
add_executable(vocoder-conv-ab ${CMAKE_SOURCE_DIR}/benchmarks/vocoder_conv_ab.cpp)
target_link_libraries(vocoder-conv-ab PRIVATE vllm::vllm)
target_include_directories(vocoder-conv-ab PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_set_warnings(vocoder-conv-ab)

# expert-stream-device-w0e: the W0e measurement harness for row
# ENG-EXPERT-STREAM-DEVICE (#1124). It exists because one generation has to
# yield three things together that no shipped command produces at once: the
# generated token ids, a per-step arrival timestamp, and the expert-stream
# counters at both ends of the run — and a live gate needs the DECODE-phase
# delta of those counters, not the total, because prefill exhaustion here is
# structural. Unlike the A/B harnesses above it is a pure client of
# `include/vllm.h` and links `vllm::shared`, so it stays inside the ABI-client
# rule and needs no `src/` include path. Not gated on CUDA: the CPU arm is the
# baseline the CUDA arm is compared against.
add_executable(expert-stream-device-w0e
               ${CMAKE_SOURCE_DIR}/benchmarks/expert_stream_device_w0e.cpp)
target_link_libraries(expert-stream-device-w0e PRIVATE vllm::shared)
vllm_cpp_set_warnings(expert-stream-device-w0e)

# vulkan-gemm-ab: the VK-C tactic A/B — cooperative matrix vs the portable scalar
# GEMM, SAME BINARY with VT_VULKAN_COOPMAT as the only variable. Built only in a
# Vulkan build, because it links the Vulkan context directly to report which
# tactic each arm actually ran (a silent fallback would post a plausible pair of
# timings that mean nothing).
if(VLLM_CPP_VULKAN)
  add_executable(vulkan-gemm-ab ${CMAKE_SOURCE_DIR}/benchmarks/vulkan_gemm_ab.cpp)
  target_link_libraries(vulkan-gemm-ab PRIVATE vllm::vllm)
  target_include_directories(vulkan-gemm-ab PRIVATE ${CMAKE_SOURCE_DIR}/src)
  vllm_cpp_set_warnings(vulkan-gemm-ab)

  # vulkan-dispatch-floor: the VK-A2 sizing measurement. Sweeps ONE op across a
  # 65,536x range of element counts to separate per-dispatch overhead from real
  # kernel cost -- two causes that produce an identical wall clock but imply
  # opposite fixes. Same Vulkan-only condition, same reason.
  add_executable(vulkan-dispatch-floor
                 ${CMAKE_SOURCE_DIR}/benchmarks/vulkan_dispatch_floor.cpp)
  target_link_libraries(vulkan-dispatch-floor PRIVATE vllm::vllm)
  target_include_directories(vulkan-dispatch-floor PRIVATE ${CMAKE_SOURCE_DIR}/src)
  vllm_cpp_set_warnings(vulkan-dispatch-floor)

  # vulkan-gemv-ab: the BACKEND-VULKAN-GEMVROWS variant sweep. Runs the decode
  # GEMV over the (k, n) shapes Qwen3.6-27B actually dispatches, so a variant can
  # be accepted or discarded without spending a 50 GB e2e run on it. Same
  # Vulkan-only condition and the same reason as the two above: it prints the
  # pipeline cache key, so an arm proves which specialization it ran.
  add_executable(vulkan-gemv-ab ${CMAKE_SOURCE_DIR}/benchmarks/vulkan_gemv_ab.cpp)
  target_link_libraries(vulkan-gemv-ab PRIVATE vllm::vllm)
  target_include_directories(vulkan-gemv-ab PRIVATE ${CMAKE_SOURCE_DIR}/src)
  vllm_cpp_set_warnings(vulkan-gemv-ab)
endif()

# w32-grouped-cols-ab: the MODEL-MM-QWEN4-EXP output-width sweep for the grouped
# 32-block GEMV (ISSUE-LOCAL-01M2TZ9AE416FW4GH23V09783Q). Runs the released
# qwen4_exp routed gate/up decode shape at VT_V4_W32_COLS 1/2/4/8 and
# VT_V4_W32_WARPS 1/2/4/8, out of ONE binary, so the width can be chosen from a
# measurement instead of an assumption -- and prints an FNV-1a hash of the whole
# output per arm, because the width axis is required to be BIT-identical and a
# fast arm that moved a float is void rather than interesting. CUDA-only and run
# by hand, like vulkan-gemv-ab above.
if(VLLM_CPP_CUDA)
  add_executable(w32-grouped-cols-ab ${CMAKE_SOURCE_DIR}/benchmarks/w32_grouped_cols_ab.cpp)
  target_link_libraries(w32-grouped-cols-ab PRIVATE vllm::vllm)
  target_include_directories(w32-grouped-cols-ab PRIVATE ${CMAKE_SOURCE_DIR}/src)
  vllm_cpp_set_warnings(w32-grouped-cols-ab)
endif()

# parakeet-transcribe: the ASR example — a THIN client of the public C ABI
# (include/vllm.h + vllm::shared ONLY, like vllm-cli), per the ONE SURFACE
# directive. The pipeline it used to own privately lives in the library behind
# vllm_engine_load + vllm_transcribe (ARCH-ONE-SURFACE ROW 1); the fold gate
# holds its transcript byte-identical to the pre-fold binary.
add_executable(parakeet-transcribe parakeet_transcribe/main.cpp)
target_link_libraries(parakeet-transcribe PRIVATE vllm::shared)
vllm_cpp_set_warnings(parakeet-transcribe)

# minimax-music3-gen: MiniMax-Music3 text-to-music end to end — a THIN client of
# the public C ABI (include/vllm.h + vllm::shared ONLY, like vllm-cli and
# parakeet-transcribe). It exists because the music family was the one
# generative family in this tree with NO example: hearing the model needed a
# running server plus a curl, or a C ABI caller nobody had written (#672).
add_executable(minimax-music3-gen minimax_music3_gen/main.cpp)
target_link_libraries(minimax-music3-gen PRIVATE vllm::shared)
vllm_cpp_set_warnings(minimax-music3-gen)
