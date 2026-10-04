cmake_minimum_required(VERSION 3.24)

# FIRST, before project() enables a compiler and before anything else can spend
# minutes getting to a failure that names the wrong thing: an in-source `cmake .`
# collides every examples/ executable with the source directory it was built
# from and dies in the linker (issue #85). cmake/InSourceGuard.cmake explains
# the collision; cmake/InSourceGuardTest.cmake pins when it fires.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(InSourceGuard)
vllm_cpp_forbid_in_source_build("${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}")

project(vllm_cpp VERSION 0.0.3 LANGUAGES CXX)

set(VLLM_CPP_BUILD_VERSION "${PROJECT_VERSION}" CACHE STRING
  "Exact version identity reported by this build")
if(VLLM_CPP_BUILD_VERSION STREQUAL "")
  message(FATAL_ERROR "VLLM_CPP_BUILD_VERSION must not be empty")
endif()

# Native Windows release ABI: the core and every consumer use the static
# MSVC/UCRT runtime (/MT in Release, /MTd in Debug), so the installed server
# does not depend on a developer-machine CRT deployment.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
if(MSVC)
  # Keep PE debug directories reproducible and free of runner/build paths.
  add_link_options("/PDBALTPATH:%_PDB%")
  # Windows headers otherwise replace qualified std::min/std::max calls with
  # legacy macros. The project intentionally uses the portable ISO CRT entry
  # points across platforms, and tokenizer tables require a UTF-8 execution
  # character set. These are platform contracts; /W4 /WX remains unchanged.
  add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Host C++ floating point is pinned to IEEE as-written semantics: no FMA
# contraction. GCC/Clang default to -ffp-contract=fast, which lets the compiler
# fuse `a*b + c` into a single-rounding fma PER LOOP, PER FUNCTION — so two
# textually identical reductions (e.g. the vt::RmsNorm golden vs the FusedChain
# Tier-1 interpreter, both `sumsq += v*v`) can compile one contracted and one
# not, breaking the bit-identity contracts the vt:: CPU reference kernels and
# their tests rely on (test_ops_fused_chain, the fused-vs-split fp8 pairs, ...).
# This bites wherever FMA is baseline ISA (aarch64 always; x86 with -march=*).
# Pinning contract=off gives every TU (kernels AND tests) the exact numerics of
# the plain x86-64 Release build on every platform. CUDA device code is NOT
# affected (nvcc flags are separate; GPU parity tests compare GPU-vs-GPU).
if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/fp:strict>)
else()
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
endif()

# Release archives must not embed the absolute build directory. FetchContent
# dependencies (parakeet.cpp's vendored ggml) live under CMAKE_BINARY_DIR/_deps,
# so their __FILE__/assert strings carry that absolute path, and the
# deterministic-archive validator (scripts/validate-release-archive.py
# --forbid-path) correctly rejects such artifacts. Rewrite the prefix instead of
# weakening the validator.
if(NOT MSVC)
  add_compile_options(
    "-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=."
    "-ffile-prefix-map=${CMAKE_CURRENT_BINARY_DIR}=.")
endif()

option(VLLM_CPP_BUILD_TESTS "Build tests" ON)
option(VLLM_CPP_BUILD_EXAMPLES "Build examples" ON)
# The OpenAI HTTP server (M3.1 Task 4) is gated so the core build stays lean when
# the vendored cpp-httplib transport header is absent. cpp-httplib is a
# header-only MIT HTTP TRANSPORT lib (llama.cpp's choice) — a transport-dep
# deviation, NOT a compute/ML dependency (consistent with the no-pytorch/no-ggml
# rule).
option(VLLM_CPP_SERVER "Build the OpenAI HTTP server (needs third_party/httplib/httplib.h)" ON)
option(VLLM_CPP_LITERAL_STATIC
  "Link the CPU-only server as a literal static executable (experimental musl release lane only)"
  OFF)
# ENG-HF-MODEL-DOWNLOAD W5 (#1280): fetching a HuggingFace repository named by
# `--model` needs transport layer security, because `huggingface.co` speaks
# nothing else. The shape mirrors llama.cpp at stock tag `b10451`
# (`vendor/cpp-httplib/CMakeLists.txt:38-154`), which retired libcurl and now
# transports over the same cpp-httplib this tree already vendors: one option for
# the feature, one for OpenSSL, one for a statically fetched BoringSSL. It also
# follows the VLLM_CPP_SERVER idiom directly above -- a missing dependency
# DOWNGRADES the feature with a WARNING rather than failing the configure, so a
# host with no OpenSSL development files still builds a working engine.
option(VLLM_CPP_HF_DOWNLOAD
  "Fetch a HuggingFace repository named by --model (resolves OFF with no TLS)" ON)
option(VLLM_CPP_OPENSSL
  "Use the system OpenSSL for HTTPS in the HuggingFace client" ON)
option(VLLM_CPP_BUILD_BORINGSSL
  "Fetch and statically link BoringSSL for HTTPS (for the literal-static lane)" OFF)
# The value the USER asked for, captured before any resolution can downgrade it.
# `tests/CMakeLists.txt` arms its transport-layer-security expectation from this
# one, so a build where the feature silently resolved OFF fails a test instead
# of quietly running the no-TLS arm of every case and reporting SUCCESS.
set(VLLM_CPP_HF_DOWNLOAD_REQUESTED ${VLLM_CPP_HF_DOWNLOAD})

if(VLLM_CPP_SERVER AND NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/httplib/httplib.h")
  message(WARNING "VLLM_CPP_SERVER=ON but third_party/httplib/httplib.h is missing; "
    "disabling the server target. Vendor cpp-httplib to enable it.")
  set(VLLM_CPP_SERVER OFF)
endif()
set(VLLM_CPP_CUDA "AUTO" CACHE STRING "Build the CUDA backend: ON, OFF, or AUTO (detect)")
# Tri-state exactly like VLLM_CPP_CUDA above (BACKEND-METAL-MLX W0). AUTO means
# "on when the host is Apple and OBJCXX is usable" — i.e. it turns itself on for
# a macOS developer and is invisible everywhere else. Resolved to a plain ON/OFF
# a few lines below the CUDA resolution.
set(VLLM_CPP_METAL "AUTO" CACHE STRING "Build the Metal backend: ON, OFF, or AUTO (detect)")
# Tri-state like the two above (BACKEND-VULKAN W0), but AUTO deliberately
# resolves OFF rather than probing for a Vulkan loader. Rationale, recorded
# because it is a DIFFERENT choice from Metal's AUTO: Metal AUTO-enables on an
# Apple host because there is no other GPU backend there, so turning it on cannot
# perturb anything. Vulkan OVERLAPS CUDA on the very box the CUDA gate models run
# on (GB10 enumerates as a Vulkan device), and silently enabling a second GPU
# backend in the gate build would register kVULKAN into the op/platform tables
# and into tests/vt/test_backend_cross_device.cpp during the CUDA regressions it
# must not touch. So Vulkan is opt-in: -DVLLM_CPP_VULKAN=ON.
set(VLLM_CPP_VULKAN "AUTO" CACHE STRING "Build the Vulkan backend: ON, OFF, or AUTO (=OFF)")
# Tri-state like Vulkan/ROCm below (BACKEND-TENSTORRENT W0). AUTO resolves OFF:
# unlike Vulkan (headers vendored, dlopen at runtime), this backend needs a real
# external tt-metal install (find_package(TT-NN)/find_package(TT-Metalium), via
# -DCMAKE_PREFIX_PATH pointing at a built tt-metal's install tree) that nothing
# on a stock box provides, so probing by default would be a silent no-op
# everywhere except a developer's own tt-metal checkout.
set(VLLM_CPP_TENSTORRENT "AUTO" CACHE STRING "Build the Tenstorrent backend: ON, OFF, or AUTO (=OFF)")
# Tri-state like the three above (BACKEND-ROCM W0). AUTO resolves OFF for the
# same reason Vulkan's does, plus one specific to this skeleton: it has NEVER
# BEEN COMPILED by anyone (no AMD GPU on the authoring machine), so AUTO-enabling
# it wherever /opt/rocm happens to exist would turn a green build red on a
# machine whose owner asked for nothing. Opt in with -DVLLM_CPP_HIP=ON, which is
# exactly what a contributor on issue #41 is asked to do first.
set(VLLM_CPP_HIP "AUTO" CACHE STRING "Build the ROCm/HIP backend: ON, OFF, or AUTO (=OFF)")
# Target gfx architecture(s), semicolon-separated, e.g. "gfx1100" or
# "gfx1100;gfx1151". EMPTY (the default) means: pass no --offload-arch at all and
# let hipcc target the installed GPU. That is the right default for a contributor
# building on the machine they will run on, which is every case today. The
# validated set is upstream's HIP_SUPPORTED_ARCHS (vllm/CMakeLists.txt:52 @ pin
# 555967922); a value outside it configures with a WARNING rather than an error,
# because a newer ROCm may know an arch that list does not.
set(VLLM_CPP_HIP_ARCHITECTURES "" CACHE STRING
  "Target gfx arch(es) for the ROCm backend, e.g. gfx1100 (empty = hipcc default)")
# ROCm installation prefix. Mirrors upstream's ROCM_PATH (vllm/CMakeLists.txt:54-60)
# with the same default and the same override, so a TheRock/nightly install in a
# non-standard prefix is one flag away.
set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation prefix")
# The MLX acceleration PROVIDER for the Metal dense GEMM (BACKEND-ACCEL-PROVIDER
# work row M5, .agents/specs/metal-mlx-reuse-study.md §6.2). Default OFF and
# deliberately NOT tri-state: unlike a PLATFORM, a provider is a CONFIGURATION
# CHOICE, and it drags in a ~19 MB libmlx.dylib plus a ~105 MB mlx.metallib —
# a real deviation from discipline.md's header-only preference, accepted only as
# an opt-in exactly like VLLM_CPP_TRITON. Native MSL stays the default and is
# what ships. Point -DMLX_ROOT at an MLX install (the pip wheel's
# `site-packages/mlx` directory works: it ships include/, lib/ and the metallib).
set(VLLM_CPP_MLX OFF CACHE BOOL "Build the optional MLX GEMM provider for the Metal backend")
set(MLX_ROOT "" CACHE PATH "Root of an MLX install (include/ + lib/) for VLLM_CPP_MLX")
# Supported values today are the sm_12x (consumer/GB10 Blackwell) family:
#   121a          GB10 / DGX Spark — the default and the only arch with a GATED
#                 runtime here (every gate model, every benchmark).
#   120a          consumer Blackwell (RTX 5070/5080/5090 class). Same family, same
#                 kernel bodies, same FEATURE-TABLE cells; BUILD-supported and
#                 verified to emit sm_120a SASS, but NOT runtime-gated here — no
#                 sm_120 board exists on this hardware. See backend-matrix.md
#                 BACKEND-CUDA-SM120 for exactly what is and is not proven.
#   120a;121a     same-family FAT binary; builds clean and carries both SASS
#                 targets. VLLM_CPP_TRITON=ON is FINE here (and is the default):
#                 a vendored AOT cubin is single-arch, but the W2 builder embeds
#                 EVERY vendored tree and picks one by exact SM at runtime, so
#                 sm_120a — which has no tree — simply uses the portable CUDA
#                 kernels. This is what the shipped ten-SM release archive and
#                 the cuda-fat-gencode CI job build. The single-arch FATAL_ERROR
#                 in TritonAOT.cmake guards the MAINTAINER regen flow only.
#   90a           Hopper (H100/H200) — a DIFFERENT family. Single-arch only, and
#                 PORTABLE-KERNELS-ONLY: all five accelerated features resolve
#                 DISABLED (we have no Hopper wgmma/TMA kernel body, so no
#                 FEATURE-TABLE cell names 9.0a), so only the portable C++/CUDA
#                 kernels compile — build-supported, NOT vLLM-competitive, and
#                 runtime-UNPROVEN (no Hopper board here). See backend-matrix.md
#                 BACKEND-CUDA-SM090 and cuda-arch-additivity.md §W9.
#   80 86 87 89   Ampere (A100 / consumer 30-series / Jetson Orin) and Ada
#   100a 103a     (sm_89), datacenter Blackwell (sm_100/sm_103), and sm_110 — the
#   110           CROSS-FAMILY BUILD-SUPPORTED FAN-OUT (spec §W10). Single-arch,
#                 PORTABLE-KERNELS-ONLY exactly like 90a: every fp4/cutlass/
#                 marlin/fa2 feature resolves DISABLED (no fast-path body for
#                 these families), so only the portable C++/CUDA kernels compile.
#                 BUILD-supported and runtime-UNVERIFIED (no such board here).
#                 sm_80 and sm_100a were compiled end to end `-Werror` 0-warn on
#                 dgx as the per-major representatives; the same-major siblings
#                 (86/87/89, 103a) share the identical portable bodies. sm_70
#                 (Volta, dropped by nvcc 13) and sm_75 (Turing, no bf16 tensor
#                 cores → the portable bf16-WMMA path will not compile) are NOT
#                 build-supported — they need a real fp16/non-tensor-core kernel
#                 body, a scoped follow-up. See backend-matrix.md
#                 BACKEND-CUDA-SM0{80,86,87,89}/SM10{0,3}/SM110 and §W10.
# Cross-family fat lists use release-W1 per-source gencode: portable TUs receive
# every requested target while feature TUs receive their exact feature-table
# intersection. The ten-SM archive gate below proves both the commands and the
# linked cubins. See .agents/specs/release-binary-matrix.md W1.
set(VLLM_CPP_CUDA_ARCHITECTURES "121a" CACHE STRING "Target CUDA architectures (GB10=121a, consumer Blackwell=120a, same-family fat binary=120a;121a; the 'a' arch-specific target is REQUIRED for the native block-scaled fp4xfp4 MMA — mma.sync kind::mxf4nvf4 is rejected on base sm_120/sm_121)")
option(VLLM_CPP_BENCH_PROFILE_CONTROL
  "Compile trace-only SIGUSR2/cudaProfiler replay control plus FIFO graceful shutdown (never for production timing builds)"
  OFF)

# Resolve the tri-state VLLM_CPP_CUDA into a plain ON/OFF. The normal variable
# set below intentionally shadows the cache entry so later if(VLLM_CPP_CUDA)
# checks see the resolved value while the user's cached choice is preserved.
include(CheckLanguage)
if(VLLM_CPP_CUDA STREQUAL "AUTO")
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA toolchain found; enabling CUDA backend")
    set(VLLM_CPP_CUDA ON)
  else()
    set(VLLM_CPP_CUDA OFF)
  endif()
elseif(VLLM_CPP_CUDA)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "VLLM_CPP_CUDA=ON but no CUDA compiler was found. "
      "Install the CUDA toolkit or configure with -DVLLM_CPP_CUDA=OFF (or AUTO).")
  endif()
  set(VLLM_CPP_CUDA ON)
else()
  set(VLLM_CPP_CUDA OFF)
endif()

# ─── Host instrumentation lanes ───────────────────────────────────────────────
# One default build is shallow coverage: it reaches neither the dynamic
# memory/UB detectors nor the race detector, so a whole defect class (bad index
# arithmetic, use-after-free across the engine's owning containers, unsynchronized
# access from the serving threads) can only be caught by a failing parity test,
# if at all. This selects an instrumentation lane for the HOST C++ build so those
# detectors run as their own CI jobs alongside the plain build.
#
# ASan/UBSan and TSan are deliberately SEPARATE lanes: their runtimes are
# mutually exclusive and their environment requirements differ. The lane is
# HOST-ONLY — nvcc-compiled device TUs are not instrumented, and a host sanitizer
# runtime reports false positives against the CUDA driver — so anything other
# than OFF requires the resolved VLLM_CPP_CUDA to be OFF. The CUDA tier's
# equivalent detector is compute-sanitizer, run against the normal build.
#
# -fno-sanitize-recover=all makes a finding ABORT instead of print-and-continue,
# so ctest FAILS the lane rather than burying the report inside passing output.
# Frame pointers plus -g1 keep file/line traces attributable without emitting
# full type/local-variable DWARF. Every test statically links the instrumented
# library, so full -g duplicates hundreds of MiB into each executable; the
# ASan+UBSan GitHub runner exhausted its filesystem before the suite finished
# linking. The lane keeps the project's -ffp-contract=off numerics so a
# bit-identity test still means what it means in the production build.
set(VLLM_CPP_SANITIZE "OFF" CACHE STRING
  "Host sanitizer lane: OFF | address | undefined | address,undefined | thread")
set_property(CACHE VLLM_CPP_SANITIZE PROPERTY STRINGS
  OFF address undefined address,undefined thread)
if(NOT VLLM_CPP_SANITIZE STREQUAL "OFF")
  set(_vllm_cpp_sanitize_allowed address undefined address,undefined thread)
  if(NOT VLLM_CPP_SANITIZE IN_LIST _vllm_cpp_sanitize_allowed)
    message(FATAL_ERROR
      "VLLM_CPP_SANITIZE='${VLLM_CPP_SANITIZE}' is not one of: "
      "OFF;${_vllm_cpp_sanitize_allowed}")
  endif()
  if(VLLM_CPP_CUDA)
    message(FATAL_ERROR
      "VLLM_CPP_SANITIZE requires VLLM_CPP_CUDA=OFF: the host sanitizer runtime "
      "does not instrument nvcc device TUs and reports false positives against "
      "the CUDA driver. Use compute-sanitizer for the CUDA tier instead.")
  endif()
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=${VLLM_CPP_SANITIZE}>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-sanitize-recover=all>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
    $<$<COMPILE_LANGUAGE:CXX>:-g1>)
  # -Werror itself is dropped for this lane in cmake/CompilerWarnings.cmake, where
  # the per-target warning flags are set; see the rationale there.
  add_link_options(-fsanitize=${VLLM_CPP_SANITIZE})
  message(STATUS "Host sanitizer lane: ${VLLM_CPP_SANITIZE} (CUDA backend off)")
endif()

# Resolve the tri-state VLLM_CPP_METAL the same way (BACKEND-METAL-MLX W0). The
# Metal backend is ObjC++ against the Metal/Foundation frameworks, so it needs
# APPLE plus a working OBJCXX compiler; on any other host AUTO resolves OFF and
# nothing about the build changes.
if(VLLM_CPP_METAL STREQUAL "AUTO")
  if(APPLE)
    check_language(OBJCXX)
    if(CMAKE_OBJCXX_COMPILER)
      message(STATUS "Apple host with an ObjC++ compiler; enabling Metal backend")
      set(VLLM_CPP_METAL ON)
    else()
      set(VLLM_CPP_METAL OFF)
    endif()
  else()
    set(VLLM_CPP_METAL OFF)
  endif()
elseif(VLLM_CPP_METAL)
  if(NOT APPLE)
    message(FATAL_ERROR "VLLM_CPP_METAL=ON but this is not an Apple host. "
      "Configure with -DVLLM_CPP_METAL=OFF (or AUTO).")
  endif()
  check_language(OBJCXX)
  if(NOT CMAKE_OBJCXX_COMPILER)
    message(FATAL_ERROR "VLLM_CPP_METAL=ON but no ObjC++ compiler was found. "
      "Install the Xcode Command Line Tools, or configure with -DVLLM_CPP_METAL=OFF.")
  endif()
  set(VLLM_CPP_METAL ON)
else()
  set(VLLM_CPP_METAL OFF)
endif()
# Resolve the tri-state VLLM_CPP_VULKAN (BACKEND-VULKAN W0). There is nothing to
# DETECT at configure time: the backend needs no SDK, no shader compiler and no
# link-time library — the Khronos headers are vendored (third_party/vulkan/), the
# SPIR-V is committed (src/vt/vulkan/vulkan_spirv.h), and libvulkan.so.1 is
# opened with dlopen at run time. Whether a Vulkan DEVICE exists is therefore a
# RUNTIME question, answered by vt::vulkan::VulkanContext::Available(), which is
# what decides whether kVULKAN registers. So ON always succeeds and AUTO means
# OFF (see the option's comment for why AUTO does not probe).
if(VLLM_CPP_VULKAN STREQUAL "AUTO")
  set(VLLM_CPP_VULKAN OFF)
elseif(VLLM_CPP_VULKAN)
  set(VLLM_CPP_VULKAN ON)
else()
  set(VLLM_CPP_VULKAN OFF)
endif()

# Resolve the tri-state VLLM_CPP_TENSTORRENT (BACKEND-TENSTORRENT W0). AUTO
# resolves OFF (see the option's comment above). ON is where detection happens
# and FAILS LOUDLY rather than silently downgrading to a CPU-only build — the
# same contract VLLM_CPP_HIP below uses, for the same reason: a contributor who
# typed -DVLLM_CPP_TENSTORRENT=ON and got a green CPU-only build would have no
# way to notice the backend was not built.
if(VLLM_CPP_TENSTORRENT STREQUAL "AUTO")
  set(VLLM_CPP_TENSTORRENT OFF)
elseif(VLLM_CPP_TENSTORRENT)
  find_package(TT-Metalium QUIET)
  find_package(TT-NN QUIET)
  if(NOT TT-Metalium_FOUND OR NOT TT-NN_FOUND)
    message(FATAL_ERROR "VLLM_CPP_TENSTORRENT=ON but TT-Metalium/TT-NN were not "
      "found. Point -DCMAKE_PREFIX_PATH at a built tt-metal install tree's "
      "lib64/cmake and share/cmake dirs (both are needed - the latter carries "
      "nlohmann_json/xtensor), or configure with -DVLLM_CPP_TENSTORRENT=OFF.")
  endif()
  set(VLLM_CPP_TENSTORRENT ON)
else()
  set(VLLM_CPP_TENSTORRENT OFF)
endif()

# Resolve the tri-state VLLM_CPP_HIP (BACKEND-ROCM W0). Unlike Vulkan there IS
# something to detect — a HIP toolchain — but AUTO still resolves OFF (see the
# option's comment). ON is where detection happens, and it FAILS LOUDLY rather
# than silently downgrading: a contributor who typed -DVLLM_CPP_HIP=ON and got a
# green CPU-only build would have no way to notice the backend was not built,
# which is the exact failure mode the CUTLASS note in docs/BUILD.md exists for.
if(VLLM_CPP_HIP STREQUAL "AUTO")
  set(VLLM_CPP_HIP OFF)
elseif(VLLM_CPP_HIP)
  # F1/F3 ABSORPTION (issue #41, jimmykarily's gfx1151 findings; layout also
  # confirmed by arch-btw on gfx1103/TheRock). Arch and TheRock dist layouts
  # install the compiler at ${ROCM_PATH}/lib/llvm/bin instead of
  # ${ROCM_PATH}/llvm/bin, so clang's walk-up-from-InstalledDir autodetection
  # derives the ROCm root as ${ROCM_PATH}/lib and finds neither the HIP headers
  # (F1) nor the hip-lang CMake package (F3: CMakeDetermineHIPCompiler
  # hard-fails). The unidentified-compiler fallout also broke the LINKER:
  # prefix expansion on the --whole-archive link options further down (F2) —
  # one root cause, three unrelated-looking failures. The report's finding:
  # ROCM_PATH was advertised by our own FATAL_ERROR below but only ever used
  # for find_library, so on those layouts the documented flag did not do what
  # its error message promised.
  #
  # Fix: derive the three hints the report set by hand from ROCM_PATH — and
  # ONLY where unset, so an explicit -DCMAKE_HIP_FLAGS /
  # -DCMAKE_HIP_COMPILER_ROCM_ROOT / exported HIPFLAGS/ROCM_PATH environment
  # always wins, and a machine whose ROCM_PATH does not exist on disk (hipcc
  # found elsewhere on PATH) configures byte-identically to before. On the
  # standard /opt/rocm layout the values filled are exactly what clang/CMake
  # autodetect anyway. Upstream vLLM honours ROCM_PATH the same way
  # (vllm/CMakeLists.txt:54-60 @ pin 555967922). All three must be set BEFORE
  # check_language/enable_language: the report measured that the ROOT hint
  # alone regresses the compiler identification, because the identification
  # try-compile then has no --rocm-path.
  if(EXISTS "${ROCM_PATH}")
    # CMake's own ROCm-root probe (F3), consulted by CMakeDetermineHIPCompiler.
    if(NOT DEFINED CMAKE_HIP_COMPILER_ROCM_ROOT)
      set(CMAKE_HIP_COMPILER_ROCM_ROOT "${ROCM_PATH}")
    endif()
    # clang's own ROCm-root flag (F1). Rides CMAKE_HIP_FLAGS so the
    # compiler-identification try-compile sees it too; pre-seeded as the cache
    # entry enable_language(HIP) would otherwise create empty. Skipped when the
    # user passed -DCMAKE_HIP_FLAGS or exported HIPFLAGS, which must keep
    # winning.
    if(NOT DEFINED CACHE{CMAKE_HIP_FLAGS} AND NOT DEFINED ENV{HIPFLAGS})
      set(CMAKE_HIP_FLAGS "--rocm-path=${ROCM_PATH}" CACHE STRING
        "Flags used by the HIP compiler during all build types.")
    endif()
    # hipcc/hipconfig read ROCM_PATH from the environment (the mechanism the
    # report validated: `ROCM_PATH=/opt/rocm hipcc` compiles where bare hipcc
    # cannot), and the environment also reaches check_language's sub-configure
    # below, which forwards no cache variables.
    if(NOT DEFINED ENV{ROCM_PATH})
      set(ENV{ROCM_PATH} "${ROCM_PATH}")
    endif()
  endif()
  # CMake's own HIP language support (>= 3.21; we require 3.24) understands .hip
  # sources, which is why upstream calls enable_language(HIP) explicitly rather
  # than leaning on torch's setup (vllm/CMakeLists.txt:196-202).
  include(CheckLanguage)
  check_language(HIP)
  if(NOT CMAKE_HIP_COMPILER)
    message(FATAL_ERROR
      "VLLM_CPP_HIP=ON but no HIP compiler was found.\n"
      "  Install ROCm (hipcc), or point at it with -DROCM_PATH=<prefix> / "
      "-DCMAKE_HIP_COMPILER=<path to hipcc>, or configure with -DVLLM_CPP_HIP=OFF.\n"
      "  Looked with ROCM_PATH=${ROCM_PATH}.")
  endif()
  enable_language(HIP)
  # Matches upstream's CMAKE_HIP_STANDARD 20 (vllm/CMakeLists.txt:20-21). The vt
  # headers the .hip TUs include are C++20.
  set(CMAKE_HIP_STANDARD 20)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
  # The same IEEE-as-written pin every other language in this build gets. It is
  # NOT cosmetic here: tests/vt/test_backend_cross_device.cpp compares this
  # backend against the CPU kernels at NMSE <= 5e-4, and a contracted multiply-add
  # in the RmsNorm reduction is exactly the kind of drift that budget is meant to
  # catch rather than absorb.
  add_compile_options($<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>)
  # Floor HIP DEVICE code at -O1 when nothing else sets an optimisation level.
  #
  # This is not a performance preference, it is a correctness workaround for the
  # ROCm runtime (issue #132, VikashLoomba, 4x gfx1100 / ROCm 7.14). The
  # documented first build — `cmake -S . -B build-hip -DVLLM_CPP_HIP=ON`, no
  # CMAKE_BUILD_TYPE — passes no -O at all, so hipcc compiles at -O0. At -O0 all
  # eight RmsNorm specialisations come out with AMDHSA metadata
  # `.uses_dynamic_stack: true` plus `hidden_hostcall_buffer` / `hidden_heap_v1`,
  # even though the kernel performs no hostcall. That makes CLR create a hostcall
  # buffer and start a listener thread on the first launch.
  #
  # CLR's listener startup/termination handshake is racy (the shared state is
  # `volatile`, not atomic, and the upstream source still carries its own
  # `FIXME_lmoriche: fix termination handshake`): under host CPU saturation a
  # delayed listener can overwrite the `kExit` written by terminate(), and DSO
  # finalisation then spins forever waiting for a thread that has already exited.
  # The symptom is `test_backend_cross_device` hanging AFTER printing
  # `Status: SUCCESS!` — 11/11 cases, 39/39 assertions, then no exit.
  #
  # The report's controlled A/B, relinking the same library with ONLY
  # rocm_rmsnorm.hip.o changed, on a 48-thread host:
  #   -O0 object (dynamic stack, hidden hostcall + heap): 14/20 teardown timeouts
  #   -O1 object (no dynamic stack, no hostcall/heap):    20/20 clean exits
  # The same finaliser loop also reproduces in a standalone raw-HIP program that
  # links neither vllm.cpp nor doctest, so the defect is CLR's, not the harness'.
  # The definitive fix belongs upstream in CLR; this keeps our documented build
  # off the path that triggers it.
  #
  # Deliberately yields to any explicit choice: if the developer set a build type
  # or put an -O in CMAKE_HIP_FLAGS, that wins and nothing is added here.
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _vllm_hip_build_type)
  if((_vllm_hip_build_type STREQUAL "" OR _vllm_hip_build_type STREQUAL "DEBUG")
     AND NOT "${CMAKE_HIP_FLAGS}" MATCHES "(^| )-O")
    add_compile_options($<$<COMPILE_LANGUAGE:HIP>:-O1>)
    message(STATUS
      "HIP device code floored at -O1: -O0 marks kernels as dynamic-stack users "
      "and trips the ROCm CLR hostcall-listener teardown deadlock (issue #132)")
  endif()
  unset(_vllm_hip_build_type)
  set(VLLM_CPP_HIP ON)
else()
  set(VLLM_CPP_HIP OFF)
endif()

if(VLLM_CPP_HIP)
  # Upstream's validated arch list, verbatim (vllm/CMakeLists.txt:52 @ pin
  # 555967922). Used only to WARN: an unknown value is passed through to hipcc,
  # which is the component that actually knows what it supports.
  set(VLLM_CPP_HIP_SUPPORTED_ARCHS
    "gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201")
  foreach(arch IN LISTS VLLM_CPP_HIP_ARCHITECTURES)
    if(NOT arch IN_LIST VLLM_CPP_HIP_SUPPORTED_ARCHS)
      message(WARNING "VLLM_CPP_HIP_ARCHITECTURES contains '${arch}', which is not in "
        "upstream vLLM's validated set. Passing it to hipcc anyway.")
    endif()
  endforeach()
  if(VLLM_CPP_HIP_ARCHITECTURES)
    message(STATUS "ROCm backend: ENABLED for arch(es) [${VLLM_CPP_HIP_ARCHITECTURES}]")
  else()
    message(STATUS "ROCm backend: ENABLED, hipcc default arch (set "
      "-DVLLM_CPP_HIP_ARCHITECTURES=gfx#### to target explicitly)")
  endif()
  message(STATUS "ROCm backend: NEVER COMPILED UPSTREAM OF YOU — this skeleton has no "
    "build report from any machine. Please report success or failure on "
    "https://github.com/mudler/vllm.cpp/issues/41")
endif()

if(VLLM_CPP_METAL)
  enable_language(OBJCXX)
  set(CMAKE_OBJCXX_STANDARD 20)
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
  # Same IEEE-as-written pin the CXX sources get (see the -ffp-contract note at
  # the top of this file): the Metal HOST code shares the vt:: numeric contracts.
  add_compile_options($<$<COMPILE_LANGUAGE:OBJCXX>:-ffp-contract=off>)
endif()

if(VLLM_CPP_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  # CACHE ... FORCE, not a plain set(): `enable_language(CUDA)` has just written
  # its own detected default into the CACHE, and a normal variable only SHADOWS
  # that for the rest of the configure. The build was always correct — the
  # gencode lines in build.ninja are the truth and they follow this value — but
  # `CMakeCache.txt` kept reporting the detected default (`75`) even for an
  # explicit `-DVLLM_CPP_CUDA_ARCHITECTURES=110` build, so the obvious way to
  # confirm which architecture a build targets silently told you the wrong thing
  # (issue #168, Jetson AGX Thor). Forcing the cache to agree costs nothing:
  # every reader already resolved to this value.
  set(CMAKE_CUDA_ARCHITECTURES ${VLLM_CPP_CUDA_ARCHITECTURES} CACHE STRING
      "CUDA architectures actually compiled; mirrors VLLM_CPP_CUDA_ARCHITECTURES"
      FORCE)
  # Per-arch capability resolution (BACKEND-CUDA-ARCH-ADDITIVITY). Every
  # arch-gated feature below asks the FEATURE TABLE instead of regex-matching the
  # whole arch list, so adding an architecture cannot silently disable fp4 /
  # cutlass-nvfp4 / cutlass-fp8 / Marlin / FA2 for the archs that DO support them.
  # Mirrors vLLM's cuda_archs_loose_intersection idiom; see the module header.
  include(cmake/CudaArchFeatures.cmake)
  message(STATUS "CUDA target architectures: ${VLLM_CPP_CUDA_ARCHITECTURES}")
  vt_cuda_feature_archs(VT_FP4_MMA_ARCHS       "fp4-mma")
  vt_cuda_feature_archs(VT_CUTLASS_NVFP4_ARCHS "cutlass-nvfp4")
  vt_cuda_feature_archs(VT_CUTLASS_NVFP4_SM100_ARCHS "cutlass-nvfp4-sm100")
  vt_cuda_feature_archs(VT_CUTLASS_FP8_ARCHS   "cutlass-fp8")
  vt_cuda_feature_archs(VT_SCALEDMM_C3X_SM90_ARCHS "scaledmm-c3x-sm90")
  vt_cuda_feature_archs(VT_SCALEDMM_C3X_SM100_ARCHS "scaledmm-c3x-sm100")
  vt_cuda_feature_archs(VT_MARLIN_NVFP4_ARCHS  "marlin-nvfp4")
  vt_cuda_feature_archs(VT_FA2_ARCHS           "fa2")
  vt_cuda_report_feature("fp4-mma"       "${VT_FP4_MMA_ARCHS}")
  vt_cuda_report_feature("cutlass-nvfp4" "${VT_CUTLASS_NVFP4_ARCHS}")
  vt_cuda_report_feature("cutlass-nvfp4-sm100" "${VT_CUTLASS_NVFP4_SM100_ARCHS}")
  vt_cuda_report_feature("cutlass-fp8"   "${VT_CUTLASS_FP8_ARCHS}")
  vt_cuda_report_feature("scaledmm-c3x-sm90" "${VT_SCALEDMM_C3X_SM90_ARCHS}")
  vt_cuda_report_feature("scaledmm-c3x-sm100" "${VT_SCALEDMM_C3X_SM100_ARCHS}")
  vt_cuda_report_feature("marlin-nvfp4"  "${VT_MARLIN_NVFP4_ARCHS}")
  vt_cuda_report_feature("fa2"           "${VT_FA2_ARCHS}")
  # The native block-scaled fp4xfp4 MMA (cuda_matmul_nvfp4.cu) needs the
  # architecture-specific sm_12xa target. Advertise it to the code so the host
  # launcher enables the native path only when it was actually compiled in.
  if(VT_FP4_MMA_ARCHS)
    add_compile_definitions(VT_FP4_MMA_SM120A=1)
  endif()
  # CUTLASS (header-only, >=v4.5.0) for the sm120a NVFP4 block-scaled fp4xfp4 GEMM
  # drop-in (cuda_matmul_nvfp4_cutlass.cu). Vendored slice under third_party/cutlass
  # by default; override with -DVLLM_CPP_CUTLASS_DIR=<cutlass root> (must contain
  # include/ + tools/util/include/). The TU is enabled only when the dir exists and
  # the arch is sm_12xa (the block-scaled fp4 collective is Blackwell-only).
  set(VLLM_CPP_CUTLASS_DIR "${CMAKE_SOURCE_DIR}/third_party/cutlass"
      CACHE PATH "CUTLASS source root (include/ + tools/util/include/)")
  # Opt-in FetchContent fallback. FlashInfer 0.6.12's installed runtime bundles
  # CUTLASS 4.5.0; its SM12 scale-factor TMA changes are required for the traced
  # 128x32/128x64 tactics (4.4.2 rejects them at compile time).
  # no vendored/pointed cutlass tree is present, -DVLLM_CPP_CUTLASS_FETCH=ON pulls
  # v4.5.0 (header-only). Default OFF so a tight-disk box (dgx) builds against an
  # existing clone via -DVLLM_CPP_CUTLASS_DIR instead of a 200MB download.
  option(VLLM_CPP_CUTLASS_FETCH "FetchContent CUTLASS v4.5.0 if not found locally" OFF)
  # CUTLASS is needed by TWO independent consumers with DIFFERENT arch scopes:
  #   (1) the sm_12x NVFP4 block-scaled GEMM  -> gated on VT_CUTLASS_NVFP4_ARCHS;
  #   (2) the vendored FlashAttention-2 kernels -> gated on VT_FA2_ARCHS, which now
  #       ALSO covers Ampere sm_8x (CudaArchFeatures.cmake fa2 cell, WA-1). FA2 only
  #       `-isystem`s the CUTLASS include tree (see the FA2 target below); it does
  #       NOT use the NVFP4 collective. So "CUTLASS headers present + >=4.5.0" is an
  #       arch-INDEPENDENT fact (VLLM_CPP_CUTLASS_HEADERS), while VLLM_CPP_CUTLASS
  #       stays the strict sm_12x NVFP4 feature flag. This is the additive host
  #       predicate the Ampere FA2 fast path needs
  #       (.agents/specs/cuda-arch-ampere-fastpath.md WA-1). sm_121 is UNCHANGED:
  #       there VT_CUTLASS_NVFP4_ARCHS resolves, so both flags are ON exactly as before.
  if((VT_CUTLASS_NVFP4_ARCHS OR VT_CUTLASS_NVFP4_SM100_ARCHS OR VT_SCALEDMM_C3X_SM90_ARCHS OR VT_SCALEDMM_C3X_SM100_ARCHS OR VT_FA2_ARCHS)
     AND NOT EXISTS "${VLLM_CPP_CUTLASS_DIR}/include/cutlass/cutlass.h"
     AND VLLM_CPP_CUTLASS_FETCH)
    include(FetchContent)
    # SOURCE_SUBDIR names a directory that does not exist IN CUTLASS ON PURPOSE.
    # That is the documented way to tell FetchContent "populate, but do not
    # add_subdirectory": we consume CUTLASS as HEADERS ONLY — every consumer
    # below `-isystem`s ${VLLM_CPP_CUTLASS_DIR}/include and nothing in this tree
    # links a CUTLASS CMake target — so configuring CUTLASS's own build system
    # buys nothing and can only fail.
    #
    # And it does fail. Building for sm_80 with CUDA 13 dies inside CUTLASS's
    # tools/library with duplicate sm_100f flags, which forced the A100 reporter
    # to add -DCUTLASS_ENABLE_TOOLS=OFF by hand before they could get to the
    # actual bugs (issue #193). Not configuring those targets at all is both the
    # narrower fix and the faster configure: the tools/library/examples/tests
    # trees are the bulk of what add_subdirectory would walk.
    FetchContent_Declare(cutlass GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
      GIT_TAG v4.5.0 GIT_SHALLOW ON
      SOURCE_SUBDIR vllm-cpp-consumes-cutlass-headers-only)
    FetchContent_MakeAvailable(cutlass)
    set(VLLM_CPP_CUTLASS_DIR "${cutlass_SOURCE_DIR}")
  endif()
  if((VT_CUTLASS_NVFP4_ARCHS OR VT_CUTLASS_NVFP4_SM100_ARCHS OR VT_SCALEDMM_C3X_SM90_ARCHS OR VT_SCALEDMM_C3X_SM100_ARCHS OR VT_FA2_ARCHS)
     AND EXISTS "${VLLM_CPP_CUTLASS_DIR}/include/cutlass/cutlass.h")
    file(STRINGS "${VLLM_CPP_CUTLASS_DIR}/include/cutlass/version.h"
      _CUTLASS_VERSION_LINES
      REGEX "^#define CUTLASS_(MAJOR|MINOR|PATCH) [0-9]+")
    foreach(_CUTLASS_VERSION_LINE IN LISTS _CUTLASS_VERSION_LINES)
      if(_CUTLASS_VERSION_LINE MATCHES "^#define CUTLASS_MAJOR ([0-9]+)")
        set(_CUTLASS_MAJOR "${CMAKE_MATCH_1}")
      elseif(_CUTLASS_VERSION_LINE MATCHES "^#define CUTLASS_MINOR ([0-9]+)")
        set(_CUTLASS_MINOR "${CMAKE_MATCH_1}")
      elseif(_CUTLASS_VERSION_LINE MATCHES "^#define CUTLASS_PATCH ([0-9]+)")
        set(_CUTLASS_PATCH "${CMAKE_MATCH_1}")
      endif()
    endforeach()
    if(NOT DEFINED _CUTLASS_MAJOR OR NOT DEFINED _CUTLASS_MINOR OR
       NOT DEFINED _CUTLASS_PATCH)
      message(FATAL_ERROR "Could not read CUTLASS version from ${VLLM_CPP_CUTLASS_DIR}")
    endif()
    math(EXPR _CUTLASS_VERSION_NUMBER
      "${_CUTLASS_MAJOR} * 10000 + ${_CUTLASS_MINOR} * 100 + ${_CUTLASS_PATCH}")
    if(_CUTLASS_VERSION_NUMBER LESS 40500)
      message(FATAL_ERROR
        "CUTLASS >=4.5.0 is required for SM12 narrow-N NVFP4 tactics and the "
        "vendored FlashAttention-2 kernels; found "
        "${_CUTLASS_MAJOR}.${_CUTLASS_MINOR}.${_CUTLASS_PATCH} at "
        "${VLLM_CPP_CUTLASS_DIR}")
    endif()
    # Headers are present + new enough: usable by FA2 on ANY arch it resolves for.
    set(VLLM_CPP_CUTLASS_HEADERS ON)
    # Datacenter sm_100a NVFP4 tcgen05 build-verify TU (BACKEND-CUDA-SM100). Its
    # own flag, independent of the sm_12x VLLM_CPP_CUTLASS gate: resolves ON only
    # for a 100a target (cutlass-nvfp4-sm100 FEATURE-TABLE cell). DERIVED+BUILD-
    # VERIFIED — compiled + SASS-proven here, no B200 board ran it.
    if(VT_CUTLASS_NVFP4_SM100_ARCHS)
      set(VLLM_CPP_CUTLASS_NVFP4_SM100 ON)
      message(STATUS "CUTLASS found at ${VLLM_CPP_CUTLASS_DIR}; enabling sm_100a NVFP4 "
                     "tcgen05 build-verify TU (DERIVED+BUILD-VERIFIED, no board ran it)")
    else()
      set(VLLM_CPP_CUTLASS_NVFP4_SM100 OFF)
    endif()
    # Hopper sm_90a CUTLASS C3x FP8 scaled-mm wgmma/TMA build-verify TU
    # (BACKEND-CUDA-SM090). Its own flag, independent of the sm_12x
    # VLLM_CPP_CUTLASS gate: resolves ON only for a 90a target (scaledmm-c3x-sm90
    # FEATURE-TABLE cell). DERIVED+BUILD-VERIFIED — compiled + SASS-proven here, no
    # Hopper board ran it.
    if(VT_SCALEDMM_C3X_SM90_ARCHS)
      set(VLLM_CPP_SCALEDMM_C3X_SM90 ON)
      message(STATUS "CUTLASS found at ${VLLM_CPP_CUTLASS_DIR}; enabling sm_90a C3x FP8 "
                     "scaled-mm wgmma/TMA build-verify TU (DERIVED+BUILD-VERIFIED, no board ran it)")
    else()
      set(VLLM_CPP_SCALEDMM_C3X_SM90 OFF)
    endif()
    # Datacenter-Blackwell sm_100a CUTLASS C3x FP8 scaled-mm tcgen05 build-verify TU
    # (BACKEND-CUDA-SM100). Its own flag, independent of the sm_12x VLLM_CPP_CUTLASS
    # gate AND the Hopper VLLM_CPP_SCALEDMM_C3X_SM90 gate: resolves ON only for a
    # 100a target (scaledmm-c3x-sm100 FEATURE-TABLE cell). DERIVED+BUILD-VERIFIED —
    # compiled + SASS-proven here, no B200 board ran it.
    if(VT_SCALEDMM_C3X_SM100_ARCHS)
      set(VLLM_CPP_SCALEDMM_C3X_SM100 ON)
      message(STATUS "CUTLASS found at ${VLLM_CPP_CUTLASS_DIR}; enabling sm_100a C3x FP8 "
                     "scaled-mm tcgen05 build-verify TU (DERIVED+BUILD-VERIFIED, no board ran it)")
    else()
      set(VLLM_CPP_SCALEDMM_C3X_SM100 OFF)
    endif()
    if(VT_CUTLASS_NVFP4_ARCHS)
      set(VLLM_CPP_CUTLASS ON)
      message(STATUS "CUTLASS found at ${VLLM_CPP_CUTLASS_DIR}; enabling sm120a NVFP4 cutlass GEMM")
    else()
      # Ampere / non-sm_12x single-arch build: headers available for FA2, but the
      # sm_12x NVFP4 GEMM is not requested for this arch, so keep that flag OFF.
      set(VLLM_CPP_CUTLASS OFF)
      message(STATUS "CUTLASS headers found at ${VLLM_CPP_CUTLASS_DIR} "
                     "(available to FA2 for arch(es) [${VT_FA2_ARCHS}]); "
                     "sm120a NVFP4 GEMM not requested for this target")
    endif()
  else()
    set(VLLM_CPP_CUTLASS OFF)
    set(VLLM_CPP_CUTLASS_HEADERS OFF)
    set(VLLM_CPP_CUTLASS_NVFP4_SM100 OFF)
    set(VLLM_CPP_SCALEDMM_C3X_SM90 OFF)
    set(VLLM_CPP_SCALEDMM_C3X_SM100 OFF)
    # Two very different situations reach this branch, and only one of them is
    # benign. If VT_FA2_ARCHS resolved, the requested arch DOES support FA2 and
    # the build is about to drop it purely because no CUTLASS tree was found —
    # a silently slower binary, which is the failure a first-time builder hits
    # (they read "CUDA feature fa2: ENABLED for [86]" in the table above and
    # never notice the STATUS line that takes it away). Warn there. If no arch
    # wanted a CUTLASS-dependent feature in the first place, nothing is lost.
    if(VT_FA2_ARCHS)
      message(WARNING
        "CUTLASS headers NOT found, so FlashAttention-2 will NOT be built even "
        "though arch(es) [${VT_FA2_ARCHS}] support it. Attention falls back to "
        "the portable path and decode/prefill will be slower. Fix with "
        "-DVLLM_CPP_CUTLASS_FETCH=ON (downloads CUTLASS 4.5.0) or "
        "-DVLLM_CPP_CUTLASS_DIR=<path to a CUTLASS >=4.5.0 checkout).")
    else()
      message(STATUS "CUTLASS not found / no CUTLASS-dependent feature for this arch; "
                     "NVFP4 GEMM + FA2 disabled "
                     "(set -DVLLM_CPP_CUTLASS_DIR=<cutlass> or -DVLLM_CPP_CUTLASS_FETCH=ON)")
    endif()
  endif()
  # Marlin NVFP4 W4A16 grouped-MoE GEMM drop-in (the 35B expert kernel). Vendored
  # torch-free slice of vLLM's moe/marlin_moe_wna16 under src/vt/cuda/marlin/
  # (marlin-dropin-feasibility.md). NO external dependency — the kernel is
  # standalone CUDA + inline PTX. Enabled on whichever architectures the
  # `marlin-nvfp4` FEATURE-TABLE cell resolves to — sm_12xa (native sm_121 SASS)
  # and, since the Thor bring-up, sm_110. The arch set is DATA, so these strings
  # report the resolved cell instead of hardcoding one architecture; the previous
  # "sm_12xa"-only wording went stale the moment the cell gained a second arch.
  # The two vendored TUs + the vt launcher are isolated with their own compile
  # options (see below); gated by VT_MARLIN_NVFP4.
  option(VLLM_CPP_MARLIN "Build the vendored Marlin NVFP4 W4A16 MoE GEMM (marlin-nvfp4 archs)" ON)
  if(VLLM_CPP_MARLIN AND VT_MARLIN_NVFP4_ARCHS
     AND EXISTS "${CMAKE_SOURCE_DIR}/src/vt/cuda/marlin/core/scalar_type.hpp")
    set(VLLM_CPP_MARLIN ON)
    message(STATUS "Marlin NVFP4 W4A16 MoE GEMM enabled (vendored) for [${VT_MARLIN_NVFP4_ARCHS}]")
  else()
    set(VLLM_CPP_MARLIN OFF)
    message(STATUS "Marlin NVFP4 MoE GEMM disabled (needs a marlin-nvfp4 arch + vendored slice)")
  endif()
endif()

include(cmake/CompilerWarnings.cmake)
# Triton AOT kernels (cubins embedded in C launchers, CUDA-only), VENDORED
# per-arch under src/vt/cuda/triton_aot_vendored/. Defines the gated option
# VLLM_CPP_TRITON (default COMPUTED: ON for a CUDA build whose vendored trees are
# present, OFF otherwise; consumes the vendored artifacts — NO Python) +
# the maintainer knob VLLM_CPP_TRITON_REGEN (regenerates them with Python+Triton;
# scripts/regen-triton-aot.sh) + add_triton_kernel()/triton_aot_finalize(); with
# the option OFF this include is inert (no sources added). Wired below under
# CUDA for the SANCTIONED GDN fast-path (see .agents/discipline.md).
include(cmake/TritonAOT.cmake)

configure_file(include/vllm/version.h.in include/vllm/version.h @ONLY)

# THE COMPILED-ARCH MANIFEST (issue #1357, umbrella #1332 M2) starts EMPTY, and
# empty means "FlashAttention-2 was not compiled". It is filled in exactly one
# place: beside the vt_cuda_set_source_gencode call that emits FA2's -gencode
# options, so the manifest cannot claim an architecture nvcc was not asked to
# emit SASS for. Nothing fills it in a non-CUDA build, in a build with
# -DVLLM_CPP_FLASH_ATTN=OFF, in a build without CUTLASS headers, or in a build
# whose requested architectures do not intersect the feature table's fa2 row --
# and "not compiled" is the correct answer in every one of those cases.
#
# The header is rendered at the END of this file, after that assignment has had
# its chance to run. Derived, never hand-written: a parallel architecture list is
# CUDA_SUPPORTED_ARCHS, and a list that goes stale is precisely the defect.
set(VLLM_CPP_CUDA_FA2_COMPILED_ARCHS "")

add_library(vllm STATIC
  src/vllm/version.cpp
  src/vllm/sampling_params.cpp
  src/vllm/lora/punica_cpu.cpp
  src/vllm/lora/layers.cpp
  src/vllm/lora/mapping.cpp
  src/vllm/lora/peft_helper.cpp
  src/vllm/lora/lora_model.cpp
  src/vllm/config/cache.cpp
  src/vllm/config/scheduler.cpp
  src/vllm/config/device.cpp
  src/vllm/config/generation.cpp
  src/vllm/config/kv_transfer.cpp
  src/vllm/config/offload.cpp
  src/vllm/config/weight_residency.cpp
  src/vllm/model_executor/device_placement.cpp
  src/vllm/model_executor/weight_offloader.cpp
  src/vllm/model_executor/weight_offload_policy.cpp
  src/vllm/model_executor/expert_slot_cache.cpp
  src/vllm/model_executor/model_loader/gguf_expert_span.cpp
  src/vllm/model_executor/expert_streamer.cpp
  src/vllm/model_executor/device_expert_slot_store.cpp
  src/vllm/model_executor/expert_stream_seam.cpp
  src/vllm/config/multimodal.cpp
  src/vllm/config/speculative.cpp
  src/vllm/outputs.cpp
  src/vllm/transformers_utils/hf_config.cpp
  src/vllm/transformers_utils/tokenizer_files.cpp
  # ENG-HF-MODEL-DOWNLOAD (#1280): the HuggingFace cache layout. Pure file
  # system, no transport, so it builds in every lane. The DFlash draft path
  # in model_loader.cpp reaches it and is not behind VLLM_CPP_SERVER.
  src/vllm/transformers_utils/hf_cache.cpp
  src/vllm/model_executor/model_loader/read_only_file_mapping.cpp
  src/vllm/model_executor/model_loader/safetensors_reader.cpp
  src/vllm/model_executor/model_loader/gguf_reader.cpp
  src/vllm/model_executor/model_loader/gguf_dequant.cpp
  src/vllm/model_executor/model_loader/gguf_keep_quant.cpp
  src/vllm/model_executor/model_loader/gguf_device_fit.cpp
  src/vllm/model_executor/model_loader/nvfp4_dequant.cpp
  src/vllm/model_executor/model_loader/awq_gptq_dequant.cpp
  src/vllm/model_executor/model_loader/mxfp4_dequant.cpp
  src/vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.cpp
  src/vllm/model_executor/models/qwen3_5_weights.cpp
  src/vllm/model_executor/models/qwen3_5_dense_weights.cpp
  # MODEL-QWEN35-EXL3 (#2495): the EXL3 linear seam, in its own TU because
  # qwen3_5.cpp cannot include dense_attn_block.h (ADL collision on
  # ResidentWeight; see the header).
  src/vllm/model_executor/models/dense_exl3_linear.cpp
  src/vllm/model_executor/models/qwen3_5_mtp.cpp
  src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp
  src/vllm/model_executor/models/qwen3_gguf_weights.cpp
  src/vllm/model_executor/models/qwen3_5.cpp
  src/vllm/model_executor/models/qwen3_5_common.cpp
  src/vllm/model_executor/models/qwen3_5_dense.cpp
  src/vllm/model_executor/models/qwen3_5_moe.cpp
  # Qwen4-Exp (Qwen3.8-Flash-Next) W4: Qwen Sparse Attention. Host reference
  # math for the indexer, its side cache and the GATHER consumer (#1991).
  src/vllm/model_executor/models/qwen4_exp_qsa.cpp
  src/vllm/model_executor/models/qwen4_exp_qsa_block.cpp
  src/vllm/model_executor/models/qwen3_dense.cpp
  src/vllm/model_executor/models/qwen3_weights.cpp
  src/vllm/model_executor/models/qwen3.cpp
  src/vllm/model_executor/models/qwen3_dflash.cpp
  src/vllm/model_executor/models/qwen3_dflash2.cpp
  src/vllm/model_executor/models/qwen3_dspark.cpp
  src/vllm/model_executor/models/qwen3_dspark_weights.cpp
  src/vllm/model_executor/models/qwen3_dflash_gguf.cpp
  src/vllm/model_executor/models/qwen3_dflash_weights.cpp
  src/vllm/model_executor/models/qwen3_moe_registry.cpp
  src/vllm/model_executor/models/qwen3_moe_weights.cpp
  src/vllm/model_executor/models/qwen3_moe.cpp
  src/vllm/model_executor/models/qwen4_exp_hc.cpp
  src/vllm/model_executor/models/deepseek_v2_registry.cpp
  src/vllm/model_executor/models/deepseek_v2_weights.cpp
  src/vllm/model_executor/models/deepseek_v2.cpp
  src/vllm/model_executor/models/deepseek_v4_registry.cpp
  src/vllm/model_executor/models/deepseek_v4_mm.cpp
  # DeepSeek-V4.1 is a SIBLING of V4, not a widening of it: upstream forked
  # `vllm/models/deepseek_v4_1/` from `vllm/models/deepseek_v4/`, and the two
  # disagree on values the V4 parser hard-validates (compress_ratios {0,1,2}
  # against {0,4,128}). Its own TUs keep V4's landed gates out of the blast
  # radius. W1 only: config descent + registration, every other arm refuses.
  src/vllm/model_executor/models/deepseek_v4_1_registry.cpp
  src/vllm/model_executor/models/deepseek_v4_1.cpp
  # W3c (indexer arm) + W3d (`kv_source_layer`-keyed KV topology) host
  # references. Nothing reaches them yet: W4 owns the host forward that
  # composes W3a-W3f (ISSUE-LOCAL-01M2TMSCJJH7X8WZAM58PB3PMB, and the spec's
  # `## Owed`).
  src/vllm/model_executor/models/deepseek_v4_1_indexer.cpp
  src/vllm/model_executor/models/deepseek_v4_weights.cpp
  src/vllm/model_executor/models/deepseek_v4_vision_weights.cpp
  src/vllm/model_executor/models/deepseek_v4.cpp
  src/vllm/model_executor/models/deepseek_v4_dsa.cpp
  src/vllm/model_executor/models/deepseek_v4_dspark.cpp
  src/vllm/model_executor/models/deepseek_v4_compressor.cpp
  src/vllm/model_executor/models/deepseek_v4_exl3_device.cpp
  src/vllm/model_executor/models/deepseek_v4_mhc.cpp
  src/vllm/model_executor/models/deepseek_v4_moe.cpp
  src/vllm/model_executor/models/deepseek_v4_device.cpp
  src/vllm/model_executor/models/deepseek_v4_1_engram.cpp
  src/vllm/model_executor/models/deepseek_v4_1_mxfp8.cpp
  src/vllm/model_executor/models/dots3_note.cpp
  src/vllm/model_executor/models/dots3_note_attn.cpp
  src/vllm/model_executor/models/dots3_note_device.cpp
  src/vllm/model_executor/models/dots3_note_vision.cpp
  src/vllm/model_executor/models/dots3_note_audio.cpp
  src/vllm/model_executor/models/dots3_note_registry.cpp
  src/vllm/model_executor/models/mimo_v2_registry.cpp
  src/vllm/model_executor/models/mimo_v2_weights.cpp
  src/vllm/model_executor/models/mimo_v2.cpp
  # Kolibri-1 (MODEL-TEXT-kolibri-1) is a NEW architecture TU pair following
  # the mimo_v2 seam: registry + weights. W1 only: config descent, the
  # two-group KV-cache spec, and the FP8-block weight loader; the CPU
  # forward refuses by name until its wave.
  src/vllm/model_executor/models/kolibri1_registry.cpp
  src/vllm/model_executor/models/kolibri1_weights.cpp
  src/vllm/model_executor/models/qwen4_exp_ple.cpp
  src/vllm/model_executor/models/qwen4_exp.cpp
  src/vllm/model_executor/models/qwen4_exp_weights.cpp
  src/vllm/model_executor/models/qwen4_exp_moe.cpp
  src/vllm/model_executor/models/qwen4_exp_ple_block.cpp
  src/vllm/model_executor/models/qwen4_exp_forward.cpp
  src/vllm/model_executor/models/qwen4_exp_registry.cpp
  src/vllm/model_executor/models/glm5_next.cpp
  src/vllm/model_executor/models/glm5_next_kda.cpp
  src/vllm/model_executor/models/glm5_next_dsa.cpp
  src/vllm/model_executor/models/glm5_next_attn.cpp
  src/vllm/model_executor/models/glm5_next_mhc.cpp
  src/vllm/model_executor/models/glm5_next_moe.cpp
  src/vllm/model_executor/models/glm5_next_kv.cpp
  src/vllm/model_executor/models/glm5_next_registry.cpp
  src/vllm/model_executor/models/glm5_next_device.cpp
  src/vllm/model_executor/models/host_token_ids.cpp
  src/vllm/model_executor/models/glm_moe_dsa.cpp
  src/vllm/model_executor/models/glm_moe_dsa_forward.cpp
  src/vllm/model_executor/models/glm_moe_dsa_loader.cpp
  src/vllm/model_executor/models/glm_moe_dsa_registry.cpp
  src/vllm/model_executor/models/laguna_registry.cpp
  src/vllm/model_executor/models/laguna_weights.cpp
  src/vllm/model_executor/models/interfaces.cpp
  src/vllm/model_executor/models/laguna_ops.cpp
  src/vllm/model_executor/models/laguna_device.cpp
  src/vllm/model_executor/models/laguna_shared_fp4.cpp
  src/vllm/model_executor/models/laguna.cpp
  src/vllm/model_executor/models/muse_glimmer_registry.cpp
  src/vllm/model_executor/models/muse_glimmer_weights.cpp
  src/vllm/model_executor/models/muse_glimmer_gguf_weights.cpp
  src/vllm/model_executor/models/qwen4_exp_gguf_weights.cpp
  src/vllm/model_executor/models/glm5_next_weights.cpp
  src/vllm/model_executor/models/glm5_next_loader.cpp
  src/vllm/model_executor/models/glm5_next_bridge.cpp
  src/vllm/model_executor/models/glm5_next_layer.cpp
  src/vllm/model_executor/models/glm5_next_forward.cpp
  src/vllm/model_executor/models/glm5_next_diag.cpp
  src/vllm/model_executor/models/muse_glimmer.cpp
  src/vllm/model_executor/models/muse_glimmer_vision.cpp
  src/vllm/model_executor/models/muse_glimmer_mm.cpp
  # RADIO ViT-H tower + the Nemotron Nano VL projector (row
  # MODEL-MM-nano-nemotron-vl-nemotron-h-nano-vl-v2).
  src/vllm/model_executor/models/radio.cpp
  src/vllm/model_executor/models/nano_nemotron_vl.cpp
  src/vllm/model_executor/models/nano_nemotron_vl_registry.cpp
  src/vllm/model_executor/models/kimi_k3_registry.cpp
  src/vllm/model_executor/models/kimi_k3_weights.cpp
  src/vllm/model_executor/models/kimi_k3.cpp
  src/vllm/model_executor/models/minimax_h3.cpp
  src/vllm/model_executor/models/minimax_h3_packing.cpp
  src/vllm/model_executor/models/minimax_h3_planner.cpp
  src/vllm/model_executor/models/minimax_h3_gguf.cpp
  src/vllm/model_executor/models/minimax_h3_nvfp4.cpp
  src/vllm/model_executor/models/minimax_h3_sharded.cpp
  src/vllm/model_executor/models/minimax_h3_audio_vae.cpp
  src/vllm/model_executor/models/vocoder1d.cpp
  src/vllm/model_executor/models/minimax_music3_loader.cpp
  src/vllm/model_executor/models/minimax_music3_quant.cpp
  src/vllm/model_executor/models/minimax_music3_ar.cpp
  src/vllm/model_executor/models/minimax_music3_acoustic.cpp
  src/vllm/model_executor/models/minimax_music3_device.cpp
  src/vllm/model_executor/models/minimax_music3_depth_device.cpp
  src/vllm/model_executor/models/minimax_music3_llm.cpp
  src/vllm/model_executor/models/minimax_music3_speech.cpp
  src/vllm/model_executor/models/gpt2.cpp
  src/vllm/model_executor/models/deberta_v2.cpp
  src/vllm/model_executor/models/gliner2.cpp
  src/vllm/model_executor/models/gliner2_weights.cpp
  src/vllm/model_executor/models/gliner2_registry.cpp
  src/vllm/model_executor/models/gliner2_ner.cpp
  src/vllm/model_executor/models/cua_s1.cpp
  src/vllm/model_executor/models/cua_s1_weights.cpp
  src/vllm/model_executor/models/cua_s1_registry.cpp
  src/vllm/model_executor/models/modernbert.cpp
  src/vllm/model_executor/models/laya.cpp
  src/vllm/model_executor/models/laya_sequence.cpp
  src/vllm/model_executor/models/laya_weights.cpp
  src/vllm/model_executor/models/laya_registry.cpp
  src/vllm/multimodal/speech_engine.cpp
  src/vllm/model_executor/models/indextts2.cpp
  src/vllm/model_executor/models/campplus.cpp
  src/vllm/model_executor/models/emovec.cpp
  src/vllm/model_executor/models/w2vbert.cpp
  src/vllm/model_executor/models/w2v_fbank.cpp
  src/vllm/model_executor/models/fvq.cpp
  src/vllm/model_executor/models/codec_encoder.cpp
  src/vllm/model_executor/models/wavenet.cpp
  src/vllm/model_executor/models/dit_tail.cpp
  src/vllm/model_executor/models/dit_skip.cpp
  src/vllm/model_executor/models/dit_front.cpp
  src/vllm/model_executor/models/dit_stack.cpp
  src/vllm/model_executor/models/bigvgan.cpp
  src/vllm/model_executor/models/bigvgan_loader.cpp
  src/vllm/model_executor/models/indextts2_s2mel_loader.cpp
  src/vllm/model_executor/models/indextts2_conditioning.cpp
  src/vllm/model_executor/models/indextts2_render.cpp
  src/vllm/model_executor/models/indextts2_talker_loader.cpp
  src/vllm/model_executor/models/vocos.cpp
  src/vllm/model_executor/models/lenreg.cpp
  src/vllm/model_executor/models/cfm.cpp
  src/vllm/model_executor/models/adaln.cpp
  src/vllm/model_executor/models/dit.cpp
  src/vllm/model_executor/models/talker.cpp
  src/vllm/model_executor/models/tiktoken_bpe.cpp
  src/vllm/model_executor/models/indextts2_pipeline.cpp
  src/vllm/model_executor/models/minimax_h3_video_vae.cpp
  src/vllm/model_executor/models/minimax_h3_video_vae_device.cpp
  src/vllm/model_executor/models/minimax_h3_vae_cnn.cpp
  src/vllm/model_executor/models/minimax_h3_encoder.cpp
  src/vllm/model_executor/models/minimax_h3_pipeline.cpp
  src/vllm/model_executor/models/minimax_h3_wav.cpp
  src/vllm/model_executor/models/minimax_h3_mux.cpp
  src/vllm/model_executor/models/minimax_h3_device_resolve.cpp
  src/vllm/model_executor/models/minimax_h3_device.cpp
  src/vllm/model_executor/models/minimax_h3_vae_loader.cpp
  src/vllm/model_executor/models/minimax_h3_encoder_gguf.cpp
  src/vllm/model_executor/models/minimax_h3_encoder_sharded.cpp
  src/vllm/model_executor/models/minimax_h3_encoder_device.cpp
  src/vllm/model_executor/models/minimax_h3_vision_gguf.cpp
  # LTX-2.5 (row MODEL-DIFFUSION-LTX25, .agents/specs/ltx-2-5.md phase L2).
  src/vllm/model_executor/models/ltx2.cpp
  src/vllm/model_executor/models/ltx2_dit.cpp
  # LTX-2.5 (row MODEL-DIFFUSION-LTX25, phase L8): the DEVICE-RESIDENT forward and
  # the (kLtx2, device) resolver. The resolver is its own TU so the vt::GetOp cast
  # links in CPU-only builds, exactly as minimax_h3_device_resolve.cpp is.
  src/vllm/model_executor/models/ltx2_device_resolve.cpp
  src/vllm/model_executor/models/ltx2_video_vae_device_resolve.cpp
  src/vllm/model_executor/models/ltx2_device.cpp
  # LTX-2.5 (ROW MODEL-DIFFUSION-LTX25, phase L4): the Conv video VAE, the audio
  # VAE and its vocoder. Additive files mirroring ltx_core's own structure.
  src/vllm/model_executor/models/ltx2_audio_vae.cpp
  src/vllm/model_executor/models/ltx2_video_vae.cpp
  # LTX-2.5 (ROW LTX25-TILED-DECODE, issue #644): the tiling algebra and the
  # STREAMING tiled decode. Own translation units so the untiled decode above
  # stays the gated reference and the tiled path cannot drift from it by sharing
  # an internal helper only one of them exercises.
  src/vllm/model_executor/models/ltx2_tiling.cpp
  src/vllm/model_executor/models/ltx2_video_vae_tiled.cpp
  # LTX-2.5 (ROW MODEL-DIFFUSION-LTX25, phase L3): the Gemma-4 multi-layer feature
  # aggregation, the two caption projections and the embedded asset pack. An
  # additive file mirroring ltx_core's own structure.
  src/vllm/model_executor/models/ltx2_text_encoder.cpp
  # LTX-2.5 (ROW MODEL-DIFFUSION-LTX25, phase L5): the flow-matching pipeline,
  # the latent spatial upsampler, the duration head and the embeddings
  # connector. Additive files mirroring ltx_core's own structure.
  src/vllm/model_executor/models/ltx2_pipeline.cpp
  # LTX-2.5 (ROW LTX25-RES2S-LOOP, issue #921): the res_2s second-order sampler.
  # Its own TU because upstream partitions it that way — a STEPPER advances one
  # substep and lives in ltx-core/components/diffusion_steps.py, which
  # ltx2_pipeline.cpp mirrors, while a SAMPLER decides how many substeps there
  # are and what is evaluated between them and lives in
  # ltx-pipelines/utils/samplers.py. Two upstream packages, two files here.
  src/vllm/model_executor/models/ltx2_samplers.cpp
  src/vllm/model_executor/models/ltx2_upsampler.cpp
  # LTX-2.5 (ROW LTX25-DFR-PIPELINE, issue #986): the DFR canvas layout — the
  # keyframe segment grid, the temporal tile ranges and the latent stitch. Its
  # own TU because it holds no weights and runs no kernel: it is pure index
  # arithmetic whose every failure mode returns a correctly shaped, finite,
  # plausible latent, so it is gated by exact index vectors rather than by a
  # numeric band, which is a different kind of test from the rest of L5.
  src/vllm/model_executor/models/ltx2_dfr.cpp
  src/vllm/model_executor/models/ltx2_duration_head.cpp
  src/vllm/model_executor/models/ltx2_connector.cpp
  # LTX-2.5 (ROW MODEL-DIFFUSION-LTX25, phase L11): the conditioning items that
  # place the VAE ENCODERS' output into the denoise state.
  src/vllm/model_executor/models/ltx2_conditioning.cpp
  # LTX-2.5 (ROW LTX25-IMAGE-COND, issue #644): the video VAE ENCODER's LOAD
  # path — key filter and config parser, which existed nowhere before — and the
  # pixel front-end an image conditioning arrives through. Separate TUs rather
  # than additions to ltx2_loader.cpp, which two concurrent rows of the same
  # campaign need to edit.
  src/vllm/model_executor/models/ltx2_video_vae_encoder_load.cpp
  src/vllm/model_executor/models/ltx2_audio_vae_encoder_load.cpp
  src/vllm/model_executor/models/ltx2_audio_input.cpp
  src/vllm/model_executor/models/ltx2_iclora_reference.cpp
  src/vllm/model_executor/models/ltx2_image_preprocess.cpp
  # LTX-2.5 RETAKE (row LTX25-RETAKE, #924): the temporal region mask, the
  # truncate-or-pad conform, and the frame-directory source. Its own TU for the
  # same reason the two above are: ltx2_video.cpp is a concurrent-edit lock.
  src/vllm/model_executor/models/ltx2_retake.cpp
  # LTX-2.5 (ROW LTX25-T2A-ONE-STAGE, issue #1005): text-to-audio — the
  # audio-only pipeline and its guided denoiser. Its own TU for the same reason
  # the two above are, plus one this row adds: it is the only LTX pipeline that
  # runs the DiT with `video = nullptr`, and keeping that call in one file is
  # what makes "does anything else pass a video stream" a readable question.
  src/vllm/model_executor/models/ltx2_t2a.cpp
  # LTX-2.5 (ROW LTX25-GUIDED-VIDEO, issue #1092): `_guided_denoise` — the pass
  # assembly four unported pipelines are each blocked on, and the piece the
  # VIDEO denoise loop never had. Its own TU because upstream has its own file
  # (ltx-pipelines utils/denoisers.py) and because ltx2_pipeline.cpp, where the
  # guiders live, is a concurrent-edit lock.
  src/vllm/model_executor/models/ltx2_denoisers.cpp
  # LTX-2.5 (ROW MODEL-DIFFUSION-LTX25, phase L6): the quantized loaders — the
  # FP8 and torchao-NVFP4 DiT arms, the torchao-NVFP4 text encoder, and the
  # load-time device staging GB10's ATS penalty makes the default.
  src/vllm/model_executor/models/ltx2_loader.cpp
  # DiT LoRA (ROW ROAD-V1-DIT-LORA): the shared load-time-fusion seam for
  # diffusion transformers. Generalized from the LTX-2.5 implementation so both
  # LTX-2.5 and MiniMax-H3 use it. Its own TU to avoid locking the loader.
  src/vllm/model_executor/models/dit_lora.cpp
  src/vllm/entrypoints/openai/video_api.cpp
  src/vllm/entrypoints/openai/speech_api.cpp
  src/vllm/entrypoints/openai/speech_api_synthesize.cpp
  src/vllm/model_executor/models/kimi_kda.cpp
  src/vllm/model_executor/models/kimi_linear_registry.cpp
  src/vllm/model_executor/models/kimi_linear_weights.cpp
  src/vllm/model_executor/models/kimi_linear_forward.cpp
  src/vllm/model_executor/models/kimi_linear.cpp
  src/vllm/model_executor/models/kimi_linear_device.cpp
  # MODEL-NEMOTRON-H W3 (#517): the additive NemotronH structural bring-up —
  # config descent, the on-disk name map, the registry entry and the het-KV
  # topology. No forward: nemotron_h.cpp is W4.
  src/vllm/model_executor/models/nemotron_h.cpp
  # A2-R (#810): the DEVICE arm — the residual stream, the 52 norms, the
  # embedding lookup and the 6 GQA attention blocks. Mamba/MoE/lm_head stay
  # host; see the file header for why the line falls where it does.
  src/vllm/model_executor/models/nemotron_h_device.cpp
  src/vllm/model_executor/models/nemotron_h_registry.cpp
  src/vllm/model_executor/models/nemotron_h_weights.cpp
  src/vllm/model_executor/models/glm4_registry.cpp
  src/vllm/model_executor/models/glm4_weights.cpp
  src/vllm/model_executor/models/glm4.cpp
  src/vllm/model_executor/models/glm4_moe_lite_registry.cpp
  src/vllm/model_executor/models/opt_registry.cpp
  src/vllm/model_executor/models/opt_weights.cpp
  src/vllm/model_executor/models/opt.cpp
  src/vllm/model_executor/models/llama_registry.cpp
  src/vllm/model_executor/models/llama_embedding_registry.cpp
  src/vllm/model_executor/models/kev_head.cpp
  src/vllm/model_executor/models/kev_registry.cpp
  src/vllm/model_executor/models/clm_head.cpp
  src/vllm/model_executor/models/clm_registry.cpp
  src/vllm/model_executor/models/gliner25_decide_head.cpp
  src/vllm/model_executor/models/gliner25_decide_registry.cpp
  src/vllm/model_executor/models/xor_registry.cpp
  src/vllm/model_executor/models/tev1_registry.cpp
  src/vllm/model_executor/models/tev1_inference.cpp
  src/vllm/model_executor/models/decision_scorer.cpp
  src/vllm/model_executor/models/nimble_inference.cpp
  src/vllm/model_executor/models/nimble_registry.cpp
  src/vllm/model_executor/models/llama_weights.cpp
  src/vllm/model_executor/models/mistral_registry.cpp
  src/vllm/model_executor/models/mistral_weights.cpp
  src/vllm/model_executor/models/gemma4_registry.cpp
  src/vllm/model_executor/models/gemma4_weights.cpp
  src/vllm/model_executor/models/gemma4_moe.cpp
  src/vllm/model_executor/models/gemma4.cpp
  src/vllm/model_executor/models/gemma4_mm.cpp
  src/vllm/model_executor/models/gemma3_registry.cpp
  src/vllm/model_executor/models/gemma3_weights.cpp
  src/vllm/model_executor/models/gemma3.cpp
  src/vllm/model_executor/models/gemma2_registry.cpp
  src/vllm/model_executor/models/gemma2_weights.cpp
  src/vllm/model_executor/models/gemma2.cpp
  src/vllm/model_executor/models/gemma_registry.cpp
  src/vllm/model_executor/models/gemma_weights.cpp
  src/vllm/model_executor/models/gemma.cpp
  src/vllm/model_executor/models/olmo2_registry.cpp
  src/vllm/model_executor/models/olmo2_weights.cpp
  src/vllm/model_executor/models/olmo2.cpp
  src/vllm/model_executor/models/phi3_registry.cpp
  src/vllm/model_executor/models/phi3_weights.cpp
  src/vllm/model_executor/models/phi3.cpp
  src/vllm/model_executor/models/granite_registry.cpp
  src/vllm/model_executor/models/granite_weights.cpp
  src/vllm/model_executor/models/granite.cpp
  src/vllm/model_executor/models/stablelm_registry.cpp
  src/vllm/model_executor/models/stablelm_weights.cpp
  src/vllm/model_executor/models/stablelm.cpp
  src/vllm/model_executor/models/minicpm_registry.cpp
  src/vllm/model_executor/models/minicpm_weights.cpp
  src/vllm/model_executor/models/minicpm.cpp
  src/vllm/model_executor/models/minicpm3_registry.cpp
  src/vllm/model_executor/models/minicpm3_weights.cpp
  src/vllm/model_executor/models/minicpm3.cpp
  src/vllm/model_executor/models/internlm2_registry.cpp
  src/vllm/model_executor/models/internlm2_weights.cpp
  src/vllm/model_executor/models/commandr_registry.cpp
  src/vllm/model_executor/models/commandr_weights.cpp
  src/vllm/model_executor/models/commandr.cpp
  src/vllm/model_executor/models/cohere2_moe_registry.cpp
  src/vllm/model_executor/models/cohere2_moe_weights.cpp
  src/vllm/model_executor/models/cohere2_moe.cpp
  src/vllm/model_executor/models/phi_registry.cpp
  src/vllm/model_executor/models/phi_weights.cpp
  src/vllm/model_executor/models/phi.cpp
  src/vllm/model_executor/models/model_registry.cpp
  src/vllm/model_executor/layers/quantization/fp8_block_quant.cpp
  src/vllm/plugins/plugins.cpp
  src/vllm/model_executor/models/qwen3_vl_vision.cpp
  # LOAD-GGUF-MMPROJ (#821): the llama.cpp `clip` mmproj reader that fills the
  # SAME Qwen3VLVisionWeights from a SECOND GGUF file beside the language one.
  src/vllm/model_executor/models/clip_mmproj_gguf.cpp
  src/vllm/model_executor/models/gemma4_vision.cpp
  src/vllm/model_executor/models/deepseek_v4_vision.cpp
  src/vllm/model_executor/models/gemma4_audio.cpp
  src/vllm/model_executor/models/whisper_audio.cpp
  src/vllm/model_executor/models/qwen3_vl_text.cpp
  src/vllm/model_executor/models/qwen3_vl.cpp
  src/vllm/model_executor/models/qwen3_vl_registry.cpp
  src/vllm/model_executor/models/voxtral.cpp
  src/vllm/multimodal/hasher.cpp
  src/vllm/multimodal/utils.cpp
  # Per-modality input-count limits and the refusal they carry
  # (ENG-MM-INPUT-PIPELINE L1, #607).
  src/vllm/multimodal/processing/context.cpp
  src/vllm/multimodal/processing/processor.cpp
  src/vllm/multimodal/dots3_note_processor.cpp
  src/vllm/multimodal/mel_filter_bank.cpp
  # Pillow's BICUBIC resampler, which is NOT torchvision's and NOT a four-tap
  # cubic (dots3-note W6c, #2537).
  src/vllm/multimodal/pil_resize.cpp
  src/vllm/multimodal/qwen3vl_processor.cpp
  # Nemotron Nano VL / Omni dynamic-resolution tiler and torch's antialiased
  # bicubic resize (a third algorithm beside Pillow's and torchvision's).
  src/vllm/multimodal/nano_nemotron_vl_processor.cpp
  src/vllm/multimodal/deepseek_v4_processor.cpp
  src/vllm/multimodal/audio_processor.cpp
  src/vllm/multimodal/audio_resample.cpp
  # Parakeet / FastConformer audio encoder + CTC head + its log-mel front end
  # (spike .agents/specs/parakeet-conformer-encoder.md row P4). A standalone
  # forward composed from the P1-P3 vt:: kernels, NOT a registry architecture —
  # vLLM does not register one either (parakeet.py is a component of
  # nano_nemotron_vl.py, and it delegates the encoder itself to transformers).
  src/vllm/model_executor/models/parakeet_encoder.cpp
  # The RNN-T / TDT transducer heads over that encoder (row P5). Upstream after
  # all: transformers `main` implements ParakeetRNNTDecoder / ParakeetForRNNT /
  # ParakeetForTDT, which the P4 spike wrongly recorded as absent because it
  # measured the LOCALLY INSTALLED 5.3.0. Corrected in the spike.
  src/vllm/model_executor/models/parakeet_transducer.cpp
  src/vllm/model_executor/models/parakeet_weights.cpp
  src/vllm/multimodal/parakeet_audio_processor.cpp
  # The ONE transcription seam every consumer drives (C ABI vllm_transcribe,
  # the server's /v1/audio/transcriptions, the parakeet-transcribe example) —
  # ARCH-ONE-SURFACE ROW 1: absorbs the pipeline examples/parakeet_transcribe
  # used to own privately. The refuse-by-task registry TU registers the three
  # Parakeet head archs so config.json RESOLVES (SupportsTranscription mirror).
  src/vllm/multimodal/parakeet_transcription.cpp
  src/vllm/model_executor/models/parakeet_registry.cpp
  # Diarization seam (wraps parakeet.cpp's C-API). Gated by
  # VLLM_CPP_WITH_DIARIZATION; the .cpp compiles as stubs when disabled.
  src/vllm/multimodal/diarization.cpp
  # The ONE video-generation seam every consumer drives (C ABI vllm_video_*,
  # the server's /v1/videos, the minimax-h3-gen example) — ARCH-ONE-SURFACE
  # ROW 2: absorbs the assembly pipeline examples/minimax_h3_gen and the
  # server's private /v1/videos twin. Builds artifacts + the ffmpeg argv and
  # SPAWNS NOTHING (the ratified process boundary).
  src/vllm/multimodal/minimax_h3_video.cpp
  # The GENERALIZED video seam (vllm::multimodal::VideoEngine) + its
  # checkpoint-detected family registry — LTX-2.5 L1, .agents/specs/ltx-2-5.md
  # §5. Knows about no family; each family self-registers from its own TU.
  src/vllm/multimodal/video_engine.cpp
  # LTX-2.5 behind that seam — the SECOND family, and the driving loop that
  # composes the L2-L6 bricks into frames + a waveform (phase L7). Additive:
  # one file plus one REGISTER_VLLM_VIDEO_FAMILY line, which is the property
  # the seam exists to have.
  # W0 of LTX25-DEVICE-RESIDENCY (#1010): where a render spends its wall, and
  # how many bytes it held while it did. Family-neutral, driven by the LTX-2.5
  # driver today.
  src/vllm/multimodal/render_phase_log.cpp
  src/vllm/multimodal/ltx2_video.cpp
  src/vllm/v1/core/encoder_cache_manager.cpp
  src/vllm/tokenizer/unicode_data.cpp
  src/vllm/tokenizer/pretokenizer.cpp
  src/vllm/tokenizer/bpe.cpp
  src/vllm/tokenizer/tokenizer.cpp
  src/vllm/v1/request.cpp
  src/vllm/v1/outputs.cpp
  src/vllm/v1/sample/metadata.cpp
  src/vllm/v1/sample/ops/penalties.cpp
  src/vllm/v1/sample/ops/bad_words.cpp
  src/vllm/v1/sample/logits_processor/builtin.cpp
  src/vllm/v1/sample/sampler.cpp
  src/vllm/v1/spec_decode/metrics.cpp
  src/vllm/v1/spec_decode/rejection_sampler.cpp
  src/vllm/v1/spec_decode/ngram_proposer.cpp
  src/vllm/v1/spec_decode/draft_model_proposer.cpp
  src/vllm/v1/worker/gpu/spec_decode/autoregressive/prepare_prefill_inputs.cpp
  src/vllm/v1/worker/gpu/spec_decode/autoregressive/prepare_decode_inputs.cpp
  src/vllm/v1/worker/gpu/spec_decode/mtp/speculator.cpp
  src/vllm/v1/worker/gpu/spec_decode/dflash/speculator.cpp
  src/vllm/v1/worker/gpu/spec_decode/dflash2/speculator.cpp
  src/vllm/v1/worker/gpu/spec_decode/dspark/speculator.cpp
  src/vllm/v1/kv_cache_interface.cpp
  src/vllm/v1/kv_cache_spec_registry.cpp
  src/vllm/v1/core/hybrid_kv_budget.cpp
  src/vllm/v1/core/kv_cache_utils.cpp
  src/vllm/distributed/kv_events.cpp
  src/vllm/v1/core/block_pool.cpp
  src/vllm/v1/core/single_type_kv_cache_manager.cpp
  src/vllm/v1/core/kv_cache_coordinator.cpp
  src/vllm/v1/core/kv_cache_manager.cpp
  src/vllm/v1/metrics/prometheus.cpp
  src/vllm/v1/metrics/loggers.cpp
  src/vllm/v1/kv_offload/cache_identity.cpp
  src/vllm/v1/kv_offload/cache_policy.cpp
  src/vllm/v1/kv_offload/cpu_manager.cpp
  src/vllm/v1/kv_offload/kv_block_transfer.cpp
  src/vllm/v1/kv_offload/fs_io.cpp
  src/vllm/v1/kv_offload/fs_tier.cpp
  src/vllm/v1/kv_offload/tiering_manager.cpp
  src/vllm/v1/kv_offload/kv_connector.cpp
  # LMCache MODE-1 (lm://) wire codec + blocking TCP client — KV-EXTERNAL-CACHE
  # W1 (codec) + W2 (client).  Pure CPU; the W3 LMCacheConnector subclass
  # (below) is the KVConnector that drives them over the W5 seam.
  src/vllm/v1/kv_offload/lmcache/memory_format.cpp
  src/vllm/v1/kv_offload/lmcache/remote_protocol.cpp
  src/vllm/v1/kv_offload/lmcache/cache_engine_key.cpp
  src/vllm/v1/kv_offload/lmcache/token_hasher.cpp
  # W4 peer-agreeing key derivation (ChunkedTokenDatabase: vLLM sha256_cbor,
  # chunk_size 256) — byte-identical CacheEngineKeys vs a real Python peer.
  src/vllm/v1/kv_offload/lmcache/chunked_token_database.cpp
  src/vllm/v1/kv_offload/lmcache/remote_client.cpp
  # LMCache lm:// KVConnector subclass over the W5 seam — KV-EXTERNAL-CACHE W3.
  src/vllm/v1/kv_offload/lmcache/lmcache_connector.cpp
  src/vllm/v1/core/sched/request_queue.cpp
  src/vllm/v1/core/sched/output.cpp
  src/vllm/v1/core/sched/scheduler.cpp
  src/vllm/v1/core/sched/async_scheduler.cpp
  src/vllm/v1/core/sched/utils.cpp
  src/vllm/v1/worker/gpu/block_table.cpp
  src/vllm/v1/worker/gpu/cudagraph_dispatch.cpp
  src/vllm/v1/worker/gpu/input_batch.cpp
  src/vllm/v1/worker/gpu/prepare_inputs.cpp
  src/vllm/v1/worker/gpu/async_output.cpp
  src/vllm/v1/worker/gpu/runner.cpp
  src/vllm/platform/process.cpp
  src/vllm/platform/console_shutdown.cpp
  src/vllm/platforms/platform.cpp
  src/vllm/platforms/cuda_arch_manifest.cpp
  src/vllm/platforms/cpu.cpp
  src/vllm/v1/attention/backend.cpp
  src/vllm/v1/attention/registry.cpp
  src/vllm/v1/attention/backends/utils.cpp
  src/vllm/v1/attention/backends/cpu_attn.cpp
  src/vllm/v1/attention/backends/gdn_attn.cpp
  src/vllm/v1/attention/backends/mla/sparse_swa.cpp
  src/vllm/v1/attention/backends/mla/indexer.cpp
  src/vllm/v1/structured_output/backend_types.cpp
  src/vllm/v1/structured_output/backend_native.cpp
  src/vllm/v1/structured_output/backend_xgrammar.cpp
  src/vllm/v1/structured_output/xgrammar_json_schema.cpp
  src/vllm/v1/structured_output/json_schema_to_gbnf.cpp
  src/vllm/v1/structured_output/jump_forward.cpp
  src/vllm/v1/structured_output/request.cpp
  src/vllm/v1/structured_output/manager.cpp
  src/vllm/v1/engine/types.cpp
  src/vllm/v1/engine/detokenizer.cpp
  src/vllm/v1/engine/core.cpp
  src/vllm/v1/engine/core_proc.cpp
  src/vllm/v1/engine/core_client.cpp
  src/vllm/v1/engine/async_llm.cpp
  src/vllm/v1/engine/input_processor.cpp
  src/vllm/v1/engine/parallel_sampling.cpp
  src/vllm/v1/engine/output_processor.cpp
  src/vllm/v1/engine/logprobs.cpp
  src/vllm/v1/engine/llm_engine.cpp
  src/vllm/entrypoints/beam_search.cpp
  src/vllm/v1/executor/executor.cpp
  src/vllm/model_executor/layers/attention/attention.cpp
  src/vllm/model_executor/layers/attention/chunked_local_attention.cpp
  src/vllm/model_executor/layers/pooler/methods.cpp
  src/vllm/model_executor/layers/pooler/activations.cpp
  src/vllm/model_executor/layers/pooler/heads.cpp
  src/vllm/model_executor/layers/pooler/poolers.cpp
  src/vllm/model_executor/layers/pooler/dispatch_pooler.cpp
  src/vllm/v1/worker/gpu/pool/pooling_runner.cpp
  src/vllm/model_executor/layers/attention/mla_attention.cpp
  src/vllm/model_executor/layers/rotary_embedding/base.cpp
  src/vllm/model_executor/layers/rotary_embedding/common.cpp
  src/vllm/model_executor/layers/rotary_embedding/dynamic_ntk_alpha_rope.cpp
  src/vllm/model_executor/layers/rotary_embedding/dynamic_ntk_scaling_rope.cpp
  src/vllm/model_executor/layers/rotary_embedding/llama3_rope.cpp
  src/vllm/model_executor/layers/rotary_embedding/linear_scaling_rope.cpp
  src/vllm/model_executor/layers/rotary_embedding/phi3_long_rope_scaled_rope.cpp
  src/vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.cpp
  src/vllm/model_executor/layers/rotary_embedding/mrope.cpp
  src/vllm/entrypoints/chat_template.cpp
  src/vllm/entrypoints/model_loader.cpp
  src/vllm/entrypoints/openai/systemone.cpp
  src/capi/vllm_c.cpp
  src/capi/chat_prompt.cpp
  src/vllm/entrypoints/openai/protocol.cpp
  src/vllm/entrypoints/openai/chat_mm.cpp
  # ENG-MM-INPUT-PIPELINE (#2475): the per-architecture multimodal chat seam
  # registry, and Qwen3-VL's registration into it. The registration TU is
  # SELF-REGISTERING and is reached through vllm's INTERFACE --whole-archive.
  src/vllm/entrypoints/openai/mm_chat_registry.cpp
  src/vllm/entrypoints/openai/mm_chat_deepseek_v4.cpp
  src/vllm/entrypoints/openai/mm_chat_nano_nemotron_vl.cpp
  src/vllm/entrypoints/openai/mm_chat_dots3note.cpp
  src/vllm/entrypoints/openai/mm_chat_qwen3vl.cpp
  src/vllm/entrypoints/openai/serving_utils.cpp
  src/vllm/entrypoints/openai/serving_completion.cpp
  src/vllm/entrypoints/openai/serving_chat.cpp
  src/vllm/entrypoints/openai/request_logger.cpp
  src/vllm/entrypoints/openai/serving_models.cpp
  src/vllm/entrypoints/openai/run_batch.cpp
  src/vllm/entrypoints/openai/tool_parsers/abstract.cpp
  src/vllm/entrypoints/openai/tool_parsers/structural_tags.cpp
  src/vllm/entrypoints/openai/tool_parsers/detect.cpp
  src/vllm/entrypoints/openai/tool_parsers/utils.cpp
  src/vllm/entrypoints/openai/tool_parsers/hermes.cpp
  src/vllm/entrypoints/openai/tool_parsers/mistral.cpp
  src/vllm/entrypoints/openai/tool_parsers/llama.cpp
  src/vllm/entrypoints/openai/tool_parsers/llama4_pythonic.cpp
  src/vllm/entrypoints/openai/tool_parsers/pythonic_core.cpp
  src/vllm/entrypoints/openai/tool_parsers/pythonic.cpp
  src/vllm/entrypoints/openai/tool_parsers/qwen3.cpp
  src/vllm/entrypoints/openai/tool_parsers/deepseek_v3.cpp
  src/vllm/entrypoints/openai/tool_parsers/deepseek_v31.cpp
  src/vllm/entrypoints/openai/tool_parsers/deepseek_v32.cpp
  src/vllm/entrypoints/openai/tool_parsers/deepseek_v4.cpp
  src/vllm/entrypoints/openai/tool_parsers/longcat.cpp
  src/vllm/entrypoints/openai/tool_parsers/granite.cpp
  src/vllm/entrypoints/openai/tool_parsers/granite4.cpp
  src/vllm/entrypoints/openai/tool_parsers/granite_20b_fc.cpp
  src/vllm/entrypoints/openai/reasoning_parsers/abstract.cpp
  src/vllm/entrypoints/openai/reasoning_parsers/detect.cpp
  src/vllm/entrypoints/openai/reasoning_parsers/think_auto.cpp
  src/vllm/entrypoints/openai/reasoning_parsers/basic.cpp
  src/vllm/entrypoints/openai/reasoning_parsers/deepseek_r1.cpp
  src/vllm/entrypoints/openai/reasoning_parsers/identity.cpp
  src/vllm/entrypoints/openai/reasoning_parsers/deepseek_v3.cpp
  src/vllm/entrypoints/openai/reasoning_parsers/mistral.cpp
  src/vllm/entrypoints/openai/reasoning_parsers/minimax_m2.cpp
  src/vllm/entrypoints/openai/reasoning_parsers/step3.cpp
  src/vllm/entrypoints/openai/reasoning_parsers/olmo3.cpp
  src/vllm/entrypoints/openai/reasoning_parsers/parser_engine_adapter.cpp
  src/vllm/entrypoints/openai/tool_parsers/xlam.cpp
  src/vllm/entrypoints/openai/tool_parsers/phi4_mini.cpp
  src/vllm/entrypoints/openai/tool_parsers/internlm.cpp
  src/vllm/entrypoints/openai/tool_parsers/jamba.cpp
  src/vllm/entrypoints/openai/tool_parsers/step3.cpp
  src/vllm/entrypoints/openai/tool_parsers/step3p5.cpp
  src/vllm/entrypoints/openai/tool_parsers/qwen3_coder.cpp
  src/vllm/entrypoints/openai/tool_parsers/minicpm5.cpp
  src/vllm/entrypoints/openai/tool_parsers/hy_v3.cpp
  src/vllm/entrypoints/openai/tool_parsers/hunyuan_a13b.cpp
  src/vllm/entrypoints/openai/tool_parsers/apertus.cpp
  src/vllm/entrypoints/openai/tool_parsers/ernie45.cpp
  src/vllm/entrypoints/openai/tool_parsers/gigachat3.cpp
  src/vllm/entrypoints/openai/tool_parsers/olmo3.cpp
  src/vllm/entrypoints/openai/tool_parsers/lfm2.cpp
  src/vllm/entrypoints/openai/tool_parsers/poolside_v1.cpp
  src/vllm/entrypoints/openai/tool_parsers/functiongemma.cpp
  src/vllm/entrypoints/openai/tool_parsers/kimi_k2.cpp
  src/vllm/entrypoints/openai/tool_parsers/glm47.cpp
  src/vllm/entrypoints/openai/tool_parsers/minimax_m2.cpp
  src/vllm/entrypoints/openai/tool_parsers/gemma4.cpp
  src/vllm/entrypoints/openai/tool_parsers/seed_oss.cpp
  src/vllm/entrypoints/openai/tool_parsers/muse_glimmer.cpp
  src/vllm/entrypoints/openai/tool_parsers/parser_engine_adapter.cpp
  src/vllm/entrypoints/openai/reasoning_parsers/muse_glimmer.cpp
  src/vllm/parser/engine/incremental_lexer.cpp
  src/vllm/parser/engine/token_id_scanner.cpp
  src/vllm/parser/engine/streaming_parser_engine.cpp
  src/vllm/parser/engine/configs.cpp
  src/vllm/parser/engine/registry.cpp
  src/vllm/parser/engine/parser_engine.cpp
  src/vllm/parser/kimi_k2.cpp
  src/vllm/parser/qwen3.cpp
  src/vllm/parser/glm47_moe.cpp
  src/vllm/parser/gemma4.cpp
  src/vllm/parser/inkling.cpp
  src/vllm/parser/parser_manager.cpp
  src/vt/dtype.cpp
  src/vt/backend.cpp
  src/vt/breakable_graph.cpp
  src/vt/persistent_step_input.cpp
  src/vt/tensor.cpp
  src/vt/arena.cpp
  src/vt/op_provider.cpp
  src/vt/communicator.cpp
  src/vt/ops.cpp
  src/vt/exl3_policy.cpp
  src/vt/fused_ops.cpp
  src/vt/merged_gemm.cpp
  src/vt/cpu/cpu_backend.cpp
  src/vt/cpu/cpu_threadpool.cpp
  src/vt/cpu/cpu_ops.cpp
  src/vt/compiled_gemma.cpp
  src/vt/cpu/cpu_compiled_gemma.cpp
  src/vt/cpu/cpu_residual_rmsnorm.cpp
  src/vt/cpu/cpu_minimax_h3.cpp
  src/vt/cpu/cpu_ltx2.cpp
  src/vt/cpu/cpu_ltx2_vae.cpp
  src/vt/cpu/cpu_qwen4_exp_ple.cpp
  src/vt/cpu/cpu_qwen4_exp.cpp
  src/vt/cpu/cpu_qwen4_exp_qsa.cpp
  src/vt/cpu/cpu_isa_arm.cpp
  src/vt/cpu/cpu_isa_x86.cpp
  src/vt/cpu/cpu_matmul_elem.cpp
  src/vt/cpu/cpu_quant_dequant.cpp
  src/vt/cpu/cpu_exl3_dequant.cpp
  src/vt/cpu/cpu_exl3_kernels.cpp
  src/vt/cpu/cpu_quant_act.cpp
  src/vt/cpu/cpu_quant_dot.cpp
  src/vt/cpu/cpu_quant_dot_sdot.cpp
  src/vt/cpu/cpu_quant_dot_arm.cpp
  src/vt/cpu/cpu_quant_repack.cpp
  src/vt/cpu/cpu_quant_repack_arm.cpp
  src/vt/cpu/cpu_quant_traits.cpp
  src/vt/cpu/cpu_quant_gemm.cpp
  src/vt/cpu/cpu_layernorm.cpp
  # Conformer / FastConformer audio-encoder kernels (spike
  # .agents/specs/parakeet-conformer-encoder.md rows P1/P2/P3). Self-registering
  # translation units, reached through vllm's INTERFACE --whole-archive option.
  src/vt/cpu/cpu_conv2d.cpp
  src/vt/cpu/cpu_conv1d_depthwise.cpp
  # BigVGAN / DAC vocoder 1-D convolutions (#672). Self-registering, reached the
  # same --whole-archive way as the conformer kernels above.
  src/vt/cpu/cpu_conv1d_general.cpp
  # General 3-D convolution (LTX25-DEVICE-RESIDENCY W5, #1007) — the primitive
  # the LTX-2.5 video VAE decode had on no device. Self-registering.
  src/vt/cpu/cpu_conv3d.cpp
  src/vt/cpu/cpu_attn_relpos.cpp
  src/vt/cpu/cpu_cache.cpp
  src/vt/cpu/cpu_mla_attn.cpp
  src/vt/cpu/cpu_mla_prefill.cpp
  # The DSA "Lightning Indexer" selection pair (dots3-note W4b-3c, #699).
  # Self-registering.
  src/vt/cpu/cpu_dsa_indexer.cpp
  src/vt/cpu/cpu_paged_attn.cpp
  src/vt/cpu/cpu_sample.cpp)
add_library(vllm::vllm ALIAS vllm)

# QUANT-GGUF-CIQ-GEMM G6: the Arm i8mm (mmla) quant-GEMM tier needs the ISA
# extension enabled so <arm_neon.h> exposes `vmmlaq_s32` and defines
# `__ARM_FEATURE_MATMUL_INT8`. The project sets no global -march, so build ONLY
# this translation unit with armv8.2-a+i8mm (surgical, exactly like the x86
# tier's dedicated F16C translation unit): every other TU stays on the default
# baseline, and the kernels are additionally runtime-gated on HWCAP2_I8MM so the
# binary still runs on non-i8mm aarch64 (portable tier serves those shapes).
# KERNEL-GEMM-CPU-ELEM-X86WIDE W1: the AVX2 elementwise tier needs the ISA
# enabled so <immintrin.h> exposes the 256-bit intrinsics. Same surgical shape
# as the aarch64 i8mm block below and the same discipline llama.cpp uses when it
# builds the CPU backend once per ISA level (ggml/src/CMakeLists.txt:371-401):
# build ONLY this TU wide, leave every other TU on the baseline, and gate entry
# on a runtime __builtin_cpu_supports probe so the binary still runs on
# pre-AVX2 x86-64 (the SSE2 tier serves those). The global -ffp-contract=off
# still applies on top, which is what keeps mul-then-add non-fused and therefore
# byte-identical to the portable tier.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  # x86-only TUs: <immintrin.h> does not exist on aarch64, so they are ADDED
  # here rather than in the unconditional source list.
  target_sources(vllm PRIVATE
    src/vt/cpu/cpu_matmul_elem_f16c.cpp
    src/vt/cpu/cpu_matmul_elem_avx2.cpp
    src/vt/cpu/cpu_matmul_elem_avx512.cpp)
  set_source_files_properties(
    src/vt/cpu/cpu_matmul_elem_f16c.cpp PROPERTIES
    COMPILE_OPTIONS
      "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/arch:AVX>;$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-mf16c>")
  set_source_files_properties(
    src/vt/cpu/cpu_matmul_elem_avx2.cpp PROPERTIES
    COMPILE_OPTIONS
      "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/arch:AVX2>;$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-mavx2;-mf16c>")
  set_source_files_properties(
    src/vt/cpu/cpu_matmul_elem_avx512.cpp PROPERTIES
    COMPILE_OPTIONS
      "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/arch:AVX512>;$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-mavx512f;-mavx512bw;-mavx512vl;-mf16c>")
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  # The portable Q8 DotProd C++ tier is valid on Linux and Darwin. Compile only
  # its TU with DotProd enabled; runtime entry is gated by Linux HWCAP or Darwin
  # sysctl. The separately scheduled A76 assembly remains Linux-only below.
  set_source_files_properties(
    src/vt/cpu/cpu_quant_dot_sdot.cpp PROPERTIES
    COMPILE_DEFINITIONS VT_CPU_ARM_Q8_DOT=1
    COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:-march=armv8.2-a+dotprod+fp16>")
  # KERNEL-CPU-A76-Q8-DOT is Linux-only because its dispatch uses HWCAP/MIDR
  # and the hand-written file carries ELF assembler directives. Other arm64
  # platforms compile the generic stub and retain their existing portable tier.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    enable_language(ASM)
    target_sources(vllm PRIVATE src/vt/cpu/cpu_quant_dot_a76.S)
    set_property(SOURCE src/vt/cpu/cpu_quant_dot_sdot.cpp APPEND PROPERTY
      COMPILE_DEFINITIONS VT_CPU_A76_Q8_DOT=1)
  endif()
  # G6 mmla tier AND G7 repack tier (cpu_quant_repack_arm.cpp) both need i8mm.
  # The global -ffp-contract=off (top of this file) still applies on top of this
  # per-file -march, which is what keeps the repack gemm/gemv's non-fused
  # `vmlaq_f32` bit-identical to the tier-0 path.
  set_source_files_properties(
    src/vt/cpu/cpu_quant_dot_arm.cpp
    src/vt/cpu/cpu_quant_repack_arm.cpp PROPERTIES
    COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:-march=armv8.2-a+i8mm+dotprod>")
endif()
# Static-init registrars (e.g. the CPU backend, the platform table, the op
# table, the attention backends) live in archive members no consumer references
# directly; force-link the whole vllm archive so every consumer (tests, C API
# shared lib, benchmarks) gets registration.
#
# BACKEND-METAL-MLX W0 item 1: this was `if(UNIX AND NOT APPLE)`, which is TRUE
# on Linux and FALSE on macOS (Apple is UNIX), so on macOS the force-link was
# silently skipped and EVERY static registrar was dropped by ld64 — measured as
# `test_backend` failing 5/7 with `vt: no backend registered for device type 0`
# (.agents/specs/backend-fanout-metal-vulkan-xpu.md § Our baseline). ld64 has no
# `--whole-archive`; the equivalent is `-force_load <archive>`. This unblocks the
# CPU backend on macOS as well as any future Metal backend.
if(MSVC)
  target_link_options(vllm INTERFACE "/WHOLEARCHIVE:$<TARGET_FILE:vllm>")
elseif(APPLE)
  target_link_options(vllm INTERFACE "LINKER:-force_load,$<TARGET_FILE:vllm>")
elseif(UNIX)
  target_link_options(vllm INTERFACE "LINKER:--whole-archive,$<TARGET_FILE:vllm>,--no-whole-archive")
endif()
target_include_directories(vllm
  PUBLIC include ${CMAKE_CURRENT_BINARY_DIR}/include)
# The C ABI impl reaches its internal test-hook header as "capi/engine_handle.h"
# (an internal, non-installed header under src/); src is a PRIVATE search path.
target_include_directories(vllm PRIVATE src)
# third_party/ is a SYSTEM include so vendored headers stay off the -Werror path.
# It carries the header-only deps reached as <dir/header>: <httplib/httplib.h>,
# <nlohmann/json.hpp>, <doctest/doctest.h>, and the vendored google/minja Jinja
# engine <minja/chat-template.hpp> / <minja/minja.hpp> that backs
# src/vllm/entrypoints/chat_template.cpp.
target_include_directories(vllm SYSTEM PUBLIC third_party)
vllm_cpp_set_warnings(vllm)
# SYSTEM covers the FRONT-END diagnostics from those vendored headers, but not
# the ones the OPTIMIZER raises after inlining: those are attributed to the TU
# being compiled, not to the header, so -isystem cannot reach them. GCC 15
# inlines minja::Parser::parseValueExpression (third_party/minja/minja.hpp),
# loses track of the allocation base of a std::vector<std::string> it is
# destroying, and reports -Wfree-nonheap-object against correct vendored code.
# Demote just that diagnostic, just here: the warning stays visible, -Werror
# stays in force for every other diagnostic and every other TU, and the
# vendored source stays unmodified. Ordered after vllm_cpp_set_warnings so the
# target-level -Werror cannot promote it again (the lesson of the MLX header
# exception below). cmake/CompilerWarnings.cmake documents the same GCC
# false-positive class for the sanitizer lanes.
set_source_files_properties(src/vllm/entrypoints/chat_template.cpp PROPERTIES
  COMPILE_OPTIONS
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-error=free-nonheap-object>")
# --- Vendored BLAKE3 (LMCache MODE-1 lm:// token hash, KV-EXTERNAL-CACHE W1) --
# The official BLAKE3 C implementation (third_party/blake3, upstream tag 1.5.5
# / commit 81f772a; dual CC0-1.0 / Apache-2.0), the same algorithm the `blake3`
# PyPI package LMCache imports binds. Built as a SEPARATE static lib with the
# portable backend forced (every SIMD path disabled via BLAKE3_NO_*): the output
# is byte-identical on x86-64 AND aarch64 (BLAKE3 is defined little-endian; the
# portable load32 in blake3_impl.h assembles words byte-wise, endian-independent)
# and this keeps the vendored C sources UNMODIFIED and off the -Werror path.
# PIC so vllm_shared can link it. Provenance: third_party/README.md,
# .agents/porting-inventory.md §9.
enable_language(C)  # blake3 is plain C; the project is otherwise CXX-only
add_library(blake3_vendored STATIC
  third_party/blake3/blake3.c
  third_party/blake3/blake3_dispatch.c
  third_party/blake3/blake3_portable.c)
target_compile_definitions(blake3_vendored PRIVATE
  BLAKE3_NO_AVX512 BLAKE3_NO_AVX2 BLAKE3_NO_SSE41 BLAKE3_NO_SSE2
  BLAKE3_USE_NEON=0)
target_include_directories(blake3_vendored PUBLIC third_party/blake3)
set_target_properties(blake3_vendored PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(vllm PUBLIC blake3_vendored)

# --- Parakeet.cpp dependency (diarization + SAS) -----------------------------
# parakeet.cpp provides the Nemotron-3-Diarization Sortformer encoder, speaker
# head, streaming diarization (AOSC), and speaker-attributed ASR (SAS) merge.
# vllm.cpp's own ParakeetTranscriber handles ASR; this adds the diarization
# side through parakeet.cpp's C-API (parakeet_capi.h).
option(VLLM_CPP_WITH_DIARIZATION "Enable diarization support via parakeet.cpp" ON)
if(VLLM_CPP_WITH_DIARIZATION)
  set(VLLM_CPP_PARAKEET_CPP_DIR "" CACHE PATH
    "Path to a parakeet.cpp source tree. If empty, FetchContent from GitHub.")
  # parakeet.cpp builds its own ggml; we only need libparakeet.a and the
  # headers. Suppress its tests, CLI and server to keep the build lean.
  set(PARAKEET_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(PARAKEET_BUILD_CLI OFF CACHE BOOL "" FORCE)
  set(PARAKEET_BUILD_SERVER OFF CACHE BOOL "" FORCE)
  # Link parakeet.cpp's ggml statically. Its default is shared, which made
  # libvllm.so NEED libggml*.so.0 through a RUNPATH into this build tree, so the
  # library stopped being self-contained. Same save-and-restore as boringssl.
  set(_vllm_saved_shared ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS OFF)
  # parakeet.cpp forces GGML_NATIVE (-march=native) unless the caller sets it.
  # A release library must run on hosts other than the builder; this project
  # selects ISA per file instead (see the -mavx2 sources above).
  if(NOT DEFINED GGML_NATIVE)
    set(GGML_NATIVE OFF CACHE BOOL "ggml: optimize the build for the current system")
  endif()
  if(VLLM_CPP_PARAKEET_CPP_DIR)
    set(parakeet_cpp_SOURCE_DIR "${VLLM_CPP_PARAKEET_CPP_DIR}")
    # Setting the source dir alone defines no `parakeet` target, so the link
    # below would fall back to a bare -lparakeet. Build the local tree.
    add_subdirectory("${parakeet_cpp_SOURCE_DIR}"
      "${CMAKE_BINARY_DIR}/_deps/parakeet_cpp-build" EXCLUDE_FROM_ALL)
  else()
    include(FetchContent)
    # Pinned to the commit the seam in src/vllm/multimodal/diarization.cpp and
    # src/capi/vllm_c.cpp was written and compiled against (ABI v8 on the
    # parakeet.cpp feat/diarization-sas branch). A branch name is not a pin:
    # `main` does not exist there (the default branch is `master`), and master
    # changed the signatures of parakeet_capi_transcribe_and_diarize and
    # parakeet_capi_free_sas_results after this seam was written. Move the pin
    # only together with the seam. GIT_SHALLOW stays off because a shallow
    # clone cannot check out an arbitrary commit hash.
    FetchContent_Declare(parakeet_cpp
      GIT_REPOSITORY https://github.com/mudler/parakeet.cpp.git
      GIT_TAG        394d270fabb1d6125f05c772aa3ca078a574b19d)
    FetchContent_MakeAvailable(parakeet_cpp)
  endif()
  set(BUILD_SHARED_LIBS ${_vllm_saved_shared})
  # parakeet.cpp finds dr_wav.h through ${CMAKE_SOURCE_DIR}/third_party, which
  # is this project's root when it is built as a subproject. Add its own
  # third_party directory so audio_io.cpp compiles here.
  target_include_directories(parakeet PRIVATE
    ${parakeet_cpp_SOURCE_DIR}/third_party)
  target_include_directories(vllm PUBLIC
    ${parakeet_cpp_SOURCE_DIR}/include)
  target_link_libraries(vllm PUBLIC parakeet)
  target_compile_definitions(vllm PUBLIC VLLM_WITH_DIARIZATION=1)
endif()
if(WIN32)
  target_link_libraries(vllm PUBLIC ws2_32)
endif()
# --- Metal backend (BACKEND-METAL-MLX W0 skeleton) --------------------------
# Purely ADDITIVE, exactly like the CUDA block below: a `target_sources` list, a
# PUBLIC compile definition tests can gate on, and two framework links. No
# existing source file, and no line of the CPU/CUDA build, is touched by turning
# it on. The MSL is compiled at RUN TIME from an embedded string
# (src/vt/metal/metal_msl.h), so there is NO offline shader-compilation step and
# no dependency on a `metal` compiler this build could fail to find.
if(VLLM_CPP_METAL)
  target_compile_definitions(vllm PUBLIC VLLM_CPP_METAL)
  target_sources(vllm PRIVATE
    src/vllm/platforms/metal.cpp
    src/vt/metal/metal_context.mm
    src/vt/metal/metal_backend.mm
    src/vt/metal/metal_ops.mm)
  target_link_libraries(vllm PUBLIC "-framework Metal" "-framework Foundation")
  # --- optional MLX GEMM provider (BACKEND-ACCEL-PROVIDER M5) ----------------
  # ONE additive block: one source file, one define, one include dir, one link.
  # Turning it OFF leaves the Metal build byte-for-byte what it was — which is
  # the point of the provider seam. The provider registers ALONGSIDE the native
  # MSL GEMM and wins by priority, deterministically, instead of racing it
  # through static-init order the way the old flat op table would have.
  if(VLLM_CPP_MLX)
    if(NOT MLX_ROOT)
      message(FATAL_ERROR "VLLM_CPP_MLX=ON requires -DMLX_ROOT=<mlx install root> "
        "(the pip wheel's site-packages/mlx directory works).")
    endif()
    find_library(MLX_LIBRARY NAMES mlx PATHS "${MLX_ROOT}/lib" NO_DEFAULT_PATH)
    if(NOT MLX_LIBRARY OR NOT EXISTS "${MLX_ROOT}/include/mlx/array.h")
      message(FATAL_ERROR "VLLM_CPP_MLX=ON but no libmlx + include/mlx/array.h under "
        "MLX_ROOT=${MLX_ROOT}")
    endif()
    include(cmake/MLXDependency.cmake)
    vllm_cpp_import_mlx("${MLX_ROOT}" "${MLX_LIBRARY}")
    target_compile_definitions(vllm PUBLIC VLLM_CPP_MLX)
    target_sources(vllm PRIVATE src/vt/metal/metal_mlx_provider.mm)
    # MLX is a third-party dependency target: its public headers are SYSTEM,
    # while vllm's own sources remain subject to the target-wide -Werror policy.
    target_link_libraries(vllm PUBLIC vllm_cpp::mlx)
    message(STATUS "MLX GEMM provider enabled: ${MLX_LIBRARY}")
  endif()
endif()
# --- MLX provider BIT-ROT GUARD (GATE-METAL-MLX-COMPILE, issue #1765) --------
# `src/vt/metal/metal_mlx_provider.mm` was built by exactly one job when this
# guard landed: `mlx_arm64` in .github/workflows/release.yml, which triggers on
# a `v*` tag or a manual dispatch -- never on a pull request and never on a push
# to `main`. `.github/workflows/ci.yml` had no Apple runner to fall back on. So
# the only build of this file happened AFTER a change landed, and a break presented
# as a blocked release instead of as a red check. Issue #1584 is the case that
# proved it: its edit to `MlxFallback` landed unbuilt and unrun.
#
# The file can be compiled here, and it is the only Metal TU that can: it
# carries ZERO Objective-C constructs. Its two `#import`s are Apple frameworks
# it names nothing from, because `metal_buffers.h` and `metal_context.h` hand it
# every Metal handle as a `void*` on purpose (metal_context.h:22). What it
# actually depends on is the `vt::OpProvider` seam, which is portable C++ -- and
# that is precisely what #1584 changed under it.
#
# Never linked, exactly like `vllm_rocm_platform_syntax_check` above: the object
# is built and discarded, so the provider's static registrar cannot run and
# every other build stays byte-identical. This is a syntax/type gate on OUR
# seam. It is NOT a build of the MLX provider -- the stubs under
# src/vt/metal/stubs are written from this file's call sites and so can never
# disagree with them, which leaves the MLX API to the jobs that build against
# the real wheel: `mlx_arm64` at release time, and `macos-metal-mlx` in ci.yml
# on every push to `main` and every 4-hourly baseline since 2026-08-23 (wave 2
# of this row). src/vt/metal/stubs/README.md states that limit; do not read a
# green here as "the MLX provider builds". This target remains the only Metal
# signal a PULL REQUEST gets, which is why it stays.
#
# Two guards, both load-bearing:
#   * NOT VLLM_CPP_MLX -- on a real MLX build the file belongs to `vllm` as
#     OBJCXX, and the LANGUAGE property below is DIRECTORY-scoped, so setting it
#     there would rewrite how the shipping build compiles it. Off that build,
#     the file is in no other target and the property reaches nothing else.
#   * NOT MSVC -- in MSVC `#import` means "import a type library", so cl.exe
#     hard-errors on those two lines. The windows-2022 jobs must not inherit a
#     red from a bit-rot guard.
if(NOT VLLM_CPP_MLX AND NOT MSVC)
  # CMake maps `.mm` to OBJCXX, which would need enable_language(OBJCXX) and an
  # ObjC++ compiler no Linux runner has. LANGUAGE CXX makes it emit `-x c++`.
  set_source_files_properties(src/vt/metal/metal_mlx_provider.mm
    PROPERTIES LANGUAGE CXX)
  add_library(vllm_metal_mlx_provider_syntax_check OBJECT
    src/vt/metal/metal_mlx_provider.mm)
  # The vt:: seam is REAL here -- that is the entire point. Only the foreign
  # dependencies are stubbed, and they are SYSTEM so their own diagnostics stay
  # out of this TU's -Werror while the provider's code stays subject to it.
  target_include_directories(vllm_metal_mlx_provider_syntax_check PRIVATE
    ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/vt/metal)
  target_include_directories(vllm_metal_mlx_provider_syntax_check SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/src/vt/metal/stubs ${CMAKE_SOURCE_DIR}/third_party)
  target_compile_features(vllm_metal_mlx_provider_syntax_check PRIVATE cxx_std_20)
  vllm_cpp_set_warnings(vllm_metal_mlx_provider_syntax_check)
  # g++ rejects ObjC's `#import` in C++ mode as
  # `error: #import is a deprecated GCC extension [-Werror=deprecated]`.
  # MEASURED to be narrow rather than assumed: with -Wno-deprecated in force, a
  # call to a [[deprecated]] function still fails as -Werror=deprecated-
  # declarations, so this gives up no diagnostic this TU could otherwise draw.
  # Rewriting the two `#import`s to `#include` was rejected -- editing a
  # production file whose real compiler is not on this host, to suit a gate,
  # is the wrong direction.
  target_compile_options(vllm_metal_mlx_provider_syntax_check PRIVATE -Wno-deprecated)
endif()
# --- Vulkan backend (BACKEND-VULKAN W0 skeleton) -----------------------------
# Purely ADDITIVE, exactly like the Metal block above: a `target_sources` list, a
# PUBLIC compile definition tests can gate on, and `${CMAKE_DL_LIBS}` for dlopen.
# No existing source file, and no line of the CPU/CUDA/Metal build, is touched by
# turning it on.
#
# NOTHING is needed at build time: the Khronos headers are vendored under
# third_party/ (single-header-adjacent, Apache-2.0, see third_party/README.md),
# the GLSL is compiled AHEAD OF TIME into the committed
# src/vt/vulkan/vulkan_spirv.h (regenerate with scripts/gen-vulkan-spirv.py), and
# libvulkan.so.1 is resolved with dlopen at run time rather than linked. That is
# a deliberate divergence from llama.cpp, which requires `glslc` on every build
# machine; neither of our boxes has one and neither grants sudo to install one.
if(VLLM_CPP_VULKAN)
  target_compile_definitions(vllm PUBLIC VLLM_CPP_VULKAN)
  target_sources(vllm PRIVATE
    src/vllm/platforms/vulkan.cpp
    src/vt/vulkan/vulkan_loader.cpp
    src/vt/vulkan/vulkan_context.cpp
    src/vt/vulkan/vulkan_backend.cpp
    src/vt/vulkan/vulkan_ops.cpp
    # GENERATED by scripts/gen-vulkan-spirv.py, alongside vulkan_spirv.h. The
    # SPIR-V words are here rather than in the header so that adding shaders costs
    # ONE TU's compile time instead of every TU that includes the table.
    src/vt/vulkan/vulkan_spirv.cpp)
  target_link_libraries(vllm PUBLIC ${CMAKE_DL_LIBS})
endif()
if(NOT WIN32)
  # The persistent cache is also exercised by CPU-only NVFP4 emulation. Its
  # current file-lock implementation is POSIX-only; Windows CUDA is out of W15.
  target_sources(vllm PRIVATE src/vt/cuda/nvfp4_persistent_cache.cpp)
endif()
# --- Tenstorrent backend (BACKEND-TENSTORRENT W0 skeleton) -------------------
# Mostly ADDITIVE like the Metal/Vulkan/ROCm blocks above: a source list, a
# PUBLIC compile definition, and one link library. TTNN::TTNN is TT-NN's own
# exported IMPORTED target and pulls in TT::Metalium/xtensor/nlohmann_json
# transitively per its own package config — no separate find_dependency needed
# here.
#
# tenstorrent_ops.cpp is the ONE exception, isolated as its own OBJECT library
# rather than a plain `vllm` source (same pattern as
# vllm_rocm_platform_syntax_check below). It includes
# ttnn/operations/matmul/matmul.hpp, which transitively pulls in tt-metal's
# tt_stl/reflection.hpp — that header uses nlohmann::json as a COMPLETE type
# without itself including the full <nlohmann/json.hpp> (it relies on some
# other TU in tt-metal's own build having already done so). `vllm`'s own
# vendored nlohmann 3.12.0 (ABI-tagged json_abi_v3_12_0, third_party/, SYSTEM
# PUBLIC on `vllm` a few hundred lines up) is on the include path for every
# `vllm` source, so an angle-bracket `<nlohmann/json.hpp>` reached from
# tt-metal's headers in a `vllm`-target TU resolves to vllm's copy instead of
# tt-metal's own (ABI-tagged json_abi_v3_11_3) — two incompatible ABI-tagged
# versions of the same library visible in one TU, "implicit instantiation of
# undefined template" and "reference to basic_json is ambiguous". Nothing else
# in this backend (or in `vllm` generally) needs matmul.hpp, so giving this one
# file its own include set — TTNN's dirs, no `third_party` — resolves it
# without touching include order for the other ~370 files in the vllm target,
# which must keep resolving vllm's OWN vendored version.
if(VLLM_CPP_TENSTORRENT)
  target_compile_definitions(vllm PUBLIC VLLM_CPP_TENSTORRENT)
  target_sources(vllm PRIVATE
    src/vllm/platforms/tenstorrent.cpp
    src/vt/tenstorrent/tenstorrent_device.cpp
    src/vt/tenstorrent/tenstorrent_backend.cpp)
  add_library(vllm_tenstorrent_ops OBJECT
    src/vt/tenstorrent/tenstorrent_ops.cpp
    src/vt/tenstorrent/tenstorrent_residency.cpp
    src/vt/tenstorrent/tenstorrent_keepquant.cpp
    src/vt/tenstorrent/tenstorrent_paged.cpp
    src/vt/tenstorrent/tenstorrent_gdn.cpp
    src/vt/tenstorrent/tenstorrent_capture.cpp)
  target_include_directories(vllm_tenstorrent_ops PRIVATE
    ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(vllm_tenstorrent_ops PRIVATE TTNN::TTNN)
  target_compile_features(vllm_tenstorrent_ops PRIVATE cxx_std_20)
  vllm_cpp_set_warnings(vllm_tenstorrent_ops)
  target_sources(vllm PRIVATE $<TARGET_OBJECTS:vllm_tenstorrent_ops>)
  target_link_libraries(vllm PUBLIC TTNN::TTNN)
endif()
# --- ROCm/HIP backend (BACKEND-ROCM W0 skeleton) ------------------------------
# Purely ADDITIVE like the Metal and Vulkan blocks above: a source list, a PUBLIC
# compile definition the tests gate on, and one link library. No existing source
# file and no line of the CPU/CUDA/Metal/Vulkan build changes when it is on.
#
# **NEVER COMPILED.** These four TUs have no build report from any machine. The
# arch parse they depend on (include/vt/rocm/rocm_arch.h) is deliberately NOT
# here: it is a HIP-free header compiled and unit-tested in every build,
# including CPU-only CI. See tests/vt/test_rocm_arch.cpp.
if(VLLM_CPP_HIP)
  target_compile_definitions(vllm PUBLIC VLLM_CPP_HIP)
  target_sources(vllm PRIVATE
    src/vllm/platforms/rocm.cpp
    src/vt/rocm/rocm_backend.hip
    src/vt/rocm/rocm_rmsnorm.hip
    src/vt/rocm/rocm_compiled_gemma.hip
    src/vt/rocm/rocm_residual_rmsnorm.hip
    src/vt/rocm/rocm_embedding.hip
    src/vt/rocm/rocm_embedding_quant.hip
    src/vt/rocm/rocm_dense_basic.hip
    src/vt/rocm/rocm_attn_gate_split.hip
    src/vt/rocm/rocm_matmul_hipblaslt.hip
    src/vt/rocm/rocm_paged_attn.hip
    src/vt/rocm/rocm_rdna3_attn_decode.hip
    src/vt/rocm/rocm_gemma4_experts.hip
    src/vt/rocm/rocm_gemma4_fused_experts.hip
    src/vt/rocm/rocm_gemma4_expert_geglu.hip
    src/vt/rocm/rocm_fp8_channel_gemv.hip
    src/vt/rocm/rocm_moe_router.hip
    src/vt/rocm/rocm_moe_chain.hip
    src/vt/rocm/rocm_moe_grouped_bf16.hip
    src/vt/rocm/rocm_grouped_gemm.hip
    src/vt/rocm/rocm_moe_gate_up_swiglu.hip
    src/vt/rocm/rocm_exl3.hip
    src/vt/rocm/rocm_sample.hip
    src/vt/rocm/rocm_gdn_state.hip
    src/vt/rocm/rocm_gdn_conv.hip
    src/vt/rocm/rocm_gdn_postconv.hip
    src/vt/rocm/rocm_gdn_scan.hip
    src/vt/rocm/rocm_kda_scan.hip
    src/vt/rocm/rocm_glm5_next_kpool.hip
    src/vt/rocm/rocm_gdn_fused.hip
    src/vt/rocm/rocm_mla_fused_norm_rope.hip
    src/vt/rocm/rocm_mla_ops.hip
    src/vt/rocm/rocm_mla_attn.hip
    src/vt/rocm/rocm_skinny_gemm.hip
    src/vt/rocm/rocm_ops.hip
    src/vt/rocm/rocm_qwen4_exp.hip
    src/vt/rocm/rocm_qwen4_exp_ple.hip
    src/vt/rocm/rocm_qwen4_exp_qsa.hip
    src/vt/rocm/rocm_dsa_indexer.hip
    src/vt/rocm/rocm_rms_norm_group.hip
    src/vt/rocm/rocm_quant_dot.hip
    src/vt/rocm/rocm_deepseek_v4.hip)
  if(VLLM_CPP_HIP_ARCHITECTURES)
    set_source_files_properties(
      src/vt/rocm/rocm_backend.hip
      src/vt/rocm/rocm_rmsnorm.hip
      src/vt/rocm/rocm_compiled_gemma.hip
      src/vt/rocm/rocm_residual_rmsnorm.hip
      src/vt/rocm/rocm_embedding.hip
      src/vt/rocm/rocm_embedding_quant.hip
      src/vt/rocm/rocm_dense_basic.hip
      src/vt/rocm/rocm_attn_gate_split.hip
      src/vt/rocm/rocm_matmul_hipblaslt.hip
      src/vt/rocm/rocm_paged_attn.hip
      src/vt/rocm/rocm_rdna3_attn_decode.hip
      src/vt/rocm/rocm_gemma4_experts.hip
      src/vt/rocm/rocm_gemma4_fused_experts.hip
      src/vt/rocm/rocm_gemma4_expert_geglu.hip
      src/vt/rocm/rocm_fp8_channel_gemv.hip
      src/vt/rocm/rocm_moe_router.hip
      src/vt/rocm/rocm_moe_chain.hip
      src/vt/rocm/rocm_moe_grouped_bf16.hip
      src/vt/rocm/rocm_grouped_gemm.hip
      src/vt/rocm/rocm_moe_gate_up_swiglu.hip
      src/vt/rocm/rocm_exl3.hip
      src/vt/rocm/rocm_sample.hip
      src/vt/rocm/rocm_gdn_state.hip
      src/vt/rocm/rocm_gdn_conv.hip
      src/vt/rocm/rocm_gdn_postconv.hip
      src/vt/rocm/rocm_gdn_scan.hip
      src/vt/rocm/rocm_kda_scan.hip
      src/vt/rocm/rocm_glm5_next_kpool.hip
      src/vt/rocm/rocm_gdn_fused.hip
      src/vt/rocm/rocm_mla_fused_norm_rope.hip
      src/vt/rocm/rocm_mla_ops.hip
      src/vt/rocm/rocm_mla_attn.hip
      src/vt/rocm/rocm_skinny_gemm.hip
      src/vt/rocm/rocm_ops.hip
      src/vt/rocm/rocm_qwen4_exp.hip
      src/vt/rocm/rocm_qwen4_exp_ple.hip
      src/vt/rocm/rocm_qwen4_exp_qsa.hip
      src/vt/rocm/rocm_dsa_indexer.hip
      src/vt/rocm/rocm_rms_norm_group.hip
      src/vt/rocm/rocm_quant_dot.hip
      src/vt/rocm/rocm_deepseek_v4.hip
      PROPERTIES HIP_ARCHITECTURES "${VLLM_CPP_HIP_ARCHITECTURES}")
  endif()
  # Prefer the absolute path inside ${ROCM_PATH}/lib, fall back to the bare name,
  # which is upstream's own resolution order (vllm/CMakeLists.txt:335-341).
  find_library(VLLM_CPP_AMDHIP64_LIB NAMES amdhip64 libamdhip64.so
    PATHS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64)
  if(VLLM_CPP_AMDHIP64_LIB)
    target_link_libraries(vllm PUBLIC ${VLLM_CPP_AMDHIP64_LIB})
  else()
    target_link_libraries(vllm PUBLIC amdhip64)
  endif()
  # hipBLAS + hipBLASLt — dense GEMM (ROCm math libraries).
  find_path(VLLM_CPP_HIPBLAS_INCLUDE hipblas/hipblas.h
    PATHS ${ROCM_PATH}/include /opt/rocm/include)
  find_library(VLLM_CPP_HIPBLAS_LIB NAMES hipblas
    PATHS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64 /opt/rocm/lib /opt/rocm/lib64)
  find_path(VLLM_CPP_HIPBLASLT_INCLUDE hipblaslt/hipblaslt.h
    PATHS ${ROCM_PATH}/include /opt/rocm/include)
  find_library(VLLM_CPP_HIPBLASLT_LIB NAMES hipblaslt
    PATHS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64 /opt/rocm/lib /opt/rocm/lib64)
  if(VLLM_CPP_HIPBLAS_INCLUDE AND VLLM_CPP_HIPBLAS_LIB)
    target_include_directories(vllm PUBLIC ${VLLM_CPP_HIPBLAS_INCLUDE})
    target_link_libraries(vllm PUBLIC ${VLLM_CPP_HIPBLAS_LIB})
    message(STATUS "ROCm hipBLAS: ${VLLM_CPP_HIPBLAS_LIB}")
  else()
    message(FATAL_ERROR "VLLM_CPP_HIP=ON requires hipBLAS (libhipblas + hipblas/hipblas.h).")
  endif()
  if(VLLM_CPP_HIPBLASLT_INCLUDE AND VLLM_CPP_HIPBLASLT_LIB)
    target_include_directories(vllm PUBLIC ${VLLM_CPP_HIPBLASLT_INCLUDE})
    target_link_libraries(vllm PUBLIC ${VLLM_CPP_HIPBLASLT_LIB})
    target_compile_definitions(vllm PUBLIC VLLM_CPP_HAS_HIPBLASLT)
    message(STATUS "ROCm hipBLASLt: ${VLLM_CPP_HIPBLASLT_LIB}")
  else()
    message(STATUS "ROCm hipBLASLt: not found (hipBLASLt matmul path disabled)")
  endif()
  # clang-17 / ROCm 5.7: __clang_hip_runtime_wrapper.h defines bfloat16 conversion
  # functions as non-inline in every TU, causing multiple-definition link errors.
  # PUBLIC (not PRIVATE) so the flag propagates to vllm_shared and every other
  # target that links the vllm archive — vllm itself is STATIC, so a PRIVATE
  # link option has no effect on it. The flag is harmless when there are no
  # duplicate definitions.
  target_link_options(vllm PUBLIC "LINKER:--allow-multiple-definition")
else()
  # BIT-ROT GUARD (BACKEND-ROCM W0). The ROCm platform leg is deliberately plain
  # C++ with no HIP header, which means it CAN be compiled on a machine with no
  # ROCm — and if we do not, nothing on any maintainer machine ever compiles it,
  # so the next change to the `Platform` interface breaks it silently and the
  # breakage is discovered by a contributor mid-bring-up. Compiling it as an
  # OBJECT library that is never linked keeps it honest at the cost of one TU.
  #
  # Not linked ANYWHERE on purpose: the object is built and discarded, so the
  # platform's static registrar cannot run and a non-HIP build is byte-identical
  # to what it was. This is a syntax/type gate, not a build of the backend — the
  # HIP TUs still have no build report from anyone.
  # The HIP-gated TEST is in here for the same reason: it is the first thing a
  # contributor runs, it is also plain C++, and a stale assertion in it would
  # read as a backend bug on their machine rather than as our drift.
  add_library(vllm_rocm_platform_syntax_check OBJECT
    src/vllm/platforms/rocm.cpp
    tests/vt/test_rocm_backend.cpp)
  target_include_directories(vllm_rocm_platform_syntax_check PRIVATE
    ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
  target_include_directories(vllm_rocm_platform_syntax_check SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party)
  target_compile_features(vllm_rocm_platform_syntax_check PRIVATE cxx_std_20)
  vllm_cpp_set_warnings(vllm_rocm_platform_syntax_check)
endif()
if(VLLM_CPP_CUDA)
  target_compile_definitions(vllm PUBLIC VLLM_CPP_CUDA)
  if(VLLM_CPP_BENCH_PROFILE_CONTROL)
    target_compile_definitions(vllm PUBLIC VT_BENCH_PROFILE_CONTROL=1)
  endif()
  target_sources(vllm PRIVATE
    src/vllm/platforms/cuda.cpp
    src/vt/cuda/cuda_arch_tactics.cu
    src/vt/cuda/cuda_backend.cu
    src/vt/cuda/nccl_communicator.cu
    src/vt/cuda/cuda_dropin.cu
    src/vt/cuda/cuda_matmul.cu
    src/vt/cuda/cuda_quant_dot.cu
    # UNCONDITIONAL ON PURPOSE (issue #960). The static per-tensor fp8 activation
    # quant has no cutlass dependency; it lived in the cutlass-fp8 TU below and so
    # went unregistered for kCUDA on every arch outside VT_CUTLASS_FP8_ARCHS,
    # where the resolver then ran the portable CPU tier over device pointers and
    # segfaulted. scripts/check-cuda-op-arch-gate.py fails if it leaves this list.
    src/vt/cuda/cuda_quant_fp8.cu
    src/vt/cuda/cuda_matmul_nvfp4.cu
    src/vt/cuda/cuda_ops.cu
    src/vt/cuda/cuda_gdn.cu
    src/vt/cuda/cuda_moe.cu
    src/vt/cuda/cuda_glue.cu
    src/vt/cuda/cuda_layernorm.cu
    src/vt/cuda/cuda_cache.cu
    src/vt/cuda/cuda_mla_attn.cu
    src/vt/cuda/cuda_dsa_indexer.cu
    src/vt/cuda/cuda_mla_prefill.cu
    src/vt/cuda/cuda_paged_attn.cu
    src/vt/cuda/cuda_combine_tokens.cu
    src/vt/cuda/cuda_sample.cu
    src/vt/cuda/cuda_deepseek_v4.cu
    src/vt/cuda/cuda_glm5_next.cu
    src/vt/cuda/cuda_exl3.cu
    src/vt/cuda/cuda_laguna.cu
    src/vt/cuda/cuda_minimax_h3.cu
    src/vt/cuda/cuda_ltx2.cu
    src/vt/cuda/cuda_ltx2_vae.cu
    # BigVGAN / DAC vocoder 1-D convolutions (#672) — the first transposed 1-D
    # convolution in the tree on any device.
    src/vt/cuda/cuda_conv1d_general.cu
    # General 3-D convolution (LTX25-DEVICE-RESIDENCY W5, #1007) — the device arm
    # the LTX-2.5 video VAE decode never had.
    src/vt/cuda/cuda_conv3d.cu
    src/vt/cuda/cuda_attention_cross.cu
    # Qwen4-Exp (Qwen3.8-Flash-Next) device arms, row MODEL-MM-QWEN4-EXP W6-CUDA:
    # the first CUDA kernels this architecture has ever had. Split to mirror the
    # cpu_qwen4_exp*.cpp units listed above. Unconditional and at
    # if(VLLM_CPP_CUDA) depth one, which is what check-cuda-op-arch-gate.py
    # requires of any TU that registers an op.
    src/vt/cuda/cuda_qwen4_exp.cu
    src/vt/cuda/cuda_qwen4_exp_ple.cu
    # W6-CUDA-B: the two QSA arms and the ungated grouped RMS norm. The norm is a
    # SHARED `vt::` seam and not a qwen4_exp op, so it gets its own TU rather
    # than joining cuda_layernorm.cu -- that file is a surface several rows
    # write, and AGENTS.md "Records" calls a shared file a lock. Both are
    # unconditional and at if(VLLM_CPP_CUDA) depth one, which is what
    # check-cuda-op-arch-gate.py requires of any TU that registers an op.
    src/vt/cuda/cuda_qwen4_exp_qsa.cu
    src/vt/cuda/cuda_rms_norm_group.cu)
  find_package(CUDAToolkit REQUIRED)
  # cublasLt is linked now so the Task 4 matmul lands without a build change.
  target_link_libraries(vllm PUBLIC CUDA::cudart CUDA::cublasLt)
  # NCCL tensor-parallel transport (BACKEND-DISTRIBUTED-COMM W2, DERIVE-AND-SHIP).
  # OFF by default so the standing single-GPU CUDA gate is untouched: without it
  # src/vt/cuda/nccl_communicator.cu compiles to a stub (no kCUDA collective
  # provider). Turn ON on a >=2-GPU host to build-verify + run the real
  # ncclAllReduce/Send/Recv path (pynccl.py mirror). AUTO-detect could be added,
  # but an explicit opt-in keeps the default CUDA build reproducible.
  option(VLLM_CPP_NCCL "Build the NCCL tensor-parallel transport (needs nccl.h + libnccl)" OFF)
  if(VLLM_CPP_NCCL)
    find_library(NCCL_LIBRARY NAMES nccl REQUIRED)
    find_path(NCCL_INCLUDE_DIR NAMES nccl.h REQUIRED)
    target_compile_definitions(vllm PRIVATE VT_NCCL=1)
    target_include_directories(vllm PRIVATE ${NCCL_INCLUDE_DIR})
    target_link_libraries(vllm PUBLIC ${NCCL_LIBRARY})
    message(STATUS "NCCL TP transport: ON (${NCCL_LIBRARY})")
  endif()
  # The cutlass sm120a NVFP4 GEMM TU is isolated: its own heavy cutlass include
  # tree + relaxed-constexpr/extended-lambda, so it never touches the general
  # build. Gated by VT_CUTLASS_NVFP4 so the host op path no-ops if not compiled.
  if(VLLM_CPP_CUTLASS)
    # The 32 NVFP4 tactics are split four instantiations per tile TU. A single
    # monolithic nvcc process for all swap/scheduler variants exceeds practical
    # compiler memory on GB10; the dispatcher and immutable W1 fallback remain
    # separate so VT_FP4_FULL_TACTICS=0 is an exact same-binary comparison.
    set(_NVFP4_FLASHINFER_TACTIC_SOURCES
      src/vt/cuda/cuda_nvfp4_tactics_128x32x128.cu
      src/vt/cuda/cuda_nvfp4_tactics_128x32x256.cu
      src/vt/cuda/cuda_nvfp4_tactics_128x64x128.cu
      src/vt/cuda/cuda_nvfp4_tactics_128x64x256.cu
      src/vt/cuda/cuda_nvfp4_tactics_128x128x128.cu
      src/vt/cuda/cuda_nvfp4_tactics_128x128x256.cu
      src/vt/cuda/cuda_nvfp4_tactics_256x128x128.cu
      src/vt/cuda/cuda_nvfp4_tactics_128x256x128.cu)
    set(_NVFP4_CUTLASS_SOURCES
      src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu
      src/vt/cuda/cuda_nvfp4_tactics_w1.cu
      ${_NVFP4_FLASHINFER_TACTIC_SOURCES})
    # Persistent tactic caches must become incompatible whenever executable
    # tactic source/ABI changes, including a dirty/source-archive build with no
    # meaningful git revision. Hash the exact compile inputs deterministically.
    set(_NVFP4_CACHE_ABI_SOURCES
      ${_NVFP4_CUTLASS_SOURCES}
      src/vt/cuda/nvfp4_cutlass_tactics.h
      src/vt/cuda/nvfp4_plan_cache.h
      src/vt/cuda/nvfp4_tactic_ids.h)
    set(_NVFP4_CACHE_BUILD_INPUT
      "cutlass=${_CUTLASS_MAJOR}.${_CUTLASS_MINOR}.${_CUTLASS_PATCH};")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
      ${_NVFP4_CACHE_ABI_SOURCES})
    foreach(_NVFP4_CACHE_ABI_SOURCE IN LISTS _NVFP4_CACHE_ABI_SOURCES)
      file(SHA256
        "${CMAKE_CURRENT_SOURCE_DIR}/${_NVFP4_CACHE_ABI_SOURCE}"
        _NVFP4_CACHE_ABI_SHA)
      string(APPEND _NVFP4_CACHE_BUILD_INPUT
        "${_NVFP4_CACHE_ABI_SOURCE}=${_NVFP4_CACHE_ABI_SHA};")
    endforeach()
    string(SHA256 _NVFP4_CACHE_BUILD_ID "${_NVFP4_CACHE_BUILD_INPUT}")
    # cutlass-fp8 is its own FEATURE-TABLE row: its ArchTag=Sm120 scaled-mm has a
    # different upstream arch set from the NVFP4 block-scaled collective (vLLM
    # keeps them as separate SCALED_MM_ARCHS / FP4_SM120_ARCHS intersections), so
    # it is resolved separately here rather than riding on the NVFP4 gate.
    # cuda_matmul_fp8_block_cutlass.cu (#1189 M5) rides the SAME cell: it is the
    # BLOCK-scaled (fine-grained 128x128) sibling of the per-tensor GEMM above,
    # built from the same ArchTag=Sm120 family -- upstream keeps both in
    # csrc/.../w8a8/cutlass/c3x behind one SCALED_MM_ARCHS intersection. It is a
    # separate TU rather than a second kernel in the same file because the two
    # share no template (dense layout tags vs cute::tuple<Layout, LayoutSF>), so
    # one file would only serialise two heavy cutlass compiles.
    set(_FP8_CUTLASS_SOURCES)
    if(VT_CUTLASS_FP8_ARCHS)
      set(_FP8_CUTLASS_SOURCES
        src/vt/cuda/cuda_matmul_fp8_cutlass.cu
        src/vt/cuda/cuda_matmul_fp8_block_cutlass.cu)
    endif()
    target_sources(vllm PRIVATE
      ${_NVFP4_CUTLASS_SOURCES}
      ${_FP8_CUTLASS_SOURCES})
    target_compile_definitions(vllm PUBLIC VT_CUTLASS_NVFP4=1)
    if(VT_CUTLASS_FP8_ARCHS)
      target_compile_definitions(vllm PUBLIC VT_CUTLASS_FP8=1)
    endif()
    target_compile_definitions(vllm PRIVATE
      VT_CUTLASS_VERSION_STRING="${_CUTLASS_MAJOR}.${_CUTLASS_MINOR}.${_CUTLASS_PATCH}"
      VT_NVFP4_CACHE_BUILD_ID="${_NVFP4_CACHE_BUILD_ID}")
    # cutlass headers reach via -isystem so their internal deprecation warnings
    # are exempt from the global -Werror=all-warnings (the general build stays
    # strict). --expt-relaxed-constexpr/--expt-extended-lambda: cutlass needs them.
    set_source_files_properties(
      ${_NVFP4_CUTLASS_SOURCES}
      ${_FP8_CUTLASS_SOURCES} PROPERTIES
      COMPILE_OPTIONS
        "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda;-isystem;${VLLM_CPP_CUTLASS_DIR}/include;-isystem;${VLLM_CPP_CUTLASS_DIR}/tools/util/include>")
    # Keep the split 32-tactic family on FlashInfer's semantic CUDA compile
    # envelope. These definitions are source-scoped: the general CUDA backend,
    # immutable W1 fallback and FP8 adapter retain the project defaults.
    set_property(SOURCE ${_NVFP4_FLASHINFER_TACTIC_SOURCES} APPEND PROPERTY
      COMPILE_DEFINITIONS
        CUTLASS_ENABLE_GDC_FOR_SM100=1
        FLASHINFER_ENABLE_FP8_E8M0
        FLASHINFER_ENABLE_FP4_E2M1
        FLASHINFER_ENABLE_F16
        FLASHINFER_ENABLE_BF16
        FLASHINFER_ENABLE_FP8_E4M3
        FLASHINFER_ENABLE_FP8_E5M2
        ENABLE_BF16
        ENABLE_FP4)
    set_property(SOURCE ${_NVFP4_FLASHINFER_TACTIC_SOURCES} APPEND PROPERTY
      COMPILE_OPTIONS
        "$<$<COMPILE_LANGUAGE:CUDA>:-static-global-template-stub=false;-use_fast_math>")
  endif()
  # Datacenter-Blackwell sm_100a NVFP4 tcgen05 GEMM BUILD-VERIFY TU
  # (BACKEND-CUDA-SM100, ROAD-V1-D1-CUDA). Compiled ONLY for a single-arch 100a
  # target — VT_CUTLASS_NVFP4_SM100 is set (above) exclusively when the
  # `cutlass-nvfp4-sm100` FEATURE-TABLE cell resolves for the requested arch, which
  # is 100a alone. It is SEPARATE from VLLM_CPP_CUTLASS (the consumer sm_12x tactic
  # sweep): the datacenter body uses ArchTag=Sm100 + KernelScheduleAuto (tcgen05),
  # a different collective/tile/cluster than the sm_120 body, so the two never
  # share a compile (widening the sm_12x gate to 100a would ptxas-reject the sm_120
  # PTX for compute_100a). On the gate arch sm_121a this flag is OFF, so the TU is
  # not added to the build at all → byte-zero impact on GB10.
  #   * -isystem cutlass so its headers are off the global -Werror=all-warnings
  #     path (mirrors the sm_12x cutlass TUs above);
  #   * --expt-relaxed-constexpr/--expt-extended-lambda: cutlass needs them;
  #   * -diag-suppress=20012: benign cutlass sm100_static_tile_scheduler.hpp
  #     "__host__/__device__ annotation ignored on defaulted function" front-end
  #     diagnostic (mirrors the marlin -diag-suppress=20280 precedent below).
  # DERIVED+BUILD-VERIFIED (testing-welcome): a green compile + cuobjdump sm_100a
  # SASS is NOT execution evidence — no B200/sm_100 board ran this.
  if(VLLM_CPP_CUTLASS_NVFP4_SM100)
    target_sources(vllm PRIVATE src/vt/cuda/cuda_matmul_nvfp4_sm100.cu)
    target_compile_definitions(vllm PUBLIC VT_CUTLASS_NVFP4_SM100=1)
    set_source_files_properties(
      src/vt/cuda/cuda_matmul_nvfp4_sm100.cu PROPERTIES
      COMPILE_OPTIONS
        "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda;-diag-suppress=20012;-isystem;${VLLM_CPP_CUTLASS_DIR}/include;-isystem;${VLLM_CPP_CUTLASS_DIR}/tools/util/include>")
  endif()
  # Hopper sm_90a CUTLASS C3x FP8 scaled-mm wgmma/TMA GEMM BUILD-VERIFY TU
  # (BACKEND-CUDA-SM090, ROAD-V1-D1-CUDA, datacenter fast-path §9 DC2). Compiled
  # ONLY for a single-arch 90a target — VLLM_CPP_SCALEDMM_C3X_SM90 is set (above)
  # exclusively when the `scaledmm-c3x-sm90` FEATURE-TABLE cell resolves for the
  # requested arch, which is 90a alone. It is SEPARATE from VLLM_CPP_CUTLASS (the
  # consumer sm_12x scaled-mm): the Hopper body uses ArchTag=Sm90 +
  # KernelTmaWarpSpecialized*FP8FastAccum (wgmma/TMA), a different collective than
  # the sm_120 body, so the two never share a compile (widening the sm_12x gate to
  # 90a would ptxas-reject the sm_120 PTX for compute_90a). On the gate arch
  # sm_121a this flag is OFF, so the TU is not added to the build at all →
  # byte-zero impact on GB10.
  #   * -isystem cutlass so its headers are off the global -Werror=all-warnings
  #     path (mirrors the sm_12x/sm_100a cutlass TUs above);
  #   * --expt-relaxed-constexpr/--expt-extended-lambda: cutlass needs them.
  # DERIVED+BUILD-VERIFIED (testing-welcome): a green compile + cuobjdump sm_90a
  # SASS is NOT execution evidence — no H100/H200/sm_90 board ran this.
  if(VLLM_CPP_SCALEDMM_C3X_SM90)
    target_sources(vllm PRIVATE src/vt/cuda/cuda_scaled_mm_c3x_sm90.cu)
    target_compile_definitions(vllm PUBLIC VT_SCALEDMM_C3X_SM90=1)
    set_source_files_properties(
      src/vt/cuda/cuda_scaled_mm_c3x_sm90.cu PROPERTIES
      COMPILE_OPTIONS
        "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda;-isystem;${VLLM_CPP_CUTLASS_DIR}/include;-isystem;${VLLM_CPP_CUTLASS_DIR}/tools/util/include>")
  endif()
  # Datacenter-Blackwell sm_100a CUTLASS C3x FP8 scaled-mm tcgen05 GEMM BUILD-VERIFY
  # TU (BACKEND-CUDA-SM100, ROAD-V1-D1-CUDA, datacenter fast-path §9 DC3). Compiled
  # ONLY for a single-arch 100a target — VLLM_CPP_SCALEDMM_C3X_SM100 is set (above)
  # exclusively when the `scaledmm-c3x-sm100` FEATURE-TABLE cell resolves for the
  # requested arch, which is 100a alone. It is SEPARATE from VLLM_CPP_CUTLASS (the
  # consumer sm_12x scaled-mm) AND from VLLM_CPP_SCALEDMM_C3X_SM90 (the Hopper leg):
  # the datacenter body uses ArchTag=Sm100 + KernelScheduleAuto (tcgen05), a
  # different collective than the sm_120 body and the Sm90 wgmma body, so no two
  # share a compile (widening the sm_12x gate to 100a would ptxas-reject the sm_120
  # PTX for compute_100a). On the gate arch sm_121a this flag is OFF, so the TU is
  # not added to the build at all → byte-zero impact on GB10.
  #   * -isystem cutlass so its headers are off the global -Werror=all-warnings
  #     path (mirrors the sm_12x/sm_90a/sm_100a-NVFP4 cutlass TUs above);
  #   * --expt-relaxed-constexpr/--expt-extended-lambda: cutlass needs them;
  #   * -diag-suppress=20012: benign cutlass sm100_static_tile_scheduler.hpp
  #     "__host__/__device__ annotation ignored on defaulted function" front-end
  #     diagnostic (same tcgen05 header the sm_100a NVFP4 TU suppresses).
  # DERIVED+BUILD-VERIFIED (testing-welcome): a green compile + cuobjdump sm_100a
  # SASS is NOT execution evidence — no B200/sm_100 board ran this.
  if(VLLM_CPP_SCALEDMM_C3X_SM100)
    target_sources(vllm PRIVATE src/vt/cuda/cuda_scaled_mm_c3x_sm100.cu)
    target_compile_definitions(vllm PUBLIC VT_SCALEDMM_C3X_SM100=1)
    set_source_files_properties(
      src/vt/cuda/cuda_scaled_mm_c3x_sm100.cu PROPERTIES
      COMPILE_OPTIONS
        "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda;-diag-suppress=20012;-isystem;${VLLM_CPP_CUTLASS_DIR}/include;-isystem;${VLLM_CPP_CUTLASS_DIR}/tools/util/include>")
  endif()
  # Marlin NVFP4 MoE: two vendored TUs (the marlin_mm dispatcher + the generated
  # bf16 NVFP4 kernel instantiations) + the vt::Tensor launcher. All three need
  #  * -static-global-template-stub=false so the __global__ Marlin<> template
  #    instantiations get external linkage across the two vendored TUs (nvcc 13
  #    defaults to static stubs -> undefined refs at link; verified on GB10);
  #  * -isystem the vendored root so kernel-side warnings (e.g. #20280) are
  #    exempt from the global -Werror=all-warnings; the general build stays strict;
  #  * --expt-relaxed-constexpr (scalar_type.hpp constexpr).
  if(VLLM_CPP_MARLIN)
    set(_MARLIN_ROOT "${CMAKE_SOURCE_DIR}/src/vt/cuda/marlin")
    set(_MARLIN_SRCS
      src/vt/cuda/marlin/libtorch_stable/moe/marlin_moe_wna16/marlin_mm_moe.cu
      src/vt/cuda/marlin/libtorch_stable/moe/marlin_moe_wna16/sm80_kernel_bfloat16_fe2m1f_bfloat16.cu
      src/vt/cuda/cuda_moe_marlin.cu
      src/vt/cuda/cuda_marlin_repack.cu
      # DENSE marlin (row KERNEL-MARLIN-DENSE-PORT): the dense marlin_mm dispatcher
      # (`namespace marlin`) + its bf16 NVFP4/MXFP4 kernel instantiations (SAME 12-param
      # Marlin<> template as the MoE TUs, distinct namespace + direct-A kernel body) +
      # the vt::Tensor launcher. Same nvcc options as the MoE TUs below.
      src/vt/cuda/marlin/libtorch_stable/quantization/marlin/marlin_mm_dense.cu
      src/vt/cuda/marlin/libtorch_stable/quantization/marlin/sm80_kernel_bfloat16_fe2m1f_bfloat16.cu
      src/vt/cuda/cuda_marlin_dense.cu)
    target_sources(vllm PRIVATE ${_MARLIN_SRCS})
    target_compile_definitions(vllm PUBLIC VT_MARLIN_NVFP4=1)
    set_source_files_properties(${_MARLIN_SRCS} PROPERTIES
      COMPILE_OPTIONS
        "$<$<COMPILE_LANGUAGE:CUDA>:-static-global-template-stub=false;--expt-relaxed-constexpr;-diag-suppress=20280;-isystem;${_MARLIN_ROOT}>")
  endif()

  # ── SANCTIONED Triton AOT fast-path: GDN delta_h (VLLM_CPP_TRITON, default ON
  # for a CUDA build) ─────────────────────────────────────────────────────────
  # The FLA chunk_delta_h state-recurrence kernel as an embedded cubin in libvllm
  # (VENDORED per-arch — the build needs no Python; VLLM_CPP_TRITON_REGEN=ON is
  # the maintainer regen); runtime dispatch (behind VT_GDN_DELTAH_TRITON) is in
  # cuda_gdn.cu with the hand-C++ GdnChunkDeltaHRegRingKernel preserved as the
  # default fallback. OFF => none of this runs and the build is byte-identical.
  # Two specializations, one per gate-model GDN shape (both K=V=128, Hg=16, BT=64;
  # only H=num_value_heads differs): 27B H=48, 35B H=32. Signature order matches
  # triton_kernels/chunk_delta_h.py arg list; pinned flags USE_G=1,USE_GK=0,
  # USE_INITIAL_STATE=1,STORE_FINAL_STATE=1,SAVE_NEW_VALUE=1,IS_VARLEN=1,USE_EXP2=0.
  if(VLLM_CPP_TRITON)
    enable_language(C)  # the AOT launchers are plain C (CUDA driver API)
    include(cmake/TritonAOTKernels.cmake)
    vllm_triton_aot_declare_all()
    vllm_triton_aot_expected_lines(_TRITON_AOT_CONTRACT_LINES)
    set_property(GLOBAL PROPERTY VLLM_TRITON_AOT_CONTRACT_LINES
      "${_TRITON_AOT_CONTRACT_LINES}")
    # Launch-config pins, PER KERNEL and PER GDN SHAPE (H=48 27B, H=32 35B),
    # set to the winning config of FLA's OWN @triton.autotune run at the exact
    # engine shapes on GB10 (tools: ~/work/finisher_fla_autotune.py; see
    # .agents/parity-ledger.md 2026-07-10 for the measured best_configs).
    # Overridable cache vars to re-sweep without editing this file.
    # gridX = cdiv(V=128, BV) baked per spec where the kernel is BV-tiled.
    foreach(_H 48 32)
      set(VT_GDN_DELTAH_BV_${_H}     64 CACHE STRING "delta_h H=${_H} Triton BV (32 or 64)")
      set(VT_GDN_DELTAH_WARPS_${_H}  4  CACHE STRING "delta_h H=${_H} Triton num_warps")
      set(VT_GDN_DELTAH_STAGES_${_H} 3  CACHE STRING "delta_h H=${_H} Triton num_stages")
    endforeach()
    set(_DELTAH_KPY "${CMAKE_SOURCE_DIR}/triton_kernels/chunk_delta_h.py")
    set(_DELTAH_KFN chunk_gated_delta_rule_fwd_kernel_h_blockdim64)
    # Signature (26 args): k v w v_new g gk h h0 ht cu_seqlens chunk_offsets T NH
    #   H Hg K V BT BV USE_G USE_GK USE_INITIAL_STATE STORE_FINAL_STATE
    #   SAVE_NEW_VALUE IS_VARLEN USE_EXP2. dtypes: k/v/w/v_new/h bf16, g/h0/ht fp32.
    set(_DELTAH_SIG_HEAD "*bf16:16, *bf16:16, *bf16:16, *bf16:16, *fp32:16, *fp32")
    foreach(_H 48 32)
      set(_DELTAH_SIG_TAIL "*bf16:16, *fp32:16, *fp32:16, *i32:16, *i32:16, i32, i32, ${_H}, 16, 128, 128, 64, ${VT_GDN_DELTAH_BV_${_H}}, 1, 0, 1, 1, 1, 1, 0")
      math(EXPR _DELTAH_GX "(128 + ${VT_GDN_DELTAH_BV_${_H}} - 1) / ${VT_GDN_DELTAH_BV_${_H}}")
      add_triton_kernel(VT_GDN_DELTAH_${_H}_SRCS
        "${_DELTAH_KPY}" ${_DELTAH_KFN} gdn_deltah_h${_H}
        "${_DELTAH_SIG_HEAD}, ${_DELTAH_SIG_TAIL}" "${_DELTAH_GX},NH,1"
        ${VT_GDN_DELTAH_WARPS_${_H}} ${VT_GDN_DELTAH_STAGES_${_H}})
      target_sources(vllm PRIVATE ${VT_GDN_DELTAH_${_H}_SRCS})
      target_include_directories(vllm PRIVATE ${VT_GDN_DELTAH_${_H}_SRCS_INCLUDE_DIR})
    endforeach()

    # ── SANCTIONED Triton AOT: GDN chunk_o (the output kernel) ──────────────────
    # FLA chunk_o.py chunk_fwd_kernel_o -> triton_kernels/chunk_o.py. Dispatch
    # TryTritonChunkO (VT_GDN_CHUNKO_TRITON) in cuda_gdn.cu; hand GdnChunkOWmmaKernel
    # preserved as fallback. Two specs per output dtype (H=48 27B, H=32 35B);
    # K=V=128, Hg=16, BT=64, BK=BV=64 (grid.x=cdiv(V,BV)=2). USE_G=1,
    # IS_VARLEN=1. o(=out) is f32 for the default GDN recurrence-output dtype and
    # bf16 for the vLLM-faithful core/z dtype follow-up. Grid (cdiv(V,BV), NT, H);
    # H=B*H baked (varlen B=1), NT is the runtime grid-carrier scalar. scale is
    # PINNED to Dk^-0.5 inside the kernel (Triton AOT mis-packs fp32 scalars), NOT
    # a sig arg.
    # Args: q k v h g o cu_seqlens chunk_indices T NT | H Hg K V BT BK BV USE_G IS_VARLEN.
    foreach(_H 48 32)
      set(VT_GDN_CHUNKO_BK_${_H}     64 CACHE STRING "chunk_o H=${_H} Triton BK")
      set(VT_GDN_CHUNKO_BV_${_H}     64 CACHE STRING "chunk_o H=${_H} Triton BV")
      set(VT_GDN_CHUNKO_WARPS_${_H}  4  CACHE STRING "chunk_o H=${_H} Triton num_warps")
      set(VT_GDN_CHUNKO_STAGES_${_H} 3  CACHE STRING "chunk_o H=${_H} Triton num_stages")
    endforeach()
    set(_CHUNKO_KPY "${CMAKE_SOURCE_DIR}/triton_kernels/chunk_o.py")
    set(_CHUNKO_SIG_HEAD_F32 "*bf16:16, *bf16:16, *bf16:16, *bf16:16, *fp32:16, *fp32:16, *i32:16, *i32:16, i32, i32")
    set(_CHUNKO_SIG_HEAD_BF16 "*bf16:16, *bf16:16, *bf16:16, *bf16:16, *fp32:16, *bf16:16, *i32:16, *i32:16, i32, i32")
    foreach(_H 48 32)
      set(_CHUNKO_SIG_TAIL "16, 128, 128, 64, ${VT_GDN_CHUNKO_BK_${_H}}, ${VT_GDN_CHUNKO_BV_${_H}}, 1, 1")
      math(EXPR _CHUNKO_GX "(128 + ${VT_GDN_CHUNKO_BV_${_H}} - 1) / ${VT_GDN_CHUNKO_BV_${_H}}")
      set(_sig "${_CHUNKO_SIG_HEAD_F32}, ${_H}, ${_CHUNKO_SIG_TAIL}")
      add_triton_kernel(VT_GDN_CHUNKO_${_H}_SRCS
        "${_CHUNKO_KPY}" chunk_fwd_kernel_o gdn_chunko_h${_H}
        "${_sig}" "${_CHUNKO_GX},NT,${_H}"
        ${VT_GDN_CHUNKO_WARPS_${_H}} ${VT_GDN_CHUNKO_STAGES_${_H}})
      target_sources(vllm PRIVATE ${VT_GDN_CHUNKO_${_H}_SRCS})
      target_include_directories(vllm PRIVATE ${VT_GDN_CHUNKO_${_H}_SRCS_INCLUDE_DIR})

      set(_sig "${_CHUNKO_SIG_HEAD_BF16}, ${_H}, ${_CHUNKO_SIG_TAIL}")
      add_triton_kernel(VT_GDN_CHUNKO_BF16_${_H}_SRCS
        "${_CHUNKO_KPY}" chunk_fwd_kernel_o gdn_chunko_bf16_h${_H}
        "${_sig}" "${_CHUNKO_GX},NT,${_H}"
        ${VT_GDN_CHUNKO_WARPS_${_H}} ${VT_GDN_CHUNKO_STAGES_${_H}})
      target_sources(vllm PRIVATE ${VT_GDN_CHUNKO_BF16_${_H}_SRCS})
      target_include_directories(vllm PRIVATE ${VT_GDN_CHUNKO_BF16_${_H}_SRCS_INCLUDE_DIR})
    endforeach()
    target_compile_definitions(vllm PUBLIC VLLM_CPP_TRITON_CHUNKO_BF16=1)

    # ── SANCTIONED Triton AOT: GDN packed pure-decode recurrence ────────────────
    # FLA fused_recurrent.py:256-336 fused_recurrent_gated_delta_rule_packed_decode
    # _kernel -> triton_kernels/fused_recurrent_packed_decode.py. Dispatch
    # TryTritonPackedDecode (VT_GDN_PACKED_DECODE_TRITON, default ON — vLLM's exact
    # token-identical FLA kernel, MIRROR policy; =0 rolls back to the hand
    # GdnPackedDecodeKernel in the same binary, which also stays the portable
    # fallback for any unsupported shape). MEASURED codegen-bound (dgx phase1 2026-07-16): the
    # register-resident [BV=32,BK=128] fp32 state tile is REG:205/0-spill under
    # Triton but REG:255+STACK:48 (spills) as hand-CUDA. Dense packed-decode
    # specializations cover Hk=16, Hv in {48,32}, K=V=128, BK=128, BV=32; the
    # 35B MoE model does not select packed decode. num_warps=1,
    # num_stages=3 (FLA launch :465-466). grid.x = cdiv(V,BV) = 4 baked; grid.y =
    # NBH (= B*HV) is the runtime carrier. scale PINNED to Dk^-0.5 in-kernel
    # (Triton AOT mis-packs fp32 scalars). Strides/dims are BAKED to each dense
    # call site and GUARDED in TryTritonPackedDecode. State-index ABI adapter:
    # skip `state_idx < 0` (our
    # cache ABI; FLA skips <= 0). Args: mixed_qkv a b A_log dt_bias o h0 ht
    # ssm_state_indices NBH | strides... H HV K V BK BV SOFTPLUS_THRESHOLD
    # USE_QK_L2NORM_IN_KERNEL. dtypes: mixed_qkv/a/b/o bf16, A_log/dt_bias/h0/ht
    # fp32, indices i32.
    set(_DECODE_KPY "${CMAKE_SOURCE_DIR}/triton_kernels/fused_recurrent_packed_decode.py")
    set(_DECODE_KFN fused_recurrent_gated_delta_rule_packed_decode_kernel)
    foreach(_HV 48 32)
      set(VT_GDN_DECODE_WARPS_${_HV} 1 CACHE STRING
          "packed decode Hv=${_HV} Triton num_warps")
      set(VT_GDN_DECODE_STAGES_${_HV} 3 CACHE STRING
          "packed decode Hv=${_HV} Triton num_stages")
      if(_HV EQUAL 48)
        set(_DECODE_MIXED_STRIDE 10240)
        set(_DECODE_BA_STRIDE 96)
        set(_DECODE_STATE_STRIDE 786432)
      else()
        set(_DECODE_MIXED_STRIDE 8192)
        set(_DECODE_BA_STRIDE 64)
        set(_DECODE_STATE_STRIDE 524288)
      endif()
      set(_DECODE_SIG "*bf16:16, *bf16:16, *bf16:16, *fp32:16, *fp32:16, *bf16:16, *fp32:16, *fp32:16, *i32:16, i32, ${_DECODE_MIXED_STRIDE}, ${_DECODE_BA_STRIDE}, ${_DECODE_BA_STRIDE}, ${_DECODE_STATE_STRIDE}, ${_DECODE_STATE_STRIDE}, 1, 16, ${_HV}, 128, 128, 128, 32, 20, 1")
      add_triton_kernel(VT_GDN_DECODE_${_HV}_SRCS
        "${_DECODE_KPY}" ${_DECODE_KFN} gdn_decode_h${_HV}
        "${_DECODE_SIG}" "4,NBH,1"
        ${VT_GDN_DECODE_WARPS_${_HV}} ${VT_GDN_DECODE_STAGES_${_HV}})
      target_sources(vllm PRIVATE ${VT_GDN_DECODE_${_HV}_SRCS})
      target_include_directories(vllm PRIVATE
        ${VT_GDN_DECODE_${_HV}_SRCS_INCLUDE_DIR})
    endforeach()

    # ── SANCTIONED Triton AOT: GDN WU pipeline (kkt -> solve_tril -> recompute_w_u)
    # The 3 FLA kernels our single fused GdnChunkWUWmmaVecKernel mirrors; ported so
    # the WU (biggest ~25% GDN) share runs on Triton codegen. Dispatch TryTritonWU
    # (VT_GDN_WU_TRITON) in cuda_gdn.cu runs all 3 into scratch A(f32)/Ai(bf16) and
    # produces w,u; hand fused kernel preserved as fallback. All grid-y = H (varlen
    # B=1, baked), grid-x = NT (runtime carrier). H=48 (27B) / 32 (35B).
    # Pins verified against FLA's OWN autotuner (GB10 2026-07-10, exact engine
    # shapes, T-stable 1024..8192; ~/work/finisher_autotune.log) AND per-kernel
    # nsys on OUR AOT variants (.agents/parity-ledger.md 2026-07-10):
    #  * delta_h BV=64/w4/s3, chunk_o BK=BV=64/w4/s3: FLA winners == old pins
    #    (CONFIRMED optimal, both H).
    #  * kkt H=48: FLA winner BK=128/w8/s3 measured −2.7% vs BK=64/w4/s3 → pinned.
    #  * tril/wu H=48: FLA winners (w8/s5, w2/s2) measured SLOWER on our AOT
    #    variants (+6.8%, +1.7% — the AOT specialization shifts the optimum) →
    #    keep the proven w4/s3 pins. Recorded as a measured dead-end.
    #  * H=32 (35B): kkt FLA winner == old pin; tril w8/s3 + wu w4/s2 are FLA
    #    winners but UNMEASURED on our AOT (and the H=48 measurement shows FLA
    #    winners don't transfer blindly) → keep proven w4/s3 pins; candidates
    #    recorded in the ledger.
    set(VT_GDN_KKT_BK_48      128 CACHE STRING "kkt H=48 Triton BK (32/64/128)")
    set(VT_GDN_KKT_WARPS_48   8   CACHE STRING "kkt H=48 Triton num_warps")
    set(VT_GDN_KKT_STAGES_48  3   CACHE STRING "kkt H=48 Triton num_stages")
    set(VT_GDN_KKT_BK_32      64  CACHE STRING "kkt H=32 Triton BK (32/64/128)")
    set(VT_GDN_KKT_WARPS_32   4   CACHE STRING "kkt H=32 Triton num_warps")
    set(VT_GDN_KKT_STAGES_32  3   CACHE STRING "kkt H=32 Triton num_stages")
    set(VT_GDN_TRIL_WARPS_48  4   CACHE STRING "solve_tril H=48 Triton num_warps")
    set(VT_GDN_TRIL_STAGES_48 3   CACHE STRING "solve_tril H=48 Triton num_stages")
    set(VT_GDN_TRIL_WARPS_32  4   CACHE STRING "solve_tril H=32 Triton num_warps")
    set(VT_GDN_TRIL_STAGES_32 3   CACHE STRING "solve_tril H=32 Triton num_stages")
    set(VT_GDN_WU_WARPS_48    4   CACHE STRING "recompute_w_u H=48 Triton num_warps")
    set(VT_GDN_WU_STAGES_48   3   CACHE STRING "recompute_w_u H=48 Triton num_stages")
    set(VT_GDN_WU_WARPS_32    4   CACHE STRING "recompute_w_u H=32 Triton num_warps")
    set(VT_GDN_WU_STAGES_32   3   CACHE STRING "recompute_w_u H=32 Triton num_stages")
    # (1) chunk_scaled_dot_kkt_fwd_kernel: k beta g A cu_seqlens chunk_indices T NT |
    #     H Hg K BT BK IS_VARLEN USE_G. A out f32.
    set(_KKT_KPY "${CMAKE_SOURCE_DIR}/triton_kernels/chunk_scaled_dot_kkt.py")
    set(_KKT_HEAD "*bf16:16, *fp32:16, *fp32:16, *fp32:16, *i32:16, *i32:16, i32, i32")
    # (2) merge_16x16_to_64x64_inverse_kernel: A Ai cu_seqlens chunk_indices T NT |
    #     H BT IS_VARLEN. A in f32, Ai out bf16. USE_TMA/DOT_PRECISION hardcoded.
    set(_TRIL_KPY "${CMAKE_SOURCE_DIR}/triton_kernels/solve_tril.py")
    set(_TRIL_HEAD "*fp32:16, *bf16:16, *i32:16, *i32:16, i32, i32")
    set(_TRIL_TAIL "64, 1")
    # (3) recompute_w_u_fwd_kernel: k v beta w u A g cu_seqlens chunk_indices T NT |
    #     H Hg K V BT BK BV IS_VARLEN. BK=BV=64 (pinned by the FLA wrapper,
    #     wy_fast.py:135-136 — NOT autotuned there). A in bf16(=Ai); w,u out bf16.
    set(_WU_KPY "${CMAKE_SOURCE_DIR}/triton_kernels/wy_fast.py")
    set(_WU_HEAD "*bf16:16, *bf16:16, *fp32:16, *bf16:16, *bf16:16, *bf16:16, *fp32:16, *i32:16, *i32:16, i32, i32")
    set(_WU_TAIL "16, 128, 128, 64, 64, 64, 1")
    foreach(_H 48 32)
      set(_KKT_TAIL "16, 128, 64, ${VT_GDN_KKT_BK_${_H}}, 1, 1")
      add_triton_kernel(VT_GDN_KKT_${_H}_SRCS
        "${_KKT_KPY}" chunk_scaled_dot_kkt_fwd_kernel gdn_kkt_h${_H}
        "${_KKT_HEAD}, ${_H}, ${_KKT_TAIL}" "NT,${_H},1"
        ${VT_GDN_KKT_WARPS_${_H}} ${VT_GDN_KKT_STAGES_${_H}})
      target_sources(vllm PRIVATE ${VT_GDN_KKT_${_H}_SRCS})
      target_include_directories(vllm PRIVATE ${VT_GDN_KKT_${_H}_SRCS_INCLUDE_DIR})

      add_triton_kernel(VT_GDN_TRIL_${_H}_SRCS
        "${_TRIL_KPY}" merge_16x16_to_64x64_inverse_kernel gdn_tril_h${_H}
        "${_TRIL_HEAD}, ${_H}, ${_TRIL_TAIL}" "NT,${_H},1"
        ${VT_GDN_TRIL_WARPS_${_H}} ${VT_GDN_TRIL_STAGES_${_H}})
      target_sources(vllm PRIVATE ${VT_GDN_TRIL_${_H}_SRCS})
      target_include_directories(vllm PRIVATE ${VT_GDN_TRIL_${_H}_SRCS_INCLUDE_DIR})

      add_triton_kernel(VT_GDN_WU_${_H}_SRCS
        "${_WU_KPY}" recompute_w_u_fwd_kernel gdn_wu_h${_H}
        "${_WU_HEAD}, ${_H}, ${_WU_TAIL}" "NT,${_H},1"
        ${VT_GDN_WU_WARPS_${_H}} ${VT_GDN_WU_STAGES_${_H}})
      target_sources(vllm PRIVATE ${VT_GDN_WU_${_H}_SRCS})
      target_include_directories(vllm PRIVATE ${VT_GDN_WU_${_H}_SRCS_INCLUDE_DIR})
    endforeach()

    # ── SANCTIONED Triton AOT: KDA chunk-prefill family (Kimi-Linear; H=32) ──────
    # FLA ops/kda.py chunk_kda_with_fused_gate forward chain (5 net-new kernels +
    # a NEW pin of chunk_delta_h.py). Dispatch TryKdaChunkPrefill (VT_KIMI_DEVICE_
    # KDA_CHUNK) in cuda_gdn.cu (vt::KdaChunkPrefill); the #104 recurrent
    # vt::KdaGatedDeltaRule stays the portable/decode fallback. Signatures pinned
    # to the Kimi KDA geometry (H=32, K=V=128, BT=64, BC=16, NC=4); scalar
    # constants (scale=K^-0.5, softplus beta/threshold, RCP_LN2, DOT_PRECISION)
    # are baked as kernel literals (Triton AOT mis-packs fp32 scalars). Each line
    # is byte-identical to the cmake/TritonAOTKernels.cmake declaration (the drift
    # check compares the two). See .agents/specs/kimi-linear.md §17.
    set(_KDA_GATE_KPY "${CMAKE_SOURCE_DIR}/triton_kernels/kda_gate_cumsum.py")
    set(_KDA_KKT_KPY "${CMAKE_SOURCE_DIR}/triton_kernels/chunk_kda_kkt.py")
    set(_KDA_WU_KPY "${CMAKE_SOURCE_DIR}/triton_kernels/recompute_w_u_kda.py")
    set(_KDA_DELTAH_KPY "${CMAKE_SOURCE_DIR}/triton_kernels/chunk_delta_h.py")
    set(_KDA_GLAO_KPY "${CMAKE_SOURCE_DIR}/triton_kernels/chunk_gla_o.py")

    add_triton_kernel(VT_KDA_GATE_CUMSUM_SRCS
      "${_KDA_GATE_KPY}" kda_gate_cumsum_fwd_kernel kda_gate_cumsum
      "*bf16:16, *fp32:16, *fp32:16, *fp32:16, *i32:16, *i32:16, i32, i32, 32, 128, 64, 64, 1, 1"
      "2,NT,32" 4 2)
    target_sources(vllm PRIVATE ${VT_KDA_GATE_CUMSUM_SRCS})
    target_include_directories(vllm PRIVATE ${VT_KDA_GATE_CUMSUM_SRCS_INCLUDE_DIR})

    add_triton_kernel(VT_KDA_KKT_INTER_SRCS
      "${_KDA_KKT_KPY}" chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter kda_kkt_inter
      "*bf16:16, *bf16:16, *fp32:16, *bf16:16, *fp32:16, *fp32:16, *i32:16, *i32:16, i32, i32, 32, 128, 64, 16, 64, 4, 1"
      "NT,16,32" 4 3)
    target_sources(vllm PRIVATE ${VT_KDA_KKT_INTER_SRCS})
    target_include_directories(vllm PRIVATE ${VT_KDA_KKT_INTER_SRCS_INCLUDE_DIR})

    add_triton_kernel(VT_KDA_KKT_INTRA_SRCS
      "${_KDA_KKT_KPY}" chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra kda_kkt_intra
      "*bf16:16, *bf16:16, *fp32:16, *bf16:16, *fp32:16, *fp32:16, *i32:16, *i32:16, i32, i32, 32, 128, 64, 16, 128, 1"
      "NT,4,32" 4 2)
    target_sources(vllm PRIVATE ${VT_KDA_KKT_INTRA_SRCS})
    target_include_directories(vllm PRIVATE ${VT_KDA_KKT_INTRA_SRCS_INCLUDE_DIR})

    add_triton_kernel(VT_KDA_WU_SRCS
      "${_KDA_WU_KPY}" recompute_w_u_fwd_kernel kda_wu
      "*bf16:16, *bf16:16, *bf16:16, *bf16:16, *bf16:16, *bf16:16, *bf16:16, *bf16:16, *bf16:16, *fp32:16, *i32:16, *i32:16, i32, i32, 32, 128, 128, 64, 64, 64, 0, 1, 1"
      "NT,32,1" 4 3)
    target_sources(vllm PRIVATE ${VT_KDA_WU_SRCS})
    target_include_directories(vllm PRIVATE ${VT_KDA_WU_SRCS_INCLUDE_DIR})

    add_triton_kernel(VT_KDA_DELTAH_H32_SRCS
      "${_KDA_DELTAH_KPY}" chunk_gated_delta_rule_fwd_kernel_h_blockdim64 kda_deltah_h32
      "*bf16:16, *bf16:16, *bf16:16, *bf16:16, *fp32, *fp32:16, *bf16:16, *fp32:16, *fp32:16, *i32:16, *i32:16, i32, i32, 32, 32, 128, 128, 64, 64, 0, 1, 1, 1, 1, 1, 1"
      "2,NH,1" 4 3)
    target_sources(vllm PRIVATE ${VT_KDA_DELTAH_H32_SRCS})
    target_include_directories(vllm PRIVATE ${VT_KDA_DELTAH_H32_SRCS_INCLUDE_DIR})

    add_triton_kernel(VT_KDA_GLA_O_SRCS
      "${_KDA_GLAO_KPY}" chunk_gla_fwd_kernel_o kda_gla_o
      "*bf16:16, *bf16:16, *fp32:16, *bf16:16, *bf16:16, *fp32:16, *i32:16, *i32:16, i32, i32, 32, 128, 128, 64, 64, 64, 1"
      "2,NT,32" 4 3)
    target_sources(vllm PRIVATE ${VT_KDA_GLA_O_SRCS})
    target_include_directories(vllm PRIVATE ${VT_KDA_GLA_O_SRCS_INCLUDE_DIR})

    # Vendored-tree bookkeeping (call ONCE, after the LAST add_triton_kernel):
    # REGEN writes the MANIFEST (generator versions + triton_kernels/*.py hashes
    # + per-base generation params); the normal builder path verifies those
    # hashes against the vendored tree and WARNS loudly on drift.
    triton_aot_finalize()

    target_compile_definitions(vllm PUBLIC VLLM_CPP_TRITON=1)
    target_link_libraries(vllm PUBLIC CUDA::cuda_driver)
  endif()

  # Vendored FlashAttention-2 sm_121a prefill + pure-decode split-KV. The exact
  # FA-2 source vLLM builds
  # as _vllm_fa2_C (vllm-project/flash-attention @ 2c839c33) — its
  # flash_fwd_splitkv kernel is what vLLM's nsys trace shows for prefill on GB10.
  # Compiled torch-free against our CUTLASS: 3 stub headers replace the ATen/c10
  # surface (PhiloxCudaState POD + C10_CUDA_CHECK), and fa2_compat_prelude.h is
  # force-included for the std headers ATen pulled in transitively — the upstream
  # (auto-generated) FA sources stay byte-for-byte pristine. DEFAULT ON (like
  # Marlin/cutlass) — it only actually builds when CUTLASS is present
  # (VLLM_CPP_CUTLASS, sm_12xa); -DVLLM_CPP_FLASH_ATTN=OFF opts out. Runtime
  # toggles VT_FA2_PREFILL and VT_FA2_DECODE (each default ON when compiled;
  # =0 restores its same-binary fallback). Prefill was validated token-for-token
  # + over the throughput bar on GB10 2026-07-10; ratio-6 decode remains GATING.
  option(VLLM_CPP_FLASH_ATTN "Build vendored FlashAttention-2 prefill/decode kernels (needs CUTLASS headers)" ON)
  # Gate on VLLM_CPP_CUTLASS_HEADERS (arch-independent header availability), NOT
  # VLLM_CPP_CUTLASS (the sm_12x NVFP4 feature flag): FA2 also resolves for Ampere
  # sm_8x now (WA-1), where the NVFP4 GEMM is absent but the headers still compile.
  if(VLLM_CPP_FLASH_ATTN AND VLLM_CPP_CUTLASS_HEADERS AND VT_FA2_ARCHS)
    set(_FA2_ROOT "${CMAKE_SOURCE_DIR}/src/vt/cuda/flash_attn")
    set(_FA2_KERNEL_SRCS
      src/vt/cuda/flash_attn/src/flash_fwd_split_hdim256_bf16_causal_sm80.cu
      src/vt/cuda/flash_attn/src/flash_fwd_split_hdim256_bf16_sm80.cu
      # d128 bf16 split-KV: the exact varlen (non-swap) reduction vLLM's
      # flash_attn_varlen_func runs for Qwen3-dense DECODE (paged KV forces the
      # split kernel even at num_splits==1). MODEL-TEXT-qwen3 strict bit-match.
      src/vt/cuda/flash_attn/src/flash_fwd_split_hdim128_bf16_causal_sm80.cu
      src/vt/cuda/flash_attn/src/flash_fwd_split_hdim128_bf16_sm80.cu
      # d192 bf16 split-KV (MLA campaign W5): the MLA PREFILL QK head dim
      # (qk_nope 128 + qk_rope 64). V is zero-padded 128 -> 192 exactly as
      # upstream does (mla/prefill/flash_attn.py:164-168), so the kernel stays a
      # symmetric instantiation of the SAME generic template — no template edit,
      # hence the d128/d256 kernels the 27B/35B/dense paths call are unchanged.
      src/vt/cuda/flash_attn/src/flash_fwd_split_hdim192_bf16_causal_sm80.cu
      src/vt/cuda/flash_attn/src/flash_fwd_split_hdim192_bf16_sm80.cu
      # d64 bf16 NON-SPLIT (multimodal-speed §17): the Whisper AUDIO ENCODER
      # self-attention. Two firsts, both deliberate — head_dim 64 (Whisper-large-v3
      # is d_model 1280 / 20 heads) and the plain batch `run_mha_fwd_` entry rather
      # than the split-KV dispatch, because the encoder is dense and non-paged where
      # every prior caller attends over a paged KV cache. vLLM reaches the same FA-2
      # forward here (whisper.py WhisperEncoderAttention:255 -> forward:298-317 ->
      # flash_attn_varlen_func, causal=False). Non-causal only: no causal hd-64
      # caller exists.
      src/vt/cuda/flash_attn/src/flash_fwd_hdim64_bf16_sm80.cu
      # d128 bf16 NON-SPLIT (#1551): the LTX-2.5 DiT self-attention, 32 heads x
      # head_dim 128 over 2352 video tokens. Same one-line ask as the hd-64 file
      # above for a different head dim; the hd-128 SPLIT-KV instantiation two
      # entries up already compiles these kernel traits and `mma.sync` bodies, so
      # this adds an entry point rather than a kernel. Non-causal only: LTX's DiT
      # self-attention is bidirectional and no causal non-split hd-128 caller
      # exists. Unlike the flash-tiled scalar op, the vendored launcher raises its
      # own dynamic shared-memory cap (flash_fwd_launch_template.h:85-88), so the
      # 48 KiB default cap that bounds AttentionDenseFlash (#1544) does not bound
      # this path.
      src/vt/cuda/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu)
    target_sources(vllm PRIVATE
      src/vt/cuda/cuda_flash_attn_fa2.cu
      ${_FA2_KERNEL_SRCS})
    target_compile_definitions(vllm PUBLIC VLLM_CPP_FLASH_ATTN)
    # NOTE (QUANT-CT-MXFP4-FLASH-AUDIT #68 + FLASH-OCCUPANCY #75): the c8 decode
    # flash gap vs vLLM (ours ~168-174 vs vLLM ~156 us/call) is a COMPILER-CODEGEN
    # gap, MEASURED to a verdict — NOT occupancy, NOT L2, NOT num_splits, and NOT a
    # flag we can set. The owed matched-c8 ncu (both engines, IDENTICAL grid
    # 1x3x64) shows occupancy is IDENTICAL 8.33% (both smem-limited to 1 CTA/SM by
    # the byte-identical 81.92 KB kSmemSize — the 216-vs-241 reg delta is moot),
    # L2 hit ~1% on BOTH (KV streamed from DRAM: no adjacency/warmth to win), and
    # the same ~60% short-scoreboard stall structure. The ONLY difference: vLLM's
    # SASS executes ~13% FEWER instructions (3.27M vs 3.70M) in this latency-bound
    # ~1-warp/scheduler regime — but that instruction count is NOT the cause. #75
    # built the exact vLLM recipe: compiling the kernel TUs for compute_80 PTX +
    # -use_fast_math REPRODUCES vLLM's SASS profile EXACTLY (241 reg, 17,008 instr
    # vs vLLM 241/17,020) and is STILL ~167 us — the SAME as our native ~165 us and
    # still ~10 us slower than vLLM's ~157 us. So matching vLLM's arch, fast-math,
    # register AND instruction counts does NOT close the gap: the kernel is not
    # instruction-bound, and the residual is vLLM's wheel-`ptxas` SASS-SCHEDULING
    # quality, not reachable from our nvcc 13.0. (native+fast-math REGRESSES to
    # 189.8 us, #68; compute_80 PTX driver-JIT alone is neutral, #75 Build B.) So
    # we keep the plain native sm_121a compile and pass no -use_fast_math. #82
    # (FLASH-PTXAS) then ARBITRATED the "ptxas lineage" claim to a NO and CORRECTED
    # it: the wheel ships NO sm_12x cubin — only CUDA-13.0 (PTX ISA 9.0) sm_80 PTX
    # driver-JIT'd on GB10 — so there is no old-CUDA-12.x ptxas in vLLM's path (and
    # ptxas 12.8 cannot even target sm_121a); a same-params cuModule A/B ties our
    # PTX and vLLM's OWN PTX at ~144 us across driver-JIT / ptxas 13.0 / 13.2, so the
    # +10 us/call gap is ENGINE CONTEXT (neighbour L2/orchestration), not this
    # kernel's codegen. See .agents/benchmark-record.md (#75, #82).
    set_source_files_properties(
      src/vt/cuda/cuda_flash_attn_fa2.cu
      ${_FA2_KERNEL_SRCS} PROPERTIES
      COMPILE_OPTIONS
        "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda;-DFLASH_NAMESPACE=flash;-include;${_FA2_ROOT}/fa2_compat_prelude.h;-isystem;${_FA2_ROOT}/src;-isystem;${_FA2_ROOT}/stubs;-isystem;${VLLM_CPP_CUTLASS_DIR}/include>")
    message(STATUS "FlashAttention-2 prefill/decode: ENABLED for arch(es) [${VT_FA2_ARCHS}] (runtime toggles VT_FA2_PREFILL, VT_FA2_DECODE)")
  endif()

  # RELEASE W1: CMake's target-wide CUDA_ARCHITECTURES property would compile
  # every CUDA TU for every requested SM, including architecture-specific
  # CUTLASS/Marlin/FA2 bodies that ptxas rejects outside their feature-table
  # cells. Disable target-wide emission for libvllm and give every CUDA source
  # exactly one explicit gencode set: portable sources receive the complete
  # requested list, feature sources receive only their resolved intersection.
  # This mirrors vLLM's set_gencode_flags_for_srcs rather than inventing a
  # release-only compilation path.
  set_property(TARGET vllm PROPERTY CUDA_ARCHITECTURES OFF)
  set(_VT_CUDA_FEATURE_SOURCES)

  if(VT_FP4_MMA_ARCHS)
    target_sources(vllm PRIVATE src/vt/cuda/cuda_nvfp4_sm12x.cu)
    vt_cuda_set_source_gencode("${VT_FP4_MMA_ARCHS}"
      src/vt/cuda/cuda_nvfp4_sm12x.cu)
    list(APPEND _VT_CUDA_FEATURE_SOURCES src/vt/cuda/cuda_nvfp4_sm12x.cu)
  endif()

  if(_NVFP4_CUTLASS_SOURCES)
    vt_cuda_set_source_gencode("${VT_CUTLASS_NVFP4_ARCHS}"
      ${_NVFP4_CUTLASS_SOURCES})
    list(APPEND _VT_CUDA_FEATURE_SOURCES ${_NVFP4_CUTLASS_SOURCES})
  endif()
  if(_FP8_CUTLASS_SOURCES)
    vt_cuda_set_source_gencode("${VT_CUTLASS_FP8_ARCHS}"
      ${_FP8_CUTLASS_SOURCES})
    list(APPEND _VT_CUDA_FEATURE_SOURCES ${_FP8_CUTLASS_SOURCES})
  endif()
  if(VLLM_CPP_CUTLASS_NVFP4_SM100)
    vt_cuda_set_source_gencode("${VT_CUTLASS_NVFP4_SM100_ARCHS}"
      src/vt/cuda/cuda_matmul_nvfp4_sm100.cu)
    list(APPEND _VT_CUDA_FEATURE_SOURCES
      src/vt/cuda/cuda_matmul_nvfp4_sm100.cu)
  endif()
  if(VLLM_CPP_SCALEDMM_C3X_SM90)
    vt_cuda_set_source_gencode("${VT_SCALEDMM_C3X_SM90_ARCHS}"
      src/vt/cuda/cuda_scaled_mm_c3x_sm90.cu)
    list(APPEND _VT_CUDA_FEATURE_SOURCES
      src/vt/cuda/cuda_scaled_mm_c3x_sm90.cu)
  endif()
  if(VLLM_CPP_SCALEDMM_C3X_SM100)
    vt_cuda_set_source_gencode("${VT_SCALEDMM_C3X_SM100_ARCHS}"
      src/vt/cuda/cuda_scaled_mm_c3x_sm100.cu)
    list(APPEND _VT_CUDA_FEATURE_SOURCES
      src/vt/cuda/cuda_scaled_mm_c3x_sm100.cu)
  endif()
  if(_MARLIN_SRCS)
    vt_cuda_set_source_gencode("${VT_MARLIN_NVFP4_ARCHS}" ${_MARLIN_SRCS})
    list(APPEND _VT_CUDA_FEATURE_SOURCES ${_MARLIN_SRCS})
  endif()
  if(_FA2_KERNEL_SRCS)
    set(_VT_FA2_SOURCES
      src/vt/cuda/cuda_flash_attn_fa2.cu
      ${_FA2_KERNEL_SRCS})
    vt_cuda_set_source_gencode("${VT_FA2_ARCHS}" ${_VT_FA2_SOURCES})
    # THE MANIFEST, taken from the SAME argument that just became nvcc's
    # -gencode options for these exact sources (issue #1357). Reaching this line
    # is what "FA2 is compiled" means, and the architectures are the ones the
    # compiler was handed -- not a second list that has to be kept honest.
    string(REPLACE ";" "," VLLM_CPP_CUDA_FA2_COMPILED_ARCHS "${VT_FA2_ARCHS}")
    list(APPEND _VT_CUDA_FEATURE_SOURCES ${_VT_FA2_SOURCES})
  endif()

  get_target_property(_VT_VLLM_TARGET_SOURCES vllm SOURCES)
  set(_VT_CUDA_PORTABLE_SOURCES ${_VT_VLLM_TARGET_SOURCES})
  list(FILTER _VT_CUDA_PORTABLE_SOURCES INCLUDE REGEX "\\.cu$")
  list(REMOVE_ITEM _VT_CUDA_PORTABLE_SOURCES ${_VT_CUDA_FEATURE_SOURCES})
  vt_cuda_set_source_gencode("${VLLM_CPP_CUDA_ARCHITECTURES}"
    ${_VT_CUDA_PORTABLE_SOURCES})
  message(STATUS
    "CUDA per-source gencode: ${VLLM_CPP_CUDA_ARCHITECTURES} portable; "
    "feature TUs narrowed by VT_CUDA_FEATURE_TABLE")
endif()

# OpenAI HTTP server (M3.1 Task 4): the api_server.cpp translation unit pulls in
# the vendored cpp-httplib transport header, so it is compiled into the library
# only when VLLM_CPP_SERVER is enabled. httplib needs threads (and sockets).
if(VLLM_CPP_SERVER)
  find_package(Threads REQUIRED)
  target_sources(vllm PRIVATE src/vllm/entrypoints/openai/api_server.cpp)
  # ARCH-ONE-SURFACE: the server ENTRY POINT (flag parsing + engine/serving
  # construction) lives in the library so examples/server can be a thin client of
  # the C ABI's vllm_server_main (v17). It shares api_server.cpp's httplib gate.
  target_sources(vllm PRIVATE src/vllm/entrypoints/openai/server_main.cpp)
  # ENG-HF-MODEL-DOWNLOAD (#1280): the hub protocol speaks HTTP through the same
  # vendored cpp-httplib header api_server.cpp uses, so it shares that gate. The
  # cache half above stays unconditional because it opens no socket.
  target_sources(vllm PRIVATE src/vllm/transformers_utils/hf_hub.cpp)
  # ENG-HF-MODEL-DOWNLOAD W3/W4 (#1280): the byte transport and the `--model`
  # grammar. Both speak HTTP through the same vendored header, so both share
  # api_server.cpp's gate. The resolver is what `server_main.cpp` calls, so a
  # build without the server has no `--model` flag to resolve either.
  target_sources(vllm PRIVATE src/vllm/transformers_utils/downloader.cpp)
  target_sources(vllm PRIVATE src/vllm/transformers_utils/model_resolver.cpp)
  # ENG-HF-MODEL-DOWNLOAD W5 (#1280): the one-definition-rule instrument. It
  # holds the SERVER half's reading (api_server.cpp) beside the FETCHER half's
  # (hf_hub.cpp) and it deliberately includes no httplib header of its own, so
  # it cannot contribute a third opinion. server_main.cpp refuses to start on a
  # disagreement.
  target_sources(vllm PRIVATE src/vllm/http_transport_abi.cpp)
  target_compile_definitions(vllm PUBLIC VLLM_CPP_SERVER)

  # ── ENG-HF-MODEL-DOWNLOAD W5 (#1280): transport layer security ─────────────
  # `huggingface.co` speaks https and nothing else, so without this block the
  # `--model org/repo` grammar that W4 landed can only reach an `http` mirror.
  #
  # THE DEFINE GOES ON THE TARGET, PUBLIC, AND NEVER PER SOURCE FILE.
  # `CPPHTTPLIB_OPENSSL_SUPPORT` is a whole-header switch: it changes the layout
  # of `httplib::Result` and `httplib::ClientConnection`
  # (third_party/httplib/httplib.h:2045-2100). This tree includes that one
  # header from the OpenAI server, from the hub client, from the byte
  # downloader and from several test binaries, and they all link together. A
  # per-file define would LINK CLEANLY and then corrupt every response object
  # crossing the seam, with no diagnostic anywhere. PUBLIC so every consumer of
  # `vllm` -- tests and examples included -- inherits the same answer.
  # `src/vllm/http_transport_abi.cpp` and
  # `tests/vllm/transformers_utils/test_tls_transport.cpp` measure that it held.
  #
  # llama.cpp made the same choice at `vendor/cpp-httplib/CMakeLists.txt:184
  # @ b10451`, `target_compile_definitions(${TARGET} PUBLIC ...)`.
  if(VLLM_CPP_HF_DOWNLOAD)
    set(_vllm_tls_source "")
    if(VLLM_CPP_BUILD_BORINGSSL)
      # The literal-static lane: a fully static libssl/libcrypto that leaves no
      # entry in `readelf -dW`, which `scripts/validate-release-archive.py`
      # refuses on that artifact. Mirrors llama.cpp `LLAMA_BUILD_BORINGSSL`
      # (`vendor/cpp-httplib/CMakeLists.txt:40-82 @ b10451`), including the
      # BUILD_SHARED_LIBS/BUILD_TESTING save-and-restore and the pinned tag.
      # It fetches from the network at CONFIGURE time, which is why it is OFF by
      # default and why nothing in the default lane depends on it.
      set(OPENSSL_NO_ASM ON CACHE BOOL "Disable OpenSSL ASM when building BoringSSL")
      set(FIPS OFF CACHE BOOL "Enable FIPS (BoringSSL)")
      set(BORINGSSL_GIT "https://boringssl.googlesource.com/boringssl"
        CACHE STRING "BoringSSL git repository")
      set(BORINGSSL_VERSION "0.20260813.0" CACHE STRING "BoringSSL version")
      message(STATUS "Fetching BoringSSL ${BORINGSSL_VERSION}")
      include(FetchContent)
      FetchContent_Declare(boringssl
        GIT_REPOSITORY ${BORINGSSL_GIT}
        GIT_TAG        ${BORINGSSL_VERSION}
        EXCLUDE_FROM_ALL)
      set(_vllm_saved_shared ${BUILD_SHARED_LIBS})
      set(_vllm_saved_testing ${BUILD_TESTING})
      set(BUILD_SHARED_LIBS OFF)
      set(BUILD_TESTING OFF)
      FetchContent_MakeAvailable(boringssl)
      set(BUILD_SHARED_LIBS ${_vllm_saved_shared})
      set(BUILD_TESTING ${_vllm_saved_testing})
      # Vendored code is not on this project's -Werror path.
      foreach(_boringssl_target ssl crypto)
        if(TARGET ${_boringssl_target})
          target_compile_options(${_boringssl_target} PRIVATE
            $<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>)
        endif()
      endforeach()
      target_link_libraries(vllm PUBLIC ssl crypto)
      set(_vllm_tls_source "BoringSSL ${BORINGSSL_VERSION} (static, fetched)")
    elseif(VLLM_CPP_OPENSSL)
      find_package(OpenSSL)
      if(OpenSSL_FOUND)
        # The version gate is llama.cpp's, verbatim in intent
        # (`vendor/cpp-httplib/CMakeLists.txt:126-146 @ b10451`): OpenSSL 3.0 or
        # later, or 1.1.1g or later for BoringSSL and LibreSSL. It is not
        # invented here, and it is not a `VERSION_GREATER_EQUAL` on
        # `OPENSSL_VERSION` either, because the version STRING does not say
        # which of the three implementations answered `find_package`. The
        # vendored header raises the same `#error` at
        # `third_party/httplib/httplib.h:411-418`, so a build that skipped this
        # check would fail later and less legibly.
        include(CheckCSourceCompiles)
        set(_vllm_saved_required_includes ${CMAKE_REQUIRED_INCLUDES})
        set(CMAKE_REQUIRED_INCLUDES ${OPENSSL_INCLUDE_DIR})
        check_c_source_compiles("
        #include <openssl/opensslv.h>
        #if defined(OPENSSL_IS_BORINGSSL) || defined(LIBRESSL_VERSION_NUMBER)
        #    if OPENSSL_VERSION_NUMBER < 0x1010107f
        #        error bad version
        #    endif
        #else
        #    if OPENSSL_VERSION_NUMBER < 0x30000000L
        #        error bad version
        #    endif
        #endif
        int main() { return 0; }
        " VLLM_CPP_OPENSSL_VERSION_SUPPORTED)
        set(CMAKE_REQUIRED_INCLUDES ${_vllm_saved_required_includes})
        if(VLLM_CPP_OPENSSL_VERSION_SUPPORTED)
          target_link_libraries(vllm PUBLIC OpenSSL::SSL OpenSSL::Crypto)
          set(_vllm_tls_source "OpenSSL ${OPENSSL_VERSION} (system, dynamic)")
        else()
          message(WARNING
            "VLLM_CPP_OPENSSL=ON but the OpenSSL at ${OPENSSL_INCLUDE_DIR} is "
            "older than 3.0 (or older than 1.1.1g for BoringSSL/LibreSSL); "
            "HTTPS support is disabled.")
        endif()
      else()
        message(WARNING
          "VLLM_CPP_OPENSSL=ON but OpenSSL was not found; HTTPS support is "
          "disabled. Install the OpenSSL development files, or configure with "
          "-DVLLM_CPP_BUILD_BORINGSSL=ON.")
      endif()
    endif()

    if(_vllm_tls_source STREQUAL "")
      # The DOWNGRADE, in the VLLM_CPP_SERVER idiom above: a WARNING and an
      # option that resolves OFF, never a failed configure. The binary still
      # builds and still serves a local checkpoint; a repository identifier
      # against an `https` endpoint is refused by
      # `HfRefuseHttpsWithoutTls` with a message naming these three options.
      message(WARNING
        "VLLM_CPP_HF_DOWNLOAD=ON but no transport layer security is available, "
        "so --model org/repo can only reach an http:// endpoint. Install the "
        "OpenSSL development files (VLLM_CPP_OPENSSL=ON, the default) or "
        "configure with -DVLLM_CPP_BUILD_BORINGSSL=ON.")
      set(VLLM_CPP_HF_DOWNLOAD OFF)
    else()
      target_compile_definitions(vllm PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT)
      if(APPLE)
        # cpp-httplib reads the macOS keychain for root certificates whenever a
        # TLS backend is enabled on an Apple clang build
        # (third_party/httplib/httplib.h:344-353), which pulls in these two
        # frameworks. Mirrors llama.cpp `vendor/cpp-httplib/CMakeLists.txt:186-190`.
        find_library(CORE_FOUNDATION_FRAMEWORK CoreFoundation REQUIRED)
        find_library(SECURITY_FRAMEWORK Security REQUIRED)
        target_link_libraries(vllm PUBLIC
          ${CORE_FOUNDATION_FRAMEWORK} ${SECURITY_FRAMEWORK})
      endif()
      if(WIN32 AND NOT MSVC)
        target_link_libraries(vllm PUBLIC crypt32)
      endif()
      message(STATUS "HuggingFace download: HTTPS through ${_vllm_tls_source}")
    endif()
  else()
    message(STATUS
      "HuggingFace download: HTTPS disabled (VLLM_CPP_HF_DOWNLOAD=OFF); "
      "--model org/repo is refused against an https:// endpoint")
  endif()
  target_link_libraries(vllm PUBLIC Threads::Threads)
  # third_party/httplib/httplib.h is reached as <httplib/httplib.h> (third_party
  # is already a SYSTEM include on the vllm target).
endif()

# ── BIT-ROT GUARD: the MiniMax-Music3 depth-stage A/B driver (#672, #1246) ────
# `tools/bench/music3_depth_stage_ab.cpp` is the sole artifact behind spec §16.6's
# 3.50x and its `f0cfeed6eee4f55d` fingerprint. Its own header argues, correctly,
# that CI must never RUN it: it allocates 2.5 GB of weights, spends seconds per
# round, and is a TWO-BUILD A/B that one target could not express anyway. None of
# that is a reason for it to be UNCOMPILABLE — it reaches internal headers and
# `LinearNoBias`, so the next signature change would rot the one file a reader
# has to compile to reproduce the measurement, silently.
#
# So both arms are compiled and NOTHING is linked or registered as a test. An
# OBJECT library builds the TU and discards it; `main` is never linked, so no
# weight is ever allocated and the build stays byte-identical to what it was.
# Two targets rather than one because the arms are `#if defined(MUSIC3_AFTER)`
# alternatives and a single configuration would leave the other half unchecked.
foreach(_music3_ab_arm before after)
  add_library(vllm_music3_depth_stage_ab_${_music3_ab_arm} OBJECT
    tools/bench/music3_depth_stage_ab.cpp)
  target_include_directories(vllm_music3_depth_stage_ab_${_music3_ab_arm} PRIVATE
    ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR}/include)
  target_include_directories(vllm_music3_depth_stage_ab_${_music3_ab_arm} SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party)
  target_compile_features(vllm_music3_depth_stage_ab_${_music3_ab_arm} PRIVATE cxx_std_20)
  vllm_cpp_set_warnings(vllm_music3_depth_stage_ab_${_music3_ab_arm})
endforeach()
target_compile_definitions(vllm_music3_depth_stage_ab_after PRIVATE MUSIC3_AFTER)

# ── The MiniMax-Music3 vocoder decode-window A/B driver (#672, #1334) ─────────
# `tools/bench/music3_vocoder_conv_ab.cpp` is the artifact behind spec §18's
# numbers, and it is registered as no test, so nothing in `ctest` runs it.
#
# WHY THIS ONE IS AN EXECUTABLE WHERE ITS SIBLING 20 LINES ABOVE IS AN OBJECT
# LIBRARY. That divergence was a review finding (2026-08-19) and it is argued
# here rather than left to be noticed twice. The depth driver's two arms are
# `#if defined(MUSIC3_AFTER)` branches of ONE file, so compiling it is the whole
# rot guard and linking `main` would only add an entry point nobody calls; an
# OBJECT library is strictly cheaper and it says so.
#
# This driver's arms are two SOURCE TREES, the shape spec §18.8 requires and that
# §16.6a was voided for not having, and the recorded recipe is a SHIPPED SCRIPT:
# `scripts/music3-vocoder-conv-ab.sh` configures two clones, runs
# `ninja ... vllm_music3_vocoder_conv_ab` in each, `sha256sum`s the two results
# and REFUSES to time anything when they hash the same, then executes them with
# `--lengths=` / `--repeats=`. Every one of those steps needs a linked binary.
# An OBJECT library would guard the compile and break the reproduction, which
# inverts the point: the rot this target exists to prevent is a §18 nobody can
# re-run.
#
# What it costs, stated rather than hoped: one ~21 MB executable in a build tree,
# no test registration, and no `install(TARGETS ...)` entry — see the install
# list below, which names `vllm` and `vllm_shared` only — so it does not ship and
# no packaged artifact changes. `main` is linked and never runs in CI; the
# allocation the depth driver's comment worries about happens on invocation, not
# at load.
add_executable(vllm_music3_vocoder_conv_ab tools/bench/music3_vocoder_conv_ab.cpp)
target_link_libraries(vllm_music3_vocoder_conv_ab PRIVATE vllm)
target_include_directories(vllm_music3_vocoder_conv_ab PRIVATE
  ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR}/include)
target_include_directories(vllm_music3_vocoder_conv_ab SYSTEM PRIVATE
  ${CMAKE_SOURCE_DIR}/third_party)
target_compile_features(vllm_music3_vocoder_conv_ab PRIVATE cxx_std_20)
vllm_cpp_set_warnings(vllm_music3_vocoder_conv_ab)

# ── The `vt::Conv1d` decomposition probe (#672, #1334) ───────────────────────
# `tools/bench/conv1d_scaling_probe.cpp` is the instrument behind the scaling
# curve and the residency ablation in `.agents/specs/vt-conv1d-time-block.md`.
# It is registered as no test and CI runs it never, for the reason its sibling
# above gives: it allocates hundreds of megabytes and spends tens of seconds per
# sweep. The build COMPILES it so that the one file a reader has to compile to
# re-derive "what limits the vocoder's parallel scaling" cannot rot behind a
# `vt::Conv1d` or `ParallelForRows` signature change -- which is the failure the
# sibling target's comment records for three earlier harnesses that lived in
# prose only.
add_executable(vllm_conv1d_scaling_probe tools/bench/conv1d_scaling_probe.cpp)
target_link_libraries(vllm_conv1d_scaling_probe PRIVATE vllm)
target_include_directories(vllm_conv1d_scaling_probe PRIVATE
  ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR}/include)
target_include_directories(vllm_conv1d_scaling_probe SYSTEM PRIVATE
  ${CMAKE_SOURCE_DIR}/third_party)
target_compile_features(vllm_conv1d_scaling_probe PRIVATE cxx_std_20)
vllm_cpp_set_warnings(vllm_conv1d_scaling_probe)

# ── The LTX-2.5 connector GEMM probe (LTX25-CONNECTOR-GEMM) ─────────────────
# `tools/bench/ltx2_connector_gemm_probe.cpp` is the instrument behind
# `.agents/specs/ltx25-connector-gemm.md`: which CPU GEMM tier the LTX-2.5 text
# connector's `vt::MatmulBT` calls actually enter, what the two weight
# orientations cost at those shapes, and how much of `Ltx2ConnectorForward` the
# GEMM accounts for at all.
#
# CI COMPILES IT AND RUNS NOTHING, the same trade its two siblings above record:
# one leg allocates hundreds of megabytes and spends tens of seconds of every
# core, which no test tier should pay. What the build buys is that the one file
# a reader has to compile to re-derive "what the connector's 224.9 s is spent
# on" cannot rot behind a `vt::MatmulBT`, `vt::AttentionCross` or
# `Ltx2ConnectorForward` signature change -- the #1246 failure, where the sole
# named harness for a measurement was compiled by nothing.
#
# The recorded runs were taken from the DIRECT g++ compile in the file's own
# header rather than from this target, because the devbox they were taken on has
# no room for a CMake build tree. The target still compiles the whole file: it
# defines no `LTX2_PROBE_NO_CONNECTOR`, so the connector and attention modes are
# in the guard as well as the GEMM one.
add_executable(vllm_ltx2_connector_gemm_probe tools/bench/ltx2_connector_gemm_probe.cpp)
target_link_libraries(vllm_ltx2_connector_gemm_probe PRIVATE vllm)
target_include_directories(vllm_ltx2_connector_gemm_probe PRIVATE
  ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR}/include)
target_include_directories(vllm_ltx2_connector_gemm_probe SYSTEM PRIVATE
  ${CMAKE_SOURCE_DIR}/third_party)
target_compile_features(vllm_ltx2_connector_gemm_probe PRIVATE cxx_std_20)
vllm_cpp_set_warnings(vllm_ltx2_connector_gemm_probe)

# ── BIT-ROT GUARD: the vt CPU per-element dispatch survey probe ──────────────
# `tools/bench/vt_cpu_elem_survey_probe.cpp` is the artifact behind the ranking
# in `.agents/specs/vt-cpu-elem-survey.md`: one `--op <name>` runs ONE CPU
# kernel alone, at a shape a shipped model actually uses, so a `perf record`
# over it reads that kernel's `LoadF32` self percentage with no other vt work in
# the process. It reaches every kernel through the production `vt::` entry
# point, so a signature change must break the build rather than the ranking.
#
# CI COMPILES IT AND RUNS NOTHING, like its siblings above: one leg allocates
# gigabytes and spends tens of seconds of a core, and the recorded runs come
# from the direct g++ compile in the file's own header, because this devbox has
# no room for a CMake build tree.
add_executable(vllm_vt_cpu_elem_survey_probe tools/bench/vt_cpu_elem_survey_probe.cpp)
target_link_libraries(vllm_vt_cpu_elem_survey_probe PRIVATE vllm)
target_include_directories(vllm_vt_cpu_elem_survey_probe PRIVATE
  ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR}/include)
target_include_directories(vllm_vt_cpu_elem_survey_probe SYSTEM PRIVATE
  ${CMAKE_SOURCE_DIR}/third_party)
target_compile_features(vllm_vt_cpu_elem_survey_probe PRIVATE cxx_std_20)
vllm_cpp_set_warnings(vllm_vt_cpu_elem_survey_probe)

# ── BIT-ROT GUARD: the BPE encode-cost harness (#1365) ───────────────────────
# `tools/bench/bpe_encode_cost.cpp` is the artifact behind the timing figures in
# `.agents/specs/bpe-quadratic-merge.md`. It exists because three earlier
# harnesses for that row lived in prose only, so no reader could re-derive one
# figure and three rounds of certified constants each failed to reproduce.
#
# CI COMPILES IT AND RUNS NOTHING, and that is not an oversight. The row's own
# ruling is that a growth assertion over these figures is NOT viable on a shared
# runner, so there is nothing here for CI to decide; and one leg on the O(n^2)
# code costs tens of seconds of one core, which no test tier should spend. What
# CI does buy is that the file cannot rot behind a `Tokenizer::Encode` or
# `FromHfJson` signature change while still being the only artifact a reader can
# reproduce the row's numbers from — the #1246 failure, where the sole named
# harness for a measurement was compiled by nothing.
#
# An OBJECT library, like `vllm_music3_depth_stage_ab_*` above and unlike
# `vllm_music3_vocoder_conv_ab`: the recorded run recipe in the file's own header
# compiles the four tokenizer translation units directly with `g++` rather than
# linking `libvllm.a`, so a linked target here would guard nothing the compile
# does not and would only add an entry point nobody calls.
add_library(vllm_bpe_encode_cost OBJECT tools/bench/bpe_encode_cost.cpp)
target_include_directories(vllm_bpe_encode_cost PRIVATE
  ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR}/include)
target_include_directories(vllm_bpe_encode_cost SYSTEM PRIVATE
  ${CMAKE_SOURCE_DIR}/third_party)
target_compile_features(vllm_bpe_encode_cost PRIVATE cxx_std_20)
vllm_cpp_set_warnings(vllm_bpe_encode_cost)

# ── Packaged shared/static libvllm (M3.5 Task 3, library-first packaging) ─────
# The static `vllm` core lib above stays as-is (tests + examples link it and
# reach internal C++ symbols). Here we ALSO package a SHARED libvllm.so that
# exports ONLY the stable C ABI (include/vllm.h): a linker version script keeps
# every `vllm_*` symbol global and localizes every C++ internal, so an FFI /
# dlopen consumer (LocalAI via purego/cgo, any C host) sees a clean, ABI-stable
# surface — `nm -D libvllm.so` shows the C ABI and nothing else. The static
# `libvllm.a` is the existing `vllm` archive; both are installed.
include(GNUInstallDirs)

# A generated stub gives the SHARED target its (empty) own source and forces the
# CXX linker driver; the real content is the whole `vllm` archive, force-linked
# below via vllm's INTERFACE --whole-archive option.
set(_vllm_shared_stub "${CMAKE_CURRENT_BINARY_DIR}/vllm_shared_stub.cpp")
file(WRITE "${_vllm_shared_stub}"
  "// Generated by CMake: linker stub for the packaged shared libvllm.\n"
  "// The C ABI + engine come from the force-linked static `vllm` archive.\n")

add_library(vllm_shared SHARED "${_vllm_shared_stub}")
add_library(vllm::shared ALIAS vllm_shared)
set_target_properties(vllm_shared PROPERTIES
  OUTPUT_NAME vllm
  ARCHIVE_OUTPUT_NAME vllm_shared
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
# PUBLIC include so a consumer linking vllm_shared gets include/vllm.h ONLY (no
# internal C++ headers).
target_include_directories(vllm_shared PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
# Force-link the whole `vllm` archive (the C ABI + engine + the CPU-backend
# static registrar) and inherit its PUBLIC deps (CUDA::cudart, Threads, ...).
# On Windows the packaged shared target also needs the vendored BLAKE3 archive
# explicitly on its own link line; relying on the static archive's usage
# requirements is not sufficient once the C ABI DLL is assembled via
# /WHOLEARCHIVE.
target_link_libraries(vllm_shared PRIVATE vllm blake3_vendored)
# Export only the C ABI: `vllm_*` stays global, everything else is localized.
# UNLIKE the force-link guard above, `UNIX AND NOT APPLE` is CORRECT here: a
# linker version script is a GNU-ld/ELF feature with no ld64 spelling (ld64 uses
# -exported_symbols_list). The macOS dylib therefore exports its default symbol
# set; tests/CMakeLists.txt gates the matching `capi_shared_exports_only_abi`
# check on the same condition for the same reason.
if(UNIX AND NOT APPLE)
  target_link_options(vllm_shared PRIVATE
    "LINKER:--version-script,${CMAKE_CURRENT_SOURCE_DIR}/cmake/vllm_export.map")
endif()

# Sanitizer-only test linkage. Every normal test force-links the complete static
# `vllm` archive so its registrars are retained. With ASan+UBSan that makes each
# executable hundreds of MiB and duplicates the same instrumented engine across
# the whole suite; GitHub's hosted runner ran out of disk while linking. Keep one
# unversioned internal shared image of the fully instrumented archive instead.
# This is deliberately separate from packaged `vllm_shared`: that library's
# version script must continue exporting only the stable C ABI, while the tests
# need the C++ internals. Production/default builds never create this target.
if(NOT VLLM_CPP_SANITIZE STREQUAL "OFF")
  add_library(vllm_sanitize_test_shared SHARED "${_vllm_shared_stub}")
  add_library(vllm::sanitize_test_shared ALIAS vllm_sanitize_test_shared)
  target_link_libraries(vllm_sanitize_test_shared PRIVATE vllm)
  target_include_directories(vllm_sanitize_test_shared INTERFACE
    "$<TARGET_PROPERTY:vllm,INTERFACE_INCLUDE_DIRECTORIES>")
  target_include_directories(vllm_sanitize_test_shared SYSTEM INTERFACE
    "$<TARGET_PROPERTY:vllm,INTERFACE_SYSTEM_INCLUDE_DIRECTORIES>")
  target_compile_definitions(vllm_sanitize_test_shared INTERFACE
    "$<TARGET_PROPERTY:vllm,INTERFACE_COMPILE_DEFINITIONS>")
  # ...and the LINK interface, forwarded by the same rule as the three lines
  # above. Those forward what a consumer needs to COMPILE against `vllm`. This
  # one forwards what it needs to LINK, and its absence is issue #1531: the
  # `CPPHTTPLIB_OPENSSL_SUPPORT` that arrives through the line above makes every
  # test including the vendored httplib header emit OpenSSL references, while
  # the only libssl anywhere near the link line was a DT_NEEDED entry of THIS
  # shared image. GNU ld refuses to resolve through one -- "DSO missing from
  # command line" -- so eight targets failed to link and both sanitizer lanes
  # reported `fail` without running a case, on every branch based after
  # a50c57d69.
  #
  # Forward the PROPERTY rather than naming OpenSSL: the answer then lives in
  # exactly one place, and the next PUBLIC LINK LIBRARY `vllm` acquires cannot be
  # forgotten here. A per-target `target_link_libraries(test_x OpenSSL::SSL)`
  # would repair the eight and leave the ninth to be discovered by a red lane.
  #
  # These four forwards are not the complete set, and saying so is the point of
  # the paragraph above. `INTERFACE_COMPILE_OPTIONS`, `INTERFACE_COMPILE_FEATURES`
  # and `INTERFACE_LINK_DIRECTORIES` are still unforwarded. All three are EMPTY on
  # `vllm` today, so nothing is broken; the first one that gains a value owes a
  # fifth line here, and will fail exactly as silently as this one did.
  #
  # It must stay INTERFACE with the property genex, and must never become
  # `PUBLIC vllm`: `vllm` carries the `--whole-archive` INTERFACE link OPTION
  # set above, and not propagating that is the entire reason this target exists.
  target_link_libraries(vllm_sanitize_test_shared INTERFACE
    "$<TARGET_PROPERTY:vllm,INTERFACE_LINK_LIBRARIES>")
endif()

# make install => a consumable libvllm.so + libvllm.a + vllm.h (llama.cpp-style).
install(TARGETS vllm vllm_shared
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES include/vllm.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(VLLM_CPP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(VLLM_CPP_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# `vllm-cli`'s own reachability gate (#1135) runs the BINARY, so it needs the
# binary built first. The dependency is declared HERE rather than in
# tests/CMakeLists.txt because `add_subdirectory(tests)` runs before
# `add_subdirectory(examples)`, so the `vllm-cli` target does not exist yet at
# that point. The `$<TARGET_FILE:vllm-cli>` compile definition in tests/ is fine
# there, because a generator expression is evaluated after every directory has
# been processed. With examples off there is no `vllm-cli` target, no
# `VLLM_CLI_BINARY` define, and the test EXITS 77 so CTest reports it Skipped
# (see tests/vllm/entrypoints/test_cli_offload_config.cpp); this guard is the only
# thing that has to know about the ordering.
if(TARGET test_cli_offload_config AND TARGET vllm-cli)
  add_dependencies(test_cli_offload_config vllm-cli)
endif()
# KV-FP8 W7 (#2619): the same ordering, for the same reason, for the benchmark
# harness's own flag gate. Without this the gate can run against a stale or
# absent `vllm-bench` and report a verdict about a binary it did not build.
if(TARGET test_bench_kv_cache_dtype AND TARGET vllm-bench)
  add_dependencies(test_bench_kv_cache_dtype vllm-bench)
endif()
# #2759: the same ordering, for the same reason, for the EOS / chat-template
# flag gate.
if(TARGET test_bench_eos_chat_template AND TARGET vllm-bench)
  add_dependencies(test_bench_eos_chat_template vllm-bench)
endif()

# Render the compiled-arch manifest (issue #1357). Last, so it observes whatever
# the FA2 gencode site above did or did not set. Generated per build into the
# build tree, so it is not a shared file any pull request has to edit.
configure_file(include/vllm/platforms/cuda_compiled_archs.h.in
  include/vllm/platforms/cuda_compiled_archs.h @ONLY)
message(STATUS
  "CUDA FA2 compiled-arch manifest: [${VLLM_CPP_CUDA_FA2_COMPILED_ARCHS}]")
