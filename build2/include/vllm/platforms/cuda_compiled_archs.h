// GENERATED — do not edit, and do not add a hand-written architecture list here.
//
// CMake renders this from `VT_FA2_ARCHS`, which is the SAME variable
// `vt_cuda_set_source_gencode` turns into the `-gencode` options nvcc receives
// for the FlashAttention-2 translation units (CMakeLists.txt:499 computes it,
// CMakeLists.txt passes it to both). The manifest is therefore DERIVED from the
// flag computation rather than tracking it, which is the entire point: a second,
// hand-maintained list is `CUDA_SUPPORTED_ARCHS`, and a list that goes stale is
// the defect issue #1357 exists to remove.
//
// Empty means FlashAttention-2 was not compiled for any requested architecture —
// `vt_cuda_report_feature` prints it DISABLED — and the predicate must answer no
// rather than fall through to a default yes.
//
// Form: CMake CUDA_ARCHITECTURES tokens, comma separated ("80,86,121a"), as
// produced by vt_cuda_archs_denormalize, which strips `+PTX`. That stripping is
// load-bearing: this manifest names targets with real SASS, and PTX is not a
// yes. See cuda_arch_manifest.h.
#ifndef VLLM_PLATFORMS_CUDA_COMPILED_ARCHS_H_
#define VLLM_PLATFORMS_CUDA_COMPILED_ARCHS_H_

#define VLLM_CPP_CUDA_FA2_COMPILED_ARCHS ""

#endif  // VLLM_PLATFORMS_CUDA_COMPILED_ARCHS_H_
