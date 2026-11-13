// vllm.cpp ORIGINAL — GGUF keep-quantized residency policy. See
// gguf_keep_quant.h for the upstream (llama.cpp @ 237ad9b96) anchors and the
// totality contract.
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"

#include <cstdlib>
#include <cstring>

#include "vllm/config/weight_residency.h"
#include "vllm/model_executor/device_placement.h"
#include "vllm/platforms/interface.h"
#include "vt/ops.h"
#include "vt/quant.h"

namespace vllm {
namespace {

// -1 means "this role never keeps its blocks"; otherwise the K (in-features)
// dimension whose block alignment decides eligibility.
//
// NOTE the deliberate absence of a `default:` label: a new GgufTensorRole that
// forgets to state its residency is a -Werror=switch BUILD failure, which is
// the compile-time half of the totality proof (the unit tests are the other).
int64_t KeepQuantKDim(GgufTensorRole role, const std::vector<int64_t>& shape) {
  switch (role) {
    case GgufTensorRole::kMatmulWeight:
      // [out, in]; ggml's src0 orientation, our MatmulBT [N, K] with nk=true.
      return shape.size() == 2 ? shape[1] : -1;
    case GgufTensorRole::kStackedExpertWeight:
      // [E, out, in]; each expert slice is whole rows of the same K.
      return shape.size() == 3 ? shape[2] : -1;
    case GgufTensorRole::kEmbeddingTable:
      // [vocab, hidden]; a GATHER, not a GEMM. The blocks stay and
      // `vt::Embedding` decodes ONE ROW per gathered id (port of llama.cpp's
      // `ggml_compute_forward_get_rows_q`, ggml/src/ggml-cpu/ops.cpp:4850
      // @ b10451). K is the row length, and the same whole-blocks rule applies:
      // a ragged row has no stride. This returned -1 until the dequantizing
      // gather existed, and the -1 was CORRECT until then — without the op a
      // kept table is a tensor nothing can read.
      return shape.size() == 2 ? shape[1] : -1;
    case GgufTensorRole::kTransformedWeight:
    case GgufTensorRole::kConvWeight:
    case GgufTensorRole::kVector:
      return -1;
  }
  return -1;  // unreachable for a valid enumerator (see -Wswitch above)
}

// L6 keep-f16 eligibility by role. Same verbatim-bytes roles as keep-quant PLUS
// the embedding table: an F16 gather table stays F16 (the gather widens f16 to
// f32 exactly like it did bf16), which is what lets a tied token_embd/lm_head be
// ONE resident f16 vocab matrix instead of a bf16 re-expansion. A value/layout
// rewrite (kTransformedWeight), a conv filter or a 1-D vector never keeps native
// bytes, exactly as for keep-quant. Returns the K (in-features) dim, or -1.
int64_t KeepF16KDim(GgufTensorRole role, const std::vector<int64_t>& shape) {
  switch (role) {
    case GgufTensorRole::kMatmulWeight:
    case GgufTensorRole::kEmbeddingTable:
      return shape.size() == 2 ? shape[1] : -1;
    case GgufTensorRole::kStackedExpertWeight:
      return shape.size() == 3 ? shape[2] : -1;
    case GgufTensorRole::kTransformedWeight:
    case GgufTensorRole::kConvWeight:
    case GgufTensorRole::kVector:
      return -1;
  }
  return -1;  // unreachable for a valid enumerator (see -Wswitch above)
}

bool EnvOn(const char* name) {
  const char* v = std::getenv(name);
  if (v == nullptr) return false;
  return !(std::strcmp(v, "") == 0 || std::strcmp(v, "0") == 0 ||
           std::strcmp(v, "false") == 0 || std::strcmp(v, "off") == 0);
}

// Tri-state: unset -> `fallback`; "0"/"false"/"off"/"" -> false; else true.
bool EnvOnOr(const char* name, bool fallback) {
  return std::getenv(name) == nullptr ? fallback : EnvOn(name);
}

}  // namespace

bool GgufQuantComputeAvailable(vt::DeviceType dev) {
  return vt::OpRegistered(vt::OpId::kMatmulBTQuant, dev);
}

bool GgufNvfp4ComputeAvailable(vt::DeviceType dev) {
  return vt::OpRegistered(vt::OpId::kMatmulNvfp4, dev);
}

// #2406. See the header for why `host_repack_active` is a parameter and not a
// call. The device term is the whole fix: everything else here is the
// pre-existing expression, so on `dev == kCPU` this is character-for-character
// the decision the loader made before, and a CPU load is unchanged.
bool QuantRepackForDevice(bool keep_quant, bool cpu_ref,
                          bool host_repack_active, vt::DeviceType dev) {
  return keep_quant && !cpu_ref && host_repack_active &&
         dev == vt::DeviceType::kCPU;
}

// See the header. The ORDER of the terms is the contract: an explicit knob is
// answered before the device is consulted, so the same-binary A/B still reaches
// a staging device.
bool GgufPrefaultForDevice(vt::DeviceType dev) {
  if (!ResolveGgufPrefault()) return false;
  if (GgufPrefaultIsExplicit()) return true;
  if (dev == vt::DeviceType::kCPU) return true;
  // A device with no registered platform cannot be asked, and this function is
  // not the place to refuse a load. Answering ON leaves such a caller with
  // exactly the behaviour it had before this term existed.
  if (!vllm::platforms::HasPlatform(dev)) return true;
  return vllm::platforms::GetPlatform(dev).host_memory_is_device_addressable();
}

const char* Name(GgufTensorRole role) {
  switch (role) {
    case GgufTensorRole::kMatmulWeight: return "matmul_weight";
    case GgufTensorRole::kStackedExpertWeight: return "stacked_expert_weight";
    case GgufTensorRole::kTransformedWeight: return "transformed_weight";
    case GgufTensorRole::kEmbeddingTable: return "embedding_table";
    case GgufTensorRole::kConvWeight: return "conv_weight";
    case GgufTensorRole::kVector: return "vector";
  }
  return "?";
}

const char* Name(GgufResidency residency) {
  switch (residency) {
    case GgufResidency::kExpandBf16: return "expand_bf16";
    case GgufResidency::kKeepQuant: return "keep_quant";
    case GgufResidency::kKeepF16: return "keep_f16";
    case GgufResidency::kNvfp4Fp4: return "nvfp4_fp4";
    case GgufResidency::kBfp8Device: return "bfp8_device";
  }
  return "?";
}

// ggml type id 1 is F16 (IEEE half); see gguf_dequant.cpp case 1.
bool KeepF16DType(uint32_t ggml_type) { return ggml_type == 1; }

bool GgufResidencyKeepsBlockWeights(GgufResidency residency) {
  return residency == GgufResidency::kKeepQuant ||
         residency == GgufResidency::kKeepF16;
}

bool GgufResidencyExpandsBf16(GgufResidency residency) {
  return residency == GgufResidency::kExpandBf16 ||
         residency == GgufResidency::kBfp8Device;
}

// ggml type id 40 is the NVFP4 fork extension; see gguf_dequant.cpp case 40.
bool KeepNvfp4DType(uint32_t ggml_type) { return ggml_type == 40; }

// Device-side keep-quant capability (review sweep on #523): the master
// boolean `GgufQuantComputeAvailable()` only says the OP is registered; a
// device's kernel set can be narrower than the CPU admission list, and on a
// discrete backend with no CPU fallback tier a format the device cannot
// execute must keep its pre-existing expand-bf16 residency -- flipping it to
// a keep-quant block throws at FORWARD time with the whole model resident.
// Per-device sets name what the registered kernels actually implement.
bool DeviceKeepQuantSupported(vt::DType dt, vt::DeviceType dev) {
  switch (dev) {
    case vt::DeviceType::kROCM:
      // rocm_grouped_gemm.hip implements Q8_0/IQ4_NL/Q4_K/Q5_K/Q6_K, while
      // rocm_quant_dot.hip adds the eight Q8_K-activation formats below on
      // both grouped and non-grouped arms. IQ4_XS is the last of the eight
      // (KERNEL-QUANT-CIQ-GEMM-ROCM-IQUANT, #1940 in pull request #3029). It
      // used to fall through to `false` here, and that is the host-RAM SIGSEGV
      // a real IQ4_XS checkpoint hit on ROCm: the loader chose exactly the
      // expand-bf16 residency this comment block warns about.
      //
      // IQ4_NL is admitted on BOTH arms or neither. It is the only entry here
      // whose activation encoding is Q8_0 rather than Q8_K, served by DotIQ4_NL
      // through IQ4NLGemmK (single) and GroupedIQ4NLK (expert towers).
      //
      // Q4_0/Q5_0/IQ2_XS/IQ3_S/MXFP4 stay on the named expand-or-refuse path.
      return dt == vt::DType::kQ8_0 || dt == vt::DType::kIQ4_NL ||
             dt == vt::DType::kQ4_K ||
             dt == vt::DType::kQ5_K || dt == vt::DType::kQ6_K ||
             dt == vt::DType::kIQ4_XS ||
             dt == vt::DType::kIQ2_XXS || dt == vt::DType::kIQ3_XXS ||
             dt == vt::DType::kQ2_K || dt == vt::DType::kQ3_K ||
             dt == vt::DType::kIQ2_S || dt == vt::DType::kIQ1_S ||
             dt == vt::DType::kIQ1_XXXS;
    case vt::DeviceType::kTENSTORRENT:
      // KEEPQUANT W3: the P150 is discrete with no CPU fallback tier, so this
      // arm admits exactly what src/vt/tenstorrent/tenstorrent_ops.cpp has a
      // registered decode for — MatmulBTQuantKernel decodes Q4_K/Q5_K/Q6_K/
      // Q8_0 through the bit-exact DecodeKeepQuantBlocksF32 chains, staged as
      // the per-weight resident i32 word shadow. The vehicle's own histogram
      // (the q4km artifact: token_embd Q6_K, attn_qkv/ssm_out Q5_K,
      // ssm_alpha/ssm_beta Q8_0) is what pulled Q5_K/Q6_K/Q8_0 from W4 into
      // W3 — kernels and predicate widened IN THE SAME CHANGE. kQ4_0 has no
      // TT arm at all; kQ2_K stayed owed until tenstorrent-gsq-keepquant
      // wave 4 admitted it (enc_sel 11; Q3_K joined in
      // QUANT-GGUF-IQ-TENSTORRENT wave 3); admitting an encoding
      // without its kernel throws at first forward with the model resident,
      // the exact failure this predicate exists to prevent.
      // QUANT-GGUF-IQ-TENSTORRENT wave 1: kIQ3_XXS joins the set — its
      // on-core decode is the int8-dot kernel's enc_sel 4
      // (kq_vec_dot_iq3_xxs_q8_K, keepquant_kernel_code.h), staged as the
      // same resident i32 word shadow (32 words = 98 B zero-padded to
      // 128 B), and dispatched on the DEFAULT path (no env gate — the
      // grouped arm has no IQ3_XXS decode to fall through to). The APEX
      // I-Nano vehicle's 164 IQ3_XXS tensors are the artifact this admits.
      // tests/vllm/test_gguf_keep_quant.cpp pins the set; widening the arm
      // without widening the kernel reds it.
      // QUANT-GGUF-IQ-TENSTORRENT wave 2: kIQ2_XXS (enc_sel 5) and kIQ2_S
      // (enc_sel 6) join the set the same way — on-core decodes
      // kq_vec_dot_iq2_xxs_q8_K / kq_vec_dot_iq2_s_q8_K
      // (keepquant_kernel_code.h), staged as the same resident i32 word
      // shadow (32 words = 66 B / 82 B zero-padded to 128 B), dispatched on
      // the DEFAULT path. The APEX I-Nano vehicle's 44 IQ2_XXS + 89 IQ2_S
      // tensors are the artifacts this admits.
      // QUANT-GGUF-IQ-TENSTORRENT wave 3: kQ3_K (enc_sel 7) closes the
      // census — kq_vec_dot_q3_k_q8_K (keepquant_kernel_code.h), 32 words
      // = 110 B zero-padded to 128 B, dispatched on the DEFAULT path. The
      // vehicle's 78 Q3_K tensors are the artifact this admits, and no
      // named missing arm remains in the census.
      // tenstorrent-gsq-keepquant wave 1: kIQ3_S (enc_sel 8) joins the set —
      // its on-core decode is kq_vec_dot_iq3_s_q8_K
      // (keepquant_kernel_code.h), staged as the same resident i32 word
      // shadow (32 words = 110 B zero-padded to 128 B), dispatched on the
      // DEFAULT path. The GSQ-RCO Qwen3.8-27B vehicle's 97 IQ3_S tensors
      // (ffn + attn, the largest census gap) are the artifact this admits.
      // tenstorrent-gsq-keepquant wave 2: kIQ4_XS (enc_sel 9) joins the set
      // the same way — its on-core decode is kq_vec_dot_iq4_xs_q8_K
      // (keepquant_kernel_code.h), staged as a resident i32 word shadow
      // (48 words = the 136-B block zero-padded to the 192-B word grid),
      // dispatched on
      // the DEFAULT path. The GSQ-RCO Qwen3.8-27B vehicle's 33 IQ4_XS tensors
      // (ssm_out + attn) are the artifact this admits: ssm_out residency is
      // what keeps the state path on-device.
      // tenstorrent-gsq-keepquant wave 3: kIQ2_XS (enc_sel 10) joins the set
      // the same way — its on-core decode is kq_vec_dot_iq2_xs_q8_K
      // (keepquant_kernel_code.h), staged as a resident i32 word shadow
      // (32 words = the 74-B block zero-padded to the 128-B word grid, the
      // IQ2_XXS/IQ2_S footprint), dispatched on the DEFAULT path. The
      // GSQ-RCO Qwen3.8-27B vehicle's 32 IQ2_XS tensors (ffn) are the
      // artifact this admits.
      // tenstorrent-gsq-keepquant wave 4: kQ2_K (enc_sel 11) joins the set
      // tenstorrent-gsq-keepquant wave 5: kIQ1_S (enc_sel 12) and kIQ1_M
      // (enc_sel 13) join the set — on-core decodes
      // kq_vec_dot_iq1_s_q8_K / kq_vec_dot_iq1_m_q8_K
      // (keepquant_kernel_code.h), staged as resident i32 word shadows
      // (16 words = the 50-B / 56-B blocks zero-padded to the 64-B word
      // grid), dispatched on the DEFAULT path. The GSQ-RCO Qwen3.8-27B
      // vehicle's 4 + 4 IQ1 tensors (ffn tail) are the artifacts this
      // admits; this closes the file's block-quant census.
      // TT-KEEPQUANT-Q8K-WEIGHTS: kQ8_K joins the TT GEMM set — the
      // UD-Q8_K_XL checkpoint's weights arm. It stages at 80 padded words
      // per 292-B block_q8_K and serves the DEQUANT-COMPOSITE arm (the E=1
      // grouped chain; NO int8-dot kernel — none exists for Q8_K weights,
      // and the fused INT8-dot arm is a separate PERF row). The
      // DeviceKeepQuantSupported contract holds: this predicate widens only
      // in the same change as the serve it names.
      return dt == vt::DType::kQ4_K || dt == vt::DType::kQ5_K ||
             dt == vt::DType::kQ6_K || dt == vt::DType::kQ8_0 ||
             dt == vt::DType::kQ8_K ||
             dt == vt::DType::kIQ3_XXS || dt == vt::DType::kIQ2_XXS ||
             dt == vt::DType::kIQ2_S || dt == vt::DType::kQ3_K ||
             dt == vt::DType::kIQ3_S || dt == vt::DType::kIQ4_XS ||
             dt == vt::DType::kIQ2_XS || dt == vt::DType::kQ2_K ||
             dt == vt::DType::kIQ1_S || dt == vt::DType::kIQ1_M;
    default:
      // CUDA falls back to the CPU kernel for anything it lacks
      // (cuda_quant_dot.cu:1841-1846); the CPU list IS the CPU capability.
      // TT-KEEPQUANT-Q8K-WEIGHTS scopes the kQ8_K weights arm to the devices
      // that name its composite serve: CPU (the to_float fallthrough IS the
      // oracle) and TENSTORRENT (the composite chain). CUDA keeps its
      // pre-arm behavior — its provider has no Q8_K-weight serve of its own,
      // so widening here would be an admitted-but-unimplemented encoding,
      // the exact failure the predicate exists to prevent.
      return dt != vt::DType::kQ8_K || dev == vt::DeviceType::kCPU ||
             dev == vt::DeviceType::kTENSTORRENT;
  }
}

// ROCm retention additionally requires the accepting loader's model dtype.
// This predicate also prevents an unwired loader admitting raw F16 model values.
bool RequiresF16ValueDType(vt::DeviceType dev) {
  return dev == vt::DeviceType::kROCM;
}

bool SupportedF16ValueDType(std::optional<vt::DType> dtype) {
  return dtype == vt::DType::kBF16 || dtype == vt::DType::kF32;
}

// The GATHER's admission rule; see the header. A block dtype with a row decoder
// is enough — `HasQuantDotKernel` is not consulted, because a gather table is
// never the `x` side of a vec_dot.
bool KeepQuantGatherDType(uint32_t ggml_type, vt::DType* out) {
  vt::DType dt = vt::DType::kF32;
  if (!vt::BlockDTypeFromGgmlTypeId(ggml_type, &dt)) return false;
  if (vt::cpu::BlockToFloat(dt) == nullptr) return false;
  if (out != nullptr) *out = dt;
  return true;
}

// Device gate for the gather arm, asked of the OP/PROVIDER TABLE and not of a
// hand-kept device list.
//
// `vt::Embedding` routes a block-quantized table to `OpId::kEmbeddingQuant`
// (vt/ops.cpp), so "can this device decode a block row" is exactly "is that op
// registered here". That is the same shape `GgufQuantComputeAvailable()` above
// uses for the GEMM arm, and it is the shape this function was owed: naming a
// device here is what `scripts/check-device-leakage.py` refuses, because the
// GGUF loader is the device-agnostic layer and a per-device set written into it
// drifts from the kernels it claims to describe.
//
// Registered today by kCPU (cpu_ops.cpp, through `BlockToFloat`) and by the
// CUDA backend (cuda_ops.cu, through cuda_quant_dequant.cuh) -- named in prose
// rather than as the enumerator on purpose, because the leakage checker greps
// the token in comments too, and rightly so: a prose mention is how the next
// hand-kept device list starts. METAL, VULKAN and
// TENSTORRENT register only `kEmbedding`, whose kernels each assert a float
// table by name, so they answer false here and keep their pre-existing
// expand-bf16 residency -- and their gather arms are owed.
//
// What this unlocks is not one op. On a device that cannot gather quantized,
// the Qwen3.8-Flash-Next n-gram table expands from 26.822 GiB of IQ4_NL to
// 95.368 GiB of bf16, against ~119.6 GiB of usable memory on the one box that
// fits the model at all -- so the device-resident quantized table is what makes
// a GPU arm possible, and llama.cpp's own n-gram path does NOT have this shape
// (#27742 pins that table to the CPU by tensor class).
bool DeviceQuantGatherSupported(vt::DeviceType dev) {
  return vt::OpRegistered(vt::OpId::kEmbeddingQuant, dev);
}

bool KeepQuantDType(uint32_t ggml_type, vt::DType* out) {
  vt::DType dt = vt::DType::kF32;
  if (!vt::BlockDTypeFromGgmlTypeId(ggml_type, &dt)) return false;
  // TT-KEEPQUANT-Q8K-WEIGHTS: kQ8_K joins the GEMM set. It has no weight-side
  // vec_dot (the selector at cpu_quant_dot.cpp returns nullptr), so it serves
  // the DEQUANT-COMPOSITE arm — the tree-standard for Q8_K weights, on the
  // CPU oracle (cpu_quant_gemm.cpp's ComposeChunk fallthrough) and on the
  // device (the TT composite chain) alike. Every other encoding still needs
  // a real vec_dot; the kQ8_K exception is named, not silent.
  if (dt != vt::DType::kQ8_K && !vt::cpu::HasQuantDotKernel(dt)) return false;
  if (out != nullptr) *out = dt;
  return true;
}

GgufResidency RouteGgufTensor(bool keep_quant, bool keep_f16, bool nvfp4_fp4,
                              bool cpu_ref, GgufTensorRole role,
                              uint32_t ggml_type,
                              const std::vector<int64_t>& shape,
                              vt::DeviceType dev,
                              std::optional<vt::DType> weight_value_dtype,
                              TtWeightResidency tt_residency) {
  // The oracle switch wins over everything (spec gate 2).
  if (cpu_ref) return GgufResidency::kExpandBf16;

  // 0. Native fp4 residency for NVFP4. Checked first because it is the ONLY
  // keep outcome type 40 has: it is not a vt block dtype (so it can never be
  // kKeepQuant) and it is not F16 (so never kKeepF16). The eligible roles are
  // exactly the verbatim-bytes ones keep-quant uses, and the alignment rule is
  // the ggml block, 64 elements — a ragged K cannot be repacked block-wise.
  //
  // The GATHER role is excluded EXPLICITLY rather than by sharing
  // `KeepQuantKDim`'s answer. It shared it until W6a, when that function
  // started returning a real K for the embedding table, and the fp4 arm would
  // then have claimed a gather table it cannot serve: `kNvfp4Fp4` produces an
  // `Nvfp4Weight` operand PAIR for `vt::MatmulNvfp4`, not a block-typed tensor,
  // and no gather can read one. The exclusion is by ROLE and not by capability,
  // so a future NVFP4 gather has to say so here.
  if (nvfp4_fp4 && KeepNvfp4DType(ggml_type) &&
      role != GgufTensorRole::kEmbeddingTable) {
    const int64_t k = KeepQuantKDim(role, shape);
    if (k > 0 && k % 64 == 0) return GgufResidency::kNvfp4Fp4;
  }

  // 1. Keep-quant blocks (a block encoding in a verbatim GEMM/expert/gather
  // role). The GEMM roles and the GATHER role share the block-alignment rule and
  // the device gate but NOT the encoding rule: a GEMM weight needs a `vec_dot`,
  // a gather table needs only a row decoder, so the two ask different questions.
  if (keep_quant) {
    const int64_t k = KeepQuantKDim(role, shape);
    const bool gather = role == GgufTensorRole::kEmbeddingTable;
    vt::DType dt = vt::DType::kF32;
    // ggml_row_size's precondition: a row is a whole number of blocks. A weight
    // whose K is ragged cannot be dotted OR decoded block-wise, so it expands.
    // The device gate (review #523): a format the RUNNING device cannot execute
    // keeps its pre-existing expand-bf16 residency instead of flipping to a
    // keep-quant block that throws at forward time on a card with no CPU
    // fallback tier.
    const bool encoding_ok = gather ? KeepQuantGatherDType(ggml_type, &dt)
                                    : KeepQuantDType(ggml_type, &dt);
    if (k > 0 && encoding_ok && k % vt::BlockElems(dt) == 0 &&
        (gather ? DeviceQuantGatherSupported(dev)
                : DeviceKeepQuantSupported(dt, dev))) {
      return GgufResidency::kKeepQuant;
    }
  }

  // 2. Keep-f16 (an F16 file weight in a verbatim role, incl. the gather table).
  // Independent of keep_quant: F16 is not a block encoding, so a weight is never
  // eligible for both. No block-alignment constraint — f16 is per-element.
  const bool f16_consumer = !RequiresF16ValueDType(dev) ||
      (SupportedF16ValueDType(weight_value_dtype) &&
       (role == GgufTensorRole::kMatmulWeight ||
        role == GgufTensorRole::kEmbeddingTable));
  if (keep_f16 && f16_consumer && KeepF16DType(ggml_type) &&
      KeepF16KDim(role, shape) > 0) {
    return GgufResidency::kKeepF16;
  }

  // 3. Tenstorrent BFP weight residency (spec
  // .agents/specs/tenstorrent-bfp-weight-residency.md). Admits exactly the
  // population that would otherwise fall through to kExpandBf16: an
  // UNQUANTIZED F32/BF16 ggml weight in a verbatim GEMM/expert role. Every
  // other role refuses by falling through — kTransformedWeight bytes are
  // rewritten, the gather table is not a GEMM operand, vectors/conv filters
  // have no BFP consumer — and a keep-quant/f16-eligible encoding never
  // reaches this point (those returned above). The alignment rule is the ttnn
  // TILE, 32 elements on BOTH dims (K via `KeepQuantKDim`, N the remaining
  // dim), because the device conversion packs BFP tiles and the staging seam
  // refuses a ragged shape by name — the route must agree with the seam. A
  // ragged shape keeps the bf16 residency. The
  // loader expands kBfp8Device exactly like kExpandBf16 — admission here
  // decides only whether the Tenstorrent staging seam converts the staged
  // weight to device-resident BFLOAT8_B.
  if (tt_residency == TtWeightResidency::kBfp8 && shape.size() >= 2 &&
      (role == GgufTensorRole::kMatmulWeight ||
       role == GgufTensorRole::kStackedExpertWeight) &&
      (ggml_type == 0 /* F32 */ || ggml_type == 30 /* BF16 */) &&
      KeepQuantKDim(role, shape) > 0 && KeepQuantKDim(role, shape) % 32 == 0 &&
      shape[shape.size() - 2] % 32 == 0) {
    return GgufResidency::kBfp8Device;
  }

  return GgufResidency::kExpandBf16;
}

GgufLoadPolicy GgufLoadPolicy::FromEnv(
    vt::DeviceType dev, std::optional<vt::DType> model_dtype) {
  GgufLoadPolicy p;
  // FIRST, because every device-dependent flag below reads it. It is the
  // ENGINE's resolved device, not `CurrentPlatform()`: see the field comment.
  p.device = dev;
  if (RequiresF16ValueDType(dev) && model_dtype) {
    VT_CHECK(SupportedF16ValueDType(model_dtype),
             "GGUF retained F16 weights require resolved BF16 or F32 model values");
    p.weight_value_dtype = model_dtype;
  }
  p.cpu_ref = EnvOn("VT_CPU_REF");
  // CIQ G4 flipped this default: keep-quant is ON wherever the running device
  // can execute the quantized GEMM. VT_GGUF_KEEP_QUANT is the two-way
  // override that survives the flip (=0 is the opt-out the spec promised).
  p.keep_quant = EnvOnOr("VT_GGUF_KEEP_QUANT", GgufQuantComputeAvailable(dev));
  // The orientation win rides the same availability condition, and the oracle
  // switch turns it off with everything else so VT_CPU_REF=1 reproduces the
  // historical load byte for byte.
  p.expand_nk = p.keep_quant && !p.cpu_ref;
  // L6 keep-f16 — now DEFAULT ON (L7, 2026-07-23). Keeps an F16 file weight
  // resident as F16 (mmap-borrowed) and computes on it natively, instead of the
  // load-time expansion to a bf16 anonymous buffer. Mirrors llama.cpp, which keeps
  // f16 weights resident and runs `ggml_vec_dot_f16` on them.
  //
  // L6 shipped this OPT-IN because it measured RSS-NEUTRAL (3.884 -> 3.832 GiB)
  // and prefill-regressive. L7's profile found WHY, and both objections are now
  // removed: (1) the "neutrality" was a q8_0 double-count — on the mmap arm the
  // repack COPY leaves the source q8_0 blocks resident in the still-alive mapping
  // (kept alive by the f16 borrows), so the q8_0 mass counted twice. Releasing
  // the repack source (OwnGgufQuantBlocks, port of llama.cpp unmap_fragment) drops
  // that 1.0 GiB, so keep-f16 now measures 3.834 -> 2.832 GiB = 1.01x llama.cpp
  // (2.798), a real weight-residency win, NOT neutral. (2) the prefill regression
  // was borrowed-weight first-touch faults landing in the timed prefill; the
  // load-time PrefaultBorrowedSpan (port of llama.cpp's mmap prefetch) faults them
  // off the timed path, restoring prefill (was 0.72x behind). Net: RSS parity,
  // greedy tokens byte-identical (native-f16 compute, md5 d235db1... unchanged).
  //
  // THE llama.cpp DENOMINATORS ABOVE ARE SUPERSEDED, and this is the one place
  // the contamination reaches a shipped DEFAULT rather than a document. They were
  // measured against 237ad9b96, our own local-only fork, 65 of this project's
  // performance commits past upstream tag b9827, six of them in ggml/src/ggml-cpu/
  // (fused gated_delta_net and a discriminated ssm_conv, both DEFAULT-ON, neither
  // of which stock upstream carries at that point). The CPU floor arm ran a qwen35
  // model whose CPU graph reaches exactly those ops. The oracle is now stock
  // b10451, and the re-take is owed under #1003.
  //
  // TWO CORRECTIONS TO WHAT THIS COMMENT USED TO SAY, both from the #1003 sweep:
  //
  // (a) THE PREFILL FIGURES HERE WERE UNSOURCED. This comment read "~205 t/s =
  //     1.16x AHEAD of pp128 176.6". NO recorded run produces either operand:
  //     `git grep -F 176.6` finds it in four other files, none of them a
  //     llama.cpp prefill number: a MoE microbenchmark mean in MICROSECONDS
  //     (benchmark-record.md), a 176.69 in an ours-vs-vLLM ledger row
  //     (parity-ledger.md), two LTX golden floats, and #1003's own index row.
  //     Do not grep the regex `176.6`: the `.` matches any character and buries
  //     these in ~360 hits. No arm of ours recorded 205 t/s. The owning record
  //     (gguf-keep-quant-loader.md:595, benchmark-record.md:10722) says 204 t/s
  //     against pp128 173.2 = 1.18x. The numbers are NOT reconciled here, because
  //     picking one would assert an attribution nobody measured; #1003 owes a
  //     single re-measured pair. Do not quote any of these ratios.
  //
  // (b) THIS DEFAULT IS NOT INDEPENDENT OF THAT FLOOR. An earlier pass claimed it
  //     was, on the grounds that the L7 acceptance is a same-binary OURS-vs-OURS
  //     A/B. That reads one row of a three-row table. The A/B trades ~10% of
  //     prefill (224 -> 204 t/s, TTFT 571 -> 625 ms) and ~1.4% of decode for
  //     1.05 GiB of peak RSS, and the recorded reason the PREFILL loss is
  //     acceptable is stated in llama.cpp's own terms: "comfortably above the
  //     competitor floor" (gguf-keep-quant-loader.md:595). That floor is the
  //     contaminated pp128 173.2. b10451 is 624 commits past b9827 and carries
  //     upstream's own fused_gdn, so the direction is NOT established: if a
  //     re-taken stock pp128 lands above 204 t/s, that clause fails and this
  //     default's only recorded justification for its prefill regression is gone.
  //     The RSS leg would still stand alone and may well suffice. Keep-f16 stays
  //     DEFAULT ON here because this is a record pass that measured nothing;
  //     revisiting it is QUANT-GGUF-KEEPQ-LOADER's decision and needs the re-take
  //     first. See .agents/specs/oracle-llamacpp-repin-stock.md, row 12.
  //
  // VT_GGUF_KEEP_F16=0 opts out. ROCm requires explicit model values; other
  // backends retain their expand_nk prerequisite. VT_CPU_REF always expands.
  const bool f16_available = RequiresF16ValueDType(dev)
      ? p.weight_value_dtype.has_value() && !p.cpu_ref : p.expand_nk;
  p.keep_f16 = EnvOnOr("VT_GGUF_KEEP_F16", f16_available) && f16_available;
  // `QUANT-GGUF-NVFP4` column C. Same shape as the keep-quant default: ON
  // wherever the running device can execute the NVFP4 GEMM (CUDA today; a CPU
  // build keeps expanding, which is correct but unquantized), with
  // VT_GGUF_NVFP4_FP4=0 the same-binary opt-out and the oracle switch forcing it
  // off so VT_CPU_REF=1 still reproduces the historical bf16 load byte for byte.
  p.nvfp4_fp4 = EnvOnOr("VT_GGUF_NVFP4_FP4", GgufNvfp4ComputeAvailable(dev)) &&
                !p.cpu_ref;
  // Both of vLLM's NVFP4 modes are mirrored; W4A4 is the default because it is
  // what the sibling compressed-tensors container of the same model runs, and
  // the GGUFs carry the `<stem>.input_scale` activation sidecars it needs.
  p.nvfp4_w4a4 = EnvOnOr("VT_GGUF_NVFP4_W4A4", true) && p.nvfp4_fp4;
  // BACKEND-TENSTORRENT weight residency — DEFAULT OFF, opt in with
  // VT_TT_WEIGHT_RESIDENCY=bfp8 (bfp4 is reserved and refused by name at the
  // staging seam), and only on a load the ENGINE resolved onto
  // kTENSTORRENT (the residency has no consumer on any other device, so
  // electing it there would change the load for nothing). Forced off by the
  // oracle switch like every other residency, so VT_CPU_REF=1 reproduces the
  // historical bf16 load byte for byte.
  p.tt_weight_residency =
      dev == vt::DeviceType::kTENSTORRENT && !p.cpu_ref
          ? ParseTtWeightResidency(std::getenv("VT_TT_WEIGHT_RESIDENCY"))
          : TtWeightResidency::kOff;
  // L5. Both ride the same availability condition as the residency they refine,
  // and both are forced off by the oracle switch, so VT_CPU_REF=1 keeps
  // reproducing the historical load byte for byte and allocation for allocation.
  // ENG-RESIDENCY-CONFIG (#1110): `VT_GGUF_MMAP` is now `--offload-config`'s
  // `vllm_cpp.mmap.enabled` as well, and ResolveGgufMmap holds the precedence —
  // env var > config > this availability default. It is the SOLE reader of the
  // variable, and it applies the same whole-value polarity `EnvOnOr` did, so an
  // environment-only run resolves byte-for-byte as before. `VT_CPU_REF` still wins
  // over both: the oracle switch is not a residency preference.
  p.mmap_residency = ResolveGgufMmap(p.keep_quant || p.keep_f16) && !p.cpu_ref;
  const bool share_available = p.expand_nk || p.keep_f16;
  p.share_tied_head = EnvOnOr("VT_GGUF_SHARE_TIED_HEAD", share_available) && share_available;
  // GDN split-projection orientation. Rides expand_nk (so VT_CPU_REF=1
  // reproduces the historical transpose); VT_GGUF_GDN_NK=0 is the narrow
  // same-binary A/B that reverts only the GDN projections to [K, N].
  p.gdn_expand_nk = EnvOnOr("VT_GGUF_GDN_NK", p.expand_nk) && p.expand_nk;
  // CIQ G7 repack-at-load. Rides keep_quant AND the i8mm probe (which itself
  // honors VT_CPU_QUANT_REPACK=0), and is forced off by the oracle switch so
  // VT_CPU_REF=1 reproduces the historical load. No separate default env read:
  // QuantRepackActive() is the single source of the on/off decision, so the
  // loader and the kernel can never disagree about whether repack is live.
  //
  // AND ON THE RESOLVED DEVICE (#2406), which it did not read until now. This
  // was the ONE device-dependent flag in this function still deciding from a
  // pure host probe, while its sibling `elem_kn_repack` eight lines below has
  // always carried the same term for the same reason. `QuantRepackForDevice`
  // holds the rule; see its header comment for why the ISA probe is passed in
  // rather than called there.
  p.quant_repack = QuantRepackForDevice(p.keep_quant, p.cpu_ref,
                                        vt::cpu::QuantRepackActive(), dev);
  // The load-time prefault, with the SAME device term and for a reason of the
  // same shape: the transform is worth paying for only where the forward reads
  // the borrowed pages. See `GgufPrefaultForDevice`. `VT_GGUF_PREFAULT` and
  // `vllm_cpp.mmap.prefault` still win over the device.
  p.prefault = GgufPrefaultForDevice(dev);
  // KERNEL-GEMM-CPU-TILED lever 2, elementwise [N,K] -> [K,N] repack-at-load.
  // OPT-IN ONLY (default false) because the repacked bytes are transposed and
  // only the CPU MatmulBTKernel honours Tensor.elem_kn_repacked today; see the
  // field comment. Forced off by the oracle switch like every other load
  // transform, so VT_CPU_REF=1 still reproduces the historical load.
  // Gated on the CPU platform, not just the env var. Only the CPU
  // MatmulBTKernel honours Tensor::elem_kn_repacked, and a staged device would
  // upload the transposed bytes and read them as [N,K]. `dev` is the same
  // seam GgufQuantComputeAvailable(dev) above already uses to decide a load
  // transform, so the policy CAN see this and should not be left to a runtime
  // backstop alone (ResidentWeight also VT_CHECKs, belt and braces).
  p.elem_kn_repack = !p.cpu_ref && EnvOnOr("VT_CPU_ELEM_KN_REPACK", false) &&
                     dev == vt::DeviceType::kCPU;
  return p;
}

// THE DEVICE A RESIDENCY DECISION IS ACTUALLY ABOUT (#2516).
//
// `GgufLoadPolicy::device` is the ENGINE's resolved device, and its own field
// comment says every device-dependent decision in the struct reads it. Hybrid
// placement (#2023, #2314) introduced a SECOND, narrower answer for exactly one
// class of tensor: `ActiveMoePlacementPlan().DeviceForLayer(l)` is the device a
// layer's routed experts are COMPUTED on, and `RunMoePlaced` hands that device
// to `MoeBlock`. A keep-quant decision asks "does the device that will execute
// this GEMM have a `vec_dot` for this encoding", so for a PLACED routed-expert
// tower it is a question about the placement device, not about the engine.
//
// On GLM-5.3 `UD-IQ1_S`, all six expert formats now stay quantized on ROCm:
// IQ1_S/IQ3_XXS/IQ2_XXS/Q2_K/Q3_K, and IQ4_XS since #3029. This seam still
// matters, because a format ROCm does not admit routes `kExpandBf16` and
// `LoadStackedExperts` then refuses a load for a tower whose bytes never reach
// the GPU (the model reads a tower only through `GlmExpertSlice`, never
// through `ResidentWeight`) and which the installed plan has already sent to
// the CPU. The CPU `vec_dot` table covers all six, and IQ2_XS, IQ4_NL and
// IQ3_S are formats where the two answers still differ.
//
// THIS IS #1136 AND #2406 ONE SEAM FURTHER ALONG. Both were the same shape: a
// residency decision resolved against a device other than the one that would
// read the bytes. The repair is the same shape too -- ask the device that will
// run it.
//
// INERT BY CONSTRUCTION, in four terms, and the last one is the load-bearing
// one. The plan is a process-global that a load with no placement never writes,
// and a DEFAULT-CONSTRUCTED plan's `engine_device()` is `kCPU`. Returning its
// answer unconditionally would therefore route a CUDA load's experts as though
// they were on the host. So the placed answer is ADOPTED only when the plan
// actively moves THIS tensor somewhere other than the device the plan itself was
// resolved for; every other path returns the policy's own device unchanged.
vt::DeviceType GgufLoadPolicy::ComputeDeviceFor(const std::string& name,
                                                GgufTensorRole role) const {
  if (role != GgufTensorRole::kStackedExpertWeight) return device;
  const MoePlacementPlan& plan = ActiveMoePlacementPlan();
  if (!plan.PlacesAnything()) return device;
  const vt::DeviceType placed = plan.DeviceForRoutedExpertTensor(name);
  if (placed == plan.engine_device()) return device;
  return placed;
}

GgufResidency GgufLoadPolicy::Route(const GgufTensorInfo& tensor,
                                    GgufTensorRole role) const {
  const GgufResidency r = RouteGgufTensor(
      keep_quant, keep_f16, nvfp4_fp4, cpu_ref, role, tensor.ggml_type,
      tensor.shape, ComputeDeviceFor(tensor.name, role), weight_value_dtype,
      tt_weight_residency);
  if (audit) audit(tensor.name, role, r);
  return r;
}

GgufLoadPolicy NoKeepQuant(const GgufLoadPolicy& policy) {
  GgufLoadPolicy p = policy;
  p.keep_quant = false;
  return p;
}

GgufResidency PeekRoute(const GgufLoadPolicy& policy, const GgufTensorInfo& tensor,
                        GgufTensorRole role) {
  // THROUGH THE SAME RESOLVER `Route` USES, not a copy of the expression. This
  // function exists so a caller can ask what the loader WILL do without
  // notifying the audit, and `GgufExpertTowersReachSlotLane` is one such caller
  // -- it peeks the very tensors the loader then routes. Two spellings of the
  // device term is exactly how the bound and the forward come to disagree about
  // one file, which is the defect #1378 records.
  return RouteGgufTensor(policy.keep_quant, policy.keep_f16, policy.nvfp4_fp4,
                         policy.cpu_ref, role, tensor.ggml_type, tensor.shape,
                         policy.ComputeDeviceFor(tensor.name, role),
                         policy.weight_value_dtype,
                         policy.tt_weight_residency);
}

}  // namespace vllm
