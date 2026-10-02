// Nemotron-H (`NemotronHForCausalLM`) — the WEIGHT LOADER (implemented in
// `nemotron_h_weights.cpp`, beside the enumeration it must agree with)
// ([spec](../../../../.agents/specs/nemotron-h-model.md) §5b, §6b "Still owed
// after W4"; issue #517).
//
// W3 enumerated all 18487 released tensors and W4 made the architecture
// compute. Neither MATERIALIZED anything, so every checkpoint load left
// `NemotronHHostWeights::materialized` false and the forward refused by name.
// This is that brick.
//
// ─── THE CHECKPOINT IS NOT UNIFORM, AND THAT IS THE WHOLE JOB ────────────────
//
// `quant_algo: MIXED_PRECISION`, `quant_method: modelopt`. One file carries
// three memory formats, and the repo name ("...-NVFP4") describes only the
// biggest of them:
//
//   routed experts, shared experts, lm_head   NVFP4 W4A16, group_size=16
//     .weight         U8      [out, in/2]     two E2M1 nibbles per byte
//     .weight_scale   F8_E4M3 [out, in/16]    one linear e4m3 per 16 inputs
//     .weight_scale_2 F32     scalar          the global scale, MULTIPLIED
//
//   mamba in_proj / out_proj (46 targets)     FP8 W8A8 static
//     .weight         F8_E4M3 [out, in]
//     .weight_scale   F32     scalar
//     .input_scale    F32     [1]             the STATIC activation scale
//
//   attention q/k/v/o, conv1d, gates, norms, embeddings   plain bf16 / f32
//   KV cache                                  fp8: k_proj.k_scale, v_proj.v_scale
//
// Reading it as UNIFORM NVFP4 is the failure a token gate cannot see: the
// answer stays numerically plausible, the tokens still match, and the load
// moves the wrong bytes. So every weight is held in the format it SHIPS in, and
// the load report states the composition as hard numbers rather than leaving it
// to be inferred.
//
// ─── WHY THE QUANTIZED FORMS ARE KEPT, NOT DEQUANTIZED AT LOAD ───────────────
//
// Two reasons, and the first one is arithmetic. The 5888 routed-expert
// projections are 29.4e9 parameters: 16.5 GB packed (14.7 GB of nibbles at
// 0.5 B/param plus 1.84 GB of group scales at 1 B per 16 — 15.4 GiB), against
// 58.7 GB at bf16. Both figures are DECIMAL GB, and that unit is the point: the
// same number written GiB is 7% wrong in the direction that flatters us. A
// dequantize-at-load loader does not fit on any box this project owns, and on a
// unified-memory box it takes the machine down rather than failing. The second
// is the rule: a widened weight is numerically correct and invisible to a token
// comparison while moving twice the bytes (AGENTS.md).
//
// The host reference forward has no NVFP4 and no FP8 GEMM — those are the CUDA
// `kMoeGroupedGemmNvfp4Marlin` and fp8-linear arms W6 selects. It therefore
// widens a quantized operand TRANSIENTLY at the GEMM call site, through the
// shared `model_loader/nvfp4_dequant.h` seam, and the report below names and
// counts that arm so it is a declared property of this path rather than
// something a reader has to find.
//
// ─── WHAT IS DEFERRED, BY NAME ───────────────────────────────────────────────
//
// The 270-tensor MTP tower (`mtp.layers.*`, unquantized bf16 because the
// `ignore` list carries `mtp*`) is W5 and is NOT materialized. It is counted and
// named in the report rather than skipped silently: 270 tensors dropped on the
// floor is 270 tensors nobody notices. The GGUF arm (W7) refuses earlier, in
// the registry TU.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/nemotron_h.h"
#include "vllm/model_executor/models/nemotron_h_forward.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

namespace vllm {

// What the load actually did, in numbers a gate can assert. Every count is over
// TENSORS as the checkpoint ships them, so `materialized + deferred` is the
// released tensor count and nothing hides in a rounding.
struct NemotronHLoadReport {
  int64_t enumerated = 0;    // what EnumerateNemotronHTensors named
  int64_t in_index = 0;      // what model.safetensors.index.json ships
  int64_t materialized = 0;  // read into a named host slot
  int64_t deferred = 0;      // named, owed to a later W, deliberately not read

  // Composition by the SHIPPED scheme. `*_weights` counts logical projections,
  // `*_tensors` the on-disk tensors they are made of (three each).
  int64_t nvfp4_weights = 0;
  int64_t nvfp4_tensors = 0;
  int64_t fp8_weights = 0;
  int64_t fp8_tensors = 0;
  // The fp8 KV scheme's own tensors (`k_proj.k_scale`, `v_proj.v_scale`), which
  // belong to no projection and are their own row of the scheme table.
  int64_t fp8_kv_scale_tensors = 0;
  // The unquantized remainder, by the dtype it ships in.
  int64_t bf16_tensors = 0;
  int64_t f32_tensors = 0;
  // Tensors whose HOST dtype is wider than their DISK dtype. On the released
  // checkpoint this is exactly the 69 SSM scalars (`A_log`, `D`, `dt_bias` on 23
  // mamba layers), bf16 on disk and f32 in memory because upstream keeps them
  // f32 (`self.A = -torch.exp(self.A_log.float())`) and `vt::Mamba2ChunkScan`
  // validates them f32. Counted so the widening cannot spread unnoticed.
  int64_t widened_tensors = 0;

  int64_t source_bytes = 0;  // bytes read out of the safetensors mappings
  int64_t host_bytes = 0;    // bytes resident in NemotronHHostWeights afterwards

  // The distinct consumer tags that were deferred, and the W that owns them.
  std::vector<std::string> deferred_by_name;
};

// The model dtype, mirroring transformers' own resolution: newer versions
// serialize `dtype` and the legacy spelling is `torch_dtype`. Refuses anything
// this forward cannot represent BY NAME rather than substituting a dtype.
vt::DType ResolveNemotronHModelDType(const HfConfig& config);

// Materialize `NemotronHHostWeights` from an already-opened safetensors
// checkpoint. `shards` is the shared `ModelSource` seam every other
// architecture's loader consumes; source pages are released as each tensor is
// copied out (`MaybeReleaseSourcePages`), so peak RSS tracks the owned mirror
// rather than mirror-plus-mapping.
//
// Throws (refuses by name) on: a tensor the enumeration names and the
// checkpoint does not ship, a tensor the checkpoint ships and the enumeration
// does not name, a dtype or shape that disagrees with the scheme its consumer
// declares, or a missing scale companion.
//
// `name_prefix` (default empty) scopes the load to the tensors under that
// prefix, stripped before the enumeration sees them. A wrapper checkpoint
// (`NemotronH_Nano_Omni_Reasoning_V3`) ships its language tower as
// `language_model.backbone.*` beside `vision_model.*`, `mlp1.*` and
// `sound_*` tensors that the wrapper accounts for itself; upstream maps the
// same prefix away (`hf_to_vllm_mapper`, nano_nemotron_vl.py:904-908). Both
// accounting directions still hold, over the prefixed set.
NemotronHHostWeights LoadNemotronHHostWeights(
    const std::vector<SafetensorsFile>& shards, const NemotronHParams& params,
    vt::DType act_dtype, NemotronHLoadReport* report,
    const std::string& name_prefix = "");

// The report of the load that produced `model`. The load happens inside the
// type-erased `ModelRegistry::Load` factory, so a structural gate has no other
// way to reach the numbers. Throws if `model` is not a NemotronH model.
const NemotronHLoadReport& NemotronHLoadReportOf(const LoadedModel& model);

}  // namespace vllm
