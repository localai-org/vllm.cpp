// VT_HOST_EMBEDDING: gather the embedding rows on the CPU and copy the [T,H]
// result to the device, so the token table never occupies device memory. The
// CPU `vt::Embedding` kernel decodes ONE ROW per gathered id — the same per-row
// discipline as llama.cpp's ggml_get_rows — for every table residency the
// loaders produce: bf16/f16/f32 and the GGUF block-quant formats.
//
// Returns false (the caller uses the device path) when the flag is off or the
// table's host bytes are gone. A forward whose embedding is ALREADY a host
// gather does not need this; it exists for the forwards that own a device
// table. An untied embedding frees the table's whole device residency; a tied
// head keeps the table resident for its GEMM, so the flag then only moves the
// gather off the device.
//
// WHO CALLS IT. Every dense forward that owns a device table and takes its ids
// as a host vector calls `EmbedGather`. The sites that deliberately do NOT are
// the ones where a host gather cannot help or would add a synchronize the path
// exists to remove: decode-graph arms whose ids already live in a device tensor
// (qwen3_moe, gemma3, deepseek_v2, nemotron_h paged, qwen4_exp), the mm embed
// hooks whose ids arrive on device (qwen3_vl, dots3_note), the draft heads
// (qwen3_dflash/dspark), and the custom-residency loaders (kimi_linear).
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"  // Dev, DBuf, OwnedTensor

namespace vllm {
namespace dense_attn {

bool HostEmbedInto(Dev d, DBuf& hidden, const std::vector<int32_t>& token_ids,
                   const OwnedTensor& table, int64_t vocab, int64_t H);

// The gather a forward should call when it owns a device table: the host arm
// above, else `ResidentWeight` + the async id override + `vt::Embedding`. The
// table and its shape are handed over UNRESOLVED so the upload happens only when
// the host arm declines. `what` names the caller in the override's shape check.
void EmbedGather(Dev d, DBuf& out, const std::vector<int32_t>& token_ids,
                 const OwnedTensor& table, int64_t vocab, int64_t H,
                 const char* what);

}  // namespace dense_attn
}  // namespace vllm
