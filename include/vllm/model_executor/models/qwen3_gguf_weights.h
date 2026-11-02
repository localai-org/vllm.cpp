// vllm.cpp ORIGINAL GGUF-format Qwen3 (dense) loader. No upstream vLLM
// mirror — vLLM has no in-tree GGUF support (deprecated in vllm#39583, moved
// to vllm-gguf-plugin). Mirrors the qwen3_5_gguf_weights.{h,cpp} seam: loads a
// GGUF Qwen3 checkpoint (general.architecture = "qwen3") into the SAME
// Qwen3DenseWeights the safetensors loader (qwen3_weights.{h,cpp}) produces,
// so the shared Qwen3DenseModel::Forward is unchanged. Only the SOURCE (ggml
// k-quant blocks, dequant via gguf_dequant.h) and the tensor NAMES differ.
//
// Tensor names + metadata keys mirror llama.cpp @ b10451 (the qwen3 arch):
//   src/llama-arch.cpp     — LLM_TENSOR_NAMES ("token_embd", "output",
//                            "output_norm", "blk.%d.attn_{norm,q,k,v,output,
//                            q_norm,k_norm}", "blk.%d.post_attention_norm",
//                            "blk.%d.ffn_{gate,up,down}") + LLM_KV_*
//                            ("%s.block_count", "%s.attention.head_count[_kv]",
//                            "%s.embedding_length", "%s.rope.freq_base",
//                            "%s.attention.layer_norm_rms_epsilon",
//                            "%s.feed_forward_length", "%s.context_length",
//                            "%s.vocab_size").
//   conversion/qwen.py     — Qwen3Model inherits Qwen2Model -> TextModel; the
//                            base TextModel.modify_tensors does NO norm
//                            modification (no w+1 shift), unlike Qwen3NextModel
//                            which DOES add +1. So this loader uses OwnBf16
//                            for all norm weights, NOT OwnNormMinus1.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/qwen3.h"
#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// Build the HfConfig from a GGUF file's metadata (arch prefix "qwen3").
// vocab_size is taken from the `qwen3.vocab_size` kv when present, else from
// token_embd's leading (out) dim. tie_word_embeddings is detected by the
// ABSENCE of an `output.weight` tensor (llama.cpp TENSOR_DUPLICATED). Sets
// architectures = {"Qwen3ForCausalLM"} so the existing registry resolves.
// Throws std::runtime_error on a missing required key or an unexpected
// architecture.
HfConfig Qwen3HfConfigFromGguf(const GgufFile& gguf);

// True when `general.architecture` is "qwen3". Mirrors IsQwen3_5Gguf's shape:
// the model that owns the family answers the question, so the dispatch in
// model_loader.cpp does not carry a second copy of the arch list.
bool IsQwen3Gguf(const GgufFile& gguf);

// Load the whole Qwen3 dense model from a GGUF file into owned host tensors,
// matching the safetensors loader's layouts and semantics. q/k/v are
// concatenated into one merged qkv_proj (rows q|k|v); gate/up into one
// gate_up_proj (rows gate|up). Norms use OwnBf16 (NO w+1 shift — Qwen3 does
// not shift, unlike Qwen3.5/3Next). Uses config.num_hidden_layers.
//
// `policy` (optional) selects per-tensor residency: null reads the process
// environment (GgufLoadPolicy::FromEnv — VT_CPU_REF / VT_GGUF_KEEP_QUANT),
// which with today's defaults reproduces the all-bf16 expansion. The initial
// implementation expands merged weights (qkv_proj, gate_up_proj) to bf16;
// keep-quant for merged weights is a follow-up.
Qwen3DenseWeights LoadQwen3FromGguf(const GgufFile& gguf,
                                     const HfConfig& config,
                                     const GgufLoadPolicy* policy = nullptr);

// Shared dense GGUF tensor map. Parts retain each source encoding and NK rows;
// callers own family-specific Q/K transforms. Qwen3's existing wrapper keeps
// its historical concatenation and residency unchanged.
Qwen3DenseWeights LoadDenseGgufWeights(const GgufFile& gguf,
                                       const HfConfig& config,
                                       const GgufLoadPolicy* policy,
                                       bool keep_projection_parts);

}  // namespace vllm
