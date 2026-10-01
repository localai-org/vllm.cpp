// Voxtral-Mini-3B e2e AUDIO->TEXT forward + weight loader (audio-track A3).
// See include/vllm/model_executor/models/voxtral.h for full provenance.
//
// Ported 1:1 from vllm/model_executor/models/voxtral.py @ e24d1b24
// (embed_multimodal:382-412, AudioLanguageAdapter:660-668, load_weights:502-568,
// VoxtralEncoderModel.mistral_remapping:674-724). The audio encoder is the A2
// WhisperAudioEncoderForward at Voxtral's encoder config; the text decoder is the
// LANDED shared dense forward (dense_attn::AttnBlock) driven from a merged
// inputs_embeds. Pure additive TU (no shared forward / runner / registry edit).
#include "vllm/model_executor/models/voxtral.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vllm/model_executor/layers/quantization/compressed_tensors/schemes/nvfp4.h"  // LinearMethod seam
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/decode_graph_sizes.h"  // DecodeGraphSizes/PadToCaptureSize
#include "vllm/model_executor/models/dense_attn_block.h"   // AttnBlock, BuildStepInputs, ResidentWeight
#include "vllm/model_executor/models/host_embedding.h"  // VT_HOST_EMBEDDING gather
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vllm/model_executor/models/qwen3_vl_text.h"       // Qwen3VLMergeMultimodal (modality-agnostic merge)
#include "vllm/model_executor/models/voxtral_loader_internal.h"  // the two mmap-reading loader steps (#772)
#include "vt/breakable_graph.h"  // ENG-CUDAGRAPH-BREAK W3: the shared capture seam
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"
#include "vt/unaligned.h"  // LoadUnaligned — mmap'd payloads have no alignment guarantee

namespace vllm {
namespace {

using dense_attn::AttnBlock;
using dense_attn::BuildStepInputs;
using dense_attn::Dev;
using dense_attn::DBuf;
using dense_attn::FusedChainAdoptEnabled;
using dense_attn::ResidentWeight;
using dense_attn::StepInputs;
using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using v1::CommonAttentionMetadata;

std::vector<float> Bf16BitsToF32(const uint16_t* p, int64_t n) {
  std::vector<float> o(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) o[static_cast<size_t>(i)] = vt::BF16ToF32(p[i]);
  return o;
}
std::vector<uint16_t> F32ToBf16Bits(const float* p, int64_t n) {
  std::vector<uint16_t> o(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) o[static_cast<size_t>(i)] = vt::F32ToBF16(p[i]);
  return o;
}
void RoundToBf16(std::vector<float>& v) {
  for (float& x : v) x = vt::BF16ToF32(vt::F32ToBF16(x));
}

// Single-sequence CommonAttentionMetadata (mirror qwen3_vl.cpp StepMeta).
CommonAttentionMetadata StepMeta(int64_t T, int64_t seq_len, int64_t first_slot) {
  CommonAttentionMetadata m;
  m.num_reqs = 1;
  m.num_actual_tokens = static_cast<int>(T);
  m.query_start_loc = {0, static_cast<int32_t>(T)};
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens = {static_cast<int32_t>(seq_len)};
  m.seq_lens_cpu = m.seq_lens;
  m.max_query_len = static_cast<int>(T);
  m.max_seq_len = static_cast<int>(seq_len);
  m.block_table_num_cols = 1;
  m.block_table_tensor = {0};
  for (int64_t t = 0; t < T; ++t) m.slot_mapping.push_back(first_slot + t);
  m.causal = true;
  return m;
}

// One Mistral/Llama decoder layer (== qwen3.cpp RunLayer, no qk-norm branch since
// Voxtral text leaves q_norm/k_norm EMPTY): input norm -> AttnBlock -> post norm ->
// SwiGLU MLP. `hidden` is the delta stream, `res` the residual accumulator.
void RunTextLayer(Dev d, const Qwen3DenseLayerWeights& layer, const HfConfig& cfg,
                  DBuf& hidden, DBuf& res, const StepInputs& si,
                  const CommonAttentionMetadata& meta, const PagedKvCache& kv, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf dhn(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dhn.t(), hidden.t(), w_in, &res.t(), vt::kFusedAddRmsNormStd, eps);
  else
    vt::RmsNorm(d.q, dhn.t(), hidden.t(), w_in, vt::RmsNormArgs{eps, false}, &res.t());

  DBuf attn = AttnBlock(d, layer.attn, cfg, dhn.t(), si, meta, kv, T);

  Tensor w_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf dh2(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dh2.t(), attn.t(), w_post, &res.t(), vt::kFusedAddRmsNormStd, eps);
  else
    vt::RmsNorm(d.q, dh2.t(), attn.t(), w_post, vt::RmsNormArgs{eps, false}, &res.t());

  auto gate_up = layers::MakeMlpGateUpMethod(layer.mlp.gate_up_proj,
                                             layer.mlp.gate_proj_fp4,
                                             layer.mlp.up_proj_fp4, I);
  DBuf act = gate_up->Apply(d, dh2.t());
  auto down = layers::MakeLinearMethod(layer.mlp.down_proj, layer.mlp.down_proj_fp4);
  hidden = down->Apply(d, act.t(), DType::kBF16);
}

// Forked dense forward from a merged inputs_embeds: N layers -> final RMSNorm ->
// UNTIED lm_head, returning the LAST row's logits [1,vocab] f32 ON DEVICE.
// Mirrors qwen3.cpp ForwardBody but (1) starts from provided device embeds and
// (2) always untied. The logits stay on device so the caller can run the greedy
// argmax there (no full-vocab D2H); the input embeds are a device Tensor so the
// decode path feeds the on-device token embed straight in (no D2H->H2D).
DBuf ForwardLastLogits(Dev d, const Tensor& hidden_in,
                       const std::vector<int32_t>& positions, int64_t T,
                       const CommonAttentionMetadata& meta,
                       const std::vector<PagedKvCache>& attn_kv,
                       const Qwen3DenseWeights& weights,
                       const HfConfig& config) {
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  const float eps = static_cast<float>(config.rms_norm_eps);

  // Working copy (device->device): RunTextLayer mutates `hidden`, so it must not
  // alias the caller's embed buffer.
  DBuf hidden(d, DType::kBF16, {T, H});
  d.b.Copy(d.q, hidden.ptr(), hidden_in.data,
           static_cast<size_t>(T) * static_cast<size_t>(H) * vt::SizeOf(DType::kBF16));
  DBuf res(d, DType::kBF16, {T, H});
  res.Zero(d);

  StepInputs si = BuildStepInputs(d, positions, meta, config);
  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    RunTextLayer(d, weights.layers[static_cast<size_t>(l)], config, hidden, res, si, meta,
                 attn_kv[static_cast<size_t>(l)], T);

  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dnorm.t(), hidden.t(), w_fn, &res.t(), vt::kFusedAddRmsNormStd, eps);
  else
    vt::RmsNorm(d.q, dnorm.t(), hidden.t(), w_fn, vt::RmsNormArgs{eps, false}, &res.t());

  // Gather the LAST row, then UNTIED lm_head (Matmul-B [H, vocab]).
  DBuf last(d, DType::kBF16, {1, H});
  d.b.Copy(d.q, last.ptr(),
           static_cast<char*>(dnorm.ptr()) +
               static_cast<size_t>((T - 1) * H) * vt::SizeOf(DType::kBF16),
           static_cast<size_t>(H) * vt::SizeOf(DType::kBF16));
  Tensor lm = ResidentWeight(d, weights.lm_head);  // [H, vocab]
  DBuf logits(d, DType::kF32, {1, vocab});
  vt::Matmul(d.q, logits.t(), last.t(), lm);
  return logits;
}

// On-GPU greedy argmax: reduce the [1,vocab] f32 logits ON DEVICE and download
// ONLY the winning int64 token id, instead of the full-vocab f32 D2H + a host
// argmax scan. Mirrors vLLM's greedy sampler device path (vt::GreedyArgmax);
// LOWEST-index tie-break (torch.argmax), byte-for-byte the same winner the host
// scan produced, so the greedy token stream is unchanged (multimodal-speed.md §5 #2).
int32_t ArgMaxDevice(Dev d, DBuf& dlogits) {
  DBuf ids(d, DType::kI64, {1});
  vt::GreedyArgmax(d.q, ids.t(), dlogits.t());
  int64_t id = 0;
  ids.Download(d, &id);
  return static_cast<int32_t>(id);
}

// ─── VoxtralDecodeGraph support (ROAD-V1-MM lever #3 W1) ─────────────────────
// Graph-driver glue mirrored 1:1 from qwen3_moe.cpp (the Qwen3-Coder full-
// attention decode-graph sibling): the SAME device-pool DBuf discipline, the SAME
// attention-only padded-decode builder, and the SAME device-logits wrappers.
using dense_attn::MakeTensor;

// Embed hidden[T,H] bf16 = embed_tokens[token_ids] into a PERSISTENT device
// buffer, kept OUTSIDE the CUDA graph (mirror qwen3_moe.cpp EmbedInto): the CUDA
// Embedding op allocates a device bounds-check flag (cudaMalloc/cudaFree) and syncs
// the stream, both illegal inside a capture region, and it consumes the HOST
// token_ids. The graph driver runs this per step into its persistent hidden buffer,
// then captures/replays ForwardLastLogits over that fixed hidden address.
void VoxtralEmbedInto(Dev d, DBuf& hidden, const std::vector<int32_t>& token_ids,
                      const Qwen3DenseWeights& weights, const HfConfig& config) {
  dense_attn::EmbedGather(d, hidden, token_ids, weights.embed_tokens,
                          config.vocab_size, config.hidden_size,
                          "voxtral embed");
}

// Overwrite dst's CONTENTS from src WITHOUT changing dst.data() when the sizes
// already match (preserves the fixed address a captured host->device copy reads
// from); reallocate only when the shape actually changed. (qwen3_moe.cpp:249.)
template <typename T>
void CopyInPlace(std::vector<T>& dst, const std::vector<T>& src) {
  if (dst.size() != src.size()) {
    dst = src;
  } else {
    std::copy(src.begin(), src.end(), dst.begin());
  }
}

// Build the S-padded PURE-DECODE inputs from the real B-request step (B<=S). The
// ATTENTION-ONLY builder (Voxtral text is pure full attention, no GDN) — a verbatim
// copy of qwen3_moe.cpp BuildPaddedDecodeAttn. At S==B this is a bit-identical
// rebuild of the eager inputs; the S-B padding rows are made inert exactly as
// vLLM's cudagraph padding (token/pos 0; slot_mapping -1 skips the KV write;
// seq_lens 1 + block_table row 0 reads block 0, whose output row is discarded).
void BuildPaddedDecodeAttn(int64_t S, const std::vector<int32_t>& tok,
                           const std::vector<int32_t>& pos,
                           const CommonAttentionMetadata& am,
                           std::vector<int32_t>& tok_out,
                           std::vector<int32_t>& pos_out,
                           CommonAttentionMetadata& am_out) {
  const int64_t cols = am.block_table_num_cols;

  // GCC-13 `-O2 -Werror=array-bounds` FALSE POSITIVE (project issue #155): with
  // S>=B guaranteed by the caller (padded decode, B<=S), these std::copy calls
  // into the freshly-sized-to-S `*_out` buffers can never overrun, but GCC-13's
  // -O2 inliner mis-derives a `[0,4]`-byte bound for the memmove and errors. The
  // suppression is scoped to exactly this builder's in-bounds copies; a comment,
  // not a broad -Wno. (Advances #155; unrelated to the W7-device change.)
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
  tok_out.assign(static_cast<size_t>(S), 0);
  pos_out.assign(static_cast<size_t>(S), 0);
  std::copy(tok.begin(), tok.end(), tok_out.begin());
  std::copy(pos.begin(), pos.end(), pos_out.begin());

  am_out = am;  // carries causal + block_table_num_cols + max_seq_len
  am_out.num_reqs = static_cast<int>(S);
  am_out.num_actual_tokens = static_cast<int>(S);
  am_out.max_query_len = 1;  // pure decode
  // W10 (#1857): a pure-decode rewrite is never spec-classified. Belt on the
  // vt shape guard's braces (S == q*S only at q == 1).
  am_out.uniform_spec_query_len = 0;
  am_out.slot_mapping.assign(static_cast<size_t>(S), -1);
  std::copy(am.slot_mapping.begin(), am.slot_mapping.end(),
            am_out.slot_mapping.begin());
  am_out.seq_lens.assign(static_cast<size_t>(S), 1);
  std::copy(am.seq_lens.begin(), am.seq_lens.end(), am_out.seq_lens.begin());
  am_out.block_table_tensor.assign(static_cast<size_t>(S * cols), 0);
  std::copy(am.block_table_tensor.begin(), am.block_table_tensor.end(),
            am_out.block_table_tensor.begin());
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
  am_out.query_start_loc.resize(static_cast<size_t>(S + 1));
  for (int64_t i = 0; i <= S; ++i)
    am_out.query_start_loc[static_cast<size_t>(i)] = static_cast<int32_t>(i);
}

// Hand a freshly-forwarded [rows,vocab] f32 DBuf out as an OWNING ForwardLogits
// (eager fallback / cold pre-warm) — the pool block's lifetime moves into a
// shared_ptr whose deleter returns it to the DevicePool (no per-step
// cudaMalloc/cudaFree). Verbatim from qwen3_moe.cpp WrapDeviceLogits.
ForwardLogits WrapDeviceLogits(Dev d, DBuf&& dlogits, int64_t rows, int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = dlogits.t();
  fl.device_storage = dlogits.ReleaseShared();
  (void)d;
  return fl;
}

// NON-OWNING [rows,vocab] f32 view over a buffer the graph slot keeps alive
// (mirror qwen3_moe.cpp ViewDeviceLogits). Stream ordering guarantees the sampler's
// later reads see the replay's writes; the next same-size replay overwrites it.
ForwardLogits ViewDeviceLogits(void* base, vt::Device device, int64_t rows,
                               int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = MakeTensor(base, DType::kF32, device, {rows, vocab});
  fl.device_storage = std::shared_ptr<void>(base, [](void*) {});
  return fl;
}

// --- weight loading helpers -----------------------------------------------------
using dense_loaders::LoadBf16Direct;
using dense_loaders::LoadBf16Transposed;
using dense_loaders::LoadMergedBf16RawNK;

void LoadEncoderWeights(const TensorResolver& get, const std::vector<float>& embed_positions,
                        const multimodal::WhisperAudioEncoderConfig& cfg,
                        multimodal::WhisperAudioEncoderWeights& w) {
  const std::string E = "mm_whisper_embeddings.whisper_encoder.";
  w.conv1_w = VoxtralStBf16ToF32(get(E + "conv_layers.0.weight"));
  w.conv1_b = VoxtralStBf16ToF32(get(E + "conv_layers.0.bias"));
  w.conv2_w = VoxtralStBf16ToF32(get(E + "conv_layers.1.weight"));
  w.conv2_b = VoxtralStBf16ToF32(get(E + "conv_layers.1.bias"));
  w.embed_positions_w = embed_positions;
  w.final_ln_w = VoxtralStBf16ToF32(get(E + "transformer.norm.weight"));
  w.final_ln_b = VoxtralStBf16ToF32(get(E + "transformer.norm.bias"));
  w.layers.resize(static_cast<size_t>(cfg.num_layers));
  for (int64_t l = 0; l < cfg.num_layers; ++l) {
    const std::string p = E + "transformer.layers." + std::to_string(l) + ".";
    multimodal::WhisperEncoderLayerWeights& lw = w.layers[static_cast<size_t>(l)];
    lw.attn_ln_w = VoxtralStBf16ToF32(get(p + "attention_norm.weight"));
    lw.attn_ln_b = VoxtralStBf16ToF32(get(p + "attention_norm.bias"));
    lw.final_ln_w = VoxtralStBf16ToF32(get(p + "ffn_norm.weight"));
    lw.final_ln_b = VoxtralStBf16ToF32(get(p + "ffn_norm.bias"));
    lw.q_w = VoxtralStBf16ToF32(get(p + "attention.wq.weight"));
    lw.q_b = VoxtralStBf16ToF32(get(p + "attention.wq.bias"));
    lw.k_w = VoxtralStBf16ToF32(get(p + "attention.wk.weight"));  // k_proj: NO bias
    lw.v_w = VoxtralStBf16ToF32(get(p + "attention.wv.weight"));
    lw.v_b = VoxtralStBf16ToF32(get(p + "attention.wv.bias"));
    lw.out_w = VoxtralStBf16ToF32(get(p + "attention.wo.weight"));
    lw.out_b = VoxtralStBf16ToF32(get(p + "attention.wo.bias"));
    lw.fc1_w = VoxtralStBf16ToF32(get(p + "feed_forward.w1.weight"));
    lw.fc1_b = VoxtralStBf16ToF32(get(p + "feed_forward.w1.bias"));
    lw.fc2_w = VoxtralStBf16ToF32(get(p + "feed_forward.w2.weight"));
    lw.fc2_b = VoxtralStBf16ToF32(get(p + "feed_forward.w2.bias"));
  }
}

// Build the merged qkv OwnedTensor [Hq*Dh + 2*Hkv*Dh, K] (rows q|k|v) with q/k
// rope-permuted and v raw — the mistral-format analog of LoadMergedBf16RawNK.
OwnedTensor BuildPermutedQKV(const TensorResolver& get, const std::string& b,
                             const HfConfig& config) {
  const StTensor& wq = get(b + "attention.wq.weight");
  const StTensor& wk = get(b + "attention.wk.weight");
  const StTensor& wv = get(b + "attention.wv.weight");
  const int64_t K = wq.shape[1];
  const int64_t qd = wq.shape[0], kd = wk.shape[0], vd = wv.shape[0];
  std::vector<uint16_t> pq = VoxtralPermuteQKBf16(wq, config.num_attention_heads);
  std::vector<uint16_t> pk = VoxtralPermuteQKBf16(wk, config.num_key_value_heads);
  OwnedTensor m = dense_loaders::MakeOwned(DType::kBF16, {qd + kd + vd, K});
  auto* dst = reinterpret_cast<uint16_t*>(m.bytes.data());
  std::memcpy(dst, pq.data(), static_cast<size_t>(qd) * K * sizeof(uint16_t));
  std::memcpy(dst + qd * K, pk.data(), static_cast<size_t>(kd) * K * sizeof(uint16_t));
  std::memcpy(dst + (qd + kd) * K, wv.data, static_cast<size_t>(vd) * K * sizeof(uint16_t));
  m.nk = true;
  return m;
}

void LoadTextWeights(const TensorResolver& get, const HfConfig& config,
                     Qwen3DenseWeights& w) {
  w.tie_word_embeddings = false;
  w.attention_bias = false;
  w.embed_tokens = LoadBf16Direct(get, "mm_whisper_embeddings.tok_embeddings.weight");
  w.final_norm = LoadBf16Direct(get, "norm.weight");
  w.lm_head = LoadBf16Transposed(get, "output.weight");  // untied [vocab,H] -> [H,vocab]
  w.layers.resize(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const std::string b = "layers." + std::to_string(l) + ".";
    Qwen3DenseLayerWeights& lw = w.layers[static_cast<size_t>(l)];
    lw.input_layernorm = LoadBf16Direct(get, b + "attention_norm.weight");
    lw.post_attention_layernorm = LoadBf16Direct(get, b + "ffn_norm.weight");
    lw.attn.qkv_proj = BuildPermutedQKV(get, b, config);  // q/k rope-permuted, v raw
    lw.attn.o_proj = LoadMergedBf16RawNK(get, {b + "attention.wo.weight"});
    // SwiGLU: w1 = gate, w3 = up, w2 = down (mistral feed_forward naming).
    lw.mlp.gate_up_proj = LoadMergedBf16RawNK(
        get, {b + "feed_forward.w1.weight", b + "feed_forward.w3.weight"});
    lw.mlp.down_proj = LoadMergedBf16RawNK(get, {b + "feed_forward.w2.weight"});
  }
}

}  // namespace

// --- bf16 StTensor -> host f32 vector (encoder + adapter weights). --------------
// `t.data` points into the safetensors mmap, whose payload offset carries NO
// alignment guarantee (issue #772), so the bytes are read through
// vt::LoadUnaligned rather than a `const uint16_t*` — the same treatment
// dense_loaders::TransposeBf16 and minimax_h3_vae_loader.cpp already use, and
// free: at -O2 it emits the identical `movzwl` and no call.
std::vector<float> VoxtralStBf16ToF32(const StTensor& t) {
  VT_CHECK(t.dtype == "BF16", "voxtral: expected BF16 tensor");
  const auto* src = static_cast<const unsigned char*>(static_cast<const void*>(t.data));
  const size_t n = t.nbytes / sizeof(uint16_t);
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i)
    out[i] = vt::BF16ToF32(vt::LoadUnaligned<uint16_t>(src + i * 2));
  return out;
}

// Permute the rows of a bf16 [n_heads*head_dim, K] q/k weight from the Meta-
// interleaved rope layout (mistral consolidated) to the HF NeoX layout vLLM's
// rotary_emb (is_neox_style=True) expects — the EXACT transform vLLM applies on
// the mistral load path (verified bit-exact: permute(wq)==vLLM q_proj). Row map
// per head: out(j*hd2 + i) <- in(2i + j), hd2 = head_dim/2. Pure byte reorder of
// bf16 values (no arithmetic) => bit-exact. wv/wo are NOT permuted.
//
// The source row address is computed in `unsigned char` (issue #772): the
// safetensors payload has no alignment guarantee, and forming
// `reinterpret_cast<const uint16_t*>(t.data)` — let alone indexing it — is
// undefined even though every access here is a memcpy that never dereferences
// it as a uint16_t. That memcpy laundering is exactly why UBSan never reported
// this site while it reported its two siblings. NOT vt::LoadUnaligned: this is a
// bulk row copy, not a scalar load, so the `* sizeof(uint16_t)` that used to be
// implicit in the pointer type is now explicit in the byte offset.
std::vector<uint16_t> VoxtralPermuteQKBf16(const StTensor& t, int64_t n_heads) {
  VT_CHECK(t.dtype == "BF16" && t.shape.size() == 2, "voxtral: q/k permute needs 2-D BF16");
  const int64_t d1 = t.shape[0], K = t.shape[1];
  const int64_t hd = d1 / n_heads, hd2 = hd / 2;
  VT_CHECK(hd * n_heads == d1 && hd2 * 2 == hd, "voxtral: q/k permute head split mismatch");
  const auto* src = static_cast<const unsigned char*>(static_cast<const void*>(t.data));
  std::vector<uint16_t> out(static_cast<size_t>(d1) * K);
  for (int64_t h = 0; h < n_heads; ++h)
    for (int64_t i = 0; i < hd2; ++i)
      for (int64_t j = 0; j < 2; ++j) {
        const int64_t out_row = h * hd + j * hd2 + i;
        const int64_t in_row = h * hd + 2 * i + j;
        std::memcpy(&out[static_cast<size_t>(out_row) * K],
                    src + static_cast<size_t>(in_row) * K * sizeof(uint16_t),
                    static_cast<size_t>(K) * sizeof(uint16_t));
      }
  return out;
}

// ─── VoxtralDecodeGraph (BF16 Mistral/Llama full-attention decode CUDA-graph) ──
// The Voxtral-text sibling of Qwen3MoeDecodeGraph (qwen3_moe.cpp) and
// Qwen3_5DenseDecodeGraph (qwen3_5.cpp): the SAME cold -> warm -> capture -> replay
// state machine, the SAME padded-batch capture set (decode_graph_sizes.h) and the
// SAME persistent fixed-address host inputs + persistent embed/logits buffers,
// driving the Voxtral text forward (ForwardLastLogits over VoxtralEmbedInto — the
// shared dense_attn::AttnBlock full attention + SwiGLU MLP + untied lm_head, NO GDN,
// NO MoE). The captured region is the EXACT op sequence the eager decode already
// ran, so at the single-sequence B==S==1 point the graph output is a bit-identical
// rebuild of the eager forward.
//
// GRAPH-SAFETY AUDIT (mirrors qwen3_moe.cpp): capture requires stable pointers and
// no host sync / stream-ordered alloc inside the region.
//   * Embedding (device flag cudaMalloc + stream sync) stays OUTSIDE (VoxtralEmbedInto).
//   * All device scratch comes from the shared DevicePool, whose blocks are recycled
//     (never returned to the driver) — the cold pre-warm step at this exact size
//     populates every size class the capture then reuses, so capture does no cudaMalloc.
//   * ResidentWeight uploads every text weight once, on first touch — done during the
//     PREFILL forward (which touches every layer + lm_head + final_norm) before decode.
//   * The paged full-attention decode (head_dim 128, GQA 32/8) uses the same
//     dense_attn::AttnBlock -> vt::PagedAttention path the already-gated Qwen3-Coder
//     decode graph captures; its host max_seq_len only sizes the split grid and the
//     per-request geometry is read from the DEVICE seq_lens, so a captured graph stays
//     correct as the sequence grows (cuda_flash_attn_fa2.cu:23-31; identical contract
//     to the landed 27B/35B/Coder decode graphs).
//   * cuBLASLt's workspace is a one-time per-context cudaMalloc (pre-warm).
struct VoxtralDecodeGraph::Impl {
  Impl(const Qwen3DenseWeights& w, const HfConfig& c, vt::Queue q, int64_t max_reqs)
      : weights(w), config(c), queue(q), max_num_reqs(max_reqs) {
    // ENG-CUDAGRAPH-BREAK W3 (#1291): the kill switch is the SEAM's, not this
    // driver's. `vt::GraphCaptureEnabled()` reads `VLLM_CPP_CUDAGRAPH` once per
    // process into a function-local static; six drivers each read that variable
    // for themselves before this row, and there was no one switch that turned
    // capture off. This driver no longer owns a copy of it.
    Backend& b = vt::GetBackend(queue.device.type);
    // The TT captured arm of this family is not gated (#2812): explicit opt-in only.
    enabled = vt::GraphCaptureEnabled() &&
              vllm::platforms::GetPlatform(queue.device.type).support_static_graph_mode() &&
              !vllm::platforms::GetPlatform(queue.device.type).static_graph_requires_opt_in() &&
              b.SupportsGraphCapture();
  }
  ~Impl() {
    if (std::getenv("VT_DECODE_GRAPH_STATS") != nullptr)
      std::fprintf(stderr,
                   "[VoxtralDecodeGraph] Voxtral text (Mistral/Llama) decode graph: "
                   "%lld total replays across %zu captured size(s)\n",
                   static_cast<long long>(replays), slots.size());
    // No DestroyGraph loop: every segment handle belongs to the slot's
    // `vt::BreakableGraph`, whose destructor releases it through
    // `Backend::DestroyGraph`. That routing is what lets ENG-CUDAGRAPH-DEDUP
    // (#1162) interpose at the backend later without editing this driver.
  }

  // One captured padded batch size. Owns its OWN persistent host inputs (the
  // captured graph's host->device copies bake these addresses), its persistent
  // embed target + logits output, and its instantiated graph.
  struct SizeSlot {
    std::vector<int32_t> token_ids;  // [S]
    std::vector<int32_t> positions;  // [S]
    CommonAttentionMetadata attn_meta;
    std::unique_ptr<DBuf> hidden;  // [S,H] bf16 persistent embed target
    std::unique_ptr<DBuf> logits;  // [S,vocab] f32 held graph output
    // ENG-CUDAGRAPH-BREAK W3 (#1291): the instantiated graph, its handle
    // ownership, its release and its `captured()` state now live in the shared
    // seam instead of in a raw `void*` plus a `bool` this driver maintained by
    // hand. `vt::BreakableGraph` is non-copyable and is constructed in place by
    // `slots[S]`, so the map still owns one per padded size.
    vt::BreakableGraph graph;
    int fa_cols = -1;  // captured block-table column count
    bool warm = false;
    int64_t replays = 0;

    // In-place refresh of the persistent host inputs (fixed addresses once the
    // slot's vectors reach size S) so a replay re-reads this step's tokens.
    void Refresh(const std::vector<int32_t>& tok, const std::vector<int32_t>& pos,
                 const CommonAttentionMetadata& am) {
      CopyInPlace(token_ids, tok);
      CopyInPlace(positions, pos);
      CopyInPlace(attn_meta.slot_mapping, am.slot_mapping);
      CopyInPlace(attn_meta.block_table_tensor, am.block_table_tensor);
      CopyInPlace(attn_meta.seq_lens, am.seq_lens);
      CopyInPlace(attn_meta.query_start_loc, am.query_start_loc);
      attn_meta.num_reqs = am.num_reqs;
      attn_meta.num_actual_tokens = am.num_actual_tokens;
      attn_meta.max_query_len = am.max_query_len;
      attn_meta.max_seq_len = am.max_seq_len;
      attn_meta.block_table_num_cols = am.block_table_num_cols;
      attn_meta.causal = am.causal;
    }
  };

  const Qwen3DenseWeights& weights;
  const HfConfig& config;
  vt::Queue queue;
  int64_t max_num_reqs = 0;  // == max_num_seqs; padded decode batch cap
  bool enabled = false;

  std::map<int64_t, SizeSlot> slots;  // padded size S -> slot
  int64_t replays = 0;                // total replays (diagnostics)
  bool any_captured = false;          // diagnostics: at least one live graph
};

VoxtralDecodeGraph::VoxtralDecodeGraph(const Qwen3DenseWeights& weights,
                                       const HfConfig& config, vt::Queue queue,
                                       int64_t max_num_reqs)
    : impl_(std::make_unique<Impl>(weights, config, queue, max_num_reqs)) {}

VoxtralDecodeGraph::~VoxtralDecodeGraph() = default;

bool VoxtralDecodeGraph::captured() const { return impl_->any_captured; }
int64_t VoxtralDecodeGraph::replay_count() const { return impl_->replays; }

ForwardLogits VoxtralDecodeGraph::Step(const std::vector<int32_t>& token_ids,
                                       const std::vector<int32_t>& positions,
                                       const CommonAttentionMetadata& attn_meta,
                                       const std::vector<PagedKvCache>& attn_kv) {
  const int64_t B = static_cast<int64_t>(token_ids.size());
  Backend& b = vt::GetBackend(impl_->queue.device.type);
  Dev d{b, impl_->queue};
  const int64_t vocab = impl_->config.vocab_size;
  const int64_t H = impl_->config.hidden_size;

  // The capturable region (ForwardLastLogits) gathers the LAST row before lm_head,
  // returning [1,vocab]; the single-sequence mm driver only ever pads to S=1, so
  // this is the B==S==1 bit-identical-rebuild case. A batch that exceeds the capture
  // set (never here) falls to the eager forward.
  const int64_t S = PadToCaptureSize(B, impl_->max_num_reqs);
  if (!impl_->enabled || S < 0 || S != 1) {
    // Eager fallback (graphs disabled / batch outside the single-seq capture set):
    // embed then run the exact ForwardLastLogits the pre-graph decode used. B is
    // always 1 in the single-sequence mm driver, so gather-last returns [B,vocab].
    DBuf hidden(d, DType::kBF16, {B, H});
    VoxtralEmbedInto(d, hidden, token_ids, impl_->weights, impl_->config);
    DBuf lg = ForwardLastLogits(d, hidden.t(), positions, B, attn_meta, attn_kv,
                                impl_->weights, impl_->config);
    return WrapDeviceLogits(d, std::move(lg), B, vocab);
  }

  // Pad this step's real B-request inputs up to S (inert padding rows), then refresh
  // THIS size's persistent host buffers in place.
  Impl::SizeSlot& s = impl_->slots[S];
  const int cols = attn_meta.block_table_num_cols;
  std::vector<int32_t> ptok, ppos;
  CommonAttentionMetadata pam;
  BuildPaddedDecodeAttn(S, token_ids, positions, attn_meta, ptok, ppos, pam);

  // A block-table column-count change reallocates the persistent block_table (the
  // captured H2D copy's source address moves) -> invalidate this slot's graph.
  const bool cols_changed = (s.fa_cols != -1 && s.fa_cols != cols);
  s.Refresh(ptok, ppos, pam);
  s.fa_cols = cols;
  if (cols_changed && s.graph.captured()) {
    // Reset() releases every segment through Backend::DestroyGraph and returns
    // the container to its as-constructed state, which is also what lets the
    // next capture open a scope on it (the scope refuses a container that
    // already holds one).
    s.graph.Reset();
    s.warm = false;
  }

  // Fast path: this size's graph is captured. Embed OUTSIDE the graph into the
  // persistent hidden buffer, then relaunch the captured layer region.
  if (s.graph.captured()) {
    VoxtralEmbedInto(d, *s.hidden, s.token_ids, impl_->weights, impl_->config);
    // Through the seam's container, never `Backend::ReplayGraph` directly: the
    // container replays its segments in order (one, here, because a decode
    // capture is kFull) and owns the G3 replay counter the reachability gate
    // reads.
    s.graph.Replay(impl_->queue);
    ++s.replays;
    ++impl_->replays;
    return ViewDeviceLogits(s.logits->ptr(), d.q.device, B, vocab);
  }

  // Warm: the pool + weight residency + per-shape kernel scratch were warmed for
  // this size by the previous (eager) step. CAPTURE the layer region once,
  // instantiate the graph, then launch it.
  if (s.warm) {
    VoxtralEmbedInto(d, *s.hidden, s.token_ids, impl_->weights, impl_->config);
    // ENG-CUDAGRAPH-BREAK W3 (#1291): the capture is the SHARED SEAM's, not this
    // driver's hand-rolled `BeginCapture`/`EndCaptureGraph` pair. The scope owns
    // the segment, the handle, its release, the drain a mid-capture throw needs
    // and the G3 counters.
    //
    // kFULL, and the mode is the whole argument. vLLM's v1 default
    // `FULL_AND_PIECEWISE` (`vllm/config/compilation.py:63`) is documented at
    // `:630-632` as a FULL graph for DECODE batches and a piecewise one for
    // prefill and mixed batches, and `decode_mode()` (`:65-66`) returns the full
    // half. This is a decode driver, so its capture is ONE segment with the
    // attention calls INSIDE it — byte-identical in shape to the region this
    // replaces. Opening it kPiecewise would turn every layer's attention into an
    // eager call between two graph replays, which is not vLLM's decode behaviour
    // and which nothing in this row's record supports.
    std::optional<DBuf> lg;
    {
      vt::GraphCaptureScope scope(b, impl_->queue, s.graph, vt::GraphCaptureMode::kFull);
      lg = ForwardLastLogits(d, s.hidden->t(), s.positions, S, s.attn_meta,
                             attn_kv, impl_->weights, impl_->config);
    }  // ~GraphCaptureScope closes the segment and files it on s.graph
    // NOT CAPTURED covers TWO states, and returning `*lg` is correct for exactly
    // one of them. `~GraphCaptureScope` must swallow a throwing `EndCaptureGraph`
    // — a destructor that propagates terminates — so a FAILED capture leaves the
    // container reporting what an INERT scope reports.
    //
    //   * INERT (`capture_failed() == false`): the backend cannot capture, or
    //     `VLLM_CPP_CUDAGRAPH=0`. The scope made no backend call, the region
    //     above ran EAGERLY, and `*lg` is a real result.
    //   * FAILED (`capture_failed() == true`): `Backend::EndCaptureGraph` threw
    //     (`src/vt/cuda/cuda_backend.cu:229`, `Check()` at `:50`). Under stream
    //     capture NOTHING between `BeginCapture` and the throw executed: every
    //     kernel was RECORDED, so `*lg` is pool-recycled memory, and returning it
    //     would hand this step uncomputed device memory as its logits — silently
    //     wrong tokens, no fault, and a token gate cannot see it.
    //
    // So the failure PROPAGATES, carrying the runtime's own exception, which is
    // exactly what the pre-W3 driver did (its `s.graph = b.EndCaptureGraph(...)`
    // was unguarded). Gated at
    // `tests/vllm/models/test_voxtral_decode_graph_seam.cpp`.
    if (!s.graph.captured()) {
      s.warm = false;  // either way this slot goes back to cold
      if (s.graph.capture_failed()) {
        const std::exception_ptr err = s.graph.capture_error();
        s.graph.Reset();  // clear the failure with the graph it described
        // The runtime's OWN diagnosis where the seam holds it. It is empty only
        // on the arm where an exception was already propagating THROUGH the
        // scope, which cannot reach this line; the refusal below is what makes
        // that unreachability an assertion rather than a claim.
        if (err) std::rethrow_exception(err);
        VT_CHECK(false,
                 "VoxtralDecodeGraph: the decode capture was ABANDONED and its logits "
                 "were never computed; refusing to return uncaptured device memory");
      }
      ForwardLogits drained = WrapDeviceLogits(d, std::move(*lg), B, vocab);
      drained.device_tensor =
          MakeTensor(drained.device_storage.get(), DType::kF32, d.q.device, {B, vocab});
      return drained;
    }
    s.logits = std::make_unique<DBuf>(std::move(*lg));
    impl_->any_captured = true;
    if (std::getenv("VT_DECODE_GRAPH_STATS") != nullptr)
      std::fprintf(stderr,
                   "[VoxtralDecodeGraph] captured Voxtral text decode graph for "
                   "padded size S=%lld (real B=%lld)\n",
                   static_cast<long long>(S), static_cast<long long>(B));
    s.graph.Replay(impl_->queue);
    s.replays = 1;
    ++impl_->replays;
    return ViewDeviceLogits(s.logits->ptr(), d.q.device, B, vocab);
  }

  // Cold size: run one EAGER step (pre-warms the DevicePool size classes, the
  // resident weights, and the paged-attention per-shape scratch for this size) and
  // defer capture to the next same-size step. This is a real decode step — nothing
  // is wasted. (Re)allocate the persistent hidden buffer to this size.
  s.hidden = std::make_unique<DBuf>(d, DType::kBF16, std::vector<int64_t>{S, H});
  VoxtralEmbedInto(d, *s.hidden, s.token_ids, impl_->weights, impl_->config);
  DBuf lg = ForwardLastLogits(d, s.hidden->t(), s.positions, S, s.attn_meta, attn_kv,
                              impl_->weights, impl_->config);
  s.warm = true;
  // lg is [S,vocab]; hand ownership out but expose only the first B (real) rows.
  ForwardLogits fl = WrapDeviceLogits(d, std::move(lg), B, vocab);
  fl.device_tensor = MakeTensor(fl.device_storage.get(), DType::kF32, d.q.device,
                                {B, vocab});
  return fl;
}

multimodal::WhisperAudioEncoderConfig VoxtralEncoderConfig() {
  multimodal::WhisperAudioEncoderConfig c;
  c.d_model = 1280;
  c.num_heads = 20;   // head_dim = 1280/20 = 64
  c.num_layers = 32;
  c.ffn_dim = 5120;
  c.num_mel_bins = 128;
  c.max_source_positions = 1500;
  c.n_frames = 3000;
  c.norm_eps = 1e-5f;
  return c;
}

VoxtralWeights LoadVoxtralWeights(const SafetensorsFile& st,
                                  const std::vector<float>& embed_positions,
                                  const HfConfig& text_config) {
  const TensorResolver get = [&st](const std::string& name) -> const StTensor& {
    return st.Get(name);
  };
  VoxtralWeights w;
  w.encoder_cfg = VoxtralEncoderConfig();
  w.downsample_factor = 4;
  w.text_hidden = text_config.hidden_size;
  LoadEncoderWeights(get, embed_positions, w.encoder_cfg, w.encoder);
  w.adapter_w_in = VoxtralStBf16ToF32(get("mm_whisper_embeddings.audio_language_projection.0.weight"));
  w.adapter_w_out = VoxtralStBf16ToF32(get("mm_whisper_embeddings.audio_language_projection.2.weight"));
  LoadTextWeights(get, text_config, w.text);
  return w;
}

std::vector<float> VoxtralProjectAudio(const std::vector<float>& enc_out,
                                       const VoxtralWeights& w, Backend& b) {
  const int64_t D = w.encoder_cfg.d_model;                 // 1280
  const int64_t ds = w.downsample_factor;                  // 4
  const int64_t n_enc = static_cast<int64_t>(enc_out.size()) / D;
  // Pad n_enc up to a multiple of ds, then reshape (concat ds consecutive frames).
  const int64_t n_tok = (n_enc + ds - 1) / ds;
  const int64_t Kin = D * ds;                              // 5120
  const int64_t Ht = w.text_hidden;                        // 3072
  std::vector<float> reshaped(static_cast<size_t>(n_tok) * Kin, 0.0f);
  for (int64_t t = 0; t < n_enc; ++t)
    std::memcpy(&reshaped[static_cast<size_t>(t) * D], &enc_out[static_cast<size_t>(t) * D],
                static_cast<size_t>(D) * sizeof(float));

  Queue q = b.CreateQueue();
  auto up = [&](const std::vector<float>& f, std::vector<int64_t> shape) {
    auto bf = F32ToBf16Bits(f.data(), static_cast<int64_t>(f.size()));
    void* p = b.Alloc(bf.size() * sizeof(uint16_t));
    b.Copy(q, p, bf.data(), bf.size() * sizeof(uint16_t));
    Tensor t;
    t.data = p;
    t.dtype = DType::kBF16;
    t.device = q.device;
    t.rank = static_cast<int>(shape.size());
    int64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      t.shape[i] = shape[static_cast<size_t>(i)];
      t.stride[i] = stride;
      stride *= shape[static_cast<size_t>(i)];
    }
    return std::pair<void*, Tensor>{p, t};
  };
  auto x = up(reshaped, {n_tok, Kin});
  auto wi = up(w.adapter_w_in, {Ht, Kin});
  auto wo = up(w.adapter_w_out, {Ht, Ht});
  void* p_h = b.Alloc(static_cast<size_t>(n_tok) * Ht * sizeof(uint16_t));
  void* p_o = b.Alloc(static_cast<size_t>(n_tok) * Ht * sizeof(uint16_t));
  Tensor hbuf, obuf;
  auto mk = [&](void* p, int64_t r, int64_t c) {
    Tensor t;
    t.data = p; t.dtype = DType::kBF16; t.device = q.device; t.rank = 2;
    t.shape[0] = r; t.shape[1] = c; t.stride[0] = c; t.stride[1] = 1;
    return t;
  };
  hbuf = mk(p_h, n_tok, Ht);
  obuf = mk(p_o, n_tok, Ht);
  vt::MatmulBT(q, hbuf, x.second, wi.second);   // w_in: [n_tok,Kin] @ [Ht,Kin]^T -> [n_tok,Ht]
  vt::GeluErf(q, hbuf, hbuf);                   // nn.GELU()
  vt::MatmulBT(q, obuf, hbuf, wo.second);       // w_out: [n_tok,Ht] @ [Ht,Ht]^T -> [n_tok,Ht]

  std::vector<uint16_t> out_bits(static_cast<size_t>(n_tok) * Ht);
  b.Copy(q, out_bits.data(), p_o, out_bits.size() * sizeof(uint16_t));
  b.Synchronize(q);
  std::vector<float> out = Bf16BitsToF32(out_bits.data(), n_tok * Ht);

  b.Free(x.first); b.Free(wi.first); b.Free(wo.first); b.Free(p_h); b.Free(p_o);
  b.DestroyQueue(q);
  return out;
}

std::vector<int32_t> VoxtralGenerateGreedy(
    const std::vector<int32_t>& prompt_ids, const std::vector<float>& audio_embeds,
    int32_t audio_token_id, int32_t eos_token_id, const VoxtralWeights& weights,
    const HfConfig& config, Queue& queue, int max_new_tokens) {
  Backend& backend = vt::GetBackend(queue.device.type);
  Dev d{backend, queue};
  const int64_t H = config.hidden_size;
  const int64_t Hkv = config.num_key_value_heads;
  const int64_t Dh = config.head_dim;
  const int64_t T0 = static_cast<int64_t>(prompt_ids.size());

  // Audio placeholder mask.
  std::vector<bool> mask(static_cast<size_t>(T0), false);
  int64_t n_audio = 0;
  for (int64_t t = 0; t < T0; ++t)
    if (prompt_ids[static_cast<size_t>(t)] == audio_token_id) { mask[static_cast<size_t>(t)] = true; ++n_audio; }
  const int64_t N = static_cast<int64_t>(audio_embeds.size()) / (H > 0 ? H : 1);
  VT_CHECK(N == n_audio, "voxtral: audio_embeds rows != audio-token count");

  // KV caches: one big block per layer sized for T0 + max_new_tokens.
  //
  // ROAD-V1-MM lever #3 W2 (multimodal-speed.md §11/§12): round the single KV
  // block_size UP to a multiple of 16 so the pure-DECODE attention routes through
  // the FA2 varlen split-KV path (LaunchDecodeVarlenFA2Bf16) instead of the naive
  // scalar PagedAttentionKernel. The FA2 decode dispatch `fa2_decode_qwen3`
  // (cuda_paged_attn.cu:2620-2628) requires `block_size % 16 == 0` (line 2621);
  // Voxtral's head_dim-128 / GQA-32q-8kv / bf16 / causal decode matches every
  // other clause of that gate, so this single rounding is all that was blocking
  // FA2. The seq (T0+max_new) still fits ONE block, and slot == abs_idx is
  // unchanged (we only enlarge the block, never re-index), so prefill/decode KV
  // addressing is identical; only the decode-attention kernel changes (39x faster
  // attention → audio TPOT 59.4→38.2 ms/tok = 0.94x, BEATS vLLM 40.8 ms). The FA2
  // f32 reduction order takes a different-but-vLLM-valid branch at the pos-33
  // 4-way EXACT bf16 tie (teacher-force PASS, gap 0.0) — see the distributional
  // near-tie gate in test_voxtral_e2e and §11.4.
  const int64_t block_size = ((T0 + max_new_tokens + 8 + 15) / 16) * 16;
  const size_t kv_bytes =
      static_cast<size_t>(2 * block_size * Hkv * Dh) * vt::SizeOf(DType::kBF16);
  std::vector<std::shared_ptr<void>> kv_storage;
  std::vector<PagedKvCache> attn_kv;
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    void* p = backend.Alloc(kv_bytes);
    backend.Memset(queue, p, 0, kv_bytes);
    kv_storage.emplace_back(p, [&backend](void* q) { backend.Free(q); });
    PagedKvCache kv;
    kv.data = p; kv.dtype = DType::kBF16; kv.num_blocks = 1; kv.block_size = block_size;
    kv.num_kv_heads = Hkv; kv.head_size = Dh;
    attn_kv.push_back(kv);
  }

  // PREFILL: embed prompt ids, download, masked-scatter audio embeds, re-upload.
  std::vector<uint16_t> emb_bits(static_cast<size_t>(T0 * H));
  {
    DBuf ids(d, DType::kI32, {T0}, prompt_ids.data());
    DBuf emb(d, DType::kBF16, {T0, H});
    Tensor tab = ResidentWeight(d, weights.text.embed_tokens, {config.vocab_size, H});
    vt::Embedding(d.q, emb.t(), tab, ids.t());
    emb.Download(d, emb_bits.data());
  }
  std::vector<float> embeds = Bf16BitsToF32(emb_bits.data(), T0 * H);
  std::vector<float> aud = audio_embeds;
  RoundToBf16(aud);
  multimodal::Qwen3VLMergeMultimodal(embeds, T0, H, aud, mask);
  std::vector<uint16_t> merged_bits = F32ToBf16Bits(embeds.data(), T0 * H);

  std::vector<int32_t> pos_prefill(static_cast<size_t>(T0));
  for (int64_t t = 0; t < T0; ++t) pos_prefill[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  const CommonAttentionMetadata pm = StepMeta(T0, T0, 0);
  std::vector<int32_t> generated;
  int32_t next;
  {
    DBuf merged(d, DType::kBF16, {T0, H}, merged_bits.data());
    DBuf dlogits =
        ForwardLastLogits(d, merged.t(), pos_prefill, T0, pm, attn_kv, weights.text, config);
    next = ArgMaxDevice(d, dlogits);
  }
  generated.push_back(next);

  // DECODE: one token per step (no audio), 1-D position abs_idx.
  //
  // ROAD-V1-MM lever #3 W1 (multimodal-speed.md §9.5): route the pure-DECODE steps
  // through a CAPTURED decode graph (VoxtralDecodeGraph) instead of the eager
  // per-step ForwardLastLogits. Voxtral's Mistral/Llama text stack had NO decode-
  // graph class (only Qwen3.5-dense/MoE/DeepSeek did); this is the audio 1.52x
  // gap-closer, because the 3B decode is NOT bandwidth-floored (unlike the 27B),
  // so the per-step launch overhead the graph removes is a REAL slice of TPOT.
  // Single-seq => B=1 => PadToCaptureSize(1,1)=1 => S==B==1, the bit-identical
  // rebuild case: the graph captures the EXACT ForwardLastLogits op sequence the
  // eager decode ran, so the token stream is unchanged. The eager path stays
  // reachable via VT_MM_DECODE_EAGER=1 (default = graph; parity-enabler-as-default),
  // mirroring the 27B mm decode-graph brick.
  const bool decode_eager =
      (std::getenv("VT_MM_DECODE_EAGER") != nullptr &&
       std::string(std::getenv("VT_MM_DECODE_EAGER")) == "1");
  std::unique_ptr<VoxtralDecodeGraph> dgraph;
  if (!decode_eager)
    dgraph = std::make_unique<VoxtralDecodeGraph>(weights.text, config, d.q,
                                                  /*max_num_reqs=*/1);
  for (int step = 1; step < max_new_tokens; ++step) {
    if (next == eos_token_id) break;
    const int64_t abs_idx = T0 + (step - 1);
    const std::vector<int32_t> one = {next};
    const std::vector<int32_t> pos1 = {static_cast<int32_t>(abs_idx)};
    const int64_t seq_len = abs_idx + 1;
    const CommonAttentionMetadata dm = StepMeta(1, seq_len, abs_idx);
    if (dgraph) {
      // Graphed pure-decode step: the fed token is embedded ON DEVICE inside Step
      // (outside the capture region), the captured layer region replays, and the
      // returned [1,vocab] logits stay on device and feed GreedyArgmax with NO
      // full-vocab D2H.
      ForwardLogits fl = dgraph->Step(one, pos1, dm, attn_kv);
      DBuf ids(d, DType::kI64, {1});
      vt::GreedyArgmax(d.q, ids.t(), fl.device_tensor);
      int64_t id = 0;
      ids.Download(d, &id);
      next = static_cast<int32_t>(id);
    } else {
      // Eager fallback (VT_MM_DECODE_EAGER=1): the original per-step forward. The
      // fed token is embedded ON DEVICE and handed straight to the forward — no
      // D2H->H2D embed round-trip — and the greedy pick runs on-GPU.
      DBuf emb(d, DType::kBF16, {1, H});
      DBuf ids(d, DType::kI32, {1}, one.data());
      Tensor tab = ResidentWeight(d, weights.text.embed_tokens, {config.vocab_size, H});
      vt::Embedding(d.q, emb.t(), tab, ids.t());
      DBuf dlogits =
          ForwardLastLogits(d, emb.t(), pos1, 1, dm, attn_kv, weights.text, config);
      next = ArgMaxDevice(d, dlogits);
    }
    generated.push_back(next);
  }
  return generated;
}

}  // namespace vllm
