// Qwen4-Exp (`Qwen/Qwen3.8-Flash-Next`) W5f — `Qwen4ExpTextModel::Forward`, the
// LAYER LOOP, and the first thing on this row that composes more than one block.
//
// Issue [#2031](https://github.com/mudler/vllm.cpp/issues/2031), reconciliation
// issue [#2336](https://github.com/mudler/vllm.cpp/issues/2336), campaign issue
// [#1978](https://github.com/mudler/vllm.cpp/issues/1978), spec
// `.agents/specs/qwen4-exp-flash-next.md`.
//
// ─── WHAT WAS MISSING, IN THE SPEC'S OWN WORDS ───────────────────────────────
// `## Now`: "**WHAT IS ACTUALLY LEFT, AFTER W5e-2: the LOOP, and only the
// loop.** … The count of missing BLOCK SEAMS is ZERO and each one resolves to a
// symbol … `RunQwen4ExpQsaBlock` / `RunQwen4ExpQsaBlockPaged` (W5b-5, W5d-3),
// `RunQwen4ExpMoeBlock` (W5d-4), `RunQwen4ExpPleBlock` (W5e-2 …)". This file
// calls all four, plus the two hyper-connection ops and the terminal mixer.
//
// ─── ORACLE ──────────────────────────────────────────────────────────────────
// vLLM registers `qwen4_exp` at NO revision, so under AGENTS.md "When vLLM has
// no implementation" the ALGORITHM oracle is transformers **5.16.0**, this row's
// accepted lane pin (`.agents/oracles/transformers.md`, sha256
// `77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459`), at
// `models/qwen4_exp/modeling_qwen4_exp.py`. Every anchor below was re-derived by
// READING that file rather than relayed — #2336's cited range was off by one at
// both ends and W5e-1 had to correct it, so relaying is not a shortcut here.
//
//   `Qwen4ExpTextModel.forward`:
//     :1415  hidden_states = inputs_embeds                  (embed_tokens gather)
//     :1416  position_embeddings = self.rotary_emb(...)      (built ONCE per step)
//     :1417  hidden_states = hidden_states.repeat(1, 1, hc_count)   THE WIDEN
//     :1419  for layer_idx, decoder_layer in enumerate(self.layers[:L])
//     :1430  hidden_states = self.hyper_connection_mixer(hidden_states)
//
//   `Qwen4ExpTextDecoderLayer.forward`:
//     :1218  hidden_states = hidden_states + self.ple(...)   PLE FIRST, on the
//            hc-WIDE stream, BEFORE the attention hyper-connection
//     :1222  hidden_states, hyper_input, injection_weights = attn_hyper_connection(hs)
//     :1224  hidden_states = self.linear_attn(...)           iff linear_attention
//     :1228  hidden_states, _ = self.self_attn(...)          otherwise
//     :1236  injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
//     :1237  hidden_states = hyper_input + injection.flatten(-2)
//     :1239  hidden_states, hyper_input, injection_weights = mlp_hyper_connection(hs)
//     :1240  hidden_states = self.mlp(hidden_states)
//     :1242-3  the SECOND injection and write-back, identical to :1236-1237
//
// FOUR ORDERING FACTS A BLOCK-LEVEL GATE CANNOT SEE, and they are why this file
// needs an end-to-end golden of its own:
//
//   1. THE PLE LAYER RUNS FIRST IN ITS DECODER LAYER, not last and not at the
//      model level. It reads the hc-WIDE stream and its output is ADDED to that
//      stream (:1218), so it is upstream of both hyper-connection sites of the
//      layer it sits on and of every later layer.
//   2. THE WIDEN IS `repeat`, NOT a tile. `repeat(1, 1, hc)` on `[B, T, H]`
//      produces `out[t, j*H + h] = in[t, h]` — each branch is a COPY of the
//      whole hidden state. `vt::IndexSelect` over a `[T*hc]` index of `i / hc`
//      is exactly that and needs no new op (#2336 §4).
//   3. `hyper_input` IS THE RAW STREAM. `Qwen4ExpTextGatedResidual` returns it
//      un-normed and it is the raw stream the write-back adds to, so the norm
//      inside the op is NOT applied to the residual path.
//   4. THERE IS NO FINAL RMSNorm. The mixer's `hc_norm` is the last
//      normalization in the model; a port that copies this tree's DeepSeek-V4
//      tail inserts one that does not exist (`Qwen4ExpWeights::mixer`).
//
// ─── THE GDN ARM IS THE QWEN3.5 BLOCK, MEASURED AND NOT ASSUMED ──────────────
// `Qwen4ExpTextGatedDeltaNet` and `Qwen3_5GatedDeltaNet` are BYTE-IDENTICAL at
// the pin — the whole class, `__init__` and `forward` — except for ONE argument:
//
//     qwen4_exp:  self.norm = RMSNormGated(head_v_dim, eps=...,
//                     activation=config.output_gate_type or config.hidden_act)
//     qwen3_5:    self.norm = RMSNormGated(head_v_dim, eps=...)
//
// (measured by diffing the two classes out of the installed 5.16.0 package).
// So `RunGdnBlockPaged` IS this architecture's linear-attention layer and a
// second GDN implementation would be the parallel path AGENTS.md forbids. What
// the difference costs is stated in `Qwen4ExpGdnHfConfig` below, and it is not
// cosmetic: the released `config.json` says `output_gate_type: "sigmoid"`, this
// tree's shared reader defaults to `"silu"`, no shape check can see the swap,
// and [#489](https://github.com/mudler/vllm.cpp/issues/489) is that exact defect
// in another model.
//
// ─── SCOPE, AND WHAT IS NOT HERE ─────────────────────────────────────────────
// MULTI-STEP DECODE IS HERE NOW (W5k), and this paragraph used to say it was not.
// It read that `multi_kv` "is refused by name" so the entry point "serves a
// SINGLE-SHOT PREFILL at `past_len == 0`". W5j narrowed that engine refusal to a
// model-declared capability and W5k settled the last two blockers against the
// running lane oracle, so on the BY-NAME channel the QSA indexer side cache lives
// in the engine's group-2 pages and the PLE layer's conv ring and n-gram history
// live in the recurrent group's third and fourth published states. A `past_len >
// 0` step returns a token.
//
// WHAT IS STILL NOT HERE is the POSITIONAL arm's second step, and the reason is
// that arm's own: nothing publishes those three states there, so they are
// per-call scratch and a per-call buffer is zeroed on entry. Refused by name, on
// the same predicate that routes. The loop itself takes the caches as operands
// and has never had either limit.
//
// ONE SEQUENCE PER CALL, as `RunQwen4ExpQsaBlockPaged` takes one: its
// `block_table` is i32 `[1, max_pages]` and `RunQwen4ExpPleBlock`'s n-gram
// history is per sequence. `num_reqs > 1` is owed with the same seam work.
//
// CPU ONLY, for the same reason `RunQwen4ExpPleBlock` is: the n-gram id build is
// a host int64 hash by construction. The spec's `## Owed` carries the device arm.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_QWEN4_EXP_FORWARD_H_
#define VLLM_MODEL_EXECUTOR_MODELS_QWEN4_EXP_FORWARD_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"     // Dev
#include "vllm/model_executor/models/qwen3_5.h"               // GdnStateCache
#include "vllm/model_executor/models/qwen3_5_weights.h"       // GdnLayerWeights
#include "vllm/model_executor/models/qwen4_exp.h"             // Qwen4ExpParams
#include "vllm/model_executor/models/qwen4_exp_ple_block.h"   // Qwen4ExpPleCaches
#include "vllm/model_executor/models/qwen4_exp_qsa_block.h"   // Qwen4ExpQsaPagedCaches
#include "vllm/model_executor/models/qwen4_exp_weights.h"     // Qwen4ExpWeights
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"                       // CommonAttentionMetadata
#include "vllm/v1/attention/backends/gdn_attn.h"              // GDNAttentionMetadata
#include "vt/tensor.h"

namespace vllm {

// Every persistent state the 48 decoder layers read and write, in DECODER-LAYER
// order within each kind.
//
// THREE VECTORS AND NOT ONE, because the three kinds are indexed differently and
// a single vector would have to encode which. `gdn[i]` is the i-th
// `kLinearAttention` layer counted from 0, `qsa[i]` the i-th
// `kQwenSparseAttention` layer, and `ple[i]` the i-th entry of
// `Qwen4ExpPleParams::layer_ids_zero_based` — which is upstream's own indexing
// (`config.ple_layer_ids.index(layer_idx + 1)`, :1202) and NOT the decoder layer
// index. The loop refuses a vector whose length disagrees with the config rather
// than indexing past it.
struct Qwen4ExpForwardCaches {
  std::vector<GdnStateCache> gdn;            // one per kLinearAttention layer
  std::vector<Qwen4ExpQsaPagedCaches> qsa;   // one per kQwenSparseAttention layer
  std::vector<Qwen4ExpPleCaches> ple;        // one per PLE layer, in ple id order
};

// Owning device-resident output of the whole text tower: `[T, hidden_size]` plus
// the shared_ptr that returns its pool block to the DevicePool when the last
// reference drops. Mirrors `GdnBlockOutput` / `Qwen4ExpQsaBlockOutput` /
// `MoeBlockOutput` / `Qwen4ExpPleBlockOutput`, so the composing caller keeps ONE
// release idiom across every level of this model.
//
// IT IS THE MIXER'S OUTPUT, NOT LOGITS. `Qwen4ExpTextModel` carries no
// `lm_head` — that is `Qwen4ExpForCausalLM` — so the `lm_head` GEMM belongs to
// the registry hook and not to the loop.
// ─── THE MODEL'S ONE STREAM DTYPE (W5k, #2031) ──────────────────────────────
// AGENTS.md "Inherit vLLM defaults": vLLM resolves ONE model dtype and every
// layer inherits it. This is that value for `qwen4_exp`, exported because THREE
// places now have to agree on it and a literal in each is three facts that can
// drift: the layer loop's activations, the PLE conv ring the recurrent group
// publishes (`MakeQwen4ExpKVCache`'s `conv_dtype`), and the per-call scratch the
// positional arm allocates.
//
// The ring MUST equal it — that is not a house convention but upstream's own
// construction, which types each cache slot from the tensor that first reaches
// it (`cache_utils.py:1019-1023` over the `hidden_states` at
// `modeling_qwen4_exp.py:1157-1159`). `RunQwen4ExpPleBlock` enforces the equality;
// this constant is what lets the producers satisfy it from one place instead of
// three.
inline constexpr vt::DType kQwen4ExpStreamDType = vt::DType::kBF16;

struct Qwen4ExpTextModelOutput {
  vt::Tensor tensor;              // [T, hidden_size] at the stream dtype
  std::shared_ptr<void> storage;  // owns the pool block (Pool().Put on release)
};

// The `HfConfig` the 36 Gated DeltaNet layers run under.
//
// DERIVED FROM THE MODEL'S OWN CONFIG, NOT SYNTHESIZED FROM `Qwen4ExpParams`,
// and that is the whole reason this function exists rather than a local struct
// literal. `ParseQwen4ExpParams` VALIDATES `output_gate_type` against upstream's
// `{sigmoid, silu}` (configuration_qwen4_exp.py:193-195) and stores NO FIELD for
// it, so `Qwen4ExpParams` cannot answer what the checkpoint asked for. The value
// lives on `HfConfig::output_gate_type`, which `Qwen4ExpHfConfigFromGguf` sets
// to `"sigmoid"` for this architecture, and `qwen3_5.cpp` reads it there to pick
// the GDN output gate's activation. A projection that rebuilt the config from
// the params alone would silently fall back to `"silu"` — plausible output, no
// crash, no shape error, and invisible to every gate that does not run the
// oracle. That is #489 in another model, and the golden gate this wave lands
// separates the two activations.
//
// `source` is refused by name when its GDN geometry disagrees with `p`: the two
// describe one checkpoint, and a disagreement means the caller composed a config
// from two models.
HfConfig Qwen4ExpGdnHfConfig(const Qwen4ExpParams& p, const HfConfig& source);

// Adapt one layer's loaded GDN weights onto the shared `GdnLayerWeights` seam.
//
// A FIELD RENAME OVER ZERO-COPY VIEWS, which is what #2336 §3 measured and what
// this function is: the `qwen4exp` and `qwen3_5` GGUF loaders read the same
// tensor names and land on the same `[N, K]`-with-`nk` orientation, so nothing
// is transposed, reordered or reallocated here — the returned `OwnedTensor`s
// BORROW the loader's bytes through `BorrowWholeOwnedTensor`.
//
// THE SHARING IS LOAD-BEARING AND IT USED TO BE A LIE (issue #2476). The same
// sentence stood above a body that spelled the pass-through as assignment, and
// `OwnedTensor`'s implicit copy deep-copies an owned buffer. `ResidentWeight`'s
// host-alias arm hands a device kernel `bytes.data()` directly, so the copy
// became the GEMM's operand and was freed when the caller's scope closed, with
// the GEMM still queued. `g` is NON-CONST because taking a keep-alive converts
// the loader's owned buffer into a shared read-only one in place.
//
// `in_proj_ba` and `in_proj_qkvz` stay EMPTY, which is exact parity with
// qwen3_5's own GGUF path and not an oversight: those two are the safetensors
// loaders' merged owners, and leaving them empty is what keeps the split fields
// live. It does mean `vt::GdnPackedDecode` never fires on this arm — a
// performance ceiling this row inherits from the GGUF path it shares, recorded
// in the spec's `## Owed` rather than worked around here.
//
// Refuses by name on a shape that disagrees with `p`, because every check it
// could skip is a wrong answer rather than a crash: the towers are all rank-2
// and a swapped pair has the right element count.
GdnLayerWeights Qwen4ExpGdnBlockWeights(Qwen4ExpGdnWeights& g,
                                        const Qwen4ExpParams& p);

// `Qwen4ExpTextModel.forward` (:1337-1433), end to end.
//
//   w          the whole model's weights, as the loader produced them. Every
//              gamma is the RAW HuggingFace value and every consumer adds the 1
//              (#2218); `linear_attn.norm.weight` is the architecture's single
//              exception and is passed through untouched.
//   config     the model's own `HfConfig`. Read for the GDN arm's
//              `output_gate_type` and for the rope parameters; see
//              `Qwen4ExpGdnHfConfig`.
//   token_ids  HOST [T], this step's token ids. Host because the PLE layer's
//              n-gram hash is a host int64 computation over token IDS.
//   positions  HOST [T] i32, the position of each new token.
//   attn_meta / gdn_meta  the step's metadata, exactly as the runner builds it.
//              The GDN step upload is built ONCE here and shared across all 36
//              linear layers, which is the reason `BuildGdnStepInputs` is a
//              separate call at all (`qwen3_5_gdn_block.h`).
//   caches     see the struct.
//   past_len   how many tokens the sequence already holds. ZERO is the PLE
//              layer's SEEDING predicate, not merely a position.
Qwen4ExpTextModelOutput Qwen4ExpTextModelForward(
    dense_attn::Dev d, Qwen4ExpWeights& w, const HfConfig& config,
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const v1::GDNAttentionMetadata& gdn_meta,
    const Qwen4ExpForwardCaches& caches, int64_t past_len);

namespace detail {

// Re-read `VT_Q4EXP_LAYER_FP` and put the fingerprint's step and tap counters
// back to zero. TEST HOOK, and the only caller is the case that gates the
// instrument.
//
// It exists because the budget is read ONCE per process. Every case in a doctest
// binary shares that process, so whichever forward runs first spends the budget
// and the case that sets the variable measures nothing and reports a pass --
// which is `assertions: 0 ... SUCCESS!` wearing the instrument's own clothes.
// Same shape and same reason as `detail::ExpertStreamSetForceFallback`.
void Qwen4ExpLayerFpResetForTest();

// Lines the instrument has printed since the last reset. The counted property a
// gate asserts on, rather than grepping the process's stderr for a prefix.
int64_t Qwen4ExpLayerFpTapsForTest();

// `VT_Q4EXP_LOGITS=<topk>`: tap the step's logits. `LayerFp` stops at `out`, so
// without this the `lm_head` decode - int5 with a group of 64 - and the final
// projection have no model-level observation at all, and the acceptance gate
// this row designs cannot see them.
//
// Off unless the variable is set, because a tap that runs regardless is a cost on
// every forward whose line stops being evidence that the run asked for it. The
// budget is counted in FORWARDS like the layer tap's, and read from
// `VT_Q4EXP_LOGITS_STEPS` (default 8) so a long server run cannot print forever.
//
// Emitted once per forward, tagged `q4logit`: the top-`k` `(id, value)` pairs of
// the first logits row - the row the sampler asked for first - plus `vmax`,
// `vmin`, `sumabs` and EIGHT fixed-seed random projections over the whole
// `[rows, vocab]` buffer. The projections are the same construction, and the same
// reason, as the layer tap's: `sumabs` is sign-insensitive and under-reports a
// real divergence by a median 75-140x, while `S w_i x_i` is a linear functional
// of the difference and cannot cancel.
void Qwen4ExpLogitsTap(dense_attn::Dev d, const vt::Tensor& logits);

// TEST HOOKS, same shape and same reason as the layer tap's: the budget is read
// once per process, so a case that sets the variable after another forward has
// already run would measure nothing and report a pass.
void Qwen4ExpLogitsTapResetForTest();
int64_t Qwen4ExpLogitsTapTapsForTest();
int64_t Qwen4ExpLogitsTapLastTop1ForTest();

}  // namespace detail

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_QWEN4_EXP_FORWARD_H_
