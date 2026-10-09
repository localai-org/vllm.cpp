// kev PointerHead -- option scorer (MODEL-KEV).
//
// Ported from jaredpalmer/kev kev/model.py @ 19dcae9b6e3e1a48200c5825aad9fc200d31e20a:
//   PointerHead  model.py:class PointerHead
//
// This is the HOST REFERENCE forward: a portable f32 implementation gated
// against the PyTorch model before any device path exists. It is not wired
// to the runner, the ABI or the server; that is Phase 3-5 of
// .agents/specs/kev.md.
//
// THE THINGS THIS ARCHITECTURE GETS WRONG QUIETLY are: (1) swapping the
// q/k projection order -- the dot product still produces a result but it is
// numerically wrong; (2) omitting the bias terms -- the model still runs
// on typical inputs; (3) using the wrong scale factor -- the logits are
// off by a constant factor but softmax still produces a distribution. Each
// is gated by a perturbation test.
#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vllm {
namespace kev {

// PointerHead configuration.
struct HeadParams {
  int64_t hidden_size = 1024;  // d: backbone hidden dimension
  int64_t head_dim = 256;      // dp: pointer dimension

  // Optional explicit scale. Unset (the default, and the only value the loader
  // ever produces) keeps the reference formula 1/sqrt(dp). It exists so a
  // caller can express "this implementation scales by the wrong dimension"
  // WITHOUT also resizing the projections: head_dim still sizes the [dp, d]
  // weights, only the scale factor changes. The wrong-scale perturbation test
  // in tests/vllm/models/test_kev.cpp uses it for exactly that simulation.
  std::optional<double> scale_override;

  double scale() const {
    return scale_override.has_value()
               ? *scale_override
               : 1.0 / std::sqrt(static_cast<double>(head_dim));
  }
};

// PointerHead weights: two Linear layers (q and k), each [dp, d] + [dp].
// Weight layout is row-major (PyTorch nn.Linear convention):
//   weight[j * d + i]  -- maps input i to output j
struct HeadWeights {
  std::vector<float> q_weight;  // [dp, d]
  std::vector<float> q_bias;    // [dp]
  std::vector<float> k_weight;  // [dp, d]
  std::vector<float> k_bias;    // [dp]
};

// Host-only f32 PointerHead forward.
//
// Computes: logits[k] = (k(h_opts[k]) . q(h_decide)) * scale
//
// h_decide:  [d]  -- hidden state at the <decide> token
// h_opts:    [K*d] -- flattened option hidden states, row-major
// n_options: K
//
// Returns logits [K] (pre-softmax).
std::vector<float> PointerHeadForward(
    const HeadParams& params, const HeadWeights& weights,
    const std::vector<float>& h_decide,
    const std::vector<float>& h_opts,
    int64_t n_options);

// Numerically stable softmax.
std::vector<float> Softmax(const std::vector<float>& logits);

// ── LoRA merge ───────────────────────────────────────────────────────
//
// Ported from the PEFT/LoRA merge formula used by jaredpalmer/kev:
// the adapter is applied once at load time, not served at runtime.
//   W' = W + scaling * (lora_B @ lora_A)
// where lora_A is [rank, in], lora_B is [out, rank], W is [out, in].
// All operands are row-major (PyTorch nn.Linear weight convention).
//
// THE THINGS THIS MERGE GETS WRONG QUIETLY are: (1) swapping the A/B
// order -- A@B has the wrong shape and would crash for non-square
// matrices, but for square matrices it silently produces wrong values;
// (2) omitting the scaling factor -- the delta is off by a constant but
// the model still runs; (3) using the wrong rank -- truncating the
// rank-r sum drops terms and produces a partial delta.  Each is gated
// by a perturbation test in test_kev.cpp.

// Merge a rank-r LoRA delta into a base weight (all f32).
//
// base:    [out * in] base weights (f32, row-major [out, in])
// lora_a:  [rank * in] (f32, row-major [rank, in])
// lora_b:  [out * rank] (f32, row-major [out, rank])
// Returns: [out * in] merged weights (f32)
std::vector<float> MergeLoraDelta(
    const std::vector<float>& base,
    const std::vector<float>& lora_a,
    const std::vector<float>& lora_b,
    int64_t out, int64_t in, int64_t rank, float scaling);

// bf16 conversion helpers (matching vt::BF16ToF32 / vt::F32ToBF16).
// bf16 is the top 16 bits of f32; F32ToBf16 uses round-to-nearest-even.
float Bf16ToF32(uint16_t b);
uint16_t F32ToBf16(float f);

// ── Phase 4: sequence construction + readout ─────────────────────────
//
// Ported from jaredpalmer/kev kev/model.py::encode() and
// kev/model.py::DecisionModel._readout() / PointerHead.forward().
//
// The encoding packs one question as a causal row:
//   [<|fim_prefix|>] state_tokens [<|fim_middle|>] instr_tokens
//   [<|box_start|>] opt0 [<|box_end|>] [<|box_start|>] opt1 [<|box_end|>]
//   ... [<|fim_suffix|>]
// The positions are continuous (0, 1, 2, ...).  decide_idx points at
// the <|fim_suffix|> token (the pointer query).  opt_idx[k] points at
// the <|box_end|> token of option k (the pointer key).
//
// THE THINGS THIS ENCODER GETS WRONG QUIETLY are: (1) swapping the
// special token roles -- the model still runs but the pointer reads the
// wrong positions; (2) discontinuous positions -- DeltaNet recurrence
// silently shifts; (3) pointing opt_idx at <|box_start|> instead of
// <|box_end|> -- the key vector is one token off.  Each is gated by a
// perturbation test.

// Qwen special-token IDs used as delimiters.  These reuse existing
// rarely-used Qwen tokens so no embedding rows are added (model.py:18).
struct KevSpecialTokens {
  int32_t fim_prefix_id = 0;  // <|fim_prefix|>  -- state start
  int32_t fim_middle_id = 0;  // <|fim_middle|>  -- question start
  int32_t box_start_id = 0;  // <|box_start|>   -- option start
  int32_t box_end_id = 0;    // <|box_end|>     -- option end
  int32_t fim_suffix_id = 0;  // <|fim_suffix|>  -- decide token
};

// Encoded question: token IDs, positions, and readout indices.
struct KevEncoded {
  std::vector<int32_t> token_ids;
  std::vector<int32_t> positions;
  int32_t decide_idx = 0;          // absolute index of <|fim_suffix|>
  std::vector<int32_t> opt_idx;    // absolute indices of <|box_end|> per option
};

// Encode one question as a causal row (model.py::encode, rows_of form).
//
//   special:       the five delimiter token IDs
//   state_tokens:  pre-tokenized state text (NO <|fim_prefix|>)
//   instr_tokens:  pre-tokenized instruction text (NO <|fim_middle|>)
//   option_tokens: pre-tokenized option texts, one vector per option
//
// Returns token_ids, positions (continuous from 0), decide_idx, opt_idx.
KevEncoded KevEncodeQuestion(
    const KevSpecialTokens& special,
    const std::vector<int32_t>& state_tokens,
    const std::vector<int32_t>& instr_tokens,
    const std::vector<std::vector<int32_t>>& option_tokens);

// Readout: extract hidden states from a ForwardHidden output and apply
// PointerHead + softmax to get per-option probabilities.
//
//   hidden:      [n_indices, D] f32, row-major.  Row 0 = h_decide
//                (at <|fim_suffix|>), rows 1..K = h_opts (at <|box_end|>).
//   D:           hidden dimension (must match params.hidden_size)
//   n_options:   K (number of options)
//   temperature: calibration scalar (1.0 = raw; the fitted value divides
//                logits before softmax, matching PointerHead.forward eval)
//
// Returns: [K] probabilities.
std::vector<float> KevReadout(
    const HeadParams& params, const HeadWeights& weights,
    const std::vector<float>& hidden,
    int64_t D, int64_t n_options, float temperature = 1.0f);

}  // namespace kev
}  // namespace vllm
