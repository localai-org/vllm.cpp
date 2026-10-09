// kev PointerHead host reference forward (MODEL-KEV).
//
// Ported from jaredpalmer/kev kev/model.py @ 19dcae9b6e3e1a48200c5825aad9fc200d31e20a:
//   class PointerHead(nn.Module):
//       def __init__(self, d, dp=256):
//           self.q = nn.Linear(d, dp)
//           self.k = nn.Linear(d, dp)
//           self.scale = 1 / math.sqrt(dp)
//
//       def forward(self, h_decide, h_opts):  # [d], [K,d] -> logits [K]
//           return (self.k(h_opts) @ self.q(h_decide)) * self.scale
//
// nn.Linear(d, dp) computes y = x @ W^T + b, where W is [dp, d].
// So q(h_decide) = h_decide @ W_q^T + b_q  ->  [dp]
//    k(h_opts)   = h_opts   @ W_k^T + b_k  ->  [K, dp]
// logits[k] = dot(k_vec[k], q_vec) * scale

#include "vllm/model_executor/models/kev.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace vllm {
namespace kev {

std::vector<float> PointerHeadForward(
    const HeadParams& params, const HeadWeights& weights,
    const std::vector<float>& h_decide,
    const std::vector<float>& h_opts,
    int64_t n_options) {
  const int64_t d = params.hidden_size;
  const int64_t dp = params.head_dim;
  const double scale = params.scale();

  // SHAPE VALIDATION FIRST. The loader reads head_dim from the config
  // (kev_registry.cpp) and the weights from head.safetensors independently,
  // so a config/checkpoint mismatch must be a loud refusal, never an
  // out-of-bounds read: the loops below index q_weight[j * d] / q_bias[j] for
  // j up to dp-1, and dp comes from the params. (The wrong-scale perturbation
  // test reached exactly that OOB read before this guard existed — ASan
  // heap-buffer-overflow at the `w[i]` read, ISSUE-LOCAL-01M4FK4ATBNXVV9VGEN5Q25HZ1.)
  if (d <= 0 || dp <= 0 || n_options < 0) {
    throw std::invalid_argument(
        "kev PointerHead: hidden_size, head_dim must be positive and "
        "n_options must not be negative, got hidden_size=" +
        std::to_string(d) + " head_dim=" + std::to_string(dp) +
        " n_options=" + std::to_string(n_options));
  }
  const size_t weight_elems = static_cast<size_t>(dp) * static_cast<size_t>(d);
  if (weights.q_weight.size() != weight_elems ||
      weights.k_weight.size() != weight_elems) {
    throw std::invalid_argument(
        "kev PointerHead: q/k weights must be [head_dim, hidden_size] = [" +
        std::to_string(dp) + ", " + std::to_string(d) + "] = " +
        std::to_string(weight_elems) + " elements, got q_weight " +
        std::to_string(weights.q_weight.size()) + " and k_weight " +
        std::to_string(weights.k_weight.size()));
  }
  const size_t bias_elems = static_cast<size_t>(dp);
  if (weights.q_bias.size() != bias_elems ||
      weights.k_bias.size() != bias_elems) {
    throw std::invalid_argument(
        "kev PointerHead: q/k biases must be [head_dim] = [" +
        std::to_string(dp) + "] = " + std::to_string(bias_elems) +
        " elements, got q_bias " +
        std::to_string(weights.q_bias.size()) + " and k_bias " +
        std::to_string(weights.k_bias.size()));
  }
  if (h_decide.size() < static_cast<size_t>(d)) {
    throw std::invalid_argument(
        "kev PointerHead: h_decide must carry at least hidden_size=" +
        std::to_string(d) + " elements, got " +
        std::to_string(h_decide.size()));
  }
  const size_t opts_elems =
      static_cast<size_t>(n_options) * static_cast<size_t>(d);
  if (h_opts.size() < opts_elems) {
    throw std::invalid_argument(
        "kev PointerHead: h_opts must carry n_options * hidden_size = " +
        std::to_string(opts_elems) + " elements for n_options=" +
        std::to_string(n_options) + ", got " + std::to_string(h_opts.size()));
  }

  // q_vec[j] = sum_i(h_decide[i] * q_weight[j * d + i]) + q_bias[j]
  std::vector<float> q_vec(static_cast<size_t>(dp), 0.0F);
  for (int64_t j = 0; j < dp; ++j) {
    const float* w = &weights.q_weight[static_cast<size_t>(j * d)];
    double acc = 0.0;
    for (int64_t i = 0; i < d; ++i) {
      acc += static_cast<double>(h_decide[static_cast<size_t>(i)]) *
             static_cast<double>(w[i]);
    }
    q_vec[static_cast<size_t>(j)] = static_cast<float>(
        acc + static_cast<double>(weights.q_bias[static_cast<size_t>(j)]));
  }

  // logits[k] = (sum_j(k_vec[k][j] * q_vec[j])) * scale
  // where k_vec[k][j] = sum_i(h_opts[k*d+i] * k_weight[j*d+i]) + k_bias[j]
  std::vector<float> logits(static_cast<size_t>(n_options), 0.0F);
  for (int64_t k = 0; k < n_options; ++k) {
    const float* opt = &h_opts[static_cast<size_t>(k * d)];
    double dot = 0.0;
    for (int64_t j = 0; j < dp; ++j) {
      const float* w = &weights.k_weight[static_cast<size_t>(j * d)];
      double kj = 0.0;
      for (int64_t i = 0; i < d; ++i) {
        kj += static_cast<double>(opt[i]) * static_cast<double>(w[i]);
      }
      kj += static_cast<double>(weights.k_bias[static_cast<size_t>(j)]);
      dot += kj * static_cast<double>(q_vec[static_cast<size_t>(j)]);
    }
    logits[static_cast<size_t>(k)] = static_cast<float>(dot * scale);
  }

  return logits;
}

std::vector<float> Softmax(const std::vector<float>& logits) {
  if (logits.empty()) return {};
  float max_val = *std::max_element(logits.begin(), logits.end());
  std::vector<float> out(logits.size());
  double sum = 0.0;
  for (size_t i = 0; i < logits.size(); ++i) {
    out[i] = std::exp(logits[i] - max_val);
    sum += static_cast<double>(out[i]);
  }
  float inv = static_cast<float>(1.0 / sum);
  for (size_t i = 0; i < logits.size(); ++i) {
    out[i] *= inv;
  }
  return out;
}

// ── LoRA merge ───────────────────────────────────────────────────────

std::vector<float> MergeLoraDelta(
    const std::vector<float>& base,
    const std::vector<float>& lora_a,
    const std::vector<float>& lora_b,
    int64_t out, int64_t in, int64_t rank, float scaling) {
  std::vector<float> merged(static_cast<size_t>(out * in), 0.0F);
  for (int64_t j = 0; j < out; ++j) {
    for (int64_t i = 0; i < in; ++i) {
      // delta = scaling * sum_k(lora_b[j, k] * lora_a[k, i])
      double delta = 0.0;
      for (int64_t k = 0; k < rank; ++k) {
        delta +=
            static_cast<double>(lora_b[static_cast<size_t>(j * rank + k)]) *
            static_cast<double>(lora_a[static_cast<size_t>(k * in + i)]);
      }
      merged[static_cast<size_t>(j * in + i)] =
          base[static_cast<size_t>(j * in + i)] +
          static_cast<float>(static_cast<double>(scaling) * delta);
    }
  }
  return merged;
}

float Bf16ToF32(uint16_t b) {
  uint32_t bits = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

uint16_t F32ToBf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  // Round to nearest even: add bias + LSB bit.
  uint32_t lsb = (bits >> 16) & 1u;
  uint32_t rounding_bias = 0x7FFFu + lsb;
  bits += rounding_bias;
  return static_cast<uint16_t>(bits >> 16);
}

// ── Phase 4: sequence construction + readout ──────────────────────────

KevEncoded KevEncodeQuestion(
    const KevSpecialTokens& special,
    const std::vector<int32_t>& state_tokens,
    const std::vector<int32_t>& instr_tokens,
    const std::vector<std::vector<int32_t>>& option_tokens) {
  KevEncoded enc;
  std::vector<int32_t>& ids = enc.token_ids;
  std::vector<int32_t>& pos = enc.positions;
  int32_t p = 0;

  // State: [<|fim_prefix|>] + state_tokens  (seg=0 in reference)
  ids.push_back(special.fim_prefix_id);
  pos.push_back(p++);
  for (int32_t tok : state_tokens) {
    ids.push_back(tok);
    pos.push_back(p++);
  }

  // Question: [<|fim_middle|>] + instr_tokens
  ids.push_back(special.fim_middle_id);
  pos.push_back(p++);
  for (int32_t tok : instr_tokens) {
    ids.push_back(tok);
    pos.push_back(p++);
  }

  // Options: [<|box_start|>] + opt_tokens + [<|box_end|>] per option
  for (const auto& opt : option_tokens) {
    ids.push_back(special.box_start_id);
    pos.push_back(p++);
    for (int32_t tok : opt) {
      ids.push_back(tok);
      pos.push_back(p++);
    }
    ids.push_back(special.box_end_id);
    pos.push_back(p++);
    enc.opt_idx.push_back(static_cast<int32_t>(ids.size()) - 1);
  }

  // Decide: [<|fim_suffix|>]
  ids.push_back(special.fim_suffix_id);
  pos.push_back(p++);
  enc.decide_idx = static_cast<int32_t>(ids.size()) - 1;

  return enc;
}

std::vector<float> KevReadout(
    const HeadParams& params, const HeadWeights& weights,
    const std::vector<float>& hidden,
    int64_t D, int64_t n_options, float temperature) {
  // Row 0 = h_decide, rows 1..K = h_opts
  std::vector<float> h_decide(hidden.begin(), hidden.begin() + D);

  std::vector<float> h_opts(static_cast<size_t>(n_options) * D);
  for (int64_t k = 0; k < n_options; ++k) {
    const float* src = hidden.data() + (k + 1) * D;
    std::copy_n(src, D, h_opts.data() + k * D);
  }

  std::vector<float> logits = PointerHeadForward(params, weights, h_decide, h_opts, n_options);

  if (temperature != 1.0f) {
    for (float& z : logits) z /= temperature;
  }

  return Softmax(logits);
}

}  // namespace kev
}  // namespace vllm
