// CLM (Contrastive-LM) System 1 decision model: head weights, text
// construction and answers (MODEL-CLM, .agents/specs/clm.md).
//
// CLM is a bi-encoder. The state text and each candidate text go SEPARATELY
// through the frozen Qwen3-8B backbone; the last token's post-norm hidden
// state is L2-normalized (vLLM's LAST pooling with normalization), projected
// by the state head or the action head, L2-normalized again, and a question's
// distribution is softmax(scale * cos / temperature).
//
// Reference: Contrastive-LM/CLM @ bb42c6c5bf914fd449bed2f6ca65be80602cb1f7,
// src/clm/heads.py (make_head, HeadPair), src/clm/schema.py (to_text,
// state_text, candidates, build_pairs, answer_from_probs), src/clm/engine.py
// (Engine.answer), src/clm/embedder.py (l2).
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm {

class SafetensorsFile;

namespace clm {

// A request the reference refuses (HTTP 400, VLLM_ERR_INVALID_ARGUMENT). The
// message is the reference's ValueError text.
class RequestError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

enum class Activation { kGelu, kRelu, kSilu };

// The checkpoint's cfg (heads.py HeadPair._load), carried in config.json as
// clm_* keys by scripts/convert-clm.py.
struct HeadParams {
  int64_t hidden_size = 4096;   // encoder width (cfg hidden_size)
  int64_t width = 1536;         // cfg width
  int64_t depth = 3;            // cfg depth: inp + (depth - 2) hidden blocks + out
  int64_t proj_dim = 512;       // cfg projection_dim
  Activation activation = Activation::kGelu;
  bool layernorm = true;
  bool residual = false;
  double logit_scale = 0.0;     // the checkpoint's raw logit_scale (a log)
};

// exp(logit_scale).clamp(max=100.0), in float32 as HeadPair._load does it.
float ClmScale(double logit_scale);

// One hidden block of make_head: Linear(width, width), then LayerNorm(width)
// when cfg layernorm is set.
struct HiddenBlock {
  std::vector<float> w, b;            // [width, width], [width]
  std::vector<float> norm_w, norm_b;  // [width] each; empty without LayerNorm
};

// One make_head module, tensors in PyTorch nn.Linear layout [out, in].
struct MlpHeadWeights {
  std::vector<float> inp_w, inp_b;  // [width, hidden], [width]
  std::vector<HiddenBlock> hidden;  // depth - 2 blocks
  std::vector<float> out_w, out_b;  // [proj, width], [proj]
};

struct HeadWeights {
  MlpHeadWeights state_head;
  MlpHeadWeights action_head;
};

// Read <prefix>.inp.*, <prefix>.hidden.<i>.*, <prefix>.norms.<i>.* and
// <prefix>.out.* for prefix state_head and action_head (F32, the .pt state-dict
// names), checking every shape against `params`. Throws std::runtime_error
// naming the first missing or mis-shaped tensor.
HeadWeights LoadHeadWeights(const std::vector<SafetensorsFile>& shards,
                            const HeadParams& params);

// Read the clm_* keys of config.json. Throws std::runtime_error naming a
// missing key; nothing is defaulted.
HeadParams ParseHeadParams(const nlohmann::json& raw, int64_t hidden_size);

// make_head's forward, then F.normalize: act(inp(x)); per hidden block
// h = act(norm(lin(x))), x = residual ? x + h : h; out(x); L2-normalize.
// `x` is the pooled (already L2-normalized) encoder embedding [hidden].
std::vector<float> ClmMlpHeadForward(const MlpHeadWeights& hw,
                                     const HeadParams& params,
                                     const std::vector<float>& x);

// The pooled embedding the head sees: the last-token hidden state,
// L2-normalized (embedder.py l2, x / (|x| + 1e-12)).
std::vector<float> PoolEmbedding(const std::vector<float>& last_hidden);

// schema.py to_text: a state, instructions or description as plain text.
std::string ToText(const nlohmann::ordered_json& x, int indent = 0);

// schema.py build_pairs for one question.
struct Pair {
  std::string state_text;
  std::vector<std::string> keys;
  std::vector<std::string> candidates;
};
Pair BuildPair(const nlohmann::ordered_json& state,
               const nlohmann::ordered_json& question);

// schema.py answer_from_logits: softmax, then the typed answer.
nlohmann::ordered_json AnswerFromLogits(const nlohmann::ordered_json& question,
                                        const std::vector<std::string>& keys,
                                        const std::vector<double>& logits);

// Engine.answer over any encoder. `embed` returns the LAST-token post-norm
// hidden state of a text (not normalized) and adds the text's token count to
// `*tokens`. Identical texts in one request are encoded once, as the
// reference embedder's dict.fromkeys does. Returns {"answers", "usage"}.
// Throws RequestError on a request the reference refuses.
using EmbedFn =
    std::function<std::vector<float>(const std::string& text, int64_t* tokens)>;
nlohmann::ordered_json Answer(const HeadWeights& heads, const HeadParams& params,
                              const nlohmann::ordered_json& body,
                              const EmbedFn& embed);

}  // namespace clm
}  // namespace vllm
