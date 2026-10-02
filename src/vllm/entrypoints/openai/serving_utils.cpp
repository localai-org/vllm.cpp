// vllm.cpp original — see serving_utils.h.
#include <cstdlib>
#include <optional>
#include "vllm/entrypoints/openai/serving_utils.h"
#include "vllm/outputs.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <utility>

namespace vllm::entrypoints::openai {

namespace {

// _get_decoded_token (generate/base/serving.py:253-271, return_as_token_id
// path off): prefer the LogprobsProcessor's decoded_token; fall back to the
// `token_id:N` placeholder when detokenization was disabled (decoded_token is
// None) — our serving layer holds no tokenizer, so this stands in for the
// `tokenizer.decode([token_id])` branch (recorded deviation).
std::string DecodedToken(const vllm::Logprob& lp, int32_t token_id) {
  // Upstream's decoded token is a Python str from tokenizer.decode([id]),
  // errors="replace": a byte-level token that is part of a UTF-8 character is
  // U+FFFD there. Ours keeps the raw byte, which json::dump refuses, so it is
  // replaced the same way (ISSUE-LOCAL-01M3SE6RVKD6SCMA2YBS7F8X0R).
  if (lp.decoded_token.has_value()) return SanitizeUtf8(*lp.decoded_token);
  return "token_id:" + std::to_string(token_id);
}

// list(token.encode("utf-8", errors="replace")): our strings are already UTF-8
// bytes, so map each byte to its unsigned value.
std::vector<int> Utf8Bytes(const std::string& s) {
  std::vector<int> out;
  out.reserve(s.size());
  for (unsigned char c : s) out.push_back(static_cast<int>(c));
  return out;
}

constexpr float kLogprobFloor = -9999.0f;  // JSON-serializable -inf floor.

}  // namespace

StreamUsageSelection ShouldIncludeUsage(
    const std::optional<StreamOptions>& stream_options,
    bool enable_force_include_usage) {
  if (enable_force_include_usage) return {true, true};
  if (!stream_options.has_value()) return {};
  const bool include_usage = stream_options->include_usage;
  return {include_usage,
          include_usage && stream_options->continuous_usage_stats};
}

std::string SanitizeUtf8(const std::string& s) {
  // Mirrors detokenizer.cpp::LossyStep: consume one valid UTF-8 character or one
  // maximal invalid subpart per step. Valid characters (and a literal U+FFFD)
  // are copied verbatim; every invalid subpart is emitted as one U+FFFD.
  static constexpr char kReplacement[] = "\xEF\xBF\xBD";
  std::string out;
  out.reserve(s.size());
  const size_t n = s.size();
  size_t i = 0;
  while (i < n) {
    const uint8_t b0 = static_cast<uint8_t>(s[i]);
    if (b0 < 0x80) {  // ASCII
      out.push_back(s[i]);
      ++i;
      continue;
    }
    size_t need;        // continuation bytes required by this lead
    uint8_t lo = 0x80;  // valid range for the FIRST continuation byte
    uint8_t hi = 0xBF;  // (encodes the overlong/surrogate/range rules)
    if (b0 >= 0xC2 && b0 <= 0xDF) {
      need = 1;
    } else if (b0 == 0xE0) {
      need = 2;
      lo = 0xA0;
    } else if (b0 >= 0xE1 && b0 <= 0xEC) {
      need = 2;
    } else if (b0 == 0xED) {
      need = 2;
      hi = 0x9F;
    } else if (b0 >= 0xEE && b0 <= 0xEF) {
      need = 2;
    } else if (b0 == 0xF0) {
      need = 3;
      lo = 0x90;
    } else if (b0 >= 0xF1 && b0 <= 0xF3) {
      need = 3;
    } else if (b0 == 0xF4) {
      need = 3;
      hi = 0x8F;
    } else {  // stray continuation byte (0x80-0xC1) or invalid lead (0xF5-0xFF)
      out += kReplacement;
      ++i;
      continue;
    }
    size_t len = 1;
    for (size_t k = 0; k < need; ++k) {
      if (i + len >= n) break;
      const uint8_t b = static_cast<uint8_t>(s[i + len]);
      const uint8_t clo = (k == 0) ? lo : uint8_t{0x80};
      const uint8_t chi = (k == 0) ? hi : uint8_t{0xBF};
      if (b < clo || b > chi) break;
      ++len;
    }
    if (len != need + 1) {  // truncated / invalid multibyte: one maximal subpart
      out += kReplacement;
      i += len;
      continue;
    }
    out.append(s, i, len);  // a valid character — copied byte-for-byte
    i += len;
  }
  return out;
}

CompletionLogProbs BuildCompletionLogProbs(const std::vector<int32_t>& token_ids,
                                           const vllm::SampleLogprobs& top_logprobs,
                                           int num_output_top_logprobs,
                                           int initial_text_offset) {
  CompletionLogProbs out;
  int last_token_len = 0;
  const std::size_t n = token_ids.size();
  for (std::size_t i = 0; i < n; ++i) {
    const int32_t token_id = token_ids[i];
    // A position may lack a logprobs dict (e.g. an echoed prompt token with no
    // prompt_logprobs): emit token + null logprob/top_logprobs (serving.py:677).
    const vllm::LogprobsOnePosition* step =
        i < top_logprobs.size() ? &top_logprobs[i] : nullptr;
    const vllm::Logprob* step_token =
        step != nullptr ? step->find(token_id) : nullptr;
    if (step == nullptr || step->empty() || step_token == nullptr) {
      out.tokens.push_back("token_id:" + std::to_string(token_id));
      out.token_logprobs.push_back(std::nullopt);
      out.top_logprobs.push_back(std::nullopt);
    } else {
      const std::string token = DecodedToken(*step_token, token_id);
      out.tokens.push_back(token);
      out.token_logprobs.push_back(std::max(step_token->logprob, kLogprobFloor));
      // Keep the first num_output_top_logprobs+1 entries (OpenAI's N+1 rule):
      // enumerate(step.items()) yields [sampled, top1, ...]; keep index <= N.
      std::map<std::string, float> top;
      int idx = 0;
      for (const int32_t tid : step->order) {
        if (idx > num_output_top_logprobs) break;
        const vllm::Logprob& lp = step->entries.at(tid);
        top[DecodedToken(lp, tid)] = std::max(lp.logprob, kLogprobFloor);
        ++idx;
      }
      out.top_logprobs.push_back(std::move(top));
    }
    const std::string& emitted = out.tokens.back();
    out.text_offset.push_back(out.text_offset.empty()
                                  ? initial_text_offset
                                  : out.text_offset.back() + last_token_len);
    last_token_len = static_cast<int>(emitted.size());
  }
  return out;
}

namespace {

// _get_top_logprobs (chat_completion/serving.py:1114-1139): keep the first
// `top_logprobs` entries (0-based cutoff; -1 => all) in dict-iteration order.
std::vector<ChatCompletionLogProb> ChatTopLogprobs(
    const vllm::LogprobsOnePosition& step, int top_logprobs) {
  std::vector<ChatCompletionLogProb> out;
  int i = 0;
  for (const int32_t tid : step.order) {
    if (!(top_logprobs == -1 || i < top_logprobs)) break;
    const vllm::Logprob& lp = step.entries.at(tid);
    ChatCompletionLogProb e;
    e.token = DecodedToken(lp, tid);
    e.logprob = std::max(lp.logprob, kLogprobFloor);
    e.bytes = Utf8Bytes(e.token);
    out.push_back(std::move(e));
    ++i;
  }
  return out;
}

}  // namespace

ChatCompletionLogProbs BuildChatLogprobs(const std::vector<int32_t>& token_ids,
                                         const vllm::SampleLogprobs& top_logprobs,
                                         int num_output_top_logprobs) {
  ChatCompletionLogProbs out;
  std::vector<ChatCompletionLogProbsContent> content;
  const std::size_t n = token_ids.size();
  for (std::size_t i = 0; i < n; ++i) {
    const int32_t token_id = token_ids[i];
    const vllm::LogprobsOnePosition* step =
        i < top_logprobs.size() ? &top_logprobs[i] : nullptr;
    const vllm::Logprob* step_token =
        step != nullptr ? step->find(token_id) : nullptr;
    ChatCompletionLogProbsContent c;
    if (step == nullptr || step->empty() || step_token == nullptr) {
      // None branch (serving.py:1159-1174): token + its bytes, no logprob/top.
      c.token = "token_id:" + std::to_string(token_id);
      c.bytes = Utf8Bytes(c.token);
    } else {
      c.token = DecodedToken(*step_token, token_id);
      c.logprob = std::max(step_token->logprob, kLogprobFloor);
      // bytes = None when decoded_token is None, else its UTF-8 bytes (:1189).
      if (step_token->decoded_token.has_value()) {
        c.bytes = Utf8Bytes(*step_token->decoded_token);
      }
      c.top_logprobs = ChatTopLogprobs(*step, num_output_top_logprobs);
    }
    content.push_back(std::move(c));
  }
  out.content = std::move(content);
  return out;
}

std::vector<vllm::CompletionOutput> SelectBestOf(
    std::vector<vllm::CompletionOutput> outputs, int return_n) {
  // INERT: nothing to trim (the default best_of == n path lands here and returns
  // the outputs — and their original indices — byte-for-byte unchanged).
  if (return_n <= 0 ||
      static_cast<int>(outputs.size()) <= return_n) {
    return outputs;
  }
  // Rank by DESCENDING cumulative logprob, stable (ties keep engine order); an
  // output without a cumulative logprob sorts last.
  std::stable_sort(outputs.begin(), outputs.end(),
                   [](const vllm::CompletionOutput& a,
                      const vllm::CompletionOutput& b) {
                     const double no_lp =
                         -std::numeric_limits<double>::infinity();
                     const double la = a.cumulative_logprob.value_or(no_lp);
                     const double lb = b.cumulative_logprob.value_or(no_lp);
                     return la > lb;
                   });
  outputs.resize(static_cast<std::size_t>(return_n));
  for (int i = 0; i < return_n; ++i) {
    outputs[static_cast<std::size_t>(i)].index = i;
  }
  return outputs;
}



bool AssignSseWaitResult(std::optional<vllm::RequestOutput> ready,
                         vllm::RequestOutput& out, std::string& chunk) {
  if (ready.has_value()) {
    out = std::move(*ready);
    return true;
  }
  // Pure SSE comment — never prefix/suffix a data frame here.
  chunk = kSsePingFrame;
  return false;
}

int SsePingIntervalSec() {
  // DEFAULT OFF (#931). Opt in with a positive VT_SERVER_SSE_PING_S; <=0, unset
  // and unparsable all disable, which takes both streams down the blocking
  // get_output() path #316 recorded as byte-identical to the behaviour before
  // it.
  //
  // The keepalive exists because a long MoE prefill produces no body bytes, so
  // a proxy or Hermes inactivity_timeout can drop the stream before the first
  // token. That is a real deployment problem, and the frame is legal SSE. It is
  // still not the default, for one reason: vLLM emits NO comment frame from any
  // streaming endpoint, so every OpenAI-compatible client written against vLLM
  // has never had to parse one — and vLLM's own `vllm bench serve` is such a
  // client. It strips each network chunk before parsing
  // (benchmarks/lib/endpoint_request_func.py:207), destroying the "\n\n"
  // separator, and resynchronises only on a `data: ` prefix (:48); one comment
  // frame ahead of a request's first token leaves a bare ":" in its buffer that
  // nothing clears, so it reports "Never received a valid chunk to calculate
  // TTFT" and counts the request FAILED while we answer 200, finish normally
  // and log nothing.
  //
  // The requests that reach a keepalive are by construction the SLOWEST, so the
  // default deleted our own worst latencies from measurements taken with vLLM's
  // client: 93/96 at 27B c16 (#577) and then 5/6 at c1 and 36/48 at c8 on
  // Qwen3.8-27B (#931, #915), where the failed requests' imputed TTFT was 92-94 s
  // against a 4.2 s p99 among the successes.
  const char* e = std::getenv("VT_SERVER_SSE_PING_S");
  if (e == nullptr || e[0] == '\0') return 0;
  char* end = nullptr;
  long v = std::strtol(e, &end, 10);
  if (end == e) return 0;
  if (v <= 0) return 0;
  if (v > 600) v = 600;
  return static_cast<int>(v);
}


// clamp_prompt_logprobs — vllm/entrypoints/generate/base/serving.py:305-317.
void ClampPromptLogprobs(std::optional<vllm::PromptLogprobs>& prompt_logprobs) {
  if (!prompt_logprobs.has_value()) return;
  for (auto& position : *prompt_logprobs) {
    if (!position.has_value()) continue;
    for (auto& [token_id, logprob] : position->entries) {
      (void)token_id;
      if (logprob.logprob == -std::numeric_limits<float>::infinity()) {
        logprob.logprob = -9999.0f;
      }
    }
  }
}

}  // namespace vllm::entrypoints::openai
