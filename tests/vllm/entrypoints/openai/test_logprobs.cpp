// ROAD-V1-C7 SAMPLE-LOGPROBS payload gate. Two parts:
//   (1) The OpenAI serialization builders (BuildCompletionLogProbs /
//       BuildChatLogprobs + their to_json) gated against a hand-computed vLLM
//       0.26 oracle — the EXACT payload vLLM's _create_completion_logprobs
//       (completion/serving.py:652-741) / _create_chat_logprobs +
//       _get_top_logprobs (chat_completion/serving.py:1114-1210) emit given the
//       same per-position Logprob dicts. RED-first: a wrong N-vs-N+1 top-k
//       cutoff, a missing sampled entry, wrong bytes, or an off text_offset
//       fails.
//   (2) The LogprobsProcessor (vllm/v1/engine/logprobs.py) consuming an
//       EngineCoreOutput.new_logprobs LogprobsTensors and producing the
//       SampleLogprobs + cumulative_logprob the OutputProcessor attaches.
//
// The logprob VALUES themselves are exact given identical logits: the Sampler's
// gather_logprobs (log_softmax + rank) is gated separately in
// tests/vllm/v1/sample/test_sampler.cpp; this file gates the OUTPUT plumbing
// and serialization schema/values on top of it.
#include "vllm/entrypoints/openai/serving_utils.h"

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/logprobs.h"
#include "vllm/v1/engine/logprobs.h"
#include "vllm/v1/engine/types.h"
#include "vllm/v1/outputs.h"

using nlohmann::json;
using vllm::Logprob;
using vllm::LogprobsOnePosition;
using vllm::SampleLogprobs;
using vllm::entrypoints::openai::BuildChatLogprobs;
using vllm::entrypoints::openai::BuildCompletionLogProbs;

namespace {

// One position with a KNOWN [sampled | top-k] dict, built via the ported
// AppendLogprobsForNextPosition so the dict order + dedup are exercised.
// sampled == 7 (rank 1), top-k = [7:-0.1, 2:-1.5, 4:-2.0]; decoded G/H/I.
LogprobsOnePosition MakePos() {
  SampleLogprobs scratch;
  vllm::AppendLogprobsForNextPosition(
      scratch, /*token_ids=*/{7, 7, 2, 4},
      /*logprobs=*/{-0.1f, -0.1f, -1.5f, -2.0f},
      /*decoded=*/{std::string("G"), std::string("G"), std::string("H"),
                   std::string("I")},
      /*rank=*/1, /*num_logprobs=*/3);
  return std::move(scratch.front());
}

}  // namespace

TEST_CASE("logprobs: dict order dedups the sampled-in-topk token, keeps 3 keys") {
  LogprobsOnePosition p = MakePos();
  REQUIRE(p.order.size() == 3);          // 7 appears once (dedup), then 2, 4
  CHECK(p.order[0] == 7);
  CHECK(p.order[1] == 2);
  CHECK(p.order[2] == 4);
  // Python dict last-write-wins on value: 7's rank is its top-1 rank (1).
  CHECK(p.find(7)->rank.value() == 1);
  CHECK(p.find(2)->rank.value() == 2);
  CHECK(p.find(4)->rank.value() == 3);
}

TEST_CASE("logprobs: completion payload matches the vLLM N+1 oracle") {
  SampleLogprobs sl;
  sl.push_back(MakePos());
  sl.push_back(MakePos());  // 2nd position to exercise text_offset accumulation
  // token_ids = the sampled ids (column 0) per position.
  auto lp = BuildCompletionLogProbs(/*token_ids=*/{7, 7}, sl,
                                    /*num_output_top_logprobs=*/2);
  json j = lp;

  // tokens + token_logprobs: the sampled token "G" and its logprob -0.1.
  CHECK(j["tokens"] == json::array({"G", "G"}));
  CHECK(j["token_logprobs"] == json::array({-0.1f, -0.1f}));
  // text_offset: first 0, then + len("G") == 1.
  CHECK(j["text_offset"] == json::array({0, 1}));
  // top_logprobs: N+1 == 3 entries (sampled + 2 alternatives).
  REQUIRE(j["top_logprobs"].size() == 2);
  const json& top0 = j["top_logprobs"][0];
  CHECK(top0.size() == 3);
  CHECK(top0["G"] == -0.1f);
  CHECK(top0["H"] == -1.5f);
  CHECK(top0["I"] == -2.0f);
}

TEST_CASE("logprobs: completion top-k honors the count (N=1 -> N+1=2 entries)") {
  SampleLogprobs sl;
  sl.push_back(MakePos());
  auto lp = BuildCompletionLogProbs(/*token_ids=*/{7}, sl,
                                    /*num_output_top_logprobs=*/1);
  json j = lp;
  const json& top0 = j["top_logprobs"][0];
  CHECK(top0.size() == 2);  // sampled + 1 alternative
  CHECK(top0.contains("G"));
  CHECK(top0.contains("H"));
  CHECK_FALSE(top0.contains("I"));
}

TEST_CASE("logprobs: chat payload matches the vLLM oracle (N entries, bytes)") {
  SampleLogprobs sl;
  sl.push_back(MakePos());
  auto lp = BuildChatLogprobs(/*token_ids=*/{7}, sl,
                              /*num_output_top_logprobs=*/2);
  json j = lp;
  REQUIRE(j["content"].size() == 1);
  const json& c0 = j["content"][0];
  CHECK(c0["token"] == "G");
  CHECK(c0["logprob"] == -0.1f);
  CHECK(c0["bytes"] == json::array({71}));  // 'G' == 0x47
  // chat top_logprobs keeps N == 2 entries (NOT N+1), 0-based cutoff.
  REQUIRE(c0["top_logprobs"].size() == 2);
  CHECK(c0["top_logprobs"][0]["token"] == "G");
  CHECK(c0["top_logprobs"][0]["logprob"] == -0.1f);
  CHECK(c0["top_logprobs"][0]["bytes"] == json::array({71}));
  CHECK(c0["top_logprobs"][1]["token"] == "H");
  CHECK(c0["top_logprobs"][1]["bytes"] == json::array({72}));  // 'H' == 0x48
}

TEST_CASE("logprobs: -9999 floor clamps -inf logprobs (OpenAI JSON-safe)") {
  LogprobsOnePosition p;
  Logprob s;
  s.logprob = -1e30f;  // effectively -inf
  s.rank = 1;
  s.decoded_token = std::string("Z");
  p.put(9, s);
  SampleLogprobs sl;
  sl.push_back(p);
  json j = BuildCompletionLogProbs(/*token_ids=*/{9}, sl, /*n=*/0);
  CHECK(j["token_logprobs"][0] == -9999.0f);
}

// ─── LogprobsProcessor: EngineCoreOutput.new_logprobs -> SampleLogprobs ───────
TEST_CASE("logprobs: LogprobsProcessor accumulates sample logprobs + cumulative") {
  vllm::SamplingParams sp;
  sp.logprobs = 2;  // request 2 sample logprobs
  vllm::v1::LogprobsProcessor proc =
      vllm::v1::LogprobsProcessor::FromNewRequest(/*tokenizer=*/nullptr, sp);

  // Build a 1-position LogprobsTensors row [sampled=7 | top 7,2,4], width = 2+1.
  vllm::v1::LogprobsTensors lt;
  lt.num_positions = 1;
  lt.num_tokens_per_position = 3;
  lt.logprob_token_ids = {7, 2, 4};
  lt.logprobs = {-0.1f, -1.5f, -2.0f};
  lt.selected_token_ranks = {1};

  vllm::v1::EngineCoreOutput eco;
  eco.new_logprobs = lt;
  proc.update_from_output(eco);
  proc.update_from_output(eco);  // second decode step

  REQUIRE(proc.logprobs().has_value());
  CHECK(proc.logprobs()->size() == 2);            // one dict per step
  const LogprobsOnePosition& pos0 = (*proc.logprobs())[0];
  CHECK(pos0.order.size() == 3);
  CHECK(pos0.find(7)->logprob == doctest::Approx(-0.1f));
  CHECK(pos0.find(7)->rank.value() == 1);
  // cumulative_logprob sums the sampled (column-0) logprob each step.
  REQUIRE(proc.cumulative_logprob().has_value());
  CHECK(*proc.cumulative_logprob() == doctest::Approx(-0.2));
}

TEST_CASE("logprobs: LogprobsProcessor inert when logprobs not requested") {
  vllm::SamplingParams sp;  // no logprobs / prompt_logprobs
  vllm::v1::LogprobsProcessor proc =
      vllm::v1::LogprobsProcessor::FromNewRequest(nullptr, sp);
  vllm::v1::EngineCoreOutput eco;  // no new_logprobs
  proc.update_from_output(eco);
  CHECK_FALSE(proc.logprobs().has_value());
  CHECK_FALSE(proc.prompt_logprobs().has_value());
  CHECK_FALSE(proc.cumulative_logprob().has_value());
}

// ISSUE-LOCAL-01M3SE6RVKD6SCMA2YBS7F8X0R: a byte-level token that is half of a
// UTF-8 character decodes, upstream, through tokenizer.decode([id]) with
// errors="replace", so its logprob token is U+FFFD and its bytes are
// EF BF BD. Our decoded_token keeps the raw byte, and serializing it threw
// inside json::dump, so a served chat request with top_logprobs answered 500
// with an empty body as soon as a partial-UTF-8 token reached the top-k.
TEST_CASE("logprobs: a partial UTF-8 token serializes as U+FFFD, not a throw") {
  SampleLogprobs scratch;
  vllm::AppendLogprobsForNextPosition(
      scratch, /*token_ids=*/{7, 7, 2},
      /*logprobs=*/{-0.1f, -0.1f, -1.5f},
      /*decoded=*/{std::string("G"), std::string("G"), std::string("\xF0\x9F")},
      /*rank=*/1, /*num_logprobs=*/2);
  auto chat = BuildChatLogprobs(/*token_ids=*/{7}, scratch,
                                /*num_output_top_logprobs=*/2);
  json j = chat;
  std::string dumped;
  CHECK_NOTHROW(dumped = j.dump());
  CHECK(j["content"][0]["top_logprobs"][1]["token"] == "\xEF\xBF\xBD");
  CHECK(j["content"][0]["top_logprobs"][1]["bytes"] == json::array({239, 191, 189}));

  auto completion = BuildCompletionLogProbs(/*token_ids=*/{7}, scratch,
                                            /*num_output_top_logprobs=*/1);
  json jc = completion;
  CHECK_NOTHROW(dumped = jc.dump());
  CHECK(jc["top_logprobs"][0].contains("\xEF\xBF\xBD"));
}
