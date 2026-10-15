// Ported from: vllm/v1/sample/sampler.py @ e24d1b24 (Sampler.forward / sample /
// gather_logprobs) + cases mirrored from tests/v1/sample/test_sampler.py
// (test_sampler_repetition_penalty, test_sampler_min_tokens, the greedy/random
// merge). The Sampler assembles the ordered pipeline over the [num_reqs, vocab]
// f32 logits (raw-logprobs snapshot BEFORE mutation -> allowed-ids -> bad-words
// -> non-argmax-invariant procs -> penalties -> sample{greedy snapshot;
// temperature; argmax-invariant min_p; top_k_top_p; random; where(temp<eps)}
// -> gather logprobs). Greedy is the bit-exact parity gate; random is
// distribution-correct (peaked distributions make the picked token deterministic
// for the assertions here). CPU is the correctness gate; a CUDA-Queue run is
// dgx-pending (the Sampler runs on whatever Queue the ops are registered for).
#include "vllm/v1/sample/sampler.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <vector>

#include "vllm/v1/sample/metadata.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

using vllm::v1::MinTokensState;
using vllm::v1::Sampler;
using vllm::v1::SamplingMetadata;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }

Tensor Logits(std::vector<float>& v, int64_t n, int64_t vocab) {
  return Tensor::Contiguous(v.data(), DType::kF32, Cpu(), {n, vocab});
}
}  // namespace

TEST_CASE("PR11 seeded sampling follows accepted positions across batch permutations") {
  Sampler sampler;
  Queue q = Q();
  constexpr int V = 31;
  auto draw = [&](const std::vector<uint64_t>& positions) {
    std::vector<float> logits(positions.size() * V, 0);
    auto t = Logits(logits, positions.size(), V);
    SamplingMetadata sm; sm.all_greedy = false; sm.all_random = true;
    sm.temperature = std::vector<float>(positions.size(), 1);
    sm.output_token_positions = positions;
    for (size_t i = 0; i < positions.size(); ++i) sm.generators[int(i)] = 719;
    return sampler.forward(q, t, sm).sampled_token_ids;
  };
  const auto first = draw({0, 1, 2, 3});
  const auto reordered = draw({3, 1, 0, 2});
  CHECK(first[0] == reordered[2]); CHECK(first[1] == reordered[1]);
  CHECK(first[2] == reordered[3]); CHECK(first[3] == reordered[0]);
  for (int i = 0; i < 4; ++i) CHECK(draw({uint64_t(i)})[0] == first[i]);
  CHECK(first == draw({0, 1, 2, 3}));
  CHECK_FALSE((first[0] == first[1] && first[0] == first[2] && first[0] == first[3]));
}

TEST_CASE("PR11 unseeded sampling advances its stream") {
  Sampler sampler; Queue q = Q(); std::vector<int32_t> tokens;
  for (int step = 0; step < 32; ++step) {
    std::vector<float> logits(128, 0); auto t = Logits(logits, 1, 128);
    SamplingMetadata sm; sm.all_greedy = false; sm.all_random = true;
    sm.temperature = std::vector<float>{1};
    tokens.push_back(sampler.forward(q, t, sm).sampled_token_ids[0][0]);
  }
  CHECK_FALSE(std::all_of(tokens.begin(), tokens.end(), [&](int32_t token) { return token == tokens.front(); }));
}

// ---------------------------------------------------------------------------
// All-greedy batch: argmax, bit-exact; no logprobs when not requested.
TEST_CASE("Sampler: all-greedy batch returns the argmax per row, no logprobs") {
  std::vector<float> logits = {0.1f, 5.0f, 0.2f, 0.3f,   // argmax 1
                               9.0f, 1.0f, 1.0f, 2.0f};  // argmax 0
  Tensor tl = Logits(logits, 2, 4);
  SamplingMetadata sm;
  sm.all_greedy = true;
  sm.all_random = false;
  sm.max_num_logprobs = std::nullopt;

  Sampler sampler;
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);

  REQUIRE(out.sampled_token_ids.size() == 2);
  REQUIRE(out.sampled_token_ids[0].size() == 1);
  CHECK(out.sampled_token_ids[0][0] == 1);
  CHECK(out.sampled_token_ids[1][0] == 0);
  CHECK_FALSE(out.logprobs_tensors.has_value());
}

// ---------------------------------------------------------------------------
// Mixed greedy + random: the temp<eps where-merge picks greedy for the greedy
// row and random for the random row.
//
// This case is DELIBERATELY constructed so an INVERTED merge predicate (picking
// random where temp<eps) fails deterministically — a peaked greedy row cannot do
// that, because greedy-argmax always equals the mode the random path samples
// from, so on a peaked row both branches coincide. Instead row 0 is UNIFORM over
// 8 tokens: its greedy argmax is index 0 (lowest-index tie-break), but its random
// draw under the fixed seed lands elsewhere (asserted below via an all-random
// reference). So merge-correct => row 0 == 0; merge-inverted => row 0 == the
// random draw (!= 0) => the test bites.
TEST_CASE("Sampler: mixed batch merges greedy (temp<eps) and random per row") {
  const int64_t V = 8;
  // Row 0 (greedy, temp 0): uniform -> argmax 0 by tie-break; random draw != 0.
  // Row 1 (random, temp 1): logit 100 at index 2 -> softmax ~= one-hot(2).
  std::vector<float> logits(2 * V, 0.0f);
  logits[V + 2] = 100.0f;
  SamplingMetadata sm;
  sm.all_greedy = false;
  sm.all_random = false;
  sm.temperature = std::vector<float>{0.0f, 1.0f};
  sm.generators[0] = 424242;    // fixed seed for the (greedy) row 0's random path
  sm.generators[1] = 20260704;  // per-request seed for the random row
  sm.max_num_logprobs = std::nullopt;

  Sampler sampler;
  Queue q = Q();
  Tensor tl = Logits(logits, 2, V);
  auto out = sampler.forward(q, tl, sm);

  CHECK(out.sampled_token_ids[0][0] == 0);  // greedy row -> argmax (tie-break)
  CHECK(out.sampled_token_ids[1][0] == 2);  // random row -> the dominant token

  // Reference: what the RANDOM path alone produces for row 0 under the same seed.
  // It must differ from the greedy argmax (0), proving the merge above genuinely
  // selected greedy — an inverted predicate would have emitted this value.
  std::vector<float> row0(logits.begin(), logits.begin() + V);
  SamplingMetadata rnd;
  rnd.all_greedy = false;
  rnd.all_random = true;
  rnd.temperature = std::vector<float>{1.0f};
  rnd.generators[0] = 424242;
  rnd.max_num_logprobs = std::nullopt;
  Tensor trow0 = Logits(row0, 1, V);
  auto ref = sampler.forward(q, trow0, rnd);
  CHECK(ref.sampled_token_ids[0][0] != 0);  // random draw diverges from greedy
}

// ---------------------------------------------------------------------------
// Penalties change the argmax: a repetition penalty on the leading token pushes
// it below the runner-up so greedy flips. (test_sampler_repetition_penalty.)
TEST_CASE("Sampler: repetition penalty flips the greedy argmax") {
  // logits [1.0, 0.9]; token 0 is in the output -> rep 2.0 divides it (0.5),
  // now below token 1 (0.9) -> argmax flips 0 -> 1.
  std::vector<float> logits = {1.0f, 0.9f};
  Tensor tl = Logits(logits, 1, 2);
  SamplingMetadata sm;
  sm.all_greedy = true;
  sm.all_random = false;
  sm.no_penalties = false;
  sm.prompt_token_ids = std::vector<std::vector<int32_t>>{{}};
  sm.output_token_ids = {{0}};
  sm.presence_penalties = {0.0f};
  sm.frequency_penalties = {0.0f};
  sm.repetition_penalties = {2.0f};
  sm.max_num_logprobs = std::nullopt;

  Sampler sampler;
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);
  CHECK(out.sampled_token_ids[0][0] == 1);
}

// ---------------------------------------------------------------------------
// min_tokens masks eos before the floor so it is never sampled.
TEST_CASE("Sampler: min_tokens masks eos below the floor") {
  // eos (token 0) has the highest logit but is masked while output_len < floor.
  std::vector<float> logits = {5.0f, 1.0f, 2.0f};
  Tensor tl = Logits(logits, 1, 3);
  SamplingMetadata sm;
  sm.all_greedy = true;
  sm.all_random = false;
  MinTokensState st;
  st.min_tokens = 5;
  st.stop_token_ids = {0};
  sm.min_tokens[0] = st;
  sm.output_token_ids = {{}};  // length 0 < 5 -> eos masked
  sm.max_num_logprobs = std::nullopt;

  Sampler sampler;
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);
  CHECK(out.sampled_token_ids[0][0] == 2);  // token 0 masked, next-highest is 2
}

// ---------------------------------------------------------------------------
// top_k restricts the support: with k=1 only the argmax survives, so even the
// random draw must return it.
TEST_CASE("Sampler: top_k=1 restricts random support to the argmax") {
  std::vector<float> logits = {0.0f, 1.0f, 3.0f, 2.0f};  // argmax index 2
  Tensor tl = Logits(logits, 1, 4);
  SamplingMetadata sm;
  sm.all_greedy = false;
  sm.all_random = true;
  sm.temperature = std::vector<float>{1.0f};
  sm.top_k = std::vector<int32_t>{1};
  sm.generators[0] = 7;
  sm.max_num_logprobs = std::nullopt;

  Sampler sampler;
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);
  CHECK(out.sampled_token_ids[0][0] == 2);
}

// ---------------------------------------------------------------------------
// Logprobs: top-k + sampled token logprob + ranks; the [n, k+1] concat shape.
// rank of the max-logprob token is 1 (batched_count_greater_than uses >=).
TEST_CASE("Sampler: logprobs gather (top-k, sampled col, ranks) all-greedy") {
  // logits [3,1,2,0] -> argmax token 0 (also the max logprob). num_logprobs=2.
  std::vector<float> logits = {3.0f, 1.0f, 2.0f, 0.0f};
  Tensor tl = Logits(logits, 1, 4);
  SamplingMetadata sm;
  sm.all_greedy = true;
  sm.all_random = false;
  sm.max_num_logprobs = 2;

  Sampler sampler;
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);

  REQUIRE(out.sampled_token_ids[0][0] == 0);
  REQUIRE(out.logprobs_tensors.has_value());
  const auto& lt = *out.logprobs_tensors;
  CHECK(lt.num_positions == 1);
  CHECK(lt.num_tokens_per_position == 3);  // k + 1
  REQUIRE(lt.logprob_token_ids.size() == 3);
  // Column 0 is the sampled token; then the top-2 (token 0, token 2).
  CHECK(lt.logprob_token_ids[0] == 0);
  CHECK(lt.logprob_token_ids[1] == 0);
  CHECK(lt.logprob_token_ids[2] == 2);
  // log_softmax reference for [3,1,2,0].
  const float lse = 3.0f + std::log(std::exp(0.0f) + std::exp(-2.0f) +
                                    std::exp(-1.0f) + std::exp(-3.0f));
  CHECK(lt.logprobs[0] == doctest::Approx(3.0f - lse));  // sampled (token 0)
  CHECK(lt.logprobs[1] == doctest::Approx(3.0f - lse));  // top-1 (token 0)
  CHECK(lt.logprobs[2] == doctest::Approx(2.0f - lse));  // top-2 (token 2)
  // Sampled token is the max logprob -> rank 1.
  CHECK(lt.selected_token_ranks[0] == 1);
}

// ---------------------------------------------------------------------------
// Ranks are computed over the RAW (pre-penalty) logprobs snapshot, so when the
// penalty flips the sampled token off the raw max the rank is > 1.
TEST_CASE("Sampler: sampled-token rank over raw logprobs when penalty flips it") {
  // Raw logits [1.0, 0.9]: raw max is token 0. Rep penalty flips greedy to
  // token 1. rank(token1) = #{raw lp >= lp[1]} = 2 (token 0 and token 1).
  std::vector<float> logits = {1.0f, 0.9f};
  Tensor tl = Logits(logits, 1, 2);
  SamplingMetadata sm;
  sm.all_greedy = true;
  sm.all_random = false;
  sm.no_penalties = false;
  sm.prompt_token_ids = std::vector<std::vector<int32_t>>{{}};
  sm.output_token_ids = {{0}};
  sm.presence_penalties = {0.0f};
  sm.frequency_penalties = {0.0f};
  sm.repetition_penalties = {2.0f};
  sm.max_num_logprobs = 1;

  Sampler sampler;
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);

  REQUIRE(out.sampled_token_ids[0][0] == 1);
  REQUIRE(out.logprobs_tensors.has_value());
  const auto& lt = *out.logprobs_tensors;
  CHECK(lt.num_tokens_per_position == 2);  // k + 1
  CHECK(lt.logprob_token_ids[0] == 1);     // sampled token in col 0
  CHECK(lt.logprob_token_ids[1] == 0);     // top-1 raw token is token 0
  CHECK(lt.selected_token_ranks[0] == 2);  // token 1 is the 2nd-highest raw lp
}

// ---------------------------------------------------------------------------
// Full ordered pipeline on a 2-req batch: req 0 greedy, req 1 random with a
// peaked distribution + logprobs, exercising the whole assembly at once.
TEST_CASE("Sampler: full pipeline on a 2-req batch (greedy + random) with logprobs") {
  std::vector<float> logits = {1.0f, 4.0f, 2.0f, 0.0f,     // greedy -> 1
                               0.0f, 0.0f, 0.0f, 50.0f};   // random -> 3 (peaked)
  Tensor tl = Logits(logits, 2, 4);
  SamplingMetadata sm;
  sm.all_greedy = false;
  sm.all_random = false;
  sm.temperature = std::vector<float>{0.0f, 1.0f};
  sm.generators[1] = 4242;
  sm.max_num_logprobs = 1;

  Sampler sampler;
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);

  CHECK(out.sampled_token_ids[0][0] == 1);
  CHECK(out.sampled_token_ids[1][0] == 3);
  REQUIRE(out.logprobs_tensors.has_value());
  const auto& lt = *out.logprobs_tensors;
  CHECK(lt.num_positions == 2);
  CHECK(lt.num_tokens_per_position == 2);
  // Col 0 of each row is the sampled token.
  CHECK(lt.logprob_token_ids[0] == 1);
  CHECK(lt.logprob_token_ids[2] == 3);
}

// ---------------------------------------------------------------------------
// Custom logits processor (ROAD-V1-C7 `custom_logit_processor`): a host callback
// that FORCES a chosen token wins greedy sampling EXACTLY, and — RED-first — the
// same batch without the processor samples a DIFFERENT token. Also proves the
// callback fires once per request with the right (token_ids, vocab_size) view.
namespace {
// Force-a-token processor state. Records the arity the sampler handed it so the
// test can assert the contract (fire count, vocab_size, token_ids ptr/len).
struct ForceProcState {
  int32_t forced = 0;      // the token id to force.
  int calls = 0;           // number of callback invocations.
  int32_t last_vocab = 0;  // vocab_size seen on the last call.
  int last_n_tokens = -1;  // n_token_ids seen on the last call.
  bool tokens_ptr_ok = true;  // token_ids non-null whenever n_token_ids > 0.
};

// C-style callback matching vllm::LogitsProcessorFn. Sets the forced token's
// logit far above every other (finite, so no nan under softmax), leaving greedy
// argmax == forced.
void ForceTokenCb(const int32_t* token_ids, int32_t n_token_ids, float* logits,
                  int32_t vocab_size, void* user_data) {
  auto* s = static_cast<ForceProcState*>(user_data);
  s->calls += 1;
  s->last_vocab = vocab_size;
  s->last_n_tokens = n_token_ids;
  if (n_token_ids > 0 && token_ids == nullptr) s->tokens_ptr_ok = false;
  for (int32_t j = 0; j < vocab_size; ++j) logits[j] = -1e30f;
  if (s->forced >= 0 && s->forced < vocab_size) logits[s->forced] = 1e30f;
}
}  // namespace

TEST_CASE("Sampler: custom logits processor forces the greedy token exactly") {
  // Baseline argmax is token 1; the processor forces token 3.
  std::vector<float> baseline = {0.1f, 5.0f, 0.2f, 0.3f};
  Sampler sampler;
  Queue q = Q();

  // RED reference: no processor -> the untouched argmax (token 1).
  {
    std::vector<float> logits = baseline;
    Tensor tl = Logits(logits, 1, 4);
    SamplingMetadata sm;
    sm.all_greedy = true;
    sm.max_num_logprobs = std::nullopt;
    auto out = sampler.forward(q, tl, sm);
    CHECK(out.sampled_token_ids[0][0] == 1);  // untouched argmax
  }

  // With the processor -> the FORCED token (3), not the baseline argmax (1).
  ForceProcState st;
  st.forced = 3;
  {
    std::vector<float> logits = baseline;
    Tensor tl = Logits(logits, 1, 4);
    SamplingMetadata sm;
    sm.all_greedy = true;
    sm.max_num_logprobs = std::nullopt;
    sm.output_token_ids = {{7, 8}};  // pretend two tokens already generated
    sm.logits_processors[0] = vllm::LogitsProcessorCallback{&ForceTokenCb, &st};
    auto out = sampler.forward(q, tl, sm);
    CHECK(out.sampled_token_ids[0][0] == 3);  // forced token wins EXACTLY
  }
  // Callback contract: fired exactly once, saw the full vocab and the request's
  // generated tokens so far (len 2, non-null pointer).
  CHECK(st.calls == 1);
  CHECK(st.last_vocab == 4);
  CHECK(st.last_n_tokens == 2);
  CHECK(st.tokens_ptr_ok);
}

// ---------------------------------------------------------------------------
// Per-request scoping: in a 2-row batch only row 1 registers a processor, so
// only row 1's argmax is forced; row 0 keeps its untouched argmax. Proves the
// callback fires once per REGISTERED request over the correct row.
TEST_CASE("Sampler: custom logits processor is per-request (only the wired row)") {
  std::vector<float> logits = {9.0f, 1.0f, 1.0f, 2.0f,    // row 0 argmax 0
                               0.1f, 0.2f, 5.0f, 0.3f};   // row 1 argmax 2
  Tensor tl = Logits(logits, 2, 4);
  ForceProcState st;
  st.forced = 1;  // force row 1 to token 1 (its untouched argmax is 2)
  SamplingMetadata sm;
  sm.all_greedy = true;
  sm.max_num_logprobs = std::nullopt;
  sm.output_token_ids = {{}, {}};
  sm.logits_processors[1] = vllm::LogitsProcessorCallback{&ForceTokenCb, &st};

  Sampler sampler;
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);

  CHECK(out.sampled_token_ids[0][0] == 0);  // row 0 untouched
  CHECK(out.sampled_token_ids[1][0] == 1);  // row 1 forced
  CHECK(st.calls == 1);                     // fired only for the registered row
  CHECK(st.last_vocab == 4);
}

// ---------------------------------------------------------------------------
// Default inertness: an EMPTY logits_processors map leaves the pipeline
// byte-identical (same argmax as with no field set at all).
TEST_CASE("Sampler: empty custom-logits-processor map is inert") {
  std::vector<float> logits = {0.1f, 5.0f, 0.2f, 0.3f};
  Tensor tl = Logits(logits, 1, 4);
  SamplingMetadata sm;
  sm.all_greedy = true;
  sm.max_num_logprobs = std::nullopt;
  // logits_processors default-constructed empty.
  Sampler sampler;
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);
  CHECK(out.sampled_token_ids[0][0] == 1);  // untouched argmax
}

// ---------------------------------------------------------------------------
// logprobs_mode (issue #238, sampler.py:87-93,255-302). Three of the four modes
// were runtime-refused stubs; these gate what each one actually returns.
//
// The distinction that matters is RAW vs PROCESSED, not logprobs vs logits: the
// raw pair is snapshotted before any mutation and describes the MODEL's
// distribution, while the processed pair is taken after temperature and
// top-k/top-p and describes the distribution actually SAMPLED from. A token
// top-k masks away reads its true value in the raw modes and -inf in the
// processed ones. Every case below uses the same logits so the four modes are
// directly comparable.

// raw_logits: the logits themselves, NOT log_softmax of them.
TEST_CASE("Sampler: logprobs_mode raw_logits returns the unnormalized logits") {
  std::vector<float> logits = {3.0f, 1.0f, 2.0f, 0.0f};
  Tensor tl = Logits(logits, 1, 4);
  SamplingMetadata sm;
  sm.all_greedy = true;
  sm.all_random = false;
  sm.max_num_logprobs = 2;

  Sampler sampler(vllm::v1::LogprobsMode::kRawLogits);
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);

  REQUIRE(out.logprobs_tensors.has_value());
  const auto& lt = *out.logprobs_tensors;
  REQUIRE(lt.logprobs.size() == 3);
  // The raw logit values, verbatim -- every one strictly greater than the
  // corresponding log_softmax value, which is what makes this mode observable.
  CHECK(lt.logprobs[0] == doctest::Approx(3.0f));  // sampled (token 0)
  CHECK(lt.logprobs[1] == doctest::Approx(3.0f));  // top-1
  CHECK(lt.logprobs[2] == doctest::Approx(2.0f));  // top-2
  CHECK(lt.logprob_token_ids[0] == 0);
  CHECK(lt.selected_token_ranks[0] == 1);
}

// The default mode over the same logits, for contrast: normalized, so strictly
// less than the raw logits above. Guards the default against this change.
TEST_CASE("Sampler: logprobs_mode raw_logprobs is unchanged and normalized") {
  std::vector<float> logits = {3.0f, 1.0f, 2.0f, 0.0f};
  Tensor tl = Logits(logits, 1, 4);
  SamplingMetadata sm;
  sm.all_greedy = true;
  sm.all_random = false;
  sm.max_num_logprobs = 2;

  Sampler sampler;  // default == kRawLogprobs
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);

  REQUIRE(out.logprobs_tensors.has_value());
  const auto& lt = *out.logprobs_tensors;
  const float lse = 3.0f + std::log(std::exp(0.0f) + std::exp(-2.0f) +
                                    std::exp(-1.0f) + std::exp(-3.0f));
  CHECK(lt.logprobs[0] == doctest::Approx(3.0f - lse));
  CHECK(lt.logprobs[0] < 3.0f);  // strictly below the raw_logits answer
}

// processed_logits under top_k=2: the two surviving tokens keep their
// temperature-scaled logits and the masked ones read -inf. This is the case the
// mode exists for, and no raw mode can produce it.
TEST_CASE("Sampler: logprobs_mode processed_logits shows the top-k mask") {
  std::vector<float> logits = {3.0f, 1.0f, 2.0f, 0.0f};
  Tensor tl = Logits(logits, 1, 4);
  SamplingMetadata sm;
  sm.all_greedy = false;
  sm.all_random = true;
  sm.temperature = std::vector<float>{1.0f};
  sm.top_k = std::vector<int32_t>{2};
  sm.max_num_logprobs = 3;  // ask for enough to see a masked token

  Sampler sampler(vllm::v1::LogprobsMode::kProcessedLogits);
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);

  REQUIRE(out.logprobs_tensors.has_value());
  const auto& lt = *out.logprobs_tensors;
  REQUIRE(lt.num_tokens_per_position == 4);  // k + 1
  // The kept tokens (0 and 2) hold their logits at temperature 1.0; the tail is
  // masked. Column 0 is the sampled token, then the top-k by value.
  CHECK(lt.logprobs[1] == doctest::Approx(3.0f));
  CHECK(lt.logprobs[2] == doctest::Approx(2.0f));
  // Whatever landed third is one of the masked tokens -> -inf.
  CHECK(lt.logprobs[3] == -std::numeric_limits<float>::infinity());
}

// processed_logprobs: the same mask, but renormalized over the surviving
// tokens, so the kept pair sums to 1 in probability space. That renormalization
// is the whole difference from processed_logits.
TEST_CASE("Sampler: logprobs_mode processed_logprobs renormalizes over the kept set") {
  std::vector<float> logits = {3.0f, 1.0f, 2.0f, 0.0f};
  Tensor tl = Logits(logits, 1, 4);
  SamplingMetadata sm;
  sm.all_greedy = false;
  sm.all_random = true;
  sm.temperature = std::vector<float>{1.0f};
  sm.top_k = std::vector<int32_t>{2};
  sm.max_num_logprobs = 3;

  Sampler sampler(vllm::v1::LogprobsMode::kProcessedLogprobs);
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);

  REQUIRE(out.logprobs_tensors.has_value());
  const auto& lt = *out.logprobs_tensors;
  // log_softmax over the SURVIVING pair {3.0, 2.0} only.
  const float kept_lse = 3.0f + std::log(std::exp(0.0f) + std::exp(-1.0f));
  CHECK(lt.logprobs[1] == doctest::Approx(3.0f - kept_lse));
  CHECK(lt.logprobs[2] == doctest::Approx(2.0f - kept_lse));
  CHECK(lt.logprobs[3] == -std::numeric_limits<float>::infinity());
  // Renormalized: the two kept tokens carry all the mass.
  const double mass = std::exp(static_cast<double>(lt.logprobs[1])) +
                      std::exp(static_cast<double>(lt.logprobs[2]));
  CHECK(mass == doctest::Approx(1.0).epsilon(1e-5));
}

// ---------------------------------------------------------------------------
// logprob_token_ids (generative scoring), sampler.py:151-225. WRITTEN, not
// ported: upstream's only unit test for the field covers the model-config
// vocab-bounds branch of verify() (tests/v1/sample/test_logprobs.py:413-425),
// which is our named engine-time deferral; the algorithm itself is exercised
// upstream only through an HTTP end-to-end over a real model.
//
// One request, explicit ids: the row is [sampled | id0 | id1], the logprobs are
// the raw logprobs AT those ids (not a top-k), and the rank is the sampled
// token's rank over the FULL vocab.
//
// RED before the port: out.logprobs_tensors has NO value — max_num_logprobs is
// unset and nothing reads sm.logprob_token_ids.
TEST_CASE("Sampler: logprob_token_ids gathers exactly the requested ids") {
  // logits [3,1,2,0] -> greedy token 0 (also the max logprob).
  std::vector<float> logits = {3.0f, 1.0f, 2.0f, 0.0f};
  Tensor tl = Logits(logits, 1, 4);
  SamplingMetadata sm;
  sm.all_greedy = true;
  sm.all_random = false;
  sm.max_num_logprobs = std::nullopt;  // ONLY logprob_token_ids is set
  sm.logprob_token_ids = std::map<int, std::vector<int32_t>>{{0, {3, 1}}};

  Sampler sampler;
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);

  REQUIRE(out.sampled_token_ids[0][0] == 0);
  REQUIRE(out.logprobs_tensors.has_value());
  const auto& lt = *out.logprobs_tensors;
  CHECK(lt.num_positions == 1);
  CHECK(lt.num_tokens_per_position == 3);  // max_num_tokens + 1
  REQUIRE(lt.logprob_token_ids.size() == 3);
  // Column 0 is the sampled token; then the requested ids IN REQUEST ORDER
  // (3 then 1) — NOT sorted by logprob, which is the whole point of the field.
  CHECK(lt.logprob_token_ids[0] == 0);
  CHECK(lt.logprob_token_ids[1] == 3);
  CHECK(lt.logprob_token_ids[2] == 1);
  const float lse = 3.0f + std::log(std::exp(0.0f) + std::exp(-2.0f) +
                                    std::exp(-1.0f) + std::exp(-3.0f));
  CHECK(lt.logprobs[0] == doctest::Approx(3.0f - lse));  // sampled (token 0)
  CHECK(lt.logprobs[1] == doctest::Approx(0.0f - lse));  // token 3
  CHECK(lt.logprobs[2] == doctest::Approx(1.0f - lse));  // token 1
  // Rank over the FULL vocab (sampler.py:212-213), not over the 2 requested
  // ids: the sampled token is the argmax, so rank 1.
  CHECK(lt.selected_token_ranks[0] == 1);
}

// Heterogeneous id lists across the batch, plus a row with NO entry at all.
// Upstream pads to [batch, max_num_tokens + 1] and masks the padding to -inf,
// keeping column 0 (the sampled token) valid for EVERY row (sampler.py:186-206).
TEST_CASE("Sampler: logprob_token_ids pads short and absent rows with -inf") {
  std::vector<float> logits = {3.0f, 1.0f, 2.0f, 0.0f,    // row 0 -> token 0
                               0.0f, 7.0f, 0.0f, 0.0f,    // row 1 -> token 1
                               0.0f, 0.0f, 0.0f, 9.0f};   // row 2 -> token 3
  Tensor tl = Logits(logits, 3, 4);
  SamplingMetadata sm;
  sm.all_greedy = true;
  sm.all_random = false;
  sm.max_num_logprobs = std::nullopt;
  // row 0 asks for two ids, row 1 for one, row 2 is absent from the map.
  sm.logprob_token_ids =
      std::map<int, std::vector<int32_t>>{{0, {2, 3}}, {1, {0}}};

  Sampler sampler;
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);

  REQUIRE(out.logprobs_tensors.has_value());
  const auto& lt = *out.logprobs_tensors;
  CHECK(lt.num_positions == 3);
  CHECK(lt.num_tokens_per_position == 3);  // max(2, 1) + 1
  REQUIRE(lt.logprob_token_ids.size() == 9);
  const float kNegInf = -std::numeric_limits<float>::infinity();

  // Row 0: full width, no padding.
  CHECK(lt.logprob_token_ids[0] == 0);
  CHECK(lt.logprob_token_ids[1] == 2);
  CHECK(lt.logprob_token_ids[2] == 3);
  CHECK(lt.logprobs[2] > kNegInf);

  // Row 1: one requested id, then one padded column.
  CHECK(lt.logprob_token_ids[3] == 1);  // sampled
  CHECK(lt.logprob_token_ids[4] == 0);  // requested
  CHECK(lt.logprobs[4] > kNegInf);
  CHECK(lt.logprobs[5] == kNegInf);     // padded

  // Row 2: absent from the map — column 0 is still its sampled token, and
  // BOTH remaining columns are padding.
  CHECK(lt.logprob_token_ids[6] == 3);
  CHECK(lt.logprobs[6] > kNegInf);
  CHECK(lt.logprobs[7] == kNegInf);
  CHECK(lt.logprobs[8] == kNegInf);

  // Ranks are still the full-vocab ranks of each row's sampled token.
  CHECK(lt.selected_token_ranks[0] == 1);
  CHECK(lt.selected_token_ranks[1] == 1);
  CHECK(lt.selected_token_ranks[2] == 1);
}

// Precedence (sampler.py:133-136): when a request supplies BOTH an explicit id
// set and a logprobs COUNT, the explicit ids win.
TEST_CASE("Sampler: logprob_token_ids wins over max_num_logprobs") {
  std::vector<float> logits = {3.0f, 1.0f, 2.0f, 0.0f};
  Tensor tl = Logits(logits, 1, 4);
  SamplingMetadata sm;
  sm.all_greedy = true;
  sm.all_random = false;
  sm.max_num_logprobs = 2;  // would gather the top-2: tokens 0 and 2
  sm.logprob_token_ids = std::map<int, std::vector<int32_t>>{{0, {3}}};

  Sampler sampler;
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);

  REQUIRE(out.logprobs_tensors.has_value());
  const auto& lt = *out.logprobs_tensors;
  // The explicit-ids shape ([sampled | 3]), NOT the top-2 shape ([0 | 0 | 2]).
  CHECK(lt.num_tokens_per_position == 2);
  REQUIRE(lt.logprob_token_ids.size() == 2);
  CHECK(lt.logprob_token_ids[0] == 0);
  CHECK(lt.logprob_token_ids[1] == 3);
}

// An out-of-range id must be REJECTED loudly, never read past the row.
// torch.gather raises on an out-of-range index, so this mirrors upstream rather
// than adding a check upstream lacks. (Issue #249's unbounded `k` lives in
// GatherLogprobs and is deliberately NOT touched here.)
TEST_CASE("Sampler: an out-of-vocab logprob_token_id throws") {
  std::vector<float> logits = {3.0f, 1.0f, 2.0f, 0.0f};
  Tensor tl = Logits(logits, 1, 4);
  SamplingMetadata sm;
  sm.all_greedy = true;
  sm.all_random = false;
  sm.max_num_logprobs = std::nullopt;
  sm.logprob_token_ids = std::map<int, std::vector<int32_t>>{{0, {4}}};  // vocab is 4

  Sampler sampler;
  Queue q = Q();
  CHECK_THROWS(sampler.forward(q, tl, sm));
}

// The empty map is FALSY in Python (`if sampling_metadata.logprob_token_ids:`),
// so a set-but-empty optional must behave exactly like an unset one.
TEST_CASE("Sampler: an empty logprob_token_ids map is inert") {
  std::vector<float> logits = {0.1f, 5.0f, 0.2f, 0.3f};
  Tensor tl = Logits(logits, 1, 4);
  SamplingMetadata sm;
  sm.all_greedy = true;
  sm.max_num_logprobs = std::nullopt;
  sm.logprob_token_ids = std::map<int, std::vector<int32_t>>{};

  Sampler sampler;
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);
  CHECK(out.sampled_token_ids[0][0] == 1);
  CHECK_FALSE(out.logprobs_tensors.has_value());
}

// ---------------------------------------------------------------------------
// The two features above COMPOSE, and this pins the composition that the
// merge of #238 (logprobs_mode) and #264 (logprob_token_ids) creates: they are
// orthogonal, so `logprobs_mode` decides WHICH tensor the step-1 snapshot holds
// and `logprob_token_ids` decides WHICH entries step 8 reads out of it.
//
// Neither PR could test this on its own -- each was written against a base
// without the other. Without it the interaction is untested, and the failure it
// guards is silent: if the snapshot were driven by `max_num_logprobs` alone
// (sampler.py:86 is an `or`, not an `and`), a request that sets ONLY
// logprob_token_ids under a processed_* mode would gather out of an EMPTY
// buffer.
//
// The tell is token 1: top_k=2 keeps only tokens 0 and 2, so a PROCESSED
// snapshot reads -inf there while either RAW snapshot would read a finite
// value. Column 0 (the sampled token) is deliberately not asserted -- this is a
// random batch, so which of the two surviving tokens lands there is not fixed.
TEST_CASE("Sampler: logprob_token_ids reads the PROCESSED snapshot under a processed mode") {
  std::vector<float> logits = {3.0f, 1.0f, 2.0f, 0.0f};
  Tensor tl = Logits(logits, 1, 4);
  SamplingMetadata sm;
  sm.all_greedy = false;
  sm.all_random = true;
  sm.temperature = std::vector<float>{1.0f};
  sm.top_k = std::vector<int32_t>{2};
  sm.max_num_logprobs = std::nullopt;  // ONLY logprob_token_ids is set
  sm.logprob_token_ids = std::map<int, std::vector<int32_t>>{{0, {2, 1}}};

  Sampler sampler(vllm::v1::LogprobsMode::kProcessedLogprobs);
  Queue q = Q();
  auto out = sampler.forward(q, tl, sm);

  REQUIRE(out.logprobs_tensors.has_value());
  const auto& lt = *out.logprobs_tensors;
  REQUIRE(lt.num_positions == 1);
  REQUIRE(lt.num_tokens_per_position == 3);  // sampled + the two requested ids

  // log_softmax over the SURVIVING pair {3.0, 2.0}, exactly as the
  // processed_logprobs case above computes it.
  const float kept_lse = 3.0f + std::log(std::exp(0.0f) + std::exp(-1.0f));

  // Requested id 2 survived top-k, so it carries the RENORMALIZED value.
  CHECK(lt.logprob_token_ids[1] == 2);
  CHECK(lt.logprobs[1] == doctest::Approx(2.0f - kept_lse));
  // Requested id 1 was masked away. -inf here, and NOT the finite raw value,
  // is the whole proof that the processed tensor is what was gathered from.
  CHECK(lt.logprob_token_ids[2] == 1);
  CHECK(lt.logprobs[2] == -std::numeric_limits<float>::infinity());
  const float raw_lse = 3.0f + std::log(std::exp(0.0f) + std::exp(-2.0f) +
                                        std::exp(-1.0f) + std::exp(-3.0f));
  CHECK(lt.logprobs[2] != doctest::Approx(1.0f - raw_lse));
}
