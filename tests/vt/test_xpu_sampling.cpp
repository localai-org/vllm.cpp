#include "xpu_test_helpers.h"
#include <array>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <limits>
#include "vt/xpu.h"
#include "vt/xpu_sampling.h"
#include "vt/sample_common.h"
#include "vllm/v1/sample/sampler.h"
#include "vllm/v1/sample/logits_processor/builtin.h"

namespace {
using xpu_test::Buffer;
using xpu_test::Queue;
using vt::DType;
void Compare(const std::vector<float>& a, const std::vector<float>& b, float tolerance = 3e-5f) {
  REQUIRE(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] == b[i] || (std::isnan(a[i]) && std::isnan(b[i]))) continue;
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || std::abs(a[i] - b[i]) > tolerance * (1 + std::abs(b[i]))) {
      CAPTURE(i);
      CAPTURE(a[i]);
      CAPTURE(b[i]);
      FAIL("sampling reference mismatch");
    }
  }
}
}
TEST_CASE("XPU sampling: temperatures, probabilities, logprobs and min-p") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (int vocab : {1, 257, 248320}) {
    Buffer a(cpu.q, DType::kF32, {4, vocab}), b(gpu.q, DType::kF32, {4, vocab});
    Buffer t(cpu.q, DType::kF32, {4}), u(gpu.q, DType::kF32, {4});
    auto values = xpu_test::Values(4 * vocab, 3, 0.5f);
    if (vocab > 1) values[1] = -std::numeric_limits<float>::infinity();
    a.put(values); b.put(values); t.put({0, 0.7f, 1.2f, 1}); u.put({0, 0.7f, 1.2f, 1});
    vt::ApplyTemperature(cpu.q, a.tensor, t.tensor, false); vt::ApplyTemperature(gpu.q, b.tensor, u.tensor, false);
    Compare(b.floats(), a.floats());
    Buffer p(cpu.q, DType::kF32, {4, vocab}), r(gpu.q, DType::kF32, {4, vocab});
    vt::ComputeProbs(cpu.q, p.tensor, a.tensor); vt::ComputeProbs(gpu.q, r.tensor, b.tensor);
    Compare(r.floats(), p.floats());
    auto probabilities = r.floats();
    for (int row = 0; row < 4; ++row) {
      double sum = 0; for (int j = 0; j < vocab; ++j) sum += probabilities[row * vocab + j];
      CHECK(std::abs(sum - 1) < 2e-5);
    }
    vt::ComputeLogprobs(cpu.q, p.tensor, a.tensor); vt::ComputeLogprobs(gpu.q, r.tensor, b.tensor);
    Compare(r.floats(), p.floats(), 3e-4f); // CPU serial sum rounds across the full vocabulary.
    vt::ComputeLogprobs(gpu.q, b.tensor, b.tensor); Compare(b.floats(), r.floats(), 1e-6f);
    a.put(values); b.put(values); t.put({0, .1f, .8f, 1}); u.put({0, .1f, .8f, 1});
    vt::ApplyMinP(cpu.q, a.tensor, t.tensor); vt::ApplyMinP(gpu.q, b.tensor, u.tensor);
    Compare(b.floats(), a.floats(), 0);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU compact draft selection ties by global ID and downloads only IDs") {
  Queue gpu(vt::DeviceType::kXPU);
  constexpr int columns = 65536, rows = 4;
  Buffer logits(gpu.q, DType::kF32, {rows, columns});
  Buffer mapping(gpu.q, DType::kI32, {columns});
  Buffer output(gpu.q, DType::kI32, {rows});
  std::vector<int32_t> global_ids(columns);
  for (int i = 0; i < columns; ++i) global_ids[i] = columns - 1 - i + 128;
  std::vector<float> values(rows * columns, -5.0f);
  values[columns] = values[2 * columns - 1] = 9.0f;
  values[2 * columns] = 10.0f;
  values[3 * columns + 32768] = 1.0f;
  mapping.upload(global_ids.data());
  logits.upload(values.data());
  vt::MappedGreedyArgmax(gpu.q, output.tensor, logits.tensor, mapping.tensor, 248320);
  auto raw = output.download();
  std::array<int32_t, rows> ids;
  std::memcpy(ids.data(), raw.data(), raw.size());
  CHECK((ids == std::array<int32_t, rows>{128, 128, 65663, global_ids[32768]}));
  CHECK(raw.size() == rows * sizeof(int32_t));

  // Nonfinite arithmetic is an explicit refusal, never a plausible proposal.
  values[123] = std::numeric_limits<float>::quiet_NaN();
  logits.upload(values.data());
  vt::MappedGreedyArgmax(gpu.q, output.tensor, logits.tensor, mapping.tensor, 248320);
  raw = output.download();
  std::memcpy(ids.data(), raw.data(), raw.size());
  CHECK(ids[0] == -1);
  CHECK(ids[1] == 128);

  global_ids[0] = 248320;
  mapping.upload(global_ids.data());
  vt::MappedGreedyArgmax(gpu.q, output.tensor, logits.tensor, mapping.tensor, 248320);
  raw = output.download();
  std::memcpy(ids.data(), raw.data(), raw.size());
  CHECK((ids == std::array<int32_t, rows>{-1, -1, -1, -1}));
  CHECK_THROWS_AS(vt::MappedGreedyArgmax(gpu.q, output.tensor, logits.tensor,
                                       mapping.tensor, 0), std::runtime_error);
}
TEST_CASE("XPU sampling: penalties, duplicate sparse biases, token and allowed masks") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  auto run = [](vt::Queue& q) {
    constexpr int N = 4, V = 33, M = 5;
    Buffer logits(q, DType::kF32, {N, V}), pm(q, DType::kI8, {N, V}), om(q, DType::kI8, {N, V}), counts(q, DType::kI32, {N, V});
    Buffer freq(q, DType::kF32, {N}), pres(q, DType::kF32, {N}), rep(q, DType::kF32, {N});
    logits.put(xpu_test::Values(N * V, 2));
    std::vector<int8_t> prompt(N * V), output(N * V); std::vector<int32_t> c(N * V);
    for (int i = 0; i < N * V; ++i) { prompt[i] = i % 3 == 0; output[i] = i % 4 == 0; c[i] = output[i] * (i % 7 + 1); }
    pm.upload(prompt.data()); om.upload(output.data()); counts.upload(c.data());
    freq.put({0, .3f, -.7f, 1}); pres.put({1, 0, .3f, -.1f}); rep.put({1, 1.2f, .8f, 2});
    vt::ApplyPenalties(q, logits.tensor, pm.tensor, counts.tensor, om.tensor, freq.tensor, pres.tensor, rep.tensor);
    Buffer rows(q, DType::kI32, {M}), cols(q, DType::kI32, {M}), bias(q, DType::kF32, {M});
    int32_t rr[] = {0, 1, 0, 3, 0}, cc[] = {3, 2, 3, 32, 3};
    rows.upload(rr); cols.upload(cc); bias.put({1e10f, .2f, -1e10f, .3f, .25f});
    vt::ApplyLogitBias(q, logits.tensor, rows.tensor, cols.tensor, bias.tensor);
    auto result = logits.floats();
    vt::ApplyTokenMask(q, logits.tensor, rows.tensor, cols.tensor);
    vt::ApplyAllowedTokenIds(q, logits.tensor, pm.tensor);
    const auto masked = logits.floats(); result.insert(result.end(), masked.begin(), masked.end());
    return result;
  };
  Compare(run(gpu.q), run(cpu.q), 0);
  CHECK(vt::GetReferenceTierHits() == 0);
}
TEST_CASE("XPU sampling: invalid sparse indices do not partially mutate logits") {
  Queue gpu(vt::DeviceType::kXPU);
  Buffer logits(gpu.q, DType::kF32, {1, 7}), rows(gpu.q, DType::kI32, {2}), cols(gpu.q, DType::kI32, {2}), bias(gpu.q, DType::kF32, {2});
  const auto original = xpu_test::Values(7); logits.put(original);
  const int32_t rr[] = {0, 1}, cc[] = {2, 2}; rows.upload(rr); cols.upload(cc); bias.put({1, 2});
  CHECK_THROWS_AS(vt::ApplyLogitBias(gpu.q, logits.tensor, rows.tensor, cols.tensor, bias.tensor), std::runtime_error);
  CHECK(logits.floats() == original);
  CHECK_THROWS_AS(vt::ApplyTokenMask(gpu.q, logits.tensor, rows.tensor, cols.tensor), std::runtime_error);
  CHECK(logits.floats() == original);
}
TEST_CASE("XPU sampling: stable top-k/top-p, ties and uneven vocabularies") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (int vocab : {1, 7, 257, 1025}) for (int mode : {0, 1, 2}) {
    CAPTURE(vocab);
    CAPTURE(mode);
    Buffer a(cpu.q, DType::kF32, {4, vocab}), b(gpu.q, DType::kF32, {4, vocab});
    Buffer kc(cpu.q, DType::kI32, {4}), kg(gpu.q, DType::kI32, {4});
    Buffer pc(cpu.q, DType::kF32, {4}), pg(gpu.q, DType::kF32, {4});
    const int32_t ks[] = {1, 3, vocab, 0}; kc.upload(ks); kg.upload(ks);
    pc.put({.05f, .37f, .83f, 1}); pg.put({.05f, .37f, .83f, 1});
    const auto values = xpu_test::Values(4 * vocab, 12, .2f); a.put(values); b.put(values);
    vt::ApplyTopKTopP(cpu.q, a.tensor, mode == 1 ? nullptr : &kc.tensor, mode == 0 ? nullptr : &pc.tensor);
    vt::ApplyTopKTopP(gpu.q, b.tensor, mode == 1 ? nullptr : &kg.tensor, mode == 0 ? nullptr : &pg.tensor);
    Compare(b.floats(), a.floats(), 0);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
  CHECK(vt::xpu::GetMemoryInfo().sampling_workspace_bytes == 16 * 1024 * 1024);
}
TEST_CASE("XPU sampling: exact top-20 route and stable-sort fallback") {
  Queue gpu(vt::DeviceType::kXPU);
  constexpr int N = 5;
  for (int vocab : {257, 1025, 248320}) {
    CAPTURE(vocab);
    Buffer reference(gpu.q, DType::kF32, {N, vocab});
    Buffer selected(gpu.q, DType::kF32, {N, vocab});
    Buffer ks(gpu.q, DType::kI32, {N}), ps(gpu.q, DType::kF32, {N});
    std::vector<float> values(size_t(N * vocab));
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = float((i * 2654435761ULL) % 1000003) * 0x1p-18f;
    for (int row = 0; row < N; ++row)
      for (int rank = 0; rank < 20; ++rank)
        values[size_t(row * vocab + (rank * 997) % vocab)] =
            20.0f - 0.35f * rank;
    const int32_t k[N] = {20, 20, 20, 20, 20};
    ks.upload(k);
    ps.put({.05f, .37f, .95f, 1.0f, .999f});
    reference.put(values); selected.put(values);
    vt::ApplyTopKTopP(gpu.q, reference.tensor, &ks.tensor, &ps.tensor);
    REQUIRE(vt::xpu::ApplyTopK20TopP(gpu.q, selected.tensor, ps.tensor));
    Compare(selected.floats(), reference.floats(), 0);
  }
  Buffer tied(gpu.q, DType::kF32, {N, 257});
  Buffer ps(gpu.q, DType::kF32, {N});
  ps.put({.95f, .95f, .95f, .95f, .95f});
  const std::vector<float> same(N * 257, 1.0f);
  tied.put(same);
  CHECK_FALSE(vt::xpu::ApplyTopK20TopP(gpu.q, tied.tensor, ps.tensor));
  CHECK(tied.floats() == same);
  Buffer boundary(gpu.q, DType::kF32, {N, 248320});
  Buffer boundary_p(gpu.q, DType::kF32, {N});
  std::vector<float> nearly_equal(size_t(N * 248320));
  for (size_t i = 0; i < nearly_equal.size(); ++i)
    nearly_equal[i] = float((i * 2654435761ULL) % 1000003) * 0x1p-18f;
  boundary.put(nearly_equal);
  boundary_p.put({.05f, .37f, .95f, 1.0f, .999f});
  CHECK_FALSE(vt::xpu::ApplyTopK20TopP(gpu.q, boundary.tensor, boundary_p.tensor));
  CHECK(boundary.floats() == nearly_equal);
  CHECK(vt::GetReferenceTierHits() == 0);
}
TEST_CASE("XPU sampling: speculative allowed tokens and bias with min-token positions") {
  Queue gpu(vt::DeviceType::kXPU);
  Buffer logits(gpu.q, DType::kF32, {5, 7});
  std::vector<float> values(35);
  for (size_t i = 0; i < values.size(); ++i) values[i] = float(i);
  logits.put(values);
  vllm::v1::SamplingMetadata metadata;
  metadata.logit_bias[0] = {{1, 2.0f}};
  metadata.logit_bias[1] = {{3, 3.0f}};
  metadata.min_tokens[0] = {4, {1}};
  metadata.min_tokens[1] = {1, {3}};
  metadata.output_token_positions = {2, 0};
  metadata.allowed_token_ids_mask = std::vector<std::vector<uint8_t>>(
      2, std::vector<uint8_t>(7, 0));
  std::fill((*metadata.allowed_token_ids_mask)[0].begin(),
            (*metadata.allowed_token_ids_mask)[0].end(), 1);
  (*metadata.allowed_token_ids_mask)[0][1] = 0;
  (*metadata.allowed_token_ids_mask)[0][5] = 0;
  vllm::v1::apply_speculative_logit_filters(
      gpu.q, logits.tensor, metadata, {0, 3, 5});
  for (int row = 0; row < 3; ++row)
    for (int token = 0; token < 7; ++token)
      if (token != 1 && token != 5)
        values[size_t(row * 7 + token)] = -std::numeric_limits<float>::infinity();
  values[1] = -std::numeric_limits<float>::infinity();
  values[8] = -std::numeric_limits<float>::infinity();
  values[15] += 2.0f;
  values[24] = -std::numeric_limits<float>::infinity();
  values[31] += 3.0f;
  Compare(logits.floats(), values, 0);
  CHECK(vt::GetReferenceTierHits() == 0);
}
TEST_CASE("XPU sampling R07: verification processors match serial target histories") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  constexpr int rows = 6, vocab = 7;
  struct Witness { std::vector<std::vector<int32_t>> history; std::vector<float> stage; };
  const auto callback = +[](const int32_t* ids, int32_t count, float* logits,
                             int32_t size, void* opaque) {
    auto& witness = *static_cast<Witness*>(opaque);
    witness.history.emplace_back(ids, ids + count);
    witness.stage.insert(witness.stage.end(), logits, logits + size);
    int sum = 0;
    for (int i = 0; i < count; ++i) sum += ids[i];
    logits[0] += float(sum) * .125f;
    logits[3] += float(count) * .5f;
  };
  Witness actual, expected;
  vllm::v1::SamplingMetadata metadata;
  metadata.no_penalties = false;
  metadata.output_token_ids = {{2, 2}, {5}};
  metadata.prompt_token_ids = std::vector<std::vector<int32_t>>{{0, 6}, {1, 3, 6}};
  metadata.output_token_positions = {2, 1};
  metadata.presence_penalties = {.5f, -.25f};
  metadata.frequency_penalties = {.125f, .5f};
  metadata.repetition_penalties = {1.25f, .75f};
  metadata.allowed_token_ids_mask = std::vector<std::vector<uint8_t>>(2, std::vector<uint8_t>(vocab, 0));
  (*metadata.allowed_token_ids_mask)[0][6] = (*metadata.allowed_token_ids_mask)[1][6] = 1;
  metadata.bad_words_token_ids[0] = {{3, 1}};
  metadata.bad_words_token_ids[1] = {{4, 3}, {4, 1, 2}};
  metadata.min_tokens[0] = {3, {5}};
  metadata.min_tokens[1] = {3, {0}};
  metadata.logit_bias[0] = {{3, 1.0f}, {5, 2.0f}};
  metadata.logit_bias[1] = {{0, .5f}, {6, 3.0f}};
  metadata.logits_processors[0] = metadata.logits_processors[1] = {callback, &actual};
  const std::vector<int32_t> offsets{0, 2, 6}, draft_inputs{0, 3, 6, 4, 1, 3};
  // Independent explicit histories: excludes both anchors and every future
  // draft. Cases cover MTP1 and MTP3 rows with different request parameters.
  const std::vector<std::vector<int32_t>> histories{
      {2, 2}, {2, 2, 3}, {5}, {5, 4}, {5, 4, 1}, {5, 4, 1, 3}};
  const std::vector<int> request_for_row{0, 0, 1, 1, 1, 1};
  std::vector<float> values(rows * vocab);
  const std::array<float, vocab> base{4, -2, 3, 5, 2, 1, -1};
  for (int row = 0; row < rows; ++row) for (int token = 0; token < vocab; ++token)
    values[row * vocab + token] = base[token] + row * .25f;
  std::vector<float> serial;
  const vllm::v1::Sampler sampler;
  for (int row = 0; row < rows; ++row) {
    const int req = request_for_row[row];
    auto one = metadata;
    one.output_token_ids = {histories[row]};
    one.prompt_token_ids = std::vector<std::vector<int32_t>>{(*metadata.prompt_token_ids)[req]};
    one.presence_penalties = {metadata.presence_penalties[req]};
    one.frequency_penalties = {metadata.frequency_penalties[req]};
    one.repetition_penalties = {metadata.repetition_penalties[req]};
    one.allowed_token_ids_mask = std::vector<std::vector<uint8_t>>{(*metadata.allowed_token_ids_mask)[req]};
    one.bad_words_token_ids = {{0, metadata.bad_words_token_ids.at(req)}};
    one.min_tokens = {{0, metadata.min_tokens.at(req)}};
    one.logit_bias = {{0, metadata.logit_bias.at(req)}};
    one.logits_processors = {{0, {callback, &expected}}};
    Buffer input(cpu.q, DType::kF32, {1, vocab});
    input.put(std::vector<float>(values.begin() + row * vocab, values.begin() + (row + 1) * vocab));
    const auto output = sampler.forward(cpu.q, input.tensor, one);
    REQUIRE(output.sampled_token_ids.size() == 1);
    const auto processed = input.floats();
    serial.insert(serial.end(), processed.begin(), processed.end());
  }
  Buffer input(gpu.q, DType::kF32, {rows, vocab});
  input.put(values);
  vllm::v1::apply_speculative_logits_processors(gpu.q, input.tensor, metadata, offsets, draft_inputs);
  Compare(input.floats(), serial);
  CHECK(actual.history == histories);
  CHECK(actual.history == expected.history);
  Compare(actual.stage, expected.stage, 0);
  for (int invalid = 0; invalid < 5; ++invalid) {
    CAPTURE(invalid);
    auto bad = metadata;
    auto bad_offsets = offsets, bad_drafts = draft_inputs;
    if (invalid == 0) bad_offsets[1] = -1;
    if (invalid == 1) bad_drafts.pop_back();
    if (invalid == 2) bad.output_token_ids.pop_back();
    if (invalid == 3) bad.frequency_penalties.pop_back();
    if (invalid == 4) bad.logit_bias[2] = {{0, 1}};
    input.put(values);
    const auto before = input.download();
    CHECK_THROWS(vllm::v1::apply_speculative_logits_processors(
        gpu.q, input.tensor, bad, bad_offsets, bad_drafts));
    xpu_test::SameBytes(input.download(), before);
  }
  vllm::v1::SamplingMetadata overflow;
  overflow.min_tokens[0] = {3, {6}};
  overflow.output_token_positions = {UINT64_MAX};
  auto two_rows = input.tensor; two_rows.shape[0] = 2;
  input.put(values); const auto unchanged = input.download();
  vllm::v1::apply_speculative_logits_processors(gpu.q, two_rows, overflow, {0, 2}, {});
  xpu_test::SameBytes(input.download(), unchanged);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU sampling: five-row production top-20 timing"
          * doctest::skip(!std::getenv("VT_B70_SAMPLING_BENCH"))) {
  Queue gpu(vt::DeviceType::kXPU);
  constexpr int N = 5, V = 248320;
  Buffer logits(gpu.q, DType::kF32, {N, V});
  Buffer ks(gpu.q, DType::kI32, {N}), ps(gpu.q, DType::kF32, {N});
  std::vector<float> values(size_t(N * V));
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = float((i * 2654435761ULL) % 1000003) * 0x1p-18f;
  for (int row = 0; row < N; ++row)
    for (int rank = 0; rank < 20; ++rank)
      values[size_t(row * V + rank * 997)] = 20.0f - 0.35f * rank;
  const int32_t k[N] = {20, 20, 20, 20, 20};
  ks.upload(k);
  ps.put({.95f, .95f, .95f, .95f, .95f});
  const char* top20_setting = std::getenv("VT_B70_FAST_TOPK20");
  const bool fast = !top20_setting || std::strcmp(top20_setting, "0") != 0;
  const auto apply = [&] {
    if (fast) CHECK(vt::xpu::ApplyTopK20TopP(gpu.q, logits.tensor, ps.tensor));
    else vt::ApplyTopKTopP(gpu.q, logits.tensor, &ks.tensor, &ps.tensor);
  };
  for (int warm = 0; warm < 2; ++warm) {
    logits.put(values);
    apply();
    vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
  }
  std::vector<double> ms;
  for (int repeat = 0; repeat < 5; ++repeat) {
    logits.put(values);
    const auto start = std::chrono::steady_clock::now();
    apply();
    vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
    ms.push_back(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count());
  }
  std::sort(ms.begin(), ms.end());
  std::cout << "SAMPLING_TOPK5 vocab=" << V << " k=20 top_p=0.95"
            << " route=" << (fast ? "top20" : "full_sort")
            << " wall_median_ms=" << ms[2] << " wall_samples_ms=";
  for (double sample : ms) std::cout << sample << ',';
  std::cout << std::endl;
  CHECK(vt::GetReferenceTierHits() == 0);
}
TEST_CASE("XPU sampling: nucleus probability mass at the real vocabulary size") {
  Queue gpu(vt::DeviceType::kXPU);
  constexpr int V = 248320, N = 4;
  Buffer logits(gpu.q, DType::kF32, {N, V}), top_p(gpu.q, DType::kF32, {N});
  // Unequal, strictly monotone rows make the nucleus boundary unambiguous.
  std::vector<float> values(N * V); const float ps[] = {.2f, .7f, .95f, 1};
  for (int row = 0; row < N; ++row) for (int j = 0; j < V; ++j) values[row * V + j] = float(j) / V * (row + 1);
  logits.put(values); top_p.upload(ps);
  vt::ApplyTopKTopP(gpu.q, logits.tensor, nullptr, &top_p.tensor);
  const auto masked = logits.floats();
  for (int row = 0; row < N; ++row) {
    double total = 0, kept = 0, first = 0;
    for (int j = 0; j < V; ++j) {
      const double probability = std::exp(double(values[row * V + j]) - (row + 1));
      total += probability;
      if (std::isfinite(masked[row * V + j])) { kept += probability; if (!first) first = probability; }
    }
    CHECK(kept / total >= ps[row] - 2e-6);
    CHECK((kept - first) / total <= ps[row] + 2e-6);
  }
}
TEST_CASE("XPU sampling: seeded exponential race and statistical distribution") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  constexpr int N = 8192, V = 4;
  Buffer pc(cpu.q, DType::kF32, {N, V}), pg(gpu.q, DType::kF32, {N, V});
  Buffer sc(cpu.q, DType::kI64, {N}), sg(gpu.q, DType::kI64, {N});
  Buffer ic(cpu.q, DType::kI64, {N}), ig(gpu.q, DType::kI64, {N});
  std::vector<float> probs(N * V); std::vector<int64_t> seeds(N);
  for (int i = 0; i < N; ++i) { seeds[i] = 977 + i; for (int j = 0; j < V; ++j) probs[i * V + j] = .1f * (j + 1); }
  pc.put(probs); pg.put(probs); sc.upload(seeds.data()); sg.upload(seeds.data());
  vt::RandomSample(cpu.q, ic.tensor, pc.tensor, sc.tensor); vt::RandomSample(gpu.q, ig.tensor, pg.tensor, sg.tensor);
  const auto first = ig.download(); xpu_test::SameBytes(first, ic.download());
  vt::RandomSample(gpu.q, ig.tensor, pg.tensor, sg.tensor); xpu_test::SameBytes(first, ig.download());
  int counts[V] = {};
  for (int i = 0; i < N; ++i) { int64_t token; std::memcpy(&token, first.data() + i * sizeof(token), sizeof(token)); REQUIRE(token >= 0); REQUIRE(token < V); ++counts[token]; }
  double chi_squared = 0;
  for (int j = 0; j < V; ++j) { const double expected = N * .1 * (j + 1); chi_squared += (counts[j] - expected) * (counts[j] - expected) / expected; }
  CHECK(chi_squared < 24); // 3 degrees of freedom, fixed seed corpus.
  CHECK(vt::GetReferenceTierHits() == 0);
}
TEST_CASE("XPU sampling: top-k and top-p draws follow the truncated distribution") {
  Queue gpu(vt::DeviceType::kXPU);
  constexpr int N = 6144, V = 8;
  const float weights[V] = {.01f, .02f, .04f, .08f, .12f, .18f, .24f, .31f};
  Buffer logits(gpu.q, DType::kF32, {N, V}), probs(gpu.q, DType::kF32, {N, V});
  Buffer ks(gpu.q, DType::kI32, {N}), ps(gpu.q, DType::kF32, {N});
  Buffer seeds(gpu.q, DType::kI64, {N}), ids(gpu.q, DType::kI64, {N});
  std::vector<float> values(N * V), p(N);
  std::vector<int32_t> k(N); std::vector<int64_t> seed(N);
  for (int i = 0; i < N; ++i) {
    // k-only and p-only retain {5,6,7}; combined filtering retains {6,7}.
    k[i] = i % 3 == 0 ? 3 : (i % 3 == 1 ? V : 4);
    p[i] = i % 3 == 0 ? 1.f : .6f; seed[i] = 419 + i;
    for (int j = 0; j < V; ++j) values[i * V + j] = std::log(weights[j]);
  }
  logits.put(values); ks.upload(k.data()); ps.put(p); seeds.upload(seed.data());
  vt::ApplyTopKTopP(gpu.q, logits.tensor, &ks.tensor, &ps.tensor);
  vt::ComputeProbs(gpu.q, probs.tensor, logits.tensor);
  vt::RandomSample(gpu.q, ids.tensor, probs.tensor, seeds.tensor);
  const auto bytes = ids.download();
  int counts[3][V] = {};
  for (int i = 0; i < N; ++i) {
    int64_t token; std::memcpy(&token, bytes.data() + i * sizeof(token), sizeof(token));
    REQUIRE(token >= (i % 3 == 2 ? 6 : 5)); REQUIRE(token < V); ++counts[i % 3][token];
  }
  for (int mode = 0; mode < 3; ++mode) {
    const int first = mode == 2 ? 6 : 5;
    const double sum = mode == 2 ? .55 : .73;
    double chi_squared = 0;
    for (int j = first; j < V; ++j) {
      const double expected = (N / 3) * weights[j] / sum;
      chi_squared += std::pow(counts[mode][j] - expected, 2) / expected;
    }
    CAPTURE(mode); CHECK(chi_squared < 24); // fixed seeds, at most 2 degrees of freedom
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}
TEST_CASE("XPU sampling: zero requests are a no-op") {
  Queue gpu(vt::DeviceType::kXPU);
  Buffer logits(gpu.q, DType::kF32, {1, 17}), params(gpu.q, DType::kF32, {1});
  Buffer ids(gpu.q, DType::kI64, {1}), seeds(gpu.q, DType::kI64, {1});
  // VT constructs positive allocations; empty batch views narrow their shape.
  for (auto* b : {&logits, &params, &ids, &seeds}) b->tensor.shape[0] = 0;
  vt::ApplyTemperature(gpu.q, logits.tensor, params.tensor, true);
  vt::ComputeProbs(gpu.q, logits.tensor, logits.tensor);
  vt::ComputeLogprobs(gpu.q, logits.tensor, logits.tensor);
  vt::ApplyMinP(gpu.q, logits.tensor, params.tensor);
  vt::ApplyTopKTopP(gpu.q, logits.tensor, nullptr, &params.tensor);
  vt::RandomSample(gpu.q, ids.tensor, logits.tensor, seeds.tensor);
  CHECK(vt::GetReferenceTierHits() == 0);
}
TEST_CASE("XPU sampling: complete mixed batch and seeded row reordering") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  vllm::v1::Sampler cs, gs;
  auto run = [](vt::Queue& q, vllm::v1::Sampler& sampler, std::vector<int> order) {
    constexpr int V = 257;
    Buffer logits(q, DType::kF32, {int64_t(order.size()), V});
    auto base = xpu_test::Values(4 * V, 19, .2f); std::vector<float> values;
    vllm::v1::SamplingMetadata sm; sm.all_greedy = false; sm.all_random = false;
    sm.temperature.emplace(); sm.top_k.emplace(); sm.top_p.emplace();
    sm.no_penalties = false; sm.prompt_token_ids.emplace();
    for (size_t r = 0; r < order.size(); ++r) {
      const int id = order[r]; values.insert(values.end(), base.begin() + id * V, base.begin() + (id + 1) * V);
      sm.temperature->push_back(id == 0 ? 0 : .7f); sm.top_k->push_back(20); sm.top_p->push_back(.87f);
      sm.generators[int(r)] = 719 + id; sm.output_token_positions.push_back(id + 3);
      sm.frequency_penalties.push_back(.2f); sm.presence_penalties.push_back(.1f); sm.repetition_penalties.push_back(1.1f);
      sm.prompt_token_ids->push_back({2, 3, 4}); sm.output_token_ids.push_back({5, 5, 7});
      sm.min_p.push_back(.1f); sm.logit_bias[int(r)][8] = 1.2f;
      sm.bad_words_token_ids[int(r)] = {{9}};
    }
    logits.put(values);
    return sampler.forward(q, logits.tensor, sm).sampled_token_ids;
  };
  const auto reference = run(cpu.q, cs, {0, 1, 2, 3}), actual = run(gpu.q, gs, {0, 1, 2, 3});
  CHECK(actual == reference);
  const auto reordered = run(gpu.q, gs, {3, 0, 2, 1});
  CHECK(reordered[0] == actual[3]); CHECK(reordered[1] == actual[0]);
  CHECK(reordered[2] == actual[2]); CHECK(reordered[3] == actual[1]);
  CHECK(vt::GetReferenceTierHits() == 0);
}
TEST_CASE("XPU sampling: workspace admission fails without modifying logits"
          * doctest::skip(!std::getenv("VT_B70_LOW_MEMORY_TEST"))) {
  Queue gpu(vt::DeviceType::kXPU);
  Buffer logits(gpu.q, DType::kF32, {4, 257}), p(gpu.q, DType::kF32, {4});
  const auto original = xpu_test::Values(4 * 257); logits.put(original); p.put({.9f, .9f, .9f, .9f});
  CHECK_THROWS_AS(vt::ApplyTopKTopP(gpu.q, logits.tensor, nullptr, &p.tensor), std::runtime_error);
  CHECK(logits.floats() == original);
  CHECK(vt::xpu::GetMemoryInfo().sampling_workspace_bytes == 0);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU sampled rejection preserves the target distribution for one-hot drafts") {
  Queue gpu(vt::DeviceType::kXPU);
  constexpr int N = 2048, R = 2 * N, V = 3;
  Buffer probs(gpu.q, DType::kF32, {R, V});
  Buffer proposal(gpu.q, DType::kI32, {R}), offsets(gpu.q, DType::kI32, {N + 1});
  Buffer seeds(gpu.q, DType::kI64, {R});
  Buffer greedy(gpu.q, DType::kI8, {R});
  Buffer choices(gpu.q, DType::kI32, {R}), accepted(gpu.q, DType::kI32, {R});
  Buffer sampled(gpu.q, DType::kI32, {N, 2}), counts(gpu.q, DType::kI32, {N});
  std::vector<float> p(size_t(R * V));
  std::vector<int32_t> d(static_cast<size_t>(R)), cu(static_cast<size_t>(N + 1));
  std::vector<int64_t> s(static_cast<size_t>(R));
  std::vector<int8_t> g(static_cast<size_t>(R), 0);
  for (int r = 0; r < N; ++r) {
    cu[size_t(r)] = 2 * r;
    for (int j = 0; j < 2; ++j) {
      const int row = 2 * r + j;
      p[size_t(row * V + 0)] = .2f;
      p[size_t(row * V + 1)] = .3f;
      p[size_t(row * V + 2)] = .5f;
      d[size_t(row)] = j == 0 ? 2 : -1;
      s[size_t(row)] = static_cast<int64_t>(vt::sample::SplitMix64(711 + 2 * r + j));
    }
  }
  cu.back() = R;
  probs.put(p); proposal.upload(d.data()); offsets.upload(cu.data()); seeds.upload(s.data());
  greedy.upload(g.data());
  vt::xpu::SampleOneHotRejection(gpu.q, sampled.tensor, counts.tensor,
                                 choices.tensor, accepted.tensor, probs.tensor,
                                 proposal.tensor, offsets.tensor, seeds.tensor, greedy.tensor);
  const auto read_i32 = [](const Buffer& buffer) {
    const auto bytes = buffer.download();
    std::vector<int32_t> out(bytes.size() / sizeof(int32_t));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
  };
  const auto tokens = read_i32(sampled), lengths = read_i32(counts);
  std::array<int, V> frequencies{};
  int accepted_drafts = 0;
  for (int r = 0; r < N; ++r) {
    const int n = lengths[size_t(r)], first = tokens[size_t(2 * r)];
    REQUIRE((n == 1 || n == 2));
    REQUIRE((first >= 0 && first < V));
    ++frequencies[size_t(first)];
    if (n == 2) {
      ++accepted_drafts;
      CHECK(first == 2);
      CHECK(tokens[size_t(2 * r + 1)] >= 0);
    } else {
      CHECK(first != 2);  // residual has zero support at the proposal id
      CHECK(tokens[size_t(2 * r + 1)] == -1);
    }
  }
  CHECK(std::abs(double(frequencies[0]) / N - .2) < .035);
  CHECK(std::abs(double(frequencies[1]) / N - .3) < .035);
  CHECK(std::abs(double(frequencies[2]) / N - .5) < .035);
  CHECK(std::abs(double(accepted_drafts) / N - .5) < .035);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU sampled rejection R07: MTP3 distributions and request-position RNG") {
  Queue gpu(vt::DeviceType::kXPU);
  constexpr int requests = 2048, width = 4, vocab = 3;
  const float probabilities[width][vocab] = {
      {.2f, .3f, .5f}, {.1f, .2f, .7f}, {.25f, .15f, .6f}, {.4f, .35f, .25f}};
  struct Result { std::vector<int32_t> tokens, counts; };
  const auto read = [](const Buffer& buffer) {
    const auto bytes = buffer.download();
    std::vector<int32_t> result(bytes.size() / sizeof(int32_t));
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
  };
  const auto run = [&](const std::vector<int>& order) {
    const int64_t n = static_cast<int64_t>(order.size()), rows = n * width;
    Buffer probs(gpu.q, DType::kF32, {rows, vocab});
    Buffer proposal(gpu.q, DType::kI32, {rows}), offsets(gpu.q, DType::kI32, {n + 1});
    Buffer seeds(gpu.q, DType::kI64, {rows}), greedy(gpu.q, DType::kI8, {rows});
    Buffer choices(gpu.q, DType::kI32, {rows}), accepted(gpu.q, DType::kI32, {rows});
    Buffer sampled(gpu.q, DType::kI32, {n, width}), counts(gpu.q, DType::kI32, {n});
    std::vector<float> p(rows * vocab);
    std::vector<int32_t> d(rows), cu(n + 1);
    std::vector<int64_t> keys(rows);
    std::vector<int8_t> deterministic(rows, 0);
    for (int64_t req = 0; req < n; ++req) {
      const int id = order[req];
      const uint64_t request_seed = 931 + uint64_t(id) * 19;
      const uint64_t committed_position = id % 7;
      cu[req] = static_cast<int32_t>(req * width);
      for (int depth = 0; depth < width; ++depth) {
        const auto row = req * width + depth;
        for (int token = 0; token < vocab; ++token) p[row * vocab + token] = probabilities[depth][token];
        d[row] = depth == width - 1 ? -1 : 2;
        // Same request/accepted-position/depth identity as the runner, with no
        // batch row in the key. Tokens0/1 are outside the deterministic proposal.
        keys[row] = static_cast<int64_t>(vt::sample::SplitMix64(
            request_seed + committed_position + uint64_t(depth)));
      }
    }
    cu.back() = static_cast<int32_t>(rows);
    probs.put(p); proposal.upload(d.data()); offsets.upload(cu.data());
    seeds.upload(keys.data()); greedy.upload(deterministic.data());
    vt::xpu::SampleOneHotRejection(gpu.q, sampled.tensor, counts.tensor, choices.tensor,
        accepted.tensor, probs.tensor, proposal.tensor, offsets.tensor, seeds.tensor, greedy.tensor);
    return Result{read(sampled), read(counts)};
  };
  std::vector<int> order(requests);
  for (int i = 0; i < requests; ++i) order[i] = i;
  const auto original = run(order);
  std::array<int, width> length_histogram{};
  int frequencies[width][vocab] = {};
  for (int req = 0; req < requests; ++req) {
    const int n = original.counts[req];
    REQUIRE(n >= 1); REQUIRE(n <= width);
    ++length_histogram[n - 1];
    for (int depth = 0; depth < n; ++depth) {
      const int token = original.tokens[req * width + depth];
      REQUIRE(token >= 0); REQUIRE(token < vocab);
      ++frequencies[depth][token];
      if (depth < n - 1) CHECK(token == 2);
      else if (n < width) CHECK(token != 2);
    }
    for (int depth = n; depth < width; ++depth) CHECK(original.tokens[req * width + depth] == -1);
  }
  for (int count : length_histogram) CHECK(count > 0);  // accepts0..3 all observed
  for (int depth = 0; depth < width; ++depth) {
    const int reached = frequencies[depth][0] + frequencies[depth][1] + frequencies[depth][2];
    REQUIRE(reached > 0);
    double chi_squared = 0;
    for (int token = 0; token < vocab; ++token) {
      const double expected = reached * double(probabilities[depth][token]);
      chi_squared += std::pow(frequencies[depth][token] - expected, 2) / expected;
    }
    CAPTURE(depth);
    CAPTURE(chi_squared);
    CHECK(chi_squared < 24);  // same fixed-corpus threshold as native sampling tests
    std::cout << "MTP3_DISTRIBUTION depth=" << depth << " reached=" << reached
              << " counts=" << frequencies[depth][0] << ',' << frequencies[depth][1]
              << ',' << frequencies[depth][2] << " chi2=" << chi_squared << '\n';
  }
  // Reorder and then condense while preserving request identity and accepted
  // position. These prove the operator's RNG contract, not a full-model C4 run.
  std::reverse(order.begin(), order.end());
  const auto reverse = run(order);
  for (int row = 0; row < requests; ++row) {
    const int id = order[row];
    CHECK(reverse.counts[row] == original.counts[id]);
    for (int depth = 0; depth < width; ++depth)
      CHECK(reverse.tokens[row * width + depth] == original.tokens[id * width + depth]);
  }
  const std::vector<int> condensed{13, 5, 7, 0};
  const auto compact = run(condensed);
  for (size_t row = 0; row < condensed.size(); ++row) {
    CHECK(compact.counts[row] == original.counts[condensed[row]]);
    for (int depth = 0; depth < width; ++depth)
      CHECK(compact.tokens[row * width + depth] == original.tokens[condensed[row] * width + depth]);
  }
  std::cout << "MTP3_ACCEPTED histogram=" << length_histogram[0] << ',' << length_histogram[1]
            << ',' << length_histogram[2] << ',' << length_histogram[3] << '\n';
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU one-hot rejection handles mixed greedy and sampled requests") {
  Queue gpu(vt::DeviceType::kXPU);
  constexpr int N = 3, R = 6, V = 3;
  Buffer probs(gpu.q, DType::kF32, {R, V});
  Buffer proposal(gpu.q, DType::kI32, {R}), offsets(gpu.q, DType::kI32, {N + 1});
  Buffer seeds(gpu.q, DType::kI64, {R}), greedy(gpu.q, DType::kI8, {R});
  Buffer choices(gpu.q, DType::kI32, {R}), accepted(gpu.q, DType::kI32, {R});
  Buffer sampled(gpu.q, DType::kI32, {N, 2}), counts(gpu.q, DType::kI32, {N});
  std::vector<float> p(R * V);
  for (int row = 0; row < R; ++row) {
    p[size_t(row * V)] = .2f;
    p[size_t(row * V + 1)] = .3f;
    p[size_t(row * V + 2)] = .5f;
  }
  const int32_t d[R] = {0, -1, 0, -1, 2, -1};
  const int32_t cu[N + 1] = {0, 2, 4, 6};
  const int64_t keys[R] = {1, 2, 3, 4, 5, 6};
  const int8_t modes[R] = {1, 1, 0, 0, 1, 1};
  probs.put(p); proposal.upload(d); offsets.upload(cu); seeds.upload(keys);
  greedy.upload(modes);
  vt::xpu::SampleOneHotRejection(gpu.q, sampled.tensor, counts.tensor,
                                 choices.tensor, accepted.tensor, probs.tensor,
                                 proposal.tensor, offsets.tensor, seeds.tensor,
                                 greedy.tensor);
  const auto token_bytes = sampled.download(), count_bytes = counts.download();
  std::vector<int32_t> tokens(R), lengths(N);
  std::memcpy(tokens.data(), token_bytes.data(), token_bytes.size());
  std::memcpy(lengths.data(), count_bytes.data(), count_bytes.size());
  CHECK(lengths[0] == 1);  // greedy proposal 0 rejected for argmax 2
  CHECK(tokens[0] == 2); CHECK(tokens[1] == -1);
  CHECK((lengths[1] == 1 || lengths[1] == 2));
  CHECK((tokens[2] >= 0 && tokens[2] < V));
  CHECK(lengths[2] == 2);  // greedy proposal 2 accepted; bonus is argmax 2
  CHECK(tokens[4] == 2); CHECK(tokens[5] == 2);
  CHECK(vt::GetReferenceTierHits() == 0);
}
