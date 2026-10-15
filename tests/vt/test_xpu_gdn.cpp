#include "xpu_test_helpers.h"
#include "vt/xpu.h"
#include "vt/breakable_graph.h"
#include <cstdlib>
#include <string>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace {
using vt::DType;
using xpu_test::Buffer;
using xpu_test::Queue;
using xpu_test::Values;
using xpu_test::Close;
using xpu_test::SameBytes;
struct SequentialReference {
  const bool had = std::getenv("VT_GDN_CHUNKED") != nullptr;
  const std::string old = had ? std::getenv("VT_GDN_CHUNKED") : "";
  SequentialReference() { setenv("VT_GDN_CHUNKED", "0", 1); }
  ~SequentialReference() { if (had) setenv("VT_GDN_CHUNKED", old.c_str(), 1); else unsetenv("VT_GDN_CHUNKED"); }
};
vt::Tensor Rows(const vt::Tensor& tensor, int64_t first, int64_t count) {
  auto view = tensor;
  view.data = static_cast<char*>(tensor.data) + first * tensor.stride[0] * vt::SizeOf(tensor.dtype);
  view.shape[0] = count;
  return view;
}
std::vector<float> Normalized(size_t count, int width, int salt) {
  auto data = Values(count, salt);
  for (size_t start = 0; start < count; start += width) {
    float sum = 1e-6f;
    for (int j = 0; j < width; ++j) sum += data[start + j] * data[start + j];
    const float inv = 1.0f / std::sqrt(sum);
    for (int j = 0; j < width; ++j) data[start + j] *= inv;
  }
  return data;
}
}

TEST_CASE("XPU conv prefill: raw history, lengths 0/1/2/3/4/63/64/65, dtype and row strides") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  const std::vector<int32_t> offsets{0, 0, 1, 3, 6, 10, 73, 137, 202};
  const int rows = 202, sequences = 8;
  for (int channels : {17, 10240}) for (auto dtype : {DType::kF32, DType::kBF16, DType::kF16})
    for (bool silu : {false, true}) {
      CAPTURE(channels);
      CAPTURE(dtype);
      CAPTURE(silu);
      std::vector<float> expected;
      std::vector<unsigned char> expected_state;
      for (auto* q : {&cpu.q, &gpu.q}) {
        Buffer x(*q, dtype, {rows, channels + 3}), w(*q, dtype, {channels, 4});
        Buffer out(*q, dtype == DType::kF16 ? DType::kF32 : dtype, {rows, channels});
        Buffer state(*q, DType::kF32, {sequences, channels, 3});
        Buffer qsl(*q, DType::kI32, {9}), init(*q, DType::kI8, {8}), bias(*q, DType::kF32, {channels});
        x.put(Values(rows * (channels + 3))); w.put(Values(channels * 4, 2));
        state.put(Values(sequences * channels * 3, 8)); bias.put(Values(channels, 6));
        const int8_t flags[] = {0, 1, 0, 1, 0, 1, 1, 0}; init.upload(flags); qsl.upload(offsets.data());
        x.tensor.shape[1] = channels;
        vt::CausalConv1dFwd(*q, out.tensor, x.tensor, w.tensor, &bias.tensor, state.tensor,
                             qsl.tensor, init.tensor, vt::CausalConv1dArgs{silu});
        if (q == &cpu.q) { expected = out.floats(); expected_state = state.download(); }
        else { Close(out.floats(), expected, dtype == DType::kBF16 ? 0.008f : 2e-6f);
               SameBytes(state.download(), expected_state); }
      }
    }
}

TEST_CASE("XPU conv long single-sequence prefill matches CPU output and state") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  constexpr int rows = 4096, channels = 17;
  std::vector<float> expected;
  std::vector<unsigned char> expected_state;
  for (auto* q : {&cpu.q, &gpu.q}) {
    Buffer x(*q, DType::kF16, {rows, channels + 3}), w(*q, DType::kF16, {channels, 4});
    Buffer out(*q, DType::kF32, {rows, channels});
    Buffer state(*q, DType::kF32, {1, channels, 3});
    Buffer qsl(*q, DType::kI32, {2}), init(*q, DType::kI8, {1});
    x.put(Values(rows * (channels + 3))); w.put(Values(channels * 4, 2));
    state.put(Values(channels * 3, 8));
    const int32_t offsets[] = {0, rows};
    const int8_t flags[] = {1};
    qsl.upload(offsets); init.upload(flags);
    x.tensor.shape[1] = channels;
    vt::CausalConv1dFwd(*q, out.tensor, x.tensor, w.tensor, nullptr, state.tensor,
                         qsl.tensor, init.tensor, {});
    if (q == &cpu.q) { expected = out.floats(); expected_state = state.download(); }
    else { Close(out.floats(), expected, 2e-6f); SameBytes(state.download(), expected_state); }
  }
}

TEST_CASE("XPU conv decode: permuted/null slots, wider history, in-place output and invalid metadata") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (auto dtype : {DType::kF32, DType::kBF16}) {
    std::vector<float> expected;
    std::vector<unsigned char> expected_state;
    for (auto* q : {&cpu.q, &gpu.q}) {
      Buffer x(*q, dtype, {3, 17}), w(*q, dtype, {17, 4}), out(*q, dtype, {3, 17});
      Buffer state(*q, DType::kF32, {4, 17, 6}), idx(*q, DType::kI32, {3});
      const int32_t indices[] = {2, -1, 0}; idx.upload(indices);
      x.put(Values(51)); out.put(Values(51)); w.put(Values(68, 3)); state.put(Values(4 * 17 * 6, 4));
      auto& target = q == &gpu.q ? x.tensor : out.tensor;
      vt::CausalConv1dUpdate(*q, target, x.tensor, w.tensor, nullptr, state.tensor,
                            vt::CausalConv1dArgs{}, &idx.tensor);
      if (q == &cpu.q) { expected = out.floats(); expected_state = state.download(); }
      else {
        Close(x.floats(), expected, dtype == DType::kBF16 ? 0.008f : 2e-6f);
        SameBytes(state.download(), expected_state);
        for (const auto& invalid : {std::vector<int32_t>{4, -1, 0}, std::vector<int32_t>{2, 2, 0}}) {
          idx.upload(invalid.data());
          CHECK_THROWS(vt::CausalConv1dUpdate(*q, out.tensor, x.tensor, w.tensor, nullptr,
                        state.tensor, vt::CausalConv1dArgs{}, &idx.tensor));
        }
      }
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == 0);
}

TEST_CASE("XPU speculative conv preserves accepted-prefix windows and rejects bad metadata") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  constexpr int channels = 7, rows = 7, taps = 4, state_len = 6;
  const std::vector<int32_t> offsets{0, 4, 6, 7};
  const std::vector<int32_t> slots{2, 0, -1};
  const auto inputs = Values(rows * (channels + 2), 3);
  const auto weights = Values(channels * taps, 4);
  const auto initial_state = Values(3 * channels * state_len, 5);
  const std::vector<float> initial_output(rows * channels, -1.0f);
  const auto biases = Values(channels, 7);
  for (auto dtype : {DType::kF32, DType::kF16})
    for (int accepted0 = 1; accepted0 <= 4; ++accepted0) {
      CAPTURE(dtype);
      CAPTURE(accepted0);
      const std::vector<int32_t> accepted{accepted0, 2, 1};
      std::vector<float> expected_output;
      std::vector<unsigned char> expected_state;
      for (auto* q : {&cpu.q, &gpu.q}) {
        Buffer x(*q, dtype, {rows, channels + 2});
        Buffer w(*q, dtype, {channels, taps});
        Buffer out(*q, q == &cpu.q ? DType::kF32 : dtype, {rows, channels});
        Buffer state(*q, DType::kF32, {3, channels, state_len});
        Buffer bias(*q, DType::kF32, {channels});
        Buffer cu(*q, DType::kI32, {4});
        Buffer idx(*q, DType::kI32, {3});
        Buffer nat(*q, DType::kI32, {3});
        x.put(inputs); x.tensor.shape[1] = channels;
        w.put(weights); out.put(initial_output); state.put(initial_state);
        bias.put(biases); cu.upload(offsets.data());
        idx.upload(slots.data()); nat.upload(accepted.data());
        vt::CausalConv1dSpecUpdate(*q, out.tensor, x.tensor, w.tensor,
                                   &bias.tensor, state.tensor, idx.tensor,
                                   nat.tensor, cu.tensor, {true});
        if (q == &cpu.q) {
          expected_output = out.floats();
          expected_state = state.download();
        } else {
          Close(out.floats(), expected_output,
                dtype == DType::kF16 ? 2e-3f : 2e-6f, 1e-4f);
          SameBytes(state.download(), expected_state);
          const auto unchanged = state.download();
          const std::vector<int32_t> bad_accepted{0, 2, 1};
          nat.upload(bad_accepted.data());
          CHECK_THROWS(vt::CausalConv1dSpecUpdate(*q, out.tensor, x.tensor,
              w.tensor, &bias.tensor, state.tensor, idx.tensor, nat.tensor,
              cu.tensor, {true}));
          SameBytes(state.download(), unchanged);
          nat.upload(accepted.data());
          const std::vector<int32_t> bad_slots{2, 2, -1};
          idx.upload(bad_slots.data());
          CHECK_THROWS(vt::CausalConv1dSpecUpdate(*q, out.tensor, x.tensor,
              w.tensor, &bias.tensor, state.tensor, idx.tensor, nat.tensor,
              cu.tensor, {true}));
          SameBytes(state.download(), unchanged);
        }
      }
    }
}

TEST_CASE("XPU speculative conv supports GPTQ FP16 persistent state") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  constexpr int channels = 5, rows = 3, taps = 4, state_len = 5;
  std::vector<float> inputs(rows * channels), weights(channels * taps);
  std::vector<float> initial_state(channels * state_len);
  for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = static_cast<float>(i % 7) / 8.0f;
  for (size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<float>(i % 5) / 4.0f;
  for (size_t i = 0; i < initial_state.size(); ++i)
    initial_state[i] = static_cast<float>(i % 9) / 8.0f;
  const int32_t offsets[] = {0, rows}, slots[] = {0};
  for (int32_t accepted_count : {1, 2, 3}) {
    std::vector<float> expected_output, expected_state;
    for (auto* q : {&cpu.q, &gpu.q}) {
      Buffer x(*q, DType::kF16, {rows, channels});
      Buffer w(*q, DType::kF16, {channels, taps});
      Buffer out(*q, q == &cpu.q ? DType::kF32 : DType::kF16,
                 {rows, channels});
      Buffer state(*q, q == &cpu.q ? DType::kF32 : DType::kF16,
                   {1, channels, state_len});
      Buffer cu(*q, DType::kI32, {2}), idx(*q, DType::kI32, {1});
      Buffer accepted(*q, DType::kI32, {1});
      x.put(inputs); w.put(weights); state.put(initial_state);
      cu.upload(offsets); idx.upload(slots); accepted.upload(&accepted_count);
      vt::CausalConv1dSpecUpdate(*q, out.tensor, x.tensor, w.tensor, nullptr,
                                 state.tensor, idx.tensor, accepted.tensor,
                                 cu.tensor, {});
      if (q == &cpu.q) {
        expected_output = out.floats();
        expected_state = state.floats();
      } else {
        Close(out.floats(), expected_output, 0.002f);
        Close(state.floats(), expected_state, 0.002f);
      }
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == 0);
}

TEST_CASE("XPU speculative conv R07 shortened verification retires rejected history") {
  Queue gpu(vt::DeviceType::kXPU);
  constexpr int channels = 10240, width = 3, slot = 1;
  for (int k : {1, 3}) for (int committed = 1; committed <= k + 1; ++committed) {
    CAPTURE(k);
    CAPTURE(committed);  // anchor plus accepted draft count 0..k
    const int count = k + 1, state_len = width + k, rows = count + 2;
    Buffer input(gpu.q, DType::kF16, {rows, channels});
    Buffer weights(gpu.q, DType::kF16, {channels, width + 1});
    Buffer cache(gpu.q, DType::kF16, {3, channels, state_len});
    Buffer serial(gpu.q, DType::kF16, {1, channels, width});
    Buffer verify(gpu.q, DType::kF16, {count, channels});
    Buffer next(gpu.q, DType::kF16, {1, channels});
    Buffer expected(gpu.q, DType::kF16, {1, channels});
    Buffer qsl(gpu.q, DType::kI32, {2}), index(gpu.q, DType::kI32, {1});
    Buffer accepted(gpu.q, DType::kI32, {1});
    input.put(Values(rows * channels, 73, .01f));
    weights.put(Values(channels * (width + 1), 74, .02f));
    const auto history = Values(channels * width, 75, .01f);
    std::vector<float> initial(3 * channels * state_len, .125f);
    for (int c = 0; c < channels; ++c) for (int j = 0; j < width; ++j)
      initial[(slot * channels + c) * state_len + j] = history[c * width + j];
    cache.put(initial); serial.put(history);
    const auto initial_bytes = cache.download();
    int32_t offsets[] = {0, count}, nat = 1, idx = slot;
    qsl.upload(offsets); accepted.upload(&nat); index.upload(&idx);
    auto provisional = Rows(input.tensor, 0, count);
    vt::CausalConv1dSpecUpdate(gpu.q, verify.tensor, provisional, weights.tensor,
        nullptr, cache.tensor, index.tensor, accepted.tensor, qsl.tensor, {true});
    // Independent one-token recurrence consumes only the valid prefix.
    for (int row = 0; row < committed; ++row) {
      auto token = Rows(input.tensor, row, 1);
      vt::CausalConv1dUpdate(gpu.q, expected.tensor, token, weights.tensor,
                            nullptr, serial.tensor, {true});
    }
    const auto before_short = cache.download();
    // A shortened provisional step must append its anchor after the two valid
    // history elements, even though the physical cache retains k spare taps.
    offsets[1] = 1; nat = committed;
    qsl.upload(offsets); accepted.upload(&nat);
    auto anchor = Rows(input.tensor, count, 1);
    vt::CausalConv1dSpecUpdate(gpu.q, next.tensor, anchor, weights.tensor,
        nullptr, cache.tensor, index.tensor, accepted.tensor, qsl.tensor, {true});
    vt::CausalConv1dUpdate(gpu.q, expected.tensor, anchor, weights.tensor,
                          nullptr, serial.tensor, {true});
    SameBytes(next.download(), expected.download());
    const auto state = cache.download(), serial_state = serial.download();
    bool spare_unchanged = true;
    for (int c = 0; c < channels; ++c) {
      const size_t base = static_cast<size_t>((slot * channels + c) * state_len) * 2;
      const size_t serial_base = static_cast<size_t>(c * width) * 2;
      spare_unchanged &= std::memcmp(state.data() + base + width * 2,
                                    before_short.data() + base + width * 2,
                                    (state_len - width) * 2) == 0;
      if (std::memcmp(state.data() + base, serial_state.data() + serial_base, width * 2)) {
        CAPTURE(c);
        FAIL_CHECK("shortened step left a rejected token in valid Conv history");
        break;
      }
    }
    CHECK(spare_unchanged);
    // Read again after the shortened step, which exposes latent stale history.
    nat = 1; accepted.upload(&nat);
    auto following = Rows(input.tensor, count + 1, 1);
    vt::CausalConv1dSpecUpdate(gpu.q, next.tensor, following, weights.tensor,
        nullptr, cache.tensor, index.tensor, accepted.tensor, qsl.tensor, {true});
    vt::CausalConv1dUpdate(gpu.q, expected.tensor, following, weights.tensor,
                          nullptr, serial.tensor, {true});
    SameBytes(next.download(), expected.download());
    const auto final_bytes = cache.download();
    const size_t page_bytes = channels * state_len * 2;
    for (int inactive : {0, 2})
      SameBytes(std::vector<unsigned char>(final_bytes.begin() + inactive * page_bytes,
                                         final_bytes.begin() + (inactive + 1) * page_bytes),
                std::vector<unsigned char>(initial_bytes.begin() + inactive * page_bytes,
                                         initial_bytes.begin() + (inactive + 1) * page_bytes));
  }
  CHECK(vt::GetReferenceTierHits() == 0);
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == 0);
}

TEST_CASE("XPU speculative GDN snapshots match CPU for every accepted prefix") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  constexpr int tokens = 5, hk = 2, hv = 4, dk = 8, dv = 8;
  const std::vector<int32_t> offsets{0, tokens};
  const std::vector<int32_t> slots{0, 1, 2, 3, 4};
  std::vector<float> initial(5 * hv * dv * dk, 0.0f);
  const auto first_slot = Values(hv * dv * dk, 9, 0.02f);
  std::copy(first_slot.begin(), first_slot.end(), initial.begin());
  auto gate = Values(tokens * hv, 4, 0.01f);
  auto beta = Values(tokens * hv, 5, 0.01f);
  for (auto& x : gate) x -= 0.2f;
  for (auto& x : beta) x += 0.5f;
  for (auto dtype : {DType::kF32, DType::kF16})
    for (int accepted_count = 1; accepted_count <= tokens; ++accepted_count) {
      CAPTURE(dtype);
      CAPTURE(accepted_count);
      std::vector<float> expected_first, expected_second, expected_state_first,
          expected_state_second;
      for (auto* q : {&cpu.q, &gpu.q}) {
        Buffer query(*q, dtype, {tokens, hk, dk});
        Buffer key(*q, dtype, {tokens, hk, dk});
        Buffer value(*q, dtype, {tokens, hv, dv});
        Buffer g(*q, DType::kF32, {tokens, hv});
        Buffer b(*q, DType::kF32, {tokens, hv});
        Buffer out(*q, q == &cpu.q ? DType::kF32 : dtype,
                   {tokens, hv, dv});
        Buffer state(*q, DType::kF32, {tokens, hv, dv, dk});
        Buffer cu(*q, DType::kI32, {2});
        Buffer idx(*q, DType::kI32, {1, tokens});
        Buffer nat(*q, DType::kI32, {1});
        query.put(Values(tokens * hk * dk, 1, 0.1f));
        key.put(Values(tokens * hk * dk, 2, 0.1f));
        value.put(Values(tokens * hv * dv, 3, 0.1f));
        g.put(gate); b.put(beta); state.put(initial);
        cu.upload(offsets.data()); idx.upload(slots.data());
        int32_t accepted = 1;
        nat.upload(&accepted);
        vt::GdnSpecDecode(*q, out.tensor, query.tensor, key.tensor,
                          value.tensor, g.tensor, b.tensor, state.tensor,
                          cu.tensor, idx.tensor, nat.tensor, {0.3535533906f});
        if (q == &cpu.q) {
          expected_first = out.floats();
          expected_state_first = state.floats();
        } else {
          Close(out.floats(), expected_first,
                dtype == DType::kF16 ? 2e-3f : 1e-4f, 1e-4f);
          Close(state.floats(), expected_state_first, 1e-4f, 1e-5f);
        }
        accepted = accepted_count;
        nat.upload(&accepted);
        vt::GdnSpecDecode(*q, out.tensor, query.tensor, key.tensor,
                          value.tensor, g.tensor, b.tensor, state.tensor,
                          cu.tensor, idx.tensor, nat.tensor, {0.3535533906f});
        if (q == &cpu.q) {
          expected_second = out.floats();
          expected_state_second = state.floats();
        } else {
          Close(out.floats(), expected_second,
                dtype == DType::kF16 ? 2e-3f : 1e-4f, 1e-4f);
          Close(state.floats(), expected_state_second, 1e-4f, 1e-5f);
          const auto unchanged = state.download();
          accepted = 0;
          nat.upload(&accepted);
          CHECK_THROWS(vt::GdnSpecDecode(*q, out.tensor, query.tensor,
              key.tensor, value.tensor, g.tensor, b.tensor, state.tensor,
              cu.tensor, idx.tensor, nat.tensor, {0.3535533906f}));
          SameBytes(state.download(), unchanged);
        }
      }
    }
}

TEST_CASE("XPU speculative GDN MTP1 uses 27B FP16 activations and FP32 snapshots") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  constexpr int tokens = 2, hk = 16, hv = 48, dk = 128, dv = 128;
  const std::vector<int32_t> offsets{0, tokens};
  const std::vector<int32_t> slots{0, 1};
  const std::vector<int32_t> accepted{1};
  std::vector<float> initial(tokens * hv * dv * dk, 0.0f);
  const auto first_slot = Values(hv * dv * dk, 11, 0.002f);
  std::copy(first_slot.begin(), first_slot.end(), initial.begin());
  auto gate = Values(tokens * hv, 12, 0.01f);
  auto beta = Values(tokens * hv, 13, 0.01f);
  for (auto& x : gate) x -= 0.2f;
  for (auto& x : beta) x += 0.5f;
  std::vector<float> expected_output, expected_state;
  for (auto* q : {&cpu.q, &gpu.q}) {
    Buffer query(*q, DType::kF16, {tokens, hk, dk});
    Buffer key(*q, DType::kF16, {tokens, hk, dk});
    Buffer value(*q, DType::kF16, {tokens, hv, dv});
    Buffer g(*q, DType::kF32, {tokens, hv});
    Buffer b(*q, DType::kF32, {tokens, hv});
    Buffer out(*q, q == &cpu.q ? DType::kF32 : DType::kF16,
               {tokens, hv, dv});
    Buffer state(*q, DType::kF32, {tokens, hv, dv, dk});
    Buffer cu(*q, DType::kI32, {2});
    Buffer idx(*q, DType::kI32, {1, tokens});
    Buffer nat(*q, DType::kI32, {1});
    query.put(Values(tokens * hk * dk, 1, 0.01f));
    key.put(Values(tokens * hk * dk, 2, 0.01f));
    value.put(Values(tokens * hv * dv, 3, 0.01f));
    g.put(gate); b.put(beta); state.put(initial);
    cu.upload(offsets.data()); idx.upload(slots.data());
    nat.upload(accepted.data());
    vt::GdnSpecDecode(*q, out.tensor, query.tensor, key.tensor,
                      value.tensor, g.tensor, b.tensor, state.tensor,
                      cu.tensor, idx.tensor, nat.tensor, {0.0883883476f});
    if (q == &cpu.q) {
      expected_output = out.floats();
      expected_state = state.floats();
    } else {
      Close(out.floats(), expected_output, 2e-3f, 1e-4f);
      Close(state.floats(), expected_state, 1e-4f, 1e-5f);
    }
  }
}

TEST_CASE("XPU P5 speculative GDN: bounded Q4 C1 C4 baseline and complete snapshots") {
  const char* request_env = std::getenv("VT_B70_GDN_BENCH_REQUESTS");
  if (!request_env) { MESSAGE("Set VT_B70_GDN_BENCH_REQUESTS for P5 baseline inspection"); return; }
  REQUIRE((std::string(request_env) == "1" || std::string(request_env) == "4"));
  const int requests = std::atoi(request_env), tokens = 4 * requests;
  constexpr int hk = 16, hv = 48, dk = 128, dv = 128, cols = 4;
  constexpr size_t slot_elements = size_t(hv) * dv * dk;
  const int slots = requests * cols + 2;
  Queue gpu(vt::DeviceType::kXPU);
  Buffer query(gpu.q, DType::kF16, {tokens, hk, dk}), key(gpu.q, DType::kF16, {tokens, hk, dk});
  Buffer value(gpu.q, DType::kF16, {tokens, hv, dv}), g(gpu.q, DType::kF32, {tokens, hv});
  const auto output_dtype = std::getenv("VT_B70_GDN_OUTPUT_F32") ? DType::kF32 : DType::kF16;
  Buffer beta(gpu.q, DType::kF32, {tokens, hv}), out(gpu.q, output_dtype, {tokens, hv, dv});
  Buffer state(gpu.q, DType::kF32, {slots, hv, dv, dk});
  Buffer cu(gpu.q, DType::kI32, {requests + 1}), ids(gpu.q, DType::kI32, {requests, cols});
  Buffer accepted(gpu.q, DType::kI32, {requests});
  query.put(Values(tokens * hk * dk, 31, .01f)); key.put(Values(tokens * hk * dk, 32, .01f));
  value.put(Values(tokens * hv * dv, 33, .01f));
  auto gates = Values(tokens * hv, 34, .01f), betas = Values(tokens * hv, 35, .01f);
  for (auto& x : gates) x -= .3f;
  for (auto& x : betas) x += .5f;
  g.put(gates); beta.put(betas);
  auto initial = Values(size_t(slots) * slot_elements, 36, .002f);
  const uint32_t poison = 0x7fc12345u;
  for (size_t j = 0; j < slot_elements; ++j) {
    std::memcpy(initial.data() + j, &poison, 4);
    std::memcpy(initial.data() + size_t(slots - 1) * slot_elements + j, &poison, 4);
  }
  state.put(initial);
  std::vector<int32_t> offsets(requests + 1), indices(requests * cols), counts(requests);
  for (int r = 0; r < requests; ++r) {
    offsets[r] = r * cols; counts[r] = r % cols + 1;
    for (int c = 0; c < cols; ++c) indices[r * cols + c] = 1 + r * cols + c;
  }
  offsets.back() = tokens; cu.upload(offsets.data()); ids.upload(indices.data()); accepted.upload(counts.data());
  const auto before = state.download(), qb = query.download(), kb = key.download(), vb = value.download();
  const auto gb = g.download(), bb = beta.download();
  auto& backend = vt::GetBackend(gpu.q.device);
  auto execute = [&] { vt::GdnSpecDecode(gpu.q, out.tensor, query.tensor, key.tensor, value.tensor,
      g.tensor, beta.tensor, state.tensor, cu.tensor, ids.tensor, accepted.tensor, {.0883883476f}); };
  execute(); const auto expected_out = out.download(), expected_state = state.download();
  for (auto x : out.floats()) REQUIRE(std::isfinite(x));
  for (int warm = 0; warm < 2; ++warm) { state.upload(before.data()); execute(); backend.Synchronize(gpu.q); }
  (void)vt::xpu::DrainProfileEvents();
  std::vector<double> ms;
  for (int sample = 0; sample < 3; ++sample) {
    // Restore the identical initial slots outside the operator timer.
    state.upload(before.data()); const auto start = std::chrono::steady_clock::now();
    execute(); backend.Synchronize(gpu.q);
    ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
  }
  SameBytes(out.download(), expected_out); SameBytes(state.download(), expected_state);
  SameBytes(query.download(), qb); SameBytes(key.download(), kb); SameBytes(value.download(), vb);
  SameBytes(g.download(), gb); SameBytes(beta.download(), bb);
  SameBytes(std::vector<unsigned char>(expected_state.begin(), expected_state.begin() + slot_elements * 4),
            std::vector<unsigned char>(before.begin(), before.begin() + slot_elements * 4));
  SameBytes(std::vector<unsigned char>(expected_state.end() - slot_elements * 4, expected_state.end()),
            std::vector<unsigned char>(before.end() - slot_elements * 4, before.end()));
  nlohmann::json report = {{"requests", requests}, {"queries_each", cols}, {"hk", hk}, {"hv", hv},
      {"dk", dk}, {"dv", dv}, {"output_f32", output_dtype == DType::kF32},
      {"accepted_selectors", counts}, {"state_bytes", state.bytes},
      {"complete_operator_wall_ms", ms}, {"seed", "deterministic synthetic distinct slots; not original model operands"},
      {"profiled", std::getenv("VT_XPU_PROFILE") && std::string(std::getenv("VT_XPU_PROFILE")) == "1"}};
  report["device_events"] = nlohmann::json::array();
  for (const auto& e : vt::xpu::DrainProfileEvents())
    if (e.stage.find("gdn_spec_decode") == 0)
      report["device_events"].push_back({{"stage", e.stage}, {"ms", (e.end_ns - e.start_ns) / 1e6}});
  if (const char* dir = std::getenv("VT_B70_GDN_BASELINE_OUTPUT")) {
    const auto root = std::filesystem::path(dir); std::filesystem::create_directories(root);
    for (const auto& entry : {std::make_pair("initial.f32", &before),
                             std::make_pair("snapshots.f32", &expected_state),
                             std::make_pair(output_dtype == DType::kF32 ? "output.f32" : "output.f16", &expected_out)}) {
      const auto path = root / entry.first; REQUIRE_FALSE(std::filesystem::exists(path));
      std::ofstream f(path, std::ios::binary); f.write(reinterpret_cast<const char*>(entry.second->data()), entry.second->size());
      f.close(); REQUIRE(f.good());
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
  std::cout << "P5_GDN_BASELINE " << report.dump() << std::endl;
}

static void CheckSpecSnapshotRollback(int first_rows) {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  const int rows = first_rows + 1;
  constexpr int hk = 16, hv = 48, dk = 128, dv = 128;
  constexpr float scale = 0.0883883476f;
  const auto q_values = Values(rows * hk * dk, 21, 0.01f);
  const auto k_values = Values(rows * hk * dk, 22, 0.01f);
  const auto v_values = Values(rows * hv * dv, 23, 0.01f);
  auto g_values = Values(rows * hv, 24, 0.01f);
  auto b_values = Values(rows * hv, 25, 0.01f);
  for (auto& x : g_values) x -= 0.2f;
  for (auto& x : b_values) x += 0.5f;
  const auto initial_row = Values(hv * dv * dk, 26, 0.002f);
  std::vector<float> initial((first_rows + 1) * hv * dv * dk, 0.125f);
  std::copy(initial_row.begin(), initial_row.end(), initial.begin());
  const int32_t slots[] = {0, 1, 2, 3, 4};

  for (int32_t previous_accepted = 1; previous_accepted <= first_rows;
       ++previous_accepted) {
    CAPTURE(previous_accepted);
    Buffer qg(gpu.q, DType::kF16, {rows, hk, dk});
    Buffer kg(gpu.q, DType::kF16, {rows, hk, dk});
    Buffer vg(gpu.q, DType::kF16, {rows, hv, dv});
    Buffer gg(gpu.q, DType::kF32, {rows, hv});
    Buffer bg(gpu.q, DType::kF32, {rows, hv});
    Buffer og(gpu.q, DType::kF16, {first_rows, hv, dv});
    Buffer last_gpu(gpu.q, DType::kF16, {1, hv, dv});
    Buffer sg(gpu.q, DType::kF32, {first_rows + 1, hv, dv, dk});
    Buffer cu(gpu.q, DType::kI32, {2});
    Buffer idx(gpu.q, DType::kI32, {1, first_rows});
    Buffer nat(gpu.q, DType::kI32, {1});
    qg.put(q_values); kg.put(k_values); vg.put(v_values);
    gg.put(g_values); bg.put(b_values); sg.put(initial);
    idx.upload(slots);
    const auto initial_bytes = sg.download();
    int32_t offsets[] = {0, first_rows};
    int32_t accepted = 1;
    cu.upload(offsets); nat.upload(&accepted);
    auto q_first = Rows(qg.tensor, 0, first_rows);
    auto k_first = Rows(kg.tensor, 0, first_rows);
    auto v_first = Rows(vg.tensor, 0, first_rows);
    auto g_first = Rows(gg.tensor, 0, first_rows);
    auto b_first = Rows(bg.tensor, 0, first_rows);
    vt::GdnSpecDecode(gpu.q, og.tensor, q_first, k_first, v_first,
                      g_first, b_first, sg.tensor, cu.tensor, idx.tensor,
                      nat.tensor, {scale});
    offsets[1] = 1;
    accepted = previous_accepted;
    cu.upload(offsets); nat.upload(&accepted);
    auto q_last = Rows(qg.tensor, first_rows, 1);
    auto k_last = Rows(kg.tensor, first_rows, 1);
    auto v_last = Rows(vg.tensor, first_rows, 1);
    auto g_last = Rows(gg.tensor, first_rows, 1);
    auto b_last = Rows(bg.tensor, first_rows, 1);
    vt::GdnSpecDecode(gpu.q, last_gpu.tensor, q_last, k_last, v_last,
                      g_last, b_last, sg.tensor, cu.tensor, idx.tensor,
                      nat.tensor, {scale});

    // Independent teacher-forced serial decode: consume only the committed
    // prefix of the first verification, then the next anchor token.
    Buffer qc(cpu.q, DType::kF16, {rows, hk, dk});
    Buffer kc(cpu.q, DType::kF16, {rows, hk, dk});
    Buffer vc(cpu.q, DType::kF16, {rows, hv, dv});
    Buffer gc(cpu.q, DType::kF32, {rows, hv});
    Buffer bc(cpu.q, DType::kF32, {rows, hv});
    Buffer oc(cpu.q, DType::kF32, {1, hv, dv});
    Buffer sc(cpu.q, DType::kF32, {1, hv, dv, dk});
    qc.put(q_values); kc.put(k_values); vc.put(v_values);
    gc.put(g_values); bc.put(b_values); sc.put(initial_row);
    for (int t = 0; t <= previous_accepted; ++t) {
      const int row = t == previous_accepted ? first_rows : t;
      auto qr = Rows(qc.tensor, row, 1);
      auto kr = Rows(kc.tensor, row, 1);
      auto vr = Rows(vc.tensor, row, 1);
      auto gr = Rows(gc.tensor, row, 1);
      auto br = Rows(bc.tensor, row, 1);
      vt::GdnDecode(cpu.q, oc.tensor, qr, kr, vr, gr, br, sc.tensor,
                    {scale});
    }
    Close(last_gpu.floats(), oc.floats(), 2e-3f, 1e-4f);
    const auto gpu_state = sg.floats();
    Close(std::vector<float>(gpu_state.begin(),
                             gpu_state.begin() + hv * dv * dk),
          sc.floats(), 1e-4f, 1e-5f);
    const auto final_bytes = sg.download();
    const size_t inactive = static_cast<size_t>(first_rows) * hv * dv * dk * sizeof(float);
    SameBytes(std::vector<unsigned char>(final_bytes.begin() + inactive, final_bytes.end()),
              std::vector<unsigned char>(initial_bytes.begin() + inactive, initial_bytes.end()));
  }
  CHECK(vt::GetReferenceTierHits() == 0);
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == 0);
}

TEST_CASE("XPU speculative GDN MTP4 restores every accepted 27B snapshot") {
  CheckSpecSnapshotRollback(5);
}

TEST_CASE("XPU speculative GDN R07 fixed MTP1/MTP3 commits every prefix") {
  for (int rows : {2, 4}) {
    CAPTURE(rows);
    CheckSpecSnapshotRollback(rows);
  }
}

TEST_CASE("XPU conv full prefill and in-place prefill equal split prefill plus decode") {
  Queue gpu(vt::DeviceType::kXPU);
  for (int length : {1, 2, 3, 4, 63, 64, 65}) {
    constexpr int channels = 17;
    Buffer input(gpu.q, DType::kBF16, {length, channels}), all(gpu.q, DType::kBF16, {length, channels});
    Buffer split(gpu.q, DType::kBF16, {length, channels}), weights(gpu.q, DType::kBF16, {channels, 4});
    Buffer state(gpu.q, DType::kF32, {1, channels, 3}), qsl(gpu.q, DType::kI32, {2}), init(gpu.q, DType::kI32, {1});
    const auto original = Values(length * channels, 3), original_state = Values(channels * 3, 7);
    input.put(original); weights.put(Values(channels * 4)); state.put(original_state);
    int32_t offsets[] = {0, length}, flags[] = {1}; qsl.upload(offsets); init.upload(flags);
    vt::CausalConv1dFwd(gpu.q, all.tensor, input.tensor, weights.tensor, nullptr,
                         state.tensor, qsl.tensor, init.tensor, {});
    const auto expected_state = state.download();
    state.put(original_state);
    vt::CausalConv1dFwd(gpu.q, input.tensor, input.tensor, weights.tensor, nullptr,
                         state.tensor, qsl.tensor, init.tensor, {});
    SameBytes(input.download(), all.download()); SameBytes(state.download(), expected_state);
    input.put(original); state.put(original_state);
    const int prefix = std::min(length, 3); offsets[1] = prefix; qsl.upload(offsets);
    auto out = Rows(split.tensor, 0, prefix);
    vt::CausalConv1dFwd(gpu.q, out, Rows(input.tensor, 0, prefix), weights.tensor, nullptr,
                         state.tensor, qsl.tensor, init.tensor, {});
    for (int t = prefix; t < length; ++t) {
      out = Rows(split.tensor, t, 1);
      vt::CausalConv1dUpdate(gpu.q, out, Rows(input.tensor, t, 1), weights.tensor, nullptr, state.tensor, {});
    }
    SameBytes(split.download(), all.download()); SameBytes(state.download(), expected_state);
  }
}

TEST_CASE("XPU GDN post-conv and gated RMSNorm: actual heads, strided gates, one normalization") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (auto dtype : {DType::kF32, DType::kBF16}) for (bool sigmoid : {false, true}) {
    constexpr int t = 3, hk = 16, hv = 48, dk = 128, dv = 128, channels = 10240;
    std::vector<std::vector<float>> expected;
    for (auto* q : {&cpu.q, &gpu.q}) {
      Buffer conv(*q, dtype, {t, channels}), a(*q, dtype, {t, hv + 3}), b(*q, dtype, {t, hv + 3});
      Buffer al(*q, DType::kF32, {hv}), dt(*q, DType::kF32, {hv});
      Buffer qo(*q, dtype, {t, hk, dk}), ko(*q, dtype, {t, hk, dk}), vo(*q, dtype, {t, hv, dv});
      Buffer go(*q, DType::kF32, {t, hv}), bo(*q, DType::kF32, {t, hv});
      conv.put(Values(t * channels)); a.put(Values(t * (hv + 3), 7, 2)); b.put(Values(t * (hv + 3), 3));
      al.put(Values(hv, 3)); dt.put(Values(hv, 9)); a.tensor.shape[1] = b.tensor.shape[1] = hv;
      vt::GdnPostConv(*q, qo.tensor, ko.tensor, vo.tensor, go.tensor, bo.tensor, conv.tensor,
                      a.tensor, b.tensor, al.tensor, dt.tensor, {1e-6f, false});
      std::vector<std::vector<float>> actual{qo.floats(), ko.floats(), vo.floats(), go.floats(), bo.floats()};
      Buffer gate(*q, dtype, {t, hv + 1, dv}), weight(*q, dtype, {dv});
      Buffer out(*q, dtype, {t, hv, dv}); gate.put(Values(t * (hv + 1) * dv, 3)); weight.put(Values(dv, 5));
      gate.tensor.shape[1] = hv;
      vt::RmsNormGated(*q, out.tensor, vo.tensor, gate.tensor, weight.tensor, {1e-6f, sigmoid});
      actual.push_back(out.floats());
      if (q == &cpu.q) expected = actual;
      else for (size_t i = 0; i < actual.size(); ++i)
        Close(actual[i], expected[i], dtype == DType::kBF16 && (i < 3 || i == 5) ? 0.008f : 3e-6f);
    }
  }
}

TEST_CASE("XPU GDN post-conv subgroup: FP16 matches native scalar") {
  Queue gpu(vt::DeviceType::kXPU);
  constexpr int hk = 16, hv = 48, dk = 128, dv = 128;
  constexpr int channels = 2 * hk * dk + hv * dv;
  for (int t : {1, 256}) {
    CAPTURE(t);
    Buffer conv(gpu.q, DType::kF16, {t, channels});
    Buffer a(gpu.q, DType::kF16, {t, hv + 3});
    Buffer b(gpu.q, DType::kF16, {t, hv + 3});
    Buffer al(gpu.q, DType::kF32, {hv});
    Buffer dt(gpu.q, DType::kF32, {hv});
    Buffer qo(gpu.q, DType::kF16, {t, hk, dk});
    Buffer ko(gpu.q, DType::kF16, {t, hk, dk});
    Buffer vo(gpu.q, DType::kF16, {t, hv, dv});
    Buffer go(gpu.q, DType::kF32, {t, hv});
    Buffer bo(gpu.q, DType::kF32, {t, hv});
    conv.put(Values(t * channels));
    a.put(Values(t * (hv + 3), 7, 2));
    b.put(Values(t * (hv + 3), 3));
    al.put(Values(hv, 3)); dt.put(Values(hv, 9));
    a.tensor.shape[1] = b.tensor.shape[1] = hv;
    const auto run = [&] {
      vt::GdnPostConv(gpu.q, qo.tensor, ko.tensor, vo.tensor, go.tensor,
                      bo.tensor, conv.tensor, a.tensor, b.tensor, al.tensor,
                      dt.tensor, {1e-6f, false});
      vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
    };
    setenv("VT_XPU_GDN_POSTCONV_SUBGROUP", "0", 1);
    if (std::getenv("VT_XPU_PROFILE"))
      (void)vt::xpu::DrainProfileEvents();
    run();
    if (std::getenv("VT_XPU_PROFILE")) {
      int scalar = 0;
      for (const auto& event : vt::xpu::DrainProfileEvents())
        scalar += event.stage == "gdn_postconv";
      CHECK(scalar == 1);
    }
    const std::vector<std::vector<float>> expected{
        qo.floats(), ko.floats(), vo.floats(), go.floats(), bo.floats()};
    unsetenv("VT_XPU_GDN_POSTCONV_SUBGROUP");
    run();
    if (std::getenv("VT_XPU_PROFILE")) {
      int subgroup = 0;
      for (const auto& event : vt::xpu::DrainProfileEvents())
        subgroup += event.stage == "gdn_postconv_subgroup";
      CHECK(subgroup == 1);
    }
    const std::vector<std::vector<float>> actual{
        qo.floats(), ko.floats(), vo.floats(), go.floats(), bo.floats()};
    for (size_t i = 0; i < actual.size(); ++i)
      Close(actual[i], expected[i], i < 3 ? 0.002f : 3e-6f);
  }
}

TEST_CASE("XPU GDN recurrence: varlen, Hv/Hk=3, complete output and F32 state") {
  SequentialReference mode;
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (auto dtype : {DType::kF32, DType::kBF16}) for (int length : {1, 2, 3, 4, 63, 64, 65}) {
    const int hk = length == 4 ? 16 : 2, hv = 3 * hk;
    const int dk = length == 4 ? 128 : 9, dv = length == 4 ? 128 : 7, tokens = length + 1;
    CAPTURE(dtype);
    CAPTURE(length);
    std::vector<float> expected, expected_state;
    for (auto* q : {&cpu.q, &gpu.q}) {
      Buffer qi(*q, dtype, {tokens, hk, dk}), ki(*q, dtype, {tokens, hk, dk});
      Buffer vi(*q, dtype, {tokens, hv, dv}), out(*q, dtype, {tokens, hv, dv});
      Buffer g(*q, DType::kF32, {tokens, hv}), beta(*q, DType::kF32, {tokens, hv});
      Buffer state(*q, DType::kF32, {3, hv, dv, dk}), qsl(*q, DType::kI32, {4});
      qi.put(Normalized(tokens * hk * dk, dk, 0)); ki.put(Normalized(tokens * hk * dk, dk, 3));
      vi.put(Values(tokens * hv * dv, 4)); g.put(std::vector<float>(tokens * hv, -0.13f));
      beta.put(std::vector<float>(tokens * hv, 0.61f)); state.put(Values(3 * hv * dv * dk, 5));
      const int32_t offsets[] = {0, 1, tokens, tokens}; qsl.upload(offsets);
      vt::GdnPrefill(*q, out.tensor, qi.tensor, ki.tensor, vi.tensor, g.tensor, beta.tensor,
                     state.tensor, qsl.tensor, {1.0f / std::sqrt(float(dk))});
      if (q == &cpu.q) { expected = out.floats(); expected_state = state.floats(); }
      else {
        Close(out.floats(), expected, dtype == DType::kBF16 ? 0.008f : 2e-5f);
        Close(state.floats(), expected_state, 2e-5f);
        const int32_t invalid[] = {0, tokens, 1, tokens}; qsl.upload(invalid);
        CHECK_THROWS(vt::GdnPrefill(*q, out.tensor, qi.tensor, ki.tensor, vi.tensor, g.tensor,
                      beta.tensor, state.tensor, qsl.tensor, {1.0f}));
      }
    }
  }
}

TEST_CASE("XPU GDN SG16 gated norm: F16, strided gate and in-place output") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  constexpr int tokens = 2, heads = 48, width = 128;
  for (bool sigmoid : {false, true}) {
    CAPTURE(sigmoid);
    std::vector<float> expected;
    for (auto* q : {&cpu.q, &gpu.q}) {
      const auto type = q == &cpu.q ? DType::kF32 : DType::kF16;
      Buffer x(*q, type, {tokens, heads, width});
      Buffer gate(*q, type, {tokens, heads + 1, width});
      Buffer weight(*q, type, {width}), out(*q, type, {tokens, heads, width});
      auto put_rounded = [&](Buffer& buffer, std::vector<float> values) {
        if (q == &cpu.q) for (auto& value : values) value = vt::F16ToF32(vt::F32ToF16(value));
        buffer.put(values);
      };
      put_rounded(x, Values(tokens * heads * width, 3));
      put_rounded(gate, Values(tokens * (heads + 1) * width, 7));
      put_rounded(weight, Values(width, 9, 0.02f));
      gate.tensor.shape[1] = heads;
      auto& target = q == &cpu.q ? out.tensor : x.tensor;
      vt::RmsNormGated(*q, target, x.tensor, gate.tensor, weight.tensor, {1e-6f, sigmoid});
      if (q == &cpu.q) expected = out.floats();
      else Close(x.floats(), expected, 0.003f, 0.0002f);
    }
  }
}

TEST_CASE("XPU GDN full prefill equals split prefill plus decode, including every state value") {
  Queue gpu(vt::DeviceType::kXPU);
  for (int length : {1, 2, 3, 4, 63, 64, 65}) {
    constexpr int hk = 2, hv = 6, dk = 9, dv = 7;
    Buffer qi(gpu.q, DType::kBF16, {length, hk, dk}), ki(gpu.q, DType::kBF16, {length, hk, dk});
    Buffer vi(gpu.q, DType::kBF16, {length, hv, dv}), all(gpu.q, DType::kBF16, {length, hv, dv});
    Buffer split(gpu.q, DType::kBF16, {length, hv, dv});
    Buffer g(gpu.q, DType::kF32, {length, hv}), beta(gpu.q, DType::kF32, {length, hv});
    Buffer state(gpu.q, DType::kF32, {1, hv, dv, dk}), qsl(gpu.q, DType::kI32, {2});
    qi.put(Normalized(length * hk * dk, dk, 0)); ki.put(Normalized(length * hk * dk, dk, 3));
    vi.put(Values(length * hv * dv, 4)); g.put(std::vector<float>(length * hv, -0.13f));
    beta.put(std::vector<float>(length * hv, 0.61f)); state.put(Values(hv * dv * dk, 5));
    int32_t offsets[] = {0, length}; qsl.upload(offsets);
    vt::GdnPrefill(gpu.q, all.tensor, qi.tensor, ki.tensor, vi.tensor, g.tensor, beta.tensor,
                   state.tensor, qsl.tensor, {1.0f / 3.0f});
    const auto expected_state = state.download(); state.put(Values(hv * dv * dk, 5));
    const int prefix = std::min(length, 3); offsets[1] = prefix; qsl.upload(offsets);
    auto o = Rows(split.tensor, 0, prefix);
    vt::GdnPrefill(gpu.q, o, Rows(qi.tensor, 0, prefix), Rows(ki.tensor, 0, prefix),
                   Rows(vi.tensor, 0, prefix), Rows(g.tensor, 0, prefix), Rows(beta.tensor, 0, prefix),
                   state.tensor, qsl.tensor, {1.0f / 3.0f});
    for (int t = prefix; t < length; ++t) {
      o = Rows(split.tensor, t, 1);
      vt::GdnDecode(gpu.q, o, Rows(qi.tensor, t, 1), Rows(ki.tensor, t, 1), Rows(vi.tensor, t, 1),
                    Rows(g.tensor, t, 1), Rows(beta.tensor, t, 1), state.tensor, {1.0f / 3.0f});
    }
    SameBytes(split.download(), all.download()); SameBytes(state.download(), expected_state);
  }
}

TEST_CASE("XPU GDN state gather/scatter and decode: permutation, null slot, preserved cache rows") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (auto cache_type : {DType::kF32, DType::kBF16}) {
    std::vector<float> expected_cache, expected_work, expected_output, expected_state;
    for (auto* q : {&cpu.q, &gpu.q}) {
      Buffer cache(*q, cache_type, {4, 6, 6}), work(*q, DType::kF32, {3, 6, 3});
      Buffer ids(*q, DType::kI32, {3}), flags(*q, DType::kI32, {3});
      const int32_t indices[] = {2, 0, 3}, initial[] = {1, 0, 1}; ids.upload(indices); flags.upload(initial);
      cache.put(Values(4 * 6 * 6));
      vt::GdnStateGather(*q, work.tensor, cache.tensor, ids.tensor, &flags.tensor);
      const auto gathered = work.floats(); work.put(Values(3 * 6 * 3, 9));
      vt::GdnStateScatter(*q, cache.tensor, work.tensor, ids.tensor);
      Buffer qi(*q, DType::kBF16, {3, 2, 9}), ki(*q, DType::kBF16, {3, 2, 9});
      Buffer vi(*q, DType::kBF16, {3, 6, 7}), out(*q, DType::kBF16, {3, 6, 7});
      Buffer g(*q, DType::kF32, {3, 6}), beta(*q, DType::kF32, {3, 6}), state(*q, DType::kF32, {4, 6, 7, 9});
      qi.put(Normalized(3 * 2 * 9, 9, 0)); ki.put(Normalized(3 * 2 * 9, 9, 3)); vi.put(Values(3 * 6 * 7));
      g.put(std::vector<float>(18, -0.13f)); beta.put(std::vector<float>(18, 0.61f)); state.put(Values(4 * 6 * 7 * 9));
      const int32_t slots[] = {2, -1, 0}; ids.upload(slots);
      vt::GdnDecode(*q, out.tensor, qi.tensor, ki.tensor, vi.tensor, g.tensor, beta.tensor, state.tensor,
                    {1.0f / 3.0f}, &ids.tensor);
      if (q == &cpu.q) {
        expected_cache = cache.floats(); expected_work = gathered;
        expected_output = out.floats(); expected_state = state.floats();
      } else {
        CHECK(cache.floats() == expected_cache); CHECK(gathered == expected_work);
        Close(out.floats(), expected_output, 0.008f); Close(state.floats(), expected_state, 2e-5f);
      }
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == 0);
}

TEST_CASE("XPU GDN SG16 decode: nonzero F32 state, repeated permuted and null slots") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  constexpr int batch = 3, slots = 4, hk = 2, hv = 6, dk = 128, dv = 128;
  std::vector<std::vector<float>> expected_outputs;
  std::vector<float> expected_state;
  for (auto* q : {&cpu.q, &gpu.q}) {
    const auto type = q == &cpu.q ? DType::kF32 : DType::kF16;
    Buffer qi(*q, type, {batch, hk, dk}), ki(*q, type, {batch, hk, dk});
    Buffer vi(*q, type, {batch, hv, dv}), out(*q, type, {batch, hv, dv});
    Buffer g(*q, DType::kF32, {batch, hv}), beta(*q, DType::kF32, {batch, hv});
    Buffer state(*q, DType::kF32, {slots, hv, dv, dk}), ids(*q, DType::kI32, {batch});
    state.put(Values(slots * hv * dv * dk, 5, 0.002f));
    std::vector<std::vector<float>> outputs;
    for (int step = 0; step < 3; ++step) {
      auto put_rounded = [&](Buffer& buffer, std::vector<float> values) {
        if (q == &cpu.q) for (auto& value : values) value = vt::F16ToF32(vt::F32ToF16(value));
        buffer.put(values);
      };
      put_rounded(qi, Normalized(batch * hk * dk, dk, step));
      put_rounded(ki, Normalized(batch * hk * dk, dk, step + 7));
      put_rounded(vi, Values(batch * hv * dv, step + 13, 0.01f));
      g.put(std::vector<float>(batch * hv, -0.13f));
      beta.put(std::vector<float>(batch * hv, 0.61f));
      const int32_t indices[batch] = {step % 2 == 0 ? 2 : 0, -1, step % 2 == 0 ? 0 : 2};
      ids.upload(indices);
      vt::GdnDecode(*q, out.tensor, qi.tensor, ki.tensor, vi.tensor, g.tensor,
                    beta.tensor, state.tensor, {1.0f / std::sqrt(float(dk))}, &ids.tensor);
      outputs.push_back(out.floats());
    }
    if (q == &cpu.q) { expected_outputs = outputs; expected_state = state.floats(); }
    else {
      for (int step = 0; step < 3; ++step) {
        CAPTURE(step);
        Close(outputs[step], expected_outputs[step], 0.002f, 0.001f);
        for (int col = 0; col < hv * dv; ++col)
          CHECK(outputs[step][hv * dv + col] == 0.0f);
      }
      Close(state.floats(), expected_state, 3e-5f, 3e-6f);
    }
  }
}

TEST_CASE("XPU GDN SG16 decode continues irregular prefill chunks and preserves final state") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  constexpr int tokens = 6, hk = 2, hv = 6, dk = 128, dv = 128;
  std::vector<float> expected_output, expected_state;
  for (auto* q : {&cpu.q, &gpu.q}) {
    const auto type = q == &cpu.q ? DType::kF32 : DType::kF16;
    Buffer qi(*q, type, {tokens, hk, dk}), ki(*q, type, {tokens, hk, dk});
    Buffer vi(*q, type, {tokens, hv, dv}), out(*q, type, {tokens, hv, dv});
    Buffer g(*q, DType::kF32, {tokens, hv}), beta(*q, DType::kF32, {tokens, hv});
    Buffer state(*q, DType::kF32, {2, hv, dv, dk}), qsl(*q, DType::kI32, {3});
    Buffer ids(*q, DType::kI32, {1});
    const int32_t slot[] = {0}; ids.upload(slot);
    auto put_rounded = [&](Buffer& buffer, std::vector<float> values) {
      if (q == &cpu.q) for (auto& value : values) value = vt::F16ToF32(vt::F32ToF16(value));
      buffer.put(values);
    };
    put_rounded(qi, Normalized(tokens * hk * dk, dk, 2));
    put_rounded(ki, Normalized(tokens * hk * dk, dk, 5));
    put_rounded(vi, Values(tokens * hv * dv, 7));
    g.put(std::vector<float>(tokens * hv, -0.13f));
    beta.put(std::vector<float>(tokens * hv, 0.61f));
    state.put(Values(2 * hv * dv * dk, 11, 0.002f));
    const vt::GdnArgs args{1.0f / std::sqrt(float(dk))};
    for (const auto [first, length] : {std::pair{0, 2}, std::pair{2, 1}}) {
      const int32_t offsets[] = {0, length, length};
      qsl.upload(offsets);
      auto output = Rows(out.tensor, first, length);
      vt::GdnPrefill(*q, output, Rows(qi.tensor, first, length), Rows(ki.tensor, first, length),
                     Rows(vi.tensor, first, length), Rows(g.tensor, first, length),
                     Rows(beta.tensor, first, length), state.tensor, qsl.tensor, args);
    }
    for (int token = 3; token < tokens; ++token) {
      auto output = Rows(out.tensor, token, 1);
      vt::GdnDecode(*q, output, Rows(qi.tensor, token, 1), Rows(ki.tensor, token, 1),
                    Rows(vi.tensor, token, 1), Rows(g.tensor, token, 1),
                    Rows(beta.tensor, token, 1), state.tensor, args, &ids.tensor);
    }
    if (q == &cpu.q) { expected_output = out.floats(); expected_state = state.floats(); }
    else {
      Close(out.floats(), expected_output, 0.002f, 0.001f);
      Close(state.floats(), expected_state, 3e-5f, 3e-6f);
    }
  }
}

TEST_CASE("XPU P7 mixed GDN output intervals match checked scatter and full FP32 states") {
  using vt::Tensor;
  Queue gpu(vt::DeviceType::kXPU);
  constexpr int ts = 4, tp = 64, hk = 16, hv = 48, dim = 128;
  constexpr int rows = ts + tp;
  Buffer query(gpu.q, DType::kF16, {rows, hk, dim});
  Buffer key(gpu.q, DType::kF16, {rows, hk, dim});
  Buffer value(gpu.q, DType::kF16, {rows, hv, dim});
  Buffer gate(gpu.q, DType::kF32, {rows, hv}), beta(gpu.q, DType::kF32, {rows, hv});
  Buffer spec_offsets(gpu.q, DType::kI32, {2}), prefill_offsets(gpu.q, DType::kI32, {2});
  Buffer slots(gpu.q, DType::kI32, {1, 4}), accepted(gpu.q, DType::kI32, {1});
  Buffer spec_map(gpu.q, DType::kI32, {ts}), prefill_map(gpu.q, DType::kI32, {tp});
  Buffer spec_out(gpu.q, DType::kF16, {ts, hv, dim}), prefill_out(gpu.q, DType::kF16, {tp, hv, dim});
  Buffer scattered(gpu.q, DType::kF16, {rows + 2, hv, dim});
  Buffer direct(gpu.q, DType::kF16, {rows + 2, hv, dim});
  Buffer old_spec_state(gpu.q, DType::kF32, {4, hv, dim, dim});
  Buffer new_spec_state(gpu.q, DType::kF32, {4, hv, dim, dim});
  Buffer old_prefill_state(gpu.q, DType::kF32, {1, hv, dim, dim});
  Buffer new_prefill_state(gpu.q, DType::kF32, {1, hv, dim, dim});
  const int32_t so[] = {0, ts}, po[] = {0, tp}, ids[] = {0, 1, 2, 3};
  spec_offsets.upload(so); prefill_offsets.upload(po); slots.upload(ids);
  vt::GdnArgs prefill_args{1.0f / std::sqrt(float(dim))};
  prefill_args.query_start_loc_host = po;
  const auto initial_spec = Values(4 * hv * dim * dim, 13, 0.003f);
  const auto initial_prefill = Values(hv * dim * dim, 17, 0.003f);
  for (bool spec_first : {true, false}) for (int count : {1, 2, 3, 4}) {
    CAPTURE(spec_first);
    CAPTURE(count);
    query.put(Normalized(rows * hk * dim, dim, count));
    key.put(Normalized(rows * hk * dim, dim, count + 7));
    value.put(Values(rows * hv * dim, count + 23));
    gate.put(std::vector<float>(rows * hv, -0.03f));
    beta.put(std::vector<float>(rows * hv, 0.4f)); accepted.upload(&count);
    old_spec_state.put(initial_spec); new_spec_state.put(initial_spec);
    old_prefill_state.put(initial_prefill); new_prefill_state.put(initial_prefill);
    const std::vector<float> poison((rows + 2) * hv * dim, std::numeric_limits<float>::quiet_NaN());
    scattered.put(poison); direct.put(poison);
    Tensor merged_old = Rows(scattered.tensor, 1, rows), merged_new = Rows(direct.tensor, 1, rows);
    const int sf = spec_first ? 0 : tp, pf = spec_first ? ts : 0;
    std::vector<int32_t> si(ts), pi(tp);
    for (int i = 0; i < ts; ++i) si[i] = sf + i;
    for (int i = 0; i < tp; ++i) pi[i] = pf + i;
    spec_map.upload(si.data()); prefill_map.upload(pi.data());
    Tensor spec_view = Rows(merged_new, sf, ts), prefill_view = Rows(merged_new, pf, tp);
    const auto spec = [&](Tensor& output, Tensor& state) {
      const auto q = Rows(query.tensor, 0, ts), k = Rows(key.tensor, 0, ts);
      const auto v = Rows(value.tensor, 0, ts), g = Rows(gate.tensor, 0, ts), b = Rows(beta.tensor, 0, ts);
      vt::GdnSpecDecode(gpu.q, output, q, k, v, g, b, state, spec_offsets.tensor,
          slots.tensor, accepted.tensor, {1.0f / std::sqrt(float(dim))});
    };
    const auto prefill = [&](Tensor& output, Tensor& state) {
      const auto q = Rows(query.tensor, ts, tp), k = Rows(key.tensor, ts, tp);
      const auto v = Rows(value.tensor, ts, tp), g = Rows(gate.tensor, ts, tp), b = Rows(beta.tensor, ts, tp);
      vt::GdnPrefill(gpu.q, output, q, k, v, g, b, state, prefill_offsets.tensor, prefill_args);
    };
    spec(spec_out.tensor, old_spec_state.tensor);
    vt::IndexCopy(gpu.q, merged_old, spec_out.tensor, spec_map.tensor);
    prefill(prefill_out.tensor, old_prefill_state.tensor);
    vt::IndexCopy(gpu.q, merged_old, prefill_out.tensor, prefill_map.tensor);
    spec(spec_view, new_spec_state.tensor); prefill(prefill_view, new_prefill_state.tensor);
    SameBytes(direct.download(), scattered.download());  // All rows and poisoned outside guards.
    SameBytes(new_spec_state.download(), old_spec_state.download());  // Every provisional snapshot.
    SameBytes(new_prefill_state.download(), old_prefill_state.download());
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU compressed conv: direct BF16 cache equals the F32 working-copy path") {
  Queue gpu(vt::DeviceType::kXPU);
  for (bool prefill : {false, true}) {
    constexpr auto cache_type = DType::kBF16;
    constexpr int channels = 17, slots = 4, tokens = 4;
    Buffer x(gpu.q, DType::kF32, {tokens, channels}), weight(gpu.q, DType::kF32, {channels, 4});
    Buffer state(gpu.q, cache_type, {slots, channels, 6});
    Buffer working(gpu.q, DType::kF32, {slots, channels, 6}), stored(gpu.q, cache_type, {slots, channels, 6});
    Buffer out(gpu.q, DType::kF32, {tokens, channels}), reference(gpu.q, DType::kF32, {tokens, channels});
    Buffer indices(gpu.q, DType::kI32, {tokens}), qsl(gpu.q, DType::kI32, {slots + 1}), flags(gpu.q, DType::kI8, {slots});
    state.put(Values(slots * channels * 6, 2)); weight.put(Values(channels * 4, 3));
    out.put(std::vector<float>(tokens * channels, -1)); reference.put(std::vector<float>(tokens * channels, -1));
    const int32_t offsets[] = {0, 0, 1, 3, 4}; const int8_t initial[] = {1, 0, 1, 1};
    qsl.upload(offsets); flags.upload(initial);
    for (int step = 0; step < 8; ++step) {
      x.put(Values(tokens * channels, step, .071f));
      int32_t ids[tokens]; for (int i = 0; i < tokens; ++i) ids[i] = i == step % slots ? -1 : (i + step) % slots;
      indices.upload(ids); vt::Copy(gpu.q, working.tensor, state.tensor);
      if (prefill) {
        vt::CausalConv1dFwd(gpu.q, out.tensor, x.tensor, weight.tensor, nullptr, state.tensor, qsl.tensor, flags.tensor, {});
        vt::CausalConv1dFwd(gpu.q, reference.tensor, x.tensor, weight.tensor, nullptr, working.tensor, qsl.tensor, flags.tensor, {});
      } else {
        vt::CausalConv1dUpdate(gpu.q, out.tensor, x.tensor, weight.tensor, nullptr, state.tensor, {}, &indices.tensor);
        vt::CausalConv1dUpdate(gpu.q, reference.tensor, x.tensor, weight.tensor, nullptr, working.tensor, {}, &indices.tensor);
      }
      vt::Copy(gpu.q, stored.tensor, working.tensor);
      SameBytes(out.download(), reference.download()); SameBytes(state.download(), stored.download());
    }
  }
}

namespace {
struct P7GatedTableEnv {
  const bool had = std::getenv("VT_XPU_GDN_GATED_SILU_TABLE") != nullptr;
  const std::string old = had ? std::getenv("VT_XPU_GDN_GATED_SILU_TABLE") : "";
  ~P7GatedTableEnv() {
    if (had) setenv("VT_XPU_GDN_GATED_SILU_TABLE", old.c_str(), 1);
    else unsetenv("VT_XPU_GDN_GATED_SILU_TABLE");
  }
  void Select(bool on) { REQUIRE(setenv("VT_XPU_GDN_GATED_SILU_TABLE", on ? "1" : "0", 1) == 0); }
};
void P7GatedSame(const Buffer& actual, const std::vector<unsigned char>& expected) {
  auto bytes = actual.download(); REQUIRE(bytes.size() == expected.size());
  size_t mismatch = 0, dual_nan = 0;
  const size_t width = vt::SizeOf(actual.tensor.dtype);
  for (size_t i = 0; i < bytes.size(); i += width) {
    if (std::memcmp(bytes.data() + i, expected.data() + i, width) == 0) continue;
    bool a_nan, b_nan;
    if (width == 2) {
      uint16_t a, b; std::memcpy(&a, bytes.data() + i, 2); std::memcpy(&b, expected.data() + i, 2);
      a_nan = (a & 0x7c00) == 0x7c00 && (a & 1023); b_nan = (b & 0x7c00) == 0x7c00 && (b & 1023);
    } else {
      float a, b; std::memcpy(&a, bytes.data() + i, 4); std::memcpy(&b, expected.data() + i, 4);
      a_nan = std::isnan(a); b_nan = std::isnan(b);
    }
    if (a_nan && b_nan) ++dual_nan; else ++mismatch;
  }
  CAPTURE(mismatch);
  CAPTURE(dual_nan); CHECK(mismatch == 0);
}
}

TEST_CASE("XPU P7 gated SiLU table: queue readiness cold capture and graph lifetime") {
  Queue gpu(vt::DeviceType::kXPU); P7GatedTableEnv mode;
  Buffer x(gpu.q, DType::kF16, {1, 48, 128}), gate(gpu.q, DType::kF16, {1, 49, 128});
  Buffer weight(gpu.q, DType::kF32, {128}), direct(gpu.q, DType::kF16, {1, 48, 128});
  Buffer table(gpu.q, DType::kF16, {1, 48, 128}), other_out(gpu.q, DType::kF16, {1, 48, 128});
  x.put(Values(48 * 128, 2)); gate.put(Values(49 * 128, 7)); gate.tensor.shape[1] = 48;
  weight.put(Values(128, 3, 0.1f)); table.put(std::vector<float>(48 * 128, -7));
  const auto unchanged = table.download(); auto& backend = vt::GetBackend(gpu.q.device);
  mode.Select(true);
  if (backend.SupportsGraphCapture()) {
    vt::BreakableGraph cold;
    CHECK_THROWS_WITH_AS(([&] {
      vt::GraphCaptureScope scope(backend, gpu.q, cold, vt::GraphCaptureMode::kFull);
      vt::RmsNormGated(gpu.q, table.tensor, x.tensor, gate.tensor, weight.tensor, {1e-6f, false});
    }()), doctest::Contains("XPU gated SiLU table must be warmed on the capture queue"), std::runtime_error);
    SameBytes(table.download(), unchanged);
  }
  mode.Select(false); vt::RmsNormGated(gpu.q, direct.tensor, x.tensor, gate.tensor, weight.tensor, {1e-6f, false});
  const auto expected = direct.download(); mode.Select(true);
  auto other = vt::CreateQueue(gpu.q.device);
  vt::RmsNormGated(gpu.q, table.tensor, x.tensor, gate.tensor, weight.tensor, {1e-6f, false});
  // No producer synchronization before the other queue joins initialization.
  vt::RmsNormGated(other, other_out.tensor, x.tensor, gate.tensor, weight.tensor, {1e-6f, false});
  backend.Synchronize(other);  // Download uses gpu.q; complete its other-queue producer first.
  SameBytes(other_out.download(), expected); SameBytes(table.download(), expected);
  CHECK(vt::xpu::GetMemoryInfo().gated_silu_table_bytes == 262144);
  if (backend.SupportsGraphCapture()) {
    vt::BreakableGraph graph;
    { vt::GraphCaptureScope scope(backend, other, graph, vt::GraphCaptureMode::kFull);
      vt::RmsNormGated(other, other_out.tensor, x.tensor, gate.tensor, weight.tensor, {1e-6f, false}); }
    REQUIRE(graph.captured());
    for (int salt : {9, 17}) {
      x.put(Values(48 * 128, salt)); mode.Select(false);
      vt::RmsNormGated(gpu.q, direct.tensor, x.tensor, gate.tensor, weight.tensor, {1e-6f, false});
      const auto wanted = direct.download(); graph.Replay(other); backend.Synchronize(other);
      SameBytes(other_out.download(), wanted);
    }
    graph.Reset(); CHECK(vt::xpu::GetMemoryInfo().graph_device_bytes == 0);
  }
  vt::DestroyQueue(other); other = vt::CreateQueue(gpu.q.device); mode.Select(true);
  if (backend.SupportsGraphCapture()) {
    vt::BreakableGraph cold;
    CHECK_THROWS_WITH_AS(([&] {
      vt::GraphCaptureScope scope(backend, other, cold, vt::GraphCaptureMode::kFull);
      vt::RmsNormGated(other, other_out.tensor, x.tensor, gate.tensor, weight.tensor, {1e-6f, false});
    }()), doctest::Contains("XPU gated SiLU table must be warmed on the capture queue"), std::runtime_error);
  }
  vt::RmsNormGated(other, other_out.tensor, x.tensor, gate.tensor, weight.tensor, {1e-6f, false});
  backend.Synchronize(other); SameBytes(other_out.download(), direct.download()); vt::DestroyQueue(other);
}

TEST_CASE("XPU P7 gated SiLU table: all gate bits FP32 boundary tails aliases and guards") {
  Queue gpu(vt::DeviceType::kXPU); P7GatedTableEnv mode;
  constexpr int rows = 512, width = 128, n = rows * width;
  Buffer x(gpu.q, DType::kF16, {rows, width}), gate(gpu.q, DType::kF16, {rows, width});
  std::vector<uint16_t> bits(n); for (int i = 0; i < n; ++i) bits[i] = uint16_t(i);
  gate.upload(bits.data()); const auto preserved_gate = gate.download();
  {
    // x=1, eps=0 and w=1 make F32 output exactly the SiLU intermediate.
    // Verify all actual F16 input bitpatterns before any output narrowing.
    x.put(std::vector<float>(n, 1.0f));
    Buffer weight(gpu.q, DType::kF32, {width}), direct(gpu.q, DType::kF32, {rows, width});
    Buffer table(gpu.q, DType::kF32, {rows, width}); weight.put(std::vector<float>(width, 1.0f));
    mode.Select(false); vt::RmsNormGated(gpu.q, direct.tensor, x.tensor, gate.tensor, weight.tensor, {0.0f, false});
    const auto expected = direct.download(); mode.Select(true);
    vt::RmsNormGated(gpu.q, table.tensor, x.tensor, gate.tensor, weight.tensor, {0.0f, false});
    P7GatedSame(table, expected);
  }
  for (auto weight_type : {DType::kF32, DType::kF16}) for (auto out_type : {DType::kF32, DType::kF16}) {
    CAPTURE(weight_type);
    CAPTURE(out_type);
    Buffer weight(gpu.q, weight_type, {width}), direct(gpu.q, out_type, {rows, width}), table(gpu.q, out_type, {rows, width});
    weight.put(Values(width, 19, 0.03125f));
    for (int salt : {0, 11, 29}) {
      CAPTURE(salt); x.put(Values(n, salt, 0.03125f)); const auto preserved_x = x.download(), preserved_w = weight.download();
      mode.Select(false); vt::RmsNormGated(gpu.q, direct.tensor, x.tensor, gate.tensor, weight.tensor, {1e-6f, false});
      const auto expected = direct.download(); mode.Select(true);
      vt::RmsNormGated(gpu.q, table.tensor, x.tensor, gate.tensor, weight.tensor, {1e-6f, false});
      P7GatedSame(table, expected); SameBytes(x.download(), preserved_x); SameBytes(weight.download(), preserved_w);
    }
  }
  SameBytes(gate.download(), preserved_gate);
  for (int count : {1, 31, 32, 33, 48, 192}) for (bool sigmoid : {false, true}) {
    CAPTURE(count);
    CAPTURE(sigmoid);
    Buffer input(gpu.q, DType::kF16, {count, width}), g(gpu.q, DType::kF16, {count, width});
    Buffer w(gpu.q, DType::kF32, {width}), direct(gpu.q, DType::kF16, {count, width});
    input.put(Values(count * width, 3)); g.put(Values(count * width, 8)); w.put(Values(width, 9));
    mode.Select(false); vt::RmsNormGated(gpu.q, direct.tensor, input.tensor, g.tensor, w.tensor, {1e-6f, sigmoid});
    const auto expected = direct.download(); mode.Select(true);
    vt::RmsNormGated(gpu.q, input.tensor, input.tensor, g.tensor, w.tensor, {1e-6f, sigmoid});
    SameBytes(input.download(), expected);
  }
  Buffer w(gpu.q, DType::kF32, {width}), out(gpu.q, DType::kF16, {rows, width}); w.put(Values(width));
  out.put(std::vector<float>(n, -7)); const auto before = out.download();
  mode.Select(true); setenv("VT_XPU_GDN_GATED_SILU_TABLE", "invalid", 1);
  CHECK_THROWS_WITH_AS(vt::RmsNormGated(gpu.q, out.tensor, x.tensor, gate.tensor, w.tensor, {1e-6f, false}),
      doctest::Contains("Invalid VT_XPU_GDN_GATED_SILU_TABLE"), std::runtime_error);
  SameBytes(out.download(), before);
  std::cout << "P7_GATED_SILU_TABLE gate_patterns=65536 activation_F32_exhaustive=1 weights=F16,F32 outputs=F16,F32 table_bytes=262144" << std::endl;
}

TEST_CASE("XPU P7 gated SiLU table: default model consumer probe") {
  if (!std::getenv("VT_B70_GATED_SILU_DEFAULT_PROBE")) {
    MESSAGE("set VT_B70_GATED_SILU_DEFAULT_PROBE in a fresh worker"); return;
  }
  Queue gpu(vt::DeviceType::kXPU); P7GatedTableEnv mode;
  REQUIRE(vt::xpu::GetMemoryInfo().gated_silu_table_bytes == 0);
  Buffer x(gpu.q, DType::kF16, {1, 48, 128}), gate(gpu.q, DType::kF16, {1, 48, 128});
  Buffer w(gpu.q, DType::kF32, {128}), direct(gpu.q, DType::kF16, {1, 48, 128}), out(gpu.q, DType::kF16, {1, 48, 128});
  x.put(Values(48 * 128, 1)); gate.put(Values(48 * 128, 5)); w.put(Values(128, 6));
  mode.Select(false); vt::RmsNormGated(gpu.q, direct.tensor, x.tensor, gate.tensor, w.tensor, {1e-6f, false});
  const auto expected = direct.download(); REQUIRE(vt::xpu::GetMemoryInfo().gated_silu_table_bytes == 0);
  REQUIRE(unsetenv("VT_XPU_GDN_GATED_SILU_TABLE") == 0);
  vt::RmsNormGated(gpu.q, out.tensor, x.tensor, gate.tensor, w.tensor, {1e-6f, false});
  SameBytes(out.download(), expected); CHECK(vt::xpu::GetMemoryInfo().gated_silu_table_bytes == 262144);
}

TEST_CASE("XPU P7 gated SiLU table: complete operator balanced timing") {
  const char* output = std::getenv("VT_B70_GATED_SILU_OUTPUT");
  if (!output) { MESSAGE("set VT_B70_GATED_SILU_OUTPUT for focused operator timing"); return; }
  Queue gpu(vt::DeviceType::kXPU); P7GatedTableEnv mode;
  nlohmann::json report = {{"schema", "b70-exl3-p7-gated-silu-operator-v1"}, {"cases", nlohmann::json::array()}};
  for (int tokens : {4, 896, 1600}) {
    const int count = tokens * 48;
    Buffer x(gpu.q, DType::kF16, {tokens, 48, 128}), gate(gpu.q, DType::kF16, {tokens, 49, 128});
    Buffer w(gpu.q, DType::kF32, {128}), direct(gpu.q, DType::kF16, {tokens, 48, 128});
    Buffer table(gpu.q, DType::kF16, {tokens, 48, 128});
    x.put(Values(count * 128, 3)); gate.put(Values(tokens * 49 * 128, 19, 0.125f)); gate.tensor.shape[1] = 48;
    w.put(Values(128, 6, 0.1f)); const auto input = x.download(), gates = gate.download(), weights = w.download();
    auto call = [&](bool on) { mode.Select(on); vt::RmsNormGated(gpu.q, on ? table.tensor : direct.tensor,
        x.tensor, gate.tensor, w.tensor, {1e-6f, false}); vt::GetBackend(gpu.q.device).Synchronize(gpu.q); };
    for (int i = 0; i < 32; ++i) call(i % 2);
    const auto expected = direct.download(); SameBytes(table.download(), expected);
    nlohmann::json trials = nlohmann::json::array();
    for (bool on : {false, true, true, false, true, false, false, true}) {
      mode.Select(on); auto start = std::chrono::steady_clock::now();
      vt::RmsNormGated(gpu.q, on ? table.tensor : direct.tensor, x.tensor, gate.tensor, w.tensor, {1e-6f, false});
      vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
      trials.push_back({{"table", on}, {"complete_operator_ms", ms}});
      SameBytes((on ? table : direct).download(), expected);
    }
    SameBytes(x.download(), input); SameBytes(gate.download(), gates); SameBytes(w.download(), weights);
    report["cases"].push_back({{"tokens", tokens}, {"heads", 48}, {"width", 128}, {"gate_token_stride", 49 * 128},
        {"warmups", 32}, {"trials", trials}, {"all_output_input_bits_exact", true}});
  }
  report["table_bytes"] = vt::xpu::GetMemoryInfo().gated_silu_table_bytes;
  std::ofstream f(output); REQUIRE(f.good()); f << report.dump(2) << std::endl;
}

namespace {
struct P7SpecSlmEnv {
  const bool had = std::getenv("VT_XPU_GDN_SPEC_SLM") != nullptr;
  const std::string old = had ? std::getenv("VT_XPU_GDN_SPEC_SLM") : "";
  ~P7SpecSlmEnv() { if (had) setenv("VT_XPU_GDN_SPEC_SLM", old.c_str(), 1); else unsetenv("VT_XPU_GDN_SPEC_SLM"); }
  void Select(bool on) { REQUIRE(setenv("VT_XPU_GDN_SPEC_SLM", on ? "1" : "0", 1) == 0); }
};
struct P7SpecTypedEnv {
  const bool had = std::getenv("VT_XPU_GDN_SPEC_SLM_TYPED") != nullptr;
  const std::string old = had ? std::getenv("VT_XPU_GDN_SPEC_SLM_TYPED") : "";
  ~P7SpecTypedEnv() {
    if (had) setenv("VT_XPU_GDN_SPEC_SLM_TYPED", old.c_str(), 1);
    else unsetenv("VT_XPU_GDN_SPEC_SLM_TYPED");
  }
  void Select(bool on) { REQUIRE(setenv("VT_XPU_GDN_SPEC_SLM_TYPED", on ? "1" : "0", 1) == 0); }
};
struct P7SpecOperands {
  static constexpr int hk = 16, hv = 48, D = 128, cols = 4;
  vt::Queue& q;
  int requests, tokens, slots;
  Buffer query, key, value, gate, beta, output, state, cu, ids, accepted;
  std::vector<int32_t> offsets, indices, counts;
  std::vector<unsigned char> initial;
  P7SpecOperands(vt::Queue& queue, const std::vector<int>& lengths, DType output_type)
      : q(queue), requests(lengths.size()), tokens([&] { int n=0; for (int x:lengths) n+=x; return n; }()),
        slots(requests * cols + 2), query(q,DType::kF16,{tokens,hk,D}), key(q,DType::kF16,{tokens,hk,D}),
        value(q,DType::kF16,{tokens,hv,D}), gate(q,DType::kF32,{tokens,hv}), beta(q,DType::kF32,{tokens,hv}),
        output(q,output_type,{tokens,hv,D}), state(q,DType::kF32,{slots,hv,D,D}),
        cu(q,DType::kI32,{requests+1}), ids(q,DType::kI32,{requests,cols}), accepted(q,DType::kI32,{requests}),
        offsets(requests+1), indices(requests*cols), counts(requests,1) {
    query.put(Values(tokens*hk*D,31)); key.put(Values(tokens*hk*D,32)); value.put(Values(tokens*hv*D,33));
    auto gs=Values(tokens*hv,34), bs=Values(tokens*hv,35);
    for (auto& x:gs) x-=.3f;
    for (auto& x:bs) x+=.5f;
    gate.put(gs); beta.put(bs);
    auto ss=Values(size_t(slots)*hv*D*D,36,.002f);
    const uint32_t poison=0x7fc12345u; const size_t stride=size_t(hv)*D*D;
    for (size_t i=0;i<stride;++i) {
      std::memcpy(ss.data()+i,&poison,4); std::memcpy(ss.data()+size_t(slots-1)*stride+i,&poison,4);
    }
    state.put(ss); initial=state.download();
    constexpr int perm[4]={2,0,3,1};
    for (int r=0;r<requests;++r) {
      offsets[r+1]=offsets[r]+lengths[r];
      for (int c=0;c<cols;++c) indices[r*cols+c]=1+(requests-1-r)*cols+perm[c];
    }
    Stage();
  }
  void Stage() { cu.upload(offsets.data()); ids.upload(indices.data()); accepted.upload(counts.data()); }
  void Reset() { state.upload(initial.data()); }
  void Run(vt::Tensor* target=nullptr) { vt::GdnSpecDecode(q,target ? *target : output.tensor,query.tensor,key.tensor,
      value.tensor,gate.tensor,beta.tensor,state.tensor,cu.tensor,ids.tensor,accepted.tensor,{.0883883476f}); }
};
}

TEST_CASE("XPU P7 speculative GDN SLM: exact full snapshots selectors null alias and graphs") {
  Queue gpu(vt::DeviceType::kXPU); P7SpecSlmEnv mode; auto& backend=vt::GetBackend(gpu.q.device);
  for (const auto& lengths:std::vector<std::vector<int>>{{1},{4},{4,4,4,4},{4,2,1,0}}) {
    for (auto type:{DType::kF32,DType::kF16}) {
      P7SpecOperands c(gpu.q,lengths,type);
      const auto qb=c.query.download(), kb=c.key.download(), vb=c.value.download(), gb=c.gate.download(), bb=c.beta.download();
      for (int selector=1;selector<=4;++selector) {
        CAPTURE(selector);
        for (int r=0;r<c.requests;++r) c.counts[r]=1+(selector+r-1)%4;
        c.Stage(); c.Reset(); mode.Select(false); c.Run();
        const auto expected=c.output.download(), snapshots=c.state.download();
        c.Reset(); mode.Select(true); c.Run();
        SameBytes(c.output.download(),expected); SameBytes(c.state.download(),snapshots);
      }
      // Repeated slots within one request are allowed, including the starting
      // slot also being a provisional output. Preserve that alias exactly.
      for (int r=0;r<c.requests;++r) for (int j=1;j<4;++j) c.indices[r*4+j]=c.indices[r*4];
      c.Stage(); c.Reset(); mode.Select(false); c.Run();
      auto expected=c.output.download(), snapshots=c.state.download();
      c.Reset(); mode.Select(true); c.Run(); SameBytes(c.output.download(),expected); SameBytes(c.state.download(),snapshots);
      c.indices[0]=-1; c.counts[0]=1; c.Stage(); c.Reset(); mode.Select(false); c.Run();
      expected=c.output.download(); snapshots=c.state.download();
      c.Reset(); mode.Select(true); c.Run(); SameBytes(c.output.download(),expected); SameBytes(c.state.download(),snapshots);
      SameBytes(c.query.download(),qb); SameBytes(c.key.download(),kb); SameBytes(c.value.download(),vb);
      SameBytes(c.gate.download(),gb); SameBytes(c.beta.download(),bb);
      if (c.requests==4 && type==DType::kF32 && backend.SupportsGraphCapture()) {
        vt::BreakableGraph graph;
        { vt::GraphCaptureScope scope(backend,gpu.q,graph,vt::GraphCaptureMode::kFull); c.Run(); }
        REQUIRE(graph.captured());
        c.Reset(); graph.Replay(gpu.q); SameBytes(c.output.download(),expected); SameBytes(c.state.download(),snapshots);
        // A changed accepted selector is actual replay metadata, never a cached
        // capture-time result. Rebuild native expected full state independently.
        c.counts[0]=2; c.Stage(); c.Reset(); mode.Select(false); c.Run();
        expected=c.output.download(); snapshots=c.state.download();
        c.Reset(); graph.Replay(gpu.q); SameBytes(c.output.download(),expected); SameBytes(c.state.download(),snapshots);
        const auto before=c.state.download(), out=c.output.download();
        c.indices[4]=c.indices[1]; c.Stage();
        CHECK_THROWS_WITH_AS(graph.Replay(gpu.q),doctest::Contains("invalid offsets, accepted count or state slot"),std::runtime_error);
        SameBytes(c.state.download(),before); SameBytes(c.output.download(),out);
        graph.Reset(); CHECK(vt::xpu::GetMemoryInfo().graph_device_bytes==0);
      }
    }
  }
  P7SpecOperands c(gpu.q,{4},DType::kF16); const auto values=c.value.download();
  c.Reset(); mode.Select(false); c.Run(&c.value.tensor); const auto alias=c.value.download(), snapshots=c.state.download();
  c.value.upload(values.data()); c.Reset(); mode.Select(true); c.Run(&c.value.tensor);
  SameBytes(c.value.download(),alias); SameBytes(c.state.download(),snapshots);
  c.output.put(std::vector<float>(c.tokens*48*128,-7.0f));
  const auto before=c.state.download(), out=c.output.download();
  setenv("VT_XPU_GDN_SPEC_SLM","invalid",1);
  CHECK_THROWS_WITH_AS(c.Run(),doctest::Contains("Invalid VT_XPU_GDN_SPEC_SLM"),std::runtime_error);
  SameBytes(c.state.download(),before); SameBytes(c.output.download(),out);
  CHECK(vt::GetReferenceTierHits()==0);
  std::cout << "P7_SPEC_SLM full_F32_snapshots=1 selectors=1,2,3,4 C1_Q1_Q4=1 C4_uniform_ragged=1 alias_null_guards_graph=1" << std::endl;
}

TEST_CASE("XPU P7 speculative GDN SLM: default selection and WG override probe") {
  if (!std::getenv("VT_B70_GDN_SLM_DEFAULT_PROBE")) { MESSAGE("run default probe with device profiling"); return; }
  REQUIRE(std::getenv("VT_XPU_GDN_SPEC_WG")==nullptr);
  Queue gpu(vt::DeviceType::kXPU); P7SpecSlmEnv mode; P7SpecOperands c(gpu.q,{4},DType::kF32);
  mode.Select(false); c.Run(); const auto expected=c.output.download(), snapshots=c.state.download();
  (void)vt::xpu::DrainProfileEvents();
  REQUIRE(unsetenv("VT_XPU_GDN_SPEC_SLM")==0); c.Reset(); c.Run();
  SameBytes(c.output.download(),expected); SameBytes(c.state.download(),snapshots);
  int selected=0; for (const auto& e:vt::xpu::DrainProfileEvents()) selected+=e.stage=="gdn_spec_decode_slm";
  CHECK(selected==1);
  REQUIRE(setenv("VT_XPU_GDN_SPEC_WG","64",1)==0); c.Reset(); c.Run();
  SameBytes(c.output.download(),expected); SameBytes(c.state.download(),snapshots);
  selected=0; for (const auto& e:vt::xpu::DrainProfileEvents()) selected+=e.stage=="gdn_spec_decode_slm";
  CHECK(selected==0); REQUIRE(unsetenv("VT_XPU_GDN_SPEC_WG")==0);
}

TEST_CASE("XPU P7 speculative GDN SLM: paired complete operator timing") {
  const char* path=std::getenv("VT_B70_GDN_SLM_OUTPUT");
  if (!path) { MESSAGE("set VT_B70_GDN_SLM_OUTPUT for focused paired timing"); return; }
  Queue gpu(vt::DeviceType::kXPU); P7SpecSlmEnv mode; auto& backend=vt::GetBackend(gpu.q.device);
  nlohmann::json report={{"schema","b70-exl3-p7-gdn-slm-operator-v1"},{"cases",nlohmann::json::array()}};
  for (const auto& lengths:std::vector<std::vector<int>>{{4},{4,4,4,4},{4,2,1,3}}) {
    P7SpecOperands c(gpu.q,lengths,DType::kF32);
    for (int r=0;r<c.requests;++r) c.counts[r]=r+1;
    c.Stage(); mode.Select(false); c.Reset(); c.Run();
    const auto expected=c.output.download(), snapshots=c.state.download();
    for (int i=0;i<16;++i) { mode.Select(i%2); c.Reset(); c.Run(); backend.Synchronize(gpu.q); }
    nlohmann::json trials=nlohmann::json::array();
    for (bool on:{false,true,true,false,true,false,false,true,false,true,true,false,true,false,false,true}) {
      mode.Select(on); c.Reset(); const auto start=std::chrono::steady_clock::now();
      c.Run(); backend.Synchronize(gpu.q);
      const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
      trials.push_back({{"SLM",on},{"complete_operator_ms",ms}});
      SameBytes(c.output.download(),expected); SameBytes(c.state.download(),snapshots);
    }
    report["cases"].push_back({{"requests",c.requests},{"lengths",lengths},{"dtype","F32 output/state, F16 qkv, F32 gates"},
        {"accepted_selectors",c.counts},{"state_bytes",c.state.bytes},{"trials",trials},{"all_output_snapshots_exact",true}});
  }
  report["SLM_per_WG_bytes"]=32768;
  report["new_device_workspace_bytes"]=0;
  std::ofstream f(path); REQUIRE(f.good()); f << report.dump(2) << std::endl;
}

TEST_CASE("XPU P7 speculative GDN SLM typed: exact snapshots F16 graphs and fallback") {
  Queue gpu(vt::DeviceType::kXPU); P7SpecSlmEnv mode; P7SpecTypedEnv typed;
  auto& backend = vt::GetBackend(gpu.q.device); mode.Select(true);
  for (const auto& lengths : std::vector<std::vector<int>>{{1}, {4}, {4,4,4,4}, {4,2,1,0}}) {
    for (auto type : {DType::kF16, DType::kF32}) {
      P7SpecOperands c(gpu.q, lengths, type);
      const auto qb=c.query.download(), kb=c.key.download(), vb=c.value.download(),
                 gb=c.gate.download(), bb=c.beta.download();
      for (int selector=1; selector<=4; ++selector) {
        for (int r=0; r<c.requests; ++r) c.counts[r]=1+(selector+r-1)%4;
        c.Stage(); c.Reset(); typed.Select(false); c.Run();
        const auto expected=c.output.download(), snapshots=c.state.download();
        c.Reset(); typed.Select(true); c.Run();
        SameBytes(c.output.download(), expected); SameBytes(c.state.download(), snapshots);
      }
      for (int r=0; r<c.requests; ++r)
        for (int j=1; j<4; ++j) c.indices[r*4+j]=c.indices[r*4];
      c.Stage(); c.Reset(); typed.Select(false); c.Run();
      auto expected=c.output.download(), snapshots=c.state.download();
      c.Reset(); typed.Select(true); c.Run();
      SameBytes(c.output.download(), expected); SameBytes(c.state.download(), snapshots);
      c.indices[0]=-1; c.counts[0]=1; c.Stage(); c.Reset(); typed.Select(false); c.Run();
      expected=c.output.download(); snapshots=c.state.download();
      c.Reset(); typed.Select(true); c.Run();
      SameBytes(c.output.download(), expected); SameBytes(c.state.download(), snapshots);
      SameBytes(c.query.download(), qb); SameBytes(c.key.download(), kb);
      SameBytes(c.value.download(), vb); SameBytes(c.gate.download(), gb); SameBytes(c.beta.download(), bb);
      if (c.requests==4 && type==DType::kF16 && backend.SupportsGraphCapture()) {
        vt::BreakableGraph graph;
        { vt::GraphCaptureScope scope(backend, gpu.q, graph, vt::GraphCaptureMode::kFull); c.Run(); }
        REQUIRE(graph.captured());
        c.Reset(); graph.Replay(gpu.q);
        SameBytes(c.output.download(), expected); SameBytes(c.state.download(), snapshots);
        c.counts[0]=2; c.Stage(); c.Reset(); typed.Select(false); c.Run();
        expected=c.output.download(); snapshots=c.state.download();
        c.Reset(); graph.Replay(gpu.q);
        SameBytes(c.output.download(), expected); SameBytes(c.state.download(), snapshots);
        const auto before=c.state.download(), out=c.output.download();
        c.indices[4]=c.indices[1]; c.Stage();
        CHECK_THROWS_WITH_AS(graph.Replay(gpu.q), doctest::Contains("invalid offsets, accepted count or state slot"), std::runtime_error);
        SameBytes(c.state.download(), before); SameBytes(c.output.download(), out);
        graph.Reset(); CHECK(vt::xpu::GetMemoryInfo().graph_device_bytes==0);
      }
    }
  }
  P7SpecOperands c(gpu.q, {4}, DType::kF16); const auto value=c.value.download();
  c.Reset(); typed.Select(false); c.Run(&c.value.tensor);
  const auto expected=c.value.download(), snapshots=c.state.download();
  c.value.upload(value.data()); c.Reset(); typed.Select(true); c.Run(&c.value.tensor);
  SameBytes(c.value.download(), expected); SameBytes(c.state.download(), snapshots);
  const auto before=c.state.download(), out=c.output.download();
  REQUIRE(setenv("VT_XPU_GDN_SPEC_SLM_TYPED", "invalid", 1)==0);
  CHECK_THROWS_WITH_AS(c.Run(), doctest::Contains("Invalid VT_XPU_GDN_SPEC_SLM_TYPED"), std::runtime_error);
  SameBytes(c.state.download(), before); SameBytes(c.output.download(), out);
  CHECK(vt::GetReferenceTierHits()==0);
}

TEST_CASE("XPU P7 speculative GDN SLM typed: paired complete operator timing") {
  const char* path=std::getenv("VT_B70_GDN_SLM_OUTPUT");
  if (!path) { MESSAGE("set VT_B70_GDN_SLM_OUTPUT for focused paired timing"); return; }
  REQUIRE_FALSE(std::filesystem::exists(path));
  Queue gpu(vt::DeviceType::kXPU); P7SpecSlmEnv mode; P7SpecTypedEnv typed;
  auto& backend=vt::GetBackend(gpu.q.device); mode.Select(true);
  nlohmann::json report={{"schema","b70-exl3-p7-gdn-slm-typed-operator-v1"},
      {"cases",nlohmann::json::array()}};
  for (const auto& lengths : std::vector<std::vector<int>>{{4}, {4,4,4,4}, {4,2,1,3}}) {
    P7SpecOperands c(gpu.q, lengths, DType::kF16);
    for (int r=0; r<c.requests; ++r) c.counts[r]=r+1;
    c.Stage(); typed.Select(false); c.Reset(); c.Run();
    const auto expected=c.output.download(), snapshots=c.state.download();
    for (int i=0; i<16; ++i) { typed.Select(i%2); c.Reset(); c.Run(); backend.Synchronize(gpu.q); }
    nlohmann::json trials=nlohmann::json::array();
    for (bool on : {false,true,true,false,true,false,false,true,false,true,true,false,true,false,false,true}) {
      typed.Select(on); c.Reset(); const auto start=std::chrono::steady_clock::now();
      c.Run(); backend.Synchronize(gpu.q);
      const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
      trials.push_back({{"typed",on},{"complete_operator_ms",ms}});
      SameBytes(c.output.download(), expected); SameBytes(c.state.download(), snapshots);
    }
    report["cases"].push_back({{"requests",c.requests},{"lengths",lengths},
        {"dtype","F16 output/qkv, F32 gates/state"},{"accepted_selectors",c.counts},
        {"state_bytes",c.state.bytes},{"trials",trials},{"all_output_snapshots_exact",true}});
  }
  report["SLM_per_WG_bytes"]=32768; report["new_device_workspace_bytes"]=0;
  std::ofstream f(path); REQUIRE(f.good()); f << report.dump(2) << std::endl;
}

TEST_CASE("XPU P7 speculative GDN SLM typed: actual selected kernel") {
  if (!std::getenv("VT_B70_GDN_SLM_DEFAULT_PROBE")) { MESSAGE("run selection probe with device profiling"); return; }
  REQUIRE(std::getenv("VT_XPU_GDN_SPEC_WG")==nullptr);
  Queue gpu(vt::DeviceType::kXPU); P7SpecSlmEnv mode; P7SpecTypedEnv typed;
  mode.Select(true);
  for (auto type : {DType::kF16, DType::kF32}) {
    P7SpecOperands c(gpu.q, {4}, type);
    for (int setting : {0, 1, 2}) {
      c.Reset();
      if (setting == 2) REQUIRE(unsetenv("VT_XPU_GDN_SPEC_SLM_TYPED") == 0);
      else typed.Select(setting == 1);
      (void)vt::xpu::DrainProfileEvents(); c.Run();
      c.output.download();
      int selected=0, fallback=0;
      for (const auto& event : vt::xpu::DrainProfileEvents()) {
        selected+=event.stage=="gdn_spec_decode_slm_typed";
        fallback+=event.stage=="gdn_spec_decode_slm";
      }
      CHECK(selected==int(setting != 0 && type==DType::kF16));
      CHECK(fallback==int(setting == 0 || type==DType::kF32));
      std::cout << "P7_SLM_TYPED_ROUTE output=" << vt::Name(type) << " setting=" << setting
                << " selected=" << selected << " fallback=" << fallback << '\n';
    }
  }
}
