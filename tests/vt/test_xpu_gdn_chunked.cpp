#include "xpu_test_helpers.h"
#include "vt/xpu.h"
#include "vt/xpu/xpu_kernels.h"
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <fstream>
#include <map>
#include <string_view>
#include <nlohmann/json.hpp>

namespace {
using vt::DType;
using xpu_test::Buffer;
using xpu_test::Queue;
constexpr int D = 128;
size_t ExpectedWorkspaceBytes() {
  const char* requested = std::getenv("VT_XPU_GDN_BATCH");
  const char* qk = std::getenv("VT_XPU_GDN_QK");
  const char* inverse = std::getenv("VT_XPU_GDN_INVERSE");
  const bool compatible = (!qk || std::string_view(qk) == "xmx") &&
      (!inverse || std::string_view(inverse) == "blocked");
  return (compatible && (!requested || std::string_view(requested) == "2")
      ? 32u : 16u) * 1024 * 1024;
}

std::vector<float> Values(int count, int salt, float scale = 0.01f) {
  std::vector<float> data(count);
  uint32_t seed = 0x83175u + salt;
  for (auto& value : data) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    value = (int(seed % 65536) - 32768) / 32768.0f * scale;
  }
  return data;
}
std::vector<float> Normalized(int count, int salt) {
  auto data = Values(count, salt, 1.0f);
  for (int row = 0; row < count; row += D) {
    float norm = 1e-6f;
    for (int d = 0; d < D; ++d) norm += data[row + d] * data[row + d];
    for (int d = 0; d < D; ++d) data[row + d] /= std::sqrt(norm);
  }
  return data;
}
void Accuracy(const std::vector<float>& actual, const std::vector<float>& expected, bool bf16 = false) {
  REQUIRE(actual.size() == expected.size());
  double error = 0, norm = 0, peak_error = 0, peak = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) FAIL("nonfinite GDN output/state");
    const double delta = double(actual[i]) - expected[i];
    error += delta * delta; norm += double(expected[i]) * expected[i];
    peak_error = std::max(peak_error, std::abs(delta)); peak = std::max(peak, std::abs(double(expected[i])));
  }
  const double rms = std::sqrt(error / std::max(norm, 1e-30));
  std::cout << "GDN_ERROR bf16=" << bf16 << " rms=" << rms << " max_abs=" << peak_error << std::endl;
  CHECK(rms <= (bf16 ? 1e-3 : 3e-5));
  CHECK(peak_error <= 2e-6 + (bf16 ? 0.008 : 3e-5) * peak);
}
struct Result { std::vector<float> output, state; };
std::vector<unsigned char> ReplayBytes(const std::string& dir,
                                       const char* stage, size_t expected) {
  const std::string path = dir + "/s-1_l0_" + stage + ".bin";
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  REQUIRE_MESSAGE(file.is_open(), "missing real GDN replay: ", path);
  REQUIRE(static_cast<size_t>(file.tellg()) == expected);
  std::vector<unsigned char> bytes(expected);
  file.seekg(0);
  REQUIRE(static_cast<bool>(file.read(reinterpret_cast<char*>(bytes.data()),
                                      static_cast<std::streamsize>(expected))));
  return bytes;
}
Result Run(vt::Queue& queue, const std::vector<int32_t>& offsets, int heads, DType output_type,
           bool initial, int split = 0, bool public_call = false, bool zero_decay = false,
           DType input_type = DType::kBF16) {
  const int tokens = offsets.back(), sequences = offsets.size() - 1, kh = heads / 3;
  Buffer q(queue, input_type, {tokens, kh, D}), k(queue, input_type, {tokens, kh, D});
  Buffer v(queue, input_type, {tokens, heads, D}), out(queue, output_type, {tokens, heads, D});
  Buffer g(queue, DType::kF32, {tokens, heads}), beta(queue, DType::kF32, {tokens, heads});
  Buffer state(queue, DType::kF32, {sequences, heads, D, D}), qsl(queue, DType::kI32, {sequences + 1});
  q.put(Normalized(tokens * kh * D, 1)); k.put(Normalized(tokens * kh * D, 2));
  v.put(Values(tokens * heads * D, 3, 0.5f));
  auto gate = Values(tokens * heads, 4, 0.2f), b = Values(tokens * heads, 5, 0.45f);
  for (int i = 0; i < tokens * heads; ++i) {
    gate[i] = zero_decay ? 0 : i % 19 ? -std::abs(gate[i]) : 0;
    b[i] = zero_decay ? float(i % 2) : b[i] + 0.5f;
  }
  g.put(gate); beta.put(b);
  state.put(initial ? Values(sequences * heads * D * D, 6, 0.1f) : std::vector<float>(sequences * heads * D * D));
  qsl.upload(offsets.data());
  if (public_call) {
    vt::GdnPrefill(queue, out.tensor, q.tensor, k.tensor, v.tensor, g.tensor, beta.tensor, state.tensor, qsl.tensor, {0.0883883476f});
  } else if (queue.device.type == vt::DeviceType::kCPU) {
    // Explicit independent oracle: CPU's BF16 chunked algorithm has additional
    // rounding sites and is not the scalar F32 recurrence required here.
    const char* value = std::getenv("VT_GDN_CHUNKED");
    const bool had = value != nullptr; const std::string old = had ? value : "";
    setenv("VT_GDN_CHUNKED", "0", 1);
    vt::GdnPrefill(queue, out.tensor, q.tensor, k.tensor, v.tensor, g.tensor, beta.tensor, state.tensor, qsl.tensor, {0.0883883476f});
    if (had) setenv("VT_GDN_CHUNKED", old.c_str(), 1); else unsetenv("VT_GDN_CHUNKED");
  } else {
    auto rows = [](const vt::Tensor& tensor, int begin, int count) {
      auto view = tensor; view.shape[0] = count;
      view.data = static_cast<char*>(view.data) + begin * view.stride[0] * vt::SizeOf(view.dtype);
      return view;
    };
    if (split) {
      REQUIRE(sequences == 1); REQUIRE(split < tokens);
      const int32_t prefix[] = {0, split}; qsl.upload(prefix);
      auto target = rows(out.tensor, 0, split);
      REQUIRE(vt::xpu::GdnChunkedPrefillKernel(queue, target, rows(q.tensor, 0, split), rows(k.tensor, 0, split),
          rows(v.tensor, 0, split), rows(g.tensor, 0, split), rows(beta.tensor, 0, split), state.tensor, qsl.tensor, {0.0883883476f}));
      for (int token = split; token < tokens; ++token) {
        target = rows(out.tensor, token, 1);
        vt::GdnDecode(queue, target, rows(q.tensor, token, 1), rows(k.tensor, token, 1), rows(v.tensor, token, 1),
                      rows(g.tensor, token, 1), rows(beta.tensor, token, 1), state.tensor, {0.0883883476f});
      }
    } else REQUIRE(vt::xpu::GdnChunkedPrefillKernel(queue, out.tensor, q.tensor, k.tensor, v.tensor,
                        g.tensor, beta.tensor, state.tensor, qsl.tensor, {0.0883883476f}));
  }
  return {out.floats(), state.floats()};
}
}

TEST_CASE("XPU GDN P4 real GPTQ layer-0 prefill replay"
          * doctest::skip(!std::getenv("VT_B70_GDN_REPLAY_DIR"))) {
  const std::string dir = std::getenv("VT_B70_GDN_REPLAY_DIR");
  Queue gpu(vt::DeviceType::kXPU);
  auto& q = gpu.q;
  constexpr int tokens = 4096, key_heads = 16, heads = 48;
  Buffer qi(q, DType::kF16, {tokens, key_heads, D});
  Buffer ki(q, DType::kF16, {tokens, key_heads, D});
  Buffer vi(q, DType::kF16, {tokens, heads, D});
  Buffer g(q, DType::kF32, {tokens, heads});
  Buffer beta(q, DType::kF32, {tokens, heads});
  Buffer state(q, DType::kF32, {1, heads, D, D});
  Buffer qsl(q, DType::kI32, {2});
  Buffer out(q, DType::kF16, {tokens, heads, D});
  qi.upload(ReplayBytes(dir, "q", qi.bytes).data());
  ki.upload(ReplayBytes(dir, "k", ki.bytes).data());
  vi.upload(ReplayBytes(dir, "v", vi.bytes).data());
  g.upload(ReplayBytes(dir, "g", g.bytes).data());
  beta.upload(ReplayBytes(dir, "beta", beta.bytes).data());
  state.upload(ReplayBytes(dir, "ssm_state_before", state.bytes).data());
  qsl.upload(ReplayBytes(dir, "query_start_loc", qsl.bytes).data());
  vt::GdnPrefill(q, out.tensor, qi.tensor, ki.tensor, vi.tensor, g.tensor,
                 beta.tensor, state.tensor, qsl.tensor, {0.0883883476f});
  xpu_test::SameBytes(out.download(), ReplayBytes(dir, "core_out", out.bytes));
  xpu_test::SameBytes(state.download(), ReplayBytes(dir, "ssm_state_after", state.bytes));

  // The second half starts from the actual state produced by the first half.
  // A 2048-token split lands exactly on a 64-token GDN chunk boundary.
  state.upload(ReplayBytes(dir, "ssm_state_before", state.bytes).data());
  const int32_t half_offsets[2] = {0, tokens / 2};
  qsl.upload(half_offsets);
  auto rows = [](vt::Tensor tensor, int begin, int count) {
    tensor.data = static_cast<char*>(tensor.data) +
        begin * tensor.stride[0] * vt::SizeOf(tensor.dtype);
    tensor.shape[0] = count;
    return tensor;
  };
  auto run_half = [&](int begin) {
    auto o = rows(out.tensor, begin, tokens / 2);
    auto qh = rows(qi.tensor, begin, tokens / 2);
    auto kh = rows(ki.tensor, begin, tokens / 2);
    auto vh = rows(vi.tensor, begin, tokens / 2);
    auto gh = rows(g.tensor, begin, tokens / 2);
    auto bh = rows(beta.tensor, begin, tokens / 2);
    vt::GdnPrefill(q, o, qh, kh, vh, gh, bh, state.tensor, qsl.tensor,
                   {0.0883883476f});
  };
  run_half(0);
  const auto intermediate = state.download();
  REQUIRE(std::any_of(intermediate.begin(), intermediate.end(),
                      [](unsigned char byte) { return byte != 0; }));
  run_half(tokens / 2);
  xpu_test::SameBytes(out.download(), ReplayBytes(dir, "core_out", out.bytes));
  xpu_test::SameBytes(state.download(), ReplayBytes(dir, "ssm_state_after", state.bytes));
}

TEST_CASE("XPU GDN P4 native Xe2 real layer diagnostic"
          * doctest::skip(!std::getenv("VT_B70_GDN_REPLAY_DIR") ||
                          !std::getenv("VT_B70_GDN_NATIVE_TEST"))) {
#ifdef VLLM_CPP_XPU_XE2_GDN
  const std::string dir = std::getenv("VT_B70_GDN_REPLAY_DIR");
  Queue gpu(vt::DeviceType::kXPU);
  auto& q = gpu.q;
  constexpr int tokens = 4096, key_heads = 16, heads = 48;
  Buffer qi(q, DType::kF16, {tokens, key_heads, D});
  Buffer ki(q, DType::kF16, {tokens, key_heads, D});
  Buffer vi(q, DType::kF16, {tokens, heads, D});
  Buffer g(q, DType::kF32, {tokens, heads});
  Buffer beta(q, DType::kF32, {tokens, heads});
  Buffer state(q, DType::kF32, {1, heads, D, D});
  Buffer qsl(q, DType::kI32, {2});
  Buffer out(q, DType::kF16, {tokens, heads, D});
  qi.upload(ReplayBytes(dir, "q", qi.bytes).data());
  ki.upload(ReplayBytes(dir, "k", ki.bytes).data());
  vi.upload(ReplayBytes(dir, "v", vi.bytes).data());
  g.upload(ReplayBytes(dir, "g", g.bytes).data());
  beta.upload(ReplayBytes(dir, "beta", beta.bytes).data());
  state.upload(ReplayBytes(dir, "ssm_state_before", state.bytes).data());
  qsl.upload(ReplayBytes(dir, "query_start_loc", qsl.bytes).data());
  REQUIRE(vt::xpu::GdnNativePrefillKernel(q, out.tensor, qi.tensor, ki.tensor,
      vi.tensor, g.tensor, beta.tensor, state.tensor, qsl.tensor,
      {0.0883883476f}));
  CHECK(vt::xpu::GetMemoryInfo().native_gdn_workspace_bytes > 0);
  CHECK(vt::xpu::GetMemoryInfo().native_gdn_workspace_bytes <= 160u * 1024 * 1024);
  const auto output = out.download();
  const auto final_state = state.download();
  auto compare = [&](const char* stage, const std::vector<unsigned char>& actual,
                     DType dtype) {
    const auto expected = ReplayBytes(dir, stage, actual.size());
    const size_t elements = actual.size() / vt::SizeOf(dtype);
    double squared_error = 0, squared_ref = 0, max_abs = 0;
    for (size_t i = 0; i < elements; ++i) {
      float a, b;
      if (dtype == DType::kF32) {
        std::memcpy(&a, actual.data() + 4 * i, 4);
        std::memcpy(&b, expected.data() + 4 * i, 4);
      } else {
        uint16_t ah, bh;
        std::memcpy(&ah, actual.data() + 2 * i, 2);
        std::memcpy(&bh, expected.data() + 2 * i, 2);
        a = vt::F16ToF32(ah); b = vt::F16ToF32(bh);
      }
      REQUIRE(std::isfinite(a));
      const double diff = double(a) - b;
      squared_error += diff * diff;
      squared_ref += double(b) * b;
      max_abs = std::max(max_abs, std::abs(diff));
    }
    const double relative_rms = std::sqrt(squared_error / std::max(squared_ref, 1e-30));
    std::cout << "GDN_NATIVE_REAL stage=" << stage << " relative_rms="
              << relative_rms << " max_abs=" << max_abs << std::endl;
  };
  compare("core_out", output, DType::kF16);
  compare("ssm_state_after", final_state, DType::kF32);
  // Reuse the real first-pass state as a nonempty continuation input. The
  // frozen Python snapshot covers only the fresh-state call, so this is a
  // diagnostic native-versus-existing-C++ comparison, not a Python gate.
  state.upload(final_state.data());
  REQUIRE(vt::xpu::GdnNativePrefillKernel(q, out.tensor, qi.tensor, ki.tensor,
      vi.tensor, g.tensor, beta.tensor, state.tensor, qsl.tensor,
      {0.0883883476f}));
  const auto native_cont_out = out.floats();
  const auto native_cont_state = state.floats();
  state.upload(final_state.data());
  REQUIRE(vt::xpu::GdnChunkedPrefillKernel(q, out.tensor, qi.tensor, ki.tensor,
      vi.tensor, g.tensor, beta.tensor, state.tensor, qsl.tensor,
      {0.0883883476f}));
  const auto old_cont_out = out.floats();
  const auto old_cont_state = state.floats();
  auto continuation_error = [&](const char* stage, const std::vector<float>& actual,
                                const std::vector<float>& reference) {
    REQUIRE(actual.size() == reference.size());
    double squared_error = 0, squared_ref = 0, max_abs = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
      REQUIRE(std::isfinite(actual[i]));
      const double diff = double(actual[i]) - reference[i];
      squared_error += diff * diff;
      squared_ref += double(reference[i]) * reference[i];
      max_abs = std::max(max_abs, std::abs(diff));
    }
    std::cout << "GDN_NATIVE_CONTINUATION stage=" << stage
              << " relative_rms=" << std::sqrt(squared_error / std::max(squared_ref, 1e-30))
              << " max_abs=" << max_abs << std::endl;
  };
  continuation_error("core_out", native_cont_out, old_cont_out);
  continuation_error("ssm_state_after", native_cont_state, old_cont_state);
  if (const char* dump = std::getenv("VT_B70_GDN_NATIVE_DUMP_DIR")) {
    std::ofstream output_file(std::string(dump) + "/native_core_out.bin", std::ios::binary);
    std::ofstream state_file(std::string(dump) + "/native_ssm_state_after.bin", std::ios::binary);
    REQUIRE(output_file.write(reinterpret_cast<const char*>(output.data()), output.size()));
    REQUIRE(state_file.write(reinterpret_cast<const char*>(final_state.data()), final_state.size()));
  }
  if (std::getenv("VT_B70_GDN_NATIVE_BENCH")) {
    auto& backend = vt::GetBackend(q.device);
    auto measure = [&](bool native) {
      std::vector<double> milliseconds;
      for (int repetition = 0; repetition < 6; ++repetition) {
        backend.Memset(q, state.tensor.data, 0, state.bytes);
        backend.Synchronize(q);
        const auto start = std::chrono::steady_clock::now();
        const bool selected = native
            ? vt::xpu::GdnNativePrefillKernel(q, out.tensor, qi.tensor, ki.tensor,
                vi.tensor, g.tensor, beta.tensor, state.tensor, qsl.tensor,
                {0.0883883476f})
            : vt::xpu::GdnChunkedPrefillKernel(q, out.tensor, qi.tensor, ki.tensor,
                vi.tensor, g.tensor, beta.tensor, state.tensor, qsl.tensor,
                {0.0883883476f});
        REQUIRE(selected);
        backend.Synchronize(q);
        const auto end = std::chrono::steady_clock::now();
        if (repetition) milliseconds.push_back(
            std::chrono::duration<double, std::milli>(end - start).count());
      }
      std::sort(milliseconds.begin(), milliseconds.end());
      return milliseconds[milliseconds.size() / 2];
    };
    const double old_ms = measure(false);
    const double native_ms = measure(true);
    std::cout << "GDN_NATIVE_REAL_BENCH old_median_ms=" << old_ms
              << " native_median_ms=" << native_ms
              << " speedup=" << old_ms / native_ms << std::endl;
  }
#endif
}

TEST_CASE("XPU GDN P4 native Xe2 long macro continuation diagnostic"
          * doctest::skip(!std::getenv("VT_B70_GDN_REPLAY_DIR") ||
                          !std::getenv("VT_B70_GDN_LONG_TEST"))) {
#ifdef VLLM_CPP_XPU_XE2_GDN
  const std::string dir = std::getenv("VT_B70_GDN_REPLAY_DIR");
  Queue gpu(vt::DeviceType::kXPU);
  auto& q = gpu.q;
  constexpr int segment = 4096, key_heads = 16, heads = 48;
  const int tokens = std::getenv("VT_B70_GDN_LONG_TOKENS")
      ? std::stoi(std::getenv("VT_B70_GDN_LONG_TOKENS")) : 2 * segment;
  const bool supported_length = tokens == 2 * segment || tokens == 4 * segment;
  REQUIRE(supported_length);
  Buffer qi(q, DType::kF16, {tokens, key_heads, D});
  Buffer ki(q, DType::kF16, {tokens, key_heads, D});
  Buffer vi(q, DType::kF16, {tokens, heads, D});
  Buffer g(q, DType::kF32, {tokens, heads});
  Buffer beta(q, DType::kF32, {tokens, heads});
  Buffer state(q, DType::kF32, {1, heads, D, D});
  Buffer qsl(q, DType::kI32, {2});
  Buffer out(q, DType::kF16, {tokens, heads, D});
  auto duplicate = [&](Buffer& buffer, const char* name) {
    const auto first = ReplayBytes(dir, name, buffer.bytes / (tokens / segment));
    std::vector<unsigned char> data(buffer.bytes);
    for (int token = 0; token < tokens; token += segment)
      std::memcpy(data.data() + token * buffer.bytes / tokens,
                  first.data(), first.size());
    buffer.upload(data.data());
  };
  duplicate(qi, "q"); duplicate(ki, "k"); duplicate(vi, "v");
  duplicate(g, "g"); duplicate(beta, "beta");
  const auto initial = ReplayBytes(dir, "ssm_state_before", state.bytes);
  state.upload(initial.data());
  const int32_t full_offsets[2] = {0, tokens};
  qsl.upload(full_offsets);
  REQUIRE(vt::xpu::GdnNativePrefillKernel(q, out.tensor, qi.tensor, ki.tensor,
      vi.tensor, g.tensor, beta.tensor, state.tensor, qsl.tensor,
      {0.0883883476f}));
  const auto native_out = out.floats();
  const auto native_state = state.floats();
  CHECK(vt::xpu::GetMemoryInfo().native_gdn_workspace_bytes <= 160u * 1024 * 1024);

  state.upload(initial.data());
  const int32_t segment_offsets[2] = {0, segment};
  qsl.upload(segment_offsets);
  auto rows = [](vt::Tensor tensor, int first, int count) {
    tensor.data = static_cast<char*>(tensor.data) +
        first * tensor.stride[0] * vt::SizeOf(tensor.dtype);
    tensor.shape[0] = count;
    return tensor;
  };
  for (int first = 0; first < tokens; first += segment) {
    auto target = rows(out.tensor, first, segment);
    REQUIRE(vt::xpu::GdnChunkedPrefillKernel(q, target,
        rows(qi.tensor, first, segment), rows(ki.tensor, first, segment),
        rows(vi.tensor, first, segment), rows(g.tensor, first, segment),
        rows(beta.tensor, first, segment), state.tensor, qsl.tensor,
        {0.0883883476f}));
  }
  const auto old_out = out.floats();
  const auto old_state = state.floats();
  auto report = [](const char* name, const std::vector<float>& actual,
                   const std::vector<float>& reference) {
    REQUIRE(actual.size() == reference.size());
    double squared_error = 0, squared_ref = 0, max_abs = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
      REQUIRE(std::isfinite(actual[i]));
      const double diff = double(actual[i]) - reference[i];
      squared_error += diff * diff;
      squared_ref += double(reference[i]) * reference[i];
      max_abs = std::max(max_abs, std::abs(diff));
    }
    std::cout << "GDN_NATIVE_LONG stage=" << name << " relative_rms="
              << std::sqrt(squared_error / std::max(squared_ref, 1e-30))
              << " max_abs=" << max_abs << std::endl;
  };
  report("core_out", native_out, old_out);
  report("ssm_state_after", native_state, old_state);
  if (std::getenv("VT_B70_GDN_LONG_BENCH")) {
    auto& backend = vt::GetBackend(q.device);
    auto measure = [&](bool native) {
      qsl.upload(native ? full_offsets : segment_offsets);
      std::vector<double> milliseconds;
      for (int repetition = 0; repetition < 6; ++repetition) {
        state.upload(initial.data());
        const auto start = std::chrono::steady_clock::now();
        if (native) {
          REQUIRE(vt::xpu::GdnNativePrefillKernel(q, out.tensor, qi.tensor,
              ki.tensor, vi.tensor, g.tensor, beta.tensor, state.tensor,
              qsl.tensor, {0.0883883476f}));
        } else {
          for (int first = 0; first < tokens; first += segment) {
            auto target = rows(out.tensor, first, segment);
            REQUIRE(vt::xpu::GdnChunkedPrefillKernel(q, target,
                rows(qi.tensor, first, segment), rows(ki.tensor, first, segment),
                rows(vi.tensor, first, segment), rows(g.tensor, first, segment),
                rows(beta.tensor, first, segment), state.tensor, qsl.tensor,
                {0.0883883476f}));
          }
        }
        backend.Synchronize(q);
        const auto end = std::chrono::steady_clock::now();
        if (repetition) milliseconds.push_back(
            std::chrono::duration<double, std::milli>(end - start).count());
      }
      std::sort(milliseconds.begin(), milliseconds.end());
      return milliseconds[milliseconds.size() / 2];
    };
    const double old_ms = measure(false);
    const double native_ms = measure(true);
    std::cout << "GDN_NATIVE_LONG_BENCH tokens=" << tokens << " old_median_ms="
              << old_ms << " native_median_ms=" << native_ms
              << " speedup=" << old_ms / native_ms << std::endl;
  }
#endif
}

TEST_CASE("XPU GDN chunk64: F16 XMX output and F32 state match sequential CPU") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (int length : {64, 65}) {
    const auto ref = Run(cpu.q, {0, length}, 48, DType::kF32, true,
                         0, false, false, DType::kF16);
    const auto got = Run(gpu.q, {0, length}, 48, DType::kF16, true,
                         0, false, false, DType::kF16);
    Accuracy(got.output, ref.output, true);
    Accuracy(got.state, ref.state);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU GDN chunk64: F16 long continuation and zero decay") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (const auto [length, zero_decay] : {std::pair{4097, false}, std::pair{1025, true}}) {
    const auto ref = Run(cpu.q, {0, length}, 6, DType::kF32, true,
                         0, false, zero_decay, DType::kF16);
    const auto got = Run(gpu.q, {0, length}, 6, DType::kF16, true,
                         0, false, zero_decay, DType::kF16);
    Accuracy(got.output, ref.output, true);
    Accuracy(got.state, ref.state);
  }
}

TEST_CASE("XPU GDN short F16 prefill selects chunked kernel"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE"))) {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  const auto ref = Run(cpu.q, {0, 64}, 48, DType::kF32, true,
                       0, false, false, DType::kF16);
  (void)vt::xpu::DrainProfileEvents();
  const auto got = Run(gpu.q, {0, 64}, 48, DType::kF16, true,
                       0, true, false, DType::kF16);
  Accuracy(got.output, ref.output, true);
  Accuracy(got.state, ref.state);
  const auto records = vt::xpu::DrainProfileEvents();
  size_t dots = 0, recurrence = 0, wu_xmx = 0, wu_xmx_prepare = 0;
  size_t state_xmx = 0, state_xmx_prepare = 0, state_tile4 = 0;
  for (const auto& record : records) {
    dots += record.stage == "gdn_chunk_dots_qk";
    recurrence += record.stage == "gdn_prefill_recurrence";
    wu_xmx += record.stage == "gdn_chunk_wu_xmx";
    wu_xmx_prepare += record.stage == "gdn_chunk_wu_xmx_prepare";
    state_xmx += record.stage == "gdn_chunk_state_xmx";
    state_xmx_prepare += record.stage == "gdn_chunk_state_xmx_prepare";
    state_tile4 += record.stage == "gdn_chunk_state_tile4";
  }
  CHECK(dots == 1);
  CHECK(recurrence == 0);
  const char* requested = std::getenv("VT_XPU_GDN_WU");
  const std::string_view wu = requested ? requested : "auto";
  const bool expect_xmx = wu == "auto" || wu == "xmx";
  CHECK(wu_xmx == static_cast<size_t>(expect_xmx));
  CHECK(wu_xmx_prepare == static_cast<size_t>(expect_xmx));
  requested = std::getenv("VT_XPU_GDN_STATE");
  const std::string_view state = requested ? requested : "auto";
  const bool expect_state_xmx = state == "auto" || state == "xmx";
  CHECK(state_xmx == static_cast<size_t>(expect_state_xmx));
  CHECK(state_xmx_prepare == static_cast<size_t>(expect_state_xmx));
  CHECK(state_tile4 == static_cast<size_t>(!expect_state_xmx));
}

TEST_CASE("XPU GDN F16 batch-two prefill groups independent stages"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE"))) {
  if (ExpectedWorkspaceBytes() != 32u * 1024 * 1024) return;
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  const auto ref = Run(cpu.q, {0, 128}, 6, DType::kF32, true,
                       0, false, false, DType::kF16);
  (void)vt::xpu::DrainProfileEvents();
  const auto got = Run(gpu.q, {0, 128}, 6, DType::kF16, true,
                       0, false, false, DType::kF16);
  Accuracy(got.output, ref.output, true);
  Accuracy(got.state, ref.state);
  std::map<std::string, size_t> counts;
  for (const auto& record : vt::xpu::DrainProfileEvents()) ++counts[record.stage];
  for (const char* stage : {"gdn_chunk_prepare_pair", "gdn_chunk_dots_qk_pair",
                            "gdn_chunk_system_pair", "gdn_chunk_inverse_blocked_pair",
                            "gdn_chunk_state_transpose"}) CHECK(counts[stage] == 1);
  CHECK(counts["gdn_chunk_prepare"] == 0);
  CHECK(vt::xpu::GetMemoryInfo().gdn_workspace_bytes == ExpectedWorkspaceBytes());
}

TEST_CASE("XPU GDN chunk64: full output and F32 state against sequential CPU") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (bool initial : {false, true}) for (int length : {1, 63, 64, 65, 127, 128, 129}) {
    CAPTURE(length);
    CAPTURE(initial);
    const auto ref = Run(cpu.q, {0, length}, 6, DType::kF32, initial);
    const auto got = Run(gpu.q, {0, length}, 6, DType::kF32, initial);
    Accuracy(got.output, ref.output); Accuracy(got.state, ref.state);
  }
  for (DType dtype : {DType::kF32, DType::kBF16}) {
    const auto ref = Run(cpu.q, {0, 129}, 48, dtype, true);
    const auto got = Run(gpu.q, {0, 129}, 48, dtype, true);
    Accuracy(got.output, ref.output, dtype == DType::kBF16); Accuracy(got.state, ref.state);
  }
  CHECK(vt::xpu::GetMemoryInfo().gdn_workspace_bytes == ExpectedWorkspaceBytes());
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU GDN profile: chunk stages and decode recurrence"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE"))) {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  const auto expected = Run(cpu.q, {0, 65}, 6, DType::kF32, true);
  (void)vt::xpu::DrainProfileEvents();
  const auto actual = Run(gpu.q, {0, 65}, 6, DType::kF32, true, 64);
  Accuracy(actual.output, expected.output);
  Accuracy(actual.state, expected.state);
  const auto records = vt::xpu::DrainProfileEvents();
  std::map<std::string, size_t> counts;
  for (const auto& record : records) {
    ++counts[record.stage];
    CHECK(record.queue_id == gpu.q.id);
    CHECK(record.start_ns > 0);
    CHECK(record.end_ns >= record.start_ns);
  }
  const char* inverse_mode = std::getenv("VT_XPU_GDN_INVERSE");
  const std::string_view mode = inverse_mode ? inverse_mode : "blocked";
  const char* inverse_stage = mode == "reference" ? "gdn_chunk_inverse"
      : mode == "blocked" ? "gdn_chunk_inverse_blocked" : "gdn_chunk_inverse_slm";
  for (const char* stage : {"gdn_chunk_prepare", "gdn_chunk_dots_qk",
                            "gdn_chunk_system", inverse_stage, "gdn_chunk_wu_tile4",
                            "gdn_chunk_delta_cross_tile4", "gdn_chunk_output_tile4", "gdn_chunk_state_tile4",
                            "gdn_decode_recurrence"}) CHECK(counts[stage] == 1);
  CHECK(records.size() == 9);
}

TEST_CASE("XPU GDN chunk64: empty and unequal sequences, long drift, decode continuation") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (const auto& offsets : {std::vector<int32_t>{0, 0, 1, 64, 129}, std::vector<int32_t>{0, 4097}}) {
    const auto ref = Run(cpu.q, offsets, 6, DType::kF32, true);
    const auto got = Run(gpu.q, offsets, 6, DType::kF32, true);
    Accuracy(got.output, ref.output); Accuracy(got.state, ref.state);
    if (offsets[1] == 0) CHECK(std::equal(got.state.begin(), got.state.begin() + 6 * D * D, ref.state.begin()));
  }
  const auto ref = Run(cpu.q, {0, 145}, 6, DType::kF32, true);
  const auto got = Run(gpu.q, {0, 145}, 6, DType::kF32, true, 79);
  Accuracy(got.output, ref.output); Accuracy(got.state, ref.state);
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == ExpectedWorkspaceBytes());
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU GDN chunk64: persistent state with zero decay and saturated gates") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  const auto ref = Run(cpu.q, {0, 1025}, 6, DType::kF32, true, 0, false, true);
  const auto got = Run(gpu.q, {0, 1025}, 6, DType::kF32, true, 0, false, true);
  Accuracy(got.output, ref.output); Accuracy(got.state, ref.state);
}

TEST_CASE("XPU GDN chunk64: workspace reuse after queue replacement") {
  Result first;
  for (int repeat = 0; repeat < 3; ++repeat) {
    Queue gpu(vt::DeviceType::kXPU);
    const auto got = Run(gpu.q, {0, 65}, 6, DType::kF32, true);
    if (!repeat) first = got;
    else { CHECK(got.output == first.output); CHECK(got.state == first.state); }
    CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == ExpectedWorkspaceBytes());
  }
}

TEST_CASE("XPU GDN chunk64: insufficient workspace budget uses native reference"
          * doctest::skip(!std::getenv("VT_B70_LOW_MEMORY_TEST"))) {
  const char* mode = std::getenv("VT_XPU_GDN_PREFILL");
  REQUIRE(mode != nullptr); REQUIRE(std::string(mode) == "chunked");
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  REQUIRE(vt::xpu::GetMemoryInfo().budget_bytes == 8 * 1024 * 1024);
  const auto ref = Run(cpu.q, {0, 65}, 6, DType::kF32, true);
  const auto got = Run(gpu.q, {0, 65}, 6, DType::kF32, true, 0, true);
  Accuracy(got.output, ref.output); Accuracy(got.state, ref.state);
  CHECK(vt::xpu::GetMemoryInfo().gdn_workspace_bytes == 0);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU GDN chunk64: timing" * doctest::skip(!std::getenv("VT_B70_GDN_BENCH"))) {
  setenv("VT_GDN_CHUNKED", "0", 1);  // public call remains the native sequential comparator
  Queue gpu(vt::DeviceType::kXPU); auto& queue = gpu.q;
  auto& backend = vt::GetBackend(queue.device);
  constexpr int heads = 48, kh = 16;
  const char* bench_f16 = std::getenv("VT_B70_GDN_BENCH_F16");
  const DType bench_type = bench_f16 && std::string_view(bench_f16) == "1"
      ? DType::kF16 : DType::kBF16;
  for (int tokens : {128, 512, 2048, 4096}) {
    Buffer q(queue, bench_type, {tokens, kh, D}), k(queue, bench_type, {tokens, kh, D});
    Buffer v(queue, bench_type, {tokens, heads, D}), out(queue, bench_type, {tokens, heads, D});
    Buffer g(queue, DType::kF32, {tokens, heads}), beta(queue, DType::kF32, {tokens, heads});
    Buffer state(queue, DType::kF32, {1, heads, D, D}), qsl(queue, DType::kI32, {2});
    q.put(Normalized(tokens * kh * D, 1)); k.put(Normalized(tokens * kh * D, 2));
    v.put(Values(tokens * heads * D, 3, 0.5f));
    g.put(std::vector<float>(tokens * heads, -0.13f)); beta.put(std::vector<float>(tokens * heads, 0.61f));
    const int32_t offsets[] = {0, tokens}; qsl.upload(offsets);
    Result reference;
    for (bool chunked : {false, true}) {
      auto run = [&] {
        if (chunked) REQUIRE(vt::xpu::GdnChunkedPrefillKernel(queue, out.tensor, q.tensor, k.tensor,
              v.tensor, g.tensor, beta.tensor, state.tensor, qsl.tensor, {0.0883883476f}));
        else vt::GdnPrefill(queue, out.tensor, q.tensor, k.tensor, v.tensor, g.tensor,
                            beta.tensor, state.tensor, qsl.tensor, {0.0883883476f});
        backend.Synchronize(queue);
      };
      auto reset = [&] { backend.Memset(queue, state.tensor.data, 0, state.bytes); backend.Synchronize(queue); };
      reset(); run();
      auto start = std::chrono::steady_clock::now();
      do { reset(); run(); } while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(200));
      std::vector<double> times;
      for (int i = 0; i < 5; ++i) {
        reset(); start = std::chrono::steady_clock::now(); run();
        times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
      }
      std::sort(times.begin(), times.end());
      std::cout << nlohmann::json{{"tokens", tokens}, {"dtype", bench_type == DType::kF16 ? "f16" : "bf16"},
          {"chunked", chunked}, {"median_ms", times[2]},
          {"samples_ms", times}, {"workspace_bytes", vt::xpu::GetMemoryInfo().gdn_workspace_bytes}}.dump() << std::endl;
      Result got{out.floats(), state.floats()};
      if (!chunked) reference = got;
      else { Accuracy(got.output, reference.output, true); Accuracy(got.state, reference.state); }
    }
  }
}
