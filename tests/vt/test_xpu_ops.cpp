#include <doctest/doctest.h>
#include "vt/backend.h"
#include "vt/breakable_graph.h"
#include "vt/ops.h"
#include "vt/xpu.h"
#include "vt/xpu_profile_span.h"
#include "vllm/v1/sample/sampler.h"
#include "vllm/platforms/interface.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp>
#include "vllm/model_executor/model_loader/safetensors_reader.h"

TEST_CASE("XPU profile spans: optional queue brackets preserve bytes and nested intervals") {
  const vt::Device gpu{vt::DeviceType::kXPU, 0};
  auto& backend = vt::GetBackend(gpu);
  auto q = vt::CreateQueue(gpu), other = vt::CreateQueue(gpu);
  auto* data = static_cast<unsigned char*>(vt::Alloc(gpu, 64));
  (void)vt::xpu::DrainProfileEvents();
  (void)vt::xpu::DrainHostProfileRecords();
  const bool device_profile = std::getenv("VT_XPU_PROFILE") &&
      std::string(std::getenv("VT_XPU_PROFILE")) == "1";
  const bool host_profile = std::getenv("VT_XPU_HOST_PROFILE") &&
      std::string(std::getenv("VT_XPU_HOST_PROFILE")) == "1";
  auto outer = vt::xpu::BeginProfileSpan(q);
  CHECK(bool(outer.state) == (device_profile || host_profile));
  backend.Memset(q, data, 0x35, 64);
  auto inner = vt::xpu::BeginProfileSpan(q);
  backend.Memset(q, data + 8, 0x72, 16);
  vt::xpu::EndProfileSpan(q, "probe_inner", std::move(inner));
  vt::xpu::EndProfileSpan(q, "probe_outer", std::move(outer));
  std::vector<unsigned char> got(64);
  backend.Copy(q, got.data(), data, got.size());
  backend.Synchronize(q);
  for (size_t i = 0; i < got.size(); ++i)
    CHECK(got[i] == (i >= 8 && i < 24 ? 0x72 : 0x35));
  const auto events = vt::xpu::DrainProfileEvents();
  const auto host = vt::xpu::DrainHostProfileRecords();
  std::vector<vt::xpu::ProfileRecord> spans;
  std::vector<vt::xpu::HostProfileRecord> host_spans;
  for (const auto& r : events)
    if (r.stage == "probe_inner" || r.stage == "probe_outer") spans.push_back(r);
  for (const auto& r : host)
    if (r.stage == "probe_inner" || r.stage == "probe_outer") host_spans.push_back(r);
  REQUIRE(spans.size() == (device_profile ? 2u : 0u));
  REQUIRE(host_spans.size() == (host_profile ? 2u : 0u));
  if (device_profile) {
    CHECK(spans[0].stage == "probe_inner");
    CHECK(spans[1].stage == "probe_outer");
    for (const auto& s : spans) {
      CHECK(s.stream_span);
      CHECK(s.queue_id == q.id);
      CHECK(s.end_ns >= s.start_ns);
    }
    CHECK(spans[1].start_ns <= spans[0].start_ns);
    CHECK(spans[1].end_ns >= spans[0].end_ns);
  }
  if (host_profile) {
    CHECK(host_spans[1].start_steady_ns <= host_spans[0].start_steady_ns);
    CHECK(host_spans[1].end_steady_ns >= host_spans[0].end_steady_ns);
  }
  if (device_profile || host_profile) {
    auto wrong_queue = vt::xpu::BeginProfileSpan(q);
    CHECK_THROWS(vt::xpu::EndProfileSpan(other, "probe_wrong_queue", std::move(wrong_queue)));
  }
  vt::Queue cpu{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  CHECK_FALSE(vt::xpu::BeginProfileSpan(cpu).state);
  vt::Free(gpu, data);
  vt::DestroyQueue(other); vt::DestroyQueue(q);
}

TEST_CASE("XPU profile copies: staged roundtrip records actual chunk bytes and callers") {
  const vt::Device gpu{vt::DeviceType::kXPU, 0};
  auto& backend = vt::GetBackend(gpu);
  auto q = vt::CreateQueue(gpu);
  constexpr size_t chunk = 4 * 1024 * 1024, bytes = chunk + 37;
  auto* data = vt::Alloc(gpu, bytes);
  std::vector<unsigned char> input(bytes), output(bytes);
  for (size_t i = 0; i < bytes; ++i) input[i] = static_cast<unsigned char>(i % 251);
  (void)vt::xpu::DrainProfileEvents();
  (void)vt::xpu::DrainHostProfileRecords();
  backend.Copy(q, data, input.data(), bytes);
  backend.Copy(q, output.data(), data, bytes);
  backend.Synchronize(q);
  CHECK(output == input);
  const auto records = vt::xpu::DrainHostProfileRecords();
  const auto events = vt::xpu::DrainProfileEvents();
  const bool device_profile = std::getenv("VT_XPU_PROFILE") &&
      std::string(std::getenv("VT_XPU_PROFILE")) == "1";
  // These are the existing memcpy commands, without extra queue anchors.
  // Their device intervals exclude commands queued before the copy.
  for (const char* stage : {"staged_h2d_copy", "staged_d2h_copy"}) {
    std::vector<vt::xpu::ProfileRecord> copies;
    for (const auto& r : events) if (r.stage == stage) copies.push_back(r);
    REQUIRE(copies.size() == (device_profile ? 2u : 0u));
    for (const auto& r : copies) {
      CHECK_FALSE(r.stream_span);
      CHECK(r.queue_id == q.id);
      CHECK(r.submit_ns > 0);
      CHECK(r.start_ns > 0);
      CHECK(r.end_ns >= r.start_ns);
    }
    if (device_profile) CHECK(copies[1].start_ns >= copies[0].end_ns);
  }
  const bool enabled = std::getenv("VT_XPU_HOST_PROFILE") &&
      std::string(std::getenv("VT_XPU_HOST_PROFILE")) == "1";
  if (!enabled) CHECK(records.empty());
  for (const char* stage : {"staged_h2d_wait", "staged_d2h_wait"}) {
    std::vector<vt::xpu::HostProfileRecord> copies;
    for (const auto& r : records) if (r.stage == stage) copies.push_back(r);
    CHECK(copies.size() == (enabled ? 2u : 0u));
    if (!enabled || copies.size() != 2) continue;
    CHECK(copies[0].copy_bytes == chunk);
    CHECK(copies[1].copy_bytes == 37);
    CHECK(copies[0].copy_bytes + copies[1].copy_bytes == bytes);
    CHECK(copies[0].caller_address != 0);
    CHECK(copies[0].caller_address == copies[1].caller_address);
    for (const auto& r : copies) {
      CHECK(r.queue_id == q.id);
      CHECK(r.end_steady_ns >= r.start_steady_ns);
    }
  }
  vt::Free(gpu, data);
  vt::DestroyQueue(q);
}

namespace {
using vt::DType;
struct Queues {
  vt::Queue cpu = vt::CreateQueue({vt::DeviceType::kCPU, 0});
  vt::Queue gpu = vt::CreateQueue({vt::DeviceType::kXPU, 0});
  ~Queues() { vt::DestroyQueue(gpu); vt::DestroyQueue(cpu); }
};
struct Buffer {
  vt::Queue& q;
  vt::Tensor t;
  size_t bytes;
  Buffer(vt::Queue& queue, DType dtype, std::initializer_list<int64_t> shape) : q(queue) {
    t = vt::Tensor::Contiguous(nullptr, dtype, q.device, shape);
    bytes = t.Bytes();
    t.data = vt::Alloc(q.device, std::max(bytes, size_t{1}));
  }
  ~Buffer() { vt::Free(q.device, t.data); }
  Buffer(const Buffer&) = delete;
  void upload(const void* data) {
    auto& b = vt::GetBackend(q.device); b.Copy(q, t.data, data, bytes); b.Synchronize(q);
  }
  void put(const std::vector<float>& values) {
    REQUIRE(values.size() * vt::SizeOf(t.dtype) == bytes);
    if (t.dtype == DType::kF32) { upload(values.data()); return; }
    std::vector<uint16_t> packed(values.size());
    for (size_t i = 0; i < values.size(); ++i)
      packed[i] = t.dtype == DType::kF16 ? vt::F32ToF16(values[i]) : vt::F32ToBF16(values[i]);
    upload(packed.data());
  }
  std::vector<unsigned char> raw() {
    std::vector<unsigned char> data(bytes);
    auto& b = vt::GetBackend(q.device); b.Copy(q, data.data(), t.data, bytes); b.Synchronize(q);
    return data;
  }
  std::vector<float> floats() {
    const auto data = raw(); std::vector<float> values(bytes / vt::SizeOf(t.dtype));
    for (size_t i = 0; i < values.size(); ++i) {
      if (t.dtype == DType::kF32) std::memcpy(&values[i], data.data() + i * 4, 4);
      else { uint16_t v; std::memcpy(&v, data.data() + i * 2, 2);
             values[i] = t.dtype == DType::kF16 ? vt::F16ToF32(v) : vt::BF16ToF32(v); }
    }
    return values;
  }
};
std::vector<float> Values(size_t n, int salt = 0) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = (static_cast<int>((i * 7 + salt) % 41) - 20) / 16.0f;
  return v;
}
void Close(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance) {
  REQUIRE(actual.size() == expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    CAPTURE(i);
    CAPTURE(actual[i]);
    CAPTURE(expected[i]);
    if (std::isnan(expected[i])) CHECK(std::isnan(actual[i]));
    else if (std::isinf(expected[i])) CHECK(actual[i] == expected[i]);
    else CHECK(std::abs(actual[i] - expected[i]) <= tolerance * (1 + std::abs(expected[i])));
  }
}
constexpr DType floats[] = {DType::kF32, DType::kF16, DType::kBF16};
}

TEST_CASE("XPU greedy rejection matches CPU for ragged GPTQ vocabulary") {
  Queues qs;
  constexpr int32_t vocab = 248320, rows = 10, requests = 4, width = 4;
  const std::vector<int32_t> offsets{0, 2, 6, 7, 10}; // k = 1, 3, 0, 2
  std::vector<float> logits(static_cast<size_t>(rows) * vocab, -5.0f);
  std::vector<int32_t> target(rows);
  for (int32_t row = 0; row < rows; ++row) {
    target[row] = (row * 7919 + 13) % vocab;
    logits[static_cast<size_t>(row) * vocab + target[row]] = 100.0f;
  }
  // Exercise the lowest-id tie rule and the all -inf row.
  logits[target[0] + 1] = 100.0f;
  for (int32_t col = 0; col < vocab; ++col)
    logits[static_cast<size_t>(rows - 1) * vocab + col] =
        -std::numeric_limits<float>::infinity();
  target[rows - 1] = 0;
  std::vector<int32_t> draft(rows, -1);
  for (int32_t req = 0; req < requests; ++req)
    for (int32_t j = 0; j < offsets[req + 1] - offsets[req] - 1; ++j) {
      const int32_t row = offsets[req] + j;
      draft[row + 1] = j == 0 ? target[row] : (target[row] + 1) % vocab;
    }
  std::vector<unsigned char> expected_tokens, expected_counts, expected_argmax;
  for (auto* q : {&qs.cpu, &qs.gpu}) {
    Buffer dl(*q, DType::kF32, {rows, vocab});
    Buffer dd(*q, DType::kI32, {rows});
    Buffer dc(*q, DType::kI32, {requests + 1});
    Buffer ds(*q, DType::kI32, {requests, width});
    Buffer dn(*q, DType::kI32, {requests});
    Buffer da(*q, DType::kI32, {rows});
    dl.put(logits); dd.upload(draft.data()); dc.upload(offsets.data());
    vt::GreedyRejectionSample(*q, ds.t, dn.t, da.t, dl.t, dd.t, dc.t);
    if (q == &qs.cpu) {
      expected_tokens = ds.raw();
      expected_counts = dn.raw();
      expected_argmax = da.raw();
    } else {
      CHECK(ds.raw() == expected_tokens);
      CHECK(dn.raw() == expected_counts);
      CHECK(da.raw() == expected_argmax);
      const std::vector<int32_t> bad_offsets{0, 2, 2, 7, 10};
      dc.upload(bad_offsets.data());
      CHECK_THROWS(vt::GreedyRejectionSample(*q, ds.t, dn.t, da.t,
                                             dl.t, dd.t, dc.t));
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == 0);
}

TEST_CASE("XPU copy and casts: all float pairs, strides, tails and overlapping transpose") {
  Queues qs;
  for (auto src_type : floats) for (auto dst_type : floats) for (int width : {1, 3, 127, 129}) {
    CAPTURE(src_type);
    CAPTURE(dst_type);
    CAPTURE(width);
    std::vector<unsigned char> expected;
    for (auto* q : {&qs.cpu, &qs.gpu}) {
      Buffer src(*q, src_type, {3, width + 4}), dst(*q, dst_type, {3, width + 2});
      auto values = Values(3 * (width + 4));
      values[0] = -0.0f; values[1] = std::numeric_limits<float>::infinity();
      values[2] = std::numeric_limits<float>::quiet_NaN();
      src.put(values); dst.put(std::vector<float>(3 * (width + 2), 17));
      src.t.shape[1] = dst.t.shape[1] = width;
      vt::Copy(*q, dst.t, src.t);
      if (q == &qs.cpu) expected = dst.raw(); else CHECK(dst.raw() == expected);
    }
  }
  for (auto* q : {&qs.cpu, &qs.gpu}) {
    Buffer b(*q, DType::kF32, {3, 3}); b.put({0,1,2,3,4,5,6,7,8});
    auto transposed = b.t; transposed.stride[0] = 1; transposed.stride[1] = 3;
    vt::Copy(*q, transposed, b.t);
    CHECK(b.floats() == std::vector<float>{0,3,6,1,4,7,2,5,8});
    Buffer bf(*q, DType::kBF16, {3, 3}), fp(*q, DType::kF32, {3, 3}), half(*q, DType::kF16, {3, 3});
    vt::CastBf16(*q, bf.t, b.t); vt::CastF32(*q, fp.t, bf.t); vt::CastF16(*q, half.t, bf.t);
    CHECK(fp.floats() == b.floats()); CHECK(half.floats() == b.floats());
  }
}

TEST_CASE("XPU Add, SiLU, MoeSiluMul and sigmoid: input rounding and aliases") {
  Queues qs;
  for (auto dtype : floats) for (int width : {1, 129}) {
    std::vector<std::vector<float>> expected;
    for (auto* q : {&qs.cpu, &qs.gpu}) {
      Buffer a(*q, dtype, {2, width}), b(*q, dtype, {width});
      Buffer result(*q, DType::kBF16, {2, width}), gateup(*q, dtype, {2, 2 * width});
      Buffer gate(*q, DType::kF32, {2, width});
      a.put(Values(2 * width)); b.put(Values(width, 3)); gateup.put(Values(4 * width, 5)); gate.put(Values(2 * width, 9));
      std::vector<std::vector<float>> observed;
      vt::Add(*q, result.t, a.t, b.t); observed.push_back(result.floats());
      vt::MoeSiluMul(*q, result.t, a.t, a.t); observed.push_back(result.floats());
      vt::SiluAndMul(*q, result.t, gateup.t); observed.push_back(result.floats());
      vt::SigmoidGateBf16(*q, result.t, gate.t, gate.t); observed.push_back(result.floats());
      vt::MoeSiluMul(*q, result.t, result.t, result.t); observed.push_back(result.floats());
      vt::Add(*q, result.t, result.t, result.t); observed.push_back(result.floats());
      if (q == &qs.cpu) {
        // Oracle is the non-overlapping CPU result. Its row-parallel kernel
        // does not promise snapshot semantics for a packed, overlapping output.
        observed.push_back(observed[2]);
      } else {
        auto alias = gateup.t; alias.dtype = DType::kBF16;
        alias.shape[1] = width; alias.stride[0] = width;
        vt::SiluAndMul(*q, alias, gateup.t);
        vt::Copy(*q, result.t, alias); observed.push_back(result.floats());
      }
      if (q == &qs.cpu) expected = observed;
      else for (size_t i = 0; i < observed.size(); ++i) Close(observed[i], expected[i], 0.008f);
    }
  }
}

TEST_CASE("XPU FP16 SwiGLU: real half midpoint packed aliases and strided rejection") {
  Queues qs;
  // Pinned original block20/token48/column16742: the SiLU is materialized
  // as FP16 before multiplying. Ordinary approximate division selects the
  // adjacent half and changes the full MLP's quantized down-projection row.
  constexpr float gate = -2.724609375f, up = -0.346923828125f;
  constexpr float expected = 0.058135986328125f;
  for (int width : {1, 129}) for (int padding : {0, 5}) for (bool alias : {false, true}) {
    CAPTURE(width);
    CAPTURE(padding);
    CAPTURE(alias);
    const int input_stride = 2 * width + padding, output_stride = width + padding;
    Buffer input(qs.gpu, DType::kF16, {3, input_stride});
    Buffer output(qs.gpu, DType::kF16, {3, output_stride});
    std::vector<float> values(3 * input_stride, std::numeric_limits<float>::quiet_NaN());
    for (int row = 0; row < 3; ++row) for (int col = 0; col < width; ++col) {
      values[row * input_stride + col] = gate;
      values[row * input_stride + width + col] = up;
    }
    input.put(values); output.put(std::vector<float>(3 * output_stride, 17));
    const auto input_before = input.raw();
    auto wanted = alias ? input_before : output.raw();
    const int result_stride = alias ? width : output_stride;
    const uint16_t half = vt::F32ToF16(expected);
    for (int row = 0; row < 3; ++row) for (int col = 0; col < width; ++col)
      std::memcpy(wanted.data() + (row * result_stride + col) * 2, &half, 2);
    auto source = input.t; source.shape[1] = 2 * width;
    auto target = alias ? input.t : output.t; target.shape[1] = width;
    if (alias) target.stride[0] = width;
    if (padding) {
      const auto output_before = output.raw();
      CHECK_THROWS_WITH_AS(vt::SiluAndMul(qs.gpu, target, source),
                          doctest::Contains("contiguous required"), std::runtime_error);
      CHECK(input.raw() == input_before);
      CHECK(output.raw() == output_before);
      continue;
    }
    vt::SiluAndMul(qs.gpu, target, source);
    CHECK((alias ? input.raw() : output.raw()) == wanted);
    if (!alias) CHECK(input.raw() == input_before);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

namespace {
struct P3SiluEnv {
  bool present = std::getenv("VT_XPU_SILU_FP16_TYPED") != nullptr;
  std::string value = present ? std::getenv("VT_XPU_SILU_FP16_TYPED") : "";
  ~P3SiluEnv() {
    if (present) setenv("VT_XPU_SILU_FP16_TYPED", value.c_str(), 1);
    else unsetenv("VT_XPU_SILU_FP16_TYPED");
  }
  void Select(bool typed) { REQUIRE(setenv("VT_XPU_SILU_FP16_TYPED", typed ? "1" : "0", 1) == 0); }
};
struct P7SiluTableEnv {
  bool present = std::getenv("VT_XPU_SILU_FP16_TABLE") != nullptr;
  std::string value = present ? std::getenv("VT_XPU_SILU_FP16_TABLE") : "";
  ~P7SiluTableEnv() {
    if (present) setenv("VT_XPU_SILU_FP16_TABLE", value.c_str(), 1);
    else unsetenv("VT_XPU_SILU_FP16_TABLE");
  }
  void Select(bool table) { REQUIRE(setenv("VT_XPU_SILU_FP16_TABLE", table ? "1" : "0", 1) == 0); }
};
void P3SameBits(const std::vector<unsigned char>& actual, const std::vector<unsigned char>& expected) {
  REQUIRE(actual.size() == expected.size());
  const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
  const auto first_byte = mismatch.first - actual.begin();
  CAPTURE(first_byte);
  REQUIRE(mismatch.first == actual.end());
}
void P3SameFp16Behavior(const std::vector<unsigned char>& actual,
                        const std::vector<unsigned char>& expected) {
  REQUIRE(actual.size() == expected.size());
  size_t mismatches = 0, nan_payloads = 0;
  for (size_t i = 0; i < actual.size(); i += 2) {
    uint16_t a, b;
    std::memcpy(&a, actual.data() + i, 2); std::memcpy(&b, expected.data() + i, 2);
    if (a == b) continue;
    const auto nan = [](uint16_t bits) { return (bits & 0x7c00) == 0x7c00 && (bits & 0x3ff) != 0; };
    // IEEE arithmetic does not specify which payload wins when both operands
    // are NaNs. Finite values, +/-Inf and signed zeros remain bit-exact; every
    // original NaN must remain a NaN, with no finite/nonfinite conversion.
    if (nan(a) && nan(b)) ++nan_payloads; else ++mismatches;
  }
  CAPTURE(mismatches);
  REQUIRE(mismatches == 0);
  if (nan_payloads) std::cout << "P3_SILU_NAN_PAYLOAD_ONLY differences=" << nan_payloads << std::endl;
}
}

TEST_CASE("XPU P3 FP16 SiLU: exhaustive gate bits preserve generic materialization and graph replay") {
  Queues qs; P3SiluEnv route;
  P7SiluTableEnv table_mode; table_mode.Select(false);
  constexpr int rows = 512, width = 128, n = 65536;
  Buffer input(qs.gpu, DType::kF16, {rows, 2 * width});
  Buffer generic(qs.gpu, DType::kF16, {rows, width});
  Buffer typed(qs.gpu, DType::kF16, {rows, width});
  std::vector<uint16_t> raw(2 * n);
  const auto fill = [&](uint16_t up, int shift = 0) {
    for (int i = 0; i < n; ++i) {
      const int offset = (i / width) * (2 * width) + i % width;
      raw[offset] = uint16_t(i + shift); raw[offset + width] = up;
    }
    input.upload(raw.data());
  };
  // Every gate bit pattern, including both signed zeros, subnormals, infinities
  // and all NaN payloads. Ups include the real midpoint, extremes, zero and
  // exceptional values. Reference is the preserved original GPU expression,
  // not a host approximation or a duplicate typed kernel in the test.
  for (uint16_t up : {uint16_t(0x3c00), uint16_t(0x0000), uint16_t(0x8000),
                     uint16_t(0x0001), uint16_t(0x7bff), uint16_t(0xfbff),
                     vt::F32ToF16(-0.346923828125f), uint16_t(0x7c00),
                     uint16_t(0xfc00), uint16_t(0x7e35)}) {
    CAPTURE(up);
    fill(up); const auto before = input.raw();
    route.Select(false); vt::SiluAndMul(qs.gpu, generic.t, input.t);
    const auto expected = generic.raw();
    route.Select(true); vt::SiluAndMul(qs.gpu, typed.t, input.t);
    P3SameFp16Behavior(typed.raw(), expected); CHECK(input.raw() == before);
  }
  // No new persistent table or scratch: a captured typed kernel re-reads fresh
  // gate/up bytes. Graph owner dies before any baked buffers.
  if (vt::GetBackend(qs.gpu.device).SupportsGraphCapture()) {
    route.Select(true); vt::SiluAndMul(qs.gpu, typed.t, input.t);
    vt::GetBackend(qs.gpu.device).Synchronize(qs.gpu);
    vt::BreakableGraph graph;
    {
      vt::GraphCaptureScope scope(vt::GetBackend(qs.gpu.device), qs.gpu, graph,
                                  vt::GraphCaptureMode::kFull);
      vt::SiluAndMul(qs.gpu, typed.t, input.t);
    }
    REQUIRE(graph.captured());
    for (int shift : {0, 17, 4097}) {
      fill(vt::F32ToF16(-0.346923828125f), shift);
      route.Select(false); vt::SiluAndMul(qs.gpu, generic.t, input.t);
      const auto expected = generic.raw();
      route.Select(true); graph.Replay(qs.gpu); P3SameBits(typed.raw(), expected);
    }
    graph.Reset(); CHECK(vt::xpu::GetMemoryInfo().graph_device_bytes == 0);
    std::cout << "P3_SILU_GRAPH captured=1 fresh_input_replays=3 retired_graph_bytes=0" << std::endl;
  }
  CHECK(vt::GetReferenceTierHits() == 0);
  std::cout << "P3_SILU_EXHAUSTIVE gate_patterns=65536 up_patterns=10 products=655360 finite_inf_signed_zero_exact=1 nan_classification_exact=1" << std::endl;
}

TEST_CASE("XPU P7 FP16 SiLU table: exhaustive bytes queue joins and graph lifetime") {
  Queues qs; P3SiluEnv typed;
  P7SiluTableEnv table_mode;
  constexpr int rows = 64, width = 1024, n = rows * width;
  Buffer input(qs.gpu, DType::kF16, {rows, 2 * width});
  Buffer direct(qs.gpu, DType::kF16, {rows, width});
  Buffer lookup(qs.gpu, DType::kF16, {rows, width});
  Buffer joined(qs.gpu, DType::kF16, {rows, width});
  auto other = vt::CreateQueue(qs.gpu.device);
  lookup.upload(std::vector<uint16_t>(n, 0x7e35).data());
  std::vector<uint16_t> data(2 * n);
  const auto fill = [&](uint16_t up, int shift = 0) {
    for (int i = 0; i < n; ++i) {
      data[(i / width) * 2 * width + i % width] = uint16_t(i + shift);
      data[(i / width) * 2 * width + width + i % width] = up;
    }
    input.upload(data.data());
  };
  typed.Select(true);
  // First capture cannot allocate/initialize the table or partially write out.
  fill(0x3c00);
  if (vt::GetBackend(qs.gpu.device).SupportsGraphCapture() &&
      vt::xpu::GetMemoryInfo().fp16_silu_table_bytes == 0) {
    setenv("VT_XPU_SILU_FP16_TABLE", "1", 1);
    const auto before = lookup.raw();
    vt::BreakableGraph graph;
    CHECK_THROWS_WITH_AS(([&] {
      vt::GraphCaptureScope scope(vt::GetBackend(qs.gpu.device), qs.gpu, graph,
                                  vt::GraphCaptureMode::kFull);
      vt::SiluAndMul(qs.gpu, lookup.t, input.t);
    }()), doctest::Contains("XPU FP16 SiLU table must be warmed on the capture queue"), std::runtime_error);
    P3SameBits(lookup.raw(), before);
    CHECK(vt::xpu::GetMemoryInfo().fp16_silu_table_bytes == 0);
    CHECK(vt::xpu::GetMemoryInfo().graph_device_bytes == 0);
  }
  for (uint16_t up : {uint16_t(0x3c00), uint16_t(0), uint16_t(0x8000),
                     uint16_t(1), uint16_t(0x7bff), uint16_t(0xfbff),
                     vt::F32ToF16(-0.346923828125f), uint16_t(0x7c00),
                     uint16_t(0xfc00), uint16_t(0x7e35)}) {
    CAPTURE(up); fill(up); const auto before = input.raw();
    setenv("VT_XPU_SILU_FP16_TABLE", "0", 1);
    vt::SiluAndMul(qs.gpu, direct.t, input.t); const auto expected = direct.raw();
    setenv("VT_XPU_SILU_FP16_TABLE", "1", 1);
    vt::SiluAndMul(qs.gpu, lookup.t, input.t);
    if (up == 0x3c00) {
      // Input is ready, but no completion wait occurs between initialization on
      // the producer and this first reader. The backend must join its event.
      vt::SiluAndMul(other, joined.t, input.t);
      vt::GetBackend(other.device).Synchronize(other);
      P3SameFp16Behavior(joined.raw(), expected);
    }
    P3SameFp16Behavior(lookup.raw(), expected);
    CHECK(input.raw() == before);
  }
  CHECK(vt::xpu::GetMemoryInfo().fp16_silu_table_bytes == 131072);
  // Reuse immutable storage from another queue without a host readiness wait.
  fill(vt::F32ToF16(-0.346923828125f), 17);
  setenv("VT_XPU_SILU_FP16_TABLE", "0", 1);
  vt::SiluAndMul(qs.gpu, direct.t, input.t); const auto expected = direct.raw();
  setenv("VT_XPU_SILU_FP16_TABLE", "1", 1);
  vt::SiluAndMul(other, lookup.t, input.t);
  vt::GetBackend(other.device).Synchronize(other);
  P3SameBits(lookup.raw(), expected);
  if (vt::GetBackend(other.device).SupportsGraphCapture()) {
    vt::BreakableGraph graph;
    {
      vt::GraphCaptureScope scope(vt::GetBackend(other.device), other, graph,
                                  vt::GraphCaptureMode::kFull);
      vt::SiluAndMul(other, lookup.t, input.t);
    }
    REQUIRE(graph.captured());
    for (int shift : {0, 4097, 33521}) {
      fill(vt::F32ToF16(-0.346923828125f), shift);
      setenv("VT_XPU_SILU_FP16_TABLE", "0", 1);
      vt::SiluAndMul(qs.gpu, direct.t, input.t); const auto wanted = direct.raw();
      graph.Replay(other); vt::GetBackend(other.device).Synchronize(other);
      P3SameBits(lookup.raw(), wanted);
    }
    graph.Reset(); CHECK(vt::xpu::GetMemoryInfo().graph_device_bytes == 0);
  }
  vt::DestroyQueue(other);
  // A new queue must acquire readiness itself even if an old native queue's
  // address is recycled. Its first capture cannot inherit the old join.
  other = vt::CreateQueue(qs.gpu.device);
  setenv("VT_XPU_SILU_FP16_TABLE", "1", 1);
  const auto wanted = direct.raw(), preserved = lookup.raw();
  if (vt::GetBackend(other.device).SupportsGraphCapture()) {
    vt::BreakableGraph cold;
    CHECK_THROWS_WITH_AS(([&] {
      vt::GraphCaptureScope scope(vt::GetBackend(other.device), other, cold,
                                  vt::GraphCaptureMode::kFull);
      vt::SiluAndMul(other, lookup.t, input.t);
    }()), doctest::Contains("XPU FP16 SiLU table must be warmed on the capture queue"), std::runtime_error);
    P3SameBits(lookup.raw(), preserved);
    CHECK(vt::xpu::GetMemoryInfo().graph_device_bytes == 0);
  }
  vt::SiluAndMul(other, lookup.t, input.t);
  vt::GetBackend(other.device).Synchronize(other);
  P3SameBits(lookup.raw(), wanted);
  vt::DestroyQueue(other);
  CHECK(vt::xpu::GetMemoryInfo().fp16_silu_table_bytes == 131072);
  CHECK(vt::GetReferenceTierHits() == 0);
  std::cout << "P7_SILU_TABLE_EXHAUSTIVE gate_patterns=65536 up_patterns=10 table_bytes=131072 other_queue=1" << std::endl;
}

TEST_CASE("XPU P3 FP16 SiLU: threshold rows tail guards and shifted aliases") {
  Queues qs; P3SiluEnv route;
  constexpr int guard = 32;
  const uint16_t gate = vt::F32ToF16(-2.724609375f), up = vt::F32ToF16(-0.346923828125f);
  const uint16_t expected = vt::F32ToF16(0.058135986328125f);
  for (int rows : {1, 3, 4, 16, 63, 64, 65}) for (int width : {1, 129})
    for (int alias_offset : {-1, 0, 17}) {
      CAPTURE(rows);
      CAPTURE(width);
      CAPTURE(alias_offset);
      Buffer input(qs.gpu, DType::kF16, {2 * guard + rows * 2 * width});
      Buffer result(qs.gpu, DType::kF16, {2 * guard + rows * width});
      std::vector<uint16_t> raw(input.bytes / 2, 0x7e35), out(result.bytes / 2, 0x7e35);
      for (int row = 0; row < rows; ++row) for (int col = 0; col < width; ++col) {
        raw[guard + row * 2 * width + col] = gate;
        raw[guard + row * 2 * width + width + col] = up;
      }
      input.upload(raw.data()); result.upload(out.data());
      auto source = vt::Tensor::Contiguous(static_cast<uint16_t*>(input.t.data) + guard,
          DType::kF16, qs.gpu.device, {rows, 2 * width});
      const bool alias = alias_offset >= 0;
      const int offset = guard + (alias ? alias_offset : 0);
      auto target = vt::Tensor::Contiguous(static_cast<uint16_t*>(alias ? input.t.data : result.t.data) + offset,
          DType::kF16, qs.gpu.device, {rows, width});
      auto wanted = alias ? raw : out;
      std::fill(wanted.begin() + offset, wanted.begin() + offset + rows * width, expected);
      std::vector<unsigned char> bits(wanted.size() * 2);
      std::memcpy(bits.data(), wanted.data(), bits.size());
      route.Select(true); vt::SiluAndMul(qs.gpu, target, source);
      P3SameBits(alias ? input.raw() : result.raw(), bits);
      if (!alias) {
        std::vector<unsigned char> input_bits(raw.size() * 2);
        std::memcpy(input_bits.data(), raw.data(), input_bits.size());
        CHECK(input.raw() == input_bits);
      }
    }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU P3 FP16 SiLU: real MLP operands complete operator gain") {
  const char* fixture = std::getenv("VT_B70_EXL3_SILU_OPERANDS");
  if (!fixture) { MESSAGE("Set VT_B70_EXL3_SILU_OPERANDS for the bounded real-row benchmark"); return; }
  Queues qs; P3SiluEnv route; P7SiluTableEnv table_mode;
  const auto original = vllm::SafetensorsFile::Open(fixture);
  const auto& gu = original.Get("p128_l20_detail_gate_up");
  const auto& act = original.Get("p128_l20_detail_swiglu");
  REQUIRE(gu.dtype == "F16"); REQUIRE(act.dtype == "F16");
  REQUIRE(gu.shape == std::vector<int64_t>{128, 34816});
  REQUIRE(act.shape == std::vector<int64_t>{128, 17408});
  constexpr int width = 17408, guard = 32;
  const bool profile = std::getenv("VT_XPU_PROFILE") && std::string(std::getenv("VT_XPU_PROFILE")) == "1";
  for (int rows : {4, 128, 896, 1600}) {
    CAPTURE(rows);
    Buffer input(qs.gpu, DType::kF16, {2 * guard + rows * 2 * width});
    Buffer result(qs.gpu, DType::kF16, {2 * guard + rows * width});
    std::vector<uint16_t> data(input.bytes / 2, 0x7e35), expected(result.bytes / 2, 0x7e35);
    for (int row = 0; row < rows; ++row) {
      // Cyclic frozen real P128 rows, not a fresh M1600 model trajectory.
      std::memcpy(data.data() + guard + row * 2 * width,
                  gu.data + (row % 128) * 2 * width * 2, 2 * width * 2);
      std::memcpy(expected.data() + guard + row * width,
                  act.data + (row % 128) * width * 2, width * 2);
    }
    input.upload(data.data()); result.upload(expected.data());
    auto source = vt::Tensor::Contiguous(static_cast<uint16_t*>(input.t.data) + guard,
        DType::kF16, qs.gpu.device, {rows, 2 * width});
    auto target = vt::Tensor::Contiguous(static_cast<uint16_t*>(result.t.data) + guard,
        DType::kF16, qs.gpu.device, {rows, width});
    const auto wanted = result.raw(), before = input.raw();
    for (int variant : {0, 1, 2}) {
      const bool typed = variant != 0, table = variant == 2;
      route.Select(typed); table_mode.Select(table);
      for (int warm = 0; warm < 2; ++warm) vt::SiluAndMul(qs.gpu, target, source);
      vt::GetBackend(qs.gpu.device).Synchronize(qs.gpu);
      (void)vt::xpu::DrainProfileEvents();
      std::vector<double> times;
      for (int sample = 0; sample < 3; ++sample) {
        const auto begin = std::chrono::steady_clock::now();
        vt::SiluAndMul(qs.gpu, target, source);
        vt::GetBackend(qs.gpu.device).Synchronize(qs.gpu);
        times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
      }
      P3SameBits(result.raw(), wanted); CHECK(input.raw() == before);
      nlohmann::json event = {{"rows", rows}, {"width", width}, {"typed_requested", typed},
          {"table_requested", table}, {"table_bytes", vt::xpu::GetMemoryInfo().fp16_silu_table_bytes},
          {"profiled", profile}, {"complete_operator_wall_ms", times}, {"exact_original_bytes", true}};
      event["device_events"] = nlohmann::json::array();
      for (const auto& e : vt::xpu::DrainProfileEvents())
        if (e.stage.find("silu_and_mul") == 0)
          event["device_events"].push_back({{"stage", e.stage}, {"ms", (e.end_ns - e.start_ns) / 1e6}});
      std::cout << "P3_SILU_BENCH " << event.dump() << std::endl;
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU RMSNorm: weight versus 1+weight, residual rounding and in-place aliases") {
  Queues qs;
  for (auto dtype : floats) for (bool gemma : {false, true}) for (int width : {1, 129, 5120}) {
    CAPTURE(dtype);
    CAPTURE(gemma);
    CAPTURE(width);
    std::vector<float> expected, expected_residual;
    for (auto* q : {&qs.cpu, &qs.gpu}) {
      Buffer x(*q, dtype, {2, width}), w(*q, dtype, {width});
      Buffer out(*q, DType::kF32, {2, width}), residual(*q, DType::kBF16, {2, width});
      x.put(Values(2 * width)); w.put(Values(width, 4)); residual.put(Values(2 * width, 11));
      vt::RmsNorm(*q, out.t, x.t, w.t, {1e-6f, gemma}, &residual.t);
      if (q == &qs.cpu) { expected = out.floats(); expected_residual = residual.floats(); }
      else { Close(out.floats(), expected, 2e-5f); CHECK(residual.floats() == expected_residual); }
    }
  }
  for (bool with_residual : {false, true}) {
    std::vector<float> expected;
    for (auto* q : {&qs.cpu, &qs.gpu}) {
      Buffer x(*q, DType::kBF16, {3, 129}), w(*q, DType::kBF16, {129});
      x.put(Values(3 * 129)); w.put(Values(129));
      vt::RmsNorm(*q, x.t, x.t, w.t, {1e-6f, true}, with_residual ? &x.t : nullptr);
      if (q == &qs.cpu) expected = x.floats(); else Close(x.floats(), expected, 0.008f);
    }
  }
}

TEST_CASE("XPU gather/scatter and embedding: padded rows, duplicate indices, invalid indices") {
  Queues qs;
  for (auto dtype : floats) {
    std::vector<float> expected_gather, expected_scatter, expected_embed;
    for (auto* q : {&qs.cpu, &qs.gpu}) {
      Buffer base(*q, dtype, {4, 132}), result(*q, dtype, {3, 129});
      Buffer idx(*q, DType::kI32, {3}), embed(*q, DType::kF32, {3, 132});
      const int32_t indices[] = {2, 0, 2}; idx.upload(indices); base.put(Values(4 * 132));
      vt::Embedding(*q, embed.t, base.t, idx.t);
      base.t.shape[1] = 129;
      vt::IndexSelect(*q, result.t, base.t, idx.t);
      if (q == &qs.cpu) { expected_gather = result.floats(); expected_embed = embed.floats(); }
      else { CHECK(result.floats() == expected_gather); CHECK(embed.floats() == expected_embed); }
      result.put(Values(3 * 129, 8)); vt::IndexCopy(*q, base.t, result.t, idx.t);
      if (q == &qs.cpu) expected_scatter = base.floats(); else CHECK(base.floats() == expected_scatter);
      const int32_t invalid[] = {0, -1, 4}; idx.upload(invalid);
      CHECK_THROWS(vt::IndexSelect(*q, result.t, base.t, idx.t));
    }
  }
  // Equal-dtype copies of I64 indices must never round through float.
  for (auto* q : {&qs.cpu, &qs.gpu}) {
    Buffer base(*q, DType::kI64, {3, 1}), idx(*q, DType::kI32, {3}), out(*q, DType::kI64, {3, 1});
    const int64_t values[] = {INT64_MAX, INT64_MIN, 16777217}; const int32_t indices[] = {2, 1, 0};
    base.upload(values); idx.upload(indices); vt::IndexSelect(*q, out.t, base.t, idx.t);
    const int64_t expected[] = {16777217, INT64_MIN, INT64_MAX};
    auto raw = out.raw(); CHECK(std::memcmp(raw.data(), expected, sizeof(expected)) == 0);
  }
}

TEST_CASE("XPU IndexCopy: wide ragged duplicate and aliased rows remain last-write-wins") {
  auto q = vt::CreateQueue({vt::DeviceType::kXPU, 0});
  for (const auto geometry : {std::pair{37, 129}, std::pair{129, 257},
                              std::pair{1600, 5120}, std::pair{1600, 6144}}) {
    const int rows = geometry.first, width = geometry.second, destination_rows = rows + 7;
    CAPTURE(rows);
    CAPTURE(width);
    for (const bool alias : {false, true}) {
      CAPTURE(alias);
      Buffer source(q, DType::kF16, {rows, width});
      Buffer destination(q, DType::kF16, {destination_rows, width});
      if (width == 6144) {
        // Actual mixed GDN core layout, with unchanged contiguous bytes.
        source.t = vt::Tensor::Contiguous(source.t.data, DType::kF16, q.device, {rows, 48, 128});
        destination.t = vt::Tensor::Contiguous(destination.t.data, DType::kF16, q.device,
                                              {destination_rows, 48, 128});
      }
      Buffer indices(q, DType::kI32, {rows});
      std::vector<uint16_t> input(size_t(rows) * width), initial(size_t(destination_rows) * width);
      std::vector<int32_t> ids(rows);
      for (int r = 0; r < destination_rows; ++r)
        for (int c = 0; c < width; ++c) {
          initial[size_t(r) * width + c] = uint16_t(0x4000 + (r * 11 + c) % 2048);
          if (r < rows) input[size_t(r) * width + c] = uint16_t(0x3000 + (r * 7 + c) % 2048);
        }
      for (int r = 0; r < rows; ++r) ids[r] = r % 7 == 0 ? 3 : (r * 13 + 7) % destination_rows;
      source.upload(input.data()); destination.upload(initial.data()); indices.upload(ids.data());
      auto in = alias ? destination.t : source.t;
      in.shape[0] = rows;
      const auto& snapshot = alias ? initial : input;
      auto expected = initial;
      for (int r = 0; r < rows; ++r)
        std::copy_n(snapshot.begin() + size_t(r) * width, width,
                    expected.begin() + size_t(ids[r]) * width);
      vt::IndexCopy(q, destination.t, in, indices.t);
      auto got = destination.raw();
      REQUIRE(got.size() == expected.size() * sizeof(uint16_t));
      CHECK(std::memcmp(got.data(), expected.data(), got.size()) == 0);
      if (!alias) {
        const auto unchanged = source.raw();
        CHECK(std::memcmp(unchanged.data(), input.data(), unchanged.size()) == 0);
        vt::BreakableGraph graph;
        {
          vt::GraphCaptureScope scope(vt::GetBackend(q.device), q, graph,
                                      vt::GraphCaptureMode::kFull);
          vt::IndexCopy(q, destination.t, in, indices.t);
        }
        REQUIRE(graph.captured());
        graph.Replay(q);
        CHECK(destination.raw() == got);
        // New duplicate routing must be recomputed on replay, not cached.
        for (int r = 0; r < rows; ++r)
          ids[r] = r % 11 == 0 ? 5 : (r * 17 + 13) % destination_rows;
        indices.upload(ids.data());
        for (int r = 0; r < rows; ++r)
          std::copy_n(input.begin() + size_t(r) * width, width,
                      expected.begin() + size_t(ids[r]) * width);
        graph.Replay(q);
        got = destination.raw();
        CHECK(std::memcmp(got.data(), expected.data(), got.size()) == 0);
        const int32_t valid = ids[0];
        ids[0] = -1; indices.upload(ids.data());
        CHECK_THROWS(graph.Replay(q));
        CHECK(destination.raw() == got);
        ids[0] = valid; indices.upload(ids.data());
      }
      for (const int32_t invalid : {-1, destination_rows}) {
        ids[0] = invalid; indices.upload(ids.data());
        CHECK_THROWS(vt::IndexCopy(q, destination.t, in, indices.t));
        CHECK(destination.raw() == got);
      }
    }
  }
  vt::DestroyQueue(q);
}

TEST_CASE("XPU Matmul and MatmulBT: mixed dtypes, strided activation and real BA dimensions") {
  Queues qs;
  for (auto a_type : floats) for (auto b_type : floats) for (bool transpose : {false, true}) {
    std::vector<float> expected;
    for (auto* q : {&qs.cpu, &qs.gpu}) {
      const int k = 13, n = 9;
      Buffer a(*q, a_type, {3, transpose ? k + 4 : k});
      Buffer b(*q, b_type, {transpose ? n : k, transpose ? k : n});
      Buffer out(*q, DType::kF32, {3, n});
      a.put(Values(a.t.Numel())); b.put(Values(b.t.Numel(), 5)); a.t.shape[1] = k;
      if (transpose) vt::MatmulBT(*q, out.t, a.t, b.t); else vt::Matmul(*q, out.t, a.t, b.t);
      if (q == &qs.cpu) expected = out.floats(); else Close(out.floats(), expected, 1e-5f);
    }
  }
  std::vector<float> expected;
  for (auto* q : {&qs.cpu, &qs.gpu}) {
    Buffer a(*q, DType::kBF16, {1, 5120}), b(*q, DType::kF16, {48, 5120});
    Buffer out(*q, DType::kBF16, {1, 48}); a.put(Values(5120)); b.put(Values(48 * 5120));
    vt::MatmulBT(*q, out.t, a.t, b.t);
    if (q == &qs.cpu) expected = out.floats(); else CHECK(out.floats() == expected);
  }
}

TEST_CASE("XPU MTP matmuls preserve FP16 activations with BF16 weights") {
  Queues qs;
  for (bool transpose : {false, true}) {
    constexpr int m = 2, k = 13, n = 9;
    const auto av = Values(m * k, 3);
    const auto bv = Values(k * n, 7);
    Buffer cpu_a(qs.cpu, DType::kF16, {m, k});
    Buffer gpu_a(qs.gpu, DType::kF16, {m, k});
    Buffer cpu_b(qs.cpu, DType::kBF16,
                 {transpose ? n : k, transpose ? k : n});
    Buffer gpu_b(qs.gpu, DType::kBF16,
                 {transpose ? n : k, transpose ? k : n});
    Buffer expected(qs.cpu, DType::kF32, {m, n});
    Buffer actual(qs.gpu, DType::kF16, {m, n});
    cpu_a.put(av); gpu_a.put(av); cpu_b.put(bv); gpu_b.put(bv);
    if (transpose) {
      vt::MatmulBT(qs.cpu, expected.t, cpu_a.t, cpu_b.t);
      vt::MatmulBT(qs.gpu, actual.t, gpu_a.t, gpu_b.t);
    } else {
      vt::Matmul(qs.cpu, expected.t, cpu_a.t, cpu_b.t);
      vt::Matmul(qs.gpu, actual.t, gpu_a.t, gpu_b.t);
    }
    const auto want = expected.floats();
    const auto got = actual.floats();
    REQUIRE(got.size() == want.size());
    for (size_t i = 0; i < got.size(); ++i) {
      CAPTURE(transpose);
      CAPTURE(i);
      CHECK(got[i] == vt::F16ToF32(vt::F32ToF16(want[i])));
    }
  }
}

TEST_CASE("XPU profile: RMSNorm and BA MatmulBT carry device timestamps"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE"))) {
  Queues qs;
  Buffer x(qs.gpu, DType::kBF16, {1, 5120}), w(qs.gpu, DType::kBF16, {5120});
  Buffer norm(qs.gpu, DType::kBF16, {1, 5120});
  Buffer ba(qs.gpu, DType::kBF16, {96, 5120}), out(qs.gpu, DType::kF32, {1, 96});
  x.put(Values(5120)); w.put(Values(5120, 4)); ba.put(Values(96 * 5120, 5));
  (void)vt::xpu::DrainProfileEvents();
  vt::RmsNorm(qs.gpu, norm.t, x.t, w.t, {1e-6f, true});
  vt::MatmulBT(qs.gpu, out.t, norm.t, ba.t);
  CHECK(out.floats().size() == 96);
  const auto records = vt::xpu::DrainProfileEvents();
  REQUIRE(records.size() == 3);
  CHECK(records[0].stage == "rms_norm");
  CHECK(records[1].stage == "matmul_bt");
  CHECK(records[2].stage == "staged_d2h_copy");
  for (const auto& record : records) {
    CHECK(record.matrix.empty());
    CHECK(record.queue_id == qs.gpu.id);
    CHECK(record.submit_ns > 0);
    CHECK(record.end_ns >= record.start_ns);
  }
}

TEST_CASE("XPU greedy: lowest tie, NaN/Inf contract, tails and production token readback") {
  Queues qs;
  for (int vocab : {1, 129, 248320}) {
    auto values = Values(6 * vocab);
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    values[0] = nan;
    for (int j = 0; j < vocab; ++j) { values[vocab + j] = -inf; values[2 * vocab + j] = inf; }
    if (vocab > 1) { values[3 * vocab + 1] = nan; values[4 * vocab] = 9; values[5 * vocab - 1] = 9; }
    std::vector<unsigned char> expected;
    for (auto* q : {&qs.cpu, &qs.gpu}) {
      Buffer logits(*q, DType::kF32, {6, vocab}), ids(*q, DType::kI64, {6}); logits.put(values);
      vt::GreedyArgmax(*q, ids.t, logits.t);
      if (q == &qs.cpu) expected = ids.raw(); else CHECK(ids.raw() == expected);
      vllm::v1::Sampler sampler; vllm::v1::SamplingMetadata metadata;
      metadata.all_greedy = true; metadata.all_random = false; metadata.no_penalties = true;
      const auto output = sampler.forward(*q, logits.t, metadata);
      REQUIRE(output.sampled_token_ids.size() == 6);
      for (size_t row = 0; row < 6; ++row) {
        int64_t id; std::memcpy(&id, expected.data() + row * sizeof(id), sizeof(id));
        REQUIRE(output.sampled_token_ids[row].size() == 1);
        CHECK(output.sampled_token_ids[row][0] == id);
      }
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
  auto& platform = vllm::platforms::GetPlatform(vt::DeviceType::kXPU);
  CHECK(platform.needs_weight_staging()); CHECK(platform.supports_graph_capture());
  CHECK_FALSE(platform.support_static_graph_mode());
  CHECK_FALSE(platform.get_device_capability().present());
}
