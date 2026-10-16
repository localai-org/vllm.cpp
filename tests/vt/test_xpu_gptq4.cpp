#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gptq4_weight.h"
#include "vllm/model_executor/models/dense_gptq4_linear.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/xpu.h"
#include "vt/xpu/xpu_common.h"
#include "vt/xpu/xpu_gptq4.h"

namespace {

struct QueueOwner {
  vt::Queue queue = vt::CreateQueue({vt::DeviceType::kXPU, 0});
  ~QueueOwner() { vt::DestroyQueue(queue); }
};

struct Buffer {
  vt::Queue& queue;
  vt::Tensor tensor;
  size_t bytes;

  Buffer(vt::Queue& q, vt::DType dtype, std::initializer_list<int64_t> shape)
      : queue(q), tensor(vt::Tensor::Contiguous(nullptr, dtype, q.device, shape)),
        bytes(tensor.Bytes()) {
    tensor.data = vt::Alloc(q.device, bytes);
  }
  ~Buffer() { vt::Free(queue.device, tensor.data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  void Upload(const void* data, size_t length) {
    REQUIRE(length == bytes);
    vt::GetBackend(queue.device).Copy(queue, tensor.data, data, length);
    vt::GetBackend(queue.device).Synchronize(queue);
  }
  std::vector<uint8_t> Read() {
    std::vector<uint8_t> result(bytes);
    vt::GetBackend(queue.device).Copy(queue, result.data(), tensor.data, bytes);
    vt::GetBackend(queue.device).Synchronize(queue);
    return result;
  }
};

float ReadHalf(const uint8_t* bytes, size_t index) {
  uint16_t bits;
  std::memcpy(&bits, bytes + index * sizeof(bits), sizeof(bits));
  return vt::F16ToF32(bits);
}

void RequireClose(const std::vector<uint8_t>& actual,
                  const std::vector<float>& expected) {
  REQUIRE(actual.size() == expected.size() * sizeof(uint16_t));
  for (size_t i = 0; i < expected.size(); ++i) {
    CAPTURE(i);
    CHECK(std::abs(ReadHalf(actual.data(), i) - expected[i]) <=
          0.02f + 0.01f * std::abs(expected[i]));
  }
}

vllm::Gptq4Weight ConstantPacked(int k, int n, uint32_t words,
                                  float scale) {
  vllm::Gptq4Weight weight;
  weight.k = k;
  weight.n = n;
  weight.qweight = vllm::dense_loaders::MakeOwned(vt::DType::kI32,
                                                   {n, k / 8});
  weight.scales = vllm::dense_loaders::MakeOwned(vt::DType::kF16,
                                                 {k / 128, n});
  weight.zero_point = vllm::dense_loaders::MakeOwned(vt::DType::kI8, {1});
  std::vector<uint32_t> packed(static_cast<size_t>(n * k / 8), words);
  std::vector<uint16_t> scales(static_cast<size_t>(n * k / 128),
                               vt::F32ToF16(scale));
  std::memcpy(weight.qweight.bytes.data(), packed.data(),
              weight.qweight.bytes.size());
  std::memcpy(weight.scales.bytes.data(), scales.data(),
              weight.scales.bytes.size());
  weight.zero_point.bytes.data()[0] = 8;
  return weight;
}

}  // namespace

TEST_CASE("XPU GPTQ FP16 convolution keeps FP16 persistent history") {
  QueueOwner owner;
  auto& q = owner.queue;
  Buffer input(q, vt::DType::kF16, {4, 3});
  Buffer weight(q, vt::DType::kF16, {3, 3});
  Buffer output(q, vt::DType::kF16, {4, 3});
  Buffer state(q, vt::DType::kF16, {1, 3, 2});
  Buffer offsets(q, vt::DType::kI32, {2});
  Buffer initial(q, vt::DType::kI8, {1});
  std::vector<uint16_t> xbits(12), wbits(9);
  std::vector<float> x(12), w(9);
  for (size_t i = 0; i < x.size(); ++i) {
    xbits[i] = vt::F32ToF16((static_cast<int>(i) - 5) * 0.125f);
    x[i] = vt::F16ToF32(xbits[i]);
  }
  for (size_t i = 0; i < w.size(); ++i) {
    wbits[i] = vt::F32ToF16((static_cast<int>(i % 4) + 1) * 0.0625f);
    w[i] = vt::F16ToF32(wbits[i]);
  }
  input.Upload(xbits.data(), input.bytes);
  weight.Upload(wbits.data(), weight.bytes);
  const std::vector<uint16_t> zeros(6, 0);
  state.Upload(zeros.data(), state.bytes);
  const int32_t qsl[] = {0, 4};
  const int8_t has_initial[] = {0};
  offsets.Upload(qsl, offsets.bytes);
  initial.Upload(has_initial, initial.bytes);
  vt::CausalConv1dFwd(q, output.tensor, input.tensor, weight.tensor, nullptr,
                      state.tensor, offsets.tensor, initial.tensor, {true});
  const auto got = output.Read();
  const auto history = state.Read();
  for (int t = 0; t < 4; ++t) for (int c = 0; c < 3; ++c) {
    float sum = 0;
    for (int tap = 0; tap < 3; ++tap) {
      const int source = t - 2 + tap;
      if (source >= 0) sum += x[source * 3 + c] * w[c * 3 + tap];
    }
    const float expected = sum / (1.0f + std::exp(-sum));
    CHECK(std::abs(ReadHalf(got.data(), t * 3 + c) - expected) < 0.001f);
  }
  for (int c = 0; c < 3; ++c) for (int tap = 0; tap < 2; ++tap)
    CHECK(ReadHalf(history.data(), c * 2 + tap) == x[(2 + tap) * 3 + c]);
}

TEST_CASE("XPU GPTQ W4A16 packed nibbles, scales and runtime reuse") {
  QueueOwner owner;
  auto& q = owner.queue;
  constexpr int m = 1, k = 5120, n = 1024, group = 128;
  Buffer activation(q, vt::DType::kF16, {m, k});
  Buffer qweight(q, vt::DType::kI32, {n, k / 8});
  Buffer scales(q, vt::DType::kF16, {k / group, n});
  Buffer zero_point(q, vt::DType::kI8, {1});
  Buffer output(q, vt::DType::kF16, {m, n});

  std::vector<uint16_t> a_bits(m * k);
  std::vector<float> a(m * k);
  for (int i = 0; i < m * k; ++i) {
    a[i] = (static_cast<int>(i % 17) - 8) / 16.0f;
    a_bits[i] = vt::F32ToF16(a[i]);
    a[i] = vt::F16ToF32(a_bits[i]);
  }
  std::vector<uint32_t> packed(n * (k / 8));
  std::vector<int> codes(n * k);
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < k; ++col) {
      const int code = 5 + (row * 3 + col * 5) % 7;
      codes[row * k + col] = code;
      packed[row * (k / 8) + col / 8] |=
          static_cast<uint32_t>(code) << (4 * (col % 8));
    }
  }
  std::vector<uint16_t> scale_bits((k / group) * n);
  std::vector<float> scale_values((k / group) * n);
  for (int group_id = 0; group_id < k / group; ++group_id) {
    for (int row = 0; row < n; ++row) {
      const auto offset = group_id * n + row;
      scale_values[offset] = vt::F16ToF32(vt::F32ToF16(
          0.0625f + 0.015625f * static_cast<float>(row % 7)));
      scale_bits[offset] = vt::F32ToF16(scale_values[offset]);
    }
  }
  const int8_t zp = 8;
  activation.Upload(a_bits.data(), a_bits.size() * sizeof(a_bits[0]));
  qweight.Upload(packed.data(), packed.size() * sizeof(packed[0]));
  scales.Upload(scale_bits.data(), scale_bits.size() * sizeof(scale_bits[0]));
  zero_point.Upload(&zp, sizeof(zp));

  std::vector<float> expected(m * n, 0.0f);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      for (int inner = 0; inner < k; ++inner) {
        const int group_id = inner / group;
        const float weight = (codes[col * k + inner] - zp) *
                             scale_values[group_id * n + col];
        expected[row * n + col] += a[row * k + inner] * weight;
      }
    }
  }

  const auto before = vt::xpu::GetGptq4RuntimeStats(q.device.index);
  for (int iteration = 0; iteration < 5; ++iteration)
    vt::MatmulGptq4W4A16(q, output.tensor, activation.tensor, qweight.tensor,
                         scales.tensor, zero_point.tensor, group);
  vt::GetBackend(q.device).Synchronize(q);
  RequireClose(output.Read(), expected);
  const auto after = vt::xpu::GetGptq4RuntimeStats(q.device.index);
  CHECK(after.engine_count == 1);
  CHECK(after.primitive_count == before.primitive_count + 1);
  CHECK(after.scratchpad_allocation_count >= before.scratchpad_allocation_count);
  CHECK(after.primitive_count ==
        vt::xpu::GetGptq4RuntimeStats(q.device.index).primitive_count);
}

TEST_CASE("XPU GPTQ dense F16 matmul with optional bias") {
  QueueOwner owner;
  auto& q = owner.queue;
  constexpr int m = 1, k = 5120, n = 96;
  Buffer activation(q, vt::DType::kF16, {m, k});
  Buffer weight(q, vt::DType::kF16, {n, k});
  Buffer bias(q, vt::DType::kF16, {n});
  Buffer output(q, vt::DType::kF16, {m, n});
  Buffer output_no_bias(q, vt::DType::kF16, {m, n});
  std::vector<uint16_t> a_bits(m * k), w_bits(n * k), b_bits(n);
  for (size_t i = 0; i < a_bits.size(); ++i)
    a_bits[i] = vt::F32ToF16((static_cast<int>(i % 23) - 11) / 16.0f);
  for (size_t i = 0; i < w_bits.size(); ++i)
    w_bits[i] = vt::F32ToF16((static_cast<int>(i % 19) - 9) / 32.0f);
  for (size_t i = 0; i < b_bits.size(); ++i)
    b_bits[i] = vt::F32ToF16((static_cast<int>(i % 7) - 3) / 8.0f);
  activation.Upload(a_bits.data(), a_bits.size() * sizeof(a_bits[0]));
  weight.Upload(w_bits.data(), w_bits.size() * sizeof(w_bits[0]));
  bias.Upload(b_bits.data(), b_bits.size() * sizeof(b_bits[0]));

  std::vector<float> expected(m * n), expected_no_bias(m * n);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int inner = 0; inner < k; ++inner)
        sum += vt::F16ToF32(a_bits[row * k + inner]) *
               vt::F16ToF32(w_bits[col * k + inner]);
      expected_no_bias[row * n + col] = sum;
      expected[row * n + col] = sum + vt::F16ToF32(b_bits[col]);
    }
  }
  const auto before = vt::xpu::GetGptq4RuntimeStats(q.device.index);
  vt::MatmulDenseF16(q, output.tensor, activation.tensor, weight.tensor,
                     &bias.tensor);
  vt::GetBackend(q.device).Synchronize(q);
  RequireClose(output.Read(), expected);
  vt::MatmulDenseF16(q, output_no_bias.tensor, activation.tensor, weight.tensor);
  vt::GetBackend(q.device).Synchronize(q);
  RequireClose(output_no_bias.Read(), expected_no_bias);
  const auto stats = vt::xpu::GetGptq4RuntimeStats(q.device.index);
  CHECK(stats.engine_count == 1);
  CHECK(stats.primitive_count == before.primitive_count + 2);
}

TEST_CASE("XPU GPTQ oneDNN profile brackets packed and dense stream spans"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE"))) {
  QueueOwner owner;
  auto& q = owner.queue;
  Buffer activation(q, vt::DType::kF16, {1, 128});
  Buffer qweight(q, vt::DType::kI32, {8, 16});
  Buffer scales(q, vt::DType::kF16, {1, 8});
  Buffer zero(q, vt::DType::kI8, {1});
  Buffer dense_weight(q, vt::DType::kF16, {8, 128});
  Buffer packed_out(q, vt::DType::kF16, {1, 8});
  Buffer dense_out(q, vt::DType::kF16, {1, 8});
  activation.Upload(std::vector<uint16_t>(128, vt::F32ToF16(1)).data(),
                    activation.bytes);
  qweight.Upload(std::vector<uint32_t>(8 * 16, 0x88888888u).data(),
                 qweight.bytes);
  scales.Upload(std::vector<uint16_t>(8, vt::F32ToF16(1)).data(), scales.bytes);
  const int8_t zero_point = 8;
  zero.Upload(&zero_point, zero.bytes);
  dense_weight.Upload(std::vector<uint16_t>(8 * 128, vt::F32ToF16(1)).data(),
                      dense_weight.bytes);
  (void)vt::xpu::DrainProfileEvents(q.device.index);
  vt::MatmulGptq4W4A16(q, packed_out.tensor, activation.tensor, qweight.tensor,
                       scales.tensor, zero.tensor, 128);
  vt::MatmulDenseF16(q, dense_out.tensor, activation.tensor,
                     dense_weight.tensor);
  vt::GetBackend(q.device).Synchronize(q);
  const auto records = vt::xpu::DrainProfileEvents(q.device.index);
  std::vector<vt::xpu::ProfileRecord> spans;
  for (const auto& record : records)
    if (record.stream_span) spans.push_back(record);
  REQUIRE(spans.size() == 2);
  CHECK(spans[0].stage == "onednn_gptq4_stream");
  CHECK(spans[1].stage == "onednn_dense_f16_stream");
  for (const auto& span : spans) {
    CHECK(span.matrix.find("M=1 K=128 N=8") != std::string::npos);
    CHECK(span.matrix.find("impl=") != std::string::npos);
    CHECK(span.host_submit_ns > 0);
    CHECK(span.end_ns > span.start_ns);
    CHECK(span.queue_id == q.id);
  }
}

TEST_CASE("XPU GPTQ typed linear seam selects packed and dense FP16 providers") {
  QueueOwner owner;
  auto& queue = owner.queue;
  Buffer input(queue, vt::DType::kF16, {1, 128});
  Buffer dense_weight(queue, vt::DType::kF16, {8, 128});
  const std::vector<uint16_t> input_bits(128, vt::F32ToF16(0.5f));
  const std::vector<uint16_t> weight_bits(8 * 128, vt::F32ToF16(0.125f));
  input.Upload(input_bits.data(), input.bytes);
  dense_weight.Upload(weight_bits.data(), dense_weight.bytes);

  vllm::Gptq4Weight packed;
  packed.k = 128;
  packed.n = 8;
  packed.qweight = vllm::dense_loaders::MakeOwned(vt::DType::kI32, {8, 16});
  packed.scales = vllm::dense_loaders::MakeOwned(vt::DType::kF16, {1, 8});
  packed.zero_point = vllm::dense_loaders::MakeOwned(vt::DType::kI8, {1});
  const std::vector<uint32_t> words(8 * 16, 0x99999999u);
  const std::vector<uint16_t> scale_bits(8, vt::F32ToF16(0.125f));
  std::memcpy(packed.qweight.bytes.data(), words.data(), packed.qweight.bytes.size());
  std::memcpy(packed.scales.bytes.data(), scale_bits.data(), packed.scales.bytes.size());
  packed.zero_point.bytes.data()[0] = 8;

  vllm::dense_attn::Dev dev{vt::GetBackend(queue.device), queue,
                            vt::DType::kF16};
  const auto before = vllm::dense_gptq4::GetDispatchCounts();
  const bool profile = std::getenv("VT_XPU_PROFILE") != nullptr;
  if (profile) (void)vt::xpu::DrainProfileEvents(queue.device.index);
  const vt::xpu::ProfileLayerScope profile_layer(7);
  auto packed_out = vllm::dense_gptq4::Packed(
      dev, input.tensor, packed, vllm::dense_gptq4::Projection::kMlpGateUp);
  auto dense_out = vllm::dense_gptq4::Dense(
      dev, input.tensor, dense_weight.tensor,
      vllm::dense_gptq4::Projection::kGdnBa);
  std::vector<uint16_t> actual_packed(8), actual_dense(8);
  auto& backend = vt::GetBackend(queue.device);
  backend.Copy(queue, actual_packed.data(), packed_out.t().data, 16);
  backend.Copy(queue, actual_dense.data(), dense_out.t().data, 16);
  backend.Synchronize(queue);
  for (int i = 0; i < 8; ++i) {
    CHECK(vt::F16ToF32(actual_packed[i]) == doctest::Approx(8.0f));
    CHECK(vt::F16ToF32(actual_dense[i]) == doctest::Approx(8.0f));
  }
  const auto after = vllm::dense_gptq4::GetDispatchCounts();
  using vllm::dense_gptq4::Projection;
  CHECK(after.calls[static_cast<size_t>(Projection::kMlpGateUp)] ==
        before.calls[static_cast<size_t>(Projection::kMlpGateUp)] + 1);
  CHECK(after.calls[static_cast<size_t>(Projection::kGdnBa)] ==
        before.calls[static_cast<size_t>(Projection::kGdnBa)] + 1);
  if (profile) {
    std::vector<vt::xpu::ProfileRecord> spans;
    for (const auto& record : vt::xpu::DrainProfileEvents(queue.device.index))
      if (record.stream_span) spans.push_back(record);
    REQUIRE(spans.size() == 2);
    CHECK(spans[0].matrix.find("layer=7 mlp_gate_up M=1 K=128 N=8") !=
          std::string::npos);
    CHECK(spans[1].matrix.find("layer=7 gdn_ba M=1 K=128 N=8") !=
          std::string::npos);
  }
}

TEST_CASE("XPU GPTQ oneDNN matmul records and replays on a SYCL command graph") {
  QueueOwner owner;
  auto& queue = owner.queue;
  auto& backend = vt::GetBackend(queue.device);
  REQUIRE(backend.SupportsGraphCapture());
  Buffer input(queue, vt::DType::kF16, {1, 128});
  Buffer qweight(queue, vt::DType::kI32, {8, 16});
  Buffer scales(queue, vt::DType::kF16, {1, 8});
  Buffer zero_point(queue, vt::DType::kI8, {1});
  Buffer output(queue, vt::DType::kF16, {1, 8});
  const std::vector<uint32_t> words(8 * 16, 0x99999999u);
  const std::vector<uint16_t> scale_bits(8, vt::F32ToF16(0.125f));
  const int8_t zp = 8;
  qweight.Upload(words.data(), words.size() * sizeof(words[0]));
  scales.Upload(scale_bits.data(), scale_bits.size() * sizeof(scale_bits[0]));
  zero_point.Upload(&zp, 1);
  const auto run = [&] {
    vt::MatmulGptq4W4A16(queue, output.tensor, input.tensor, qweight.tensor,
                         scales.tensor, zero_point.tensor, 128);
  };
  const std::vector<uint16_t> half_input(128, vt::F32ToF16(0.5f));
  input.Upload(half_input.data(), input.bytes);
  run();
  backend.Synchronize(queue);
  RequireClose(output.Read(), std::vector<float>(8, 8.0f));
  const auto before = vt::xpu::GetGptq4RuntimeStats(queue.device.index);
  void* graph = nullptr;
  backend.BeginCapture(queue);
  try {
    run();
    graph = backend.EndCaptureGraph(queue);
  } catch (const std::exception& error) {
    try {
      void* partial = backend.EndCaptureGraph(queue);
      backend.DestroyGraph(partial);
    } catch (...) {}
    FAIL("pinned oneDNN GPTQ matmul cannot be captured: " << error.what());
  }
  const auto captured = vt::xpu::GetGptq4RuntimeStats(queue.device.index);
  CHECK(captured.primitive_count == before.primitive_count);
  CHECK(captured.scratchpad_allocation_count == before.scratchpad_allocation_count);
  for (float activation : {1.0f, 0.25f}) {
    const std::vector<uint16_t> bits(128, vt::F32ToF16(activation));
    input.Upload(bits.data(), input.bytes);
    backend.ReplayGraph(queue, graph);
    RequireClose(output.Read(), std::vector<float>(8, 16.0f * activation));
  }
  backend.DestroyGraph(graph);
}

TEST_CASE("XPU GPTQ captured slots retain distinct bindings and shared scratchpad") {
  QueueOwner owner;
  auto& queue = owner.queue;
  auto& backend = vt::GetBackend(queue.device);
  REQUIRE(backend.SupportsGraphCapture());
  Buffer qweight(queue, vt::DType::kI32, {8, 16});
  Buffer scales(queue, vt::DType::kF16, {1, 8});
  Buffer zero_point(queue, vt::DType::kI8, {1});
  const std::vector<uint32_t> words(8 * 16, 0x99999999u);
  const std::vector<uint16_t> scale_bits(8, vt::F32ToF16(0.125f));
  const int8_t zp = 8;
  qweight.Upload(words.data(), words.size() * sizeof(words[0]));
  scales.Upload(scale_bits.data(), scale_bits.size() * sizeof(scale_bits[0]));
  zero_point.Upload(&zp, 1);
  std::array<std::unique_ptr<Buffer>, 2> input, output;
  std::array<void*, 2> graphs{};
  const auto run = [&](int slot) {
    vt::MatmulGptq4W4A16(queue, output[slot]->tensor, input[slot]->tensor,
                         qweight.tensor, scales.tensor, zero_point.tensor, 128);
  };
  for (int slot = 0; slot < 2; ++slot) {
    input[slot] = std::make_unique<Buffer>(queue, vt::DType::kF16,
                                           std::initializer_list<int64_t>{1, 128});
    output[slot] = std::make_unique<Buffer>(queue, vt::DType::kF16,
                                            std::initializer_list<int64_t>{1, 8});
    const std::vector<uint16_t> bits(128, vt::F32ToF16(0.5f));
    input[slot]->Upload(bits.data(), input[slot]->bytes);
    run(slot);
    backend.Synchronize(queue);
  }
  const auto warm = vt::xpu::GetGptq4RuntimeStats(queue.device.index);
  for (int slot = 0; slot < 2; ++slot) {
    backend.BeginCapture(queue);
    run(slot);
    graphs[slot] = backend.EndCaptureGraph(queue);
  }
  const auto captured = vt::xpu::GetGptq4RuntimeStats(queue.device.index);
  CHECK(captured.primitive_count == warm.primitive_count);
  CHECK(captured.scratchpad_allocation_count == warm.scratchpad_allocation_count);
  for (int iteration = 0; iteration < 3; ++iteration) {
    for (int slot = 0; slot < 2; ++slot) {
      const float activation = 0.25f * (1 + slot + iteration);
      const std::vector<uint16_t> bits(128, vt::F32ToF16(activation));
      input[slot]->Upload(bits.data(), input[slot]->bytes);
    }
    backend.ReplayGraph(queue, graphs[0]);
    backend.ReplayGraph(queue, graphs[1]);
    for (int slot = 0; slot < 2; ++slot)
      RequireClose(output[slot]->Read(),
                   std::vector<float>(8, 4.0f * (1 + slot + iteration)));
  }
  backend.DestroyGraph(graphs[0]);
  graphs[0] = nullptr;
  backend.BeginCapture(queue);
  run(0);
  graphs[0] = backend.EndCaptureGraph(queue);
  backend.ReplayGraph(queue, graphs[0]);
  RequireClose(output[0]->Read(), std::vector<float>(8, 12.0f));
  for (void* graph : graphs) backend.DestroyGraph(graph);
}

TEST_CASE("XPU GPTQ merged SwiGLU uses packed gate-up and down projections") {
  QueueOwner owner;
  auto& queue = owner.queue;
  Buffer input(queue, vt::DType::kF16, {1, 128});
  const std::vector<uint16_t> input_bits(128, vt::F32ToF16(0.5f));
  input.Upload(input_bits.data(), input.bytes);
  auto gate_up = ConstantPacked(128, 256, 0x99999999u, 0.015625f);
  auto down = ConstantPacked(128, 128, 0x99999999u, 0.015625f);
  vllm::dense_attn::Dev dev{vt::GetBackend(queue.device), queue,
                            vt::DType::kF16};
  const auto before = vllm::dense_gptq4::GetDispatchCounts();
  auto output = vllm::dense_gptq4::Mlp(dev, input.tensor, gate_up, down, 128);
  std::vector<uint16_t> actual(128);
  auto& backend = vt::GetBackend(queue.device);
  backend.Copy(queue, actual.data(), output.t().data, actual.size() * 2);
  backend.Synchronize(queue);
  const float projection = 128 * 0.5f * 0.015625f;
  const float activated = vt::F16ToF32(vt::F32ToF16(
      projection * projection / (1.0f + std::exp(-projection))));
  const float expected = 128 * activated * 0.015625f;
  for (uint16_t bits : actual)
    CHECK(std::abs(vt::F16ToF32(bits) - expected) < 0.01f);
  const auto after = vllm::dense_gptq4::GetDispatchCounts();
  using vllm::dense_gptq4::Projection;
  for (Projection projection : {Projection::kMlpGateUp, Projection::kMlpDown}) {
    const size_t i = static_cast<size_t>(projection);
    CHECK(after.calls[i] == before.calls[i] + 1);
  }
}

TEST_CASE("XPU GPTQ merged attention QKV preserves Q gate K V row order") {
  QueueOwner owner;
  auto& queue = owner.queue;
  Buffer input(queue, vt::DType::kF16, {2, 128});
  std::vector<uint16_t> input_bits(256);
  std::fill(input_bits.begin(), input_bits.begin() + 128,
            vt::F32ToF16(0.5f));
  std::fill(input_bits.begin() + 128, input_bits.end(),
            vt::F32ToF16(0.25f));
  input.Upload(input_bits.data(), input.bytes);
  auto qkv = ConstantPacked(128, 384, 0x99999999u, 0.125f);
  auto* words = reinterpret_cast<uint32_t*>(qkv.qweight.bytes.data());
  for (int row = 128; row < 256; ++row)
    std::fill(words + row * 16, words + (row + 1) * 16, 0xaaaaaaaau);
  for (int row = 256; row < 384; ++row)
    std::fill(words + row * 16, words + (row + 1) * 16, 0xbbbbbbbbu);
  vllm::dense_attn::Dev dev{vt::GetBackend(queue.device), queue,
                            vt::DType::kF16};
  const auto before = vllm::dense_gptq4::GetDispatchCounts();
  auto output = vllm::dense_gptq4::AttentionQkv(
      dev, input.tensor, qkv, 128, 128);
  std::vector<uint16_t> actual(2 * 384);
  auto& backend = vt::GetBackend(queue.device);
  backend.Copy(queue, actual.data(), output.t().data, actual.size() * 2);
  backend.Synchronize(queue);
  for (int token = 0; token < 2; ++token)
    for (int segment = 0; segment < 3; ++segment)
      for (int col = 0; col < 128; ++col) {
        const float expected = (token == 0 ? 8.0f : 4.0f) * (segment + 1);
        CHECK(vt::F16ToF32(actual[token * 384 + segment * 128 + col]) ==
              doctest::Approx(expected));
      }
  const auto after = vllm::dense_gptq4::GetDispatchCounts();
  const size_t i = static_cast<size_t>(vllm::dense_gptq4::Projection::kAttnQkv);
  CHECK(after.calls[i] == before.calls[i] + 1);
}

TEST_CASE("XPU GPTQ FP16 BF16 F32 RMSNorm and SwiGLU boundaries") {
  QueueOwner owner;
  auto& queue = owner.queue;
  const auto encode = [](const std::vector<float>& values, vt::DType dtype) {
    std::vector<uint8_t> bytes(values.size() * vt::SizeOf(dtype));
    for (size_t i = 0; i < values.size(); ++i) {
      if (dtype == vt::DType::kF16) {
        const uint16_t bits = vt::F32ToF16(values[i]);
        std::memcpy(bytes.data() + i * 2, &bits, 2);
      } else if (dtype == vt::DType::kBF16) {
        const uint16_t bits = vt::F32ToBF16(values[i]);
        std::memcpy(bytes.data() + i * 2, &bits, 2);
      } else {
        std::memcpy(bytes.data() + i * 4, &values[i], 4);
      }
    }
    return bytes;
  };
  const auto decode = [](const std::vector<uint8_t>& bytes, vt::DType dtype,
                         size_t index) {
    if (dtype == vt::DType::kF16 || dtype == vt::DType::kBF16) {
      uint16_t bits;
      std::memcpy(&bits, bytes.data() + index * 2, 2);
      return dtype == vt::DType::kF16 ? vt::F16ToF32(bits)
                                       : vt::BF16ToF32(bits);
    }
    float value;
    std::memcpy(&value, bytes.data() + index * 4, 4);
    return value;
  };
  const auto round = [&encode, &decode](float value, vt::DType dtype) {
    return decode(encode({value}, dtype), dtype, 0);
  };
  for (vt::DType dtype : {vt::DType::kF16, vt::DType::kBF16,
                          vt::DType::kF32}) {
    std::vector<float> input(16), residual_input(16), norm_weight(8);
    for (size_t i = 0; i < 16; ++i) {
      input[i] = (static_cast<int>(i % 9) - 4) / 8.0f;
      residual_input[i] = (static_cast<int>(i % 7) - 3) / 16.0f;
    }
    for (size_t i = 0; i < 8; ++i)
      norm_weight[i] = (static_cast<int>(i) - 4) / 32.0f;
    Buffer x(queue, dtype, {2, 8});
    Buffer residual(queue, dtype, {2, 8});
    Buffer weight(queue, dtype, {8});
    Buffer output(queue, dtype, {2, 8});
    const auto x_bytes = encode(input, dtype);
    const auto r_bytes = encode(residual_input, dtype);
    const auto w_bytes = encode(norm_weight, dtype);
    x.Upload(x_bytes.data(), x_bytes.size());
    residual.Upload(r_bytes.data(), r_bytes.size());
    weight.Upload(w_bytes.data(), w_bytes.size());
    vt::RmsNorm(queue, output.tensor, x.tensor, weight.tensor,
                {1e-6f, true}, &residual.tensor);
    const auto output_bytes = output.Read();
    const auto residual_bytes = residual.Read();
    for (int row = 0; row < 2; ++row) {
      float combined[8];
      float sum = 0.0f;
      for (int col = 0; col < 8; ++col) {
        const size_t index = row * 8 + col;
        combined[col] = round(decode(x_bytes, dtype, index) +
                                  decode(r_bytes, dtype, index), dtype);
        sum += combined[col] * combined[col];
        CHECK(std::abs(decode(residual_bytes, dtype, index) - combined[col]) <
              0.001f);
      }
      const float scale = 1.0f / std::sqrt(sum / 8.0f + 1e-6f);
      for (int col = 0; col < 8; ++col) {
        const size_t index = row * 8 + col;
        const float expected = round(
            combined[col] * scale * (1.0f + decode(w_bytes, dtype, col)),
            dtype);
        CHECK(std::abs(decode(output_bytes, dtype, index) - expected) < 0.02f);
      }
    }
    Buffer gate_up(queue, dtype, {2, 16});
    Buffer swiglu(queue, dtype, {2, 8});
    std::vector<float> gate_up_values(32);
    for (size_t i = 0; i < 32; ++i)
      gate_up_values[i] = (static_cast<int>(i % 13) - 6) / 8.0f;
    const auto gate_up_bytes = encode(gate_up_values, dtype);
    gate_up.Upload(gate_up_bytes.data(), gate_up_bytes.size());
    vt::SiluAndMul(queue, swiglu.tensor, gate_up.tensor);
    const auto swiglu_bytes = swiglu.Read();
    for (int row = 0; row < 2; ++row) {
      for (int col = 0; col < 8; ++col) {
        const float gate = decode(gate_up_bytes, dtype, row * 16 + col);
        const float up = decode(gate_up_bytes, dtype, row * 16 + col + 8);
        const float silu = round(gate / (1.0f + std::exp(-gate)), dtype);
        const float expected = round(silu * up, dtype);
        CHECK(std::abs(decode(swiglu_bytes, dtype, row * 8 + col) - expected) <
              0.02f);
      }
    }
  }
}

TEST_CASE("XPU GPTQ FP16 embedding add and gated norm boundaries") {
  QueueOwner owner;
  auto& queue = owner.queue;
  std::vector<uint16_t> table_values(32);
  for (size_t i = 0; i < table_values.size(); ++i)
    table_values[i] = vt::F32ToF16((static_cast<int>(i) - 15) / 32.0f);
  const std::array<int32_t, 3> ids{3, 0, 2};
  Buffer table(queue, vt::DType::kF16, {4, 8});
  Buffer indices(queue, vt::DType::kI32, {3});
  Buffer gathered(queue, vt::DType::kF16, {3, 8});
  table.Upload(table_values.data(), table_values.size() * 2);
  indices.Upload(ids.data(), ids.size() * sizeof(ids[0]));
  vt::Embedding(queue, gathered.tensor, table.tensor, indices.tensor);
  const auto embedding = gathered.Read();
  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < 8; ++col) {
      uint16_t actual;
      std::memcpy(&actual, embedding.data() + (row * 8 + col) * 2, 2);
      CHECK(actual == table_values[ids[row] * 8 + col]);
    }

  std::vector<uint16_t> bias_values(8);
  for (int col = 0; col < 8; ++col)
    bias_values[col] = vt::F32ToF16((col - 4) / 16.0f);
  Buffer bias(queue, vt::DType::kF16, {8});
  Buffer added(queue, vt::DType::kF16, {3, 8});
  bias.Upload(bias_values.data(), bias_values.size() * 2);
  vt::Add(queue, added.tensor, gathered.tensor, bias.tensor);
  const auto add_result = added.Read();
  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < 8; ++col) {
      const float expected = vt::F16ToF32(vt::F32ToF16(
          vt::F16ToF32(table_values[ids[row] * 8 + col]) +
          vt::F16ToF32(bias_values[col])));
      CHECK(std::abs(ReadHalf(add_result.data(), row * 8 + col) - expected) <
            0.001f);
    }

  std::vector<uint16_t> gate_values(24);
  for (size_t i = 0; i < gate_values.size(); ++i)
    gate_values[i] = vt::F32ToF16((static_cast<int>(i % 11) - 5) / 8.0f);
  Buffer gate(queue, vt::DType::kF16, {3, 8});
  Buffer normed(queue, vt::DType::kF16, {3, 8});
  gate.Upload(gate_values.data(), gate_values.size() * 2);
  vt::RmsNormGated(queue, normed.tensor, added.tensor, gate.tensor,
                   bias.tensor, {1e-6f, false});
  const auto norm_result = normed.Read();
  for (int row = 0; row < 3; ++row) {
    float sum = 0.0f;
    for (int col = 0; col < 8; ++col) {
      const float x = ReadHalf(add_result.data(), row * 8 + col);
      sum += x * x;
    }
    const float inv = 1.0f / std::sqrt(sum / 8.0f + 1e-6f);
    for (int col = 0; col < 8; ++col) {
      const float x = ReadHalf(add_result.data(), row * 8 + col);
      const float g = vt::F16ToF32(gate_values[row * 8 + col]);
      const float w = vt::F16ToF32(bias_values[col]);
      const float expected = x * inv * w * g / (1.0f + std::exp(-g));
      CHECK(std::abs(ReadHalf(norm_result.data(), row * 8 + col) - expected) <
            0.02f);
    }
  }
}

TEST_CASE("XPU GPTQ FP16 GDN post-convolution keeps gate state FP32") {
  QueueOwner owner;
  auto& queue = owner.queue;
  Buffer conv(queue, vt::DType::kF16, {2, 16});
  Buffer araw(queue, vt::DType::kF16, {2, 2});
  Buffer braw(queue, vt::DType::kF16, {2, 2});
  Buffer alog(queue, vt::DType::kF32, {2});
  Buffer dt_bias(queue, vt::DType::kF32, {2});
  Buffer qout(queue, vt::DType::kF16, {2, 1, 4});
  Buffer kout(queue, vt::DType::kF16, {2, 1, 4});
  Buffer vout(queue, vt::DType::kF16, {2, 2, 4});
  Buffer gout(queue, vt::DType::kF32, {2, 2});
  Buffer beta(queue, vt::DType::kF32, {2, 2});
  std::vector<uint16_t> conv_values(32), a_values(4), b_values(4);
  for (size_t i = 0; i < conv_values.size(); ++i)
    conv_values[i] = vt::F32ToF16((static_cast<int>(i % 11) - 5) / 8.0f);
  for (size_t i = 0; i < 4; ++i) {
    a_values[i] = vt::F32ToF16((static_cast<int>(i) - 2) / 4.0f);
    b_values[i] = vt::F32ToF16((static_cast<int>(i) - 1) / 4.0f);
  }
  const std::array<float, 2> alog_values{-1.0f, -0.5f};
  const std::array<float, 2> bias_values{0.25f, -0.25f};
  conv.Upload(conv_values.data(), conv_values.size() * 2);
  araw.Upload(a_values.data(), a_values.size() * 2);
  braw.Upload(b_values.data(), b_values.size() * 2);
  alog.Upload(alog_values.data(), alog_values.size() * 4);
  dt_bias.Upload(bias_values.data(), bias_values.size() * 4);
  vt::GdnPostConv(queue, qout.tensor, kout.tensor, vout.tensor, gout.tensor,
                  beta.tensor, conv.tensor, araw.tensor, braw.tensor,
                  alog.tensor, dt_bias.tensor, {1e-6f, false});
  const auto q_result = qout.Read();
  const auto k_result = kout.Read();
  const auto v_result = vout.Read();
  const auto g_result = gout.Read();
  const auto beta_result = beta.Read();
  for (int token = 0; token < 2; ++token) {
    float qs = 0.0f, ks = 0.0f;
    for (int col = 0; col < 4; ++col) {
      const float qv = vt::F16ToF32(conv_values[token * 16 + col]);
      const float kv = vt::F16ToF32(conv_values[token * 16 + 4 + col]);
      qs += qv * qv;
      ks += kv * kv;
    }
    for (int col = 0; col < 4; ++col) {
      const float qv = vt::F16ToF32(conv_values[token * 16 + col]);
      const float kv = vt::F16ToF32(conv_values[token * 16 + 4 + col]);
      CHECK(std::abs(ReadHalf(q_result.data(), token * 4 + col) -
                     qv / std::sqrt(qs + 1e-6f)) < 0.002f);
      CHECK(std::abs(ReadHalf(k_result.data(), token * 4 + col) -
                     kv / std::sqrt(ks + 1e-6f)) < 0.002f);
    }
    for (int col = 0; col < 8; ++col)
      CHECK(ReadHalf(v_result.data(), token * 8 + col) ==
            vt::F16ToF32(conv_values[token * 16 + 8 + col]));
    for (int head = 0; head < 2; ++head) {
      const size_t index = token * 2 + head;
      const float a = vt::F16ToF32(a_values[index]) + bias_values[head];
      const float b = vt::F16ToF32(b_values[index]);
      const float expected_g = -std::exp(alog_values[head]) *
                               std::log1p(std::exp(a));
      const float expected_beta = 1.0f / (1.0f + std::exp(-b));
      float actual_g, actual_beta;
      std::memcpy(&actual_g, g_result.data() + index * 4, 4);
      std::memcpy(&actual_beta, beta_result.data() + index * 4, 4);
      CHECK(std::abs(actual_g - expected_g) < 1e-5f);
      CHECK(std::abs(actual_beta - expected_beta) < 1e-5f);
    }
  }
}

TEST_CASE("XPU GPTQ FP16 GDN recurrence matches FP32 output and state") {
  QueueOwner owner;
  auto& queue = owner.queue;
  Buffer qin(queue, vt::DType::kF16, {2, 1, 4});
  Buffer kin(queue, vt::DType::kF16, {2, 1, 4});
  Buffer vin(queue, vt::DType::kF16, {2, 2, 4});
  Buffer g(queue, vt::DType::kF32, {2, 2});
  Buffer beta(queue, vt::DType::kF32, {2, 2});
  Buffer state_f16(queue, vt::DType::kF32, {1, 2, 4, 4});
  Buffer state_f32(queue, vt::DType::kF32, {1, 2, 4, 4});
  Buffer qsl(queue, vt::DType::kI32, {2});
  Buffer output_f16(queue, vt::DType::kF16, {2, 2, 4});
  Buffer output_f32(queue, vt::DType::kF32, {2, 2, 4});
  std::array<uint16_t, 8> q_values{}, k_values{};
  std::array<uint16_t, 16> v_values{};
  for (size_t i = 0; i < q_values.size(); ++i) {
    q_values[i] = vt::F32ToF16((static_cast<int>(i) - 3) / 8.0f);
    k_values[i] = vt::F32ToF16((static_cast<int>(i % 5) - 2) / 8.0f);
  }
  for (size_t i = 0; i < v_values.size(); ++i)
    v_values[i] = vt::F32ToF16((static_cast<int>(i % 9) - 4) / 8.0f);
  const std::array<float, 4> g_values{-0.2f, -0.4f, -0.3f, -0.5f};
  const std::array<float, 4> beta_values{0.2f, 0.3f, 0.4f, 0.5f};
  const std::array<int32_t, 2> offsets{0, 2};
  const std::array<float, 32> zeros{};
  qin.Upload(q_values.data(), q_values.size() * 2);
  kin.Upload(k_values.data(), k_values.size() * 2);
  vin.Upload(v_values.data(), v_values.size() * 2);
  g.Upload(g_values.data(), g_values.size() * 4);
  beta.Upload(beta_values.data(), beta_values.size() * 4);
  state_f16.Upload(zeros.data(), zeros.size() * 4);
  state_f32.Upload(zeros.data(), zeros.size() * 4);
  qsl.Upload(offsets.data(), offsets.size() * 4);
  vt::GdnPrefill(queue, output_f16.tensor, qin.tensor, kin.tensor, vin.tensor,
                 g.tensor, beta.tensor, state_f16.tensor, qsl.tensor, {0.5f});
  vt::GdnPrefill(queue, output_f32.tensor, qin.tensor, kin.tensor, vin.tensor,
                 g.tensor, beta.tensor, state_f32.tensor, qsl.tensor, {0.5f});
  const auto half_result = output_f16.Read();
  const auto float_result = output_f32.Read();
  const auto half_state = state_f16.Read();
  const auto float_state = state_f32.Read();
  CHECK(half_state == float_state);
  for (size_t i = 0; i < 16; ++i) {
    float value;
    std::memcpy(&value, float_result.data() + i * 4, 4);
    CHECK(std::abs(ReadHalf(half_result.data(), i) - value) < 0.002f);
  }
}

TEST_CASE("XPU GPTQ captured attention QKV merged equals split and Python") {
  const char* fixture_dir = std::getenv("VLLM_CPP_GPTQ4_ORACLE_DIR");
  if (fixture_dir == nullptr) {
    MESSAGE("Set VLLM_CPP_GPTQ4_ORACLE_DIR for the captured QKV operation test");
    return;
  }
  const auto weights_file = vllm::SafetensorsFile::Open(
      std::string(fixture_dir) + "/attention_qkv_weights.safetensors");
  constexpr int64_t k = 5120;
  constexpr int64_t n = 14336;
  constexpr int64_t qn = 12288;
  constexpr int64_t kn = 1024;
  constexpr int64_t vn = 1024;
  vllm::Gptq4Weight merged;
  merged.k = k;
  merged.n = n;
  merged.qweight = vllm::dense_loaders::MakeOwned(vt::DType::kI32,
                                                   {n, k / 8});
  merged.scales = vllm::dense_loaders::MakeOwned(vt::DType::kF16,
                                                  {k / 128, n});
  merged.zero_point = vllm::dense_loaders::MakeOwned(vt::DType::kI8, {1});
  const auto& packed = weights_file.Get("qweight_nt_int32");
  const auto& scales = weights_file.Get("scales_f16");
  const auto& zero = weights_file.Get("effective_zero_point_i8");
  REQUIRE(packed.nbytes == merged.qweight.bytes.size());
  REQUIRE(scales.nbytes == merged.scales.bytes.size());
  REQUIRE(zero.nbytes == 1);
  std::memcpy(merged.qweight.bytes.data(), packed.data, packed.nbytes);
  std::memcpy(merged.scales.bytes.data(), scales.data, scales.nbytes);
  merged.zero_point.bytes.data()[0] = zero.data[0];

  const auto split = [&merged](int64_t begin, int64_t width) {
    vllm::Gptq4Weight part;
    part.k = k;
    part.n = width;
    part.qweight = vllm::dense_loaders::MakeOwned(vt::DType::kI32,
                                                   {width, k / 8});
    part.scales = vllm::dense_loaders::MakeOwned(vt::DType::kF16,
                                                  {k / 128, width});
    part.zero_point = vllm::dense_loaders::MakeOwned(vt::DType::kI8, {1});
    std::memcpy(part.qweight.bytes.data(),
                merged.qweight.bytes.data() + begin * (k / 8) * 4,
                part.qweight.bytes.size());
    for (int64_t group = 0; group < k / 128; ++group)
      std::memcpy(part.scales.bytes.data() + group * width * 2,
                  merged.scales.bytes.data() + (group * n + begin) * 2,
                  width * 2);
    part.zero_point.bytes.data()[0] = merged.zero_point.bytes.data()[0];
    return part;
  };
  auto qpart = split(0, qn);
  auto kpart = split(qn, kn);
  auto vpart = split(qn + kn, vn);
  QueueOwner owner;
  auto& queue = owner.queue;
  const auto merged_gpu = vllm::PrepareGptq4Resident(merged, queue);
  const auto qgpu = vllm::PrepareGptq4Resident(qpart, queue);
  const auto kgpu = vllm::PrepareGptq4Resident(kpart, queue);
  const auto vgpu = vllm::PrepareGptq4Resident(vpart, queue);
  for (const auto* fixture : {"attention_qkv_m1.safetensors",
                              "attention_qkv_m16.safetensors"}) {
    const auto sample = vllm::SafetensorsFile::Open(
        std::string(fixture_dir) + "/" + fixture);
    const auto& input = sample.Get("activation_fp16");
    const auto& expected = sample.Get("output_fp16");
    REQUIRE(input.shape.size() == 2);
    const int64_t m = input.shape[0];
    REQUIRE(input.shape[1] == k);
    REQUIRE(expected.shape.size() == 2);
    REQUIRE(expected.shape[0] == m);
    REQUIRE(expected.shape[1] == n);
    Buffer activation(queue, vt::DType::kF16, {m, k});
    Buffer combined(queue, vt::DType::kF16, {m, n});
    Buffer qout(queue, vt::DType::kF16, {m, qn});
    Buffer kout(queue, vt::DType::kF16, {m, kn});
    Buffer vout(queue, vt::DType::kF16, {m, vn});
    activation.Upload(input.data, input.nbytes);
    vt::MatmulGptq4W4A16(queue, combined.tensor, activation.tensor,
                         merged_gpu.qweight, merged_gpu.scales,
                         merged_gpu.zero_point, 128);
    vt::MatmulGptq4W4A16(queue, qout.tensor, activation.tensor,
                         qgpu.qweight, qgpu.scales, qgpu.zero_point, 128);
    vt::MatmulGptq4W4A16(queue, kout.tensor, activation.tensor,
                         kgpu.qweight, kgpu.scales, kgpu.zero_point, 128);
    vt::MatmulGptq4W4A16(queue, vout.tensor, activation.tensor,
                         vgpu.qweight, vgpu.scales, vgpu.zero_point, 128);
    vt::GetBackend(queue.device).Synchronize(queue);
    const auto merged_result = combined.Read();
    const auto q_result = qout.Read();
    const auto k_result = kout.Read();
    const auto v_result = vout.Read();
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        const float actual = ReadHalf(merged_result.data(), row * n + col);
        const float reference = ReadHalf(expected.data, row * n + col);
        const float separate = col < qn
            ? ReadHalf(q_result.data(), row * qn + col)
            : col < qn + kn
                ? ReadHalf(k_result.data(), row * kn + col - qn)
                : ReadHalf(v_result.data(), row * vn + col - qn - kn);
        const float tolerance = 0.02f + 0.01f * std::abs(reference);
        CHECK(std::abs(actual - reference) <= tolerance);
        CHECK(std::abs(actual - separate) <= tolerance);
      }
    }
  }
}

namespace {

void CheckCapturedTwoWayMerge(vt::Queue& queue, const std::string& directory,
                              const std::string& family,
                              int64_t first_width, int64_t second_width) {
  constexpr int64_t k = 5120;
  const int64_t n = first_width + second_width;
  const auto captured = vllm::SafetensorsFile::Open(
      directory + "/" + family + "_weights.safetensors");
  const auto& packed = captured.Get("qweight_nt_int32");
  const auto& scales = captured.Get("scales_f16");
  const auto& zero = captured.Get("effective_zero_point_i8");
  vllm::Gptq4Weight merged;
  merged.k = k;
  merged.n = n;
  merged.qweight = vllm::dense_loaders::MakeOwned(vt::DType::kI32,
                                                  {n, k / 8});
  merged.scales = vllm::dense_loaders::MakeOwned(vt::DType::kF16,
                                                 {k / 128, n});
  merged.zero_point = vllm::dense_loaders::MakeOwned(vt::DType::kI8, {1});
  REQUIRE(packed.nbytes == merged.qweight.bytes.size());
  REQUIRE(scales.nbytes == merged.scales.bytes.size());
  REQUIRE(zero.nbytes == 1);
  std::memcpy(merged.qweight.bytes.data(), packed.data, packed.nbytes);
  std::memcpy(merged.scales.bytes.data(), scales.data, scales.nbytes);
  merged.zero_point.bytes.data()[0] = zero.data[0];
  const auto slice = [&merged, n](int64_t begin, int64_t width) {
    vllm::Gptq4Weight part;
    part.k = k;
    part.n = width;
    part.qweight = vllm::dense_loaders::MakeOwned(vt::DType::kI32,
                                                  {width, k / 8});
    part.scales = vllm::dense_loaders::MakeOwned(vt::DType::kF16,
                                                 {k / 128, width});
    part.zero_point = vllm::dense_loaders::MakeOwned(vt::DType::kI8, {1});
    std::memcpy(part.qweight.bytes.data(),
                merged.qweight.bytes.data() + begin * (k / 8) * 4,
                part.qweight.bytes.size());
    for (int64_t group = 0; group < k / 128; ++group)
      std::memcpy(part.scales.bytes.data() + group * width * 2,
                  merged.scales.bytes.data() + (group * n + begin) * 2,
                  width * 2);
    part.zero_point.bytes.data()[0] = merged.zero_point.bytes.data()[0];
    return part;
  };
  auto first = slice(0, first_width);
  auto second = slice(first_width, second_width);
  const auto merged_gpu = vllm::PrepareGptq4Resident(merged, queue);
  const auto first_gpu = vllm::PrepareGptq4Resident(first, queue);
  const auto second_gpu = vllm::PrepareGptq4Resident(second, queue);
  CHECK(merged.ResidentBytes() == first.ResidentBytes() +
                                    second.ResidentBytes() - 1);
  CHECK(!merged.qweight.HasHostBytes());
  for (const char* batch : {"m1", "m16"}) {
    const auto sample = vllm::SafetensorsFile::Open(
        directory + "/" + family + "_" + batch + ".safetensors");
    const auto& input = sample.Get("activation_fp16");
    const auto& expected = sample.Get("output_fp16");
    REQUIRE(input.shape.size() == 2);
    const int64_t m = input.shape[0];
    REQUIRE(input.shape[1] == k);
    REQUIRE(expected.shape.size() == 2);
    REQUIRE(expected.shape[0] == m);
    REQUIRE(expected.shape[1] == n);
    Buffer activation(queue, vt::DType::kF16, {m, k});
    Buffer combined(queue, vt::DType::kF16, {m, n});
    Buffer first_out(queue, vt::DType::kF16, {m, first_width});
    Buffer second_out(queue, vt::DType::kF16, {m, second_width});
    activation.Upload(input.data, input.nbytes);
    vt::MatmulGptq4W4A16(queue, combined.tensor, activation.tensor,
                         merged_gpu.qweight, merged_gpu.scales,
                         merged_gpu.zero_point, 128);
    vt::MatmulGptq4W4A16(queue, first_out.tensor, activation.tensor,
                         first_gpu.qweight, first_gpu.scales,
                         first_gpu.zero_point, 128);
    vt::MatmulGptq4W4A16(queue, second_out.tensor, activation.tensor,
                         second_gpu.qweight, second_gpu.scales,
                         second_gpu.zero_point, 128);
    vt::GetBackend(queue.device).Synchronize(queue);
    const auto actual = combined.Read();
    const auto first_result = first_out.Read();
    const auto second_result = second_out.Read();
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        const float reference = ReadHalf(expected.data, row * n + col);
        const float combined_value = ReadHalf(actual.data(), row * n + col);
        const float separate = col < first_width
            ? ReadHalf(first_result.data(), row * first_width + col)
            : ReadHalf(second_result.data(),
                       row * second_width + col - first_width);
        const float tolerance = 0.02f + 0.01f * std::abs(reference);
        CHECK(std::abs(combined_value - reference) <= tolerance);
        CHECK(std::abs(combined_value - separate) <= tolerance);
      }
    }
  }
}

}  // namespace

TEST_CASE("XPU GPTQ captured GDN and MLP merges equal split and Python") {
  const char* directory = std::getenv("VLLM_CPP_GPTQ4_ORACLE_DIR");
  if (directory == nullptr) {
    MESSAGE("Set VLLM_CPP_GPTQ4_ORACLE_DIR for captured merge tests");
    return;
  }
  QueueOwner owner;
  CheckCapturedTwoWayMerge(owner.queue, directory, "gdn_qkvz", 10240, 6144);
  CheckCapturedTwoWayMerge(owner.queue, directory, "mlp_gate_up", 17408, 17408);
}

TEST_CASE("XPU GPTQ full checkpoint streams one resident per merge") {
  const char* checkpoint = std::getenv("VLLM_CPP_GPTQ4_CHECKPOINT_DIR");
  if (checkpoint == nullptr) {
    MESSAGE("Set VLLM_CPP_GPTQ4_CHECKPOINT_DIR for full XPU staging");
    return;
  }
  QueueOwner owner;
  const auto config = vllm::LoadHfConfig(
      std::string(checkpoint) + "/config.json");
  std::vector<vllm::SafetensorsFile> shards;
  for (int i = 1; i <= 5; ++i)
    shards.push_back(vllm::SafetensorsFile::Open(
        std::string(checkpoint) + "/model-0000" + std::to_string(i) +
        "-of-00005.safetensors"));
  const auto model = vllm::LoadQwen3_5Dense(shards, config, &owner.queue);
  REQUIRE(model.gptq4_checkpoint);
  REQUIRE(model.layers.size() == 64);
  size_t resident_bytes = 0;
  for (const auto& layer : model.layers) {
    resident_bytes += layer.gptq4.ResidentBytes();
    const auto check = [](const vllm::Gptq4Weight& weight) {
      if (weight.k == 0) return;
      CHECK(weight.resident);
      CHECK(weight.qweight.d_dev != nullptr);
      CHECK(weight.scales.d_dev != nullptr);
      CHECK(weight.zero_point.d_dev != nullptr);
      CHECK(!weight.qweight.HasHostBytes());
      CHECK(!weight.scales.HasHostBytes());
      CHECK(!weight.zero_point.HasHostBytes());
    };
    check(layer.gptq4.gdn_qkvz);
    check(layer.gptq4.gdn_out);
    check(layer.gptq4.attn_qkv);
    check(layer.gptq4.attn_out);
    check(layer.gptq4.mlp_gate_up);
    check(layer.gptq4.mlp_down);
  }
  CHECK(resident_bytes > 10ULL * 1024 * 1024 * 1024);
  CHECK(resident_bytes < 20ULL * 1024 * 1024 * 1024);
  CHECK(model.Gptq4ResidentBytes() == resident_bytes);
  const size_t uploaded = vllm::load_stats::Snapshot().device_upload_bytes;
  for (const auto& layer : model.layers) layer.gptq4.PrepareResident(owner.queue);
  CHECK(vllm::load_stats::Snapshot().device_upload_bytes == uploaded);
  CHECK(model.lm_head.dtype == vt::DType::kF16);
  CHECK(model.embed_tokens.dtype == vt::DType::kF16);
}
