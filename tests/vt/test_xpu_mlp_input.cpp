#include "vt/exl3_w8a8_initialization_policy.h"
#include "xpu_test_helpers.h"
#include "vt/breakable_graph.h"
#include "vt/xpu.h"
#include "vt/xpu/xpu_common.h"
#include "vt/xpu/xpu_silu_input.h"
#include "vt/xpu/xpu_exl3_prepare.h"
#include "vt/exl3_grouped.h"
#include "vt/unaligned.h"
#include <iostream>
#include <string>

namespace {
using xpu_test::Buffer;
using xpu_test::Queue;
using vt::DType;
struct Env {
  const char* key;
  bool had;
  std::string value;
  Env(const char* name, const char* setting) : key(name), had(std::getenv(name)),
      value(had ? std::getenv(name) : "") { setenv(key, setting, 1); }
  ~Env() { if (had) setenv(key, value.c_str(), 1); else unsetenv(key); }
};
struct ProducerProbe {
  const sycl::half* input;
  const uint16_t* table;
  sycl::half* output;
  int32_t* valid;
  int width;
  void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
    using namespace sycl::ext::intel::esimd;
    const size_t tile = item.get_global_linear_id();
    const size_t row = tile / (width / 128), column = tile % (width / 128) * 128;
    const auto block = vt::xpu::detail::LoadSiluMulFp16(
        input, row * (2 * width) + column, width, table);
    block_store<sycl::half, 128>(output + row * width + column, block.values);
    simd<int32_t, 128> flags = 0;
    flags.merge(1, block.finite);
    block_store<int32_t, 128>(valid + row * width + column, flags);
  }
};
void Produce(vt::Queue& q, Buffer& out, Buffer& valid, const Buffer& in,
             int rows, int width) {
  vt::xpu::RecordGraphWrite(q, out.tensor.data, out.bytes);
  vt::xpu::RecordGraphWrite(q, valid.tensor.data, valid.bytes);
  const bool ready = vt::xpu::WithFp16SiluValues(q, [&](const void* table) {
    ProducerProbe kernel{static_cast<const sycl::half*>(in.tensor.data),
        static_cast<const uint16_t*>(table), static_cast<sycl::half*>(out.tensor.data),
        static_cast<int32_t*>(valid.tensor.data), width};
    vt::xpu::NativeQueue(q).parallel_for(
        sycl::nd_range<1>(size_t(rows) * (width / 128), 1), kernel);
  });
  REQUIRE(ready);
}
uint16_t Word(const std::vector<unsigned char>& raw, size_t index) {
  uint16_t value; std::memcpy(&value, raw.data() + 2 * index, 2); return value;
}
void Compare(const std::vector<unsigned char>& got,
             const std::vector<unsigned char>& valid,
             const std::vector<unsigned char>& reference, size_t count) {
  size_t differences = 0, validity_errors = 0;
  for (size_t i = 0; i < count; ++i) {
    int32_t flag; std::memcpy(&flag, valid.data() + 4 * i, 4);
    const uint16_t expected = Word(reference, i);
    const bool finite = (expected & 0x7c00u) != 0x7c00u;
    validity_errors += flag != int(finite);
    differences += Word(got, i) != (finite ? expected : uint16_t{0});
  }
  CHECK(validity_errors == 0);
  CHECK(differences == 0);
}
}

TEST_CASE("XPU MLP input producer: exhaustive FP16 checkpoints and checked invalid lanes") {
  Queue gpu(vt::DeviceType::kXPU);
  Env typed("VT_XPU_SILU_FP16_TYPED", "1"), table("VT_XPU_SILU_FP16_TABLE", "0");
  constexpr int rows = 64, width = 1024, count = rows * width;
  Buffer input(gpu.q, DType::kF16, {rows, 2 * width});
  Buffer reference(gpu.q, DType::kF16, {rows, width});
  Buffer output(gpu.q, DType::kF16, {count + 128});
  Buffer valid(gpu.q, DType::kI32, {count + 32});
  const std::vector<unsigned char> poison(output.bytes, 0xa5), vp(valid.bytes, 0x5a);
  std::vector<uint16_t> values(2 * count);
  for (uint16_t up : {uint16_t{0}, uint16_t{0x8000}, uint16_t{1}, uint16_t{0x3c00},
       uint16_t{0xbc00}, vt::F32ToF16(-0.346923828125f), uint16_t{0x7bff},
       uint16_t{0xfbff}, uint16_t{0x7c00}, uint16_t{0xfc00}, uint16_t{0x7e35}}) {
    CAPTURE(up);
    for (int i = 0; i < count; ++i) {
      values[(i / width) * 2 * width + i % width] = uint16_t(i);
      values[(i / width) * 2 * width + width + i % width] = up;
    }
    input.upload(values.data());
    const auto before = input.download();
    vt::SiluAndMul(gpu.q, reference.tensor, input.tensor);
    output.upload(poison.data()); valid.upload(vp.data());
    Produce(gpu.q, output, valid, input, rows, width);
    const auto got = output.download(), flags = valid.download();
    Compare(got, flags, reference.download(), count);
    CHECK(std::equal(got.begin() + count * 2, got.end(), poison.begin() + count * 2));
    CHECK(std::equal(flags.begin() + count * 4, flags.end(), vp.begin() + count * 4));
    xpu_test::SameBytes(input.download(), before);
  }
  CHECK(vt::xpu::GetMemoryInfo().fp16_silu_table_bytes == 131072);
  std::cout << "MLP_INPUT_PRODUCER patterns=65536 up_patterns=11 finite_x_bits_exact=1 invalid_private_zero=1 input_canaries=1\n";
}

TEST_CASE("XPU MLP input producer: shared immutable table independent graph queues") {
  Queue first(vt::DeviceType::kXPU), second(vt::DeviceType::kXPU);
  Env typed("VT_XPU_SILU_FP16_TYPED", "1"), table("VT_XPU_SILU_FP16_TABLE", "0");
  constexpr int rows = 64, width = 1024, count = rows * width;
  Buffer a(first.q, DType::kF16, {rows, 2 * width}),
      b(second.q, DType::kF16, {rows, 2 * width}),
      oa(first.q, DType::kF16, {rows, width}),
      ob(second.q, DType::kF16, {rows, width}),
      va(first.q, DType::kI32, {rows, width}),
      vb(second.q, DType::kI32, {rows, width}),
      reference(first.q, DType::kF16, {rows, width});
  std::vector<uint16_t> values(2 * count);
  const auto fill = [&](int shift) {
    for (int i = 0; i < count; ++i) {
      values[(i / width) * 2 * width + i % width] = uint16_t(i + shift);
      values[(i / width) * 2 * width + width + i % width] = vt::F32ToF16(0.3125f);
    }
    a.upload(values.data()); b.upload(values.data());
  };
  fill(0); Produce(first.q, oa, va, a, rows, width);
  Produce(second.q, ob, vb, b, rows, width);
  auto& backend = vt::GetBackend(first.q.device); backend.Synchronize(first.q);
  backend.Synchronize(second.q); REQUIRE(backend.SupportsGraphCapture());
  vt::BreakableGraph ga, gb;
  { vt::GraphCaptureScope capture(backend, first.q, ga, vt::GraphCaptureMode::kFull);
    Produce(first.q, oa, va, a, rows, width); }
  { vt::GraphCaptureScope capture(backend, second.q, gb, vt::GraphCaptureMode::kFull);
    Produce(second.q, ob, vb, b, rows, width); }
  REQUIRE(ga.captured()); REQUIRE(gb.captured());
  const auto warm = vt::xpu::GetMemoryInfo();
  for (int shift : {17, 4097, 33521}) {
    fill(shift); vt::SiluAndMul(first.q, reference.tensor, a.tensor);
    const auto expected = reference.download();
    gb.Replay(second.q); ga.Replay(first.q);
    Compare(oa.download(), va.download(), expected, count);
    Compare(ob.download(), vb.download(), expected, count);
  }
  const auto end = vt::xpu::GetMemoryInfo();
  CHECK(end.fp16_silu_table_bytes == 131072);
  CHECK(end.allocated_bytes == warm.allocated_bytes);
  CHECK(end.pinned_bytes == warm.pinned_bytes);
  CHECK(end.graph_count == warm.graph_count);
}

namespace {
struct PreparationSnapshot {
  sycl::half* activation;
  sycl::half* product;
  sycl::half* transformed;
  ESIMD_INLINE void Record(const ::exl3::HadInQ8Kernel<sycl::half>& input,
      int row, int group, int block,
      const sycl::ext::intel::esimd::simd<sycl::half, 128>& x,
      const sycl::ext::intel::esimd::simd<sycl::half, 128>& p,
      const sycl::ext::intel::esimd::simd<sycl::half, 128>& h) const SYCL_ESIMD_KERNEL {
    using namespace sycl::ext::intel::esimd;
    const size_t offset = (size_t(group) * input.M + row) * input.Kdim + block * 128;
    block_store<sycl::half, 128>(activation + offset, x);
    block_store<sycl::half, 128>(product + offset, p);
    block_store<sycl::half, 128>(transformed + offset, h);
  }
};
struct PreparedProbe {
  vt::Queue& q;
  int rows, k, groups, padded;
  Buffer quant, scales, status, x, product, transformed;
  PreparedProbe(vt::Queue& queue, int m, int width, int count)
      : q(queue), rows(m), k(width), groups(count), padded((m + 255) / 256 * 256),
        quant(q, DType::kI8, {groups, padded, k}),
        scales(q, DType::kF32, {groups, padded}), status(q, DType::kI32, {1}),
        x(q, DType::kF16, {groups, rows, k}),
        product(q, DType::kF16, {groups, rows, k}),
        transformed(q, DType::kF16, {groups, rows, k}) {}
  void Run(const Buffer& input, const Buffer& suh, bool fused, bool observe = true, bool skip_full_init = false) {
    std::vector<unsigned char> zero(quant.bytes, skip_full_init ? 0xa5 : 0); quant.upload(zero.data());
    scales.put(std::vector<float>(size_t(groups) * padded, skip_full_init ? -987.0f : 1.0f));
    int32_t one = 1; status.upload(&one);
    for (Buffer* b : {&x, &product, &transformed}) {
      std::vector<unsigned char> poison(b->bytes, 0xa5); b->upload(poison.data());
    }
    ::exl3::HadInQ8Kernel<sycl::half> args{
        static_cast<const sycl::half*>(input.tensor.data),
        static_cast<const sycl::half*>(suh.tensor.data),
        static_cast<int8_t*>(quant.tensor.data), static_cast<float*>(scales.tensor.data),
        rows, k, groups, fused ? 2 * k : k, padded};
    if (skip_full_init) vt::xpu::detail::InitializePreparedRows(q,args,quant.bytes,true);
    PreparationSnapshot observer{static_cast<sycl::half*>(x.tensor.data),
        static_cast<sycl::half*>(product.tensor.data),
        static_cast<sycl::half*>(transformed.tensor.data)};
    auto* flag = static_cast<uint32_t*>(status.tensor.data);
    const auto launch = [&](const uint16_t* table) {
      const auto submit = [&]<int Threads, int Blocks>() {
        if (!fused && !observe) {
          const vt::xpu::detail::PrepareRows<Threads, Blocks> kernel{args, flag};
          vt::xpu::NativeQueue(q).parallel_for(
              sycl::nd_range<1>(size_t(groups) * rows * Threads, Threads), kernel);
        } else if (fused && !observe) {
          const vt::xpu::detail::PrepareRows<Threads, Blocks,
              vt::xpu::detail::SiluMulActivationInput>
              kernel{args, flag, {table}};
          vt::xpu::NativeQueue(q).parallel_for(
              sycl::nd_range<1>(size_t(groups) * rows * Threads, Threads), kernel);
        } else if (fused) {
          const vt::xpu::detail::PrepareRows<Threads, Blocks,
              vt::xpu::detail::SiluMulActivationInput, PreparationSnapshot>
              kernel{args, flag, {table}, observer};
          vt::xpu::NativeQueue(q).parallel_for(
              sycl::nd_range<1>(size_t(groups) * rows * Threads, Threads), kernel);
        } else {
          const vt::xpu::detail::PrepareRows<Threads, Blocks,
              vt::xpu::detail::TensorActivationInput, PreparationSnapshot>
              kernel{args, flag, {}, observer};
          vt::xpu::NativeQueue(q).parallel_for(
              sycl::nd_range<1>(size_t(groups) * rows * Threads, Threads), kernel);
        }
      };
      if (k / 128 <= 48) submit.template operator()<8, 6>();
      else submit.template operator()<16, 9>();
    };
    if (fused) REQUIRE(vt::xpu::WithFp16SiluValues(q,
        [&](const void* table) { launch(static_cast<const uint16_t*>(table)); }));
    else launch(nullptr);
  }
};
}

TEST_CASE("XPU MLP preparation: exact materialized checkpoints INT8 scales and real operator") {
  Queue gpu(vt::DeviceType::kXPU);
  Env typed("VT_XPU_SILU_FP16_TYPED", "1"), table("VT_XPU_SILU_FP16_TABLE", "0"),
      prepare("VT_XPU_W8A8_PREPARE", "1");
  for (const auto shape : {std::pair{129, 128}, std::pair{255, 5120},
                            std::pair{256, 17408}, std::pair{257, 17408}}) {
    const auto [rows, k] = shape;
    CAPTURE(rows);
    CAPTURE(k);
    constexpr int groups = 2, n = 256;
    Buffer gu(gpu.q, DType::kF16, {rows, 2 * k}), act(gpu.q, DType::kF16, {rows, k}),
        suh(gpu.q, DType::kF16, {groups, k}), svh(gpu.q, DType::kF16, {n}),
        packed(gpu.q, DType::kI8, {k / 16, n / 16, 128}),
        map(gpu.q, DType::kI32, {n / 128}), out(gpu.q, DType::kF16, {rows, n});
    std::vector<uint16_t> values(size_t(rows) * 2 * k);
    for (int r = 0; r < rows; ++r) for (int c = 0; c < k; ++c) {
      values[size_t(r) * 2 * k + c] = vt::F32ToF16((int((r * 13 + c * 7) % 257) - 128) * 0.0625f);
      values[size_t(r) * 2 * k + k + c] = vt::F32ToF16((int((r * 7 + c * 11) % 251) - 125) * 0.0078125f);
    }
    gu.upload(values.data());
    std::vector<float> signs(size_t(groups) * k);
    for (int g = 0; g < groups; ++g) for (int c = 0; c < k; ++c)
      signs[size_t(g) * k + c] = (c + g) % 3 ? 1.125f : -0.75f;
    suh.put(signs); svh.put(std::vector<float>(n, 1.0f));
    const int32_t mapping[] = {0, 1}; map.upload(mapping);
    std::vector<unsigned char> packed_zero(packed.bytes, 0); packed.upload(packed_zero.data());
    vt::SiluAndMul(gpu.q, act.tensor, gu.tensor);
    PreparedProbe materialized(gpu.q, rows, k, groups), fused(gpu.q, rows, k, groups);
    materialized.Run(act, suh, false); fused.Run(gu, suh, true);
    xpu_test::SameBytes(fused.x.download(), materialized.x.download());
    xpu_test::SameBytes(fused.product.download(), materialized.product.download());
    xpu_test::SameBytes(fused.transformed.download(), materialized.transformed.download());
    xpu_test::SameBytes(fused.quant.download(), materialized.quant.download());
    xpu_test::SameBytes(fused.scales.download(), materialized.scales.download());
    REQUIRE(vt::LoadUnaligned<uint32_t>(fused.status.download().data()) == 1);
    // The actual producer specialization has NO checkpoint stores. Verify it
    // too: diagnostic stores must not be what forces the required narrowing.
    const auto observed_quant = fused.quant.download(), observed_scales = fused.scales.download();
    fused.Run(gu, suh, true, false);
    xpu_test::SameBytes(fused.quant.download(), observed_quant);
    xpu_test::SameBytes(fused.scales.download(), observed_scales);
    REQUIRE(vt::LoadUnaligned<uint32_t>(fused.status.download().data()) == 1);
    const auto plan = vt::PlanExl3W8A8(rows, k, n, groups, 4);
    Buffer workspace(gpu.q, DType::kI8, {int64_t(plan.workspace_bytes)}),
        panel(gpu.q, DType::kI8, {k, plan.weight_panel_columns});
    vt::Exl3GroupedW8A8(gpu.q, out.tensor, act.tensor, packed.tensor,
        suh.tensor, svh.tensor, map.tensor, workspace.tensor, panel.tensor, {4, 2});
    const auto published = workspace.download();
    const auto qb = fused.quant.download(), sb = fused.scales.download();
    CHECK(std::equal(qb.begin(), qb.end(), published.begin() + plan.activation_offset));
    CHECK(std::equal(sb.begin(), sb.end(), published.begin() + plan.row_scale_offset));
    const auto old_out = out.download(), old_gu = gu.download();
    // The final up lane is invalid: both checked producers must reject it.
    values.back() = 0x7c00u; gu.upload(values.data());
    vt::SiluAndMul(gpu.q, act.tensor, gu.tensor);
    fused.Run(gu, suh, true);
    REQUIRE(vt::LoadUnaligned<uint32_t>(fused.status.download().data()) == 0);
    CHECK_THROWS_WITH_AS(vt::Exl3GroupedW8A8(gpu.q, out.tensor, act.tensor,
        packed.tensor, suh.tensor, svh.tensor, map.tensor, workspace.tensor,
        panel.tensor, {4, 2}), doctest::Contains("finite inputs"), std::runtime_error);
    xpu_test::SameBytes(out.download(), old_out);
    values.back() = vt::LoadUnaligned<uint16_t>(old_gu.data() + old_gu.size() - 2);
    gu.upload(values.data()); xpu_test::SameBytes(gu.download(), old_gu);
  }
}

TEST_CASE("XPU MLP fused W8A8: complete outputs guards refusal and managed graph contract") {
  Queue gpu(vt::DeviceType::kXPU);
  Env typed("VT_XPU_SILU_FP16_TYPED", "1"), table("VT_XPU_SILU_FP16_TABLE", "0"),
      prepare("VT_XPU_W8A8_PREPARE", "1"), direct("VT_XPU_W8A8_DIRECT_PREPARE", "1"),
      metadata("VT_XPU_W8A8_MODEL_MAP", "1");
  for (const auto shape : {std::pair{129, 128}, std::pair{256, 128}, std::pair{257, 17408}}) {
    const auto [rows, k] = shape;
    CAPTURE(rows);
    CAPTURE(k);
    constexpr int groups = 2, n = 256;
    Buffer gu(gpu.q, DType::kF16, {rows, 2 * k}), act(gpu.q, DType::kF16, {rows, k}),
        suh(gpu.q, DType::kF16, {groups, k}), svh(gpu.q, DType::kF16, {n}),
        packed(gpu.q, DType::kI8, {k / 16, n / 16, 128}),
        expected(gpu.q, DType::kF16, {rows, n}), actual(gpu.q, DType::kF16, {rows, n});
    auto& backend = vt::GetBackend(gpu.q.device);
    auto owner = std::shared_ptr<void>(vt::Alloc(gpu.q.device, 8),
        [device = gpu.q.device](void* p) { vt::Free(device, p); });
    auto map = vt::Tensor::Contiguous(owner.get(), DType::kI32, gpu.q.device, {2});
    const int32_t ids[] = {0, 1}; backend.Copy(gpu.q, map.data, ids, sizeof(ids));
    backend.Synchronize(gpu.q);
    vt::SharedPtrCache<const vt::Exl3W8A8ModelMap> cache;
    std::vector<uint16_t> values(size_t(rows) * 2 * k);
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = vt::F32ToF16((int((i * 37) % 257) - 128) * 0.015625f);
    gu.upload(values.data());
    std::vector<float> signs(size_t(groups) * k);
    for (size_t i = 0; i < signs.size(); ++i) signs[i] = i % 3 ? 1.125f : -0.75f;
    suh.put(signs); svh.put(std::vector<float>(n, 1.0f));
    std::vector<unsigned char> zeros(packed.bytes, 0); packed.upload(zeros.data());
    const auto before_input = gu.download();
    vt::SiluAndMul(gpu.q, act.tensor, gu.tensor);
    vt::detail::Exl3GroupedW8A8Model(gpu.q, expected.tensor, act.tensor,
        packed.tensor, suh.tensor, svh.tensor, map, {4, 2}, owner, cache);
    const auto want = expected.download();
    const auto run = [&] {
      return vt::detail::TryExl3GroupedW8A8SiluModel(gpu.q, actual.tensor, gu.tensor,
          packed.tensor, suh.tensor, svh.tensor, map, {4, 2}, owner, cache);
    };
    for (const char* direct_mode : {"1", "0"}) for (const char* map_mode : {"1", "0"}) {
      setenv("VT_XPU_W8A8_DIRECT_PREPARE", direct_mode, 1);
      setenv("VT_XPU_W8A8_MODEL_MAP", map_mode, 1);
      REQUIRE(run()); xpu_test::SameBytes(actual.download(), want);
    }
    setenv("VT_XPU_W8A8_DIRECT_PREPARE", "1", 1);
    setenv("VT_XPU_W8A8_MODEL_MAP", "1", 1);
    const auto warm = vt::xpu::GetMemoryInfo();
    for (int repeat = 0; repeat < 2; ++repeat) {
      REQUIRE(run()); xpu_test::SameBytes(actual.download(), want);
    }
    const auto steady = vt::xpu::GetMemoryInfo();
    CHECK(steady.allocated_bytes == warm.allocated_bytes);
    CHECK(steady.w8a8_preparation_bytes == warm.w8a8_preparation_bytes);
    CHECK(steady.fp16_silu_table_bytes == 131072);
    setenv("VT_XPU_W8A8_PREPARE", "0", 1);
    CHECK_FALSE(run()); xpu_test::SameBytes(actual.download(), want);
    setenv("VT_XPU_W8A8_PREPARE", "1", 1);
    auto bad = gu.tensor; bad.stride[0] -= 1;
    CHECK_THROWS(vt::detail::TryExl3GroupedW8A8SiluModel(gpu.q, actual.tensor,
        bad, packed.tensor, suh.tensor, svh.tensor, map, {4, 2}, owner, cache));
    bad = actual.tensor; bad.data = gu.tensor.data;
    CHECK_THROWS(vt::detail::TryExl3GroupedW8A8SiluModel(gpu.q, bad,
        gu.tensor, packed.tensor, suh.tensor, svh.tensor, map, {4, 2}, owner, cache));
    xpu_test::SameBytes(gu.download(), before_input);
    values.back() = 0x7e35; gu.upload(values.data());
    CHECK_THROWS_WITH_AS(run(), doctest::Contains("finite inputs"), std::runtime_error);
    xpu_test::SameBytes(actual.download(), want);
    values.back() = vt::LoadUnaligned<uint16_t>(before_input.data() + before_input.size() - 2);
    gu.upload(values.data());
    auto bad_scale = std::vector<float>(n, 1.0f);
    bad_scale.back() = vt::F16ToF32(0x7c00); svh.put(bad_scale);
    CHECK_THROWS_WITH_AS(run(), doctest::Contains("finite svh"), std::runtime_error);
    xpu_test::SameBytes(actual.download(), want);
    svh.put(std::vector<float>(n, 1.0f));
    if (backend.SupportsGraphCapture()) {
      backend.BeginCapture(gpu.q);
      CHECK_THROWS_WITH_AS(run(), doctest::Contains("eager-only"), std::runtime_error);
      void* graph = backend.EndCaptureGraph(gpu.q); backend.DestroyGraph(graph);
      xpu_test::SameBytes(actual.download(), want);
    }
    REQUIRE(run()); xpu_test::SameBytes(actual.download(), want);
    // Public tensor entry points discard private producer payloads.
    vt::Exl3GroupedLinearArgs args{4, 2};
    args.input_producer = vt::Exl3W8A8InputProducer::kSiluMul;
    vt::Exl3GroupedW8A8(gpu.q, actual.tensor, act.tensor, packed.tensor,
        suh.tensor, svh.tensor, map, args);
    xpu_test::SameBytes(actual.download(), want);
  }
}

TEST_CASE("XPU MLP fused W8A8: oversized private preparation refuses without publication") {
  Queue gpu(vt::DeviceType::kXPU);
  Env prepare("VT_XPU_W8A8_PREPARE", "1");
  constexpr int rows = 4096, k = 17408, groups = 2, n = 256;
  Buffer gu(gpu.q, DType::kF16, {rows, 2 * k}),
      suh(gpu.q, DType::kF16, {groups, k}), svh(gpu.q, DType::kF16, {n}),
      packed(gpu.q, DType::kI8, {k / 16, n / 16, 128}),
      out(gpu.q, DType::kF16, {rows, n});
  auto& backend = vt::GetBackend(gpu.q.device);
  auto owner = std::shared_ptr<void>(vt::Alloc(gpu.q.device, 8),
      [device = gpu.q.device](void* p) { vt::Free(device, p); });
  auto map = vt::Tensor::Contiguous(owner.get(), DType::kI32, gpu.q.device, {2});
  const int32_t ids[] = {0, 1}; backend.Copy(gpu.q, map.data, ids, sizeof(ids));
  backend.Synchronize(gpu.q);
  std::vector<unsigned char> poison(out.bytes, 0xa5); out.upload(poison.data());
  vt::SharedPtrCache<const vt::Exl3W8A8ModelMap> cache;
  const auto before = vt::xpu::GetMemoryInfo();
  CHECK_FALSE(vt::detail::TryExl3GroupedW8A8SiluModel(gpu.q, out.tensor, gu.tensor,
      packed.tensor, suh.tensor, svh.tensor, map, {4, 2}, owner, cache));
  xpu_test::SameBytes(out.download(), poison);
  const auto after = vt::xpu::GetMemoryInfo();
  CHECK(after.w8a8_preparation_bytes == before.w8a8_preparation_bytes);
  CHECK(after.w8a8_workspace_bytes == before.w8a8_workspace_bytes);
}

#include "vt/xpu_mlp_policy.h"
TEST_CASE("XPU MLP prepared policy: long exact geometry optin and small fallback") {
  const char* key = "VT_XPU_MLP_SILU_PREPARE";
  Env flag(key, "0");
  unsetenv(key); CHECK(vt::XpuMlpSiluPrepare(4096, 17408, 5120));
  setenv(key, "0", 1); CHECK_FALSE(vt::XpuMlpSiluPrepare(4096, 17408, 5120));
  setenv(key, "1", 1);
  CHECK(vt::XpuMlpSiluPrepare(512, 17408, 5120));
  CHECK(vt::XpuMlpSiluPrepare(4096, 17408, 5120));
  for (int rows : {0, 1, 4, 128, 129, 511, 4097})
    CHECK_FALSE(vt::XpuMlpSiluPrepare(rows, 17408, 5120));
  CHECK_FALSE(vt::XpuMlpSiluPrepare(4096, 17407, 5120));
  CHECK_FALSE(vt::XpuMlpSiluPrepare(4096, 17408, 5121));
  for (const char* invalid : {"", "2", "01", " 1", "1x"}) {
    setenv(key, invalid, 1);
    CHECK_THROWS(vt::XpuMlpSiluPrepare(4096, 17408, 5120));
    CHECK_FALSE(vt::XpuMlpSiluPrepare(1, 17408, 5120));
    CHECK_FALSE(vt::XpuMlpSiluPrepare(4, 17408, 5120));
  }
}

TEST_CASE("XPU preparation full-init removal: poisoned complete rows and untouched partial contract") {
  Queue gpu(vt::DeviceType::kXPU);
  for (int rows : {255,256,257,3839,3840,3841,4095,4096})
  for (int k : {128,5120,17408}) for (int groups : {1,2,3,4}) {
    CAPTURE(rows);
    CAPTURE(k);
    CAPTURE(groups);
    Buffer gu(gpu.q,DType::kF16,{rows,2*k}),act(gpu.q,DType::kF16,{rows,k}),suh(gpu.q,DType::kF16,{groups,k});
    std::vector<uint16_t> values(size_t(rows)*2*k);
    for(size_t i=0;i<values.size();++i)values[i]=vt::F32ToF16((int(i%251)-125)*.015625f);
    gu.upload(values.data());vt::SiluAndMul(gpu.q,act.tensor,gu.tensor);
    suh.put(xpu_test::Values(size_t(groups)*k,7,.125f));
    PreparedProbe baseline(gpu.q,rows,k,groups),candidate(gpu.q,rows,k,groups);
    baseline.Run(act,suh,false,false);candidate.Run(act,suh,false,false,true);
    xpu_test::SameBytes(candidate.quant.download(),baseline.quant.download());
    xpu_test::SameBytes(candidate.scales.download(),baseline.scales.download());
    xpu_test::SameBytes(candidate.status.download(),baseline.status.download());
    // The fused producer has the identical complete-row write geometry.
    candidate.Run(gu,suh,true,false,true);
    xpu_test::SameBytes(candidate.quant.download(),baseline.quant.download());
    xpu_test::SameBytes(candidate.scales.download(),baseline.scales.download());
    values.back()=0x7c00;gu.upload(values.data());candidate.Run(gu,suh,true,false,true);
    CHECK(vt::LoadUnaligned<uint32_t>(candidate.status.download().data())==0);
  }
}
TEST_CASE("XPU preparation full-init policy: exact full geometry only") {
  Env flag("VT_XPU_W8A8_SKIP_FULL_INIT","0");unsetenv(flag.key);
  CHECK_FALSE(vt::Exl3W8A8SkipFullInitialization(4096,4096));setenv(flag.key,"1",1);
  CHECK(vt::Exl3W8A8SkipFullInitialization(256,256));CHECK(vt::Exl3W8A8SkipFullInitialization(4096,4096));
  CHECK_FALSE(vt::Exl3W8A8SkipFullInitialization(255,256));CHECK_FALSE(vt::Exl3W8A8SkipFullInitialization(4095,4096));
  setenv(flag.key,"bad",1);CHECK_THROWS(vt::Exl3W8A8SkipFullInitialization(4096,4096));CHECK_FALSE(vt::Exl3W8A8SkipFullInitialization(255,256));
}

TEST_CASE("XPU preparation full-init removal: public workspace panel output atomicity") {
  Queue gpu(vt::DeviceType::kXPU);
  Env flag("VT_XPU_W8A8_SKIP_FULL_INIT","1"),prepare("VT_XPU_W8A8_PREPARE","1");
  for (const char* skip : {"0","1"}) for (const char* checked : {"0","1"}) {
  setenv(flag.key,skip,1);setenv(prepare.key,checked,1);
  CAPTURE(skip);
  CAPTURE(checked);
  constexpr int m=256,k=128,n=256,groups=2;
  const auto plan=vt::PlanExl3W8A8(m,k,n,groups,4,128);
  Buffer input(gpu.q,DType::kF16,{m,k}),suh(gpu.q,DType::kF16,{groups,k}),svh(gpu.q,DType::kF16,{n}),map(gpu.q,DType::kI32,{n/128}),packed(gpu.q,DType::kI8,{k/16,n/16,128}),out(gpu.q,DType::kF16,{m,n}),workspace(gpu.q,DType::kI8,{int64_t(plan.workspace_bytes)}),panel(gpu.q,DType::kI8,{k,plan.weight_panel_columns});
  input.put(xpu_test::Values(m*k,3,.125f));suh.put(xpu_test::Values(groups*k,7,.125f));svh.put(std::vector<float>(n,1.f));const int32_t ids[]={0,1};map.upload(ids);packed.upload(std::vector<unsigned char>(packed.bytes,0).data());
  workspace.upload(std::vector<unsigned char>(workspace.bytes,0xa5).data());panel.upload(std::vector<unsigned char>(panel.bytes,0x5a).data());
  const auto run=[&]{vt::Exl3GroupedW8A8(gpu.q,out.tensor,input.tensor,packed.tensor,suh.tensor,svh.tensor,map.tensor,workspace.tensor,panel.tensor,{4,2,"full-rows-public-atomicity",128});};
  run();const auto old_out=out.download(),old_workspace=workspace.download(),old_panel=panel.download(),old_input=input.download();
  const auto unchanged=[&]{xpu_test::SameBytes(out.download(),old_out);xpu_test::SameBytes(workspace.download(),old_workspace);xpu_test::SameBytes(panel.download(),old_panel);};
  auto bad_input=old_input;const uint16_t invalid=0x7e35;std::memcpy(bad_input.data()+bad_input.size()-2,&invalid,2);input.upload(bad_input.data());
  CHECK_THROWS_WITH_AS(run(),doctest::Contains("finite inputs"),std::runtime_error);unchanged();input.upload(old_input.data());
  std::vector<float> bad_svh(n,1.f);bad_svh.back()=std::numeric_limits<float>::infinity();svh.put(bad_svh);
  CHECK_THROWS_WITH_AS(run(),doctest::Contains("finite svh"),std::runtime_error);unchanged();svh.put(std::vector<float>(n,1.f));
  run();xpu_test::SameBytes(out.download(),old_out);
  }
}
