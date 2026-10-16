#include "xpu_test_helpers.h"
#include "vt/xpu.h"
#include "vt/fp8_kv.h"
#include <limits>
#include <numeric>

namespace {
using vt::DType;
using xpu_test::Buffer;
using xpu_test::Queue;
using xpu_test::Values;
using xpu_test::Close;
using xpu_test::SameBytes;
vt::Tensor FlatHeads(const vt::Tensor& tensor) {
  return vt::Tensor::Contiguous(tensor.data, tensor.dtype, tensor.device,
                                {tensor.shape[0] * tensor.shape[1], tensor.shape[2]});
}
// Actual cache allocation is [blocks,2,page,Hkv,D]; K/V are unbind views.
vt::Tensor CacheView(const vt::Tensor& storage, int side, int64_t blocks, int64_t page,
                      int64_t heads, int64_t dim) {
  auto view = vt::Tensor::Contiguous(static_cast<char*>(storage.data) + side * page * heads * dim * vt::SizeOf(storage.dtype),
                                    storage.dtype, storage.device, {blocks, page, heads, dim});
  view.stride[0] *= 2;
  return view;
}
}

TEST_CASE("XPU unpaged attention: CPU agreement, model dtypes, causal GQA and aliases") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (int dim : {7, 256}) for (bool causal : {false, true})
    for (auto input : {DType::kF32, DType::kBF16, DType::kF16}) {
      CAPTURE(dim);
      CAPTURE(causal);
      CAPTURE(input);
      constexpr int tokens = 5, hq = 6, hk = 2;
      std::vector<float> expected;
      for (auto* q : {&cpu.q, &gpu.q}) {
        Buffer query(*q, input, {tokens, hq, dim});
        Buffer key(*q, input, {tokens, hk, dim}), value(*q, input, {tokens, hk, dim});
        query.put(Values(tokens * hq * dim, 1, 0.1f));
        key.put(Values(tokens * hk * dim, 3, 0.1f));
        value.put(Values(tokens * hk * dim, 7, 0.1f));
        const vt::AttentionArgs args{1.0f / std::sqrt(float(dim)), causal};
        if (q == &cpu.q) {
          Buffer out(*q, DType::kF32, {tokens, hq, dim});
          vt::Attention(*q, out.tensor, query.tensor, key.tensor, value.tensor, args);
          expected = out.floats();
        } else {
          for (auto output : {DType::kF32, DType::kBF16, DType::kF16}) {
            CAPTURE(output);
            Buffer out(*q, output, {tokens, hq, dim});
            vt::Attention(*q, out.tensor, query.tensor, key.tensor, value.tensor, args);
            // The independent CPU reference accumulates in F32. Device
            // reductions need not be bit-identical, including at a half tie.
            const float tolerance = output == DType::kF32 ? 2e-5f :
                                    output == DType::kF16 ? 0.001f : 0.008f;
            Close(out.floats(), expected, tolerance, tolerance);
            const auto result = out.floats();
            CHECK(std::all_of(result.begin(), result.end(),
                              [](float v) { return std::isfinite(v); }));
          }
          // Same input/output storage must preserve the query until all rows
          // have consumed it. Exercise the existing native alias snapshot.
          vt::Attention(*q, query.tensor, query.tensor, key.tensor, value.tensor, args);
          const float tolerance = input == DType::kF32 ? 2e-5f :
                                  input == DType::kF16 ? 0.001f : 0.008f;
          Close(query.floats(), expected, tolerance, tolerance);
        }
      }
    }
  // CPU keeps its existing F32/BF16 output contract. The F16 extension is
  // scoped to XPU, independently of the shared floating input support.
  Buffer half(cpu.q, DType::kF16, {1, 1, 7});
  CHECK_THROWS_WITH_AS(vt::Attention(cpu.q, half.tensor, half.tensor,
                                    half.tensor, half.tensor, {1.0f, true}),
                       doctest::Contains("out must be f32 or bf16"),
                       std::runtime_error);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU attention preamble: Q/gate split, Q/K RMSNorm, partial RoPE at real geometry") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  constexpr int hq = 24, hk = 4, dim = 256, rot = 64;
  for (int tokens : {1, 5}) for (auto packed_type : {DType::kF32, DType::kBF16})
    for (auto dtype : {DType::kF32, DType::kBF16}) {
      std::vector<std::vector<float>> expected;
      for (auto* q : {&cpu.q, &gpu.q}) {
        Buffer packed(*q, packed_type, {tokens, hq * 2 * dim}), queries(*q, dtype, {tokens, hq, dim});
        Buffer gates(*q, DType::kF32, {tokens, hq, dim}), keys(*q, dtype, {tokens, hk, dim});
        Buffer qw(*q, DType::kF32, {dim}), kw(*q, DType::kF32, {dim}), pos(*q, DType::kI64, {tokens});
        packed.put(Values(tokens * hq * 2 * dim)); keys.put(Values(tokens * hk * dim, 5));
        qw.put(Values(dim, 3)); kw.put(Values(dim, 7));
        const int64_t positions[] = {0, 1, 137, 65535, 262143}; pos.upload(positions);
        vt::AttnGateSplit(*q, queries.tensor, gates.tensor, packed.tensor);
        auto qflat = FlatHeads(queries.tensor), kflat = FlatHeads(keys.tensor);
        vt::RmsNorm(*q, qflat, qflat, qw.tensor, {1e-6f, true});
        vt::RmsNorm(*q, kflat, kflat, kw.tensor, {1e-6f, true});
        std::vector<std::vector<float>> actual{queries.floats(), keys.floats(), gates.floats()};
        vt::RopeNeox(*q, queries.tensor, keys.tensor, pos.tensor, {10000000.0f, rot});
        actual.push_back(queries.floats()); actual.push_back(keys.floats());
        // RoPE must leave every non-rotary channel untouched.
        for (int side = 0; side < 2; ++side) {
          const int heads = side == 0 ? hq : hk;
          bool unchanged = true;
          for (int row = 0; row < tokens * heads; ++row) for (int col = rot; col < dim; ++col)
            unchanged &= actual[side][row * dim + col] == actual[side + 3][row * dim + col];
          CHECK(unchanged);
        }
        if (q == &cpu.q) expected = actual;
        else for (size_t i = 0; i < actual.size(); ++i)
          Close(actual[i], expected[i], dtype == DType::kBF16 && i != 2 ? 0.008f : 4e-6f);
      }
    }
}

TEST_CASE("XPU F16 attention preamble and output gate preserve model dtype") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  constexpr int tokens = 2, hq = 2, hk = 1, dim = 8, rot = 4;
  const auto packed_values = Values(tokens * hq * 2 * dim, 11, 0.2f);
  const auto key_values = Values(tokens * hk * dim, 12, 0.2f);
  std::vector<float> ref_q, ref_k, ref_gate;
  {
    Buffer packed(cpu.q, DType::kF32, {tokens, hq * 2 * dim});
    Buffer key(cpu.q, DType::kF16, {tokens, hk * dim});
    Buffer query(cpu.q, DType::kF32, {tokens, hq, dim});
    Buffer keys(cpu.q, DType::kF32, {tokens, hk, dim});
    Buffer gate(cpu.q, DType::kF32, {tokens, hq, dim});
    Buffer qw(cpu.q, DType::kF32, {dim}), kw(cpu.q, DType::kF32, {dim});
    Buffer pos(cpu.q, DType::kI32, {tokens});
    auto rounded = packed_values;
    for (auto& value : rounded) value = vt::F16ToF32(vt::F32ToF16(value));
    packed.put(rounded); key.put(key_values);
    qw.put(std::vector<float>(dim, 1.0f)); kw.put(std::vector<float>(dim, 1.0f));
    const int32_t positions[] = {1, 17}; pos.upload(positions);
    vt::AttnGateSplit(cpu.q, query.tensor, gate.tensor, packed.tensor);
    auto query_flat = FlatHeads(query.tensor), keys_flat = FlatHeads(keys.tensor);
    auto key_flat = vt::Tensor::Contiguous(key.tensor.data, DType::kF16,
                                           cpu.q.device, {tokens * hk, dim});
    vt::RmsNorm(cpu.q, query_flat, query_flat, qw.tensor, {1e-6f, true});
    vt::RmsNorm(cpu.q, keys_flat, key_flat, kw.tensor, {1e-6f, true});
    vt::RopeNeox(cpu.q, query.tensor, keys.tensor, pos.tensor,
                 {10000000.0f, rot});
    ref_q = query.floats(); ref_k = keys.floats(); ref_gate = gate.floats();
  }
  Buffer packed(gpu.q, DType::kF16, {tokens, hq * 2 * dim});
  Buffer key(gpu.q, DType::kF16, {tokens, hk * dim});
  Buffer query(gpu.q, DType::kF16, {tokens, hq, dim});
  Buffer keys(gpu.q, DType::kF16, {tokens, hk, dim});
  Buffer gate(gpu.q, DType::kF32, {tokens, hq, dim});
  Buffer qw(gpu.q, DType::kF32, {dim}), kw(gpu.q, DType::kF32, {dim});
  Buffer pos(gpu.q, DType::kI32, {tokens}), cos_sin(gpu.q, DType::kF32, {tokens, rot});
  packed.put(packed_values); key.put(key_values);
  qw.put(std::vector<float>(dim, 1.0f)); kw.put(std::vector<float>(dim, 1.0f));
  const int32_t positions[] = {1, 17}; pos.upload(positions);
  vt::RopeCosSinCache(gpu.q, cos_sin.tensor, pos.tensor, {10000000.0f, rot});
  vt::AttnQkNormRopeGate(gpu.q, query.tensor, keys.tensor, gate.tensor,
                         packed.tensor, key.tensor, qw.tensor, kw.tensor,
                         cos_sin.tensor, {1e-6f, true}, {10000000.0f, rot});
  Close(query.floats(), ref_q, 0.003f, 1e-4f);
  Close(keys.floats(), ref_k, 0.003f, 1e-4f);
  Close(gate.floats(), ref_gate, 2e-6f);
  Buffer widened(gpu.q, DType::kF32, {tokens, hq, dim});
  vt::CastF32(gpu.q, widened.tensor, query.tensor);
  Close(widened.floats(), query.floats(), 0.0f);
  Buffer attn(gpu.q, DType::kF16, {tokens, hq, dim});
  Buffer gated(gpu.q, DType::kF16, {tokens, hq, dim});
  attn.put(std::vector<float>(tokens * hq * dim, 0.25f));
  vt::SigmoidGateBf16(gpu.q, gated.tensor, attn.tensor, gate.tensor);
  const auto actual = gated.floats();
  for (size_t i = 0; i < actual.size(); ++i)
    CHECK(std::abs(actual[i] - 0.25f / (1.0f + std::exp(-ref_gate[i]))) < 0.0003f);
}

TEST_CASE("XPU SG16 attention preamble: F16 real geometry, strided inputs and partial RoPE") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  constexpr int tokens = 3, hq = 24, hk = 4, dim = 256, rot = 64;
  Buffer pos(gpu.q, DType::kI32, {tokens}), cos_gpu(gpu.q, DType::kF32, {tokens, rot});
  const int32_t positions[] = {1, 137, 4097};
  pos.upload(positions);
  vt::RopeCosSinCache(gpu.q, cos_gpu.tensor, pos.tensor, {10000000.0f, rot});
  const auto cos_values = cos_gpu.floats();
  std::vector<float> expected_q, expected_k, expected_gate;
  for (auto* q : {&cpu.q, &gpu.q}) {
    const auto type = q == &cpu.q ? DType::kF32 : DType::kF16;
    Buffer packed(*q, type, {tokens, 2 * hq * dim + 7});
    Buffer key(*q, type, {tokens, hk * dim + 3});
    Buffer query(*q, type, {tokens, hq, dim}), keys(*q, type, {tokens, hk, dim});
    Buffer gate(*q, DType::kF32, {tokens, hq, dim});
    Buffer qw(*q, DType::kF32, {dim}), kw(*q, DType::kF32, {dim});
    Buffer cos(*q, DType::kF32, {tokens, rot});
    auto put_rounded = [&](Buffer& buffer, std::vector<float> values) {
      if (q == &cpu.q) for (auto& value : values) value = vt::F16ToF32(vt::F32ToF16(value));
      buffer.put(values);
    };
    put_rounded(packed, Values(tokens * (2 * hq * dim + 7), 4));
    put_rounded(key, Values(tokens * (hk * dim + 3), 9));
    packed.tensor.shape[1] = 2 * hq * dim;
    key.tensor.shape[1] = hk * dim;
    qw.put(Values(dim, 2, 0.02f));
    kw.put(Values(dim, 3, 0.02f));
    cos.put(cos_values);
    vt::AttnQkNormRopeGate(*q, query.tensor, keys.tensor, gate.tensor,
                           packed.tensor, key.tensor, qw.tensor, kw.tensor,
                           cos.tensor, {1e-6f, true}, {10000000.0f, rot});
    if (q == &cpu.q) {
      expected_q = query.floats(); expected_k = keys.floats(); expected_gate = gate.floats();
    } else {
      Close(query.floats(), expected_q, 0.003f, 0.0005f);
      Close(keys.floats(), expected_k, 0.003f, 0.0005f);
      Close(gate.floats(), expected_gate, 0.0f);
    }
  }
}

TEST_CASE("XPU cached RoPE: supplied positions, strided heads, optional K and both pair styles") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  constexpr int tokens = 4, hq = 24, hk = 4, dim = 256, rot = 64, count = 130;
  for (int scaling : {0, 1, 2}) for (bool neox : {false, true}) for (bool have_keys : {false, true}) {
    CAPTURE(scaling);
    CAPTURE(neox);
    CAPTURE(have_keys);
    std::vector<float> expected_cache, expected_q, expected_k;
    for (auto* q : {&cpu.q, &gpu.q}) {
      Buffer queries(*q, DType::kF32, {tokens, hq + 1, dim + 3});
      Buffer keys(*q, DType::kF32, {tokens, hk + 2, dim + 7});
      Buffer cache(*q, DType::kF32, {count, rot}), allpos(*q, DType::kI32, {count}), pos(*q, DType::kI32, {tokens});
      queries.put(Values(tokens * (hq + 1) * (dim + 3))); keys.put(Values(tokens * (hk + 2) * (dim + 7), 4));
      queries.tensor.shape[1] = hq; queries.tensor.shape[2] = dim;
      keys.tensor.shape[1] = hk; keys.tensor.shape[2] = dim;
      std::vector<int32_t> all_positions(count); std::iota(all_positions.begin(), all_positions.end(), 0);
      allpos.upload(all_positions.data()); const int32_t positions[] = {129, 3, 64, 0}; pos.upload(positions);
      vt::RopeArgs args{10000000.0f, rot}; args.is_neox_style = neox;
      if (scaling == 1) args.linear_scaling_factor = 2.0f;
      if (scaling == 2) {
        args.llama3_scaling_factor = 8; args.llama3_low_freq_factor = 1;
        args.llama3_high_freq_factor = 4; args.llama3_orig_max_position = 8192;
      }
      vt::RopeCosSinCache(*q, cache.tensor, allpos.tensor, args);
      vt::RopeFromCache(*q, queries.tensor, have_keys ? &keys.tensor : nullptr, pos.tensor, cache.tensor, args);
      if (q == &cpu.q) { expected_cache = cache.floats(); expected_q = queries.floats(); expected_k = keys.floats(); }
      else {
        // CPU vector powf/sincos and device math need not round identically.
        // F32 inverse-frequency rounding becomes an angle error with position.
        const float absolute = scaling == 1 ? 1e-5f : 1e-6f;
        Close(cache.floats(), expected_cache, 4e-6f, absolute);
        Close(queries.floats(), expected_q, 4e-6f, absolute);
        Close(keys.floats(), expected_k, 4e-6f, absolute);
        if (scaling == 1) {
          std::vector<float> scalar(count * rot);
          for (int p = 0; p < count; ++p) for (int i = 0; i < rot / 2; ++i) {
            const float exponent = float(2 * i) / float(rot);
            const float power = float(std::pow(double(args.base), double(exponent)));
            const float inv = float(1.0 / double(power));
            const float angle = (float(p) / args.linear_scaling_factor) * inv;
            scalar[p * rot + i] = float(std::cos(double(angle)));
            scalar[p * rot + rot / 2 + i] = float(std::sin(double(angle)));
          }
          Close(cache.floats(), scalar, 1e-6f);
        }
        const int32_t invalid[] = {130, 3, 64, 0}; pos.upload(invalid);
        CHECK_THROWS(vt::RopeFromCache(*q, queries.tensor, nullptr, pos.tensor, cache.tensor, args));
      }
    }
  }
}

TEST_CASE("XPU KV write: unbind strides, padded inputs, null/repeated slots and raw bit preservation") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  constexpr int tokens = 7, blocks = 4, heads = 4, dim = 256;
  for (int page : {3, 16}) for (auto dtype : {DType::kF32, DType::kBF16, DType::kF16}) {
    std::vector<unsigned char> expected;
    for (auto* q : {&cpu.q, &gpu.q}) {
      Buffer keys(*q, dtype, {tokens, heads + 1, dim}), values(*q, dtype, {tokens, heads + 2, dim});
      Buffer storage(*q, dtype, {blocks, 2 * page, heads, dim}), slots(*q, DType::kI64, {tokens});
      // Includes NaNs and signed zero; a KV store is a bit copy, not a cast.
      std::vector<uint32_t> raw_keys((keys.bytes + 3) / 4), raw_values((values.bytes + 3) / 4);
      for (size_t i = 0; i < raw_keys.size(); ++i) raw_keys[i] = uint32_t(i * 2654435761u);
      for (size_t i = 0; i < raw_values.size(); ++i) raw_values[i] = uint32_t(i * 2246822519u);
      keys.upload(raw_keys.data()); values.upload(raw_values.data()); storage.put(Values(storage.tensor.Numel(), 7));
      keys.tensor.shape[1] = values.tensor.shape[1] = heads;
      auto kc = CacheView(storage.tensor, 0, blocks, page, heads, dim);
      auto vc = CacheView(storage.tensor, 1, blocks, page, heads, dim);
      const int64_t mapping[] = {2 * page, -1, page - 1, page, 0, blocks * page - 1, 0}; slots.upload(mapping);
      vt::ReshapeAndCache(*q, keys.tensor, values.tensor, kc, vc, slots.tensor);
      if (q == &cpu.q) expected = storage.download();
      else {
        SameBytes(storage.download(), expected);
        const int64_t invalid[] = {blocks * page, -1, 0, 0, 0, 0, 0}; slots.upload(invalid);
        CHECK_THROWS(vt::ReshapeAndCache(*q, keys.tensor, values.tensor, kc, vc, slots.tensor));
        SameBytes(storage.download(), expected);
      }
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == vt::xpu::GetMemoryInfo().attention_workspace_bytes);
}

TEST_CASE("XPU paged attention: appended queries, GQA, causal alignment, strides and BF16 cache") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (int requests : {1, 4}) for (int chunk : {1, 5}) for (int page : {3, 16})
    for (auto dtype : {DType::kF32, DType::kBF16}) for (int mode : {0, 1, 2})
    for (bool fp8 : {false, true}) {
      CAPTURE(requests);
      CAPTURE(chunk);
      CAPTURE(page);
      CAPTURE(dtype);
      CAPTURE(mode);
      constexpr int heads = 24, kvheads = 4, dim = 256, columns = 6;
      const int blocks = requests * columns;
      std::vector<int32_t> offsets{0}, lengths(requests), table(requests * (2 * columns + 1), -1);
      std::vector<int64_t> slots;
      for (int r = 0; r < requests; ++r) {
        const int count = chunk == 1 ? 1 : chunk - r;
        const int context = page + r + 1;
        offsets.push_back(offsets.back() + count); lengths[r] = context + count;
        for (int b = 0; b < columns; ++b) table[r * (2 * columns + 1) + 2 * b] = blocks - 1 - (r * columns + b);
        for (int t = 0; t < count; ++t) {
          const auto pos = context + t;
          slots.push_back(int64_t(table[r * (2 * columns + 1) + 2 * (pos / page)]) * page + pos % page);
        }
      }
      const int tokens = offsets.back();
      std::vector<float> expected;
      for (auto* q : {&cpu.q, &gpu.q}) {
        Buffer query(*q, dtype, {tokens, heads, dim}), output(*q, dtype, {tokens, heads, dim});
        Buffer storage(*q, fp8 ? DType::kI8 : DType::kBF16, {blocks, 2 * page, kvheads, dim});
        Buffer keys(*q, DType::kBF16, {tokens, kvheads, dim}), values(*q, DType::kBF16, {tokens, kvheads, dim});
        Buffer bt(*q, DType::kI32, {requests, 2 * columns + 1}), lens(*q, DType::kI32, {requests});
        Buffer qsl(*q, DType::kI32, {requests + 1}), ids(*q, DType::kI64, {tokens});
        query.put(Values(query.tensor.Numel(), 7, 0.2f));
        auto initial = Values(storage.tensor.Numel(), 3, 0.08f);
        if (fp8) {
          std::vector<uint8_t> bytes(initial.size());
          for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = vt::F32ToF8E4M3(initial[i]);
          storage.upload(bytes.data());
        } else storage.put(initial);
        keys.put(Values(keys.tensor.Numel(), 13, 0.2f)); values.put(Values(values.tensor.Numel(), 11));
        bt.upload(table.data()); lens.upload(lengths.data()); qsl.upload(offsets.data()); ids.upload(slots.data());
        bt.tensor.shape[1] = columns; bt.tensor.stride[1] = 2;
        auto kc = CacheView(storage.tensor, 0, blocks, page, kvheads, dim);
        auto vc = CacheView(storage.tensor, 1, blocks, page, kvheads, dim);
        vt::PagedAttentionArgs args;
        if (fp8) {
          args.kv_cache_dtype = vt::Fp8KVCacheDataType::kFp8E4M3;
          args.k_scale = 0.3f; args.v_scale = 1.7f;
          vt::ReshapeAndCacheFp8(*q, keys.tensor, values.tensor, kc, vc, ids.tensor,
                                 args.kv_cache_dtype, args.k_scale, args.v_scale);
        } else vt::ReshapeAndCache(*q, keys.tensor, values.tensor, kc, vc, ids.tensor);
        args.scale = 1.0f / 16.0f; args.causal = mode != 1;
        if (mode == 2) { args.window_size = vt::AttentionWindow{3, 2}; args.logits_soft_cap = 0.7f; }
        // Also exercise alias snapshot on the one-token decode path.
        auto& target = q == &gpu.q && chunk == 1 ? query.tensor : output.tensor;
        vt::PagedAttention(*q, target, query.tensor, kc, vc, bt.tensor, lens.tensor, qsl.tensor, args);
        if (q == &cpu.q) expected = output.floats();
        else Close(chunk == 1 ? query.floats() : output.floats(), expected,
                     dtype == DType::kBF16 ? 0.008f : 1e-4f, dtype == DType::kBF16 ? 2e-5f : 5e-6f);
      }
    }
  CHECK(vt::GetReferenceTierHits() == 0);
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == vt::xpu::GetMemoryInfo().attention_workspace_bytes);
}

TEST_CASE("XPU paged attention: head tails, empty requests, F16 queries and invalid device metadata") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  constexpr int tokens = 3, heads = 6, kvheads = 2, dim = 13, page = 3, blocks = 4;
  std::vector<float> expected;
  for (auto* q : {&cpu.q, &gpu.q}) {
    Buffer query(*q, DType::kF16, {tokens, heads, dim}), output(*q, DType::kF32, {tokens, heads, dim});
    Buffer storage(*q, DType::kBF16, {blocks, 2 * page, kvheads, dim});
    Buffer table(*q, DType::kI32, {4, 2}), lens(*q, DType::kI32, {4}), qsl(*q, DType::kI32, {5});
    query.put(Values(query.tensor.Numel(), 4)); storage.put(Values(storage.tensor.Numel(), 9));
    const int32_t offsets[] = {0, 0, 2, 2, 3}, lengths[] = {-1, 5, -1, 4}, ids[] = {-1, -1, 2, 0, -1, -1, 3, 1};
    qsl.upload(offsets); lens.upload(lengths); table.upload(ids);
    auto kc = CacheView(storage.tensor, 0, blocks, page, kvheads, dim);
    auto vc = CacheView(storage.tensor, 1, blocks, page, kvheads, dim);
    vt::PagedAttentionArgs args; args.scale = 1.0f / std::sqrt(float(dim));
    auto run = [&] { vt::PagedAttention(*q, output.tensor, query.tensor, kc, vc, table.tensor, lens.tensor, qsl.tensor, args); };
    run();
    if (q == &cpu.q) expected = output.floats();
    else {
      Close(output.floats(), expected, 2e-5f);
      const int32_t bad_offsets[] = {0, 2, 1, 2, 3}; qsl.upload(bad_offsets); CHECK_THROWS(run()); qsl.upload(offsets);
      for (const auto& bad : {std::vector<int32_t>{-1, 1, -1, 4}, std::vector<int32_t>{-1, 7, -1, 4}}) {
        lens.upload(bad.data()); CHECK_THROWS(run());
      }
      lens.upload(lengths);
      for (int invalid : {-1, blocks}) {
        auto bad = std::vector<int32_t>(ids, ids + 8); bad[2] = invalid;
        table.upload(bad.data()); CHECK_THROWS(run());
      }
      Close(output.floats(), expected, 2e-5f);
    }
  }
}

TEST_CASE("XPU E4M3 KV codec: all bytes, ties, adjacent floats, saturation and independent scales") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  std::vector<float> input;
  for (int code = 0; code < 126; ++code) {
    const float a = vt::F8E4M3ToF32(code), b = vt::F8E4M3ToF32(code + 1);
    const float mid = (a + b) * 0.5f;
    for (float v : {a, std::nextafter(mid, a), mid, std::nextafter(mid, b)}) {
      input.push_back(v); input.push_back(-v);
    }
  }
  for (float v : {0.0f, -0.0f, 448.0f, -448.0f, 1000.0f, -1000.0f,
                 std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                 std::numeric_limits<float>::quiet_NaN()}) input.push_back(v);
  const int64_t dim = input.size();
  for (auto dtype : {DType::kF32, DType::kBF16, DType::kF16}) for (float scale : {1.0f, 0.125f, 2.7f}) {
    CAPTURE(dtype);
    CAPTURE(scale);
    std::vector<unsigned char> expected;
    for (auto* q : {&cpu.q, &gpu.q}) {
      Buffer k(*q, dtype, {3, 2, dim}), v(*q, dtype, {3, 2, dim});
      Buffer cache(*q, DType::kI8, {2, 4, 1, dim}), slots(*q, DType::kI64, {3});
      std::vector<float> source(6 * dim);
      for (int i = 0; i < 6; ++i) std::copy(input.begin(), input.end(), source.begin() + i * dim);
      for (int64_t i = 0; i < 4 * dim; ++i) source[i] = -source[i]; // earlier duplicate differs
      k.put(source); v.put(source);
      k.tensor.shape[1] = v.tensor.shape[1] = 1; // padded token strides
      const int64_t mapping[] = {3, -1, 3}; slots.upload(mapping);
      std::vector<uint8_t> initial(cache.bytes, 0x23); cache.upload(initial.data());
      auto kc = CacheView(cache.tensor, 0, 2, 2, 1, dim);
      auto vc = CacheView(cache.tensor, 1, 2, 2, 1, dim);
      vt::ReshapeAndCacheFp8(*q, k.tensor, v.tensor, kc, vc, slots.tensor,
                              vt::Fp8KVCacheDataType::kFp8E4M3, scale, 3.0f * scale);
      if (q == &cpu.q) expected = cache.download();
      else {
        SameBytes(cache.download(), expected);
        const int64_t invalid[] = {3, -1, 4}; slots.upload(invalid);
        CHECK_THROWS(vt::ReshapeAndCacheFp8(*q, k.tensor, v.tensor, kc, vc, slots.tensor,
                                           vt::Fp8KVCacheDataType::kFp8E4M3, scale, scale));
        SameBytes(cache.download(), expected);
      }
    }
  }
  // A single visible key makes attention return V unchanged, independently
  // checking every decoder byte (including zero and the two NaNs).
  for (float scale : {1.0f, 0.125f, 2.7f}) {
    Buffer query(gpu.q, DType::kF32, {1, 1, 256}), out(gpu.q, DType::kF32, {1, 1, 256});
    Buffer keys(gpu.q, DType::kI8, {1, 1, 1, 256}), values(gpu.q, DType::kI8, {1, 1, 1, 256});
    Buffer table(gpu.q, DType::kI32, {1, 1}), lengths(gpu.q, DType::kI32, {1}), offsets(gpu.q, DType::kI32, {2});
    query.put(std::vector<float>(256, 0)); std::vector<uint8_t> bytes(256, 0); keys.upload(bytes.data());
    std::iota(bytes.begin(), bytes.end(), 0); values.upload(bytes.data());
    const int32_t zero = 0, one = 1, qsl[] = {0, 1};
    table.upload(&zero); lengths.upload(&one); offsets.upload(qsl);
    vt::PagedAttentionArgs args; args.scale = 1; args.v_scale = scale;
    args.kv_cache_dtype = vt::Fp8KVCacheDataType::kFp8E4M3;
    vt::PagedAttention(gpu.q, out.tensor, query.tensor, keys.tensor, values.tensor,
                       table.tensor, lengths.tensor, offsets.tensor, args);
    const auto actual = out.floats();
    for (int i = 0; i < 256; ++i) {
      CAPTURE(i);
      CAPTURE(scale);
      if ((i & 127) == 127) CHECK(std::isnan(actual[i]));
      else CHECK(actual[i] == vt::F8E4M3ToF32(i) * scale);
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}
