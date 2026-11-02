// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Ported scalar contracts from vLLM a7c23ac96d7806e7c7e7d862eadbce5a33529b94.
// tests/kernels/core/test_layernorm.py:21-132, test_activation.py:137-221,
// test_rotary_embedding.py:29-93. See the wave evidence for harness adaptations.
#include <doctest/doctest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/op_provider.h"
#include "vt/recipes.h"
#include "vt/quant.h"
void CudaFp16ScalarReferenceSilu(vt::Queue&, vt::Tensor&, const vt::Tensor&);

namespace {
using namespace vt;
float Round(float x, DType dt) {
  if (dt == DType::kF16) return F16ToF32(F32ToF16(x));
  if (dt == DType::kBF16) return BF16ToF32(F32ToBF16(x));
  return x;
}
std::vector<uint8_t> Pack(const std::vector<float>& v, DType dt) {
  std::vector<uint8_t> b(v.size()*SizeOf(dt));
  for (size_t i = 0; i<v.size(); ++i) {
    if (dt == DType::kF32) std::memcpy(b.data()+i*4, &v[i], 4);
    else {
      const uint16_t h = dt == DType::kF16?F32ToF16(v[i]):F32ToBF16(v[i]);
      std::memcpy(b.data()+i*2, &h, 2);
    }
  }
  return b;
}
struct Session {
  Backend& b = GetBackend(DeviceType::kCUDA);
  Queue q = b.CreateQueue();
  ~Session() {
    b.DestroyQueue(q);
  }
};
struct Buffer {
  Session& s;
  Tensor t;
  void* p;
  Buffer(Session& ss, DType dt, std::initializer_list<int64_t> shape, const void* host = nullptr):s(ss) {
    t = Tensor::Contiguous(nullptr, dt, s.q.device, shape);
    const size_t bytes = RowSizeBytes(dt, t.shape[t.rank - 1]) *
                         (t.Numel() / t.shape[t.rank - 1]);
    p = s.b.Alloc(std::max<size_t>(1, bytes));
    t.data = p;
    if (host) s.b.Copy(s.q, p, host, bytes);
  }
  ~Buffer() {
    s.b.Free(p);
  }
  std::vector<float> read() {
    std::vector<uint8_t> b(t.Bytes());
    s.b.Copy(s.q, b.data(), p, b.size());
    s.b.Synchronize(s.q);
    std::vector<float> v(t.Numel());
    for (size_t i = 0; i<v.size(); ++i) {
      if (t.dtype == DType::kF32)std::memcpy(&v[i], b.data()+i*4, 4);
      else {
        uint16_t h;
        std::memcpy(&h, b.data()+i*2, 2);
        v[i] = t.dtype == DType::kF16?F16ToF32(h):BF16ToF32(h);
      }
    }
    return v;
  }
};
std::vector<float> Random(size_t n, float scale = 1, float mean = 0) {
  std::mt19937 rng(0);
  std::normal_distribution<float> dist(mean, scale);
  std::vector<float> v(n);
  for (auto& x:v)x = dist(rng);
  return v;
}
void Close(const std::vector<float>& got, const std::vector<float>& want, float atol, float rtol) {
  REQUIRE(got.size() == want.size());
  size_t bad = 0, first = 0;
  for (size_t i = 0; i<got.size(); ++i)if (!(std::fabs(got[i]-want[i]) <= atol+rtol*std::fabs(want[i]))) {
    if (!bad)first = i;
    ++bad;
  }
  CAPTURE(first);
  CAPTURE(got[first]);
  CAPTURE(want[first]);
  CAPTURE(bad);
  CHECK(bad == 0);
}
}

TEST_CASE("CUDA FP16 scalar reverse cast preserves half format and row strides") {
  Session s;
  std::vector<uint16_t> raw = {
    F32ToF16(1.0009765625f), 1, 0x7bff, 0x8000, 0, 0, F32ToF16(-2.00390625f), 0x0400, 0x3555, 0x3c01, 0, 0
  };
  Buffer in(s, DType::kF16, {2, 6}, raw.data()), out(s, DType::kF32, {2, 4});
  in.t.shape[1] = 4;
  CastF32(s.q, out.t, in.t);
  std::vector<float> want;
  for (int r = 0; r<2; ++r)
  for (int j = 0; j<4; ++j)want.push_back(F16ToF32(raw[r*6+j]));
  Close(out.read(), want, 0, 0);
}

TEST_CASE("CUDA FP16 scalar add and broadcast use half stores") {
  Session s;
  for (bool broadcast:{false, true}) {
    std::vector<float> a = {
      1.0009765625f, 65504, -65504, 0.00006103515625f, -2.00390625f, 0.333251953125f
    };
    std::vector<float> b = broadcast?std::vector<float> {
      0.00048828125f, 32, 16
    }
    :std::vector<float> {
      0.00048828125f, 32, -32, -0.000061f, 2, 0.0002f
    };
    auto ap = Pack(a, DType::kF16), bp = Pack(b, DType::kF16);
    Buffer x(s, DType::kF16, {2, 3}, ap.data()), y(s, DType::kF16, {broadcast?1:2, 3}, bp.data()), out(s, DType::kF16, {2, 3});
    if (broadcast)y.t = Tensor::Contiguous(y.p, DType::kF16, s.q.device, {3});
    Add(s.q, out.t, x.t, y.t);
    std::vector<float> want(6);
    for (int i = 0; i<6; ++i)want[i] = Round(a[i]+Round(b[broadcast?i%3:i], DType::kF16), DType::kF16);
    auto got = out.read();
    for (size_t i = 0; i<got.size(); ++i) {
      if (std::isinf(want[i]))CHECK(got[i] == want[i]);
      else CHECK(got[i] == want[i]);
    }
  }
}

TEST_CASE("CUDA FP16 scalar RMSNorm ports upstream shapes residual strides and weightless") {
  Session s;
  for (auto dt:{DType::kF16, DType::kBF16, DType::kF32})
  for (int tokens:{7, 83, 4096})
  for (int hidden:{8, 768, 769, 5120, 5125, 8192})
  for (bool residual:{false, true})
  for (bool strided:{false, true})
  for (bool weightless:{false, true}) {
    CAPTURE(std::string(Name(dt)));
    CAPTURE(tokens);
    CAPTURE(hidden);
    CAPTURE(residual);
    CAPTURE(strided);
    CAPTURE(weightless);
    const int stride = strided?2*hidden:hidden;
    const float scale = weightless ? 1.0f : 1.0f / (2 * hidden);
    auto x = Random(size_t(tokens)*stride, scale), r = Random(size_t(tokens)*stride, scale);
    auto w = weightless?std::vector<float>(hidden, 1):Random(hidden, 0.1f, 1);
    for (auto& z:x)z = Round(z, dt);
    for (auto& z:r)z = Round(z, dt);
    for (auto& z:w)z = Round(z, dt);
    auto xp = Pack(x, dt), rp = Pack(r, dt), wp = Pack(w, dt);
    Buffer dx(s, dt, {tokens, stride}, xp.data()), dr(s, dt, {tokens, stride}, rp.data()), dw(s, dt, {hidden}, wp.data()), out(s, dt, {tokens, hidden});
    dx.t.shape[1] = hidden;
    dr.t.shape[1] = hidden;
    RmsNorm(s.q, out.t, dx.t, dw.t, RmsNormArgs{1e-6f, false}, residual?&dr.t:nullptr);
    std::vector<float> want(size_t(tokens)*hidden);
    // Native primary IR computes variance over the F32 residual sum. Its CUDA
    // fused kernel instead squares the rounded residual. Preserve the native
    // comparison and its upstream tolerance, then check stored residuals exactly.
    for (int t = 0; t < tokens; ++t) {
      double sum = 0;
      for (int j = 0; j < hidden; ++j) {
        const size_t i = size_t(t)*stride+j;
        const float value = residual ? x[i] + r[i] : x[i];
        sum += double(value)*value;
      }
      const float inv = 1 / std::sqrt(float(sum / hidden) + 1e-6f);
      for (int j = 0; j < hidden; ++j) {
        const size_t i = size_t(t)*stride+j;
        const float value = residual ? x[i] + r[i] : x[i];
        const float normalized = value * inv;
        want[size_t(t)*hidden+j] = weightless ? Round(normalized, dt) :
            Round(Round(normalized, dt) * w[j], dt);
        if (residual) r[i] = Round(value, dt);
      }
    }
    const float atol = weightless && dt == DType::kF32 ? 1e-5f :
                       weightless && dt == DType::kBF16 ? 1e-3f : 1e-2f;
    const float rtol = !weightless ? 1e-2f : dt == DType::kF16 ? 2e-3f :
                       dt == DType::kBF16 ? 1.6e-2f : 1.3e-6f;
    Close(out.read(), want, atol, rtol);
    if (residual) {
      dr.t.shape[1] = stride;
      Close(dr.read(), r, 0, 0);
    }
  }
}

TEST_CASE("CUDA FP16 scalar FusedChain preserves residual rounding") {
  Session s;
  auto x = Pack({1.0009765625f, 0.00048828125f, -1.0009765625f, 0.333251953125f}, DType::kF16), r = Pack({0.00048828125f, 0.000000059604644775390625f, 1, 0.0001220703125f}, DType::kF16), w = Pack({1, 1, 1, 1}, DType::kF16);
  Buffer dx(s, DType::kF16, {1, 4}, x.data()), r1(s, DType::kF16, {1, 4}, r.data()), r2(s, DType::kF16, {1, 4}, r.data()), dw(s, DType::kF16, {4}, w.data()), a(s, DType::kF16, {1, 4}), b(s, DType::kF16, {1, 4});
  RmsNorm(s.q, a.t, dx.t, dw.t, RmsNormArgs{}, &r1.t);
  FusedChain(s.q, b.t, dx.t, dw.t, &r2.t, kFusedAddRmsNormStd, 1e-6f);
  Close(b.read(), a.read(), 0, 0);
  Close(r2.read(), r1.read(), 0, 0);
}

TEST_CASE("CUDA FP16 scalar split and merged SiLU port the exact primary matrix") {
  Session s;
  for (auto dt : {DType::kF16, DType::kBF16, DType::kF32}) {
    for (int tokens : {7, 83, 2048}) {
      for (int d : {512, 13824}) {
        CAPTURE(std::string(Name(dt)));
        CAPTURE(tokens);
        CAPTURE(d);
        auto x = Random(size_t(tokens) * 2 * d);
        for (auto& z : x) z = Round(z, dt);
        auto xp = Pack(x, dt);
        Buffer dx(s, dt, {tokens, 2 * d}, xp.data());
        Buffer out(s, dt, {tokens, d});
        Buffer reference(s, dt, {tokens, d});
        CudaFp16ScalarReferenceSilu(s.q, reference.t, dx.t);
        SiluAndMul(s.q, out.t, dx.t);
        const auto expected = reference.read();
        Close(out.read(), expected, 0, 0);
        std::vector<float> gate(size_t(tokens) * d), up(gate.size());
        for (int t = 0; t < tokens; ++t) {
          for (int j = 0; j < d; ++j) {
            const size_t i = size_t(t) * d + j;
            gate[i] = x[size_t(t) * 2 * d + j];
            up[i] = x[size_t(t) * 2 * d + d + j];
          }
        }
        auto gp = Pack(gate, dt), up_pack = Pack(up, dt);
        Buffer dg(s, dt, {tokens, d}, gp.data());
        Buffer du(s, dt, {tokens, d}, up_pack.data());
        MoeSiluMul(s.q, out.t, dg.t, du.t);
        Close(out.read(), expected, 0, 0);
      }
    }
  }
}

TEST_CASE("CUDA FP16 scalar ordinary and quantized embedding gather typed outputs") {
  Session s;
  std::vector<float> table(5*32);
  for (size_t i = 0; i<table.size(); ++i)table[i] = float(int(i)%31-15)*0.125f;
  const int32_t ids[] = {
    4, 0, 3, 4
  };
  Buffer di(s, DType::kI32, {4}, ids), out(s, DType::kF16, {4, 32});
  for (auto dt:{DType::kF32, DType::kBF16, DType::kF16}) {
    auto p = Pack(table, dt);
    Buffer tab(s, dt, {5, 32}, p.data());
    Embedding(s.q, out.t, tab.t, di.t);
    std::vector<float> want(128);
    for (int t = 0; t<4; ++t)
  for (int j = 0; j<32; ++j)want[t*32+j] = Round(Round(table[ids[t]*32+j], dt), DType::kF16);
    Close(out.read(), want, 0, 0);
  }
  // Q8_0: original packed blocks, half scale followed by 32 signed codes.
  std::vector<uint8_t> raw(5*34);
  std::vector<float> decoded(160);
  for (int v = 0; v<5; ++v) {
    uint16_t scale = F32ToF16(0.031280517578125f);
    std::memcpy(raw.data()+v*34, &scale, 2);
    for (int j = 0; j<32; ++j) {
      int8_t code = int8_t(j-16+v);
      raw[v*34+2+j] = uint8_t(code);
      decoded[v*32+j] = F16ToF32(scale)*code;
    }
  }
  Buffer tab(s, DType::kQ8_0, {5, 32}, raw.data());
  Embedding(s.q, out.t, tab.t, di.t);
  std::vector<float> want(128);
  for (int t = 0; t<4; ++t)
  for (int j = 0; j<32; ++j)want[t*32+j] = Round(decoded[ids[t]*32+j], DType::kF16);
  Close(out.read(), want, 0, 0);
}

TEST_CASE("CUDA FP16 scalar dense Matmul and MatmulBT preserve orientation strides and output") {
  Session s;
  for (bool bt:{false, true})
  for (auto outdt:{DType::kF16, DType::kF32})
  for (int m:{1, 7}) {
    const int k = 17, n = 11, stride = bt?k+3:k;
    auto a = Random(m*stride), b = Random(k*n);
    for (auto& z:a)z = Round(z, DType::kF16);
    for (auto& z:b)z = Round(z, DType::kF16);
    auto ap = Pack(a, DType::kF16), bp = Pack(b, DType::kF16);
    Buffer da(s, DType::kF16, {m, stride}, ap.data()), db(s, DType::kF16, {bt?n:k, bt?k:n}, bp.data()), out(s, outdt, {m, n});
    da.t.shape[1] = k;
    if (bt)MatmulBT(s.q, out.t, da.t, db.t);
    else Matmul(s.q, out.t, da.t, db.t);
    std::vector<float> want(m*n);
    for (int i = 0; i<m; ++i)
  for (int j = 0; j<n; ++j) {
      float sum = 0;
      for (int h = 0; h<k; ++h)sum+=a[i*stride+h]*b[bt?j*k+h:h*n+j];
      want[i*n+j] = Round(sum, outdt);
    }
    Close(out.read(), want, 1e-3f, 1e-3f);
  }
}

TEST_CASE("CUDA FP16 scalar cached RoPE ports upstream conventions optional keys and head strides") {
  Session s;
  for (auto dtype:{DType::kF32, DType::kBF16, DType::kF16})
  for (int maxpos:{11, 4096, 32768})
  for (bool neox:{false, true})
  for (int head:{32, 108})
  for (int tokens:{11, 1024})
  for (bool key:{false, true})
  for (bool padded:{false, true}) {
    const int stride = head+(padded?64:0), heads = 7;
    auto x = Random(size_t(tokens)*heads*stride);
    for (auto& z:x)z = Round(z, dtype);
    std::vector<float> cache(size_t(maxpos)*32);
    for (int p = 0; p<maxpos; ++p)
  for (int j = 0; j<16; ++j) {
      double angle = p*std::pow(10000., -2.*j/32);
      cache[p*32+j] = Round(float(std::cos(angle)), dtype);
      cache[p*32+16+j] = Round(float(std::sin(angle)), dtype);
    }
    std::vector<int64_t> pos(tokens);
    std::mt19937 rng(0);
    for (auto& p:pos)p = rng()%maxpos;
    auto xp = Pack(x, dtype), cp = Pack(cache, dtype);
    Buffer q(s, dtype, {tokens, heads, stride}, xp.data()), k(s, dtype, {tokens, heads, stride}, xp.data()), c(s, dtype, {maxpos, 32}, cp.data()), p(s, DType::kI64, {tokens}, pos.data());
    q.t.shape[2] = head;
    k.t.shape[2] = head;
    RopeArgs args;
    args.rotary_dim = 32;
    args.is_neox_style = neox;
    RopeFromCache(s.q, q.t, key?&k.t:nullptr, p.t, c.t, args);
    auto want = x;
    for (int t = 0; t<tokens; ++t)
  for (int h = 0; h<heads; ++h)
  for (int j = 0; j<16; ++j) {
      size_t off = (size_t(t)*heads+h)*stride;
      int a = neox?j:2*j, b = neox?j+16:2*j+1;
      float co = cache[pos[t]*32+j], si = cache[pos[t]*32+16+j];
      want[off+a] = Round(std::fma(-x[off+b], si, x[off+a]*co), dtype);
      want[off+b] = Round(std::fma(x[off+a], si, x[off+b]*co), dtype);
    }
    q.t.shape[2] = stride;
    k.t.shape[2] = stride;
    Close(q.read(), want, 1e-3f, 1e-3f);
    Close(k.read(), key?want:x, 1e-3f, 1e-3f);
    if (padded) {
      // Upstream also flattens the sliced head axes in this stride variant.
      // Materialize its logical values, then reshape to the rank-3 VT ABI.
      std::vector<float> flat_x(size_t(tokens)*heads*head), flat_want(flat_x.size());
      for (int t = 0; t<tokens; ++t)
      for (int h = 0; h<heads; ++h)
      for (int j = 0; j<head; ++j) {
        const size_t dst = (size_t(t)*heads+h)*head+j;
        const size_t src = (size_t(t)*heads+h)*stride+j;
        flat_x[dst] = x[src];
        flat_want[dst] = want[src];
      }
      auto flat_packed = Pack(flat_x, dtype);
      Buffer flat_q(s, dtype, {tokens, heads*head}, flat_packed.data()),
             flat_k(s, dtype, {tokens, heads*head}, flat_packed.data());
      auto q_view = flat_q.t.View({tokens, heads, head});
      auto k_view = flat_k.t.View({tokens, heads, head});
      RopeFromCache(s.q, q_view, key?&k_view:nullptr, p.t, c.t, args);
      Close(flat_q.read(), flat_want, 1e-3f, 1e-3f);
      Close(flat_k.read(), key?flat_want:flat_x, 1e-3f, 1e-3f);
    }
  }
}

TEST_CASE("CUDA FP16 scalar direct RoPE rounds cache trigonometry to the model format") {
  Session s;
  auto x = Random(2*2*32);
  for (auto& z:x)z = Round(z, DType::kF16);
  auto xp = Pack(x, DType::kF16);
  int32_t pos[] = {
    3, 101
  };
  Buffer q(s, DType::kF16, {2, 2, 32}, xp.data()), k(s, DType::kF16, {2, 2, 32}, xp.data()), p(s, DType::kI32, {2}, pos);
  RopeArgs args;
  args.rotary_dim = 32;
  RopeNeox(s.q, q.t, k.t, p.t, args);
  auto want = x;
  for (int t = 0; t<2; ++t)
  for (int h = 0; h<2; ++h)
  for (int j = 0; j<16; ++j) {
    size_t o = (t*2+h)*32;
    double angle = pos[t]*std::pow(10000., -2.*j/32);
    float c = Round(float(std::cos(angle)), DType::kF16), sn = Round(float(std::sin(angle)), DType::kF16);
    want[o+j] = Round(std::fma(-x[o+j+16], sn, x[o+j]*c), DType::kF16);
    want[o+j+16] = Round(std::fma(x[o+j], sn, x[o+j+16]*c), DType::kF16);
  }
  Close(q.read(), want, 1e-3f, 1e-3f);
  Close(k.read(), want, 1e-3f, 1e-3f);
}

TEST_CASE("CUDA FP16 scalar admission belongs to the selected provider") {
  // Existing four-field aggregate initializers remain source compatible.
  // These probe providers have no executable operation and never touch memory.
  const auto noop = +[]() {
  };
  OpProvider old_shape {
    "fp16-lower", 100, nullptr, reinterpret_cast<void*>(noop)
  };
  RegisterOpProvider(OpId::kDropinProbe, DeviceType::kCPU, old_shape);
  RegisterOpScalarType(OpId::kDropinProbe, DeviceType::kCPU, "fp16-lower", DType::kF16);
  CHECK(OpSupportsScalarType(OpId::kDropinProbe, DeviceType::kCPU, DType::kF16));
  RegisterOpProvider(OpId::kDropinProbe, DeviceType::kCPU, {"untyped-higher", 101, nullptr, reinterpret_cast<void*>(noop)});
  CHECK_FALSE(OpSupportsScalarType(OpId::kDropinProbe, DeviceType::kCPU, DType::kF16));
  CHECK_FALSE(OpSupportsScalarType(OpId::kAdd, DeviceType::kCPU, DType::kF16));
  // Untouched shared operations retain their output refusals.
  Session s;
  auto raw = Pack({1, 2, 3, 4}, DType::kF16);
  Buffer x(s, DType::kF16, {1, 4}, raw.data()), o(s, DType::kF16, {1, 2});
  CHECK_THROWS(GeluAndMul(s.q, o.t, x.t));
}

TEST_CASE("CUDA FP16 scalar SiLU rounds before multiplication exactly") {
  Session s;
  const auto merged = Pack({-2.703125f, 3.140625f}, DType::kF16);
  const auto gate = Pack({-2.703125f}, DType::kF16);
  const auto up = Pack({3.140625f}, DType::kF16);
  Buffer x(s, DType::kF16, {1, 2}, merged.data());
  Buffer g(s, DType::kF16, {1, 1}, gate.data());
  Buffer u(s, DType::kF16, {1, 1}, up.data());
  Buffer out(s, DType::kF16, {1, 1});
  SiluAndMul(s.q, out.t, x.t);
  CHECK(out.read()[0] == -0.53271484375f);
  MoeSiluMul(s.q, out.t, g.t, u.t);
  CHECK(out.read()[0] == -0.53271484375f);
}

TEST_CASE("CUDA FP16 scalar norm strided rows bypass contiguous fast kernels") {
  Session s;
  for (auto dt : {DType::kF32, DType::kBF16, DType::kF16}) {
    constexpr int tokens = 7, hidden = 5120, stride = 2 * hidden;
    std::vector<float> x(tokens * stride, 0.25f), residual(tokens * stride, 0.5f);
    for (int t = 0; t < tokens; ++t) {
      std::fill(x.begin() + t * stride + hidden, x.begin() + (t + 1) * stride, -7.0f);
      std::fill(residual.begin() + t * stride + hidden,
                residual.begin() + (t + 1) * stride, 9.0f);
    }
    auto xp = Pack(x, dt), rp = Pack(residual, dt);
    auto wp = Pack(std::vector<float>(hidden, 1.0f), dt);
    Buffer dx(s, dt, {tokens, stride}, xp.data());
    Buffer dr(s, dt, {tokens, stride}, rp.data());
    Buffer weight(s, dt, {hidden}, wp.data());
    Buffer out(s, dt, {tokens, hidden});
    dx.t.shape[1] = hidden;
    dr.t.shape[1] = hidden;
    RmsNorm(s.q, out.t, dx.t, weight.t, RmsNormArgs{}, &dr.t);
    Close(out.read(), std::vector<float>(tokens * hidden, Round(0.75f / std::sqrt(0.5625f + 1e-6f), dt)), 1e-5f, 1e-5f);
    for (int t = 0; t < tokens; ++t)
      std::fill(residual.begin() + t * stride, residual.begin() + t * stride + hidden, 0.75f);
    dr.t.shape[1] = stride;
    Close(dr.read(), residual, 0, 0);
  }
}

TEST_CASE("CUDA FP16 scalar quantized embedding stores all resident codecs as half") {
  Session s;
  const DType codecs[] = {
      DType::kQ4_0, DType::kQ5_0, DType::kQ8_0, DType::kQ2_K,
      DType::kQ3_K, DType::kQ4_K, DType::kQ5_K, DType::kQ6_K,
      DType::kQ8_K, DType::kIQ2_XXS, DType::kIQ3_XXS, DType::kIQ2_S,
      DType::kIQ2_XS, DType::kIQ1_S, DType::kIQ1_XXXS, DType::kIQ4_NL,
      DType::kIQ4_XS, DType::kMXFP4, DType::kIQ3_S};
  for (DType dt : codecs) {
    CAPTURE(std::string(Name(dt)));
    constexpr int rows = 3;
    const int width = 2 * BlockElems(dt), bytes = BlockBytes(dt);
    std::vector<uint8_t> packed(rows * 2 * bytes);
    std::vector<float> decoded(rows * width);
    std::mt19937 rng(7);
    const auto decode = vt::cpu::BlockToFloat(dt);
    REQUIRE(decode != nullptr);
    for (int block = 0; block < rows * 2; ++block) {
      auto* data = packed.data() + block * bytes;
      auto* values = decoded.data() + block * BlockElems(dt);
      for (int attempt = 0; ; ++attempt) {
        REQUIRE(attempt < 200);
        for (int i = 0; i < bytes; ++i) data[i] = uint8_t(rng());
        decode(data, values, BlockElems(dt));
        bool finite = true;
        for (int i = 0; i < BlockElems(dt); ++i)
          finite = finite && std::isfinite(values[i]) && std::fabs(values[i]) < 60000.0f;
        if (finite) break;
      }
    }
    Buffer table(s, dt, {rows, width}, packed.data());
    for (bool ids64 : {false, true}) {
      const int32_t i32[] = {2, 0, 1, 2, 0};
      const int64_t i64[] = {2, 0, 1, 2, 0};
      Buffer ids(s, ids64 ? DType::kI64 : DType::kI32, {5},
                 ids64 ? static_cast<const void*>(i64) : static_cast<const void*>(i32));
      Buffer out(s, DType::kF16, {5, width});
      Embedding(s.q, out.t, table.t, ids.t);
      std::vector<float> want(5 * width);
      for (int t = 0; t < 5; ++t)
        for (int j = 0; j < width; ++j)
          want[t * width + j] = Round(decoded[i32[t] * width + j], DType::kF16);
      Close(out.read(), want, 0, 0);
    }
  }
}

TEST_CASE("CUDA FP16 scalar norm rounds the residual before variance") {
  Session s;
  const auto x = Pack({-0.14794921875f, -0.5068359375f, -1.4462890625f,
                       1.4658203125f}, DType::kF16);
  const auto residual = Pack({-1.974609375f, 0.01113128662109375f,
                              1.5927734375f, -1.6767578125f}, DType::kF16);
  const auto gamma = Pack({1, 1, 1, 1}, DType::kF16);
  Buffer input(s, DType::kF16, {1, 4}, x.data());
  Buffer r(s, DType::kF16, {1, 4}, residual.data());
  Buffer weight(s, DType::kF16, {4}, gamma.data());
  Buffer out(s, DType::kF16, {1, 4});
  RmsNorm(s.q, out.t, input.t, weight.t, RmsNormArgs{}, &r.t);
  Close(r.read(), {-2.123046875f, -0.49560546875f, 0.146484375f,
                   -0.2109375f}, 0, 0);
  Close(out.read(), {-1.9345703125f, -0.451416015625f, 0.1334228515625f,
                     -0.192138671875f}, 0, 0);
}
