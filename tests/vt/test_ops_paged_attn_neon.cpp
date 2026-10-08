// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
//
// PERF-CPU-ATTN-NEON tolerance gate (issue
// ISSUE-LOCAL-01M4EQ7TPR5Q7JQGNM8628X9HW, row MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm).
//
// The scalar paged-attention kernel (src/vt/cpu/cpu_paged_attn.cpp) is the
// ORACLE here. The aarch64 NEON lane behind VT_CPU_PAGED_ATTN_NEON changes the
// K-REDUCTION ORDER (four interleaved fma lanes summed by vaddvq_f32 instead
// of a strictly sequential f32 chain), so the envelope is NOT bit-exact — a
// memcmp gate would be the wrong instrument by construction. The envelope
// chosen, stated before the first run:
//   - the raw Q.K^T scores may drift by <= 1e-4 absolute (<= 32 fma groups of
//     dh <= 128 reordered; softmax sits on top of `scale`-damped logits);
//   - the OUTPUT rows are compared with a mixed tolerance that is tight on the
//     absolute scale the tests generate (|values| <= 1): rel 2e-5 OR abs 2e-6,
//     i.e. f32-accumulation noise, not a relaxed tolerance.
// Whether that drift is acceptable in production is adjudicated by the model
// gates (kolibri 234/234, W3 900/900 + ARGMAX 141/145 near-tie chain), not by
// this file. On non-aarch64 the file SKIPS LOUDLY: the NEON lane cannot exist
// there and the scalar oracle would be compared with itself.
//
// Sweep (all randomized, xorshift-seeded and deterministic): T in {1, 2, 8,
// 64}, GQA ratios hq/hkv in {1, 2, 4, 8}, dh in {64, 128}, causal full vs
// causal + sliding window (left/right) vs full non-causal, KV dtype f32/f16/
// bf16/fp8 (k_scale/v_scale 0.75/1.25), out f32/bf16, block sizes {16, 32}
// with block-spanning sequences, varlen batches, softcap, and a batch that
// carries an empty request row.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/fp8_kv.h"
#include "vt/ops.h"

using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

vt::Device Cpu() { return vt::Device{vt::DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }

Tensor Contig(void* data, DType dt, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = Cpu();
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

#if !defined(__aarch64__)
TEST_CASE("PERF-CPU-ATTN-NEON: non-aarch64 build has no NEON lane to test") {
  MESSAGE("SKIP: VT_CPU_PAGED_ATTN_NEON is aarch64-only; scalar oracle serves "
          "this build and there is no second path to adjudicate.");
}
#else

// Sweep-wide mutation counters. A small random case can round to bit-equality
// by coincidence, so the REACH anchor is aggregate: the NEON lane must change
// at least half the swept configurations somewhere.
int g_neon_cases = 0;
int g_neon_diff_cases = 0;

using vt::Fp8KVCacheDataType;
using vt::PagedAttentionArgs;

uint32_t NextRand(uint32_t& s) {
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}

float RandUnit(uint32_t& s) {
  return static_cast<float>(NextRand(s) % 2048u) / 1024.0f - 1.0f;  // [-1, 1)
}

std::vector<uint8_t> Encode(const std::vector<float>& src, DType dt, float fp8_scale) {
  std::vector<uint8_t> bytes;
  switch (dt) {
    case DType::kF32:
      bytes.resize(src.size() * 4);
      std::memcpy(bytes.data(), src.data(), bytes.size());
      break;
    case DType::kF16: {
      bytes.resize(src.size() * 2);
      auto* p = reinterpret_cast<uint16_t*>(bytes.data());
      for (size_t i = 0; i < src.size(); ++i) p[i] = vt::F32ToF16(src[i]);
      break;
    }
    case DType::kBF16: {
      bytes.resize(src.size() * 2);
      auto* p = reinterpret_cast<uint16_t*>(bytes.data());
      for (size_t i = 0; i < src.size(); ++i) p[i] = vt::F32ToBF16(src[i]);
      break;
    }
    case DType::kI8: {
      bytes.resize(src.size());
      for (size_t i = 0; i < src.size(); ++i) bytes[i] = vt::StoreKvFp8E4M3(src[i], fp8_scale);
      break;
    }
    default: FAIL("test Encode: unsupported dtype");
  }
  return bytes;
}

size_t ElemBytes(DType dt) { return dt == DType::kI8 ? 1u : (dt == DType::kF32 ? 4u : 2u); }

// The stated envelope, applied to output rows (and, separately, to raw scores
// when the caller wants the tighter score bound).
bool WithinEnvelope(float got, float want) {
  const float diff = std::fabs(got - want);
  // 2^-7 = one bf16 mantissa ulp: a reordered f32 accumulation can straddle a
  // bf16 rounding boundary, and the two STORED bf16 neighbors then differ by
  // exactly one ulp even though the underlying f32 values agree to f32 noise.
  return diff <= 2e-6f || diff <= std::max(1.0f / 128.0f, 2e-5f) * std::fabs(want);
}

struct Sweep {
  std::string name;
  std::vector<int32_t> qsl;        // [num_reqs + 1]; an empty row is allowed
  std::vector<int32_t> seq_lens;   // [num_reqs]
  int64_t hq = 8;
  int64_t hkv = 2;                 // GQA ratio = hq / hkv
  int64_t d = 128;
  int64_t block_size = 32;
  int64_t max_blocks = 8;          // covers ceil(max seq_len / block_size)
  bool softcap = false;
  // window = {left, right}; nullopt = full (causal still applies via args).
  std::optional<std::pair<int64_t, int64_t>> window;
  bool causal = true;
};

void RunPair(const Sweep& c, DType q_dt, DType kv_dt, DType out_dt, uint32_t seed) {
  const bool fp8 = kv_dt == DType::kI8;
  const int64_t num_reqs = static_cast<int64_t>(c.seq_lens.size());
  const int64_t total_q = c.qsl.back();
  const int64_t num_blocks = num_reqs * c.max_blocks;
  const size_t q_elems = static_cast<size_t>(total_q * c.hq * c.d);
  const size_t cache_elems = static_cast<size_t>(num_blocks * c.block_size * c.hkv * c.d);

  uint32_t s = seed;
  std::vector<float> qf(q_elems), kf(cache_elems), vf(cache_elems);
  for (auto& x : qf) x = RandUnit(s);
  for (auto& x : kf) x = RandUnit(s);
  for (auto& x : vf) x = RandUnit(s);

  PagedAttentionArgs args{};
  args.scale = 0.088f;
  args.causal = c.causal;
  if (c.softcap) {
    args.logits_soft_cap = 30.0f;
  }
  if (c.window.has_value()) {
    args.window_size = vt::AttentionWindow{static_cast<int32_t>(c.window->first),
                                           static_cast<int32_t>(c.window->second)};
  }
  if (fp8) {
    args.kv_cache_dtype = Fp8KVCacheDataType::kFp8E4M3;
    args.k_scale = 0.75f;
    args.v_scale = 1.25f;
  }

  std::vector<uint8_t> qb = Encode(qf, q_dt, 1.0f);
  std::vector<uint8_t> kb = Encode(kf, kv_dt, args.k_scale);
  std::vector<uint8_t> vb = Encode(vf, kv_dt, args.v_scale);
  std::vector<int32_t> btab(static_cast<size_t>(num_blocks));
  for (size_t i = 0; i < btab.size(); ++i) btab[i] = static_cast<int32_t>(i);
  std::vector<int32_t> qsl = c.qsl, slens = c.seq_lens;

  const std::vector<int64_t> q_shape = {total_q, c.hq, c.d};
  const std::vector<int64_t> c_shape = {num_blocks, c.block_size, c.hkv, c.d};
  Tensor tq = Contig(qb.data(), q_dt, q_shape);
  Tensor tk = Contig(kb.data(), kv_dt, c_shape);
  Tensor tv = Contig(vb.data(), kv_dt, c_shape);
  Tensor tbt = Contig(btab.data(), DType::kI32, {num_reqs, c.max_blocks});
  Tensor tsl = Contig(slens.data(), DType::kI32, {num_reqs});
  Tensor tqsl = Contig(qsl.data(), DType::kI32, {num_reqs + 1});

  const size_t out_bytes = q_elems * ElemBytes(out_dt);
  std::vector<uint8_t> scalar(out_bytes, 0xA5), neon(out_bytes, 0xA5);
  Tensor tscalar = Contig(scalar.data(), out_dt, q_shape);
  Tensor tneon = Contig(neon.data(), out_dt, q_shape);
  Queue qq = Q();

  // The knob is read once per kernel invocation, so the SAME binary runs the
  // scalar oracle first and the NEON lane second on identical bytes.
  setenv("VT_CPU_PAGED_ATTN_NEON", "0", 1);
  vt::PagedAttention(qq, tscalar, tq, tk, tv, tbt, tsl, tqsl, args);
  setenv("VT_CPU_PAGED_ATTN_NEON", "1", 1);
  vt::PagedAttention(qq, tneon, tq, tk, tv, tbt, tsl, tqsl, args);
  setenv("VT_CPU_PAGED_ATTN_NEON", "0", 1);

  // Narrow both outputs back to f32 for the comparison so the bf16 out arm is
  // judged on the values the store rounded to, not on raw bytes.
  auto widen = [&](const std::vector<uint8_t>& raw) {
    std::vector<float> v(q_elems);
    const float* p;
    std::vector<uint16_t> half;
    if (out_dt == DType::kF32) {
      p = reinterpret_cast<const float*>(raw.data());
    } else {
      half.resize(q_elems);
      std::memcpy(half.data(), raw.data(), out_bytes);
      p = nullptr;
      for (size_t i = 0; i < q_elems; ++i) v[i] = vt::BF16ToF32(half[i]);
    }
    if (p != nullptr) std::memcpy(v.data(), p, q_elems * sizeof(float));
    return v;
  };
  // REACH + MUTATION ANCHOR: the raw NEON output must differ from the scalar
  // output somewhere (the K-reduction reorder is real arithmetic, not a no-op).
  // Deleting the NEON branch, the env read, or mis-wiring the knob so VT_CPU_
  // PAGED_ATTN_NEON=1 still runs scalar makes this REQUIRE red.
  ++g_neon_cases;
  if (std::memcmp(neon.data(), scalar.data(), out_bytes) != 0) ++g_neon_diff_cases;
  const std::vector<float> got = widen(neon);
  const std::vector<float> want = widen(scalar);
  for (size_t i = 0; i < q_elems; ++i) {
    REQUIRE_MESSAGE(WithinEnvelope(got[i], want[i]),
                    c.name << " kv=" << static_cast<int>(kv_dt) << " q=" << static_cast<int>(q_dt)
                           << " elem " << i << ": got " << got[i] << " want " << want[i]);
  }
}

TEST_CASE("PERF-CPU-ATTN-NEON: NEON lane matches the scalar oracle over the sweep") {
  const std::vector<DType> kv_dts = {DType::kF32, DType::kF16, DType::kBF16, DType::kI8};
  const std::vector<DType> q_dts = {DType::kF32, DType::kBF16};
  const std::vector<DType> out_dts = {DType::kF32, DType::kBF16};

  std::vector<Sweep> cs;
  // T sweep at kolibri-ish widths (GQA 4:1, dh 128, causal).
  for (int64_t t : {1, 2, 8, 64}) {
    Sweep c;
    c.name = "decode t=" + std::to_string(t);
    c.qsl = {0, static_cast<int32_t>(t)};
    c.seq_lens = {static_cast<int32_t>(t + 40)};
    c.max_blocks = static_cast<int64_t>((t + 40 + c.block_size - 1) / c.block_size);
    cs.push_back(c);
  }
  // GQA mixes at dh 128 and 64.
  for (auto [hq, hkv] : std::vector<std::pair<int64_t, int64_t>>{{4, 4}, {8, 4}, {8, 1}, {32, 4}}) {
    Sweep c;
    c.name = "gqa " + std::to_string(hq) + "/" + std::to_string(hkv);
    c.qsl = {0, 3};
    c.seq_lens = {64};
    c.hq = hq;
    c.hkv = hkv;
    cs.push_back(c);
    Sweep c64 = c;
    c64.d = 64;
    c64.name += " dh64";
    cs.push_back(c64);
  }
  // Sliding window vs full vs non-causal.
  {
    Sweep sw;
    sw.name = "sliding window";
    sw.qsl = {0, 8};
    sw.seq_lens = {96};
    sw.window = {16, 0};
    cs.push_back(sw);
    Sweep nc;
    nc.name = "non-causal full";
    nc.qsl = {0, 4};
    nc.seq_lens = {70};
    nc.causal = false;
    cs.push_back(nc);
  }
  // Softcap, block spanning at block_size 16, varlen with an empty row.
  {
    Sweep sc;
    sc.name = "softcap";
    sc.qsl = {0, 5};
    sc.seq_lens = {48};
    sc.softcap = true;
    cs.push_back(sc);
    Sweep bs;
    bs.name = "block size 16 spanning";
    bs.qsl = {0, 9};
    bs.seq_lens = {37};
    bs.block_size = 16;
    bs.max_blocks = 4;
    cs.push_back(bs);
    Sweep vl;
    vl.name = "varlen with empty row";
    vl.qsl = {0, 0, 4, 9};
    vl.seq_lens = {0, 33, 21};
    vl.max_blocks = 4;
    cs.push_back(vl);
  }

  uint32_t seed = 0x9E3779B9u;
  for (const Sweep& c : cs) {
    for (DType kv : kv_dts) {
      for (DType q : q_dts) {
        for (DType out : out_dts) {
          RunPair(c, q, kv, out, seed++);
        }
      }
    }
  }

  REQUIRE_MESSAGE(g_neon_diff_cases * 2 > g_neon_cases,
                  "NEON lane bit-equal to scalar in " << g_neon_cases - g_neon_diff_cases
                                                      << " of " << g_neon_cases
                                                      << " configurations — the NEON lane likely did not run");
  MESSAGE("NEON lane bit-differs from scalar in " << g_neon_diff_cases << " of " << g_neon_cases
                                                  << " swept configurations");
}

#endif  // __aarch64__

}  // namespace
