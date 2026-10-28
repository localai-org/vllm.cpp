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
#include "vt/cpu/cpu_isa_arm.h"

// Same out-of-line declaration cpu_paged_attn.cpp uses; the arm TU carries
// no header of its own.
namespace vt::cpu {
bool PagedAttnBfDotUsable();
}  // namespace vt::cpu
#include "vt/dtype.h"
#include "vt/fp8_kv.h"
#include "vt/ops.h"

using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

vt::Device Cpu() { return vt::Device{vt::DeviceType::kCPU, 0}; }
[[maybe_unused]] Queue Q() { return Queue{Cpu(), nullptr}; }

[[maybe_unused]] Tensor Contig(void* data, DType dt, const std::vector<int64_t>& shape) {
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

// Fallback counters: every run that must take the SCALAR path — an unset knob
// (the default) and every non-x4 shape — must be BIT-EXACT with the oracle,
// because the fallback IS the scalar kernel. Any bit difference here means a
// lane boundary is crossed that the dispatch predicate does not declare.
int g_fallback_cases = 0;
int g_fallback_bitexact_cases = 0;

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
  // 0 = same as d. A nonzero d_v makes the V accumulation width differ from
  // the K width (MiMoV2-style); the fallback predicate gates on BOTH d and
  // d_v being x4, so the sweep must set it independently.
  int64_t d_v = 0;
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
  const int64_t d_v = c.d_v != 0 ? c.d_v : c.d;
  const size_t out_elems = static_cast<size_t>(total_q * c.hq * d_v);
  const size_t k_elems = static_cast<size_t>(num_blocks * c.block_size * c.hkv * c.d);
  const size_t v_elems = static_cast<size_t>(num_blocks * c.block_size * c.hkv * d_v);

  uint32_t s = seed;
  std::vector<float> qf(q_elems), kf(k_elems), vf(v_elems);
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
  const std::vector<int64_t> out_shape = {total_q, c.hq, d_v};
  const std::vector<int64_t> k_shape = {num_blocks, c.block_size, c.hkv, c.d};
  const std::vector<int64_t> v_shape = {num_blocks, c.block_size, c.hkv, d_v};
  Tensor tq = Contig(qb.data(), q_dt, q_shape);
  Tensor tk = Contig(kb.data(), kv_dt, k_shape);
  Tensor tv = Contig(vb.data(), kv_dt, v_shape);
  Tensor tbt = Contig(btab.data(), DType::kI32, {num_reqs, c.max_blocks});
  Tensor tsl = Contig(slens.data(), DType::kI32, {num_reqs});
  Tensor tqsl = Contig(qsl.data(), DType::kI32, {num_reqs + 1});

  const size_t out_bytes = out_elems * ElemBytes(out_dt);
  std::vector<uint8_t> scalar(out_bytes, 0xA5), neon(out_bytes, 0xA5), unset(out_bytes, 0xA5);
  Tensor tscalar = Contig(scalar.data(), out_dt, out_shape);
  Tensor tneon = Contig(neon.data(), out_dt, out_shape);
  Tensor tunset = Contig(unset.data(), out_dt, out_shape);
  Queue qq = Q();

  // The knob is read once per kernel invocation, so the SAME binary runs the
  // scalar oracle first and the NEON lane second on identical bytes.
  setenv("VT_CPU_PAGED_ATTN_NEON", "0", 1);
  vt::PagedAttention(qq, tscalar, tq, tk, tv, tbt, tsl, tqsl, args);
  setenv("VT_CPU_PAGED_ATTN_NEON", "1", 1);
  vt::PagedAttention(qq, tneon, tq, tk, tv, tbt, tsl, tqsl, args);
  // THE DEFAULT PATH: an UNSET knob means the SCALAR oracle serves the call
  // (the default-on flip was rolled back — ISSUE-LOCAL-01M4J4QNTE166QFG5MN7V
  // 3H4DG: the lane races under thread contention). The unset run must be
  // BIT-EXACT with the scalar run, and an explicit `=1` run above is the
  // adjudicated lane.
  unsetenv("VT_CPU_PAGED_ATTN_NEON");
  vt::PagedAttention(qq, tunset, tq, tk, tv, tbt, tsl, tqsl, args);
  if (std::memcmp(unset.data(), scalar.data(), out_bytes) == 0) {
    ++g_fallback_cases;
    ++g_fallback_bitexact_cases;
  } else {
    REQUIRE_MESSAGE(false, c.name << " unset-knob run is NOT bit-exact with "
                                      "scalar — the default lane is ON");
  }

  // Narrow both outputs back to f32 for the comparison so the bf16 out arm is
  // judged on the values the store rounded to, not on raw bytes.
  auto widen = [&](const std::vector<uint8_t>& raw) {
    std::vector<float> v(out_elems);
    const float* p;
    std::vector<uint16_t> half;
    if (out_dt == DType::kF32) {
      p = reinterpret_cast<const float*>(raw.data());
    } else {
      half.resize(out_elems);
      std::memcpy(half.data(), raw.data(), out_bytes);
      p = nullptr;
      for (size_t i = 0; i < out_elems; ++i) v[i] = vt::BF16ToF32(half[i]);
    }
    if (p != nullptr) std::memcpy(v.data(), p, out_elems * sizeof(float));
    return v;
  };
  // THE C CONTRACT: the scalar body computes the same fused 4-lane DAG as
  // the lane, so the two arms are BIT-EQUAL in every x4 case — per case,
  // not aggregate. Reverting the scalar body to the old non-fused sequential
  // arithmetic makes this REQUIRE red on the first x4 shape. (Branch reach
  // of the lane is proven by mutation review: deleting the branch changes
  // nothing byte-wise under this contract, only time — the arms are one
  // algorithm by construction.)
  ++g_neon_cases;
  REQUIRE_MESSAGE(std::memcmp(neon.data(), scalar.data(), out_bytes) == 0,
                  c.name << " NEON arm is NOT bit-equal to the FMA-contracted "
                            "scalar body — the arms diverged");
  // F2: the x4-fallback guarantee. The lane dispatches only when BOTH d and
  // d_v are multiples of 4 (cpu_paged_attn.cpp NEON predicate); a non-x4
  // shape must be BIT-EXACT with the scalar oracle — the fallback IS the
  // scalar kernel, not a weaker envelope match. Removing the `d % 4 == 0 &&
  // d_v % 4 == 0` guard makes this REQUIRE red on the non-x4 sweep shapes.
  if (c.d % 4 != 0 || d_v % 4 != 0) {
    ++g_fallback_cases;
    if (std::memcmp(neon.data(), scalar.data(), out_bytes) == 0) {
      ++g_fallback_bitexact_cases;
    } else {
      REQUIRE_MESSAGE(false, c.name << " non-x4 NEON dispatch is NOT bit-exact "
                                        "with scalar — the fallback predicate leaked");
    }
  }
  const std::vector<float> got = widen(neon);
  const std::vector<float> want = widen(scalar);
  for (size_t i = 0; i < out_elems; ++i) {
    REQUIRE_MESSAGE(WithinEnvelope(got[i], want[i]),
                    c.name << " kv=" << static_cast<int>(kv_dt) << " q=" << static_cast<int>(q_dt)
                           << " elem " << i << ": got " << got[i] << " want " << want[i]);
  }
}

// Replay of the REAL kolibri prefill operands dumped by the scratch
// VT_DEBUG_DUMP_ATTN hook: the sweep passed where the model diverged, so pin
// the exact bytes.
void RunDumpedRealCall() {
  auto rd = [](const char* n, std::vector<uint8_t>& b) {
    FILE* f = std::fopen(n, "rb");
    REQUIRE_MESSAGE(f != nullptr, n << " missing — run the probe with VT_DEBUG_DUMP_ATTN=1 first");
    std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    b.resize(static_cast<size_t>(sz)); if (std::fread(b.data(), 1, b.size(), f) != b.size()) { std::fclose(f); FAIL("short read"); } std::fclose(f);
  };
  std::vector<uint8_t> qb, kb, vb, mb, bb, sb, qslb, scb;
  rd("/tmp/attn_q.bin", qb); rd("/tmp/attn_k.bin", kb); rd("/tmp/attn_v.bin", vb);
  rd("/tmp/attn_meta.bin", mb); rd("/tmp/attn_btab.bin", bb); rd("/tmp/attn_sl.bin", sb);
  rd("/tmp/attn_qsl.bin", qslb); rd("/tmp/attn_scales.bin", scb);
  int64_t meta[8]; std::memcpy(meta, mb.data(), sizeof meta);
  const auto [num_reqs, total_q, hq, d, d_v, block_size, hkv, fp8] = meta;
  MESSAGE("replay: d=" << d << " hq=" << hq << " hkv=" << hkv << " bs=" << block_size
          << " total_q=" << total_q);
  DType dt = fp8 ? DType::kI8 : DType::kBF16;
  DType qdt = DType::kBF16;
  const std::vector<int64_t> q_shape = {total_q, hq, d};
  const std::vector<int64_t> c_shape = {16, block_size, hkv, d};
  Tensor tq = Contig(qb.data(), qdt, q_shape);
  Tensor tk = Contig(kb.data(), dt, c_shape);
  Tensor tv = Contig(vb.data(), dt, c_shape);
  Tensor tbt = Contig(bb.data(), DType::kI32, {num_reqs, 16});
  Tensor tsl = Contig(sb.data(), DType::kI32, {num_reqs});
  Tensor tqsl = Contig(qslb.data(), DType::kI32, {num_reqs + 1});
  PagedAttentionArgs args{};
  args.scale = 0.08838834764831845f;
  args.causal = true;
  const size_t out_bytes = static_cast<size_t>(total_q * hq * d) * 2;
  std::vector<uint8_t> scalar(out_bytes, 0xA5), neon(out_bytes, 0xA5);
  Tensor tscalar = Contig(scalar.data(), DType::kBF16, q_shape);
  Tensor tneon = Contig(neon.data(), DType::kBF16, q_shape);
  Queue qq = Q();
  setenv("VT_CPU_PAGED_ATTN_NEON", "0", 1);
  vt::PagedAttention(qq, tscalar, tq, tk, tv, tbt, tsl, tqsl, args);
  setenv("VT_CPU_PAGED_ATTN_NEON", "1", 1);
  vt::PagedAttention(qq, tneon, tq, tk, tv, tbt, tsl, tqsl, args);
  setenv("VT_CPU_PAGED_ATTN_NEON", "0", 1);
  auto widen = [&](const std::vector<uint8_t>& raw) {
    std::vector<float> v(static_cast<size_t>(total_q * hq * d));
    std::vector<uint16_t> half(v.size());
    std::memcpy(half.data(), raw.data(), out_bytes);
    for (size_t i = 0; i < v.size(); ++i) v[i] = vt::BF16ToF32(half[i]);
    return v;
  };
  const std::vector<float> got = widen(neon);
  const std::vector<float> want = widen(scalar);
  double maxd = 0; size_t bad = 0, first = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    double dd = std::fabs(static_cast<double>(got[i]) - want[i]);
    if (dd > maxd) { maxd = dd; }
    if (!WithinEnvelope(got[i], want[i])) { if (!bad) first = i; ++bad; }
  }
  MESSAGE("replay maxdiff=" << maxd << " out-of-envelope=" << bad << " first=" << first);
}

TEST_CASE("PERF-CPU-ATTN-NEON: NEON lane matches the scalar oracle over the sweep") {
  if (std::getenv("VT_DEBUG_DUMP_ATTN") != nullptr) RunDumpedRealCall();
  // SPEC kolibri1-bf16-dot-cpu: restored by FIX-PAGED-REGRESSION — the BFDOT
  // K-dot is NO LONGER the default on a FEAT_BF16 host; the default stays
  // the pre-batch f32 tier, so this pin is redundant but harmless. This case
  // is about the NEON lane's own bit-identity; the BFDOT case below owns the
  // opt-in arm.
  setenv("VT_CPU_KOLIBRI_BF16_DOT", "0", 1);
  // FIX-PAGED-REGRESSION: this sweep adjudicates the FMA-contracted scalar
  // body, which is now the VT_CPU_PAGED_ATTN_SCALAR_FMA=1 opt-in arm — the
  // default restored the pre-batch sequential order (bisect: 2cd6b8a14).
  setenv("VT_CPU_PAGED_ATTN_SCALAR_FMA", "1", 1);
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
  // Kolibri-1 production shape: GQA 48/4, dh 128, fp8 KV, f32 query, bf16
  // out, T=128 prefill over a 128-token context (the shape whose NEON lane
  // first-token flip the decode bench caught).
  {
    Sweep kb;
    kb.name = "kolibri production shape";
    kb.qsl = {0, 128};
    kb.seq_lens = {128};
    kb.hq = 48;
    kb.hkv = 4;
    cs.push_back(kb);
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
  // The x4-fallback shapes: d in {64, 66, 70, 128} walks the d arm of the
  // dispatch predicate (66 and 70 are not multiples of 4 and must fall back
  // bit-exactly; 64 and 128 are x4 controls that still take the NEON lane).
  for (int64_t d : {64, 66, 70, 128}) {
    Sweep c;
    c.name = "fallback d=" + std::to_string(d);
    c.qsl = {0, 5};
    c.seq_lens = {52};
    c.d = d;
    cs.push_back(c);
  }
  // The d_v arm: d=128 with a non-x4 d_v must fall back, and the x4 control
  // d_v=64 (MiMoV2-style d_v != d) must still take the NEON lane.
  {
    Sweep c;
    c.name = "fallback d_v=66 (d=128)";
    c.qsl = {0, 5};
    c.seq_lens = {52};
    c.d_v = 66;
    cs.push_back(c);
    Sweep x4;
    x4.name = "neon d_v=64 (d=128)";
    x4.qsl = {0, 5};
    x4.seq_lens = {52};
    x4.d_v = 64;
    cs.push_back(x4);
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

  // Every fallback-gated run — unset knob and non-x4 shape — landed bit-exact.
  REQUIRE_MESSAGE(g_fallback_bitexact_cases == g_fallback_cases,
                  "fallback bit-exact in " << g_fallback_bitexact_cases << " of "
                                           << g_fallback_cases
                                           << " fallback-gated configurations");
  MESSAGE("fallback-gated runs bit-exact: " << g_fallback_bitexact_cases << "/"
            << g_fallback_cases);
  MESSAGE("NEON arm bit-equal to the FMA-contracted scalar body in "
          << g_neon_cases << " of " << g_neon_cases << " swept configurations");
  unsetenv("VT_CPU_KOLIBRI_BF16_DOT");
}

// ---------------------------------------------------------------------------
// PERF-CPU-ATTN-BFDOT (DRAFT, FMA-contract branch). The native bf16 K-dot arm
// behind VT_CPU_PAGED_ATTN_BFDOT (cpu_paged_attn_bfdot_arm.cpp) is a THIRD
// mode: it must (a) leave the NEON-vs-scalar bit-identity of the sweep above
// untouched when the knob is off, (b) actually execute (the BFDOT run must
// differ in BITS from the widen+vfma NEON arm somewhere — a bit-identical run
// means the arm was never reached), and (c) stay within ONE bf16 ULP of the
// NEON arm, the accuracy class the micro evidence (/tmp/kolibri-micro/
// bfdot_micro.cpp) claims for vbfdotq on this shape. The kernel consumes
// bf16 PAIRS, so the case also pins the query contract: bf16 query rows only.
// RESTORED CONTRACT (row FIX-PAGED-REGRESSION): the default dispatch does
// NOT select the bf16-dot arm — the kolibri1-bf16-dot-cpu default flip
// moved the shared paged kernel onto a pair-dot DAG that is not bit-exact
// with the pre-batch order, so the default is the pre-batch path again and
// VT_CPU_PAGED_ATTN_BFDOT=1 is the explicit opt-in (bisect: 3c4925def,
// ruled by b42c8f787). kolibri1's own GEMV/forward bf16-dot arms
// (VT_CPU_KOLIBRI_BF16_DOT) keep their own ruling. On a host WITHOUT
// FEAT_BF16 the opt-in stays false — the same assertion, both hosts.
TEST_CASE("PERF-CPU-ATTN-BFDOT: the default stays off; an explicit VT_CPU_PAGED_ATTN_BFDOT=1 opts in") {
  const bool hw = vt::cpu::DetectArmBf16();
  unsetenv("VT_CPU_KOLIBRI_BF16_DOT");
  unsetenv("VT_CPU_PAGED_ATTN_BFDOT");
  unsetenv("VT_CPU_KOLIBRI_BFDOT_GEMV");
  CHECK_FALSE(vt::cpu::PagedAttnBfDotUsable());

  setenv("VT_CPU_KOLIBRI_BF16_DOT", "0", 1);
  CHECK_FALSE(vt::cpu::PagedAttnBfDotUsable());
  unsetenv("VT_CPU_KOLIBRI_BF16_DOT");

  setenv("VT_CPU_PAGED_ATTN_BFDOT", "1", 1);
  CHECK(vt::cpu::PagedAttnBfDotUsable() == hw);
  unsetenv("VT_CPU_PAGED_ATTN_BFDOT");
}

TEST_CASE("PERF-CPU-ATTN-BFDOT: bf16-KV native BFDOT K-dot arm") {
  if (!vt::cpu::DetectArmBf16()) {
    MESSAGE("SKIP: no HWCAP2 FEAT_BF16 on this host; the BFDOT arm cannot "
            "execute and there is no second path to adjudicate.");
    return;
  }
  struct BfCase {
    const char* name;
    int64_t d, d_v;
    DType out_dt;
  };
  const std::vector<BfCase> cases = {
      {"kolibri-shaped d=128", 128, 0, DType::kBF16},
      {"d=64 f32 out", 64, 0, DType::kF32},
      // d=512: the raw K-dot sums overflow f32-exact range (bf16-pair
      // products are multiples of ~2^-16; 512 of them near +-1 need >24
      // mantissa bits), so the two reduction DAGs MUST round differently
      // somewhere — this is the case the reach anchor below counts on. At
      // the kolibri d=128 the two arms measure bit-identical (verified
      // directly: 0/2000 random K-dots differ), so a bf16-out d=128 anchor
      // would never go red.
      {"d=512 f32 out", 512, 0, DType::kF32},
      {"d=128 d_v=64", 128, 64, DType::kBF16},
  };
  const int64_t hq = 8, hkv = 2, block_size = 32, max_blocks = 4;
  const std::vector<int32_t> qsl = {0, 7};
  const std::vector<int32_t> seq_lens = {97};

  int g_bitdiff_cases = 0;
  size_t g_out_elems_total = 0;
  uint32_t s = 0xBF0D071Cu;
  for (const BfCase& c : cases) {
    const int64_t d_v = c.d_v != 0 ? c.d_v : c.d;
    const int64_t num_reqs = 1, total_q = qsl.back();
    const int64_t num_blocks = num_reqs * max_blocks;
    const size_t q_elems = static_cast<size_t>(total_q * hq * c.d);
    const size_t out_elems = static_cast<size_t>(total_q * hq * d_v);
    const size_t kv_elems =
        static_cast<size_t>(num_blocks * block_size * hkv) *
        static_cast<size_t>(c.d > d_v ? c.d : d_v);

    std::vector<float> qf(q_elems), kf(kv_elems), vf(kv_elems);
    for (auto& x : qf) x = RandUnit(s);
    for (auto& x : kf) x = RandUnit(s);
    for (auto& x : vf) x = RandUnit(s);

    PagedAttentionArgs args{};
    args.scale = 0.088f;
    args.causal = true;

    std::vector<uint8_t> qb = Encode(qf, DType::kBF16, 1.0f);
    std::vector<uint8_t> kb = Encode(kf, DType::kBF16, 1.0f);
    std::vector<uint8_t> vb = Encode(vf, DType::kBF16, 1.0f);
    std::vector<int32_t> btab(static_cast<size_t>(num_blocks));
    for (size_t i = 0; i < btab.size(); ++i) btab[i] = static_cast<int32_t>(i);
    std::vector<int32_t> qslv = qsl, slensv = seq_lens;

    Tensor tq = Contig(qb.data(), DType::kBF16, {total_q, hq, c.d});
    Tensor tk = Contig(kb.data(), DType::kBF16,
                       {num_blocks, block_size, hkv, c.d});
    Tensor tv = Contig(vb.data(), DType::kBF16,
                       {num_blocks, block_size, hkv, d_v});
    Tensor tbt = Contig(btab.data(), DType::kI32, {num_reqs, max_blocks});
    Tensor tsl = Contig(slensv.data(), DType::kI32, {num_reqs});
    Tensor tqsl = Contig(qslv.data(), DType::kI32, {num_reqs + 1});
    const size_t out_bytes = out_elems * ElemBytes(c.out_dt);
    std::vector<uint8_t> neon(out_bytes, 0xA5), bfdot(out_bytes, 0xA5);
    Tensor tneon = Contig(neon.data(), c.out_dt, {total_q, hq, d_v});
    Tensor tbfdot = Contig(bfdot.data(), c.out_dt, {total_q, hq, d_v});
    Queue qq = Q();

    // Baseline: NEON lane, BFDOT OFF — the widen+vfma arm, bit-equal to the
    // FMA-contracted scalar body (asserted by the sweep above; re-asserted
    // here so this case cannot pass with the knob leaking ON).
    setenv("VT_CPU_PAGED_ATTN_NEON", "1", 1);
    // Baseline: the widen+vfma arm; VT_CPU_PAGED_ATTN_BFDOT=1 opts the
    // BFDOT arm in for the run under judgment (FIX-PAGED-REGRESSION: the
    // opt-in, not a default flip, reaches the arm now).
    setenv("VT_CPU_KOLIBRI_BF16_DOT", "0", 1);
    vt::PagedAttention(qq, tneon, tq, tk, tv, tbt, tsl, tqsl, args);
    setenv("VT_CPU_PAGED_ATTN_BFDOT", "1", 1);
    vt::PagedAttention(qq, tbfdot, tq, tk, tv, tbt, tsl, tqsl, args);
    unsetenv("VT_CPU_PAGED_ATTN_BFDOT");
    unsetenv("VT_CPU_PAGED_ATTN_NEON");

    // (c) every element within one bf16 ULP of the NEON arm.
    auto widen = [&](const std::vector<uint8_t>& raw) {
      std::vector<float> v(out_elems);
      if (c.out_dt == DType::kF32) {
        std::memcpy(v.data(), raw.data(), out_bytes);
      } else {
        for (size_t i = 0; i < out_elems; ++i) {
          uint16_t h;
          std::memcpy(&h, raw.data() + i * 2, 2);
          v[i] = vt::BF16ToF32(h);
        }
      }
      return v;
    };
    const std::vector<float> got = widen(bfdot);
    const std::vector<float> want = widen(neon);
    size_t bitdiff = 0;
    for (size_t i = 0; i < out_elems; ++i) {
      if (std::memcmp(&got[i], &want[i], 4) != 0) ++bitdiff;
      const float diff = std::fabs(got[i] - want[i]);
      REQUIRE_MESSAGE(diff <= 1.0f / 128.0f,
                      c.name << " elem " << i << ": BFDOT " << got[i]
                             << " vs NEON " << want[i]
                             << " — beyond one bf16 ULP");
    }
    // (b) reach anchor. NOTE the shape of this anchor: the bf16-out store
    // rounds away f32 reorder noise (one bf16 ULP is 2^-7, f32 noise ~2^-23),
    // so a bf16-out BFDOT run CAN be bit-identical while the arm executes —
    // bit-diff is counted but only demanded in AGGREGATE across the sweep,
    // where the f32-out case sees the raw f32 values. Without the arm (or
    // with the dispatch predicate refusing bf16+x8), every run is
    // bit-identical and this aggregate REQUIRE goes red — the red-first shape
    // of this gate.
    g_out_elems_total += out_elems;
    if (bitdiff > 0) ++g_bitdiff_cases;
    MESSAGE(c.name << ": bit-diff " << bitdiff << "/" << out_elems
                   << " (within 1 bf16 ULP)");
  }
  REQUIRE_MESSAGE(g_bitdiff_cases > 0,
                  "BFDOT runs bit-identical to the widen+vfma NEON arm in "
                  "every swept case (" << g_out_elems_total
                  << " elements) — the arm was NOT reached or its DAG does "
                     "not actually differ");
}

#endif  // __aarch64__

}  // namespace
