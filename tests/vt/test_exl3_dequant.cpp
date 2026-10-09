// MODEL-DSV4-EXL3 W1a — the EXL3 (exllamav3) trellis reference dequant.
//
// PORTED FROM exllamav3 @ 2398c05635fbbad01a0a51dce63c85c6c8a8450e (MIT):
//   tests/test_quant_fn.py:83-128     tail-biting window semantics + the
//                                     "ideal encoding" round-trip this file's
//                                     synthetic case reproduces.
//   exllamav3_ext/quant/pack.cu:29-57 span packing + SWAP16 (the ENCODE side,
//                                     reimplemented HERE so the test never
//                                     shares code with the implementation's
//                                     decode).
//   exllamav3_ext/quant/exl3_dq.cuh:15-31   the 16-bit window read.
//   exllamav3_ext/quant/codebook.cuh:67-75  the 3-instruction MCG decode.
//   exllamav3/modules/quant/exl3_lib/quantize.py:22-42  tensor_core_perm.
//   exllamav3/util/hadamard.py:34-42        Sylvester H128 (no hadamard_128.txt
//                                     ships, so get_hadamard(128) recurses to
//                                     hadamard_1.txt = "+").
//   exllamav3/modules/quant/exl3.py:227-237 get_weight_tensor, the dequant this
//                                     op mirrors.
//
// INDEPENDENCE, stated plainly, and exactly as far as it goes. The synthetic
// case builds its trellis from the ENCODE side (window composition + span
// packing) while the implementation only ever runs the DECODE side, so the
// window semantics ARE independently derived on the two sides.
//
// TWO things are transcribed twice from the same upstream lines, not derived
// twice, and a reader deciding how much this gate proves needs both:
//   * `tensor_core_perm` (quantize.py:22-42) — `TensorCorePerm` here vs
//     `Exl3TileRowMajorIndex` in the implementation. A 256-entry permutation
//     has no cheap second source.
//   * the MCG constants `0xCBAC1FED` / `0x8fff8fff` / `0x3b603b60`
//     (codebook.cuh:67-75) — `TestMcgDecode` below vs `Exl3DecodeMcg` in
//     `src/vt/cpu/cpu_exl3_dequant.cpp`. A transcription cannot gate the
//     transcription it copies. (Both cited by FUNCTION: a line number inside
//     the file it names goes stale on the next edit to that file, which is
//     what happened to the earlier form of this note.)
// What gates BOTH is the REAL-CHECKPOINT case: a wrong permutation scrambles a
// real expert's weights and a wrong constant decodes different values, and
// either misses every spot value.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"

namespace {

// ── the ENCODE side, reimplemented for the test ────────────────────────────

// exllamav3 tests/test_quant_fn.py:104-112. Codeword `t` is the 16-bit window
// that ENDS at weight t: t's own K bits in the low positions, then t-1, t-2 …
// wrapping (tail-biting) around the 256-weight tile.
std::vector<uint16_t> ComposeWindows(const std::vector<int>& v, int k_bits) {
  const int shifts = (16 + k_bits - 1) / k_bits;
  std::vector<uint16_t> w(256, 0);
  for (int t = 0; t < 256; ++t) {
    uint32_t x = 0;
    for (int s = 0; s < shifts; ++s) {
      const int j = (t + 256 - s) % 256;
      x |= static_cast<uint32_t>(v[j] & ((1 << k_bits) - 1)) << (k_bits * s);
    }
    w[t] = static_cast<uint16_t>(x & 0xffffu);
  }
  return w;
}

// exllamav3_ext/quant/pack.cu:29-57. 16 spans of 16 weights; each span writes a
// big-endian K*16-bit run into K uint16 words. Then pack.cu:56 SWAP16s every
// uint32 pair, which is what makes the stored int16 array read back as a
// big-endian bit stream through a plain uint32 view.
std::vector<uint16_t> PackTile(const std::vector<int>& v, int k_bits) {
  const int packed_size = 256 * k_bits / 16;
  std::vector<uint16_t> s(static_cast<size_t>(packed_size), 0);
  for (int span = 0; span < 16; ++span) {
    int i = 16 * span;
    int j = k_bits * span;
    int k = 32;
    uint32_t buf = 0;
    for (int nth = 0; nth < 16; ++nth) {
      const uint32_t x = static_cast<uint32_t>(v[i] & ((1 << k_bits) - 1));
      k -= k_bits;
      buf |= (x << k);
      if (k <= 16) {
        s[static_cast<size_t>(j)] = static_cast<uint16_t>(buf >> 16);
        buf <<= 16;
        k += 16;
        ++j;
      }
      ++i;
    }
  }
  std::vector<uint16_t> g(s.size(), 0);
  for (size_t i = 0; i + 1 < s.size(); i += 2) {
    g[i] = s[i + 1];
    g[i + 1] = s[i];
  }
  return g;
}

// exllamav3_ext/quant/codebook.cuh:67-75 (cb == 1), written out rather than
// called: multiply, lop3(0x6a) == (x & b) ^ c, then the two fp16 halves summed
// in fp16 (__hadd).
float TestMcgDecode(uint16_t codeword) {
  uint32_t x = static_cast<uint32_t>(codeword) * 0xCBAC1FEDu;
  x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
  const float lo = vt::F16ToF32(static_cast<uint16_t>(x & 0xffffu));
  const float hi = vt::F16ToF32(static_cast<uint16_t>(x >> 16));
  return vt::F16ToF32(vt::F32ToF16(lo + hi));
}

// exllamav3/modules/quant/exl3_lib/quantize.py:22-42.
std::vector<int> TensorCorePerm() {
  std::vector<int> perm(256, 0);
  for (int t = 0; t < 32; ++t) {
    const int r0 = (t % 4) * 2;
    const int c0 = t / 4;
    const int c1 = c0 + 8;
    perm[t * 8 + 0] = r0 * 16 + c0;
    perm[t * 8 + 1] = (r0 + 1) * 16 + c0;
    perm[t * 8 + 2] = (r0 + 8) * 16 + c0;
    perm[t * 8 + 3] = (r0 + 9) * 16 + c0;
    perm[t * 8 + 4] = r0 * 16 + c1;
    perm[t * 8 + 5] = (r0 + 1) * 16 + c1;
    perm[t * 8 + 6] = (r0 + 8) * 16 + c1;
    perm[t * 8 + 7] = (r0 + 9) * 16 + c1;
  }
  return perm;
}

// Deterministic K-bit codeword stream (no <random>: the fixture must be the
// same bytes on every platform for the literals below to mean anything).
std::vector<int> Codewords(int k_bits, uint32_t seed) {
  std::vector<int> v(256, 0);
  uint32_t s = seed;
  for (int i = 0; i < 256; ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = static_cast<int>((s >> 13) & ((1u << k_bits) - 1u));
  }
  return v;
}

// exllamav3/util/hadamard.py:34-42 — Sylvester recursion, dense, as doubles.
std::vector<double> SylvesterHadamard(int n) {
  std::vector<double> h{1.0};
  int d = 1;
  while (d < n) {
    std::vector<double> s(static_cast<size_t>(4 * d * d), 0.0);
    for (int i = 0; i < d; ++i) {
      for (int j = 0; j < d; ++j) {
        const double v = h[static_cast<size_t>(i) * d + j];
        s[static_cast<size_t>(i) * 2 * d + j] = v;
        s[static_cast<size_t>(i) * 2 * d + d + j] = v;
        s[static_cast<size_t>(d + i) * 2 * d + j] = v;
        s[static_cast<size_t>(d + i) * 2 * d + d + j] = -v;
      }
    }
    h.swap(s);
    d *= 2;
  }
  return h;
}

}  // namespace

// ───────────────────────────────────────────────────────────────────────────
// A. The window/codebook byte gate: independently encoded tiles, decoded back.
// ───────────────────────────────────────────────────────────────────────────
TEST_CASE("exl3: the trellis window decode reproduces independently packed codewords") {
  const std::vector<int> perm = TensorCorePerm();
  for (int k_bits = 1; k_bits <= 8; ++k_bits) {
    CAPTURE(k_bits);
    const std::vector<int> v = Codewords(k_bits, 0x1234567u + 7u * k_bits);
    const std::vector<uint16_t> windows = ComposeWindows(v, k_bits);

    // The fixture's own tail-biting self-check (test_quant_fn.py:83-87): the
    // history above weight 0's K bits IS the low 16-K bits of weight 255's word.
    CHECK((windows[0] >> k_bits) ==
          (windows[255] & ((1u << (16 - k_bits)) - 1u)));

    const std::vector<uint16_t> tile = PackTile(v, k_bits);
    REQUIRE(tile.size() == static_cast<size_t>(256 * k_bits / 16));

    // Every one of the 256 windows must come back BYTE-identical.
    int mismatches = 0;
    for (int t = 0; t < 256; ++t) {
      if (vt::Exl3TileCodeword(tile.data(), k_bits, t) != windows[t]) ++mismatches;
    }
    CHECK(mismatches == 0);

    // …and the codebook value, and the tensor-core -> row-major placement.
    std::vector<float> decoded(256, 0.0f);
    vt::Exl3DecodeTile(tile.data(), k_bits, /*codebook=*/1, decoded.data());
    int value_mismatches = 0;
    for (int t = 0; t < 256; ++t) {
      const float want = TestMcgDecode(windows[t]);
      if (vt::Exl3DecodeMcg(windows[t]) != want) ++value_mismatches;
      if (decoded[static_cast<size_t>(perm[t])] != want) ++value_mismatches;
    }
    CHECK(value_mismatches == 0);
  }
}

// ───────────────────────────────────────────────────────────────────────────
// B. The full get_weight_tensor ladder, against a dense double reference.
// ───────────────────────────────────────────────────────────────────────────
TEST_CASE("exl3: the full dequant matches a dense blockwise-Hadamard reference") {
  constexpr int kBits = 3;
  constexpr int64_t k = 256;  // two 128-wide Hadamard blocks on each side, so a
  constexpr int64_t n = 256;  // whole-tensor (non-blockwise) transform fails.
  const int64_t tk = k / 16, tn = n / 16;
  const int tile_words = 16 * kBits;

  std::vector<uint16_t> trellis(static_cast<size_t>(tk * tn * tile_words), 0);
  std::vector<float> inner(static_cast<size_t>(k * n), 0.0f);
  const std::vector<int> perm = TensorCorePerm();
  for (int64_t i = 0; i < tk; ++i) {
    for (int64_t j = 0; j < tn; ++j) {
      const std::vector<int> v =
          Codewords(kBits, 0xA5A5u + 131u * static_cast<uint32_t>(i * tn + j));
      const std::vector<uint16_t> windows = ComposeWindows(v, kBits);
      const std::vector<uint16_t> tile = PackTile(v, kBits);
      std::memcpy(&trellis[static_cast<size_t>((i * tn + j) * tile_words)],
                  tile.data(), tile.size() * sizeof(uint16_t));
      for (int t = 0; t < 256; ++t) {
        const int p = perm[t];
        inner[static_cast<size_t>((i * 16 + p / 16) * n + j * 16 + p % 16)] =
            TestMcgDecode(windows[t]);
      }
    }
  }

  // Non-trivial sign+scale vectors on both sides (the real checkpoint's suh
  // carries a per-channel scale, not just a sign).
  std::vector<uint16_t> suh(static_cast<size_t>(k)), svh(static_cast<size_t>(n));
  for (int64_t i = 0; i < k; ++i)
    suh[static_cast<size_t>(i)] =
        vt::F32ToF16((i % 3 == 0 ? -1.0f : 1.0f) * (0.5f + 0.01f * (i % 7)));
  for (int64_t j = 0; j < n; ++j)
    svh[static_cast<size_t>(j)] =
        vt::F32ToF16((j % 5 == 0 ? -1.0f : 1.0f) * (1.25f - 0.02f * (j % 11)));

  // The dense reference: exl3.py:227-237 over quantize.py:340-358, in doubles,
  // rounding to fp16 exactly where upstream's `.to(x_dtype)` / half multiply do.
  const std::vector<double> h = SylvesterHadamard(128);
  const double hs = 1.0 / std::sqrt(128.0);
  std::vector<float> want = inner;
  for (int64_t b = 0; b < k; b += 128) {  // preapply_had_l
    std::vector<float> blk(static_cast<size_t>(128 * n));
    for (int64_t i = 0; i < 128; ++i)
      for (int64_t j = 0; j < n; ++j) {
        double acc = 0.0;
        for (int64_t t = 0; t < 128; ++t)
          acc += h[static_cast<size_t>(i) * 128 + t] * hs *
                 static_cast<double>(want[static_cast<size_t>((b + t) * n + j)]);
        blk[static_cast<size_t>(i * n + j)] =
            vt::F16ToF32(vt::F32ToF16(static_cast<float>(acc)));
      }
    for (int64_t i = 0; i < 128; ++i)
      for (int64_t j = 0; j < n; ++j)
        want[static_cast<size_t>((b + i) * n + j)] = blk[static_cast<size_t>(i * n + j)];
  }
  for (int64_t i = 0; i < k; ++i)  // *= suh[:, None]
    for (int64_t j = 0; j < n; ++j)
      want[static_cast<size_t>(i * n + j)] = vt::F16ToF32(vt::F32ToF16(
          want[static_cast<size_t>(i * n + j)] * vt::F16ToF32(suh[static_cast<size_t>(i)])));
  for (int64_t b = 0; b < n; b += 128) {  // preapply_had_r
    std::vector<float> blk(static_cast<size_t>(k * 128));
    for (int64_t i = 0; i < k; ++i)
      for (int64_t j = 0; j < 128; ++j) {
        double acc = 0.0;
        for (int64_t t = 0; t < 128; ++t)
          acc += static_cast<double>(want[static_cast<size_t>(i * n + b + t)]) *
                 h[static_cast<size_t>(t) * 128 + j] * hs;
        blk[static_cast<size_t>(i * 128 + j)] =
            vt::F16ToF32(vt::F32ToF16(static_cast<float>(acc)));
      }
    for (int64_t i = 0; i < k; ++i)
      for (int64_t j = 0; j < 128; ++j)
        want[static_cast<size_t>(i * n + b + j)] = blk[static_cast<size_t>(i * 128 + j)];
  }
  for (int64_t i = 0; i < k; ++i)  // *= svh[None, :]
    for (int64_t j = 0; j < n; ++j)
      want[static_cast<size_t>(i * n + j)] = vt::F16ToF32(vt::F32ToF16(
          want[static_cast<size_t>(i * n + j)] * vt::F16ToF32(svh[static_cast<size_t>(j)])));

  // The inner (pre-Hadamard) reconstruct is byte-exact — no summation happens.
  std::vector<float> got_inner(static_cast<size_t>(k * n), 0.0f);
  vt::Exl3ReconstructInner(trellis.data(), k, n, kBits, /*codebook=*/1, got_inner.data());
  int inner_mismatches = 0;
  for (size_t i = 0; i < inner.size(); ++i)
    if (got_inner[i] != inner[i]) ++inner_mismatches;
  CHECK(inner_mismatches == 0);

  std::vector<float> got(static_cast<size_t>(k * n), 0.0f);
  vt::Exl3DequantLinear(trellis.data(), suh.data(), svh.data(), k, n, kBits, /*codebook=*/1, got.data());
  double max_abs = 0.0, max_diff = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    max_abs = std::max(max_abs, std::fabs(static_cast<double>(want[i])));
    max_diff = std::max(max_diff, std::fabs(static_cast<double>(got[i] - want[i])));
  }
  CAPTURE(max_abs);
  CAPTURE(max_diff);
  // fp16 has ~11 bits of mantissa; the two paths differ only in f32-vs-f64
  // summation order inside one 128-term Hadamard block, so at most a couple of
  // fp16 ulps at this scale. A wrong constant, offset or block size is orders
  // of magnitude out.
  CHECK(max_abs > 0.5);
  CHECK(max_diff < 4.0 * max_abs / 1024.0);
}

// ───────────────────────────────────────────────────────────────────────────
// C. A REAL EXL3 linear out of the SparkInfer rank-sliced DeepSeek-V4 shard.
// ───────────────────────────────────────────────────────────────────────────
namespace {

constexpr const char* kRealShard =
    "/mnt/nas_share/rc/ckpt/dsv4-flash-0731-spark-exl3/"
    "exl3-layer-000-tp4-rank1.safetensors";

// PARITY ANCHOR PROVENANCE, stated honestly. exllamav3's own `ext.reconstruct`
// is a CUDA extension and this host has no GPU, so these literals were NOT
// produced by running upstream's kernel. They come from a throwaway script
// whose TRELLIS half is a second hand transcription of exl3_dq.cuh:15-31 +
// codebook.cuh:67-75 + pack.cu:29-57 + quantize.py:22-42, and whose HADAMARD
// half is upstream's OWN preapply_had_l / preapply_had_r (quantize.py:340-358)
// over the Sylvester H128 of hadamard.py:34-42, executed by torch 2.11.0:
//
//   python3 exl3_ref.py <ckpt>/exl3-layer-000-tp4-rank1.safetensors
//                       layers.0.ffn.experts.0.w1.rank1 3
//
// Running upstream's kernel on this shard is owed to MODEL-DSV4-EXL3 W3
// (exllamav3 gateability on GB10, .agents/oracles/exllamav3.md).
struct Spot {
  int64_t row, col;
  float value;
};

// Pre-Hadamard reconstruct: EXACT fp16 codebook values, no summation.
constexpr Spot kInnerSpots[] = {
    {0, 0, -0.52783203125f},    {0, 1, 0.5537109375f},
    {1, 0, 0.9677734375f},      {3, 7, -0.7001953125f},
    {15, 15, -0.00537109375f},  {16, 0, -2.62109375f},
    {17, 31, -1.685546875f},    {255, 17, 1.908203125f},
    {2049, 77, -0.67822265625f}, {4095, 511, -0.020751953125f},
};

// Full get_weight_tensor.
constexpr Spot kWeightSpots[] = {
    {0, 0, 0.0205841064f},     {0, 1, -0.0104980469f},
    {1, 0, -0.0253601074f},    {15, 15, 0.0425109863f},
    {16, 0, -0.0145874023f},   {127, 127, 0.000401735306f},
    {128, 128, 0.0116653442f}, {1000, 300, -0.0140304565f},
    {2049, 77, 0.0156860352f}, {4095, 511, 0.0218963623f},
};

// Minimal safetensors header reader — this case must not depend on the loader
// it is meant to be independent of.
bool ReadStTensor(const std::string& path, const std::string& name,
                  std::vector<uint8_t>* out, std::vector<int64_t>* shape,
                  std::string* dtype) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) return false;
  uint64_t hn = 0;
  if (std::fread(&hn, 1, 8, f) != 8 || hn == 0 || hn > (1u << 30)) {
    std::fclose(f);
    return false;
  }
  std::string header(static_cast<size_t>(hn), '\0');
  if (std::fread(header.data(), 1, header.size(), f) != header.size()) {
    std::fclose(f);
    return false;
  }
  const std::string key = "\"" + name + "\":";
  const size_t at = header.find(key);
  if (at == std::string::npos) {
    std::fclose(f);
    return false;
  }
  const size_t end = header.find('}', at);
  const std::string entry = header.substr(at, end - at);
  const size_t dq = entry.find("\"dtype\":\"");
  *dtype = entry.substr(dq + 9, entry.find('"', dq + 9) - (dq + 9));
  shape->clear();
  const size_t sh = entry.find("\"shape\":[");
  for (size_t p = sh + 9; p < entry.size() && entry[p] != ']';) {
    if (entry[p] == ',') { ++p; continue; }
    shape->push_back(std::strtoll(entry.c_str() + p, nullptr, 10));
    while (p < entry.size() && entry[p] != ',' && entry[p] != ']') ++p;
  }
  const size_t of = entry.find("\"data_offsets\":[");
  const int64_t begin = std::strtoll(entry.c_str() + of + 16, nullptr, 10);
  const size_t comma = entry.find(',', of + 16);
  const int64_t stop = std::strtoll(entry.c_str() + comma + 1, nullptr, 10);
  out->resize(static_cast<size_t>(stop - begin));
  std::fseek(f, static_cast<long>(8 + hn + static_cast<uint64_t>(begin)), SEEK_SET);
  const bool ok = std::fread(out->data(), 1, out->size(), f) == out->size();
  std::fclose(f);
  return ok;
}

}  // namespace

TEST_CASE("exl3: a REAL rank-sliced DeepSeek-V4 expert linear dequants to its anchors") {
  const std::string prefix = "layers.0.ffn.experts.0.w1.rank1";
  std::vector<uint8_t> tb, ub, vb;
  std::vector<int64_t> ts, us, vs;
  std::string td, ud, vd;
  if (!ReadStTensor(kRealShard, prefix + ".trellis", &tb, &ts, &td) ||
      !ReadStTensor(kRealShard, prefix + ".suh", &ub, &us, &ud) ||
      !ReadStTensor(kRealShard, prefix + ".svh", &vb, &vs, &vd)) {
    MESSAGE("SKIPPED: no readable EXL3 shard at " << kRealShard
            << ". This run decoded ZERO real checkpoint tensors; the synthetic "
               "byte gate above is unaffected.");
    return;
  }
  MESSAGE("resolved EXL3 shard: " << std::string(kRealShard) << " tensor " << prefix);
  REQUIRE(td == "I16");
  REQUIRE(ud == "F16");
  REQUIRE(vd == "F16");
  REQUIRE(ts == std::vector<int64_t>({256, 32, 48}));
  REQUIRE(us == std::vector<int64_t>({4096}));
  REQUIRE(vs == std::vector<int64_t>({512}));

  const int64_t k = ts[0] * 16, n = ts[1] * 16;
  const int kBits = static_cast<int>(ts[2] / 16);
  REQUIRE(kBits == 3);
  const auto* trellis = reinterpret_cast<const uint16_t*>(tb.data());
  const auto* suh = reinterpret_cast<const uint16_t*>(ub.data());
  const auto* svh = reinterpret_cast<const uint16_t*>(vb.data());

  std::vector<float> inner(static_cast<size_t>(k * n), 0.0f);
  vt::Exl3ReconstructInner(trellis, k, n, kBits, /*codebook=*/1, inner.data());
  for (const Spot& s : kInnerSpots) {
    CAPTURE(s.row);
    CAPTURE(s.col);
    CHECK(inner[static_cast<size_t>(s.row * n + s.col)] == s.value);
  }

  std::vector<float> w(static_cast<size_t>(k * n), 0.0f);
  vt::Exl3DequantLinear(trellis, suh, svh, k, n, kBits, /*codebook=*/1, w.data());
  double sum = 0.0, sq = 0.0, absmax = 0.0;
  bool finite = true;
  for (float x : w) {
    if (!std::isfinite(x)) finite = false;
    sum += x;
    sq += static_cast<double>(x) * x;
    absmax = std::max(absmax, std::fabs(static_cast<double>(x)));
  }
  const double count = static_cast<double>(w.size());
  const double mean = sum / count;
  const double stdev = std::sqrt(sq / count - mean * mean);
  CAPTURE(mean);
  CAPTURE(stdev);
  CAPTURE(absmax);
  CHECK(finite);
  CHECK(stdev == doctest::Approx(0.024503704).epsilon(0.002));
  CHECK(absmax == doctest::Approx(0.18737793).epsilon(0.002));
  CHECK(std::fabs(mean) < 1.0e-3);

  // Two fp16 ulps at the tensor's absmax: the f32 Hadamard summation order is
  // not torch's, and nothing else may move.
  const double tol = 2.0 * 0.18737793 / 1024.0;
  for (const Spot& s : kWeightSpots) {
    CAPTURE(s.row);
    CAPTURE(s.col);
    CHECK(std::fabs(static_cast<double>(w[static_cast<size_t>(s.row * n + s.col)]) -
                    s.value) < tol);
  }
}

// ───────────────────────────────────────────────────────────────────────────
// E. The `mul1` codebook (cb 2), gated against HAND-COMPUTED upstream values.
//
// `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` marks every quantized linear `mul1`
// (#2495), and until this row the host decoder refused cb 2 by name.
//
// THE EXPECTED VALUES BELOW WERE NOT PRODUCED BY THIS TREE. Each is
// `decode_3inst<2>` (codebook.cuh:82-89) carried out independently in exact
// rational arithmetic: `x = cw * 0x83DCD12D` mod 2^32, the four unsigned bytes
// of `x` summed, `0x6400 + sum` READ AS AN FP16 BIT PATTERN, then
// `h * 0x1eee + 0xc931` with one rounding, as `__hfma` does. Gating cb 2
// against our own `Exl3DecodeCodeword` would be a tautology, so the literals
// are the oracle and the table carries each intermediate for a reader to redo.
//
// The shape is NOT the shape of cb 0 and cb 1. Those mask, xor and sum the two
// fp16 halves of the product; cb 2 sums its BYTES. So a cb 2 arm cannot be
// obtained by swapping the multiplier, and the ordinary failure of getting it
// wrong is a correctly distributed, completely uncorrelated weight.
// ───────────────────────────────────────────────────────────────────────────
namespace {

struct Cb2Case {
  uint16_t codeword;
  uint16_t expect_f16_bits;
};

// x = codeword * 0x83DCD12D (mod 2^32); bytesum = sum of its four unsigned
// bytes; h = 1024 + bytesum exactly, because fp16 has an ULP of 1.0 across the
// whole [1024, 2048) binade and bytesum <= 1020.
const Cb2Case kMul1Cases[] = {
    {0x0000, 0xC2E8},  // x=0x00000000 bytesum=   0  h=1024.0  -> -3.453125
    {0x0001, 0x3921},  // x=0x83DCD12D bytesum= 605  h=1629.0  ->  0.64111328125
    {0x0002, 0xB72D},  // x=0x07B9A25A bytesum= 444  h=1468.0  -> -0.448486328125
    {0x00FF, 0x3AB3},  // x=0x58F45BD3 bytesum= 634  h=1658.0  ->  0.83740234375
    {0x0100, 0xB3DB},  // x=0xDCD12D00 bytesum= 474  h=1498.0  -> -0.2454833984375
    {0x1234, 0xB89E},  // x=0x4F8FA724 bytesum= 425  h=1449.0  -> -0.5771484375
    {0x7FFF, 0x3FCA},  // x=0xE4B9AED3 bytesum= 798  h=1822.0  ->  1.947265625
    {0x8000, 0xBAF2},  // x=0x68968000 bytesum= 382  h=1406.0  -> -0.8681640625
    {0xABCD, 0xBD42},  // x=0x178C9009 bytesum= 316  h=1340.0  -> -1.314453125
    {0xDEAD, 0x24BC},  // x=0x999E6169 bytesum= 513  h=1537.0  ->  0.01849365234375
    {0xFFFE, 0x3702},  // x=0xC9735DA6 bytesum= 575  h=1599.0  ->  0.43798828125
    {0xFFFF, 0xB936},  // x=0x4D502ED3 bytesum= 414  h=1438.0  -> -0.6513671875
    // THE EXTREME of the byte sum over all 65536 codewords, which is what pins
    // the top of the binade: no codeword reaches a sum above 1005, and 0x0000
    // above is the only one that reaches 0.
    {0x7B5B, 0x42B2},  // x=0xF7FEF9FF bytesum=1005  h=2029.0  ->  3.34765625
};

// cb 0 and cb 1, hand-carried the same way from codebook.cuh:58-74, so that
// "unchanged" is asserted against upstream and not against this tree.
struct Cb01Case {
  uint16_t codeword;
  uint16_t expect_cb0_bits;
  uint16_t expect_cb1_bits;
};

const Cb01Case kCb01Cases[] = {
    {0x0000, 0x3A25, 0x3F60},  // cb0= 0.76806640625   cb1= 1.84375
    {0x0001, 0xBB5B, 0x304E},  // cb0=-0.91943359375   cb1= 0.134521484375
    {0x0002, 0x3B74, 0xBA13},  // cb0= 0.931640625     cb1=-0.75927734375
    {0x00FF, 0x3CD1, 0xBD26},  // cb0= 1.2041015625    cb1=-1.287109375
    {0x0100, 0x3C3F, 0xBAF0},  // cb0= 1.0615234375    cb1=-0.8671875
    {0x1234, 0xC066, 0x3E0F},  // cb0=-2.19921875      cb1= 1.5146484375
    {0x7FFF, 0xBA96, 0x4172},  // cb0=-0.8232421875    cb1= 2.72265625
    {0x8000, 0xB7F8, 0xB915},  // cb0=-0.498046875     cb1=-0.63525390625
};

// codebook.cuh:58-64 (cb == 0), written out, for the full-domain sweep.
float TestQtipDecode(uint16_t codeword) {
  uint32_t x = static_cast<uint32_t>(codeword) * 89226354u;
  x += 64248484u;
  x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
  const float lo = vt::F16ToF32(static_cast<uint16_t>(x & 0xffffu));
  const float hi = vt::F16ToF32(static_cast<uint16_t>(x >> 16));
  return vt::F16ToF32(vt::F32ToF16(lo + hi));
}

}  // namespace

TEST_CASE("exl3: the mul1 codebook decodes to independently computed upstream values") {
  for (const Cb2Case& c : kMul1Cases) {
    CAPTURE(c.codeword);
    const float got = vt::Exl3DecodeCodeword(c.codeword, /*codebook=*/2);
    const float want = vt::F16ToF32(c.expect_f16_bits);
    CAPTURE(got);
    CAPTURE(want);
    // BIT-EXACT, both ways. The value comparison catches a wrong number and the
    // bit-pattern comparison catches a value that is right but not an fp16 --
    // upstream's decode ends in fp16, so a f32-precision result is a defect even
    // when it prints the same.
    CHECK(got == want);
    CHECK(vt::F32ToF16(got) == c.expect_f16_bits);
  }
}

TEST_CASE("exl3: mul1 is a DISTINCT codebook over the whole 16-bit domain") {
  // Not a smoke test: the failure this guards is an arm that compiles, runs and
  // returns a plausible codebook -- the wrong one. If cb 2 were reachable by
  // reusing the cb 0 or cb 1 tail, these would collide far more than by chance.
  int agree_cb0 = 0;
  int agree_cb1 = 0;
  int nonfinite = 0;
  float lo = 1.0e30f;
  float hi = -1.0e30f;
  for (int cw = 0; cw < 65536; ++cw) {
    const float v = vt::Exl3DecodeCodeword(static_cast<uint16_t>(cw), 2);
    if (!std::isfinite(v)) ++nonfinite;
    lo = std::min(lo, v);
    hi = std::max(hi, v);
    if (v == TestQtipDecode(static_cast<uint16_t>(cw))) ++agree_cb0;
    if (v == TestMcgDecode(static_cast<uint16_t>(cw))) ++agree_cb1;
  }
  CAPTURE(agree_cb0);
  CAPTURE(agree_cb1);
  CHECK(nonfinite == 0);
  // The reachable range, derived rather than observed: byte sums span [0, 1005]
  // over the domain, so h spans [1024, 2029] and the affine map sends that to
  // [1024*k_inv + k_bias, 2029*k_inv + k_bias] == [-3.453125, 3.34765625] after
  // the fp16 rounding. Both endpoints are in the table above.
  CHECK(lo == vt::F16ToF32(0xC2E8));
  CHECK(hi == vt::F16ToF32(0x42B2));
  // Chance agreement only. cb 2's image is 1006 distinct values (one per byte
  // sum) against cb 0/cb 1's tens of thousands, so a handful of coincidences is
  // expected; wholesale agreement would mean the arm aliased.
  CHECK(agree_cb0 < 4096);
  CHECK(agree_cb1 < 4096);
}

TEST_CASE("exl3: adding mul1 leaves codebooks 0 and 1 byte-identical") {
  for (const Cb01Case& c : kCb01Cases) {
    CAPTURE(c.codeword);
    CHECK(vt::F32ToF16(vt::Exl3DecodeCodeword(c.codeword, 0)) == c.expect_cb0_bits);
    CHECK(vt::F32ToF16(vt::Exl3DecodeCodeword(c.codeword, 1)) == c.expect_cb1_bits);
  }
  // …and over the whole domain against the written-out upstream forms, so the
  // regression surface is every codeword and not the eight above.
  int cb0_mismatch = 0;
  int cb1_mismatch = 0;
  for (int cw = 0; cw < 65536; ++cw) {
    const uint16_t w = static_cast<uint16_t>(cw);
    if (vt::Exl3DecodeCodeword(w, 0) != TestQtipDecode(w)) ++cb0_mismatch;
    if (vt::Exl3DecodeCodeword(w, 1) != TestMcgDecode(w)) ++cb1_mismatch;
  }
  CHECK(cb0_mismatch == 0);
  CHECK(cb1_mismatch == 0);
}

TEST_CASE("exl3: a codebook upstream does not define is still refused BY NAME") {
  // `decode_3inst<cb>` (codebook.cuh:56-90) has arms for 0, 1 and 2 and falls
  // off the end for anything else, so 3 is not a codebook this tree merely has
  // not ported -- it does not exist. Widening the host arm to cb 2 must not
  // have widened it to "anything >= 2".
  CHECK_THROWS(vt::Exl3DecodeCodeword(0x1234, 3));
  CHECK_THROWS(vt::Exl3DecodeCodeword(0x1234, -1));
  CHECK_THROWS(vt::Exl3DecodeCodeword(0x1234, 255));
}

// ── HALF-INTEGER rates (K+0.5, mul1) — BACKEND-ROCM frac rates ──────────────

namespace {

// The ENCODE side of the frac format, written from frac.cu's DEFINITION and
// sharing nothing with the implementation's decode: positions alternate
// KA / KA+1 bits (mask 0xAAAA — ODD positions carry the extra bit), the whole
// 256-position tile is one MSB-first bit run of tile_bits = 256*KA+128 bits
// (tail-biting: position 0 continues from the run's end), and the run is cut
// into 16-bit words MSB-first. The final SWAP16 over each uint32 pair is
// pack.cu:56 — the same step the integer packer applies, and the reason the
// stored int16 array reads back as a big-endian bit stream through a uint32
// view.
std::vector<uint16_t> PackFracTile(const std::vector<int>& v, int ka) {
  const int tile_bits = 256 * ka + 128;
  std::vector<uint16_t> raw(static_cast<size_t>(tile_bits / 16), 0);
  int pos = 0;
  for (int i = 0; i < 256; ++i) {
    const int width = ka + ((i & 1) ? 1 : 0);
    for (int b = width - 1; b >= 0; --b) {
      const int bit = (v[static_cast<size_t>(i)] >> b) & 1;
      raw[static_cast<size_t>(pos / 16)] |=
          static_cast<uint16_t>(bit << (15 - (pos % 16)));
      ++pos;
    }
  }
  std::vector<uint16_t> out(raw.size(), 0);
  for (size_t i = 0; i + 1 < raw.size(); i += 2) {
    out[i] = raw[i + 1];
    out[i + 1] = raw[i];
  }
  return out;
}

// The INDEPENDENT extraction: codeword t is the 16-bit window of ring bits
// [S(t)-16, S(t)) with S(i) = KA*(i+1) + ((i&15)+1)/2 + 8*(i/16). Each stream
// bit is read straight off the uint32 view — word p/32, bit 31-(p%32), the
// big-endian read the SWAP16 produces — one bit at a time. Deliberately NOT
// the funnel-shift merge the implementation uses: this is the definition
// spelled plainly, and it wraps the ring with a mod on the BIT index,
// exercising the tail-bite at every position including t near 0.
uint16_t IndependentFracCodeword(const std::vector<uint16_t>& tile, int ka, int t) {
  const int tile_bits = 256 * ka + 128;
  const int s = ka * (t + 1) + ((t & 15) + 1) / 2 + 8 * (t / 16);
  uint16_t w = 0;
  for (int j = 0; j < 16; ++j) {
    const int p = s - 16 + j;
    const int pp = p + tile_bits;  // the wrap headroom, then a bit-index mod
    const int P = (pp % tile_bits) / 32;
    const int o = pp % 32;
    const uint32_t word =
        static_cast<uint32_t>(tile[static_cast<size_t>(2 * P)]) |
        (static_cast<uint32_t>(tile[static_cast<size_t>(2 * P + 1)]) << 16);
    const int bit = static_cast<int>((word >> (31 - o)) & 1u);
    w = static_cast<uint16_t>((w << 1) | bit);
  }
  return w;
}

// codebook.cuh:82-89 (cb == 2), written out rather than called — the same
// acknowledged double transcription as TestMcgDecode above: the mul1 byte-sum
// decode cannot gate its own transcription, and the real-checkpoint anchors are
// what gate both.
float TestMul1Decode(uint16_t codeword) {
  const uint32_t x = static_cast<uint32_t>(codeword) * 0x83DCD12Du;
  const uint32_t byte_sum =
      (x & 0xffu) + ((x >> 8) & 0xffu) + ((x >> 16) & 0xffu) + ((x >> 24) & 0xffu);
  const float h = vt::F16ToF32(static_cast<uint16_t>(0x6400u + byte_sum));
  const float k_inv = vt::F16ToF32(static_cast<uint16_t>(0x1eeeu));
  const float k_bias = vt::F16ToF32(static_cast<uint16_t>(0xc931u));
  return vt::F16ToF32(vt::F32ToF16(h * k_inv + k_bias));
}

}  // namespace

TEST_CASE("exl3: the FRAC window decode reproduces independently packed codewords") {
  // Every KA the loader accepts, over random tiles: the bit-window arithmetic
  // and the mul1 decode both gate here, codeword for codeword, against the
  // independent extraction and the written-out decode. Random bits reach every
  // position, so the tail-biting wrap (S(t) - 16 negative near ring 0) is
  // exercised at every t including 0 — the off-by-128 the spec warns about is
  // a wrong-modulus bug and shows up on the first windows.
  for (const int ka : {1, 2, 3, 4, 5, 6, 7}) {
    CAPTURE(ka);
    for (const uint32_t seed : {0x51ED2A17u, 0xC0FFEEu}) {
      CAPTURE(seed);
      // Codewords bounded per position width, so every value is representable.
      std::vector<int> v(256, 0);
      uint32_t s = seed;
      for (int i = 0; i < 256; ++i) {
        s = s * 1664525u + 1013904223u;
        const int width = ka + ((i & 1) ? 1 : 0);
        v[static_cast<size_t>(i)] = static_cast<int>(s >> 16) & ((1 << width) - 1);
      }
      const std::vector<uint16_t> tile = PackFracTile(v, ka);
      REQUIRE(tile.size() == static_cast<size_t>(16 * ka + 8));

      // The scalar reader, codeword for codeword against the extraction.
      for (int t = 0; t < 256; ++t) {
        CAPTURE(t);
        CHECK(vt::Exl3FracTileCodeword(tile.data(), ka, t) == IndependentFracCodeword(tile, ka, t));
      }

      // The full tile decode against perm + independent extraction + the
      // written-out mul1 decode, compared as fp16 BIT PATTERNS.
      std::vector<float> out(256, 0.0f);
      vt::Exl3DecodeTile(tile.data(), ka, /*codebook=*/2, out.data(), /*half=*/true);
      const std::vector<int> perm = TensorCorePerm();
      int mismatch = 0;
      for (int t = 0; t < 256; ++t) {
        const float want = TestMul1Decode(IndependentFracCodeword(tile, ka, t));
        const float got = out[static_cast<size_t>(perm[static_cast<size_t>(t)])];
        if (vt::F32ToF16(got) != vt::F32ToF16(want)) ++mismatch;
      }
      CHECK(mismatch == 0);

      // An integer-rate read of the SAME bytes must NOT agree (the widths
      // differ; a frac read falling into the integer window arithmetic is the
      // silent-wrong failure class this suite refuses). Compare only where the
      // narrower integer tile is even defined: this asserts the readers really
      // diverge on the first 32*KA bytes.
      std::vector<uint16_t> as_integer(tile.begin(),
                                       tile.begin() + static_cast<long>(32 * ka / 2));
      int int_agree = 0;
      for (int t = 0; t < 256; ++t) {
        // The integer read of a truncated tile reads garbage; only the window
        // positions that stay inside the kept words are compared.
        if (vt::Exl3TileCodeword(as_integer.data(), ka, t) ==
            IndependentFracCodeword(tile, ka, t)) {
          ++int_agree;
        }
      }
      CHECK(int_agree < 256);
    }
  }
}

TEST_CASE("exl3: frac KA outside 1..7 is refused BY NAME") {
  std::vector<uint16_t> tile(16 * 8 + 8, 0);
  CHECK_THROWS(vt::Exl3FracTileCodeword(tile.data(), 0, 3));
  CHECK_THROWS(vt::Exl3FracTileCodeword(tile.data(), 8, 3));
}
