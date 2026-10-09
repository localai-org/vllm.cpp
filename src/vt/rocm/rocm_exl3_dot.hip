// EXL3 dot-product decode arm — ROCm. Row BACKEND-ROCM, spec
// .agents/specs/rocm-exl3-gemv-residual.md (R4), issue
// .agents/issues/BACKEND-ROCM/ISSUE-LOCAL-01M3RM2TM98FY569AZ6CEASBD3.md.
//
// WHY A THIRD ARM. The rocprofv3 trace on gfx1101 serving
// Qwen3.5-9B-EXL3-4.00bpw shows the ported exl3_gemv arm ALU-bound at
// ~100 GB/s: its emulated m16n8k16 MMA gathers each lane's A rows and B
// columns with 32 in-warp shuffles per 16x16 tile, and the gfx11 WMMA arm
// pays the same cost in fragment conversion (measured ~5.8x SLOWER end to
// end — VT_ROCM_EXL3_WMMA stays evaluation-only). The decoded weights are
// only ever multiplied against a handful of activation rows (m <= 8), so
// the MMA fragment machinery is overhead this hardware cannot amortize.
//
// This kernel is the OTHER shape of the same math: consume codewords where
// they land. Lane l of a warp owns the eight codewords t = 8*l..8*l+7 of one
// 16x16 tile, which RowMajorIndex places at rows (l%4)*2 + {0,1,8,9} and
// columns l/4 and l/4+8. A lane needs NO shuffle to consume what it decodes:
// it reads its codeword span straight from global, decodes with the shared
// helpers, and multiplies into two column accumulators. Two xor-shuffles
// over the four lanes sharing a column finish the tile. Accumulation is
// f32 scalar — strictly tighter than the GEMV arm's fp16 fragments — and B
// traffic is invariant in m, which is why this arm also serves mid-m
// chunking and the 6 bpw lm_head that upstream's GEMV envelope (bits 2..4
// only) cannot.
//
// The bit-window extraction is Exl3TileCodeword's own arithmetic (the
// tail-biting 16-bit window, `+ 256*bits` wrap), lifted to a per-lane
// 4-word register span so each lane touches each tile once.
#include <hip/hip_runtime.h>
#include "rocm_env.h"
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/ops.h"
#include "vt/rocm/rocm_device_bind.h"
#include "vt/rocm/rocm_exl3_decode.h"

namespace vt::rocm {
namespace {

inline void CheckD(hipError_t err, const char* what) {
  if (err != hipSuccess)
    throw std::runtime_error(std::string("vt rocm exl3 dot: ") + what + ": " +
                             hipGetErrorString(err));
}

// Lane l owns codewords 8*l..8*l+7. Exl3TileCodeword reads each codeword's
// tail-biting 16-bit window as (word32[i0]<<32 | word32[i1]) >> s0 where i0
// and i1 are VIRTUAL word indices in the +256*bits-shifted stream and only
// their residues modulo words32 map back into the tile. The per-lane span
// therefore starts at i00 = (8*l*bits + bits - 16 + 256*bits) / 32 — the
// virtual index of codeword 8*l's first word — and five u32 words cover
// i0..i1 of all eight codewords (spread 7*bits+47 bits at worst; verified
// codeword-for-codeword against Exl3TileCodeword for every width 1..8 on
// random tiles).
constexpr int kDotSpanWords = 5;

// Codeword s of a lane's span: the virtual-index arithmetic in
// Exl3TileCodeword collapses once the span base i00 is subtracted —
// codeword s's window sits at span bit e0 + s*bits (e0 = i00's bit
// residue), so its two words are w[eb>>5] and w[(eb+15)>>5] and the
// merged shift is (i1+1)*32 - b1 spelled in span terms: 48 - (eb&31). And
// (a<<32|b)>>s is ONE instruction on gfx11 — __funnelshift_r lowers to
// v_fshr/v_alignbit — so the whole extraction is a shift, two register
// selects, and the funnel shift itself. Verified codeword-for-codeword
// against Exl3TileCodeword for every width 1..8 on random tiles.
// The codeword is the LOW 16 bits of the returned word; the high half is
// window spill that the mul1 pair decode never reads, so no mask is spent.
__device__ __forceinline__ uint32_t Exl3SpanCodeword(
    const uint32_t w[kDotSpanWords], int e0, int s, int bits) {
  const int eb = e0 + s * bits;
  return __funnelshift_r(w[(eb + 15) >> 5], w[eb >> 5], 48 - (eb & 31));
}


// bits==4: e0 = (32*l + 1012) & 31 = 20 for EVERY lane — each window starts
// at compile-time span bit 20+4s, so the funnel-shift operands resolve to
// registers at compile time.
__device__ __forceinline__ uint32_t Exl3SpanCodeword4(
    const uint32_t w[kDotSpanWords], int s) {
  const int eb = 20 + s * 4;
  return __funnelshift_r(w[(eb + 15) >> 5], w[eb >> 5], 48 - (eb & 31));
}

// bits==6: e0 = (48*l + 1526) & 31 = 22 (even lane) or 6 (odd lane) — two
// compile-time residue patterns selected on lane parity. bits==5: e0 =
// (40*l + 1269) & 31 = (8*l - 11) & 31 — four patterns ({21,29,5,13} for
// l%4 = 0..3), runtime e0 but still a three-word window. The `bits`
// arithmetic is compile-time; only the window base varies.
template <int BITS>
__device__ __forceinline__ uint32_t Exl3SpanCodeword6(
    const uint32_t w[kDotSpanWords], int lane, int s) {
  const int e0 = BITS == 6 ? 22 - 16 * (lane & 1)
                           : (BITS == 5 ? (8 * lane - 11) & 31 : 0);
  const int eb = e0 + s * BITS;
  return __funnelshift_r(w[(eb + 15) >> 5], w[eb >> 5], 48 - (eb & 31));
}

// The same extraction with the two words fetched by the caller: for
// bits==5, e0 is a runtime value (four lane patterns), so indexing a
// register array w[] by eb>>5 would spill it to scratch. The caller fetches
// the tile words the window needs and passes them as w_lo = span word eb>>5
// and w_hi = span word (eb+15)>>5.
__device__ __forceinline__ uint32_t Exl3SpanCodewordDyn(
    uint32_t w_lo, uint32_t w_hi, int e0, int s, int bits) {
  const int eb = e0 + s * bits;
  return __funnelshift_r(w_hi, w_lo, 48 - (eb & 31));
}

// Coalesced tile fetch for the templated arms: lane l loads tile word l
// (bits==4: 32 words = exactly one lane per word; bits==6: 48 words, lanes
// 0..15 load a second). The lane's codeword span is then assembled from
// in-warp shuffles of the words its neighbours hold — every tile byte is
// fetched exactly once, unlike the per-lane span load which reads each
// shared word four or five times.

// Fetch tile word `word` (a runtime index into the tile) from whichever
// lane loaded it. BITS==4: word is already < 32, so one shuffle. BITS==6:
// word < 48; lanes j hold tile word j and (j<16 ? j+16 : none), so the
// fetch is two shuffles + one select. Span assembly needs at most three
// words (eb <= 87), so three of these calls per lane per tile.
// Two 6-bpw codewords decoded straight into one f16 pair: the mcg/3inst
// map is a pure multiply (+add for 3inst), the mask/xor applies bitwise to
// both lanes' u32s, and hadd2 on the gathered halves replaces the per-
// codeword DF16ToF32 + add + round. Numerically identical to the scalar
// decode because f32-add of two f16-exact halves is exact.
template <int BITS>
__device__ __forceinline__ uint32_t DotFetchWord(uint32_t w0, uint32_t w1,
                                                 int word) {
  if constexpr (BITS == 4) {
    return __shfl_sync(0xffffffffffffffffULL, w0, word & 31);
  } else {
    const uint32_t lo = __shfl_sync(0xffffffffffffffffULL, w0, word & 31);
    const uint32_t hi = __shfl_sync(0xffffffffffffffffULL, w1, word & 31);
    return word < 32 ? lo : hi;
  }
}

// Two codewords decoded straight into one f16 pair: the mcg/3inst map is a
// pure multiply (+add for 3inst), the mask/xor applies bitwise to both u32s,
// and __lows2half2/__highs2half2 + __hadd2 produce (wA, wB) — replacing two
// scalar decode chains (each fshift -> mul -> mask -> xor -> 2xDF16ToF32 ->
// fadd -> RoundHalf) with one packed round trip. Numerically identical: the
// f32 sum of two f16-exact halves is exact, so __hadd2's per-half RN is the
// same rounding the scalar arm did.
__device__ __forceinline__ half2 Exl3DecodePair6(uint32_t cwa, uint32_t cwb,
                                               int cb) {
  if (cb == 2) {
    // The shared byte-dot pair decode — two ~10-op scalar Exl3DecodeCodeword
    // calls collapse to one v_dot4_u32_u8 + __hfma2 each. Bit-identical:
    // the CPU arm proves both roundings agree over every reachable byte sum.
    return Exl3DecodePairCb2Dp4a(cwa, cwb);
  }
  uint32_t xa = cwa & 0xffffu, xb = cwb & 0xffffu;
  if (cb == 0) {
    xa = xa * 89226354u + 64248484u;
    xb = xb * 89226354u + 64248484u;
  } else {
    xa *= 0xCBAC1FEDu;
    xb *= 0xCBAC1FEDu;
  }
  xa = 0x3b603b60u ^ (xa & 0x8fff8fffu);
  xb = 0x3b603b60u ^ (xb & 0x8fff8fffu);
  const half2 ha = __builtin_bit_cast(half2, xa);
  const half2 hb = __builtin_bit_cast(half2, xb);
  return __hadd2(__lows2half2(ha, hb), __highs2half2(ha, hb));
}

// Codebook-resolved twin of Exl3DecodePair6 for the one codebook the served
// 6 bpw row uses: the cb dispatch leaves the m=1 loop body and the pair decode
// is `Exl3DecodePairCb2Dp4a` directly. Every other codebook keeps the
// runtime-cb form, so the cb 0/1 arithmetic is not duplicated here.
template <int CBS>
__device__ __forceinline__ half2 Exl3DecodePair6C(uint32_t cwa, uint32_t cwb) {
  static_assert(CBS == 2, "only the mul1 codebook has a compile-time twin");
  return Exl3DecodePairCb2Dp4a(cwa, cwb);
}

// Two adjacent fp16 activations as one u32 load. RowMajorIndex places each
// lane's k-rows at r0+{0,1} and r0+{8,9}, both adjacent pairs — so the m-loop
// reads two u32 words per (tile, m) instead of eight scalar f16 loads.
__device__ __forceinline__ uint32_t LoadH2(const half* p) {
  uint32_t v;
  __builtin_memcpy(&v, p, sizeof(v));
  return v;
}

// ─── the half-integer (K+0.5) decode — exl3_dq.cuh:254-286 `dq8_half` ────────
//
// HALF (BACKEND-ROCM frac rates). Positions alternate KA / KA+1 state bits
// (mask 0xAAAA: ODD positions carry the extra bit), so a position pair consumes
// bits2 = 2*KA+1 bits and S(t) for odd t — every group END below — is exactly
// ((t>>1)+1)*bits2. A group of four windows (positions t0..t0+3, t0 even)
// spans gspan = 16 + (KA+1) + KA + (KA+1) = 18+3*KA ≤ 27 < 32 bits, so one
// funnel shift assembles the whole span and THREE shifts derive the other
// three windows: the window ends descend by KA+1, KA, KA+1 going down from the
// top window. `+ 128*bits2` is one whole frac ring (tile_bits = 256*KA+128 =
// 128*bits2), keeping every word index non-negative — the same headroom trick
// as the integer path's +256*bits. Bit-exactness against the scalar
// Exl3FracTileCodeword reader is gated by tests/vt/test_exl3_rocm.cpp (host
// golden) and the fixture in tests/vt/test_exl3_dequant.cpp.
//
// KA is a template parameter (2 and 3 are the rates observed in the wild;
// other widths take the scalar Exl3FracTileCodeword arm). Codewords return in
// the LOW 16 bits of each element, high halves garbage — what
// Exl3DecodePairCb2Dp4a reads.
template <int KA>
__device__ __forceinline__ void Exl3Dq8Half(const uint32_t* tp, int t0,
                                            uint32_t (&cw)[8]) {
  constexpr int bits2 = 2 * KA + 1;
  constexpr int gspan = 18 + 3 * KA;
  const int W = 4 * bits2;  // u32 words per frac tile (8*KA + 4)
#pragma unroll
  for (int g = 0; g < 2; ++g) {
    const int t = t0 + 4 * g + 3;  // the group's top window (t odd)
    const int e = ((t >> 1) + 1 + 128) * bits2;  // S(t) + one ring
    const int hi = (e - 1) >> 5;
    const int lo = (e - gspan) >> 5;
    const int s = ((hi + 1) << 5) - e;
    const uint32_t wa = tp[lo % W];
    const uint32_t wb = tp[hi % W];
    // LOWER stream word in the HIGH half — the same (a<<32)|b merge the scalar
    // reader uses; swapping the operands silently inverts the stream.
    const uint64_t merged = (static_cast<uint64_t>(wa) << 32) | wb;
    const uint32_t w = static_cast<uint32_t>((merged >> s) & 0xffffffffu);
    cw[4 * g + 3] = w & 0xffffu;
    cw[4 * g + 2] = (w >> (KA + 1)) & 0xffffu;
    cw[4 * g + 1] = (w >> (KA + 1 + KA)) & 0xffffu;
    cw[4 * g + 0] = (w >> (3 * KA + 2)) & 0xffffu;
  }
}


// The m<=8 fused decode+dot kernel. Grid = 8-tile column groups times a
// host-chosen k-split; block 256 = 8 warps; each warp owns one 16-column
// tile per group and walks its k-range. Per warp per k-tile: 4 streaming
// u32 loads (the lane's codeword span), 8 window extracts + codebook
// decodes, and 8*size_m f32 FMAs. The two xor-shuffles merge the four lanes
// that share a column, lanes l%4==0 write the warp's 16 f32 partials into
// block LDS, and warp 0 flushes the group — a plain store when the k-split
// is 1, an atomicAdd when several blocks each produced a partial.
//
// A is read straight from global: a row is at most 8*size_k*2 bytes and
// stays L1/L2-resident across the n loop, so the per-tile activation reads
// are hits, not traffic.
// BITS is a template parameter: the checkpoint's 4, 5 and 6 bpw decode with
// compile-time shifts, and the generic arm (BITS==0) keeps runtime `bits`
// for every width the templated instantiations do not cover.
// Block = 256 threads (8 warps), ONE 16-column tile per block: the eight
// warps split the tile's k-range 8 ways — the k-parallelism the GEMV arm
// gets from its 16 warps is what this rebalances to, and it is what
// measured the difference between ~60 and ~200 GB/s on the lm_head. Every
// warp owns the SAME tile and a disjoint k-slice; the epilogue folds the
// eight warp partials through LDS (a plain store when the host k-split is
// 1, atomicAdd when several blocks each produced a partial).
// NW: warps per block, 8 throughout — the only instantiation (block 256).
template <int BITS, int NW = 8, int CBT = -1, bool HALF = false>
__global__ __launch_bounds__(NW * 32) void Exl3DotKImpl(
    const half* __restrict__ A, const uint16_t* __restrict__ B,
    float* __restrict__ C, int size_m, int size_k, int size_n, int bits_rt, int cb,
    int ksplit, int half_rt) {
  (void)half_rt;  // HALF instantiations fold the flag into the type
  const int bits = BITS > 0 ? BITS : bits_rt;
  const int warp = threadIdx.x >> 5;  // 0..NW-1
  const int lane = threadIdx.x & 31;
  const int ntiles = size_n >> 4;   // n/16 tile columns
  const int ktiles = size_k >> 4;   // k/16 tile rows
  const int words32 = bits * 8;     // u32 words per 16x16 tile
  const int tile = blockIdx.x % ntiles;
  const int kslice = blockIdx.x / ntiles;

  // This warp's k-tile range: host k-split slices sit inside the 8-way
  // warp split — kslice*8+warp picks this warp's slice of
  // ceil(ktiles / (ksplit*8)) tiles.
  const int chunk = (ktiles + ksplit * NW - 1) / (ksplit * NW);
  const int kt0 = (kslice * NW + warp) * chunk;
  const int kt1 = min(ktiles, kt0 + chunk);

  // per-lane codeword span (tail-biting, modulo words32)
  // the virtual word index where this lane's span starts (see the helper
  // above): residues modulo words32 select tile words.
  const int i00 = (8 * lane * bits + bits - 16 + 256 * bits) >> 5;
  const int e0 = (8 * lane * bits + bits - 16 + 256 * bits) & 31;
  const uint32_t* B32 = reinterpret_cast<const uint32_t*>(B);
  __shared__ float sh[NW][8][16];  // [warp][m][col within tile]
  // bits==6 only: the warp's tile words staged through LDS. Cross-lane
  // fetch via __shfl costs ~30 cycles of dependent latency per word and
  // dominates the m=1 chain; ds_read is ~4 cycles and banks are conflict-
  // free here because lanes' 5-word spans overlap but never share a bank.
  [[maybe_unused]] __shared__ uint32_t sh_tile[NW][96];  // 48 u32 × 2 tiles

  {
    float acc[8][2] = {};  // acc[m][col-half]: c = l/4 (+8)
    const int r0 = (lane & 3) * 2;

    // tile = blockIdx.x % ntiles over a grid of ntiles*ksplit blocks, so
    // every block owns a valid tile; no guard is needed here.
    {
      if constexpr (HALF) {
        // The half-integer arm (K+0.5, mul1 — validated at the launch site).
        // Same lane/column mapping and same epilogue as the integer arms; only
        // the codeword source differs. KA 2 and 3 run the dq8_half fast port
        // (Exl3Dq8Half); every other KA takes the width-generic scalar reader
        // — the correctness floor, not a tuned path.
        const int Wf = 8 * bits + 4;  // u32 words per frac tile
        for (int kt = kt0; kt < kt1; ++kt) {
          half2 wv2[4];
          if constexpr (BITS == 2 || BITS == 3) {
            uint32_t cw[8];
            Exl3Dq8Half<BITS>(B32 + (static_cast<size_t>(kt) * ntiles + tile) * Wf,
                              8 * lane, cw);
#pragma unroll
            for (int q = 0; q < 4; ++q)
              wv2[q] = Exl3DecodePairCb2Dp4a(cw[2 * q], cw[2 * q + 1]);
          } else {
            // Generic KA: one scalar window read per codeword, identical
            // arithmetic to the host reference Exl3FracTileCodeword. The tile
            // stride is 16*KA+8 uint16 words.
            const uint16_t* tp =
                B + (static_cast<size_t>(kt) * ntiles + tile) * (16 * bits + 8);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              const uint16_t cwa = Exl3FracTileCodeword(tp, bits, 8 * lane + 2 * q);
              const uint16_t cwb = Exl3FracTileCodeword(tp, bits, 8 * lane + 2 * q + 1);
              wv2[q] = Exl3DecodePairCb2Dp4a(cwa, cwb);
            }
          }

          const int kbase = kt * 16 + r0;
#pragma unroll
          for (int mm = 0; mm < 8; ++mm) {
            if (mm >= size_m) break;
            const half* arow = A + (size_t)mm * size_k + kbase;
            const half2 a01 = __builtin_bit_cast(half2, LoadH2(arow));
            const half2 a89 = __builtin_bit_cast(half2, LoadH2(arow + 8));
            acc[mm][0] = amd_mixed_dot(wv2[0], a01, acc[mm][0], false);
            acc[mm][0] = amd_mixed_dot(wv2[1], a89, acc[mm][0], false);
            acc[mm][1] = amd_mixed_dot(wv2[2], a01, acc[mm][1], false);
            acc[mm][1] = amd_mixed_dot(wv2[3], a89, acc[mm][1], false);
          }
        }
      } else if constexpr (BITS == 4 || BITS == 5 || BITS == 6) {
        // Coalesced fetch: the warp reads the tile's words32 u32s exactly
        // once — lane l holds word l, plus word 32+l for bits==6 — and the
        // lane's codeword span is assembled by shuffling its neighbours'
        // words back. This replaces the per-lane span load, which re-read
        // every shared word four or five times.
        uint32_t ring[4][2] = {};
        uint32_t aring[4][2] = {};  // A pairs (a01, a89) for the same slot
        auto ld_tile_b = [&](int kt, uint32_t& o0, uint32_t& o1) {
          const uint32_t* tp =
              B32 + ((size_t)kt * ntiles + tile) * words32;
          o0 = tp[lane];
          if constexpr (BITS > 4)
            o1 = (lane < BITS * 8 - 32) ? tp[lane + 32] : 0u;
        };
        auto ld_tile = [&](int kt, uint32_t& o0, uint32_t& o1,
                           uint32_t& a0, uint32_t& a1) {
          ld_tile_b(kt, o0, o1);
          if (size_m == 1 && BITS > 4) {
            a0 = LoadH2(A + kt * 16 + r0);
            a1 = LoadH2(A + kt * 16 + r0 + 8);
          }
        };
#pragma unroll
        for (int d = 0; d < 4; ++d)
          if (kt0 + d < kt1)
            ld_tile(kt0 + d, ring[d][0], ring[d][1], aring[d][0], aring[d][1]);

        if (size_m == 1 && BITS > 4) {
          // m=1 (the lm_head): decode is a per-lane serial chain — 8 codewords
          // per tile through funnel-shift + Exl3DecodeCodeword — so one tile
          // per loop iteration leaves the warp issue-idle on ALU latency.
          // Two tiles per iteration give the scheduler two independent
          // chains; the ring already covers their loads.
          // Roving pointers: consecutive kt of one tile column sit ntiles
          // tiles apart (a full k-ROW — 2.98 MB at the lm_head), so the
          // per-load (size_t)kt * ntiles * words32 mad chain collapses to a
          // pointer add. Same addresses, same words, same math.
          const size_t krow = (size_t)ntiles * words32;
          const uint32_t* tp0 = B32 + ((size_t)kt0 * ntiles + tile) * words32;
          const half* ap0 = A + (size_t)kt0 * 16 + r0;
          const uint32_t* tpf = tp0 + 4 * krow;
          const half* apf = ap0 + 4 * 16;
          auto ld_tile_r = [&](const uint32_t* tp, const half* ap, uint32_t& o0,
                               uint32_t& o1, uint32_t& a0, uint32_t& a1) {
            o0 = tp[lane];
            o1 = (lane < BITS * 8 - 32) ? tp[lane + 32] : 0u;
            a0 = LoadH2(ap);
            a1 = LoadH2(ap + 8);
          };
          auto dec6 = [&](uint32_t a, uint32_t b) -> half2 {
            if constexpr (CBT >= 0)
              return Exl3DecodePair6C<CBT>(a, b);
            else
              return Exl3DecodePair6(a, b, cb);
          };
          int kt = kt0;
          for (; kt + 1 < kt1; kt += 2) {
            uint32_t c0a = ring[0][0], c1a = ring[0][1];
            uint32_t c0b = ring[1][0], c1b = ring[1][1];
#pragma unroll
            for (int d = 0; d < 2; ++d) {
              ring[d][0] = ring[d + 2][0];
              ring[d][1] = ring[d + 2][1];
            }
            uint32_t a0a = aring[0][0], a1a = aring[0][1];
            uint32_t a0b = aring[1][0], a1b = aring[1][1];
#pragma unroll
            for (int d = 0; d < 2; ++d) {
              aring[d][0] = aring[d + 2][0];
              aring[d][1] = aring[d + 2][1];
            }
            if (kt + 4 < kt1)
              ld_tile_r(tpf, apf, ring[2][0], ring[2][1], aring[2][0], aring[2][1]);
            if (kt + 5 < kt1)
              ld_tile_r(tpf + krow, apf + 16, ring[3][0], ring[3][1], aring[3][0],
                        aring[3][1]);
            tpf += 2 * krow;
            apf += 32;

            sh_tile[warp][lane] = c0a;
            if (lane < BITS * 8 - 32) sh_tile[warp][32 + lane] = c1a;
            sh_tile[warp][48 + lane] = c0b;
            if (lane < BITS * 8 - 32) sh_tile[warp][48 + 32 + lane] = c1b;
            __syncwarp();
            const int i0v = i00 % words32;
            // Exl3SpanCodeword6's widest index is (eb+15)>>5 = 2 for s<=7 —
            // three words carry the whole 16-bit codeword span.
            [[maybe_unused]] uint32_t wa[3], wb[3];
            if constexpr (BITS != 5) {
#pragma unroll
              for (int u = 0; u < 3; ++u) {
                wa[u] = sh_tile[warp][(i0v + u) % words32];
                wb[u] = sh_tile[warp][48 + (i0v + u) % words32];
              }
            }
            __syncwarp();

            half2 wv2a[4], wv2b[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              if constexpr (BITS == 5) {
                // Runtime e0: fetch each codeword's two window words
                // straight from LDS — indexing a register array by eb>>5
                // would spill it, and ds_read is ~4 cycles bank-conflict
                // free here.
                const int ea = e0 + 2 * q * 5;
                const int eaa = e0 + (2 * q + 1) * 5;
                wv2a[q] = dec6(
                    Exl3SpanCodewordDyn(
                        sh_tile[warp][(i0v + (ea >> 5)) % 40],
                        sh_tile[warp][(i0v + ((ea + 15) >> 5)) % 40],
                        e0, 2 * q, 5),
                    Exl3SpanCodewordDyn(
                        sh_tile[warp][(i0v + (eaa >> 5)) % 40],
                        sh_tile[warp][(i0v + ((eaa + 15) >> 5)) % 40],
                        e0, 2 * q + 1, 5));
                wv2b[q] = dec6(
                    Exl3SpanCodewordDyn(
                        sh_tile[warp][48 + (i0v + (ea >> 5)) % 40],
                        sh_tile[warp][48 + (i0v + ((ea + 15) >> 5)) % 40],
                        e0, 2 * q, 5),
                    Exl3SpanCodewordDyn(
                        sh_tile[warp][48 + (i0v + (eaa >> 5)) % 40],
                        sh_tile[warp][48 + (i0v + ((eaa + 15) >> 5)) % 40],
                        e0, 2 * q + 1, 5));
              } else {
                wv2a[q] = dec6(Exl3SpanCodeword6<BITS>(wa, lane, 2 * q),
                               Exl3SpanCodeword6<BITS>(wa, lane, 2 * q + 1));
                wv2b[q] = dec6(Exl3SpanCodeword6<BITS>(wb, lane, 2 * q),
                               Exl3SpanCodeword6<BITS>(wb, lane, 2 * q + 1));
              }
            }
            // bits==5 reads sh_tile inside the decode loop above, past the
            // last syncwarp — one more barrier before the next iteration's
            // staging writes can begin.
            if constexpr (BITS == 5) __syncwarp();

            const half2 a01a = __builtin_bit_cast(half2, a0a);
            const half2 a89a = __builtin_bit_cast(half2, a1a);
            const half2 a01b = __builtin_bit_cast(half2, a0b);
            const half2 a89b = __builtin_bit_cast(half2, a1b);
            acc[0][0] = amd_mixed_dot(wv2a[0], a01a, acc[0][0], false);
            acc[0][0] = amd_mixed_dot(wv2a[1], a89a, acc[0][0], false);
            acc[0][1] = amd_mixed_dot(wv2a[2], a01a, acc[0][1], false);
            acc[0][1] = amd_mixed_dot(wv2a[3], a89a, acc[0][1], false);
            acc[0][0] = amd_mixed_dot(wv2b[0], a01b, acc[0][0], false);
            acc[0][0] = amd_mixed_dot(wv2b[1], a89b, acc[0][0], false);
            acc[0][1] = amd_mixed_dot(wv2b[2], a01b, acc[0][1], false);
            acc[0][1] = amd_mixed_dot(wv2b[3], a89b, acc[0][1], false);
          }
          for (; kt < kt1; ++kt) {
            uint32_t c0 = ring[0][0], c1 = ring[0][1];
            uint32_t a0t = aring[0][0], a1t = aring[0][1];
#pragma unroll
            for (int d = 0; d < 3; ++d) {
              ring[d][0] = ring[d + 1][0];
              ring[d][1] = ring[d + 1][1];
              aring[d][0] = aring[d + 1][0];
              aring[d][1] = aring[d + 1][1];
            }
            aring[3][0] = a0t; aring[3][1] = a1t;  // placeholder, load below
            if (kt + 4 < kt1)
              ld_tile_r(tpf, apf, ring[3][0], ring[3][1], aring[3][0], aring[3][1]);
            tpf += krow;
            apf += 16;
            sh_tile[warp][lane] = c0;
            if (lane < BITS * 8 - 32) sh_tile[warp][32 + lane] = c1;
            __syncwarp();
            const int i0v = i00 % words32;
            [[maybe_unused]] uint32_t w[3];
            if constexpr (BITS != 5) {
#pragma unroll
              for (int u = 0; u < 3; ++u)
                w[u] = sh_tile[warp][(i0v + u) % words32];
            }
            __syncwarp();
            half2 wv2[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              if constexpr (BITS == 5) {
                const int ea = e0 + 2 * q * 5;
                const int eaa = e0 + (2 * q + 1) * 5;
                wv2[q] = dec6(
                    Exl3SpanCodewordDyn(
                        sh_tile[warp][(i0v + (ea >> 5)) % 40],
                        sh_tile[warp][(i0v + ((ea + 15) >> 5)) % 40],
                        e0, 2 * q, 5),
                    Exl3SpanCodewordDyn(
                        sh_tile[warp][(i0v + (eaa >> 5)) % 40],
                        sh_tile[warp][(i0v + ((eaa + 15) >> 5)) % 40],
                        e0, 2 * q + 1, 5));
              } else {
                wv2[q] = dec6(Exl3SpanCodeword6<BITS>(w, lane, 2 * q),
                              Exl3SpanCodeword6<BITS>(w, lane, 2 * q + 1));
              }
            }
            if constexpr (BITS == 5)
              __syncwarp();  // sh_tile read above, written next iter
            const half2 a01 = __builtin_bit_cast(half2, a0t);
            const half2 a89 = __builtin_bit_cast(half2, a1t);
            acc[0][0] = amd_mixed_dot(wv2[0], a01, acc[0][0], false);
            acc[0][0] = amd_mixed_dot(wv2[1], a89, acc[0][0], false);
            acc[0][1] = amd_mixed_dot(wv2[2], a01, acc[0][1], false);
            acc[0][1] = amd_mixed_dot(wv2[3], a89, acc[0][1], false);
          }
        } else {
          for (int kt = kt0; kt < kt1; ++kt) {
            // rotate: current = oldest ring slot, prefetch four tiles ahead
            uint32_t c0 = ring[0][0], c1 = ring[0][1];
#pragma unroll
            for (int d = 0; d < 3; ++d) {
              ring[d][0] = ring[d + 1][0];
              ring[d][1] = ring[d + 1][1];
            }
            if (kt + 4 < kt1) ld_tile_b(kt + 4, ring[3][0], ring[3][1]);

            // Assemble the lane's three-word span from the warp's words.
            const int i0v = i00 % words32;  // span word 0's tile index
            [[maybe_unused]] uint32_t w[3];
            if constexpr (BITS != 5) {
              w[0] = DotFetchWord<BITS>(c0, c1, i0v);
              w[1] = DotFetchWord<BITS>(c0, c1, (i0v + 1) % words32);
              w[2] = DotFetchWord<BITS>(c0, c1, (i0v + 2) % words32);
            }

            float wv[8];
#pragma unroll
            for (int s = 0; s < 8; ++s) {
              uint16_t cw;
              if constexpr (BITS == 4) {
                cw = Exl3SpanCodeword4(w, s);
              } else if constexpr (BITS == 5) {
                // Runtime e0: fetch the codeword's two window words via
                // shuffles rather than indexing a register array — a dynamic
                // w[] index would spill the array to scratch.
                const int eb = e0 + s * 5;
                const uint32_t wl =
                    DotFetchWord<BITS>(c0, c1, (i0v + (eb >> 5)) % words32);
                const uint32_t wh = DotFetchWord<BITS>(
                    c0, c1, (i0v + ((eb + 15) >> 5)) % words32);
                cw = Exl3SpanCodewordDyn(wl, wh, e0, s, 5);
              } else {
                cw = Exl3SpanCodeword6<BITS>(w, lane, s);
              }
              wv[s] = Exl3DecodeCodeword(cw, cb);
            }
            // Decoded weights are f16-exact (Exl3DecodeCodeword returns an
            // fp16-rounded value), so the dot pairs run as PACKED f16 through
            // amd_mixed_dot — v_dot2_f32_f16, which is a VOPD dual-issue
            // candidate (V_DUAL_DOT2ACC_F32_F16): two pair-dots per issue
            // slot, and half the FMA count of the scalar form.
            half2 wv2[4];
#pragma unroll
            for (int q = 0; q < 4; ++q)
              wv2[q] = __floats2half2_rn(wv[2 * q], wv[2 * q + 1]);

            const int kbase = kt * 16 + r0;
            // r0+{0,1} and r0+{8,9} are adjacent fp16 pairs — one u32 load
            // per pair, reinterpreted as half2 for the dot.
#pragma unroll
            for (int mm = 0; mm < 8; ++mm) {
              if (mm >= size_m) break;
              const half* arow = A + (size_t)mm * size_k + kbase;
              const half2 a01 = __builtin_bit_cast(half2, LoadH2(arow));
              const half2 a89 = __builtin_bit_cast(half2, LoadH2(arow + 8));
              // s<4 -> rows r0+q (q=s), col c0; s>=4 -> same rows, col c0+8.
              acc[mm][0] = amd_mixed_dot(wv2[0], a01, acc[mm][0], false);
              acc[mm][0] = amd_mixed_dot(wv2[1], a89, acc[mm][0], false);
              acc[mm][1] = amd_mixed_dot(wv2[2], a01, acc[mm][1], false);
              acc[mm][1] = amd_mixed_dot(wv2[3], a89, acc[mm][1], false);
            }
          }
        }
      } else {
        // Generic arm (runtime bits): the per-lane span load.
        uint32_t pf0[kDotSpanWords], pf1[kDotSpanWords];
        auto ld_span = [&](int kt, uint32_t* w) {
          const uint32_t* tp =
              B32 + ((size_t)kt * ntiles + tile) * words32;
#pragma unroll
          for (int j = 0; j < kDotSpanWords; ++j)
            w[j] = tp[(i00 + j) % words32];
        };
        if (kt0 < kt1) ld_span(kt0, pf0);
        if (kt0 + 1 < kt1) ld_span(kt0 + 1, pf1);

        for (int kt = kt0; kt < kt1; ++kt) {
          uint32_t w[kDotSpanWords];
#pragma unroll
          for (int j = 0; j < kDotSpanWords; ++j) w[j] = pf0[j];
#pragma unroll
          for (int j = 0; j < kDotSpanWords; ++j) pf0[j] = pf1[j];
          if (kt + 2 < kt1) ld_span(kt + 2, pf1);

          float wv[8];
#pragma unroll
          for (int s = 0; s < 8; ++s)
            wv[s] = Exl3DecodeCodeword(Exl3SpanCodeword(w, e0, s, bits), cb);

          const int kbase = kt * 16 + r0;
          half2 wv2[4];
#pragma unroll
          for (int q = 0; q < 4; ++q)
            wv2[q] = __floats2half2_rn(wv[2 * q], wv[2 * q + 1]);
#pragma unroll
          for (int mm = 0; mm < 8; ++mm) {
            if (mm >= size_m) break;
            const half* arow = A + (size_t)mm * size_k + kbase;
            const half2 a01 = __builtin_bit_cast(half2, LoadH2(arow));
            const half2 a89 = __builtin_bit_cast(half2, LoadH2(arow + 8));
            acc[mm][0] = amd_mixed_dot(wv2[0], a01, acc[mm][0], false);
            acc[mm][0] = amd_mixed_dot(wv2[1], a89, acc[mm][0], false);
            acc[mm][1] = amd_mixed_dot(wv2[2], a01, acc[mm][1], false);
            acc[mm][1] = amd_mixed_dot(wv2[3], a89, acc[mm][1], false);
          }
        }
      }
    }

    // Reduce the four lanes sharing each column (l, l^1, l^2). Empty
    // k-ranges still write zeros: ksplit>1 slices accumulate through
    // atomicAdd, so a stale LDS value would corrupt the result.
    for (int mm = 0; mm < size_m; ++mm) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float v = acc[mm][h];
        v += __shfl_xor_sync(0xffffffffffffffffULL, v, 1);
        v += __shfl_xor_sync(0xffffffffffffffffULL, v, 2);
        if ((lane & 3) == 0) sh[warp][mm][(lane >> 2) + 8 * h] = v;
      }
    }
    __syncthreads();

    // Fold the warps' partials for this one tile: every (m, col) sums
    // sh[w][m][c] over w — the warp-split k-slices. Warp 0 owns the fold and
    // the writeout; m*16 cells / 32 lanes is 4 passes at m=8.
    if (warp == 0) {
      const int col0 = tile * 16;
      for (int i = lane; i < size_m * 16; i += 32) {
        const int mm = i >> 4;
        const int c = i & 15;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; ++w) sum += sh[w][mm][c];
        const size_t off = (size_t)mm * size_n + col0 + c;
        if (ksplit > 1)
          atomicAdd(C + off, sum);
        else
          C[off] = sum;
      }
    }
  }
}

}  // namespace

// The dot arm's launch contract. Serves every shape the GEMV arm's envelope
// declines — all bit widths 1..8 (upstream's GEMV caps at 4), all three
// codebooks — plus m<=8 shapes the host wants on the lighter ALU profile.
// Returns false on the preconditions the op layer does not already
// guarantee; a false is a decline, never a failure.
bool Exl3DotTryLaunchRocm(hipStream_t st, const Tensor& a_had, const Tensor& trellis,
                          float* raw, int size_m, int size_k, int size_n, int device,
                          const Exl3GemmArgs& args) {
  if (size_m < 1 || size_m > 8) return false;
  if (size_k % 128 != 0 || size_n % 128 != 0) return false;
  if (args.bits < 1 || args.bits > 8) return false;
  if (args.codebook < 0 || args.codebook > 2) return false;
  // HALF-INTEGER rate (K+0.5): mul1 only, and the KAs the oracle bounds to
  // [1, 7]. Refused by name rather than dropped into an integer arm — the
  // arm set below is compile-time, so a frac tensor MUST take a HALF
  // instantiation or none at all.
  if (args.half && args.codebook != 2) return false;
  if (args.half && (args.bits < 1 || args.bits > 7)) return false;

  int warp_size = 0;
  CheckD(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, device),
         "warp size query");
  if (warp_size != 32) return false;  // the lane/col mapping is wave32

  const int ntiles = size_n / 16;
  // One block per 16-column tile (each block's eight warps split k 8 ways),
  // ksplit block-level slices for deeper parallelism on narrow shapes. The
  // atomic tail needs a zeroed C, which is why raw scratch zeroing happens
  // here and not in the kernel.
  static const int kKsplitEnv = EnvIntStrict("VT_EXL3_DOT_KSPLIT", 0);
  int ksplit = 1;
  while (ntiles * ksplit < 128 && (size_k / 16) / (ksplit * 2) >= 8) ksplit <<= 1;
  if (kKsplitEnv > 0) {
    // Tuning probe, not a contract — still capped: an unbounded split makes
    // ntiles*ksplit grid.x and the in-kernel kslice*NW bounds overflow int.
    const int cap = std::max(1, static_cast<int>((size_k / 16) / 8));
    ksplit = std::min(kKsplitEnv, std::min(cap, 1 << 20));
  }
  if (ksplit > 1)
    CheckD(hipMemsetAsync(raw, 0, (size_t)size_m * size_n * sizeof(float), st),
           "exl3_dot zero");

  const half* a_ptr = a_had.Ptr<half>();
  const uint16_t* b_ptr = trellis.Ptr<uint16_t>();
  int bits = args.bits, cb = args.codebook;
  int half_flag = args.half ? 1 : 0;
  void* kargs[] = {static_cast<void*>(&a_ptr),  static_cast<void*>(&b_ptr),
                   static_cast<void*>(&raw),    static_cast<void*>(&size_m),
                   static_cast<void*>(&size_k), static_cast<void*>(&size_n),
                   static_cast<void*>(&bits),   static_cast<void*>(&cb),
                   static_cast<void*>(&ksplit), static_cast<void*>(&half_flag)};
  // The templated instantiations fold bits into the type: the checkpoint's
  // 4 and 6 bpw decode with compile-time shifts; every other width launches
  // Exl3DotK<0>, the runtime-bits arm. The HALF rows (K+0.5, mul1) compile
  // the dq8_half fast port for KA 2 and 3 — the rates observed in the wild —
  // and the width-generic scalar reader for the rest.
  const void* kernel;
  unsigned block_sz = 256;
  if (args.half) {
    if (bits == 2) {
      kernel = reinterpret_cast<const void*>(&Exl3DotKImpl<2, 8, -1, true>);
    } else if (bits == 3) {
      kernel = reinterpret_cast<const void*>(&Exl3DotKImpl<3, 8, -1, true>);
    } else {
      kernel = reinterpret_cast<const void*>(&Exl3DotKImpl<0, 8, -1, true>);
    }
  } else if (bits == 4) {
    kernel = reinterpret_cast<const void*>(&Exl3DotKImpl<4, 8>);
  } else if (bits == 5 && cb == 2) {
    kernel = reinterpret_cast<const void*>(&Exl3DotKImpl<5, 8, 2>);
  } else if (bits == 5) {
    kernel = reinterpret_cast<const void*>(&Exl3DotKImpl<5, 8>);
  } else if (bits == 6 && cb == 2) {
    kernel = reinterpret_cast<const void*>(&Exl3DotKImpl<6, 8, 2>);
  } else if (bits == 6) {
    kernel = reinterpret_cast<const void*>(&Exl3DotKImpl<6, 8>);
  } else {
    kernel = reinterpret_cast<const void*>(&Exl3DotKImpl<0, 8>);
  }
  CheckD(hipLaunchKernel(kernel,
                         dim3(static_cast<unsigned>(ntiles * ksplit)),
                         dim3(block_sz), kargs, 0, st),
         "exl3_dot launch");
  CheckD(hipGetLastError(), "exl3_dot launch");
  return true;
}

// The chunked variant for 8 < m <= 32 (spec R2a, dot flavour): the same
// 8-row slicing the GEMV chunker uses. Each chunk re-reads B — upstream's
// own trade-off for the narrow arm — and past 32 rows the reconstruct arm
// is the right shape.
bool Exl3DotChunkedTryLaunchRocm(hipStream_t st, const Tensor& a_had,
                                 const Tensor& trellis, float* raw, int size_m,
                                 int size_k, int size_n, int device,
                                 const Exl3GemmArgs& args) {
  for (int base = 0; base < size_m; base += 8) {
    const int rows = std::min(8, size_m - base);
    Tensor a_slice = a_had;
    a_slice.data = static_cast<char*>(a_had.data) +
                   static_cast<size_t>(base) * static_cast<size_t>(size_k) * sizeof(half);
    float* raw_slice = raw + static_cast<size_t>(base) * static_cast<size_t>(size_n);
    if (!Exl3DotTryLaunchRocm(st, a_slice, trellis, raw_slice, rows, size_k, size_n,
                              device, args))
      return false;
  }
  return true;
}

}  // namespace vt::rocm
