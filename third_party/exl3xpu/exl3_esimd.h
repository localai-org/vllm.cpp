#pragma once
// EXL3 (exllamav3 trellis) kernels for Intel Xe2 (BMG / Arc B70) in ESIMD.
//
// Trellis tile (16x16 weights, K bits each) = 256*K bits = 8*K uint32 words, MSB-first stream.
// Value t (0..255) is decoded from the 16-bit window ending at bit (t+1)*K (circular in the tile),
// and lands at tile row/col:
//     row = 8*((t>>1)&1) + 2*((t>>3)&3) + (t&1)       (K / input dim)
//     col = (t>>5) + 8*((t>>2)&1)                       (N / output dim)
//
// Bit periodicity: with g = gcd(K, 32), every D = K/g words hold V = 32/g whole values, so value
// t = V*grp + u always sits at the same bit offsets relative to word D*grp. We vectorise over grp
// (G = 256/V lanes) and fully unroll u. For V >= 8 the output column half h = (u>>2)&1 depends on
// u only and the base column t>>5 depends on grp only, so accumulators are acc[h][grp] and the
// final reduction sums groups of 32/V adjacent lanes.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>

namespace exl3 {

using namespace sycl::ext::intel::esimd;
using fp16 = sycl::half;

constexpr int cgcd(int a, int b) { return b == 0 ? a : cgcd(b, a % b); }

template <int K> struct Geo {
    static constexpr int g = cgcd(K, 32);
    static constexpr int D = K / g;          // words per period
    static constexpr int V = 32 / g;         // values per period
    static constexpr int G = 256 / V;        // periods per tile (vector lanes)
    static constexpr int WORDS = 8 * K;      // words per tile
    static_assert(V >= 8, "K with gcd(K,32) > 4 (K=8) not supported by this kernel");
};

// ------------------------------------------------------------------------------------------------
// Codebooks: 16-bit state -> fp16 value (returned widened to float), bit-exact with exllamav3.

template <int CB, int N>
ESIMD_INLINE simd<fp16, N> decode_cb_h(simd<uint32_t, N> st) {
#ifdef EXL3_DEBUG_NODECODE
    return convert<fp16>(st);
#endif
    // Bit-exact with exllamav3 decode_3inst<cb> (CUDA): same integer ops, same fp16 roundings.
    if constexpr (CB == 2) {  // mul1
#ifdef EXL3_MUL16
        // st < 2^16: st * 0x83DCD12D mod 2^32 == st*0xD12D + ((st*0x83DC) mod 2^16) << 16, i.e. two
        // full-rate 16x16 multiplies instead of one reduced-rate 32x32 multiply (same bits).
        simd<uint16_t, N> s16 = convert<uint16_t>(st);
        simd<uint32_t, N> lo = s16 * simd<uint16_t, N>(0xD12D);
        simd<uint16_t, N> hi = s16 * simd<uint16_t, N>(0x83DC);
        simd<uint32_t, N> x = lo + (convert<uint32_t>(hi) << 16);
#else
        simd<uint32_t, N> x = st * 0x83DCD12Du;
#endif
        // 0x6400 + byte sum in one instruction; its low 16 bits ARE fp16(1024 + byte sum) (< 2048)
        simd<uint32_t, N> bs = dp4a<uint32_t, uint32_t, uint32_t, uint32_t, N>(
            simd<uint32_t, N>(0x6400u), x, simd<uint32_t, N>(0x01010101u));
#ifdef EXL3_DECODE_INLINE
        // strided fp16 view straight into the mad (<2;1,0>:hf is a legal source region; no copy)
        return bs.template bit_cast_view<fp16>().template select<N, 2>(0) * fp16(0.00676727294921875f)
               + fp16(-10.3828125f);
#else
        simd<fp16, N> h = bs.template bit_cast_view<fp16>().template select<N, 2>(0);
        // __hfma(h, 0x1eee, 0xc931): single fp16 rounding, bit-exact with exllamav3
        return h * fp16(0.00676727294921875f) + fp16(-10.3828125f);
#endif
    } else {
        simd<uint32_t, N> x;
        if constexpr (CB == 1) x = st * 0xCBAC1FEDu;
        else x = st * 89226354u + 64248484u;
        x = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
        simd<uint16_t, N> lo = convert<uint16_t>(x & 0xFFFFu);
        simd<uint16_t, N> hi = convert<uint16_t>(x >> 16);
        simd<fp16, N> a = lo.template bit_cast_view<fp16>().read();
        simd<fp16, N> b = hi.template bit_cast_view<fp16>().read();
        return a + b;   // __hadd
    }
}

// mul1 codebook without its affine: h = fp16(1024 + bytesum), exact. The affine w = h*c1 + c2 is applied to the
// dot product instead (sum_k a_k w_k = c1 * sum_k a_k h_k + c2 * sum_k a_k), saving one fp16 mad per value.
constexpr float kMul1C1 = 0.00676727294921875f, kMul1C2 = -10.3828125f;
template <int N>
ESIMD_INLINE simd<fp16, N> decode_mul1_raw(simd<uint32_t, N> st) {
    simd<uint32_t, N> x = st * 0x83DCD12Du;
    simd<uint32_t, N> bs = dp4a<uint32_t, uint32_t, uint32_t, uint32_t, N>(
        simd<uint32_t, N>(0x6400u), x, simd<uint32_t, N>(0x01010101u));
    return bs.template bit_cast_view<fp16>().template select<N, 2>(0);
}

template <int CB, int N>
ESIMD_INLINE simd<float, N> decode_cb(simd<uint32_t, N> st) {
    return convert<float>(decode_cb_h<CB, N>(st));
}

#ifdef EXL3_LUT
// Codebook lookup: lut[state] holds the fp16 value decode_cb_h<CB> computes for that state (built by
// LutKernel from the same function, so bit-exact). Replaces mul + dp4a + mad per weight with one L1-cached gather.
template <int N>
ESIMD_INLINE simd<fp16, N> decode_lut(const fp16* lut, simd<uint32_t, N> st) {
    return gather<fp16, N>(lut, st << 1);
}

template <int CB>
struct LutKernel {
    fp16* lut;
    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        uint32_t base = it.get_global_id(0) * 16;
        simd<uint32_t, 16> st(base, 1);
        block_store<fp16, 16>(lut + base, decode_cb_h<CB, 16>(st));
    }
};
#endif

// state for value u of every period of TP adjacent tiles: words dw [TP*WORDS], prev[x] = dw[x - 1]
// (circular per tile). Tiles are contiguous and WORDS is a multiple of D, so a stride-D select runs
// straight across tile boundaries: TP=2 turns K=6's 16-lane vectors into 32-lane ones.
template <int K, int u, int TP = 1, bool PL = false>
ESIMD_INLINE simd<uint32_t, Geo<K>::G * TP> tile_states(simd<uint32_t, Geo<K>::WORDS * TP> dw,
                                                         simd<uint32_t, Geo<K>::WORDS * TP> prev) {
    using Gm = Geo<K>;
    constexpr int D = Gm::D, G = Gm::G * TP;
    // PL: dw/prev are planar (plane i = words D*g + i over g, contiguous), so the D>1 stride-D gathers
    // become contiguous reads (Xe regions only stride by powers of two; stride 3 costs per-element moves)
    constexpr int SS = PL ? 1 : D;      // select stride
    constexpr int PS = PL ? G : 1;      // plane step
    constexpr int e = (u + 1) * K;                       // end bit within period
    constexpr int i1 = (e - 1) / 32;                     // word holding last bit
    constexpr int b0 = e - 16;                           // start bit (may be negative)
    constexpr int i0 = b0 >= 0 ? b0 / 32 : -1;           // word holding first bit
    constexpr int s = (i1 + 1) * 32 - e;                 // right shift aligning window end
    simd<uint32_t, G> B = dw.template select<G, SS>(i1 * PS);
#ifdef EXL3_DEBUG_NOSTATE
    return B;
#endif
    simd<uint32_t, G> st;
    if constexpr (i0 == i1) {
        st = B >> s;
    } else {
        simd<uint32_t, G> A;
        if constexpr (i0 >= 0) A = dw.template select<G, SS>(i0 * PS);
        else A = prev.template select<G, SS>(0);         // previous period's last word
        if constexpr (s == 0) st = B;
        else st = (A << (32 - s)) | (B >> s);
    }
    return st & 0xFFFFu;
}

// Load NT tiles' words and their circular previous-word vectors: prev is words shifted one dword in
// registers with lane 0 of each tile patched (EXL3_PREV_FROM_MEM restores the old second load at p-1).
template <int NT, int W>
ESIMD_INLINE void load_words(const uint32_t* p, bool first, simd<uint32_t, NT * W>& words,
                             simd<uint32_t, NT * W>& prev) {
    words = block_load<uint32_t, NT * W>(p);
#ifndef EXL3_PREV_FROM_MEM
    // prev is words shifted by one dword; every tile's prev[0] is patched below, so the second (p-1) load
    // was redundant. Measured on B70, all linears: M=1 32.5->29.2 ms, M=16 39.0->37.8, M=64 76.8->67.7.
    first = true;
#endif
    if (!first) {
        prev = block_load<uint32_t, NT * W>(p - 1);
    } else {
        prev.template select<NT * W - 1, 1>(1) = words.template select<NT * W - 1, 1>(0);
    }
#pragma unroll
    for (int j = 0; j < NT; ++j) prev[j * W] = words[j * W + W - 1];
}

#ifdef EXL3_L1_PF
#ifndef EXL3_PF_DIST
#define EXL3_PF_DIST 2
#endif
// Cache-only prefetch (no registers) of the trellis words EXL3_PF_DIST K-rows ahead, in 64-dword blocks.
template <int N>
ESIMD_INLINE void pf_words(const uint32_t* p) {
    static_assert(N % 32 == 0 || N < 32, "prefetch granularity");
    if constexpr (N % 64 != 0) {
#pragma unroll
        for (int c = 0; c < (N + 31) / 32; ++c)
            sycl::ext::intel::experimental::esimd::lsc_prefetch<uint32_t, 32, sycl::ext::intel::experimental::esimd::lsc_data_size::default_size,
                sycl::ext::intel::experimental::esimd::cache_hint::cached, sycl::ext::intel::experimental::esimd::cache_hint::cached>(p + 32 * c);
    } else if constexpr (N < 64) {
        sycl::ext::intel::experimental::esimd::lsc_prefetch<uint32_t, N, sycl::ext::intel::experimental::esimd::lsc_data_size::default_size,
            sycl::ext::intel::experimental::esimd::cache_hint::cached, sycl::ext::intel::experimental::esimd::cache_hint::cached>(p);
    } else {
#pragma unroll
        for (int c = 0; c < N / 64; ++c)
            sycl::ext::intel::experimental::esimd::lsc_prefetch<uint32_t, 64, sycl::ext::intel::experimental::esimd::lsc_data_size::default_size,
                sycl::ext::intel::experimental::esimd::cache_hint::cached, sycl::ext::intel::experimental::esimd::cache_hint::cached>(p + 64 * c);
    }
}
#endif

// Reorder each group of D*GV words from interleaved (word D*g + i) to planar (plane i, lane g).
template <int NGRP, int D, int GV>
ESIMD_INLINE void planarize(simd<uint32_t, NGRP * D * GV>& v) {
    simd<uint32_t, NGRP * D * GV> t = v;
#pragma unroll
    for (int jp = 0; jp < NGRP; ++jp)
#pragma unroll
        for (int i = 0; i < D; ++i)
            v.template select<GV, 1>(jp * D * GV + i * GV) = t.template select<GV, D>(jp * D * GV + i);
}

// ------------------------------------------------------------------------------------------------
// Fast Walsh-Hadamard (Sylvester order) of 128 floats, unnormalised

ESIMD_INLINE void fwht128(simd<float, 128>& v) {
#define EXL3_BFLY(H)                                                                     \
    {                                                                                    \
        auto m = v.template bit_cast_view<float, 128 / (2 * H), 2 * H>();               \
        simd<float, 64> a = m.template select<128 / (2 * H), 1, H, 1>(0, 0).read();     \
        simd<float, 64> b = m.template select<128 / (2 * H), 1, H, 1>(0, H).read();     \
        m.template select<128 / (2 * H), 1, H, 1>(0, 0) = a + b;                         \
        m.template select<128 / (2 * H), 1, H, 1>(0, H) = a - b;                         \
    }
    EXL3_BFLY(1) EXL3_BFLY(2) EXL3_BFLY(4) EXL3_BFLY(8) EXL3_BFLY(16) EXL3_BFLY(32) EXL3_BFLY(64)
#undef EXL3_BFLY
}

constexpr float kRsqrt128 = 0.08838834764831845f;

// ------------------------------------------------------------------------------------------------
// Fusion helpers (single-kernel linear for small M):
//  - input Hadamard computed per thread for its own 128-row k block, bit-identical to HadInKernel
//  - split-K reduction + output Hadamard by the last-arriving thread of each 128-column block

struct FusedArgs {
    const fp16* x; int x_stride;       // [M, Kdim] row-major fp16 activations
    const fp16* suh;                   // [S, Kdim]
    const fp16* svh;                   // [N]
    fp16* out; int out_stride;         // [M, N] fp16
    uint32_t* counters;                // [N/128], zero on entry, left zero on exit
};

// xblk[m*128 + i] = fp16( H(fp16(x[m, kb*128 + .] * suh[shard, .])) / sqrt(128) )[i]
template <int MROWS>
ESIMD_INLINE void fused_had_in(const FusedArgs& fa, int M, int Kdim, int shard, int kb, simd<fp16, MROWS * 128>& xblk) {
    simd<float, 128> su = convert<float>(block_load<fp16, 128>(fa.suh + (size_t)shard * Kdim + kb * 128));
#pragma unroll
    for (int m = 0; m < MROWS; ++m) {
        if (m < M) {
            simd<float, 128> v = convert<float>(block_load<fp16, 128>(fa.x + (size_t)m * fa.x_stride + kb * 128)) * su;
            v = convert<float>(convert<fp16>(v));
            fwht128(v);
            v *= kRsqrt128;
            xblk.template select<128, 1>(m * 128) = convert<fp16>(v);
        } else {
            xblk.template select<128, 1>(m * 128) = 0;
        }
    }
}

// Called by every thread after writing its partials for columns [col0, col0 + ncols) of split p.
// The thread that completes a 128-column block (arrivals == P * 128/ncols) reduces it.
ESIMD_INLINE void fused_had_out(const FusedArgs& fa, const float* part, int M, int N, int P, int col0, int ncols) {
    fence<memory_kind::global, fence_flush_op::none, fence_scope::gpu>();
    int nb = col0 / 128;
    uint32_t target = (uint32_t)(P * (128 / ncols));
    simd<uint32_t, 1> off(nb * (uint32_t)sizeof(uint32_t));
    simd<uint32_t, 1> old = atomic_update<atomic_op::inc, uint32_t, 1>(fa.counters, off);
    if (old[0] != target - 1) return;
    fence<memory_kind::global, fence_flush_op::none, fence_scope::gpu>();
    atomic_update<atomic_op::store, uint32_t, 1>(fa.counters, off, simd<uint32_t, 1>(0u));
    simd<float, 128> sv = convert<float>(block_load<fp16, 128>(fa.svh + nb * 128)) * kRsqrt128;
    for (int m = 0; m < M; ++m) {
        simd<float, 128> v = 0.0f;
        for (int pp = 0; pp < P; ++pp) {
            // L1 is not coherent across Xe cores: read other threads' partials past it (LSC max 64 x d32)
            const float* src = part + ((size_t)pp * M + m) * N + nb * 128;
            constexpr auto props = properties{cache_hint_L1<cache_hint::uncached>, cache_hint_L2<cache_hint::cached>};
            v.template select<64, 1>(0) += block_load<float, 64>(src, props);
            v.template select<64, 1>(64) += block_load<float, 64>(src + 64, props);
        }
        fwht128(v);
        block_store<fp16, 128>(fa.out + (size_t)m * fa.out_stride + nb * 128, convert<fp16>(v * sv));
    }
}

// ------------------------------------------------------------------------------------------------
// had_in: xh[g, m, :] = fp16( H(x[m, :] * suh[g, :]) / sqrt(128) ), one thread per (g, m, 128-block)

template <typename TIn>
struct HadInKernel {
    const TIn* x; const fp16* suh; fp16* xh;
    int M, Kdim, S, x_stride, Mp;   // Mp: padded row count of the blocked xh layout
    int row_major;                  // 1: plain [S][M][Kdim] output (prefill GEMM operand)
    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int id = it.get_global_id(0);
        int kb_n = Kdim / 128;
        int kb = id % kb_n;
        int m = (id / kb_n) % M;
        int g = id / (kb_n * M);
        if (g >= S) return;
        simd<TIn, 128> xi = block_load<TIn, 128>(x + (size_t)m * x_stride + kb * 128);
        simd<fp16, 128> su = block_load<fp16, 128>(suh + (size_t)g * Kdim + kb * 128);
        simd<float, 128> v = convert<float>(xi) * convert<float>(su);
        // CUDA path rounds x*suh to fp16 before the transform
        v = convert<float>(convert<fp16>(v));
        fwht128(v);
        v *= kRsqrt128;
        simd<fp16, 128> vh = convert<fp16>(v);
        if (row_major) {
            block_store<fp16, 128>(xh + ((size_t)g * M + m) * Kdim + kb * 128, vh);
            return;
        }
        // blocked layout xh[g][k/16][m][16]: a tile-row's A block for all M rows is contiguous
        int kt = Kdim / 16;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            block_store<fp16, 16>(xh + (((size_t)g * kt + kb * 8 + i) * Mp + m) * 16, vh.template select<16, 1>(i * 16));
    }
};

// ------------------------------------------------------------------------------------------------
// had_out: out[m, nb*128:+128] = svh * H(sum_p part[p, m, :]) / sqrt(128)

template <typename TOut, typename TPart = float>
struct HadOutKernel {
    const TPart* part; const fp16* svh; TOut* out;
    int M, N, P, out_stride;
    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int id = it.get_global_id(0);
        int nb_n = N / 128;
        int nb = id % nb_n;
        int m = id / nb_n;
        if (m >= M) return;
        simd<float, 128> v = 0.0f;
        for (int p = 0; p < P; ++p)
            v += convert<float>(block_load<TPart, 128>(part + ((size_t)p * M + m) * N + nb * 128));
        fwht128(v);
        simd<fp16, 128> sv = block_load<fp16, 128>(svh + nb * 128);
        v = v * kRsqrt128 * convert<float>(sv);
        block_store<TOut, 128>(out + (size_t)m * out_stride + nb * 128, convert<TOut>(v));
    }
};

// ------------------------------------------------------------------------------------------------
// Prefill W8A8: had_in to int8 with one scale per (group, row). Thread = (g, m), two passes over the row
// (max, then quantize); the pre-quantization values are exactly the fp16 path's xh.

template <typename TIn>
struct HadInQ8Kernel {
    const TIn* x; const fp16* suh; int8_t* xq; float* sx;
    int M, Kdim, S, x_stride, Ms;   // Ms: row stride of xq / sx (M padded for the GEMM)
    ESIMD_INLINE simd<float, 128> blk(int g, int m, int kb) const {
        simd<TIn, 128> xi = block_load<TIn, 128>(x + (size_t)m * x_stride + kb * 128);
        simd<fp16, 128> su = block_load<fp16, 128>(suh + (size_t)g * Kdim + kb * 128);
        simd<float, 128> v = convert<float>(xi) * convert<float>(su);
        v = convert<float>(convert<fp16>(v));
        fwht128(v);
        v *= kRsqrt128;
        return convert<float>(convert<fp16>(v));
    }
    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int id = it.get_global_id(0);
        int m = id % M, g = id / M;
        if (g >= S) return;
        int kb_n = Kdim / 128;
        simd<float, 128> mx = 0.0f;
        for (int kb = 0; kb < kb_n; ++kb)
            mx = max(mx, abs(blk(g, m, kb)));
        float amax = hmax<float>(mx);
        float sc = amax > 0.0f ? amax / 127.0f : 1.0f;
        float inv = 1.0f / sc;
        sx[(size_t)g * Ms + m] = sc;
        for (int kb = 0; kb < kb_n; ++kb)
            block_store<int8_t, 128>(xq + ((size_t)g * Ms + m) * Kdim + kb * 128,
                                     convert<int8_t>(rnde<float>(blk(g, m, kb) * inv)));
    }
};

// Same result as HadInQ8Kernel, one work-group of TPR threads per (g, m): each thread transforms up to NB
// 128-blocks into registers, the row max is reduced through SLM, then each thread quantizes its own blocks.
template <typename TIn, int TPR, int NB>
struct HadInQ8WgKernel {
    const TIn* x; const fp16* suh; int8_t* xq; float* sx;
    int M, Kdim, S, x_stride, Ms;
    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        slm_init<TPR * sizeof(float)>();
        int t = it.get_local_id(0);
        int row = it.get_group(0);
        int m = row % M, g = row / M;
        int kb_n = Kdim / 128;
        simd<fp16, 128 * NB> buf;
        simd<float, 128> mx = 0.0f;
#pragma unroll
        for (int j = 0; j < NB; ++j) {
            int kb = t + j * TPR;
            if (kb < kb_n) {
                simd<TIn, 128> xi = block_load<TIn, 128>(x + (size_t)m * x_stride + kb * 128);
                simd<fp16, 128> su = block_load<fp16, 128>(suh + (size_t)g * Kdim + kb * 128);
                simd<float, 128> v = convert<float>(xi) * convert<float>(su);
                v = convert<float>(convert<fp16>(v));
                fwht128(v);
                v *= kRsqrt128;
                simd<fp16, 128> vh = convert<fp16>(v);
                buf.template select<128, 1>(j * 128) = vh;
                mx = max(mx, abs(convert<float>(vh)));
            }
        }
        slm_scalar_store<float>(t * sizeof(float), hmax<float>(mx));
        barrier();
        float amax = hmax<float>(slm_block_load<float, TPR>(0));
        float sc = amax > 0.0f ? amax / 127.0f : 1.0f;
        float inv = 1.0f / sc;
        if (t == 0) sx[(size_t)g * Ms + m] = sc;
#pragma unroll
        for (int j = 0; j < NB; ++j) {
            int kb = t + j * TPR;
            if (kb < kb_n) {
                simd<fp16, 128> b = buf.template select<128, 1>(j * 128);
                simd<float, 128> r = rnde<float>(convert<float>(b) * inv);
                block_store<int8_t, 128>(xq + ((size_t)g * Ms + m) * Kdim + kb * 128, convert<int8_t>(r));
            }
        }
    }
};

// out[m, nb*128:+128] = svh * H(y_i32[m, :] * sx[m] * sw) / sqrt(128), for one fused group's columns
template <typename TOut>
struct HadOutQ8Kernel {
    const int32_t* y; const float* sx; const fp16* svh; TOut* out;
    int M, N, out_stride; float sw;
    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int id = it.get_global_id(0);
        int nb_n = N / 128;
        int nb = id % nb_n, m = id / nb_n;
        if (m >= M) return;
        simd<float, 128> v = convert<float>(block_load<int32_t, 128>(y + (size_t)m * N + nb * 128)) * (sx[m] * sw);
        fwht128(v);
        v = v * kRsqrt128 * convert<float>(block_load<fp16, 128>(svh + nb * 128));
        block_store<TOut, 128>(out + (size_t)m * out_stride + nb * 128, convert<TOut>(v));
    }
};

// ------------------------------------------------------------------------------------------------
// Fused trellis-decode GEMV/GEMM for small M.
//   part[p, m, n] = sum_{k in split p} xh[shard(n), m, k] * W_inner[k, n]
// Thread = (column strip of NT tiles, K split). Loops over its tile-rows.

template <int K, int CB, int MR, int NT, bool FUSED = false>
struct GemvKernel {
    const fp16* xh;          // [S, M, Kdim]  (unused when FUSED)
    const uint32_t* tr;      // [Kdim/16, N/16, 8K]
    const int* shard_of_nb;  // [N/128]
    float* part;             // [P, M, N]
    int M, Kdim, N, tiles_n, rows_per_split, n_strips;
    FusedArgs fa;
    int Psplit;     // number of K splits (fused split-K reduction target)
    const fp16* lut = nullptr;   // EXL3_LUT: 65536-entry codebook table

    static constexpr int G = Geo<K>::G;
    static constexpr int V = Geo<K>::V;
    static constexpr int W = Geo<K>::WORDS;
    static constexpr int D = Geo<K>::D;
#ifdef EXL3_NO_PLANAR
    static constexpr bool PLANAR = false;
#else
    static constexpr bool PLANAR = D > 1;
#endif
    static constexpr int TP = G >= 32 ? 1 : 32 / G;     // tiles decoded per vector op (K=6: 2)
    static constexpr int GV = G * TP;                   // vector lanes
    static constexpr int NP = NT / TP;                  // tile groups per thread
    static_assert(NT % TP == 0, "NT must be a multiple of the tile pairing");
    // accumulator layout: acc[((m * NP + jp) * 2 + h) * GV + lane]
    static constexpr int ACC = MR * NP * 2 * GV;

    template <int u>
    static ESIMD_INLINE void step(simd<uint32_t, NT * W>& words, simd<uint32_t, NT * W>& prev,
                                  simd<fp16, MR * 16>& xr, simd<fp16, ACC>& hacc, const fp16* lut) {
        if constexpr (u < V) {
            constexpr int h = (u >> 2) & 1;
            // row(grp, u) = R0 + STR * (grp % P): a strided replicate of the 16-row x block
            constexpr int P = V == 8 ? 4 : (V == 16 ? 2 : 1);
            constexpr int STR = V == 8 ? 2 : (V == 16 ? 4 : 1);
            constexpr int R0 = 8 * ((u >> 1) & 1) + 2 * ((u >> 3) & 3) + (u & 1);
#pragma unroll
            for (int jp = 0; jp < NP; ++jp) {
                simd<uint32_t, W * TP> dw = words.template select<W * TP, 1>(jp * W * TP);
                simd<uint32_t, W * TP> pw = prev.template select<W * TP, 1>(jp * W * TP);
#ifdef EXL3_LUT
                simd<fp16, GV> v = decode_lut<GV>(lut, tile_states<K, u, TP, PLANAR>(dw, pw));
#else
                simd<fp16, GV> v = decode_cb_h<CB, GV>(tile_states<K, u, TP, PLANAR>(dw, pw));
#endif
#pragma unroll
                for (int m = 0; m < MR; ++m) {
                    auto a = hacc.template select<GV, 1>(((m * NP + jp) * 2 + h) * GV);
                    a += v * xr.template replicate_vs_w_hs<GV / P, 0, P, STR>(m * 16 + R0);
                }
            }
            step<u + 1>(words, prev, xr, hacc, lut);
        }
    }

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int id = it.get_global_id(0);
        int strip = id % n_strips;
        int p = id / n_strips;
        if (p * rows_per_split >= Kdim / 16) return;
        int tile_n0 = strip * NT;
        int shard = shard_of_nb[(tile_n0 * 16) / 128];
        int r0 = p * rows_per_split;
        int r1 = r0 + rows_per_split;
        if (r1 > Kdim / 16) r1 = Kdim / 16;

        simd<float, ACC> acc = 0.0f;
        if constexpr (FUSED) {
            // r0, r1 are multiples of 8 tile-rows: one input-Hadamard block per 8 tile-rows
            for (int rb = r0; rb < r1; rb += 8) {
                simd<fp16, MR * 128> xblk;
                fused_had_in<MR>(fa, M, Kdim, shard, rb / 8, xblk);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int r = rb + i;
                    size_t t0 = (size_t)r * tiles_n + tile_n0;
                    simd<uint32_t, NT * W> words, prev;
                    load_words<NT, W>(tr + t0 * W, t0 == 0, words, prev);
                    if constexpr (PLANAR) { planarize<NP, D, GV>(words); planarize<NP, D, GV>(prev); }
                    simd<fp16, MR * 16> xr;
#pragma unroll
                    for (int m = 0; m < MR; ++m)
                        xr.template select<16, 1>(m * 16) = xblk.template select<16, 1>(m * 128 + i * 16);
                    simd<fp16, ACC> hacc = 0;
                    step<0>(words, prev, xr, hacc, lut);
                    acc += convert<float>(hacc);
                }
            }
        } else {
        const fp16* xbase = xh + (size_t)shard * M * Kdim;   // blocked [k/16][M][16]

        for (int r = r0; r < r1; ++r) {
            size_t t0 = (size_t)r * tiles_n + tile_n0;
            simd<uint32_t, NT * W> words, prev;
#ifdef EXL3_L1_PF
            if (r + EXL3_PF_DIST < r1) pf_words<NT * W>(tr + (t0 + (size_t)EXL3_PF_DIST * tiles_n) * W);
#endif
            load_words<NT, W>(tr + t0 * W, t0 == 0, words, prev);
            if constexpr (PLANAR) { planarize<NP, D, GV>(words); planarize<NP, D, GV>(prev); }
            simd<fp16, MR * 16> xr = 0;
            if (M == MR) {
                xr = block_load<fp16, MR * 16>(xbase + (size_t)r * M * 16);
            } else {
#pragma unroll
                for (int m = 0; m < MR; ++m)
                    if (m < M)
                        xr.template select<16, 1>(m * 16) = block_load<fp16, 16>(xbase + ((size_t)r * M + m) * 16);
            }
            // fp16 partial dot products over one tile-row (few terms per lane), folded into fp32
            simd<fp16, ACC> hacc = 0;
            step<0>(words, prev, xr, hacc, lut);
            acc += convert<float>(hacc);
        }
        }

        // reduce groups of L = 32/V adjacent lanes -> 8 columns per half, per tile
        constexpr int L = 32 / V;
#pragma unroll
        for (int m = 0; m < MR; ++m) {
            if (m >= M) break;
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                simd<float, 16> o;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    int base = ((m * NP + j / TP) * 2 + h) * GV + (j % TP) * G;
                    simd<float, 8> s = acc.template select<8, L>(base);
#pragma unroll
                    for (int l = 1; l < L; ++l) s += acc.template select<8, L>(base + l);
                    o.template select<8, 1>(h * 8) = s;
                }
                block_store<float, 16>(part + ((size_t)p * M + m) * N + (tile_n0 + j) * 16, o);
            }
        }
        if constexpr (FUSED) fused_had_out(fa, part, M, N, Psplit, tile_n0 * 16, NT * 16);
    }
};

// ------------------------------------------------------------------------------------------------
// DPAS (XMX) trellis GEMM for batched decode: one EXL3 16x16 tile == one DPAS B operand
// (K16 x N16, VNNI: element (k, n) at ((k>>1)*16 + n)*2 + (k&1)). Decoding each tile once feeds
// MB/8 DPAS ops, so decode cost is amortised over the whole batch block.
//
// Writing decoded lanes straight into VNNI: with grp = P*a + b (P = 32/V... see below) the value
// (grp, u) lands at VNNI matrix [8][32] row = R(u) + MS*b, col = 16*h(u) + (u&1) + 2*a, a in 0..7.
// Reading the trellis words in (b-major, a-minor) lane order (a transpose folded into the word
// read) lets each b-row be written with one strided region move.

template <int K, int CB, int MB, int NT, bool FUSED = false>
struct DpasKernel {
    const fp16* xh;          // [S, M, Kdim]
    const uint32_t* tr;      // [Kdim/16, N/16, 8K]
    const int* shard_of_nb;  // [N/128]
    float* part;             // [P, M, N]
    int M, Kdim, N, tiles_n, rows_per_split, n_strips, m_blocks, Mp;   // xh rows padded to Mp = m_blocks * MB
    FusedArgs fa;
    int Psplit;     // number of K splits (fused split-K reduction target)

    static constexpr int G = Geo<K>::G, V = Geo<K>::V, D = Geo<K>::D, W = Geo<K>::WORDS;
    static constexpr int P = V == 8 ? 4 : (V == 16 ? 2 : 1);   // grp period in rows
    static constexpr int A_ = G / P;                            // == 8
    static constexpr int MS = 4 / P;                            // VNNI-row stride per b
    static_assert(A_ == 8, "lane geometry");

#ifdef EXL3_NO_PLANAR
    static constexpr bool PLANAR = false;
#else
    static constexpr bool PLANAR = D > 1;    // K=6: de-interleave words so the B-operand gathers use stride P
#endif
    // lane L = 8*b + a reads word D*(P*a + b) + i: interleaved <P, D, 8, D*P>, planar <P, 1, 8, P> at plane i
#ifndef EXL3_FOLD   // opt-in: +3.5% at M<=8 but an unexplained gemm_raw gate anomaly at M=24/40 (see PROGRESS)
    static constexpr bool FOLD = false;
#else
    // codebook affine folded out of the decode loop where decode is the bottleneck (small row blocks)
    static constexpr bool FOLD = (CB == 2) && (MB <= 8) && !FUSED;   // MB=16: extra accumulator spills (35 -> 70 ms)
#endif
    template <int N>
    static ESIMD_INLINE simd<fp16, N> dec(simd<uint32_t, N> st) {
        if constexpr (FOLD) return decode_mul1_raw<N>(st);
        else return decode_cb_h<CB, N>(st);
    }
    static constexpr int RVS = PLANAR ? 1 : D;
    static constexpr int RHS = PLANAR ? P : D * P;
    static constexpr int RPS = PLANAR ? G : 1;

    template <int u, class BT>
    static ESIMD_INLINE void build(simd<uint32_t, W>& dw, simd<uint32_t, W>& dwprev, BT&& Bv) {
        if constexpr (u < V) {
            constexpr int e = (u + 1) * K;
            constexpr int i1 = (e - 1) / 32;
            constexpr int b0 = e - 16;
            constexpr int i0 = b0 >= 0 ? b0 / 32 : -1;
            constexpr int s = (i1 + 1) * 32 - e;
            // transpose-read: lane L = 8*b + a  <->  word D*(P*a + b) + i
            simd<uint32_t, G> Bw = dw.template replicate_vs_w_hs<P, RVS, 8, RHS>(i1 * RPS);
            simd<uint32_t, G> st;
            if constexpr (i0 == i1) {
                st = Bw >> s;
            } else {
                simd<uint32_t, G> Aw;
                if constexpr (i0 >= 0) Aw = dw.template replicate_vs_w_hs<P, RVS, 8, RHS>(i0 * RPS);
                else Aw = dwprev.template replicate_vs_w_hs<P, RVS, 8, RHS>(0);
                if constexpr (s == 0) st = Bw;
                else st = (Aw << (32 - s)) | (Bw >> s);
            }
            simd<fp16, G> v = dec<G>(st & 0xFFFFu);
            constexpr int h = (u >> 2) & 1;
            constexpr int R = 4 * ((u >> 1) & 1) + ((u >> 3) & 3);      // row0(u) / 2
            constexpr int C0 = 16 * h + (u & 1);
            auto Bm = Bv.template bit_cast_view<fp16, 8, 32>();
#pragma unroll
            for (int b = 0; b < P; ++b)
                Bm.template select<1, 1, 8, 2>(R + MS * b, C0) = v.template select<8, 1>(8 * b);
            build<u + 1>(dw, dwprev, Bv);
        }
    }

    // K=4 fast path: build the VNNI B operand with contiguous 32-lane writes and no scatter moves.
    // Word w (0..31) holds t = 8w + j; with w = m + 4c' and j = q0 + 2q1 + 4h the VNNI index of t is
    // 128q1 + 32m + 2c' + 16h + q0. Lanes L = q0 + 2c' + 16h for fixed (q1, m) are therefore one
    // contiguous 32-half chunk at 32m + 128q1. Each lane reads word m + 4c' (a region broadcast) and
    // shifts by a per-lane amount: state = (prev << 4 + 4j) | (cur >> 28 - 4j), low 16 bits.
#ifndef EXL3_HALVES_MAXMB
#define EXL3_HALVES_MAXMB 32
#endif
    template <class BT>
    static ESIMD_INLINE void build_k4(simd<uint32_t, W>& dw, simd<uint32_t, W>& prev, BT&& Bv) {
#ifdef EXL3_K4_HALVES   // REJECTED: wrong DPAS B layout at MB<=32 (caught by extended Gate A1)
      if constexpr (MB <= EXL3_HALVES_MAXMB) {
        // Same values, as two 16-lane halves (h = 0, 1) that read the word region <4;2,0> directly instead of
        // materialising a duplicated 32-lane copy of it: lane l = q0 + 2c' within a half, j = q0 + 2q1 + 4h.
        // All linears: M=4 30.6 -> 29.4 ms, M=8 32.3 -> 31.0, M=16 34.7 -> 33.1-33.9; MB=64 regresses (+4%).
        simd<uint32_t, 16> l16(0, 1);
        simd<uint32_t, 16> q0h = l16 & 1u;
        simd<uint32_t, W> prev2 = prev << 2u;
#pragma unroll
        for (int q1 = 0; q1 < 2; ++q1) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                simd<uint32_t, 16> j = q0h + (2u * q1 + 4u * h);
                simd<uint32_t, 16> sr = 28u - 4u * j;
                simd<uint32_t, 16> sl = 2u + 4u * j;
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    simd<uint32_t, 16> st = ((prev2.template replicate_vs_w_hs<8, 4, 2, 0>(m) << sl)
                                             | (dw.template replicate_vs_w_hs<8, 4, 2, 0>(m) >> sr)) & 0xFFFFu;
                    Bv.template select<16, 1>(32 * m + 128 * q1 + 16 * h) = dec<16>(st);
                }
            }
        }
      } else
#endif
      {
        simd<uint32_t, 32> lane(0, 1);
        simd<uint32_t, 32> q0 = lane & 1u, h = lane >> 4;
        simd<uint32_t, W> prev2 = prev << 2u;                // hoisted: one shift per tile, not per group
#pragma unroll
        for (int q1 = 0; q1 < 2; ++q1) {
            simd<uint32_t, 32> j = q0 + 2u * q1 + 4u * h;
            simd<uint32_t, 32> sr = 28u - 4u * j;          // right shift of the current word (0..28)
            simd<uint32_t, 32> sl = 2u + 4u * j;           // prev << 2 << sl  ==  prev << (4 + 4j), 32 -> 0
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                simd<uint32_t, 32> cw, pw;
                cw.template select<16, 1>(0) = dw.template replicate_vs_w_hs<8, 4, 2, 0>(m);
                cw.template select<16, 1>(16) = cw.template select<16, 1>(0);
                pw.template select<16, 1>(0) = prev2.template replicate_vs_w_hs<8, 4, 2, 0>(m);
                pw.template select<16, 1>(16) = pw.template select<16, 1>(0);
                simd<uint32_t, 32> st = ((pw << sl) | (cw >> sr)) & 0xFFFFu;
                Bv.template select<32, 1>(32 * m + 128 * q1) = dec<32>(st);
            }
        }
      }
    }

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int id = it.get_global_id(0);
        int strip = id % n_strips;
        int rest = id / n_strips;
        int mb = rest % m_blocks;
        int p = rest / m_blocks;
        if (p * rows_per_split >= Kdim / 16) return;
        int tile_n0 = strip * NT;
        int shard = shard_of_nb[(tile_n0 * 16) / 128];
        int r0 = p * rows_per_split;
        int r1 = r0 + rows_per_split;
        if (r1 > Kdim / 16) r1 = Kdim / 16;
        int m0 = mb * MB;
        int mrows = M - m0; if (mrows > MB) mrows = MB;

        simd<float, MB * 16 * NT> acc = 0.0f;   // [NT][MB][16]
        simd<float, FOLD ? MB * 16 : 16> accs = 0.0f;   // FOLD: activation row sums (all 16 columns equal)
        if constexpr (FUSED) {
            static_assert(MB == 8, "fused DPAS path is for one 8-row block");
            for (int rb = r0; rb < r1; rb += 8) {
                simd<fp16, MB * 128> xblk;
                fused_had_in<MB>(fa, M, Kdim, shard, rb / 8, xblk);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int r = rb + i;
                    size_t t0 = (size_t)r * tiles_n + tile_n0;
                    simd<uint32_t, NT * W> words, prevs;
                    load_words<NT, W>(tr + t0 * W, t0 == 0, words, prevs);
                    if constexpr (PLANAR) { planarize<NT, D, G>(words); planarize<NT, D, G>(prevs); }
                    simd<fp16, MB * 16> Am;
#pragma unroll
                    for (int m = 0; m < MB; ++m)
                        Am.template select<16, 1>(m * 16) = xblk.template select<16, 1>(m * 128 + i * 16);
            #pragma unroll
                        for (int j = 0; j < NT; ++j) {
                            simd<uint32_t, W> dw = words.template select<W, 1>(j * W);
                            simd<uint32_t, W> dwprev = prevs.template select<W, 1>(j * W);
                            simd<fp16, 256> Bv;
                            if constexpr (K == 4) build_k4(dw, dwprev, Bv);
                            else build<0>(dw, dwprev, Bv);
                            auto c = acc.template select<128, 1>(j * MB * 16);
                            c = xmx::dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(c), Bv, Am);
                        }
                }
            }
        } else {
        const fp16* xbase = xh + (size_t)shard * Mp * Kdim;   // blocked [k/16][Mp][16], pad rows zero

#ifdef EXL3_DPAS_PREFETCH
        // REJECTED (register pressure: M=16 39 -> 86 ms); kept for reference.
        // software pipeline: the next K-row's trellis words (the DRAM stream) load while this row computes
        simd<uint32_t, NT * W> nwords, nprevs;
        {
            size_t t0 = (size_t)r0 * tiles_n + tile_n0;
            load_words<NT, W>(tr + t0 * W, t0 == 0, nwords, nprevs);
        }
#endif
        for (int r = r0; r < r1; ++r) {
            size_t t0 = (size_t)r * tiles_n + tile_n0;
            simd<uint32_t, NT * W> words, prevs;
#ifdef EXL3_DPAS_PREFETCH
            words = nwords; prevs = nprevs;
            if constexpr (PLANAR) { planarize<NT, D, G>(words); planarize<NT, D, G>(prevs); }
            if (r + 1 < r1) {
                size_t t1 = t0 + tiles_n;
                load_words<NT, W>(tr + t1 * W, false, nwords, nprevs);
            }
#else
#ifdef EXL3_L1_PF
            if (r + EXL3_PF_DIST < r1) pf_words<NT * W>(tr + (t0 + (size_t)EXL3_PF_DIST * tiles_n) * W);
#endif
            load_words<NT, W>(tr + t0 * W, t0 == 0, words, prevs);
            if constexpr (PLANAR) { planarize<NT, D, G>(words); planarize<NT, D, G>(prevs); }
#endif
#ifdef EXL3_DEBUG_A_ONCE
            // timing probe only (wrong results): activation block loaded once per thread, not per K-row
            simd<fp16, MB * 16> Am = block_load<fp16, MB * 16>(xbase + ((size_t)r0 * Mp + m0) * 16);
#else
            simd<fp16, MB * 16> Am = block_load<fp16, MB * 16>(xbase + ((size_t)r * Mp + m0) * 16);
#endif
#ifdef EXL3_BV_ALL
            // one B register block per tile: decoding tile j+1 no longer waits on tile j's dpas reading a shared
            // B register (write-after-read), so the ALU decode overlaps the asynchronous XMX work
            simd<fp16, 256 * NT> Ball;
#endif
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                simd<uint32_t, W> dw = words.template select<W, 1>(j * W);
                simd<uint32_t, W> dwprev = prevs.template select<W, 1>(j * W);
#ifdef EXL3_BV_ALL
                auto Bv = Ball.template select<256, 1>(256 * j);
#else
                simd<fp16, 256> Bv;
#endif
#ifdef EXL3_DEBUG_B_FAKE
                // timing probe only (wrong results): B operand = raw words as fp16, no trellis decode
                if constexpr (W >= 32) {
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                        Bv.template select<64, 1>(64 * q) = dw.template select<32, 1>(0).template bit_cast_view<fp16>();
                }
#else
                if constexpr (K == 4) build_k4(dw, dwprev, Bv);
                else build<0>(dw, dwprev, Bv);
#endif
                // DPAS repeat count RC rows per instruction: 8, or MB itself for MB=4 (no zero-row work)
                constexpr int RC = MB < 8 ? MB : 8;
#pragma unroll
                for (int rb = 0; rb < MB / RC; ++rb) {
                    auto c = acc.template select<RC * 16, 1>((j * MB + rb * RC) * 16);
                    c = xmx::dpas<8, RC, float, float, fp16, fp16>(
                        simd<float, RC * 16>(c), simd<fp16, 256>(Bv), simd<fp16, RC * 16>(Am.template select<RC * 16, 1>(rb * RC * 16)));
                }
            }
            if constexpr (FOLD) {
                // sum_k a_k per row, on the (idle) XMX unit: one dpas against an all-ones B per row block
                constexpr int RC = MB < 8 ? MB : 8;
                simd<fp16, 256> ones = fp16(1.0f);
#pragma unroll
                for (int rb = 0; rb < MB / RC; ++rb) {
                    auto c = accs.template select<RC * 16, 1>(rb * RC * 16);
                    c = xmx::dpas<8, RC, float, float, fp16, fp16>(
                        simd<float, RC * 16>(c), ones, simd<fp16, RC * 16>(Am.template select<RC * 16, 1>(rb * RC * 16)));
                }
            }
        }
        }
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int m = 0; m < MB; ++m)
                if (m < mrows) {
                    if constexpr (FOLD) {
                        simd<float, 16> o = acc.template select<16, 1>((j * MB + m) * 16);
                        o = o * kMul1C1 + simd<float, 16>(accs.template select<16, 1>(m * 16)) * kMul1C2;
                        block_store<float, 16>(part + ((size_t)p * M + m0 + m) * N + (tile_n0 + j) * 16, o);
                    } else {
                        block_store<float, 16>(part + ((size_t)p * M + m0 + m) * N + (tile_n0 + j) * 16,
                                               acc.template select<16, 1>((j * MB + m) * 16));
                    }
                }
        if constexpr (FUSED) fused_had_out(fa, part, M, N, Psplit, tile_n0 * 16, NT * 16);
    }
};


// ------------------------------------------------------------------------------------------------
// Reconstruct W_inner (Hadamard domain) as fp16 [Kdim, n_out], columns [n0, n0 + n_out) of the
// packed tensor. Thread = (tile-row, strip of NT tiles). Bit-exact.

template <int K, int CB, int NT, typename TW = fp16>
struct ReconstructKernel {
    const uint32_t* tr; TW* w;
    int tiles_n, tile_n0, n_out, n_strips, tk;
    float q8_inv = 0.0f;   // TW = int8_t: w = rne(W_inner * q8_inv)
    static constexpr int G = Geo<K>::G, V = Geo<K>::V, D = Geo<K>::D, W = Geo<K>::WORDS;
    static constexpr int P = V == 8 ? 4 : (V == 16 ? 2 : 1);
    static constexpr int STR = V == 8 ? 2 : (V == 16 ? 4 : 1);

    template <int u>
    static ESIMD_INLINE void build(simd<uint32_t, W>& dw, simd<uint32_t, W>& dwprev, simd<fp16, 256>& T) {
        if constexpr (u < V) {
            constexpr int e = (u + 1) * K;
            constexpr int i1 = (e - 1) / 32;
            constexpr int b0 = e - 16;
            constexpr int i0 = b0 >= 0 ? b0 / 32 : -1;
            constexpr int s = (i1 + 1) * 32 - e;
            simd<uint32_t, G> Bw = dw.template replicate_vs_w_hs<P, D, 8, D * P>(i1);
            simd<uint32_t, G> st;
            if constexpr (i0 == i1) st = Bw >> s;
            else {
                simd<uint32_t, G> Aw;
                if constexpr (i0 >= 0) Aw = dw.template replicate_vs_w_hs<P, D, 8, D * P>(i0);
                else Aw = dwprev.template replicate_vs_w_hs<P, D, 8, D * P>(0);
                if constexpr (s == 0) st = Bw; else st = (Aw << (32 - s)) | (Bw >> s);
            }
            simd<fp16, G> v = decode_cb_h<CB, G>(st & 0xFFFFu);
            constexpr int h = (u >> 2) & 1;
            constexpr int R0 = 8 * ((u >> 1) & 1) + 2 * ((u >> 3) & 3) + (u & 1);
            auto Tm = T.template bit_cast_view<fp16, 16, 16>();
#pragma unroll
            for (int b = 0; b < P; ++b)
                Tm.template select<1, 1, 8, 1>(R0 + STR * b, 8 * h) = v.template select<8, 1>(8 * b);
            build<u + 1>(dw, dwprev, T);
        }
    }

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int id = it.get_global_id(0);
        int strip = id % n_strips;
        int r = id / n_strips;
        if (r >= tk) return;
        int t0 = tile_n0 + strip * NT;
        simd<uint32_t, NT * W> words = block_load<uint32_t, NT * W>(tr + ((size_t)r * tiles_n + t0) * W);
        simd<fp16, 16 * 16 * NT> out;   // [16 rows][16*NT cols]
        auto Om = out.template bit_cast_view<fp16, 16, 16 * NT>();
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            simd<uint32_t, W> dw = words.template select<W, 1>(j * W);
            simd<uint32_t, W> dwprev;
            dwprev.template select<W - 1, 1>(1) = dw.template select<W - 1, 1>(0);
            dwprev[0] = dw[W - 1];
            simd<fp16, 256> T;
            build<0>(dw, dwprev, T);
            Om.template select<16, 1, 16, 1>(0, 16 * j) = T.template bit_cast_view<fp16, 16, 16>();
        }
#pragma unroll
        for (int rr = 0; rr < 16; ++rr)
        {
            simd<fp16, 16 * NT> row = out.template select<16 * NT, 1>(rr * 16 * NT);
            if constexpr (std::is_same_v<TW, fp16>)
                block_store<fp16, 16 * NT>(w + ((size_t)r * 16 + rr) * n_out + strip * NT * 16, row);
            else
                block_store<TW, 16 * NT>(w + ((size_t)r * 16 + rr) * n_out + strip * NT * 16,
                                         convert<TW>(rnde<float>(convert<float>(row) * q8_inv)));
        }
    }
};

}  // namespace exl3
