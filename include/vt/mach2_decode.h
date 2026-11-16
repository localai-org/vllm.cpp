// QUANT-MACH2-ADDITIVE U1 — the Mach-2 packed-format decode primitives.
//
// PORTED FROM SyzygyResearch/Mach-2-Additive-Medium `decode.py`
// (format tag `fn_packed_v2`), read 2026-11-16. Function anchors are named on
// each declaration. vLLM implements none of this, so the pin is
// `.agents/oracles/mach2-native.md` and the row is QUANT-MACH2-ADDITIVE.
//
// This header is the ORACLE half of the codec: it turns packed code streams into
// weights and does no matmul. Every operation here is integer-exact except
// `Bf16Round`, which is the release decoder's own deliberate rounding.
//
// BOTH rotations are here. `decode.py`'s `hadamard` carries two different
// radix-12/20 matrices - `_HS`, a hand-written sign table, for the spine path and
// the quadratic-residue construction for the expert - and they also differ in
// NORMALISATION: the spine branch runs the bare butterflies over its inner block
// and divides the whole product by sqrt(N), while the expert branch recurses into
// a normalised transform and divides by sqrt(radix). Mixing the two scales the
// result, which is exactly the defect the differential gate first found.
//
// `_HS` is transcribed here from the published `decode.py`, which reproduces its
// own `H @ H.T == radix * I` assertion in-file, so the tables are verifiable at
// rest rather than trusted.
//
// Two constants of the format are worth stating up front because they are easy to
// get wrong and the tests pin them: `wave_gamma` is indexed by anti-diagonal
// START, not by tile, so it must hold at least `mb + nb` entries (a 1x1 tiling
// reads index 1, and a one-element array is out of range); and each codeword's
// four values become four consecutive COLUMNS of one row, not four rows.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vt::mach2 {

// Constants as `decode.py` states them.
constexpr int kExpertValues = 4;      // EX_V: weights per codeword
constexpr int kExpertWindow = 16;     // EX_L: window width in bits
constexpr int kTile = 16;             // EX_TD / NE_TD: tile side
constexpr int kSpineValues = 2;       // NE_V: weights per codeword
constexpr int kSpineWindow = 16;      // NE_L
constexpr int kSpineStep = 8;         // NE_K * NE_V = 4 * 2
constexpr int kSpineTlutBits = 9;     // NE_TLUT_BITS
constexpr int kSpineTlutRows = 1 << kSpineTlutBits;  // 512 trained rows
constexpr int kSpineStates = 1 << 16;                // folded onto every window

namespace detail {
// The release decoder's `Bf16Round`, applied to one value.
float Bf16RoundOne(float x);
}  // namespace detail

// `wave_map(Mb, Nb)`: the order in which per-tile scales are stored. The walk
// starts at the bottom-left corner, then along the bottom row, and travels
// up-right along each anti-diagonal. Returns Mb*Nb indices, row-major.
std::vector<int32_t> WaveMap(int mb, int nb);

// `_ex_states` / `_ne_states`: one 16-bit window per codeword out of a packed
// `uint16` tape. Bits are consumed MSB first; the window advances by
// `step_bits` per codeword and the tape TAIL-BITES — the first
// `window - step_bits` bits are appended so the last codewords see a full
// window. `nstep` codewords are produced.
//
// Requires `step_bits * nstep <= 16 * nwords` and `step_bits <= window`.
std::vector<uint16_t> TrellisStates(const uint16_t* words, std::size_t nwords,
                                    int step_bits, std::int64_t nstep);

// `_lut(cb, K4)` + the gather in `decode_expert`: one LUT row per window state.
// `lut` holds `1 << window` rows of `kExpertValues` floats; every row is a D4
// lattice point times one shared scale, which is what makes the dot additive.
// `out` receives `nstep * kExpertValues` floats.
void ExpandExpertUnit(const uint16_t* states, std::int64_t nstep, const float* lut,
                      float* out);

// `_ne_full_lut` + `decode_ne_tensor`'s gather: the 512 trained rows folded onto
// all 2^16 window states. For state `s` the folded row is
// `row = (p >> 6) & (kSpineTlutRows - 1)` with `p = s * (s + 1)`, and bit 15 of
// `p` negates the row's FIRST component.
// `tlut` holds `kSpineTlutRows` rows of `kSpineValues` floats.
std::vector<float> FoldSpineLut(const float* tlut);

// `rc_grid(mx, k)`: the per-row/per-column rescale the spine path applies after
// the rotation, `mx * k / 127`.
void RcGrid(float mx, const std::int32_t* k, std::size_t n, float* out);

// `unpack_int5`: eight signed values packed into five bytes. The five bytes are
// the low 40 bits of a little-endian 64-bit word and each value is
// `((word >> (5 * i)) & 31) - 16`.
// `packed` holds `5 * n / 8` bytes; `n` must be a multiple of 8.
std::vector<std::int8_t> UnpackInt5(const std::uint8_t* packed, std::size_t n);

// `decode_embed`: asymmetric grouped integer dequantisation, the embedding tier.
// A code `q` in group `g` decodes to `mn[g] + q * (mx[g] - mn[g]) / (2^bits - 1)`,
// with the release decoder's `1e-8` floor on a degenerate group.
void DequantGrouped(const std::uint8_t* codes, const float* mn, const float* mx,
                    int bits, int group, std::size_t cols, float* out);

// The release decoder's `bf16_round`, elementwise: round-to-nearest-even at
// 16 bits. Used where `decode_ne_tensor` rounds its intermediate.
void Bf16Round(const float* in, std::size_t n, float* out);

// `hadamard`'s power-of-two branch (`_butterflies`): an orthonormal Walsh-Hadamard
// transform along the LAST axis, in place. `cols` must be a power of two.
void Butterflies(float* x, std::size_t rows, int cols);

// `_hmat`: the quadratic-residue construction behind `hadamard_12()` and
// `hadamard_20()`, the expert pair. Returns a `radix x radix` matrix `H` that
// satisfies `H == H^T` and `H H^T == radix * I` exactly. `radix` must be 12 or
// 20; the 12 case uses the character over Z/5 and the 20 case over Z/3[i].
std::vector<float> RadixMatrixGf(int radix);

// `H @ x` across the middle axis of `x` viewed as `(blocks, radix, cols)`,
// divided by `sqrt(norm)`. This is the radix half of `hadamard` with `kind`
// "expert": the caller has already run the butterflies on the `cols` axis.
void ApplyRadix(float* x, std::size_t blocks, int radix, int cols, double norm);

// The orthonormal transform of size `cols` for the expert path: butterflies when
// `cols` is a power of two, otherwise butterflies on the `cols / radix` axis
// followed by the quadratic-residue radix matrix, normalised by `sqrt(cols)`.
// `cols` must be a power of two, `12 * 2^k`, or `20 * 2^k`.
void HadamardExpert(float* x, std::size_t rows, int cols);

// `expert_signs(seeds, dim)`: the `+-1` vector the expert rotation is left- and
// right-multiplied by. `seed` is the 56-bit value `sign_seed()` derives as
// `sha256("{ns}|L{layer}|e{expert}|{proj}|{side}")[:7]`, big-endian; the SHA-256
// itself is the loader's job (it is not a decode primitive) and stays owed.
//
// The expansion is a Philox-style counter mixing: ten rounds over two key halves,
// 32-bit truncating multiplies, then the first output word read as a fraction of
// a turn and the second as a uniform - a lane is `-1` only when the turn lands in
// the upper half. Returns `dim` values of `+1` or `-1`.
std::vector<float> ExpertSigns(std::uint64_t seed, int dim);

// `hadamard(x, "spine")`: the spine path's orthonormal transform, radix 12 or 20
// times a power-of-two inner block. Note the normalisation difference from
// `HadamardExpert`: here the inner butterflies are bare and the whole product is
// divided by sqrt(cols). `cols` must be a power of two, `12 * 2^k` or `20 * 2^k`.
void HadamardSpine(float* x, std::size_t rows, int cols);

// `decode_ne_tensor`: one non-expert tensor from its packed tape.
//
//   unit = fold[states] as (mb, nb, kTile, kTile) -> (m, kTile, nb, kTile) -> (m, n)
//   unit *= wscale
//   unit = hadamard_spine(unit) * sign(su)          // along n
//   unit = hadamard_spine(unit^T) * sign(sv)        // along m
//   out = rc_grid(rc_max_m, |sv|) row-wise * bf16_round(out) * rc_grid(rc_max_n, |su|)
//
// `su` holds `n` signed values and `sv` holds `m`: each carries BOTH the rotation
// sign and, through its magnitude, the `rc_grid` multiplier. A codeword's two
// values become two consecutive COLUMNS of one row (the spine tape holds 8 bits
// per codeword: 4 bits for each of two weights).
//
// `states` holds `nstep = m * n / kSpineValues` windows, `tlut` is
// `kSpineTlutRows` rows of `kSpineValues`, and `out` receives `m * n` floats.
void DecodeNeTensor(const uint16_t* states, std::int64_t nstep, const float* tlut, int mb, int nb,
                    float wscale, const std::int32_t* su, const std::int32_t* sv, float rc_max_m,
                    float rc_max_n, float* out);

// `decode_expert`'s assembly for one expert: the packed tape's windows become
// tiles, each tile is scaled by its own `wave_gamma` and the expert's `wscale`,
// and the result is rotated on both axes and signed.
//
//   unit = lut[states] as (mb, nb, kTile, kTile) -> (mb, kTile, nb, kTile) -> (m, n)
//   unit[tile] *= gamma[wave_map(mb, nb)[tile]]
//   unit *= wscale
//   unit = hadamard(unit) * su            // along n, one sign per column
//   unit = hadamard(unit^T) * sv          // along m, one sign per row
//
// `m = mb * kTile`, `n = nb * kTile`, `states` holds `nstep` windows where
// `nstep = m * n / kExpertValues`, `lut` is `1 << kExpertWindow` rows of
// `kExpertValues`, `gamma` holds at least `mb + nb` entries (see above), and
// `su`/`sv` hold `n`/`m` signs. `out` receives `m * n` floats, row-major.
void DecodeExpertTile(const uint16_t* states, std::int64_t nstep, const float* lut,
                      int mb, int nb, const float* gamma, float wscale, const float* su,
                      const float* sv, float* out);

}  // namespace vt::mach2
