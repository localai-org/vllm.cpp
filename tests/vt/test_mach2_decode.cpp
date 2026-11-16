// QUANT-MACH2-ADDITIVE U1 — the Mach-2 packed-format decode primitives.
//
// PORTED FROM SyzygyResearch/Mach-2-Additive-Medium `decode.py` (format tag
// `fn_packed_v2`), read 2026-11-16. The expectations here do NOT call the
// implementation under test: each case re-derives its answer a second way, from
// the bit layout the source states, so a mirrored index or a swapped sign shows
// up as a disagreement rather than as two copies of the same mistake. The
// byte-exact differential against the release decoder is U2 and needs the
// fixtures; this suite is what makes U1 worth having before that download.
#include <doctest/doctest.h>

#include <vt/mach2_decode.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

// The MSB-first bit stream `decode.py` reads out of a uint16 tape, rebuilt here
// from the words themselves rather than by calling `vt::mach2::TrellisStates`.
std::vector<uint8_t> IndependentBits(const std::vector<uint16_t>& words) {
  std::vector<uint8_t> bits;
  bits.reserve(words.size() * 16);
  for (const auto w : words) {
    for (int j = 15; j >= 0; --j) bits.push_back(static_cast<uint8_t>((w >> j) & 1u));
  }
  return bits;
}

float BitsToFloat(uint32_t u) {
  float f = 0.0f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

uint32_t FloatToBits(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

}  // namespace

TEST_CASE("mach2 wave map walks the left edge up, then the bottom row left") {
  // mb=2, nb=3. `decode.py` builds its start list as the leftmost tile of each
  // row from the bottom up, then the bottom row right to left, and assigns the
  // START index to every tile on that anti-diagonal - the index is constant along
  // the walk, not a running counter. Two consequences are load-bearing:
  //
  //   * the bottom-left corner appears in BOTH lists, so its diagonal is visited
  //     at index 1 and again at 2, and the later visit wins;
  //   * the anti-diagonal through (0,2) and (1,1) therefore reads 2, and index 1
  //     ends up used by no tile at all.
  //
  // The first version of this expectation assumed a running counter and failed
  // here, which is what made the distinction visible; U2 confirms the matrix
  // against the oracle itself.
  const auto idx = vt::mach2::WaveMap(2, 3);
  REQUIRE(idx.size() == 6);
  const int32_t want[2][3] = {{4, 3, 2}, {3, 2, 0}};
  for (int r = 0; r < 2; ++r) {
    for (int c = 0; c < 3; ++c) {
      CHECK(idx[static_cast<size_t>(r * 3 + c)] == want[r][c]);
    }
  }
  // A 1x1 tiling is the whole degenerate case: the single start is produced by
  // both lists, so the tile is visited at index 0 and then at 1.
  CHECK(vt::mach2::WaveMap(1, 1) == std::vector<int32_t>{1});
}

TEST_CASE("mach2 trellis windows advance by step bits and tail-bite the tape") {
  // K4 = 6: four weights per codeword, six new bits each.
  const std::vector<uint16_t> tape = {0xABCD, 0x1234};
  const std::int64_t nstep = 5;
  const auto states = vt::mach2::TrellisStates(tape.data(), tape.size(), 6, nstep);
  REQUIRE(states.size() == static_cast<size_t>(nstep));

  // Independent derivation: truncate the stream at step*nstep, append the first
  // (window - step) bits, then read a 16-bit window per codeword.
  auto bits = IndependentBits(tape);
  const size_t need = static_cast<size_t>(6 * nstep);
  REQUIRE(bits.size() >= need);
  bits.resize(need);
  const size_t tail = static_cast<size_t>(vt::mach2::kExpertWindow - 6);
  const auto head = IndependentBits(tape);
  bits.insert(bits.end(), head.begin(), head.begin() + static_cast<ptrdiff_t>(tail));

  for (std::int64_t i = 0; i < nstep; ++i) {
    uint16_t want = 0;
    for (int j = 0; j < 16; ++j) {
      want = static_cast<uint16_t>((want << 1) | bits[static_cast<size_t>(i) * 6 + static_cast<size_t>(j)]);
    }
    CHECK(states[static_cast<size_t>(i)] == want);
  }
  // The first window is the first tape word verbatim: MSB-first means the first
  // sixteen bits are that word.
  CHECK(states[0] == 0xABCD);
  // A tape too short for the requested codewords yields no states rather than
  // reading past the end.
  const auto short_tape = vt::mach2::TrellisStates(tape.data(), 1, 6, 1000);
  for (const auto s : short_tape) CHECK(s == 0);
}

TEST_CASE("mach2 expert LUT expansion is a per-state row gather") {
  // A four-row LUT is enough to pin the mapping; the real table has 2^16 rows
  // and the gather is the same indexing.
  std::vector<float> lut(4 * vt::mach2::kExpertValues, 0.0f);
  for (size_t r = 0; r < 4; ++r) {
    for (int v = 0; v < vt::mach2::kExpertValues; ++v) {
      lut[r * vt::mach2::kExpertValues + static_cast<size_t>(v)] =
          static_cast<float>(100 * r + v);
    }
  }
  const std::vector<uint16_t> states = {3, 0, 2, 1};
  std::vector<float> out(states.size() * vt::mach2::kExpertValues, 0.0f);
  vt::mach2::ExpandExpertUnit(states.data(), static_cast<std::int64_t>(states.size()),
                              lut.data(), out.data());
  const float want[4][4] = {{300, 301, 302, 303},
                            {0, 1, 2, 3},
                            {200, 201, 202, 203},
                            {100, 101, 102, 103}};
  for (size_t s = 0; s < states.size(); ++s) {
    for (int v = 0; v < vt::mach2::kExpertValues; ++v) {
      CHECK(out[s * vt::mach2::kExpertValues + static_cast<size_t>(v)] ==
            want[s][static_cast<size_t>(v)]);
    }
  }
}

TEST_CASE("mach2 spine LUT folds 512 rows onto every window state") {
  std::vector<float> tlut(vt::mach2::kSpineTlutRows * vt::mach2::kSpineValues, 0.0f);
  for (size_t r = 0; r < vt::mach2::kSpineTlutRows; ++r) {
    tlut[r * 2] = 1.0f + static_cast<float>(r);
    tlut[r * 2 + 1] = -2.0f - static_cast<float>(r);
  }
  const auto full = vt::mach2::FoldSpineLut(tlut.data());
  REQUIRE(full.size() == static_cast<size_t>(vt::mach2::kSpineStates) * 2);
  for (std::int64_t s = 0; s < vt::mach2::kSpineStates; s += 997) {
    // Independent derivation of `p`, the folded row and the sign bit.
    const std::int64_t p = s * (s + 1);
    const auto row = static_cast<size_t>((p >> 6) % vt::mach2::kSpineTlutRows);
    const float sign = ((p >> 15) & 1) ? -1.0f : 1.0f;
    CHECK(full[static_cast<size_t>(s) * 2] == sign * (1.0f + static_cast<float>(row)));
    // Only the FIRST component takes the sign.
    CHECK(full[static_cast<size_t>(s) * 2 + 1] == -2.0f - static_cast<float>(row));
  }
  // State 0 is the all-zero window and must be untouched by the sign.
  CHECK(full[0] == 1.0f);
}

TEST_CASE("mach2 rc grid scales by mx over 127") {
  const std::int32_t k[5] = {0, 1, 2, 3, 127};
  float out[5] = {};
  vt::mach2::RcGrid(127.0f, k, 5, out);
  CHECK(out[0] == 0.0f);
  CHECK(out[1] == 1.0f);
  CHECK(out[2] == 2.0f);
  CHECK(out[3] == 3.0f);
  CHECK(out[4] == 127.0f);
  // A different ceiling scales linearly, which is what makes it a rescale and
  // not a lookup.
  float half[5] = {};
  vt::mach2::RcGrid(63.5f, k, 5, half);
  CHECK(half[4] == doctest::Approx(63.5f));
  CHECK(half[2] == doctest::Approx(1.0f));
}

TEST_CASE("mach2 int5 groups are five little-endian bytes holding eight values") {
  // 0x21 = 0b00100001: the low five bits are 1, so value 0 is 1 - 16 = -15.
  const std::uint8_t packed[5] = {0x21, 0x43, 0x65, 0x87, 0xA9};
  const auto vals = vt::mach2::UnpackInt5(packed, 8);
  REQUIRE(vals.size() == 8);

  // Independent derivation: assemble the 40-bit word little-endian, then take
  // five-bit fields from the bottom up.
  std::uint64_t word = 0;
  for (int b = 0; b < 5; ++b) word |= static_cast<std::uint64_t>(packed[b]) << (8 * b);
  for (int i = 0; i < 8; ++i) {
    const int want = static_cast<int>((word >> (5 * i)) & 31u) - 16;
    CHECK(static_cast<int>(vals[static_cast<size_t>(i)]) == want);
  }
  CHECK(static_cast<int>(vals[0]) == -15);
  // The range is [-16, 15] and never leaves it.
  for (const auto v : vals) {
    CHECK(static_cast<int>(v) >= -16);
    CHECK(static_cast<int>(v) <= 15);
  }
}

TEST_CASE("mach2 grouped dequant is affine per group and survives a flat group") {
  const std::uint8_t codes[6] = {0, 15, 0, 7, 15, 3};
  const float mn[2] = {0.0f, -1.0f};
  const float mx[2] = {15.0f, -1.0f};  // the second group is degenerate
  float out[6] = {};
  vt::mach2::DequantGrouped(codes, mn, mx, 4, 3, 6, out);
  CHECK(out[0] == doctest::Approx(0.0f));
  CHECK(out[1] == doctest::Approx(15.0f));   // the top code lands on `mx`
  CHECK(out[2] == doctest::Approx(0.0f));
  // A group whose mn equals mx must not divide by zero: the release decoder
  // floors the range at 1e-8, so every code in it decodes to `mn`.
  CHECK(out[3] == doctest::Approx(-1.0f));
  CHECK(out[4] == doctest::Approx(-1.0f));
  CHECK(out[5] == doctest::Approx(-1.0f));
}

TEST_CASE("mach2 bf16 round is nearest-even at sixteen bits") {
  // Exactly representable: unchanged.
  CHECK(vt::mach2::detail::Bf16RoundOne(1.0f) == 1.0f);
  CHECK(FloatToBits(vt::mach2::detail::Bf16RoundOne(1.0f)) == 0x3F800000u);
  // 0.1f is 0x3DCCCCCD; the low half 0xCCCD is above the halfway point, so it
  // rounds up to 0x3DCD0000 = 0.10009765625.
  CHECK(FloatToBits(vt::mach2::detail::Bf16RoundOne(0.1f)) == 0x3DCD0000u);
  CHECK(vt::mach2::detail::Bf16RoundOne(0.1f) == BitsToFloat(0x3DCD0000u));
  // A value EXACTLY halfway between two bf16 numbers. 0x3F808000 sits between
  // 1.0 (0x3F800000) and 1.00390625 (0x3F810000); nearest-EVEN rounds it down,
  // because the upper candidate's low mantissa bit is 0. This case exists because
  // the first version of this suite did not contain a tie at all, and a mutation
  // of the bias from 0x7FFF to 0x8000 (round-half-up) survived it. The odd
  // neighbour is the control: 0x3F818000 rounds UP to 0x3F820000.
  CHECK(FloatToBits(vt::mach2::detail::Bf16RoundOne(BitsToFloat(0x3F808000u))) == 0x3F800000u);
  CHECK(FloatToBits(vt::mach2::detail::Bf16RoundOne(BitsToFloat(0x3F818000u))) == 0x3F820000u);
  // The result is always on the bf16 grid, i.e. its low sixteen bits are zero.
  const float probes[4] = {3.14159f, -2.71828f, 1e-8f, 65504.0f};
  float rounded[4] = {};
  vt::mach2::Bf16Round(probes, 4, rounded);
  for (float r : rounded) CHECK((FloatToBits(r) & 0xFFFFu) == 0u);
  // Rounding is idempotent, which is what lets the spine path round an
  // intermediate and still land on the same grid twice.
  float again[4] = {};
  vt::mach2::Bf16Round(rounded, 4, again);
  for (int i = 0; i < 4; ++i) CHECK(again[i] == rounded[i]);
}

TEST_CASE("mach2 butterﬂy hadamard is its own inverse up to the size") {
  // 32 columns: the power-of-two branch.
  const int cols = 32;
  std::vector<float> x(static_cast<size_t>(cols)), orig(static_cast<size_t>(cols));
  for (int i = 0; i < cols; ++i) orig[static_cast<size_t>(i)] = static_cast<float>(i) - 7.0f;
  x = orig;
  vt::mach2::Butterflies(x.data(), 1, cols);
  // The unnormalised transform is H; applying it twice multiplies by `cols`.
  vt::mach2::Butterflies(x.data(), 1, cols);
  for (int i = 0; i < cols; ++i) CHECK(x[static_cast<size_t>(i)] == doctest::Approx(orig[static_cast<size_t>(i)] * cols));
}

TEST_CASE("mach2 radix matrices are symmetric and orthogonal to exact equality") {
  for (const int radix : {12, 20}) {
    const auto h = vt::mach2::RadixMatrixGf(radix);
    REQUIRE(h.size() == static_cast<size_t>(radix * radix));
    // H == H^T
    for (int i = 0; i < radix; ++i) {
      for (int j = 0; j < radix; ++j) {
        CHECK(h[static_cast<size_t>(i * radix + j)] == h[static_cast<size_t>(j * radix + i)]);
      }
    }
    // H H^T == radix * I, exactly: every entry is a small integer, so this is
    // an equality and not a tolerance.
    for (int i = 0; i < radix; ++i) {
      for (int j = 0; j < radix; ++j) {
        double acc = 0.0;
        for (int k = 0; k < radix; ++k) {
          acc += static_cast<double>(h[static_cast<size_t>(i * radix + k)]) *
                 static_cast<double>(h[static_cast<size_t>(j * radix + k)]);
        }
        CHECK(acc == doctest::Approx(i == j ? radix : 0).epsilon(1e-12));
      }
    }
    // The bordered construction puts a zero at C(0,0), so the first row is the
    // B corner pattern: +1, -1, repeating.
    CHECK(h[0] == 1.0f);
    CHECK(h[1] == -1.0f);
  }
}

TEST_CASE("mach2 expert hadamard is orthonormal on both radix and power-of-two sizes") {
  // 20 columns takes the radix-20 path with a one-wide butterfly block; 256 takes
  // the pure butterfly path. Applying the transform twice returns the input.
  for (const int cols : {20, 256}) {
    std::vector<float> x(static_cast<size_t>(cols)), orig(static_cast<size_t>(cols));
    for (int i = 0; i < cols; ++i) orig[static_cast<size_t>(i)] = 0.25f * static_cast<float>(i) - 3.0f;
    x = orig;
    vt::mach2::HadamardExpert(x.data(), 1, cols);
    vt::mach2::HadamardExpert(x.data(), 1, cols);
    for (int i = 0; i < cols; ++i) {
      CHECK(x[static_cast<size_t>(i)] == doctest::Approx(orig[static_cast<size_t>(i)]).epsilon(1e-4));
    }
  }
}

TEST_CASE("mach2 expert signs match the release decoder's expansion") {
  // The seeds are `sign_seed()`'s first seven SHA-256 bytes for the stated
  // (layer, expert, projection, side); the vectors are the release decoder's
  // `expert_signs` output, produced by a transcription of it. That means the
  // constants - the two multipliers, the two key increments, the ten rounds and
  // the 32-bit truncation between them - are all pinned, not just the shape: a
  // single changed digit anywhere moves every value in this table.
  struct Case {
    std::uint64_t seed_hi;
    std::uint64_t seed_lo;
    int first8[8];
    int ones;
  };
  const Case cases[] = {
      {4703694u, 21246858u, {1, 1, 1, -1, -1, 1, -1, 1}, 9},    // L0 e0 gate SU
      {4827869u, 212509842u, {-1, 1, -1, 1, -1, -1, 1, -1}, 5},  // L0 e1 gate SU
      {10181321u, 3738221035u, {-1, -1, -1, -1, -1, -1, 1, -1}, 6},  // L7 e300 down SV
  };
  for (const auto& c : cases) {
    const std::uint64_t seed = (c.seed_hi << 32) | c.seed_lo;
    const auto signs = vt::mach2::ExpertSigns(seed, 16);
    REQUIRE(signs.size() == 16);
    for (int i = 0; i < 8; ++i) CHECK(signs[static_cast<size_t>(i)] == static_cast<float>(c.first8[i]));
    int ones = 0;
    for (const float s : signs) {
      CHECK((s == 1.0f || s == -1.0f));
      if (s > 0.0f) ++ones;
    }
    CHECK(ones == c.ones);
  }
  // Same seed, same vector; a neighbouring seed must not reuse it. Without this a
  // stub returning all ones would pass the shape checks above on the third case
  // only by luck, and never distinguish two experts of the same layer.
  CHECK(vt::mach2::ExpertSigns(12345u, 16) == vt::mach2::ExpertSigns(12345u, 16));
  CHECK(vt::mach2::ExpertSigns(12345u, 16) != vt::mach2::ExpertSigns(12346u, 16));
}

TEST_CASE("mach2 expert tile assembly matches the release decoder's pipeline") {
  // A 16x16 tile is 64 codewords of four weights, i.e. 24 six-bit windows' worth
  // of tape. The LUT, gamma, wscale and signs here are synthetic but fixed, and
  // the expectations are `decode_expert`'s own output for them, computed by a
  // transcription of the release decoder.
  //
  // This case pins the OPERATION ORDER - gather, per-tile gamma, wscale, row
  // rotation and signs, then column rotation and signs - because any reordering
  // moves the checksum. It also pins the two layout facts that are easiest to get
  // wrong: a codeword's four values become four consecutive COLUMNS of one row,
  // and gamma is indexed by anti-diagonal start, so a 1x1 tiling reads index 1.
  std::vector<uint16_t> words(24);
  for (int i = 0; i < 24; ++i) words[static_cast<size_t>(i)] = static_cast<uint16_t>((i * 0x9E37 + 0x1234) & 0xFFFF);
  const auto states = vt::mach2::TrellisStates(words.data(), words.size(), 6, 64);
  REQUIRE(states.size() == 64);
  // The window walk is pinned against the same transcription, so a disagreement
  // here would mean the tape reader and the tile assembler disagree about which
  // codeword each window belongs to.
  const uint16_t head[] = {4660, 36140, 19206, 49581, 27470, 54184};
  for (int i = 0; i < 6; ++i) CHECK(states[static_cast<size_t>(i)] == head[i]);

  std::vector<float> lut(static_cast<size_t>(1 << 16) * 4);
  for (int s = 0; s < (1 << 16); ++s) {
    for (int v = 0; v < 4; ++v) {
      lut[static_cast<size_t>(s) * 4 + static_cast<size_t>(v)] =
          static_cast<float>(((s % 33) * 7 + v * 13) % 33 - 16) * 0.5f;
    }
  }
  // Two entries: the wave map of a 1x1 tiling is {{1}}, so a one-element array
  // would index out of range. The release decoder has the same requirement.
  const float gamma[2] = {0.75f, 0.5f};
  std::vector<float> su(16), sv(16);
  for (int j = 0; j < 16; ++j) {
    su[static_cast<size_t>(j)] = (j * 3) % 5 ? 1.0f : -1.0f;
    sv[static_cast<size_t>(j)] = (j * 7) % 4 ? 1.0f : -1.0f;
  }

  std::vector<float> tile(256);
  vt::mach2::DecodeExpertTile(states.data(), 64, lut.data(), 1, 1, gamma, 1.25f, su.data(),
                              sv.data(), tile.data());
  double sum = 0.0;
  double sumsq = 0.0;
  for (const float v : tile) {
    sum += static_cast<double>(v);
    sumsq += static_cast<double>(v) * static_cast<double>(v);
  }
  CHECK(sum == doctest::Approx(-5.0).epsilon(1e-9));
  CHECK(sumsq == doctest::Approx(2186.03515625).epsilon(1e-9));
  CHECK(tile[0] == doctest::Approx(0.95703125).epsilon(1e-6));
  CHECK(tile[15] == doctest::Approx(-1.93359375).epsilon(1e-6));
  CHECK(tile[15 * 16] == doctest::Approx(-3.53515625).epsilon(1e-6));
  CHECK(tile[7 * 16 + 9] == doctest::Approx(-4.51171875).epsilon(1e-6));
}
