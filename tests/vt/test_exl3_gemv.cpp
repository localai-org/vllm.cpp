// The EXL3 m<=8 GEMV arm and its selection envelope — MODEL-DSV4-EXL3 W2c.
//
// PORTED FROM exllamav3 @ 2398c05635fbbad01a0a51dce63c85c6c8a8450e (MIT):
//   exllamav3_ext/quant/exl3_gemv.cu:29-42    the two env knobs
//   exllamav3_ext/quant/exl3_gemv.cu:46-72    exl3_gemv_cfg, the shape envelope
//   exllamav3_ext/quant/exl3_gemv.cu:108-114  the hard eligibility checks
//   exllamav3_ext/quant/exl3_gemv_kernel.cuh:31  EXL3_GEMV_MAX_M
//
// WHAT IS GATED HERE, AND WHAT IS NOT.
//
// The ENVELOPE is pure integer arithmetic over (cc, m, k, n, K, cb, mode,
// narrow_coresident) and is gated on any machine. That matters more here than it
// did for the GEMM shape table, because the sentence this row inherited about
// this envelope — "Ada/Blackwell are memory-bound here and keep the regular
// kernel" — describes a guard that is COMMENTED OUT at `exl3_gemv.cu:53`. A
// quoted comment is not a gate; these cases are.
//
// The KERNEL is not gated here on a machine with no GPU, and its bound is not
// tier 3. It accumulates in fp16 and folds to f32 only every FOLD iterations
// (`exl3_gemv_kernel.cuh:37-52,317-330`), which is a different NUMERIC arm from
// the f32-accumulating regular kernel. `.agents/specs/model-dsv4-exl3.md`
// `## W2cd design` W2c-3 states its own bound, tier 3c, and the device case
// below is where it is measured. That case SKIPS loudly and still asserts.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

#include "exl3_fixture.h"

namespace {

using exl3_test::Exl3Fixture;
using exl3_test::MakeFixture;
using exl3_test::Rng;
using exl3_test::UlpF16;

bool HasCudaExl3() {
  try {
    (void)vt::GetBackend(vt::DeviceType::kCUDA);
    return vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kCUDA);
  } catch (const std::runtime_error&) {
    return false;
  }
}

// The two shapes this checkpoint's TP1-coalesced experts have (spec
// `## The format`): w1/w3 are k=4096 n=2048, w2 is k=2048 n=4096.
constexpr int kW13K = 4096, kW13N = 2048;
constexpr int kW2K = 2048, kW2N = 4096;

}  // namespace

// ─── W2c-1: the compute-capability guard is DISABLED upstream ────────────────

TEST_CASE("exl3 gemv: no compute-capability test is live in the envelope") {
  // `exl3_gemv.cu:53` is `//if (cc != CC_AMPERE) return -1;`. If that line were
  // live, every non-Ampere bucket would return -1 for every shape. It is not, so
  // w2's shape is eligible on EVERY bucket that reaches the shape tests, and the
  // buckets differ from each other only where upstream's LIVE branches say they
  // do (`:65`, the Ada K==3 row).
  const int kMode = 1;  // the heuristic, upstream's default
  for (vt::Exl3Cc cc : {vt::Exl3Cc::kOld, vt::Exl3Cc::kAmpere, vt::Exl3Cc::kAda,
                        vt::Exl3Cc::kHopper, vt::Exl3Cc::kBlackwell}) {
    // w2, K = 3, cb = 1: `:67` `size_k <= 2048 && size_n <= 8192` fires for
    // every bucket, so config 0 (narrow), with NO occupancy input.
    CHECK(vt::Exl3GemvSelectConfig(cc, 1, kW2K, kW2N, 3, 1, kMode,
                                   /*narrow_coresident=*/0) == 0);
  }
}

// ─── W2c-2: what the envelope resolves to for THIS checkpoint ────────────────

TEST_CASE("exl3 gemv: w2 is eligible on GB10 and w1/w3 rest on an occupancy query") {
  const vt::Exl3Cc bw = vt::Exl3Cc::kBlackwell;
  const int kMode = 1;

  // w2 (k=2048, n=4096): `:67` fires. Independent of occupancy, so BOTH ends of
  // the occupancy range agree.
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, kW2K, kW2N, 3, 1, kMode, 0) == 0);
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, kW2K, kW2N, 3, 1, kMode, 1 << 20) == 0);

  // w1/w3 (k=4096, n=2048): `:67` does NOT fire, so `:68` `K == 3` returns -1
  // UNLESS `:66` fires first, which needs `2048 / 32 = 64 <= narrow_coresident`.
  // The threshold is EXACTLY 64 and both sides of it are pinned, because the
  // whole point of the entry is that the verdict is a device query this row has
  // not made.
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, kW13K, kW13N, 3, 1, kMode, 63) == -1);
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, kW13K, kW13N, 3, 1, kMode, 64) == 0);
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, kW13K, kW13N, 3, 1, kMode, 65) == 0);
}

TEST_CASE("exl3 gemv: the other branches of the envelope are upstream's too") {
  const int kMode = 1;
  // `:64` K == 2 splits on n at 8192, on EVERY bucket.
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, 4096, 8192, 2, 1, kMode, 0) == 0);
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, 4096, 8320, 2, 1, kMode, 0) == 1);
  // `:65` K == 3 on ADA takes the same split; on Blackwell it does not (that is
  // the one branch where the buckets genuinely differ).
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kAda, 1, 4096, 8192, 3, 1, kMode, 0) == 0);
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kAda, 1, 4096, 8320, 3, 1, kMode, 0) == 1);
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, 4096, 8320, 3, 1, kMode, 0) == -1);
  // `:69` K == 4, big n, small k -> the wide config.
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, 4096, 8192, 4, 1, kMode, 0) == 1);
  // `:70` is Ampere-only even though nothing else is.
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kAmpere, 1, 5120, 10240, 4, 1, kMode, 0) == 1);
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, 5120, 10240, 4, 1, kMode, 0) == -1);
  // `:48` mode 0 turns the whole path off; `:54` mode 2 takes it wherever the
  // hard constraints allow; `:55`/`:56` force one config.
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, kW2K, kW2N, 3, 1, 0, 0) == -1);
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, kW13K, kW13N, 3, 1, 2, 0) == 0);
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, kW13K, kW13N, 3, 1, 3, 0) == 0);
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, kW13K, kW13N, 3, 1, 4, 0) == 1);
}

TEST_CASE("exl3 gemv: the hard constraints refuse before any heuristic runs") {
  // `exl3_gemv.cu:110-114`, in upstream's own order.
  CHECK(vt::Exl3GemvHardEligible(1, kW2K, kW2N, 3, 1, /*has_su_sv=*/true));
  CHECK_FALSE(vt::Exl3GemvHardEligible(1, kW2K, kW2N, 3, 1, /*has_su_sv=*/false));
  CHECK_FALSE(vt::Exl3GemvHardEligible(1, kW2K, kW2N, 1, 1, true));   // K < 2
  CHECK_FALSE(vt::Exl3GemvHardEligible(1, kW2K, kW2N, 5, 1, true));   // K > 4
  CHECK_FALSE(vt::Exl3GemvHardEligible(1, kW2K, kW2N, 3, 0, true));   // K != 4 && cb == 0
  CHECK(vt::Exl3GemvHardEligible(1, kW2K, kW2N, 4, 0, true));         // K == 4 admits cb 0
  CHECK(vt::Exl3GemvHardEligible(vt::kExl3GemvMaxM, kW2K, kW2N, 3, 1, true));
  CHECK_FALSE(vt::Exl3GemvHardEligible(vt::kExl3GemvMaxM + 1, kW2K, kW2N, 3, 1, true));
  CHECK_FALSE(vt::Exl3GemvHardEligible(1, 2048 + 16, kW2N, 3, 1, true));  // size_k % 128
  CHECK_FALSE(vt::Exl3GemvHardEligible(1, kW2K, 4096 + 16, 3, 1, true));  // size_n % 128
  // OUR EXTENSION (BACKEND-ROCM frac rates): a HALF-INTEGER rate (K+0.5) is
  // never GEMV-eligible, no matter what integer shape its KA pretends to be —
  // (3, 2) below IS an instantiated arm, and the flag is what keeps a frac
  // tensor out of it. The frac GEMV arm is owed.
  CHECK(vt::Exl3GemvHardEligible(1, kW2K, kW2N, 3, 2, true));            // integer twin passes
  CHECK_FALSE(vt::Exl3GemvHardEligible(1, kW2K, kW2N, 3, 2, true, /*half=*/true));
  CHECK_FALSE(vt::Exl3GemvHardEligible(1, kW2K, kW2N, 2, 2, true, /*half=*/true));
  CHECK_FALSE(vt::Exl3GemvHardEligible(vt::kExl3GemvMaxM, kW2K, kW2N, 4, 2, true,
                                       /*half=*/true));
  // The constant itself, so a change to it is a red rather than a silent
  // widening (`exl3_gemv_kernel.cuh:31`).
  CHECK(vt::kExl3GemvMaxM == 8);
  // The envelope also enforces the hard bound on m, independently, because
  // upstream repeats it inside `exl3_gemv_cfg` (`:51`).
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, vt::kExl3GemvMaxM + 1, kW2K, kW2N, 3, 1,
                                 1, 0) == -1);
}

// ─── QUANT-EXL3-PERF A3: the envelope AT THE CHECKPOINT'S OWN SHAPES ────────
//
// The spec's admission table, executable. Read from the local safetensors
// headers of `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` on 2026-09-02: 409 trellis
// modules, all `mul1` (cb 2), widths {3: 137, 4: 270, 5: 1, 6: 1}, every one
// 128-aligned on both k and n.
//
// WHY THIS CASE EXISTS. Instantiating an arm is necessary and it is NOT
// sufficient: `Exl3GemvSelectConfig` returns -1 to DECLINE, and on Blackwell
// every bits-3 branch above the `if (K == 3) return -1;` line depends on
// `narrow_coresident`, which is an OCCUPANCY QUERY and therefore the one term a
// unit test cannot supply. So this case pins the THRESHOLD instead: it asserts
// the exact `narrow_coresident` at which each shape flips, which makes the
// spec's table a gate and turns a device measurement into a lookup rather than
// a guess. A zero end-to-end effect with the arm declined and a zero with the
// arm taken are DIFFERENT results, and this is what tells them apart.
TEST_CASE("exl3 gemv: the envelope's verdict at the #2495 checkpoint's real shapes") {
  constexpr int kMode = 1;  // the DEFAULT: production, not a forced testing mode
  const auto bw = vt::Exl3Cc::kBlackwell;

  // bits 3, cb 2 — 137 modules, two shapes. On Blackwell the only branch that
  // can admit them is `size_n / 32 <= narrow_coresident`; `size_k <= 2048` is
  // false at both (k is 5120 and 17408) and `if (K == 3) return -1;` follows.
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, 5120, 17408, 3, 2, kMode, 543) == -1);
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, 5120, 17408, 3, 2, kMode, 544) == 0);
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, 17408, 5120, 3, 2, kMode, 159) == -1);
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, 17408, 5120, 3, 2, kMode, 160) == 0);

  // The envelope is per-K and NOT per-cb, so (3, 2) inherits (3, 1)'s exactly.
  // If that ever stops holding, this row's whole "no envelope change" claim is
  // wrong, and it fails here rather than in a benchmark.
  for (int n : {17408, 5120}) {
    const int k = n == 17408 ? 5120 : 17408;
    for (int nc : {0, 159, 160, 543, 544, 1 << 20}) {
      CHECK(vt::Exl3GemvSelectConfig(bw, 1, k, n, 3, 1, kMode, nc) ==
            vt::Exl3GemvSelectConfig(bw, 1, k, n, 3, 2, kMode, nc));
    }
  }

  // bits 4, cb 2 — 270 modules, the largest single population and the one #2570
  // leads with. The arm is INSTANTIATED as of QUANT-EXL3-PERF slice B, so every
  // shape below is a live admission decision rather than a record for later.
  //
  // ALL EIGHT SHAPES, both sides of each threshold. The census is
  // {n=1024: 34, n=6144: 48, n=10240: 48, n=12288: 17, n=17408: 38,
  //  n=5120: 64 + 1 + 20}, and `size_n / 32` is the only term that moves.
  struct B4Shape {
    int k, n, threshold, modules;
  };
  const B4Shape kB4[] = {
      {5120, 1024, 32, 34},   {6144, 5120, 160, 64},    {10240, 5120, 160, 1},
      {17408, 5120, 160, 20}, {5120, 6144, 192, 48},    {5120, 10240, 320, 48},
      {5120, 12288, 384, 17}, {5120, 17408, 544, 38},
  };
  int total = 0;
  for (const B4Shape& s : kB4) {
    CAPTURE(s.k);
    CAPTURE(s.n);
    CHECK(s.n / 32 == s.threshold);
    CHECK(vt::Exl3GemvSelectConfig(bw, 1, s.k, s.n, 4, 2, kMode, s.threshold) == 0);
    CHECK(vt::Exl3GemvSelectConfig(bw, 1, s.k, s.n, 4, 2, kMode, s.threshold - 1) == -1);
    total += s.modules;
  }
  CHECK(total == 270);

  // AND THE PART THAT IS EASY TO GET BACKWARDS. K == 4 does not take the
  // bits-3 early `return -1;`, so the door stays open one line longer — and on
  // THIS checkpoint it leads nowhere. The next branch is the wide-config band
  // `size_n >= 8192 && size_k <= 4096`, and the smallest 4-bit `k` in the
  // artifact is 5120. So the wide config is NOT the escape from the narrow
  // config's occupancy ceiling here, at any shape this checkpoint has; what
  // bits 4 actually buys is LOWER thresholds (32 and 160 against the bits-3
  // shapes' 160 and 544), not a second admitting branch.
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, 5120, 17408, 4, 2, kMode, 543) == -1);  // not 1
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, 5120, 10240, 4, 2, kMode, 319) == -1);  // not 1
  // The band is real, and it is reachable only at a k this artifact does not
  // have. Both sides of that boundary, so the claim above is a gate.
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, 4096, 10240, 4, 2, kMode, 319) == 1);
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, 4224, 10240, 4, 2, kMode, 319) == -1);

  // The envelope is per-K, so `(4, 2)` inherits `(4, 1)`'s and `(4, 0)`'s
  // exactly, the same way `(3, 2)` inherits `(3, 1)`'s. This is what makes
  // "slice B adds no envelope change" a gate rather than a sentence.
  for (const B4Shape& s : kB4) {
    for (int nc : {0, s.threshold - 1, s.threshold, 1 << 20}) {
      CHECK(vt::Exl3GemvSelectConfig(bw, 1, s.k, s.n, 4, 2, kMode, nc) ==
            vt::Exl3GemvSelectConfig(bw, 1, s.k, s.n, 4, 1, kMode, nc));
      CHECK(vt::Exl3GemvSelectConfig(bw, 1, s.k, s.n, 4, 2, kMode, nc) ==
            vt::Exl3GemvSelectConfig(bw, 1, s.k, s.n, 4, 0, kMode, nc));
    }
  }

  // bits 5 and 6 have NO GEMV upstream either (`exl3_gemv.cu:110-111`), so the
  // one 5-bit tensor and the 6-bit lm_head falling to the regular shape table
  // is upstream's arrangement and not a gap. Asserted so it is not re-filed.
  CHECK_FALSE(vt::Exl3GemvHardEligible(1, 6144, 5120, 5, 2, true));
  CHECK_FALSE(vt::Exl3GemvHardEligible(1, 5120, 248320, 6, 2, true));

  // Mode 2 is upstream's "wherever the hard constraints allow" testing mode
  // (`exl3_gemv.cu:22`). It is what the diagnostic leg of the A/B uses, and it
  // must admit every bits-3 shape here regardless of occupancy, or that leg
  // measures the same path twice.
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, 5120, 17408, 3, 2, 2, 0) == 1);
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, 17408, 5120, 3, 2, 2, 0) == 0);
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, 5120, 17408, 4, 2, 2, 0) == 1);
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, 6144, 5120, 4, 2, 2, 0) == 0);
}

TEST_CASE("exl3 gemv: the env knobs parse exactly as upstream's do") {
  // `exl3_gemv.cu:29-34`: unset is 1, everything else is atoi.
  CHECK(vt::Exl3GemvParseMode(nullptr) == 1);
  CHECK(vt::Exl3GemvParseMode("0") == 0);
  CHECK(vt::Exl3GemvParseMode("1") == 1);
  CHECK(vt::Exl3GemvParseMode("2") == 2);
  CHECK(vt::Exl3GemvParseMode("4") == 4);
  // `exl3_gemv.cu:37-42`: unset is -1.
  CHECK(vt::Exl3GemvParseSmemMode(nullptr) == -1);
  CHECK(vt::Exl3GemvParseSmemMode("0") == 0);
  CHECK(vt::Exl3GemvParseSmemMode("1") == 1);
}

// ─── W2c-3: the device arm, and the bound that is NOT tier 3 ─────────────────

TEST_CASE("exl3 device: every instantiated GEMV arm meets tier 3c") {
  if (!HasCudaExl3()) {
    MESSAGE(
        "SKIPPED, no CUDA device: MODEL-DSV4-EXL3 W2c's tier-3c bound is PENDING, and so is "
        "`narrow_coresident`, the occupancy query that alone decides whether the w1/w3 shape "
        "is GEMV-eligible at all. dgx.casa is flapping. Reproduce with: "
        "rc run --device dgx:gpu0 -- ctest --test-dir build-cuda -R test_exl3_gemv -V");
    // A skip that asserts NOTHING reports `assertions: 0`, which reads as a pass.
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kCUDA));
    return;
  }
  // FORCED, through `Exl3GemmArgs::force_gemv`, which mirrors upstream's own
  // direct entry point (`exl3_gemv.cu:171-241`, "errors if the call is not
  // hard-eligible"). Forcing is what makes this a gate rather than a coin flip
  // on a heuristic: without it a device whose occupancy declines the shape would
  // measure the REGULAR kernel and report tier 3c green.
  vt::Backend& cb_be = vt::GetBackend(vt::DeviceType::kCUDA);
  vt::Queue hq = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();

  const int64_t kSingleM = 1;

  // EVERY INSTANTIATED ARM, and both CONFIGS of the envelope.
  //
  // (3, 1) is the SparkInfer DeepSeek-V4 artifact's arm. (3, 2) is 137 of the
  // 409 trellis modules of `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` — every MLP
  // projection quantized at the low end of its 3.5 bpw average. (4, 2) is the
  // other 270, the largest single population in that checkpoint and the one
  // #2570 leads with (QUANT-EXL3-PERF slice B).
  //
  // `force_gemv` drives the envelope at mode 2, whose whole rule is
  // `size_n <= 8192 ? 0 : 1` (`exl3_gemv.cu:54`). So n picks the config, and
  // the bits-4 arm is gated at BOTH: the narrow config (512 threads, WNT 2,
  // LSTRIDE 32, PF 4, FOLD 4) and the wide one (256 threads, WNT 4, PF 2,
  // FOLD 2). Those are different compiled kernels with different geometry, and
  // gating one of them would leave the other measured by nothing. The wide
  // config is upstream's own `K == 4` band (`:69`) and is the one arm of this
  // port that the #2495 checkpoint's shapes cannot reach, because its smallest
  // 4-bit `k` is 5120 and that band needs `size_k <= 4096`.
  struct Arm {
    int bits, cb;
    int64_t k, n;
    // m is the row count: 1 is the dq8/direct arm, 2..8 take the batched
    // Exl3GemvMK3 arm added under spec gfx1100-exl3-decode-60tps (the arm that
    // decodes each trellis tile once for the whole batch instead of
    // re-streaming it per row).
    int64_t m;
    // `std::string`, NOT `const char*`. Under doctest 2.5.2 a `const char*`
    // streamed into `MESSAGE` or `CAPTURE` decays to BOOL and prints `1`, so a
    // parameterised suite's per-arm diagnostic silently stops naming the arm --
    // and every number read off it is then attributed by assumed loop order.
    // That has already rotated three measured values across three axes in this
    // tree. The label is what the lease's evidence is read by, so it is typed
    // to print.
    std::string what;
  };
  const Arm kArms[] = {
      {3, 1, kW2K, kW2N, 1, "(3,1) narrow"},
      {3, 2, kW2K, kW2N, 1, "(3,2) narrow"},
      {3, 2, kW2K, kW2N, 8, "(3,2) narrow m8"},
      {4, 2, kW2K, kW2N, 1, "(4,2) narrow"},
      {4, 2, 2048, 8320, 1, "(4,2) wide"},
  };

  // The (3,1)/(3,2) device outputs are kept for the cross-arm check below.
  std::vector<std::vector<uint16_t>> bits3_per_cb;

  for (const Arm& arm : kArms) {
    CAPTURE(arm.what);
    CAPTURE(arm.bits);
    CAPTURE(arm.cb);
    const int64_t k = arm.k, n = arm.n;
    const int64_t m = arm.m;
    const int64_t tile_bytes = 32 * arm.bits;
    Exl3Fixture f = MakeFixture(k, n, arm.bits, 0x5EEDu);
    std::vector<uint16_t> a(static_cast<size_t>(m * k));
    Rng rng;
    for (auto& v : a) v = vt::F32ToF16(rng.next(1.0f));

    vt::Queue dq = cb_be.CreateQueue();

    // The reference is the CPU arm, which `test_exl3_gemm` already gates against
    // the f64 chain at tier 3, and which decodes EVERY width and codebook.
    // Comparing against it rather than re-deriving f64 here keeps ONE reference
    // for every device arm.
    //
    // `sibling` is the SAME width decoded with the OTHER codebook. It is what
    // makes the tolerance mean something: a `cb` threaded wrongly neither fails
    // to compile nor changes a shape, so it yields a weight with the right
    // DISTRIBUTION and no correlation to the true one. The assertion is that
    // the device output is close to its OWN codebook's reference and FAR from
    // the sibling's. That generalises to any arm, including one whose confusable
    // partner is not instantiated on the device at all — which is exactly
    // `(4, 2)`, whose partner `(4, 1)` this tree deliberately does not compile.
    const int sibling_cb = arm.cb == 2 ? 1 : 2;
    std::vector<uint16_t> ref(static_cast<size_t>(m * n), 0);
    std::vector<uint16_t> sib(static_cast<size_t>(m * n), 0);
    std::vector<uint16_t> got(static_cast<size_t>(m * n), 0);
    std::vector<uint16_t> a_had_h(static_cast<size_t>(m * k), 0);
    vt::Exl3GemmArgs ha;
    ha.bits = arm.bits;
    ha.codebook = arm.cb;
    for (int which = 0; which < 2; ++which) {
      vt::Exl3GemmArgs hargs = ha;
      hargs.codebook = which == 0 ? arm.cb : sibling_cb;
      uint16_t* out = which == 0 ? ref.data() : sib.data();
      vt::Tensor ta = vt::Tensor::Contiguous(a.data(), vt::DType::kF16, hq.device, {m, k});
      vt::Tensor tah = vt::Tensor::Contiguous(a_had_h.data(), vt::DType::kF16, hq.device, {m, k});
      vt::Tensor tc = vt::Tensor::Contiguous(out, vt::DType::kF16, hq.device, {m, n});
      vt::Tensor tb = vt::Tensor::Contiguous(f.trellis.data(), vt::DType::kI8, hq.device,
                                             {k / 16, n / 16, tile_bytes});
      vt::Tensor tsuh = vt::Tensor::Contiguous(f.suh.data(), vt::DType::kF16, hq.device, {k});
      vt::Tensor tsvh = vt::Tensor::Contiguous(f.svh.data(), vt::DType::kF16, hq.device, {n});
      vt::Exl3Gemm(hq, tc, ta, tb, tsuh, tsvh, tah, hargs);
    }

    void* d_a = cb_be.Alloc(a.size() * 2);
    void* d_ah = cb_be.Alloc(a.size() * 2);
    void* d_c = cb_be.Alloc(got.size() * 2);
    void* d_b = cb_be.Alloc(f.trellis.size() * 2);
    void* d_suh = cb_be.Alloc(f.suh.size() * 2);
    void* d_svh = cb_be.Alloc(f.svh.size() * 2);
    cb_be.Copy(dq, d_a, a.data(), a.size() * 2);
    cb_be.Copy(dq, d_b, f.trellis.data(), f.trellis.size() * 2);
    cb_be.Copy(dq, d_suh, f.suh.data(), f.suh.size() * 2);
    cb_be.Copy(dq, d_svh, f.svh.data(), f.svh.size() * 2);
    vt::Tensor da = vt::Tensor::Contiguous(d_a, vt::DType::kF16, dq.device, {m, k});
    vt::Tensor dah = vt::Tensor::Contiguous(d_ah, vt::DType::kF16, dq.device, {m, k});
    vt::Tensor dc = vt::Tensor::Contiguous(d_c, vt::DType::kF16, dq.device, {m, n});
    vt::Tensor db =
        vt::Tensor::Contiguous(d_b, vt::DType::kI8, dq.device, {k / 16, n / 16, tile_bytes});
    vt::Tensor dsuh = vt::Tensor::Contiguous(d_suh, vt::DType::kF16, dq.device, {k});
    vt::Tensor dsvh = vt::Tensor::Contiguous(d_svh, vt::DType::kF16, dq.device, {n});
    vt::Exl3GemmArgs da_args = ha;
    da_args.force_gemv = 1;  // upstream's `force`: bypasses the heuristic, not the hard checks
    vt::Exl3Gemm(dq, dc, da, db, dsuh, dsvh, dah, da_args);
    cb_be.Synchronize(dq);
    cb_be.Copy(dq, got.data(), d_c, got.size() * 2);
    cb_be.Synchronize(dq);

    // NOTHING LANDS DEAD. Everything above is FORCED, and a forced arm proves
    // the kernel works, never that anything reaches it. This second call leaves
    // `force_gemv` at its `Exl3GemmArgs` DEFAULT of -1 — exactly what a model's
    // linear method passes through `ModelRegistry::Forward` — so the envelope
    // decides at mode 1, on this device, with no test-only lever anywhere.
    //
    // At `k == 2048, n == 4096` the branch `size_k <= 2048 && size_n <= 8192`
    // (`exl3_gemv.cu:67`) fires on EVERY compute-capability bucket and takes NO
    // occupancy input, so this is the one shape whose default-mode verdict is a
    // constant rather than a device query. `arm.n == kW2N` selects it and the
    // wide leg, at n = 8320, is excluded because its verdict is not.
    //
    // The assertion is BYTE equality with the forced result, and that is what
    // makes it a reachability check rather than a second tolerance: the regular
    // shape-table kernel accumulates in f32 where this one accumulates in fp16,
    // so a fall-through would agree to tier 3c and disagree here. Deleting the
    // `Exl3GemvTryLaunch` call site turns this from equal to unequal.
    if (arm.n == kW2N) {
      std::vector<uint16_t> unforced(static_cast<size_t>(m * n), 0);
      void* d_c2 = cb_be.Alloc(unforced.size() * 2);
      vt::Tensor dc2 = vt::Tensor::Contiguous(d_c2, vt::DType::kF16, dq.device, {m, n});
      vt::Exl3GemmArgs prod = ha;  // force_gemv stays -1, force_shape_idx stays 0
      CHECK(prod.force_gemv == -1);
      CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, static_cast<int>(m),
                                     static_cast<int>(k), static_cast<int>(n), arm.bits, arm.cb,
                                     /*mode=*/1, /*narrow_coresident=*/0) == 0);
      vt::Exl3Gemm(dq, dc2, da, db, dsuh, dsvh, dah, prod);
      cb_be.Synchronize(dq);
      cb_be.Copy(dq, unforced.data(), d_c2, unforced.size() * 2);
      cb_be.Synchronize(dq);
      size_t same = 0;
      for (size_t i = 0; i < got.size(); ++i)
        if (unforced[i] == got[i]) ++same;
      MESSAGE(arm.what, " reached UNFORCED at mode 1: ", same, " of ", got.size(),
              " outputs byte-equal to the forced launch");
      CHECK(same == got.size());
      cb_be.Free(d_c2);
    }

    double sq = 0.0, rq = 0.0, worst = 0.0, sq_sib = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
      const double r = vt::F16ToF32(ref[i]);
      const double s = vt::F16ToF32(sib[i]);
      const double g = vt::F16ToF32(got[i]);
      sq += (g - r) * (g - r);
      sq_sib += (g - s) * (g - s);
      rq += r * r;
      worst = std::max(worst, std::fabs(g - r));
    }
    const double rms_ref = std::sqrt(rq / static_cast<double>(got.size()));
    const double rel = std::sqrt(sq / static_cast<double>(got.size())) / rms_ref;
    const double rel_sib = std::sqrt(sq_sib / static_cast<double>(got.size())) / rms_ref;
    MESSAGE(arm.what, " tier 3c: relative RMS ", rel, ", worst elementwise ", worst,
            ", relative RMS against codebook ", sibling_cb, " ", rel_sib);
    // `## W2cd design` W2c-3. NOT tier 3's 1.0e-3: this arm accumulates in fp16.
    // A NEW arm INHERITS this bound; it is never widened to admit one.
    CHECK(rel <= 6.0e-3);
    CHECK(worst <= 64.0 * UlpF16(rms_ref));
    // A GEMV that silently declined would leave `got` at its allocated content
    // and could still pass a tolerance against a reference that is also near
    // zero. It cannot pass this.
    CHECK(rms_ref > 0.0);
    // THE DISCRIMINATION CHECK, per arm. Two decodes of the same bits under
    // different codebooks are uncorrelated, so the distance to the sibling is
    // O(1) relative and the distance to the truth is O(1e-4). A hundredfold
    // margin below is not a tuned constant: it sits three orders of magnitude
    // above the measured `rel` and two below the measured `rel_sib`, so it
    // cannot be met by an arm that decoded with the wrong codebook.
    CHECK(rel_sib > 100.0 * 6.0e-3);

    if (arm.bits == 3 && arm.m == kSingleM) bits3_per_cb.push_back(got);

    cb_be.Free(d_a);
    cb_be.Free(d_ah);
    cb_be.Free(d_c);
    cb_be.Free(d_b);
    cb_be.Free(d_suh);
    cb_be.Free(d_svh);
    cb_be.DestroyQueue(dq);
  }

  // THE CROSS-ARM CHECK, kept from slice A. `(3, 1)` and `(3, 2)` are both
  // compiled here, so the two DEVICE outputs can be compared directly rather
  // than each against a host reference. If they agree the case is measuring one
  // arm twice and its tolerances mean nothing — which is the spec's stop
  // condition, not a tolerance to widen.
  // Same-m rows only: the m=8 batched arm writes a different output extent.
  REQUIRE(bits3_per_cb.size() == 2);
  size_t differing = 0;
  for (size_t i = 0; i < bits3_per_cb[0].size(); ++i)
    if (bits3_per_cb[0][i] != bits3_per_cb[1][i]) ++differing;
  MESSAGE("(3,1) vs (3,2) differ in ", differing, " of ", bits3_per_cb[0].size(), " outputs");
  CHECK(differing > bits3_per_cb[0].size() / 2);

  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(hq);
}
