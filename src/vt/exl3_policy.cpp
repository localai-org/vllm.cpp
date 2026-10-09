// EXL3 kernel-shape policy — MODEL-DSV4-EXL3 W2b, host tier.
//
// PORTED 1:1 FROM exllamav3 @ 2398c05635fbbad01a0a51dce63c85c6c8a8450e (MIT):
//   exllamav3_ext/quant/exl3_kernel_map.cuh:53-60  the shape + geometry macros
//   exllamav3_ext/quant/exl3_kernel_map.cu:23-91   select_gemm_shape,
//                                                  exl3_gemm_shape_compat
//   exllamav3_ext/quant/exl3_kernel_map.cu:153-160 the empty-block clamp
//   exllamav3_ext/quant/exl3_devctx.cu:32-46       the compute-capability bucket
//   exllamav3_ext/quant/exl3_gemv.cu:29-42,46-72,110-114  the m<=8 GEMV envelope
//                                                  and its two env knobs (W2c)
//   exllamav3_ext/quant/exl3_moe.cu:14-18          exl3_moe_max_concurrency (W2d)
//   exllamav3/modules/block_sparse_mlp.py:1079-1105  the MoE token sort (W2d)
//
// WHY THIS IS A .cpp AND NOT PART OF THE .cu. The table decides which kernel a
// shape gets, and it is pure integer arithmetic over (cc, m, k, n, bits). Left
// inside the CUDA translation unit it would be checkable only on a machine with
// a GPU, and a table nobody can check is a table that drifts. Here it is gated
// by `tests/vt/test_exl3_gemm.cpp` on any host, and the CUDA launcher calls
// exactly these functions rather than a second copy of them.
#include <cstdlib>
#include <string>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"

namespace vt {
namespace {

// exl3_kernel_map.cuh:53-56, in the macro's own order:
//   TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES
// plus EXL3_GEMM_BLOCKDIM (:60), whose slot 0 is upstream's unused `nullptr`
// entry. Index 0 exists so `shape_idx` stays 1-based exactly as upstream's
// instance arrays are.
constexpr Exl3GemmShape kShapes[5] = {
    {0, 0, 0, 0, 0, 0},
    {16, 16, 128, 6, 5, 256},
    {16, 32, 128, 4, 3, 512},
    {16, 32, 256, 4, 3, 512},
    {16, 16, 512, 4, 3, 256},
};

}  // namespace

Exl3Cc Exl3CcFromSm(int sm_major, int sm_minor) {
  // exl3_devctx.cu:39-43, verbatim, including the `major >= 8 && minor >= 9`
  // Ada test that reads the MINOR only after the major has qualified.
  if (sm_major >= 10) return Exl3Cc::kBlackwell;
  if (sm_major >= 9) return Exl3Cc::kHopper;
  if (sm_major >= 8 && sm_minor >= 9) return Exl3Cc::kAda;
  if (sm_major >= 8) return Exl3Cc::kAmpere;
  return Exl3Cc::kOld;
}

int Exl3GemmNumShapes() { return 4; }  // EXL3_GEMM_NUM_SHAPES (exl3_kernel_map.cuh:62)

Exl3GemmShape Exl3GemmShapeParams(int shape_idx) {
  VT_CHECK(shape_idx >= 0 && shape_idx <= Exl3GemmNumShapes(),
           "exl3_gemm: shape index out of range (0.." +
               std::to_string(Exl3GemmNumShapes()) +
               "); got " + std::to_string(shape_idx));
  return kShapes[shape_idx];
}

int Exl3SelectGemmShape(Exl3Cc cc, int size_m, int size_k, int size_n, int bits,
                        bool multi) {
  // exl3_kernel_map.cu:23-75. `size_m` is upstream's parameter and upstream
  // reads it nowhere in the body; it is kept so the port reads against its
  // anchor line for line.
  (void)size_m;
  const bool mod_256 = (size_n % 256 == 0);
  const bool mod_512 = (size_n % 512 == 0);
  const int K = bits;

  switch (cc) {
    case Exl3Cc::kOld:
    case Exl3Cc::kAmpere:
      if (mod_256 && K <= 4) {
        if (size_n <= 2048 || size_k <= 2048) return 2;
        return 3;
      }
      if (mod_256 && size_n < 4096) return size_k > 8192 ? 3 : 2;
      if (mod_512 && (size_n * size_k) > (4096 * 4096) && K <= 6) return 4;
      if (mod_256) return 3;
      return 2;

    case Exl3Cc::kAda:
      if (mod_256 && K <= 3) {
        if (size_k <= 2048 && !multi) return 2;
        if (size_n < 4096 && size_k <= 12288) return 2;
        return 3;
      }
      if (size_n <= 16384) return 2;
      if (mod_512 && size_n >= 32768) return 4;
      if (mod_256) return 3;
      return 2;

    case Exl3Cc::kHopper:
    case Exl3Cc::kBlackwell:
      if ((K == 4 || K == 2) && !multi) {
        if (size_k <= 2048) return 1;
      }
      if (K >= 7) {
        if (mod_256 && size_n <= 8192) return size_k > 32768 ? 3 : 2;
        if (mod_512 && size_n > 32768) return 4;
        return 2;
      }
      if (mod_256 && size_n <= 4096) return size_k > 8192 && K >= 3 ? 3 : 2;
      if (mod_512 && size_n > 16384) return 4;
      if (mod_256) return 3;
      return 2;
  }
  return 0;
}

bool Exl3GemmShapeCompat(int shape_idx, int size_k, int size_n) {
  // exl3_kernel_map.cu:86-91. Upstream also takes `size_m` and `K` and reads
  // neither; both are dropped rather than carried as parameters `-Werror` would
  // then need told about. Index 0 has no geometry and is compatible with
  // nothing, which is upstream's `nullptr` slot expressed as a predicate.
  const Exl3GemmShape s = Exl3GemmShapeParams(shape_idx);
  if (s.tile_k == 0 || s.tile_n == 0) return false;
  return (size_k % s.tile_k == 0) && (size_n % s.tile_n == 0);
}

int Exl3GemmNumSms(int shape_idx, int size_k, int size_n, int device_sms) {
  // exl3_kernel_map.cu:153-160: *num_sms = MAX(MIN(max_slices, *num_sms), 1),
  // with max_slices = size_k / tilesize_k * size_n / tilesize_n. The clamp is
  // what keeps a small problem from launching blocks with no tile to take, and
  // the trailing MAX(.., 1) is what keeps a launch from asking for zero.
  const Exl3GemmShape s = Exl3GemmShapeParams(shape_idx);
  VT_CHECK(s.tile_k > 0 && s.tile_n > 0,
           "exl3_gemm: shape " + std::to_string(shape_idx) + " has no geometry");
  const int max_slices = (size_k / s.tile_k) * (size_n / s.tile_n);
  const int lo = max_slices < device_sms ? max_slices : device_sms;
  return lo > 1 ? lo : 1;
}

// ─── W2c: the m<=8 GEMV envelope ─────────────────────────────────────────────

bool Exl3GemvHardEligible(int size_m, int size_k, int size_n, int bits, int codebook,
                          bool has_su_sv, bool half) {
  // OUR EXTENSION, NOT UPSTREAM'S: a HALF-INTEGER rate (K+0.5, mul1) must not
  // reach the GEMV. Its Exl3GemmArgs carries bits = KA, and e.g. a KA=3 frac
  // tensor passes every test below as (3, 2) — an arm that IS instantiated —
  // and would then decode the frac bitstream through the INTEGER-3 window
  // arithmetic: silently wrong, the failure class this tree refuses by name.
  // The frac GEMV arm is owed (BACKEND-ROCM, spec `## Owed`).
  if (half) return false;
  // exl3_gemv.cu:110-114, in upstream's own order. These are the tests upstream
  // takes BEFORE reading the environment or querying the device, because they
  // are free and they exclude almost every call.
  if (!has_su_sv) return false;
  if (bits < 2 || bits > 4) return false;
  if (bits != 4 && codebook == 0) return false;
  if (size_m > kExl3GemvMaxM) return false;
  if (size_k % 128 != 0 || size_n % 128 != 0) return false;
  return true;
}

int Exl3GemvSelectConfig(Exl3Cc cc, int size_m, int size_k, int size_n, int bits, int codebook,
                         int mode, int narrow_coresident) {
  // exl3_gemv.cu:46-72, verbatim. The `cc != CC_AMPERE` early return at :53 is
  // COMMENTED OUT upstream and is therefore absent here too; a line reproduced
  // as live that upstream has disabled would make every non-Ampere shape
  // ineligible and quietly delete the arm this row exists to evaluate.
  const int K = bits;
  const int cb = codebook;
  if (mode == 0) return -1;
  if (K < 2 || K > 4) return -1;
  if (K != 4 && cb == 0) return -1;
  if (size_m > kExl3GemvMaxM) return -1;
  if (size_k % 128 != 0 || size_n % 128 != 0) return -1;
  if (mode == 2) return size_n <= 8192 ? 0 : 1;
  if (mode == 3) return 0;  // testing: force narrow
  if (mode == 4) return 1;  // testing: force wide

  if (K == 2) return size_n <= 8192 ? 0 : 1;
  if (K == 3 && cc == Exl3Cc::kAda) return size_n <= 8192 ? 0 : 1;
  if (size_n / 32 <= narrow_coresident) return 0;
  if (size_k <= 2048 && size_n <= 8192) return 0;
  if (K == 3) return -1;
  if (size_n >= 8192 && size_k <= 4096) return 1;
  if (size_n >= 8192 && size_n <= 10240 && size_k <= 5120 && cc == Exl3Cc::kAmpere) return 1;
  return -1;
}

int Exl3GemvParseMode(const char* env_value) {
  // exl3_gemv.cu:29-34: unset is 1, everything else is atoi, which yields 0 for
  // an unparseable value. Kept as upstream spells it rather than made stricter,
  // because a knob that behaves differently from the oracle's is a knob that
  // cannot reproduce the oracle's run.
  if (env_value == nullptr) return 1;
  return std::atoi(env_value);
}

int Exl3GemvMode() {
  static const int mode = Exl3GemvParseMode(std::getenv("VT_EXL3_GEMV"));
  return mode;
}

int Exl3GemvParseSmemMode(const char* env_value) {
  // exl3_gemv.cu:37-42: unset is -1 (the per-bits default).
  if (env_value == nullptr) return -1;
  return std::atoi(env_value);
}

int Exl3GemvSmemMode() {
  static const int mode = Exl3GemvParseSmemMode(std::getenv("VT_EXL3_GEMV_SMEM"));
  return mode;
}

// ─── W2d: the fused MoE, host half ───────────────────────────────────────────

int Exl3MoeMaxConcurrency(int device_sms) {
  // exl3_moe.cu:14-18. The clamp at both ends is ours and is stated rather than
  // implied: upstream computes `num_sms / MOE_SMS_PER_EXPERT` on a real device,
  // where the value is neither zero nor above MOE_MAX_GROUPS, and this function
  // is called with values a test may choose.
  int c = device_sms / kExl3MoeSmsPerExpert;
  if (c > kExl3MoeMaxGroups) c = kExl3MoeMaxGroups;
  if (c < 1) c = 1;
  return c;
}

int Exl3MoeSortTokensByExpert(const int64_t* topk_ids, const float* topk_weights,
                              int64_t num_tokens, int64_t topk, int64_t num_experts,
                              int64_t max_tokens_per_expert, int64_t* expert_count,
                              int64_t* token_sorted, uint16_t* weight_sorted) {
  // block_sparse_mlp.py:1079-1105.
  VT_CHECK(num_tokens >= 0 && topk > 0 && num_experts > 0,
           "exl3_moe: the token sort needs topk > 0 and num_experts > 0");
  const int64_t assignments = num_tokens * topk;

  // `bincount(flat_expert, minlength = E + 1)` (:1100). The trailing slot is
  // upstream's sentinel for an assignment outside this shard's expert range.
  for (int64_t e = 0; e <= num_experts; ++e) expert_count[e] = 0;
  for (int64_t i = 0; i < assignments; ++i) {
    const int64_t e = topk_ids[i];
    VT_CHECK(e >= 0 && e <= num_experts,
             "exl3_moe: routed expert id " + std::to_string(e) + " is outside [0, " +
                 std::to_string(num_experts) + "]");
    ++expert_count[e];
  }

  // The grouping (:1095-1097). A counting sort, which is stable: the prefix sum
  // gives each expert its segment and each assignment lands in the order it was
  // read. Upstream's `argsort` is not stable and does not need to be.
  std::vector<int64_t> cursor(static_cast<size_t>(num_experts + 1), 0);
  int64_t running = 0;
  for (int64_t e = 0; e <= num_experts; ++e) {
    cursor[static_cast<size_t>(e)] = running;
    running += expert_count[e];
  }
  for (int64_t t = 0; t < num_tokens; ++t) {
    for (int64_t j = 0; j < topk; ++j) {
      const int64_t i = t * topk + j;
      const int64_t e = topk_ids[i];
      const int64_t slot = cursor[static_cast<size_t>(e)]++;
      token_sorted[slot] = t;  // `flat_token` is the interleaved arange (:1083)
      weight_sorted[slot] = F32ToF16(topk_weights[i]);
    }
  }

  // `num_active` (:1105): the experts the FUSED kernel will process. An expert
  // with no tokens has nothing to do; one above the cut does not fit the temp
  // buffers and is left to the caller's per-expert path (:1141,1151-1156).
  int num_active = 0;
  for (int64_t e = 0; e < num_experts; ++e)
    if (expert_count[e] > 0 && expert_count[e] <= max_tokens_per_expert) ++num_active;
  return num_active;
}

}  // namespace vt
