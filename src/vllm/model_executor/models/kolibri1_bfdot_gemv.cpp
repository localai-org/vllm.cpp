// Kolibri-1 BFDOT bf16 GEMV arm implementation (see kolibri1_bfdot_gemv.h
// for the contract, the numerics caveat, and the gating rationale). Compile
// shape mirrors cpu_paged_attn_bfdot_arm.cpp: on aarch64 THIS TU is built
// with -march=armv8.2-a+bf16 so <arm_neon.h> exposes vbfdotq_f32, and every
// entry is runtime-gated on the DetectArmBf16() HWCAP2 probe, so the binary
// still runs on non-bf16 aarch64 (portable scalar stub below serves those).
#include "vllm/model_executor/models/kolibri1_bfdot_gemv.h"

#include <cstdlib>

#include "vllm/model_executor/models/host_parallel.h"
#include "vt/cpu/cpu_isa_arm.h"
#include "vt/dtype.h"

#if defined(__aarch64__) && defined(__ARM_FEATURE_BF16)
#include <arm_neon.h>
#endif

namespace vllm {
namespace kolibri1_bfdot_gemv {
namespace {

bool EnvOne(const char* name) {
  const char* e = std::getenv(name);
  return e != nullptr && e[0] == '1';
}

bool EnvZero(const char* name) {
  const char* e = std::getenv(name);
  return e != nullptr && e[0] == '0';
}

}  // namespace

bool KnobRequested() { return EnvOne("VT_CPU_KOLIBRI_BFDOT_GEMV"); }

bool KnobActive() {
  if (!KnobRequested()) return false;
  if (vt::cpu::DetectArmBf16() == false) {
    // Refuse BY NAME: the operator asked for the arm and must never wonder
    // whether it engaged. Note the deliberate ABSENCE of a dotprod
    // requirement — FEAT_BF16 is independent of ASIMDDP and is the only
    // thing this kernel needs (see the header).
    VT_CHECK(false,
             "VT_CPU_KOLIBRI_BFDOT_GEMV=1 requires the aarch64 FEAT_BF16 "
             "extension (HWCAP2_BF16); this host does not report it, "
             "refusing the BFDOT kolibri1 GEMV arm");
  }
  return true;
}

bool Bf16DotActive() {
  // Legacy per-arm opt-in keeps working as an explicit `=1` (deprecated,
  // one-release bridge): it keeps the refuse-by-name contract above.
  if (KnobRequested()) return KnobActive();
  // The DEFAULT dispatch is feature-gated, not env-gated: vLLM dispatches
  // CPU numerics kernels on feature probes (SPEC kolibri1-bf16-dot-cpu
  // Design). The single escape hatch VT_CPU_KOLIBRI_BF16_DOT=0 forces the
  // f32 tier for every kolibri1 bf16-dot arm.
  if (EnvZero("VT_CPU_KOLIBRI_BF16_DOT")) return false;
  return vt::cpu::DetectArmBf16();
}

template <bool kF32Out>
inline void StoreOut(void* out_ptr, int64_t row, float acc) {
  if (kF32Out)
    static_cast<float*>(out_ptr)[row] = acc;
  else
    static_cast<uint16_t*>(out_ptr)[row] = vt::F32ToBF16(acc);
}

template <bool kF32Out>
void GemvBodyRows(const uint16_t* w_bf16, const uint16_t* x_bf16, void* out_ptr,
                  int64_t t, int64_t n, int64_t k) {
  // Threaded over OUTPUT elements only (t*n of them): every output element is
  // one worker's same-order instruction sequence (the host_parallel
  // contract), so the threaded result is bit-identical to the serial loop by
  // construction — AND to the t == 1 call of the same row, because the
  // per-element body below does not read t at all. The split may differ from
  // the t == 1 run's split; the host_parallel contract makes that a
  // scheduling difference, not a numerics one.
  host_parallel::ForOutputRows(t * n, k, [&](int64_t e0, int64_t e1) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_BF16)
    for (int64_t idx = e0; idx < e1; ++idx) {
      const int64_t i = idx / n;
      const int64_t row = idx % n;
      const uint16_t* w = w_bf16 + row * k;
      const uint16_t* x = x_bf16 + i * k;
      float32x4_t vacc = vdupq_n_f32(0.0f);
      int64_t e = 0;
      for (; e + 8 <= k; e += 8) {
        const uint16x8_t wv = vld1q_u16(w + e);
        const uint16x8_t xv = vld1q_u16(x + e);
        vacc = vbfdotq_f32(vacc, vreinterpretq_bf16_u16(wv),
                           vreinterpretq_bf16_u16(xv));
      }
      float acc = vaddvq_f32(vacc);
      // Scalar tail (k % 8): same bf16-pair semantics elementwise, f32
      // accumulate, so the tail never silently truncates. Kolibri's linear
      // K values (2560, 512, 6144) never hit it.
      for (; e < k; ++e) {
        acc = acc + vt::BF16ToF32(w[e]) * vt::BF16ToF32(x[e]);
      }
      StoreOut<kF32Out>(out_ptr, idx, acc);
    }
#else
    // Portable fallback: never selected at runtime (KnobActive refuses
    // without FEAT_BF16), kept so the TU links and the selection logic is
    // exercisable on any host.
    for (int64_t idx = e0; idx < e1; ++idx) {
      const int64_t i = idx / n;
      const int64_t row = idx % n;
      const uint16_t* w = w_bf16 + row * k;
      const uint16_t* x = x_bf16 + i * k;
      float acc = 0.0f;
      for (int64_t e = 0; e < k; ++e) {
        acc = acc + vt::BF16ToF32(w[e]) * vt::BF16ToF32(x[e]);
      }
      StoreOut<kF32Out>(out_ptr, idx, acc);
    }
#endif
  });
}

void Gemv(const uint16_t* w_bf16, const uint16_t* x_bf16, uint16_t* out_bf16,
          int64_t n, int64_t k) {
  GemvBodyRows<false>(w_bf16, x_bf16, out_bf16, 1, n, k);
}

void GemvF32Out(const uint16_t* w_bf16, const uint16_t* x_bf16, float* out_f32,
                int64_t n, int64_t k) {
  GemvBodyRows<true>(w_bf16, x_bf16, out_f32, 1, n, k);
}

void GemvRows(const uint16_t* w_bf16, const uint16_t* x_bf16, uint16_t* out_bf16,
              int64_t t, int64_t n, int64_t k) {
  GemvBodyRows<false>(w_bf16, x_bf16, out_bf16, t, n, k);
}

void GemvF32OutRows(const uint16_t* w_bf16, const uint16_t* x_bf16,
                    float* out_f32, int64_t t, int64_t n, int64_t k) {
  GemvBodyRows<true>(w_bf16, x_bf16, out_f32, t, n, k);
}

}  // namespace kolibri1_bfdot_gemv
}  // namespace vllm
