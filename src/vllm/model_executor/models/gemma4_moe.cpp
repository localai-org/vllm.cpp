// Gemma-4 MoE: BF16 fused or FP8 per-expert + optional device resident.
#include "vllm/model_executor/models/gemma4_moe.h"
#include "vllm/model_executor/models/gemma4_indexed_gate.h"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vllm/model_executor/models/dense_attn_block.h"
#include "vllm/model_executor/models/device_pool.h"
#include "vllm/model_executor/layers/linear.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/unaligned.h"
#include "vt/ops.h"
#include "vt/fused_ops.h"

namespace vllm {
namespace {

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::ResidentWeight;
using vt::DType;
using vt::Tensor;

// Scratch reused across top-k experts within a token (and host H2D weight slots).
struct ExpertScratch {
  DBuf gu;    // [T, 2I] fused gate|up activations
  DBuf act;   // [T, I]
  DBuf gu_w;  // host-path [2I, H] weight upload
  DBuf down_w;
  // Sticky H2D key (expert identity). Do NOT key on buffer pointers — ephemeral
  // dequant reuses the same gu_tmp/dn_tmp addresses for every expert.
  const void* sticky_key = nullptr;
  ExpertScratch(Dev d, int64_t T, int64_t I, int64_t H)
      : gu(d, DType::kBF16, {T, 2 * I}),
        act(d, DType::kBF16, {T, I}),
        gu_w(d, DType::kBF16, {2 * I, H}),
        down_w(d, DType::kBF16, {H, I}) {}
};

void ExpertGeGLUHost(Dev d, DBuf& out, const Tensor& x, const uint16_t* gate_up_e,
                     const uint16_t* down_e, int64_t I, int64_t H, ExpertScratch& s,
                     const void* sticky_key = nullptr) {
  const int64_t T = x.shape[0];
  VT_CHECK(out.t().shape[0] >= T && out.t().shape[1] == H, "ExpertGeGLUHost out shape");
  VT_CHECK(s.gu.t().shape[0] >= T && s.act.t().shape[0] >= T, "ExpertGeGLUHost scratch T");
  const size_t gu_b = static_cast<size_t>(2 * I * H) * sizeof(uint16_t);
  const size_t dn_b = static_cast<size_t>(H * I) * sizeof(uint16_t);
  const void* key = sticky_key != nullptr ? sticky_key : static_cast<const void*>(gate_up_e);
  if (s.sticky_key != key) {
    d.b.Copy(d.q, s.gu_w.ptr(), gate_up_e, gu_b);
    d.b.Copy(d.q, s.down_w.ptr(), down_e, dn_b);
    s.sticky_key = key;
  }
  const vt::Device dev = d.q.device;
  Tensor gu_act =
      Tensor::Contiguous(s.gu.ptr(), DType::kBF16, dev, {T, 2 * I});
  Tensor act = Tensor::Contiguous(s.act.ptr(), DType::kBF16, dev, {T, I});
  Tensor out_view = Tensor::Contiguous(out.ptr(), DType::kBF16, dev, {T, H});
  vt::MatmulBT(d.q, gu_act, x, s.gu_w.t());
  vt::GeluAndMul(d.q, act, gu_act);
  vt::MatmulBT(d.q, out_view, act, s.down_w.t());
  // Per-expert drain: once-per-token-only still lost the server after pollution
  // (connection refused). Keep barrier until a safer fused device path exists.
  d.b.Synchronize(d.q);
}

void ExpertGeGLUDeviceAccum(Dev d, DBuf& out, const Tensor& x, const uint16_t* gate_up_e,
                            const uint16_t* down_e, int64_t I, int64_t H, ExpertScratch& s,
                            float alpha, float beta) {
  const int64_t T = x.shape[0];
  VT_CHECK(s.gu.t().shape[0] >= T && s.act.t().shape[0] >= T, "ExpertGeGLUDeviceAccum scratch T");
  const vt::Device dev = d.q.device;
  // gate_up_e is contiguous [2I, H] — one BT GEMM instead of two.
  Tensor gu_w =
      Tensor::Contiguous(const_cast<uint16_t*>(gate_up_e), DType::kBF16, dev, {2 * I, H});
  Tensor gu_act =
      Tensor::Contiguous(s.gu.ptr(), DType::kBF16, dev, {T, 2 * I});
  Tensor act = Tensor::Contiguous(s.act.ptr(), DType::kBF16, dev, {T, I});
  vt::MatmulBT(d.q, gu_act, x, gu_w);
  vt::GeluAndMul(d.q, act, gu_act);
  vt::MatmulBTAlphaBeta(d.q, out.ptr(), act.data, down_e, static_cast<int>(T),
                                  static_cast<int>(H), static_cast<int>(I), alpha, beta,
                                  DType::kBF16);
}

void ExpertGeGLUFp8Native(Dev d, DBuf& out, const Tensor& x, const void* fp8_gu,
                          const void* s_gu, const void* fp8_dn, const void* s_dn, int64_t I,
                          int64_t H, ExpertScratch& s, float alpha, float beta,
                          const void* weight_id = nullptr) {
  const int64_t T = x.shape[0];
  VT_CHECK(out.t().shape[0] >= T && out.t().shape[1] == H, "ExpertGeGLUFp8Native out");
  VT_CHECK(s.gu.t().shape[0] >= T && s.act.t().shape[0] >= T, "ExpertGeGLUFp8Native scratch T");
  // T==1 + beta=0: fused ExpertGeGLU (decode). Weights stay FP8.
  if (T == 1 && beta == 0.f) {
    const float wts[1] = {alpha};
    if (vt::ExpertGeGLUFp8TopKM1(d.q, out.ptr(), x.data, &fp8_gu, &s_gu, &fp8_dn, &s_dn, wts,
                                 /*G=*/1, static_cast<int>(I), static_cast<int>(H))) {
      return;
    }
  }
  // Prefill T>1: custom Fp8ChannelGemmMKernel is serial-in-M (slow). Dequant FP8→BF16
  // once on GPU, then hipBLAS MatmulBT (same path as ExpertGeGLUDeviceAccum).
  // weight_id: stable expert identity when fp8_gu is an ephemeral peer staging ptr
  // (same address for every expert — must NOT key sticky on that pointer alone).
  const vt::Device dev = d.q.device;
  const void* key = weight_id != nullptr ? weight_id : fp8_gu;
  if (s.sticky_key != key) {
    vt::DequantFp8ChannelBf16(d.q, s.gu_w.ptr(), fp8_gu, s_gu, static_cast<int>(2 * I),
                              static_cast<int>(H));
    vt::DequantFp8ChannelBf16(d.q, s.down_w.ptr(), fp8_dn, s_dn, static_cast<int>(H),
                              static_cast<int>(I));
    s.sticky_key = key;
  }
  Tensor gu_w = Tensor::Contiguous(s.gu_w.ptr(), DType::kBF16, dev, {2 * I, H});
  Tensor gu_act = Tensor::Contiguous(s.gu.ptr(), DType::kBF16, dev, {T, 2 * I});
  Tensor act = Tensor::Contiguous(s.act.ptr(), DType::kBF16, dev, {T, I});
  vt::MatmulBT(d.q, gu_act, x, gu_w);
  vt::GeluAndMul(d.q, act, gu_act);
  vt::MatmulBTAlphaBeta(d.q, out.ptr(), act.data, s.down_w.ptr(), static_cast<int>(T),
                        static_cast<int>(H), static_cast<int>(I), alpha, beta, DType::kBF16);
}

// Top-k FP8 native: gate_up×G → one GeluAndMul → down×G (alpha/beta mix). Decode M=1.
bool ExpertGeGLUFp8TopKFusedGelu(Dev d, DBuf& ysum, const Tensor& x, const void* const* fp8_gu,
                                 const void* const* s_gu, const void* const* fp8_dn,
                                 const void* const* s_dn, const float* wts, int G, int64_t I,
                                 int64_t H) {
  if (G <= 0 || x.shape[0] != 1) return false;
  struct Tls {
    int dev = -1;
    int Gcap = 0;
    int64_t I = 0, H = 0;
    std::optional<DBuf> gu;   // [G, 2I]
    std::optional<DBuf> act;  // [G, I]
  };
  static thread_local Tls tls;
  if (tls.dev != d.q.device.index || tls.Gcap < G || tls.I != I || tls.H != H) {
    tls.gu.emplace(d, DType::kBF16, std::vector<int64_t>{G, 2 * I});
    tls.act.emplace(d, DType::kBF16, std::vector<int64_t>{G, I});
    tls.dev = d.q.device.index;
    tls.Gcap = G;
    tls.I = I;
    tls.H = H;
  }
  const vt::Device dev = d.q.device;
  const size_t gu_row = static_cast<size_t>(2 * I) * 2;
  const size_t act_row = static_cast<size_t>(I) * 2;
  const int Ngu = static_cast<int>(2 * I);
  const int Nh = static_cast<int>(H);
  const int Ki = static_cast<int>(I);
  const int Kh = static_cast<int>(H);

  for (int g = 0; g < G; ++g) {
    void* gu_out = static_cast<char*>(tls.gu->ptr()) + static_cast<size_t>(g) * gu_row;
    vt::MatmulBTFp8Channel(d.q, gu_out, x.data, fp8_gu[g], s_gu[g], /*M=*/1, Ngu, Kh, 1.f, 0.f);
  }
  Tensor gu_all = Tensor::Contiguous(static_cast<uint16_t*>(tls.gu->ptr()), DType::kBF16, dev,
                                     {G, 2 * I});
  Tensor act_all = Tensor::Contiguous(static_cast<uint16_t*>(tls.act->ptr()), DType::kBF16, dev,
                                      {G, I});
  vt::GeluAndMul(d.q, act_all, gu_all);
  for (int g = 0; g < G; ++g) {
    const float alpha = wts[g];
    const float beta = (g == 0) ? 0.f : 1.f;
    void* act_g = static_cast<char*>(tls.act->ptr()) + static_cast<size_t>(g) * act_row;
    vt::MatmulBTFp8Channel(d.q, ysum.ptr(), act_g, fp8_dn[g], s_dn[g], /*M=*/1, Nh, Ki, alpha,
                           beta);
  }
  return true;
}

// Top-k experts: all gate_up GEMMs → one GeluAndMul → all down GEMMs (alpha/beta mix).
// Cuts (top_k-1) Gelu launches vs per-expert ExpertGeGLUDeviceAccum.
// Uses MatmulBTAlphaBetaRocm directly (no vt::MatmulBT dispatch overhead).
bool ExpertGeGLUTopKFusedGelu(Dev d, DBuf& ysum, const Tensor& x, const uint16_t* const* gu_ptrs,
                              const uint16_t* const* dn_ptrs, const float* wts, int G, int64_t I,
                              int64_t H) {
  if (G <= 0 || x.shape[0] != 1) return false;
  struct Tls {
    int dev = -1;
    int Gcap = 0;
    int64_t I = 0, H = 0;
    std::optional<DBuf> gu;   // [G, 2I]
    std::optional<DBuf> act;  // [G, I]
  };
  static thread_local Tls tls;
  if (tls.dev != d.q.device.index || tls.Gcap < G || tls.I != I || tls.H != H) {
    tls.gu.emplace(d, DType::kBF16, std::vector<int64_t>{G, 2 * I});
    tls.act.emplace(d, DType::kBF16, std::vector<int64_t>{G, I});
    tls.dev = d.q.device.index;
    tls.Gcap = G;
    tls.I = I;
    tls.H = H;
  }
  const vt::Device dev = d.q.device;
  const size_t gu_row = static_cast<size_t>(2 * I) * 2;
  const size_t act_row = static_cast<size_t>(I) * 2;
  const int Ngu = static_cast<int>(2 * I);
  const int Nh = static_cast<int>(H);
  const int Ki = static_cast<int>(I);
  const int Kh = static_cast<int>(H);

  // Phase 1: gate_up GEMMs into packed [G, 2I]
  for (int g = 0; g < G; ++g) {
    void* gu_out = static_cast<char*>(tls.gu->ptr()) + static_cast<size_t>(g) * gu_row;
    vt::MatmulBTAlphaBeta(d.q, gu_out, x.data, gu_ptrs[g], /*M=*/1, Ngu, Kh, 1.f, 0.f,
                                    DType::kBF16);
  }

  // Phase 2: single GeluAndMul over all experts
  Tensor gu_all = Tensor::Contiguous(static_cast<uint16_t*>(tls.gu->ptr()), DType::kBF16, dev,
                                     {G, 2 * I});
  Tensor act_all = Tensor::Contiguous(static_cast<uint16_t*>(tls.act->ptr()), DType::kBF16, dev,
                                      {G, I});
  vt::GeluAndMul(d.q, act_all, gu_all);

  // Phase 3: down GEMMs with alpha/beta accumulate into ysum
  for (int g = 0; g < G; ++g) {
    const float alpha = wts[g];
    const float beta = (g == 0) ? 0.f : 1.f;
    void* act_g = static_cast<char*>(tls.act->ptr()) + static_cast<size_t>(g) * act_row;
    vt::MatmulBTAlphaBeta(d.q, ysum.ptr(), act_g, dn_ptrs[g], /*M=*/1, Nh, Ki, alpha,
                                    beta, DType::kBF16);
  }
  return true;
}

// Batched top-k path (gather+strided or pointer-batch): currently disabled.
// Lab: gather+strided produced wrong tokens (~23 t/s); pointer-batch ~0.8 t/s.
// Serial / fused-gelu top-k remains the correct path (~34 t/s).
bool ExpertGeGLUDeviceBatched(Dev /*d*/, DBuf& /*ysum*/, const Tensor& /*x*/,
                              const std::vector<const uint16_t*>& /*gu_ptrs*/,
                              const std::vector<const uint16_t*>& /*dn_ptrs*/,
                              const std::vector<float>& /*wts*/, int64_t /*I*/, int64_t /*H*/) {
  return false;
}

}  // namespace

namespace {
// Bound permanent host BF16 expert packs. Unbounded cache + hipHostRegister OOM'd
// the 30G box (~27G RSS) during pollution. Default 2 GiB; override VT_GEMMA4_HOST_EXPERT_MB.
struct HostExpertLru {
  struct Slot {
    const Gemma4Fp8ExpertMats* ex = nullptr;
    size_t bytes = 0;
    uint64_t tick = 0;
  };
  std::vector<Slot> slots;
  size_t used = 0;
  uint64_t tick = 1;

  size_t BudgetBytes() const {
    static const size_t b = []() -> size_t {
      size_t mb = 2048;
      if (const char* e = std::getenv("VT_GEMMA4_HOST_EXPERT_MB")) {
        const long v = std::strtol(e, nullptr, 10);
        if (v == 0) return size_t{0};
        if (v > 0) mb = static_cast<size_t>(v);
      }
      return mb * static_cast<size_t>(1024ull * 1024ull);
    }();
    return b;
  }

  void EvictOne() {
    if (slots.empty()) return;
    size_t victim = 0;
    for (size_t i = 1; i < slots.size(); ++i) {
      if (slots[i].tick < slots[victim].tick) victim = i;
    }
    auto& s = slots[victim];
    if (s.ex) {
      UnpinGemma4Fp8ExpertHostCache(*s.ex);
      s.ex->cached_gu.clear();
      s.ex->cached_gu.shrink_to_fit();
      s.ex->cached_dn.clear();
      s.ex->cached_dn.shrink_to_fit();
    }
    used = used >= s.bytes ? used - s.bytes : 0;
    slots.erase(slots.begin() + static_cast<std::ptrdiff_t>(victim));
  }

  void MakeRoom(size_t need) {
    const size_t bud = BudgetBytes();
    if (bud == 0) {
      // No permanent host cache — caller should use ephemeral path.
      return;
    }
    while (used + need > bud && !slots.empty()) EvictOne();
  }

  void Note(const Gemma4Fp8ExpertMats* ex, size_t bytes) {
    const size_t bud = BudgetBytes();
    if (bud == 0) return;
    // Already tracked?
    for (auto& s : slots) {
      if (s.ex == ex) {
        s.tick = tick++;
        return;
      }
    }
    MakeRoom(bytes);
    if (used + bytes > bud) {
      // Still no room for a single expert — keep this one untracked; drop immediately
      // after use is caller's problem. Prefer: allow one oversize by evicting all.
      while (!slots.empty()) EvictOne();
    }
    if (used + bytes > bud) return;
    slots.push_back(Slot{ex, bytes, tick++});
    used += bytes;
  }

  void Touch(const Gemma4Fp8ExpertMats* ex) {
    for (auto& s : slots) {
      if (s.ex == ex) {
        s.tick = tick++;
        return;
      }
    }
  }
};

HostExpertLru& HostCacheLru() {
  static HostExpertLru lru;
  return lru;
}
}  // namespace

// Ensure FP8 expert has BF16 cache filled (idempotent), under host LRU budget.
void EnsureGemma4Fp8ExpertCached(const Gemma4Fp8ExpertMats& ex, int64_t I, int64_t H) {
  if (!ex.cached_gu.empty() && !ex.cached_dn.empty() &&
      static_cast<int64_t>(ex.cached_gu.size()) == 2 * I * H &&
      static_cast<int64_t>(ex.cached_dn.size()) == H * I) {
    HostCacheLru().Touch(&ex);
    return;
  }
  const size_t bytes =
      (static_cast<size_t>(2 * I * H) + static_cast<size_t>(H * I)) * sizeof(uint16_t);
  // Budget 0 → do not retain permanent packs (ephemeral-only mode).
  if (HostCacheLru().BudgetBytes() == 0) return;

  HostCacheLru().MakeRoom(bytes);
  ex.cached_gu.resize(static_cast<size_t>(2 * I * H));
  ex.cached_dn.resize(static_cast<size_t>(H * I));
  DequantFp8ChannelToBf16(ex.gate_w.bytes.data(),
                          reinterpret_cast<const uint16_t*>(ex.gate_s.bytes.data()), I, H,
                          ex.cached_gu.data());
  DequantFp8ChannelToBf16(ex.up_w.bytes.data(),
                          reinterpret_cast<const uint16_t*>(ex.up_s.bytes.data()), I, H,
                          ex.cached_gu.data() + I * H);
  DequantFp8ChannelToBf16(ex.down_w.bytes.data(),
                          reinterpret_cast<const uint16_t*>(ex.down_s.bytes.data()), H, I,
                          ex.cached_dn.data());
  PinGemma4Fp8ExpertHostCache(ex);
  HostCacheLru().Note(&ex, bytes);
}

// Dequant into caller buffers without retaining a permanent host BF16 cache.
// Used by dual-GPU resident upload (must not pin ~1.5GiB/layer on host).
void DequantGemma4Fp8ExpertToBf16Ephemeral(const Gemma4Fp8ExpertMats& ex, int64_t I,
                                           int64_t H, uint16_t* gate_up_out,
                                           uint16_t* down_out) {
  VT_CHECK(gate_up_out && down_out, "fp8 expert ephemeral dequant null out");
  if (!ex.cached_gu.empty() && !ex.cached_dn.empty() &&
      static_cast<int64_t>(ex.cached_gu.size()) == 2 * I * H &&
      static_cast<int64_t>(ex.cached_dn.size()) == H * I) {
    std::memcpy(gate_up_out, ex.cached_gu.data(), ex.cached_gu.size() * sizeof(uint16_t));
    std::memcpy(down_out, ex.cached_dn.data(), ex.cached_dn.size() * sizeof(uint16_t));
    return;
  }
  DequantFp8ChannelToBf16(ex.gate_w.bytes.data(),
                          reinterpret_cast<const uint16_t*>(ex.gate_s.bytes.data()), I, H,
                          gate_up_out);
  DequantFp8ChannelToBf16(ex.up_w.bytes.data(),
                          reinterpret_cast<const uint16_t*>(ex.up_s.bytes.data()), I, H,
                          gate_up_out + I * H);
  DequantFp8ChannelToBf16(ex.down_w.bytes.data(),
                          reinterpret_cast<const uint16_t*>(ex.down_s.bytes.data()), H, I,
                          down_out);
}

// Host BF16 cache + optional device upload. H2D is async on d.q.
// VT_GEMMA4_EXPERT_VRAM_MB: unset/0 = device expert LRU off; N>0 = N MiB fill-only
// budget (evict only with VT_GEMMA4_EXPERT_EVICT=1). Free VRAM probed via
// Backend::DeviceMemoryInfo when admitting new experts.
namespace {
struct DevExpertLru {
  struct Slot {
    const Gemma4Fp8ExpertMats* ex = nullptr;
    void* gu = nullptr;
    void* dn = nullptr;
    void* fp8_gu = nullptr;
    void* fp8_dn = nullptr;
    void* s_gu = nullptr;
    void* s_dn = nullptr;
    size_t bytes = 0;
    uint64_t tick = 0;
  };
  std::vector<Slot> slots;
  size_t used = 0;
  size_t budget = 0;
  bool budget_set = false;
  uint64_t tick = 1;
  int dev = -1;

  size_t BudgetBytes() {
    if (budget_set) return budget;
    // unset → 2048 MiB fill-only cache (no eviction by default). "0" → off.
    // N>0 → N MiB. Eviction opt-in: VT_GEMMA4_EXPERT_EVICT=1.
    if (const char* e = std::getenv("VT_GEMMA4_EXPERT_VRAM_MB")) {
      const long v = std::strtol(e, nullptr, 10);
      if (v == 0) {
        budget = 0;
        budget_set = true;
        return 0;
      }
      if (v > 0) {
        budget = static_cast<size_t>(v) * 1024ull * 1024ull;
        budget_set = true;
        return budget;
      }
    }
    budget = 2048ull * 1024ull * 1024ull;  // fill-only default
    budget_set = true;
    return budget;
  }

  bool Enabled() { return BudgetBytes() > 0; }

  // Free VRAM via Backend::DeviceMemoryInfo. No HIP in this TU.
  //
  // ROCm AND CUDA answer this today. `CudaBackend::DeviceMemoryInfo` exists
  // (`src/vt/cuda/cuda_backend.cu:93`, added by PERF-QWEN35-STAGE-WEIGHTS), so
  // `MakeRoom` below now ADMITS on a CUDA device where it used to refuse, and
  // this cache is live there rather than dead. The text here said the opposite
  // until #2623 re-read it; that claim was true when it was written and the
  // CUDA override falsified it without touching this file. Two device-expert
  // upload paths depended on the dead premise for their safety, which is why
  // both now carry an explicit op-availability guard instead.
  //
  // The refuse-on-unknown polarity is unchanged and still correct: an Alloc
  // without headroom has hung hipMalloc, so a backend that cannot answer must
  // not be admitted.
  static bool FreeBytes(Dev d, size_t* free_out) {
    *free_out = 0;
    size_t free_b = 0, tot_b = 0;
    if (!d.b.DeviceMemoryInfo(&free_b, &tot_b)) return false;
    *free_out = free_b;
    return true;
  }

  void EvictOne(Dev d) {
    if (slots.empty()) return;
    // CRITICAL: async H2D/GEMM may still reference the victim. hipFree without
    // a stream barrier races the compute stream and has been observed as a
    // permanent kfd_wait hang (GPU idle, prefill done, no decode tokens).
    d.b.Synchronize(d.q);
    size_t victim = 0;
    for (size_t i = 1; i < slots.size(); ++i)
      if (slots[i].tick < slots[victim].tick) victim = i;
    Slot s = slots[victim];
    if (s.gu) d.b.Free(s.gu);
    if (s.dn) d.b.Free(s.dn);
    if (s.fp8_gu) d.b.Free(s.fp8_gu);
    if (s.fp8_dn) d.b.Free(s.fp8_dn);
    if (s.s_gu) d.b.Free(s.s_gu);
    if (s.s_dn) d.b.Free(s.s_dn);
    if (s.ex) {
      s.ex->dev_gu = nullptr;
      s.ex->dev_dn = nullptr;
      s.ex->dev_fp8_gu = nullptr;
      s.ex->dev_fp8_dn = nullptr;
      s.ex->dev_s_gu = nullptr;
      s.ex->dev_s_dn = nullptr;
    }
    used = used >= s.bytes ? used - s.bytes : 0;
    slots.erase(slots.begin() + static_cast<std::ptrdiff_t>(victim));
  }

  // Evict until bookkeeping budget AND free VRAM (if knowable) can take `need`.
  // DEFAULT: no hipFree eviction — ROCm hangs in kfd_wait when we free under
  // load (hoist/pollution). Fill until full, then caller falls back to host H2D.
  // Opt-in eviction: VT_GEMMA4_EXPERT_EVICT=1.
  bool MakeRoom(Dev d, size_t need) {
    const size_t bud = BudgetBytes();
    if (bud == 0) return false;
    static const bool allow_evict = [] {
      const char* e = std::getenv("VT_GEMMA4_EXPERT_EVICT");
      return e && e[0] == '1';
    }();
    constexpr size_t kHeadroom = 1536ull << 20;  // 1.5 GiB after expert — hipMalloc hung at 512MiB
    constexpr size_t kMaxSlots = 24;             // hard cap; further experts stay host
    if (slots.size() >= kMaxSlots) return false;
    if (allow_evict) {
      while (used + need > bud && !slots.empty()) EvictOne(d);
    }
    if (used + need > bud) return false;
    size_t free_b = 0;
    // Refuse device upload if free VRAM unknown — Alloc-without-headroom has
    // hung hipMalloc (hoist start without hoist-done under pollution).
    if (!FreeBytes(d, &free_b)) return false;
    if (allow_evict) {
      int guard = 0;
      while (free_b < need + kHeadroom && !slots.empty() && guard++ < 256) {
        EvictOne(d);
        if (!FreeBytes(d, &free_b)) return false;
      }
    }
    return free_b >= need + kHeadroom;
  }

  void Note(const Gemma4Fp8ExpertMats* ex, void* gu, void* dn, size_t bytes, Dev d,
            void* fp8_gu = nullptr, void* fp8_dn = nullptr, void* s_gu = nullptr,
            void* s_dn = nullptr) {
    if (dev != d.q.device.index) {
      // Device change: drop bookkeeping only (buffers owned by prior device).
      slots.clear();
      used = 0;
      dev = d.q.device.index;
    }
    const size_t bud = BudgetBytes();
    if (used + bytes > bud) return;  // no eviction — drop tracking if over
    slots.push_back(Slot{ex, gu, dn, fp8_gu, fp8_dn, s_gu, s_dn, bytes, tick++});
    used += bytes;
  }

  void Touch(const Gemma4Fp8ExpertMats* ex) {
    for (auto& s : slots) {
      if (s.ex == ex) {
        s.tick = tick++;
        return;
      }
    }
  }
};

DevExpertLru& ExpertLru() {
  static DevExpertLru lru;
  return lru;
}
}  // namespace

bool EnsureGemma4Fp8ExpertOnDevice(Dev d, const Gemma4Fp8ExpertMats& ex, int64_t I,
                                   int64_t H) {
  // Refuse BEFORE the upload on a device whose down-projection GEMM does not
  // exist. Returning true here is a PROMISE that the caller may run the
  // device-resident arm, and every caller that takes that promise ends in
  // `vt::MatmulBTAlphaBeta` — `ExpertGeGLUDeviceAccum` (:76-93) and
  // `ExpertGeGLUTopKFusedGelu` (:181-234) both do. That call has exactly one
  // implementation in the tree, `rocm::MatmulBTAlphaBetaRocm`, so off ROCm it
  // throws (issue #1205). The upload's own `try`/`catch (...)` below does NOT
  // cover the compute, so without this line the exception leaves the decode step
  // instead of degrading: the `else` arms at the call sites already fall back to
  // `EnsureGemma4Fp8ExpertCached` + `ExpertGeGLUHost`, which is slower and
  // rounds twice more, but answers.
  //
  // It is latent rather than live today only because `MakeRoom` needs
  // `Backend::DeviceMemoryInfo`, which only ROCm overrides. #1126 step 1 is
  // exactly the change that adds the CUDA override, which is why the refusal has
  // to be here before it lands and not after.
  //
  // Keyed on whether the arm EXISTS, not on a device name or a build macro:
  // `vt::HasMatmulBTAlphaBeta` is the same predicate the dispatch itself uses, so
  // writing the CUDA kernel wakes this path with no edit here, and on ROCm the
  // answer is true and nothing about this function changes.
  if (!vt::HasMatmulBTAlphaBeta(d.q)) return false;
  // When device LRU disabled, do NOT host-cache-dequant here — that path was
  // unbounded (every expert forever) and OOM'd the 30G host (~27G RSS) under pollution.
  if (!ExpertLru().Enabled()) return false;
  if (ex.dev_gu != nullptr && ex.dev_dn != nullptr) {
    ExpertLru().Touch(&ex);
    return true;
  }
  EnsureGemma4Fp8ExpertCached(ex, I, H);
  const size_t gu_b = static_cast<size_t>(2 * I * H) * sizeof(uint16_t);
  const size_t dn_b = static_cast<size_t>(H * I) * sizeof(uint16_t);
  const size_t total = gu_b + dn_b;
  void* gu = nullptr;
  void* dn = nullptr;
  try {
    auto& lru = ExpertLru();
    if (!lru.MakeRoom(d, total)) return false;
    gu = d.b.Alloc(gu_b);
    dn = d.b.Alloc(dn_b);
    d.b.Copy(d.q, gu, ex.cached_gu.data(), gu_b);
    d.b.Copy(d.q, dn, ex.cached_dn.data(), dn_b);
    // Ensure H2D lands before any later free/evict on another admission path.
    d.b.Synchronize(d.q);
    ex.dev_gu = gu;
    ex.dev_dn = dn;
    lru.Note(&ex, gu, dn, total, d);
    return true;
  } catch (...) {
    if (gu) d.b.Free(gu);
    if (dn) d.b.Free(dn);
    static std::atomic<int> fails{0};
    const int n = fails.fetch_add(1) + 1;
    if (n == 1 || n % 64 == 0)
      std::fprintf(stderr, "gemma4 moe: device expert upload fail #%d (falling back to H2D)\n",
                   n);
    return false;
  }
}

// Upload FP8 weights + channel scales (no BF16 dequant). Half weight VRAM vs BF16 path.
bool EnsureGemma4Fp8NativeOnDevice(Dev d, const Gemma4Fp8ExpertMats& ex, int64_t I, int64_t H) {
  // Refuse BEFORE the upload on a device that cannot run the arm this `true`
  // promises, exactly as `EnsureGemma4Fp8ExpertOnDevice` does above (#2623).
  // This is the DEFAULT arm -- `VT_GEMMA4_FP8_NATIVE` unset returns true at
  // :1071-1077 -- so it is reached before the guarded BF16 sibling is.
  //
  // THREE TERMS, NOT THE SIBLING'S ONE, because this promise commits the caller
  // to three different ops and the sibling's predicate names only one of them.
  // Every consumer of the four slots this function fills ends in an op whose
  // sole arm is ROCm's: `ExpertGeGLUFp8TopKFusedGelu` (:134) calls
  // `vt::MatmulBTFp8Channel` (:165, :176), and `ExpertGeGLUFp8Native` (:96)
  // calls `vt::DequantFp8ChannelBf16` (:118, :120) and `vt::MatmulBTAlphaBeta`
  // (:129). None of those calls sits inside the try/catch below, which wraps
  // only the upload, so without this line the throw leaves the DECODE STEP
  // instead of degrading to the host fallback the `else` arms already hold.
  //
  // TWO DIFFERENT OPS ACTUALLY FIRE, which is why one predicate will not do, and
  // this is measured rather than argued. Three call sites admit through this
  // function -- the `use_dev_expert_lru` prefetch (:1419), the `!fp8_res_peer`
  // branch (:1495) and the per-expert branch (:1620) -- and they do not all land
  // in the same place. Deleting this guard makes
  // `test_gemma4_moe_fp8_native_arm_guard` throw
  // `vt::MatmulBTFp8Channel: ROCm-only in this build`, through
  // `ExpertGeGLUFp8TopKFusedGelu` after `vt::ExpertGeGLUFp8TopKM1` declines by
  // returning false. Deleting it AND disabling :1419 and :1495 makes the same
  // test throw `vt::DequantFp8ChannelBf16` instead, through
  // `ExpertGeGLUFp8Native` at :1620. Neither of those two ops is the one
  // `HasMatmulBTAlphaBeta` names, so the sibling's single predicate would have
  // guarded against the op this path reaches LAST and missed both of the ops it
  // actually reaches.
  //
  // WHAT THE GATE DOES NOT PROVE, stated because a green mutation is not a
  // silent one. All three predicates are the SAME EXPRESSION today
  // (`VLLM_CPP_HIP` and `kROCM`), so any one term alone already refuses on every
  // device this suite can run: deleting the first term is GREEN, and keeping
  // only the first term is GREEN. The three terms are therefore justified by the
  // ops this function commits to and NOT by a mutation that separates them, and
  // nothing here will notice if a future device gains one kernel of the three.
  // Recorded under `## Owed` in .agents/specs/gemma4-fp8-native-arm-guard.md.
  //
  // Each term asks whether the arm EXISTS rather than testing a device name or
  // a build macro, and each op dispatches on its own predicate
  // (`src/vt/fused_ops.cpp`), so a predicate cannot drift from the function it
  // describes and writing all three kernels for a new device wakes this arm with
  // no edit here. Writing only one of them correctly does NOT wake it.
  if (!vt::HasMatmulBTFp8Channel(d.q) || !vt::HasDequantFp8ChannelBf16(d.q) ||
      !vt::HasMatmulBTAlphaBeta(d.q))
    return false;
  // When device LRU disabled, do NOT host-cache-dequant here.
  if (!ExpertLru().Enabled()) return false;
  if (ex.dev_fp8_gu && ex.dev_fp8_dn && ex.dev_s_gu && ex.dev_s_dn) {
    ExpertLru().Touch(&ex);
    return true;
  }
  VT_CHECK(ex.gate_w.HasHostBytes() && ex.up_w.HasHostBytes() && ex.down_w.HasHostBytes(),
           "fp8 native: missing weights");
  VT_CHECK(ex.gate_s.HasHostBytes() && ex.up_s.HasHostBytes() && ex.down_s.HasHostBytes(),
           "fp8 native: missing scales");
  const size_t gu_b = static_cast<size_t>(2 * I * H);       // u8
  const size_t dn_b = static_cast<size_t>(H * I);           // u8
  const size_t sgu_b = static_cast<size_t>(2 * I) * 2;      // bf16
  const size_t sdn_b = static_cast<size_t>(H) * 2;          // bf16
  const size_t total = gu_b + dn_b + sgu_b + sdn_b;
  void *fgu = nullptr, *fdn = nullptr, *sgu = nullptr, *sdn = nullptr;
  try {
    auto& lru = ExpertLru();
    if (!lru.MakeRoom(d, total)) return false;
    fgu = d.b.Alloc(gu_b);
    fdn = d.b.Alloc(dn_b);
    sgu = d.b.Alloc(sgu_b);
    sdn = d.b.Alloc(sdn_b);
    // Pack gate|up FP8 rows
    d.b.Copy(d.q, fgu, ex.gate_w.bytes.data(), static_cast<size_t>(I * H));
    d.b.Copy(d.q, static_cast<char*>(fgu) + static_cast<size_t>(I * H), ex.up_w.bytes.data(),
             static_cast<size_t>(I * H));
    d.b.Copy(d.q, fdn, ex.down_w.bytes.data(), dn_b);
    d.b.Copy(d.q, sgu, ex.gate_s.bytes.data(), static_cast<size_t>(I) * 2);
    d.b.Copy(d.q, static_cast<char*>(sgu) + static_cast<size_t>(I) * 2, ex.up_s.bytes.data(),
             static_cast<size_t>(I) * 2);
    d.b.Copy(d.q, sdn, ex.down_s.bytes.data(), sdn_b);
    d.b.Synchronize(d.q);
    ex.dev_fp8_gu = fgu;
    ex.dev_fp8_dn = fdn;
    ex.dev_s_gu = sgu;
    ex.dev_s_dn = sdn;
    lru.Note(&ex, nullptr, nullptr, total, d, fgu, fdn, sgu, sdn);
    return true;
  } catch (...) {
    if (fgu) d.b.Free(fgu);
    if (fdn) d.b.Free(fdn);
    if (sgu) d.b.Free(sgu);
    if (sdn) d.b.Free(sdn);
    return false;
  }
}

void DequantGemma4Fp8ExpertToBf16(const Gemma4Fp8ExpertMats& ex, int64_t I, int64_t H,
                                  uint16_t* gate_up_out, uint16_t* down_out) {
  VT_CHECK(gate_up_out && down_out, "fp8 expert dequant null out");
  EnsureGemma4Fp8ExpertCached(ex, I, H);
  std::memcpy(gate_up_out, ex.cached_gu.data(), ex.cached_gu.size() * sizeof(uint16_t));
  std::memcpy(down_out, ex.cached_dn.data(), ex.cached_dn.size() * sizeof(uint16_t));
}

Gemma4MoeScratch RunGemma4Moe(vt::Queue& q, const Gemma4MoeLayerWeights& moe,
                              const vt::Tensor& router_in, const vt::Tensor& expert_in,
                              int64_t T, int64_t H, float rms_eps) {
  using dense_attn::DBuf;
  using dense_attn::Dev;
  using dense_attn::ResidentWeight;
  using vt::DType;
  using vt::Tensor;

  VT_CHECK(moe.enabled && !moe.experts.Empty(), "gemma4 moe: disabled");
  VT_CHECK(router_in.shape[0] == T && router_in.shape[1] == H, "gemma4 moe: router_in");
  VT_CHECK(expert_in.shape[0] == T && expert_in.shape[1] == H, "gemma4 moe: expert_in");
  const int64_t E = moe.experts.num_experts;
  const int64_t I = moe.experts.intermediate;
  const int top_k = moe.top_k;
  VT_CHECK(E > 0 && I > 0 && top_k > 0 && top_k <= E, "gemma4 moe: dims");
  VT_CHECK(moe.experts.hidden == H, "gemma4 moe: H mismatch");

  Dev d{vt::GetBackend(q.device.type), q};
  const vt::RmsNormArgs plain{rms_eps, false};
  const int compute_dev = q.device.index;

  static const bool profile = [] {
    const char* e = std::getenv("VT_GEMMA4_PROFILE");
    return e && e[0] == '1';
  }();
  using clock = std::chrono::steady_clock;
  const auto t_all0 = profile ? clock::now() : clock::time_point{};

  // T=1 graph-stable router temps (capture bakes pointers).
  struct RouterTls {
    int dev = -1;
    int64_t T = 0, H = 0, E = 0, K = 0;
    std::optional<DBuf> rn, logits, rw, ri;
  };
  static thread_local RouterTls rt;
  if (rt.dev != compute_dev || rt.T != T || rt.H != H || rt.E != E || rt.K != top_k ||
      !rt.rn || !rt.logits || !rt.rw || !rt.ri) {
    rt.rn.emplace(d, DType::kBF16, std::vector<int64_t>{T, H});
    rt.logits.emplace(d, DType::kF32, std::vector<int64_t>{T, E});
    rt.rw.emplace(d, DType::kF32, std::vector<int64_t>{T, top_k});
    rt.ri.emplace(d, DType::kI32, std::vector<int64_t>{T, top_k});
    rt.dev = compute_dev;
    rt.T = T;
    rt.H = H;
    rt.E = E;
    rt.K = top_k;
  }
  DBuf& rn = *rt.rn;
  // Identity RMS weight (ones) — TLS, upload once (was H2D every layer/token).
  {
    struct OnesTls {
      int dev = -1;
      int64_t H = 0;
      std::optional<DBuf> w;
    };
    static thread_local OnesTls ot;
    if (ot.dev != compute_dev || ot.H != H || !ot.w) {
      std::vector<uint16_t> ones(static_cast<size_t>(H), vt::F32ToBF16(1.f));
      ot.w.emplace(d, DType::kBF16, std::vector<int64_t>{H}, ones.data());
      ot.dev = compute_dev;
      ot.H = H;
    }
    vt::RmsNorm(d.q, rn.t(), router_in, ot.w->t(), plain);
  }

  const OwnedTensor& rproj =
      !moe.router_proj_fused.Empty() ? moe.router_proj_fused : moe.router_proj;
  VT_CHECK(rproj.HasHostBytes() && rproj.nk && rproj.shape[0] == E && rproj.shape[1] == H,
           "gemma4 moe: router proj");
  Tensor wp = ResidentWeight(d, rproj);
  DBuf& logits = *rt.logits;
  vt::MatmulBT(d.q, logits.t(), rn.t(), wp);

  // Device router top-k (softmax + greedy). Only D2H [T,K] weights/indices.
  DBuf& rw = *rt.rw;
  DBuf& ri = *rt.ri;
  vt::MoeRouterTopKArgs rargs;
  rargs.top_k = top_k;
  rargs.renormalize = true;
  vt::MoeRouterTopK(d.q, rw.t(), ri.t(), logits.t(), rargs);


  if (!moe.experts.compact.empty()) {
    VT_CHECK(q.device.type == vt::DeviceType::kCPU,
             "gemma4 CT: compact integer experts are implemented for CPU only");
    VT_CHECK(moe.experts.compact.size() == static_cast<size_t>(E),
             "gemma4 CT: compact expert count mismatch");
    DBuf acc(d, DType::kBF16, {T, H});
    DBuf y(d, DType::kBF16, {1, H});
    auto* output = static_cast<uint16_t*>(acc.ptr());
    const auto* ids = static_cast<const int32_t*>(ri.ptr());
    const auto* route = static_cast<const float*>(rw.ptr());
    const auto* scales = moe.per_expert_scale.bytes.data();
    for (int64_t t = 0; t < T; ++t) {
      Tensor x = expert_in.Slice(0, t, t + 1);
      // Routed expert reduction accumulates in F32 before its BF16 output store.
      std::vector<float> sum(static_cast<size_t>(H), 0.0f);
      for (int i = 0; i < top_k; ++i) {
        const int32_t e = ids[t * top_k + i];
        VT_CHECK(e >= 0 && e < E, "gemma4 CT: routed expert index out of range");
        const auto& ew = moe.experts.compact[static_cast<size_t>(e)];
        layers::DenseShardMlpGateUpGeluMethod method(&ew.gate_up, I);
        DBuf act = method.Apply(d, x);
        Tensor down = ResidentWeight(d, ew.down);
        vt::MatmulBT(d.q, y.t(), act.t(), down);
        const float scale = vt::BF16ToF32(vt::LoadUnaligned<uint16_t>(scales + e * 2));
        const float weight = route[t * top_k + i] * scale;
        const auto* result = static_cast<const uint16_t*>(y.ptr());
        for (int64_t h = 0; h < H; ++h)
          sum[static_cast<size_t>(h)] += weight * vt::BF16ToF32(result[h]);
      }
      for (int64_t h = 0; h < H; ++h)
        output[t * H + h] = vt::F32ToBF16(sum[static_cast<size_t>(h)]);
    }
    Gemma4MoeScratch result;
    result.tensor = acc.t();
    result.storage = acc.ReleaseShared();
    return result;
  }

  const auto& ex = moe.experts;
  const int64_t gu_stride = 2 * I * H;
  const int64_t dn_stride = H * I;
  const bool same_dev =
      ex.gate_up_dev != nullptr && ex.down_dev != nullptr && ex.dev_id == compute_dev;
  const bool fp8_res =
      ex.fp8_native_resident && !ex.fp8.empty() && ex.fp8[0].dev_fp8_gu != nullptr &&
      ex.fp8_gu_base != nullptr && ex.dev_id >= 0;
  const bool fp8_res_same = fp8_res && ex.dev_id == compute_dev;
  const bool fp8_res_peer = fp8_res && ex.dev_id != compute_dev;

  // Device-indexed FP8 MoE for T=1..min(MAX, batch_min-1). No router D2H.
  // T=1: hipGraph-stable TLS acc. T>1: owned [T,H] + per-token existing helpers.
  // VT_GEMMA4_DECODE_INDEXED_MAX_T: default 63; =1 → T=1 only; clamp [1,63].
  const int64_t indexed_max_t = Gemma4DecodeIndexedMaxT();
  const bool indexed_eligible = Gemma4IndexedOkT(T, indexed_max_t, top_k, fp8_res);
  auto emit_moe_dispatch = [&](const char* path, bool fallthrough) {
    if (!profile) return;
    std::fprintf(stderr,
                 "gemma4 moe dispatch: T=%lld indexed_max_t=%lld path=%s eligible=%d top_k=%d "
                 "compute_dev=%d expert_dev=%d fallthrough=%d\n",
                 static_cast<long long>(T), static_cast<long long>(indexed_max_t), path,
                 indexed_eligible ? 1 : 0, top_k, compute_dev, ex.dev_id, fallthrough ? 1 : 0);
    std::fflush(stderr);
  };
  if (Gemma4IndexedOkT(T, indexed_max_t, top_k, fp8_res)) {
    // per-expert scale on device (once per layer/E; apply each token — helper is G-wide).
    struct EscTls {
      int dev = -1;
      int64_t E = 0;
      const void* host_key = nullptr;
      std::optional<DBuf> sc;
    };
    static thread_local EscTls esc;
    float* escale_ptr = nullptr;
    if (moe.per_expert_scale.HasHostBytes()) {
      const void* hk = moe.per_expert_scale.bytes.data();
      if (esc.dev != compute_dev || esc.E != E || esc.host_key != hk || !esc.sc) {
        std::vector<float> hs(static_cast<size_t>(E), 1.f);
        const auto* pe = reinterpret_cast<const uint16_t*>(moe.per_expert_scale.bytes.data());
        for (int64_t e = 0; e < E; ++e) hs[static_cast<size_t>(e)] = vt::BF16ToF32(pe[e]);
        esc.sc.emplace(d, DType::kF32, std::vector<int64_t>{E}, hs.data());
        esc.dev = compute_dev;
        esc.E = E;
        esc.host_key = hk;
      }
      escale_ptr = static_cast<float*>(esc.sc->ptr());
    }

    // Never mutate router `rw` in place — fallback must still see unscaled weights.
    // T=1: TLS-stable copy (hipGraph bakes the pointer). T>1: per-call pooled DBuf.
    struct RwIdxTls {
      int dev = -1;
      int64_t n = 0;  // T*top_k
      std::optional<DBuf> buf;
    };
    static thread_local RwIdxTls rwt;
    std::optional<DBuf> rw_idx_owned;
    const float* helper_rw = static_cast<const float*>(rw.ptr());
    if (escale_ptr) {
      const int64_t n = T * top_k;
      const bool t1_tls = Gemma4IndexedScratchKindFor(T) == Gemma4IndexedScratchKind::TlsT1;
      DBuf* scaled = nullptr;
      if (t1_tls) {
        if (rwt.dev != compute_dev || rwt.n != n || !rwt.buf) {
          rwt.buf.emplace(d, DType::kF32, std::vector<int64_t>{T, top_k});
          rwt.dev = compute_dev;
          rwt.n = n;
        }
        scaled = &*rwt.buf;
      } else {
        rw_idx_owned.emplace(d, DType::kF32, std::vector<int64_t>{T, top_k});
        scaled = &*rw_idx_owned;
      }
      d.b.Copy(d.q, scaled->ptr(), rw.ptr(), static_cast<size_t>(n) * sizeof(float));
      for (int64_t t = 0; t < T; ++t) {
        const auto off = Gemma4IndexedTokenOffsets(t, H, top_k);
        vt::ApplyExpertScaleRw(d.q, static_cast<float*>(scaled->ptr()) + off.route,
                               static_cast<const int32_t*>(ri.ptr()) + off.route, escale_ptr, top_k,
                               static_cast<int>(E));
      }
      helper_rw = static_cast<const float*>(scaled->ptr());
    }

    const auto indexed_arm = Gemma4IndexedSelectArm(fp8_res_same, fp8_res_peer);
    auto run_one = [&](void* yout, const void* xin, const int32_t* ri_t, const float* rw_t) -> bool {
      const auto args = Gemma4IndexedPackArgs(yout, xin, ri_t, rw_t);
      return Gemma4IndexedRunSelectedArm(
          indexed_arm,
          [&] {
            return vt::ExpertGeGLUFp8TopKIndexed(d.q, args.y, args.x, ex.fp8_gu_base, ex.fp8_dn_base,
                                                 ex.fp8_sgu_base, ex.fp8_sdn_base, args.ri, args.rw,
                                                 top_k, static_cast<int>(I), static_cast<int>(H));
          },
          [&] {
            return RunGemma4Fp8TopKIndexedOnExpertDevice(
                d.q, ex.dev_id, args.y, args.x, ex.fp8_gu_base, ex.fp8_dn_base, ex.fp8_sgu_base,
                ex.fp8_sdn_base, args.ri, args.rw, top_k, static_cast<int>(I), static_cast<int>(H));
          });
    };
    auto restore_compute = [] {};
    bool indexed_retired = false;
    auto retire_indexed = [&]() -> bool {
      if (indexed_retired) return true;
      bool ok = true;
      if (fp8_res_peer) {
        ok = RetireGemma4Fp8TopKIndexedPeer(d.q, ex.dev_id);
      } else {
        d.b.Synchronize(d.q);
      }
      indexed_retired = ok;
      return ok;
    };

    if (Gemma4IndexedScratchKindFor(T) == Gemma4IndexedScratchKind::TlsT1) {
      // Stable T=1 acc for hipGraph (do not pool-Release). rw_idx is RwIdxTls.
      struct AccFastTls {
        int dev = -1;
        int64_t H = 0;
        std::optional<DBuf> acc;
      };
      static thread_local AccFastTls aft;
      if (aft.dev != d.q.device.index || aft.H != H || !aft.acc) {
        aft.acc.emplace(d, DType::kBF16, std::vector<int64_t>{1, H});
        aft.dev = d.q.device.index;
        aft.H = H;
      }
      DBuf& acc_fast = *aft.acc;
      const auto disp = Gemma4IndexedDispatchTokens(
          1, H, top_k, fp8_res_peer, static_cast<uint16_t*>(acc_fast.ptr()),
          static_cast<const uint16_t*>(expert_in.data), static_cast<const int32_t*>(ri.ptr()),
          helper_rw, [&](const Gemma4IndexedCall<uint16_t, uint16_t>& c) {
            return run_one(c.y, c.x, c.ri, c.rw);
          },
          restore_compute);
      if (disp.ok) {
        emit_moe_dispatch(indexed_arm == Gemma4IndexedArm::Peer ? "indexed_peer" : "indexed_same",
                          /*fallthrough=*/false);
        const auto t_router1 = profile ? clock::now() : clock::time_point{};
        Gemma4MoeScratch r;
        r.tensor = acc_fast.t();
        r.storage = std::shared_ptr<void>(acc_fast.ptr(), [](void*) {});
        if (profile) {
          d.b.Synchronize(d.q);
          const auto t_all1 = clock::now();
          static std::atomic<uint64_t> ncalls{0};
          static std::atomic<uint64_t> us_router{0};
          static std::atomic<uint64_t> us_total{0};
          const auto ur =
              std::chrono::duration_cast<std::chrono::microseconds>(t_router1 - t_all0).count();
          const auto ut =
              std::chrono::duration_cast<std::chrono::microseconds>(t_all1 - t_all0).count();
          us_router.fetch_add(static_cast<uint64_t>(ur), std::memory_order_relaxed);
          us_total.fetch_add(static_cast<uint64_t>(ut), std::memory_order_relaxed);
          const uint64_t c = ncalls.fetch_add(1, std::memory_order_relaxed) + 1;
          if (c == 1 || c % 64 == 0) {
            const uint64_t tr = us_router.load(std::memory_order_relaxed);
            const uint64_t tt = us_total.load(std::memory_order_relaxed);
            std::fprintf(stderr,
                         "gemma4 moe profile: calls=%llu router_us/call=%.1f expert+rest_us/call=%.1f "
                         "total_us/call=%.1f (router%%=%.0f) [indexed-device]\n",
                         static_cast<unsigned long long>(c), static_cast<double>(tr) / c,
                         static_cast<double>(tt - tr) / c, static_cast<double>(tt) / c,
                         tt ? 100.0 * static_cast<double>(tr) / static_cast<double>(tt) : 0.0);
          }
        }
        return r;
      }
      (void)retire_indexed();  // T=1 TLS acc/rw_idx are not pooled; retire before leaving arm
    } else {
      std::optional<DBuf> acc_idx;
      acc_idx.emplace(d, DType::kBF16, std::vector<int64_t>{T, H});
      auto* x_base = static_cast<const uint16_t*>(expert_in.data);
      auto* y_base = static_cast<uint16_t*>(acc_idx->ptr());
      auto* ri_base = static_cast<const int32_t*>(ri.ptr());
      const Gemma4IndexedScratchChoice scratch_choice{Gemma4IndexedScratchKindFor(T), y_base,
                                                      T * H};
      if (!Gemma4IndexedScratchValidForT(scratch_choice, T, H)) {
        if (!retire_indexed()) (void)acc_idx->Release();  // quarantine
        // fall through with acc_idx still in scope until this block ends
      } else {
        const auto disp = Gemma4IndexedDispatchTokens(
            T, H, top_k, fp8_res_peer, y_base, x_base, ri_base, helper_rw,
            [&](const Gemma4IndexedCall<uint16_t, uint16_t>& c) {
              return run_one(c.y, c.x, c.ri, c.rw);
            },
            restore_compute);
        if (disp.ok) {
          emit_moe_dispatch(indexed_arm == Gemma4IndexedArm::Peer ? "indexed_peer" : "indexed_same",
                            /*fallthrough=*/false);
          Gemma4MoeScratch r;
          r.tensor = acc_idx->t();
          r.storage = acc_idx->ReleaseShared();
          return r;
        }
        // retire-before-acc_idx-dtor: still lexically inside acc_idx scope
        if (!retire_indexed()) (void)acc_idx->Release();  // quarantine, do not Put
      }
    }
    if (!indexed_retired) (void)retire_indexed();
    if (!indexed_retired && rw_idx_owned) (void)rw_idx_owned->Release();
    // fall through to legacy host-gather path
  }

  emit_moe_dispatch("legacy", /*fallthrough=*/indexed_eligible);

  std::vector<float> hw(static_cast<size_t>(T * top_k));
  std::vector<int32_t> hi(static_cast<size_t>(T * top_k));
  d.b.Copy(d.q, hw.data(), rw.ptr(), hw.size() * sizeof(float));
  d.b.Copy(d.q, hi.data(), ri.ptr(), hi.size() * sizeof(int32_t));
  d.b.Synchronize(d.q);

  const auto t_router1 = profile ? clock::now() : clock::time_point{};

  std::vector<float> hscale(static_cast<size_t>(E), 1.f);
  if (moe.per_expert_scale.HasHostBytes()) {
    const auto* pe = reinterpret_cast<const uint16_t*>(moe.per_expert_scale.bytes.data());
    for (int64_t e = 0; e < E; ++e) hscale[static_cast<size_t>(e)] = vt::BF16ToF32(pe[e]);
  }
  // Apply per-expert scale to selected weights (once; indexed scratch is a copy).
  Gemma4ApplyHostExpertScaleOnce(hw.data(), hi.data(), hscale.data(), T, top_k, E,
                                 /*already_scaled=*/false);
  const bool need_peer_sc = (!same_dev && ex.gate_up_dev && ex.down_dev && ex.dev_id >= 0) ||
                            fp8_res_peer;

  const auto* gu_host = ex.gate_up.Empty()
                            ? nullptr
                            : reinterpret_cast<const uint16_t*>(ex.gate_up.bytes.data());
  const auto* dn_host =
      ex.down.Empty() ? nullptr : reinterpret_cast<const uint16_t*>(ex.down.bytes.data());

  // Graph-stable decode scratch: do not pool-Release acc when T==1 (hipGraph bakes ptr).
  struct AccTls {
    int dev = -1;
    int64_t T = 0, H = 0;
    std::optional<DBuf> acc;
  };
  static thread_local AccTls acc_tls;
  DBuf* acc_ptr = nullptr;
  std::optional<DBuf> acc_owned;
  if (T == 1) {
    if (acc_tls.dev != d.q.device.index || acc_tls.T != T || acc_tls.H != H || !acc_tls.acc) {
      acc_tls.acc.emplace(d, DType::kBF16, std::vector<int64_t>{T, H});
      acc_tls.dev = d.q.device.index;
      acc_tls.T = T;
      acc_tls.H = H;
    }
    acc_ptr = &*acc_tls.acc;
  } else {
    acc_owned.emplace(d, DType::kBF16, std::vector<int64_t>{T, H});
    acc_ptr = &*acc_owned;
  }
  DBuf& acc = *acc_ptr;
  acc.Zero(d);

  // Reuse MoE decode scratch across layers (30 layers × every token was thrashing the pool).
  struct MoeTlsScratch {
    int dev = -1;
    int64_t I = 0, H = 0;
    std::unique_ptr<ExpertScratch> esc;
    std::optional<DBuf> xin, ysum, y, ysc;
    std::optional<DBuf> gu_sc, dn_sc;
    std::optional<DBuf> fp8_gu_sc, fp8_dn_sc, fp8_sgu_sc, fp8_sdn_sc;
    bool have_peer = false;
    bool have_fp8_peer = false;
    std::vector<uint16_t> gu_tmp, dn_tmp;
  };
  static thread_local MoeTlsScratch tls;
  if (tls.dev != compute_dev || tls.I != I || tls.H != H) {
    tls.esc = std::make_unique<ExpertScratch>(d, /*T=*/1, I, H);
    tls.xin.emplace(d, DType::kBF16, std::vector<int64_t>{1, H});
    tls.ysum.emplace(d, DType::kBF16, std::vector<int64_t>{1, H});
    tls.y.emplace(d, DType::kBF16, std::vector<int64_t>{1, H});
    tls.ysc.emplace(d, DType::kBF16, std::vector<int64_t>{1, H});
    tls.gu_sc.reset();
    tls.dn_sc.reset();
    tls.fp8_gu_sc.reset();
    tls.fp8_dn_sc.reset();
    tls.fp8_sgu_sc.reset();
    tls.fp8_sdn_sc.reset();
    tls.have_peer = false;
    tls.have_fp8_peer = false;
    tls.gu_tmp.clear();
    tls.dn_tmp.clear();
    tls.dev = compute_dev;
    tls.I = I;
    tls.H = H;
  }
  ExpertScratch& esc = *tls.esc;
  DBuf& xin = *tls.xin;
  DBuf& ysum = *tls.ysum;
  DBuf& y = *tls.y;
  DBuf& ysc = *tls.ysc;
  if (need_peer_sc && !fp8_res_peer && !tls.have_peer) {
    tls.gu_sc.emplace(d, DType::kBF16, std::vector<int64_t>{2 * I, H});
    tls.dn_sc.emplace(d, DType::kBF16, std::vector<int64_t>{H, I});
    tls.have_peer = true;
  }
  if (fp8_res_peer && !tls.have_fp8_peer) {
    tls.fp8_gu_sc.emplace(d, DType::kI8, std::vector<int64_t>{2 * I * H});
    tls.fp8_dn_sc.emplace(d, DType::kI8, std::vector<int64_t>{H * I});
    tls.fp8_sgu_sc.emplace(d, DType::kBF16, std::vector<int64_t>{2 * I});
    tls.fp8_sdn_sc.emplace(d, DType::kBF16, std::vector<int64_t>{H});
    tls.have_fp8_peer = true;
  }
  std::optional<DBuf>& gu_sc = tls.gu_sc;
  std::optional<DBuf>& dn_sc = tls.dn_sc;
  if (ex.is_fp8 && tls.gu_tmp.size() != static_cast<size_t>(gu_stride)) {
    tls.gu_tmp.resize(static_cast<size_t>(gu_stride));
    tls.dn_tmp.resize(static_cast<size_t>(dn_stride));
  }
  std::vector<uint16_t>& gu_tmp = tls.gu_tmp;
  std::vector<uint16_t>& dn_tmp = tls.dn_tmp;
  static const bool host_axpy = [] {
    const char* e = std::getenv("VT_GEMMA4_HOST_AXPY");
    return e && e[0] == '1';
  }();
  static const bool batch_experts = [] {
    const char* e = std::getenv("VT_GEMMA4_BATCH_EXPERTS");
    return e && e[0] == '1';
  }();
  static const bool fp8_native = [] {
    const char* e = std::getenv("VT_GEMMA4_FP8_NATIVE");
    // Default ON: use fused FP8 expert kernels when packs/LRU available.
    // =0 forces BF16 device/host paths only.
    if (e == nullptr) return true;
    return e[0] == '1';
  }();
  static const bool custom_expert = [] {
    const char* e = std::getenv("VT_GEMMA4_CUSTOM_EXPERT");
    return e && e[0] == '1';
  }();
  std::vector<uint16_t> hsum;
  if (host_axpy) hsum.assign(static_cast<size_t>(H), vt::F32ToBF16(0.f));

  // Device expert cache: DECODE-ONLY (T==1), fill-only. NO bulk hoist — pre-upload
  // of top_k experts hung hipMalloc (hoist without hoist-done). Lazy Ensure in the
  // expert loop only. Prefill stays host. Opt out: EXPERT_VRAM_MB=0.
  static const bool moe_trace = [] {
    const char* e = std::getenv("VT_GEMMA4_LAYER_TRACE");
    return e && e[0] == '2';
  }();
  const bool use_dev_expert_lru =
      (T == 1) && ex.is_fp8 && !same_dev && ex.gate_up_dev == nullptr && ExpertLru().Enabled();
  // Lazy Ensure only (below). Bulk hoist removed — was the pollution hang site.
  (void)moe_trace;

  // Prefill batch MoE: group tokens by expert, one hipBLAS GEMM per expert chunk.
  // Auto ON when BF16 device-resident same-GPU OR native FP8 resident (any GPU —
  // dequant ephemeral → BF16 GEMM). Native FP8 decode stays M=1 fused GEMV;
  // without this, prefill is serial M=1 × T and crawls (~13 tok/s on 40k).
  // Explicit VT_GEMMA4_PREFILL_BATCH_MOE=0/1 overrides.
  // hipBLASLt FP8 W8A8 was microbench'd slower than BF16 GemmEx at all M on
  // gfx1201 — do not route prefill through Lt FP8 (see fp8_lt_prefill_m log).
  static const int prefill_batch_env = [] {
    const char* e = std::getenv("VT_GEMMA4_PREFILL_BATCH_MOE");
    if (e == nullptr) return -1;  // auto
    return (e[0] == '1') ? 1 : 0;
  }();
  // Prefer fused M=1 FP8 for short T (decode + tiny prefills). Batch dequant+GEMM
  // only pays off once enough tokens share experts (lab: T=818 ~3×, T=6k ~6×;
  // T=13 was slower than fused M=1).
  constexpr int64_t kPrefillBatchMinT = kGemma4PrefillBatchMinT;
  const bool prefill_batch_moe =
      (T >= kPrefillBatchMinT) && !host_axpy &&
      ((prefill_batch_env == 1) ||
       (prefill_batch_env < 0 && (same_dev || fp8_res)));
  if (prefill_batch_moe) {
    // One-shot path breadcrumb (always on) so we can prove this arm fired.
    static std::atomic<int> batch_enter{0};
    if (batch_enter.fetch_add(1) < 3) {
      std::fprintf(stderr,
                   "INFO gemma4-moe ENTER prefill-batch T=%lld same_dev=%d fp8_res=%d "
                   "fp8_res_same=%d env=%d\n",
                   static_cast<long long>(T), same_dev ? 1 : 0, fp8_res ? 1 : 0,
                   fp8_res_same ? 1 : 0, prefill_batch_env);
      std::fflush(stderr);
    }
    const vt::Device dev = d.q.device;
    std::vector<std::vector<int32_t>> etok(static_cast<size_t>(E));
    std::vector<std::vector<float>> ewt(static_cast<size_t>(E));
    for (int64_t t = 0; t < T; ++t) {
      for (int k = 0; k < top_k; ++k) {
        const size_t o = static_cast<size_t>(t * top_k + k);
        const int e = static_cast<int>(hi[o]);
        if (e < 0 || e >= static_cast<int>(E)) continue;
        etok[static_cast<size_t>(e)].push_back(static_cast<int32_t>(t));
        ewt[static_cast<size_t>(e)].push_back(hw[o]);
      }
    }
    int64_t max_n = 0;
    for (int64_t e = 0; e < E; ++e) {
      max_n = std::max<int64_t>(max_n, static_cast<int64_t>(etok[static_cast<size_t>(e)].size()));
    }
    if (max_n > 0) {
      // Cap GeGLU M per expert chunk. Larger M cuts launch count on long prefills
      // (lab A/B via VT_GEMMA4_PREFILL_GEMM_M). Cap 8192 for hot experts.
      static const int64_t kMaxGemmM = []() -> int64_t {
        if (const char* e = std::getenv("VT_GEMMA4_PREFILL_GEMM_M")) {
          const long v = std::strtol(e, nullptr, 10);
          if (v >= 16 && v <= 8192) return static_cast<int64_t>(v);
        }
        return 2048;  // 2026-08-10: 512→2048 ~+80 eng @11k vs WMMA baseline
      }();
      // Peer-act: run GeGLU on expert GPU; peer only activations.
      // Default ON — 2026-08-10 A/B @~11k: ~2016 → ~2056 eng (+40) after SharedK-WMMA+GEMM_M=2048.
      // VT_GEMMA4_PREFILL_PEER_ACT=0 to disable (weight PeerCopy path).
      static const bool kPrefillPeerAct = [] {
        const char* e = std::getenv("VT_GEMMA4_PREFILL_PEER_ACT");
        if (e == nullptr) return true;
        return e[0] != '0';
      }();
      const int64_t scratch_n = std::min(max_n, kMaxGemmM);
      ExpertScratch besc(d, scratch_n, I, H);
      DBuf bx(d, DType::kBF16, {scratch_n, H});
      DBuf by(d, DType::kBF16, {scratch_n, H});
      // Device token-id / weight rows for gather+scatter (pure GPU path).
      DBuf d_tids(d, DType::kI32, {scratch_n});
      DBuf d_wts(d, DType::kF32, {scratch_n});
      std::vector<int32_t> h_tids(static_cast<size_t>(scratch_n));
      std::vector<float> h_wts(static_cast<size_t>(scratch_n));
      // Zero MoE output accumulator on device (no host hacc).
      vt::MoeZeroBf16(d.q, acc.ptr(), static_cast<int64_t>(T) * H);
      int experts_run = 0;
      int peer_act_hits = 0;
      bool batch_ok = true;
      for (int64_t e = 0; e < E; ++e) {
              const auto& toks = etok[static_cast<size_t>(e)];
              if (toks.empty()) continue;
              const auto& wts_e = ewt[static_cast<size_t>(e)];
              const int64_t n_all = static_cast<int64_t>(toks.size());

              const uint16_t* gu_p = nullptr;
              const uint16_t* dn_p = nullptr;
              const uint16_t* gu_host_e = nullptr;
              const uint16_t* dn_host_e = nullptr;
              const void* fp8_gu_p = nullptr;
              const void* fp8_dn_p = nullptr;
              const void* s_gu_p = nullptr;
              const void* s_dn_p = nullptr;
              const void* fp8_weight_id = nullptr;
              bool use_device_w = false;
              bool use_fp8_dev = false;
              bool use_peer_act = false;
              if (same_dev) {
                gu_p = static_cast<const uint16_t*>(ex.gate_up_dev) + e * gu_stride;
                dn_p = static_cast<const uint16_t*>(ex.down_dev) + e * dn_stride;
                use_device_w = true;
              } else if (ex.is_fp8) {
                auto& fex = ex.fp8[static_cast<size_t>(e)];
                // Pure GPU: native FP8 packs. Same-device direct; peer-act preferred.
                if (fex.dev_fp8_gu && fex.dev_fp8_dn && fex.dev_s_gu && fex.dev_s_dn) {
                  if (ex.dev_id == compute_dev || ex.dev_id < 0) {
                    fp8_gu_p = fex.dev_fp8_gu;
                    fp8_dn_p = fex.dev_fp8_dn;
                    s_gu_p = fex.dev_s_gu;
                    s_dn_p = fex.dev_s_dn;
                    fp8_weight_id = fex.dev_fp8_gu;
                    use_fp8_dev = true;
                  } else if (kPrefillPeerAct) {
                    // Weights stay on expert_dev; GeGLU runs there (see chunk loop).
                    fp8_gu_p = fex.dev_fp8_gu;
                    fp8_dn_p = fex.dev_fp8_dn;
                    s_gu_p = fex.dev_s_gu;
                    s_dn_p = fex.dev_s_dn;
                    fp8_weight_id = fex.dev_fp8_gu;
                    use_peer_act = true;
                  } else {
                    struct PeerTls {
                      int dev = -1;
                      int64_t I = 0, H = 0;
                      std::optional<DBuf> gu, dn, sgu, sdn;
                    };
                    static thread_local PeerTls pt;
                    if (pt.dev != compute_dev || pt.I != I || pt.H != H) {
                      pt.gu.emplace(d, DType::kI8, std::vector<int64_t>{2 * I * H});
                      pt.dn.emplace(d, DType::kI8, std::vector<int64_t>{H * I});
                      pt.sgu.emplace(d, DType::kBF16, std::vector<int64_t>{2 * I});
                      pt.sdn.emplace(d, DType::kBF16, std::vector<int64_t>{H});
                      pt.dev = compute_dev;
                      pt.I = I;
                      pt.H = H;
                    }
                    if (!PeerCopyGemma4Fp8ExpertSlice(ex.dev_id, fex.dev_fp8_gu, fex.dev_fp8_dn,
                                                     fex.dev_s_gu, fex.dev_s_dn, I, H, compute_dev,
                                                     pt.gu->ptr(), pt.dn->ptr(), pt.sgu->ptr(),
                                                     pt.sdn->ptr())) {
                      batch_ok = false;
                      break;
                    }
                    fp8_gu_p = pt.gu->ptr();
                    fp8_dn_p = pt.dn->ptr();
                    s_gu_p = pt.sgu->ptr();
                    s_dn_p = pt.sdn->ptr();
                    fp8_weight_id = fex.dev_fp8_gu;
                    use_fp8_dev = true;
                  }
                } else if (use_dev_expert_lru && !fp8_native &&
                           EnsureGemma4Fp8ExpertOnDevice(d, fex, I, H)) {
                  gu_p = static_cast<const uint16_t*>(fex.dev_gu);
                  dn_p = static_cast<const uint16_t*>(fex.dev_dn);
                  use_device_w = true;
                } else {
                  batch_ok = false;
                  break;
                }
              } else if (gu_host && dn_host) {
                gu_host_e = gu_host + e * gu_stride;
                dn_host_e = dn_host + e * dn_stride;
              } else {
                batch_ok = false;
                break;
              }

              for (int64_t base = 0; base < n_all; base += kMaxGemmM) {
                const int64_t n = std::min(kMaxGemmM, n_all - base);
                for (int64_t i = 0; i < n; ++i) {
                  h_tids[static_cast<size_t>(i)] = toks[static_cast<size_t>(base + i)];
                  h_wts[static_cast<size_t>(i)] = wts_e[static_cast<size_t>(base + i)];
                }
                d.b.Copy(d.q, d_tids.ptr(), h_tids.data(),
                         static_cast<size_t>(n) * sizeof(int32_t));
                d.b.Copy(d.q, d_wts.ptr(), h_wts.data(), static_cast<size_t>(n) * sizeof(float));
                // Gather expert_in rows → bx on GPU
                vt::MoeGatherRows(d.q, bx.ptr(), expert_in.data,
                                  static_cast<const int32_t*>(d_tids.ptr()), static_cast<int>(n),
                                  static_cast<int>(H));
                Tensor x = Tensor::Contiguous(bx.ptr(), DType::kBF16, dev, {n, H});
                if (use_peer_act) {
                  if (!RunGemma4Fp8ExpertGeGLUPrefillOnExpertDevice(
                          d.q, ex.dev_id, by.ptr(), bx.ptr(), fp8_gu_p, s_gu_p, fp8_dn_p, s_dn_p,
                          static_cast<int>(n), static_cast<int>(I), static_cast<int>(H))) {
                    batch_ok = false;
                    break;
                  }
                  ++peer_act_hits;
                } else if (use_fp8_dev) {
                  ExpertGeGLUFp8Native(d, by, x, fp8_gu_p, s_gu_p, fp8_dn_p, s_dn_p, I, H, besc, 1.f,
                                       0.f, fp8_weight_id);
                } else if (use_device_w) {
                  ExpertGeGLUDeviceAccum(d, by, x, gu_p, dn_p, I, H, besc, 1.f, 0.f);
                } else {
                  ExpertGeGLUHost(d, by, x, gu_host_e, dn_host_e, I, H, besc,
                                  ex.is_fp8 ? &ex.fp8[static_cast<size_t>(e)] : nullptr);
                }
                // Weighted scatter-add into acc on GPU (sequential experts → same token OK)
                vt::MoeWeightedScatterAdd(d.q, acc.ptr(), by.ptr(),
                                          static_cast<const int32_t*>(d_tids.ptr()),
                                          static_cast<const float*>(d_wts.ptr()), static_cast<int>(n),
                                          static_cast<int>(H));
              }
              if (!batch_ok) break;
              ++experts_run;
            }
            if (batch_ok) {
            // No host Synchronize: acc stays on-device; DualRmsNorm/next layer share
            // the same queue. (Was a full-device stall after every MoE layer prefill.)
            if (moe_trace) {
              d.b.Synchronize(d.q);  // only so stderr timing is meaningful
              std::fprintf(stderr,
                           "INFO gemma4-moe prefill-batch T=%lld experts_run=%d max_n=%lld "
                           "chunk=%lld fp8_res=%d same_dev=%d peer_act_hits=%d "
                           "(gpu-fp8+scatter)\n",
                           static_cast<long long>(T), experts_run, static_cast<long long>(max_n),
                           static_cast<long long>(kMaxGemmM), fp8_res ? 1 : 0, same_dev ? 1 : 0,
                           peer_act_hits);
              std::fflush(stderr);
            }
            static std::atomic<int> peer_act_log{0};
            if (peer_act_hits > 0 && peer_act_log.fetch_add(1) < 3) {
              std::fprintf(stderr,
                           "INFO gemma4-moe prefill PEER_ACT hits=%d experts_run=%d T=%lld "
                           "(weights stayed on expert GPU)\n",
                           peer_act_hits, experts_run, static_cast<long long>(T));
              std::fflush(stderr);
            }
      Gemma4MoeScratch r;
      r.tensor = acc.t();
      r.storage = acc.ReleaseShared();
      if (profile) {
        const auto t_all1 = clock::now();
        static std::atomic<uint64_t> ncalls{0};
        static std::atomic<uint64_t> us_router{0};
        static std::atomic<uint64_t> us_total{0};
        const auto ur =
            std::chrono::duration_cast<std::chrono::microseconds>(t_router1 - t_all0).count();
        const auto ut =
            std::chrono::duration_cast<std::chrono::microseconds>(t_all1 - t_all0).count();
        us_router.fetch_add(static_cast<uint64_t>(ur), std::memory_order_relaxed);
        us_total.fetch_add(static_cast<uint64_t>(ut), std::memory_order_relaxed);
        const uint64_t c = ncalls.fetch_add(1, std::memory_order_relaxed) + 1;
        if (c == 1 || c % 64 == 0) {
          const uint64_t tr = us_router.load(std::memory_order_relaxed);
          const uint64_t tt = us_total.load(std::memory_order_relaxed);
          std::fprintf(stderr,
                       "gemma4 moe profile: calls=%llu router_us/call=%.1f expert+rest_us/call=%.1f "
                       "total_us/call=%.1f (router%%=%.0f) [prefill-batch-gpu-scatter]\n",
                       static_cast<unsigned long long>(c), static_cast<double>(tr) / c,
                       static_cast<double>(tt - tr) / c, static_cast<double>(tt) / c,
                       tt ? 100.0 * static_cast<double>(tr) / static_cast<double>(tt) : 0.0);
        }
      }
      return r;
      }  // batch_ok — else fall through to serial GPU MoE
    }
  }

  for (int64_t t = 0; t < T; ++t) {
    // Drain previous token's device work before starting the next (prefill only).
    if (T > 1 && t > 0) d.b.Synchronize(d.q);

    std::vector<int> idx(static_cast<size_t>(top_k));
    std::vector<float> wts(static_cast<size_t>(top_k));
    for (int i = 0; i < top_k; ++i) {
      const size_t o = static_cast<size_t>(t * top_k + i);
      idx[static_cast<size_t>(i)] = static_cast<int>(hi[o]);
      wts[static_cast<size_t>(i)] = hw[o];
    }

    d.b.Copy(d.q, xin.ptr(),
             static_cast<const char*>(expert_in.data) +
                 static_cast<size_t>(t) * static_cast<size_t>(H) * 2,
             static_cast<size_t>(H) * 2);

    if (host_axpy) {
      std::fill(hsum.begin(), hsum.end(), vt::F32ToBF16(0.f));
    }
    // device path: first expert MulScalar writes ysum (no Zero needed)

    // Decode-only device Ensure (Touch/fallback). Prefill skips device LRU.
    if (use_dev_expert_lru) {
      for (int i = 0; i < top_k; ++i) {
        const int e = idx[static_cast<size_t>(i)];
        if (e >= 0 && e < static_cast<int>(E)) {
          if (fp8_native)
            (void)EnsureGemma4Fp8NativeOnDevice(d, ex.fp8[static_cast<size_t>(e)], I, H);
          else
            (void)EnsureGemma4Fp8ExpertOnDevice(d, ex.fp8[static_cast<size_t>(e)], I, H);
        }
      }
    }

    // Fused top-k ExpertGeGLU (VT_GEMMA4_FUSED_EXPERTS=1).
    if (!host_axpy && ex.is_fp8 && T == 1) {
      std::vector<const uint16_t*> gu_ptrs, dn_ptrs;
      gu_ptrs.reserve(static_cast<size_t>(top_k));
      dn_ptrs.reserve(static_cast<size_t>(top_k));
      bool all_dev = true;
      for (int i = 0; i < top_k; ++i) {
        const int e = idx[static_cast<size_t>(i)];
        const auto& fex = ex.fp8[static_cast<size_t>(e)];
        if (!fex.dev_gu || !fex.dev_dn) {
          all_dev = false;
          break;
        }
        gu_ptrs.push_back(static_cast<const uint16_t*>(fex.dev_gu));
        dn_ptrs.push_back(static_cast<const uint16_t*>(fex.dev_dn));
      }
      if (all_dev &&
          RunGemma4FusedTopkExpertGeGLU(d.q, ysum.ptr(), xin.ptr(), gu_ptrs.data(),
                                        dn_ptrs.data(), wts.data(), top_k, I, H)) {
        d.b.Copy(
            d.q,
            static_cast<char*>(acc.ptr()) + static_cast<size_t>(t) * static_cast<size_t>(H) * 2,
            ysum.ptr(), static_cast<size_t>(H) * 2);
        continue;
      }
    }

    // Batched path: VT_GEMMA4_BATCH_EXPERTS=1 (default off). Decode-only device.
    if (batch_experts && ex.is_fp8 && !host_axpy && T == 1) {
      std::vector<const uint16_t*> gu_ptrs;
      std::vector<const uint16_t*> dn_ptrs;
      gu_ptrs.reserve(static_cast<size_t>(top_k));
      dn_ptrs.reserve(static_cast<size_t>(top_k));
      bool all_dev = true;
      for (int i = 0; i < top_k; ++i) {
        const int e = idx[static_cast<size_t>(i)];
        const auto& fex = ex.fp8[static_cast<size_t>(e)];
        if (!EnsureGemma4Fp8ExpertOnDevice(d, fex, I, H)) {
          all_dev = false;
          break;
        }
        gu_ptrs.push_back(static_cast<const uint16_t*>(fex.dev_gu));
        dn_ptrs.push_back(static_cast<const uint16_t*>(fex.dev_dn));
      }
      if (all_dev && ExpertGeGLUDeviceBatched(d, ysum, xin.t(), gu_ptrs, dn_ptrs, wts, I, H)) {
        d.b.Copy(
            d.q,
            static_cast<char*>(acc.ptr()) + static_cast<size_t>(t) * static_cast<size_t>(H) * 2,
            ysum.ptr(), static_cast<size_t>(H) * 2);
        continue;  // next token
      }
    }

    // Fused-Gelu top-k FP8 native (resident or LRU). xin is always 1×H in this loop.
    if (!host_axpy && (fp8_native || fp8_res) && ex.is_fp8) {
      std::vector<const void*> fgu, sgu, fdn, sdn;
      fgu.reserve(static_cast<size_t>(top_k));
      sgu.reserve(static_cast<size_t>(top_k));
      fdn.reserve(static_cast<size_t>(top_k));
      sdn.reserve(static_cast<size_t>(top_k));
      bool ok = true;
      for (int i = 0; i < top_k && ok; ++i) {
        const int e = idx[static_cast<size_t>(i)];
        const auto& fex = ex.fp8[static_cast<size_t>(e)];
        if (fex.dev_fp8_gu && fex.dev_fp8_dn && fex.dev_s_gu && fex.dev_s_dn) {
          fgu.push_back(fex.dev_fp8_gu);
          sgu.push_back(fex.dev_s_gu);
          fdn.push_back(fex.dev_fp8_dn);
          sdn.push_back(fex.dev_s_dn);
        } else if (!fp8_res_peer && EnsureGemma4Fp8NativeOnDevice(d, fex, I, H)) {
          fgu.push_back(fex.dev_fp8_gu);
          sgu.push_back(fex.dev_s_gu);
          fdn.push_back(fex.dev_fp8_dn);
          sdn.push_back(fex.dev_s_dn);
        } else {
          ok = false;
        }
      }
      bool ran = false;
      void* acc_row = static_cast<char*>(acc.ptr()) +
                      static_cast<size_t>(t) * static_cast<size_t>(H) * 2;
      if (ok && static_cast<int>(fgu.size()) == top_k) {
        if (fp8_res_peer) {
          // Weights stay on expert GPU; move activations only (Phase-1 peer tax fix).
          // Write straight into acc row — skip ysum staging copy.
          ran = RunGemma4Fp8TopKOnExpertDevice(
              d.q, ex.dev_id, acc_row, xin.ptr(), fgu.data(), sgu.data(), fdn.data(),
              sdn.data(), wts.data(), top_k, static_cast<int>(I), static_cast<int>(H));
        } else {
          ran = vt::ExpertGeGLUFp8TopKM1(d.q, acc_row, xin.ptr(), fgu.data(), sgu.data(),
                                         fdn.data(), sdn.data(), wts.data(), top_k,
                                         static_cast<int>(I), static_cast<int>(H));
          if (!ran) {
            ran = ExpertGeGLUFp8TopKFusedGelu(d, ysum, xin.t(), fgu.data(), sgu.data(), fdn.data(),
                                              sdn.data(), wts.data(), top_k, I, H);
            if (ran) {
              d.b.Copy(d.q, acc_row, ysum.ptr(), static_cast<size_t>(H) * 2);
            }
          }
        }
      }
      if (ran) {
        continue;
      }
    }

    // Fused-Gelu top-k BF16 device path (resident gate_up_dev or LRU BF16).
    // Optional custom RDNA4 expert kernels: VT_GEMMA4_CUSTOM_EXPERT=1
    // Always try when BF16 stacks are on this GPU; peer BF16 uses serial+PeerCopy.
    if (!host_axpy && !fp8_res) {
      std::vector<const uint16_t*> gu_p, dn_p;
      gu_p.reserve(static_cast<size_t>(top_k));
      dn_p.reserve(static_cast<size_t>(top_k));
      bool ok = true;
      for (int i = 0; i < top_k && ok; ++i) {
        const int e = idx[static_cast<size_t>(i)];
        if (same_dev) {
          gu_p.push_back(static_cast<const uint16_t*>(ex.gate_up_dev) +
                         static_cast<int64_t>(e) * gu_stride);
          dn_p.push_back(static_cast<const uint16_t*>(ex.down_dev) +
                         static_cast<int64_t>(e) * dn_stride);
        } else if (ex.is_fp8) {
          const auto& fex = ex.fp8[static_cast<size_t>(e)];
          if (!fex.dev_gu || !fex.dev_dn) {
            if (!EnsureGemma4Fp8ExpertOnDevice(d, fex, I, H)) {
              ok = false;
              break;
            }
          }
          gu_p.push_back(static_cast<const uint16_t*>(fex.dev_gu));
          dn_p.push_back(static_cast<const uint16_t*>(fex.dev_dn));
        } else {
          ok = false;
        }
      }
      if (ok && static_cast<int>(gu_p.size()) == top_k) {
        bool ran = false;
        if (custom_expert) {
          std::vector<const void*> gu_v(static_cast<size_t>(top_k));
          std::vector<const void*> dn_v(static_cast<size_t>(top_k));
          for (int g = 0; g < top_k; ++g) {
            gu_v[static_cast<size_t>(g)] = gu_p[static_cast<size_t>(g)];
            dn_v[static_cast<size_t>(g)] = dn_p[static_cast<size_t>(g)];
          }
          ran = vt::ExpertGeGLUBf16TopKM1(d.q, ysum.ptr(), xin.ptr(), gu_v.data(),
                                                    dn_v.data(), wts.data(), top_k,
                                                    static_cast<int>(I), static_cast<int>(H));
        }
        if (!ran) {
          ran = ExpertGeGLUTopKFusedGelu(d, ysum, xin.t(), gu_p.data(), dn_p.data(), wts.data(),
                                         top_k, I, H);
        }
        if (ran) {
          d.b.Copy(
              d.q,
              static_cast<char*>(acc.ptr()) + static_cast<size_t>(t) * static_cast<size_t>(H) * 2,
              ysum.ptr(), static_cast<size_t>(H) * 2);
          d.b.Synchronize(d.q);
          continue;
        }
      }
    }

    for (int i = 0; i < top_k; ++i) {
      const int e = idx[static_cast<size_t>(i)];
      const float ww = wts[static_cast<size_t>(i)];
      const float beta = (i == 0) ? 0.f : 1.f;
      bool fused_mix = false;

      if (same_dev) {
        auto* gu = static_cast<const uint16_t*>(ex.gate_up_dev) +
                   static_cast<int64_t>(e) * gu_stride;
        auto* dn =
            static_cast<const uint16_t*>(ex.down_dev) + static_cast<int64_t>(e) * dn_stride;
        ExpertGeGLUDeviceAccum(d, ysum, xin.t(), gu, dn, I, H, esc, ww, beta);
        fused_mix = true;
      } else if (fp8_res_same && ex.is_fp8) {
        const auto& fex = ex.fp8[static_cast<size_t>(e)];
        ExpertGeGLUFp8Native(d, ysum, xin.t(), fex.dev_fp8_gu, fex.dev_s_gu, fex.dev_fp8_dn,
                             fex.dev_s_dn, I, H, esc, ww, beta);
        fused_mix = true;
      } else if (fp8_res_peer && ex.is_fp8 && tls.fp8_gu_sc && tls.fp8_dn_sc && tls.fp8_sgu_sc &&
                 tls.fp8_sdn_sc) {
        const auto& fex = ex.fp8[static_cast<size_t>(e)];
        if (PeerCopyGemma4Fp8ExpertSlice(ex.dev_id, fex.dev_fp8_gu, fex.dev_fp8_dn, fex.dev_s_gu,
                                         fex.dev_s_dn, I, H, compute_dev, tls.fp8_gu_sc->ptr(),
                                         tls.fp8_dn_sc->ptr(), tls.fp8_sgu_sc->ptr(),
                                         tls.fp8_sdn_sc->ptr())) {
          ExpertGeGLUFp8Native(d, ysum, xin.t(), tls.fp8_gu_sc->ptr(), tls.fp8_sgu_sc->ptr(),
                               tls.fp8_dn_sc->ptr(), tls.fp8_sdn_sc->ptr(), I, H, esc, ww, beta);
          fused_mix = true;
        }
      } else if (fp8_native && ex.is_fp8) {
        const auto& fex = ex.fp8[static_cast<size_t>(e)];
        if (EnsureGemma4Fp8NativeOnDevice(d, fex, I, H)) {
          ExpertGeGLUFp8Native(d, ysum, xin.t(), fex.dev_fp8_gu, fex.dev_s_gu, fex.dev_fp8_dn,
                               fex.dev_s_dn, I, H, esc, ww, beta);
          fused_mix = true;
        } else {
          // Native FP8 device arm unavailable — fall back to host BF16 dequant,
          // exactly as the !fp8_native sibling below.  Without this else the
          // expert is skipped and `y` is never written, so the accumulator gets
          // whatever the TLS scratch held (zero under TSan, garbage otherwise).
          EnsureGemma4Fp8ExpertCached(fex, I, H);
          if (!fex.cached_gu.empty() && !fex.cached_dn.empty()) {
            ExpertGeGLUHost(d, y, xin.t(), fex.cached_gu.data(), fex.cached_dn.data(), I, H, esc,
                            &fex);
          } else {
            DequantGemma4Fp8ExpertToBf16Ephemeral(fex, I, H, gu_tmp.data(), dn_tmp.data());
            ExpertGeGLUHost(d, y, xin.t(), gu_tmp.data(), dn_tmp.data(), I, H, esc, &fex);
          }
        }
      } else if (need_peer_sc && gu_sc && dn_sc) {
        if (PeerCopyGemma4ExpertSlice(ex.dev_id, ex.gate_up_dev, ex.down_dev, e, I, H,
                                      compute_dev, gu_sc->ptr(), dn_sc->ptr())) {
          ExpertGeGLUDeviceAccum(d, ysum, xin.t(), static_cast<const uint16_t*>(gu_sc->ptr()),
                                 static_cast<const uint16_t*>(dn_sc->ptr()), I, H, esc, ww,
                                 beta);
          fused_mix = true;
        } else if (ex.is_fp8) {
          const auto& fex = ex.fp8[static_cast<size_t>(e)];
          DequantGemma4Fp8ExpertToBf16Ephemeral(fex, I, H, gu_tmp.data(), dn_tmp.data());
          ExpertGeGLUHost(d, y, xin.t(), gu_tmp.data(), dn_tmp.data(), I, H, esc, &fex);
        } else {
          VT_CHECK(gu_host && dn_host, "gemma4 moe: peer fail no host");
          ExpertGeGLUHost(d, y, xin.t(), gu_host + static_cast<int64_t>(e) * gu_stride,
                          dn_host + static_cast<int64_t>(e) * dn_stride, I, H, esc,
                          gu_host + static_cast<int64_t>(e) * gu_stride);
        }
      } else if (ex.is_fp8) {
      const auto& fex = ex.fp8[static_cast<size_t>(e)];
      if (EnsureGemma4Fp8ExpertOnDevice(d, fex, I, H)) {
        ExpertGeGLUDeviceAccum(d, ysum, xin.t(), static_cast<const uint16_t*>(fex.dev_gu),
                               static_cast<const uint16_t*>(fex.dev_dn), I, H, esc, ww,
                               beta);
        fused_mix = true;
      } else {
        // Prefer bounded host BF16 LRU; fall back to stack ephemeral dequant.
        EnsureGemma4Fp8ExpertCached(fex, I, H);
        if (!fex.cached_gu.empty() && !fex.cached_dn.empty()) {
          ExpertGeGLUHost(d, y, xin.t(), fex.cached_gu.data(), fex.cached_dn.data(), I, H, esc,
                          &fex);
        } else {
          DequantGemma4Fp8ExpertToBf16Ephemeral(fex, I, H, gu_tmp.data(), dn_tmp.data());
          ExpertGeGLUHost(d, y, xin.t(), gu_tmp.data(), dn_tmp.data(), I, H, esc, &fex);
        }
      }
      } else if (gu_host && dn_host) {
      ExpertGeGLUHost(d, y, xin.t(), gu_host + static_cast<int64_t>(e) * gu_stride,
                      dn_host + static_cast<int64_t>(e) * dn_stride, I, H, esc,
                      gu_host + static_cast<int64_t>(e) * gu_stride);
      } else {
      VT_CHECK(false, "gemma4 moe: no expert weights");
      }

      if (fused_mix) {
        continue;  // already accumulated into ysum (same stream)
      }

      if (host_axpy) {
        d.b.Synchronize(d.q);
        std::vector<uint16_t> hy(static_cast<size_t>(H));
        d.b.Copy(d.q, hy.data(), y.ptr(), hy.size() * 2);
        d.b.Synchronize(d.q);
        for (int64_t j = 0; j < H; ++j)
          hsum[static_cast<size_t>(j)] = vt::F32ToBF16(
              vt::BF16ToF32(hsum[static_cast<size_t>(j)]) +
              ww * vt::BF16ToF32(hy[static_cast<size_t>(j)]));
      } else if (i == 0) {
        vt::MulScalar(d.q, ysum.t(), y.t(), static_cast<double>(ww));
      } else {
        vt::MulScalar(d.q, ysc.t(), y.t(), static_cast<double>(ww));
        vt::Add(d.q, ysum.t(), ysum.t(), ysc.t());
      }
    }
    if (host_axpy) {
      d.b.Copy(d.q, ysum.ptr(), hsum.data(), hsum.size() * 2);
    }
    d.b.Copy(d.q,
             static_cast<char*>(acc.ptr()) + static_cast<size_t>(t) * static_cast<size_t>(H) * 2,
             ysum.ptr(), static_cast<size_t>(H) * 2);
    // One drain per token after all top_k experts (prefill + decode).
    d.b.Synchronize(d.q);
  }

  Gemma4MoeScratch r;
  r.tensor = acc.t();
  if (T == 1) {
    // TLS-owned: non-owning view for DualRmsNorm; next call overwrites same buffer.
    r.storage = std::shared_ptr<void>(acc.ptr(), [](void*) {});
  } else {
    r.storage = acc.ReleaseShared();
  }

  if (profile) {
    d.b.Synchronize(d.q);
    const auto t_all1 = clock::now();
    static std::atomic<uint64_t> ncalls{0};
    static std::atomic<uint64_t> us_router{0};
    static std::atomic<uint64_t> us_total{0};
    const auto ur = std::chrono::duration_cast<std::chrono::microseconds>(t_router1 - t_all0).count();
    const auto ut = std::chrono::duration_cast<std::chrono::microseconds>(t_all1 - t_all0).count();
    us_router.fetch_add(static_cast<uint64_t>(ur), std::memory_order_relaxed);
    us_total.fetch_add(static_cast<uint64_t>(ut), std::memory_order_relaxed);
    const uint64_t c = ncalls.fetch_add(1, std::memory_order_relaxed) + 1;
    if (c == 1 || c % 64 == 0) {
      const uint64_t tr = us_router.load(std::memory_order_relaxed);
      const uint64_t tt = us_total.load(std::memory_order_relaxed);
      std::fprintf(stderr,
                   "gemma4 moe profile: calls=%llu router_us/call=%.1f expert+rest_us/call=%.1f "
                   "total_us/call=%.1f (router%%=%.0f)\n",
                   static_cast<unsigned long long>(c), static_cast<double>(tr) / c,
                   static_cast<double>(tt - tr) / c, static_cast<double>(tt) / c,
                   tt ? 100.0 * static_cast<double>(tr) / static_cast<double>(tt) : 0.0);
    }
  }
  return r;
}

#ifndef VLLM_CPP_HIP
// Resident-expert preload is a discrete-ROCm optimization; its real
// implementation lives in src/vt/rocm/rocm_gemma4_experts.hip and is compiled
// only under -DVLLM_CPP_HIP. Non-HIP builds need these symbols to LINK (the
// call site in gemma4_registry.cpp is gated on VT_GEMMA4_RESIDENT_EXPERTS=1 and
// is never reached off-ROCm, but the reference must still resolve). Loud no-op:
// if a caller ever asks for resident experts without a HIP backend, say so.
size_t UploadGemma4ExpertsResident(std::vector<Gemma4MoeLayerWeights>& layers,
                                   int num_gpus) {
  (void)layers;
  (void)num_gpus;
  std::fprintf(stderr,
               "[gemma4] VT_GEMMA4_RESIDENT_EXPERTS requested but this binary "
               "was built without -DVLLM_CPP_HIP; resident preload is a no-op.\n");
  return 0;
}
size_t UploadGemma4ExpertsResidentForWeights(Gemma4Weights& weights,
                                             int num_gpus) {
  (void)weights;
  (void)num_gpus;
  std::fprintf(stderr,
               "[gemma4] VT_GEMMA4_RESIDENT_EXPERTS requested but this binary "
               "was built without -DVLLM_CPP_HIP; resident preload is a no-op.\n");
  return 0;
}
bool RunGemma4FusedTopkExpertGeGLU(vt::Queue&, void*, const void*, const uint16_t* const*,
                                   const uint16_t* const*, const float*, int, int64_t, int64_t) {
  return false;
}
bool PeerCopyGemma4ExpertSlice(int, const void*, const void*, int, int64_t, int64_t, int, void*,
                               void*) {
  return false;
}
bool PeerCopyGemma4Fp8ExpertSlice(int, const void*, const void*, const void*, const void*, int64_t,
                                  int64_t, int, void*, void*, void*, void*) {
  return false;
}
bool RunGemma4Fp8TopKOnExpertDevice(vt::Queue&, int, void*, const void*, const void* const*,
                                    const void* const*, const void* const*, const void* const*,
                                    const float*, int, int, int) {
  return false;
}
bool RunGemma4Fp8ExpertGeGLUPrefillOnExpertDevice(vt::Queue&, int, void*, const void*, const void*,
                                                  const void*, const void*, const void*, int, int,
                                                  int) {
  return false;
}
bool RunGemma4Fp8TopKIndexedOnExpertDevice(vt::Queue&, int, void*, const void*, const void*,
                                           const void*, const void*, const void*, const int32_t*,
                                           const float*, int, int, int) {
  return false;
}
bool RetireGemma4Fp8TopKIndexedPeer(vt::Queue&, int) { return true; }
void PinGemma4Fp8ExpertHostCache(const Gemma4Fp8ExpertMats&) {}
void UnpinGemma4Fp8ExpertHostCache(const Gemma4Fp8ExpertMats&) {}
#endif  // VLLM_CPP_HIP

}  // namespace vllm
