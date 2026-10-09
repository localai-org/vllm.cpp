// vllm.cpp original (vt runtime, inventory deviation §9.1) — implementation of
// the acceleration-provider seam declared in include/vt/op_provider.h
// (BACKEND-ACCEL-PROVIDER, .agents/specs/metal-mlx-reuse-study.md §6).
//
// Structure is ported from src/vt/cuda/cuda_arch_tactics.cu:64-170, which proved
// the shape one layer down: fixed-capacity static storage so registration is
// safe from any static-init order, table-fill-only registration that never
// throws, a linear capability scan at selection, and atomic instrumentation.
//
// The ONE thing changed in the port is the property that made the flat op table
// a bug: selection is `(priority DESC, name ASC)`, NOT registration order. The
// tactic registry could afford registration order because exactly one tactic is
// registered per family; the op table cannot, because the whole point is two
// providers coexisting, and registration order across TUs is unspecified.
//
// These three symbols (`RegisterOp`, `GetOp`, `OpRegistered`) were previously
// defined in src/vt/ops.cpp over a flat `void* [OpId][DeviceType]` array. They
// moved HERE rather than growing that file — it is the single hottest shared TU
// in the tree and every op wrapper in it is untouched by this change.
#include "vt/op_provider.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>

#include "vt/backend.h"
#include "vt/ops.h"

namespace vt {
namespace {

// Enough for the native kernel plus three accelerator providers on one op. A
// higher count is a design smell (four libraries claiming one op), and a bounded
// array is what keeps registration allocation-free and static-init-order-safe.
constexpr int kMaxProvidersPerOp = 4;
constexpr size_t kOpCount = static_cast<size_t>(OpId::kCount);

struct Slot {
  OpProvider providers[kMaxProvidersPerOp];
  int count = 0;
  // Resolved selection cache. GetOp is on the hot path of every op in the tree,
  // so the steady state must stay one relaxed load, exactly like the array load
  // it replaces. Registration and SetDeviceProviderCaps() clear it.
  std::atomic<void*> selected{nullptr};
  // Memoized NEGATIVE resolution. `OpRegistered` is consulted per call by the
  // fused-recipe fast-realization ladder (src/vt/ops.cpp) for ops a backend does
  // NOT have, so the "nothing supports this" answer must be as cheap as the
  // positive one — otherwise the seam would put a capability scan on the 35B
  // decode path. Cleared by registration and by SetDeviceProviderCaps().
  std::atomic<bool> resolved_none{false};
  // Cached NATIVE-availability for OpRegistered: -1 unknown, 0 none, 1 present.
  // OpRegistered means "is there a NATIVE kernel" (the fused-recipe ladder and
  // the cross-device harness depend on that meaning), so it deliberately IGNORES
  // any reference-tier fallback provider in this slot. Cleared exactly like the
  // selection cache. This is separate from `selected` because `selected` CAN
  // hold a reference-tier fn (a real GetOp answer) while native stays absent.
  std::atomic<int8_t> native_registered{-1};
  // One-time loud reference-tier warning per (op, device) — see Resolve.
  std::atomic<bool> ref_announced{false};
  // Is the CURRENT selection the portable CPU reference tier? Read on every
  // GetOp, so it is a plain relaxed bool rather than a string compare.
  std::atomic<bool> ref_selected{false};
  std::atomic<const char*> last_selected{nullptr};
  std::atomic<unsigned long long> selections{0};
  std::atomic<unsigned long long> declines{0};
  std::atomic<unsigned long long> fallbacks{0};
  std::atomic<bool> announced{false};
};

// OBSERVABILITY: total reference-tier selections across all (op, device). A
// process that ran a model entirely on native kernels leaves this at 0; any
// value > 0 is proof the portable CPU path fired and the run is not a
// performance measurement. Ported in spirit from the ArchTacticStats counters.
std::atomic<unsigned long long>& RefTierHits() {
  static std::atomic<unsigned long long> hits{0};
  return hits;
}

// Zero-initialized static storage: usable from any static-init order, no
// dynamic allocation, no dependence on another TU's constructor. ~66 KiB of BSS.
Slot* Table() {
  static Slot table[kOpCount * kNumDeviceTypes];
  return table;
}

Slot& At(OpId op, DeviceType device) {
  const size_t o = static_cast<size_t>(op);
  const size_t d = static_cast<size_t>(device);
  VT_CHECK(o < kOpCount, "invalid op id");
  VT_CHECK(d < kNumDeviceTypes, "invalid device type");
  return Table()[o * kNumDeviceTypes + d];
}

// JSONL records count provider LOOKUPS, not registrations or completed kernels.
// The environment is sampled once, before the first lookup. No file is opened
// on the default path. Names are JSON-escaped (provider names are extensible).
bool TraceEnabled() {
  static const bool enabled = [] {
    const char* path = std::getenv("VT_OP_PROVIDER_TRACE");
    return path != nullptr && *path != '\0';
  }();
  return enabled;
}

struct Trace {
    std::mutex mutex;
    std::ofstream file{std::getenv("VT_OP_PROVIDER_TRACE"), std::ios::app};
    unsigned long long sequence = 0;
    unsigned long long reference_hits = 0;
    std::map<std::string, unsigned long long> counts;
};
Trace& TraceState() {
  static Trace trace;
  return trace;
}
void TraceSelection(OpId op, DeviceType device, const char* provider, const char* event) {
  if (!TraceEnabled()) return;
  auto& trace = TraceState();
  std::lock_guard<std::mutex> lock(trace.mutex);
  VT_CHECK(trace.file.is_open(), "cannot open VT_OP_PROVIDER_TRACE");
  const bool reference = std::strcmp(provider, kReferenceProviderName) == 0;
  if (reference) ++trace.reference_hits;
  const std::string key = std::string(OpName(op)) + "/" + DeviceTypeName(device) + "/" + provider;
  const nlohmann::json row = {
      {"event", event}, {"op", OpName(op)}, {"device", DeviceTypeName(device)},
      {"provider", provider}, {"cpu_reference", reference},
      {"sequence", ++trace.sequence}, {"provider_selections", ++trace.counts[key]},
      {"reference_tier_hits", trace.reference_hits}, {"count_kind", "selection"}};
  trace.file << row.dump() << '\n';
  trace.file.flush();
  VT_CHECK(trace.file.good(), "cannot write VT_OP_PROVIDER_TRACE");
}

// --- device capability records ---------------------------------------------
struct CapsTable {
  std::mutex mu;
  ProviderCaps caps[kNumDeviceTypes];
};
CapsTable& Caps() {
  static CapsTable t;
  return t;
}

// --- runtime provider disable list (VT_OP_PROVIDER_DISABLE / DisableOpProvider)
constexpr int kMaxDisabled = 8;
constexpr size_t kMaxNameLen = 63;
struct DisableList {
  std::mutex mu;
  char names[kMaxDisabled][kMaxNameLen + 1] = {};
  int count = 0;
  bool env_parsed = false;
  // Lock-free fast path. `Supported()` runs inside the capability scan, which
  // `OpRegistered` reaches on a cold slot; taking a mutex there would be a hot
  // path regression. The overwhelmingly common state is "nothing disabled", and
  // this atomic answers that without touching `mu`.
  std::atomic<int> live_count{-1};  // -1 == env not parsed yet
};
DisableList& Disabled() {
  static DisableList d;
  return d;
}

void AddDisabledLocked(DisableList& d, const char* begin, const char* end) {
  size_t n = static_cast<size_t>(end - begin);
  if (n == 0 || n > kMaxNameLen || d.count >= kMaxDisabled) return;
  for (int i = 0; i < d.count; ++i) {
    if (std::strncmp(d.names[i], begin, n) == 0 && d.names[i][n] == '\0') return;
  }
  std::memcpy(d.names[d.count], begin, n);
  d.names[d.count][n] = '\0';
  ++d.count;
  d.live_count.store(d.count, std::memory_order_relaxed);
}

void EnsureEnvParsedLocked(DisableList& d) {
  if (d.env_parsed) return;
  d.env_parsed = true;
  const char* e = std::getenv("VT_OP_PROVIDER_DISABLE");
  if (e != nullptr) {
    const char* p = e;
    while (*p != '\0') {
      const char* q = p;
      while (*q != '\0' && *q != ',') ++q;
      AddDisabledLocked(d, p, q);
      p = (*q == '\0') ? q : q + 1;
    }
  }
  d.live_count.store(d.count, std::memory_order_relaxed);
}

std::atomic<bool>& CallStatsFlag() {
  static std::atomic<bool> on{[] {
    const char* e = std::getenv("VT_OP_PROVIDER_STATS");
    return e != nullptr && e[0] == '1';
  }()};
  return on;
}

bool AnnounceEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_OP_PROVIDER_STATS");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// The DETERMINISTIC total order: higher priority first, then name ascending by
// strcmp. Both keys are compile-time constants of the registering TU, so the
// winner does not depend on which static initializer ran first.
bool Better(const OpProvider& a, const OpProvider& b) {
  if (a.priority != b.priority) return a.priority > b.priority;
  return std::strcmp(a.name, b.name) < 0;
}

bool Supported(const OpProvider& p, const ProviderCaps& caps) {
  if (p.fn == nullptr) return false;
  if (OpProviderDisabled(p.name)) return false;
  return p.supports == nullptr || p.supports(caps);
}

// Best supported provider strictly below `floor` in the order, or the best
// overall when `floor` is null. One linear scan; kMaxProvidersPerOp is 4.
const OpProvider* Choose(Slot& slot, const ProviderCaps& caps, const OpProvider* floor) {
  const OpProvider* best = nullptr;
  for (int i = 0; i < slot.count; ++i) {
    const OpProvider& p = slot.providers[i];
    if (!Supported(p, caps)) continue;
    if (floor != nullptr && !Better(*floor, p)) continue;
    if (best == nullptr || Better(p, *best)) best = &p;
  }
  return best;
}

// The portable reference tier (S5). When (op, device) has no provider and the
// device is a UNIFIED-MEMORY accelerator, install the CPU kernel as a
// negative-priority fallback so GetOp returns a working (if slow) kernel instead
// of throwing — our equivalent of vLLM CustomOp.forward_native. Returns true iff
// a fallback was installed. The unified-memory gate lives in
// ReferenceTierEligible; a discrete device NEVER reaches RegisterOpProvider here.
bool MaybeInstallReferenceTier(OpId op, DeviceType device) {
  if (!ReferenceTierEligible(device)) return false;
  // Already installed for this (op, device)? Report no-install so an eager
  // second pass (RegisterReferenceTier) is exactly idempotent — RegisterOpProvider
  // rejects the duplicate name, so counting it would over-report.
  Slot& target = At(op, device);
  for (int i = 0; i < target.count; ++i) {
    if (std::strcmp(target.providers[i].name, kReferenceProviderName) == 0) return false;
  }
  Slot& cpu = At(op, DeviceType::kCPU);
  if (cpu.count == 0) return false;  // no CPU reference either — a genuine throw
  const ProviderCaps cpu_caps = GetDeviceProviderCaps(DeviceType::kCPU);
  const OpProvider* src = Choose(cpu, cpu_caps, nullptr);
  if (src == nullptr) return false;
  OpProvider ref;
  ref.name = kReferenceProviderName;
  ref.priority = kReferenceTierPriority;
  ref.supports = nullptr;  // portable: runs wherever host==device memory
  ref.fn = src->fn;        // the SAME host kernel the CPU device dispatches
  RegisterOpProvider(op, device, ref);
  return true;
}

// Why the portable CPU reference tier did not answer for (op, device). Called
// only on the refusal path, so it may take the slow, readable route. The clauses
// name the tier's preconditions in the order MaybeInstallReferenceTier and
// ReferenceTierEligible test them; the last one asks OpRegistered on the CPU
// slot rather than repeating that function's Choose call, which is the same
// question for the CPU device, where no reference provider ever exists.
std::string ReferenceTierRefusalReason(OpId op, DeviceType device) {
  if (device == DeviceType::kCPU) {
    return "the portable CPU reference tier is the SOURCE of that kernel, not a "
           "fallback for it";
  }
  Backend* b = TryGetBackend(device);
  if (b == nullptr) {
    return "no backend is registered for that device in this build, so the "
           "portable CPU reference tier has nothing to install onto";
  }
  if (!b->DeviceMemoryIsHostAddressable()) {
    std::string why =
        std::string("the portable CPU reference tier is NOT eligible: this "
                    "backend does not report its device memory "
                    "host-addressable, so a host kernel may not dereference "
                    "what it allocated (unified memory is ") +
        (b->UnifiedMemory() ? "true" : "false") +
        ", which is a DIFFERENT property). Build a native kernel for this op "
        "or run it on the CPU device";
    // A backend that WITHDREW the property deliberately gets to say why, right
    // here, where the user meets the consequence. The generic sentence above is
    // true on every such device and useless on the one where the false is a
    // decision rather than the hardware -- ROCm on an XNACK-less integrated
    // part, issue #2511, is the case this exists for. Null on every backend
    // with nothing to add, which leaves the message byte-identical.
    if (const char* note = b->HostAddressabilityNote(); note != nullptr) {
      why += ". ";
      why += note;
    }
    return why;
  }
  if (!OpRegistered(op, DeviceType::kCPU)) {
    return "the portable CPU reference tier is eligible but has no CPU kernel "
           "for this op to install";
  }
  return "the portable CPU reference tier declined it";
}

void Announce(OpId op, DeviceType device, Slot& slot, const OpProvider* chosen,
              const ProviderCaps& caps) {
  if (!AnnounceEnabled()) return;
  if (slot.announced.exchange(true, std::memory_order_relaxed)) return;
  std::fprintf(stderr,
               "[vt op-provider] op=%d device=%d selected=%s priority=%d "
               "registered=%d caps=%d.%d/%s\n",
               static_cast<int>(op), static_cast<int>(device),
               chosen != nullptr ? chosen->name : "<none>",
               chosen != nullptr ? chosen->priority : 0, slot.count, caps.compute_major,
               caps.compute_minor, caps.valid ? "valid" : "unprobed");
}

// The canonical spelling of an op, for user-facing messages. Mirrors
// DeviceTypeName (include/vt/device.h): a data list, not a device-specific
// branch. The switch is deliberately EXHAUSTIVE and carries no `default` — with
// -Wall -Werror, appending an OpId without naming it fails the build, which is
// what keeps this from drifting behind the enum.
const char* OpNameImpl(OpId op) {
  switch (op) {
    case OpId::kMatmul:
      return "Matmul";
    case OpId::kMatmulGptq4W4A16:
      return "MatmulGptq4W4A16";
    case OpId::kMatmulDenseF16:
      return "MatmulDenseF16";
    case OpId::kVisionPosEmbedInterpolate:
      return "VisionPosEmbedInterpolate";
    case OpId::kVisionRopeGrid:
      return "VisionRopeGrid";
    case OpId::kVisionRopeApply:
      return "VisionRopeApply";
    case OpId::kRmsNorm:
      return "RmsNorm";
    case OpId::kResidualRmsNorm:
      return "ResidualRmsNorm";
    case OpId::kScaledRmsNorm:
      return "ScaledRmsNorm";
    case OpId::kSandwichRmsNorm:
      return "SandwichRmsNorm";
    case OpId::kCompiledGeluErfMul:
      return "CompiledGeluErfMul";
    case OpId::kSiluAndMul:
      return "SiluAndMul";
    case OpId::kRopeNeox:
      return "RopeNeox";
    case OpId::kEmbedding:
      return "Embedding";
    case OpId::kEmbeddingQuant:
      return "EmbeddingQuant";
    case OpId::kCausalConv1dFwd:
      return "CausalConv1dFwd";
    case OpId::kCausalConv1dUpdate:
      return "CausalConv1dUpdate";
    case OpId::kCausalConv1dSpecUpdate:
      return "CausalConv1dSpecUpdate";
    case OpId::kL2Norm:
      return "L2Norm";
    case OpId::kRmsNormGated:
      return "RmsNormGated";
    case OpId::kGdnPrefill:
      return "GdnPrefill";
    case OpId::kGdnPrefillRawGate:
      return "GdnPrefillRawGate";
    case OpId::kGdnDecode:
      return "GdnDecode";
    case OpId::kGdnSpecDecode:
      return "GdnSpecDecode";
    case OpId::kGdnPackedDecode:
      return "GdnPackedDecode";
    case OpId::kKdaGatedDeltaRule:
      return "KdaGatedDeltaRule";
    case OpId::kKdaChunkPrefill:
      return "KdaChunkPrefill";
    case OpId::kMoeRouterTopK:
      return "MoeRouterTopK";
    case OpId::kMoeCombine:
      return "MoeCombine";
    case OpId::kAttention:
      return "Attention";
    case OpId::kAttentionCross:
      return "AttentionCross";
    case OpId::kAttentionDenseFast:
      return "AttentionDenseFast";
    case OpId::kAttentionDenseFlash:
      return "AttentionDenseFlash";
    case OpId::kDFlashBlockAttention:
      return "DFlashBlockAttention";
    case OpId::kDFlashPagedBlockAttention:
      return "DFlashPagedBlockAttention";
    case OpId::kDFlashGroupedConv:
      return "DFlashGroupedConv";
    case OpId::kDflash2SelectorEdges:
      return "Dflash2SelectorEdges";
    case OpId::kDflash2PathWalk:
      return "Dflash2PathWalk";
    case OpId::kTopKValuesIndices:
      return "TopKValuesIndices";
    case OpId::kReshapeAndCache:
      return "ReshapeAndCache";
    case OpId::kConcatAndCacheMla:
      return "ConcatAndCacheMla";
    case OpId::kConcatAndCacheDsMla:
      return "ConcatAndCacheDsMla";
    case OpId::kDequantAndGatherDsMla:
      return "DequantAndGatherDsMla";
    case OpId::kMlaDecodeAttention:
      return "MlaDecodeAttention";
    case OpId::kMlaPrefillAttention:
      return "MlaPrefillAttention";
    case OpId::kGatherMlaCache:
      return "GatherMlaCache";
    case OpId::kMergeAttnStates:
      return "MergeAttnStates";
    case OpId::kPagedAttention:
      return "PagedAttention";
    case OpId::kApplyTemperature:
      return "ApplyTemperature";
    case OpId::kGreedyArgmax:
      return "GreedyArgmax";
    case OpId::kMappedGreedyArgmax:
      return "MappedGreedyArgmax";
    case OpId::kApplyTopKTopP:
      return "ApplyTopKTopP";
    case OpId::kComputeProbs:
      return "ComputeProbs";
    case OpId::kComputeLogprobs:
      return "ComputeLogprobs";
    case OpId::kRandomSample:
      return "RandomSample";
    case OpId::kApplyPenalties:
      return "ApplyPenalties";
    case OpId::kApplyMinP:
      return "ApplyMinP";
    case OpId::kApplyLogitBias:
      return "ApplyLogitBias";
    case OpId::kApplyTokenMask:
      return "ApplyTokenMask";
    case OpId::kApplyAllowedTokenIds:
      return "ApplyAllowedTokenIds";
    case OpId::kMatmulNvfp4:
      return "MatmulNvfp4";
    case OpId::kScaledFp4Quant:
      return "ScaledFp4Quant";
    case OpId::kSiluMulFp4Quant:
      return "SiluMulFp4Quant";
    case OpId::kSiluAndMulFp4Quant:
      return "SiluAndMulFp4Quant";
    case OpId::kSigmoidGateFp4Quant:
      return "SigmoidGateFp4Quant";
    case OpId::kMatmulNvfp4Fp4:
      return "MatmulNvfp4Fp4";
    case OpId::kMatmulNvfp4Cutlass:
      return "MatmulNvfp4Cutlass";
    case OpId::kMatmulFp8Cutlass:
      return "MatmulFp8Cutlass";
    case OpId::kMatmulFp8CublasLt:
      return "MatmulFp8CublasLt";
    case OpId::kQuantFp8Static:
      return "QuantFp8Static";
    case OpId::kSwizzleBlockscale:
      return "SwizzleBlockscale";
    case OpId::kMoeGroupedGemmNvfp4:
      return "MoeGroupedGemmNvfp4";
    case OpId::kMoeSiluMul:
      return "MoeSiluMul";
    case OpId::kMoeRelu2:
      return "MoeRelu2";
    case OpId::kClampedSwiGLU:
      return "ClampedSwiGLU";
    case OpId::kCastBf16:
      return "CastBf16";
    case OpId::kCastF16:
      return "CastF16";
    case OpId::kCastF32:
      return "CastF32";
    case OpId::kPermuteVHeads:
      return "PermuteVHeads";
    case OpId::kMulColVecF32:
      return "MulColVecF32";
    case OpId::kAttnGateSplit:
      return "AttnGateSplit";
    case OpId::kSigmoidGateBf16:
      return "SigmoidGateBf16";
    case OpId::kGdnGBeta:
      return "GdnGBeta";
    case OpId::kGdnConvSplit:
      return "GdnConvSplit";
    case OpId::kQkvSplit:
      return "QkvSplit";
    case OpId::kSharedExpertGate:
      return "SharedExpertGate";
    case OpId::kMoeCombineGate:
      return "MoeCombineGate";
    case OpId::kMoeGroupedGemmNvfp4Marlin:
      return "MoeGroupedGemmNvfp4Marlin";
    case OpId::kGdnPostConv:
      return "GdnPostConv";
    case OpId::kRopeCosSinCache:
      return "RopeCosSinCache";
    case OpId::kAttnQkNormRopeGate:
      return "AttnQkNormRopeGate";
    case OpId::kAttnQkNormRope:
      return "AttnQkNormRope";
    case OpId::kFusedChain:
      return "FusedChain";
    case OpId::kRmsNormQuantFp8:
      return "RmsNormQuantFp8";
    case OpId::kRmsNormGatedQuantFp8:
      return "RmsNormGatedQuantFp8";
    case OpId::kMatmulBT:
      return "MatmulBT";
    case OpId::kMatmulBTQuant:
      return "MatmulBTQuant";
    case OpId::kMatmulBTQuantGrouped:
      return "MatmulBTQuantGrouped";
    case OpId::kDropinProbe:
      return "DropinProbe";
    case OpId::kRopeFromCache:
      return "RopeFromCache";
    case OpId::kGdnStateGather:
      return "GdnStateGather";
    case OpId::kGdnStateScatter:
      return "GdnStateScatter";
    case OpId::kIndexSelect:
      return "IndexSelect";
    case OpId::kIndexCopy:
      return "IndexCopy";
    case OpId::kVHeadPermute:
      return "VHeadPermute";
    case OpId::kMoeGroupedGemmBf16:
      return "MoeGroupedGemmBf16";
    case OpId::kLayerNorm:
      return "LayerNorm";
    case OpId::kRelu:
      return "Relu";
    case OpId::kGeluTanh:
      return "GeluTanh";
    case OpId::kGeluErf:
      return "GeluErf";
    case OpId::kAdd:
      return "Add";
    case OpId::kBatchedMatmul:
      return "BatchedMatmul";
    case OpId::kConcatMlaNopeRope:
      return "ConcatMlaNopeRope";
    case OpId::kDsaIndexerLogits:
      return "DsaIndexerLogits";
    case OpId::kDsaTopkSelect:
      return "DsaTopkSelect";
    case OpId::kGeluAndMul:
      return "GeluAndMul";
    case OpId::kMulScalar:
      return "MulScalar";
    case OpId::kSoftCap:
      return "SoftCap";
    case OpId::kGreedyRejectionSample:
      return "GreedyRejectionSample";
    case OpId::kAllReduce:
      return "AllReduce";
    case OpId::kAllGather:
      return "AllGather";
    case OpId::kSend:
      return "Send";
    case OpId::kRecv:
      return "Recv";
    case OpId::kDeepseekV4Mhc:
      return "DeepseekV4Mhc";
    case OpId::kDeepseekV4Dsa:
      return "DeepseekV4Dsa";
    case OpId::kDeepseekV4Compressor:
      return "DeepseekV4Compressor";
    case OpId::kDeepseekV4Moe:
      return "DeepseekV4Moe";
    case OpId::kReshapeAndCacheFp8:
      return "ReshapeAndCacheFp8";
    case OpId::kMoeGateUpSwiGLUGrouped:
      return "MoeGateUpSwiGLUGrouped";
    case OpId::kFusedNormRope:
      return "FusedNormRope";
    case OpId::kMoeGroupedGemmBf16GateUpSilu:
      return "MoeGroupedGemmBf16GateUpSilu";
    case OpId::kMoeGroupedGemmBf16GateUpSiluNative:
      return "MoeGroupedGemmBf16GateUpSiluNative";
    case OpId::kMoeGroupedGemmBf16Weighted:
      return "MoeGroupedGemmBf16Weighted";
    case OpId::kMoeCombinePreweighted:
      return "MoeCombinePreweighted";
    case OpId::kLaguna:
      return "Laguna";
    case OpId::kMarlinDenseGemm:
      return "MarlinDenseGemm";
    case OpId::kMiniMaxH3:
      return "MiniMaxH3";
    // Absorbed from origin/main, named in enum order. This exhaustive switch IS
    // the drift guard (0541cbeaa), and it carries no `default`, so merging main
    // is precisely when it is supposed to fire.
    case OpId::kAttentionDenseFa2:
      return "AttentionDenseFa2";
    case OpId::kMatmulFp8CublasLtAlphaVec:
      return "MatmulFp8CublasLtAlphaVec";
    case OpId::kMamba2ChunkScan:
      return "Mamba2ChunkScan";
    case OpId::kMamba2StateUpdate:
      return "Mamba2StateUpdate";
    case OpId::kRmsNormGatedGroup:
      return "RmsNormGatedGroup";
    case OpId::kLtx2:
      return "Ltx2";
    case OpId::kLtx2Vae:
      return "Ltx2Vae";
    case OpId::kQwen4ExpPleConv:
      return "Qwen4ExpPleConv";
    case OpId::kRmsNormGroup:
      return "RmsNormGroup";
    case OpId::kQwen4ExpPleGate:
      return "Qwen4ExpPleGate";
    case OpId::kQwen4ExpGatedResidual:
      return "Qwen4ExpGatedResidual";
    case OpId::kQwen4ExpGatedResidualWriteBack:
      return "Qwen4ExpGatedResidualWriteBack";
    case OpId::kQwen4ExpQsaCompress:
      return "Qwen4ExpQsaCompress";
    case OpId::kGlm5NextKpoolCompress:
      return "Glm5NextKpoolCompress";
    case OpId::kGlm5NextKpoolSelect:
      return "Glm5NextKpoolSelect";
    case OpId::kQwen4ExpQsaGatherAttention:
      return "Qwen4ExpQsaGatherAttention";
    case OpId::kConv2d:
      return "Conv2d";
    case OpId::kDepthwiseConv1d:
      return "DepthwiseConv1d";
    case OpId::kConv1d:
      return "Conv1d";
    case OpId::kConvTranspose1d:
      return "ConvTranspose1d";
    case OpId::kAttentionRelPos:
      return "AttentionRelPos";
    case OpId::kQuantFp8Group:
      return "QuantFp8Group";
    case OpId::kMatmulFp8BlockScaled:
      return "MatmulFp8BlockScaled";
    case OpId::kConv3d:
      return "Conv3d";
    case OpId::kExl3HadR128:
      return "Exl3HadR128";
    case OpId::kExl3Gemm:
      return "Exl3Gemm";
    case OpId::kExl3GroupedLinear:
      return "Exl3GroupedLinear";
    case OpId::kExl3GroupedW8A8:
      return "Exl3GroupedW8A8";
    case OpId::kExl3ReconstructGemm:
      return "Exl3ReconstructGemm";
    case OpId::kExl3MoeMlp:
      return "Exl3MoeMlp";
    case OpId::kKeepQuantDecode:
      return "KeepQuantDecode";
    case OpId::kCount:
      break;
    case OpId::kCopy:
      return "Copy";
  }
  return "unknown";
}

void* Resolve(OpId op, DeviceType device, Slot& slot) {
  ProviderCaps caps = GetDeviceProviderCaps(device);
  const OpProvider* chosen = Choose(slot, caps, nullptr);
  // MISS: on a unified-memory accelerator, install the CPU reference tier and
  // re-select. The native kernel — if one existed — would already have been
  // chosen above (priority >= 0 beats the tier's negative priority), so this
  // path is reached ONLY when the device genuinely lacks a native kernel.
  if (chosen == nullptr && MaybeInstallReferenceTier(op, device)) {
    chosen = Choose(slot, caps, nullptr);
  }
  Announce(op, device, slot, chosen, caps);
  if (chosen == nullptr) {
    slot.fallbacks.fetch_add(1, std::memory_order_relaxed);
    slot.resolved_none.store(true, std::memory_order_relaxed);
    // Refuse BY NAME. The integers stay for grep-ability, but a reader must not
    // have to count enumerators in include/vt/ops.h to learn what was refused —
    // nor open this file to learn that a portable fallback exists and was
    // withheld. The trailing clause says which of the tier's preconditions the
    // device failed, which is the sentence #844 asked for.
    VT_CHECK(false, std::string("no kernel for op ") + OpNameImpl(op) + " (id " +
                        std::to_string(static_cast<int>(op)) + ") on device " +
                        DeviceTypeName(device) + " (type " +
                        std::to_string(static_cast<int>(device)) + "), and " +
                        ReferenceTierRefusalReason(op, device));
    return nullptr;
  }
  // Reference-tier accounting: count it, and warn LOUDLY exactly once per
  // (op, device) so "this backend ran op X on the portable tier" is never silent.
  if (std::strcmp(chosen->name, kReferenceProviderName) == 0) {
    slot.ref_selected.store(true, std::memory_order_relaxed);
    if (!slot.ref_announced.exchange(true, std::memory_order_relaxed)) {
      // NOT "correct but slow". Those three words asserted a property instead of
      // naming it, and #844 / docs/USAGE.md both quote them as what misled a
      // reader past a SIGSEGV. What makes this dispatch valid is one checkable
      // fact — the backend reports its device memory host-addressable — so the
      // line states that fact and lets the reader verify it.
      std::fprintf(stderr,
                   "[vt reference-tier] op=%s device=%s has NO native kernel; "
                   "running the PORTABLE CPU host kernel, which this backend "
                   "permits because it reports its device memory "
                   "host-addressable. It is SLOW: this run is not a performance "
                   "measurement\n",
                   OpNameImpl(op), DeviceTypeName(device));
    }
  }
  slot.last_selected.store(chosen->name, std::memory_order_relaxed);
  slot.selected.store(chosen->fn, std::memory_order_relaxed);
  return chosen->fn;
}

void InvalidateAll() {
  Slot* t = Table();
  for (size_t i = 0; i < kOpCount * kNumDeviceTypes; ++i) {
    t[i].selected.store(nullptr, std::memory_order_relaxed);
    t[i].resolved_none.store(false, std::memory_order_relaxed);
    t[i].ref_selected.store(false, std::memory_order_relaxed);
    t[i].native_registered.store(-1, std::memory_order_relaxed);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
void SetDeviceProviderCaps(DeviceType device, const ProviderCaps& caps) {
  VT_CHECK(static_cast<size_t>(device) < kNumDeviceTypes, "invalid device type");
  {
    CapsTable& t = Caps();
    std::lock_guard<std::mutex> lock(t.mu);
    t.caps[static_cast<size_t>(device)] = caps;
    t.caps[static_cast<size_t>(device)].device = device;
  }
  // A predicate may have declined against unprobed capabilities; drop the cache
  // so the next dispatch re-selects against the real record.
  InvalidateAll();
}

ProviderCaps GetDeviceProviderCaps(DeviceType device) {
  VT_CHECK(static_cast<size_t>(device) < kNumDeviceTypes, "invalid device type");
  CapsTable& t = Caps();
  std::lock_guard<std::mutex> lock(t.mu);
  ProviderCaps c = t.caps[static_cast<size_t>(device)];
  c.device = device;
  return c;
}

void RegisterOpProvider(OpId op, DeviceType device, const OpProvider& provider) {
  // Table fill only; NEVER throws (registrars run before main, where a throw has
  // no receiver) — the RegisterArchTactic contract, cuda_arch_tactics.cu:126-131.
  if (static_cast<size_t>(op) >= kOpCount) return;
  if (static_cast<size_t>(device) >= kNumDeviceTypes) return;
  if (provider.fn == nullptr || provider.name == nullptr) return;
  Slot& slot = At(op, device);
  if (slot.count >= kMaxProvidersPerOp) return;
  for (int i = 0; i < slot.count; ++i) {
    // A duplicate name would make the order non-total. First registration wins,
    // which is deterministic for the only case that can legitimately occur (a
    // single provider registering twice); two DIFFERENT providers sharing a name
    // is a naming bug the test suite catches.
    if (std::strcmp(slot.providers[i].name, provider.name) == 0) return;
  }
  slot.providers[slot.count++] = provider;
  slot.selected.store(nullptr, std::memory_order_relaxed);
  slot.resolved_none.store(false, std::memory_order_relaxed);
  slot.ref_selected.store(false, std::memory_order_relaxed);
  slot.native_registered.store(-1, std::memory_order_relaxed);
}

void RegisterOp(OpId op, DeviceType device, void* fn) {
  VT_CHECK(static_cast<size_t>(op) < kOpCount, "invalid op id");
  VT_CHECK(static_cast<size_t>(device) < kNumDeviceTypes, "invalid device type");
  OpProvider p;
  p.name = kNativeProviderName;
  p.priority = 0;
  p.supports = nullptr;
  p.fn = fn;
  RegisterOpProvider(op, device, p);
}

const char* OpName(OpId op) { return OpNameImpl(op); }

void TraceOpTensors(OpId op, const Queue& q, std::initializer_list<const Tensor*> tensors) {
  if (!TraceEnabled()) return;
  nlohmann::json args = nlohmann::json::array();
  for (const auto* t : tensors) {
    if (!t) { args.push_back(nullptr); continue; }
    args.push_back({{"dtype", Name(t->dtype)}, {"shape", std::vector<int64_t>(t->shape, t->shape + t->rank)},
                    {"stride", std::vector<int64_t>(t->stride, t->stride + t->rank)}});
  }
  const nlohmann::json row = {{"event", "tensor_arguments"}, {"op", OpName(op)},
      {"device", DeviceTypeName(q.device.type)}, {"device_index", q.device.index},
      {"queue_id", q.id}, {"tensors", args}};
  auto& trace = TraceState();
  std::lock_guard<std::mutex> lock(trace.mutex);
  VT_CHECK(trace.file.is_open(), "cannot open VT_OP_PROVIDER_TRACE");
  trace.file << row.dump() << '\n';
  trace.file.flush();
  VT_CHECK(trace.file.good(), "cannot write VT_OP_PROVIDER_TRACE");
}

void* GetOp(OpId op, DeviceType device) {
  Slot& slot = At(op, device);
  void* fn = slot.selected.load(std::memory_order_relaxed);
  if (fn == nullptr) fn = Resolve(op, device, slot);
  // The portable reference tier is a HOST kernel about to read and write DEVICE
  // memory directly (it is only ever installed on unified memory, op_provider.h
  // SAFETY). A backend that defers submission must therefore drain before it
  // runs, or the CPU reads bytes the GPU has not written yet. No-op on every
  // eagerly-submitting backend, and skipped entirely on the common path.
  if (slot.ref_selected.load(std::memory_order_relaxed)) {
    RefTierHits().fetch_add(1, std::memory_order_relaxed);
    Backend* b = TryGetBackend(device);
    if (b != nullptr) b->FlushPending();
  }
  if (CallStatsFlag().load(std::memory_order_relaxed) || TraceEnabled()) {
    slot.selections.fetch_add(1, std::memory_order_relaxed);
  }
  TraceSelection(op, device, slot.last_selected.load(std::memory_order_relaxed), "selection");
  return fn;
}

namespace {

// The one body behind both spellings of a fallback resolution. `count` is the
// ONLY difference: the resolution order, the reference-tier install, the drain
// and every throw are shared, so the two spellings can never answer differently
// ([#1584](https://github.com/mudler/vllm.cpp/issues/1584)).
void* ResolveFallback(OpId op, DeviceType device, const char* declining_provider, bool count) {
  VT_CHECK(declining_provider != nullptr, "op provider fallback requires a provider name");
  Slot& slot = At(op, device);
  const OpProvider* floor = nullptr;
  for (int i = 0; i < slot.count; ++i) {
    if (std::strcmp(slot.providers[i].name, declining_provider) == 0) {
      floor = &slot.providers[i];
      break;
    }
  }
  VT_CHECK(floor != nullptr, std::string("declining provider '") + declining_provider +
                                 "' is not registered for this op/device");
  const ProviderCaps caps = GetDeviceProviderCaps(device);
  const OpProvider* next = Choose(slot, caps, floor);
  // NOTHING BELOW: install the portable reference tier and re-select. Resolve()
  // installs it only on a GetOp MISS, so an op whose device HAS a native kernel
  // never gets one — and a native kernel that declines per-call then had nothing
  // to fall back to and threw. That is the wrong answer twice over: the tier is
  // the whole reason a declining kernel is preferable to a throwing one
  // (op_provider.h § DECLINE-AND-FALL-BACK), and before the native kernel existed
  // this very shape resolved to the tier and worked. `floor` stays valid across
  // the install because providers[] is a fixed array appended in place.
  if (next == nullptr && MaybeInstallReferenceTier(op, device)) {
    next = Choose(slot, caps, floor);
  }
  // BEFORE the check below and not after: a decline that finds nothing underneath
  // it is still a decline, and tests/vt/test_op_provider.cpp asserts that count.
  if (count) slot.declines.fetch_add(1, std::memory_order_relaxed);
  VT_CHECK(next != nullptr,
           std::string("provider '") + declining_provider + "' declined op " +
               std::to_string(static_cast<int>(op)) + " on device type " +
               std::to_string(static_cast<int>(device)) + " and no provider is below it");
  // The reference tier is a HOST kernel about to read and write DEVICE memory,
  // exactly as in GetOp above — and GetOp's drain is keyed on the SELECTED
  // provider, which on this path is the declining NATIVE one, so it never fires.
  // Without this a backend that defers submission (Vulkan batches command
  // buffers) hands the host kernel bytes the device has not written yet, and it
  // does so SILENTLY.
  if (std::strcmp(next->name, kReferenceProviderName) == 0) {
    if (count) RefTierHits().fetch_add(1, std::memory_order_relaxed);
    Backend* b = TryGetBackend(device);
    if (b != nullptr) b->FlushPending();
  }
  if (count) TraceSelection(op, device, next->name, "fallback_selection");
  return next->fn;
}

}  // namespace

void* GetOpFallback(OpId op, DeviceType device, const char* declining_provider) {
  return ResolveFallback(op, device, declining_provider, /*count=*/true);
}

void* GetOpFallbackUncounted(OpId op, DeviceType device, const char* declining_provider) {
  return ResolveFallback(op, device, declining_provider, /*count=*/false);
}

void NoteOpDecline(OpId op, DeviceType device) {
  At(op, device).declines.fetch_add(1, std::memory_order_relaxed);
}

bool OpRegistered(OpId op, DeviceType device) {
  // Meaning (unchanged, and load-bearing for the fused-recipe fast-realization
  // ladder and the cross-device harness): is there a NATIVE kernel for
  // (op, device)? The portable reference tier is a FALLBACK, not a native
  // kernel, so it is deliberately EXCLUDED here — otherwise a unified accelerator
  // would report every op as "registered" the moment its fallback installed, and
  // the ladder would stop choosing its portable composite path. It cannot use the
  // `selected` cache because `selected` can legitimately hold a reference-tier fn
  // (a real GetOp answer); it keeps its own native-only memo instead.
  Slot& slot = At(op, device);
  const int8_t cached = slot.native_registered.load(std::memory_order_relaxed);
  if (cached >= 0) return cached == 1;
  const ProviderCaps caps = GetDeviceProviderCaps(device);
  bool has_native = false;
  for (int i = 0; i < slot.count; ++i) {
    const OpProvider& p = slot.providers[i];
    if (std::strcmp(p.name, kReferenceProviderName) == 0) continue;  // fallback, not native
    if (Supported(p, caps)) {
      has_native = true;
      break;
    }
  }
  slot.native_registered.store(has_native ? 1 : 0, std::memory_order_relaxed);
  return has_native;
}

int OpProviderCount(OpId op, DeviceType device) { return At(op, device).count; }

const char* OpProviderNameAt(OpId op, DeviceType device, int i) {
  Slot& slot = At(op, device);
  if (i < 0 || i >= slot.count) return nullptr;
  // Report in SELECTION order, not storage order — the observable order is the
  // one the seam promises, so a test can assert it directly.
  const OpProvider* prev = nullptr;
  for (int rank = 0; rank <= i; ++rank) {
    const OpProvider* best = nullptr;
    for (int j = 0; j < slot.count; ++j) {
      const OpProvider& p = slot.providers[j];
      if (prev != nullptr && !Better(*prev, p)) continue;
      if (best == nullptr || Better(p, *best)) best = &p;
    }
    if (best == nullptr) return nullptr;
    prev = best;
  }
  return prev->name;
}

OpProviderStats GetOpProviderStats(OpId op, DeviceType device) {
  Slot& slot = At(op, device);
  OpProviderStats s;
  s.last_selected = slot.last_selected.load(std::memory_order_relaxed);
  s.selections = slot.selections.load(std::memory_order_relaxed);
  s.declines = slot.declines.load(std::memory_order_relaxed);
  s.fallbacks = slot.fallbacks.load(std::memory_order_relaxed);
  return s;
}

void EnableOpProviderCallStats(bool on) {
  CallStatsFlag().store(on, std::memory_order_relaxed);
}

void ResetOpProviderStats(OpId op, DeviceType device) {
  Slot& slot = At(op, device);
  slot.selections.store(0, std::memory_order_relaxed);
  slot.declines.store(0, std::memory_order_relaxed);
  slot.fallbacks.store(0, std::memory_order_relaxed);
}

void DisableOpProvider(const char* name, bool disabled) {
  if (name == nullptr) return;
  DisableList& d = Disabled();
  {
    std::lock_guard<std::mutex> lock(d.mu);
    EnsureEnvParsedLocked(d);
    const size_t n = std::strlen(name);
    if (disabled) {
      AddDisabledLocked(d, name, name + n);
    } else {
      for (int i = 0; i < d.count; ++i) {
        if (std::strcmp(d.names[i], name) == 0) {
          for (int j = i; j + 1 < d.count; ++j) {
            std::memcpy(d.names[j], d.names[j + 1], kMaxNameLen + 1);
          }
          --d.count;
          break;
        }
      }
    }
  }
  InvalidateAll();
}

bool OpProviderDisabled(const char* name) {
  if (name == nullptr) return false;
  DisableList& d = Disabled();
  if (d.live_count.load(std::memory_order_relaxed) == 0) return false;  // lock-free
  std::lock_guard<std::mutex> lock(d.mu);
  EnsureEnvParsedLocked(d);
  for (int i = 0; i < d.count; ++i) {
    if (std::strcmp(d.names[i], name) == 0) return true;
  }
  return false;
}

// --- Portable reference tier (S5) ------------------------------------------
bool ReferenceTierEligible(DeviceType device) {
  // The CPU is the SOURCE of the reference kernels, never a fallback target
  // (falling back to itself is a no-op at best and self-reference at worst).
  if (device == DeviceType::kCPU) return false;
  // THE SAFETY GATE. A CPU kernel dereferences pointers that came out of
  // `Backend::Alloc`, so the question is whether the HOST MAY DEREFERENCE THEM
  // — `DeviceMemoryIsHostAddressable()` — and not whether the two address spaces
  // happen to sit on the same physical RAM. Gate on the property of the ACTUAL
  // registered backend, not on DeviceType. A device with no backend in this
  // build is trivially ineligible.
  //
  // THIS USED TO ASK `UnifiedMemory()`, AND THAT COST TWO CRASHES (#844, #1435).
  // The two properties are not the same, and include/vt/backend.h says so beside
  // the narrow one: CUDA on GB10 reports unified memory because host and device
  // address the same physical RAM, yet a plain `cudaMalloc` pointer is still not
  // host-dereferenceable. `CudaBackend::Alloc` calls exactly that, so every CUDA
  // op without a native kernel installed the CPU host kernel over device
  // pointers and the process took SIGSEGV — `vt::QuantFp8Static` on sm_110
  // (#960) and `vt::MatmulFp8BlockScaled` on a CUTLASS-less build (#1435).
  //
  // Metal (StorageModeShared) and ROCm (managed or pageable-access integrated
  // allocations) answer the narrow predicate with the value they already report
  // for unified memory, so neither changes. VULKAN WIDENS, and safely: it
  // answers the narrow predicate `true` unconditionally, while its
  // `UnifiedMemory()` is false when no HOST_VISIBLE|HOST_COHERENT|DEVICE_LOCAL
  // memory type exists. On such a device the tier was WITHHELD before and is
  // installed now. That is sound rather than lucky — `VulkanContext` falls back
  // to the host flags alone and then VT_CHECKs that a host-visible, host-coherent
  // type was found, so every allocation it hands out is host memory the GPU also
  // reads. A backend added later gets the safe default, which is `false`.
  Backend* b = TryGetBackend(device);
  return b != nullptr && b->DeviceMemoryIsHostAddressable();
}

int RegisterReferenceTier(DeviceType target) {
  if (!ReferenceTierEligible(target)) return 0;
  int installed = 0;
  for (size_t o = 0; o < kOpCount; ++o) {
    const OpId op = static_cast<OpId>(o);
    // Skip ops the target already serves natively — the tier is a fallback for
    // MISSING kernels only, and installing under a native provider is wasted
    // capacity (the native one wins by priority regardless).
    if (OpRegistered(op, target)) continue;
    if (MaybeInstallReferenceTier(op, target)) ++installed;
  }
  return installed;
}

unsigned long long GetReferenceTierHits() {
  return RefTierHits().load(std::memory_order_relaxed);
}

}  // namespace vt
