// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
//
// The acceleration-provider seam's own gate (BACKEND-ACCEL-PROVIDER,
// include/vt/op_provider.h, .agents/specs/metal-mlx-reuse-study.md §6).
//
// WHAT IS UNDER TEST, precisely. Before this seam, `src/vt/ops.cpp` held one
// `void*` per (OpId, DeviceType) and `RegisterOp` overwrote it silently. Two
// providers of one op on one device therefore resolved by STATIC-INIT ORDER
// ACROSS TRANSLATION UNITS, which is unspecified — a nondeterministic BUILD.
// These cases assert the three properties that replace it:
//   1. DETERMINISM — the same provider set selects the same winner regardless of
//      the order the registrars ran. Proved by registering the identical set in
//      OPPOSITE orders on two slots and requiring the same answer.
//   2. DECLINE-AND-FALL-BACK — a provider that cannot serve a call forwards to
//      the next provider DOWN, with the ~70 op entry points unmodified.
//   3. OBSERVABILITY — which provider ran is answerable from the process, not
//      inferred from a green assertion.
//
// Synthetic registrations use an unused device slot selected by TestDevice().
// Production backend registrations remain available for the integration case.
#include <doctest/doctest.h>

#include <cstring>
#include <set>
#include <string>

#include "vt/backend.h"
#include "vt/op_provider.h"
#include "vt/ops.h"

namespace {

using vt::DeviceType;
using vt::OpId;
using vt::OpProvider;
using vt::ProviderCaps;

// Synthetic registrations need an empty device slot. XPU now has real kernels
// in an XPU build; choose another unused slot without disturbing any backend.
DeviceType TestDevice() {
  static const DeviceType device = [] {
    for (auto d : {DeviceType::kXPU, DeviceType::kMETAL, DeviceType::kVULKAN,
                   DeviceType::kTENSTORRENT, DeviceType::kROCM, DeviceType::kCUDA}) {
      if (vt::TryGetBackend(d)) continue;
      bool empty = true;
      for (int op = 0; op < static_cast<int>(OpId::kCount); ++op)
        if (vt::OpProviderCount(static_cast<OpId>(op), d)) empty = false;
      if (empty) return d;
    }
    throw std::runtime_error("provider test requires one unused device registry slot");
  }();
  return device;
}


// Distinct kernel bodies so the selected pointer identifies the provider. Their
// signature is irrelevant to the seam (`fn` is type-erased exactly as the old
// table's `void*` was); what matters is that the addresses differ.
void KernelA() {}
void KernelB() {}
void KernelC() {}

void* Fn(void (*f)()) { return reinterpret_cast<void*>(f); }

OpProvider P(const char* name, int priority, void* fn,
             vt::OpProviderSupportsFn supports = nullptr) {
  OpProvider p;
  p.name = name;
  p.priority = priority;
  p.supports = supports;
  p.fn = fn;
  return p;
}

bool NeverSupports(const ProviderCaps&) { return false; }
bool NeedsComputeMajor9(const ProviderCaps& c) { return c.valid && c.compute_major >= 9; }

}  // namespace

// ---------------------------------------------------------------------------
// 1. DETERMINISM — the property whose absence was the bug.
// ---------------------------------------------------------------------------
TEST_CASE("op provider: selection is independent of registration order") {
  // Same three providers, registered in OPPOSITE orders on two different ops.
  // Under the old flat table this was exactly the case that resolved by whichever
  // static initializer happened to run last.
  vt::RegisterOpProvider(OpId::kAdd, TestDevice(), P("vt-native", 0, Fn(&KernelA)));
  vt::RegisterOpProvider(OpId::kAdd, TestDevice(), P("accel-lo", 5, Fn(&KernelB)));
  vt::RegisterOpProvider(OpId::kAdd, TestDevice(), P("accel-hi", 9, Fn(&KernelC)));

  vt::RegisterOpProvider(OpId::kRelu, TestDevice(), P("accel-hi", 9, Fn(&KernelC)));
  vt::RegisterOpProvider(OpId::kRelu, TestDevice(), P("accel-lo", 5, Fn(&KernelB)));
  vt::RegisterOpProvider(OpId::kRelu, TestDevice(), P("vt-native", 0, Fn(&KernelA)));

  CHECK(vt::GetOp(OpId::kAdd, TestDevice()) == Fn(&KernelC));
  CHECK(vt::GetOp(OpId::kRelu, TestDevice()) == Fn(&KernelC));

  // ... and the whole ORDER, not just the winner, is the same both ways.
  for (int i = 0; i < 3; ++i) {
    const char* a = vt::OpProviderNameAt(OpId::kAdd, TestDevice(), i);
    const char* b = vt::OpProviderNameAt(OpId::kRelu, TestDevice(), i);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    CAPTURE(i);
    CHECK(std::string(a) == std::string(b));
  }
  CHECK(std::string(vt::OpProviderNameAt(OpId::kAdd, TestDevice(), 0)) == "accel-hi");
  CHECK(std::string(vt::OpProviderNameAt(OpId::kAdd, TestDevice(), 1)) == "accel-lo");
  CHECK(std::string(vt::OpProviderNameAt(OpId::kAdd, TestDevice(), 2)) == "vt-native");
  CHECK(vt::OpProviderNameAt(OpId::kAdd, TestDevice(), 3) == nullptr);
}

TEST_CASE("op provider: equal priority breaks by name, not by registration order") {
  vt::RegisterOpProvider(OpId::kLayerNorm, TestDevice(), P("zzz-provider", 3, Fn(&KernelA)));
  vt::RegisterOpProvider(OpId::kLayerNorm, TestDevice(), P("aaa-provider", 3, Fn(&KernelB)));
  CHECK(vt::GetOp(OpId::kLayerNorm, TestDevice()) == Fn(&KernelB));
  CHECK(std::string(vt::OpProviderNameAt(OpId::kLayerNorm, TestDevice(), 0)) == "aaa-provider");
}

TEST_CASE("op provider: a duplicate name is rejected so the order stays total") {
  vt::RegisterOpProvider(OpId::kL2Norm, TestDevice(), P("dup", 1, Fn(&KernelA)));
  vt::RegisterOpProvider(OpId::kL2Norm, TestDevice(), P("dup", 7, Fn(&KernelB)));
  CHECK(vt::OpProviderCount(OpId::kL2Norm, TestDevice()) == 1);
  CHECK(vt::GetOp(OpId::kL2Norm, TestDevice()) == Fn(&KernelA));
}

// ---------------------------------------------------------------------------
// 2. CAPABILITY PREDICATE + DECLINE-AND-FALL-BACK (the two fallback axes).
// ---------------------------------------------------------------------------
TEST_CASE("op provider: a provider whose capability predicate fails is skipped") {
  vt::RegisterOpProvider(OpId::kEmbedding, TestDevice(), P("vt-native", 0, Fn(&KernelA)));
  vt::RegisterOpProvider(OpId::kEmbedding, TestDevice(),
                         P("never", 100, Fn(&KernelB), &NeverSupports));
  CHECK(vt::GetOp(OpId::kEmbedding, TestDevice()) == Fn(&KernelA));
  CHECK(std::string(vt::GetOpProviderStats(OpId::kEmbedding, TestDevice()).last_selected) ==
        "vt-native");
}

TEST_CASE("op provider: capability predicates re-resolve when the device caps are published") {
  vt::RegisterOpProvider(OpId::kQkvSplit, TestDevice(), P("vt-native", 0, Fn(&KernelA)));
  vt::RegisterOpProvider(OpId::kQkvSplit, TestDevice(),
                         P("sm9-only", 50, Fn(&KernelB), &NeedsComputeMajor9));
  // Unprobed caps: the accelerator must decline rather than guess.
  CHECK(vt::GetOp(OpId::kQkvSplit, TestDevice()) == Fn(&KernelA));

  ProviderCaps caps;
  caps.valid = true;
  caps.compute_major = 9;
  vt::SetDeviceProviderCaps(TestDevice(), caps);
  CHECK(vt::GetOp(OpId::kQkvSplit, TestDevice()) == Fn(&KernelB));

  ProviderCaps old;  // restore the unprobed record for the rest of the file
  vt::SetDeviceProviderCaps(TestDevice(), old);
  CHECK(vt::GetOp(OpId::kQkvSplit, TestDevice()) == Fn(&KernelA));
}

TEST_CASE("op provider: a provider declines a call and falls back to the one below it") {
  // This is the SECOND fallback axis — per-CALL refusal (a shape or dtype the
  // accelerator does not handle), which `supports()` cannot express because
  // `GetOp` has no shape to inspect. The declining kernel asks for the provider
  // below itself and forwards. That is what keeps the ~70 op entry points in
  // src/vt/ops.cpp free of any edit: the decline lives inside the kernel.
  vt::RegisterOpProvider(OpId::kMulColVecF32, TestDevice(), P("vt-native", 0, Fn(&KernelA)));
  vt::RegisterOpProvider(OpId::kMulColVecF32, TestDevice(), P("accel", 10, Fn(&KernelB)));
  vt::RegisterOpProvider(OpId::kMulColVecF32, TestDevice(), P("accel-mid", 5, Fn(&KernelC)));

  CHECK(vt::GetOp(OpId::kMulColVecF32, TestDevice()) == Fn(&KernelB));
  // "accel" declines -> the next one DOWN in the deterministic order.
  CHECK(vt::GetOpFallback(OpId::kMulColVecF32, TestDevice(), "accel") == Fn(&KernelC));
  // ... and if that one declines too, the native kernel.
  CHECK(vt::GetOpFallback(OpId::kMulColVecF32, TestDevice(), "accel-mid") == Fn(&KernelA));
  // Nothing below the native kernel: a decline there is a hard error, never a
  // silent no-op.
  CHECK_THROWS(vt::GetOpFallback(OpId::kMulColVecF32, TestDevice(), "vt-native"));
  CHECK(vt::GetOpProviderStats(OpId::kMulColVecF32, TestDevice()).declines >= 3);
}

// ---------------------------------------------------------------------------
// 2b. THE CACHED-FALLBACK PATTERN, and whether its COUNT is exact
//     ([#1584](https://github.com/mudler/vllm.cpp/issues/1584)).
// ---------------------------------------------------------------------------
// `op_provider.h` prescribes this shape for a SHAPE-GATED provider -- one that
// declines on the HOT path rather than rarely: resolve the fallback ONCE into a
// function-local static and count each decline with `NoteOpDecline`, so a decode
// run that declines ~21,500 times pays no provider-stack walk per decline. The
// header states that `OpProviderStats::declines` "stays exact" under that shape.
// This case is what makes the statement checkable, because `declines` is the
// ONLY thing that separates a served call from a forwarded one for a provider
// that is always the SELECTED one.
//
// Built to be red in a FULL run as well as under `-tc=`. The defect it pins is
// one extra increment per STATIC, not per process, so a case that shares its
// static with an earlier case reads correct and hides it -- which is exactly how
// it hid in tests/vt/test_ops_attention_cross.cpp until a single-case filter was
// run. This slot and this static are reachable from this case and nothing else,
// and the reset opens the counted window BEFORE the static resolves.
constexpr OpId kCachedFallbackOp = OpId::kMoeCombine;
constexpr const char* kShapeGated = "shape-gated";

void* CachedFallbackPtr() {
  static void* f = vt::GetOpFallbackUncounted(kCachedFallbackOp, TestDevice(), kShapeGated);
  return f;
}

TEST_CASE("op provider: the cached fallback pattern counts exactly one decline per decline") {
  vt::RegisterOpProvider(kCachedFallbackOp, TestDevice(), P("vt-native", 0, Fn(&KernelA)));
  vt::RegisterOpProvider(kCachedFallbackOp, TestDevice(), P(kShapeGated, 10, Fn(&KernelB)));
  REQUIRE(vt::GetOp(kCachedFallbackOp, TestDevice()) == Fn(&KernelB));

  vt::ResetOpProviderStats(kCachedFallbackOp, TestDevice());
  constexpr int kDeclines = 3;
  for (int i = 0; i < kDeclines; ++i) {
    // The order the two live providers use: count, then forward through the
    // cached pointer (cuda_attention_cross.cu, metal_mlx_provider.mm).
    vt::NoteOpDecline(kCachedFallbackOp, TestDevice());
    CHECK(CachedFallbackPtr() == Fn(&KernelA));
  }
  // EXACT, and two-sided. 4 is the #1584 double count of the first decline; 1 or
  // 0 is a decline that went unrecorded, which is the failure every
  // `declines == 0` assertion in this tree is built to catch.
  CHECK(vt::GetOpProviderStats(kCachedFallbackOp, TestDevice()).declines == kDeclines);
}

TEST_CASE("op provider: GetOpFallbackUncounted resolves the same provider without counting") {
  const OpId op = OpId::kAttnGateSplit;
  vt::RegisterOpProvider(op, TestDevice(), P("vt-native", 0, Fn(&KernelA)));
  vt::RegisterOpProvider(op, TestDevice(), P("accel", 10, Fn(&KernelB)));
  vt::ResetOpProviderStats(op, TestDevice());

  // Same answer as the counting spelling, and no increment.
  CHECK(vt::GetOpFallbackUncounted(op, TestDevice(), "accel") == Fn(&KernelA));
  CHECK(vt::GetOpProviderStats(op, TestDevice()).declines == 0);
  // The counting spelling still counts. This is what keeps the five per-call
  // callers that never cache -- vulkan_ops.cpp:950/1067/1488/1509 and
  // tenstorrent_ops.cpp:1341 -- reporting exactly what they report today.
  CHECK(vt::GetOpFallback(op, TestDevice(), "accel") == Fn(&KernelA));
  CHECK(vt::GetOpProviderStats(op, TestDevice()).declines == 1);
  // Nothing below the floor still throws in BOTH spellings, and the counting one
  // still counts a decline that throws -- the position of the increment relative
  // to that check is load-bearing for the `>= 3` assertion above.
  CHECK_THROWS(vt::GetOpFallbackUncounted(op, TestDevice(), "vt-native"));
  CHECK(vt::GetOpProviderStats(op, TestDevice()).declines == 1);
  CHECK_THROWS(vt::GetOpFallback(op, TestDevice(), "vt-native"));
  CHECK(vt::GetOpProviderStats(op, TestDevice()).declines == 2);
}

// ---------------------------------------------------------------------------
// 3. OBSERVABILITY — "did the accelerator actually run?" must be answerable.
// ---------------------------------------------------------------------------
TEST_CASE("op provider: selection is observable through the stats counters") {
  vt::RegisterOpProvider(OpId::kSigmoidGateBf16, TestDevice(), P("vt-native", 0, Fn(&KernelA)));
  vt::RegisterOpProvider(OpId::kSigmoidGateBf16, TestDevice(), P("accel", 10, Fn(&KernelB)));
  vt::ResetOpProviderStats(OpId::kSigmoidGateBf16, TestDevice());
  vt::EnableOpProviderCallStats(true);
  for (int i = 0; i < 5; ++i) (void)vt::GetOp(OpId::kSigmoidGateBf16, TestDevice());
  vt::EnableOpProviderCallStats(false);

  const auto s = vt::GetOpProviderStats(OpId::kSigmoidGateBf16, TestDevice());
  REQUIRE(s.last_selected != nullptr);
  CHECK(std::string(s.last_selected) == "accel");
  CHECK(s.selections == 5);
}

TEST_CASE("op provider: a provider can be disabled at run time for a same-binary A/B") {
  vt::RegisterOpProvider(OpId::kCastF32, TestDevice(), P("vt-native", 0, Fn(&KernelA)));
  vt::RegisterOpProvider(OpId::kCastF32, TestDevice(), P("ab-accel", 10, Fn(&KernelB)));
  CHECK(vt::GetOp(OpId::kCastF32, TestDevice()) == Fn(&KernelB));

  vt::DisableOpProvider("ab-accel", true);
  CHECK(vt::OpProviderDisabled("ab-accel"));
  CHECK(vt::GetOp(OpId::kCastF32, TestDevice()) == Fn(&KernelA));

  vt::DisableOpProvider("ab-accel", false);
  CHECK_FALSE(vt::OpProviderDisabled("ab-accel"));
  CHECK(vt::GetOp(OpId::kCastF32, TestDevice()) == Fn(&KernelB));
}

// ---------------------------------------------------------------------------
// 4. BEHAVIOUR PRESERVATION — every backend in the tree still registers through
//    plain RegisterOp() and must keep winning exactly as before.
// ---------------------------------------------------------------------------
TEST_CASE("op provider: RegisterOp is the priority-0 vt-native provider, unchanged") {
  vt::RegisterOp(OpId::kCastBf16, TestDevice(), Fn(&KernelA));
  CHECK(vt::OpProviderCount(OpId::kCastBf16, TestDevice()) == 1);
  CHECK(std::string(vt::OpProviderNameAt(OpId::kCastBf16, TestDevice(), 0)) ==
        vt::kNativeProviderName);
  CHECK(vt::OpRegistered(OpId::kCastBf16, TestDevice()));
  CHECK(vt::GetOp(OpId::kCastBf16, TestDevice()) == Fn(&KernelA));
}

TEST_CASE("op provider: an unrealized (op, device) still probes false and throws") {
  CHECK_FALSE(vt::OpRegistered(OpId::kPagedAttention, TestDevice()));
  CHECK_THROWS(vt::GetOp(OpId::kPagedAttention, TestDevice()));
  // The probe stays false after the throw (the negative resolution is memoized,
  // it is not a one-shot).
  CHECK_FALSE(vt::OpRegistered(OpId::kPagedAttention, TestDevice()));
}

// A refusal that says "op 30 on device type 4" makes the reader open the header
// and count enumerators. Every other refusal in this tree names what it refused;
// this one is the seam's own, so it is held to the same bar.
TEST_CASE("op provider: the refusal NAMES the op and the device, not their integers") {
  std::string msg;
  try {
    (void)vt::GetOp(OpId::kMlaPrefillAttention, TestDevice());
    FAIL("an unrealized (op, device) must refuse");
  } catch (const std::exception& e) {
    msg = e.what();
  }
  CHECK(msg.find("MlaPrefillAttention") != std::string::npos);
  CHECK(msg.find(vt::DeviceTypeName(TestDevice())) != std::string::npos);

  // vt::OpName is TOTAL over the enum: every op has a real name, and no two ops
  // share one. A missing case would compile to the "unknown" fallthrough and
  // hand every future reader the same unusable message this case exists to stop.
  std::set<std::string> seen;
  for (int i = 0; i < static_cast<int>(OpId::kCount); ++i) {
    CAPTURE(i);
    const char* name = vt::OpName(static_cast<OpId>(i));
    REQUIRE(name != nullptr);
    CHECK(std::string(name) != "unknown");
    CHECK(seen.insert(std::string(name)).second);
  }
}

TEST_CASE("op provider: the real CPU backend resolves to vt-native") {
  CHECK(vt::OpRegistered(OpId::kMatmul, DeviceType::kCPU));
  (void)vt::GetOp(OpId::kMatmul, DeviceType::kCPU);
  const auto s = vt::GetOpProviderStats(OpId::kMatmul, DeviceType::kCPU);
  REQUIRE(s.last_selected != nullptr);
  CHECK(std::string(s.last_selected) == vt::kNativeProviderName);
}
