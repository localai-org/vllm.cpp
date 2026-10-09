// Gates on the GCN-arch capability parse (include/vt/rocm/rocm_arch.h), the one
// piece of the ROCm skeleton that holds a decision rather than an API call.
//
// COMPILED IN EVERY BUILD, including a CPU-only build on a machine with no AMD
// GPU and no ROCm toolchain — that is the entire reason the parse lives in a
// plain header. The rest of the ROCm skeleton (rocm_backend.hip, the RmsNorm
// kernel, the platform TU) is gated on VLLM_CPP_HIP and cannot be exercised
// here; see tests/vt/test_rocm_backend.cpp.
//
// The cases are upstream's own worked examples, from the docstring of
// vllm/platforms/rocm.py:223-291 `_capability_from_gcn_arch`, plus the three
// boards offered on issue #41 (gfx1100, gfx1103, gfx1151).
#include <array>
#include <string>

#include <doctest/doctest.h>

#include "vt/rocm/rocm_arch.h"
#include "vt/rocm/rocm_attn_wmma_arch.h"
#include "vt/rocm/rocm_quant_wmma_arch.h"
#include "vt/rocm/rocm_skinny_gemm_arch.h"

using vt::rocm::CapabilityFromGcnArch;

TEST_CASE("SharedK attention WMMA admits measured gfx1100 and retains gfx12") {
  using vt::rocm::GcnArchNameHasSharedKAttentionWmma;
  for (const char* arch : {"gfx1100", "gfx1100:xnack-", "gfx1100:sramecc+:xnack-",
                           "gfx1101", "gfx1101:xnack-",
                           "gfx1200", "gfx1201", "gfx1201:xnack-"}) {
    CHECK_MESSAGE(GcnArchNameHasSharedKAttentionWmma(arch), arch);
  }
  for (const char* arch : {"", "gfx110", "gfx11000", "gfx1100junk", "foogfx1100",
                           "gfx11010", "gfx1102", "gfx1103", "gfx1151", "gfx942",
                           "gfx1202", "gfx12010"}) {
    CHECK_FALSE_MESSAGE(GcnArchNameHasSharedKAttentionWmma(arch), arch);
  }
  // The existing predicate also serves quantized dispatch on this base.
  CHECK_FALSE(vt::rocm::GcnArchNameIsGfx12PrefillWmma("gfx1100"));
}

TEST_CASE("SharedK gfx1100 WMMA is enabled by default after full model parity") {
  using vt::rocm::SharedKAttentionWmmaEnabled;
  for (const char* arch : {"gfx1100", "gfx1100:xnack-"}) {
    CHECK(SharedKAttentionWmmaEnabled(arch, nullptr));
    CHECK_FALSE(SharedKAttentionWmmaEnabled(arch, "0"));
    CHECK(SharedKAttentionWmmaEnabled(arch, "1"));
  }
  // gfx1101: same gfx11 WMMA family as gfx1100, admitted for prefill only
  // (GcnArchNameHasGemmaDecodeWmma keeps decode gfx1100-only; the case below
  // still asserts that).
  for (const char* arch : {"gfx1101", "gfx1101:xnack-"}) {
    CHECK(SharedKAttentionWmmaEnabled(arch, nullptr));
    CHECK_FALSE(SharedKAttentionWmmaEnabled(arch, "0"));
    CHECK(SharedKAttentionWmmaEnabled(arch, "1"));
  }
  for (const char* arch : {"gfx1200", "gfx1201", "gfx1201:xnack-"}) {
    CHECK(SharedKAttentionWmmaEnabled(arch, nullptr));
    CHECK_FALSE(SharedKAttentionWmmaEnabled(arch, "0"));
    CHECK(SharedKAttentionWmmaEnabled(arch, "1"));
  }
  for (const char* arch : {"gfx1151", "gfx12010", ""}) {
    CHECK_FALSE(SharedKAttentionWmmaEnabled(arch, nullptr));
    CHECK_FALSE(SharedKAttentionWmmaEnabled(arch, "1"));
  }
}

TEST_CASE("Gemma matrix decode is admitted only on measured gfx1100") {
  CHECK(vt::rocm::GcnArchNameHasGemmaDecodeWmma("gfx1100"));
  CHECK(vt::rocm::GcnArchNameHasGemmaDecodeWmma("gfx1100:xnack-"));
  for (const char* arch : {"gfx1101", "gfx1100junk", "gfx1151", "gfx1200", "gfx1201", ""})
    CHECK_FALSE(vt::rocm::GcnArchNameHasGemmaDecodeWmma(arch));
}

namespace {
std::array<int, 4> skinny_arch_resolves{};

std::string SimulatedSkinnyArch(int device_index) noexcept {
  if (device_index >= 0 && device_index < static_cast<int>(skinny_arch_resolves.size())) {
    ++skinny_arch_resolves[static_cast<size_t>(device_index)];
  }
  switch (device_index) {
    case 0: return "gfx1100";
    case 1: return "gfx942:sramecc+:xnack-";
    case 2: return "gfx1201";
    default: return "future-arch";
  }
}

// Reads as (major, minor) at the call site instead of .first / .second.
void CheckArch(const char* gcn, int major, int minor) {
  const auto cap = CapabilityFromGcnArch(gcn);
  REQUIRE_MESSAGE(cap.has_value(), gcn);
  CHECK_MESSAGE(cap->first == major, gcn);
  CHECK_MESSAGE(cap->second == minor, gcn);
}
}  // namespace

TEST_CASE("gfx9 family parses as 1-digit major, upstream's worked examples") {
  CheckArch("gfx90a", 9, 0);  // MI210 / MI250
  CheckArch("gfx942", 9, 4);  // MI300X / MI325X
  CheckArch("gfx950", 9, 5);  // MI355
}

TEST_CASE("gfx10xx and later parse as 2-digit major") {
  CheckArch("gfx1100", 11, 0);  // RDNA3 dGPU, 7900 XTX (issue #41)
  CheckArch("gfx1101", 11, 0);
  CheckArch("gfx1103", 11, 0);  // RDNA3 iGPU, Radeon 780M (issue #41)
  CheckArch("gfx1151", 11, 5);  // Strix Halo APU (issue #41)
  CheckArch("gfx1200", 12, 0);  // RDNA4
}

TEST_CASE("the HIP feature suffix on gcnArchName is stripped") {
  // hipDeviceProp_t::gcnArchName carries target features on gfx9 parts. The
  // capability must not change because ECC or xnack is on.
  CheckArch("gfx942:sramecc+:xnack-", 9, 4);
  CheckArch("gfx90a:sramecc+:xnack+", 9, 0);
}

TEST_CASE("non-gfx strings are declined, not guessed") {
  // Upstream returns None here (not a ROCm arch string at all) so the caller can
  // fall back. Ours reports "capability unknown" the same way.
  CHECK_FALSE(CapabilityFromGcnArch("").has_value());
  CHECK_FALSE(CapabilityFromGcnArch("sm_90a").has_value());
  CHECK_FALSE(CapabilityFromGcnArch("Apple M4").has_value());
  CHECK_FALSE(CapabilityFromGcnArch("agfx1100").has_value());  // not anchored at 0
}

TEST_CASE("layouts outside the known MMms shape are declined, never split by guess") {
  // Upstream RAISES on each of these rather than returning a value. A skeleton
  // that invented (1, 0) for "gfx1" would hand a wrong capability to a future
  // tactic selector, which is the failure this test exists to prevent.
  CHECK_FALSE(CapabilityFromGcnArch("gfx").has_value());      // no digits
  CHECK_FALSE(CapabilityFromGcnArch("gfx9").has_value());     // too few to split
  CHECK_FALSE(CapabilityFromGcnArch("gfx11000").has_value());  // beyond 4 digits
}

TEST_CASE("major outside [9, 12] is declined") {
  // Both of upstream's sanity rails. A parse that yields major < 9 or > 12 means
  // the layout assumption did not hold, so the answer is not trustworthy.
  CHECK_FALSE(CapabilityFromGcnArch("gfx803").has_value());   // pre-gfx9 (Polaris)
  CHECK_FALSE(CapabilityFromGcnArch("gfx1300").has_value());  // no such generation
}

TEST_CASE("SharedK WMMA host gate is gfx1200/gfx1201 prefix, not substring") {
  using vt::rocm::GcnArchNameIsGfx12PrefillWmma;
  CHECK(GcnArchNameIsGfx12PrefillWmma("gfx1200"));
  CHECK(GcnArchNameIsGfx12PrefillWmma("gfx1201"));
  CHECK(GcnArchNameIsGfx12PrefillWmma("gfx1201:xnack-"));
  CHECK(GcnArchNameIsGfx12PrefillWmma("gfx1201:sramecc+"));
  CHECK(GcnArchNameIsGfx12PrefillWmma("gfx1200:xnack-"));
  CHECK_FALSE(GcnArchNameIsGfx12PrefillWmma(""));
  CHECK_FALSE(GcnArchNameIsGfx12PrefillWmma("gfx1100"));
  CHECK_FALSE(GcnArchNameIsGfx12PrefillWmma("gfx1202"));
  CHECK_FALSE(GcnArchNameIsGfx12PrefillWmma("gfx1210"));
  CHECK_FALSE(GcnArchNameIsGfx12PrefillWmma("foogfx1201"));
  CHECK_FALSE(GcnArchNameIsGfx12PrefillWmma("agfx1201"));
  CHECK_FALSE(GcnArchNameIsGfx12PrefillWmma("gfx12010"));
  CHECK_FALSE(GcnArchNameIsGfx12PrefillWmma("gfx120"));
  static_assert(GcnArchNameIsGfx12PrefillWmma("gfx1201:xnack-"));
  static_assert(!GcnArchNameIsGfx12PrefillWmma("foogfx1201"));
  static_assert(!GcnArchNameIsGfx12PrefillWmma("gfx12010"));
}

TEST_CASE("quantized WMMA admits only measured device names and HIP features") {
  using vt::rocm::GcnArchNameHasQuantWmma;
  for (const char* name : {"gfx1100", "gfx1200", "gfx1201", "gfx1100:xnack-",
                           "gfx1100:sramecc+:xnack-", "gfx1200:sramecc-",
                           "gfx1201:xnack+:sramecc+"}) {
    CAPTURE(std::string(name));
    CHECK(GcnArchNameHasQuantWmma(name));
  }
  for (const char* name : {"", "gfx110", "gfx11000", "gfx1100garbage",
                           "agfx1100", "gfx1100:", "gfx1100:xnack", "gfx1100:foo+",
                           "gfx1100:xnack-:", "gfx1100::xnack-", "gfx1201garbage",
                           "gfx1100:xnack-:xnack+", "gfx1200:sramecc+:sramecc+",
                           "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151",
                           "gfx1202", "gfx1210", "gfx942", "gfx1030"}) {
    CAPTURE(std::string(name));
    CHECK_FALSE(GcnArchNameHasQuantWmma(name));
  }
  CHECK_FALSE(vt::rocm::GcnArchNameIsGfx12PrefillWmma("gfx1100"));
  CHECK_FALSE(vt::rocm::GcnArchNameIsGfx12PrefillWmma("gfx1100:xnack-"));
}

TEST_CASE("the parse is constexpr, so a wrong answer is a compile error") {
  // Not decoration: it is what lets the capability be asserted without a device.
  static_assert(CapabilityFromGcnArch("gfx1100")->first == 11);
  static_assert(CapabilityFromGcnArch("gfx1151")->second == 5);
  static_assert(!CapabilityFromGcnArch("sm_121a").has_value());
  CHECK(true);
}

TEST_CASE("skinny GEMM architecture eligibility follows device hops") {
  skinny_arch_resolves.fill(0);
  CHECK(vt::rocm::SkinnyGemmArchOk(0, SimulatedSkinnyArch));
  CHECK_FALSE(vt::rocm::SkinnyGemmArchOk(1, SimulatedSkinnyArch));
  CHECK(vt::rocm::SkinnyGemmArchOk(2, SimulatedSkinnyArch));
  CHECK_FALSE(vt::rocm::SkinnyGemmArchOk(3, SimulatedSkinnyArch));
  CHECK(vt::rocm::SkinnyGemmArchOk(0, SimulatedSkinnyArch));
  CHECK(skinny_arch_resolves == std::array<int, 4>{1, 1, 1, 1});
}

// --- The memory-model decision (issue #2511) --------------------------------
//
// `ResolveMemoryPolicy` decides which allocator `RocmBackend::Alloc` uses and
// whether the backend may claim its allocations are host-addressable. It is the
// second piece of the ROCm skeleton that holds a DECISION rather than an API
// call, and it is here for the same reason the capability parse is: the whole
// table is checkable on a machine with no AMD GPU, while the board-gated case
// in tests/vt/test_rocm_backend.cpp can only ever check the ONE row its own
// silicon happens to be.
//
// The rows are the real parts. gfx1151 (Strix Halo) and gfx1103 (Radeon 780M)
// probed Integrated/ManagedMemory/ConcurrentManagedAccess = 1 and
// PageableMemoryAccess = 0 (issue #41 F6 attribute table, re-read on the board
// for #2511). A discrete Radeon reports Integrated = 0. The fourth row is the
// NVIDIA-style integrated shape -- everything 1 -- which is the only shape the
// managed branch still serves.
using vt::rocm::ParseManagedAllocOverride;
using vt::rocm::ResolveMemoryPolicy;
using vt::rocm::RocmManagedAllocOverride;
using vt::rocm::RocmMemoryAttributes;

namespace {
constexpr RocmMemoryAttributes kXnackless{/*integrated=*/true, /*managed=*/true,
                                          /*concurrent=*/true, /*pageable=*/false};
constexpr RocmMemoryAttributes kDiscrete{/*integrated=*/false, /*managed=*/false,
                                         /*concurrent=*/false, /*pageable=*/false};
constexpr RocmMemoryAttributes kFullyIntegrated{/*integrated=*/true, /*managed=*/true,
                                                /*concurrent=*/true, /*pageable=*/true};
// Integrated and pageable, but WITHOUT the managed attributes: the W0 ground
// stands on its own and the managed branch is not available at all.
constexpr RocmMemoryAttributes kPageableNoManaged{/*integrated=*/true, /*managed=*/false,
                                                  /*concurrent=*/false, /*pageable=*/true};
}  // namespace

TEST_CASE("#2511: a part that cannot fault and recover gets no managed allocation") {
  // THE ROW THIS CHANGE EXISTS FOR. gfx1151 measured 17 GPU faults in 21 legs on
  // the managed branch and 0 in 21 without it.
  constexpr auto p = ResolveMemoryPolicy(kXnackless, RocmManagedAllocOverride::kUnset);
  static_assert(!p.managed_alloc, "an XNACK-less APU must not get migratable memory");
  // And the capability answer FOLLOWS the allocator. This is the consequence the
  // decision was taken with, not an accident: the CPU reference tier goes with
  // it, and an op that loses its fallback refuses.
  static_assert(!p.unified_memory);
  static_assert(p.host_addressability_note != nullptr,
                "a DELIBERATE withdrawal must be able to say so in the refusal");
  CHECK(std::string(p.host_addressability_note).find("PageableMemoryAccess") !=
        std::string::npos);
  CHECK(std::string(p.host_addressability_note).find("2511") != std::string::npos);
  CHECK(std::string(p.host_addressability_note).find("VT_ROCM_MANAGED_ALLOC=1") !=
        std::string::npos);
}

TEST_CASE("#2511: the managed branch survives where the device CAN fault and recover") {
  // The narrowing is a narrowing, not a removal. An integrated part reporting
  // PageableMemoryAccess = 1 keeps approach (b) exactly as #41 F6 ratified it.
  constexpr auto p = ResolveMemoryPolicy(kFullyIntegrated, RocmManagedAllocOverride::kUnset);
  static_assert(p.managed_alloc);
  static_assert(p.unified_memory);
  static_assert(p.host_addressability_note == nullptr,
                "nothing was withheld here, so the refusal has nothing extra to say");
}

TEST_CASE("the discrete path stays byte-identical to W0, whatever the knob says") {
  // A discrete card must never take the managed branch and must never claim
  // host-addressable memory, because a CPU reference kernel there is corruption
  // rather than a slow path. The override cannot conjure a capability.
  for (auto ov : {RocmManagedAllocOverride::kUnset, RocmManagedAllocOverride::kForceOff,
                  RocmManagedAllocOverride::kForceOn}) {
    const auto p = ResolveMemoryPolicy(kDiscrete, ov);
    CHECK_FALSE(p.managed_alloc);
    CHECK_FALSE(p.unified_memory);
    // Unremarkable, so no note: the generic refusal already says everything a
    // reader needs about a discrete card.
    CHECK(p.host_addressability_note == nullptr);
  }
}

TEST_CASE("the W0 conjunction stands on its own without the managed attributes") {
  constexpr auto p = ResolveMemoryPolicy(kPageableNoManaged, RocmManagedAllocOverride::kUnset);
  static_assert(!p.managed_alloc);
  static_assert(p.unified_memory, "Integrated AND PageableMemoryAccess is ground 1");
  static_assert(p.host_addressability_note == nullptr);
}

TEST_CASE("VT_ROCM_MANAGED_ALLOC=1 restores the pre-#2511 configuration WHOLE") {
  // A red-before arm that moves only the allocator and leaves the capability
  // claim where it was is not the configuration it claims to reproduce, and the
  // single-binary interleaved A/B on strix:gpu0 depends on this being exact.
  constexpr auto on = ResolveMemoryPolicy(kXnackless, RocmManagedAllocOverride::kForceOn);
  static_assert(on.managed_alloc);
  static_assert(on.unified_memory);
  static_assert(on.host_addressability_note == nullptr);

  // `=0` is the other direction and is now redundant on this part -- the default
  // already declines -- but it must still never turn the branch back on.
  constexpr auto off = ResolveMemoryPolicy(kXnackless, RocmManagedAllocOverride::kForceOff);
  static_assert(!off.managed_alloc);
  static_assert(!off.unified_memory);

  // On a part that CAN fault and recover, `=0` is the live A/B lever: it drops
  // the allocator while ground 1 keeps the unified claim, which is exactly the
  // isolation the W9 measurement needed.
  constexpr auto lever = ResolveMemoryPolicy(kFullyIntegrated, RocmManagedAllocOverride::kForceOff);
  static_assert(!lever.managed_alloc);
  static_assert(lever.unified_memory);
  static_assert(lever.host_addressability_note == nullptr);
  CHECK(true);
}

TEST_CASE("the knob parses to the shipped default on anything it does not recognise") {
  CHECK(ParseManagedAllocOverride("0") == RocmManagedAllocOverride::kForceOff);
  CHECK(ParseManagedAllocOverride("1") == RocmManagedAllocOverride::kForceOn);
  // Unset, empty, and a typo all take the shipped default rather than a
  // surprising arm. "00" and "true" are the two shapes a reader most plausibly
  // writes by hand.
  CHECK(ParseManagedAllocOverride("") == RocmManagedAllocOverride::kUnset);
  CHECK(ParseManagedAllocOverride("00") == RocmManagedAllocOverride::kUnset);
  CHECK(ParseManagedAllocOverride("true") == RocmManagedAllocOverride::kUnset);
  CHECK(ParseManagedAllocOverride("2") == RocmManagedAllocOverride::kUnset);
}
