// vllm.cpp original (vt runtime, inventory deviation §9.1) — the ACCELERATION
// PROVIDER seam (BACKEND-ACCEL-PROVIDER, .agents/specs/metal-mlx-reuse-study.md
// §6, work row W0b-2).
//
// THE GAP THIS CLOSES. `src/vt/ops.cpp` keyed exactly ONE `void*` per
// (OpId, DeviceType) and `RegisterOp` overwrote it with no check and no warning.
// Two implementations of one op on one device therefore could not coexist: which
// one you got depended on static-initialization order ACROSS translation units,
// which the standard leaves unspecified. "Register MLX's matmul alongside our
// MSL matmul" was a NONDETERMINISTIC BUILD, not a configuration. That is the bug
// this header fixes; MLX is merely the first caller.
//
// THE SEAM. This generalizes `src/vt/cuda/cuda_arch_tactics.h` — which already
// had every property the op table lacked — up one layer, out of `vt::cuda`, and
// keys it on (OpId, DeviceType):
//   * capacity-bounded static storage, registration is TABLE FILL ONLY and never
//     throws or allocates, so it is safe from any static-init order;
//   * a capability predicate (`supports`) evaluated against a DEVICE-NEUTRAL
//     `ProviderCaps` — the one genuine CUDA coupling of the tactic registry
//     (`DeviceCaps`) generalized away;
//   * a portable FALLBACK: a provider that cannot handle a particular call
//     declines and forwards to the next provider down (`GetOpFallback`), which
//     is the `ArchTacticLaunchFn`-returns-false axis expressed at this layer;
//   * SELECTION IS OBSERVABLE (`GetOpProviderStats`, `VT_OP_PROVIDER_STATS=1`),
//     because a passing test does not prove WHICH provider executed;
//   * SELECTION IS DETERMINISTIC — see below.
//
// DETERMINISM (the actual defect, stated precisely). Selection order is
// `(priority DESC, name ASC)` with `name` compared by `strcmp`. Both keys are
// compile-time constants of the registering TU, so the selected provider is a
// pure function of WHICH providers are linked in — never of the order their
// static initializers happened to run. Registering the same `name` twice for one
// (OpId, DeviceType) is a programming error and is rejected (first wins), so the
// order is also a strict total order. `tests/vt/test_op_provider.cpp` asserts
// this by registering providers in both orders and requiring the same winner.
//
// RECONCILIATION WITH `.agents/specs/dropin-kernel-abi.md` (BACKEND-ABI-VT).
// That spec is the ARGUMENT half of a provider seam: how a raw-C launcher
// (CUTLASS/Marlin/cuBLASLt/flashinfer) receives pointers, shapes, strides,
// semantic types, workspace and stream. It is deliberately silent on WHICH
// launcher runs. This header is the SELECTION half and is deliberately silent on
// the argument shape — `OpProvider::fn` is the same type-erased `void*` the op
// table always held, so a drop-in adapter registered through `RegisterTypedOp`
// is a provider like any other. They compose; neither replaces the other.
//
// FOUR BACKENDS, ONE MECHANISM (the test the user set for this design):
//   * Metal  — MLX (`steel_matmul`/`qmm`), a C++ OBJECT-MODEL library; the shape
//              `dropin-kernel-abi.md` cannot express, and the reason this exists.
//   * CUDA   — cuBLASLt / CUTLASS / flashinfer / Marlin: raw C launchers whose
//              ARGUMENTS come from `dropin-kernel-abi.md` and whose SELECTION
//              comes from here; `supports()` reads `ProviderCaps::compute_major`.
//   * Vulkan — llama.cpp-style coopmat/coopmat2 shader tactics, selected on
//              `ProviderCaps::features` bits published by the Vulkan context.
//   * CPU    — llama.cpp `vec_dot`/repack tiers, selected on ISA feature bits.
// It is not an MLX seam. MLX is one row.
#pragma once

#include <cstdint>
#include <initializer_list>

#include "vt/device.h"

namespace vt {

enum class OpId : uint8_t;  // defined in vt/ops.h
struct Tensor;
// Opt-in JSONL argument metadata, never tensor contents. Complements provider
// selection rows in VT_OP_PROVIDER_TRACE; does not count a second selection.
void TraceOpTensors(OpId op, const Queue& q, std::initializer_list<const Tensor*> tensors);

// Device-neutral capability record. The CUDA tactic registry's `DeviceCaps`
// (src/vt/cuda/cuda_device_caps.h) is CUDA-shaped — `sm_major`, `sm_minor`,
// `max_shared_memory_per_block_optin`. This is the same idea with the vendor
// nouns removed, so one predicate signature serves five platforms:
//   CUDA   compute_major/minor = SM version; features = (unused today)
//   Metal  compute_major       = MTLGPUFamily generation (Apple7 -> 7)
//   Vulkan features            = coopmat / coopmat2 / fp16-storage bits
//   CPU    features            = ISA bits (AVX-512, NEON dotprod, ...)
// A backend publishes its record ONCE at init via SetDeviceProviderCaps(); until
// it does, `valid` is false and a predicate that needs real capabilities must
// decline. Selection is cached, so publication must precede first op dispatch —
// SetDeviceProviderCaps() invalidates the cache to make that safe either way.
struct ProviderCaps {
  DeviceType device = DeviceType::kCPU;
  bool valid = false;
  int32_t compute_major = 0;
  int32_t compute_minor = 0;
  int32_t compute_unit_count = 0;
  // Backend-defined capability bits. Deliberately opaque here: the meaning
  // belongs to the platform that publishes it and the providers that read it,
  // and putting a union of five vendors' feature enums in a shared header would
  // recreate exactly the coupling this seam removes.
  uint64_t features = 0;
};

void SetDeviceProviderCaps(DeviceType device, const ProviderCaps& caps);
ProviderCaps GetDeviceProviderCaps(DeviceType device);

// Answers "can this provider run on THIS device at all". Must be cheap and
// side-effect free. `nullptr` means "always" — which is what every first-party
// backend kernel registered through RegisterOp() gets, preserving behaviour
// exactly. Per-CALL refusal (a shape or dtype this provider does not handle) is
// NOT expressed here: it is expressed by the kernel itself calling
// GetOpFallback() and forwarding, because `GetOp` has no shape to inspect and
// all ~70 op entry points must keep working with ZERO call-site edits.
using OpProviderSupportsFn = bool (*)(const ProviderCaps&);

// The name every first-party backend kernel registers under. Chosen to sort
// AFTER short vendor names under strcmp is NOT relied upon — priority, not the
// name, is what makes an accelerator win; the name only breaks priority ties.
inline constexpr const char* kNativeProviderName = "vt-native";

struct OpProvider {
  const char* name = nullptr;      // stable identity: "vt-native", "mlx", ...
  int priority = 0;                // higher wins; ties broken by name (strcmp)
  OpProviderSupportsFn supports = nullptr;  // nullptr == unconditionally
  void* fn = nullptr;              // the kernel pointer, type-erased as before
};

// Registers `provider` for (op, device). Table fill only: never allocates, never
// throws, and never depends on another TU's constructor having run. Silently
// ignored when the capacity is exhausted, `fn` is null, `name` is null, or
// `name` is already registered for this (op, device) — matching the
// RegisterArchTactic() contract, whose rationale is that a registrar runs before
// main() where throwing has no receiver.
void RegisterOpProvider(OpId op, DeviceType device, const OpProvider& provider);

// The pre-existing single-kernel registration, unchanged for every caller: it is
// exactly RegisterOpProvider() with name=kNativeProviderName, priority 0 and no
// capability predicate. Every backend in the tree keeps registering this way and
// keeps winning, because nothing else registers at priority 0 under that name.
void RegisterOp(OpId op, DeviceType device, void* fn);

// The canonical spelling of an op — the enumerator without its `k`. The
// counterpart of DeviceTypeName (include/vt/device.h), and what every refusal
// and reference-tier warning in this seam reports, so no reader has to count
// enumerators in include/vt/ops.h to find out what was refused. Total over the
// enum: the defining switch is exhaustive and `default`-free, so appending an
// OpId without naming it fails the -Werror build.
const char* OpName(OpId op);

// The selected provider's kernel for (op, device): highest priority whose
// `supports(caps)` holds, ties by name. Throws when nothing is registered or
// nothing supports the device — the pre-existing GetOp contract, naming the op
// and the device rather than their integers.
void* GetOp(OpId op, DeviceType device);

// DECLINE-AND-FALL-BACK. A provider kernel that cannot serve a particular call
// looks up the next provider strictly BELOW itself in the selection order and
// forwards to it. This is the second fallback axis of the tactic registry
// (`ArchTacticLaunchFn` returning false) expressed where a shape is actually
// visible — inside the kernel — which is what keeps the ~70 op entry points
// edit-free. Counts ONE decline in `OpProviderStats::declines` -- including a
// decline that finds nothing below it and throws. Throws if nothing is below.
void* GetOpFallback(OpId op, DeviceType device, const char* declining_provider);

// The same resolution, counting NOTHING. Exists for exactly one shape: a
// SHAPE-GATED provider that declines on the hot path and therefore hoists the
// lookup into a function-local static, then counts each decline with
// `NoteOpDecline`. Hoisting a COUNTING resolver silently hoists the accounting
// with it, and the static's one-time initialisation then adds a second
// increment to the first decline of the process -- which is what this header
// used to prescribe, and what
// [#1584](https://github.com/mudler/vllm.cpp/issues/1584) measured.
//
// Reach for it only WITH `NoteOpDecline` beside it. A caller that uses it and
// counts nothing reports a forwarded call as a served one, and no gate in this
// tree can see the difference. `GetOpFallback` stays the ordinary spelling and
// stays correct by default; this one is the opt-out and says so in its name.
void* GetOpFallbackUncounted(OpId op, DeviceType device, const char* declining_provider);

// Non-throwing probes. `OpRegistered` keeps its exact prior meaning: is ANY
// kernel realized for (op, device) — the fused-recipe fast-realization ladder
// and the cross-device coverage harness both depend on it.
bool OpRegistered(OpId op, DeviceType device);
int OpProviderCount(OpId op, DeviceType device);
// i-th provider in DETERMINISTIC selection order (nullptr when out of range).
const char* OpProviderNameAt(OpId op, DeviceType device, int i);

// POSITIVE SIGNAL — ported from ArchTacticStats. A green test does not prove a
// provider ran, and the fan-out spike's Risk 4 is exactly a probe failing
// SILENTLY into the slow path. `selections` counts dispatches through GetOp once
// per-call counting is on; `last_selected` names the bound provider; `declines`
// counts GetOpFallback() forwards; `fallbacks` counts resolutions that found no
// supporting provider. VT_OP_PROVIDER_STATS=1 prints one line per (op, device)
// the first time it resolves, so a device run leaves evidence in its own log.
struct OpProviderStats {
  const char* last_selected = nullptr;
  unsigned long long selections = 0;
  unsigned long long declines = 0;
  unsigned long long fallbacks = 0;
};
OpProviderStats GetOpProviderStats(OpId op, DeviceType device);

// Per-call selection counting. OFF by default so the hot path stays a cached
// pointer load; VT_OP_PROVIDER_STATS=1 turns it on, and tests turn it on
// explicitly rather than depending on the environment.
void EnableOpProviderCallStats(bool on);
// Counts one decline WITHOUT re-resolving the fallback. `GetOpFallback` walks the
// provider stack and re-reads device caps, which is fine when declines are rare
// but not when a SHAPE-GATED provider declines on the hot path -- a decode run
// declines ~21,500 times. Such a provider caches `GetOpFallbackUncounted` once
// and calls this per decline, and `OpProviderStats::declines` is then exact from
// the FIRST decline while the lookup cost disappears. Pairing it with the
// COUNTING `GetOpFallback` instead is the #1584 defect: the count reads N+1 over
// N declines, and a routing gate asserting an exact count then passes or fails
// on whether something else in the same binary already warmed the static.
void NoteOpDecline(OpId op, DeviceType device);

void ResetOpProviderStats(OpId op, DeviceType device);

// Runtime A/B lever: disables a provider by name, forcing selection to fall to
// the next one down. Also settable as VT_OP_PROVIDER_DISABLE=name[,name...].
// This is how "same binary, MLX on vs MLX off" is measured without a rebuild —
// the same-binary A/B the benchmark protocol requires.
void DisableOpProvider(const char* name, bool disabled);
bool OpProviderDisabled(const char* name);

// --- Portable reference tier (S5, .agents/specs/accelerator-seam-audit.md) ----
//
// vLLM's `CustomOp.forward_native` (custom_op.py:138) is a pure-torch body every
// op carries, so a brand-new platform that implements ZERO kernels is still
// CORRECT — just slow — because torch is the universal portable op layer. Our
// GetOp() THROWS on an unregistered op, so a partial backend (Metal: 18/75 ops,
// Vulkan: a skeleton) could not run a model at all. This is our equivalent of
// `forward_native`: register the existing CPU kernel as a NEGATIVE-priority
// provider on a UNIFIED-MEMORY device, so an op the device lacks a native kernel
// for falls back to the CPU reference instead of throwing.
//
// SAFETY (the load-bearing invariant). A CPU kernel dereferences the pointers it
// is given. That is correct ONLY where the HOST MAY DEREFERENCE what
// `Backend::Alloc` returned — `Backend::DeviceMemoryIsHostAddressable()`. The
// tier is gated on that property, never on DeviceType blindly.
//
// IT IS NOT `Backend::UnifiedMemory()`, and the difference is two crashes
// (#844, #1435). Unified memory says host and device address the same physical
// RAM; it does NOT say a device allocation is host-dereferenceable. CUDA on GB10
// reports unified memory and allocates with `cudaMalloc`, which the host may not
// touch, so gating on the wide property ran host kernels over device pointers
// and the process took SIGSEGV under a banner claiming a correct fallback.
// include/vt/backend.h carries the same distinction beside the narrow predicate,
// including why its default is `false`.
//
// DETERMINISM / no-change-on-native. The fallback registers at
// kReferenceTierPriority (strictly below every native kernel's priority >= 0), so
// a native kernel always wins when present — a backend that HAS the kernel is
// byte-identical to before. And the tier installs LAZILY, only on a genuine
// GetOp miss, so a backend that never misses (CUDA on the gate models) is
// completely untouched: no provider added, no table change.
inline constexpr const char* kReferenceProviderName = "vt-cpu-ref";
inline constexpr int kReferenceTierPriority = -1000;  // strictly below any native

// THE SAFETY GATE. True iff `device` may host the CPU reference tier: it is not
// the CPU source device itself, a backend is registered for it, and that backend
// reports DeviceMemoryIsHostAddressable() == true. Consulted at registration
// time; a device that answers false NEVER gets a CPU fallback installed, and
// GetOp refuses it by name instead, naming which precondition it failed.
bool ReferenceTierEligible(DeviceType device);

// Eagerly install the reference tier for `target`: for every op that has a CPU
// kernel and no native kernel on `target`, install the CPU fn as a
// kReferenceProviderName provider. Returns the number installed; a no-op
// returning 0 unless ReferenceTierEligible(target). Production need not call
// this — the tier auto-installs on the first GetOp miss — but a backend may call
// it to warm up, and tests use it to assert the gate.
int RegisterReferenceTier(DeviceType target);

// OBSERVABILITY (Risk 7 — a slow fallback must never be silent). The number of
// distinct (op, device) resolutions that selected the reference tier since
// process start. It MUST be 0 in any performance arm: a non-zero value means a
// native kernel is missing and the portable CPU path ran. VT_OP_PROVIDER_STATS=1
// additionally prints a one-time stderr line the first time each (op, device)
// falls back.
unsigned long long GetReferenceTierHits();

}  // namespace vt
