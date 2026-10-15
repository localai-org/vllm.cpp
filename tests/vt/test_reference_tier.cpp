// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream vLLM
// test mirrors it (vLLM's forward_native fallback is implicit in torch).
//
// The portable reference tier's own gate (S5,
// .agents/specs/accelerator-seam-audit.md § Work breakdown row S5). It proves the
// three properties the tier must have, WITHOUT needing Metal or Vulkan hardware:
//
//   1. SAFETY (the load-bearing invariant) — the CPU fallback is installed ONLY
//      where the HOST MAY DEREFERENCE what Backend::Alloc returned
//      (Backend::DeviceMemoryIsHostAddressable()). A device that cannot promise
//      that NEVER receives a CPU fallback; GetOp refuses by name instead,
//      because a CPU kernel against device memory is corruption. Asserted twice:
//      against a fake DISCRETE backend, and against a fake backend carrying the
//      GB10 CUDA pair — unified memory TRUE, host-addressable FALSE. The gate
//      read UnifiedMemory() until #844 / #1435, and the second fake is the shape
//      that made that a SIGSEGV rather than a refusal.
//
//   2. CORRECTNESS-WITH-ZERO-KERNELS — a device with NO native kernel for an op
//      still produces the right answer through the reference tier. Asserted
//      against a fake UNIFIED, host-addressable backend (host-memory allocator
//      standing in for Metal StorageModeShared and integrated Vulkan): vt::Relu
//      dispatched on that device falls back to the CPU kernel and returns
//      bit-identical output.
//
//   3. NATIVE ALWAYS WINS + OBSERVABILITY — a registered native kernel outranks
//      the tier (priority), the tier is never silent (GetReferenceTierHits, and a
//      one-time stderr line), and OpRegistered still means "native kernel present"
//      so the fused-recipe ladder is unchanged.
//
// This file is its own executable (tests/CMakeLists.txt: one add_executable per
// test), so registering a backend on the unused TestDevice() and kTENSTORRENT
// slots cannot leak into any other test binary. The two slots are separate
// because a reference-tier provider cannot be uninstalled once registered, so a
// case that must observe an EMPTY table needs a table nothing else has touched.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/op_provider.h"
#include "vt/ops.h"

namespace {

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::OpId;
using vt::Queue;
using vt::Tensor;

// Keep synthetic host allocations away from real device-only XPU kernels.
// TENSTORRENT remains reserved for the separate non-host-addressable case.
DeviceType TestDevice() {
  static const DeviceType device = [] {
    for (auto d : {DeviceType::kXPU, DeviceType::kMETAL, DeviceType::kVULKAN,
                   DeviceType::kROCM, DeviceType::kCUDA}) {
      if (vt::TryGetBackend(d)) continue;
      bool empty = true;
      for (int op = 0; op < static_cast<int>(OpId::kCount); ++op)
        if (vt::OpProviderCount(static_cast<OpId>(op), d)) empty = false;
      if (empty) return d;
    }
    throw std::runtime_error("reference-tier test requires an unused device registry slot");
  }();
  return device;
}

// A minimal backend over ordinary HOST memory. `unified` is the only thing under
// test: a unified instance stands in for Metal/GB10/integrated-Vulkan (host and
// device pointers alias, so the CPU reference tier is sound); a discrete instance
// stands in for a real GPU where it must be refused. Everything else is a plain
// host allocator so a fallback kernel dispatched here actually runs.
class FakeBackend final : public Backend {
 public:
  // `unified` and `host_addressable` are the two properties under test, and they
  // are SEPARATE because a real backend separates them. `UnifiedMemory()` says
  // host and device address the same physical RAM; `DeviceMemoryIsHostAddressable()`
  // says a pointer from `Alloc()` may be dereferenced by the host. CUDA on GB10
  // answers true then false (include/vt/backend.h), which is the pair the third
  // instance below carries. Everything else is a plain host allocator so a
  // fallback kernel dispatched here actually runs.
  FakeBackend(bool unified, bool host_addressable, const char* note = nullptr)
      : unified_(unified), host_addressable_(host_addressable), note_(note) {}
  void* Alloc(size_t bytes) override { return std::malloc(bytes == 0 ? 1 : bytes); }
  void Free(void* p) override { std::free(p); }
  void Memset(Queue&, void* p, int v, size_t bytes) override { std::memset(p, v, bytes); }
  void Copy(Queue&, void* dst, const void* src, size_t bytes) override {
    std::memcpy(dst, src, bytes);
  }
  Queue CreateQueue() override { return Queue{Device{TestDevice(), 0}, nullptr}; }
  bool UnifiedMemory() const override { return unified_; }
  bool DeviceMemoryIsHostAddressable() const override { return host_addressable_; }
  const char* HostAddressabilityNote() const override { return note_; }

 private:
  bool unified_;
  bool host_addressable_;
  const char* note_ = nullptr;
};

FakeBackend& Unified() {
  static FakeBackend b(true, true);
  return b;
}
FakeBackend& Discrete() {
  static FakeBackend b(false, false);
  return b;
}
// The GB10 CUDA shape, expressed without a GPU: host and device share the RAM,
// and a `cudaMalloc` pointer is still not host-dereferenceable.
FakeBackend& UnifiedNotHostAddressable() {
  static FakeBackend b(true, false);
  return b;
}
// A backend that WITHDREW host addressability deliberately and says why. This is
// the ROCm-on-an-XNACK-less-APU shape (issue #2511): the managed allocator that
// made those boards claim host-addressable memory is what page-faulted them, so
// the claim was withdrawn and the reference tier went with it. The literal is
// this test's own, not the ROCm one, because the assertion under test is that
// the REFUSAL CARRIES THE BACKEND'S NOTE -- not what any particular backend
// chose to write in it.
inline constexpr const char* kNote =
    "TESTNOTE: this device withdrew host addressability on purpose";
FakeBackend& DiscreteWithNote() {
  static FakeBackend b(false, false, kNote);
  return b;
}

void* AsVoid(void (*f)()) { return reinterpret_cast<void*>(f); }
void NativeKernel() {}

}  // namespace

// ---------------------------------------------------------------------------
// 1. SAFETY GATE — a DISCRETE device is never given a CPU fallback.
// ---------------------------------------------------------------------------
TEST_CASE("reference tier: a discrete (non-unified) device is refused and still throws") {
  vt::RegisterBackend(TestDevice(), &Discrete());

  // The gate itself.
  CHECK_FALSE(vt::ReferenceTierEligible(TestDevice()));
  // Eager install is a strict no-op: nothing registered, count 0.
  CHECK(vt::RegisterReferenceTier(TestDevice()) == 0);
  CHECK(vt::OpProviderCount(OpId::kRelu, TestDevice()) == 0);
  // And a dispatch on an op it lacks THROWS — a CPU kernel must not be allowed to
  // run against what this device claims is discrete memory.
  CHECK_FALSE(vt::OpRegistered(OpId::kRelu, TestDevice()));
  CHECK_THROWS(vt::GetOp(OpId::kRelu, TestDevice()));
}

// A device that withdrew host addressability DELIBERATELY gets to say why, in
// the refusal, where the user meets it. The generic sentence is true on every
// such device and useless on the one where the false is a decision rather than
// the hardware -- ROCm on gfx1151 after #2511, which is what
// `Backend::HostAddressabilityNote()` exists for.
//
// Ordered immediately after the discrete case ON PURPOSE: it shares kXPU, and
// the eligible case further down installs providers on that slot permanently.
// The REQUIRE below turns a reordering into a diagnosable failure instead of a
// silently vacuous pass.
TEST_CASE("reference tier: the refusal carries the backend's own reason") {
  REQUIRE(vt::OpProviderCount(OpId::kRelu, TestDevice()) == 0);
  vt::RegisterBackend(TestDevice(), &DiscreteWithNote());
  REQUIRE_FALSE(vt::ReferenceTierEligible(TestDevice()));

  std::string message;
  try {
    vt::GetOp(OpId::kRelu, TestDevice());
  } catch (const std::exception& e) {
    message = e.what();
  }
  REQUIRE_FALSE(message.empty());
  // The generic clauses still stand -- the note ADDS, it never replaces.
  CHECK(message.find("Relu") != std::string::npos);
  CHECK(message.find("host-addressable") != std::string::npos);
  // And the backend's own sentence is in it, verbatim.
  CHECK(message.find(kNote) != std::string::npos);

  // A backend with NOTHING to add leaves the message byte-identical, so the
  // note cannot become a sentence every refusal drags along. Same slot, same
  // op, same refusal path: the only difference is the note.
  vt::RegisterBackend(TestDevice(), &Discrete());
  std::string plain;
  try {
    vt::GetOp(OpId::kRelu, TestDevice());
  } catch (const std::exception& e) {
    plain = e.what();
  }
  REQUIRE_FALSE(plain.empty());
  CHECK(plain.find(kNote) == std::string::npos);
  CHECK(plain.size() < message.size());
}

// The CPU device is the SOURCE of the reference kernels and is never a target.
TEST_CASE("reference tier: the CPU source device is never eligible") {
  CHECK_FALSE(vt::ReferenceTierEligible(DeviceType::kCPU));
}

// ---------------------------------------------------------------------------
// 1b. THE SAFETY GATE IS HOST-ADDRESSABILITY, NOT UNIFIED MEMORY (#844, #1435).
//
// This is the shape that crashed twice on GB10. `Backend::UnifiedMemory()` is
// true there, because host and device address the same physical RAM. It is NOT
// the property a host kernel needs: `CudaBackend::Alloc` calls `cudaMalloc`, and
// `include/vt/backend.h` records beside `DeviceMemoryIsHostAddressable()` that
// such a pointer "is still not host-dereferenceable". Gating the tier on the
// wide property handed the CPU kernel device pointers, and the process took
// SIGSEGV under a banner reading "correct but slow" -- `vt::QuantFp8Static` on
// sm_110 (#960) and `vt::MatmulFp8BlockScaled` on a CUTLASS-less build (#1435,
// exit 139 with `test cases: 0  assertions: 0`).
//
// A SEPARATE device slot from the cases around it, deliberately: the tier cannot
// be uninstalled once a provider is registered, so sharing kXPU with the unified
// case below would make this assertion depend on doctest's execution order.
//
// AND THAT SLOT IS SKIPPED, NOT ASSERTED EMPTY. kTENSTORRENT is not an unused
// enumerator: src/vt/tenstorrent/tenstorrent_ops.cpp registers 21 op sites on it,
// `OpId::kRelu` among them, and `vllm` carries an INTERFACE `--whole-archive`
// (CMakeLists.txt), so on a `-DVLLM_CPP_TENSTORRENT=ON` build that registrar runs
// inside THIS binary. An earlier draft REQUIREd the table empty and called that
// failing loudly. It is not: the failure is unconditional on a legitimate
// configuration nobody's CI builds, so it would land as a permanent red for
// whoever next turns that backend on, and RegisterBackend would displace a live
// backend in-process on the way. Skipping is the honest answer, because the
// property under test needs a slot no other registrar has claimed and this build
// then has none.
// ---------------------------------------------------------------------------
TEST_CASE("reference tier: unified memory the host cannot address is refused, by name") {
  constexpr DeviceType kSlot = DeviceType::kTENSTORRENT;
  if (vt::TryGetBackend(kSlot) != nullptr ||
      vt::OpProviderCount(OpId::kRelu, kSlot) != 0) {
    MESSAGE("SKIP: DeviceType::kTENSTORRENT carries a backend or providers in "
            "this build, so it is not a free slot; this case needs one no other "
            "registrar has claimed");
    return;
  }
  vt::RegisterBackend(kSlot, &UnifiedNotHostAddressable());

  // The property pair under test: the OLD gate says yes, the right one says no.
  Backend& dev = vt::GetBackend(kSlot);
  REQUIRE(dev.UnifiedMemory());
  REQUIRE_FALSE(dev.DeviceMemoryIsHostAddressable());

  // THE INVARIANT. A host kernel is never installed for a device whose memory
  // the host cannot dereference.
  CHECK_FALSE(vt::ReferenceTierEligible(kSlot));
  CHECK(vt::RegisterReferenceTier(kSlot) == 0);
  CHECK(vt::OpProviderCount(OpId::kRelu, kSlot) == 0);

  // And the dispatch REFUSES rather than returning the host kernel. Comparing
  // against the CPU fn pointer is what separates a refusal from a fallback: the
  // tier installs `src->fn`, the very same pointer, so an equal answer here is
  // the defect even when nothing throws.
  REQUIRE(vt::OpRegistered(OpId::kRelu, DeviceType::kCPU));
  void* thrown_fn = nullptr;
  std::string message;
  try {
    thrown_fn = vt::GetOp(OpId::kRelu, kSlot);
  } catch (const std::exception& e) {
    message = e.what();
  }
  CHECK(thrown_fn == nullptr);
  CHECK(thrown_fn != vt::GetOp(OpId::kRelu, DeviceType::kCPU));

  // Refused BY NAME: the op, the device, and why the portable tier did not run.
  // A reader who sees this must not have to open op_provider.cpp to learn that a
  // fallback exists and was withheld.
  CHECK(message.find("Relu") != std::string::npos);
  CHECK(message.find("tenstorrent") != std::string::npos);
  CHECK(message.find("host-addressable") != std::string::npos);
}

// ---------------------------------------------------------------------------
// 2. CORRECTNESS WITH ZERO NATIVE KERNELS — a unified device runs via the tier.
// ---------------------------------------------------------------------------
TEST_CASE("reference tier: a unified device with NO native Relu falls back to the CPU kernel") {
  vt::RegisterBackend(TestDevice(), &Unified());
  CHECK(vt::ReferenceTierEligible(TestDevice()));

  // There is no native kMETAL/kVULKAN kernel here; the test device has none registered.
  REQUIRE_FALSE(vt::OpRegistered(OpId::kRelu, TestDevice()));

  constexpr int64_t kRows = 5, kCols = 32;
  constexpr size_t kN = kRows * kCols;
  std::vector<float> in(kN);
  for (size_t i = 0; i < kN; ++i) in[i] = static_cast<float>(static_cast<int>(i) - 40) * 0.25f;

  // CPU oracle through the very same vt::Relu entry point.
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue cq = cpu.CreateQueue();
  const Device cd{DeviceType::kCPU, 0};
  std::vector<float> ref(kN);
  {
    std::vector<float> ci = in;
    Tensor ti = Tensor::Contiguous(ci.data(), DType::kF32, cd, {kRows, kCols});
    Tensor to = Tensor::Contiguous(ref.data(), DType::kF32, cd, {kRows, kCols});
    vt::Relu(cq, to, ti);
  }
  cpu.DestroyQueue(cq);

  const unsigned long long hits_before = vt::GetReferenceTierHits();

  // Now the same op on the UNIFIED "accelerator" — with zero native kernels.
  Backend& dev = vt::GetBackend(TestDevice());
  Queue q = dev.CreateQueue();
  const Device d{TestDevice(), 0};
  void* pin = dev.Alloc(kN * sizeof(float));
  void* pout = dev.Alloc(kN * sizeof(float));
  dev.Copy(q, pin, in.data(), kN * sizeof(float));
  Tensor ti = Tensor::Contiguous(pin, DType::kF32, d, {kRows, kCols});
  Tensor to = Tensor::Contiguous(pout, DType::kF32, d, {kRows, kCols});
  vt::Relu(q, to, ti);  // dispatch -> reference tier installs + runs the CPU kernel
  dev.Synchronize(q);

  std::vector<float> got(kN);
  dev.Copy(q, got.data(), pout, kN * sizeof(float));
  // The tier reuses the EXACT CPU kernel, so the result is bit-identical, not
  // merely close: memcmp, no tolerance.
  CHECK(std::memcmp(ref.data(), got.data(), kN * sizeof(float)) == 0);
  dev.Free(pin);
  dev.Free(pout);
  dev.DestroyQueue(q);

  // OBSERVABILITY — the fallback is loud, not silent.
  CHECK(vt::GetReferenceTierHits() > hits_before);
  REQUIRE(vt::OpProviderCount(OpId::kRelu, TestDevice()) >= 1);
  CHECK(std::string(vt::OpProviderNameAt(OpId::kRelu, TestDevice(), 0)) ==
        vt::kReferenceProviderName);
  const auto s = vt::GetOpProviderStats(OpId::kRelu, TestDevice());
  REQUIRE(s.last_selected != nullptr);
  CHECK(std::string(s.last_selected) == vt::kReferenceProviderName);

  // And it is NOT counted as a native kernel — the meaning OpRegistered must keep
  // for the fused-recipe ladder — even though GetOp now returns a working fn.
  CHECK_FALSE(vt::OpRegistered(OpId::kRelu, TestDevice()));
}

// ---------------------------------------------------------------------------
// 3. NATIVE ALWAYS WINS + eager registration shape.
// ---------------------------------------------------------------------------
TEST_CASE("reference tier: a native kernel outranks the fallback and is left untouched") {
  vt::RegisterBackend(TestDevice(), &Unified());

  // A native provider for kEmbedding on the "accelerator".
  vt::RegisterOp(OpId::kEmbedding, TestDevice(), AsVoid(&NativeKernel));
  REQUIRE(vt::OpRegistered(OpId::kEmbedding, TestDevice()));

  // Eager install must SKIP kEmbedding (already native) and install for the ops
  // that only have a CPU kernel. It returns a positive count and never displaces
  // the native provider.
  const int installed = vt::RegisterReferenceTier(TestDevice());
  CHECK(installed > 0);
  CHECK(vt::GetOp(OpId::kEmbedding, TestDevice()) == AsVoid(&NativeKernel));
  CHECK(vt::OpProviderCount(OpId::kEmbedding, TestDevice()) == 1);  // no fallback added

  // An op that only had a CPU kernel now resolves through the tier.
  CHECK(vt::OpRegistered(OpId::kMatmul, DeviceType::kCPU));
  CHECK(vt::GetOp(OpId::kMatmul, TestDevice()) ==
        vt::GetOp(OpId::kMatmul, DeviceType::kCPU));  // same host kernel pointer
}

// A second eager pass is idempotent — the tier providers are already present and
// RegisterOpProvider rejects the duplicate name, so nothing new is installed.
TEST_CASE("reference tier: eager registration is idempotent") {
  vt::RegisterBackend(TestDevice(), &Unified());
  (void)vt::RegisterReferenceTier(TestDevice());
  const int second = vt::RegisterReferenceTier(TestDevice());
  CHECK(second == 0);
}
