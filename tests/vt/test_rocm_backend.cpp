// ROCm backend skeleton gates (BACKEND-ROCM, W0). Newly authored — vLLM has no
// C++ ROCm backend tests to port. Mirrors tests/vt/test_metal_backend.cpp, which
// mirrors tests/vt/test_backend.cpp, so all three read side by side.
//
// RUN STATE, per issue #41: the W0 cases in this file ran green on community
// boards — gfx1151, gfx1103, gfx1100 and gfx1201 (5 cases, 1044 assertions in
// the posted tables). The two approach-(b) cases (alloc path / host-readable)
// have NEVER RUN: no AMD GPU exists on the authoring machine. The file is
// LINKED into a test binary only in a HIP build (tests/CMakeLists.txt gates it
// on VLLM_CPP_HIP) but COMPILED everywhere: a non-HIP build object-compiles it
// as a bit-rot guard (see the CMake block next to the ROCm sources), so its
// types are checked on CI even with no ROCm installed. Compiled is not run. If
// a new case fails on your board, that is far more likely a bug in the blind
// change than in your setup — paste the output into
// https://github.com/mudler/vllm.cpp/issues/41 with the arch it printed.
//
// Deliberately plain C++ with no HIP header: every assertion goes through the
// public vt:: / vllm::platforms:: seams. If the skeleton needed HIP in a test to
// be checkable, the seam would be leaking.
//
// NOT HERE: cross-device numeric equality against the CPU oracle. That lives in
// tests/vt/test_backend_cross_device.cpp, which discovers every registered
// non-CPU backend and so covers ROCm automatically — including the RmsNorm this
// skeleton registers, at NMSE <= 5e-4. Run both.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "vllm/platforms/interface.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/registry.h"
#include "vt/backend.h"
#include "vt/op_provider.h"
#include "vt/ops.h"
#include "vt/quant.h"
#include "vt/rocm/rocm_arch.h"
#include "vt/rocm/rocm_runtime.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {
// Every test is a no-op when the build has HIP but the box has no AMD GPU (a
// contributor cross-compiling, or CI). Skipping is correct; failing would be a
// lie about the hardware.
bool NoDevice() { return !vt::rocm::DeviceAvailable(); }
}  // namespace

TEST_CASE("ROCm embedding rejects bad IDs and recovers on the same queue") {
  if (NoDevice()) return;
  Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  Queue q = rocm.CreateQueue();
  constexpr int64_t vocab = 7, width = 8;
  std::vector<float> values(vocab * width);
  for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(i);
  void* table = rocm.Alloc(values.size() * sizeof(float));
  void* ids = rocm.Alloc(sizeof(int64_t));
  void* output = rocm.Alloc(width * sizeof(float));
  rocm.Copy(q, table, values.data(), values.size() * sizeof(float));
  Tensor t = Tensor::Contiguous(table, DType::kF32, q.device, {vocab, width});
  Tensor out = Tensor::Contiguous(output, DType::kF32, q.device, {1, width});
  for (DType dtype : {DType::kI32, DType::kI64}) {
    Tensor index = Tensor::Contiguous(ids, dtype, q.device, {1});
    for (int64_t value : {int64_t{-1}, vocab, int64_t{0}, vocab - 1, int64_t{0}}) {
      const int32_t narrow = static_cast<int32_t>(value);
      rocm.Copy(q, ids,
                dtype == DType::kI32 ? static_cast<const void*>(&narrow)
                                     : static_cast<const void*>(&value),
                vt::SizeOf(dtype));
      if (value < 0 || value >= vocab) {
        CHECK_THROWS_WITH_AS(vt::Embedding(q, out, t, index), doctest::Contains("out of range"),
                             std::runtime_error);
      } else {
        vt::Embedding(q, out, t, index);
        std::vector<float> actual(width);
        rocm.Copy(q, actual.data(), output, width * sizeof(float));
        rocm.Synchronize(q);
        for (int64_t i = 0; i < width; ++i) CHECK(actual[i] == values[value * width + i]);
      }
    }
  }
  rocm.Synchronize(q);
  rocm.Free(table);
  rocm.Free(ids);
  rocm.Free(output);
  rocm.DestroyQueue(q);
}

// vLLM e126687a9a sampler.py: greedy sampling is torch.argmax(dim=-1).
// Large and ragged rows enter the same public selector used by Gemma decode.
TEST_CASE("ROCm large-vocabulary greedy selection preserves the first maximum") {
  if (NoDevice()) return;
  Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  Queue q = rocm.CreateQueue();
  constexpr int64_t rows = 4;
  for (int64_t cols : {int64_t{65537}, int64_t{262208}}) {
    std::vector<float> values(rows * cols, -std::numeric_limits<float>::infinity());
    values[511] = values[1024] = values[cols - 1] = 8.f;
    values[2 * cols - 1] = 2.f;
    values[3 * cols + 1023] = 9.f;
    std::vector<uint16_t> bf16(values.size());
    for (size_t i = 0; i < values.size(); ++i) bf16[i] = vt::F32ToBF16(values[i]);
    for (DType dtype : {DType::kF32, DType::kBF16}) {
      const size_t bytes = values.size() * vt::SizeOf(dtype);
      void* device_values = rocm.Alloc(bytes);
      void* device_ids = rocm.Alloc(rows * sizeof(int64_t));
      rocm.Copy(q, device_values,
                dtype == DType::kF32 ? static_cast<void*>(values.data())
                                     : static_cast<void*>(bf16.data()),
                bytes);
      Tensor logits = Tensor::Contiguous(device_values, dtype, q.device, {rows, cols});
      Tensor ids = Tensor::Contiguous(device_ids, DType::kI64, q.device, {rows});
      if (dtype == DType::kBF16) {
        // The public sampling contract accepts FP32 logits; the runner widens
        // Gemma's BF16 projection before reaching this operation.
        CHECK_THROWS_WITH_AS(vt::GreedyArgmax(q, ids, logits),
                             doctest::Contains("logits must be f32"), std::runtime_error);
      } else {
        vt::GreedyArgmax(q, ids, logits);
        std::vector<int64_t> actual(rows);
        rocm.Copy(q, actual.data(), device_ids, rows * sizeof(int64_t));
        rocm.Synchronize(q);
        CHECK(actual == std::vector<int64_t>{511, cols - 1, 0, 1023});
      }
      rocm.Synchronize(q);
      rocm.Free(device_values);
      rocm.Free(device_ids);
    }
  }
  rocm.DestroyQueue(q);
}

TEST_CASE("ROCm backend registers when a device is present") {
  if (NoDevice()) return;
  Backend& rocm = vt::GetBackend(DeviceType::kROCM);

  // The arch string is the single most useful thing a first bug report can
  // carry, so print it unconditionally rather than only on failure.
  const std::string arch = vt::rocm::DeviceArchName(0);
  MESSAGE("ROCm device 0 gcnArchName: ", arch);
  CHECK_FALSE(arch.empty());

  // The capability must agree with the parse the CPU-tier test already gates
  // (tests/vt/test_rocm_arch.cpp). A disagreement means the backend took its
  // fallback path (props.major/.minor) because the string did not parse, which
  // is worth knowing loudly.
  const auto parsed = vt::rocm::CapabilityFromGcnArch(arch);
  CHECK_MESSAGE(parsed.has_value(), "gcnArchName did not parse: ", arch);
  if (parsed) {
    CHECK(rocm.DeviceCapabilityMajor() == parsed->first);
    CHECK(rocm.DeviceCapabilityMinor() == parsed->second);
  }
}

TEST_CASE("alloc / copy / memset round-trip through the ROCm backend") {
  if (NoDevice()) return;
  Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  Queue q = rocm.CreateQueue();
  REQUIRE(q.device.type == DeviceType::kROCM);

  constexpr size_t kN = 1024;
  std::vector<float> host(kN);
  for (size_t i = 0; i < kN; ++i) host[i] = static_cast<float>(i) * 0.5f;
  std::vector<float> back(kN, -1.0f);

  void* dev = rocm.Alloc(kN * sizeof(float));
  REQUIRE(dev != nullptr);
  // Backend::Alloc owes >= 64B alignment (StepArena depends on it).
  CHECK((reinterpret_cast<uintptr_t>(dev) % 64) == 0);

  rocm.Copy(q, dev, host.data(), kN * sizeof(float));
  rocm.Copy(q, back.data(), dev, kN * sizeof(float));
  rocm.Synchronize(q);
  // A pure copy path is BIT-exact — nothing is reassociated, so anything less
  // would be hiding a bug (the contract in test_backend_cross_device.cpp).
  CHECK(std::memcmp(host.data(), back.data(), kN * sizeof(float)) == 0);

  rocm.Memset(q, dev, 0, kN * sizeof(float));
  rocm.Copy(q, back.data(), dev, kN * sizeof(float));
  rocm.Synchronize(q);
  for (size_t i = 0; i < kN; ++i) REQUIRE(back[i] == 0.0f);

  rocm.Free(dev);
  rocm.DestroyQueue(q);
  CHECK(q.handle == nullptr);
}

TEST_CASE("the reference tier follows UnifiedMemory, which is the memory-safety gate") {
  if (NoDevice()) return;
  Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  const bool unified = rocm.UnifiedMemory();
  MESSAGE("ROCm device 0 UnifiedMemory(): ", unified);

  // THE property this whole skeleton turns on. The portable CPU reference tier
  // serves unimplemented ops by running CPU kernels on the SAME pointers, which
  // is valid only where host and device memory alias. On an APU that is what
  // makes a model run end to end with one registered kernel; on a discrete card
  // it would be memory corruption, so eligibility must track UnifiedMemory()
  // exactly — not the device type, not the arch name.
  CHECK(vt::ReferenceTierEligible(DeviceType::kROCM) == unified);

  if (unified) {
    // Installing is idempotent and must not displace the native RmsNorm: the
    // tier registers strictly below any native kernel.
    const int installed = vt::RegisterReferenceTier(DeviceType::kROCM);
    MESSAGE("reference-tier ops installed for kROCM: ", installed);
    CHECK(installed > 0);
  }
}

TEST_CASE("#2511: the alloc path, the unified claim and the fault window move together") {
  if (NoDevice()) return;
  Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  const bool integrated = vt::rocm::IntegratedDevice(0);
  const bool managed = vt::rocm::ManagedAllocActive(0);
  const bool unified = rocm.UnifiedMemory();
  const bool host_pageable = vt::rocm::HostMemoryIsDeviceAddressable(0);
  // Printed unconditionally: this quadruple is the first thing a bring-up report
  // on issue #41 or #2511 should carry.
  MESSAGE("ROCm device 0 integrated: ", integrated, " managed-alloc: ", managed,
          " UnifiedMemory(): ", unified, " integrated+pageable: ", host_pageable);

  if (!integrated) {
    // DISCRETE (7900 XTX, R9700): the managed branch must be provably dead and
    // the unified claim false — the byte-identical-to-W0 half of the (b)
    // decision. A CPU fallback here would be memory corruption, not a slow
    // path, so these two CHECKs are the memory-safety gate itself.
    CHECK_FALSE(managed);
    CHECK_FALSE(unified);
    return;
  }

  // INTEGRATED. This case asserted the OPPOSITE until #2511, and the reason it
  // changed is measured rather than argued: approach (b) gave every integrated
  // managed-capable device migratable memory, and on an XNACK-less part —
  // PageableMemoryAccess = 0, so no recoverable page fault — the driver
  // migrating a page out from under a live queue is a hard violation. gfx1151
  // measured 17 GPU faults in 21 legs on that branch and 0 in 21 without it.
  //
  // So the managed branch now requires the part to be able to fault and recover,
  // which is exactly `HostMemoryIsDeviceAddressable(0)` on an integrated device.
  CHECK_MESSAGE(managed == host_pageable,
                "the managed branch must be taken EXACTLY where the device can "
                "take a recoverable page fault (#2511) — post the quadruple "
                "above on https://github.com/mudler/vllm.cpp/issues/2511");
  // And the capability claim follows the ALLOCATOR, never the capability the
  // allocator declined. Under the shipped default the managed branch implies
  // ground 1, so this collapses to the W0 conjunction on every integrated part.
  CHECK_MESSAGE(unified == host_pageable,
                "UnifiedMemory() must follow the allocation path actually taken");
  CHECK(rocm.DeviceMemoryIsHostAddressable() == unified);

  // The reference tier is the consequence, and it is the whole reason this
  // needed a decision. Where the claim was withdrawn, the backend must be able
  // to SAY WHY: an op that loses its fallback is refused with this sentence
  // appended, so a Strix Halo owner learns the cause at the point of failure.
  if (!unified) {
    CHECK_FALSE(vt::ReferenceTierEligible(DeviceType::kROCM));
    REQUIRE(rocm.HostAddressabilityNote() != nullptr);
    CHECK(std::string(rocm.HostAddressabilityNote()).find("2511") != std::string::npos);
  } else {
    CHECK(rocm.HostAddressabilityNote() == nullptr);
  }
}

TEST_CASE("unified path: a kernel-written value is host-readable with no copy") {
  if (NoDevice()) return;
  Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  // On a discrete card a host dereference of Backend::Alloc memory is
  // undefined behavior, so this gate only exists where UnifiedMemory() claims
  // it is safe — which is exactly the claim under test.
  if (!rocm.UnifiedMemory()) return;

  // Issue #41 F6's decisive experiment ("a kernel writes ...; the host reads
  // ... back directly, no hipMemcpy"), turned into the standing gate. The
  // kernel is the one op this backend registers (RmsNorm), so the file stays
  // free of HIP: host WRITES the inputs directly (what a reference-tier CPU
  // kernel does), the native device kernel reads them, and the host READS the
  // device-written output directly. Same golden row as the native-RmsNorm case
  // below, so a numeric mismatch here isolates COHERENCE, not arithmetic.
  Queue q = rocm.CreateQueue();
  float* dx = static_cast<float*>(rocm.Alloc(2 * sizeof(float)));
  float* dw = static_cast<float*>(rocm.Alloc(2 * sizeof(float)));
  float* dout = static_cast<float*>(rocm.Alloc(2 * sizeof(float)));
  REQUIRE(dx != nullptr);
  REQUIRE(dw != nullptr);
  REQUIRE(dout != nullptr);

  // Host writes, no Copy staging.
  dx[0] = 3.0f;
  dx[1] = 4.0f;
  dw[0] = 2.0f;
  dw[1] = 0.5f;
  dout[0] = -1.0f;
  dout[1] = -1.0f;

  const Device dev{DeviceType::kROCM, 0};
  Tensor tx = Tensor::Contiguous(dx, DType::kF32, dev, {1, 2});
  Tensor tw = Tensor::Contiguous(dw, DType::kF32, dev, {2});
  Tensor to = Tensor::Contiguous(dout, DType::kF32, dev, {1, 2});
  vt::RmsNorm(q, to, tx, tw, vt::RmsNormArgs{0.0f, false});
  rocm.Synchronize(q);

  // Host reads the device-written output directly, no Copy back. If this
  // faults or reads the -1.0f sentinels, UnifiedMemory() lied — the exact
  // failure mode approach (b) exists to make impossible.
  CHECK(dout[0] == doctest::Approx(1.697056f));
  CHECK(dout[1] == doctest::Approx(0.565685f));

  rocm.Free(dx);
  rocm.Free(dw);
  rocm.Free(dout);
  rocm.DestroyQueue(q);
}

TEST_CASE("FlushPending orders in-flight device work before a host read (#2498)") {
  if (NoDevice()) return;
  Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  // Same precondition as the unified-path case above: a host dereference of
  // Backend::Alloc memory is only defined where UnifiedMemory() says so. That is
  // also exactly where the portable CPU reference tier installs
  // (op_provider.h SAFETY), so this case runs on precisely the boards where the
  // behavior under test is reachable.
  if (!rocm.UnifiedMemory()) return;

  // WHAT THIS PINS. `GetOp` calls Backend::FlushPending() before handing back a
  // reference-tier kernel (src/vt/op_provider.cpp:707-711), and so does the
  // decline path (762-765), because that kernel is a HOST function about to read
  // DEVICE memory with no Queue in hand. Metal and Vulkan override it. ROCm
  // inherited the `{}` default from vt/backend.h, whose contract says it "suits
  // every backend that submits eagerly" -- and HIP submits eagerly but COMPLETES
  // asynchronously, since CreateQueue builds a real hipStreamCreate stream.
  //
  // So the assertion is deliberately about ORDERING and not about arithmetic:
  // the host must not observe its own pre-launch sentinels after FlushPending().
  // Note what is NOT called below -- there is no Synchronize(q) anywhere. The
  // only thing between the launches and the host read is FlushPending().
  const int64_t rows = 2048;
  const int64_t cols = 1024;
  const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t bytes = n * sizeof(float);

  float* dx = static_cast<float*>(rocm.Alloc(bytes));
  float* dw = static_cast<float*>(rocm.Alloc(cols * sizeof(float)));
  float* dout = static_cast<float*>(rocm.Alloc(bytes));
  REQUIRE(dx != nullptr);
  REQUIRE(dw != nullptr);
  REQUIRE(dout != nullptr);

  // Host stores BEFORE anything is enqueued, so this fill races nothing. Every
  // element of a row is 3, so mean(x^2) = 9 and rms = 3 exactly; with eps 0 the
  // row reduces to out[j] = (3/3) * w[j] = w[j] = 2. An exact power-of-two-free
  // identity, chosen so a partially-written row cannot round into the answer.
  for (size_t i = 0; i < n; ++i) dx[i] = 3.0f;
  for (int64_t j = 0; j < cols; ++j) dw[j] = 2.0f;
  for (size_t i = 0; i < n; ++i) dout[i] = -1.0f;

  const Device dev{DeviceType::kROCM, 0};
  Tensor tx = Tensor::Contiguous(dx, DType::kF32, dev, {rows, cols});
  Tensor tw = Tensor::Contiguous(dw, DType::kF32, dev, {cols});
  Tensor to = Tensor::Contiguous(dout, DType::kF32, dev, {rows, cols});

  Queue q = rocm.CreateQueue();
  // Enough queued work that the stream is genuinely busy when the host regains
  // control. One launch would make the outcome a coin toss on launch latency;
  // this makes the un-drained window milliseconds wide, so the mutation below
  // fails for the reason it names rather than by luck.
  for (int i = 0; i < 128; ++i) {
    vt::RmsNorm(q, to, tx, tw, vt::RmsNormArgs{0.0f, false});
  }

  rocm.FlushPending();

  // The host reads managed memory directly, exactly as a reference-tier kernel
  // would. Reading -1.0f here is the defect: it means the host observed bytes
  // the device had not written.
  CHECK(dout[0] == doctest::Approx(2.0f));
  CHECK(dout[n / 2] == doctest::Approx(2.0f));
  CHECK(dout[n - 1] == doctest::Approx(2.0f));

  rocm.Synchronize(q);
  rocm.Free(dx);
  rocm.Free(dw);
  rocm.Free(dout);
  rocm.DestroyQueue(q);
}

TEST_CASE("RmsNorm is registered natively, and the tier does not displace it") {
  if (NoDevice()) return;
  // Seam 3: the op table. One op today (src/vt/rocm/rocm_ops.hip).
  CHECK(vt::OpRegistered(vt::OpId::kRmsNorm, DeviceType::kROCM));

  Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  Queue q = rocm.CreateQueue();

  // x = [3,4]; mean(x^2) = 12.5; rms = sqrt(12.5); w = [2, 0.5]
  // out = [3/3.53553*2, 4/3.53553*0.5] = [1.697056, 0.565685]
  // Same golden row as tests/vt/test_ops_rmsnorm.cpp, so the two backends are
  // pinned to one arithmetic statement rather than to each other.
  const std::vector<float> x = {3.0f, 4.0f};
  const std::vector<float> w = {2.0f, 0.5f};
  std::vector<float> out(2, 0.0f);

  void* dx = rocm.Alloc(x.size() * sizeof(float));
  void* dw = rocm.Alloc(w.size() * sizeof(float));
  void* dout = rocm.Alloc(out.size() * sizeof(float));
  rocm.Copy(q, dx, x.data(), x.size() * sizeof(float));
  rocm.Copy(q, dw, w.data(), w.size() * sizeof(float));

  const Device dev{DeviceType::kROCM, 0};
  Tensor tx = Tensor::Contiguous(dx, DType::kF32, dev, {1, 2});
  Tensor tw = Tensor::Contiguous(dw, DType::kF32, dev, {2});
  Tensor to = Tensor::Contiguous(dout, DType::kF32, dev, {1, 2});
  vt::RmsNorm(q, to, tx, tw, vt::RmsNormArgs{0.0f, false});

  rocm.Copy(q, out.data(), dout, out.size() * sizeof(float));
  rocm.Synchronize(q);
  CHECK(out[0] == doctest::Approx(1.697056f));
  CHECK(out[1] == doctest::Approx(0.565685f));

  // The native kernel RAN — it was not silently served by the CPU fallback. A
  // reference-tier hit here would mean the registration in rocm_ops.hip did not
  // take effect, and the numbers above would still pass, which is exactly the
  // kind of pass that teaches nothing.
  CHECK(vt::GetReferenceTierHits() == 0);

  rocm.Free(dx);
  rocm.Free(dw);
  rocm.Free(dout);
  rocm.DestroyQueue(q);
}

TEST_CASE("the ROCm platform self-registers and is selected over CPU") {
  if (NoDevice()) return;
  using vllm::platforms::CurrentPlatform;
  using vllm::platforms::GetPlatform;
  using vllm::platforms::HasPlatform;

  REQUIRE(HasPlatform(DeviceType::kROCM));
  const auto& rocm = GetPlatform(DeviceType::kROCM);
  CHECK(rocm.device_type() == DeviceType::kROCM);
  CHECK(rocm.get_device_capability().present());
  CHECK(rocm.supported_dtypes().size() == 3);

  // Accelerator-first: on an AMD box with no CUDA, the process platform is ROCm.
  // This is what the kCurrentPriority walk in platform.cpp decides, and the CPU
  // tier gates its membership (tests/vllm/platforms/test_platform.cpp).
  if (!HasPlatform(DeviceType::kCUDA)) {
    CHECK(&CurrentPlatform() == &rocm);
  }

  // M3: ROCM_ATTN is registered for kROCM and the priority list mirrors
  // rocm.py:424-434 (AITER entries gated off on RDNA3). The dense walk resolves
  // to the first REGISTERED name — ROCM_ATTN — and constructs the named backend.
  const auto dense_prio = rocm.get_attn_backend_priority({});
  // Verbatim mirror of rocm.py:407-441 _get_backend_priorities (dense branch)
  // at pin 555967922 — the AITER entries are gated on is_mha_enabled() /
  // is_aiter_found_and_supported() upstream (:434,:436) and are named-but-
  // unregistered placeholders here, skipped by the walk.
  const std::vector<std::string> expected_dense{
      "ROCM_ATTN", "ROCM_AITER_FA", "ROCM_AITER_UNIFIED_ATTN",
      "TRITON_ATTN", "TURBOQUANT"};
  CHECK(dense_prio == expected_dense);
  CHECK(vllm::v1::HasAttentionBackend(DeviceType::kROCM, "ROCM_ATTN"));
  CHECK(vllm::v1::SelectAttentionBackendName(rocm) == "ROCM_ATTN");
  std::unique_ptr<vllm::v1::AttentionBackend> b =
      vllm::v1::SelectAttentionBackend(rocm);
  REQUIRE(b != nullptr);
  CHECK(b->get_name() == "ROCM_ATTN");
  // The NHD KV shape the local ROCm kernel reads (KV-layout deviation, backend.h).
  const std::vector<int64_t> shape = b->get_kv_cache_shape(10, 16, 2, 128);
  const std::vector<int64_t> expected_shape{10, 2, 16, 2, 128};
  CHECK(shape == expected_shape);
  // TRITON_ATTN / TURBOQUANT are named but unregistered -> skipped, not picked.
  CHECK_FALSE(vllm::v1::HasAttentionBackend(DeviceType::kROCM, "TRITON_ATTN"));
  CHECK_FALSE(vllm::v1::HasAttentionBackend(DeviceType::kROCM, "TURBOQUANT"));
}

// Mirrors "CUDA backend: graph capture/replay re-executes captured ops" in
// tests/vt/test_cuda_backend.cpp assertion for assertion. Same shape, same
// persistent-buffer contract, hipGraph underneath.
//
// Step 5 is the load-bearing one and the reason this test exists. Replaying
// must RE-EXECUTE the captured copy over the persistent buffers, not replay a
// snapshot of their contents — that is precisely how a decode graph picks up
// each new token's inputs. A capture that bakes values instead of addresses
// passes step 4 and fails step 5, which is the silent-correctness-bug shape
// rocm_backend.hip's scope note warns about.
TEST_CASE("ROCm backend: graph capture/replay re-executes captured ops") {
  if (NoDevice()) return;
  Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  CHECK(rocm.SupportsGraphCapture());

  Queue q = rocm.CreateQueue();
  constexpr size_t kBytes = 64 * 1024;

  // Allocated ONCE; the pointers stay fixed across every replay below. Only
  // their CONTENTS change — the capture contract.
  void* src = rocm.Alloc(kBytes);
  void* dst = rocm.Alloc(kBytes);

  std::vector<unsigned char> pattern_a(kBytes, 0x11);
  std::vector<unsigned char> pattern_b(kBytes, 0x22);
  std::vector<unsigned char> back(kBytes, 0);

  rocm.Copy(q, src, pattern_a.data(), kBytes);
  rocm.Memset(q, dst, 0, kBytes);
  rocm.Synchronize(q);

  // Recorded, NOT executed: dst must still be zero after EndCapture.
  rocm.BeginCapture(q);
  rocm.Copy(q, dst, src, kBytes);
  rocm.EndCapture(q);
  rocm.Copy(q, back.data(), dst, kBytes);
  rocm.Synchronize(q);
  CHECK(back.front() == 0x00);

  // Replay #1 -> pattern A. Proves the graph ran at all.
  rocm.Replay(q);
  rocm.Synchronize(q);
  rocm.Copy(q, back.data(), dst, kBytes);
  rocm.Synchronize(q);
  CHECK(back.front() == 0x11);
  CHECK(back.back() == 0x11);

  // Mutate src in place (SAME address) -> replay must observe the new contents.
  rocm.Copy(q, src, pattern_b.data(), kBytes);
  rocm.Replay(q);
  rocm.Synchronize(q);
  rocm.Copy(q, back.data(), dst, kBytes);
  rocm.Synchronize(q);
  CHECK(back.front() == 0x22);
  CHECK(back.back() == 0x22);

  // Handle variant — the path decode graphs actually take, since they keep one
  // exec per padded batch size rather than a single stored graph. Captures
  // into a DIFFERENT destination (dst2) than the stored-graph path above
  // (dst), not the same op replayed twice: a mutation that returns the stale
  // member exec_ (still the stored-graph path's Copy(dst, src)) instead of the
  // freshly captured local exec must be distinguishable from the correct
  // behaviour, and only fails here because the two graphs write different
  // buffers.
  void* dst2 = rocm.Alloc(kBytes);
  rocm.Copy(q, src, pattern_a.data(), kBytes);
  rocm.Memset(q, dst2, 0, kBytes);
  rocm.Synchronize(q);

  rocm.BeginCapture(q);
  rocm.Copy(q, dst2, src, kBytes);
  void* graph = rocm.EndCaptureGraph(q);
  REQUIRE(graph != nullptr);

  rocm.ReplayGraph(q, graph);
  rocm.Synchronize(q);
  rocm.Copy(q, back.data(), dst2, kBytes);
  rocm.Synchronize(q);
  CHECK(back.front() == 0x11);

  rocm.Copy(q, src, pattern_b.data(), kBytes);
  rocm.ReplayGraph(q, graph);
  rocm.Synchronize(q);
  rocm.Copy(q, back.data(), dst2, kBytes);
  rocm.Synchronize(q);
  CHECK(back.front() == 0x22);
  CHECK(back.back() == 0x22);

  rocm.DestroyGraph(graph);
  rocm.Free(src);
  rocm.Free(dst);
  rocm.Free(dst2);
  rocm.DestroyQueue(q);
}

// The capture contract's allocation clause, asserted rather than assumed
// (.agents/specs/rocm-decode-graph.md D1). hipBLASLt sizes its workspace lazily
// inside the GEMM path — LtWorkspace() in rocm_matmul_hipblaslt.hip does
// hipFree+hipMalloc when a shape needs more than the current high-water mark —
// and hipblasCreate() likewise initialises on first use. Both are illegal
// mid-capture.
//
// MEASURED on gfx1200 during W1: capturing a cold GEMM fails loudly, with
// `hipMalloc: operation not permitted when stream is capturing` (and hipFree,
// and hipblasCreate INTERNAL_ERROR) — never silent corruption. Running the
// identical GEMM once beforehand grows the workspace and creates the handle, so
// the in-capture call is a pure pool hit.
//
// This case pins the MITIGATION, not the hazard: it asserts that a pre-warmed
// GEMM captures and replays correctly. Deliberately not asserting that the cold
// path throws — a future capture-safe allocator would be an improvement, and a
// test that forbade it would be a ratchet in the wrong direction.
TEST_CASE("ROCm backend: a pre-warmed GEMM captures and replays") {
  if (NoDevice()) return;
  Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  if (!rocm.SupportsGraphCapture()) return;

  Queue q = rocm.CreateQueue();
  const Device dev{DeviceType::kROCM, 0};
  constexpr int kM = 1, kN = 2048, kK = 2048;  // a decode-shaped GEMM

  const std::vector<float> ha(kM * kK, 0.01f);
  const std::vector<float> hb(kN * kK, 0.02f);
  const float expect = 0.01f * 0.02f * static_cast<float>(kK);

  void* da = rocm.Alloc(ha.size() * sizeof(float));
  void* db = rocm.Alloc(hb.size() * sizeof(float));
  void* dc = rocm.Alloc(kM * kN * sizeof(float));
  rocm.Copy(q, da, ha.data(), ha.size() * sizeof(float));
  rocm.Copy(q, db, hb.data(), hb.size() * sizeof(float));
  rocm.Synchronize(q);

  Tensor ta = Tensor::Contiguous(da, DType::kF32, dev, {kM, kK});
  Tensor tb = Tensor::Contiguous(db, DType::kF32, dev, {kN, kK});
  Tensor tc = Tensor::Contiguous(dc, DType::kF32, dev, {kM, kN});

  // Pre-warm: grows LtWorkspace's cap and creates the hipBLAS handle.
  vt::MatmulBT(q, tc, ta, tb);
  rocm.Synchronize(q);
  rocm.Memset(q, dc, 0, kM * kN * sizeof(float));
  rocm.Synchronize(q);

  rocm.BeginCapture(q);
  vt::MatmulBT(q, tc, ta, tb);
  rocm.EndCapture(q);

  rocm.Replay(q);
  rocm.Synchronize(q);
  std::vector<float> back(kM * kN, 0.0f);
  rocm.Copy(q, back.data(), dc, back.size() * sizeof(float));
  rocm.Synchronize(q);
  CHECK(back.front() == doctest::Approx(expect).epsilon(0.01));
  CHECK(back.back() == doctest::Approx(expect).epsilon(0.01));

  rocm.Free(da);
  rocm.Free(db);
  rocm.Free(dc);
  rocm.DestroyQueue(q);
}

// NO COMMA IN THE NAME BELOW, DELIBERATELY. doctest's -tc filter splits its
// argument on commas, so a case whose name contains one can never be selected by
// name: the filter matches nothing, the binary runs zero cases and exits 0, and
// that reads as a pass. Any mutation run that selects this case by -tc would then
// be measuring nothing at all.
//
// DELIBERATELY LAST IN THIS FILE. It installs the portable reference tier, which
// makes GetReferenceTierHits() non-zero for the rest of the process, and the
// native-RmsNorm case above asserts that counter is exactly 0. doctest runs cases
// in declaration order, so keeping this one at the end is what keeps the two
// compatible. Do not move it, and do not weaken the absolute assertion above.
TEST_CASE("the reference tier's flush is REACHED through GetOp and not merely callable (#2498)") {
  if (NoDevice()) return;
  Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  if (!rocm.UnifiedMemory()) return;

  // WHY THIS CASE EXISTS SEPARATELY FROM THE ONE ABOVE. That case calls
  // Backend::FlushPending() by hand, so it proves the override does its job --
  // and it would go on passing if the production call site were deleted, because
  // it never goes through one. This case enters where production enters:
  // vt::GetOp, which is what dispatches every op in the engine. Deleting the
  // guarded flush at op_provider.cpp:707-711 must make THIS fail.
  //
  // kBatchedMatmul is chosen because it has a CPU kernel and no native ROCm one,
  // which is exactly the shape that installs the tier. If a later wave ports it
  // to ROCm the REQUIRE below fails loudly, which is the intended signal to
  // repoint this case at another still-unported op rather than to delete it. As
  // of this commit the ROCm MLA/DSA arm supplies seven more candidates.
  REQUIRE_MESSAGE(!vt::OpRegistered(vt::OpId::kBatchedMatmul, DeviceType::kROCM),
                  "kBatchedMatmul now has a native ROCm kernel; repoint this case "
                  "at an op that still has none (see the MLA/DSA arm)");
  REQUIRE(vt::OpRegistered(vt::OpId::kBatchedMatmul, DeviceType::kCPU));

  const int64_t rows = 2048;
  const int64_t cols = 1024;
  const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t bytes = n * sizeof(float);

  float* dx = static_cast<float*>(rocm.Alloc(bytes));
  float* dw = static_cast<float*>(rocm.Alloc(cols * sizeof(float)));
  float* dout = static_cast<float*>(rocm.Alloc(bytes));
  REQUIRE(dx != nullptr);
  REQUIRE(dw != nullptr);
  REQUIRE(dout != nullptr);

  // Same identity as the case above: every element 3, so rms is exactly 3 and
  // out[j] = w[j] = 2. Host stores happen before anything is enqueued.
  for (size_t i = 0; i < n; ++i) dx[i] = 3.0f;
  for (int64_t j = 0; j < cols; ++j) dw[j] = 2.0f;
  for (size_t i = 0; i < n; ++i) dout[i] = -1.0f;

  const Device dev{DeviceType::kROCM, 0};
  Tensor tx = Tensor::Contiguous(dx, DType::kF32, dev, {rows, cols});
  Tensor tw = Tensor::Contiguous(dw, DType::kF32, dev, {cols});
  Tensor to = Tensor::Contiguous(dout, DType::kF32, dev, {rows, cols});

  Queue q = rocm.CreateQueue();
  for (int i = 0; i < 128; ++i) {
    vt::RmsNorm(q, to, tx, tw, vt::RmsNormArgs{0.0f, false});
  }

  // THE PRODUCTION ENTRY POINT. Resolving an op with no native ROCm kernel
  // installs the CPU host kernel and, because the slot is now reference-tier,
  // GetOp drains the device before handing it back. Nothing else here syncs.
  const unsigned long long before = vt::GetReferenceTierHits();
  void* fn = vt::GetOp(vt::OpId::kBatchedMatmul, DeviceType::kROCM);
  CHECK(fn != nullptr);
  // The tier really was what answered -- otherwise the drain above never ran and
  // the reads below would be measuring nothing.
  CHECK(vt::GetReferenceTierHits() > before);

  // A reference-tier kernel would now dereference these host-addressable
  // pointers. Sentinels here mean it would have read bytes the device had not
  // written.
  CHECK(dout[0] == doctest::Approx(2.0f));
  CHECK(dout[n / 2] == doctest::Approx(2.0f));
  CHECK(dout[n - 1] == doctest::Approx(2.0f));

  rocm.Synchronize(q);
  rocm.Free(dx);
  rocm.Free(dw);
  rocm.Free(dout);
  rocm.DestroyQueue(q);
}

// RmsNormRowKernel's arithmetic, emulated on the host operation for operation:
// thread t of the 256-thread block accumulates j = t, t+256, ... in order, the
// shared-memory tree halves 128..1, then out = (v * inv) * w. The device result
// must match BYTE for BYTE, so any change to which thread owns an element or to
// the order it is accumulated in fails here, not in a served token.
namespace {

float RoundBf16(float v) { return vt::BF16ToF32(vt::F32ToBF16(v)); }

void RmsNormRowEmulated(std::vector<float>& out, std::vector<float>& res, const std::vector<float>& x,
                        const std::vector<float>& w, int64_t rows, int64_t h, float eps, bool gemma,
                        bool has_res, bool bf16_store) {
  constexpr int kBlock = 256;
  for (int64_t r = 0; r < rows; ++r) {
    float partial[kBlock];
    for (int t = 0; t < kBlock; ++t) {
      float acc = 0.0f;
      for (int64_t j = t; j < h; j += kBlock) {
        float v = x[r * h + j];
        if (has_res) {
          v = v + res[r * h + j];
          if (bf16_store) v = RoundBf16(v);
          res[r * h + j] = v;
        }
        acc += v * v;
      }
      partial[t] = acc;
    }
    for (int s = kBlock / 2; s > 0; s /= 2)
      for (int t = 0; t < s; ++t) partial[t] += partial[t + s];
    const float inv = 1.0f / std::sqrt(partial[0] / static_cast<float>(h) + eps);
    for (int64_t j = 0; j < h; ++j) {
      const float v = has_res ? res[r * h + j] : x[r * h + j];
      float wj = w[j];
      if (gemma) wj += 1.0f;
      const float o = v * inv * wj;
      out[r * h + j] = bf16_store ? RoundBf16(o) : o;
    }
  }
}

}  // namespace

TEST_CASE("rocm RmsNorm row kernel is byte-identical to its host emulation") {
  if (NoDevice()) return;
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue q = gpu.CreateQueue();
  std::mt19937 rng(0x5EED);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  for (const DType dt : {DType::kBF16, DType::kF32}) {
    const bool bf16 = dt == DType::kBF16;
    const size_t es = bf16 ? 2 : 4;
    for (const int64_t h : {int64_t{5120}, int64_t{2051}, int64_t{17408}, int64_t{17}}) {
      for (const bool has_res : {true, false}) {
        for (const bool gemma : {false, true}) {
          CAPTURE(bf16);
          CAPTURE(h);
          CAPTURE(has_res);
          CAPTURE(gemma);
          const int64_t rows = 3;
          std::vector<float> x(rows * h), r(rows * h), w(h);
          for (auto* v : {&x, &r, &w})
            for (auto& e : *v) e = bf16 ? RoundBf16(dist(rng)) : dist(rng);
          auto pack = [&](const std::vector<float>& f) {
            std::vector<uint8_t> b(f.size() * es);
            for (size_t i = 0; i < f.size(); ++i) {
              if (bf16) {
                const uint16_t u = vt::F32ToBF16(f[i]);
                std::memcpy(b.data() + 2 * i, &u, 2);
              } else {
                std::memcpy(b.data() + 4 * i, &f[i], 4);
              }
            }
            return b;
          };
          const auto xb = pack(x), rb = pack(r), wb = pack(w);
          void* dx = gpu.Alloc(xb.size());
          void* dr = gpu.Alloc(rb.size());
          void* dw = gpu.Alloc(wb.size());
          void* dout = gpu.Alloc(xb.size());
          gpu.Copy(q, dx, xb.data(), xb.size());
          gpu.Copy(q, dr, rb.data(), rb.size());
          gpu.Copy(q, dw, wb.data(), wb.size());
          const Device dev{DeviceType::kROCM, 0};
          Tensor tx = Tensor::Contiguous(dx, dt, dev, {rows, h});
          Tensor tr = Tensor::Contiguous(dr, dt, dev, {rows, h});
          Tensor tw = Tensor::Contiguous(dw, dt, dev, {h});
          Tensor tout = Tensor::Contiguous(dout, dt, dev, {rows, h});
          vt::RmsNormArgs args;
          args.eps = 1e-6f;
          args.gemma = gemma;
          vt::RmsNorm(q, tout, tx, tw, args, has_res ? &tr : nullptr);
          std::vector<uint8_t> got_out(xb.size()), got_res(rb.size());
          gpu.Copy(q, got_out.data(), dout, got_out.size());
          gpu.Copy(q, got_res.data(), dr, got_res.size());
          gpu.Synchronize(q);
          std::vector<float> want_out(rows * h), want_res = r;
          RmsNormRowEmulated(want_out, want_res, x, w, rows, h, 1e-6f, gemma, has_res, bf16);
          CHECK(got_out == pack(want_out));
          if (has_res) CHECK(got_res == pack(want_res));
          gpu.Free(dx);
          gpu.Free(dr);
          gpu.Free(dw);
          gpu.Free(dout);
        }
      }
    }
  }
  gpu.DestroyQueue(q);
}
