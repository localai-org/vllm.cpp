// ENG-LOAD-DIRECT-UPLOAD (issue #150) — the MECHANISM, not a timing.
//
// The claim under test is not "loading got faster"; it is that a weight the
// device consumes VERBATIM never materializes an owned host buffer at all. That
// is an observable, deterministic property of the loaded container:
//
//   * `bytes.data()` IS the address inside the safetensors mmap (so nothing was
//     copied), and `bytes.borrowed()` is true;
//   * the borrow keeps the mapping alive past ~SafetensorsFile, which is what
//     makes a LATER device upload legal (the whole design problem);
//   * the byte accounting shows the range under `borrowed`, and NOT under
//     `host_copy`;
//   * every non-verbatim path — transpose, dtype conversion, concatenation —
//     and every size/dtype mismatch still copies, i.e. the lever fails closed.
//
// A timing test would pass with the copy still in place. These do not.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>
#if defined(__linux__) && defined(__GLIBC__)
#include <malloc.h>  // mallopt — pinned for the mincore residency observation below
#endif

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_device_glue.h"  // dense_attn::Dev
#include "vllm/model_executor/models/dense_nvfp4_gemm.h"   // dense_nvfp4::ResidentNvfp4
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vllm/model_executor/models/owned_bytes.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vt/backend.h"

namespace {

std::string U64Le(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) s[i] = static_cast<char>((v >> (8 * i)) & 0xff);
  return s;
}

class TempFile {
 public:
  explicit TempFile(const std::string& bytes) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_direct_upload_test_" + std::to_string(counter++) +
              ".safetensors"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  ~TempFile() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// Two BF16 [4,2] tensors ("w", "v") and one F32 [4] tensor ("f"), with distinct
// contents so a wrong copy is visible in the values, not only in the pointer.
constexpr int64_t kRows = 4;
constexpr int64_t kCols = 2;
constexpr size_t kBf16Bytes = static_cast<size_t>(kRows * kCols) * 2;
constexpr size_t kF32Bytes = static_cast<size_t>(kRows) * 4;

std::string Body() {
  std::string data(kBf16Bytes * 2 + kF32Bytes, '\0');
  std::vector<uint16_t> w(kRows * kCols);
  std::vector<uint16_t> v(kRows * kCols);
  for (size_t i = 0; i < w.size(); ++i) {
    w[i] = static_cast<uint16_t>(0x3f00 + i);
    v[i] = static_cast<uint16_t>(0x4100 + i);
  }
  std::memcpy(data.data(), w.data(), kBf16Bytes);
  std::memcpy(data.data() + kBf16Bytes, v.data(), kBf16Bytes);
  const float f[kRows] = {1.5F, 2.5F, 3.5F, 4.5F};
  std::memcpy(data.data() + 2 * kBf16Bytes, f, kF32Bytes);
  return data;
}

std::string Header() {
  return
      R"({"w":{"dtype":"BF16","shape":[4,2],"data_offsets":[0,16]},)"
      R"("v":{"dtype":"BF16","shape":[4,2],"data_offsets":[16,32]},)"
      R"("f":{"dtype":"F32","shape":[4],"data_offsets":[32,48]}})";
}

std::string File() {
  const std::string h = Header();
  return U64Le(h.size()) + h + Body();
}

vllm::TensorResolver ResolverFor(const vllm::SafetensorsFile& st) {
  return [&st](const std::string& name) -> const vllm::StTensor& {
    return st.Get(name);
  };
}

// RAII around the process-wide test seams so one failing CHECK cannot leak a
// forced decision into the next TEST_CASE.
class ForcedArm {
 public:
  explicit ForcedArm(bool direct) {
    vllm::detail::SetLoadDirectUploadOverrideForTesting(direct);
    // The windowed source-page release madvises the source range away after a
    // copy. Reading a copied tensor's SOURCE afterwards is legal (the pages
    // re-fault) but the assertions below compare against it, so keep it off to
    // isolate the one behavior under test.
    vllm::detail::SetLoadWindowedReleaseOverrideForTesting(false);
  }
  ~ForcedArm() {
    vllm::detail::SetLoadDirectUploadOverrideForTesting(std::nullopt);
    vllm::detail::SetLoadWindowedReleaseOverrideForTesting(std::nullopt);
  }
};

}  // namespace

TEST_CASE("direct upload: a verbatim BF16 weight VIEWS the mapping, no host copy") {
  ForcedArm arm(true);
  TempFile f(File());
  vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
  const vllm::StTensor& src = st.Get("w");

  const vllm::load_stats::Counters before = vllm::load_stats::Snapshot();
  const vllm::OwnedTensor w = vllm::dense_loaders::LoadBf16Direct(ResolverFor(st), "w");
  const vllm::load_stats::Counters after = vllm::load_stats::Snapshot();

  // THE MECHANISM: the weight's bytes ARE the mapping's bytes. No owned buffer
  // was allocated, so there is nothing for a device upload to read but the file.
  CHECK(w.bytes.borrowed());
  CHECK(static_cast<const void*>(w.bytes.data()) ==
        static_cast<const void*>(src.data));
  CHECK(w.bytes.size() == kBf16Bytes);
  CHECK(w.mmap_src == static_cast<const void*>(src.data));
  CHECK(w.mmap_src_bytes == kBf16Bytes);

  // ... and the accounting agrees: these bytes were BORROWED, never copied.
  CHECK(after.borrowed_bytes - before.borrowed_bytes == kBf16Bytes);
  CHECK(after.host_copy_bytes == before.host_copy_bytes);

  // Metadata is the same as the copy arm would produce.
  CHECK(w.dtype == vt::DType::kBF16);
  CHECK(w.rank == 2);
  CHECK(w.shape[0] == kRows);
  CHECK(w.shape[1] == kCols);
}

TEST_CASE("direct upload: the borrow keeps the mapping alive past ~SafetensorsFile") {
  ForcedArm arm(true);
  TempFile f(File());
  std::vector<uint16_t> expected;
  vllm::OwnedTensor w;
  {
    vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
    w = vllm::dense_loaders::LoadBf16Direct(ResolverFor(st), "w");
    expected.resize(kRows * kCols);
    for (size_t i = 0; i < expected.size(); ++i) {
      std::memcpy(&expected[i], w.bytes.data() + i * sizeof(uint16_t),
                  sizeof(uint16_t));
    }
  }
  // The SafetensorsFile is gone. This is the lifetime question the lazy upload
  // poses: the mapping must still be readable here, because ResidentWeight runs
  // long after the loader returned and the shards were released.
  REQUIRE(w.bytes.size() == kBf16Bytes);
  for (size_t i = 0; i < expected.size(); ++i) {
    uint16_t got;
    std::memcpy(&got, w.bytes.data() + i * sizeof(uint16_t), sizeof(got));
    CHECK(got == expected[i]);
    CHECK(got == static_cast<uint16_t>(0x3f00 + i));
  }
}

TEST_CASE("direct upload: OFF copies, and both arms load the SAME bytes") {
  TempFile f(File());
  std::vector<uint8_t> direct_bytes;
  std::vector<uint8_t> copied_bytes;
  {
    ForcedArm arm(true);
    vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
    const vllm::OwnedTensor w =
        vllm::dense_loaders::LoadBf16Direct(ResolverFor(st), "w");
    REQUIRE(w.bytes.borrowed());
    direct_bytes.assign(w.bytes.begin(), w.bytes.end());
  }
  {
    ForcedArm arm(false);
    vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
    const vllm::StTensor& src = st.Get("w");
    const vllm::load_stats::Counters before = vllm::load_stats::Snapshot();
    const vllm::OwnedTensor w =
        vllm::dense_loaders::LoadBf16Direct(ResolverFor(st), "w");
    const vllm::load_stats::Counters after = vllm::load_stats::Snapshot();
    CHECK_FALSE(w.bytes.borrowed());
    CHECK(static_cast<const void*>(w.bytes.data()) !=
          static_cast<const void*>(src.data));
    CHECK(w.mmap_src == nullptr);
    CHECK(after.host_copy_bytes - before.host_copy_bytes == kBf16Bytes);
    CHECK(after.borrowed_bytes == before.borrowed_bytes);
    copied_bytes.assign(w.bytes.begin(), w.bytes.end());
  }
  // A residency change may not change one byte of the model.
  CHECK(direct_bytes == copied_bytes);
}

TEST_CASE("direct upload: a TRANSPOSE is not a verbatim copy and still owns its buffer") {
  ForcedArm arm(true);
  TempFile f(File());
  vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
  const vllm::StTensor& src = st.Get("w");
  const vllm::OwnedTensor t =
      vllm::dense_loaders::LoadBf16Transposed(ResolverFor(st), "w");
  CHECK_FALSE(t.bytes.borrowed());
  CHECK(t.mmap_src == nullptr);
  CHECK(static_cast<const void*>(t.bytes.data()) !=
        static_cast<const void*>(src.data));
  // [out=4, in=2] on disk -> [in=2, out=4] in memory, transposed values.
  REQUIRE(t.rank == 2);
  CHECK(t.shape[0] == kCols);
  CHECK(t.shape[1] == kRows);
  const auto* got = reinterpret_cast<const uint16_t*>(t.bytes.data());
  for (int64_t r = 0; r < kRows; ++r) {
    for (int64_t c = 0; c < kCols; ++c) {
      CHECK(got[c * kRows + r] == static_cast<uint16_t>(0x3f00 + r * kCols + c));
    }
  }
}

TEST_CASE("direct upload: a CONCATENATION of two shards owns its merged buffer") {
  ForcedArm arm(true);
  TempFile f(File());
  vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
  const vllm::OwnedTensor merged =
      vllm::dense_loaders::LoadMergedBf16RawNK(ResolverFor(st), {"w", "v"});
  CHECK_FALSE(merged.bytes.borrowed());
  CHECK(merged.mmap_src == nullptr);
  REQUIRE(merged.bytes.size() == 2 * kBf16Bytes);
  const auto* got = reinterpret_cast<const uint16_t*>(merged.bytes.data());
  for (size_t i = 0; i < kRows * kCols; ++i) {
    CHECK(got[i] == static_cast<uint16_t>(0x3f00 + i));
    CHECK(got[kRows * kCols + i] == static_cast<uint16_t>(0x4100 + i));
  }
}

TEST_CASE("direct upload: BorrowStTensorBytes FAILS CLOSED on a size or dtype mismatch") {
  ForcedArm arm(true);
  TempFile f(File());
  vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
  const vllm::StTensor& w = st.Get("w");
  const vllm::StTensor& fl = st.Get("f");

  // Right dtype, WRONG element count for the span.
  vllm::OwnedTensor too_small;
  CHECK_FALSE(vllm::BorrowStTensorBytes(too_small, w, vt::DType::kBF16, {2, 2}));
  CHECK(too_small.bytes.empty());
  CHECK(too_small.mmap_src == nullptr);

  // Right element count, WRONG dtype width: an f32 source is twice a bf16
  // destination, which is exactly the f32->bf16 conversion arm that must copy.
  vllm::OwnedTensor wrong_dtype;
  CHECK_FALSE(vllm::BorrowStTensorBytes(wrong_dtype, fl, vt::DType::kBF16, {4}));
  CHECK(wrong_dtype.bytes.empty());

  // A synthetic StTensor with no mapping keep-alive can never be borrowed: a
  // borrow with nothing holding the memory alive is the bug this fails closed on.
  vllm::StTensor detached;
  detached.dtype = "BF16";
  detached.shape = {4, 2};
  detached.nbytes = kBf16Bytes;
  detached.data = w.data;
  detached.mapping = nullptr;
  vllm::OwnedTensor no_owner;
  CHECK_FALSE(
      vllm::BorrowStTensorBytes(no_owner, detached, vt::DType::kBF16, {4, 2}));
  CHECK(no_owner.bytes.empty());

  // The matching case still borrows, so the checks above rejected for their
  // stated reason rather than because the arm was off.
  vllm::OwnedTensor ok;
  CHECK(vllm::BorrowStTensorBytes(ok, w, vt::DType::kBF16, {4, 2}));
  CHECK(ok.bytes.borrowed());
}

// --- Post-upload residency: the adopt branch, and the ORDER it does things ---
//
// The branch under test is `AdoptDeviceBytesAsHost`'s direct-upload arm
// (`mmap_src != nullptr && bytes.borrowed()`), the one place ENG-LOAD-DIRECT-
// UPLOAD and BACKEND-VULKAN-LOADMEM meet. Nothing pinned it: deleting the whole
// arm left every existing suite green, because no test ever built the
// (borrowed + `mmap_src` + host-addressable backend) state it keys on.
//
// It has two guarantees, and ONE of them is an ORDERING, which is why these
// tests need a mapping that can be OBSERVED at the instant it is dropped:
//
//   1. The consumed source pages are released BEFORE `bytes` is re-pointed at
//      the device allocation. That assignment destroys the OwnedBytes carrying
//      this tensor's keep-alive on the mapping, and by adoption time the shard's
//      SafetensorsFile is already gone -- so for the last adopted weight of a
//      shard the assignment MUNMAPS, and a release after it would madvise an
//      address range that is no longer mapped.
//   2. The release also runs on the two early-return paths -- a backend without
//      host-addressable device memory, and `VT_ADOPT_DEVICE_BYTES=0` -- so the
//      adoption A/B does not silently move the release lever as well.
//
// THE OBSERVABLE. The stand-in mapping is a private ANONYMOUS mapping filled
// with a pattern. `MADV_DONTNEED` on such a mapping DISCARDS its contents (a
// later read returns zeroes), so "were the pages released yet?" is answerable by
// reading one byte -- including from inside the keep-alive's deleter, which runs
// at exactly the instant the mapping is dropped. Correct order => the deleter
// sees 0. Reversed order => the deleter sees the pattern.
//
// PLATFORM NOTE, recorded rather than abstracted away. That zero-fill is LINUX
// semantics: `madvise(MADV_DONTNEED)` on a private anonymous mapping frees the
// pages and the next read faults in a fresh zero page. On the BSDs and macOS
// `MADV_DONTNEED` is advisory and leaves the CONTENTS intact, so every `== 0`
// assertion in this section would read `kSrcPattern` there. The assumption is not
// new with these cases -- the already-merged adopt cases rest on it identically,
// and so does the production behavior they pin (the release exists to return RSS,
// which is what MADV_DONTNEED does on Linux). vllm.cpp gates on Linux only; a
// macOS port owes this section a residency observation with different semantics,
// not a tolerance widening.
namespace {

// Forces the direct-upload arm AND the windowed release (the release is the
// behavior under test here, unlike the loader cases above which switch it off).
class ForcedResidencyArm {
 public:
  explicit ForcedResidencyArm(bool release = true) {
    vllm::detail::SetLoadDirectUploadOverrideForTesting(true);
    vllm::detail::SetLoadWindowedReleaseOverrideForTesting(release);
  }
  ~ForcedResidencyArm() {
    vllm::detail::SetLoadDirectUploadOverrideForTesting(std::nullopt);
    vllm::detail::SetLoadWindowedReleaseOverrideForTesting(std::nullopt);
  }
  ForcedResidencyArm(const ForcedResidencyArm&) = delete;
  ForcedResidencyArm& operator=(const ForcedResidencyArm&) = delete;
};

class ScopedEnvVar {
 public:
  ScopedEnvVar(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) {
      had_old_ = true;
      old_ = old;
    }
    ::setenv(name, value, 1);
  }
  ~ScopedEnvVar() {
    if (had_old_) {
      ::setenv(name_.c_str(), old_.c_str(), 1);
    } else {
      ::unsetenv(name_.c_str());
    }
  }
  ScopedEnvVar(const ScopedEnvVar&) = delete;
  ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

 private:
  std::string name_;
  std::string old_;
  bool had_old_ = false;
};

// Minimal Backend whose "device" memory is plain host memory, so the adoption
// is observable with no accelerator. `host_addressable` is the axis under test.
class FakeBackend final : public vt::Backend {
 public:
  explicit FakeBackend(bool host_addressable)
      : host_addressable_(host_addressable) {}

  void* Alloc(size_t bytes) override { return std::malloc(bytes == 0 ? 1 : bytes); }
  void Free(void* p) override {
    ++frees;
    std::free(p);
  }
  void Memset(vt::Queue&, void* p, int value, size_t bytes) override {
    std::memset(p, value, bytes);
  }
  void Copy(vt::Queue&, void* dst, const void* src, size_t bytes) override {
    std::memcpy(dst, src, bytes);
  }
  vt::Queue CreateQueue() override {
    return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  }
  bool UnifiedMemory() const override { return true; }
  bool DeviceMemoryIsHostAddressable() const override { return host_addressable_; }

  int frees = 0;

 private:
  bool host_addressable_;
};

constexpr uint8_t kSrcPattern = 0xC3;

// A stand-in shard mapping whose drop is OBSERVABLE.
struct ObservableMapping {
  uint8_t* addr = nullptr;
  size_t size = 0;
  bool dropped = false;
  // The source byte as it read AT THE MOMENT the mapping was dropped. 0 means
  // the pages had already been released; kSrcPattern means they had not.
  uint8_t byte_at_drop = 0xFF;
};

size_t PageSize() {
  const long ps = ::sysconf(_SC_PAGESIZE);
  return static_cast<size_t>(ps > 0 ? ps : 4096);
}

// A weight that BORROWS `bytes` from an observable stand-in shard mapping and
// records the source range in `mmap_src`, with the tensor's borrow as the LAST
// reference to the mapping (the SafetensorsFile is gone by adoption time). This
// is the state the loader leaves a qualifying weight in, BEFORE any upload.
// `size` must be a whole number of pages so the interior-page release covers it.
vllm::OwnedTensor BorrowedWeight(ObservableMapping& m, size_t size) {
  m.size = size;
  void* raw = ::mmap(nullptr, m.size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  REQUIRE(raw != MAP_FAILED);
  m.addr = static_cast<uint8_t*>(raw);
  std::memset(m.addr, kSrcPattern, m.size);

  ObservableMapping* mp = &m;
  std::shared_ptr<const void> keep(static_cast<const void*>(m.addr),
                                   [mp](const void*) {
                                     mp->byte_at_drop = mp->addr[0];
                                     mp->dropped = true;
                                     ::munmap(mp->addr, mp->size);
                                   });

  vllm::OwnedTensor w;
  w.dtype = vt::DType::kI8;
  w.rank = 1;
  w.shape[0] = static_cast<int64_t>(m.size);
  w.bytes = vllm::OwnedBytes::Borrow(m.addr, m.size, keep);
  w.mmap_src = m.addr;
  w.mmap_src_bytes = m.size;
  return w;  // `keep` dies here: the weight's borrow is the only reference left
}

// Build a borrowed, uploaded weight standing exactly where ResidentWeight leaves
// one: the borrow above, plus the device copy published on `d_dev`.
vllm::OwnedTensor BorrowedUploadedWeight(vt::Backend& b, vt::Queue& q,
                                         ObservableMapping& m) {
  vllm::OwnedTensor w = BorrowedWeight(m, 2 * PageSize());
  void* p = b.Alloc(m.size);
  b.Copy(q, p, w.bytes.data(), m.size);
  vt::Backend* bk = &b;
  w.d_dev = std::shared_ptr<void>(p, [bk](void* x) { bk->Free(x); });
  return w;
}

}  // namespace

TEST_CASE("adopt: a direct-upload borrow RELEASES its source pages BEFORE it re-points bytes") {
  ForcedResidencyArm arm;
  ScopedEnvVar adopt_default("VT_ADOPT_DEVICE_BYTES", "1");
  FakeBackend b(/*host_addressable=*/true);
  vt::Queue q = b.CreateQueue();

  ObservableMapping m;
  vllm::OwnedTensor w = BorrowedUploadedWeight(b, q, m);
  REQUIRE(w.bytes.borrowed());
  REQUIRE(static_cast<const void*>(w.bytes.data()) == static_cast<const void*>(m.addr));
  REQUIRE_FALSE(m.dropped);

  vllm::AdoptDeviceBytesAsHost(b, w);

  // THE ADOPTION. The host view IS the device allocation now -- the branch that
  // no test reached before.
  CHECK(w.bytes.borrowed());
  CHECK(static_cast<const void*>(w.bytes.data()) == w.d_dev.get());
  CHECK(w.bytes.size() == m.size);
  CHECK(w.bytes.data()[0] == kSrcPattern);  // the surviving copy has the bytes
  CHECK(w.mmap_src == nullptr);             // no longer a direct-upload borrow
  CHECK(w.mmap_src_bytes == 0u);
  CHECK_FALSE(w.Empty());
  CHECK_FALSE(w.host_released);

  // THE MAPPING WENT. Re-pointing `bytes` dropped this tensor's keep-alive, and
  // it was the last one -- which is exactly why the ordering matters.
  CHECK(m.dropped);

  // THE ORDERING. Read at the instant of the drop: 0 means the source pages had
  // ALREADY been released, i.e. the madvise ran while the range was still
  // mapped. kSrcPattern here means the release was moved after the assignment
  // and would have madvise'd an unmapped range.
  CHECK(m.byte_at_drop == 0);
}

TEST_CASE("adopt: the source pages are released even where device memory is NOT host-addressable") {
  ForcedResidencyArm arm;
  ScopedEnvVar adopt_default("VT_ADOPT_DEVICE_BYTES", "1");
  FakeBackend b(/*host_addressable=*/false);
  vt::Queue q = b.CreateQueue();

  ObservableMapping m;
  vllm::OwnedTensor w = BorrowedUploadedWeight(b, q, m);

  vllm::AdoptDeviceBytesAsHost(b, w);

  // No adoption: the borrow stays a valid, re-faultable view of the mapping...
  CHECK(w.bytes.borrowed());
  CHECK(static_cast<const void*>(w.bytes.data()) == static_cast<const void*>(m.addr));
  CHECK(w.mmap_src == static_cast<const void*>(m.addr));
  CHECK_FALSE(m.dropped);
  // ... but the consumed pages were released all the same.
  CHECK(m.addr[0] == 0);
}

TEST_CASE("alias: an ALIASED direct-upload borrow still releases its source pages") {
  // ENG-EXPERT-STREAM-DEVICE W0f (#1299), found by a fresh review. W0f gave
  // `ResidentWeight` a third residency: on a platform whose kernels can
  // dereference host storage it ALIASES the bytes and never populates `d_dev`,
  // so `AdoptDeviceBytesAsHost` is never called for that weight. That function
  // is the only other caller of `ReleaseDirectUploadSource`, so the alias branch
  // became a third path past a release whose own ordering comment insists it
  // happens on EVERY path, including the `VT_ADOPT_DEVICE_BYTES=0` arm.
  //
  // A borrow reaches the alias branch when it is already
  // `kDeviceAliasAlignment`-aligned, and an `mmap` return always is, so this is
  // not a corner case: it is every direct-upload borrow on such a platform.
  // Without the fix the mapping's consumed pages stay resident for the life of
  // the process, silently, and issue #150's measurement stops being true.
  ForcedResidencyArm arm;
  ObservableMapping m;
  vllm::OwnedTensor w = BorrowedWeight(m, 2 * PageSize());
  REQUIRE(w.bytes.borrowed());
  REQUIRE(w.mmap_src == static_cast<const void*>(m.addr));
  // Asserted, not assumed: the case proves nothing if the borrow declines.
  REQUIRE(reinterpret_cast<uintptr_t>(w.bytes.data()) %
              vllm::kDeviceAliasAlignment == 0);

  CHECK(vllm::MakeHostBytesDeviceAliasable(w));

  // The borrow is UNTOUCHED, because an alias moves nothing...
  CHECK(w.bytes.borrowed());
  CHECK(static_cast<const void*>(w.bytes.data()) == static_cast<const void*>(m.addr));
  CHECK_FALSE(m.dropped);
  // ...and the consumed source pages were released all the same. Zero means the
  // madvise ran; the source pattern would mean it did not.
  CHECK(m.addr[0] == 0);
  // The record is SPENT, and clearing it is the memo. See the once-only case
  // below for why this branch needs one at all.
  CHECK(w.mmap_src == nullptr);
  CHECK(w.mmap_src_bytes == 0);
}

TEST_CASE("alias: the source pages are released ONCE, not once per forward step") {
  // THE REPEAT THE PAGE-RELEASE FIX INTRODUCED (a fresh review of #1299 caught
  // it). `AdoptDeviceBytesAsHost` calls `ReleaseDirectUploadSource` from behind
  // `if (!w.d_dev)`, so it ran exactly once per weight for the life of the
  // process. The alias branch has NO such memo: `ResidentWeight` re-tests the
  // alignment on every call, about 1,361 times per forward step on the target
  // checkpoint. Left uncleared, `mmap_src` would make every one of those calls
  // `madvise(MADV_DONTNEED)` the pages the GPU is about to read, and the kernel
  // would fault them straight back in — correct output, and a throughput
  // regression invisible to every token gate.
  //
  // THE FIRST VERSION OF THIS CASE ASSERTED ONLY THAT THE RELEASE HAPPENED. That
  // is satisfied by a release that happens every time, which is the defect. This
  // asserts the COUNT, by restoring the pattern and looking for it again.
  ForcedResidencyArm arm;
  ObservableMapping m;
  vllm::OwnedTensor w = BorrowedWeight(m, 2 * PageSize());
  REQUIRE(reinterpret_cast<uintptr_t>(w.bytes.data()) %
              vllm::kDeviceAliasAlignment == 0);

  CHECK(vllm::MakeHostBytesDeviceAliasable(w));
  REQUIRE(m.addr[0] == 0);  // the first release ran; the case is not vacuous

  // Re-arm the observation, then take the SAME branch again the way a second
  // decode step would.
  std::memset(m.addr, kSrcPattern, m.size);
  CHECK(vllm::MakeHostBytesDeviceAliasable(w));
  CHECK(vllm::MakeHostBytesDeviceAliasable(w));

  // The pattern SURVIVES: no second madvise. The alias itself still succeeded
  // above, so this is a count and not a disabled branch.
  CHECK(m.addr[0] == kSrcPattern);
  CHECK(w.bytes.borrowed());
  CHECK(static_cast<const void*>(w.bytes.data()) == static_cast<const void*>(m.addr));
}

TEST_CASE("adopt: VT_ADOPT_DEVICE_BYTES=0 moves ONLY the adoption, not the page release") {
  ForcedResidencyArm arm;
  ScopedEnvVar adopt_off("VT_ADOPT_DEVICE_BYTES", "0");
  FakeBackend b(/*host_addressable=*/true);
  vt::Queue q = b.CreateQueue();

  ObservableMapping m;
  vllm::OwnedTensor w = BorrowedUploadedWeight(b, q, m);

  vllm::AdoptDeviceBytesAsHost(b, w);

  // The documented A/B is back to the two-copy behavior...
  CHECK(w.bytes.borrowed());
  CHECK(static_cast<const void*>(w.bytes.data()) == static_cast<const void*>(m.addr));
  CHECK(static_cast<const void*>(w.bytes.data()) != w.d_dev.get());
  CHECK(w.mmap_src == static_cast<const void*>(m.addr));
  CHECK_FALSE(m.dropped);
  // ... and it did NOT also switch off the direct-upload source release, which
  // is a separate lever with its own flag (VT_LOAD_WINDOWED_RELEASE).
  CHECK(m.addr[0] == 0);
}

TEST_CASE("adopt: the windowed-release flag still governs the direct-upload release") {
  // The negative control for the three cases above: with the release gate OFF,
  // the source pages survive, so those `== 0` assertions are reading the release
  // and not some unrelated zeroing.
  ForcedResidencyArm arm(/*release=*/false);
  ScopedEnvVar adopt_default("VT_ADOPT_DEVICE_BYTES", "1");
  FakeBackend b(/*host_addressable=*/false);
  vt::Queue q = b.CreateQueue();

  ObservableMapping m;
  vllm::OwnedTensor w = BorrowedUploadedWeight(b, q, m);

  vllm::AdoptDeviceBytesAsHost(b, w);

  CHECK(m.addr[0] == kSrcPattern);
  CHECK_FALSE(m.dropped);
}

// --- The fp4 resident upload: ResidentNvfp4's accounting AND its residency ---
//
// `LoadCtNvfp4W4A16` / `LoadCtMxfp4W4A16` / `LoadCtNvfp4Raw` BORROW an fp4
// weight's `packed` and `scale` from the shard mmap, so `ResidentNvfp4` is the
// ONE host->device move of those bytes. Round 2 of this row therefore gave it
// three statements per buffer, in both copies of the function
// (`dense_nvfp4_gemm.h` and the private one in `qwen3_5.cpp`):
//
//   1. `load_stats::AddDeviceUpload(nb)`   — account the move;
//   2. `w.packed.d_dev = <owner>`          — publish the allocation on the
//      OwnedTensor, which is BOTH the only reason `AdoptDeviceBytesAsHost` can
//      act on it (that function keys on `d_dev`) and the ONE owner of those
//      bytes: an Nvfp4Weight has no second device handle to forget to release;
//   3. `AdoptDeviceBytesAsHost(d.b, w.packed)` — the post-upload residency step
//      every other qualifying weight already got.
//
// NOTHING PINNED THEM. `test_load_direct_upload` is the only suite asserting on
// `load_stats`, and it never reached `ResidentNvfp4`; the sole `ResidentNvfp4`
// case (`test_qwen36_weights.cpp`) is CUDA + 35B-shard gated and asserts nothing
// about upload accounting or post-upload residency. All six statements could be
// reverted with every fp4-touching suite still green. These cases close that:
// they drive the SHARED `dense_nvfp4::ResidentNvfp4` over the same FakeBackend +
// ObservableMapping harness the adopt cases above use, so each of the three
// statements has an assertion that fails when it is removed. (The `qwen3_5.cpp`
// duplicate lives in an anonymous namespace inside a 8.5k-line translation unit
// and is unreachable from a test; `scripts/check-fp4-resident-consistency.py`
// is what keeps it from diverging from the copy pinned here.)
namespace {

// Whole-page `packed` and `scale` sizes, consistent with an [n, k] NVFP4 weight
// (`packed` = n*k/2 bytes, `scale` = n*k/group_size bytes at group_size 16), so
// the interior-whole-page source release covers both buffers completely.
// Derived from the page size rather than hardcoded: a 64 KiB-page arm64 host
// would otherwise release nothing out of a 4 KiB `scale`.
struct Fp4Dims {
  int64_t n = 0;
  int64_t k = 256;
  size_t packed_bytes = 0;  // 8 pages
  size_t scale_bytes = 0;   // 1 page
};

Fp4Dims MakeFp4Dims() {
  Fp4Dims d;
  d.n = static_cast<int64_t>(PageSize() / 16);  // n*k == 16 * page_size
  d.packed_bytes = static_cast<size_t>(d.n * d.k) / 2;
  d.scale_bytes = static_cast<size_t>(d.n * d.k) / 16;
  return d;
}

// An fp4 weight whose packed+scale BORROW two observable stand-in mappings —
// exactly what the compressed-tensors loaders hand `ResidentNvfp4` under
// ENG-LOAD-DIRECT-UPLOAD. No device handles yet: the upload under test is what
// creates them.
vllm::Nvfp4Weight BorrowedFp4Weight(const Fp4Dims& d, ObservableMapping& mp,
                                    ObservableMapping& ms) {
  vllm::Nvfp4Weight w;
  w.n = d.n;
  w.k = d.k;
  w.group_size = 16;
  w.scale2 = 1.0F;
  w.packed = BorrowedWeight(mp, d.packed_bytes);
  w.scale = BorrowedWeight(ms, d.scale_bytes);
  return w;
}

// The OTHER residency an fp4 weight can arrive in: an OWNED anonymous buffer
// (`mmap_src == nullptr`), which is what every non-borrowing arm still produces.
// A recognizable, position-dependent fill, so reading the wrong buffer after the
// adoption is visible in the VALUES and not only in the pointer.
vllm::OwnedTensor OwnedPatternWeight(size_t size) {
  std::vector<uint8_t> v(size);
  for (size_t i = 0; i < size; ++i) v[i] = static_cast<uint8_t>(i % 251);
  vllm::OwnedTensor w;
  w.dtype = vt::DType::kI8;
  w.rank = 1;
  w.shape[0] = static_cast<int64_t>(size);
  w.bytes = vllm::OwnedBytes(std::move(v));
  return w;
}

bool AllBytesMatchPattern(const uint8_t* p, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    if (p[i] != static_cast<uint8_t>(i % 251)) return false;
  }
  return true;
}

}  // namespace

TEST_CASE("fp4 resident: ResidentNvfp4 COUNTS its upload, publishes d_dev, and adopts") {
  ForcedResidencyArm arm;
  ScopedEnvVar adopt_default("VT_ADOPT_DEVICE_BYTES", "1");
  FakeBackend b(/*host_addressable=*/true);
  vt::Queue q = b.CreateQueue();
  const Fp4Dims dims = MakeFp4Dims();

  {
    ObservableMapping mp;
    ObservableMapping ms;
    vllm::Nvfp4Weight w = BorrowedFp4Weight(dims, mp, ms);
    REQUIRE(w.packed.bytes.size() == dims.packed_bytes);
    REQUIRE(w.scale.bytes.size() == dims.scale_bytes);
    REQUIRE(w.packed.d_dev == nullptr);
    REQUIRE(w.scale.d_dev == nullptr);

    const vllm::load_stats::Counters before = vllm::load_stats::Snapshot();
    vllm::dense_attn::Dev d{b, q};
    const vllm::dense_nvfp4::Nvfp4Dev dev = vllm::dense_nvfp4::ResidentNvfp4(d, w);
    const vllm::load_stats::Counters after = vllm::load_stats::Snapshot();

    // (1) ACCOUNTING. Both buffers reached the device, and both were counted —
    // this is the ONLY place those bytes are ever counted, since they were
    // borrowed rather than copied on the way in.
    CHECK(after.device_upload_bytes - before.device_upload_bytes ==
          dims.packed_bytes + dims.scale_bytes);
    // ... and the upload is not miscounted as a host materialization.
    CHECK(after.host_copy_bytes == before.host_copy_bytes);
    CHECK(after.borrowed_bytes == before.borrowed_bytes);

    // (2) PUBLICATION. The allocation is on the OwnedTensor, not only on the
    // Nvfp4Weight's own handle. Without this the adoption below cannot happen
    // at all: AdoptDeviceBytesAsHost returns immediately on a null `d_dev`.
    REQUIRE(w.packed.d_dev != nullptr);
    REQUIRE(w.scale.d_dev != nullptr);

    // (3) ADOPTION. The host view IS the device allocation, and the weight is no
    // longer a direct-upload borrow.
    CHECK(w.packed.bytes.borrowed());
    CHECK(w.scale.bytes.borrowed());
    CHECK(static_cast<const void*>(w.packed.bytes.data()) == w.packed.d_dev.get());
    CHECK(static_cast<const void*>(w.scale.bytes.data()) == w.scale.d_dev.get());
    CHECK(w.packed.mmap_src == nullptr);
    CHECK(w.scale.mmap_src == nullptr);
    CHECK(w.packed.mmap_src_bytes == 0u);
    CHECK(w.scale.mmap_src_bytes == 0u);
    // The surviving copy carries the bytes, so every `.bytes` reader (the CPU
    // dequant fallback among them) still reads the weight.
    CHECK(w.packed.bytes.size() == dims.packed_bytes);
    CHECK(w.scale.bytes.size() == dims.scale_bytes);
    CHECK(w.packed.bytes.data()[0] == kSrcPattern);
    CHECK(w.scale.bytes.data()[0] == kSrcPattern);
    CHECK_FALSE(w.packed.host_released);
    CHECK_FALSE(w.Empty());

    // (4) THE SOURCE WENT, IN THE RIGHT ORDER. Both mappings were dropped by the
    // re-point, and both read 0 at the instant of the drop — i.e. the madvise ran
    // while the range was still mapped. kSrcPattern here would mean the release
    // had moved after the assignment, onto an unmapped range.
    CHECK(mp.dropped);
    CHECK(ms.dropped);
    CHECK(mp.byte_at_drop == 0);
    CHECK(ms.byte_at_drop == 0);

    // (5) The returned device views are the uploaded buffers.
    CHECK(dev.packed.data == w.packed.d_dev.get());
    CHECK(dev.scale.data == w.scale.d_dev.get());
  }

  // ONE control block per buffer: `packed.d_dev` is the ONE owner and the
  // adopted `bytes` view aliases the same block, so the device memory is freed
  // exactly once, through the vt Backend.
  CHECK(b.frees == 2);
}

// THE LEAK THE SINGLE OWNER REPAIRS. After a Marlin repack the fp4 originals are
// dead weight, and the repack builders are their only owner. Pre-fix, the
// release they performed dropped only the type-specific `d_packed`/`d_scale`
// pair, and `packed.d_dev` was a SECOND handle on the same control block, so the
// block survived every repack for the process lifetime -- once per repacked
// weight and once per expert per projection on an MoE model. That silently
// undid the repack's whole point (`dense_nvfp4_gemm.h`: "then FREE the fp4
// originals"), on exactly the 24 GiB cards the NVFP4 arms are served from.
// The invariant below is the one the leak violated: a release through the
// weight's own resident state must actually free BOTH buffers.
TEST_CASE("fp4 resident: releasing the resident frees both device buffers") {
  ForcedResidencyArm arm;
  ScopedEnvVar adopt_default("VT_ADOPT_DEVICE_BYTES", "1");
  // A discrete (non-host-addressable) device: no adoption, so before the fix
  // `packed.d_dev` and the type-specific handle were two references to one
  // block. This is the arm every consumer Blackwell card takes.
  FakeBackend b(/*host_addressable=*/false);
  vt::Queue q = b.CreateQueue();
  const Fp4Dims dims = MakeFp4Dims();

  ObservableMapping mp;
  ObservableMapping ms;
  vllm::Nvfp4Weight w = BorrowedFp4Weight(dims, mp, ms);
  vllm::dense_attn::Dev d{b, q};
  const vllm::dense_nvfp4::Nvfp4Dev dev = vllm::dense_nvfp4::ResidentNvfp4(d, w);
  REQUIRE(dev.packed.data != nullptr);
  REQUIRE(dev.scale.data != nullptr);
  CHECK(b.frees == 0);

  // The one release a repack builder performs once the repack has landed. RED
  // before the single-owner change: the builders reset the type-specific pair
  // and the surviving `packed.d_dev`/`scale.d_dev` alias kept both blocks, so
  // this read 0 rather than 2.
  w.ReleaseResident();

  CHECK(b.frees == 2);
}

TEST_CASE("fp4 resident: a NON-host-addressable device still counts and still releases") {
  ForcedResidencyArm arm;
  ScopedEnvVar adopt_default("VT_ADOPT_DEVICE_BYTES", "1");
  FakeBackend b(/*host_addressable=*/false);
  vt::Queue q = b.CreateQueue();
  const Fp4Dims dims = MakeFp4Dims();

  ObservableMapping mp;
  ObservableMapping ms;
  vllm::Nvfp4Weight w = BorrowedFp4Weight(dims, mp, ms);

  const vllm::load_stats::Counters before = vllm::load_stats::Snapshot();
  vllm::dense_attn::Dev d{b, q};
  vllm::dense_nvfp4::ResidentNvfp4(d, w);
  const vllm::load_stats::Counters after = vllm::load_stats::Snapshot();

  CHECK(after.device_upload_bytes - before.device_upload_bytes ==
        dims.packed_bytes + dims.scale_bytes);

  // No adoption: the borrows stay valid, re-faultable views of their mappings...
  CHECK(static_cast<const void*>(w.packed.bytes.data()) ==
        static_cast<const void*>(mp.addr));
  CHECK(static_cast<const void*>(w.scale.bytes.data()) ==
        static_cast<const void*>(ms.addr));
  CHECK(w.packed.mmap_src == static_cast<const void*>(mp.addr));
  CHECK_FALSE(mp.dropped);
  CHECK_FALSE(ms.dropped);
  // ... but the consumed source pages were dropped all the same.
  CHECK(mp.addr[0] == 0);
  CHECK(ms.addr[0] == 0);
}

TEST_CASE("fp4 resident: a host-addressable device adopts an OWNED fp4 mirror too") {
  // THE PREVIOUSLY UNEXERCISED REGIME. Every fp4 arm that does NOT borrow — a
  // dequant/repack producer, a checkpoint the borrow helper failed closed on,
  // any non-safetensors source — hands ResidentNvfp4 an OWNED `packed`/`scale`
  // with a null `mmap_src`. The round-2 AdoptDeviceBytesAsHost call then takes
  // the GENERAL branch, which madvises and frees the host fp4 mirror and
  // re-points `bytes` at the device buffer. That is new behavior on Vulkan and
  // no gate covered it; the assertion that matters is that every `.bytes` reader
  // still sees the weight's VALUES afterwards, since the CPU dequant fallback is
  // one of them.
  ForcedResidencyArm arm;
  ScopedEnvVar adopt_default("VT_ADOPT_DEVICE_BYTES", "1");
  FakeBackend b(/*host_addressable=*/true);
  vt::Queue q = b.CreateQueue();
  const Fp4Dims dims = MakeFp4Dims();

  vllm::Nvfp4Weight w;
  w.n = dims.n;
  w.k = dims.k;
  w.group_size = 16;
  w.packed = OwnedPatternWeight(dims.packed_bytes);
  w.scale = OwnedPatternWeight(dims.scale_bytes);
  REQUIRE_FALSE(w.packed.bytes.borrowed());
  REQUIRE(w.packed.mmap_src == nullptr);

  const vllm::load_stats::Counters before = vllm::load_stats::Snapshot();
  vllm::dense_attn::Dev d{b, q};
  vllm::dense_nvfp4::ResidentNvfp4(d, w);
  const vllm::load_stats::Counters after = vllm::load_stats::Snapshot();

  CHECK(after.device_upload_bytes - before.device_upload_bytes ==
        dims.packed_bytes + dims.scale_bytes);
  REQUIRE(w.packed.d_dev != nullptr);
  CHECK(w.scale.d_dev != nullptr);

  // Adopted: ONE copy, and it is the device one.
  CHECK(w.packed.bytes.borrowed());
  CHECK(w.scale.bytes.borrowed());
  CHECK(static_cast<const void*>(w.packed.bytes.data()) == w.packed.d_dev.get());
  CHECK(static_cast<const void*>(w.scale.bytes.data()) == w.scale.d_dev.get());
  CHECK(w.packed.bytes.size() == dims.packed_bytes);
  CHECK(w.scale.bytes.size() == dims.scale_bytes);
  CHECK_FALSE(w.packed.host_released);
  CHECK_FALSE(w.Empty());

  // THE VALUES SURVIVED the host mirror being madvise'd away and freed.
  CHECK(AllBytesMatchPattern(w.packed.bytes.data(), dims.packed_bytes));
  CHECK(AllBytesMatchPattern(w.scale.bytes.data(), dims.scale_bytes));
}

// --- The general adopt branch's madvise: the RSS half, made observable --------
//
// The case above proves the general branch keeps the weight's VALUES. It does not
// prove the branch reclaims anything, and reclaiming is the entire thesis of this
// row. `AdoptDeviceBytesAsHost`'s general branch MADV_DONTNEEDs the host mirror's
// interior whole pages and only THEN frees the vector, for a reason its own
// comment states: glibc raises its dynamic mmap threshold as large blocks are
// freed, so `free()` alone leaves weight buffers on the arena free-list with every
// page still RESIDENT.
//
// NOTHING BIT WHEN THAT MADVISE WAS DELETED, and a fresh reviewer showed it: every
// assertion in this file reads VALUES, and the values live in the surviving device
// copy either way. A residency claim needs a residency observation, which is
// `mincore()` -- it answers "is this page in core" for an address RANGE, and being
// a syscall on the range rather than a dereference it stays legal after the block
// has gone back to the allocator.
//
// TWO ALLOCATOR FACTS MUST BE PINNED or the observation is not about the madvise:
//
//   * `M_MMAP_MAX=0` keeps the mirror in the sbrk arena instead of a private mmap.
//     `free()` on an mmap'd block MUNMAPS it, and then the pages are gone with or
//     without the madvise -- the test would pass on the mutant.
//   * `M_TRIM_THRESHOLD` off plus a guard allocation made AFTER the mirror keep the
//     freed block off the heap top, so glibc cannot sbrk-trim it away for exactly
//     the same reason.
//
// The in-run negative control is `before`: the same pages, counted the same way,
// must show FULLY resident immediately beforehand.
//
// `#if __GLIBC__` PROVES THE HEADERS, NOT THE ALLOCATOR. An LD_PRELOADed
// jemalloc/tcmalloc interposes malloc/free, leaves the mallopt calls above inert,
// and may purge or munmap freed pages BY ITSELF -- under which the deleted-madvise
// mutant would pass. So the case also carries a RUNTIME guard (`the free() control`
// below): it allocates a block the same way, frees it, and requires the pages to
// stay mapped AND resident. That is exactly the allocator property the observation
// rests on, and if it does not hold the case goes red and says which half failed
// instead of silently vouching for a mutant.
//
// A PROCESS-LIFETIME SIDE EFFECT REMAINS, recorded rather than fixed. Every
// `mallopt` that touches a threshold sets glibc's `no_dyn_threshold`, which no
// destructor can clear: the thresholds stop adapting for the rest of the process,
// pinned at the documented defaults the destructor restores. It is NOT specific to
// `M_TRIM_THRESHOLD`, so dropping that call would not remove the coupling --
// MEASURED on glibc 2.39, probing whether freeing a 1 MiB mmap'd block still raises
// `mmap_threshold` (`mallinfo2().hblks` on the next 1 MiB allocation): no mallopt
// -> threshold LIVE; `M_TRIM_THRESHOLD` set-then-restored -> DISABLED;
// `M_MMAP_MAX` set-then-restored -> DISABLED. `M_MMAP_MAX=0` is load-bearing here
// (without it `free()` munmaps the mirror and the mutant passes), so the flag is
// unavoidable for this observation. No cross-case effect was observed across 30
// randomized-order runs, and only this case reads residency at all.
#if defined(__linux__) && defined(__GLIBC__)
namespace {

class ScopedArenaOnlyMalloc {
 public:
  ScopedArenaOnlyMalloc() {
    ::mallopt(M_MMAP_MAX, 0);
    ::mallopt(M_TRIM_THRESHOLD, -1);
  }
  // Restores glibc's documented defaults. It cannot restore `no_dyn_threshold`,
  // which either call above already set for the process; see the note above.
  ~ScopedArenaOnlyMalloc() {
    ::mallopt(M_MMAP_MAX, 65536);             // glibc DEFAULT_MMAP_MAX
    ::mallopt(M_TRIM_THRESHOLD, 128 * 1024);  // glibc DEFAULT_TRIM_THRESHOLD
  }
  ScopedArenaOnlyMalloc(const ScopedArenaOnlyMalloc&) = delete;
  ScopedArenaOnlyMalloc& operator=(const ScopedArenaOnlyMalloc&) = delete;
};

struct PageResidency {
  int total = -1;     // interior whole pages in the range
  int resident = -1;  // how many of them mincore() reports in core
};

// Residency of the interior whole pages of [begin, begin+nb) -- the exact range
// the adopt branch madvises. `scratch` is the caller's, so counting allocates
// NOTHING: an allocation between the free and the count could be served out of the
// freed block and re-fault the very pages being measured.
PageResidency InteriorResidency(const uint8_t* begin, size_t nb,
                                std::vector<unsigned char>& scratch) {
  const auto ps = static_cast<uintptr_t>(PageSize());
  const auto b = reinterpret_cast<uintptr_t>(begin);
  const uintptr_t page_begin = (b + ps - 1) & ~(ps - 1);
  const uintptr_t page_end = (b + nb) & ~(ps - 1);
  PageResidency r;
  if (page_end <= page_begin) return r;
  const size_t n = (page_end - page_begin) / ps;
  if (scratch.size() < n) return r;
  if (::mincore(reinterpret_cast<void*>(page_begin),
                static_cast<size_t>(page_end - page_begin), scratch.data()) != 0) {
    return r;
  }
  r.total = static_cast<int>(n);
  r.resident = 0;
  for (size_t i = 0; i < n; ++i) r.resident += (scratch[i] & 1);
  return r;
}

}  // namespace

TEST_CASE("adopt: the general branch DROPS the host mirror's resident pages") {
  ScopedArenaOnlyMalloc arena;
  ForcedResidencyArm arm;
  ScopedEnvVar adopt_default("VT_ADOPT_DEVICE_BYTES", "1");
  FakeBackend b(/*host_addressable=*/true);
  vt::Queue q = b.CreateQueue();

  const size_t nb = 32 * PageSize();
  std::vector<unsigned char> scratch(nb / PageSize() + 1);

  // THE RUNTIME ALLOCATOR GUARD. `#if __GLIBC__` compiled this case in because the
  // HEADERS are glibc's; it does not prove glibc's allocator is the one running. An
  // interposed allocator makes the mallopt calls above inert and may drop freed
  // pages by itself, which would let the deleted-madvise mutant pass. Prove in-run
  // that `free()` ALONE keeps the pages mapped and resident -- the exact property
  // the observation below rests on -- using a block allocated and freed the same way.
  {
    std::vector<uint8_t> control(nb, 0x3C);
    const uint8_t* cp = control.data();
    std::vector<uint8_t> control_guard(nb, 0x3C);  // keeps `control` off the heap top
    const PageResidency live = InteriorResidency(cp, nb, scratch);
    std::vector<uint8_t>().swap(control);          // free, nothing else
    const PageResidency freed = InteriorResidency(cp, nb, scratch);
    REQUIRE(live.total > 0);
    REQUIRE(live.resident == live.total);
    // Still MAPPED: mincore would fail with ENOMEM (total stays -1) if free() had
    // munmapped the block, as it does for an mmap'd chunk or under jemalloc.
    REQUIRE(freed.total == live.total);
    // Still RESIDENT: an allocator that purges on free would make the madvise
    // below unobservable, and this case's verdict meaningless.
    REQUIRE(freed.resident == freed.total);
    CHECK(control_guard[0] == 0x3C);
  }

  // The host mirror: an OWNED buffer, every page touched by the pattern fill.
  vllm::OwnedTensor w = OwnedPatternWeight(nb);
  // Allocated AFTER it, so freeing the mirror cannot consolidate into the heap top.
  std::vector<uint8_t> guard(nb, 0x5A);
  REQUIRE_FALSE(w.bytes.borrowed());
  REQUIRE(w.mmap_src == nullptr);

  // Upload it, exactly as ResidentWeight / ResidentNvfp4 leave a weight.
  void* p = b.Alloc(nb);
  b.Copy(q, p, w.bytes.data(), nb);
  vt::Backend* bk = &b;
  w.d_dev = std::shared_ptr<void>(p, [bk](void* x) { bk->Free(x); });

  const uint8_t* mirror = w.bytes.data();
  const PageResidency before = InteriorResidency(mirror, nb, scratch);

  vllm::AdoptDeviceBytesAsHost(b, w);

  const PageResidency after = InteriorResidency(mirror, nb, scratch);

  // The control: every interior page of the mirror was in core going in.
  REQUIRE(before.total > 0);
  REQUIRE(before.resident == before.total);
  // THE RECLAIM. Deleting the madvise leaves this at `before.total`: free() hands
  // the block back to the arena with its pages still resident, and the RSS this
  // row exists to return is never returned.
  REQUIRE(after.total == before.total);
  CHECK(after.resident == 0);

  // ... and the weight is still readable, out of the surviving device copy.
  CHECK(w.bytes.borrowed());
  CHECK(static_cast<const void*>(w.bytes.data()) == w.d_dev.get());
  CHECK(w.bytes.size() == nb);
  CHECK(AllBytesMatchPattern(w.bytes.data(), nb));
  CHECK(guard[0] == 0x5A);
}
#endif  // __linux__ && __GLIBC__
