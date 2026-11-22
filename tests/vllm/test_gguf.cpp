#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "gguf_builder.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"

namespace {

// Byte-builder helpers (LE encoders, Header, GStr, PadTo, TempFile) live in
// gguf_builder.h, shared with the tokenizer GGUF-vocab tests.
using gguf_test::GStr;
using gguf_test::Header;
using gguf_test::PadTo;
using gguf_test::TempFile;
using gguf_test::U32Le;
using gguf_test::U64Le;
using gguf_test::UniqueTempPath;
using gguf_test::Utf8Path;

// One Q8_0 block: f16 scale + 32 int8 quants = 34 bytes.
std::string Q8Block(uint16_t scale_f16_bits) {
  std::string b = U32Le(scale_f16_bits).substr(0, 2);
  for (int i = 0; i < 32; ++i) b.push_back(static_cast<char>(i));
  return b;
}

// A full, valid GGUF file: 4 kvs (string, u32 alignment, string array, u64)
// and 2 tensors — "t_f32" F32 ggml-dims [4,2] at offset 0 and "t_q8" Q8_0
// [32] at offset 32 — with the data section aligned to 32.
std::string BuildValid(uint32_t version = 3) {
  std::string f = Header(version, /*tensors=*/2, /*kvs=*/4);
  f += GStr("general.architecture") + U32Le(8) + GStr("llama");
  f += GStr("general.alignment") + U32Le(4) + U32Le(32);
  f += GStr("tokenizer.ggml.tokens") + U32Le(9) +  // array kv
       U32Le(8) + U64Le(3) + GStr("a") + GStr("bb") + GStr("ccc");
  f += GStr("some.count") + U32Le(10) + U64Le(UINT64_C(1234567890123));
  // Tensor infos: name, u32 n_dims, u64 dims (ggml order), u32 type, u64 off.
  f += GStr("t_f32") + U32Le(2) + U64Le(4) + U64Le(2) + U32Le(0) + U64Le(0);
  f += GStr("t_q8") + U32Le(1) + U64Le(32) + U32Le(8) + U64Le(32);
  PadTo(f, 32);
  const float vals[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  f.append(reinterpret_cast<const char*>(vals), 32);
  f += Q8Block(0x3c00);  // f16 1.0
  return f;
}

class NamedTempFile {
 public:
  NamedTempFile(const std::filesystem::path& name, const std::string& bytes)
      : directory_(UniqueTempPath("vllm_gguf_named_", "")),
        path_(directory_ / name) {
    if (!std::filesystem::create_directory(directory_))
      throw std::runtime_error("failed to create GGUF test directory");
    std::ofstream out(path_, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("failed to write GGUF test file");
  }
  ~NamedTempFile() {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
    std::filesystem::remove(directory_, ignored);
  }
  NamedTempFile(const NamedTempFile&) = delete;
  NamedTempFile& operator=(const NamedTempFile&) = delete;
  std::string utf8_path() const { return Utf8Path(path_); }

 private:
  std::filesystem::path directory_;
  std::filesystem::path path_;
};

void SetEnvironment(const char* name, const char* value) {
#if defined(_WIN32)
  if (::_putenv_s(name, value) != 0)
    throw std::runtime_error("failed to update test environment");
#else
  const int result =
      value[0] == '\0' ? ::unsetenv(name) : ::setenv(name, value, 1);
  if (result != 0)
    throw std::runtime_error("failed to update test environment");
#endif
}

class ScopedEnvironment final {
 public:
  ScopedEnvironment(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name); old != nullptr) old_ = old;
    SetEnvironment(name_.c_str(), value);
  }
  ~ScopedEnvironment() {
    try {
      SetEnvironment(name_.c_str(), old_.has_value() ? old_->c_str() : "");
    } catch (...) {
    }
  }
  ScopedEnvironment(const ScopedEnvironment&) = delete;
  ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;

 private:
  std::string name_;
  std::optional<std::string> old_;
};

}  // namespace

TEST_CASE("gguf: valid v3 file parses kvs and tensors") {
  TempFile f(BuildValid());
  vllm::GgufFile g = vllm::GgufFile::Open(f.path());

  // Scalar and string kvs.
  const vllm::GgufValue* arch = g.FindKv("general.architecture");
  REQUIRE(arch != nullptr);
  CHECK(arch->TypeId() == vllm::kGgufString);
  CHECK(std::get<std::string>(arch->v) == "llama");
  const vllm::GgufValue* align = g.FindKv("general.alignment");
  REQUIRE(align != nullptr);
  CHECK(std::get<uint32_t>(align->v) == 32);
  const vllm::GgufValue* count = g.FindKv("some.count");
  REQUIRE(count != nullptr);
  CHECK(std::get<uint64_t>(count->v) == UINT64_C(1234567890123));

  // String array kv (tokenizer vocab shape, needed by M0.5).
  const vllm::GgufValue* toks = g.FindKv("tokenizer.ggml.tokens");
  REQUIRE(toks != nullptr);
  REQUIRE(toks->TypeId() == vllm::kGgufArray);
  const vllm::GgufArray& arr = std::get<vllm::GgufArray>(toks->v);
  CHECK(arr.elem_type == vllm::kGgufString);
  REQUIRE(arr.elems.size() == 3);
  CHECK(std::get<std::string>(arr.elems[0].v) == "a");
  CHECK(std::get<std::string>(arr.elems[2].v) == "ccc");

  // Absent kv -> nullptr, not a throw.
  CHECK(g.FindKv("no.such.key") == nullptr);

  // Tensors, in file order; ggml dims [4,2] reversed to torch shape [2,4].
  REQUIRE(g.Tensors().size() == 2);
  const vllm::GgufTensorInfo& t0 = g.Tensors()[0];
  CHECK(t0.name == "t_f32");
  CHECK(t0.shape == std::vector<int64_t>({2, 4}));
  CHECK(t0.ggml_type == 0);
  REQUIRE(t0.nbytes == 32);
  float vals[8];
  std::memcpy(vals, t0.data, 32);
  CHECK(vals[0] == 0.0f);
  CHECK(vals[7] == 7.0f);

  const vllm::GgufTensorInfo& t1 = g.Get("t_q8");
  CHECK(t1.shape == std::vector<int64_t>({32}));
  CHECK(t1.ggml_type == 8);
  REQUIRE(t1.nbytes == 34);
  CHECK(t1.data[0] == 0x00);  // f16 1.0 scale, low byte
  CHECK(t1.data[1] == 0x3c);
  CHECK(t1.data[2 + 5] == 5);  // 6th quant
}

TEST_CASE("gguf: v2 file parses identically (same LE layout as v3)") {
  TempFile f(BuildValid(/*version=*/2));
  vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  CHECK(g.Tensors().size() == 2);
  CHECK(g.Get("t_f32").nbytes == 32);
}

TEST_CASE("gguf: general.alignment kv moves the data section start") {
  // One F32 tensor [2] at offset 0 with alignment 64: the header ends well
  // before 64, so honoring the kv is observable in the data the span sees.
  std::string f = Header(3, 1, 1);
  f += GStr("general.alignment") + U32Le(4) + U32Le(64);
  f += GStr("t") + U32Le(1) + U64Le(2) + U32Le(0) + U64Le(0);
  // The kv must be observable: default alignment 32 would place the data
  // section at a different offset than the declared 64.
  auto align_up = [](size_t v, size_t a) { return (v + a - 1) / a * a; };
  REQUIRE(align_up(f.size(), 32) != align_up(f.size(), 64));
  PadTo(f, 64);
  const float vals[2] = {5.0f, 6.0f};
  f.append(reinterpret_cast<const char*>(vals), 8);
  TempFile tf(f);
  vllm::GgufFile g = vllm::GgufFile::Open(tf.path());
  float got[2];
  std::memcpy(got, g.Get("t").data, 8);
  CHECK(got[0] == 5.0f);
  CHECK(got[1] == 6.0f);
}

TEST_CASE("gguf: move semantics keep the mapping alive") {
  TempFile f(BuildValid());
  vllm::GgufFile a = vllm::GgufFile::Open(f.path());
  vllm::GgufFile b = std::move(a);
  CHECK(b.Get("t_f32").nbytes == 32);
  vllm::GgufFile c = vllm::GgufFile::Open(f.path());
  c = std::move(b);
  CHECK(c.Get("t_q8").nbytes == 34);
  float first;
  std::memcpy(&first, c.Get("t_f32").data, 4);
  CHECK(first == 0.0f);
}  // moved-from a and b destroyed here; must not double-munmap

TEST_CASE("gguf: borrowed mapping outlives a moved and destroyed reader") {
  TempFile f(BuildValid());
  vllm::GgufTensorInfo borrowed;
  std::shared_ptr<const vllm::GgufMapping> owner;
  std::weak_ptr<const vllm::GgufMapping> weak;
  {
    vllm::GgufFile source = vllm::GgufFile::Open(f.path());
    vllm::GgufFile moved = std::move(source);
    borrowed = moved.Get("t_q8");
    owner = moved.Mapping();
    weak = owner;
  }

  REQUIRE_FALSE(weak.expired());
  CHECK(borrowed.data[0] == 0x00);
  CHECK(borrowed.data[1] == 0x3c);
  owner.reset();
  CHECK(weak.expired());
}

TEST_CASE("gguf: non-ASCII filesystem path parses without ANSI conversion") {
  NamedTempFile f(std::filesystem::path(U"vllm_gguf_caf\u00e9_\u6a21\u578b.gguf"),
                  BuildValid());
  vllm::GgufFile g = vllm::GgufFile::Open(f.utf8_path());
  CHECK(g.Get("t_q8").data[1] == 0x3c);
}

TEST_CASE("gguf: Get on absent tensor throws with name") {
  TempFile f(BuildValid());
  vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  CHECK_THROWS_WITH_AS(g.Get("nope"), doctest::Contains("nope"),
                       std::runtime_error);
}

TEST_CASE("gguf: missing file throws with path") {
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open("/nonexistent/no.gguf"),
                       doctest::Contains("/nonexistent/no.gguf"),
                       std::runtime_error);
}

TEST_CASE("gguf: bad magic throws") {
  std::string f = BuildValid();
  f.replace(0, 4, "GGML");
  TempFile tf(f);
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open(tf.path()),
                       doctest::Contains("magic"), std::runtime_error);
}

TEST_CASE("gguf: unsupported version throws") {
  std::string f = BuildValid();
  f.replace(4, 4, U32Le(1));
  TempFile tf(f);
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open(tf.path()),
                       doctest::Contains("version"), std::runtime_error);
}

TEST_CASE("gguf: big-endian file is rejected with a clear message") {
  std::string f = BuildValid();
  f.replace(4, 4, std::string("\x00\x00\x00\x03", 4));  // byte-swapped v3
  TempFile tf(f);
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open(tf.path()),
                       doctest::Contains("big-endian"), std::runtime_error);
}

TEST_CASE("gguf: truncated file throws") {
  std::string f = BuildValid();
  TempFile half(f.substr(0, f.size() / 2));
  CHECK_THROWS_AS(vllm::GgufFile::Open(half.path()), std::runtime_error);
  TempFile tiny(std::string("GGUF"));
  CHECK_THROWS_AS(vllm::GgufFile::Open(tiny.path()), std::runtime_error);
  TempFile empty("");
  CHECK_THROWS_AS(vllm::GgufFile::Open(empty.path()), std::runtime_error);
}

TEST_CASE("gguf: huge kv count throws before any allocation") {
  TempFile f(Header(3, 0, UINT64_C(1) << 40));
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open(f.path()),
                       doctest::Contains("kv count"), std::runtime_error);
}

TEST_CASE("gguf: huge tensor count throws before any allocation") {
  TempFile f(Header(3, UINT64_C(1) << 40, 0));
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open(f.path()),
                       doctest::Contains("tensor count"), std::runtime_error);
}

TEST_CASE("gguf: string length beyond file size throws") {
  // kv key claims 2^40 bytes.
  std::string f = Header(3, 0, 1) + U64Le(UINT64_C(1) << 40) + "x";
  TempFile tf(f);
  CHECK_THROWS_AS(vllm::GgufFile::Open(tf.path()), std::runtime_error);
}

TEST_CASE("gguf: array count beyond file size throws") {
  std::string f = Header(3, 0, 1) + GStr("k") + U32Le(9) +  // array kv
                  U32Le(7) + U64Le(UINT64_C(1) << 40);       // bool x 2^40
  TempFile tf(f);
  CHECK_THROWS_AS(vllm::GgufFile::Open(tf.path()), std::runtime_error);
}

TEST_CASE("gguf: array element budget across the file throws") {
  // A bool array declaring 2e6 elements fits in ~2MB of file bytes, so the
  // per-array remaining-bytes check passes; only the file-wide element
  // budget (1e6) rejects it before the ~40x memory amplification.
  constexpr uint64_t kElems = 2000000;
  std::string f = Header(3, 0, 1) + GStr("k") + U32Le(9) +  // array kv
                  U32Le(7) + U64Le(kElems);                 // bool x 2e6
  f += std::string(kElems, '\1');
  TempFile tf(f);
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open(tf.path()),
                       doctest::Contains("array element budget exceeded"),
                       std::runtime_error);
}

TEST_CASE("gguf: empty array with unknown element type throws") {
  std::string f = Header(3, 0, 1) + GStr("k") + U32Le(9) +  // array kv
                  U32Le(999) + U64Le(0);  // elem type 999, count 0
  TempFile tf(f);
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open(tf.path()),
                       doctest::Contains("999"), std::runtime_error);
}

TEST_CASE("gguf: unknown kv value type throws") {
  std::string f = Header(3, 0, 1) + GStr("k") + U32Le(99) + U64Le(0);
  TempFile tf(f);
  CHECK_THROWS_AS(vllm::GgufFile::Open(tf.path()), std::runtime_error);
}

TEST_CASE("gguf: tensor offset beyond data section throws") {
  std::string f = Header(3, 1, 0);
  f += GStr("t") + U32Le(1) + U64Le(2) + U32Le(0) + U64Le(1024);  // off 1024
  PadTo(f, 32);
  f += std::string(8, '\0');  // section is only 8 bytes
  TempFile tf(f);
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open(tf.path()),
                       doctest::Contains("t"), std::runtime_error);
}

TEST_CASE("gguf: tensor nbytes beyond data section throws") {
  std::string f = Header(3, 1, 0);
  f += GStr("t") + U32Le(1) + U64Le(64) + U32Le(0) + U64Le(0);  // needs 256B
  PadTo(f, 32);
  f += std::string(8, '\0');
  TempFile tf(f);
  CHECK_THROWS_AS(vllm::GgufFile::Open(tf.path()), std::runtime_error);
}

TEST_CASE("gguf: numel not divisible by quant block throws") {
  std::string f = Header(3, 1, 0);
  f += GStr("t") + U32Le(1) + U64Le(33) + U32Le(8) + U64Le(0);  // Q8_0 [33]
  PadTo(f, 32);
  f += std::string(64, '\0');
  TempFile tf(f);
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open(tf.path()),
                       doctest::Contains("divisible"), std::runtime_error);
}

TEST_CASE("gguf: unknown ggml type id in a tensor throws") {
  std::string f = Header(3, 1, 0);
  f += GStr("t") + U32Le(1) + U64Le(2) + U32Le(999) + U64Le(0);
  PadTo(f, 32);
  f += std::string(8, '\0');
  TempFile tf(f);
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open(tf.path()),
                       doctest::Contains("999"), std::runtime_error);
}

TEST_CASE("gguf: duplicate tensor names throw") {
  std::string f = Header(3, 2, 0);
  f += GStr("t") + U32Le(1) + U64Le(2) + U32Le(0) + U64Le(0);
  f += GStr("t") + U32Le(1) + U64Le(2) + U32Le(0) + U64Le(32);
  PadTo(f, 32);
  f += std::string(64, '\0');
  TempFile tf(f);
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open(tf.path()),
                       doctest::Contains("duplicate"), std::runtime_error);
}

TEST_CASE("gguf: dims product overflow throws, not wraps") {
  std::string f = Header(3, 1, 0);
  f += GStr("t") + U32Le(3) + U64Le(UINT64_C(1) << 40) +
       U64Le(UINT64_C(1) << 40) + U64Le(UINT64_C(1) << 40) +  // 2^120 elems
       U32Le(0) + U64Le(0);
  PadTo(f, 32);
  f += std::string(8, '\0');
  TempFile tf(f);
  CHECK_THROWS_AS(vllm::GgufFile::Open(tf.path()), std::runtime_error);
}

TEST_CASE("gguf: misaligned tensor offset throws") {
  std::string f = Header(3, 1, 0);
  f += GStr("t") + U32Le(1) + U64Le(2) + U32Le(0) + U64Le(7);  // 7 % 32 != 0
  PadTo(f, 32);
  f += std::string(64, '\0');
  TempFile tf(f);
  CHECK_THROWS_AS(vllm::GgufFile::Open(tf.path()), std::runtime_error);
}

TEST_CASE("gguf: non-power-of-two general.alignment throws") {
  std::string f = Header(3, 0, 1);
  f += GStr("general.alignment") + U32Le(4) + U32Le(24);
  TempFile tf(f);
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open(tf.path()),
                       doctest::Contains("alignment"), std::runtime_error);
}

TEST_CASE("gguf: zero-tensor file with kvs only is fine") {
  std::string f = Header(3, 0, 1) + GStr("k") + U32Le(5) + U32Le(0xfffffff6);
  TempFile tf(f);
  vllm::GgufFile g = vllm::GgufFile::Open(tf.path());
  CHECK(g.Tensors().empty());
  CHECK(std::get<int32_t>(g.FindKv("k")->v) == -10);
}

namespace {
// One shard of a synthetic split GGUF: split.no/count KVs + a single F32[4]
// tensor whose first element is `v0`. Shard 00001 also carries the "real"
// metadata (general.architecture), the others do not (mirrors llama.cpp
// gguf-split, whose non-first shards hold only split.* KVs).
std::string SplitShard(const std::string& tname, float v0, uint32_t split_no,
                       uint32_t split_count, bool full_kv) {
  gguf_test::GgufModelBuilder b;
  if (full_kv) b.AddKv(gguf_test::StrKv("general.architecture", "deepseek4"));
  b.AddKv(gguf_test::U32Kv("split.no", split_no));
  b.AddKv(gguf_test::U32Kv("split.count", split_count));
  std::string data(16, '\0');
  for (int i = 0; i < 4; ++i) {
    const float f = v0 + static_cast<float>(i);
    std::memcpy(&data[i * 4], &f, 4);
  }
  b.AddTensor(tname, {4}, /*F32=*/0, data);
  return b.Build();
}
// Writes two shard files under a unique split-named base and removes them on
// destruction. `Open(shard1())` must transparently merge both.
struct SplitFiles {
  explicit SplitFiles(const std::string& tag) {
    const std::filesystem::path base =
        UniqueTempPath("vllm_gguf_split_" + tag + "_", "");
    s1_ = base;
    s1_ += "-00001-of-00002.gguf";
    s2_ = base;
    s2_ += "-00002-of-00002.gguf";
  }
  void Write(const std::string& a, const std::string& b) const {
    std::ofstream(s1_, std::ios::binary)
        .write(a.data(), static_cast<std::streamsize>(a.size()));
    std::ofstream(s2_, std::ios::binary)
        .write(b.data(), static_cast<std::streamsize>(b.size()));
  }
  std::string shard1() const { return Utf8Path(s1_); }
  ~SplitFiles() {
    std::error_code ignored;
    std::filesystem::remove(s1_, ignored);
    std::filesystem::remove(s2_, ignored);
  }

 private:
  std::filesystem::path s1_;
  std::filesystem::path s2_;
};
}  // namespace

TEST_CASE("gguf: split shards are merged into one logical file") {
  SplitFiles f("merge");
  f.Write(SplitShard("t_shard0", 10.0f, 0, 2, /*full_kv=*/true),
          SplitShard("t_shard1", 20.0f, 1, 2, /*full_kv=*/false));

  vllm::GgufFile g = vllm::GgufFile::Open(f.shard1());
  // Both shards' tensors are visible through the merged file.
  REQUIRE(g.Tensors().size() == 2);
  // KV metadata comes from shard 00001.
  const vllm::GgufValue* arch = g.FindKv("general.architecture");
  REQUIRE(arch != nullptr);
  CHECK(std::get<std::string>(arch->v) == "deepseek4");
  // The shard-2 tensor's bytes stay readable — its mapping is kept alive by the
  // primary shard's mapping (the keep-quant borrow contract), and OwnsSpan sees
  // spans in either shard.
  const vllm::GgufTensorInfo& t1 = g.Get("t_shard1");
  float v0 = 0;
  std::memcpy(&v0, t1.data, 4);
  CHECK(v0 == 20.0f);
  CHECK(g.OwnsSpan(t1.data, t1.nbytes));
  const vllm::GgufTensorInfo& t0 = g.Get("t_shard0");
  CHECK(g.OwnsSpan(t0.data, t0.nbytes));
}

TEST_CASE("gguf: VT_GGUF_NO_SPLIT opens a single shard as-is") {
  SplitFiles f("nosplit");
  f.Write(SplitShard("t_shard0", 1.0f, 0, 2, true),
          SplitShard("t_shard1", 2.0f, 1, 2, false));
  ScopedEnvironment no_split("VT_GGUF_NO_SPLIT", "1");
  vllm::GgufFile g = vllm::GgufFile::Open(f.shard1());
  CHECK(g.Tensors().size() == 1);  // only shard 00001's tensor
}

TEST_CASE("gguf: split.count disagreeing with the -of- filename throws") {
  SplitFiles f("mismatch");
  // Filename says -of-00002, but the header KV claims 3 shards.
  f.Write(SplitShard("t_shard0", 1.0f, 0, /*count=*/3, true),
          SplitShard("t_shard1", 2.0f, 1, /*count=*/3, false));
  CHECK_THROWS_AS(vllm::GgufFile::Open(f.shard1()), std::runtime_error);
}

TEST_CASE("ggml traits: standard table values") {
  CHECK(vllm::GgmlTraits(0).block_elems == 1);   // F32
  CHECK(vllm::GgmlTraits(0).block_bytes == 4);
  CHECK(vllm::GgmlTraits(0).name == std::string("F32"));
  CHECK(vllm::GgmlTraits(1).block_bytes == 2);   // F16
  CHECK(vllm::GgmlTraits(2).block_elems == 32);  // Q4_0
  CHECK(vllm::GgmlTraits(2).block_bytes == 18);
  CHECK(vllm::GgmlTraits(8).block_bytes == 34);  // Q8_0
  CHECK(vllm::GgmlTraits(10).block_bytes == 84);   // Q2_K
  CHECK(vllm::GgmlTraits(11).block_bytes == 110);  // Q3_K
  CHECK(vllm::GgmlTraits(12).block_elems == 256);  // Q4_K
  CHECK(vllm::GgmlTraits(12).block_bytes == 144);
  CHECK(vllm::GgmlTraits(13).block_bytes == 176);  // Q5_K
  CHECK(vllm::GgmlTraits(14).block_bytes == 210);  // Q6_K
  CHECK(vllm::GgmlTraits(22).block_elems == 256);  // IQ2_S (APEX Mini)
  CHECK(vllm::GgmlTraits(22).block_bytes == 82);
  CHECK(vllm::GgmlTraits(23).block_elems == 256);  // IQ4_XS (APEX Quality)
  CHECK(vllm::GgmlTraits(23).block_bytes == 136);
  CHECK(vllm::GgmlTraits(24).block_bytes == 1);    // I8
  CHECK(vllm::GgmlTraits(25).block_bytes == 2);    // I16
  CHECK(vllm::GgmlTraits(26).block_bytes == 4);    // I32
  CHECK(vllm::GgmlTraits(27).block_bytes == 8);    // I64
  CHECK(vllm::GgmlTraits(28).block_bytes == 8);    // F64
  CHECK(vllm::GgmlTraits(30).block_bytes == 2);    // BF16
  CHECK(vllm::GgmlTraits(30).name == std::string("BF16"));
}

TEST_CASE("ggml traits: killgate fork extension ids (NVFP4, Q1_0, MXFP4)") {
  // Ids/geometry from mudler's killgate llama.cpp fork
  // (ggml/include/ggml.h:429-431, ggml/src/ggml-common.h). See
  // .agents/specs/gguf-nvfp4-notes.md.
  CHECK(vllm::GgmlTraits(39).block_elems == 32);  // MXFP4: 1 E8M0 + 16 qs
  CHECK(vllm::GgmlTraits(39).block_bytes == 17);
  CHECK(vllm::GgmlTraits(39).name == std::string("MXFP4"));
  CHECK(vllm::GgmlTraits(40).block_elems == 64);  // NVFP4: 4 UE4M3 + 32 qs
  CHECK(vllm::GgmlTraits(40).block_bytes == 36);
  CHECK(vllm::GgmlTraits(40).name == std::string("NVFP4"));
  CHECK(vllm::GgmlTraits(41).block_elems == 128);  // Q1_0: f16 d + 16 qs
  CHECK(vllm::GgmlTraits(41).block_bytes == 18);
  CHECK(vllm::GgmlTraits(41).name == std::string("Q1_0"));
}

TEST_CASE("gguf: synthetic NVFP4 tensor (fork type id 40) nbytes math") {
  // One tensor, ggml dims [64, 3] (192 elements = 3 NVFP4 blocks), so
  // nbytes must be 3 * 36 = 108. Layout per killgate fork block_nvfp4:
  // 4 UE4M3 sub-block scales then 32 bytes of packed e2m1 nibbles.
  std::string f = Header(3, /*tensors=*/1, /*kvs=*/1);
  f += GStr("general.alignment") + U32Le(4) + U32Le(32);
  f += GStr("w_nvfp4") + U32Le(2) + U64Le(64) + U64Le(3) + U32Le(40) +
       U64Le(0);
  PadTo(f, 32);
  std::string block;
  for (int i = 0; i < 4; ++i) block.push_back(static_cast<char>(0x40 + i));
  for (int i = 0; i < 32; ++i) block.push_back(static_cast<char>(i));
  for (int b = 0; b < 3; ++b) f += block;
  TempFile tf(f);

  vllm::GgufFile g = vllm::GgufFile::Open(tf.path());
  const vllm::GgufTensorInfo& t = g.Get("w_nvfp4");
  CHECK(t.ggml_type == 40);
  CHECK(t.shape == std::vector<int64_t>({3, 64}));
  REQUIRE(t.nbytes == 108);  // 192 / 64 * 36
  CHECK(t.data[0] == 0x40);         // first sub-block scale
  CHECK(t.data[4] == 0x00);         // first packed e2m1 byte
  CHECK(t.data[36 + 3] == 0x43);    // block 1, 4th scale
  CHECK(t.data[2 * 36 + 4 + 31] == 31);  // last qs byte of block 2

  // A 96-element NVFP4 tensor is not divisible by the 64-element block.
  std::string bad = Header(3, 1, 0);
  bad += GStr("w_bad") + U32Le(1) + U64Le(96) + U32Le(40) + U64Le(0);
  PadTo(bad, 32);
  bad += std::string(54, '\0');
  TempFile tbad(bad);
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open(tbad.path()),
                       doctest::Contains("NVFP4"), std::runtime_error);
}

TEST_CASE("ggml traits: unknown type id throws") {
  CHECK_THROWS_WITH_AS(vllm::GgmlTraits(999), doctest::Contains("999"),
                       std::runtime_error);
  // Real ggml ids we have not tabulated (e.g. 9 = Q8_1, removed ids 4/5)
  // must also throw rather than return garbage size math. Ids 3 (Q4_1) and
  // 7 (Q5_1) are NOW tabulated for the inkling2_loop APEX GGUF and are
  // asserted in test_inkling2_loop_gguf.cpp.
  CHECK_THROWS_WITH_AS(vllm::GgmlTraits(9), doctest::Contains("9"),
                       std::runtime_error);
  CHECK(vllm::GgmlTraits(3).block_elems == 32);
  CHECK(vllm::GgmlTraits(3).block_bytes == 20);
  CHECK(vllm::GgmlTraits(7).block_elems == 32);
  CHECK(vllm::GgmlTraits(7).block_bytes == 21);
}

#if !defined(_WIN32)
TEST_CASE("gguf: SourceOfSpan names the OWNING shard's fd and that shard's own offset") {
  // ENG-EXPERT-STREAM F4 (#912). `SourceOfSpan` had ZERO coverage, and two
  // mutations of it survived the whole gate: deleting the sibling walk, so every
  // span resolves to shard 0's descriptor, and forcing `out.offset = 0`.
  //
  // Both are the "wrong shard at a plausible offset" failure. Nothing downstream
  // can tell: a descriptor is a descriptor and an offset is in range, so the
  // pread succeeds, returns the requested number of bytes, and the model
  // multiplies whatever happened to live there. This is the one place in the
  // streaming chain that decides WHICH FILE a weight is read out of.
  //
  // A split GGUF is the whole reason the function is not `ptr - base`: each
  // shard is its own mapping with its own zero, and a merged file hands out
  // spans from any of them.
  SplitFiles f("spansrc");
  f.Write(SplitShard("t_shard0", 10.0f, 0, 2, /*full_kv=*/true),
          SplitShard("t_shard1", 20.0f, 1, 2, /*full_kv=*/false));

  vllm::GgufFile g = vllm::GgufFile::Open(f.shard1());
  REQUIRE(g.Tensors().size() == 2);
  const vllm::GgufTensorInfo& t0 = g.Get("t_shard0");
  const vllm::GgufTensorInfo& t1 = g.Get("t_shard1");

  const vllm::GgufFile::SpanSource s0 = g.SourceOfSpan(t0.data, t0.nbytes);
  const vllm::GgufFile::SpanSource s1 = g.SourceOfSpan(t1.data, t1.nbytes);
  REQUIRE(s0.fd >= 0);
  REQUIRE(s1.fd >= 0);

  // THE SHARD-DISCRIMINATING ASSERTION. Deleting the sibling walk makes every
  // span resolve through the primary mapping, and these two descriptors become
  // the same one.
  CHECK(s0.fd != s1.fd);

  // The offset is relative to the OWNING shard's own zero, so shard 1's tensor
  // sits at a small offset in a small file rather than at an address computed
  // against shard 0.
  const auto shard_size = [](int fd) {
    struct ::stat st {};
    REQUIRE(::fstat(fd, &st) == 0);
    return static_cast<size_t>(st.st_size);
  };
  CHECK(s1.offset + t1.nbytes <= shard_size(s1.fd));
  CHECK(s0.offset + t0.nbytes <= shard_size(s0.fd));
  // Neither is zero here, so a mutation that forces `offset = 0` is not
  // accidentally right.
  CHECK(s0.offset > 0);
  CHECK(s1.offset > 0);

  // AND THE END-TO-END PROPERTY, which is the one that actually matters:
  // pread(fd, offset) must reproduce the MAPPED bytes byte for byte. This is the
  // substitution the expert streamer makes, so if it does not hold the streamer
  // silently feeds the GEMM another tensor.
  const auto pread_span = [](int fd, size_t off, size_t n) {
    std::vector<uint8_t> buf(n);
    size_t done = 0;
    while (done < n) {
      const ssize_t r = ::pread(fd, buf.data() + done, n - done,
                                static_cast<off_t>(off + done));
      REQUIRE(r > 0);
      done += static_cast<size_t>(r);
    }
    return buf;
  };
  const std::vector<uint8_t> got1 = pread_span(s1.fd, s1.offset, t1.nbytes);
  CHECK(std::memcmp(got1.data(), t1.data, t1.nbytes) == 0);
  const std::vector<uint8_t> got0 = pread_span(s0.fd, s0.offset, t0.nbytes);
  CHECK(std::memcmp(got0.data(), t0.data, t0.nbytes) == 0);

  // The two shards hold DIFFERENT bytes, so the memcmp above can actually fail:
  // a test whose two candidates are identical proves nothing about which one was
  // chosen.
  REQUIRE(t0.nbytes == t1.nbytes);
  CHECK(std::memcmp(t0.data, t1.data, t0.nbytes) != 0);

  // A span this file does not own is reported as "no descriptor", which the
  // caller must read as "go through the mapping" rather than as an error.
  const std::vector<uint8_t> foreign(64, 0x7E);
  const vllm::GgufFile::SpanSource none = g.SourceOfSpan(foreign.data(), foreign.size());
  CHECK(none.fd == -1);
}
#endif  // !_WIN32
