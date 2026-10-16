// Full real K/N on GPU; independent CPU evaluation of the first, middle and
// final 128-column Hadamard blocks bounds host work, including the 6-bit head.
#include "xpu_test_helpers.h"
#include "exl3_fixture.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vt/unaligned.h"
#include "vt/xpu.h"
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <tuple>

namespace {
using vt::DType;
using xpu_test::Buffer;
using xpu_test::Queue;
using xpu_test::SameBytes;
}

TEST_CASE("XPU EXL3 checkpoint: all real bit/K/N families, F16 and F32 to BF16") {
  const char* env = std::getenv("VT_B70_MODEL_DIR");
  if (!env) {
    std::cerr << "SKIP: set VT_B70_MODEL_DIR to the pinned local EXL3 checkpoint; no weight downloads.\n";
    std::exit(77);
  }
  const std::filesystem::path root(env);
  const auto index = vllm::LoadSafetensorsIndex((root / "model.safetensors.index.json").string());
  std::map<std::string, vllm::SafetensorsFile> files;
  auto get = [&](const std::string& name) -> const vllm::StTensor& {
    const auto& shard = index.at(name);
    if (!files.contains(shard)) files.emplace(shard, vllm::SafetensorsFile::Open((root / shard).string()));
    return files.at(shard).Get(name);
  };
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  std::set<std::tuple<int, int64_t, int64_t>> seen;
  for (const auto& [name, shard] : index) {
    (void)shard;
    if (!name.ends_with(".trellis") || name.starts_with("mtp.")) continue;
    const auto& packed = get(name);
    REQUIRE(packed.dtype == "I16"); REQUIRE(packed.shape.size() == 3);
    const int bits = packed.shape[2] / 16;
    const int64_t k = packed.shape[0] * 16, n = packed.shape[1] * 16;
    if (!seen.emplace(bits, k, n).second) continue;
    const auto prefix = name.substr(0, name.size() - 8);
    CAPTURE(prefix);
    CAPTURE(bits);
    CAPTURE(k);
    CAPTURE(n);
    const auto& suh = get(prefix + ".suh"); const auto& svh = get(prefix + ".svh");
    const auto& marker = get(prefix + ".mul1");
    REQUIRE(vt::LoadUnaligned<uint32_t>(marker.data) == 0x83DCD12Du);
    REQUIRE(suh.dtype == "F16"); REQUIRE(svh.dtype == "F16");
    Buffer b(gpu.q, DType::kI8, {k / 16, n / 16, 32 * bits});
    Buffer u(gpu.q, DType::kF16, {k}), v(gpu.q, DType::kF16, {n});
    REQUIRE(b.bytes == packed.nbytes);
    b.upload(packed.data); u.upload(suh.data); v.upload(svh.data);
    std::vector<uint16_t> input(k);
    exl3_test::Rng rng;
    for (auto& value : input) value = vt::F32ToF16(rng.next(0.2f));
    Buffer a(gpu.q, DType::kF16, {1, k}), ah(gpu.q, DType::kF16, {1, k});
    a.upload(input.data());
    for (auto dtype : {DType::kF16, DType::kF32}) {
      CAPTURE(dtype);
      Buffer out(gpu.q, dtype, {1, n});
      vt::Exl3Gemm(gpu.q, out.tensor, a.tensor, b.tensor, u.tensor, v.tensor, ah.tensor,
                   vt::Exl3GemmArgs{bits, 2});
      const auto actual = out.download();
      std::vector<unsigned char> actual_bf16;
      if (dtype == DType::kF32) {
        Buffer bf(gpu.q, DType::kBF16, {1, n});
        vt::CastBf16(gpu.q, bf.tensor, out.tensor); actual_bf16 = bf.download();
      }
      for (int64_t column : {int64_t{0}, (n / 256) * 128, n - 128}) {
        CAPTURE(column);
        std::vector<unsigned char> panel(k / 16 * 8 * 32 * bits);
        for (int64_t tile = 0; tile < k / 16; ++tile)
          std::memcpy(panel.data() + tile * 8 * 32 * bits,
                      packed.data + (tile * (n / 16) + column / 16) * 32 * bits, 8 * 32 * bits);
        Buffer ca(cpu.q, DType::kF16, {1, k}), ch(cpu.q, DType::kF16, {1, k});
        Buffer cb(cpu.q, DType::kI8, {k / 16, 8, 32 * bits});
        Buffer cu(cpu.q, DType::kF16, {k}), cv(cpu.q, DType::kF16, {128});
        Buffer co(cpu.q, dtype, {1, 128});
        ca.upload(input.data()); cb.upload(panel.data()); cu.upload(suh.data); cv.upload(svh.data + column * 2);
        vt::Exl3Gemm(cpu.q, co.tensor, ca.tensor, cb.tensor, cu.tensor, cv.tensor, ch.tensor,
                     vt::Exl3GemmArgs{bits, 2});
        const size_t begin = column * vt::SizeOf(dtype);
        SameBytes(std::vector<unsigned char>(actual.begin() + begin,
                    actual.begin() + begin + co.bytes), co.download());
        if (dtype == DType::kF32) {
          Buffer bf(cpu.q, DType::kBF16, {1, 128});
          vt::CastBf16(cpu.q, bf.tensor, co.tensor);
          SameBytes(std::vector<unsigned char>(actual_bf16.begin() + column * 2,
                      actual_bf16.begin() + (column + 128) * 2), bf.download());
        }
      }
      std::cout << "PASS " << prefix << " bits=" << bits << " K=" << k << " N=" << n
                << " output=" << vt::Name(dtype) << " (three CPU-checked column blocks)\n";
    }
  }
  REQUIRE(seen.size() == 11);
  CHECK(vt::GetReferenceTierHits() == 0);
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == vt::xpu::GetMemoryInfo().exl3_workspace_bytes);
}
