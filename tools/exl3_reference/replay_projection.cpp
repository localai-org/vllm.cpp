// Focused S0b replay of real packed fixtures through the native XPU leaves.
// No Torch, model expansion, server, or alternate checkpoint is involved.
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vt/xpu/xpu_common.h"
#include "vt/xpu/xpu_exl3.h"
#include "vt/xpu/xpu_kernels.h"
#include "vt/unaligned.h"
#include <nlohmann/json.hpp>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <vector>

namespace {
using vt::DType;
struct Queue {
  vt::Queue q = vt::CreateQueue({vt::DeviceType::kXPU, 0});
  ~Queue() { vt::DestroyQueue(q); }
};
struct Buffer {
  vt::Queue& q;
  vt::Tensor tensor;
  Buffer(vt::Queue& queue, DType dtype, std::initializer_list<int64_t> shape) : q(queue) {
    tensor = vt::Tensor::Contiguous(nullptr, dtype, q.device, shape);
    tensor.data = vt::Alloc(q.device, tensor.Bytes());
  }
  ~Buffer() { vt::Free(q.device, tensor.data); }
  Buffer(const Buffer&) = delete;
  void upload(const vllm::StTensor& source) {
    VT_CHECK(tensor.Bytes() == source.nbytes, "replay upload byte count mismatch");
    vt::GetBackend(q.device).Copy(q, tensor.data, source.data, source.nbytes);
  }
  std::vector<uint8_t> download() {
    std::vector<uint8_t> raw(tensor.Bytes());
    vt::GetBackend(q.device).Copy(q, raw.data(), tensor.data, raw.size());
    vt::GetBackend(q.device).Synchronize(q);
    return raw;
  }
};
struct Blob {
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> raw;
};
using Blobs = std::map<std::string, Blob>;

void Check(const vllm::StTensor& tensor, const char* dtype, std::vector<int64_t> shape) {
  VT_CHECK(tensor.dtype == dtype && tensor.shape == shape, "replay fixture dtype/shape mismatch");
}
void Save(Blobs& blobs, const std::string& name, Buffer& buffer, const char* dtype) {
  blobs.emplace(name, Blob{dtype,
      {buffer.tensor.shape, buffer.tensor.shape + buffer.tensor.rank}, buffer.download()});
}
void Write(const std::string& path, const Blobs& blobs, const std::string& device) {
  nlohmann::json header = {{"__metadata__", {{"device", device},
      {"route", "direct native XPU leaves, default dispatch; not a full engine build"}}}};
  size_t offset = 0;
  for (const auto& [name, blob] : blobs) {
    header[name] = {{"dtype", blob.dtype}, {"shape", blob.shape},
                    {"data_offsets", {offset, offset + blob.raw.size()}}};
    offset += blob.raw.size();
  }
  std::string json = header.dump();
  json.append((8 - json.size() % 8) % 8, ' ');
  const uint64_t length = json.size();
  int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
  VT_CHECK(fd >= 0, "cannot create exclusive native capture");
  try {
    auto write_all = [&](const void* data, size_t bytes) {
      const auto* p = static_cast<const uint8_t*>(data);
      while (bytes) {
        const auto count = ::write(fd, p, bytes);
        VT_CHECK(count > 0, "native capture write failed");
        p += count; bytes -= count;
      }
    };
    // Safetensors header length is little-endian regardless of host order.
    uint8_t prefix[8];
    for (int i = 0; i < 8; ++i) prefix[i] = (length >> (8 * i)) & 255;
    write_all(prefix, sizeof(prefix)); write_all(json.data(), json.size());
    for (const auto& [name, blob] : blobs) { (void)name; write_all(blob.raw.data(), blob.raw.size()); }
    VT_CHECK(::close(fd) == 0, "native capture close failed"); fd = -1;
  } catch (...) { if (fd >= 0) ::close(fd); throw; }
}
}

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::cerr << "usage: exl3_projection_replay FIXTURE ORACLE OUTPUT [--expect-route-rejection]\n";
    return 2;
  }
  try {
    const bool reject_route = argc == 5 && std::string(argv[4]) == "--expect-route-rejection";
    VT_CHECK(argc == 4 || reject_route, "unknown replay option");
    VT_CHECK(!std::filesystem::exists(argv[3]), "refusing to overwrite native capture");
    auto fixture = vllm::SafetensorsFile::Open(argv[1]);
    auto oracle = vllm::SafetensorsFile::Open(argv[2]);
    const auto& tr = fixture.Get("trellis");
    VT_CHECK(tr.dtype == "I16" && tr.shape.size() == 3, "invalid replay trellis");
    const int bits = tr.shape[2] / 16;
    const int64_t k = tr.shape[0] * 16, n = tr.shape[1] * 16;
    VT_CHECK((bits == 4 || bits == 6) && tr.shape[2] == 16 * bits &&
             k > 0 && n > 0 && k % 128 == 0 && n % 128 == 0, "invalid replay dimensions");
    const auto& u = fixture.Get("suh"); const auto& v = fixture.Get("svh");
    Check(u, "F16", {k}); Check(v, "F16", {n});
    const auto& marker = fixture.Get("mul1"); Check(marker, "I32", {});
    VT_CHECK(vt::LoadUnaligned<uint32_t>(marker.data) == 0x83DCD12Du, "invalid replay mul1 marker");
    Queue owner; auto& q = owner.q;
    vt::RegisterOp(vt::OpId::kCopy, vt::DeviceType::kXPU,
        reinterpret_cast<void*>(static_cast<vt::CopyFn>(&vt::xpu::CopyKernel)));
    const auto device = vt::xpu::DeviceDescription(0);
    VT_CHECK(nlohmann::json::parse(device).at("name").get<std::string>().find("B70") != std::string::npos,
             "replay requires the B70");
    Buffer packed(q, DType::kI8, {k / 16, n / 16, 32 * bits});
    Buffer suh(q, DType::kF16, {k}), svh(q, DType::kF16, {n});
    packed.upload(tr); suh.upload(u); svh.upload(v);
    Blobs blobs;
    if (!reject_route) {
      Buffer decoded(q, DType::kF16, {k, n});
      auto* dst = static_cast<sycl::half*>(decoded.tensor.data);
      const auto* src = static_cast<const unsigned char*>(packed.tensor.data);
      vt::xpu::NativeQueue(q).parallel_for(sycl::range<1>(k * n), [=](sycl::id<1> i) {
        const int64_t row = i[0] / n, col = i[0] % n;
        const auto* tile = src + ((row / 16) * (n / 16) + col / 16) * 32 * bits;
        dst[i[0]] = sycl::half(vt::xpu::exl3::Decode(
            vt::xpu::exl3::Codeword(tile, bits, vt::xpu::exl3::Fragment(row % 16, col % 16)), 2));
      });
      Save(blobs, "decoded_weight_f16", decoded, "F16");
    }
    for (int64_t m : {1, 4}) {
      const std::string key = "m" + std::to_string(m);
      const auto& x = fixture.Get("activation_" + key + "_fp16"); Check(x, "F16", {m, k});
      Buffer input(q, DType::kF16, {m, k}), had(q, DType::kF16, {m, k});
      Buffer out(q, DType::kF16, {m, n}), raw(q, DType::kF32, {m, n});
      input.upload(x);
      vt::Exl3GemmArgs args{bits, 2};
      auto replay = [&](vt::Tensor& target) {
        vt::xpu::Exl3GemmReplayKernel(q, out.tensor, input.tensor, packed.tensor,
            suh.tensor, svh.tensor, had.tensor, args, target);
      };
      if (reject_route) {
        try { replay(raw.tensor); }
        catch (const std::runtime_error& e) {
          VT_CHECK(std::string(e.what()).find("cannot capture fused/prefill") != std::string::npos,
                   "unexpected route refusal");
          std::cout << "PASS replay refused unsupported route without changing dispatch\n";
          return 0;
        }
        VT_CHECK(false, "replay accepted an unsupported capture route");
      }
      vt::xpu::Exl3GemmKernel(q, out.tensor, input.tensor, packed.tensor,
          suh.tensor, svh.tensor, had.tensor, args);
      const auto normal_out = out.download(), normal_had = had.download();
      // Reject incorrect raw storage before changing any input/output bytes.
      for (int invalid = 0; invalid < 5; ++invalid) {
        auto bad = raw.tensor;
        if (invalid == 0) bad.dtype = DType::kF16;
        if (invalid == 1) --bad.shape[1];
        if (invalid == 2) bad.data = out.tensor.data;
        if (invalid == 3) ++bad.device.index;
        if (invalid == 4) ++bad.stride[0];
        bool refused = false;
        try { replay(bad); } catch (const std::runtime_error&) { refused = true; }
        VT_CHECK(refused && normal_out == out.download() && normal_had == had.download(),
                 "invalid raw storage was accepted or modified operands");
      }
      replay(raw.tensor);
      VT_CHECK(normal_out == out.download() && normal_had == had.download(),
               "replay hook changed normal kernel results");
      Save(blobs, key + "_input_hadamard_f16", had, "F16");
      Save(blobs, key + "_activation_f16", input, "F16");
      Save(blobs, key + "_native_raw_f32", raw, "F32");
      Save(blobs, key + "_production_output_f16", out, "F16");
      // Isolate output transform on the EXACT producer probe operand.
      const auto& operand = oracle.Get(key + "_output_hadamard_operand_f16");
      Check(operand, "F16", {m, n});
      Buffer probe(q, DType::kF16, {m, n}); probe.upload(operand);
      Save(blobs, key + "_output_hadamard_operand_f16", probe, "F16");
      vt::xpu::Exl3HadR128Kernel(q, out.tensor, probe.tensor, {nullptr, &svh.tensor, 1.0f});
      Save(blobs, key + "_output_hadamard_probe_f16", out, "F16");
      std::cout << "PASS " << key << " replay hook equals normal output; five invalid-storage cases refused\n";
    }
    Write(argv[3], blobs, device);
    std::cout << "native capture: " << argv[3] << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "exl3_projection_replay: " << e.what() << '\n';
    return 1;
  }
}
