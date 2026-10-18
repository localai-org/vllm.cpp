// Tenstorrent verbatim operand staging + dtype-pivot readback
// (MODEL-TEXT-kolibri-1-tenstorrent, B2b-i first slice; spec
// .agents/specs/kolibri-tt.md ### B2 scope — B2b addendum, slice i).
//
// The B2b-i dense-resident slice's operands stage on the shared mesh
// device BYTES VERBATIM per the wave-A dtype decision: FP8_E4M3 operands
// native ROW_MAJOR (the pin marks FP8_E4M3 "Blackhole only, ROW-MAJOR only
// for now", tt_metal/api/tt-metalium/tensor/tensor_types.hpp:35-37), bf16
// and f32 ROW_MAJOR. Tilized consumption is the compute wave's business
// (spec §FP8 constraints: "tilized consumption happens inside the compute
// kernels"); this TU stages storage, not compute layouts.
//
// TWO staging routes, both byte-verbatim:
//  - FP8_E4M3: the pin's from_vector/from_span factories VALUE-CONVERT
//    through the to_dtype lattice (tt_metal/impl/tensor/
//    host_tensor_factory.cpp:97-113 — a uint8 buffer handed to an
//    FP8_E4M3 spec would convert the INTEGER values, destroying the fp8
//    bit patterns), so the host tensor is built directly from the raw
//    bytes (ttnn::Tensor(HostBuffer, TensorSpec) takes physical data
//    encoded in the buffer) and moved to device.
//  - bf16/f32: the established bulk from_span route
//    (tenstorrent_residency.cpp UploadRowsBf16) with a ROW_MAJOR spec —
//    from_span on a same-dtype buffer moves the bytes (no conversion).
//
// Readback: the pin's to_vector has no FP8 case and ttnn::to_dtype needs
// host storage (ttnn/core/tensor/tensor_ops.cpp:595-599), so an FP8_E4M3
// staged operand is brought to host (cpu() — a raw copy for interleaved
// ROW_MAJOR tensors) and pivoted host-side FP8→F32, the same pivot the
// pin's own print path uses (ttnn/core/tensor/tensor_impl.cpp:267-275).
#include "vt/tenstorrent/tenstorrent_internal.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace vt::tenstorrent {
namespace {

// The staged-operand registry (the #1486 never-destroyed residency
// pattern): handles are indices into this vector, stable for the process
// lifetime. The smoke's staged resident slice is the bring-up's
// deliverable; nothing re-stages or evicts it.
std::mutex& StagedOperandMutex() {
  static std::mutex m;
  return m;
}

std::vector<ttnn::Tensor>& StagedOperands() {
  static std::vector<ttnn::Tensor>* v = new std::vector<ttnn::Tensor>();
  return *v;
}

// One FP8_E4M3 ROW_MAJOR operand, packed bytes verbatim, on device.
ttnn::Tensor StageFp8E4M3(const void* host, uint32_t rows, uint32_t cols,
                          MeshDevice& device) {
  const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  std::vector<uint8_t> bytes(static_cast<const uint8_t*>(host),
                             static_cast<const uint8_t*>(host) + n);
  auto spec = tt::tt_metal::TensorSpec(
      tt::tt_metal::Shape({rows, cols}),
      tt::tt_metal::TensorLayout(
          ttnn::DataType::FP8_E4M3,
          tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
          tt::tt_metal::MemoryConfig{}));
  // The raw bytes ARE the tensor's physical encoding (one fp8 byte per
  // element, row-major) — no value conversion anywhere on this path.
  ttnn::Tensor host_tensor(tt::tt_metal::HostBuffer(std::move(bytes)),
                           spec);
  return host_tensor.to_device(&device, spec.memory_config());
}

// One bf16 ROW_MAJOR operand, bytes verbatim, on device.
ttnn::Tensor StageBf16(const void* host, uint32_t rows, uint32_t cols,
                       MeshDevice& device) {
  const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const bfloat16* src = static_cast<const bfloat16*>(host);
  return ttnn::Tensor::from_span(
      ttsl::Span<const bfloat16>(src, n),
      SpecOf(tt::tt_metal::Shape({rows, cols}), ttnn::DataType::BFLOAT16,
             ttnn::Layout::ROW_MAJOR),
      &device);
}

// One f32 ROW_MAJOR operand, bytes verbatim, on device.
ttnn::Tensor StageF32(const void* host, uint32_t rows, uint32_t cols,
                      MeshDevice& device) {
  const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const float* src = static_cast<const float*>(host);
  return ttnn::Tensor::from_span(
      ttsl::Span<const float>(src, n),
      SpecOf(tt::tt_metal::Shape({rows, cols}), ttnn::DataType::FLOAT32,
             ttnn::Layout::ROW_MAJOR),
      &device);
}

int64_t OperandBytes(const TtStageOperand& op) {
  switch (op.kind) {
    case TtStageOperand::Kind::kFp8E4M3:
      return op.rows * op.cols;
    case TtStageOperand::Kind::kBf16:
      return op.rows * op.cols * 2;
    case TtStageOperand::Kind::kF32:
      return op.rows * op.cols * 4;
  }
  return 0;
}

}  // namespace

TtStagingReport StageResidentOperands(Queue& q,
                                      const std::vector<TtStageOperand>& ops) {
  VT_CHECK(q.device.type == DeviceType::kTENSTORRENT,
           "tenstorrent StageResidentOperands: the queue must be a "
           "kTENSTORRENT queue");
  MeshDevice& device = SharedMeshDevice();
  TtStagingReport report;
  report.chips = static_cast<int64_t>(device.num_devices());
  if (report.chips > 0) {
    report.first_chip_id =
        static_cast<int64_t>(device.get_devices().front()->id());
  }
  report.dram_total_bytes = DeviceDramTotalBytes();
  const bool progress = std::getenv("VT_KOLIBRI1_TT_B2I_PROGRESS") != nullptr;
  if (progress) {
    fprintf(stderr, "[b2i-progress] mesh open: chips=%lld first_chip=%lld "
                    "dram_total=%lld ops=%zu\n",
            static_cast<long long>(report.chips),
            static_cast<long long>(report.first_chip_id),
            static_cast<long long>(report.dram_total_bytes), ops.size());
  }
  {
    std::lock_guard<std::mutex> g(StagedOperandMutex());
    for (const TtStageOperand& op : ops) {
      VT_CHECK(op.host != nullptr && op.rows > 0 && op.cols > 0,
               "tenstorrent StageResidentOperands: operand with a null host "
               "or a degenerate geometry");
      const uint32_t rows = static_cast<uint32_t>(op.rows);
      const uint32_t cols = static_cast<uint32_t>(op.cols);
      ttnn::Tensor staged =
          op.kind == TtStageOperand::Kind::kFp8E4M3
              ? StageFp8E4M3(op.host, rows, cols, device)
              : op.kind == TtStageOperand::Kind::kBf16
                    ? StageBf16(op.host, rows, cols, device)
                    : StageF32(op.host, rows, cols, device);
      report.bytes_staged += OperandBytes(op);
      ++report.tensors;
      if (op.kind == TtStageOperand::Kind::kFp8E4M3) {
        ++report.fp8_tensors;
      } else if (op.kind == TtStageOperand::Kind::kBf16) {
        ++report.bf16_tensors;
      } else {
        ++report.f32_tensors;
      }
      StagedOperands().push_back(std::move(staged));
      if (progress &&
          (report.tensors % 8 == 0 || report.tensors <= 8)) {
        fprintf(stderr,
                "[b2i-progress] staged %lld/%zu tensors, %lld bytes "
                "(last: kind=%d %ux%u = %lld B)\n",
                static_cast<long long>(report.tensors), ops.size(),
                static_cast<long long>(report.bytes_staged),
                static_cast<int>(op.kind), rows, cols,
                static_cast<long long>(OperandBytes(op)));
      }
    }
  }
  report.dram_free_after_bytes = DeviceDramFreeBytes();
  if (progress) {
    fprintf(stderr, "[b2i-progress] staging done: %lld tensors, %lld bytes, "
                    "dram_free_after=%lld\n",
            static_cast<long long>(report.tensors),
            static_cast<long long>(report.bytes_staged),
            static_cast<long long>(report.dram_free_after_bytes));
  }
  return report;
}

std::vector<float> ReadbackStagedOperandF32(Queue& q, int64_t handle) {
  VT_CHECK(q.device.type == DeviceType::kTENSTORRENT,
           "tenstorrent ReadbackStagedOperandF32: the queue must be a "
           "kTENSTORRENT queue");
  ttnn::Tensor staged;
  {
    std::lock_guard<std::mutex> g(StagedOperandMutex());
    VT_CHECK(handle >= 0 &&
                 handle < static_cast<int64_t>(StagedOperands().size()),
             "tenstorrent ReadbackStagedOperandF32: handle out of range");
    staged = StagedOperands()[static_cast<size_t>(handle)];
  }
  // Blocking D2H, then the dtype pivot the pin requires for FP8_E4M3
  // (see the file header). bf16/f32 read back directly.
  if (std::getenv("VT_KOLIBRI1_TT_B2I_PROGRESS") != nullptr) {
    static int64_t n = 0;
    if ((++n % 64) == 0) {
      fprintf(stderr, "[b2i-progress] readback %lld (handle=%lld)\n",
              static_cast<long long>(n), static_cast<long long>(handle));
    }
  }
  ttnn::Tensor host = staged.cpu(/*blocking=*/true);
  if (host.dtype() == ttnn::DataType::FP8_E4M3) {
    host = ttnn::to_dtype(host, ttnn::DataType::FLOAT32);
  }
  return host.to_vector<float>();
}

std::vector<uint8_t> ReadbackStagedOperandBytes(Queue& q, int64_t handle) {
  VT_CHECK(q.device.type == DeviceType::kTENSTORRENT,
           "tenstorrent ReadbackStagedOperandBytes: the queue must be a "
           "kTENSTORRENT queue");
  ttnn::Tensor staged;
  {
    std::lock_guard<std::mutex> g(StagedOperandMutex());
    VT_CHECK(handle >= 0 &&
                 handle < static_cast<int64_t>(StagedOperands().size()),
             "tenstorrent ReadbackStagedOperandBytes: handle out of range");
    staged = StagedOperands()[static_cast<size_t>(handle)];
  }
  ttnn::Tensor host = staged.cpu(/*blocking=*/true);
  // Dtype-agnostic raw bytes off the host buffer (to_vector/get_as have no
  // FP8_E4M3 case; see the header comment). Single-host mesh: one shard.
  std::vector<uint8_t> out;
  host.host_storage()
      .host_tensor()
      .buffer()
      .apply([&](const tt::tt_metal::HostBuffer& hb) {
        const auto bytes = hb.view_bytes();
        const auto* first = reinterpret_cast<const uint8_t*>(bytes.data());
        out.assign(first, first + bytes.size());
      });
  return out;
}

}  // namespace vt::tenstorrent
