// repro_trace_config_pages.cpp — minimal reproducer for per-launch CB/DFB
// config-page scaling of trace-recorded command sequences (tt-metal,
// Blackhole P150, eager+trace, program cache on).
//
// One trivial op (ttnn::multiply, bf16, zero data payload, kernel binary
// relayed by reference) is captured inside a trace twice: once at a 1-core
// extent (interleaved [32,32]) and once spanning the FULL device grid
// (height-sharded L1 over every compute core). The trace descriptor's
// total_trace_size (MeshTraceBuffer::desc, host-side assembled command
// stream) is printed for both, plus an N-launch loop to show per-launch
// linear growth. Expected if config pages are per-launch: full-grid trace
// >> 1-core trace, and the N-launch trace scales ~N x.

#include <ttnn/device.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/trace.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/mesh_trace_id.hpp>

#include <cstdio>
#include <vector>

using namespace tt::tt_metal;

static uint32_t Capture(ttnn::MeshDevice& dev, const std::function<void()>& body) {
  auto tid = ttnn::operations::trace::begin_trace_capture(&dev, std::nullopt);
  body();
  ttnn::operations::trace::end_trace_capture(&dev, tid, std::nullopt);
  const uint32_t bytes = dev.get_trace_buffers_size();
  ttnn::operations::trace::execute_trace(&dev, tid, std::nullopt, /*blocking=*/true);
  ttnn::operations::trace::release_trace(&dev, tid);
  return bytes;
}

int main() {
  auto dev = ttnn::open_mesh_device(/*device_id=*/0, /*l1_small_size=*/DEFAULT_L1_SMALL_SIZE,
                                    /*trace_region_size=*/1ull << 28);
  dev->enable_program_cache();
  const CoreCoord g = dev->compute_with_storage_grid_size();
  const uint32_t cores = g.x * g.y;
  printf("device compute grid: %ux%u (%u cores)\n", (unsigned)g.x, (unsigned)g.y, cores);

  // Case 1: [32,32] bf16, interleaved -> a 1-core extent.
  const tt::tt_metal::TensorSpec small_spec(
      tt::tt_metal::Shape({32, 32}),
      tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                                 tt::tt_metal::MemoryConfig{}));
  ttnn::Tensor a1 = ttnn::Tensor::from_vector<float>(std::vector<float>(32 * 32, 1.0f), small_spec, dev.get());
  ttnn::Tensor b1 = ttnn::Tensor::from_vector<float>(std::vector<float>(32 * 32, 2.0f), small_spec, dev.get());
  ttnn::multiply(a1, b1);  // warm: binaries must be loaded before capture
  const uint32_t one_core = Capture(*dev, [&] { ttnn::multiply(a1, b1); });

  // Case 2: same op, height-sharded across the FULL compute grid.
  const uint32_t rows = cores * 32;
  const tt::tt_metal::ShardSpec shard(
      CoreRangeSet(CoreRange({0, 0}, {g.x - 1, g.y - 1})), {32, 64}, ShardOrientation::ROW_MAJOR);
  const tt::tt_metal::TensorSpec big_spec(
      tt::tt_metal::Shape({rows, 64}),
      tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                                 tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                                                            tt::tt_metal::BufferType::L1, shard}));
  ttnn::Tensor a2 = ttnn::Tensor::from_vector<float>(std::vector<float>(rows * 64, 1.0f), big_spec, dev.get());
  ttnn::Tensor b2 = ttnn::Tensor::from_vector<float>(std::vector<float>(rows * 64, 2.0f), big_spec, dev.get());
  ttnn::Tensor out2 = ttnn::multiply(a2, b2);  // warm
  const auto& mc = out2.memory_config();
  printf("case2 memory layout=%d buffer=%d shard grid cores=%u\n",
         (int)mc.memory_layout(), (int)mc.buffer_type(),
         (unsigned)(mc.shard_spec() ? mc.shard_spec()->grid.num_cores() : 0));
  const uint32_t full_grid = Capture(*dev, [&] { ttnn::multiply(a2, b2); });

  // Case 3: 8 launches of the same full-grid op inside ONE trace.
  const uint32_t full_grid_x8 =
      Capture(*dev, [&] { for (int i = 0; i < 8; ++i) ttnn::multiply(a2, b2); });

  printf("trace bytes, 1 launch, 1-core extent   ([32,32] interleaved):   %u\n", one_core);
  printf("trace bytes, 1 launch, full-grid extent (%u cores sharded):   %u\n", cores, full_grid);
  printf("trace bytes, 8 launches, full-grid extent:                    %u (%u/launch)\n",
         full_grid_x8, full_grid_x8 / 8);
  return 0;
}
