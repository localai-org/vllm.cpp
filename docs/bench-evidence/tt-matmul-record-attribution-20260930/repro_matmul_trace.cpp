// repro_matmul_trace.cpp — per-launch trace-record cost of stock
// ttnn::matmul at realistic 27B shapes (tt-metal, Blackhole P150).
//
// Question: does a stock matmul at 27B-like shapes record ~MB per launch
// (class is tt-metal-side) or ~KB (class is specific to our MatmulBT /
// attention program construction)? Mirrors repro_trace_config_pages.cpp:
// warm, capture one launch (and an N-launch loop to divide out the fixed
// trace-descriptor overhead), read get_trace_buffers_size().

#include <ttnn/device.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/trace.hpp>
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

static tt::tt_metal::TensorSpec SpecBf16(std::vector<uint32_t> shape) {
  return tt::tt_metal::TensorSpec(tt::tt_metal::Shape(std::move(shape)),
                                  tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16,
                                                             tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                                                             tt::tt_metal::MemoryConfig{}));
}

static uint32_t MeasureMatmul(ttnn::MeshDevice& dev, std::vector<uint32_t> shape_a,
                              std::vector<uint32_t> shape_b, const char* label) {
  ttnn::Tensor a = ttnn::Tensor::from_vector<float>(
      std::vector<float>(shape_a[0] * shape_a[1], 0.5f), SpecBf16(shape_a), &dev);
  ttnn::Tensor b = ttnn::Tensor::from_vector<float>(
      std::vector<float>(shape_b[0] * shape_b[1], 0.25f), SpecBf16(shape_b), &dev);
  ttnn::matmul(a, b);  // warm: binaries must be loaded before capture
  // 1 launch minus the empty-trace floor; then 8 launches and divide, to
  // separate per-launch bytes from any fixed capture overhead.
  const uint32_t zero = Capture(dev, [] {});
  const uint32_t one = Capture(dev, [&] { ttnn::matmul(a, b); });
  const uint32_t eight = Capture(dev, [&] { for (int i = 0; i < 8; ++i) ttnn::matmul(a, b); });
  const uint32_t per_launch = (eight - one) / 7;
  printf("%-34s [%u,%u]x[%u,%u] bf16: empty=%u  1launch=%u  8launch=%u  per-launch(delta)=%u\n",
         label, shape_a[0], shape_a[1], shape_b[0], shape_b[1], zero, one, eight, per_launch);
  return per_launch;
}

int main() {
  auto dev = ttnn::open_mesh_device(/*device_id=*/0, /*l1_small_size=*/DEFAULT_L1_SMALL_SIZE,
                                    /*trace_region_size=*/1ull << 30);
  dev->enable_program_cache();
  const CoreCoord g = dev->compute_with_storage_grid_size();
  printf("device compute grid: %ux%u (%u cores)\n", (unsigned)g.x, (unsigned)g.y,
         (unsigned)(g.x * g.y));

  MeasureMatmul(*dev, {64, 5120}, {5120, 5120}, "27B-like hidden GEMM");
  MeasureMatmul(*dev, {64, 256}, {256, 256}, "small GEMM");
  MeasureMatmul(*dev, {64, 5120}, {5120, 1024}, "27B-like down-proj");

  // Control, run 2026-09-30 (log /tmp/mmrepro-run3.log): an inline H2D
  // from_vector issued INSIDE a capture does NOT record - the new pin
  // FATALS at fd_mesh_command_queue.cpp:830 ("Writes are not supported
  // during trace capture", trace id 9) and aborts. The old audit's
  // inline-H2D mechanism is therefore impossible on pin 6449cf13f7b.
  // Left out of the default run so EXIT=0; re-enable to reproduce the abort.
  return 0;
}
