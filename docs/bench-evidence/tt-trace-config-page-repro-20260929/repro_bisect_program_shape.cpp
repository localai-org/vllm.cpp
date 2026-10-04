// repro_bisect_program_shape.cpp — the N-kernel/N-CB/binary-size bisect the
// tt-trace-config-page-repro-20260929 note §4 names. Synthetic raw-tt-metal
// programs shaped like the keepquant int8-dot program (DataMovement kernel over
// the full grid, CBs, common RTAs), one knob at a time, each warmed then traced:
//   argv: [cb_count] [kernel_count] [table_kb] [cb_page_kb] [core_span]
// Prints trace bytes per launch per variant. Registering the realtime program
// profiler prints per-program dispatch byte classes (BINARY / RTARGS / CB_CONFIG).

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/mesh_trace_id.hpp>
#include <ttnn/device.hpp>
#include <ttnn/operations/trace.hpp>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
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

static std::string KernelSrc(uint32_t table_kb) {
  // table_kb > 0 pads the binary with a used const table (lands in .rodata,
  // exactly like the keepquant IQ tables).
  std::string s = "#include \"api/dataflow/dataflow_api.h\"\n";
  s += "#include <cstdint>\n";
  if (table_kb) {
    const uint32_t n = table_kb * 256u;
    s += "static const uint32_t kTable[" + std::to_string(n) + "] = {";
    for (uint32_t i = 0; i < n; ++i)
      s += (i ? "," : "") + std::to_string(0x01020304u + i * 0x9E3779B9u) + "u";
    s += "};\n";
  }
  s += R"(
void kernel_main() {
  constexpr uint32_t out_base = get_compile_time_arg_val(0);
  constexpr uint32_t page = get_compile_time_arg_val(1);
  uint32_t a0 = get_common_arg_val<uint32_t>(0);
  uint32_t a1 = get_common_arg_val<uint32_t>(1);
  uint64_t src = get_noc_addr(a0);
  uint32_t local = out_base;
  for (uint32_t i = 0; i < page; i += 16) {
    uint32_t w = a0 + a1 + i;
    volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(local + i);
    p[0] = w;
  }
  noc_async_write(out_base, src + 64, page);
  noc_async_write_barrier();
}
)";
  return s;
}

struct Variant {
  const char* name;
  uint32_t cbs;
  uint32_t kernels;
  uint32_t table_kb;
  uint32_t cb_page_kb;
};

int main(int argc, char** argv) {
  std::vector<Variant> variants;
  if (argc >= 5) {
    variants.push_back({"custom", static_cast<uint32_t>(std::atoi(argv[1])),
                        static_cast<uint32_t>(std::atoi(argv[2])),
                        static_cast<uint32_t>(std::atoi(argv[3])),
                        static_cast<uint32_t>(std::atoi(argv[4]))});
  } else {
    variants = {
        {"tiny-kern, 4cb, small pages", 4, 1, 0, 4},
        {"tiny-kern, 4cb, 16K pages  ", 4, 1, 0, 16},
        {"tiny-kern, 8cb             ", 8, 1, 0, 4},
        {"2 kernels, 4cb             ", 4, 2, 0, 4},
        {"big-table 32KB             ", 4, 1, 32, 4},
        {"big-table 64KB             ", 4, 1, 64, 4},
        {"big-table 128KB            ", 4, 1, 128, 4},
        {"big-table 256KB            ", 4, 1, 256, 4},
    };
  }

  auto dev = ttnn::open_mesh_device(0, DEFAULT_L1_SMALL_SIZE, 1ull << 28);
  dev->enable_program_cache();
  const CoreCoord g = dev->compute_with_storage_grid_size();
  printf("grid: %ux%u\n", (unsigned)g.x, (unsigned)g.y);

  for (const auto& v : variants) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    const CoreRange full({0, 0}, {g.x - 1, g.y - 1});
    const uint32_t cb_page = v.cb_page_kb * 1024u;
    for (uint32_t c = 0; c < v.cbs; ++c) {
      CircularBufferConfig cfg(v.cbs * cb_page,
                               {{static_cast<tt::CBIndex>(static_cast<int>(tt::CBIndex::c_0) + static_cast<int>(c)),
                                 tt::DataFormat::Float32}});
      cfg.set_page_size(static_cast<tt::CBIndex>(static_cast<int>(tt::CBIndex::c_0) + static_cast<int>(c)), cb_page);
      CreateCircularBuffer(program, full, cfg);
    }
    const std::string src = KernelSrc(v.table_kb);
    for (uint32_t k = 0; k < v.kernels; ++k) {
      // Disjoint vertical halves so two kernels never share a core.
      CoreRange kr({0u, k * 5u}, {g.x - 1, k * 5u + 4u});
      std::vector<uint32_t> cargs = {1024 * 64 + k * 16, cb_page};
      CreateKernelFromString(program, src, kr,
                             DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                                                .noc = NOC::RISCV_0_default,
                                                .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
                                                .compile_args = cargs});
    }
    SetCommonRuntimeArgs(program, 0, std::vector<uint32_t>{1024 * 1024, 12345});
    tt::tt_metal::distributed::MeshWorkload workload;
    workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange(dev->shape()), std::move(program));
    tt::tt_metal::Program& prog = workload.get_programs().begin()->second;
    // warm (load binaries) then trace one launch
    tt::tt_metal::distributed::EnqueueMeshWorkload(dev->mesh_command_queue(), workload, false);
    dev->mesh_command_queue().finish();
    const uint32_t bytes = Capture(*dev, [&] {
      prog.set_runtime_id(1);
      tt::tt_metal::distributed::EnqueueMeshWorkload(dev->mesh_command_queue(), workload, false);
    });
    printf("%s : cbs=%u kern=%u table=%uKB cbpage=%uKB -> trace %u B (%u/launch)\n",
           v.name, v.cbs, v.kernels, v.table_kb, v.cb_page_kb, bytes, bytes);
  }
  return 0;
}
