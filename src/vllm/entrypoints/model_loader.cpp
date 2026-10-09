#include "vllm/v1/core/recurrent_prefix_snapshot.h"
// See include/vllm/entrypoints/model_loader.h. ORIGINAL packaging helper — the
// shared model-load + engine-stack wiring behind both the OpenAI server and the
// C ABI. Mirrors the M1.8 LLMEngine __init__ (vllm/v1/engine/llm_engine.py @
// e24d1b24) as exercised by examples/server/main.cpp and the test harness.
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/layers/attention/attention.h"
#include "vllm/model_executor/models/qwen3_dflash_gguf.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/config/cache.h"  // KV-FP8 W3: --kv-cache-dtype vs the checkpoint
#include "vllm/config/weight_residency.h"  // LOAD-IO: GGUF prefault bytes/seconds
#include "vllm/model_executor/layers/quantization/kv_cache.h"  // k/v scale arms
#include "vllm/model_executor/device_placement.h"
#include "vllm/model_executor/weight_offloader.h"
#include "vllm/model_executor/expert_stream_seam.h"  // MODEL-TEXT-GLM-MOE-DSA W3 (#2214): the load-time slot-capacity refusal
#include "vllm/model_executor/model_loader/gguf_device_fit.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/clip_mmproj_gguf.h"  // LOAD-GGUF-MMPROJ, #821
#include "vllm/model_executor/models/deepseek_v4.h"  // deepseek4 GGUF dispatch arm
#include "vllm/model_executor/models/deepseek_v4_1.h"  // the OWED deepseek41 GGUF refusal (W8)
#include "vllm/model_executor/models/dots3_note.h"  // the OWED dots3note GGUF refusal (#2882)
#include "vllm/model_executor/models/interfaces.h"  // #607 L3 SkipTowerForModalities
#include "vllm/model_executor/models/glm5_next_weights.h"  // glm5next GGUF arm
#include "vllm/model_executor/models/glm_moe_dsa.h"  // glm-dsa GGUF arm
#include "vllm/model_executor/models/laguna.h"  // laguna GGUF arm
#include "vllm/model_executor/models/muse_glimmer_gguf_weights.h"  // muse-glimmer GGUF arm
#include "vllm/model_executor/models/qwen4_exp_gguf_weights.h"  // qwen4exp GGUF arm
#include "vllm/model_executor/models/nemotron_h.h"  // the OWED nemotron_h* GGUF refusal (#809)
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"
#include "vllm/model_executor/models/qwen3_gguf_weights.h"  // qwen3 (dense) GGUF arm
#include "vllm/model_executor/models/qwen3_5_mtp.h"  // SPEC-MTP I5d-pre draft load
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/model_executor/models/qwen3_5_dense_mm.h"
#include "vllm/model_executor/layers/quantization/exl3_checkpoint.h"
#include "vllm/multimodal/qwen3vl_processor.h"
#include "vllm/model_executor/models/qwen3_5_common.h"  // SPEC-MTP I5d KV widening
#include "vllm/model_executor/models/qwen3_dflash.h"  // SPEC-DFLASH D5 draft load
#include "vllm/transformers_utils/hf_cache.h"  // ENG-HF-MODEL-DOWNLOAD (#1280)
#include "vllm/transformers_utils/hf_config.h"  // SPEC-DFLASH D5 draft config
#include "vllm/transformers_utils/tokenizer_files.h"
#include "vllm/platforms/interface.h"  // CurrentPlatform() — SelectQueue
#include "vllm/v1/core/hybrid_kv_budget.h"
#include "vllm/v1/core/kv_cache_utils.h"  // check_enough_kv_cache_memory (M4)
#include "vllm/v1/kv_cache_interface.h"  // FIX-KV-GROUP-LAYER-COUNT resolver
#if defined(VLLM_CPP_TENSTORRENT)
#include "vt/tenstorrent/tenstorrent_device.h"  // W4d W0 (#3042) alloc trace
#endif
#include "vllm/v1/structured_output/backend_native.h"  // MakeNativeBackendFactory
#include "vllm/v1/structured_output/jump_forward.h"     // JumpForwardEnabled (SW3)
#include "vt/dtype.h"
#include "vt/tensor.h"
#if defined(VLLM_CPP_CUDA) && defined(VT_CUTLASS_NVFP4)
#include "vt/cuda/nvfp4_autotune.h"
#endif

namespace vllm::entrypoints {

namespace fs = std::filesystem;

// W4d W0 (#3042) attribution: DRAM bank snapshots around the load stages.
#if defined(VLLM_CPP_TENSTORRENT)
static void TtAllocTraceStage(const char* stage) {
  vt::tenstorrent::AllocTraceSnapshot(vt::tenstorrent::SharedMeshDevice(),
                                      stage);
}
#else
static inline void TtAllocTraceStage(const char*) {}
#endif

// `architecture` is the model's registered architecture string. It is what lets
// a PARTIAL backend decline a model whose kernels it has not registered, instead
// of being selected and then failing deep inside a kernel bind. Empty means "no
// model resolved yet", which is treated as no constraint.
//
// ARCH-ONE-SURFACE ROW 8: `device` is the caller's explicit selection
// (EngineParams::device / vllm_model_params.device). kAuto keeps the
// accelerator-first probe below byte-identical; an EXPLICIT selection routes
// through LoadedEngine::ResolveExplicitDeviceType and — unlike the auto arm —
// a failure to serve the named device PROPAGATES instead of falling back to
// CPU (mirror of vLLM never substituting an explicitly named device,
// vllm/config/device.py:61-66).
namespace {

// The auto arm of the resolution below, WITHOUT creating a queue. Extracted so
// the queue selector and the load-time device-fit refusal (issue #1123) read one
// description of "which device will this model run on" rather than two that can
// drift. May throw, exactly as `CurrentPlatform()` can, and every caller keeps
// the try/catch the original code had around it.
vt::DeviceType AutoAcceleratorDeviceType(std::string_view architecture) {
  const vllm::platforms::Platform& plat = vllm::platforms::CurrentPlatform();
  const vt::DeviceType dev = plat.device_type();
  // A PARTIAL backend (Metal today: 15 of 75 ops) must be able to decline a
  // model whose kernels it has not registered. The default answer is `true`,
  // so CUDA and CPU selection is byte-unchanged.
  if (dev != vt::DeviceType::kCPU &&
      (architecture.empty() || plat.supports_model_architecture(architecture))) {
    return dev;
  }
  return vt::DeviceType::kCPU;
}

// The AUTO arm, resolved by ATTEMPTING the queue. One implementation, so
// `ResolveModelDeviceType` and `SelectQueueForModel` cannot answer differently.
//
// Asking `CurrentPlatform()` alone is not enough, and #1136 measured why. This
// arm has always fallen back to CPU when `CreateQueue()` throws — "a platform can
// be registered while CreateQueue still fails, and CPU must remain reachable" —
// so on such a box a platform query answers `kCUDA` while the load runs on the
// CPU queue. The load-time device-fit refusal reads the query, and it therefore
// refused a checkpoint by naming a device nothing was going to run on, removing a
// load that previously served on CPU. Whether `CreateQueue()` fails is knowable
// only by calling it, so it is called here, once, and the queue goes to whichever
// caller wants one.
struct AutoDeviceResolution {
  vt::DeviceType device = vt::DeviceType::kCPU;
  // Set exactly when `device != kCPU`: the queue whose creation PROVED it.
  std::optional<vt::Queue> queue;
};

AutoDeviceResolution ResolveAutoDevice(std::string_view architecture) {
  AutoDeviceResolution out;
  try {
    const vt::DeviceType dev = AutoAcceleratorDeviceType(architecture);
    if (dev != vt::DeviceType::kCPU) {
      // Order matters: `device` is set only AFTER the queue exists, so a throw
      // leaves the CPU answer rather than a device nothing can serve.
      vt::Queue q = vt::GetBackend(dev).CreateQueue();
      out.queue = q;
      out.device = dev;
    }
  } catch (const std::exception&) {
    // No usable accelerator; CPU, which is what this arm has always returned.
  }
  return out;
}

}  // namespace

vt::DeviceType ResolveModelDeviceType(std::string_view architecture,
                                      vllm::Device device) {
  if (device != vllm::Device::kAuto) {
    const vllm::platforms::Platform* named_platform =
        vllm::platforms::FindPlatformByName(vllm::DeviceName(device));
    // Propagates for an explicitly named absent device, which is the refusal
    // vllm/config/device.py:61-66 mirrors and must not be swallowed here.
    return LoadedEngine::ResolveExplicitDeviceType(
        device, named_platform == nullptr
                    ? std::nullopt
                    : std::optional{named_platform->device_type()});
  }
  AutoDeviceResolution resolved = ResolveAutoDevice(architecture);
  // The queue was created only to learn whether it CAN be created. `vt::Queue` is
  // a NON-OWNING handle (a raw `cudaStream_t`) with no destructor, so dropping the
  // value would leak the stream.
  //
  // Through the FREE `vt::DestroyQueue`, not `Backend::DestroyQueue`: that is what
  // this file's only other queue teardown does (`load_queue`, below), it is what
  // `vt/backend.h` asks of new code so device index and queue cleanup are never
  // ambient, and it adds the `Synchronize` and the handle/id clearing the method
  // does not. The CREATE side deliberately stays `GetBackend(...).CreateQueue()`,
  // because that is the call this arm has always made and switching it would move
  // the production queue-selection path onto the drop-in resource ABI — a
  // behaviour change, which this repair is not.
  if (resolved.queue.has_value()) vt::DestroyQueue(*resolved.queue);
  return resolved.device;
}

// ENG-HYBRID-PLACEMENT W2 (#2023): build the resolved placement and SAY it.
//
// Called from `SelectQueueForModel`, which is the one function every load path
// goes through and which already owns the device decision — its neighbour comment
// records that it and `ResolveModelDeviceType` cannot answer differently. That
// makes this "at model build" in the only sense the spec means, and it covers the
// GGUF and safetensors paths with one call site rather than three.
//
// W2 RESOLVES AND REPORTS; IT MOVES NOTHING. The returned placement is dropped
// here on purpose: W3 owns the routing that reads it. What lands today is the
// refusal of an impossible placement and the line an operator needs to attribute
// a slow run, and a placement that changes nothing prints nothing at all, so an
// ordinary load is byte-identical on stderr as well as in behaviour.
void ReportDevicePlacement(vt::DeviceType engine_device) {
  const std::vector<vllm::PlacementOverride> overrides =
      vllm::ResolvePlacementOverrides();
  if (overrides.empty()) return;
  const vllm::DevicePlacement placement =
      vllm::DevicePlacement::FromOverrides(overrides, engine_device);
  const std::string described = placement.Describe();
  if (described.empty()) {
    // Non-empty overrides that place NOTHING away from this device: `cpu_moe` on
    // a CPU engine is exactly this. Say so, because the operator asked for
    // something and is entitled to know it was inert rather than ignored.
    std::cerr << "engine: device placement: " << overrides.size()
              << (overrides.size() == 1 ? " override resolves"
                                        : " overrides resolve")
              << " to the engine's own device (" << vt::DeviceTypeName(engine_device)
              << "), so nothing is placed" << std::endl;
    return;
  }
  std::cerr << "engine: device placement: " << described << std::endl;
}

// ENG-HYBRID-PLACEMENT (#2314): INSTALL the resolved plan, which is the step that
// makes every part above actually move a weight.
//
// W2's neighbour comment says it "RESOLVES AND REPORTS; IT MOVES NOTHING ... W3
// owns the routing that reads it". W3 built that routing — the `RunMoePlaced`
// seam and five architectures on it — and never added this call, so the seam read
// a global nothing ever wrote. `ActiveMoePlacementPlan()` returned the default on
// every load, `PlacesAnything()` was always false, and NO expert was ever placed.
//
// THE ANNOUNCEMENT IS WHY THAT SURVIVED. `ReportDevicePlacement` prints the
// RESOLVED plan, so an operator running `VT_CPU_MOE=1` read
// "device placement: N layers on cpu" on stderr and had every reason to believe
// it. A token gate cannot see it either: with nothing placed the placed arm is
// byte-identical to the unplaced one, so it passes for the wrong reason.
//
// UNCONDITIONAL, including when nothing is placed. The plan lives in a
// process-wide global, so a second load in the same process must overwrite the
// first model's plan rather than inherit it; an early return on "no overrides"
// would leave a stale placement installed against the wrong model.
void InstallMoePlacementPlan(vt::DeviceType engine_device,
                             int64_t num_hidden_layers,
                             const vllm::GgufFile* gguf) {
  // An EXPLICIT `--fit` beside a manual placement is refused here rather than at
  // parse time, which closes two holes the parse-time check cannot see: a
  // multi-document merge, and the environment. A DEFAULTED fit is not a
  // collision — it yields, and the manual placement wins.
  if (const std::string collision = vllm::DescribePlacementFitCollision();
      !collision.empty()) {
    throw std::invalid_argument(collision);
  }

  std::vector<vllm::PlacementOverride> overrides =
      vllm::ResolvePlacementOverrides();
  vllm::PlacementOrigin origin = overrides.empty()
                                     ? vllm::PlacementOrigin::kNone
                                     : vllm::PlacementOrigin::kStated;
  vllm::MoeFitResolution fit;

  // W4 (#2384): `--fit` asks the resolver to decide the placement instead of
  // stating it. Mutually exclusive with a manual placement, which the config
  // parse already refuses, so reaching here with both is not expected.
  if (vllm::ResolvePlacementFit() && overrides.empty()) {
    if (gguf == nullptr) {
      // INERT WHEN DEFAULTED, FATAL WHEN ASKED. `--fit` is on by default
      // (mirroring llama.cpp), so refusing every safetensors load over a feature
      // nobody requested would make that default a breaking change. But an
      // operator who explicitly asked must NOT be told silently that it did not
      // happen -- that is the #2382 failure, where a placement was announced and
      // never installed.
      if (!vllm::PlacementFitWasRequested()) {
        std::cerr << "engine: device placement: --fit is on by default but "
                     "cannot apply to a safetensors checkpoint (its weight "
                     "footprint is not known where the placement must be "
                     "installed); continuing with no placement"
                  << std::endl;
        vllm::MoePlacementPlan bare = vllm::MoePlacementPlan::Resolve(
            vllm::DevicePlacement::FromOverrides({}, engine_device),
            num_hidden_layers);
        bare.set_origin(vllm::PlacementOrigin::kNone);
        vllm::SetActiveMoePlacementPlan(bare);
        return;
      }
      // REFUSE BY NAME rather than resolve to nothing. The model's weight
      // footprint is not available at this point on the safetensors path -- the
      // shards open downstream of where the plan must be installed, because
      // `ResidentWeight` aliases host bytes on a CPU `Dev` and uploads
      // otherwise, so installing after the upload pays the round trip the
      // placement exists to avoid. A silent "fit resolved nothing" here is the
      // #2382 failure again: the operator asks for a placement, sees no error,
      // and gets none.
      throw std::invalid_argument(
          "device placement: \"vllm_cpp.placement.fit\" (--fit) is implemented "
          "for GGUF checkpoints only; this is a safetensors checkpoint, whose "
          "weight footprint is not known at the point the placement must be "
          "installed. State the placement instead with \"cpu_moe\", "
          "\"n_cpu_moe\" or \"overrides\"");
    }
    const size_t budget = vllm::DeviceWeightBudgetBytes(
        vllm::platforms::GetPlatform(engine_device)
            .residency_policy()
            .device_memory_total_bytes);
    const vllm::GgufStagedFootprint footprint =
        vllm::GgufStagedWeightFootprint(*gguf);
    fit = vllm::ResolveMoeFitFromSizes(
        footprint.lower_bound_bytes, budget,
        vllm::GgufRoutedExpertBytesPerLayer(*gguf, num_hidden_layers));

    if (fit.resolved && fit.placed_layers > 0) {
      overrides.clear();
      for (int64_t l = num_hidden_layers - fit.placed_layers;
           l < num_hidden_layers; ++l)
        overrides.push_back({vllm::LlmFfnExpsBlockRegex(l), "cpu"});
      origin = vllm::PlacementOrigin::kFit;
    }
    // A resolver that declined says why on stderr, because "--fit did nothing"
    // is otherwise indistinguishable from "--fit is broken".
    if (!fit.resolved) {
      std::cerr << "engine: device placement: --fit resolved NO placement: "
                << fit.reason << std::endl;
    } else if (fit.placed_layers == 0) {
      std::cerr << "engine: device placement: --fit places nothing; the model "
                   "already fits the budget" << std::endl;
    }
  }

  vllm::MoePlacementPlan plan = vllm::MoePlacementPlan::Resolve(
      vllm::DevicePlacement::FromOverrides(overrides, engine_device),
      num_hidden_layers);
  plan.set_origin(plan.PlacesAnything() ? origin : vllm::PlacementOrigin::kNone);
  plan.set_fit(fit);
  vllm::SetActiveMoePlacementPlan(plan);

  // Printed FROM THE INSTALLED PLAN, and this is the distinction that matters
  // rather than a duplicate of `ReportDevicePlacement`. That line prints what
  // RESOLVED, so it appeared unchanged through the whole period when nothing was
  // installed and nothing was placed — it is what made #2314 invisible to the one
  // signal an operator checks. This line cannot appear unless the plan reached
  // the seam's global, so a gate can distinguish a real placement from a vacuous
  // one, which a token comparison alone cannot do.
  //
  // Only when it actually places, so an ordinary load stays byte-identical on
  // stderr.
  if (plan.PlacesAnything()) {
    std::cerr << "engine: device placement INSTALLED: " << plan.Describe()
              << " (resolved against " << plan.resolved_layer_count()
              << " layers, origin "
              << vllm::PlacementOriginName(plan.origin()) << ")" << std::endl;
    if (plan.origin() == vllm::PlacementOrigin::kFit) {
      // The ARITHMETIC, not just the verdict. An operator who cannot see the
      // budget and the footprint cannot tell a wrong placement from a wrong
      // budget. The whole-layer granularity is stated here too, because this
      // resolver cannot place the half-layer upstream can and a user comparing
      // against llama.cpp would otherwise have to infer that from a number.
      std::cerr << "engine: device placement: --fit placed " << fit.placed_layers
                << " layer(s) (" << fit.placed_bytes << " B) to bring a "
                << fit.footprint_bytes << " B footprint under a "
                << fit.budget_bytes
                << " B budget; WHOLE layers only, so a partial layer is taken "
                   "entirely"
                << (fit.still_exceeds
                        ? "; STILL EXCEEDS the budget with every layer placed"
                        : "")
                << std::endl;
    }
  }
}

vt::Queue SelectQueueForModel(std::string_view architecture,
                              vllm::Device device) {
  if (device != vllm::Device::kAuto) {
    const vt::DeviceType resolved =
        ResolveModelDeviceType(architecture, device);
    // NOT reported here. An EXPLICIT device is reported far earlier, in
    // `FromModelDir`'s up-front device-resolution block, because that is ahead of
    // all path and weight I/O and is therefore where an operator still sees the
    // line when the load then fails. Reporting in both places would print twice.
    if (resolved == vt::DeviceType::kCPU) {
      return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
    }
    // No try/catch here on purpose: an explicit accelerator whose queue cannot
    // be created must FAIL the load loudly, never silently serve on CPU.
    return vt::GetBackend(resolved).CreateQueue();
  }
  // The AUTO path reports here and not earlier: the architecture participates in
  // the answer (`ResolveAutoDevice`), so the up-front block cannot know it yet.
  // The explicit path is the other way round and is reported there.
  ReportDevicePlacement(ResolveModelDeviceType(architecture, device));

  // M2.2b: run the engine forward on the ACCELERATOR when one is available, so
  // (on CUDA/GB10) the fp4-resident MoE/lm_head weights hit vt::MatmulNvfp4
  // on-device instead of the CPU dequant reference.
  //
  // W0b-1 item 1 (.agents/specs/metal-mlx-reuse-study.md §3.3), closed by work
  // row M3a: this hardcoded `GetBackend(kCUDA)`, which made the engine CPU-only
  // on every non-NVIDIA accelerator no matter how complete that backend was —
  // the single line that stood between the Metal backend and running a model.
  // It now asks the PLATFORM seam, which is the tree's own answer to "which
  // device is this process running on": CurrentPlatform() walks
  // {kCUDA, kROCM, kXPU, kVULKAN, kMETAL, kTENSTORRENT, kCPU} and returns the
  // first whose backend actually probed a device
  // (src/vllm/platforms/platform.cpp:91-98), so on a CUDA box this selects
  // EXACTLY the queue the old code did, byte for byte, and on the M4 it selects
  // Metal. The try/catch stays, now inside `ResolveAutoDevice`: a platform can be
  // registered while CreateQueue still fails, and CPU must remain reachable.
  AutoDeviceResolution resolved = ResolveAutoDevice(architecture);
  if (resolved.queue.has_value()) return *resolved.queue;
  return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
}

namespace {

// --- Issue #150 load-time instrumentation -----------------------------------
// "Measure it properly, then cut it": `VT_LOAD_STATS=1` prints the wall time of
// each load phase and the bytes the load actually MOVED, so the cost of the
// weight path is a measured number instead of an inferred one. Off by default
// and read once; when off this costs two clock reads per load.
bool LoadStatsEnabled() {
  static const bool enabled = [] {
    const char* e = std::getenv("VT_LOAD_STATS");
    return e != nullptr && e[0] != '0';
  }();
  return enabled;
}

double SecondsSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count();
}

void ReportLoadPhase(const char* phase, double seconds) {
  if (!LoadStatsEnabled()) return;
  std::fprintf(stderr, "[vt load] %-14s %8.3f s\n", phase, seconds);
}

void PrintLoadBytes(const char* when) {
  const vllm::load_stats::Counters c = vllm::load_stats::Snapshot();
  const double gib = 1024.0 * 1024.0 * 1024.0;
  std::fprintf(stderr,
               "[vt load] bytes@%-9s host_copy=%.3f GiB borrowed=%.3f GiB "
               "device_upload=%.3f GiB\n",
               when, static_cast<double>(c.host_copy_bytes) / gib,
               static_cast<double>(c.borrowed_bytes) / gib,
               static_cast<double>(c.device_upload_bytes) / gib);
}

// LOAD-IO. The GGUF branch does NOT call ReportLoadBytes: every one of those
// counters is incremented on the safetensors path only (`AddHostCopy` in
// safetensors_reader.cpp, `AddBorrowed` in qwen3_5_weights.cpp's
// `BorrowStTensorBytes`), so on a `.gguf` load that line prints three zeros for
// an artifact it just moved 67.56 GiB of. A zero that means "nobody counted"
// printed beside two real timings is worse than no line, because it reads as a
// measurement. This reports the counters a GGUF load DOES keep.
void ReportGgufLoadIo() {
  if (!LoadStatsEnabled()) return;
  const uint64_t bytes = vllm::GgufPrefaultedBytes();
  const double seconds = vllm::GgufPrefaultSeconds();
  const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
  std::fprintf(stderr,
               "[vt load] gguf prefault spans=%llu paged_in=%.3f GiB in %.3f s",
               static_cast<unsigned long long>(vllm::GgufPrefaultedSpanCount()),
               gib, seconds);
  if (seconds > 0.0) {
    std::fprintf(stderr, " (%.1f MiB/s)", gib * 1024.0 / seconds);
  }
  std::fprintf(stderr, "\n");
}

void ReportLoadBytes() {
  if (!LoadStatsEnabled()) return;
  PrintLoadBytes("load-end");
  // The device uploads are LAZY -- ResidentWeight runs at first forward use,
  // after this function returns -- so the load-end snapshot always reads
  // device_upload=0. Print the final totals at exit as well, which is where the
  // "bytes moved by this process" question is actually answered. Registered
  // once; std::atexit handlers cannot take an argument, hence the wrapper.
  static const bool once = [] {
    std::atexit([] { PrintLoadBytes("exit"); });
    return true;
  }();
  (void)once;
}

bool DirectDeviceLoadRequested() {
  const char* release = std::getenv("VT_RELEASE_HOST_WEIGHTS");
  if (release != nullptr && release[0] == '0') return false;
  const char* direct = std::getenv("VT_DIRECT_DEVICE_LOAD");
  return direct == nullptr || direct[0] != '0';
}

std::vector<vllm::SafetensorsFile> LoadShards(const std::string& model_dir) {
  std::vector<std::string> paths;
  for (const auto& e : fs::directory_iterator(model_dir)) {
    if (e.is_regular_file() && e.path().extension() == ".safetensors") {
      paths.push_back(e.path().string());
    }
  }
  if (paths.empty()) {
    throw std::runtime_error("no *.safetensors shards found in " + model_dir);
  }
  std::sort(paths.begin(), paths.end());
  std::vector<vllm::SafetensorsFile> shards;
  shards.reserve(paths.size());
  for (const std::string& p : paths) {
    shards.push_back(vllm::SafetensorsFile::Open(p));
  }
  return shards;
}

#if defined(VLLM_CPP_CUDA) && defined(VT_CUTLASS_NVFP4)
bool EnvironmentEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr || value[0] != '0';
}
#endif

// ── SPEC-DFLASH D5: separate DFlash draft-checkpoint load ────────────────────
// Resolve the DFlash draft path: a local directory (with config.json) is used
// as-is; an HF repo id ("z-lab/Qwen3.6-27B-DFlash") resolves to the newest
// ~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<hash>/ dir.
// True when `path` names a DFlash draft packaged as a single `dflash`-arch GGUF
// rather than a safetensors directory (SPEC-DFLASH-GGUF GD3). Checked before the
// config.json probe because a .gguf file has no config.json and would otherwise
// fall through to the HF-cache search and be reported as "not found".
bool IsDflashGgufDraft(const std::string& path) {
  std::error_code ec;
  return fs::is_regular_file(path, ec) &&
         fs::path(path).extension() == ".gguf";
}

std::string ResolveDflashDraftDir(const std::string& path) {
  // ENG-HF-MODEL-DOWNLOAD W2 (#1280): this used to carry its own copy of the
  // HuggingFace cache walk. The walk now lives in
  // vllm::transformers_utils::ResolveCachedSnapshotDir, so the tree holds ONE
  // implementation of it.
  //
  // The cache root stays $HOME/.cache/huggingface/hub, which is what this path
  // has always read, rather than HfHubCacheDir(). This move is a relocation and
  // must not change what a DFlash run resolves; a container that sets HF_HOME
  // would otherwise start resolving a different directory as a side effect. The
  // migration onto HfHubCacheDir() is listed under `## Owed` in
  // .agents/specs/hf-model-download.md.
  const char* home = std::getenv("HOME");
  const fs::path hub_dir =
      home == nullptr ? fs::path() : fs::path(home) / ".cache/huggingface/hub";
  return vllm::transformers_utils::ResolveCachedSnapshotDir(path, hub_dir);
}

// The loader-local `LoadNamedBf16` that used to live here (the memcpy read of
// the draft's shared tensors) is gone as of SPEC-DFLASH2 W9 (#1849): its one
// remaining caller was the shared-embed read, which now goes through the
// exported borrow-first `LoadDflashSharedEmbedBf16` (qwen3_dflash.h) so the
// borrow is gated at the exact function production calls.

// SPEC-DFLASH-GGUF B1: WHERE the draft's SHARED bf16 embed_tokens + lm_head
// come from.
//
// A DFlash draft owns neither. It runs the TARGET's embedding table and the
// TARGET's lm_head over its own hidden states, which is why the z-lab
// safetensors checkpoint and llama.cpp's DFLASH GGUF arch both omit them. Until
// axis B that sharing was expressed as a `const std::vector<SafetensorsFile>&`
// parameter on LoadDflashDraft, and THAT TYPE was the axis-B blocker: a GGUF
// target has no shards to point at, so the whole feature was refused in
// FromModelDir's GGUF branch. This is the same seam re-expressed as a SOURCE,
// which makes every line of draft-side loading identical for both containers.
//
// It holds a non-owning pointer to whichever the caller has: both live inside
// FromModelDir for the duration of the load.
class SharedHeadSource {
 public:
  explicit SharedHeadSource(const std::vector<vllm::SafetensorsFile>* shards)
      : shards_(shards) {}
  explicit SharedHeadSource(const vllm::GgufFile* gguf) : gguf_(gguf) {}

  // Fill the draft's two shared tensors. Both arms produce the SAME thing: bf16
  // `[vocab, H]` with nk=false for the gather table and the same `[vocab, H]`
  // with nk=true for the MatmulBT head. Throws naming the source on absence.
  //
  // `head_was_quantized` is REQUIRED and has no default, which is a deliberate
  // change from the defaulted `= nullptr` it carried through W3. SPEC-DFLASH2's
  // fresh review found that deleting the third ARGUMENT at the DFlash call site
  // below compiled clean and left all 38 dflash/gguf suites green after a full
  // relink: the default silently turned the carry off, `lm_head_dequantized`
  // stayed false, and D12's `RefuseQuantizedDflash2LmHead` guard lost its
  // trigger while every gate stayed green. That is a silent-wrong the type system
  // can refuse outright, so it does: every caller now names what it wants, and
  // dropping the argument is a COMPILE ERROR rather than a green run. The DSpark
  // caller passes `nullptr` explicitly and says why.
  //
  // `head_fp4` is the SPEC-DFLASH2-QUANT-LMHEAD (#1628) owner and is REQUIRED
  // for the same reason `head_was_quantized` is: a defaulted argument is what
  // silently turned the D12 carry off. `nullptr` means "this lane cannot compute
  // with a packed head", and the safetensors arm then refuses a quantized target
  // head exactly as it did before that row. The DSpark caller passes it and says
  // why; the GGUF arm never sets it, because a GGUF target's `output.weight` is
  // dequantized on the way in and is the case D12 refuses.
  //
  // `head_exl3` is the MODEL-QWEN35-EXL3-HEAD (#2495 item 6) owner and is
  // REQUIRED for the third time and the third reason that is the SAME reason:
  // `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` ships `lm_head.{trellis,suh,svh,mul1}`
  // and no `lm_head.weight` at all, so a lane that quietly declined the trellis
  // would report a complete head as absent. `nullptr` means "this lane cannot
  // compute with a trellis head" and the read then refuses BY NAME. The DSpark
  // caller passes it and says why; the GGUF arm never sets it, because a GGUF
  // target's `output.weight` is dequantized on the way in — the same sentence
  // that already applies to `head_fp4`, and for the same reason.
  void LoadInto(vllm::OwnedTensor* embed, vllm::OwnedTensor* head,
                bool* head_was_quantized, vllm::Nvfp4Weight* head_fp4,
                vllm::Exl3Weight* head_exl3) const {
    if (head_was_quantized != nullptr) *head_was_quantized = false;
    if (head_fp4 != nullptr) *head_fp4 = vllm::Nvfp4Weight{};
    if (head_exl3 != nullptr) *head_exl3 = vllm::Exl3Weight{};
    if (gguf_ != nullptr) {
      vllm::LoadGgufSharedEmbedAndHeadBf16(*gguf_, embed, head, head_was_quantized);
    } else {
      // SPEC-DFLASH2 W9 (#1849): both shared reads are borrow-first now (the
      // fail-closed BorrowStTensorBytes seam) — on a real target each copy
      // this replaces was a ~2.54 GB anonymous buffer.
      *embed = vllm::LoadDflashSharedEmbedBf16(
          *shards_, "model.language_model.embed_tokens.weight");
      vllm::LoadDflashSharedLmHead(*shards_, head, head_fp4, head_exl3);
    }
    // The head half of this check is now a belt to `LoadDflashSharedLmHead`'s
    // own braces (#2569): that function used to fall off the end of its bf16
    // loop and return silently with every owner empty, which made THIS line the
    // only refusal — and it names bf16 tensors, which is the wrong sentence for
    // a target whose head is present and simply not dense. It throws by name
    // itself now. This stays because the GGUF arm reaches it too.
    if (embed->Empty() ||
        (head->Empty() && (head_fp4 == nullptr || head_fp4->Empty()) &&
         (head_exl3 == nullptr || head_exl3->Empty()))) {
      throw std::runtime_error(
          "dflash: the target's bf16 embed_tokens + lm_head (which the draft "
          "SHARES) were not found in " +
          Describe());
    }
  }

  std::string Describe() const {
    return gguf_ != nullptr ? std::string("the GGUF target file")
                            : std::string("the target safetensors shards");
  }

 private:
  const std::vector<vllm::SafetensorsFile>* shards_ = nullptr;
  const vllm::GgufFile* gguf_ = nullptr;
};

// Build the DSpark draft HfConfig from its config.json (SPEC-DSPARK W5).
//
// A DSpark config differs from the DFlash one in three ways that all bite:
//   * `mask_token_id` / `target_layer_ids` sit at the TOP LEVEL, not inside a
//     nested `dflash_config`. The inherited backbone helpers read
//     `dflash_config`, exactly as upstream's _dflash_layer_causal does on a
//     DSpark config (getattr -> {} -> fall back to layer_types), so we synthesize
//     that sub-object here rather than fork the helpers.
//   * `sliding_window` may be JSON null (the 4B/8B drafts are all full-attention).
//   * `rope_theta` lives under `rope_parameters`, not at the top level.
// Speculators-format configs are translated to this shape first
// (Qwen3DSparkModel::TranslateSpeculatorsDsparkConfig).
vllm::HfConfig MakeDsparkDraftConfig(const nlohmann::json& c) {
  vllm::HfConfig cfg;
  cfg.hidden_size = c.at("hidden_size").get<int64_t>();
  cfg.num_attention_heads = c.at("num_attention_heads").get<int64_t>();
  cfg.num_key_value_heads = c.at("num_key_value_heads").get<int64_t>();
  cfg.head_dim = c.at("head_dim").get<int64_t>();
  cfg.rotary_dim = cfg.head_dim;
  cfg.rope_theta = 10000.0;
  if (c.contains("rope_theta") && c.at("rope_theta").is_number()) {
    cfg.rope_theta = c.at("rope_theta").get<double>();
  } else if (c.contains("rope_parameters") && c.at("rope_parameters").is_object() &&
             c.at("rope_parameters").contains("rope_theta")) {
    cfg.rope_theta = c.at("rope_parameters").at("rope_theta").get<double>();
  }
  cfg.intermediate_size = c.at("intermediate_size").get<int64_t>();
  cfg.vocab_size = c.at("vocab_size").get<int64_t>();
  cfg.num_hidden_layers = c.at("num_hidden_layers").get<int64_t>();
  cfg.rms_norm_eps = c.at("rms_norm_eps").get<double>();
  if (c.contains("sliding_window") && c.at("sliding_window").is_number_integer()) {
    cfg.sliding_window = c.at("sliding_window").get<int64_t>();
  }
  if (c.contains("layer_types") && c.at("layer_types").is_array()) {
    cfg.layer_types = c.at("layer_types").get<std::vector<std::string>>();
  }
  cfg.raw = c;
  // Synthesize the nested dflash_config the inherited backbone reads.
  nlohmann::json dflash_config = nlohmann::json::object();
  if (c.contains("mask_token_id")) dflash_config["mask_token_id"] = c.at("mask_token_id");
  if (c.contains("target_layer_ids")) {
    dflash_config["target_layer_ids"] = c.at("target_layer_ids");
  }
  cfg.raw["dflash_config"] = dflash_config;
  return cfg;
}

// SPEC-DFLASH2 W1 (#1314): the draft's declared architectures, read off its own
// config.json, or an EMPTY list when there is nothing to read. A `.gguf` draft
// (ResolveDflashDraftDir hands back the file itself) and an uncached HF repo id
// both land there, and both already have their own precise error further down the
// load; classifying them here would replace "draft checkpoint not found" with a
// classification failure. An absent or malformed `architectures` key is the same
// case: this engine does not classify on the ABSENCE of evidence.
std::vector<std::string> ReadDflashDraftArchitectures(const std::string& path) {
  std::vector<std::string> architectures;
  std::error_code ec;
  const fs::path cfg = fs::path(ResolveDflashDraftDir(path)) / "config.json";
  if (!fs::exists(cfg, ec)) return architectures;
  std::ifstream f(cfg.string());
  nlohmann::json doc;
  try {
    f >> doc;
  } catch (const nlohmann::json::exception&) {
    return architectures;  // LoadDflashDraft parses it again and reports this
  }
  if (!doc.is_object()) return architectures;
  if (doc.contains("architectures") && doc.at("architectures").is_array()) {
    for (const nlohmann::json& a : doc.at("architectures")) {
      if (a.is_string()) architectures.push_back(a.get<std::string>());
    }
  }
  return architectures;
}

// Classify a DFlash2 draft BY NAME, before any weight is read, and refuse the arm
// that is still missing.
//
// Upstream selects a different model class and a different speculator on the
// `DFlash2DraftModel` architecture (registry.py:628 and
// v1/worker/gpu/spec_decode/__init__.py:12-17 @ vllm-project/vllm#52816 head
// `19c9351904df4c63042671bc67a866ca48dc7d6f`). This engine selects the draft lane
// from the CLI method string alone, and a DFlash2 checkpoint's tensor set is
// DFlash1's PLUS the conv and selector tensors -- so without this classification
// it loads through the DFlash1 loader with nothing missing and nothing thrown,
// and drafts with both new mechanisms simply absent. That draft proposes worse
// tokens, the verify is lossless, so the emitted tokens are still the target's
// and only acceptance falls.
//
// SPEC-DFLASH2 W2 (#1314) SPLITS the two container arms, because they are no
// longer in the same state:
//
//  * SAFETENSORS is ADMITTED. Its grouped dynamic depthwise convolution is
//    implemented (`vt::DFlashGroupedConv`), loaded (`LoadQwen3DFlash` reads the
//    per-layer `attention_conv`/`mlp_conv` tensors) and RUN (every layer body of
//    `Qwen3DFlashModel`), and as of W3 so is its CANDIDATE SELECTOR
//    (`vllm::v1::Dflash2SelectCandidates`). What is still missing is the PATH
//    WALK, and that is refused BY NAME one step later, after both have executed,
//    at `RefuseDflash2PathWalk`. Refusing here instead would leave every
//    line of W2 and W3 unreachable from any production entry point -- AGENTS.md
//    `## Nothing lands dead`. The notice below is what a user gets at STARTUP so
//    the later refusal is not a surprise; it is a notice and not a warning about
//    a degraded result, because there is no degraded result: the engine refuses.
//  * GGUF was REFUSED through W4, because the GGUF drafter ARM was wave W5:
//    neither the config reader nor the weight path had a name for a conv tensor,
//    so admitting the file would have loaded a DFlash1 draft out of a DFlash2
//    checkpoint -- the exact silent degradation this function exists to prevent.
//    W5 (#1314) LANDS that arm, so the refusal is gone and this container gets
//    the same startup notice the other one does. Both arms now print and neither
//    throws; the function stays because the notice is what tells a user the port
//    is beyond the parity pin and carries no throughput number, which is not
//    readable off any checkpoint.
void CheckDflash2DraftArm(const std::string& draft_model_path) {
  // WHAT IDENTIFIED THE FILE, which differs by container and is quoted back to
  // the user because the two arms are otherwise indistinguishable in a message.
  std::string identity;
  const std::string resolved = ResolveDflashDraftDir(draft_model_path);
  if (IsDflashGgufDraft(resolved)) {
    // The GGUF arm. A GGUF carries no `architectures` array, and the published
    // DFlash2 drafter declares `general.architecture = "dflash"` -- the SAME
    // string a DFlash1 drafter writes -- so the architecture cannot separate
    // them and the file would load through the DFlash1 lane. The discriminator
    // is the DFlash2-only metadata (`IsDflash2Gguf`). A file this cannot open or
    // parse is not classified: `LoadDflashDraft` opens it again and owns that
    // error.
    std::string matched;
    try {
      const vllm::GgufFile g = vllm::GgufFile::Open(resolved);
      if (!vllm::IsDflash2Gguf(g, &matched)) return;
    } catch (const std::exception&) {
      return;
    }
    identity = "carries the DFlash2-only metadata key \"" + matched +
               "\" (a GGUF declares no architecture this could read: the "
               "published DFlash2 drafter writes the same \"dflash\" a DFlash1 "
               "drafter writes)";
  } else {
    // The safetensors arm, ADMITTED as of W2 and drafting as of W4. This is the
    // container upstream classifies on, and the only one that HAS an
    // architecture string to classify with.
    const std::vector<std::string> architectures =
        ReadDflashDraftArchitectures(draft_model_path);
    if (!vllm::SpeculativeConfig::IsDflash2Draft(architectures)) return;
    identity = "declares architecture \"DFlash2DraftModel\"";
  }
  // W4 (#1314) removed the boundary on the safetensors arm; W5 removes it on the
  // GGUF one, so this notice is now the function's WHOLE output and both
  // containers reach it. It still prints, because three things a user cannot
  // read off the checkpoint are true -- the port is beyond the parity pin, no
  // throughput number has been taken for it, and the GGUF arm dequantizes a
  // quantized drafter to bf16 -- and because a notice that vanished the moment
  // the lane started working would leave a DFlash2 run indistinguishable from a
  // DFlash1 one in the log.
  std::cerr
      << "vllm.cpp: the draft checkpoint at \"" << draft_model_path << "\" "
      << identity
      << ". Its grouped dynamic depthwise convolution, its CANDIDATE SELECTOR "
         "and its PATH WALK are all implemented and will run, so this draft "
         "DRAFTS -- from safetensors and from GGUF alike (row SPEC-DFLASH2 waves "
         "W1-W5, .agents/specs/dflash2-spec-decode.md, issue #1314). Two things "
         "are still owed and neither is silent: a GGUF drafter is DEQUANTIZED "
         "wholesale to bf16 at load, so a k-quant draft costs its bf16 residency "
         "rather than its file size, and no throughput number has been taken for "
         "this architecture -- a DFlash2 draft runs its block forward off the "
         "paged CUDA-graph fast path, because the candidate selector needs the "
         "hidden states of the same forward its logits came from. This port "
         "mirrors vllm-project/vllm#52816, which MERGED upstream on 2026-08-21 "
         "at head 3406ec1dae9916f920b90f0dbf90dcf54923d042, merge commit "
         "b389ac29465b33f9e9c534df221ea3c129e9793f. It does not advance the "
         "parity pin, and the port is not yet reconciled onto that merged head "
         "(issue #1561).\n";
}

// SPEC-DSPARK-QWEN3-ROUTING (#1193): the two keys upstream classifies a DSpark
// draft by — `architectures` and `model_type` (speculative.py:882-887 and
// :934-944 @ 555967922, plus vllm-project/vllm#52197). Nothing else is read.
struct DsparkDraftIdentity {
  std::vector<std::string> architectures;
  std::string model_type;
};

// Read them off the draft's config.json, or nullopt when there is no config.json
// to read. A GGUF draft (ResolveDflashDraftDir hands back the .gguf file itself)
// and an HF repo id that is not in the local cache both land there, and both
// already have their own precise error further down the load; refusing them HERE
// would replace "draft checkpoint not found" with a classification failure.
std::optional<DsparkDraftIdentity> ReadDsparkDraftIdentity(const std::string& path) {
  std::error_code ec;
  const fs::path cfg = fs::path(ResolveDflashDraftDir(path)) / "config.json";
  if (!fs::exists(cfg, ec)) return std::nullopt;
  std::ifstream f(cfg.string());
  nlohmann::json doc;
  try {
    f >> doc;
  } catch (const nlohmann::json::exception&) {
    return std::nullopt;  // LoadDsparkDraft parses it again and reports this
  }
  if (!doc.is_object()) return std::nullopt;
  // Classify the SAME document LoadDsparkDraft will load: the speculators layout
  // carries no top-level `architectures`, and its translation writes
  // ["Qwen3DSparkModel"] (qwen3_dspark.cpp, update_dspark).
  if (vllm::Qwen3DSparkModel::IsSpeculatorsDsparkConfig(doc)) {
    doc = vllm::Qwen3DSparkModel::TranslateSpeculatorsDsparkConfig(doc);
  }
  DsparkDraftIdentity id;
  if (doc.contains("architectures") && doc.at("architectures").is_array()) {
    for (const nlohmann::json& a : doc.at("architectures")) {
      if (a.is_string()) id.architectures.push_back(a.get<std::string>());
    }
  }
  // A config that DECLARES no architecture is not classified at all. Upstream
  // reads the key off a HuggingFace `ModelConfig`, where an absent key is `[]`,
  // and its catch-all would send that empty list to DeepSeek-V4. Refusing on it
  // here would refuse a draft on the ABSENCE of evidence, and the native
  // `deepseek-ai/dspark_qwen3_*_block7` layouts have not been read on this host
  // to confirm they declare it. The narrowing is deliberate and is recorded
  // under `## Owed` in .agents/specs/dspark-qwen3-routing.md.
  if (id.architectures.empty()) return std::nullopt;
  if (doc.contains("model_type") && doc.at("model_type").is_string()) {
    id.model_type = doc.at("model_type").get<std::string>();
  }
  return id;
}

// The two DSpark resolution keys, read off the draft checkpoint's own
// config.json (SPEC-DSPARK-BLOCK-SIZE-GUARD, #1225).
struct DsparkDraftKeys {
  std::optional<int> n_predict = std::nullopt;
  std::optional<int> block_floor = std::nullopt;
  // The key `block_floor` was actually read from, so the refusal can name it.
  // Upstream's `dspark_block_size` unless the fallback below supplied it, which
  // on both published Qwen3 drafts is always.
  const char* block_floor_key = "dspark_block_size";
};

// Mirror of the getattr() reads upstream performs on the draft's hf_config
// before it resolves k, all @ 555967922:
//
//   * n_predict                                                :973-975
//   * the Gemma4 normalization n_predict = block_size          :957-961
//     (guarded by "Gemma4DSparkModel" in architectures, exactly as upstream
//     guards it -- it does NOT apply to a Qwen3 DSpark draft)
//   * dspark_block_size, the block floor                       :1011-1015
//
// ONE DIVERGENCE, argued in .agents/specs/dspark-block-size-guard.md section 2:
// when `dspark_block_size` is absent the floor falls back to `block_size`.
// Upstream reads only `dspark_block_size`, and that identifier occurs in no file
// of the pinned checkout except speculative.py, so it can only arrive from a
// draft config.json -- and NEITHER published Qwen3 draft carries it.
// deepseek-ai/dspark_qwen3_4b_block7 and RadixArk/Qwen3.8-27B-DSpark @ 85ef153b
// both ship `block_size: 7` with no n_predict, and the :957-961 normalization is
// Gemma4-only, so upstream accepts k=6 against a block-7 Qwen3 draft. A literal
// port would key the floor on a field no checkpoint we support sets. Our draft
// block is sized by k alone (spec_decode/dspark/speculator.h:56) and no weight
// is block-shaped, so a short k raises no shape error: it drafts a structurally
// wrong block in silence. The explicit key still wins when a checkpoint does
// carry it, so a later pin that adds it changes nothing here.
//
// Both values stay std::nullopt when the draft checkpoint is not on disk. That
// keeps ResolveSpecConfig resolving a path it cannot read exactly as it did
// before this change; LoadDsparkDraft owns the "not found" message and names the
// directory it looked in.
DsparkDraftKeys ReadDsparkDraftKeys(const std::optional<std::string>& draft_model_path) {
  DsparkDraftKeys keys;
  if (!draft_model_path.has_value()) return keys;
  const std::string draft_dir = ResolveDflashDraftDir(*draft_model_path);
  std::error_code ec;
  const fs::path config_path = fs::path(draft_dir) / "config.json";
  if (!fs::exists(config_path, ec)) return keys;

  nlohmann::json cj;
  try {
    std::ifstream cf(config_path.string());
    cf >> cj;
  } catch (const std::exception&) {
    return keys;  // LoadDsparkDraft re-reads it and reports the parse failure.
  }
  if (!cj.is_object()) return keys;
  // Read through the SAME normalized shape LoadDsparkDraft loads from, so both
  // published config layouts resolve identically.
  if (vllm::Qwen3DSparkModel::IsSpeculatorsDsparkConfig(cj)) {
    cj = vllm::Qwen3DSparkModel::TranslateSpeculatorsDsparkConfig(cj);
  }

  const auto read_int = [&cj](const char* key) -> std::optional<int> {
    if (cj.contains(key) && cj.at(key).is_number_integer()) {
      return cj.at(key).get<int>();
    }
    return std::nullopt;
  };

  keys.n_predict = read_int("n_predict");
  if (!keys.n_predict.has_value() && cj.contains("architectures") &&
      cj.at("architectures").is_array()) {
    for (const auto& arch : cj.at("architectures")) {
      if (arch.is_string() && arch.get<std::string>() == "Gemma4DSparkModel") {
        keys.n_predict = read_int("block_size");  // speculative.py:957-961
        break;
      }
    }
  }

  keys.block_floor = read_int("dspark_block_size");
  if (!keys.block_floor.has_value()) {
    keys.block_floor = read_int("block_size");  // the divergence, above
    keys.block_floor_key = "block_size";
  }
  return keys;
}

// Load a DSpark draft: the DFlash backbone plus the Markov head plus, for a
// reduced draft vocab, the d2t map (SPEC-DSPARK W5). Both published config
// layouts are accepted; the tensor layout is identical between them.
std::unique_ptr<DflashDraft> LoadDsparkDraft(const vllm::SpeculativeConfig& spec,
                                             const SharedHeadSource& shared) {
  if (spec.method != "dspark") return nullptr;
  if (!spec.draft_model_path.has_value()) {
    throw std::runtime_error("dspark: resolved config missing draft_model_path");
  }
  const std::string draft_dir = ResolveDflashDraftDir(*spec.draft_model_path);
  std::error_code ec;
  if (!fs::exists(fs::path(draft_dir) / "config.json", ec)) {
    throw std::runtime_error("dspark: draft checkpoint not found at " + draft_dir +
                             " (from \"" + *spec.draft_model_path + "\")");
  }
  std::ifstream cf((fs::path(draft_dir) / "config.json").string());
  nlohmann::json cj;
  cf >> cj;
  // Speculators format -> the native shape the rest of the path expects.
  if (vllm::Qwen3DSparkModel::IsSpeculatorsDsparkConfig(cj)) {
    cj = vllm::Qwen3DSparkModel::TranslateSpeculatorsDsparkConfig(cj);
  }

  auto draft = std::make_unique<DflashDraft>();
  draft->k = spec.ResolvedNumSpeculativeTokens();
  draft->config = MakeDsparkDraftConfig(cj);
  // Native Qwen3 DSpark configs default to sampling from the anchor
  // (dspark/speculator.py:50-52 getattr(..., True)); the Speculators translation
  // has already written the key explicitly with its own FALSE default.
  draft->sample_from_anchor =
      !cj.contains("sample_from_anchor") || cj.at("sample_from_anchor").get<bool>();

  const nlohmann::json& dcfg = draft->config.raw.at("dflash_config");
  if (!dcfg.contains("target_layer_ids") || !dcfg.contains("mask_token_id")) {
    throw std::runtime_error(
        "dspark: the draft config must carry target_layer_ids and mask_token_id");
  }
  const int64_t num_taps = static_cast<int64_t>(dcfg.at("target_layer_ids").size());
  const int32_t mask_id = dcfg.at("mask_token_id").get<int32_t>();

  std::vector<vllm::SafetensorsFile> dshards = LoadShards(draft_dir);
  draft->dspark = std::make_unique<vllm::Qwen3DSparkWeights>(
      vllm::LoadQwen3DSpark(dshards, draft->config, num_taps, mask_id));

  // A DSpark checkpoint usually SHIPS embed_tokens + lm_head (both published
  // families do), unlike the z-lab DFlash draft. Share the target's ONLY when the
  // draft omits them, which is what load_dspark_model's _should_share decides
  // (dspark/utils.py:56-73) -- overwriting a shipped head would silently swap the
  // draft's own (possibly reduced-vocab) output layer for the target's.
  if (draft->dspark->backbone.embed_tokens.Empty() ||
      draft->dspark->backbone.lm_head.Empty()) {
    vllm::OwnedTensor shared_embed;
    vllm::OwnedTensor shared_lm_head;
    // ALL THREE nullptr, and NONE a default. The DSpark lane has no DFlash2
    // selector, so no guard reads a dequantized-head flag here; and its backbone
    // holds ONE bf16 `lm_head` owner, so there is nowhere to put a packed head
    // of EITHER kind — NVFP4 or EXL3 trellis. A quantized target head therefore
    // still refuses at `LoadDflashSharedLmHead`, by name and at startup. All
    // three are NAMED rather than omitted so none can be deleted at the DFlash
    // call site without a build failure. Owed:
    // .agents/specs/dflash2-spec-decode.md `## Owed` O26, issue #1628; the
    // trellis owner is owed by .agents/specs/model-qwen35-exl3-head.md, #2495.
    shared.LoadInto(&shared_embed, &shared_lm_head, /*head_was_quantized=*/nullptr,
                    /*head_fp4=*/nullptr, /*head_exl3=*/nullptr);
    if (draft->dspark->backbone.embed_tokens.Empty()) {
      draft->dspark->backbone.embed_tokens = std::move(shared_embed);
    }
    if (draft->dspark->backbone.lm_head.Empty()) {
      draft->dspark->backbone.lm_head = std::move(shared_lm_head);
      draft->dspark->backbone.draft_vocab_size =
          draft->dspark->backbone.lm_head.shape[0];
      draft->dspark->draft_vocab_size = draft->dspark->backbone.draft_vocab_size;
    }
  }
  return draft;
}

// Load the whole DFlash draft (layer weights + fc + norms from the draft
// checkpoint, safetensors dir or `dflash`-arch GGUF; embed_tokens + lm_head
// SHARED bf16 from the TARGET via `shared`) plus the resolved draft config + k.
// The source `shared` points at must still be alive (both are inside
// FromModelDir). Returns null when the config carries no dflash draft path.
std::unique_ptr<DflashDraft> LoadDflashDraft(
    const vllm::SpeculativeConfig& spec, const SharedHeadSource& shared) {
  if (spec.method != "dflash") return nullptr;
  if (!spec.draft_model_path.has_value()) {
    throw std::runtime_error("dflash: resolved config missing draft_model_path");
  }
  const std::string draft_dir = ResolveDflashDraftDir(*spec.draft_model_path);
  std::error_code ec;
  // GGUF draft (SPEC-DFLASH-GGUF axis A): config + weights both come out of the
  // single file. The target-shared embed/lm_head still come from *target_shards*
  // below, exactly as for a safetensors draft - the GGUF DFLASH arch omits
  // token_embd/output precisely because the draft shares the target's.
  auto draft = std::make_unique<DflashDraft>();
  draft->k = spec.ResolvedNumSpeculativeTokens();
  int64_t num_taps = 0;
  int32_t mask_id = -1;
  const char* source_kind = nullptr;
  if (IsDflashGgufDraft(draft_dir)) {
    vllm::GgufFile dg = vllm::GgufFile::Open(draft_dir);
    draft->config = vllm::MakeDflashGgufConfig(dg);
    const nlohmann::json& dcfg = draft->config.raw.at("dflash_config");
    num_taps = static_cast<int64_t>(dcfg.at("target_layer_ids").size());
    mask_id = dcfg.at("mask_token_id").get<int32_t>();
    draft->weights =
        vllm::LoadQwen3DFlashFromGguf(dg, draft->config, num_taps, mask_id);
    source_kind = "GGUF";
  } else {
    if (!fs::exists(fs::path(draft_dir) / "config.json", ec)) {
      throw std::runtime_error("dflash: draft checkpoint not found at " +
                               draft_dir + " (from \"" +
                               *spec.draft_model_path + "\")");
    }
    std::ifstream cf((fs::path(draft_dir) / "config.json").string());
    nlohmann::json cj;
    cf >> cj;
    draft->config = vllm::MakeQwen3DFlashDraftConfig(cj);
    num_taps = static_cast<int64_t>(
        cj.at("dflash_config").at("target_layer_ids").size());
    mask_id = cj.at("dflash_config").at("mask_token_id").get<int32_t>();
    std::vector<vllm::SafetensorsFile> dshards = LoadShards(draft_dir);
    draft->weights =
        vllm::LoadQwen3DFlash(dshards, draft->config, num_taps, mask_id);
    source_kind = "safetensors";
  }

  // The draft SHARES the target's embed_tokens + lm_head, exactly as vLLM's
  // skip_substrs(embed_tokens)/tie handling. Common to BOTH draft sources and
  // BOTH target containers since B1 - the source abstraction is what lets the
  // four combinations share one code path.
  //
  // SPEC-DFLASH2-QUANT-LMHEAD (#1628) corrects what this comment used to claim,
  // which was that the head is "bf16 in both containers of the NVFP4 27B". It is
  // not: `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` stores `lm_head.weight` as
  // ModelOpt NVFP4 (U8), and the read here refused it by stored dtype. It is now
  // taken PACKED into `lm_head_fp4`, which is what the target itself computes
  // with, so the DFlash2 selector reads the target head's exact top-K rather
  // than a widened head's - the state D12 refuses and the state it admits are
  // different, and the dtype cannot tell them apart.
  shared.LoadInto(&draft->weights.embed_tokens, &draft->weights.lm_head,
                  &draft->weights.lm_head_dequantized,
                  &draft->weights.lm_head_fp4, &draft->weights.lm_head_exl3);
  // MODEL-QWEN35-EXL3-HEAD (#2495 item 6): the vocab comes from whichever of the
  // THREE owners the read filled, and never from a declared number. The trellis
  // arm reports it as `OutFeatures()` — the trellis geometry, `n/16` output
  // tiles of 16 — which is the same rule `Exl3Weight` applies to `bits`: the
  // tensor is the authority and the config scalar is not.
  draft->weights.draft_vocab_size =
      !draft->weights.lm_head_exl3.Empty()
          ? draft->weights.lm_head_exl3.OutFeatures()
          : (draft->weights.lm_head_fp4.Empty()
                 ? draft->weights.lm_head.shape[0]
                 : draft->weights.lm_head_fp4.n);
  // A DFLASH GGUF draft carries NO vocab KV and no embedding tensor (it SHARES
  // the target's), so MakeDflashGgufConfig leaves vocab_size 0 - right for the
  // config, fatal for the forward: the draft sizes its embedding lookup as
  // `{config.vocab_size, H}` (qwen3_dflash.cpp:245,477,1008,1043), so 0 is an
  // EMPTY table and the first propose throws "cuda embedding: empty table
  // (vocab 0) with nonempty ids". A safetensors draft never hit this because
  // its config.json declares vocab_size, which is why the condition is on the
  // VALUE and not on the draft source. Take the row count from the TARGET
  // tensor actually being indexed rather than from any declared number, so the
  // view and the buffer cannot disagree. Found by SPEC-DFLASH-GGUF GD4, the
  // first run that ever GENERATED through a GGUF-sourced DFlash draft; still
  // the rule at B1, where the rows now come from a GGUF target's token_embd.
  if (draft->config.vocab_size == 0) {
    draft->config.vocab_size = draft->weights.embed_tokens.shape[0];
  }
  // SPEC-DFLASH2 W2 (#1314): the conv's block is the QUERY block, and the
  // resolved `k` is its authority -- exactly as upstream sizes the conv from
  // `1 + speculative_config.num_speculative_tokens` and never from the config key
  // (`DFlash2Qwen3DecoderLayer.__init__` @ vllm-project/vllm#52816 head
  // `66e5414c6d75a8529473d977f7458c140bbab8a0`). A conv masking against the wrong
  // block is acceptance-only and token-invisible, so it is set from ONE place.
  //
  // W4 (#1314) made that one place the ONLY place: `LoadQwen3DFlash` no longer
  // seeds the field from the checkpoint's `block_size`, so deleting this line
  // leaves 0 rather than a plausible default and the first DFlash2 forward
  // refuses by name (`Qwen3DFlashModel`'s `conv_block_size must be set` check).
  // That is the discharge of spec `## Owed` O5's first item: the line was
  // mutation-proven UNGATED by W2 -- deleting it compiled clean and left every
  // suite green -- and it is now covered by the production reachability gate,
  // tests/vllm/v1/spec_decode/test_dflash2_runner_reach.cpp, which generates
  // through this loader path.
  if (draft->weights.IsDflash2()) {
    draft->weights.conv_block_size = draft->k + 1;
    // GEOMETRY ONLY, and deliberately NOT the boundary. This line used to append
    // "the candidate selector is NOT implemented", which W3 made false in the
    // same wave that wrote it: the selector landed and the boundary moved to the
    // path walk. A user loading a real DFlash2 directory then got that sentence
    // AND `CheckDflash2DraftArm`'s corrected notice, which contradict each other,
    // and the stale one named the wave that had just shipped as still owing the
    // mechanism.
    //
    // The boundary has exactly ONE owner, for the reason `FromModelDir` already
    // states beside its own call: `CheckDflash2DraftArm` runs on every path that
    // reaches this function -- `ResolveSpecConfig` for the constructor and
    // `FromModelDir` line ~1889 ahead of every load -- so its notice has already
    // been printed by the time anything gets here. A second copy adds nothing a
    // user can act on and cannot be gated from any entry point this repository
    // can drive (spec `## Owed` O5), so it went stale within one wave and would
    // again at W4 and W5. What this line uniquely knows is the RESOLVED conv
    // geometry, which the notice above cannot report because it runs before any
    // weight is read, so that is all it says.
    std::cerr << "vllm.cpp: DFlash2 draft: grouped conv taps="
              << draft->weights.conv_taps
              << " group=" << draft->weights.conv_group_size
              << " block=" << draft->weights.conv_block_size << "\n";
  }
  std::cerr << "vllm.cpp: DFlash draft loaded from " << source_kind << " "
            << draft_dir << " (k=" << draft->k << ", taps=" << num_taps
            << ", mask=" << mask_id << ", vocab=" << draft->config.vocab_size
            << ", shared head from " << shared.Describe() << ")\n";
  return draft;
}

// KV-EXTERNAL-CACHE (LMCache): build the external KV connector selected by
// EngineParams::kv_transfer_config, injecting the runner's resolved
// full-attention KV geometry into its extra_config so the connector's KV_2LTD
// chunk layout matches the physical KV page exactly. Returns nullptr when no
// connector is configured (the default-off inert path). Mirrors vLLM building
// the connector from --kv-transfer-config after the KV caches exist.
std::unique_ptr<vllm::v1::kv_offload::KVConnector> BuildKvConnector(
    const EngineParams& params, const vllm::v1::GPUModelRunner& runner) {
  using namespace vllm::v1::kv_offload;
  if (!params.kv_transfer_config.has_value()) return nullptr;
  vllm::KVTransferConfig cfg = *params.kv_transfer_config;
  if (!cfg.kv_connector.has_value() || cfg.kv_connector->empty()) return nullptr;
  // kv_role is required whenever kv_connector is set; default to kBoth so a
  // caller that only names the connector still yields a valid config.
  if (!cfg.kv_role.has_value()) cfg.kv_role = vllm::KVRole::kBoth;

  // D1 SAFETY GUARD, before anything is constructed. A connector's scheduler
  // half shortcuts prefill for externally matched blocks; if its worker half
  // cannot write those bytes into THIS device's KV pages, the model would
  // attend over never-written KV and emit plausible, wrong output with no error
  // anywhere. Refuse instead. The predicate is registered per connector
  // (KVConnectorWorkerTransferFn), so admitting a future worker half is one
  // override plus one registration argument — no ladder to extend here.
  // Checked BEFORE Create() on purpose: a connector ctor can fail first for an
  // unrelated precondition, and a refusal that surfaces as somebody else's
  // error message is not actionable.
  EnsureWorkerTransferSupported(*cfg.kv_connector, runner.device().type);

  const std::vector<vllm::PagedKvCache>& kv = runner.attn_kv();
  if (kv.empty()) {
    throw std::runtime_error(
        "LMCache connector requires a full-attention KV group, but this model "
        "has none");
  }
  const int num_layers = static_cast<int>(kv.size());
  const int hidden_dim =
      static_cast<int>(kv[0].num_kv_heads * kv[0].head_size);
  const int fa_block = static_cast<int>(kv[0].block_size);
  // Inject the geometry (extra_config overrides win in CreateFromConfig). The
  // connector keys block-aligned (chunk_tokens == the full-attention block).
  cfg.kv_connector_extra_config["num_layers"] = std::to_string(num_layers);
  cfg.kv_connector_extra_config["hidden_dim"] = std::to_string(hidden_dim);
  cfg.kv_connector_extra_config["chunk_tokens"] = std::to_string(fa_block);

  KVConnectorContext ctx;
  ctx.config = &cfg;
  ctx.role = KVConnectorRole::kScheduler;
  ctx.block_size = fa_block;
  return KVConnectorFactory::Create(ctx);
}

// The GGUF architectures this build dispatches, keyed by llama.cpp's
// `general.architecture`, in the order they are tried. ONE table rather than a
// ladder plus a hand-written list: the refusal below names the supported set by
// READING this, so an added arm cannot drift from what a user is told it can
// load. Additive by construction — a new GGUF-loadable arch adds ONE row here
// and owns its config builder in its own TU. Everything downstream
// (Resolve -> tokenizer -> Load) is arch-agnostic.
//
//  * `deepseek4` -> DeepseekV4HfConfigFromGguf, which maps it onto the
//    registered DeepseekV4ForCausalLM.
//  * `muse-glimmer` -> the k-quant arm whose config builder recovers the query
//    pre-scale from the folded attn_q_norm and the iRoPE mask from
//    sliding_window_pattern (muse_glimmer_gguf_weights.h).
//  * the three qwen3_5 keys -> HfConfigFromGguf, which owns all three itself
//    (qwen3_5_gguf_weights.cpp).
//  * `qwen4exp` -> Qwen4ExpHfConfigFromGguf. It does NOT reuse HfConfigFromGguf,
//    which asserts its own three architectures by name; a fourth family routed
//    there would refuse as "qwen3_5 gguf: unexpected architecture", which is the
//    #809 defect this table exists to prevent (see the default arm below).
//  * `glm-dsa` -> GlmMoeDsaHfConfigFromGguf. GLM-5.3, and the first row here
//    whose family vLLM DOES implement at the pin -- `registry.py:117` routes
//    `GlmMoeDsaForCausalLM` into `deepseek_v2` -- while our own DeepSeek-V2
//    loader refuses the checkpoint at its `index_topk` tripwire. It is a
//    SEPARATE row rather than a `deepseek2` one for that reason: the tripwire
//    stays a wall for DeepSeek-V2 and this family gets its own config, its own
//    registration and its own refusals. The builder synthesizes an HF-shaped
//    config and hands it to the same `ParseGlmMoeDsaParams` a config.json
//    descends through.
//  * `glm5next` -> Glm5NextHfConfigFromGguf, whose builder synthesizes an
//    HF-shaped config and hands it to the SAME `ParseGlm5NextParams` a
//    config.json descends through, so both sources meet one validator. This is
//    the row that discharges O9: `scripts/convert-glm5-next-gguf.py` is the
//    only writer of that container -- no upstream tool can produce one -- and
//    until this row existed the file it wrote was refused as unrecognized.
//  * `laguna` -> LagunaHfConfigFromGguf. The Laguna registry has carried a GGUF
//    load arm (and `LagunaParamsFromGguf`) since W5, but no row here reached
//    it, so every Laguna GGUF was refused at this table and only
//    `examples/laguna_gen` could load one (#2841). The builder is derived from
//    the loader's own GGUF resolve, so the two cannot disagree on geometry.
struct GgufArchArm {
  const char* arch;
  HfConfig (*build)(const vllm::GgufFile&);
};

constexpr GgufArchArm kGgufArchArms[] = {
    {"deepseek4", &vllm::DeepseekV4HfConfigFromGguf},
    {vllm::kMuseGlimmerGgufArch, &vllm::MuseGlimmerHfConfigFromGguf},
    {"qwen35", &vllm::HfConfigFromGguf},
    {"qwen35moe", &vllm::HfConfigFromGguf},
    {"qwen3next", &vllm::HfConfigFromGguf},
    {"qwen3", &vllm::Qwen3HfConfigFromGguf},
    {vllm::kQwen4ExpGgufArch, &vllm::Qwen4ExpHfConfigFromGguf},
    {vllm::kGlm5NextGgufArch, &vllm::Glm5NextHfConfigFromGguf},
    {vllm::kGlmMoeDsaGgufArch, &vllm::GlmMoeDsaHfConfigFromGguf},
    {"laguna", &vllm::LagunaHfConfigFromGguf},
};

std::string SupportedGgufArchitectures() {
  std::string list;
  for (const GgufArchArm& arm : kGgufArchArms) {
    if (!list.empty()) list += ", ";
    list += arm.arch;
  }
  return list;
}

// Top-level GGUF architecture dispatch: `general.architecture` selects the
// family's HfConfig builder.
//
// The default is EXPLICIT and refuses by name. It used to fall through to
// `vllm::HfConfigFromGguf`, which is qwen3_5's builder and hard-asserts its own
// three keys — so every unsupported architecture, `nemotron_h_moe` included,
// died with "qwen3_5 gguf: unexpected architecture", naming a model that has
// nothing to do with the file the user passed and sending the reader into an
// unrelated translation unit (#809). A refusal that names the wrong model is
// worse than none.
HfConfig HfConfigFromGgufDispatch(const vllm::GgufFile& gguf) {
  const vllm::GgufValue* arch_kv = gguf.FindKv("general.architecture");
  if (arch_kv == nullptr || arch_kv->TypeId() != vllm::kGgufString) {
    throw std::runtime_error(
        "GGUF: this file carries no string `general.architecture` key, so no "
        "architecture can be selected. GGUF architectures supported by this "
        "build: " +
        SupportedGgufArchitectures());
  }
  const std::string arch = std::get<std::string>(arch_kv->v);
  for (const GgufArchArm& arm : kGgufArchArms) {
    if (arch == arm.arch) return arm.build(gguf);
  }
  // KNOWN architectures whose GGUF arm is OWED, not absent. Each refuses with
  // the message its OWN model writes, so the reader lands in the translation
  // unit that owes the work and on the spec section that tracks it. Without
  // this the arm below would refuse them as merely unrecognized, which
  // understates them: the file IS one this project knows.
  if (vllm::IsNemotronHGguf(gguf)) {
    throw std::runtime_error(vllm::NemotronHGgufRefusal());
  }
  if (vllm::IsDots3NoteGguf(gguf)) {
    throw std::runtime_error(vllm::Dots3NoteGgufRefusal());
  }
  // DeepSeek-V4.1: this build RESOLVES the architecture and VALIDATES its
  // config.json (W1), so a `deepseek41` container is a file this project knows
  // about and the generic "unrecognized architecture" below would understate
  // it. The GGUF arm itself is OWED to W8.
  if (vllm::IsDeepseekV41Gguf(gguf)) {
    throw std::runtime_error(vllm::DeepseekV41GgufRefusal());
  }
  throw std::runtime_error(
      "GGUF architecture '" + arch +
      "' is not supported by this build. GGUF architectures supported by this "
      "build: " +
      SupportedGgufArchitectures());
}

}  // namespace

// Resolve the per-step token budget (max_num_batched_tokens) for chunked
// prefill. An explicit EngineParams override wins; otherwise a PER-ARCH bounded
// default that does NOT scale with max_num_seqs, so a long/many-request prefill
// is split across steps and the per-step GDN chunked-scan activation stays
// bounded regardless of concurrency (the 27B 8x1024 conc-8 OOM fix — the old
// max_model_len*max_num_seqs product ran the whole 8192-token prefill in one
// step).
//
//  * DENSE arch (27B W4A4): 2048 FLAT — mirrors vLLM's own scheduler default
//    (DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048, vllm/config/scheduler.py:42 @
//    e24d1b24). The dense prefill is expensive per token: at mnbt=8192 one
//    giant mixed step runs several full prompts' prefill eagerly and every
//    decode stream stalls behind it (TTFT ~2x, decode starved). MEASURED (27B
//    NVFP4, GB10, in1024/out128): conc32/np96 mnbt=2048 999.16 tok/s vs 8192
//    895.90 (+11.5%); the conc16/conc32 default-vs-8192 A/Bs are in
//    .agents/parity-ledger.md (2026-07-10).
//  * MoE arch (35B A3B W4A16): keep the GB10-tuned CONCURRENCY-AWARE budget —
//    8192 at high concurrency (>=32), else 4096. Its cheap A3B expert prefill
//    wants the bigger chunk: at the 35B gate (conc-64, in1024/out128)
//    mnbt=8192 is +2.7% over 4096 (itself +8.2% over 2048) — bigger prefill
//    chunks amortize per-token GEMM/attention work across the many running
//    seqs. At LOW concurrency 8192 loses pipelining, so those keep 4096.
//    Memory-safe on GB10's 119GB (35B conc-64 peak 54GB; 16384 OOMs).
//
// Invariants mirrored from SchedulerConfig.verify_max_model_len
// (vllm/config/scheduler.py:87): the budget must be >= max_num_seqs (every
// running seq needs at least one token/step). For tiny models whose whole
// workload is smaller than the default we keep the old
// (max_model_len*max_num_seqs) ceiling so no behavior changes there.
int LoadedEngine::ResolveMaxNumBatchedTokens(const EngineParams& params,
                                             int max_model_len,
                                             bool is_dense_arch) {
  const int seqs = params.max_num_seqs > 0 ? params.max_num_seqs : 8;
  if (params.max_num_batched_tokens > 0) {
    // Explicit override; still honor the >= max_num_seqs invariant.
    return std::max(params.max_num_batched_tokens, seqs);
  }
  int budget = is_dense_arch ? 2048 : (seqs >= 32 ? 8192 : 4096);
  // Never exceed the whole workload's ceiling (tiny-model no-op preservation).
  const long ceiling = static_cast<long>(max_model_len) * seqs;
  if (ceiling > 0 && static_cast<long>(budget) > ceiling) {
    budget = static_cast<int>(ceiling);
  }
  return std::max(budget, seqs);
}

bool LoadedEngine::ResolveEnablePrefixCaching(const EngineParams& params,
                                               const ModelInfo& model_info) {
  if (params.enable_prefix_caching.has_value()) {
    return *params.enable_prefix_caching;
  }
  // ModelConfig.is_prefix_caching_supported at the parity pin: generative
  // hybrid and attention-free models default OFF while ordinary decoder-only
  // models default ON. ModelInfo.has_inner_state covers the attention-free
  // family in the native registry; is_hybrid covers Qwen3.5/3.6 GDN.
  return !model_info.is_hybrid && !model_info.has_inner_state;
}

// ARCH-ONE-SURFACE ROW 8: the explicit arms of the device-selection policy —
// see the contract in model_loader.h. Ported semantics:
// vllm/config/device.py:61-66 @ 555967922 (an explicit device string is
// assigned VERBATIM — never substituted), with the loud failure upstream
// raises when the named device cannot serve (torch/worker init on an absent
// CUDA device; our analogue is the unregistered kCUDA platform,
// src/vllm/platforms/cuda.cpp Registrar — kCUDA registers only when a usable
// GPU probed).
vt::DeviceType LoadedEngine::ResolveExplicitDeviceType(
    vllm::Device requested,
    std::optional<vt::DeviceType> named_platform_type) {
  switch (requested) {
    case vllm::Device::kCPU:
      // Explicit CPU never consults the accelerator probe: even on a
      // CUDA-capable build/process this selects the CPU queue.
      return vt::DeviceType::kCPU;
    case vllm::Device::kNamedPlatform:
      if (!named_platform_type.has_value()) {
        throw std::runtime_error(
            "device 'cuda' was requested but no CUDA platform is available in "
            "this build/process (an explicitly named device is never silently "
            "replaced — mirror of vllm/config/device.py:61-66; use device=auto "
            "or device=cpu, or run a CUDA build on a machine with a usable "
            "GPU)");
      }
      return *named_platform_type;
    case vllm::Device::kAuto:
      break;  // auto resolves through the probe in SelectQueue, not here.
  }
  throw std::invalid_argument(
      "ResolveExplicitDeviceType resolves only explicit device selections "
      "(cpu/cuda); auto resolves through the accelerator-first probe");
}

bool LoadedEngine::EnsureNoneHash() {
  // Idempotent: init_none_hash just (re)assigns the NONE_HASH global.
  //
  // The seed is resolved INSIDE init_none_hash (explicit >
  // $VLLM_PREFIX_CACHING_HASH_SEED > $PYTHONHASHSEED > the fixed built-in
  // default), so block hashes are DETERMINISTIC ACROSS PROCESSES by default.
  // The previous comment here claimed the unseeded value "does not affect
  // determinism" because prefix caching is inert below one block; that is true
  // only for the sub-block case and does not generalise — every full block
  // hash chains from NONE_HASH, so a per-process-random seed makes any
  // content-addressed persisted cache score 0% on restart.
  vllm::v1::init_none_hash(vllm::v1::sha256_cbor);
  return true;
}

namespace {
bool NativeQwen3_5ImagesEnabled(const HfConfig& config,
                               std::string_view architecture,
                               const vllm::MultiModalConfig& multimodal) {
  return architecture == "Qwen3_5ForConditionalGeneration" &&
         IsExl3Checkpoint(config) && multimodal.GetLimitPerPrompt("image") > 0;
}
}  // namespace

int ResolveNativeQwen3_5MaxNumSeqs(
    int resolved, const HfConfig& config, std::string_view architecture,
    vt::DeviceType device, const vllm::MultiModalConfig& multimodal) {
  if (device != vt::DeviceType::kXPU ||
      !NativeQwen3_5ImagesEnabled(config, architecture, multimodal)) return resolved;
  if (resolved <= 0)
    throw std::runtime_error("native Qwen3.5 image concurrency must be positive");
  if (resolved > kNativeQwen3_5MaxConcurrentRequests) {
    std::cerr << "INFO native Qwen3.5 image admission: reduced max_num_seqs from "
              << resolved << " to " << kNativeQwen3_5MaxConcurrentRequests
              << ". Extra clients queue within the C1-C4 native image envelope.\n";
    return kNativeQwen3_5MaxConcurrentRequests;
  }
  return resolved;
}

void ConfigureNativeQwen3_5EncoderBudget(
    vllm::SchedulerConfig& scheduler, const HfConfig& config,
    std::string_view architecture, vt::DeviceType device,
    const vllm::MultiModalConfig& multimodal_config) {
  if (device != vt::DeviceType::kXPU ||
      !NativeQwen3_5ImagesEnabled(config, architecture, multimodal_config)) return;
  const auto tower = Qwen3_5DenseVisionConfig(config);
  if (tower.patch_size <= 0 || tower.patch_size > 64 ||
      tower.spatial_merge_size <= 0 || tower.spatial_merge_size > 128 ||
      scheduler.max_num_seqs <= 0) {
    throw std::runtime_error("native Qwen3.5 encoder budget: unsupported geometry or concurrency");
  }
  const int64_t patch_area = int64_t{tower.patch_size} * tower.patch_size;
  const int64_t merge_area = int64_t{tower.spatial_merge_size} * tower.spatial_merge_size;
  const int64_t rows = std::min(multimodal::kNativeQwen3_5MaxImagePixels / patch_area,
                                multimodal::kNativeQwen3_5MaxImagePatches) / merge_area;
  // Two images is the native serving envelope. Reserve all admitted requests'
  // outputs independently of decoder chunking, so one item's bidirectional
  // encoder cannot become permanently unschedulable at a 1600-token budget.
  const int64_t images = std::min(2, multimodal_config.GetLimitPerPrompt("image"));
  const int64_t compute = rows * images;
  const int64_t capacity = compute * scheduler.max_num_seqs;
  if (rows <= 0 || images <= 0 || capacity > std::numeric_limits<int>::max())
    throw std::runtime_error("native Qwen3.5 encoder budget exceeds supported capacity");
  scheduler.max_num_encoder_input_tokens = std::max(
      scheduler.max_num_encoder_input_tokens, static_cast<int>(compute));
  scheduler.encoder_cache_size = std::max(
      scheduler.encoder_cache_size, static_cast<int>(capacity));
}

vllm::SchedulerConfig LoadedEngine::MakeSchedulerConfig(
    int max_model_len, int max_num_seqs, int max_num_batched_tokens,
    vllm::SchedulerPolicy policy) const {
  vllm::SchedulerConfig cfg;
  cfg.max_num_seqs = max_num_seqs;
  // Bounded per-step budget (chunked prefill). See ResolveMaxNumBatchedTokens.
  cfg.max_num_batched_tokens = max_num_batched_tokens;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = max_model_len;
  cfg.watermark = 0.0;
  // Scheduling policy (fcfs default; kPriority selects the priority queue).
  cfg.policy = policy;
  ConfigureNativeQwen3_5EncoderBudget(
      cfg, config_, model_->registration().architecture, runner_.device().type,
      mm_config_);
  return cfg;
}

// The construction-time async-scheduling resolution (W3 enable-flip). Mirrors
// vLLM: async_scheduling=None resolves to True when compatible
// (vllm/config/vllm.py:990-1038) gated on the runner advertising the async
// device path, then the house VT_ASYNC_SCHED=0 rollback env force-disables it in
// the same binary. `async_scheduling` stays nullopt on MakeSchedulerConfig, so
// ResolveAsyncScheduling(runner_supports_async) yields runner_supports_async
// (when otherwise compatible).
bool LoadedEngine::ResolveAsyncEnabled(
    const vllm::SchedulerConfig& scheduler_config, bool runner_supports_async,
    bool is_pooling_model, bool spec_decode_incompatible) {
  // Pooling models resolve async scheduling OFF (the mirror of vLLM disabling
  // it by default for pooling models, vllm/config/vllm.py:1068-1073) — the
  // landed is_pooling_model arm of ResolveAsyncScheduling, wired here since
  // ARCH-ONE-SURFACE ROW 6. false (every text arch) is byte-identical.
  // spec_decode_incompatible (SPEC-DFLASH2 W7, #1824) is the vllm.py:1076-1087
  // arm: a speculative method OUTSIDE the Eagle-type family resolves OFF; an
  // Eagle-type one (mtp/dflash/dspark) passes false and stays ON.
  return vllm::AsyncSchedulingEnabled(scheduler_config.ResolveAsyncScheduling(
      runner_supports_async, is_pooling_model, spec_decode_incompatible));
}

std::pair<int, int> LoadedEngine::ResolveSchedulerBlockSizes(
    const vllm::v1::KVCacheConfig& kv_cfg, int block_size,
    bool enable_prefix_caching) {
  // STEP 1 — re-derive the cache block size from the groups that were actually
  // built. `block_size` is what the engine ASKED to page at (raised to the
  // architecture's `kv_block_size_floor`); a group may publish a SMALLER page
  // regardless, because upstream fixes some of them by tensor sharing rather
  // than by configuration — DeepSeek-V4's SWA cache is a hard 64 tokens
  // (`vllm/v1/attention/backends/mla/sparse_swa.py:76-83`, `self.block_size =
  // 64`) and its compressor states are 4 and 8 (`compressor.py:174-186`).
  // Upstream reconciles the two by writing the minimum back over the
  // configured value before anything reads it as a scheduling quantity:
  // `cache_config.block_size = min(g.kv_cache_spec.block_size for g in
  // kv_cache_groups)` (`vllm/v1/engine/core.py:335-338` @ `e126687a9a`),
  // guarded by `if kv_cache_groups:` because an attention-free model publishes
  // none and then keeps the size it paged at.
  int cache_block_size = block_size;
  for (const auto& group : kv_cfg.kv_cache_groups) {
    if (!group.kv_cache_spec) continue;
    if (group.kv_cache_spec->block_size < cache_block_size) {
      cache_block_size = group.kv_cache_spec->block_size;
    }
  }
  // STEP 2 — split that into the scheduler's token-alignment invariant and the
  // prefix-hash granularity (`core.py:158-160`). For one group both are the
  // cache block size; for several they are the LCM and the GCD. This call is
  // what makes `resolve_kv_cache_block_sizes` REACHED: it was ported at
  // `kv_cache_utils.cpp:640` with no production caller, so the engine was
  // deriving neither value and handing the coordinator the configured size for
  // both.
  return vllm::v1::resolve_kv_cache_block_sizes(
      kv_cfg, cache_block_size, /*prefix_match_unit=*/std::nullopt,
      enable_prefix_caching, /*connector_enabled=*/false,
      /*dcp_world_size=*/1);
}

std::unique_ptr<vllm::v1::Scheduler> LoadedEngine::MakeScheduler(
    bool async_enabled, vllm::SchedulerConfig scheduler_config,
    vllm::v1::KVCacheConfig kv_cache_config, int block_size,
    int hash_block_size, bool enable_caching,
    vllm::v1::StructuredOutputManager* structured_output_manager,
    std::optional<vllm::SpeculativeConfig> speculative_config) {
  if (async_enabled) {
    // get_scheduler_cls -> AsyncScheduler (scheduler.py:180-189). SPEC-DFLASH2
    // W7 (#1824): the speculative_config now rides into the AsyncScheduler —
    // an Eagle-type speculator keeps async scheduling and the AsyncScheduler
    // needs the config for num_lookahead_tokens, the spec budget, and the
    // -1 placeholder assignment.
    return std::make_unique<vllm::v1::AsyncScheduler>(
        std::move(scheduler_config), std::move(kv_cache_config), block_size,
        enable_caching, structured_output_manager,
        std::move(speculative_config), hash_block_size);
  }
  return std::make_unique<vllm::v1::Scheduler>(
      std::move(scheduler_config), std::move(kv_cache_config), block_size,
      enable_caching, structured_output_manager, std::move(speculative_config),
      /*kv_events_config=*/nullptr, /*data_parallel_rank=*/0, hash_block_size);
}

std::optional<vllm::SpeculativeConfig> LoadedEngine::ResolveSpecConfig(
    const EngineParams& params, const HfConfig& config) {
  if (!params.speculative_config.has_value()) {
    return std::nullopt;  // production default: no speculation.
  }
  const vllm::SpeculativeConfig& cli = *params.speculative_config;
  // SPEC-DRAFTER-CHAIN W1 (#1522): THE PRODUCTION READER of the parsed chain,
  // and the reason the field does not land dead.
  //
  // W1 lands the field, its validation and its refusals, and NO chain
  // behaviour. Nothing resolves a chain and nothing changes which speculator
  // drafts. That leaves exactly one way for the wave to be harmful, and it is
  // not a malformed document: it is a WELL-FORMED one that parses, stores, and
  // is then ignored. The engine would draft with one speculator — or with none —
  // under a document whose author believes it configures several, and any
  // measurement taken there is a measurement of a configuration nobody chose.
  // That is #1160's failure with a larger blast radius, so the chain is REFUSED
  // BY NAME rather than silently reduced.
  //
  // FIRST in this function, before every method branch, because two of those
  // branches read a draft checkpoint's own config.json off disk
  // (`CheckDflash2DraftArm`, `ReadDsparkDraftKeys`) and because `cli.method` is
  // EMPTY on a chain config — it would otherwise fall through to the "only
  // methods mtp, dflash and ngram are supported" line at the bottom and tell the
  // user that "" is not one of them.
  //
  // `FromModelDir` calls this function ahead of its path resolution so the
  // refusal also precedes every weight operation, which is what G5 means by
  // "before any weight I/O"; `tests/vllm/entrypoints/test_drafter_chain_reach.cpp`
  // asserts that ordering by refusing a chain against a path that does not
  // exist.
  if (cli.use_drafter_chain()) {
    std::string listed;
    for (const vllm::SpeculativeChainEntry& entry : cli.drafter_chain) {
      if (!listed.empty()) listed += ", ";
      listed += "\"" + entry.method + "\"";
    }
    throw std::invalid_argument(
        "speculative-config: \"vllm_cpp.drafter_chain\" [" + listed +
        "] parses and validates here, but NOTHING resolves a chain yet: this "
        "engine still drafts with exactly one speculator per step. It is "
        "refused rather than silently reduced to one drafter or to no "
        "speculation at all, because a run under a configuration nobody chose "
        "is worse than a run that does not start. Per-sequence resolution is "
        "owed by row SPEC-DRAFTER-CHAIN wave W3, after the per-drafter "
        "attribution of W2 (.agents/specs/drafter-chain.md), issue "
        "https://github.com/mudler/vllm.cpp/issues/1522. Name one method at the "
        "top level to run today.");
  }
  // SPEC-DFLASH D5: the block-diffusion drafter. num_speculative_tokens is REQUIRED
  // (= the draft block_size, e.g. 16; speculative.py raises if None) and there is no
  // n_predict-module divisibility (the drafter is non-autoregressive). The concrete
  // draft checkpoint is loaded separately (LoadDflashDraft); this only finalizes the
  // scheduler-facing config and carries the draft path forward. The extra scheduler
  // lookahead slot comes from NumLookaheadTokens() = k+1 (already coded).
  if (cli.method == "dflash") {
    if (!cli.num_speculative_tokens.has_value()) {
      throw std::invalid_argument(
          "speculative-config: method \"dflash\" requires num_speculative_tokens "
          "(the draft block_size, e.g. 16)");
    }
    // SPEC-DFLASH2 W1 (#1314): classify the draft by its OWN config.json before
    // resolving anything else, exactly as SPEC-DSPARK-QWEN3-ROUTING does for
    // DSpark below. Upstream reads the architecture off the draft config too
    // (v1/worker/gpu/spec_decode/__init__.py:12 @ the PR head); this engine read
    // nothing, so a DFlash2 checkpoint drafted through the DFlash1 lane by
    // OMISSION rather than by decision. This is the production caller
    // `SpeculativeConfig::IsDflash2Draft` would otherwise lack.
    if (cli.draft_model_path.has_value()) {
      CheckDflash2DraftArm(*cli.draft_model_path);
    }
    vllm::SpeculativeConfig cfg =
        vllm::SpeculativeConfig::ResolveDflash(*cli.num_speculative_tokens);
    cfg.draft_model_path = cli.draft_model_path;
    return cfg;
  }
  // SPEC-NGRAM (ROAD-V1-D3): the draft-FREE n-gram proposer. num_speculative_tokens
  // (k) is REQUIRED (speculative.py:1224-1234); the prompt_lookup window defaults to
  // 5/5. There is NO draft checkpoint to load and NO n_predict-module constraint —
  // the drafts come from matching the sequence's own suffix. The GDN spec verify
  // machinery (widened conv + k+1 state slots) is still needed and is reused via
  // MakeQwen3_5KVCacheSpec(num_spec>0) exactly like MTP; the fa_draft draft-KV group
  // it allocates is simply unused (ngram has no draft-model forward).
  if (cli.method == "ngram") {
    if (!cli.num_speculative_tokens.has_value()) {
      throw std::invalid_argument(
          "speculative-config: method \"ngram\" requires num_speculative_tokens");
    }
    return vllm::SpeculativeConfig::ResolveNgram(*cli.num_speculative_tokens,
                                                 cli.prompt_lookup_min,
                                                 cli.prompt_lookup_max);
  }
  // SPEC-DSPARK W5: the semi-autoregressive block drafter. Like DFlash it names a
  // SEPARATE draft checkpoint and takes k from the CLI (a native Qwen3 DSpark
  // config carries no n_predict, speculative.py:973-994).
  //
  // SPEC-DSPARK-BLOCK-SIZE-GUARD (#1225): read the draft's own n_predict and
  // block floor and PASS them. Both arguments were std::nullopt here, so
  // ResolveDspark's k >= block floor (speculative.h:179-185, from
  // speculative.py:1003-1027) reached no user and a k below the checkpoint's
  // block was accepted in silence.
  if (cli.method == "dspark") {
    const DsparkDraftKeys keys = ReadDsparkDraftKeys(cli.draft_model_path);
    // speculative.py:990-994. Kept ahead of ResolveDspark only for the case it
    // was written for -- a native Qwen3 draft, which carries no n_predict to
    // default from. With one present, :973-979 defaults k and this must not
    // pre-empt it, or the n_predict threaded above would be unreachable.
    if (!cli.num_speculative_tokens.has_value() && !keys.n_predict.has_value()) {
      throw std::invalid_argument(
          "speculative-config: method \"dspark\" requires num_speculative_tokens "
          "(a DSpark draft config carries no n_predict)");
    }
    // SPEC-DSPARK-QWEN3-ROUTING (#1193): classify the draft by its OWN config
    // before resolving anything else. Upstream picks the DSpark lane from the
    // draft's architectures and model_type (speculative.py:882-887 and :934-944
    // @ 555967922, plus vllm#52197); this engine picked it from the CLI method
    // string alone, so a `DSparkDraftModel` checkpoint loaded as a Qwen3 draft by
    // OMISSION rather than by decision, and a DeepSeek-V4 one loaded far enough
    // to fail on a missing key. This is the production caller
    // `SpeculativeConfig::IsDsparkDraft` lacked.
    if (cli.draft_model_path.has_value()) {
      const std::optional<DsparkDraftIdentity> ident =
          ReadDsparkDraftIdentity(*cli.draft_model_path);
      if (ident.has_value()) {
        std::string listed;
        for (const std::string& arch : ident->architectures) {
          if (!listed.empty()) listed += ", ";
          listed += "\"" + arch + "\"";
        }
        // Upstream's detection (speculative.py:882-887 + #52197 hunk 1). A draft
        // that fails it is precisely the set upstream's fallback rewrites into
        // the DeepSeek-V4 lane, so it is refused with that lane named.
        if (!vllm::SpeculativeConfig::IsDsparkDraft(
                *cli.draft_model_path, ident->architectures, ident->model_type)) {
          throw std::invalid_argument(
              "speculative-config: the draft checkpoint at \"" +
              *cli.draft_model_path +
              "\" does not identify as a Qwen3 or Gemma4 DSpark draft: its "
              "model id carries no \"dspark\", and its architectures [" +
              listed + "] with model_type \"" + ident->model_type +
              "\" name none of \"Qwen3DSparkModel\", \"Gemma4DSparkModel\" or "
              "the \"DSparkDraftModel\" + \"qwen3\" pair "
              "(vllm/config/speculative.py:882-887 @ 555967922 + "
              "vllm-project/vllm#52197). Upstream routes exactly this set into "
              "the DeepSeek-V4 DSpark lane (:934-944), and that lane is not "
              "implemented here: DeepseekV4Model is a stub and it needs two "
              "Sparks. Owed by row SPEC-DSPARK-QWEN3-ROUTING "
              "(.agents/specs/dspark-qwen3-routing.md).");
        }
        // Upstream's normalization (#52197 hunk 2), called for its REFUSAL. At
        // this pin `SpeculativeConfig::ResolveDsparkArchitecture` is TOTAL over
        // its two outcomes: it answers "Qwen3DSparkModel" or it throws the
        // DeepSeek-V4 refusal by name. So the returned lane is always the one
        // `LoadDsparkDraft` implements, and a `lane != "Qwen3DSparkModel"` guard
        // here would be a branch nothing can enter — dead code, which the earlier
        // shape of this call site carried and a mutation caught. When a further
        // upstream lane arrives (#52197's own context already carries a
        // `K3DSparkModel` arm absent from this pin), the returned value becomes a
        // decision and the dispatch lands WITH the lane that needs it; the
        // pin-advance item under `## Owed` in
        // .agents/specs/dspark-qwen3-routing.md carries that.
        vllm::SpeculativeConfig::ResolveDsparkArchitecture(ident->architectures,
                                                           ident->model_type);
      }
    }
    vllm::SpeculativeConfig resolved = vllm::SpeculativeConfig::ResolveDspark(
        keys.n_predict, keys.block_floor, cli.num_speculative_tokens,
        keys.block_floor_key);
    resolved.draft_model_path = cli.draft_model_path;
    return resolved;
  }
  if (cli.method != "mtp") {
    throw std::invalid_argument(
        "speculative-config: only methods \"mtp\", \"dflash\" and \"ngram\" are "
        "supported (got \"" +
        cli.method + "\")");
  }
  // mtp_num_hidden_layers from the checkpoint (text_config, default 1 — both gate
  // checkpoints). Mirrors qwen3_5_mtp.cpp NumMtpLayers.
  int64_t mtp_layers = 1;
  if (config.raw.is_object()) {
    const nlohmann::json* text = &config.raw;
    if (config.raw.contains("text_config") &&
        config.raw.at("text_config").is_object()) {
      text = &config.raw.at("text_config");
    }
    if (text->is_object()) {
      mtp_layers = text->value("mtp_num_hidden_layers", int64_t{1});
    }
  }
  // SPEC-MTP-K-GT-1 (#81): the resolved k is SERVED, at any depth. Depth above 1
  // was refused on this line between commits, because the propose path carried
  // only upstream's k=1 early exit (autoregressive/speculator.py:236-238) and
  // would have billed the user for a depth it drafted one token for. The
  // multi-step propose (MtpProposeDrafts) landed the loop behind that early
  // exit, so the refusal is GONE rather than widened.
  return vllm::SpeculativeConfig::ResolveMtp(static_cast<int>(mtp_layers),
                                             cli.num_speculative_tokens);
}

vllm::v1::KVCacheConfig LoadedEngine::MakeKVCacheMaybeSpec(
    const LoadedModel& model, const HfConfig& config, int block_size,
    int num_blocks, const std::optional<vllm::SpeculativeConfig>& spec) {
  // The architecture may require a larger KV block than the engine's default.
  // Upstream DERIVES that geometry from the model rather than taking it from an
  // operator, so raising it here is the mirror; leaving it to a flag would make
  // the model unreachable on its default configuration, and `vllm-cli` does not
  // expose one at all.
  const int resolved_bs =
      ModelRegistry::ResolveKVBlockSize(model.registration(), block_size);
  if (resolved_bs != block_size) {
    std::fprintf(stderr,
                 "[vllm] kv-cache: raising block_size %d -> %d, the floor this "
                 "architecture derives (a compress_ratio page cannot be smaller "
                 "than one token)\n",
                 block_size, resolved_bs);
    block_size = resolved_bs;
  }
  vllm::v1::KVCacheConfig kv;
  if (spec.has_value()) {
    // Speculation is Qwen3.5/3.6-only at this pin (both gate checkpoints); build
    // the widened spec KV directly (extra GDN k+1 state slots + widened conv row
    // + the `fa_draft` full-attn group). MakeQwen3_5KVCacheSpec(num_spec>0).
    kv = vllm::MakeQwen3_5KVCacheSpec(config, block_size, num_blocks,
                                      spec->ResolvedNumSpeculativeTokens(),
                                      /*share_mtp_pages=*/spec->method == "mtp");
  } else {
    kv = ModelRegistry::MakeKVCache(model, config, block_size, num_blocks);
  }
  // FIX-KV-GROUP-LAYER-COUNT (#1963, #1966). THE single funnel: both branches
  // above return through here, so one call reaches every architecture and both
  // the probe and the resized config MakeKVCacheResolved builds.
  //
  // Thirty-three of the thirty-four registries publish ONE placeholder name per
  // KV group, and `KVBytesPerBlock` / `recurrent_state_bytes` read
  // `layer_names.size()` as the layer count. Without this line a
  // `--kv-cache-memory` budget is divided by ONE layer's page and then
  // multiplied by every layer when the runner allocates: 1 GiB in, 8.5 GiB
  // allocated on the 27B, and the #371 recurrent-state OOM guard reads 0.90 GiB
  // against a 43.40 GiB allocation. Upstream cannot have that bug because the
  // count that divides the budget and the count that sizes the allocation are
  // the same expression over the same list (`kv_cache_utils.py:1399`,
  // `:1005-1008`, `:1409-1416`).
  //
  // ORDERING, against `ResolveMaxNumSeqs` (#1983, which landed first): that
  // resolver reads `kv_cfg_`, which is `MakeKVCacheResolved`'s result, so it
  // always sees names this call has already resolved. Its seat count is
  // `num_blocks`-linear, and `num_blocks` is the one input of its arithmetic
  // this change moves — which is the point: upstream's `num_blocks` is
  // per-layer (`kv_cache_utils.py:1008` divides by `num_layers`), and that is
  // the meaning its unification against one attention page assumes. Before this
  // line the byte-budget path handed it a count inflated by the layer count, so
  // its clamp was too permissive. The two fixes agree; they do not fight.
  vllm::v1::ResolveKVCacheGroupLayerNames(kv, config.num_hidden_layers,
                                          config.layer_types);
  return kv;
}

int LoadedEngine::ResolveNumBlocks(const EngineParams& params,
                                   const vllm::v1::KVCacheConfig& probe) {
  // 1. Explicit override wins (vLLM num_gpu_blocks_override).
  if (params.num_blocks > 0) {
    return params.num_blocks;
  }
  // 2. Absolute KV-pool budget. IGNORES gpu_memory_utilization, exactly like
  //    vLLM CacheConfig (cache.py:189). num_blocks = budget / bytes-per-block.
  if (params.kv_cache_memory_bytes > 0) {
    const int64_t bytes_per_block = vllm::v1::KVBytesPerBlock(probe);
    if (bytes_per_block <= 0) {
      throw std::runtime_error(
          "ResolveNumBlocks: model reports zero KV bytes per block; cannot size "
          "the pool from --kv-cache-memory");
    }
    const int64_t n = params.kv_cache_memory_bytes / bytes_per_block;
    if (n <= 0) {
      throw std::invalid_argument(
          "kv_cache_memory_bytes (" +
          std::to_string(params.kv_cache_memory_bytes) +
          ") is smaller than a single KV block (" +
          std::to_string(bytes_per_block) +
          " bytes); raise --kv-cache-memory or set an explicit --num-blocks");
    }
    return static_cast<int>(n);
  }
  // 3. gpu_memory_utilization profile path (ROAD-V1-MEM M3): needs a real device
  //    profile run to measure the non-KV footprint before the free-memory
  //    fraction can be turned into a block count. Until that lands, fall back to
  //    the historical default so the default path is byte-identical.
  // TODO(ROAD-V1-MEM M3): profile run -> available_kv = free*util - non_kv.
  constexpr int kFallbackNumBlocks = 256;
  // FIX-GPU-MEM-UTIL-INERT (#1165): this line is where an explicitly chosen
  // fraction gets discarded, so this is where the engine has to say so. The
  // flag is NOT refused: roadmap_v1.md:71 records the intent that it keeps
  // vLLM's exact name and fraction semantics so a published vLLM launch line
  // ports unchanged. What was wrong was accepting the value in silence, which
  // left a user believing they had sized the KV pool when they had sized
  // nothing.
  //
  // Only an EXPLICIT value warns. A default nobody set has nothing to report,
  // and a line on every start is noise rather than a warning.
  if (params.gpu_memory_utilization.has_value()) {
    std::cerr
        << "vllm.cpp: WARNING --gpu-memory-utilization "
        << *params.gpu_memory_utilization
        << " was accepted but did NOT size the KV cache.\n"
           "vllm.cpp:   The profile run that turns a free-memory fraction into "
           "a block count is not\n"
           "vllm.cpp:   implemented yet (ROAD-V1-MEM M3, "
           "https://github.com/mudler/vllm.cpp/issues/83).\n"
           "vllm.cpp:   The pool fell back to "
        << kFallbackNumBlocks
        << " blocks. To size it today, pass\n"
           "vllm.cpp:   --kv-cache-memory <bytes> for an absolute KV budget, or "
           "--num-blocks <n> for an\n"
           "vllm.cpp:   exact block count.\n";
    // Unbuffered by the time the loader's next line lands, so the notice cannot
    // be separated from the load it belongs to (same reason as the auto-fit
    // INFO line in ResolveMaxModelLen).
    std::cerr.flush();
  }
  return kFallbackNumBlocks;
}

vllm::v1::KVCacheConfig LoadedEngine::MakeKVCacheResolved(
    const LoadedModel& model, const HfConfig& config, int block_size,
    const EngineParams& params,
    const std::optional<vllm::SpeculativeConfig>& spec) {
  // The per-block byte geometry is independent of the block count, so build a
  // probe at the override-or-256 count, read its geometry to resolve the real
  // count, and only rebuild when the resolved count differs.
  // Once per load — this function is the single call site (the LoadedEngine
  // ctor's kv_cfg_ member initializer), and ApplyResolvedCacheDType below runs
  // up to twice.
  WarnUncalibratedKvScales(params);
  const int probe_blocks = params.num_blocks > 0 ? params.num_blocks : 256;
  vllm::v1::KVCacheConfig probe =
      MakeKVCacheMaybeSpec(model, config, block_size, probe_blocks, spec);
  // KV-FP8 W3 — the storage dtype is applied to the PROBE, before
  // ResolveNumBlocks reads its geometry. That ordering IS the halved-block
  // feature: `KVBytesPerBlock(probe)` is the divisor knob 2 sizes the pool with,
  // so an fp8 page (1 byte/element vs bf16's 2) halves the divisor and doubles
  // the block count at the same --kv-cache-memory. Applying it after would size
  // the pool from a bf16 page and then serve it as fp8, which is the same pool
  // in half the bytes rather than twice the pool.
  ApplyResolvedCacheDType(params, probe);
  const int resolved = ResolveNumBlocks(params, probe);
  auto with_recurrent_prefix = [&](vllm::v1::KVCacheConfig result) {
    if (model.registration().architecture == "Qwen3_5ForConditionalGeneration" &&
        ResolveEnablePrefixCaching(params, model.registration().info)) {
      VT_CHECK(!spec.has_value() || spec->method == "mtp",
               "Qwen recurrent prefix speculation requires the native MTP state protocol");
      result.recurrent_prefix_snapshots = std::make_shared<vllm::v1::RecurrentPrefixSnapshotIndex>(4, block_size);
    }
    return result;
  };
  if (resolved == probe_blocks) {
    return with_recurrent_prefix(std::move(probe));
  }
  vllm::v1::KVCacheConfig sized =
      MakeKVCacheMaybeSpec(model, config, block_size, resolved, spec);
  ApplyResolvedCacheDType(params, sized);
  return with_recurrent_prefix(std::move(sized));
}

// Upstream's `logger.warning_once` for the defaulted-scale case
// (`kv_cache.py:150-156`), lifted out of ApplyResolvedCacheDType because that
// function runs up to TWICE per load (probe, then the resized config) and a
// warning that fires twice reads as two engines. Once per LOAD, not once per
// process: two engines in one process both deserve the line.
void LoadedEngine::WarnUncalibratedKvScales(const EngineParams& params) {
  const vllm::ResolvedKvCacheScales scales = vllm::ResolveKvCacheScales(
      params.kv_cache_dtype, /*calculate_kv_scales=*/false,
      vllm::kKvScaleUnloaded, vllm::kKvScaleUnloaded);
  if (scales.origin == vllm::KvScaleOrigin::kNotQuantized || !scales.uncalibrated) {
    return;
  }
  std::cerr << vllm::UncalibratedKvScaleWarning(params.kv_cache_dtype);
  std::cerr.flush();
}

void LoadedEngine::ApplyResolvedCacheDType(const EngineParams& params,
                                           vllm::v1::KVCacheConfig& cfg) {
  // `params.kv_cache_dtype` is already resolved against the checkpoint by
  // FromModelDir; here it only has to become bytes.
  const vllm::v1::ResolvedCacheDType resolved = vllm::v1::ParseCacheDType(
      params.kv_cache_dtype, vllm::v1::ResolveKvCacheDType());
  // The scale pair, resolved through the SAME four-arm mirror of
  // `BaseKVCacheMethod.process_weights_after_loading` that upstream uses. Both
  // loaded scales are the `KVCacheScaleParameter` sentinel because no model's
  // weight loader extracts `k_scale`/`v_scale` yet (owed, see the spec's
  // `## Owed`), so a declaring checkpoint lands on `kDeclaredButAbsent`, which
  // is a DIFFERENT state from a checkpoint that declared nothing. The #1574
  // subject `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` reaches this arm only when an
  // operator types `--kv-cache-dtype fp8`: its inline
  // `config.json:quantization_config` declares no `kv_cache_*` key, and that is
  // the document both engines read (`transformers_utils/config.py:751-761`).
  const vllm::ResolvedKvCacheScales scales = vllm::ResolveKvCacheScales(
      params.kv_cache_dtype, /*calculate_kv_scales=*/false,
      vllm::kKvScaleUnloaded, vllm::kKvScaleUnloaded);
  if (scales.origin == vllm::KvScaleOrigin::kNotQuantized) {
    // Nothing declared an fp8 KV cache. Do NOT reach for a scale: ApplyCacheDType
    // returns immediately on the auto/bf16 path anyway, and asking
    // ScalesForFp8Store here would be the invented-default this port refuses.
    vllm::v1::ApplyCacheDType(cfg, resolved, /*k_scale=*/1.0F, /*v_scale=*/1.0F);
    return;
  }
  float k_scale = 0.0F;
  float v_scale = 0.0F;
  // The uncalibrated line is NOT printed here. This function runs up to twice
  // per load (the probe, then the resized config), and a process-static
  // `call_once` would fix that by printing once for the process instead — which
  // silences the SECOND engine in the same process rather than the second call
  // of the same load. `WarnUncalibratedKvScales` above is the once-per-load
  // owner, and it is the only caller of the message.
  vllm::ScalesForFp8Store(scales, &k_scale, &v_scale);
  vllm::v1::ApplyCacheDType(cfg, resolved, k_scale, v_scale);
}

int LoadedEngine::ResolveMaxNumSeqs(const EngineParams& params,
                                    const vllm::v1::KVCacheConfig& kv_cfg,
                                    bool serves_one_sequence_per_step) {
  const int configured = params.max_num_seqs > 0 ? params.max_num_seqs : 8;
  const vllm::v1::HybridKvBudget budget =
      vllm::v1::ComputeHybridKvBudget(kv_cfg);
  int resolved = vllm::v1::ClampMaxNumSeqsToStateBudget(configured, budget);
  // MODEL-MM-QWEN4-EXP W5L (#2031): THE MODEL'S OWN CEILING, and it is not a
  // tuning knob. A forward that refuses `num_reqs > 1` throws from inside the
  // EngineCore busy loop, which treats a throw as FATAL — the socket stays open
  // and every request from then on is a 500. The default `max_num_seqs` is 128,
  // so without this clamp the first pair of overlapping requests kills the
  // server. Clamping here makes the same engine serve them one after another.
  //
  // AFTER the budget clamp because the two bound different quantities and the
  // smaller has to win; `resolved` is therefore the min of both. The line below
  // is printed whenever this clamp is the binding one, on the same "a number the
  // operator did not choose is never silent" rule as the budget message.
  if (serves_one_sequence_per_step && resolved > 1) {
    std::cerr << "INFO model concurrency: reduced max_num_seqs from " << resolved
              << " to 1. This architecture's forward serves ONE sequence per "
                 "step and refuses a batched one, and an EngineCore that meets "
                 "that refusal dies rather than degrades. Batching it needs the "
                 "ragged multi-request plumbing owed under "
                 ".agents/specs/qwen4-exp-flash-next.md (issue #2031).\n";
    std::cerr.flush();
    return 1;
  }
  if (resolved >= configured) {
    // Attention-only models, pure-recurrent models, and every hybrid whose
    // budget already holds the configured concurrency land here: no line, no
    // change, byte-identical to before this resolver existed.
    return configured;
  }
  // A reduction is never silent. Upstream logs its own block-size raise and its
  // auto-fit for the same reason: a number the operator did not choose, that
  // changes what the engine serves, has to appear in the engine's own output.
  // It names WHAT was compared against WHAT, because a bound whose wiring is
  // invisible in the report cannot be audited.
  std::cerr << "INFO recurrent-state budget: reduced max_num_seqs from "
            << configured << " to " << resolved << ". The KV pool ("
            << kv_cfg.num_blocks << " blocks) holds "
            << budget.unified_num_blocks
            << " unified pages of " << budget.unified_block_tokens
            << " tokens (one page = one " << budget.mamba_page_bytes
            << "-byte GDN state), and each sequence owns "
            << budget.slots_per_seq
            << " of them. Raise --num-blocks / --kv-cache-memory for more"
               " concurrent sequences, or lower num_speculative_tokens.\n";
  std::cerr.flush();
  return resolved;
}

int LoadedEngine::ResolveMaxModelLen(const EngineParams& params,
                                     const HfConfig& config,
                                     const vllm::v1::KVCacheConfig& kv_cfg,
                                     int block_size, bool is_dense_arch) {
  // kv_cache_utils.py:2304-2327 @ e126687a9a. See model_loader.h for the two
  // arms and why this post-condition matters.
  //
  // FIX-KV-POOL-MIN-FIT (ISSUE-LOCAL-01M36QVG0KGKEP4MSMKT18MMZ4): both arms
  // count ONE max-length request over EVERY KV cache group and reserve the null
  // block, because that is what the scheduler's allocator claims: one block
  // table per group, all drawn from this one pool. The one-table count this
  // replaced accepted a Qwen3.5 speculative pool (`fa`, `gdn`, `fa_draft`) at a
  // third of its need, and the admitted max-length request then waited forever.
  (void)block_size;  // each group's spec carries its own block size
  const int64_t bytes_per_block = vllm::v1::KVBytesPerBlock(kv_cfg);
  const int64_t available =
      static_cast<int64_t>(kv_cfg.num_blocks) * bytes_per_block;

  if (params.max_model_len > 0) {
    // The caller pinned a length. Refuse if the pool cannot serve it. A config
    // with no paged KV (attention-free, or pure Mamba/GDN, where
    // KVBytesPerBlock is 0) has nothing to run out of and is never refused,
    // which is upstream's `if kv_cache_spec:` (kv_cache_utils.py:950).
    // `max_num_batched_tokens` is the value the scheduler's managers get, so
    // a sliding-window count here is the count their admission cap uses.
    vllm::v1::check_enough_kv_cache_memory(
        kv_cfg, available, params.max_model_len,
        ResolveMaxNumBatchedTokens(params, params.max_model_len,
                                   is_dense_arch));
    return params.max_model_len;
  }

  // Unpinned: serve the checkpoint's own context, auto-fitted down to the pool.
  const int64_t derived = config.max_position_embeddings;
  if (derived <= 0) {
    // No context length in the config at all. There is nothing to fit against,
    // and it is not this function's job to invent one.
    return static_cast<int>(derived);
  }
  // The batched-token budget at the DERIVED length. The budget can only shrink
  // at a shorter length, so a fitted length is never one the scheduler's
  // admission cap would refuse.
  const int64_t fitted = vllm::v1::auto_fit_max_model_len(
      kv_cfg, available, derived,
      ResolveMaxNumBatchedTokens(params, static_cast<int>(derived),
                                 is_dense_arch));
  if (fitted < derived) {
    // kv_cache_utils.py:2150-2157 logs the reduction. Silence here would make a
    // shortened context look like a model-config surprise later.
    std::cerr << "INFO auto-fit max_model_len: reduced from " << derived
              << " to " << fitted << " to fit the KV cache ("
              << kv_cfg.num_blocks << " blocks, one of them the null block, "
              << "shared by " << kv_cfg.kv_cache_groups.size()
              << " KV cache groups). Raise --num-blocks / --kv-cache-memory for "
                 "a longer context.\n";
    std::cerr.flush();
  }
  return static_cast<int>(fitted);
}

// SPEC-MTP-K-GT-1 (#81): the in-memory mirror of FromModelDir's
// `maybe_attach_mtp`. A caller holding weights in memory had no way to supply
// the `mtp.*` draft head, so an in-memory speculative engine could only run with
// a NULL drafter -- which a depth gate must never mistake for working
// speculation. Attaching before the LoadedEngine body runs is what matters: the
// constructor asks `model_->supports_mtp_draft()` and calls BuildMtpDraft in its
// member-initialiser list, so a later attach would be too late.
std::unique_ptr<LoadedModel> AttachMtp(std::unique_ptr<LoadedModel> model,
                                       std::optional<Qwen3_5MTPWeights> mtp) {
  if (mtp.has_value()) model->AttachMtpDraftWeights(std::move(*mtp));
  return model;
}

// #1946. The header carries the argument; this is the mechanism.
bool BindDflashDraftSharedEmbed(DflashDraft& draft, const LoadedModel& target) {
  // The DSpark lane is NOT this lane. A DSpark checkpoint usually ships its own
  // table and keeps it by value inside Qwen3DSparkWeights, and the loader's
  // fallback fills THAT rather than `weights.embed_tokens` -- so binding here
  // would rebind a field the DSpark forward never reads while leaving the copy
  // that costs the memory in place. Skipped by name, and owed as its own row.
  if (draft.dspark != nullptr) return false;
  const OwnedTensor* shared = target.shared_embed_tokens();
  if (shared == nullptr) return false;
  const OwnedTensor& own = draft.weights.embed_tokens;
  // Nothing was read into the draft (an in-memory draft built without one), or
  // it is ALREADY the target's tensor. Either way there is no second copy and
  // no rebind to make; returning false keeps `EmbedTable()` on whatever the
  // caller set up.
  if (own.Empty() || shared == &own) return false;

  // The one discriminator, and it is load-bearing rather than defensive: a GGUF
  // target may keep its table F16 in place (`LoadEmbedAndHead`'s kKeepF16 arm)
  // while `LoadGgufSharedEmbedAndHeadBf16` always hands the draft BF16. Those
  // are different bytes at different widths, and aliasing them would make the
  // draft gather f16 rows through a bf16 view -- silently wrong tokens, which no
  // memory gate would ever see.
  bool same = own.dtype == shared->dtype && own.rank == shared->rank;
  for (int i = 0; same && i < own.rank; ++i) same = own.shape[i] == shared->shape[i];
  if (!same) {
    const auto describe = [](const OwnedTensor& w) {
      std::string s(vt::Name(w.dtype));
      s += " [";
      for (int i = 0; i < w.rank; ++i) {
        if (i != 0) s += ", ";
        s += std::to_string(w.shape[i]);
      }
      return s + "]";
    };
    std::cerr << "vllm.cpp: DFlash draft embed NOT shared with the target: the "
                 "draft read "
              << describe(own) << " and the target holds " << describe(*shared)
              << ". Both tables stay resident; tokens are unaffected.\n";
    return false;
  }

  // The bytes this rebind stops uploading a second time. Read from the draft's
  // OWN copy, which is the allocation that goes away, and read BEFORE the clear.
  //
  // DERIVED FROM THE GEOMETRY, not from `bytes.size()` (#1946 review). The two
  // agree for every table this arm can see, because the guard above has just
  // proved dtype/rank/shape equal on both sides. They stop agreeing the moment a
  // buffer is a BORROWED view that is longer than its tensor -- an over-long
  // safetensors mapping slice, a shared bf16 expansion -- and then the number on
  // stderr would be the buffer's length rather than the table's. `RowSizeBytes`
  // is what the DEVICE allocation will be, which is the quantity the line
  // claims, and it is defined for block-quant and elementwise dtypes alike.
  const size_t saved = vt::RowSizeBytes(own.dtype, own.Numel());
  draft.weights.shared_embed_tokens = shared;
  // `del draft_inner.embed_tokens` (utils.py:73). Not tidiness: leaving the
  // draft's OwnedTensor populated leaves a second `d_dev` FIELD reachable, so a
  // future call site that reads `weights.embed_tokens` instead of `EmbedTable()`
  // would re-open the whole 2.5 GB with every gate green. An empty table makes
  // that call site refuse at its first `ResidentWeight` instead, by name.
  //
  // THAT REFUSAL IS A CHECK, NOT A PROPERTY OF THE EMPTY TENSOR (#1953).
  // This comment used to say `vt::Embedding` would refuse an empty table. It
  // would not have: `vt::Embedding` validates ranks, shapes, dtypes, contiguity
  // and device, and `ResidentWeight` takes the shape from the caller, so an empty
  // tensor passed every one of those and the failure was a SIGSEGV on the host
  // arm and a zero-byte allocation read out of bounds on a device arm. The
  // refusal now lives in `ResidentWeight` itself (dense_attn_block.h), and
  // `test_dflash2_embed_dedup.cpp` drives this exact call site to gate it.
  draft.weights.embed_tokens = OwnedTensor{};
  std::cerr << "vllm.cpp: DFlash draft embed SHARED with the target (one device "
               "copy, "
            << saved << " B saved)\n";
  return true;
}

// #1946: the member-initialiser wrapper for BindDflashDraftSharedEmbed, so the
// rebind happens WHILE the engine is being built rather than in a body that a
// later member could forward before. Null model or null draft passes through
// untouched, which is every non-speculative load.
//
// `static`, because it is generically named, sits in the public
// `vllm::entrypoints` namespace and no header declares it: without this it was
// an external symbol (`nm -C` reported `T vllm::entrypoints::BindSharedEmbed`)
// that a future translation unit could collide with. `BindDflashDraftSharedEmbed`
// is the exported lever, and it is exported deliberately so the gate reaches the
// function production calls; this wrapper is not.
static std::unique_ptr<DflashDraft> BindSharedEmbed(
    std::unique_ptr<DflashDraft> draft, const LoadedModel* target) {
  if (draft != nullptr && target != nullptr) {
    (void)BindDflashDraftSharedEmbed(*draft, *target);
  }
  return draft;
}

// XPU FlashAttention starts with a 64-token block. Python then grows its
// attention page until it can hold one Qwen Mamba state page. Derive the same
// size from the resolved KV dtype and speculative depth, before sizing the
// pool or constructing the runner. A larger caller override is retained.
static int ResolvePythonXpuHybridBlockSize(
    const LoadedModel& model, const HfConfig& config,
    const EngineParams& params,
    const std::optional<vllm::SpeculativeConfig>& spec) {
  const int requested = params.block_size > 0 ? params.block_size : 32;
  const int base = ModelRegistry::ResolveKVBlockSize(model.registration(), requested);
  if (model.registration().architecture != "Qwen3_5ForConditionalGeneration" ||
      params.device == vllm::Device::kCPU ||
      vllm::platforms::CurrentPlatform().device_type() != vt::DeviceType::kXPU) {
    return base;
  }
  constexpr int kXpuAttentionAlignment = 64;
  const int depth = spec.has_value() ? spec->ResolvedNumSpeculativeTokens() : 0;
  auto probe = vllm::MakeQwen3_5KVCacheSpec(
      config, kXpuAttentionAlignment, /*num_blocks=*/1, depth);
  const auto dtype = vllm::v1::ParseCacheDType(
      params.kv_cache_dtype, vllm::v1::ResolveKvCacheDType());
  vllm::v1::ApplyCacheDType(probe, dtype, /*k_scale=*/1.0F, /*v_scale=*/1.0F);
  const auto geometry = vllm::v1::ComputeHybridKvBudget(
      probe, kXpuAttentionAlignment);
  VT_CHECK(geometry.unified_block_tokens > 0 &&
               geometry.unified_block_tokens <= INT_MAX,
           "Qwen XPU hybrid attention page does not fit in int");
  const int resolved = std::max(base, static_cast<int>(geometry.unified_block_tokens));
  if (resolved != base) {
    std::cerr << "[vllm] kv-cache: XPU Qwen hybrid attention page " << base
              << " -> " << resolved << " tokens (Mamba page "
              << geometry.mamba_page_bytes << " B, attention token "
              << geometry.attn_bytes_per_token << " B, MTP depth " << depth
              << ")\n";
  }
  return resolved;
}

LoadedEngine::MainQueueOwner::~MainQueueOwner() {
  if (queue.handle != nullptr) vt::DestroyQueue(queue);
}

LoadedEngine::LoadedEngine(HfConfig config, Qwen3_5MoeWeights weights,
                           tok::Tokenizer tokenizer, const EngineParams& params,
                           std::optional<Qwen3_5MTPWeights> mtp_weights)
    : LoadedEngine(std::move(config),
                   AttachMtp(MakeQwen3_5MoeLoadedModel(std::move(weights)),
                             std::move(mtp_weights)),
                   std::move(tokenizer), params) {}

LoadedEngine::LoadedEngine(HfConfig config, Qwen3_5DenseWeights weights,
                           tok::Tokenizer tokenizer, const EngineParams& params,
                           std::optional<Qwen3_5MTPWeights> mtp_weights)
    : LoadedEngine(config,
                   AttachMtp(MakeQwen3_5DenseLoadedModel(std::move(weights), config),
                             std::move(mtp_weights)),
                   std::move(tokenizer), params) {}

// SPEC-DFLASH2 W3 (#1314): the DFlash counterpart of the overload above. See the
// header for why it exists and what it makes reachable.
LoadedEngine::LoadedEngine(HfConfig config, Qwen3_5DenseWeights weights,
                           tok::Tokenizer tokenizer, const EngineParams& params,
                           std::unique_ptr<DflashDraft> dflash_draft)
    : LoadedEngine(config, MakeQwen3_5DenseLoadedModel(std::move(weights), config),
                   std::move(tokenizer), params, /*preselected_queue=*/nullptr,
                   std::move(dflash_draft)) {}

LoadedEngine::LoadedEngine(HfConfig config,
                           std::unique_ptr<LoadedModel> model,
                           tok::Tokenizer tokenizer,
                           const EngineParams& params,
                           vt::Queue* preselected_queue,
                           std::unique_ptr<DflashDraft> dflash_draft,
                           std::optional<multimodal::Qwen3VLVisionWeights>
                               vision_tower,
                           multimodal::Qwen3VLVisionConfig vision_config,
                           bool mmproj_tower_skipped)
    : hash_ready_(EnsureNoneHash()),
      config_(std::move(config)),
      // LOAD-GGUF-MMPROJ: the `clip` projector's tower, already loaded and
      // already refused-or-accepted by FromModelDir before the tokenizer.
      // nullopt on every load that named no --mmproj.
      vision_tower_(std::move(vision_tower)),
      vision_config_(vision_config),
      // #607 L3: the projector was named and deliberately left unread. Reported
      // by skipped_towers() beside the model's own skipped stages.
      mmproj_tower_skipped_(mmproj_tower_skipped),
      // SPEC-MTP I5d: finalize the speculative config against the checkpoint
      // (n_predict + resolved k). nullopt on the production default path.
      resolved_spec_config_(ResolveSpecConfig(params, config_)),
      model_(std::move(model)),
      // SPEC-DFLASH D5: the separately-loaded DFlash draft (null for mtp/non-spec).
      // #1946: rebound onto the target's embedding table on the way in, which is
      // why `model_` is initialised (and declared) first. THE PRODUCTION CALL
      // SITE: all three draft loaders -- the GGUF branch, the two safetensors
      // branches, and the in-memory W3 overload -- funnel into this constructor,
      // and it is the first point at which the target and the draft both exist.
      dflash_draft_(BindSharedEmbed(std::move(dflash_draft), model_.get())),
      tokenizer_(std::move(tokenizer)),
      // #607 L2: carry the multimodal input limits onto the engine, so the ONE
      // config object every consumer asks (GetLimitPerPrompt) is the one the
      // server flags / the C ABI set. Default-constructed == the pre-L2 999.
      mm_config_(params.multimodal),
      // ROAD-V1-MEM M1: resolve the block count from the sizing knobs
      // (num_blocks override > kv_cache_memory_bytes > util fallback) against the
      // model's own per-block byte geometry. FIRST, because max_model_len_ is
      // resolved against this pool.
      // Resolved ONCE before anything reads it. On XPU Qwen, the hybrid page
      // grows to the Python Mamba/attention geometry; the registry floor still
      // applies. The funnel below re-resolves only for direct callers.
      block_size_(ResolvePythonXpuHybridBlockSize(
          *model_, config_, params, resolved_spec_config_)),
      kv_cfg_(MakeKVCacheResolved(*model_, config_, block_size_, params,
                                  resolved_spec_config_)),
      // The serving length, checked (pinned) or auto-fitted (unpinned) against
      // kv_cfg_. See ResolveMaxModelLen.
      max_model_len_(ResolveMaxModelLen(params, config_, kv_cfg_, block_size_,
                                        ModelRegistry::IsDenseModel(*model_))),
      // The serving concurrency, clamped to the recurrent-state budget the KV
      // pool affords. See ResolveMaxNumSeqs (issue #1983).
      max_num_seqs_([&] {
        const int resolved = ResolveMaxNumSeqs(
            params, kv_cfg_,
            model_->registration().factory->serves_one_sequence_per_step);
        const auto architecture = model_->registration().architecture;
        if (!NativeQwen3_5ImagesEnabled(config_, architecture, mm_config_))
          return resolved;
        const auto device = preselected_queue != nullptr
            ? preselected_queue->device.type
            : ResolveModelDeviceType(architecture, params.device);
        return ResolveNativeQwen3_5MaxNumSeqs(
            resolved, config_, architecture, device, mm_config_);
      }()),
      max_num_batched_tokens_(ResolveMaxNumBatchedTokens(
          params, max_model_len_, ModelRegistry::IsDenseModel(*model_))),
      prefix_caching_enabled_(ResolveEnablePrefixCaching(
          params, model_->registration().info)),
      // The scheduler's two block sizes, derived from the BUILT `kv_cfg_`
      // rather than from `block_size_`. Upstream derives them in exactly this
      // position — after the KV cache config exists and before the Scheduler is
      // constructed (`vllm/v1/engine/core.py:335-338` then `:158-170` @
      // `e126687a9a`). See ResolveSchedulerBlockSizes.
      scheduler_block_size_(
          ResolveSchedulerBlockSizes(kv_cfg_, block_size_,
                                     prefix_caching_enabled_)
              .first),
      hash_block_size_(
          ResolveSchedulerBlockSizes(kv_cfg_, block_size_,
                                     prefix_caching_enabled_)
              .second),
      // ENG-SGLANG-BEHAVIOR-FLAG SW3: resolve jump-forward once (config field +
      // VT_ENABLE_JUMP_FORWARD env override). Default nullopt+no-env => false =>
      // the byte-identical decode path (jump-forward is inert until enabled).
      jump_forward_enabled_(
          vllm::v1::JumpForwardEnabled(params.enable_jump_forward)),
      // runner_ FIRST (W3): the async-scheduling flip reads
      // runner_.runner_supports_async(). SPEC-MTP I5d: when speculation is on,
      // pass the resolved config + the MTP draft (built from the retained mtp.*
      // weights, sharing the target embed/lm_head). The draft KV `fa_draft` group
      // is allocated by the runner from kv_cfg_ (empty vector here), so the loop
      // reaches it via runner-owned storage. nullopt/null on the default path.
      runner_(config_, *model_, kv_cfg_,
              [&] {
                main_queue_owner_.queue = preselected_queue != nullptr
                    ? *preselected_queue
                    : SelectQueueForModel(model_->registration().architecture, params.device);
                // Ownership transfers immediately. The loader's exception
                // guard must not destroy it a second time if runner init fails.
                if (preselected_queue != nullptr) *preselected_queue = {};
                return main_queue_owner_.queue;
              }(),
              /*max_num_reqs=*/max_num_seqs_,
              max_model_len_,
              /*max_num_batched_tokens=*/max_num_batched_tokens_,
              resolved_spec_config_,
              // Only the MTP method builds an in-target MTP draft; DFlash (D5)
              // loads a SEPARATE draft, wired via set_dflash_draft in the body.
              resolved_spec_config_.has_value() &&
                      resolved_spec_config_->method == "mtp" &&
                      model_->supports_mtp_draft()
                  ? model_->BuildMtpDraft(config_)
                  : nullptr,
              /*draft_kv=*/{}),
      // Resolve the enable-flip from the now-constructed runner + VT_ASYNC_SCHED,
      // then size the batch-queue depth (2 under async scheduling → depth-2
      // step_with_batch_queue; 1 otherwise). Since the 2026-07-17 flip the default
      // (no env) resolves ON (VT_ASYNC_RUNNER default ON), mirroring vLLM;
      // VT_ASYNC_RUNNER=0 / VT_ASYNC_SCHED=0 roll back to the synchronous path.
      // SPEC-DFLASH2 W7 (#1824): async scheduling now SURVIVES an Eagle-type
      // speculator (mtp/dflash/dspark — vllm/config/vllm.py:1064-1112 at the
      // pin), the draft-in-output path having landed: the AsyncScheduler ships
      // -1 placeholder drafts, the runner fills them from its own propose, and
      // post_step is skipped under async. A method upstream refuses (host
      // ngram, draft_model at the pin) still forces the synchronous scheduler
      // through the spec_decode_incompatible arm. This line is the production
      // reach for the whole wave — the reachability mutation reverts it to the
      // pre-W7 `!resolved_spec_config_.has_value() &&` form.
      async_scheduling_enabled_(ResolveAsyncEnabled(
          MakeSchedulerConfig(
              max_model_len_, max_num_seqs_,
              max_num_batched_tokens_, params.policy),
          runner_.runner_supports_async(),
          model_->registration().info.is_pooling_model,
          /*spec_decode_incompatible=*/resolved_spec_config_.has_value() &&
              !resolved_spec_config_->async_scheduling_compatible())),
      max_concurrent_batches_(MakeSchedulerConfig(
                                  max_model_len_, max_num_seqs_,
                                  max_num_batched_tokens_, params.policy)
                                  .MaxConcurrentBatches(async_scheduling_enabled_)),
      // The engine-wide structured-output manager, native backend over the
      // tokenizer (upstream EngineCore constructs one unconditionally,
      // core.py:134). Wired into the scheduler + engine cores below so
      // response_format / C-ABI structured constraints gate decoding.
      structured_output_manager_(
          max_num_seqs_, [this] {
            // The factory runs lazily, after input_processor_ is constructed.
            // Use its resolved model EOS set, not only tokenizer.json's EOS:
            // chat/generation EOS ids must be valid grammar stop tokens too.
            return std::make_unique<vllm::v1::NativeStructuredOutputBackend>(
                tokenizer_, static_cast<int>(config_.vocab_size),
                input_processor_.eos_token_ids());
          }),
      // AsyncScheduler when the flip resolved ON, else the synchronous Scheduler.
      scheduler_(MakeScheduler(
          async_scheduling_enabled_,
          MakeSchedulerConfig(
              max_model_len_, max_num_seqs_,
              max_num_batched_tokens_, params.policy),
          kv_cfg_, scheduler_block_size_, hash_block_size_,
          /*enable_caching=*/prefix_caching_enabled_,
          &structured_output_manager_, resolved_spec_config_)),
      executor_(runner_),
      // SPEC-MTP I5d: with a speculator configured, EngineCore pulls the runner's
      // out-of-band drafts each step (take_draft_token_ids -> update_draft_token_ids)
      // so the next verify step schedules them. Default false (no-op post_step).
      engine_core_(*scheduler_, executor_, &structured_output_manager_,
                   /*check_for_draft_tokens=*/resolved_spec_config_.has_value()),
      // The admission-time prompt-length check validates against the RESOLVED
      // serving length, which is what upstream's model_config.max_model_len is
      // (input_processor.py:399-401). Passing config_ alone would check against
      // the raw checkpoint context and let through prompts the pool cannot hold.
      input_processor_(tokenizer_, config_, max_model_len_),
      output_processor_(&tokenizer_),
      // `hash_block_size_`, not `block_size_`: the hasher's granularity must be
      // the one the coordinator matches against, which is upstream's own
      // pairing (`get_request_block_hasher(hash_block_size, caching_hash_fn)`,
      // `vllm/v1/engine/core.py:232` @ `e126687a9a`). Hashing at the configured
      // size while the pool pages smaller produces hashes no lookup can hit.
      block_hasher_(prefix_caching_enabled_
                        ? vllm::v1::get_request_block_hasher(
                              hash_block_size_, vllm::v1::sha256_cbor,
                              resolved_spec_config_.has_value() &&
                                  resolved_spec_config_->method == "mtp")
                        : nullptr),
      engine_(input_processor_, engine_core_, output_processor_, block_hasher_) {
  (void)hash_ready_;
  // ENG-ATTENTION-WINDOW W3 (#2388): install the model-level sliding-window
  // switch, mirroring `ModelConfig.disable_sliding_window`
  // (`vllm/config/model.py:248` @ `5559679229`). UNCONDITIONAL, so a second load
  // in one process overwrites the first model's answer instead of inheriting it
  // -- the same hazard the MoE placement plan carried until #2382, and the same
  // fix. `value_or(false)` is upstream's own default.
  //
  // NO ENVIRONMENT VARIABLE, deliberately. This replaces `VT_GEMMA2_SLIDING` and
  // `VT_GEMMA3_SLIDING`, and adding a third env spelling would re-create the
  // surface W3 exists to remove. Upstream has no env var here to mirror.
  vllm::SetDisableSlidingWindow(params.disable_sliding_window.value_or(false));
  // FOUR consumers page at `block_size_`: the KV config, the max-model-len fit,
  // the scheduler's block table, and the prefix-cache hasher. Before the floor
  // existed each spelled the same fallback expression and so could not disagree;
  // with a floor they can, and the failure is silent -- a pool paged at 256 read
  // through a block table striding 32 attends over the wrong tokens and returns
  // plausible output. Assert what the disagreement would look like. Written as a
  // property of the built config rather than a repetition of the assignment,
  // because repeating the assignment would gate nothing.
  for (const auto& group : kv_cfg_.kv_cache_groups) {
    if (!group.kv_cache_spec) continue;
    VT_CHECK(group.kv_cache_spec->block_size <= block_size_,
             std::string("kv-cache: group pages ") +
                 std::to_string(group.kv_cache_spec->block_size) +
                 " tokens but the engine's block table strides " +
                 std::to_string(block_size_) +
                 ". A page wider than the stride is read as the wrong tokens, "
                 "not as an error.");
  }
  // issue #371: REFUSE an unservable recurrent-state budget instead of
  // allocating it. Speculation widens the Mamba/GDN state to k+1 snapshot slots
  // per sequence (runner.cpp:449-451), so a k=15 draft costs SIXTEEN times the
  // spec-off state; on a unified-memory box the resulting allocation takes the
  // machine down rather than failing, which is exactly what it did four times on
  // 2026-08-11. Upstream checks the equivalent budget up front and raises
  // (kv_cache_utils.py:751-787, with MambaSpec counting num_speculative_blocks at
  // kv_cache_interface.py:713-718); this is that check for the state term.
  //
  // An UNKNOWN budget (MemAvailable unreadable) never refuses.
  {
    // The RESOLVED concurrency, not the configured one: the guard must weigh
    // what the runner allocates (issue #1983). ResolveMaxNumSeqs is the single
    // source of truth for that number.
    const int seqs = max_num_seqs_;
    const int64_t state_needed =
        vllm::v1::recurrent_state_bytes(kv_cfg_, seqs);
    const int64_t host_available = vllm::v1::host_available_memory_bytes();
    if (state_needed > 0 && host_available > 0) {
      vllm::v1::check_enough_state_memory(
          host_available, state_needed, seqs,
          resolved_spec_config_.has_value()
              ? resolved_spec_config_->ResolvedNumSpeculativeTokens()
              : 0);
    }
  }
  // SPEC-DFLASH D5: wire the separately-loaded DFlash draft into the runner's
  // verify/propose loop. Done here (after runner_ is fully constructed, before
  // WarmupKernels) so the runner holds a stable borrow of dflash_draft_ (which
  // outlives it). Null for mtp/non-spec, so this is inert on every other path.
  if (dflash_draft_ != nullptr) {
    // Both block drafters CONDITION on the target's aux multi-tap. A target
    // architecture whose forward cannot produce it yields an engine that dies on
    // the first propose with "missing target aux multi-tap" -- which is exactly
    // how the first DSpark e2e failed, against classic-dense Qwen3ForCausalLM.
    // Refuse at LOAD, by name, with the reason.
    if (!model_->supports_aux_multi_tap()) {
      const std::string method =
          dflash_draft_->dspark != nullptr ? "dspark" : "dflash";
      const std::string arch = config_.architectures.empty()
                                   ? std::string("this model")
                                   : config_.architectures.front();
      throw std::runtime_error(
          "speculative-config: method \"" + method +
          "\" needs a target architecture that captures the aux multi-tap (the "
          "residual stream at the draft's target_layer_ids); " + arch +
          " does not. Supported targets today are the Qwen3.5/3.6 dense and MoE "
          "families.");
    }
    if (dflash_draft_->dspark != nullptr) {
      // SPEC-DSPARK W5: wires the inherited backbone through set_dflash_draft
      // internally, so the shared machinery is byte-identical to the DFlash lane.
      runner_.set_dspark_draft(dflash_draft_->dspark.get(), &dflash_draft_->config,
                               dflash_draft_->k,
                               dflash_draft_->sample_from_anchor);
    } else {
      runner_.set_dflash_draft(&dflash_draft_->weights, &dflash_draft_->config,
                               dflash_draft_->k);
    }
  }
  // KV-EXTERNAL-CACHE (LMCache): build + wire the external KV connector when the
  // caller selected one via EngineParams::kv_transfer_config. Default (unset)
  // leaves kv_connector_ null and BOTH the scheduler and the runner unchanged —
  // byte-identical to the production engine. Built here (ctor body), after
  // runner_ and scheduler_ are fully constructed, so the runner's KV geometry is
  // available for the connector's chunk layout.
  kv_connector_ = BuildKvConnector(params, runner_);
  if (kv_connector_ != nullptr) {
    scheduler_->set_kv_connector(kv_connector_.get());
    runner_.set_kv_connector(kv_connector_.get());
    std::cerr << "vllm.cpp: KV external cache connector '"
              << *params.kv_transfer_config->kv_connector
              << "' wired ON (scheduler + worker)\n";
  }
  // Mirror vLLM's "Asynchronous scheduling is enabled/disabled" resolution log
  // (vllm/config/vllm.py:990-1038) so the DGX A/B can audit which arm is live.
  std::cerr << "vllm.cpp: Asynchronous scheduling is "
            << (async_scheduling_enabled_ ? "enabled" : "disabled")
            << " (max_concurrent_batches=" << max_concurrent_batches_ << ")\n";
  // SPEC-DFLASH2 W7 (#1824): tell the runner which scheduling mode resolved —
  // under async + a speculator it fills the scheduler's -1 draft placeholders
  // from its own propose and corrects the optimistic num_computed_tokens.
  // Before any step runs (WarmupKernels below is the first).
  runner_.set_async_scheduling(async_scheduling_enabled_);
  WarmupKernels();
  // W4d W3: the ctor's cold pre-warm committed full-batch-shape activation
  // slots that the captured decode never reads. Recipe-gated release —
  // here, at the end of BOTH LoadedEngine ctors, because the GGUF dense
  // load reaches the engine through more than one loader branch.
#if defined(VLLM_CPP_TENSTORRENT)
  if (const char* wr = std::getenv("VT_TT_RELEASE_WARM_ROWS");
      wr != nullptr && wr[0] != '\0') {
    vt::tenstorrent::ReleaseWarmShapeSlots(
        static_cast<uint32_t>(std::strtoul(wr, nullptr, 10)));
  }
#endif
}

void LoadedEngine::WarmupKernels() {
#if defined(VLLM_CPP_CUDA) && defined(VT_CUTLASS_NVFP4)
  if (!model_->uses_nvfp4_w4a4() ||
      !vllm::platforms::GetPlatform(runner_.device().type).cutlass_fp4_supported() ||
      !EnvironmentEnabled("VT_FP4_PRE_SERVE_WARMUP") ||
      !EnvironmentEnabled("VT_FP4_AUTOTUNE") ||
      !EnvironmentEnabled("VT_FP4_PLAN_CACHE")) {
    return;
  }

  int32_t dummy_token = -1;
  for (int32_t token = 0; token < tokenizer_.VocabSize(); ++token) {
    if (tokenizer_.HasToken(token) && !tokenizer_.IsSpecial(token)) {
      dummy_token = token;
      break;
    }
  }
  if (dummy_token < 0) {
    throw std::runtime_error(
        "NVFP4 pre-serve warmup could not find a non-special tokenizer token");
  }

  SamplingParams sampling;
  sampling.max_tokens = 1;
  sampling.temperature = 0.0;
  sampling.ignore_eos = true;
  sampling.PostInit();
  std::vector<int32_t> prompt(
      static_cast<size_t>(max_num_batched_tokens_), dummy_token);

  std::cerr << "vllm.cpp: warming FlashInfer-parity NVFP4 profiles at "
            << max_num_batched_tokens_ << " tokens before serving\n";
  vt::cuda::Nvfp4AutotuneWarmupScope warmup(
      static_cast<uint32_t>(max_num_batched_tokens_), runner_.device().index);
  engine_core_.add_request(std::make_unique<vllm::v1::Request>(
      "_vllm_cpp_nvfp4_warmup", std::move(prompt), std::move(sampling),
      /*arrival_time=*/0.0, /*block_hasher=*/nullptr));
  while (scheduler_->get_num_unfinished_requests() > 0) {
    (void)engine_core_.step();
  }
  if (scheduler_->has_finished_requests()) {
    (void)engine_core_.step();
  }
  if (scheduler_->get_num_unfinished_requests() != 0 ||
      scheduler_->has_finished_requests()) {
    throw std::runtime_error(
        "NVFP4 pre-serve warmup left scheduler state behind");
  }
  warmup.Complete();
#endif
}

vllm::v1::AsyncLLM& LoadedEngine::async_engine() {
  std::lock_guard<std::mutex> lock(async_engine_mutex_);
  if (async_engine_ == nullptr) {
    // Thread the resolved depth-2 batch-queue size so the async frontend runs
    // step_with_batch_queue under async scheduling (W3 enable-flip); the sync
    // default keeps max_concurrent_batches_ == 1 (depth-1 step()).
    async_engine_ = std::make_unique<vllm::v1::AsyncLLM>(
        input_processor_, *scheduler_, executor_, output_processor_,
        block_hasher_, /*shutdown_timeout_s=*/0, max_concurrent_batches_,
        &structured_output_manager_,
        // The speculative-decode flag EngineCoreProc needs to run post_step.
        // Without it every speculator's drafts were proposed and dropped on
        // this (the production CLI/server) path.
        /*check_for_draft_tokens=*/resolved_spec_config_.has_value());
  }
  return *async_engine_;
}

std::unique_ptr<LoadedEngine> LoadedEngine::FromModelDir(
    const std::string& model_dir, const EngineParams& params_in) {
  // SPEC-DRAFTER-CHAIN W1 (#1522): refuse a drafter chain BEFORE anything else
  // this function does.
  //
  // G5 requires the refusal to land "before any weight I/O", and the ordering is
  // the requirement rather than tidiness. Below this line the function reads the
  // checkpoint's quantization config, installs two process-global
  // configurations, resolves a device, stats the model directory, parses a
  // config, builds a tokenizer and maps weights. A chain config cannot produce
  // an engine, so every one of those is work spent on a load that will not
  // happen, and the two installs are not free, because one of them latches
  // decisions that a later engine in the same process cannot retake.
  //
  // It delegates to `ResolveSpecConfig` rather than repeating the test, so there
  // is ONE chain refusal with ONE message. `ResolveSpecConfig` runs again in the
  // constructor further down, which is what refuses a chain on the loader paths
  // that do not come through this function.
  //
  // It reads `params_in`, the UNRESOLVED argument, deliberately: it must run
  // ahead of the KV-FP8 W3 stanza below, whose `ReadQuantConfigJson` opens a
  // file inside `model_dir`. The case
  // "kv-fp8 W3 G10: the drafter-chain refusal runs BEFORE the KV resolution" in
  // `tests/vllm/entrypoints/test_kv_cache_fp8_wiring.cpp` gates that order, by
  // pointing at a directory that EXISTS and declares fp8 -- which
  // `test_drafter_chain_reach.cpp` cannot do, because its nonexistent path makes
  // `ReadQuantConfigJson` answer "" without opening anything.
  if (params_in.speculative_config.has_value() &&
      params_in.speculative_config->use_drafter_chain()) {
    (void)ResolveSpecConfig(params_in, vllm::HfConfig{});
  }
  // KV-FP8 W3: resolve `--kv-cache-dtype` against the CHECKPOINT once, here,
  // before any consumer reads it. Upstream does exactly this and in exactly this
  // position: `EngineArgs.create_engine_config` calls
  // `resolve_kv_cache_dtype_string(self.kv_cache_dtype, model_config)` and hands
  // the RESULT to `CacheConfig(cache_dtype=...)` (`arg_utils.py:1915-1929`), so
  // nothing downstream of the config ever sees the unresolved "auto".
  //
  // `params` shadows the argument from here on, so every later reference in this
  // function reads the resolved value and no call site had to change. The only
  // field that differs is `kv_cache_dtype`.
  //
  // THIS IS THE ONLY PLACE A CHECKPOINT'S DECLARATION IS READ. The direct
  // `LoadedEngine(config, weights, ...)` constructors take `kv_cache_dtype`
  // verbatim, which is right: they are handed weights that are already in
  // memory and there is no checkpoint directory to ask.
  EngineParams params = params_in;
  {
    const vllm::ResolvedCacheDTypeString resolved =
        vllm::ResolveKvCacheDTypeString(params.kv_cache_dtype,
                                        vllm::ReadQuantConfigJson(model_dir));
    if (resolved.declared_by_checkpoint) {
      // Say it. An operator who typed no flag and gets a quantized KV cache
      // because the checkpoint asked for one deserves to read that sentence
      // rather than infer it from a block count that doubled.
      std::cerr << "engine: the checkpoint declares kv_cache_quant_algo -> "
                   "--kv-cache-dtype "
                << resolved.cache_dtype
                << " (pass an explicit --kv-cache-dtype to override)"
                << std::endl;
    }
    params.kv_cache_dtype = resolved.cache_dtype;
  }
  // ENG-RESIDENCY-CONFIG (#1110): install the host-RAM -> DISK residency config
  // FIRST — before the offloader below, before the device resolution, before any
  // path or weight operation.
  //
  // The ordering is the whole requirement, not tidiness. Each knob this config
  // feeds (`VT_GGUF_MMAP`, `VT_GGUF_PREFAULT`, `VT_MOE_EXPERT_STREAM` and its two
  // sizes) is read during weight load, and two of them cannot be taken back: the
  // expert-stream decision is cached in a function-local static, and the slot
  // store's geometry is fixed when the store is built. A config that arrived after
  // either would be silently ignored, which is why `SetWeightResidencyConfig`
  // throws when a document would CHANGE one of those two decisions rather than
  // accepting it. It throws on nothing else: a second engine in one process is legal,
  // so a late `mmap` or `prefault` (both resolved per load), a document that omits a
  // decided field, and a document that asks for what was decided are all installed,
  // and the install MERGES field by field so a partial document does not drop the
  // first engine's. It is placed ahead of `CreateWeightOffloader`
  // deliberately:
  // that call can THROW for a configured-but-unwired backend, and a document
  // carrying both tiers must still have installed its residency half first.
  //
  // Absent (the default, and every caller that predates this row) installs
  // nothing, so every knob resolves exactly as it did before.
  if (params.weight_residency.has_value() &&
      !params.weight_residency->empty()) {
    vllm::SetWeightResidencyConfig(*params.weight_residency);
    // ONE line, naming THE DOCUMENT THAT WAS INSTALLED — the fields the operator
    // set, not the values the engine will resolve. The two differ exactly when a
    // variable overrides the document, which is why the second line exists: it
    // names every variable that would WIN over a field of it, by variable and by
    // field. The environment deliberately wins (those variables exist so a
    // benchmark arm is switchable without a restart), and a document silently
    // overridden by something exported weeks ago is the one way that precedence
    // hurts.
    //
    // It does not print RESOLVED values, and ONE of the five is the reason:
    // `expert_stream` is cached on first read, so resolving it here would move that
    // decision ahead of the load — the exact ordering this block exists to hold. The
    // other four could be resolved at this point (`prefault` and `slots` outright;
    // `mmap` and `slot_bytes` need a built-in default only their caller has), so
    // printing the document rather than a mixture of asked-for and resolved values is
    // a consistency decision on top of that one constraint. An operator reading
    // `expert_stream=on` beside `VT_MOE_EXPERT_STREAM (...) OVERRIDES` is being told
    // the document said on and the variable decides.
    //
    // IT READS BACK THE INSTALLED GLOBAL, not `params`, and that is what makes the
    // line evidence that the install RAN. Measured: with the line printing from
    // `params`, the reachability mutation — deleting the `SetWeightResidencyConfig`
    // call above — left the server-level suite GREEN, because the log and the
    // install were independent statements.
    const vllm::WeightResidencyConfig installed =
        vllm::ActiveWeightResidencyConfig();
    if (!installed.empty()) {
      std::cerr << "engine: weight residency (offload_config vllm_cpp): "
                << installed.Describe() << std::endl;
      const std::string shadowed = installed.DescribeEnvOverrides();
      if (!shadowed.empty()) {
        std::cerr << "engine: weight residency: the environment OVERRIDES the "
                     "config for "
                  << shadowed << std::endl;
      }
    }
    // ENG-HYBRID-PLACEMENT (#2018): the one pairing inside this key that is legal
    // and slow rather than legal and fine. Printed unconditionally of
    // `installed.empty()`, because the placement can come from the environment
    // alone, in which case the installed DOCUMENT is empty and the collision is
    // still real.
    const std::string collision = vllm::DescribePlacementResidencyCollision();
    if (!collision.empty()) {
      std::cerr << "engine: weight residency: " << collision << std::endl;
    }
  }
  // ENG-WEIGHT-OFFLOAD W1: install the weight offloader BEFORE any weight I/O,
  // mirroring vLLM setting the process-global at
  // v1/worker/gpu_model_runner.py:939. `ModelRegistry::Prepare` reads it back
  // (our analogue of models/utils.py:824). An absent config installs the no-op,
  // which is the current engine path, so this is inert by default.
  //
  // W1 has no backend, so a config that asks for one gets the no-op and ONE
  // line saying so. A configured `cpu_offload_gb` that silently frees nothing
  // is a memory bug the operator cannot see, and an honest line is cheaper than
  // the bug report it prevents.
  {
    vllm::WeightOffloaderChoice choice = vllm::CreateWeightOffloader(
        params.offload_config.value_or(vllm::OffloadConfig{}));
    if (!choice.selected_backend_pending.empty()) {
      // A backend this build cannot construct at all.
      std::cerr << "engine: offload_config selected backend \""
                << choice.selected_backend_pending
                << "\" but this build has no offloader for it "
                   "(ENG-WEIGHT-OFFLOAD W5); NO weight is offloaded"
                << std::endl;
    } else if (choice.offloader != nullptr && choice.offloader->moves_weights()) {
      // A backend that IS constructed but that no loader consults yet. The
      // distinction matters to whoever set a budget: the first case is "not
      // built", the second is "built and not wired". Both offload nothing, and
      // saying so differently is what lets a bug report name the right one.
      std::cerr << "engine: offload_config installed " << choice.offloader->name()
                << " but no loader consults it yet "
                   "(ENG-WEIGHT-OFFLOAD W2c); NO weight is offloaded"
                << std::endl;
    }
    vllm::SetWeightOffloader(std::move(choice.offloader));
  }
  // ARCH-ONE-SURFACE ROW 8: resolve an EXPLICIT device selection up front,
  // BEFORE any path/config/weight I/O — the mirror of vLLM resolving
  // DeviceConfig at config-creation time, ahead of the model load
  // (vllm/engine/arg_utils.py:1878 builds DeviceConfig first;
  // device.py __post_init__ resolves immediately). An explicitly named absent
  // device therefore fails HERE, loudly, and is never masked by a later
  // path/tokenizer error. The result is discarded: SelectQueueForModel re-runs
  // the SAME ResolveExplicitDeviceType when it actually creates the queue, so
  // the policy has exactly one owner.
  if (params.device != vllm::Device::kAuto) {
    const vllm::platforms::Platform* named_platform =
        vllm::platforms::FindPlatformByName(vllm::DeviceName(params.device));
    // ENG-HYBRID-PLACEMENT W2 (#2023): the EXPLICIT device is fully known here,
    // because `ResolveModelDeviceType` ignores the architecture on this branch, so
    // the placement can be resolved and reported ahead of all path and weight I/O
    // — which is where an operator still reads the line when the load then fails.
    ReportDevicePlacement(ResolveModelDeviceType(/*architecture=*/"",
                                                 params.device));
    (void)ResolveExplicitDeviceType(
        params.device, named_platform == nullptr
                           ? std::nullopt
                           : std::optional{named_platform->device_type()});
  }
  const fs::path dir(model_dir);

  // SPEC-DFLASH2 W1 (#1314): classify a DFlash draft here, BEFORE any path,
  // config, tokenizer or weight I/O, for the same reason
  // SPEC-DSPARK-BLOCK-SIZE-GUARD resolves the DSpark config up front further
  // down: the dflash draft load below runs BEFORE the LoadedEngine constructor
  // reaches ResolveSpecConfig and resolves from the CLI config directly, so
  // without this the constructor's refusal would arrive after the draft
  // checkpoint had already been read through the DFlash1 loader. Both target
  // containers pass through this line, and the `.gguf` branch immediately below
  // returns before the later one. This is a REFUSAL and not a second resolution:
  // it calls the same `CheckDflash2DraftArm` the constructor's `ResolveSpecConfig`
  // calls and decides nothing else, so the classification keeps one owner and
  // one message.
  if (params.speculative_config.has_value() &&
      params.speculative_config->method == "dflash" &&
      params.speculative_config->draft_model_path.has_value()) {
    CheckDflash2DraftArm(*params.speculative_config->draft_model_path);
  }

  // LOAD-GGUF-MMPROJ (#821): a projector is a SECOND GGUF beside a GGUF
  // language file, and nothing else. A safetensors checkpoint carries its
  // vision tower in its own shards, so accepting --mmproj there and quietly
  // ignoring it would load a tower the user did not ask for and drop the one
  // they named. Refuse by name instead, before any path or config I/O.
  if (!params.mmproj_path.empty() &&
      !(fs::is_regular_file(dir) && dir.extension() == ".gguf")) {
    throw std::runtime_error(
        "--mmproj: a multimodal projector attaches to a .gguf language file, "
        "and '" +
        model_dir +
        "' is not one. A safetensors checkpoint carries its vision tower in "
        "its own shards and needs no projector file");
  }

  // A single `.gguf` file: config + weights + tokenizer all come from the
  // GGUF (M0.10). The engine stack below is unchanged.
  if (fs::is_regular_file(dir) && dir.extension() == ".gguf") {
    // LOAD-IO: `VT_LOAD_STATS=1` reported NOTHING on a GGUF load. The two
    // ReportLoadPhase call sites both sat on the safetensors branch, so the one
    // instrument this repository has for "where did the load time go" was silent
    // on every `.gguf` model -- and silence reads as "no phases", not as "not
    // measured". That is the instrument-failure-looks-like-a-result shape, and it
    // cost a 74-minute load its breakdown (row MODEL-MM-QWEN4-EXP, W5n).
    const auto t_gguf_open = std::chrono::steady_clock::now();
    vllm::GgufFile gguf = vllm::GgufFile::Open(model_dir);
    ReportLoadPhase("mmap+header", SecondsSince(t_gguf_open));
    HfConfig config = HfConfigFromGgufDispatch(gguf);
    // Resolve before tokenizer/weight work so unsupported architecture errors
    // are deterministic and match registry.py rather than being masked by a
    // later source-specific missing-tensor/tokenizer error.
    const ModelRegistration& gguf_arch = ModelRegistry::Resolve(config);
    // ENG-GGUF-RESIDENCY-RESOLVED-DEVICE F2: resolved ONCE, here, and carried.
    //
    // `ResolveModelDeviceType` is NOT pure on `--device auto`: `ResolveAutoDevice`
    // decides by ATTEMPTING `CreateQueue()` and answers `kCPU` when that throws.
    // Between the fit check below and the `ModelSource` further down, this
    // function opens the tokenizer and the mmproj vision tower, so host memory
    // grows; `cudaStreamCreate` failing after exactly that growth is documented
    // on this project's own target box (examples/laguna_gen/main.cpp:181).
    //
    // Two calls could therefore disagree WITHIN ONE LOAD: the fit check bounds a
    // CUDA load, every registry then builds a CPU policy and keeps the n-gram
    // table block-resident, and the runner's own resolution hands the forward a
    // CUDA queue whose `EmbeddingKernelCuda` cannot decode blocks — the exact
    // first-forward throw with the model fully resident that
    // `DeviceQuantGatherSupported` exists to prevent. One resolution cannot
    // disagree with itself.
    //
    // The MoE placement plan below was a THIRD call to the same function when
    // #2314 landed under this row. It takes the carried value for the same
    // reason the other two do: a placement plan installed for one device while
    // the residency policy resolves another is the same class of defect this
    // row exists to remove.
    const vt::DeviceType gguf_device =
        ResolveModelDeviceType(gguf_arch.architecture, params.device);
    // #2314: before ANY weight I/O, so a CPU-placed layer is never staged onto
    // the device first — `ResidentWeight` aliases host bytes on a CPU `Dev` and
    // uploads otherwise, so this ordering is what makes the placement free
    // rather than a round trip.
    InstallMoePlacementPlan(gguf_device, config.num_hidden_layers, &gguf);
    // Issue #1123: refuse a GGUF whose weights cannot be STAGED onto the target
    // device, here, before any weight I/O and before the tokenizer.
    //
    // `Qwen3.8-2.4T-A95B UD-Q1_0` (369.96 GiB) reached a serving state on
    // `--device cuda` on a 119.631 GiB GB10 after 26 minutes and then died on
    // the FIRST forward with `vt cuda: cudaMalloc: out of memory`. The load
    // succeeds because a keep-quant expert tower is BORROWED from this mapping
    // and costs zero anonymous bytes; the forward dies because a
    // weight-staging device copies each tower into device memory
    // (`ResidentWeight`, qwen3_5.cpp:1011 -- 276 towers of 1,275,068,416
    // bytes plus 3 of 2,818,572,288, so 335.62 GiB). Loading for 26 minutes and dying
    // mid-stream is the worst of the available behaviours.
    //
    // Placed AFTER Resolve so an unsupported-architecture error keeps its
    // priority and the error ordering this branch documents is unchanged, and
    // BEFORE the tokenizer and the weights because everything after this point
    // is the cost the refusal exists to avoid paying. The predicate lives in
    // `gguf_device_fit.h`; it decides nothing on a platform that does not stage
    // weights (every CPU load) and nothing when no budget is known.
    {
      const platforms::Platform& target = platforms::GetPlatform(gguf_device);
      // ENG-EXPERT-STREAM-DEVICE W0d (issue #1124). The bound above sums the
      // WHOLE tensor table, so on `Qwen3.8-2.4T-A95B UD-Q1_0` it counts all
      // 335.62 GiB of `*_exps` and refuses before any forward exists to take the
      // slot arm. That is right whenever those towers really are staged, and
      // wrong when the streaming lane serves them: then they are not staged at
      // all, and what the device pays instead is the slot arena.
      //
      // THE FOUR CONDITIONS, in this order because the last one LATCHES.
      // `ResolveExpertStreamRequested()` fixes the process's answer for good, so
      // it is asked only once the platform has already said it both stages and
      // can read host slots, and only once the resolved MODEL has said its
      // forward reads experts through the slot seam — which is false on every CPU
      // load, on every discrete device and on every architecture that does not
      // stream, i.e. everywhere the latch would be a side effect rather than the
      // question. The offload config is installed at the top of this function,
      // well before here, so the latched answer is the configured one.
      //
      // THE ARCHITECTURE TERM IS NOT COSMETIC. Without it this block is keyed on
      // the tensor NAME alone, and `_exps.weight` is what a llama.cpp MoE export
      // writes for families this tree does not stream:
      // `deepseek_v4_weights.cpp` and `laguna_weights.cpp` both emit it, and
      // neither model composes `RunMoeBlock`, so neither ever reaches
      // `KqExpertSlice`. `CheckDeviceWeightFit` has ONE production call site —
      // this one — and every GGUF architecture arrives at it, so a `deepseek4`
      // load on a GB10 with `VT_MOE_EXPERT_STREAM=1` would have had its whole
      // expert set dropped from the bound and an arena added that nothing
      // allocates. That is the UNSAFE direction, unlike the two over-counts this
      // bound documents (#1136): it deletes a correct refusal and puts back the
      // 26-minute load and the `cudaMalloc: out of memory` first forward that
      // #1123 exists to prevent. The capability is declared on the factory beside
      // the forward that implements it, and a model that does not declare it gets
      // the whole bound.
      //
      // THE RESIDENCY TERM IS THE SAME FINDING ONE LEVEL DEEPER (#1378). The four
      // conditions above are all properties of the DEVICE and the ARCHITECTURE,
      // and a `qwen35moe` GGUF satisfies every one of them while still staging
      // every tower: `LoadExpertsOrNvfp4` routes the SAME `_exps.weight` tensors
      // to `expert_*_fp4` or to `expert_*` whenever the residency is not a keep
      // residency, and only the `expert_*_kq` arm reaches `KqExpertSlice`. With
      // `VT_GGUF_KEEP_QUANT=0`, or on an NVFP4 GGUF, this block therefore dropped
      // 335.62 GiB of towers from the bound on a load that stages all of them and
      // deleted the #1123 refusal outright. `GgufExpertTowersReachSlotLane` asks
      // the model loader's own routing function about this file under this
      // process's policy, so the bound and the forward cannot disagree.
      //
      // It is asked LAST, after `ResolveExpertStreamRequested()`, only because
      // that call LATCHES: moving a non-latching file scan in front of it would
      // change which loads fix the process's streaming answer, and this repair
      // has no business moving that.
      //
      // GGUF-DEVICE-FIT-EXPAND-POLICY (#1870): resolved ONCE and reused below by
      // `CheckDeviceWeightFit` as well, so the two calls that ask this process's
      // residency policy about this file can never resolve two different
      // answers to the same `getenv` reads.
      //
      // ENG-GGUF-RESIDENCY-RESOLVED-DEVICE: from `target.device_type()`, which
      // is `gguf_device` — the load's ONE resolution, the same value the fit
      // check is bounded with and the same value the `ModelSource` carries. It used to be `FromEnv()`, which probed
      // `platforms::CurrentPlatform()` — so on a CUDA-capable process an
      // explicit `--device cpu` bounded a CPU load with the CUDA residency
      // policy. That is #1136's finding one level down: the bound and the
      // policy the bound describes must name the same device.
      const GgufLoadPolicy gguf_load_policy =
          GgufLoadPolicy::FromEnv(target.device_type());
      static constexpr std::string_view kStreamedExpertSuffix = "_exps.weight";
      StreamedExpertLane lane;
      // BACKEND-ROCM-LANE-GUARD (#2507): `allocates_bounded_device_memory()`,
      // for the SAME reason the refusal below reads it, and this guard has to
      // read the same one BECAUSE the lane is that refusal's own exemption.
      // Towers the lane serves leave the bound and the arena enters it; keying
      // the two halves of one sum on two predicates lets them disagree, and on
      // ROCm they deliberately do. `needs_weight_staging()` is false there and
      // `allocates_bounded_device_memory()` is true, so the load drew the one
      // combination that loses: the refusal fired and the exemption did not,
      // and GLM-5.3's `UD-IQ1_S` was charged all 187.3125 GiB of the towers it
      // declares it streams. On CUDA both are true, which is why the same
      // checkpoint has been generating on a GB10 throughout.
      //
      // THE TWO QUESTIONS THIS GUARD ASKS, and neither is answered by
      // `needs_weight_staging()`:
      //
      //   Is a fit computation running at all? `CheckDeviceWeightFit` returns
      //   before computing anything when its gating argument is false, so a
      //   lane built for such a load is pure side effect — `Reserve` fixes the
      //   slot store's geometry for the process and `ResolveExpertStreamRequested`
      //   latches its streaming answer, and neither belongs on a load whose fit
      //   check is inert. Only the predicate the fit check is KEYED on can
      //   answer this, which is what makes the choice forced rather than
      //   selected.
      //
      //   Will the lane actually serve? That is the SECOND term, unchanged, and
      //   it is unchanged because it was already right: `ExpertSlice`
      //   (`expert_stream_seam.cpp`) admits the lane at RUNTIME on
      //   `cpu || host_memory_is_device_addressable()`, with no staging term.
      //   The load-time guard and the runtime seam now read the same predicate,
      //   which is the property that stops the bound and the forward
      //   disagreeing about who serves a tower.
      //
      // `needs_weight_staging()` answers a THIRD question that is not asked
      // here: should the fully-optimized device-resident forward run — the
      // indexed GDN state I/O, the merged/packed GDN projections, the fp8/bf16
      // GDN resident prep. It stood in for the first question only because
      // before #1934 there was no separate predicate for it to stand on. That
      // row built one and moved the refusal onto it; this one moves the
      // exemption. NO platform's answer to either predicate changes here.
      //
      // NOT a `||`. That would admit ROCm as well and would be wrong: it
      // re-admits a platform that stages but reports no bounded pool, for which
      // the fit check computes nothing and the lane is again pure side effect.
      // `test_gguf_device_fit_reach` pins the plain predicate in both
      // directions.
      if (target.allocates_bounded_device_memory() &&
          target.host_memory_is_device_addressable() &&
          gguf_arch.factory != nullptr &&
          gguf_arch.factory->streams_routed_experts &&
          ResolveExpertStreamRequested() &&
          GgufExpertTowersReachSlotLane(gguf, kStreamedExpertSuffix,
                                        gguf_load_policy)) {
        // `_exps.weight` is exactly the set `KqExpertSlice` streams: the stacked
        // `blk.<n>.ffn_{gate,up,down}_exps.weight` towers a llama.cpp MoE export
        // writes. The arena is the store's own arithmetic — `slots *
        // slot_bytes` through the same two resolvers `Qwen35ExpertStream`'s
        // constructor uses, with the largest per-expert slice in this file as
        // the computed default, so the number here is the number that gets
        // allocated and not an estimate of it.
        lane.tensor_name_suffix = kStreamedExpertSuffix;
        // MODEL-TEXT-GLM-MOE-DSA W3 (#2214, spec §3.3). The budget must hold
        // ONE decode step's whole slice working set, because every `Acquire`
        // marks its entry protected until `EndStep` clears it. Below that, the
        // cache does not fail: `Slice` returns nullptr and the caller reads the
        // tower IN PLACE out of the mmap, counted on stderr and reported as
        // success. On the model this row targets that is a 187 GiB random read
        // per token, and a benchmark measuring it would publish a page-cache
        // number under a streaming label.
        //
        // HERE, not inside the store's constructor, because "at load" is the
        // point of it: the constructor first runs on the FIRST expert slice of
        // the first forward, after the weights are read and the device pool is
        // built, which is exactly the 26-minute-then-die shape the refusal a few
        // lines above exists to avoid. This block is already the one place that
        // has both the file and the resolved budget.
        //
        // Reached on every streaming load, not only this row's: the geometry
        // comes from the file, so Qwen3.5 is checked by the same call. Streaming
        // is default OFF and this branch needs `ResolveExpertStreamRequested()`,
        // so a run that never asked for the lane cannot reach the refusal.
        const GgufExpertLaneGeometry lane_geom =
            GgufStreamedExpertLaneGeometry(gguf, lane.tensor_name_suffix);
        expert_stream::RequireSlotCapacity(
            std::string(gguf_arch.architecture), lane_geom.streamed_tower_count,
            lane_geom.experts_per_tok, ResolveExpertStreamSlots());
        const size_t slice =
            GgufLargestExpertSliceBytes(gguf, lane.tensor_name_suffix);
        // MODEL-TEXT-GLM-MOE-DSA (#2214, spec O33). RESERVE the same number the
        // bound below is about to charge the device for, so the arena
        // `CheckDeviceWeightFit` prices and the arena the store allocates cannot
        // be two different numbers.
        //
        // They WERE two different numbers, and it cost a step. `ExpertStreamLane`
        // is a process-lifetime singleton built on the FIRST slice anyone asks
        // for, and a model whose forward reserves per LAYER sizes it from
        // whichever layer runs first. On GLM-5.3's `UD-IQ1_S` that is `blk.3`,
        // whose `ffn_down_exps` is IQ3_XXS at 4,816,896 B; `blk.8`'s is IQ4_XS at
        // 6,684,672 B, so the store was built 28% too small and refused by name
        // mid-step after streaming 527 slices — measured on `dgx:gpu0`,
        // 2026-08-31. A `UD-*` arm mixes encodings ACROSS layers by design, so no
        // per-layer maximum is the model's maximum.
        //
        // Here rather than in a model, because this is the one place that has the
        // whole FILE. `Reserve` takes a maximum and is inert unless streaming was
        // requested, so a model that reserves for itself as well (Qwen3.5 does)
        // is unaffected except by being given a floor that is at least correct.
        if (slice > 0) expert_stream::ExpertStreamLane::Reserve(slice);
        if (slice > 0) {
          lane.arena_bytes =
              static_cast<size_t>(ResolveExpertStreamSlots()) *
              static_cast<size_t>(
                  ResolveExpertStreamSlotBytes(static_cast<int64_t>(slice)));
        }
      }
      // GGUF-DEVICE-FIT-EXPAND-POLICY (#1870): `RouteGgufTensor`'s own totality
      // guarantee ("anything else is kExpandBf16") makes this condition an
      // EXACT description of every tensor's residency, not a guess — see the
      // header on `GgufStagedWeightFootprint`. `cpu_ref` needs no term of its
      // own: it is a CPU-only oracle switch, and `allocates_bounded_device_memory`
      // — the predicate BOTH the lane guard above and the refusal below now read
      // (#2507) — already excludes every load it could apply to, because it
      // delegates to `needs_weight_staging()` on CPU and that is false there.
      // The exclusion this sentence names is unchanged; only the predicate that
      // provides it moved.
      const bool policy_forces_full_expand =
          GgufPolicyForcesFullExpand(gguf_load_policy);
      // BACKEND-ROCM (#1934): `allocates_bounded_device_memory()`, not
      // `needs_weight_staging()`. The two questions differ (see the interface
      // doc): this one asks whether `ResidentWeight` draws from a bounded
      // device pool at all -- true on every non-CPU platform since issue
      // #125's `is_cpu()` fix -- while `needs_weight_staging()` asks whether
      // the FULLY-OPTIMIZED device-resident forward (several GDN kernel
      // defaults) should run. Using the narrower predicate here is what makes
      // this refusal reachable on ROCm without moving any of the other one's
      // consumers; the row's spec records why that flag stays untouched.
      // #2517: THE PLAN INSTALLED 220 LINES ABOVE, credited here.
      //
      // `InstallMoePlacementPlan` runs at the top of this branch, before any
      // weight I/O, and announces what it placed. Until this argument existed
      // the refusal on the very next line quoted the UN-reduced footprint: one
      // load printed "56 layers run their routed experts on cpu ... to bring a
      // 216433205760 B footprint under a 68719476736 B budget" and then refused
      // needing 216433205760 B. Two lines of one load contradicting each other,
      // and on `strix:gpu0` the contradiction was the whole distance between a
      // placed GLM-5.3 load and a forward.
      //
      // The GLOBAL rather than a local, because `SetActiveMoePlacementPlan` is
      // where the plan lives and the forward reads the same object; a copy taken
      // here could be the one the seam never sees. It is installed
      // unconditionally, including when nothing is placed, so this is never a
      // stale plan from a previous load in the same process.
      const DeviceWeightFit fit = CheckDeviceWeightFit(
          gguf, vt::DeviceTypeName(target.device_type()),
          target.allocates_bounded_device_memory(),
          DeviceWeightBudgetBytes(
              target.residency_policy().device_memory_total_bytes),
          /*model_dtype_bytes=*/2, lane, policy_forces_full_expand,
          &vllm::ActiveMoePlacementPlan());
      if (fit.refuse) throw std::runtime_error(fit.message);
    }
    // QUANT-QWEN38-27B-GGUF-ARM (#821): refuse a qwen3_5-family GGUF carrying
    // tensors NOTHING in its loader reads, here, before the projector, the
    // tokenizer and every weight byte.
    //
    // The reason it is worth a refusal rather than a warning is one file.
    // `Qwen3.8-27B-Q4_K_M.gguf` states `qwen35.block_count = 65` and
    // `qwen35.nextn_predict_layers = 1`: 64 decoder blocks plus an MTP drafter
    // at `blk.64`. A reader that spends the whole 65 on the trunk gets a model
    // that loads, decodes fluently, and is the wrong graph — and the ten of
    // `blk.64`'s tensors that stop being read are the ONLY evidence of it,
    // because nothing downstream ever asks about a tensor it did not want.
    //
    // Placed AFTER the device-fit refusal so the existing error ordering this
    // branch documents is unchanged, and gated on the architecture the same way
    // `HfConfigFromGgufDispatch` gates its builders, so a family whose
    // enumeration lives elsewhere is not accounted against qwen3_5's.
    if (vllm::IsQwen3_5Gguf(gguf)) {
      vllm::RefuseUnaccountedQwen3_5Gguf(gguf, config);
      // KEEPQUANT W4a wave-3b-2 (#3030): the accounting above deliberately
      // PASSES a declared head, because its tensors ARE enumerated as
      // expected — and the trunk-only load below then leaves them unread.
      // Say so, by name, before any weight byte moves. The one config that
      // reads the head is speculative method "mtp" (the attach below); on
      // every other run the skip is real and the line is its record. The
      // pinned llama.cpp b10451 oracle ignores the same tensors, so a gate
      // against it is matched work only with this skip loud.
      const bool mtp_head_loads = params.speculative_config.has_value() &&
                                  params.speculative_config->method == "mtp";
      if (!mtp_head_loads) {
        vllm::LogQwen3_5GgufMtpHeadSkip(gguf, config);
      }
    }
    // LOAD-GGUF-MMPROJ (#821): the SECOND file. Opened, validated and READ
    // here — after the architecture resolve and the device-fit refusal, and
    // BEFORE the tokenizer and every weight byte — for the same reason the fit
    // refusal sits there: a projector this build cannot load must cost the user
    // a message, not a 17 GB map followed by one.
    //
    // llama.cpp's `--mmproj` is the user-facing convention this mirrors
    // (b10451 `tools/mtmd/mtmd-cli.cpp`); the file is a `clip`-architecture
    // GGUF and NOT a shard of the language file, which is why
    // `GgufFile::Open`'s own `DetectSplit` shard merge is not the seam.
    std::optional<multimodal::Qwen3VLVisionWeights> vision_tower;
    multimodal::Qwen3VLVisionConfig vision_config;
    // #607 L3: whether the read below was deliberately not done. Distinct from
    // `!vision_tower` — that is also true of every load that named no
    // `--mmproj`, and "there is no projector here" is not "the projector was
    // freed" (the same distinction the Muse Glimmer text-only case draws).
    bool mmproj_tower_skipped = false;
    std::optional<vllm::GgufFile> mmproj;
    // MODEL-MM-deepseek-v4 W4 (#2411): the `deepseek4v` arm, and the ONE thing
    // this function does for it is decide which reader gets the file.
    //
    // A DeepSeek-V4 projector is NOT read here. Its reader fills the W2 vision
    // types, and the tower has to end up on the LANGUAGE model rather than on
    // the engine, because the registered `encode_mm` hook reaches it through
    // `LoadedModel`. The engine's own `vision_tower_` is typed
    // `Qwen3VLVisionWeights`, which is exactly why it could not carry a second
    // architecture's tower. So the file travels down on `ModelSource::mmproj`
    // and `LoadDeepseekV4ForCausalLM` opens it -- a family's on-disk name map
    // living inside that family's `load_weights`, as everywhere else here.
    //
    // `RefuseUnsupportedClipMmproj` below keeps refusing `deepseek4v` and the
    // W3A gate keeps asserting that it does. Widening it would send a DeepSeek
    // projector into the Qwen3-VL reader and build a tower that runs and is
    // wrong; the branch is what stops it being reached at all.
    const vllm::GgufFile* deepseek_v4_mmproj = nullptr;
    if (!params.mmproj_path.empty()) {
      mmproj = vllm::GgufFile::Open(params.mmproj_path);
      if (vllm::IsClipMmprojGguf(*mmproj) &&
          vllm::ClipProjectorType(*mmproj) == vllm::kClipProjectorDeepSeekV4) {
        deepseek_v4_mmproj = &*mmproj;
        // REFUSE HERE, before the tokenizer and every weight byte, for the same
        // reason the Qwen3-VL arm below refuses here: a projector this build
        // cannot load must cost the user a message, not a 91 GiB map followed
        // by one. Only the READ is deferred, and it happens inside
        // `LoadDeepseekV4ForCausalLM` because the tower belongs on the model.
        multimodal::DeepSeekV4VisionConfig deepseek_v4_vision_config;
        vllm::RefuseDeepSeekV4ClipMmprojArm(*mmproj, params.mmproj_path,
                                            &deepseek_v4_vision_config);
      } else {
        vllm::RefuseUnsupportedClipMmproj(*mmproj, params.mmproj_path);
        vision_config = vllm::ClipMmprojVisionConfig(*mmproj);
        // QUANT-QWEN38-27B-GGUF-ARM (#821): the projector's own accounting, and
        // it runs BEFORE the read, so a file this reader would only partly
        // consume costs a message rather than a silently incomplete tower.
        vllm::RefuseUnaccountedClipMmproj(*mmproj, vision_config,
                                          params.mmproj_path);
        // #607 L3, the THIRD production tower load, and the one the first cut of
        // this row missed. It is a tower like the other two: `--mmproj` names a
        // projector, this reads every one of its tensors into owned host f32, and
        // the engine holds them for the process lifetime. So
        // `--language-model-only` zeroed every limit, refused every image request,
        // AND STILL PAID FOR THE PROJECTOR — the exact L2 failure this row exists
        // to close, surviving on the one architecture nothing was looking at.
        //
        // Gated on the same predicate and the same modality set as the two
        // safetensors sites ({"image","video"} — interfaces.py:293 and
        // qwen3_vl.py:1747), because this projector IS the Qwen3-VL tower, read
        // out of a `clip` GGUF instead of out of the model's own shards. `image:
        // 0` alone must therefore not skip it here either.
        //
        // ONLY THE READ IS CONDITIONAL. `GgufFile::Open`, both refusals and
        // `ClipMmprojVisionConfig` above still run: that is the construct half of
        // construct-without-initialise (utils.py:762), so the geometry resolves
        // either way and a `--mmproj` this build cannot load is still refused by
        // name rather than accepted in silence at zero limits. What stops is the
        // storage — and with it the reader's own missing-tensor refusals, which is
        // the mirror of `StageMissingLayer` keeping a skipped stage out of the
        // loader's key accounting (utils.py:693-695).
        //
        // `vision_tower` stays nullopt, which is already a supported engine state:
        // it is what every load that named no `--mmproj` produces.
        if (vllm::SkipTowerForModalities(&params.multimodal, {"image", "video"})) {
          mmproj_tower_skipped = true;
        } else {
          vision_tower =
              vllm::LoadQwen3VLVisionFromClipMmproj(*mmproj, vision_config);
        }
      }
    }
    tok::Tokenizer tokenizer = tok::Tokenizer::FromGguf(gguf);
    // Dense-vs-MoE GGUF dispatch now happens through the registry: the bench
    // branch's inline `IsDenseArch` split is superseded by
    // `HfConfigFromGguf` mapping the GGUF `general.architecture` key
    // (`qwen35` dense / `qwen35moe` / `qwen3next`) onto the registered
    // architecture ID, which resolves to the owning arch TU's GGUF loader.
    // SPEC-DFLASH-GGUF B2: dflash used to be REFUSED here, because the draft
    // shares the target's bf16 embed_tokens + lm_head and LoadDflashDraft read
    // them through a safetensors-TYPED seam. B1 replaced that parameter with
    // SharedHeadSource, which serves the same two tensors out of a GGUF, so the
    // refusal has no premise left and is gone. MTP left this ladder earlier for
    // its own reason - see below.
    //
    // SPEC-MTP-GGUF: MTP over GGUF used to be refused alongside dflash, on the
    // premise that GGUF exports carry no `mtp.*` tensors. They can: llama.cpp's
    // Qwen3.5 converter emits the head under layer-indexed `nextn` names and
    // announces it with `<arch>.nextn_predict_layers`, which HfConfigFromGguf
    // reads. A GGUF that genuinely lacks the head is still refused, but now for
    // the true reason and with the fix in the message.
    if (params.speculative_config.has_value() &&
        params.speculative_config->method == "mtp" &&
        !config.raw.contains("mtp_num_hidden_layers")) {
      throw std::runtime_error(
          "mtp speculative decoding needs a GGUF exported WITH the MTP head: "
          "this file declares no <arch>.nextn_predict_layers (it was converted "
          "with --no-mtp, or predates llama.cpp's Qwen3.5 MTP support)");
    }
    // #607 L3, THE PRODUCTION CALL SITE for the tower skip (one of three, all in
    // this function). The engine's multimodal limits are borrowed for the load,
    // so a loader that owns a tower can leave it uninitialised when every
    // modality it serves is at limit 0 — the mirror of upstream reading
    // `vllm_config.model_config.multimodal_config` inside the model's __init__
    // (interfaces.py:288-293). `params` outlives the call. Deleting this
    // assignment leaves the flag accepted and inert, which is exactly the
    // failure L2 recorded and L3 exists to close; test_tower_skip's reachability
    // case is the gate that catches it.
    // ENG-GGUF-RESIDENCY-RESOLVED-DEVICE: the RESOLVED device travels with the
    // source, so every GGUF registry hook builds its residency policy from what
    // the engine chose rather than from `platforms::CurrentPlatform()`. The
    // SAME VALUE the #1123 fit check above was bounded with — not a second call
    // to the same function, which on `--device auto` can answer differently
    // (see `gguf_device`).
    ModelSource gguf_source = ModelSource::FromGguf(gguf, gguf_device);
    gguf_source.multimodal = &params.multimodal;
    // MODEL-MM-deepseek-v4 W4 (#2411): hand the projector to the architecture's
    // own loader. Null for every other arm, and `ModelRegistry::Load` refuses a
    // non-null one whose registration does not declare `consumes_mmproj`, so a
    // DeepSeek projector named beside another family's language file costs a
    // message rather than a tower-free engine that answers images as text.
    gguf_source.mmproj = deepseek_v4_mmproj;
    if (deepseek_v4_mmproj != nullptr) {
      gguf_source.mmproj_path = params.mmproj_path;
    }
    const auto t_gguf_weights = std::chrono::steady_clock::now();
    TtAllocTraceStage("load/stage/pre-registry-load");
    std::unique_ptr<LoadedModel> model = ModelRegistry::Load(config, gguf_source);
    TtAllocTraceStage("load/stage/post-registry-load");
    ReportLoadPhase("weights", SecondsSince(t_gguf_weights));
    ReportGgufLoadIo();
    // SPEC-MTP-GGUF: attach the head from the SAME file, mirroring the
    // safetensors branch's maybe_attach_mtp. The GGUF is still mapped here; the
    // loader owns its dequantized copies, so nothing borrows past this scope.
    if (params.speculative_config.has_value() &&
        params.speculative_config->method == "mtp") {
      const ModelRegistration& gguf_reg = ModelRegistry::Resolve(config);
      const Qwen3_5MTPKind kind = gguf_reg.factory->is_dense_model
                                      ? Qwen3_5MTPKind::kDense
                                      : Qwen3_5MTPKind::kMoe;
      model->AttachMtpDraftWeights(vllm::LoadQwen3_5MTPFromGguf(
          gguf, config, kind,
          // The SAME resolved device the target's own load used, taken off the
          // source rather than resolved a second time.
          GgufLoadPolicy::FromEnv(gguf_source.device)));
    }
    // SPEC-DFLASH-GGUF B3: the axis-B wiring. Structurally the same three lines
    // as the safetensors branch's maybe_load_dflash - ResolveSpecConfig re-runs
    // on the target config inside the LoadedEngine ctor, so the draft path + k
    // are resolved here from the CLI config directly - and the ONE difference is
    // the shared-head source. The draft is loaded while `gguf` is still mapped,
    // and it copies out (its resolver owns its dequantized bf16), so nothing
    // borrows past this scope.
    // SPEC-DSPARK W5: a DSpark draft is a safetensors checkpoint at this pin.
    // A GGUF TARGET would otherwise silently produce a spec-ON engine with NO
    // draft (the propose path finds no weights and yields nothing), so refuse by
    // name. The GGUF draft axis is the DSpark analogue of SPEC-DFLASH-GGUF and is
    // tracked separately.
    if (params.speculative_config.has_value() &&
        params.speculative_config->method == "dspark") {
      throw std::invalid_argument(
          "speculative-config: method \"dspark\" needs a safetensors target at "
          "this pin (a GGUF DSpark draft/target axis is not ported yet)");
    }
    std::unique_ptr<DflashDraft> dflash;
    if (params.speculative_config.has_value() &&
        params.speculative_config->method == "dflash") {
      vllm::SpeculativeConfig resolved = vllm::SpeculativeConfig::ResolveDflash(
          params.speculative_config->ResolvedNumSpeculativeTokens());
      resolved.draft_model_path = params.speculative_config->draft_model_path;
      dflash = LoadDflashDraft(resolved, SharedHeadSource(&gguf));
    }
    TtAllocTraceStage("load/stage/pre-engine-ctor");
    std::unique_ptr<LoadedEngine> engine(new LoadedEngine(
        std::move(config), std::move(model), std::move(tokenizer), params,
        /*preselected_queue=*/nullptr, std::move(dflash),
        std::move(vision_tower), vision_config, mmproj_tower_skipped));
    TtAllocTraceStage("load/stage/post-engine-ctor");
    // W4d W3: the ctor's cold pre-warm committed full-batch-shape activation
    // slots that the captured decode never reads. Recipe-gated release.
#if defined(VLLM_CPP_TENSTORRENT)
    if (const char* wr = std::getenv("VT_TT_RELEASE_WARM_ROWS");
        wr != nullptr && wr[0] != '\0') {
      vt::tenstorrent::ReleaseWarmShapeSlots(
          static_cast<uint32_t>(std::strtoul(wr, nullptr, 10)));
    }
#endif
    return engine;
  }

  // SPEC-DSPARK-BLOCK-SIZE-GUARD (#1225): resolve the DSpark speculative config
  // ONCE, here, and hand the result to the draft load further down.
  //
  // The draft load used to resolve it a second time, with its own argument list,
  // and it runs BEFORE the LoadedEngine constructor reaches ResolveSpecConfig —
  // so that second copy, not this function, was the first resolution a DSpark run
  // ever met. It passed `ResolvedNumSpeculativeTokens()`
  // (`include/vllm/config/speculative.h::ResolvedNumSpeculativeTokens`), which is
  // `num_speculative_tokens.value_or(n_predict)` and therefore ZERO when the user
  // named no k, because nothing fills `n_predict` on the CLI-side config. Once the
  // block floor became reachable that refused an absent k against a k of 0, naming
  // a key the checkpoint does not carry and a number nobody typed, on the native
  // Qwen3 lane that works today. The `n_predict` default and the "requires
  // num_speculative_tokens" message were both unreachable in production for the
  // same reason, while this file's tests asserted them through ResolveSpecConfig.
  //
  // Delegating deletes the second implementation instead of repairing it: one
  // resolution, one set of messages, one place the floor is applied. The dspark
  // branch of `ResolveSpecConfig` reads nothing off the target `HfConfig` — it
  // resolves from the CLI config and the draft's own config.json — so the empty
  // config here yields exactly what the constructor's re-resolution against the
  // real one will yield.
  //
  // Placed AFTER the `.gguf` branch above, which keeps its own named refusal for
  // a GGUF target, and BEFORE every path, config, tokenizer and weight operation
  // below. A speculative length the draft cannot serve is then refused before the
  // loader spends twenty minutes mapping a target it will not get to use, which is
  // the same ordering the device resolution above exists to give.
  std::optional<vllm::SpeculativeConfig> dspark_spec;
  if (params.speculative_config.has_value() &&
      params.speculative_config->method == "dspark") {
    dspark_spec = LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{});
  }

  if (!fs::exists(dir) || !fs::is_directory(dir)) {
    throw std::runtime_error("model path is not a directory: " + model_dir);
  }
  std::string config_path = (dir / "config.json").string();
  // cua-s1-forms models (MODEL-CUA-S1-FORMS) ship cua-s1-forms.json instead
  // of config.json. Fall back so the standard loading path can resolve it.
  if (!fs::exists(config_path)) {
    const std::string cuas1_path = (dir / "cua-s1-forms.json").string();
    if (fs::exists(cuas1_path)) {
      config_path = cuas1_path;
    }
  }
  // Laya models (MODEL-LAYA) ship rl_agent_config.json instead of
  // config.json. Fall back so the standard loading path can resolve it.
  if (!fs::exists(config_path)) {
    const std::string rl_path = (dir / "rl_agent_config.json").string();
    if (fs::exists(rl_path)) {
      config_path = rl_path;
    }
  }
  // Root tokenizer.json, else tokenizer/tokenizer.json (MODEL-LAYA ships it
  // there). ISSUE-LOCAL-01M3SDXGYYTS9FDXKZDAE24N0R.
  const std::string tokenizer_path =
      vllm::ResolveTokenizerFile(dir, "tokenizer.json").string();

  // Refuse-by-task (ARCH-ONE-SURFACE ROW 1), BEFORE the full HfConfig parse: a
  // SupportsTranscription-ONLY architecture (Parakeet CTC/RNNT/TDT) has no
  // text-generation path, so the text engine must not be built around it —
  // mirror of vLLM excluding "generate" from supported_tasks for
  // supports_transcription_only models (interfaces.py:1118). The peek is
  // deliberately narrow: only a config whose architectures RESOLVE to a
  // transcription-only registration takes this exit (its config shape — e.g.
  // hidden_size nested under encoder_config — would otherwise fail the text
  // HfConfig parse below with a misleading message); every other model, known
  // or unknown, falls through with error ordering unchanged. The C ABI routes
  // such a directory to the transcription stack before reaching here
  // (vllm_c.cpp), so this fires only for a text-only consumer (server --task
  // generate, vllm-cli, bench).
  if (const std::vector<std::string> archs =
          vllm::PeekHfArchitectures(config_path);
      !archs.empty()) {
    const ModelRegistration* peek = nullptr;
    try {
      peek = &ModelRegistry::Resolve(std::span<const std::string>(archs));
    } catch (const std::exception&) {
      peek = nullptr;  // unknown arch: the existing path owns the diagnosis
    }
    if (peek != nullptr && peek->info.supports_transcription_only) {
      throw std::runtime_error(
          "Model architecture " + std::string(peek->architecture) +
          " supports transcription only (no text generation). Use "
          "vllm_transcribe on the C ABI or the server's "
          "/v1/audio/transcriptions instead of the text-generation entry "
          "points.");
    }
  }
  HfConfig config = vllm::LoadHfConfig(config_path);
  const ModelRegistration& registration = ModelRegistry::Resolve(config);
  // #2314, and see the GGUF branch above for why this precedes weight I/O.
  // No `GgufFile` on this path: `--fit` refuses by name rather than resolving to
  // nothing, because the safetensors footprint is not knowable here.
  InstallMoePlacementPlan(
      ResolveModelDeviceType(registration.architecture, params.device),
      config.num_hidden_layers, /*gguf=*/nullptr);
  // ENG-WEIGHT-OFFLOAD totality guard. Refuse a configured offload against a
  // model whose loader does not consult the offloader, BEFORE any weight I/O.
  // Without this the budget would be accepted and free nothing, with no error
  // anywhere, because there is no single upload seam that could enforce the
  // obligation structurally.
  vllm::RefuseUnsupportedWeightOffload(
      params.offload_config.value_or(vllm::OffloadConfig{}),
      registration.architecture,
      registration.factory != nullptr &&
          registration.factory->supports_weight_offload);
  // Models without a standard tokenizer (e.g. cua-s1-forms, which uses
  // ByteCollator) ship no tokenizer.json. Empty() returns a Tokenizer whose
  // Encode/Decode are never called for those models.
  tok::Tokenizer tokenizer = fs::exists(tokenizer_path)
      ? tok::Tokenizer::FromHfJson(tokenizer_path)
      : tok::Tokenizer::Empty();

  // Shared ownership so a loader may retain the mmap'd shards past the load: the
  // Qwen3.6-35B MoE loader defers its routed-expert host copies and streams them
  // per layer during PrepareMarlinResident (bounds load-phase peak PSS). The
  // deferred-expert closure holds the last reference and releases the shards once
  // the device Marlin resident is built; loaders that don't retain it drop the
  // shards when this local `shards` and the model's ModelSource go out of scope.
  const auto t_open = std::chrono::steady_clock::now();
  auto shards = std::make_shared<const std::vector<vllm::SafetensorsFile>>(
      LoadShards(model_dir));
  ReportLoadPhase("mmap+header", SecondsSince(t_open));

  // SPEC-MTP I5d-pre: when a speculative (MTP) config is set, load the `mtp.*`
  // draft weights from the SAME shards and retain them on the loaded target
  // model, so the runner has a typed path to build the draft
  // (LoadedModel::BuildMtpDraft). This runs INSIDE FromModelDir, while the local
  // `shards` shared_ptr is still alive — the dense direct-device-load path
  // releases the shards once the target is on device, so the draft tensors must
  // be materialized here before this function returns. Inert (never called) on
  // the production default where speculative_config is unset. The verify/propose
  // runner loop is I5d; this only retains the weights.
  const auto maybe_attach_mtp = [&](LoadedModel& loaded) {
    if (!params.speculative_config.has_value() ||
        params.speculative_config->method != "mtp") {
      return;
    }
    const Qwen3_5MTPKind kind = registration.factory->is_dense_model
                                    ? Qwen3_5MTPKind::kDense
                                    : Qwen3_5MTPKind::kMoe;
    const vt::DeviceType draft_device =
        ResolveModelDeviceType(registration.architecture, params.device);
    Qwen3_5MTPWeights draft =
        vllm::LoadQwen3_5MTP(*shards, config, kind, draft_device);
    if (draft.IsExl3() && kind == Qwen3_5MTPKind::kDense &&
        draft_device == vt::DeviceType::kXPU) {
      const char* path = std::getenv("EXL3_DRAFT_VOCAB");
      VT_CHECK(path != nullptr && path[0] != '\0',
               "EXL3 MTP requires EXL3_DRAFT_VOCAB pointing to the pinned 65536-token subset");
      std::ifstream input(path, std::ios::binary);
      VT_CHECK(input.good(), "EXL3 MTP: cannot read draft vocabulary");
      const std::string bytes((std::istreambuf_iterator<char>(input)),
                               std::istreambuf_iterator<char>());
      const std::string hash = v1::sha256_bytes(bytes);
      constexpr uint8_t expected[] = {
          0xb4,0xea,0xdc,0x08,0x80,0x59,0x19,0x09,0x83,0xfe,0x04,0x98,0xaf,0x11,0x86,0x4f,
          0x5a,0xaa,0x2e,0xaf,0x2e,0xc5,0x8a,0xe9,0x86,0x4f,0x07,0x15,0x63,0x4d,0x31,0x3d};
      VT_CHECK(hash.size() == sizeof(expected) &&
                   std::memcmp(hash.data(), expected, sizeof(expected)) == 0,
               "EXL3 MTP: draft subset differs from the pinned production artifact");
      draft.draft_head_exl3 =
          LoadExl3DraftHead(*shards, nlohmann::json::parse(bytes), config.vocab_size);
    }
    const bool gptq4 = config.raw.contains("quantization_config") &&
        config.raw.at("quantization_config").is_object() &&
        config.raw.at("quantization_config").value("quant_method", std::string()) ==
            "gptq";
    if (gptq4 && kind == Qwen3_5MTPKind::kDense &&
        draft_device == vt::DeviceType::kXPU) {
      const StTensor* head = nullptr;
      for (const SafetensorsFile& shard : *shards) {
        if (std::find(shard.Names().begin(), shard.Names().end(),
                      "lm_head.weight") != shard.Names().end()) {
          VT_CHECK(head == nullptr, "qwen3_5 MTP: duplicate lm_head.weight");
          head = &shard.Get("lm_head.weight");
        }
      }
      VT_CHECK(head != nullptr, "qwen3_5 MTP: missing FP16 target lm_head.weight");
      PackQwen3_5MTPGptqDraft(draft, *head);
    }
    loaded.AttachMtpDraftWeights(std::move(draft));
  };

  // SPEC-DFLASH D5: when a dflash config is set, load the SEPARATE z-lab draft
  // checkpoint (host bf16 weights + the target-SHARED embed/lm_head read from
  // *shards, which are still alive here) into a DflashDraft bundle the engine
  // owns and wires into the runner. Null (never built) on every other path.
  const auto maybe_load_dflash = [&]() -> std::unique_ptr<DflashDraft> {
    if (!params.speculative_config.has_value()) return nullptr;
    // SPEC-DSPARK W5: the DSpark draft rides the same seam and the same bundle.
    if (params.speculative_config->method == "dspark") {
      // SPEC-DSPARK-BLOCK-SIZE-GUARD (#1225): use the config resolved at the top
      // of this function, which is where the block floor is applied. Resolving
      // again here is what put a SECOND, differently-argued copy of the
      // resolution ahead of the constructor's; `LoadDsparkDraft` sizes the draft
      // block from this k alone, so the k it gets must be the refused-or-accepted
      // one and not a second opinion.
      return LoadDsparkDraft(*dspark_spec, SharedHeadSource(shards.get()));
    }
    if (params.speculative_config->method != "dflash") {
      return nullptr;
    }
    // ResolveSpecConfig re-runs on the target config in the LoadedEngine ctor, so
    // resolve the draft here from the CLI config (path + k) directly.
    vllm::SpeculativeConfig resolved = vllm::SpeculativeConfig::ResolveDflash(
        params.speculative_config->ResolvedNumSpeculativeTokens());
    resolved.draft_model_path = params.speculative_config->draft_model_path;
    return LoadDflashDraft(resolved, SharedHeadSource(shards.get()));
  };

  // Live architecture dispatch: consume config.architectures in order and let
  // the matched registration own the weight-name map/loader. Unknown dense
  // configs now reject instead of falling through num_experts == 0.
  //
  // ROW 7 (kimi-linear.md §20.3): a factory with `stage_on_load` (Kimi-Linear's
  // 91.5 GiB bf16-resident loader) takes the queue-selected branch below so the
  // CUDA context exists BEFORE the weights load and each tensor stages then
  // releases its host mirror (the §13 GB10 recipe). Every other arch resolves
  // this condition exactly as before — byte-identical.
  const bool queue_load =
      (registration.factory->is_dense_model && DirectDeviceLoadRequested()) ||
      registration.factory->stage_on_load;
  if (!queue_load) {
    const auto t_weights = std::chrono::steady_clock::now();
    // #607 L3 production call site (see the GGUF branch above for the argument).
    ModelSource source = ModelSource::FromSafetensorsOwned(shards);
    source.multimodal = &params.multimodal;
    std::unique_ptr<LoadedModel> model = ModelRegistry::Load(config, source);
    ReportLoadPhase("weights", SecondsSince(t_weights));
    ReportLoadBytes();

    // ENG-WEIGHT-OFFLOAD (#2386): the SECOND guard, and the only one that can
    // see this particular lie. `RefuseUnsupportedWeightOffload` runs BEFORE the
    // load and can only ask whether the architecture DECLARES support. A loader
    // that declares it and then never calls `ConsiderWeight` offloads nothing,
    // and the run frees no memory with no error anywhere — zero consulted
    // weights is the only evidence that proves it, and that count does not exist
    // until the load has finished. Hence here, and not beside its sibling.
    //
    // It had no production caller until now: declaration, definition and four
    // test assertions, while `docs/WEIGHT-OFFLOAD.md` described it in the
    // present tense as shipped behaviour. It is currently VACUOUS rather than
    // wrong — no registration sets `supports_weight_offload`, so the first guard
    // refuses every configured offload before this one is reached — but it stops
    // being vacuous the day anyone wires a loader, which is exactly when its
    // absence would be silent.
    vllm::VerifyWeightOffloadWasConsulted(
        vllm::GetWeightOffloader(), registration.architecture,
        registration.factory != nullptr &&
            registration.factory->supports_weight_offload);

    maybe_attach_mtp(*model);
    std::unique_ptr<DflashDraft> dflash = maybe_load_dflash();
    return std::unique_ptr<LoadedEngine>(new LoadedEngine(
        std::move(config), std::move(model), std::move(tokenizer), params,
        /*preselected_queue=*/nullptr, std::move(dflash)));
  }

  // Select before loading so an eligible discrete-CUDA dense loader stages each
  // completed layer to the exact queue the runner will use. If construction
  // fails before the runner takes over, destroy the selected native stream.
  vt::Queue load_queue =
      SelectQueueForModel(registration.architecture, params.device);
  try {
    const auto t_weights = std::chrono::steady_clock::now();
    // #607 L3 production call site (see the GGUF branch above for the argument).
    ModelSource source = ModelSource::FromSafetensorsOwned(shards, &load_queue);
    source.multimodal = &params.multimodal;
    std::unique_ptr<LoadedModel> model = ModelRegistry::Load(config, source);
    ReportLoadPhase("weights", SecondsSince(t_weights));
    ReportLoadBytes();

    // ENG-WEIGHT-OFFLOAD (#2386): the SECOND guard, and the only one that can
    // see this particular lie. `RefuseUnsupportedWeightOffload` runs BEFORE the
    // load and can only ask whether the architecture DECLARES support. A loader
    // that declares it and then never calls `ConsiderWeight` offloads nothing,
    // and the run frees no memory with no error anywhere — zero consulted
    // weights is the only evidence that proves it, and that count does not exist
    // until the load has finished. Hence here, and not beside its sibling.
    //
    // It had no production caller until now: declaration, definition and four
    // test assertions, while `docs/WEIGHT-OFFLOAD.md` described it in the
    // present tense as shipped behaviour. It is currently VACUOUS rather than
    // wrong — no registration sets `supports_weight_offload`, so the first guard
    // refuses every configured offload before this one is reached — but it stops
    // being vacuous the day anyone wires a loader, which is exactly when its
    // absence would be silent.
    vllm::VerifyWeightOffloadWasConsulted(
        vllm::GetWeightOffloader(), registration.architecture,
        registration.factory != nullptr &&
            registration.factory->supports_weight_offload);

    maybe_attach_mtp(*model);
    std::unique_ptr<DflashDraft> dflash = maybe_load_dflash();
    return std::unique_ptr<LoadedEngine>(new LoadedEngine(
        std::move(config), std::move(model), std::move(tokenizer), params,
        &load_queue, std::move(dflash)));
  } catch (...) {
    vt::DestroyQueue(load_queue);
    throw;
  }
}

}  // namespace vllm::entrypoints
