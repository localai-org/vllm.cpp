#pragma once
// Activation dumping for the qwen3.5 dense paged forward.
//
// ROCM-TIER-DIVERGENCE (#2590). Issue:
// https://github.com/mudler/vllm.cpp/issues/2590
// Spec: .agents/specs/rocm-tier-hidden-state-bisect.md
//
// The keying and the file I/O live here, apart from the device reads, so a test
// can drive the writer without a backend. The device-touching wrappers
// (`DumpTensor`, `MaybeDumpStream`) stay beside the forward in qwen3_5.cpp,
// where the `Dev`/`DBuf` glue is.

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "vt/device.h"
#include "vt/dtype.h"

namespace vllm {

// ─── ROCM-TIER-DIVERGENCE (#2590): the activation-dump writer ────────────────
//
// Every env-gated activation dump in the qwen3.5 dense paged forward goes through
// `actdump`. It exists
// because three dumps were already in that file and none of them could support the
// comparison #2590 needs, for reasons that were invisible in their output:
//
//   * A FAILED OPEN WROTE NOTHING AND SAID NOTHING. Each writer was
//     `if (f != nullptr) { fwrite; fclose; }`, so a dump into a directory that
//     does not exist is byte-for-byte the same experience as a dump that was
//     never switched on. That is the failure `.agents/verification.md` names
//     ("an instrument that writes nothing looks exactly like an instrument that
//     is off"), and the sibling logit-dump row burned a lease on exactly it.
//     Here every open, write and close is a VT_CHECK carrying the knob, the
//     path and errno, so a silent skip is now a refusal by name.
//   * NOTHING RECORDED WHAT WAS DUMPED. The blobs were raw bytes with no dtype,
//     shape or provenance, so a reader had to infer the element width from the
//     file size and two dumps of different dtypes compared as garbage without
//     either side saying so. Every blob now files a `manifest.tsv` row, and the
//     comparator joins on it rather than on a naming convention.
//   * THE KEY WAS AN INVOCATION ORDINAL. `RunDenseLayerPaged`'s sub-stage dump
//     keyed its files on a `static thread_local` counter that is never reset, so
//     one extra forward call on either side silently shifts the whole join. The
//     key is now (step, layer), taken explicitly.
//
// Debug-only on both knobs: each blob costs a Download, which synchronizes. A
// capture path must never set either variable.
namespace actdump {

// The two knobs, read ONCE. `VT_DUMP_ACT` is the residual-stream dump (both
// halves, per layer); `VT_DUMP_ACT_SUB` is the per-layer sub-stage dump. Both
// are pre-existing names and both already sit on scripts/env-doc-allowlist.txt,
// so this facility adds no environment variable.
inline const char* StreamDir() {
  static const char* d = std::getenv("VT_DUMP_ACT");
  return d;
}
inline const char* StageDir() {
  static const char* d = std::getenv("VT_DUMP_ACT_SUB");
  return d;
}
inline bool Enabled() { return StreamDir() != nullptr || StageDir() != nullptr; }

// Where the dump currently is. `step` is the forward-call ordinal of the paged
// dense forward; `layer` is the decoder-layer index, or -1 for the pre-layer
// (post-embedding) snapshot. A nested dump inside the GDN or attention block
// reads this instead of keeping a counter of its own, which is what stops two
// dumps in one process from disagreeing about where they are.
struct Where {
  int64_t step = -1;
  int64_t layer = -1;
};
inline Where& Current() {
  static thread_local Where w;
  return w;
}

// Focused eager diagnostics can observe one forward ordinal. Count every
// forward in BeginStep, but reject unselected steps before device downloads.
// This programmatic scope adds no serving knob and restores the prior filter.
inline int64_t& SelectedStep() {
  static thread_local int64_t step = -1;
  return step;
}
inline bool StepSelected(int64_t step) {
  return step >= 0 && (SelectedStep() < 0 || SelectedStep() == step);
}
struct StepSelectionScope {
  explicit StepSelectionScope(int64_t step) : previous_(SelectedStep()) {
    VT_CHECK(step >= -1, "activation dump selection requires a nonnegative ordinal or -1");
    SelectedStep() = step;
  }
  ~StepSelectionScope() { SelectedStep() = previous_; }
  StepSelectionScope(const StepSelectionScope&) = delete;
  StepSelectionScope& operator=(const StepSelectionScope&) = delete;
 private:
  int64_t previous_;
};
inline const char* ActiveStreamDir() {
  const char* dir = StreamDir();
  return dir != nullptr && StepSelected(Current().step) ? dir : nullptr;
}
inline int64_t& SelectedStageLayer() {
  static thread_local int64_t layer = -1;
  return layer;
}
struct StageLayerSelectionScope {
  explicit StageLayerSelectionScope(int64_t layer) : previous_(SelectedStageLayer()) {
    VT_CHECK(layer >= -1, "activation substage selection requires a nonnegative layer or -1");
    SelectedStageLayer() = layer;
  }
  ~StageLayerSelectionScope() { SelectedStageLayer() = previous_; }
  StageLayerSelectionScope(const StageLayerSelectionScope&) = delete;
  StageLayerSelectionScope& operator=(const StageLayerSelectionScope&) = delete;
 private:
  int64_t previous_;
};
inline bool StageLayerSelected(int64_t layer) {
  return SelectedStageLayer() < 0 || SelectedStageLayer() == layer;
}
inline const char* ActiveStageDir() {
  const char* dir = StageDir();
  return dir != nullptr && StepSelected(Current().step) &&
      StageLayerSelected(Current().layer) ? dir : nullptr;
}

inline std::atomic<int64_t> g_step{-1};
inline std::atomic<int64_t> g_blobs{0};       // whole process
inline std::atomic<int64_t> g_blobs_step{0};  // reset at each BeginStep
// The RESIDUAL-STREAM blobs alone, counted apart from every other stage.
//
// A total is not enough, and a reachability mutation proved it: deleting the
// stream call sites from the paged loop left the GDN stage probes writing, the
// total stayed non-zero, and a gate keyed on the total passed over a forward
// that no longer dumped a single hidden state. A count that cannot go to zero
// when the thing it measures is removed measures nothing.
inline std::atomic<int64_t> g_stream_blobs_step{0};

// One line per process, before the first blob, naming what the instrument was
// pointed at. A reader of the log can then audit the wiring without reading
// this file, which is the repair `.agents/verification.md` asks for.
inline void NarrateOnce(vt::DeviceType dev, const char* model_class, int64_t layers,
                 int64_t T, int64_t H) {
  static std::atomic<bool> done{false};
  bool expected = false;
  if (!done.compare_exchange_strong(expected, true)) return;
  std::fprintf(stderr,
               "[VT_DUMP_ACT] instrument ON: model=%s device=%s layers=%lld "
               "rows=%lld cols=%lld stream_dir=%s stage_dir=%s\n",
               model_class, vt::DeviceTypeName(dev),
               static_cast<long long>(layers), static_cast<long long>(T),
               static_cast<long long>(H),
               StreamDir() == nullptr ? "(unset)" : StreamDir(),
               StageDir() == nullptr ? "(unset)" : StageDir());
  std::fflush(stderr);
}

// Append one manifest row. Separate from the blob so a truncated blob and a
// missing manifest row cannot both be silent; the comparator refuses when the
// two disagree.
inline void FileManifestRow(const char* knob, const std::string& dir,
                     const std::string& file, int64_t step, int64_t layer,
                     const char* stage, vt::DType dtype, int64_t rows,
                     int64_t cols, size_t nbytes) {
  const std::string path = dir + "/manifest.tsv";
  std::FILE* m = std::fopen(path.c_str(), "a");
  VT_CHECK(m != nullptr, std::string(knob) + ": cannot open the manifest '" +
                             path + "': " + std::strerror(errno) +
                             ". The dump directory must exist and be writable; "
                             "a dump that cannot record what it wrote is not a "
                             "dump");
  const int n = std::fprintf(
      m, "%lld\t%lld\t%s\t%s\t%lld\t%lld\t%zu\t%s\n",
      static_cast<long long>(step), static_cast<long long>(layer), stage,
      vt::Name(dtype), static_cast<long long>(rows),
      static_cast<long long>(cols), nbytes, file.c_str());
  const int rc = std::fclose(m);
  VT_CHECK(n > 0 && rc == 0,
           std::string(knob) + ": short write on the manifest '" + path +
               "'; the dump would describe fewer blobs than it wrote");
}

// Write one blob and its manifest row. Refuses on every I/O failure.
inline void WriteBlob(const char* knob, const std::string& dir, int64_t step,
               int64_t layer, const char* stage, vt::DType dtype, int64_t rows,
               int64_t cols, const void* bytes, size_t nbytes) {
  const std::string file = "s" + std::to_string(step) + "_l" +
                           std::to_string(layer) + "_" + stage + ".bin";
  const std::string path = dir + "/" + file;
  std::FILE* f = std::fopen(path.c_str(), "wb");
  VT_CHECK(f != nullptr,
           std::string(knob) + ": cannot open the dump file '" + path + "': " +
               std::strerror(errno) +
               ". The dump directory must exist and be writable; a dump that "
               "writes nothing is indistinguishable from a dump that is off");
  const size_t w = std::fwrite(bytes, 1, nbytes, f);
  const int rc = std::fclose(f);
  VT_CHECK(w == nbytes && rc == 0,
           std::string(knob) + ": short write on '" + path + "' (" +
               std::to_string(w) + " of " + std::to_string(nbytes) +
               " bytes); a truncated blob reads as a shape mismatch later");
  FileManifestRow(knob, dir, file, step, layer, stage, dtype, rows, cols,
                  nbytes);
  g_blobs.fetch_add(1, std::memory_order_relaxed);
  g_blobs_step.fetch_add(1, std::memory_order_relaxed);
}

// Scope the (step, layer) key for one decoder layer, so a dump nested inside the
// GDN or attention block does not have to be handed the index.
struct LayerScope {
  LayerScope(int64_t step, int64_t layer) {
    prev_ = Current();
    Current() = Where{step, layer};
  }
  ~LayerScope() { Current() = prev_; }
  LayerScope(const LayerScope&) = delete;
  LayerScope& operator=(const LayerScope&) = delete;

 private:
  Where prev_;
};

// Open a forward step. Returns the step ordinal, or -1 when neither knob is set,
// in which case every dump below is inert.
inline int64_t BeginStep() {
  if (!Enabled()) return -1;
  const int64_t s = g_step.fetch_add(1, std::memory_order_relaxed) + 1;
  g_blobs_step.store(0, std::memory_order_relaxed);
  g_stream_blobs_step.store(0, std::memory_order_relaxed);
  Current() = Where{s, -1};
  return s;
}

// Close a forward step, and say how many blobs it wrote. ZERO IS A REFUSAL, not
// a silence: an enabled instrument that produced nothing for a whole forward has
// failed, and the run must not continue and be read as a clean profile.
//
// TWO floors, because one of them cannot see the failure the other exists for.
// `stream_due` counts residual-stream blobs alone; `any_due` counts every blob.
// A reachability mutation that deleted both stream call sites left the GDN stage
// probes writing, so the total stayed high while not one hidden state was
// dumped — which a single total-keyed floor reports as success.
inline void EndStep(int64_t step, int64_t stream_due, int64_t any_due) {
  if (!StepSelected(step)) return;
  const int64_t n = g_blobs_step.load(std::memory_order_relaxed);
  const int64_t sn = g_stream_blobs_step.load(std::memory_order_relaxed);
  // `stream_blobs` is printed SEPARATELY from the total, and it is the number a
  // gate keys on. The total includes the GDN stage probes, which fire off the
  // same variable, so it stays non-zero even when the residual-stream dump is
  // gone entirely.
  std::fprintf(stderr,
               "[VT_DUMP_ACT] step=%lld stream_blobs=%lld blobs_this_step=%lld "
               "blobs_total=%lld\n",
               static_cast<long long>(step), static_cast<long long>(sn),
               static_cast<long long>(n),
               static_cast<long long>(g_blobs.load(std::memory_order_relaxed)));
  std::fflush(stderr);
  VT_CHECK(sn >= stream_due,
           "VT_DUMP_ACT: the instrument is ON and wrote " + std::to_string(sn) +
               " residual-stream blobs where " + std::to_string(stream_due) +
               " were due for this step. A dump that writes nothing looks "
               "exactly like a dump that is off, so this is a refusal");
  // `any_due` is the caller's, because only the caller knows which dumps its own
  // path carries: the MoE loop has no sub-stage probes, so a SUB-only run there
  // legitimately writes nothing and a hardwired "> 0" would refuse a correct
  // run. It is a separate floor from `stream_due` and not a weaker copy of it.
  VT_CHECK(n >= any_due,
           "VT_DUMP_ACT/VT_DUMP_ACT_SUB: the instrument is ON and wrote " +
               std::to_string(n) + " blobs where at least " +
               std::to_string(any_due) + " were due for this step");
}

}  // namespace actdump

}  // namespace vllm
