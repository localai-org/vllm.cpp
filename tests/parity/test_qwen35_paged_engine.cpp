// vllm.cpp original (ADDITIVE-MODEL W4 — THE SACRED correctness gate); no
// upstream mirror.
//
// THE PAGED-ENGINE Qwen3.5-0.8B (GDN) GREEDY CORRECTNESS GATE. The GDN sibling of
// test_qwen3_paged_engine.cpp: drives the standard prompt battery through the FULL
// PAGED LLMEngine stack (InputProcessor -> Scheduler -> GDN recurrence + paged
// attention on the full-attention layers + Sampler -> OutputProcessor) via
// LoadedEngine::FromModelDir, and checks greedy (temperature-0) decode against the
// pinned vLLM oracle (commit 555967922, runtime 0.23.1rc1.dev1511+g555967922).
//
// ORACLE PROVENANCE — DIFFERENT FROM THE QWEN3-DENSE GATE. There is no dgx/CUDA
// capture for this model. The 0.8B GDN checkpoint runs on the gfx1100 box, which
// also hosts the vLLM-ROCm oracle. Issue #2772 refreshed the committed base
// files from the restored immutable image in production mode. The default
// production oracle and the local default both select wvSplitK. The files are:
//   greedy_ids.npy  [N,T]   i32  the pinned ROCm oracle's per-prompt greedy
//                                (K=10 per-prompt runs DETERMINISTIC in every
//                                cell — see greedy_dist.npy).
//   our_ids.npy     [N,T]   i32  OUR ROCm engine's greedy (the anchor).
//   neartie_gap_mnats.npy [N,T]  oracle teacher-forced gap (milli-nats) for OUR
//                                token given OUR prefix.
// On kROCM these base files are the gate; on ANY other device the gate SKIPS
// (exit 77) rather than compare another engine's tokens against ROCm-derived
// goldens — fail-safe by device, not luck — with ONE ratified exception: the
// Tenstorrent lane, which gets the Mistral gate's device-golden pair treatment
// (our_ids_tenstorrent.npy + neartie_gap_mnats_tenstorrent.npy; gap teacher-
// forced via scripts/qwen3-neartie-gap-transformers.py — vLLM has no TT
// backend, so `transformers` is the secondary oracle per AGENTS.md's registry).
//
// PROVENANCE OF THE GREEN: the committed our_ids/near-tie pair is the fixed
// engine's sequence. The oracle was re-derived after the AttnQkNormRopeGate
// output-dtype dispatch fix in row/ROCM-GDN-08B-FIX. Issue #2772 then re-derived
// all four ROCm arrays in production mode. The historical eager capture is
// diagnostic evidence only. It selected a different branch of an exact tie.
// The pre-fix capture had 13/16 forward-divergent prompts and a maximum first-
// divergence gap of 1.062 nats. That result remains in
// .agents/specs/rocm-m4-oracle.md and the parity ledger.
//
// METHODOLOGY — identical to the Qwen3-dense gate (see that file's header and
// [[near-tie-distributional-gate]]): STRICT token-exact is reported, but the
// PASS bar is the near-tie band: given OUR EXACT PREFIX, the oracle's OWN logits
// must place OUR token within kNearTieMnats of the oracle argmax. Strict where
// well-posed, near-tie-robust only where the oracle itself cannot separate the
// tokens.
//
// BACKEND PROOF — GDN op set (not the Qwen3-dense list): the GDN layers dispatch
// kCausalConv1dFwd/Update (prefill/decode conv), kGdnPrefill/kGdnDecode (the
// recurrence), kGdnPostConv, kRmsNormGated, kSigmoidGateBf16,
// kAttnQkNormRopeGate (the GDN preamble), and the full-attention layers dispatch
// kReshapeAndCache + kPagedAttention + kAttnQkNormRope; shared: kEmbedding,
// kMatmulBT, kRmsNorm, kSiluAndMul, kGreedyArgmax. All must show selections>0
// and declines==0 on the running device.
//
// Checkpoint-gated: resolves the HF snapshot under ~/.cache/huggingface/hub/.
// Absent snapshot (CI/CPU) => loud SKIP.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "hf_snapshot.h"
#include "npy.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/sampling_params.h"
#include "vt/breakable_graph.h"  // GetGraphBreakStats: the served-replay gate's replays/captures counters
#include "vt/op_provider.h"  // the "which backend actually ran" proof
#include "vt/ops.h"
#include "vt/tenstorrent/tenstorrent_device.h"

namespace fs = std::filesystem;

namespace {

// Near-tie acceptance threshold in milli-nats — same bar as the Qwen3-dense
// gate (0.5 nats); see that file for the derivation.
constexpr int32_t kNearTieMnats = 500;
// Stable test-only sentinel: distinct from process success and CTest skip 77.
constexpr int kPrerequisiteProbeCompleteExit = 86;

// The standard prompt battery — MUST match scripts/qwen3-oracle-capture.py (and
// qwen3-neartie-gap.py) exactly; those scripts are model-agnostic and take
// --model/--out-dir/--golden-dir.
const std::vector<std::string>& Prompts() {
  static const std::vector<std::string> p = {
      "The capital of France is",
      "Once upon a time,",
      "In the beginning God created",
      "The quick brown fox jumps over",
      "def fibonacci(n):",
      "Water boils at a temperature of",
      "The theory of relativity was developed by",
      "To be or not to be, that is",
      "The largest planet in our solar system is",
      "Machine learning is a subfield of",
      "The mitochondria is the powerhouse of",
      "Roses are red, violets are",
      "The first president of the United States was",
      "E equals m c",
      "A journey of a thousand miles begins with",
      "The chemical symbol for gold is",
  };
  return p;
}

vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.PostInit();
  return sp;
}

const int32_t* AsI32(const parity::NpyArray& a) {
  return reinterpret_cast<const int32_t*>(a.data.data());
}

// A gate that cannot run must not report success: exit 77 makes CTest report
// **Skipped**, never a false green (issue #463; SKIP_RETURN_CODE is wired for
// every parity test in tests/CMakeLists.txt).
[[noreturn]] void SkipGate(const char* label, const std::string& why) {
  std::fprintf(stderr,
               "\n*** GATE NOT RUN — SKIPPED (exit 77), this is NOT a pass ***\n"
               "*** %s: %s\n\n",
               label, why.c_str());
  std::fflush(stderr);
  std::exit(77);
}

enum class GateArtifactState { kReady, kBootstrap };

GateArtifactState RequireGateArtifacts(const fs::path& gdir, const char* label,
                                       bool dump) {
  if (!fs::exists(gdir / "greedy_ids.npy")) {
    SkipGate(label,
             "required greedy_ids.npy is absent. Capture it with the pinned "
             "ROCm oracle using qwen3-oracle-capture.py --per-prompt");
  }

  const bool have_anchor = fs::exists(gdir / "our_ids.npy");
  const bool have_gap = fs::exists(gdir / "neartie_gap_mnats.npy");
  if (have_anchor && have_gap) return GateArtifactState::kReady;
  if (dump) return GateArtifactState::kBootstrap;

  std::string missing;
  if (!have_anchor) missing = "our_ids.npy";
  if (!have_gap) {
    if (!missing.empty()) missing += " and ";
    missing += "neartie_gap_mnats.npy";
  }
  SkipGate(label, "required " + missing +
                      " is absent. Run with VT_DUMP_IDS=1, then run "
                      "qwen3-neartie-gap.py");
}

// `model` is the checkpoint the gate loads: the bf16 HF snapshot for the
// SACRED ROCm pair, the q4km GGUF path for the keep-quant vehicle arm.
// `keep_quant` selects the dispatch set the backend-proof block asserts: the
// bf16 vehicle dispatches plain kMatmulBT and kSiluAndMul, while the
// keep-quant vehicle swaps those two for the W3 decode surface — every matmul
// weight stays block-encoded and decodes on-core (kMatmulBTQuant) and the
// split gate/up dense MLP routes through kMoeSiluMul (qwen3_5.cpp:7610) — so
// the same selections>0/declines==0 proof covers the quant path's e2e reach.
void RunGate(const std::string& golden_subdir, const char* label,
             const std::string& model, const bool keep_quant = false) {
  const char* probe_dir = std::getenv("VT_QWEN35_GATE_PREREQ_PROBE_DIR");
  const bool probe = probe_dir != nullptr;
  const std::string snap = probe ? std::string() : model;
  if (!probe && snap.empty()) {
    SkipGate(label, "model artifact not cached — resolve the snapshot or GGUF "
                    "path this gate is pinned to first");
  }
  // W4d W3 (#3042): the 27B checkpoint's auto-fit (max_model_len 8192 -> a
  // 256-block KV pool and a 13104-token capture shape) drove the 32 GiB card
  // to 99 percent bank occupancy and OOM'd mid-generation; the battery needs
  // ~40 tokens of context. Bound the engine to the workload the way vLLM's
  // own smoke tests pass max_model_len. The 0.8B vehicle keeps the default.
  vllm::entrypoints::EngineParams gate_params;
  if (label == std::string_view("qwen38-gguf-q4km-27b"))
    gate_params.max_model_len = 1024;
  const fs::path gdir = probe ? fs::path(probe_dir)
                              : fs::path(PARITY_GOLDENS_DIR) / golden_subdir;
  const bool dump = !probe && std::getenv("VT_DUMP_IDS") != nullptr;
  const GateArtifactState artifacts =
      RequireGateArtifacts(gdir, label, dump);
  if (probe) {
    std::fprintf(
        stderr,
        "\n*** TEST PROBE COMPLETE (exit %d), production gate NOT RUN ***\n"
        "*** %s: all three required artifact paths are present ***\n\n",
        kPrerequisiteProbeCompleteExit, label);
    std::fflush(stderr);
    std::exit(kPrerequisiteProbeCompleteExit);
  }
  // BOOTSTRAP: with VT_DUMP_IDS set and no gap golden yet, generate + dump OUR
  // token ids (our_ids.i32) so qwen3-neartie-gap.py can build the gap golden.
  if (artifacts == GateArtifactState::kBootstrap) {
    MESSAGE(label << ": BOOTSTRAP dump (gap golden absent) via FromModelDir(" << snap << ")...");
    std::unique_ptr<vllm::entrypoints::LoadedEngine> le =
        vllm::entrypoints::LoadedEngine::FromModelDir(snap, gate_params);
    const parity::NpyArray gg = parity::LoadNpy((gdir / "greedy_ids.npy").string());
    const int64_t NN = gg.shape[0], TT = gg.shape[1];
    std::vector<int32_t> buf(static_cast<size_t>(NN * TT), -1);
    for (int64_t i = 0; i < NN; ++i) {
      const vllm::RequestOutput out = le->engine().generate(
          Prompts()[static_cast<size_t>(i)], Greedy(static_cast<int>(TT)),
          "boot" + std::to_string(i));
      const std::vector<int32_t>& got = out.outputs[0].token_ids;
      for (int64_t j = 0; j < TT && j < static_cast<int64_t>(got.size()); ++j)
        buf[static_cast<size_t>(i * TT + j)] = got[static_cast<size_t>(j)];
    }
    const std::string path = (gdir / "our_ids.i32").string();
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f != nullptr) { std::fwrite(buf.data(), sizeof(int32_t), buf.size(), f); std::fclose(f); }
    MESSAGE(label << " BOOTSTRAP dumped our token ids -> " << path);
    SkipGate(label, "bootstrap does not run the correctness gate. Run "
                    "qwen3-neartie-gap.py, then rerun without VT_DUMP_IDS");
  }

  const parity::NpyArray g = parity::LoadNpy((gdir / "greedy_ids.npy").string());
  const parity::NpyArray o = parity::LoadNpy((gdir / "our_ids.npy").string());
  const parity::NpyArray gap = parity::LoadNpy((gdir / "neartie_gap_mnats.npy").string());
  REQUIRE(g.dtype == "<i4");
  REQUIRE(o.dtype == "<i4");
  REQUIRE(gap.dtype == "<i4");
  REQUIRE(g.shape.size() == 2);
  const int64_t N = g.shape[0];
  const int64_t T = g.shape[1];
  REQUIRE(o.shape.size() == 2);
  REQUIRE(o.shape[0] == N);
  REQUIRE(o.shape[1] == T);
  REQUIRE(gap.shape.size() == 2);
  REQUIRE(gap.shape[0] == N);
  REQUIRE(gap.shape[1] == T);
  REQUIRE(static_cast<size_t>(N) == Prompts().size());
  const int32_t* gd = AsI32(g);
  const int32_t* od = AsI32(o);
  const int32_t* gapd = AsI32(gap);

  std::vector<int32_t> our_dump;
  if (dump) our_dump.assign(static_cast<size_t>(N * T), -1);

  MESSAGE(label << ": loading via FromModelDir(" << snap << ")...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(snap, gate_params);

  // The base golden pair is production-mode ROCm evidence after issue #2772.
  // The Tenstorrent device lane carries its OWN oracle-backed golden pair
  // (the Mistral gate's treatment); every other device still skips loudly.
  const vt::DeviceType run_dev = loaded->runner().device().type;
  const bool rocm = run_dev == vt::DeviceType::kROCM;
  const bool tenstorrent = run_dev == vt::DeviceType::kTENSTORRENT;
  const bool device_golden = tenstorrent;
  // FAIL SAFE BY DEVICE: this model's only oracle-backed goldens are
  // ROCm-captured (the pinned vLLM-ROCm oracle on gfx1100 — no dgx/CUDA capture
  // exists). On any device without its OWN pair the anchor would be compared
  // against another engine's sequence, so skip loudly rather than misattribute
  // a drift. Tenstorrent is the one lane with a committed device pair.
  if (!rocm && !device_golden) {
    SkipGate(label, "goldens are ROCm-oracle-captured; this run is on device "
                    "type " + std::to_string(static_cast<int>(run_dev)) +
                    " — capture that device's pair first");
  }

  // The GDN op set this model dispatches — all must be proven on the running
  // device (selections > 0, declines == 0; fan-out spike Risk 4). The last two
  // entries are the GEMM/MLP pair that differs per arm (see RunGate's
  // keep_quant comment).
  const std::vector<vt::OpId> kGdnOps = {
      vt::OpId::kEmbedding,        vt::OpId::kRmsNorm,
      vt::OpId::kRmsNormGated,
      vt::OpId::kCausalConv1dFwd,  vt::OpId::kCausalConv1dUpdate,
      vt::OpId::kGdnPrefill,       vt::OpId::kGdnDecode,
      vt::OpId::kGdnPostConv,      vt::OpId::kSigmoidGateBf16,
      vt::OpId::kAttnQkNormRopeGate,
      vt::OpId::kReshapeAndCache,  vt::OpId::kPagedAttention,
      vt::OpId::kGreedyArgmax,
      keep_quant ? vt::OpId::kMatmulBTQuant : vt::OpId::kMatmulBT,
      keep_quant ? vt::OpId::kMoeSiluMul : vt::OpId::kSiluAndMul};
  if (rocm || device_golden) {
    for (vt::OpId op : kGdnOps) {
      CHECK(vt::OpRegistered(op, run_dev));
      vt::ResetOpProviderStats(op, run_dev);
    }
    vt::EnableOpProviderCallStats(true);
    MESSAGE(label << ": running on device type " << static_cast<int>(run_dev)
            << " (5=ROCM, 6=TENSTORRENT) — gated against "
               << (device_golden ? "this device's OWN oracle-backed golden"
                                 : "the ROCm oracle-backed golden pair"));
  }

  // Device-appropriate anchor + teacher-forced gap goldens. Base = the ROCm
  // pair. Tenstorrent carries its own pair (captured via VT_DUMP_IDS=1 on the
  // P150, then qwen3-neartie-gap-transformers.py teacher-forces `transformers`
  // on that sequence — NOT vLLM, which has no Tenstorrent backend; same
  // secondary-oracle lane and precedent as the Qwen3-0.6B and Mistral-7B TT
  // goldens).
  //
  // Each Tenstorrent decode ARM gates against its OWN captured pair (#2115).
  // The ambient pair (refreshed 2026-09-05 with the device-pure GDN wave:
  // the PA bf16 decode cast unlocked device sdpa_decode on the eager arm,
  // and the boundary re-capture + conv-shadow gate fixed the captured arm)
  // is a LEGITIMATE ALTERNATE GREEDY PATH, not a drift: teacher-forced
  // against the transformers oracle it diverges from greedy_ids at 81 cells
  // with max gap 375 mnats (69 of 81 cells our token IS the teacher-forced
  // argmax), inside the 500-mnat band. The captured pair diverges at 68
  // cells, also max gap 375 mnats. The two arms differ from each other at
  // 59 of 256 cells — near-ties on both paths, the same shape the Mistral
  // gate records. At exact ties the arm may take the tied runner-up, so the
  // near-tie band (kNearTieMnats) and the exact-match anchor REQUIRE stay
  // as they are. The pre-wave pairs' "uniform tok3 divergence" was a real
  // defect (prefill dual-role transitions replaced the conv/ssm shadow
  // device tensors the captured graph had baked; see the row spec), not an
  // alternate path — retired with the fix, not adjudicated.
  const char* hf_env = std::getenv("VT_TT_HOST_FREE_DECODE");
  const bool host_free_off =
      tenstorrent && hf_env != nullptr && std::string_view(hf_env) == "0";
  // Selection follows the engine's arm: the #1625 flip defaults TT capture
  // on for the evidence families (Qwen3.5-GDN joined with its own committed
  // pair), so — like the qwen3 test — the ambient leg keys on
  // DecodeCaptureEnabled() alone ("0" opts out to the eager arm and its
  // pair). A hardcoded eager name would adjudicate a captured run against
  // eager goldens when the arm IS captured.
  const bool tt_capture =
      tenstorrent && !host_free_off && vt::tenstorrent::DecodeCaptureEnabled();
  const char* ids_name = tenstorrent
                             ? (tt_capture
                                    ? "our_ids_tenstorrent_capture.npy"
                                    : (host_free_off
                                           ? "our_ids_tenstorrent_host_free_off.npy"
                                           : "our_ids_tenstorrent.npy"))
                             : "our_ids.npy";
  const char* gap_name = tenstorrent
                             ? (tt_capture
                                    ? "neartie_gap_mnats_tenstorrent_capture.npy"
                                    : (host_free_off
                                           ? "neartie_gap_mnats_tenstorrent_host_free_off.npy"
                                           : "neartie_gap_mnats_tenstorrent.npy"))
                             : "neartie_gap_mnats.npy";
  if (tt_capture && !fs::exists(gdir / ids_name) && !dump) {
    MESSAGE(label << " TT capture pair absent (the committed pair was removed "
            "or renamed); skipping on Tenstorrent");
    return;
  }
  parity::NpyArray o_dev, gap_dev;  // keep the device arrays alive for the loop
  bool bootstrap_only = false;
  if (device_golden) {
    const bool have_dev = fs::exists(gdir / ids_name) && fs::exists(gdir / gap_name);
    if (!have_dev && dump) {
      // Bootstrap dump path: generate tokens, write raw i32, skip the gate.
      bootstrap_only = true;
      MESSAGE(label << ": BOOTSTRAP dump (device golden absent) for Tenstorrent"
                    << (host_free_off ? std::string(" host-free opt-out arm")
                                      : std::string(" ambient arm"))
                    << " — dumping ids for " << std::string(ids_name) << " / "
                    << std::string(gap_name) << "...");
    } else {
      REQUIRE_MESSAGE(have_dev,
                      label << ": device oracle golden absent (" << ids_name << " / "
                            << gap_name
                            << ") — capture the sequence with VT_DUMP_IDS=1, then "
                               "teacher-force it: qwen3-neartie-gap-transformers.py "
                               "-> device golden pair");
      o_dev = parity::LoadNpy((gdir / ids_name).string());
      gap_dev = parity::LoadNpy((gdir / gap_name).string());
      REQUIRE(o_dev.dtype == "<i4");
      REQUIRE(gap_dev.dtype == "<i4");
      REQUIRE(o_dev.shape.size() == 2);
      REQUIRE(o_dev.shape[0] == N);
      REQUIRE(o_dev.shape[1] == T);
      REQUIRE(gap_dev.shape.size() == 2);
      REQUIRE(gap_dev.shape[0] == N);
      REQUIRE(gap_dev.shape[1] == T);
      od = AsI32(o_dev);
      gapd = AsI32(gap_dev);
    }
  }

  const int32_t* anchor_ids = od;
  const int32_t* gap_ids = gapd;

  // KEEPQUANT W3 capture-safety probe: the staging counter counts ONLY
  // capture-active word-shadow misses (EnsureKeepQuantWords refuses a
  // capture-time arrival by name, so a miss would have CHECK-aborted before
  // this point). The engine's pre-capture eager step warms every shadow, so
  // across a captured run the count must read ZERO — a positive count is the
  // #2812 class surviving the W3 fix. On the bf16 arm no keep-quant weight
  // exists and the count is trivially zero; the invariant costs one atomic read.
  if (tenstorrent && tt_capture)
    vt::tenstorrent::ResetKeepQuantCaptureStagingWritesForTest();

  int strict_exact = 0;
  int neartie_only = 0;
  int fail = 0;
  int32_t worst_gap = 0;
  int worst_i = -1, worst_j = -1;
  for (int64_t i = 0; i < N; ++i) {
    const vllm::RequestOutput out = loaded->engine().generate(
        Prompts()[static_cast<size_t>(i)], Greedy(static_cast<int>(T)),
        "gate" + std::to_string(i));
    REQUIRE(out.finished);
    REQUIRE(out.outputs.size() == 1);
    const std::vector<int32_t>& got = out.outputs[0].token_ids;
    REQUIRE(static_cast<int64_t>(got.size()) == T);
    if (dump) {
      for (int64_t j = 0; j < T; ++j)
        our_dump[static_cast<size_t>(i * T + j)] = got[static_cast<size_t>(j)];
    }
    if (bootstrap_only) continue;  // dump-only path; no anchor/gap yet

    // Anchor: the committed anchor is the exact deterministic sequence OUR ROCm
    // engine produces. Drift is a hard REQUIRE — no cross-device latitude.
    int first_div = -1;
    for (int64_t j = 0; j < T; ++j) {
      if (got[static_cast<size_t>(j)] != anchor_ids[i * T + j]) {
        first_div = static_cast<int>(j);
        break;
      }
    }
    const bool anchor_ok = dump || first_div < 0;
    // A drift is a REGRESSION until proven otherwise: this anchor is the fixed
    // engine's oracle-verified sequence. Refreshing it is the LAST step of a
    // justified re-capture, never the response to a failure.
    REQUIRE_MESSAGE(anchor_ok,
                    label << " anchor drift prompt[" << i << "] tok=" << first_div
                    << " engine=" << (first_div < 0 ? -1 : got[static_cast<size_t>(first_div)])
                    << " committed anchor="
                    << (first_div < 0 ? -1 : anchor_ids[i * T + first_div])
                    << " — REGRESSION SUSPECTED: bisect the engine change first; "
                       "only re-derive goldens (VT_DUMP_IDS=1 + "
                       "qwen3-neartie-gap.py) after the drift is proven to be a "
                       "justified numerical change, never to silence the gate");

    bool exact = true;
    bool prompt_ok = true;
    int first_bad = -1;
    for (int64_t j = 0; j < T; ++j) {
      if (got[static_cast<size_t>(j)] != gd[i * T + j]) exact = false;
      const int32_t mn = gap_ids[i * T + j];
      if (mn > worst_gap) {
        worst_gap = mn;
        worst_i = static_cast<int>(i);
        worst_j = static_cast<int>(j);
      }
      if (mn > kNearTieMnats) {
        prompt_ok = false;
        if (first_bad < 0) first_bad = static_cast<int>(j);
      }
    }
    if (!prompt_ok) {
      ++fail;
      MESSAGE(label << " FORWARD DIVERGENCE prompt[" << i << "] tok=" << first_bad
              << " our=" << got[static_cast<size_t>(first_bad)]
              << " vLLM_greedy=" << gd[i * T + first_bad]
              << " gap=" << (gap_ids[i * T + first_bad] / 1000.0) << " nats (> "
              << (kNearTieMnats / 1000.0) << ") \"" << out.outputs[0].text << "\"");
    } else if (exact) {
      ++strict_exact;
    } else {
      ++neartie_only;
    }
    CHECK(prompt_ok);
  }

  if (tenstorrent && tt_capture) {
    const int64_t staged = vt::tenstorrent::KeepQuantCaptureStagingWrites();
    CHECK_MESSAGE(staged == 0,
                  label << ": keep-quant decode staged " << staged
                        << " word uploads DURING the captured e2e (the #2812 "
                           "class — a captured graph reading bytes its replay "
                           "cannot refresh)");
    // W4a wave-3b-1: the device-reported live trace demand as of the last
    // capture end (the wave-3a LastTraceBytesForTest pattern), recorded on
    // both sides of the dense packed-word switch. Under the wave-3b-1c
    // dynamic-first region policy (TraceRegionSizeFromEnv) tt-metal grows the
    // capture region to the demand, so the number IS the axis: the measured
    // command stream, not a carve-out against a fixed size.
    MESSAGE(label << ": device trace demand at last capture end = "
                  << vt::tenstorrent::LastTraceBytesForTest()
                  << " B (dynamic trace region; VT_TT_TRACE_REGION_MB opts "
                     "into a fixed one)");
  }

  // Backend proof: token equality alone does not prove which device ran.
  // The bootstrap dump path does not exercise the full op set to a comparison,
  // so its stats prove reachability only (still selections > 0, declines == 0).
  if (device_golden || rocm) vt::EnableOpProviderCallStats(false);
  if (device_golden || rocm) {
    for (vt::OpId op : kGdnOps) {
      const auto st = vt::GetOpProviderStats(op, run_dev);
      CHECK_MESSAGE(st.selections > 0,
                    label << ": op " << static_cast<int>(op)
                          << " was never dispatched on device type "
                          << static_cast<int>(run_dev));
      CHECK_MESSAGE(st.declines == 0,
                    label << ": op " << static_cast<int>(op)
                          << " DECLINED and fell back");
    }
    MESSAGE(label << ": BACKEND PROOF — Qwen3.5 GDN ops on device type "
            << static_cast<int>(run_dev) << " with 0 declines (kPagedAttention "
               "selections="
            << vt::GetOpProviderStats(vt::OpId::kPagedAttention, run_dev).selections
            << ", kGdnDecode selections="
            << vt::GetOpProviderStats(vt::OpId::kGdnDecode, run_dev).selections
            << ", kCausalConv1dUpdate selections="
            << vt::GetOpProviderStats(vt::OpId::kCausalConv1dUpdate, run_dev).selections
            << ")");
  }

  if (dump) {
    const std::string dump_name =
        tt_capture ? "our_ids_tenstorrent_capture.i32"
                   : (tenstorrent ? "our_ids_tenstorrent.i32" : "our_ids.i32");
    const std::string path = (gdir / dump_name).string();
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f != nullptr) {
      std::fwrite(our_dump.data(), sizeof(int32_t), our_dump.size(), f);
      std::fclose(f);
      MESSAGE(label << " dumped our token ids -> " << path);
    }
  }
  if (bootstrap_only) {
    // BOOTSTRAP complete — no correctness bar was applied this run (every
    // prompt hit the `continue` above). The summary below would print
    // "0/16 prompts PASS ... 0 forward-divergent" over a vacuous REQUIRE —
    // the Mistral gate returns here for the same reason.
    MESSAGE(label << ": BOOTSTRAP complete -- ids dumped, NO correctness bar "
                     "was applied this run. Teacher-force the dumped sequence, "
                     "commit the golden pair, then re-run without VT_DUMP_IDS.");
    return;
  }
  MESSAGE(label << " correctness gate: " << (strict_exact + neartie_only) << "/" << N
          << " prompts PASS  (STRICT token-exact vs oracle per-prompt greedy: "
          << strict_exact << "/" << N << "; near-tie-band only: " << neartie_only
          << "/" << N << "; max gap " << (worst_gap / 1000.0) << " nats @ prompt["
          << worst_i << "] tok=" << worst_j << "; " << fail
          << " forward-divergent)");
  REQUIRE(fail == 0);
}

// KEEPQUANT W4c (issue #3079): the int8-dot LANE's own end-to-end gate.
//
// The lane (W4b's `VT_TT_KEEPQUANT_INT8DOT`, op-level bit-exact vs the pinned
// llama.cpp `ggml_vec_dot_*` integer domain) cannot be adjudicated by the
// ROCm-domain pair — the W4b outcome recorded the ≤500-mnat band failing at
// cell (5,7) with 1125 mnats there. The lane's denominator is therefore the
// `llama-cpp` oracle ITSELF (registered, gateable = yes): this battery runs
// the vehicle's captured production path over the standard 16-prompt battery
// with the lever ON and adjudicates against the LANE pair committed under
// tests/parity/goldens/qwen35_gguf_q4km_lanegate/ — the pin's teacher-forced
// gap on OUR captured prefix, in the established pair convention. Adjudication
// mirrors RunGate exactly: anchor drift is a hard REQUIRE (regression
// suspect), 0 forward-divergent cells, every flip inside the ratified band,
// an exact tie with a different id is legit (gap 0 is inside any band). The
// determinism leg re-runs the battery in-process through a SECOND engine load
// (a second decode-graph capture — the op-level capture test's x2
// byte-identity mechanism at vehicle scale) and REQUIREs the byte-identical
// token stream, reporting the device trace demand per run.
//
// Opt-out only: the lever is the default ON since the flip
// (tenstorrent-keepquant-int8dot-default), so this gate runs on the default
// configuration; an explicit VT_TT_KEEPQUANT_INT8DOT=0 opts out to the W4a
// grouped arm and this gate skips loudly.
constexpr int32_t kLaneBandMnats = 500;  // the row's ratified starting band

void RunLaneGate(const std::string& golden_subdir, const char* label,
                 const std::string& model) {
  const fs::path gdir = fs::path(PARITY_GOLDENS_DIR) / golden_subdir;
  const bool dump = std::getenv("VT_DUMP_IDS") != nullptr;
  const bool have_anchor = fs::exists(gdir / "our_ids.npy");
  const bool have_gap = fs::exists(gdir / "neartie_gap_mnats.npy");
  if (!have_anchor || !have_gap) {
    if (dump) {
      // BOOTSTRAP: the lane pair does not exist yet — generate the 16
      // prompts with the lever ON and dump OUR token ids so the pinned
      // llama.cpp oracle can teacher-force them
      // (scripts/qwen3-neartie-gap-llamacpp-oracle.py).
      MESSAGE(label << ": BOOTSTRAP dump (lane pair absent) via "
                       "FromModelDir("
                    << model << ") with VT_TT_KEEPQUANT_INT8DOT=1...");
      std::unique_ptr<vllm::entrypoints::LoadedEngine> le =
          vllm::entrypoints::LoadedEngine::FromModelDir(
              model, vllm::entrypoints::EngineParams{});
      std::vector<int32_t> buf;
      for (size_t i = 0; i < Prompts().size(); ++i) {
        const vllm::RequestOutput out = le->engine().generate(
            Prompts()[i], Greedy(16), "laneboot" + std::to_string(i));
        REQUIRE(out.finished);
        const std::vector<int32_t>& got = out.outputs[0].token_ids;
        REQUIRE(static_cast<int64_t>(got.size()) == 16);
        buf.insert(buf.end(), got.begin(), got.end());
      }
      const fs::path path = gdir / "our_ids.i32";
      std::FILE* f = std::fopen(path.string().c_str(), "wb");
      if (f != nullptr) {
        std::fwrite(buf.data(), sizeof(int32_t), buf.size(), f);
        std::fclose(f);
      }
      MESSAGE(label << " BOOTSTRAP dumped our token ids -> " << path);
      SkipGate(label, "bootstrap does not run the correctness gate. Run "
                      "scripts/qwen3-neartie-gap-llamacpp-oracle.py, commit "
                      "the lane pair, then rerun without VT_DUMP_IDS");
    }
    // RED-BY-ABSENCE: a loud skip naming the missing pair is NOT a pass —
    // the opt-in battery must FAIL until the lane pair exists.
    const bool have_pair = have_anchor && have_gap;
    REQUIRE_MESSAGE(have_pair == true,
                    label << ": the LANE PAIR IS ABSENT in " << gdir
                          << " (our_ids.npy: " << have_anchor
                          << ", neartie_gap_mnats.npy: " << have_gap
                          << ") — capture with VT_DUMP_IDS=1 under "
                             "VT_TT_KEEPQUANT_INT8DOT=1, build the pair with "
                             "scripts/qwen3-neartie-gap-llamacpp-oracle.py "
                             "(pinned llama.cpp b10451 oracle), and commit it. "
                             "This failure is the red-first evidence");
  }

  MESSAGE(label << ": loading via FromModelDir(" << model << ")...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(model,
                                                    vllm::entrypoints::EngineParams{});
  const vt::DeviceType run_dev = loaded->runner().device().type;
  REQUIRE_MESSAGE(run_dev == vt::DeviceType::kTENSTORRENT,
                  label << ": the lane battery is Tenstorrent-only; this run "
                           "is on device type "
                        << static_cast<int>(run_dev));
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuant, run_dev));
  vt::ResetOpProviderStats(vt::OpId::kMatmulBTQuant, run_dev);
  vt::EnableOpProviderCallStats(true);

  const parity::NpyArray g = parity::LoadNpy((gdir / "greedy_ids.npy").string());
  const parity::NpyArray o = parity::LoadNpy((gdir / "our_ids.npy").string());
  const parity::NpyArray gap = parity::LoadNpy((gdir / "neartie_gap_mnats.npy").string());
  REQUIRE(g.dtype == "<i4");
  REQUIRE(o.dtype == "<i4");
  REQUIRE(gap.dtype == "<i4");
  const int64_t N = g.shape[0];
  const int64_t T = g.shape[1];
  REQUIRE(static_cast<size_t>(N) == Prompts().size());
  REQUIRE(o.shape.size() == 2);
  REQUIRE(o.shape[0] == N);
  REQUIRE(o.shape[1] == T);
  REQUIRE(gap.shape.size() == 2);
  REQUIRE(gap.shape[0] == N);
  REQUIRE(gap.shape[1] == T);
  const int32_t* gd = AsI32(g);
  const int32_t* od = AsI32(o);
  const int32_t* gapd = AsI32(gap);

  if (vt::tenstorrent::DecodeCaptureEnabled())
    vt::tenstorrent::ResetKeepQuantCaptureStagingWritesForTest();

  int strict_exact = 0;
  int neartie_only = 0;
  int fail = 0;
  int32_t worst_gap = 0;
  int worst_i = -1, worst_j = -1;
  std::vector<int32_t> run1;
  for (int64_t i = 0; i < N; ++i) {
    const vllm::RequestOutput out = loaded->engine().generate(
        Prompts()[static_cast<size_t>(i)], Greedy(static_cast<int>(T)),
        "lane" + std::to_string(i));
    REQUIRE(out.finished);
    const std::vector<int32_t>& got = out.outputs[0].token_ids;
    REQUIRE(static_cast<int64_t>(got.size()) == T);
    run1.insert(run1.end(), got.begin(), got.end());

    int first_div = -1;
    for (int64_t j = 0; j < T; ++j) {
      if (got[static_cast<size_t>(j)] != od[i * T + j]) {
        first_div = static_cast<int>(j);
        break;
      }
    }
    REQUIRE_MESSAGE(first_div < 0,
                    label << " anchor drift prompt[" << i << "] tok=" << first_div
                          << " engine="
                          << (first_div < 0 ? -1 : got[static_cast<size_t>(first_div)])
                          << " committed lane anchor="
                          << (first_div < 0 ? -1 : od[i * T + first_div])
                          << " — REGRESSION SUSPECTED: the lane capture is the "
                             "fixed lever path's sequence; re-derive only after "
                             "the drift is proven a justified numerical change");
    bool exact = true;
    bool prompt_ok = true;
    int first_bad = -1;
    for (int64_t j = 0; j < T; ++j) {
      if (got[static_cast<size_t>(j)] != gd[i * T + j]) exact = false;
      const int32_t mn = gapd[i * T + j];
      if (mn > worst_gap) {
        worst_gap = mn;
        worst_i = static_cast<int>(i);
        worst_j = static_cast<int>(j);
      }
      if (mn > kLaneBandMnats) {
        prompt_ok = false;
        if (first_bad < 0) first_bad = static_cast<int>(j);
      }
    }
    if (!prompt_ok) {
      ++fail;
      MESSAGE(label << " FORWARD DIVERGENCE prompt[" << i << "] tok=" << first_bad
              << " our=" << got[static_cast<size_t>(first_bad)]
              << " llamacpp_greedy=" << gd[i * T + first_bad]
              << " gap=" << (gapd[i * T + first_bad] / 1000.0) << " nats (> "
              << (kLaneBandMnats / 1000.0) << ") \"" << out.outputs[0].text
              << "\"");
    } else if (exact) {
      ++strict_exact;
    } else {
      ++neartie_only;
    }
    CHECK(prompt_ok);
  }
  {
    const int64_t staged = vt::tenstorrent::KeepQuantCaptureStagingWrites();
    CHECK_MESSAGE(staged == 0,
                  label << ": keep-quant decode staged " << staged
                        << " word uploads DURING the captured e2e (the #2812 "
                           "class)");
    MESSAGE(label << ": device trace demand at last capture end = "
            << vt::tenstorrent::LastTraceBytesForTest()
            << " B (lane run 1; dynamic trace region)");
  }
  const int64_t selections =
      vt::GetOpProviderStats(vt::OpId::kMatmulBTQuant, run_dev).selections;
  const int64_t declines =
      vt::GetOpProviderStats(vt::OpId::kMatmulBTQuant, run_dev).declines;
  CHECK_MESSAGE(selections > 0,
                label << ": kMatmulBTQuant never dispatched (lane e2e reach)");
  CHECK_MESSAGE(declines == 0,
                label << ": kMatmulBTQuant DECLINED and fell back");
  vt::EnableOpProviderCallStats(false);

  // Determinism leg: a SECOND engine load in this process is a SECOND decode
  // capture (the op-level int8-dot capture test's x2 byte-identity mechanism
  // at vehicle scale). The 16-prompt token stream must be byte-identical.
  loaded.reset();
  MESSAGE(label << ": determinism leg — reloading the engine (capture x2)...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded2 =
      vllm::entrypoints::LoadedEngine::FromModelDir(model,
                                                    vllm::entrypoints::EngineParams{});
  std::vector<int32_t> run2;
  for (int64_t i = 0; i < N; ++i) {
    const vllm::RequestOutput out = loaded2->engine().generate(
        Prompts()[static_cast<size_t>(i)], Greedy(static_cast<int>(T)),
        "lane2-" + std::to_string(i));
    REQUIRE(out.finished);
    const std::vector<int32_t>& got = out.outputs[0].token_ids;
    REQUIRE(static_cast<int64_t>(got.size()) == T);
    run2.insert(run2.end(), got.begin(), got.end());
  }
  REQUIRE_MESSAGE(run1 == run2,
                  label << ": capture x2 token streams DIVERGE at vehicle "
                           "scale — the lane decode is not deterministic "
                           "across two in-process captures");
  MESSAGE(label << ": capture x2 byte-identity at vehicle scale: PASS; trace "
                  "demand run 2 = "
          << vt::tenstorrent::LastTraceBytesForTest() << " B");

  MESSAGE(label << " lane correctness gate: " << (strict_exact + neartie_only)
          << "/" << N << " prompts PASS  (STRICT token-exact vs pinned "
                          "llama.cpp greedy: "
          << strict_exact << "/" << N << "; near-tie-band only: " << neartie_only
          << "/" << N << "; max gap " << (worst_gap / 1000.0) << " nats @ prompt["
          << worst_i << "] tok=" << worst_j << "; band " << kLaneBandMnats
          << " mnats; " << fail << " forward-divergent)");
  REQUIRE(fail == 0);
}

}  // namespace

// Qwen3.5-0.8B (GDN hybrid: linear-attention recurrence + full-attention
// layers) — the first GDN-architecture gate, ROCm-oracle-backed (issue #41 M4).
TEST_CASE("qwen3.5-0.8B GDN paged-engine greedy near-tie correctness gate (ROCm, SACRED)") {
  RunGate("qwen35_greedy_0_8b", "qwen3.5-0.8B", parity::Qwen35_08BSnapshot());
}

// KEEPQUANT W3 (issue #2959): the SAME gate shape driven through the
// quantized vehicle — unsloth's Qwen3.5-0.8B Q4_K_M GGUF, the mixed-quant
// artifact that forced W3's decode set (Q6_K token_embd, Q5_K attn_qkv/
// ssm_out, Q8_0 ssm_alpha/ssm_beta; see the row spec's falsification
// section). On Tenstorrent the matmul weights keep their blocks (the
// widened kTENSTORRENT predicate) and every GEMM decodes on-core from the
// resident i32 word shadow; the Q6_K embedding table still expands (the
// gather arm needs kEmbeddingQuant, unregistered on TT) — bounded at 0.8B.
// The Tenstorrent lane gates against its OWN captured pair with the
// teacher-forced near-tie band, exactly the bf16 gate's treatment; the
// oracle is `transformers` on the DEQUANTIZED artifact (never the bf16
// safetensors checkpoint: those logits are a different model's).
// Checkpoint-gated: absent VLLM_CPP_QWEN35_Q4KM_GGUF -> loud SKIP.
TEST_CASE("qwen3.5-0.8B GGUF Q4_K_M paged-engine greedy near-tie gate (Tenstorrent, checkpoint-gated)") {
  const char* gguf = std::getenv("VLLM_CPP_QWEN35_Q4KM_GGUF");
  if (gguf == nullptr || gguf[0] == '\0') {
    SkipGate("qwen35-gguf-q4km",
             "VLLM_CPP_QWEN35_Q4KM_GGUF is absent — set it to the local "
             "Qwen3.5-0.8B-Q4_K_M.gguf (unsloth/Qwen3.5-0.8B-GGUF @ 6ab46149, "
             "sha256 bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a0"
             "6121dc517, 532517120 bytes) to run the keep-quant vehicle gate");
  }
  RunGate("qwen35_gguf_q4km", "qwen35-gguf-q4km", std::string(gguf),
          /*keep_quant=*/true);
}

// KEEPQUANT W4a (issue #3030): the SAME gate shape on the 27B dense arm —
// unsloth's Qwen3.8-27B Q4_K_M GGUF, the checkpoint the 0823 token-gate
// evidence (docs/bench-evidence/qwen38-27b-q4km-token-gate-20260823.md)
// measured. This file ships the head tensors (the 15 blk.64.* nextn/MTP
// tensors) and the loader NAMES what it leaves unread on a spec-off load; the
// gate loads the trunk through the same production surface and proves the
// keep-quant decode set end to end at 27B scale. The oracle is the pinned
// llama.cpp b10451 (see .agents/oracles/llama-cpp.md — the executed path is
// pinned: repack ON, flash-attn enabled, ubatch 512), NOT transformers: the
// artifact is the quantized file itself, and llama.cpp is the registry's
// CPU/GGUF k-quant floor for it.
// Checkpoint-gated: absent VLLM_CPP_QWEN38_27B_GGUF -> loud SKIP.
TEST_CASE("qwen3.8-27B GGUF Q4_K_M paged-engine greedy near-tie gate (Tenstorrent, checkpoint-gated)") {
  const char* gguf = std::getenv("VLLM_CPP_QWEN38_27B_GGUF");
  if (gguf == nullptr || gguf[0] == '\0') {
    SkipGate("qwen38-gguf-q4km-27b",
             "VLLM_CPP_QWEN38_27B_GGUF is absent — set it to the local "
             "Qwen3.8-27B-Q4_K_M.gguf (unsloth/Qwen3.8-27B-GGUF @ fe1e2a23, "
             "sha256 7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b"
             "6c6fe169, 17106775008 bytes) to run the 27B keep-quant gate");
  }
  RunGate("qwen38_gguf_q4km_27b", "qwen38-gguf-q4km-27b", std::string(gguf),
          /*keep_quant=*/true);
}

// KEEPQUANT W4c (issue #3079): the int8-dot lane's e2e battery. The default
// flipped ON (tenstorrent-keepquant-int8dot-default), so unset runs it; an
// explicit `=0` opt-out keeps the W4a default-config vehicle battery.
TEST_CASE("qwen3.5-0.8B GGUF Q4_K_M int8-dot lane e2e battery (Tenstorrent, lever opt-in)") {
  const char* lever = std::getenv("VT_TT_KEEPQUANT_INT8DOT");
  if (lever != nullptr && lever[0] != '\0' && std::string_view(lever) == "0") {
    MESSAGE("SKIPPED: VT_TT_KEEPQUANT_INT8DOT=0 opts out to the W4a grouped "
            "arm — the int8-dot lane is the default since the flip "
            "(tenstorrent-keepquant-int8dot-default); this battery runs on "
            "the default configuration or with =1");
    return;
  }
  const char* gguf = std::getenv("VLLM_CPP_QWEN35_Q4KM_GGUF");
  if (gguf == nullptr || gguf[0] == '\0') {
    SkipGate("qwen35-gguf-q4km-lanegate",
             "VLLM_CPP_QWEN35_Q4KM_GGUF is absent — set it to the local "
             "Qwen3.5-0.8B-Q4_K_M.gguf (unsloth/Qwen3.5-0.8B-GGUF @ 6ab46149, "
             "sha256 bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a0"
             "6121dc517, 532517120 bytes) to run the int8-dot lane battery");
  }
  RunLaneGate("qwen35_gguf_q4km_lanegate", "qwen35-gguf-q4km-lanegate",
              std::string(gguf));
}

// ─── TT-GDN-REGION-REPLAY: the served-replay gates ──────────────────────────
//
// The dense captured arm's replays must SERVE, and the SERVED stream must be
// byte-identical to the eager stream. The design is recorded in
// ISSUE-LOCAL-01M3G75X89F89R331165AE6TMS and the spec
// .agents/specs/tenstorrent-gdn-region-replay.md (gate 1): the arm's GDN state
// slots bind persistently in-region (the W3 discipline extended to the ssm and
// conv caches — the captured trace reads and writes the SAME device shadow),
// the two post-replay expected_cur_pos continuation increments make replays
// serve (qwen3.cpp:971/:1118's mirrors, landed by the #2469 continuation port),
// and the served tokens equal the eager stream.
//
// RED-FIRST CONTRACT (both cases): on the pre-fix tree the mechanical half
// passed (replays > captures — the increments' red) while the token half
// failed — the anchor's first served replay wrote 248,320/248,320 exact-zero
// logits (argmax 0 → token 0), the bf16 9B's first served replay diverged
// coherently ([220,16,220,220] vs the eager [220,16,220,16]) — the stale
// state-slot binding the fix removes (the trace's baked shadow-read address
// was replaced every step by the out-of-place state commit).
//
// The workload is the recorded anchor leg verbatim
// (docs/bench-evidence/tt-dense-embed-in-region-20260927.md): the first four
// prompts of tests/fixtures/tt-int8dot-sweep-sharegpt-64-20260923.json (their
// exact `conversations[0].value` strings, embedded so the gate needs no
// dataset plumbing and the prompt bytes cannot drift), greedy temperature-0,
// --output-len 4, --ignore-eos, concurrency 1, and the anchor leg's own
// engine geometry (max_num_seqs 1, block 32, 256 blocks, auto-fit
// max_model_len — the parameters the capture runs cleanly at).
//
// The expected streams are the recorded EAGER references: the anchor's
// main-line stream (sha256 13c3f70b3611ff6b59fef413357704ca2bcd84251c72baff45
// 0a935de1a049cb of the [4,4] token matrix) and the 9B's fresh-process eager
// leg ([220,16,220,16] on the row-0 prompt). The 9B's FOURTH cell is an
// exact tie in the transformers oracle (both tied tokens at 0.00 mnats —
// the case's comment carries the adjudication), so that one cell accepts
// either tied token; the anchor is byte-identical everywhere.
namespace {

// The first four prompts of the sweep fixture, verbatim (see above).
const std::vector<std::string>& SweepFixturePrompts() {
  static const std::vector<std::string> p = {
      "1 2 world 1 hello 1 2 hello 1 world 2 hello 1 1 hello world 2 world "
      "world hello 2 world 1 1 hello 2 1 1 2 world hello 2 2 2 2 2 2 1 world "
      "hello 2",
      "world hello world 1 world 2 world hello 2 hello world world hello world "
      "2 1 hello hello 2 world world hello 2 1 2 world hello hello hello world "
      "hello 1 1 2 2 world hello 2 world 2 hello 1 1 world 1 hello",
      "2 world 1 2 world world 1 hello world world hello 2 2 world 2 2 hello "
      "world 2 hello hello hello hello hello world 1 1 world 2 hello hello "
      "hello 2 1 2 2 world 1 hello 2 1 hello 1 2",
      "hello world world 1 1 1 1 2 1 1 hello 1 hello hello hello hello world 1 "
      "world 2 1 hello hello hello 1 world 1 world world 2 hello hello 1 2 2 "
      "world world hello 2 hello hello hello world hello 1",
  };
  return p;
}

// Drive the served-replay gate: every prompt's greedy stream must equal the
// recorded eager reference, and the decode graph must have SERVED replays
// (replays strictly greater than the captures' own launches). `oracle_ties`
// names cells where the ORACLE ITSELF cannot separate the tokens — the
// teacher-forced adjudication recorded in the owning case's comment — and
// accepts any listed token there; every other cell is byte-identity.
void RunServedReplayGate(
    const char* label, const std::string& model,
    const std::vector<std::string>& prompts,
    const std::vector<std::vector<int32_t>>& expected,
    const std::map<std::pair<size_t, size_t>, std::vector<int32_t>>&
        oracle_ties = {}) {
  // The anchor leg's engine geometry (see the block comment above).
  vllm::entrypoints::EngineParams params;
  params.max_num_seqs = 1;
  params.block_size = 32;
  params.num_blocks = 256;
  MESSAGE(label << ": loading via FromModelDir(" << model << ")...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(model, params);
  const vt::DeviceType run_dev = loaded->runner().device().type;
  if (run_dev != vt::DeviceType::kTENSTORRENT) {
    SkipGate(label, "the served-replay gate is the Tenstorrent captured arm's; "
                    "this run is on device type " +
                        std::to_string(static_cast<int>(run_dev)));
  }
  if (!vt::GraphCaptureEnabled() || !vt::tenstorrent::DecodeCaptureEnabled()) {
    SkipGate(label, "capture is disabled on this run (VLLM_CPP_CUDAGRAPH=0 or "
                    "VT_TT_HOST_FREE_DECODE=0) — the served-replay gate is the "
                    "captured arm's");
  }
  vt::ResetGraphBreakStats();
  std::vector<std::vector<int32_t>> got;
  for (size_t i = 0; i < prompts.size(); ++i) {
    vllm::SamplingParams sp;
    sp.temperature = 0.0;
    sp.max_tokens = static_cast<int>(expected[i].size());
    sp.ignore_eos = true;
    sp.PostInit();
    const vllm::RequestOutput out =
        loaded->engine().generate(prompts[i], sp, "sr" + std::to_string(i));
    got.push_back(out.outputs[0].token_ids);
  }
  const vt::GraphBreakStats stats = vt::GetGraphBreakStats();
  CHECK_MESSAGE(stats.replays > stats.segments_captured,
                label << ": the dense decode graph never SERVED a replay: "
                         "replays (" << stats.replays << ") never exceeded the "
                         "captures' own launches (segments_captured "
                      << stats.segments_captured
                      << ") — the #2469 continuation-port masking defect (the "
                         "missing expected_cur_pos increments)");
  REQUIRE(got.size() == expected.size());
  bool all_ok = true;
  for (size_t i = 0; i < got.size(); ++i) {
    bool row_ok = got[i].size() == expected[i].size();
    if (row_ok) {
      for (size_t k = 0; k < got[i].size(); ++k) {
        const auto tie = oracle_ties.find({i, k});
        if (tie != oracle_ties.end()) {
          bool in_tie = false;
          for (int32_t t : tie->second) in_tie = in_tie || got[i][k] == t;
          if (!in_tie) {
            row_ok = false;
            std::fprintf(stderr,
                         "%s: row %zu token %zu: served %d, oracle-tied "
                         "alternatives {",
                         label, i, k, got[i][k]);
            for (size_t a = 0; a < tie->second.size(); ++a)
              std::fprintf(stderr, "%s%d", a ? ", " : "", tie->second[a]);
            std::fprintf(stderr, "}\n");
          }
          continue;
        }
        if (got[i][k] != expected[i][k]) {
          row_ok = false;
          std::fprintf(stderr,
                       "%s: row %zu token %zu: served %d, eager reference %d\n",
                       label, i, k, got[i][k], expected[i][k]);
        }
      }
    } else {
      std::fprintf(stderr, "%s: row %zu: %zu served tokens vs %zu expected\n",
                   label, i, got[i].size(), expected[i].size());
    }
    if (!row_ok) all_ok = false;
  }
  std::ostringstream rows;
  for (size_t i = 0; i < got.size(); ++i) {
    rows << "[";
    for (size_t k = 0; k < got[i].size(); ++k) {
      if (k) rows << ",";
      rows << got[i][k];
    }
    rows << "]";
    if (i + 1 < got.size()) rows << ",";
  }
  CHECK_MESSAGE(all_ok, label << ": the SERVED stream diverged from the eager "
                                 "reference; served=[" << rows.str() << "]");
}

}  // namespace

// The anchor (27B APEX I-Nano, quantized): served stream must equal the
// recorded anchor [[220,17,220,17],[220,17,220,17],[220,16,220,17],
// [220,16,220,16]] — the same bytes as main's recorded anchor leg.
// Checkpoint-gated: absent VLLM_CPP_QWEN38_27B_APEX_NANO_GGUF -> loud SKIP.
TEST_CASE("qwen3.8-27B APEX dense captured arm SERVES replays byte-identical to eager (Tenstorrent, checkpoint-gated)") {
  const char* gguf = std::getenv("VLLM_CPP_QWEN38_27B_APEX_NANO_GGUF");
  if (gguf == nullptr || gguf[0] == '\0') {
    SkipGate("qwen38-apex-nano-served-replay",
             "VLLM_CPP_QWEN38_27B_APEX_NANO_GGUF is absent — set it to the "
             "local Qwen3.8-27B-APEX-I-Nano.gguf (mudler-qwen3.8-27B-APEX-gguf "
             "local artifact, 11,240,605,152 bytes) to run the served-replay "
             "gate");
  }
  RunServedReplayGate("qwen38-apex-nano-served-replay", std::string(gguf),
                      SweepFixturePrompts(),
                      {{220, 17, 220, 17},
                       {220, 17, 220, 17},
                       {220, 16, 220, 17},
                       {220, 16, 220, 16}});
}

// The bf16 9B (Qwen3.5-9B-hf, dense GDN-hybrid): the CLEANER instrument of
// the pair. The first three served tokens are byte-identical to the recorded
// eager leg ([220,16,220,16] on the row-0 prompt; the pre-fix defect diverged
// the fourth to 220 — a different-distribution value — and corrupted every
// later step). The FOURTH cell is an EXACT TIE in the transformers oracle:
// teacher-forced on the shared prefix (…,220,16,220), the oracle's top-2 are
// {17, 16} at IDENTICAL log-prob (-0.706930 both; gaps 0.00 mnats, pinned
// oracle env torch 2.7.1+cpu / transformers 5.8.1, bf16, CPU — the keepquant
// row's oracle pin; evidence in this row's bench record) — the eager arm
// took 16, the served arm takes 17, and both are the oracle's argmax. The
// recorded tie treatment applies (the 0.8B gate's own words: "at exact ties
// the arm may take the tied runner-up"), so the cell accepts either tied
// token; the ORACLE's own greedy stream for this prompt is [220,16,220,16].
// Checkpoint-gated: absent VLLM_CPP_QWEN35_9B_DIR -> loud SKIP.
TEST_CASE("qwen3.5-9B dense captured arm SERVES replays byte-identical to eager up to the oracle-tied cell (Tenstorrent, checkpoint-gated)") {
  const char* dir = std::getenv("VLLM_CPP_QWEN35_9B_DIR");
  if (dir == nullptr || dir[0] == '\0') {
    SkipGate("qwen35-9b-served-replay",
             "VLLM_CPP_QWEN35_9B_DIR is absent — set it to the local "
             "Qwen3.5-9B-hf snapshot directory (Qwen3_5ForConditionalGeneration, "
             "32 layers, GDN-hybrid dense) to run the served-replay gate");
  }
  RunServedReplayGate("qwen35-9b-served-replay", std::string(dir),
                      {SweepFixturePrompts()[0]},
                      {{220, 16, 220, 16}},
                      /*oracle_ties=*/{{{0, 3}, {16, 17}}});
}

// TT-GDN-SLOT-CHURN (ISSUE-LOCAL-01M433M0TNT8FWC6SMT4R3700W): the decode
// graph's GDN state-slot bindings are baked at trace-capture time (the
// content-keyed one-hot/scatter ids of GdnSsmIdxEntry / GdnConvIdxEntry,
// tenstorrent_gdn.cpp:558/:1629). Continuous admission — a request retiring
// while others decode, or a queued wave re-admitting onto freed slots —
// changes the batch's state-index CONTENT at a constant padded size S, where
// every earlier single-wave gate leg kept it frozen. On HEAD that is the
// engine-fatal at :558 when a capture lands after a membership change, and
// the replays between membership changes gather the stale capture-time slots
// (the c2 run-to-run nondeterminism). This gate drives exactly that load on
// the 0.8B keep-quant vehicle: six requests with STAGGERED output lengths in
// two overlapping waves, so the decode batch's membership churns at a
// constant S while the captured arm is live. GREEN requires: every request
// finishes, and two independent engine runs over the identical load produce
// byte-identical streams (the determinism the churn artifacts broke).
// Checkpoint-gated: absent VLLM_CPP_QWEN35_Q4KM_GGUF -> loud SKIP.
namespace {

struct MultiWaveRun {
  bool all_finished = false;
  std::vector<std::vector<int32_t>> streams;
};

MultiWaveRun RunMultiWaveChurn(const char* gguf) {
  MultiWaveRun run;
  // max_num_seqs=4 keeps the decode batch AT the capture size S=4 for every
  // step with a non-empty queue: the queued wave backfills a retiring slot on
  // the same step, which is exactly the re-admission path the single-wave
  // legs never ran. (The B < S padded arm is a separate owed item; this gate
  // never claims it.)
  vllm::entrypoints::EngineParams params;
  params.max_num_seqs = 4;
  std::unique_ptr<vllm::entrypoints::LoadedEngine> le =
      vllm::entrypoints::LoadedEngine::FromModelDir(std::string(gguf), params);
  // Wave 1: four short decodes with STAGGERED output lengths, so the
  // batch's membership changes on several consecutive steps.
  std::vector<vllm::v1::AsyncRequest> live =
      le->async_engine().add_request_wave({{"mw0", Prompts()[0], Greedy(2)},
                                           {"mw1", Prompts()[1], Greedy(3)},
                                           {"mw2", Prompts()[2], Greedy(4)},
                                           {"mw3", Prompts()[3], Greedy(5)}});
  // Wave 2: queued IMMEDIATELY behind it, so every wave-1 retirement is
  // backfilled from this wave — the slot set at S=4 churns while the
  // captured decode graph is live.
  std::vector<vllm::v1::AsyncRequest> late =
      le->async_engine().add_request_wave({{"mw4", Prompts()[4], Greedy(20)},
                                           {"mw5", Prompts()[5], Greedy(20)},
                                           {"mw6", Prompts()[6], Greedy(20)},
                                           {"mw7", Prompts()[7], Greedy(20)}});
  // get_output returns STREAMING chunks; drain to the terminal output the
  // same loop AsyncLLM::generate uses.
  auto drain = [&](const vllm::v1::AsyncRequest& req) {
    vllm::RequestOutput out{};
    std::vector<int32_t> ids;
    bool finished = false;
    for (;;) {
      std::optional<vllm::RequestOutput> ready =
          le->async_engine().get_output_nowait(req);
      out = ready.has_value() ? std::move(*ready)
                              : le->async_engine().get_output(req);
      if (!out.outputs.empty())
        for (const auto& o : out.outputs)
          ids.insert(ids.end(), o.token_ids.begin(), o.token_ids.end());
      if (out.finished) { finished = true; break; }
    }
    return std::pair<bool, std::vector<int32_t>>{finished, std::move(ids)};
  };
  run.all_finished = true;
  size_t idx = 0;
  for (const auto& req : live) {
    auto [fin, ids] = drain(req);
    if (!fin) run.all_finished = false;
    MESSAGE("live[" << idx << "] finished=" << fin << " tokens=" << ids.size());
    ++idx;
    run.streams.push_back(std::move(ids));
  }
  idx = 0;
  for (const auto& req : late) {
    auto [fin, ids] = drain(req);
    if (!fin) run.all_finished = false;
    MESSAGE("late[" << idx << "] finished=" << fin << " tokens=" << ids.size());
    ++idx;
    run.streams.push_back(std::move(ids));
  }
  return run;
}

}  // namespace

TEST_CASE("qwen3.5-0.8B GGUF multi-wave GDN slot-churn re-admission serves deterministically (Tenstorrent, checkpoint-gated)") {
  const char* gguf = std::getenv("VLLM_CPP_QWEN35_Q4KM_GGUF");
  if (gguf == nullptr || gguf[0] == '\0') {
    SkipGate("qwen35-gguf-q4km-multivave",
             "VLLM_CPP_QWEN35_Q4KM_GGUF is absent — set it to the local "
             "Qwen3.5-0.8B-Q4_K_M.gguf to run the multi-wave slot-churn gate");
  }
  const MultiWaveRun first = RunMultiWaveChurn(gguf);
  REQUIRE(first.all_finished);
  REQUIRE(first.streams.size() == 6);
  for (size_t i = 0; i < first.streams.size(); ++i)
    REQUIRE_MESSAGE(!first.streams[i].empty(),
                    "wave request " << i << " produced no tokens");
  // Determinism: a second identical load must reproduce the first run's
  // streams byte-for-byte. The stale-binding replays made this flaky (the
  // c2 INT8DOT A/B legs' run-to-run flip).
  const MultiWaveRun second = RunMultiWaveChurn(gguf);
  REQUIRE(second.all_finished);
  for (size_t i = 0; i < first.streams.size(); ++i) {
    REQUIRE_MESSAGE(second.streams[i] == first.streams[i],
                    "run-to-run stream mismatch on request " << i);
  }
}
