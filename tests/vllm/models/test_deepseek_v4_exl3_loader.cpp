// MODEL-DSV4-EXL3 W1b — the rank-sliced EXL3 loader arm.
//
// The SparkInfer artifact `0xSero/deepseek-v4-flash-0731-spark` stores its 216
// routed experts as EXL3 trellis tensors SLICED ACROSS FOUR TENSOR-PARALLEL
// RANKS and everything else ("carried") as the un-requantized DeepSeek-V4 FP8
// source tensors. Its `config.json` declares
// `quantization_config.quant_method == "exl3"` with
// `version == "rank-sliced-deepseek-v4-v1"` and
// `hybrid_tr3_tail.tensor_schema ==
//  "layers.{L}.ffn.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}"`.
//
// The slicing is upstream's own `LinearEXL3.tp_import_split`
// (exllamav3 @ 2398c056, `modules/quant/exl3.py:296-313`): a split on the OUT
// features takes `svh[first:last]` and `trellis[:, first//16:last//16]`, and a
// split on the IN features takes `suh[first:last]` and
// `trellis[first//16:last//16, :]`. w1 and w3 are out-split, w2 is in-split, so
// coalescing back to TP1 is pure concatenation and is LOSSLESS: 16x16 trellis
// tiles are independent and every rank boundary here is a multiple of 128, the
// Hadamard block size (`exl3_lib/quantize.py:15`).
//
// This suite drives the PRODUCTION loader entry
// (`vllm::LoadDeepseekV4ForCausalLMWeights`, what
// `deepseek_v4_registry.cpp:89` calls) over a hermetic four-rank fixture the
// test writes itself, and it checks the coalesced owners BYTE FOR BYTE against
// the rank inputs it wrote. Real-checkpoint residency (this arm copies the
// tower into host owner buffers, which is ~100 GB on the real artifact) is the
// named MODEL-DSV4-EXL3 W2 residual; see the spec's `## Owed`.
#include <doctest/doctest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/mxfp4_dequant.h"   // E8M0ToF32
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"   // F8E4M3ToF32
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/deepseek_v4.h"
#include "vllm/model_executor/models/deepseek_v4_compressor.h"  // Fp8DsMlaPageLayout
#include "vllm/model_executor/models/model_registry.h"
#include "vt/ops.h"                                // kFp8DsMla* geometry
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/model_executor/models/qwen3_5.h"   // PagedKvCache, GdnStateCache
#include "vllm/v1/attention/backend.h"             // CommonAttentionMetadata
#include "vllm/v1/attention/backends/gdn_attn.h"   // GDNAttentionMetadata
#include "vllm/v1/core/kv_cache_utils.h"  // host_available_memory_bytes
#include "vt/backend.h"
#include "vt/dtype.h"

// The fixture is SHARED with tests/vllm/models/test_deepseek_v4_exl3_forward.cpp,
// which drives the same production loader entry and then runs a forward over
// what it produced. One fixture is what makes the reachability claim falsifiable
// (#1923): the forward suite used to build `DeepseekV4Weights` by hand.
#include "dsv4_exl3_fixture.h"

using namespace dsv4_exl3_fixture;  // NOLINT(build/namespaces) — test fixture

// ───────────────────────────────────────────────────────────────────────────
TEST_CASE("dsv4 exl3: the rank-sliced arm is detected and coalesces TP4 -> TP1") {
  auto f = BuildFixture();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);

  REQUIRE(w.has_exl3_weights);
  CHECK(w.exl3.tp == kTp);
  CHECK(w.exl3.bits == kBits);
  CHECK(w.exl3.codebook == "mcg");
  CHECK(w.exl3.version == "rank-sliced-deepseek-v4-v1");
  REQUIRE(w.exl3.layers.size() == static_cast<size_t>(kLayers));
  REQUIRE(w.exl3.layers[0].experts.size() == static_cast<size_t>(kExperts));

  // 35 carried + 1 layer * 2 experts * 3 projections * 4 ranks * 4 tensors.
  CHECK(w.accounted_tensors == 35 + kLayers * kExperts * 3 * kTp * 4);
  // The MTP tail is skipped exactly as vLLM's own DeepSeek-V4 loader skips it
  // (`AutoWeightsLoader(skip_substrs=["mtp."])`, nvidia/model.py:1474) — but
  // COUNTED, so the skip is visible rather than silent. On the real artifact
  // these are 3985 NVFP4 draft-head tensors.
  CHECK(w.exl3.skipped_mtp_tensors == 2);

  const vllm::DeepseekV4Exl3Expert& e0 = w.exl3.layers[0].experts[0];
  for (const vllm::DeepseekV4Exl3Linear* lin : {&e0.w1, &e0.w3}) {
    CHECK(lin->in_features == kHidden);
    CHECK(lin->out_features == kInter);
    CHECK(lin->bits == kBits);
    CHECK(lin->suh.size() == static_cast<size_t>(kHidden));
    CHECK(lin->svh.size() == static_cast<size_t>(kInter));
    CHECK(lin->trellis.size() == static_cast<size_t>(TrellisElems(kHidden, kInter)));
  }
  CHECK(e0.w2.in_features == kInter);
  CHECK(e0.w2.out_features == kHidden);
  CHECK(e0.w2.suh.size() == static_cast<size_t>(kInter));
  CHECK(e0.w2.svh.size() == static_cast<size_t>(kHidden));
  CHECK(e0.w2.trellis.size() == static_cast<size_t>(TrellisElems(kInter, kHidden)));
  CHECK(e0.w1.mcg == -877912083);
}

TEST_CASE("dsv4 exl3: the coalesced owners equal the rank inputs BYTE FOR BYTE") {
  auto f = BuildFixture();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);
  REQUIRE(w.has_exl3_weights);

  int trellis_mismatch = 0, sign_mismatch = 0;
  for (int x = 0; x < kExperts; ++x) {
    const vllm::DeepseekV4Exl3Expert& ex = w.exl3.layers[0].experts[x];
    int proj_index = 0;
    for (const char* proj : {"w1", "w2", "w3"}) {
      const vllm::DeepseekV4Exl3Linear& lin =
          std::strcmp(proj, "w1") == 0 ? ex.w1
                                       : (std::strcmp(proj, "w2") == 0 ? ex.w2 : ex.w3);
      const bool out_split = SplitsOut(proj);
      const int64_t k = lin.in_features, n = lin.out_features;
      const int64_t tn = n / 16;
      const int64_t words = 16 * kBits;
      for (int r = 0; r < kTp; ++r) {
        // The per-rank slice this rank file was written with.
        const int64_t rk = out_split ? k : k / kTp;
        const int64_t rn = out_split ? n / kTp : n;
        const int64_t rtk = rk / 16, rtn = rn / 16;
        for (int64_t i = 0; i < rtk; ++i) {
          for (int64_t j = 0; j < rtn; ++j) {
            for (int64_t t = 0; t < words; ++t) {
              const uint16_t want =
                  TrellisWord(x, proj_index, r, (i * rtn + j) * words + t);
              // OUT split concatenates along trellis dim 1, IN split along dim 0.
              const int64_t gi = out_split ? i : (r * rtk + i);
              const int64_t gj = out_split ? (r * rtn + j) : j;
              if (lin.trellis[static_cast<size_t>((gi * tn + gj) * words + t)] != want)
                ++trellis_mismatch;
            }
          }
        }
        // The SPLIT sign vector is concatenated; the INVARIANT one came whole.
        if (out_split) {
          for (int64_t j = 0; j < rn; ++j)
            if (lin.svh[static_cast<size_t>(r * rn + j)] !=
                SignWord(x, proj_index, r, j, 1))
              ++sign_mismatch;
        } else {
          for (int64_t i = 0; i < rk; ++i)
            if (lin.suh[static_cast<size_t>(r * rk + i)] !=
                SignWord(x, proj_index, r, i, 0))
              ++sign_mismatch;
        }
      }
      for (int64_t i = 0; i < (out_split ? k : 0); ++i)
        if (lin.suh[static_cast<size_t>(i)] != SignWord(x, proj_index, 0, i, 0))
          ++sign_mismatch;
      for (int64_t j = 0; j < (out_split ? 0 : n); ++j)
        if (lin.svh[static_cast<size_t>(j)] != SignWord(x, proj_index, 0, j, 1))
          ++sign_mismatch;
      ++proj_index;
    }
  }
  CHECK(trellis_mismatch == 0);
  CHECK(sign_mismatch == 0);

  // The split inputs are NOT retained: the tower holds exactly the coalesced
  // bytes, never the four rank copies as well.
  CHECK(vllm::DeepseekV4Exl3ResidentBytes(w) == ExpectedTowerBytes());
}

TEST_CASE("dsv4 exl3: the LOAD reports the tower's residency and refuses one that does not fit") {
  // WHY THIS CASE EXISTS. `DeepseekV4Exl3ResidentBytes` had no production caller
  // at all: only the case above called it, which measures a function rather than
  // a capability ("Nothing lands dead", AGENTS.md). It is also exactly the
  // instrument the row's residency RISK needs — the real artifact's trellis alone
  // is ~83.5 GiB (43 layers x 216 experts x 3 projections) on a box whose unified
  // memory OOM-reboots. So the load itself now reports the figure and refuses a
  // projection the host cannot hold, and this case gates both halves.
  auto f = BuildFixture();
  vllm::DeepseekV4Weights w;
  const std::string log = CaptureStderr([&] {
    w = vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);
  });
  REQUIRE(w.has_exl3_weights);

  // REACHABILITY: this line is emitted BY THE LOAD, so deleting the production
  // call site takes this case red.
  CAPTURE(log);
  CHECK(log.find("[vt load] dsv4-exl3:") != std::string::npos);
  CHECK(log.find("resident_bytes=" + std::to_string(ExpectedTowerBytes())) !=
        std::string::npos);

  // ...AND IT MUST CARRY THE REAL BUDGET. The two assertions above match BOTH
  // branches of the report, which is how the fresh review (2026-08-24, MINOR-1)
  // wired `host_available` to a literal 0 at the production call site — symbol
  // still referenced, so it compiled (ninja rc=0, 3 steps) — and watched this
  // suite stay 6/6 66/66 SUCCESS while the refusal went silently inert and the
  // load printed `/proc/meminfo unreadable` on a host where it reads fine. The
  // budget the load ACTUALLY used is therefore asserted, branched on whether
  // THIS host can read the pool at all.
  const int64_t budget_now = vllm::v1::host_available_memory_bytes();
  CAPTURE(budget_now);
  if (budget_now > 0) {
    CHECK(log.find("host MemAvailable=") != std::string::npos);
    CHECK(log.find("MemAvailable unknown") == std::string::npos);
    // Same POOL, not merely some non-zero constant. The window is deliberately
    // wide (1/8x .. 8x) because MemAvailable moves under other work on the box
    // between the load and this second read; it is still far tighter than the
    // 1 MiB and 1 TiB brackets the refusal cases below inject, so a call site
    // rewired to either of those constants fails here.
    const double reported = ReportedMemAvailableGiB(log);
    const double now_gib = static_cast<double>(budget_now) / (1024.0 * 1024.0 * 1024.0);
    CAPTURE(reported);
    CAPTURE(now_gib);
    CHECK(reported > 0.0);
    CHECK(reported >= now_gib / 8.0);
    CHECK(reported <= now_gib * 8.0);
  } else {
    // /proc/meminfo is genuinely unreadable here, so the unknown branch is the
    // CORRECT report and the refusal is correctly inert.
    CHECK(log.find("MemAvailable unknown") != std::string::npos);
  }

  // The REFUSAL, driven through the same production function the load calls,
  // with the budget INJECTED. `check_enough_state_memory`
  // (`vllm/v1/core/kv_cache_utils.h`) is parameterised for exactly this reason:
  // a refusal observable only on a box of a chosen size is not gateable.
  const std::string refusal = ThrowMessage([&] {
    (void)vllm::ReportDeepseekV4Exl3Residency(w, /*layers_done=*/1,
                                              /*layers_total=*/43,
                                              /*host_available_bytes=*/1 << 20);
  });
  CAPTURE(refusal);
  CHECK(Mentions(refusal, "MODEL-DSV4-EXL3"));
  CHECK(Mentions(refusal, "MemAvailable"));
  // An UNKNOWN budget never refuses — `host_available_memory_bytes()` returns 0
  // when /proc/meminfo is unreadable, and `VT_DSV4_EXL3_HOST_BUDGET=0` hands the
  // reporter the same 0 on purpose. An unknown budget must not become a false
  // refusal (`host_available_memory_bytes`, `kv_cache_utils.cpp`, keeps the same
  // polarity).
  CHECK(ThrowMessage([&] {
          (void)vllm::ReportDeepseekV4Exl3Residency(w, 1, 43, 0);
        }).empty());
  // A budget that holds the projection does not refuse either.
  CHECK(ThrowMessage([&] {
          (void)vllm::ReportDeepseekV4Exl3Residency(w, 1, 43, int64_t{1} << 40);
        }).empty());
  // THE INCLUSIVE EDGE (fresh review, NIT-1). The two cases above bracket the
  // threshold at 1 MiB and 1 TiB against a ~304 KiB tower, which catches a
  // direction flip but NOT `projected <= budget` narrowing to `projected <
  // budget`. A projection that EQUALS the budget must load: with layers_done=1
  // the projection is exactly `tower * layers_total` PLUS the materialized
  // carried tower, which W1c made non-zero and which the refusal now prices
  // (a refusal that measured only one of the two towers would let the other
  // take the box down).
  const int64_t host_bytes = vllm::DeepseekV4HostResidentBytes(w);
  CHECK(host_bytes > 0);
  CHECK(ThrowMessage([&] {
          (void)vllm::ReportDeepseekV4Exl3Residency(
              w, 1, 43, ExpectedTowerBytes() * 43 + host_bytes);
        }).empty());
  // ...and ONE byte under it refuses. This is what makes the host term
  // load-bearing rather than decorative: dropping it from the sum leaves the
  // edge above green and takes this one red.
  CHECK(!ThrowMessage([&] {
           (void)vllm::ReportDeepseekV4Exl3Residency(
               w, 1, 43, ExpectedTowerBytes() * 43 + host_bytes - 1);
         }).empty());
  // The report names the carried tower too, so a reader sees both numbers.
  CHECK(log.find("host_bytes=" + std::to_string(host_bytes)) != std::string::npos);
}

TEST_CASE(
    "dsv4 exl3: VT_DSV4_EXL3_HOST_BUDGET defaults ON; only a '0'-leading value "
    "disables the refusal") {
  // The refusal's budget is a HEURISTIC (`/proc/meminfo` MemAvailable ignores
  // swap, and in a container reports the HOST's pool rather than the cgroup's
  // — see the caveats at the read site in `LoadDeepseekV4Exl3`), so it ships
  // with a same-binary escape hatch. This pins the PARSE, which is factored into
  // the header precisely so it is gateable without mutating the environment
  // (house shape: `AsyncRunnerFlagIsOn`, tests/vllm/v1/worker/
  // test_async_runner_flag.cpp).
  using vllm::Dsv4Exl3HostBudgetFlagIsOn;
  // Default (unset) is ON: the refusal ships enabled, because on the
  // unified-memory box this arm targets an over-commit reboots the machine.
  CHECK(Dsv4Exl3HostBudgetFlagIsOn(nullptr));
  // Non-'0'-leading values stay ON, including the explicit opt-in and junk.
  CHECK(Dsv4Exl3HostBudgetFlagIsOn("1"));
  CHECK(Dsv4Exl3HostBudgetFlagIsOn(""));
  CHECK(Dsv4Exl3HostBudgetFlagIsOn("on"));
  CHECK(Dsv4Exl3HostBudgetFlagIsOn("true"));
  CHECK(Dsv4Exl3HostBudgetFlagIsOn(" 0"));  // leading space, not a '0' first char
  CHECK(Dsv4Exl3HostBudgetFlagIsOn("10"));
  // Disabled: FIRST character '0'.
  CHECK_FALSE(Dsv4Exl3HostBudgetFlagIsOn("0"));
  CHECK_FALSE(Dsv4Exl3HostBudgetFlagIsOn("00"));
  CHECK_FALSE(Dsv4Exl3HostBudgetFlagIsOn("0abc"));
}

TEST_CASE("dsv4 exl3: a NON-exl3 quantization_config takes the DENSE arm") {
  // The detection predicate was gated only in the POSITIVE direction: every case
  // above hands it an EXL3 checkpoint, so widening `quant_method == "exl3"` to an
  // always-true test left four dsv4 suites green (fresh review, 2026-08-24).
  // That matters because the plain vehicle is not "no quantization_config": the
  // `deepseek_v4_fp8` artifact carries one, and this very checkpoint stores it
  // verbatim as `quantization_config.base_quantization_config`. A regression that
  // widened the predicate would route FP8 into the trellis arm undetected.
  SUBCASE("quant_method fp8 — the deepseek_v4_fp8 vehicle") {
    FixtureOptions opt;
    opt.quant_method = "fp8";
    opt.dense_routed_experts = true;
    auto f = BuildFixture(opt);
    vllm::DeepseekV4Weights w;
    // Through ThrowMessage rather than bare: a widened predicate sends this
    // fixture into the EXL3 arm, which THROWS on the absent `version`, and an
    // uncaught throw is a failed CASE with `assertions: N | N passed` — a red
    // that reads as a pass in the summary line. Captured, it is a named
    // assertion that prints the refusal it got instead.
    const std::string msg = ThrowMessage(
        [&] { w = vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    REQUIRE(msg.empty());
    CHECK_FALSE(w.has_exl3_weights);
    CHECK(w.exl3.layers.empty());
    CHECK(vllm::DeepseekV4Exl3ResidentBytes(w) == 0);
    // 35 carried + 1 layer * 2 experts * 3 projections * 4 NVFP4 suffixes. The
    // dense arm walks its own name-map, which the EXL3 arm does not.
    CHECK(w.accounted_tensors == 35 + kLayers * kExperts * 3 * 4);
  }
  SUBCASE("no quantization_config at all") {
    // This one does NOT discriminate the `== "exl3"` widening, because the null
    // guard above it already returns false — it guards the OTHER widening, a
    // predicate that drops that guard.
    //
    // HOW IT DISCRIMINATES, AND WHY THAT IS NOT AN ASSERTION (fresh review,
    // NIT-2). Dropping `qc == nullptr` from `IsExl3Checkpoint` is not a
    // behavioural widening this subcase can observe by value: it is a null
    // dereference inside the predicate itself, so the process dies with SIGSEGV
    // before any CHECK below runs. MEASURED, not assumed: the binary exits 139
    // and doctest prints `5 | 4 passed | 1 failed | 2 skipped` with
    // `assertions: 63 | 63 passed | 0 failed` and `Status: FAILURE!` — a real
    // ctest red under a CLEAN assertion counter, with two later cases never run.
    // DO NOT grep that counter for this case's verdict; read the exit status.
    //
    // It was left this way on purpose. Making it fail by assertion instead needs
    // the deref replaced with a checked accessor, which means deleting the very
    // guard under test; and the sibling `!qc->is_object()` half cannot be
    // discriminated at all, because `nlohmann::json::contains` is already safe
    // on a non-object, so a fixture with a string-valued `quantization_config`
    // would take the dense arm either way and would assert nothing. A
    // memory-safety defect's discriminator is a crash or a sanitizer, not a
    // CHECK.
    FixtureOptions opt;
    opt.omit_quant_config = true;
    opt.dense_routed_experts = true;
    auto f = BuildFixture(opt);
    vllm::DeepseekV4Weights w;
    const std::string msg = ThrowMessage(
        [&] { w = vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    REQUIRE(msg.empty());
    CHECK_FALSE(w.has_exl3_weights);
    CHECK(vllm::DeepseekV4Exl3ResidentBytes(w) == 0);
    CHECK(w.accounted_tensors == 35 + kLayers * kExperts * 3 * 4);
  }
}

TEST_CASE("dsv4 exl3: the arm is REACHED from the registry's production load") {
  auto f = BuildFixture();
  const vllm::ModelSource source = vllm::ModelSource::FromSafetensors(f->shards);
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = vllm::ModelRegistry::Load(f->config, source));
  CHECK(model != nullptr);
}

// ───────────────────────────────────────────────────────────────────────────
// MODEL-DSV4-EXL3 W1c — the CARRIED tower is MATERIALIZED, not merely counted.
// ───────────────────────────────────────────────────────────────────────────

TEST_CASE("dsv4 exl3 W1c: the load materializes the carried tower and sets the flag") {
  // W1b accounted for these tensors with a presence check and wrote NOTHING, so
  // `has_host_weights` stayed false and every forward entry point refused a
  // checkpoint that had just loaded (#1923: zero tokens from `vllm-server`).
  // Both halves are asserted here — the flag, and the tower the flag claims.
  FixtureOptions opt;
  opt.layers = 2;
  opt.num_hash_layers = 1;
  opt.topk = 2;
  opt.compress_ratios = {0, 4};
  opt.index_n_heads = 2;
  opt.index_head_dim = 4;
  opt.index_topk = 3;
  // The REAL artifact's geometry, which is the only one upstream emits at
  // `cr == 4`: `coff = 1 + (compress_ratio == 4)` is 2
  // (`vllm/models/deepseek_v4/compressor.py:247-248`), so the compressor and
  // indexer-compressor families are doubled while both norms and `weights_proj`
  // are not, and `indexer.wq_b`'s K is `q_lora_rank` (`attention.py:721-726`).
  // This is the tower the loader must materialize; whether the host forward can
  // INDEX it is a separate question, answered by the refusal gated in
  // `test_deepseek_v4_exl3_forward.cpp` (#1970).
  opt.real_dsa_geometry = true;
  auto f = BuildFixture(opt);
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);

  REQUIRE(w.has_exl3_weights);
  REQUIRE(w.has_host_weights);
  REQUIRE(w.host.layers.size() == 2);
  const int64_t H = kHidden;
  const int64_t hc = kHcMult;
  const int64_t hc3 = (2 + hc) * hc;
  const int64_t in_per_group = kHeads * kHeadDim / kOGroups;

  CHECK(w.host.embed.size() == static_cast<size_t>(kVocab * H));
  CHECK(w.host.lm_head.size() == static_cast<size_t>(kVocab * H));
  CHECK(w.host.final_norm_weight.size() == static_cast<size_t>(H));
  CHECK(w.host.hc_head_fn.size() == static_cast<size_t>(hc * hc * H));
  CHECK(w.host.hc_head_base.size() == static_cast<size_t>(hc));

  const vllm::DeepseekV4LayerHostWeights& L0 = w.host.layers[0];
  CHECK(L0.wq_a.size() == static_cast<size_t>(kQLora * H));
  CHECK(L0.wq_b.size() == static_cast<size_t>(kHeads * kHeadDim * kQLora));
  CHECK(L0.wkv.size() == static_cast<size_t>(kHeadDim * H));
  CHECK(L0.wo_a.size() == static_cast<size_t>(kOGroups * kOLora * in_per_group));
  CHECK(L0.wo_b.size() == static_cast<size_t>(H * kOGroups * kOLora));
  CHECK(L0.attn_sink.size() == static_cast<size_t>(kHeads));
  CHECK(L0.hc_attn_fn.size() == static_cast<size_t>(hc3 * hc * H));
  CHECK(L0.shared_w1.size() == static_cast<size_t>(kInter * H));
  CHECK(L0.shared_w2.size() == static_cast<size_t>(H * kInter));
  CHECK(L0.shared_w3.size() == static_cast<size_t>(kInter * H));
  CHECK(L0.gate_weight.size() == static_cast<size_t>(kExperts * H));
  // Layer 0 is the HASH layer: the I64 `tid2eid` narrows to int32, exactly as
  // the GGUF arm narrows it, and the noaux_tc bias is absent.
  CHECK(L0.tid2eid.size() == static_cast<size_t>(kVocab * opt.topk));
  CHECK(L0.gate_bias.empty());

  // Layer 1 carries the DSA compressor + Lightning-Indexer at the width UPSTREAM
  // DERIVES for `cr == 4`. The doubled and undoubled members are asserted side
  // by side on purpose: a loader that widened the whole family would pass an
  // assertion that only checked the four doubled ones.
  const vllm::DeepseekV4LayerHostWeights& L1 = w.host.layers[1];
  CHECK(L1.tid2eid.empty());
  CHECK(L1.gate_bias.size() == static_cast<size_t>(kExperts));
  CHECK(L1.comp_wgate.size() == static_cast<size_t>(2 * kHeadDim * H));
  CHECK(L1.comp_ape.size() == static_cast<size_t>(4 * 2 * kHeadDim));
  CHECK(L1.idx_wk.size() == static_cast<size_t>(2 * opt.index_head_dim * H));
  CHECK(L1.idx_wq.size() ==
        static_cast<size_t>(opt.index_n_heads * opt.index_head_dim * kQLora));
  // NOT widened by `coff`: the compressor norm (`compressor.py:288` is
  // `RMSNorm(self.head_dim, self.rms_norm_eps)`) and `weights_proj`, which is
  // `[n_head, hidden_size]`.
  CHECK(L1.comp_norm_weight.size() == static_cast<size_t>(kHeadDim));
  CHECK(L1.idx_wproj.size() == static_cast<size_t>(opt.index_n_heads * H));

  // The routed experts are the TRELLIS tower; a second host copy of them would
  // be an unreachable duplicate of the thing this row exists to run.
  CHECK(L0.exp_w1.empty());
  CHECK(L0.exp_w2.empty());
  CHECK(L0.exp_w3.empty());
}

TEST_CASE("dsv4 exl3 W1c: the materialized VALUES are the checkpoint's, decoded") {
  // Sizes alone would pass for a tower filled with zeros, or for one whose FP8
  // block scale was ignored. Each family is recomputed from the fixture's own
  // generator and compared elementwise, so a decode that dropped the scale, read
  // the wrong scale block, or mistook BF16 for F16 fails here.
  auto f = BuildFixture();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);
  REQUIRE(w.has_host_weights);
  const int64_t H = kHidden;

  // BF16 -> f32, exactly (bf16 widening is lossless).
  int64_t embed_mismatch = 0;
  for (int64_t i = 0; i < kVocab * H; ++i) {
    const float want =
        vt::BF16ToF32(vt::F32ToBF16(CarriedValue("embed.weight", i, 0.8f, 0.0f)));
    if (w.host.embed[static_cast<size_t>(i)] != want) ++embed_mismatch;
  }
  CHECK(embed_mismatch == 0);

  // F32 straight through.
  int64_t hc_mismatch = 0;
  for (int64_t i = 0; i < kHcMult * kHcMult * H; ++i)
    if (w.host.hc_head_fn[static_cast<size_t>(i)] !=
        CarriedValue("hc_head_fn", i, 0.2f, 0.0f))
      ++hc_mismatch;
  CHECK(hc_mismatch == 0);

  // Block-wise FP8: E4M3 byte times the UE8M0 scale of ITS 128x128 block. The
  // expectation is built from the two decode helpers directly, not from the
  // loader's own path, so the two cannot agree by sharing a bug.
  const std::string base = "layers.0.attn.wq_a";
  const int64_t N = kQLora, K = H;
  const int64_t kb = (K + kBlockK - 1) / kBlockK;
  // THE INSTRUMENT'S OWN PRECONDITION. This case can only see a decode that
  // reads the wrong scale BLOCK if the blocks carry different scales. They did
  // not when the generator had a three-value alphabet, and the mutation that
  // pins the scale to block [0] passed. Assert it rather than assume it.
  REQUIRE(kb > 1);
  CHECK(CarriedScaleByte(base + ".scale", 0) != CarriedScaleByte(base + ".scale", 1));
  int64_t fp8_mismatch = 0;
  int64_t nonzero = 0;
  int64_t narrowing_lost_a_bit = 0;
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t k = 0; k < K; ++k) {
      const uint8_t wb = CarriedFp8Byte(base + ".weight", n * K + k);
      const uint8_t sb =
          CarriedScaleByte(base + ".scale", (n / kBlockN) * kb + (k / kBlockK));
      const float want = vllm::F8E4M3ToF32(wb) * vllm::E8M0ToF32(sb);
      // W1d (#2186): the slot holds BF16 now, so the expectation is narrowed to
      // the same dtype. Read the stored bit pattern back through `vt::BF16ToF32`
      // rather than comparing raw `uint16_t`s, so a change that stored something
      // other than a bf16 pattern in these slots still fails here.
      const float got = vt::BF16ToF32(w.host.layers[0].wq_a[static_cast<size_t>(n * K + k)]);
      if (got != vt::BF16ToF32(vt::F32ToBF16(want))) ++fp8_mismatch;
      // AND THE NARROWING IS LOSSLESS, which is the whole argument for W1d rather
      // than a precision trade. E4M3 carries FOUR significand bits (3 stored + 1
      // implicit) and E8M0 is a pure power of two, so their product needs four
      // significand bits; bf16 has EIGHT (7 stored + 1 implicit) and f32's exponent
      // range verbatim. Every value of this tower is therefore exactly
      // representable at bf16, and this counter must be zero -- not small. If a
      // future checkpoint recipe widens the carried source (a real f32 or a
      // higher-mantissa fp8), this fires and says so instead of quietly rounding.
      if (want != vt::BF16ToF32(vt::F32ToBF16(want))) ++narrowing_lost_a_bit;
      if (want != 0.0f) ++nonzero;
    }
  }
  CHECK(fp8_mismatch == 0);
  CHECK(narrowing_lost_a_bit == 0);
  // A tower of zeros would satisfy an equality check against a zero
  // expectation; it cannot satisfy this.
  CHECK(nonzero == N * K);
  // And the SCALE is load-bearing: the same weight bytes read without their
  // block scale differ from what landed.
  int64_t unscaled_agrees = 0;
  for (int64_t i = 0; i < N * K; ++i)
    if (vt::BF16ToF32(w.host.layers[0].wq_a[static_cast<size_t>(i)]) ==
        vllm::F8E4M3ToF32(CarriedFp8Byte(base + ".weight", i)))
      ++unscaled_agrees;
  CHECK(unscaled_agrees < N * K);
}

TEST_CASE("dsv4 exl3 W1d: the inlined bf16 widening equals vt::BF16ToF32") {
  // `HostBf16ToF32` exists because `vt::BF16ToF32` is out of line in
  // `src/vt/dtype.cpp` and this build has no LTO: calling it per element in the
  // innermost loop of a carried-tower GEMV would cost more than halving the
  // memory traffic saves. A duplicated bit operation is a thing that DRIFTS,
  // though, and a token gate would not see it -- so pin the two together over
  // the WHOLE domain rather than over a sample. bf16 has 65536 patterns; there
  // is no reason to test fewer than all of them.
  //
  // Compared as BITS, not as floats: `NaN != NaN`, so a float comparison would
  // silently pass over every NaN pattern -- the exact region where a widening
  // that mangled the payload would show up.
  int64_t disagree = 0;
  for (int32_t i = 0; i < 65536; ++i) {
    const uint16_t b = static_cast<uint16_t>(i);
    const float mine = vllm::HostBf16ToF32(b);
    const float theirs = vt::BF16ToF32(b);
    if (std::bit_cast<uint32_t>(mine) != std::bit_cast<uint32_t>(theirs)) ++disagree;
  }
  CHECK(disagree == 0);
  // And the widening is REAL rather than a constant: distinct patterns stay
  // distinct, so a helper that returned 0.0f for everything cannot pass above by
  // agreeing with a broken `vt::BF16ToF32`.
  CHECK(vllm::HostBf16ToF32(0x3F80) == 1.0f);   // bf16 1.0
  CHECK(vllm::HostBf16ToF32(0xBF80) == -1.0f);  // bf16 -1.0
  CHECK(vllm::HostBf16ToF32(0x0000) == 0.0f);
}

TEST_CASE("dsv4 exl3 W1d: the carried tower is PRICED at each field's own width") {
  // W1d (#2186) made the carried tower two dtypes: the FP8-sourced half is
  // `HostBf16` at two bytes and the rest stays f32 at four. `DeepseekV4HostResidentBytes`
  // is the number the load refusal prices the artifact with -- on the real
  // DeepSeek-V4-Flash it decides between a 108.59 GiB refusal and a ~97.7 GiB load
  // against 119.63 GiB physical -- so an element size hardcoded inside it would
  // report the pre-W1d total and refuse a tower that now fits.
  //
  // Nothing else can see that. Every OTHER residency case in this file compares
  // the function against itself: it takes `host_bytes` from this same call and
  // brackets the threshold around it, which stays green for any self-consistent
  // formula, right or wrong. So this case rebuilds the total from the LOADED
  // struct, taking each field's width from its OWN `value_type` rather than
  // naming a number. Hardcode `sizeof(float)` back into the accounting and the
  // two totals part company here.
  auto f = BuildFixture();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);
  REQUIRE(w.has_exl3_weights);

  auto B = [](const auto& v) {
    using E = typename std::decay_t<decltype(v)>::value_type;
    return static_cast<int64_t>(v.size()) * static_cast<int64_t>(sizeof(E));
  };
  const vllm::DeepseekV4HostWeights& h = w.host;
  int64_t want = B(h.embed) + B(h.lm_head) + B(h.final_norm_weight) + B(h.hc_head_fn) +
                 B(h.hc_head_base) + static_cast<int64_t>(sizeof(float));
  int64_t bf16_elems = 0;
  for (const vllm::DeepseekV4LayerHostWeights& L : h.layers) {
    want += B(L.attn_norm_weight) + B(L.ffn_norm_weight) + B(L.hc_attn_fn) +
            B(L.hc_attn_base) + B(L.hc_attn_scale) + B(L.hc_ffn_fn) + B(L.hc_ffn_base) +
            B(L.hc_ffn_scale) + B(L.wq_a) + B(L.q_norm_weight) + B(L.wq_b) + B(L.wkv) +
            B(L.kv_norm_weight) + B(L.attn_sink) + B(L.wo_a) + B(L.wo_b) + B(L.idx_wq) +
            B(L.idx_wk) + B(L.idx_wproj) + B(L.comp_wgate) + B(L.comp_ape) +
            B(L.comp_norm_weight) + B(L.gate_weight) + B(L.gate_bias) + B(L.tid2eid) +
            B(L.shared_w1) + B(L.shared_w3) + B(L.shared_w2) + B(L.exp_w1) + B(L.exp_w3) +
            B(L.exp_w2);
    bf16_elems += static_cast<int64_t>(L.wq_a.size() + L.wq_b.size() + L.wkv.size() +
                                       L.wo_a.size() + L.wo_b.size() + L.idx_wq.size() +
                                       L.shared_w1.size() + L.shared_w2.size() +
                                       L.shared_w3.size());
  }
  CHECK(vllm::DeepseekV4HostResidentBytes(w) == want);

  // THE PRECONDITION OF THE CASE ABOVE. If the fixture happened to carry no
  // FP8-sourced elements, the equality would hold for a hardcoded `sizeof(float)`
  // too and this file would gate nothing. Assert that the discriminating
  // population is non-empty, and that it is priced at TWO bytes -- the f32
  // counterfactual is a different number, and it is the one that refuses.
  REQUIRE(bf16_elems > 0);
  // What that population is worth, reported rather than asserted: the saving is a
  // property of the real checkpoint's dimensions, not of this fixture's, so a
  // number pinned here would gate the fixture instead of the change. The claim
  // that 26.64 GiB becomes ~15.7 GiB belongs to the device gate on the real
  // artifact (#2186), which is the only place it can be measured.
  MESSAGE("W1d: " << bf16_elems << " FP8-sourced elements held at 2 B, saving "
                  << 2 * bf16_elems << " B against f32 on this fixture");
}

TEST_CASE("dsv4 exl3 W1c: the carried tower is read from MISALIGNED payloads") {
  // THE ALIGNMENT CONTRACT (#1923 follow-up). A safetensors payload starts at
  // `8 + header_bytes` and each tensor at whatever `data_offsets` names, so the
  // mmap'd address of a tensor satisfies NO alignment above 1. The first W1c
  // materialization formed `const uint16_t*` and `const int64_t*` straight into
  // that mapping and indexed them, which is undefined behaviour. x86 executes
  // the misaligned load and returns the right answer, so every local suite was
  // green; only CI's `sanitize-cpu (address,undefined)` lane reported it, on
  // the BF16 arm of `Exl3CarriedReader::Float` and the I64 arm of its
  // `HashTable`. The readers now go through `vt::LoadUnaligned`, the seam issue
  // #627 established for exactly this.
  //
  // This case is the durable half of that fix, and it pins TWO things.
  //
  // (1) THE FIXTURE'S OWN PRECONDITION. A fixture whose payload happened to land
  //     aligned would exercise nothing, on any lane, and the sanitizer would go
  //     quiet without the bug being gone. `WriteSafetensors` therefore pads its
  //     header until the payload base is ODD, and the REQUIREs below assert that
  //     the three tensors the widening readers actually consume are each
  //     misaligned for their own element type. A change that re-aligns the
  //     fixture reds HERE rather than silently muting the sanitizer lane.
  //
  // (2) THAT THE LOADER READS THEM CORRECTLY ANYWAY. The values are recomputed
  //     from the fixture's generators, so a reader that mis-assembles the bytes
  //     of an unaligned scalar fails on the numbers, not only on a report.
  //
  // What this case CANNOT do on x86 is red on the raw cast by itself: the
  // hardware performs that load. Under the sanitizer build it does, and that is
  // the lane this pin is aimed at. `Float`'s F16 arm and `HashTable`'s I32 arm
  // are not covered: the artifact carries neither (the F16 `suh`/`svh` belong to
  // the trellis tower, read by a different path), and the I32 arm is a bulk
  // `memcpy`, which has no alignment precondition at all.
  //
  // The hash-layer options, so `layers.0.ffn.gate.tid2eid` — the ONLY I64 the
  // carried tower has, and the second site UBSan reported — actually exists.
  // The default fixture has `num_hash_layers = 0` and leaves that arm unread.
  FixtureOptions opt;
  opt.num_hash_layers = 1;
  opt.topk = 2;
  auto f = BuildFixture(opt);

  const vllm::StTensor* embed = nullptr;
  const vllm::StTensor* hc = nullptr;
  const vllm::StTensor* tid = nullptr;
  for (const vllm::SafetensorsFile& shard : f->shards) {
    for (const std::string& name : shard.Names()) {
      if (name == "embed.weight") embed = &shard.Get(name);
      if (name == "hc_head_fn") hc = &shard.Get(name);
      if (name == "layers.0.ffn.gate.tid2eid") tid = &shard.Get(name);
    }
  }
  REQUIRE(embed != nullptr);
  REQUIRE(hc != nullptr);
  REQUIRE(tid != nullptr);
  CHECK(embed->dtype == "BF16");
  CHECK(hc->dtype == "F32");
  CHECK(tid->dtype == "I64");

  auto misaligned_for = [](const vllm::StTensor* t, size_t align) {
    return (reinterpret_cast<uintptr_t>(t->data) % align) != 0;
  };
  // BF16 is what UBSan reported at :431, I64 what it reported at :458. F32 is
  // read by a bulk memcpy today; it is asserted so that converting that arm to a
  // typed load can never be done onto an accidentally-aligned fixture.
  REQUIRE(misaligned_for(embed, alignof(uint16_t)));
  REQUIRE(misaligned_for(hc, alignof(float)));
  REQUIRE(misaligned_for(tid, alignof(int64_t)));

  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);
  REQUIRE(w.has_host_weights);

  // And the misaligned bytes decoded to the right numbers.
  int64_t bf16_mismatch = 0;
  for (int64_t i = 0; i < kVocab * kHidden; ++i)
    if (w.host.embed[static_cast<size_t>(i)] !=
        vt::BF16ToF32(vt::F32ToBF16(CarriedValue("embed.weight", i, 0.8f, 0.0f))))
      ++bf16_mismatch;
  CHECK(bf16_mismatch == 0);

  int64_t f32_mismatch = 0;
  for (int64_t i = 0; i < kHcMult * kHcMult * kHidden; ++i)
    if (w.host.hc_head_fn[static_cast<size_t>(i)] !=
        CarriedValue("hc_head_fn", i, 0.2f, 0.0f))
      ++f32_mismatch;
  CHECK(f32_mismatch == 0);

  // The I64 hash table, narrowed to int32 exactly as the GGUF arm narrows it.
  const int64_t topk = static_cast<int64_t>(w.host.layers[0].tid2eid.size()) / kVocab;
  REQUIRE(topk >= 1);
  REQUIRE(w.host.layers[0].tid2eid.size() ==
          static_cast<size_t>(kVocab * topk));
  int64_t i64_mismatch = 0;
  for (int64_t i = 0; i < kVocab * topk; ++i)
    if (w.host.layers[0].tid2eid[static_cast<size_t>(i)] !=
        static_cast<int32_t>(NameHash("layers.0.ffn.gate.tid2eid", i) % kExperts))
      ++i64_mismatch;
  CHECK(i64_mismatch == 0);
}

TEST_CASE("dsv4 exl3 W1c: a carried tensor this arm cannot route REFUSES BY NAME") {
  // RETIRED by #1970: this subcase asserted that the loader REFUSES the real
  // artifact's `coff = 2` compressor, which is exactly the behaviour option C
  // removes. The refusal did not disappear — it moved to the forward, where the
  // mis-index it prevents actually lives, and it is re-gated end to end by
  // `test_deepseek_v4_exl3_forward.cpp` ("the REAL DSA geometry LOADS and the
  // FORWARD refuses by name"), which drives the production loader AND the
  // production forward rather than the loader alone.
  //
  // Its fixture was also wrong about the artifact: it set `compress_ratios =
  // {128}` while doubling the width, and upstream's `coff` is
  // `1 + (compress_ratio == 4)` (`vllm/models/deepseek_v4/compressor.py:247-248`),
  // so a `cr == 128` layer is NOT doubled and that shape is one the checkpoint
  // does not store. The fixture flag it used is now `real_dsa_geometry` and
  // applies the per-layer rule instead.
  SUBCASE("a DSA width that is NEITHER geometry") {
    // #1970 moved the compressor family's width from a CONSTANT to upstream's
    // own derivation, `coff = 1 + (compress_ratio == 4)`
    // (`compressor.py:247-248`). It did not stop checking, and a fixture flag
    // that only toggled between the derived width and the collapsed one could
    // not tell a derivation from a two-value allow-list. This writes a third
    // width no oracle emits.
    FixtureOptions opt;
    opt.layers = 1;
    opt.compress_ratios = {128};
    opt.bogus_dsa_width = true;
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "compressor.wgate.weight"));
    CHECK(Mentions(msg, "MODEL-DSV4-EXL3"));
    CHECK(Mentions(msg, "W1c"));
    // The DERIVED width is named, so the message says what WOULD be routable
    // rather than only that this one is not. At cr == 128 `coff` is 1.
    CHECK(Mentions(msg, "512"));
    CHECK(Mentions(msg, "1536"));  // the refused 3 * head_dim
  }
  SUBCASE("a COLLAPSED `cr == 4` family, which upstream cannot emit") {
    // The width is DERIVED from `coff = 1 + (compress_ratio == 4)`
    // (`vllm/models/deepseek_v4/compressor.py:247-248` at the parity pin
    // `5559679229bc961848b121ccdeaa8fa5d79bec98`), so at `cr == 4` upstream
    // emits ONE width and it is the doubled one. A `cr == 4` checkpoint whose
    // compressor family is UNDOUBLED is a checkpoint upstream cannot load at
    // all, and this loader must not accept it either.
    //
    // This is the case a two-width loader accepted. It is separated from the
    // "neither geometry" subcase on purpose: a third width proves the loader
    // still checks SOMETHING, and only this one proves it checks the width
    // upstream actually derives rather than a set chosen to suit a fixture.
    FixtureOptions opt;
    opt.layers = 1;
    opt.compress_ratios = {4};
    opt.index_n_heads = 2;
    opt.index_head_dim = 4;
    opt.index_topk = 3;
    opt.real_dsa_geometry = false;  // the collapsed, undoubled family
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "compressor.wgate.weight"));
    CHECK(Mentions(msg, "coff"));
    CHECK(Mentions(msg, "compressor.py:247-248"));
    CHECK(Mentions(msg, "MODEL-DSV4-EXL3"));
    CHECK(Mentions(msg, "W1c"));
    CHECK(Mentions(msg, "1024"));  // the derived `coff * head_dim`
    CHECK(Mentions(msg, "512"));   // the collapsed width the checkpoint carries
  }
  SUBCASE("`indexer.wq_b` at the WRONG INPUT SPACE") {
    // `wq_b` is the one DSA tensor whose width is NOT a `coff` width. Upstream
    // builds it as `ReplicatedLinear(q_lora_rank, head_dim * n_head)`
    // (`attention.py:721-726`) and calls it on `qr`, the q-LoRA latent
    // (`:835`), so its K is `q_lora_rank` and nothing about it is doubled.
    //
    // It gets its own case because the rest of the family refuses FIRST: a
    // wholly collapsed checkpoint reds on `compressor.wgate.weight` and never
    // reaches this check, which would leave the loader free to accept a `wq_b`
    // at `hidden_size` unnoticed. So this writes the real geometry everywhere
    // ELSE and collapses only this one tensor.
    FixtureOptions opt;
    opt.layers = 1;
    opt.compress_ratios = {4};
    opt.index_n_heads = 2;
    opt.index_head_dim = 4;
    opt.index_topk = 3;
    opt.real_dsa_geometry = true;
    opt.collapsed_indexer_wq_b = true;
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "indexer.wq_b"));
    CHECK(Mentions(msg, "q_lora_rank"));
    CHECK(Mentions(msg, "attention.py:721-726"));
    // The message must NOT blame `coff`, which has nothing to do with this K.
    CHECK(!Mentions(msg, "compressor.py:247-248"));
    CHECK(Mentions(msg, std::to_string(kQLora)));  // 128, the required K
    CHECK(Mentions(msg, std::to_string(kHidden)));  // 256, the K carried
  }
  SUBCASE("the INDEXER's own compressor at the COLLAPSED width") {
    // The third derivation. `coff` widens the indexer's own
    // `DeepseekCompressor` family too, because upstream builds it at
    // `head_dim = index_head_dim` with the SAME ratio
    // (`vllm/models/deepseek_v4/attention.py:768-776`, the `DeepseekV4Indexer`
    // one — `attention.py:335` is the attention's own) and `coff` is a property
    // of the ratio, not of the head width. So `indexer.compressor.wkv.weight`
    // is derived at `coff * index_head_dim` by a rule that is the SAME as the
    // main compressor's but reads a different head width.
    //
    // It gets its own case for the reason `collapsed_indexer_wq_b` does, and
    // the round-2 fresh review found the hole by mutation: with the whole
    // family collapsed the MAIN compressor refuses first, so this derivation's
    // message was never read and DELETING the check left both suites green.
    // Here everything else is at the real geometry and only this tensor is
    // collapsed, so the loader must refuse BY NAME on this one.
    FixtureOptions opt;
    opt.layers = 1;
    opt.compress_ratios = {4};
    opt.index_n_heads = 2;
    opt.index_head_dim = 4;
    opt.index_topk = 3;
    opt.real_dsa_geometry = true;
    opt.collapsed_indexer_wkv = true;
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "indexer.compressor.wkv.weight"));
    // Its OWN derivation, named: `coff` over `index_head_dim`, and WHY `coff`
    // is 2 here at all. A message that named only `coff * head_dim` would be
    // describing the main compressor's rule.
    CHECK(Mentions(msg, "coff * index_head_dim"));
    CHECK(Mentions(msg, "compress_ratio == 4"));
    CHECK(Mentions(msg, "attention.py:274"));
    CHECK(Mentions(msg, "compressor.py:247-248"));
    CHECK(Mentions(msg, "MODEL-DSV4-EXL3"));
    CHECK(Mentions(msg, "W1c"));
    // Both widths, bound to this tensor: 8 = coff * index_head_dim is what
    // upstream derives, 4 = index_head_dim is what this checkpoint carries.
    CHECK(Mentions(msg, "dimension 0 must be 8"));
    CHECK(Mentions(msg, "got 4 in [4,256]"));
    // The main compressor is at the REAL width here, so it must NOT be what
    // refused — otherwise this case would pass without ever reaching the
    // derivation it exists to gate.
    CHECK(!Mentions(msg, "compressor.wgate.weight"));
    CHECK(!Mentions(msg, "indexer.wq_b"));
  }
  SUBCASE("no recipe for the carried FP8 half") {
    // The carried MLA linears are block-wise FP8 and the block size comes from
    // the checkpoint, not from a constant in this loader. Without it there is
    // nothing to assume.
    FixtureOptions opt;
    opt.omit_base_quant_config = true;
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "base_quantization_config"));
    CHECK(Mentions(msg, "MODEL-DSV4-EXL3"));
    CHECK(Mentions(msg, "W1c"));
  }
}

TEST_CASE("dsv4 exl3: unrepresentable inputs REFUSE BY NAME") {
  SUBCASE("an unknown rank-sliced schema version") {
    FixtureOptions opt;
    opt.version = "rank-sliced-deepseek-v4-v2";
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "rank-sliced-deepseek-v4-v2"));
    CHECK(Mentions(msg, "MODEL-DSV4-EXL3"));
  }
  // `mul1` is NOT an unrepresentable input: 974c3b7f8 (QUANT-EXL3 W1-W2)
  // made the loader decode it, so a mul1 fixture loads and this subcase's
  // refusal would never fire. The unrepresentable example is a name outside
  // the accepted set {mcg, mul1}.
  SUBCASE("a codebook other than mcg or mul1") {
    FixtureOptions opt;
    opt.codebook = "not_a_codebook";
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "not_a_codebook"));
    CHECK(Mentions(msg, "MODEL-DSV4-EXL3"));
  }
  SUBCASE("a missing rank tensor") {
    FixtureOptions opt;
    opt.drop_tensor = "layers.0.ffn.experts.1.w2.rank2.suh";
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "layers.0.ffn.experts.1.w2.rank2.suh"));
    CHECK(Mentions(msg, "MODEL-DSV4-EXL3"));
  }
  SUBCASE("a whole missing rank") {
    FixtureOptions opt;
    opt.ranks_written = 3;
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "rank3"));
  }
  SUBCASE("a carried tensor no arm routes") {
    FixtureOptions opt;
    opt.extra_carried = "layers.0.attn.wq_c.weight";
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "layers.0.attn.wq_c.weight"));
    CHECK(Mentions(msg, "MODEL-DSV4-EXL3"));
  }
  SUBCASE("w2 sliced on the wrong axis") {
    FixtureOptions opt;
    opt.swap_w2_slice_axis = true;
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "w2"));
    CHECK(Mentions(msg, "MODEL-DSV4-EXL3"));
    // Pin the REASON, not just the fact of a throw: a refusal test that passes
    // because some earlier check happened to fire gates nothing. This must be
    // the replicated-side comparison noticing that w2's svh differs per rank.
    CHECK(Mentions(msg, "IN-split"));
    CHECK(Mentions(msg, "svh"));
  }
}

TEST_CASE("dsv4 exl3 R1: the MTP tail is CLASSIFIED by the production load (#1314)") {
  // REACHABILITY, not arithmetic. `test_deepseek_v4_mtp_inventory` states the
  // real artifact's shapes and exercises `ClassifyDeepseekV4MtpTail` directly;
  // this case proves the production loader actually CALLS it, by going through
  // `LoadDeepseekV4ForCausalLMWeights` and reading the result off the weights.
  //
  // The distinction is the point. Before #1314 the loader met 3985 draft-head
  // tensors on every load of the real checkpoint and dropped them while counting
  // only how many there were, and `deepseek-v4-mtp.md` §4 recorded the MTP gate
  // as weight-BLOCKED for a week as a result.
  FixtureOptions opt;
  opt.layers = 1;
  opt.mtp_heads = 3;  // the measured artifact carries three, not the config's one
  auto f = BuildFixture(opt);
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);

  // Still SKIPPED, exactly as vLLM skips it (nvidia/model.py:1474) -- classifying
  // is not loading, and this arm must not start consuming the tail by accident.
  // 6 tensors x 3 heads. The base fixture's two plain `mtp.0.*` stand down when a
  // real tail is requested, because they use the same names.
  CHECK(w.exl3.skipped_mtp_tensors == 18);

  CHECK(w.exl3.mtp.refusal.empty());
  CHECK(w.exl3.mtp.num_heads == 3);
  CHECK(w.exl3.mtp.plain == 6);      // (attn_norm + attn_sink) x 3
  CHECK(w.exl3.mtp.fp8_block == 3);  // wkv, per head
  CHECK(w.exl3.mtp.mxfp4 == 3);      // one routed expert, per head
}

TEST_CASE("dsv4 exl3 R1: a tail of PLAIN tensors classifies without refusing") {
  // The base fixture's tail is two unquantized `mtp.0.*` tensors. A tail carrying
  // no quantized weight at all must still classify cleanly: a refusal here would
  // break every load that works today, which is why the classifier REPORTS rather
  // than throws.
  FixtureOptions opt;
  opt.layers = 1;
  auto f = BuildFixture(opt);
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);
  CHECK(w.exl3.skipped_mtp_tensors == 2);
  CHECK(w.exl3.mtp.num_heads == 1);
  CHECK(w.exl3.mtp.plain == 2);
  CHECK(w.exl3.mtp.fp8_block == 0);
  CHECK(w.exl3.mtp.mxfp4 == 0);
  CHECK(w.exl3.mtp.refusal.empty());
}

TEST_CASE("W1: the compressor arm is REACHED from a production forward (#2286)") {
  // The claim this earns: `CompressorLayerStep` is not a function that merely
  // works. `test_deepseek_v4_gguf_load` records why the GGUF paged arm can never
  // reach it -- `dsa_dense` makes every layer dense there -- so the proof has to
  // come through the non-GGUF paged entry, which is what this drives.
  FixtureOptions opt;
  opt.layers = 2;
  opt.compress_ratios = {0, 128};  // layer 1 carries the compressor
  opt.real_dsa_geometry = false;   // the collapsed width the host forward indexes
  auto f = BuildFixture(opt);
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);
  REQUIRE(w.has_exl3_weights);

  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const int64_t nlayers = w.params.num_hidden_layers;
  const int64_t hd = w.params.head_dim;
  const int64_t bs = 8, nb = 4;
  std::vector<std::vector<float>> storage(static_cast<size_t>(nlayers));
  std::vector<vt::Tensor> pages(static_cast<size_t>(nlayers));
  for (int64_t l = 0; l < nlayers; ++l) {
    storage[static_cast<size_t>(l)].assign(static_cast<size_t>(nb * bs * hd), 0.0f);
    pages[static_cast<size_t>(l)] = vt::Tensor::Contiguous(
        storage[static_cast<size_t>(l)].data(), vt::DType::kF32, q.device,
        {nb, bs, hd});
  }

  vllm::DeepseekV4CompressorState cs;
  cs.Resize(nlayers);

  int64_t kv_base = 0;
  for (int step = 0; step < 2; ++step) {
    const std::vector<int32_t> tok{static_cast<int32_t>(1 + step)};
    const std::vector<int32_t> pos{static_cast<int32_t>(kv_base)};
    const auto lg = vllm::DeepseekV4ForwardExl3Paged(w, q, pages, kv_base, tok, pos,
                                                     {0}, &cs);
    REQUIRE(lg.size() == static_cast<size_t>(w.params.vocab_size));
    for (const float v : lg) REQUIRE(std::isfinite(v));
    kv_base += 1;
  }

  // THE REACHABILITY CLAIM: the compressor layer's state grew, one row per step,
  // which only the composed arm produces. A dense fallback leaves it empty.
  CHECK(cs.state_kv[1].size() == static_cast<size_t>(2 * hd));
  CHECK(cs.state_score[1].size() == static_cast<size_t>(2 * hd));
  // Layer 0 has no compressor, so its state must stay untouched.
  CHECK(cs.state_kv[0].empty());
  // Two tokens close no window at ratio 128.
  CHECK(cs.comp_rows[1].empty());
}

TEST_CASE("W1: without compressor state the same layer REFUSES (#2286)") {
  // The guard is what stops a compressor layer attending the raw prefix. Adding
  // the arm must not make its absence silent.
  FixtureOptions opt;
  opt.layers = 2;
  opt.compress_ratios = {0, 128};
  opt.real_dsa_geometry = false;
  auto f = BuildFixture(opt);
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const int64_t nlayers = w.params.num_hidden_layers, hd = w.params.head_dim;
  const int64_t bs = 8, nb = 4;
  std::vector<std::vector<float>> storage(static_cast<size_t>(nlayers));
  std::vector<vt::Tensor> pages(static_cast<size_t>(nlayers));
  for (int64_t l = 0; l < nlayers; ++l) {
    storage[static_cast<size_t>(l)].assign(static_cast<size_t>(nb * bs * hd), 0.0f);
    pages[static_cast<size_t>(l)] = vt::Tensor::Contiguous(
        storage[static_cast<size_t>(l)].data(), vt::DType::kF32, q.device,
        {nb, bs, hd});
  }
  const std::vector<int32_t> tok{1}, pos{0};
  CHECK_THROWS(vllm::DeepseekV4ForwardExl3Paged(w, q, pages, 0, tok, pos, {0},
                                                /*compressor=*/nullptr));
}

TEST_CASE("W3: the compressor pools its OWN projection, not the MLA latent (#2286)") {
  // Upstream's compressor owns a `fused_wkv_wgate` emitting both its KV and its
  // gate from the hidden state (`compressor.py:279-287`); it never reuses the
  // MLA's `kraw`. This tree accounted `compressor.wkv.weight` and dropped it
  // until W3, so the composition pooled the MLA latent -- finite, plausible, and
  // not what upstream pools.
  //
  // The claim here is that the loaded projection is LOAD-BEARING: perturbing
  // `comp_wkv` alone must move the logits. If it did not, the forward would still
  // be pooling the latent and the materialization would be decoration.
  FixtureOptions opt;
  opt.layers = 2;
  opt.compress_ratios = {0, 128};
  opt.real_dsa_geometry = false;
  auto f = BuildFixture(opt);
  vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);
  REQUIRE(w.has_exl3_weights);
  // It is actually loaded, at the gate's own width.
  REQUIRE(!w.host.layers[1].comp_wkv.empty());
  CHECK(static_cast<int64_t>(w.host.layers[1].comp_wkv.size()) ==
        static_cast<int64_t>(w.host.layers[1].comp_wgate.size()));

  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const int64_t nlayers = w.params.num_hidden_layers, hd = w.params.head_dim;
  const int64_t bs = 16, nb = 16;  // >= 128 slots, so the closing window fits
  const auto run = [&](const vllm::DeepseekV4Weights& ww) {
    std::vector<std::vector<float>> storage(static_cast<size_t>(nlayers));
    std::vector<vt::Tensor> pages(static_cast<size_t>(nlayers));
    for (int64_t l = 0; l < nlayers; ++l) {
      storage[static_cast<size_t>(l)].assign(static_cast<size_t>(nb * bs * hd), 0.0f);
      pages[static_cast<size_t>(l)] = vt::Tensor::Contiguous(
          storage[static_cast<size_t>(l)].data(), vt::DType::kF32, q.device,
          {nb, bs, hd});
    }
    vllm::DeepseekV4CompressorState cs;
    cs.Resize(nlayers);
    // SEED the state so this step CLOSES a window. At ratio 128 a single token at
    // position 0 closes none, the pooled row never reaches the logits, and the
    // projection under test would read as inert -- which is exactly what the
    // first version of this case measured. 127 prior rows plus this token's makes
    // `(127 + 1) % 128 == 0`, and `kv_base = 127` keeps the prefix guard happy.
    const int64_t cr = 128;
    cs.state_kv[1].assign(static_cast<size_t>((cr - 1) * hd), 0.05f);
    cs.state_score[1].assign(static_cast<size_t>((cr - 1) * hd), 0.02f);
    const std::vector<int32_t> tok{1}, pos{static_cast<int32_t>(cr - 1)};
    return vllm::DeepseekV4ForwardExl3Paged(ww, q, pages, cr - 1, tok, pos, {0}, &cs);
  };

  const std::vector<float> before = run(w);
  // Perturb ONLY the compressor's KV projection.
  vllm::DeepseekV4Weights w2 = w;
  for (float& v : w2.host.layers[1].comp_wkv) v += 0.25f;
  const std::vector<float> after = run(w2);

  REQUIRE(before.size() == after.size());
  double diff = 0.0, mag = 0.0;
  for (size_t i = 0; i < before.size(); ++i) {
    REQUIRE(std::isfinite(after[i]));
    mag = std::max(mag, std::abs(static_cast<double>(before[i])));
    diff = std::max(diff, std::abs(static_cast<double>(before[i] - after[i])));
  }
  REQUIRE(mag > 1e-6);
  CHECK(diff > 1e-6);  // the projection is READ, not decoration
}

TEST_CASE("W3: the INDEXER's compressor family is materialized (#2286)") {
  // The indexer carries its own `DeepseekCompressor` at `head_dim =
  // index_head_dim` (`attention.py:768-776`). Three of its four tensors were
  // accounted and dropped; only `idx_wk` had a host slot. They load now.
  //
  // Loading them does NOT make the indexer's compressor run, and this case does
  // not pretend otherwise: the forward still refuses on `idx_wk`'s width. That
  // order is deliberate -- narrowing a refusal before the capability exists turns
  // it into a silent wrong answer, which this row has established twice.
  FixtureOptions opt;
  opt.layers = 2;
  opt.compress_ratios = {0, 4};      // the indexer exists ONLY at cr == 4
  opt.real_dsa_geometry = true;      // the artifact's widths, coff == 2
  opt.index_n_heads = 2;
  opt.index_head_dim = 8;
  opt.index_topk = 2;
  auto f = BuildFixture(opt);
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);
  REQUIRE(w.has_exl3_weights);

  const auto& L = w.host.layers[1];
  const int64_t ihd = w.params.index_head_dim, H = w.params.hidden_size;
  const int64_t cr = w.params.compress_ratio(1);
  const int64_t iw = 2 * ihd;  // coff == 2, always, where the indexer exists

  // Present, and at upstream's widths: the ape and the fused gate carry `coff`,
  // the norm does NOT (`compressor.py:288`).
  CHECK(static_cast<int64_t>(L.idx_comp_ape.size()) == cr * iw);
  CHECK(static_cast<int64_t>(L.idx_comp_wgate.size()) == iw * H);
  CHECK(static_cast<int64_t>(L.idx_comp_norm_weight.size()) == ihd);
  // Its KV projection was already loaded and keeps the same width as the gate.
  CHECK(static_cast<int64_t>(L.idx_wk.size()) == iw * H);

  // NON-TRIVIAL: an all-zero slot would satisfy every size check above while
  // carrying nothing the checkpoint wrote.
  double mag = 0.0;
  for (const float v : L.idx_comp_wgate) mag = std::max(mag, std::abs((double)v));
  CHECK(mag > 1e-6);

  // The refusal is UNCHANGED, because the cycle does not run yet.
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const std::string msg = ThrowMessage([&] {
    (void)vllm::DeepseekV4Model::Forward({1}, {0}, {}, {}, w, q, {0});
  });
  CAPTURE(msg);
  CHECK(dsv4_exl3_fixture::Mentions(msg, "indexer.compressor.wkv.weight"));
}

TEST_CASE("W3: the cr == 4 arm is REACHED, and its INDEXER state accumulates (#2286)") {
  // The `cr == 128` arm was already reached. This is the `cr == 4` family, which
  // differs by running a SECOND compressor -- the indexer's -- and using its
  // selection to narrow which closed rows the attention merges.
  //
  // The claim is reachability of that second machine specifically: its state must
  // grow, separately from the attention compressor's, or the selection is being
  // built from nothing and every step would silently attend the window alone.
  FixtureOptions opt;
  opt.layers = 2;
  opt.compress_ratios = {0, 4};  // layer 1 carries BOTH a compressor and an indexer
  // The REAL widths: the loader derives `coff` for a cr == 4 layer and refuses a
  // collapsed one, since upstream cannot emit it at this ratio.
  opt.real_dsa_geometry = true;
  opt.index_n_heads = 2;
  // >= the model's `qk_rope_head_dim` (64 in this fixture). The indexer's pooled
  // key is rotated on its rope TAIL, and that tail is the MODEL's rope width
  // whatever the indexer's head is (`compressor.py:240`), so a head narrower than
  // the rope width is a geometry upstream cannot produce.
  opt.index_head_dim = 128;
  opt.index_topk = 2;
  auto f = BuildFixture(opt);
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);
  REQUIRE(w.has_exl3_weights);

  // Report the widths before running, so a mismatch names the tensor instead of
  // surfacing as an anonymous MatVec error from inside the forward.
  {
    const auto& L1 = w.host.layers[1];
    const int64_t H1 = w.params.hidden_size, hd1 = w.params.head_dim;
    const int64_t ihd1 = w.params.index_head_dim, inh1 = w.params.index_n_heads;
    CAPTURE(L1.comp_wgate.size()); CAPTURE(L1.comp_wkv.size());
    CAPTURE(L1.comp_ape.size());   CAPTURE(L1.idx_wk.size());
    CAPTURE(L1.idx_comp_wgate.size()); CAPTURE(L1.idx_comp_ape.size());
    CAPTURE(L1.idx_wq.size());     CAPTURE(L1.idx_wproj.size());
    CAPTURE(H1); CAPTURE(hd1); CAPTURE(ihd1); CAPTURE(inh1);
    CAPTURE(w.params.q_lora_rank);
    CHECK(static_cast<int64_t>(L1.idx_wproj.size()) == inh1 * H1);
    CHECK(static_cast<int64_t>(L1.idx_wk.size()) == 2 * ihd1 * H1);
    CHECK(static_cast<int64_t>(L1.comp_wkv.size()) ==
          static_cast<int64_t>(L1.comp_wgate.size()));
    // The one the forward DISPATCHES on: if this is neither shape it falls to the
    // hidden-state branch and MatVec throws anonymously.
    CHECK(static_cast<int64_t>(L1.idx_wq.size()) ==
          inh1 * ihd1 * w.params.q_lora_rank);
  }
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const int64_t nlayers = w.params.num_hidden_layers, hd = w.params.head_dim;
  const int64_t bs = 16, nb = 8;
  std::vector<std::vector<float>> storage(static_cast<size_t>(nlayers));
  std::vector<vt::Tensor> pages(static_cast<size_t>(nlayers));
  for (int64_t l = 0; l < nlayers; ++l) {
    storage[static_cast<size_t>(l)].assign(static_cast<size_t>(nb * bs * hd), 0.0f);
    pages[static_cast<size_t>(l)] = vt::Tensor::Contiguous(
        storage[static_cast<size_t>(l)].data(), vt::DType::kF32, q.device,
        {nb, bs, hd});
  }

  vllm::DeepseekV4CompressorState cs;
  cs.Resize(nlayers);

  // Enough steps to cross a cr == 4 boundary, so rows actually close and the
  // selection has something to choose from.
  int64_t kv_base = 0;
  for (int step = 0; step < 8; ++step) {
    const std::vector<int32_t> tok{static_cast<int32_t>(1 + (step % 3))};
    const std::vector<int32_t> pos{static_cast<int32_t>(kv_base)};
    const auto lg = vllm::DeepseekV4ForwardExl3Paged(w, q, pages, kv_base, tok, pos,
                                                     {0}, &cs);
    REQUIRE(lg.size() == static_cast<size_t>(w.params.vocab_size));
    for (const float v : lg) REQUIRE(std::isfinite(v));
    kv_base += 1;
  }

  // BOTH state machines ran on layer 1, and they are separate.
  CHECK(!cs.state_kv[1].empty());       // the attention compressor's
  CHECK(!cs.idx_state_kv[1].empty());   // the INDEXER's -- the reachability claim
  CHECK(!cs.idx_state_score[1].empty());
  // Eight tokens at ratio 4 close two windows, so the indexer produced two keys.
  const int64_t ihd = w.params.index_head_dim;
  CHECK(static_cast<int64_t>(cs.idx_comp_rows[1].size()) == 2 * ihd);
  // Layer 0 is dense: neither machine touched it.
  CHECK(cs.state_kv[0].empty());
  CHECK(cs.idx_state_kv[0].empty());
}

// ── MODEL-DSV4-PAGED-ENTRY (#2447) ─────────────────────────────────────────
//
// THE REACHABILITY GATE. `DeepseekV4ForwardExl3Paged` worked before this row and
// had only test callers, so every case above it proves a class rather than a
// capability. This one enters through `ModelRegistry::Forward` -- the production
// entry point AGENTS.md §"Nothing lands dead" names -- and it leaves
// `gather_logits` AT ITS DEFAULT, because that default is the defect: it is true
// on every step the runner builds, so it selected `ForwardDevice`, which binds
// neither `paged_kv` nor `compressor`.
//
// WHAT IT DOES NOT DO. `DeepseekV4LoadedModel` lives in an anonymous namespace,
// so no downcast can read its compressor state, and it must not gain a public
// accessor for the purpose -- that would weaken the claim to "the member
// exists". The two reachable observables carry it instead:
//
//   (A) the page bytes for the compressor layer are non-zero after step 1. Only
//       `vt::ConcatAndCacheMla` writes them and only the paged arm calls it, so
//       this is red if the new registry branch is deleted.
//   (B) step 2 SUCCEEDS at `kv_base == 1`. `CompressorLayerStep` refuses unless
//       the carried state has seen exactly `kv_base` tokens, so this is red if
//       the state is constructed per call rather than persisted on the model.
//
// Two production defects, one assertion each.
namespace {
// One count, not one assertion per element.
int64_t NonFinite(const std::vector<float>& v) {
  int64_t n = 0;
  for (const float x : v)
    if (!std::isfinite(x)) ++n;
  return n;
}
}  // namespace

TEST_CASE("PAGED-ENTRY: ModelRegistry::Forward REACHES the EXL3 paged arm (#2447)") {
  dsv4_exl3_fixture::FixtureOptions opt;
  opt.layers = 2;
  opt.compress_ratios = {0, 128};  // layer 1 carries the compressor
  opt.real_dsa_geometry = false;   // the collapsed width the host forward indexes
  auto f = dsv4_exl3_fixture::BuildFixture(opt);

  const vllm::ModelSource source = vllm::ModelSource::FromSafetensors(f->shards);
  std::unique_ptr<vllm::LoadedModel> model =
      vllm::ModelRegistry::Load(f->config, source);
  REQUIRE(model != nullptr);

  const vllm::DeepseekV4Params params = vllm::ParseDeepseekV4Params(f->config);
  const int64_t nlayers = params.num_hidden_layers;
  const int64_t hd = params.head_dim;
  REQUIRE(nlayers == 2);
  REQUIRE(params.has_compressor(1));

  // The pages the runner publishes, under the names `MakeDeepseekV4KVCache`
  // publishes them under -- resolved BY NAME, exactly as the adapter does it.
  // f32 rather than the shipped `kI8`: `vt::ConcatAndCacheMla` takes a float
  // cache only, so the packed arm is refused by name and is owed to
  // `KV-DSV4-MULTICACHE` W8.
  const int64_t nb = 4, bs = 8;
  std::vector<std::vector<float>> storage(static_cast<size_t>(nlayers));
  std::vector<vllm::PagedKvCache> attn_kv(static_cast<size_t>(nlayers));
  std::vector<std::string> names;
  for (int64_t l = 0; l < nlayers; ++l) {
    const size_t i = static_cast<size_t>(l);
    storage[i].assign(static_cast<size_t>(nb * bs * hd), 0.0f);
    attn_kv[i].data = storage[i].data();
    attn_kv[i].dtype = vt::DType::kF32;
    attn_kv[i].num_blocks = nb;
    attn_kv[i].block_size = bs;
    attn_kv[i].num_kv_heads = 1;
    attn_kv[i].head_size = hd;
    names.push_back("model.layers." + std::to_string(l) + ".attn.swa_cache");
  }
  vllm::MultiKvCacheIndex mk;
  mk.layer_names = &names;

  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  std::vector<vllm::GdnStateCache> gdn_state;
  const vllm::v1::GDNAttentionMetadata gdn_meta{};

  const auto step = [&](int32_t token, int32_t position, int computed, int nreqs) {
    const std::vector<int32_t> tok{token};
    const std::vector<int32_t> pos{position};
    const std::vector<int32_t> li{0};
    vllm::v1::CommonAttentionMetadata attn_meta{};
    attn_meta.num_reqs = nreqs;
    attn_meta.num_computed_tokens_cpu = {computed};
    vllm::ModelForwardInput in{.token_ids = tok,
                               .positions = pos,
                               .attn_meta = attn_meta,
                               .gdn_meta = gdn_meta,
                               .attn_kv = attn_kv,
                               .gdn_state = gdn_state,
                               .config = f->config,
                               .queue = queue,
                               .logits_indices = li,
                               .num_reqs = nreqs};
    // NOT in the aggregate initializer, exactly as the runner sets it.
    in.multi_kv = &mk;
    // `gather_logits` is DELIBERATELY LEFT AT ITS DEFAULT (true). A test that
    // set it false would route through the branch no runner step takes and
    // would prove nothing about the path that runs.
    REQUIRE(in.gather_logits);
    return vllm::ModelRegistry::Forward(*model, in);
  };

  // ── step 1: kv_base == 0 ────────────────────────────────────────────────
  const vllm::ForwardLogits s1 = step(/*token=*/1, /*position=*/0, /*computed=*/0,
                                      /*nreqs=*/1);
  CHECK(s1.rows == 1);
  CHECK(s1.vocab == params.vocab_size);
  REQUIRE(s1.host.size() == static_cast<size_t>(params.vocab_size));
  // AGGREGATED, never one assertion per vocabulary entry: a per-entry loop
  // buries the assertion COUNT, and a changed count is itself the signal a
  // mutation is read by.
  CHECK(NonFinite(s1.host) == 0);

  // (A) THE PAGE WAS WRITTEN, on the compressor layer. Aggregated rather than one
  // assertion per element: a changed assertion COUNT is itself the signal a
  // mutation is read by, and `nb * bs * hd` REQUIREs would swamp it.
  int64_t nonzero_l1 = 0;
  double sum_l1 = 0.0;
  for (const float v : storage[1]) {
    if (v != 0.0f) ++nonzero_l1;
    sum_l1 += std::abs(static_cast<double>(v));
  }
  CHECK(NonFinite(storage[1]) == 0);
  MESSAGE("compressor-layer page: " << nonzero_l1 << " non-zero of "
                                    << storage[1].size() << ", L1 " << sum_l1);
  CHECK(nonzero_l1 > 0);
  // Exactly the FIRST slot's row, and nothing beyond it: `kv_base == 0` writes
  // slot 0. A route that wrote the whole page would also pass "non-zero".
  int64_t nonzero_beyond = 0;
  for (size_t i = static_cast<size_t>(hd); i < storage[1].size(); ++i)
    if (storage[1][i] != 0.0f) ++nonzero_beyond;
  CHECK(nonzero_beyond == 0);

  // ── step 2: kv_base == 1 ────────────────────────────────────────────────
  // (B) THE STATE PERSISTED. `CompressorLayerStep` refuses unless the carried
  // state has seen exactly `kv_base` tokens, so a state built per call has seen
  // 0 here and this throws.
  const vllm::ForwardLogits s2 = step(/*token=*/2, /*position=*/1, /*computed=*/1,
                                      /*nreqs=*/1);
  CHECK(s2.rows == 1);
  REQUIRE(s2.host.size() == static_cast<size_t>(params.vocab_size));
  CHECK(NonFinite(s2.host) == 0);
  // And the second token's row landed in slot 1, so the two steps did not write
  // the same slot.
  int64_t nonzero_slot1 = 0;
  for (int64_t d = 0; d < hd; ++d)
    if (storage[1][static_cast<size_t>(hd + d)] != 0.0f) ++nonzero_slot1;
  CHECK(nonzero_slot1 > 0);

  // ── the route's own refusal, which is the RESOLVER'S and not a copy ─────
  // One `kv_base` cannot serve two context lengths, and the compressor state has
  // no request dimension -- two requests at EQUAL length would sail past
  // `seen == kv_base` into a plausible answer over a mixed history.
  CHECK_THROWS(step(/*token=*/3, /*position=*/2, /*computed=*/2, /*nreqs=*/2));
}

// ───────────────────────────────────────────────────────────────────────────
// ENG-ASYNC-DEVICE-IDS-2544 — THE ASYNCHRONOUS DEVICE-IDENTIFIER GATE.
//
// Spec `.agents/specs/eng-async-device-ids-2544.md`, issue
// [#2544](https://github.com/mudler/vllm.cpp/issues/2544); the contract is
// [#1305](https://github.com/mudler/vllm.cpp/issues/1305)'s.
//
// WHAT IT MEASURES. On the asynchronous serving path the runner's combine
// splices each decode row's sampled token into the DEVICE identifiers on the
// main queue and leaves the host `token_ids` deliberately stale for decode rows
// (`src/vllm/v1/worker/gpu/runner.cpp`, the mirror arm, which is the DEFAULT on
// CUDA). A forward that embeds the host vector generates every step after the
// first from the previous step's identifiers — and because `token_ids_cpu` is
// zero-initialised, from TOKEN ID 0. Silently, at rc=0, with plausible output.
//
// WIRED BUT NOT CONVICTED, and the case says so rather than implying otherwise.
// #2544 named this registration on a `grep` and called it "a candidate, not a
// conviction"; this row has no staged DeepSeek-V4 checkpoint and no GPU time for
// one. What is proved here is that the identifiers reach the embed through
// `ModelRegistry::Forward` — not that service ever hands this model a live
// `device_token_ids`. Spec `## Owed` O4.
//
// THREE RUNS, because two of them cannot separate the cases:
//
//     A  right host id, no mirror            -> the reference
//     B  ZERO host id, no mirror             -> must DIFFER from A
//     C  ZERO host id, mirror carries A's    -> must EQUAL A, bit for bit
//
// B is the control. Without it a forward that ignored its identifiers entirely
// would satisfy C. ZERO is the wrong value on purpose: it is what the real
// defect feeds. A's identifier is NON-ZERO, so C cannot agree for the wrong
// reason.
//
// ONE MODEL PER RUN. This arm carries a per-model compressor state that refuses
// unless it has seen exactly `kv_base` tokens, so three runs sharing one
// `LoadedModel` would be three different histories rather than three readings of
// one step.
TEST_CASE("dsv4 #2544: the paged arm embeds the async mirror's DEVICE id, not the stale host one") {
  dsv4_exl3_fixture::FixtureOptions opt;
  opt.layers = 2;
  opt.compress_ratios = {0, 128};
  opt.real_dsa_geometry = false;
  auto f = dsv4_exl3_fixture::BuildFixture(opt);
  const vllm::DeepseekV4Params params = vllm::ParseDeepseekV4Params(f->config);
  const int64_t nlayers = params.num_hidden_layers;
  const int64_t hd = params.head_dim;

  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vt::Backend& be = vt::GetBackend(queue.device.type);

  // The identifier the runner's combine would have written. NON-ZERO, and inside
  // the fixture's vocabulary so run B is a wrong ANSWER and not a refusal.
  constexpr int32_t kTrue = 3;
  REQUIRE(kTrue < params.vocab_size);

  // One step through `ModelRegistry::Forward` on a FRESH model, with the host
  // identifier and the mirror chosen by the caller.
  const auto run = [&](int32_t host_id, bool mirror) {
    const vllm::ModelSource source = vllm::ModelSource::FromSafetensors(f->shards);
    std::unique_ptr<vllm::LoadedModel> model =
        vllm::ModelRegistry::Load(f->config, source);
    REQUIRE(model != nullptr);

    const int64_t nb = 4, bs = 8;
    std::vector<std::vector<float>> storage(static_cast<size_t>(nlayers));
    std::vector<vllm::PagedKvCache> attn_kv(static_cast<size_t>(nlayers));
    std::vector<std::string> names;
    for (int64_t l = 0; l < nlayers; ++l) {
      const size_t i = static_cast<size_t>(l);
      storage[i].assign(static_cast<size_t>(nb * bs * hd), 0.0f);
      attn_kv[i].data = storage[i].data();
      attn_kv[i].dtype = vt::DType::kF32;
      attn_kv[i].num_blocks = nb;
      attn_kv[i].block_size = bs;
      attn_kv[i].num_kv_heads = 1;
      attn_kv[i].head_size = hd;
      names.push_back("model.layers." + std::to_string(l) + ".attn.swa_cache");
    }
    vllm::MultiKvCacheIndex mk;
    mk.layer_names = &names;

    const std::vector<int32_t> tok{host_id};
    const std::vector<int32_t> pos{0};
    const std::vector<int32_t> li{0};
    std::vector<vllm::GdnStateCache> gdn_state;
    const vllm::v1::GDNAttentionMetadata gdn_meta{};
    vllm::v1::CommonAttentionMetadata attn_meta{};
    attn_meta.num_reqs = 1;
    attn_meta.num_computed_tokens_cpu = {0};
    vllm::ModelForwardInput in{.token_ids = tok,
                               .positions = pos,
                               .attn_meta = attn_meta,
                               .gdn_meta = gdn_meta,
                               .attn_kv = attn_kv,
                               .gdn_state = gdn_state,
                               .config = f->config,
                               .queue = queue,
                               .logits_indices = li,
                               .num_reqs = 1};
    in.multi_kv = &mk;
    // A REAL backend allocation, never the host vector's address. On CPU the two
    // are the same kind of pointer, so this buys nothing today; it is what makes
    // the case correct by construction on a device.
    int32_t* dev = nullptr;
    if (mirror) {
      dev = static_cast<int32_t*>(be.Alloc(sizeof(int32_t)));
      const int32_t truth = kTrue;
      be.Copy(queue, dev, &truth, sizeof(int32_t));
      be.Synchronize(queue);
      // Set AFTER aggregate construction, exactly as `runner.cpp` sets it.
      in.device_token_ids = dev;
    }
    const vllm::ForwardLogits out = vllm::ModelRegistry::Forward(*model, in);
    if (dev != nullptr) be.Free(dev);
    REQUIRE(out.host.size() == static_cast<size_t>(params.vocab_size));
    return out.host;
  };

  const std::vector<float> ref = run(kTrue, /*mirror=*/false);
  // FINITENESS BEFORE ANY COMPARISON. Against a NaN both `==` and `!=` are
  // false, so an equality gate and a difference gate BOTH report success on a
  // forward that produced no numbers at all.
  REQUIRE(NonFinite(ref) == 0);

  const std::vector<float> stale = run(/*host_id=*/0, /*mirror=*/false);
  REQUIRE(NonFinite(stale) == 0);
  size_t moved = 0;
  for (size_t i = 0; i < ref.size(); ++i)
    if (std::memcmp(&ref[i], &stale[i], sizeof(float)) != 0) ++moved;
  CHECK_MESSAGE(moved > 0,
                "zeroing the host id changed nothing, so this fixture cannot "
                "see an identifier at all and the gate below proves nothing");

  const std::vector<float> via_device = run(/*host_id=*/0, /*mirror=*/true);
  size_t differing = 0;
  for (size_t i = 0; i < ref.size(); ++i)
    if (std::memcmp(&ref[i], &via_device[i], sizeof(float)) != 0) ++differing;
  CHECK_MESSAGE(differing == 0,
                "registry forward, mirror vs host reference: " << differing
                    << " of " << ref.size() << " logits differ, so this forward "
                    "embedded the STALE host id (#2544, #1305)");
  MESSAGE("dsv4 device-ids: control moved " << moved << " floats; mirror vs "
          "reference differ in " << differing << " of " << ref.size());
}

// ── KV-DSV4-MULTICACHE W8 slice 4 (#2455): THE EXL3 ARM OF THE SAME SEAM ─────
//
// The case above binds FLOAT pages and says, in a comment written before slice
// 4, that "the packed arm is refused by name and is owed to KV-DSV4-MULTICACHE
// W8". Slice 4 is what that comment was waiting for, and this case is the other
// half of the wiring it landed.
//
// WHY IT EXISTS SEPARATELY FROM THE GGUF CASE. `ForwardDeepseekV4ForCausalLM`
// has TWO junctions that carry the storage row count from
// `ResolveDeepseekV4SwaPages` into a forward, and they are different lines with
// different arguments: this one passes `dsa_dense=false` and
// `have_compressor_state=true`, so its refusals and its DECODE-ONLY bound are
// not the GGUF arm's. The slice-4 review proved the GGUF junction ungated;
// nothing measured this one either, and one gate cannot cover both lines.
//
// THE DECODE-ONLY BOUND IS WHY THIS STEP IS ONE TOKEN. With carried compressor
// state the resolver refuses any step carrying more than one token, because the
// window/compressed merge reshapes the two LSE buffers rather than transposing
// them (MODEL-DSV4-PAGED-ENTRY, #2447).
TEST_CASE("W8 slice 4: ModelRegistry::Forward carries rows_per_block on the EXL3 arm (#2455)") {
  dsv4_exl3_fixture::FixtureOptions opt;
  opt.layers = 2;
  // NO COMPRESSOR ON ANY LAYER. A compressor layer holding a PACKED page is
  // refused by name -- `CompressorLayerStep` attends through
  // `vt::MlaDecodeAttention`, which takes a rank-3 float cache -- and that
  // refusal ends the whole STEP rather than skipping the layer.
  opt.compress_ratios = {0, 0};
  opt.real_dsa_geometry = false;
  auto f = dsv4_exl3_fixture::BuildFixture(opt);

  const vllm::ModelSource source = vllm::ModelSource::FromSafetensors(f->shards);
  std::unique_ptr<vllm::LoadedModel> model =
      vllm::ModelRegistry::Load(f->config, source);
  REQUIRE(model != nullptr);

  const vllm::DeepseekV4Params params = vllm::ParseDeepseekV4Params(f->config);
  const int64_t nlayers = params.num_hidden_layers;
  const int64_t hd = params.head_dim;
  REQUIRE(nlayers == 2);
  // The packed layout is upstream's literals, not parameters
  // (`cache_utils.py:180-183`). This fixture already writes the real 512-wide
  // MLA head with a 64-wide RoPE half, which is why a packed page is
  // expressible here at all.
  REQUIRE(hd == vt::kFp8DsMlaInputDim);

  const int64_t rows = 8;  // STORAGE rows per block -- the value under test
  const int64_t num_blocks = 4;
  // From the SHARED packer. A local `RoundUp(rows * 584, 576)` would be a second
  // derivation of the padding rule, which is the drift the one packer prevents.
  const vllm::deepseek_v4::Fp8DsMlaPageLayout P =
      vllm::deepseek_v4::MakeFp8DsMlaPageLayout(
          vllm::deepseek_v4::MakeFp8DsMlaLayout(vt::kFp8DsMlaNopeDim,
                                                vt::kFp8DsMlaRopeDim,
                                                vt::kFp8DsMlaQuantBlock),
          rows);
  const int64_t block_bytes = P.padded_block_bytes;
  // ASSERTED BEFORE IT IS USED AS A SIZE. `padded_block_bytes` is an `int64_t`
  // the shared packer returns and this TU cannot prove non-negative, so the
  // `static_cast<size_t>` below otherwise carries the [2^63, 2^64) range into a
  // `memset` bound once -O3 inlines the `assign` -- the CPU Release failure the
  // GGUF sibling hit at `-Werror=stringop-overflow=`.
  REQUIRE(block_bytes > 0);
  REQUIRE(num_blocks > 0);

  // POISON, so "the store ran" is an observation and not a check that zeros
  // stayed zero.
  std::vector<std::vector<uint8_t>> pstore(static_cast<size_t>(nlayers));
  std::vector<vllm::PagedKvCache> attn_kv(static_cast<size_t>(nlayers));
  std::vector<std::string> names;
  for (int64_t l = 0; l < nlayers; ++l) {
    const size_t i = static_cast<size_t>(l);
    pstore[i].assign(static_cast<size_t>(num_blocks * block_bytes), 0xA5);
    attn_kv[i].data = pstore[i].data();
    attn_kv[i].dtype = vt::DType::kI8;  // the architecture's published default
    attn_kv[i].num_blocks = num_blocks;
    attn_kv[i].block_size = rows;
    attn_kv[i].num_kv_heads = 1;
    attn_kv[i].head_size = hd;
    // Filled ONLY by `GPUModelRunner::initialize_kv_cache` in production, and
    // the reason the field exists: the view and the allocated page disagree.
    attn_kv[i].page_size_bytes = block_bytes;
    names.push_back("model.layers." + std::to_string(l) + ".attn.swa_cache");
  }
  vllm::MultiKvCacheIndex mk;
  mk.layer_names = &names;

  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  std::vector<vllm::GdnStateCache> gdn_state;
  const vllm::v1::GDNAttentionMetadata gdn_meta{};

  // THE STEP RESUMES AT A NON-ZERO kv_base, for the same reason as the GGUF
  // case: `slots[t] = kv_base + t`, so the storage row the store touches is the
  // observable consequence of the registry reading `kv_base` off the step. At 0
  // a carried `kv_base` and a dropped one write the same byte, and the link is
  // ungated. Safe here because this fixture has NO compressor layer, so
  // `CompressorLayerStep`'s `seen == kv_base` guard (`deepseek_v4_dsa.cpp:423`)
  // never runs -- that guard is what a resumed step would otherwise meet.
  //
  // THAT REASON IS ABOUT THE GUARD, NOT ABOUT THE HISTORY, and an earlier
  // wording stopped there and left the other half unsaid. The rows below
  // `kv_base` still have to be WRITTEN by something, or this case resumes over
  // bytes nothing ever stored. They are, by the warm steps below.
  const int64_t kv_base = 4;
  REQUIRE(kv_base < rows);
  const std::vector<int32_t> tok{1};
  // ABSOLUTE position, matching the resumed context.
  const std::vector<int32_t> pos{static_cast<int32_t>(kv_base)};
  const std::vector<int32_t> li{0};
  vllm::v1::CommonAttentionMetadata attn_meta{};
  attn_meta.num_reqs = 1;
  attn_meta.num_computed_tokens_cpu = {static_cast<int32_t>(kv_base)};
  vllm::ModelForwardInput in{.token_ids = tok,
                             .positions = pos,
                             .attn_meta = attn_meta,
                             .gdn_meta = gdn_meta,
                             .attn_kv = attn_kv,
                             .gdn_state = gdn_state,
                             .config = f->config,
                             .queue = queue,
                             .logits_indices = li,
                             .num_reqs = 1};
  in.multi_kv = &mk;
  // LEFT AT ITS DEFAULT (true), exactly as the runner sets it. This arm is
  // tested BEFORE the `gather_logits` branch, which is why it is the arm a
  // default decode step actually takes.
  REQUIRE(in.gather_logits);

  // ── THE PRIOR STEPS, and why this case drives FOUR of them ──────────────
  //
  // A resumed step ATTENDS its history: `n_keys = kv_base + T`, so rows
  // 0..kv_base-1 are read back and dequantised (`deepseek_v4.cpp:1268`). Left
  // at 0xA5 that history is FICTITIOUS: the case would resume over bytes
  // nothing ever wrote, `CHECK(NonFinite(out.host) == 0)` below would be a
  // property of the fp8 decoder applied to poison rather than a statement about
  // this arm, and the decode shape would be one production never takes. The
  // GGUF sibling was given a real history for exactly this reason.
  //
  // FOUR STEPS RATHER THAN ONE, and that is forced rather than stylistic. This
  // junction passes `have_compressor_state = true`
  // (`deepseek_v4_registry.cpp:184`), so `deepseek_v4.cpp:3743` REFUSES any step
  // carrying more than one token: the composed arm's window/compressed merge
  // reshapes the two LSE buffers instead of transposing them. The GGUF sibling
  // passes `dsa_dense = true` with no carried state and can warm its history in
  // one four-token step; this arm cannot. So the history is written one token
  // per step at `kv_base` 0, 1, 2 and 3 -- which is also exactly the decode
  // shape a served request takes.
  for (int64_t w = 0; w < kv_base; ++w) {
    vllm::v1::CommonAttentionMetadata warm_meta{};
    warm_meta.num_reqs = 1;
    warm_meta.num_computed_tokens_cpu = {static_cast<int32_t>(w)};
    const std::vector<int32_t> warm_tok{1};
    // ABSOLUTE position, matching the context this step resumes at.
    const std::vector<int32_t> warm_pos{static_cast<int32_t>(w)};
    vllm::ModelForwardInput warm{.token_ids = warm_tok,
                                 .positions = warm_pos,
                                 .attn_meta = warm_meta,
                                 .gdn_meta = gdn_meta,
                                 .attn_kv = attn_kv,
                                 .gdn_state = gdn_state,
                                 .config = f->config,
                                 .queue = queue,
                                 .logits_indices = li,
                                 .num_reqs = 1};
    warm.multi_kv = &mk;
    const vllm::ForwardLogits warm_out =
        vllm::ModelRegistry::Forward(*model, warm);
    CAPTURE(w);
    REQUIRE(warm_out.host.size() == static_cast<size_t>(params.vocab_size));
    // A REAL statement about this arm: each warm step after the first resumes
    // over rows its predecessors actually wrote.
    CHECK(NonFinite(warm_out.host) == 0);
  }

  // THE HISTORY AS THE ENGINE LEFT IT. Rows 0..kv_base-1 now hold real packed
  // tokens, and the step under test must not touch ONE BYTE of them.
  const std::vector<std::vector<uint8_t>> before = pstore;

  // THE STEP. A wrong row count throws out of here rather than returning.
  const vllm::ForwardLogits out = vllm::ModelRegistry::Forward(*model, in);
  CHECK(out.rows == 1);
  CHECK(out.vocab == params.vocab_size);
  REQUIRE(out.host.size() == static_cast<size_t>(params.vocab_size));
  CHECK(NonFinite(out.host) == 0);

  // 1. THE WRITE REACHED THE PACKED PAGE THROUGH THE REGISTRY, on every layer.
  //    The 8th scale byte is the explicit zero pad upstream writes
  //    (`cache_utils.py:148-149`), so it is asserted rather than assumed.
  for (int64_t l = 0; l < nlayers; ++l) {
    const std::vector<uint8_t>& blk = pstore[static_cast<size_t>(l)];
    CAPTURE(l);
    bool moved = false;
    // ROW `kv_base`, where `slots[0] = kv_base + 0` puts this step's one token.
    const uint8_t* data = blk.data() + kv_base * P.token_data_size;
    for (int64_t i = 0; i < P.token_data_size; ++i)
      if (data[i] != 0xA5) { moved = true; break; }
    CHECK(moved);
    const uint8_t* sc =
        blk.data() + P.scale_region_offset + kv_base * P.scale_dim;
    CHECK(sc[P.token.n_nope_blocks] == 0);
    // AND NOTHING BELOW `kv_base` MOVED -- the half that catches a registry
    // which dropped the `kv_base` read and wrote row 0 instead. Compared
    // against the SNAPSHOT rather than against poison, because those rows hold
    // real packed tokens by now.
    //
    // BOTH REGIONS, and the data region alone is not enough. A token's 8 scale
    // bytes live at `scale_region_offset + row * scale_dim`, in a DIFFERENT
    // region from its 576 data bytes (`cache_utils.py:59-66`), so a defect that
    // rewrote only the history's SCALE bytes moves no data byte and passed a
    // data-only comparison unseen.
    const std::vector<uint8_t>& was = before[static_cast<size_t>(l)];
    int64_t below = 0;
    for (int64_t r = 0; r < kv_base; ++r) {
      const uint8_t* now_d = blk.data() + r * P.token_data_size;
      const uint8_t* was_d = was.data() + r * P.token_data_size;
      for (int64_t i = 0; i < P.token_data_size; ++i)
        if (now_d[i] != was_d[i]) ++below;
      const uint8_t* now_s =
          blk.data() + P.scale_region_offset + r * P.scale_dim;
      const uint8_t* was_s =
          was.data() + P.scale_region_offset + r * P.scale_dim;
      for (int64_t i = 0; i < P.scale_dim; ++i)
        if (now_s[i] != was_s[i]) ++below;
    }
    CHECK(below == 0);
    // 2. AND IT STAYED INSIDE ITS BLOCK. Everything from `rows * 584` on is
    //    alignment padding that no store may reach.
    int64_t clobbered = 0;
    for (int64_t i = P.real_block_bytes; i < block_bytes; ++i)
      if (blk[static_cast<size_t>(i)] != 0xA5) ++clobbered;
    CHECK(clobbered == 0);
  }
}
