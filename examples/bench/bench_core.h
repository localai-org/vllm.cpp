// vllm.cpp benchmark core (M2.1). ORIGINAL tool (no upstream 1:1 mirror), but the
// METRICS and their report format deliberately mirror vLLM's `vllm bench serve`
// (vllm/benchmarks/serve.py @ the pinned oracle, class BenchmarkMetrics +
// benchmark()/save-and-print path) and `vllm bench throughput`
// (vllm/benchmarks/throughput.py) so the numbers this prints are directly
// comparable to a vLLM run on the same box + same model + same workload. See
// examples/bench/main.cpp for the CLI and the dgx invocation.
//
// Mirrored from serve.py:
//   - request_throughput = completed / dur_s               (serve.py:731)
//   - output_throughput  = sum(output_lens) / dur_s        (serve.py:733)
//   - total_token_throughput = (in+out) / dur_s            (serve.py:734)
//   - ttft: time-to-first-token per request                (serve.py:615)
//   - tpot: (latency - ttft) / (output_len - 1)            (serve.py:609-611)
//   - itl:  inter-token latency per streamed chunk         (serve.py:614)
//   - e2el: end-to-end per-request latency                 (serve.py:616)
//   mean/median/std/percentile reduction over each         (serve.py:726-758)
// Our ADDITION (labeled separately): a prefill-vs-decode token-throughput split
// — Input tok/s = total_input/dur, Output(decode) tok/s = total_output/dur
// (== serve.py output_throughput) — which is exactly the gate #1 measurement
// ("prefill AND decode throughput at large concurrency", .agents/gates.md).
// Both divide by the WHOLE run, which is right for a saturated many-request run
// and misleading at low concurrency; a separate `prefill_throughput` divides by
// the time actually spent in prefill and is the figure comparable to
// llama-bench's pp128. See the field comment for how conflating the two
// produced a bogus cross-engine prefill ratio. We drive the engine in DELTA output mode (like a streaming
// serve client) so TTFT/ITL are observed the same way serve.py observes them.
//
// This header is deliberately header-only + inline so both the CLI
// (examples/bench/main.cpp) and the ctest smoke (tests/examples/test_bench.cpp)
// compile the exact same measurement path.
#ifndef VLLM_EXAMPLES_BENCH_BENCH_CORE_H_
#define VLLM_EXAMPLES_BENCH_BENCH_CORE_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/config/speculative.h"
#include "vllm/entrypoints/chat_template.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/outputs.h"
#include "vllm/sampling_params.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/engine/async_llm.h"
#include "vllm/v1/engine/llm_engine.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"
#include "vt/fp8_kv.h"
#include "vt/tensor.h"

namespace vllm::bench {

enum class OutputWaitMode {
  kPoll,
  kBlockingC1,
};

#ifdef VLLM_BENCH_TEST_OUTPUT_WAIT_TRACE
// Test-only observation seam. Production benchmark translation units do not
// define VLLM_BENCH_TEST_OUTPUT_WAIT_TRACE, so tracing adds no state or work to
// the measured loop.
enum class OutputWaitTestEvent {
  kNowaitReady,
  kNowaitEmpty,
  kBlockingWait,
};

// Runtime probe controls used only by test_bench. The seeded collector value is
// input to the real AsyncLLM method; the observer below runs only after that
// method has returned, so neither control can replace the production call.
struct OutputWaitTestProbe {
  bool seed_first_collector = false;
  bool stop_after_first_nowait = false;
};
#endif

inline const char* OutputWaitModeName(OutputWaitMode mode) {
  switch (mode) {
    case OutputWaitMode::kPoll:
      return "poll";
    case OutputWaitMode::kBlockingC1:
      return "blocking-c1";
  }
  throw std::invalid_argument("unknown benchmark output wait mode");
}

inline OutputWaitMode ParseOutputWaitMode(std::string_view name) {
  if (name == "poll") return OutputWaitMode::kPoll;
  if (name == "blocking-c1") return OutputWaitMode::kBlockingC1;
  throw std::invalid_argument("unknown benchmark output wait '" +
                              std::string(name) +
                              "' (expected poll or blocking-c1)");
}

// ── Config (mirrors the `vllm bench throughput` / `serve` knobs we support) ────
struct BenchConfig {
  // Empty => SYNTHETIC tiny CPU engine (smoke). Otherwise a model dir / .gguf.
  std::string model_path;
  // Optional ShareGPT JSON file. When set, consume the first conversation turn
  // from each entry so vLLM and vllm.cpp can run byte-identical prompts.
  std::string dataset_path;
  // Optional JSON output path containing generated IDs in submission order.
  std::string output_token_ids_path;
  // Optional structured result artifact. This is separate from
  // output_token_ids_path so the established array-only token fixture stays
  // byte-compatible.
  std::string output_json_path;
  // Optional speculative-decoding config JSON, e.g.
  // '{"method":"mtp","num_speculative_tokens":1}'. Empty => spec decode OFF
  // (production non-spec path, byte-identical to pre-existing runs). Parsed via
  // vllm::ParseSpeculativeConfigJson and forwarded to EngineParams for the real
  // checkpoint path only (the synthetic engine ignores it).
  std::string speculative_config;
  int num_prompts = 8;     // N: total requests to submit.
  int input_len = 16;      // L: target prompt tokens per request.
  int output_len = 16;     // O: max_tokens per request (greedy => exactly O).
  int concurrency = 4;     // C: max in-flight requests admitted to the engine.
  // `poll` preserves the production benchmark's scan+yield loop. The
  // concurrency-1-only `blocking-c1` control performs the same nowait probe,
  // then blocks on the sole collector so Pi measurements can isolate polling.
  OutputWaitMode output_wait = OutputWaitMode::kPoll;
#ifdef VLLM_BENCH_TEST_OUTPUT_WAIT_TRACE
  std::vector<OutputWaitTestEvent>* output_wait_test_trace = nullptr;
  OutputWaitTestProbe output_wait_test_probe;
#endif
  uint64_t seed = 0;       // prompt-generation RNG seed.
  double temperature = 0;  // <= 0 => greedy (deterministic).
  bool quiet = false;      // suppress per-progress logging to stderr.
  // Per-step token budget (chunked-prefill knob). 0 => the engine's bounded
  // default (ResolveMaxNumBatchedTokens). Exposed so the GB10 memory ramp can
  // bound the per-step GDN prefill activation explicitly (the 27B 8x1024 OOM
  // fix): a smaller budget splits a big prefill batch across more steps.
  int max_num_batched_tokens = 0;
  // KV/GDN-state cache blocks to preallocate. 0 => the heuristic below
  // (concurrency * seq_blocks * 2). Exposed for the GB10 memory ramp: the
  // preallocated f32 KV + GDN mamba-state cache scales with num_blocks and is the
  // dominant unified-memory consumer at high concurrency, SEPARATE from the
  // (chunked-prefill-bounded) per-step activation. Setting it to just enough for
  // C concurrent (input+output)-long sequences keeps peak RAM bounded.
  int num_blocks = 0;
  // KV-FP8 W7 (#2619): vLLM's `CacheConfig.cache_dtype` (config/cache.py:19-36,76),
  // spelled `--kv-cache-dtype` on this harness exactly as on `vllm-server` and
  // `vllm-cli`. "auto" is the default and is byte-identical to before the flag
  // existed. The string is passed to the engine VERBATIM: `vllm::v1::
  // ParseCacheDType` is the one place a CacheDType name is accepted or refused,
  // and a benchmark client that re-listed the legal names would be a second
  // spelling of that contract, free to drift from it.
  std::string kv_cache_dtype = "auto";
  // #2759: EOS polarity. `MakeSampling` used to hardcode `sp.ignore_eos = true`,
  // so no request this harness ever admitted could terminate naturally, and no
  // report said so. Spelled `--ignore-eos` exactly as vLLM's own benchmark
  // client (`vllm/benchmarks/serve.py:1757-1762`) and negated as
  // `--no-ignore-eos` under this tree's own convention (`server_main.cpp`'s
  // `--no-enable-thinking`). The DEFAULT is true, which is the previous
  // behavior byte for byte: every landed figure was measured with EOS
  // suppressed, and flipping the default would change what those numbers mean
  // without re-measuring one of them.
  bool ignore_eos = true;
  // #2759: chat-template polarity. `--skip-chat-template` from
  // `vllm/benchmarks/datasets/datasets.py:1654-1658`, same negation, same
  // default-preserving polarity: true sends the raw prompt string, which is
  // what this harness has always done. The comparator for
  // `BENCH-QWEN38-27B-SOTA` posts to `/v1/chat/completions`, so its protocol
  // cannot be matched without the other setting.
  bool skip_chat_template = true;
  // #2759: the template SOURCE -- a file path, or the template itself in
  // single-line form. `vllm/entrypoints/launchers/cli_args.py:80-82` verbatim.
  // Empty selects the model's own template through `LoadChatTemplateForModel`,
  // the identical selection path `server_main.cpp` uses.
  std::string chat_template;
  // #2759: the SERVER-level `chat_template_kwargs` default, spelled
  // `--enable-thinking` / `--no-enable-thinking` exactly as `server_main.cpp`
  // and resolved by the same rule (`DefaultChatTemplateKwargs`). Unset leaves
  // `enable_thinking` Jinja-undefined, which is NOT the same as false (#1681),
  // and the comparator sends false.
  std::optional<bool> enable_thinking;
};

// ── Per-request timing record (client-side, exactly what serve.py records). ────
struct RequestRecord {
  double arrival_s = 0.0;       // when we admitted it to the engine.
  double first_token_s = -1.0;  // when its first output token was observed.
  double completion_s = 0.0;    // when it finished.
  double last_token_s = 0.0;    // running: previous token arrival (for ITL).
  int prompt_tokens = 0;
  int output_tokens = 0;
  std::vector<int32_t> prompt_token_ids;
  std::vector<int32_t> output_token_ids;
  std::vector<double> itls;  // inter-token latencies (s), one per chunk>1st.
  bool finished = false;
};

// ── Aggregated result (field names mirror serve.py BenchmarkMetrics). ──────────
struct BenchResult {
  // Auditable engine selection. The comparison harness uses the production
  // AsyncLLM frontend even when async scheduling resolves OFF (then the core's
  // queue depth is one), matching vLLM's frontend across the ON/OFF control.
  bool async_frontend = false;
  bool async_scheduling_enabled = false;
  // True when prompt tokenization was completed before the benchmark clock and
  // timed admission used AsyncLLM's TokensPrompt overload. Auditable in every
  // result so a benchmark artifact cannot silently mix frontend modes.
  bool pretokenized_admission = false;
  int max_concurrent_batches = 1;
  OutputWaitMode output_wait = OutputWaitMode::kPoll;
  // Number of calls made to AsyncLLM::get_output after a failed nowait probe.
  // This proves a blocking-c1 artifact actually exercised its control path.
  int64_t blocking_wait_calls = 0;
  int completed = 0;
  double duration_s = 0.0;
  int64_t total_input = 0;
  int64_t total_output = 0;
  // Per-request generated IDs in submission order. This makes the benchmark
  // workload usable as a token-for-token correctness gate before timing it.
  std::vector<std::vector<int32_t>> prompt_token_ids;
  std::vector<std::vector<int32_t>> output_token_ids;
  double request_throughput = 0.0;       // req/s
  double output_throughput = 0.0;        // tok/s  (decode)
  double input_throughput = 0.0;         // tok/s  input/WHOLE run — our split
  // TRUE prefill rate: input tokens divided by the time actually spent in
  // prefill (the sum of TTFTs), NOT by the whole run. `input_throughput` above
  // divides by total duration, which is the right shape for a saturated
  // many-request run but degenerates badly at low concurrency: for a 1-prompt
  // 32-in/8-out run it reports 32/E2EL, i.e. it charges the whole DECODE to
  // prefill. That is how a "~500x behind llama.cpp on prefill" figure got
  // quoted against llama-bench's pp128, which is prefill-ONLY. This field is
  // the one comparable to pp128.
  double prefill_throughput = 0.0;       // tok/s  input/sum(TTFT)
  double total_token_throughput = 0.0;   // tok/s
  double mean_per_stream_decode = 0.0;   // 1000/mean_tpot, tok/s per stream
  // ms statistics (mean/median/p99), like serve.py.
  double mean_ttft_ms = 0, median_ttft_ms = 0, p99_ttft_ms = 0;
  double mean_tpot_ms = 0, median_tpot_ms = 0, p99_tpot_ms = 0;
  double mean_itl_ms = 0, median_itl_ms = 0, p99_itl_ms = 0;
  double mean_e2el_ms = 0, median_e2el_ms = 0, p99_e2el_ms = 0;
  // Speculative-decoding telemetry (only meaningful when spec decode is ON).
  // spec_proposed = draft tokens VERIFIED; spec_accepted = draft tokens
  // ACCEPTED. Acceptance rate = accepted / proposed. Both 0 when spec is OFF.
  bool spec_on = false;
  int64_t spec_proposed = 0;
  int64_t spec_accepted = 0;
  // KV-FP8 W7 (#2619): the KV storage dtype the loader ACTUALLY sized blocks
  // from, read back out of `LoadedEngine::kv_cache_config()` rather than echoed
  // from the flag. The two differ whenever the checkpoint declares
  // `kv_cache_quant_algo` and no flag was typed -- `FromModelDir` honours the
  // declaration, so a report that echoed the flag would print "auto" over an
  // fp8 measurement. Reading the spec back means this line cannot drift from
  // the allocation, because it IS the allocation.
  std::string resolved_kv_cache_dtype = "unknown";
  // #2759: what the run actually did, beside what it was asked to do. The EOS
  // value is read back out of the `SamplingParams` object `MakeSampling` builds
  // for an admitted request rather than copied from the config, for the same
  // reason `resolved_kv_cache_dtype` is read back out of the cache the loader
  // sized: a report that echoes the request cannot detect a flag that parses
  // and is then dropped, which is exactly the defect #2759 names.
  bool resolved_ignore_eos = true;
  std::string resolved_chat_template = "skipped (raw prompt)";
  std::string resolved_chat_template_kwargs = "{}";
};

// KV-FP8 W7 (#2619): name the KV storage dtype of a RESOLVED KV cache config.
// `vt::Name` alone would print "i8" for an fp8 page, which is the storage width
// and not the thing a published number has to state; the fp8 interpretation on
// the same spec is what says which fp8. Returns "unknown" for a config with no
// attention group (a pure recurrent model allocates no paged KV).
inline std::string ResolvedKvCacheDTypeName(const vllm::v1::KVCacheConfig& cfg) {
  for (const auto& group : cfg.kv_cache_groups) {
    const auto* attn =
        dynamic_cast<const vllm::v1::AttentionSpec*>(group.kv_cache_spec.get());
    if (attn == nullptr) continue;
    switch (attn->fp8_kind) {
      case vt::Fp8KVCacheDataType::kFp8E4M3:
        return "fp8_e4m3 (1-byte pages)";
      case vt::Fp8KVCacheDataType::kFp8E5M2:
        return "fp8_e5m2 (1-byte pages)";
      case vt::Fp8KVCacheDataType::kAuto:
        break;
    }
    return vt::Name(attn->dtype);
  }
  return "unknown (no attention KV group)";
}

// ── numpy-style linear-interpolation percentile (matches np.percentile). ───────
inline double Percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  if (v.size() == 1) return v.front();
  const double rank = (p / 100.0) * static_cast<double>(v.size() - 1);
  const auto lo = static_cast<size_t>(std::floor(rank));
  const auto hi = static_cast<size_t>(std::ceil(rank));
  const double frac = rank - static_cast<double>(lo);
  return v[lo] + (v[hi] - v[lo]) * frac;
}

inline double Mean(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  double s = 0.0;
  for (double x : v) s += x;
  return s / static_cast<double>(v.size());
}

// Benchmark frontend parity selector. The pinned vLLM comparison tokenizes all
// prompts before starting its closed-loop clock and submits TokensPrompt IDs.
// Keep that production-parity path as the safe default: only exact `0` selects
// the timed string-admission rollback; invalid spellings stay default-ON.
inline bool ResolveBenchPretokenizedAdmission(const char* env_value) {
  return env_value == nullptr || std::string_view(env_value) != "0";
}

// Pure production dispatch seam: exactly one callback is invoked. Keeping the
// selector separate from the benchmark loop lets the CPU contract test detect
// parser inversion and accidental double admission without hot-path counters.
template <typename PretokenizedCallback, typename TimedStringCallback>
inline decltype(auto) DispatchBenchPromptAdmission(
    const char* env_value, PretokenizedCallback&& pretokenized_callback,
    TimedStringCallback&& timed_string_callback) {
  if (ResolveBenchPretokenizedAdmission(env_value)) {
    return std::forward<PretokenizedCallback>(pretokenized_callback)();
  }
  return std::forward<TimedStringCallback>(timed_string_callback)();
}

// One benchmark refill is one observable engine wave. Record every selected
// request's arrival before invoking exactly one engine batch publish; in rollback
// mode, string tokenization therefore remains timed but begins only after the
// complete wave has the same arrival boundary as the pretokenized path.
template <typename Engine, typename ArrivalCallback,
          typename PretokenizedWaveFactory, typename TimedStringWaveFactory>
inline decltype(auto) DispatchBenchPromptWaveAdmission(
    Engine& engine, const char* env_value, std::size_t wave_size,
    ArrivalCallback&& arrival_callback,
    PretokenizedWaveFactory&& pretokenized_wave_factory,
    TimedStringWaveFactory&& timed_string_wave_factory) {
  for (std::size_t offset = 0; offset < wave_size; ++offset) {
    std::forward<ArrivalCallback>(arrival_callback)(offset);
  }
  if (ResolveBenchPretokenizedAdmission(env_value)) {
    return engine.add_request_wave(
        std::forward<PretokenizedWaveFactory>(pretokenized_wave_factory)());
  }
  return engine.add_request_wave(
      std::forward<TimedStringWaveFactory>(timed_string_wave_factory)());
}

// Own the ordering boundary between workload preparation and measurement.
// The clock callback is invoked exactly once and only after every default-path
// prompt has been encoded with the same special-token policy as InputProcessor.
// Supplying the clock keeps this pure host contract deterministic in tests.
template <typename TokenizerLike, typename StartClockCallback>
inline auto PretokenizeBenchPromptsThenStartClock(
    bool pretokenized_admission, const TokenizerLike& tokenizer,
    const std::vector<std::string>& prompts,
    StartClockCallback&& start_clock_callback) {
  std::vector<std::vector<int32_t>> pretokenized_prompts;
  if (pretokenized_admission) {
    pretokenized_prompts.reserve(prompts.size());
    for (const std::string& prompt : prompts) {
      pretokenized_prompts.push_back(
          tokenizer.EncodeWithSpecialTokens(prompt));
    }
  }
  auto start = std::forward<StartClockCallback>(start_clock_callback)();
  return std::make_pair(std::move(pretokenized_prompts), std::move(start));
}

// ────────────────────────────── Synthetic model ───────────────────────────────
// A tiny hybrid-MoE Qwen3.6 (mirrors tests/vllm/v1/test_llm_engine.cpp) so the
// harness runs end-to-end on the CPU box with no checkpoint. The NUMBERS from
// this engine are meaningless (toy weights); it proves the harness drives the
// real V1 engine loop + produces sane metrics. Real numbers come from a GB10
// run with --model.
namespace detail {

inline uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
inline float RandV(uint64_t seed) {
  const double u =
      static_cast<double>(Mix(seed) >> 40) / static_cast<double>(1 << 24);
  return static_cast<float>(u * 0.16 - 0.08);
}
inline OwnedTensor MakeOwned(vt::DType dt, std::vector<int64_t> shape,
                             uint64_t seed) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  if (dt == vt::DType::kBF16) {
    t.bytes.resize(static_cast<size_t>(n) * 2);
    auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i)
      p[i] = vt::F32ToBF16(RandV(seed + static_cast<uint64_t>(i)));
  } else {
    t.bytes.resize(static_cast<size_t>(n) * 4);
    auto* p = reinterpret_cast<float*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) p[i] = RandV(seed + static_cast<uint64_t>(i));
  }
  return t;
}

constexpr int kVocab = 24;  // == the tiny BPE fixture's assigned ids (0..23).

inline HfConfig MakeSyntheticConfig(int max_model_len) {
  HfConfig c;
  c.model_type = "qwen3_5_moe_text";
  c.architectures = {"Qwen3_5MoeForConditionalGeneration"};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;  // [LA, LA, LA, FA]
  c.vocab_size = kVocab;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention",
                   "full_attention"};
  c.num_experts = 4;
  c.num_experts_per_tok = 2;
  c.moe_intermediate_size = 16;
  c.shared_expert_intermediate_size = 16;
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 4;
  c.linear_key_head_dim = 8;
  c.linear_value_head_dim = 8;
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 4;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = max_model_len;
  c.raw = nlohmann::json::object();  // no eos => runs to max_tokens.
  return c;
}

inline vllm::MoeBlockWeights MakeMoe(const HfConfig& c, uint64_t s) {
  vllm::MoeBlockWeights m;
  const int64_t H = c.hidden_size, E = c.num_experts, I = c.moe_intermediate_size,
                Is = c.shared_expert_intermediate_size;
  m.router_gate = MakeOwned(vt::DType::kBF16, {H, E}, s + 1);
  m.shared_gate = MakeOwned(vt::DType::kBF16, {H, 1}, s + 2);
  for (int64_t e = 0; e < E; ++e) {
    m.expert_gate.push_back(MakeOwned(vt::DType::kBF16, {H, I}, s + 100 + e * 7));
    m.expert_up.push_back(MakeOwned(vt::DType::kBF16, {H, I}, s + 200 + e * 7));
    m.expert_down.push_back(MakeOwned(vt::DType::kBF16, {I, H}, s + 300 + e * 7));
  }
  m.shared_gate_proj = MakeOwned(vt::DType::kBF16, {H, Is}, s + 3);
  m.shared_up_proj = MakeOwned(vt::DType::kBF16, {H, Is}, s + 4);
  m.shared_down_proj = MakeOwned(vt::DType::kBF16, {Is, H}, s + 5);
  return m;
}

inline vllm::Qwen3_5MoeWeights MakeSyntheticWeights(const HfConfig& c) {
  vllm::Qwen3_5MoeWeights w;
  const int64_t H = c.hidden_size, V = c.vocab_size;
  const int64_t Hq = c.num_attention_heads, Hkv = c.num_key_value_heads,
                Dh = c.head_dim;
  const int64_t Hk = c.linear_num_key_heads, Hv = c.linear_num_value_heads,
                Dk = c.linear_key_head_dim, Dv = c.linear_value_head_dim,
                Kw = c.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk, value_dim = Hv * Dv,
                conv_dim = 2 * key_dim + value_dim;
  w.embed_tokens = MakeOwned(vt::DType::kBF16, {V, H}, 11);
  w.final_norm = MakeOwned(vt::DType::kBF16, {H}, 12);
  w.lm_head = MakeOwned(vt::DType::kBF16, {H, V}, 13);
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const uint64_t s = 1000 + static_cast<uint64_t>(l) * 5000;
    vllm::Qwen3_5MoeLayerWeights lw;
    lw.is_linear_attention =
        (c.layer_types[static_cast<size_t>(l)] == "linear_attention");
    lw.input_layernorm = MakeOwned(vt::DType::kBF16, {H}, s + 1);
    lw.post_attention_layernorm = MakeOwned(vt::DType::kBF16, {H}, s + 2);
    if (lw.is_linear_attention) {
      lw.gdn.in_proj_qkv = MakeOwned(vt::DType::kBF16, {H, conv_dim}, s + 10);
      lw.gdn.in_proj_z = MakeOwned(vt::DType::kBF16, {H, value_dim}, s + 20);
      lw.gdn.in_proj_b = MakeOwned(vt::DType::kBF16, {H, Hv}, s + 30);
      lw.gdn.in_proj_a = MakeOwned(vt::DType::kBF16, {H, Hv}, s + 40);
      lw.gdn.conv1d_weight = MakeOwned(vt::DType::kBF16, {conv_dim, Kw}, s + 50);
      lw.gdn.a_log = MakeOwned(vt::DType::kF32, {Hv}, s + 60);
      lw.gdn.dt_bias = MakeOwned(vt::DType::kF32, {Hv}, s + 70);
      lw.gdn.norm_weight = MakeOwned(vt::DType::kBF16, {Dv}, s + 80);
      lw.gdn.out_proj = MakeOwned(vt::DType::kBF16, {value_dim, H}, s + 90);
    } else {
      lw.attn.q_proj = MakeOwned(vt::DType::kBF16, {H, 2 * Hq * Dh}, s + 10);
      lw.attn.k_proj = MakeOwned(vt::DType::kBF16, {H, Hkv * Dh}, s + 20);
      lw.attn.v_proj = MakeOwned(vt::DType::kBF16, {H, Hkv * Dh}, s + 30);
      lw.attn.o_proj = MakeOwned(vt::DType::kBF16, {Hq * Dh, H}, s + 40);
      lw.attn.q_norm = MakeOwned(vt::DType::kBF16, {Dh}, s + 50);
      lw.attn.k_norm = MakeOwned(vt::DType::kBF16, {Dh}, s + 60);
    }
    lw.moe = MakeMoe(c, s + 500);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// The tiny oracle-verified BPE fixture (ids 0..23, no holes) from
// tests/vllm/v1/test_llm_engine.cpp: "hello"=13, " world"=17, "1"=8, "2"=9, ...
inline tok::Tokenizer BuildSyntheticTokenizer() {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_bench_tok_" + std::to_string(counter++) + ".json"))
          .string();
  nlohmann::json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = nlohmann::json::array(
      {{{"id", 19}, {"content", "<|end|>"}, {"special", true}},
       {{"id", 20}, {"content", "<tool>"}, {"special", false}},
       {{"id", 21}, {"content", "<|end|>of"}, {"special", true}}});
  doc["normalizer"] = nullptr;
  doc["pre_tokenizer"] = {
      {"type", "Sequence"},
      {"pretokenizers",
       nlohmann::json::array(
           {{{"type", "Split"},
             {"pattern",
              {{"Regex",
                R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"}}},
             {"behavior", "Isolated"},
             {"invert", false}},
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"trim_offsets", false},
             {"use_regex", false}}})}};
  // Keep a real TemplateProcessing distinction in the benchmark fixture. The
  // production string path applies this post-processor through InputProcessor;
  // the pretokenized path must therefore use EncodeWithSpecialTokens too.
  doc["post_processor"] = nlohmann::json::parse(R"json({
    "type": "TemplateProcessing",
    "single": [
      {"SpecialToken": {"id": "<tool>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}}
    ],
    "pair": [],
    "special_tokens": {
      "<tool>": {"id": "<tool>", "ids": [20], "tokens": ["<tool>"]}
    }
  })json");
  nlohmann::json vocab = {
      {"h", 0},   {"e", 1},   {"l", 2},    {"o", 3},     {"w", 4},
      {"r", 5},   {"d", 6},   {"Ġ", 7}, {"1", 8},   {"2", 9},
      {"ll", 10}, {"he", 11}, {"llo", 12}, {"hello", 13}, {"Ġw", 14},
      {"or", 15}, {"orld", 16}, {"Ġworld", 17}, {"ld", 18}};
  vocab[tok::MapBytesToUnicode("\xF0\x9F")] = 22;
  vocab[tok::MapBytesToUnicode("\x8C\x8D")] = 23;
  doc["model"] = {
      {"type", "BPE"},
      {"ignore_merges", false},
      {"vocab", vocab},
      {"merges",
       nlohmann::json::array(
           {nlohmann::json::array({"l", "l"}), nlohmann::json::array({"h", "e"}),
            nlohmann::json::array({"ll", "o"}),
            nlohmann::json::array({"he", "llo"}),
            nlohmann::json::array({"Ġ", "w"}),
            nlohmann::json::array({"o", "r"}), nlohmann::json::array({"l", "d"}),
            nlohmann::json::array({"or", "ld"}),
            nlohmann::json::array({"Ġw", "orld"})})}};
  std::ofstream(path, std::ios::binary) << doc.dump();
  tok::Tokenizer t = tok::Tokenizer::FromHfJson(path);
  std::remove(path.c_str());
  return t;
}

// Build a prompt string that tokenizes to ~target tokens under `tok`. The
// generated-workload mode intentionally remains human-readable rather than
// sampling arbitrary IDs; default admission pre-encodes this string before the
// clock and the rollback tokenizes the same string inside add_request. The
// harness reports the measured tokenized counts, so throughput stays honest
// regardless of the small over/undershoot.
inline std::string BuildPrompt(const tok::Tokenizer& t, int target,
                               uint64_t seed) {
  // NOTE: for the SYNTHETIC engine, only bytes in the tiny fixture's alphabet
  // (h e l o w r d and digits 1 2) are encodable; a real tokenizer accepts any.
  // These filler words stay within that alphabet so both paths share this code.
  static const char* kWords[] = {"hello", "world", "1", "2"};
  constexpr size_t kNumWords = sizeof(kWords) / sizeof(kWords[0]);
  std::mt19937_64 rng(seed);
  std::string p;
  // Append a few words at a time, re-checking length, to keep this ~linear.
  while (static_cast<int>(t.Encode(p).size()) < target) {
    for (int k = 0; k < 4 && static_cast<int>(t.Encode(p).size()) < target;
         ++k) {
      if (!p.empty()) p += ' ';
      p += kWords[rng() % kNumWords];
    }
  }
  return p;
}

inline std::vector<std::string> LoadShareGptPrompts(const std::string& path,
                                                    int num_prompts) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open dataset: " + path);
  nlohmann::json data;
  in >> data;
  if (!data.is_array())
    throw std::runtime_error("ShareGPT dataset must be a JSON array: " + path);

  std::vector<std::string> prompts;
  prompts.reserve(static_cast<size_t>(num_prompts));
  for (const auto& entry : data) {
    if (!entry.contains("conversations") ||
        !entry["conversations"].is_array() ||
        entry["conversations"].empty())
      continue;
    const auto& turn = entry["conversations"][0];
    if (!turn.contains("value") || !turn["value"].is_string()) continue;
    prompts.push_back(turn["value"].get<std::string>());
    if (static_cast<int>(prompts.size()) == num_prompts) break;
  }
  if (static_cast<int>(prompts.size()) != num_prompts) {
    throw std::runtime_error("dataset has fewer valid prompts than --num-prompts");
  }
  return prompts;
}

}  // namespace detail

// Build a SamplingParams for one bench request (greedy unless temperature>0).
inline SamplingParams MakeSampling(const BenchConfig& cfg, int req_index) {
  SamplingParams sp;
  sp.temperature = cfg.temperature;  // <= 0 => greedy.
  sp.max_tokens = cfg.output_len;
  // #2759: the EOS polarity is now a SETTING, and its default is the fixed-length
  // workload this line used to hardcode: generate EXACTLY output_len tokens
  // (never stop early on EOS), so throughput/latency are measured on the
  // intended token budget and match `vllm bench serve --ignore-eos`
  // apples-to-apples, and the documented "greedy => exactly O" contract holds.
  // `--no-ignore-eos` gives up that contract deliberately, which is what a
  // chat-completions comparator's protocol requires
  // (`vllm/benchmarks/serve.py:1757-1762`, default false upstream).
  sp.ignore_eos = cfg.ignore_eos;
  sp.output_kind = RequestOutputKind::kDelta;  // observe TTFT/ITL like a client.
  if (cfg.temperature > 0.0) {
    sp.seed = static_cast<int64_t>(cfg.seed + static_cast<uint64_t>(req_index));
  }
  return sp;
}

// #2759: resolve the chat-template SOURCE the way the server does, and say
// where it came from. `--chat-template` names a file OR carries the template
// itself in single-line form (`vllm/entrypoints/launchers/cli_args.py:80-82`);
// with neither, the model's own template is loaded through
// `LoadChatTemplateForModel`, the identical selection path `server_main.cpp`
// uses (tokenizer_config.json first, then a `.gguf`'s `tokenizer.chat_template`
// metadata).
//
// Every failure here THROWS rather than falling back to the role-join prompt
// the server falls back to. A served model must answer something. A benchmark
// that silently measured a different prompt shape than the one requested is
// worse than one that stops, because the number it prints is publishable and
// wrong.
inline std::string ResolveBenchChatTemplate(const BenchConfig& cfg,
                                            std::string& source) {
  if (!cfg.chat_template.empty()) {
    std::error_code ec;
    if (std::filesystem::is_regular_file(cfg.chat_template, ec)) {
      std::ifstream in(cfg.chat_template, std::ios::binary);
      if (!in) {
        throw std::runtime_error("cannot read the chat template file: " +
                                 cfg.chat_template);
      }
      const std::string body((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
      source = "--chat-template file " + cfg.chat_template;
      return body;
    }
    source = "--chat-template literal";
    return cfg.chat_template;
  }
  if (cfg.model_path.empty()) {
    throw std::runtime_error(
        "--no-skip-chat-template needs a chat template, and the synthetic "
        "engine (no --model) ships none: pass --chat-template <file or "
        "single-line template>, or --model a checkpoint that carries one");
  }
  const std::filesystem::path dir(cfg.model_path);
  std::string template_source;
  std::string body = vllm::entrypoints::LoadChatTemplateForModel(
      (dir / "tokenizer_config.json").string(), cfg.model_path,
      template_source);
  source = template_source;
  return body;
}

// ─────────────────────────────── The harness ──────────────────────────────────
// Creates the engine (synthetic if cfg.model_path is empty, else loaded from the
// dir/.gguf), builds N prompts, then drives the production AsyncLLM frontend,
// admitting up to C requests at a time until all N finish — timing everything
// with steady_clock. Async scheduling OFF still uses AsyncLLM with a depth-1
// core queue, so the ON/OFF comparison changes only scheduler/runner behavior.
// Returns the aggregated metrics.
inline BenchResult RunBench(const BenchConfig& cfg) {
  using Clock = std::chrono::steady_clock;

  if (cfg.output_wait == OutputWaitMode::kBlockingC1 && cfg.concurrency != 1) {
    throw std::invalid_argument(
        "benchmark output wait 'blocking-c1' requires --concurrency 1");
  }
  // #2759: refuse a flag that would be silently ignored. Accepting
  // `--chat-template` while the template is skipped would reproduce, inside the
  // fix, the exact defect the issue names: a setting that parses, is dropped,
  // and leaves a report that looks like it was honoured.
  if (cfg.skip_chat_template && !cfg.chat_template.empty()) {
    throw std::runtime_error(
        "--chat-template was given but the chat template is skipped; pass "
        "--no-skip-chat-template to apply it");
  }

  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded;
  std::vector<std::string> prompts;
  prompts.reserve(static_cast<size_t>(cfg.num_prompts));
  if (!cfg.dataset_path.empty()) {
    prompts = detail::LoadShareGptPrompts(cfg.dataset_path, cfg.num_prompts);
  }

  if (cfg.model_path.empty()) {
    // Synthetic: build tokenizer first (to size the engine + measure prompts),
    // then move it into the LoadedEngine.
    tok::Tokenizer tok = detail::BuildSyntheticTokenizer();
    int max_prompt = 1;
    if (prompts.empty()) {
      for (int i = 0; i < cfg.num_prompts; ++i) {
        prompts.push_back(detail::BuildPrompt(
            tok, cfg.input_len, cfg.seed + static_cast<uint64_t>(i)));
      }
    }
    for (const std::string& prompt : prompts) {
      max_prompt =
          std::max(max_prompt, static_cast<int>(tok.Encode(prompt).size()));
    }
    const int seq_budget = max_prompt + cfg.output_len + 4;
    vllm::entrypoints::EngineParams params;
    // Unified block (hybrid-KV constraint): one block size for the GDN and the
    // full-attention groups, still one block per sequence. ALIGNED UP to a
    // multiple of 16 — the kernel block size every registered attention
    // backend declares (CPU_ATTN and FLASH_ATTN both declare {16}), so
    // supports_block_size accepts it; the raw seq_budget is arbitrary (e.g.
    // 36) and was refused out of engine construction ("block_size not
    // supported") since CPU_ATTN was registered (9ecaf1bb3).
    params.block_size = (seq_budget + 15) / 16 * 16;
    params.max_model_len = seq_budget;
    params.max_num_seqs = std::max(cfg.concurrency, 1);
    params.num_blocks = std::max(cfg.concurrency * 4, 16);
    // KV-FP8 W7 (#2619): the synthetic arm takes the flag too. Not symmetry --
    // the synthetic model is a Qwen3.5-MoE, whose attention block W3 routed, so
    // this is what lets `--kv-cache-dtype fp8` allocate 1-byte pages and decode
    // through them with no checkpoint and no device. The direct constructor
    // takes the value verbatim: there is no model directory to resolve against.
    params.kv_cache_dtype = cfg.kv_cache_dtype;
    HfConfig c = detail::MakeSyntheticConfig(seq_budget);
    vllm::Qwen3_5MoeWeights w = detail::MakeSyntheticWeights(c);
    loaded = std::make_unique<vllm::entrypoints::LoadedEngine>(
        std::move(c), std::move(w), std::move(tok), params);
  } else {
    vllm::entrypoints::EngineParams params;
    params.max_num_seqs = std::max(cfg.concurrency, 1);
    // Real checkpoint: max_model_len comes from config; blocks sized for C
    // concurrent (input+output)-long sequences at the default block size.
    params.block_size = 32;
    const int seq_blocks = (cfg.input_len + cfg.output_len + 31) / 32 + 1;
    params.num_blocks = cfg.num_blocks > 0
                            ? cfg.num_blocks
                            : std::max(cfg.concurrency * seq_blocks * 2, 256);
    // KV-FP8 W7 (#2619): `--kv-cache-dtype` reaches the engine. FromModelDir
    // resolves it against the checkpoint's own `kv_cache_quant_algo` before any
    // consumer reads it, and an explicit value always wins (torch_utils.py:
    // 380-381), so the harness hands over the raw string and reads the OUTCOME
    // back out of kv_cache_config() below.
    params.kv_cache_dtype = cfg.kv_cache_dtype;
    // Chunked-prefill per-step budget (0 => engine bounded default).
    params.max_num_batched_tokens = cfg.max_num_batched_tokens;
    // Speculative decoding (MTP): OFF unless a config JSON is supplied. When
    // set, resolve the CLI method here; FromModelDir loads the mtp.* head and
    // wires the verify/propose loop (spec-OFF path is byte-unchanged).
    if (!cfg.speculative_config.empty()) {
      params.speculative_config =
          vllm::ParseSpeculativeConfigJson(cfg.speculative_config);
    }
    loaded = vllm::entrypoints::LoadedEngine::FromModelDir(cfg.model_path, params);
    if (prompts.empty()) {
      for (int i = 0; i < cfg.num_prompts; ++i) {
        prompts.push_back(detail::BuildPrompt(
            loaded->tokenizer(), cfg.input_len,
            cfg.seed + static_cast<uint64_t>(i)));
      }
    }
  }

  // #2759: the chat-template render, in ONE place and BEFORE any tokenization,
  // so both admission paths -- the pretokenized default and the
  // `VT_BENCH_PRETOKENIZE=0` timed-string path -- carry the identical rendered
  // prompt and the prompt-token accounting reflects it. One user message,
  // `add_generation_prompt=true`, no tools:
  // `vllm/benchmarks/datasets/datasets.py:2610-2615` exactly. The
  // per-request kwargs object is empty because a benchmark has no per-request
  // kwargs; `enable_thinking` rides the SERVER-level default, which is where
  // `server_main.cpp` puts it too.
  std::string resolved_chat_template = "skipped (raw prompt)";
  const nlohmann::ordered_json chat_template_kwargs =
      vllm::entrypoints::DefaultChatTemplateKwargs(cfg.enable_thinking);
  if (!cfg.skip_chat_template) {
    std::string source;
    const std::string template_str = ResolveBenchChatTemplate(cfg, source);
    const tok::Tokenizer& tokenizer = loaded->tokenizer();
    const std::string bos =
        tokenizer.BosId() >= 0 ? tokenizer.Decode({tokenizer.BosId()}) : "";
    const std::string eos =
        tokenizer.EosId() >= 0 ? tokenizer.Decode({tokenizer.EosId()}) : "";
    const vllm::entrypoints::openai::ChatPromptFn prompt_fn =
        vllm::entrypoints::MakeChatTemplatePromptFn(template_str, bos, eos,
                                                    chat_template_kwargs);
    for (std::string& prompt : prompts) {
      prompt = prompt_fn(
          {vllm::entrypoints::openai::ChatMessage{"user", prompt}},
          /*add_generation_prompt=*/true, /*tools=*/{},
          nlohmann::ordered_json::object());
    }
    resolved_chat_template = "applied (" +
                             std::to_string(template_str.size()) +
                             " chars) from " + source;
  }

  vllm::v1::AsyncLLM& engine = loaded->async_engine();

  // Match the pinned vLLM comparison frontend: tokenize the complete workload
  // before t0, preserving submission order and the string path's special-token
  // processing, then move those IDs through AsyncLLM's TokensPrompt overload.
  // Exact `VT_BENCH_PRETOKENIZE=0` retains the previous timed string overload
  // and does not allocate this precomputed workload.
  const char* const pretokenize_env = std::getenv("VT_BENCH_PRETOKENIZE");
  const bool pretokenized_admission =
      ResolveBenchPretokenizedAdmission(pretokenize_env);
  auto [pretokenized_prompts, t0] = PretokenizeBenchPromptsThenStartClock(
      pretokenized_admission, loaded->tokenizer(), prompts,
      []() { return Clock::now(); });

  std::map<std::string, RequestRecord> records;
  std::map<std::string, vllm::v1::AsyncRequest> active;
  auto now_s = [&]() {
    return std::chrono::duration<double>(Clock::now() - t0).count();
  };

  int next = 0;       // next prompt index to submit.
  int in_flight = 0;  // requests currently admitted + unfinished.
  int done = 0;
  int64_t blocking_wait_calls = 0;

  auto admit = [&]() {
    const int available = cfg.concurrency - in_flight;
    const int wave_size = std::min(cfg.num_prompts - next, available);
    if (wave_size <= 0) return;
    const int wave_begin = next;

    std::vector<vllm::v1::AsyncRequest> admitted =
        DispatchBenchPromptWaveAdmission(
            engine, pretokenize_env, static_cast<std::size_t>(wave_size),
            [&](std::size_t offset) {
              const int request_index =
                  wave_begin + static_cast<int>(offset);
              RequestRecord record;
              record.arrival_s = now_s();
              records[std::to_string(request_index)] = std::move(record);
            },
            [&]() {
              std::vector<vllm::v1::AsyncTokensRequestInput> wave;
              wave.reserve(static_cast<std::size_t>(wave_size));
              for (int offset = 0; offset < wave_size; ++offset) {
                const int request_index = wave_begin + offset;
                const std::size_t prompt_index =
                    static_cast<std::size_t>(request_index);
                wave.push_back(vllm::v1::AsyncTokensRequestInput{
                    std::to_string(request_index),
                    std::move(pretokenized_prompts[prompt_index]),
                    MakeSampling(cfg, request_index), 0});
              }
              return wave;
            },
            [&]() {
              std::vector<vllm::v1::AsyncStringRequestInput> wave;
              wave.reserve(static_cast<std::size_t>(wave_size));
              for (int offset = 0; offset < wave_size; ++offset) {
                const int request_index = wave_begin + offset;
                const std::size_t prompt_index =
                    static_cast<std::size_t>(request_index);
                wave.push_back(vllm::v1::AsyncStringRequestInput{
                    std::to_string(request_index), prompts[prompt_index],
                    MakeSampling(cfg, request_index), 0});
              }
              return wave;
            });
    if (admitted.size() != static_cast<std::size_t>(wave_size)) {
      throw std::runtime_error("AsyncLLM admitted an incomplete bench wave");
    }
    for (vllm::v1::AsyncRequest& request : admitted) {
      active.emplace(request.request_id, std::move(request));
    }
    next += wave_size;
    in_flight += wave_size;
  };

  auto consume_output = [&](auto it, RequestOutput out) {
    RequestRecord& rec = records[out.request_id];
    if (rec.prompt_tokens == 0 && !out.prompt_token_ids.empty()) {
      rec.prompt_tokens = static_cast<int>(out.prompt_token_ids.size());
      rec.prompt_token_ids = out.prompt_token_ids;
    }
    if (!out.outputs.empty() && !out.outputs[0].token_ids.empty()) {
      const double t = now_s();
      const int n_new = static_cast<int>(out.outputs[0].token_ids.size());
      if (rec.first_token_s < 0.0) {
        rec.first_token_s = t;  // TTFT reference.
      } else {
        rec.itls.push_back(t - rec.last_token_s);  // one ITL per chunk.
      }
      rec.last_token_s = t;
      rec.output_tokens += n_new;
      rec.output_token_ids.insert(rec.output_token_ids.end(),
                                  out.outputs[0].token_ids.begin(),
                                  out.outputs[0].token_ids.end());
    }
    if (out.finished && !rec.finished) {
      rec.finished = true;
      rec.completion_s = now_s();
      --in_flight;
      ++done;
      return active.erase(it);
    }
    return std::next(it);
  };

  admit();
#ifdef VLLM_BENCH_TEST_OUTPUT_WAIT_TRACE
  if (cfg.output_wait_test_probe.seed_first_collector) {
    if (active.size() != 1 || active.begin()->second.collector == nullptr) {
      throw std::logic_error(
          "nowait runtime probe requires exactly one active collector");
    }
    RequestOutput seeded;
    seeded.request_id = active.begin()->first;
    active.begin()->second.collector->put(std::move(seeded));
  }
#endif
  while (done < cfg.num_prompts) {
    bool observed_output = false;
    for (auto it = active.begin(); it != active.end();) {
      // Keep this evaluation unconditional: test instrumentation observes the
      // already-evaluated value below and cannot replace the production call.
      std::optional<RequestOutput> ready =
          engine.get_output_nowait(it->second);
#ifdef VLLM_BENCH_TEST_OUTPUT_WAIT_TRACE
      if (cfg.output_wait_test_trace != nullptr) {
        cfg.output_wait_test_trace->push_back(
            ready.has_value() ? OutputWaitTestEvent::kNowaitReady
                              : OutputWaitTestEvent::kNowaitEmpty);
      }
      if (cfg.output_wait_test_probe.stop_after_first_nowait) {
        throw std::logic_error(
            ready.has_value() ? "nowait runtime probe observed seeded output"
                              : "nowait runtime probe missed seeded output");
      }
#endif
      if (!ready.has_value()) {
        ++it;
        continue;
      }
      observed_output = true;
      it = consume_output(it, std::move(*ready));
    }
    if (!observed_output) {
      if (cfg.output_wait == OutputWaitMode::kBlockingC1) {
        // Validation above guarantees the sole active collector is also the
        // sole admitted request. The scan already performed the required
        // nowait probe, matching AsyncLLM::generate before this blocking wait.
        if (active.size() != 1) {
          throw std::logic_error(
              "blocking-c1 benchmark must have exactly one active request");
        }
        auto it = active.begin();
#ifdef VLLM_BENCH_TEST_OUTPUT_WAIT_TRACE
        if (cfg.output_wait_test_trace != nullptr) {
          cfg.output_wait_test_trace->push_back(
              OutputWaitTestEvent::kBlockingWait);
        }
#endif
        ++blocking_wait_calls;
        it = consume_output(it, engine.get_output(it->second));
        (void)it;
      } else {
        // Exact historical policy: yield rather than block on an arbitrary
        // request, which could delay a ready peer at concurrency greater than 1.
        std::this_thread::yield();
      }
    }
    admit();  // keep C in flight as requests finish.
  }

  const double dur_s = now_s();

  // ── Reduce to aggregate metrics (serve.py:726-758 semantics). ────────────────
  std::vector<double> ttfts, tpots, itls, e2els;
  int64_t total_in = 0, total_out = 0;
  double sum_prefill = 0.0, sum_decode = 0.0;
  int64_t decode_tokens = 0;
  for (const auto& kv : records) {
    const RequestRecord& r = kv.second;
    total_in += r.prompt_tokens;
    total_out += r.output_tokens;
    const double ttft = std::max(0.0, r.first_token_s - r.arrival_s);
    const double e2el = std::max(0.0, r.completion_s - r.arrival_s);
    ttfts.push_back(ttft);
    e2els.push_back(e2el);
    if (r.output_tokens > 1) {
      tpots.push_back((e2el - ttft) / static_cast<double>(r.output_tokens - 1));
      sum_decode += (e2el - ttft);
      decode_tokens += (r.output_tokens - 1);
    }
    sum_prefill += ttft;
    for (double x : r.itls) itls.push_back(x);
  }

  BenchResult res;
  res.async_frontend = true;
  res.async_scheduling_enabled = loaded->async_scheduling_enabled();
  res.pretokenized_admission = pretokenized_admission;
  res.max_concurrent_batches = loaded->max_concurrent_batches();
  res.output_wait = cfg.output_wait;
  res.blocking_wait_calls = blocking_wait_calls;
  // KV-FP8 W7 (#2619): the resolved KV dtype is read back out of the config the
  // loader sized, so the report line IS the allocation (see the field comment).
  res.resolved_kv_cache_dtype = ResolvedKvCacheDTypeName(loaded->kv_cache_config());
  // #2759: read the EOS setting back out of `MakeSampling`, the one function
  // both admission paths call to build a request's `SamplingParams`. Reading
  // `cfg.ignore_eos` here instead would make the report a transcription of the
  // request and blind to the pass-through being deleted.
  res.resolved_ignore_eos = MakeSampling(cfg, 0).ignore_eos;
  res.resolved_chat_template = resolved_chat_template;
  res.resolved_chat_template_kwargs = chat_template_kwargs.dump();
  res.prompt_token_ids.resize(static_cast<size_t>(cfg.num_prompts));
  res.output_token_ids.resize(static_cast<size_t>(cfg.num_prompts));
  res.completed = done;
  res.duration_s = dur_s;
  res.total_input = total_in;
  res.total_output = total_out;
  res.request_throughput = dur_s > 0 ? static_cast<double>(done) / dur_s : 0.0;
  res.output_throughput =
      dur_s > 0 ? static_cast<double>(total_out) / dur_s : 0.0;
  res.input_throughput = dur_s > 0 ? static_cast<double>(total_in) / dur_s : 0.0;
  res.total_token_throughput =
      dur_s > 0 ? static_cast<double>(total_in + total_out) / dur_s : 0.0;
  // sum_prefill is the sum of per-request TTFTs, i.e. the time actually spent
  // producing the input tokens. It was already being accumulated and then
  // discarded.
  res.prefill_throughput =
      sum_prefill > 0 ? static_cast<double>(total_in) / sum_prefill : 0.0;
  const double mean_tpot = Mean(tpots);
  res.mean_per_stream_decode = mean_tpot > 0 ? 1.0 / mean_tpot : 0.0;
  (void)sum_decode;
  (void)decode_tokens;

  res.mean_ttft_ms = Mean(ttfts) * 1000.0;
  res.median_ttft_ms = Percentile(ttfts, 50) * 1000.0;
  res.p99_ttft_ms = Percentile(ttfts, 99) * 1000.0;
  res.mean_tpot_ms = mean_tpot * 1000.0;
  res.median_tpot_ms = Percentile(tpots, 50) * 1000.0;
  res.p99_tpot_ms = Percentile(tpots, 99) * 1000.0;
  res.mean_itl_ms = Mean(itls) * 1000.0;
  res.median_itl_ms = Percentile(itls, 50) * 1000.0;
  res.p99_itl_ms = Percentile(itls, 99) * 1000.0;
  res.mean_e2el_ms = Mean(e2els) * 1000.0;
  res.median_e2el_ms = Percentile(e2els, 50) * 1000.0;
  res.p99_e2el_ms = Percentile(e2els, 99) * 1000.0;
  for (const auto& kv : records) {
    const size_t request_index = static_cast<size_t>(std::stoul(kv.first));
    if (request_index >= res.output_token_ids.size()) {
      throw std::runtime_error("benchmark request id is out of range");
    }
    res.prompt_token_ids[request_index] = kv.second.prompt_token_ids;
    res.output_token_ids[request_index] = kv.second.output_token_ids;
  }
  // Speculative-decoding acceptance telemetry (real-checkpoint spec-ON only;
  // the synthetic engine has no draft loop).
  if (!cfg.model_path.empty() && !cfg.speculative_config.empty()) {
    res.spec_on = true;
    res.spec_proposed = loaded->runner().spec_drafts_proposed();
    res.spec_accepted = loaded->runner().spec_drafts_accepted();
  }
  return res;
}

inline void WriteOutputTokenIds(const std::string& path,
                                const BenchResult& result) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write output token IDs: " + path);
  out << nlohmann::json(result.output_token_ids).dump() << '\n';
}

inline nlohmann::json ResultJson(const BenchConfig& cfg,
                                 const BenchResult& result) {
  return {{"output_wait", OutputWaitModeName(result.output_wait)},
          {"blocking_wait_calls", result.blocking_wait_calls},
          {"concurrency", cfg.concurrency},
          {"completed", result.completed},
          {"duration_s", result.duration_s},
          {"total_input", result.total_input},
          {"total_output", result.total_output},
          {"request_throughput", result.request_throughput},
          {"output_throughput", result.output_throughput},
          {"prefill_throughput", result.prefill_throughput},
          {"mean_per_stream_decode", result.mean_per_stream_decode},
          {"mean_ttft_ms", result.mean_ttft_ms},
          {"mean_tpot_ms", result.mean_tpot_ms},
          {"mean_e2el_ms", result.mean_e2el_ms},
          {"output_token_ids", result.output_token_ids}};
}

inline void WriteResultJson(const std::string& path, const BenchConfig& cfg,
                            const BenchResult& result) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write benchmark JSON: " + path);
  out << ResultJson(cfg, result).dump() << '\n';
}

// Print the summary table, mirroring serve.py's "Serving Benchmark Result"
// block + our prefill/decode split. Fixed-width columns like serve.py
// ("{:<40} {:<10.2f}").
inline void PrintReport(const BenchConfig& cfg, const BenchResult& r,
                        std::FILE* out) {
  auto line_i = [&](const char* label, long long v) {
    std::fprintf(out, "%-42s %-12lld\n", label, v);
  };
  auto line_f = [&](const char* label, double v) {
    std::fprintf(out, "%-42s %-12.2f\n", label, v);
  };
  auto line_s = [&](const char* label, const char* v) {
    std::fprintf(out, "%-42s %s\n", label, v);
  };
  auto sep = [&](const char* title) {
    std::fprintf(out, "%.*s %s %.*s\n", 8,
                 "----------------------------------------", title, 8,
                 "----------------------------------------");
  };

  std::fprintf(out, "\n============= vllm.cpp Benchmark Result =============\n");
  std::fprintf(out, "%-42s %-12s\n", "Engine frontend:",
               r.async_frontend ? "AsyncLLM" : "LLMEngine");
  std::fprintf(out, "%-42s %-12s\n", "Output wait:",
               OutputWaitModeName(r.output_wait));
  line_i("Blocking wait calls:",
         static_cast<long long>(r.blocking_wait_calls));
  line_i("Async scheduling enabled:", r.async_scheduling_enabled ? 1 : 0);
  line_i("Pretokenized prompt admission:", r.pretokenized_admission ? 1 : 0);
  // KV-FP8 W7 (#2619): a published number states the KV dtype it was measured
  // on. Two lines, because they answer different questions and can disagree:
  // the first is what the operator asked for, the second is what the loader
  // sized -- and a checkpoint declaring kv_cache_quant_algo makes the second
  // fp8 while the first still reads "auto".
  line_s("KV cache dtype (requested):", cfg.kv_cache_dtype.c_str());
  line_s("KV cache dtype (resolved storage):", r.resolved_kv_cache_dtype.c_str());
  // #2759: a benchmark that does not say whether it stopped on EOS is not
  // reproducible, and every figure this harness published before now was taken
  // with EOS suppressed without saying so. Two lines for the same reason the KV
  // pair above has two: the first is what the operator asked for, the second is
  // read back out of the `SamplingParams` the admission path builds, and only
  // the second can disagree with the flag.
  line_i("Ignore EOS (requested):", cfg.ignore_eos ? 1 : 0);
  line_i("Ignore EOS (resolved sampling):", r.resolved_ignore_eos ? 1 : 0);
  line_s("Chat template:", r.resolved_chat_template.c_str());
  line_s("Chat template kwargs:", r.resolved_chat_template_kwargs.c_str());
  line_i("Maximum concurrent batches:", r.max_concurrent_batches);
  line_i("Successful requests:", r.completed);
  line_i("Maximum request concurrency:", cfg.concurrency);
  line_f("Benchmark duration (s):", r.duration_s);
  line_i("Total input tokens:", static_cast<long long>(r.total_input));
  line_i("Total generated tokens:", static_cast<long long>(r.total_output));
  line_f("Request throughput (req/s):", r.request_throughput);
  line_f("Output token throughput (tok/s):", r.output_throughput);
  line_f("Total token throughput (tok/s):", r.total_token_throughput);
  sep("Time to First Token");
  line_f("Mean TTFT (ms):", r.mean_ttft_ms);
  line_f("Median TTFT (ms):", r.median_ttft_ms);
  line_f("P99 TTFT (ms):", r.p99_ttft_ms);
  sep("Time per Output Token (excl. 1st token)");
  line_f("Mean TPOT (ms):", r.mean_tpot_ms);
  line_f("Median TPOT (ms):", r.median_tpot_ms);
  line_f("P99 TPOT (ms):", r.p99_tpot_ms);
  sep("Inter-token Latency");
  line_f("Mean ITL (ms):", r.mean_itl_ms);
  line_f("Median ITL (ms):", r.median_itl_ms);
  line_f("P99 ITL (ms):", r.p99_itl_ms);
  sep("End-to-end Latency");
  line_f("Mean E2EL (ms):", r.mean_e2el_ms);
  line_f("Median E2EL (ms):", r.median_e2el_ms);
  line_f("P99 E2EL (ms):", r.p99_e2el_ms);
  sep("Prefill vs Decode split (gate #1)");
  line_f("Input token throughput (tok/s, over whole run):", r.input_throughput);
  line_f("Prefill token throughput (tok/s, in/TTFT):", r.prefill_throughput);
  line_f("Output (decode) token throughput (tok/s):", r.output_throughput);
  line_f("Mean per-stream decode rate (tok/s):", r.mean_per_stream_decode);
  if (r.spec_on) {
    sep("Speculative decoding (MTP)");
    line_i("Draft tokens proposed:", static_cast<long long>(r.spec_proposed));
    line_i("Draft tokens accepted:", static_cast<long long>(r.spec_accepted));
    line_f("Acceptance rate (accepted/proposed):",
           r.spec_proposed > 0
               ? static_cast<double>(r.spec_accepted) /
                     static_cast<double>(r.spec_proposed)
               : 0.0);
  }
  std::fprintf(out, "====================================================\n");
}

}  // namespace vllm::bench

#endif  // VLLM_EXAMPLES_BENCH_BENCH_CORE_H_
