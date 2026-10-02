// server: an OpenAI-compatible HTTP server over the vllm.cpp LLMEngine (M3.1
// Task 4). Loads a supported model (safetensors or GGUF weights + tokenizer + a KV-cache config →
// LLMEngine), constructs the OpenAI serving handlers (chat wired with the real
// chat template via MakeChatTemplatePromptFn(LoadChatTemplateFromConfig(...)))
// and serves /v1/completions, /v1/chat/completions, /v1/models, /health,
// /version.
//
//   server --model <dir> [--host 0.0.0.0] [--port 8000]
//          [--tokenizer-config <tokenizer_config.json>]
//          [--served-model-name <name>]
//          [--block-size N] [--num-blocks N] [--max-model-len N]
//          [--gpu-memory-utilization F] [--kv-cache-memory BYTES]
//          [--kv-cache-dtype auto|bfloat16|fp8|fp8_e4m3]
//          [--max-num-seqs N] [--max-num-batched-tokens N]
//          [--enable-force-include-usage]
//          [--[no-]enable-prefix-caching]
//          [--scheduling-policy fcfs|priority]
//          [--generation-config auto|vllm|<dir>]
//          [--tool-call-parser <name>|auto|none]
//          [--reasoning-parser <name>|auto|none]
//          [--kv-transfer-config '<json>'] [--offload-config '<json>']
//
// A directory holds config.json, tokenizer.json and supported safetensors
// shards. A supported GGUF file is also accepted and supplies model metadata
// plus embedded vocabulary. If --tokenizer-config is omitted for a directory it
// defaults to <dir>/tokenizer_config.json; when that file has no chat_template
// the chat endpoint falls back to the simple role-join prompt.
//
// NOTE: loading the real 35B checkpoint is a GPU/dgx concern; on the CPU CI box
// this binary is only built + smoke-tested against a synthetic engine (see
// tests/vllm/entrypoints/openai/test_api_server.cpp). The wiring below is the
// same either way.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <atomic>
#include <cerrno>
#include <functional>
#include <thread>
#if defined(_WIN32)
#include <process.h>
#endif
// DSR-ALLOW(ARCH-ONE-SURFACE): VT_BENCH_PROFILE_CONTROL is a build-option guard for the CUDA-graph-replay profiler, not a device fork; #189 moved it here verbatim from examples/server/main.cpp, which the DSR scanner never covered.
#if defined(VT_BENCH_PROFILE_CONTROL) && !defined(_WIN32)
#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>  // ::getpid below; guarded because MSVC has no such header
#include <thread>
#endif

#include "vllm.h"
#include "vllm/config/device.h"
#include "vllm/config/kv_transfer.h"
#include "vllm/config/multimodal.h"
#include "vllm/config/scheduler.h"
#include "vllm/entrypoints/chat_template.h"
#include "vllm/config/offload.h"
#include "vllm/http_transport_abi.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/transformers_utils/model_resolver.h"
#include <fstream>
#include "vllm/entrypoints/openai/server_main.h"
#include "vllm/entrypoints/openai/api_server.h"
#include "vllm/entrypoints/openai/chat_mm.h"
#include "vllm/entrypoints/openai/mm_chat_registry.h"
#include "vllm/entrypoints/openai/request_logger.h"
#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/config/generation.h"
#include "vllm/entrypoints/openai/serving_completion.h"
#include "vllm/entrypoints/openai/serving_models.h"
#include "vllm/v1/metrics/loggers.h"
#include "vllm/entrypoints/openai/reasoning_parsers/detect.h"
#include "vllm/entrypoints/openai/tool_parsers/detect.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/platform/console_shutdown.h"
#include "vllm/platform/process.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/transformers_utils/tokenizer_files.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/gliner2_ner.h"  // Gliner2NerInference (MODEL-GLINER25)
#include "vllm/model_executor/models/cua_s1_inference.h"  // CuaS1ScoreInference (MODEL-CUA-S1-FORMS)
#include "vllm/model_executor/models/laya_inference.h"  // LayaInference (MODEL-LAYA)
#include "vllm/model_executor/models/kev_inference.h"  // KevInference (MODEL-KEV)
#include "vllm/model_executor/models/nimble_inference.h"  // NimbleDecide (MODEL-NIMBLE)
#include "vllm/model_executor/models/tev1_inference.h"  // Tev1Decide (MODEL-TEV1)
#include "vllm/model_executor/models/clm_inference.h"  // ClmDecide (MODEL-CLM)
#include "vllm/model_executor/models/gliner25_decide_inference.h"  // Gliner25DecideInference (MODEL-GLINER25-DECIDE)
#include "vllm/model_executor/models/xor_inference.h"  // XorInference (MODEL-XOR)
#include "vllm/multimodal/minimax_h3_video.h"
#include "vllm/multimodal/parakeet_transcription.h"
#include "vllm/multimodal/video_engine.h"
#include "vllm/multimodal/speech_engine.h"
#include "vllm/model_executor/models/minimax_music3_speech.h"
#include "vllm/entrypoints/openai/speech_api.h"
#include "vllm/model_executor/models/minimax_h3.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/version.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/core.h"
#include "vllm/v1/engine/input_processor.h"
#include "vllm/v1/engine/llm_engine.h"
#include "vllm/v1/engine/output_processor.h"
#include "vllm/v1/executor/executor.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/kv_offload/kv_connector.h"
#include "vllm/v1/worker/gpu/runner.h"
#include "vt/backend.h"
// DSR-ALLOW(ARCH-ONE-SURFACE): VT_BENCH_PROFILE_CONTROL is a build-option guard for the CUDA-graph-replay profiler, not a device fork; #189 moved it here verbatim from examples/server/main.cpp, which the DSR scanner never covered.
#if defined(VT_BENCH_PROFILE_CONTROL) && !defined(_WIN32)
#include "vt/cuda/cuda_profiler_control.h"
#endif
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace {

namespace fs = std::filesystem;
using vllm::HfConfig;
using vllm::Qwen3_5MoeWeights;

fs::path NativeUtf8Path(const std::string& value) {
#if defined(_WIN32)
  const std::u8string utf8(
      reinterpret_cast<const char8_t*>(value.data()), value.size());
  return fs::path(utf8);
#else
  return fs::path(value);
#endif
}

std::string PathUtf8(const fs::path& path) {
#if defined(_WIN32)
  const std::u8string utf8 = path.u8string();
  return std::string(reinterpret_cast<const char*>(utf8.data()), utf8.size());
#else
  return path.string();
#endif
}

void SetEnvironment(const char* name, const char* value) {
#if defined(_WIN32)
  if (_putenv_s(name, value) != 0) {
    throw std::runtime_error(std::string("could not set environment variable ") +
                             name);
  }
#else
  if (setenv(name, value, /*overwrite=*/1) != 0) {
    throw std::runtime_error(std::string("could not set environment variable ") +
                             name);
  }
#endif
}

// DSR-ALLOW(ENG-RELEASE-WINDOWS): the profiler process-ID adapter is POSIX-only; #117 keeps native Windows on its CRT process boundary, not a device-dispatch fork.
#if defined(VT_BENCH_PROFILE_CONTROL) && !defined(_WIN32)
uint64_t CurrentProcessId() {
  return static_cast<uint64_t>(::getpid());
}
#endif

// Run an argv to completion and return its exit status — the ONE process
// spawn in the MiniMax-H3 path, and it lives HERE, in examples/, by the
// developer-ratified 2026-08-03 decision: the library (the
// MiniMaxH3VideoEngine seam behind /v1/videos) writes the artifacts and
// BUILDS this argv, and spawns nothing.
int RunFfmpegArgv(const std::vector<std::string>& args) {
  return vllm::platform::RunProcessArgv(args);
}

struct Args {
  std::string model_dir;
  // ENG-HF-MODEL-DOWNLOAD W4 (#1280): what the user TYPED after `--model`,
  // kept because the resolver replaces `model_dir` with a cache path and the
  // served model name still defaults to the name the user asked for, exactly
  // as vLLM's `served_model_name` defaults to `model` (`arg_utils.py:839`).
  std::string model_argument;
  // vLLM's own `--revision` (`arg_utils.py:839`, `config/model.py:183`).
  // Applies to both HuggingFace forms. There is deliberately no inline
  // `org/repo@rev` syntax, because vLLM does not spell it that way.
  std::string revision;
  // vLLM's own `--download-dir` (`config/model.py:183`): the directory that
  // holds the `models--org--repo` folders.
  std::string download_dir;
  std::string host = "0.0.0.0";
  int port = 8000;
  std::string tokenizer_config;  // default: <model_dir>/tokenizer_config.json
  std::string served_model_name;  // default: the model dir name
  int block_size = 32;
  // --num-blocks is the KV block-count OVERRIDE (0 => auto: sized by the knobs
  // below, else the 256-block fallback). ROAD-V1-MEM M1.
  int num_blocks = 0;
  // --gpu-memory-utilization: fraction of free device memory for the whole
  // engine (needs the M3 profile run; inert until then). --kv-cache-memory: an
  // absolute KV-pool size in bytes that sizes the block count directly (0 =>
  // unset).
  //
  // TRI-STATE (FIX-GPU-MEM-UTIL-INERT, #1165), like enable_prefix_caching
  // below: nullopt means the flag was NOT passed. It carries a value only when
  // the user typed one, which is what lets the engine warn that a chosen
  // fraction sized nothing without warning on a default nobody chose. A plain
  // double pre-filled with 0.92 could not express the difference.
  std::optional<double> gpu_memory_utilization = std::nullopt;
  long long kv_cache_memory_bytes = 0;
  // --kv-cache-dtype: vLLM CacheConfig.cache_dtype (config/cache.py:19-36,76).
  // "auto" (the default) uses the model dtype and is byte-identical to before
  // the flag existed; "fp8"/"fp8_e4m3" stores the paged K/V as 1-byte fp8, which
  // HALVES the bytes per KV block and so doubles the pool at the same
  // --kv-cache-memory. The value is resolved against the checkpoint's own
  // `kv_cache_quant_algo` inside LoadedEngine::FromModelDir, and an explicit
  // value always beats the checkpoint (torch_utils.py:380-381).
  std::string kv_cache_dtype = "auto";
  int max_model_len = 0;  // 0 => config.max_position_embeddings
  int max_num_seqs = 32;  // see model_loader.h: 8 clamped c8 batching.
  int max_num_batched_tokens = 0;  // 0 => per-architecture default.
  // --device: explicit device selection for the TEXT engine (ARCH-ONE-SURFACE
  // ROW 8), the vLLM DeviceConfig.device names this build serves: "auto"
  // (default — the accelerator-first probe, byte-identical to before the flag),
  // "cpu" (force the CPU queue), "cuda" (require CUDA; an absent device fails
  // startup LOUDLY, never a silent fallback). The video engine keeps its own
  // --video-device below: the two engines are loaded from different
  // checkpoints and may legitimately serve on different devices.
  std::string device = "auto";
  // --- MiniMax-H3 video generation (opt-in; absent => /v1/videos is unregistered
  // and the server behaves exactly as before). ---
  std::string video_dit, video_vae, video_vae_config, audio_vae, audio_vae_config;
  std::string video_prompt_embeds, video_workdir = "/tmp/vllm_h3_videos";
  std::string video_encoder, video_tokenizer;
  int video_encoder_max_layers = 50;
  std::string video_ffmpeg = "ffmpeg", video_device = "cuda";
  std::string video_partition;  // served partition (fl2va|ref2va); see the #77 guard
  // The video model FAMILY to load. EMPTY keeps detection, which is the default
  // and what every pre-L9B invocation gets. See the --video-family block below
  // for why it now exists.
  std::string video_family;
  // ── Speech + music generation (W6 of #672). OPT-IN: absent =>
  // /v1/audio/speech is never registered and the server answers 404 exactly as
  // it did before this flag existed. ────────────────────────────────────────
  // The checkpoint SET directory (MiniMax-Music3 ships six component
  // directories beside a modular_model_index.json).
  std::string speech_model;
  // The family to load. EMPTY keeps DETECTION, which inspects the artifact; an
  // unregistered name is refused at ParseArgs naming what is registered, and is
  // never treated as a hint, because the wrong family would not fail — it would
  // render noise.
  std::string speech_family;
  // WHERE the speech family runs. 0 = CPU, the default and the arm every Music3
  // correctness gate was taken on; 1 = the accelerator this build resolves.
  // A value the seam cannot serve is refused BY NAME at load, not substituted.
  int32_t speech_device = 0;
  // FAMILY-SPECIFIC load knobs, `--video-extra KEY=VALUE`, repeatable. Pinning a
  // family is useless if that family's required load knobs are unreachable:
  // LTX-2.5 cannot load without `dit_config_path` (the shipped FP8 DiT carries
  // no __metadata__) and `audio_prompt_embeds_path` (its audio stream conditions
  // at a second width), and neither has — or should have — a dedicated flag on a
  // family-generic server. `--video-partition` remains the documented alias for
  // the H3 key "partition".
  std::vector<std::pair<std::string, std::string>> video_extras;
  // Keep-quant is the library seam's DEFAULT arm; --video-dequant-bf16 selects
  // the bf16 dequant/stream arm (the throughput trade the gen example ships).
  // --video-keep-quant is still accepted (it names the default).
  bool video_dequant_bf16 = false;
  int cuda_profile_graph_replays = 0;  // trace-only diagnostic build seam.
  int cuda_profile_graph_batch = 0;  // 0 => accepted c16 trace contract.
  std::string benchmark_shutdown_fifo;  // paired trace-only control path.
  std::optional<bool> enable_prefix_caching = std::nullopt;
  bool enable_force_include_usage = false;
  // GET /tokenizer_info gate. Mirrors vLLM's --enable-tokenizer-info-endpoint
  // (entrypoints/openai/cli_args.py:140, default False; the route is registered
  // only when serve/tokenize/api_router.py:95 sees the flag). Default off → the
  // route 404s, byte-identical to before.
  bool enable_tokenizer_info_endpoint = false;
  // Dev/admin endpoint gate. Mirrors vLLM's VLLM_SERVER_DEV_MODE env
  // (envs.py:157, default 0): build_app registers the dev/rlhf + dev/cache
  // routers only under `if envs.VLLM_SERVER_DEV_MODE` (api_server.py:238). Off by
  // default → /abort_requests 404s. Enables the /abort_requests production wiring.
  bool enable_server_dev_mode = false;
  bool verbose = false;
  // Our spelling of vLLM's `--default-chat-template-kwargs enable_thinking`.
  // TRI-STATE, and the third state is the default and the point (#1681):
  // upstream's own default is `None`, so unless somebody asks, `enable_thinking`
  // is not a template variable at all and a template that gates on
  // `{% if enable_thinking is undefined %}` gets its own answer. Storing a plain
  // `false` here made that test permanently false and silently inverted the
  // Qwen3.8 family's reasoning default against vLLM and SGLang.
  std::optional<bool> enable_thinking;
  // Request logging (Python vLLM --enable-log-requests parity). Default ON.
  bool enable_log_requests = true;
  bool enable_log_outputs = false;
  int max_log_len = 256;
  // Attach Prometheus logger + GET /metrics (default ON for solid Hermes serve).
  bool enable_metrics = true;
  // Scheduling policy: "fcfs" (default), "priority" (mirrors vLLM's
  // --scheduling-policy / SchedulerConfig.policy), or "lpm" (SGLang's
  // cache-aware longest-prefix-match admission ordering, ENG-SGLANG-BEHAVIOR-FLAG;
  // opt-in, output-neutral, resolves to fcfs when prefix caching is off).
  // --schedule-policy is accepted as an SGLang-compatible alias.
  std::string scheduling_policy = "fcfs";
  // Jump-forward decoding (ENG-SGLANG-BEHAVIOR-FLAG SW3): tri-state, mirrors the
  // C-ABI vllm_model_params.enable_jump_forward. Unset (default) => OFF unless
  // VT_ENABLE_JUMP_FORWARD is set; --enable-jump-forward forces on,
  // --disable-jump-forward forces off (the env var still overrides). The
  // token-unique forced-run subset only; see .agents/specs/sglang-enablement.md.
  std::optional<bool> enable_jump_forward = std::nullopt;
  // C-ABI vllm_model_params.disable_sliding_window (ABI v26). Unset (default) =>
  // the sliding window is ENABLED, which is upstream's own `= False`.
  std::optional<bool> disable_sliding_window = std::nullopt;
  // Tool-call / reasoning dialect selection (mirrors vLLM's --tool-call-parser
  // and --reasoning-parser). THE DEFAULTS ARE TODAY'S HARDCODED BEHAVIOUR:
  // "hermes" is exactly what OpenAIServingChat was constructed with before this
  // flag existed, and "none" is the empty reasoning-parser name it passed. An
  // invocation that names neither flag is therefore unchanged, byte for byte.
  // "auto" opts into the chat-template detection the C ABI uses.
  std::string tool_call_parser = "hermes";
  // vLLM's --generation-config (config/model.py:298-304). "auto" (the DEFAULT)
  // takes the checkpoint's own generation_config.json sampling keys as the
  // server-wide request defaults; "vllm" loads no file and keeps vLLM's neutral
  // defaults; a DIRECTORY path reads a generation_config.json from there.
  //
  // "auto" is what upstream defaults to, so it is what a parity benchmark
  // compares against, and #1985 is what taking anything else cost. The escape
  // hatch is exactly the one upstream's own warning names.
  std::string generation_config = "auto";
  std::string reasoning_parser = "none";
  // vLLM's --enable-auto-tool-choice (cli_args.py:105, default False). Accepted
  // and INERT here (see kAcceptedInertArgs below) but still RECORDED, because
  // cli_args.py:395 validates it against --tool-call-parser and that validation
  // is mirrored, not dropped.
  bool enable_auto_tool_choice = false;
  // vLLM's --kv-transfer-config: the external KV connector selection, as the
  // same JSON object vLLM takes. Empty (default) == no connector == the inert
  // production path. See docs/KV-OFFLOAD.md.
  std::string kv_transfer_config;
  // ENG-WEIGHT-OFFLOAD W0b — vLLM's OffloadConfig, the WEIGHT-offload selection
  // (distinct from --kv-transfer-config, which offloads KV blocks), as the same
  // JSON object vLLM takes. Empty (default) == no offloading == the inert
  // production path. Accepted and validated today; the offloader that MOVES a
  // weight is W2/W5.
  std::string offload_config;
  // vLLM's --speculative-config: the speculative-decoding selection, as the same
  // JSON object vLLM takes (e.g. '{"method":"mtp","num_speculative_tokens":1}').
  // Empty (default) == no speculation == the inert production path (SPEC-MTP I5d).
  std::string speculative_config;
  // ── Multimodal input limits (ENG-MM-INPUT-PIPELINE wave L2, #607) ─────────
  // --language-model-only (arg_utils.py:555,1276,1691) and --limit-mm-per-prompt
  // (arg_utils.py:556,1279,1692), the two flags 43 of the 157 official recipes
  // need. They are sugar over vllm::MultiModalConfig, which is the mechanism:
  // the flag sets every modality limit to 0, and the REFUSAL those zeros produce
  // (BaseProcessingInfo::ValidateNumItems, #607 L1) is the observable effect.
  //
  // Defaults reproduce today's behaviour exactly: language_model_only false and
  // an empty map resolve to the 999-per-modality default, so a server started
  // without either flag refuses nothing it used to serve.
  vllm::MultiModalConfig multimodal;
  // ── The `clip` multimodal projector (row `LOAD-GGUF-MMPROJ`, #821) ────────
  // llama.cpp's `--mmproj`: the SECOND GGUF file, beside a `.gguf` --model.
  // Empty (default) == no projector == every load that existed before the row.
  // Deliberately explicit: a sibling `mmproj*.gguf` is NOT auto-discovered,
  // because a directory holding two unrelated models must not silently fuse
  // them.
  std::string mmproj_path;
};

// ── Accepted-and-inert serve arguments (SERVE-RECIPE-ARGS, #606) ────────────
// A published `vllm serve` line must reach model load. Measured over
// vllm-project/recipes @ 86c7777a (157 official model recipes), 89 pass
// `--enable-auto-tool-choice` and 82 pass `--trust-remote-code`; neither means
// anything to this engine, and both used to stop the server at the unknown-
// argument guard below before a single weight was read.
//
// THIS IS NOT A CATCH-ALL, and that is the whole point. A flag that is inert
// because we LACK the capability -- --tensor-parallel-size, the EP flags,
// --mm-encoder-tp-mode -- keeps aborting, because silently swallowing it would
// let a user believe they got tensor parallelism. Only a flag that is inert BY
// CONSTRUCTION earns an entry, and the per-entry `reason` is the enforcement:
// an entry that cannot state an honest reason cannot be written.
//
// Inert is also not UNVALIDATED. --enable-auto-tool-choice is recorded on Args
// and still checked against --tool-call-parser after the loop, mirroring
// vllm/entrypoints/openai/cli_args.py:395.
struct InertArg {
  const char* flag;
  bool takes_value;   // consume the following argv entry as this flag's value
  const char* reason;  // printed on use; see rule 2 -- accepting is ANNOUNCED
};

constexpr bool kNoValue = false;

// NOTE on `takes_value`: no entry needs it today (both flags are bare), but the
// field is part of the table's shape rather than a later retrofit -- a value-
// taking recipe flag must not silently leave its value to be re-parsed as the
// next flag.
constexpr InertArg kAcceptedInertArgs[] = {
    // cli_args.py:105 (`enable_auto_tool_choice: bool = False`), threaded on as
    // `enable_auto_tools` at api_server.py:426,441,529,544.
    //
    // The trailing caveat clause is deliberate and must stay: upstream defaults
    // --tool-call-parser to None, we default it to "hermes"
    // (docs/USAGE.md), so upstream's flag genuinely gates something and ours
    // cannot. The notice must not let a reader infer the two agree when the
    // parser is unset. Reconciling that default is out of scope here.
    {"--enable-auto-tool-choice", kNoValue,
     // Do not write `open (` here: check-windows-portability.py scans this file
     // with comments stripped but STRING LITERALS intact, and its POSIX-call
     // pattern matches the bare word `open` before a parenthesis. The semicolon
     // keeps the sentence and keeps the Windows gate green.
     "tool parsing is already unconditional once --tool-call-parser resolves, "
     "so there is no second gate to open; note --tool-call-parser defaults to "
     "hermes here, where upstream's defaults to unset"},
    // Authorizes executing Python from the checkpoint. N/A BY CONSTRUCTION, not
    // unimplemented: this engine has no Python runtime.
    {"--trust-remote-code", kNoValue,
     "no Python runtime, so there is no remote code to trust"},
};

const InertArg* FindAcceptedInertArg(const std::string& flag) {
  for (const InertArg& entry : kAcceptedInertArgs) {
    if (flag == entry.flag) return &entry;
  }
  return nullptr;
}

[[noreturn]] void Usage(const char* argv0, int code) {
  std::cerr
      << "usage: " << argv0
      << " --model <dir> [--host H] [--port P] [--tokenizer-config F]\n"
         "               [--served-model-name N] [--block-size N] "
         "[--num-blocks N] [--max-model-len N]\n"
         "               [--gpu-memory-utilization F] "
         "[--kv-cache-memory BYTES]\n"
         "               [--kv-cache-dtype auto|bfloat16|fp8|fp8_e4m3]\n"
         "               [--max-num-seqs N] "
         "[--max-num-batched-tokens N]\n"
         "               [--device auto|cpu|cuda]\n"
         "               [--cuda-profile-graph-replays N]\n"
         "               [--cuda-profile-graph-batch N]\n"
         "               [--benchmark-shutdown-fifo F]\n"
         "               [--enable-force-include-usage]\n"
         "               [--enable-tokenizer-info-endpoint]\n"
         "               [--enable-server-dev-mode]\n"
         "               [--revision REF] [--download-dir DIR]\n"
         "               [--verbose]\n"
         "               [--enable-thinking|--no-enable-thinking]\n"
         "               [--enable-log-requests|--disable-log-requests]\n"
         "               [--enable-log-outputs] [--max-log-len N]\n"
         "               [--enable-metrics|--disable-metrics]\n"
         "               [--[no-]enable-prefix-caching]\n"
         "               [--[no-]enable-radix-attention]\n"
         "               [--scheduling-policy fcfs|priority|lpm]\n"
         "               [--[enable|disable]-jump-forward]\n"
         "               [--generation-config auto|vllm|<dir>]\n"
         "               [--tool-call-parser <name>|auto|none]\n"
         "               [--reasoning-parser <name>|auto|none]\n"
         "               [--kv-transfer-config '<json>']\n"
         "               [--offload-config '<json>']\n"
         "               [--speculative-config '<json>']\n"
         "               [--[no-]language-model-only]\n"
         "               [--limit-mm-per-prompt '<json>']\n"
         "               [--mmproj <mmproj-*.gguf>]\n"
         "               [--speech-model <checkpoint-dir>] "
         "[--speech-family <name>]\n"
         "               [--speech-device 0|1]\n"
         "               [--version]\n"
         "  --speech-model WITHOUT --model serves /v1/audio/speech ALONE (no "
         "text model is loaded)\n"
         "  --speech-device 0 (default) runs the speech family on the CPU, 1 on "
         "the accelerator this build resolves\n"
         "  accepted for published-recipe compatibility, NO effect: "
         "--enable-auto-tool-choice, --trust-remote-code\n";
  std::exit(code);
}

std::string NextArg(int argc, char** argv, int& i, const char* argv0) {
  if (i + 1 >= argc) Usage(argv0, 2);
  return argv[++i];
}

Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--model") {
      a.model_dir = NextArg(argc, argv, i, argv[0]);
      a.model_argument = a.model_dir;
    } else if (flag == "--revision") {
      a.revision = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--download-dir") {
      a.download_dir = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--host") {
      a.host = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--port") {
      a.port = std::stoi(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--tokenizer-config") {
      a.tokenizer_config = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--served-model-name") {
      a.served_model_name = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--block-size") {
      a.block_size = std::stoi(NextArg(argc, argv, i, argv[0]));
      // The attention backends (FLASH_ATTN / ROCM_ATTN get_kv_cache_shape)
      // enforce block_size % 16 == 0 and the runner validates at init — fail
      // here with a clear message rather than at engine init.
      if (a.block_size <= 0 || a.block_size % 16 != 0) {
        std::cerr << argv[0] << ": --block-size must be a positive multiple of 16"
                  << " (got " << a.block_size << ")\n";
        Usage(argv[0], 2);
      }
    } else if (flag == "--num-blocks") {
      a.num_blocks = std::stoi(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--gpu-memory-utilization") {
      a.gpu_memory_utilization = std::stod(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--kv-cache-memory") {
      a.kv_cache_memory_bytes = std::stoll(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--kv-cache-dtype") {
      a.kv_cache_dtype = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--max-model-len") {
      a.max_model_len = std::stoi(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--max-num-seqs") {
      a.max_num_seqs = std::stoi(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--max-num-batched-tokens") {
      a.max_num_batched_tokens = std::stoi(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--device") {
      // Text-engine device selection (mirrors vLLM's DeviceConfig.device
      // names). Validated by vllm::DeviceFromString at engine construction;
      // --video-device (below) stays the video engine's separate knob.
      a.device = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--cuda-profile-graph-replays") {
      a.cuda_profile_graph_replays =
          std::stoi(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--cuda-profile-graph-batch") {
      a.cuda_profile_graph_batch =
          std::stoi(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--benchmark-shutdown-fifo") {
      a.benchmark_shutdown_fifo = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--enable-force-include-usage") {
      a.enable_force_include_usage = true;
    } else if (flag == "--enable-tokenizer-info-endpoint") {
      a.enable_tokenizer_info_endpoint = true;
    } else if (flag == "--video-dit") {
      a.video_dit = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--video-vae") {
      a.video_vae = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--video-vae-config") {
      a.video_vae_config = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--audio-vae") {
      a.audio_vae = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--audio-vae-config") {
      a.audio_vae_config = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--video-encoder") {
      a.video_encoder = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--video-tokenizer") {
      a.video_tokenizer = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--video-encoder-max-layers") {
      a.video_encoder_max_layers = std::atoi(NextArg(argc, argv, i, argv[0]).c_str());
    } else if (flag == "--video-prompt-embeds") {
      a.video_prompt_embeds = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--video-workdir") {
      a.video_workdir = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--video-ffmpeg") {
      a.video_ffmpeg = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--video-device") {
      a.video_device = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--video-partition") {
      a.video_partition = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--video-family") {
      a.video_family = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--speech-model") {
      a.speech_model = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--speech-family") {
      a.speech_family = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--speech-device") {
      a.speech_device = std::stoi(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--video-extra") {
      const std::string kv = NextArg(argc, argv, i, argv[0]);
      const std::string::size_type eq = kv.find('=');
      // A bare KEY is refused rather than read as KEY="": an extra whose value
      // silently became empty is how a mistyped knob renders the default and
      // looks like the feature not working, which is the same failure the
      // families' own unknown-extra refusals exist to stop.
      if (eq == std::string::npos || eq == 0) {
        std::cerr << "server: --video-extra takes KEY=VALUE, got '" << kv << "'\n";
        Usage(argv[0], 2);
      }
      a.video_extras.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
    } else if (flag == "--video-keep-quant") {
      // the seam's default arm; accepted for pre-fold CLI compatibility
      a.video_dequant_bf16 = false;
    } else if (flag == "--video-dequant-bf16") {
      a.video_dequant_bf16 = true;
    } else if (flag == "--enable-server-dev-mode") {
      a.enable_server_dev_mode = true;
    } else if (flag == "--verbose" || flag == "-v") {
      a.verbose = true;
    } else if (flag == "--enable-thinking") {
      a.enable_thinking = true;
    } else if (flag == "--no-enable-thinking") {
      a.enable_thinking = false;  // an EXPLICIT false, unlike passing neither
    } else if (flag == "--enable-log-requests") {
      a.enable_log_requests = true;
    } else if (flag == "--disable-log-requests") {
      a.enable_log_requests = false;
    } else if (flag == "--enable-log-outputs") {
      a.enable_log_outputs = true;
    } else if (flag == "--max-log-len") {
      a.max_log_len = std::stoi(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--enable-metrics") {
      a.enable_metrics = true;
    } else if (flag == "--disable-metrics") {
      a.enable_metrics = false;
    } else if (flag == "--enable-prefix-caching" ||
               flag == "--no-enable-prefix-caching" ||
               flag == "--enable-radix-attention" ||
               flag == "--disable-radix-attention") {
      // --[no-]enable-prefix-caching is vLLM's flag. --enable-radix-attention /
      // --disable-radix-attention are SGLang-compatible ALIASES for the SAME
      // toggle (RadixAttention is fused into our block-hash APC — there is no
      // distinct radix code path; see .agents/specs/sglang-radixattention.md §1).
      // They set the identical tri-state as the vLLM flag; last-wins is rejected
      // (mirrors passing the vLLM flag twice) so a contradictory pair is caught.
      if (a.enable_prefix_caching.has_value()) {
        std::cerr << "server: prefix-caching flag (--[no-]enable-prefix-caching "
                     "/ --[disable|enable]-radix-attention) specified more than "
                     "once\n";
        Usage(argv[0], 2);
      }
      a.enable_prefix_caching =
          flag == "--enable-prefix-caching" || flag == "--enable-radix-attention";
    } else if (flag == "--scheduling-policy" || flag == "--schedule-policy") {
      // --scheduling-policy is vLLM's flag; --schedule-policy is SGLang's name,
      // accepted as an alias. Both take fcfs|priority|lpm.
      a.scheduling_policy = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--enable-jump-forward" ||
               flag == "--disable-jump-forward") {
      // ENG-SGLANG-BEHAVIOR-FLAG SW3: opt into (or force off) jump-forward
      // decoding — the token-unique grammar-speed subset (see
      // .agents/specs/sglang-enablement.md). Absent => the default (OFF unless
      // VT_ENABLE_JUMP_FORWARD is set). The env var, when set, still overrides.
      if (a.enable_jump_forward.has_value()) {
        std::cerr << "server: jump-forward flag "
                     "(--[enable|disable]-jump-forward) specified more than "
                     "once\n";
        Usage(argv[0], 2);
      }
      a.enable_jump_forward = flag == "--enable-jump-forward";
    } else if (flag == "--disable-sliding-window" ||
               flag == "--enable-sliding-window") {
      // ENG-ATTENTION-WINDOW W3 (#2388): vLLM spells this
      // `--disable-sliding-window` (`ModelConfig.disable_sliding_window`,
      // config/model.py:248), so this carries the upstream name. The `--enable-`
      // spelling is the explicit opposite, matching the jump-forward pair above;
      // upstream has no such flag because its field defaults to False, and
      // neither does anything different from omitting both.
      if (a.disable_sliding_window.has_value()) {
        std::cerr << "server: sliding-window flag "
                     "(--[enable|disable]-sliding-window) specified more than "
                     "once\n";
        Usage(argv[0], 2);
      }
      a.disable_sliding_window = flag == "--disable-sliding-window";
    } else if (flag == "--generation-config") {
      a.generation_config = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--tool-call-parser") {
      a.tool_call_parser = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--reasoning-parser") {
      a.reasoning_parser = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--kv-transfer-config") {
      a.kv_transfer_config = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--offload-config") {
      a.offload_config = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--speculative-config") {
      a.speculative_config = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--mmproj") {
      // llama.cpp's own spelling for the second file (`b10451`
      // `tools/mtmd/mtmd-cli.cpp`). NOT validated here: unlike the JSON flags
      // above, whose value can be refused without touching the disk, this is a
      // path whose contents decide the answer — and the loader already refuses
      // it BY NAME before the tokenizer and before any language weight byte, so
      // a second check here would be a second implementation of the same
      // refusal rather than an earlier one.
      a.mmproj_path = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--language-model-only" ||
               flag == "--no-language-model-only") {
      // arg_utils.py:1276 over a bool field, which _compute_kwargs gives
      // argparse.BooleanOptionalAction (arg_utils.py:346-348) — so upstream
      // accepts BOTH spellings and the negative one is the default. Mirrored,
      // rather than accepting only the positive: a recipe that switches the flag
      // off explicitly must not die on an unknown argument.
      a.multimodal.language_model_only = flag == "--language-model-only";
    } else if (flag == "--limit-mm-per-prompt") {
      // arg_utils.py:1279 over a dict field => type=parse_type(json.loads)
      // (arg_utils.py:379-381 — the plain-dict branch, NOT the
      // `union_dict_and_str` one at :374-378, which needs a `str` arm or a
      // non-builtin hint that `dict[str, BaseDummyOptions]` does not have):
      // the value is a JSON OBJECT. Parsed HERE, before
      // the multi-GB model load, for the same reason the parser dialects are:
      // a malformed limit costs a second rather than a full load, and it is
      // REFUSED rather than defaulted — a typo that silently became 999 is a
      // limit that is not there.
      //
      // NAMED RESIDUAL, and it is not this flag's: upstream's dotted spelling
      // (`--limit-mm-per-prompt.image 2`) is a FlexibleArgumentParser feature
      // that rewrites any dotted key into the JSON form before argparse sees it
      // (argparse_utils.py:389-425). It applies equally to --kv-transfer-config
      // and --speculative-config, which this server also takes as JSON only, so
      // adding it for one flag would be the bespoke path. It belongs to a parser
      // brick covering all three.
      const std::string value = NextArg(argc, argv, i, argv[0]);
      std::vector<std::string> ignored_options;
      try {
        a.multimodal.limit_per_prompt =
            vllm::ParseLimitMmPerPromptJson(value, &ignored_options);
      } catch (const std::exception& e) {
        std::cerr << "server: --limit-mm-per-prompt: " << e.what() << "\n";
        Usage(argv[0], 2);
      }
      // Accepting is ANNOUNCED (the kAcceptedInertArgs rule, applied to a
      // partially-inert VALUE): the dummy-profiling options are validated and
      // then dropped, so a reader learns `num_frames` did nothing here instead
      // of inferring that it worked.
      for (const std::string& key : ignored_options) {
        std::cerr << "server: --limit-mm-per-prompt " << key
                  << " accepted and IGNORED: only the modality's count is read "
                     "here. A key upstream DECLARES sizes dummy inputs for "
                     "memory profiling, a surface this engine does not have; a "
                     "key it does not declare is one its BaseDummyOptions "
                     "fallback drops too\n";
      }
    } else if (flag == "--version") {
      std::cout << "vllm.cpp " << vllm::Version()
                << " c-abi=" << VLLM_ABI_VERSION << "\n";
      std::exit(0);
    } else if (flag == "-h" || flag == "--help") {
      Usage(argv[0], 0);
    } else if (const InertArg* inert = FindAcceptedInertArg(flag);
               inert != nullptr) {
      // Rule 2: accepting is ANNOUNCED. One notice per accepted flag AS USED,
      // naming the flag and its reason, so a log reader learns the flag did
      // nothing rather than inferring that it worked.
      if (inert->takes_value) (void)NextArg(argc, argv, i, argv[0]);
      if (flag == "--enable-auto-tool-choice") {
        a.enable_auto_tool_choice = true;  // validated below, cli_args.py:395
      }
      std::cerr << "server: accepted '" << flag
                << "' for published-recipe compatibility; it has no effect "
                   "here: "
                << inert->reason << "\n";
    } else {
      std::cerr << "server: unknown argument '" << flag << "'\n";
      Usage(argv[0], 2);
    }
  }
  // --model is required EXCEPT for a speech-only server (#672). Upstream's own
  // spelling for MiniMax-Music3 is `sgl-omni serve --model MiniMaxAI/MiniMax-Music3`
  // and nothing else — there is no text model in that command, because a music
  // model is not an accessory to one. Requiring --model here made the documented
  // recipe unrunnable on any box whose smallest text checkpoint is tens of GB:
  // serving a 28.5 GB music model also staged a model no request would touch.
  //
  // This is a NEW ACCEPTED COMBINATION, not a change of behaviour: `--model` with
  // or without `--speech-model` resolves exactly as before, and the case that
  // changes verdict — NEITHER flag — was and remains an error. It mirrors the
  // task-conditional dispatch already in this file for pooling and
  // transcription-only models (vLLM api_server.py:255-265): the routes a server
  // registers follow from what it loaded.
  if (a.model_dir.empty() && a.speech_model.empty()) {
    std::cerr << "server: --model <dir> is required (or --speech-model <dir> for "
                 "a speech/music-only server)\n";
    Usage(argv[0], 2);
  }
  if (a.max_num_seqs <= 0 || a.max_num_batched_tokens < 0 ||
      a.cuda_profile_graph_replays < 0 || a.cuda_profile_graph_batch < 0) {
    std::cerr << "server: scheduler capacities must be positive "
                 "(--max-num-batched-tokens may be 0 for auto)\n";
    Usage(argv[0], 2);
  }
  if ((a.cuda_profile_graph_replays > 0) !=
      !a.benchmark_shutdown_fifo.empty()) {
    std::cerr << "server: --cuda-profile-graph-replays and "
                 "--benchmark-shutdown-fifo must be specified together\n";
    Usage(argv[0], 2);
  }
  if (a.cuda_profile_graph_replays == 0 && a.cuda_profile_graph_batch != 0) {
    std::cerr << "server: --cuda-profile-graph-batch requires "
                 "--cuda-profile-graph-replays\n";
    Usage(argv[0], 2);
  }
  if (a.cuda_profile_graph_replays > 0 && a.cuda_profile_graph_batch == 0) {
    a.cuda_profile_graph_batch = 16;
  }
  if (a.cuda_profile_graph_batch > a.max_num_seqs) {
    std::cerr << "server: --cuda-profile-graph-batch exceeds --max-num-seqs\n";
    Usage(argv[0], 2);
  }
  // The declared video family, checked HERE for the same reason the parser
  // dialects below are: `LoadVideoEngine` would refuse an unregistered name
  // anyway, but only after the TEXT model has loaded, so a typo would cost a
  // multi-GB load instead of a second. The REGISTRY is the authority — never a
  // literal list here — so a family added in its own file is accepted with no
  // edit to this one, which is the whole point of the registration seam.
  if (!a.video_family.empty()) {
    const std::vector<std::string> registered =
        vllm::multimodal::RegisteredVideoFamilies();
    if (std::find(registered.begin(), registered.end(), a.video_family) ==
        registered.end()) {
      std::cerr << "server: --video-family '" << a.video_family
                << "' is not a registered video family. Registered families:";
      for (const std::string& name : registered) std::cerr << " " << name;
      std::cerr << "\n";
      Usage(argv[0], 2);
    }
  }
  if (!a.speech_family.empty()) {
    vllm::multimodal::SpeechRegistry& registry = vllm::multimodal::GlobalSpeechRegistry();
    vllm::models::music3::RegisterBuiltinSpeechFamilies(registry);
    const std::vector<std::string> registered = registry.families();
    if (std::find(registered.begin(), registered.end(), a.speech_family) == registered.end()) {
      std::cerr << "server: --speech-family '" << a.speech_family
                << "' is not a registered speech family. Registered families:";
      for (const std::string& name : registered) std::cerr << " " << name;
      std::cerr << "\n";
      Usage(argv[0], 2);
    }
  }
  if (!a.speech_family.empty() && a.speech_model.empty()) {
    std::cerr << "server: --speech-family names a family but --speech-model names no "
                 "checkpoint; there is nothing to load it from\n";
    Usage(argv[0], 2);
  }
  if (a.speech_device != 0 && a.speech_model.empty()) {
    // Same shape as the --speech-family check above, and for the same reason: a
    // knob that silently applies to nothing reads as "it was honoured".
    std::cerr << "server: --speech-device selects where the speech family runs but "
                 "--speech-model names no checkpoint; there is nothing to place\n";
    Usage(argv[0], 2);
  }
  // Mirrors vllm/entrypoints/openai/cli_args.py:395 — upstream raises
  //   TypeError("Error: --enable-auto-tool-choice requires --tool-call-parser")
  // when the flag is set and `args.tool_call_parser` is falsy. Upstream's falsy
  // value is the unset default `None`; ours is the explicit selection "none",
  // because our --tool-call-parser defaults to "hermes" and so is never unset.
  // Accepting the flag as inert does NOT drop the validation it came with.
  // Ordered before the dialect check below so a contradiction is reported as
  // the contradiction; "none" is itself a registered selection, so that check
  // would pass and say nothing about the conflict.
  if (a.enable_auto_tool_choice && a.tool_call_parser == "none") {
    std::cerr << "server: Error: --enable-auto-tool-choice requires "
                 "--tool-call-parser\n"
                 "server: (--tool-call-parser none selects NO parser; name a "
                 "parser, or drop --tool-call-parser to keep the hermes "
                 "default)\n";
    Usage(argv[0], 2);
  }
  // Validate a NAMED parser dialect here, before the (multi-GB) model load, so a
  // typo costs a second rather than a full load. "auto" cannot be checked yet —
  // it resolves against the chat template — but detection only ever returns
  // registered names, so it cannot fail later either.
  namespace oai = vllm::entrypoints::openai;
  if (a.tool_call_parser != "auto") {
    (void)oai::ResolveToolParserName(a.tool_call_parser, "");
  }
  if (a.reasoning_parser != "auto") {
    (void)oai::ResolveReasoningParserName(a.reasoning_parser, "");
  }
  return a;
}


// ENG-HF-MODEL-DOWNLOAD W4 (#1280): turn what the user typed after `--model`
// into a path the loader can open, fetching the checkpoint when it names a
// HuggingFace repository.
//
// THIS IS THE PRODUCTION CALL SITE, and the ONLY one. `vllm-server` reaches
// `--model` through `VllmServerMain`, so a repository identifier that is not
// resolved here is not resolved anywhere. Deleting this call leaves the loader
// opening `org/repo` as a relative path, which is the behavior this row
// replaces, and the end-to-end case in
// `tests/vllm/entrypoints/openai/test_serve_hf_model.cpp` turns red.
//
// `vllm-cli` deliberately does NOT reach it. That example includes `vllm.h` and
// nothing else, because an example is an application binary interface client
// only, and this row adds no ABI function, so `vllm-cli --model org/repo` still
// takes a local path. It is recorded under `## Owed` in
// `.agents/specs/hf-model-download.md`.
//
// An existing directory and an existing `.gguf` file come back unchanged and
// open no socket, so a local run is byte-identical to the one before this row.
void ResolveModelArgument(Args& a) {
  if (a.model_dir.empty()) return;
  vllm::transformers_utils::ModelResolveOptions opts;
  opts.revision = a.revision;
  opts.download_dir = a.download_dir;
  opts.verbose = a.verbose;
  const std::string resolved =
      vllm::transformers_utils::ResolveModelPath(a.model_dir, opts);
  if (resolved == a.model_dir) return;
  // The SERVED name stays the name the user asked for. Without this the
  // default would become the 40 character commit directory the cache happens
  // to hold, and no client could name the model it just started.
  if (a.served_model_name.empty()) a.served_model_name = a.model_argument;
  std::cerr << "server: --model " << a.model_argument << " resolved to "
            << resolved << "\n";
  a.model_dir = resolved;
}

}  // namespace

namespace vllm {
namespace entrypoints {
namespace openai {

int VllmServerMain(int argc, char** argv) {
  try {
    // ENG-HF-MODEL-DOWNLOAD W5 (#1280). BEFORE anything is parsed, bound or
    // fetched: `CPPHTTPLIB_OPENSSL_SUPPORT` is a whole-header switch that
    // changes the layout of `httplib::Result`, and this binary carries both a
    // listener and a hub client compiled from that one header. A build that
    // defined it for some translation units and not for others LINKS CLEANLY
    // and then corrupts every response object handed across the seam. Refusing
    // here is the only place a user finds out before a wrong answer.
    const std::string transport_mismatch = vllm::HttpTransportAbiMismatch();
    if (!transport_mismatch.empty()) {
      std::cerr << transport_mismatch << "\n";
      return 2;
    }
    Args args = ParseArgs(argc, argv);
    if (args.verbose) {
      SetEnvironment("VT_SERVER_VERBOSE", "1");
      std::cerr << "server: verbose stage logging enabled (debug_stages)\n";
    }
    ResolveModelArgument(args);
    {
      vllm::entrypoints::openai::RequestLogConfig log_cfg;
      log_cfg.enable_log_requests = args.enable_log_requests;
      log_cfg.enable_log_outputs = args.enable_log_outputs || args.verbose;
      log_cfg.max_log_len = args.max_log_len;
      log_cfg.debug_stages = args.verbose ||
                             (std::getenv("VT_SERVER_VERBOSE") &&
                              std::getenv("VT_SERVER_VERBOSE")[0] == '1');
      vllm::entrypoints::openai::ConfigureRequestLogger(log_cfg);
      std::cerr << "server: request logging "
                << (log_cfg.enable_log_requests ? "ON" : "OFF")
                << " outputs=" << (log_cfg.enable_log_outputs ? "ON" : "OFF")
                << " max_log_len=" << log_cfg.max_log_len
                << " debug_stages=" << (log_cfg.debug_stages ? "ON" : "OFF")
                << "\n";
    }

    // The RESOLVED multimodal limits, printed before the model load so the
    // startup log records what the two flags actually produced rather than what
    // was typed. GetLimitPerPrompt is the single accessor every consumer asks
    // (multimodal.py:321-336), so printing through it is what makes the flag's
    // PRECEDENCE visible: --language-model-only alongside an explicit
    // `image=4` prints image=0, because the flag is checked before the map.
    {
      std::cerr << "server: multimodal limits language-model-only="
                << (args.multimodal.language_model_only ? "ON" : "OFF");
      // Print every modality the user named, plus the ones the flag zeroes even
      // though nobody named them, so "--language-model-only" is not a line that
      // says only "ON" with no consequence next to it.
      std::vector<std::string> shown;
      for (const auto& [modality, unused] : args.multimodal.limit_per_prompt) {
        (void)unused;
        shown.push_back(modality);
      }
      if (shown.empty() && args.multimodal.language_model_only) {
        shown = {"audio", "image", "video"};
      }
      for (const std::string& modality : shown) {
        std::cerr << " " << modality << "="
                  << args.multimodal.GetLimitPerPrompt(modality);
      }
      if (shown.empty()) {
        std::cerr << " (no per-modality limit set; default "
                  << vllm::kDefaultLimitPerPrompt << ")";
      }
      std::cerr << "\n";
    }

    // ── SPEECH TASK DISPATCH (#672): --speech-model with NO --model serves
    // /v1/audio/speech and nothing else. It is the same task-conditional shape
    // the pooling and transcription-only branches below take, and the same one
    // vLLM's api_server.py:255-265 uses: a server registers the routes its
    // loaded task can answer, rather than every route it knows how to spell.
    //
    // It exists because upstream's own recipe is `sgl-omni serve --model
    // MiniMaxAI/MiniMax-Music3` — one model, no text tower — and because pairing
    // a 28.5 GB music checkpoint with an unrelated text checkpoint nobody
    // queries is not a smaller cost than the music model itself. NOTHING here
    // touches the combined path: --model + --speech-model still loads both and
    // registers both, byte for byte as before.
    if (args.model_dir.empty()) {
      vllm::multimodal::SpeechRegistry& registry = vllm::multimodal::GlobalSpeechRegistry();
      vllm::models::music3::RegisterBuiltinSpeechFamilies(registry);
      vllm::multimodal::SpeechModelParams smp;
      smp.path = args.speech_model;
      smp.family = args.speech_family;  // empty => DETECT by inspecting the artifact
      smp.device = args.speech_device;   // 0 => CPU, the byte-identical default
      std::string why;
      std::unique_ptr<vllm::multimodal::SpeechEngine> loaded_speech = registry.Load(smp, &why);
      if (loaded_speech == nullptr) {
        throw std::runtime_error("server: --speech-model " + args.speech_model + ": " + why);
      }
      std::shared_ptr<vllm::multimodal::SpeechEngine> speech_only(std::move(loaded_speech));
      // The served name defaults to the FAMILY rather than to the directory
      // basename, because a speech-only server has no config.json to name and
      // the family is what a client puts in `"model"`. An explicit
      // --served-model-name still wins.
      const std::string speech_served_name =
          args.served_model_name.empty() ? speech_only->family() : args.served_model_name;
      vllm::openai::SpeechCapabilities caps;
      caps.family = speech_only->family();
      caps.sample_rate = speech_only->sample_rate();
      caps.requires_reference_audio = speech_only->requires_reference_audio();
      std::cerr << "server: speech/music-only model (family=" << caps.family << ", "
                << caps.sample_rate << " Hz, "
                << (caps.requires_reference_audio ? "reference clip REQUIRED"
                                                  : "text-only synthesis")
                << ", family "
                << (args.speech_family.empty() ? "DETECTED" : "DECLARED (--speech-family)")
                // GRANTED, not requested. A log that echoed --speech-device back
                // would say "cuda" on a build that resolved CPU.
                << ", device " << vt::DeviceTypeName(speech_only->device().type)
                << "); serving /v1/audio/speech\n";
      namespace oai = vllm::entrypoints::openai;
      oai::OpenAIServingModels speech_models(speech_served_name);
      oai::ApiServer speech_server(speech_models, vllm::Version());
      speech_server.set_synthesizer(
          [speech_only](const vllm::openai::SpeechRequest& req)
              -> vllm::openai::SpeechResponse {
            return vllm::openai::SynthesizeSpeechRequest(*speech_only, req);
          },
          caps);
      std::cerr << "server: listening on http://" << args.host << ":" << args.port
                << " (model '" << speech_served_name << "')\n";
      vllm::platform::ConsoleShutdown shutdown_on_signal(
          [&]() { speech_server.stop(); });
      if (!speech_server.listen(args.host, args.port)) {
        std::cerr << "server: failed to bind " << args.host << ":" << args.port << "\n";
        return 1;
      }
      return 0;
    }

    const fs::path dir = NativeUtf8Path(args.model_dir);
    std::string config_path = PathUtf8(dir / "config.json");
    // cua-s1-forms models (MODEL-CUA-S1-FORMS) ship cua-s1-forms.json
    // instead of config.json. Fall back so the dispatch and loader can
    // resolve it.
    if (!fs::exists(config_path)) {
      const std::string cuas1_path =
          PathUtf8(dir / "cua-s1-forms.json");
      if (fs::exists(cuas1_path)) {
        config_path = cuas1_path;
      }
    }
    // Laya models (MODEL-LAYA) ship rl_agent_config.json instead of
    // config.json. Fall back so the dispatch and loader can resolve it.
    if (!fs::exists(config_path)) {
      const std::string rl_path =
          PathUtf8(dir / "rl_agent_config.json");
      if (fs::exists(rl_path)) {
        config_path = rl_path;
      }
    }
    // Root file, else the tokenizer/ subdirectory; the same resolution the
    // loader applies (ISSUE-LOCAL-01M3SDXGYYTS9FDXKZDAE24N0R).
    const std::string tokenizer_path =
        PathUtf8(vllm::ResolveTokenizerFile(dir, "tokenizer.json"));
    const std::string tokenizer_config_path =
        args.tokenizer_config.empty()
            ? PathUtf8(vllm::ResolveTokenizerFile(dir, "tokenizer_config.json"))
            : args.tokenizer_config;
    const std::string served_model_name =
        args.served_model_name.empty()
            ? PathUtf8(dir.has_filename() ? dir.filename()
                                          : dir.parent_path().filename())
            : args.served_model_name;

    // ── --offload-config, PARSED ONCE, AHEAD OF THE TASK BRANCH (#1135) ───────
    //
    // It used to be parsed after the branch below, in the text path only, so the
    // pooling and transcription-only paths built their engine parameters without
    // it. An embedding server started with `--offload-config` placed its weights
    // as though the flag were absent and said nothing. That dropped BOTH halves
    // of the document — vLLM's mirrored `uva`/`prefetch` weight offload and the
    // vllm.cpp `vllm_cpp` residency extension — and had done so since before the
    // extension existed.
    //
    // ONE PARSE, and every branch below now answers for the document: the text
    // path and the pooling path take both halves, and the transcription-only
    // path refuses a non-empty flag rather than dropping it.
    //
    // Both halves come out of the SAME string:
    // `parse_offload_config_json` reads vLLM's three keys and
    // `parse_weight_residency_extension_json` reads the extension and closes the
    // document against a typo at every level. Two parsers over one string is what
    // keeps `include/vllm/config/offload.h` a byte-faithful transcription of
    // `vllm/config/offload.py` (which has no disk tier) while the operator still
    // types one flag for one concept.
    //
    // Parsing HERE also refuses a malformed document, an unknown key or a
    // validator violation BEFORE the architecture is peeked and before the
    // `server: loading model from` line, rather than after both. Upstream's three
    // backend/field mismatches are WARNINGS in vLLM and stay warnings here.
    std::optional<vllm::OffloadConfig> parsed_offload_config;
    std::optional<vllm::WeightResidencyConfig> parsed_weight_residency;
    if (!args.offload_config.empty()) {
      vllm::OffloadConfig off_cfg =
          vllm::parse_offload_config_json(args.offload_config);
      off_cfg.Validate();
      for (const std::string& w : off_cfg.warnings) {
        std::fprintf(stderr, "[vllm.cpp] offload_config: %s\n", w.c_str());
      }
      parsed_offload_config = std::move(off_cfg);
      // ENG-RESIDENCY-CONFIG (#1110): the SAME document also carries the
      // vllm.cpp-original `vllm_cpp` key, which governs the tier BELOW vLLM's —
      // weights borrowed out of the file mapping rather than moved to host RAM.
      //
      // Its parser REFUSES a key it does not know, which the mirrored parser does
      // not do, and that refusal is load-bearing: a silently ignored
      // `{"vllm_cpp":{"mmapp":...}}` starts a server running this tier at its
      // defaults and is met as an out-of-memory kill instead of an error.
      vllm::WeightResidencyConfig res_cfg =
          vllm::parse_weight_residency_extension_json(args.offload_config);
      if (!res_cfg.empty()) parsed_weight_residency = std::move(res_cfg);
    }

    // ── TASK DISPATCH (ARCH-ONE-SURFACE ROW 1): a model dir whose
    // architectures resolve to a SupportsTranscription-ONLY registration
    // (Parakeet CTC/RNNT/TDT) serves /v1/audio/transcriptions through the ONE
    // library seam — the same ParakeetTranscriber vllm_transcribe drives — and
    // registers NO generate routes (vLLM's task-conditional registration,
    // api_server.py:255-265). Every other model takes the text path below,
    // byte-identical to before. ────────────────────────────────────────────────
    {
      bool transcription_only = false;
      const std::vector<std::string> archs =
          vllm::PeekHfArchitectures(config_path);
      if (!archs.empty()) {
        try {
          transcription_only =
              vllm::ModelRegistry::Resolve(std::span<const std::string>(archs))
                  .info.supports_transcription_only;
        } catch (const std::exception&) {
          transcription_only = false;  // unknown arch: the text path diagnoses
        }
      }
      // ── GLINER2.5 NER TASK DISPATCH (MODEL-GLINER25): a model dir whose
      // architectures resolve to "BoundaryExtractor" (GLiNER2.5) serves
      // /v1/ner and /v1/systemone through the ONE library seam
      // (Gliner2NerInference — the same path vllm_gliner_ner drives) and
      // registers NO generate or embedding routes. This check runs BEFORE
      // the pooling_model check because BoundaryExtractor IS a pooling model
      // (is_pooling_model = true), but it needs the NER server, not the
      // embedding server.
      bool gliner2_model = false;
      if (!archs.empty()) {
        try {
          gliner2_model =
              vllm::ModelRegistry::Resolve(std::span<const std::string>(archs))
                  .architecture == "BoundaryExtractor";
        } catch (const std::exception&) {
          gliner2_model = false;
        }
      }
      if (gliner2_model) {
        std::cerr << "server: GLiNER2.5 model (" << archs[0]
                  << "); serving /v1/ner and /v1/systemone\n";
        vllm::entrypoints::EngineParams ner_params;
        ner_params.block_size = args.block_size;
        ner_params.num_blocks = args.num_blocks;
        ner_params.gpu_memory_utilization = args.gpu_memory_utilization;
        ner_params.kv_cache_memory_bytes = args.kv_cache_memory_bytes;
        ner_params.max_model_len = args.max_model_len;
        ner_params.max_num_seqs = args.max_num_seqs;
        ner_params.max_num_batched_tokens = args.max_num_batched_tokens;
        ner_params.enable_prefix_caching = args.enable_prefix_caching;
        ner_params.offload_config = parsed_offload_config;
        ner_params.weight_residency = parsed_weight_residency;
        auto loaded_ner = std::shared_ptr<vllm::entrypoints::LoadedEngine>(
            vllm::entrypoints::LoadedEngine::FromModelDir(args.model_dir,
                                                          ner_params));
        namespace oai = vllm::entrypoints::openai;
        oai::OpenAIServingModels ner_models(served_model_name);
        oai::ApiServer ner_server(ner_models, vllm::Version());
        auto ner_mutex = std::make_shared<std::mutex>();
        ner_server.set_ner(
            [loaded_ner, ner_mutex](
                const std::string& text,
                const std::vector<std::string>& labels,
                float threshold, int64_t max_width)
                -> oai::ApiServer::NerResult {
              std::lock_guard<std::mutex> lock(*ner_mutex);
              const vllm::LoadedModel& model = loaded_ner->loaded_model();
              const vllm::tok::Tokenizer& tokenizer = loaded_ner->tokenizer();
              vllm::gliner2::NerParams params;
              params.threshold = threshold;
              params.max_width = (max_width > 0) ? max_width : 12;
              vllm::Gliner2NerResult result =
                  vllm::Gliner2NerInference(model, tokenizer, text,
                                            labels, params);
              oai::ApiServer::NerResult out;
              out.prompt_tokens = static_cast<int64_t>(
                  tokenizer.Encode(text).size());
              for (const vllm::gliner2::NerEntity& e : result.entities) {
                out.entities.push_back(
                    {e.label, e.text, e.char_start, e.char_end,
                     e.confidence});
              }
              return out;
            });
        std::cerr << "server: listening on http://" << args.host << ":"
                  << args.port << "\n";
        vllm::platform::ConsoleShutdown shutdown_on_signal(
            [&]() { ner_server.stop(); });
        if (!ner_server.listen(args.host, args.port)) {
          std::cerr << "server: failed to bind " << args.host << ":"
                    << args.port << "\n";
          return 1;
        }
        return 0;
      }

      // ── CUA-S1-FORMS SCORE DISPATCH (MODEL-CUA-S1-FORMS): a model dir
      // whose architectures resolve to "CuaS1Forms" serves /v1/score
      // through the score callback (CuaS1ScoreInference — the same path
      // the C ABI will drive) and registers NO generate, embedding, NER,
      // or systemone routes. This check runs BEFORE the pooling_model check
      // because CuaS1Forms IS a pooling model (is_pooling_model = true),
      // but it needs the score server, not the embedding server.
      bool cuas1_model = false;
      if (!archs.empty()) {
        try {
          cuas1_model =
              vllm::ModelRegistry::Resolve(std::span<const std::string>(archs))
                  .architecture == "CuaS1Forms";
        } catch (const std::exception&) {
          cuas1_model = false;
        }
      }
      if (cuas1_model) {
        std::cerr << "server: cua-s1-forms model (" << archs[0]
                  << "); serving /v1/score\n";
        vllm::entrypoints::EngineParams score_params;
        score_params.block_size = args.block_size;
        score_params.num_blocks = args.num_blocks;
        score_params.gpu_memory_utilization = args.gpu_memory_utilization;
        score_params.kv_cache_memory_bytes = args.kv_cache_memory_bytes;
        score_params.max_model_len = args.max_model_len;
        score_params.max_num_seqs = args.max_num_seqs;
        score_params.max_num_batched_tokens = args.max_num_batched_tokens;
        score_params.enable_prefix_caching = args.enable_prefix_caching;
        score_params.offload_config = parsed_offload_config;
        score_params.weight_residency = parsed_weight_residency;
        auto loaded_score = std::shared_ptr<vllm::entrypoints::LoadedEngine>(
            vllm::entrypoints::LoadedEngine::FromModelDir(args.model_dir,
                                                          score_params));
        namespace oai = vllm::entrypoints::openai;
        oai::OpenAIServingModels score_models(served_model_name);
        oai::ApiServer score_server(score_models, vllm::Version());
        auto score_mutex = std::make_shared<std::mutex>();
        score_server.set_score(
            [loaded_score, score_mutex](
                const std::string& context,
                const std::vector<std::string>& options)
                -> oai::ApiServer::ScoreResult {
              std::lock_guard<std::mutex> lock(*score_mutex);
              const vllm::LoadedModel& model =
                  loaded_score->loaded_model();
              vllm::cua_s1::CuaS1ScoreResult result =
                  vllm::CuaS1ScoreInference(model, context, options);
              oai::ApiServer::ScoreResult out;
              out.probabilities = std::move(result.probabilities);
              out.winner = result.winner;
              out.confidence = result.confidence;
              out.prompt_tokens = result.prompt_tokens;
              return out;
            });
        std::cerr << "server: listening on http://" << args.host << ":"
                  << args.port << "\n";
        vllm::platform::ConsoleShutdown shutdown_on_signal(
            [&]() { score_server.stop(); });
        if (!score_server.listen(args.host, args.port)) {
          std::cerr << "server: failed to bind " << args.host << ":"
                    << args.port << "\n";
          return 1;
        }
        return 0;
      }

      // ── LAYA DECISION TASK DISPATCH (MODEL-LAYA): a model dir whose
      // architectures resolve to "LayaModel" serves /v1/systemone through the
      // decision callback (LayaInference — the same path the C ABI will drive)
      // and registers NO generate, embedding, or NER routes. This check runs
      // BEFORE the pooling_model check because LayaModel IS a pooling model
      // (is_pooling_model = true), but it needs the decision server, not the
      // embedding server.
      bool laya_model = false;
      if (!archs.empty()) {
        try {
          laya_model =
              vllm::ModelRegistry::Resolve(std::span<const std::string>(archs))
                  .architecture == "LayaModel";
        } catch (const std::exception&) {
          laya_model = false;
        }
      }
      if (laya_model) {
        std::cerr << "server: Laya decision model (" << archs[0]
                  << "); serving /v1/systemone\n";
        vllm::entrypoints::EngineParams decision_params;
        decision_params.block_size = args.block_size;
        decision_params.num_blocks = args.num_blocks;
        decision_params.gpu_memory_utilization = args.gpu_memory_utilization;
        decision_params.kv_cache_memory_bytes = args.kv_cache_memory_bytes;
        decision_params.max_model_len = args.max_model_len;
        decision_params.max_num_seqs = args.max_num_seqs;
        decision_params.max_num_batched_tokens = args.max_num_batched_tokens;
        decision_params.enable_prefix_caching = args.enable_prefix_caching;
        decision_params.offload_config = parsed_offload_config;
        decision_params.weight_residency = parsed_weight_residency;
        auto loaded_decision = std::shared_ptr<vllm::entrypoints::LoadedEngine>(
            vllm::entrypoints::LoadedEngine::FromModelDir(args.model_dir,
                                                          decision_params));
        namespace oai = vllm::entrypoints::openai;
        oai::OpenAIServingModels decision_models(served_model_name);
        oai::ApiServer decision_server(decision_models, vllm::Version());
        auto decision_mutex = std::make_shared<std::mutex>();
        decision_server.set_decision(
            [loaded_decision, decision_mutex](
                const std::string& state,
                const std::string& qtype,
                const std::string& instructions,
                const std::vector<std::string>& options)
                -> oai::ApiServer::DecisionResult {
              std::lock_guard<std::mutex> lock(*decision_mutex);
              const vllm::LoadedModel& model =
                  loaded_decision->loaded_model();
              const vllm::tok::Tokenizer& tokenizer =
                  loaded_decision->tokenizer();
              vllm::LayaDecisionResult result =
                  vllm::LayaInference(model, tokenizer, state, qtype,
                                      instructions, options);
              oai::ApiServer::DecisionResult out;
              out.scores = std::move(result.scores);
              out.act_logits = std::move(result.act_logits);
              out.prompt_tokens = result.prompt_tokens;
              return out;
            });
        std::cerr << "server: listening on http://" << args.host << ":"
                  << args.port << "\n";
        vllm::platform::ConsoleShutdown shutdown_on_signal(
            [&]() { decision_server.stop(); });
        if (!decision_server.listen(args.host, args.port)) {
          std::cerr << "server: failed to bind " << args.host << ":"
                    << args.port << "\n";
          return 1;
        }
        return 0;
      }

      // ── NIMBLE DECISION TASK DISPATCH (MODEL-NIMBLE): a model dir whose
      // architectures resolve to "NimbleModel" (scripts/convert-nimble.py
      // output) serves /v1/systemone through the REQUEST-level seam
      // (NimbleDecide, the same function vllm_decide calls), and registers no
      // generate, embedding, or NER routes. Checked before the pooling check
      // for the same reason as kev below.
      bool nimble_model = false;
      if (!archs.empty()) {
        try {
          nimble_model =
              vllm::ModelRegistry::Resolve(std::span<const std::string>(archs))
                  .architecture == "NimbleModel";
        } catch (const std::exception&) {
          nimble_model = false;
        }
      }
      if (nimble_model) {
        std::cerr << "server: nimble decision model (" << archs[0]
                  << "); serving /v1/systemone\n";
        vllm::entrypoints::EngineParams decision_params;
        decision_params.block_size = args.block_size;
        decision_params.num_blocks = args.num_blocks;
        decision_params.gpu_memory_utilization = args.gpu_memory_utilization;
        decision_params.kv_cache_memory_bytes = args.kv_cache_memory_bytes;
        decision_params.max_model_len = args.max_model_len;
        decision_params.max_num_seqs = args.max_num_seqs;
        decision_params.max_num_batched_tokens = args.max_num_batched_tokens;
        decision_params.enable_prefix_caching = args.enable_prefix_caching;
        decision_params.offload_config = parsed_offload_config;
        decision_params.weight_residency = parsed_weight_residency;
        auto loaded_decision = std::shared_ptr<vllm::entrypoints::LoadedEngine>(
            vllm::entrypoints::LoadedEngine::FromModelDir(args.model_dir,
                                                          decision_params));
        namespace oai = vllm::entrypoints::openai;
        oai::OpenAIServingModels decision_models(served_model_name);
        oai::ApiServer decision_server(decision_models, vllm::Version());
        auto decision_mutex = std::make_shared<std::mutex>();
        decision_server.set_systemone_request(
            [loaded_decision, decision_mutex](const nlohmann::ordered_json& body)
                -> nlohmann::ordered_json {
              std::lock_guard<std::mutex> lock(*decision_mutex);
              vllm::NimbleResponse r = vllm::NimbleDecide(
                  loaded_decision->loaded_model(), loaded_decision->tokenizer(),
                  body);
              nlohmann::ordered_json out = nlohmann::ordered_json::object();
              out["answers"] = std::move(r.answers);
              out["usage"] = {{"input_tokens", r.input_tokens},
                              {"output_tokens", 0}};
              return out;
            });
        std::cerr << "server: listening on http://" << args.host << ":"
                  << args.port << "\n";
        vllm::platform::ConsoleShutdown shutdown_on_signal(
            [&]() { decision_server.stop(); });
        if (!decision_server.listen(args.host, args.port)) {
          std::cerr << "server: failed to bind " << args.host << ":"
                    << args.port << "\n";
          return 1;
        }
        return 0;
      }

      // ── KEV DECISION TASK DISPATCH (MODEL-KEV): a model dir whose
      // architectures resolve to "KevModel" serves /v1/systemone through the
      // decision callback (KevInference — the same path the C ABI will drive)
      // and registers NO generate, embedding, or NER routes. This check runs
      // BEFORE the pooling_model check because KevModel IS a pooling model
      // (is_pooling_model = true), but it needs the decision server, not the
      // embedding server.
      bool kev_model = false;
      if (!archs.empty()) {
        try {
          kev_model =
              vllm::ModelRegistry::Resolve(std::span<const std::string>(archs))
                  .architecture == "KevModel";
        } catch (const std::exception&) {
          kev_model = false;
        }
      }
      if (kev_model) {
        std::cerr << "server: kev decision model (" << archs[0]
                  << "); serving /v1/systemone\n";
        vllm::entrypoints::EngineParams decision_params;
        decision_params.block_size = args.block_size;
        decision_params.num_blocks = args.num_blocks;
        decision_params.gpu_memory_utilization = args.gpu_memory_utilization;
        decision_params.kv_cache_memory_bytes = args.kv_cache_memory_bytes;
        decision_params.max_model_len = args.max_model_len;
        decision_params.max_num_seqs = args.max_num_seqs;
        decision_params.max_num_batched_tokens = args.max_num_batched_tokens;
        decision_params.enable_prefix_caching = args.enable_prefix_caching;
        decision_params.offload_config = parsed_offload_config;
        decision_params.weight_residency = parsed_weight_residency;
        auto loaded_decision = std::shared_ptr<vllm::entrypoints::LoadedEngine>(
            vllm::entrypoints::LoadedEngine::FromModelDir(args.model_dir,
                                                          decision_params));
        namespace oai = vllm::entrypoints::openai;
        oai::OpenAIServingModels decision_models(served_model_name);
        oai::ApiServer decision_server(decision_models, vllm::Version());
        auto decision_mutex = std::make_shared<std::mutex>();
        decision_server.set_decision(
            [loaded_decision, decision_mutex](
                const std::string& state,
                const std::string& qtype,
                const std::string& instructions,
                const std::vector<std::string>& options)
                -> oai::ApiServer::DecisionResult {
              std::lock_guard<std::mutex> lock(*decision_mutex);
              const vllm::LoadedModel& model =
                  loaded_decision->loaded_model();
              const vllm::tok::Tokenizer& tokenizer =
                  loaded_decision->tokenizer();
              vllm::KevDecisionResult result =
                  vllm::KevInference(model, tokenizer, state, qtype,
                                     instructions, options);
              oai::ApiServer::DecisionResult out;
              out.scores = std::move(result.scores);
              out.act_logits = std::move(result.act_logits);
              out.prompt_tokens = result.prompt_tokens;
              return out;
            });
        std::cerr << "server: listening on http://" << args.host << ":"
                  << args.port << "\n";
        vllm::platform::ConsoleShutdown shutdown_on_signal(
            [&]() { decision_server.stop(); });
        if (!decision_server.listen(args.host, args.port)) {
          std::cerr << "server: failed to bind " << args.host << ":"
                    << args.port << "\n";
          return 1;
        }
        return 0;
      }

      // ── XOR DECISION TASK DISPATCH (MODEL-XOR):
      bool xor_model = false;
      if (!archs.empty()) {
        try {
          xor_model =
              vllm::ModelRegistry::Resolve(std::span<const std::string>(archs))
                  .architecture == "XorModel";
        } catch (const std::exception&) {
          xor_model = false;
        }
      }
      if (xor_model) {
        std::cerr << "server: xor decision model (" << archs[0]
                  << "); serving /v1/systemone\n";
        vllm::entrypoints::EngineParams decision_params;
        decision_params.block_size = args.block_size;
        decision_params.num_blocks = args.num_blocks;
        decision_params.gpu_memory_utilization = args.gpu_memory_utilization;
        decision_params.kv_cache_memory_bytes = args.kv_cache_memory_bytes;
        decision_params.max_model_len = args.max_model_len;
        decision_params.max_num_seqs = args.max_num_seqs;
        decision_params.max_num_batched_tokens = args.max_num_batched_tokens;
        decision_params.enable_prefix_caching = args.enable_prefix_caching;
        decision_params.offload_config = parsed_offload_config;
        decision_params.weight_residency = parsed_weight_residency;
        auto loaded_decision = std::shared_ptr<vllm::entrypoints::LoadedEngine>(
            vllm::entrypoints::LoadedEngine::FromModelDir(args.model_dir,
                                                          decision_params));
        namespace oai = vllm::entrypoints::openai;
        oai::OpenAIServingModels decision_models(served_model_name);
        oai::ApiServer decision_server(decision_models, vllm::Version());
        auto decision_mutex = std::make_shared<std::mutex>();
        decision_server.set_decision(
            [loaded_decision, decision_mutex](
                const std::string& state,
                const std::string& qtype,
                const std::string& instructions,
                const std::vector<std::string>& options)
                -> oai::ApiServer::DecisionResult {
              std::lock_guard<std::mutex> lock(*decision_mutex);
              const vllm::LoadedModel& model =
                  loaded_decision->loaded_model();
              const vllm::tok::Tokenizer& tokenizer =
                  loaded_decision->tokenizer();
              vllm::XorDecisionResult result =
                  vllm::XorInference(model, tokenizer, state, qtype,
                                     instructions, options);
              oai::ApiServer::DecisionResult out;
              out.scores = std::move(result.scores);
              out.prompt_tokens = result.prompt_tokens;
              return out;
            });
        std::cerr << "server: listening on http://" << args.host << ":"
                  << args.port << "\n";
        vllm::platform::ConsoleShutdown shutdown_on_signal(
            [&]() { decision_server.stop(); });
        if (!decision_server.listen(args.host, args.port)) {
          std::cerr << "server: failed to bind " << args.host << ":"
                    << args.port << "\n";
          return 1;
        }
        return 0;
      }

      // ── CLM DECISION TASK DISPATCH (MODEL-CLM):
      bool clm_model = false;
      if (!archs.empty()) {
        try {
          clm_model =
              vllm::ModelRegistry::Resolve(std::span<const std::string>(archs))
                  .architecture == "ClmModel";
        } catch (const std::exception&) {
          clm_model = false;
        }
      }
      if (clm_model) {
        std::cerr << "server: clm decision model (" << archs[0]
                  << "); serving /v1/systemone\n";
        vllm::entrypoints::EngineParams decision_params;
        decision_params.block_size = args.block_size;
        decision_params.num_blocks = args.num_blocks;
        decision_params.gpu_memory_utilization = args.gpu_memory_utilization;
        decision_params.kv_cache_memory_bytes = args.kv_cache_memory_bytes;
        decision_params.max_model_len = args.max_model_len;
        decision_params.max_num_seqs = args.max_num_seqs;
        decision_params.max_num_batched_tokens = args.max_num_batched_tokens;
        decision_params.enable_prefix_caching = args.enable_prefix_caching;
        decision_params.offload_config = parsed_offload_config;
        decision_params.weight_residency = parsed_weight_residency;
        auto loaded_decision = std::shared_ptr<vllm::entrypoints::LoadedEngine>(
            vllm::entrypoints::LoadedEngine::FromModelDir(args.model_dir,
                                                          decision_params));
        namespace oai = vllm::entrypoints::openai;
        oai::OpenAIServingModels decision_models(served_model_name);
        oai::ApiServer decision_server(decision_models, vllm::Version());
        auto decision_mutex = std::make_shared<std::mutex>();
        // Request-level, like Nimble: ClmDecide renders the raw state and
        // criteria JSON the way the reference does, which the per-question
        // DecisionFn strings cannot carry, and answers in the reference's
        // own shape.
        decision_server.set_systemone_request(
            [loaded_decision, decision_mutex](const nlohmann::ordered_json& body)
                -> nlohmann::ordered_json {
              std::lock_guard<std::mutex> lock(*decision_mutex);
              return vllm::ClmDecide(loaded_decision->loaded_model(),
                                     loaded_decision->tokenizer(), body);
            });
        std::cerr << "server: listening on http://" << args.host << ":"
                  << args.port << "\n";
        vllm::platform::ConsoleShutdown shutdown_on_signal(
            [&]() { decision_server.stop(); });
        if (!decision_server.listen(args.host, args.port)) {
          std::cerr << "server: failed to bind " << args.host << ":"
                    << args.port << "\n";
          return 1;
        }
        return 0;
      }

      // ── GLINER25-DECIDE DECISION TASK DISPATCH (MODEL-GLINER25-DECIDE):
      bool gliner25_decide_model = false;
      if (!archs.empty()) {
        try {
          gliner25_decide_model =
              vllm::ModelRegistry::Resolve(std::span<const std::string>(archs))
                  .architecture == "SpanExtractor";
        } catch (const std::exception&) {
          gliner25_decide_model = false;
        }
      }
      if (gliner25_decide_model) {
        std::cerr << "server: GLiNER2.5-Decide decision model ("
                  << archs[0] << "); serving /v1/systemone\n";
        vllm::entrypoints::EngineParams decision_params;
        decision_params.block_size = args.block_size;
        decision_params.num_blocks = args.num_blocks;
        decision_params.gpu_memory_utilization = args.gpu_memory_utilization;
        decision_params.kv_cache_memory_bytes = args.kv_cache_memory_bytes;
        decision_params.max_model_len = args.max_model_len;
        decision_params.max_num_seqs = args.max_num_seqs;
        decision_params.max_num_batched_tokens = args.max_num_batched_tokens;
        decision_params.enable_prefix_caching = args.enable_prefix_caching;
        decision_params.offload_config = parsed_offload_config;
        decision_params.weight_residency = parsed_weight_residency;
        auto loaded_decision = std::shared_ptr<vllm::entrypoints::LoadedEngine>(
            vllm::entrypoints::LoadedEngine::FromModelDir(args.model_dir,
                                                          decision_params));
        namespace oai = vllm::entrypoints::openai;
        oai::OpenAIServingModels decision_models(served_model_name);
        oai::ApiServer decision_server(decision_models, vllm::Version());
        auto decision_mutex = std::make_shared<std::mutex>();
        decision_server.set_decision(
            [loaded_decision, decision_mutex](
                const std::string& state,
                const std::string& qtype,
                const std::string& instructions,
                const std::vector<std::string>& options)
                -> oai::ApiServer::DecisionResult {
              std::lock_guard<std::mutex> lock(*decision_mutex);
              const vllm::LoadedModel& model =
                  loaded_decision->loaded_model();
              const vllm::tok::Tokenizer& tokenizer =
                  loaded_decision->tokenizer();
              vllm::Gliner25DecideResult result =
                  vllm::Gliner25DecideInference(model, tokenizer, state,
                                                 qtype, instructions, options);
              oai::ApiServer::DecisionResult out;
              out.scores = std::move(result.scores);
              out.prompt_tokens = result.prompt_tokens;
              return out;
            });
        std::cerr << "server: listening on http://" << args.host << ":"
                  << args.port << "\n";
        vllm::platform::ConsoleShutdown shutdown_on_signal(
            [&]() { decision_server.stop(); });
        if (!decision_server.listen(args.host, args.port)) {
          std::cerr << "server: failed to bind " << args.host << ":"
                    << args.port << "\n";
          return 1;
        }
        return 0;
      }

      // ── POOLING TASK DISPATCH (ARCH-ONE-SURFACE ROW 6): a model dir whose
      // architectures resolve to a POOLING registration (is_pooling_model,
      // e.g. "LlamaModel" — vLLM _EMBEDDING_MODELS registry.py:230) serves
      // /v1/embeddings through the ONE engine path (LoadedEngine ->
      // LLMEngine::embed -> registry forward -> PoolingRunner) — the same
      // path vllm_embed drives — and registers NO generate routes (vLLM's
      // task-conditional registration, api_server.py:255-265). ──────────────
      bool pooling_model = false;
      if (!archs.empty()) {
        try {
          pooling_model =
              vllm::ModelRegistry::Resolve(std::span<const std::string>(archs))
                  .info.is_pooling_model;
        } catch (const std::exception&) {
          pooling_model = false;  // unknown arch: the text path diagnoses
        }
      }
      if (pooling_model) {
        std::cerr << "server: pooling (embedding) model (" << archs[0]
                  << "); serving /v1/embeddings\n";
        vllm::entrypoints::EngineParams embed_params;
        embed_params.block_size = args.block_size;
        embed_params.num_blocks = args.num_blocks;
        embed_params.gpu_memory_utilization = args.gpu_memory_utilization;
        embed_params.kv_cache_memory_bytes = args.kv_cache_memory_bytes;
        embed_params.max_model_len = args.max_model_len;
        embed_params.max_num_seqs = args.max_num_seqs;
        embed_params.max_num_batched_tokens = args.max_num_batched_tokens;
        embed_params.enable_prefix_caching = args.enable_prefix_caching;
        // #1135: BOTH halves of `--offload-config`, on the same terms as the text
        // path. This branch reaches `LoadedEngine::FromModelDir`, which installs
        // the residency document and the weight offloader ahead of every path and
        // weight operation it performs — so an embedding model is loaded by the
        // same loader and nothing about pooling makes either half inapplicable.
        // Before this, both were dropped here in silence.
        //
        // The other engine flags this block still drops, `--device` among them,
        // are #1196. They are the same shape over a wider set and belong to the
        // row that owns this dispatch.
        embed_params.offload_config = parsed_offload_config;
        embed_params.weight_residency = parsed_weight_residency;
        auto loaded_embed = std::shared_ptr<vllm::entrypoints::LoadedEngine>(
            vllm::entrypoints::LoadedEngine::FromModelDir(args.model_dir,
                                                          embed_params));
        namespace oai = vllm::entrypoints::openai;
        oai::OpenAIServingModels embed_models(served_model_name);
        oai::ApiServer embed_server(embed_models, vllm::Version());
        auto embed_mutex = std::make_shared<std::mutex>();
        auto embed_counter = std::make_shared<std::atomic<uint64_t>>(0);
        embed_server.set_embedder(
            [loaded_embed, embed_mutex, embed_counter](
                const std::vector<std::string>& inputs) {
              // Serialize batches: the pooling path drives the SYNCHRONOUS
              // LLMEngine (async scheduling resolves OFF for pooling models).
              std::lock_guard<std::mutex> lock(*embed_mutex);
              oai::ApiServer::EmbeddingBatch batch;
              for (const std::string& text : inputs) {
                std::vector<int32_t> ids =
                    loaded_embed->tokenizer().EncodeWithSpecialTokens(text);
                if (ids.empty()) {
                  throw std::runtime_error(
                      "input tokenized to an empty prompt");
                }
                batch.prompt_tokens += static_cast<int64_t>(ids.size());
                vllm::RequestOutput ro = loaded_embed->engine().embed(
                    std::move(ids), vllm::PoolingParams{},
                    "embd-" + std::to_string(embed_counter->fetch_add(1)));
                if (!ro.finished || !ro.pooling_output.has_value()) {
                  throw std::runtime_error(
                      "engine produced no pooled output");
                }
                batch.embeddings.push_back(std::move(*ro.pooling_output));
              }
              return batch;
            });
        std::cerr << "server: listening on http://" << args.host << ":"
                  << args.port << "\n";
        vllm::platform::ConsoleShutdown shutdown_on_signal(
            [&]() { embed_server.stop(); });
        if (!embed_server.listen(args.host, args.port)) {
          std::cerr << "server: failed to bind " << args.host << ":"
                    << args.port << "\n";
          return 1;
        }
        return 0;
      }

      if (transcription_only) {
        // #1135, and this arm is a REFUSAL rather than a wiring. The
        // transcription stack has no seam either half of `--offload-config`
        // could reach: `ParakeetTranscriber::FromDir` builds no `EngineParams`
        // and calls no `LoadedEngine::FromModelDir`, so this path runs no
        // `SetWeightResidencyConfig`, no `CreateWeightOffloader`, no GGUF
        // mapping and no expert slot store. There is no field of either half
        // that any code here could read.
        //
        // AGENTS.md: refuse an unimplemented arm with a message that names the
        // missing part, and record the arm as owed. Accepting the flag and
        // warning would leave a server running while it holds a placement
        // instruction it does not follow, which is the failure #1135 was filed
        // about. The document is still PARSED above, so a typo in it is refused
        // here on the same terms as everywhere else.
        if (!args.offload_config.empty()) {
          throw std::invalid_argument(
              "--offload-config is not supported on a transcription-only model "
              "(" + archs[0] +
              "). THE MISSING PART: this path serves /v1/audio/transcriptions "
              "through ParakeetTranscriber, which loads its own weights and "
              "never builds an engine, so neither vLLM's uva/prefetch weight "
              "offload nor vllm.cpp's vllm_cpp weight-residency tier has a call "
              "site on it. Both halves of the document would be accepted and "
              "then ignored, which is issue #1135. Remove the flag, or serve a "
              "text-generation or embedding model, which honour it. Tracked as "
              "issue #1195");
        }
        std::cerr << "server: transcription-only model (" << archs[0]
                  << "); serving /v1/audio/transcriptions\n";
        auto transcriber =
            std::make_shared<vllm::multimodal::ParakeetTranscriber>(
                vllm::multimodal::ParakeetTranscriber::FromDir(args.model_dir));
        namespace oai = vllm::entrypoints::openai;
        oai::OpenAIServingModels asr_models(served_model_name);
        oai::ApiServer asr_server(asr_models, vllm::Version());
        asr_server.set_transcriber(
            [transcriber](const uint8_t* wav, size_t n) {
              return transcriber->TranscribeWavBytes(wav, n);
            });
        std::cerr << "server: listening on http://" << args.host << ":"
                  << args.port << "\n";
        vllm::platform::ConsoleShutdown shutdown_on_signal(
            [&]() { asr_server.stop(); });
        if (!asr_server.listen(args.host, args.port)) {
          std::cerr << "server: failed to bind " << args.host << ":"
                    << args.port << "\n";
          return 1;
        }
        return 0;
      }
    }

    // ── Load the model + build the full engine stack via the shared loader
    // (src/vllm/entrypoints/model_loader.cpp) — the same path the C ABI drives.
    // It loads config.json + tokenizer.json + *.safetensors and wires the M1.8
    // LLMEngine over Scheduler + runner + KV + processors. ─────────────────────
    std::cerr << "server: loading model from " << args.model_dir << " (config "
              << config_path << ", tokenizer " << tokenizer_path << ")\n";
    vllm::entrypoints::EngineParams engine_params;
    engine_params.block_size = args.block_size;
    engine_params.num_blocks = args.num_blocks;
    engine_params.gpu_memory_utilization = args.gpu_memory_utilization;
    engine_params.kv_cache_memory_bytes = args.kv_cache_memory_bytes;
    engine_params.kv_cache_dtype = args.kv_cache_dtype;
    engine_params.max_model_len = args.max_model_len;  // 0 => from config.
    engine_params.max_num_seqs = args.max_num_seqs;
    engine_params.max_num_batched_tokens = args.max_num_batched_tokens;
    engine_params.enable_prefix_caching = args.enable_prefix_caching;
    // #607 L2: the multimodal input limits go onto the ENGINE, not into a
    // server-local variable, so the C ABI and this server resolve one limit the
    // same way and the chat seam can borrow the engine's copy.
    engine_params.multimodal = args.multimodal;
    // --device: explicit device selection (ARCH-ONE-SURFACE ROW 8). "auto"
    // (default) keeps the accelerator-first probe byte-identical; an unknown
    // name throws HERE (a startup error), and an explicitly named ABSENT
    // device fails FromModelDir loudly — never a silent fallback
    // (vllm/config/device.py:61-66).
    engine_params.device = vllm::DeviceFromString(args.device);
    // Reject an unknown policy string (mirrors upstream SchedulingPolicy(value)).
    engine_params.policy = vllm::SchedulerPolicyFromString(args.scheduling_policy);
    // ENG-SGLANG-BEHAVIOR-FLAG (SW1): `lpm` needs prefix caching to have any
    // cache to match against; with APC explicitly off it degrades to fcfs
    // (the scheduler leaves arrival order intact). Warn once at load so the
    // no-op is visible (mirrors the spec's lpm+cache-off resolution).
    if (engine_params.policy == vllm::SchedulerPolicy::kLPM &&
        args.enable_prefix_caching.has_value() &&
        !args.enable_prefix_caching.value()) {
      std::cerr << "server: --scheduling-policy lpm has no effect with prefix "
                   "caching disabled; falling back to fcfs admission order\n";
    }
    // ENG-SGLANG-BEHAVIOR-FLAG SW3: jump-forward decoding. Unset => the default
    // (env-resolved, OFF); --[enable|disable]-jump-forward forces it, and
    // VT_ENABLE_JUMP_FORWARD still overrides at resolution time.
    engine_params.enable_jump_forward = args.enable_jump_forward;
    engine_params.disable_sliding_window = args.disable_sliding_window;
    // --kv-transfer-config: the external KV connector, mirroring vLLM's own
    // flag and JSON shape. Absent (default) leaves the optional unset, which is
    // the inert no-connector path the server has always run. A malformed
    // document, an unknown key/role, or a connector whose worker half cannot
    // move bytes on this device (the D1 guard, inside LoadedEngine) all throw
    // out of here and are reported at startup by the catch in main.
    if (!args.kv_transfer_config.empty()) {
      vllm::KVTransferConfig kv_cfg =
          vllm::ParseKVTransferConfigJson(args.kv_transfer_config);
      if (kv_cfg.kv_connector.has_value() &&
          !vllm::v1::kv_offload::KVConnectorFactory::IsRegistered(
              *kv_cfg.kv_connector)) {
        std::string msg = "unknown kv_connector \"" + *kv_cfg.kv_connector +
                          "\" (registered connectors: ";
        const std::vector<std::string> names =
            vllm::v1::kv_offload::KVConnectorFactory::RegisteredNames();
        for (size_t n = 0; n < names.size(); ++n) {
          if (n != 0) msg += ", ";
          msg += names[n];
        }
        msg += ")";
        throw std::invalid_argument(msg);
      }
      engine_params.kv_transfer_config = std::move(kv_cfg);
    }
    // --offload-config: WEIGHT offload (ENG-WEIGHT-OFFLOAD W0b) and the
    // vllm.cpp `vllm_cpp` residency tier (ENG-RESIDENCY-CONFIG). Both halves were
    // parsed and validated above, ahead of the task branch, so that the pooling
    // path gets them too and a malformed document is refused before the
    // architecture peek (#1135). Empty (default) leaves both optionals unset —
    // the byte-identical no-offload path.
    engine_params.offload_config = parsed_offload_config;
    engine_params.weight_residency = parsed_weight_residency;
    // --mmproj: the second GGUF (LOAD-GGUF-MMPROJ, #821). Empty leaves the
    // loader on its single-file path, byte-identically.
    engine_params.mmproj_path = args.mmproj_path;
    // --speculative-config: speculative decoding (SPEC-MTP I5d). Absent (default)
    // leaves the optional unset — the byte-identical no-speculation path. The
    // parse validates method/k here; n_predict + the resolved k are finalized in
    // LoadedEngine once the checkpoint's mtp_num_hidden_layers is known. A
    // malformed document or unsupported method throws and is reported at startup.
    if (!args.speculative_config.empty()) {
      engine_params.speculative_config =
          vllm::ParseSpeculativeConfigJson(args.speculative_config);
    }
    // Declared before LoadedEngine so reverse destruction order shuts down and
    // joins AsyncLLM before releasing its non-owning metrics logger pointer.
    std::unique_ptr<vllm::v1::metrics::PrometheusStatLogger> prom_logger;
    std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
        vllm::entrypoints::LoadedEngine::FromModelDir(args.model_dir,
                                                      engine_params);
    std::cerr << "server: prefix caching "
              << (loaded->prefix_caching_enabled() ? "enabled" : "disabled")
              << "\n";
    // #607 L3: what the zero limits actually FREED, read back off the loaded
    // MODEL rather than off `args`. The limits line above says what was asked
    // for; this one says what happened, and until L3 the two could differ
    // completely — the flag was accepted, every limit went to 0, and the tower
    // was loaded anyway. Printing from `args` would restore exactly that gap,
    // which is the mistake the weight-residency line records having made
    // (model_loader.cpp, `ActiveWeightResidencyConfig`). Silent when nothing was
    // skipped, so a text model and a multimodal model at their default limits
    // both print nothing new.
    {
      const std::vector<std::string> skipped = loaded->skipped_towers();
      if (!skipped.empty()) {
        std::cerr << "server: multimodal towers NOT loaded (every modality they "
                     "serve is at limit 0):";
        for (const std::string& stage : skipped) std::cerr << " " << stage;
        std::cerr << "\n";
      }
    }
    // W2: the production server uses AsyncLLM over EngineCoreProc's dedicated
    // engine thread. HTTP workers submit independently and stream from their
    // per-request collectors; no server-wide engine mutex remains.
    vllm::v1::AsyncLLM& engine = loaded->async_engine();
    const vllm::tok::Tokenizer& tokenizer = loaded->tokenizer();

    if (args.cuda_profile_graph_replays > 0) {
// DSR-ALLOW(ARCH-ONE-SURFACE): VT_BENCH_PROFILE_CONTROL is a build-option guard for the CUDA-graph-replay profiler, not a device fork; #189 moved it here verbatim from examples/server/main.cpp, which the DSR scanner never covered.
#if defined(VT_BENCH_PROFILE_CONTROL) && !defined(_WIN32)
      vt::cuda::ConfigureCudaGraphReplayProfiler(
          static_cast<uint32_t>(args.cuda_profile_graph_replays),
          static_cast<uint32_t>(args.cuda_profile_graph_batch));
      std::cerr << "[VT_CUDA_PROFILE] ready pid=" << CurrentProcessId()
                << " signal=SIGUSR2 target_replays="
                << args.cuda_profile_graph_replays << "\n";
#else
      throw std::invalid_argument(
          "--cuda-profile-graph-replays requires "
          "VLLM_CPP_BENCH_PROFILE_CONTROL=ON");
#endif
    }

    // ── OpenAI serving handlers. The chat handler is wired with the real chat
    // template (Task 3) when tokenizer_config.json carries one; otherwise it
    // keeps the default role-join fallback. ────────────────────────────────
    namespace oai = vllm::entrypoints::openai;
    oai::OpenAIServingModels models(served_model_name);
    oai::OpenAIServingCompletion completion(
        engine, served_model_name, args.enable_force_include_usage);

    oai::ChatPromptFn chat_prompt_fn = oai::DefaultChatPromptFallback;
    // Kept outside the try so the parser resolution below can sniff it when
    // --tool-call-parser/--reasoning-parser are "auto"; empty when the model
    // ships no template (auto then falls back to hermes / disabled).
    std::string chat_template;
    try {
      // The selection path (config first, then GGUF metadata) lives in
      // LoadChatTemplateForModel so a gate can drive it without starting a
      // server. `source` records where the template came from so the log does
      // not report a GGUF-loaded template as coming from tokenizer_config_path.
      std::string template_source;
      chat_template = vllm::entrypoints::LoadChatTemplateForModel(
          tokenizer_config_path, args.model_dir, template_source);
      const std::string bos =
          tokenizer.BosId() >= 0 ? tokenizer.Decode({tokenizer.BosId()}) : "";
      const std::string eos =
          tokenizer.EosId() >= 0 ? tokenizer.Decode({tokenizer.EosId()}) : "";
      chat_prompt_fn = vllm::entrypoints::MakeChatTemplatePromptFn(
          chat_template, bos, eos,
          vllm::entrypoints::DefaultChatTemplateKwargs(args.enable_thinking));
      std::cerr << "server: using chat template (" << chat_template.size()
                << " chars) from " << template_source
                << " enable_thinking="
                << (args.enable_thinking.has_value()
                        ? (*args.enable_thinking ? "true" : "false")
                        : "unset (the template's own default)")
                << "\n";
    } catch (const std::exception& e) {
      std::cerr << "server: no chat template (" << e.what()
                << "); falling back to the simple role-join prompt\n";
    }
    // Dialect selection. Defaults reproduce the previously hardcoded pair
    // ("hermes", "") exactly; an unknown name throws std::invalid_argument
    // listing every registered parser and aborts startup, rather than leaving
    // tool/reasoning parsing silently off for the life of the process.
    const std::string tool_parser_name =
        oai::ResolveToolParserName(args.tool_call_parser, chat_template);
    const std::string reasoning_parser_name =
        oai::ResolveReasoningParserName(args.reasoning_parser, chat_template);
    std::cerr << "server: tool-call parser "
              << (tool_parser_name.empty() ? "disabled" : tool_parser_name)
              << ", reasoning parser "
              << (reasoning_parser_name.empty() ? "disabled"
                                                : reasoning_parser_name)
              << "\n";
    oai::OpenAIServingChat chat(engine, served_model_name, chat_prompt_fn,
                                tool_parser_name, reasoning_parser_name,
                                args.enable_force_include_usage);

    // SAMPLE-BEAM (C7): enable use_beam_search on the production AsyncLLM path.
    // Both handlers need the tokenizer (prompt tok + per-beam detok) and the eos
    // id (beam retirement); a use_beam_search request then routes through
    // BeamSearchAsync (online.py) over the async engine. Without this, beam
    // requests reject with "requires an engine and a tokenizer".
    const std::optional<int32_t> beam_eos =
        tokenizer.EosId() >= 0
            ? std::optional<int32_t>(tokenizer.EosId())
            : std::nullopt;
    completion.set_beam_search_tokenizer(&tokenizer, beam_eos);
    chat.set_beam_search_tokenizer(&tokenizer, beam_eos);

    // ── #1985: the checkpoint's own sampling defaults. Upstream resolves these
    // once per server (ModelConfig.get_diff_sampling_param ->
    // OpenAIServing*.default_sampling_params) and every request that OMITS a
    // knob then takes them. THIS IS THE PRODUCTION CALL SITE: without these two
    // lines the whole feature is unreachable, and the reachability gate in
    // tests/vllm/entrypoints/openai/test_serving_generation_defaults.cpp goes
    // red when either is deleted. ─────────────────────────────────────────────
    const vllm::DefaultSamplingParams default_sampling_params =
        vllm::GetDiffSamplingParam(loaded->config(), args.generation_config);
    completion.set_default_sampling_params(default_sampling_params);
    chat.set_default_sampling_params(default_sampling_params);
    if (!default_sampling_params.empty()) {
      // Upstream logs the same thing at startup and names the source, so the
      // two engines' logs can be lined up: "Default vLLM sampling parameters
      // have been overridden by the model's generation_config.json".
      std::cerr << "server: default sampling params from "
                << (args.generation_config == "auto"
                        ? "the model's generation_config.json"
                        : args.generation_config)
                << ": " << default_sampling_params.ToString()
                << " (pass --generation-config vllm to use vLLM's neutral"
                   " defaults instead)\n";
    }

    // ── MM-SERVE-E2E: install the multimodal chat seam BY ARCHITECTURE.
    //
    // ENG-MM-INPUT-PIPELINE (#2475). This block used to decide whether the
    // server could serve images by asking whether
    // `<model_dir>/preprocessor_config.json` existed, and then construct
    // `Qwen3VLImageProcessor` + `MakeQwen3VLImageChatFn` with no branch on the
    // architecture anywhere in it. `preprocessor_config.json` is not a Qwen3-VL
    // marker, so a second multimodal checkpoint got Qwen3-VL's processor read
    // against its own `vision_config` shape, or threw into a swallowed `catch`
    // and left the server silently on the text-only path. Dispatch is now the
    // registry's, keyed on the architecture the model registry resolved
    // (mm_chat_registry.h; mirrors vllm/multimodal/registry.py:176-186, which
    // resolves `get_model_architecture(model_config)` and refuses BY NAME).
    //
    // The three outcomes are `MultiModalChatInstall`: a text-only architecture
    // installs nothing and the chat path stays byte-identical; a registered
    // architecture gets its seam; an architecture that declares multimodal
    // support and cannot build one gets a seam that REFUSES with HTTP 400,
    // never a silent text answer.
    //
    // An image_url request served through an installed seam renders the
    // placeholder marker → tokenizes to the single image_pad id → EXPANDS to N
    // image tokens + mm_features carried onto the engine request, and is
    // answered by `ModelRegistry::Forward`: the runner calls the registration's
    // `encode_mm` per item, `embed_mm` per step, and fills
    // `ModelForwardInput::mm` (runner.cpp `execute_mm_encoder`,
    // `ModelRegistry::EmbedMm`, `forward_input.mm = mm_buffers->mm`). What
    // ENG-MM-INPUT-PIPELINE P2 (#2379) left owed is named in
    // `.agents/specs/multimodal-track.md` `## Owed`: one request per step, the
    // host round-trip inside the merge, and image as the only modality that
    // reaches the runner.
    //
    // The seam OWNS its processor and its `BaseProcessingInfo` (the factory
    // captures them), so no locals live here for them and a future architecture
    // cannot get their lifetime wrong by forgetting to add one. `chat` is
    // declared AFTER `loaded`, and destruction runs in REVERSE of declaration
    // order, so the closure is destroyed BEFORE the `MultiModalConfig` its
    // `BaseProcessingInfo` holds by reference. Reordering those two
    // declarations reverses that and leaves the info holding a dangling
    // `const MultiModalConfig&`, so the order is load-bearing rather than
    // incidental.
    oai::MultiModalChatContext mm_ctx;
    mm_ctx.architecture = loaded->architecture();
    mm_ctx.model_dir = PathUtf8(dir);
    mm_ctx.config_path = config_path;
    mm_ctx.served_model_name = served_model_name;
    mm_ctx.tokenizer = &tokenizer;
    mm_ctx.prompt_fn = chat_prompt_fn;
    // The ONE image codec, shared with the `vllm_chat` install through
    // `include/vllm.h` so the two entry points of one library cannot accept
    // different containers. The container-format decode (PNG/JPEG -> RGB) is a
    // NAMED MM-SERVE residual and this codec refuses it by name; it belongs to
    // the LIBRARY and not to an architecture, which is why it is supplied here
    // once for every factory rather than grown per model.
    mm_ctx.codec = oai::DefaultImageCodec();
    // #607 L2 / #686: where --limit-mm-per-prompt / --language-model-only
    // landed. The seam folds them by min() against the architecture's own
    // ceiling, so a three-image request is answered with HTTP 400 "At most 1
    // image(s) may be provided in one prompt." instead of being served with its
    // first image.
    mm_ctx.mm_config = &loaded->mm_config();
    // The RESOLVED config, not `config.json`. A `.gguf` checkpoint has no such
    // file, and a factory whose processor is keyed on `vocab_size` -- as
    // DeepSeek-V4's is, because it spells every image position
    // `vocab_size + type` -- would otherwise have to guess it.
    mm_ctx.config = &loaded->config();
    // The `--mmproj` second file, so a factory can refuse a tower-free load at
    // INSTALL rather than inside the engine's busy loop.
    mm_ctx.mmproj_path = args.mmproj_path;
    mm_ctx.max_model_len = loaded->max_model_len();
    oai::InstallMultiModalChatSeam(chat, loaded->is_multimodal_model(), mm_ctx,
                                   std::cerr);

    // Diagnostic opt-out exists only for same-binary attribution. Production
    // defaults to the capacity-derived fixed pool.
    const char* fixed_pool_env = std::getenv("VLLM_CPP_HTTP_FIXED_POOL");
    const auto worker_pool_mode =
        fixed_pool_env != nullptr && std::string(fixed_pool_env) == "0"
            ? oai::ApiServer::HttpWorkerPoolMode::kLegacyDynamic
            : oai::ApiServer::HttpWorkerPoolMode::kCapacityFixed;
    oai::ApiServer server(completion, chat, models, vllm::Version(),
                          static_cast<size_t>(args.max_num_seqs),
                          worker_pool_mode);

    // ── C8 opt-in utility/admin endpoints (SERVE-UTILITY-ENDPOINTS /
    // SERVE-ADMIN). Wire the setters from the LIVE engine + tokenizer through the
    // single shared seam so the production server actually serves /tokenize,
    // /detokenize, /tokenizer_info (flag) and /abort_requests (dev-mode flag),
    // mirroring vLLM 0.26's per-endpoint default gating. /metrics and
    // /reset_prefix_cache stay unwired (no live backing on the AsyncLLM path) —
    // see ConfigureUtilityEndpoints + specs/{utility,admin}-endpoints.md. ────────
    // ── MiniMax-H3 video generation. OPT-IN: with no --video-dit the routes are
    // never registered and the server is byte-identical to before. The whole
    // pipeline lives in the LIBRARY seam (vllm::multimodal::MiniMaxH3VideoEngine,
    // ARCH-ONE-SURFACE ROW 2) — the SAME entry point the C ABI's vllm_video_*
    // and the minimax-h3-gen example drive, so HTTP and FFI cannot drift. This
    // file keeps exactly what an example may own: flag plumbing, the job
    // directory, and the ONE process spawn (ffmpeg, ratified 2026-08-03 — the
    // library builds the argv and spawns nothing). ────────────────────────────
    std::shared_ptr<vllm::multimodal::VideoEngine> video_engine;
    if (!args.video_dit.empty()) {
      std::cerr << "server: loading video checkpoints...\n";
      vllm::multimodal::VideoModelParams vmp;
      vmp.dit_path = args.video_dit;
      vmp.encoder_path = args.video_encoder;
      vmp.tokenizer_path = args.video_tokenizer;
      vmp.video_vae_path = args.video_vae;
      vmp.video_vae_config_path = args.video_vae_config;
      vmp.audio_vae_path = args.audio_vae;
      vmp.audio_vae_config_path = args.audio_vae_config;
      vmp.prompt_embeds_path = args.video_prompt_embeds;
      // The H3-specific partition rides in the generic extras (LTX-2.5 L1).
      if (!args.video_partition.empty()) vmp.extras["partition"] = args.video_partition;
      // ...and every other family-specific knob rides there too, from
      // --video-extra KEY=VALUE. Applied AFTER the partition alias so the two
      // spellings of one key cannot disagree silently: a --video-extra
      // partition=X that contradicts --video-partition Y is refused by name
      // rather than resolved by whichever assignment ran last.
      for (const auto& kv : args.video_extras) {
        const auto existing = vmp.extras.find(kv.first);
        if (existing != vmp.extras.end() && existing->second != kv.second) {
          throw std::runtime_error("server: --video-extra " + kv.first + "=" + kv.second +
                                   " contradicts the value already supplied for '" + kv.first +
                                   "' ('" + existing->second +
                                   "'). Refusing rather than preferring one.");
        }
        vmp.extras[kv.first] = kv.second;
      }
      vmp.device = args.video_device == "cuda" ? 1 : 0;
      vmp.dequant_bf16 = args.video_dequant_bf16 ? 1 : 0;
      vmp.encoder_max_layers = args.video_encoder_max_layers;
      // --video-family PINS the family; empty keeps detection, which is what
      // every invocation before this flag existed got and still gets. The flag
      // exists now because a SECOND family is registered (LTX-2.5), and the two
      // shipped LTX DiTs are separate files whose comparison is only a statement
      // about the files if the family is declared rather than inferred. It is
      // never a hint: an unregistered name was already refused at ParseArgs, and
      // a declared family that cannot load the checkpoint fails loudly instead
      // of falling back to detection.
      vmp.family = args.video_family;
      video_engine = vllm::multimodal::LoadVideoEngine(vmp);
      std::cerr << "server: video family "
                << (args.video_family.empty() ? "DETECTED" : "DECLARED (--video-family)")
                << "\n";
      std::cerr << "server: /v1/videos on (family=" << video_engine->family()
                << ", device=" << args.video_device
                << (args.video_dequant_bf16 ? ", dequant-bf16" : ", keep-quant") << ")\n";
      // HONEST LIMIT, stated at startup rather than buried: turning a PROMPT
      // into conditioning needs the H3-Encoder; without one every request is
      // conditioned on the SAME supplied embeddings.
      if (video_engine->has_encoder()) {
        std::cerr << "server: /v1/videos conditions on the request PROMPT\n";
      } else if (!video_engine->has_prompt_embeds()) {
        std::cerr << "server: WARNING /v1/videos has neither --video-encoder nor "
                     "--video-prompt-embeds; requests will be REJECTED\n";
      } else {
        std::cerr << "server: WARNING /v1/videos ignores the request PROMPT — pass "
                     "--video-encoder to condition on it\n";
      }

      auto counter = std::make_shared<std::atomic<int64_t>>(0);
      const std::string workdir = args.video_workdir;
      const std::string ffmpeg = args.video_ffmpeg;
      server.set_video_runner([video_engine, counter, workdir,
                               ffmpeg](const vllm::openai::VideoRequest& req) -> std::string {
        const int64_t id = counter->fetch_add(1);
        const std::string dir = workdir + "/job" + std::to_string(id);
        // The library-owned request mapping + generation: conditioning, task
        // resolution, the #77 partition guard, reference encoding, artifacts.
        const vllm::multimodal::VideoResult out =
            video_engine->Generate(vllm::multimodal::VideoGenParamsFromRequest(req, dir));
        // The ONE process spawn: exec the argv the library composed.
        std::vector<std::string> argv_mux = out.mux_argv;
        if (!argv_mux.empty()) argv_mux[0] = ffmpeg;
        const int status = RunFfmpegArgv(argv_mux);
        if (status != 0) {
          throw std::runtime_error("ffmpeg exited " + std::to_string(status));
        }
        return out.mux_output_path;
      });
    }

    // ── Speech + music generation (W6 of #672). OPT-IN: with no --speech-model
    // the route is never registered and the server is byte-identical to one
    // built before this existed. ─────────────────────────────────────────────
    std::shared_ptr<vllm::multimodal::SpeechEngine> speech_engine;
    if (!args.speech_model.empty()) {
      vllm::multimodal::SpeechRegistry& registry = vllm::multimodal::GlobalSpeechRegistry();
      vllm::models::music3::RegisterBuiltinSpeechFamilies(registry);
      vllm::multimodal::SpeechModelParams smp;
      smp.path = args.speech_model;
      smp.family = args.speech_family;  // empty => DETECT by inspecting the artifact
      smp.device = args.speech_device;   // 0 => CPU, the byte-identical default
      std::string why;
      // NOT `loaded`: that name is already taken by the TEXT engine in the
      // enclosing scope, and MSVC's C4456 is a warning-as-error there, so the
      // shadow red every pull request's `windows-msvc-*` pair with a failure
      // about a line the author had not touched (#965).
      std::unique_ptr<vllm::multimodal::SpeechEngine> loaded_speech_engine =
          registry.Load(smp, &why);
      if (loaded_speech_engine == nullptr) {
        // `why` names every family that was tried and the path, so a startup
        // failure is evidence rather than a verdict.
        throw std::runtime_error("server: --speech-model " + args.speech_model + ": " + why);
      }
      speech_engine = std::move(loaded_speech_engine);
      vllm::openai::SpeechCapabilities caps;
      caps.family = speech_engine->family();
      caps.sample_rate = speech_engine->sample_rate();
      caps.requires_reference_audio = speech_engine->requires_reference_audio();
      std::cerr << "server: /v1/audio/speech on (family=" << caps.family << ", "
                << caps.sample_rate << " Hz, "
                << (caps.requires_reference_audio ? "reference clip REQUIRED"
                                                  : "text-only synthesis")
                << ", speech family "
                << (args.speech_family.empty() ? "DETECTED" : "DECLARED (--speech-family)")
                << ", device " << vt::DeviceTypeName(speech_engine->device().type)
                << ")\n";
      // The request mapping is `vllm::openai::SynthesizeSpeechRequest` (library,
      // speech_api.h) rather than a lambda body here: a mapping that only a
      // running server can reach is a mapping no gate can call, and this route
      // is the one an end-to-end claim is made about.
      server.set_synthesizer(
          [speech_engine](const vllm::openai::SpeechRequest& req)
              -> vllm::openai::SpeechResponse {
            return vllm::openai::SynthesizeSpeechRequest(*speech_engine, req);
          },
          caps);
    }

    oai::UtilityEndpointOptions endpoint_opts;
    endpoint_opts.enable_tokenizer_info_endpoint =
        args.enable_tokenizer_info_endpoint;
    endpoint_opts.enable_server_dev_mode = args.enable_server_dev_mode;
    oai::ConfigureUtilityEndpoints(server, tokenizer, loaded->max_model_len(),
                                   engine, endpoint_opts);

    // ── Tev1 on /v1/systemone (MODEL-TEV1 Phase 6). A model dir whose
    // architectures name "Tev1Model" keeps every generation route and ALSO
    // answers /v1/systemone through the request-level seam: Tev1Decide, the
    // function vllm_decide calls. It scores through this same AsyncLLM, so no
    // lock is needed beside the chat traffic.
    if (loaded->architecture() == "Tev1Model") {
      const int64_t tev1_max_model_len = loaded->max_model_len();
      server.set_systemone_request(
          [&engine, &tokenizer, tev1_max_model_len](
              const nlohmann::ordered_json& body) -> nlohmann::ordered_json {
            vllm::Tev1Response r =
                vllm::Tev1Decide(engine, tokenizer, tev1_max_model_len, body);
            nlohmann::ordered_json out = nlohmann::ordered_json::object();
            out["answers"] = std::move(r.answers);
            out["usage"] = {{"input_tokens", r.input_tokens},
                            {"output_tokens", r.output_tokens}};
            return out;
          });
      std::cerr << "server: Tev1 decision model; /v1/systemone on\n";
    }
    std::cerr << "server: utility endpoints: /tokenize /detokenize on"
              << (args.enable_tokenizer_info_endpoint ? ", /tokenizer_info on"
                                                      : "")
              << (args.enable_server_dev_mode ? ", /abort_requests on (dev-mode)"
                                              : "")
              << "\n";

    // Prometheus /metrics (Python vLLM always-on family names).
    if (args.enable_metrics) {
      // #2770: the spec_decode metric families are config-gated exactly as
      // upstream gates them (loggers.py:477-481 hands the logger
      // `vllm_config.speculative_config`). This is the ONLY production
      // construction of a PrometheusStatLogger, so it is the call site that
      // decides whether a served speculative engine reports acceptance at all.
      // The RESOLVED config, not `args.speculative_config`: the loader
      // re-resolves k against the checkpoint, and the scheduler sizes its
      // per-position vectors from that same resolved value.
      const int num_speculative_tokens =
          loaded->speculative_config().has_value()
              ? loaded->speculative_config()->ResolvedNumSpeculativeTokens()
              : 0;
      prom_logger = std::make_unique<vllm::v1::metrics::PrometheusStatLogger>(
          served_model_name, loaded->max_model_len(), /*engine_index=*/0,
          num_speculative_tokens);
      // BOTH frontends record into the SAME logger, so a scrape reports this
      // process whichever one served the request. The async engine is the one
      // that matters here — every HTTP route is served from
      // loaded->async_engine() — and until #277 it recorded nothing, so
      // /metrics answered with a well-formed catalog that never moved.
      loaded->engine().set_stat_logger(prom_logger.get());
      loaded->async_engine().set_stat_logger(prom_logger.get());
      server.set_metrics_logger(prom_logger.get());
      std::cerr << "server: GET /metrics enabled (PrometheusStatLogger"
                << (num_speculative_tokens > 0
                        ? ", spec_decode families on)\n"
                        : ")\n");
    }

    std::cerr << "server: listening on http://" << args.host << ":" << args.port
              << " (model '" << served_model_name << "', HTTP worker pool ";
    if (server.http_worker_count() == 0) {
      std::cerr << "legacy-dynamic";
    } else {
      std::cerr << server.http_worker_count() << " fixed";
    }
    std::cerr << ")\n";

// DSR-ALLOW(ARCH-ONE-SURFACE): VT_BENCH_PROFILE_CONTROL is a build-option guard for the CUDA-graph-replay profiler, not a device fork; #189 moved it here verbatim from examples/server/main.cpp, which the DSR scanner never covered.
#if defined(VT_BENCH_PROFILE_CONTROL) && !defined(_WIN32)
    std::atomic<bool> benchmark_shutdown_waiter_ready{false};
    std::atomic<bool> benchmark_shutdown_received{false};
    std::atomic<bool> benchmark_shutdown_failed{false};
    std::atomic<bool> benchmark_shutdown_cancelled{false};
    std::thread benchmark_shutdown_thread;
    if (args.cuda_profile_graph_replays > 0) {
      benchmark_shutdown_thread = std::thread([&]() {
        const int shutdown_fd =
            open(args.benchmark_shutdown_fifo.c_str(),
                 O_RDONLY | O_NONBLOCK | O_CLOEXEC | O_NOFOLLOW);
        if (shutdown_fd < 0) {
          const int status = errno;
          std::cerr << "[VT_BENCH_SHUTDOWN] failed operation=open status="
                    << status << "\n";
          benchmark_shutdown_failed.store(true, std::memory_order_release);
          return;
        }
        struct stat shutdown_stat {};
        const int stat_status = fstat(shutdown_fd, &shutdown_stat);
        if (stat_status != 0 || !S_ISFIFO(shutdown_stat.st_mode)) {
          const int status = stat_status != 0 ? errno : EINVAL;
          std::cerr << "[VT_BENCH_SHUTDOWN] failed operation=fstat status="
                    << status << "\n";
          close(shutdown_fd);
          benchmark_shutdown_failed.store(true, std::memory_order_release);
          return;
        }
        benchmark_shutdown_waiter_ready.store(true, std::memory_order_release);
        std::cerr << "[VT_BENCH_SHUTDOWN] ready pid=" << CurrentProcessId()
                  << " control=fifo\n";
        while (!benchmark_shutdown_cancelled.load(std::memory_order_acquire)) {
          char command = '\0';
          const ssize_t bytes = read(shutdown_fd, &command, 1);
          if (bytes == 1) {
            if (command == 'Q') {
              benchmark_shutdown_received.store(true,
                                                 std::memory_order_release);
              std::cerr
                  << "[VT_BENCH_SHUTDOWN] requested control=fifo\n";
              close(shutdown_fd);
              server.stop();
              return;
            }
            std::cerr
                << "[VT_BENCH_SHUTDOWN] failed operation=command status="
                << static_cast<unsigned int>(
                       static_cast<unsigned char>(command))
                << "\n";
            close(shutdown_fd);
            benchmark_shutdown_failed.store(true, std::memory_order_release);
            server.stop();
            return;
          }
          if (bytes < 0 && errno != EAGAIN && errno != EWOULDBLOCK &&
              errno != EINTR) {
            const int status = errno;
            std::cerr << "[VT_BENCH_SHUTDOWN] failed operation=read status="
                      << status << "\n";
            close(shutdown_fd);
            benchmark_shutdown_failed.store(true, std::memory_order_release);
            server.stop();
            return;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        close(shutdown_fd);
      });
      while (!benchmark_shutdown_waiter_ready.load(std::memory_order_acquire) &&
             !benchmark_shutdown_failed.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      if (benchmark_shutdown_failed.load(std::memory_order_acquire)) {
        benchmark_shutdown_cancelled.store(true, std::memory_order_release);
        benchmark_shutdown_thread.join();
        return 1;
      }
    }
#endif

    vllm::platform::ConsoleShutdown shutdown_on_signal(
        [&]() { server.stop(); });
    const bool listen_ok = server.listen(args.host, args.port);

// DSR-ALLOW(ARCH-ONE-SURFACE): VT_BENCH_PROFILE_CONTROL is a build-option guard for the CUDA-graph-replay profiler, not a device fork; #189 moved it here verbatim from examples/server/main.cpp, which the DSR scanner never covered.
#if defined(VT_BENCH_PROFILE_CONTROL) && !defined(_WIN32)
    if (benchmark_shutdown_thread.joinable()) {
      benchmark_shutdown_cancelled.store(true, std::memory_order_release);
      benchmark_shutdown_thread.join();
      if (benchmark_shutdown_received.load(std::memory_order_acquire)) {
        std::cerr << "[VT_BENCH_SHUTDOWN] completed control=fifo\n";
      } else {
        if (!benchmark_shutdown_failed.load(std::memory_order_acquire)) {
          std::cerr
              << "[VT_BENCH_SHUTDOWN] failed operation=cancelled status=0\n";
        }
        if (listen_ok) {
          return 1;
        }
      }
    }
#endif

    if (!listen_ok) {
      std::cerr << "server: failed to bind " << args.host << ":" << args.port
                << "\n";
      return 1;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "server: fatal: " << e.what() << "\n";
    return 1;
  }
}

}  // namespace openai
}  // namespace entrypoints
}  // namespace vllm
