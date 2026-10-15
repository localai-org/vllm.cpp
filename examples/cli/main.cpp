// vllm-cli — a llama.cpp-style command-line client for vllm.cpp, driven ENTIRELY
// through the stable C ABI (include/vllm.h). It links libvllm and uses only the
// public `vllm_*` surface (no C++ engine headers), so it doubles as a worked
// example of the library-first packaging: load a model dir, run one completion
// (blocking or streaming), print it, free everything, report errors via
// vllm_last_error().
//
//   vllm-cli --model <dir> --prompt "<text>"
//            [--tokenizer-config <path>] [--device auto|cpu|cuda]
//            [--max-tokens N] [--temperature T] [--top-p P] [--top-k K]
//            [--seed S] [--stream] [--repeat N]
//            [--gpu-memory-utilization F] [--kv-cache-memory BYTES]
//            [--max-num-seqs N] [--kv-cache-dtype auto|bfloat16|fp8|fp8_e4m3]
//
// <dir> holds config.json, tokenizer.json and the *.safetensors shards (T0:
// safetensors only). Loading a real checkpoint is a GPU/dgx concern; on a CPU
// box `--help` / bad-args still work without a model (smoke-tested in CI).
// --repeat N runs N completions after one load (warm bench / decode tok-s).
#include "vllm.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string model_dir;
  std::string tokenizer_config;  // optional override
  std::string prompt;
  bool have_prompt = false;
  std::string prompt_file;
  int max_model_len = 0;
  int max_num_batched_tokens = 0;
  int num_blocks = 0;
  int max_tokens = 16;
  float temperature = 0.0f;  // default: greedy (deterministic).
  float top_p = 1.0f;
  int top_k = 0;
  unsigned long long seed = 0;
  bool have_seed = false;
  bool stream = false;
  int repeat = 1;  // load once, complete N times (warm tok/s)
  std::string speculative_config;  // vLLM --speculative-config JSON; "" => off.
  // --offload-config (#1135): the same JSON document `vllm-server` takes, and
  // the same ABI field. It carries TWO halves — vLLM's mirrored `uva`/`prefetch`
  // weight offload, and vllm.cpp's `vllm_cpp` weight-residency tier, which is
  // what makes a checkpoint larger than host RAM loadable. `vllm-cli` had no
  // such flag, so the only way to reach the residency tier from it was the
  // `VT_*` environment variables. "" => NULL => the byte-identical default.
  std::string offload_config;
  // --device (ABI v14): "auto" (default probe), "cpu", or "cuda" — the names
  // of vLLM's DeviceConfig.device this build serves. Mapped to the int the ABI
  // takes (0/1/2) in ParseArgs; an unknown name is rejected there.
  int32_t device = 0;
  // --gpu-memory-utilization / --kv-cache-memory (ABI v16): KV-pool sizing.
  // gpu_memory_utilization is inert until the M3 profile run lands;
  // kv_cache_memory_bytes (> 0) sizes the block count directly.
  //
  // 0.0, NOT 0.92 (FIX-GPU-MEM-UTIL-INERT, #1165). 0.0 is the ABI's documented
  // "unset" spelling (vllm.h: 0.0 => 0.92), so this stays 0.0 until the user
  // types --gpu-memory-utilization. The engine now warns that a CHOSEN fraction
  // sized nothing, and pre-filling 0.92 here would make every plain `vllm-cli`
  // run look like an explicit ask and print that warning.
  double gpu_memory_utilization = 0.0;
  long long kv_cache_memory_bytes = 0;
  // --max-num-seqs: max concurrent sequences. Exposed because it is the knob the
  // recurrent-state budget check names (issue #371): under speculative decoding
  // a GDN model's state is max_num_seqs * (k+1) * per-slot, so this is what a
  // user must lower to make a speculative run fit. 0 = leave the engine default.
  int max_num_seqs = 0;
  // --kv-cache-dtype: vLLM CacheConfig.cache_dtype. "" => auto (the default).
  std::string kv_cache_dtype;
};

void Usage(const char* argv0, std::FILE* out) {
  std::fprintf(
      out,
      "usage: %s --model <dir> --prompt \"<text>\"\n"
      "          [--prompt-file PATH (instead of --prompt)]\n"
      "          [--tokenizer-config <path>] [--device auto|cpu|cuda]\n"
      "          [--max-tokens N] [--temperature T] [--top-p P] [--top-k K]\n"
      "          [--seed S] [--stream] [--repeat N]\n"
      "          [--gpu-memory-utilization F] [--kv-cache-memory BYTES]\n"
      "          [--max-num-seqs N]\n"
      "          [--max-model-len N] [--max-num-batched-tokens N] [--num-blocks N]\n"
      "          [--kv-cache-dtype auto|bfloat16|fp8|fp8_e4m3]\n"
      "          [--speculative-config '<json>'] [--offload-config '<json>']\n"
      "\n"
      "Runs completion(s) over the vllm.cpp C ABI (libvllm). <dir> holds\n"
      "config.json, tokenizer.json and the *.safetensors shards.\n"
      "--repeat N: load once, run N blocking completions (for warm tok/s).\n",
      argv0);
}

// Returns the value following `argv[i]`, advancing i; prints usage + exits on a
// missing operand.
const char* NextArg(int argc, char** argv, int& i) {
  if (i + 1 >= argc) {
    std::fprintf(stderr, "vllm-cli: missing value for '%s'\n", argv[i]);
    Usage(argv[0], stderr);
    std::exit(2);
  }
  return argv[++i];
}

// Parse argv. Returns false (after printing usage) if the args are invalid or
// --help was given; *exit_code carries the process exit status in that case.
bool ParseArgs(int argc, char** argv, Args& a, int& exit_code) {
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--model") {
      a.model_dir = NextArg(argc, argv, i);
    } else if (flag == "--tokenizer-config") {
      a.tokenizer_config = NextArg(argc, argv, i);
    } else if (flag == "--prompt") {
      a.prompt = NextArg(argc, argv, i);
      a.have_prompt = true;
    } else if (flag == "--prompt-file") {
      a.prompt_file = NextArg(argc, argv, i);
    } else if (flag == "--max-model-len" || flag == "--max-num-batched-tokens" ||
               flag == "--num-blocks") {
      const std::string value = NextArg(argc, argv, i);
      size_t used = 0;
      long long n = 0;
      try { n = std::stoll(value, &used); } catch (...) { used = 0; }
      if (used != value.size() || used == 0 || n <= 0 ||
          n > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "vllm-cli: %s requires a positive int\n", flag.c_str());
        exit_code = 2;
        return false;
      }
      if (flag == "--max-model-len") a.max_model_len = static_cast<int>(n);
      else if (flag == "--max-num-batched-tokens") a.max_num_batched_tokens = static_cast<int>(n);
      else a.num_blocks = static_cast<int>(n);
    } else if (flag == "--max-tokens") {
      a.max_tokens = std::atoi(NextArg(argc, argv, i));
    } else if (flag == "--temperature") {
      a.temperature = static_cast<float>(std::atof(NextArg(argc, argv, i)));
    } else if (flag == "--top-p") {
      a.top_p = static_cast<float>(std::atof(NextArg(argc, argv, i)));
    } else if (flag == "--top-k") {
      a.top_k = std::atoi(NextArg(argc, argv, i));
    } else if (flag == "--seed") {
      a.seed = std::strtoull(NextArg(argc, argv, i), nullptr, 10);
      a.have_seed = true;
    } else if (flag == "--stream") {
      a.stream = true;
    } else if (flag == "--repeat") {
      a.repeat = std::atoi(NextArg(argc, argv, i));
      if (a.repeat < 1) a.repeat = 1;
    } else if (flag == "--speculative-config") {
      a.speculative_config = NextArg(argc, argv, i);
    } else if (flag == "--offload-config") {
      a.offload_config = NextArg(argc, argv, i);
    } else if (flag == "--gpu-memory-utilization") {
      a.gpu_memory_utilization = std::atof(NextArg(argc, argv, i));
    } else if (flag == "--kv-cache-memory") {
      a.kv_cache_memory_bytes = std::strtoll(NextArg(argc, argv, i), nullptr, 10);
    } else if (std::strcmp(argv[i], "--max-num-seqs") == 0) {
      a.max_num_seqs = std::atoi(NextArg(argc, argv, i));
    } else if (flag == "--kv-cache-dtype") {
      a.kv_cache_dtype = NextArg(argc, argv, i);
    } else if (flag == "--device") {
      // The vLLM DeviceConfig.device names (auto/cpu/cuda) -> the ABI int
      // (vllm_model_params.device: 0=auto, 1=cpu, 2=cuda). An unknown name is
      // a usage error, mirroring vLLM rejecting a non-Literal device value.
      const std::string device = NextArg(argc, argv, i);
      if (device == "auto") {
        a.device = 0;
      } else if (device == "cpu") {
        a.device = 1;
      } else if (device == "cuda") {
        a.device = 2;
      } else {
        std::fprintf(stderr,
                     "vllm-cli: unknown --device '%s' (expected auto, cpu, or "
                     "cuda)\n",
                     device.c_str());
        Usage(argv[0], stderr);
        exit_code = 2;
        return false;
      }
    } else if (flag == "-h" || flag == "--help") {
      Usage(argv[0], stdout);
      exit_code = 0;
      return false;
    } else {
      std::fprintf(stderr, "vllm-cli: unknown argument '%s'\n", flag.c_str());
      Usage(argv[0], stderr);
      exit_code = 2;
      return false;
    }
  }
  if (a.model_dir.empty()) {
    std::fprintf(stderr, "vllm-cli: --model <dir> is required\n");
    Usage(argv[0], stderr);
    exit_code = 2;
    return false;
  }
  if (!a.prompt_file.empty()) {
    if (a.have_prompt) {
      std::fprintf(stderr, "vllm-cli: choose --prompt or --prompt-file\n");
      exit_code = 2;
      return false;
    }
    std::ifstream input(a.prompt_file, std::ios::binary);
    if (!input) {
      std::fprintf(stderr, "vllm-cli: cannot read prompt file %s\n", a.prompt_file.c_str());
      exit_code = 2;
      return false;
    }
    a.prompt.assign(std::istreambuf_iterator<char>(input), {});
    if (input.bad() || a.prompt.find('\0') != std::string::npos) {
      std::fprintf(stderr, "vllm-cli: prompt file must be readable text without NUL\n");
      exit_code = 2;
      return false;
    }
    a.have_prompt = true;
  }
  if (!a.have_prompt) {
    std::fprintf(stderr, "vllm-cli: --prompt \"<text>\" is required\n");
    Usage(argv[0], stderr);
    exit_code = 2;
    return false;
  }
  return true;
}

// Streaming callback: print each delta as it arrives (flushing so it streams to
// the terminal), keep going until the engine finishes. Matches
// vllm_token_callback exactly.
bool StreamPrintCb(const char* delta_text, bool finished, void* user_data) {
  (void)finished;
  (void)user_data;
  if (delta_text != nullptr) {
    std::fputs(delta_text, stdout);
    std::fflush(stdout);
  }
  return true;  // never stop early from the CLI.
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  int exit_code = 0;
  if (!ParseArgs(argc, argv, args, exit_code)) {
    return exit_code;
  }

  // ── Load the model + build the engine stack via the C ABI. ─────────────────
  vllm_model_params mp = vllm_model_params_default();
  mp.model_path = args.model_dir.c_str();
  if (!args.tokenizer_config.empty()) {
    mp.tokenizer_config_path = args.tokenizer_config.c_str();
  }
  // --speculative-config: enable speculative decoding (SPEC-MTP). Absent =>
  // NULL => the byte-identical no-speculation engine. A malformed/unsupported
  // document fails the load below with the parse error.
  if (!args.speculative_config.empty()) {
    mp.speculative_config = args.speculative_config.c_str();
  }
  // --offload-config (#1135): one string, both halves, parsed by the library at
  // `vllm_engine_load` (src/capi/vllm_c.cpp) exactly as `vllm-server` parses it.
  // This example stays a thin ABI client and adds no parsing of its own, so a
  // malformed document, an unknown key at any level of it, or a residency
  // document that a decision has already fixed all fail the load below with the
  // library's own message.
  if (!args.offload_config.empty()) {
    mp.offload_config = args.offload_config.c_str();
  }
  // --device: explicit device selection (ABI v14). 0 (the default) keeps the
  // accelerator-first probe; an explicitly named absent device fails the load
  // below with the library's message (never a silent fallback).
  mp.device = args.device;
  // KV-pool sizing knobs (ABI v16). Defaults leave the historical 256-block
  // behaviour; --kv-cache-memory sizes the pool from an absolute byte budget.
  // The assignment OVERWRITES vllm_model_params_default()'s pre-filled 0.92
  // with 0.0 unless --gpu-memory-utilization was typed, which is what keeps a
  // plain run out of the #1165 warning. Do not guard it with `> 0.0`: that
  // would leave the 0.92 in place and warn on every start.
  mp.gpu_memory_utilization = args.gpu_memory_utilization;
  mp.kv_cache_memory_bytes = args.kv_cache_memory_bytes;
  if (args.max_num_seqs > 0) mp.max_num_seqs = args.max_num_seqs;
  if (args.max_model_len > 0) mp.max_model_len = args.max_model_len;
  if (args.max_num_batched_tokens > 0) mp.max_num_batched_tokens = args.max_num_batched_tokens;
  if (args.num_blocks > 0) mp.num_blocks = args.num_blocks;
  if (!args.kv_cache_dtype.empty()) mp.kv_cache_dtype = args.kv_cache_dtype.c_str();

  vllm_engine* engine = nullptr;
  std::fprintf(stderr, "vllm-cli: loading model from %s\n",
               args.model_dir.c_str());
  vllm_status st = vllm_engine_load(&mp, &engine);
  if (st != VLLM_OK) {
    std::fprintf(stderr, "vllm-cli: model load failed (status %d): %s\n",
                 static_cast<int>(st), vllm_last_error());
    return 1;
  }

  // ── Sampling params from the CLI flags. ────────────────────────────────────
  vllm_sampling_params sp = vllm_sampling_params_default();
  sp.temperature = args.temperature;  // <= 0 => greedy.
  sp.top_p = args.top_p;
  sp.top_k = args.top_k;
  sp.max_tokens = args.max_tokens;
  if (args.have_seed) {
    sp.has_seed = 1;
    sp.seed = static_cast<uint64_t>(args.seed);
  }

  int rc = 0;
  if (args.stream) {
    // ── Streaming: print deltas as they arrive. ──────────────────────────────
    if (args.repeat != 1) {
      std::fprintf(stderr, "vllm-cli: --repeat with --stream not supported; using 1\n");
    }
    st = vllm_complete_stream(engine, args.prompt.c_str(), &sp, &StreamPrintCb,
                              nullptr);
    std::fputc('\n', stdout);
    if (st != VLLM_OK) {
      std::fprintf(stderr, "vllm-cli: streaming completion failed (status %d): %s\n",
                   static_cast<int>(st), vllm_last_error());
      rc = 1;
    }
  } else {
    // ── Blocking: load once, optionally repeat for warm tok/s. ───────────────
    for (int r = 0; r < args.repeat; ++r) {
      vllm_completion out{};
      // TWO CLOCKS, DELIBERATELY, because they answer different questions.
      // steady_clock measures the DURATION and cannot jump; system_clock names
      // the INSTANT, which is the only thing an out-of-process observer can
      // line its own samples up against. Reporting a duration from the wall
      // clock, or an instant from the monotonic one, would each be wrong.
      // THE SPECULATIVE COUNTERS EITHER SIDE OF THE LEG (#2832). They run for
      // the life of the handle and --repeat keeps ONE handle for every leg, so
      // the raw value is a running total and only the difference belongs to
      // this leg. `spec_ok` is both reads succeeding: a partial pair would make
      // the difference meaningless, and a benchmark reading a meaningless
      // number is worse than one reading none.
      //
      // OUTSIDE BOTH CLOCKS, deliberately. The first read is taken before `w0`
      // and the second after `w1`, so neither the duration nor the leg span
      // this instrument annotates contains the instrument.
      vllm_spec_acceptance spec_before{};
      vllm_spec_acceptance spec_after{};
      bool spec_ok =
          vllm_engine_spec_acceptance(engine, &spec_before) == VLLM_OK;
      const auto w0 = std::chrono::system_clock::now();
      const auto t0 = std::chrono::steady_clock::now();
      st = vllm_complete(engine, args.prompt.c_str(), &sp, &out);
      const auto t1 = std::chrono::steady_clock::now();
      const auto w1 = std::chrono::system_clock::now();
      spec_ok = spec_ok &&
                vllm_engine_spec_acceptance(engine, &spec_after) == VLLM_OK;
      const double secs =
          std::chrono::duration<double>(t1 - t0).count();
      if (st != VLLM_OK) {
        std::fprintf(stderr, "vllm-cli: completion failed (status %d): %s\n",
                     static_cast<int>(st), vllm_last_error());
        rc = 1;
        break;
      }
      if (r == 0 || args.repeat == 1) {
        std::fputs(out.text != nullptr ? out.text : "", stdout);
        std::fputc('\n', stdout);
      }
      const int ct = out.completion_tokens;
      const double tps = (secs > 0.0 && ct > 0) ? (static_cast<double>(ct) / secs) : 0.0;
      std::fprintf(stderr,
                   "vllm-cli: run=%d/%d finish_reason=%s prompt_tokens=%d "
                   "completion_tokens=%d secs=%.3f tok_s=%.3f\n",
                   r + 1, args.repeat,
                   out.finish_reason != nullptr ? out.finish_reason : "(none)",
                   out.prompt_tokens, ct, secs, tps);
      // THE LEG'S BOUNDARIES, as Unix epoch seconds (#1671). A benchmark
      // sampling the GPU from outside this process sees the PROCESS, and one
      // `vllm-cli` run is a 52 GiB checkpoint read followed by a few seconds of
      // generation: the DFlash2 speed gate's window came out 18.37% busy over
      // 3222 samples and its clock rule refused it, correctly, because the
      // retained window did not describe the measured work. Only this process
      // knows when the generation was, so only this process can say. Epoch
      // seconds rather than an ISO stamp because there is no time zone to get
      // wrong, and printed on their own line so the timing line above keeps the
      // format every existing reader already parses.
      std::fprintf(stderr,
                   "vllm-cli: run=%d/%d generate_start_unix=%.6f "
                   "generate_end_unix=%.6f\n",
                   r + 1, args.repeat,
                   std::chrono::duration<double>(w0.time_since_epoch()).count(),
                   std::chrono::duration<double>(w1.time_since_epoch()).count());
      // THE LEG'S SPECULATIVE ACCEPTANCE (#2832). A third line, for the reason
      // the second one is a line of its own: the timing line is parsed by
      // evidence and readers that predate this marker, so its bytes do not
      // move.
      //
      // UNITS, because the two reference engines disagree on one of them:
      // spec_drafts_accepted EXCLUDES the bonus token a verify step always
      // emits, exactly as vLLM's `spec_decode_num_accepted_tokens` does, so
      // accepted/proposed is the acceptance RATE and
      // 1 + accepted/drafted_request_steps is vLLM's `mean_acceptance_length`,
      // which DOES include that token. It is NOT SGLang's `accept_length`,
      // whose numerator also carries the prefill token; see `vllm.h`.
      //
      // AND THE THIRD COUNT IS PER (REQUEST, STEP), NOT PER FORWARD PASS, which
      // is why it is not called a verify-step count: at concurrency 8 one verify
      // forward over 8 drafted requests adds 8. Both oracles count it the same
      // way, so the comparison is exact; a reader who takes it for a forward
      // count is wrong by the batch size.
      //
      // NOT PRINTED AT ALL when either read failed. A silent line is a refusal
      // in the harness that reads it; a line of zeroes would be a measurement
      // of an engine that accepted nothing.
      if (spec_ok) {
        std::fprintf(stderr,
                     "vllm-cli: run=%d/%d spec_drafts_proposed=%lld "
                     "spec_drafts_accepted=%lld "
                     "spec_drafted_request_steps=%lld\n",
                     r + 1, args.repeat,
                     static_cast<long long>(spec_after.drafts_proposed -
                                            spec_before.drafts_proposed),
                     static_cast<long long>(spec_after.drafts_accepted -
                                            spec_before.drafts_accepted),
                     static_cast<long long>(spec_after.drafted_request_steps -
                                            spec_before.drafted_request_steps));
      }
      std::fflush(stderr);
      vllm_completion_free(&out);
    }
  }

  vllm_engine_free(engine);
  return rc;
}
