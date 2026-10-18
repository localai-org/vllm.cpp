// vllm-bench — the M2.1 throughput/latency benchmark harness for vllm.cpp.
//
// This is the tool that MEASURES gate #1 (throughput parity vs vLLM,
// .agents/gates.md): it drives the production V1 AsyncLLM frontend at a fixed
// concurrency and reports request/output/total token throughput plus
// TTFT/TPOT/ITL/E2EL, with a prefill-vs-decode split. The metrics + report
// format mirror `vllm bench serve` / `vllm bench throughput`
// (vllm/benchmarks/serve.py + throughput.py) so the numbers are directly
// comparable to a vLLM run on the same box, model and workload. See
// examples/bench/bench_core.h for the cited metric mapping.
//
//   vllm-bench [--model <dir|.gguf>] [--dataset-path <sharegpt.json>]
//              [--num-prompts N] [--input-len L]
//              [--output-len O] [--concurrency C] [--seed S]
//              [--temperature T] [--output-wait poll|blocking-c1]
//              [--output-token-ids <json>] [--output-json <json>]
//              [--kv-cache-dtype auto|bfloat16|fp8|fp8_e4m3]
//              [--ignore-eos | --no-ignore-eos]
//              [--skip-chat-template | --no-skip-chat-template]
//              [--chat-template <file or single-line template>]
//              [--enable-thinking | --no-enable-thinking]
//
// With NO --model it builds a SYNTHETIC tiny CPU engine so the harness runs on
// the dev box (toy weights => meaningless numbers; the point is the harness).
// On dgx.casa with a real --model it is the parity-measurement tool, e.g.:
//
//   ./vllm-bench --model <snapshot-dir-of-Qwen3.6-35B-A3B-NVFP4>
//                --num-prompts 200 --input-len 1024 --output-len 128
//                --concurrency 64
//
// Compare its throughput block against, on the SAME box + model files:
//   ~/venvs/vllm-oracle/bin/vllm bench throughput --model <same>
//                --input-len 1024 --output-len 128 --num-prompts 200
// Both runs go in .agents/parity-ledger.md (the lead records them).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

#include "bench_core.h"

namespace {

using vllm::bench::BenchConfig;

void Usage(const char* argv0, std::FILE* out) {
  std::fprintf(
      out,
      "usage: %s [--model <dir|.gguf>] [--num-prompts N] [--input-len L]\n"
      "          [--dataset-path <sharegpt.json>]\n"
      "          [--output-len O] [--concurrency C] [--seed S]\n"
      "          [--temperature T] [--max-num-batched-tokens B]\n"
      "          [--output-token-ids <json>]\n"
      "          [--output-json <json>]\n"
      "          [--output-wait poll|blocking-c1]\n"
      "          [--speculative-config <json>]\n"
      "          [--kv-cache-dtype auto|bfloat16|fp8|fp8_e4m3]\n"
      "          [--ignore-eos | --no-ignore-eos]\n"
      "          [--skip-chat-template | --no-skip-chat-template]\n"
      "          [--chat-template <file or single-line template>]\n"
      "          [--enable-thinking | --no-enable-thinking]\n"
      "\n"
      "Throughput/latency benchmark over the vllm.cpp V1 AsyncLLM, mirroring\n"
      "`vllm bench serve` metrics. With no --model, a synthetic CPU engine runs\n"
      "(numbers meaningless; smoke only). Defaults: --num-prompts 8 "
      "--input-len 16\n"
      "--output-len 16 --concurrency 4 --seed 0 --temperature 0 (greedy)\n"
      "--kv-cache-dtype auto. The KV dtype is printed in the report header, so a\n"
      "published number states the cache it was measured on.\n"
      "\n"
      "#2759: --ignore-eos (default) runs every request to exactly --output-len,\n"
      "which is what every landed figure was measured with; --no-ignore-eos lets\n"
      "a request stop where the model would have stopped, which is what a\n"
      "/v1/chat/completions comparator measures. --skip-chat-template (default)\n"
      "sends the raw prompt; --no-skip-chat-template renders one user message\n"
      "through the model's own template, or through --chat-template. Both\n"
      "settings are printed in the report header, so a published number states\n"
      "its stopping rule and its prompt shape.\n",
      argv0);
}

const char* NextArg(int argc, char** argv, int& i) {
  if (i + 1 >= argc) {
    std::fprintf(stderr, "vllm-bench: missing value for '%s'\n", argv[i]);
    Usage(argv[0], stderr);
    std::exit(2);
  }
  return argv[++i];
}

bool ParseArgs(int argc, char** argv, BenchConfig& cfg, int& exit_code) {
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--model") {
      cfg.model_path = NextArg(argc, argv, i);
    } else if (flag == "--dataset-path") {
      cfg.dataset_path = NextArg(argc, argv, i);
    } else if (flag == "--num-prompts") {
      cfg.num_prompts = std::atoi(NextArg(argc, argv, i));
    } else if (flag == "--input-len") {
      cfg.input_len = std::atoi(NextArg(argc, argv, i));
    } else if (flag == "--output-len") {
      cfg.output_len = std::atoi(NextArg(argc, argv, i));
    } else if (flag == "--concurrency") {
      cfg.concurrency = std::atoi(NextArg(argc, argv, i));
    } else if (flag == "--seed") {
      cfg.seed = std::strtoull(NextArg(argc, argv, i), nullptr, 10);
    } else if (flag == "--temperature") {
      cfg.temperature = std::atof(NextArg(argc, argv, i));
    } else if (flag == "--output-token-ids") {
      cfg.output_token_ids_path = NextArg(argc, argv, i);
    } else if (flag == "--output-json") {
      cfg.output_json_path = NextArg(argc, argv, i);
    } else if (flag == "--output-wait") {
      try {
        cfg.output_wait =
            vllm::bench::ParseOutputWaitMode(NextArg(argc, argv, i));
      } catch (const std::invalid_argument& e) {
        std::fprintf(stderr, "vllm-bench: %s\n", e.what());
        exit_code = 2;
        return false;
      }
    } else if (flag == "--max-num-batched-tokens") {
      cfg.max_num_batched_tokens = std::atoi(NextArg(argc, argv, i));
    } else if (flag == "--num-blocks") {
      cfg.num_blocks = std::atoi(NextArg(argc, argv, i));
    } else if (flag == "--speculative-config") {
      cfg.speculative_config = NextArg(argc, argv, i);
    } else if (flag == "--kv-cache-dtype") {
      // KV-FP8 W7 (#2619). Mirrors server_main.cpp and examples/cli/main.cpp:
      // take the raw string, validate NOTHING here. `vllm::v1::ParseCacheDType`
      // owns the accepted set and refuses an unknown name BY NAME; a second
      // copy of that list in a benchmark client could only ever drift from it.
      cfg.kv_cache_dtype = NextArg(argc, argv, i);
    } else if (flag == "--ignore-eos") {
      // #2759. The NAME is vLLM's (`vllm/benchmarks/serve.py:1757-1762`, where
      // it is `store_true`); the `--no-` negation is this tree's
      // (`server_main.cpp` `--no-enable-thinking`). Both arms exist even though
      // one restates the default, because a recipe that states its stopping
      // rule explicitly is the point of the flag.
      cfg.ignore_eos = true;
    } else if (flag == "--no-ignore-eos") {
      cfg.ignore_eos = false;
    } else if (flag == "--skip-chat-template") {
      // #2759. Name from `vllm/benchmarks/datasets/datasets.py:1654-1658`.
      cfg.skip_chat_template = true;
    } else if (flag == "--no-skip-chat-template") {
      cfg.skip_chat_template = false;
    } else if (flag == "--chat-template") {
      // #2759. "The file path to the chat template, or the template in
      // single-line form" -- `vllm/entrypoints/launchers/cli_args.py:80-82`.
      // Taken VERBATIM: `ResolveBenchChatTemplate` owns the file-or-literal
      // decision, and a second copy of it here could only ever drift.
      cfg.chat_template = NextArg(argc, argv, i);
    } else if (flag == "--enable-thinking") {
      // #2759. `server_main.cpp`'s spelling and `DefaultChatTemplateKwargs`'s
      // rule: UNSET is not false, and only unset leaves `enable_thinking`
      // Jinja-undefined (#1681).
      cfg.enable_thinking = true;
    } else if (flag == "--no-enable-thinking") {
      cfg.enable_thinking = false;
    } else if (flag == "-h" || flag == "--help") {
      Usage(argv[0], stdout);
      exit_code = 0;
      return false;
    } else {
      std::fprintf(stderr, "vllm-bench: unknown argument '%s'\n", flag.c_str());
      Usage(argv[0], stderr);
      exit_code = 2;
      return false;
    }
  }
  if (cfg.num_prompts <= 0 || cfg.input_len <= 0 || cfg.output_len <= 0 ||
      cfg.concurrency <= 0) {
    std::fprintf(stderr,
                 "vllm-bench: --num-prompts/--input-len/--output-len/"
                 "--concurrency must be positive\n");
    exit_code = 2;
    return false;
  }
  if (cfg.output_wait == vllm::bench::OutputWaitMode::kBlockingC1 &&
      cfg.concurrency != 1) {
    std::fprintf(stderr,
                 "vllm-bench: benchmark output wait 'blocking-c1' requires "
                 "--concurrency 1\n");
    exit_code = 2;
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  BenchConfig cfg;
  int exit_code = 0;
  if (!ParseArgs(argc, argv, cfg, exit_code)) {
    return exit_code;
  }

  std::fprintf(stderr,
               "vllm-bench: %s engine | num_prompts=%d input_len=%d "
               "output_len=%d concurrency=%d seed=%llu temp=%.2f dataset=%s "
               "output_wait=%s kv_cache_dtype=%s ignore_eos=%d chat_template=%s\n",
               cfg.model_path.empty() ? "SYNTHETIC (no --model)"
                                       : cfg.model_path.c_str(),
               cfg.num_prompts, cfg.input_len, cfg.output_len, cfg.concurrency,
               static_cast<unsigned long long>(cfg.seed), cfg.temperature,
               cfg.dataset_path.empty() ? "generated" : cfg.dataset_path.c_str(),
               vllm::bench::OutputWaitModeName(cfg.output_wait),
               cfg.kv_cache_dtype.c_str(), cfg.ignore_eos ? 1 : 0,
               cfg.skip_chat_template
                   ? "skipped"
                   : (cfg.chat_template.empty() ? "the model's own"
                                                : cfg.chat_template.c_str()));

  try {
    const vllm::bench::BenchResult res = vllm::bench::RunBench(cfg);
    vllm::bench::PrintReport(cfg, res, stdout);
    if (!cfg.output_token_ids_path.empty()) {
      vllm::bench::WriteOutputTokenIds(cfg.output_token_ids_path, res);
    }
    if (!cfg.output_json_path.empty()) {
      vllm::bench::WriteResultJson(cfg.output_json_path, cfg, res);
    }
    if (res.completed != cfg.num_prompts) {
      std::fprintf(stderr,
                   "vllm-bench: only %d/%d requests completed\n", res.completed,
                   cfg.num_prompts);
      return 1;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "vllm-bench: failed: %s\n", e.what());
    return 1;
  }
  return 0;
}
