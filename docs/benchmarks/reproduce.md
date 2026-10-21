# Reproduce

| Benchmark | Entry point |
|---|---|
| vLLM online grid | `.agents/specs/competitive-benchmarks.md`, evidence under `dgx:~/work/vllm.cpp-online-gate/evidence/` |
| Clock-controlled pin series | `$HOME/gpu.lock` FIRST, then `sudo -n nvidia-smi -lgc 2190` under an always-fires `-rgc` trap; oracle by EXPLICIT PATH, identity asserted per leg; a `gpu_clock_state.py` window per leg |
| CPU vs llama.cpp | Same GGUF both arms, 3 reps under one `flock $HOME/gpu.lock`; `VT_GGUF_KEEP_F16=0` reproduces the pre-L7 baseline |
| Kolibri-1 CPU, dequant-cache default | `numactl --cpunodebind=3 --interleave=all` with `VLLM_CPP_CPU_THREADS=32`, the whole series under one `flock -x $HOME/gpu.lock`; the **default** leg leaves `VT_KOLIBRI1_DEQUANT_CACHE_MB` UNSET (the derived budget), the **cache-off** leg sets it to `0`, the **explicit** leg sets `16384`. The serving number is the in-repo probe `tests/test_kolibri1_spec_probe`; the decode and prefill legs are the session bench driver (`VT_KB_PROMPT`/`VT_KB_GEN`). Commands and limits: [`kolibri1-cpu-dequant-cache-default.md`](kolibri1-cpu-dequant-cache-default.md) |
| Laguna NVFP4 decode | `flock $HOME/gpu.lock ./build-cuda/examples/laguna-gen --model ~/laguna-xs-nvfp4 --gpu` (that directory holds the S-2.1 checkpoint); `drop_caches` first, create the CUDA context before loading weights |
| DeepSeek-V4-Flash decode | `deepseek-v4-gen --gpu --kv-cache` on `ds4flash.gguf`, captured under tmux |
| Metal vs MLX-LM | Paired A/B harness, interleaved runs, cold legs discarded |
| Vulkan vs llama.cpp Vulkan | Same GGUF both arms: ours `-DVLLM_CPP_VULKAN=ON`, llama.cpp `-DGGML_VULKAN=ON` at `237ad9b96`, SUPERSEDED, via `llama-bench`; clean legs only, one `flock $HOME/gpu.lock`. GEMV sweep: `benchmarks/vulkan_gemv_ab.cpp` |
| Qwen3.8-27B EXL3 on GB10 | Step-by-step in [`qwen38-27b-exl3-gb10.md`](qwen38-27b-exl3-gb10.md#reproduce-this-run): both checkpoints pinned by revision and sha256, the 164-problem HumanEval set, and both arms on one binary. `VT_DFLASH_PAGED=0` is required ([#2274](https://github.com/mudler/vllm.cpp/issues/2274)) |
| Variadic serving load, either engine | `benchmarks/variadic/`, method in [`variadic-load-methodology.md`](variadic-load-methodology.md#running-it): `build_corpus.py` from three sha256-pinned corpora, `client.py` closed-loop at a chosen concurrency, `report.py` for the tables, `job.sh` for the whole sweep inside an `rc` lease. The report recomputes every number from the per-request records, so a corrected statistic needs no GPU |
| Which llama.cpp a figure ran | Three revisions on this page, all SUPERSEDED (#1003): fork `237ad9b96` (GB10 CPU, Vulkan, x86, kernel matrix), stock `b9892` (Pi 5), stock `7044859` (Muse Glimmer, #391). Pin is stock `b10451`, unbuilt (#857) |
| Revisions repo-wide | **Five**, not three, enumerated in the [spec](../../.agents/specs/oracle-llamacpp-repin-stock.md). Absent here: stock `030ebb5` (NON-BINDING) and a Poolside fork BRANCH with no commit recorded, behind Laguna's `27.8 tok/s` |

Build flags, environment variables, and the full gate list are in
[BUILD.md](../BUILD.md) and [ENVIRONMENT.md](../ENVIRONMENT.md).
