# TinyLlama same-binary Pascal attention A/B

Binary: `/tmp/pascal-attention/actual-frozen`.
SHA256: `656228b1744c8ab20ad041546e71dc9bd2498cd92f35f2e9cbca5dedb92968fc`. Attention source head: `22d433baf`.

Started 2026-10-29T13:59:14.022778+00:00; finished 2026-10-29T14:01:19.811606+00:00.
The approved local nonfleet P620 was held through `/home/user/gpu.lock` for the entire campaign. Before: Quadro P620, GPU-fc83f069-1409-46ec-84be-618f60c8d567, 580.178.04, P0, 1265 MHz, 2504 MHz, 38, [N/A], 7 MiB, 0 %. No compute processes were listed. After: Quadro P620, GPU-fc83f069-1409-46ec-84be-618f60c8d567, 580.178.04, P0, 1354 MHz, 2504 MHz, 64, [N/A], 7 MiB, 87 %. No compute processes remained. No inherited VT_ATTN/VT_FA2 environment variables were present.

Only `VT_ATTN_DECODE_OPT` changes: ON=`1`, rollback OFF=`0`. Pair order alternates ON/OFF and OFF/ON across the four axes. Each process loads the same checkpoint and executes one cold completion plus three warm completions. All completion calls include prefill and greedy generation. Context256, blocksize16, pool32, batch/concurrency1, temperature0, seed0, ignoreEOS. No recompile or changed model artifact occurs between arms.

| Prompt / generated | ON median s | OFF median s | ON tok/s | OFF tok/s | ON/OFF throughput | Latency reduction |
|---|---:|---:|---:|---:|---:|---:|
| p16-g16 | 1.547378 | 1.590105 | 10.3401 | 10.0622 | 1.0276x | 2.69% |
| p16-g64 | 4.334521 | 4.696210 | 14.7652 | 13.6280 | 1.0834x | 7.70% |
| p128-g16 | 1.555517 | 1.802811 | 10.2860 | 8.8750 | 1.1590x | 13.72% |
| p128-g64 | 4.325300 | 5.551068 | 14.7967 | 11.5293 | 1.2834x | 22.08% |

Each axis/arm passes every cold and warm token ID against the three identical pinned-oracle `greedy_ids.npy` rows. Total32/32 completions,1280 generated token IDs exact;24 measured warm repetitions. All subprocesses exit0. Oracle artifact paths and SHA256 hashes appear in each per-arm JSON. The immutable TinyLlama checkpoint hash is `9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`.

NVML sampling every10ms reports equal ON/OFF incremental peak GPU memory for each axis:960626688 bytes at prompt16,969015296 bytes at prompt128. No memory reduction claim. Per-arm JSON records process wall time, each cold/warm completion time, median/mean/min/max warm times, throughput, baseline/peak/incremental memory and exact command/environment.

The adjacent raw logs, manifest, per-arm measurements and summary preserve the measured campaign. `run.py` preserves the reproduction runner. Run it through `flock /home/user/gpu.lock python3 run.py` in the recorded environment after preserving existing output logs. The registered operator gate remains197 numerical cases, not385.

These comparisons establish a d64 attention improvement against the same binary's rollback on four real TinyLlama axes. They do not establish an absolute pinned-oracle performance floor or current integration correctness.

The measured attention source is `22d433baf`, before the later d128 explicit-override repair `c93e92b94`. Independent review passes that repair; d64 remains unchanged. This executed binary does not include the later Pascal Q5/Q6 decode corrections or the D4 prefill quantization changes. Its 32 exact completions do not validate those later quant candidates. At this evidence commit, current Smol token parity and the later Q6-prefill Tiny regression remain FAIL, as reported by the operator. No final all-kernel gate or correctness waiver is claimed.

The executed link uses `/tmp/realmodel-review/public-test.o`, `/tmp/pascal-attention/attention-final.o`, and `/tmp/tiny-repair/qwen-green.o`, with `/tmp/pascal-attention/isolated.a`. The thin archive references the pre-quant-correction Tiny repair archive and excludes its old attention object. The Tiny output-head fix is present. `executed-objects.json` records retained object hashes and the exact measured executable hash. No binaries, model arrays or trace databases are committed.
