Fresh promoted batched-Q1 Intel Arc Pro B70 serving comparison: **vLLM.cpp vs Python vLLM**, same pinned Qwen3.8-27B EXL3 4.00-bpw model, FP16 / FP32 GDN / FP8 KV / MTP3.

Both engines completed the original 20 scenarios / 70 measured waves / 124 requests plus three isolated 64K prefix requests. All measured prompts and request metadata match the frozen originals; all responses reached 1,024 tokens with zero preemptions.

**Single-request decode** (original repeat counts; 200K is one observation):

| Input budget | C++ decode tok/s | Python decode tok/s | C++ vs Python |
|---:|---:|---:|---:|
| 512 | 54.86 | 56.19 | -2.4% |
| 2,048 | 51.86 | 59.23 | -12.4% |
| 4,096 | 54.04 | 56.79 | -4.8% |
| 8,192 | 49.57 | 55.60 | -10.8% |
| 16,384 | 52.01 | 56.10 | -7.3% |
| 32,768 | 49.56 | 51.90 | -4.5% |
| 65,536 | 44.86 | 50.49 | -11.1% |
| 131,072 | 39.48 | 42.58 | -7.3% |
| 199,680 | 36.41 | 37.52 | -2.9% |

**Four simultaneous requests**, aggregate throughput in the fully overlapping decode window:

| Input budget / request | C++ C4 aggregate tok/s | Python C4 aggregate tok/s | C++ vs Python |
|---:|---:|---:|---:|
| 2,048 | 149.81 | 184.43 | -18.8% |
| 4,096 | 141.05 | 186.74 | -24.5% |
| 16,384 | 129.26 | 161.46 | -19.9% |

Positive percentages mean higher C++ throughput; negative percentages mean lower throughput. They are not extra-duration percentages. The full English HTML report and PNG/SVG charts include prefill, prefix reuse, TTFT, MTP acceptance, ranges, exact conditions and raw data.

**Counter-window qualification:** A post-capture audit found one leading final-prefill sample in a native C4-4K window. The same deterministic final-prefill plateau rule is applied to every window in both engines before unchanged strict counter checks. The native wave changes141.018→140.186 tok/s; the other69 native matrix windows are unchanged. Raw data/driver values remain intact; no rerun or relaxed gate. TTFT/E2E, C1 request-duration metrics and output/work counts are unchanged.

**Change and limits:** Batched pure-Q1 B2–B4 is now default 1 (explicit 0 rollback), with auto/2, SiLU 1 and mixed-prefill 0. Its separate same-binary three-pair screen showed C4-16K aggregate decode +6.07% to +6.92% (median +6.60%), median TTFT +0.013% and median E2E −2.65%. All 4K gains were below observed A/A variation and are not independently noise-qualified. Expanded SiLU 2 remains experimental. The opt-in mixed-prefill correction completed its bounded MTP0 C2 case in 101.980s at terminal positions 65,500/65,500; default MTP0 C2 is not freshly qualified. Serving completion is not universal quality PASS. The known strict closing-fence failure persists despite recovered code passing32/32 checks. Inherited cold costs and lost short warm hits remain disclosed. Engine scheduling, library policies and stochastic acceptance differ.

Native binary: `c87739a6986b13de1240213ba0ed12075cea05b007ec6cc79c754e5042ed9f07`. Model revision: `113cf7ab958054860e43fb7f3063b1af19171095`.

Attach the supplied context-throughput, relative-throughput, concurrency-throughput and prefix-ttft PNG files with the downloadable HTML report.
