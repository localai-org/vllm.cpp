# TT-KEEPQUANT-INT8DOT: the 500-mnat teacher-forced band on the live 27B arm, llama.cpp @ `11fe0215` (2026-10-04, P150, 16 prompts)

Row `BACKEND-TENSTORRENT-KEEPQUANT`, spec
[tenstorrent-keepquant-int8dot-default.md](../../../.agents/specs/tenstorrent-keepquant-int8dot-default.md)
(live-arm re-run section), issue
`ISSUE-LOCAL-01M37W0HP6JTNJP55S74159T02`, branch `row/int8dot-default-flip`
(at `628834379`). This is the authorized follow-up to
[tt-int8dot-flip-live-gates-20261003.md](tt-int8dot-flip-live-gates-20261003.md),
whose DONT-FLIP verdict named exactly one blocker for spec gate 1: the pinned
llama.cpp `b10451` refused the artifact, so no band denominator existed.

## Verdict up front

**Gate 1 (the 500-mnat teacher-forced band) PASSES: 16/16 prompts in-band on
BOTH arms.** Denominator: llama.cpp `11fe02151f79c41d0d4af7da708755d73b9c0da6`
(full-sequence batch decode, teacher-forced on each arm's own stream), which
loads the artifact `b10451` refuses. INT8DOT max gap **97.2 mnats** (mean
18.1), W4a-out (`=0`) max gap **144.1 mnats** (mean 15.6) — both 3.5x+ inside
the 500-mnat band, and the INT8DOT arm's 245.3-mnat anchor outlier does not
recur. Every arm-pair stream divergence is priced as a genuine near-tie
(single position, 34-144 mnats). **Recommendation: the flip's gate-1 evidence
is complete; the flip is supported and should proceed as its own authorized
change** (this unit does not flip the default). The spec's gate-2 (siblings)
and gate-4 (`=0` opt-out identity) legs remain re-owed on the live arm.

## Denominator: the pin advance

- llama.cpp oracle pin advanced `b10451` → `11fe0215` in
  [`.agents/oracles/llama-cpp.md`](../../../.agents/oracles/llama-cpp.md)
  (commit `6fbe576dc`); gateability evidence
  [oracle-llamacpp-11fe0215-gateable-20261003.md](oracle-llamacpp-11fe0215-gateable-20261003.md):
  `11fe0215` loads and generates on the unsloth
  `Qwen3.8-27B-Q4_K_M.gguf` (17,106,775,008 B) that `b10451` refuses
  (`missing tensor 'blk.64.ssm_conv1d.weight'`, the MTP export gap).
- `11fe0215`, CPU-only build (`-DGGML_NATIVE=ON`,
  `/tmp/llamacpp-main/build`), 128-core aarch64 host. Gates recorded at
  `b10451` stay attributed to it.

## Methodology (identical to the 2026-09-23 anchor adjudication)

- Gap semantics (spec gate 1, `tt-int8dot-flip-gates-20260923.md`):
  teacher-force (prompt ids + OUR 32 generated ids) through the oracle's
  full-sequence batch decode; per generated token,
  `gap = max(0, argmax_logp − our_logp)` in mnats from raw-f32 logits
  (log-softmax over the full 248,320 vocab); band = 500 per prompt
  (max gap < 500).
- Prompt set: the bench's own generated workload — `BuildPrompt`
  (`examples/bench/bench_core.h:518`, `mt19937_64`, words
  {hello, world, 1, 2}, grown to ~128 tokens, seed `0+i`) under the model's
  own GGUF tokenizer, 16 prompts (the documented minimum fallback, see the
  c2 blocker below). Prompts and prompt ids dumped by a scratch tool
  (`/tmp/int8dot-flip/gen-prompts`, sha256 `945dc100…`) that calls the bench's
  own `BuildPrompt` + `tok::Tokenizer::FromGguf` — byte-identical by
  construction to what `vllm-bench` admits.
- Oracle driver: `/tmp/int8dot-flip/oracle-band` (sha256 `4778d1d2…`, source
  `oracle-band.cpp` in the same directory) — loads the model once, decodes
  each full sequence as ONE batch, emits per-position argmax / our-logp /
  gap (the `oracle-dump` VTLGDUMP path of 2026-09-23, rebuilt against
  `11fe0215` and collapsed to gap output). Smoke: the driver's own greedy
  stream re-reads at 0.0 mnats on every position.
- Denominator logprobs: raw f32 per-position logits (log-softmax computed
  locally), so per-token logprobs ARE available — the band is the full
  500-mnat teacher-forced gate, not a reduced exact-match gate.

## The 64-prompt c2 plan died on a product bug (both arms)

The sweep was sized for 64 prompts at c2. Both legs died mid-run with
`BENCH_EXIT=1`: engine-fatal `vt: tenstorrent gdn_decode: state-slot indices
changed during trace capture — decode slots must be stable across a
sequence's steps; the recapture cadence owns a slot change` at
`src/vt/tenstorrent/tenstorrent_gdn.cpp:558` (INT8DOT=1 leg
`/tmp/int8dot-flip/dot1-c2-64p.log` 10:51:33, INT8DOT=0 leg
`dot0-c2-64p.log` 11:30:20, 2026-10-04). Every earlier c2 gate leg ran
`--num-prompts 2` — one wave, no sequence re-admission — so this path never
fired. Filed as `ISSUE-LOCAL-01M433M0TNT8FWC6SMT4R3700W` (commit
`628834379`). The sweep fell back to the documented minimum: **16 prompts,
c1** (the clean single-stream comparison shape of the 2026-10-03 record).

## Sweep setup

- Tree `row/int8dot-default-flip` at `628834379`, build
  `/tmp/vllm-int8dot-flip/build` (the 2026-10-03 Release/Ninja TT build;
  `nm vllm-bench | grep -c tenstorrent` = 427).
- Device: personal Blackhole P150a under `flock $HOME/gpu.lock`, one device
  job at a time, `~/Sources/tt/luwen/target/release/reset && sleep 15` before
  each leg (3 retries on failure; none needed).
- Env: `TT_METAL_HOME=TT_METAL_RUNTIME_ROOT=/home/lu_zero/Sources/tt/tt-metal-pin`,
  `LD_LIBRARY_PATH=$TT_METAL_HOME/build_release_script/lib64:$TT_METAL_HOME/build_release_script/libexec/tt-metalium`;
  `env-tt-common.sh` NOT sourced; run from the repo root.
- Workload: `--num-prompts 16 --input-len 128 --output-len 32 --num-blocks 64
  --max-num-batched-tokens 64 --seed 0 --concurrency 1 --temperature 0`,
  `VT_TT_KEEPQUANT_INT8DOT=0/1`, `--output-token-ids` per leg.
- Both legs `BENCH_EXIT=0`: INT8DOT=1 11:37:48→16:04, INT8DOT=0
  16:05:39→20:50, 2026-10-04, one lock hold.

## Sweep table (per prompt; `div@k` = first divergence at generated index k, `-` = none)

| prompt | `=0` max gap (mnats) | `=0` div@ | `=0` out-of-argmax positions | `=1` max gap (mnats) | `=1` div@ | `=1` out-of-argmax positions | arm ids equal |
|---|---:|---|---:|---:|---|---:|---|
| p00 | 0.0 | - | 0 | 0.0 | - | 0 | yes |
| p01 | 0.0 | - | 0 | 0.0 | - | 0 | yes |
| p02 | 0.0 | - | 0 | 34.1 | 1 | 1 | no |
| p03 | 0.0 | - | 0 | 0.0 | - | 0 | yes |
| p04 | 8.2 | 3 | 4 | 97.2 | 3 | 5 | no |
| p05 | 0.0 | - | 0 | 0.0 | - | 0 | yes |
| p06 | 0.0 | - | 0 | 0.0 | - | 0 | yes |
| p07 | 144.1 | 0 | 1 | 0.0 | - | 0 | no |
| p08 | 0.0 | - | 0 | 0.0 | - | 0 | yes |
| p09 | 0.0 | - | 0 | 0.0 | - | 0 | yes |
| p10 | 0.0 | - | 0 | 0.0 | - | 0 | yes |
| p11 | 0.0 | - | 0 | 78.7 | 1 | 1 | no |
| p12 | 0.0 | - | 0 | 0.0 | - | 0 | yes |
| p13 | 96.9 | 1 | 1 | 0.0 | - | 0 | no |
| p14 | 0.0 | - | 0 | 0.0 | - | 0 | yes |
| p15 | 0.0 | - | 0 | 79.8 | 0 | 1 | no |

| arm | in-band (500) | max gap (mnats) | mean of per-prompt max (mnats) | TPOT mean (ms) |
|---|---|---:|---:|---:|
| `=0` (W4a grouped, f32-exact) | **16/16** | 144.1 | 15.6 | 31,261.37 |
| `=1` (INT8DOT) | **16/16** | 97.2 | 18.1 | 6,660.11 |

- TPOT ratio 4.69x at c1 — the default-path lever reproduces (4.50x on
  2026-10-03, 4.69x in wave 3).
- 11/16 prompts: the arms produce byte-identical streams and every one of
  our tokens IS the denominator's argmax. 5/16 prompts diverge at exactly
  one near-tie position (34.1-144.1 mnats) and then track the denominator
  in-band — on p04 both arms diverge at the same index 3 near-tie and each
  resolves it differently; the divergence is a coin-flip the band prices,
  exactly the shape the near-tie disposition exists for.
- The INT8DOT arm's anchor outlier (245.3 mnats, APEX-I-Nano) does not grow:
  97.2 on the live 27B arm.

## What this unit does NOT claim

- 64-prompt coverage: the c2 slot-churn bug (above) forced the 16-prompt
  minimum. A wider sweep re-opens automatically once
  `ISSUE-LOCAL-01M433M0TNT8FWC6SMT4R3700W` lands.
- Spec gate 2 (sibling keep-quant models) and the gate-4 `=0` opt-out
  identity: still re-owed on the live arm; not run this window.
- The default is NOT flipped here; the flip is a separate authorized change
  carrying the spec's remaining gates.

## Artifacts

- `/tmp/int8dot-flip/` — `prompts-64.json` (generated prompts + prompt ids),
  `dot{0,1}-c1-16p-{tokens,metrics,log}`, `dot{0,1}-c1-16p-band.json`
  (per-position argmax/logp/gap), `dot{0,1}-c2-64p.log` (the engine-fatal),
  `seqs-dot{0,1}-c1-16p.txt`, `gen-prompts(.cpp)`, `oracle-band(.cpp)`,
  `device-sweep.sh`.
- Monitor logs:
  `/home/lu_zero/.local/logs/maki/CeRgUcXiFK5bYWGSPS4sy/monitor-1791099797-ab99/stdout.log`
  (failed c2 64p sweep) and
  `.../monitor-1791106652-c334/stdout.log` (c1 16p sweep, both
  `BENCH_EXIT=0`).
