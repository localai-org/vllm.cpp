# TT capture economics — the TPOT-vs-output-length curve on the serving arm (2026-09-27)

Branch `row/TT-CAPTURE-ECONOMICS` @ `514441044` (the spec commit) + this
row's change. Spec: `.agents/specs/tenstorrent-capture-economics.md`. Issue:
`ISSUE-LOCAL-01M3HB24YKDRXTXY85R4CG7VF6`. Host: personal Tenstorrent Blackhole
P150a workstation (aarch64, GCC 16); every device command under `flock -x
$HOME/gpu.lock`, `~/Sources/tt/luwen/target/release/reset` (+15 s, retried 3x
on the 101 flake), `source ~/Sources/tt/env-tt-common.sh`. vllm.cpp: Release
Ninja build `/tmp/row-capture-econ/build` (`-DVLLM_BUILD_TESTS=ON
-DVLLM_CPP_TENSTORRENT=ON`, the tt-metal-pin cmake prefix), the recorded pin
`vllm-cpp-pin/20260925` (`d20b8e27f29`) — build-linkage and runtime trees
verified at the same commit, both clean. Model: the anchor APEX I-Nano
(`/mnt/models/mudler-qwen3.8-27B-APEX-gguf/Qwen3.8-27B-APEX-I-Nano.gguf`,
11,240,605,152 bytes). Recipe: the recorded anchor leg verbatim (`--num-prompts
4 --concurrency 1 --seed 0 --temperature 0`, ignore-eos by the bench default,
fixture `tests/fixtures/tt-int8dot-sweep-sharegpt-64-20260923.json`, raw
prompts), `VT_TT_STEP_PHASES=1 VT_DECODE_GRAPH_STATS=1`, `--output-token-ids`
per leg, ONE FRESH PROCESS PER LEG. Logs under `/tmp/capture-econ/`.

## The build workaround (a found bug, owed to its own unit)

A fresh configure no longer fetches parakeet.cpp: `CMakeLists.txt:1597` pins
`GIT_TAG main`, and `mudler/parakeet.cpp` no longer has a `main` branch (the
default is `master` @ `2bf8895`, whose header lacks `parakeet_capi_diarize_pcm`
— the diarization line lives on `feat/diarization-sas` @ `5495716`, which
carries the API `src/vllm/multimodal/diarization.cpp:89` calls). This row's
build used a local clone of `feat/diarization-sas` @ `5495716` through
`-DFETCHCONTENT_SOURCE_DIR_PARAKEET_CPP=/tmp/capture-econ/parakeet.cpp` (the
documented FetchContent source override; the
`VLLM_CPP_PARAKEET_CPP_DIR` knob does not define the `parakeet` target and
dead-ends at `-lparakeet` at link time). The clone needed its ggml submodule
(`e705c5fed490514458bdd2eaddc43bd098fcce9b`) and two
`CMAKE_SOURCE_DIR` → `CMAKE_CURRENT_SOURCE_DIR` corrections in its own
CMakeLists (the third_party include dir at :112 and the ggml-patch runner at
:48-52), because parakeet.cpp's CMake assumes a top-level build and breaks as
a FetchContent subproject. No vllm.cpp tree file changed; the binary is the
same tree with the same compile definitions as the #3327/#3322 builds. The
`GIT_TAG main` repair is a one-line change on a product path and is named as a
next target below.

## The measurement

Four legs — `--output-len` 4, 16, 64, 256 — each a fresh process (the
per-request retention ceiling, ~5 requests/process, stands; the ttnn
deferred-reader issue is still open upstream), 4 prompts, concurrency 1,
greedy, ignore-eos, the committed fixture's first four prompts. The phase
instruments are read-only (`VT_TT_STEP_PHASES`'s documented invariance: the
anchor's token stream is byte-identical with the knob set; each leg's
`--output-token-ids` file records what the leg actually served). The
served-replay gate ran green on this fresh binary BEFORE the legs (the
correctness state): `test_qwen35_paged_engine -tc='*SERVES replays*'`, 2/2
cases, 6/6 assertions, SUCCESS (`served-replay-test.log`).

## The curve

| output_len | TPOT mean (ms) | TPOT median (ms) | TPOT P99 (ms) | TTFT mean (ms) | E2EL mean (ms) | bench duration (s) | replays (served + capture-own) | tokens sha256 |
|---|---|---|---|---|---|---|---|---|
| 4 | 34,280.9 | 34,330.8 | 34,366.5 | 297,661.3 | 400,504.1 | 1,602.0 | 8 (4 + 4) | `13c3f70b…` (the recorded anchor, byte-identical) |
| 16 | 31,754.1 | 31,749.1 | 31,793.9 | 297,896.2 | 774,207.1 | 3,096.8 | 56 (52 + 4) | `53c7be18…` |
| 64 | 31,283.2 | 31,277.2 | 31,313.6 | 297,796.7 | 2,268,636.4 | 9,074.6 | 248 (244 + 4) | `38bdb003…` |
| 256 | in flight | in flight | in flight | — | — | — | — | launched 18:27, one fresh process, the same recipe; the flat 244-replay line puts the fit's prediction at ~31,166 ms — recorded when it lands, never before |

The 4-token leg reproduces #3327's recorded 34,165.25 ms mean at +0.34%
(within noise; same 1,602 s duration, same phase mix, and the served stream
sha256 byte-identical to the recorded anchor `13c3f70b…`) — the
reproducibility gate.

## The phase mix

Per request, the decode graph runs `output_len - 1` steps: one eager COLD
step (host-dispatched forward), one CAPTURE step (the record pass + tt-metal
trace finalize), and `output_len - 3` served REPLAY steps. The step-phase
line's `wall_ms` is the step's own host window; `gap_prev_ms` is the CALLER's
window (runner + sampler) between the previous step's return and this step's
entry.

| leg | cold wall/step | capture wall/step | replay wall/step | replay `gap_prev`/step | cold `gap_prev` (the prefill) |
|---|---|---|---|---|---|
| 4 | 35,128 / 34,149 / 34,306 / 34,465 ms | 4,782 / 6,621 / 6,290 / 6,328 ms | 17.7 / 49.9 / 51.7 / 58.9 ms | **31,165.7 / 31,119.1 / 31,119.3 / 31,114.2 ms** | 323,296 / 322,875 / 322,751 ms |
| 16 | 35,067 / 34,066 / 34,922 / 34,244 ms | 4,861 / 5,941 / 6,067 / 5,923 ms | 7.2-55.4 ms (mean 14.7; first-of-request 14.5 / 55.4 / 52.2 / 52.5 ms, the rest 7-20 ms) | **31,107-31,170 ms — EVERY replay step, metronome-flat** | 323,456 / 323,132 / 322,982 ms |
| 64 | 35,071 / 34,102 / 34,348 / 34,239 ms | 4,713 / 5,935 / 6,054 / 5,954 ms | 7.4-46.8 ms (mean 15.3; first-of-request 41.0-46.8 ms, the rest 7.4-23 ms) | **31,100-31,172 ms — every one of the 244 replay steps** | 323,530 / 322,654 / 323,036 ms |
| 256 | in flight (launched 18:27); the row records when it lands |

Two structural facts fall out:

1. **The ~31.1 s caller window is PER REPLAY STEP, not per request.** At
   output_len 4 the arm runs one replay per request, so the recorded
   decomposition (#3322: "the trace-execution wait, 31.1 s/request, in the
   next step's gap") could not distinguish per-request from per-token. At 16
   and 64 it recurs before EVERY replay step, metronome-flat (31,107-31,170
   ms), and it does not grow with the request count (request 1's replay gap
   31,169 ms vs request 4's 31,114 ms at len 4 — flat while the retention
   staircase advances) nor with the position within a request.
2. **The served step's own wall stays 12 ms class** (7-20 ms steady state;
   each request's first replay ~55 ms), and the retention read-out is flat
   (dram_free 5,102 MiB through every replay of a request; per-request entry
   6,280 → 4,969 → 4,967 → 4,967 — the staircase steps once at request 1's
   capture and then holds; 4,967 MiB still free at request 4). Nothing that
   the length could grow — GDN state windows, KV pages, keepquant transients —
   shows up in either the wall or the gap.

The seam-side `sync` lines (4 per process, one per request at its completion)
read `wait_ms` 1.6-7.9 ms for `read=688128` bytes: the per-request completion
read does NOT sit in a 31 s device wait. The ~31.1 s is spent in the caller
window between step returns, where no instrument currently brackets.

## The model

The bench's TPOT divides the decode window by `output_len - 1`; the physical
per-request decode wall (TPOT × (len−1)) closes at each leg as cold + capture
+ (len−2) × 31.13 s, leaving a fixed residue of 40.5 / 40.1 / 40.8 s (the
cold + capture pair, 40.1 s mean):

```text
decode(len) ≈ 40.5 s  +  31.13 s × (len - 2)
```

(the wait lands once after the capture step, the completion window once after
the last replay; both are the same ~31.1 s class). Fitting the spec's
`TPOT(len) = fixed/len + per_token` by least squares over the completed legs
(4, 16, 64):

```text
TPOT(len) = fixed/len + per_token            (the spec's literal form):
  fixed     =  12,986 ms   (13.0 s)
  per_token =  31,019 ms   (31.0 s)
  len=   4: measured 34,280.9  fitted 34,265.6  resid +15.3
  len=  16: measured 31,754.1  fitted 31,830.7  resid -76.6
  len=  64: measured 31,283.2  fitted 31,221.9  resid +61.2

TPOT(len) = fixed/(len-1) + per_token   (the bench's divisor; the reported fit):
  fixed     =   9,454 ms   (9.5 s)
  per_token =  31,129 ms   (31.1 s)
  len=   4: measured 34,280.9  fitted 34,280.1  resid  +0.8
  len=  16: measured 31,754.1  fitted 31,759.1  resid  -5.0
  len=  64: measured 31,283.2  fitted 31,278.9  resid  +4.2
```

The (len−1) fit lands `per_token` at 31,129 ms — the measured per-replay-step
caller window (31,100-31,172 ms) — and its residuals are ±0.02% of the
measurement; the model is the curve.

## The verdict

**The per-token term dominates at every real generation length.** The fixed
pool (`cold` ~34.4 s + `capture` ~5.7 s, ~40.1 s once per request) amortizes
as 40.1/(len-1): 13.4 s/token at output_len 4 (39% of TPOT), 2.7 s/token at
16 (8%), 0.6 s/token at 64 (2%) — and the fit's per-token coefficient,
31,129 ms, is the measured per-replay-step caller window (31,100-31,172 ms)
to within 0.05%. The curve's entire bend lives in the first 16 tokens; from
16 on it is the flat 31.1 s/token line, and the in-flight 256 leg exists to
confirm the asymptote the fit already pins at ~31,166 ms. The
capture-economics fixed costs are NOT what real-length generation pays:
**the ~31.1 s caller window between served replay steps is 98-99% of the
bill at any length past a handful of tokens.**

The spec's provisional reading bound is resolved the other way: per-token
does NOT stay 12 ms class, and it does NOT grow with length either — it is
constant. The "served replays are 12 ms" of #3327 was the STEP's own wall;
the token PERIOD is the step plus the ~31.1 s caller window that follows it.
The #3322 attribution ("the trace-execution wait, ~30 ms per replayed
command over the 1,037-entry trace, 31.1 s") predicted this exact value, and
its mechanism is the leading candidate for the caller window: the replayed
trace executes at ~30 ms/command on device (its kernels sum to ~1.3 s), and
the caller waits for it before the next step. The alternative candidate is a
per-step deferred-reader materialization (tt-metal#57970's retention flushed
by the per-step token read-back); the flat-across-requests gaps (31,169 ms
at request 1 vs 31,114 ms at request 4, while the retention advanced) and
the flat within-request walls both fit the constant per-command trace
overhead better than a retention-proportional flush, but the caller window
itself is unbracketed — no instrument today splits host-wait from
device-busy inside it.

## The named next targets

1. **The per-replay-step ~31.1 s caller window** (the dispatch row's
   denominator, 98-99% of real-length TPOT): bracket it. A step-decompose-2 row
   with a per-replay-step completion probe (the `=sync` serialized probe
   extended past the layer loop, or a device-side timestamp on the trace's
   last command) splits host-wait vs device-busy and names the tt-metal lane
   (~30 ms/command trace-replay overhead) vs a host-side lane. This row's
   legs are its red-free baseline: 31,100-31,172 ms, flat in length, request
   count, and position.
2. **The trace's per-command overhead itself**: 1,037 commands × ~30 ms =
   31.1 s of device trace execution for ~1.3 s of kernels — the same
   magnitude as the eager host dispatch it replaced (the cold body's ~32
   ms/op). The captured arm currently buys ~nothing over eager; the lever is
   per-command, not per-trace.
3. **The cold step and the capture pass** (the actual capture economics,
   ~40.5 s/request): real once amortized over ≥16 tokens, but it bounds the
   short-length arm (39% of TPOT at len 4) and the capture pass is the only
   part with a known cheap fix direction (#3322: record 2.6 s + finalize
   1.7 s of the 6.1 s).
4. **The `GIT_TAG main` pin rot** (this row's build blocker): a one-line
   repair on `CMakeLists.txt:1597` to a ref that exists
   (`feat/diarization-sas` @ `5495716` or a fork pin) — a fresh configure is
   currently broken for every row.

## Gates

| Gate | Verdict |
|---|---|
| Served-replay test green on this binary (the correctness state) | **PASS** — `test_qwen35_paged_engine -tc='*SERVES replays*'`: 2/2 cases (the anchor byte-identical, the 9B up to the oracle-tied cell), 6/6 assertions, SUCCESS, exit 0. Log `served-replay-test.log` |
| Reproducible: the 4-token leg vs #3327's 34,165.25 ms | **PASS** — 34,280.9 ms mean (+0.34%), same 1,602.0 s duration, same phase mix, served tokens sha256 `13c3f70b…` byte-identical to the recorded anchor. Log `leg-len4.log` |
| Tokens at each leg recorded | **PASS** — `--output-token-ids` per leg; len 4 byte-identical to the recorded anchor; len 16/64 streams recorded (`53c7be18…`, `38bdb003…`; different lengths produce different streams — byte-comparison across lengths is not a gate per the spec); len 256 in flight |
| Phase instruments read-only | **PASS by the recorded invariance** — the len-4 leg with both knobs set reproduces the recorded anchor's token stream byte-for-byte (the ENVIRONMENT.md contract), and the legs' phase lines are the only stderr traffic |
| The curve completes | **4/16/64 COMPLETE, PASS** — each in one fresh process (the ~5-request retention ceiling held at 4 prompts; per-request entry DRAM 6,280 → 4,969 → 4,967 → 4,967 MiB, flat after request 1; 12/60/252 step lines, no device fault, no reset retry needed). **256 IN FLIGHT** (launched 18:27, detached, same recipe; ETA ~9 h at the flat rate — the spec's design: 4/16/64 discover, 256 confirms; recorded when it lands) |
| check-commit-style + check-commit-trailers (`origin/main..HEAD`) | **PASS** (rerun on this branch's commits before the handoff) |
| check-agent-record | **RED at HEAD, pre-existing/main-inherited** — 598 dangling-link errors across other rows' `.agents/issues/`, `.agents/completed/`, plus two claim rows referencing unknown rows and the model-matrix module count (309/245 vs 310/261); none name this row's files or issue. The task brief records this as repo-wide rot owned by another unit; not fixed here |

## The legs' raw record

- `leg-len4.log` — 12 step lines, TPOT 34,280.9/34,330.8/34,366.5 ms, 8
  replays, EXIT=0.
- `leg-len16.log` — 60 step lines, TPOT 31,754.1/31,749.1/31,793.9 ms, 56
  replays, EXIT=0.
- `leg-len64.log` — 252 step lines, TPOT 31,283.2/31,277.2/31,313.6 ms, 248
  replays, EXIT=0, 9,074.6 s.
- `leg-len256.log` — in flight (launched 18:27); the row updates when it lands.
- `served-replay-test.log` — the correctness gate, 2/2 cases SUCCESS.
- `check-agent-record-head.log` — the pre-existing record rot, 602 lines.
