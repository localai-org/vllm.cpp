# Spec: region-scoped decode capture for the Tenstorrent 27B — the whole graph does not fit, so capture what recurs

Row: `tt-27b-region-capture`. State: DRAFT (2026-09-28).
Issue: `ISSUE-LOCAL-01M3JXEFQKSZP23PP2HWY9G0VQ` (Blocker B owns the
trace-budget analysis this row implements; the row closes the 27B
decode-trace DRAM-fit site of that issue).
Builds on: the capture-safety doctrine landed on
`row/tt-27b-capture-write` — EnsureDevice2D one-chain (`ab7cdb359`),
the fresh-slot Memset shadow in both passes bounded to ≤ 64 KiB
(`965822766`), the batched decode RAC per-user device path
(`e39f2cf3f`), and the PA stale-guard deletion + page-table width
guard (`809067ccb`). The OWED RAC flake from `e39f2cf3f` (the doctest
order-sensitive under full-suite program-cache history) carries forward,
not into this row's scope.
Precedent: `.agents/specs/tenstorrent-gdn-region-replay.md` — region-
scoped capture already serves on the 9B (served == eager byte-identical,
8 replays > 4 captures, the in-place state commit at
`src/vt/tenstorrent/tenstorrent_gdn.cpp:933` and `:2441`).
Git integration: ONE pull request (spec + implementation + gates),
developer decision recorded 2026-09-28 in
`.agents/developer-preferences.md` under `## Git integration`.

## Scope

The Qwen3.8-27B dense GDN-hybrid decode step on Tenstorrent (the c2 leg,
Q4_K_M anchor arm) cannot use whole-graph trace capture: the captured
graph records 1,037 tt-metal commands and `end_trace_capture` asks for
one 3,153,969,152 B DRAM staging buffer (3.04 MB per recorded command)
against 298,568,896 B free; even a full tt-metal#57970 retention
recovery (~948 MiB back, to ~2.2 GiB) leaves 3.15 GB > 2.2 GB. The
demand is per-command launch RECORDS — descriptors, CB/semaphore state,
runtime args — not tensor staging (every in-region tensor is persistent
under the warmup discipline) and not program binaries (cached outside
the trace). This row replaces whole-graph capture for the 27B with
per-layer compute regions captured and replayed per layer from a host
loop, keeping the heterogeneous preamble, RAC, and sampling eager.

Whole-graph capture stays the shape for every model it already fits. The
row ships a size/fit predicate (below), not a flip of the default.

## Upstream anchors

- vLLM capture semantics: CUDA-graph capture in vLLM is per-iteration,
  whole-forward, and falls back to eager when capture does not fit
  (`vllm/worker/gpu_model_runner.py` — the capture-size/eager fallback
  path). vLLM never captures the sampler or the per-step host logic;
  the captured region is the model forward. This row mirrors that
  polarity one level finer: capture the per-layer forward compute, keep
  the step plumbing eager. The eager fallback behavior (decline capture,
  serve eager, name the missing part) is the vLLM behavior this row's
  predicate inherits.
- tt-metal `MeshTrace` constraints: the trace buffer is the replay
  staging DRAM for the whole recorded command stream;
  `MeshTrace::populate_mesh_buffer` / `bank_manager.cpp:495` OOM at
  3,128,655,872 B requested vs 278,858,624 B free / 266,655,872 B
  largest block (the c2c leg, `/tmp/leg-27b-c2c.log`, 2026-09-28), and
  3,153,969,152 B vs 298,568,896 B free on the c1 leg. Mid-trace host
  writes and readbacks are fatals (`fd_mesh_command_queue.cpp:826` and
  `:873`), which is the constraint the landed doctrine already serves.
- The per-command economics: ~30 ms/command trace execution
  (`tt-27b-step-decompose-20260926.md`, the legE in-capture census,
  1,037 entries identical across all four captures) and
  `tt-capture-economics-20260927.md` (the ~31.1 s caller window is
  98-99% of real-length TPOT).

## Design

### What is captured: per-layer compute regions

The 27B decode step is a homogeneous inner loop: 64 layers of
GDN/attention + GEMM + norms between one heterogeneous preamble and the
RAC/sampling tail. Capture ONE region per layer (or per contiguous block
of layers, if the boundary discipline below prices blocks cheaper): the
region contains that layer's kernel-dense compute chain and nothing
else. The eager preamble (embedding, first norms), the RAC KV write,
the LM head, and the sampler stay OUTSIDE the trace — the same
polarity vLLM captures with (model forward only, never the sampler),
applied per layer.

### Region budget math

- The GDN precedent's fit number: the chunked E=1 arm targets a 50 MiB
  trace region (`src/vt/tenstorrent/tenstorrent_capture.cpp:90`,
  `LastTraceBytes()` discipline) and the 9B GDN region captured and
  replayed inside it.
- Today's free pool admits ~90 commands: 90 × 3.04 MB ≈ 274 MB against
  298,568,896 B free.
- The 50 MiB budget admits ~16 commands: 16 × 3.04 MB ≈ 48.6 MiB.
- The 27B layer census: 1,037 commands / 64 layers ≈ 16.2 commands per
  layer. **The derived budget: ≤ 50 MiB per region, i.e. ≤ ~16 commands
  per captured region** — one layer per region lands on the 50 MiB
  precedent's budget exactly. A layer block that stays ≤ ~90 commands
  (~274 MB) is admissible only if the boundary-count risk (below)
  prices it cheaper; the 50 MiB per-region cap is the spec's default,
  and `LastTraceBytes()` asserts it at capture time (a region over cap
  declines capture for that region and serves it eager, named).

### Region boundary and state handoff discipline

Every cross-region value (residual stream, GDN ssm/conv state slots, KV
pages, rope outputs) lives in the persistent preallocated shadows the
trace reads and writes IN PLACE — the proven W3/W4 in-place commit
pattern (the 9B GDN region's `tenstorrent_gdn.cpp:933`/`:2441`
discipline, extended fleet-wide by `965822766` and `e39f2cf3f`). No
region boundary installs, frees, or re-shadows a state tensor: a fresh
device tensor at a commit invalidates the address a captured region
baked (the #3327-class defect — the region re-reads the freed block and
re-scatters garbage over the live slot). Warmup runs every region
eagerly first, so every captured call is a program-cache hit — the W4
doctrine, one chain in both passes, no capture-active branch. The host
loop replays region after region, re-patching per-region runtime args
(the RAC per-user `override_runtime_arguments` mechanism) between
regions; per-region `LastTraceBytes()` and the boundary counter are the
fit and handoff instruments.

### Whole-graph stays for small models: the size/fit predicate

Capture scope is decided per model by a fit predicate evaluated before
the first capture: estimate the whole-graph demand (the recorded command
count × the measured per-command cost, both from `VT_TT_TRACE_DEBUG`'s
census and `LastTraceBytes()`) against the free DRAM headroom; when it
fits (the 9B and the small classic lane), capture whole-graph as today.
When it does not, take the region-scoped path. The predicate uses
measured numbers, not model size alone; a model whose whole graph fits
never changes behavior. The 27B region-scoped arm is a named decline of
whole-graph with a message that says why, per the refused-arm rule.

## Risks

- **Region count × per-region overhead vs the ~31.1 s window.** 64
  regions × ~16 commands × ~30 ms/command ≈ 30.7 s/step of trace
  execution — the same command stream the whole graph replays, so the
  per-command term is unchanged; the NEW cost is the per-replay fixed
  overhead (host loop dispatch, runtime-arg re-patch, boundary
  bookkeeping). At a generous 5-10 ms per region boundary that is
  0.32-0.64 s/step, 1-2% of the 31.1 s window — noise against the win:
  the whole-graph arm SERVES NOTHING today (BENCH_EXIT=1 at the trace
  OOM), so region-scoped's ~31 s/step is not a regression against 31 s,
  it is the difference between a served 27B TT arm and none. The
  remaining lever on the 31.1 s itself is the dispatch-cost row's work,
  per-command, not this row's.
- **Boundary re-capture storms.** A region that declines at replay
  (shape change, page-table width change) falls to eager and may
  re-capture; a per-step storm would multiply the fixed cost 64x. The
  c2c leg survived SIX whole-graph boundary re-captures serving ~12
  minutes of batched decode (11:24:06 → 11:36:40) — re-capture is
  bounded in practice, and the landed width-change guards
  (`809067ccb`, the batched-lane guard) removed the known decline
  causes. The gate records `boundary` counts per step; a boundary rate
  above the c2c precedent fails the gate.
- **State-slot binding across regions (the #3327 class).** The whole-
  graph row's founding defect (a replay reading a freed block) recurs at
  region scale if ANY region's captured trace references a non-
  persistent buffer. Mitigation is the standing doctrine plus a red-first
  test per state class (below), not hope: every cross-region slot must
  be a persistent shadow bound before capture, and the test suite
  mutates each binding to prove the tests would catch a fresh-tensor
  commit.
- **The region-scoped arm becomes a parallel path.** Route through the
  existing capture machinery (`tenstorrent_capture.cpp`, the
  `LastTraceBytes` discipline, the RAC re-patch seam) — no new capture
  implementation beside it. A genuinely unreachable upstream behavior
  needs one exact tracked exception, not a hand-written twin.

## Tests

Red-first per region handoff (each test fails on the pre-row tree for
the right reason, mirroring the fix-train's doctest pattern):

1. **Per-layer region capture**: the 27B-shaped decode graph captures N
   per-layer regions, each `LastTraceBytes()` ≤ 50 MiB, and replays
   token-exact against the eager reference.
2. **State handoff across boundaries**: for each cross-region state
   class (residual, GDN ssm/conv, KV pages), a multi-region replay whose
   correctness requires the in-place commit — the capture installs a
   fresh tensor in the mutated variant and the test must fail there
   (the reviewer's mutation target).
3. **The fit predicate**: a model whose whole graph fits takes the
   whole-graph path (byte-identical behavior to today); one that does
   not takes the region path; neither changes the other's served stream.
4. **The predicate decline names the missing part**: an over-budget
   region declines with a message that names the command count and the
   budget.
5. **Device gate — the 27B c2 leg**: `BENCH_EXIT=0` on the anchor arm
   (`--concurrency 2`, both `VT_TT_KEEPQUANT_INT8DOT=0` and `=1`), the
   served stream token-exact vs eager, and the TPOT table recorded per
   leg (the economics curve's recipe, one fresh process per leg). The
   leg's replay-vs-eager token identity is the KV/token-exactness gate.

## Gates

1. Every test above red-before on the pre-row tree, green after.
2. The device suite at the standing bar (94/95 + the OWED RAC flake
   recorded, not silently absorbed; this row must not depend on the
   flaky ordering).
3. The 27B c2 leg `BENCH_EXIT=0` with the TPOT table and the boundary
   count at or under the c2c precedent (six re-captures per ~12 min).
4. Served == eager token-exactness on the anchor arm, both INT8DOT
   settings.
5. Standard gates: `agent-preflight.sh --staged`, `check-commit-style`,
   `check-commit-trailers`, `check-agent-record` on the changed files.

## Evidence plan

Dated `docs/bench-evidence/tt-region-capture-<date>.md`: the per-region
`LastTraceBytes()` census, the boundary counts, the TPOT table over the
same legs as `tt-capture-economics-20260927.md` (4/16/64, one fresh
process per leg, the recorded anchor recipe), the fit predicate's
decisions per model, and the red/green test logs. The bench-evidence
file is this row's single measurement record; no number lives in two
files.

## Stop conditions

- A per-layer region cannot be captured under 50 MiB (the layer's
  command count is structurally over budget) AND a block-of-layers
  region cannot hold the boundary discipline → stop, record the census,
  the arm stays masked on the tt-metal lane (per-command trace cost is
  not shrinkable from vllm.cpp — escalate with the numbers beside
  tt-metal#57970).
- A cross-region state slot cannot bind persistently in-region → stop,
  record; the 9B precedent says this should not happen, but a
  design-level impossibility stops the row, not a workaround.
- Region-scoped replay serves tokens that diverge coherently from eager
  on the anchor → stop, the adjudication row decides which side matches
  the vLLM oracle.

## Owed

- The RAC doctest flake from `e39f2cf3f` (order-sensitive under
  full-suite program-cache history; 126/128 K elems, user-1 second
  head): NOT this row's fix, but this row's suite must stay green with
  it recorded; its bisect (which case poisons the variant) stays a
  separate unit.
- INT8DOT re-measure downstream: the INT8DOT lever's throughput verdict
  is re-measured on the served region-scoped arm once it lands; this
  row records only correctness on both INT8DOT settings.
- The sampler-bracket interaction: the dispatch-cost row
  (`.agents/specs/tenstorrent-dispatch-cost.md`,
  `row/TT-DISPATCH-COST`) brackets inside the ~31.1 s caller window
  assuming a single whole-graph trace execution; region scope splits
  that window into 64 per-region executions. Its baseline
  (31,100-31,172 ms, flat) must be re-recorded on the region-scoped arm
  before its lever verdict binds; sequence the bracketing measurement
  BEFORE this row lands, or re-baseline it after — one of the two,
  recorded in both specs.

## Git integration

One pull request: spec, implementation, tests, evidence, and this row's
records land together (developer decision, 2026-09-28, recorded in
`.agents/developer-preferences.md`). Branch `row/tt-27b-region-capture`.

## Now

2026-09-28: DRAFT. The trace-budget analysis (issue Blocker B) named
region-scoped capture as the recommendation; this spec commits the
design before any implementation. Next: the red-first region handoff
tests, then the region capture machinery on the existing
`tenstorrent_capture.cpp` seams.
