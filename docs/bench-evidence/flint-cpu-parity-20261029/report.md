# FLINT-CPU-PARITY — the llama.cpp floor on flint Q4_K_M, and what our CPU side can serve

Row: `FLINT-CPU-PARITY`
Spec: `.agents/specs/flint-cpu-parity.md`
Issue: `ISSUE-LOCAL-01M61G9R7D4EAM54RKXMES0Z6J`
Date: 2026-10-29. Host: the shared aarch64 128-core dev box, bench zone
`taskset -c 32-63`, governor **schedutil** on all 128 cores (observed; the row
brief expected `performance` — the recorded state is schedutil, left untouched).
Idle gate (1-min load < 16, no other test/bench processes) re-checked before
every measured leg. Box claimed in `~/.config/box-claims.txt`.

Artifact: `/mnt/models/flint-GGUF/flint-Q4_K_M.gguf`, sha256
`20c89707f762af73f0840f7b2c2911193617e60cac2ae30a96cb2cda7c15927e` — the SAME
byte-identical file measured on both sides.

## Deliverable 1 — the oracle gate: PASS, `gateable = yes`

llama.cpp PR #30928 head `1b44e3077` (build at `/tmp/llamacpp-flint-oracle`,
`LD_LIBRARY_PATH=/tmp/llamacpp-flint-oracle/build/bin`), CPU backend.

- tg rates, 3-rep medians, prompt 0, `llama bench -p 0 -n 200,400 -r 3`
  (`llama-bench-tg.log`):

| threads | tg200 (tok/s) | tg400 (tok/s) |
|---:|---:|---:|
| 16 | **26.29 ± 0.05** | **25.66 ± 0.08** |
| 32 | **26.38 ± 0.10** | **24.69 ± 1.75** |

- Greedy token goldens, temperature 0, seed 42, finish reason length:
  - `golden-sayhi-raw.log` — prompt "Say hi.", 24 completions: " The Australia
    is a country is located in thecur southern hemisphere. It is the world's
    sixth-largest island, but often described"
  - `golden-count-raw.log` — prompt "Count from 1 to 30 in words: one, two,
    three,", 40 completions: " four, five, fifteen, sixteen, ... twenty-seven"
  - `golden-200tok-raw.log` — 190-token computing-history prompt, 32
    completions: " The history of computing is not just a story of circuits
    and code. It is a story of how we think, how we organize knowledge, and
    how we imagine"
- Harness quirk, recorded in the pin block: the container's chat-template
  metadata carries a `\x00` escape that llama.cpp's template lexer rejects and
  template init throws (`golden-sayhi.log` shows the failure). The golden legs
  therefore pass an explicit raw-passthrough chat template
  (`--chat-template '{% for message in messages %}{{ message["content"] }}{%
  endfor %}' --jinja`), which yields a raw completion — the prompt reaches the
  model unwrapped, matching our side's raw-prompt serving. Numerics are
  unaffected; the graph is the same.

The pin `.agents/oracles/llama-cpp-flint.md` flips to `gateable = yes` with
this directory as evidence, and the AGENTS.md registry row lands in the same
change (the pre-existing draft row and the committed row were merged into ONE
row carrying both descriptions — committing both would have created a
duplicate `llama-cpp-flint` registry id).

## Deliverable 2 — the like-for-like floor

The two 16-thread and 32-thread cells above ARE the floor, pinned to the bench
zone (`taskset -c 32-63`), steady-state windows (200 and 400 completions, the
bench's reported rate excludes warmup; `-r 3` medians). **The floor number is
26.29 tok/s (tg200, 16 threads); nothing else is quotable as the floor.**

Note on the zone: the earlier external-session numbers (27.99/27.33) were
measured on CPUs 0-31; these cells are the 32-63 zone this row was scoped to.

## Deliverable 3 — our side, and the row's central finding

**vllm.cpp has NO CPU keep-quant serving arm.** `VT_FLINT_KEEP_QUANT=1` loads
the block-carrying tree (`LoadFlintKeepQuantFromGguf`), whose only consumer is
the Tenstorrent device forward (`flint_tt.cpp`); the CPU reference forward
refuses that tree by name (flint.cpp, the BF16-OWNED audit), and the **paged**
CPU arm — the actual serving path this row's harness drives — did not carry
the audit.

- `keepquant-refusal.log` (BEFORE the fix): the paged arm let the packed bytes
  reach a bf16 matmul and died on the opaque `vt: matmul: inner dims mismatch
  at src/vt/ops.cpp:150`. A safe throw, not a silent misread, but AGENTS.md
  requires an unimplemented arm be refused with a message that names the
  missing part. Fixed in this row: the same named audit now sits at the head
  of `FlintForwardPaged`, and `keepquant-refusal.log` (AFTER) records the
  refusal firing from the serving path with the full reload instruction.

So the keepquant serving number this row was scoped to produce **cannot be
produced on CPU** — the arm refuses by design. Measured instead, clearly
labeled: our only CPU GGUF-serving arm is the bf16-owned paged arm (dequantize
to bf16 elementwise weights), same harness (`build/examples/vllm-bench`, full
`LoadedEngine` stack), same zone, same budgets, single stream, greedy,
`--ignore-eos`, windows 200/400 completions (`ours-t16.log`, `ours-t32.log`):

| threads | our 200 (tok/s) | our 400 (tok/s) |
|---:|---:|---:|
| 16 | 3.25 | 3.28 |
| 32 | 2.33 (run 2; run 1, 2.45, taken with an external rust-analyzer active — both runs agree within 7%, quoted run 2) | 2.50 (run 2; run 1, 2.60) |

`ours-t32-contaminated-run1.log` keeps run 1. Leg-end 1-min load readings of
24-32 are our own 32 runnable bench threads, not contamination; the pre-leg
gates all passed at < 16.

What differs from the oracle's windows: llama.cpp `llama bench` reports the
median rate over 3 repetitions of a prompt-0 generation with its own scheduler
in-process; ours schedules through the full engine stack (input processor,
scheduler, paged attention, sampler, output processor) with the first-token
prefill inside the same run. Both quote steady-state generation windows; the
harness overhead differs, and ours includes a 8-token prefill the oracle leg
does not.

## Deliverable 4 — the shexp check and the token cross-check

- **shexp slicing: VERIFIED CORRECT.** Our loader reads
  `shared_expert_intermediate_size` from the GGUF metadata (= 1024) and loads
  `ffn_gate/up/down_shexp` as ONE rank-2 projection of that width
  (flint_gguf_weights.cpp:492, 598-602, 623-625) — one shared expert of DOUBLE
  width stored as 2 slots of 512, exactly llama.cpp's `expert_shared_count =
  2` semantics. Never two shared experts. The keepquant loader arm
  (`OwnProjectionKeepQuant`) uses the same widths.
- **Token cross-check**: the keepquant arm produces no tokens on CPU (the
  refusal above), so a keepquant-vs-golden comparison is not producible. The
  applicable correctness record for the arm that DOES serve is the row's
  committed gate: `test_flint_gguf` replays the pinned oracle golden on this
  container — 46/53 greedy positions token-exact, 0 hard flips, everything
  else inside the house 2.5-nat near-tie band (docs/USAGE.md flint section).
  No divergence beyond the band; no defect finding on numerics. The one
  defect this row found and fixed is the missing paged-arm refusal, above.

## The gap record

- Floor (llama.cpp Q4_K_M, this box, this zone): **26.29 / 25.66 tok/s**
  (tg200/tg400, 16 t); 26.38 / 24.69 (32 t).
- Ours, keepquant arm (the arm the row scoped): **NOT MEASURABLE on CPU** —
  the tree refuses by name; its consumer is the Tenstorrent device. The gap is
  therefore **unrecorded pending a CPU quantized dispatch arm**; the named
  lever is a keep-quant CPU consumer for the flint paged arm (the int8-dot /
  per-dtype dispatch the TT arm already uses), which files its OWN row per the
  spec — this row does not implement kernel work.
- Ours, bf16 paged arm, context only, NOT a keepquant comparison: 3.25 / 3.28
  (16 t), 2.33 / 2.50 (32 t). Against the quantized floor that reads as an
  ~8-10x difference, but it compares a dequantize-to-bf16 elementwise arm
  against packed k-quant GEMV — different kernel class, different memory
  traffic; it is a bound, not a gap.

## Raw logs in this directory

`llama-bench-tg.log`, `golden-sayhi.log` (the chat-template failure),
`golden-sayhi-raw.log`, `golden-count-raw.log`, `golden-200tok-raw.log`,
`keepquant-refusal.log` (post-fix named refusal),
`ours-t16.log`, `ours-t32.log`, `ours-t32-contaminated-run1.log`.
