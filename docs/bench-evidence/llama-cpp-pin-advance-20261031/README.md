# Evidence: `LLAMA-CPP-PIN-ADVANCE` — stock llama.cpp pin advance to `b11765`

Date: 2026-10-31. Host: local aarch64 workstation (128 CPUs, 4 NUMA nodes,
node1 = CPUs 32-63 is the bench zone). Governor `performance`, logged, not
changed. Device-free row. Oracle records touched:
[`llama-cpp`](../../../.agents/oracles/llama-cpp.md) (pin advanced) and
[`llama-cpp-kolibri1`](../../../.agents/oracles/llama-cpp-kolibri1.md)
(re-anchoring note only). `llama-cpp-flint` untouched.

## Contents

| file | what it is |
|---|---|
| [tag-identification.md](tag-identification.md) | why `b11765` (`9df0b73c707900b7277a46b5ae52ff7e7e567d13`) is the first release tag carrying #30673 and #30865 |
| [configure.log](configure.log), [build.log](build.log) | the gateability build |
| [build-sha256.txt](build-sha256.txt) | binary sha256 record |
| [golden-kolibri1-imicro.log](golden-kolibri1-imicro.log) | greedy generate on the kolibri1 APEX I-Micro |
| [golden-qwen38-q4km.log](golden-qwen38-q4km.log) | greedy generate on the Qwen3.8-27B Q4_K_M floor artifact |
| [tg-legs.log](tg-legs.log) | first idle-gated `llama-bench` legs (3 reps; t16 leg empty — see below) |
| [tg-legs-kolibri-rerun.log](tg-legs-kolibri-rerun.log) | the kolibri1 legs re-taken at 5 reps: **17.98 ± 0.12 (16t) / 20.18 ± 0.61 (32t) tok/s** — the recorded numbers |
| [golden-flint-refusal.log](golden-flint-refusal.log) | the flint artifact's expected refusal at the stock pin |
| [reconciliation.md](reconciliation.md) | every consumer of the prior pin, dispositioned |

## Build

- Source: shallow clone of `ggml-org/llama.cpp`, tag `b11765`, HEAD
  `9df0b73c707900b7277a46b5ae52ff7e7e567d13`, `git status --porcelain` EMPTY
  at build time (tree assertion per the oracle record).
- Recipe: `cmake -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON`,
  Ninja, `taskset -c 64-127 -j 64` (build zone = node3/node2, outside the bench
  zone). 336 targets, zero warnings fatal.
- Binary identity: `llama-cli --version` → `version: 0.7.0-dev (build 1,
  commit 9df0b73)`, `built with GNU 16.2.0 for Linux aarch64`.

## Gateability legs

`gateable = yes` carries its usual meaning: the pin BUILDS and RUNS the floor
models. It says nothing about any number beyond what a leg explicitly records.

1. **Kolibri-1 APEX I-Micro** (`Kolibri-1-APEX-I-Micro.gguf`, sha256
   `3edfaf09aac60d60cde56744fe1e0a7e5362c5549d5eb12d97ec4a9c83c90e0e` — the
   recorded artifact): loads (`IQ1_M - 1.75 bpw`), generates coherent greedy
   text end to end (golden log). This is the artifact the row exists for.
2. **Qwen3.8-27B Q4_K_M** (`/mnt/models/unsloth-qwen3.8-27B-gguf/
   Qwen3.8-27B-Q4_K_M.gguf`, sha256 `7e78da5d...`): loads and generates
   coherently — the qwen38 ladder artifact class, proving no regression in the
   pre-existing floor the pin supplies.
3. **flint Q4_K_M REFUSES at the stock pin** (`unknown model architecture:
   'flint'`): expected and correct — the flint architecture is not upstream;
   that artifact belongs to the separate [`llama-cpp-flint`](../../../.agents/oracles/llama-cpp-flint.md)
   PR-head oracle, which stays pinned as is. Recorded per the row's leg list.

## The quant-matched kolibri1 comparison (the row's payoff number)

Same artifact (`Kolibri-1-APEX-I-Micro.gguf`), same bench zone (`taskset
-c 32-63`), same thread counts as our GGUF arm's recorded 16/32-thread figures
(4.97 / 6.55 tok/s). Idle gate: no foreign process pinned in 32-63 and 1-min
load < 16 before each leg; load recorded before and after each leg.

<!-- COMPARISON_TABLE -->

Same artifact, same bench zone (`taskset -c 32-63`, NUMA node1), same thread
counts, idle-gated, governor `performance` (logged, unchanged), `llama-bench
-p 0 -n 128 -r 5`:

| arm | 16 threads | 32 threads |
|---|---|---|
| llama.cpp release `b11765` (`9df0b73c`), tg128, 5 reps | **17.98 ± 0.12 tok/s** | **20.18 ± 0.61 tok/s** |
| our kolibri1 GGUF arm (recorded, `QUANT-KOLIBRI1-GGUF-ARM`, 2026-10-30) | 4.97 tok/s | 6.55 tok/s |
| llama.cpp / ours | **3.62x** | **3.08x** |

This is the kolibri1 GGUF arm's first quantization-matched external
denominator: same `Kolibri-1-APEX-I-Micro.gguf` file
(sha256 `3edfaf09...`), same hardware, same thread count. The floor is
**~3x our arm**, so the gap is real and now named and measured rather than
projected — this is what the KOLIBRI1-GGUF-ARM row asked this row to make
possible. Rep spread is small (±0.7% / ±3%); box 1-min load was 13-15 before
and 17-24 after the legs (other sessions' builds outside the bench zone, whose
cores are NUMA-local to the workload). Earlier same-day legs at 3 reps
([tg-legs.log](tg-legs.log)) agree: 18.41 ± 3.48 at 32 threads.

Correctness context carried from the scoped record: the kolibri1 router
arithmetic changed between our committed topk-row capture (`62318cd57`) and
this pin (#30865 removed the forced-F32 router precision), so no NEW
token-agreement comparison against those rows was taken here; the re-capture
at `b11765` remains owed by
`ISSUE-LOCAL-01M5H2VRQ270C05ZHSA6V8EQDA`. This row's comparison is
throughput-only on the APEX artifact and does not touch those rows.

## Stop conditions

Neither fired: the tag builds and runs on aarch64, and no reconciled
consumer's floor moved (no recorded llama.cpp floor number was measured
against the retired live pin — see [reconciliation.md](reconciliation.md)).
