# Reconciliation inventory: every consumer of the prior pin

Prior live pin: `62318cd57dc63900ecb153272c0b10f78550855d` (labelled "main",
actually the upstream release tag `b11709`), pinned 2026-10-20. Earlier stock
pin `b10451`, retired 2026-10-20. New pin: `9df0b73c707900b7277a46b5ae52ff7e7e567d13`
(release `b11765`), 2026-10-31. Identified per
[tag-identification.md](tag-identification.md).

Read 2026-10-31 via `grep -rn 'b10451'` and `grep -rn '62318cd57'` over the
tree. Each consumer is either RE-MEASURED on the new pin in this row, or
RE-ANCHORED with a dated note. Nothing was left silently behind.

## The pin surfaces (edited in this row)

| consumer | disposition | change |
|---|---|---|
| `.agents/oracles/llama-cpp.md` oracle-pin block | advanced | pin `62318cd57` → `9df0b73c` (`b11765`), `pinned_on = 2026-10-31`, `evidence = docs/bench-evidence/llama-cpp-pin-advance-20261031`; new Prior-pin paragraph for `62318cd57`/`b11709` |
| `.agents/oracles/llama-cpp-kolibri1.md` | re-anchored | already pinned `b11765` since 2026-10-22; dated note records that the stock pin advanced onto the same object and that its "Why the llama-cpp file cannot carry this pin" paragraph is now historical |
| `docs/FEATURES.md` (Reference versions) | re-stated | `b10451` → `b11765` (the surface had drifted two pins behind; the 2026-10-20 advance left it stale) |
| `docs/benchmarks/how-we-measure.md` | re-stated | was "stock `b10451` since 2026-08-16, gateable = no" (three pins stale); now names `b11765`/`9df0b73c`, `gateable = yes`, with the pin history |
| `README.md` (denominators note) | re-stated | "pin is now stock `b10451`" → `b11765` since 2026-10-31, with the intermediate pins named |
| `.agents/environment.md` | re-stated | same sentence pattern updated; superseded-fork warning kept |
| `docs/USAGE.md` (cohere2moe refused arm) | re-stated | pin named is now `b11765`; `b10451` kept as the architecture's first definer |

## Attribution-only consumers (kept, no rewrite)

Per AGENTS.md and the oracle record itself, gates recorded against a prior pin
stay attributed to it. These name `b10451`/`62318cd57` as the revision a
RECORDED number or golden was taken at, and rewriting them would falsify
provenance: `.agents/benchmark-record.md` (all hits), `.agents/backend-matrix.md`,
`.agents/engine-matrix.md`, `.agents/feature-matrix.md`, `.agents/model-matrix.md`,
`.agents/quantization-matrix.md`, `docs/benchmarks/open-gaps.md`,
`docs/benchmarks/reproduce.md`, `docs/USAGE.md` (muse-glimmer and k-quant token
gate attributions), `tests/scripts/fixtures/glm5_next_kquant_golden_b10451.json`,
`tests/support/glm5_next_gguf_fixture.h`, and every file under
`docs/bench-evidence/` (evidence is immutable), `.agents/issues/`,
`.agents/claims/`, `.agents/completed/`, and the landed specs.

## Floor consumers and their dispositions

| consumer | reads which pin | disposition | delta |
|---|---|---|---|
| `tests/scripts/test_cpu_x86_llamacpp_floor.py` + `scripts/cpu-x86-llamacpp-floor.sh` | none mechanically — the harness takes a prebuilt `llama-bench` path (`LB=`) and the recorded evidence names its revision | RE-ANCHORED with dated note: the harness ran on an x86_64 host against a `b9892`-era binary (see its evidence file); this row's host is aarch64, so the x86 series cannot be re-measured here. The x86 floor stays attributed to its recorded binary; a re-take on the new pin remains owed by #1003 | none taken (wrong architecture on this host) |
| qwen38 ladder (`tests/scripts/test_qwen38_endpoint_bench.py`, tensorfold gap scripts) | none — no llama.cpp pin coupling found by grep | NO-OP: these read our own bench endpoints, not the llama-cpp oracle | n/a |
| qwen4exp ladder (`tests/scripts/test_qwen4exp_llamacpp_ladder.py`) | its OWN scoped oracle `llama-cpp-qwen4exp.md`, asserted by `test_build_script_pin_is_the_oracle_records_pin` | NO-OP: the qwen4exp/glm5next/dsv4vision/unsloth scoped oracle records pin their own objects and are untouched by this advance | n/a |
| `scripts/llamacpp/llamacpp_oracle.cpp`, `scripts/qwen3-neartie-gap-llamacpp-oracle.py`, `scripts/check-gate-commands.py` | grep hits are coincidental substrings, not pin references (verified by reading the hits) | NO-OP | n/a |
| kolibri1 topk rows (`tests/vllm/models/kolibri1_llamacpp_topk_rows.json`, provenance `62318cd57`) | recorded capture | RE-ANCHORED (pre-existing): the scoped record already names the `b11765` re-capture owed by `ISSUE-LOCAL-01M5H2VRQ270C05ZHSA6V8EQDA`; this row produces no NEW comparison against those rows, so the owe stands | none taken |
| `.agents/oracles/llama-cpp-flint.md` | separate PR-head pin `1b44e3077` | UNTOUCHED by design (spec §What this row is NOT) | n/a |

## Floors that moved

None. No recorded llama.cpp floor number in this tree was measured against the
retired live pin `62318cd57` — the cross-check evidence file pins correctness
only and says so, and every throughput floor predates even `b10451` (the #1003
re-take debt). The new numbers this row measures (below) are NEW gateability
and comparison data, not re-baselines of any landed floor, so no landed claim
is invalidated and the spec's second stop condition does not fire.

## New measurements on the new pin

See [tg-legs.log](tg-legs.log) and [README.md](README.md) in this directory.
