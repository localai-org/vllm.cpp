# Oracle: llama.cpp release `b11765`, the Kolibri-1 architecture with its router fix

A scoped, second llama.cpp record. It exists for one reason: `b11765` is the
first stock llama.cpp release that carries the `kolibri1` architecture together
with the router change that follows it, and the row
`MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm` needs a llama.cpp denominator and
a third independent reference on the artifact users download. It does not
replace the [`llama-cpp`](llama-cpp.md) oracle, it does not outrank vLLM, and it
is never a mirror source.

## Why the `llama-cpp` file cannot carry this pin

`scripts/check-oracle-pins.py` admits exactly one ` ```oracle-pin ` block per
file, so one file holds one revision. The `llama-cpp` pin is deliberately a
STOCK RELEASE, and every floor already measured against it means "what a user
gets from that release". It currently reads `62318cd57` (release `b11709`),
the first stock release that converts, loads and runs a Kolibri-1 checkpoint.
Advancing it to `b11765` would silently redefine what those recorded numbers
measured, 56 commits after the fact.

The two records therefore say two different true things.
[`llama-cpp`](llama-cpp.md) says what a release does for every model. This
file says what one release does for one architecture, and nothing else.

## Scope, and what this oracle may not do

Use it ONLY for the `kolibri1` architecture: the GGUF conversion in
`conversion/kolibri1.py`, the graph in `src/models/kolibri1.cpp`, the
architecture and hyper-parameter registration in `src/llama-arch.cpp` and
`src/llama-model.cpp`, the MoE router with its sigmoid gating, the shared
expert, the hybrid sliding and full attention, and the CPU and GGUF k-quant
speed and memory numbers those produce on a Kolibri-1 checkpoint.

Upstream describes that architecture in the merge commit that introduced it,
and this record takes the description from there rather than from a second
source: 50 layers, 384 routed experts and one shared expert, GQA with 48 query
heads against 4 KV heads at head_dim 128, a QK-RMSNorm and four RMSNorms per
block, attention interleaving four sliding layers at a 513 window with one full
layer, NeoX RoPE on the sliding layers and NoPE on the full ones. Routing picks
the experts from the logits plus the expert bias and weights them with the
sigmoid of the unbiased logits, without renorm. The converter reuses
`AfmoeModel` for the per-expert checkpoint layout and the expert selection
bias, and the released FP8 checkpoint goes through the generic
`weight_scale_inv` path.

For every other path, including CPU and GGUF k-quant floors generally, the
oracle is [`llama-cpp`](llama-cpp.md) at its own pin.

The model author's own runtime is the algorithm oracle for every behavior both
implement. The `aleph-alpha-inference` plugin registers `Kolibri1ForCausalLM`,
and upstream vLLM implements nothing for this architecture, so where the
plugin serves, the answer it produces is vLLM's answer for Kolibri. See
[`aleph-alpha-inference`](aleph-alpha-inference.md). Where vLLM implements the
behavior at all, vLLM wins. llama.cpp's structure is not vLLM's, and a
difference in structure is never a reason to diverge.

## The pin, verified rather than relayed

Read on 22 October 2026 from `ggml-org/llama.cpp` through the GitHub API, and
from the read-only mirror named by `LLAMACPP_SOURCE` in `.env`. No working tree
was modified, and no number in this file comes from a local build.

| Claim | Query | Result |
|---|---|---|
| the architecture merge | `gh api repos/ggml-org/llama.cpp/commits/62318cd57dc63900ecb153272c0b10f78550855d` | `model : add Kolibri-1 (Aleph-Alpha) support (#30673)`, committed 17 October 2026 at 11:00:17Z |
| the same object in the local mirror | `git show --no-patch 62318cd57` in the `LLAMACPP_SOURCE` mirror | the same subject, and the mirror's current branch has it as an ancestor |
| the router change | `gh api repos/ggml-org/llama.cpp/commits/9df0b73c707900b7277a46b5ae52ff7e7e567d13` | `model : drop the redundant precision overrides in the Kolibri-1 router (#30865)`, committed 20 October 2026 at 12:58:09Z |
| the router diff touches one file | the same call, `files[]` | `src/models/kolibri1.cpp`, 1 insertion and 4 deletions |
| the router arithmetic it removes | the same call, `files[0].patch` | `ggml_prec_set_acc(logits, GGML_PREC_F32)` and `ggml_prec_set_src(logits, GGML_PREC_F32, 1)` leave the router matmul |
| the architecture release | `gh api repos/ggml-org/llama.cpp/git/ref/tags/b11709` and `compare/b11709...62318cd57` | the tag resolves to `62318cd57dc63900ecb153272c0b10f78550855d`, and the compare reports `identical` |
| the router release | `gh api repos/ggml-org/llama.cpp/git/ref/tags/b11765` and `compare/b11765...9df0b73c707900b7277a46b5ae52ff7e7e567d13` | the tag resolves to `9df0b73c707900b7277a46b5ae52ff7e7e567d13`, and the compare reports `identical` |
| the distance between the two releases | `compare/b11709...b11765` | `ahead`, `ahead_by=56` |

## Gateability

`gateable = yes`. The flag says one thing: llama.cpp BUILT and RAN the
Kolibri-1 model and produced tokens. The evidence is
[the Kolibri-1 llama.cpp cross-check](../../docs/bench-evidence/kolibri1-llamacpp-crosscheck-20261020.md),
and the sentence in it that carries the flag is its own:

> The runtime LOADED the artifact and generated coherent text end-to-end on all
> eight prompts: this oracle pin is gateable for kolibri-1.

That run was measured at `62318cd57` (release `b11709`), the release that
carries the architecture. The [`llama-cpp`](llama-cpp.md) record says the same
about its own pin history: it advanced because `11fe0215` predates the
Kolibri-1 support merge, and it names the same evidence file.

**Gateable is not measured.** The flag says the oracle runs. It says nothing
about any number. The evidence file pins correctness only, and it says so: the
speed and memory floors against this oracle remain owed by the re-take debt,
and the run produced no timing worth quoting. Do not promote a throughput, a
latency or a memory figure out of that file.

The one number this record carries is the one the evidence file states, at the
revision it states: the llama.cpp arm's top-1 agreement with the primary oracle
over the 202 aligned positions of the 8-prompt battery, 198 of 202, measured at
`62318cd57`. The evidence file marks that label provisional, because the
primary rows carried a one-position lag at the time, and the primary rows were
repaired after that run. Read it as two independent implementations agreeing,
not as a settled statistic.

**The rows in this tree were captured BEFORE the router change, and a
re-capture at this pin is owed.** The committed per-position rows,
`tests/vllm/models/kolibri1_llamacpp_topk_rows.json`, carry the provenance
"llama.cpp commit 62318cd57 (Kolibri-1 support), CPU aarch64, bf16 GGUF" and
the artifact sha256 `76ddb05e458cce1a023d42cd4f13dc8e8152fbb6501c1f680918fc7d2a0fcefd`.
Upstream #30865 changed the router's arithmetic between that capture and this
pin, because the router matmul no longer runs at a forced F32 precision. Those
rows therefore do not describe the outputs of the pinned oracle, and a NEW
llama.cpp number compared against them would compare against a different
arithmetic. The re-capture at `b11765` is owed by
[ISSUE-LOCAL-01M5H2VRQ270C05ZHSA6V8EQDA](../issues/MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm/ISSUE-LOCAL-01M5H2VRQ270C05ZHSA6V8EQDA.md),
and it must land before any NEW llama.cpp number is compared. That
re-capture covers the run at the pinned object as well, so a reader who
requires the build-and-run at the exact revision finds it owed by the same
issue.

## Pin

```oracle-pin
id = llama-cpp-kolibri1
role = secondary
upstream = https://github.com/ggml-org/llama.cpp
scope = the kolibri1 architecture only: its GGUF conversion, its graph, its arch and hyper-parameter registration, its sigmoid-gated MoE router, its shared expert and its hybrid sliding/full attention, and the CPU and GGUF k-quant floor those produce on a Kolibri-1 checkpoint
pin = 9df0b73c707900b7277a46b5ae52ff7e7e567d13
pin_label = release b11765
pinned_on = 2026-10-22
gateable = yes
evidence = docs/bench-evidence/kolibri1-llamacpp-crosscheck-20261020.md
```
