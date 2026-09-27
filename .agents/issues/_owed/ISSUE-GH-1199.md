ID: ISSUE-GH-1199
Title: `scripts/check-symbol-anchors.py --upstream-root` resolved 354 `vllm/...::Symbol` citations against the parity pin `555967922`: **343 fresh, 0 stale**, which is the measurement that decides the symbol convention over line anchors, since the same pin advance broke every line anchor [#1139](https://github.com/mudler/vllm.cpp/issues/1139) examined. The 11 that did not resolve name a FILE absent at the pin: six `.agents/model-matrix.md` rows (`olmo.py`, `olmo2.py`, `ouro.py`, `persimmon.py`, `plamo2.py`, `fuyu.py` — `registry.py:658` routes `OlmoForCausalLM` to the `transformers` fallback and `registry.py:765` lists `FuyuForCausalLM` as removed at 0.25.0, so our rows claim a mirror source vLLM deleted), two malformed `vllm/tests/kernels/...` paths, `vllm/v1/worker/gpu/worker.py` for `gpu_worker.py`, and one fixture-text false positive. Filed rather than repaired because the model-matrix half is a claim about vLLM, not a path edit. Owned under `## Owed` in [`citation-anchor-freshness.md`](../../specs/citation-anchor-freshness.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1199
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:394`

### Frozen archive evidence

> | [#1199](https://github.com/mudler/vllm.cpp/issues/1199) | — | `scripts/check-symbol-anchors.py --upstream-root` resolved 354 `vllm/...::Symbol` citations against the parity pin `555967922`: **343 fresh, 0 stale**, which is the measurement that decides the symbol convention over line anchors, since the same pin advance broke every line anchor [#1139](https://github.com/mudler/vllm.cpp/issues/1139) examined. The 11 that did not resolve name a FILE absent at the pin: six `.agents/model-matrix.md` rows (`olmo.py`, `olmo2.py`, `ouro.py`, `persimmon.py`, `plamo2.py`, `fuyu.py` — `registry.py:658` routes `OlmoForCausalLM` to the `transformers` fallback and `registry.py:765` lists `FuyuForCausalLM` as removed at 0.25.0, so our rows claim a mirror source vLLM deleted), two malformed `vllm/tests/kernels/...` paths, `vllm/v1/worker/gpu/worker.py` for `gpu_worker.py`, and one fixture-text false positive. Filed rather than repaired because the model-matrix half is a claim about vLLM, not a path edit. Owned under `## Owed` in [`citation-anchor-freshness.md`](../../specs/citation-anchor-freshness.md) | bug |

## Resolution

-
