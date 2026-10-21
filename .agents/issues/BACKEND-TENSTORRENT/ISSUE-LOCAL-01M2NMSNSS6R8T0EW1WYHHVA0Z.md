ID: ISSUE-LOCAL-01M2NMSNSS6R8T0EW1WYHHVA0Z
Title: APEX 27B e2e: greedy token-correctness gate vs the llama.cpp oracle
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-16
Updated: 2026-10-21
Closed: -

## Problem

The APEX-I-Nano 27B now completes the e2e bench (PR #3212: exit 0, 64/64 tokens) but nothing pins that the generated tokens are CORRECT. Before any bench record or the APEX e2e owed closes, the greedy decode needs a gate against the pinned llama.cpp b10451 oracle (the registry's k-quant floor oracle) on the identical workload: same GGUF artifact, same generated-dataset prompts (BuildPrompt, seeds 0 and 1, input_len 128), greedy temp 0, 32 output tokens, no chat template. Deliverable: recorded token/text comparison, disposition, and the recipe.

## Resolution

Evidence 2026-09-16: gate EXECUTED, verdict FAIL (the defect it exists to
catch is real; fix owned by ISSUE-LOCAL-01M2NSDATJQ1YNW1PA9ZBMAAM5).

- Workload: identical both legs — artifact sha256
  `47b627b7de17c2bcfa9cebb7cf2d9d81e68b4b61f8cad16cf85f839604c694cb`,
  BuildPrompt seeds 0/1 (351-byte filler prompts, byte-identical files
  /tmp/apex_prompt{0,1}.txt), greedy temp 0, 32 tokens, no chat template.
- Oracle leg: pinned llama.cpp b10451 `llama-server` /completion with
  `return_tokens` (/tmp/oracle_tokens.json) — real token ids, both prompts.
- Our leg: `vllm-bench --output-token-ids` (/tmp/apex_ids.json) — **all
  zeros**, 2x32. `VT_DEBUG_SAMPLED=1` confirms the sampler itself emits
  tok=0 at every step (also on a 1x16 / 4-token run, /tmp/apex_debug2.log).
- Attribution: the logits slot IS committed `tracked=1` at `[1,248320]`
  and `EnsureHost` downloads from it (TT-SLOT lines 3020/4481/5928/7375),
  so the zeros are IN the committed device tensor: the E=1 chunked
  keep-quant arm produces garbage at the lm_head decode shapes. The 27B
  e2e completions were structurally complete but token content garbage —
  the gate did its job before any bench record.

2026-10-21: STILL TRUE. The gate this issue asks for ran, and its verdict was
FAIL. The record already carries that execution: identical workload on both
legs, artifact sha256 `47b627b7de17c2bcfa9cebb7cf2d9d81e68b4b61f8cad16cf85f839604c694cb`,
the oracle leg on the pinned llama.cpp b10451 `llama-server`, our leg through
`vllm-bench --output-token-ids`, greedy temp 0, 32 tokens, no chat template --
and all 64 of our token ids were zero.

The defect the gate caught is closed (`ISSUE-LOCAL-01M2NSDATJQ1YNW1PA9ZBMAAM5`,
CLOSED 2026-09-18), but that record's own resolution states that token-exact
parity with the llama.cpp oracle "is NOT yet achieved and stays open under the
gate spec", and the gate spec `.agents/specs/tenstorrent-apex-greedy-gate.md`
is still ACTIVE with an open `## Owed` section.

What settles it: a P150 rerun of the same two legs on the current pin, with the
recorded comparison and the disposition written back here. No offline check can
establish the token parity, so the record stays open.
