# CLAIM-MODEL-GLINER25-DECIDE

Separate claim from `CLAIM-MODEL-GLINER25`, not a second line on it: `check-agent-record.py` rejects a duplicate active claim ID, so one claim owns exactly one active row. The two rows share a tower and differ in a head, and the split follows that difference.

| Claim | Row IDs | Agent | Worktree / remote dir | Branch | Owned scope | State | Last update |
|---|---|---|---|---|---|---|---|
| `CLAIM-MODEL-GLINER25-DECIDE` | `MODEL-GLINER25-DECIDE` (`SPIKE`) | Claude Code (glm5.2), operator role | `.wt/gliner25` | `row/MODEL-GLINER25` | GLiNER2.5-Decide: the CLASSIFICATION-HEAD sibling of `MODEL-GLINER25`, reusing its DeBERTa v2 encoder and disentangled attention rather than reimplementing them, plus the `/v1/systemone` route and the `vllm_decide` ABI it already shares with kev and Laya. The head is Linear -> ReLU -> Linear to one output, replacing the NER boundary pooler; everything below the head is the sibling's. Bounded-choice decision, one forward, no prompt template and no generated tokens | `SPIKE` | 2026-09-28 |
