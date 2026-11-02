# Independent legacy metadata repair review

## Findings

NONE.

## Verdict

PASS for the scoped metadata regression repair at `b0807acb8fbff3efbb026e1e346782fa934c9007`.

## Static review

Read the committed spec, reviewer contract, complete three-commit repair diff, actual production descriptor, shared validation predicates, selector, and surrounding host test suite. Spec commit `be9784105` precedes test commit `1bc166214`. The final commit adds evidence only. Production source, headers, defaults, and newer GPU dispatch are unchanged from `4ade7c6a1`.

The three added cases enter the registered production selector. They pin valid and invalid head geometry, the inherited block-multiple rule and zero-as-unspecified behavior, explicit auto and BF16 KV acceptance, quantized and FP16 KV refusal, decoder-only attention, inherited unsupported request modes, and cache allocation geometry. Original legacy capability and newer FLASH_ATTN controls remain.

## Full gate

In independent scratch worktree `/tmp/lumen-metadata-review`, pinned to the reviewed immutable head, compiled the actual platform and descriptor plus the complete host selector test translation unit. Linked those objects against one frozen production archive after removing its duplicate platform and descriptor objects. This retains actual production registry, validation, priority, and other backend registrars.

The unchanged complete host suite ran exactly once before mutation: exit 0, 25 cases and 148 assertions. Log: `/tmp/independent-metadata-review/immutable-full-gate-run.log`.

`python3 scripts/check-agent-record.py`, `git diff --check`, and final `git status --short` also pass. The scratch tracked tree remains byte-for-byte unchanged.

## Independent mutation evidence

Only external scratch descriptor source/object and scratch executable changed. No reviewed source, branch, index, or original Release build changed. Each mutated descriptor compiles and links successfully. Each focused run exits 1 for the intended failed assertion:

- Head predicate widened to all positive values: invalid geometry no longer throws.
- Kernel block multiple changed from 16 to 1: block sizes 8 and 24 no longer throw.
- KV connector predicate changed to true: the unsupported connector request no longer throws.
- Explicit `bfloat16` removed from supported KV types: the valid BF16 requests fail selection.
- Cache block divisibility exception disabled: invalid block size 24 no longer throws.

Logs: `/tmp/independent-metadata-review/{head,block,connector,bf16,cacheblock}-run.log`. Exact independent recipe: `/tmp/independent-metadata-review.py`. Driver log: `/tmp/independent-metadata-review-driver.log`.

Restored immutable production descriptor, recompiled and linked, and reran the complete suite: exit 0, 25 cases and 148 assertions. Log: `/tmp/independent-metadata-review/restored-run.log`. No mutation uses a compiler failure as evidence.

## Remaining concern

This scoped review closes the five reported metadata coverage gaps. It does not replace the overall CUDA full Release gate, operator rerun, actual checkpoint correctness or performance measurements, or unavailable P100 and Volta hardware validation. Earlier independent reviews cover unchanged product behavior and other metadata mutations. No full-row acceptance or green repository preflight is claimed.
