ID: ISSUE-GH-2009
Title: **DFlash2's draft-context position invariant is ungated: deleting it leaves the suite green.** Found while fixing [#2008](https://github.com/mudler/vllm.cpp/issues/2008) and owed by [`specs/dflash2-request-scoped-context.md`](../../specs/dflash2-request-scoped-context.md). `src/vllm/v1/worker/gpu/runner.cpp:2939-2945` is the guard the whole draft-context accumulation rests on and the reason #2008 was a loud refusal rather than a silent wrong-context draft. Measured: deleted on the pre-#2008 code, `test_dflash2_concurrency` stays green at 2 cases / 30 assertions, the row move resets the survivor's store to empty, it drafts from a context that is not its own, and nothing notices — because the verify is lossless, so a draft from the wrong context costs acceptance and never a token, and every token-shaped gate in this tree is blind to it by construction. #2008's own gate cannot close this: the natural leg, comparing drafted blocks against a solo control, is a **tautology** on the shared DFlash2 fixture — with the invariant deleted and the context reset at every row move the draft still emits `12 12 12` at every step of both runs, because its seeded-noise weights over a 24-token vocabulary collapse the selector walk to one id, so nine passing string comparisons measured nothing. That leg was written, run and removed rather than shipped. `test_dflash2_runner_reach`'s value-sensitivity case is unaffected — it moves the drafts by changing the selector's WEIGHTS, not the context. Closing this needs a fixture whose drafted block is demonstrably sensitive to the draft CONTEXT, which is a fixture problem before it is a test problem and is the same instrument several DFlash2 rows would benefit from: today the tree can prove a draft moves with its weights and cannot prove it moves with its context
Row: -
State: UNKNOWN
Kind: bug
GitHub: 2009
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:753`

### Frozen archive evidence

> | [#2009](https://github.com/mudler/vllm.cpp/issues/2009) | — | **DFlash2's draft-context position invariant is ungated: deleting it leaves the suite green.** Found while fixing [#2008](https://github.com/mudler/vllm.cpp/issues/2008) and owed by [`specs/dflash2-request-scoped-context.md`](../../specs/dflash2-request-scoped-context.md). `src/vllm/v1/worker/gpu/runner.cpp:2939-2945` is the guard the whole draft-context accumulation rests on and the reason #2008 was a loud refusal rather than a silent wrong-context draft. Measured: deleted on the pre-#2008 code, `test_dflash2_concurrency` stays green at 2 cases / 30 assertions, the row move resets the survivor's store to empty, it drafts from a context that is not its own, and nothing notices — because the verify is lossless, so a draft from the wrong context costs acceptance and never a token, and every token-shaped gate in this tree is blind to it by construction. #2008's own gate cannot close this: the natural leg, comparing drafted blocks against a solo control, is a **tautology** on the shared DFlash2 fixture — with the invariant deleted and the context reset at every row move the draft still emits `12 12 12` at every step of both runs, because its seeded-noise weights over a 24-token vocabulary collapse the selector walk to one id, so nine passing string comparisons measured nothing. That leg was written, run and removed rather than shipped. `test_dflash2_runner_reach`'s value-sensitivity case is unaffected — it moves the drafts by changing the selector's WEIGHTS, not the context. Closing this needs a fixture whose drafted block is demonstrably sensitive to the draft CONTEXT, which is a fixture problem before it is a test problem and is the same instrument several DFlash2 rows would benefit from: today the tree can prove a draft moves with its weights and cannot prove it moves with its context | bug |

## Resolution

-
