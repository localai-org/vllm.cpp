ID: ISSUE-GH-2443
Title: The qwen4_exp spec still states the CUDA gather gap that KGATHER closed, including a grep that changed answer
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: UNKNOWN
GitHub: 2443
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `MODEL-MM-QWEN4-EXP`
>
> Spec: [`.agents/specs/qwen4-exp-flash-next.md`](../../specs/qwen4-exp-flash-next.md),
> section `### The neighbouring suites, and the LOAD-TIME refusal that settles reachability`.
>
> ## The defect
>
> #2391 (`27aaf199a`) and #2396 (`f937e8063`) merged within an hour of each other.
> #2391's spec text describes the CUDA gather gap as open; #2396 closed it. Three
> statements are live on `main` and false:
>
> | the spec says | `main` actually says |
> |---|---|
> | "there is no `kEmbeddingQuant` OpId at all — `grep -c kEmbeddingQuant include/vt/ops.h` is 0" | `grep -c` returns **2** |
> | "`DeviceQuantGatherSupported` returns `dev == vt::DeviceType::kCPU` and nothing else" | `return vt::OpRegistered(vt::OpId::kEmbeddingQuant, dev);` |
> | "on a CUDA build this architecture **does not load**" | it loads; the refusal no longer fires |
>
> A fourth is misleading rather than plainly false: "`EmbeddingKernelCuda` refuses
> a table that is not f32/bf16 by name". That string is still greppable, but it no
> longer covers the block-quantized case, so citing it as the gather's blocker
> points a reader at a refusal that no longer refuses.
>
> The first one is the worst of the four, because it publishes a literal command
> and its expected output. A reader who runs it gets a different answer than the
> document promises, and nothing in the tree gates that.
>
> ## Why it survived
>
> #2396's author DID reconcile the two surfaces #2391 flagged — `docs/FEATURES.md`
> (the row now reads "ALL SIX ... NOW HAVE CUDA ARMS ... **THAT LEAVES THE GATHER,
> AND KGATHER LANDED IT**", one consistent story) and the SCOPE paragraph in
> `tests/vllm/models/test_qwen4_exp_cuda.cpp`. The spec was the third location and
> was missed. #2391's own `## Owed` entry named all three in advance, which is how
> this was found.
>
> ## Fix
>
> Correct the three statements in place rather than delete the passage: the
> measurement below them is what the device gate observed and a record that
> quietly loses a superseded reading cannot be audited. Mark the reading as
> superseded, give the current values, and say what did NOT change — the three
> host reads in `qwen4_exp_qsa_block.cpp` and `IsCudaKeepQuantSupported`'s IQ4_NL
> exclusion are both still on `main`, so this row still claims no GPU run and no
> token.
>

## Resolution

-
