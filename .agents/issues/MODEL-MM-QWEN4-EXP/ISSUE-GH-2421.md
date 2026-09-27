ID: ISSUE-GH-2421
Title: qwen4_exp QSA block refuses a device-resident block table, rope pair and kv_lens, so no CUDA step can enter it
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: UNKNOWN
GitHub: 2421
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
> Spec: [`.agents/specs/qwen4-exp-qsa-device-residency.md`](../../specs/qwen4-exp-qsa-device-residency.md)
>
> ## What this tracks
>
> `src/vllm/model_executor/models/qwen4_exp_qsa_block.cpp` resolves three values on
> the host and REFUSES BY NAME when the tensor carrying them is not CPU-resident:
>
> | site | value | refusal |
> |---|---|---|
> | `IndexerRows` | group 2's block table | "the indexer block table is read on the HOST to resolve a physical row, so it must be CPU-resident" |
> | `CheckRopeLayoutsAgree` | the two rope layouts' probe rows | "the two rope layouts are cross-checked on the host, so both must be CPU-resident" |
> | `Qwen4ExpQsaIndex` | `kv_lens`, to build the scoring window | "kv_lens is read on the host to build the scoring window" |
>
> Every operand the production forward hands the block is a
> `dense_attn::DBuf`, whose constructor ends
> `t_ = MakeTensor(p_, dt, d.q.device, shape)`. On a CUDA queue that carries the
> CUDA device, so the first two refuse on every CUDA step. The spec's `## Owed`
> already names both under "the QSA CUDA arm must give the translation a
> device-side home or argue it away".
>
> ## What this change does
>
> Makes the three reads work on a device-resident operand by copying the words
> they read to the host explicitly, rather than refusing or dereferencing a device
> pointer. Each is O(pages) or O(T) i32/f32 words, and none has a device-side
> consumer to hand the work to: the page table picks an index, the probe rows feed
> a host comparison, and the window bounds are built once and uploaded.
>
> ## What stays owed after it
>
> The copy costs a queue synchronize per QSA layer per step on a device arm. That
> is a real decode cost on a 48-layer model with 12 QSA layers and it is recorded
> rather than hidden; folding the page translation into `vt::Qwen4ExpQsaCompress`'s
> own address mode removes it, which is the entry the spec already carries.
>

## Resolution

-
