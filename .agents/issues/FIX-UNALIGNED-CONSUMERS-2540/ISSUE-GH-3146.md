ID: ISSUE-GH-3146
Title: dots3_note_vision.cpp reads a borrowed bf16 weight through a misaligned uint16_t* dereference
Row: FIX-UNALIGNED-CONSUMERS-2540
State: CLOSED
Kind: UNKNOWN
GitHub: 3146
Mirror: SYNCED
Availability: FULL
Created: 2026-09-11
Updated: 2026-09-11
Closed: 2026-09-11

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `FIX-UNALIGNED-CONSUMERS-2540`
>
> Owed by `.agents/specs/unaligned-safetensors-consumers.md` under `## Owed`.
>
> ## What was measured
>
> The `sanitize-cpu (address,undefined)` lane aborts at
> `src/vllm/model_executor/models/dots3_note_vision.cpp:367` with a misaligned
> load:
>
> ```
> runtime error: load of misaligned address 0x... for type const uint16_t,
> which requires 2 byte alignment
> ```
>
> The site casts borrowed safetensors bytes to `const uint16_t*` and dereferences
> the typed pointer directly:
>
> ```cpp
> const auto* src = reinterpret_cast<const uint16_t*>(w.bytes.data());
> const auto bf16 = [src] (int64_t idx) {
>   uint32_t bits = static_cast<uint32_t>(src[idx]) << 16;  // line 367
> ```
>
> When `BorrowStTensorBytes` hands the weight at an odd byte address, the
> `src[idx]` load is undefined behavior and the address sanitizer aborts.
>
> ## Why this site was not in #2579
>
> #2579 grepped at `aedad724c` (2026-09-02) and listed 20 sites across ten files.
> `dots3_note_vision.cpp` did not exist in the tree at that SHA — it was
> introduced by `c8dad3857` (2026-09-01) but the grep ran after a merge that did
> not carry it, so the file was absent. It matches the same
> `reinterpret_cast<const uint16_t*>(<OwnedTensor>.bytes.data())` shape but was
> not enumerated.
>
> ## Fix
>
> Replace `src[idx]` with `vt::LoadUnaligned<uint16_t>(src + idx)`, the same
> `memcpy`-based load that #3138 applied to the CPU GEMM scalar tails and that
> `FIX-UNALIGNED-CONSUMERS-2540` applied to the three consumers in its original
> scope. Add `#include "vt/unaligned.h"`. No loaded value, shape, dtype, or
> behavior changes.
>
> ## Red-before
>
> `./build-sanitize/tests/test_dots3_note_vision` (or the sanitizer lane) aborts
> at line 367 under `-fsanitize=address,undefined -fno-sanitize-recover=all`.
>
> ## Green-after
>
> The same test passes under the sanitizer with the `LoadUnaligned` replacement.

## Resolution

Fixed in PR #3148: replaced src[idx] with vt::LoadUnaligned<uint16_t>(src + idx) at dots3_note_vision.cpp:367
