ID: ISSUE-GH-772
Title: **FIXED 2026-08-14, `row/FIX-UNALIGNED-LOADERS-772`, spec [`unaligned-safetensors-loaders.md`](../../specs/unaligned-safetensors-loaders.md).** FOURTH recurrence of the unaligned-safetensors class after [#301](https://github.com/mudler/vllm.cpp/issues/301), [#627](https://github.com/mudler/vllm.cpp/issues/627) and [#674](https://github.com/mudler/vllm.cpp/issues/674): `voxtral.cpp:51`/`:344`, `qwen3_vl.cpp:78` and `qwen3_5_mtp.cpp:71` each formed a `const uint16_t*` over an mmap'd safetensors payload, whose offset (`8 + <JSON header length> + <preceding tensor sizes>`) carries no alignment guarantee. TWO OF THE FOUR ARE INVISIBLE TO UBSAN BY CONSTRUCTION — `PermuteQKBf16` and `CopyRawNK` launder every access through `std::memcpy`, so `-fsanitize=alignment` never fires, and all three earlier recurrences were UBSan finds. Fixed by `vt::LoadUnaligned` at the two scalar-load sites and by `unsigned char` byte arithmetic at the two bulk-copy sites; `minimax_h3_vae_loader.cpp:96-101`, which hand-rolled the same `memcpy` instead of calling the seam, folded in. Gated by `test_loader_unaligned_offsets`, which FORCES an odd payload offset with one space of JSON header padding and REQUIREs the mapped-address parity
Row: FIX-UNALIGNED-LOADERS-772
State: UNKNOWN
Kind: bug
GitHub: 772
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:145`

### Frozen archive evidence

> | [#772](https://github.com/mudler/vllm.cpp/issues/772) | `FIX-UNALIGNED-LOADERS-772` | **FIXED 2026-08-14, `row/FIX-UNALIGNED-LOADERS-772`, spec [`unaligned-safetensors-loaders.md`](../../specs/unaligned-safetensors-loaders.md).** FOURTH recurrence of the unaligned-safetensors class after [#301](https://github.com/mudler/vllm.cpp/issues/301), [#627](https://github.com/mudler/vllm.cpp/issues/627) and [#674](https://github.com/mudler/vllm.cpp/issues/674): `voxtral.cpp:51`/`:344`, `qwen3_vl.cpp:78` and `qwen3_5_mtp.cpp:71` each formed a `const uint16_t*` over an mmap'd safetensors payload, whose offset (`8 + <JSON header length> + <preceding tensor sizes>`) carries no alignment guarantee. TWO OF THE FOUR ARE INVISIBLE TO UBSAN BY CONSTRUCTION — `PermuteQKBf16` and `CopyRawNK` launder every access through `std::memcpy`, so `-fsanitize=alignment` never fires, and all three earlier recurrences were UBSan finds. Fixed by `vt::LoadUnaligned` at the two scalar-load sites and by `unsigned char` byte arithmetic at the two bulk-copy sites; `minimax_h3_vae_loader.cpp:96-101`, which hand-rolled the same `memcpy` instead of calling the seam, folded in. Gated by `test_loader_unaligned_offsets`, which FORCES an odd payload offset with one space of JSON header padding and REQUIREs the mapped-address parity | bug |

## Resolution

-
