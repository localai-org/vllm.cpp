ID: ISSUE-GH-1015
Title: `Ltx2WidenDitToF32` (`src/vllm/model_executor/models/ltx2_loader.cpp:694-710 @ 332aed738`) holds the **bf16 originals AND the f32 copies at once, permanently**: `:707` does `checkpoint.storage.push_back(std::move(widened))` — an APPEND — and repoints `view.data`, but nothing drops the original `storage` entry. At the **21.00B** geometry MEASURED from the checkpoint header ([`ltx-2.5.md`](../../specs/ltx-2-5.md):55; `include/vllm/multimodal/ltx2_video.h:109 @ 332aed738`, the line this row cites, states the same 21.00B) a fully-bf16 DiT is 2+4 bytes per parameter = **~117.3 GiB COMPUTED on a 119 GiB box**, of which the ~39.1 GiB of bf16 originals is dead the instant the loop finishes. **Two bounds on that figure, stated up front.** `ltx2_video.h:109` also says "~76 GB" for the f32 half in the SAME sentence, and 21.00B in f32 is ~84 GB, so the header contradicts itself — a small separate record defect, unowned. And the loop widens only views whose dtype is `kBF16` (`:697`), so on an FP8 checkpoint it touches only the bf16 biases and norms: 117.3 GiB is the **bf16-arm bound**, not a cost every host-arm load pays, and no host-arm load has been measured against it. Reached on every host-arm load via `src/vllm/multimodal/ltx2_video.cpp:786 @ 332aed738` (`widen_to_f32 = !on_device`). Found while attributing [#1014](https://github.com/mudler/vllm.cpp/issues/1014); it is NOT that fall, which follows a flat plateau this cost precedes. Fix must not free a buffer another view still points at (same class as [#949](https://github.com/mudler/vllm.cpp/issues/949)). Listed under `## Owed` in [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1015
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:291`

### Frozen archive evidence

> | [#1015](https://github.com/mudler/vllm.cpp/issues/1015) | — | `Ltx2WidenDitToF32` (`src/vllm/model_executor/models/ltx2_loader.cpp:694-710 @ 332aed738`) holds the **bf16 originals AND the f32 copies at once, permanently**: `:707` does `checkpoint.storage.push_back(std::move(widened))` — an APPEND — and repoints `view.data`, but nothing drops the original `storage` entry. At the **21.00B** geometry MEASURED from the checkpoint header ([`ltx-2.5.md`](../../specs/ltx-2-5.md):55; `include/vllm/multimodal/ltx2_video.h:109 @ 332aed738`, the line this row cites, states the same 21.00B) a fully-bf16 DiT is 2+4 bytes per parameter = **~117.3 GiB COMPUTED on a 119 GiB box**, of which the ~39.1 GiB of bf16 originals is dead the instant the loop finishes. **Two bounds on that figure, stated up front.** `ltx2_video.h:109` also says "~76 GB" for the f32 half in the SAME sentence, and 21.00B in f32 is ~84 GB, so the header contradicts itself — a small separate record defect, unowned. And the loop widens only views whose dtype is `kBF16` (`:697`), so on an FP8 checkpoint it touches only the bf16 biases and norms: 117.3 GiB is the **bf16-arm bound**, not a cost every host-arm load pays, and no host-arm load has been measured against it. Reached on every host-arm load via `src/vllm/multimodal/ltx2_video.cpp:786 @ 332aed738` (`widen_to_f32 = !on_device`). Found while attributing [#1014](https://github.com/mudler/vllm.cpp/issues/1014); it is NOT that fall, which follows a flat plateau this cost precedes. Fix must not free a buffer another view still points at (same class as [#949](https://github.com/mudler/vllm.cpp/issues/949)). Listed under `## Owed` in [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md) | bug |

## Resolution

-
