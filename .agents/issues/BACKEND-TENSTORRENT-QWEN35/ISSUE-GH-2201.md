ID: ISSUE-GH-2201
Title: BACKEND-TENSTORRENT-QWEN35 W3: the GDN d2h counter misses two download paths, and EnsureGdnCacheDevice's fast path accepts a conv-transposed host pointer
Row: BACKEND-TENSTORRENT-QWEN35
State: CLOSED
Kind: bug
GitHub: 2201
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-28
Updated: 2026-08-29
Closed: 2026-08-29

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> The GDN row's reviewer left two verification debts, recorded as W3 in [`.agents/specs/tenstorrent-qwen35.md`](../../specs/tenstorrent-qwen35.md) (Work breakdown item 5, "Reviewer leftovers from the GDN row"):
>
> **(a) The d2h traffic counter is incomplete.** `GdnStateD2hBytes()` (`src/vt/tenstorrent/tenstorrent_ops.cpp:4164`, surfaced as `state_d2h_bytes` in `tenstorrent_device.h:244`, snapshotted at `:6223`) counts the downloads at `:5039` and `:5109` but misses:
> 1. the `EnsureGdnCacheDevice` slow-path download (`tenstorrent_ops.cpp:4216`) — a host-refresh upload that also reads device state back;
> 2. the `CommitConvTransposed` untracked-buffer fallback (`tenstorrent_ops.cpp:4563`, commit sites `:4774`) — when the buffer is not the tracked one, the path still moves bytes to the host and records nothing.
> So any d2h attribution or budget read off `state_d2h_bytes` is a lower bound by an unmeasured amount, and the two W2b-era legs that assert on the counter cannot see these paths at all.
>
> **(b) A role-confusion hardening, not a live bug.** `EnsureGdnCacheDevice`'s fast path keys on the host pointer alone and does not check the `conv_transposed` role, so a host pointer reused across roles (GDN state vs conv-transposed state) would be served the cached device tensor with the WRONG geometry. `src/vllm/model_executor/models/qwen3_5.cpp` uses distinct buffers per role today, so nothing live hits it — the fix is a refusal (or key widening) plus a test that pins the refusal.
>
> FIX (test-first per the row's method): write the smallest failing test per leftover in the TT suite (`tests/vt/test_tenstorrent_backend.cpp`), capture red, then (a) add the two missing `fetch_add`s so the counter is complete, and (b) make the fast path reject (or correctly key) a host pointer presented under a different role, with the refusal naming the mismatch. Numerics must not move: the counter and the refusal are observability/hardening, and the sacred 16/16 golden legs must stay byte-identical.
>
> Owner: row `BACKEND-TENSTORRENT-QWEN35`, spec `tenstorrent-qwen35.md`.

## Resolution

GitHub records closing pull request #2217 (https://github.com/mudler/vllm.cpp/pull/2217) merged on 2026-08-29 as commit `a456e6eaf98a43e04a328179baa9bb058c39c824`. GitHub closed issue #2201 on 2026-08-29.
