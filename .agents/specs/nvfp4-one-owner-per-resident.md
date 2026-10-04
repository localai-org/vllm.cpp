# `Nvfp4Weight` publishes one device allocation on two owning handles — ISSUE-LOCAL-01M3QDT6A8JEM8WNJTPR16MXHR

`ResidentNvfp4` hands the same device block to `Nvfp4Weight::d_packed` and to
`OwnedTensor::d_dev`. Every Marlin repack builder released only the first, so the
second kept a full packed+scale device copy alive for the process lifetime — once
per repacked weight, and once per expert per projection on an MoE model. The
repack is the step that is supposed to keep peak weight memory flat, so the
surviving alias silently undoes the memory design on exactly the cards that need
it: the NVFP4 arms serve on 24 GiB consumer Blackwell (`sm_120a`), where a second
full copy of the fp4 originals does not fit beside the repacked weights.

Issue: [ISSUE-LOCAL-01M3QDT6A8JEM8WNJTPR16MXHR](../issues/LOAD-MODELOPT-NVFP4-BORROW/ISSUE-LOCAL-01M3QDT6A8JEM8WNJTPR16MXHR.md).
Owning row: `LOAD-MODELOPT-NVFP4-BORROW` ([engine-matrix.md](../engine-matrix.md)),
the row that owns `ResidentNvfp4`'s upload and adoption path and the spec
[`load-modelopt-nvfp4-borrow.md`](load-modelopt-nvfp4-borrow.md).

## The defect, grounded

| Where (line anchors at this branch's base, `b45a94273`) | What |
|---|---|
| `include/vllm/model_executor/models/qwen3_5_weights.h:707-708` | `Nvfp4Weight::d_packed`/`d_scale`, documented as "the shared_ptr deleter frees through the vt Backend". |
| `.../qwen3_5_weights.h:195` | `OwnedTensor::d_dev`, the generic raw-twin slot `AdoptDeviceBytesAsHost` keys on (`src/vllm/model_executor/models/qwen3_5_weights.cpp:420-429`). |
| `.../dense_nvfp4_gemm.h:319-347` | `ResidentNvfp4` sets `w.d_packed = shared_ptr(p, Free)` and then `w.packed.d_dev = w.d_packed`: **two owning handles, one control block.** |
| `src/vllm/model_executor/models/qwen3_5.cpp:1449-1476` | The private twin of the same function, the same two-handle shape. |
| `.../dense_nvfp4_gemm.h:448-449`, `:667-670` | `BuildMarlinDenseResident` and `BuildMarlinDensePairResident` release the type-specific pair only. |
| `src/vllm/model_executor/models/qwen3_5.cpp:2952-2953`, `:3132-3135`, `:6890-6891` | The 27B dense and MoE repack builders, the same release shape. |
| `src/vllm/model_executor/models/laguna.cpp:650-655` | Six resets per expert, the same shape. |

The alias is not an accident of one builder: it is the design (`packed.d_dev`
is what `AdoptDeviceBytesAsHost` can act on), and the fix is to make it the ONLY
owner rather than to teach six call sites to drop two handles.

## The design call, and what it costs

Alternatives rejected:

1. **Keep both handles and reset both at every caller.** Correct at each site and
   wrong for the codebase: six release sites, one of which (a new builder) will
   forget. The bug is the duplicated ownership, not the six callers.
2. **Make `d_dev` observe `d_packed` without owning it** (a raw pointer or a
   `weak_ptr`). `AdoptDeviceBytesAsHost` needs the block alive while the adopted
   host view exists, and the residency machinery keys on `d_dev` everywhere else;
   splitting the lifetime rules between two slots reintroduces the same class of
   mistake.
3. **Have `ReleaseResident` free through the deleter directly.** It cannot: the
   adopted `bytes` view holds a copy of the shared_ptr on a host-addressable
   device, so the block outlives the weight's handle by construction. The view
   must be dropped first, which is what `ReleaseResident` does.

## Design

One owner per allocation: `packed.d_dev` / `scale.d_dev`. The type-specific
`d_packed`/`d_scale` members are deleted, so a reader that still expects them
fails to **compile** — the migration cannot be half-done.

Two additions carry the release:

- `OwnedTensor::HostViewIsDeviceTwin()` — true when `d_dev` is set, `bytes` is an
  adopted (borrowed, `mmap_src == nullptr`) view, and it points at that same
  block. This is the predicate for "the host view is an alias of this tensor's
  own device allocation", i.e. it holds the block alive.
- `Nvfp4Weight::ReleaseResident()` — drops an adopted twin view first, then
  resets `d_dev` on both tensors. Called at all six release sites.

`ResidentNvfp4` (both copies) uploads into `packed.d_dev`/`scale.d_dev`
directly. The old upload guard `!w.d_packed` becomes `!w.packed.d_dev`; that is
equivalent because the upload is the only writer of either slot for an
`Nvfp4Weight`'s packed/scale tensors (`grep -rn 'd_dev = ' src include`).

**Behavior is unchanged: same bytes, same adoption point, same release points.**
The only new effect is that the adopted alias view is dropped before the block is
freed, which is required for the free to happen at all.

## What this does NOT claim

No end-to-end GiB number is measured here. The unit gate proves that a release
through the weight's own resident state frees both buffers; it does not measure
the process RSS of a real repacking serve on a real checkpoint. That measurement
needs a model and a GPU and is the natural follow-up, not a precondition for the
correctness fix.

## Tests

`tests/vllm/test_load_direct_upload.cpp`, which already owns the `ResidentNvfp4`
accounting/adoption cases over the `FakeBackend` + `ObservableMapping` harness:

- New: `fp4 resident: releasing the resident frees both device buffers`. Uploads
  a borrowed fp4 weight on a non-host-addressable fake device (no adoption, so
  the alias is the only extra reference), then performs the release and asserts
  `b.frees == 2`.
  - **RED pre-fix**, captured:
    `CHECK( b.frees == 2 )` → `values: CHECK( 0 == 2 )`, because the release
    reset only the type-specific pair.
  - **GREEN post-fix**: the release is `w.ReleaseResident()`.
- The three existing cases are updated to the single-owner shape
  (`w.packed.d_dev` / `w.scale.d_dev` in place of `w.d_packed` / `w.d_scale`),
  and their final `CHECK(b.frees == 2)` (the weight out of scope) is unchanged.

What the gate does not prove: that a real CUDA driver frees at that instant. The
`FakeBackend` counts `Free` calls; it cannot observe the driver.

## Gates

- `cmake --build build --target test_load_direct_upload && ./build/tests/test_load_direct_upload`
  — RED before, GREEN after, all 17 cases.
- The full `ctest --test-dir build` on the affected tree.
- `scripts/agent-preflight.sh` (the record gates for the changed files).
- The build itself is the completeness check for the deleted members.

## Stop conditions

- A caller of `d_packed`/`d_scale` outside the sites listed above: stop and
  re-scope, because the deletion is no longer complete.
- A regression in the adoption cases (host-addressable device): stop, because
  the release ordering is wrong.
