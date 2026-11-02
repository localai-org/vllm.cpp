# CLAIM-FLINT-TT-DENSE-MLP-STAGING

| Field | Value |
|---|---|
| Row | `FLINT-TT-DENSE-MLP-STAGING` |
| Worktree | `/tmp/flint-p150-attn` |
| Branch | `row/FLINT-TT-DENSE-MLP-STAGING` (base: FLINT-P150-ATTENTION content, spec+issue committed `ed1f3d919`) |
| Owner | Maki (zai-glm-5.3-flash), fresh implementer per the committed spec |
| State | `RUNNING` — diagnose MEASURED (attribution table below), fix NOT landed |
| Last update | 2026-11-01 (leg (a) landed-order running) |

Owes: per-tensor rel_rms at L0/L46 under three staging orders, the fix that
makes the W0 band allocation-order-insensitive, the standing gates, and the
evidence dir `docs/bench-evidence/flint-tt-dense-mlp-staging-20261101/`.

## Session handoff (2026-11-02, diagnose measured, fix NOT landed)

Measured (monitor logs in docs/bench-evidence/flint-tt-dense-mlp-staging-20261101/):

- W0 band under 4 staging-order legs: landed / reversed / extended-DBuf all
  2.683e-1; oproj-scoped 7.224e-1 — BOTH P150-ATTENTION record numbers
  reproduced exactly (the 7.224e-1 leg is leg-d-oproj-scoped.log).
- The W2 "dense gate_proj outlier" (TT −14168 vs CPU +15.3) is a TAG MISMATCH
  (TT g is pre-activation, the CPU dump's g is post-activation); like-for-like
  the gate matches to 0.09% (silu|gu rel_rms 3.2e-3).
- The corruption under oproj-scoped sits in the DOWN projection's OUTPUT
  slot: the matmul's CommitDevice2D lands the result, but the download reads
  a slot whose served shadow is path-dependent. down|recomp (fresh buffer,
  re-uploaded a bytes) is bit-correct in every leg; down|live (the model's
  own slot) carries the corruption. down rel_rms: 1.10e-2 landed vs 1.8059
  oproj-scoped.
- The zero-out leg (VT_FLINT_TT_ZERO_OUT=1) is INSTRUMENT-POISONED: the Zero
  call sat between the matmul and the download — rerun with the Zero BEFORE
  the matmul before trusting any read from it.

Next lever (unchanged from the spec's option (a)): explicit commit boundaries
at the dense staging hand-offs, THEN the standing gates (probe 64/64 both
window modes, paged 208/208, gguf serving, MiMoV2 sink, dtype bit-identity).
No commits made — the worktree carries the instrument + the diagnostic edits
uncommitted.

## Orientation facts (verified this session, 2026-11-01)

- Spec: `.agents/specs/flint-tt-dense-mlp-staging.md` — D1 instrument
  per-tensor rel_rms at L0/L46 under (a) landed (b) reversed (c) extended
  DBuf lifetimes; D2 the fix that makes the W0 band order-INSENSITIVE under
  every DBuf lifetime permutation; D3 the evidence dir with per-tensor
  tables, the attribution verdict (order vs tensor), and raw logs.
- `MatmulWeight` (flint_tt.cpp:538-546) is the dispatch seam: block weights
  route `vt::MatmulBT`, float weights `vt::Matmul`.
- `DumpOn()` (flint_tt.cpp:100) is the env-dump gate to mirror
  (FLINT_TT_W0_DUMP / FLINT_CPU_DUMP, test_flint_tt_w0.cpp pattern).
- ForwardLogits (qwen3_5.h:182): host vector + device_storage shared_ptr +
  device_tensor view. DBuf: dense_device_glue.h, pooled move-only RAII.
- dense_attn namespace: dense_attn_block.h:51-841.
- Test registration: `vllm_cpp_add_test()` tests/CMakeLists.txt:18.
- Device recipe: `exec 9>$HOME/gpu.lock; flock -x 9` mutex, luwen reset +
  15 s, build /tmp/build-p150staging
  (`-DVLLM_CPP_TENSTORRENT=ON`, CMAKE_PREFIX_PATH=/tmp/umdtrial-install/
  lib64/cmake;/tmp/umdtrial-install/share/cmake), device legs
  `taskset -c 32-63`, governor `performance` logged, never changed.

## Next steps

1. Instrument flint_tt.cpp: env-gated per-tensor rel_rms dumps of the dense
   gate/up/down staging at L0 and L46 vs the CPU shadow.
2. Build /tmp/build-p150staging; run the 3 staging-order permutations.
3. Attribution verdict (order vs tensor); fix; gates (probe 64/64, paged
   208, gguf serving, dtype bit-identity); evidence dir; commits
   (diagnose / fix / evidence, trailers LAST); push to `mine` (never origin).
