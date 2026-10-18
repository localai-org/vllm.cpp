# Qualify the historical Kolibri-1 TT capture study

## Scope

Documentation correction for `BACKEND-TENSTORRENT`, tracked by
`ISSUE-LOCAL-01M4YR6QSEEZEAAW2FJNANM67V`. Base: `50d425bec`.
Preserve the capture study and all original measurements. Identify which
observations predate the deterministic-decode and golden-index fixes.
No implementation, lifecycle, default, or benchmark disposition changes.

## Source and design

| Surface | Source | Change |
|---|---|---|
| Prototype timings | `docs/gap-kolibri1-tt-captured-decode.md` | Label historical results and qualify their interpretation |
| Cache lifetime fixes | `src/vllm/model_executor/models/kolibri1_tt_forward.cpp`, `src/vt/tenstorrent/tenstorrent_residency.cpp`, commit `ad8d9b45b` | Explain recorded repeatability after fixes |
| Correctness comparison | `tests/vllm/models/test_kolibri1_tt_b2ii.cpp`, commit `50d425bec` | Explain corrected decode indexing and remaining failures |
| Retained evidence | `.agents/issues/BACKEND-TENSTORRENT/ISSUE-LOCAL-01M4P21TZ1Q9CA7P66XBH0BQTX.md` | Link the recorded results without claiming a fresh run |

Add a short current-status note near the start so readers see the limitation
before the analysis. Mark the prototype section as historical. Qualify its
nondeterminism attribution and its conclusion about bias caching. Preserve
original timings and structural capture limitations. Avoid a new speed ratio.

## Tests and gates

This is prose-only work. No upstream tests need porting and no runtime oracle
or accelerator is required. Verify every new assertion against the sources
above. Check that all three timing rows remain byte-identical. Check changed
links locally, run `git diff --check`, `scripts/check-site.py`, and
`scripts/check-agent-record.py`. Run the full preflight and report unavailable
tooling separately from the focused documentation gate. A fresh reviewer
checks the immutable implementation commit and the source evidence.

## Risks and stop conditions

Repeatable output does not prove oracle correctness. The recorded token gate
still fails. Historical timings cannot establish current performance. Stop if
a proposed claim requires new accelerator measurements. Do not edit README
news: this corrects an old study, and existing PRs own the new CPU features.

## Work breakdown

1. Commit this scope and local issue before editing public prose.
2. A fresh implementer edits only the study and records source checks.
3. A fresh reviewer checks the committed diff. The operator reruns the focused
   gate before pushing to the maintenance fork and opening an upstream PR.

## Implementation evidence (15 October 2026)

The pre-edit study attributed differing greedy chains to a seven-flip residue
and concluded that bias caching was not a decode lever. The revised study
labels that interpretation as historical and links the subsequent resolution.
The three timing rows remain byte-identical to the committed spec base.

Source checks:

- `kolibri1_tt_forward.cpp:810-822` evicts before emplacement.
- `tenstorrent_residency.cpp:665-693` removes shadows within a retiring
  allocation. Calls at lines 1126 and 1179 cover freeing and scratch reuse.
- `test_kolibri1_tt_b2ii.cpp:1137-1147` compares decode positions.
  Lines 1233-1241 correct the teacher-forced prefix.
- The retained issue's acceptance section records three identical chains per
  arm. Its fixed-index evidence records two identical runs per arm with
  correctness checks still failing. These are recorded results, not new runs.

Verification used the supplied Python environment:

```sh
export PATH=/tmp/vllm-doc-python/root/usr/bin:$PATH
export LD_LIBRARY_PATH=/tmp/vllm-doc-python/root/usr/lib
export PYTHONHOME=/tmp/vllm-doc-python/root/usr
```

| Check | Result |
|---|---|
| `git diff --check` | Exit 0 |
| `python3 scripts/check-site.py` | Exit 0, 13 published documents |
| `python3 scripts/check-agent-record.py` | Exit 0, `ANCHOR-ROT=0` |
| Python comparison against `git show HEAD:docs/gap-kolibri1-tt-captured-decode.md` | Exit 0, all three timing rows identical |
| Local Markdown link existence check | Exit 0 |
| `scripts/agent-preflight.sh` | Incomplete, stopped at the coordinator's request after duplicate baseline failures |
| `timeout 10 scripts/agent-preflight.sh --staged` | Exit 143, bounded duplicate preflight, incomplete |

Preflight reported missing PyYAML and a test-registration configuration
failure. The coordinator's baseline identifies unavailable CMake as the
configuration blocker. The coordinator also reports that the rendered-site
test lacks Hugo. Its full preflight also encounters release-tooling failures
and subprocesses that cannot load the isolated Python library. Full preflight
remains FAILED or INCOMPLETE because of host tooling. The focused
documentation checks pass independently.
No runtime test, benchmark, GPU execution, or mutation test applies to this
prose-only correction. No tests or executable guarantees changed.
