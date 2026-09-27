ID: ISSUE-GH-1885
Title: **`test_ltx2_video` reports a DIFFERENT assertion count on every run of the same binary, so any count comparison across a diff measures noise.** Seven runs across two binaries at `5c789015e` (Release, x86-64, `SERVER=ON`) gave 4337, 4338, 4339, 4340, 4341, 4342 and 4343 — a 7-assertion spread — while `test cases: 105 | 105 passed | 0 failed | 0 skipped` held on every one. The two binaries are two worktrees at the same commit, so it is neither a source nor a configuration difference; [#1445](https://github.com/mudler/vllm.cpp/issues/1445)'s spec first explained it as `SERVER=OFF` against `SERVER=ON` and that explanation is wrong. It MATTERS rather than being cosmetic because this repository quotes assertion counts as evidence — `docs/FEATURES.md` carries `test_ltx2_dfr` 11/11, 652 assertions and `test_ltx2_tiling` 10/10, 915 assertions — and a reviewer comparing 4343 before against 4337 after will hunt for six assertions a diff deleted when it deleted none, which is a failure that reads as a result. It also puts a genuine five-assertion deletion inside the observed noise. NOT ESTABLISHED and deliberately not guessed at in the record: WHICH cases vary (timing-dependent loops in the phase-log live-tick and RSS-style cases are the obvious candidates and nothing has isolated them), and whether any other suite shares the property — `test_ltx2_pipeline` returned 3475 on three runs, which is three runs and not a claim. Owned by row `LTX25-DISTILLED-LORA-REQUIRED` and listed under `## Owed` in [ltx25-distilled-lora-required.md](../../specs/ltx25-distilled-lora-required.md). Not fixed in flow: the varying cases are pre-existing, that row touches none of them, and isolating them needs a bisection over 105 cases
Row: LTX25-DISTILLED-LORA-REQUIRED
State: UNKNOWN
Kind: bug
GitHub: 1885
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:712`

### Frozen archive evidence

> | [#1885](https://github.com/mudler/vllm.cpp/issues/1885) | `LTX25-DISTILLED-LORA-REQUIRED` | **`test_ltx2_video` reports a DIFFERENT assertion count on every run of the same binary, so any count comparison across a diff measures noise.** Seven runs across two binaries at `5c789015e` (Release, x86-64, `SERVER=ON`) gave 4337, 4338, 4339, 4340, 4341, 4342 and 4343 — a 7-assertion spread — while `test cases: 105 \| 105 passed \| 0 failed \| 0 skipped` held on every one. The two binaries are two worktrees at the same commit, so it is neither a source nor a configuration difference; [#1445](https://github.com/mudler/vllm.cpp/issues/1445)'s spec first explained it as `SERVER=OFF` against `SERVER=ON` and that explanation is wrong. It MATTERS rather than being cosmetic because this repository quotes assertion counts as evidence — `docs/FEATURES.md` carries `test_ltx2_dfr` 11/11, 652 assertions and `test_ltx2_tiling` 10/10, 915 assertions — and a reviewer comparing 4343 before against 4337 after will hunt for six assertions a diff deleted when it deleted none, which is a failure that reads as a result. It also puts a genuine five-assertion deletion inside the observed noise. NOT ESTABLISHED and deliberately not guessed at in the record: WHICH cases vary (timing-dependent loops in the phase-log live-tick and RSS-style cases are the obvious candidates and nothing has isolated them), and whether any other suite shares the property — `test_ltx2_pipeline` returned 3475 on three runs, which is three runs and not a claim. Owned by row `LTX25-DISTILLED-LORA-REQUIRED` and listed under `## Owed` in [ltx25-distilled-lora-required.md](../../specs/ltx25-distilled-lora-required.md). Not fixed in flow: the varying cases are pre-existing, that row touches none of them, and isolating them needs a bisection over 105 cases | bug |

## Resolution

-
