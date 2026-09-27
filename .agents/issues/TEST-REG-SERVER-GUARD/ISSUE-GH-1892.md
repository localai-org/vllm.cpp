ID: ISSUE-GH-1892
Title: `check-pr-size.py`'s checker-change evidence harness ran `tests.scripts.test_check_test_registration` with neither `cmake`, `ctest` nor `ninja` reachable: `EVIDENCE_REQUIRED_TOOLS` named only the windows-portability module, so the sanitized `PATH` (`os.defpath` plus an empty private tools directory) could not start the programs that module drives. CI reported `FileNotFoundError: 'cmake'` with 22-26 errors, all charged to the checker under change rather than to the harness -- the broken-instrument shape, and the same gap [#458](https://github.com/mudler/vllm.cpp/issues/458) closed for the other module. Invisible until now because the harness only runs when a checker and its evidence file change together. Fixed in flow with #1883: the module declares its tools, and the test DERIVES the expectation from the checker's own argument-list literals rather than transcribing a list, so the `ctest` that the first fix missed reds locally instead of in CI. Spec [test-registration-server-guard.md](../../specs/test-registration-server-guard.md) §10
Row: TEST-REG-SERVER-GUARD
State: UNKNOWN
Kind: bug
GitHub: 1892
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:709`

### Frozen archive evidence

> | [#1892](https://github.com/mudler/vllm.cpp/issues/1892) | `TEST-REG-SERVER-GUARD` | `check-pr-size.py`'s checker-change evidence harness ran `tests.scripts.test_check_test_registration` with neither `cmake`, `ctest` nor `ninja` reachable: `EVIDENCE_REQUIRED_TOOLS` named only the windows-portability module, so the sanitized `PATH` (`os.defpath` plus an empty private tools directory) could not start the programs that module drives. CI reported `FileNotFoundError: 'cmake'` with 22-26 errors, all charged to the checker under change rather than to the harness -- the broken-instrument shape, and the same gap [#458](https://github.com/mudler/vllm.cpp/issues/458) closed for the other module. Invisible until now because the harness only runs when a checker and its evidence file change together. Fixed in flow with #1883: the module declares its tools, and the test DERIVES the expectation from the checker's own argument-list literals rather than transcribing a list, so the `ctest` that the first fix missed reds locally instead of in CI. Spec [test-registration-server-guard.md](../../specs/test-registration-server-guard.md) §10 | bug |

## Resolution

-
