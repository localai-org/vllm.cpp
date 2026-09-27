ID: ISSUE-GH-1883
Title: `test_minimax_music3_e2e_real` was registered outside `if(VLLM_CPP_SERVER)` while calling `ApiServer`, whose translation unit `CMakeLists.txt` compiles only inside it, so every `-DVLLM_CPP_SERVER=OFF` configure succeeded and then failed at `ld` with 16 `undefined reference` lines -- the standing red on `main`'s `build-test-cpu-arm64-full` job. Fixed in flow: the registration moves inside the guard beside the other server-linking suites, and `scripts/check-test-registration.py` gains a static `server_guard_errors` check that derives the gated translation units from the top-level CMake, resolves their declaring headers, walks the tree's own include graph transitively, and refuses any `vllm_cpp_add_test` target outside the guard that reaches one. Spec [test-registration-server-guard.md](../../specs/test-registration-server-guard.md)
Row: TEST-REG-SERVER-GUARD
State: UNKNOWN
Kind: bug
GitHub: 1883
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:708`

### Frozen archive evidence

> | [#1883](https://github.com/mudler/vllm.cpp/issues/1883) | `TEST-REG-SERVER-GUARD` | `test_minimax_music3_e2e_real` was registered outside `if(VLLM_CPP_SERVER)` while calling `ApiServer`, whose translation unit `CMakeLists.txt` compiles only inside it, so every `-DVLLM_CPP_SERVER=OFF` configure succeeded and then failed at `ld` with 16 `undefined reference` lines -- the standing red on `main`'s `build-test-cpu-arm64-full` job. Fixed in flow: the registration moves inside the guard beside the other server-linking suites, and `scripts/check-test-registration.py` gains a static `server_guard_errors` check that derives the gated translation units from the top-level CMake, resolves their declaring headers, walks the tree's own include graph transitively, and refuses any `vllm_cpp_add_test` target outside the guard that reaches one. Spec [test-registration-server-guard.md](../../specs/test-registration-server-guard.md) | bug |

## Resolution

-
