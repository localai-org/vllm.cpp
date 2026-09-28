ID: ISSUE-LOCAL-01M3MQKE9M5BCGS4N80K2RDWG0
Title: CMakeLists.txt fetches parakeet.cpp at GIT_TAG main, a ref that repository does not have, so every configuring build job in the pull request lane fails at configure
Row: MODEL-AUDIO-PARAKEET-TRANSDUCER
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-28
Updated: 2026-10-01
Closed: 2026-10-01

## Problem

CMakeLists.txt:1596-1599 declares FetchContent for parakeet.cpp with GIT_REPOSITORY https://github.com/mudler/parakeet.cpp.git and GIT_TAG main. That repository has no main branch: git ls-remote --symref reports ref: refs/heads/master HEAD, and the head listing contains no refs/heads/main. FetchContent therefore fails during configure with fatal: invalid reference: main and CMake Error at parakeet_cpp-populate-gitclone.cmake:61 (message) Failed to checkout tag: main, which aborts configuration before any compilation. VLLM_CPP_WITH_DIARIZATION is ON by default, so every job that configures with the default feature set is affected. Measured on the pull request lane at base 1de097c46: build-test-cpu, build-test-vulkan, cuda-fat-build, build-newest-gcc, build-test-cpu-arm64, sanitize-cpu (address,undefined), sanitize-cpu (thread) and both verify jobs all fail with this one error, on every open pull request, because they share the same CMakeLists.txt. The introducing commit is 2833d6300 (feat(diarization): add diarization and SAS support via parakeet.cpp (ABI v30)), which is an ancestor of both 1de097c46 and current main, so the break is pre-existing on main and is not caused by any open branch. The push lane does not surface it because its build job does not configure the diarization feature set. All nine parakeet C-API symbols that vllm.cpp actually calls are present on master: parakeet_capi_load, parakeet_capi_free, parakeet_capi_diarize_path, parakeet_capi_diarize_pcm, parakeet_capi_transcribe_and_diarize, parakeet_capi_free_sas_results, parakeet_capi_free_string, plus the parakeet_ctx and parakeet_sas_result types. master also carries include/parakeet.h and include/parakeet_capi.h and defines the parakeet library target that CMakeLists.txt:1608 links, so the ref is the only thing standing between the tree and a working configure.

## Resolution

upstream/main fixed this on 2026-09-30 in e3b3f7971 (fix(SERVE-C-ABI):
pin parakeet.cpp to the commit the diarization seam compiles against): the
FetchContent GIT_TAG is now the immutable commit
394d270fabb1d6125f05c772aa3ca078a574b19d, the commit the diarization seam
in src/vllm/multimodal/diarization.cpp and src/capi/vllm_c.cpp was written
and compiled against (ABI v8 on the parakeet.cpp feat/diarization-sas
branch). That is the immutable pin the review of this branch asked for. The
branch change this issue was filed against (GIT_TAG master, a moving branch,
paired with GIT_SHALLOW ON) is superseded, and both halves are contradicted
upstream: a branch name is not a pin, and GIT_SHALLOW stays off because a
shallow clone cannot check out an arbitrary commit hash. The rebase therefore
takes upstream's CMakeLists.txt side wholesale and carries no CMakeLists
change of its own; what it adds is this record, closed here against
e3b3f7971.
