ID: ISSUE-LOCAL-01M5ECNRB11FFA72MJK5KTAG9F
Title: windows-msvc is red on main: kolibri1_forward.cpp shadows 'prof' (C4456) under -WX
Row: ENG-RELEASE-WINDOWS
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-21
Updated: 2026-10-21
Closed: 2026-10-21

## Problem

origin/main fails windows-msvc-cpu and windows-msvc-vulkan in the 'Build and execute the native Windows CPU focused gate' step (and its Vulkan twin) because ONE source file shadows a local, and the lane's /W4 /WX turns the warning into an error.

CI-log evidence (job logs, verbatim):

  job 118938856642 (windows-msvc-cpu, run 39513156212, head 46c6846b93, 2026-10-20T23:09:47Z):
    D:\a\vllm.cpp\vllm.cpp\src\vllm\model_executor\models\kolibri1_forward.cpp(326,17): error C2220: the following warning is treated as an error
    D:\a\vllm.cpp\vllm.cpp\src\vllm\model_executor\models\kolibri1_forward.cpp(326,17): warning C4456: declaration of 'prof' hides previous local declaration
          D:\a\vllm.cpp\vllm.cpp\src\vllm\model_executor\models\kolibri1_forward.cpp(306,15):
          see declaration of 'prof'
    ... cmake exited with status 1

  job 118938857142 (windows-msvc-vulkan, same run): the identical two lines.

  The same two lines are the red in the NEXT run, job 119042514950 (windows-msvc-cpu)
  and job 119042515207 (windows-msvc-vulkan), run 39545458600, head 46c6846b93,
  2026-10-21T04:41:56Z, so the signature is current and not a stale read. That log
  also carries ZERO C4101 diagnostics.

The same signature is the red on job 118305585185 (2026-10-19, head 61d333e525) and job 118149718329 (2026-10-19, head 4b3a0091c7); the whole build then ends with 'cmake exited with status 1'.

Cause: src/vllm/model_executor/models/kolibri1_forward.cpp declares 'prof::Scope prof("linear_gemm");' at :306 (the function-scope stage profiler) and a second 'prof::Scope prof("linear_gemm");' at :326 inside the BFDOT bf16-GEMV arm. Three main commits landed after that head (d9e669582, e451be705, 3ce14f35c) and moved the site to :349 with the outer scope at :329, without changing the shape; their windows-msvc jobs are pull-request-only and were skipped, so no newer MSVC log exists for them. The inner declaration hides the outer local, which is MSVC C4456. It entered with 6be7f0214 (the knob-gated BFDOT bf16 GEMV arm, 2026-10-11). GCC and Clang do not warn on this shape by default, so it built green on Linux and landed.

This is a DIFFERENT defect from the one ISSUE-LOCAL-01M58B5J4DHKN5AEVCRR6Y7YWE closed on 2026-10-18 (the C4101 unreferenced 'numa_guard' and the C4456 shadowed 'e' in kolibri1_tt_forward.cpp). That pair was fixed on main by ecc64d147 and the current job logs confirm it is gone: the latest windows-msvc logs contain ZERO C4101 diagnostics and no diagnostic at all against kolibri1_weights.cpp, kolibri1_dequant_cache.h or kolibri1_tt_forward.cpp. The lane stayed red because this third, later diagnostic was still there. The fix is a rename of the inner variable; the profiling key string and every value stay identical.

## Resolution

2026-10-21: the inner scope is renamed bfdot_prof at src/vllm/model_executor/models/kolibri1_forward.cpp:326, so the C4456 is gone by construction. The profiling key, the scope boundaries and the accumulated time are unchanged, and no value moves.

RED: tests/scripts/test_msvc_wx_diagnostics.py compiles the production translation units with the host compiler's -Wshadow and fails on the pre-fix source with 'kolibri1_forward.cpp:326:17: declaration of prof shadows a previous local', the same file, line and column the MSVC job log names. The same suite holds the C4101 class, which no host compiler can express, because GCC's -Wunused-variable does not fire for a class with a non-trivial constructor. FALSIFIED on 2026-10-21; see the correction below.

GREEN: the Linux build is clean (cmake --build build -j 4, exit 0, zero warnings in the log) and the kolibri1 suites are identical on pristine origin/main (3ce14f35c) and on this branch: test_kolibri1_dequant_cache 13/13 cases 66150 assertions, test_kolibri1_numa 3/3 cases 11 assertions, test_kolibri1_dequant 3/3 cases 10 assertions, test_kolibri1 27/27 cases 234 assertions, test_kolibri1_w2 7/7 cases 1608 assertions. The branch is rebased onto 3ce14f35c, the main head this issue was filed against plus the three commits that moved the site to :349; the rename and the probe were re-verified at that location, where the pre-fix source reports kolibri1_forward.cpp:349:15. The Windows lane itself is not observable from this host, so the MSVC verdict rests on the host compile probe at the exact MSVC site.

MEASURED AND NOT FIXED HERE, for the record. src/vllm/multimodal/ltx2_video.cpp:3931-3933 holds three local shadows that GCC's -Wshadow reports. MSVC does not report them, and the Windows lane compiled that file clean (job 118938856642, 2026-10-20T23:23:15Z); Clang does not report them either, so they are not a lane red. They are a deliberate, documented shadow. The Linux lane on main is red for a different class entirely: the CI container refuses set_mempolicy(MPOL_INTERLEAVE), which fails test_kolibri1_numa, test_kolibri1_dequant_cache, test_kolibri1_w2, test_kolibri1, test_ops_paged_attn_dtype and test_glm4_moe_lite_paged_engine. The last two fail identically on pristine origin/main and on this branch, so they are not this change's.

CORRECTION, 2026-10-21. The C4101 claim above is false. An unreferenced local of an EMPTY class with a trivial default constructor does trigger -Wunused-variable, and both host compilers report it. The reason the class did not appear is the SHAPE the probe compiled: on Linux kolibri1_numa::PolicyGuard has a non-trivial constructor, so the host never saw an empty class. The probe now compiles the production translation units a second time in the non-Linux shape, with '-U__linux__', which selects the same empty-guard branch of kolibri1_numa.h that _WIN32 selects there, and it carries -Wunused-variable in that shape. The probe also reports a diagnostic in a repository header, because the C4101 site in kolibri1_dequant_cache.h is a header site, and it checks that the probed units reach every production PolicyGuard local so the class cannot go quiet when a guard moves.

MUTATION EVIDENCE, 2026-10-21, one file at a time, each restored byte-for-byte (sha256 verified before and after). C4456, a shadow in a different translation unit: kolibri1_tt_forward.cpp:535:19. C4456, a shadow in an included header: kolibri1_numa.h:93:9. C4101, the header site: kolibri1_dequant_cache.h:351:38. C4101, the translation unit site: kolibri1_weights.cpp:225:36. The C4101 sites are reported by g++ 16.2.0 and by clang++ 22.1.6, and by neither in the host shape, which is what the shape switch buys.
