ID: ISSUE-LOCAL-01M7GZ3ZNQ59M7647B2SBACBZX
Title: Nothing tests whether vLLM's argmax depends on the decode path it was measured through
Row: MODEL-TEXT-GLM4-MOE-LITE-GATE-2839
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-15
Updated: 2026-11-15
Closed: 2026-11-15

## Problem

The committed GLM-4.7-Flash goldens say the STRICT bar is licensed because five greedy runs agree with each other. Five runs of the same call share one numeric path: prefill kernels, batch shape and decode kernels are identical between them, so the experiment cannot see a disagreement between the greedy-decode path and the teacher-forced prefill prompt_logprobs path. At the seven first divergences the committed gap is under half a millinat and the argmax it was measured against is OUR token, which is what a path-dependent oracle would look like. K2 measures it directly: run the oracle greedy decode and the teacher-forced prefill on the same prompt and prefix, under the same and different max_num_seqs, and record where the two paths' argmaxes differ. No assertion moves; the probe produces the evidence a ratification or a forward repair would rest on.

## Resolution

K2 landed on row/MODEL-TEXT-GLM4-MOE-LITE-GATE-2839-K2 based at 13542b40a. scripts/glm4-moe-lite-path-determinism.py runs the greedy-decode path that produced greedy_ids.npy and the teacher-forced prefill path that measured neartie_gap_mnats.npy, at each --max-num-seqs, and compares them with compare_paths; it also records whether the greedy path reproduces the committed goldens. It writes path_determinism.json beside the goldens and prints ORACLE PATH-DEPENDENT or ORACLE PATH-STABLE. Red before: tests/scripts/test_glm4_path_determinism.py failed 5 of 5 with the tool absent. Green after: 5 of 5, covering agreeing paths, an argmax that moves to our token, a no-distribution -1 that is not a comparison, an exact prompt with no first divergence, and a refused shape mismatch. The vLLM-facing functions are unexecuted on this host: no GPU and no 31.2B snapshot, stated in the tool header. No golden regenerated and no assertion moved.
