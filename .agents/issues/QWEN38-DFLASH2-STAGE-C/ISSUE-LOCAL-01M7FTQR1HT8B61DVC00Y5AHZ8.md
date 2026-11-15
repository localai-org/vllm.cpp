ID: ISSUE-LOCAL-01M7FTQR1HT8B61DVC00Y5AHZ8
Title: The verify's greedy rejection sampler has no Tenstorrent kernel (op id 107) — the next by-name refusal after kGdnSpecDecode
Row: QWEN38-DFLASH2-STAGE-C
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-15
Updated: 2026-11-15
Closed: 2026-11-15

## Problem

The W3 evidence named kGdnSpecDecode (id 12) as the loop's next blocker; with that kernel ported, the w4-on-probe leg runs the draft propose, the (S, 9, spec) verify forward, the GDN spec step — and dies at the NEXT by-name refusal: 'no kernel for op GreedyRejectionSample (id 107) on device tenstorrent'. vt::GreedyRejectionSample is CPU+CUDA only (cpu_sample.cpp:315, cuda_sample.cu:1013); the verify's block logits are TT-device tensors, so the sampler must resolve on TT. The TT arm follows the backend's sampling precedent (kGreedyArgmaxKernel is host-staged: EnsureHost, the same strict-> lowest-index tie-break, CommitHost) and the direct-argmax draft arm already downloads these block logits every verify step, so the eager arm pays no new traffic class. Ported 1:1 from the CPU reference — the TT arm is bit-exact vs the CPU oracle by construction (same host arithmetic, no device float path). The captured spec-verify arm stays the standing named gap (the acceptance metadata has no persistent device refresh); the spec verify runs eager (VLLM_CPP_CUDAGRAPH=0).

## Resolution

Ported in the same wave 2026-11-15: kGreedyRejectionSample registered on Tenstorrent (tenstorrent_ops.cpp, host-staged per the kGreedyArgmax precedent, ported 1:1 from the CPU reference — the TT arm is bit-exact vs the CPU oracle by construction). The w4-on-probe leg runs the draft propose, the (S,9,spec) verify, the GDN spec step and the sampler to completion (exit 0, acceptance telemetry live) — the refusal is gone. Focused gate: 3 bit-exact cases (full accept, reject-at-depth, two requests) in tests/vt/test_tenstorrent_backend.cpp, 4/4 test cases EXIT 0 (w4-gdnspec-green.log). The captured-sampler arm stays the standing named gap (the acceptance metadata has no persistent device refresh; the spec verify runs eager).
