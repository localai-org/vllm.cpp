ID: ISSUE-LOCAL-01M578VP02HPVBYSMRP32GWZGF
Title: kolibri1 P150: tt-metal rms_norm shows a 0.7% input-dependent deviation on real in-model rows — kernel precision, not call-site config
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-18
Updated: 2026-10-21
Closed: -

## Problem

The tt-metal rms_norm arm (ttnn::rms_norm over f32 tiles, patched fp32_acc=true/approx_mode=false in the pinned fork's layernorm_default_compute_config) deviates ~0.7% from the f64 oracle on kolibri's REAL in-model activation distributions while reading clean (0.9997) on the micro's random N(0,0.5) inputs. DIAGNOSTIC RESULT (2026-10-18, P150): the standalone micro (test_kolibri1_tt_b2ii 'SCRATCH dbg rmsnorm micro', VT_TT_RMSNORM_MICRO_INPUT) replaying #3466-era captured rows (VT_TT_NORM_DUMP_DIR, /tmp/normdump) REPRODUCES the deviation — it is input-dependent kernel precision, not a call-site config difference. Numbers (tt sum / f64-oracle sum): the d=2560 sandwich-norm rows with large magnitudes (max|x| 3.55 and 5.5) read 1.0072 and 1.0077 (+0.7%); the d=128 per-head q/k rows read 0.9984 (clean). The same real rows also show the CPU arm further from the oracle than the TT arm (cpu/host2 1.0156 and 1.0130), so the tt-metal kernel is not the worst mover on these inputs; the composed f32 arm exists for A/B (VT_TT_RMSNORM_COMPOSED=1) but the model's tensor geometry refuses it on this stack (binary_ng rejects the subtile broadcast). FIX DIRECTION: inside the pinned tt-metal RmsNormKernel (layernorm.cpp compute kernel): the sum-of-squares already reduces in f32 when fp32_dest_acc_en is set (float32_reduction = fp32_dest_acc_en && !legacy_reduction), so the remaining suspects are the SFPU rsqrt approximation (LEGACY_RSQRT path) and the fp16-b intermediate data formats — both require a tt-metal-side change, beyond a vllm.cpp-level fix.

## Resolution

-

## Update 2026-10-19: probed tt-metal upstream head (2500c1d8f63)

The bias is NOT improved upstream and regressed on the largest-magnitude captured row: tt/host replay 1.0072 -> 0.9809 (normrow-000, 1x2560, max|x|=3.55, max_abs 0.4375), 1.0077 -> 1.0033 (normrow-004), 0.9984 -> 0.9970 (normrow-001). Upstream switched rsqrt_tile<LEGACY_RSQRT> to ckl::Rsqrt<Approx::Exact> in a new eltwise_chain — did not fix the input-dependence. Full details: ISSUE-LOCAL-01M5B60NEFSEHYZXMR6FSS1RAF (includes the SFPI 7.88 rvtt_live ICE blocker and the still-needed local patch audit). Re-check trigger: SFPI >= 7.89 or an upstream rms reduce/round rework.

## Update 2026-10-21: the 0.9809 reading is the compute config, not an upstream change

The 2026-10-19 update above attributes the 0.9809 reading to upstream's rsqrt rework. A re-measurement on 2026-10-21 (P150, docs/bench-evidence/tt-rms-norm-compute-config-20261021.md) refutes that attribution and keeps the defect open with a sharper mechanism.

The two recorded replays ran two different tt-metal substrates, and the number follows the substrate's DEFAULT rms_norm compute configuration, not the tt-metal version. A standalone ttnn probe replays the same captured rows with an explicitly chosen configuration and reads 0.980947 with fp32_acc=false and 1.007197 with fp32_acc=true and approx_mode=false, on the pinned substrate and on upstream main alike. Upstream main and the pinned fork give the same number for the same configuration, so the upstream rsqrt rework changed no measured value.

The pinned 1.0072 reading is the patched-substrate regime. The 0.9809 reading is the unpatched regime, which the trial substrate of 2026-10-09 also had before the local precision edit was re-installed on 2026-10-10 17:36. The vllm.cpp rms path is byte-identical across the two sessions and passes no compute configuration, so no caller-side parameter changed.

The claim that the deviation is a property of the tt-metal kernel stands. The claim that upstream regressed does not. The 0.9984 figure quoted above for the d=128 rows is normrow-002, not normrow-001 (normrow-001 reads 0.995782 on the pin). The substrate's local precision edit is not in the recorded pin, which is filed as ISSUE-LOCAL-01M5EM67GWF0D79SSY3NHZRKQA.
