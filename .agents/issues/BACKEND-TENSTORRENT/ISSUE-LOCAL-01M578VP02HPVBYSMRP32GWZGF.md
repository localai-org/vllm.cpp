ID: ISSUE-LOCAL-01M578VP02HPVBYSMRP32GWZGF
Title: kolibri1 P150: tt-metal rms_norm shows a 0.7% input-dependent deviation on real in-model rows — kernel precision, not call-site config
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-18
Updated: 2026-10-18
Closed: -

## Problem

The tt-metal rms_norm arm (ttnn::rms_norm over f32 tiles, patched fp32_acc=true/approx_mode=false in the pinned fork's layernorm_default_compute_config) deviates ~0.7% from the f64 oracle on kolibri's REAL in-model activation distributions while reading clean (0.9997) on the micro's random N(0,0.5) inputs. DIAGNOSTIC RESULT (2026-10-18, P150): the standalone micro (test_kolibri1_tt_b2ii 'SCRATCH dbg rmsnorm micro', VT_TT_RMSNORM_MICRO_INPUT) replaying #3466-era captured rows (VT_TT_NORM_DUMP_DIR, /tmp/normdump) REPRODUCES the deviation — it is input-dependent kernel precision, not a call-site config difference. Numbers (tt sum / f64-oracle sum): the d=2560 sandwich-norm rows with large magnitudes (max|x| 3.55 and 5.5) read 1.0072 and 1.0077 (+0.7%); the d=128 per-head q/k rows read 0.9984 (clean). The same real rows also show the CPU arm further from the oracle than the TT arm (cpu/host2 1.0156 and 1.0130), so the tt-metal kernel is not the worst mover on these inputs; the composed f32 arm exists for A/B (VT_TT_RMSNORM_COMPOSED=1) but the model's tensor geometry refuses it on this stack (binary_ng rejects the subtile broadcast). FIX DIRECTION: inside the pinned tt-metal RmsNormKernel (layernorm.cpp compute kernel): the sum-of-squares already reduces in f32 when fp32_dest_acc_en is set (float32_reduction = fp32_dest_acc_en && !legacy_reduction), so the remaining suspects are the SFPU rsqrt approximation (LEGACY_RSQRT path) and the fp16-b intermediate data formats — both require a tt-metal-side change, beyond a vllm.cpp-level fix.

## Resolution

-
