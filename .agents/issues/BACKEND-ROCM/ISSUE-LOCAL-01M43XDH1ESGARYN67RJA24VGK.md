ID: ISSUE-LOCAL-01M43XDH1ESGARYN67RJA24VGK
Title: Dead code on rocm-gfx11-exl3-perf: orphan exports, measured-dead knob arms, unselectable instantiations
Row: BACKEND-ROCM
State: OPEN
Kind: cleanup
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-04
Updated: 2026-10-04
Closed: -

## Problem

Adversarial dead-code sweep 2026-10-04 (full table in the campaign record). Certain removals: Exl3GemvShapeEligibleRocm + Exl3GemvChunkedTryLaunchRocm (born orphans, zero callers); dead bits>=5 carve-outs at rocm_exl3_gemv.hip:1758-1763 and 1785-1790 (GemvArmInstantiated returns false first); duplicated clause in GemvArmInstantiated:1472-1476 with a false comment (6bpw m1 is served by the dot arm, not a span kernel); unreachable (void)cb; 8 Exl3GemvM1KFused<2/3,*> instantiations unreachable behind the bits!=4 gate at :1964; HadK<2,2>/<1,2> instantiations no caller can name; DotTileWords empty struct; adump0/adump1 never-written locals; always-true tile<ntiles guard; duplicate (4,0) test row. Measured-dead knob arms the spec itself records as regressions: VT_ROCM_EXL3_WMMA (3.3x slower, ~200 lines + 12 instantiations), VT_EXL3_GEMV_CFG (14.7 tok/s), VT_EXL3_GEMV_KSPLIT and VT_EXL3_DOT_KSPLIT (regressions), VT_EXL3_DOT_FIRST (3x slower). Also check-env-doc.py fails on 4 undocumented knobs (VT_EXL3_DOT_KSPLIT, VT_EXL3_GEMV_KSPLIT, VT_EXL3_RECON_NO_LT, VT_ATTN_PREFILL_F32Q_B16KV) and 2 stale defaults (VT_ATTN_PREAMBLE_COOP, VT_GDN_NORMGATED_COOP now default ON).

## Resolution

2026-10-04 (this worktree): landed — Exl3GemvShapeEligibleRocm +
Exl3GemvChunkedTryLaunchRocm deleted; bits>=5 carve-outs and the duplicated
GemvArmInstantiated disjunct removed with the comment corrected to name the
dot TU; GemvM1FusedKernelFor bits 2/3 rows deleted (call-site gate is
bits==4); HadK<1,2>/<2,2> instantiations deleted; DotTileWords struct and
adump0/adump1 dead stores deleted; the (void)cb trap resolved by dropping
the cb parameter. Deliberately retained: the always-true tile<ntiles guard
(harmless defensive check), VT_EXL3_FUSED_HAD's arm (A/B lever, ctest entry
exercises it), and the LaunchHadIO dispatch table's other pairs (documented
contract, cheap). The measured-dead knob arms note still stands open.
