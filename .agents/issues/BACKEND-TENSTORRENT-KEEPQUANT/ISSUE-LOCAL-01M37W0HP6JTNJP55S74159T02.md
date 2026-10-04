ID: ISSUE-LOCAL-01M37W0HP6JTNJP55S74159T02
Title: Flip VT_TT_KEEPQUANT_INT8DOT to default-on: the default-off denominator is invalidated and both arms are in-band
Row: BACKEND-TENSTORRENT-KEEPQUANT
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-23
Updated: 2026-10-04
Closed: -

## Problem

The W4b lever stayed default-off (#3031) because its e2e lane failed the 500-mnat band against llama.cpp b10451 INCREMENTAL greedy. The 2026-09-22 GSQ investigation proved that denominator invalid (the pin's own batch decode disagrees with its incremental greedy at identical causal positions). Re-adjudicated against the batch decode, both arms are in-band (default 26.6, INT8DOT 245.3 mnats, 16/16 each) and the lever cuts TPOT 1253->642 ms (2x) and trace demand -33.8%. The default flip is the row's pending decision; this spec is its gate plan.

## Resolution

-

- 2026-10-04 (`row/int8dot-default-flip`): gates re-run on the LIVE 27B arm
  after wave 3 (evidence `docs/bench-evidence/tt-int8dot-flip-live-gates-20261003.md`,
  spec section "Live-arm re-run"). c1 arms token-IDENTICAL (64/64 ids) and
  the c1 A/B reads 4.50x (31,277 vs 6,957 ms TPOT); the c2 arm-pair
  divergence is within-arm run-to-run nondeterminism (a c2 `=0` re-run
  reproduced the `=1` stream byte-for-byte), not a lever effect. The flip
  stays blocked: the pinned llama.cpp b10451 REFUSES the unsloth Q4_K_M
  artifact (`missing tensor 'blk.64.ssm_conv1d.weight'`; nextn KV override
  inert), so the 500-mnat teacher-forced band of spec gate 1 has no
  denominator for this checkpoint; sibling gate and `=0` opt-out identity
  are re-owed on the live arm. VERDICT: DONT-FLIP, lever stays opt-in,
  issue stays OPEN pending a working oracle denominator for this artifact.
