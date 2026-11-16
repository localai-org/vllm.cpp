ID: ISSUE-LOCAL-01M7HMM79822JF0D68KQAJJDRX
Title: The non-expert (spine) half of the Mach-2 decoder is missing, so no attention or dense tensor can be decoded
Row: QUANT-MACH2-ADDITIVE
State: CLOSED
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-16
Updated: 2026-11-16
Closed: 2026-11-16

## Problem

U1 landed the expert path only. decode_ne_tensor, _ne_states, _ne_full_lut and rc_grid were unimplemented, and the spine rotation needs the _HS string tables that the earlier fetch had mangled, so nothing that is not an MoE expert could be turned into weights - which is most of the model's non-expert parameters.

## Resolution

The spine path is ported: HadamardSpine with both _HS tables, DecodeNeTensor, plus the TrellisStates/FoldSpineLut/RcGrid pieces already present. Differential against the unmodified decode.py: 51,184/51,200 values bit-identical at 160x320 (radix 20) and 4,601/4,608 at 96x48 (radix 12), residual about one ulp. _HS is generated from the published file rather than typed. Two findings: _HS is orthogonal but not symmetric, so the spine transform is not an involution and the correct invariant is Parseval - the first test asserted twice-is-identity and was wrong; and rc_grid multiplies by 1/127 rather than dividing, which differs on 100 of 889 (mx,k) pairs and needed a pinned discriminating pair before the mutation failed. Six mutations, all caught. 16 cases and 3513 assertions green.
