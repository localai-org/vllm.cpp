# Experimental mixed-prefill oneDNN admission on B70

The reviewed auto/2 baseline is commit `e69198adc06401dac1d26411bc03fb0ebd1b1e7a`.
The optional `VT_XPU_ONEDNN_MIXED_PREFILL=1` admits a one-query decode row
alongside existing multi-query prefill rows in the current oneDNN attention
path. The default remains `0`; pure Q1 batches retain their existing route.
Exact `0`/`1` parsing and graph-policy identity include this control. Existing
dtype, layout, scale, length and per-request bounds remain. The implementation
reuses current logical-row padding, causal masking and output copies.

## Cause and operator control

The original fresh 64K/MTP0/C2 diagnostic stopped with raw FAIL/exit 137 after
480 seconds plus stop grace. It completed 27 steps and remains unchanged.
The last geometry was `[0,1,4096]`, one decode query plus 4,095 prefill queries.
This is distinct from a batched pure-Q1 workload.

A new bounded baseline capture reproduces the first such mixed step at actual
positions `[65470,8258]`: 12.584 seconds overall, with 9.087 seconds of tiled
prefill-attention kernel time and 0.302 seconds of convolution kernel time.
All 16 attention layers select the native prefill route because oneDNN rejects
query rows below two. All 48 convolution calls use the multi-sequence fallback;
that is a secondary cost here. Host copy waits overlap GPU completion and are
not independent compute costs.

A diagnostic oneDNN prototype changes only the minimum query rows from two to
one. Across all 16 real attention calls, identical query/KV/page/length/offset
operands give 11.022 seconds total baseline complete-call time versus 0.418
seconds for the prototype, including its cold partition work. Baseline outputs
are restored before the model consumes them. All outputs are finite; maximum
relative L2 is 0.00007751 and maximum absolute difference is 0.03125. The
unchanged matrix checks (relative L2 <=0.001, maximum absolute <=3e-6 plus
0.002 times reference peak) pass. These are scoped same-input operator controls,
not paired serving throughput or whole-step speedup claims.

## Focused checks

The new detecting test first fails at the three missing mixed-route admissions.
The final GPU test passes 1 case / 113 assertions: default and explicit-off
fallbacks, enabled B2/B3/B4 mixed packets, empty rows, heterogeneous exact
lengths, strided page tables, output aliasing, immutable KV/metadata, pure-Q1
fallback and invalid flag spellings. Existing graph-policy checks pass 2 cases /
109 assertions including the new control. No full suite was run.

With the opt-in route, a fresh original MTP0/C2 run completes both requests:

- Wall time including engine startup: 101.980 seconds, exit 0.
- Predeclared limit: 180 seconds plus 15 seconds shutdown; at most 100 steps.
- Actual steps: 64; 32 output tokens per request, finish reason length.
- Actually computed prompt work: 65,469 tokens per request.
- Final consumed positions: 65,500 and 65,500.
- Profiler settings remain off; original greedy seed 770000 and original IDs.

This establishes bounded completion for this opt-in case. It does not establish
universal MTP0 liveness, MTP3 throughput gain or strict semantic/format quality.
The default still uses the reviewed route and retains the historical limitation.

## Identity and disposition

Candidate server SHA-256: `fd66ae79e967d83e9bb581ff72298089dc2c65dee2cd6453cba3bd5085f03a93`.
Product library SHA-256: `c5e040d31de3398181575cb953cc6e87ddb686122f759dbad6e424e58777eb68`.
Completion harness SHA-256: `afc88ded4106d2a95d1d4dd6df1ab380c89265a45bc7b36f9d7fa4caf5775ec9`.

Raw build commands, source hashes, RED/GREEN logs, same-input operator controls,
progress rows and terminal receipts are retained in the supplied local evidence.
The known strict code-fence failure remains separate. Disposition: retained
experimental, default disabled pending the combined promotion decision.
