ID: ISSUE-LOCAL-01M3HB24YKDRXTXY85R4CG7VF6
Title: The capture-economics curve: TPOT vs output length on the serving arm
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-27
Updated: 2026-09-27
Closed: 2026-09-27

## Problem

The serving arm's TPOT at output_len 4 is ~34 s/token, cold-dominated: the per-REQUEST fixed costs (cold body ~33 s, capture pass ~5.7 s, trace-wait ~31 s) dwarf the per-TOKEN served-replay cost (12 ms wall). Whether this matters is a function of generation length: at output_len 256 the fixed costs amortize over ~251 served steps and TPOT should collapse -- IF nothing grows with length (GDN state windows, KV pages, per-step keepquant repair transients). The curve (TPOT at output_len 4/16/64/256, phase mix per leg, the fitted TPOT(len) = fixed/len + per_token model) is the evidence the next optimization targets hang from. Spec: tenstorrent-capture-economics.md.

## Resolution

The curve is measured and the model fits: docs/bench-evidence/tt-capture-economics-20260927.md. Legs 4/16/64 completed in one fresh process each (the 4-token leg reproduces #3327's 34,165.25 ms at +0.34% with the served stream sha256 byte-identical to the recorded anchor; the served-replay gate ran green on the fresh binary first, 2/2 cases 6/6 assertions). The discovery: the ~31.1 s caller window (gap_prev) before every served replay step is PER TOKEN, not the per-request trace-wait the len-4 decomposition read it as — at 16 and 64 it recurs metronome-flat (31,100-31,172 ms) before every replay of every request, while the served step's own wall stays 12 ms class and the retention read-out stays flat. TPOT(4)=34,280.9, TPOT(16)=31,754.1, TPOT(64)=31,283.2 ms; the fit TPOT(len)=fixed/(len-1)+per_token gives fixed=9,454 ms, per_token=31,129 ms with residuals +-5 ms (+-0.02%), per_token matching the measured caller window to 0.05%. Verdict: the per-token term dominates at every real length (98-99% past a handful of tokens); the capture-economics fixed pool (cold 34.4 s + capture 5.7 s) amortizes to 39% of TPOT at len 4, 8% at 16, 2% at 64. Per-token does not grow with length — it is constant; the growth-mechanism question resolves to a constant per-step cost the len-4 single-replay-per-request reading misattributed. Next targets: bracket the per-replay-step caller window (the #3322 ~30 ms/command trace-execution reading is the leading candidate, the deferred-reader flush the alternative), the trace per-command overhead itself (1,037 commands x ~30 ms for ~1.3 s of kernels — the capture buys ~nothing over eager), the cold+capture pass for the short arm, and the GIT_TAG main parakeet pin rot that blocked this row's fresh build. The 256-token confirmation leg launched detached at 18:27 (ETA ~9 h at the flat rate); the spec's design is 4/16/64 discover, 256 confirm — the evidence doc records it in flight and updates when it lands.
