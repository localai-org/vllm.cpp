# B70 Pro-plan disposition — 2026-10-21

The bounded work packages are implemented or closed by measured disposition. The promoted production change is batched pure-Q1 attention for B2–B4, default 1 with explicit0 rollback. Request-stable auto alignment / long-Q1 mode2 remain the accepted baseline. Expanded SiLU 2 and mixed-prefill 1 are retained experimental; ordinary defaults remain SiLU 1/mixed-prefill 0. No merge, release or production restart is authorized by this document.

| Work package | Disposition | Evidence and practical result |
| --- | --- | --- |
| 0: reviewed auto/2 integration | Integrated | Existing PR3408 baseline e69198adc06401dac1d26411bc03fb0ebd1b1e7a, previous full comparison retained. The fresh post-promotion comparison is separate. |
| 1: cold/C4 and mixed-work attribution | Completed; mixed correction experimental | Steady/cold target/draft/EXL3/GDN/SiLU/attention profiling and source-correlated oneDNN attribution are retained. The mixed `[0,1,4096]` packet was rejected by a minimum-two-query oneDNN guard. Opt-in corrected MTP0/C2 completed 64 steps/101.980s, both 32-token requests finished, terminal positions 65,500/65,500. Original27-step 480+15s FAIL remains. |
| 2: small-row SiLU extension | Experimental, default 1 unchanged | Added exact mode2 for rows1/2/3/4/8/12/16 at width17408 with original FP16 intermediate rounding. Complete M16 call improved about 82–84%; target64-MLP GPU chain improved9.7/9.9%, roughly2.3ms. Six serving pairs do not reliably separate general gains from observed A/A variation. |
| 3: batched pure-Q1 attention | Promoted, default 1 | Exact B2–B4 offsets `[0,1,...,B]`; same bounded donor, scaled checked 16-MiB scratch. Complete-call improvements82.5–88.3%; C4-16K three-pair aggregate decode +6.074/+6.603/+6.925% (median+6.603%). C4-4K median+2.777% is below observed A/A variation. C1 sentinel effectively unchanged. |
| 4A: snapshot batching | Skipped from measured attribution | At most one publishing owner per step in original C4 traces; 96 arrays are not96 synchronizations. No speculative ownership/publication rewrite. |
| 4B: short repeated-prefix deployment choice | Existing explicit force retained | Previous descriptive short pair: auto cold1.762s / warm1.773s with0hits4091work; force cold1.925s / warm1.130s with1600hits2491work. No case-specific automatic threshold. |

The corrected liveness result records `last_positions` from `last_step.seq_lens`:65,500/65,500 are completed sequence lengths / exclusive request ends, not zero-based last-token indices.

## Source and executable identity

Product revisions: mixed-prefill `db6ef69e9e0fc609900869eb471bd16112d9114e`; SiLU `ec0875cc503231f4c76f98a0880d442cf5a2716a`; final Q1 `5fcad5dcab817a7462b5c884243c9b168d4b2843`. `source-review/manifest.json` exports78 exact committed source/test paths, four isolated diffs and the full-comparison executable binding. Build began at the prior recorded commit plus a working delta, verified content-equivalent to the committed product revision. It is not described as a clean-HEAD build.

| Measurement | Executable SHA-256 | Source binding / fallback |
| --- | --- | --- |
| Mixed completion | fd66ae79e967d83e9bb581ff72298089dc2c65dee2cd6453cba3bd5085f03a93 | phase-next-pro-mixed-candidate-v1 exact snapshots; mixed default 0 preserves fallback. |
| SiLU serving screen | 2e783827d43fb22fb0bee58e9f3dd099f39f544f9f034213ff3a575632f61b20 | phase-next-pro-silu-v1 exact snapshots; default 1 keeps original rows1/4. |
| Q1 same-binary paired screen | 0c821133e1f6e4a5548938b7ee4972473d7aeab462e552139abf7abfc59d56b8 | phase-next-pro-q1-candidate-v1/final-focused-identities.json, source-final; explicit0 versus1. |
| Promoted-default full comparison | c87739a6986b13de1240213ba0ed12075cea05b007ec6cc79c754e5042ed9f07 | phase-next-pro-q1-promoted-v1/identity.json; eight source/test hashes equal final Q1 revision. Only post-screen production change default 0→1; extra tests are test-only. |

Q1 retains all hardware/compiler/driver/dtype/layout/scale/causal/page guards and live metadata validation; B1, Q4, ragged and mixed routes remain. Last capacity, overflow fallback, NaN scratch/neighbor poison, short one-split contexts, live graph metadata, B4→3→2→1, actual owner replacement/cancellation and cross-queue lifetime have detecting coverage. Final promoted build: GPU 4 cases / 1880 assertions; policy 1 / 36; graph 3 / 250, all PASS. Original B1 retention1426 assertions PASS. Actual engine lifecycle46.200s/43steps: six finishes, cancelled request3 after54tokens without resurrection, actual B2/B3/B4 admissions4/6/62; all six finished greedy sequences match retained auto/2.

The unchanged operator gate is relative-L2≤0.001 and max-absolute≤3e−6+0.002×reference peak. Captured Q1 worst relative-L2 was0.000333713, max-absolute0.015625. Exact structural ownership/copy/lifetime checks are separate from allowed arithmetic variation. Stochastic C4 outputs and MTP acceptance can differ even between A/A workers.

## Serving screen and retained limits

`qualification/phase-next-pro-q1-serving-v1/analysis.json` and the corresponding SiLU analysis retain all 16 original worker receipts per screen, six neighboring alternating A/B pairs, A/A controls, one C1 sentinel, raw sample vectors, outputs, active environment, computed/cache work, full and overlap MTP counts, preemptions and host cgroups. No repeat-until-preferred-sign sampling occurred. Current Q1 C4-4K A/A decode differed−3.515%; previous comparable auto/2 A/A differed−4.835%. These are observed pairs, not confidence bounds. Every 16K Q1 gain exceeded both differences; median TTFT+0.013%, E2E−2.654%. The new cold work/caches are unchanged and every worker has zero preemptions.

The earlier auto/2-versus-off/1 16K C4 costs (median decode−2.546%, TTFT+5.017%) are accepted baseline configuration tradeoffs, not new Q1 regressions. Lost forced short-prefix hits under auto remain explicit.

Known strict Python-module closing-fence failure remains. Recovered code passes 32/32 functional checks; this does not turn strict requested format into PASS. Earlier extraction128 and arithmetic96 fixed cases are inherited quality evidence, not a newly universal quality qualification of the promoted build. The original raw FAIL / exit 137 last completed request positions were 65,479 (`c2-proof-3`) and 45,113 (`c2-proof-4`), not terminal finishes. The default64K MTP0/C2 completion remains unqualified; only the opt-in mixed-prefill correction completed its new bounded case. No full-matrix GPU allocator peak is claimed; host cgroup peak is not VRAM.

All28 C4 primary counter windows across both screens pass the unchanged strict
gates and final live-computed-prefill end levels. One Q1 C1-candidate auxiliary
aggregate-counter view fails occupancy/prefill/token-accounting checks. That bad
view is preserved and is not used: the predeclared sentinel primary is the
independently measured client post-first-token rate, checked exactly against its
raw request summary. This does not relabel the auxiliary view as PASS.

## Offline counter-window correction

The native worker completed all70 original waves/124 requests, but the additional
strict counter audit detected one C4-4K leading sample containing final prefill
publication. `qualify_counter_windows.py` applies the same deterministic final
committed-prefill plateau rule to every engine/block/wave, followed by the frozen
original occupancy, reset, integer-round and exact token-conservation gates.
Only leading pre-plateau samples are removed; the endpoint is unchanged and
never searched by throughput or residual. This is a disclosed post-capture
analysis correction, not a preregistered boundary rule or benchmark rerun.
The native case changes24.918729→24.667270s and141.018428→140.185761 tok/s,
with3,458 generated =2,134 accepted +3,972 drafted/3. Other69 native matrix windows
are unchanged. Original raw samples, driver fields, requests and receipts remain.
TTFT/E2E, C1 request-duration decode, work/caches and preemptions are unaffected.
The final activity plateau is tied to each wave’s actual final live-counter
value: native computed tokens, Python prompt tokens published on first prefill
output. Python actual-computed histograms publish only at request completion
and remain the independent prefill-throughput work metric. The initial attempt
to require that late histogram inside decode correctly fails; source-verified
live publication replaces only the activity guard, without relaxing the frozen
strict counter checks. Source files/hashes and the actual Python unchanged-case
replay are in prefill-counter-publication.json. The actual detecting replay passes; mismatched-draft and counter-reset controls
remain rejected. Full records are in counter-window-analysis.json after all four
workers complete; the native-only proof is already recorded separately.

## Fresh full comparison

All four fresh sequential workers passed the terminal audit: native and Python each completed70 original matrix waves/124 requests and three additional isolated64K prefix requests,254 measured requests total. All original payloads match, every response has1024 tokens/length finish, zero preemptions, unchanged committed/cached-token accounting and valid fully overlapping concurrency windows. Source/binary hashes are unchanged. Visual review is recorded separately.

| C4 input budget | C++ aggregate tok/s | Python aggregate tok/s | C++ relative throughput | C++ / Python TTFT seconds | C++ / Python E2E seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 149.81 | 184.43 | -18.8% | 4.162 / 2.375 | 31.878 / 25.442 |
| 4,096 | 141.05 | 186.74 | -24.5% | 6.422 / 4.846 | 38.913 / 28.913 |
| 16,384 | 129.26 | 161.46 | -19.9% | 26.612 / 19.906 | 71.407 / 54.724 |

C1 post-first-token decode and all original prefill/prefix rows are in report/index.html and comparison.json/CSV. Relative throughput is100×(C++/Python−1), not an extra-duration percentage. A fresh cross-engine measurement is not substituted for the separate causal Q1 same-binary screen.
