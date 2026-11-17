# B70 EXL3 serving continuity

This is a bounded integration check on 2026-10-06, not a new Python speed gate.
The native development control remains at `4a5485c1297f06482b1d5b41946ed1997b0f4d4f`.
The contribution starts from main `0fbd7c994fd40f4e62f9a75ad135146b00f0952c`;
it imports selected product changes without that development ancestry.

## Recipe and method

One Arc Pro B70, unchanged 180 W, `icpx` 2026.1.1, IGC 2.41.5,
driver 1.17.39758+10, oneDNN 3.13.0 and the source pins in
[EXL3_XPU.md](../EXL3_XPU.md). Checkpoint revision
`113cf7ab958054860e43fb7f3063b1af19171095` has a 4 bpw body and 6 bpw full head.
The matching 65536-token draft map, FP16 activations, FP32 recurrence and FP8 KV
remain unchanged. Each arm runs alone; model and receipt paths are explicit inputs.

The sentinel is in-process C1, P4096/O256, MTP3, greedy with `ignore_eos=true`,
page 1600, batch budget 1600, four admitted slots and 180 blocks. Each fresh process
first warms a 32-output request, then resets prefix state before the scored request.
Instrumentation is OFF. The same frozen 4096 IDs are supplied to both binaries.
Request timing starts at admission; TTFT includes prefill and first emission.
TPOT is `(last emission - first emission) / 255`, using emitted-token counts.
MTP emits chunks, so timestamps are chunk observations rather than 256 independent
per-token clocks. The client never counts one token per SSE frame.

| Native arm | Request wall | TTFT | Mean TPOT |
|---|---:|---:|---:|
| Unchanged development control | 8.1719 s | 2293.26 ms | 23.0534 ms |
| Source-built contribution | 8.1692 s | 2285.91 ms | 23.0717 ms |

All 256 output IDs and 113 non-timing scheduler/verification cycles match exactly.
The one-pair wall difference is -0.033%; this does not establish a speed gain.
The existing short lifecycle result JSON also matches exactly, with 483 assertions
per arm covering turnover, cancellation, poisoned spares and graph/eager/graph.
The 32K/O64 native prefix pair starts cold at 0 and warm at 30400, with identical IDs.
Native sentinel peak tracked device allocation is 30307785027 bytes; graph bytes
return to zero after release. Captured cgroups report no OOM-kill increment or swap.

## Public server and distinct limits

The actual `examples/vllm-server` passes bounded target-only and MTP3 HTTP checks
for text/usage, template, streaming, C1–C4 client requests, cancellation/EOS and
successful follow-up requests. HTTP alone does not prove GPU overlap or physical
slot identity. Draft-token counters advance in MTP; target-only requires no
advance, allowing registered constant counters. Exact draft depth is launch evidence.
These short HTTP checks are functional evidence, not a serving-throughput matrix.
The public client and launch recipe are in [EXL3_XPU.md](../EXL3_XPU.md).

An aligned 4K HTTP diagnostic restores 1600 tokens through the existing snapshot
trace and reproduces its cold text. Its strict cached-token metric check fails:
`prompt_tokens_cached_total` stays zero. Streaming logprobs and complete MTP
logprob positions are also unqualified. Keep transport observability separate
from the native position/ID checks.

The earlier 2026-10-06 sampled in-process matrix used batch 4096, 20 scenarios,
70 waves and 124 requests with O1024, compared with the published Python EXL3
recipe from 2026-10-02. Historical C1 decode gaps were roughly 17–22% through 32K
and 29–35% at longer prompts; C4 aggregate gaps were 33–36%. No matrix was rerun
for this composition, and those figures are not current-pin or HTTP speed parity.
Native admission was four, while the attempted 16-slot recipe hung or failed.
The longest measured input was 199673+1024; configured 262144 is not maximum-boundary
qualification. The batch 4096 16K/64K prefix resend limitation remains open.

Default D27/D29 TV/KL and integrated MTP/C4 state gates remain failed. A controlled
arithmetic reference is a separate label. The ordinary vLLM pin remains
`a7c23ac96d7806e7c7e7d862eadbce5a33529b94`; its inspected stock XPU platform does
not register EXL3. No runtime qualification against that pin was run here.
Independent static/mutation review is pending. This checkpoint supports experimental
review, not a claim that the backend is fully reference-qualified or merge-ready.

## Publication follow-up on 2026-10-07

The unpublished contribution now uses main
`452617154b8231a0fea4b5331b246427d96622b8`, preserving the upstream TT slot-churn
repair. Tested code head: `11cb381f61b37c9aba88ca63d8f60180e4cf5ef2`.
CPU/XPU rebuilds and corrected target/MTP3 HTTP checks pass. Required scheduler
gauges are present, finite, nonnegative and observed drained under a bounded
deadline. The compact map is included with its exact byte hash and MIT provenance.

The fresh unchanged-control/contribution sentinel matches256 IDs and113 non-timing
cycles, with610 assertions per arm. The new native lifecycle matches the unchanged
historical control JSON, with483 assertions. These checks retain their original
workload and tolerances. Timing is a single continuity observation; no performance
matrix or reference qualification was repeated. The original table above retains
its historical identity.

The [public evidence appendix](b70-exl3-publication-evidence/README.md) contains
current commands, exits, JSON, binary hashes and separately labelled historical
failure excerpts. The full preflight was not repeated after this rebase. All
remaining red/PENDING qualification gates stay open.
