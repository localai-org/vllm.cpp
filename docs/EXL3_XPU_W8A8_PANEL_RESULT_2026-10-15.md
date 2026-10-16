# B70 follow-up: bounded W8A8 panel policy, 15 October 2026

This is the section 9.3 follow-up authorized after performance checkpoint
`48c3cfb76ecd13dbfb5655140b5620f7ff1ef495` was committed and pushed to Draft PR
3408. The supplied Pro plan has named Waves 0–4, not Waves 5–8. This experiment
uses the existing 1024/2048-column control; it introduces no product code,
precision, source-group, checkpoint, driver or library change.

## Hypothesis and focused checks

Larger bounded panels could reduce reconstruction/oneDNN submissions while
retaining checked preparation, source-group boundaries, FP16 Y, all scaling,
output Hadamard and completion-owned workspace lifetimes. Both widths run on
the same committed native binary with the prior selected improvements fixed.

An external adaptation of the existing isolated model-owned operator test runs
three original-reference fixtures in four fresh ABBA workers each. Every worker
uses the complete checked model operation, three warmups and twelve measured
calls. All twelve cases / 120 assertions pass, including exact output bytes and
accounted workspace/preparation bounds. These are independent fixtures, not
claims about all model matrices or arbitrary inputs.

| Original case | 1024 complete ms | 2048 complete ms | Change | Extra tracked peak bytes |
|---|---:|---:|---:|---:|
| M4096/K17408/N5120 down, one group | 3.036386 | 3.047624 | +0.37% | 17,825,792 |
| M4096/K5120/N16384 GDN, two groups | 3.199198 | 3.135496 | -1.99% | 5,242,880 |
| M1600/K5120/N34816 MLP, two groups | 3.356151 | 3.235651 | -3.59% | 5,242,880 |

The existing real wider-panel test additionally passes 1 case / 36 assertions
at M1600 for 128/1024/2048: quantization/scales, original FP16 Y, final output,
reconstructed tail and allocation guard remain exact. The 2048 primitive does
not buy speed by changing an arithmetic tolerance. Initial external build
attempts lacked the private include path and SYCL compile option; the corrected
build and all original failed receipts are retained separately.

## Actual full-model route and memory

Two separate fresh P4096/O64 diagnostics link the current native library. Their
64 raw output IDs match. Live oneDNN records prove N2048 dispatch and bounded
N1024 tails rather than inferring selection from an environment variable.

| Diagnostic, target-only prefill | 1024 | 2048 |
|---|---:|---:|
| Reconstruction calls | 3808 | 2048 |
| oneDNN calls | 3808 | 2048 |
| oneDNN stream-span sum ms | 777.320 | 774.630 |
| Weight reconstruction kernel sum ms | 61.866 | 56.327 |
| Preparation kernel sum ms | 102.631 | 103.068 |
| Output Hadamard kernel sum ms | 125.219 | 125.336 |
| Accounted shared W8A8 workspace bytes | 290,455,616 | 295,698,496 |
| Accounted preparation bytes | 71,319,552 | 71,319,552 |
| Accounted peak device bytes | 27,002,444,580 | 27,007,687,460 |
| Graph device bytes / count | 8192 / 2 | 8192 / 2 |

Observed descriptors retain `jit:gemm:any`, s8/s8, F16 destination, F32 weight
and row scales, and zero user scratch for these matrix shapes. The native and
pinned Python source use the same output-scale/post-op construction; Python
reconstructs full source groups whereas native retains bounded panels. No
oneDNN upgrade was attempted. The 46% submission-count reduction leaves its
summed region time nearly unchanged; it is not a 46% serving-speed improvement.

Instrumentation is separate from scoring. No nested graph span is added to its
kernels. Tracked peak increases by exactly 5MiB in this full-model diagnostic;
this is not a full-context capacity qualification. Driver free-byte readings
vary between workers and are not used to claim a VRAM reduction.

## Trace-free serving and decision

Three counterbalanced fresh-process pairs per mode preserve C1/P4096/O256,
greedy/ignore-EOS, prefix off, FP8 KV, all prior selected defaults and the same
two-CPU/24GiB/no-swap recipe. No profiler is present in scores. All 48 native
replies retain exact paired public token representations, text, usage and MTP
counters; raw prompt IDs agree. Each pair uses four requests and the warm
last-three median. These are independent blocks, not independent-token samples.

| Mode | Paired median prefill change | Prefill range across blocks | Paired median decode-duration change | Paired median total change | Total range |
|---|---:|---:|---:|---:|---:|
| MTP0 | -0.251% | -1.571% to +0.097% | +0.125% | +0.096% | -0.727% to +0.210% |
| MTP3 | -0.519% | -0.588% to -0.448% | -0.017% | -0.167% | -0.271% to -0.131% |

Decision: retain the 1024-column default. The small MTP3 benefit is consistent
in these three blocks, but target-only prefill and total changes reverse sign.
The target-only median total is slightly higher, within observed variation;
neither a repeatable target-only regression nor a serving win is established.
The 3% unaffected-duration limit is met, but that tolerance is not proof of a
gain. A roughly 0.17% MTP3 total improvement with 5MiB more tracked peak is not
a compelling general default change from this mixed result. The existing 2048
override remains available; no additional mode-specific policy was invented.

The prior selected code/defaults and their contemporary Python comparison stay
in force. No new Python or broad vision matrix was needed to evaluate an
unselected existing control. In particular this experiment does not close the
remaining prefill/decode gap or the deferred long natural-generation quality
check for the approved subgroup GDN path.

Commands, actual descriptors, exact fixtures and binary hashes, every sample,
failed setup and receipts are retained under
`pro-optimization-2026-10-15/wave4/panel2048` in the local recovery root.
