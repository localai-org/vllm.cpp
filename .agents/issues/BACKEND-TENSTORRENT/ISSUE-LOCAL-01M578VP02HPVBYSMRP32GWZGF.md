ID: ISSUE-LOCAL-01M578VP02HPVBYSMRP32GWZGF
Title: kolibri1 P150: tt-metal rms_norm shows a 0.7% input-dependent deviation on real in-model rows — kernel precision, not call-site config
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-18
Updated: 2026-10-22
Closed: -

## Problem

The tt-metal rms_norm arm (ttnn::rms_norm over f32 tiles, patched fp32_acc=true/approx_mode=false in the pinned fork's layernorm_default_compute_config) deviates ~0.7% from the f64 oracle on kolibri's REAL in-model activation distributions while reading clean (0.9997) on the micro's random N(0,0.5) inputs. DIAGNOSTIC RESULT (2026-10-18, P150): the standalone micro (test_kolibri1_tt_b2ii 'SCRATCH dbg rmsnorm micro', VT_TT_RMSNORM_MICRO_INPUT) replaying #3466-era captured rows (VT_TT_NORM_DUMP_DIR, /tmp/normdump) REPRODUCES the deviation — it is input-dependent kernel precision, not a call-site config difference. Numbers (tt sum / f64-oracle sum): the d=2560 sandwich-norm rows with large magnitudes (max|x| 3.55 and 5.5) read 1.0072 and 1.0077 (+0.7%); the d=128 per-head q/k rows read 0.9984 (clean). The same real rows also show the CPU arm further from the oracle than the TT arm (cpu/host2 1.0156 and 1.0130), so the tt-metal kernel is not the worst mover on these inputs; the composed f32 arm exists for A/B (VT_TT_RMSNORM_COMPOSED=1) but the model's tensor geometry refuses it on this stack (binary_ng rejects the subtile broadcast). FIX DIRECTION: inside the pinned tt-metal RmsNormKernel (layernorm.cpp compute kernel): the sum-of-squares already reduces in f32 when fp32_dest_acc_en is set (float32_reduction = fp32_dest_acc_en && !legacy_reduction), so the remaining suspects are the SFPU rsqrt approximation (LEGACY_RSQRT path) and the fp16-b intermediate data formats — both require a tt-metal-side change, beyond a vllm.cpp-level fix.

## Resolution

2026-10-21: STILL TRUE, and the mechanism is sharper than the 2026-10-19
update below states. The deviation is a property of the tt-metal rms_norm arm
on REAL in-model rows. The 0.9809 reading that the 2026-10-19 update
attributed to upstream's rsqrt rework is in fact the substrate's DEFAULT
compute configuration (`fp32_acc=false`), and upstream main reads the same
value for the same configuration, so upstream changed no measured value: see
the 2026-10-21 update at the end of this record.

What settles it: the re-check trigger this record names -- SFPI >= 7.89, or an
upstream rms reduce and round rework -- followed by the micro replay on the
captured in-model rows. The substrate characterisation and the local patch
audit continue under `ISSUE-LOCAL-01M5B60NEFSEHYZXMR6FSS1RAF` in the
`MODEL-TEXT-kolibri-1-tenstorrent` row.

This record also absorbs the duplicate that closed on 2026-10-21,
`ISSUE-LOCAL-01M54CKBVMKZEGP8ZWM3G0CCT6` (the same substrate defect, opened the
day before). Its unique evidence is preserved in that record's closure -- the
pre-scale fix is falsified, because scaling probes x0.25/x1/x4 leave the
tt/host-double sum ratio unchanged (0.995/0.992), so the bias is
scale-invariant; 74% of the elements match the host-double oracle exactly and
the differing ~26% are net-low 1-3 ULP artifacts.

## Update 2026-10-19: probed tt-metal upstream head (2500c1d8f63)

The bias is NOT improved upstream and regressed on the largest-magnitude captured row: tt/host replay 1.0072 -> 0.9809 (normrow-000, 1x2560, max|x|=3.55, max_abs 0.4375), 1.0077 -> 1.0033 (normrow-004), 0.9984 -> 0.9970 (normrow-001). Upstream switched rsqrt_tile<LEGACY_RSQRT> to ckl::Rsqrt<Approx::Exact> in a new eltwise_chain — did not fix the input-dependence. Full details: ISSUE-LOCAL-01M5B60NEFSEHYZXMR6FSS1RAF (includes the SFPI 7.88 rvtt_live ICE blocker and the still-needed local patch audit). Re-check trigger: SFPI >= 7.89 or an upstream rms reduce/round rework.

## Update 2026-10-21: the 0.9809 reading is the compute config, not an upstream change

The 2026-10-19 update above attributes the 0.9809 reading to upstream's rsqrt rework. A re-measurement on 2026-10-21 (P150, docs/bench-evidence/tt-rms-norm-compute-config-20261021.md) refutes that attribution and keeps the defect open with a sharper mechanism.

The two recorded replays ran two different tt-metal substrates, and the number follows the substrate's DEFAULT rms_norm compute configuration, not the tt-metal version. A standalone ttnn probe replays the same captured rows with an explicitly chosen configuration and reads 0.980947 with fp32_acc=false and 1.007197 with fp32_acc=true and approx_mode=false, on the pinned substrate and on upstream main alike. Upstream main and the pinned fork give the same number for the same configuration, so the upstream rsqrt rework changed no measured value.

The pinned 1.0072 reading is the patched-substrate regime. The 0.9809 reading is the unpatched regime, which the trial substrate of 2026-10-09 also had before the local precision edit was re-installed on 2026-10-10 17:36. The vllm.cpp rms path is byte-identical across the two sessions and passes no compute configuration, so no caller-side parameter changed.

The claim that the deviation is a property of the tt-metal kernel stands. The claim that upstream regressed does not. The 0.9984 figure quoted above for the d=128 rows is normrow-002, not normrow-001 (normrow-001 reads 0.995782 on the pin). The substrate's local precision edit is not in the recorded pin, which is filed as ISSUE-LOCAL-01M5EM67GWF0D79SSY3NHZRKQA.

## Update 2026-10-22: at layer 0 and at positions 0 and 5, the device path sits about 3.4x further from our CPU path than our CPU path sits from llama.cpp

This entry sizes the compute-config effect at layer 0, at positions 0 and 5. It does not change the mechanism this record already names, and it does not touch the band gate, which lives in another record.

### The artifacts

A peer agent from mistral.rs measured the same model on the same layer-0 taps. The peer contributed one distance row, the selection geometry, and the width reading. The device side is `/tmp/kolibri1-tt-taps-serial.tsv`, the Tenstorrent SERIAL walk, which is the device's production path, captured on the P150 on 2026-10-21 in the same run as the chunked-prefill null arm. The CPU side is `/tmp/kolibri1-layerdump-vllmcpp.hid.tsv`, our CPU dump and the artifact handed to the peer. The llama.cpp side is `/tmp/opencode/llamacpp-lout.tsv`, the llama.cpp dump in our vocabulary.

The device file and our CPU dump carry the CPU dump's own format, `layer<TAB>quantity<TAB>position<TAB>hidden_index<TAB>value`, and both hold 1,024,000 rows each: 50 layers, 4 quantities, 2 positions (0 and 5), and 2560 hidden indices. The two files differ in their values, so the comparison below compares two runs and not one file with itself.

The llama.cpp dump carries the same format, the same 50 layers, and the same 2560 hidden indices. It holds the `l_out` quantity only and 6 positions (0 to 5), and it holds 768,000 rows and 20,925,916 bytes. The comparison below uses positions 0 and 5, which both sides carry.

### The like-for-like comparison at layer 0

Every row below carries the same quantity on both sides, `l_out`, and the formula is the one this record already states, `rel/cpu = rms(a - b) / rms(cpu)`. The rows pair by the key `layer`, `quantity`, `position`, `hidden_index` and not by row index, because the files hold their rows in a different order.

| layer 0 quantity | position | pair | rel/our CPU path | rel/the other stream | measured by |
|---|---|---|---|---|---|
| `l_out` | 0 | device vs our CPU path | 6.7138e-03 | 6.7148e-03 | the coordinator |
| `l_out` | 5 | device vs our CPU path | 4.8405e-03 | 4.8432e-03 | the coordinator |
| `l_out` | 0 | llama.cpp vs our CPU path | 2.0016e-03 | 2.0019e-03 | the coordinator |
| `l_out` | 5 | llama.cpp vs our CPU path | 1.4344e-03 | 1.4345e-03 | the coordinator |

The third column divides by our own CPU stream in every row, and the fourth column divides by the other stream of the pair. The two llama.cpp rows are new in this entry. The coordinator computed them from `/tmp/opencode/llamacpp-lout.tsv` and `/tmp/kolibri1-layerdump-vllmcpp.hid.tsv`. The peer's own llama.cpp dump carries all four quantities and holds the same values for `l_out`, so it reads the same two rows.

At layer 0's `l_out` the device path sits about 3.4x further from our CPU path than our CPU path sits from llama.cpp. The ratio is 3.35x at position 0 and 3.37x at position 5, whichever column supplies the llama.cpp row.

### The device against our CPU path at every quantity, at layer 0

The coordinator recomputed the device-against-CPU rows independently from the device file and our CPU dump, with `rel/cpu` = rms(a-b)/rms(cpu) and `rel/dev` = rms(a-b)/rms(device), where `a` is the device value and `b` the CPU value at the same layer, quantity, position, and hidden index.

| layer 0 quantity | position | rel/cpu | rel/dev |
|---|---|---|---|
| `attn_out_pre_norm` | 0 | 1.2249e-02 | 1.2300e-02 |
| `attn_out_pre_norm` | 5 | 1.1331e-02 | 1.1366e-02 |
| `attn_post_norm` | 0 | 1.1137e-02 | 1.1154e-02 |
| `attn_post_norm` | 5 | 1.2000e-02 | 1.2045e-02 |
| `ffn_post_norm` | 0 | 7.8719e-03 | 7.8772e-03 |
| `ffn_post_norm` | 5 | 9.3196e-03 | 9.3273e-03 |
| `l_out` | 0 | 6.7138e-03 | 6.7148e-03 |
| `l_out` | 5 | 4.8405e-03 | 4.8432e-03 |

The table holds layer 0 only, at positions 0 and 5. The gap holds the same order at every quantity and at both positions. It is not one norm that behaves differently, and it does not close by the end of the layer.

### The peer's distance, attributed

At layer 0, `attn_out_pre_norm`, position 0, the peer measured 1.66e-03 between llama.cpp and our CPU path. The coordinator recomputed that row from the peer's llama.cpp dump and our CPU dump and read 1.6569e-03, which reproduces the peer's figure and confirms that the row compares llama.cpp with our CPU path.

### The selection geometry, measured by the peer

At position 5, the peer compared the six experts that each implementation's router selects. Set agreement is the percentage of the six that the two implementations have in common. Exact set is the percentage of positions at which the two select the identical six.

| pair | set agreement | exact set |
|---|---|---|
| device vs our CPU path | 47.00 % | 4.00 % |
| device vs llama.cpp | 58.84 % | 4.08 % |
| device vs the transcription | 57.67 % | 4.00 % |
| llama.cpp vs our CPU path | 89.15 % | 49.49 % |
| llama.cpp vs the transcription | 89.94 % | 55.93 % |

The peer's table calls the vllm.cpp capture `the peer CPU path`, and the recomputation of the peer's distance row confirms that name. This record uses `our CPU path` for it. The transcription is the peer's label for a further selection reference. Its artifact is not in this tree, so this record carries the peer's number for it without an independent recomputation.

The device's selection is the outlier among the four implementations in this table. It selects the identical six experts as another implementation in 4.00 % to 4.08 % of the positions, while llama.cpp and our CPU path select the identical six in 49.49 % of the positions. The router flip is therefore a consequence of the divergence and not a knife-edge that better numerics would rescue.

### Why this is not a width effect

The peer read the gap as a compute-config effect rather than a width effect. The device computes its activations in f32, and llama.cpp and our CPU path keep the residual stream in f32 as well. The gap is therefore not a dtype-width difference. The candidate mechanism stays the one this record already names, the rms compute configuration, that is `approx_mode` and `fp32_acc`.

### The limitation on the peer's own capture

The peer's own layer dump is byte-identical to the two files that are named as our dump. `/tmp/opencode/kolibri1_layer_taps.tsv`, `/tmp/opencode/our_taps_runA.tsv`, and `/tmp/opencode/our_taps_runB.tsv` all carry the md5 `ee7ff773160d1c57b6a4e19d60f26e28`, and our own dump `/tmp/kolibri1-layerdump-vllmcpp.hid.tsv` carries `0ed9733915d84db1014e996f3eaafbe5`. The peer's 1.66e-03 row therefore does not stand as a distance between two CPU implementations, and this record does not use the peer's rows against its own capture.

### The band gate stays in its own record

This record is about the magnitude of the rms compute configuration's effect. The band gate's own verdict and its worst in-band divergence live in `ISSUE-LOCAL-01M4P21TZ1Q9CA7P66XBH0BQTX`. This record neither restates that verdict nor re-measures it.

### What this entry does not show

- It does not identify a mechanism beyond the candidate this record already names. It sizes the effect and it does not narrow the mechanism inside it.
- It does not measure the device's decode-time behaviour. The taps come from one teacher-forced forward.
- It does not re-measure the band gate.
- It does not measure any layer other than layer 0, and no position other than 0 and 5.
- It does not measure a token-level distance. The comparison holds on the layer taps only.

The evidence file with the provenance, the method, and the tables is `docs/bench-evidence/kolibri1-tt-device-vs-cpu-layer0-20261022.md`.
