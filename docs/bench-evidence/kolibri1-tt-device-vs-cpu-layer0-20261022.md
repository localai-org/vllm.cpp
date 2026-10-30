# Kolibri-1 on the P150: the device path against our CPU path and against llama.cpp at layer 0 (2026-10-22)

Row: BACKEND-TENSTORRENT. Issue
`ISSUE-LOCAL-01M578VP02HPVBYSMRP32GWZGF`, the rms compute-config record.
Branch `row/TT-DEVICE-VS-CPU-LAYER0`, base `845b3c59d` = `origin/main`.

This document is a measurement record. No vllm.cpp product code changed, no
model was run for it, and the Tenstorrent device was not touched. The device
numbers come from taps that a previous run already captured, and the CPU
numbers come from dumps that already existed.

## 1. The question

How far is the Tenstorrent device path from our CPU path at layer 0, at
positions 0 and 5, and how does that distance compare with the distance our
CPU path holds from llama.cpp on the same quantity?

The answer this document carries: at layer 0's `l_out`, the device path sits
about 3.4x further from our CPU path than our CPU path sits from llama.cpp.

## 2. The artifacts and their provenance

| side | artifact | rows | what it is |
|---|---|---|---|
| device | `/tmp/kolibri1-tt-taps-serial.tsv` | 1,024,000 | the Tenstorrent SERIAL walk, which is the device's production path, captured on the P150 on 2026-10-21 in the same run as the chunked-prefill null arm |
| our CPU path | `/tmp/kolibri1-layerdump-vllmcpp.hid.tsv` | 1,024,000 | our CPU dump. The artifact handed to the peer agent was a DIFFERENT capture of the same run (see the md5 evidence in section 9) |
| llama.cpp | `/tmp/opencode/llamacpp-lout.tsv` | 768,000 | the llama.cpp dump in our vocabulary, the `l_out` quantity for all 50 layers and all 6 positions, 20,925,916 bytes |

The device file and our CPU dump carry the CPU dump's own format,
`layer<TAB>quantity<TAB>position<TAB>hidden_index<TAB>value`. Both hold
1,024,000 rows each, and the rows pair one to one on the key `layer`,
`quantity`, `position`, `hidden_index`. The grid is 50 layers, 4 quantities
(`attn_out_pre_norm`, `attn_post_norm`, `ffn_post_norm`, `l_out`), 2 positions
(0 and 5), and 2560 hidden indices. The two files differ in their values, so
the comparison below compares two runs and not one file with itself.

The llama.cpp dump carries the same format, the same 50 layers, and the same
2560 hidden indices. It holds the `l_out` quantity only and 6 positions (0 to
5). The comparison below uses positions 0 and 5, which both sides carry.

All three artifacts live in the system temporary directory, outside this tree.
This document is the record.

## 3. The method

For one pair of implementations and one key, the relative rms of the
difference is

```text
rel(a, b) = rms(a - b) / rms(reference)
```

where `a` is the first value, `b` is the second value, and the reference is
the stream that divides. The coordinator's recomputation reports two
references for the device against CPU pair, so every row carries both:

```text
rel/cpu = rms(a - b) / rms(cpu)
rel/dev = rms(a - b) / rms(device)
```

The two columns differ only in the denominator, and they differ from each
other by 0.01 % to 0.42 % of the figure, which is the size of the difference
between the two streams' own scales.

The rows pair by the key `layer`, `quantity`, `position`, `hidden_index` and
not by row index, because the files hold their rows in a different order.

## 4. The like-for-like comparison at layer 0

Every row below carries the same quantity on both sides, `l_out`.

| layer 0 quantity | position | pair | rel/our CPU path | rel/the other stream | measured by |
|---|---|---|---|---|---|
| `l_out` | 0 | device vs our CPU path | 6.7138e-03 | 6.7148e-03 | the coordinator |
| `l_out` | 5 | device vs our CPU path | 4.8405e-03 | 4.8432e-03 | the coordinator |
| `l_out` | 0 | llama.cpp vs our CPU path | 2.0016e-03 | 2.0019e-03 | the coordinator |
| `l_out` | 5 | llama.cpp vs our CPU path | 1.4344e-03 | 1.4345e-03 | the coordinator |

The third column divides by our own CPU stream in every row, and the fourth
column divides by the other stream of the pair.

The two llama.cpp rows are new in this entry. The coordinator computed them
from `/tmp/opencode/llamacpp-lout.tsv` and
`/tmp/kolibri1-layerdump-vllmcpp.hid.tsv`, with the method of section 3 and
pairing by key. The peer's own llama.cpp dump,
`/tmp/opencode/llamacpp-taps/hid.tsv.0000`, carries all four quantities and
holds the same values for `l_out`, so it reads the same two rows.

At layer 0's `l_out` the device path sits about 3.4x further from our CPU
path than our CPU path sits from llama.cpp. The ratio is 3.35x at position 0
and 3.37x at position 5, whichever column supplies the llama.cpp row.

## 5. The device against our CPU path at every quantity, at layer 0

The coordinator recomputed the device against CPU row independently from the
two artifacts of section 2, with the method of section 3:

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

The table holds layer 0 only, at positions 0 and 5. The gap holds the same
order at every quantity and at both positions. It is not one norm that
behaves differently, and it does not close by the end of the layer.

## 6. The peer's distance, attributed

At layer 0, `attn_out_pre_norm`, position 0, the peer measured 1.66e-03
between llama.cpp and our CPU path. The coordinator recomputed that row from
the peer's llama.cpp dump and our CPU dump and read 1.6569e-03, which
reproduces the peer's figure and confirms that the row compares llama.cpp with
our CPU path.

## 7. The selection geometry, measured by the peer

At position 5, the peer compared the six experts that each implementation's
router selects. Set agreement is the percentage of the six that the two
implementations have in common. Exact set is the percentage of positions at
which the two select the identical six.

| pair | set agreement | exact set |
|---|---|---|
| device vs our CPU path | 47.00 % | 4.00 % |
| device vs llama.cpp | 58.84 % | 4.08 % |
| device vs the transcription | 57.67 % | 4.00 % |
| llama.cpp vs our CPU path | 89.15 % | 49.49 % |
| llama.cpp vs the transcription | 89.94 % | 55.93 % |

The peer's table calls the vllm.cpp capture `the peer CPU path`, and the
recomputation of the peer's distance row confirms that name. This document
uses `our CPU path` for it. The transcription is the peer's label for a
further selection reference. Its artifact is not in this tree, so this
document carries the peer's number for it without an independent
recomputation.

The device's selection is the outlier among the four implementations in this
table. It selects the identical six experts as another implementation in 4.00 %
to 4.08 % of the positions, while llama.cpp and our CPU path select the
identical six in 49.49 % of the positions. The router flip is therefore a
consequence of the divergence and not a knife-edge that better numerics would
rescue.

## 8. Why the gap is not a width effect

The peer read the gap as a compute-config effect rather than a width effect.
The device computes its activations in f32, and llama.cpp and our CPU path
keep the residual stream in f32 as well. The gap is therefore not a
dtype-width difference. The candidate mechanism stays the one the record
already names, the rms compute configuration, that is `approx_mode` and
`fp32_acc`.

## 9. What this does not show

- It does not identify a mechanism. The candidate is the one the record
  already names, the rms compute configuration. This measurement sizes the
  effect and it does not narrow the mechanism inside it.
- It does not measure the device's decode-time behaviour. The taps come from
  one teacher-forced forward.
- It does not re-measure the band gate. The band gate's own verdict and its
  worst in-band divergence live in
  `ISSUE-LOCAL-01M4P21TZ1Q9CA7P66XBH0BQTX`.
- It does not measure throughput, latency, or memory.
- It does not measure any layer other than layer 0, and no position other
  than 0 and 5.
- It does not measure a token-level distance. The comparison holds on the
  layer taps only.
- It does not carry the artifacts behind the peer's selection geometry, so
  those numbers are quoted here and cannot be recomputed from this tree.
- The peer's own layer dump is byte-identical to the two files that are named
  as our dump. `/tmp/opencode/kolibri1_layer_taps.tsv`,
  `/tmp/opencode/our_taps_runA.tsv`, and `/tmp/opencode/our_taps_runB.tsv` all
  carry the md5 `ee7ff773160d1c57b6a4e19d60f26e28`, and our own dump
  `/tmp/kolibri1-layerdump-vllmcpp.hid.tsv` carries
  `0ed9733915d84db1014e996f3eaafbe5`. The peer's 1.66e-03 row therefore does
  not stand as a distance between two CPU implementations, and this document
  does not use the peer's rows against its own capture.

## 10. The record this updates

The measurement is carried by
`.agents/issues/BACKEND-TENSTORRENT/ISSUE-LOCAL-01M578VP02HPVBYSMRP32GWZGF.md`
under its dated entry of 2026-10-22. The evidence file for the rms
compute-config question itself is
`docs/bench-evidence/tt-rms-norm-compute-config-20261021.md`.
