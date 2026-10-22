# The chunked prefill path is not reproducible run to run, so its flip set is not a precision signature (2026-10-22)

Row MODEL-TEXT-kolibri-1-tenstorrent (spec `.agents/specs/kolibri-tt.md`).
Issues: ISSUE-LOCAL-01M5HHEV4DWX0D4S5E955VE0AN (the null arm this file
measures) and ISSUE-LOCAL-01M54CKCJNWHRDJB7PYRG31Y1D (the chunked-prefill
record this file corrects). Branch `row/TT-CHUNKED-NULL-ARM`, base
`origin/main` 845b3c59d. Device: one P150 (Blackhole). No product code
changed. This file is the record, and the code change is the null arm the
`SCRATCH dbg chunk sweep` case now carries in
`tests/vllm/models/test_kolibri1_tt_b2ii.cpp`.

## 1. The question

The chunked-prefill record states a root cause: "inherent multi-token
prefill kernel precision, NOT chunk logic". The evidence it cites is a
chunk-size sweep that compares the CHUNKED path with the SERIAL path. That
comparison had no same-mode null arm. Nobody measured what the chunked path
does to ITSELF, twice.

This file measures that null arm, localises the divergence with the
per-layer taps, and states what the measurement settles and what it does
not.

## 2. Provenance

| item | value |
|---|---|
| device | P150 (Blackhole), one card |
| base commit of the measurement | `origin/main` 3ce14f35c |
| build directory | `/tmp/build-tt-taps` |
| harness | `tests/vllm/models/test_kolibri1_tt_taps.cpp` |
| prompt | 6 teacher-forced tokens: 5249, 25079, 74493, 15774, 121598, 116026 |
| model | `/mnt/models/Aleph-Alpha/Kolibri-1` |
| tt-metal runtime root | `TT_METAL_RUNTIME_ROOT=/tmp/umdtrial-install/libexec/tt-metalium` |
| tt-metal source pin | `TT_METAL_HOME=/home/lu_zero/Sources/tt/tt-metal-pin` |
| CPU pinning | `taskset -c 32-63` |
| raw run output | `/tmp/tt-reps/run5.log` |
| device reset output | `/tmp/tt-reps/smi-reset.log` |

The run log records four arms in this order.

| arm | mode | repeats | device reset | window |
|---|---|---|---|---|
| `p1-cold` | chunked | 3 | yes | 15:54:13 to 15:56:36 |
| `p1-serial` | serial | 2 | yes | 15:56:36 to 15:58:48 |
| `p2-warm` | chunked | 3 | no | 15:58:48 to 15:59:46 |
| `p3-warm` | chunked | 3 | no | 15:59:46 to 16:00:42 |

All four arms ran on 2026-10-21 with the governor in the performance state.

## 3. The null arm: the raw output

### 3.1 Chunked, three repeats in one process

The same forward run again with the KV cache zeroed between the repeats.
`p1-cold` and `p2-warm` produced the same three readings, so the device
reset and the kernel-cache clear changed nothing.

```text
KOLIBRI1_TT_TAPS first_token_id=42046 top5=42046(6.930672), 15774(6.102934), 25079(5.940235), 74493(5.928247), 32(5.635765) mode=chunked rep=0
KOLIBRI1_TT_TAPS first_token_id=42046 top5=42046(7.048378), 25079(6.062962), 15774(6.035669), 74493(5.998163), 121598(5.658829) mode=chunked rep=1
KOLIBRI1_TT_TAPS first_token_id=42046 top5=42046(6.747927), 15774(5.883314), 74493(5.759626), 25079(5.745075), 32(5.410065) mode=chunked rep=2
```

The argmax is unchanged and the top-1 logit moves by about 0.3 nats.

### 3.2 Chunked, a second process in a different device state

`p3-warm`, no reset and no kernel-cache clear.

```text
KOLIBRI1_TT_TAPS first_token_id=42046 top5=42046(3.597120), 74493(3.498388), 25079(3.216179), 32(3.203287), 46(3.193529) mode=chunked rep=0
KOLIBRI1_TT_TAPS first_token_id=42046 top5=42046(6.964228), 25079(5.862806), 15774(5.855246), 74493(5.810308), 32(5.424648) mode=chunked rep=1
KOLIBRI1_TT_TAPS first_token_id=42046 top5=42046(6.917763), 15774(6.070782), 25079(5.992756), 74493(5.914759), 32(5.606716) mode=chunked rep=2
```

The first repeat reads 3.4 nats away from the two that follow it, so the
chunked path's own spread is 3.4 nats across device states.

### 3.3 Serial, two repeats

```text
KOLIBRI1_TT_TAPS first_token_id=74493 top5=74493(8.055458), 15774(8.038990), 121598(6.574747), 25079(5.707886), 34(5.195730) mode=serial rep=0
KOLIBRI1_TT_TAPS first_token_id=74493 top5=74493(8.055458), 15774(8.038990), 121598(6.574747), 25079(5.707886), 34(5.195730) mode=serial rep=1
```

The serial walk is byte identical.

### 3.4 The device reset and the kernel-cache clear

`/tmp/tt-reps/smi-reset.log` records the reset of the UMD logical ids. The
`p1-cold` arm, which ran after that reset, and the `p2-warm` arm, which ran
without one, produced the same three readings. The spread is therefore not a
cold or warm artefact.

## 4. The tap artifacts, labelled by mode

The per-layer taps hold one row per layer, per tap point, per query row, per
hidden element. The shape is `layer`, tap name, query row, hidden index,
value. The dumps carry the query rows 0 and 5.

| artifact | mode | reproducible | md5 |
|---|---|---|---|
| `/tmp/det-chunk-c1-taps.tsv` | CHUNKED prefill, run 1 | NO | `8fc2d866d5439aad72b42dbf6e208b2a` |
| `/tmp/det-chunk-c2-taps.tsv` | CHUNKED prefill, run 2 | NO | `0e9515d2ae37ec7c715042354b32caca` |
| `/tmp/det-serial-s1-taps.tsv` | SERIAL walk, run 1 | YES | `5ae26be91df198a74725ebf893f06138` |
| `/tmp/det-serial-s2-taps.tsv` | SERIAL walk, run 2 | YES | `5ae26be91df198a74725ebf893f06138`, byte identical to run 1 |
| `/tmp/kolibri1-layerdump-vllmcpp.hid.tsv` | CPU reference dump, not a device walk | n/a | `0ed9733915d84db1014e996f3eaafbe5` |

The router selection per layer rides beside the taps.

| artifact | mode | reproducible | md5 |
|---|---|---|---|
| `/tmp/det-chunk-c1-ids.tsv` | CHUNKED prefill, run 1 | NO | `98fa0312fef419816a2adf20aec2cf8f` |
| `/tmp/det-chunk-c2-ids.tsv` | CHUNKED prefill, run 2 | NO | `a904346b51857c9f9efe8756cfd4f551` |
| `/tmp/det-serial-s1-ids.tsv` | SERIAL walk, run 1 | YES | `25016de5eb871109d824587468683db2` |
| `/tmp/det-serial-s2-ids.tsv` | SERIAL walk, run 2 | YES | `25016de5eb871109d824587468683db2`, byte identical to run 1 |

The path `tests/vllm/models/kolibri1_layerdump_vllmcpp.hid.tsv` is NOT in the
tree. The file with that shape on this box is the `/tmp` path in the table
above, and it is a CPU reference dump, not a device walk. No artifact in
this repository carries a layer dump today.

## 5. The per-layer localisation

Two CHUNKED runs against each other (the null arm), and one CHUNKED run
against the SERIAL walk, on the same taps. `rms(d)` is the root mean square
of the element differences, and the row `rms(d) over the differing elements`
counts only the elements that differ.

| tap point | comparison | rms(d) over all 2560 elements | rms(d) over the differing elements | max &#124;d&#124; | elements differing |
|---|---|---|---|---|---|
| layer 0, `attn_out_pre_norm`, row 0 | chunk run 1 vs run 2 | 0 | 0 | 0 | 0 of 2560 |
| layer 0, `attn_out_pre_norm`, row 5 | chunk run 1 vs run 2 | 4.95e-05 | 6.96e-05 | 9.77e-04 | 1296 of 2560 |
| layer 0, `attn_out_pre_norm`, row 5 | chunk vs serial | 4.74e-03 | 4.75e-03 | 4.35e-02 | 2554 of 2560 |
| layer 0, `l_out`, row 5 | chunk run 1 vs run 2 | 1.15e-03 | 1.35e-03 | 1.56e-02 | 1871 of 2560 |
| layer 49, `l_out`, row 5 | chunk run 1 vs run 2 | 3.24 | 3.24 | 21.6 | 2556 of 2560 |
| layer 49, `l_out`, row 5 | chunk vs serial | 16.41 | 16.41 | 272 | 2559 of 2560 |

The divergence is born at layer 0's LAST query row, `attn_out_pre_norm` row
5. The maximum difference there, 9.77e-04, is one bf16 ULP at that exponent,
and about half of the elements differ by 1 to 2 ULP. The query row 0 of
layer 0 is bit identical at every tap point, so the difference starts at the
last row and not before it. The dumps carry rows 0 and 5 only, so they carry
no reading for rows 1 to 4.

The divergence grows with depth. The `l_out` rms(d) runs from 1.35e-03 at
layer 0 to 3.24 at layer 49, with a maximum element difference of 21.6 at
layer 49.

## 6. The chunk-vs-serial difference against the path's own spread

The measurement of record puts the chunk-vs-serial difference at 5x to 600x
the run-to-run spread at every layer, with the two anchors at layer 49
`l_out` 16.41 against 3.24, a 5.06x ratio, and layer 0 `attn_out_pre_norm`
2.17e-02 against 6.96e-05, a 312x ratio.

The reproduction check of section 8 recomputes the ratio from the tap
artifacts of section 4 and bounds the per-layer largest ratio at 5x to 123x.
The direction of the ratio does not change: the chunk-vs-serial difference
is far larger than the path's own spread at every layer.

## 7. The router order flip

The router's expert ORDER flips in 36 of 50 layers between two CHUNKED runs
of the same path. At layer 0 the two runs read:

```text
run 1: 359,197,246,203,11,68
run 2: 359,246,197,203,11,68
```

Ranks 2 and 3 are swapped and the six experts are the same. Of the 36 layers
whose order changes, 17 keep the same six experts in a different order and 19
change the expert set. The SERIAL walk's two runs read the same 6 experts
per layer in the same order, at every one of the 50 layers.

Against the SERIAL walk the CHUNKED path differs in the expert set at all 50
layers, for example at layer 0 `359,197,246,203,11,68` against
`359,203,11,68,197,151`.

## 8. Reproduction check on 2026-10-22

The numbers of sections 5 to 7 were recomputed on 2026-10-22 from the tap
artifacts of section 4, on this host, with no device and no model run. The
method is the plain one: for every `(layer, tap name, query row)` key, take
the element-wise difference of the two files and take the root mean square of
it, once over all 2560 elements and once over the elements that differ.

What reproduces:

- the run-to-run values of section 5, including 6.96e-05 and the maximum
  9.77e-04 at layer 0 `attn_out_pre_norm` row 5, and 1.35e-03 at layer 0
  `l_out` row 5 and 3.24 at layer 49 `l_out` row 5.
- the chunk-vs-serial value 16.41 at layer 49 `l_out` row 5.
- the per-layer largest ratio, at 5x to 123x, taken as the largest ratio
  of the chunk-vs-serial rms(d) to the run-to-run rms(d) inside the layer,
  and at 2.9x to 77.5x when both rms(d) values pool every tap of the
  layer.
- the router order flip in 36 of 50 layers, and the byte identity of the two
  SERIAL runs.

What does not reproduce:

- the 600x upper bound of the ratio. The largest ratio the artifacts carry
  is 123x.
- the layer 0 `attn_out_pre_norm` chunk-vs-serial anchor 2.17e-02. The
  artifacts carry 4.74e-03, a 68x ratio against the run-to-run 6.96e-05.

Both unreproduced values are owed to be reconciled. Neither changes the
direction of the ratio, and neither changes what this file settles.

## 9. WHAT THIS DOES NOT SHOW

- It does not identify the mechanism. The taps show where the divergence is
  born and how it grows with depth. They do not show why it happens. No
  record may cite a mechanism from this file.
- It does not show that the serial path is affected. The serial walk is byte
  identical across two repeats, and the serial walk is the reference of every
  comparison in this file.
- It does not re-measure the chunk sweep. The sweep's own numbers were not
  re-run. This file adds the null arm that the sweep lacked, and the sweep
  stays owed.
- It does not show that the chunked path's nondeterminism accounts for the
  whole chunk-vs-serial difference. That difference is larger than the
  path's own spread at every layer, so something in the chunked path differs
  from the serial path. What that something is stays open.

## 10. What is owed

- The chunked prefill path's reproducibility. Its run-to-run spread must sit
  far below the chunk-vs-serial difference before any chunked comparison can
  carry a reading.
- A re-run of the chunk sweep with the null arm the instrument now carries.
  The `SCRATCH dbg chunk sweep` case runs the same chunked walk twice per
  chunk size and prints the chunked-versus-chunked difference, and its ratio
  to the chunked-versus-serial difference, beside the chunked-versus-serial
  one.
- The mechanism hunt, and the reconciliation of the two unreproduced values
  of section 8.
