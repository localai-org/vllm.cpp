ID: ISSUE-GH-1471
Title: **`vt::Conv2d` and `vt::DepthwiseConv1d` compute their output extents with C++ integer division, which truncates toward zero, where torch's shape contract FLOORS.** `src/vt/ops.cpp:2749-2750` (`hout`/`wout`) and `:2883` (`lout`). The two agree for a non-negative numerator and disagree for a negative one whenever `stride > 1`: at `in = 2, k = 3, stride = 2, pad = 0` the numerator is -1, so torch computes `floor(-1/2) + 1 = 0` and raises "Output size is too small" while truncation computes `-1/2 + 1 = 1` and ACCEPTS an extent of 1, convolving over taps the stride skipped. The `VT_CHECK(... > 0)` guard below each line cannot catch it, because truncation has already produced a positive extent. Found by the fresh review of [#1007](https://github.com/mudler/vllm.cpp/issues/1007) (finding F7) against the sibling `vt::Conv3d`, which carried the identical expression and is FIXED on that row: the span is separated from the division and a negative span is refused by name, proven by mutation — reverting the guard reds *conv3d: the shape contract refuses by name* at exit 1 (`CHECK_THROWS` did not throw, `tests/vt/test_ops_conv3d.cpp:446`), restoring it gives 4/4 and 2036 assertions, both arms BUILT=YES with 0 compile errors. NOT FIXED IN FLOW: extending the repair means two more red-first cases and a fresh review over ops #1007 does not own, and **no caller is yet known to reach either path** — the LTX-2.5 `CausalConv3d` geometry that motivated the Conv3d fix materialises a pad of at least the kernel, and the audio-encoder `kConv2d`/`kDepthwiseConv1d` callers are unaudited, which is part of the owed work. `Conv1dOutLength` (`:2886`) and `ConvTranspose1dOutLength` already guard it correctly with an explicit negative-span return, so three of the five conv wrappers use the right shape already. Listed under `## Owed` in [`ltx25-device-residency.md`](../../specs/ltx25-device-residency.md)
Row: LTX25-DEVICE-RESIDENCY
State: UNKNOWN
Kind: bug
GitHub: 1471
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:514`

### Frozen archive evidence

> | [#1471](https://github.com/mudler/vllm.cpp/issues/1471) | `LTX25-DEVICE-RESIDENCY` | **`vt::Conv2d` and `vt::DepthwiseConv1d` compute their output extents with C++ integer division, which truncates toward zero, where torch's shape contract FLOORS.** `src/vt/ops.cpp:2749-2750` (`hout`/`wout`) and `:2883` (`lout`). The two agree for a non-negative numerator and disagree for a negative one whenever `stride > 1`: at `in = 2, k = 3, stride = 2, pad = 0` the numerator is -1, so torch computes `floor(-1/2) + 1 = 0` and raises "Output size is too small" while truncation computes `-1/2 + 1 = 1` and ACCEPTS an extent of 1, convolving over taps the stride skipped. The `VT_CHECK(... > 0)` guard below each line cannot catch it, because truncation has already produced a positive extent. Found by the fresh review of [#1007](https://github.com/mudler/vllm.cpp/issues/1007) (finding F7) against the sibling `vt::Conv3d`, which carried the identical expression and is FIXED on that row: the span is separated from the division and a negative span is refused by name, proven by mutation — reverting the guard reds *conv3d: the shape contract refuses by name* at exit 1 (`CHECK_THROWS` did not throw, `tests/vt/test_ops_conv3d.cpp:446`), restoring it gives 4/4 and 2036 assertions, both arms BUILT=YES with 0 compile errors. NOT FIXED IN FLOW: extending the repair means two more red-first cases and a fresh review over ops #1007 does not own, and **no caller is yet known to reach either path** — the LTX-2.5 `CausalConv3d` geometry that motivated the Conv3d fix materialises a pad of at least the kernel, and the audio-encoder `kConv2d`/`kDepthwiseConv1d` callers are unaudited, which is part of the owed work. `Conv1dOutLength` (`:2886`) and `ConvTranspose1dOutLength` already guard it correctly with an explicit negative-span return, so three of the five conv wrappers use the right shape already. Listed under `## Owed` in [`ltx25-device-residency.md`](../../specs/ltx25-device-residency.md) | bug |

## Resolution

-
