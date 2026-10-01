# the wave-2 whole-decode fuse: the captured MatmulBT launch records 23,552 B where the chain recorded 147,456 B (2026-10-02)

Implements the safe route the
[tt-matmul-class-split-20261001.md](tt-matmul-class-split-20261001.md)
verdict names: the matmul itself stays stock, and the NON-matmul chain
around it collapses. Branch `row/tt-matmul-fusion` (worktree
`/tmp/vllm-region-capture-spec`, build `build2/build2-rescue2` @ pin
`6449cf13f7b`), device `thor:gpu0`-class Blackhole under
`flock $HOME/gpu.lock`, `luwen reset` + 15 s, pin-first env
(`TT_METAL_HOME=/home/lu_zero/Sources/tt/tt-metal-pin`,
`build_release_script/lib64` + `libexec/tt-metalium`), no other
`TT_METAL*`.

## 1. The fused arm (src/vt/tenstorrent/tenstorrent_keepquant.cpp, E=1 dense arm)

When the encoding is Q6_K and the WHOLE decoded f32 plane (plus the
exact-f32 arm's prod plane) fits the chunk loop's own
`kKeepQuantChunkPlaneBytes` budget, the chunk loop is bypassed:

1. ONE `kKeepQuantDecodeFusedKernelSrc` launch decodes the entire word
   shadow (whole-extent rows — the window where the chain's `slice()`
   aliases its input; no slice op, no per-chunk programs).
2. `ttnn::typecast` f32→bf16, `ttnn::to_layout` TILE — the chain's own
   operand prep, one program each.
3. ONE stock `ttnn::operations::matmul` (bf16 TILE, transpose_b).
4. The output typecasts f32 IN the TILE domain and commits directly —
   the chain's ROW_MAJOR partial round-trip exists for the concat domain
   a multi-chunk assembly needs, and the single-tile arm does not; the
   typecast is elementwise, so every value is bit-identical.

Per-launch tt-metal programs: ~6 (fused decode, typecast, to_layout,
matmul, partial typecast, commit) against the chain's ~20.

Declines (each named, never silent): non-Q6_K encodings, `VT_TT_KEEPQUANT_FUSED=0`,
`VT_TT_KEEPQUANT_MM_CHAIN=1` (the new force-chain kill switch), an
over-budget plane, and the fused decode's own L1 decline — all fall to
the proven chunk chain, identically in the eager and captured passes.

## 2. Region record (the gate)

Focused doctest `kTENSTORRENT matmul region record class split
(VT_TT_MMCLASS=1)`, extended to the wave-2 gate (the fused leg runs
first — the region probe reads the live trace-buffer total, so only the
first region in a process closes clean; the chain leg runs second as an
informational baseline).

| arm | region close, [64,5120]x[5120,5120], 1 launch |
|---|---|
| chain, HEAD (RED run, stash of the fused arm) | **147,456 B** — byte-exact the class-split audit's number |
| fused, wave 2 (GREEN) | **23,552 B** |
| chain re-measured under `VT_TT_KEEPQUANT_MM_CHAIN=1` | 147,456 B (red run), informational |

Gate `<= 32,768 B`: **GREEN** at 23,552 B — 6.26x, `147456 → 23552`.
Logs `/tmp/mm-red-full.log`, `/tmp/mm-green-full.log`.

## 3. Bit-exactness

Doctest `kTENSTORRENT wave-2 fused MatmulBT launch is bit-exact to the
chunk chain`: the SAME launch on both arms (`VT_TT_KEEPQUANT_MM_CHAIN=1`
selects the chain), `memcmp` on the f32 outputs. Four shapes, all
byte-identical on device:

| P | N | K | arm | covers |
|---|---|---|---|---|
| 64 | 333 | 5120 | prefill | partial-last-core + idle-core tails (333 % 110 != 0), zero d/scale blocks |
| 1 | 64 | 5120 | prefill | single chunk, broadcast form |
| 1 | 333 | 5120 | exact-f32 | tails, the fused decode + chain permuted tail |
| 1 | 130 | 2560 | exact-f32 | single chunk |

The wave-1 capture-x2 byte-identity harness runs unchanged in the full
suite (below) with the fused arm default-on.

## 4. Full TT suite

`build2/build2-rescue2` `test_tenstorrent_backend`, whole binary, one
process under the lock. See the row spec's `## Now` and the issue entry
for the result line.

## 5. Records

Same issue (`ISSUE-LOCAL-01M3JXEFQKSZP23PP2HWY9G0VQ`), same row
(TT-DECODE-FUSION wave 2). Commits `d20f5ae5c` (the fused arm + the
red-first gate), `443e70bc0` (the golden).
