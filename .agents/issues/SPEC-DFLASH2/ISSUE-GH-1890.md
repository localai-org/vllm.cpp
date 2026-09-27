ID: ISSUE-GH-1890
Title: **The DFlash draft block's attention never got the FA-2 treatment: `DFlashPagedBlockAttentionWarpKernel` at 449.7 us/call x 5.1 calls/step = 2.29 ms/step, against SGLang's 14.3 us/call for the same work** — while W10's TARGET VERIFY, on the FA-2 split-KV lane, runs at 17.1 us/call and is marginally ours. 16.2 verify + 5.1 draft = 21.3 attention calls/step on both engines: same call count, one lane 31x slower. **The blocking property is KV RESIDENCY, and it is none of the four #1890 names** (page size 16, identity block table, bf16, head dim — all admit): `vt::DFlashPagedBlockAttention` reads the block's own (1+k) K/V out of contiguous per-layer tensors that are in NO paged cache, and every split-KV launcher addresses K and V exclusively through a block table, so handing one the store's pools would drop every block row — a wrong answer, not a slow one. W11 makes those rows RESIDENT (`vt::ReshapeAndCache` into the store's own pages) and reads the whole combined sequence as ONE `vt::PagedAttention`, which is the presentation upstream uses for the same work and is why SGLang issues one kernel for both lanes. The mask maps exactly with no new mask code — `PagedAttentionArgs` already carries FlashAttention's bottom-right alignment, so full / SWA / plain-causal become non-causal / causal+`{W-1,0}` / causal. **BYTE-IDENTICAL on CPU, asserted rather than argued**: the two kernels are the same three-pass online softmax in the same j-ascending order over the same bf16 bits, gated element-for-element across five mask and layout cases plus a drafted-token A/B through the production runner. CUDA additionally widens the W10 admission and `LaunchSpecDecodeFA2Bf16` to head dim 128 and to the three masks (the d256 verify arm stays dispatch-identical), and a classified batch the dispatch cannot serve now NARRATES its failed conjunct group once to stderr. `VT_FA2_DFLASH_BLOCK=0` is the same-binary rollback. The GPU number, the on/off A/B, the GPU token battery and the first CUDA compile are owed, operator-run, in [dflash2-draft-block-fa2.md](../../specs/dflash2-draft-block-fa2.md)
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: perf
GitHub: 1890
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:705`

### Frozen archive evidence

> | [#1890](https://github.com/mudler/vllm.cpp/issues/1890) | `SPEC-DFLASH2` | **The DFlash draft block's attention never got the FA-2 treatment: `DFlashPagedBlockAttentionWarpKernel` at 449.7 us/call x 5.1 calls/step = 2.29 ms/step, against SGLang's 14.3 us/call for the same work** — while W10's TARGET VERIFY, on the FA-2 split-KV lane, runs at 17.1 us/call and is marginally ours. 16.2 verify + 5.1 draft = 21.3 attention calls/step on both engines: same call count, one lane 31x slower. **The blocking property is KV RESIDENCY, and it is none of the four #1890 names** (page size 16, identity block table, bf16, head dim — all admit): `vt::DFlashPagedBlockAttention` reads the block's own (1+k) K/V out of contiguous per-layer tensors that are in NO paged cache, and every split-KV launcher addresses K and V exclusively through a block table, so handing one the store's pools would drop every block row — a wrong answer, not a slow one. W11 makes those rows RESIDENT (`vt::ReshapeAndCache` into the store's own pages) and reads the whole combined sequence as ONE `vt::PagedAttention`, which is the presentation upstream uses for the same work and is why SGLang issues one kernel for both lanes. The mask maps exactly with no new mask code — `PagedAttentionArgs` already carries FlashAttention's bottom-right alignment, so full / SWA / plain-causal become non-causal / causal+`{W-1,0}` / causal. **BYTE-IDENTICAL on CPU, asserted rather than argued**: the two kernels are the same three-pass online softmax in the same j-ascending order over the same bf16 bits, gated element-for-element across five mask and layout cases plus a drafted-token A/B through the production runner. CUDA additionally widens the W10 admission and `LaunchSpecDecodeFA2Bf16` to head dim 128 and to the three masks (the d256 verify arm stays dispatch-identical), and a classified batch the dispatch cannot serve now NARRATES its failed conjunct group once to stderr. `VT_FA2_DFLASH_BLOCK=0` is the same-binary rollback. The GPU number, the on/off A/B, the GPU token battery and the first CUDA compile are owed, operator-run, in [dflash2-draft-block-fa2.md](../../specs/dflash2-draft-block-fa2.md) | perf |

## Resolution

-
