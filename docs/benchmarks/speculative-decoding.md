# Speculative decoding

**THE PIN MOVED UNDER TWO OF THESE ROWS.** The parity pin
advanced to <!--pin:commit-->`a7c23ac96d`<!--/pin--> on 2026-09-26
([#2817](https://github.com/mudler/vllm.cpp/issues/2817)). The **MTP** row and
the **DFlash** row carry a vLLM denominator measured against the PREVIOUS pin
`555967922` with FlashInfer `0.6.15.post1`, which is the oracle's attention
backend on that path and moves to `0.6.18` at the new pin. Both owe a
re-measurement ([#2818](https://github.com/mudler/vllm.cpp/issues/2818)), and the
DFlash re-take must record the oracle's SELECTED backend rather than assume it
carries over. The advance preceded the re-measurement by a developer ruling that
inverts the sync cycle's step order; a red there requires reverting the pin.
Nothing here is withdrawn today.

| Speculator | Model | Result | Status |
|---|---|---|---|
| MTP | Qwen3.6-27B NVFP4 | token-identical to vLLM MTP, **~4% faster at c1**; on-par at c2-c8 | `DONE` |
| DFlash | Qwen3.6-27B NVFP4 | **2.9x over spec-off** (10.16 → 29.32 tok/s), at/above vLLM DFlash-on (**1.003x**, non-overlapping bands) | `DONE` |
| n-gram | Qwen3.6-27B NVFP4 | draft-free (`SPEC-NGRAM`); 27B 5/5 STRICT our-ngram-ON == vLLM-ngram-ON, 180/180 drafts accepted (correctness only, no speed row yet) | `DONE` |
| DSpark | 27B NVFP4 dense k=15; 35B-A3B MoE k=8 | MoE 35B-A3B: **0.835x** paired on kairos-17dd (matched 89 tokens, warm oracle cache). Prior 0.957-0.989 came from a different machine with a cold oracle (#442) | `ACTIVE` |
| DSpark block floor | Qwen3.8-27B + `RadixArk/Qwen3.8-27B-DSpark` @ `85ef153b` | a `k` below the draft's block is refused instead of drafted; the run gate that exhibits the garbling is **owed** and needs a GPU lease (#1225) | `ACTIVE` |
| DSpark draft routing | Qwen3.8-27B + `RadixArk/Qwen3.8-27B-DSpark` @ `85ef153b` | **PENDING**, no number. The token-exact run gate needs the 2.53 GiB draft and GPU time, and neither authority is recorded; only the CPU classification gate has run (`.agents/specs/dspark-qwen3-routing.md` §6) | `ACTIVE` |
| DFlash2 | `Qwen3.8-27B` + `z-lab/Qwen3.8-27B-DFlash2`, k=7, GB10, oracle BEYOND-PIN @`66e5414c` on `TRITON_ATTN` (#1456, #1562) | **0.8017x** decode throughput (13.051 vs 16.279 tok/s, 16 warm legs each): **RECORDED, NO floor declared -- a ratio, NOT a pass**. 3 of 4 axes NOT MEASURED. Correctness READ: 4/4 token-exact, 45/47 draft blocks identical | Ours is OFF the paged CUDA-graph fast path, the oracle GRAPHS its draft step. 5 draft layers resolved `FlashAttentionBackend` (#1685); arm clock windows differ (#1673). [record](../../.agents/benchmark-record.md) | `ACTIVE` |
| Drafter chain (`SPEC-DRAFTER-CHAIN`, [#1522](https://github.com/mudler/vllm.cpp/issues/1522)) | n/a, nothing runs a chain yet | **PENDING, no number admissible.** W1 landed the field and its refusals only. The premise is an ACCEPTANCE claim, invisible to a token gate, so W2/W4 must read first. G1 and G5 PASS on CPU | `ACTIVE` |
| Breadth (EAGLE1/3, suffix, ngram-gpu, dynamic-k, ...) | n/a | enumerated from vLLM source + `INVENTORIED` 2026-08-06 (`.agents/specs/spec-decode-inventory.md`), unmeasured | `INVENTORIED` |
