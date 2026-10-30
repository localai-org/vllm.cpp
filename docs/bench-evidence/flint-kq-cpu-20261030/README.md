# FLINT-KQ-CPU-DISPATCH — evidence index

Row: `FLINT-KQ-CPU-DISPATCH` — the CPU paged arm serves the keep-quant tree.
Spec: `.agents/specs/flint-kq-cpu-dispatch.md`. Detail: [`report.md`](report.md).

## Review repair (M5) — the keep-quant gate's tree-identity gap (2026-10-30)

The fresh review's single finding: the keep-quant paged token gate proved "the
paged arm serves in-band", not "the packed tree served" — a loader that
silently ignored `VT_FLINT_KEEP_QUANT` (the bf16 arm) would have passed it
unchanged.

Repair: the gate now asserts on the loaded tree itself, before the forward
loop (`tests/vllm/models/test_flint_gguf.cpp`, "keepquant tree identity"
REQUIREs) — dtype checks only, no data scan:

- `REQUIRE(vt::IsBlockQuant(kq_tree.lm_head.dtype))` — the untied head.
- one routed-expert `gate_proj.w` across the layers must carry a block
  encoding, read back through the production `LoadedModel` via the new
  `vllm::FlintLoadedModelWeights` accessor (declared in
  `flint_gguf_weights.h`, defined at `flint.cpp:1355`, the kolibri1 seam).

RED-under-mutation (loader forced to always `keep_quant=false` in
`flint_gguf_weights.cpp LoadFlintKeepQuantFromGguf`, scratch copy, restored
byte-for-byte after):

```
./tests/vllm/models/test_flint_gguf.cpp:285: FATAL ERROR:
REQUIRE( vt::IsBlockQuant(kq_tree.lm_head.dtype) ) is NOT correct!
values: REQUIRE( false )
[doctest] assertions: 4 | 3 passed | 1 failed |  Status: FAILURE!
```

GREEN at head (mutation reverted, rebuilt):

```
MESSAGE: keepquant tree identity: expert gate_proj dtype 1 (block), lm_head dtype 1 (block)
MESSAGE: keepquant paged token gate: 53/60 greedy tokens exact, 7 near-tie, 0 hard, worst gap 2.48648
[doctest] test cases: 4 | 4 passed  | assertions: 184 | 184 passed | Status: SUCCESS!
```

Sibling suites unchanged: `test_flint_paged` 208/208; bf16 token gate 46/53,
0 hard flips.
