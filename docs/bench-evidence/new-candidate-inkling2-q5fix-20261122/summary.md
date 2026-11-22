# Evidence — Q5_1 block_bytes corrective (ISSUE-LOCAL-01M82AMJXV6DH2BD3SC40YG8DB)

Date: 2026-11-22. Derived from PR localai-org/vllm.cpp#3586 (merged), where
ggml type id 7 (Q5_1) was tabulated as `{32, 21, "Q5_1"}`.

## Defect

`llama.cpp ggml/src/ggml-common.h:237-249`:
`block_q5_1` = `ggml_half d` (2) + `ggml_half m` (2) + `uint8_t qh[4]` (4) +
`uint8_t qs[QK5_1/2 = 16]` (16) = **24 bytes**, with an in-file
`static_assert(sizeof(block_q5_1) == 2*sizeof(ggml_half) + sizeof(uint32_t) + QK5_1/2)`.
The four `qh` bytes carry the fifth (high) bit of each of the 32 quants, one
bit per quant — it is NOT a single `u8 qh`. The 21-byte row made every Q5_1
tensor's `nbytes` short by 3 bytes per block (`R*3` bytes over R rows). The
reader still opened the file because the span check only tests the upper bound
against the data section, but any consumer computing offsets from the tabulated
nbytes would read shifted data.

## Independent confirmation from the artifact header

In the committed walk `docs/bench-evidence/new-candidate-inkling2-20261121/apex-balanced-walk.txt`,
`blk.10.ffn_down_exps.weight` (type 7, shape `[1152,2304,128]`) sits at offset
9,197,330,176 and the next tensor at 9,452,134,144: difference 254,803,968 =
`numel/32 * 24` (numel = 339,738,624), not `numel/32 * 21` (222,953,472).
The new test asserts `t.nbytes == numel / 32 * 24` for every type-7 tensor, so
the regression is caught against the real artifact rather than a hand-picked
constant.

## Fix and verification

- `src/vllm/model_executor/model_loader/gguf_reader.cpp` case 7 → `{32, 24, "Q5_1"}`
  with a comment citing `ggml-common.h:237-249`.
- `tests/vllm/models/test_inkling2_loop_gguf.cpp` and `tests/vllm/test_gguf.cpp`
  assertions updated to 24; the new nbytes-vs-header assertion added.
- The committed W0 spec and evidence summary were corrected from 21 to 24.

`focused.log`: `ctest --test-dir /tmp/opencode/vllm-q5fix-build -R
'^(test_inkling2_loop_gguf|test_gguf)$' --output-on-failure` → **2/2 passed**.
Direct `test_inkling2_loop_gguf` → **11/11 cases, 86/86 assertions, SUCCESS**.
With `VT_INKLING2_APEX_GGUF_PATH=/nonexistent/absent.gguf` the binary still
prints the SKIP reason and exits 77.

Discovery: burn-vllm peer `oc-d1315680f6d21180` cross-checked ggml-common.h
while exchanging Inkling-2 notes. Corrective ownership confirmed by `mk-C`.
