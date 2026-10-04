# `PermuteVHeads` on CUDA: the missing arm of the GDN column-permuted keep-quant path — ISSUE-LOCAL-01M3QFKXY1N2RJV6PKDA19Q9WK

`vt::PermuteVHeads` is the runtime gather that lets `out_proj` stay tiled-order
Q5_K instead of expanding to bf16 at load. The op was registered on CPU and ROCm
only, so the opt-in `VT_GDN_COLPERM_KEEP_QUANT` arm refused by name on CUDA — the
one backend the released 27B GGUF arm is served from.

Issue: [ISSUE-LOCAL-01M3QFKXY1N2RJV6PKDA19Q9WK](../issues/QUANT-QWEN38-27B-GGUF-ARM/ISSUE-LOCAL-01M3QFKXY1N2RJV6PKDA19Q9WK.md).
Owning row: `QUANT-QWEN38-27B-GGUF-ARM` ([quantization-matrix.md](../quantization-matrix.md));
the row's spec is [qwen38-27b-quant-arms.md](qwen38-27b-quant-arms.md).

## Premise, grounded

| Where (line anchors at this branch's base, `b45a94273`) | What |
|---|---|
| `src/vllm/model_executor/models/qwen3_5.cpp:1810-1820` | `GdnOutProjMatmul`: when `w.out_proj_tiled`, the gated-norm output is permuted grouped→tiled with `vt::PermuteVHeads` before the K-quant GEMV; otherwise no permutation (the bf16 expanded weight has `ReorderVCols` applied). |
| `src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp:1136`, `:1272` | `VT_GDN_COLPERM_KEEP_QUANT` gates keeping the tiled-order Q5_K weight instead of expanding it. |
| `src/vt/cpu/cpu_ops.cpp:4027-4044`, `:4471-4474` | The CPU kernel and its registration — the index mapping the CUDA arm must reproduce. |
| `src/vt/rocm/rocm_ops.hip:242-244` | The ROCm registration. |
| `include/vt/ops.h:2410-2411`, `src/vt/op_provider.cpp:433` | The op's signature and name. |
| `docs/ENVIRONMENT.md` | `VT_GDN_COLPERM_KEEP_QUANT`: "Saves ~4x weight bandwidth (Q5_K ~5 MB vs bf16 20 MB per call)". |

Two record gaps go with the code gap, and this issue names both:

- The arm was **not recorded as owed** anywhere in `.agents`: no `T25` or
  `VT_GDN_COLPERM_KEEP_QUANT` reference exists. A reader could not have found it.
- The op had **no test on any backend**. The CPU mapping the CUDA kernel claims to
  match was itself unpinned, so a wrong mapping and a right one were
  indistinguishable.

## Design

One CUDA kernel, a flat gather over `T * num_k * rpk * dv` elements, with the
index map written exactly as the CPU kernel computes it: for output element
`(row, t, h)`, `r = t / num_k`, `k = t % num_k`, source head
`g = k * rpk + r` — i.e. `out[row, (r*num_k + k)*dv + h] = in[row, (k*rpk + r)*dv + h]`.
Registered under `DeviceType::kCUDA` for `OpId::kPermuteVHeads` beside the other
elementwise glue ops in `cuda_glue.cu`. bf16 in / bf16 out, same as the CPU and
ROCm arms.

## Tests

`tests/vt/test_cuda_ops.cpp` gains the op's **first** coverage on any backend:

- The expected permutation is computed in the test from the formula above, so the
  CPU arm is checked against the specification rather than against itself.
- The CUDA arm is compared **bit-exactly** against the CPU output — a gather has
  no reduction, so equality is exact and a tolerance would hide a wrong index.
- Before this change the CUDA half refuses (`vt::GetOp` throws for an
  unregistered `(kPermuteVHeads, kCUDA)`), which is the RED state.
- On a build without CUDA the case still runs the CPU half and says so.

## Evidence

- The parity case GREEN on a CUDA build; the mutation that removes the
  registration makes it RED (the refusal is the pre-change state).
- CPU-only builds keep the CPU half.

## Gates

- `ctest --test-dir build -R test_cuda_ops` on a CUDA build with
  `-DVLLM_CPP_CUDA_ARCHITECTURES=120a` on the local `sm_120a` card.
- The full `ctest --test-dir build` on the same build.
- `scripts/agent-preflight.sh --staged`.

## Owed

- An end-to-end `VT_GDN_COLPERM_KEEP_QUANT=1` measurement on the released
  Qwen3.8-27B GGUF artifact: token identity against the arm as shipped, plus the
  weight-bandwidth and decode-speed delta. The arm stays opt-in until then, so no
  performance claim is made here.
