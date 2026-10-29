# gfx1100 quantized WMMA evidence

Row: `KERNEL-QUANT-CIQ-GEMM-ROCM-RDNA3`.
[Specification](../../../.agents/specs/rocm-rdna3-quant-wmma.md).
Architecture admission is `DONE`. Whole-model performance floors remain `FAILING`.
This packaging change preserves the reviewed implementation, tests, and validation harness.

## Retrieve the complete evidence

[Release](https://github.com/VikashLoomba/vllm.cpp/releases/tag/rdna3-wmma-evidence-d6e40c91f634) ·
[Download archive](https://github.com/VikashLoomba/vllm.cpp/releases/download/rdna3-wmma-evidence-d6e40c91f634/rdna3-wmma-evidence-d6e40c91f634.tar.gz).
Source revision: `d6e40c91f634a041c873c7a04516d55c4d05772a`.
The 511,480-byte archive preserves all 236 original files under this evidence path.
It adds a root `SHA256SUMS` manifest. Archive SHA256:
`996ed227235b9e85196055b9005e2adf7a1d8c86a61ef610249fcfd0be36eb0d`.

Run these commands with Python 3.12 or later. Extraction uses a new private directory.

```sh
(
set -eu
archive=rdna3-wmma-evidence-d6e40c91f634.tar.gz
curl -q --fail --location --output "$archive" \
  "https://github.com/VikashLoomba/vllm.cpp/releases/download/rdna3-wmma-evidence-d6e40c91f634/$archive"
printf '%s  %s\n' \
  996ed227235b9e85196055b9005e2adf7a1d8c86a61ef610249fcfd0be36eb0d "$archive" | sha256sum --check --strict
dest=$(mktemp -d)
python3 - "$archive" "$dest" <<'PYARCHIVE'
import pathlib, sys, tarfile
with tarfile.open(sys.argv[1], "r:gz") as archive:
    for member in archive.getmembers():
        path = pathlib.PurePosixPath(member.name)
        assert not path.is_absolute() and ".." not in path.parts
        assert member.isfile() or member.isdir()
    archive.extractall(sys.argv[2], filter="data")
PYARCHIVE
cd "$dest"
sha256sum --check --strict SHA256SUMS
printf 'Extracted evidence to: %s\n' "$PWD"
)
```

Open `docs/bench-evidence/rocm-rdna3-quant-wmma/README.md` and `model-summary.md` inside the extraction for the full reports.
Their original cross-links outside the evidence directory require the source checkout at the revision above.
The archive retains commands, manifests, failed attempts, compiler identities, and receipt qualifications unchanged.
Large raw matrices, model logits, and full profiler traces were already external at their sealed source paths.

## Correctness, review, and scope

Implementation: `c3fe98ba6c55ce71e75746e1b944a27640464e0f`.
Measurements used physical RX 7900 XTX `gfx1100` on 13 September 2026.
Every GPU invocation held `/home/vikash/gpu.lock` and selected device 0.
Native tooling used HIP 7.15, Clang 23, and rocWMMA 2.2.1. The archive seals exact toolchain hashes.

| Gate | Result | Archived receipts |
|---|---|---|
| G1 architecture policy | Four assertions fail before admission. Green: 16 cases, 109 assertions. Attention admission stays unchanged. | `arch-red.log`, `arch-green.log` |
| G1-G2 physical tiles | Red: four cases fail ten dispatch assertions. Green and separate scalar control: 46 assertions each. | `hardware-red.log`, `hardware-green.log`, `hardware-scalar.log` |
| G2 arithmetic and ISA | Both tile bodies remain byte-identical. Eight signed-int8 WMMA instructions span Q4_K/Q6_K and F32/BF16. | `tile-bodies.json`, `gfx1100-wmma-isa.txt` |
| G2 original primary fixtures | All 240 cases pass against the dense reference and plugin outputs with original modes and tolerances. | `original-mmq-summary.json`, `original-mmq-comparison.json` |
| G3 production reachability | Prompts of 16 and 37 tokens enter both formats. All 1024 logits and eight completion tokens match scalar. | `public-red.log`, `public-green.log` |
| G5 fresh review | `PASS`, no findings. Mutated admission, launches, scalar override, and finite corruption fail their gates. Source hashes restore exactly. | `review-review-report.json`, `production-mutations.json` |
| G5 operator verification | Independent architecture, public, and 240-case gates pass. HIP: 29 registered tests, zero failures, five baseline skips. | `operator-receipt.json`, `model-operator-mmq-comparison.json` |

Physical cases retain F32/BF16 outputs, joint and asymmetric tails, and the partial four-wave block `M=32,N=48,K=512`.
Original fixtures retain `M=7,83,128,2048`, `K=256,1024`, seed zero, and F16/BF16/F32 inputs.
The native harness narrows F32 to F16 for upstream F16 output comparisons. Native Q8_K activation and output contracts remain unchanged.
Source chain: plugin `tests/test_kernels.py:145-196` → `ops.py:200` → `csrc/gguf/gguf_kernel.hip:221` → `mmq_hip.cuh:490,591`.

HIP skips require three model fixtures and two-visible-device coverage. Only device 0 was exposed.
Historical full preflights preserve 12 skips. The operator's original role failure and its scoped branch-rename resolution remain recorded.
Seven NumPy suites subsequently pass in isolation. Five adherence checkpoint subcases remain unavailable without `VT_LTX25_ADHERENCE_MODEL`.
The archived final-records preflight retains its onboarding fixture failure and the passing isolated rerun of all 39 cases.
The later source-head readiness run has zero failed checks and five argument-dependent skips, with exit 1 under readiness policy.
PR classification passed separately. ARM, CPU, CUDA, and Triton argument-dependent checks concern unchanged architecture paths.
These qualified results do not constitute an all-green readiness claim.

## Model identity and observed results

Checkpoint: `Qwen3.5-4B-Q4_K_M-unsloth-e87f1764.gguf`, 2,740,937,888 bytes.
Source: `unsloth/Qwen3.5-4B-GGUF@e87f176479d0855a907a41277aca2f8ee7a09523`.
SHA256: `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`.
Task snapshots: vLLM `39545e475d3627287ff69c25465dc0bd405f67e1`,
llama.cpp `093a2f86c3e37c54fa3e1f9efb17b304f3433abd`,
GGUF plugin `d4c1f0d082fc7cd4350da56689109a01c1f29d6c`.
These developer-selected snapshots do not advance the repository parity pin.

Eight requests alternate 183/174 input tokens, totaling 1428 inputs and 128 generated IDs.
Concurrency, temperature, and seed are 1, 0, and 0. Each request generates 16 tokens without chat templates or EOS-logit masks.
All eight input arrays match the primary. All six native process outputs match both oracles exactly.
Native `rocprofv3` captures contain 152 WMMA calls per prefill, 1216 total, and zero during decode or with WMMA disabled.
Both traced completion arrays match the primary. Native traces do not establish cross-engine invocation parity.

| Observed axis | WMMA enabled | WMMA disabled | Enabled/scalar |
|---|---:|---:|---:|
| Prefill input tokens per summed first-token latency | 224.12 tokens/s | 173.86 tokens/s | 1.2891 |
| Median first-token latency | 386.51 ms | 617.09 ms | 0.6263 |
| Mean per-stream decode rate | 53.33 tokens/s | 53.15 tokens/s | 1.0034 |
| Sampled whole-device memory peak | 4.8638 GB | 4.8567 GB | 1.0015 |

Native medians cover three complete processes per mode in `on1,off1,off2,on2,on3,off3` order, including first-use effects.
Each oracle loads once and captures all four two-request legs. Primary compilation caches were warmed by its failed first attempt.
Clocks vary dynamically. Monitor windows include initialization and teardown, so accepted clock attribution remains `PENDING`.
Engine cache capacities differ. The primary reserves approximately 17 GiB, preventing equal-capacity memory conclusions.
The llama.cpp client includes extra logit, memory, and finite-value instrumentation. Its decode-call-only timings are a separate scope.
Memory values are sampled device-wide or process-tree peaks. Small decode or memory differences do not establish stable changes.
Native/primary total whole-run throughput is 0.9443 and decode is 0.8880. Native/llama.cpp ratios are 0.2439 and 0.5743.
The [open performance issue](../../../.agents/issues/_owed/ISSUE-LOCAL-01M2F4WCD6ZK5VH5S8TF83APD6.md) retains every below-floor axis, comparable timing windows, matching oracle traces, and accepted clock reproduction.
The archived `model-summary.md` retains every value and ratio. Architecture admission claims no full-model parity or performance ceiling.

> EVIDENCE-LOST (2026-10-30): `arch-green.log`, `arch-red.log`, `hardware-green.log`, `hardware-red.log`,
> `hardware-scalar.log`, `public-green.log`, `public-red.log`
> The raw logs named above are unrecoverable: the pre-2026-10-30 global `*.log` rule
> dropped them at `git add` and the worktree removal took the untracked files; the reviews
> verified them at review time. Recorded 2026-10-30 by FIX-EVIDENCE-LOG-GATE.
