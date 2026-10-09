# EXL3 projection fixtures and replay

These tools capture and compare bounded EXL3 projection, attention, recurrent
state and model outputs. Supply the model, reference manifest, source/runtime
identity and artifact paths explicitly. The tools do not download weights,
change services or device settings, or acquire resource leases. Run original
and native GPU work sequentially; compilation and host comparisons need no GPU.

See [EXL3_XPU.md](../../docs/EXL3_XPU.md) for the current model, pinned native
dependencies, build commands and capability limits. An isolated operator pass
does not qualify integrated MTP state, ordinary verifier admission, sampling
equivalence or maximum-context generation. The frozen default-reference
failures and separately controlled reference retain their distinct labels.

## Artifact admission

`extract_projection.py` preserves real trellis, scale and marker bits and
generates deterministic FP16 M1/M4 inputs. It checks checkpoint metadata and
whole 128-column Hadamard blocks, and refuses to overwrite outputs.
`capture_projection.py` validates the fixture and its adjacent JSON manifest
before loading the original GPU runtime:

```sh
python3 tools/exl3_reference/capture_projection.py \
  --artifact-root "/path/to/artifacts" --fixture head_first.safetensors --check-only
```

`--artifact-root` overrides `EXL3_ARTIFACT_ROOT`; otherwise relative inputs use
the working directory. A configured root also constrains absolute filenames
and symlink targets. Parent traversal is rejected. Capture outputs use the
explicit `--output` destination. Relocation preserves content digests;
historical locations are metadata, not artifact identity.

Missing inputs normally fail. `--optional-artifacts` permits an explicitly
optional external check to exit77 for absent files; corrupt or malformed inputs
still fail. Required qualification must reject a missing-input outcome.
The generated host tests need no model or Torch. See
[EXL3_TEST_ARTIFACTS.md](../../docs/EXL3_TEST_ARTIFACTS.md) for the separate native
SmallM external-test admission contract.

## Standalone projection replay

Build in a supplied Intel oneAPI compiler environment. No particular container
name or mount layout is required:

```sh
cmake -S tools/exl3_reference -B build-projection \
  -DCMAKE_CXX_COMPILER=icpx -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-projection --target exl3_projection_replay -j2
```

This target builds native XPU leaves, queue/allocator, Copy and safetensors
reader code with strict floating-point flags. It requires neither Torch nor
oneDNN/SYCL-TLA linkage. It is an operator replay, not a full-engine build.

The recorded original projection runtime is
`sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918`;
the recorded compiler image is
`sha256:ae6950731b3c031f812a95c0eb615a572239426440bf67fd1b3c9c9cbfce9eca`.
These are measured identities, not substitutes for dependency/source provenance.
Capture must use the runtime identity declared by the fixture manifest. Choose
distinct output names for each run:

```sh
python3 tools/exl3_reference/capture_projection.py \
  --fixture /path/to/head_first.safetensors \
  --output /path/to/head_first_oracle.safetensors \
  --image-identity sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918

build-projection/exl3_projection_replay \
  /path/to/head_first.safetensors /path/to/head_first_oracle.safetensors \
  /path/to/head_first_native.safetensors

python3 tools/exl3_reference/compare_projection.py \
  --fixture /path/to/head_first.safetensors \
  --oracle /path/to/head_first_oracle.safetensors \
  --native /path/to/head_first_native.safetensors \
  --binary /path/to/build-projection/exl3_projection_replay \
  --report /path/to/new-comparison.json
```

The comparison preserves dtype, shape, hashes, finite status and numerical
differences. Its numerical failure is not an artifact skip. Diagnostic split-K
and output-Hadamard replays remain separately attributed; inference does not
import captured tensors or the original runtime's Torch implementation.

## Additional bounded captures

Use each tool's `--help` for required manifests, geometry and output arguments:

| Tools | Scope |
|---|---|
| `capture_grouped.py`, `capture_w8a8.py`, `capture_swiglu.py` | Packed source groups, W8A8 intermediates and activation boundaries |
| `capture_block.py`, `capture_gdn.py`, `capture_variable_gdn.py` | Block and recurrent-state boundaries on declared operands |
| `capture_attention.py`, `replay_attention_rope.py` | Initialized FP8 KV entries and explicit RoPE materialization |
| `capture_verify*.py`, `capture_exact_k_attention.py` | Declared verification shapes and attention diagnostics |
| `capture_target.py`, `compare_target.py` | Full target-head rows on recorded prefixes |
| `capture_mtp_draft.py`, `capture_mtp_gdn.py` | Isolated draft outputs and recurrent operands |
| `capture_integrated_mtp.py`, `capture_mtp_transition.py` | Integrated state observations and active-row transitions |
| `capture_runtime_layout.py`, `serving_timing.py` | Runtime metadata and request/chunk observations |

Ordinary, observed and forced-trace sequences are distinct evidence. A trace
replay can match inputs without proving identical greedy choices. Compare only
initialized KV entries and declared state arrays. Keep default and controlled
arithmetic captures separate; do not replace a failed default golden.

`compare_target.py --gate` returns1 for a failed logit gate. `--report-only`
preserves failed metrics and `gate_exit_code: 1` while returning0 for successful
report creation. That return code does not promote numerical qualification.

Host guard tests are under `tests/scripts/test_exl3_*.py`; select the test for
the tool being changed. External-model/native checks require the explicitly
supplied fixtures and the pinned declared runtime. Missing external data never
establishes an inference pass.
