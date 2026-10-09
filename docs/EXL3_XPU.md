# EXL3 on the native XPU backend

This is an experimental Intel SYCL/Level Zero backend. The measured model is
`turboderp/Qwen3.8-27B-exl3`, revision
`113cf7ab958054860e43fb7f3063b1af19171095`, with a 4-bpw EXL3 body, a full
6-bpw target head and a 65536-token compact draft vocabulary. Activations are
FP16, recurrent SSM state is FP32 and attention KV is FP8. Large projections
use rotated EXL3 W8A8; this is not a GPTQ W4A8 engine. Provide model files and
the draft vocabulary explicitly. Configuration and tests do not download them.

## Source build

CPU builds require no oneAPI, GPU, model or oneDNN. A focused host configuration
is:

```sh
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_XPU=OFF \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_HIP=OFF -DVLLM_CPP_VULKAN=OFF \
  -DVLLM_CPP_METAL=OFF -DVLLM_CPP_TENSTORRENT=OFF \
  -DVLLM_CPP_XPU_ONEDNN=OFF -DVLLM_CPP_XPU_GPTQ4=OFF \
  -DVLLM_CPP_SERVER=OFF -DVLLM_CPP_HF_DOWNLOAD=OFF \
  -DVLLM_CPP_BUILD_EXAMPLES=OFF -DVLLM_CPP_WITH_DIARIZATION=OFF
cmake --build build-cpu --target exl3_external_artifact_probe test_shared_ptr_cache -j2
ctest --test-dir build-cpu --output-on-failure \
  -R '^(test_exl3_external_artifacts|test_shared_ptr_cache)$'
```

For XPU, provide an Intel oneAPI compiler environment, a Level Zero driver,
a SYCL/GPU installation of exactly oneDNN 3.13.0 and the pinned SYCL-TLA source
checkout. The measured toolchain is `icpx` 2026.1.1 (20260724), IGC 2.41.5,
driver `1.17.39758+10`, Arc Pro B70 device ID57891. This identifies the tested
environment; it does not qualify other software or devices.

```sh
cmake -S . -B build-xpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DVLLM_CPP_BUILD_TESTS=ON \
  -DVLLM_CPP_XPU=ON -DVLLM_CPP_XPU_ONEDNN=ON -DVLLM_CPP_XPU_GPTQ4=OFF \
  -Ddnnl_DIR=/path/to/onednn-install/lib/cmake/dnnl \
  -DVLLM_CPP_SYCL_TLA_DIR=/path/to/sycl-tla \
  -DVLLM_CPP_XPU_XE2_GDN=ON -DVLLM_CPP_XPU_XE2_PREFILL=OFF \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_HIP=OFF -DVLLM_CPP_VULKAN=OFF \
  -DVLLM_CPP_METAL=OFF -DVLLM_CPP_TENSTORRENT=OFF \
  -DVLLM_CPP_SERVER=ON -DVLLM_CPP_HF_DOWNLOAD=OFF \
  -DVLLM_CPP_BUILD_EXAMPLES=ON -DVLLM_CPP_WITH_DIARIZATION=OFF
cmake --build build-xpu --target server test_xpu_backend test_xpu_exl3_mtp test_gptq4_weight -j2
```

`VLLM_CPP_XPU_GPTQ4` is a compatibility alias for the same optional oneDNN
capability. Both it and `VLLM_CPP_XPU_ONEDNN` default OFF; either ON requires
XPU. The alias does not qualify GPTQ. The current configuration builds the
GDN donor and shared attention verification/decode sources, while the separate
prefill donor option is OFF. Preserve the source-specific floating-point flags.

Standalone development benchmarks and local receipt orchestration are excluded
from this contribution composition. Functional operators and the focused tests
remain available; GPTQ compatibility code is diagnostic, not model qualification.

| Dependency | Source pin / license |
|---|---|
| EXL3 ESIMD | `c59d9442aba8610188837e37724600f1517d7335`; MIT |
| Xe2 attention | `6d92b1bfbf32767ecda8e819613eb151e70030ad`; Apache-2.0 and individual BSD-3-Clause notices |
| Xe2 GDN | `ddf336d86e3c8602888572a3502f951abd51df12`; Apache-2.0, with separate Intel BSD-3-Clause helpers |
| External SYCL-TLA | `87f6850680a580654b9ea2c80dbc01aeb36ad231`; BSD-3-Clause |
| External oneDNN 3.13.0 | `0e2a5bfeef1bfbffc3137464606540233086ce9b`; Apache-2.0 |

[NOTICE](../NOTICE) and the individual source licenses remain required. Source
pins alone do not establish the provenance of an independently supplied wheel
or prebuilt library. Record the actual compiler, dependency build and binary
identity when reproducing a measurement.

## Public HTTP server

The build above produces `build-xpu/examples/vllm-server`, target `server`.
Load the oneAPI environment and make the supplied oneDNN library visible to the
runtime loader. Use explicit local model and draft-map paths. No Python or Torch
runs in the native inference process.

The exact compact MTP map is included at
`third_party/exl3xpu/qwen3.8-27b-draft-vocab.json`, under its retained MIT license.
See that directory's README for the distinct metadata source revision and blob.
From the repository root, verify and copy it without reformatting:

```sh
printf '%s  %s\n' \
  b4eadc088059190983fe0498af11864f5aaa2eaf2ec58ae9864f0715634d313d \
  third_party/exl3xpu/qwen3.8-27b-draft-vocab.json | sha256sum --check
cp third_party/exl3xpu/qwen3.8-27b-draft-vocab.json /path/to/draft_vocab.json
```

The JSON schema contains `block_size` (128), `n_blocks` (512), `blocks` (512
sorted unique block indices), `tokens` (65536), and source `vocab` (248077).
The retained `model` string and corpus statistics describe its donor selection;
they are not runtime paths or weights. The native full head has 248320 padded
rows. The loader verifies the exact 2654-byte file's hash before using its map;
a different selection or reserialized JSON is rejected.

```sh
EXL3_DRAFT_VOCAB=/path/to/draft_vocab.json \
VT_ASYNC_SCHED=0 VT_ASYNC_RUNNER=0 VT_XPU_GRAPH=1 \
VLLM_CPP_CUDAGRAPH=1 VLLM_CPP_DENSE_DECODE_GRAPH=1 \
VT_XPU_ATTENTION=auto VT_XPU_XE2_VERIFY=1 \
VT_XPU_PROFILE=0 VT_XPU_HOST_PROFILE=0 \
VT_XPU_GDN_MIXED_TOKEN_VIEWS=1 \
VT_XPU_SILU_FP16_TABLE=1 VT_XPU_SILU_FP16_TYPED=1 \
VT_XPU_W8A8_MODEL_MAP=1 VT_XPU_W8A8_PREPARE=1 \
VT_XPU_SMALLM_MODEL_MAP=1 VT_XPU_GDN_GATED_SILU_TABLE=1 \
build-xpu/examples/vllm-server \
  --model /path/to/model --served-model-name qwen38-exl3 \
  --host 127.0.0.1 --port 8000 --device auto \
  --block-size 1600 --num-blocks 180 --max-model-len 262144 \
  --max-num-seqs 4 --max-num-batched-tokens 4096 --kv-cache-dtype fp8 \
  --enable-prefix-caching --no-enable-thinking \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

The existing CLI accepts `auto`, `cpu` and `cuda`; it does not accept `xpu`.
In this XPU build with CUDA/HIP disabled, `auto` resolves the XPU platform.
For target-only operation omit `--speculative-config`; no draft map is required.
`EXL3_DRAFT_VOCAB` is required only when loading the compact MTP head.
The checkpoint's generation configuration remains the default; the smoke below
explicitly requests greedy sampling. Vision requests are outside this text path.

Run the portable client against the server, selecting its actual speculation mode.
Use a new output path. Python is a client/test dependency only:

```sh
python3 tools/bench/exl3_http_smoke.py \
  --url http://127.0.0.1:8000 --model qwen38-exl3 --mode mtp3 \
  --out /path/to/new-http-result.json
```

The client checks nonstreaming/streaming text and terminal usage, C1–C4 client
requests, cancellation and EOS followed by successful requests, chat-template
token accounting and speculation counters. Simultaneous clients do not prove
GPU overlap or the identity of a reused physical slot; the native lifecycle
test supplies separate state-ownership evidence. Target-only requires no draft
counter advance, allowing registered constant counters. MTP requires draft
activity; its exact depth of three must be checked in the launch configuration,
since this server's info endpoint does not expose that resolved setting.
Both scheduler gauges must be present, finite, nonnegative and become zero
within `--drain-timeout` (default two seconds, maximum ten). Drain observations
retain values and timestamps, including on failure.
The client counts output through API usage, not SSE frame counts.
Streaming logprobs are not provided by the existing transport. Nonstreaming MTP
logprob positions may be absent and appear as `token_id` placeholders; this smoke
is not logprob parity or exact-token-ID qualification. The native control check
below provides the separate token-ID comparison.

For an aligned prefix check, restart with `--max-num-batched-tokens 1600` and add
`--prompt-ids /path/to/prompt.json`. The file supplies exactly 4096 IDs under
`prompt_token_ids` or the documented native workload's first request. The client
requires an exact public detokenize/tokenize round trip, identical cold/warm output
and a positive cached-token increment. This budget is a separate control:
4096-budget prefix reuse remains limited for the historical 16K/64K resend cases.
The HTTP cached-token metric check currently fails even at budget1600: a separate
existing snapshot trace proves a1600-token restore for the4K request, but
`prompt_tokens_cached_total` stays zero. Treat the client's optional prefix mode
as a strict diagnostic, not a passed qualification. The native32K/O64 test instead
checks the warm scheduled position directly and reuses30400 tokens.

## Focused model check

After providing the model and matching draft vocabulary, a native lifecycle
check can be run explicitly. The output file must not already exist:

```sh
VT_B70_EXL3_MODEL=/path/to/model EXL3_DRAFT_VOCAB=/path/to/draft_vocab.json \
VT_B70_EXL3_ENGINE_OUTPUT=/path/to/new-lifecycle.json \
VT_B70_EXL3_ENGINE_DEPTH=3 VT_B70_EXL3_ENGINE_GRAPH=1 \
VT_ASYNC_SCHED=0 VT_ASYNC_RUNNER=0 VT_XPU_GRAPH=1 \
VLLM_CPP_CUDAGRAPH=1 VLLM_CPP_DENSE_DECODE_GRAPH=1 \
VT_XPU_ATTENTION=auto VT_XPU_XE2_VERIFY=1 \
build-xpu/tests/test_xpu_exl3_mtp --test-case='*R08 lifecycle*'
```

The verifier opt-in above is experimental. The test covers native1/4/2/1,
mixed prefill/speculation, EOS, cancellation, slot reuse, poisoned spares and
graph/eager/graph. Its sentinel layers0/47 are not a full per-token reference
state proof. Missing model-test environment values exit77; a required
qualification runner must reject that outcome. This legacy model check does
not implement `EXL3_REQUIRE_ARTIFACTS`. The separate portable SmallM external
checks do implement required/optional admission, documented in
[EXL3_TEST_ARTIFACTS.md](EXL3_TEST_ARTIFACTS.md). Present invalid data and numerical
failures must not be converted into skips. Run only the selected focused check;
routine tests do not stop services or acquire resource leases.

## Capabilities and runtime admission

| Scope | Evidence / limit |
|---|---|
| XPU substrate, SmallM/W8A8 and ownership | Focused synthetic and real-operand tests; model-owned direct preparation preserves full declared intermediates and lifetime behavior. Public workspaces retain their layout. |
| Eager target, held-out P128/D1 | Two complete hybrid-state comparisons pass; this is not integrated MTP/C4/long-context state qualification. |
| Public HTTP server | Target-only and MTP3 greedy text/usage, template, streaming, C1–C4, cancellation/EOS/reuse smoke checks pass. MTP draft counters advance. Streaming and MTP logprob parity remain unqualified. |
| Prefix observability over HTTP | Aligned4K functional trace restores1600 tokens; the cached-token metric check fails. Keep this separate from native32K position/ID checks. |
| Native MTP3, graphs and request lifecycle | Native regression/ownership checks pass in the tested scopes. Integrated original/native C1 and C4→C2→C1 state comparisons fail; keep the verifier experimental. |
| P32768/O64 prefix | Each implementation reproduces its own cold output with30400 cached prompt tokens. Native and original outputs differ; no comparable native/original prefix timing is established. |
| Default reference | Frozen D27/D29 TV/KL gates remain failed. A separately controlled arithmetic reference is a separate label, not a replacement golden. |
| Maximum context / sampling | 262144 is configured, not a full262K generation proof. Stochastic/RNG equivalence is unqualified. |

The guard index is the implementation authority, rather than a universal
hardware support claim: `PagedAttentionKernel` in `xpu_attention.cpp` selects
the route; `PagedAttentionXe2VerifyQueryLength` and
`PagedAttentionXe2VerifyKernel` validate C1 Q2–Q5 or uniform C2/C3/C4 Q4,
FP16/FP8 layouts, metadata, page, device, compiler and driver contracts.
Nonuniform verification does not qualify for that compiled route.
`PagedAttentionXe2DecodeKernel`, `PagedAttentionPrefillKernel`,
`GdnPrefillKernel`, `GdnSpecDecodeKernel` and the FP16 producer wrappers own
their distinct guards. The measured page protocol is1600; a source predicate
accepting another layout is not model qualification for it. Guard refusal uses
an available fallback or an explicit diagnostic error; never remove guards to
advertise another device. `VT_XPU_XE2_VERIFY=1` is not ordinary-route admission.

Direct model-owned W8A8 preparation defaults ON under its checked certificate,
K/128<=144 and64MiB preparation budget. `VT_XPU_W8A8_DIRECT_PREPARE=0` is the
diagnostic publication fallback. This retains the original completion fence;
the measured event-lease variant was rejected and is not implemented.

## Diagnostic controls

Only two `VT_B70_*` names occur in product sources at this checkpoint:

| Control | Classification / behavior |
|---|---|
| `VT_B70_FAST_TOPK20` | Runtime diagnostic override. Unset enables the existing guarded top20/top-p attempt;0 selects the general path. This does not establish sampled-RNG parity. |
| `VT_B70_FAST_TOPK_TRACE` | Optional stderr trace of the top20 fallback witness; ordinary inference leaves it unset. |

The other `VT_B70_*` capture/model/fixture/report names are test/tool inputs,
not engine configuration. `VT_B70_EXL3_ENGINE_*`, `VT_B70_EXL3_MODEL`, SmallM
fixture names and serving/profile controls belong to their individual test
executables or tools. Dumps can be large and do not prove in-flight safety.
`VT_XPU_PROFILE` and `VT_XPU_HOST_PROFILE` enable measurement instrumentation;
serving scores require them OFF. Do not treat a profiling score, a native
regression result or an optional external skip as reference qualification.
