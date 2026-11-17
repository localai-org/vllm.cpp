# Experimental native EXL3 image serving on XPU

Native PNG/JPEG image requests, streaming, two ordered images and image-aware
MTP3 are implemented for the measured Qwen3.8-27B EXL3 checkpoint on Intel Arc
Pro B70. The repaired FP32 vision LayerNorm and Xe2 attention pass the four
frozen full-tower geometry gates with unchanged relative L2<=0.003 and maximum
absolute<=0.05 limits. Bounded public HTTP/C-ABI, MTP, cache reuse and cancellation
qualification passed on the selected native product. Greedy outputs can differ
across batch shapes; supported same-geometry isolation and correct MTP retained
states remain required. Comparable shape dependence was observed independently
in the pinned Python reference. This is not cross-runtime tensor equality,
all-shape determinism or complete Python vision-feature parity.
The existing [EXL3 XPU guide](EXL3_XPU.md) remains authoritative for text serving,
model revision, compact draft-map identity, dependency pins and inherited limits.

## Build dependencies

Use the oneAPI/XPU, oneDNN 3.13.0 and pinned SYCL-TLA environment described in
the EXL3 guide. Image containers additionally require libpng >= 1.6 and a
libjpeg-compatible development package. CMake finds `PNG::PNG` and `JPEG::JPEG`
locally; it does not fetch these codecs. Retain their original dependency
license notices when distributing binaries; see [NOTICE](../NOTICE).

The executed codec builds used GCC 16.2.1 with libpng 1.6.58 / libjpeg-turbo
3.2.0 on the host, and oneAPI 2026.1.1 with libpng 1.6.43 / libjpeg-turbo 2.1.5
in the XPU builder. These identify tested dependency combinations, not a
qualification of all codec versions or other devices.

From the repository root, with dependency paths supplied by the builder:

```sh
cmake -S . -B build-xpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx \
  -DVLLM_CPP_XPU=ON -DVLLM_CPP_XPU_ONEDNN=ON \
  -Ddnnl_DIR=/path/to/onednn-install/lib/cmake/dnnl \
  -DVLLM_CPP_SYCL_TLA_DIR=/path/to/sycl-tla \
  -DVLLM_CPP_XPU_XE2_GDN=ON -DVLLM_CPP_XPU_XE2_PREFILL=OFF \
  -DVLLM_CPP_XPU_GPTQ4=OFF -DVLLM_CPP_IMAGE_CODECS=ON \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_HIP=OFF -DVLLM_CPP_VULKAN=OFF \
  -DVLLM_CPP_METAL=OFF -DVLLM_CPP_TENSTORRENT=OFF \
  -DVLLM_CPP_SERVER=ON -DVLLM_CPP_BUILD_EXAMPLES=ON \
  -DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_HF_DOWNLOAD=OFF \
  -DVLLM_CPP_WITH_DIARIZATION=OFF
cmake --build build-xpu --target vllm-server native-vision-chat -j2
```

Without `VLLM_CPP_IMAGE_CODECS=ON`, PNG/JPEG requests are explicitly refused.
With the pinned SYCL-TLA checkout and oneDNN enabled, FP16 Xe2 vision attention
is built independently of `VLLM_CPP_XPU_XE2_PREFILL`. The supported B70
device/layout automatically uses it; other device/layout admissions retain
oneDNN. No vision route environment selector is needed.
The native inference process has no Torch/Python dependency. Python below is
only a client or a reference-fixture generator. Learned tower, merger, target
and draft execution remain on XPU; media decode/resize run on the host.

## Bounded request envelope

| Property | Implemented or exercised scope |
|---|---|
| Input | PNG/JPEG data URIs; tested 8-bit color/grayscale/palette, transparency and EXIF orientation cases |
| Images | At most two still images per request, preserving conversation order |
| Container admission | At most 32 MiB decoded container bytes, 16777216 decoded pixels and 32768 pixels on either axis |
| Processed image | Effective area ceiling 4194304 pixels, at most 16384 patches / 4096 merged visual rows |
| Tower | 27 blocks, FP16 execution; stored BF16 parameters are numerically converted to FP16 |
| Language state | FP16 activations, FP32 recurrent SSM state, FP8 paged attention KV |
| Scheduling | KV page1600; exercised token budgets1600 and4096; bounded C1/C2/C4 admission |
| Deferred | Remote/file URLs, animated images, video/audio, 16-bit PNG and unqualified JPEG precision/color spaces |

A C4 lifecycle result does not establish that four maximum images and contexts
fit simultaneously. The text model-length setting is not qualification of a
262K image-conditioned conversation. Malformed containers or unsupported
processor geometry produce client errors before scheduled model work.

## HTTP and streaming

Load the same native EXL3 runtime environment as the text-serving recipe,
including its actual oneDNN library path and verified compact draft map for
MTP. A bounded image-serving launch uses:

```sh
build-xpu/examples/vllm-server \
  --model /path/to/model --served-model-name qwen38-exl3 \
  --host 127.0.0.1 --port 8000 --device auto \
  --block-size 1600 --num-blocks 180 --max-model-len 262144 \
  --max-num-seqs 4 --max-num-batched-tokens 1600 --kv-cache-dtype fp8 \
  --no-enable-prefix-caching --no-enable-thinking \
  --limit-mm-per-prompt '{"image":2,"video":0}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

This CLI uses `--device auto` for the XPU-only build. Target-only mode omits
`--speculative-config`; MTP3 additionally requires the explicit verified
`EXL3_DRAFT_VOCAB` path from the EXL3 guide. Prefix-cache lifecycle checks are
separate from this uncached launch example.

Create a request using a checked-in small image, then send it to the server:

```sh
python3 - <<'PY'
import base64, json
from pathlib import Path
raw = Path('tests/fixtures/native_vision_http/orbit.png').read_bytes()
body = {
    'model': 'qwen38-exl3',
    'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {
            'url': 'data:image/png;base64,' + base64.b64encode(raw).decode()}},
        {'type': 'text', 'text': 'Read the heading and describe the shapes.'}]}],
    'temperature': 0, 'max_tokens': 64,
    'chat_template_kwargs': {'enable_thinking': False}}
Path('vision-request.json').write_text(json.dumps(body))
PY
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @vision-request.json
```

For SSE, add `"stream": true` and
`"stream_options": {"include_usage": true}` to the request and use `curl -N`.
A second `image_url` content part supplies a second image; retain the intended
order of image/text parts across all messages. HTTP and the exported C ABI
share the same media decode, processor and serving path.

## Focused reproduction

The existing source-based HTTP client accepts an already running server and
never manages services. It checks four PNG/JPEG functional fixtures and one
SSE case, usage and drained scheduler gauges. It does not prove numerical,
MTP, mixed-request or throughput parity:

```sh
python3 scripts/mm/validate_native_exl3_vision_http.py \
  --url http://127.0.0.1:8000 --model qwen38-exl3 \
  --output /path/to/new-vision-http-result.json
```

The `native-vision-chat` example links only the exported shared C ABI and checks
ordinary/streamed chat from a bounded JSON request bundle. Generate the existing
portable request bundle using the checkpoint directory basename, then run each
mode in a separate process with the same native runtime environment:

```sh
python3 scripts/mm/generate_native_exl3_vision_capi_requests.py \
  --model CHECKPOINT_DIRECTORY_BASENAME --output capi-requests.json
build-xpu/examples/native-vision-chat /path/to/model capi-requests.json target.json 0
build-xpu/examples/native-vision-chat /path/to/model capi-requests.json mtp.json 3
```

Internal attribution utilities are separate from this public interface.

Focused host targets include `test_native_image_codec`, `test_torchvision_resize`
and `test_qwen3vl_processor`. XPU operator/tower/MTP targets are
`test_xpu_vision_ops`, `test_xpu_vision_tower` and `test_xpu_vision_mtp`.
Synthetic cases are runnable independently. External model/fixture cases
report framework skips when inputs are optional; required qualification uses
`EXL3_REQUIRE_ARTIFACTS=1` or `VLLM_CPP_REQUIRE_EXL3_TEST_ARTIFACTS=ON` and fails
on missing/corrupt inputs. A skipped case is not a numerical pass.

Use the checked-in [tower capture](../scripts/mm/capture_native_exl3_vision_tower_reference.py)
and [block capture](../scripts/mm/capture_native_exl3_vision_block_reference.py)
tools with their explicit model/runtime arguments for external references.
Record the actual model/config/processor, compiler, dependency, source and
executable hashes with each result. Keep failed tower/strict greedy checks
under their original names; no semantic success or private reference injection
qualifies a production numerical contract.

The bounded alpha attribution tools use a compact
[public pin manifest](../tests/fixtures/native_vision_attribution/README.md)
instead of development receipt capsules. Their original large captures remain
explicit external inputs; missing or changed artifacts fail. The focused
model-free `test_native_vision_attribution_pins` packaging test does not load a
GPU, checkpoint or Python inference runtime.

## Measured bounded qualification

The 2026-10-09 B70 qualification used the default selected vision route, FP8
paged language KV and the existing compact MTP3 mapping. It exercised public
PNG/JPEG, streaming and two ordered images in target-only and MTP3 arms; warm
C2 and held-out C4; partial-image prefill/decode cancellation with unaffected
siblings and ordered retries; A/A/B/A prefix reuse; and maximum/small eviction.
After engine free the public C-ABI check observed zero graphs and live pool
blocks. Repeated same-phase cache/pool observations reached a stable plateau;
intentionally resident contexts and free scratch pools were accounted separately.

| Original tower geometry | Merger relative L2 | Maximum absolute |
|---|---:|---:|
| Aligned 512x384 | 0.000021037 | 0.00390625 |
| Resized 513x385 | 0.000018182 | 0.00390625 |
| Portrait 512x768 | 0 | 0 |
| Maximum 2048x2048 | 0 | 0 |

A paired native pre-vision P4096/O256,C1,MTP3 text sentinel preserved all four
responses,usage and public token representations. The last-three-sample warm
median was 6.592s versus 6.566s overall (+0.39%),2.271s versus 2.264s prefill,
and 4.321s versus 4.297s decode. The post-first-token rate was about59.0 versus
59.3 tokens/s including MTP3. This is one bounded native text regression check,
not a Python throughput comparison or an image-prefill benchmark. Broader
text-reference/performance limits remain those of the existing EXL3 guide.

Fresh selected full-state MTP captures used actual accepted0 packets; previous
actual accepted1/2 and forced-selector controls remain separately scoped
mechanism evidence. Historical strict mixed-token failures remain diagnostics.
The qualified cases do not establish all-context,image-answer,stochastic or
complete accepted3-state parity.
