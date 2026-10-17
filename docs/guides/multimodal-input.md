# Send multimodal input

The OpenAI-compatible server routes supported image and audio content parts
through `/v1/chat/completions` to the model. This guide covers these examples:

| Server architecture | Image items per prompt | Audio items per prompt | Video |
|---|---:|---:|---|
| `Qwen3VLForConditionalGeneration` | 1 | 0 | Refused |
| `Dots3NoteForCausalLM` | 512 | 128 with `audio_config`; otherwise 0 | Refused |

These are implementation ceilings. `--limit-mm-per-prompt` can lower them;
`--language-model-only` sets them to zero. Dots3-note can mix images and audio
in one prompt. Qwen3-VL serves one sequence per model step.

CPU tests exercise the serving path with synthetic weights. They establish
that media reaches the model, not token parity on real checkpoints:
[Qwen3-VL tests](../../tests/vllm/entrypoints/openai/test_api_server_mm_forward.cpp)
and [dots3-note tests](../../tests/vllm/entrypoints/openai/test_api_server_dots3_mm_forward.cpp).
No real-checkpoint token-parity or accelerator-performance claim follows from
these tests. Other multimodal architectures do not gain HTTP support from
registration alone. The text CLI does not accept these media requests.
The C API uses the same multimodal chat handler as the server. See
[C API limits](#c-api-limits).

## Send an image

The server accepts square RGB images as packed, unsigned 8-bit bytes, with
three channels per pixel and no file header. Use `image/x-raw-rgb` in the data
URI. PNG and JPEG files are refused; changing their MIME type does not convert
them to raw RGB.

Start a server with a local Qwen3-VL safetensors checkpoint:

```sh
./build/examples/vllm-server \
  --model /models/Qwen3-VL-4B-Instruct \
  --served-model-name qwen3-vl --port 8000
```

Save a square raw RGB image as `image.rgb`. For a 448 by 448 image, the file
must contain exactly 602,112 bytes. Run this request with Python's standard
library; it reads the image from disk and sends its bytes:

```python
import base64
import json
import math
from pathlib import Path
from urllib.request import Request, urlopen

rgb = Path("image.rgb").read_bytes()
side = math.isqrt(len(rgb) // 3)
if side == 0 or side * side * 3 != len(rgb):
    raise ValueError("image.rgb must be a square HxWx3 raw RGB buffer")

image_url = "data:image/x-raw-rgb;base64," + base64.b64encode(rgb).decode("ascii")
payload = {
    "model": "qwen3-vl",
    "max_tokens": 32,
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]}],
}
request = Request(
    "http://localhost:8000/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urlopen(request) as response:
    print(json.load(response))
```

Dots3-note uses the same image format. Its audio path accepts `input_audio`
or `audio_url` parts carrying PCM16 RIFF/WAVE data. It mixes channels to mono
and resamples to the checkpoint's configured sample rate. MP3, FLAC, and Ogg
are refused. Parsing a part type does not establish model support: both
registered chat paths refuse `video_url`.

## Add a `clip` multimodal projector to GGUF

The Qwen3-VL projector path uses two files: the language `.gguf` and a
`clip`-architecture `mmproj-*.gguf` carrying the vision tower. Name the second
one with `--mmproj` (`vllm-server`) or `vllm_model_params.mmproj_path` (C ABI
v22); it is never auto-discovered from a sibling filename, because a directory
holding two unrelated models must not silently fuse them.

```console
./build/examples/vllm-server \
  --model /models/Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj /models/mmproj-BF16.gguf
```

With default limits, the loader opens the projector and reads its `clip.*`
metadata and `v.*` and `mm.*` tensors. The engine then holds the same vision tower that the
safetensors path builds. No forward pass consumes this Qwen3-VL projector tower yet.
Neither the server nor the C API can use this projector path to produce an
image answer
([#821](https://github.com/mudler/vllm.cpp/issues/821)).

With default limits, the loader refuses these conditions before it reads the
tokenizer or language-model weights. Zero limits skip some tensor checks, as
explained under [Per-prompt input limits](#per-prompt-input-limits).

- `--model` is not a `.gguf`. A safetensors checkpoint carries its tower in its
  own shards and needs no projector file.
- the file's `general.architecture` is not `clip` (this is what you get for
  passing the language file twice).
- its `clip.projector_type` is not `qwen3vl_merger`. DeepSeek-V4 projectors use
  a separate loader. A `muse-glimmer` projector
  is routed to MuseGlimmer's own recorded refusal instead, which names the
  missing axis.
- it carries `v.patch_embd.weight` without `v.patch_embd.weight.1`. llama.cpp
  writes the temporal patch embedding as two halves; with one of them absent,
  loading would mean inventing the other, and the result would be a fluent,
  wrong model rather than an error.


The [checkpoint registry](../USAGE.md#checkpoint-registry) owns the exact file
sizes, repository revision, and checksums. The
[Qwen3.8 27B model recipe](../models/qwen3-8-27b.md) owns current arm support and
limitations. Loader and gate evidence remains in
[the Qwen3.8 quantized-arm spec](../../.agents/specs/qwen38-27b-quant-arms.md).

## Per-prompt input limits

Use `--limit-mm-per-prompt` to cap media items per prompt. The server uses the
lower of your configured limit and the architecture's ceiling.
For example, `--limit-mm-per-prompt '{"image": 99}'` still allows only one image
on Qwen3-VL. On dots3-note, it allows 99 images.

Requests that exceed a limit return HTTP 400. For example, a Qwen3-VL request
with two images receives:

```json
{"error":{"type":"BadRequestError",
          "message":"At most 1 image(s) may be provided in one prompt."}}
```

The error adds ``Set `--limit-mm-per-prompt` to increase this limit.`` only when
raising your configured limit would permit the request. The architecture's ceiling
still applies. The refusal does not otherwise distinguish a configured limit
from an unsupported modality.

### Disable media and skip unused towers

Use `--language-model-only` to set every modality limit to zero and refuse media
requests. You can also set individual limits to zero:

```sh
./build/examples/vllm-server \
  --model /models/Qwen3.5-4B \
  --limit-mm-per-prompt '{"image": 0, "video": 0}'
```

Loaders that implement tower skipping leave its tensors unread when every
modality served by that tower has a zero limit. These loaders include:

- Qwen3.5 dense safetensors, when the checkpoint contains a vision tower.
- Qwen3-VL safetensors.
- MuseGlimmer safetensors, when the checkpoint contains a vision tower.
- The Qwen3-VL tower in a `qwen3vl_merger` `clip` GGUF projector supplied through `--mmproj`.

These towers serve image and video. Set both limits to zero to skip them.
Setting only `image` to zero leaves the tower loaded, even when the server
cannot accept video requests. `--language-model-only` skips the same towers.
The dots3-note loader still loads its supported vision and audio towers at zero
limits.

The Qwen3.5 dense loader checks for vision tensor names before it skips the
tower. The skip bypasses both vision tensor loading and vision configuration
parsing. It does not add multimodal HTTP support for Qwen3.5 dense models.

For a Qwen3-VL `clip` projector, zero limits do not bypass all validation:

- The loader still opens the file, checks its architecture and projector type,
  resolves its geometry, and refuses tensors that its reader cannot consume.
- The loader skips tensor reads and their validation. For example, a missing
  `v.patch_embd.weight.1` fails at default limits but does not fail with
  `--language-model-only`.

The server reports towers that the loaded model actually skipped:

```text
server: multimodal limits language-model-only=ON audio=0 image=0 video=0
server: multimodal towers NOT loaded (every modality they serve is at limit 0): vision_tower
```

The tower message is absent when the loader skips nothing. Memory savings depend
on the checkpoint and loader. See [Memory benchmarks](../benchmarks/memory.md)
for the Qwen3-VL-4B load-time host-memory measurements, their history, and
unmeasured models. Those measurements do not establish serving memory usage or
VRAM savings.

### C API limits

`vllm_model_params.language_model_only` and `vllm_model_params.limit_mm_per_prompt`
configure the engine's limits. `vllm_model_params.mmproj_path` selects a projector.
`vllm_chat` and `vllm_chat_stream` use the server's multimodal chat handler and
codec. Pass media through the request JSON's content parts.

A registered multimodal handler processes the request subject to its limits.
An unregistered multimodal architecture, or a handler that cannot initialize,
refuses media with `VLLM_ERR_INVALID_ARGUMENT`. Read `vllm_last_error()` for the
architecture and missing capability. PNG, JPEG, and remote image fetching remain
unavailable with the default codec. `vllm_generate` accepts text only.
