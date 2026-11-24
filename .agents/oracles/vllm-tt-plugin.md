# `vllm-tt-plugin` — vLLM's Tenstorrent platform plugin

Announced 2026-09-07
([blog](https://vllm.ai/blog/2026-09-07-vllm-tt-plugin)):
Tenstorrent accelerators reach vLLM through the standard out-of-tree platform
plugin (`vllm.platform_plugins` entry point `tt`), active whenever `ttnn` is
importable. This supersedes the premise this repository recorded until now —
that vLLM has no Tenstorrent platform anywhere (`tt-forge.md`,
[`../porting-inventory.md`](../porting-inventory.md) §15). Where the plugin
serves, the answer it produces is vLLM's answer, and the
`BACKEND-TENSTORRENT` row is no longer an extension with "no upstream analog":
`TTQwen3_5ForConditionalGeneration` covers the exact architecture family the
GSQ and APEX rows port.

```oracle-pin
id = vllm-tt-plugin
role = secondary
upstream = https://github.com/tenstorrent/vllm-tt-plugin
scope = the qwen35 (Qwen3.5/3.6) architecture and the other TT-prefixed architectures the plugin registers, served on Tenstorrent mesh devices through hand-written TTNN implementations that ship in tt-metal; the weights are HF checkpoints, so this is NOT a GGUF-decode reference
pin = 6a3482a660bd31b51b49b5079e7050d58a453b05
pin_label = main HEAD, 2026-11-24; source read only, not built or served
pinned_on = 2026-11-24
gateable = no
evidence = #3261
```

## What it is, and what it is not

- **The model code lives in tt-metal**, not the plugin: each registered
  architecture is a vLLM-facing wrapper around a hand-written TTNN
  implementation. That is the same tt-metal stack whose native qwen35 numbers
  are the performance denominator the Tenstorrent rows benchmark against, so
  plugin-on-P150 is the closest thing to an apples-to-apples
  primary-oracle-rank reference this repository can get for device behavior.
- **Weights are HF checkpoints.** The plugin registers architectures and maps
  HF weights; GGUF is served by a DIFFERENT plugin
  ([`vllm-gguf-plugin.md`](vllm-gguf-plugin.md), CUDA/ROCm-side). A combined
  GGUF-on-TT serving path is not established by either plugin, so for the
  sub-bit GGUF decode arms the pinned llama.cpp exact oracle
  ([`llama-cpp.md`](llama-cpp.md)) stays the decode reference. What this
  plugin denominates is the ARCHITECTURE on the device: graph correctness at
  bf16 residency, per-layer logits, scheduling, sampling, and speed.
- **Setup is from source**: a vLLM build with `VLLM_TARGET_DEVICE=empty`
  against vLLM 0.26.0 at `7250ddfaa9`, and against vLLM 0.32.0 at the
  current pin, inside a tt-metal environment, then the plugin install
  script (blog, "Try it out"). `TTPlatform` selects only when `ttnn` imports,
  so the install cannot hijack a CUDA/ROCm environment.

## First gateability attempt (2026-09-21, FAILED — reproducible warmup hang)

Environment assembled on the row's own aarch64 host: vLLM 0.26.0
(`VLLM_TARGET_DEVICE=empty`) + torchvision 0.26.0 CPU + the plugin at
`7250ddfaa9`, the pin at that date, in the tt-metal python_env; Qwen/Qwen3.5-9B downloaded
(17 GB); `TTQwen3_5ForConditionalGeneration` registers and the TT platform
activates; the serve reaches tt-metal's qwen36 TTNN model, loads all
weights, compiles the decode trace, and then HANGS in
`capture_prefill_trace_chunked` (chunk=2048) — EngineCore pinned at ~100%
of one core with no log output for 45 and 63 minutes across two runs (the
second with the tensor cache already warm). tt-metal build:
v0.79.0-dev20260911-82-g81f3bbf3b40. Not yet bisected: whether the hang is
in the plugin's serve path, tt-metal's qwen36 warmup on this build, or an
insufficiently slow-but-finite compile (no introspection available —
ptrace_scope=1, no sudo). The tt-metal-side control
(`models/demos/blackhole/qwen36/demo/text_demo.py -k traced_128`) has not
been run.

## What this record does NOT license

`gateable = no`, owed to
[#3261](https://github.com/mudler/vllm.cpp/issues/3261) (local record
`ISSUE-LOCAL-01M32FVCE8XBW2PDTE48ZD8TWN`). The blog, the registry entry, and
the source read establish feasibility, not our gate. Measuring gateability on
the row's own P150 — one qwen35 checkpoint served, tokens emitted — is the
owed step, and it is ALSO
the row's performance denominator opportunity: our ~1800×→~385× gap figures
were computed against tt-metal's native qwen35 rate, and running OUR engine
and THAT stack side by side on the same card is the comparison the
benchmarking protocol has wanted all along.

## Deadlock characterized (2026-09-22, faulthandler stack capture)

Third attempt, with a faulthandler SIGUSR1 hook injected via
`PYTHONPATH=/tmp/fhhook sitecustomize.py` so the SPAWNED EngineCore child
registers it too. Stack dump at 27 minutes of silence:

- The main thread is blocked inside `ttnn.as_tensor` called from
  `models/demos/blackhole/qwen36/tt/gdn/weights.py:116 load_gdn_weights`
  (the fused `qkv_proj.weight` [4096,8192] bf16 -> bfloat8_b TILE
  conversion, `cache_file_name` set), during `initialize_vllm_model`.
- EVERY thread of the EngineCore process is in `futex_do_wait` — a
  tt-metal dispatch deadlock, not a slow compile. (Earlier "100% CPU"
  readings were the lifetime average; instantaneous state is all-sleep.)
- The deadlock point is NONDETERMINISTIC: attempt 1 got through weight
  load and decode-trace compile and blocked in the 2048-token prefill
  warmup; attempts 2 and 3 blocked earlier, during GDN weight load.
- The tt-metal sysmem warning ("using regular pages; pre-allocate
  hugepages") is present in every run.
- CONTROL RESULT (evidence in the prior section): tt-metal's own
  validated `traced_128` demo grinds identically (DEMO_EXIT=124 at 90
  min), so the deadlock is in the tt-metal python stack on this
  host/build — v0.79.0-dev20260911-82-g81f3bbf3b40, aarch64 — and NOT in
  the plugin's serve path. Our own C++-dispatched engine runs the same
  card without it.

Practical consequence: #3261 stays blocked until either the tt-metal
build is refreshed past the deadlock, hugepages are provisioned (needs
root; the warning names it), or upstream (tenstorrent/vllm-tt-plugin or
tt-metal) confirms a known fix. The stack capture above is the repro for
that report.

## Pin update (2026-11-24, source read only)

The pin moves from `7250ddfaa988cc7417f518266dce52e425745472` to
`6a3482a660bd31b51b49b5079e7050d58a453b05`, 81 commits later. Nothing was
built or served at the new pin: the developer disallowed the TT serve on this
host. Every measurement in this file above was taken at the earlier pin.

- **vLLM target.** The README and the install script target vLLM 0.32.0
  (`__version__ = "0.32.0"` in `src/vllm_tt_plugin/__init__.py`). Python must
  be 3.11 to 3.13. The `compat/vllm-0.25.1` and `compat/vllm-0.24.0` tags hold
  the last plugin state for those vLLM versions.
- **Model families registered at the new pin.** Llama 3.1, 3.2 and 3.3; Qwen
  2.5 and 3; Qwen 3.5; Qwen 2.5-VL and 3-VL; Mistral and Mistral 3; Gemma 3
  and 4; DiffusionGemma; DeepSeek V3; GPT-OSS.
- **No Inkling architecture.** `src/vllm_tt_plugin/platform.py`,
  `src/vllm_tt_plugin/model_registry.py`, `docs/MODEL_CAPABILITIES.md` and
  `README.md` contain no match for `inkling` at this pin. This covers Inkling-2
  and Inkling-2-Mini.
- **Kolibri-1 is partly present.** `src/vllm_tt_plugin/parsers/kolibri1.py`
  is a reasoning and tool-call parser, and `platform.py` registers
  `model_type: kolibri1` with `AutoConfig` so the config loads. Kolibri-1 is
  not in the README's list of supported model families.
- **Out-of-tree models.** `EXTRA_MODELS_DIR` registers a model from a bundle
  folder at startup, each holding a `vllm_metadata.json` and an adapter class.
  Inkling-2-Mini reaches a P150 this way: `Lottolabs/Inkling-2-Mini-TT-Mixed-BFP4-BFP8-P150`
  ships its own model code, a `vllm-tt-plugin-eligible.patch`, and a Docker
  image. That package is a third-party release, not a plugin feature. Its
  weights are BFP4 and BFP8, so it denominates that quantization on the
  device, not BF16.
- **Gateability is unchanged.** `gateable = no` still stands, and
  ISSUE-LOCAL-01M32FVCE8XBW2PDTE48ZD8TWN (#3261) still owns the first served
  token at a pinned plugin revision.

ISSUE-LOCAL-01M87G9R15Y8D7FQD9YC2V8J8X owns this update.
