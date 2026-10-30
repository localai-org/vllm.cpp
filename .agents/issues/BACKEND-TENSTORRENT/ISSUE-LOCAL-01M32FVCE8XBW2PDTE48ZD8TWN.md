ID: ISSUE-LOCAL-01M32FVCE8XBW2PDTE48ZD8TWN
Title: Measure vllm-tt-plugin gateability on the row's P150 (qwen35 served, tokens emitted)
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: 3261
Mirror: SYNCED
Availability: FULL
Created: 2026-09-21
Updated: 2026-10-21
Closed: -

## Problem

The vllm-tt-plugin oracle record (.agents/oracles/vllm-tt-plugin.md, pin 7250ddfaa, 2026-09-21) is filed with gateable = no: nothing from the plugin has executed in this repository. The oracle-registry checker requires an owing issue named on a gateable=no record; this is it. Owed measurement: install vLLM 0.26.0 (VLLM_TARGET_DEVICE=empty) plus the plugin at the pinned HEAD inside a tt-metal environment on the row's P150, serve one registered qwen35 checkpoint, and record tokens emitted. Establishing gateability also unlocks the primary-rank denominators the Tenstorrent rows want: on-device qwen35 graph correctness at bf16 (vs our GGUF-decoded bf16 arm, per-layer like the llama.cpp bisect) and the native ~50 tok/s qwen35 rate measured side by side on the same card. Source: vllm.ai blog 2026-09-07; supported architectures include TTQwen3_5ForConditionalGeneration. NOT established by this measurement: GGUF-decode parity (the plugin serves HF weights; llama.cpp stays the GGUF decode oracle).

## Resolution

2026-10-21: NOT VERIFIABLE OFFLINE; the measurement is still owed. The oracle
record is unchanged: `.agents/oracles/vllm-tt-plugin.md` still carries
`gateable = no` on pin 7250ddfaa (2026-09-21), and it names this issue as the
owing issue, which is what the registry checker requires of a gateable=no
record.

What settles it: the measurement this issue already specifies -- install vLLM
0.26.0 (VLLM_TARGET_DEVICE=empty) plus the plugin at the pinned HEAD inside a
tt-metal environment on the row's P150, serve one registered qwen35 checkpoint,
and record the tokens emitted. It needs the device and the plugin install, so
the tree cannot settle it.
