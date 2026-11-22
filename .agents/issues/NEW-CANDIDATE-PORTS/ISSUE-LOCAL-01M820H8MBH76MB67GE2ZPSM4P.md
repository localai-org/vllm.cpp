ID: ISSUE-LOCAL-01M820H8MBH76MB67GE2ZPSM4P
Title: Gateability probe: build and run vLLM inkling2_loop at ab18334a4 on a leased GPU
Row: NEW-CANDIDATE-PORTS
State: OPEN
Kind: task
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-22
Updated: 2026-11-22
Closed: -

## Problem

Upstream vLLM PR #69028 (merge ab18334a4f60da74db7de89fe95594e5ed0cda36, 2026-11-21) implements Inkling2LoopForConditionalGeneration nvidia-only (csrc/inkling2/*.cu; kernel selection branches only on device capability, no CPU path in vllm/models/inkling2/nvidia/ops/__init__.py). The gateability probe for the pin advance recorded in specs/inkling2-vllm-pin-advance.md therefore cannot run on a CPU-only host: gateable=no as of 2026-11-22. Discharge: take a GPU lease, build vLLM from source at ab18334a4, download thinkingmachines/Inkling-2-Mini HF safetensors (~26 GB, ungated), and demonstrably RUN greedy generation on a prompt. A config construction alone proves nothing. Record the measured parity-pin identity strings for the eventual sync cycle.

## Resolution

-
