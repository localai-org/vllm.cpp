ID: ISSUE-LOCAL-01M87G9R15Y8D7FQD9YC2V8J8X
Title: vllm-tt-plugin oracle pin is two months stale and omits the Inkling-2 route
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: docs
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-24
Updated: 2026-11-24
Closed: 2026-11-24

## Problem

The vllm-tt-plugin oracle file pins 7250ddfaa988cc7417f518266dce52e425745472 (main HEAD 2026-09-21, built against vLLM 0.26.0). Upstream main is 81 commits ahead at 6a3482a660bd31b51b49b5079e7050d58a453b05 (2026-11-24) and now targets vLLM 0.32.0 (plugin __version__ 0.32.0). The record does not say what the plugin registers at the current head, so two facts were re-derived by hand this session: the plugin registers no Inkling architecture at 6a3482a660bd (zero matches for inkling in platform.py, model_registry.py, docs/MODEL_CAPABILITIES.md and README.md), and it carries Kolibri-1 only as a reasoning and tool-call parser plus the AutoConfig hook for model_type kolibri1. Inkling-2-Mini reaches a P150 through the EXTRA_MODELS_DIR bundle mechanism with third-party model code and a patched plugin (Lottolabs/Inkling-2-Mini-TT-Mixed-BFP4-BFP8-P150), not through the upstream plugin. The plugin was not run at the new pin: the developer disallowed the TT serve on this host, so gateable stays no and the gateability debt of ISSUE-LOCAL-01M32FVCE8XBW2PDTE48ZD8TWN (#3261) is unchanged.

## Resolution

Resolved 2026-11-24 by the record edit in .agents/oracles/vllm-tt-plugin.md: the oracle pin block now names 6a3482a660bd31b51b49b5079e7050d58a453b05 (main HEAD, 2026-11-24), and a dated section records the vLLM 0.32.0 target, the registered model families, the absence of any Inkling architecture, the partial Kolibri-1 support, and the EXTRA_MODELS_DIR route by which Inkling-2-Mini reaches a P150 through third-party model code. Source read only: nothing was built or served at the new pin, so gateable stays no and ISSUE-LOCAL-01M32FVCE8XBW2PDTE48ZD8TWN (#3261) is unchanged.
