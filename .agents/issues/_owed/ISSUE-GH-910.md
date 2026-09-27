ID: ISSUE-GH-910
Title: Our on-device argmax resolves an EXACT logit tie toward the HIGHER token id where `torch.argmax` takes the lower, so vLLM and vllm.cpp deterministically pick different tokens at any tied position. Found while adjudicating the 35B bf16 MoE token gate ([#864](https://github.com/mudler/vllm.cpp/issues/864)): one position in 112 diverged on a bit-identical logprob `-0.8293954133987427`, `top2_gap_mnats = 0.0`, which PASSES the ratified near-tie doctrine and still costs a STRICT gate that would otherwise read 7/7. Benign here, permanent everywhere, and a behavioral divergence from the reference we mirror. Needs a RED-first test that constructs a real tie — one asserting only "argmax returns a maximum" passes both conventions — plus inertness across the existing greedy goldens. Listed under `## Owed` in [`moe-bf16-tower-arms.md`](../../specs/moe-bf16-tower-arms.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 910
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:240`

### Frozen archive evidence

> | [#910](https://github.com/mudler/vllm.cpp/issues/910) | — | Our on-device argmax resolves an EXACT logit tie toward the HIGHER token id where `torch.argmax` takes the lower, so vLLM and vllm.cpp deterministically pick different tokens at any tied position. Found while adjudicating the 35B bf16 MoE token gate ([#864](https://github.com/mudler/vllm.cpp/issues/864)): one position in 112 diverged on a bit-identical logprob `-0.8293954133987427`, `top2_gap_mnats = 0.0`, which PASSES the ratified near-tie doctrine and still costs a STRICT gate that would otherwise read 7/7. Benign here, permanent everywhere, and a behavioral divergence from the reference we mirror. Needs a RED-first test that constructs a real tie — one asserting only "argmax returns a maximum" passes both conventions — plus inertness across the existing greedy goldens. Listed under `## Owed` in [`moe-bf16-tower-arms.md`](../../specs/moe-bf16-tower-arms.md) | bug |

## Resolution

-
