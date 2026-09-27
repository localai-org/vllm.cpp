ID: ISSUE-GH-1466
Title: **`tests/vllm/models/test_muse_glimmer_text.cpp:532`'s `CHECK(diff <= 5e-4)` is a rounded-up W1 measurement, and one correct kernel change already spent 45 points of its 76-point margin.** `3a54c4b7d`'s own body quotes the number the constant was rounded up from ("max abs diff 1.21e-4 on logits of max 4.88e-2"); there is no derivation beside it. `4712dac40` narrowed `act(gate)` to the input dtype — upstream's polarity, and the only form that reproduces the committed `silu_and_mul_bf16_8x256` oracle golden bit-exactly — and grew the envelope 2.8x to 3.43e-04, which is 0.687 of the bound (measured at `aeba0de6f`, CPU-only Release, x86_64). It is the last member of that class in this file: [#1458](https://github.com/mudler/vllm.cpp/issues/1458) replaces the other one (`bdiff <= 1e-5`) in the same case and leaves this one to its own derivation, because a bigger constant is not a repair. Found in flow while gating #1458 (PR [#1461](https://github.com/mudler/vllm.cpp/pull/1461)); also listed under `## Owed` in [`muse-glimmer.md`](../../specs/muse-glimmer.md)
Row: MODEL-MUSE-GLIMMER
State: UNKNOWN
Kind: bug
GitHub: 1466
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:515`

### Frozen archive evidence

> | [#1466](https://github.com/mudler/vllm.cpp/issues/1466) | `MODEL-MUSE-GLIMMER` | **`tests/vllm/models/test_muse_glimmer_text.cpp:532`'s `CHECK(diff <= 5e-4)` is a rounded-up W1 measurement, and one correct kernel change already spent 45 points of its 76-point margin.** `3a54c4b7d`'s own body quotes the number the constant was rounded up from ("max abs diff 1.21e-4 on logits of max 4.88e-2"); there is no derivation beside it. `4712dac40` narrowed `act(gate)` to the input dtype — upstream's polarity, and the only form that reproduces the committed `silu_and_mul_bf16_8x256` oracle golden bit-exactly — and grew the envelope 2.8x to 3.43e-04, which is 0.687 of the bound (measured at `aeba0de6f`, CPU-only Release, x86_64). It is the last member of that class in this file: [#1458](https://github.com/mudler/vllm.cpp/issues/1458) replaces the other one (`bdiff <= 1e-5`) in the same case and leaves this one to its own derivation, because a bigger constant is not a repair. Found in flow while gating #1458 (PR [#1461](https://github.com/mudler/vllm.cpp/pull/1461)); also listed under `## Owed` in [`muse-glimmer.md`](../../specs/muse-glimmer.md) | bug |

## Resolution

-
