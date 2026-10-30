ID: ISSUE-LOCAL-01M5QBGBM3Z7NT3D9TA0K1HT7T
Title: FlintModel::Forward memcpys bf16 embedding bits into an f32 buffer
Row: MODEL-TEXT-FLINT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-24
Updated: 2026-10-24
Closed: 2026-10-24

## Problem

The W1 CPU reference forward (commit ea5c3dade) loads the embedding row with std::memcpy(&x[i*h], emb + tok*h, h*sizeof(uint16_t)) from the bf16 embed_tokens into the f32 hidden stream. That copies RAW bf16 BITS into the low half of each float, leaving the high half zero — every embedding becomes a ~1e-42 denormal, the residual stream starts at garbage, and the whole forward is noise. The W1 synthetic gate passed because its expected probe values were derived through the same conversion (circular), and the synthetic config's bf16 bit patterns happen to look like plausible small floats at the harness's tolerance. Found at the flint GGUF real-weights token gate: 8/8 golden chains diverged at step 0 with gaps up to 17 nats while the pinned llama-cpp-flint oracle and an independent transformers (FlintForCausalLM, weights dequanted from the same container) reference both answer 12834 (' Paris'). Fix: convert each element, x[i*h+d] = Bf16(emb[tok*h+d]).

## Resolution

Fixed 2026-10-24 in commit 299592c36: the embedding lookup converts bf16 to f32 element-wise; adjudicated against the pinned oracle and a transformers FlintForCausalLM reference, both top-1 12834 (' Paris').
