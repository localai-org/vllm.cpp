ID: ISSUE-GH-2191
Title: **`hidden_act` is not mirrored, so a non-silu dots3-note config runs SwiGLU silently where vLLM raises.** `DeepseekV2MoE.__init__` refuses anything but silu before it builds a thing — `deepseek_v2.py:310-314` @ pin `bc2d63e650`, `ValueError(f"Unsupported activation: {config.hidden_act}. Only silu is supported for now.")` — and `Dots3NoteLanguageModelForCausalLM` (`model.py:681`) subclasses `DeepseekV32ForCausalLM`, so that is the `__init__` W5 ports. `grep -c hidden_act` over `dots3_note.cpp` and `dots3_note.h` is **0**: the key is never parsed, `Dots3NoteParams` has no field for it, and `Dots3NoteDeviceRefusal` never mentions it, so `hidden_act: "gelu"` loads and runs `vt::MoeSiluMul` / `vt::MoeGroupedGemmBf16GateUpSilu` with no refusal. Same hole in `deepseek_v2.cpp` and `deepseek_v2_weights.cpp` (`grep -c` = 0 on both), so it is a MIRROR GAP inherited by both ports rather than a W5 regression; `parakeet_transducer.cpp:108-112` is the shape this owes, throwing by name for anything but its one ported activation. The released `config.json` carries `"hidden_act": "silu"`, so nothing shipped is affected. Owed: parse it and refuse by name mirroring upstream's message; whether the same guard lands on the SACRED `deepseek_v2.cpp` path is a separate decision with its own red-before evidence. Found by the fresh review of [#2187](https://github.com/mudler/vllm.cpp/pull/2187) as F6. Under `## Owed` in [specs/dots3-note.md](../../specs/dots3-note.md)
Row: MODEL-MM-dots3-note-dots3-note-for-causal-lm
State: UNKNOWN
Kind: bug
GitHub: 2191
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:849`

### Frozen archive evidence

> | [#2191](https://github.com/mudler/vllm.cpp/issues/2191) | `MODEL-MM-dots3-note` | **`hidden_act` is not mirrored, so a non-silu dots3-note config runs SwiGLU silently where vLLM raises.** `DeepseekV2MoE.__init__` refuses anything but silu before it builds a thing — `deepseek_v2.py:310-314` @ pin `bc2d63e650`, `ValueError(f"Unsupported activation: {config.hidden_act}. Only silu is supported for now.")` — and `Dots3NoteLanguageModelForCausalLM` (`model.py:681`) subclasses `DeepseekV32ForCausalLM`, so that is the `__init__` W5 ports. `grep -c hidden_act` over `dots3_note.cpp` and `dots3_note.h` is **0**: the key is never parsed, `Dots3NoteParams` has no field for it, and `Dots3NoteDeviceRefusal` never mentions it, so `hidden_act: "gelu"` loads and runs `vt::MoeSiluMul` / `vt::MoeGroupedGemmBf16GateUpSilu` with no refusal. Same hole in `deepseek_v2.cpp` and `deepseek_v2_weights.cpp` (`grep -c` = 0 on both), so it is a MIRROR GAP inherited by both ports rather than a W5 regression; `parakeet_transducer.cpp:108-112` is the shape this owes, throwing by name for anything but its one ported activation. The released `config.json` carries `"hidden_act": "silu"`, so nothing shipped is affected. Owed: parse it and refuse by name mirroring upstream's message; whether the same guard lands on the SACRED `deepseek_v2.cpp` path is a separate decision with its own red-before evidence. Found by the fresh review of [#2187](https://github.com/mudler/vllm.cpp/pull/2187) as F6. Under `## Owed` in [specs/dots3-note.md](../../specs/dots3-note.md) | bug |

## Resolution

-
