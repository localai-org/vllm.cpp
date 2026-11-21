ID: ISSUE-LOCAL-01M7ZYAY4W7STQCBPSF6NQRF78
Title: NEW-CANDIDATE-PORTS (inkling2): enumerate the mudler/Inkling-2-Mini-APEX-GGUF release (arch inkling2_loop); support path owed
Row: NEW-CANDIDATE-PORTS
State: OPEN
Kind: task
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-21
Updated: 2026-11-21
Closed: -

## Problem

Developer downloaded ettore's new release 2026-11-21: mudler/Inkling-2-Mini-APEX-GGUF @ d85a58007995c3365eb2323b177937ce4d20b8f0 (Apache-2.0), mounted at /mnt/models/inkling2-mini-apex-gguf/ (Balanced arm, 20.68 GB, sha256 2c753df3cd190777... verified against the LFS pointer). GGUF v3, arch 'inkling2_loop' (custom), 704 tensors, 59 KVs; full walk committed at docs/bench-evidence/new-candidate-inkling2-20261121/apex-balanced-walk.txt. Config: 24 blocks, hidden 2304, ctx 1M, vocab 201024 (unpadded 200058), experts 128 top-8 scale 8.0, expert ff 1152, head_dim 128, sliding window 1024, shortconv l_cache 4, rope dim 0 (NoPE), attn_res block 6, softcap 30, relative attention (length 16, extent 1024, qk_clip 22.63, rel_clip 8), log-scale floor 8192 alpha 0.1, situ activation (beta 4.0 / linear_beta 25.0), loop start 6 end 18 count 3 gate_rank 64 varying=false. Tensor census: type ids {14 Q6_K x131, 0 F32 x478, 13 Q5_K x15, 8 Q8_0 x66, 7 Q5_1 x14} — ALL standard ggml ids, but OUR READER DOES NOT TABULATE id 7 (Q5_1; table jumps case 6 -> case 8 at gguf_reader.cpp:200+) — that is the first support step (plus id 3 Q4_1 for completeness). Name map: per-layer attn_{q,k,v,output,gate} + conv + rel + sink_proj + norms + ffn_norm + ffn_res_query/attn_res_query + exp_probs_b.bias + ffn_gate_inp + experts ffn_down_exps/ffn_gate_up_exps (mixed Q5_1/Q8_0/Q5_K/Q6_K by layer) + loop_* controller weights (12 core layers x q/k/v/ffn x gain/forget/down + loop_msg_query) — the looped-core message machinery. vLLM oracle: PR vllm#69028 MERGED 2026-11-21 (ab18334a4f60...) adding Inkling2LoopForConditionalGeneration (model_type inkling2_loop) — day-0 support, looped core layers 6-17 x 3 passes with per-pass KV+conv caches and gated messages; our vLLM pin predates it, so the port needs a pin advance or a lane-scoped oracle record per upstream-sync rules. Base checkpoint thinkingmachines/Inkling-2-Mini (26B total / 3.5B active multimodal MoE; text-only in this GGUF — no vision tower). Support path owed: (1) tabulate Q5_1/Q4_1 in gguf_reader, (2) inkling2_loop config parser + loader row (NEW-CANDIDATE-PARSERS pattern), (3) name-map census like olala W1, (4) looped forward (per-pass caches + message controllers) against the vLLM reference, (5) DSpark speculator is native to the model (separate row). Other arms in the release: Quality 20.07 GB, Compact 14.10 GB, I-Mini 12.36 GB, I-Nano 11.34 GB (imatrix variants), imatrix.gguf 90 MB — not downloaded.

## Resolution

-
