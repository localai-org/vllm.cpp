# Olala-v1.0-mini-7A1B GGUF name-map census (W1 enumeration, NEW-CANDIDATE-PORTS)

Artifacts (HF OVHaiLLM/Olala-v1.0-mini-7A1B-GGUF, snapshot b9f5d20a...): bf16 (13.59 GB), q6_k, q5_k_m, q8_0-safe2.
Walks: olala-tensor-walk-bf16.txt / olala-tensor-walk-q6k.txt (dump_container manifest, 989 tensors each).

## Globals (no blk. prefix)
  output.weight            BF16  [151936,1536]
  token_embd.weight        BF16  [151936,1536]

## Layer census: 36 blocks, two layer kinds
- V (full-attention) layers at 4, 9, 14, 19, 24, 29, 34 (7) — identified by attn_q.weight presence;
  cross-checked against the olala.layer_types KV read directly from the file: 36 chars, V at exactly
  [4,9,14,19,24,29,34] — the two sources AGREE.
- M (SSM) layers: the other 29.

## Common per-layer tensors (all 36, from blk.0)
  attn_output.weight       BF16  [1536,3072]
  exp_probs_b              F32   [256]
  ffn_down_exps.weight     BF16  [256,384,768]
  ffn_down_shexp.weight    BF16  [1536,1536]
  ffn_gate_inp.weight      F32   [256,1536]
  ffn_latent_down.weight   BF16  [384,1536]
  ffn_latent_up.weight     BF16  [1536,384]
  ffn_up_exps.weight       BF16  [256,768,384]
  ffn_up_shexp.weight      BF16  [1536,1536]
  geo_mixer_bias           F32   [1]
  geo_mixer_scale          F32   [1]
  geo_mlp_bias             F32   [1]
  geo_mlp_scale            F32   [1]

## M-layer-only tensors (29 layers, from blk.0)
  ssm_b_bias               F32   [48,4,128]
  ssm_b_norm.weight        F32   [128]
  ssm_c_bias               F32   [48,4,128]
  ssm_c_norm.weight        F32   [128]
  ssm_d                    F32   [48]
  ssm_dt_bias              F32   [48]
  ssm_in.weight            BF16  [6288,1536]
  ssm_in_dyn.weight        BF16  [1056,1536]
  ssm_m3_bias              F32   [48,8,128]
  ssm_m3_misc              F32   [2,48]
  ssm_m3_mxz               F32   [48,8,64]
  ssm_m3_norms             F32   [2,128]
  ssm_mimo_o               F32   [48,4,64]
  ssm_mimo_x               F32   [48,4,64]
  ssm_mimo_z               F32   [48,4,64]

## V-layer-only tensors (7 layers, from blk.4; attn_output excluded above)
  attn_gate.weight         BF16  [4608,1536]
  attn_k_norm.weight       F32   [128]
  attn_lambda.weight       BF16  [12,1536]
  attn_q.weight            BF16  [6144,1536]
  attn_q_norm.weight       F32   [128]
  attn_shift_k.weight      BF16  [12,1536]
  attn_shift_v.weight      BF16  [12,1536]
  attn_softmax_scaler      F32   [48]
  attn_wa_k.weight         BF16  [48,1536]
  attn_wa_v.weight         BF16  [48,1536]
  attn_wb_k.weight         BF16  [512,1536]
  attn_wb_v.weight         BF16  [512,1536]

## Dtype census
- bf16 artifact: {'BF16': 375, 'F32': 614}
- q6_k artifact: {'Q6_K': 303, 'F32': 614, 'BF16': 36, 'Q8_0': 36}
- q6_k non-Q6_K set (the "safe" tensors): all F32 biases/norms/router (exp_probs_b, ffn_gate_inp, geo_*, ssm_* biases/norms, attn q/k norms, attn_softmax_scaler), BF16 ffn_latent_up, Q8_0 ffn_up_exps (routed up-projection).

## Config cross-check (olala-kvs.txt from the parser row)
- hidden 1536 = token_embd.shape[1]; vocab 151936 = token_embd.shape[0] = output.shape[0] (untied head, both present)
- experts 256: ffn_*_exps.shape[0]; expert_used_count 6; moe_latent_size 384: ffn_*_exps.shape[1]; expert_feed_forward_length 768: shape[2]
- M-layer attn_output [1536,3072] = ssm.inner_size 3072; V-layer attn_output [1536,4608], attn_q [6144,1536] = 48 heads x key_length 128
- ssm_mimo_* [48,4,64]: heads 48 x mimo_dim 4 x headdim 64; ssm_b/c_bias [48,4,128]: heads x mimo x state_size 128
