# `clm` - the Contrastive-LM/CLM reference, which vLLM does not implement

vLLM has no CLM architecture. CLM is two small MLP projection heads over the
last-token pooled embedding of an unmodified Qwen3-8B, scored by scaled cosine
similarity. The model author's own runtime, `Contrastive-LM/CLM`, is the only
definition of the head layout (`src/clm/heads.py`), the text the encoder sees
(`src/clm/schema.py` `build_pairs`, `to_text`) and the answer
(`answer_from_logits`). The checkpoint is `Contrastive-LM/CLM-v0.1-8B`
(`CLM_v0.1-8B.pt`).

This oracle answers: what correct CLM answers look like for a SystemOne
request. The encoder half of the reference is a vLLM pooling server; where a
comparison runs without one, it substitutes `transformers` (last token of the
post-norm hidden state, L2-normalized) and says so.

```oracle-pin
id = clm
role = secondary
upstream = https://github.com/Contrastive-LM/CLM
scope = the CLM bi-encoder decision model (Qwen3-8B last-token pooling + state/action projection heads, InfoNCE scaled-cosine scoring), which neither vLLM nor vLLM-Omni registers
pin = bb42c6c5bf914fd449bed2f6ca65be80602cb1f7
pin_label = main, 2026-09-30
pinned_on = 2026-09-30
gateable = yes
evidence = docs/models/clm.md
```

`gateable = yes`: on 2026-09-30 the pinned reference ran `Engine.answer` on
CPU over the published checkpoint (encoder: Qwen3-8B @
`b968826d9c46dd6066d109eabc6255188de91218` through `transformers` 5.3.0) and
its answers are the goldens in `tests/vllm/models/clm_goldens.inc`
(`scripts/gen-clm-goldens.py`). The measured comparison is in
[docs/models/clm.md](../../docs/models/clm.md).
