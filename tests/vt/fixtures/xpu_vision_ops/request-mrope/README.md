# Executed request M-RoPE positions

`generate.py MODEL_CONFIG OUTPUT_DIRECTORY` executes the installed pinned
`Qwen3_5ForConditionalGeneration.get_mrope_input_positions` with CPU image-grid
metadata. The method is inherited from Qwen3-VL. The fixture records its source
file SHA-256 and the reference image. This capture needs no GPU, model weights
or complete worker forward; it does not emulate the position formula.

Five cases cover text-only input, an image at the start, inside and at the end
of a prompt, and two image features supplied out of offset order. Native
registered positions and the maximum-position delta must match exactly. These
integer checks do not qualify image-conditioned logits, speculative positions,
chunked runner positions or complete conversation semantics.
