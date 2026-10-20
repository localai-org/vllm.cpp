# Decoded image geometry fixture

Rows are `[budget, input_height, input_width, target_height, target_width]`.
Captured by executing the unmodified AST body of
`get_aspect_ratio_preserving_size` from Transformers
`a08ace4bbd97e721c98751deec37d87b026acadc`,
`src/transformers/models/gemma4/image_processing_gemma4.py` (SHA256
`5d280d5448b1c219183a27e95b6aa7178b350275772d07881c91c65f13fa815e`).
Arguments are patch_size=16, max_patches=budget*9, pooling_kernel_size=3.
The C++ test encodes these tiny rows directly; independent scalar loops check
all patch values against an RGB ramp, not production indexing helpers.

This is source geometry evidence, not execution of the full HF processor.
Torchvision is absent on the capture host. Resampling and full processor parity
remain unproven; the supported C++ subset refuses every nonidentity target.
Google defaults come from the adjacent processor_config.json, pinned in the
manifest to ee0ef6023621cff504d758262d4e04895a5af4a2.
