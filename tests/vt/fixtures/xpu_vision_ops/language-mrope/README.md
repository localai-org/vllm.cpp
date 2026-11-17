# Executed language M-RoPE coefficients

`language-mrope.json` records the pinned reference image, Torch build, source
hash and construction device. `generate.py OUTPUT_DIRECTORY` executes the
installed `MRotaryEmbedding.forward_xpu` with FP16 queries that expose its
selected cosine/sine coefficients directly. It does not emulate axis selection
in Python. Run it inside the recorded image with the XPU available; the native
product has no Python/Torch dependency.

The eight cases cover the target geometry (head 256, rotary width 64, theta
10,000,000, sections 11/11/10), both axis layouts, asymmetric and empty sections,
equal axes, and positions up to 262143. The reference constructor's fourfold
cache expansion is bounded to 16384 or 262144 rows. Only selected coefficients
are stored. Its construction runs on XPU; the complete worker's cache
initialization device is not established by this standalone capture.

The fixed native gate is storage-exact after casting the produced FP32 cache to
FP16. The focused test also checks both integer widths, unchanged inputs,
repeatability, unchanged live allocation size, equal-axis identity with the
existing 1-D producer, invalid shape/section/overlap refusal and zero reference
tier hits. This qualifies the producer, not full vision-conditioned logits.
