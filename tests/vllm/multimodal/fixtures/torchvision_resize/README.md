# Executed native processor references

Generated arithmetic RGB patterns; no private images, weights or model answers.
`generate.py` records the pinned environment, entrypoint, shapes and SHA-256 of
each output in `manifest.json`. It runs torchvision **v2** tensor resize and the
resolved Transformers Qwen2VLImageProcessor class, not a NumPy/C++ mirror.

Ten compact UInt8 resize cases cover down/up/mixed/one-axis/identity/tiny sizes.
Four compact processor cases use differing per-channel means/stds and retain
exact Float32 and directly cast FP16 patch arrays. The smaller patch geometry
keeps these public fixtures compact; the separate local executed worker captures
exercise the actual checkpoint's 16×16/two-frame/merge-two geometry.

Reproduce without model weights or GPU access in reference image
`sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918`:

```sh
/opt/venv/bin/python generate.py OUTPUT_DIR
```

UInt8 resize outputs pin the reference CPU dispatch, while normalized FP16
patches pin the actual EXL3 model dtype. Existing BF16 goldens are unchanged.
