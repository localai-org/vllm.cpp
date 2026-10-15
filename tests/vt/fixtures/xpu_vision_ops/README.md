# Executed XPU vision operator fixtures

`generate_layernorm.py` records `torch.nn.functional.layer_norm` on the
reference Intel Arc Pro B70. `layernorm.json` pins the runtime image, Torch
revision, device, file hashes and comparison contracts. The generator uses
deterministic synthetic inputs and no model weights. It repeats every reference
call and requires identical output bytes before writing the fixture.

The matching XPU source pin is `bc294243807debb350dca694049ba7a806dfdceb`,
resolved from the recorded Torch revision's `third_party/xpu.txt`. The native
adaptation follows `src/ATen/native/xpu/sycl/LayerNormKernels.cpp`,
`SharedReduceOps.h` and `GroupReduceUtils.h`; the Apache-2.0 license and Intel
copyright notice are retained under `third_party/torch_xpu_layer_norm/`.

`generate_gelu.py` additionally records both `torch.nn.functional.gelu`
variants on XPU for FP16, BF16 and Float32, including zero/near-zero values and
negative/positive tails. Its optional second output argument is an existing
local tower-capture directory: it runs standalone Erf-GELU on the two captured
merger fc1 outputs and writes the large operator references there. These local
files are not distributed. This is operator replay, not evidence that the
native merger or complete tower runs. The matching source is
`src/ATen/native/xpu/sycl/ActivationGeluKernel.cpp` at the same Intel pin.

The seven cases exercise FP16 tower/merger widths, BF16 and Float32, tails,
constant and nearly constant rows, a large input offset, optional affine
parameters, width one and an empty batch. Relative L2 and maximum absolute
error limits are declared in the generator before native comparisons. Native
tests separately check repeatability, in-place aliasing and invalid metadata.

`generate_dense.py` records three executed FP16 `F.linear` cases, with/without
bias and non-aligned tails. Its optional capture/model arguments also replay
the actual patch and merger projection inputs with checkpoint FP16 weights;
these larger tensors stay local. The actual patch Conv3d layer executes
`F.linear` for its non-overlapping kernel/stride configuration. The fixed
comparison contract is relative L2 3e-4 and maximum absolute error 0.02, without
the existing dense unit test's additional per-element percentage allowance.
Real merger reference replays have small differences across repetitions;
native deterministic oneDNN projection repeats are checked byte-for-byte.
Native tests additionally refuse partially overlapping output ranges and allow
adjacent views. They do not establish complete tower execution or performance.

`generate_position.py` executes the unchanged production
`Qwen3_VisionTransformer.fast_pos_embed_interpolate` method on XPU, observing its
selected Triton route. Four compact FP16 fixtures cover irregular interpolation,
the actual 24×32 grid, hidden-width tails, spatial reorder, frame repetition and
degenerate axes. Repetition here tests this operator; it does not enable video
serving. Optional capture/model arguments also replay the checkpoint's FP16
position table and require byte-identity with both saved actual worker metadata
boundaries. These large tables and references stay local. Native comparison
additionally checks exact output bytes and repeats, invalid geometry/dtypes/
strides/devices, partial aliases and legal adjacent views. The executed Triton
kernel contracts half arithmetic: the first `w01*e01` product rounds before
three successive half FMAs. Merely rounding separate products/sums differs.

`generate_qkv.py` records executed FP16/BF16/Float32 Torch splits, including
unequal widths, the actual 16×72 head geometry and raw special-value patterns.
Native comparisons require unchanged bytes, including signed zeros, subnormals
and NaN payloads. Optional capture/model arguments also execute `F.linear` and
splits on the two actual first-block norm outputs with checkpoint FP16 QKV
parameters. These references are standalone operator replays, not additional
full-worker QKV boundaries. The native tests also compose dense FP16 projection
and QKV splitting, check repeats, six overlap pairs, adjacent output views,
metadata refusal and the unchanged CPU BF16/Float32 contract/FP16 refusal.

`generate_vision_rope.py` executes production `ApplyRotaryEmb` on XPU with
NeoX ordering and FP32 compute disabled. Three compact FP16 fixtures cover
even head-width tails and the actual 16×72 geometry; repeated reference calls
must match bytes. The contract requires exact bytes. Native aligned-row tests
also compose the GPU base-cache producer, FP16 cast and spatial grid with the
two real-input Q/K rotations from `vision-rope-reference.json`. Those large
operator replays stay local and do not establish a complete tower or serving.

`generate_attention.py` executes unchanged production `MMEncoderAttention` on
three compact non-causal FP16 cases at 16×72, optionally adding the two real-input
rotated Q/K and V replays. It observes successful `_vllm_fa2_C.varlen_fwd` calls
and any Python fallback separately; repeated references must match bytes. The
comparison contract, declared before native execution, is the existing matching
FP16 unpaged-attention test's per-element `abs(delta) <= 0.001 + 0.001*abs(ref)`.
Native tests report global L2/maximum errors as diagnostics, check each element,
repeats, input preservation and allocation stability, and save optional device/
host profiles. Timings describe these operator calls, not a serving benchmark.

Run `test_xpu_vision_ops` in an XPU-enabled build. The optional real-model
replays require `VT_B70_VISION_REFERENCE_DIR` to contain the executed
`vision-boundaries-capture.json`, its tensor files and the generated
`gelu-real-merger.json`. LayerNorm additionally needs `VT_B70_VISION_MODEL_DIR`
to contain the matching checkpoint. Dense real-boundary tests require that
checkpoint and `dense-real-projections.json` as well. Missing settings produce
an explicit skip message. These large captures and checkpoint weights
are not distributed with the fixtures. Operator comparisons do not establish
end-to-end native vision serving.

The real position test needs `position-real-replays.json` plus the original
worker boundaries in `VT_B70_VISION_REFERENCE_DIR`. It also composes recorded
patch projection outputs with native position interpolation/Add and compares
the actual first norm inputs. This checks that boundary, not a full tower.
The real QKV test needs `qkv-real-replays.json` and the checkpoint directories
set as above. It applies the same predeclared projection limits as the dense
replays. No checkpoint weights or large real-input references are distributed.
