# Legacy split SwiGLU boundary

Issue: `ISSUE-LOCAL-01M63W80J6R0M1JXTQ5P1ENWGN`.
Spec-first commit: `f8c67056c`.

A public registered `vt::MoeSiluMul` invocation with BF16 gate and up tensors
and an F32 output exposes the width boundary without hiding it in the final
BF16 store. The unchanged CUDA kernel differs from CPU on 254 of 256 values.
The test uses fixed seeds 214 and 713 and skips only devices at sm75 or newer,
whose behavior this change deliberately preserves.

The repaired kernel rounds SiLU through the gate template type only below sm75.
This matches the already repaired `SiluAndMulKernel` and pinned vLLM
`a7c23ac96d` `csrc/libtorch_stable/activation_kernels.cu:165-167`.
The complete CUDA ops suite passes 14 cases and 450 assertions on P620.
The actual production CUDA translation unit compiles for sm60, sm61, and sm70;
`cuobjdump --list-elf /tmp/quant-cuda_moe.o` confirms all three cubins.

A scratch production source replaces the architecture guard with false,
recompiles for sm61, and links the same registered-op regression. The mutant
again fails on 254 values. Restored full-suite output remains green. Commands
come from `/tmp/quant-parts-compile.py` and `/tmp/quant-split-link.py`, using the
existing Release compile database and a single scratch archive. All GPU runs
hold `/home/user/gpu.lock`. The reviewed source and original build are not
mutated during the negative check.

This supplies the activation boundary for the quantized dense shard method.
Public Llama GGUF loader reachability, real-model token gates, full integration
rebuild, and independent review remain pending. No performance claim is made.
