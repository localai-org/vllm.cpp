ID: ISSUE-GH-41
Title: ROCm (AMD GPU) backend
Row: BACKEND-ROCM
State: OPEN
Kind: feature
GitHub: 41
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-05
Updated: 2026-08-29
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM`
>
> Someone offered to help bring up ROCm support, so this issue tracks the work.
>
> ROCm is listed as roadmap in the README backend table. The codebase was structured for exactly this kind of addition: the engine core (scheduler, KV/block management, persistent batch, sampler, serving) is backend-agnostic, mirroring upstream vLLM, and a new device lands as additive files through three seams (see [.agents/backends.md](../../backends.md)):
>
> 1. Platform: a mirror of `vllm/platforms/`. Upstream already ships `platforms/rocm.py`, so ours is a port of that. Platforms self-register and expose device probing, memory model, stream semantics and graph-capture capability.
> 2. Attention backend registry: backends self-register per `(DeviceType, name)`. Adding one is a single self-registering translation unit plus a priority slot.
> 3. `vt::` op tables: per-device kernel registration for GEMM, norms, rope, activations, MoE and sampling.
>
> Adding a platform never touches engine code. The drop-in kernel ABI ([.agents/specs/dropin-kernel-abi.md](../../specs/dropin-kernel-abi.md)) matches upstream kernel entry-point signatures at the vt-op boundary, and upstream's ROCm kernels mirror the CUDA signatures in `csrc/`, so kernels lifted from vLLM's ROCm path should bind without redesign.
>
> vLLM's ROCm platform and kernels are the primary upstream to mirror. SGLang has native ROCm support too and is a secondary reference. Build recipes and per-backend state are in [docs/BUILD.md](../../../docs/BUILD.md).
>
> ## Milestones
>
> - [ ] M0, build: `-DVLLM_CPP_HIP=ON` CMake path, HIP toolchain detection, portable layer compiles for a target `gfx` arch.
> - [ ] M1, platform: `platforms/rocm` mirroring `vllm/platforms/rocm.py`, self-registered.
> - [ ] M2, first model end to end: a small dense model produces coherent output on an AMD GPU via the portable kernel path, token parity vs the CPU reference backend.
> - [ ] M3, attention: register a ROCm attention backend. Upstream uses Triton/AITER flash attention on ROCm; start from whatever vLLM selects for the target arch.
> - [ ] M4, correctness gate: greedy token parity vs a vLLM-ROCm oracle on the same hardware, following the project gate methodology.
> - [ ] M5, speed: benchmark vs vLLM on the same box, quant-matched. The bar is vLLM, not other engines.
>
> ## If you want to pick this up
>
> Comment with the hardware and stack you have (MI300, consumer RDNA, Strix Halo APU, ROCm version). It changes what the first target model and quantization should be for M2, and whether a vLLM-ROCm oracle can run on the same box for M4. I'm happy to walk you through the codebase and split the milestones into separate issues once work starts.

## Resolution

-
