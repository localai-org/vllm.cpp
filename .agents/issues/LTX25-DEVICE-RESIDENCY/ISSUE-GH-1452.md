ID: ISSUE-GH-1452
Title: **`src/vt/cuda/cuda_conv3d.cu` landed with [#1007](https://github.com/mudler/vllm.cpp/issues/1007) and has never been compiled or executed anywhere in this project's reach.** The row that wrote it was dispatched with the GPU lease withheld, the box has no `nvcc`, and no CI job in `.github/workflows/ci.yml` has a GPU runner — `cuda-fat-build` compiles the file but CI does not complete in that environment, so even the COMPILE verdict is unread. What IS gated, and the scope is exact: `tests/vllm/multimodal/test_diffusion_device_seam.cpp` runs the whole device arm (upload, a `vt::Conv3d` dispatch on a device that is not `kCPU`, download) against the fake unified-memory XPU backend and requires `memcmp`-identical pixels — that gates the MARSHALLING, and it cannot gate this kernel because it registers the CPU kernel for the fake device. Owed: a compile; a `memcmp` CPU-vs-CUDA arm over `tests/vt/test_ops_conv3d.cpp`'s shape table in the shape `tests/vt/test_ops_conv1d_general.cpp` already ships for `kConv1d`; an end-to-end 320x192/25f render with pixels byte-compared and the maximum absolute difference recorded BEFORE any wall-clock number; and the f16/bf16 storage the arm currently refuses by name while the op contract and the CPU arm admit it. The BIT-IDENTITY claim (`__fmul_rn`/`__fadd_rn` against the host's `-ffp-contract=off`) is a DESIGN ARGUMENT and not a measurement. Listed under `## Owed` in [`ltx25-device-residency.md`](../../specs/ltx25-device-residency.md)
Row: LTX25-DEVICE-RESIDENCY
State: UNKNOWN
Kind: bug
GitHub: 1452
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:513`

### Frozen archive evidence

> | [#1452](https://github.com/mudler/vllm.cpp/issues/1452) | `LTX25-DEVICE-RESIDENCY` | **`src/vt/cuda/cuda_conv3d.cu` landed with [#1007](https://github.com/mudler/vllm.cpp/issues/1007) and has never been compiled or executed anywhere in this project's reach.** The row that wrote it was dispatched with the GPU lease withheld, the box has no `nvcc`, and no CI job in `.github/workflows/ci.yml` has a GPU runner — `cuda-fat-build` compiles the file but CI does not complete in that environment, so even the COMPILE verdict is unread. What IS gated, and the scope is exact: `tests/vllm/multimodal/test_diffusion_device_seam.cpp` runs the whole device arm (upload, a `vt::Conv3d` dispatch on a device that is not `kCPU`, download) against the fake unified-memory XPU backend and requires `memcmp`-identical pixels — that gates the MARSHALLING, and it cannot gate this kernel because it registers the CPU kernel for the fake device. Owed: a compile; a `memcmp` CPU-vs-CUDA arm over `tests/vt/test_ops_conv3d.cpp`'s shape table in the shape `tests/vt/test_ops_conv1d_general.cpp` already ships for `kConv1d`; an end-to-end 320x192/25f render with pixels byte-compared and the maximum absolute difference recorded BEFORE any wall-clock number; and the f16/bf16 storage the arm currently refuses by name while the op contract and the CPU arm admit it. The BIT-IDENTITY claim (`__fmul_rn`/`__fadd_rn` against the host's `-ffp-contract=off`) is a DESIGN ARGUMENT and not a measurement. Listed under `## Owed` in [`ltx25-device-residency.md`](../../specs/ltx25-device-residency.md) | bug |

## Resolution

-
