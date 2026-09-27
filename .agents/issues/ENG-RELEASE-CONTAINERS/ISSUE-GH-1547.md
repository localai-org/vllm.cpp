ID: ISSUE-GH-1547
Title: **The `vulkan` lane cannot build on arm64, and the failed leg leaves the pushed `cpu` digests with no tag.** Run [32447481128](https://github.com/mudler/vllm.cpp/actions/runs/32447481128), job `publish (vulkan, linux/arm64, ubuntu-24.04-arm)`, failed at 406 seconds with `accelerator release metadata error: unsupported Linux accelerator artifact 'linux-aarch64-glibc-vulkan'`, and the re-run failed identically. `docker/Dockerfile:145-146` selects that id for every non-amd64 build while `scripts/release_accelerator_metadata.py` `ARTIFACTS` and `scripts/release_manifest.py` `_artifact_policy` both stop at `linux-x86_64-glibc-vulkan`. The blast radius is the whole publish: `manifest` declares `needs: [plan, publish]`, so one failed leg skipped the job that turns pushed digests into tags, and the `cpu` images built, validated and pushed by digest on both architectures and sit in the registry unreachable. FIXED HERE by registering the tuple rather than removing the lane. The lane is intended: `release/container-matrix.json` declares `vulkan` on both platforms at `channel: preview`, `scripts/check-container-matrix.py` refuses any lane that does not declare both, and removing the leg would mean weakening that gate for every lane. The registration is deliberately NARROW. Two registries describe this tuple and are edited; the third, `scripts/release_pipeline.py` `PRIMARY_ARTIFACT_FORMATS`, describes DOWNLOADABLE archives and is not, because `release_pipeline.py:137` requires `release/release-matrix.json` to equal it exactly and `.github/workflows/release.yml` builds no aarch64 Vulkan tarball. The channel set is `{"preview"}` and not `{"preview", "stable"}`: `.agents/roadmap_v1.md` records both arm64 container legs as unbuilt, no arm64 Vulkan leg has ever run here, and the produced manifest already records `correctness`, `runtime` and `performance` as `absent`. A test refuses `stable` by name and a second test pins the deliberate absence from the download matrix. Spec [fix-ci-container-publish.md](../../specs/fix-ci-container-publish.md)
Row: ENG-RELEASE-CONTAINERS
State: UNKNOWN
Kind: bug
GitHub: 1547
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:544`

### Frozen archive evidence

> | [#1547](https://github.com/mudler/vllm.cpp/issues/1547) | `ENG-RELEASE-CONTAINERS` | **The `vulkan` lane cannot build on arm64, and the failed leg leaves the pushed `cpu` digests with no tag.** Run [32447481128](https://github.com/mudler/vllm.cpp/actions/runs/32447481128), job `publish (vulkan, linux/arm64, ubuntu-24.04-arm)`, failed at 406 seconds with `accelerator release metadata error: unsupported Linux accelerator artifact 'linux-aarch64-glibc-vulkan'`, and the re-run failed identically. `docker/Dockerfile:145-146` selects that id for every non-amd64 build while `scripts/release_accelerator_metadata.py` `ARTIFACTS` and `scripts/release_manifest.py` `_artifact_policy` both stop at `linux-x86_64-glibc-vulkan`. The blast radius is the whole publish: `manifest` declares `needs: [plan, publish]`, so one failed leg skipped the job that turns pushed digests into tags, and the `cpu` images built, validated and pushed by digest on both architectures and sit in the registry unreachable. FIXED HERE by registering the tuple rather than removing the lane. The lane is intended: `release/container-matrix.json` declares `vulkan` on both platforms at `channel: preview`, `scripts/check-container-matrix.py` refuses any lane that does not declare both, and removing the leg would mean weakening that gate for every lane. The registration is deliberately NARROW. Two registries describe this tuple and are edited; the third, `scripts/release_pipeline.py` `PRIMARY_ARTIFACT_FORMATS`, describes DOWNLOADABLE archives and is not, because `release_pipeline.py:137` requires `release/release-matrix.json` to equal it exactly and `.github/workflows/release.yml` builds no aarch64 Vulkan tarball. The channel set is `{"preview"}` and not `{"preview", "stable"}`: `.agents/roadmap_v1.md` records both arm64 container legs as unbuilt, no arm64 Vulkan leg has ever run here, and the produced manifest already records `correctness`, `runtime` and `performance` as `absent`. A test refuses `stable` by name and a second test pins the deliberate absence from the download matrix. Spec [fix-ci-container-publish.md](../../specs/fix-ci-container-publish.md) | bug |

## Resolution

-
