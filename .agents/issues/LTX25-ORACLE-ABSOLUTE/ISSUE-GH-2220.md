ID: ISSUE-GH-2220
Title: **A CUDA toolkit staged off CIFS compiles but cannot be LINKED against, and the precondition that should catch it checks the one link that works.** `/workspace` is CIFS and stores no symlink, so a staged toolkit carries only `libcudart.so.13.3.29` and `libcublasLt.so.13.6.0.2`. `ltx25-oracle-absolute-render.sh` rebuilt the links with `b=${f%%.so.*}; ln -sf "$f" "$b.so"; ln -sf "$f" "$b.so.${f#*.so.}"` -- and `${f#*.so.}` strips the SHORTEST prefix, so it expands to `13.3.29` rather than `13`. The second `ln` therefore links `libcudart.so.13.3.29` to ITSELF and **`libcudart.so.13`, the SONAME, is never created**. That is the name `ld` resolves versioned undefined symbols against, so CMake reports `Found CUDAToolkit`, every CUDA TU compiles, and the job dies 21 minutes later with 38 `undefined reference to ...@libcudart.so.13` / `@libcublasLt.so.13` and `ninja: build stopped`. `need_ok` tested `[ -f .../libcublasLt.so ]`, which is exactly the link the loop DID create, so it passed on an unlinkable toolkit -- a precondition that cannot fail. LATENT, not new: the staging branch is a FALLBACK, and every earlier lease found `/usr/local/cuda` 13.0.88 and never took it; `dgx:gpu0` went `unhealthy ... worker_lost` for 3h20m on 2026-08-28 and returned without a toolkit, which exercised it for the first time. The two runs A/B in their own configure logs: `20260827T220845Z` `/usr/local/cuda` 13.0.88 built in 1192 s, `20260828T224529Z` `/root/cudatk` 13.3.73 failed at link. Fixed in flow: take the MAJOR (`v=${f#*.so.}; ${v%%.*}`), prefer `ldconfig -n` which reads each object's own `DT_SONAME`, and assert `<stem>.so` resolves AND `<stem>.so.<MAJOR>` exists for both libraries BEFORE the build. Red-before/green-after on a replica of the CIFS layout: old loop creates no `.so.13`, new logic creates both, and the guard FAILS on the old layout, PASSES on the new, and FAILS on the real NAS source. `rc` job `1ad519b1-4e75-41d7-9386-9932076390f1`, exit 34. Also recorded in [`environment.md`](../../environment.md) as a lease-environment fact, because it will bite the next row
Row: LTX25-ORACLE-ABSOLUTE
State: UNKNOWN
Kind: bug
GitHub: 2220
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:878`

### Frozen archive evidence

> | [#2220](https://github.com/mudler/vllm.cpp/issues/2220) | `LTX25-ORACLE-ABSOLUTE` | **A CUDA toolkit staged off CIFS compiles but cannot be LINKED against, and the precondition that should catch it checks the one link that works.** `/workspace` is CIFS and stores no symlink, so a staged toolkit carries only `libcudart.so.13.3.29` and `libcublasLt.so.13.6.0.2`. `ltx25-oracle-absolute-render.sh` rebuilt the links with `b=${f%%.so.*}; ln -sf "$f" "$b.so"; ln -sf "$f" "$b.so.${f#*.so.}"` -- and `${f#*.so.}` strips the SHORTEST prefix, so it expands to `13.3.29` rather than `13`. The second `ln` therefore links `libcudart.so.13.3.29` to ITSELF and **`libcudart.so.13`, the SONAME, is never created**. That is the name `ld` resolves versioned undefined symbols against, so CMake reports `Found CUDAToolkit`, every CUDA TU compiles, and the job dies 21 minutes later with 38 `undefined reference to ...@libcudart.so.13` / `@libcublasLt.so.13` and `ninja: build stopped`. `need_ok` tested `[ -f .../libcublasLt.so ]`, which is exactly the link the loop DID create, so it passed on an unlinkable toolkit -- a precondition that cannot fail. LATENT, not new: the staging branch is a FALLBACK, and every earlier lease found `/usr/local/cuda` 13.0.88 and never took it; `dgx:gpu0` went `unhealthy ... worker_lost` for 3h20m on 2026-08-28 and returned without a toolkit, which exercised it for the first time. The two runs A/B in their own configure logs: `20260827T220845Z` `/usr/local/cuda` 13.0.88 built in 1192 s, `20260828T224529Z` `/root/cudatk` 13.3.73 failed at link. Fixed in flow: take the MAJOR (`v=${f#*.so.}; ${v%%.*}`), prefer `ldconfig -n` which reads each object's own `DT_SONAME`, and assert `<stem>.so` resolves AND `<stem>.so.<MAJOR>` exists for both libraries BEFORE the build. Red-before/green-after on a replica of the CIFS layout: old loop creates no `.so.13`, new logic creates both, and the guard FAILS on the old layout, PASSES on the new, and FAILS on the real NAS source. `rc` job `1ad519b1-4e75-41d7-9386-9932076390f1`, exit 34. Also recorded in [`environment.md`](../../environment.md) as a lease-environment fact, because it will bite the next row | bug |

## Resolution

-
