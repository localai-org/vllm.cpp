ID: ISSUE-GH-2345
Title: **`hybrid-placement.md` claimed `RunMoeBlockPlaced` executed under `test_placed_moe_roundtrip` "byte-identical to the direct call and mutation-proven", after both had been deleted -- so the spec advertised a byte-for-byte placement gate that does not exist.** `866075b2f` ([#2309](https://github.com/mudler/vllm.cpp/issues/2309)) deleted the helper once W3c had made it dead; `6416aab85` ([#2331](https://github.com/mudler/vllm.cpp/issues/2331)) deleted the test, which was calling a symbol that no longer existed and had stopped `main` building. **The claim cannot simply be rewritten, for a structural reason:** `RunMoePlaced` short-circuits when the placement device equals the engine device (`if (placed_on == engine_device) return body(engine, dh);` -- no copy, no allocation), so the transfer path is reachable ONLY cross-device. The deleted test reached it by passing `kCPU` as the placement device explicitly, which the seam no longer accepts. FIXED by correcting the bullet to say there is no such gate and pointing at the cross-device gate recorded under `## Owed` in [expert-stream-device-slots.md](../../specs/expert-stream-device-slots.md). **Filed rather than quietly edited, and this is the point of the row:** it is the fallout of #2331, which was my own change, and a record that OVERSTATES coverage is precisely the defect that cost real time hours earlier the same day -- [#2302](https://github.com/mudler/vllm.cpp/issues/2302), a wrong dependency written into a spec because two stale records agreed with each other and neither was the tree. Same shape, same treatment: a traceable correction rather than a silent one. The narrative at `hybrid-placement.md:442` is accurate HISTORY of how the code got here and is deliberately untouched; only the live-coverage claim was wrong
Row: ENG-HYBRID-PLACEMENT
State: UNKNOWN
Kind: bug
GitHub: 2345
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:910`

### Frozen archive evidence

> | [#2345](https://github.com/mudler/vllm.cpp/issues/2345) | `ENG-HYBRID-PLACEMENT` | **`hybrid-placement.md` claimed `RunMoeBlockPlaced` executed under `test_placed_moe_roundtrip` "byte-identical to the direct call and mutation-proven", after both had been deleted -- so the spec advertised a byte-for-byte placement gate that does not exist.** `866075b2f` ([#2309](https://github.com/mudler/vllm.cpp/issues/2309)) deleted the helper once W3c had made it dead; `6416aab85` ([#2331](https://github.com/mudler/vllm.cpp/issues/2331)) deleted the test, which was calling a symbol that no longer existed and had stopped `main` building. **The claim cannot simply be rewritten, for a structural reason:** `RunMoePlaced` short-circuits when the placement device equals the engine device (`if (placed_on == engine_device) return body(engine, dh);` -- no copy, no allocation), so the transfer path is reachable ONLY cross-device. The deleted test reached it by passing `kCPU` as the placement device explicitly, which the seam no longer accepts. FIXED by correcting the bullet to say there is no such gate and pointing at the cross-device gate recorded under `## Owed` in [expert-stream-device-slots.md](../../specs/expert-stream-device-slots.md). **Filed rather than quietly edited, and this is the point of the row:** it is the fallout of #2331, which was my own change, and a record that OVERSTATES coverage is precisely the defect that cost real time hours earlier the same day -- [#2302](https://github.com/mudler/vllm.cpp/issues/2302), a wrong dependency written into a spec because two stale records agreed with each other and neither was the tree. Same shape, same treatment: a traceable correction rather than a silent one. The narrative at `hybrid-placement.md:442` is accurate HISTORY of how the code got here and is deliberately untouched; only the live-coverage claim was wrong | bug |

## Resolution

-
