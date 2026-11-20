ID: ISSUE-LOCAL-01M7PKXYE5WHFDA8EJ6J1YA2QC
Title: Measure mach2-native oracle gateability: run the release decode.py and lease a host that holds the 26.7 GB artifact
Row: QUANT-MACH2-ADDITIVE
State: OPEN
Kind: measurement
GitHub: 3573
Mirror: SYNCED
Availability: FULL
Created: 2026-11-18
Updated: 2026-11-18
Closed: -

## Problem

.agents/oracles/mach2-native.md records `gateable = no`, and what keeps it there is the ARTIFACT, not the decoder. The file-level risk is already retired: `decode.py` downloads clean at 17,766 bytes with LF endings and intact indentation, sha256 `e27dbbe500250364506f8b7e8afb46b7c86971c782c7e0824e49a851a1158b1e`, `python3 -m py_compile decode.py` exits 0, and U1 plus U2 ran that unmodified file over our own fixtures with 51,184/51,200 values bit-identical at 160x320 and 4,601/4,608 at 96x48. Two facts about the published weights are still unmeasured. First, that a real published shard decodes: every fixture so far is one this tree generated, no file from the 26.7 GB text-weights release has been downloaded, and the `format = fn_packed_v2` metadata every expert shard is said to carry has never been read from the artifact that carries it. Second, that a leased host can hold it: about 26.7 GB of packed text weights beside a 102.4 GB bf16 n-gram table in host RAM. Until both land, `.agents/specs/quant-mach2-additive.md` G2 is measured against a decoder that has never seen the weights it is supposed to gate, and `gateable` cannot honestly become `yes`.

## Resolution

-
