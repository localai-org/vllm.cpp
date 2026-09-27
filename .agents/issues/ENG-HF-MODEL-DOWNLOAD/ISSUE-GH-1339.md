ID: ISSUE-GH-1339
Title: SUPERSEDES the integrity sentence in the [#1280](https://github.com/mudler/vllm.cpp/issues/1280) row above, which still reads "refuses a listing where two distinct files carry the same identifier". That any-duplicate rule was refuted in review and never shipped: `lfs.oid` is the sha256 of the contents and the plain `oid` is the git blob sha1, so two byte-identical files share one identifier by construction and the rule rejected legitimate repositories. Two narrower rules shipped instead, neither depending on the token. An identifier whose characters are all the same is refused, which is the 17 August 2026 `Lightricks/LTX-2.5` shape. A shared identifier whose entries disagree on a size the listing REPORTED is refused, whether or not their paths differ, and an agreed size is accepted. The second review then found two holes in that size rule and both are repaired here. It kept the first entry carrying an identifier as that identifier's owner whatever its size state and compared only against a known-size owner, so ONE entry reporting no size disarmed the rule for that identifier for the rest of the listing, and `HF_ENDPOINT` is user configurable. The owner is now the first entry whose size was reported, which is enough rather than a list because size equality is transitive. It also carried a distinct-path guard that was pinned by nothing and that accepted one path listed twice at 4096 and 2048 bytes, the exact self-contradiction the rule exists to catch; the guard is removed. `HfFile::size` becomes `std::optional<uint64_t>` in the same repair, because `0` spelled both a zero-byte file and an unreported size and W3 sizes a byte range from that field. Spec [`hf-model-download.md`](../../specs/hf-model-download.md)
Row: ENG-HF-MODEL-DOWNLOAD
State: UNKNOWN
Kind: bug
GitHub: 1339
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:442`

### Frozen archive evidence

> | [#1339](https://github.com/mudler/vllm.cpp/issues/1339) | `ENG-HF-MODEL-DOWNLOAD` | SUPERSEDES the integrity sentence in the [#1280](https://github.com/mudler/vllm.cpp/issues/1280) row above, which still reads "refuses a listing where two distinct files carry the same identifier". That any-duplicate rule was refuted in review and never shipped: `lfs.oid` is the sha256 of the contents and the plain `oid` is the git blob sha1, so two byte-identical files share one identifier by construction and the rule rejected legitimate repositories. Two narrower rules shipped instead, neither depending on the token. An identifier whose characters are all the same is refused, which is the 17 August 2026 `Lightricks/LTX-2.5` shape. A shared identifier whose entries disagree on a size the listing REPORTED is refused, whether or not their paths differ, and an agreed size is accepted. The second review then found two holes in that size rule and both are repaired here. It kept the first entry carrying an identifier as that identifier's owner whatever its size state and compared only against a known-size owner, so ONE entry reporting no size disarmed the rule for that identifier for the rest of the listing, and `HF_ENDPOINT` is user configurable. The owner is now the first entry whose size was reported, which is enough rather than a list because size equality is transitive. It also carried a distinct-path guard that was pinned by nothing and that accepted one path listed twice at 4096 and 2048 bytes, the exact self-contradiction the rule exists to catch; the guard is removed. `HfFile::size` becomes `std::optional<uint64_t>` in the same repair, because `0` spelled both a zero-byte file and an unreported size and W3 sizes a byte range from that field. Spec [`hf-model-download.md`](../../specs/hf-model-download.md) | bug |

## Resolution

-
