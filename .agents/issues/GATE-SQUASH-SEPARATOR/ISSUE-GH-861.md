ID: ISSUE-GH-861
Title: `PR_BODY` did not remove the `---------` separator: GitHub writes it above the `Co-authored-by:` block it appends, under both settings, so `617d6f452` landed failing the trailer gate with ONE block and one marker. #829 and #850 concluded otherwise from a simulation that omitted it. The fuse now steps over the separator when trailers sit on both sides (spec [`squash-separator.md`](../../specs/squash-separator.md))
Row: GATE-SQUASH-SEPARATOR
State: UNKNOWN
Kind: bug
GitHub: 861
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:215`

### Frozen archive evidence

> | [#861](https://github.com/mudler/vllm.cpp/issues/861) | `GATE-SQUASH-SEPARATOR` | `PR_BODY` did not remove the `---------` separator: GitHub writes it above the `Co-authored-by:` block it appends, under both settings, so `617d6f452` landed failing the trailer gate with ONE block and one marker. #829 and #850 concluded otherwise from a simulation that omitted it. The fuse now steps over the separator when trailers sit on both sides (spec [`squash-separator.md`](../../specs/squash-separator.md)) | bug |

## Resolution

-
