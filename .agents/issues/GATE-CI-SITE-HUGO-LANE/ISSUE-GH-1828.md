ID: ISSUE-GH-1828
Title: **`tests/scripts/test_check_site.py` asserts a literal `10` rendered benchmark detail links, which is a stored count of `docs/benchmarks/*.md` inside another file, and #1754's lane fix is what ARMS it**: until Hugo was installed the case never reached that line in CI. Measured on `d60692c89` with Hugo 0.146.3+extended: 10 slugs, 10 table hrefs, no duplicated target and no unlinked slug, so the relationship is a bijection and the literal is derivable at read time -- with its own non-vacuity floor, since `sorted(x) == sorted(y)` is satisfied by two empty sets. NOT FIXED IN FLOW: deriving it changes what the case asserts rather than how it spells a number, which `AGENTS.md` routes through the normal row, spec and fresh-review path, and the file already has an open pull request against the same case. Listed under `## Owed` in [gate-ci-site-hugo-lane.md](../../specs/gate-ci-site-hugo-lane.md)
Row: GATE-CI-SITE-HUGO-LANE
State: UNKNOWN
Kind: bug
GitHub: 1828
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:687`

### Frozen archive evidence

> | [#1828](https://github.com/mudler/vllm.cpp/issues/1828) | `GATE-CI-SITE-HUGO-LANE` | **`tests/scripts/test_check_site.py` asserts a literal `10` rendered benchmark detail links, which is a stored count of `docs/benchmarks/*.md` inside another file, and #1754's lane fix is what ARMS it**: until Hugo was installed the case never reached that line in CI. Measured on `d60692c89` with Hugo 0.146.3+extended: 10 slugs, 10 table hrefs, no duplicated target and no unlinked slug, so the relationship is a bijection and the literal is derivable at read time -- with its own non-vacuity floor, since `sorted(x) == sorted(y)` is satisfied by two empty sets. NOT FIXED IN FLOW: deriving it changes what the case asserts rather than how it spells a number, which `AGENTS.md` routes through the normal row, spec and fresh-review path, and the file already has an open pull request against the same case. Listed under `## Owed` in [gate-ci-site-hugo-lane.md](../../specs/gate-ci-site-hugo-lane.md) | bug |

## Resolution

-
