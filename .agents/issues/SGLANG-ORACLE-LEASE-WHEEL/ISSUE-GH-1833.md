ID: ISSUE-GH-1833
Title: **Both registrations of `test_sglang_lease_identity` are deletable at `rc=0`, and the control shows the `SUITES` array itself is ungated.** Raised by the fresh review of PR #1831 and NOT repaired there: the control proves it is a property of the array rather than anything W2 introduced, and repairing `check-test-registration.py` is a semantic checker change owing its own row, spec and red-first evidence. The suite is registered TWICE, deliberately: `scripts/agent-preflight.sh:176` and `.github/workflows/ci.yml:341`. Two mutations at `85c247580`, each proven applied (non-empty diffstat, mutated file still parses) and each restored byte-for-byte: deleting the `SUITES` entry leaves `python3 scripts/check-test-registration.py` at `rc=0`, and deleting the whole 11-line CI step leaves it at `rc=0` -- `grep -c` on the workflow returns 0 and `yaml.safe_load` still parses it. CONTROL: deleting the unrelated `test_tower_skip_rss_report` from the same array behaves identically, `rc=0`. So ANY entry can be removed from `SUITES` with no gate noticing, and the "Registered in TWO places, deliberately" pattern that `.github/workflows/ci.yml:339-343` documents buys no protection in either direction. NOT covered by the neighbours: [#408](https://github.com/mudler/vllm.cpp/issues/408) names suites executed by NOTHING and correctly diagnoses `REQUIRED_TESTS` as a self-guard rather than a population guard, and [#1730](https://github.com/mudler/vllm.cpp/issues/1730) names a suite with ONE registration absent from `SUITES`; neither states that deleting a `SUITES` entry is itself ungated, which is what the control measures. Triage may prefer to fold this into #408. Repair, when taken: give the checker a POPULATION rule -- every `tests/scripts/test_*.py` must appear in `SUITES` and in a CI lane, red when either falls away -- which will red on the twelve suites #408 lists, and that is the point. Listed under `## Owed` in [sglang-wheel-in-lease.md](../../specs/sglang-wheel-in-lease.md)
Row: SGLANG-ORACLE-LEASE-WHEEL
State: UNKNOWN
Kind: bug
GitHub: 1833
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:684`

### Frozen archive evidence

> | [#1833](https://github.com/mudler/vllm.cpp/issues/1833) | `SGLANG-ORACLE-LEASE-WHEEL` | **Both registrations of `test_sglang_lease_identity` are deletable at `rc=0`, and the control shows the `SUITES` array itself is ungated.** Raised by the fresh review of PR #1831 and NOT repaired there: the control proves it is a property of the array rather than anything W2 introduced, and repairing `check-test-registration.py` is a semantic checker change owing its own row, spec and red-first evidence. The suite is registered TWICE, deliberately: `scripts/agent-preflight.sh:176` and `.github/workflows/ci.yml:341`. Two mutations at `85c247580`, each proven applied (non-empty diffstat, mutated file still parses) and each restored byte-for-byte: deleting the `SUITES` entry leaves `python3 scripts/check-test-registration.py` at `rc=0`, and deleting the whole 11-line CI step leaves it at `rc=0` -- `grep -c` on the workflow returns 0 and `yaml.safe_load` still parses it. CONTROL: deleting the unrelated `test_tower_skip_rss_report` from the same array behaves identically, `rc=0`. So ANY entry can be removed from `SUITES` with no gate noticing, and the "Registered in TWO places, deliberately" pattern that `.github/workflows/ci.yml:339-343` documents buys no protection in either direction. NOT covered by the neighbours: [#408](https://github.com/mudler/vllm.cpp/issues/408) names suites executed by NOTHING and correctly diagnoses `REQUIRED_TESTS` as a self-guard rather than a population guard, and [#1730](https://github.com/mudler/vllm.cpp/issues/1730) names a suite with ONE registration absent from `SUITES`; neither states that deleting a `SUITES` entry is itself ungated, which is what the control measures. Triage may prefer to fold this into #408. Repair, when taken: give the checker a POPULATION rule -- every `tests/scripts/test_*.py` must appear in `SUITES` and in a CI lane, red when either falls away -- which will red on the twelve suites #408 lists, and that is the point. Listed under `## Owed` in [sglang-wheel-in-lease.md](../../specs/sglang-wheel-in-lease.md) | bug |

## Resolution

-
