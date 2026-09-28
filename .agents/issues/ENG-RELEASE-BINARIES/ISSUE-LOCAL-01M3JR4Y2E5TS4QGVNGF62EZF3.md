ID: ISSUE-LOCAL-01M3JR4Y2E5TS4QGVNGF62EZF3
Title: the roadmap REL row carries no release lifecycle, so two release gates are red, and its "no published binary exists" clause is false against tag v0.0.2
Row: ENG-RELEASE-BINARIES
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-27
Updated: 2026-09-27
Closed: -

## Problem

Two release gates are red because the roadmap's REL row does not carry the release lifecycle, and the cell that does carry it contradicts the repository.

check-release-binary-contract and check-windows-release-state both read the `REL` / `ROAD-V1-RELEASE` row of .agents/roadmap_v1.md. The binary-contract checker requires its State cell to be `ACTIVE` (it is) and its Next-gate cell to contain three fragments -- "v0.0.2 published eight primary archive/checksum/provenance triplets", "Windows W14-W16 are implemented for one PR", and "publication and 32-asset audit remain pending". The windows-state checker additionally requires .agents/roadmap_v1.md to carry exactly one `<!-- ENG-RELEASE-WINDOWS: state=ACTIVE publication=pending artifact=unpublished -->` anchor and the string "v0.0.3-pre.1` publication and 32-asset audit remain pending".

None of that text has ever been in the roadmap. `git log -S "v0.0.2 published eight primary archive" -- .agents/roadmap_v1.md` returns nothing, and at 1de097c46 the file contains no "v0.0.2", no "v0.0.3-pre.1" and no anchor. The v0.0.2 facts live in the ENG-RELEASE-BINARIES row of .agents/engine-matrix.md instead, so the two gates disagree about which file owns the lifecycle. This is the same shape as check-benchmark-index: a checker requiring text no revision ever carried.

The Next-gate cell also asserts something false. It reads "Hosted ten-SM completion, the full eight-tuple dry run, matching-hardware evidence, merge, and tagged publication remain pending; no published binary exists." The tag `v0.0.2` EXISTS in this repository (`git tag` lists v0.0.2, v0.0.2-alpha, v0.0.2-alpha1-ci-test), and the engine-matrix row records the publication: "v0.0.2 published eight archive/checksum/provenance triplets plus two indexes from 7020de93652ca920424a10ac5255b34810dd2f24 in run 31466516224 (26 assets)". "No published binary exists" and "tagged publication remain pending" are both false, and they are false about the row the roadmap points at.

The three fragments the checker wants are all TRUE and independently sourced, which is why they can be written rather than invented:

- v0.0.2: tag exists; engine-matrix ENG-RELEASE-BINARIES carries the run and asset count.
- Windows W14-W16 implemented: .agents/specs/windows-binary-release.md:5-7 reads "Status: ACTIVE. Design approved by the developer on 2026-08-11. W14-W16 are implemented locally; native hosted Windows evidence, the merged-SHA ten-tuple dry run, prerelease publication, and the 32-asset audit remain pending." Its wave table at :390-394 assigns W14/W15/W16 and states "W14, W15, and W16 land in one PR".
- 32-asset audit pending: the same spec at :451, "v0.0.3-pre.1 publication, attestations, and the exact 32-asset API audit are not inferred from Linux and remain required hosted gates."

This change writes that lifecycle into the Next-gate cell, sourced from the engine-matrix row and the Windows spec, adds the one anchor the windows checker requires, and corrects the false "no published binary exists" clause. It does not move the row's State, does not touch either checker, and does not claim Windows is published -- the row stays ACTIVE/pending/preview and the spec's own stop conditions are unchanged.

## Resolution

-
