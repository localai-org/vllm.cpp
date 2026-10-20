ID: ISSUE-LOCAL-01M4P1A9D5ED9CY2YT4229BNB9
Title: Prepare Gemma engine state from prompt IDs once per request
Row: MODEL-MM-gemma4-mm-gemma4-for-conditional-generation
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-11
Updated: 2026-10-11
Closed: 2026-10-11

## Problem

ParserEngine parse_delta cannot carry prompt IDs, so prompt-open reasoning is emitted as content before the first close.

## Resolution

2026-10-11: Engine-local prompt preparation implemented after prospective fcc0b0125 and executable 12-assertion RED. Final six CPU targets, 28 metadata, 44 syntax consumers and 11 compiled sensitivity mutants pass; evidence assets/work/2026-10-11-e4b-prompt-state/README.md. Root verification and fresh review remain owed; no serving or original-token claim.
