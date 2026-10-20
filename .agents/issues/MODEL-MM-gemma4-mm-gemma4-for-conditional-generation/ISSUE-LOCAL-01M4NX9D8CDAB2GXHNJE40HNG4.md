ID: ISSUE-LOCAL-01M4NX9D8CDAB2GXHNJE40HNG4
Title: Match Google empty schema mapping truthiness
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

Independent T03A review F1 at f340f218 finds extra items:{} for array items={} versus original Google Jinja2 bytes. Audit analogous mapping truth tests and add independent nested regressions before exact-hash-only repair.

## Resolution

2026-10-11: prospective spec8c32404ac preceded actual executable RED (24 byte failures). Exact-SHA source adapter now qualifies params and array items truth tests, preserving mapping-only branches. Independent unchanged-Google Jinja oracle180 passes; standalone48/415, loader46/7196, audio5/5225, image4/201, vision5/1172, all zero skips; metadata28 passes. Evidence assets/work/2026-10-11-e4b-template-fix/README.md. Root verification and fresh mutation review pending; no runtime or full T03 acceptance. Full preflight prohibited/PENDING.
