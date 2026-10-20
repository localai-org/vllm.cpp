ID: ISSUE-LOCAL-01M4R157C4X783JF2XYZH9S19M
Title: Bridge the production tokenizer to the parser engine
Row: MODEL-MM-gemma4-mm-gemma4-for-conditional-generation
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-12
Updated: 2026-10-12
Closed: -

## Problem

T03C5 requires a borrowed immutable tok::Tokenizer bridge with an unambiguous vocabulary domain and actual provider budget gate; serving integration remains deferred.

## Resolution

-

## Bounded implementation evidence

T03C5 bridge passes43/6319 with real providers, six CPU targets,28 metadata,
five compiled mutations and two header consumers. Root verification and fresh
review pending. Remains OPEN for T10 serving wiring/prompt timing, explicitly
excluded from this bounded borrowed-bridge change. No readiness claim.
