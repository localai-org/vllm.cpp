ID: ISSUE-LOCAL-01M4KSETGY63TDEDPF6EHQHRA8
Title: Establish the original Google Gemma-4 E4B multimodal baseline
Row: MODEL-MM-gemma4-mm-gemma4-for-conditional-generation
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-10
Updated: 2026-10-10
Closed: -

## Problem

The original google/gemma-4-E4B-it checkpoint lacks validated production text, tool, image, audio-input, and video-frame inference. Static inspection at 6c43232260b300e6992f9375811f782fe24c172a finds both multimodal hooks absent from kGemma4Factory. Historical ISSUE-GH-2173 remains UNKNOWN/METADATA_ONLY, not an open milestone record. Reuse the existing ENG-MM-INPUT-PIPELINE runner. The canonical design is .agents/specs/gemma4-e4b-google-baseline.md. This issue stays OPEN until the complete baseline lands with exact-checkpoint and production-reachability evidence.

## Resolution

-
