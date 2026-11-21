ID: ISSUE-LOCAL-01M7H8933N4XNQMKF2PDB1MKE0
Title: Mach-2 Additive has no row, no spec and no oracle, so its packed format cannot be ported or gated
Row: QUANT-MACH2-ADDITIVE
State: CLOSED
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-15
Updated: 2026-11-15
Closed: 2026-11-15

## Problem

Syzygy Research publishes Mach-2 Additive Medium, a 1.70-bit-per-weight QTIP bitshift trellis over Qwen3.8-Flash-Next whose codewords are D4 lattice integers applied with addition only. Nothing in this tree recorded the row, the format, or the oracle. Its native format is specified only by decode.py in the weight repository (format tag fn_packed_v2), and vLLM implements none of it, so it is a secondary-oracle case that needs a registry entry and a pin before any code can be written. The model card also states the model loops at greedy sampling, so no token-exact gate may exist for it, which is why the decode oracle has to come first.

## Resolution

Landed on row/QUANT-MACH2-ADDITIVE at 13542b40a as a records-only pull request, per the developer's 2026-11-16 split decision. Registered: the QUANT-MACH2-ADDITIVE row in .agents/quantization-matrix.md (agent record QUANT 88 -> 89); the spec .agents/specs/quant-mach2-additive.md with the codec read from the release decoder and the campaign split into U1-U5; the oracle pin .agents/oracles/mach2-native.md with gateable = no plus its mach2-native entry in the AGENTS.md registry; and CLAIM-QUANT-MACH2-ADDITIVE. The GGUF arm is excluded because the fork documents payload v3 for Mach-1 and names no Mach-2 payload version. Nothing was downloaded, leased or executed; U1 (the CPU decode oracle) opens its own issue.
