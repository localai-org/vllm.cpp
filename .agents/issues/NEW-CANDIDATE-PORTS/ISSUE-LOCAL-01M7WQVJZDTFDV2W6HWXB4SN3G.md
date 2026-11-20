ID: ISSUE-LOCAL-01M7WQVJZDTFDV2W6HWXB4SN3G
Title: NEW-CANDIDATE-PORTS W1 (olala): GGUF name-map + census-validating weights arm for the pinned Olala-v1.0-mini-7A1B GGUFs; forward still refuses by name
Row: NEW-CANDIDATE-PORTS
State: OPEN
Kind: task
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-20
Updated: 2026-11-20
Closed: -

## Problem

Parser row NEW-CANDIDATE-PARSERS landed (parse + refuse-by-name); this row owns the arch ports. W1 for olala = the GGUF weights arm: (1) W1 tensor walks + name-map census committed under docs/bench-evidence/new-candidate-ports-olala-20261120/ (989 tensors, 36 blocks, 29 M/SSM + 7 V layers at 4,9,14,19,24,29,34 by attn_q presence; layer_types KV is 31 chars vs block_count 36 — discrepancy recorded, tensor walk authoritative); (2) olala_gguf_weights.{h,cpp} with pattern->role name-map, CensusOlalaGguf (every tensor mapped, unmapped listed), ValidateOlalaGguf(gguf, config) checking counts+shapes vs the parsed HfConfig; (3) loader calls ValidateOlalaGguf on the IsOlalaGguf path after config parse, before the existing OlalaForwardRefusal — census reached from the production entry point; (4) CPU test test_olala_gguf_weights.cpp pinned-path-gated over bf16 + q6_k artifacts (989 tensors, 0 unmapped, per-role counts, dtype census bf16=375 BF16/614 F32, q6_k=303 Q6_K/614 F32/36 Q8_0/36 BF16, shape-vs-config assertions, negative unmapped case); (5) USAGE.md status entry. Forward (mamba-hybrid SSM + gated sliding-window attention + latent MoE) is W2+ and still refuses by name. Spec: .agents/specs/new-candidate-ports-olala.md

## Resolution

-
