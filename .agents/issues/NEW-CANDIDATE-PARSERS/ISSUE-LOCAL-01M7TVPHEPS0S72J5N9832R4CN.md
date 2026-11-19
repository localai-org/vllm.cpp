ID: ISSUE-LOCAL-01M7TVPHEPS0S72J5N9832R4CN
Title: NEW-CANDIDATE-PARSERS: the CPU-side config parsers for olala, hy_v4_flash, and Maxwell — parse and refuse-by-name, no forwards
Row: NEW-CANDIDATE-PARSERS
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-19
Updated: 2026-11-19
Closed: -

## Problem

Three new candidate artifacts are downloaded and pinned but NONE parses today: (1) /mnt/models/olala-v1-mini/ — GGUF v3, general.architecture = 'olala' (custom, 989 KVs, mamba-hybrid MoE 7A1B); (2) /mnt/models/hy4-flash-gguf/ — GGUF v3, general.architecture = 'hy_v4_flash' (custom, 1415 KVs, Q4_K_M 2 shards + mmproj; the tencent Hy4-flash with Engram tables/compressed indexer/DFLY); (3) /mnt/models/maxwell-v1-flash/snapshot/ — safetensors 35 shards + config.json with REMOTE CODE (MaxwellMoeConfig auto_map), 36.8B. The loader's additive arch table (model_loader.cpp:1223) has no rows for olala or hy_v4_flash, and Maxwell's safetensors+config path has no parser. Owed (parsers ONLY, no forwards): the W0 enumerations committed (the GGUF header walks + the safetensors header walk + the config.json field map), the three config builders (OlalaHfConfigFromGguf / HyV4FlashHfConfigFromGguf / MaxwellConfigFromJson — mapping standard fields by name, unmapped fields carried verbatim, Maxwell's remote code NOT executed), ONE loader row per arch, refuse-by-name at the forwards, CPU-only tests against the three pinned paths, docs entries (parses yes / forwards no). The arch ports are their own rows. Spec: .agents/specs/new-candidate-parsers.md

## Resolution

-
