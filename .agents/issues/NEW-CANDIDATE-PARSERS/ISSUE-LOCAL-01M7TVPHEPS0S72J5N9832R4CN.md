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

- 2026-11-19: LANDED (parsers only; forwards still OWED to row
  NEW-CANDIDATE-PORTS). Enumerations committed
  (docs/bench-evidence/new-candidates-parsers-20261119/: olala-kvs.txt — GGUF
  v3, 989 TENSORS / 52 metadata KVs; hy4-kvs.txt — 1415 tensors / 76 KVs,
  shard 1; maxwell-fields.txt — config.json field map + all 35 shard headers,
  34270 tensors). Builders: OlalaHfConfigFromGguf
  (src/vllm/model_executor/models/olala_gguf_config.cpp),
  HyV4FlashHfConfigFromGguf (hy_v4_flash_gguf_config.cpp), MaxwellConfigFromJson
  (maxwell_config.cpp); shared KV readers gguf_config_kv.h; GgufFile::Kvs()
  accessor added to gguf_reader.h for the carry-verbatim walk. Loader rows
  added to kGgufArchArms (model_loader.cpp) plus the Maxwell branch on the
  safetensors path; refuse-by-name at the forward boundary names
  NEW-CANDIDATE-PORTS. CPU tests: tests/vllm/models/
  test_new_candidate_parsers.cpp, 11/11 cases PASS against the pinned
  artifacts (fixture paths skip loudly when absent). Findings recorded:
  olala vocab is 151936 (token_embd, not a KV); olala has no per-block
  attn_norm (geo_* family); hy indexer.types covers block_count minus the 1
  nextn layer (42/43); hy shard 2 carries a fork-specific ggml type 43
  (per_layer_token_embd) the reader does not tabulate, so a merged open
  refuses in the READER — port-row work; Maxwell heads*head_dim (4096) is
  twice hidden (2048), confirmed by q_proj [4096, 2048] — recorded, not
  assumed away. Remote code for Maxwell is never executed; the fields it
  would default are recorded UNKNOWN under raw["maxwell_unknown"].
