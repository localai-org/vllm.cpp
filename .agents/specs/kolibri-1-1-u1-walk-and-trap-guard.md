# SPEC — Kolibri-1.1 wave U1: per-arm GGUF walk, arch-collision guard, fit evidence

- Owning row: `MODEL-TEXT-kolibri-1-1-kolibri1-1-for-causal-lm`.
  Guard issue: ISSUE-LOCAL-01M84HPYRT3BNNF61BKGNRQ740.
- Base: pinned assessment head `c96876895d7e840b8c63db8ddc037a8bedc9acb2`
  (branch `row/MODEL-TEXT-KOLIBRI11-ASSESS`, pushed to `mine`, no PR). Branching
  from it keeps the canonical issue, the assessment spec and the oracle records
  in the same U1 PR, so they become base-reachable together. One PR carries spec
  and implementation.
- Developer/coordinator authority: oc-41 assigned U1 with explicit go-ahead;
  user authorized the download and study. Worktree `/tmp/vllm-kolibri11-u1`,
  branch `row/MODEL-TEXT-kolibri-1-1-U1`.
- Status: SPEC FIRST. No implementation in this commit.

## Trigger / problem

The assessment (`docs/bench-evidence/kolibri11-assessment-20261123/ASSESSMENT.md`,
commit `c96876895`) found that llama.cpp's Kolibri-1.1 converter writes
`general.architecture = "kolibri1"` for 1.1 GGUFs. Our GGUF arm claims that same
string (`kKolibri1GgufArch`, `include/vllm/model_executor/models/kolibri1_gguf_weights.h:25`,
checked in `IsKolibri1Gguf` at `src/vllm/model_executor/models/kolibri1_gguf_weights.cpp:183-187`
and dispatched at `src/vllm/entrypoints/model_loader.cpp:1297`). A 1.1 GGUF would
therefore be silently claimed as 1.0: indexer KVs ignored, `indexed_attention`
layers called "full", and missing shared-KV tensors surfacing only as an
incidental load error. This must be a named refusal, not a misleading load.
U1 also owes the per-arm header/tensor walks that ground later waves and the
fit finding that decides which arm is worth porting first.

## Upstream anchors

- llama.cpp PR `ggml-org/llama.cpp#32891` (OPEN 2026-11-23, head `5f4d222898dd`):
  converter `conversion/kolibri1.py` reuses `MODEL_ARCH.KOLIBRI1` for 1.1;
  `src/models/kolibri1.cpp` implements the lightning-indexer path. Unmerged, so
  it cannot be a pinned oracle and no token-parity result is claimed here.
- Official GGUF: `Aleph-Alpha/Kolibri-1.1-GGUF` @
  `90116812a1669494a894b729e677ed75d6068320`, Q4_K_M LFS oid `ee511452…1ab10`.
- Assessment evidence: `ASSESSMENT.md` and the 995-line `walk-q4km.txt` in
  `docs/bench-evidence/kolibri11-assessment-20261123/`.

## Scope

IN:
1. Download and verify four pinned arms into `/mnt/models/Aleph-Alpha-Kolibri-1.1-GGUF`:
   IQ1_M (15.44 GiB), IQ2_XXS (17.65), IQ2_XS (19.87), Q4_K_M (42.38). Each
   file is verified against its HF LFS sha256 from the blobs API; the expected
   values are recorded in U1 evidence. No walk starts before verification.
2. One header/tensor walk per verified arm: metadata KVs, tensor census, ggml
   type ids, KV names, offsets and the indexer/KV-source KV set. Evidence under
   `docs/bench-evidence/kolibri11-u1-20261123/` (flat, allowed extensions).
3. The trap guard: `Kolibri1HfConfigFromGguf` (or `IsKolibri1Gguf`) refuses a
   `kolibri1` GGUF that carries any Kolibri-1.1 marker KV:
   `kolibri1.attention.indexer.top_k`, `kolibri1.attention.indexer.types`,
   `kolibri1.attention.kv_source_map`, `kolibri1.attention.indexer.candidate_block_size`
   or `...candidate_top_k_blocks`. The message names the missing `kolibri1_1`
   support and this row, per AGENTS.md "Refuse an unimplemented arm with a
   message that names the missing part".
4. A red-first, mutation-proven unit test over a hermetic synthetic GGUF header
   carrying `general.architecture=kolibri1` plus an indexer KV, asserting the
   refusal by name; plus a control that a plain 1.0 GGUF still loads (no false
   positive). Production-path reach: the test enters through
   `HfConfigFromGgufDispatch`/`LoadedEngine::FromModelDir`, not by calling the
   helper directly, so deleting the loader row or the guard turns the gate red.
5. Fit evidence in the spec/row evidence: Q4_K_M at 42.38 GiB cannot fit a
   32 GB P150 alone; only the three GSQ-RCO arms are near that budget, with KV
   and activation headroom still an open calculation. Accepted by oc-41 as U1
   evidence.

OUT: the indexed-attention kernel, the dense-layer prefix, paged NVFP4 KV,
token parity, model weights pins in `docs/USAGE.md`, and any oracle pin advance.
There is no gateable CPU oracle today, so U1 claims no token-parity result.

## Our baseline

- `Kolibri1HfConfigFromGguf` at `kolibri1_gguf_weights.cpp:219` builds a 1.0
  HfConfig and synthesizes `architectures={"Kolibri1ForCausalLM"}`; it reads no
  indexer or KV-source KVs. `EnumerateKolibri1GgufTensors` (:191) expects every
  layer to carry `attn_k`/`attn_v`, which 1.1 shared-KV layers drop.
- The HF safetensors path is already correct: `ModelRegistry::Resolve` keys on
  the exact architecture string, so `Kolibri1_1ForCausalLM` is refused with an
  unsupported-arch message naming the architecture. The trap is GGUF-only.
- Hermetic GGUF builder precedent: `tests/vllm/gguf_builder.h`, used by the
  inkling2 loader-reachability test in PR #3586.

## Tests

1. RED: add the synthetic-GGUF production-path test; confirm it fails because
   the current tree accepts the 1.1-shaped header. Capture the red output.
2. Add the guard; focused test green, control (plain 1.0) still green.
3. Mutation proof in a scratch copy: (a) remove the guard, (b) remove the
   `kGgufArchArms[]` row at `model_loader.cpp:1297`; each must turn the focused
   gate red. Restore byte-for-byte after each.
4. Run the related ctest subset and the full gate.

## Gates / evidence

- `sha256sum -c` equivalent against the recorded HF LFS oids for all four arms,
  captured as evidence before any walk.
- `scripts/agent-preflight.sh --staged`, `check-agent-record.py --report` rot 0,
  commit style/trailers, `agent-pr-body.py --pr <N>` exit 0.
- No throughput, latency or token gate is claimed in U1. "No gateable CPU
  oracle" is the honest report, not a waiver.

## Dependencies / risks

- Pinned artifacts must be available and verifiable; the download is in flight
  at spec time. If any arm fails verification, that arm's walk is PENDING and
  is named, not guessed.
- The marker-KV set comes from the assessment walk; if a verified arm shows a
  different marker, the guard list is updated with evidence in the same PR.
- `kv_cache_qat=true` (NVFP4 KV) is a later wave; U1 does not implement it.

## Work breakdown

U1a verification + four walks; U1b trap guard + mutation-proven test; U1c fit
evidence and records. One PR, spec first.

## Stop conditions

`NEEDS_CONTEXT` if an arm cannot be verified or the pinned base is unavailable;
`NEEDS_DECISION` if a marker KV contradicts the assessment. Do not stop at a
correctable test failure. Do not claim token parity or publish a benchmark.

## Git integration

One PR on `row/MODEL-TEXT-kolibri-1-1-U1` to `localai-org/vllm.cpp:main`, push
only to `mine`. Trailers and PR body per AGENTS.md. Fresh reviewer must PASS
before the unit is called merge-ready. Never merge or close a PR here.

## Now

SPEC — download in progress; no walk or guard written yet. Implementation waits
for per-arm sha256 verification.

## Owed

- U2-U6 from the assessment: dense prefix, indexed attention, paged NVFP4 KV,
  serving surface and quantized-arm gates.
- A gateable oracle (llama.cpp PR #32891 merge/pin, or a CUDA lease for the v2
  plugin) before any token-parity claim.
