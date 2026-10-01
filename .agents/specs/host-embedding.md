# Host-resident token table: gather the embedding rows on the CPU — ISSUE-LOCAL-01M3QETSTM8X8AJKBM9QCTGBKM

A dense forward that owns a device embedding table uploads `[vocab, H]` once and
gathers there. On a 24 GiB card that is 1.5 GiB of pool for a 151k x 5120 bf16
table, and on a tied head the table is kept for its GEMM whatever the gather does.
The gather itself is small: one row per id.

Issue: [ISSUE-LOCAL-01M3QETSTM8X8AJKBM9QCTGBKM](../issues/ENG-HOST-EMBEDDING/ISSUE-LOCAL-01M3QETSTM8X8AJKBM9QCTGBKM.md).
Owning row: `ENG-HOST-EMBEDDING` ([engine-matrix.md](../engine-matrix.md)), added
with this spec.

## Scope

IN: the host-resident arm of the token-table gather — the `VT_HOST_EMBEDDING`
switch, the CPU gather through `vt::Embedding`, the `[T,H]` copy to the device,
the `EmbedGather` seam that owns both arms, the wake-up of every dense forward
that owns a device table, `docs/ENVIRONMENT.md`, and the tests that pin the arm.

OUT: the tied-head GEMM and every other consumer of the table (the table's host
bytes stay available for them); the decode-graph arms whose ids live on a device
tensor (they keep the device gather by design); any change to the device arm's
bytes or dtype; and the end-to-end VRAM/throughput measurement, which is a row
deliverable recorded under `## Owed`, not a precondition for the arm.

## Upstream chain

**vLLM has no equivalent.** Its embedding is a device-resident
`VocabParallelEmbedding` parameter and the gather runs on the device. This is a
vllm.cpp-original capability, so the primary oracle has nothing to mirror and the
shape comes from a **secondary oracle**: llama.cpp's `ggml_get_rows` /
`ggml_compute_forward_get_rows_q` (`ggml/src/ggml-cpu/ops.cpp:4850` @ `b10451`),
the dequantizing host gather that reads ONE ROW per id and never materializes the
table. That is the same function `tests/vt/test_ops_embedding_quant.cpp` cites for
the device-side `vt::Embedding` op, so both arms are compared against one
denominator.

## Our baseline

Before this change every dense forward that owns a device table staged it with
`ResidentWeight(d, table, {vocab, H})` and gathered with `vt::Embedding(d.q, ...)`.
For an fp8/non-CUDA-device path `ResidentWeight` aliases the host bytes, but on a
staging backend (CUDA, non-unified) it uploads `[vocab, H]` and keeps it resident
for the process. There was no way to keep the table host-side, and no seam that
owned both arms.

## Port map

| Piece | Where |
|---|---|
| The seam and the flag | `include/vllm/model_executor/models/host_embedding.h`, `src/vllm/model_executor/models/host_embedding.cpp` |
| The wake-up call sites | the Qwen3.5 family, MuseGlimmer, the shared Qwen3 dense driver, and the classic dense families (Gemma 1-4, GLM4, Granite, MiniCPM 1/3, OLMo2, OPT, Phi, Phi3, StableLM, Command-R, DeepSeek-V2, GLM-MoE-DSA, Dots3-Note, Nemotron-H, Voxtral) |
| The async id override consumed before the gather | `src/vllm/model_executor/models/qwen3_5_internal.h` (`detail::TakeDeviceTokenIds`, then `detail::ApplyDeviceTokenIds` for the splice) |
| The documented knobs | `docs/ENVIRONMENT.md` (`VT_HOST_EMBEDDING`, `VT_HOST_EMBED_TRACE`) |
| Build registration | the new TU in `CMakeLists.txt` |

## Tests to port

The oracle fixtures already exist and are reused rather than re-derived:
`tests/vt/iq4nl_q5_0_golden_vectors.h` carries the pinned IQ4_NL/Q5_0 golden
vectors (real bytes of the shipped `per_layer_token_embd.weight` and the pinned
oracle's own `dequantize_row_iq4_nl` output) that pin the device-side op
bit-exactly. The host arm is held to the same vectors, so the two arms are
compared against one denominator, and the CPU per-row decode is the llama.cpp
`ggml_get_rows` behaviour this arm ports.

## Design

`EmbedGather(d, out, token_ids, table, vocab, H, what)` is the ONE call a dense
forward makes when it owns a device table. It owns both arms:

1. **Host arm** (`VT_HOST_EMBEDDING=1`): the table stays in `table.bytes` and is
   never uploaded. A process-lifetime CPU queue runs `vt::Embedding` with a
   `ViewOn` of the host bytes, which decodes one row per gathered id for every
   table residency the loaders produce (bf16/f16/f32 and the GGUF block formats).
   A bf16 table into a bf16 output takes a byte-copy fast path. The `[T,H]`
   staging buffer is copied to `out`.
   - The async runner may have spliced this step's sampled token into a
     device-resident ids buffer, so the host arm consumes
     `detail::TakeDeviceTokenIds()` and reads the ids back before the gather. The
     splice runs through the same `detail::ApplyDeviceTokenIds` body the device
     arm uses, so an override longer than the embed input is refused with the
     caller's name instead of writing past the `[T]` buffer, and a SHORTER
     override replaces exactly its prefix while the padded host tail stays.
   - `VT_HOST_EMBED_TRACE=1` prints the arm and `T`.
   - The flag is read ONCE per process (a function-static), so a serving process
     cannot switch arms mid-run; a same-binary A/B is two processes.
2. **Device arm** (the shipped behaviour): `ResidentWeight` upload +
   `ApplyDeviceTokenIds` + `vt::Embedding`, unchanged. Taken when the flag is off,
   when the host bytes are gone (`bytes.empty()` or `host_released`), or when the
   host arm declines for any reason; a one-shot `[host-embed] DISABLED` line names
   the reason.

## Dependencies

- `vt::Embedding`'s CPU kernel and the CPU backend (in-tree, no new library).
- No GPU is needed for the host arm's correctness; a staging backend is needed to
  exercise the `d_dev == nullptr` half.
- The async id override (`detail::TakeDeviceTokenIds`) must exist for the serving
  loop; it is already in-tree from ENG-ASYNC-SCHED W4.

## Work breakdown

- **W1** (this PR): the seam, the flag, the CPU gather, the CPU-queue singleton,
  the bf16 fast path, the async-override consumption, the wake-up of the dense
  forward family, the docs, and the test.
- **W2** (owed): the end-to-end device-memory number and a decode-throughput A/B
  on the target 24 GiB `sm_120a` card, both arms in one binary. Not a precondition
  for the arm; the reason it is not in W1 is that a helper PR makes no speed claim.
- **W3** (later, if wanted): a config surface (`--offload-config`'s `vllm_cpp`
  key) instead of an environment variable, if the operator prefers the config
  document over env.

## Risks and decisions

| Risk / decision | Handling |
|---|---|
| A host gather adds a synchronize on the serving loop | The async runner's device-resident ids are consumed BEFORE the gather, so the host arm reads the spliced ids back rather than racing them; the decode-graph arms that hold ids on device keep the device gather and are listed in the header |
| A tied head keeps the table resident, so the memory saving is smaller | Recorded in the knob's documentation and in the header: the saving is the table's device residency for an untied embedding, and only the gather moves for a tied one |
| The flag is process-static, so a test binary cannot exercise both arms | The test binary enables it before `main` and is flag-ON by construction; the off arm is a plain early return and every other model suite runs it |
| `d_dev == nullptr` does not prove the arm ran on the CPU backend | `ResidentWeight` aliases host bytes when `is_cpu()`, so the test captures the one-shot `[host-embed]` banner and checks `HostEmbedInto`'s return value; the mutation that disables the arm makes 4 assertions fail |
| The host table's bytes are released by another path | The host arm declines on `bytes.empty()` / `host_released`, and `EmbedGather` then takes the device arm; the decline is a test case |
| The runner and the model disagree about this step's row count | The override's count is bounded against the embed input by `detail::ApplyDeviceTokenIds` on BOTH arms; a longer override throws with `what` naming the caller, a shorter one is the padded case and keeps the host upload's tail. Two test cases pin the boundary |

## Evidence

- The host arm's rows equal the pinned oracle through the same golden vectors the
  device op is gated on; the arm is proven to have run (captured banner + return
  value), not merely to have produced right numbers.
- The table is never uploaded on the host arm (`d_dev == nullptr`).
- The async override's shape is bounded: a longer-than-`T` override throws, and a
  shorter one splices its prefix over the host upload while preserving the tail
  (`tests/vllm/models/test_host_embedding.cpp`, the two override cases).
- `tests/vt/test_ops_embedding_quant` (6/6, 1637 assertions) is unchanged.

## Gates

- `ctest --test-dir build -R test_host_embedding`.
- `tests/vt/test_ops_embedding_quant` stays green (the device op is unchanged).
- `python3 scripts/check-env-doc.py` — the two new env vars are documented in
  `docs/ENVIRONMENT.md` in this change (the checker's only remaining complaint is
  the pre-existing `VT_VK_*` gap).
- The full `ctest --test-dir build`.
- `scripts/agent-preflight.sh --staged`.

## Owed

- The W2 measurement above (a 27B NVFP4/Q8mix GGUF on the 24 GiB `sm_120a` card,
  the flag on and off, device memory and decode throughput), owed to the row.
- A dedicated config key (W3) only if the operator asks for one.
