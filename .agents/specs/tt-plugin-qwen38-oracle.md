# SPEC — `TT-PLUGIN-QWEN38-ORACLE` — pin the dflash2-vision serving bundle as a vllm-tt-plugin oracle instance for qwen3.8-27B

- Owning row: `TT-PLUGIN-QWEN38-ORACLE`
- Kind: oracle registration (a secondary oracle instance; numerics reference, never a mirror source)
- Issue: created in the same flow (row-scoped local issue)

## The artifact

`mando2222/qwen3.8-27b-dflash2-vision-p300x2-q4kv-v6-thin-cp4` @
`7855cf0d2982064350991bdb8e1ec1db82f914b3` — a self-contained tt-model
serving BUNDLE (code, not weights): vLLM + the tt-model plugin + custom
kernels, DFlash2 speculative decoding (draft + verify on-card, host
touches each step once), TWO p300c boards = 4 Blackhole chips (1x4 mesh,
FABRIC_1D). Weights download on first use (`Qwen/Qwen3.8-27B`, bf16
safetensors). Greedy output is token-for-token exact to the target
model; sampled output follows the target distribution exactly
(rejection sampling). OpenAI-compatible endpoints; 8 slots; 262,144-token
context.

Its measured serving numbers (their README, one user): 78–110 tok/s
DFlash, 43 plain lane, ~93 @4 users, ~140 @8 users. TTFT cached 0.4–0.6s.

## Why this is an oracle — and what kind

`vllm-tt-plugin` is already a registered secondary oracle (vLLM's own TT
platform plugin). This bundle is a **concrete, self-contained instance**
of that oracle serving qwen3.8-27B — the model the QWEN38-P150-REBENCH
row just measured at 0.03 tok/s output on our single-P150 path. Its
greedy output being exact to the target model makes it a correctness
reference for OUR qwen3.8-27B TT path — the reference the P150 path has
never had (the 09-14 re-bench compared only against itself).

**Scope limits, recorded up front (the registry's honesty contract)**:
- **NOT a throughput denominator at any shared precision**: the bundle
  runs 4 Blackhole chips + DFlash2 speculative decoding + upstream's
  kernel set — the gap to our 0.03 tok/s decomposes into hardware (4x),
  speculative decoding (multi-token/step), kernel maturity, and stack.
  Their number is recorded as *their* number; ours is recorded against
  our own path's profile.
- **NOT numerics-comparable at the token level across hardware**: a
  4-chip mesh's reduction order differs from a single P150's — greedy
  outputs may diverge on near-tie positions for reasons that are the
  ARCHITECTURE's, not either implementation's. Any token comparison is
  architecture-dependent and must be labeled as such (the kolibri1
  band-gate principle: same-weights comparisons only where the
  reduction order matches).

## Deliverable

1. **The oracle record**: `.agents/oracles/vllm-tt-plugin-qwen38.md` (a
   new scoped record in the llama-cpp-flint shape — the registry's
   one-pin-per-file rule) pinning the bundle revision
   `7855cf0d2982064350991bdb8e1ec1db82f914b3` with `gateable = no` and
   the owing issue = this row's issue, OR `gateable = yes` if the
   bundle is actually runnable on THIS host's hardware (it is not: the
   bundle requires 4 Blackhole chips; this host has one P150 — so the
   honest state is `gateable = no` with the hardware mismatch named).
2. **The AGENTS.md registry row**: `vllm-tt-plugin-qwen38` in the oracle
   table — "the dflash2-vision serving bundle for qwen3.8-27B: a
   self-contained vllm-tt-plugin instance on 4x Blackhole; a
   correctness reference for the qwen3.8-27B TT path where the
   architecture matches, never a throughput denominator on different
   hardware".
3. **The gateability adjudication**: the bundle CANNOT run on this
   host's single P150 (4-chip mesh required). Its file records
   `gateable = no` with the hardware mismatch as the reason and the
   named external resource (a 4x-Blackhole box) that would make it
   gateable. The qwen3.8-27B correctness reference this row wants is
   therefore DEFERRED until a gateable instance exists — the row
   delivers the registration and the honest gateability record, not a
   comparison.
4. **The QWEN38-P150-REBENCH cross-reference**: the re-bench row's
   comparison table gains a note pointing at the oracle record — the
   2600x gap to the bundle decomposes (hardware 4x, speculative
   decoding, kernel maturity, stack) and the oracle is the reference
   the gap decomposes AGAINST, when a gateable instance exists.

## Non-goals

- No attempt to run the bundle on this host (impossible: hardware
  mismatch). No speculative-decoding architecture row (a separate,
  much larger row if wanted). No change to the vllm-tt-plugin record's
  own pin. No throughput claim involving the bundle.

## Gates

check-oracle-pins (+ self-test), check-agent-record, style, trailers,
preflight (named host dotnet failure). Device-free row.

## Stop conditions

- If the registry's ID grammar rejects the row ID or the checker's
  surface rules reject the record shape, record the constraint and
  follow the kernel-matrix precedent (a matrix row under the checker's
  namespace, alias recorded in the spec).

## Git integration

One pull request, `row/TT-PLUGIN-QWEN38-ORACLE`, spec committed first.
