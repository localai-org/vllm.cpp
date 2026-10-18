# Kolibri-1 TT B2b-i dense-resident forward — completion evidence (2026-10-08)

Row MODEL-TEXT-kolibri-1-tenstorrent, slice B2b-i (spec
`.agents/specs/kolibri-tt.md` ### B2 scope — B2b addendum, slice i; issue
ISSUE-LOCAL-01M4E22DM790W0E69M07D5XA9D). Branch `row/kolibri-tt-b2bi-fwd`,
implementation commit `cbdd7cce5` (+ this evidence commit).

## 1. Build recipe

Worktree `/tmp/vllm-kolibri-tt-fwd`, fresh build dir `/tmp/build-kolibri-tt-fwd2`:

```sh
cmake -S /tmp/vllm-kolibri-tt-fwd -B /tmp/build-kolibri-tt-fwd2 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_TENSTORRENT=ON \
  -DCMAKE_PREFIX_PATH="/tmp/pin-build/lib64/cmake;/tmp/pin-build/share/cmake"
```

The build links against the PIN tree's fresh libs (`/tmp/pin-build`,
symbol-verified 2026-10-08). NOTE recorded against the earlier expectation:
the NON-pin tree (`~/Sources/tt/tt-metal`, d20b8e27f29) does NOT link this
row's TT test binaries — its `build_Release/lib64/_ttnncpp.so` (2026-09-18)
exports `chunk_gated_delta_rule` WITHOUT the `use_mcast` parameter and with a
by-value `ttnn::ComputeKernelConfig`, so the declaration in
`src/vt/tenstorrent/tenstorrent_internal.h:113` does not resolve and the link
fails with an undefined reference. The "no-op stub" in the Owed section is
therefore not reachable for a vLLM TT link; the pin fresh libs are the only
working link on this host. Also: `~/Sources/tt/env-tt-common.sh` sets
`LD_LIBRARY_PATH` to the non-pin lib64, which preempts the binary's RUNPATH
(`/tmp/pin-build/lib64`) and aborts at load — device runs must use
`LD_LIBRARY_PATH=/tmp/pin-build/lib64:/tmp/pin-build/libexec/tt-metalium:...`
with `TT_METAL_HOME=TT_METAL_RUNTIME_ROOT=~/Sources/tt/tt-metal-pin`.

## 2. Lease identity and window

Host-local P150, file mutex `${HOME}/gpu.lock` (this box is not a fleet
device). Window 2026-10-08 ~19:47–19:53 UTC: card reset
(`~/Sources/tt/luwen/target/release/reset` + 15 s + `rm -rf
~/.cache/tt-metal-cache/*`), then the device leg under `flock`. KMD 2.10.1 /
fw 19.7.1 (from the device leg log).

## 3. Host-side gates (no card)

| Gate | Result |
|---|---|
| test_kolibri1_tt_b2bi (NEW, host cases) | 8 cases, 62 assertions — first run RED (3 failures: a wrong tie-break expectation in the routing case — e0/e2 tie at biased score 2.0 breaks to the LOWER index, which is the correct contract; a layer-count bug in the byte expectation; a too-tight manifest-rounding bound), then GREEN after fixing the TEST expectations. No product assertion was weakened. |
| test_kolibri1 | 234/234 |
| test_kolibri1_w2 | 1608/1608 |
| test_kolibri1_w3 | 900/900; ARGMAX CHAIN 141/145, 4 near-tie flips, 0 hard — identical to the landed CPU-row baseline (the kolibri1_shared.h relocation is byte-identical) |
| test_kolibri1_tt | 235/235 (B2a planner contracts unchanged) |
| test_kolibri1_tt_b2i | 204/204 |
| test_kolibri1_moe_glue | 21/21 |
| test_kolibri1_dequant | 10/10 |
| test_kolibri1_decode_bench | anchor last token 109726, 2/2 |
| test_kolibri1_dequant_cache | 48/49 — PRE-EXISTING at HEAD `129e997a7` (verified by building that test at HEAD in a scratch worktree): `fork()` returns -1 in the default-off probe (line 335) on this host/TT env; unrelated to this change |

`scripts/agent-preflight.sh --staged`: green (PF_RC=0) with the foreign
claim-file edit set aside byte-for-byte; the file was restored uncommitted
afterwards. The foreign edit (CLAIM-KERNEL-CUDA-DECODE-MEGAKERNEL, SPIKE→
ACTIVE) itself breaks `check-agent-record` in the shared worktree — it is
another agent's live state, never committed here.

## 4. Staged byte totals on device

From the device leg log:

- context build: **350 projections** (7 per layer × 50 layers: attention
  q/k/v/o + shared expert gate/up/down), **3,801,088,000 B bf16 = 3.540 GiB**
  uploaded in 1.0 s.
- The spec byte-math plan's resident fp8 set (attention 1.587 GiB + shared
  expert 0.188 GiB) times 2 (bf16 dequant) matches: the manifest-driven
  host case measures 350 resident fp8 projections and a bf16 cost exactly
  2× the fp8 bytes, within 8 MiB of the plan's 3-decimal rounding (measured
  delta 5.1 MiB).
- The bf16 modules (embed, lm_head, norms, q/k norms, router gate bf16 +
  bias) ride the shared `dense_attn::ResidentWeight` seam, per the wave-A
  design.

## 5. Completion condition: ONE greedy decode on the card

```
[kolibri1-tt-b2bi] device leg start: model=/mnt/models/Aleph-Alpha/Kolibri-1
[kolibri1-tt-b2bi] load: 18.9 s
[kolibri1-tt-b2bi] context: 350 projections, 3801088000 B bf16 (3.540 GiB) in 1.0 s
[kolibri1-tt-b2bi] prefill done: 6 tokens -> first argmax 25079
[kolibri1-tt-b2bi] decode step 0..7: argmax 109602, 45, 127907, 55598, 127907, 55598, 127907, 55598
[kolibri1-tt-b2bi] GREEDY DECODE COMPLETE on card: 15 tokens in 63.8 s;
  routed-expert refusals fired 450 times (by name)
[doctest] assertions: 78 | 78 passed | 0 failed
[doctest] Status: SUCCESS!
```

The routed-expert refusal fired **450 = 50 layers × (1 prefill + 8 decode)
steps** times BY NAME (message names the missing part, the owning slice
B2b-ii, the row, and the issue; full text once per process in the raw log
`/tmp/b2bi-device.log`). The decode ran the resident components end-to-end
through the production seam (`ModelRegistry::Prepare` builds the device
context; `ModelRegistry::Forward` runs every step; greedy argmax on the
downloaded f32 logits).

The goldens' expected tokens are NOT asserted, deliberately: the goldens are
full-model decodes and cannot be replayed without the routed experts. The
141/145 token gate stays owed to B2b-ii — the refusal firing by name is the
recorded proof, per the addendum's gate ordering.

## 6. Per-op device-vs-CPU agreement

Stated envelope (inherited from the b2i bring-up, measured 2026-10-08, worst
ratio 1.375): per element `|dev − cpu| ≤ 8·2⁻⁸·Σₖ|a_ik·w_jk| +
max(ulp(|cpu|), ulp(|dev|))`. The 2-ulp accumulation-order premise stays
falsified and is not restated.

- The embedding gather is bit-exact by construction (row gather, no
  arithmetic); its agreement is asserted through the decode: a wrong gather
  cannot reach the same first argmax the resident path produces.
- Every resident GEMM consumes identical bf16 dequant operands on both sides
  (memoized `DequantRowsBf16`); the host case pins those dequants
  BIT-EXACTLY against the CPU row's function over the fixture, and the b2i
  bring-up already measured the device GEMM envelope on this path (2026-10-08,
  max_err_ratio 1.375, violations 0).
- No new GEMM envelope was re-measured in this leg: the slice introduces no
  new kernel, only the memoized-residency difference documented in the
  forward's header.

## 7. Red-first evidence

- The inherited draft had NEVER been built: the first build failed with two
  compile errors in the draft code (`kolibri1_registry.cpp:316` const-char*
  + const-char*; unused variable in `kolibri1_tt_forward.cpp:280`). Fixed
  forward, minimum change.
- The new gate TU's first run was red (3 test-expectation failures, §3);
  each was a defect in the TEST's expectation, not in the product code, and
  the routing one double-pins the tie-break contract.
- The `test_kolibri1`/`w2`/`w3` suites pin the kolibri1_shared.h relocation
  byte-identical (identical counts to the landed baseline).

## 8. What remains owed

- B2b-ii: the routed-expert streaming tier (slot pool, fetch executor, the
  router readback dtype pivot, the runtime stream-bound assert).
- The full-model token gate (141/145 argmax, 4 flips in the 2.5-nat band,
  0 hard) and the production bench anchor — only after B2b-ii.
- Durable copy of `/tmp/pin-build` lib64 into the pin tree (operator
  decision, unchanged).

## 9. Review repair — host-side coverage for the three mutation findings (2026-10-08)

The fresh mutation review returned FAIL with three findings, all of one form:
the guarantee had no host-side coverage, so every mutation left all card-less
gates green. The repair adds a host op census over the PRODUCTION forward —
a host-memory backend + platform registered in the kTENSTORRENT slot (the
`test_resident_weight_host_addressable.cpp` pattern) plus recording op
providers over the EXISTING `vt::OpProvider` seam (priority 100, one
test-only provider name, DISABLED on scope exit, previous backend/platform
restored) — so the forward runs end-to-end on the host while the census
records which ops fired, in what order, consuming which norm weight. The
tiny fixture's norm weights now carry DISTINCT bf16 sentinels (input_ln 1.0,
post_attn 2.0, post_attention 3.0, post_ffn 4.0, q_norm 0.25, k_norm 0.3125,
final 0.5), so the census identifies WHICH weight a recorded norm consumed.
No device kernel executes; no parallel forward path exists; the device leg's
semantics are untouched (`TenstorrentPresent()` excludes the stand-in, the
guard restores the real backend and disables the recorders).

New host cases (test_kolibri1_tt_b2bi.cpp):
- "HOST: the forward's op census" (line 825): asserts the per-step op counts
  and ORDER (1 embedding; per layer q,k,v,o + router + shared gate,up,down
  matmuls, per-head q/k norms, RoPE on the sliding layer only, KV write +
  paged attention, MoeSiluMul; 1 lm_head matmul) and the norm IDENTITY
  sequence — each sandwich norm consuming ITS OWN sentinel weight in the CPU
  row's order, with the residual-carrying norms exactly input_ln /
  post_attention_layernorm / final — plus the refusal-firing contract
  HOST-SIDE: `refusals >= layers x (1 + 2 steps)` (the device leg's
  assertion, decoupled from the card).
- "HOST: the registry's kTENSTORRENT dispatch arm resolves the
  dense-resident forward" (line 918): Resolve -> Load -> Prepare on a TT
  queue -> ModelRegistry::Forward completes and the refusal fires, THROUGH
  the production seam.

Red-first capture (each mutation is the reviewer's EXACT mutation; the full
battery below stayed at its landed counts except the named cases):

- MUTATION 1 — `NoteRoutedExpertRequest` counter increment silenced
  (kolibri1_tt_forward.cpp:103): RED host-side —
  `test_kolibri1_tt_b2bi.cpp:851 CHECK(refusals >= 2*(1+2)) NOT correct` and
  `:963 CHECK(count - before >= 2) NOT correct` (10 cases: 8 passed,
  2 failed; 180/182). Restored byte-for-byte: 10/10, 182/182 GREEN.
- MUTATION 2 — post_attn_norm swapped to `lw.input_layernorm`
  (kolibri1_tt_forward.cpp:478): RED host-side — `:907 CHECK(r.weight ==
  expected.word) NOT correct`, norm records 3 and 9 recorded sentinel
  0x3F80 (input_ln) instead of 0x4000 (post_attn) in BOTH layers (10 cases:
  9 passed, 1 failed; 180/182). Restored: 10/10, 182/182 GREEN.
- MUTATION 3 — kTENSTORRENT dispatch arm replaced with a throw
  (kolibri1_registry.cpp:308): RED host-side — `:918 test case THREW
  exception: Kolibri1ForCausalLM: the Tenstorrent forward arm is not
  implemented.` (10 cases: 9 passed, 1 failed). Restored: 10/10, 182/182
  GREEN.

Post-repair host battery (build dir `/tmp/build-b2bi-repair`, pin libs,
2026-10-08): test_kolibri1 234/234, test_kolibri1_tt 235/235,
test_kolibri1_tt_b2i 204/204, test_kolibri1_tt_b2bi 10 cases / 182
assertions (was 8/62), test_kolibri1_dequant 10/10, test_kolibri1_moe_glue
21/21, test_kolibri1_w2 1608/1608, test_kolibri1_decode_bench 2/2 (anchor
109726), test_kolibri1_w3 900/900 ARGMAX CHAIN 141/145 (4 near-tie flips,
0 hard) — identical to the landed baseline. test_kolibri1_dequant_cache
48/49: the documented PRE-EXISTING fork() failure in the default-off probe
(line 335) reproduced identically on this host/TT env (retried isolated
twice); unrelated to this change, which that binary does not compile.

No device leg was re-run for this repair (host-side only, per the review
scope); the device leg's semantics are exactly as landed at `cbdd7cce5`.
