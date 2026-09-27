# Chunked GDN prefill on every backend, with the sequential arm behind a flag — `KERNEL-GDN-CHUNKED-MIRROR`

Issue [#2612](https://github.com/mudler/vllm.cpp/issues/2612). Intended row
`KERNEL-GDN-CHUNKED-MIRROR` — **not yet in [kernel-matrix.md](../kernel-matrix.md)**.
The spec comes first (AGENTS.md: a row cannot become `READY` or `ACTIVE` without
a committed spec), and this one is a `SPIKE`, so the matrix row and the
`check-agent-record.py:402` count bump `58 -> 59` are owed by the change that
moves it to `READY`, not by this one. Until then #2612 is owned by this file's
`## Owed`, which is the alternative AGENTS.md names. vLLM parity pin
`5559679229bc961848b121ccdeaa8fa5d79bec98`
([upstream-sync.md](../upstream-sync.md)). Every upstream anchor below was read
at that revision, in the vendored tree
`vllm/third_party/flash_linear_attention/ops/`.

Measurement:
[`docs/bench-evidence/gdn-chunked-decomposition-20260902.md`](../../docs/bench-evidence/gdn-chunked-decomposition-20260902.md)
(wave GDNDECOMP) and
[`docs/bench-evidence/qwen4exp-cuda-prefill-divergence-20260902.md`](../../docs/bench-evidence/qwen4exp-cuda-prefill-divergence-20260902.md)
(wave PREFILLDIV).

## Scope

vLLM runs the chunked WY decomposition for **all** GDN prefill, on every path.
`QwenGatedDeltaNetAttention._forward_core` has no sequential branch
(`qwen_gdn_linear_attn.py:1424-1450`), and `forward_native` (`:266-294`) is the
same Triton chunk kernel rather than a torch fallback. The sequential kernel our
CPU arm ports, `fused_recurrent_gated_delta_rule`, is not referenced by that
layer at all; **among the model layers** its only callers are
`olmo_gdn_linear_attn.py:430,473`, both decode. Repo-wide it has two more
references, `tests/kernels/test_fused_sigmoid_gating_delta_rule.py:71,166`, which
use it as the reference an upstream test compares the fused decode kernel
against. That does not weaken the layer claim; it does mean "referenced only by
`olmo`" is false as a repo-wide statement and is not written that way here.

`vt::GdnPrefill` runs the chunked algorithm on CUDA
(`src/vt/cuda/cuda_gdn.cu:6117 GdnPrefillChunkedCuda`, default on) and on
Tenstorrent. It runs the exact sequential recurrence on **CPU**
(`src/vt/cpu/cpu_ops.cpp:1862 GdnPrefillKernel` -> `:1811 GdnHeadTokenStep`),
**ROCm** and **Vulkan**. Those three arms do not mirror vLLM, no record says so,
and `porting-inventory.md` §9 carries no entry for it.

**The developer has decided: default to mirroring vLLM, and put the algorithm
choice behind a flag.**

**In scope.** A chunked GDN prefill for the CPU, ROCm and Vulkan arms, with
upstream's dtype placement; the flag that selects it, on every backend; the
retention of the sequential recurrence as one tracked exception; the goldens and
gates that change meaning when the default moves; and the records that today
state the old arrangement.

**Out of scope, each named again under `## Owed`.** GDN *decode*, which upstream
does run sequentially and where our arms already mirror it. The `qwen4_exp` MoE
residue of `7.269e-05`/layer that PREFILLDIV §3 isolated and did not diagnose.
Speed: this row's default flip is a correctness move and carries no throughput
claim; the chunk-parallel form exists upstream for speed, but establishing that
on a CPU arm is a separate row.

**What this spec does NOT promise: token agreement.** PREFILLDIV measured
`VT_GDN_CHUNKED=0` on CUDA cutting prefill divergence 332x while agreeing on
*fewer* token ids (3 of 8 against 5 of 8). Agreement between two of our arms is
an argmax over near-ties and is not a monotone function of the distance between
them. No gate in this spec is a CPU-vs-CUDA token gate, and no reviewer should
accept one as evidence that this row succeeded.

## The measurement that decides the design

Full method and validation in the evidence file. The three numbers that bind
this spec:

| quantity | value |
|---|---|
| vLLM's chunked kernel, distance from the exact recurrence | `2.286426e-04` out, `2.248661e-03` state |
| of which the **chunked reassociation** | `2.428613e-17` out, `1.110223e-16` state |
| of which the **bf16 intermediates** | the entire remainder |

The chunked WY decomposition is an algebraic identity and behaves like one:
evaluated in `float64` it reproduces the sequential recurrence to `2.4e-17`.
Carried at f32 it costs `3.86e-08`, 3.4x our sequential arm's `1.15e-08` and
four orders below the bf16 term. A numpy replica given nothing but upstream's
bf16 placement reproduces the real kernel's distance from exact to seven
significant figures.

**Consequence, and it is the whole design.** At the production output dtype
(bf16, 8192 elements of the golden):

| arm | bf16 output elements differing from vLLM's kernel |
|---|---|
| chunked with upstream's bf16 placement | **62 / 8192** |
| chunked with f32 intermediates | **4157 / 8192** |
| our sequential arm today | **4157 / 8192** |

An f32-intermediate chunked port differs from our own sequential arm in **2 of
8192** elements. It is not a third answer. **It is our current answer wearing
vLLM's algorithm** — the mirror claim would be unearned and any gate resting on
it would be a tautology.

## Design

### D0. The default is chunked FOR BF16 AND SEQUENTIAL FOR F32, because that is upstream

`chunk.py:213-215` asserts `q.dtype != torch.float32`. **vLLM cannot run its
chunked prefill on f32 inputs and does not have a chunked f32 path to mirror.**
Given f32, mirroring vLLM means running the sequential recurrence, because that
is the only gated delta rule upstream will execute at that dtype.

The tree already knows this in two places. `test_op_parity.cpp:590-595` forces
`VT_GDN_CHUNKED=0` for f32 goldens on CUDA, with the reason in a comment. And
the three f32 goldens are dumped from
`fla/ops/fused_recurrent.py::fused_recurrent_gated_delta_rule` — the SEQUENTIAL
kernel — at `atol = rtol = 1e-05`, precisely because the chunked wrapper refused
to produce them. Only `gdn_prefill_bf16_realdims` is a chunked dump.

**So the default predicate is `chunked = (q.dtype != f32) && VT_GDN_CHUNKED != 0`,
on every backend.** This is not a concession to the goldens; it is the mirror.
An implementer who reads "chunked by default" as "chunked always" will make the
three f32 goldens fail by one to two orders of magnitude — `1e-05` against a
recorded chunk-vs-sequential gap of `2.44e-04` — and the correct repair is not to
loosen them. It is `## Stop conditions` item 3.

**The alternative was considered and rejected.** Upstream does not *fall back*
on f32, it *asserts*, and AGENTS.md asks an unimplemented arm to refuse with a
message naming the missing part. Refusing f32 by name would be the closer
mirror of the assertion — and it would break three committed goldens and every
f32 GDN op test in the tree for a dtype our own engine never puts on the model
path (`state` is f32, but `q`/`k`/`v` reach `GdnPrefill` in the model dtype).
Routing f32 to the sequential recurrence keeps that coverage alive and computes
the same recurrence upstream's own f32 kernel computes. Tenstorrent, which has
no sequential arm at all, is the one backend that must refuse rather than route
(D3).

**D0 CHANGES CUDA's DEFAULT TOO, and that is a second backend whose production
path moves.** `ChunkedPrefillEnabled()` (`cuda_gdn.cu:3265-3271`) carries no
dtype term, and neither does the six-term conjunction at `:6218` — `wmma_ok`
constrains bf16 dims only. **CUDA therefore takes the chunked arm for f32
inputs today.** That is why `tests/parity/test_op_parity.cpp:595` has to set
`ScopedEnv("VT_GDN_CHUNKED","0")` at all: without it the f32 goldens would
replay through a chunked kernel upstream refuses to run. Under D0 that
`ScopedEnv` becomes redundant, because the dtype predicate does what the comment
above it describes. Leave it in place, restate the comment, and say in the
commit that CUDA's f32 default moved — an f32 CUDA request that runs chunked
today runs sequentially after this row, which is a behaviour change on a shipped
backend and not only a new arm on three others. Whether to then DELETE the
`ScopedEnv` is the implementer's call; deleting it removes a belt while the
braces are new, so keeping it is the recommendation.

A consequence worth stating: the f32 chunked arm is then **not covered by any
upstream test**, because upstream has no f32 chunked case to port. Whatever f32
coverage we ship there is ours and must be recorded as an adaptation, never as a
port.

### D1. The chunked CPU kernel mirrors upstream's dtype placement

Structure follows `chunk.py:37-82` and the CUDA arm's decomposition
(`cuda_gdn.cu:3243-3460`), which is already stage-for-stage aligned with the FLA
sub-ops. Per chunk of `BT = FLA_CHUNK_SIZE = 64` (`utils.py:31`, matching
`cuda_gdn.cu:169 kChunk`):

```
G      = inclusive cumsum of g within the chunk               f32   cumsum.py:160-280
A[i,j] = beta_i (k_i . k_j) exp(G_i-G_j),  j<i                f32   chunk_scaled_dot_kkt.py:86-116,161
Ai     = (I + A)^-1                       f32 solve, STORED   BF16  solve_tril.py:227-505 + chunk.py:50
u      = Ai @ bf16(beta v)                f32 acc,  STORED    BF16  wy_fast.py:92-94
w      = Ai @ bf16(beta exp(G) k)         f32 acc,  STORED    BF16  wy_fast.py:114-116
h_c    = the chunk-start state                        STORED  BF16  chunk_delta_h.py:352
v_new  = u - w @ bf16(H)^T                f32 acc,  STORED    BF16  chunk_delta_h.py:178,206,357
H      = H exp(G_last) + k^T bf16(v_new exp(G_last-G))  H in  F32   chunk_delta_h.py:208-302
o      = exp(G)(q.h_c)*scale + (bf16(A_o) @ v_new)*scale STORED BF16 chunk_o.py:111-138
final_state                                                    F32  chunk_delta_h.py:353-355
```

**Two rounding sites the existing record does not name, and both are load
bearing.** The two `*scale` factors above are written separately on purpose:
`chunk_o.py:137` is `b_o * scale + tl.dot(...) * scale`, not
`(b_o + tl.dot(...)) * scale`, and the two are not the same in f32.

`chunk_delta_h.py:178` casts the f32 register state down to bf16 *as
an operand* of `w @ h^T` — the state is f32 and the product that reads it is
not. `chunk.py:50` passes `output_dtype=k.dtype` to `solve_tril`, so the
triangular inverse is stored bf16 even though `A` itself is f32. Neither appears
in the dtype table of `qwen4exp-cuda-prefill-divergence-20260902.md` §4. Per-site
ablation (evidence §3) shows no single site is the mechanism: removing any one
leaves the gap intact, and each alone is worth `1e-6` to `2e-4`.

**`A` and `final_state` stay f32.** That is upstream, not a concession
(`chunk.py:47`, `chunk_delta_h.py:353-355`).

**The `K K^T` dot is ieee f32, not TF32.** `chunk_scaled_dot_kkt.py:103` keeps
the promoted f32 operand on non-RDNA. A TF32 variant was measured and lands
*further* from the real kernel (`1.2207e-04` against `6.1035e-05`), which is the
evidence that the ieee path is the one that runs.

### D2. `A^-1` by forward substitution, and why that is still a mirror

Upstream inverts `(I+A)` blockwise: four 16x16 unit-triangular inverses by
in-register substitution, then six off-diagonal blocks by `tl.dot` chains
(`solve_tril.py:356-394`). Our CUDA arm instead fuses the solve into the two WY
columns by forward substitution (`cuda_gdn.cu:3322-3348`), which is the same
linear algebra with a different association and no materialised inverse.

**The CPU arm may take either shape, but it must round where upstream rounds.**
Forward substitution has no `A^-1` to store, so the `chunk.py:50` bf16 site has
no home in it. A port that fuses the solve therefore skips a rounding site that
the oracle performs. Ablation puts that one site alone at `2.6174e-06` on `out`
— small, and it is the only site of the nine that a fused solve structurally
cannot reproduce. **Decision: materialise `A^-1`, round it to bf16, then apply
it.** The cost is one `[64,64]` per chunk-head. Fusing it is an optimisation this
row does not take, and if a later row takes it, it takes it with this number
named.

### D3. The flag: `VT_GDN_CHUNKED`, extended, not duplicated

`VT_GDN_CHUNKED` exists today, CUDA-only, read once at
`src/vt/cuda/cuda_gdn.cu:3269` by a bespoke `ChunkedPrefillEnabled()`, and
consumed as the first term of a six-term conjunction at `:6218`. It is on the
kernel-internal allowlist (`scripts/env-doc-allowlist.txt:71`) and deliberately
absent from `docs/ENVIRONMENT.md`.

**The CPU, ROCm and Vulkan arms join that same flag. They do not get their own.**
One name, one meaning: *select the sequential recurrence instead of vLLM's
chunked decomposition*. Two flags for one semantic choice is how a `--device cpu`
run and a `--device cuda` run end up on different algorithms because only one of
them was set, which is the exact confusion this row exists to remove.

Mechanically: lift `ChunkedPrefillEnabled()` out of `cuda_gdn.cu` into the shared
op layer beside the other `vt::` env readers, so every backend's `GdnPrefill`
entry point reads the same predicate. The existing bespoke parse
(`e == nullptr || e[0] != '0'`) is kept verbatim rather than rewritten to
`EnvOnOr`, because changing the accepted spellings of a flag whose off-value is
recorded in four evidence files and six spec lines is a semantic change wearing a
cleanup.

**Tenstorrent has no sequential arm and no flag.** Its prefill
(`src/vt/tenstorrent/tenstorrent_ops.cpp:4262`) delegates unconditionally to
`ttnn::transformer::chunk_gated_delta_rule`. It joins the predicate for the f32
half of D0 — an f32 request must not silently reach a bf16-only chunked op — and
it either gains a sequential fallback or refuses f32 by name. Refusing by name
is acceptable and is the pattern AGENTS.md asks for; refusing silently is not.
Metal has no GDN kernel at all (`src/vt/metal/metal_ops.mm:29`) and is out of
scope.

**`VT_GDN_CHUNKED` gains a row in `docs/ENVIRONMENT.md`.** Today it is on the
kernel-internal allowlist (`scripts/env-doc-allowlist.txt:71`) and absent from
the documented surface, which is correct for a CUDA-only bisect knob. After this
row it selects, on every device, which of two documented algorithms the engine
runs, which is exactly the "supported operational lever" the ENVIRONMENT.md
preamble carves out. Its sibling `VT_GDN_OUT_BF16` is already documented there
(`docs/ENVIRONMENT.md:228`) for the same reason. Add it to the
`## Rollback and bisect switches` table, whose stated semantics ("default-on
fast paths, each with an off switch") it now matches.

**The flag stays out of `include/vllm.h`.** This is the one place the spec
declines an obvious extension, and the reason is that the ABI has no
algorithm-choice surface at all: `vllm_model_params` (`include/vllm.h:377-683`)
selects device, offload backend and scheduler policy, and stops. There is no
kernel key, no numerics key, no precedent to follow. AGENTS.md requires every
shipped *capability* to be reachable through the ABI; the shipped capability here
is **GDN prefill that mirrors vLLM**, and after this row that is what
`vllm_engine_load` gives you on every device, with no key to set. The flag
selects a *deliberately non-mirroring* arm. Exposing it in the ABI would make the
non-mirror a supported product surface, which is the opposite of what D4
records it as.

If a later row needs it in the ABI, the precedent to copy is
`enable_jump_forward` (`include/vllm.h:553-568`): a tri-state `int32_t` whose 0
is the byte-identical default and over which the env var still wins. That row
would owe a reason this row does not have.

### D4. The tracked exception

AGENTS.md: *"Extend a shared seam when it cannot represent the upstream
behavior. Otherwise, record one exact tracked exception. Never write a parallel
path by hand."*

The sequential recurrence is retained as **one** tracked exception, recorded as a
new entry in `porting-inventory.md` §9, with this reason:

> The gated delta rule's sequential form is not an alternative implementation of
> vLLM's prefill. It is the recurrence vLLM's prefill is a reassociation of, and
> it is measurably nearer to it: `1.15e-08` from the exact answer against
> `2.29e-04`. It is retained for two jobs no mirror can do. It is the reference
> an implementation error in the chunked arm is caught by, because a chunked arm
> cannot check itself against another chunked arm. And it is the only gated delta
> rule upstream itself will run on f32 inputs (`chunk.py:213-215`), so on that
> dtype it IS the mirror. It is off by default wherever upstream's own kernel
> would run, and it is not an ABI surface.

It is one exception, not four: the same predicate, the same kernel family, the
same reason on every backend. It is **not** licence for the arms to drift — the
sequential arm must stay bit-identical across CPU, ROCm and Vulkan, which is
today's cross-device gate and stays as-is under the flag.

## Blast radius, enumerated

### The control token sequence

`11751 13 15767 411 2029 11 1092 369` is the CPU arm's output for
`The capital of France is` on the released `Qwen3.8-Flash-Next-GGUF` UD-IQ1_S.
**Flipping the CPU default to chunked changes it.** It appears in 11 tracked
files on `origin/main` in that exact spelling — twelve once this spec lands,
which is why the count below is stated against the base and re-derivable with
`git grep -l "11751 13 15767 411 2029 11 1092 369"`:

| file | line(s) | role |
|---|---|---|
| `src/vllm/model_executor/models/qwen4_exp_registry.cpp` | 212 | product code (comment) |
| `docs/USAGE.md` | 670 | doc, checkpoint registry |
| `docs/bench-evidence/qwen4exp-cuda-decode-identifiers-20260902.md` | 27 | evidence |
| `docs/bench-evidence/qwen4exp-cuda-prefill-divergence-20260902.md` | 60 | evidence |
| `docs/bench-evidence/qwen4exp-moe-selection-20260902.md` | 229 | evidence |
| `docs/bench-evidence/qwen4exp-moe-selection-20260902/results.txt` | 17, 30, 42 | evidence |
| `.agents/specs/debtfix-glue-rank-bound-and-repack-device.md` | 123, 346 | spec |
| `.agents/specs/qwen4-exp-cuda-decode-ngram-history-sync.md` | 19, 198 | spec |
| `.agents/specs/qwen4-exp-cuda-rmsnorm-weight-dtype.md` | 221 | spec |
| `.agents/specs/qwen4-exp-flash-next.md` | 8406, 8461, 8535, 8689, 9160 | spec |
| `.agents/specs/qwen4-exp-qsa-q-bf16.md` | 123 | spec |

**Three more carry it in a different spelling and a `git grep` of the
space-separated form misses them.** `tests/vllm/models/test_qwen4_exp_layer_loop.cpp:2595-2596`
wraps it across two comment lines; `.agents/specs/qwen4-exp-flash-next.md:4376`
also spells it comma-separated; and `docs/FEATURES.md:128` plus
`docs/bench-evidence/qwen4exp-released-checkpoint-tokens-20260831.md:12,123`
carry only the decoded text `" Paris. Given this fact, what is"`. **Fourteen
tracked files, not eleven.** An implementer who greps one spelling will leave
three behind.

`.agents/specs/qwen4-exp-qsa-q-bf16.md:123` states the sequence "must not move".
That line is a gate on a *different* row's change and this row invalidates its
premise. Reconcile it in the same flow; do not silently break it.

An evidence file records what was measured on the tree it names. **Do not rewrite
history.** The evidence files and the specs' historical passages keep their
numbers; what changes is any line that presents the sequence as the *current*
expectation.

### `docs/USAGE.md:670` — CORRECTED IN THIS FLOW

It read: **"`--device cpu` is the arm to use when the exact ids matter."**

On the mirror criterion that is backwards, and it was the single canonical
statement of the claim (no other doc makes it). This spec's own commit replaces
it with a presently-true form — the CPU arm is the more *accurate* one and the
less *faithful* one. Verbatim, as landed in the same commit as this spec, in a
fenced block because its two links are relative to `docs/`, not to this file:

```text
**`--device cuda` is the arm that MIRRORS vLLM; `--device cpu` is the arm that
is more accurate.** Those are different things and the CPU arm is not the
authority: it runs an exact sequential recurrence and lands `1.15e-08` from the
exact answer, where vLLM's own chunked kernel lands `2.29e-04`
([decomposition](../../docs/bench-evidence/gdn-chunked-decomposition-20260902.md),
[#2612](https://github.com/mudler/vllm.cpp/issues/2612)), so the CPU ids are the
ids of an answer vLLM does not compute. Reach for `--device cpu` when you want
the ids this table records, not when you want the ids vLLM would emit.
```

**Sequencing.** That correction is true of the tree as it stands and does not
depend on the default flip, which is why it lands here rather than waiting. The
*further* rewrite — dropping the CPU/CUDA distinction once both arms are chunked,
and re-deriving the ids the cell records — lands with the implementation, and the
row is not `DONE` until the cell stops promising an arm that no longer exists.

### What goes red — and the answer is ALMOST NOTHING, which is the cost

An earlier draft of this spec listed five gates as "red by construction". **That
list does not reproduce, and the real cost is the opposite one.** Every test in
it is f32-only, and D0's predicate is
`chunked = (q.dtype != f32) && VT_GDN_CHUNKED != 0`, so under this spec's own
design none of them ever takes the chunked arm:

| predicted red | why it stays GREEN |
|---|---|
| `tests/vt/test_ops_gdn.cpp:111` | the hand-computed 3-token table builds every tensor through `T2`/`T3`/`T4` (`:53-60`), which hardwire `DType::kF32` |
| `tests/vt/test_ops_kda_recurrence.cpp:144` | every tensor in the case is `kF32` (`:169-191`) |
| `tests/vt/test_cpu_threadpool.cpp:459` | every `GdnPrefill` tensor in the corpus is `kF32` (`:325-378`) |
| `tests/parity/test_op_parity.cpp:590-600` | D0 IS the fix — these three goldens are f32 dumps of upstream's sequential kernel and must stay green, which is G2 |
| `tests/vt/test_ops_gdn.cpp:3466` | it runs `kCudaCombos[0]` only (`:795-799`), which is f32 in / f32 out |

**So the cost this spec has to state is a coverage gap, not a red list.**
Essentially the whole CPU unit-test corpus for `vt::GdnPrefill` is f32. Under D0
the new chunked arm therefore lands reached by almost nothing that exists today:
the single bf16 golden `gdn_prefill_bf16_realdims` and whatever T1 and T5 below
add. A change that lands green because its tests cannot see it is the failure
this repository names in `.agents/reachability.md`, and a suite that stays green
across the flip is evidence of nothing.

**Closing the gap is part of the implementer's job, not a follow-up.** Each site
in the table above gets a bf16 arm beside its f32 one, computing the same case at
the model dtype so that the chunked kernel is the thing under test:

| site | owed |
|---|---|
| `tests/vt/test_ops_gdn.cpp:111` | a bf16 companion case. The hand-computed f64 table is NOT the expectation for it — the chunked arm is a different answer by design — so the bf16 case asserts against the golden-derived bar of T1's family, or against the sequential arm at the recorded chunk-vs-sequential distance, and says which in a comment |
| `tests/vt/test_ops_kda_recurrence.cpp:144` | a bf16 arm, under whichever resolution T7 takes |
| `tests/vt/test_cpu_threadpool.cpp:459` | a bf16 `GdnPrefill` entry in the corpus at `:348`/`:375`, because thread byte-identity is only proved on the arm that actually runs |
| `tests/vt/test_backend_cross_device.cpp:1676` | see the trap below: this gate is all-f32 and is NOT where a chunked-arm comparison belongs |

**`tests/vt/test_ops_gdn.cpp:3452` fails for a DIFFERENT reason, in the opposite
direction, and it names a live hazard.** `RunGdnCudaCase` executes a bare
`setenv("VT_GDN_CHUNKED", "0", 1)` at `:1798`. It is process-wide, it is never
unset, and there is no scoped-restore wrapper on it. Two consequences:

1. Under D3's shared flag it pins **both** legs sequential, not just the CUDA
   one, so the CPU-vs-CUDA pair does not de-pin — it agrees for a reason that
   has nothing to do with either arm being right. This is the inverse of what
   an earlier draft claimed.
2. **The setting leaks forward through the whole binary.** Once `:1798` (or the
   toggle loop at `:2212`, or any of the nine `setenv(..., "1")` calls at
   `:3800-4077`) has fired, every later CPU `GdnPrefill` in the same doctest
   process inherits it. Today that is invisible because CPU ignores the flag.
   The moment D3 makes CPU read it, test-case ordering silently selects the CPU
   algorithm.

**Decision: the port replaces every bare `setenv("VT_GDN_CHUNKED", ...)` in
`tests/vt/test_ops_gdn.cpp` with the scoped setter the tree already has**,
`ScopedEnv` (used at `tests/parity/test_op_parity.cpp:594-595`), so the value is
restored at scope exit. This is not a cleanup: without it a chunked-CPU test and
a flag-pinning CUDA test in one binary interact through the environment, and
which one wins depends on doctest's case order. Do it in the same change that
makes CPU read the flag.

**The cross-device trap is real, and it does NOT belong on
`test_backend_cross_device.cpp:1676`.** That case is all-f32 — its own comment
at `:1677` says "§7/§8. All f32." — so under D0 no device takes the chunked arm
there and it compares two sequential arms before and after this row. Its
mechanics are otherwise exactly as this spec described them (CPU reference at
`:1699-1711`, `RegisteredDevices()` loop at `:1714`, `kNmseTol = 5e-4` at `:58`),
and it stays a gate; it is simply not the gate the trap applies to.

**The trap attaches to `tests/vt/test_vulkan_backend.cpp:2723-2747` instead.**
That is the bf16 arm of the Vulkan GDN prefill case, the one cross-device GDN
prefill comparison in the tree that runs at the model dtype: it builds
`kBF16` q/k/v/out on both devices (`:2723-2730`), runs the CPU reference at
`:2732` and Vulkan at `:2733`, and checks `nmse16 <= kGdnNmseTol` at `:2747`
with `kGdnNmseTol = 5e-4` (`:1961`). **Once BOTH arms are chunked they will
agree with each other far more tightly than either agrees with today's
baseline, and a green there is not evidence that either is right.** During the
transition — CPU chunked, Vulkan still sequential — the same gate compares two
different algorithms across a recorded `2.44e-04` max-abs gap. Both states are
green-for-the-wrong-reason in opposite directions.

**Fails outright, for a reason other than the chunked arm.**

| site | assertion | why it breaks |
|---|---|---|
| `tests/vt/test_ops_gdn.cpp:3452` | CUDA-vs-CPU over all three `kCudaCombos`, including bf16 | not the algorithm: the `setenv` at `:1798` pins both legs once CPU reads the flag. Fixed by the `ScopedEnv` decision above, not by touching tolerances |

**At risk, and each needs a measurement rather than a guess.**

| gate | bar | note |
|---|---|---|
| `tests/vt/test_vulkan_backend.cpp:2723-2747` (bf16 arm) | `nmse16 <= kGdnNmseTol`, `kGdnNmseTol = 5e-4` (`:1961`) | **the trap gate**, above. The f32 arm of the same case (`:2601-2702`) is unaffected by D0; its `:2701-2702` `memcmp` that an EMPTY sequence's state block is untouched must still hold on the chunked arm |
| `tests/vt/test_backend_cross_device.cpp:1676` | `Nmse <= kNmseTol`, `kNmseTol = 5e-4` (`:58`) | all-f32, so unmoved by D0. It remains the ROCm/Vulkan/Tenstorrent/CUDA gate at once (`RegisteredDevices()` at `:1714` against the CPU reference at `:1699-1711`), and ROCm's kernel header names it as its contract (`src/vt/rocm/rocm_gdn_scan.hip:14-16`). It gates the SEQUENTIAL arm after this row, which is a thing worth gating and is not a mirror check |
| `tests/vt/test_vulkan_backend.cpp:2890` | `memcmp == 0` on the wide-`Dk` decline path (`:2967`) | Vulkan declines to the CPU provider, so this compares the new CPU kernel to itself. It stays green — and it therefore proves nothing about the change, which is the trap in its purest form |
| `tests/vt/test_tenstorrent_backend.cpp:1965` | `tol_o` 0.05/0.08, `tol_s` 0.05 (`:2068-2076`) | its title says "matches the CPU f32 oracle" and the case is f32, so D0 leaves it comparing an already-chunked Tenstorrent arm against a still-sequential CPU f32 reference. Its recorded state margin is `1.44e-2` against 0.05 — 3.10x headroom (`.agents/specs/tenstorrent-gdn.md:263-264`). Unmoved by this row; re-based only if a later row gives it a bf16 arm |

**Where the re-baseline actually comes from.** Not from any of the tables above.
`tests/parity/goldens/gdn_prefill_bf16_realdims/` is a dump of the real Triton
chunk kernel and is the only artifact in this tree that can say whether a
chunked port is correct. It is currently satisfied by the sequential C++ at a
tolerance loosened to `5e-3` in M0.7 *because of* the `2.29e-04` gap its own
manifest note records. **A chunked CPU arm must pass it far tighter, and that
tightening is this row's primary gate.**

### Harnesses and evidence whose premise moves

- `scripts/tier-hidden-delta.py` — the ROCm-tier-vs-CPU-tier activation bisector
  (#2590) carries no numeric bar by design, but its whole premise is that the
  CPU tier is the reference. Every recorded profile is re-based.
- `docs/bench-evidence/qwen4exp-cuda-prefill-divergence-20260902.md`,
  `docs/bench-evidence/qwen4exp-moe-selection-20260902.md` (`:77`, `:119`,
  `:228`) and `.agents/specs/qwen4-exp-flash-next.md:4079,8671-8831` all rest on
  "the CPU arm runs an exact sequential recurrence". They keep their numbers —
  an evidence file records what was measured on the tree it names — but any
  passage presenting that as the CURRENT arrangement is reconciled.

### The production path this lands on

`vt::GdnPrefill` on `kCPU` is reached on the default, unflagged path of two
model trunks, at every linear-attention layer of every prefill step:
`runner.cpp:2814` -> `model_registry.cpp:492` -> (`qwen3_5_moe.cpp:109,210` or
`qwen4_exp_registry.cpp:295,956` -> `qwen4_exp_forward.cpp:589` ->
`qwen3_5.cpp:8116`) -> `qwen3_5.cpp:5103 GdnBlockPaged` -> `:5484`/`:5493` ->
`vt/ops.cpp:2190,2196` -> `cpu_ops.cpp:4036` -> `:1862`. The only branches are
`layer.is_linear_attention` and `np > 0`. That is the call site G6 deletes.

**Two dtypes on that path, and neither is what an op test uses.** `out` is
**bf16** by default — `GdnOutDType()` (`qwen3_5.cpp:193-197`) reads
`VT_GDN_OUT_BF16`, unset means on, and there is no device term, so CPU gets bf16
exactly as CUDA does. `state` is **f32** unconditionally: the persistent cache is
gathered into an f32 working buffer before the call (`qwen3_5.cpp:5451-5455`).
The chunked CPU arm must therefore accept a bf16 `out` store and an f32 state,
and any CPU-vs-CPU A/B must fix `VT_GDN_OUT_BF16` rather than inherit it.

## Tests — red first, and the honest note about how little goes red

**Read `## What goes red` first.** Only T1 has a genuine red-before on
`origin/main` today, because the existing CPU corpus is f32 and D0 routes f32 to
the sequential arm. T2, T3 and T4 are red in the trivial sense that the thing
they assert does not exist yet. The rest are coverage the chunked arm does not
have and must be given, and their red-before is captured by writing the bf16
case FIRST and watching it fail against the tight bar on the sequential arm.

An implementer who reports "everything was already green" has confirmed the
coverage gap, not passed the gate.

1. **T1, the tolerance the sequential arm cannot meet.** `test_op_parity.cpp`
   runs `gdn_prefill_bf16_realdims` at `atol = rtol = 5e-3`, loosened in M0.7
   precisely to admit the sequential arm. Add a second, *tight* case over the
   same golden that the chunked CPU arm must pass. The bar is derived from the
   evidence, not chosen: the bf16-placement replica lands `6.1035e-05` from the
   real kernel and 62 of 8192 bf16 output elements differ. **Bar:
   `max|d| <= 1.5e-04` on `out` and `<= 1.5e-03` on `state`**, which the
   f32-intermediate arm (`2.4414e-04` / `2.2486e-03`) and today's sequential arm
   (`2.2865e-04` / `2.2487e-03`) both fail and a bf16-faithful port passes with
   margin. RED-BEFORE: fails on `main` at `2.2865e-04`.

   **THE BAR IS SINGLE-CHUNK-DERIVED, and G1 says so.** The golden is two
   sequences of 20 and 12 tokens at `BT = 64`, so `run_golden.py` processes
   `2 seqs x 2 heads x 1 chunk = 4` chunks and asserts that count. **No
   cross-chunk state carry is measured against a real kernel dump anywhere in
   this tree.** The multi-chunk regime exists only against the replica's own f64
   reference (`run_final.py`, `T` up to 1024), where the oracle is the author's
   implementation rather than a dumped kernel. Evidence §3 shows that the two
   sites which compound across chunks — `vdec` (`chunk_delta_h.py:274`) and the
   `wy_fast` stores — are precisely the state-path ones, and `vdec` does not
   reach `out` at all in a single-chunk run. So `1.5e-04` / `1.5e-03` is a bar
   with 2.46x / 2.96x headroom over a residual that a multi-chunk workload has
   never been asked to produce, and it may be optimistic there. It is the bar
   this row gates on, because it is the only bar an oracle dump supports.

   **What would establish a multi-chunk bar**, and it is owed rather than
   assumed: a second golden dumped from `fla/ops/chunk.py::chunk_gated_delta_rule`
   at `T >= 129` on the same real dims, giving at least three chunks including a
   partial tail, with a sequence carrying an initial state. Until that dump
   exists, a T1 failure at `T > 64` on a port that passes G3 is the
   `## Owed` `6.1e-05` hypothesis, not automatically an implementation defect —
   and it is the trigger to dump the golden rather than to loosen the bar.

2. **T2, the dtype gate a token gate cannot see.** Assert the *memory format* of
   the chunked CPU scratch: `u`, `w`, `v_new`, the per-chunk `h` snapshot and
   `A^-1` are bf16 buffers; `A`, the running state and `final_state` are f32.
   AGENTS.md: a token gate cannot detect a dtype that is too wide, and evidence
   §4 is the proof that the wide version passes every value gate the sequential
   arm passes. RED-BEFORE: no such buffers exist.

3. **T3, the dtype predicate of D0 routes.** An f32 request must take the
   sequential arm and a bf16 request the chunked one, on the same binary, with
   `VT_GDN_CHUNKED` unset. Assert it by observing the two arms' outputs differ
   for bf16 and are bit-identical to today's for f32. RED-BEFORE: no predicate.

4. **T4, the flag routes, in both directions.** `VT_GDN_CHUNKED=0` and `=1` in
   one process over the same bf16 inputs must produce *different* outputs on
   every backend that has both arms, and the `=0` output must be bit-identical
   to today's sequential output. Assert both: a flag that is read and ignored,
   and a flag whose two arms coincide, each reads as a pass otherwise. Mirror
   `tests/vt/test_ops_gdn.cpp:1982-1983`, which already does this for CUDA.
   RED-BEFORE: on CPU both settings give one answer.

5. **T5, the chunk boundary.** `T = 5` (one partial chunk — the real prompt
   length PREFILLDIV measured), `T = 64` (exactly one chunk), `T = 65`,
   `T = 129`, and a varlen batch mixing them with an EMPTY sequence. Evidence §3
   shows the state-path rounding sites do not reach `out` at all in a
   single-chunk run, so a suite that only exercises `T <= 64` cannot see half the
   kernel. The empty-sequence case is not optional:
   `tests/vt/test_vulkan_backend.cpp:2701-2702` already `memcmp`-asserts that
   block untouched.

6. **T6, byte-identity across threads survives.** The corpus at
   `tests/vt/test_cpu_threadpool.cpp:348`/`:375` is entirely `kF32`
   (`:325-378`), so it will never reach the chunked arm on its own. **Add a
   bf16 `GdnPrefill` entry** and check byte-identity at `n_threads` 1 / 3 / 20
   on it. Today's kernel is byte-identical by construction
   (`cpu_ops.cpp:1878-1880`); a chunked arm must be too, which constrains how
   chunks may be parallelised. RED-BEFORE: run the new bf16 entry under the
   existing harness before the parallelisation is settled. Reporting the
   existing f32 corpus green is not this test.

7. **T7, the KDA reduction.** `tests/vt/test_ops_kda_recurrence.cpp:144` asserts
   `vt::KdaGatedDeltaRule` reduces to `vt::GdnPrefill` **bit for bit** on CPU
   (`out_diff == 0`, `st_diff == 0`). Its tensors are all `kF32` (`:169-191`),
   so **D0 leaves it green** — the claim silently narrows from "the two agree"
   to "the two agree at f32", and nothing in the tree records that narrowing.
   That is worse than a red, because it is a true assertion standing in for one
   that stopped being true. **Decide it in the spec, not in the test**: either
   KDA moves in lockstep, or the reduction is restated as holding on the
   sequential arm. Restating it is the smaller change and is what this spec
   recommends, because the KDA row's claim is about the *recurrence*, not about
   which evaluation order our GDN default happens to take. Under that
   resolution the test states the narrowing in a comment carrying this row's ID,
   and **adds a bf16 arm that asserts the two DIFFER** — a reduction claim that
   is silent about the dtype where it fails is the shape of gate this spec keeps
   refusing. If instead KDA moves in lockstep, the bf16 arm asserts `== 0` there
   too. Do not weaken `== 0` to a tolerance (`## Stop conditions` item 6).

8. **T8, upstream's own tests.** Port the FLA chunked-path cases with their
   parameters, tolerances and the pin anchor, per AGENTS.md. Constraint, and it
   must be recorded rather than papered over: `chunk.py:213-215` means upstream
   has **no f32 chunked case to port**, so whatever f32 chunked coverage exists
   is ours. Record it as an adaptation, never as a port.

9. **T9, the f32-only corpus gets bf16 arms, and the flag setters get scoped.**
   The four sites tabulated under `## What goes red` each gain a bf16 companion,
   and every bare `setenv("VT_GDN_CHUNKED", ...)` in `tests/vt/test_ops_gdn.cpp`
   (`:1798`, `:2212`, and `:3800`-`:4077`) becomes a scoped `ScopedEnv` so the
   value does not leak into a later case in the same process. RED-BEFORE for the
   scoping half is mechanical and worth capturing anyway: set the flag in one
   case, assert in a LATER case that `getenv("VT_GDN_CHUNKED")` is unset, and
   watch it fail on `main`. This item is not optional polish. Without it the CPU
   algorithm a test runs depends on doctest's case order.

**Two doctest traps this suite must avoid.** `assertions: 0 ... SUCCESS!` at
rc 0 is a skip wearing a pass — assert a counted property in every case. And
`-tc` splits a filter on commas, so no case name added here may contain one
(#2605).

## Gates

The operator reruns each of these itself; an implementer or reviewer report is
an input, never a result.

| gate | what | bar |
|---|---|---|
| G1 tight golden | T1, `gdn_prefill_bf16_realdims`, CPU arm | `max\|d\| <= 1.5e-04` out, `<= 1.5e-03` state. **SINGLE-CHUNK-DERIVED**: the golden is 20 and 12 tokens at `BT = 64`, so the residual this bar has 2.46x/2.96x headroom over was measured at one chunk per sequence, and no cross-chunk state carry is gated against a real kernel dump anywhere. See T1 for what would establish a multi-chunk bar |
| G2 f32 goldens UNMOVED | `gdn_prefill_f32_small`, `_noinit`, `_realdims` | still green at their committed `atol = rtol = 1e-05`, with the tolerances **byte-unchanged**. This is the gate that catches a mis-read of D0. |
| G3 dtype format | T2 | every buffer has the dtype D1 names |
| G4 routing | T3, T4 | dtype predicate and flag both reach both arms; `=0` bit-identical to today |
| G5 breadth | T5, T6, T8 | green on every listed `T`, on varlen with an empty sequence, and at 1/3/20 threads — **on bf16 entries**, since the f32 corpus does not reach the arm |
| G5b coverage | T9 | every site tabulated under `## What goes red` has a bf16 arm, and no `setenv("VT_GDN_CHUNKED", ...)` in `tests/vt/` escapes its scope. Mutate for it: delete one bf16 arm and confirm the suite still passes — if it does not go quieter, the arm was not the thing under test |
| G6 KDA | T7 | the reduction claim is true as written, and the test states which arm it holds on |
| G7 cross-device re-baseline | `tests/vt/test_vulkan_backend.cpp:2723-2747` (the bf16 arm — the ONLY cross-device GDN prefill gate D0 moves); `tests/vt/test_backend_cross_device.cpp:1676` and `tests/vt/test_tenstorrent_backend.cpp:1965` are all-f32 and are confirmed UNMOVED rather than re-derived | the bf16 bar re-derived **against the golden**, never against the CPU arm, and restated with the arm and tree it was measured on. For the two f32 gates the required evidence is the opposite: show they did not move, and say it is because D0 routes f32 sequential on every arm |
| G8 reachability | delete `qwen3_5.cpp:5484`/`:5493` in a scratch copy, rerun G1-G6 | at least one gate goes red; restore the tree byte-for-byte |
| G9 full | `scripts/agent-preflight.sh` | grep for `gate(s) failed` and `NOT a green`; never trust the exit code |

**G7 is the gate most likely to be reported green while meaning nothing**, for
the reason stated under `## What goes red`. State for each backend which arm
produced the baseline, at which dtype, and on what tree. An evidence table that
does not name its tree is not evidence, and one that does not name its dtype
cannot say whether the chunked arm ran at all.

## Risks

**R1. The implementer ships f32 intermediates.** The likeliest failure, because
f32 is the obvious CPU choice and every existing value gate passes with it. It
is caught by G1 and G3 and by nothing else in the tree. This is a
`## Stop conditions` item, not merely a risk.

**R2. The implementer reads "chunked by default" as "chunked always."** The
second likeliest, and it fails three committed goldens by one to two orders of
magnitude. D0 and G2 exist for it.

**R3. bf16 on CPU costs a rounding pass at nine sites.** Our CPU arm has no bf16
arithmetic; the sites are round-to-bf16-then-widen, not bf16 math. That is what
upstream does too (Triton accumulates f32 and stores bf16), so it is a mirror.
No speed claim is made and none is owed; if the cost is material it is a
separate row's problem.

**R4. Chunk parallelism breaks thread byte-identity.** Today's kernel is
byte-identical across thread counts because `(sequence, head)` work is disjoint
with no shared reduction. The cross-chunk state recurrence is sequential and must
stay so; the intra-chunk work may be parallelised only in a fixed order. T6/G5.

**R5. `A^-1` conditioning.** `(I+A)` is unit lower triangular with `|A| < 0.11`
and `cond = 1.13` on the golden; a dense inverse is safe there and not in
general. Assert unit-diagonal forward substitution rather than a general solver,
and add a case with `beta -> 1` and near-parallel `k`.

**R5 MEASURED, and one implication in it is wrong.** The port takes unit-diagonal
forward substitution as asked. But the un-normalised-`k` blow-up is NOT the
inverse's: over 12 seeds at `T=70`, `Dk=Dv=128` with un-normalised `q`/`k`, the
SEQUENTIAL recurrence reaches `max|out|` of `1e23` to `1e31` and the chunked arm
is SMALLER in every one of them. With `|k|^2 ~ Dk` the update
`S <- S(I - beta k k^T) + beta v k^T` has a per-token factor of order
`(1 - beta|k|^2)`, so both arms diverge geometrically and L2 normalisation is
what makes the recurrence a contraction. The precondition R5 names is real; the
mechanism it attributes it to is not, and an earlier draft of this row's test
comments carried that misattribution. The `beta -> 1` / near-parallel-`k` case
R5 asks for is still OWED.

**R6. The re-baselined cross-device gates go green for the wrong reason.**
Covered by the G7 note.

**R7. The row lands and the tree still says the old thing.** Fourteen files carry
the control sequence, one doc cell states the inverted claim, four records
describe an arrangement that stopped being true when M2.3 landed CUDA-only, and
three evidence documents rest on "the CPU arm runs an exact sequential
recurrence". Every one is listed above or under `## Records owed`; none is
discovered later.

**R9. The chunked arm lands with almost no coverage and every gate is green.**
The likeliest OUTCOME, as distinct from the likeliest defect. The CPU corpus is
f32; D0 routes f32 sequential; so the new arm is reached by one golden and the
cases T1/T5/T9 add. A suite that passes across the flip without those cases has
measured the arm that did not change. T9 and G5b exist for it, and the mutation
G5b names is the only thing that distinguishes coverage from its appearance.

**R10. `VT_GDN_CHUNKED` leaks between test cases and picks the algorithm.**
`tests/vt/test_ops_gdn.cpp` sets it with a bare, never-unset `setenv` in eleven
places. Harmless today, because only CUDA reads it; a silent, order-dependent
algorithm selector the moment D3 lands. T9 converts every one to `ScopedEnv`.

**R8. Someone reads a token count as the result.** Named in `## Scope` and
restated here, because PREFILLDIV already measured that the more accurate arm
agrees on *fewer* ids.

## Records owed

Each rides in the pull request whose change makes it stale, per AGENTS.md.

| record | today | owed |
|---|---|---|
| `.agents/porting-inventory.md:124` | Tier `T0 🚧 ead59d6 (correctness-grade sequential; chunked perf kernel M2.3)` | M2.3 landed chunked on **CUDA only**. **Corrected in this flow** — the record is already stale on `main`. |
| `docs/bench-evidence/qwen4exp-cuda-prefill-divergence-20260902.md:110-111` | calls `5559679229` "a FORWARD REFERENCE beyond this row's pin"; its §4 dtype table omits two bf16 sites | **It IS the pin.** Both **corrected in this flow**. |
| `docs/USAGE.md:670` | "`--device cpu` is the arm to use when the exact ids matter" | Inverted on the mirror criterion. **First half corrected in this flow**; the full rewrite, dropping the CPU/CUDA distinction, lands with the implementation. |
| `.agents/porting-inventory.md` §9 | no entry | The tracked exception of D4, as a numbered list item in the section's prose format (it is NOT a table; and its numbering is already non-monotonic and must not be "fixed"). Lands with the implementation. |
| `.agents/specs/gdn-semantics.md` §7 | calls the chunked path the "chunked oracle" and the sequential kernel "what the M0.7 C++ implements directly" | The second half is what changes. Restate which arm runs which algorithm, and add D1's dtype placement — §7 carries none today. Lands with the implementation. |
| `docs/ENVIRONMENT.md` `## Rollback and bisect switches` | `VT_GDN_CHUNKED` absent; allowlisted as kernel-internal at `scripts/env-doc-allowlist.txt:71` | It becomes a cross-backend documented lever. Add the row; its sibling `VT_GDN_OUT_BF16` is already there (`:228`) for the same reason. Lands with the implementation. |
| `scripts/check-pr-size.py` `BENCH_EVIDENCE_RUN` | the extension list was `txt\|log\|gz\|sh\|cu`, so this wave's SIX `.py` probes under `docs/bench-evidence/gdn-chunked-decomposition-20260902/` were unclassified; `classify_path` fails closed, and the whole-tree sweep in `tests/scripts/test_check_pr_size.py` went red on the branch carrying them | **Corrected in this flow** ([#2629](https://github.com/mudler/vllm.cpp/issues/2629)), with a red-before mutation and two new cases. At least the seventh instance of the class — #856, #668, #989, #1448, then `8496d93dd` (#2316) and `6d1335568` (#2609) — and the second in two days; repaired by naming the surface, not by widening a rule |
| `.agents/kernel-matrix.md` + `scripts/check-agent-record.py:402` | no `KERNEL-GDN-CHUNKED-MIRROR` row; `KERNEL` count `58` | The row and the `58 -> 59` bump, owed by the change that moves this spec to `READY`. |
| `.agents/specs/tenstorrent-gdn.md:263-264` | tolerance table measured against a sequential CPU reference | Re-based by G7. |
| `.agents/specs/qwen4-exp-qsa-q-bf16.md:123` | "must not move" on the control sequence | Reconcile with the implementation. |
| `docs/FEATURES.md:128` | carries the control sequence in decoded form | Touch only if the flip changes a user-visible arm. |

## Owed

- [#2612](https://github.com/mudler/vllm.cpp/issues/2612) — the finding this row
  was created from. This spec is its owner; the pull request body says so.
- [#2845](https://github.com/mudler/vllm.cpp/issues/2845) — the implementation.
  **Closed by the CPU arm.** What remains after it is listed under `## Now`:
  ROCm and Vulkan chunked arms, G7's bf16 cross-device re-baseline, and a CUDA
  gate run for D0's changed f32 default. This spec owns each of them.
- [#2858](https://github.com/mudler/vllm.cpp/issues/2858) — re-deriving the
  control token sequence. **Done and closed by wave ARMTOKENS**; the result is
  under `## Now` and in
  [the evidence file](../../docs/bench-evidence/qwen4exp-gdn-chunked-token-ids-20260904.md).
- [#2861](https://github.com/mudler/vllm.cpp/issues/2861) — `main` not compiling
  with `-DVLLM_CPP_CUDA=ON`, this row's own regression. **Found and fixed in the
  ARMTOKENS flow and closed by its pull request.** Listed here because AGENTS.md
  asks an issue a change cites to name its owner, and this spec is that owner.
- **The MoE prefill residue is now the DOMINANT CPU-vs-CUDA divergence**, not
  merely a second source. ARMTOKENS measured `moe` at `2.289e-04` from an input
  differing by `4.324e-05`, against PREFILLDIV's `1.269e-04` from `2.1e-05`.
  Still named, still undiagnosed, still not this row — but it is what any future
  work on these ids has to attack, because the Gated DeltaNet term is spent.
- **GDN decode on the non-CUDA arms.** Out of scope above. vLLM does run decode
  sequentially, so the arms already mirror it, but nobody has measured that claim
  the way prefill has now been measured. No issue yet; file one before this row
  reaches `DONE`.
- **The f32 chunked arm has no upstream test**, because `chunk.py:213-215`
  refuses f32. Whatever coverage exists there is ours and unmirrored. Owed as a
  named coverage gap, not as coverage.
- **The `qwen4_exp` MoE prefill residue**, `7.269e-05` per layer with the GDN
  source removed (PREFILLDIV §3). A second, independent divergence source, named
  and undiagnosed. Not this row.
- [#2629](https://github.com/mudler/vllm.cpp/issues/2629) — `check-pr-size.py`
  had no class for a `.py` probe in a per-run evidence directory, so this
  branch's own six scripts were unclassified and the checker's whole-tree sweep
  was red. **Fixed in this flow and closed by this pull request**; it is listed
  here because AGENTS.md asks an issue a change cites to name its owner, and
  this spec is the thing that owns it. What the issue records beyond the fix is
  the RATE: at least seven repairs of one shape in that file, two of them one
  day apart, and the argument for why no durable fix was taken.

- **A MULTI-CHUNK oracle dump.** Every number in this spec, G1's bar included,
  is derived from a golden of 20 and 12 tokens at `BT = 64` — one chunk per
  sequence. The cross-chunk state carry is exercised only against the replica's
  own f64 reference. A second dump from
  `fla/ops/chunk.py::chunk_gated_delta_rule` at `T >= 129` on the same real
  dims, with a partial tail chunk and a sequence carrying an initial state,
  would turn T1's bar from single-chunk-derived into a bar. No issue yet; file
  one before this row reaches `DONE`. It needs a GPU lease and is why it is not
  in this change.

- **The `6.1e-05` the replica does not model.** The bound on how much of vLLM's
  output a bf16-faithful C++ port could still miss for reasons that are not dtype
  placement: Triton tile reduction order, the blocked `solve_tril` merge,
  `tl.dot` operand precision. If G1 fails at the tight bar for a port that passes
  G3, this is the first hypothesis and it needs its own row.

## Stop conditions

Stop and return `NEEDS_DECISION` rather than proceeding, if:

1. **The chunked arm is written with f32 intermediates.** Evidence §4: that arm
   is our current sequential output to within 2 of 8192 bf16 elements. It would
   land a mirror claim that is false and a gate that cannot detect it. This is
   the failure this spec exists to prevent.
2. **G1's tight bar is not met by a port that passes G3.** Do not loosen G1. The
   bar came from a measurement; moving it to admit the implementation is making a
   red gate green by widening its scope.
3. **The three f32 goldens are made green by editing their tolerances.** They are
   dumps of upstream's own sequential kernel at `1e-05`. If they are red, D0 was
   not implemented. Loosening them would delete the only evidence that the dtype
   predicate exists.
4. **A cross-device gate can only be made green by re-baselining it against the
   new CPU arm rather than against the golden.** That is a circular gate.
5. **The work needs a second flag.** D3 says one name. If the choice cannot be
   expressed with one predicate, the seam is wrong and the design needs
   revisiting, not a second env var.
6. **The KDA bit-identity reduction is made green by weakening its assertion from
   `== 0` to a tolerance.** T7 offers two honest resolutions and that is not one
   of them.
7. **A token-exactness claim appears anywhere in the change.** Scope says this
   row does not promise it.

## Now

Implementation issue:
[#2845](https://github.com/mudler/vllm.cpp/issues/2845).

**`ACTIVE` (wave GDNCPUPORT, 2026-09-03). The CPU chunked arm is written, the
flag is shared, the default is dtype-conditioned, and G1-G6 pass on CPU. The row
is NOT `DONE`: three gates need hardware this wave did not have.**

Landed:

- **The chunked CPU kernel**, `GdnChunkedHeadPrefill` in `src/vt/cpu/cpu_ops.cpp`,
  carrying upstream's bf16 intermediate placement site for site (D1). It lands
  `6.103516e-05` on `out` and `5.059987e-04` on `state` against
  `gdn_prefill_bf16_realdims`, reproducing the numpy replica's `chunk_up` arm to
  the printed precision -- so **stop condition 1 does not fire**, and G1 passes
  with 2.46x / 2.96x headroom. `A^-1` is materialised and rounded, per D2.
- **One shared predicate on every backend**, `vt::GdnUseChunkedPrefill(dtype)`
  over `vt::GdnChunkedPrefillEnabled()` (`src/vt/ops.cpp`). `cuda_gdn.cu`'s
  bespoke reader delegates to it, and the bespoke `e[0] != '0'` parse moved
  verbatim. One flag, not two (stop condition 5).
- **D0's dtype term on CUDA as well as CPU**, so an f32 CUDA `GdnPrefill` that
  ran chunked before this row runs the sequential scan after it. That is a
  behaviour change on a shipped backend.
- **T9's `ScopedEnv` conversion**: all eleven bare `setenv("VT_GDN_CHUNKED", ...)`
  in `tests/vt/test_ops_gdn.cpp` are gone, and a case registered last asserts the
  flag did not leak. Mutating one guard back to a bare `setenv` reds that case
  in a whole-suite run **and passes it under `-tc`** -- which is the order
  dependence R10 names, reproduced.
- **The bf16 coverage R9 says the arm would otherwise land without**: T1 (the
  tight golden bar, with the sequential arm's failure of it asserted in the same
  case), T3/T4 (the predicate and the flag both route), T5/T8 (upstream's own
  `PREFILL_SEQ_LENS`, `NUM_HEADS`, `CHUNK_HEAD_DIMS` and `atol=rtol=1e-2` ported
  from `tests/kernels/mamba/cpu/test_cpu_gdn_ops.py`, plus its two-call split and
  our empty-sequence case), T6 (a bf16 entry in the thread byte-identity corpus),
  T7 (the KDA reduction restated onto the sequential arm with a bf16 arm
  asserting the two differ).

**What changed in the reference, and it is new since this spec was written.**
vLLM now ships a **chunked CPU kernel in C++**, `chunk_gated_delta_rule_cpu`
(`csrc/cpu/sgl-kernels/fla.cpp:2178`), reached from
`vllm/model_executor/layers/mamba/ops/cpu/gdn_attention.py:247,629,705`. It
confirms D0 (it type-checks bf16 only, `:2205-2207`) and confirms D1's dtype
FAMILY on a CPU target. It differs from Triton in three secondary sites, and
**following it instead of Triton would fail G1**: pre-scaling `q` into bf16
alone moves `out` from `6.103516e-05` to `2.441406e-04` against a `1.5e-04` bar.
The port follows Triton and records why; the measurement is in
`.agents/specs/gdn-semantics.md` §7.

**G8 FAILS, AND IT IS THIS ROW'S OWN REACHABILITY GATE.** With all four
production `vt::GdnPrefill` call sites in `qwen3_5.cpp` disabled in a scratch
copy (4 markers counted, build rc read separately from test rc), G1-G6 come back
BYTE-IDENTICAL: 68/2285, 14/161, 9/19606, 4/9 (cases/assertions for
`test_ops_gdn`, `test_op_parity`, `test_cpu_threadpool`,
`test_ops_kda_recurrence`, re-counted at this head). The mutation bit — the positive
control is that `test_qwen35_paged_forward` and `test_qwen3_5_gdn_spec_routing`
flip green to red — so the op IS reached from `ModelRegistry::Forward`. What no
gate in this row measures is the REACH: every test here enters through
`vt::GdnPrefill` directly. Per `.agents/reachability.md` that measures a class,
not a capability, and it is recorded here rather than left for a reader to
rediscover. Two model-level tests now do discriminate the arms after the
reconciliations below, which is a partial repair and not the gate G8 asks for.

**THREE production-path tests went red and are reconciled here. The third was
found by the full `ctest` sweep, not by the review** — which is the argument for
running the whole suite rather than the row's own targets, and the reason
`## Gates` G9 says what it says. All three are ONE failure mode wearing three
faces: a scheduler-split prefill is bit-identical under the sequential
recurrence and is not under vLLM's chunked one.

- `tests/vllm/v1/test_llm_engine.cpp`'s
  "chunked prefill accumulates the identical prompt logprobs" runs a
  `linear_attention` config at `max_num_batched_tokens=1`, i.e. a chunk boundary
  between EVERY pair of prompt tokens — the maximum discontinuity the algorithm
  admits — and asserted `Approx(...).epsilon(1e-5)`. Sequential arm: exactly 0.
  Chunked arm: max|d| `3.28e-03`, max relative `1.2e-03`. Its subject is the
  ACCUMULATION MACHINERY (`gpu_model_runner.py:5646-5706`), which is
  arm-independent, so the row/position/token-id assertions stay exact on both
  arms and only the logprob VALUE splits: `== ` on the sequential arm, `5e-3`
  relative on the chunked one.
**The two the review found:**

- `test_qwen27_paged_forward.cpp`'s state-continuity case read `0.00277987` and
  `0.00376107` against a `< 1e-4` bar. **The port is right and the bar was
  wrong**, and the bar was wrong in a way that only a chunked arm can expose:
  one-shot == split is EXACT for the sequential recurrence and is not a property
  of vLLM's algorithm at all. Each extra chunk boundary sends the interactions
  across it through the bf16 state snapshot (`chunk_delta_h.py:178,352`) instead
  of the intra-chunk f32 path. The committed replica reproduces it independently
  of our C++ at this case's own T=6 split {3,3} — sequential `0.0`, chunked
  upstream-bf16 `1.694679e-03`, chunked with f32 intermediates `2.980232e-08` —
  so the discontinuity is the bf16 PLACEMENT, not the reassociation, and cannot
  be engineered away without giving up the mirror. **The driver is committed**
  (`docs/bench-evidence/gdn-chunked-decomposition-20260902/run_split.py`), so
  every number these three test comments quote as a derivation is reproducible
  without writing one. Upstream says the same thing
  in its own words: `test_chunk_gated_delta_rule_cpu_two_call_split` gates this
  exact property at `1e-3` state / `2e-2` output with the comment "State must be
  near-exact; output allows a looser bound for the bf16 round-trip".
  **Reconciled, not loosened**: the exactness claim moves onto the sequential arm
  and is TIGHTENED there from `< 1e-4` to `== 0`, and the chunked arm gets a
  `1e-2` bar with an IN-TEST POSITIVE CONTROL that drops the carried state and
  asserts it exceeds that bar. **State the cost honestly: on the sequential arm
  the ratio between a correct and a dropped state is infinite; on the chunked
  arm it is finite — measured 9.75x at tail=3 and 10.37x at tail=2 (0.00277987
  against 0.0270987, and 0.00376107 against 0.0389975).** Mirroring vLLM buys that
  discriminating power down and no bar in that window buys it back.

- `test_qwen4_exp_layer_loop.cpp`'s `CHECK(moved > 0.0)` read exactly `0`.
  **The gate is a mute switch and always was** ([#2851](https://github.com/mudler/vllm.cpp/issues/2851)).
  `logits_indices` is `{T-1}` and both prompts are EOS-terminated, so the one
  compared row is the SAME final token's logits. Varying only `ids2`, the
  sequential arm — byte-identical to `origin/main` on CPU — reads exactly `0`
  for two of six valid prompts and the SAME `0.0546875` for four others, which
  is a fixed `7/128` artifact rather than prompt response (`max|logit|` is
  `95090.7`). The forward is not prompt-blind on either arm: with `ids2 = t + 2`
  the chunked arm moves `31.8438`. Repaired by comparing ALL `T` rows, of which
  `0..T-2` carry a different token, plus a same-prompt rerun control that makes
  `> 0.0` mean something. The repaired form passes for all six prompts on both
  arms where the old one failed two of six on `main`.

**Three kernel defects the review found, fixed here.** The predicate was
`dtype != f32`, so an f16 request took the chunked arm and was silently
bf16-rounded at all nine sites; it is now `dtype == bf16`, which is what both
upstream implementations accept. The output was rounded to the INPUT dtype,
which silently made `VT_GDN_OUT_BF16=0` a no-op on CPU; the store now rounds to
the DESTINATION dtype (upstream's own `o = q.options()` rule), the golden runner
hands it a bf16 buffer as upstream does, and T2 gates both halves. And
`chunk.py:212`'s `q.dtype == k.dtype == v.dtype` is now carried.

Not done, and each is a gate rather than a nicety:

- **ROCm and Vulkan have no chunked arm.** The shared predicate reaches them;
  the kernel does not. They still run the sequential recurrence on every dtype.
- **G7 is NOT established.** `tests/vt/test_backend_cross_device.cpp:1676` is
  confirmed UNMOVED (all-f32, so D0 routes it sequential on every arm, and it is
  green). The trap gate, `tests/vt/test_vulkan_backend.cpp:2723-2747`, is the
  bf16 arm and needs a Vulkan device; this wave had none, and `test_vulkan_*` is
  not even a build target here. Until it is measured, the CPU arm is chunked and
  the Vulkan arm is sequential, which is the transition state the spec names as
  green-for-the-wrong-reason in one direction.
- **No CUDA gate was run.** D0 changed CUDA's f32 default and nothing on a GPU
  has executed since. `tests/vt/test_ops_gdn.cpp:3452`'s CUDA-vs-CPU case is the
  one the `ScopedEnv` conversion was supposed to unbreak, and it has not run.
- **D0 DEFANGED SIX CUDA A/B CALLS AND THE REPAIR IS UNRUN.**
  `RunGdnChunkedVsSequentialOnQueue` had no must-differ assertion, so the six
  f32 calls in the chunked-vs-sequential ladder became self-comparisons the
  moment both toggles routed f32 sequential — passing trivially and retiring
  three of four rungs. The same guard this row applied to the CPU A/B (T4) is
  now inside the helper, armed by `vt::GdnUseChunkedPrefill` itself rather than
  a hardcoded dtype list, and each f32 rung has gained a bf16 twin so the ladder
  still isolates chunk-count, varlen and GQA. **None of the twins has run on a
  GPU**, so their `3e-2` tolerance is inherited from the one bf16 rung that
  already existed at these shapes and is not measured.
- ~~**The control token sequence has not been re-derived.**~~ **DONE, wave
  ARMTOKENS, 2026-09-04**
  ([evidence](../../docs/bench-evidence/qwen4exp-gdn-chunked-token-ids-20260904.md),
  [#2858](https://github.com/mudler/vllm.cpp/issues/2858)). **Neither sequence
  moved.** On the released UD-IQ1_S artifact, production configuration, greedy,
  `thor:gpu0`: `--device cpu` emits `11751 13 15767 411 2029 11 1092 369` and
  `--device cuda` emits `11751 13 15767 411 1928 11 628 567`, both byte-identical
  to PREFILLDIV's, still **5 of 8**. `VT_GDN_CHUNKED=0` emits the same eight ids
  as the default, so the annotations naming it as the way to reproduce the old
  sequence were correct but no longer load-bearing.
  **The arm is not inert, and it was NOT enough to read the ids to know that.**
  All three CPU arms agreeing left T4's exact ambiguity — a flag whose two arms
  coincide, or an arm that is not reached — and no log line names the arm. The
  committed `VT_Q4EXP_LAYER_FP` fingerprint settled it: `VT_GDN_CHUNKED` moves
  `L00 blk` by `3.702e-04` from a bit-identical input, 26 of 42 taps differ
  (**annotation 2026-09-04, [#2877](https://github.com/mudler/vllm.cpp/issues/2877):
  `42` is not model coverage — the instrument printed 1311 taps and the differ's
  `load()` collapsed them to 14 tags x 3 steps because it splits on `'='` while the
  tap prints `L%+03lld`. It is not "layer-0 coverage" either, which is how this
  line first put it: 11 of the 14 tags are tapped inside the decoder loop and keep
  layer 0, but `emb`, `wide` and `out` are tapped at `il = -1` and are never layer
  0 — the differ's own largest value is at `('2', None, 'out')`. The routing
  conclusion survives, since it needs one moved tap and not 42, but read nothing
  about layers 1..47 into it**),
  and
  the CPU arm lands `1.772e-05` from CUDA where the sequential arm sat
  `3.525e-04` away — **19.9x**, with `s.attn` 24.1x and `mhc.mix` 11.6x. The
  CPU-sequential and CUDA `L00 blk` readings reproduce PREFILLDIV's to the
  printed digit on a different tree, which is what makes this a re-measurement.
  **What the row bought is a 20x reduction at the block and no id.** `out`
  improves 1.23x only, because the MoE residue moved the other way
  (`moe` 1.269e-04 -> 2.289e-04 from an input 11.6x closer). With the Gated
  DeltaNet source closed, that residue is the whole-model divergence. Scope's
  refusal to promise token agreement is vindicated in both directions: agreement
  did not improve, and it did not degrade either.

  **ANNOTATION 2026-09-04 — EVERY RATIO IN THE BULLET ABOVE IS WITHDRAWN, IN BOTH
  DIRECTIONS** ([#2877](https://github.com/mudler/vllm.cpp/issues/2877), full
  reasoning, both framings and the committed control in [the ARMTOKENS evidence
  file](../../docs/bench-evidence/qwen4exp-gdn-chunked-token-ids-20260904.md)). The
  sentences are kept so the shape of the error stays visible. The replacement claim
  is NOT "the residue did not grow" — that is the same overreach with its sign
  flipped. (a) **THE ONE THAT SURVIVES:** `LayerFp` returns early on
  `s.step >= s.budget` (`qwen4_exp_forward.cpp:118`), so `VT_Q4EXP_LAYER_FP=3`
  covers forwards 0, 1, 2 = tokens `11751 13 15767`, which **agree on both arms**,
  while the three disagreeing ids are emitted at forwards 4, 6 and 7. **No
  `VT_Q4EXP_LAYER_FP` fingerprint has yet observed a disagreeing step.** Those
  forwards are causally upstream of forward 4 through the recurrent state, so the
  taps are not irrelevant to the disagreement; they simply never observe it.
  **CORRECTION 2026-09-05
  ([#2969](https://github.com/mudler/vllm.cpp/issues/2969)): this bullet said "no
  instrument", which is wrong for the other tap.** `VT_MOE_SEL_FP` counts MoE
  block invocations and not forwards, and `qwen4_exp` runs 48 MoE blocks in each
  forward, so wave MOEDIV's 384 calls reached forwards 4, 6 and 7. That reading is
  VOID for a different reason: its CUDA arm answered the degenerate pre-#2550
  sequence `11751 271 271 271 271 271 0 0`. A non-degenerate arm was what was
  missing, not a wider window. **SECOND CORRECTION 2026-09-06
  ([#2998](https://github.com/mudler/vllm.cpp/issues/2998)): that arm has been
  run, and forwards 4, 6 and 7 are read**
  ([evidence](../../docs/bench-evidence/qwen4exp-moe-selection-fwd-20260906.md)).
  It does NOT restore the attribution this bullet withdrew: the disagreeing
  forwards flip 127 of 144 expert slots and the agreeing ones 116 of 192, and
  agreeing forward 5 flips 48 of 48 while disagreeing forward 4 flips 33 of 48, so
  a flip count separates nothing in either direction. (b) `rel(sumabs)`
  is a difference of NORMS, not a norm of DIFFERENCES, and its under-report is a
  DISTRIBUTION rather than the `~122x` single seed draw this line first quoted: at
  `n = 12800` over 400 seeds the median is 75x (sigma 1e-3) to 140x (sigma 1e-4),
  p05..p95 34..1500, and at a **fixed** true divergence two readings differ by a
  median **2.1x** and **24x** at p95 — 400 seeds, not the six this annotation's
  own first version (`c51c484db`) read `4.64x` from. Two significant figures is
  what 400 draws buy; the three-figure set this line carried until #2879 came from
  a script that was never committed. `MetricSpread` now READS this spec off disk
  and compares every figure quoted here to the value it draws
  (`test_the_PUBLISHER_reproduces_both_specs`), so the two cannot move apart.
  (c) **One framing, applied to every tap.** `3.525e-04` and `1.269e-04` are the
  same PREFILLDIV column, so a comparison that is mismatched for the MoE row is
  mismatched for the block row. Read as algorithm-**matched** CPU-vs-CUDA pairs,
  `L00 blk` moved **16.7x FURTHER** (1.062e-06 seq/seq -> 1.772e-05
  chunked/chunked) where the bullet claims 19.9x closer, and every other layer-0
  tap moved further too (`s.attn` 1.71x, `mhc.mix` 2.02x, `moe` 3.15x, `s.mlp`
  2.34x, `out` 2.29x; `mhc.inj` alone to exactly zero). That direction is what the
  chunked decomposition's larger reassociation freedom predicts — it lands
  `2.29e-04` from the exact answer where sequential lands `1.15e-08` — and it is
  not a defect. Against (b): 19.9x and 16.7x sit at 6% and 7% of the metric's
  own no-change distribution, and no change at all produces 1.80x, 2.02x, 2.34x
  and 3.15x in **59%, 52%, 45% and 33%** of draws — this line first called that
  "between its 33rd and 59th percentile", which states the complement and inverts
  the ranking: 3.15x sits at the **67th** percentile of no change, not the 33rd.
  **Nothing here is ranked, in either direction.** The
  residue's mechanism was already named by
  [#2552](https://github.com/mudler/vllm.cpp/issues/2552) — the keep-quant grouped
  expert GEMM's reassociation plus a bimodal top-k term at a 32.9% exact-tie rate,
  both floors, both faithful mirrors and neither a defect. **STILL NOT CLOSED, for
  a narrower reason since 2026-09-06**
  ([#2998](https://github.com/mudler/vllm.cpp/issues/2998),
  [reading](../../docs/bench-evidence/qwen4exp-moe-selection-fwd-20260906.md)):
  this line said the matched pair's `4.324e-05` lands inside #2552's layer-0 flip
  bracket (2.139e-05 no flip .. 4.999e-04 flip) and that `VT_MOE_SEL_FP` was never
  run on it. **It has been run, and layer 0 does NOT flip**, so the no-flip bound
  lifts to `4.324e-05` and the layer-0 residue at this pair is the keep-quant
  expert GEMM's reassociation rather than the top-k term. The cause of the three
  disagreeing ids is what stays open, under
  [#2999](https://github.com/mudler/vllm.cpp/issues/2999); it needs the 68 GB
  artifact and a GPU.

- **`main` did not compile with CUDA, and that is how this wave found out**
  ([#2861](https://github.com/mudler/vllm.cpp/issues/2861), fixed in the same
  flow). `f2bda11e3` left `vt::cuda::<unnamed>::ChunkedPrefillEnabled` with no
  caller and `-Werror=all-warnings` rejects it (nvcc 177-D), so the first lease
  spent 920 s and produced no arm. The bullet above about no CUDA gate having run
  understated it: the CUDA arm of this row was **unbuildable**, not merely
  ungated, from `73db7a8a3` until the fix.
- **T2/G3 is partial.** The two dtype properties a caller can observe are gated
  (the output is bf16-representable on bf16 input; the state is f32 and not
  bf16-rounded). The nine INTERIOR placement sites are gated only in aggregate,
  by G1's bar, which separates the bf16 placement from an f32-intermediate one
  by 4x. A per-buffer format assertion would need a testing hook the seam does
  not have.
