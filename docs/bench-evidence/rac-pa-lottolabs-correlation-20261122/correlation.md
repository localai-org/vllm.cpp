# RAC / PA latent defects vs the Lottolabs P150 trace-capture fixes — written correlation

Row: `QWEN38-DFLASH2-STAGE-C`. Issue: ISSUE-LOCAL-01M82AGA1RXW8A4GRQHYYT67HE.
Date: 2026-11-22. Status: **evidence-based hypothesis, NOT a fix**. This note
changes no source. W32 (mk-C's in-flight dispatcher command-stream trace) owns
any code change on the staging seam.

Sources compared:

- Our evidence chain: W28 (`../qwen38-dflash2-stage-c-w28-20261118/README.md`),
  W30 (`../qwen38-dflash2-stage-c-w30-20261120/README.md`), W31
  (`../qwen38-dflash2-stage-c-w31-20261122/README.md`), and the two latent
  issues `.agents/issues/_owed/ISSUE-LOCAL-01M7JYHQJEANK9Z6S0SJWKXWBC.md` (RAC)
  and `...ISSUE-LOCAL-01M7JYJ1WDJJ3M7E23VQCYS6KC.md` (PA cache-disabled).
- External reference: `Lottolabs/Inkling-2-Mini-TT-Mixed-BFP4-BFP8-P150` @
  `b0ad7060246f192aa1823b6fb8bd17f6ead194ae`, file
  `runtime/inkling2/tt/NOTES-model.md`, section "Second-sequence hang: root
  cause and fix" (lines 97-124 of the downloaded copy). Local study note:
  `/tmp/opencode/inkling-tt-study/STUDY.md`.

## Our defect class, as established by W28/W30/W31

1. **Write-set and timing (W28).** The writer stamps exactly 512 bytes: the
   zeroed BFP8_B exponent headers of the first 8 tiles (the first page of each
   of 8 DRAM banks) of a freshly staged weight. The bytes are already zero at
   the first canary read (`ARRIVED-ZEROED`, strikes at call indexes 52/58/61),
   while the stage-time read of the same buffer was clean in 185/185 checks.
   Nothing is enqueued between the two reads on the single CQ, so the write is
   out of band of the command queue. The allocator block table is clean
   (`blocks=1`, no overlap) and the typecast is exonerated down to a 0xFF
   prefill target. The slice-based probe underreports (1/8 vs 8/8), so only
   direct reads are trusted.
2. **Trace replay falsified (W28).** The struck leg runs fully eager; no
   `TraceEndCapture` fires and `capture_hit=0`, yet the strikes reproduce
   byte-identically. A capture-replay mechanism cannot explain this writer.
3. **Engine dependence (W30).** Three seam-level repros (pure install loop,
   program-cache-warm op mix, resident-plus-churn) are all clean. The writer
   needs the fuller engine state: ~25 GB residency, GGUF mmap sources, drafter
   staging, trace-region / async-executor machinery. The DNR repair is the
   working mitigation; the "disable DNR by default" premise failed and was
   reverted, with a measured ~3.73x acceptance multiple.
4. **Launch-path exhaustion (W31).** TA/RTA instrumentation of every patched
   tensor binding and every legacy per-core runtime arg found no victim-window
   address on any kernel launch path. No cached unbound kernel retains a victim
   address in its CRTA, and no legacy RTA carries one. The refined hypothesis
   is (a) a stale dispatch-level command replay embedding a freed staging-temp
   base, or (b) an on-device agent computing its write address from a stale
   descriptor. W32 is tracing host command construction to separate them.
5. **Latent issues share the signature, not yet the mechanism.** The RAC issue
   reports recycled-memory bits at slot 2 / KV head 2, deterministic per binary
   layout, passing in isolation and at the pre-merge head but failing in the
   suite. The PA cache-disabled issue reports value-level garbage at seq>=2
   only under rich suite state, with per-call recompile kernel-arg/scratch reuse
   as a suspect. Both are "garbage depends on allocator/JIT history", which is
   the same *class* as the W28 writer, but neither has been shown to be the
   same writer or the same address path.

## Per-fix correlation

For each Lottolabs fix: what our class predicts, whether the mechanism is the
same, and whether it transfers.

### Fix 1 — allocate every persistent tensor in the constructor

- **Their mechanism.** The first device allocation after a trace capture logs
  "Allocating device buffers is unsafe due to the existence of an active
  trace". Replay writes the DRAM/L1 that the capture's intermediates occupied,
  so a buffer allocated after capture can sit there and be overwritten. They
  moved all persistent tensors (all bucket tables, the gather index, the trace
  input) into the constructor.
- **What our class predicts.** A stale address held across free-and-reuse: the
  W28 victim block was recycled onto the live staged weight, and W31 shows the
  writer holds an address no current launch arg supplies. The *lifetime shape*
  matches — an address becomes valid for reuse while some other agent still
  treats it as live — but W28 explicitly falsified capture replay in the
  struck leg, and the RAC issue's failing leg is EAGER. So the trigger is not
  capture replay in our case.
- **Same mechanism?** No. Same symptom class, different owner of the stale
  reference (their replay vs our dispatch command / device descriptor).
- **Transfers?** Partly, as hardening only: our traced decode path should also
  avoid post-capture allocation. It is not a repair for W28/W31 or either
  latent issue, and it must not be recorded as one.

### Fix 2 — share one prefill gather index across layers

- **Their mechanism.** Attention prefill gathered convolution-state rows via a
  device index (`ttnn.embedding`); allocating it lazily on the first prefill
  (after capture) cost 77 MB per layer. One index per kind, allocated in the
  constructor, cut it to 154 MB total from up to 1.85 GB.
- **What our class predicts.** Nothing specific. Our BFP8 staging seam
  (`Bfp8TypecastInstall` in `src/vt/tenstorrent/tenstorrent_residency.cpp:953`)
  allocates a fresh bf16 H2D temp and a typecast output per call; there is no
  per-layer gather index and W28 exonerated the allocator and the typecast.
- **Same mechanism?** Only the "lazy allocation after a capture" half, which
  our struck legs do not have.
- **Transfers?** No code transfer. The memory-footprint discipline is a
  useful lesson if we ever adopt a per-layer prefill index, nothing more.

### Fix 3 — route the heads reshape through row-major (closest match)

- **Their mechanism.** A tile-layout `ttnn.reshape` moving data
  (`_heads`: `[1,1,T,n*w]` → `[1,T,n,w]`) built a page-mapping tensor on the
  device on a program-cache miss and retained it with the cached program.
  `reshape_tile_mode=RECREATE` was ignored in their tt-metal build. Prefill
  programs compile lazily, i.e. after capture, so the maps landed in the
  trace's freed intermediates; replays overwrote them and the next prefill
  with the same shapes read garbage maps and hung. Fixed by untilize →
  reshape → permute → tilize, which carries no device-side state.
- **What our class predicts.** Program-cache-associated device state aliasing
  recycled memory under rich JIT history. This is exactly the shape of the PA
  cache-disabled issue (value corruption only with 80+ cases of allocator and
  program history ahead, suspects: per-call recompile kernel-arg/scratch
  reuse), and it is adjacent to W31's "stale descriptor" branch. It is NOT the
  W28 writer: their symptom is a hang from garbage page maps, ours is 512 B of
  zeroed exponent headers or wrong computed values.
- **Same mechanism?** Plausibly the same *family* — device state created on a
  program-cache miss and then aliased — but unproven for our kernels. It is
  the one Lottolabs fix with a concrete, testable transfer hypothesis.
- **Transfers?** Yes as an audit rule, not as a patch: audit our decode/PA
  kernels and their program factories for device tensors (page maps, reshape
  maps, scratch) created on a cache miss and retained with the program, and
  prefer a stateless layout path. W32 should decide whether the writer is a
  baked command or a device descriptor before any kernel is changed.

### Fix 4 — compile all decode buckets before any capture

- **Their mechanism.** Decode programs own reshape maps too, so every bucket
  is compiled before the first trace is captured. One trace per power-of-two
  bucket, with per-bucket attention tables, all allocated in the constructor.
- **What our class predicts.** W28's struck leg has no capture at all
  (`capture_hit=0`), and W30's seam repro was clean with a warm program cache.
  There is no evidence that lazy bucket compilation drives our writer.
- **Same mechanism?** No.
- **Transfers?** Only if we later adopt traced decode with buckets. Not
  applicable to the current engine.

### Fix 5 — assert `new_low_blocks` empty after every sequence

- **Their mechanism.** Record the bottom-up DRAM blocks allocated right after
  capture and assert no new ones appear after any sequence. This caught both
  post-capture allocations in their hang and guards the fix.
- **What our class predicts.** Our W28 check is `blocks=1` on the block table
  at each strike — it asks "did the allocator double-book this range?", a
  different question than "did anything allocate below the post-capture
  baseline?". A `new_low_blocks` equivalent would not catch the W28 writer,
  because there is no capture in that leg; it would catch a post-capture
  allocation in the RAC capture-safe test.
- **Same mechanism?** No, but it is a mechanism-agnostic detector.
- **Transfers?** Yes as instrumentation. Add a baseline-block assertion to the
  capture-safe suite if W32 confirms a capture-adjacent path. It is a gate,
  not a repair, and it belongs to the owner of that suite.

## Secondary — the P150 reference map for our benchmark ladder

From the same package (NOTES-model.md, NOTES-attn.md, README.md); these are
reference numbers from a different model and quantizer, not gates for us.

- **Decode buckets.** Power-of-two 4096..max_len, one trace and one
  `AttnDecodeTables` per bucket, all allocated in the constructor; 72 MiB trace
  region per bucket, 512 MiB total. Traced step cost grows 68.4 ms at 4096 to
  94.1 ms at 131072. Their reason for buckets: a max-context trace costs
  +26 ms per step.
- **Device-side sampling.** Temperature, top_k unset or 1-64, top_p, min_p,
  penalties and seeds run on device over the full vocabulary; exact
  speculative rejection sampling for sampled requests; seeded spec == plain.
  Host fallback only for logprobs, logit bias, structured output, temperature
  > 8 and top_k > 64 with top_p < 1.
- **Headline serving numbers.** Greedy plain 87.1 tok/s mean, greedy spec 108.0
  (1.24x), sampled T=1.0 spec 98.5; 1M-token context served at 23.0 tok/s with
  3/3 pass keys. These are the numbers a healthy P150 Inkling-2 port would
  compare against.
- **Our current reference point.** W30 measures the DNR-mitigated decode at
  ~24.8 s/step versus ~6.65 s/step with DNR off, a ~3.73x acceptance multiple
  caused by the unfixed writer. A reference map should record both their
  healthy traced path and our repair-cost multiple, and keep them explicitly
  separated by model and quantization.

## Minimal falsification experiments (run before any code change)

Both surviving hypotheses are cheap to test and should be falsified or kept
alongside W32's dispatcher trace, not ported on the strength of this note.

1. **Program-cache-miss device state (PA cache-disabled / W31 stale descriptor).**
   In the pin-side diagnostic tree only, log at each program-cache miss on the
   decode / PA path whether a program factory allocates or retains a device
   tensor (reshape map, page map, scratch), with its DRAM address. Run the
   `ProgramCacheToggleForTest(false)` arm inside the *full suite* - the arm is
   green in a fresh process and red only with 80+ cases of history ahead.
   Correlate any logged address with the value-garbage window. Falsified if no
   device state is created or retained on a miss, or if the garbage appears at
   the same positions with logging disabled and with the state suppressed. A
   second, independent falsifier is a scratch pin that replaces the offending
   tile-layout reshape with the untilize/reshape/permute/tilize path: if the
   suite arm stays red, this mechanism is not the PA defect.
2. **`new_low_blocks` detector (RAC capture-safe suite).** Snapshot the
   bottom-up DRAM block list immediately after trace capture in the RAC
   capture-safe test, then assert after every case that no new low block has
   appeared. Run the *suite*, not the isolation case, since the defect only
   appears there. Falsified if the assertion stays clean on the case that
   currently fails (slot 2 / KV head 2) - that would show post-capture growth
   is not the trigger and the detector has no diagnostic value for this row.
   This is test-only instrumentation; it must not change production paths.

## Bottom line

One of the five Lottolabs fixes (program-cache-miss page-map state) is a
credible *audit hypothesis* for the PA cache-disabled defect and W31's stale
descriptor branch. One (baseline-block assertion) is a credible detector for
the capture-safe suite. The other three address a capture-replay trigger that
W28 falsified in our struck leg. None is a fix for our writer, and none should
be ported as one. W32's dispatcher command trace is the deciding instrument.
