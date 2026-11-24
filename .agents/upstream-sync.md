# Upstream sync protocol

The reference checkout, fetch remote and eligible gate host come from the
untracked `developer-preferences.md`. The parity pin and classification rules
are repository-wide; machine paths are not.

**Active pin:** the authoritative pin, runtime strings and gate constraints are
the `parity-pin` block in this file (line 101); that block is what `tools/bench/`
reads and it was advanced to `a7c23ac96d` on 2026-09-26. Older prose here named
the 2026-09-03 `e126687a9a` advance as current, which is no longer true; its
history stays in
[specs/upstream-pin-advance-e126687.md](specs/upstream-pin-advance-e126687.md)
and [`sync/2026-09-03-e126687-advance.md`](sync/2026-09-03-e126687-advance.md)
([#2817](https://github.com/mudler/vllm.cpp/issues/2817)).

**This advance re-validated NOTHING, and that is the difference from the last
one.** The 2026-07-26 advance re-captured goldens on the new oracle on GB10 and
recorded zero real drift. This one ran no build, no lease, no GPU and no golden
re-capture. Every committed golden in this tree was captured against `555967922`
or earlier. **This sentence used to end "and the declared token-exact gate at
this pin is owed by [#2794](https://github.com/mudler/vllm.cpp/issues/2794)",
and that is no longer true.** Read live 2026-09-12,
`gh issue view 2794 --repo mudler/vllm.cpp --json state,closedAt,stateReason`
gives `CLOSED`, `COMPLETED`, `2026-09-06T07:00:58Z`; the gate ran at this pin on
2026-09-04 and PASSED, as [`NOW.md`](NOW.md) records. The declared token-exact
gate — the OPT-125m oracle gate at six prompts and sixteen tokens — is therefore
DISCHARGED, not owed. **Two obligations remain owed at this pin, and this
sentence must not be read as naming only one.** (1) The BENCHMARK step-6
re-baseline, [#2818](https://github.com/mudler/vllm.cpp/issues/2818), still
`OPEN` at the same read. (2) **The other four strict goldens** — 27B W4A4,
32B-NVFP4A16, 35B, Coder — which the pin advance re-captured at `5559679229` and
none of which has been re-captured at `e126687a9a`. That second obligation is
deliberately NOT anchored on an issue, because #2794 is closed and an `## Owed`
bullet pointing at a closed issue is how an obligation stops being read: it
lives under `## Owed` in
[`specs/upstream-sync-headpin-tokengate.md`](specs/upstream-sync-headpin-tokengate.md)
(the bullet at `:569`), and whoever takes it files the issue then. The same
split is recorded at [`oracles/vllm.md`](oracles/vllm.md) `:77` and `:93` and in
[`sync/2026-09-05-e126687-pingate.md`](sync/2026-09-05-e126687-pingate.md)
`:45`. Step 5 did not run
either, so **at least 177** files whose `Ported from:` header names `55596792`
are now BEHIND the pin, counted over `include/`, `src/` and `tests/`. **It is a
FLOOR, not a count**, and the reason is in
[`sync/2026-09-03-e126687-advance.md`](sync/2026-09-03-e126687-advance.md) §5.3:
a header wraps, so no single probe sees all of them, and the figure is the union
of three probes each with a different blind spot. 556 files carry such a header
at all, and at least 330 still name `e24d1b24`, the pin before that one — they
were already behind, and this advance neither causes nor fixes that. This
paragraph is the ledger note §Concepts asks for, and the queue is
[#2611](https://github.com/mudler/vllm.cpp/issues/2611).

**Step 6 ran AFTER step 7, by developer ruling, and the ruling is recorded as
one.** §"The sync cycle" orders the re-measurement before the advance. Wave STEP6
([#2771](https://github.com/mudler/vllm.cpp/issues/2771), landed as
[#2783](https://github.com/mudler/vllm.cpp/pull/2783)) established by hermetic
probe that this order cannot be executed here: `tools/bench/online_gate.py`
refuses every oracle that is not the pinned one, on the distribution version, the
runtime version, the commit and the FlashInfer version, and all four come out of
the block below by way of `tools/bench/serve_low_common.py`. Editing the block IS
step 7, so step 6 is a precondition of its own precondition. STEP6 filed the
ordering `NEEDS_DECISION`.

> **The developer ruled that the pin advances first and step 6 re-runs against
> it.** That inverts a documented step order. It is a developer decision, not a
> reading this protocol permits, and its consequence is part of the ruling:
> **step 6 becomes post-hoc validation, and a red there requires REVERTING the
> pin, not holding it while the rows are re-argued.** Owed by
> [#2818](https://github.com/mudler/vllm.cpp/issues/2818), which covers the five
> affected rows across `vllm-online-serving` and `speculative-decoding` and the
> withdrawn `nvidia-cutlass-dsl` discharge (`C1c`). No checker enforces the
> inversion and none is added; a gate over step order would have to read the same
> block it is deciding about.

**Prior pin (advanced 2026-07-26):** `555967922` (vLLM 0.26.0.dev0 + transformers
5.14.1, Torch 2.13.0, FlashInfer 0.6.15.post1, CUTLASS DSL 4.6.0), FLIPPED from
`e24d1b24`/0.25.0 at W5 (`bc415a3e`), see
[specs/pin-advance.md](specs/pin-advance.md) §7. That advance re-validated with
zero real golden drift (27B-W4A4 + 32B-NVFP4A16 bit-identical, 35B/Coder
byte-stable) and unblocked DFlash (vllm#40898), Gemma-4 (`transformers.gemma4`),
and OLMo-3 (nested rope); a `vllm-oracle-v0.25.0-stage` rollback is preserved.

**The pin, as a running oracle reports itself.** The paragraph above names the
release; the block below carries the exact strings a runtime identity check can
compare. Every value in it was MEASURED at the target and none was transcribed:
the runtime and distribution strings from `IMPORT VLLM_VERSION` and
`DIST DIST_VERSION` read from `cd /`
([`sync/2026-09-02-e126687.md`](sync/2026-09-02-e126687.md) §5.4, reproduced by
the source build in [`sync/2026-09-03-e126687-runhalf.md`](sync/2026-09-03-e126687-runhalf.md)
§2 and §5), the FlashInfer version from `PIPLIST flashinfer-python` in the same
lease (§5.2). They are not derivable from the release number — the oracle reports
`0.28.1rc1.dev132+ge126687a9`, not `0.28.1rc1`. **This sentence used to end "and
its distribution metadata adds a `.precompiled` suffix its runtime string
lacks", which the correction below made false and which nothing then removed;**
on a source build the two strings are equal, and a source build is the only mode
that runs on this architecture. `tools/bench/` reads this
block rather than duplicating it (#520); the duplicate drifted once, and the
harness spent 17 days *refusing* the oracle this record required. Advance it only
as part of a sync cycle, from a measured oracle, never by transcribing a version
number.

```parity-pin
vllm_commit = a7c23ac96d7806e7c7e7d862eadbce5a33529b94
vllm_runtime_version = 0.3.0.dev267+ga7c23ac96d
vllm_distribution_version = 0.3.0.dev267+ga7c23ac96d
flashinfer_version = 0.6.15.post1
```

**This field now records the SOURCE build, corrected 2026-09-04 on developer
instruction (#2896).** The `.precompiled` suffix is a BUILD-MODE property, and
recording it made the harness unsatisfiable on this architecture: `online_gate.py`
compares this field for EQUALITY, and on aarch64 `VLLM_USE_PRECOMPILED=1`
downloads nothing and leaves a 13,872-byte editable install with **no compiled
extensions** — so the only mode that matched the string could not execute a
kernel, while the only mode that can execute was refused. No cross-engine
measurement was possible at this pin in either direction.

**The new value is measured, not transcribed**, which is the constraint this block
exists to enforce. `VLLM_USE_PRECOMPILED=0 VLLM_TARGET_DEVICE=cuda` at this
revision produced
`vllm-0.28.1rc1.dev132+ge126687a9-cp312-cp312-linux_aarch64.whl`, with no suffix
([`sync/2026-09-03-e126687-runhalf.md`](sync/2026-09-03-e126687-runhalf.md) §2,
lines 52 and 56). Writing a string nobody read is the #520 failure and is not what
happened here.

**A second build has since ATTESTED the same string on a second device, from the
wheel name.** Job `7386f034-246a-4af5-9a04-f98aafffce54` built this revision from
source on `dgx:gpu0` (GB10, `sm_121a`) on 2026-09-04 and produced
`vllm-0.28.1rc1.dev132+ge126687a9-cp312-cp312-linux_aarch64.whl`, with no suffix
(its log at L111 and L114). A wheel filename carries the DISTRIBUTION version, so
it attests THIS field — but as a second corroboration OF THE SAME KIND
`5d97007c2` already took from the Thor build's wheel name, on a second device,
not as a reading of the field.

**That job did not read the field, and an earlier version of this paragraph said
it did.** Its identity block is `cd /` then `import torch, vllm` then
`print("VLLM_VERSION", vllm.__version__)`
([`scripts/tokengate-e126687-job.sh`](scripts/tokengate-e126687-job.sh) lines
234-241 at `17fcdd5e3^`, the byte sequence that ran; sha256 `bf77dfbb3383d837`
is that WHOLE FILE, which is the hash the share copy was matched on, and lines
234-241 on their own hash `23c52138b006feda`).
The directory is right and the FIELD is wrong: that is the RUNTIME string. All
316 of the job's log lines carry zero `importlib` and zero `metadata`, against
positive controls of one `VLLM_VERSION`, six `autotun` and four `kernel_warmup`
counted the same way, once on the raw lines and once on the whitespace-stripped
file so a wrap could not defeat the search. Of every block in this repository
this is the one where that substitution cannot be made, because the
`.precompiled` question IS the difference between the two fields, and this block
is the one whose own prose says every value in it was measured and none
transcribed. The job's full record is
[`../docs/bench-evidence/opt125m-token-gate-e126687-dgx-20260904.md`](../docs/bench-evidence/opt125m-token-gate-e126687-dgx-20260904.md).

**The distribution field itself was read on 2026-09-05, by a different job.**
`d7908816-96a3-40ed-8024-c3ad0cc34d77` read `importlib.metadata.version("vllm")`
from `cd /` against a source build on `dgx:gpu0`, and recorded
`metadata_vllm='0.28.1rc1.dev132+ge126687a9'` beside an equal
`runtime_vllm___version__` in `/workspace/step6-c1a-2818/versions.txt`. Its own
log records the read succeeding: `VERSION_READ_RC=0` at `step35.log:73`, the
`metadata_vllm=` block at `:75`, and `step35.sh:51` is the line that writes the
file. That is the reading
[#2818](https://github.com/mudler/vllm.cpp/issues/2818) item 1 asked a lease to
take, and it is what this field now rests on.

**It was NOT taken by `8c4f639c-0ebb-44de-b5e4-0e192143f7c3`, which this file
credited until now.** That job attempted the same read and it FAILED:
`VERSION_READ_RC=1` at `stepA.log:1119`, its `cat` of `versions.txt` produced
nothing, and `versions.err` is
`ModuleNotFoundError: No module named 'vllm'` — its venv `/tmp/oracle-e126687`
never got the wheel installed, and the job records that too:
`INSTALL_VLLM_RC=1` at `stepA.log:912`, after `Failed to build installable
wheels for some pyproject.toml based projects` / `instanttensor` at `:909-910`. The job nevertheless finished `succeeded` with
`exit_code 0`, because `stepA.sh` echoes `VERSION_READ_RC=$?` instead of
propagating it. **An exit code attests the job, never a step inside it**, and
that exit 0 was the whole of the corroboration this attribution rested on. Three
things place the file with `d7908816`: its mtime `2026-09-05 22:42:39Z` falls
inside that job's `22:33:48Z`–`23:08:22Z` window and after `8c4f639c` finished
at `22:33:48Z` (`rc jobs -n 200 --device dgx:gpu0 -o json`, read directly); it
carries
`vllm___file__='/root/step6/venv/lib/python3.12/site-packages/vllm/__init__.py'`,
which is `step35.sh`'s venv and not `stepA.sh`'s; and `stepA.sh`'s own `cat` of
it is empty.

`rc logs` returns nothing today for EITHER job — a positive control on the same
client still returns `7386f034`'s 316 lines — so the share artifacts under
`/workspace/step6-c1a-2818/` are the only record, which is why the record exists.
[`sync/2026-09-05-e126687-step6-c1a.md`](sync/2026-09-05-e126687-step6-c1a.md)
§R1 states the same reading and still names the wrong job; that record belongs
to another wave and its repair is owed by
[#2997](https://github.com/mudler/vllm.cpp/issues/2997).

Whoever next builds the oracle records WHICH mode they used and, if the field is
wrong for it, corrects it from THAT measurement — never by editing the block to
make a run pass. **A build that carries `.precompiled` on aarch64 is not a
runnable oracle**, so a future correction back toward it needs a measurement
showing the install can actually serve.

**`vllm_runtime_version` must carry a `+g<sha>` segment naming `vllm_commit`.**
That is a permanent constraint of this design, not a property of today's values:
`assert_oracle_commit` extracts the segment and requires it to prefix
`vllm_commit`. A released-wheel shape (`0.26.0`, no local version segment) is
therefore unusable — set the block to one and the harness refuses every oracle
including the pin itself; measured 2026-08-12, 34 of the 233 `tests/tools` cases
go red. Fail-closed and CI-caught rather than silent, but it means a pin advance
is taken from a source build's measured `vllm.__version__`, never transcribed
from a release number. If a future pin is genuinely a released wheel, give the
commit its own asserted field first; do not delete the assertion to make the
block parse.

**A shallow clone rewrites the version prefix, and the discrepancy that taught us
this is still OPEN.** A build of the PRIOR pin inside an `rc` lease on
`dgx:gpu0`, 2026-08-18, reported `vllm.__version__ = 0.1.dev1+g555967922` against
that pin's recorded `0.23.1rc1.dev1511+g555967922`
([#1185](https://github.com/mudler/vllm.cpp/issues/1185),
[`specs/oracle-wheel-in-lease.md`](specs/oracle-wheel-in-lease.md)). The binding
constraint held, because the `+g<sha>` segment named `vllm_commit`. The cause is
the shallow fetch that build used: `setuptools_scm` cannot count the commits
since the last tag and falls back to a default. The mechanism is general and
still applies. It is also why the CURRENT block's full string is trustworthy: the
build that measured it recorded `SHALLOW=false REVCOUNT=20591
GIT_DESCRIBE=v0.28.1rc0-132-ge126687a9`
([`sync/2026-09-03-e126687-runhalf.md`](sync/2026-09-03-e126687-runhalf.md) §2),
so `setuptools_scm` had the history it needs. A gate that compares the FULL string
needs either a deep fetch or an explicit pretend-version carrying that reason. Do
NOT edit the block above to match a build. The block records the pin, and a
shallow clone is a property of one job.

**vLLM-Omni's pin does NOT live here.** It is a separate repository, and under
AGENTS.md §"When vLLM has no implementation" every oracle carries its own file:
[`.agents/oracles/vllm-omni.md`](oracles/vllm-omni.md), whose `oracle-pin` block
is the one place its revision is recorded. Do not add a second pin block to this
file; one file per oracle, read by glob, is what keeps a pin from becoming a
surface every change has to write. What belongs HERE is only the part that is
about the relationship between the two, which the oracle file cannot state on its
own: see the omni rules under §Rules, and
[specs/upstream-omni-pin.md](specs/upstream-omni-pin.md) (#633) for why they hold.

**Prior cycle (2026-07-12):** audited target v0.25.0 `702f4814fe54`; report
[`sync/2026-07-12-702f481.md`](sync/2026-07-12-702f481.md). The exact 145-commit
`e24d1b24..702f481` delta was classified (94 `INVENTORY`, 51 `IGNORE`, no
trace-independent `PORT-NOW` change in the implemented Qwen T0 slice). Immutable
`9cc7191` established the first new 27B denominator, while immutable `3f256ab`
superseded it at **55/124 axes pass, 69 fail**; the current binding is `9ecd9d0`
**114/124** (see roadmap_v1.md).

- **Reference checkout:** `${VLLM_SOURCE}`, branch selected by the developer
  preferences (normally `main`)
  (https://github.com/vllm-project/vllm).
- **STARTING PIN (MVP phase):** `e24d1b24` (2026-07-02) — the vLLM commit we
  port *from*. During the MVP build-out this is **not a parity claim**: we are
  not "at" this pin, we are building toward it, and the feature gaps vs this
  pin are documented in [porting-inventory.md](porting-inventory.md) (tier
  assignments + status markers = the gap record). Golden dumps, file headers,
  and benchmark baselines are all taken at this pin so the target stays fixed.
- Once the MVP gates pass, this line becomes the **PARITY PIN** — a statement
  of equivalence for the T0 surface — and moves only via sync cycles.

## Concepts

- **Starting pin (MVP)** — the fixed upstream snapshot the MVP is built
  against. Gaps vs it are normal and tracked in the inventory, not hidden.
- **Parity pin (post-MVP)** — one repo-wide vLLM commit. "We have feature X"
  always means "X as of the pin". Never compare against a moving target.
- **Omni parity pin** — the same idea for `vllm-project/vllm-omni`, recorded in
  [`.agents/oracles/vllm-omni.md`](oracles/vllm-omni.md), not here. It is a
  second pin rather than a second value of this one: it names the vLLM commit
  *it* ran against, which need not be ours.
- **Per-file pins** — every ported file's header records the upstream path +
  the upstream commit it matches. Normally equal to the parity pin; a file may
  be temporarily ahead (hot-fix port) but never behind without a ledger note.
- **Sync cycle** — the repeatable task that advances the pin. Bounded,
  mechanical, agent-runnable on a cadence (weekly, or on demand, e.g. when a
  needed upstream fix lands).

## The sync cycle (repeatable task)

1. **Fetch & choose target.** `git fetch origin main` in the reference
   checkout. Target = `origin/main` HEAD (or a specific commit/tag if the user
   asked for one). Do NOT move the working tree yet if mid-cycle work exists.
2. **Enumerate.** `git log --oneline PIN..TARGET -- <subtree>` for each
   subtree we mirror. Because our structure mirrors upstream 1:1, the ported
   surface is derivable: map `src/vllm/**/*.{h,cpp}` back to `vllm/**/*.py`
   (plus `tools/parity/` dump scripts and ported test files).
3. **Classify** every commit touching those paths (upstream PR # is in the
   subject line):
   - **PORT-NOW** — changes behavior of code we've already ported.
   - **INVENTORY** — adds a feature we haven't ported; add/annotate it in
     [porting-inventory.md](porting-inventory.md) with its `vllm#` ref and a
     tier; do not port unless tiered T0/T1 and scheduled.
   - **IGNORE** — touches ported paths but is irrelevant to us (Python-only
     refactors, torch.compile plumbing, platforms we don't target); record
     the reason.
4. **Write the sync report** to `.agents/sync/YYYY-MM-DD-<target7>.md`:
   the three lists above, each entry `vllm#NNNNN <subject> — <disposition,
   one-line reason>`. The report is the reviewable artifact and the work
   queue for step 5.
5. **Port the PORT-NOW queue**, one upstream PR per commit where feasible:
   read the upstream diff (`git show` / `git diff <file-pin>..TARGET -- path`),
   translate it into the mirrored C++ file(s), bump those file headers to
   TARGET, add/adjust tests, append a [parity-ledger.md](parity-ledger.md)
   row per ported PR (upstream ref goes in the "Upstream equivalent" column).
6. **Re-verify.** Regenerate golden dumps at TARGET on an eligible host selected
   by the developer preferences, then run op/behavioral/model suites. If the
   required release-gate hardware is unavailable, retain the gate as `PENDING`
   with an exact handoff. If benchmarks are baselined, the vLLM baseline must
   be re-measured at TARGET before comparing.
   **This step cannot run at TARGET through the committed online-serving harness
   until step 7 has moved the block**, because `online_gate.py` refuses every
   oracle that is not the pinned one on four values it reads from that block
   (measured by [`sync/2026-09-03-e126687-step6.md`](sync/2026-09-03-e126687-step6.md)
   §6.1). The 2026-09-03 cycle resolved that circularity by a developer ruling,
   recorded above and in
   [specs/upstream-pin-advance-e126687.md](specs/upstream-pin-advance-e126687.md)
   §1. The ruling is that cycle's, not a change to this order.
7. **Advance the pin.** Fast-forward the reference checkout to TARGET, update
   the PARITY PIN line above, append an indexed immutable state event linking the
   sync report. A cycle that stalls mid-way keeps the old pin and records
   what's left in the report ("carry-over" section) — the next cycle picks it
   up.

## Rules

- Ledger and inventory updates are part of the cycle, not optional follow-ups.
- An omni-gated number is labeled with BOTH commits and is never cited in a
  vLLM-side parity claim, a binding grid, or a `docs/BENCHMARKS.md` row owned by
  a core-pinned row.
- Advancing the omni pin does not re-open the vLLM-side binding grids PROVIDED
  the omni oracle is installed in its own virtualenv and touches neither
  `${VLLM_SOURCE}` nor the environment the core pin measures itself from. If that
  isolation does not hold, the advance is a core sync cycle and is re-validated
  as one — the dependency tree under the denominator moved.
- Never mix a sync cycle with feature work in the same commit.
- If an upstream change conflicts with a recorded deviation (inventory §9),
  the deviation doc gets updated in the same cycle — deviations must always
  describe the current truth.
- Tooling: `tools/sync/` (roadmap unit P1) automates steps 2–4 (enumerate,
  map to ported files via headers, draft the report). Until it exists, do the
  steps by hand exactly as written.
