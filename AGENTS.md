# AGENTS.md: the rules

This file contains the complete policy for `vllm.cpp`. It is the only file that
every agent loads automatically, so every rule lives here. Files under
`.agents/` are task guides. They explain how to do a specific job. They cannot
add or weaken a rule in this file.

The project mirrors vLLM in C++ without PyTorch or a ggml dependency. vLLM
defines the reference behavior and the performance target.

## Start here

1. Run `scripts/agent-start.py`. Pass `--intent operator|helper|read-only` and
   `--row <ID>` when you know them. Otherwise, relay its welcome and ask what
   work is intended. Follow the printed action, then run the command again.
2. Declare a role. Use `scripts/agent-role.py claim operator` for a multi-step
   integration campaign. Use `claim helper --row <ID>` for one scoped task.
   Use `claim read-only` for inspection. The operator claim records the current
   worktree as a coordinator. Another coordinator does not block the claim.
   Add `--headless` only when the developer explicitly says the run is
   unattended. Never infer this setting.
3. Run `scripts/now.py` to get the live position. Read `.agents/NOW.md` for the
   operator's current gate and next actions. The command output is derived.
   The file is authored and fits on one screen.
4. Read only the claimed row, its spec, its evidence, and the task guide for the
   current job.
5. Run `scripts/agent-preflight.sh` before you edit a file.

Never infer a role, host, permission, or developer preference. Resolve `.env`
and `.agents/developer-preferences.md` from the shared checkout. Ask only for
the one value that the current gate needs. If a value is unavailable, leave its
gate `PENDING`. Never convert a missing value into an assumption. Preferences
control operations only. They cannot reduce a correctness, evidence,
attribution, or testing obligation.

**Create both files on first use.** Neither is tracked, so a fresh checkout has
neither, and `scripts/agent-start.py` reports the absence and routes you to ask.
Ask the developer for the one value the current gate needs. Record an
environment value with `scripts/agent-onboard.py --env-set KEY=VALUE`, which
refuses any key `.env.example` does not declare. Record a preference by copying
`.agents/developer-preferences.example.md` and editing the one entry. Leave
every key you did not ask about empty, because empty means unavailable and its
gate stays `PENDING`. A host name, a share path, or a checkout path written in
a repository document is another developer's resolved value. It is never a
default, and reading one instead of asking is the failure this rule names.

## History is git

The project has no state log. Git is the history, and the history must agree
with the tree.

| Question | Command |
|---|---|
| Did this row already land? | `git log --oneline --grep '<ROW-ID>'` |
| When did this symbol change? | `git log -S'<symbol>' --oneline -- <path>` |
| What happened to this file? | `git log --follow --oneline -- <path>` |
| What is on main that I lack? | `git log --oneline HEAD..origin/main` |
| Why is this line like this? | `git log -L '<start>,<end>:<path>'` |
| What did that commit change? | `git show --stat <sha>` |

The roadmap row states the current position of a row. Git and the row's spec
state how it got there. Before you conclude anything about past work, read the
spec and run `git log -S`. Do not derive the history again.

## Every change starts from an issue

**Do not start work without an open canonical local issue.** Issue authority is
one tracked file under `.agents/issues/`; GitHub is an optional mirror. Create a
local issue before claiming a row or writing code:

```sh
python3 scripts/agent-issue.py create \
  --title "..." --kind bug --problem "..." --row ROW-ID
```

Import an existing GitHub issue with `agent-issue.py import-github N`. Imported
text is quoted as historical evidence; it does not override the local record.
Use `update`, `close`, and the explicit `mirror` command for later changes.
Local validation and writes happen before any remote write.

A row-owned issue lives at
`.agents/issues/<ROW-ID>/ISSUE-GH-<number>.md` or
`.agents/issues/<ROW-ID>/ISSUE-LOCAL-<ULID>.md`; its `Row:` field equals the
directory. A rowless issue lives under `_owed`, and exactly one spec lists its
stable local ID under `## Owed`. Assigning a row moves the file and removes that
spec reference in the same change. `_intake` contains migration-only unavailable
archive residue. It is not valid ownership for new work or a full record.

`scripts/agent-issue-index.py --refresh` validates the local files and renders
the untracked `.agents/issue-index.generated.md` convenience view without a
network call. `scripts/check-agent-record.py` validates every canonical issue
and every issue reference in changed files and commit bodies offline. The four
reference forms are `ISSUE-GH-<number>`, `ISSUE-LOCAL-<ULID>`, `#<number>`, and
`issues/<number>`.

A bug that you find during other work still needs a local issue. Filing the
issue does not defer the fix. File it, fix it in the same flow, reference it in
the commit, and close it. The person who found the bug has the context to fix
it. Traceability is the goal, not another round trip.

**An issue closes when the work lands.** Record dated resolution evidence in
the local file. Mirror afterward only when a GitHub copy is wanted. Do not leave
a fixed issue open for a later sweep.

**An issue the tree falsifies closes with that evidence.** When the file,
checker, or behaviour an issue describes no longer exists, close it locally and
say what falsified it. Do not re-spec it, and do not re-verify it a third time.
Intake without an exit is how 701 open issues accumulated in 24 days, six of
them already answered by the tree (#2298).

This in-flow rule applies to small and clear fixes. Use the normal row, spec,
and fresh-review path for a surprising fix. Use that path when a fix needs its
own spec or changes checker semantics. The in-flow rule is not a bypass.

## Spec before code

A row cannot become `READY` or `ACTIVE` without a committed
`.agents/specs/<slug>.md`. Commit the spec before implementation. Never write
the spec after the implementation. The spec contains scope, upstream anchors,
design, risks, tests, gates, evidence, and stop conditions.

At row claim, before you write the spec, ask the developer whether the spec and
implementation use one pull request or separate pull requests. Recommend one
pull request. Record the answer under `## Git integration` in
`.agents/developer-preferences.md`, and do not ask again for that row. When no
answer is recorded and no split case applies, use one pull request. This default
is repository policy, not an inferred preference.

The single pull request is a recommendation, not an enforcement rule. Use
separate pull requests whenever the developer selects that shape. A split is
often useful for these cases:

- A helper dispatch needs a base-reachable committed spec before the helper can
  start.
- A large campaign benefits from agreement on the scope before implementation
  waves start.
- A change deliberately adds roadmap items, issues, or specs without product
  code.

Commit order still proves that the spec came first when one pull request carries
the spec and implementation. Do not add a checker that selects the pull request
shape.

Before you claim a row, verify the gap against the current code, tests, issues,
pull requests, `NOW.md`, and the owning row. If the work already landed, already
has an owner, or no longer matches its record, reconcile the record first and do
not implement.

When a row reaches `DONE`, add an `## Outcome` section to its spec. Record what
you measured, what you rejected and why, and why each default has its value.
The code and Git history do not record these decisions.

## How we write

Use [the commit and pull request guide](.agents/style/commits.md) for commit
subjects, commit bodies, pull request titles, pull request descriptions, branch
names, changelog entries, and release-note lines. Use [the technical-English
guide](.agents/style/prose.md) for repository documents and all session prose.
Session prose includes progress updates, decisions, questions, and final
reports.

These guides control language and document structure. They cannot add unrelated
project policy or weaken this file. This file takes precedence when a guide
conflicts with a repository convention or rule.

Two of those rules are gated, and `scripts/check-commit-style.py` enforces them.
A commit subject does not end in a period. A commit has an authored body,
because the reader of a commit already has the diff and lacks the reason. One
sentence satisfies the body rule.

Subject length and body wrapping stay guidance in the guide, and no gate
enforces them. Measured over the last 200 non-merge commits, a 72-character
subject limit fails 164 commits and a 72-column body wrap fails 3658 lines. A
gate that fires on ordinary work is the defect, not the discipline.

The guides bind new prose. Rewriting an existing file to satisfy a style rule is
out of scope unless a row asks for that rewrite.

## How work gets done

Work is delegated. Another person or agent reviews every implementation, and the
operator verifies the result. This sequence is the method, not a suggestion:

1. A **fresh implementer** works from the committed spec. The implementer ports
   or writes the smallest test that fails for the intended reason. They capture
   the red result, make the minimum complete change, and get focused green.
   Then they run the full gate.
2. A **fresh reviewer** reviews the immutable head. The reviewer cannot be the
   agent that wrote the code. They inspect the change statically and mutate each
   claimed guarantee in a scratch copy. This mutation proves that the tests
   detect the defect. Mutate the guarantee. Do not only read it. The reviewer
   restores the tree byte-for-byte after each mutation.
3. A **fresh implementer** repairs each finding. The coordinating session never
   repairs a finding. Repeat the focused gate, full gate, and fresh scoped
   review until the result is `PASS`. Attempt budgets control scheduling. They
   never stop a correctable finding. Only explicit developer direction or a
   precise external blocker stops the loop.
4. The **operator reruns the row's gate itself**. An implementer or reviewer
   report is an input, never a gate result.

Every delegated task states the goal, context, exact scope, exclusions,
constraints, completion condition, required evidence, authority, output
contract, and stop conditions. Return `NEEDS_CONTEXT` when binding context is
missing. Do not guess. Return `NEEDS_DECISION` for a material disagreement
rather than changing the scope silently. Use the versioned contracts in
[`.agents/prompts/`](.agents/prompts/).

The operator is a **coordinator**. It holds the plan and the graphics processing
unit (GPU). It merges reviewed pull requests and dispatches agents into separate
worktrees. It does not write an implementation that needs independent review.

**More than one operator can run at the same time.**
`scripts/agent-role.py claim operator` records the coordinator and never refuses
because another operator exists. The `show` command lists the other operators.

**Never force-push `main`.** Nobody can use `--force` or `--force-with-lease` on
`main`. A plain `git push` rejects a non-fast-forward update. Git provides the
coordination lock. After a rejected push, fetch, merge again, rerun the gate,
and push again. Never force the update.

## vLLM is the reference

**Mirror vLLM.** When vLLM defines behavior, mirror every applicable mode,
default, error, and edge case. Escalate only a genuine product decision. Never ask how a
mirrored feature must behave.

**Pin vLLM.** Compare against the pinned oracle in
[`.agents/upstream-sync.md`](.agents/upstream-sync.md). Advance the pin only
after you reconcile every affected row and gate. An oracle is only gateable once
it demonstrably builds and runs the model. Constructing a config proves
nothing.

**Run the oracle and read its source.** Check every change in two ways. Run the
pinned vLLM on the identical workload. Read the matching upstream path. Ground
the conclusion in the complete executing chain. This chain can include
FlashInfer, CUTLASS, cuBLASLt, DeepGEMM, torch or Inductor, generated code, and
local dispatch. Cite the `file:line` that you ported. Dump the generated kernel
before you call a lever unreachable. Record scratch implementations as such in
the porting inventory.

**Inherit vLLM defaults.** vLLM resolves one model dtype, and every layer
inherits it. An `f32` value is a rare, annotated exception. Mirror this
polarity. Add a one-line reason beside a model-path buffer or GEMM output that
names `f32`. **A token gate cannot detect a dtype that is too wide.** The tokens
still match, and the goldens still pass, although the path moves twice the
bytes. Compare the memory format with the oracle as
[`.agents/porting.md`](.agents/porting.md) requires.

**Port the upstream tests in the same change.** Preserve parameters, modes,
fixtures, tolerances, failure cases, and the upstream revision anchor. Document
only an unavoidable adaptation of the harness.

**Trace both sides with the same tool.** Use an identical workload before any
throughput comparison. Source inspection identifies candidates. Matching traces
identify the executed path. A GEMM or GEMV invocation-parity claim proves the
output dtype, compute type, scale type, entry point, algorithm policy, and
resolved template dtypes in the same tool.

## When vLLM has no implementation

vLLM, including `vllm-project/vllm-omni`, is the primary reference. It is the
only reference wherever it implements the behavior. Where it implements nothing, the work is
still not ungated. Run it against a **secondary oracle**. A secondary oracle is valid
only when it appears in this table and has a recorded pin:

<!-- oracle-registry:begin -->

| Oracle | Registry id | Reach for it when |
|---|---|---|
| vLLM | `vllm` | always, as the primary wherever it implements the behavior |
| vLLM-Omni | `vllm-omni` | diffusion, TTS, and omni-only architectures that vLLM does not register |
| HuggingFace `transformers` | `transformers` | a model, processor, or tokenizer reference implementation that vLLM mirrors |
| DeepSeek `DeepSeek-V4-Flash-Vision-Exp` | `deepseek-v4-vision` | the model-author prompt encoder, image processor, ViT, aligner and image-span visibility, which vLLM and Transformers do not implement |
| `diffusers` | `diffusers` | schedulers, VAEs, and diffusion pipelines |
| Lightricks `LTX-2` | `ltx-2` | the LTX-2.5 architecture and pipeline recipes, from the model author's own runtime, for the generations and defaults vLLM-Omni's `ltx2` registration does not reach |
| SGLang | `sglang` | a model or serving path that SGLang implements and vLLM does not |
| SGLang-Omni | `sglang-omni` | omni, speech, TTS, and music models served by SGLang's pipeline runtime, in a third repository that is not SGLang |
| llama.cpp | `llama-cpp` | CPU and GGUF k-quant floors |
| `unslothai/llama.cpp` fork | `llama-cpp-unsloth` | the sub-IQ1_S quant encodings (IQ1_XS, IQ1_XXS, IQ1_XXXS) that no upstream llama.cpp defines, and that a published Qwen3.8-2.4T checkpoint stores its experts in |
| llama.cpp release `b10766` | `llama-cpp-dsv4vision` | the `deepseek4` VISION variant: the `deepseek4v` clip projector, its mmproj container, the `exp_probs_b_vl` media routing bias and the non-causal image-span window, none of which the stock `llama-cpp` pin at `b10451` contains |
| `ggml-org/llama.cpp` PR #27742 | `llama-cpp-qwen4exp` | the `qwen4exp` architecture, its GGUF conversion and its graph, which no released llama.cpp defines, so the stock `llama-cpp` pin cannot supply a denominator for it |
| `ggml-org/llama.cpp` PR #27752 | `llama-cpp-glm5next` | the `glm5next` TEXT architecture, its GGUF conversion and its graph, which no released llama.cpp defines either; it is the head whose architecture string matches the published artifact, and it carries no vision tower |
| turboderp-org `exllamav3` | `exllamav3` | the EXL3 trellis quantization format and its kernels, and the DeepSeek-V4 support the pinned HEAD carries, which neither vLLM nor vLLM-Omni implements |
| `vllm-project/vllm-gguf-plugin` | `vllm-gguf-plugin` | GGUF, which the pinned vLLM carries no in-tree support for: it was deprecated in `vllm#39583` and moved to this first-party repository, so a GGUF path it serves is a path the PRIMARY oracle serves. `gateable = yes` since 2026-09-10 (emitted byte-identical tokens on gfx1151); the CUDA `sm_110` forward gap ([#2624](https://github.com/mudler/vllm.cpp/issues/2624)) is a vLLM wheel arch issue, not a plugin defect |
| Tenstorrent tt-forge | `tt-forge` | Tenstorrent execution where vLLM has no implementation; vLLM's TT backend now exists as [`vllm-tt-plugin`](.agents/oracles/vllm-tt-plugin.md), which takes precedence where it serves |
| vLLM TT plugin | `vllm-tt-plugin` | the qwen35 and other TT-prefixed architectures on Tenstorrent mesh devices, from HF weights — vLLM's own TT platform plugin (blog 2026-09-07); NOT a GGUF-decode reference |
| Aleph Alpha inference plugin | `aleph-alpha-inference` | the kolibri1 (Kolibri1ForCausalLM) architecture, from HF weights — the model-author's own vLLM plugin (upstream vLLM implements nothing for it); [`aleph-alpha-inference.md`](.agents/oracles/aleph-alpha-inference.md) |
| `ddickmann/vllm-factory` | `vllm-factory` | GLiNER2.5 on vLLM (DeBERTa v2 encoder with disentangled attention + GLiNER2 pooler head + IO processor), which vLLM does not implement at the pin; no DeBERTa, no disentangled attention, and no GLiNER pooler exists in-tree or upstream |
| `trycua/cua` | `cua` | the cua-s1-forms TinyTransformerScorer (byte-level 2-layer transformer encoder + cross-attention scoring head) for GUI form-filling decisions, which neither vLLM nor vLLM-Omni registers |
| `Contrastive-LM/CLM` | `clm` | the CLM bi-encoder decision model (Qwen3-8B pooling + state/action projection heads), which neither vLLM nor vLLM-Omni registers |

<!-- oracle-registry:end -->

**Pin every oracle.** Each oracle has its own file under
[`.agents/oracles/<id>.md`](.agents/oracles/). The file records the revision,
measurement date, gateability, and evidence. An unpinned upstream is a moving
target, not an oracle. Its measurements are not reproducible. The vLLM parity
pin remains in [`.agents/upstream-sync.md`](.agents/upstream-sync.md). The vLLM
oracle file points there instead of copying the value. Use one file per oracle
and read the files with a glob. Never require every change to write one shared
table.

**A secondary oracle answers one question:** what correct output looks like on
a path that vLLM cannot run. It never outranks vLLM, and it never becomes the mirror
source. Mirror vLLM behavior, defaults, structure, and naming wherever vLLM
defines them. "SGLang does it differently" is never on its own a reason to diverge.
When vLLM implements the path, reconcile the row onto vLLM and record the
change in the spec.

**Measure gateability.** The primary rule applies to every oracle. The oracle
must demonstrably build and run the model. Until then, its file records
`gateable = no` and names the issue that owes the measurement. This record makes
the ungateable lane visible debt.

## Gates

Correctness always comes first. Establish the declared token-exact gate before
you accept a performance result. Use an explicitly ratified distributional gate
only when the oracle's greedy decode is non-deterministic. Never trade
correctness for throughput.

Use the pinned oracle on each side. Use identical model artifacts, prompts,
token counts, batching, concurrency, and sampling. Use vLLM's production
configuration as the denominator. Never use `--enforce-eager` as the
denominator.

Record values and ratios for every required throughput, latency, and memory
axis. An axis below its floor remains an open gap. Record the exact build and
run recipe, revisions, model hashes, environment, and contention state.
Reproduce the result on an idle host with a same-binary A/B test before you
accept it.

**Never declare a ceiling.** An apparent same-architecture performance limit is
an unresolved implementation difference. Keep the gap open and name the next
traceable hypothesis.

Report exactly one result for each applicable rule. The result is satisfied, narrowly
waived, pending a named external authority or resource, or failing. A permanent
report-only state is not a result.

## Shared seams

A capability is not done until the shared surface can reach it.

- Route model fusion through `vt::FusedChain`.
- Route mergeable multilayer perceptron (MLP) projections through
  `layers::MlpGateUpMethodBase` and `vt::MergedGemmGroup`.
- Route decode through `ModelRegistry::Forward`, `dense_attn::AttnBlock`, and
  on-device sampling.
- Expose every shipped capability through `include/vllm.h`. Examples and
  servers are thin clients of this application binary interface (ABI). They
  never include internal headers.

Extend a shared seam when it cannot represent the upstream behavior. Otherwise,
record one exact tracked exception. Never write a parallel path by hand.

Add new files for new hardware, architectures, and models. Mirror the vLLM file
structure.

A model port includes the **quantized arms, not only bf16**. GGUF k-quants are a
standing requirement. They are not a choice for each model. Most users run the
quantized arms, and a quant-matched llama.cpp comparison needs them. Refuse an
unimplemented arm with a message that names the missing part. Record the arm as
owed. Never leave the missing path to be discovered later. Use
[`.agents/porting-a-model.md`](.agents/porting-a-model.md) as the checklist.

**Say which weights, and from where.** Every ported model documents the
checkpoints it was built and gated against in [`docs/USAGE.md`](docs/USAGE.md),
in the same change that makes the capability reachable: file name, size, and the
exact HuggingFace repo *and revision*, grouped by arm, with a sha256 for a
quantized artifact and the refused arms named beside them. A repo id alone is not
a pin, because checkpoints get re-quantized in place under an unchanged name.
Weights are the one part of a port a reader cannot infer from the code, so code
shipped without them is something nobody can feed.

## Nothing lands dead

A shared seam says where a capability routes. This section says whether anything
reaches it. The two failures are different, and this tree has carried the second
one green.

What lands is reachable from a production entry point at its own merge commit:
`include/vllm.h`, the loader, `ModelRegistry::Forward`, or a registered server
or command-line path on its default configuration. An example's internals are
not one, and a test is not one. The smallest failing test enters the new code
through that entry point. A unit test that constructs the type by hand proves
that the class works, never that anything reaches it.

A staged slice may land unreached only when the commit body and the pull request
body name what is unreached, the row ID that owns the wiring, and the issue that
tracks it, and the row's spec lists it under `## Owed`. There is no registry, and
silence is not an exception.

The fresh reviewer mutates for this. Delete the production call site in a scratch
copy and rerun the focused gate. A gate that stays green without the call site
measures a class, not a capability.

Method, the shapes dead code takes, and why no checker enforces this:
[`.agents/reachability.md`](.agents/reachability.md).

## Records

Every inventory item has a stable ID. It records the upstream source, local
anchor, tests and evidence, spec, lifecycle state, owner, and issue in the
correct matrix. When a lifecycle state changes, update the roadmap row and its
owning matrix row in the same change.

For a concurrent edit to a keyed record, take the complete target-branch
version. Apply the scoped edit again. Verify that unrelated keys remain
byte-for-byte equal. Never accept an automatic three-way merge of a keyed record.

**Do not create a surface that every pull request must write.** If N concurrent
pull requests edit file F, that file is a lock. A record surface can have only
one of these shapes:

- One file per row, read with a glob.
- A value that is derived at read time and is not stored.

Rewrite every other record surface into one of these shapes.

**An append-only file with `merge=union` is NOT one of them.** That shape was
admissible here until 2026-08-29 and it never worked, because GitHub does not run
`.gitattributes` merge drivers. The driver resolves the collision on your machine
and the forge conflicts anyway, so the file reads as safe locally and is a lock
in the only place that decides whether work can land. The retired
`.agents/issue-index.md` (now in [`.agents/completed/`](.agents/completed/))
was a correct implementation of that shape and still became the repository's
dominant conflict source: 115 of 200 commits wrote it, and 16 of 21 open pull
requests conflicted on it, four of them on that file alone (#883, #2290).

A conflicted pull request is worse than a failing one. GitHub never schedules CI
for it, so it carries zero check-runs and reads as unverified rather than red
(#2248). Do not reach for `merge=union` to make a shared file safe. Derive it, or
give it one file per writer.

Two rules follow. **Limit an entry, not a shared file.** A shared-file budget
forces each addition to remove another entry. Merging two such edits cleanly is worse than
conflicting, because it applies both removals. **Never store a measurement of one file inside another file.** A
number that changes after each edit couples every pull request to lines that it
does not own.

A gate often creates the lock. If a checker requires every change to edit one
shared file, the checker is defective. Move the obligation to a per-row surface.
Do not delete the obligation.

Move superseded detail into `.agents/completed/`. Keep its links and provenance.
Never delete evidence to reduce context.

## Public documents

Each public document has one purpose and one trigger. These documents are
projections, not narratives. Each fact lives in one document.

| Surface | Changes when |
|---|---|
| `docs/BENCHMARKS.md` | a public benchmark ID is added, removed, or changes disposition |
| `docs/FEATURES.md` | a feature, model, backend, or quantization surface changes |
| `docs/USAGE.md` | a command, C API, config key, install step, or workflow changes |
| `README.md` | a user-visible headline, positioning, or quick start changes |
| the moved row spec's `## Now` | a row changes lifecycle state |

Editing `src/`, `include/`, or `tests/` on its own owes none of these. A
lifecycle change owes only the moved row spec's `## Now`; Git and the row's
records carry detailed status and history. Publishing a benchmark owns one
`docs/benchmarks/<benchmark-id>.md` detail file and its index row.
`.agents/NOW.md` is authored only at operator cadence and is never a per-row
lifecycle write.

## Work on a GPU happens inside a lease

The shared GPUs are managed by
[resource-controller](https://github.com/mudler/resource-controller), whose
client is `rc`. **`dgx:gpu0`, `thor:gpu0` and `orin:gpu0` are the fleet devices.
Claim a fleet device with `rc run` or `rc hold` before any GPU work, and never
`ssh` to one to run work directly.** The lease is the required path to those
boxes, and it replaces the `flock` file mutex as the default. The three names
are written here so that membership stays checkable when the client is not at
hand.

**The condition is the device, not the shell you are typing in.** A missing
local `rc`, a controller that does not answer, and a refused authentication are
each a reason to get the client or to report the controller down. None of them
turns a fleet device into a box you may reach by `ssh` plus `flock`, because the
fleet cannot see that mutex. `thor:gpu0` read `unknown (no contact 1m0s)` on
2026-08-17, so a controller that loses contact is a live state and not a
hypothetical. The list of three is a lower bound and never an upper one. A
device that `rc devices` reports is a fleet device even when this file has not
caught up with the fleet.

**On a GPU that is not a fleet device, take the file mutex
`${GPU_LOCK:-$HOME/gpu.lock}`.** The rule is conditional because the hosts are
not identical, and a reader on a personal machine still needs an instruction.
Where both apply, the mutex runs inside the lease and never instead of one. The
lease decides who gets the box. The mutex serialises the work of whoever holds
it.

**Two mutexes that do not exclude each other are worse than one, and this
already cost a measurement.** On 2026-08-17 one session took the file mutex over
`ssh` while another session held the same box through `rc`. Neither mutex
excluded the other, and `.agents/specs/minimax-music3.md` §13.10 retains a whole
speed axis as VOID because of it. That is the #777 failure again, in which this
repository carried two GPU mutexes and neither serialised the other. A bypass
also makes the fleet report the box free while somebody is on it.

**A lease runs as root, and the worker provisions itself.** The leased worker
reads and writes the shared `/workspace`, and on `dgx:gpu0` it is an Ubuntu 24.04
container carrying `git`, `curl`, `wget`, `gcc`, `cmake`, `ninja`, `python3` and
`pip`. It installs what it lacks and it compiles in place. On 2026-08-18 one job
apt-installed `cuda-nvcc-13-0` and built this tree for `sm_121a`
([#1213](https://github.com/mudler/vllm.cpp/issues/1213)).

**Four limits are still real, and they shape staging.** No CUDA toolkit is
preinstalled, so a job that compiles CUDA installs one first. A global install
leaks into the next job, so put project dependencies in a virtual environment
under `/workspace`. `/workspace` is CIFS and holds no symlink, so build in `/tmp`
and copy out with `cp -rL`. Unconstrained parallelism has OOM-rebooted this box,
so use `-j 4`. The HOST `dgx.casa` has no egress to `github.com` while the
container does, so name the side you mean.
[`.agents/environment.md`](.agents/environment.md) carries the fleet, the
measurement, and the procedure.

## Work happens in a worktree

**Do every unit of work in its own linked worktree and task branch.** A unit of
work is one issue's change together with the records that change invalidates. A
feature, a fix, a policy change, a document, and a one-line gate repair are each
a unit. A claimed row uses `row/<ID>`. Pin the base SHA when you create the
worktree.

**A record edit rides in the pull request whose change made the record stale.**
It is not a unit of work by itself. A record-only pull request is still correct
when the record is the work: a stale row, a corrected pin, a newly filed gap. It
is wrong when it only restates what just landed, because that costs a branch, a
gate run, and a fresh review to say something the landing change already knew.
This narrows what counts as a unit. It never licenses bundling unrelated work
into one branch.

Keep the shared checkout on a clean `main`. **Never use it as a work surface.**
Other worktrees branch from it, so it must remain current and safe. Never edit,
commit, or stash in the shared checkout.

Remove the worktree and delete its branch when the work merges, closes, or is
abandoned. A worktree is a complete checkout. Stale worktrees fill the disk and
can make gates fail when temporary space runs out.

## Landing work

Move work to `main` from its task branch, never from the shared checkout. A
helper opens a reviewed `row/<ID>` pull request. An operator with recorded merge
authority can merge the branch locally with a commit that names it. This local
merge keeps an in-flow repair to one step. The task branch still makes the
change visible, reversible, and attributable.

Run the applicable gate before every push. Chain the successful gate directly
to the exact-SHA push. **Never force-push `main`, and never add a force option to
a script that can target it.** The scope is `main`, exactly as in §"How work gets
done" -- a row branch you own is yours to rebase and force-push, and the
`merge=union` retirement makes that the ordinary way to clear a stale index
conflict. Read this sentence as a blanket ban and ordinary work stalls (#1808).
A rejected push on `main` protects another merge. Fetch, merge again, rerun the gate, and
push again. Hooks are bypassable convenience, not evidence. If you cannot query
the remote, report `REMOTE_UNVERIFIED`. Unknown is not absence or success, and
it does not authorize cleanup.

Merge each verified pull request in the current session. Close an obsolete pull
request and record the reason. Never end a session with a verified, unmerged
pull request.

**The pull request body is the landed commit message.** The repository sets
`squash_merge_commit_message = PR_BODY`, so write the body to the same standard
as a commit message and end it with the trailer block below. This setting is
part of the contract, not a convenience. Under the previous `COMMIT_MESSAGES`
value GitHub wrote a `---------` separator between the concatenated commit
messages, and on a multi-commit squash the last one fell between the trailer
block and the `Co-authored-by:` line GitHub appends. That orphaned the block,
and the gate reported trailers the commit plainly carried as missing.

No gate can hold that setting, because reading it needs a network call and no
checker here may make one. If the landed commits start failing the trailer gate
again, read the setting first. `tests/scripts/test_check_commit_trailers.py`
pins both squash shapes, so the difference between them is executable.

**Read the body before you merge it.** The CI job that holds a body to this
contract runs in a queue and can be outrun. On #1257 it was still `pending` when
the merge went ahead, so nothing ever read the body, and its malformed
attribution value became a message on `main` that cannot be repaired (#1262,
#1263). Run the same check yourself, on the bytes that are about to become the
commit:

```sh
python3 scripts/agent-pr-body.py --pr <N>
```

It exits 0 when the body will land clean, 1 when the body was read and fails the
contract, and 3 when it could not be read at all. A 3 is `REMOTE_UNVERIFIED` and
is never a pass. The command is a belt to the CI guard's braces and not a
replacement for it: the forge reads the body again from its own event payload,
which is what catches an edit made after you looked.

Every commit contains a bare `FOLLOWING_AGENTS_PROTOCOL` paragraph and these
trailers:

```text
Following-Agents-Protocol: true
AI-Assisted: true
Assisted-by: AGENT:MODEL [TOOL]
```

AI tools never add `Signed-off-by` or `Co-Authored-By`. The human submitter owns
and reviews the change.

This prohibition stops an AI from claiming authorship. It does not stop the
forge from recording the submitter. GitHub can add a `Co-authored-by` trailer
for the account that opens a pull request. The accepted form uses an
`@users.noreply.github.com` address. This trailer records the submitter, even
when the account is a bot. `AI-Assisted` and `Assisted-by` record AI involvement
and are always required. `Signed-off-by` has no exception because it is a legal
assertion, not attribution.

Classify policy, checker, document, script, test, continuous integration (CI),
generated, and product paths explicitly. Never hide mutable files behind a
general directory exemption. The project has no line budget. It retired the
per-class limits on 10 August 2026. Nine of the previous 22 merged pull requests
exceeded the product limit. Tests were one-third to one-half of each large
change. The gate therefore failed normal work, and it charged red-first
mutation tests against the same allowance as kernel code. Size is a review decision. Split a change when parts
help the reviewer, not when a counter requires it.

## Changing the rules or a checker

A checker's message defines what it enforces. This file states the rule in
prose. No gate checks whether the two descriptions agree, and that is deliberate.
Keeping two descriptions in sync is the failure mode that this protocol was
built to remove.

A semantic checker change needs a spec, a red-before test or mutation, and
green-after evidence. Never make a red gate green by deleting an assertion or
widening its scope.

The project has no waiver registry. An exception registry is a state log, and
this protocol has no state log. A commit that needs an exception argues for it
in its own message. The reason then stays with the diff, author, and date. It
cannot drift from the tree because it is part of Git history. Use
`git log --grep` to find it. An exception is visible debt, not success. A
reviewer who rejects the reason does not merge the change.

## Task guides

Read the guide for the current job.

| Doing this | Read |
|---|---|
| Porting a MODEL (the coverage checklist) | [`.agents/porting-a-model.md`](.agents/porting-a-model.md) |
| Porting a model, kernel, or feature from vLLM | [`.agents/porting.md`](.agents/porting.md) |
| Running gates, proving correctness, reviewing | [`.agents/verification.md`](.agents/verification.md) |
| Proving a change is reached | [`.agents/reachability.md`](.agents/reachability.md) |
| Measuring performance | [`.agents/benchmarking.md`](.agents/benchmarking.md) |
| Fixing a bug | [`.agents/bugfixing.md`](.agents/bugfixing.md) |
| Working on a specific host or GPU | [`.agents/environment.md`](.agents/environment.md) |
| Coordinating parallel work | [`.agents/workflow.md`](.agents/workflow.md) |
| Chasing a parity lever | [`.agents/parity-lever-protocol.md`](.agents/parity-lever-protocol.md) |
| Syncing the vLLM pin | [`.agents/upstream-sync.md`](.agents/upstream-sync.md) |

## Commands

```sh
scripts/agent-start.py                          # always first
python3 scripts/agent-role.py show
scripts/agent-preflight.sh                      # before edits
scripts/agent-preflight.sh --staged             # before commit
python3 scripts/agent-ready.py                  # before remote handoff
python3 scripts/agent-pr-body.py --pr <N>      # before merging: the body IS the message
python3 scripts/agent-integration.py --base origin/main
rc devices                                      # the GPU fleet, and who holds it
```

Never push, merge, manage services, use external compute, or download large
assets without authority recorded in developer preferences or given for the current
task.
