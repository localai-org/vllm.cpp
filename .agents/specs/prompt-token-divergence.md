# SPEC-PROMPT-TOKEN-DIVERGENCE — the served prompt-token count, and what it is not

Issue: [#2948](https://github.com/mudler/vllm.cpp/issues/2948)
Row: `SPEC-PROMPT-TOKEN-DIVERGENCE`
State: `READY` (diagnosis committed; the defect is not yet located)

**The original issue no longer resolves.** This row was filed as
[#1355](https://github.com/mudler/vllm.cpp/issues/1355), which now returns
`Could not resolve to an issue`. #2948 refiles it with the description recovered
from `.agents/completed/issue-index.md:448` and carries the grounding below.
`.agents/benchmark-record.md:25533,25560` still cite the dead number.

**The ID collides with a prefix.** `SPEC-` is this tree's SPECULATIVE-DECODING
row prefix (`SPEC-MTP-K-GT-1`, `SPEC-DSPARK-BLOCK-SIZE-GUARD`, `SPEC-DFLASH2`),
so this ID reads as a spec-decode row and is not one. It is recorded here rather
than renamed, because the ID is already committed, indexed and referenced.

## Now

The benchmark campaign of 19 August 2026 measured our server at **5,942** prompt
tokens where the pinned vLLM oracle measured **6,144** for a corpus both arms
took from byte-identical invocations of one `vllm bench serve` client. This spec
records what that divergence **is not**, with reproductions, and names the one
probe that can still locate it. It changes no product code.

## Scope

In scope: attributing the count. Out of scope: the speed verdict itself, and
with it every ours-over-vLLM ratio. What this spec corrects is OUR OWN arm's
numerator, on our own arm only (`## Consequence for the campaign's numbers`).

## Upstream anchors (pin `5559679229`, `/home/mudler/_git/vllm`)

- `vllm/benchmarks/datasets/datasets.py:557` `RandomDataset` — a prompt is a
  contiguous slice of vocabulary ids, DECODED to text and then re-encoded.
- `vllm/benchmarks/datasets/datasets.py:495-549` `gen_prompt_decode_to_target_len`
  — the decode/re-encode loop that pads or truncates until
  `len(tokenizer.encode(prompt, add_special_tokens=False)) == target`, retrying
  ten times and reporting the residual as `token_mismatch`.
- `vllm/benchmarks/datasets/utils.py:41` `get_sampling_params`.
- `vllm/benchmarks/serve.py:74-144` `_align_prompts_to_server_tokenizer` — posts
  the FIRST prompt to the server's `/tokenize`, and on a disagreement re-tokenizes
  every prompt through the server and truncates it to `prompt_len` ids.
- `vllm/benchmarks/lib/endpoint_request_func.py:245-247` — `output.prompt_len`
  starts at the dataset's value and is OVERWRITTEN by `usage.prompt_tokens` from a
  usage-only SSE frame.
- `vllm/benchmarks/serve.py:606,734,1222` — `total_input`, hence
  `total_token_throughput`, and `input_lens`.

**vLLM reporting exactly 1024 is not the surprising half.** The dataset
calibrates each prompt string until the client's own tokenizer encodes it to
exactly `--random-input-len` tokens, so a server whose tokenizer agrees with the
client's reports 1024 by construction. Regenerated locally at the pin: 48/48
prompts, `token_mismatch == 0`.

## What was measured

Corpus regenerated from `--seed 0`, `--random-input-len 1024`,
`--random-range-ratio 0`, tokenizer `/mnt/nas_share/rc/ckpt/qwen3.8-27b-hf`
(`tokenizer.json` sha256 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`,
byte-identical to the `ckpt-sha256.txt` manifest the campaign wrote from the
STAGED copy the server read).

1. **The corpus is the campaign's corpus.** State the measurement precisely,
   because the shortfall did NOT reproduce and cannot reproduce here: our
   tokenizer encodes all 48 regenerated prompts to 1024, so they come short
   nowhere on x86-64. What agrees is the INDEX SET. The 19 regenerated prompts
   that carry 74-150 combining marks are exactly the 19 indices the campaign's
   `bench-20260819T035148Z/c8-r1.json` reports short — 0, 1, 9, 10, 11, 13, 14,
   15, 16, 19, 21, 22, 25, 28, 29, 30, 33, 45, 47. Prompts 0-5 of the c8 leg are
   byte-identical to the c1 leg because `numpy` short-circuits a zero-width
   `integers()` draw and consumes no state, so both legs share one offset
   stream. A 48-bit index agreement between an independently drawn corpus and
   the served counts is not a coincidence: these are the campaign's prompts.
2. **Our tokenizer sources are byte-exact ON X86-64.**
   `vllm::tok::Tokenizer::FromHfJson` + `Encode` at this revision returns the
   SAME IDS, not merely the same count, as HF `tokenizers` for all 48 prompts. The tokenizer sources are unchanged
   between the campaign's build `1dac4f9a7` and `origin/main`
   (`git diff 1dac4f9a7 origin/main -- src/vllm/tokenizer include/vllm/tokenizer`
   is empty), so the comparison is against the code that ran.
3. **The transport PRIMITIVES are not it.** The same result over a real socket
   through `third_party/httplib` + `nlohmann::json::parse` +
   `EncodeWithSpecialTokens`, with the body sent three ways: `ensure_ascii`
   escapes (what `aiohttp` sends), raw UTF-8, and `Transfer-Encoding: chunked`.
   48/48 at 1024. **This was a standalone harness and NOT the production request
   path.** It never ran `from_json(const nlohmann::json&, CompletionRequest&)`,
   which is where the `prompt` string is actually extracted
   (`protocol.cpp:330`), and it never ran `serving_completion.cpp`. That
   segment is x86-testable and is UNTESTED, and it sits inside the gap this
   elimination would otherwise be read as closing.
4. **Not a build-flag or memory defect.** `-O0`, `-O3`, `-O3 -funsigned-char`
   (the aarch64 `char` signedness, which is the obvious portability suspect) and
   `-fsanitize=address,undefined` all return 1024 on all 48 with no diagnostic.
5. **Not any pre-tokenizer or BPE option TRIED.** Every wrong `SplitPattern`
   (`kQwen2Classic`, `kLlama3`, `kTekken`, `kGpt4o`, `kGpt2`) and every
   misclassification of `\p{M}` produces MORE tokens (1117-1306), never fewer.
   **That direction is a property of the cases tried, not a law**, and this spec
   claims only the former. A COARSER split PERMITS merges a finer split forbids,
   so it can lower a count rather than raise it — that is the mechanism behind
   `\p{N}` against Qwen's `\p{N}{1,3}` — and two measured cases already sit
   below 1024: `ignore_merges=true` gives 1022, and the fresh review measured a
   Split pre-tokenizer with `behavior: "Removed"` giving 0. The reason every
   pattern tried went UP is narrower and sufficient: the merge table was learned
   under one canonical split, so a deviation from it costs tokens. Mechanisms
   that can lower a count are therefore NOT excluded here, and three are named
   in `## Diagnosis`.
6. **Not the client's alignment pass, in EITHER direction.** `_fix_one`
   (`serve.py:132-138`) returns the request UNCHANGED when
   `len(tokens) <= req.prompt_len`, and truncates only above it. No prompt in
   this corpus exceeds 1024 server ids — the anomaly is 915, which is below —
   so the bytes on the wire are the same whether the pass ran or not, and
   whether or not the warning that says it ran was truncated
   (`## Consequence for the campaign's numbers`). Simulated against our
   tokenizer under every wrong pattern it likewise always predicts 1024, never
   915.
7. **Not Unicode normalisation.** The checkpoint declares `normalizer: NFC` and
   `tokenizer.cpp:527-530` records the deviation that we accept it and do not
   apply it. Every one of the 48 prompts is ALREADY NFC, so the deviation is
   inert here. (It is still owed: a client that sends non-NFC text gets a
   different tokenization from HF. The tokenizer parity goldens feed it no
   non-NFC text either, so nothing in this tree exercises the deviation at all.
   Not this bug.)

The divergence correlates perfectly with combining marks: all 19 prompts at the
campaign's short indices carry 74-150 characters in `Mn`/`Mc`, and all 29 at its
exact indices carry 0-2. Equivalently
they are the prompts drawn from the multilingual region of the vocabulary
(bytes-per-character 1.4-2.2), against pure-ASCII (1.0) and pure-CJK (3.0)
neighbours that are exact. No transformation tried — deleting or replacing any
category, page or codepoint range, prefix truncation, NFC/NFD/NFKC — lands on the
served numbers.

## Diagnosis

**This is not a harness artifact, not a corpus property, and not a defect in the
tokenizer SOURCES AS EXERCISED ON X86-64.** All three were candidate
explanations and all three are falsified above. That qualifier is load-bearing:
what item 2 and item 4 establish is that these sources produce the reference ids
on x86-64 at four optimisation settings and under ASan and UBSan. The server
binary that reported the anomaly is aarch64, and the probe below exists to catch
a code-generation difference in exactly that class. What remains is that the
server binary (`bin/vllm-server`, sha256 `7d0c3cafb224…`, aarch64, built from
`1dac4f9a7`) reported a count that the identical source, the identical
`tokenizer.json` and the identical prompt bytes do not produce on x86-64.

**That sha256 is the run's, and the conflicting value beside the artifacts is
already adjudicated.** `NOTES.txt` in the evidence directory records
`ab0b9a1e6144…`, which is the binary of the PREVIOUS day's run
(`out/bench-20260818T213222Z/job.log:17,22`). Three artifacts of the run this
spec diagnoses give `7d0c3cafb224…` instead: `sha256sum` over
`/mnt/nas_share/rc/q38bf16/bin/vllm-server`, `out/RESULT.txt:5`, and
`out/bench-20260819T035148Z/job.log:17,18`, which asserted `WANT_BIN_SHA256`
equal to `GOT_BIN_SHA256` at launch.
`.agents/benchmark-record.md:24510-24514` settled this exact conflict in favour
of the executing artifact, so it is not re-opened here. The hash is not
load-bearing for the diagnosis — both builds carry
`WANT_SHA=1dac4f9a7…` and the identical `SRC_SHA256=c74c45d1…`, two
non-reproducible builds of one tree, and the argument rests on the source and on
aarch64 — but it is load-bearing for reproducing the run.

**The conclusion rests on the data path, not on an enumeration.** No list of
things that can go wrong inside a tokenizer is needed, and none is offered,
because a missed entry would not weaken this:

- `usage.prompt_tokens` is `response.prompt_token_ids.size()`
  (`serving_completion.cpp:78`, the streaming path `vllm bench serve` drives;
  `res.prompt_token_ids.size()` at `:283` is its non-streaming sibling);
- that vector is `RequestOutput::prompt_token_ids`
  (`output_processor.cpp:284`), copied from the request state the engine
  recorded at admission (`output_processor.cpp:174`);
- and that state carries the ids `InputProcessor` produced by calling
  `tokenizer_.EncodeWithSpecialTokens(prompt)` (`input_processor.cpp:259-260`).

The reported number IS the id vector the model consumed. 915 ids where the
client calibrated the prompt to 1024 therefore means the model consumed a
different prompt, WHATEVER the mechanism, and that holds against a mechanism
nobody has listed. **The model saw a different prompt.** If the mechanism is
confirmed, it reaches every token-exact gate driven over HTTP with non-ASCII
prompts, not only benchmarking.

**Three mechanisms that nothing above excludes**, recorded so that the search is
not read as narrower than it is:

- **A merge-table or vocabulary ORDER difference.** A hash order or a
  comparator that resolves differently on aarch64 can change which merges rank
  first and so LOWER a count with no text lost at all. This is the aarch64
  hypothesis itself, and the x86-64 id-for-id agreement in item 2 cannot see it.
- **A UTF-8 decode truncation inside `Encode`**, which loses text rather than
  re-ranking merges.
- **The `/v1/completions` request-parse segment**, which item 3 did not
  exercise: `CompletionRequest`'s `from_json` (`protocol.cpp:330`) and
  `serving_completion.cpp`.

## The one probe left

Everything reproducible without hardware has been reproduced. The remaining
question needs the box, and needs no GPU beyond a server that is already up:

1. On `dgx:gpu0` inside a lease, against a running `vllm-server` on this
   checkpoint, POST prompt 0 of the regenerated corpus to `/tokenize`
   (`add_special_tokens: false`) and to `/v1/completions`.

   **There is ONE tokenizer instance and five borrows of it**, so this probe
   does not compare two tokenizer states and must not be read as doing so.
   `server_main.cpp:1229` takes
   `const vllm::tok::Tokenizer& tokenizer = loaded->tokenizer()`
   (`model_loader.h:381`, returning `LoadedEngine::tokenizer_` at
   `model_loader.h:528`); `api_server.cpp:1229` passes that object's ADDRESS
   into the non-owning `ApiServer::tokenizer_` (`api_server.h:340`); and
   `model_loader.cpp:1528` constructs `input_processor_(tokenizer_, …)`, whose
   member is a REFERENCE (`input_processor.h:152`) to the same object.
   `model_loader.cpp:1529` and `:1508` borrow it a fourth and fifth time, by
   ADDRESS for `output_processor_(&tokenizer_)` and by REFERENCE for
   `MakeNativeBackendFactory(tokenizer_, …)`, whose parameter is declared
   `const tok::Tokenizer&` and which the header requires the object to outlive
   (`backend_native.h:155-158`). The one owning instance is
   move-constructed at `model_loader.cpp:1429`, and the count is of THIS path:
   the C ABI takes a sixth borrow at `vllm_c.cpp:1395`, off the HTTP path this
   probe drives. What the two endpoints do NOT share is the HANDLER around it:
   body parse, string extraction, and `Encode` against
   `EncodeWithSpecialTokens`.

   Read the three outcomes as follows:

   - **They disagree** — the difference is in the two HANDLERS, not between two
     tokenizer states.
   - **They agree at 915** — this does NOT isolate the tokenizer. Both handlers
     sit behind the same httplib and the same nlohmann, so a common parse or
     transport truncation and an ARM tokenizer defect are indistinguishable at
     this point.
   - **They agree at 1024** — `## Stop conditions`.
2. **The discriminating test is `examples/tokenize` built natively on aarch64**
   and run over the same prompt bytes, with no server, no HTTP and no JSON in
   the path. It is the step that separates an ARM code-generation difference
   from everything on the server path, and on an agreement at 915 it is the next
   step rather than a fallback.
3. Whichever way it falls, the prompt that reproduces it is the smallest failing
   test, and it enters through the HTTP entry point rather than by constructing a
   tokenizer by hand.

## The aarch64 hypothesis does not reproduce (2026-09-05, `c796fea41`)

`## The one probe left` says the discriminating test is `examples/tokenize` built
natively on aarch64. **It does not need the box.** CI already installs
`qemu-user` and runs `qemu-aarch64 -cpu cortex-a53` (`ci.yml:1563-1577`), so the
same comparison cross-builds and runs on any x86-64 host.

`src/vllm/tokenizer/{tokenizer,bpe,pretokenizer,unicode_data}.cpp` were built for
x86-64 (`g++ 13.3`) and for aarch64 (`aarch64-linux-gnu-g++ 13.3, -static`)
against the committed `tests/parity/goldens/tokenizer_qwen36/tokenizer.json`
(sha256 `87a7830d…72de4`, equal to the value `encodings.json` records for its
oracle) and run over two corpora. Instrument precondition: the x86-64 build
reproduces the committed goldens **53/53**, ids for ids.

| corpus | x86-64 vs aarch64 | ours vs HF `tokenizers` 0.22.2 |
|---|---|---|
| committed golden corpus, 1,417 tokens | identical | 53/53 |
| combining-mark-dense, 74-150 marks/line, 2,245 marks, 4,240 tokens | identical | 15/20 mismatch |
| the same corpus NFC-normalised, 1,999 marks retained | identical | **20/20** |

Identical also under `-cpu cortex-a53`, and under **both** `-fsigned-char` and
`-funsigned-char` — the `char`-signedness hypothesis the arm CI job's own comment
names.

**The 15/20 mismatch is the recorded NFC deviation, and the first corpus injected
it.** The mismatching line set equals the non-NFC line set exactly (15 == 15, set
equality), and the direction is wrong for this bug: ours came out **longer** than
HF, where the anomaly is shorter. Item 7 already establishes that the campaign's
48 prompts were themselves NFC, so this is not the campaign's mechanism.

**Mechanisms 1 and 2 therefore do not reproduce on today's sources.** None of the
following is closed by that, and the row stays open on each:

- qemu-user is not aarch64 silicon, and the cross-compiler is not the box's.
- The golden Qwen3.6 `tokenizer.json` is not the campaign's Qwen3.8-27B one.
- The four tokenizer sources moved **864 lines** since the campaign build
  `1dac4f9a7`, so a defect present then may have been repaired by unrelated work.

**Mechanism 3 is untested and is now the leading candidate.** The
`/v1/completions` request-parse segment — `from_json(const nlohmann::json&,
CompletionRequest&)` (`protocol.cpp:330`) and `serving_completion.cpp` —
which item 3 did not exercise, and which the arm lane cannot reach because it
sets `VLLM_CPP_SERVER=OFF`.

## Owed

- ~~**The aarch64 lane runs no tokenizer gate.**~~ **Landed 2026-08-23**
  (`af320abb2`, GATE-CI-AARCH64-COVERAGE W1b, [#1385](https://github.com/mudler/vllm.cpp/issues/1385)).
  `.github/workflows/ci.yml:1533-1538` now builds nine tokenizer targets on
  `ubuntu-24.04-arm`, `test_tokenizer_parity` among them, and that test compares
  IDS rather than counts. **It was necessary and not sufficient, and the gap is
  now measured rather than estimated.** Before this change the committed corpus
  carried **28** combining marks across its 64 entries, at most **13** in any
  single entry and no entry longer than 109 characters, against the **74-150 per
  prompt** that produced the anomaly. Six entries in that regime were added to
  `tests/parity/goldens/tokenizer_qwen36/corpus.txt`
  ([#2948](https://github.com/mudler/vllm.cpp/issues/2948)); the corpus now
  carries **687** marks, and the aarch64 lane executes them.
- The NFC deviation at `src/vllm/tokenizer/tokenizer.cpp:527-530` is accepted
  rather than applied. Neither the golden corpus nor any other gate feeds it
  non-NFC text, so the deviation is unexercised rather than passing. **This is
  deliberate in the added entries too**: every one is NFC-stable, so they probe
  the `\p{M}`-aware split and the merge ranking rather than re-measuring a
  deviation that is already recorded. Feeding non-NFC text is a separate change,
  because it would gate a behaviour this tree has chosen not to implement.

## Consequence for the campaign's numbers

`total_token_throughput = (total_input + total_output) / duration`
(`serve.py:734`) takes `total_input` from the server's own report. Our arm's
`total_input` is short by the missing prompt tokens, so OUR OWN published figure
is deflated against the corpus the client actually built, and the axis is not
comparable between the two arms at all. `.agents/benchmark-record.md:24201`
already annotates the oracle's cell "do not set this beside ours" for that
reason.

This spec corrects OUR numerator and stops there. Both columns are our arm:

| our leg | as published | with input = 1024 x N, the corpus the client built | correction |
|---|---|---|---|
| c1, mean of 3 | 38.4819 | 39.6404 | +3.01 % |
| c8, mean of 3 | 195.9628 | 203.6222 | +3.91 % |

**No ours-over-vLLM ratio is derived here, and none may be derived from these
numbers.** `tools/bench/gpu_clock_state.py compare` returned
`PAIRING_VERDICT=DISCARD` on all three c1 pairings; `docs/BENCHMARKS.md:216`
records the c1 ratio as OWED for that reason; and
`.agents/benchmark-record.md:24229` states that no ours-over-vLLM ratio is
derived from those absolutes, here or anywhere else. A number the clock gate
discarded is not a number, and a corrected numerator does not turn one into a
number. Nor would such a ratio be independent evidence if the gate ever allowed
it: with the inputs imputed equal on both arms and the outputs equal at 768, a
corpus-matched total-token ratio reduces algebraically to the ratio of the two
durations, which is exactly what `output_throughput` already divides. The speed
verdict is out of scope (`## Scope`).

`output_throughput` is biased upward because the shortfall is real work our
server did not do: 3.29 % fewer prompt tokens at c1, 4.23 % at c8. Weighted by
the prefill share of wall time the bias is **0.10-0.15 % at c1** and **about
0.6 % at c8** — measured run-to-run CV over the three reps is **0.048 %** (c1)
and **0.251 %** (c8), so the bias exceeds the noise on both legs and is not
absorbed by it. (The campaign published **0.039 %** and **0.205 %**. That is the
SAME measurement in the population rather than the sample form, not an
independent check of it: the ratio is exactly the Bessel factor sqrt(3/2) =
1.2247 on three reps, and 0.039 % x 1.2247 = 0.048 %, 0.205 % x 1.2247 =
0.251 %.)

One correction to the campaign's record: the absence of a
`WARNING: tokenizer mismatch` line is NOT evidence that the alignment pass found
the tokenizers in agreement. `bench.sh:237` pipes the client through `tail -40`,
and both that warning and `WARNING: /tokenize unavailable` are printed before the
result block, so both are cut. The saved logs cannot say which happened. Nothing
about the served counts turns on which it was, because `_fix_one` is inert at
915 either way (item 6).

## Reproduction

No GPU, no lease, no download. `python3` with `numpy` + `transformers`, and a
C++20 compiler.

1. Mirror `RandomDataset` at the pin (seed 0, `input_len` 1024, `range_ratio` 0,
   `prefix_len` 0) against the checkpoint's tokenizer; dump the prompts.
2. Compile `src/vllm/tokenizer/{tokenizer,bpe,pretokenizer,unicode_data}.cpp`
   with a stub for `vllm::GgufFile::FindKv` (needed only to link `FromGguf`) and
   encode each prompt.
3. Compare ids against HF and counts against
   `bench-20260819T035148Z/c8-r1.json`'s `input_lens`.

## Stop conditions

Stop and report `NEEDS_DECISION` if the probe shows `/tokenize` and
`/v1/completions` agreeing at 1024 on the box, because that would mean the
campaign's `usage` frames did not come from the tokenizer at all and the
attribution above is wrong.
