# SPEC-BPE-QUADRATIC-MERGE — the merge loop is quadratic, and the premise that made it look safe

**Issue:** [#1365](https://github.com/mudler/vllm.cpp/issues/1365), re-titled and
re-scoped on 2026-08-19 from "a reproducible ~4 s TTFT outlier on request 3 of
every leg" to the cause measured below.
**Kind:** one algorithm replaced inside the tokenizer, mirrored from HF
`tokenizers`. No kernel, no device code, no model numerics, no new public
surface. Every gate in this spec runs on a CPU host.
**Row:** `SPEC-BPE-QUADRATIC-MERGE` in
[`engine-matrix.md`](../engine-matrix.md), section "Loading, tokenizer, and config".
**Base:** `31f93787c`, merged with `origin/main` at `c9724b5ee`. Every local
line number in this document is read at that merge. `git diff 31f93787c HEAD --
src/vllm/tokenizer include/vllm/tokenizer src/vllm/v1/engine/input_processor.cpp
src/vllm/entrypoints/openai/api_server.cpp` is empty, so the merge moves none of
them.
**Pull request shape:** ONE pull request carrying W1 through W4. No answer is
recorded for this row under `## Git integration` in
`.agents/developer-preferences.md`. The only recorded answer there is
`SPEC-DFLASH2`'s, and AGENTS.md makes one pull request the default when no
answer is recorded. The shape is not cosmetic here: W1 lands the cost assertion
of `## Tests to port` item 2 RED against unchanged code, so a split would leave
`main` red between W1 and W3. See `## Work breakdown`.

## Now

**Closed. The row is `DONE`.** `src/vllm/tokenizer/bpe.cpp::BpeMerge` is the
heap of HF `tokenizers` `Word::merge_all`
(`tokenizers/src/models/bpe/word.rs:162-250`), `MergeRanks` is an
identifier-keyed table mirroring `MergeMap`
(`tokenizers/src/models/bpe/model.rs:19`), and a merge naming a token absent
from the vocabulary is refused at load on both load surfaces, mirroring
`MergeTokenOutOfVocabulary` (`model.rs:174-192`). The fix landed as
`67823aee2`, which is the owner the row now names, and the closing commit
promoted `GATING` -> `DONE` on 2026-08-21 after the rerun recorded under
`### The closing rerun`. Nothing about this row is open. The next step on this
surface belongs to a different row: [#1541](https://github.com/mudler/vllm.cpp/issues/1541),
the refusing request-boundary length guard, which is a second layer after the
algorithmic fix and never a substitute for it.

`## Tests to port` item 2 was landed RED against the shipped code and turned
green by W3. `tests/vllm/test_bpe_equivalence.cpp` holds 80 corpus entries
across both committed goldens and both special-token modes, 320 id vectors,
every one of them recorded from HF `tokenizers` 0.22.2 rather than from our own
output. The identifiers did not move. `## Outcome` carries the idle-host
re-measure, what was rejected, and the two limitations this row is disclosing
rather than closing.

## Scope

In scope:

- `src/vllm/tokenizer/bpe.cpp::BpeMerge`, replaced by a heap-driven merge that
  mirrors HF `tokenizers` `Word::merge_all`.
- Whatever `include/vllm/tokenizer/bpe.h` has to expose so that
  `src/vllm/tokenizer/tokenizer.cpp::EncodePlain` and
  `src/vllm/tokenizer/tokenizer.cpp::EncodePlainSp`, the only two callers, reach
  the new form.
- The interning decision described under `## Design`, and the load-time refusal
  it needs, on **both** load surfaces. `merge_ranks_` is built from
  `tokenizer.json` by `src/vllm/tokenizer/tokenizer.cpp::FromHfJson`
  (`InsertMerge` at `src/vllm/tokenizer/tokenizer.cpp:697`) and from a GGUF's
  `tokenizer.ggml.merges` by `src/vllm/tokenizer/tokenizer.cpp::FromGguf`
  (`src/vllm/tokenizer/tokenizer.cpp:758`, the same `InsertMerge` at `:864`).
  Both reach the same table and the same `BpeMerge`, so the refusal has to be
  written once where the table is built and gated on both. `FromGguf` is already
  gated: `tests/vllm/test_bpe.cpp` builds a zero-tensor GGUF and loads it
  through `Tokenizer::FromGguf`.
- Red-first equivalence and cost cases in `tests/vllm/test_bpe.cpp` and one new
  CPU-tier binary described under `## Tests`.
- `docs/STATUS.md` and `docs/BENCHMARKS.md` when the row moves to `DONE`.

Out of scope, each with a stated reason:

- **A cap on pretoken length.** Argued and rejected under `## Design`.
- **The pretokenizer.** `src/vllm/tokenizer/pretokenizer.cpp` mirrors an
  upstream regex. Its unbounded runs are the regex's behaviour, not a defect,
  and shortening one changes the token identifiers we emit.
- **A word cache.** Upstream has one (`tokenizers/src/models/bpe/model.rs:475-496`)
  and it stores only sequences below `MAX_LENGTH = 256`
  (`tokenizers/src/utils/cache.rs:10`), so it does not touch the case this row
  exists for. Adding it is a separate, measurable question.
- **A request-size limit at the HTTP boundary.** Discussed under
  `## Defence in depth`, and deliberately not bundled here.
- **`ignore_merges`, the added-token split, the detokenizer.** Untouched.

## Our baseline

Every figure below was measured on an AMD Ryzen 9 9950X3D (20 cores) at `-O2`,
against a driver built from this tree's own tokenizer sources. **Three harnesses
produced them and the tables say which.**

- **Harness A**, the one that found the defect: `BpeSplit` alone, against the
  `merges.txt` of `qwen3.8-27b-hf`, 247,586 merges, sha256
  `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d`, at a load
  average between 130 and 190. Those are heavily contended single-shot numbers.
- **Harness B**, the re-derivation: full `Tokenizer::Encode` through the two
  committed goldens, min-of-k, at a load average between 4 and 90 on the same
  box, 2026-08-19. Every figure carries its `k` and the load. The box was not
  idle: another session was compiling throughout, which is why the minimum of a
  repetition set is reported rather than a mean. The minimum is the least
  contended estimate available, not a claim of an idle host.
- **Harness C**, the confirmation: the same Harness B binaries re-run on
  2026-08-19 at load average 23 to 57, which is BUSIER than either earlier run.
  It exists to answer one question, whether the recorded constants reproduce.
  They do not, and the direction is the one contention predicts: `a` x 65,535
  read 37,564.80 ms against Harness B's 24,358.86 ms, a 54% inflation at roughly
  double the load. The growth ratios it re-derived are gone with the assertion
  they were derived for; `## Tests to port` item 3 records that ruling.

**None of those three harnesses was committed, and that is the root cause of
three rounds of irreproducible constants.** `tools/bench/bpe_encode_cost.cpp`
is the committed one, added with this revision. It is not any of A, B or C: it
is a fourth driver, it re-derived the English figures within the factor that
load alone explains, and its build and run commands are in `## Gates`. Every
figure below predates it and none can be re-derived from the tree at the
revision that recorded it.

The **shape** is what all three harnesses agree on, and it is the result. Every
absolute figure below is a contended minimum, and Harness C is the evidence that
none of them is quotable as a constant: the same input on the same binary moved
54% on load alone. A same-binary idle-host re-measure still belongs to the
implementing row, and `## Gates` requires it before any speed claim is accepted.

### The benchmark prompts that found it

Harness A. The six reconstructed `vllm bench serve` prompts of the #1365 legs,
timing `BpeSplit` only:

| prompt | bytes | pretokens | longest pretoken | current | heap prototype |
|---|---:|---:|---:|---:|---:|
| 0 | 8,798 | 698 | 88 | 3.70 ms | 3.27 ms |
| 1 | 6,629 | 344 | 119 | 2.94 ms | 2.93 ms |
| **2** | **8,034** | **1** | **8,034** | **1,356.89 ms** | **6.75 ms** |
| 3 | 6,682 | 723 | 47 | 1.65 ms | 1.93 ms |
| 4 | 6,939 | 737 | 91 | 1.69 ms | 2.01 ms |
| 5 | 5,863 | 709 | 73 | 3.72 ms | 2.59 ms |

Index 2 is the outlier request of #1365's four legs. It is one pretoken.

### The growth is quadratic

Harness A. Prompt 2 truncated and tripled, one pretoken throughout:

| bytes | BPE merge |
|---:|---:|
| 2,007 | 81.57 ms |
| 4,017 | 350.61 ms |
| 6,024 | 865.44 ms |
| 8,034 | 1,933.60 ms |
| 24,102 | 14,794.38 ms |

Three times the bytes costs 7.65 times the time.

### Five rules are unbounded, not one

The Qwen and Llama-3 alternation has exactly seven rules, dispatched at
`src/vllm/tokenizer/pretokenizer.cpp:810-821`. **Five of them scan
`while (p < t.size())` with no cap:**

| rule | function | bound |
|---:|---|---|
| 1 | `MatchContraction` (`:82`) | **bounded**: a fixed alternation, at most three bytes |
| 2 | `MatchLetterRun` (`:106`); `MatchTekkenLetterRun` (`:193`) on Tekken | unbounded |
| 3 | `MatchNumbers` (`:219`) | **capped** at `max_digits`: one codepoint for Qwen, three for Llama-3 |
| 4 | `MatchPunctRun` (`:241`) | unbounded |
| 5 | `MatchWsNewlines` (`:269`) | unbounded |
| 6 | `MatchWsNotBeforeNonSpace` (`:287`) | unbounded |
| 7 | `MatchWs` (`:303`) | unbounded |

Rule 5 was missed by the first revision of this spec and by #1365 as first
re-scoped. It matches `\s*[\r\n]+`, so a block of newlines is one pretoken and
pays the same cost. A pasted log and a pasted diff both have that shape. Two
rules are bounded, and only one of those two is bounded by a *cap*: saying
"only `MatchNumbers` is capped" is true of rule 3 and says nothing about rule 1,
which is bounded by its own alternation rather than by a counter.

Harness B, 8,192 bytes of one character, `Tokenizer::Encode` through the
committed `tests/parity/goldens/tokenizer_qwen36/tokenizer.json`, min-of-5, load
average 33 to 37:

| input | rule | pretokens | current | heap prototype |
|---|---|---:|---:|---:|
| `a` x 8,192 | 2, letter run | 1 | 467.59 ms | 1.96 ms |
| `中` x 2,730 (+2 B) | 2, letter run | 2, longest 8,190 | 433.12 ms | 0.81 ms |
| `ก` x 2,730 (+2 B) | 2, letter run | 2, longest 8,190 | 532.33 ms | 1.14 ms |
| `的` x 2,730 (+2 B) | 2, letter run | 2, longest 8,190 | 389.74 ms | 0.82 ms |
| `~` x 8,192 | 4, punct run | 1 | 414.21 ms | 1.91 ms |
| **newline x 8,192** | **5, `\s*[\r\n]+`** | **1** | **580.44 ms** | **2.60 ms** |
| space x 8,192 | 7, whitespace | 1 | 618.56 ms | 2.84 ms |

The pretoken counts are read from `vllm::tok::Pretokenize` itself, not inferred:
the CJK and Thai inputs are 2,730 codepoints plus a 2-byte remainder, which is a
second pretoken, so their longest run is 8,190 rather than 8,192.

The trigger is a long run of one character class, not one script. ASCII reaches
it, and so does an empty-looking block of newlines.

### At 65,535 bytes

Harness B, `Tokenizer::Encode` through the committed Qwen3.6 golden. Every input
is one pretoken of 65,535 bytes, read from `vllm::tok::Pretokenize`. Current
min-of-2, prototype min-of-5, load average 25 to 90. The current column is
min-of-2 rather than min-of-5 because each repetition costs about 30 s of one
core, which is itself the finding:

| input | rule | current | heap prototype | ratio |
|---|---|---:|---:|---:|
| `a` x 65,535 | 2 | 24,358.86 ms | 20.71 ms | 1,176x |
| `中` x 21,845 | 2 | 26,823.67 ms | 8.00 ms | 3,353x |
| `ก` x 21,845 | 2 | 39,122.71 ms | 10.72 ms | 3,649x |
| `的` x 21,845 | 2 | 30,309.46 ms | 7.40 ms | 4,096x |
| `~` x 65,535 | 4 | 32,468.48 ms | 18.85 ms | 1,723x |
| newline x 65,535 | 5 | 40,538.57 ms | 24.14 ms | 1,679x |
| space x 65,535 | 7 | 45,780.21 ms | 28.07 ms | 1,631x |

### The SentencePiece family pays it on ordinary English

`src/vllm/tokenizer/tokenizer.cpp::EncodePlainSp` calls the same `BpeMerge`, and
its pretokenization is Metaspace, not the split regex. When `metaspace_split_`
is false it calls `encode_piece` on the whole input
(`src/vllm/tokenizer/tokenizer.cpp:1022-1024`). Two shipped families are in that
state: `src/vllm/tokenizer/tokenizer.cpp:583` hardcodes `false` for the Gemma
metaspace layout, and the committed Mistral golden
`tests/parity/goldens/tokenizer_mistral/tokenizer.json` declares
`"split": false` in its own `pre_tokenizer`.

Full `Tokenizer::Encode` through that committed golden, on "The quick brown fox
jumps over the lazy dog. " repeated to length. Two controls run the identical
text: the committed Qwen3.6 golden, which is the byte-level arm of our own
tokenizer, and **HF `tokenizers` 0.22.2 reading the same Mistral
`tokenizer.json` file**, which is the code the pinned `transformers` executes
and therefore the oracle for both the identifiers and the cost. Harness B,
min-of-3 for the two Mistral columns, min-of-5 for the Qwen3.6 control, load
average 4 to 12:

| prompt bytes | ids | ours, Mistral golden | HF `tokenizers` 0.22.2, same file | ours / HF | ours, Qwen3.6 control |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 267 | 5.61 ms | 0.14 ms | 40x | 0.058 ms |
| 8,000 | 2,134 | 358.83 ms | 1.09 ms | 329x | 0.457 ms |
| 32,000 | 8,534 | 6,005.56 ms | 4.92 ms | 1,221x | 1.883 ms |
| 65,536 | 17,476 | 25,345.61 ms | 10.11 ms | **2,507x** | 3.877 ms |

**The identifiers are the same on both sides**: the `ids` column is one number
because ours and HF's agree at every size, and at 8,000 bytes the two id
sequences were compared element by element and are byte-identical (2,134 of
2,134). So the 2,507x is a pure cost difference on identical output, not a
different answer computed faster.

Fitting our Mistral column, 65.536x the bytes costs 4,518x the time, an exponent
of **2.01**. Read at a single 4x step it is plainer still: 8,000 to 32,000 bytes
costs 16.74x, against the 16x that `n^2` predicts and the ~4.6x that
`n log n` does (`4 * log2(32000) / log2(8000)`, the word being the whole prompt
on this arm, so `n` is the prompt length in codepoints). The Qwen3.6 control
over the same 65.536x range costs 66.8x, which is linear. HF over the same range
costs 72x, also linear.

A Mistral or Gemma server spends 25 s of one core to tokenize a 64 KB English
document that HF tokenizes in 10 ms, and a user who pastes a document sends
exactly that.

### Why no gate saw it

The committed 64-entry corpus at `tests/parity/goldens/tokenizer_qwen36/` has a
longest pretoken of **54 bytes**. The failing regime starts two orders of
magnitude above it. A token gate also cannot see this class of defect at all,
because the identifiers it compares are correct.

## Reachability

The encode is on the request path of a shipped server, and it runs before the
length check that a reader would expect to bound it:

```
$ sed -n '259,265p' src/vllm/v1/engine/input_processor.cpp
  std::vector<int32_t> prompt_token_ids =
      tokenizer_.EncodeWithSpecialTokens(prompt);
  ...
  ValidatePromptLen(prompt_token_ids.size());
```

`src/vllm/v1/engine/input_processor.cpp::ValidatePromptLen` runs at `:265`, five
lines after the encode at `:260`. `max_model_len` therefore bounds nothing here:
the cost is paid in full, and only then does the server decide the prompt is too
long to serve. `src/vllm/entrypoints/openai/api_server.cpp:763-764` reaches the
same encode from `/tokenize`, which needs no engine. The worker pool is sized by
`src/vllm/entrypoints/openai/api_server.cpp::HttpWorkerCount` as
`max_concurrent_streams + kControlWorkerHeadroom`, so one caller can occupy
every worker at once.

This is a remote denial of service in a shipped path. It is not a latency
curiosity, and the row is scoped that way.

## Upstream chain

vLLM does not implement a tokenizer. `vllm/tokenizers/hf.py` delegates to
HuggingFace `transformers`, which delegates to the Rust `tokenizers` crate, and
that crate holds the merge loop. So the executing chain for a vLLM encode is:

| Layer | What it decides | Anchor |
|---|---|---|
| vLLM | which tokenizer to build | `vllm/tokenizers/registry.py:176`, `vllm/tokenizers/hf.py:163` |
| `transformers` 5.14.1 | the `tokenizer.json` contract | the pin resolved inside the pinned vLLM environment, [`../oracles/transformers.md`](../oracles/transformers.md) |
| `tokenizers` 0.22.2 | **the merge algorithm** | `tokenizers/src/models/bpe/word.rs:162-250` (`Word::merge_all`) |
| `tokenizers` 0.22.2 | the merge-table representation | `tokenizers/src/models/bpe/mod.rs:9` (`type Pair = (u32, u32)`), `model.rs:19` (`MergeMap`), `model.rs:174-192` (built at load, refuses an out-of-vocabulary merge) |
| `tokenizers` 0.22.2 | the tie-break order | `tokenizers/src/models/bpe/word.rs:28-36` (`Ord for Merge`) |
| `tokenizers` 0.22.2 | what a stale heap entry is | `tokenizers/src/models/bpe/word.rs:197-205`, which compares `new_id`, NOT the pair |
| `tokenizers` 0.22.2 | the symbol list | `tokenizers/src/models/bpe/word.rs:37-53` (`Symbol`, `merge_with`) |
| `tokenizers` 0.22.2 | word construction, byte fallback, unk fusing | `tokenizers/src/models/bpe/model.rs:382-460` (`merge_word`) |
| `tokenizers` 0.22.2 | the word cache, out of scope here | `model.rs:475-496`, `tokenizers/src/utils/cache.rs:7,10` |

The crate source read for this spec is `tokenizers-0.22.2.crate`, sha256
`b238e22d44a15349529690fb07bd645cf58149a1b1e44d6cb5bd1641ff1a6223`, fetched from
`static.crates.io`. `tokenizers` is not a registered oracle in its own right; it
is the code `transformers` executes, and
[`../oracles/transformers.md`](../oracles/transformers.md) is the pin that
selects this version. The implementing row records the version it read.

## Port map

| Upstream | Ours today | After |
|---|---|---|
| `word.rs:162-250` `Word::merge_all` | `src/vllm/tokenizer/bpe.cpp::BpeMerge`, an O(n^2) rescan | the same signature, heap-driven |
| `word.rs:37-43` `Symbol {c, prev, next, len}` | `std::vector<std::string>` with `erase` per merge | a linked symbol list with `len = 0` tombstones |
| `word.rs:28-35` `Ord for Merge` | `strict <` inside the scan | an explicit comparator, with the tie pinned by a test |
| `mod.rs:9`, `model.rs:19` identifier-keyed `MergeMap` | `MergeRanks = unordered_map<string, int32_t>` in `include/vllm/tokenizer/bpe.h` | an identifier-pair-keyed table built at load |
| `MergeKey` in `src/vllm/tokenizer/bpe.cpp`, deleted by this row | one `std::string` per probe | deleted; there is no key to build |
| `model.rs:180-189` `MergeTokenOutOfVocabulary` | no rule; the failure appears per request in `src/vllm/tokenizer/tokenizer.cpp::EncodePlain` | refused at load, naming the missing token, on `FromHfJson` AND `FromGguf` |
| `model.rs:169-173`, `:186` `new_token = format!("{}{}", a, &b[prefix_len..])` | no counterpart: the `MergeKey` this row deletes concatenated `a` and `b` whole | still whole. `prefix_len` is `continuing_subword_prefix.len()`, and `src/vllm/tokenizer/tokenizer.cpp:641-648` already REFUSES a non-empty `continuing_subword_prefix` at load, so `prefix_len` is 0 on every checkpoint we accept and the term is inert for us. Port the concatenation without it, and do not silently drop the refusal that makes that legal |
| `model.rs:382-460` `merge_word` | `src/vllm/tokenizer/tokenizer.cpp::EncodePlainSp`'s symbol builder | unchanged in behaviour, emitting identifiers |
| `model.rs:475-496` word cache | absent | still absent, and out of scope |

## Design

**Mirror `Word::merge_all`.** HF `tokenizers` 0.22.2, the backend the pinned
`transformers` 5.14.1 resolves, solves this at
`tokenizers/src/models/bpe/word.rs:162-250`. The crate source is
`tokenizers-0.22.2.crate`, sha256
`b238e22d44a15349529690fb07bd645cf58149a1b1e44d6cb5bd1641ff1a6223`. Its shape:

1. A doubly linked list of symbols, each carrying `c`, `prev`, `next`, `len`
   (`word.rs:37-43`). A removed symbol is tagged `len = 0` rather than erased,
   so no index shifts (`word.rs:210`) and the final `retain` drops them all at
   once (`word.rs:249`).
2. A `QuaternaryHeap` of candidate merges, seeded once from every adjacent pair
   (`word.rs:163-178`).
3. Pop the best candidate, apply it, and push only the two pairs the merge
   creates: the one with the previous symbol and the one with the next
   (`word.rs:217-244`).
4. A popped candidate is validated before use rather than removed when it goes
   stale: skip it when its left symbol was removed (`word.rs:187`), when it has
   no right neighbour (`word.rs:191`), or when the merge table's `new_id` for
   the pair now at that position differs from the entry's own `new_id`
   (`word.rs:197-205`). **The third test compares `new_id`, not the pair.**
   Upstream's line is
   `merges.get(&target_new_pair).is_none_or(|(_, new_id)| *new_id != top.new_id)`
   (`word.rs:200-203`), so a table with two distinct pairs mapping to one
   `new_id` accepts the entry where a pair comparison would reject it. Port the
   `new_id` comparison, and note that this is only expressible once the table is
   identifier-keyed, which is the second reason W2 comes before W3.

That is O(n log n) merges of O(1) work each, against our O(n) rescan per merge.

**The order must stay identical, and upstream makes it explicit.** `Ord for
Merge` (`word.rs:28-36`) inverts both comparisons, so the max-heap yields the
lowest `rank` first and the lowest `pos` on a tie. That is the same leftmost
rule our loop gets from `strict <`. Our implementation states this in a comment
and a test pins it, because a heap silently reversing a tie is exactly the
change a token gate over ordinary text would not catch.

**Intern the symbols, which removes the string key rather than optimising it.**
Upstream's merge table is `AHashMap<(u32, u32), (u32 rank, u32 new_id)>`
(`tokenizers/src/models/bpe/mod.rs:9`, `model.rs:19`), built once at load
(`model.rs:174-192`). A symbol is a vocabulary identifier, a pair is two of
them, and `MergeKey` has no counterpart upstream. Our `MergeRanks` is
`unordered_map<string, int32_t>`, so we build one key string per probe. Mirroring
upstream deletes that whole class of work.

Interning has one precondition, and upstream states it as a refusal: a merge
whose left token, right token, or concatenation is absent from the vocabulary is
`MergeTokenOutOfVocabulary` at load (`model.rs:180-189`). Our string form has no
such rule, so a merge table can currently produce a symbol that
`src/vllm/tokenizer/tokenizer.cpp::EncodePlain` then fails on at encode time
with `symbol "..." not in vocab`. Mirroring the upstream refusal moves that
failure from per-request to load, which is where upstream puts it and where a
user can act on it. **This is a behaviour change on malformed inputs and it must
be called out in review**: a checkpoint that loads today and fails on some
prompts would be refused at load instead. That is the upstream behaviour, so it
is the behaviour we mirror, but the implementing row records which committed
goldens it verified still load.

**The two callers differ and both must be carried.**
`src/vllm/tokenizer/tokenizer.cpp::EncodePlain` starts from single mapped
codepoints, each of which is in the vocabulary for a complete byte-level
alphabet. `src/vllm/tokenizer/tokenizer.cpp::EncodePlainSp` builds symbols
first, with byte-fallback decomposition and a `kUnk` sentinel that is
deliberately not a real symbol, and only then merges. Upstream does the same
split (`model.rs:382-460` builds the `Word`; `merge_all` runs after), so the
seam survives. The sentinel needs a reserved identifier that no merge can name.

**Rejected: a cap on pretoken length.** It changes the token identifiers we
emit, so it breaks the mirror on exactly the inputs it claims to protect.
Truncation is worse, because it changes the output silently. Nobody should
reach for this as a shortcut, which is why it is written here.

**Removing the per-probe `std::string` on its own is not a separate option, and
that is an argument about the design rather than a measurement.** Interning is
forced by the algorithm: upstream's heap entry carries `new_id`
(`word.rs:8-12`), its staleness test compares `new_id` (`word.rs:197-205`), and
its table is keyed on an identifier pair (`mod.rs:9`, `model.rs:19`). A merge
step that has to rebuild a `std::string` to name a pair cannot express that
test. So the key disappears as a consequence of mirroring `merge_all`, not as a
tuning step taken beside it. The rescan is the `O(n^2)` term and the key is a
constant on top of it. Removing a constant from a quadratic leaves a quadratic,
which is why nobody should reach for the key alone as the cheap half.

**How much the key alone is worth has NOT been measured on an idle host, and
this spec makes no claim about it.** A first revision of this spec asserted that
a reused key buffer had been measured and was *slower*, and a same-binary A/B
did not reproduce that direction: the sign reversed on the largest case. The
original figures were taken at load average 130-190, which is where a reversed
sign comes from. The claim is withdrawn rather than restated with a new sign,
because neither reading has an idle-host A/B behind it and the design argument
above does not need one. What IS measured, in `## Prototype evidence`, is the
heap against the current loop, and that ratio is three orders of magnitude, not
a margin any allocation constant could account for.

## Prototype evidence

A prototype of the design, built against this tree's sources by substituting one
`bpe.cpp` in the link line, produced output **bit-identical** to `BpeMerge`
everywhere it has been run.

**Harness A**, symbol output only, against the `qwen3.8-27b-hf` merge table:
ten inputs: the six benchmark prompts, 8,192 spaces, 8,192 `~`, 8,192 `a`,
2,731 `中`, and the 65,535-byte `的` case. **Every one of those is byte-level.**
None of them exercises `EncodePlainSp`, which is the arm this row exists for,
so on its own that set does not cover the SentencePiece path at all.

**Harness B** closes exactly that gap, comparing token identifiers out of full
`Tokenizer::Encode` through **both** committed goldens, the SentencePiece
`tokenizer_mistral` and the byte-level `tokenizer_qwen36`, over ten inputs each:
2,000 and 8,000 bytes of English prose, 4,096 bytes each of spaces, `a`, `~` and
newlines, 2,001 bytes of `的`, 3,000 bytes each of Chinese and Thai prose, and
4,000 bytes of mixed script with punctuation, digits and whitespace. All 20
comparisons are identical, and **all 20 also match HF `tokenizers` 0.22.2
reading the same two files**, which is what makes them an oracle result rather
than a self-consistency check.

It is a prototype, not the implementation. Three differences are named so the
implementer does not inherit them by accident:

- It validates staleness by comparing the merged **string**, where upstream
  compares `new_id` (`word.rs:197-205`) and the implementation must too. On a
  merge table with two distinct pairs sharing one `new_id`, the two tests are
  not the same test.
- It uses `std::priority_queue` where upstream uses a 4-ary heap. The order is
  the same because the comparator is the same. The constant is not.
- It still builds a `MergeKey` string per candidate and a merged string per
  heap entry, because it reuses the string-keyed `MergeRanks`. The
  implementation interns, so its constant is lower and its growth ratio is
  closer to `n log n` than the prototype's. **A bound derived from the
  prototype's ratio is therefore not a bound for the implementation.** No such
  bound is asserted anywhere in this row: `## Tests to port` item 3 records the
  ruling that dropped it.

## Defence in depth

**Nothing else in the stack bounds this, and that is checked rather than
assumed.** Two facts, both read at this base:

- **There is no authentication anywhere in `src/vllm/entrypoints/`.** A
  case-insensitive grep for `api_key`, `api-key`, `bearer`, `authorization` and
  `authenticat` over all 143 files of `src/vllm/entrypoints/` and
  `include/vllm/entrypoints/` returns nothing, exit status 1. So there is no
  credential between an unauthenticated caller and `/tokenize`, which needs no
  engine and no model.
- **The only size bound in the stack is httplib's default**,
  `CPPHTTPLIB_PAYLOAD_MAX_LENGTH` at `third_party/httplib/httplib.h:129-130`,
  which is `100 * 1024 * 1024`. Nothing in `src/`, `include/`, `cmake/` or
  `CMakeLists.txt` overrides it and nothing calls `set_payload_max_length`, so
  100 MB is what a request body may be.

Extrapolate the measured `n^2.01` fit, taking `t = c n^2` with `c` from the
25,345.61 ms at 65,536 bytes — one session's reading at load average 4 to 12,
which is the term that has since been shown to move a figure like this one by
54%. A 100 MB body of a single character class is then on the order of `6e7` s,
several hundred CPU-days of one core, for one request.
That extrapolation spans three decades beyond the largest measured point, so
read it as an order of magnitude and not as a measurement. What is measured is
that 64 KB already costs 25 s. Either way the conclusion does not depend on the
constant: the cost is unbounded in the request body, unauthenticated, and paid
on the HTTP worker before any check.

A length guard at the API boundary is worth having **after** the algorithmic
fix, not instead of it, and it is not in this row's scope. Two constraints bind
it if a later row adds one:

- It must **refuse**, with an explicit error naming the limit. Truncation would
  silently change the token identifiers we return, which is the same defect as
  the rejected cap.
- It belongs where a caller can be told, at the request boundary, not inside the
  tokenizer. `src/vllm/v1/engine/input_processor.cpp::ValidatePromptLen` is the
  wrong place, because it needs the token count that the expensive step
  produces.

After the fix, 64 KB encodes in tens of milliseconds. The prototype takes
15.485 ms on the English case and 7.40 ms to 28.07 ms on the single-class cases.
The guard therefore protects against a future regression rather than against
today's cost. File it as its own issue when the row lands.

## Dependencies

**Code:** none. The row is self-contained host code. It needs no oracle run, no
lease, no checkpoint mount, and no other row's code. The two goldens its gates
read are already committed. It does not block on
[#915](https://github.com/mudler/vllm.cpp/issues/915) and #915 does not block on
it, although closing this row removes the outlier that
[`qwen38-27b-bf16-gate.md`](qwen38-27b-bf16-gate.md) records.

**Record:** none, and this row appends no
[`issue-index.md`](../completed/issue-index.md) row of its own.

#1365 already has its row. `9e1a5e573`
([PR #1369](https://github.com/mudler/vllm.cpp/pull/1369)) appended it at
13:05:11Z on 19 August 2026, and it was correctly anchored when it landed. A second row for the same issue number is
what `scripts/check-agent-record.py` reports as `issue #1365 listed twice`,
because under `merge=union` a duplicate is exactly what two branches appending
one issue look like. So no row is appended here, and that is the finished state
rather than a deferral.

**The landed row's text is stale, and it is not repaired by editing it.** #1365
was renamed and re-scoped in place at 17:05:51Z on 19 August 2026, from the
symptom, a reproducible ~4 s TTFT outlier on request 3 of every leg, onto the
cause this spec addresses. The index row was written before that and still reads
as the original symptom report. The issue now legitimately covers both, and its
body keeps the original observation verbatim under
`## Original report, kept verbatim`. The index is append-only, so the row stays
as it landed and this paragraph is where the reader is told that the row
describes the symptom while the issue describes the cause.

## Work breakdown

Non-overlapping. W1 is the whole correctness argument and can be reviewed
before any algorithm changes.

| W | Deliverable | Reviewable on its own because |
|---|---|---|
| **W1** | The equivalence corpus and harness of `## Tests to port` item 1, plus the cost assertion of item 2, all against UNCHANGED code | items 1 and 6 pass, item 2 is red. That red is the defect, recorded before anything moves |
| **W2** | The identifier-keyed merge table and the load-time refusal, with `BpeMerge` still doing the O(n^2) scan over identifiers | W1 stays exactly as green and as red as it was. Only the representation moved |
| **W3** | The heap merge, the linked symbol list, the comparator, the staleness checks | W1's item 2 turns green and nothing else changes |
| **W4** | The idle-host re-measure, the `## Outcome` section, `docs/STATUS.md` and `docs/BENCHMARKS.md` | the code is frozen; this is the record |

W2 before W3 is deliberate. Interning and the heap are separable in the tree
even though they are one design, and splitting them means a reviewer never has
to hold a representation change and an algorithm change in mind at once. It is
also the order the `new_id` staleness test needs: that comparison cannot be
written against a string-keyed table.

**These four waves are four commits in ONE pull request, and the split is a
review aid rather than a landing plan.** W1 lands item 2 red against
unchanged code, so between W1 and W3 the gate is failing by construction. A
separate W1 pull request would put that red on `main` and leave every other
row's gate unable to tell its own failure from this one. AGENTS.md's default
with no recorded preference is one pull request, none of its three split cases
applies here, because no helper needs a base-reachable spec, the scope is one
function, and W1 through W3 all write product code. The commit order still
proves the spec came first. If the developer prefers a split, the red must land
`SKIP`-ped with the reason and the owning wave named, and W3 must remove the
skip in the same change that turns it green. A permanently skipped assertion
is not an acceptable resting state, which is also why item 3 is dropped outright
rather than deferred.

## Risks

| Risk | Why it is real | Control |
|---|---|---|
| A tie is resolved differently and one token identifier changes | The heap's comparator is where leftmost-wins lives, and a reversed tie is invisible on most text | Port `Ord for Merge` (`word.rs:28-35`) with its inversion; a direct tie case in `test_bpe.cpp`, which already pins `BpeSplit("ĠĠĠ", r2) == {"ĠĠ", "Ġ"}` |
| A stale heap entry is applied | Positions and neighbours change under the heap | Mirror all three validations (`word.rs:187`, `:191`, `:197-205`), the third of them comparing `new_id` rather than the pair; a red-first case that reaches a stale entry |
| Interning refuses a checkpoint that loads today | The load-time vocabulary rule is new for us | Verify every committed golden still loads; record which ones in `## Outcome` |
| A llama.cpp-converted GGUF is refused where `tokenizer.json` was not | `src/vllm/tokenizer/tokenizer.cpp::FromGguf` builds the same `merge_ranks_` at `:864`, and `tokenizer.ggml.merges` is a legacy `"left right"` string list produced by a CONVERTER rather than copied from the original checkpoint. A converter that drops or renames a vocabulary entry names a token the vocabulary does not carry, and after W2 that is a load-time refusal on a file that loads today. Neither we nor upstream have a GGUF case for this: HF `tokenizers` never reads GGUF, so upstream cannot be the oracle for the GGUF arm | The refusal message names the missing token, so the failure is actionable at load rather than per request. `tests/vllm/test_bpe.cpp` already loads a GGUF through `FromGguf`; the implementing row adds one case that a GGUF merge naming an absent token is refused at load with the name in the message, and one that the existing well-formed GGUF still loads. If a real converted checkpoint is refused, that is `NEEDS_DECISION` under `## Stop conditions`, not a quiet widening |
| The `kUnk` sentinel collides with a real identifier | The SentencePiece path merges a value that is not a vocabulary entry | Reserve an identifier outside the vocabulary range and assert it |
| The equivalence corpus is too small to be evidence | The existing 64-entry corpus missed this defect at 54 bytes | The corpus below is required to contain the long-pretoken regime |

## Tests to port

Red first, in this order. Each states what it fails on today and what it must
prove after.

1. **Equivalence over a corpus, token identifiers only.** A CPU-tier binary that
   encodes a corpus through the committed goldens and compares against recorded
   identifiers. The corpus is the existing 64 entries, plus the six #1365
   prompts, plus real multilingual prose in Chinese, Japanese, Thai, Lao and
   Khmer, plus the single-class runs above at a size the suite can afford. This
   test **passes on today's code** and must keep passing.
   **The recorded identifiers come from HF `tokenizers` 0.22.2, not from our own
   output.** A baseline captured from the code under test is a change detector:
   it can only prove that the answer did not move, never that the answer is
   right, so recording our own output would make the sentence "it cannot go
   green on a faster wrong answer" false. The capture is cheap and has already
   been done for the 20 comparisons of `## Prototype evidence`: `Tokenizer.from_file`
   on the same committed `tokenizer.json` file the C++ side reads, then
   `encode(text, add_special_tokens=False)` against `Tokenizer::Encode` (the
   `add_special_tokens=True` arm pairs with `EncodeWithSpecialTokens`), comparing
   `Encoding.ids` element by element. The generator script, the `tokenizers`
   version, and the golden file's sha256 are committed beside the recorded
   identifiers so the capture is reproducible. Where an entry cannot be captured
   from HF, the test records that entry as self-referential in its own comment
   rather than letting the whole corpus inherit an oracle it does not have.
2. **A cost bound that fails today. It is the ONLY timing assertion this row
   ships, and item 3 records why.** Encode one pretoken of a stated size and
   assert an upper bound on wall time. State the bound as an absolute figure
   with the margin argued, not as a ratio against a same-run baseline. At 65,536
   bytes of English through the Mistral golden the two sides read 25,345.61 ms
   and, for the prototype, 15.485 ms — one session's readings at load average 4
   to 12, tabulated in `## The SentencePiece family pays it on ordinary
   English`, not constants. What survives the load is the SEPARATION: a bound
   placed between them has **three orders of magnitude** of headroom, so a
   runner would have to be a thousand times slower than that box to cross it.

   That margin is the whole argument, and it is cheap to check that load does
   not eat it. Re-derived for this revision with the committed harness
   `tools/bench/bpe_encode_cost.cpp` on the same box on 2026-08-19, min-of-3,
   at load average 194/209/196 — an order of magnitude busier than the table
   above — English through the Mistral golden read 12.802 ms at 1,000 bytes and
   606.424 ms at 8,000 bytes, against that table's 5.61 ms and 358.83 ms. So
   roughly 1.7x on load alone, and the identifier counts (267 and 2,134) came
   back exactly. **The identifiers reproduce and the times do not**, which is
   the whole shape of this row in two numbers, and 1.7x against a thousandfold
   margin is why item 2 and not item 3 is the gate.
3. **A growth-shape assertion. CONSIDERED AND REJECTED — do not implement it.**
   The shape would have been: encode at n and at 4n, and assert the ratio sits
   below a bound that quadratic growth cannot satisfy and the implemented
   algorithm can. It is not in this row's scope, it is not deferred to the
   implementing row, and it must not land `SKIP`-ped. **Operator ruling,
   2026-08-19.** It is recorded here at length so that nobody proposes it again
   from the same reasoning that produced it twice.

   The ruling rests on a re-derivation by an independent reviewer, on the
   20-core box at load average 176 to 268:

   - **The red and green distributions OVERLAP at that load.** Ten sweeps of
     8,000 to 32,000 bytes on the CURRENT, defective code gave 4.654, 9.213,
     9.629, 11.989, 12.115, 13.206, 15.241, 16.017, 16.376 and 17.896. Five of
     the ten fall below 12.680, which a previous revision of this document
     certified as the defective code's MINIMUM, and one falls below 8.920,
     which the same revision certified as the prototype's MAXIMUM. A bound
     cannot be placed between two distributions that overlap.
   - **The mechanism is this row's own diagnosis turned against its own gate.**
     In the sweep that produced 4.654, the 8,000-byte leg read 4,383 ms against
     the other nine sweeps' 650 to 1,878 ms. A preempted DENOMINATOR alone
     dropped the defective code to a ratio the correct algorithm is supposed to
     own. Min-of-k inside each half does not help: the two halves of a ratio are
     separate, independently preemptible measurement windows, and a minimum
     taken inside one of them says nothing about how the other one was
     scheduled.
   - **The margin argument for a 2x step was FALSIFIED.** A previous revision
     argued that a 2x step is unusable because a correct algorithm was observed
     at 4.414, above the 4.0 that quadratic predicts. That 4.414 was a
     single-session artifact. Twenty-four min-of-8 sweeps of HF `tokenizers`
     0.22.2 — the reference implementation this design mirrors, at the version
     [`../oracles/transformers.md`](../oracles/transformers.md) pins — ranged
     0.756 to 3.349 at load average 198 to 228, and never once exceeded 4.0.
     The crossing does not reproduce, so it is recorded here as a negative
     result and is not evidence for anything.

   **Item 2 carries the timing gate alone, and that is the right split rather
   than a reduction.** Item 2's separation is about three orders of magnitude
   and is immune to load in the only sense that matters: a 1.7x inflation, which
   is what an order-of-magnitude load increase actually cost above, does not
   approach it. The best separation this assertion was ever certified at was
   1.42x, a figure that did not itself reproduce, and the separation actually
   observed above is none at all. A gate whose margin is smaller than the noise of the
   host it runs on does not measure the code; it measures the box, and on a red
   day it reports the defect as fixed.

   Three revisions of this document each certified a growth window, and no
   reviewer reproduced any of them: a 2x single-shot ratio called "the assertion
   that survives a slow runner"; a 4x min-of-8 sweep with a certified 9% spread
   and a (6.03, 16) band; and an eight-sweep re-derivation with a 1.42x window.
   The pattern is not carelessness in any one of them. It is that this quantity
   is not stable on a shared host, and the correct response to a third
   irreproducible certification is to stop certifying it.

   What the implementing row owes instead is nothing here: it implements item 2,
   and if it wants a growth reading for its `## Outcome`, it takes one with
   `tools/bench/bpe_encode_cost.cpp`, records the sweep, the `k`, the load
   average and the host beside it, and states it as a session reading. A reading
   is not a gate, and this item is the record of why this one cannot become one.
4. **The leftmost tie.** Extend the existing `test_bpe.cpp` case into one that
   distinguishes the two orders on a longer symbol list.
5. **A stale-entry case.** A merge sequence where a queued candidate is
   invalidated by an earlier merge, constructed so that applying it produces
   different identifiers.
6. **The SentencePiece arm.** The same equivalence and cost cases through
   `tests/parity/goldens/tokenizer_mistral/tokenizer.json`, whose `"split":
   false` makes the whole prompt one word. Ordinary English prose is the input,
   because that is what fails.
7. **The load-time vocabulary refusal, on BOTH load surfaces.** A merge table
   naming a token absent from the vocabulary is refused at load, with the
   missing name in the message, once through
   `src/vllm/tokenizer/tokenizer.cpp::FromHfJson` and once through
   `src/vllm/tokenizer/tokenizer.cpp::FromGguf`, whose
   `tokenizer.ggml.merges` reaches the same `InsertMerge` at
   `src/vllm/tokenizer/tokenizer.cpp:881`. `tests/vllm/test_bpe.cpp` already
   builds a zero-tensor GGUF and loads it through `FromGguf`, so the GGUF case
   is one more kv block in an existing fixture, not new machinery. Also assert
   that the existing well-formed GGUF still loads, because that is the
   regression the refusal can cause.

The reachability mutation for the fresh reviewer: delete the call to the new
merge from `src/vllm/tokenizer/tokenizer.cpp::EncodePlain` in a scratch copy and
rerun the focused gate. A gate that stays green measured a class, not the
encode path.

## Gates

**This row is unusually cheap to gate, and the plan takes that seriously.** The
tokenizer is pure host code. Nothing here needs a GPU, a lease, or a checkpoint
mount, and the two goldens the gates need are already committed:
`tests/parity/goldens/tokenizer_qwen36/tokenizer.json` (248k vocabulary) and
`tests/parity/goldens/tokenizer_mistral/tokenizer.json`.

The CPU tier can prove, with no other host:

- Token-identifier equivalence before and after, on both tokenizer families,
  over a corpus that contains the failing regime.
- The tie-break order, the stale-entry handling, and the load-time refusal.
- That the cost bound of `## Tests to port` item 2 is red before the change
  and green after. It is the row's only timing assertion; item 3 records the
  growth-shape assertion that was considered and rejected, and no gate here
  lands `SKIP`-ped.
- Through `test_tokenizer_parity`, `test_tokenizer_parity_mistral`,
  `test_tokenizer_parity_deepseek` and `test_tokenizer_parity_gpt4o`, that all
  four already-gated families still produce their recorded identifiers.
- Through `test_bpe`, `test_tokenizer_metaspace_split` and `test_detokenizer`,
  that the units this change actually edits still hold. These three are the
  suites nearest the diff and the first revision of this spec omitted all of
  them. `test_bpe` owns `BpeSplit`, `BpeMerge`, `MergeKey` and the `FromGguf`
  load path. `test_tokenizer_metaspace_split` owns the `metaspace_split_` flag
  that decides whether the SentencePiece arm makes the whole prompt one word.
  `test_detokenizer` is the round trip that a changed symbol boundary would
  break. All three are declared in `tests/CMakeLists.txt` and run in the CPU
  tier today.

What the CPU tier cannot prove: the end-to-end TTFT of #1365's legs on GB10.
That needs the bf16 27B server on `dgx:gpu0` under an `rc` lease, and it is the
row's closing evidence rather than its correctness gate. Run it only after the
CPU gates are green.

**The harness is committed, and this is the recipe AGENTS.md §Gates requires.**
`tools/bench/bpe_encode_cost.cpp` times `Tokenizer::Encode` on one synthetic
input at stated sizes through a stated `tokenizer.json`, min-of-k, and prints
the 1/5/15-minute load average beside every row it emits. Its own header carries
the same two commands and states that its output is a session reading and never
a bound. Build it, from the repository root:

```sh
g++ -O2 -std=c++20 -I include -I src -isystem third_party \
    tools/bench/bpe_encode_cost.cpp \
    src/vllm/tokenizer/bpe.cpp src/vllm/tokenizer/tokenizer.cpp \
    src/vllm/tokenizer/pretokenizer.cpp src/vllm/tokenizer/unicode_data.cpp \
    src/vllm/model_executor/model_loader/gguf_reader.cpp \
    src/vllm/model_executor/model_loader/read_only_file_mapping.cpp \
    -o /tmp/bpe_encode_cost
```

and run it against either committed golden:

```sh
/tmp/bpe_encode_cost tests/parity/goldens/tokenizer_mistral/tokenizer.json \
    --case english --repeats 5 --sizes 1000,8000
/tmp/bpe_encode_cost tests/parity/goldens/tokenizer_qwen36/tokenizer.json \
    --case a --case newline --repeats 5 --sizes 1000,4096
```

It is registered as no test and CI runs it never. CI does COMPILE it, as the
never-linked OBJECT library `vllm_bpe_encode_cost` in `CMakeLists.txt`, so it
cannot rot behind a `Tokenizer::Encode` or `FromHfJson` signature change while
still being the only artifact these figures can be reproduced from. **It is
deliberately not a gate**: item 3 rules that a growth gate is not viable on a
shared runner, and the cost bound of item 2 belongs in the test suite rather
than in a bench driver. This harness exists so that a human or an agent can
re-derive a figure deliberately, on a host whose load they have looked at.

Required before any speed figure in this spec is accepted: an idle-host,
same-binary A/B re-measure with the load average recorded. No figure above was
taken on an idle host. Harness A ran at load average 130 to 190 and Harness B at
4 to 90, on a 20-core box with another session compiling and testing throughout.
Harness B reports the minimum of a repetition set, which is the least contended
estimate the box could give, and that is not the same thing as an idle
measurement.

**The harnesses agree on the SHAPE and they do not agree on the constants.** An
earlier revision of this section said the harnesses agreed "on every absolute
figure to within about 25%". That sentence named Harness A and B and was never
updated when Harness C was added, and Harness C contradicts it directly: the
same binary on the same input read 37,564.80 ms against Harness B's 24,358.86 ms
at roughly double the load, which is **54%**. A fourth reading taken for this
revision with the committed harness moved the 8,000-byte English figure by about
1.7x on load alone. So the agreement that makes this row's argument is the
exponent, the ratio against HF on identical output, and the direction — every
one of which is invariant across all four — and no absolute figure in this
document is quotable as a constant.

## Outcome

Recorded at the close of W4, on the branch, before the merge.

### What landed, and where the mirror is exact

`BpeMerge` is `Word::merge_all` (`word.rs:162-250`): a linked symbol list with
`len = 0` tombstones, a 4-ary max-heap seeded once from every adjacent pair,
two pushes per applied merge, and the three validations of `word.rs:187,191`
and `:197-205`. `MergeRanks` is the identifier-keyed `MergeMap` of `model.rs:19`
and `mod.rs:9`, built at load, and `InsertMerge` applies the
`MergeTokenOutOfVocabulary` rule of `model.rs:174-192` on BOTH load surfaces.
`MergeKey` is deleted.

**One adaptation, and it is the only divergence.** Upstream interns to
VOCABULARY ids because the model owns the vocabulary; `MergeRanks` owns none, so
it assigns its own dense identifiers in order of first appearance among the
merge entries. The numberings differ and the equivalence they induce does not:
upstream's `new_id` is the vocabulary id of the concatenated string, and ours is
keyed on that same string, so two merges share an identifier under exactly the
same condition. `prefix_len` is 0 throughout, because `FromHfJson` refuses a
non-empty `continuing_subword_prefix` at load; that refusal is what makes the
whole-string concatenation legal and it was NOT dropped.

### What was measured

Same-harness A/B: one `tools/bench/bpe_encode_cost.cpp` source compiled against
the shipped tokenizer (`a50c57d69`) and against this branch, with the BEFORE and
AFTER legs interleaved so a drift in box state cannot land entirely on one side.

**Which instrument produced which number.** The HF legs were produced by a
scratch script, `/tmp/bpe_ab/hf_cost.py`, whose banner (`# hf_cost --`) is in the
log. `tools/bench/bpe_encode_cost_hf.py` is committed here as its id-equivalent
generalization -- same case units, same repeat-and-truncate rule, same min-of-k,
same load line, plus argument parsing the scratch version did not have -- and it
is NOT the binary that produced the figures below. It is committed for the same
reason its C++ sibling was: the reference arm of this comparison had no
committed artifact, and a recipe nobody can execute is not a recipe. The HF legs
also ran LAST rather than interleaved, after both reps of both arms, which is
why the load beside them is not the load beside ours.

**What "idle host" meant here, stated because it is not what the word usually
means.** `## Gates` requires an idle-host re-measure and this box does not go
idle: concurrent sessions built on it throughout. The run therefore waited for
an UNCONTENDED window -- a 1-minute load average below 4.00, so at least sixteen
of twenty cores free, sustained over three consecutive 30-second samples --
which is the condition a SINGLE-THREADED measurement actually needs. `AB.log`
records that wait as 23:57:37 to 00:37:08 on 2026-08-21, 39 minutes 31 seconds,
with the 1-minute load reaching 95.39 inside it. A threshold nothing can satisfy
yields no measurement rather than a careful one, so the threshold is written
down rather than quietly met.

An EARLIER attempt polled the same box for a 2.00 threshold and never got a
window. Its log was truncated by the rerun that replaced it, so its figures are
not quotable and none is quoted. That it happened is recorded; what it read is
not, because this row has already burned three generations of constants no
reviewer could re-derive and an unlogged fourth is the same defect.

| case (tokenizer) | bytes | ids | BEFORE `a50c57d69` | AFTER this branch | before/after | HF 0.22.2 |
|---|---:|---:|---:|---:|---:|---:|
| Mistral, English prose | 1,000 | 267 | 5.376 | 0.066 | 81x | 0.116 |
| | 4,096 | 1,093 | 90.990 | 0.440 | 207x | 0.547 |
| | 16,384 | 4,370 | 1,458.602 | 1.793 | 814x | 2.384 |
| | 65,536 | 17,476 | **23,620.695** | **7.797** | **3,029x** | 10.546 |
| Qwen3.6, one repeated `a` | 1,000 | 125 | 5.233 | 0.101 | 52x | 0.187 |
| | 4,096 | 512 | 88.308 | 0.539 | 164x | 0.847 |
| | 16,384 | 2,048 | 1,423.630 | 2.388 | 596x | 4.262 |
| | 65,536 | 8,192 | **22,813.108** | **10.563** | **2,160x** | 15.402 |

Milliseconds, min over k, from the rep-1 legs. Every BEFORE and AFTER figure
above was taken inside the window, at a 1-minute load of 1.98 to 2.59 AT LEG
START; the last of them, the BEFORE `a` 65,536 leg, closed at 9.74 as the window
gave out. The window opened at 00:37:08 UTC on 2026-08-21, after 39 minutes 31
seconds and 80 samples of polling.

**The shape is the finding, not the ratio.** Read down a BEFORE column and each
4x of input costs 16.9x, 16.0x, 16.2x on Mistral and 16.9x, 16.1x, 16.0x on
Qwen3.6 -- 4^2, to three readings each. Read down an AFTER column and the same
4x costs 6.7x, 4.1x, 4.3x and 5.3x, 4.4x, 4.4x -- 4^1, once the constant
factors stop dominating at the smallest size. That is the whole claim: the
exponent moved from 2 to 1. The before/after ratio quadruples per step for
exactly that reason, so the 3,029x is the largest input measured and not a
property of the change.

**Against HF the honest reading is PARITY, and it needs the rep-2 legs to be
checkable.** The window closed before rep 2 and before the HF legs, which is why
rep 2 is tabulated here rather than dismissed: it is the only one of our legs
taken at a load comparable to HF's.

| 65,536 B, Mistral English | 1-minute load | ms |
|---|---:|---:|
| AFTER, rep 1 (in the window) | 1.98 | 7.797 |
| AFTER, rep 2 (window closed) | 10.49 | **10.563** |
| HF `tokenizers` 0.22.2 (ran last) | 13.96 | **10.546** |

Setting rep 1 beside HF spans a 7x load gap and reads as a win over the
reference. It is not one, and it is not claimed. The like-for-like pair is rep 2
against HF: **10.563 against 10.546 ms, a ratio of 1.002**, which is parity
inside the noise of a shared box -- and even that still favours HF slightly on
load, so parity is the ceiling of what these legs support, not a floor.

Two claims survive, and only two. **Ours against our own past** is clean: both
sides of every before/after figure were compiled from one harness source and
interleaved inside one window. **Ours against the reference we mirror** is
parity on the input this row exists for, where before it was three orders
larger. We are not faster than HF `tokenizers` and this row does not say so.

Every figure above is a SESSION READING and none is quotable as a constant.
This row's own history is why that sentence is here: three revisions of the
spec each certified a growth window that no reviewer reproduced, and one binary
on one input moved 54% on load alone. What the readings support is the
SEPARATION and the DIRECTION, and both are three orders of magnitude wide.

The committed gate is the shorter statement. `test_bpe_equivalence`'s cost case
read 23,918.5 ms and 23,077.3 ms against the shipped code and 8.273 ms and
11.814 ms after W3, against a 2,000 ms bound: crossed by 12x before, cleared by
170x after.

### Why each default has its value

**The bound is 2,000 ms and not tighter.** It sits an order of magnitude below
the measured defective figures and two orders above the measured fixed ones. A
tighter bound buys nothing — the defect is 2,800x, not 2x — and a bound whose
margin approaches the noise of a shared runner measures the box. That is the
failure `## Tests to port` item 3 records for the growth-ratio assertion, and
nothing here reintroduces one in any form.

**The heap is 4-ary because upstream's is.** The ORDER is decided by the
comparator alone: two entries compare equal only when rank and position are both
equal, rank is the merge's index in the checkpoint's merge list, so equal rank
means the same pair and the same `new_id`, and fully identical entries are
interchangeable. Any correct heap yields the same sequence. The arity is a
constant, and it is mirrored rather than chosen because upstream's is the
constant this design was measured against.

**The refusal is at load, on both surfaces, and it names the token.** All seven
`tokenizer.json` files committed under `tests/` carry zero offending merges,
checked before the rule was written; `test_bpe` asserts that the four parity
goldens still load, and that the well-formed GGUF fixture still loads. The GGUF
arm has no oracle — HF `tokenizers` never reads GGUF — so the message names the
missing token, which is what makes a converter defect actionable at load rather
than per request.

### What was rejected

- **A growth-shape assertion.** Ruled out by `## Tests to port` item 3 before
  this row started. Not implemented, not skipped, not deferred.
- **A cap on pretoken length.** It changes the identifiers on exactly the
  inputs it claims to protect.
- **Upstream's word cache** (`model.rs:475-496`). It stores only sequences below
  `MAX_LENGTH = 256` (`tokenizers/src/utils/cache.rs:10`), so it never touches
  this regime. Still out of scope, still a separate measurable question.
- **Removing the string key alone.** W2 measures what it is worth and the
  answer does not change the ruling: interning took the 65,536-byte Mistral case
  from 23,918.5 ms to 8,335.3 ms, at load 7.65 and 46 respectively, so the two
  are not comparable and neither is quotable. It was still four times over the
  bound and still quadratic. Removing a constant from a quadratic leaves a
  quadratic.

### Limitations, disclosed rather than closed

1. **The six literal #1365 prompts are not reproducible from this tree.** They
   were read out of `out/bench-20260819T035148Z/`, a `vllm bench serve
   --dataset-name random` run on a leased GB10 against a checkpoint that is not
   committed. The corpus therefore carries the SHAPE of prompt 2 — 8,034 bytes
   that pretokenize to one pretoken, no whitespace or class boundary in it —
   under the name `issue1365/prompt2-shape`, and says so at its definition
   rather than claiming a provenance it does not have.
2. **The second staleness validation is a memory-safety guard, and no value
   assertion catches its removal.** `word.rs:191` prevents an index cast of `-1`;
   deleting it leaves both `test_bpe` and the 80-entry corpus green, because the
   `Symbol` read one slot before the vector holds an identifier that names no
   merge. Under ASan the same deletion is a heap-buffer-overflow READ of size 16
   at `bpe.cpp:284` on the fixture added for it, and the guarded build is clean.
   The `sanitize-cpu` leg is what fails on its removal; the added case exists so
   the branch is reached at all. This is recorded because a green mutation is a
   finding, not a detail. The guarded code was also swept under ASan over the
   whole committed corpus -- 160 encodes, 80 entries through both goldens,
   `checked=160 mismatch=0` and no sanitizer report -- so the guard is not
   merely present, the path it protects is exercised clean.
3. **No GB10 end-to-end re-measure of #1365's legs.** `## Gates` names it as the
   row's closing evidence rather than its correctness gate, and it needs the
   bf16 27B server under an `rc` lease. Not taken here: `## Stop conditions`
   forbids taking a lease for anything in this row's tests, and the operator
   owns the closing run.

### Owed at landing, and not filed here -- PAID

`## Defence in depth` asks for one issue to be filed when the row lands: a
REFUSING length guard at the request boundary, which is worth having after the
algorithmic fix and never instead of it. It was not filed by the implementing
branch, because that task carried no recorded remote-write authority and an
issue opened without one is a remote write nobody asked for. It was named here
so the operator files it at the merge rather than discovering the sentence
later, and that is what happened.

**The debt is paid.** The operator filed it as
[#1541](https://github.com/mudler/vllm.cpp/issues/1541) against `67823aee2`, and
the closing commit appends its [`issue-index.md`](../completed/issue-index.md) row and
lists it under `## Owed` below, which is what makes it an owned filing rather
than a deferred one. Both binding constraints survive into the issue text: it
must REFUSE with an error naming the limit, never truncate, and it belongs at
the request boundary rather than in
`src/vllm/v1/engine/input_processor.cpp::ValidatePromptLen`, which needs the
token count the expensive step produces. Nothing else was owed at landing.

### The closing rerun

Run on 2026-08-21 by a fresh session that did not write the implementation, at
base `6b48edb2c`, in its own linked worktree, from a FRESH build directory
configured `-DVLLM_CPP_BUILD_TESTS=ON` with no build type -- 505 of 505 Ninja
targets compiled and linked, zero compiler warnings, so no stale binary can be
printing this green. Every suite was run as its own executable so that
`Status:` could be read beside `assertions:`, and every one reports a NON-ZERO
case count, which is the shape a `-tc` filter typo or a class after a `__main__`
guard would break silently.

| suite | cases | assertions | `Status:` | exit |
|---|---:|---:|---|---:|
| `test_bpe` | 24 | 971 | `SUCCESS!` | 0 |
| `test_bpe_equivalence` | 2 | 334 | `SUCCESS!` | 0 |
| `test_tokenizer_metaspace_split` | 7 | 28 | `SUCCESS!` | 0 |
| `test_detokenizer` | 12 | 221 | `SUCCESS!` | 0 |
| `test_tokenizer_parity` | 4 | 1175 | `SUCCESS!` | 0 |
| `test_tokenizer_parity_mistral` | 6 | 421 | `SUCCESS!` | 0 |
| `test_tokenizer_parity_deepseek` | 6 | 2461 | `SUCCESS!` | 0 |
| `test_tokenizer_parity_gpt4o` | 5 | 1000 | `SUCCESS!` | 0 |

**The equivalence still holds and was counted, not quoted.** The corpus case
prints `compared 320 id vectors; longest corpus entry 8034 bytes`, and
`CHECK(compared == entries.size() * 4)` ties that 320 to 80 committed entries
rather than to a literal, so a corpus that silently lost entries cannot report a
full sweep. 320 of 320 match HF `tokenizers` 0.22.2, zero mismatches.

**The one timing assertion, re-derived, and what its numbers are worth.** The
box was CONTENDED throughout: 1-minute load average 24.44 at configure, 34.20 at
the run, and 38.48 recorded by the test itself on both sides of each timed leg.
`mistral/english/65536` read **72.486 ms** for 17,476 ids and `qwen36/a/65536`
**83.647 ms** for 8,192 ids, against the 2,000 ms bound: 27x and 24x of margin.
Those are about 9x and 8x the idle-window figures `### What was measured`
records (7.797 and 10.563 ms), which is the load talking and is exactly why the
gate is an ABSOLUTE bound with three orders of headroom rather than a ratio.
Read them as gate margin. They are session readings and neither is a constant.
Nothing was re-benchmarked: the W4 A/B stands as recorded, and no growth-ratio
assertion was reintroduced in any form.

**Rerun again after the merge, because the base moved under the branch.**
`origin/main` advanced to `483cd3198` while the closing commit was being gated,
so it was merged and all eight suites were run again on the merged head with no
rebuild needed (`ninja: no work to do` -- the merge carries `README.md`, one
spec and one index row, and touches no source). Same eight `exit 0`, same eight
`SUCCESS!`, same case and assertion counts, and the same **320 id vectors, 0
mismatches**. The cost case read 76.624 ms and 84.892 ms at a 1-minute load
average of 91.79, against the same 2,000 ms bound. Two runs at two loads, 38.48
and 91.79, moved those figures by about 6% and 1.5% while the bound has three
orders of headroom, which is the whole argument for an absolute bound over a
ratio, restated as a measurement instead of a claim.

### Promotion

**Promoted `GATING` -> `DONE` on 2026-08-21.** The rerun above is the operator's
own, not an implementer's report, and it supports the move: every declared gate
in `## Gates` is green at a merge of this row's records onto `6b48edb2c`. The
two things a `DONE` row owes now exist. The exact parity-ledger link is
[parity-ledger.md#L945](../parity-ledger.md#L945), appended by the closing
commit. The owner is `67823aee2`, the hexadecimal commit that landed the fix
(PR [#1539](https://github.com/mudler/vllm.cpp/pull/1539)); the closing commit
carries records only and changes no product code, so naming it as the owner
would point a reader at a diff that contains none of the behaviour.

The GB10 end-to-end re-measure of #1365's legs is NOT part of this promotion and
was not taken. `## Gates` calls it the row's closing evidence rather than its
correctness gate, `### Limitations` item 3 discloses it, and `## Stop
conditions` forbids taking a lease for anything in this row's tests. The row is
`DONE` on the gate it declared, with that limitation on the record.

## Owed

Nothing this row leaves behind, and no defect. What stands open is one FOLLOW-ON
issue, and one index row that says something true of the past:

- ISSUE-GH-1541 -- the REFUSING
  request-boundary length guard of `## Defence in depth`, filed by the operator
  at the merge against `67823aee2` and listed here so the index row that the
  closing commit appends for it names an owner. It is a separate row's work and
  a second layer after the algorithmic fix, never a substitute for it, and it is
  NOT a defect this row leaves behind. `## Outcome`'s `### Owed at landing`
  records the debt as paid.
- ISSUE-GH-1365 itself is FIXED, and
  it is named in this list only so that its append-only index row keeps an owner
  without anyone editing it. See the next paragraph, which is the whole
  reconciliation.

**The landed #1365 index row reads as the SYMPTOM, the issue was re-scoped onto
the CAUSE, and the row is left standing because the rule forbids touching it.**
That row -- [`issue-index.md`](../completed/issue-index.md), the `#1365` line -- describes
a reproducible ~4 s TTFT outlier at a fixed request index, found in
`out/bench-20260819T035148Z/`, and it says the cause is deliberately not chased
and is owed under `## Owed` in
[qwen38-27b-bf16-gate.md](qwen38-27b-bf16-gate.md). That was true when it was
written. On 2026-08-19 the issue was re-titled and re-scoped IN PLACE onto the
measured cause, the O(n^2) merge loop, which is the row this spec owns; a
pretoken long enough to make that loop visible is the mechanism behind the
outlier. So the index row now under-describes its own issue.

**It is not repaired, and that is the correct outcome rather than a shortfall.**
AGENTS.md makes [`issue-index.md`](../completed/issue-index.md) append-only: never edit a
row, never delete one. Appending a second `#1365` row is not the escape either,
because `scripts/check-agent-record.py` reports exactly that as `issue #1365
listed twice` -- under `merge=union` a duplicate is what two branches appending
the same issue look like, which is the failure the append-only rule exists to
prevent. Both available edits are gate failures, so the reconciliation is PROSE,
it lives here, and this paragraph is it. A reader who arrives from the index row
alone finds a stale scope; a reader who follows the issue link finds the current
one; and this spec, which the index row's own owner-spec chain reaches, says why
the two differ.

**One thing did move, because it could move without an edit to the index.**
Ownership of #1365's cause is no longer
[qwen38-27b-bf16-gate.md](qwen38-27b-bf16-gate.md)'s. That spec's `## Owed`
entry is rewritten by the closing commit to record that the cause was chased and
fixed here, and this row is where #1365 now lives. Specs are per-row files with
one writer, so moving the ownership costs no shared surface and no append.

## Stop conditions

- Stop and return `NEEDS_DECISION` if token-identifier equivalence fails on any
  corpus entry. A single changed identifier is a correctness regression, not a
  tuning question, and no speed result excuses it.
- Stop and return `NEEDS_DECISION` if the load-time vocabulary refusal rejects
  any committed golden. That would mean the mirror costs us a checkpoint that
  works today, and the trade is a developer decision.
- Stop if the fix needs a change to `src/vllm/tokenizer/pretokenizer.cpp`. The
  pretokenizer mirrors an upstream regex and is out of scope; a design that
  needs it is the wrong design.
- Do not take a GPU or an `rc` lease for anything in `## Tests`. If a gate seems
  to need one, it is measuring the wrong thing.
