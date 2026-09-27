ID: ISSUE-GH-1083
Title: `CleanCaption`'s italic unwrap emulated upstream's `(?!\*)` with a CAPTURED `($|[^*])` (`src/vllm/model_executor/models/minimax_music3_ar.cpp:85,114` @ `a332fb98d`), and a captured group is not a zero-width assertion: consuming the trailing neighbour advanced `regex_replace` past it, so an emphasis span opening within ONE character of the previous close was never examined and the surviving asterisks re-paired ACROSS the intended spans. `*a* *b* *c*` -> `a *b c*` where `_clean_caption` (`encoders.py:72` @ diffusers `c6da9936`) gives `a b c`, and `Warm *lo-fi* *jazzy* keys with a *soft* *brushed* snare` -> `Warm lo-fi *jazzy keys with a soft brushed* snare` — a re-association, not a leftover marker, so the caption handed to the tokenizer is a string upstream would never emit. `encoders.py`'s own header states that whitespace-level prompt changes change the generated audio, so this is a checkpoint-contract break. Invisible to the gate because `markdown_and_tags`, the only golden with italics, carries ONE span per line and the defect needs adjacency. FIXED by porting the trailing side LITERALLY as `(?!\*)` — std::regex's ECMAScript grammar has negative lookahead though not lookbehind, so only the leading `(?<!\*)` stays emulated as `(^|[^*])`, which is safe because that character sits before the span rather than between it and the next one. Differentially over 16 012 inputs against the pinned oracle: 147 mismatches before, 85 after, 0 newly broken. The 85 residual are ONE separate degenerate class, recorded as owed in §10.7 of [`minimax-music3.md`](../../specs/minimax-music3.md) and NOT chased here: a caption that is entirely a horizontal rule, where Python's `re.MULTILINE` lets `\s` span a newline. Found by the [#672](https://github.com/mudler/vllm.cpp/issues/672) oracle sweep
Row: MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation
State: UNKNOWN
Kind: bug
GitHub: 1083
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:311`

### Frozen archive evidence

> | [#1083](https://github.com/mudler/vllm.cpp/issues/1083) | `MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation` | `CleanCaption`'s italic unwrap emulated upstream's `(?!\*)` with a CAPTURED `($\|[^*])` (`src/vllm/model_executor/models/minimax_music3_ar.cpp:85,114` @ `a332fb98d`), and a captured group is not a zero-width assertion: consuming the trailing neighbour advanced `regex_replace` past it, so an emphasis span opening within ONE character of the previous close was never examined and the surviving asterisks re-paired ACROSS the intended spans. `*a* *b* *c*` -> `a *b c*` where `_clean_caption` (`encoders.py:72` @ diffusers `c6da9936`) gives `a b c`, and `Warm *lo-fi* *jazzy* keys with a *soft* *brushed* snare` -> `Warm lo-fi *jazzy keys with a soft brushed* snare` — a re-association, not a leftover marker, so the caption handed to the tokenizer is a string upstream would never emit. `encoders.py`'s own header states that whitespace-level prompt changes change the generated audio, so this is a checkpoint-contract break. Invisible to the gate because `markdown_and_tags`, the only golden with italics, carries ONE span per line and the defect needs adjacency. FIXED by porting the trailing side LITERALLY as `(?!\*)` — std::regex's ECMAScript grammar has negative lookahead though not lookbehind, so only the leading `(?<!\*)` stays emulated as `(^\|[^*])`, which is safe because that character sits before the span rather than between it and the next one. Differentially over 16 012 inputs against the pinned oracle: 147 mismatches before, 85 after, 0 newly broken. The 85 residual are ONE separate degenerate class, recorded as owed in §10.7 of [`minimax-music3.md`](../../specs/minimax-music3.md) and NOT chased here: a caption that is entirely a horizontal rule, where Python's `re.MULTILINE` lets `\s` span a newline. Found by the [#672](https://github.com/mudler/vllm.cpp/issues/672) oracle sweep | bug |

## Resolution

-
