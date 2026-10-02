/*
 * vllm.h — the vllm.cpp stable C ABI (libvllm).
 *
 * ORIGINAL packaging layer — NOT a 1:1 upstream mirror. vLLM ships no C ABI;
 * this header is the llama.cpp-style (see llama.h) handle-based C surface over
 * the C++ LLMEngine, so downstream FFI consumers (LocalAI via purego/cgo, any
 * C/C++ host) can load + drive vllm.cpp without the C++ headers. Recorded as a
 * deviation in .agents/porting-inventory.md §9 (C-ABI packaging, alongside the
 * vt:: runtime and the cpp-httplib transport).
 *
 * ── ABI contract ────────────────────────────────────────────────────────────
 * - PURE C: this header compiles as C (C11) and as C++. It uses only C types —
 *   opaque handle typedefs, POD param/result structs, primitive scalars and
 *   `const char*`. No C++ leaks across the boundary.
 * - NO-THROW: every entry point catches all C++ exceptions internally and maps
 *   them to a `vllm_status`; nothing throws across the ABI. On a non-OK status,
 *   `vllm_last_error()` returns a human-readable message (thread-local).
 * - OWNERSHIP is documented per pointer below. In short: the caller frees
 *   heap `char*` returned in results (via vllm_string_free / vllm_completion_free)
 *   and frees engine handles (via vllm_engine_free); `const char*` returns
 *   (finish_reason, vllm_last_error, vllm_version) point to storage the library
 *   owns and the caller must NOT free.
 * - Scope (M3.5 Task 1): load a model + blocking completion. Streaming callback
 *   + sampled generation land in Task 2; the shared/static lib packaging + CLI
 *   + dlopen smoke test in Task 3.
 */
#ifndef VLLM_H_
#define VLLM_H_

#include <stddef.h>
#include <stdint.h>

/* `bool` in the streaming callback signature: native in C++, needs <stdbool.h>
 * when this header is compiled as C. */
#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── ABI version ──────────────────────────────────────────────────────────────
 * Bumped on any incompatible change to the structs / signatures below. A
 * consumer can compare this against vllm_abi_version() to detect a mismatch
 * between the header it compiled against and the loaded library.
 * v2: structured-output constraint fields appended to vllm_sampling_params
 * (structured_json / structured_regex / structured_choice / structured_grammar /
 * structured_json_object).
 * v3: OpenAI-style chat entry points (vllm_chat / vllm_chat_stream) — chat
 * templating, tool_choice lowering, and streaming tool-call parsing run
 * ENGINE-SIDE.
 * v4: tool_parser field appended to vllm_model_params — selects the tool-call
 * v5: vllm_model_params.reasoning_parser (chain-of-thought split selection).
 * parser for the chat entry points, or AUTO-detects it from the chat template
 * when NULL/empty.
 * v6: vllm_model_params.speculative_config — the speculative-decoding selection
 * as the JSON object vLLM takes (e.g. '{"method":"mtp"}'); NULL/empty disables
 * speculation (the byte-identical default).
 * v7: vllm_model_params.enable_prefix_caching — tri-state APC toggle
 * (0=model default, 1=on, 2=off). The server's --enable-radix-attention is a
 * documented alias for the ON state (RadixAttention is fused into our APC).
 * v8: vllm_sampling_params.logits_processor / logits_processor_user_data — a
 * per-request custom logits-processor callback (vllm_logits_processor) invoked
 * each decode step to modify the request's logits before sampling. Mirrors
 * vLLM's SamplingParams.logits_processors and SGLang's CustomLogitProcessor.
 * NULL => no processor (the byte-identical default).
 * v9: vllm_model_params.max_num_batched_tokens / scheduling_policy /
 * kv_transfer_config, and tokenizer_config_path is now honoured. These close the
 * gap between what the bundled OpenAI server can configure and what an embedder
 * reaches through this ABI: the chunked-prefill token budget, the scheduler
 * admission policy, and the external KV connector (LMCache `lm://`). Every one is
 * inert at its default, so zero-filling the struct growth keeps the pre-v9
 * engine byte-identical. Malformed speculative_config / kv_transfer_config
 * documents now report VLLM_ERR_INVALID_ARGUMENT (a caller error) rather than
 * VLLM_ERR_MODEL_LOAD, matching what v6 already documented.
 * v10: vllm_model_params.enable_jump_forward — a tri-state toggle for
 * jump-forward decoding (0=default/env-resolved-OFF, 1=on, 2=off), the SGLang
 * grammar-speed subset (ENG-SGLANG-BEHAVIOR-FLAG SW3). Appended at the END of
 * vllm_model_params, after the v9 fields. The VT_ENABLE_JUMP_FORWARD env var,
 * when set, overrides this field. 0 (the default) is the byte-identical
 * default, so zero-filling the struct growth keeps a v9 engine byte-identical.
 * Scheduler policy (incl. SGLang's cache-aware LPM) is selected through the v9
 * string field .scheduling_policy = "lpm" — there is no separate int knob.
 * v11: AUDIO TRANSCRIPTION (ARCH-ONE-SURFACE fold #4) — vllm_transcribe /
 * vllm_transcription_params(_default) / vllm_transcription(_free), appended so
 * zero values preserve behaviour. vllm_engine_load now RESOLVES a
 * transcription-only checkpoint (Parakeet CTC/RNNT/TDT, the vLLM
 * SupportsTranscription mirror) into a transcription engine: the text entry
 * points on such a handle report VLLM_ERR_INVALID_ARGUMENT with an actionable
 * message instead of serving, and vllm_transcribe on a TEXT handle does the
 * same. A pre-v11 caller that never loads a Parakeet directory is
 * byte-identical.
 * v12: VIDEO+AUDIO GENERATION (ARCH-ONE-SURFACE ROW 2, MiniMax-H3) — the
 * ratified video slice: an opaque vllm_video_engine loaded from the H3
 * checkpoint set (vllm_video_engine_load/free, vllm_video_model_params +
 * _default), one blocking vllm_video_generate (vllm_video_params + _default)
 * producing a vllm_video_result (frame dir + WAV + geometry + the ffmpeg argv
 * the CALLER may exec — the library spawns nothing), vllm_video_result_free,
 * and the standalone vllm_video_mux_argv(+_free) composer the mux example
 * client uses. Appended so zero values preserve behaviour; a pre-v12 caller
 * that never touches the video symbols is byte-identical. Text and video
 * handles refuse each other's checkpoints LOUDLY: vllm_video_engine_load on a
 * text-model directory names vllm_engine_load, and vllm_engine_load on an H3
 * checkpoint directory keeps failing exactly as at v11 (no config.json).
 * v13: vllm_complete_tokens — blocking completion from a PRE-TOKENIZED prompt
 * (vLLM's TokensPrompt), returning the generated token ids (and optionally the
 * detokenized vllm_completion). The entry point for embedders that manage
 * their own tokenization and for token-exact gates/benchmarks that compare
 * whole token streams against a reference (the Kimi-Linear paged-runner fold
 * battery, ARCH-ONE-SURFACE ROW 7, is the first consumer). Purely additive —
 * no struct changed.
 * v14: vllm_model_params.device — EXPLICIT DEVICE SELECTION (ARCH-ONE-SURFACE
 * fold ROW 8), the mirror of vLLM's DeviceConfig.device names
 * (vllm/config/device.py:13): 0=auto (the byte-identical default — the
 * accelerator-first platform probe that has always selected the queue), 1=cpu
 * (force the CPU queue even on an accelerator build), 2=cuda (require the
 * CUDA platform; when it is absent the load FAILS with VLLM_ERR_MODEL_LOAD —
 * an explicitly named device is never silently substituted, device.py:61-66).
 * Appended at the END of vllm_model_params so a zero-initialized struct keeps
 * the pre-v14 engine byte-identical.
 * v15: vllm_embed / vllm_embedding_result(_free) — EMBEDDINGS through the ONE
 * surface (ARCH-ONE-SURFACE fold ROW 6). An engine loaded from a POOLING
 * (embedding) checkpoint — a directory whose config.json architectures resolve
 * to a pooling registration, e.g. "LlamaModel" (the mirror of vLLM's
 * _EMBEDDING_MODELS registry.py:230 + as_embedding_model adapters.py:230) —
 * embeds text through the SAME registry forward + PoolingRunner engine step
 * the server's /v1/embeddings drives. Text and pooling handles refuse each
 * other's tasks LOUDLY: vllm_complete/vllm_chat on a pooling engine name
 * vllm_embed, and vllm_embed on a text engine names vllm_complete — the
 * SupportsTranscription-refusal precedent (v11) applied to the pooling task.
 * Purely additive — no struct changed; zero values preserve behaviour.
 *
 * v17 — vllm_server_main: RUN THE OPENAI SERVER from the public surface.
 * examples/server was the deepest ONE SURFACE breach (36 internal headers: the
 * engine, the serving layers, metrics, the video and ASR seams), which is why it
 * carried an example-abi-allowlist entry. That construction moved INTO the
 * library (vllm/entrypoints/openai/server_main.h) and the example is now a thin
 * client of this call. argv rather than a params struct is deliberate: the
 * server takes ~57 flags and gains more with every serving feature, and a
 * mirrored C struct would put that churn in the ABI where every field is
 * permanent. The flag surface mirrors vLLM's cli_args.py, which is the real
 * contract. Embedders wanting programmatic control keep the granular entry
 * points. Purely additive. */
/* v18 — THE GENERALIZED VIDEO SEAM (LTX-2.5 L1, .agents/specs/ltx-2-5.md §5,
 * issue #435). The v12 video slice was H3-typed inside; it is now the C face of
 * vllm::multimodal::VideoEngine, an abstract seam with a checkpoint-detected
 * family registry, so a second video family (LTX-2.5) is an additive file
 * rather than a second ABI. Three additions, all APPENDED:
 *   - vllm_video_model_params.family — the family to load. NULL/empty (the
 *     zero value) means DETECT it from the checkpoint, which is what v12
 *     callers already get, since MiniMax-H3 is what a v12 checkpoint set is.
 *     A name that is not registered is REFUSED naming what is registered; it
 *     is never treated as a hint, and a checkpoint no family claims is refused
 *     rather than handed to the only family present — an H3 DiT loaded as LTX
 *     would not fail, it would render noise.
 *   - extra_keys / extra_values / n_extras on vllm_video_model_params and on
 *     vllm_video_params — parallel arrays carrying FAMILY-SPECIFIC settings as
 *     strings, so a new family adds no permanent field to a struct every other
 *     family must then ignore. n_extras 0 (the zero value) is "none".
 *     vllm_video_model_params.partition is now the documented ALIAS for the
 *     load extra "partition"; supplying both with DIFFERENT values is
 *     VLLM_ERR_INVALID_ARGUMENT rather than a silent winner.
 *   - vllm_video_engine_family() — which family a handle actually resolved to,
 *     so detection is visible to a C caller rather than implicit.
 * A v12 caller is byte-identical: it zero-fills the struct growth, so family
 * stays NULL (detect), n_extras stays 0, `partition` keeps its exact v12
 * meaning, and every v12 status/message contract is unchanged (the
 * text-checkpoint refusal still names vllm_engine_load). */
/* v19 — MULTIMODAL INPUT LIMITS on vllm_model_params (ENG-MM-INPUT-PIPELINE
 * wave L2, issue #607). Two APPENDED fields mirroring vLLM's MultiModalConfig
 * (vllm/config/multimodal.py:78,81) as its own two serve flags expose it
 * (vllm/engine/arg_utils.py:555-556,1276-1279,1691-1692):
 *   - vllm_model_params.language_model_only — the flag whose name misleads and
 *     whose docstring does not: it "disables all multimodal inputs by setting
 *     all modality limits to 0" (multimodal.py:78-80). It is sugar; the limits
 *     are the mechanism.
 *   - vllm_model_params.limit_mm_per_prompt — the per-modality input-count
 *     limits, as the SAME JSON object the flag takes ('{"image": 2,
 *     "video": 0}', or the option form '{"video": {"count": 1}}'), following
 *     the v9 precedent that a dict-valued vLLM flag crosses this ABI as its own
 *     JSON rather than as a fixed struct of modalities the ABI would then owe
 *     forever. A malformed document, a negative count, or an unknown option on
 *     one of the three modalities upstream gives an `extra="forbid"` dataclass
 *     (image/video/audio) fails vllm_engine_load with
 *     VLLM_ERR_INVALID_ARGUMENT rather than defaulting — mirroring the pydantic
 *     validation upstream does at parse time (multimodal.py:17-45,212-236).
 *     An unknown option on any OTHER modality is dropped, not refused, because
 *     the BaseDummyOptions it falls back to (:17-21,233) is the one such
 *     dataclass declared without extra="forbid".
 * WHERE THEY BITE, stated exactly, because this contract is permanent. Both
 * fields land on the engine's ONE MultiModalConfig
 * (vllm_engine_load -> EngineParams::multimodal -> LoadedEngine::mm_config()),
 * and that config is what BaseProcessingInfo::ValidateNumItems refuses against.
 * The caller that reaches ValidateNumItems on a live request is the multimodal
 * chat seam, installed by `oai::InstallMultiModalChatSeam(...)`, which since
 * #2475 is the ONE production caller of `set_multimodal_chat_fn` and dispatches
 * on the model's architecture; serving_chat.cpp gates the whole multimodal
 * branch on that seam being set. So an engine loaded with
 * language_model_only=1 answers a multimodal chat request with
 * "At most 0 image(s) may be provided in one prompt." rather than serving it.
 *
 * THIS ABI CARRIES A MULTIMODAL CHAT REQUEST PATH since MODEL-MM-deepseek-v4 W5
 * (issue #2411). It used to have none: `server_main.cpp` was the sole caller of
 * that install, so a chat body carrying an `image_url` part was answered as
 * TEXT with the part silently dropped, and every shipped multimodal capability
 * was reachable only from the bundled HTTP server. `vllm_chat` and
 * `vllm_chat_stream` now install the SAME seam with the SAME context, so the
 * three outcomes a C-ABI caller can get are exactly the server's:
 *   - a TEXT architecture installs nothing and the chat path is byte-identical
 *     to every earlier version;
 *   - a registered multimodal architecture SERVES the image, subject to these
 *     two fields;
 *   - a multimodal architecture with no registered chat seam, or one whose
 *     factory refuses, REFUSES the request with VLLM_ERR_INVALID_ARGUMENT and
 *     a vllm_last_error() naming the architecture and the missing part — never
 *     a silent text answer, because an image request answered as text looks
 *     like a working engine.
 * The image bytes travel in the request JSON itself, as an OpenAI `image_url`
 * content part; there is no new ABI symbol and no struct field for media.
 * The CONTAINER-FORMAT decode (PNG/JPEG -> RGB) and the http(s) fetch are NAMED
 * residuals: the one codec this library ships decodes raw RGB
 * (`image/x-raw-rgb`) and refuses everything else by name. A request that hits
 * either is a caller error and is reported as one.
 * That is pinned behaviourally by tests/capi/test_capi.cpp ("capi: a multimodal
 * chat request is ANSWERED or REFUSED, never silently served as text"), so this
 * paragraph cannot silently become false.
 * NO ABI VERSION BUMP CARRIES THIS: no symbol and no struct field changed, so
 * a client compiled against v26 links and runs unchanged. What changed is what
 * an engine DOES with a request it already accepted, which is why the change is
 * recorded in this paragraph and pinned by that test rather than by a number.
 * The memory win upstream also gets from zero limits (skipping the vision tower
 * weights, interfaces.py:293) is NOT in this version — it is wave L3, and until
 * it lands and is MEASURED this field must not be described as freeing VRAM.
 * Appended at the END of vllm_model_params, so a zero-initialized v18 struct is
 * byte-identical: language_model_only 0 (off) and limit_mm_per_prompt NULL (no
 * limits configured => the 999-per-modality default, multimodal.py:331-333). */
/* v20 — SPEECH AND MUSIC GENERATION (.agents/specs/minimax-music3.md §4.1 W6,
 * issue #672). The C face of vllm::multimodal::SpeechEngine, the seam the
 * IndexTTS-2.5 lane landed and MiniMax-Music3 is the first family to implement:
 * an opaque vllm_speech_engine loaded from a checkpoint SET
 * (vllm_speech_engine_load/free, vllm_speech_model_params + _default), one
 * blocking vllm_synthesize (vllm_speech_params + _default) producing a
 * vllm_speech_result — the float waveform AND the RIFF/WAVE bytes, so a server
 * hands a client a playable file without a second encoder — plus
 * vllm_speech_result_free, and three interrogations of the loaded handle
 * (vllm_speech_engine_family / _sample_rate / _requires_reference_audio).
 *
 * WHY THE HANDLE ANSWERS QUESTIONS. `sample_rate` is the family's NATIVE rate
 * (44100 stereo for Music3, 22050 mono for IndexTTS-2.5) and never a resampled
 * one, so the caller decides whether to resample.
 * `requires_reference_audio` exists so a server can REFUSE a request before
 * staging tens of gigabytes: true for a family with no text-only synthesis,
 * false for one conditioned on text alone.
 *
 * A MUSIC family takes TWO texts. `lyrics` and `description` are separate
 * fields rather than one `text` behind a separator, because upstream runs a
 * DIFFERENT normalizer over each (encoders.py:54-91); a one-utterance family
 * keeps using `text` and ignores them. Every generation control is inert at its
 * zero value, which selects the family's own default — EXCEPT `guidance_scale`,
 * which carries `has_guidance_scale` beside it (the vllm_video_params.has_seed
 * precedent) because 0 IS A LEGAL GUIDANCE SCALE and a 0-means-default sentinel
 * would make the unconditional branch unreachable.
 *
 * Purely additive: no struct changed, no existing signature moved, and a pre-v19
 * caller that never touches a speech symbol is byte-identical. A directory no
 * speech family claims is refused NAMING every family that was tried, because
 * the wrong family does not fail, it renders noise. */
/* v21 — vllm_speech_model_params.device + vllm_speech_engine_device, THE SPEECH
 * LANE'S DEVICE SELECTOR (issue #672). v20 could only run a speech family on the
 * CPU: MiniMax-Music3's queue was a compile-time constant, so a 28.5 GB music
 * model was a host-only model whatever hardware the box had.
 *
 * 0 = CPU, 1 = the accelerator this build resolves. That is the
 * vllm_video_model_params.device spelling, NOT vllm_model_params.device's
 * 0=auto/1=cpu/2=cuda: the text engine's `auto` selects an accelerator when one
 * exists, whereas a speech family's CPU path is what its correctness gates were
 * taken on, so a ZERO-VALUE CALLER MUST KEEP GETTING THE CPU ARM. Device 1 on a
 * build with no accelerator backend, or on a PARTIAL backend that declines the
 * family's architecture, is REFUSED BY NAME at load rather than substituted.
 *
 * vllm_speech_engine_device reports what was GRANTED, not what was asked for —
 * two different facts, and a benchmark that conflates them measures the CPU arm
 * twice.
 *
 * Appended at the END of vllm_speech_model_params, so a zero-initialized v20
 * struct keeps the CPU engine byte-identical. WHAT DEVICE 1 MOVES for Music3 is
 * the 8.6B language model and nothing else yet; the RVQ depth decoder and the
 * acoustic half are still host reference loops (see docs/FEATURES.md). */
/* v22 — vllm_model_params.mmproj_path, THE SECOND GGUF FILE (row
 * `LOAD-GGUF-MMPROJ`, issue #821). A GGUF multimodal model ships as two files:
 * the language `.gguf` and a `clip`-architecture `mmproj-*.gguf` carrying the
 * vision tower. Before this field the ABI could name only the first, so a GGUF
 * multimodal checkpoint could be loaded only as a text model and the projector
 * had nowhere to arrive.
 *
 * The spelling mirrors llama.cpp's user-facing `--mmproj`, which is the flag
 * every holder of these artifacts already types. It is EXPLICIT by design:
 * auto-discovery of a sibling `mmproj*.gguf` is deliberately not implemented,
 * because a directory holding two unrelated models would then silently fuse
 * them and the failure would be a wrong-shaped model rather than an error.
 *
 * SCOPE, and it carries the same weight as the field: this loads the tower and
 * hands it to the engine, and since MODEL-MM-deepseek-v4 W5 (issue #2411)
 * `vllm_chat` / `vllm_chat_stream` can FEED it — see the v19 note above for the
 * three outcomes a multimodal chat request can get. The path is the request
 * JSON's own `image_url` content part; `vllm_generate` still takes text only.
 * A two-file vehicle whose second file was NOT named refuses an image request
 * at install rather than inside the engine's busy loop, so the omission costs
 * one refusal naming `--mmproj` rather than every later request.
 *
 * Appended at the END of vllm_model_params, so a zero-initialized v21 struct is
 * byte-identical: NULL/empty means no projector, which is every load that
 * existed before. */

/* v23 — vllm_video_last_phase_log, WHERE A RENDER SPENT ITS WALL (issue #1010,
 * row LTX25-DEVICE-RESIDENCY stage W0).
 *
 * Before this the LTX-2.5 render path emitted one line per render, so a render
 * that took two hours could not say which of its phases took them. Every
 * attempt to act on that profile has since failed on a measurement defect: the
 * evidence existed only on a host that stopped answering (#1040), the 1731 s
 * phase #1087 measures is unnamed, and #1024's GPU-zero window is known to be
 * neither the denoise nor the decode.
 *
 * A completed generation now WRITES a phase table — `phase-log.json`, beside
 * the frames, on the shipped default and behind no flag — carrying, per phase,
 * a monotone timestamp, a duration, a peak host byte count and a peak device
 * byte count, plus the wall and the `unaccounted_seconds` the named phases did
 * NOT cover. That residue is emitted rather than smeared over the phases,
 * because a table whose parts do not add up has a phase nobody named.
 *
 * PURELY ADDITIVE: no struct changed and no existing signature moved, which is
 * why the path is a QUERY on the handle rather than a new member on
 * vllm_video_result. Growing an OUTPUT struct is the one append a caller cannot
 * absorb by zero-initializing, since the library writes the field with its own
 * sizeof. A family that emits no table returns NULL.
 *
 * IT IS v23 AND IT WAS WRITTEN AS v22. `vllm_model_params.mmproj_path` (issue
 * #821) took v22 on `main` while this branch was open, and both features had
 * merged into a tree that defined 22 twice. Two additions under one version is
 * not a textual conflict a merge tool resolves — the number is the caller's only
 * question ("does the library I loaded have this?"), and one that answers yes
 * for a feature the build does not carry is worse than no version at all. So
 * this one moved. The dependent sites moved with it: the >= floor in
 * `tests/capi/test_capi.cpp`, the table row and the version line in
 * `docs/USAGE.md`, and the surface row in `docs/FEATURES.md`. */
/* v24 — vllm_model_params.kv_cache_dtype, KV-CACHE STORAGE DTYPE (row
 * `KV-FP8` W6, fork issue #7). Mirrors vLLM CacheConfig.cache_dtype and its
 * `--kv-cache-dtype` flag (config/cache.py:19-36,76). NULL or "auto" (the
 * zero value) uses the model dtype and is byte-identical to before; "fp8" or
 * "fp8_e4m3" stores 1-byte fp8-e4m3 K/V. Appended at the END of
 * vllm_model_params, so a zero-initialized v23 struct is byte-identical. */
/* v25 — vllm_engine_spec_acceptance, SPECULATIVE ACCEPTANCE TELEMETRY (row
 * `SPEC-DFLASH2`, issue #2832). A pure READ of three counters the engine
 * already increments; it computes nothing and changes no decision. Added
 * because the DFlash2 speed gate drives `examples/cli`, a pure client of this
 * header, so the acceptance both reference engines publish had no route out of
 * ours and a 0.928x deficit could not be attributed to execution speed or to
 * acceptance. A NEW ENTRY POINT, so every existing struct and call is
 * byte-identical. */
/* v26 — vllm_model_params.disable_sliding_window, the MODEL-LEVEL SLIDING-WINDOW
 * KILL SWITCH (row `ENG-ATTENTION-WINDOW` W3, issue #2388). Mirrors vLLM's
 * `ModelConfig.disable_sliding_window` (`vllm/config/model.py:248` @ pin
 * `5559679229`), which is ONE model-agnostic field rather than a per-architecture
 * toggle. It replaces `VT_GEMMA2_SLIDING` and `VT_GEMMA3_SLIDING`, two
 * kernel-internal environment knobs that gave this control to two of the five
 * model families that have a window and to none of the other three. A new
 * TRAILING field on a struct whose callers zero-initialize, and 0 is the existing
 * default, so a zero-initialized v25 struct is byte-identical. */
/* v27 — vllm_gliner_ner / vllm_ner_result(_free), ZERO-SHOT NER through the
 * ONE surface (row `MODEL-GLINER25`). A GLiNER2.5 engine (architecture
 * "BoundaryExtractor", a pooling model) runs the full NER pipeline — DeBERTa
 * v2 encoder → boundary head → candidate decoder — and returns extracted
 * entities with character offsets. The entry point is additive: a
 * non-GLiNER2 engine is refused by name, and every existing struct and call is
 * byte-identical. */
/* v28 — vllm_systemone / vllm_score: SystemOne decision API and cua-s1 score
 * API through the ONE surface. A kev/laya engine (architecture "KevModel" or
 * "LayaModel") runs the decision forward and returns a JSON response string;
 * a cua-s1 engine (architecture "CuaS1Forms") runs the score forward. Both are
 * additive: non-matching architectures are refused by name, and every existing
 * struct and call is byte-identical. */
/* v29 — vllm_decide: unifies vllm_systemone and vllm_score into one
 * category-level entry point. The engine architecture determines the pipeline:
 * KevModel/LayaModel runs the decision forward, CuaS1Forms runs the score
 * forward. Non-matching architectures are refused by name. Every existing
 * struct and call is byte-identical. */
#define VLLM_ABI_VERSION 30

/* ── Export macro ─────────────────────────────────────────────────────────────
 * Marks the symbols that make up the stable ABI. Default visibility now; Task 3
 * hides everything else (visibility=hidden + version script) so only VLLM_API
 * symbols are exported from libvllm.so. */
#ifndef VLLM_API
#if defined(_WIN32)
#define VLLM_API __declspec(dllexport)
#else
#define VLLM_API __attribute__((visibility("default")))
#endif
#endif

/* ── Status codes ─────────────────────────────────────────────────────────────
 * Every fallible entry point returns one of these. VLLM_OK == 0. On any error,
 * vllm_last_error() (thread-local) carries the detail. */
typedef enum vllm_status {
  VLLM_OK = 0,
  VLLM_ERR_INVALID_ARGUMENT = 1, /* null/out-of-range argument */
  VLLM_ERR_MODEL_LOAD = 2,       /* config/tokenizer/weights load failed */
  VLLM_ERR_RUNTIME = 3,          /* generation / engine runtime failure */
  VLLM_ERR_UNKNOWN = 4           /* a non-std::exception escaped internally */
} vllm_status;

/* ── Opaque handles ───────────────────────────────────────────────────────────
 * The engine handle owns the whole C++ stack (LLMEngine + Scheduler + runner +
 * KV cache + processors + tokenizer). Created by vllm_engine_load, destroyed by
 * vllm_engine_free. W2 completion submissions are thread-safe and share one
 * AsyncLLM scheduler; destruction still requires all request handles freed. */
typedef struct vllm_engine vllm_engine;

/* A non-blocking request returned by vllm_request_submit. The engine MUST
 * outlive every request created from it. Free with vllm_request_free. */
typedef struct vllm_request vllm_request;

/* ── Model / load parameters ──────────────────────────────────────────────────
 * POD. Populate with vllm_model_params_default() then override. All `const char*`
 * fields are borrowed for the duration of the vllm_engine_load call only — the
 * library copies what it needs; the caller retains ownership. */
typedef struct vllm_model_params {
  /* Supported model directory or GGUF file. Required. */
  const char* model_path;
  /* Optional override for tokenizer_config.json; NULL => <model_path>/
   * tokenizer_config.json. Honoured since ABI v9: it selects the file the chat
   * entry points read `chat_template` from, mirroring the server's
   * --tokenizer-config. Ignored for a .gguf model_path, whose template comes
   * from the GGUF `tokenizer.chat_template` metadata. */
  const char* tokenizer_config_path;
  /* KV-cache block size (tokens per block). <= 0 => 32.
     MUST be a multiple of 16: the attention backends' get_kv_cache_shape
     refuses any other value, so a non-multiple throws from vllm_engine_load. */
  int32_t block_size;
  /* KV-cache block count OVERRIDE (vLLM num_gpu_blocks_override). > 0 pins the
   * pool to exactly this many blocks. <= 0 => AUTO: the pool is sized by the
   * v16 sizing knobs below (kv_cache_memory_bytes, then gpu_memory_utilization),
   * falling back to 256 when neither is set. A zero-initialized struct therefore
   * still yields 256 blocks — behaviour is unchanged from before the sizing
   * knobs existed. */
  int32_t num_blocks;
  /* Max sequence length. <= 0 => config.max_position_embeddings. */
  int32_t max_model_len;
  /* Max concurrent sequences the scheduler admits. <= 0 => 8. */
  int32_t max_num_seqs;
  /* ── Tool-call parser (ABI v4) ──────────────────────────────────────────────
   * Selects the parser that turns the model's raw tool-call output into
   * structured tool_calls for the chat entry points (vllm_chat /
   * vllm_chat_stream). NULL or "" => AUTO: the parser is detected from the
   * model's chat template at the first chat call (a template that wraps calls
   * in <tool_call> selects "hermes"; the fallback when nothing is detected is
   * also "hermes"). A non-empty value MUST name a registered parser (e.g.
   * "hermes", "qwen3"); an unknown name fails the first chat call with
   * VLLM_ERR_INVALID_ARGUMENT. Borrowed for the vllm_engine_load call only. */
  const char* tool_parser;
  /* ── Reasoning parser (ABI v5) ──────────────────────────────────────────────
   * Selects the parser that splits chain-of-thought from user-visible content
   * for the chat entry points; reasoning streams as the `reasoning` field of
   * the chat chunks and is stripped BEFORE tool-call parsing. NULL or "" =>
   * AUTO: detected from the model's chat template at the first chat call
   * ("[THINK]" selects "mistral", "<think>" selects "deepseek_r1"; nothing
   * detected => reasoning parsing disabled). "none" => force-disabled. Any
   * other value MUST name a registered reasoning parser (e.g. "deepseek_r1",
   * "mistral", "minimax_m2", "step3", "olmo3"); an unknown name fails the
   * first chat call with VLLM_ERR_INVALID_ARGUMENT. Borrowed for the
   * vllm_engine_load call only. */
  const char* reasoning_parser;
  /* ── Speculative-decoding config (ABI v6) ──────────────────────────────────
   * The JSON object vLLM's --speculative-config takes, e.g.
   * '{"method":"mtp"}' or '{"method":"mtp","num_speculative_tokens":3}'.
   * For "mtp", num_speculative_tokens is the draft DEPTH. It defaults to the
   * checkpoint's mtp_num_hidden_layers, which is 1 on both gate checkpoints, and
   * a deeper value loops the single head autoregressively. Depth cannot change
   * the emitted tokens under greedy sampling, so it is a throughput knob.
   * NULL or "" => speculation DISABLED, the byte-identical default engine.
   * A malformed document or unsupported method fails vllm_engine_load with
   * VLLM_ERR_INVALID_ARGUMENT. Only MTP is supported today, on the Qwen3.5/3.6
   * checkpoints that ship an mtp.* head (safetensors only). Borrowed for the
   * vllm_engine_load call only. */
  const char* speculative_config;
  /* ── Automatic prefix caching (ABI v7) ─────────────────────────────────────
   * Tri-state toggle for automatic prefix caching (APC), mirroring vLLM's
   * --[no-]enable-prefix-caching resolution against the model default:
   *   0 => MODEL DEFAULT (the byte-identical default): dense full-attention
   *        models default ON, hybrid / attention-free models default OFF;
   *   1 => force ON;
   *   2 => force OFF.
   * The server's --enable-radix-attention flag is a documented ALIAS for state
   * 1 and --disable-radix-attention for state 2: SGLang's "RadixAttention" is
   * functionally fused into our block-hash APC (see
   * .agents/specs/sglang-radixattention.md §1), so the radix flag switches
   * exactly this field — there is no distinct radix code path. Any value other
   * than 0/1/2 fails vllm_engine_load with VLLM_ERR_INVALID_ARGUMENT. */
  int32_t enable_prefix_caching;
  /* ── Chunked-prefill token budget (ABI v9) ─────────────────────────────────
   * The per-step token budget the scheduler admits (vLLM's
   * --max-num-batched-tokens). <= 0 => the bounded PER-ARCH default: 2048 flat
   * for a dense arch (vLLM's own DEFAULT_MAX_NUM_BATCHED_TOKENS), 8192/4096 for
   * MoE depending on max_num_seqs. Raising it lets more prefill land in one
   * step at the cost of decode latency behind it (and, on the hybrid archs, a
   * larger per-step GDN activation — the reason the default does not scale with
   * max_num_seqs). Constrained to >= max_num_seqs by the scheduler. */
  int32_t max_num_batched_tokens;
  /* ── Scheduler admission policy (ABI v9) ───────────────────────────────────
   * One of "fcfs" (arrival order, the default), "priority" ((priority,
   * arrival_time) ordering — requests then carry a `priority` field), or "lpm"
   * (SGLang's cache-aware longest-prefix-match ordering; output-neutral, and it
   * degrades to fcfs when prefix caching is off since there is no cache to match
   * against). NULL or "" => "fcfs". An unknown name fails vllm_engine_load with
   * VLLM_ERR_INVALID_ARGUMENT. Borrowed for the call only. */
  const char* scheduling_policy;
  /* ── External KV connector / LMCache (ABI v9) ──────────────────────────────
   * The JSON object vLLM's --kv-transfer-config takes, selecting an external
   * KV-cache connector, e.g.
   *   {"kv_connector":"LMCacheConnector","kv_role":"kv_both",
   *    "kv_connector_extra_config":{"host":"127.0.0.1","port":65432}}
   * NULL or "" => NO connector == the byte-identical default engine. A malformed
   * document, an unknown role, or a connector name that is not registered in
   * this build fails vllm_engine_load with VLLM_ERR_INVALID_ARGUMENT; a
   * connector whose worker half cannot move bytes on this device is refused at
   * engine construction (VLLM_ERR_MODEL_LOAD). `kv_role` is REQUIRED whenever
   * `kv_connector` is set. Borrowed for the call only. See docs/KV-OFFLOAD.md. */
  const char* kv_transfer_config;
  /* ── Weight offload / OffloadConfig (ABI v21) ──────────────────────────────
   * The JSON object vLLM's OffloadConfig takes, selecting the WEIGHT-offload
   * backend. This is a DIFFERENT subject from kv_transfer_config above, which
   * offloads KV blocks: this one offloads model WEIGHTS. Example:
   *   {"offload_backend":"uva","uva":{"cpu_offload_gb":10,
   *                                   "cpu_offload_params":["experts"]}}
   * NULL or "" => the default inert config: no backend selected, nothing
   * offloaded, the byte-identical engine. Mirrors vllm/config/offload.py at the
   * pin: `offload_backend` is one of "auto" (default; prefetch when
   * offload_group_size > 0, else uva when cpu_offload_gb > 0, else nothing),
   * "uva" or "prefetch". A malformed document, an unknown backend name, a
   * wrong-typed field, or a config that fails upstream's validator
   * (offload_num_in_group > offload_group_size, or offload_prefetch_step < 1
   * when prefetch is enabled) fails vllm_engine_load with
   * VLLM_ERR_INVALID_ARGUMENT.
   *
   * THE MIRRORED KEYS ARE ACCEPTED BUT NOT YET ACTED ON: `ENG-WEIGHT-OFFLOAD`
   * W0b wires the `offload_backend`/`uva`/`prefetch` half end to end (CLI -> ABI
   * -> EngineParams) and validates it; the offloader that would MOVE a weight to
   * host RAM is W2/W5. So that half parses, validates and is recorded, and no
   * weight moves yet. It is spelled out here rather than left silent because a
   * user who sets cpu_offload_gb and sees no memory change deserves to know it
   * was accepted and is inert, not ignored. This sentence covers ONLY those three
   * keys.
   *
   * THE `vllm_cpp` KEY IS LIVE, and it moves weights (ABI v21, row
   * `ENG-RESIDENCY-CONFIG`). The same string carries a vllm.cpp-ORIGINAL object
   * for the tier BELOW upstream's: weights borrowed out of the GGUF file mapping
   * rather than copied to host RAM, plus a bounded host slot cache for routed
   * expert slices. Upstream has no disk tier, so there is nothing to mirror and
   * the key names itself. Schema, every field optional and an absent field
   * meaning unchanged:
   *   {"vllm_cpp":{"mmap":{"enabled":bool,"prefault":bool},
   *                "expert_stream":{"enabled":bool,"slots":int,
   *                                 "slot_bytes":int},
   *                "device_fit":{"weight_budget_bytes":int}}}
   * Precedence per field is environment variable > this document > built-in
   * default, so an exported VT_GGUF_MMAP / VT_GGUF_PREFAULT / VT_MOE_EXPERT_STREAM
   * / VT_MOE_EXPERT_STREAM_SLOTS / VT_MOE_EXPERT_STREAM_SLOT_BYTES /
   * VT_DEVICE_WEIGHT_BUDGET_BYTES still wins; the engine prints one line on
   * stderr naming what it installed, plus a second line
   * naming the variables that override it when there are any. The engine acts on it
   * during weight load, so it must be installed before then, which vllm_engine_load
   * does. Loading a SECOND engine in one process is legal: an absent field means
   * unchanged, so a partial document is merged over what is installed rather than
   * replacing it, and only a document that would CHANGE a decision the process has
   * already taken — the streaming answer, or the slot store's geometry — is refused.
   *
   * REFUSALS ADDED WITH THAT KEY, all VLLM_ERR_INVALID_ARGUMENT before any model
   * I/O: an UNKNOWN key anywhere in the document — a misspelled top-level key
   * (`{"vllm-cpp":...}` with a hyphen, `{"uvaa":...}`), a misspelled key inside
   * `vllm_cpp` or inside any of its THREE objects
   * (`{"vllm_cpp":{"mmapp":...}}`,
   * `{"vllm_cpp":{"device_fit":{"weight_budget":0}}}`), and a misspelled key
   * inside the mirrored `uva` or `prefetch` object
   * (`{"uva":{"cpu_offload_GB":10}}`); a wrong-typed field; a non-positive
   * `slots` or `slot_bytes`; and a NEGATIVE `weight_budget_bytes`. A typo is
   * refused rather than defaulted because a silently disabled residency tier, or
   * a budget the operator believes is set, is met as an out-of-memory kill rather
   * than as an error. Upstream refuses one too: every vLLM config dataclass
   * carries `extra="forbid"`. The four legal top-level keys are
   * `offload_backend`, `uva`, `prefetch` and `vllm_cpp`.
   *
   * `weight_budget_bytes` is the ONE field of the six that ACCEPTS `0`, and the
   * asymmetry is the reason the key exists. It is a BUDGET, not a size: `0` is
   * the documented spelling of "suppress the load-time device-fit refusal and get
   * the late failure back", because the fit check reads a zero budget as UNKNOWN
   * and decides nothing — exactly what `VT_DEVICE_WEIGHT_BUDGET_BYTES=0` already
   * means. `slots` and `slot_bytes` are sizes, and a slot count that silently
   * became its default is a cache the operator does not have, so those two keep
   * refusing `0`. Only a NEGATIVE budget is refused, and the message says "must
   * not be negative" rather than "must be positive". See docs/USAGE.md.
   * Borrowed for the call only. */
  const char* offload_config;
  /* ── Jump-forward decoding (ABI v10) ───────────────────────────────────────
   * Tri-state toggle for jump-forward decoding — the SGLang grammar-speed
   * behavior (ENG-SGLANG-BEHAVIOR-FLAG SW3): when the structured-output grammar
   * has a token-unique forced continuation, emit it without a model step. Only
   * the PROVABLY byte-identical token-unique subset is jumped (see
   * .agents/specs/sglang-enablement.md for the residual):
   *   0 => DEFAULT: OFF unless the VT_ENABLE_JUMP_FORWARD env var turns it on
   *        (the byte-identical default);
   *   1 => force ON (the env var, if set, still overrides);
   *   2 => force OFF (likewise env-overridable).
   * The VT_ENABLE_JUMP_FORWARD env var, WHEN SET, always wins over this field
   * (mirrors the VT_ASYNC_SCHED convention). Any value other than 0/1/2 fails
   * vllm_engine_load with VLLM_ERR_INVALID_ARGUMENT. Scheduler policy (incl.
   * SGLang's cache-aware LPM) is a SEPARATE knob — the v9 string field
   * .scheduling_policy = "lpm", not an int here. */
  int32_t enable_jump_forward;
  /* ── Sliding window (ABI v26) ──────────────────────────────────────────────
   * Disables the MODEL-LEVEL sliding window for every model that has one,
   * mirroring vLLM's ModelConfig.disable_sliding_window
   * (vllm/config/model.py:248). Upstream's own docstring settles what happens to
   * a model without one: "If the model does not support sliding window, this
   * argument is ignored."
   *   0 => DEFAULT: sliding window enabled (upstream's `= False`, and the
   *        byte-identical behaviour of every release before v26).
   *   1 => disable it.
   *   2 => explicitly enable it.
   * Any other value fails vllm_engine_load with VLLM_ERR_INVALID_ARGUMENT.
   * A per-layer window still wins over this switch, which is upstream's
   * precedence (attention.py:204-236): the flag clears the MODEL-level value. */
  int32_t disable_sliding_window;
  /* ── Device selection (ABI v14) ────────────────────────────────────────────
   * Which device the text-generation engine serves on, mirroring vLLM's
   * DeviceConfig.device names (vllm/config/device.py:13 — Device =
   * Literal["auto", "cuda", "cpu", ...]):
   *   0 => AUTO (the byte-identical default): the accelerator-first platform
   *        probe that has always selected the queue (CUDA first, CPU
   *        fallback; on other builds ROCm/XPU/Vulkan/Metal probe in between);
   *   1 => CPU: force the CPU queue even when an accelerator is available;
   *   2 => CUDA: require the CUDA platform. When it is absent in this
   *        build/process the load FAILS with VLLM_ERR_MODEL_LOAD and a
   *        message naming the device — an explicitly named device is NEVER
   *        silently replaced by another (mirror of vLLM assigning an explicit
   *        device verbatim, device.py:61-66).
   * 0 must stay auto so a zero-initialized struct preserves pre-v14 behaviour;
   * the cpu-before-accelerator value order follows the v12 precedent
   * (vllm_video_model_params.device: 0 cpu, 1 the resolved accelerator)
   * shifted by the auto slot.
   * Any other value fails vllm_engine_load with VLLM_ERR_INVALID_ARGUMENT. */
  int32_t device;
  /* ── KV-pool sizing (ABI v16) ──────────────────────────────────────────────
   * Two knobs mirroring vLLM's CacheConfig, resolved together with num_blocks
   * above in the order: num_blocks (override) > kv_cache_memory_bytes > the
   * gpu_memory_utilization profile path. num_blocks retains its meaning as the
   * explicit override (num_gpu_blocks_override); leave it 0 to reach these.
   *
   * gpu_memory_utilization: the fraction of free device memory the whole engine
   * may consume (weights + activations + KV), mirroring vLLM
   * CacheConfig.gpu_memory_utilization (default 0.92). 0.0 => 0.92. This is the
   * DEFAULT sizing path in vLLM, but it needs a device profile run that measures
   * the non-KV footprint; that profile run is not implemented yet
   * (ROAD-V1-MEM M3), so until it lands a struct with both other knobs unset
   * still falls back to the historical 256-block default — the zero-initialized
   * struct's behaviour is unchanged from pre-v16.
   *
   * Since #1165 that fallback is no longer SILENT: a value > 0.0 here, with
   * num_blocks and kv_cache_memory_bytes both unset, prints one warning per
   * vllm_engine_load naming the block count that resolved instead and the two
   * knobs that do bind today. Accepting a fraction and sizing nothing without
   * saying so left callers believing they had sized the pool.
   *
   * Note that vllm_model_params_default() pre-fills this field with 0.92, so a
   * caller who never touched it is indistinguishable from one who chose 0.92
   * and does get the warning. That is deliberate: on this ABI there is no
   * "flag not typed" state, and a struct carrying 0.92 into an engine that
   * ignores it is exactly the case the warning is for. To opt out, spell the
   * unset sentinel: set the field to 0.0. */
  double gpu_memory_utilization;
  /* kv_cache_memory_bytes: an ABSOLUTE KV-pool size in bytes. When > 0 it sizes
   * the block count directly (num_blocks = kv_cache_memory_bytes / bytes-per-
   * block, computed from the model's own KV geometry) and IGNORES
   * gpu_memory_utilization, mirroring vLLM CacheConfig.kv_cache_memory_bytes
   * (cache.py:182,189). 0 => unset. A budget smaller than a single KV block
   * fails vllm_engine_load with VLLM_ERR_INVALID_ARGUMENT. */
  int64_t kv_cache_memory_bytes;
  /* ── Multimodal input limits (ABI v19) ────────────────────────────────────
   * The mirror of vLLM's --language-model-only / --limit-mm-per-prompt
   * (arg_utils.py:555-556,1276-1279,1691-1692) over MultiModalConfig
   * (multimodal.py:78,81). These are ONE mechanism, not two: the flag is
   * defined as "disables all multimodal inputs by setting all modality limits
   * to 0" (:78-80), so it is checked BEFORE the map and an explicit non-zero
   * entry does not survive it (get_limit_per_prompt, :321-336).
   *
   * language_model_only: 0 => off (the zero value, byte-identical to pre-v19);
   * nonzero => every modality limit resolves to 0 on this engine's
   * MultiModalConfig. On the OPENAI-SERVER path that is a refusal — HTTP 400
   * "At most 0 <modality>(s) may be provided in one prompt." — because the
   * server installs the multimodal chat seam that reaches ValidateNumItems. On
   * this ABI's own vllm_chat there is no multimodal request to refuse yet, so
   * the field configures the engine without changing any C-ABI call's result;
   * see the v19 note in the version log above for exactly what a C-ABI caller
   * gets today. The tower-skip memory win upstream also produces is not here
   * either (wave L3). */
  int32_t language_model_only;
  /* limit_mm_per_prompt: the per-modality maximum input-item count, as the same
   * JSON object the flag takes. NULL/empty (the zero value) => no limit
   * configured => 999 per modality (:331-333), NOT zero — an empty map is "no
   * limits", not "nothing allowed". Accepted spellings, all upstream's own
   * (:87-96,212-236):
   *     {"image": 16, "video": 2}                        (count only)
   *     {"video": {"count": 1, "num_frames": 32}}        (with options)
   *     {"image": 16, "video": {"count": 1}}             (mixed)
   * The option keys are validated exactly as upstream's per-modality
   * dataclasses do (video: num_frames/width/height, image: width/height, audio:
   * length; each an integer > 0; anything else on those three is refused,
   * `extra="forbid"`, :24,33,41) and then DROPPED: they size dummy inputs for
   * memory profiling, which this engine does not do, and only `count` feeds the
   * limit (:335). A modality outside those three is upstream's bare
   * BaseDummyOptions (:233), which has no `extra="forbid"`, so its unknown keys
   * are dropped rather than refused — mirrored, not invented.
   * Invalid JSON, a non-object document, a negative count, or a refused option
   * per the paragraph above fails vllm_engine_load with
   * VLLM_ERR_INVALID_ARGUMENT. Borrowed for the call only. */
  const char* limit_mm_per_prompt;
  /* ── The `clip` multimodal projector (ABI v22) ────────────────────────────
   * Path to the SECOND GGUF file — `mmproj-*.gguf`, `general.architecture` =
   * `clip` — beside a `.gguf` model path. NULL or empty (the zero value) means
   * no projector, which is byte-identical to pre-v22.
   *
   * Refused BY NAME with VLLM_ERR_MODEL_LOAD — the code every FromModelDir
   * failure reports, exactly as an absent named device does — when: the model path is
   * not a `.gguf` (a safetensors checkpoint carries its tower in its own
   * shards); the file is not a `clip` projector; its `clip.projector_type` is
   * not one this build loads; or it carries only the first half of the
   * temporal patch embedding, which cannot be completed without inventing the
   * other half. Every one of those fires BEFORE the tokenizer and before any
   * language-model weight byte is read. Borrowed for the call only. */
  const char* mmproj_path;
  /* ── KV-cache storage dtype (ABI v24) ──────────────────────────────────────
   * Mirrors vLLM CacheConfig.cache_dtype and its `--kv-cache-dtype` flag
   * (config/cache.py:19-36,76). NULL or "auto" (the default) uses the model
   * dtype and is byte-identical to before this field existed; "fp8" or
   * "fp8_e4m3" stores 1-byte fp8-e4m3 K/V, halving the bytes per KV block.
   * Borrowed for the call only. */
  const char* kv_cache_dtype;
} vllm_model_params;

/* ── Custom logits processor (ABI v8) ─────────────────────────────────────────
 * A host callback the sampler invokes ONCE PER DECODE STEP for the request,
 * BEFORE sampling, to inspect the tokens generated so far and modify the logits.
 * It runs at vLLM's non-argmax-invariant logits-processor stage — after
 * allowed_token_ids / bad_words / min_tokens / logit_bias, before penalties —
 * mirroring SamplingParams.logits_processors (and satisfying SGLang's
 * custom_logit_processor capability with the same callback).
 *   - token_ids / n_token_ids: the request's generated output token ids so far
 *     (n_token_ids == 0 on the first decode step). BORROWED — valid only for the
 *     duration of the call; copy them if you need to retain them.
 *   - logits / vocab_size: a MUTABLE view of THIS request's logits row, a
 *     contiguous float[vocab_size]. Edit it in place (add a bias, mask tokens to
 *     -inf, force a token to +inf, ...); the edited row is what the sampler then
 *     samples from. Greedy decoding takes the argmax of the edited row.
 *   - user_data: the opaque pointer registered in
 *     vllm_sampling_params.logits_processor_user_data, round-tripped unchanged.
 * The callback is C code and MUST NOT throw across the ABI boundary. */
typedef void (*vllm_logits_processor)(const int32_t* token_ids,
                                      int32_t n_token_ids, float* logits,
                                      int32_t vocab_size, void* user_data);

/* ── Sampling parameters ──────────────────────────────────────────────────────
 * POD mirror of the T0 fields of vllm::SamplingParams. Populate with
 * vllm_sampling_params_default() then override; a zero-initialized struct is NOT
 * valid (repetition_penalty must be > 0). temperature <= 0 selects greedy.
 * `stop` is borrowed for the duration of the vllm_complete call; the library
 * copies the strings. */
typedef struct vllm_sampling_params {
  float temperature;         /* randomness; <= eps => greedy (argmax). */
  float top_p;               /* nucleus cutoff in (0, 1]. */
  int32_t top_k;             /* top-k; 0 (or -1) => all tokens. */
  float min_p;               /* min token prob relative to the max, in [0, 1]. */
  int32_t max_tokens;        /* max tokens to generate; <= 0 => unbounded. */
  uint64_t seed;             /* RNG seed; used only when has_seed != 0. */
  int32_t has_seed;          /* 0 => unseeded (nondeterministic sampling). */
  float presence_penalty;    /* new-token presence penalty. */
  float frequency_penalty;   /* new-token frequency penalty. */
  float repetition_penalty;  /* repetition penalty; must be > 0 (default 1). */
  int32_t min_tokens;        /* min tokens before EOS/stop can end generation. */
  int32_t ignore_eos;        /* 0/1: keep generating past EOS. */
  const char* const* stop;   /* array of stop strings (may be NULL). */
  int32_t n_stop;            /* number of entries in `stop`. */
  /* ── Structured output (ABI v2) ─────────────────────────────────────────────
   * POD mirror of vllm::StructuredOutputsParams (the same constraints the
   * OpenAI response_format layer lowers to). AT MOST ONE of the five
   * constraints below may be set (non-NULL / non-zero); more than one is
   * rejected with an error status (upstream's exactly-one rule). The strings
   * are borrowed for the duration of the call; the library copies them.
   * Enforcement is engine-side per-step constrained decoding (a grammar
   * bitmask over the logits), on every completion entry point.
   *   - structured_json: a JSON-Schema document, as a JSON string; the output
   *     is constrained to instances of that schema.
   *   - structured_regex: the output matches the regular expression.
   *   - structured_choice / n_structured_choice: the output is exactly one of
   *     the given strings.
   *   - structured_grammar: a GBNF (llama.cpp-style) grammar.
   *   - structured_json_object: != 0 => some valid JSON object (schema-free
   *     "JSON mode"). */
  const char* structured_json;          /* JSON-Schema string, or NULL. */
  const char* structured_regex;         /* regular expression, or NULL. */
  const char* const* structured_choice; /* array of choices (may be NULL). */
  int32_t n_structured_choice;          /* entries in structured_choice. */
  const char* structured_grammar;       /* GBNF grammar, or NULL. */
  int32_t structured_json_object;       /* 0/1: schema-free JSON-object mode. */
  /* ── Custom logits processor (ABI v8) ───────────────────────────────────────
   * Per-request host callback invoked each decode step to modify the logits
   * before sampling (see vllm_logits_processor). NULL => no processor (the
   * byte-identical default: the sampler path is unchanged). The function pointer
   * and user_data are borrowed for the duration of the generation call; the
   * library copies the pair. */
  vllm_logits_processor logits_processor;   /* NULL => no custom processor. */
  void* logits_processor_user_data;         /* opaque; passed to the callback. */
} vllm_sampling_params;

/* ── Completion result ────────────────────────────────────────────────────────
 * Filled by vllm_complete. OWNERSHIP:
 *   - text: heap-allocated, NUL-terminated. The CALLER owns it and must free it
 *     via vllm_completion_free(out) (or vllm_string_free(out->text)). Set to
 *     NULL on any non-OK status.
 *   - finish_reason: borrowed pointer into library-owned static storage
 *     ("stop" / "length" / "abort" / ...). The caller must NOT free it and must
 *     not use it after the string literal's program lifetime (it is static, so
 *     it is always valid). NULL if the request did not finish.
 *   - prompt_tokens / completion_tokens: token counts (prompt vs generated). */
typedef struct vllm_completion {
  char* text;
  const char* finish_reason;
  int32_t prompt_tokens;
  int32_t completion_tokens;
} vllm_completion;

/* ── Defaults ─────────────────────────────────────────────────────────────────
 * Return structs pre-filled with the upstream SamplingParams / sane load
 * defaults. Use these as the base and override fields, so future struct growth
 * stays source-compatible. */
VLLM_API vllm_model_params vllm_model_params_default(void);
VLLM_API vllm_sampling_params vllm_sampling_params_default(void);

/* ── Lifecycle ────────────────────────────────────────────────────────────────
 * vllm_engine_load: build the full engine stack from `params->model_path`.
 * On success returns VLLM_OK and stores a handle in *out (caller frees via
 * vllm_engine_free). On failure returns a VLLM_ERR_* code, sets vllm_last_error(),
 * and leaves *out == NULL. `params` and `out` must be non-NULL. */
VLLM_API vllm_status vllm_engine_load(const vllm_model_params* params,
                                      vllm_engine** out);

/* Destroy an engine handle and everything it owns. NULL is a no-op. */
VLLM_API void vllm_engine_free(vllm_engine* engine);

/* ── Speculative-decoding acceptance (ABI v25) ────────────────────────────────
 * The counters the engine already keeps for its own speculative decode, read
 * back so a benchmark driven through this ABI can say WHY a throughput number
 * is what it is. CUMULATIVE over the handle's whole life: a caller that wants a
 * per-request or per-leg figure subtracts two reads.
 *
 * THE UNITS ARE STATED HERE BECAUSE THE TWO REFERENCE ENGINES DISAGREE ON ONE
 * OF THEM, and conflating the two is a one-token-per-step error:
 *
 *   drafts_proposed        draft tokens the target VERIFIED. Mirrors vLLM's
 *                          `vllm:spec_decode_num_draft_tokens`.
 *   drafts_accepted        draft tokens the rejection sampler ACCEPTED.
 *                          EXCLUDES the bonus/replacement token a verify step
 *                          always emits, which is what vLLM's
 *                          `vllm:spec_decode_num_accepted_tokens` counts.
 *   drafted_request_steps  (REQUEST, STEP) PAIRS that carried at least one
 *                          draft — NOT forward passes. The engine increments it
 *                          once per request per step, so a single verify step
 *                          over a batch of 8 drafted requests adds 8. It equals
 *                          the number of verify forwards ONLY at
 *                          `max_num_seqs=1`.
 *
 * THE FIELD IS NAMED FOR WHAT IT COUNTS, because a reader who divides by a
 * "verify step" count that is really a request-step count is wrong by the batch
 * size — and the concurrent rungs are exactly where this instrument gets
 * pointed. The unit is nevertheless the SAME one both reference engines
 * publish, so the comparison is exact and needs no correction: vLLM's
 * `vllm:spec_decode_num_drafts` is incremented per request per step
 * (`vllm/v1/spec_decode/metrics.py:41-42`, called from
 * `vllm/v1/core/sched/scheduler.py:1691` under the same `num_draft_tokens > 0`
 * predicate at `:2445`), and SGLang's `spec_verify_ct` likewise
 * (`python/sglang/srt/managers/schedule_batch.py:961`).
 *
 * So `drafts_accepted / drafts_proposed` is the acceptance RATE with the bonus
 * token OUT of it, and
 * `1 + drafts_accepted / drafted_request_steps` is vLLM's
 * `mean_acceptance_length` EXACTLY (`vllm/v1/spec_decode/metrics.py:114`), with
 * the bonus token IN it.
 *
 * IT IS NOT SGLang's `accept_length`, and the two must not be quoted as one
 * number. SGLang divides `completion_tokens` by `spec_verify_ct`
 * (`python/sglang/srt/managers/tokenizer_manager.py:2363`) and that numerator
 * includes the request's PREFILL token, which came from no verify step at all.
 * The difference is about one token per request — roughly +0.07 on a 64-token
 * request over ~15 steps, which is a seventh of the 4.67-versus-4.23 spread
 * this instrument exists to explain.
 *
 * All three read 0 on an engine that never speculated, which is a true answer
 * and not an error. Returns VLLM_ERR_INVALID_ARGUMENT for a NULL argument or a
 * handle with no text-generation path (*out is zeroed first either way). */
typedef struct vllm_spec_acceptance {
  int64_t drafts_proposed;
  int64_t drafts_accepted;
  int64_t drafted_request_steps;
} vllm_spec_acceptance;

VLLM_API vllm_status vllm_engine_spec_acceptance(const vllm_engine* engine,
                                                 vllm_spec_acceptance* out);

/* ── Completion (blocking) ────────────────────────────────────────────────────
 * Run a single blocking completion for `prompt` with `params`, filling *out.
 * Returns VLLM_OK on success (out->text is a heap string the caller frees), or a
 * VLLM_ERR_* code (out is zeroed, out->text == NULL, vllm_last_error() set).
 * `engine`, `prompt`, `params` and `out` must be non-NULL. */
VLLM_API vllm_status vllm_complete(vllm_engine* engine, const char* prompt,
                                   const vllm_sampling_params* params,
                                   vllm_completion* out);

/* ── Pre-tokenized completion (ABI v13) ───────────────────────────────────────
 * Run a single blocking completion for a PRE-TOKENIZED prompt (vLLM's
 * TokensPrompt): tokenization is skipped and generation starts from
 * `prompt_tokens` directly.
 *   - prompt_tokens / n_prompt_tokens: the prompt token ids, BORROWED for the
 *     duration of the call. n_prompt_tokens must be > 0.
 *   - out_tokens / max_out_tokens: caller-owned buffer that receives the
 *     GENERATED token ids; *n_out_tokens is set to the number written
 *     (<= max_out_tokens). Generation length is bounded by params->max_tokens
 *     as usual — a smaller buffer only truncates what is REPORTED, never the
 *     generation. out_tokens may be NULL iff max_out_tokens == 0.
 *   - out: OPTIONAL (may be NULL). When non-NULL it is filled exactly like
 *     vllm_complete (detokenized text owned by the caller, finish_reason,
 *     token counts).
 * Returns VLLM_OK on success; a VLLM_ERR_* code with vllm_last_error() set on
 * failure (*n_out_tokens zeroed, out zeroed when supplied). */
VLLM_API vllm_status vllm_complete_tokens(
    vllm_engine* engine, const int32_t* prompt_tokens, int32_t n_prompt_tokens,
    const vllm_sampling_params* params, int32_t* out_tokens,
    int32_t max_out_tokens, int32_t* n_out_tokens, vllm_completion* out);

/* ── Streaming completion (M3.5 Task 2) ───────────────────────────────────────
 * vllm_token_callback: invoked once per engine-step delta for the streaming
 * request, then once more with finished == true to carry the finish.
 *   - delta_text: the incremental text produced since the previous call, as a
 *     NUL-terminated, well-formed UTF-8 C string. It is BORROWED — valid ONLY
 *     for the duration of the call; copy it if you need to retain it. The final
 *     finished == true call may carry an empty delta_text ("").
 *   - finished: true on the terminal call (the request ended: EOS / stop /
 *     length / abort). The callback is not invoked again for this request.
 *   - user_data: the opaque pointer passed to vllm_complete_stream, round-tripped
 *     unchanged (e.g. an accumulator the callback appends to).
 * RETURN false to STOP generation early: the library aborts the in-flight
 * request (tears it down so the engine stays usable) and returns VLLM_OK. Return
 * true to keep generating. The callback is C code and MUST NOT throw across the
 * ABI (any C++ exception it raises is caught and mapped to a status). */
typedef bool (*vllm_token_callback)(const char* delta_text, bool finished,
                                    void* user_data);

/* Run a single streaming completion for `prompt` with `params`, invoking `cb`
 * per delta (see vllm_token_callback). BLOCKING: drives the engine loop to a
 * natural finish, an early stop (cb returned false), or an error before
 * returning. Returns VLLM_OK on success (including an early stop), or a
 * VLLM_ERR_* code (vllm_last_error() set). `engine`, `prompt`, `params` and `cb`
 * must be non-NULL; `user_data` may be NULL. Sampled generation (temperature > 0
 * with a seed) is supported and deterministic for a fixed seed. */
VLLM_API vllm_status vllm_complete_stream(vllm_engine* engine,
                                          const char* prompt,
                                          const vllm_sampling_params* params,
                                          vllm_token_callback cb,
                                          void* user_data);

/* ── Non-blocking streaming requests (async-serving W2) ─────────────────────
 * Submit returns after validation/enqueue. A library-owned delivery thread
 * invokes `cb` for each RequestOutput delta while the shared AsyncLLM engine
 * continues batching other requests. `out` receives an owned request handle.
 * The callback/user_data borrow follows vllm_complete_stream's contract.
 * The engine must outlive the request handle. */
VLLM_API vllm_status vllm_request_submit(
    vllm_engine* engine, const char* prompt,
    const vllm_sampling_params* params, vllm_token_callback cb,
    void* user_data, vllm_request** out);

/* Abort an in-flight request. Idempotent; its delivery thread exits after the
 * terminal abort output is consumed. */
VLLM_API vllm_status vllm_request_cancel(vllm_request* request);

/* Wait for callback delivery to finish and return its terminal status. On an
 * error, vllm_last_error() on the WAITING thread receives the request error.
 * Do not call wait/free from that request's own callback. */
VLLM_API vllm_status vllm_request_wait(vllm_request* request);

/* Non-blocking completion probe (false for NULL). */
VLLM_API bool vllm_request_done(const vllm_request* request);

/* Request-owned diagnostic string. Empty while the request is still running;
 * after vllm_request_done returns true (or wait returns), valid until
 * vllm_request_free. Never NULL. */
VLLM_API const char* vllm_request_error(const vllm_request* request);

/* Cancel if needed, join the delivery thread, and destroy the handle. NULL is
 * a no-op. The parent engine must still be alive. Must not be called from the
 * request's own callback; use cancel there and free from another thread. */
VLLM_API void vllm_request_free(vllm_request* request);

/* ── Chat completions (ABI v3) ────────────────────────────────────────────────
 * OpenAI-style chat entry points over the SAME engine handle. The heavy
 * lifting runs ENGINE-SIDE, exactly like the bundled OpenAI server:
 *   - request_json is one OpenAI /v1/chat/completions request object
 *     (messages, tools, tool_choice, temperature, top_p, max_tokens, stop,
 *     ...). The `model` and `stream` fields are ignored (the handle's model
 *     serves; streaming is selected by the entry point).
 *   - The chat template is applied by the engine's renderer. It is resolved
 *     at vllm_engine_load: <model_dir>/tokenizer_config.json `chat_template`,
 *     or the GGUF `tokenizer.chat_template` metadata for a .gguf model; when
 *     neither exists, a plain "<role>: <content>" join is used.
 *   - request_json may carry `chat_template_kwargs`, an object of extra Jinja
 *     variables for that template, the same field vLLM takes. A key nobody
 *     sends is not a template variable at all, so `{% if enable_thinking is
 *     undefined %}` answers true and the checkpoint's own default applies;
 *     this ABI sets no default of its own. A key that names something the
 *     renderer supplies is REFUSED, not honoured (`messages`, `tools`,
 *     `chat_template`, `tokenize`): the call returns
 *     VLLM_ERR_INVALID_ARGUMENT and vllm_last_error() says which key, so no
 *     request can replace the conversation the caller passed in `messages`.
 *   - tools + tool_choice lower to the engine's structural-tag DECODE
 *     constraint: `auto` is LAZY (the ENGINE decides when a tool engages —
 *     text is unconstrained until the model emits the tool trigger, then the
 *     call is grammar-constrained); `required`/named force a call; `none`
 *     disables. Tool-call output is parsed engine-side (streaming-stateful
 *     Hermes-style parser) into structured tool_calls deltas.
 *
 * vllm_chat: BLOCKING non-streaming completion. On VLLM_OK, *out_response_json
 * is a heap NUL-terminated ChatCompletionResponse JSON object (choices with
 * message.content / message.tool_calls, finish_reason "stop"/"length"/
 * "tool_calls", usage) that the CALLER frees via vllm_string_free. On error,
 * *out_response_json is NULL and vllm_last_error() is set (a malformed
 * request_json maps to VLLM_ERR_INVALID_ARGUMENT). */
VLLM_API vllm_status vllm_chat(vllm_engine* engine, const char* request_json,
                               char** out_response_json);

/* vllm_chat_stream: BLOCKING streaming completion. `cb` receives ONE OpenAI
 * chat.completion.chunk JSON object per invocation in delta_text (no SSE
 * framing): first the role chunk, then content and/or tool_calls delta chunks
 * as the engine-side parser emits them, then the finish chunk; after the last
 * chunk the callback is invoked once more with finished == true and an empty
 * delta. Returning false from the callback aborts the in-flight request
 * (VLLM_OK is still returned). The borrow/threading contract of
 * vllm_token_callback applies unchanged. */
VLLM_API vllm_status vllm_chat_stream(vllm_engine* engine,
                                      const char* request_json,
                                      vllm_token_callback cb, void* user_data);

/* ── Audio transcription (ABI v11) ────────────────────────────────────────────
 * The transcription slice of the ONE-SURFACE fold: an engine loaded from a
 * transcription-only checkpoint (a directory whose config.json architectures
 * resolve to the Parakeet CTC / RNN-T / TDT family — the mirror of vLLM's
 * SupportsTranscription protocol) transcribes audio to text through the SAME
 * library pipeline the bundled server's /v1/audio/transcriptions route and the
 * parakeet-transcribe example drive. Greedy decode, 16 kHz mono input (the
 * extractor refuses to resample, mirroring the HF feature extractor).
 *
 * Input is EXACTLY ONE of:
 *   - audio_path: a 16-bit PCM mono RIFF/WAVE file path;
 *   - pcm + n_samples + sample_rate: a mono float32 waveform in [-1, 1).
 * Setting both or neither is VLLM_ERR_INVALID_ARGUMENT. All pointers are
 * borrowed for the duration of the call. */
typedef struct vllm_transcription_params {
  const char* audio_path;  /* WAV path, or NULL when pcm is used. */
  const float* pcm;        /* mono f32 samples, or NULL when audio_path is used. */
  int64_t n_samples;       /* number of entries in pcm. */
  int32_t sample_rate;     /* pcm sampling rate in Hz (16000). */
} vllm_transcription_params;

/* One transcription result. OWNERSHIP: the caller frees text + token_ids via
 * vllm_transcription_free(out) (text may also go through vllm_string_free).
 *   - text: heap NUL-terminated transcript. NULL when has_text == 0 (the
 *     checkpoint ships no tokenizer.json: ids only, exactly like the
 *     pre-fold example).
 *   - token_ids / n_token_ids: the decoded token ids (CTC: after the greedy
 *     collapse; transducer: blanks and the start token dropped).
 *   - has_text: 0/1 — whether `text` was produced. */
typedef struct vllm_transcription {
  char* text;
  int32_t* token_ids;
  int32_t n_token_ids;
  int32_t has_text;
} vllm_transcription;

/* Zero-initialized params (no input selected — the caller sets exactly one). */
VLLM_API vllm_transcription_params vllm_transcription_params_default(void);

/* Run a blocking transcription on a transcription-capable engine handle,
 * filling *out. Returns VLLM_OK on success; VLLM_ERR_INVALID_ARGUMENT for a
 * text-generation handle (use the completion/chat entry points there) or a bad
 * input selection; VLLM_ERR_RUNTIME when the audio cannot be read/decoded
 * (non-PCM16-mono WAV, wrong sample rate) or the forward fails — matching
 * vllm_complete's convention. On any non-OK status *out is zeroed and
 * vllm_last_error() carries the detail. */
VLLM_API vllm_status vllm_transcribe(vllm_engine* engine,
                                     const vllm_transcription_params* params,
                                     vllm_transcription* out);

/* Free the owned members of a transcription result and zero the struct. The
 * struct itself is caller storage. NULL is a no-op. */
VLLM_API void vllm_transcription_free(vllm_transcription* out);


/* ── Speaker diarization (ABI v30) ───────────────────────────────────────────
 * When the library is built with VLLM_CPP_WITH_DIARIZATION=ON (the default),
 * a second engine handle can be loaded from a Nemotron-3-Diarization GGUF
 * file. The diarization engine identifies who spoke when in a mono 16 kHz
 * audio stream. It is independent of the ASR (Parakeet) engine — the two
 * can be combined via vllm_transcribe_and_diarize.
 *
 * When VLLM_CPP_WITH_DIARIZATION=OFF, every function below returns
 * VLLM_ERR_INVALID_ARGUMENT with a "not compiled in" message. */

/* One speaker segment. */
typedef struct vllm_speaker_segment {
  int32_t speaker;   /* 0-indexed speaker ID */
  float start;       /* seconds from audio start */
  float end;
} vllm_speaker_segment;

/* Diarization result. OWNERSHIP: free with vllm_diarization_free. */
typedef struct vllm_diarization {
  vllm_speaker_segment* segments;
  int32_t n_segments;
} vllm_diarization;

/* Load a diarization GGUF file. Returns NULL on error
 * (vllm_last_error carries the detail). */
VLLM_API vllm_engine* vllm_diarization_load(const char* gguf_path);

/* Diarize a WAV file. Returns VLLM_OK on success. */
VLLM_API vllm_status vllm_diarize_path(vllm_engine* diar_engine,
                                       const char* wav_path,
                                       vllm_diarization* out);

/* Diarize raw PCM (mono float32, 16 kHz). */
VLLM_API vllm_status vllm_diarize_pcm(vllm_engine* diar_engine,
                                      const float* pcm, int64_t n_samples,
                                      int32_t sample_rate,
                                      vllm_diarization* out);

/* Free a diarization result. NULL is a no-op. */
VLLM_API void vllm_diarization_free(vllm_diarization* out);


/* ── Speaker-attributed ASR (ABI v30) ───────────────────────────────────────
 * Combined transcription + diarization: runs both models on the same audio
 * and merges word timestamps with speaker segments. The ASR engine must be
 * a Parakeet checkpoint; the diarization engine must be a GGUF loaded with
 * vllm_diarization_load. */

typedef struct vllm_speaker_utterance {
  int32_t speaker;
  char* text;
  float start;
  float end;
  float conf;
} vllm_speaker_utterance;

typedef struct vllm_sas_result {
  vllm_speaker_utterance* utterances;
  int32_t n_utterances;
} vllm_sas_result;

/* Run combined ASR + diarization on a WAV file. */
VLLM_API vllm_status vllm_transcribe_and_diarize(
    vllm_engine* asr_engine, vllm_engine* diar_engine,
    const char* wav_path,
    vllm_sas_result* out);

/* Run combined ASR + diarization on raw PCM. */
VLLM_API vllm_status vllm_transcribe_and_diarize_pcm(
    vllm_engine* asr_engine, vllm_engine* diar_engine,
    const float* pcm, int64_t n_samples, int32_t sample_rate,
    vllm_sas_result* out);

/* Free a SAS result. Each utterance's .text is freed, then the array. */
VLLM_API void vllm_sas_result_free(vllm_sas_result* out);


/* ── Embeddings (ABI v15) ─────────────────────────────────────────────────────
 * The embeddings/pooling slice of the ONE-SURFACE fold: an engine loaded from
 * a POOLING (embedding) checkpoint — config.json architectures resolving to a
 * pooling registration such as "LlamaModel" — turns text into L2-normalized
 * embedding vectors through the SAME registry forward + PoolingRunner engine
 * step the bundled server's /v1/embeddings drives (task=embed, LAST-token
 * pooling: the mirror of vLLM's as_embedding_model conversion). Loading such a
 * checkpoint uses the ordinary vllm_engine_load; the handle then serves ONLY
 * the embedding entry point (the text/chat entry points refuse, naming this
 * one, and vice versa on a text handle). */

/* One embedding batch result. OWNERSHIP: `values` is library-allocated; free
 * via vllm_embedding_result_free(out). Row-major: embedding i occupies
 * values[i*dim .. (i+1)*dim). */
typedef struct vllm_embedding_result {
  float* values;          /* n_embeddings * dim floats, row-major */
  int32_t n_embeddings;   /* == the number of input texts */
  int32_t dim;            /* the model's hidden size */
  int32_t prompt_tokens;  /* total input tokens (the OpenAI usage mirror) */
} vllm_embedding_result;

/* Embed n_texts NUL-terminated UTF-8 strings on a pooling-capable engine
 * handle, filling *out (one embedding per text, input order). BLOCKING; the
 * texts are tokenized with the checkpoint's tokenizer and each prompt runs one
 * engine prefill + pool step. Returns VLLM_OK on success;
 * VLLM_ERR_INVALID_ARGUMENT for a text-generation handle (use vllm_complete /
 * vllm_chat there), a NULL texts/out, an n_texts <= 0, or a NULL entry in
 * texts; VLLM_ERR_RUNTIME when tokenization or the forward fails. On any
 * non-OK status *out is zeroed and vllm_last_error() carries the detail. */
VLLM_API vllm_status vllm_embed(vllm_engine* engine,
                                const char* const* texts, int32_t n_texts,
                                vllm_embedding_result* out);

/* Free the owned members of an embedding result and zero the struct. The
 * struct itself is caller storage. NULL is a no-op. */
VLLM_API void vllm_embedding_result_free(vllm_embedding_result* out);


/* ── Zero-shot NER (ABI v27, MODEL-GLINER25) ────────────────────────────────
 * The NER slice of the ONE-SURFACE fold: the SAME library pipeline the bundled
 * server's /v1/ner endpoint drives (vllm::Gliner2NerInference), reachable by
 * any embedder. A GLiNER2.5 engine (architecture "BoundaryExtractor") runs the
 * DeBERTa v2 encoder, the boundary head, and the candidate decoder, returning
 * extracted entities with token and character offsets. */

/* One extracted entity. OWNERSHIP: label and text are library-allocated; free
 * the containing vllm_ner_result via vllm_ner_result_free. */
typedef struct vllm_ner_entity {
  char* label;          /* entity type / query name (NUL-terminated, library-alloc) */
  char* text;           /* surface text from the input (NUL-terminated, library-alloc) */
  int32_t char_start;   /* character start offset in the input (inclusive) */
  int32_t char_end;     /* character end offset in the input (exclusive) */
  int32_t token_start;  /* token start index (inclusive) */
  int32_t token_end;    /* token end index (exclusive) */
  float confidence;     /* sigmoid probability of the span score */
} vllm_ner_entity;

/* NER result: an array of entities. OWNERSHIP: entities and their label/text
 * strings are library-allocated; free via vllm_ner_result_free. */
typedef struct vllm_ner_result {
  vllm_ner_entity* entities;  /* n_entities entities, or NULL when none */
  int32_t n_entities;         /* number of entities (may be 0) */
} vllm_ner_result;

/* Run zero-shot NER on `text` with `n_labels` entity types on a GLiNER2.5
 * engine handle. `labels` is an array of NUL-terminated entity-type strings
 * (e.g. {"person", "organization", "location"}). `threshold` is the minimum
 * sigmoid probability to keep (default 0.5); `max_width` is the maximum span
 * width in tokens (default 12, 0 = use the model default). BLOCKING. Returns
 * VLLM_OK on success; VLLM_ERR_INVALID_ARGUMENT for a non-GLiNER2 engine, a
 * NULL text/labels/out, or n_labels <= 0; VLLM_ERR_RUNTIME on forward failure.
 * On any non-OK status *out is zeroed and vllm_last_error() carries the detail. */
VLLM_API vllm_status vllm_gliner_ner(vllm_engine* engine, const char* text,
                                     const char* const* labels, int32_t n_labels,
                                     float threshold, int32_t max_width,
                                     vllm_ner_result* out);

/* Free the owned members of a NER result and zero the struct. NULL is a no-op. */
VLLM_API void vllm_ner_result_free(vllm_ner_result* out);


/* ── Decision + Score (ABI v29, MODEL-KEV / MODEL-LAYA / MODEL-CUA-S1-FORMS) ──
 *
 * vllm_decide runs the decision or scoring pipeline depending on the engine
 * architecture. A kev engine ("KevModel") or laya engine ("LayaModel") runs
 * the decision forward and returns the /v1/systemone JSON response, as do
 * "ClmModel", "SpanExtractor", "XorModel" and "NimbleModel". A Tev1 engine
 * ("Tev1Model", a generation engine) answers the same /v1/systemone request
 * by scoring each question's answer letters through its own scheduler, so the
 * call may run concurrently with vllm_chat / vllm_complete on the same handle;
 * its usage.output_tokens counts the one token sampled per question. A cua-s1
 * engine ("CuaS1Forms") runs the score forward and returns the /v1/score JSON
 * response. Other architectures are refused by name.
 *
 * request_json is the raw body of the POST /v1/systemone or POST /v1/score
 * request. On success *out_json is a heap-allocated NUL-terminated string
 * (caller frees with vllm_decide_free). On any non-OK status *out_json is NULL
 * and vllm_last_error() carries detail. */
VLLM_API vllm_status vllm_decide(vllm_engine* engine,
                                  const char* request_json,
                                  char** out_json);

/* Free a string returned by vllm_decide. NULL is a no-op. */
VLLM_API void vllm_decide_free(char* json);


/* ── Video+audio generation (ABI v12, MiniMax-H3) ────────────────────────────
 * The video slice of the ONE-SURFACE fold: the SAME library pipeline the
 * bundled server's /v1/videos routes and the minimax-h3-gen example drive
 * (vllm::multimodal::MiniMaxH3VideoEngine), reachable by any embedder.
 *
 * A video engine is loaded from the H3 checkpoint SET (the release ships the
 * DiT, the text encoder and the two VAEs as separate artifacts), not from one
 * model directory — which is why this is a separate handle from vllm_engine.
 * Loading a TEXT checkpoint here fails with VLLM_ERR_MODEL_LOAD naming
 * vllm_engine_load; loading an H3 checkpoint with vllm_engine_load keeps
 * failing exactly as before v12 (its directory carries no config.json).
 *
 * THE PROCESS BOUNDARY: the library writes frames + WAV and COMPOSES the
 * ffmpeg argv, and spawns nothing. The caller execs mux_argv to get an MP4. */
typedef struct vllm_video_engine vllm_video_engine;

/* All paths; NULL/empty means "not supplied". Borrowed for the load call. */
typedef struct vllm_video_model_params {
  const char* dit_path;       /* GGUF | NVFP4 safetensors | bf16 shard dir */
  const char* encoder_path;   /* H3-Encoder GGUF or bf16 shard dir */
  const char* tokenizer_path; /* tokenizer.json (with an encoder) */
  const char* video_vae_path;
  const char* video_vae_config_path;
  const char* audio_vae_path;
  const char* audio_vae_config_path;
  /* Fallback conditioning when no encoder is supplied: rows of text_dim,
   * little-endian f32 (the pre-fold --prompt-embeds file). */
  const char* prompt_embeds_path;
  /* The served checkpoint PARTITION: "fl2va" (serves t2va+fl2va) or "ref2va".
   * Community GGUF/NVFP4 files strip the release metadata and the two DiTs
   * are byte-structurally identical, so it must be DECLARED; NULL/empty makes
   * every generate refuse with the guidance (the #77 guard). */
  const char* partition;
  /* 0 is the CPU; 1 is THE ACCELERATOR THIS BUILD RESOLVES, through the
   * platform seam (CurrentPlatform + TryGetBackend +
   * supports_model_architecture), never the enum value 1. It is therefore CUDA
   * on a CUDA build and refused BY NAME on a build with no accelerator backend,
   * or one whose partial backend declines this architecture (#659, #660). The
   * ABI value is unchanged; what it means was never "cuda". */
  int32_t device;
  int32_t dequant_bf16; /* 0 keep-quant, 1 dequant/stream bf16 */
  int32_t fp4_resident; /* NVFP4+cuda: keep FP4 packed, Marlin W4A16 GEMM */
  /* ── v18 additions (the generalized seam) ─────────────────────────────────
   * The model family to load, e.g. "minimax-h3". NULL/empty DETECTS it from
   * what the checkpoint holds; an unregistered name is refused naming the
   * registered ones. Never a hint — a checkpoint no family claims is refused,
   * because the wrong family does not fail, it renders noise. */
  const char* family;
  /* FAMILY-SPECIFIC load settings as parallel arrays of n_extras borrowed
   * key/value strings (both arrays must hold n_extras non-NULL entries).
   * `partition` above is the documented alias for the key "partition";
   * supplying both with DIFFERENT values is VLLM_ERR_INVALID_ARGUMENT. */
  const char* const* extra_keys;
  const char* const* extra_values;
  int32_t n_extras; /* 0 => none */
} vllm_video_model_params;

typedef struct vllm_video_params {
  const char* prompt;        /* encoded when the engine has an encoder */
  int32_t width, height;     /* <= 0 => aspect-derived default canvas */
  int32_t num_frames;        /* <= 1 => per-task default */
  int32_t steps;             /* <= 0 => H3 default (50) */
  uint64_t seed;
  int32_t has_seed;          /* 0 => the fixed default noise streams */
  const char* first_frame;   /* fl2va keyframes: binary PPM (P6) paths; */
  const char* last_frame;    /* pin frame 0 / the last frame OF THE OUTPUT */
  const char* ref_image;     /* ref2va: ONE whole reference image (PPM) */
  const char* ref_video;     /* ref2va: a DIRECTORY of frame_%06d.ppm */
  const char* ref_audio;     /* ref2va: a 16-bit PCM WAV path */
  float noise_aug;           /* keyframe pinning strength; <= 0 => 1.0 */
  /* Where frame_%06d.ppm + audio.wav land (created if absent). REQUIRED. */
  const char* output_dir;
  /* v18: FAMILY-SPECIFIC per-generation settings, same parallel-array shape as
   * the load-time extras. Every family refuses a key it does not know rather
   * than ignoring it. 0 => none.
   *   MiniMax-H3: none.
   *   LTX-2.5:    "image_crf" — the H.264 CRF an image conditioning is
   *               re-compressed at. Only "0" is served; an LTX-2.5 checkpoint
   *               RESOLVES 18 when this is absent and the codec round trip is
   *               unported, so leaving it out refuses BY NAME rather than
   *               rendering. "0" is upstream-legal and out of distribution;
   *               see docs/USAGE.md. No ABI change was needed for it, which is
   *               what this parallel-array shape exists for. */
  const char* const* extra_keys;
  const char* const* extra_values;
  int32_t n_extras;
} vllm_video_params;

/* One finished generation. OWNERSHIP: every member is library-allocated;
 * free the whole struct's members via vllm_video_result_free(out). */
typedef struct vllm_video_result {
  char* frame_dir;  /* holds frame_%06d.ppm */
  char* audio_path; /* 16-bit PCM WAV */
  int32_t frame_count, width, height, fps, sample_rate;
  /* The ffmpeg argv the CALLER may exec to mux <output_dir>/video.mp4;
   * argv[0] is "ffmpeg" (substitute a custom binary before exec'ing).
   * mux_argv[mux_argc] is NULL, so it is execvp-ready. */
  char** mux_argv;
  int32_t mux_argc;
} vllm_video_result;

/* Zero-initialized params (every path NULL, cpu, keep-quant). */
VLLM_API vllm_video_model_params vllm_video_model_params_default(void);
/* Default generation params (all defaults resolved engine-side). */
VLLM_API vllm_video_params vllm_video_params_default(void);

/* Load the H3 checkpoint set and stage its weights once. On VLLM_OK, *out is
 * a handle the caller frees via vllm_video_engine_free. On error, *out is
 * NULL and vllm_last_error() carries the detail (a text-model directory maps
 * to VLLM_ERR_MODEL_LOAD naming vllm_engine_load). */
VLLM_API vllm_status vllm_video_engine_load(const vllm_video_model_params* params,
                                            vllm_video_engine** out);
VLLM_API void vllm_video_engine_free(vllm_video_engine* engine);

/* v18: the family this handle RESOLVED to ("minimax-h3", ...) — the answer to
 * "what did detection decide?". Points at storage the library owns for the
 * lifetime of the handle; the caller must NOT free it. NULL engine => NULL. */
VLLM_API const char* vllm_video_engine_family(const vllm_video_engine* engine);

/* Run one BLOCKING generation, filling *out. Serialized per engine handle.
 * VLLM_ERR_INVALID_ARGUMENT for a missing output_dir / illegal reference
 * combination; VLLM_ERR_RUNTIME when the pipeline refuses (undeclared or
 * mismatched partition, missing conditioning, unreadable references) or the
 * forward fails. On any non-OK status *out is zeroed. */
VLLM_API vllm_status vllm_video_generate(vllm_video_engine* engine,
                                         const vllm_video_params* params,
                                         vllm_video_result* out);

/* Free the owned members of a result and zero the struct. The struct itself
 * is caller storage. NULL is a no-op. */
VLLM_API void vllm_video_result_free(vllm_video_result* out);

/* v23: the phase table the LAST completed vllm_video_generate on this handle
 * wrote — an absolute path to a JSON file holding, per phase, a monotone
 * timestamp, a duration, and peak host and device byte counts, plus the wall
 * and the time the named phases did not cover.
 *
 * Points at storage the library owns for the lifetime of the handle (the
 * vllm_video_engine_family precedent); the caller must NOT free it, and the
 * next generate replaces it. NULL when this handle has completed no
 * generation, when the family emits no table, or when the engine is NULL.
 *
 * The FILE is the deliverable, not this string: a number that lives only in a
 * process nobody attached to is the evidence class issue #1040 is made of. */
VLLM_API const char* vllm_video_last_phase_log(const vllm_video_engine* engine);

/* ── Standalone MP4 mux-argv composer ─────────────────────────────────────────
 * The encoding contract (h264/yuv420p + AAC, -shortest, +faststart) is the
 * library's; the caller execs. This is the engine-free entry the
 * minimax-h3-mux example client uses to mux EXISTING frames + WAV. */
typedef struct vllm_video_mux_params {
  const char* frames;      /* printf-style pattern, e.g. dir/frame_%06d.ppm */
  const char* audio_path;  /* NULL/empty => a silent clip */
  const char* output_path; /* the .mp4 to write */
  int32_t fps;             /* <= 0 => the H3 default (24) */
  int32_t crf;             /* <= 0 => the library default (18) */
} vllm_video_mux_params;

VLLM_API vllm_video_mux_params vllm_video_mux_params_default(void);

/* Compose the argv (argv[0] "ffmpeg"; *out_argv[*out_argc] is NULL so it is
 * execvp-ready). The caller frees via vllm_video_mux_argv_free. */
VLLM_API vllm_status vllm_video_mux_argv(const vllm_video_mux_params* params,
                                         char*** out_argv, int32_t* out_argc);
VLLM_API void vllm_video_mux_argv_free(char** argv, int32_t argc);

/* ── Speech + music generation (ABI v20) ─────────────────────────────────────
 * The C face of vllm::multimodal::SpeechEngine: text (and, for a music family,
 * lyrics + a structured description) in, a waveform out, through the SAME
 * library seam the bundled server's /v1/audio/speech route drives — so HTTP and
 * FFI cannot drift.
 *
 * A speech engine is loaded from a checkpoint SET (MiniMax-Music3 ships six
 * component directories beside a modular_model_index.json), not from one model
 * directory, which is why this is a separate handle from vllm_engine. Loading a
 * TEXT checkpoint here fails naming vllm_engine_load; a directory NO registered
 * family claims is refused naming every family that was tried, because the
 * wrong family would not fail — it would render noise. */
typedef struct vllm_speech_engine vllm_speech_engine;

typedef struct vllm_speech_model_params {
  /* The checkpoint set's root directory. REQUIRED. */
  const char* path;
  /* The family to load, e.g. "minimax-music3". NULL/empty (the zero value)
   * DETECTS it by inspecting the artifact. An unregistered name is refused
   * naming what IS registered; it is never treated as a hint. */
  const char* family;
  /* v21 — WHERE the family runs. 0 = CPU (the zero value, and the arm every
   * Music3 correctness gate was taken on), 1 = the accelerator this build
   * resolves. Anything else is refused. Device 1 with no accelerator backend
   * registered, or on a PARTIAL backend that declines this family's
   * architecture, is refused BY NAME at load — never silently substituted.
   * Appended, so a zero-initialized v20 struct is byte-identical. */
  int32_t device;
} vllm_speech_model_params;

typedef struct vllm_speech_params {
  /* ── One-utterance families (IndexTTS-2.5) ─────────────────────────────── */
  const char* text;
  const char* language; /* NULL/empty => the family's default */
  /* The reference clip, for a family whose requires_reference_audio() is 1.
   * `reference_audio` is n_reference_audio interleaved-free mono f32 samples in
   * [-1, 1); NULL/0 means none. Borrowed for the call. */
  const float* reference_audio;
  int64_t n_reference_audio;
  int32_t reference_sample_rate;

  /* ── Music families (MiniMax-Music3) ───────────────────────────────────────
   * TWO texts, not one: upstream normalizes the sung lyrics and the structured
   * description differently, so packing both into `text` behind a separator
   * would be a private protocol. A one-utterance family ignores both. */
  const char* lyrics;      /* with [Verse]/[Chorus] section tags */
  const char* description; /* genre, BPM, key, instrumentation, mood */

  /* ── Generation controls; every zero selects the family's own default ───── */
  double audio_duration_s;     /* <= 0 => family default */
  int32_t num_inference_steps; /* <= 0 => family default */
  double guidance_scale;       /* honoured ONLY when has_guidance_scale != 0 */
  /* 0 IS A LEGAL guidance scale (it selects the unconditional branch), so the
   * "use the family default" signal cannot be the value 0 and is this flag
   * instead — the vllm_video_params.has_seed precedent. */
  int32_t has_guidance_scale;
  int64_t seed;
} vllm_speech_params;

/* One rendered waveform. OWNERSHIP: every pointer is library-allocated; free
 * the struct's members via vllm_speech_result_free(out).
 *   - samples: CHANNEL-MAJOR f32, channels * n_samples entries — channel c
 *     occupies samples[c*n_samples .. (c+1)*n_samples);
 *   - n_samples: samples PER CHANNEL;
 *   - sample_rate: the family's NATIVE rate, never resampled;
 *   - wav / n_wav: the same waveform as RIFF/WAVE 16-bit PCM, interleaved, so
 *     a caller can write or serve a playable file without a second encoder. */
typedef struct vllm_speech_result {
  float* samples;
  int64_t n_samples;
  int32_t sample_rate;
  int32_t channels;
  char* wav;
  int64_t n_wav;
} vllm_speech_result;

/* Zero-initialized params. For the model params that means "detect the family";
 * for the generation params it means every control at its family default. */
VLLM_API vllm_speech_model_params vllm_speech_model_params_default(void);
VLLM_API vllm_speech_params vllm_speech_params_default(void);

/* Resolve the checkpoint set to a registered family and stage its weights once.
 * On VLLM_OK, *out is a handle the caller frees via vllm_speech_engine_free.
 * On error, *out is NULL and vllm_last_error() carries the detail: a directory
 * nothing claims maps to VLLM_ERR_MODEL_LOAD listing the families tried. */
VLLM_API vllm_status vllm_speech_engine_load(const vllm_speech_model_params* params,
                                             vllm_speech_engine** out);
VLLM_API void vllm_speech_engine_free(vllm_speech_engine* engine);

/* Which family this handle RESOLVED to. Library-owned storage, valid for the
 * lifetime of the handle; the caller must NOT free it. NULL engine => NULL. */
VLLM_API const char* vllm_speech_engine_family(const vllm_speech_engine* engine);
/* The family's NATIVE output rate in Hz (44100 for MiniMax-Music3). 0 for a
 * NULL handle. */
VLLM_API int32_t vllm_speech_engine_sample_rate(const vllm_speech_engine* engine);
/* 1 when the family cannot synthesize without a reference clip, 0 when it can
 * synthesize from text alone, and 0 for a NULL handle. Ask BEFORE building a
 * request, so a missing clip is a caller-side refusal rather than a failed job. */
VLLM_API int32_t vllm_speech_engine_requires_reference_audio(const vllm_speech_engine* engine);
/* v21 — the device this handle actually RESOLVED to, in the same encoding
 * vllm_speech_model_params.device uses: 0 = CPU, 1 = an accelerator. 0 for a
 * NULL handle.
 *
 * It reports what was GRANTED, which is not what was asked for. A caller that
 * echoes back its own request cannot tell a device arm from a CPU arm with a
 * flag set, and a speed comparison built on that measures one arm twice. */
VLLM_API int32_t vllm_speech_engine_device(const vllm_speech_engine* engine);

/* Run one BLOCKING synthesis, filling *out. Serialized per engine handle.
 * VLLM_ERR_INVALID_ARGUMENT for a NULL engine/params/out;
 * VLLM_ERR_RUNTIME when the family refuses the request (a field it cannot
 * honour, a missing reference clip, a stage that is not implemented) or the
 * forward fails. On any non-OK status *out is zeroed. */
VLLM_API vllm_status vllm_synthesize(vllm_speech_engine* engine,
                                     const vllm_speech_params* params,
                                     vllm_speech_result* out);

/* Free the owned members of a result and zero the struct. The struct itself is
 * caller storage. NULL is a no-op. */
VLLM_API void vllm_speech_result_free(vllm_speech_result* out);

/* ── Memory helpers ───────────────────────────────────────────────────────────
 * Free a heap string returned by the library. NULL is a no-op. */
VLLM_API void vllm_string_free(char* s);

/* Free the owned members of a completion (out->text) and zero the struct. The
 * `out` struct itself is caller-provided storage and is not freed. NULL is a
 * no-op. */
VLLM_API void vllm_completion_free(vllm_completion* out);

/* ── Diagnostics / versioning ─────────────────────────────────────────────────
 * The last error on the CURRENT thread, as a NUL-terminated string owned by the
 * library (thread-local). Never NULL (empty string if no error). Valid until the
 * next C API call on the same thread; the caller must NOT free it. */
VLLM_API const char* vllm_last_error(void);

/* The library version string ("MAJOR.MINOR.PATCH[+cuda]"), static storage; do
 * not free. */
VLLM_API const char* vllm_version(void);

/* ── OpenAI-compatible server ─────────────────────────────────────────────────
 * Parse `argv` and RUN the OpenAI-compatible HTTP server until it exits,
 * returning the process exit code (0 on clean shutdown). `--help` prints usage
 * and returns 0; a bad argument or a startup failure prints the reason and
 * returns non-zero. Never throws across this boundary.
 *
 * This is what `vllm-server` is: examples/server is a thin client of this call.
 * It serves /v1/chat/completions, /v1/completions, /v1/models, /v1/embeddings,
 * and — when the matching flags are supplied — /v1/videos (MiniMax-H3) and
 * transcription, all through the SAME library seams the granular entry points
 * below drive, so HTTP and FFI cannot drift.
 *
 * BLOCKS for the lifetime of the server. `argv` must hold `argc` NUL-terminated
 * strings and stay valid for the duration; the library does not take ownership.
 * The conventional argv[0] program name is expected at index 0. */
VLLM_API int32_t vllm_server_main(int32_t argc, char** argv);

/* The ABI version the library was built with (compare against VLLM_ABI_VERSION). */
VLLM_API int32_t vllm_abi_version(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* VLLM_H_ */
