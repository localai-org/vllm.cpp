# Native image HTTP fixtures

The three fixed RGB PNGs and one JPEG have synthetic headings and distinguishable shapes.
`orbit.png` and `comet.png` share 512x384 geometry and use the same textual
prompt. `comet-unaligned.png` is 513x385 and needs the configured processor
resize. `fixtures.json` pins their hashes, expected words and side-specific
colors/shapes. No runtime font, Pillow or image-generation dependency is needed.

`scripts/mm/validate_native_exl3_vision_http.py` submits these four ordinary
requests and one live SSE request to a separately started native server. It
checks image semantics, image-dependent answers, 64 tokens or natural EOS,
usage, SSE termination, ordinary/streamed text equality and finite drained
scheduler gauges. It exits nonzero on failure and preserves the report at the
explicit output path. It never starts a server or uses a Python model fallback.

The initial mode is target-only C1, PNG/JPEG codecs enabled, FP8 KV and the
captured 4.2 MP processor envelope. This small PNG/JPEG gate does not establish complete-tower numerical parity, speculative decode, C2/C4,
prefix/chunk qualification or throughput parity. The source has a synchronous
streaming arm too; the executed end-to-end gate exercises AsyncLLM/live SSE.

Example against an already started local native server:

```sh
python3 scripts/mm/validate_native_exl3_vision_http.py \
  --url http://127.0.0.1:8000 --model native-vision-test \
  --output /tmp/native-vision-http-result.json
```


The compact `chunk-mrope-reference.json` contains SHA-256 digests of six actual
expanded prompts and executed CPU reference positions. They place the image
at offsets 1599/1600/1601 and 4095/4096/4097. No full model capture is included.
`generate_chunk_reference.py` runs the installed conditional model's request
position method inside the recorded reference image, without weights or GPU.
It requires measured native request reports and the checkpoint config:

```sh
python3 tests/fixtures/native_vision_http/generate_chunk_reference.py \
  /path/to/model/config.json /tmp/chunk-reference.json \
  --requests /tmp/boundary-1600-requests.json /tmp/boundary-4096-requests.json \
  --reference-image sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918
```

Start an isolated native server with `VT_NATIVE_VISION_TRACE=2`, prefix caching
disabled and target-only C1, then retain its three-request boundary report and
fresh log. The portable checker validates positions, masks, explicit-parent
source rows and one encoder submission across the cohort:

```sh
python3 scripts/mm/check_native_exl3_vision_chunk_trace.py \
  --requests /tmp/boundary-requests.json --trace-log /tmp/native-server.log \
  --budget 1600 --output /tmp/chunk-check.json
```

The boundary request producer is now portable. On a fresh otherwise unused C1
server, use page 1600, prefix caching OFF, trace level2, and the declared prefill
budget and speculative depth. The three requests put ORBIT at boundary-1,
boundary, boundary+1. The producer checks the frozen expanded-prompt hashes
before image work, then image facts, usage, finite drain gauges and actual
device-idle witnesses. It also requires actual draft-counter growth for MTP3.

```sh
python3 -O scripts/mm/capture_native_exl3_vision_chunk.py \
  --url http://127.0.0.1:8000 --model YOUR_SERVED_MODEL \
  --boundary 4096 --budget 1600 --spec-depth 3 \
  --trace-log native-server.log --output chunk-requests.json
```

The client never starts/stops services. After stopping the isolated instance,
its supervisor records actual `exit_code`, `server_removed`,
`production_stopped`, source `head`, `binary_sha256`, and `requests_sha256` of
the client output. Supply that separate receipt to the position/slice checker:

```sh
python3 -O scripts/mm/check_native_exl3_vision_chunk_trace.py \
  --requests chunk-requests.json --trace-log native-server.log \
  --execution chunk-execution.json --budget 1600 --spec-depth 3 \
  --output chunk-check.json
```

The checker verifies the observed first chunk agrees with the declared C1
budget and requires exactly one encode across all three requests. The execution
receipt is a supervisor observation, not independent cleanup verification by
the checker. Earlier archived reports still work without the optional receipt.

To compare exact committed token IDs across target-only/MTP3 and chunk budgets,
list completed runs in JSON. Each entry has `boundary`, `budget`, `depth`,
`requests`, `trace`, and `execution`; file names resolve relative to this list.
Include both depths 0 and 3 for each listed boundary/budget, with the same native
binary/source head. For example:

```json
[
  {"boundary":1600,"budget":1600,"depth":0,"requests":"target.json","trace":"target.log","execution":"target-exit.json"},
  {"boundary":1600,"budget":1600,"depth":3,"requests":"mtp.json","trace":"mtp.log","execution":"mtp-exit.json"}
]
```

```sh
python3 -O scripts/mm/check_native_exl3_vision_chunk_pairs.py \
  --runs chunk-runs.json --generation-config MODEL_DIR/generation_config.json \
  --output chunk-pairs-check.json
```

Every run must first pass the reference position/slice checks and show advancing
C1 decode graphs. The paired checker compares raw committed IDs, response text,
usage and finish reason for each image offset. EOS is checked against the
checkpoint generation configuration; sampler tokens after the observed stop
or declared output limit are excluded only within the speculative-depth bound.
These metadata checks do not qualify prefix restoration or complete-tower
numerical parity. Budget 8192 remains outside the existing W8A8 M<=4096 bound;
budget 4096 can provide a single-chunk contrast for the smaller 1840-token
recipe without changing that operator limit. Trace timings are not benchmarks.


For the aligned prefix workload, use page/budget 1600, prefix caching enabled,
`VT_NATIVE_VISION_TRACE=2` and `VT_PREFIX_SNAPSHOT_TRACE=1`. The four requests
are A cold, A repeat, B with identical text/shape and A repeat after B; their
image placeholder starts at token 1599. Validate the fresh report/log with:

```sh
python3 scripts/mm/check_native_exl3_vision_prefix_trace.py \
  --requests /tmp/prefix-requests.json --trace-log /tmp/native-server.log \
  --output /tmp/prefix-check.json
```

The checker requires two actual 1600-token restore events, exact oracle cold
positions and warm suffix metadata, distinct image hashes, only two encoder
submissions and finite drained gauges. It uses the implemented prefix-hit
counter; the generic cached-prompt counter is explicitly deferred in the
output processor and cannot establish a hit. This is an aligned C1 restore
check, not eviction/cancellation qualification. Request production now uses a
portable standard-library client. Start a fresh C1 target-only server with the
recipe above, then submit the frozen requests:

```sh
python3 -O scripts/mm/capture_native_exl3_vision_prefix.py \
  --url http://127.0.0.1:8000 --model YOUR_SERVED_MODEL \
  --trace-log native-server.log --output prefix-requests.json
```

The client verifies both PNG hashes and the expanded prompt digest against the
executed CPU reference. It requires heading and left/right facts, exact warm
response choices/usage, prefix-hit deltas0/1600/0/1600, and four drained/idle
releases. The generic `prompt_tokens_cached` counter remains diagnostic and
deferred in `output_processor`; the later prefill-timing fix did not implement
that counter. The initial portable v1 run incorrectly required both counters
to agree and failed despite a real1600-token restore; that failed receipt is
retained. No product arithmetic or cache policy was changed to correct this
test assumption.

After actual cleanup, supply a supervisor receipt with client `exit_code`,
`server_removed`, `production_stopped`, `head`, `binary_sha256` and the client
report's `requests_sha256`:

```sh
python3 -O scripts/mm/check_native_exl3_vision_prefix_trace.py \
  --requests prefix-requests.json --trace-log native-server.log \
  --execution prefix-execution.json --output prefix-check.json
```

This independently checks actual restore IDs, exact cold/reference positions,
warm suffix metadata, two distinct encoder submissions and four idle-owner
records. Cleanup is an outer-supervisor observation, bound to the report hash.
The previous archived request format remains supported without `--execution`.
Trace timings are not serving performance.


A bounded encoder eviction wave uses the existing 2048-row cache with eleven
distinct fixed-size images, then repeats the evicted original A/B images. Nine
variants change only COMET's corner pixel, to retain its semantic task. The
fourteen requests use C1/target-only, prefix caching off and budget 192. A small
prefill budget alone does not reduce the current encoder-cache capacity.
Check the retained report and fresh trace-level2 log with:

```sh
python3 scripts/mm/check_native_exl3_vision_eviction_trace.py \
  --requests /tmp/eviction-requests.json --trace-log /tmp/native-server.log \
  --output /tmp/eviction-check.json
```

The check requires actual cache insertions/evictions, re-encodes of A/B, stable
warm backend USM allocations, bounded live output bytes and exact repeated
answers/usage. Backend counts include retained pools and model weights, and
exclude driver memory. Request/variant production is still a local harness;
no generated variants or raw response captures are shipped in these fixtures.

That 2048-row wave is historical and predates the current model-scoped
maximum-image encoder budget. Its checker preserves that measured recipe;
it is not evidence for eviction at the enlarged current capacity.

The separately frozen `orbit-max-2048.png` is a 2048-square RGB PNG: the
original ORBIT image enlarged fourfold with nearest-neighbor resampling and
pasted on a white square at y=256. It is an authored test image, without model
weights or captured learned features. Its hash is in
`max-eviction-mrope-reference.json`; the four-case `fixtures.json` remains
unchanged. The new current-capacity C1 MTP3 wave is max,max,COMET,COMET,COMET,
repeated twice, with page/chunk 1600, prefix caching off and image=1. It requires
four encoder submissions, three evictions, device-owner release, exact
repeat token sequences and the warm small-image allocation baseline:

```sh
python3 scripts/mm/check_native_exl3_vision_max_eviction_trace.py \
  --requests /tmp/max-eviction-result.json --trace-log /tmp/native-server.log \
  --output /tmp/max-eviction-check.json
```

The compact max/small position digests were produced by actual Python
request-position code on CPU, with qualified grid shapes and no learned
inference. Regenerate only in the pinned reference environment without GPU
exposure, retaining a separately named result:

```sh
python3 tests/fixtures/native_vision_http/generate_max_eviction_mrope_reference.py \
  --requests /tmp/max-eviction-result.json --config /path/to/model/config.json \
  --output /tmp/max-eviction-reference.json
```

The max/small request producer is now portable. Use an otherwise unused native
server with the declared C1 MTP3/image=1 recipe, page/chunk 1600, prefix caching
OFF, and `VT_NATIVE_VISION_TRACE=2`. Give the client its fresh local server log:

```sh
python3 -O scripts/mm/capture_native_exl3_vision_max_eviction.py \
  --url http://127.0.0.1:8000 --model YOUR_SERVED_MODEL \
  --trace-log native-server.log --output max-eviction-result.json
```

The client verifies fixture hashes, exact frozen expanded prompts, image facts,
usage, repeats, scheduler drain gauges, device-idle witnesses and small-image
memory release. It does not start/stop services or record fictitious cleanup.
After stopping the isolated server, its supervisor records a separate execution
JSON with actual `exit_code` (client), `server_removed`, `production_stopped`,
`binary_sha256`, source `head`, and `requests_sha256` of the client output.
Then check the original trace plus that receipt:

```sh
python3 -O scripts/mm/check_native_exl3_vision_max_eviction_trace.py \
  --requests max-eviction-result.json --trace-log native-server.log \
  --execution max-eviction-execution.json --output max-eviction-check.json
```

Cleanup fields are supervisor observations, not independently verified by the
checker. The optional execution argument also preserves the previous archived
report format, which carried these observations in the request report itself.
The new portable client has passed the actual ten-request GPU wave with four
encodes, three evictions, 640 exact repeat IDs and reference positions. This C1
envelope does not qualify four simultaneous maximum images or full-tower
numerical parity. Do not interpret traced request wall times as performance.


The C1 cancellation workload uses page/budget 1600, prefix caching off,
trace level2, `VT_SERVER_PREFILL_PROGRESS=1` and `--enable-server-dev-mode`.
It aborts the explicitly identified request after encode submission while the
first partial-image prefix is active, then executes three ordinary retries:

```sh
python3 scripts/mm/check_native_exl3_vision_cancel_trace.py \
  --requests /tmp/cancel-requests.json --trace-log /tmp/native-server.log \
  --output /tmp/cancel-check.json
```

The check requires zero generated tokens for the cancelled stream, actual
partial row0 consumption, no completed cancelled prefill, idle owner release,
finite drained gauges, one total encode and stable retry allocations/results.
It qualifies C1 request cancellation, not interruption inside a GPU kernel or
asynchronous C4 overlap. Request orchestration is now portable. Start an unused
C1 target-only server with page/chunk 1600, prefix caching OFF, native trace2,
prefill progress and dev mode, then run:

```sh
python3 -O scripts/mm/capture_native_exl3_vision_cancel.py \
  --url http://127.0.0.1:8000 --model YOUR_SERVED_MODEL \
  --trace-log native-server.log --memory-contract graph-pool-owned \
  --output cancel-requests.json
```

The client calibrates and verifies the frozen offset1599 prompt. It observes
the actual encoder plus first partial-image target submission before explicitly
aborting that live request through `/abort_requests`. This is not automatic
HTTP-disconnect handling. It requires zero generated tokens, terminated SSE,
three identical 64-token retries, one encode and four drained device releases.
It never starts/stops services. After actual cleanup, the supervisor records
`exit_code`, `server_removed`, `production_stopped`, `head`, `binary_sha256` and
`requests_sha256`; supply that receipt to the independent checker:

```sh
python3 -O scripts/mm/check_native_exl3_vision_cancel_trace.py \
  --requests cancel-requests.json --trace-log native-server.log \
  --execution cancel-execution.json --output cancel-check.json
```

The default `resident-exact` client contract preserves the original requirement
that all three retry totals are equal. That strict current graph-enabled run
remains FAIL: its first later prefill adds 1,026,124 bytes of free retained
scratch. Trace attribution proves this entire increase is in the main scratch
pool; graph count/device bytes and checked-out pool blocks do not increase.
The explicitly selected `graph-pool-owned` contract instead requires exact
backend bytes excluding free main scratch at all three retries, unchanged live
pool blocks and graph residency, plus an exact total/free-pool/allocation-miss
plateau at the last two. It has a separate actual PASS receipt; it does not
rewrite the original strict failure. The baseline includes resident caches and
weights, and excludes driver allocations. No memory tolerance is widened.
Earlier archived receipts still use the original total-byte predicate.

For paired C1 target-only/MTP1 or MTP3 receipts produced with trace level 2:

```sh
python3 scripts/mm/check_native_exl3_vision_mtp_trace.py \
  --target-report target-result.json --target-log target-server.log \
  --mtp-report mtp-result.json --mtp-log mtp-server.log --depth 3
```

The checker requires five actual PNG/JPEG/SSE cases, all 320 committed IDs,
reference-derived three-axis positions and real proposal/verification counters.
It consumes existing receipts; a portable paired request producer is still
pending. It makes no Python numerical or speed-parity claim.

For image MTP chunk receipts, add `--spec-depth 3` to the chunk trace checker.
This mode uses committed token histories for verification positions and checks
the draft's shifted slices, including the pinned unencoded lookahead fallback.
The default remains target-only.

Add `--require-graphs` to the paired MTP checker when qualifying native target
Decode graphs. It requires progressing graph execution observations as well as
the exact committed sequence; an otherwise passing eager receipt fails this
gate. The native draft remains eager in the current implementation.

The mixed two-image cancellation producer uses a fresh MTP3 server with four
**effective** request slots, two permitted images, page/prefill budget 1600,
FP8 KV, prefix caching OFF, native trace2, `VT_SERVER_PREFILL_PROGRESS=1` and
`--enable-server-dev-mode`. Existing admission limits still apply. The actual
qualification retained the default requested C32 and overlarge multimodal
limits to exercise their native C4/two-image/video=0 clamp. It does not reserve
four maximum images. Run against the isolated server and its new trace:

```sh
python3 -O scripts/mm/capture_native_exl3_vision_mixed_cancel.py \
  --url http://127.0.0.1:8000 --model YOUR_SERVED_MODEL \
  --trace-log native-server.log --output mixed-cancel-requests.json
```

The standard-library client verifies both frozen PNG hashes and the expanded
2022-token prompt whose first image starts at1599. It aborts the cold identified
request only after actual first-row submission, waits for drained gauges and a
device-idle witness, and completes the same padded prompt. A four-client barrier
then submits one streamed two-image victim, a COMET image companion and two text
completions. Before aborting that victim it requires content, an executed C4
captured graph and running=4/waiting=0. All three companions finish their full
64/96/96-token quotas. Three later two-image retries must preserve order, exact
response choices/usage, two total encodes, bounded image owners and an exact
final-two backend-byte plateau. Trace consumption is incremental and retains
only compact lifecycle records; client polling is outside performance timing.

After actual shutdown, record supervisor observations `exit_code`,
`server_removed`, `production_stopped`, source `head`, `binary_sha256` and
`requests_sha256` of the client result. Then independently check:

```sh
python3 -O scripts/mm/check_native_exl3_vision_mixed_cancel_trace.py \
  --requests mixed-cancel-requests.json --trace-log native-server.log \
  --execution mixed-cancel-execution.json --output mixed-cancel-check.json
```

The checker also proves the cancelled target/draft slices and masks from the
raw trace; these are not inferred from the client summary. Cleanup remains an
outer-supervisor observation. Omitting `--execution` preserves the previous
archived report format, with cleanup fields in that report. The actual portable
run and checker PASS, including 375 actual draft tokens; missing cleanup, wrong
report hash and removed idle-owner witness each FAIL under Python `-O`.
This qualifies explicit dev-mode cancellation and bounded lifecycle/reuse,
not transport disconnects, interruption inside a kernel, strict mixed-batch
Token-ID invariance, full-tower numerical parity or serving performance.

The same client can qualify the separate conversation-history case on another
fresh C1 target-only prefix-enabled instance. The frozen previous assistant
message comes from an earlier actual native image response. A pure text user
turn follows the image-containing first message; it is run cold and warm:

```sh
python3 -O scripts/mm/capture_native_exl3_vision_prefix.py --history \
  --url http://127.0.0.1:8000 --model YOUR_SERVED_MODEL \
  --trace-log native-server.log --output history-requests.json
python3 -O scripts/mm/check_native_exl3_vision_prefix_trace.py --history \
  --requests history-requests.json --trace-log native-server.log \
  --execution history-execution.json --output history-check.json
```

`history-mrope-reference.json` records the actual pinned CPU position-method
source, frozen assistant text/hash and compact token/position digests. The
portable actual run has1957 prompt tokens, one encoder submission, hit0/1600,
exact cold/reference positions and exact warm suffix metadata, two terminated
64-token responses and two idle-owner releases. Its assistant history already
contains image facts, so this is structural/position/cache qualification, not
an independent image-understanding or held-out quality test. Those use the
separate fresh image and held-out suites. No two-response exact memory plateau
or MTP-history claim is made here.

Reproduce the compact CPU reference using the installed pinned environment
without `/dev/dri` or model weights; only config and the actual client report
are needed. Use a new output path and compare it with the frozen reference:

```sh
python3 -O tests/fixtures/native_vision_http/generate_history_reference.py \
  MODEL_CONFIG new-history-reference.json --requests history-requests.json \
  --reference-image sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918
```

The executing source hash and declared runtime must match the pins. A fresh
actual CPU run reproduces the complete compact reference exactly. The native
checker also rejects altered history text and an altered submitted position,
in addition to wrong hit counters and missing restore/cleanup/report binding.
The generic cached-prompt counter remains deferred, and full Merger, mixed
batch token invariance and resulting PR-head qualification remain open.

For bounded real-serving phase attribution, build the optional internal XPU
utility `native-vision-serving-profile`. It uses the existing production C-API
engine and backend profile seam, not a standalone language generator. Generate
the same frozen C-API bundle and use its first backup-dialog screenshot:

```sh
cmake --build YOUR_BUILD --target native-vision-serving-profile -j2
python3 scripts/mm/generate_native_exl3_vision_capi_requests.py \
  --model YOUR_MODEL_DIRECTORY_BASENAME --output capi-requests.json
VT_XPU_PROFILE=1 YOUR_BUILD/tests/native-vision-serving-profile \
  MODEL_DIR capi-requests.json target-profile.json 0
VT_XPU_PROFILE=1 YOUR_BUILD/tests/native-vision-serving-profile \
  MODEL_DIR capi-requests.json mtp-profile.json 3
```

Run the two modes sequentially in the pinned native XPU recipe, with its
existing draft subset/map, oneDNN/library environment and GPU selection. Disable
native/host/prefix snapshot traces, keep production stopped and never overlap
another GPU instance. C1, prefix OFF, page/budget1600, image1, four repeats and
bounded output are set by the utility. It drains loading events before the
first request and profile records only after whole responses. No product
synchronization or inference arithmetic changes. The initial target span is
language prefill for this357-token single-chunk prompt; the initial MTP span
includes draft-prefill setup plus two extra draft-decode steps. Queue spans can
include host gaps and are not sums of kernel times or trace-free benchmarks.

The supervisor separately records actual `exit_code`, `instance_removed`,
`production_stopped`, binary/source `head`, `binary_sha256`, input bundle
`requests_sha256` and output `result_sha256`. Then check both runs:

```sh
python3 -O scripts/mm/check_native_exl3_vision_serving_profile.py \
  --target target-profile.json --target-execution target-execution.json \
  --mtp mtp-profile.json --mtp-execution mtp-execution.json \
  --requests capi-requests.json --output serving-profile-check.json
```

The checker binds actual cleanup reports, validates screenshot facts, exact
repeat and cross-mode responses, finite device-clock intervals, initial-phase
attribution, actual MTP counters and backend memory. Both actual cohorts PASS.
This covers the measured queue spans, not individual draft-prefill kernels,
Python speed parity, driver allocations or full-tower numerical correctness.
