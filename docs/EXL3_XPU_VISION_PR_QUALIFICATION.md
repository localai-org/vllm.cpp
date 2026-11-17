# Resulting draft-PR image qualification

The native product at `5bb81a30b7862aa978f1ad7b24b89296e631c6ed` passes the bounded resulting-head checks
below. Main `5ce71e490d8e6b9615540f1244ec1b4c35f31f59` was merged independently, then the
442 selected paths from qualified vision source `959e13c5a` were imported.
The upstream lease guard and ordered tool-schema changes remain intact.
Among the 442 transferred files, only CMake registrations and C-ABI parsing
differ from the frozen vision files;
those differences preserve the corresponding Main additions.

| Executed check | Actual result |
|---|---|
| Pre-import Main text base | Native server build; four lease cases / 28 assertions; one real P4096/O32 MTP3 text-prefix control PASS |
| Resulting native server and focused binaries | Build PASS; hashes in companion JSON |
| Vision operators, including actual external/scalar fixtures | 35 cases / 1,854 assertions PASS; no skipped case |
| Non-checkpoint tower scalar/owner checks | Four cases / 101 assertions PASS; full-checkpoint case explicitly filtered out |
| Variable GDN batch | One selected case / 31 assertions PASS |
| Synthetic hybrid-model graph/eager state ownership | One selected case / 4,737 assertions PASS |
| Actual captured packed FP8 C4/C1 verifier | One selected case / 220 assertions PASS; changing metadata and pre-write negative guards |
| HTTP target-only | Seven PNG/JPEG, SSE and two-image ordinary/SSE requests PASS; zero proposed drafts |
| HTTP MTP3 | Seven matching image cases PASS; 300 proposed draft tokens; strict usage/gauges |
| Public C ABI and ownership teardown | Six backup-dialog fact requests PASS; 144 proposed drafts; zero graphs and live pool blocks after engine free |
| Resulting-head text | One actual P4096/O32 MTP3 output prefix, usage and public token representations match frozen pre-vision baseline |
| Cache/EOS observer regression | Eleven focused Python tests PASS under `-O` |

The actual server/tool processes were isolated and sequential on B70. Production
remained stopped; each temporary native instance was removed. The companion
JSON binds source, binary, fixture/config and external command/log/result hashes.
Filtered-out tests are not reported as executed passes.

## Reproduce the focused checks

Use the dependency/runtime pins and build recipe in
[EXL3_XPU_VISION_SERVING.md](EXL3_XPU_VISION_SERVING.md). The native build used
oneAPI 2026.1.1, oneDNN 3.13.0 and the pinned SYCL-TLA checkout with text Xe2
prefill OFF; vision Xe2 attention is built independently. Set external reference
and checkpoint paths explicitly. Missing required artifacts must fail:

```sh
EXL3_REQUIRE_ARTIFACTS=1 VT_B70_VISION_REFERENCE_DIR=/path/to/reference \
  VT_B70_VISION_MODEL_DIR=/path/to/model build-xpu/tests/test_xpu_vision_ops
build-xpu/tests/test_xpu_qwen_exl3_fp16 \
  --test-case='XPU dense EXL3 MM: MTP3 graph owns*'
python3 -O tests/scripts/test_native_vision_cache_trace.py
python3 -O tests/scripts/test_native_vision_completion.py
```

Run the public fixture client against separately launched target and MTP3
servers as described in the guide. The existing C-ABI request generator and
`native-vision-chat` example are portable public reproduction tools. The local
qualification used the `native-vision-serving-profile` public load/chat calls
for its bounded memory/teardown check; it did not claim to execute that example.

## Retained scope and remaining limits

N4 at vision source `959e13c5a` passed the original four tower geometries,
known/held-out mixed checks, selected actual state retention, A/A/B/A and
maximum/small warm-owner sequences, actual cancellation/reuse and paired
P4096/O256 native text timing. Those earlier results retain their source and
binary identities; the full N4 matrix was not rerun on this PR product.
The guide records those metrics separately from the fresh integration checks.

The developer approved supported same-geometry isolation and correct actual
MTP target-prefix retention, allowing shape-dependent greedy text. Independent
Python self-controls show comparable shape dependence, not cross-runtime
numerical equality. Tower limits remain relative L2<=0.003 and max absolute<=0.05;
no replacement tolerance or second runtime profile was adopted. Previous
actual accepted 1/2 and forced-selector controls retain their original scope.

This remains an experimental draft for at most two PNG/JPEG still images in the
measured B70 envelope. Video/audio, remote URLs, arbitrary devices/codecs/shapes,
262K image contexts and complete Python-feature/answer-quality/stochastic-state
parity are not qualified. Existing text-reference and throughput gaps remain.
No full preflight/CI campaign was run; historical automatic publication failures
and skips remain in the earlier evidence appendix. This report does not mark
the PR Ready, merge upstream or qualify production use.
