# Evidence — NEW-CANDIDATE-PORTS W0 (inkling2_loop)

Issue ISSUE-LOCAL-01M80WA9T38SWMMWYDJ9T938K8. Spec
`.agents/specs/inkling2-loop-w0-gguf-types-and-parser.md` (committed first,
`171a1d964`). CPU only; no GPU lease, no vLLM pin change.

## Fixture

`/mnt/models/inkling2-mini-apex-gguf/Inkling-2-Mini-APEX-Balanced.gguf` (20,684,174,784 bytes),
`mudler/Inkling-2-Mini-APEX-GGUF @ d85a58007995c3365eb2323b177937ce4d20b8f0`.
Full walk: `docs/bench-evidence/new-candidate-inkling2-20261121/apex-balanced-walk.txt`
(704 tensors, 59 KVs, 14 `ffn_down_exps` tensors at ggml id 7).

## TDD red → green

`red.txt` — the focused binary built from the deliberate red stub and the old
type table fails 6/7 cases: `GgmlTraits(3)` throws `unknown ggml type id 3` and
`GgufFile::Open` throws `... unknown ggml type id 7` on
`blk.5.ffn_down_exps.weight`, exactly the reported defect.

`focused.log` — after adding ids 3/7 and the real parser: 7/7 cases, 67/67
assertions, `Status: SUCCESS!`.

## Commands

Configure (CPU-only):
```
cmake -S . -B /tmp/opencode/vllm-inkling2-build -G Ninja \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_HIP=OFF -DVLLM_CPP_VULKAN=OFF \
  -DVLLM_CPP_METAL=OFF -DVLLM_CPP_MLX=OFF -DVLLM_CPP_TRITON=OFF \
  -DVLLM_CPP_BUILD_BORINGSSL=OFF -DVLLM_CPP_BUILD_TESTS=ON \
  -DVLLM_CPP_BUILD_EXAMPLES=OFF -DVLLM_CPP_SERVER=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/vllm-inkling2-build --target test_inkling2_loop_gguf -j4
```

`skip-check.log` — artifact present: 7/7 pass; with
`VT_INKLING2_APEX_GGUF_PATH=/nonexistent/absent.gguf`, the binary prints the
SKIP reason and exits 77.

`ctest.log` — CTest reports `Passed` with the artifact and `***Skipped` with
the override (SKIP_RETURN_CODE 77 wired by `vllm_cpp_add_test`), then four
regression tests all pass: `test_gguf`, `test_new_candidate_parsers`,
`test_olala_gguf_weights`, `test_inkling2_loop_gguf`. The first `test_gguf`
run FAILED because its old assertion named id 7 as untabulated; that assertion
was updated to use id 9 (Q8_1) and to assert the new id-3/id-7 geometry, and
the rerun is in the same log.

## Oracle grounding

- vLLM PR #69028 (merged 2026-11-21): `Inkling2LoopForConditionalGeneration`,
  `model_type: inkling2_loop`; `vllm/models/inkling2/configs.py` loop_config
  fields and validation, `vllm/model_executor/models/registry.py` registration,
  `vllm/transformers_utils/config.py` config-class mapping. Upstream test
  `tests/models/inkling2/test_model.py::test_loop_serving_options_validation`
  ported where CPU-parse-applicable; its GPU/forward cases are port-row scope.
- ggml block geometry: `/home/lu_zero/Sources/llama.cpp/ggml/include/ggml.h:391-407`
  (`GGML_TYPE_Q4_1 = 3`, `GGML_TYPE_Q5_1 = 7`, `Q6_K = 14`) and the matching
  `ggml-common.h` block layouts.
