# External EXL3 test admission

External tests must distinguish unavailable inputs from invalid inputs and
numerical failures. `tests/support/exl3_external_artifacts.h` provides the
test-only entry points `ExternalEnvironment` and `ExternalPath`. Check the
complete required file set before initializing a GPU. It neither downloads
artifacts nor changes services, and it does not validate tensor contents.

Missing optional environment values, files or directories exit **77**. Register
each external case separately with CTest `SKIP_RETURN_CODE 77`; an unavailable
case must not prevent an unrelated case from running. A skip is unavailable
qualification, even when the overall CTest result is green.

Set `EXL3_REQUIRE_ARTIFACTS=1` in a required qualification job: missing inputs
then exit **1**. Unset or `0` selects optional admission. Other values, wrong
path types and filesystem inspection errors fail. Existing payloads still need
their own manifest, digest, shape and numerical checks after admission.

The host probe and generated-input test exercise this contract without a model,
oracle, GPU or oneAPI installation:

```sh
cmake --build build-cpu --target exl3_external_artifact_probe -j2
ctest --test-dir build-cpu --output-on-failure -R '^test_exl3_external_artifacts$'
```

Use a CPU configuration with `VLLM_CPP_BUILD_TESTS=ON`. The probe also accepts
`env VARIABLE`, `file PATH` or `dir PATH` for a single availability check.
All input locations are explicit; paths containing spaces are supported.
The host test covers optional skips, required failures, valid paths, wrong
types and invalid modes. It proves admission semantics, not inference parity.
