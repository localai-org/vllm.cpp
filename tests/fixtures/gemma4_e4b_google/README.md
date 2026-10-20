# Original Google E4B metadata

These four unchanged files come from Google DeepMind's
[`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it/tree/ee0ef6023621cff504d758262d4e04895a5af4a2).
`manifest.json` records immutable resolve URLs, sizes, and SHA-256 hashes.
The validator has independent pins so a modified manifest cannot authorize
replacement bytes. No weights, tokenizer vocabulary, media, or inference
outputs are included.

## Attribution and license

Google DeepMind is the upstream author of `config.json`,
`processor_config.json`, `generation_config.json`, and `chat_template.jinja`.
They are unchanged. The
[pinned model card](https://huggingface.co/google/gemma-4-E4B-it/blob/ee0ef6023621cff504d758262d4e04895a5af4a2/README.md)
declares `license: apache-2.0` and links to the
[Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license).
`LICENSE` contains the Apache License, Version 2.0 text, copied from
[the pinned Transformers license](https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/LICENSE).
The model repository's pinned file listing contains no separate LICENSE or
NOTICE file. Preserve this attribution with these fixtures.

`upstream-tests.json` is a locally generated index, not vendored test code.
It lists source URLs, hashes, declared test symbols, line numbers, and decorator
parameters from pinned vLLM and Transformers test sources. Decorator excerpts
retain upstream test parameters and skip reasons. Those sources are Apache-2.0,
authored by contributors to vLLM and the Hugging Face team respectively.
No upstream test was run. Inherited tests and parameter-expanded collected
names need the actual oracle environment and are not claimed by this index.

## Reproduce the metadata check

Run from the repository root:

```sh
python3 scripts/mm/check_gemma4_e4b_manifest.py \
  --metadata tests/fixtures/gemma4_e4b_google/manifest.json --offline
```

To reproduce source bytes, download each manifest URL into a separate scratch
directory and compare its byte count and SHA-256 to the manifest. Do not replace
pins on a mismatch. The original planning source manifest supplied these bytes.
No network is used by the validator or its unit tests.

Exit 0 means **metadata-only**, never runtime acceptance or original-model
gateability. `--report` is mutually exclusive with `--offline`. It checks a
strict structural matrix, then fails because numeric bars and measured oracle
compatibility are PENDING. A report cannot set its own thresholds or substitute
another checkpoint. Even a zero-error synthetic report is not accepted.

The structural report has the four identity fields from the manifest and a
`cases` list. Each case has exactly `id`, `status`, `executed`, `skipped`,
`token_mismatches`, `integer_mismatches`, `max_abs_error`, and `max_rel_error`.
The mandatory case IDs are defined in the validator. Counts must be integers,
execution positive, skips and mismatches zero, and numeric errors finite and
nonnegative. This minimum matrix is not the future stage-reference harness.
T11 still owes artifact provenance, per-stage arrays and bounds, per-modality
counts, engine identities, and all production evidence required by the spec.
