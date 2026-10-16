# Frozen external alpha attribution identities

`pins.json` is a compact, path-free derivative of four historical vision
attribution capsules. Their names and SHA256 identities are retained in
`derived_from`; the development capsules are not required at runtime. This
file contains artifact/model identities only, not activation tensors, model
weights, expected numerical outputs or a gate waiver.

The loader in `scripts/mm/native_vision_attribution_pins.py` pins the complete
3185-byte manifest to SHA256
`90c0050f30cdcc7502df5e872a56c6cac5f98c4618911caadcf6d2d27eee4fb6`.
Missing, changed or oversized manifests fail before GPU/reference initialization,
including under Python `-O`. Do not replace frozen identities with a freshly
captured scheduling layout to make an existing comparison pass.

Consumers:

- `analyze_native_exl3_vision_adjacent_prefix.py`: original prefix21 metadata.
- `replay_native_exl3_vision_target_states.py`: adjacent/first-MTP selected
  metadata and original last-prefix metadata.
- `capture_native_exl3_vision_python_alpha.py`: first-MTP metadata, original
  failed mixed-result identity and exact checkpoint index/shard identities.

The real large tensor captures and checkpoint remain separately supplied
external artifacts. Caller-supplied capture directories must contain the
original metadata and payloads; each existing consumer retains its hash,
shape, request, position, state and full-head checks. This manifest does not
make a new capture identical to the historical failed case. No artifacts are
downloaded or substituted by the loader.

A model-free packaging/admission test imports the consumers from a relocated
source subset with no `docs` directory, verifies the public manifest, rejects
missing external captures and confirms that no Torch/vLLM runtime initializes:

```sh
python3 -O tests/scripts/test_native_vision_attribution_pins.py
```

The same focused test is registered as `test_native_vision_attribution_pins`
in CTest when tests are enabled. Its success is a portability/admission
result, not numerical or serving qualification.
