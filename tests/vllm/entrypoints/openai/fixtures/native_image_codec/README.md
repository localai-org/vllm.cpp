# Native codec reference fixtures

These images contain deterministic generated pixel patterns. No model weights
or private images are included. `manifest.json` records container and RGB hashes.
The pinned reference is vLLM `ImageMediaIO.load_bytes`, including EXIF handling
and its default white transparency conversion, rather than direct Pillow RGB
conversion. Generate inside that reference environment with:

```sh
python generate.py OUTPUT_DIRECTORY
```

`oversize-header.png` deliberately contains CRC-valid 8192×8192 dimensions with
small unrelated image data. It must fail admission before a pixel allocation.
Other invalid/truncated cases are constructed from these bytes by the tests.

`orientations.json` adds separately generated asymmetric rectangular RGB
fixtures for EXIF orientations 1–8 in both PNG and JPEG, using both TIFF byte
orders. The expected RGB bytes come from the executed pinned `ImageMediaIO`,
including JPEG decoding before orientation. Regenerate these added fixtures
inside the same reference environment with:

```sh
python generate_orientations.py OUTPUT_DIRECTORY
```

This producer leaves the original frozen fixtures and manifest untouched.
