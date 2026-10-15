"""Portable local artifact resolution and Safetensors identity checks.

No runtime imports, downloads, inference, or service operations occur here.
"""
import os
from pathlib import Path

from extract_projection import digest, headers


class MissingArtifact(FileNotFoundError):
    """An external payload or its manifest is unavailable, rather than invalid."""


class ArtifactRoot:
    """An explicit root wins over EXL3_ARTIFACT_ROOT; otherwise use cwd.

    Existing explicit absolute filenames remain supported without a configured
    root. Configured roots constrain absolute filenames and symlink targets too.
    """
    def __init__(self, root=None):
        configured = root if root is not None else os.environ.get("EXL3_ARTIFACT_ROOT")
        self.configured = configured is not None and str(configured) != ""
        self.path = Path(configured if self.configured else Path.cwd()).expanduser().resolve()

    def resolve(self, name, *, constrain=False):
        name = Path(name)
        headers.require(".." not in name.parts, "artifact path traversal is not allowed")
        path = (name if name.is_absolute() else self.path / name).resolve()
        if self.configured or constrain or not name.is_absolute():
            headers.require(path.is_relative_to(self.path), "artifact escapes the configured root")
        return path


def require_file(path):
    if not path.exists():
        raise MissingArtifact(f"missing artifact: {path}")
    headers.require(path.is_file(), f"artifact is not a regular file: {path}")


def _sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def checked_fixture(path):
    """Check existing projection sidecars; paths are not content identities.

    Legacy unversioned sidecars remain readable. A declared schema must be the
    supported version, and declared shape/dtype/hash metadata is always checked.
    """
    path = Path(path)
    sidecar = path.with_suffix(".json")
    require_file(path)
    require_file(sidecar)
    receipt = headers.read_json(sidecar)
    if "schema" in receipt:
        headers.require(type(receipt["schema"]) is int and receipt["schema"] == 1,
                        "incompatible projection fixture schema")
    expected = receipt.get("tensor_hashes")
    headers.require(isinstance(expected, dict) and expected, "invalid fixture tensor manifest")
    sha = receipt.get("fixture_sha256")
    headers.require(_sha256(sha) and digest(path.read_bytes()) == sha,
                    "fixture SHA-256 mismatch")
    report = headers.read_shard_header(path)
    entries = {t["name"]: t for t in report["tensors"]}
    headers.require(entries.keys() == expected.keys(), "fixture tensor set mismatch")
    with path.open("rb") as stream:
        base = 8 + report["header_bytes"]
        for name, entry in entries.items():
            spec = expected[name]
            headers.require(isinstance(spec, dict), f"invalid fixture tensor metadata: {name}")
            dtype, shape = spec.get("dtype"), spec.get("shape")
            headers.require(isinstance(dtype, str) and dtype in headers.DTYPE_BYTES
                            and isinstance(shape, list)
                            and all(headers.unsigned_integer(d, headers.MAX_DIM) for d in shape)
                            and (entry["dtype"], entry["shape"]) == (dtype, shape),
                            f"fixture dtype/shape mismatch: {name}")
            begin, end = entry["data_offsets"]
            stream.seek(base + begin)
            sha = spec.get("sha256")
            headers.require(_sha256(sha) and digest(stream.read(end - begin)) == sha,
                            f"fixture tensor SHA-256 mismatch: {name}")
    return receipt
