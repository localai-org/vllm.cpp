"""Read-only tensor layout records for the pinned worker capture.

No Torch import or cache payload read is needed for a descriptor. Reading small
initialized metadata values is a separate explicit operation; cache capacity
buffers and unused block-table columns must never be serialized as values.
"""
import dataclasses
import enum
import math


def tensor_layout(tensor):
    shape, strides = list(tensor.shape), list(tensor.stride())
    offset, item_size = tensor.storage_offset(), tensor.element_size()
    storage = tensor.untyped_storage()
    storage_bytes = storage.nbytes()
    if len(shape) != len(strides) or any(type(v) is not int or v < 0 for v in shape + strides):
        raise ValueError("invalid shape/stride")
    if type(offset) is not int or offset < 0 or type(item_size) is not int or item_size <= 0:
        raise ValueError("invalid offset/element size")
    numel = math.prod(shape)
    span_elements = 0 if numel == 0 else 1 + sum((d - 1) * s for d, s in zip(shape, strides))
    end_bytes = (offset + span_elements) * item_size
    if end_bytes > storage_bytes:
        raise ValueError("view exceeds owning storage")
    base, pointer = storage.data_ptr(), tensor.data_ptr()
    if numel and pointer != base + offset * item_size:
        raise ValueError("tensor pointer disagrees with storage offset")
    return {"dtype": str(tensor.dtype), "device": str(tensor.device), "shape": shape,
            "strides_elements": strides, "storage_offset_elements": offset,
            "element_bytes": item_size, "logical_bytes": numel * item_size,
            "view_span_bytes": span_elements * item_size, "storage_bytes": storage_bytes,
            "storage_data_ptr": base, "data_ptr": pointer}


def initialized_values(tensor, max_elements=4096):
    """Only call on a known initialized active slice; enforce the host-copy bound."""
    layout = tensor_layout(tensor)
    if math.prod(layout["shape"]) > max_elements:
        raise ValueError("metadata value read exceeds bound")
    values = tensor.detach().cpu().tolist()

    def finite(value):
        if isinstance(value, list):
            return all(finite(v) for v in value)
        return not isinstance(value, float) or math.isfinite(value)

    if not finite(values):
        raise ValueError("nonfinite initialized metadata")
    return {"layout": layout, "values": values, "value_scope": "explicit initialized active slice"}


def describe(value, depth=0):
    """Describe nested runtime configuration without reading tensor values."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite runtime field")
        return value
    if hasattr(value, "untyped_storage") and hasattr(value, "stride"):
        return {"tensor_layout": tensor_layout(value)}
    if isinstance(value, enum.Enum):
        return describe(value.value, depth)
    if type(value).__module__ == "torch" and type(value).__name__ in ("dtype", "device"):
        return str(value)
    kind = type(value).__module__ + "." + type(value).__qualname__
    if depth >= 5:
        return {"object_type": kind, "scope": "nested fields omitted"}
    if isinstance(value, dict):
        return {str(k): describe(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [describe(v, depth + 1) for v in value]
    if dataclasses.is_dataclass(value):
        return {"object_type": kind, "fields": {
            f.name: describe(getattr(value, f.name), depth + 1) for f in dataclasses.fields(value)}}
    return {"object_type": kind}


def attention_call_layout(arguments):
    """Describe actual attention arguments without copying Q/K/V/output payloads.

    Only active length metadata, scalar/broadcast scales and the first page ID
    are read. The remaining block-table capacity may be uninitialized.
    """
    result = {"arguments": describe(arguments), "active_values": {}}
    for name in ("cu_seqlens_q", "seqused_k", "k_descale", "v_descale"):
        value = arguments.get(name)
        if hasattr(value, "untyped_storage"):
            result["active_values"][name] = initialized_values(value)
    table = arguments.get("block_table")
    if hasattr(table, "untyped_storage") and table.ndim == 2 and table.shape[1] > 0:
        result["active_values"]["block_table_first_column"] = initialized_values(table[:, :1])
    result["value_scope"] = "Active metadata/scales and first block column only; no Q/K/V/output values"
    return result
