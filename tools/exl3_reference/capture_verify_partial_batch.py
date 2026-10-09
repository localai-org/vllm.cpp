#!/usr/bin/env python3
"""Executed original C2/C3 Q4 operators on derived pinned worker operands.

Bounded operator fixtures, not model trajectories or reference promotion.
"""
import argparse
import json
import os
from pathlib import Path

from capture_runtime_layout import PROFILE, verify_inputs
from capture_verify_batch import query_bytes
from capture_verify_family import BASE_ROWS, PAGE, SOURCE_SHA, VERIFY_SHA
from compare_projection import blob
from extract_projection import digest, headers, write_safetensors

BLOCKS, COLUMNS = 63, 164
REQUEST_BLOCKS = [list(range(r * 21, (r + 1) * 21)) for r in range(3)]


def partial_cases():
    cases = []
    for n in (2, 3):
        for length, name in ((4096, "short"), (32768, "long")):
            variants = [("ordinary", list(range(n)), None), ("rounded", list(range(n)), None)]
            if length == 32768:
                variants += [("permuted", list(reversed(range(n))), None),
                             ("tail-values-r1", list(range(n)), 1)]
            for variant, order, mutate in variants:
                base = [length - r * 17 for r in range(3)]
                maximum = (length + PAGE - 1) // PAGE * PAGE if variant == "rounded" else length
                cases.append(dict(id=f"c{n}-q4-{name}-{variant}", request_ids=order, max_keys=maximum,
                                  base_lengths=base, lengths=[base[r] for r in order],
                                  mutate_request=mutate))
    return cases


def cache_bytes(keys, values, lengths, mutate=None, poison=0):
    if len(keys) != BASE_ROWS * 1024 or len(values) != len(keys):
        raise ValueError("requires pinned4100-row initialized cache")
    storage = bytearray([poison]) * (BLOCKS * PAGE * 2048)
    for request, length in enumerate(lengths):
        for row in range(length):
            block = REQUEST_BLOCKS[request][row // PAGE]
            for head in range(4):
                src = (((row + request * 337) % BASE_ROWS) * 4 + head) * 256
                dst = block * PAGE * 2048 + (row % PAGE) * 2048 + head * 512
                storage[dst:dst + 256] = keys[src:src + 256]
                storage[dst + 256:dst + 512] = (
                    bytes([0x38]) * 256 if request == mutate and row >= length - 3
                    else values[src:src + 256])
    return storage


def capture(args):
    headers.require(not args.output.exists() and not args.output.with_suffix('.json').exists(),
                    "refusing existing partial-batch capture")
    headers.require(digest(args.source.read_bytes()) == SOURCE_SHA, "source worker changed")
    reference = verify_inputs(args.reference_manifest, args.model, args.image_identity)
    index = headers.read_shard_header(args.source)
    base_query = blob(args.source, index, 'query')[1]
    keys = b''.join(blob(args.source, index, f'k_page{i}')[1] for i in range(3))
    values = b''.join(blob(args.source, index, f'v_page{i}')[1] for i in range(3))
    import yaml
    for key, value in yaml.safe_load(PROFILE.read_text())['env'].items():
        if value is not None:
            os.environ[key] = str(value).replace('{model_dir}', str(PROFILE.parent))
    import torch
    from exl3xpu import shared_kv_verify as verify
    from torch.utils._python_dispatch import TorchDispatchMode
    library = Path(os.environ['EXL3_M04_LIBRARY'])
    headers.require(digest(library.read_bytes()) == os.environ['EXL3_M04_LIBRARY_SHA256'],
                    "pinned library changed")
    headers.require(digest(Path(verify.__file__).read_bytes()) == VERIFY_SHA,
                    "pinned wrapper changed")
    torch.ops.load_library(str(library))
    tensors = {'base_keys': ('U8', [BASE_ROWS, 4, 256], keys),
               'base_values': ('U8', [BASE_ROWS, 4, 256], values)}
    records, canonical = [], {}
    for number, case in enumerate(partial_cases()):
        order = case['request_ids']; n = len(order)
        host = cache_bytes(keys, values, case['base_lengths'], case['mutate_request'])
        storage = torch.frombuffer(host, dtype=torch.uint8).to('xpu')
        key = storage.as_strided((BLOCKS, PAGE, 4, 256),
                                (PAGE * 2048, 2048, 512, 1)).view(torch.float8_e4m3fn)
        value = storage.as_strided(key.shape, key.stride(), 256).view(torch.float8_e4m3fn)
        query_raw = b''.join(query_bytes(base_query, r) for r in order)
        query = torch.frombuffer(bytearray(query_raw), dtype=torch.float16).reshape(n * 4, 24, 256).to('xpu')
        table = [REQUEST_BLOCKS[r] + [-1] * (COLUMNS - 21) for r in order]
        logical = list(range(0, 4 * n + 1, 4))
        descale = torch.ones(1, dtype=torch.float32, device='xpu')
        data = dict(q=query, k=key, v=value, out=torch.empty_like(query),
                    max_seqlen_q=4, max_seqlen_k=case['max_keys'],
                    softmax_scale=0.0625, causal=True,
                    cu_seqlens_q=torch.tensor(logical, dtype=torch.int32, device='xpu'),
                    seqused_k=torch.tensor(case['lengths'], dtype=torch.int32, device='xpu'),
                    block_table=torch.tensor(table, dtype=torch.int32, device='xpu'),
                    k_descale=descale, v_descale=descale)
        headers.require(verify.eligible(data), 'original declined partial uniformQ4')
        calls = []

        class Witness(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if str(func) == 'b70_exl3_attention.shared_kv_verify_out.default':
                    calls.append(dict(packed_query_shape=list(args[0].shape),
                                      physical_offsets=args[4].cpu().tolist(),
                                      max_keys=args[-3], splits=args[-2], tile=args[-1]))
                return func(*args, **(kwargs or {}))

        with Witness():
            result = verify.run(data)
        torch.xpu.synchronize()
        raw = result.cpu().contiguous().numpy().tobytes()
        headers.require(len(calls) == 1 and calls[0]['physical_offsets'] == list(range(n + 1)),
                        'actual single packed batch invocation missing')
        headers.require(result.data_ptr() == data['out'].data_ptr(), 'original lost output owner')
        headers.require(torch.isfinite(result).all().item(), 'nonfinite original output')
        headers.require(raw == verify.run(data).cpu().contiguous().numpy().tobytes(), 'repeat changed')
        headers.require(storage.cpu().numpy().tobytes() == bytes(host), 'original wrote KV')
        headers.require(query.cpu().numpy().tobytes() == query_raw, 'original wrote query')
        for slot, request in enumerate(order):
            piece = raw[slot * 49152:(slot + 1) * 49152]
            label = (n, tuple(case['base_lengths']), case['max_keys'], request)
            if case['mutate_request'] is None:
                headers.require(piece == canonical.setdefault(label, piece), 'permutation changed output')
            elif request != case['mutate_request']:
                headers.require(piece == canonical[label], 'mutation crossed request')
            else:
                headers.require(piece[:12288] == canonical[label][:12288], 'row0 saw future tail')
                headers.require(piece[12288:] != canonical[label][12288:], 'tail mutation has no witness')
        prefix = f'case{number}'
        tensors[prefix + '_query'] = ('F16', [n * 4, 24, 256], query_raw)
        tensors[prefix + '_output'] = ('F16', [n * 4, 24, 256], raw)
        records.append(case | dict(prefix=prefix, route=calls[0], logical_offsets=logical))
        print('ORIGINAL_PARTIAL_Q4_PASS', case['id'], flush=True)
        del data, result, query, key, value, storage, host, descale
    write_safetensors(args.output, tensors, {'scope': __doc__})
    report = dict(schema='b70-exl3-C2-C3-Q4-original-v1', cases=records,
                  source_sha256=SOURCE_SHA, image=args.image_identity,
                  checkpoint=reference['checkpoint']['identity'], wrapper_sha256=VERIFY_SHA,
                  library_sha256=digest(library.read_bytes()), tool_sha256=digest(Path(__file__).read_bytes()),
                  physical_blocks=BLOCKS, page=PAGE, columns=COLUMNS,
                  request_blocks=REQUEST_BLOCKS, base_rows=BASE_ROWS,
                  capture_sha256=digest(args.output.read_bytes()),
                  peak_allocated_bytes=torch.xpu.max_memory_allocated(), scope=__doc__)
    with args.output.with_suffix('.json').open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'reference-manifest', 'model', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--image-identity', required=True)
    capture(parser.parse_args())
