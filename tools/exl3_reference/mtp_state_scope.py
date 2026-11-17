"""Initialized read scopes for bounded integrated target/MTP observations.

Host-only admission helpers. These do not read tensors, replace model math, or
establish numerical parity. Slot ownership comes from actual active metadata;
initialization comes from previously observed completed producers.
"""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def integrated_batch_reads(request_ids, tokens, positions, query_start_loc, seq_lens):
    """Admit actual unpadded C1..4 cold P128 or bounded speculative rows.

    Request identity and row offsets come from the prepared batch. This does
    not infer a transition from requested concurrency or from output timing.
    """
    _require(isinstance(request_ids, list) and 1 <= len(request_ids) <= 4 and
             all(isinstance(v, str) and v for v in request_ids) and
             len(set(request_ids)) == len(request_ids), "invalid live request ownership")
    count = len(request_ids)
    _require(len(query_start_loc) == count + 1 and query_start_loc[0] == 0 and
             all(type(v) is int for v in query_start_loc) and
             query_start_loc[-1] == len(tokens) and 0 < len(tokens) <= 512,
             "invalid active token offsets")
    _require(all(type(t) is int and 0 <= t < 248320 for t in tokens), "invalid active token IDs")
    _require(len(seq_lens) == count and all(type(s) is int and 128 <= s <= 160 for s in seq_lens),
             "invalid bounded sequence lengths")
    result, expected_positions = [], []
    for row, request in enumerate(request_ids):
        begin, end = query_start_loc[row:row + 2]
        length, seq = end - begin, seq_lens[row]
        cold = length == seq == 128
        _require(cold or (1 <= length <= 4 and seq - length >= 128), "unsupported target row")
        result.append({"request_id": request, "row": row, "begin": begin, "end": end,
                       "cold_prefill": cold})
        expected_positions.extend(range(seq - length, seq))
    _require(positions == [expected_positions] * 3, "actual positions differ from live rows")
    _require(all(v["cold_prefill"] == result[0]["cold_prefill"] for v in result),
             "mixed prefill/spec scope is not this bounded transition")
    return result


def speculative_state_reads(query_start_loc, state_indices, accepted,
                            capacity, conv_widths, ssm_slots):
    """Select consumed seeds and every written FP32 token snapshot.

    The previous accepted count selects SSM column accepted-1 and a three-value
    Conv history at offset accepted-1. The new Conv producer writes 2+Q values;
    SSM writes columns [0,Q). Physical capacity beyond those scopes is unread.
    This matches the pinned causal_conv1d.py contract and native spec kernels.
    """
    _require(type(capacity) is int and capacity > 0, "invalid state capacity")
    _require(isinstance(state_indices, list) and 1 <= len(state_indices) <= 4,
             "requires one to four active speculative requests")
    count = len(state_indices)
    _require(len(query_start_loc) == count + 1 and query_start_loc[0] == 0 and
             all(type(v) is int for v in query_start_loc), "invalid query offsets")
    _require(len(accepted) == count, "missing accepted lengths")
    owned = set()
    result = []
    for row, indices in enumerate(state_indices):
        _require(isinstance(indices, list) and len(indices) == 4 and
                 all(type(v) is int and 0 <= v < capacity for v in indices),
                 "requires four valid snapshot slots per request")
        _require(len(set(indices)) == 4 and not owned.intersection(indices),
                 "aliased request/snapshot ownership")
        owned.update(indices)
        length = query_start_loc[row + 1] - query_start_loc[row]
        n = accepted[row]
        _require(1 <= length <= 4 and type(n) is int and 1 <= n <= 4,
                 "invalid query/accepted length")
        base, initial = indices[0], indices[n - 1]
        _require(initial in ssm_slots, "uninitialized consumed SSM snapshot")
        _require(conv_widths.get(base, 0) >= n + 2,
                 "uninitialized consumed Conv history")
        result.append({"row": row, "query_length": length,
                       "previous_accepted_tokens": n,
                       "conv_slot": base, "conv_before": [n - 1, n + 2],
                       "conv_after": [0, 2 + length],
                       "ssm_before": initial, "ssm_after": indices[:length]})
    return result


def initialized_kv_addresses(page_size, capacity, seq_len, query_positions,
                             blocks, slots, stage):
    """Map only initialized logical rows, excluding unwritten cache capacity.

    Blocks must already be sliced to the columns required by this live row.
    Provisional KV from an earlier iteration beyond the current prefix is not
    read. Before excludes this query; after includes its actual written rows.
    """
    _require(type(page_size) is int and page_size > 0 and
             type(capacity) is int and capacity > 0 and
             type(seq_len) is int and 1 <= seq_len <= 160,
             "invalid bounded KV geometry")
    _require(stage in ("before", "after"), "invalid KV boundary")
    _require(isinstance(query_positions, list) and bool(query_positions) and
             all(type(v) is int for v in query_positions) and
             query_positions == list(range(seq_len - len(query_positions), seq_len)) and
             query_positions[0] >= 0, "noncontiguous query positions")
    _require(len(blocks) == (seq_len + page_size - 1) // page_size and
             all(type(b) is int and 0 <= b < capacity for b in blocks) and
             len(set(blocks)) == len(blocks), "invalid active KV pages")
    addresses = [(blocks[p // page_size], p % page_size) for p in range(seq_len)]
    _require(len(slots) == len(query_positions) and
             all(type(s) is int for s in slots) and
             slots == [addresses[p][0] * page_size + addresses[p][1]
                       for p in query_positions], "slot mapping differs from query")
    return addresses[:query_positions[0] if stage == "before" else seq_len]
