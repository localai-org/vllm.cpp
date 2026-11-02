# KV offload and external KV caches

LumenEngine can keep KV blocks outside the GPU's own paged cache and reuse them
later: a local CPU + disk tier, or a remote [LMCache](https://github.com/LMCache/LMCache)
store over the `lm://` wire. Both are **opt-in and off by default**; with no KV
transfer configured the engine runs exactly as it always has.

Selection mirrors vLLM's own CLI: `--kv-transfer-config '<json>'`, taking the
same JSON object vLLM takes, so a config written for vLLM works here.

> **Read the [Limitations](#limitations) section before enabling anything.** One
> of the two connectors is scheduler-side only and the engine will refuse to run
> it. That refusal is deliberate, and the reason is correctness.

## The flag

```
server --model <dir> --kv-transfer-config '<json>'
```

The JSON is parsed into the same `vllm::KVTransferConfig` the C++ / C-ABI API
takes programmatically (`EngineParams::kv_transfer_config`):

| Key | Required | Meaning |
|---|---|---|
| `kv_connector` | yes | Connector name. Registered: `LMCacheConnector`, `OffloadingConnector` |
| `kv_role` | yes when `kv_connector` is set | `kv_producer`, `kv_consumer`, or `kv_both` |
| `kv_connector_extra_config` | no | Per-connector settings (tables below). Values may be strings, numbers or booleans |
| `engine_id` | no | Engine identity for KV transfers; defaults to a fixed placeholder |
| `kv_load_failure_policy` | no | `fail` (default) or `recompute`. `fail` surfaces a torn/missing block as an error instead of quietly recomputing it |

Anything else is refused at startup: malformed JSON, a non-object document, an
unknown key, a wrongly-typed value, an unknown role, or a connector name that is
not registered (the error lists the registered names). Omit the flag entirely and
no connector is built.

## LMCache (`lm://` remote KV), the connector that works end to end

This is the connector to use. It is a pure-C++ `lm://` client (no in-process
`lmcache`, no Python) whose keys agree byte-for-byte with a real Python
vLLM + LMCache peer, and whose worker half is implemented: the runner moves K and
V planes between the KV pages and the store around each forward.

Start an LMCache server (from a normal `lmcache` install):

```bash
python -m lmcache.v1.server localhost 65432
```

Then run the server against it:

```bash
server --model /models/Qwen3-0.6B \
  --kv-transfer-config '{
    "kv_connector": "LMCacheConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "host": "127.0.0.1",
      "port": 65432,
      "hash_algo": "vllm",
      "chunk_tokens": 256
    }
  }'
```

`kv_connector_extra_config` keys:

| Key | Default | Meaning |
|---|---|---|
| `host` | `127.0.0.1` (or `VT_LMCACHE_HOST`) | LMCache server host |
| `port` | `65432` (or `VT_LMCACHE_PORT`) | LMCache server port |
| `hash_algo` | `blake3` | Key derivation. `vllm` (alias `sha256_cbor`) is the **peer-interoperable** mode: vLLM's own `sha256_cbor` rolling chunk hash, byte-identical to a Python vLLM+LMCache peer. `blake3` is our self-consistent mode, fine when we are the only writer |
| `chunk_tokens` | `256` in `vllm` mode, the KV block size in `blake3` mode | Tokens per stored chunk. In peer mode this must match the peer's LMCache `chunk_size` |
| `model_name` | from the model identity | Folded into the cache key; two different models can never collide |
| `world_size` / `worker_id` | `1` / `0` | Folded into the key, mirroring LMCache's per-worker sharding |
| `offload_prompt_only` | `true` | Store only prompt blocks, never generated tokens |
| `save_unfull_chunk` | `false` | Whether a trailing partial chunk is stored |
| `none_hash_seed` | `0` | Seed of the rolling hash chain; must match the peer |

To interoperate with a **real vLLM + LMCache peer**, use `"hash_algo": "vllm"`
and the same `chunk_tokens` as the peer's LMCache `chunk_size` (LMCache's default
is 256). Under those settings our `CacheEngineKey` strings, chunk boundaries and
folded hashes are byte-identical to LMCache's own `ChunkedTokenDatabase`, and a
chunk written by the peer is found and decoded by us (and vice versa). See the
LMCache W4/W5 rows in [the benchmark record](BENCHMARKS.md) for the exact evidence.

### Identity and refusal

A cache entry that does not provably belong to this model/configuration is
**refused, never served**. The KV geometry recorded in the connector (layer
count, KV hidden dim, chunk tokens, element size) is re-verified against the
runner's actual KV cache before any byte is moved in either direction, and a
mismatch aborts the operation rather than copying under a wrong layout. A chunk
the scheduler recorded as a prefix hit that then turns out to be absent is an
error, not a silent recompute: it means a torn store or an eviction mid-decode,
and hiding it would surface as plausible-but-wrong output.

The disk tier applies the same principle harder, with a 27-field identity header
(model, config digest, weight quantization, KV dtype and quant mode, rope config,
sliding window, page/block sizes, hash algorithm, hash-chain seed, parallelism,
format version, …). Upstream vLLM's `fs` tier writes a `config.json` it never
reads back; we refuse on every one of those fields instead.

## CPU + disk offload (`OffloadingConnector`), REFUSED by the engine today

`kv_connector: "OffloadingConnector"` selects the local tiered CPU + disk cache.
Its scheduler half is real and gated: on a restart, a prefix that is only on disk
is promoted and prefill is shortcut (32 of 48 tokens saved, 2 of 3 blocks, in the
e2e test).

**Its worker half does not exist**, so the engine refuses to wire it:

```
$ server --model /models/Qwen3-0.6B \
    --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
                           "kv_connector_extra_config":{"root_dir":"/var/tmp/kv"}}'
server: fatal: REFUSING KV connector 'OffloadingConnector' on device 'cuda': this
connector has no worker half on this device, but its scheduler half shortcuts
prefill for externally matched blocks. Running it would make the model attend
over KV that is never written into the device's KV pages, i.e. silently wrong
output. Connectors whose worker half can move KV bytes on 'cuda':
LMCacheConnector. Select one of those, or unset the KV transfer config to run
with no connector.
```

This is not a configuration mistake you can work around, and it is not device
specific: the refusal fires on CPU too, because the missing piece is the byte
movement itself, not a device port. Until the worker half lands, the disk
connector is usable only through its scheduler-level API (which is what the test
suite exercises), never through a serving engine.

For reference, its `kv_connector_extra_config` keys are `root_dir` (required),
`num_cpu_blocks` or `cpu_bytes_to_use`, `fs_capacity_bytes`, `eviction_policy`
(`lru`), `group_idx`, `offload_block_tokens` and `offload_prompt_only`.

## Limitations

Stated plainly, because a KV cache that quietly serves the wrong bytes is worse
than no KV cache:

- **The CPU/disk `OffloadingConnector` has no worker half.** Its
  `ConnectorLoadJob`s are consumed by nothing and its bytes live in a host
  buffer that is never copied into a KV page. It is therefore **refused at engine
  construction on every device** rather than silently producing wrong output.
  Implementing that worker half (the mirror of the LMCache store/load path in
  `src/vllm/v1/worker/gpu/runner.cpp`) is open, tracked work. It is **not**
  claimed as shipped.
- **No binding throughput number for LMCache yet.** Correctness is closed:
  connector-ON generation is bit-identical to connector-OFF, both after an
  in-process restart and from a cold second process. Speed is not: the only
  end-to-end loop measured so far is a 125M-parameter model, where the TCP
  round-trip is comparable to the prefill it saves, so no speedup is claimed. The
  owed measurement is an every-axis TTFT/throughput grid on a larger model with a
  long shared-prefix corpus, against vLLM launched with the equivalent LMCache
  `--kv-transfer-config`, with hits proven in every arm.
- **Layerwise pipelining is not implemented.** `wait_for_layer_load` /
  `save_kv_layer` are no-ops; loads and stores happen around the forward, not
  interleaved with it.
- **Only the full-attention KV group is offloaded.** Recurrent/GDN state is not
  block-addressed and is never offloaded, so hybrid models offload their
  attention group only.
- `VT_LMCACHE_HOST` / `VT_LMCACHE_PORT` / `VT_LMCACHE_HASH_ALGO` are read as
  defaults by the client; `kv_connector_extra_config` overrides them.

## Consuming it programmatically

The flag is a thin wrapper over the library surface, which has been available all
along:

```cpp
vllm::entrypoints::EngineParams params;
params.kv_transfer_config = vllm::ParseKVTransferConfigJson(R"({
    "kv_connector": "LMCacheConnector", "kv_role": "kv_both",
    "kv_connector_extra_config": {"host": "127.0.0.1", "port": 65432}})");
auto engine = vllm::entrypoints::LoadedEngine::FromModelDir(model_dir, params);
```

Or build the `vllm::KVTransferConfig` field by field and skip the JSON. Either
way the same guard applies.
