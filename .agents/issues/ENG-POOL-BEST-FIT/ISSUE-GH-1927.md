ID: ISSUE-GH-1927
Title: `GPUModelRunner::connector_stored_blocks_` is a `request-id -> int` map that is inserted into and **never erased anywhere in the tree**, so a KV-connector server leaks one entry per request served for the life of the process. Found while auditing every request-keyed container in the runner for #1922. Honest scale: tens of bytes per request, and inert unless a worker-capable KV connector is installed, so it is NOT #1922's 2 GiB per request and is not claimed to be — but it is an unbounded request-keyed container in the request path, which is exactly the class of defect #1922 sent someone looking for, and the entry is dead the moment the request finishes (the count exists to deduplicate stores within ONE request's chunked prefill). NOT fixed in the `ENG-POOL-BEST-FIT` flow: it is a different behaviour with a different test surface — the gate is a KV-connector test that serves two requests and asserts the first is not retained — in a file that row's allocator change does not touch. Listed under `## Owed` O4 in [`pool-best-fit-retention.md`](../../specs/pool-best-fit-retention.md)
Row: ENG-POOL-BEST-FIT
State: UNKNOWN
Kind: bug
GitHub: 1927
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:724`

### Frozen archive evidence

> | [#1927](https://github.com/mudler/vllm.cpp/issues/1927) | `ENG-POOL-BEST-FIT` | `GPUModelRunner::connector_stored_blocks_` is a `request-id -> int` map that is inserted into and **never erased anywhere in the tree**, so a KV-connector server leaks one entry per request served for the life of the process. Found while auditing every request-keyed container in the runner for #1922. Honest scale: tens of bytes per request, and inert unless a worker-capable KV connector is installed, so it is NOT #1922's 2 GiB per request and is not claimed to be — but it is an unbounded request-keyed container in the request path, which is exactly the class of defect #1922 sent someone looking for, and the entry is dead the moment the request finishes (the count exists to deduplicate stores within ONE request's chunked prefill). NOT fixed in the `ENG-POOL-BEST-FIT` flow: it is a different behaviour with a different test surface — the gate is a KV-connector test that serves two requests and asserts the first is not retained — in a file that row's allocator change does not touch. Listed under `## Owed` O4 in [`pool-best-fit-retention.md`](../../specs/pool-best-fit-retention.md) | bug |

## Resolution

-
