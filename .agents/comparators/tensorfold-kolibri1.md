# TensorFold Kolibri 1 comparator

TensorFold is pinned here only as an external implementation and performance
comparator, in the same sense as the Qwen3.8 Flash Next record. It may ground
source inspection, mechanism attribution, and measured throughput. It does not
supply correct output and does not replace vLLM (through the
`aleph-alpha-inference` oracle plugin), Transformers, llama.cpp, or the existing
component fixtures in a correctness gate. It never enters the secondary-oracle
registry in `AGENTS.md`.

The correctness authorities for Kolibri 1 remain the `vllm` primary via the
`aleph-alpha-inference` plugin oracle and the HuggingFace `transformers`
fixtures. Nothing in this record may be read as a ratio, winner, gap, or
ceiling over any vllm.cpp arm.

The three published figures below are UNVERIFIED PUBLISHER AND PR-TITLE CLAIMS,
recorded as claims only. They are Apple-Silicon MLX GPU figures and are not
comparable to any x86 CPU arm of this repository:

- Pull request 328, `feat(kolibri1 cuda)`: runs the Kolibri 1 FP8 checkpoint
  (78.1B-A3.46B MoE) on CUDA. Head SHA
  `42382566bf9836bdf11d0d9b97f6280d75dabf96`.
- Pull request 433, `feat(mlx): Kolibri 1 on Macs, row-exact decode with
  context-copy drafts`: claims +24-29 percent decode with thinking on,
  "same scores". Head SHA `dcc9d4616dc2762e58b9753bc1374962017b35b3`.
- Pull request 437, `fix(mlx): Kolibri 1's router gives every row its one-row
  bits, so streams share forwards again`: claims 4-stream aggregate decode
  111-117 to 179-195 tok/s after making routing per-batch. Head SHA
  `b4e1e2fd0ec97be722f34e05892dafc76cdc454b`.
- Issue 434, `Kolibri 1 (MLX): shared multi-stream forwards are not bit-exact,
  so streams take turns`: the constraint the router fix answers.

All three pull requests were closed on 2026-10-06. None of the three head SHAs
is an ancestor of the pinned upstream main: each branch diverged from main
(behind by 6, 7, and 9 commits respectively at resolution time), so every claim
above attaches to an unmerged pull-request head, not to the repository pin.

The deployment recipe pin is the TensorFold head that carries the per-batch
routing fix, because the MLX serving configuration and its published multi-
stream numbers attach to that head, not to main.

What this comparator MAY answer: mechanism attribution only — per-batch MoE
routing as the enabler of multi-stream shared forwards, context-copy drafting,
and multi-stream bit-exactness constraints on shared forwards; and whether a
future same-harness, same-hardware campaign shows a vllm.cpp change narrowing a
measured gap. What it MAY NOT answer: correctness, cross-hardware ratios
against any CPU arm, or historical byte-identity of outputs.

Two named mechanism leads for vllm.cpp, recorded as leads and not commitments:
multi-stream router batching on the CPU serving path, and reuse of the
context-copy drafting idea. Neither is scheduled by this record.

```comparator-pin
id = tensorfold-kolibri1
role = implementation-performance
upstream = https://github.com/ashhart/TensorFold
pin = ed78d6fc204d89d90b045bf033d6551e7714f3a1
recipe_upstream = https://github.com/ashhart/TensorFold
recipe_pin = b4e1e2fd0ec97be722f34e05892dafc76cdc454b
scope = Kolibri 1 implementation and MLX decode performance at the unmerged per-batch-routing pull-request head
correctness = not-an-oracle
pinned_on = 2026-10-07
```
