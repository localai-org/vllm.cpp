# SPEC — `TT-DRAFTER-STUDY` — the DFlash2 pattern: study the bundle's on-card speculative-decode runtime as a design reference for our models

- Owning row: `TT-DRAFTER-STUDY`
- Kind: study/design (a reference-implementation study with a written deliverable; no product code)
- Issue: created in the same flow (row-scoped local issue)

## Why this row exists

The developer's reframe (2026-11-03): the dflash2-vision bundle's
advantage over our P150 numbers is **~500x, and most of it is the drafter
and other shared code we will need for other models they do not cover as
well**. The bundle is Apache-2.0 end to end (verified: the bundle, the
kernel sources' SPDX headers, the base model, the drafter), so the study
is a licensed read of a working reference implementation — not a
clean-room exercise.

The bundle: `mando2222/qwen3.8-27b-dflash2-vision-p300x2-q4kv-v6-thin-cp4`
@ `7855cf0d2982064350991bdb8e1ec1db82f914b3`; the drafter model:
`incoai/Qwen3.8-27B-DFlash2` @ `dedf8df68adfb1afeaf7b7480c0a0243108177b4`;
the plugin source: `plugin-step-rebuild` @ `d564ae09a6fe1d63428251ec4c665e7dff8652ae`;
the model source: `thin/src-cp81` @ `e3ab64576fb6ccca73e1bf12322b9702fada99e1`
(all from the bundle's `PROVENANCE.json`, which also carries the runtime
lock: python 3.12.14, uv 0.12.11, constraints.txt sha).

## The questions the study answers

1. **The DFlash2 mechanism**: what is the draft/verify/acceptance loop?
   (their README: "the whole DFlash iteration — drafting, verification,
   acceptance, sampling — runs on the card; the host touches each decode
   step once"; greedy is token-for-token exact; sampled follows the
   target distribution exactly via rejection sampling). HOW: read the
   plugin's decode path (extra_code/qwen38/attention/decode.py,
   attn_decode kernels) and the drafter integration.
2. **The on-card decode loop**: the property our host-free decode work
   chases — "the host touches each decode step once" and "decode keeps
   96–97% of idle speed under host load". What is the loop's structure
   (the capture/replay pattern, the command queue, the slot machinery)?
3. **The mesh execution**: 1×4 FABRIC_1D — how are the layers/heads/experts
   sharded across 4 chips? (This is the hardware multiplier our single
   P150 cannot take, but the pattern matters for a future multi-chip
   box.)
4. **Portability to OUR models**: what does the drafter pattern need
   from a model? (a draft model — trained or derived; the verify pass;
   the acceptance test). Which of OUR models could carry it (kolibri1?
   flint? the MoE ones?), and what is the lift?
5. **The adoption sketch**: the deliverable is a written design note —
   what adopting the pattern for one of our models would take, in
   stages, with the risks named. NOT an implementation.

## Deliverable

`docs/design/tt-drafter-study-20261103.md` (or the docs/ location the
tree prefers): the study note answering the five questions, with
file/line citations into the bundle's sources (downloaded to a scratch
dir, NOT committed — it is external code; the note cites it, the tree
does not carry it), the license/attribution paragraph (Apache-2.0 end to
end, the SPDX notices, the PROVENANCE chain), and the adoption sketch
for our models.

## Method

- `tt-model pull` is not available on this host (no tt-model client);
  the bundle's files are fetched directly from HuggingFace (the
  extra_code/ tree + PROVENANCE.json + README + constraints.txt — the
  code is small; the weights are NOT downloaded).
- Read the sources; cite file:line; no assumptions from behavior —
  every claim in the note is grounded in a cited line.

## Gates

check-agent-record, style, trailers, preflight (named host dotnet
failure). Device-free row. The note is a public document (docs/), so
the public-docs trigger applies: it lands with its own citation.

## Non-goals

- No product code. No tt-model installation. No weight downloads. No
  performance claims. No adoption — the sketch is the deliverable; the
  adoption row files after the study is read.

## Git integration

One pull request, `row/TT-DRAFTER-STUDY`, spec committed first.
