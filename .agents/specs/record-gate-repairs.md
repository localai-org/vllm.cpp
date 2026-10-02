# Spec: preflight record/doc gate repairs (five red gates on main)

## Scope

Repair the five red record/doc gates on `origin/main` by reconciling the
records to each checker's contract. No checker is weakened, no benchmark or
document is deleted.

1. `scripts/check-release-binary-contract.py` — `.agents/roadmap_v1.md` lost
   the `| REL | ROAD-V1-RELEASE |` release-lifecycle anchors (v0.0.2 published;
   Windows W14-W16 pending). Restore the historical row text (last carried at
   `10a64f827`, `docs(release): add native Windows prerelease lanes (#117)`),
   which is the state the checker's `EXACT_MACHINE_FIELDS`
   (`archive_claims = published-v0.0.2`) and `PUBLIC_PENDING_MUTATIONS` pin.
2. `scripts/check-windows-release-state.py` — the roadmap must carry exactly
   one `<!-- ENG-RELEASE-WINDOWS: state=ACTIVE publication=pending
   artifact=unpublished -->` anchor plus the pending `v0.0.3-pre.1`
   publication/audit phrase. Both were dropped from the roadmap in the same
   restructure; restore them (anchor placement per `a0a8fedbe`).
3. `scripts/check-benchmark-index.py` — index every detail file under
   `docs/benchmarks/` in the `## Benchmark detail index` table of
   `docs/BENCHMARKS.md`, one row per file, ID equal to the file stem.
4. `scripts/check-env-doc.py` — classify `VT_VK_DISABLE` and
   `VT_VK_DISABLE_PAGED_ATTN` (kernel-internal bisect hooks,
   `src/vt/vulkan/vulkan_ops.cpp:982` and `:1296`) onto
   `scripts/env-doc-allowlist.txt`; document `VT_VK_FENCE_TIMEOUT_MS` (a
   user-facing operational watchdog that aborts on a wedged Vulkan dispatch,
   `src/vt/vulkan/vulkan_context.cpp:174`) in `docs/ENVIRONMENT.md`.
5. `scripts/check-oracle-pins.py` — the sync `4f11dfc10` (2026-09-26) advanced
   the parity pin to `a7c23ac96d` / `0.3.0.dev267` but left the restated
   `<!--pin:commit-->`/`<!--pin:label-->` spans in `.agents/NOW.md`,
   `docs/FEATURES.md`, `docs/benchmarks/how-we-measure.md`,
   `docs/benchmarks/speculative-decoding.md` and
   `docs/benchmarks/vllm-online-serving.md` at the prior pin. Restate the
   current pin; the spans are the checker's only read of these surfaces.

## Gates

All five checkers exit 0; `check-agent-record.py` stays 0; the record suite
passes.

## Stop conditions

A checker whose expectation contradicts the tree's true state is reconciled to
the true state and the conflict named in the commit body, never silenced.

## Owed

- ISSUE-LOCAL-01M3WMKTBEDAHF226YSATSR7BF — the five-gate preflight repair this spec owns until it lands
