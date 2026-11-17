# B70 EXL3 publication evidence

This appendix records the bounded publication repair on 2026-10-07. Upstream
base is `452617154b8231a0fea4b5331b246427d96622b8`; the tested code head is
`11cb381f61b37c9aba88ca63d8f60180e4cf5ef2`. A later documentation-only commit
packages these receipts; its exact final identity and commit range are in the
review archive's PUBLIC_TRANSFER_MANIFEST.json. The source files used by the
binaries do not change in that packaging commit.

Every JSON here is a public-safe derivative. Its `sources` identify original
artifacts by logical ID, original byte count and SHA-256. Local paths and ANSI
color are replaced or removed; selected failure excerpts state omitted output.
Historical raw receipts are retained privately without rewriting their hashes.
MANIFEST.json supplies separate hashes for these new derivatives and this README.
No HANDOFF.txt, model weights, private machine configuration or full state tensors
are included. Native prompt/output IDs and small cycle records are retained.

## Current bounded results

- R1: seven model-free full-client tests pass; the old missing-gauge defect fails
  six subcases before the repair. CTest registers the new suite.
- CPU library/model and XPU production server/model rebuilds exit zero after
  the rebase; two affected CPU CTest entries pass. Exact commands/logs/exits are
  in `rebase-*-build.json` and `rebase-cpu-test.json`. Fresh original configurations
  and earlier focused suites retain their separate historical identities.
- The corrected actual-server target and MTP3 clients each pass eleven bounded
  checks. Target runs without a draft-map mount/environment variable. MTP3 uses
  the bundled exact metadata. Both gauges are present and zero, with observed
  timestamps. Counter activity does not prove depth; launch commands record three.
- Fresh unchanged-control/current native sentinel runs each pass 610 assertions:
  all 256 output IDs and 113 non-timing cycles match. Current native lifecycle
  passes 483 assertions and matches the historical unchanged-control JSON,
  including 1/4/2/1 turnover, poisoned spare state and graph/eager/graph.
- The common Qwen conflict preserves both upstream TT state-slot invalidation
  and XPU attention-policy invalidation. The merge-tree check is clean; source
  checks are recorded. No Tenstorrent runtime is available to certify that device.

Executable hashes, native resource observations, HTTP JSON and launch configurations
are included. These are author-run scoped results; neither a self-audit nor the
narrow Pro review discharges independent full static/mutation or reference review.
The one sentinel pair is continuity evidence, not a speedup or Python benchmark.

## Red and unavailable gates

The full automatic preflight was not repeated on the new base/repair head.
`historical-preflight-comparison.json` and failed-section excerpts describe the
actual earlier pristine/final runs, which exited one. They are not current green
results. `failure-classification.json` distinguishes baseline-proven gate names,
unresolved individual causes, the newly observed automatic Windows failure and
unavailable checks. The focused historical static portability pass uses a different
configuration and is not an MSVC build. Maintainer disposition is still needed;
a draft does not waive required gates.

The strict historical HTTP cached-token metric remains FAIL despite a distinct
functional restore trace. Default numerical/reference and integrated state gates
remain open, as do stock current-pin EXL3 runtime qualification, sixteen slots,
maximum context and full stochastic/logprob parity. See the
[continuity report](../b70-exl3-serving-continuity.md) for the tested envelope.
