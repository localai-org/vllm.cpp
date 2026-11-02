# FP16 Wave 1 cached RoPE scoped re-review

Verdict: PASS. Findings: NONE. The original P2 and P3 findings are resolved.
Product/test head reviewed: `0dacdf4c7`; final evidence-only head: `4c416296c`.
Spec-first repair commit: `035fc6692`. Original Wave1 head: `6c785c4a8`.
Independent scratch: `/tmp/lumen-rope-rereview`.
Evidence: `/tmp/rope-rereview-evidence`.

The repair restores the original pinned vLLM F32/BF16 cached-RoPE matrix and retains the F16 extension. It covers 288 dtype/shape combinations: maximum positions11/4096/32768, both conventions, rotary32, heads32/108, sequences11/1024, optional keys and both head strides. All input, cache, output and expected rounding use the selected dtype. The 144 alternate flattened calls materialize logical values and use the existing rank3 View interface. The existing strided calls preserve padding and nonrotary tails. No product rank contract changes.

The stale gamma-only comment now accurately describes the shared scalar arm. Removing comments yields byte-identical old and new CUDA source; both files have4253 lines. Registered guards, all numerical product code and the independent activation reference are unchanged. The final evidence commit changes no source/test bytes.

Independent source compilation built the repaired scalar test and three separately constructed production-call deletion variants from the frozen checkout. Each variant removes exactly one F32, BF16 or F16 RopeFromCache production call while leaving other dispatch intact. Binaries and sources have unique paths; production files are never mutated.

Canonical record validation, repair-range commit style/trailers and whitespace checks pass. The linked review tree and index are clean.

Full preflight is not green. The original full run was interrupted before its default eight-worker compile stage. The unchanged standalone checker then finished with exit1,44/829 failures, dominated by CUDA12.4 rejecting default compute121a and the existing syntax-only nvlink scratch-object failure. Earlier full-run failures include the branding baseline already corrected by the coordinator and rocprof invalid-page metadata. This scoped test/comment repair does not repeat that unrelated full run. Actual legacy CUDA source compilation and the prior broad Wave1 independent numerical/mutation review remain separately recorded.

This review supplies no full primary-model runtime, auto FP16 model activation, real-model token parity, performance acceptance, or unavailable-board runtime claim. Wave4 still owns production model-dtype propagation.

## Independent runtime and mutation results

The newly compiled full scalar suite passes14 cases and3428 assertions, with no skipped cases. The restored cached-RoPE suite passes1 case and1728 assertions, covering288 original/extended combinations plus144 flattened-view calls. Each independent production-call deletion compiles/links successfully and then exits1 for the intended missing dtype: F32 at its first case, BF16 after576 prior assertions, F16 after1152 prior assertions. The clean cached suite then exits0. Original coverage-gap proof remains recorded in the prior independent review. No production source or test bytes were edited for mutation; external variants and unique executables are under the evidence directory.

All GPU commands use approved escalated execution through flock /home/user/gpu.lock. The full scalar and mutation executables were independently compiled from the immutable linked checkout; unchanged numerical product objects are reused from the preceding independent review after verifying source identity. Full legacy architecture compiles are not repeated for a comment-only source delta. All three independently constructed mutation binaries and the restored scalar binary have unique immutable paths.

Final restoration.json verifies reviewed source/test hashes unchanged, the exact0dacdf4c7 head, and an empty tracked-tree/index status. The final4c416296c evidence-only commit has identical product/test bytes. The operator separately reports its own full14/3428 scalar rerun PASS. This review does not substitute that report for the independent runtime results above.
