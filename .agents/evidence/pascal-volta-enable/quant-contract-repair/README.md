# Dense shard descriptor contract repair

Issue: `ISSUE-LOCAL-01M63WWWN3EETEG29PF1ME9BNN`.
Spec-first commit: `85bd504e5`.

Production code is unchanged. Three host cases call `vt::MergedGemmDense`
with malformed descriptors and operands. They require the exact diagnostic
and `std::runtime_error` before tensor access. Count mismatch, minimum arity,
null arrays, fused providers, unsupported epilogue and arity, unexpected
output, and missing SwiGLU output each have explicit checks.

The four production guard deletions pass the original valid-only tests.
`baseline.txt` records four exit-zero mutations. After adding tests, each
mutant compiles and links, then exits 1 on failed exception assertions.
`repaired.txt` records the compile, link, and runtime status. The selected
malformed case uses allocated operands or zero contractions, so these four
deletion runs do not depend on a crash. Separate valid-count null-array and
missing-output cases pin the early rejection contract in the restored suite.

The restored complete host suite passes 13 cases and 107 assertions.
`restored-run.txt` contains the full result. No GPU command runs. Scratch
objects use the actual changed linear test and merged GEMM production source,
with immutable Qwen and CUDA MoE objects linked against a separate archive.
The recipe adapts `/tmp/review-quant.py`, using one archive and executable.

`git diff --check` and `scripts/check-agent-record.py` pass. Startup preflight
was invoked and interrupted to avoid another parallel repository-wide run.
It is not green evidence. Full CUDA, public model, oracle, and performance
validation remain separate operator gates. The issue stays OPEN until landing.
