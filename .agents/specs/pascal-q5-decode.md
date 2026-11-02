# Pascal Q5_0 decode activation sums

## Now

ACTIVE. The existing decode path diverges from the pinned GPU oracle.

## Scope

Correct dense Q5_0 single-row CUDA decode on Pascal. Preserve prefill, grouped paths, and sm70+ defaults.
Issue: ISSUE-LOCAL-01M63YETKYW03J7A5RESW13C6D.

## Upstream anchors

The primary vLLM pin cannot run this Pascal GGUF workload. The registered llama-cpp secondary pin is `62318cd57dc63900ecb153272c0b10f78550855d`.
Its `ggml/src/ggml-cuda/quantize.cu:64-101` stores half scales and original activation sums.
Its `ggml/src/ggml-cuda/vecdotq.cuh:175-203` subtracts the original sum for Q5_0.
The parent holds matching executed GPU traces and actual decode tensor evidence.

## Design

Add a Pascal hardware helper for Q8_1 activation quantization and Q5_0 dot products.
Use warp reductions, division before roundf, and half conversion as the oracle does.
Route dense Q5_0 decode through this helper only when the actual device is Pascal.
Keep output width and the public MatmulBTQuant entry point unchanged.

## Risks

Float reduction order can differ from the oracle. Q6_K and Q8_0 numerical differences remain separate hypotheses.

## Tests and gates

First add an independent original-sum regression through MatmulBTQuant and capture its failure.
Require focused green, prefill preservation, and a production call-site removal failure.
Compile legacy architectures. The operator reruns all SmolLM token axes against the same GPU oracle.

## Evidence

Actual decode layer 0 Q5_0 output error falls from RMSE 0.00498 to 6.4e-8 with the oracle activation formula.

## Stop conditions

Stop for a required public ABI change or evidence that the intended device scope is incorrect.

## Owed

Full model correctness and performance acceptance remain with the operator.

## Candidate evidence

On 29 October 2026, the original-sum regression failed at -22.0459 against -22.2344 before implementation.
The candidate and the restored production route return -22.2344.
Replacing the quant translation unit with its base version reproduces the same failure.
The focused suite passes 19 cases and 254599 assertions with the large GLM MoE geometry excluded.
The full suite fails that geometry because cudaMalloc exceeds the P620 2 GiB memory.
The subsequent original-sum case then observes the CUDA allocation error.
The changed translation unit compiles for sm60, sm61, and sm70.
The operator actual SmolLM gate still differs at generated token 8, so model correctness remains FAILING.
These results establish an operation correction, not model parity or performance acceptance.
Logs live under `/tmp/pascal-q5` for this session.
