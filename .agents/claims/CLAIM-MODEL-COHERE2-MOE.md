# CLAIM-MODEL-COHERE2-MOE

| Claim | Row IDs | Agent | Worktree / remote dir | Branch | Owned scope | State | Last update |
|---|---|---|---|---|---|---|---|
| `CLAIM-MODEL-COHERE2-MOE` | `MODEL-TEXT-cohere2-moe-cohere2-moe-for-causal-lm` (`ACTIVE`) | Claude (claude-opus-5-5), helper for `ISSUE-LOCAL-01M3S23H9B25EFPFJN75Y1XBWY`; a fresh implementer and a fresh reviewer | local worktree `/home/mudler/_git/vllm.cpp-wt-cohere2-moe`; CPU only | `feat/cohere2-moe` (developer direction: local commits, no pull request) | The [spec](../specs/cohere2-moe.md): config, loader, forward and registration of `Cohere2MoeForCausalLM` with the per-layer window, the RoPE/NoPE split, the sigmoid router and the dense prefix; synthetic, real-tensor and reachability gates. Excludes GGUF, quantized siblings, the drafter, the e2e token gate and speed | `ACTIVE` | 2026-09-30 - implementation committed on `feat/cohere2-moe` (CPU gates green); fresh review, GPU token gate, GGUF and quantized arms owed |
