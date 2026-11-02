# Real-checkpoint gate KV pool

The original public ABI gate failed before inference on SmolLM2.
The exact operator failure appears in `red.txt`.

The harness sets `max_model_len = 256` and `block_size = 16`.
Sixteen blocks contain 256 token slots, but one block is reserved.
The usable capacity is 240 tokens.

`src/vllm/entrypoints/model_loader.cpp:2033` reserves the null block in the fit check.
`src/vllm/v1/core/block_pool.cpp:453` excludes that block from free capacity.

The harness now allocates 32 blocks.
The context limit remains 256 tokens.
No model, scheduler, or KV planner behavior changes.

The operator owns the GPU and reruns the real workloads after integration.
Runtime verification remains PENDING until that run.
