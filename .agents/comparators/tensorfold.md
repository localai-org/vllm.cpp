# TensorFold Qwen3.8 Flash Next comparator

TensorFold is pinned here only as an external implementation and performance
comparator. It may ground source inspection, mechanism attribution, and measured
throughput. It does not supply correct output and does not replace vLLM,
Transformers, llama.cpp, or the existing component fixtures in a correctness
gate.

The deployment recipe is a separate pinned object because it carries the image
recipe, runtime flags, and eight patches used for the published DGX Spark setup.
Neither repository alone reconstructs that setup.

```comparator-pin
id = tensorfold
role = implementation-performance
upstream = https://github.com/ashhart/TensorFold
pin = 191188075bca56a7c71074a79375eb4c1cb22e1c
recipe_upstream = https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
recipe_pin = 856bb6be4b58ce6a6727e6d071fb1c52f3f80e6e
scope = Qwen3.8 Flash Next implementation organization and performance on the pinned MiaAI DGX Spark recipe
correctness = not-an-oracle
pinned_on = 2026-09-29
```
