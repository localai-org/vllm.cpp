// Llama GGUF source adapter into the shared dense model; no alternate forward.
#pragma once
#include "vllm/model_executor/models/llama.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
namespace vllm {
HfConfig LlamaHfConfigFromGguf(const GgufFile& gguf);
LlamaWeights LoadLlamaFromGguf(const GgufFile& gguf, const HfConfig& config,
                              const GgufLoadPolicy* policy = nullptr);
}
