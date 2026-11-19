// vllm.cpp ORIGINAL — the `olala` GGUF config builder TU. See the header and
// docs/bench-evidence/new-candidates-parsers-20261119/olala-kvs.txt (the
// committed 52-KV walk of the pinned artifact, which is the key-name record
// this file is written against).
#include "vllm/model_executor/models/olala_gguf_config.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/gguf_config_kv.h"

namespace vllm {
using gguf_config::CarryUnmapped;
using gguf_config::HasTensor;
using gguf_config::KvToJson;
using gguf_config::OptInt;
using gguf_config::ReqFloat;
using gguf_config::ReqInt;

bool IsOlalaGguf(const GgufFile& gguf) {
  const GgufValue* v = gguf.FindKv("general.architecture");
  return v != nullptr && v->TypeId() == kGgufString &&
         std::get<std::string>(v->v) == kOlalaGgufArch;
}

std::string OlalaForwardRefusal() {
  return std::string("olala: the GGUF config parsed (OlalaHfConfigFromGguf); "
                     "the FORWARD is not implemented in this build and is "
                     "owed to row NEW-CANDIDATE-PORTS (mamba-hybrid MoE arch "
                     "port). Refused at the forward boundary — nothing ran.");
}

HfConfig OlalaHfConfigFromGguf(const GgufFile& gguf) {
  if (!IsOlalaGguf(gguf)) {
    throw std::runtime_error(
        "olala gguf: general.architecture must be '" +
        std::string(kOlalaGgufArch) + "'");
  }
  const std::string p = "olala.";

  HfConfig c;
  c.model_type = "olala";
  // SYNTHESIZED class name — no upstream registry (vLLM or llama.cpp) carries
  // an `olala` arm, so the real class spelling is the port row's decision.
  // Recorded as UNKNOWN-pending-port rather than guessed further.
  c.architectures = {"OlalaForCausalLM"};

  // --- standard fields, by name (the GGUF keys the walk records) ----------
  c.num_hidden_layers = ReqInt(gguf, p + "block_count");
  c.hidden_size = ReqInt(gguf, p + "embedding_length");
  c.num_attention_heads = ReqInt(gguf, p + "attention.head_count");
  c.num_key_value_heads = ReqInt(gguf, p + "attention.head_count_kv");
  c.head_dim = ReqInt(gguf, p + "attention.key_length");
  c.max_position_embeddings = OptInt(gguf, p + "context_length", 0);
  c.sliding_window = OptInt(gguf, p + "attention.sliding_window", 0);
  c.num_experts = ReqInt(gguf, p + "expert_count");
  c.num_experts_per_tok = ReqInt(gguf, p + "expert_used_count");
  {
    const GgufTensorInfo& t = gguf.Get("token_embd.weight");
    if (t.shape.size() != 2) {
      throw std::runtime_error("olala gguf: token_embd.weight is not 2-D");
    }
    c.vocab_size = t.shape[0];
    // Geometry: the embedding table must be [vocab, hidden].
    if (t.shape[1] != c.hidden_size) {
      throw std::runtime_error(
          "olala gguf: token_embd.shape[1] " + std::to_string(t.shape[1]) +
          " != embedding_length " + std::to_string(c.hidden_size));
    }
  }
  // NOT set: layer_types. `olala.layer_types` is a compact 'M'/'V' string;
  // its expansion into "linear_attention"/"full_attention" per layer is
  // port-row semantics this row does not guess. Carried verbatim below.
  // NOT set: intermediate_size. `olala.feed_forward_length` is 0 on the
  // pinned artifact (every FFN is routed or shared-expert), and a 0 would
  // read downstream as a real value; the expert sizes are carried instead.
  // The heads*head_dim != hidden product is EXPECTED here (tpa_rank 4
  // decomposition); recorded, not fatal, and the walk shows the same numbers.

  // --- geometry consistency -----------------------------------------------
  if (const GgufValue* v = gguf.FindKv(p + "layer_types");
      v != nullptr && v->TypeId() == kGgufString) {
    const std::string& s = std::get<std::string>(v->v);
    if (static_cast<int64_t>(s.size()) != c.num_hidden_layers) {
      throw std::runtime_error(
          "olala gguf: layer_types length " + std::to_string(s.size()) +
          " != block_count " + std::to_string(c.num_hidden_layers));
    }
  }
  // Every decoder block the file declares must have a tensor table entry.
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    // The walk shows this family has NO attn_norm per block; its per-block
    // tensors are the geo_* family. Presence probe on one of those.
    const std::string name = "blk." + std::to_string(l) + ".geo_mixer_bias";
    if (!HasTensor(gguf, name)) {
      throw std::runtime_error("olala gguf: declared block " +
                               std::to_string(l) + " has no " + name);
    }
  }

  // --- UNMAPPED KVs carried verbatim ---------------------------------------
  // The mapped set is EXACTLY the keys read above; everything else (the ssm.*
  // family, mamba_*, tpa_rank, tokenizer.*, general.*, ...) lands here with
  // its GGUF spelling. The enumeration records the same set on disk.
  CarryUnmapped(
      gguf,
      {"general.architecture",      "olala.block_count",
       "olala.embedding_length",    "olala.attention.head_count",
       "olala.attention.head_count_kv", "olala.attention.key_length",
       "olala.context_length",      "olala.attention.sliding_window",
       "olala.expert_count",        "olala.expert_used_count"},
      c);
  // `olala.layer_types` is READ above for the length validation but left in
  // the carry: its 'M'/'V' expansion is port-row semantics, and the verbatim
  // string is what the port row should inherit.
  return c;
}

}  // namespace vllm
