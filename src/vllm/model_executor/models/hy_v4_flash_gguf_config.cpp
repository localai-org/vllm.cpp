// vllm.cpp ORIGINAL — the `hy_v4_flash` GGUF config builder TU. See the header
// and docs/bench-evidence/new-candidates-parsers-20261119/hy4-kvs.txt (the
// committed 76-KV walk of the pinned artifact's first shard, which is the
// key-name record this file is written against).
#include "vllm/model_executor/models/hy_v4_flash_gguf_config.h"

#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "vllm/model_executor/models/gguf_config_kv.h"

namespace vllm {
using gguf_config::CarryUnmapped;
using gguf_config::KvToJson;
using gguf_config::OptInt;
using gguf_config::ReqFloat;
using gguf_config::ReqInt;

namespace {

// One per-layer array whose length must equal `want` (element-type agnostic).
void ReqArrayLen(const GgufFile& g, const std::string& key, int64_t want) {
  const GgufValue* v = g.FindKv(key);
  if (v == nullptr || v->TypeId() != kGgufArray) {
    throw std::runtime_error("hy_v4_flash gguf: missing per-layer array " +
                             key);
  }
  const int64_t n =
      static_cast<int64_t>(std::get<GgufArray>(v->v).elems.size());
  if (n != want) {
    throw std::runtime_error("hy_v4_flash gguf: " + key + " length " +
                             std::to_string(n) + " != " +
                             std::to_string(want));
  }
}

// One per-layer u32 array whose length must equal block_count.
std::vector<int64_t> ReqPerLayerArray(const GgufFile& g, const std::string& key,
                                      int64_t block_count) {
  const GgufValue* v = g.FindKv(key);
  if (v == nullptr || v->TypeId() != kGgufArray) {
    throw std::runtime_error("hy_v4_flash gguf: missing per-layer array " +
                             key);
  }
  const GgufArray& a = std::get<GgufArray>(v->v);
  if (static_cast<int64_t>(a.elems.size()) != block_count) {
    throw std::runtime_error(
        "hy_v4_flash gguf: " + key + " length " +
        std::to_string(a.elems.size()) + " != block_count " +
        std::to_string(block_count));
  }
  std::vector<int64_t> out;
  out.reserve(a.elems.size());
  for (const GgufValue& e : a.elems) out.push_back(gguf_config::KvInt(e, key));
  return out;
}

}  // namespace

bool IsHyV4FlashGguf(const GgufFile& gguf) {
  const GgufValue* v = gguf.FindKv("general.architecture");
  return v != nullptr && v->TypeId() == kGgufString &&
         std::get<std::string>(v->v) == kHyV4FlashGgufArch;
}

std::string HyV4FlashForwardRefusal() {
  return std::string(
      "hy_v4_flash: the GGUF config parsed (HyV4FlashHfConfigFromGguf); the "
      "FORWARD is not implemented in this build and is owed to row "
      "NEW-CANDIDATE-PORTS (Engram n-gram tables, compressed indexer, DFLY "
      "MTP, vision tower — all port-row scope). Refused at the forward "
      "boundary — nothing ran.");
}

HfConfig HyV4FlashHfConfigFromGguf(const GgufFile& gguf) {
  if (!IsHyV4FlashGguf(gguf)) {
    throw std::runtime_error(
        "hy_v4_flash gguf: general.architecture must be '" +
        std::string(kHyV4FlashGgufArch) + "'");
  }
  const std::string p = "hy_v4_flash.";

  HfConfig c;
  c.model_type = "hy_v4_flash";
  // SYNTHESIZED class name — no upstream registry carries a `hy_v4_flash` arm;
  // the real spelling is the port row's decision (UNKNOWN-pending-port).
  c.architectures = {"HyV4FlashForCausalLM"};

  // --- standard fields, by name -------------------------------------------
  c.num_hidden_layers = ReqInt(gguf, p + "block_count");
  c.hidden_size = ReqInt(gguf, p + "embedding_length");
  c.intermediate_size = ReqInt(gguf, p + "feed_forward_length");
  c.num_key_value_heads = ReqInt(gguf, p + "attention.head_count_kv");
  c.max_position_embeddings = OptInt(gguf, p + "context_length", 0);
  c.sliding_window = OptInt(gguf, p + "attention.sliding_window", 0);
  c.num_experts = ReqInt(gguf, p + "expert_count");
  c.num_experts_per_tok = ReqInt(gguf, p + "expert_used_count");
  // `attention.head_count` is a PER-LAYER ARRAY on this family (the walk,
  // entry 39), not a scalar. HfConfig carries one scalar, so the FIRST entry
  // populates it and the full array is carried verbatim below; a consumer
  // that needs per-layer heads must read the carried array, not the scalar.
  const std::vector<int64_t> head_count =
      ReqPerLayerArray(gguf, p + "attention.head_count", c.num_hidden_layers);
  c.num_attention_heads = head_count.empty() ? 0 : head_count.front();
  c.head_dim = ReqInt(gguf, p + "attention.key_length");
  // vocab_size is BOTH an explicit KV (entry 19) and the embedding tensor's
  // first dim; the two must agree.
  c.vocab_size = ReqInt(gguf, p + "vocab_size");
  {
    const GgufTensorInfo& t = gguf.Get("token_embd.weight");
    if (t.shape.size() != 2 || t.shape[0] != c.vocab_size ||
        t.shape[1] != c.hidden_size) {
      throw std::runtime_error(
          "hy_v4_flash gguf: token_embd.weight shape disagrees with "
          "vocab_size/embedding_length");
    }
  }

  // --- geometry consistency -----------------------------------------------
  // Per-layer arrays. `attention.indexer.types` covers block_count MINUS the
  // DFLY next-n (MTP) layers on the pinned artifact (42 entries against
  // block_count 43, nextn_predict_layers 1) — accepted as either, recorded.
  const int64_t nextn = OptInt(gguf, p + "nextn_predict_layers", 0);
  {
    const GgufValue* v = gguf.FindKv(p + "attention.indexer.types");
    if (v == nullptr || v->TypeId() != kGgufArray) {
      throw std::runtime_error("hy_v4_flash gguf: missing per-layer array " +
                               p + "attention.indexer.types");
    }
    const int64_t n = static_cast<int64_t>(
        std::get<GgufArray>(v->v).elems.size());
    if (n != c.num_hidden_layers && n != c.num_hidden_layers - nextn) {
      throw std::runtime_error(
          "hy_v4_flash gguf: attention.indexer.types length " +
          std::to_string(n) + " matches neither block_count " +
          std::to_string(c.num_hidden_layers) + " nor block_count - nextn " +
          std::to_string(c.num_hidden_layers - nextn));
    }
    c.raw["hy_v4_flash_indexer_types_entries"] = n;
  }
  ReqPerLayerArray(gguf, p + "attention.sliding_window_pattern",
                   c.num_hidden_layers);
  ReqArrayLen(gguf, p + "swiglu_clamp_exp", c.num_hidden_layers);
  (void)ReqFloat(gguf, p + "rope.freq_base");  // presence, carried below

  // --- UNMAPPED KVs carried verbatim ---------------------------------------
  // Indexer, Engram, PLE, hyper-connection, nextn (DFLY MTP), tokenizer and
  // the per-layer arrays all land here with their GGUF spellings — the port
  // row inherits exactly what the file declares. The mmproj (vision) is a
  // SEPARATE file the loader opens; its KVs are not carried here.
  CarryUnmapped(
      gguf,
      {"general.architecture",
       "hy_v4_flash.block_count",
       "hy_v4_flash.embedding_length",
       "hy_v4_flash.feed_forward_length",
       "hy_v4_flash.attention.head_count",
       "hy_v4_flash.attention.head_count_kv",
       "hy_v4_flash.attention.key_length",
       "hy_v4_flash.context_length",
       "hy_v4_flash.attention.sliding_window",
       "hy_v4_flash.expert_count",
       "hy_v4_flash.expert_used_count",
       "hy_v4_flash.vocab_size",
       "hy_v4_flash.rope.freq_base",
       "hy_v4_flash.attention.indexer.types",
       "hy_v4_flash.attention.sliding_window_pattern",
       "hy_v4_flash.swiglu_clamp_exp"},
      c);
  // The per-layer head_count ARRAY is carried explicitly (it is in the mapped
  // set, so CarryUnmapped skips it): the scalar num_attention_heads above took
  // its FIRST entry only, and a consumer that needs per-layer heads reads this.
  nlohmann::json heads = nlohmann::json::array();
  for (int64_t h : head_count) heads.push_back(h);
  c.raw["hy_v4_flash_attention_head_count"] = heads;
  return c;
}

}  // namespace vllm
