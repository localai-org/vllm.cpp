// vllm.cpp ORIGINAL — the `inkling2_loop` GGUF config builder TU. See the header
// for the artifact and oracle anchors. The KV enumeration is the committed walk
// docs/bench-evidence/new-candidate-inkling2-20261121/apex-balanced-walk.txt.
#include "vllm/model_executor/models/inkling2_loop_gguf_config.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/gguf_config_kv.h"

namespace vllm {
using gguf_config::CarryUnmapped;
using gguf_config::KvInt;
using gguf_config::OptInt;
using gguf_config::ReqFloat;
using gguf_config::ReqInt;

namespace {

// One per-layer array whose length must equal block_count (type agnostic).
void ReqArrayLen(const GgufFile& g, const std::string& key, int64_t want) {
  const GgufValue* v = g.FindKv(key);
  if (v == nullptr || v->TypeId() != kGgufArray) {
    throw std::runtime_error("inkling2_loop gguf: missing per-layer array " +
                             key);
  }
  const int64_t n =
      static_cast<int64_t>(std::get<GgufArray>(v->v).elems.size());
  if (n != want) {
    throw std::runtime_error("inkling2_loop gguf: " + key + " length " +
                             std::to_string(n) + " != " +
                             std::to_string(want));
  }
}

// One per-layer integer array; the first entry feeds HfConfig's scalar and the
// whole vector is carried verbatim (hy_v4_flash precedent).
std::vector<int64_t> ReqPerLayerArray(const GgufFile& g, const std::string& key,
                                      int64_t block_count) {
  const GgufValue* v = g.FindKv(key);
  if (v == nullptr || v->TypeId() != kGgufArray) {
    throw std::runtime_error("inkling2_loop gguf: missing per-layer array " +
                             key);
  }
  const GgufArray& a = std::get<GgufArray>(v->v);
  if (static_cast<int64_t>(a.elems.size()) != block_count) {
    throw std::runtime_error("inkling2_loop gguf: " + key + " length " +
                             std::to_string(a.elems.size()) +
                             " != block_count " + std::to_string(block_count));
  }
  std::vector<int64_t> out;
  out.reserve(a.elems.size());
  for (const GgufValue& e : a.elems) out.push_back(KvInt(e, key));
  return out;
}

bool ReqBool(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  if (v == nullptr || v->TypeId() != kGgufBool) {
    throw std::runtime_error("inkling2_loop gguf: missing bool metadata key " +
                             key);
  }
  return std::get<bool>(v->v);
}

}  // namespace

bool IsInkling2LoopGguf(const GgufFile& gguf) {
  const GgufValue* v = gguf.FindKv("general.architecture");
  return v != nullptr && v->TypeId() == kGgufString &&
         std::get<std::string>(v->v) == kInkling2LoopGgufArch;
}

std::string Inkling2LoopForwardRefusal() {
  return std::string(
      "inkling2_loop: the GGUF config parsed (Inkling2LoopHfConfigFromGguf); "
      "the FORWARD is not implemented in this build and is owed to row "
      "NEW-CANDIDATE-PORTS (looped core layers 6-17 x 3 passes with per-pass "
      "KV/conv caches and gated message controllers — all port-row scope). "
      "Refused at the forward boundary — nothing ran.");
}

HfConfig Inkling2LoopHfConfigFromGguf(const GgufFile& gguf) {
  if (!IsInkling2LoopGguf(gguf)) {
    throw std::runtime_error(
        "inkling2_loop gguf: general.architecture must be '" +
        std::string(kInkling2LoopGgufArch) + "'");
  }
  const std::string p = "inkling2_loop.";

  HfConfig c;
  c.model_type = "inkling2_loop";
  // From the oracle: vllm/models/inkling2/configs.py Inkling2LoopMMConfig
  // (model_type inkling2_loop, architecture Inkling2LoopForConditionalGeneration)
  // and vllm/model_executor/models/registry.py's registration. Not guessed.
  c.architectures = {"Inkling2LoopForConditionalGeneration"};

  // --- standard fields, by name -------------------------------------------
  c.num_hidden_layers = ReqInt(gguf, p + "block_count");
  c.hidden_size = ReqInt(gguf, p + "embedding_length");
  c.intermediate_size = ReqInt(gguf, p + "feed_forward_length");
  const std::vector<int64_t> head_count =
      ReqPerLayerArray(gguf, p + "attention.head_count", c.num_hidden_layers);
  c.num_attention_heads = head_count.empty() ? 0 : head_count.front();
  const std::vector<int64_t> kv_count = ReqPerLayerArray(
      gguf, p + "attention.head_count_kv", c.num_hidden_layers);
  c.num_key_value_heads = kv_count.empty() ? 0 : kv_count.front();
  c.head_dim = ReqInt(gguf, p + "attention.key_length");
  c.num_experts = ReqInt(gguf, p + "expert_count");
  c.num_experts_per_tok = ReqInt(gguf, p + "expert_used_count");
  c.moe_intermediate_size = ReqInt(gguf, p + "expert_feed_forward_length");
  c.max_position_embeddings = OptInt(gguf, p + "context_length", 0);
  c.sliding_window = OptInt(gguf, p + "attention.sliding_window", 0);
  c.rms_norm_eps = ReqFloat(gguf, p + "attention.layer_norm_rms_epsilon");
  c.vocab_size = ReqInt(gguf, p + "vocab_size");
  {
    const GgufTensorInfo& t = gguf.Get("token_embd.weight");
    if (t.shape.size() != 2 || t.shape[0] != c.vocab_size ||
        t.shape[1] != c.hidden_size) {
      throw std::runtime_error(
          "inkling2_loop gguf: token_embd.weight shape disagrees with "
          "vocab_size/embedding_length");
    }
  }

  // --- geometry consistency -----------------------------------------------
  // The bool pattern is per-layer layer-type semantics: validate its length,
  // but leave it in the verbatim carry (olala.layer_types precedent).
  ReqArrayLen(gguf, p + "attention.sliding_window_pattern",
              c.num_hidden_layers);

  // --- loop_config: upstream field names, GGUF values ----------------------
  // vllm/models/inkling2/configs.py:108-138 requires the four int fields and
  // one bool, bounds the core range to the block count, gate_rank >= 1, and
  // at least two passes for a fixed (non-varloop) checkpoint.
  const int64_t core_start = ReqInt(gguf, p + "loop.start");
  const int64_t core_end = ReqInt(gguf, p + "loop.end");
  const int64_t num_passes = ReqInt(gguf, p + "loop.count");
  const int64_t gate_rank = ReqInt(gguf, p + "loop.gate_rank");
  const bool is_varloop = ReqBool(gguf, p + "loop.varying");
  if (!(0 < core_start && core_start < core_end &&
        core_end <= c.num_hidden_layers)) {
    throw std::runtime_error(
        "inkling2_loop gguf: loop core layers are out of range");
  }
  if (gate_rank < 1) {
    throw std::runtime_error("inkling2_loop gguf: loop gate_rank must be >= 1");
  }
  if (!is_varloop && num_passes < 2) {
    throw std::runtime_error(
        "inkling2_loop gguf: a fixed-loop checkpoint needs at least 2 passes");
  }
  nlohmann::json loop = nlohmann::json::object();
  loop["core_start"] = core_start;
  loop["core_end"] = core_end;
  loop["num_passes"] = num_passes;
  loop["gate_rank"] = gate_rank;
  loop["is_varloop"] = is_varloop;
  c.raw["loop_config"] = loop;
  // RECORDED DEVIATION: upstream also requires attn_res_normalize_values=True
  // for a loop config (configs.py:123-126), but this GGUF container carries no
  // such KV. The forward port must obtain it from the checkpoint, not invent it.

  // --- UNMAPPED KVs carried verbatim ---------------------------------------
  CarryUnmapped(
      gguf,
      {"general.architecture",
       "inkling2_loop.block_count",
       "inkling2_loop.embedding_length",
       "inkling2_loop.feed_forward_length",
       "inkling2_loop.attention.head_count",
       "inkling2_loop.attention.head_count_kv",
       "inkling2_loop.attention.key_length",
       "inkling2_loop.expert_count",
       "inkling2_loop.expert_used_count",
       "inkling2_loop.expert_feed_forward_length",
       "inkling2_loop.context_length",
       "inkling2_loop.attention.sliding_window",
       "inkling2_loop.attention.layer_norm_rms_epsilon",
       "inkling2_loop.vocab_size",
       "inkling2_loop.loop.start",
       "inkling2_loop.loop.end",
       "inkling2_loop.loop.count",
       "inkling2_loop.loop.gate_rank",
       "inkling2_loop.loop.varying"},
      c);
  // head_count is in the mapped set, so CarryUnmapped skips it; carry the
  // per-layer vectors explicitly because HfConfig holds only their first entry.
  nlohmann::json heads = nlohmann::json::array();
  for (int64_t h : head_count) heads.push_back(h);
  c.raw["inkling2_loop_attention_head_count"] = heads;
  nlohmann::json kvs = nlohmann::json::array();
  for (int64_t h : kv_count) kvs.push_back(h);
  c.raw["inkling2_loop_attention_head_count_kv"] = kvs;
  return c;
}

}  // namespace vllm
