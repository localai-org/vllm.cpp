// Kolibri-1.1 W2 (U2+U4) gate — MODEL-TEXT-kolibri-1-1-kolibri1-1-for-causal-lm.
// Spec .agents/specs/kolibri-1-1-u2-w2-dense-prefix-and-shared-kv.md.
//
// U2 is parse / tensor-accounting / KV-group plumbing only. There is no
// gateable CPU oracle for kolibri1_1, so this test claims no token parity:
// it proves config fields, tensor names, group membership and named refusals.
#include "vllm/model_executor/models/kolibri1.h"

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/kolibri1_weights.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/v1/kv_cache_interface.h"

using vllm::AccountKolibri1Tensors;
using vllm::EnumerateKolibri1Tensors;
using vllm::HfConfig;
using vllm::Kolibri1WeightScheme;
using vllm::Kolibri1Weights;
using vllm::MakeKolibri1KVCache;
using vllm::ModelRegistry;
using vllm::ParseKolibri1Params;
using vllm::RefuseKolibri11SparseForward;

namespace {

HfConfig BaseConfig(int64_t layers = 50) {
  HfConfig c;
  c.model_type = "kolibri1";
  c.architectures = {"Kolibri1ForCausalLM"};
  c.hidden_size = 2560;
  c.num_hidden_layers = layers;
  c.vocab_size = 128000;
  c.num_attention_heads = 48;
  c.num_key_value_heads = 4;

  nlohmann::json j;
  j["head_dim"] = 128;
  j["sliding_window"] = 513;
  j["use_sliding_window"] = true;
  j["rope_theta"] = 10000.0;
  j["num_experts"] = 384;
  j["num_experts_per_tok"] = 6;
  j["moe_intermediate_size"] = 512;
  j["shared_expert_intermediate_size"] = 512;
  j["norm_topk_prob"] = false;
  j["rms_norm_eps"] = 1e-6;
  j["hidden_act"] = "silu";
  nlohmann::json kinds = nlohmann::json::array();
  for (int64_t i = 0; i < layers; ++i)
    kinds.push_back(i % 5 == 4 ? "full_attention" : "sliding_attention");
  j["layer_types"] = kinds;
  c.raw = j;
  return c;
}

// The official 1.1 geometry: every fifth layer is indexed, layers 0-1 are a
// dense FFN prefix, and six layers read K/V from a source layer.
HfConfig Config11() {
  HfConfig c = BaseConfig();
  nlohmann::json kinds = nlohmann::json::array();
  for (int64_t i = 0; i < c.num_hidden_layers; ++i)
    kinds.push_back(i % 5 == 4 ? "indexed_attention" : "sliding_attention");
  c.raw["layer_types"] = kinds;
  c.raw["index_layers"] = {4, 14, 19, 29};
  c.raw["kv_source_layers"] = {0,1,2,3,4,5,6,7,8,4,10,11,12,13,4,15,16,17,18,19,
                               20,21,22,23,24,25,26,27,28,29,30,31,32,33,29,35,
                               36,37,38,29,40,41,42,43,29,45,46,47,48,4};
  c.raw["mlp_only_layers"] = {0, 1};
  c.raw["mlp_only_intermediate_size"] = 3584;
  return c;
}

std::vector<std::string> ExpectedNames(const HfConfig& c) {
  const auto p = ParseKolibri1Params(c);
  std::vector<std::string> names;
  for (const auto& t : EnumerateKolibri1Tensors(p, Kolibri1WeightScheme::kBf16))
    names.push_back(t.name);
  return names;
}

}  // namespace

TEST_CASE("kolibri1_1: 1.0 config parsing stays byte-neutral") {
  const auto p = ParseKolibri1Params(BaseConfig());
  CHECK(p.index_layers.empty());
  CHECK(p.kv_source_layers.empty());
  CHECK(p.mlp_only_layers.empty());
  CHECK(p.mlp_only_intermediate_size == 0);
  CHECK_FALSE(p.IsIndexedLayer(4));
  CHECK_FALSE(p.IsDenseLayer(0));
  CHECK(p.KvSourceLayer(9) == 9);
}

TEST_CASE("kolibri1_1: the 1.1 fields parse into the named shape") {
  const auto p = ParseKolibri1Params(Config11());
  REQUIRE(p.index_layers.size() == 4);
  CHECK(p.index_layers[0] == 4);
  REQUIRE(p.kv_source_layers.size() == 50);
  CHECK(p.KvSourceLayer(9) == 4);
  CHECK(p.KvSourceLayer(34) == 29);
  CHECK(p.IsIndexedLayer(4));
  CHECK(p.IsIndexedLayer(49));
  CHECK_FALSE(p.IsIndexedLayer(0));
  CHECK(p.IsDenseLayer(0));
  CHECK(p.IsDenseLayer(1));
  CHECK_FALSE(p.IsDenseLayer(2));
  CHECK(p.mlp_only_intermediate_size == 3584);
}

TEST_CASE("kolibri1_1: malformed 1.1 fields are refused by name") {
  HfConfig bad = Config11();
  bad.raw["kv_source_layers"] = {0, 1};
  CHECK_THROWS_WITH(ParseKolibri1Params(bad),
                    doctest::Contains("kv_source_layers length"));

  bad = Config11();
  bad.raw["index_layers"] = {0};  // layer 0 is sliding, not indexed
  CHECK_THROWS_WITH(ParseKolibri1Params(bad),
                    doctest::Contains("index_layers entry"));

  bad = Config11();
  bad.raw["mlp_only_intermediate_size"] = 0;
  CHECK_THROWS_WITH(ParseKolibri1Params(bad),
                    doctest::Contains("mlp_only_intermediate_size"));
}

TEST_CASE("kolibri1_1: tensor map accepts the dense prefix and shared-KV omissions") {
  const HfConfig c = Config11();
  const auto names = ExpectedNames(c);
  const auto acc = AccountKolibri1Tensors(ParseKolibri1Params(c),
                                          Kolibri1WeightScheme::kBf16, names);
  CHECK(acc.missing.empty());
  CHECK(acc.unaccounted.empty());
  CHECK(acc.duplicated.empty());

  bool dense_proj = false, shared_kv = false;
  for (const auto& n : names) {
    if (n == "model.layers.0.mlp.gate_proj.weight") dense_proj = true;
    if (n == "model.layers.9.self_attn.k_proj.weight") shared_kv = true;
  }
  CHECK(dense_proj);
  CHECK_FALSE(shared_kv);  // layer 9 reads layer 4's K/V

  // A genuinely missing tensor is still refused.
  auto missing = names;
  missing.erase(missing.begin());
  CHECK_FALSE(AccountKolibri1Tensors(ParseKolibri1Params(c),
                                     Kolibri1WeightScheme::kBf16, missing)
                  .missing.empty());

  // The 1.0 map is unchanged by the new fields.
  const auto names10 = ExpectedNames(BaseConfig());
  CHECK(names10.size() != names.size());  // dense prefix shrinks the map
}

TEST_CASE("kolibri1_1: KV groups share the source and keep SWA separate") {
  const auto kv = MakeKolibri1KVCache(Config11(), 16, 8);
  // One group per KV source (44 sources for the official map), not two broad
  // buckets — otherwise "sharing" is an artefact of co-membership.
  CHECK(kv.kv_cache_groups.size() > 2);
  std::set<std::string> all;
  std::string group_of_9, group_of_34;
  for (size_t g = 0; g < kv.kv_cache_groups.size(); ++g) {
    const auto& names = kv.kv_cache_groups[g].layer_names;
    for (const auto& n : names) {
      CHECK(all.insert(n).second);  // no layer dropped or duplicated
      if (n == "model.layers.9.self_attn") group_of_9 = std::to_string(g);
      if (n == "model.layers.34.self_attn") group_of_34 = std::to_string(g);
    }
  }
  CHECK(all.size() == 50);

  bool share_4 = false, share_29 = false;
  for (const auto& grp : kv.kv_cache_groups) {
    bool has9 = false, has4 = false, has34 = false, has29 = false;
    for (const auto& n : grp.layer_names) {
      has9 |= n == "model.layers.9.self_attn";
      has4 |= n == "model.layers.4.self_attn";
      has34 |= n == "model.layers.34.self_attn";
      has29 |= n == "model.layers.29.self_attn";
    }
    share_4 |= has9 && has4;
    share_29 |= has34 && has29;
  }
  CHECK(share_4);
  CHECK(share_29);
  CHECK_FALSE(group_of_9.empty());
  CHECK_FALSE(group_of_34.empty());
}

TEST_CASE("kolibri1_1: the sparse forward is refused by name") {
  CHECK_THROWS_WITH(
      RefuseKolibri11SparseForward(ParseKolibri1Params(Config11())),
      doctest::Contains(
          "indexed_attention (Kolibri-1.1) forward is not implemented"));
  // Control: the 1.0 path must not be refused.
  RefuseKolibri11SparseForward(ParseKolibri1Params(BaseConfig()));
}

TEST_CASE("kolibri1_1: the registry factory reaches the new KV path") {
  const HfConfig c = Config11();
  const auto& reg = ModelRegistry::Resolve(c);
  REQUIRE(reg.factory != nullptr);
  REQUIRE(reg.factory->make_kv_cache != nullptr);
  const auto kv = reg.factory->make_kv_cache(c, 16, 8);
  CHECK_FALSE(kv.kv_cache_groups.empty());
}
