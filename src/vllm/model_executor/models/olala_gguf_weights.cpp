// vllm.cpp ORIGINAL — the `olala` GGUF weights arm (name-map + census), row
// NEW-CANDIDATE-PORTS W1. See the header and
// .agents/specs/new-candidate-ports-olala.md. The name table below is written
// against the committed walk of the pinned artifact
// (docs/bench-evidence/new-candidate-ports-olala-20261120/olala-tensor-walk-bf16.txt):
// 989 tensors, 36 blocks, V layers at 4,9,14,19,24,29,34 (attn_q presence),
// matching olala.layer_types (36 chars, V at the same indices).
#include "vllm/model_executor/models/olala_gguf_weights.h"

#include <set>
#include <stdexcept>
#include <utility>

#include "vllm/model_executor/models/olala_gguf_config.h"

namespace vllm {
namespace {

struct RoleSpec {
  const char* suffix;  // tensor name after "blk.N."
  OlalaTensorRole role;
  bool layer_tensor = true;
};

// The per-layer common set (all 36 blocks) plus the M-only ssm_* set and the
// V-only attn_* set, from the namemap census.
constexpr RoleSpec kLayerRoles[] = {
    {"attn_output.weight", OlalaTensorRole::kAttnOutput},
    {"exp_probs_b", OlalaTensorRole::kRouterBias},
    {"ffn_gate_inp.weight", OlalaTensorRole::kRouterWeight},
    {"ffn_up_exps.weight", OlalaTensorRole::kFfnUpExps},
    {"ffn_down_exps.weight", OlalaTensorRole::kFfnDownExps},
    {"ffn_up_shexp.weight", OlalaTensorRole::kFfnUpShexp},
    {"ffn_down_shexp.weight", OlalaTensorRole::kFfnDownShexp},
    {"ffn_latent_up.weight", OlalaTensorRole::kFfnLatentUp},
    {"ffn_latent_down.weight", OlalaTensorRole::kFfnLatentDown},
    {"geo_mixer_bias", OlalaTensorRole::kGeoMixerBias},
    {"geo_mixer_scale", OlalaTensorRole::kGeoMixerScale},
    {"geo_mlp_bias", OlalaTensorRole::kGeoMlpBias},
    {"geo_mlp_scale", OlalaTensorRole::kGeoMlpScale},
    // M layers (29):
    {"ssm_in.weight", OlalaTensorRole::kSsmIn},
    {"ssm_in_dyn.weight", OlalaTensorRole::kSsmInDyn},
    {"ssm_b_bias", OlalaTensorRole::kSsmBBias},
    {"ssm_b_norm.weight", OlalaTensorRole::kSsmBNorm},
    {"ssm_c_bias", OlalaTensorRole::kSsmCBias},
    {"ssm_c_norm.weight", OlalaTensorRole::kSsmCNorm},
    {"ssm_d", OlalaTensorRole::kSsmD},
    {"ssm_dt_bias", OlalaTensorRole::kSsmDtBias},
    {"ssm_m3_bias", OlalaTensorRole::kSsmM3Bias},
    {"ssm_m3_misc", OlalaTensorRole::kSsmM3Misc},
    {"ssm_m3_mxz", OlalaTensorRole::kSsmM3Mxz},
    {"ssm_m3_norms", OlalaTensorRole::kSsmM3Norms},
    {"ssm_mimo_x", OlalaTensorRole::kSsmMimoX},
    {"ssm_mimo_z", OlalaTensorRole::kSsmMimoZ},
    {"ssm_mimo_o", OlalaTensorRole::kSsmMimoO},
    // V layers (7):
    {"attn_q.weight", OlalaTensorRole::kAttnQ},
    {"attn_gate.weight", OlalaTensorRole::kAttnGate},
    {"attn_q_norm.weight", OlalaTensorRole::kAttnQNorm},
    {"attn_k_norm.weight", OlalaTensorRole::kAttnKNorm},
    {"attn_lambda.weight", OlalaTensorRole::kAttnLambda},
    {"attn_shift_k.weight", OlalaTensorRole::kAttnShiftK},
    {"attn_shift_v.weight", OlalaTensorRole::kAttnShiftV},
    {"attn_softmax_scaler", OlalaTensorRole::kAttnSoftmaxScaler},
    {"attn_wa_k.weight", OlalaTensorRole::kAttnWaK},
    {"attn_wa_v.weight", OlalaTensorRole::kAttnWaV},
    {"attn_wb_k.weight", OlalaTensorRole::kAttnWbK},
    {"attn_wb_v.weight", OlalaTensorRole::kAttnWbV},
};

constexpr RoleSpec kGlobalRoles[] = {
    {"token_embd.weight", OlalaTensorRole::kEmbed, false},
    {"output.weight", OlalaTensorRole::kLmHead, false},
};

OlalaTensorRole Classify(const std::string& name, int64_t* layer_out) {
  static const std::string kBlk = "blk.";
  if (name.compare(0, kBlk.size(), kBlk) == 0) {
    const size_t dot = name.find('.', kBlk.size());
    if (dot == std::string::npos) return OlalaTensorRole::kUnmapped;
    const std::string num = name.substr(kBlk.size(), dot - kBlk.size());
    if (num.empty() ||
        num.find_first_not_of("0123456789") != std::string::npos) {
      return OlalaTensorRole::kUnmapped;
    }
    *layer_out = std::stoll(num);
    const std::string rest = name.substr(dot + 1);
    for (const RoleSpec& spec : kLayerRoles) {
      if (rest == spec.suffix) return spec.role;
    }
    return OlalaTensorRole::kUnmapped;
  }
  for (const RoleSpec& spec : kGlobalRoles) {
    if (name == spec.suffix) return spec.role;
  }
  return OlalaTensorRole::kUnmapped;
}

std::string ShapeStr(const std::vector<int64_t>& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    if (i) out += ",";
    out += std::to_string(s[i]);
  }
  return out + "]";
}

}  // namespace

const char* OlalaRoleName(OlalaTensorRole role) {
  switch (role) {
    case OlalaTensorRole::kEmbed: return "embed";
    case OlalaTensorRole::kLmHead: return "lm_head";
    case OlalaTensorRole::kAttnOutput: return "attn_output";
    case OlalaTensorRole::kRouterBias: return "router_bias";
    case OlalaTensorRole::kRouterWeight: return "router_weight";
    case OlalaTensorRole::kFfnUpExps: return "ffn_up_exps";
    case OlalaTensorRole::kFfnDownExps: return "ffn_down_exps";
    case OlalaTensorRole::kFfnUpShexp: return "ffn_up_shexp";
    case OlalaTensorRole::kFfnDownShexp: return "ffn_down_shexp";
    case OlalaTensorRole::kFfnLatentUp: return "ffn_latent_up";
    case OlalaTensorRole::kFfnLatentDown: return "ffn_latent_down";
    case OlalaTensorRole::kGeoMixerBias: return "geo_mixer_bias";
    case OlalaTensorRole::kGeoMixerScale: return "geo_mixer_scale";
    case OlalaTensorRole::kGeoMlpBias: return "geo_mlp_bias";
    case OlalaTensorRole::kGeoMlpScale: return "geo_mlp_scale";
    case OlalaTensorRole::kSsmIn: return "ssm_in";
    case OlalaTensorRole::kSsmInDyn: return "ssm_in_dyn";
    case OlalaTensorRole::kSsmBBias: return "ssm_b_bias";
    case OlalaTensorRole::kSsmBNorm: return "ssm_b_norm";
    case OlalaTensorRole::kSsmCBias: return "ssm_c_bias";
    case OlalaTensorRole::kSsmCNorm: return "ssm_c_norm";
    case OlalaTensorRole::kSsmD: return "ssm_d";
    case OlalaTensorRole::kSsmDtBias: return "ssm_dt_bias";
    case OlalaTensorRole::kSsmM3Bias: return "ssm_m3_bias";
    case OlalaTensorRole::kSsmM3Misc: return "ssm_m3_misc";
    case OlalaTensorRole::kSsmM3Mxz: return "ssm_m3_mxz";
    case OlalaTensorRole::kSsmM3Norms: return "ssm_m3_norms";
    case OlalaTensorRole::kSsmMimoX: return "ssm_mimo_x";
    case OlalaTensorRole::kSsmMimoZ: return "ssm_mimo_z";
    case OlalaTensorRole::kSsmMimoO: return "ssm_mimo_o";
    case OlalaTensorRole::kAttnQ: return "attn_q";
    case OlalaTensorRole::kAttnGate: return "attn_gate";
    case OlalaTensorRole::kAttnQNorm: return "attn_q_norm";
    case OlalaTensorRole::kAttnKNorm: return "attn_k_norm";
    case OlalaTensorRole::kAttnLambda: return "attn_lambda";
    case OlalaTensorRole::kAttnShiftK: return "attn_shift_k";
    case OlalaTensorRole::kAttnShiftV: return "attn_shift_v";
    case OlalaTensorRole::kAttnSoftmaxScaler: return "attn_softmax_scaler";
    case OlalaTensorRole::kAttnWaK: return "attn_wa_k";
    case OlalaTensorRole::kAttnWaV: return "attn_wa_v";
    case OlalaTensorRole::kAttnWbK: return "attn_wb_k";
    case OlalaTensorRole::kAttnWbV: return "attn_wb_v";
    case OlalaTensorRole::kUnmapped: return "UNMAPPED";
  }
  return "?";
}

OlalaGgufCensus CensusOlalaGguf(const GgufFile& gguf) {
  OlalaGgufCensus c;
  std::map<int64_t, bool> layer_has_attn_q;
  for (const GgufTensorInfo& t : gguf.Tensors()) {
    OlalaCensusEntry e;
    e.name = t.name;
    e.ggml_type = t.ggml_type;
    e.shape = t.shape;
    e.role = Classify(t.name, &e.layer);
    if (e.role == OlalaTensorRole::kAttnQ) layer_has_attn_q[e.layer] = true;
    c.entries.push_back(std::move(e));
    c.role_counts[c.entries.back().role]++;
    if (c.entries.back().role == OlalaTensorRole::kUnmapped) {
      c.unmapped.push_back(t.name);
    }
  }
  for (const auto& kv : layer_has_attn_q) c.v_layers.push_back(kv.first);
  std::map<int64_t, bool> layer_seen;
  for (const OlalaCensusEntry& e : c.entries) {
    if (e.layer >= 0) layer_seen[e.layer] = true;
  }
  for (const auto& kv : layer_seen) {
    c.num_layers++;
    if (!layer_has_attn_q.count(kv.first)) c.m_layers.push_back(kv.first);
  }
  return c;
}

void ValidateOlalaGguf(const GgufFile& gguf, const HfConfig& cfg) {
  if (!IsOlalaGguf(gguf)) {
    throw std::runtime_error("olala gguf weights: general.architecture is not '" +
                             std::string(kOlalaGgufArch) + "'");
  }
  OlalaGgufCensus c = CensusOlalaGguf(gguf);
  std::string err;
  auto fail = [&](const std::string& m) {
    if (err.empty()) err = "olala gguf weights: " + m;
  };

  if (!c.unmapped.empty()) {
    std::string list;
    for (size_t i = 0; i < c.unmapped.size() && i < 8; ++i) {
      list += (i ? ", " : "") + c.unmapped[i];
    }
    fail(std::to_string(c.unmapped.size()) +
         " unmapped tensor(s): " + list +
         (c.unmapped.size() > 8 ? ", ..." : ""));
  }

  if (c.num_layers != cfg.num_hidden_layers) {
    fail("layer count " + std::to_string(c.num_layers) + " != block_count " +
         std::to_string(cfg.num_hidden_layers));
  }
  const int64_t n_attn = 12;     // V-only roles (attn_output is common)
  if (c.role_counts[OlalaTensorRole::kAttnOutput] != c.num_layers) {
    fail("attn_output count " +
         std::to_string(c.role_counts[OlalaTensorRole::kAttnOutput]) +
         " != layer count " + std::to_string(c.num_layers));
  }
  for (OlalaTensorRole r : {OlalaTensorRole::kRouterBias, OlalaTensorRole::kRouterWeight,
                            OlalaTensorRole::kFfnUpExps, OlalaTensorRole::kFfnDownExps,
                            OlalaTensorRole::kFfnUpShexp, OlalaTensorRole::kFfnDownShexp,
                            OlalaTensorRole::kFfnLatentUp, OlalaTensorRole::kFfnLatentDown,
                            OlalaTensorRole::kGeoMixerBias, OlalaTensorRole::kGeoMixerScale,
                            OlalaTensorRole::kGeoMlpBias, OlalaTensorRole::kGeoMlpScale}) {
    if (c.role_counts[r] != c.num_layers) {
      fail(std::string("role ") + OlalaRoleName(r) + " count " +
           std::to_string(c.role_counts[r]) + " != layer count " +
           std::to_string(c.num_layers));
    }
  }
  const int64_t n_ssm_roles = 15;
  for (int i = 0; i < n_ssm_roles; ++i) {
    OlalaTensorRole r = static_cast<OlalaTensorRole>(
        static_cast<int>(OlalaTensorRole::kSsmIn) + i);
    if (c.role_counts[r] != static_cast<int64_t>(c.m_layers.size())) {
      fail(std::string("M-role ") + OlalaRoleName(r) + " count " +
           std::to_string(c.role_counts[r]) + " != M-layer count " +
           std::to_string(c.m_layers.size()));
    }
  }
  for (int i = 0; i < n_attn; ++i) {
    OlalaTensorRole r = static_cast<OlalaTensorRole>(
        static_cast<int>(OlalaTensorRole::kAttnQ) + i);
    if (c.role_counts[r] != static_cast<int64_t>(c.v_layers.size())) {
      fail(std::string("V-role ") + OlalaRoleName(r) + " count " +
           std::to_string(c.role_counts[r]) + " != V-layer count " +
           std::to_string(c.v_layers.size()));
    }
  }

  // Globals vs config.
  const GgufTensorInfo* embed = nullptr;
  const GgufTensorInfo* head = nullptr;
  for (const GgufTensorInfo& t : gguf.Tensors()) {
    if (t.name == "token_embd.weight") embed = &t;
    if (t.name == "output.weight") head = &t;
  }
  if (embed == nullptr || head == nullptr) {
    fail("missing token_embd.weight or output.weight");
  } else {
    if (embed->shape.size() != 2 || embed->shape[1] != cfg.hidden_size) {
      fail("token_embd shape " + ShapeStr(embed->shape) +
           " != [vocab, hidden=" + std::to_string(cfg.hidden_size) + "]");
    }
    if (embed->shape != head->shape) {
      fail("output.weight " + ShapeStr(head->shape) +
           " != token_embd " + ShapeStr(embed->shape));
    }
    if (cfg.vocab_size != 0 && embed->shape[0] != cfg.vocab_size) {
      fail("token_embd rows " + std::to_string(embed->shape[0]) +
           " != config vocab_size " + std::to_string(cfg.vocab_size));
    }
  }

  // Router + experts vs config.
  for (const OlalaCensusEntry& e : c.entries) {
    if (e.role == OlalaTensorRole::kRouterWeight &&
        (e.shape.size() != 2 || e.shape[0] != cfg.num_experts ||
         e.shape[1] != cfg.hidden_size)) {
      fail("ffn_gate_inp " + e.name + " " + ShapeStr(e.shape) +
           " != [experts=" + std::to_string(cfg.num_experts) +
           ", hidden=" + std::to_string(cfg.hidden_size) + "]");
    }
    if (e.role == OlalaTensorRole::kRouterBias &&
        (e.shape.size() != 1 || e.shape[0] != cfg.num_experts)) {
      fail("exp_probs_b " + e.name + " " + ShapeStr(e.shape) +
           " != [experts=" + std::to_string(cfg.num_experts) + "]");
    }
    if (e.role == OlalaTensorRole::kFfnUpExps &&
        (e.shape.size() != 3 || e.shape[0] != cfg.num_experts)) {
      fail("ffn_up_exps " + e.name + " " + ShapeStr(e.shape) +
           " != [experts, latent, ff] with experts=" +
           std::to_string(cfg.num_experts));
    }
    if (e.role == OlalaTensorRole::kAttnQ &&
        (e.shape.size() != 2 ||
         e.shape[0] != cfg.num_attention_heads * cfg.head_dim ||
         e.shape[1] != cfg.hidden_size)) {
      fail("attn_q " + e.name + " " + ShapeStr(e.shape) +
           " != [heads*key_length=" +
           std::to_string(cfg.num_attention_heads * cfg.head_dim) +
           ", hidden=" + std::to_string(cfg.hidden_size) + "]");
    }
  }

  // Within a layer kind, each role must have one consistent shape.
  std::set<int64_t> v_set(c.v_layers.begin(), c.v_layers.end());
  std::map<OlalaTensorRole, std::vector<int64_t>> role_shape_m, role_shape_v;
  for (const OlalaCensusEntry& e : c.entries) {
    if (e.layer < 0 || e.role == OlalaTensorRole::kUnmapped) continue;
    const bool is_v = v_set.count(e.layer) != 0;
    auto& slot = is_v ? role_shape_v : role_shape_m;
    auto it = slot.find(e.role);
    if (it == slot.end()) {
      slot[e.role] = e.shape;
    } else if (it->second != e.shape) {
      fail(std::string("role ") + OlalaRoleName(e.role) +
           " has inconsistent shapes across layers: " + e.name + " " +
           ShapeStr(e.shape) + " vs " + ShapeStr(it->second));
    }
  }

  if (!err.empty()) throw std::runtime_error(err);
}

}  // namespace vllm
