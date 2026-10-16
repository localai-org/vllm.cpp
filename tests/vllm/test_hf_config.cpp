// Sliding-window cases port ModelConfig normalization from
// vllm/config/model.py:542-559,654-660,723-726,1232-1234. RoPE cases port
// vllm/transformers_utils/config.py:439-509 and rotary_embedding/__init__.py:
// 33-112,243-283 @ e24d1b24fe96.
#include <doctest/doctest.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/transformers_utils/hf_config.h"

namespace {

// Writes `body` to a unique file under the system temp dir and returns its
// path. Removed in the destructor so test runs don't accumulate files.
class TempJson {
 public:
  explicit TempJson(const std::string& body) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_hf_config_test_" + std::to_string(counter++) + ".json"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out << body;
  }
  ~TempJson() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// Writes config.json (+ optional generation_config.json) into a unique temp
// DIRECTORY and returns the config.json path, so sibling-file resolution is
// exercised the way a real checkpoint lays out. Removed in the destructor.
class TempModelDir {
 public:
  TempModelDir(const std::string& config_body, const std::string& gen_body) {
    static int counter = 0;
    dir_ = (std::filesystem::temp_directory_path() /
            ("vllm_hf_model_test_" + std::to_string(counter++)))
               .string();
    std::filesystem::create_directories(dir_);
    std::ofstream(dir_ + "/config.json", std::ios::binary) << config_body;
    if (!gen_body.empty()) {
      std::ofstream(dir_ + "/generation_config.json", std::ios::binary) << gen_body;
    }
  }
  ~TempModelDir() { std::filesystem::remove_all(dir_); }
  std::string config_path() const { return dir_ + "/config.json"; }

 private:
  std::string dir_;
};

// Qwen3-Next-like hybrid MoE config (key names per upstream
// vllm/transformers_utils/configs/qwen3_next.py).
constexpr const char* kHybridJson = R"({
  "model_type": "qwen3_5_moe",
  "architectures": ["Qwen3NextForCausalLM"],
  "hidden_size": 2048,
  "num_hidden_layers": 4,
  "vocab_size": 151936,
  "num_attention_heads": 16,
  "num_key_value_heads": 2,
  "head_dim": 256,
  "layer_types": ["linear_attention", "linear_attention", "linear_attention",
                  "full_attention"],
  "intermediate_size": 5120,
  "num_experts": 256,
  "num_experts_per_tok": 8,
  "moe_intermediate_size": 512,
  "shared_expert_intermediate_size": 512,
  "linear_num_key_heads": 16,
  "linear_num_value_heads": 32,
  "linear_key_head_dim": 128,
  "linear_value_head_dim": 128,
  "linear_conv_kernel_dim": 4,
  "mamba_ssm_dtype": "float32",
  "rope_theta": 5000000.0,
  "partial_rotary_factor": 0.25,
  "rms_norm_eps": 1e-06,
  "max_position_embeddings": 262144,
  "torch_dtype": "bfloat16"
})";

// Minimal llama-like config: no MoE/GDN keys, no explicit head_dim, no
// partial_rotary_factor, no layer_types.
constexpr const char* kLlamaJson = R"({
  "model_type": "llama",
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "vocab_size": 32000,
  "num_attention_heads": 32,
  "intermediate_size": 11008,
  "rms_norm_eps": 1e-05,
  "max_position_embeddings": 4096,
  "torch_dtype": "float16"
})";

// Multimodal wrapper config (Qwen3_5MoeForConditionalGeneration): the
// text-model fields are nested under `text_config`; only architectures,
// model_type and (here) torch_dtype live at the top level. Mirrors the real 35B
// config.json structure that LoadHfConfig must resolve via the upstream
// get_text_config() path.
constexpr const char* kNestedWrapperJson = R"({
  "architectures": ["Qwen3_5MoeForConditionalGeneration"],
  "model_type": "qwen3_5_moe",
  "torch_dtype": "bfloat16",
  "vision_config": {
    "model_type": "qwen3_5_moe",
    "depth": 27,
    "hidden_size": 1152
  },
  "text_config": {
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 2048,
    "num_hidden_layers": 40,
    "vocab_size": 248320,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention",
                    "full_attention", "linear_attention", "linear_attention",
                    "linear_attention", "full_attention", "linear_attention",
                    "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention",
                    "full_attention", "linear_attention", "linear_attention",
                    "linear_attention", "full_attention", "linear_attention",
                    "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention",
                    "full_attention", "linear_attention", "linear_attention",
                    "linear_attention", "full_attention", "linear_attention",
                    "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention",
                    "full_attention"],
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "mamba_ssm_dtype": "float32",
    "rope_theta": 5000000.0,
    "rms_norm_eps": 1e-06,
    "max_position_embeddings": 32768
  }
})";

}  // namespace

TEST_CASE("LoadHfConfig resolves nested text_config for a wrapper config") {
  TempJson f(kNestedWrapperJson);
  vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());

  // architectures + model_type come from the top-level wrapper.
  REQUIRE(cfg.architectures.size() == 1);
  CHECK(cfg.architectures[0] == "Qwen3_5MoeForConditionalGeneration");
  CHECK(cfg.model_type == "qwen3_5_moe");

  // Text-model fields resolved from the nested text_config, not the wrapper.
  CHECK(cfg.num_hidden_layers == 40);
  CHECK(cfg.num_experts == 256);
  CHECK(cfg.num_experts_per_tok == 8);
  REQUIRE(cfg.layer_types.size() == 40);
  CHECK(cfg.layer_types[0] == "linear_attention");
  CHECK(cfg.layer_types[3] == "full_attention");
  CHECK(cfg.layer_types[39] == "full_attention");
  CHECK(cfg.hidden_size == 2048);
  CHECK(cfg.vocab_size == 248320);
  CHECK(cfg.num_attention_heads == 16);
  CHECK(cfg.num_key_value_heads == 2);
  CHECK(cfg.head_dim == 256);
  CHECK(cfg.moe_intermediate_size == 512);
  CHECK(cfg.shared_expert_intermediate_size == 512);
  CHECK(cfg.linear_num_key_heads == 16);
  CHECK(cfg.linear_num_value_heads == 32);
  CHECK(cfg.linear_key_head_dim == 128);
  CHECK(cfg.linear_value_head_dim == 128);
  CHECK(cfg.linear_conv_kernel_dim == 4);
  CHECK(cfg.mamba_ssm_dtype == "float32");
  CHECK(cfg.rope_theta == doctest::Approx(5000000.0));
  // `.scale(0.0)` is load-bearing: doctest's Approx adds a scale term defaulting
  // to 1.0, so a bare `Approx(1e-6)` has an absolute tolerance of ~1.19e-5 and
  // accepts ANY eps below it -- including the 1e-05 this very file uses for
  // another family. Verified: mutating the fixture to 1e-05 left the bare form
  // passing and fails here.
  CHECK(cfg.rms_norm_eps == doctest::Approx(1e-6).scale(0.0));
  CHECK(cfg.max_position_embeddings == 32768);

  // partial_rotary_factor absent -> qwen family default 0.25 applies from the
  // resolved config (0.25 * 256 = 64).
  CHECK(cfg.rotary_dim == 64);

  // torch_dtype lives only at the top level here; fall back to the wrapper.
  CHECK(cfg.torch_dtype == "bfloat16");

  // raw keeps the full wrapper doc, including nested sub-configs.
  CHECK(cfg.raw.at("text_config").at("num_hidden_layers").get<int64_t>() == 40);
}

TEST_CASE("LoadHfConfig resolves modern dtype without changing legacy defaults") {
  auto plain = nlohmann::json::parse(kLlamaJson);
  const auto legacy = vllm::ParseHfConfig(plain, "legacy");
  CHECK(legacy.torch_dtype == "float16");
  CHECK(legacy.dtype_source == "torch_dtype");

  plain.erase("torch_dtype");
  plain["dtype"] = "float16";
  const auto modern = vllm::ParseHfConfig(plain, "modern");
  CHECK(modern.torch_dtype == "float16");
  CHECK(modern.dtype_source == "dtype");

  auto wrapper = nlohmann::json::parse(kNestedWrapperJson);
  wrapper.erase("torch_dtype");
  wrapper["text_config"]["dtype"] = "float16";
  const auto nested = vllm::ParseHfConfig(wrapper, "nested");
  CHECK(nested.torch_dtype == "float16");
  CHECK(nested.dtype_source == "text.dtype");

  plain["torch_dtype"] = "bfloat16";
  CHECK_THROWS_WITH_AS(vllm::ParseHfConfig(plain, "conflict"),
                       doctest::Contains("conflicting dtype declarations"),
                       std::runtime_error);
  plain.erase("torch_dtype");
  plain.erase("dtype");
  const auto missing = vllm::ParseHfConfig(plain, "missing");
  CHECK(missing.torch_dtype.empty());
  CHECK(missing.dtype_source.empty());
}

TEST_CASE("LoadHfConfig resolves llm_config alias and thinker_config.text_config") {
  SUBCASE("llm_config alias (upstream _CONFIG_ATTRS_MAPPING)") {
    TempJson f(R"({
      "architectures": ["SomeWrapperForCausalLM"],
      "model_type": "wrapper",
      "llm_config": {
        "hidden_size": 3584,
        "num_hidden_layers": 28,
        "num_attention_heads": 28
      }
    })");
    vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
    CHECK(cfg.model_type == "wrapper");
    CHECK(cfg.hidden_size == 3584);
    CHECK(cfg.num_hidden_layers == 28);
  }
  SUBCASE("thinker_config.text_config") {
    TempJson f(R"({
      "architectures": ["ThinkerForConditionalGeneration"],
      "model_type": "thinker_wrapper",
      "thinker_config": {
        "text_config": {
          "hidden_size": 4096,
          "num_hidden_layers": 36,
          "num_attention_heads": 32
        }
      }
    })");
    vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
    CHECK(cfg.model_type == "thinker_wrapper");
    CHECK(cfg.hidden_size == 4096);
    CHECK(cfg.num_hidden_layers == 36);
  }
}

TEST_CASE("LoadHfConfig on the real 35B config.json (skipped if absent)") {
  // Point VLLM_CPP_QWEN35_CONFIG at a real
  // Qwen3_5MoeForConditionalGeneration config.json to exercise this on dgx.
  const char* env = std::getenv("VLLM_CPP_QWEN35_CONFIG");
  if (env == nullptr || !std::filesystem::exists(env)) {
    MESSAGE("skipping: VLLM_CPP_QWEN35_CONFIG not set to an existing file");
    return;
  }
  vllm::HfConfig cfg = vllm::LoadHfConfig(env);
  CHECK(cfg.num_hidden_layers == 40);
  CHECK(cfg.num_experts == 256);
  CHECK(cfg.layer_types.size() == 40);
}

TEST_CASE("LoadHfConfig parses a Qwen3-Next-like hybrid MoE config") {
  TempJson f(kHybridJson);
  vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());

  CHECK(cfg.model_type == "qwen3_5_moe");
  REQUIRE(cfg.architectures.size() == 1);
  CHECK(cfg.architectures[0] == "Qwen3NextForCausalLM");
  CHECK(cfg.hidden_size == 2048);
  CHECK(cfg.num_hidden_layers == 4);
  CHECK(cfg.vocab_size == 151936);
  CHECK(cfg.num_attention_heads == 16);
  CHECK(cfg.num_key_value_heads == 2);
  CHECK(cfg.head_dim == 256);  // explicit key wins over hidden/heads = 128

  REQUIRE(cfg.layer_types.size() == 4);
  CHECK(cfg.layer_types[0] == "linear_attention");
  CHECK(cfg.layer_types[3] == "full_attention");

  CHECK(cfg.intermediate_size == 5120);
  CHECK(cfg.num_experts == 256);
  CHECK(cfg.num_experts_per_tok == 8);
  CHECK(cfg.moe_intermediate_size == 512);
  CHECK(cfg.shared_expert_intermediate_size == 512);

  CHECK(cfg.linear_num_key_heads == 16);
  CHECK(cfg.linear_num_value_heads == 32);
  CHECK(cfg.linear_key_head_dim == 128);
  CHECK(cfg.linear_value_head_dim == 128);
  CHECK(cfg.linear_conv_kernel_dim == 4);
  CHECK(cfg.mamba_ssm_dtype == "float32");

  CHECK(cfg.rope_theta == doctest::Approx(5000000.0));
  CHECK(cfg.rotary_dim == 64);  // 0.25 * 256
  // `.scale(0.0)` is load-bearing: doctest's Approx adds a scale term defaulting
  // to 1.0, so a bare `Approx(1e-6)` has an absolute tolerance of ~1.19e-5 and
  // accepts ANY eps below it -- including the 1e-05 this very file uses for
  // another family. Verified: mutating the fixture to 1e-05 left the bare form
  // passing and fails here.
  CHECK(cfg.rms_norm_eps == doctest::Approx(1e-6).scale(0.0));
  CHECK(cfg.max_position_embeddings == 262144);
  CHECK(cfg.torch_dtype == "bfloat16");

  // Raw doc keeps untyped fields accessible.
  CHECK(cfg.raw.at("partial_rotary_factor").get<double>() ==
        doctest::Approx(0.25));
}

TEST_CASE("LoadHfConfig parses a minimal llama-like config with defaults") {
  TempJson f(kLlamaJson);
  vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());

  CHECK(cfg.model_type == "llama");
  CHECK(cfg.hidden_size == 4096);
  CHECK(cfg.num_hidden_layers == 32);
  CHECK(cfg.num_attention_heads == 32);
  // num_key_value_heads absent -> num_attention_heads (MHA).
  CHECK(cfg.num_key_value_heads == 32);
  // head_dim absent -> hidden_size / num_attention_heads.
  CHECK(cfg.head_dim == 128);
  // No partial_rotary_factor -> full rotary.
  CHECK(cfg.rotary_dim == 128);
  // rope_theta absent -> upstream default 10000.0.
  CHECK(cfg.rope_theta == doctest::Approx(10000.0));

  CHECK(cfg.layer_types.empty());  // empty = all full attention

  // MoE / GDN keys absent -> zeros.
  CHECK(cfg.num_experts == 0);
  CHECK(cfg.num_experts_per_tok == 0);
  CHECK(cfg.moe_intermediate_size == 0);
  CHECK(cfg.shared_expert_intermediate_size == 0);
  CHECK(cfg.linear_num_key_heads == 0);
  CHECK(cfg.linear_num_value_heads == 0);
  CHECK(cfg.linear_key_head_dim == 0);
  CHECK(cfg.linear_value_head_dim == 0);
  CHECK(cfg.linear_conv_kernel_dim == 0);

  CHECK(cfg.torch_dtype == "float16");
}

TEST_CASE("LoadHfConfig applies upstream rotary and head_dim defaults") {
  SUBCASE("qwen family without partial_rotary_factor defaults to 0.25") {
    // Upstream Qwen config classes default partial_rotary_factor to 0.25
    // (qwen3_next.py:240, qwen3_5_moe.py:92).
    TempJson f(R"({
      "model_type": "qwen3_5_moe",
      "hidden_size": 2048,
      "num_hidden_layers": 4,
      "num_attention_heads": 16,
      "head_dim": 256
    })");
    vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
    CHECK(cfg.head_dim == 256);
    CHECK(cfg.rotary_dim == 64);  // head_dim / 4
  }
  SUBCASE("non-qwen model without partial_rotary_factor uses full rotary") {
    TempJson f(R"({
      "model_type": "llama",
      "hidden_size": 4096,
      "num_hidden_layers": 2,
      "num_attention_heads": 32
    })");
    vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
    CHECK(cfg.head_dim == 128);
    CHECK(cfg.rotary_dim == 128);
  }
  SUBCASE("explicit head_dim of 0 falls back to derived value") {
    // Upstream honors explicit head_dim only when > 0
    // (model_arch_config_convertor.py:61-75).
    TempJson f(R"({
      "model_type": "llama",
      "hidden_size": 4096,
      "num_hidden_layers": 2,
      "num_attention_heads": 32,
      "head_dim": 0
    })");
    vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
    CHECK(cfg.head_dim == 128);   // hidden_size / num_attention_heads
    CHECK(cfg.rotary_dim == 128);
  }
}

TEST_CASE("LoadHfConfig mirrors sliding_window normalization from the text config") {
  SUBCASE("positive top-level window is retained") {
    TempJson f(R"({
      "model_type": "starcoder2",
      "hidden_size": 64,
      "num_hidden_layers": 2,
      "num_attention_heads": 4,
      "sliding_window": 4096
    })");
    const vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
    REQUIRE(cfg.sliding_window.has_value());
    CHECK(*cfg.sliding_window == 4096);
  }

  SUBCASE("zero and null mean disabled") {
    for (const char* value : {"0", "null"}) {
      TempJson f(std::string(R"({
        "model_type": "local",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "sliding_window": )") + value + "}");
      CHECK_FALSE(vllm::LoadHfConfig(f.path()).sliding_window.has_value());
    }
  }

  SUBCASE("nested text_config is the source") {
    TempJson f(R"({
      "model_type": "wrapper",
      "architectures": ["WrapperForCausalLM"],
      "sliding_window": 999,
      "text_config": {
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "sliding_window": 128
      }
    })");
    const vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
    REQUIRE(cfg.sliding_window.has_value());
    CHECK(*cfg.sliding_window == 128);
  }
}

// ROW 7 / kimi-linear.md §20.3 B1 — KV enablement for the shared paged runner.
// Kimi-Linear's config carries NO `layer_types` and NONE of the qwen3_5-style
// explicit `linear_*` GDN-geometry keys: its KDA/full-attn split lives in the
// nested `linear_attn_config` (transformers_utils/configs/kimi_linear.py:34-148,
// `is_kda_layer(l) := (l+1) in kda_layers` :144-148 — the lists are 1-INDEXED).
// LoadHfConfig must SYNTHESIZE the runner-facing fields from it so the runner's
// MambaSpec shape check (runner.cpp:489-493) and the per-layer KDA-state /
// MLA-page allocation loop (runner.cpp:625-) see the same geometry
// MakeKimiLinearKVCache declares — instead of {0,0},{0,0,0} and an abort.
TEST_CASE("LoadHfConfig synthesizes layer_types + GDN geometry from linear_attn_config") {
  TempJson f(R"({
    "model_type": "kimi_linear",
    "architectures": ["KimiLinearForCausalLM"],
    "hidden_size": 2304,
    "num_hidden_layers": 8,
    "vocab_size": 163840,
    "num_attention_heads": 32,
    "intermediate_size": 9216,
    "rms_norm_eps": 1e-05,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "mla_use_nope": true,
    "linear_attn_config": {
      "kda_layers": [1, 2, 3, 5, 6, 7],
      "full_attn_layers": [4, 8],
      "num_heads": 32,
      "head_dim": 128,
      "short_conv_kernel_size": 4
    },
    "max_position_embeddings": 1048576
  })");
  vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());

  // GDN-group geometry sourced from linear_attn_config (num_k == num_v == the
  // KDA num_heads; Dk == Dv == head_dim; conv kernel == short_conv_kernel_size),
  // so the runner derives conv_dim = 2*Hk*Dk + Hv*Dv = 3*32*128 = 12288 and the
  // ssm shape {32,128,128} — exactly MakeKimiLinearKVCache's MambaSpec.
  CHECK(cfg.linear_num_key_heads == 32);
  CHECK(cfg.linear_num_value_heads == 32);
  CHECK(cfg.linear_key_head_dim == 128);
  CHECK(cfg.linear_value_head_dim == 128);
  CHECK(cfg.linear_conv_kernel_dim == 4);

  // layer_types synthesized from the 1-indexed kda_layers list.
  REQUIRE(cfg.layer_types.size() == 8);
  const std::vector<std::string> expect = {
      "linear_attention", "linear_attention", "linear_attention",
      "full_attention",   "linear_attention", "linear_attention",
      "linear_attention", "full_attention"};
  for (size_t i = 0; i < expect.size(); ++i) CHECK(cfg.layer_types[i] == expect[i]);

  // The raw doc still carries linear_attn_config for ParseKimiLinearParams.
  CHECK(cfg.raw.contains("linear_attn_config"));
}

// The synthesis must be ADDITIVE: a config that carries the explicit qwen3_5
// fields (kHybridJson has both layer_types AND linear_*) is untouched by it —
// that path is byte-identical (asserted by the hybrid TEST_CASE above), and a
// config with NEITHER (plain dense llama) stays all-zero / empty.
TEST_CASE("LoadHfConfig linear_attn_config synthesis leaves non-Kimi configs untouched") {
  TempJson f(kLlamaJson);
  vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
  CHECK(cfg.linear_num_key_heads == 0);
  CHECK(cfg.linear_num_value_heads == 0);
  CHECK(cfg.linear_conv_kernel_dim == 0);
  CHECK(cfg.layer_types.empty());
}

// The PRIORITY conjunct (`cfg.linear_num_key_heads == 0`): a config carrying
// BOTH the explicit qwen3_5-style keys AND a linear_attn_config keeps the
// EXPLICIT values — the synthesis never overwrites them. Drop the conjunct and
// this case REDs (num_value_heads would become 4 == num_heads, key_head_dim 16,
// conv 4). This makes B1 additive-by-construction, pinned rather than assumed.
TEST_CASE("LoadHfConfig explicit linear_* fields WIN over linear_attn_config") {
  TempJson f(R"({
    "model_type": "qwen3_5_moe",
    "architectures": ["Qwen3NextForCausalLM"],
    "hidden_size": 64,
    "num_hidden_layers": 3,
    "vocab_size": 128,
    "num_attention_heads": 4,
    "layer_types": ["linear_attention", "full_attention", "linear_attention"],
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 8,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 64,
    "linear_conv_kernel_dim": 3,
    "linear_attn_config": {
      "kda_layers": [1, 2, 3],
      "full_attn_layers": [],
      "num_heads": 4,
      "head_dim": 16,
      "short_conv_kernel_size": 4
    }
  })");
  vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
  // The explicit (qwen3_5-style, asymmetric) geometry survives verbatim ...
  CHECK(cfg.linear_num_key_heads == 2);
  CHECK(cfg.linear_num_value_heads == 8);
  CHECK(cfg.linear_key_head_dim == 32);
  CHECK(cfg.linear_value_head_dim == 64);
  CHECK(cfg.linear_conv_kernel_dim == 3);
  // ... and the explicit layer_types are NOT resynthesized from kda_layers
  // (which would flip index 1 to linear_attention).
  REQUIRE(cfg.layer_types.size() == 3);
  CHECK(cfg.layer_types[1] == "full_attention");
}

TEST_CASE("LoadHfConfig throws when required fields are missing") {
  SUBCASE("missing model_type") {
    TempJson f(R"({"hidden_size": 64, "num_hidden_layers": 2})");
    CHECK_THROWS_AS(vllm::LoadHfConfig(f.path()), std::runtime_error);
  }
  SUBCASE("missing hidden_size") {
    TempJson f(R"({"model_type": "llama", "num_hidden_layers": 2})");
    CHECK_THROWS_AS(vllm::LoadHfConfig(f.path()), std::runtime_error);
  }
  SUBCASE("missing num_hidden_layers") {
    TempJson f(R"({"model_type": "llama", "hidden_size": 64})");
    CHECK_THROWS_AS(vllm::LoadHfConfig(f.path()), std::runtime_error);
  }
}

TEST_CASE("LoadHfConfig error paths mention the file path") {
  SUBCASE("file not found") {
    const std::string missing = "/nonexistent/vllm_cpp_no_such_config.json";
    try {
      vllm::LoadHfConfig(missing);
      FAIL("expected std::runtime_error");
    } catch (const std::runtime_error& e) {
      CHECK(std::string(e.what()).find(missing) != std::string::npos);
    }
  }
  SUBCASE("invalid JSON") {
    TempJson f("{ not json !!");
    try {
      vllm::LoadHfConfig(f.path());
      FAIL("expected std::runtime_error");
    } catch (const std::runtime_error& e) {
      CHECK(std::string(e.what()).find(f.path()) != std::string::npos);
    }
  }
}

TEST_CASE("LoadHfConfig retains the complete typed YaRN parameter slice") {
  TempJson f(R"({
    "model_type": "nomic_bert",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "max_position_embeddings": 8192,
    "rope_parameters": {
      "rope_type": "yarn",
      "rope_theta": 1000.0,
      "rope_dim": 12,
      "factor": 4.0,
      "original_max_position_embeddings": 2048,
      "extrapolation_factor": 1.25,
      "attn_factor": 0.75,
      "beta_fast": 24,
      "beta_slow": 2,
      "apply_yarn_scaling": false,
      "truncate": false,
      "mrope_section": [2, 2, 2],
      "mrope_interleaved": true
    }
  })");
  const vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
  REQUIRE(cfg.has_rope_parameters);
  CHECK(cfg.rope_theta == doctest::Approx(1000.0));
  CHECK(cfg.rotary_dim == 12);
  CHECK(cfg.rope_parameters.rope_type == "yarn");
  REQUIRE(cfg.rope_parameters.factor.has_value());
  CHECK(*cfg.rope_parameters.factor == doctest::Approx(4.0));
  REQUIRE(cfg.rope_parameters.original_max_position_embeddings.has_value());
  CHECK(*cfg.rope_parameters.original_max_position_embeddings == 2048);
  CHECK(cfg.rope_parameters.extrapolation_factor == doctest::Approx(1.25));
  CHECK(cfg.rope_parameters.attn_factor == doctest::Approx(0.75));
  CHECK(cfg.rope_parameters.beta_fast == 24);
  CHECK(cfg.rope_parameters.beta_slow == 2);
  CHECK_FALSE(cfg.rope_parameters.apply_yarn_scaling);
  CHECK_FALSE(cfg.rope_parameters.truncate);
  CHECK(cfg.rope_parameters.mrope_section ==
        std::vector<int64_t>{2, 2, 2});
  CHECK(cfg.rope_parameters.mrope_interleaved);
}

TEST_CASE("LoadHfConfig retains the complete typed Llama 3 parameter slice") {
  TempJson f(R"({
    "model_type": "llama",
    "hidden_size": 96,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "max_position_embeddings": 128,
    "rope_parameters": {
      "rope_type": "llama3",
      "rope_theta": 10000.0,
      "rope_dim": 16,
      "factor": 4.0,
      "low_freq_factor": 1.0,
      "high_freq_factor": 4.0,
      "original_max_position_embeddings": 32
    }
  })");
  const vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
  REQUIRE(cfg.has_rope_parameters);
  CHECK(cfg.rope_parameters.rope_type == "llama3");
  CHECK(cfg.rotary_dim == 16);
  REQUIRE(cfg.rope_parameters.factor.has_value());
  REQUIRE(cfg.rope_parameters.low_freq_factor.has_value());
  REQUIRE(cfg.rope_parameters.high_freq_factor.has_value());
  REQUIRE(cfg.rope_parameters.original_max_position_embeddings.has_value());
  CHECK(*cfg.rope_parameters.factor == doctest::Approx(4.0));
  CHECK(*cfg.rope_parameters.low_freq_factor == doctest::Approx(1.0));
  CHECK(*cfg.rope_parameters.high_freq_factor == doctest::Approx(4.0));
  CHECK(*cfg.rope_parameters.original_max_position_embeddings == 32);
}

TEST_CASE("LoadHfConfig retains and normalizes Phi-3 LongRoPE parameters") {
  TempJson f(R"({
    "model_type": "phi3",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "max_position_embeddings": 128,
    "rope_scaling": {
      "type": "su",
      "rope_theta": 10000.0,
      "rope_dim": 8,
      "original_max_position_embeddings": 32,
      "short_factor": [1.0, 1.1, 1.2, 1.3],
      "long_factor": [2.0, 2.2, 2.4, 2.6],
      "short_mscale": 0.75,
      "long_mscale": 1.25
    }
  })");
  const vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
  REQUIRE(cfg.has_rope_parameters);
  CHECK(cfg.rope_parameters.rope_type == "longrope");
  CHECK(cfg.rotary_dim == 8);
  CHECK(cfg.rope_parameters.short_factor ==
        std::vector<double>{1.0, 1.1, 1.2, 1.3});
  CHECK(cfg.rope_parameters.long_factor ==
        std::vector<double>{2.0, 2.2, 2.4, 2.6});
  REQUIRE(cfg.rope_parameters.short_mscale.has_value());
  REQUIRE(cfg.rope_parameters.long_mscale.has_value());
  CHECK(*cfg.rope_parameters.short_mscale == doctest::Approx(0.75));
  CHECK(*cfg.rope_parameters.long_mscale == doctest::Approx(1.25));
  REQUIRE(cfg.rope_parameters.original_max_position_embeddings.has_value());
  CHECK(*cfg.rope_parameters.original_max_position_embeddings == 32);
}

TEST_CASE("LoadHfConfig retains both dynamic NTK dispatch forms") {
  SUBCASE("factor mode retains trained context") {
    TempJson f(R"({
      "model_type": "llama",
      "hidden_size": 64,
      "num_hidden_layers": 2,
      "num_attention_heads": 4,
      "max_position_embeddings": 128,
      "rope_parameters": {
        "rope_type": "dynamic",
        "rope_theta": 10000.0,
        "rope_dim": 8,
        "factor": 4.0,
        "max_trained_positions": 32
      }
    })");
    const vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
    CHECK(cfg.rope_parameters.rope_type == "dynamic");
    REQUIRE(cfg.rope_parameters.factor.has_value());
    REQUIRE(cfg.rope_parameters.max_trained_positions.has_value());
    CHECK(*cfg.rope_parameters.factor == doctest::Approx(4.0));
    CHECK(*cfg.rope_parameters.max_trained_positions == 32);
  }

  SUBCASE("alpha mode is typed independently") {
    TempJson f(R"({
      "model_type": "hunyuan",
      "hidden_size": 64,
      "num_hidden_layers": 2,
      "num_attention_heads": 4,
      "max_position_embeddings": 128,
      "rope_parameters": {
        "rope_type": "dynamic",
        "rope_dim": 8,
        "alpha": 2.0
      }
    })");
    const vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
    REQUIRE(cfg.rope_parameters.alpha.has_value());
    CHECK(*cfg.rope_parameters.alpha == doctest::Approx(2.0));
    CHECK_FALSE(cfg.rope_parameters.factor.has_value());
  }
}

TEST_CASE("LoadHfConfig normalizes legacy and default MRoPE dictionaries") {
  SUBCASE("legacy rope_scaling type yarn is standardized") {
    TempJson f(R"({
      "model_type": "legacy",
      "hidden_size": 64,
      "num_hidden_layers": 2,
      "num_attention_heads": 4,
      "rope_theta": 5000.0,
      "rope_scaling": {
        "type": "yarn",
        "factor": 2.0,
        "original_max_position_embeddings": 32
      }
    })");
    const vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
    CHECK(cfg.rope_parameters.rope_type == "yarn");
    CHECK(cfg.rope_parameters.rope_theta == doctest::Approx(5000.0));
    CHECK(*cfg.rope_parameters.factor == doctest::Approx(2.0));
  }

  SUBCASE("legacy mrope type becomes default and retains sections") {
    TempJson f(R"({
      "model_type": "qwen_vl",
      "hidden_size": 64,
      "num_hidden_layers": 2,
      "num_attention_heads": 4,
      "rope_parameters": {
        "type": "mrope",
        "rope_theta": 1000000.0,
        "mrope_section": [2, 3, 3],
        "mrope_interleaved": true
      }
    })");
    const vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
    CHECK(cfg.rope_parameters.rope_type == "default");
    CHECK(cfg.rope_parameters.mrope_section ==
          std::vector<int64_t>{2, 3, 3});
    CHECK(cfg.rope_parameters.mrope_interleaved);
  }
}

TEST_CASE("LoadHfConfig keeps unsupported or malformed RoPE loud") {
  const std::string prefix = R"({
    "model_type": "rope_test",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "rope_parameters": )";
  SUBCASE("future formula family remains rejected until its leaf lands") {
    TempJson f(prefix + R"({"rope_type":"future_formula"}})");
    CHECK_THROWS_WITH_AS(vllm::LoadHfConfig(f.path()),
                         doctest::Contains("does not implement yet"),
                         std::runtime_error);
  }
  SUBCASE("YaRN requires both dispatch fields") {
    TempJson f(prefix + R"({"rope_type":"yarn","factor":4.0}})");
    CHECK_THROWS_WITH_AS(vllm::LoadHfConfig(f.path()),
                         doctest::Contains("requires factor"),
                         std::runtime_error);
  }
  SUBCASE("Llama 3 requires every frequency-band field") {
    TempJson f(prefix +
               R"({"rope_type":"llama3","factor":4.0,"low_freq_factor":1.0}})");
    CHECK_THROWS_WITH_AS(vllm::LoadHfConfig(f.path()),
                         doctest::Contains("llama3 rope requires"),
                         std::runtime_error);
  }
  SUBCASE("LongRoPE requires both factor arrays and original context") {
    TempJson f(prefix +
               R"({"rope_type":"longrope","short_factor":[1.0,1.0]}})");
    CHECK_THROWS_WITH_AS(vllm::LoadHfConfig(f.path()),
                         doctest::Contains("longrope requires"),
                         std::runtime_error);
  }
  SUBCASE("dynamic NTK requires alpha or factor") {
    TempJson f(prefix + R"({"rope_type":"dynamic"}})");
    CHECK_THROWS_WITH_AS(vllm::LoadHfConfig(f.path()),
                         doctest::Contains("requires either alpha or factor"),
                         std::runtime_error);
  }
  SUBCASE("nested per-layer rope loads; owning model reads it from raw (G1b)") {
    // Gemma-4-shaped per-layer-TYPE rope. As of Gemma-4 G1b this is SUPPORTED
    // (mirrors vLLM, which keeps per-layer-type rope configs on the model): the
    // loader no longer aborts. There is no single flat typed rope for a
    // heterogeneous-rope model, so the typed fields stay at their defaults and
    // the nested structure is preserved verbatim in config.raw for the owning
    // forward to consume (gemma4.cpp::RopeField). This is additive: no
    // previously-loadable model reached this branch (it used to throw).
    TempJson f(prefix +
               R"({"full_attention":{"rope_type":"proportional","rope_theta":1000000.0,"partial_rotary_factor":0.25},"sliding_attention":{"rope_type":"default","rope_theta":10000.0}}})");
    vllm::HfConfig cfg;
    REQUIRE_NOTHROW(cfg = vllm::LoadHfConfig(f.path()));
    CHECK(cfg.has_rope_parameters);
    // The per-layer-type rope is intact in raw (what the model actually reads).
    const auto& rp = cfg.raw.at("rope_parameters");
    CHECK(rp.at("full_attention").at("rope_theta").get<double>() == 1000000.0);
    CHECK(rp.at("sliding_attention").at("rope_theta").get<double>() == 10000.0);
  }
}

TEST_CASE("LoadHfConfig admits Gemma 3 linear RoPE with its factor") {
  TempJson f(R"({"model_type":"gemma3_text","architectures":["Gemma3ForCausalLM"],
    "hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,
    "rope_theta":1000000.0,"rope_scaling":{"rope_type":"linear","factor":8.0}})");
  const auto cfg = vllm::LoadHfConfig(f.path());
  CHECK(cfg.rope_parameters.rope_type == "linear");
  REQUIRE(cfg.rope_parameters.factor.has_value());
  CHECK(*cfg.rope_parameters.factor == 8.0);
  CHECK(cfg.rope_parameters.rope_theta == 1000000.0);
}

// ─── generation_config.json eos ids (vllm/config/model.py try_get_generation_
// config -> sampling_params.py:645-655) ──────────────────────────────────────
// Upstream loads generation_config.json alongside config.json whenever
// --generation-config is "auto" (the default) or "vllm". Its eos_token_id is a
// DIFFERENT, usually larger set than config.json's: Gemma-4-26B ships
// config.json [1, 106] and generation_config.json [1, 106, 50].
TEST_CASE("LoadHfConfig reads sibling generation_config.json eos ids") {
  constexpr const char* kConfig = R"({
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 8,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "vocab_size": 32,
    "max_position_embeddings": 128,
    "eos_token_id": [1, 106]
  })";

  SUBCASE("array eos is unioned, config.json raw left untouched") {
    TempModelDir model(kConfig, R"({"eos_token_id": [1, 106, 50]})");
    const vllm::HfConfig cfg = vllm::LoadHfConfig(model.config_path());
    // Sorted, de-duplicated union.
    std::vector<int32_t> expected = {1, 50, 106};
    CHECK(cfg.generation_config_eos_ids == expected);
    // raw["eos_token_id"] is the checkpoint's own field and must not be
    // rewritten: the PRIMARY eos id is derived from it downstream.
    CHECK(cfg.raw["eos_token_id"] == nlohmann::json::array({1, 106}));
  }

  SUBCASE("scalar eos in generation_config.json") {
    TempModelDir model(kConfig, R"({"eos_token_id": 50})");
    const vllm::HfConfig cfg = vllm::LoadHfConfig(model.config_path());
    std::vector<int32_t> expected = {50};
    CHECK(cfg.generation_config_eos_ids == expected);
  }

  SUBCASE("absent generation_config.json leaves the list empty") {
    TempModelDir model(kConfig, "");
    const vllm::HfConfig cfg = vllm::LoadHfConfig(model.config_path());
    CHECK(cfg.generation_config_eos_ids.empty());
  }

  SUBCASE("malformed generation_config.json is a silent no-op, not a throw") {
    TempModelDir model(kConfig, "{ this is not json");
    vllm::HfConfig cfg;
    REQUIRE_NOTHROW(cfg = vllm::LoadHfConfig(model.config_path()));
    CHECK(cfg.generation_config_eos_ids.empty());
  }

  SUBCASE("generation_config.json without an eos field") {
    TempModelDir model(kConfig, R"({"temperature": 0.7})");
    const vllm::HfConfig cfg = vllm::LoadHfConfig(model.config_path());
    CHECK(cfg.generation_config_eos_ids.empty());
  }
}

namespace {

// GDN gate-activation fixtures. The text-model body is shared so the flat and
// nested arms differ ONLY in where the key lives -- the exact confusion the
// nested arm exists to catch.
std::string GdnTextFields(const std::string& gate_entry) {
  return std::string(R"(
    "hidden_size": 2048,
    "num_hidden_layers": 4,
    "vocab_size": 151936,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention",
                    "full_attention"],
    "intermediate_size": 5120,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "rms_norm_eps": 1e-06,
    "max_position_embeddings": 262144)") +
         gate_entry;
}

// A flat text-only checkpoint (Qwen3-Next layout): the key sits at top level.
std::string FlatGdnJson(const std::string& gate_entry) {
  return std::string(R"({
    "model_type": "qwen3_5",
    "architectures": ["Qwen3_5ForCausalLM"],)") +
         "\n" + GdnTextFields(gate_entry) + "\n}";
}

// A composite VL wrapper (Qwen3_5MoeForConditionalGeneration layout): the key
// sits inside `text_config`, and the wrapper level carries a DIFFERENT
// (deliberately unrecognized) value that must never be read.
std::string NestedGdnJson(const std::string& gate_entry) {
  return std::string(R"({
  "model_type": "qwen3_5_moe",
  "architectures": ["Qwen3_5MoeForConditionalGeneration"],
  "output_gate_type": "not_a_gate",
  "text_config": {
    "model_type": "qwen3_5_moe_text",)") +
         "\n" + GdnTextFields(gate_entry) + "\n  }\n}";
}

const std::string kSilu = R"(,
    "output_gate_type": "silu")";
const std::string kSwish = R"(,
    "output_gate_type": "swish")";
const std::string kSigmoid = R"(,
    "output_gate_type": "sigmoid")";
const std::string kBogus = R"(,
    "output_gate_type": "gelu")";
// PRESENT but unusable. Upstream's `getattr(config, "output_gate_type",
// "silu")` only substitutes the default when the attribute is ABSENT: a
// present None or "" is returned as-is and then fails
// `assert output_gate_type in ["silu", "swish", "sigmoid"]`.
const std::string kNull = R"(,
    "output_gate_type": null)";
const std::string kEmpty = R"(,
    "output_gate_type": "")";
// PRESENT and not a string at all. `getattr` returns whatever the attribute
// holds regardless of type, so a number or a bool reaches upstream's assert the
// same way None does. Locally these take the `dump()` arm of the parse, which
// null alone was covering.
const std::string kNumber = R"(,
    "output_gate_type": 3)";
const std::string kBool = R"(,
    "output_gate_type": true)";

}  // namespace

// Ports vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:452-456
// @555967922:
//   output_gate_type = getattr(config, "output_gate_type", "silu")
//   if output_gate_type == "swish": output_gate_type = "silu"
//   assert output_gate_type in ["silu", "swish", "sigmoid"]
// Canonicalized at PARSE time (spec gdn-output-gate-type.md "Design"), so
// HfConfig only ever holds "silu" or "sigmoid" and no call site can
// reintroduce the default by forgetting to normalize. Upstream asserts on an
// unrecognized value; we refuse at load with a message naming the key and the
// accepted set rather than silently defaulting.
TEST_CASE("LoadHfConfig normalizes the GDN output_gate_type") {
  SUBCASE("absent key resolves to silu") {
    TempJson f(kHybridJson);  // carries no output_gate_type
    CHECK(vllm::LoadHfConfig(f.path()).output_gate_type == "silu");
  }

  SUBCASE("explicit silu stays silu") {
    TempJson f(FlatGdnJson(kSilu));
    CHECK(vllm::LoadHfConfig(f.path()).output_gate_type == "silu");
  }

  SUBCASE("swish collapses to silu, exactly as upstream") {
    TempJson f(FlatGdnJson(kSwish));
    CHECK(vllm::LoadHfConfig(f.path()).output_gate_type == "silu");
  }

  SUBCASE("sigmoid is preserved") {
    TempJson f(FlatGdnJson(kSigmoid));
    CHECK(vllm::LoadHfConfig(f.path()).output_gate_type == "sigmoid");
  }

  SUBCASE("an unrecognized value is refused, never silently defaulted") {
    TempJson f(FlatGdnJson(kBogus));
    CHECK_THROWS_WITH_AS(vllm::LoadHfConfig(f.path()),
                         doctest::Contains("output_gate_type"),
                         std::runtime_error);
    CHECK_THROWS_WITH_AS(vllm::LoadHfConfig(f.path()),
                         doctest::Contains("sigmoid"), std::runtime_error);
  }

  // ABSENT and PRESENT-BUT-NULL are different states, and only the first one
  // takes upstream's `getattr` default. A `"output_gate_type": null` reaches
  // upstream as None, fails the assert, and errors — so it must be refused
  // here rather than collapsing into silu, which is what docs/USAGE.md already
  // tells a user this loader does.
  SUBCASE("a present null is refused, never treated as absent") {
    TempJson f(FlatGdnJson(kNull));
    CHECK_THROWS_WITH_AS(vllm::LoadHfConfig(f.path()),
                         doctest::Contains("output_gate_type"),
                         std::runtime_error);
    CHECK_THROWS_WITH_AS(vllm::LoadHfConfig(f.path()),
                         doctest::Contains("sigmoid"), std::runtime_error);

    // Same in a nested wrapper: the resolved text config is what decides.
    TempJson nested(NestedGdnJson(kNull));
    CHECK_THROWS_AS(vllm::LoadHfConfig(nested.path()), std::runtime_error);
  }

  // An empty string is a value, not an absence: upstream returns "" from
  // getattr and the assert rejects it.
  SUBCASE("a present empty string is refused, never treated as absent") {
    TempJson f(FlatGdnJson(kEmpty));
    CHECK_THROWS_WITH_AS(vllm::LoadHfConfig(f.path()),
                         doctest::Contains("output_gate_type"),
                         std::runtime_error);
    CHECK_THROWS_WITH_AS(vllm::LoadHfConfig(f.path()),
                         doctest::Contains("sigmoid"), std::runtime_error);

    TempJson nested(NestedGdnJson(kEmpty));
    CHECK_THROWS_AS(vllm::LoadHfConfig(nested.path()), std::runtime_error);
  }

  // `null` is only ONE of the non-string JSON types the parse routes through
  // `dump()`, and it was the only one exercised. A number and a bool are just
  // as reachable from a hand-edited or generator-written config, take the same
  // arm, and must be refused with the offending value QUOTED into the message —
  // the whole reason that arm dumps rather than substituting a default.
  SUBCASE("a present non-string value is refused, naming what was found") {
    TempJson num(FlatGdnJson(kNumber));
    CHECK_THROWS_WITH_AS(vllm::LoadHfConfig(num.path()),
                         doctest::Contains("output_gate_type"),
                         std::runtime_error);
    CHECK_THROWS_WITH_AS(vllm::LoadHfConfig(num.path()),
                         doctest::Contains("\"3\""), std::runtime_error);

    TempJson yes(FlatGdnJson(kBool));
    CHECK_THROWS_WITH_AS(vllm::LoadHfConfig(yes.path()),
                         doctest::Contains("output_gate_type"),
                         std::runtime_error);
    CHECK_THROWS_WITH_AS(vllm::LoadHfConfig(yes.path()),
                         doctest::Contains("\"true\""), std::runtime_error);

    // Same in a nested wrapper: the resolved text config is what decides.
    TempJson nested(NestedGdnJson(kNumber));
    CHECK_THROWS_AS(vllm::LoadHfConfig(nested.path()), std::runtime_error);
  }

  SUBCASE("the key is read from the RESOLVED text config, not the wrapper") {
    TempJson f(NestedGdnJson(kSigmoid));
    CHECK(vllm::LoadHfConfig(f.path()).output_gate_type == "sigmoid");

    TempJson swish(NestedGdnJson(kSwish));
    CHECK(vllm::LoadHfConfig(swish.path()).output_gate_type == "silu");

    // The nested fixtures put "not_a_gate" on the WRAPPER: reading the top-level
    // doc instead of the resolved text config would throw here.
    TempJson absent(NestedGdnJson(""));
    CHECK(vllm::LoadHfConfig(absent.path()).output_gate_type == "silu");
  }

  SUBCASE("a nested unrecognized value is still refused") {
    TempJson f(NestedGdnJson(kBogus));
    CHECK_THROWS_AS(vllm::LoadHfConfig(f.path()), std::runtime_error);
  }
}

// ─── generation_config.json SAMPLING keys (#1985) ───────────────────────────
// Upstream reads the same file for the same reason and then narrows it in
// ModelConfig.get_diff_sampling_param (vllm/config/model.py). We read
// eos_token_id out of it already; these cases cover the six sampling keys the
// narrowing consumes, and the polarity that makes the defaulting rule work:
// UNSET means the checkpoint said nothing, which is not the same as the
// checkpoint saying the neutral value.
TEST_CASE("LoadHfConfig reads the sibling generation_config.json sampling keys") {
  constexpr const char* kConfig = R"({
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 8,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "vocab_size": 32,
    "max_position_embeddings": 128
  })";

  SUBCASE("the Qwen3.8-27B file, verbatim") {
    // Read live 2026-08-26 from
    // https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/generation_config.json
    TempModelDir model(kConfig, R"({
      "bos_token_id": 248044,
      "do_sample": true,
      "eos_token_id": [248046, 248044],
      "pad_token_id": 248044,
      "temperature": 1.0,
      "top_k": 20,
      "top_p": 0.95
    })");
    const vllm::HfConfig cfg = vllm::LoadHfConfig(model.config_path());
    const auto& g = cfg.generation_config_sampling;
    REQUIRE(g.temperature.has_value());
    CHECK(*g.temperature == doctest::Approx(1.0));
    REQUIRE(g.top_k.has_value());
    CHECK(*g.top_k == 20);
    REQUIRE(g.top_p.has_value());
    CHECK(*g.top_p == doctest::Approx(0.95));
    // Declared by no Qwen3.8-27B file, so they stay UNSET rather than becoming
    // the neutral value. This is the distinction the defaulting rule turns on.
    CHECK_FALSE(g.min_p.has_value());
    CHECK_FALSE(g.repetition_penalty.has_value());
    CHECK_FALSE(g.max_new_tokens.has_value());
    // The eos read is unaffected by the new keys.
    std::vector<int32_t> eos = {248044, 248046};
    CHECK(cfg.generation_config_eos_ids == eos);
  }

  SUBCASE("all six keys") {
    TempModelDir model(kConfig, R"({
      "repetition_penalty": 1.05,
      "temperature": 0.7,
      "top_k": 40,
      "top_p": 0.8,
      "min_p": 0.05,
      "max_new_tokens": 512
    })");
    const auto g = vllm::LoadHfConfig(model.config_path()).generation_config_sampling;
    CHECK(*g.repetition_penalty == doctest::Approx(1.05));
    CHECK(*g.temperature == doctest::Approx(0.7));
    CHECK(*g.top_k == 40);
    CHECK(*g.top_p == doctest::Approx(0.8));
    CHECK(*g.min_p == doctest::Approx(0.05));
    CHECK(*g.max_new_tokens == 512);
  }

  SUBCASE("absent, malformed, null and wrong-typed all leave every key UNSET") {
    // try_get_generation_config returns {} on every failure path and never
    // raises, so none of these may throw and none may invent a value.
    for (const char* body : {"", "{ this is not json", "[1,2,3]",
                             R"({"temperature": null, "top_k": null})",
                             R"({"temperature": "hot", "top_k": [20]})"}) {
      CAPTURE(body);
      TempModelDir model(kConfig, body);
      vllm::HfConfig cfg;
      REQUIRE_NOTHROW(cfg = vllm::LoadHfConfig(model.config_path()));
      const auto& g = cfg.generation_config_sampling;
      CHECK_FALSE(g.temperature.has_value());
      CHECK_FALSE(g.top_k.has_value());
      CHECK_FALSE(g.top_p.has_value());
      CHECK_FALSE(g.min_p.has_value());
      CHECK_FALSE(g.repetition_penalty.has_value());
      CHECK_FALSE(g.max_new_tokens.has_value());
    }
  }

  SUBCASE("ParseHfConfig has no sibling to read, so it stays UNSET") {
    // The in-memory entry point must not pick up whatever generation_config.json
    // happens to sit in the working directory -- the same reason
    // generation_config_eos_ids stays empty there.
    const auto doc = nlohmann::json::parse(kConfig);
    const auto g = vllm::ParseHfConfig(doc, "<memory>").generation_config_sampling;
    CHECK_FALSE(g.temperature.has_value());
    CHECK_FALSE(g.top_k.has_value());
  }
}
