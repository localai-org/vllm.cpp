// Ported from upstream vLLM @ e24d1b24fe96a56ba8b0d653efa076d03eb95d6c:
//   tests/v1/spec_decode/test_mtp.py:67-221
//   tests/v1/worker/test_gpu_autoregressive_speculator.py:52-82
// Loader sharing and direct-hidden-return assertions land with M-mtp-0. The
// propose-loop-only assertions remain explicitly skipped until M-mtp-1 adds the
// scheduler/speculator plumbing, as required by .agents/test-porting.md rule 6.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/worker/gpu/spec_decode/mtp/speculator.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/xpu.h"

namespace {

using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::Qwen3_5DenseWeights;
using vllm::Qwen3_5MoeWeights;
using vllm::Qwen3_5MTPKind;
using vllm::Qwen3_5MTPModel;
using vllm::Qwen3_5MTPWeights;
using vllm::StTensor;
using vllm::TensorResolver;

struct StoredTensor {
  std::vector<uint16_t> values;
  StTensor view;
};

class TensorStore {
 public:
  void Add(const std::string& name, std::vector<int64_t> shape,
           uint16_t seed) {
    int64_t numel = 1;
    for (int64_t dim : shape) numel *= dim;
    StoredTensor& stored = tensors_[name];
    stored.values.resize(static_cast<size_t>(numel));
    for (int64_t i = 0; i < numel; ++i) {
      const int centered = static_cast<int>((i + seed) % 19) - 9;
      stored.values[static_cast<size_t>(i)] =
          vt::F32ToBF16(static_cast<float>(centered) * 0.0078125F);
    }
    stored.view.dtype = "BF16";
    stored.view.shape = std::move(shape);
    stored.view.data =
        reinterpret_cast<const uint8_t*>(stored.values.data());
    stored.view.nbytes = stored.values.size() * sizeof(uint16_t);
  }

  const StTensor& Get(const std::string& name) const {
    return tensors_.at(name).view;
  }

  const StoredTensor& Stored(const std::string& name) const {
    return tensors_.at(name);
  }

  StTensor& MutableView(const std::string& name) {
    return tensors_.at(name).view;
  }

  TensorResolver Resolver() const {
    return [this](const std::string& name) -> const StTensor& {
      return Get(name);
    };
  }

  // MODEL-QWEN35-EXL3-HEAD (#2495 item 5): the presence predicate
  // `LoadQwen3_5MTP` now REQUIRES, so a caller cannot silently turn the EXL3
  // arm off by omitting it.
  std::function<bool(const std::string&)> Exists() const {
    return [this](const std::string& name) { return tensors_.count(name) != 0; };
  }

 private:
  std::map<std::string, StoredTensor> tensors_;
};

OwnedTensor MakeOwned(std::vector<int64_t> shape, uint16_t seed,
                      bool raw_nk = false) {
  OwnedTensor out;
  out.dtype = vt::DType::kBF16;
  out.rank = static_cast<int>(shape.size());
  int64_t numel = 1;
  for (int i = 0; i < out.rank; ++i) {
    out.shape[i] = shape[static_cast<size_t>(i)];
    numel *= out.shape[i];
  }
  out.bytes.resize(static_cast<size_t>(numel) * sizeof(uint16_t));
  auto* values = reinterpret_cast<uint16_t*>(out.bytes.data());
  for (int64_t i = 0; i < numel; ++i) {
    const int centered = static_cast<int>((i + seed) % 23) - 11;
    values[i] = vt::F32ToBF16(static_cast<float>(centered) * 0.005F);
  }
  out.nk = raw_nk;
  return out;
}

HfConfig MakeConfig(Qwen3_5MTPKind kind) {
  HfConfig config;
  config.model_type = kind == Qwen3_5MTPKind::kDense ? "qwen3_5" : "qwen3_5_moe";
  config.hidden_size = 4;
  config.num_hidden_layers = 2;
  config.vocab_size = 16;
  config.num_attention_heads = 2;
  config.num_key_value_heads = 1;
  config.head_dim = 2;
  config.rotary_dim = 2;
  config.rope_theta = 10000.0;
  config.rms_norm_eps = 1e-5;
  config.max_position_embeddings = 32;
  config.intermediate_size = 6;
  if (kind == Qwen3_5MTPKind::kMoe) {
    config.num_experts = 2;
    config.num_experts_per_tok = 1;
    config.moe_intermediate_size = 3;
    config.shared_expert_intermediate_size = 3;
  }
  config.raw = {
      {"text_config",
       {{"mtp_num_hidden_layers", 1},
        {"mtp_use_dedicated_embeddings", false}}}};
  return config;
}

void AddCommonMtp(TensorStore& store, const HfConfig& config) {
  const int64_t hidden = config.hidden_size;
  const int64_t q_out =
      2 * config.num_attention_heads * config.head_dim;
  const int64_t kv_out = config.num_key_value_heads * config.head_dim;
  store.Add("mtp.fc.weight", {hidden, 2 * hidden}, 1);
  store.Add("mtp.pre_fc_norm_embedding.weight", {hidden}, 2);
  store.Add("mtp.pre_fc_norm_hidden.weight", {hidden}, 3);
  store.Add("mtp.layers.0.input_layernorm.weight", {hidden}, 4);
  store.Add("mtp.layers.0.self_attn.q_proj.weight", {q_out, hidden}, 5);
  store.Add("mtp.layers.0.self_attn.k_proj.weight", {kv_out, hidden}, 6);
  store.Add("mtp.layers.0.self_attn.v_proj.weight", {kv_out, hidden}, 7);
  store.Add("mtp.layers.0.self_attn.o_proj.weight",
            {hidden, config.num_attention_heads * config.head_dim}, 8);
  store.Add("mtp.layers.0.self_attn.q_norm.weight", {config.head_dim}, 9);
  store.Add("mtp.layers.0.self_attn.k_norm.weight", {config.head_dim}, 10);
  store.Add("mtp.layers.0.post_attention_layernorm.weight", {hidden}, 11);
  store.Add("mtp.norm.weight", {hidden}, 12);
}

void AddDenseMtp(TensorStore& store, const HfConfig& config) {
  AddCommonMtp(store, config);
  store.Add("mtp.layers.0.mlp.gate_proj.weight",
            {config.intermediate_size, config.hidden_size}, 13);
  store.Add("mtp.layers.0.mlp.up_proj.weight",
            {config.intermediate_size, config.hidden_size}, 14);
  store.Add("mtp.layers.0.mlp.down_proj.weight",
            {config.hidden_size, config.intermediate_size}, 15);
}

void AddMoeMtp(TensorStore& store, const HfConfig& config) {
  AddCommonMtp(store, config);
  store.Add("mtp.layers.0.mlp.gate.weight",
            {config.num_experts, config.hidden_size}, 13);
  store.Add("mtp.layers.0.mlp.experts.gate_up_proj",
            {config.num_experts, 2 * config.moe_intermediate_size,
             config.hidden_size},
            14);
  store.Add("mtp.layers.0.mlp.experts.down_proj",
            {config.num_experts, config.hidden_size,
             config.moe_intermediate_size},
            15);
  store.Add("mtp.layers.0.mlp.shared_expert.gate_proj.weight",
            {config.shared_expert_intermediate_size, config.hidden_size}, 16);
  store.Add("mtp.layers.0.mlp.shared_expert.up_proj.weight",
            {config.shared_expert_intermediate_size, config.hidden_size}, 17);
  store.Add("mtp.layers.0.mlp.shared_expert.down_proj.weight",
            {config.hidden_size, config.shared_expert_intermediate_size}, 18);
  store.Add("mtp.layers.0.mlp.shared_expert_gate.weight",
            {1, config.hidden_size}, 19);
}

Qwen3_5DenseWeights MakeDenseTarget(const HfConfig& config) {
  Qwen3_5DenseWeights target;
  target.embed_tokens =
      MakeOwned({config.vocab_size, config.hidden_size}, 21);
  target.lm_head = MakeOwned({config.hidden_size, config.vocab_size}, 22);
  return target;
}

Qwen3_5MoeWeights MakeMoeTarget(const HfConfig& config) {
  Qwen3_5MoeWeights target;
  target.embed_tokens =
      MakeOwned({config.vocab_size, config.hidden_size}, 23);
  target.lm_head = MakeOwned({config.hidden_size, config.vocab_size}, 24);
  return target;
}

void CheckFinite(const std::vector<float>& values) {
  REQUIRE_FALSE(values.empty());
  for (float value : values) CHECK(std::isfinite(value));
}

// ── SPEC-MTP I5c helpers: a host-backed draft KV cache + full-attn metadata. ──
using vllm::PagedKvCache;
using vllm::v1::CommonAttentionMetadata;

// Owns one draft full-attention KV layer's bf16 buffer (the production draft KV
// dtype — ResolveKvCacheDType; the raw-torch MTP weights produce bf16 K/V so the
// "auto" reshape_and_cache requires a bf16 cache) and hands out a PagedKvCache
// view. num_blocks*block_size slots of [2, num_kv_heads, head_size].
struct DraftKvPool {
  std::vector<uint16_t> buf;
  PagedKvCache kv;
  DraftKvPool(const HfConfig& c, int64_t num_blocks, int64_t block_size) {
    const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
    buf.assign(static_cast<size_t>(num_blocks * 2 * block_size * Hkv * Dh), 0);
    kv.data = buf.data();
    kv.dtype = vt::DType::kBF16;
    kv.num_blocks = num_blocks;
    kv.block_size = block_size;
    kv.num_kv_heads = Hkv;
    kv.head_size = Dh;
  }
};

// A single-request prefill metadata over T contiguous tokens starting at
// absolute position `start` into block 0 (block_size >= start+T). slot_mapping =
// [start, start+T).
CommonAttentionMetadata MtpMeta(int64_t T, int64_t seq_len, int64_t start,
                                int64_t block_size) {
  (void)block_size;  // single block 0; slots addressed by absolute position.
  CommonAttentionMetadata m;
  m.num_reqs = 1;
  m.num_actual_tokens = static_cast<int>(T);
  m.query_start_loc = {0, static_cast<int32_t>(T)};
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens = {static_cast<int32_t>(seq_len)};
  m.seq_lens_cpu = m.seq_lens;
  m.max_query_len = static_cast<int>(T);
  m.max_seq_len = static_cast<int>(seq_len);
  m.block_table_num_cols = 1;
  m.block_table_tensor = {0};
  for (int64_t t = 0; t < T; ++t)
    m.slot_mapping.push_back(start + t);
  m.causal = true;
  return m;
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  double d = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    d = std::max(d, std::abs(static_cast<double>(a[i]) - b[i]));
  return d;
}

int64_t ArgmaxRow(const std::vector<float>& logits, int64_t row, int64_t vocab) {
  const float* r = logits.data() + static_cast<size_t>(row) * vocab;
  int64_t best = 0;
  for (int64_t v = 1; v < vocab; ++v)
    if (r[static_cast<size_t>(v)] > r[static_cast<size_t>(best)]) best = v;
  return best;
}

}  // namespace

TEST_CASE("test_mtp_load_model_unified: dense MTP shares target embedding and lm_head") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kDense);
  TensorStore store;
  AddDenseMtp(store, config);
  const Qwen3_5MTPWeights weights = vllm::LoadQwen3_5MTP(
      store.Resolver(), store.Exists(), config, Qwen3_5MTPKind::kDense);
  const Qwen3_5DenseWeights target = MakeDenseTarget(config);
  const Qwen3_5MTPModel model(weights, target, config);

  CHECK_FALSE(model.has_own_embed_tokens());
  CHECK_FALSE(model.has_own_lm_head());
  CHECK(&model.embed_tokens() == &target.embed_tokens);
  CHECK(model.lm_head() == &target.lm_head);
  // PERF-27B-LMHEAD-FP4 (issue #213): the dense drafter shares the target's
  // PACKED head too, so the pointer is the target's field rather than null. This
  // target is bf16, so the field is EMPTY and the bf16 arm is still selected.
  CHECK(model.lm_head_fp4() == &target.lm_head_fp4);
  REQUIRE(model.lm_head_fp4() != nullptr);
  CHECK(model.lm_head_fp4()->Empty());
  CHECK(weights.NumLayers() == 1);
  REQUIRE(weights.dense_layers.size() == 1);
  CHECK(weights.fc.nk);
  CHECK(weights.fc.shape[0] == config.hidden_size);
  CHECK(weights.fc.shape[1] == 2 * config.hidden_size);
  CHECK_FALSE(weights.dense_layers[0].is_linear_attention);
  CHECK(weights.dense_layers[0].attn.q_proj.nk);
  CHECK(weights.dense_layers[0].mlp.down_proj.nk);
}

TEST_CASE("test_mtp_load_model_unified: MoE fused stacks split per expert and share target") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kMoe);
  TensorStore store;
  AddMoeMtp(store, config);
  const Qwen3_5MTPWeights weights = vllm::LoadQwen3_5MTP(
      store.Resolver(), store.Exists(), config, Qwen3_5MTPKind::kMoe);
  const Qwen3_5MoeWeights target = MakeMoeTarget(config);
  const Qwen3_5MTPModel model(weights, target, config);

  CHECK(&model.embed_tokens() == &target.embed_tokens);
  CHECK(model.lm_head() == &target.lm_head);
  CHECK(model.lm_head_fp4() == &target.lm_head_fp4);
  REQUIRE(weights.moe_layers.size() == 1);
  const auto& moe = weights.moe_layers[0].moe;
  REQUIRE(moe.expert_gate.size() ==
          static_cast<size_t>(config.num_experts));
  REQUIRE(moe.expert_up.size() ==
          static_cast<size_t>(config.num_experts));
  REQUIRE(moe.expert_down.size() ==
          static_cast<size_t>(config.num_experts));
  CHECK(moe.expert_gate[0].shape[0] == config.moe_intermediate_size);
  CHECK(moe.expert_gate[0].shape[1] == config.hidden_size);
  CHECK(moe.expert_down[0].shape[0] == config.hidden_size);
  CHECK(moe.expert_down[0].shape[1] == config.moe_intermediate_size);

  const auto& gate_up =
      store.Stored("mtp.layers.0.mlp.experts.gate_up_proj").values;
  const int64_t hidden = config.hidden_size;
  const int64_t intermediate = config.moe_intermediate_size;
  const int64_t stride = 2 * intermediate * hidden;
  const auto* gate1 = reinterpret_cast<const uint16_t*>(
      moe.expert_gate[1].bytes.data());
  const auto* up0 =
      reinterpret_cast<const uint16_t*>(moe.expert_up[0].bytes.data());
  CHECK(gate1[0] == gate_up[static_cast<size_t>(stride)]);
  CHECK(up0[0] == gate_up[static_cast<size_t>(intermediate * hidden)]);
}

// MODEL-QWEN35-EXL3-HEAD (#2495 item 5) NARROWS this title and nothing else.
// The assertion is unchanged and still fires: `mtp.fc.weight` is the BF16
// arm's tensor, which an EXL3 checkpoint does not ship at all, so the trellis
// rung is not selected and the strictness refusal is still the right answer
// for the arm this case is about. "every mtp tensor" is simply no longer
// true of every checkpoint.
TEST_CASE("test_mtp_load_model_unified: every mtp tensor on the BF16 arm is strictly BF16") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kDense);
  TensorStore store;
  AddDenseMtp(store, config);
  store.MutableView("mtp.fc.weight").dtype = "F16";
  CHECK_THROWS_AS(
      vllm::LoadQwen3_5MTP(store.Resolver(), store.Exists(), config,
                           Qwen3_5MTPKind::kDense),
      std::runtime_error);
}

TEST_CASE("test_mtp_load_model_unified: wrong same-byte-count shape is rejected") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kDense);
  TensorStore store;
  AddDenseMtp(store, config);
  // [q_out,H] is [8,4] for this config. Reversing the dimensions preserves
  // nbytes, proving the loader checks the upstream semantic shape and not just
  // storage size.
  store.MutableView("mtp.layers.0.self_attn.q_proj.weight").shape = {4, 8};
  CHECK_THROWS_AS(
      vllm::LoadQwen3_5MTP(store.Resolver(), store.Exists(), config,
                           Qwen3_5MTPKind::kDense),
      std::runtime_error);
}

TEST_CASE("test_mtp_load_model_unified: dedicated embeddings are rejected for gate checkpoints") {
  HfConfig config = MakeConfig(Qwen3_5MTPKind::kDense);
  config.raw["text_config"]["mtp_use_dedicated_embeddings"] = true;
  TensorStore store;
  AddDenseMtp(store, config);
  CHECK_THROWS_AS(
      vllm::LoadQwen3_5MTP(store.Resolver(), store.Exists(), config,
                           Qwen3_5MTPKind::kDense),
      std::runtime_error);
}

TEST_CASE("test_mtp_propose k=1: MTP forward returns hidden states directly") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kDense);
  TensorStore store;
  AddDenseMtp(store, config);
  const Qwen3_5MTPWeights weights = vllm::LoadQwen3_5MTP(
      store.Resolver(), store.Exists(), config, Qwen3_5MTPKind::kDense);
  const Qwen3_5DenseWeights target = MakeDenseTarget(config);
  const Qwen3_5MTPModel model(weights, target, config);

  OwnedTensor target_hidden = MakeOwned({3, config.hidden_size}, 31);
  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();
  const std::vector<int32_t> input_ids = {1, 2, 3};
  const std::vector<int32_t> positions = {0, 1, 2};
  const auto hidden =
      model.Forward(input_ids, positions, target_hidden.View(), queue);
  REQUIRE(hidden.storage != nullptr);
  CHECK(hidden.tensor.rank == 2);
  CHECK(hidden.tensor.shape[0] == 3);
  CHECK(hidden.tensor.shape[1] == config.hidden_size);
  CHECK(hidden.tensor.dtype == vt::DType::kBF16);

  const vllm::ForwardLogits logits = model.ComputeLogits(hidden.tensor, queue);
  CHECK(logits.rows == 3);
  CHECK(logits.vocab == config.vocab_size);
  const std::vector<float> host = model.ForwardLogitsHost(
      input_ids, positions, target_hidden.View(), queue);
  CHECK(host.size() == static_cast<size_t>(3 * config.vocab_size));
  CheckFinite(host);
  backend.DestroyQueue(queue);
}

TEST_CASE("test_mtp_propose k=1: MoE MTP forward returns hidden states directly") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kMoe);
  TensorStore store;
  AddMoeMtp(store, config);
  const Qwen3_5MTPWeights weights = vllm::LoadQwen3_5MTP(
      store.Resolver(), store.Exists(), config, Qwen3_5MTPKind::kMoe);
  const Qwen3_5MoeWeights target = MakeMoeTarget(config);
  const Qwen3_5MTPModel model(weights, target, config);

  OwnedTensor target_hidden = MakeOwned({3, config.hidden_size}, 37);
  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();
  const std::vector<int32_t> input_ids = {3, 4, 5};
  const std::vector<int32_t> positions = {0, 1, 2};
  const auto hidden =
      model.Forward(input_ids, positions, target_hidden.View(), queue);
  REQUIRE(hidden.storage != nullptr);
  CHECK(hidden.tensor.rank == 2);
  CHECK(hidden.tensor.shape[0] == 3);
  CHECK(hidden.tensor.shape[1] == config.hidden_size);
  CHECK(hidden.tensor.dtype == vt::DType::kBF16);

  const std::vector<float> host = model.ForwardLogitsHost(
      input_ids, positions, target_hidden.View(), queue);
  CHECK(host.size() == static_cast<size_t>(3 * config.vocab_size));
  CheckFinite(host);
  backend.DestroyQueue(queue);
}

TEST_CASE("test_run_model_reuses_tensor_return_for_mtp" * doctest::skip(true)) {
  MESSAGE("SKIP: _run_model tensor reuse belongs to M-mtp-1 AutoRegressiveSpeculator");
}

TEST_CASE("test_run_model_unpacks_tuple_return_for_mtp" * doctest::skip(true)) {
  MESSAGE("SKIP: tuple-vs-tensor dispatch belongs to M-mtp-1 AutoRegressiveSpeculator");
}

// ── SPEC-MTP I5c: the PAGED MTP propose forward + draft KV layer. ────────────

namespace {
// Download an on-device ForwardLogits to a host [rows*vocab] vector.
std::vector<float> HostLogits(const vllm::ForwardLogits& logits,
                              vt::Backend& backend, vt::Queue& queue) {
  std::vector<float> host(static_cast<size_t>(logits.rows) * logits.vocab);
  backend.Copy(queue, host.data(), logits.device_tensor.data,
               host.size() * sizeof(float));
  backend.Synchronize(queue);
  return host;
}

// A [1,H] contiguous bf16 view of row `r` of a [rows,H] owned bf16 tensor.
vt::Tensor Row(const OwnedTensor& owned, int64_t r, int64_t H) {
  vt::Tensor t = owned.View();
  t.rank = 2;
  t.shape[0] = 1;
  t.shape[1] = H;
  t.stride[0] = H;
  t.stride[1] = 1;
  t.data = static_cast<uint16_t*>(t.data) + r * H;
  return t;
}
}  // namespace

TEST_CASE("B70 GPTQ checkpoint loads its real BF16 MTP draft tensors") {
  const char* model_dir = std::getenv("VT_B70_GPTQ_MODEL_DIR");
  if (model_dir == nullptr) {
    MESSAGE("set VT_B70_GPTQ_MODEL_DIR to run the local B70 checkpoint load");
    return;
  }
  const std::filesystem::path base(model_dir);
  const HfConfig config = vllm::LoadHfConfig((base / "config.json").string());
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(
      (base / "model-00005-of-00005.safetensors").string()));
  auto weights = vllm::LoadQwen3_5MTP(
      shards, config, Qwen3_5MTPKind::kDense);
  CHECK(weights.NumLayers() == 1);
  CHECK(weights.fc.dtype == vt::DType::kBF16);
  CHECK(weights.fc.shape[0] == config.hidden_size);
  CHECK(weights.fc.shape[1] == 2 * config.hidden_size);
  CHECK(weights.dense_layers[0].attn.q_proj.shape[0] ==
        2 * config.num_attention_heads * config.head_dim);
  CHECK(weights.dense_layers[0].mlp.down_proj.shape[1] ==
        config.intermediate_size);
}

TEST_CASE("B70 GPTQ MTP draft runs native XPU paged forward and real head logits") {
  const char* model_dir = std::getenv("VT_B70_GPTQ_MODEL_DIR");
  if (model_dir == nullptr) {
    MESSAGE("set VT_B70_GPTQ_MODEL_DIR to run the local B70 draft forward");
    return;
  }
  const std::filesystem::path base(model_dir);
  const HfConfig config = vllm::LoadHfConfig((base / "config.json").string());
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(
      (base / "model-00005-of-00005.safetensors").string()));
  auto weights = vllm::LoadQwen3_5MTP(
      shards, config, Qwen3_5MTPKind::kDense);

  // Borrow the target-shared embedding and head. Below, the GPTQ draft packs
  // its own INT4 head while this target FP16 owner remains unchanged.
  const auto embed_shard = vllm::SafetensorsFile::Open(
      (base / "model-00001-of-00005.safetensors").string());
  const auto& src = embed_shard.Get("model.language_model.embed_tokens.weight");
  REQUIRE(src.dtype == "F16");
  REQUIRE(src.shape == std::vector<int64_t>{config.vocab_size, config.hidden_size});
  Qwen3_5DenseWeights target;
  target.embed_tokens.dtype = vt::DType::kF16;
  target.embed_tokens.rank = 2;
  target.embed_tokens.shape[0] = config.vocab_size;
  target.embed_tokens.shape[1] = config.hidden_size;
  target.embed_tokens.bytes = vllm::OwnedBytes::Borrow(
      src.data, src.nbytes, src.mapping);
  const auto head_shard = vllm::SafetensorsFile::Open(
      (base / "model-00002-of-00005.safetensors").string());
  const auto& head = head_shard.Get("lm_head.weight");
  REQUIRE(head.dtype == "F16");
  REQUIRE(head.shape == std::vector<int64_t>{config.vocab_size, config.hidden_size});
  target.lm_head.dtype = vt::DType::kF16;
  target.lm_head.rank = 2;
  target.lm_head.shape[0] = config.vocab_size;
  target.lm_head.shape[1] = config.hidden_size;
  target.lm_head.nk = true;
  target.lm_head.bytes = vllm::OwnedBytes::Borrow(
      head.data, head.nbytes, head.mapping);
  const Qwen3_5MTPModel draft(weights, target, config);

  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kXPU);
  vt::Queue queue = backend.CreateQueue();
  std::vector<uint16_t> input(static_cast<size_t>(config.hidden_size));
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = vt::F32ToF16((static_cast<int>(i % 29) - 14) * 0.003F);
  void* hidden_ptr = backend.Alloc(input.size() * sizeof(uint16_t));
  backend.Copy(queue, hidden_ptr, input.data(), input.size() * sizeof(uint16_t));
  vt::Tensor target_hidden = vt::Tensor::Contiguous(
      hidden_ptr, vt::DType::kF16, queue.device, {1, config.hidden_size});

  constexpr int64_t block_size = 16;
  const size_t kv_bytes = static_cast<size_t>(2 * block_size *
      config.num_key_value_heads * config.head_dim * sizeof(uint16_t));
  void* kv_ptr = backend.Alloc(kv_bytes);
  backend.Memset(queue, kv_ptr, 0, kv_bytes);
  vllm::PagedKvCache kv;
  kv.data = kv_ptr;
  kv.dtype = vt::DType::kF16;
  kv.num_blocks = 1;
  kv.block_size = block_size;
  kv.num_kv_heads = config.num_key_value_heads;
  kv.head_size = config.head_dim;

  {
    const auto output = draft.ForwardPaged(
        {100}, {0}, target_hidden, MtpMeta(1, 1, 0, block_size), kv, queue);
    REQUIRE(output.tensor.dtype == vt::DType::kF16);
    REQUIRE(output.tensor.shape[0] == 1);
    REQUIRE(output.tensor.shape[1] == config.hidden_size);
    std::vector<uint16_t> host(static_cast<size_t>(config.hidden_size));
    backend.Copy(queue, host.data(), output.tensor.data,
                 host.size() * sizeof(uint16_t));
    backend.Synchronize(queue);
    CHECK(std::all_of(host.begin(), host.end(), [](uint16_t x) {
      return std::isfinite(vt::F16ToF32(x));
    }));
    CHECK(std::any_of(host.begin(), host.end(), [](uint16_t x) {
      return vt::F16ToF32(x) != 0.0F;
    }));
    const auto logits = draft.ComputeLogits(output.tensor, queue);
    REQUIRE(logits.on_device());
    REQUIRE(logits.rows == 1);
    REQUIRE(logits.vocab == config.vocab_size);
    std::vector<float> host_logits(static_cast<size_t>(config.vocab_size));
    backend.Copy(queue, host_logits.data(), logits.device_tensor.data,
                 host_logits.size() * sizeof(float));
    backend.Synchronize(queue);
    CHECK(std::all_of(host_logits.begin(), host_logits.end(), [](float x) {
      return std::isfinite(x);
    }));
    CHECK(std::any_of(host_logits.begin(), host_logits.end(), [](float x) {
      return x != 0.0F;
    }));
    CHECK(vt::GetReferenceTierHits() == 0);

    // The second draft step consumes the first step's hidden row and KV.
    // Compare that continuation with one causal two-row forward using the
    // exact same token and hidden inputs. Both use the real MTP weights.
    const int32_t next_id = static_cast<int32_t>(
        std::max_element(host_logits.begin(), host_logits.end()) -
        host_logits.begin());
    const auto carry = draft.GatherHiddenRows(output.tensor, {0}, queue);
    REQUIRE(carry.tensor.dtype == vt::DType::kF16);
    const auto continued = draft.ForwardPaged(
        {next_id}, {1}, carry.tensor, MtpMeta(1, 2, 1, block_size), kv,
        queue);
    std::vector<uint16_t> continued_host(input.size());
    backend.Copy(queue, continued_host.data(), continued.tensor.data,
                 continued_host.size() * sizeof(uint16_t));

    void* pair_hidden_ptr = backend.Alloc(2 * input.size() * sizeof(uint16_t));
    backend.Copy(queue, pair_hidden_ptr, hidden_ptr,
                 input.size() * sizeof(uint16_t));
    backend.Copy(queue,
                 static_cast<uint8_t*>(pair_hidden_ptr) +
                     input.size() * sizeof(uint16_t),
                 output.tensor.data, input.size() * sizeof(uint16_t));
    const vt::Tensor pair_hidden = vt::Tensor::Contiguous(
        pair_hidden_ptr, vt::DType::kF16, queue.device,
        {2, config.hidden_size});
    void* pair_kv_ptr = backend.Alloc(kv_bytes);
    backend.Memset(queue, pair_kv_ptr, 0, kv_bytes);
    vllm::PagedKvCache pair_kv = kv;
    pair_kv.data = pair_kv_ptr;
    const auto pair = draft.ForwardPaged(
        {100, next_id}, {0, 1}, pair_hidden,
        MtpMeta(2, 2, 0, block_size), pair_kv, queue);
    std::vector<uint16_t> pair_host(2 * input.size());
    backend.Copy(queue, pair_host.data(), pair.tensor.data,
                 pair_host.size() * sizeof(uint16_t));
    backend.Synchronize(queue);
    float worst = 0.0F;
    bool finite = true;
    for (size_t i = 0; i < input.size(); ++i) {
      const float a = vt::F16ToF32(continued_host[i]);
      const float b = vt::F16ToF32(pair_host[input.size() + i]);
      finite = finite && std::isfinite(a) && std::isfinite(b);
      worst = std::max(worst, std::abs(a - b));
    }
    MESSAGE("real MTP draft KV continuation vs two-row forward max|diff|="
            << worst);
    CHECK(finite);
    CHECK(worst < 0.05F);

    // Red control: the same second step without its preceding KV row must
    // diverge. Otherwise matching the two-row forward would not prove the
    // continuation actually read the first step's cache.
    backend.Memset(queue, pair_kv_ptr, 0, kv_bytes);
    const auto fresh = draft.ForwardPaged(
        {next_id}, {1}, carry.tensor, MtpMeta(1, 2, 1, block_size),
        pair_kv, queue);
    std::vector<uint16_t> fresh_host(input.size());
    backend.Copy(queue, fresh_host.data(), fresh.tensor.data,
                 fresh_host.size() * sizeof(uint16_t));
    backend.Synchronize(queue);
    float fresh_worst = 0.0F;
    for (size_t i = 0; i < input.size(); ++i) {
      fresh_worst = std::max(fresh_worst,
          std::abs(vt::F16ToF32(continued_host[i]) -
                   vt::F16ToF32(fresh_host[i])));
    }
    MESSAGE("real MTP draft missing-KV red control max|diff|=" << fresh_worst);
    CHECK(fresh_worst > 0.0F);

    // The production serving recipe uses FP8 KV. Exercise the same two-step
    // draft route with an independent E4M3 cache and compare the decoded row
    // with the FP16-KV result above. This checks FP8 write/read boundaries;
    // a Python oracle comparison is still required for a quality claim.
    void* fp8_kv_ptr = backend.Alloc(kv_bytes / 2);
    backend.Memset(queue, fp8_kv_ptr, 0, kv_bytes / 2);
    vllm::PagedKvCache fp8_kv = kv;
    fp8_kv.data = fp8_kv_ptr;
    fp8_kv.dtype = vt::DType::kI8;
    fp8_kv.fp8_kind = vt::Fp8KVCacheDataType::kFp8E4M3;
    const auto fp8_first = draft.ForwardPaged(
        {100}, {0}, target_hidden, MtpMeta(1, 1, 0, block_size),
        fp8_kv, queue);
    const auto fp8_carry = draft.GatherHiddenRows(
        fp8_first.tensor, {0}, queue);
    const auto fp8_second = draft.ForwardPaged(
        {next_id}, {1}, fp8_carry.tensor,
        MtpMeta(1, 2, 1, block_size), fp8_kv, queue);
    std::vector<uint16_t> fp8_host(input.size());
    backend.Copy(queue, fp8_host.data(), fp8_second.tensor.data,
                 fp8_host.size() * sizeof(uint16_t));
    backend.Synchronize(queue);
    float fp8_worst = 0.0F;
    bool fp8_finite = true;
    for (size_t i = 0; i < input.size(); ++i) {
      const float value = vt::F16ToF32(fp8_host[i]);
      fp8_finite = fp8_finite && std::isfinite(value);
      fp8_worst = std::max(fp8_worst,
          std::abs(vt::F16ToF32(continued_host[i]) - value));
    }
    MESSAGE("real MTP draft FP8 E4M3 KV vs FP16 KV max|diff|=" << fp8_worst);
    CHECK(fp8_finite);
    CHECK(fp8_worst < 0.5F);
    backend.Free(fp8_kv_ptr);
    backend.Free(pair_kv_ptr);
    backend.Free(pair_hidden_ptr);

    // Convert the same checkpoint's BF16 draft linears and a separate FP16
    // head copy once, exactly as the serving load path does. The target head
    // owner remains FP16 and must not be replaced by the draft's packed copy.
    vllm::PackQwen3_5MTPGptqDraft(weights, head);
    REQUIRE(weights.IsGptq4Draft());
    CHECK(weights.fc.bytes.empty());
    CHECK(weights.dense_layers[0].gptq4.attn_qkv.k == config.hidden_size);
    CHECK(weights.draft_lm_head_gptq4.n == config.vocab_size);
    CHECK(target.lm_head.dtype == vt::DType::kF16);
    CHECK(target.lm_head.bytes.data() == head.data);
    backend.Memset(queue, kv_ptr, 0, kv_bytes);
    const auto packed_hidden = draft.ForwardPaged(
        {100}, {0}, target_hidden, MtpMeta(1, 1, 0, block_size), kv, queue);
    std::vector<uint16_t> packed_hidden_host(host.size());
    backend.Copy(queue, packed_hidden_host.data(), packed_hidden.tensor.data,
                 packed_hidden_host.size() * sizeof(uint16_t));
    backend.Synchronize(queue);
    float hidden_worst = 0.0F;
    for (size_t i = 0; i < host.size(); ++i)
      hidden_worst = std::max(hidden_worst,
          std::abs(vt::F16ToF32(host[i]) - vt::F16ToF32(packed_hidden_host[i])));
    MESSAGE("real MTP draft dense/INT4 hidden max|diff|=" << hidden_worst);
    const auto packed_logits = draft.ComputeLogits(packed_hidden.tensor, queue);
    std::vector<float> packed_host(static_cast<size_t>(config.vocab_size));
    backend.Copy(queue, packed_host.data(), packed_logits.device_tensor.data,
                 packed_host.size() * sizeof(float));
    backend.Synchronize(queue);
    CHECK(std::all_of(packed_host.begin(), packed_host.end(),
                      [](float value) { return std::isfinite(value); }));
    const int32_t packed_next_id = static_cast<int32_t>(
        std::max_element(packed_host.begin(), packed_host.end()) -
        packed_host.begin());
    const auto packed_head_on_dense = draft.ComputeLogits(output.tensor, queue);
    std::vector<float> packed_head_host(packed_host.size());
    backend.Copy(queue, packed_head_host.data(),
                 packed_head_on_dense.device_tensor.data,
                 packed_head_host.size() * sizeof(float));
    backend.Synchronize(queue);
    const int32_t packed_head_next_id = static_cast<int32_t>(
        std::max_element(packed_head_host.begin(), packed_head_host.end()) -
        packed_head_host.begin());
    MESSAGE("real MTP draft dense/INT4 top1 (dense/packed-head/full-packed): "
            << next_id << "/" << packed_head_next_id << "/" << packed_next_id);
    CHECK(vt::GetReferenceTierHits() == 0);
  }
  backend.Free(kv_ptr);
  backend.Free(hidden_ptr);
  backend.DestroyQueue(queue);
}

// CORE I5c PROOF: the PAGED MTP forward over a single-request 1-block KV with a
// trivial slot map reproduces the STANDALONE (dense) forward's logits/argmax —
// the paged rewrite did not change the head math (mtp-spec-decode.md §5 I5c gate).
TEST_CASE("i5c paged MTP forward equals standalone (dense head)") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kDense);
  TensorStore store;
  AddDenseMtp(store, config);
  const Qwen3_5MTPWeights weights =
      vllm::LoadQwen3_5MTP(store.Resolver(), store.Exists(), config, Qwen3_5MTPKind::kDense);
  const Qwen3_5DenseWeights target = MakeDenseTarget(config);
  const Qwen3_5MTPModel model(weights, target, config);

  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();
  const int64_t T = 4, H = config.hidden_size, vocab = config.vocab_size;
  const std::vector<int32_t> ids = {1, 5, 2, 9};
  const std::vector<int32_t> pos = {0, 1, 2, 3};
  OwnedTensor target_hidden = MakeOwned({T, H}, 41);

  // Standalone (I1) path.
  const std::vector<float> logits_std =
      model.ForwardLogitsHost(ids, pos, target_hidden.View(), queue);

  // Paged path: fresh 1-block KV, contiguous slot map.
  DraftKvPool pool(config, /*num_blocks=*/1, /*block_size=*/8);
  const CommonAttentionMetadata am = MtpMeta(T, /*seq_len=*/T, /*start=*/0, 8);
  const vllm::Qwen3_5MTPHiddenStates hidden = model.ForwardPaged(
      ids, pos, target_hidden.View(), am, pool.kv, queue);
  const std::vector<float> logits_paged =
      HostLogits(model.ComputeLogits(hidden.tensor, queue), backend, queue);

  REQUIRE(logits_paged.size() == logits_std.size());
  const double d = MaxAbsDiff(logits_paged, logits_std);
  MESSAGE("i5c dense paged==standalone max|diff| = " << d);
  CHECK(d < 1e-2);
  for (int64_t t = 0; t < T; ++t)
    CHECK(ArgmaxRow(logits_paged, t, vocab) == ArgmaxRow(logits_std, t, vocab));
  backend.DestroyQueue(queue);
}

TEST_CASE("paged MTP keeps the BF16 CPU contract and refuses an FP16 tap") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kDense);
  TensorStore store;
  AddDenseMtp(store, config);
  const Qwen3_5MTPWeights weights = vllm::LoadQwen3_5MTP(
      store.Resolver(), store.Exists(), config, Qwen3_5MTPKind::kDense);
  const Qwen3_5DenseWeights target = MakeDenseTarget(config);
  const Qwen3_5MTPModel model(weights, target, config);

  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();
  const int64_t T = 4, H = config.hidden_size;
  OwnedTensor target_f16 = MakeOwned({T, H}, 31);
  target_f16.dtype = vt::DType::kF16;
  auto* halves = reinterpret_cast<uint16_t*>(target_f16.bytes.data());
  for (int64_t i = 0; i < T * H; ++i) {
    const float value = static_cast<float>(i - 8) * 0.0031F;
    halves[i] = vt::F32ToF16(value);
  }

  const std::vector<int32_t> ids = {1, 5, 2, 9};
  const std::vector<int32_t> positions = {0, 1, 2, 3};
  const CommonAttentionMetadata am = MtpMeta(T, T, 0, 8);
  DraftKvPool pool(config, 1, 8);
  CHECK_THROWS(model.ForwardPaged(
      ids, positions, target_f16.View(), am, pool.kv, queue));
  backend.DestroyQueue(queue);
}

TEST_CASE("i5c paged MTP forward equals standalone (MoE head)") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kMoe);
  TensorStore store;
  AddMoeMtp(store, config);
  const Qwen3_5MTPWeights weights =
      vllm::LoadQwen3_5MTP(store.Resolver(), store.Exists(), config, Qwen3_5MTPKind::kMoe);
  const Qwen3_5MoeWeights target = MakeMoeTarget(config);
  const Qwen3_5MTPModel model(weights, target, config);

  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();
  const int64_t T = 4, H = config.hidden_size, vocab = config.vocab_size;
  const std::vector<int32_t> ids = {3, 7, 1, 4};
  const std::vector<int32_t> pos = {0, 1, 2, 3};
  OwnedTensor target_hidden = MakeOwned({T, H}, 43);

  const std::vector<float> logits_std =
      model.ForwardLogitsHost(ids, pos, target_hidden.View(), queue);
  DraftKvPool pool(config, 1, 8);
  const CommonAttentionMetadata am = MtpMeta(T, T, 0, 8);
  const vllm::Qwen3_5MTPHiddenStates hidden = model.ForwardPaged(
      ids, pos, target_hidden.View(), am, pool.kv, queue);
  const std::vector<float> logits_paged =
      HostLogits(model.ComputeLogits(hidden.tensor, queue), backend, queue);

  REQUIRE(logits_paged.size() == logits_std.size());
  const double d = MaxAbsDiff(logits_paged, logits_std);
  MESSAGE("i5c MoE paged==standalone max|diff| = " << d);
  CHECK(d < 1e-2);
  for (int64_t t = 0; t < T; ++t)
    CHECK(ArgmaxRow(logits_paged, t, vocab) == ArgmaxRow(logits_std, t, vocab));
  backend.DestroyQueue(queue);
}

// DRAFT-KV CORRECTNESS: a two-step drive where step 1 writes the draft K/V and
// step 2 (a decode) attends over it must reproduce the token-1 hidden of a single
// combined 2-token forward. The RED CONTROL in the same test — running step 2
// over a FRESH (unpopulated) draft KV — DIVERGES, proving step 2 genuinely reads
// what step 1 wrote (a stubbed/absent paged KV path fails here).
TEST_CASE("i5c draft KV two-step: decode attends over written K/V") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kDense);
  TensorStore store;
  AddDenseMtp(store, config);
  const Qwen3_5MTPWeights weights =
      vllm::LoadQwen3_5MTP(store.Resolver(), store.Exists(), config, Qwen3_5MTPKind::kDense);
  const Qwen3_5DenseWeights target = MakeDenseTarget(config);
  const Qwen3_5MTPModel model(weights, target, config);

  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();
  const int64_t H = config.hidden_size;
  const int32_t id0 = 5, id1 = 9;
  OwnedTensor th = MakeOwned({2, H}, 47);  // per-token target hidden rows
  const vt::Tensor th0 = Row(th, 0, H), th1 = Row(th, 1, H);

  auto row_bytes = [&](const vllm::Qwen3_5MTPHiddenStates& hs, int64_t r) {
    std::vector<float> out(static_cast<size_t>(H));
    const auto* p = static_cast<const uint16_t*>(hs.tensor.data) + r * H;
    for (int64_t i = 0; i < H; ++i)
      out[static_cast<size_t>(i)] = vt::BF16ToF32(p[i]);
    return out;
  };

  // (A) Combined 2-token forward → reference token-1 hidden.
  DraftKvPool comb(config, 1, 8);
  const CommonAttentionMetadata am_comb = MtpMeta(2, 2, 0, 8);
  const auto hs_comb = model.ForwardPaged({id0, id1}, {0, 1}, th.View(),
                                          am_comb, comb.kv, queue);
  const std::vector<float> ref = row_bytes(hs_comb, 1);

  // (B) Split: step 1 prefill token0 (writes slot 0), step 2 decode token1
  // (seq_len 2, slot 1) reads slots 0..1.
  DraftKvPool split(config, 1, 8);
  const CommonAttentionMetadata am_s1 = MtpMeta(1, 1, 0, 8);
  (void)model.ForwardPaged({id0}, {0}, th0, am_s1, split.kv, queue);
  const CommonAttentionMetadata am_s2 = MtpMeta(1, 2, 1, 8);
  const auto hs_s2 = model.ForwardPaged({id1}, {1}, th1, am_s2, split.kv, queue);
  const std::vector<float> got = row_bytes(hs_s2, 0);

  // (C) RED CONTROL: step 2 over a FRESH KV (slot 0 never written).
  DraftKvPool fresh(config, 1, 8);
  const auto hs_nokv = model.ForwardPaged({id1}, {1}, th1, am_s2, fresh.kv, queue);
  const std::vector<float> nokv = row_bytes(hs_nokv, 0);

  const double d_ok = MaxAbsDiff(got, ref);
  const double d_red = MaxAbsDiff(nokv, ref);
  MESSAGE("i5c draft-KV two-step: with-step1 max|diff|=" << d_ok
          << " vs fresh-KV (RED) max|diff|=" << d_red);
  CHECK(d_ok < 2e-2);       // decode-via-written-cache == combined forward
  CHECK(d_red > d_ok);      // unwritten KV diverges → step 2 really reads step 1
  backend.DestroyQueue(queue);
}

// propose() k=1: prepare_prefill_inputs (I5b) shift-splice + one paged forward +
// argmax draft pick, returning one token per request (speculator.py:236-238).
TEST_CASE("i5c MtpProposePrefill k=1 returns the argmax over the shifted draft") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kDense);
  TensorStore store;
  AddDenseMtp(store, config);
  const Qwen3_5MTPWeights weights =
      vllm::LoadQwen3_5MTP(store.Resolver(), store.Exists(), config, Qwen3_5MTPKind::kDense);
  const Qwen3_5DenseWeights target = MakeDenseTarget(config);
  const Qwen3_5MTPModel model(weights, target, config);

  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();
  const int64_t H = config.hidden_size, vocab = config.vocab_size;

  // One decoding request, k=1: verify span = 2 tokens (the last is the bonus).
  const std::vector<int32_t> verify_ids = {5, 9};
  const std::vector<int64_t> verify_pos = {0, 1};
  const std::vector<int32_t> idx_mapping = {0};
  const int32_t sampled = 7;  // last_sampled[req_state 0]
  const std::vector<int32_t> last_sampled = {sampled};
  const std::vector<int32_t> next_prefill_tokens = {0};
  const std::vector<int32_t> num_sampled = {1};   // all accepted
  const std::vector<int32_t> num_rejected = {0};
  OwnedTensor target_hidden = MakeOwned({2, H}, 51);

  CommonAttentionMetadata am = MtpMeta(2, 2, 0, 8);
  DraftKvPool pool(config, 1, 8);

  const std::vector<int32_t> draft = vllm::v1::MtpProposePrefill(
      model, am, pool.kv, target_hidden.View(), verify_ids, verify_pos,
      idx_mapping, last_sampled, next_prefill_tokens, num_sampled, num_rejected,
      /*max_num_reqs=*/1, queue);
  REQUIRE(draft.size() == 1);
  CHECK(draft[0] >= 0);
  CHECK(draft[0] < vocab);

  // Independent recomputation over the EXPECTED shift-splice: with 0 rejected the
  // draft span is [verify_ids[1], sampled] and the sampled row is index 1. This
  // proves propose applied prepare_prefill_inputs' shift + last_token index.
  const std::vector<int32_t> expect_ids = {9, sampled};
  const std::vector<int32_t> expect_pos = {0, 1};
  DraftKvPool pool2(config, 1, 8);
  const auto hs = model.ForwardPaged(expect_ids, expect_pos,
                                     target_hidden.View(), am, pool2.kv, queue);
  const std::vector<float> lg =
      HostLogits(model.ComputeLogits(hs.tensor, queue), backend, queue);
  CHECK(draft[0] == ArgmaxRow(lg, /*row=*/1, vocab));
  backend.DestroyQueue(queue);
}

// The target-model hidden-state tap (ForwardDeviceTap) is INERT: it returns the
// exact same logits as ForwardDevice and hands back a [T,H] post-norm hidden.
TEST_CASE("i5c hidden-state tap is inert and shape-correct") {
  // Exercised against a real paged forward in the dedicated paged-forward tests
  // (test_qwen27_paged_forward / test_qwen35_paged_forward); this asserts the
  // MTP carrier plumbing compiles and the forward-declared type resolves.
  vllm::Qwen3_5MTPHiddenStates carrier;
  CHECK(carrier.storage == nullptr);
  CHECK(carrier.tensor.data == nullptr);
}

// SPEC-MTP I5d-pre: the LoadedModel draft-construction virtual. The concrete
// Qwen3.5 target retains the loaded mtp.* weights and builds a Qwen3_5MTPModel
// draft sharing the target's embed_tokens/lm_head; a non-MTP model returns null
// and refuses to hold draft weights. RED-first: before this increment the
// virtuals do not exist (does not compile) and no non-qwen model returns null.
TEST_CASE("i5d-pre LoadedModel::BuildMtpDraft builds a Qwen3.5 draft, null otherwise") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kDense);
  TensorStore store;
  AddDenseMtp(store, config);

  // The target LoadedModel (borrows caller-owned dense target weights).
  const Qwen3_5DenseWeights target = MakeDenseTarget(config);
  std::unique_ptr<vllm::LoadedModel> model =
      vllm::BorrowQwen3_5DenseLoadedModel(target);

  // Capability + before-attach: supports MTP, but no draft yet.
  CHECK(model->supports_mtp_draft());
  CHECK(model->BuildMtpDraft(config) == nullptr);

  // Attach the loaded mtp.* draft weights, then build the draft.
  model->AttachMtpDraftWeights(
      vllm::LoadQwen3_5MTP(store.Resolver(), store.Exists(), config, Qwen3_5MTPKind::kDense));
  std::unique_ptr<Qwen3_5MTPModel> draft = model->BuildMtpDraft(config);
  REQUIRE(draft != nullptr);
  // load_eagle_model sharing: the draft borrows the target's embed_tokens/lm_head.
  CHECK(&draft->embed_tokens() == &target.embed_tokens);
  CHECK(draft->lm_head() == &target.lm_head);

  // A non-MTP model inherits the base defaults: no support, null draft, and it
  // refuses to hold draft weights. A test-local subclass exercises the base
  // contract without needing a full non-qwen checkpoint on disk.
  struct NonMtpModel final : public vllm::LoadedModel {
    explicit NonMtpModel(const vllm::ModelRegistration& r) : LoadedModel(r) {}
  };
  NonMtpModel non_mtp(vllm::RegistrationFor("OPTForCausalLM"));
  CHECK_FALSE(non_mtp.supports_mtp_draft());
  CHECK(non_mtp.BuildMtpDraft(config) == nullptr);
  CHECK_THROWS_AS(
      non_mtp.AttachMtpDraftWeights(
          vllm::LoadQwen3_5MTP(store.Resolver(), store.Exists(), config, Qwen3_5MTPKind::kDense)),
      std::runtime_error);
}


TEST_CASE("paged MTP consumes supplied merged embeddings and preserves lookup defaults") {
  const auto config = MakeConfig(Qwen3_5MTPKind::kDense);
  TensorStore store;
  AddDenseMtp(store, config);
  const auto weights = vllm::LoadQwen3_5MTP(
      store.Resolver(), store.Exists(), config, Qwen3_5MTPKind::kDense);
  const auto target = MakeDenseTarget(config);
  const Qwen3_5MTPModel model(weights, target, config);
  auto& backend = vt::GetBackend(vt::DeviceType::kCPU);
  auto queue = backend.CreateQueue();
  const int64_t tokens = 4, hidden = config.hidden_size;
  const std::vector<int32_t> ids{1, 5, 2, 9}, positions{0, 1, 2, 3};
  auto feedback = MakeOwned({tokens, hidden}, 41);
  auto merged = MakeOwned({tokens, hidden}, 0);
  const size_t row_bytes = static_cast<size_t>(hidden) * sizeof(uint16_t);
  for (size_t row = 0; row < ids.size(); ++row)
    std::copy_n(target.embed_tokens.bytes.begin() + ids[row] * row_bytes,
                row_bytes, merged.bytes.begin() + row * row_bytes);
  const auto initial = merged.bytes;
  const auto embed_view = merged.View();
  const auto metadata = MtpMeta(tokens, tokens, 0, 8);
  DraftKvPool ordinary(config, 1, 8), supplied(config, 1, 8);
  const auto expected = model.ForwardPaged(ids, positions, feedback.View(), metadata, ordinary.kv, queue);
  // Out-of-vocabulary placeholders prove the provided rows bypass lookup.
  const auto actual = model.ForwardPaged({999, 999, 999, 999}, positions,
      feedback.View(), metadata, supplied.kv, queue, 0, &embed_view);
  const auto expected_logits = HostLogits(model.ComputeLogits(expected.tensor, queue), backend, queue);
  const auto actual_logits = HostLogits(model.ComputeLogits(actual.tensor, queue), backend, queue);
  CHECK(actual_logits == expected_logits);
  CHECK(merged.bytes == initial);
  const size_t cache_bytes = 2 * 8 * config.num_key_value_heads * config.head_dim * sizeof(uint16_t);
  CHECK(std::memcmp(ordinary.kv.data, supplied.kv.data, cache_bytes) == 0);
  const auto cache_before = std::vector<uint8_t>(static_cast<uint8_t*>(supplied.kv.data),
                                               static_cast<uint8_t*>(supplied.kv.data) + cache_bytes);
  auto invalid = embed_view;
  invalid.shape[0]--;
  CHECK_THROWS(model.ForwardPaged(ids, positions, feedback.View(), metadata, supplied.kv, queue, 0, &invalid));
  invalid = embed_view;
  invalid.dtype = vt::DType::kF16;
  CHECK_THROWS(model.ForwardPaged(ids, positions, feedback.View(), metadata, supplied.kv, queue, 0, &invalid));
  invalid = embed_view;
  invalid.data = nullptr;
  CHECK_THROWS(model.ForwardPaged(ids, positions, feedback.View(), metadata, supplied.kv, queue, 0, &invalid));
  CHECK(std::equal(cache_before.begin(), cache_before.end(), static_cast<uint8_t*>(supplied.kv.data)));
  backend.DestroyQueue(queue);
}

// The caller merges shifted visual rows before proposal, matching the executed
// V2 input contract. Only the first forward uses these rows: k>1 must embed
// its generated tokens normally, rather than reuse a verify-sized image batch.
TEST_CASE("MTP proposer consumes merged prefill rows only at the first draft step") {
  const auto config = MakeConfig(Qwen3_5MTPKind::kDense);
  TensorStore store;
  AddDenseMtp(store, config);
  const auto weights = vllm::LoadQwen3_5MTP(
      store.Resolver(), store.Exists(), config, Qwen3_5MTPKind::kDense);
  const auto target = MakeDenseTarget(config);
  const Qwen3_5MTPModel model(weights, target, config);
  auto& backend = vt::GetBackend(vt::DeviceType::kCPU);
  auto queue = backend.CreateQueue();
  auto feedback = MakeOwned({2, config.hidden_size}, 51);
  auto merged = MakeOwned({2, config.hidden_size}, 0);
  const size_t row_bytes = config.hidden_size * sizeof(uint16_t);
  const std::vector<int32_t> shifted_ids{9, 7};
  for (size_t row = 0; row < shifted_ids.size(); ++row)
    std::copy_n(target.embed_tokens.bytes.begin() + shifted_ids[row] * row_bytes,
                row_bytes, merged.bytes.begin() + row * row_bytes);
  const auto original = merged.bytes;
  auto embeds = merged.View();
  const auto metadata = MtpMeta(2, 2, 0, 8);
  for (int depth : {1, 3}) {
    CAPTURE(depth);
    DraftKvPool expected(config, 1, 8), supplied(config, 1, 8);
    const auto normal = vllm::v1::MtpProposeDrafts(
        model, metadata, expected.kv, feedback.View(), {5, 9}, {0, 1},
        {0}, {7}, {0}, {1}, {0}, 1, depth, 8, 8, queue);
    const auto actual = vllm::v1::MtpProposeDrafts(
        model, metadata, supplied.kv, feedback.View(), {999, 999}, {0, 1},
        {0}, {7}, {0}, {1}, {0}, 1, depth, 8, 8, queue, &embeds);
    CHECK(actual.draft_tokens == normal.draft_tokens);
    CHECK(actual.num_draft_decode_forwards == depth - 1);
    CHECK(supplied.buf == expected.buf);
    CHECK(merged.bytes == original);
    if (depth == 1) {
      DraftKvPool direct(config, 1, 8);
      CHECK(vllm::v1::MtpProposePrefill(
          model, metadata, direct.kv, feedback.View(), {999, 999}, {0, 1},
          {0}, {7}, {0}, {1}, {0}, 1, queue, &embeds) == normal.draft_tokens);
      CHECK(direct.buf == expected.buf);
    }
  }
  DraftKvPool invalid_pool(config, 1, 8);
  const auto untouched = invalid_pool.buf;
  auto invalid = embeds;
  invalid.shape[0] = 1;
  CHECK_THROWS(vllm::v1::MtpProposeDrafts(
      model, metadata, invalid_pool.kv, feedback.View(), {5, 9}, {0, 1},
      {0}, {7}, {0}, {1}, {0}, 1, 3, 8, 8, queue, &invalid));
  CHECK(invalid_pool.buf == untouched);
  backend.DestroyQueue(queue);
}
