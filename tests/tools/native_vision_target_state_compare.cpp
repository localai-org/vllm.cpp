// Bounded full native target replay for the existing alpha MTP capture.
// No scheduler, drafter, sampler, new precision, or serving dispatch.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string_view>
#include <vector>
#include <nlohmann/json.hpp>
#include "vllm/config/multimodal.h"
#include "vllm/model_executor/models/act_dump.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_dense_mm.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/dense_device_glue.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/v1/worker/gpu/prepare_inputs.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vt/op_provider.h"
#include "vt/unaligned.h"
namespace {
using json = nlohmann::json;
using Bytes = std::vector<unsigned char>;
using Snapshot = std::map<std::string, Bytes>;
using vllm::dense_attn::DBuf;
using vllm::dense_attn::Dev;
void Require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
std::string Hash(const Bytes& bytes) {
  const auto raw = vllm::v1::sha256_bytes(std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (unsigned char x : raw) { result += hex[x >> 4]; result += hex[x & 15]; }
  return result;
}
Bytes Read(const std::filesystem::path& path, size_t bound) {
  const auto size = std::filesystem::file_size(path);
  Require(size > 0 && size <= bound, "input byte bound exceeded");
  std::ifstream file(path, std::ios::binary);
  Bytes result{std::istreambuf_iterator<char>(file), {}};
  Require(result.size() == size, "incomplete bounded input");
  return result;
}
json ReadJson(const std::filesystem::path& path) { return json::parse(Read(path, 1024 * 1024)); }
struct Capture {
  json metadata;
  std::map<std::string, json> entries;
  Snapshot blobs;
  explicit Capture(const std::filesystem::path& root, const char* name) : metadata(ReadJson(root / name)) {
    size_t total = 0;
    for (const auto& entry : metadata.at("blobs")) {
      const auto key = entry.at("name").get<std::string>(), file = entry.at("file").get<std::string>();
      Require(std::filesystem::path(file).filename() == file && !entries.count(key), "unsafe/duplicated capture blob");
      const auto raw = Read(root / file, 4 * 1024 * 1024);
      total += raw.size();
      Require(total <= 256 * 1024 * 1024 && raw.size() == entry.at("bytes") &&
              Hash(raw) == entry.at("sha256").get<std::string>(), "capture payload hash/bound differs");
      entries.emplace(key, entry); blobs.emplace(key, raw);
    }
  }
  const Bytes& State(const std::string& name, int dtype, const json& shape) const {
    const auto& entry = entries.at(name);
    Require(entry.at("dtype") == dtype && entry.at("shape") == shape, "captured state layout differs");
    return blobs.at(name);
  }
};
Bytes Window(const Bytes& raw, int width, int offset) {
  Require(width >= 3 && offset >= 0 && offset + 3 <= width && raw.size() == size_t(10240 * width * 2),
          "invalid Conv canonical window");
  Bytes result(10240 * 3 * 2);
  for (int ch = 0; ch < 10240; ++ch)
    std::memcpy(result.data() + ch * 6, raw.data() + (ch * width + offset) * 2, 6);
  return result;
}
struct QueueOwner {
  std::optional<vt::Queue> value;
  ~QueueOwner() { if (value) vt::DestroyQueue(*value); }
};
struct Caches {
  Dev d;
  bool spec;
  std::vector<DBuf> owners;
  std::vector<Bytes> init_conv;  // Keep all H2D inputs through the initialization drain.
  std::vector<vllm::GdnStateCache> gdn;
  std::vector<vllm::PagedKvCache> kv;
  const std::vector<int32_t> ordinary_slots{2, 0, 5, 7};
  const std::vector<int32_t> pages{1, 0, 3, 2};
  const int selected_page = 2;
  const int selected_slot = 3;
  int ordinary_row = 0;
  Caches(Dev device, const Capture& initial, bool speculative, int initial_context = 18,
         int ordinary_rows = 4, int selected_ordinary_row = 0, bool poison_unused = false)
      : d(device), spec(speculative), ordinary_row(selected_ordinary_row) {
    Require(initial_context > 0 && initial_context < 1600 && ordinary_rows >= 1 && ordinary_rows <= 4 &&
            ordinary_row >= 0 && ordinary_row < ordinary_rows, "invalid selected initial cache geometry");
    owners.reserve(112); init_conv.reserve(48);
    const int width = spec ? 6 : 3;
    const auto live = spec ? std::vector<int32_t>{selected_slot}
                           : std::vector<int32_t>(ordinary_slots.begin(), ordinary_slots.begin() + ordinary_rows);
    for (int layer = 0; layer < 48; ++layer) {
      const auto base = "gdn" + std::to_string(layer);
      const int initial_width = initial.entries.at(base + "-conv").at("shape").at(1);
      Require(initial_width == 3 || initial_width == 6, "invalid initial Conv width");
      const auto narrow = Window(initial.State(base + "-conv", 1, json::array({10240, initial_width})), initial_width, 0);
      init_conv.emplace_back(8 * 10240 * width * 2, poison_unused ? 0xa5 : 0);
      auto& conv = init_conv.back();
      for (int slot : live) for (int ch = 0; ch < 10240; ++ch)
        std::memcpy(conv.data() + (slot * 10240 + ch) * width * 2, narrow.data() + ch * 6, 6);
      owners.emplace_back(d, vt::DType::kF16, std::vector<int64_t>{8, 10240, width}, conv.data());
      vllm::GdnStateCache state; state.conv_state = owners.back().t();
      owners.emplace_back(d, vt::DType::kF32, std::vector<int64_t>{8, 48, 128, 128});
      state.ssm_state = owners.back().t();
      d.b.Memset(d.q, state.ssm_state.data, poison_unused ? 0xa5 : 0, state.ssm_state.Bytes());
      const auto& ssm = initial.State(base + "-ssm", 0, json::array({48, 128, 128}));
      for (int slot : live)
        d.b.Copy(d.q, static_cast<char*>(state.ssm_state.data) + slot * ssm.size(), ssm.data(), ssm.size());
      gdn.push_back(state);
    }
    for (int layer = 0; layer < 16; ++layer) {
      const auto base = "attn" + std::to_string(layer);
      const auto& entry = initial.entries.at(base + "-k");
      Require(entry.at("shape") == json::array({initial_context, 4, 256}) && entry.at("dtype") == int(vt::DType::kI8),
              "actual FP8 KV geometry differs");
      owners.emplace_back(d, vt::DType::kI8, std::vector<int64_t>{4, 2, 1600, 1024});
      vllm::PagedKvCache cache;
      cache.data = owners.back().t().data; cache.dtype = vt::DType::kI8;
      cache.num_blocks = 4; cache.block_size = 1600; cache.num_kv_heads = 4; cache.head_size = 256;
      cache.page_size_bytes = 2 * 1600 * 1024;
      cache.fp8_kind = static_cast<vt::Fp8KVCacheDataType>(entry.at("fp8_kind").get<int>());
      cache.k_scale = entry.at("k_scale"); cache.v_scale = entry.at("v_scale");
      d.b.Memset(d.q, cache.data, poison_unused ? 0xa5 : 0, cache.num_blocks * cache.page_size_bytes);
      const auto active_pages = spec ? std::vector<int32_t>{selected_page}
                                    : std::vector<int32_t>(pages.begin(), pages.begin() + ordinary_rows);
      for (int page : active_pages) for (int which = 0; which < 2; ++which) {
        const auto key = base + (which ? "-v" : "-k");
        const auto& raw = initial.State(key, int(vt::DType::kI8), json::array({initial_context, 4, 256}));
        const auto& value = initial.entries.at(key);
        Require(value.at("fp8_kind") == entry.at("fp8_kind") && value.at("k_scale") == entry.at("k_scale") &&
                value.at("v_scale") == entry.at("v_scale"), "K/V precision contract differs");
        d.b.Copy(d.q, static_cast<char*>(cache.data) + (2 * page + which) * 1600 * 1024, raw.data(), raw.size());
      }
      kv.push_back(cache);
    }
    d.b.Synchronize(d.q);  // Diagnostic initialization: inputs remain live until here.
  }
  Bytes Download(const void* data, size_t size) {
    Bytes result(size); d.b.Copy(d.q, result.data(), data, size); d.b.Synchronize(d.q); return result;
  }
  void CopyFrom(const Caches& from) {
    Require(spec == from.spec && owners.size() == from.owners.size(), "different cache clone layout");
    for (size_t i = 0; i < owners.size(); ++i) {
      Require(owners[i].t().Bytes() == from.owners[i].t().Bytes(), "different cache clone extent");
      d.b.Copy(d.q, owners[i].t().data, from.owners[i].t().data, owners[i].t().Bytes());
    }
  }
  Snapshot Observe(int token = 0, int first_context = 19) {
    Snapshot result;
    const auto copy = [&](const std::string& name, const void* data, size_t bytes) {
      auto& output = result[name]; output.resize(bytes);
      d.b.Copy(d.q, output.data(), data, bytes);
    };
    const int slot = spec ? std::vector<int32_t>{3, 1, 6, 4}.at(token) : ordinary_slots[ordinary_row];
    const int page = spec ? selected_page : pages[ordinary_row];
    for (size_t layer = 0; layer < gdn.size(); ++layer) {
      const auto& state = gdn[layer];
      const auto base = "gdn" + std::to_string(layer);
      const auto conv_bytes = state.conv_state.Bytes() / 8, ssm_bytes = state.ssm_state.Bytes() / 8;
      const int conv_slot = spec ? selected_slot : slot;
      copy(base + "-conv", static_cast<char*>(state.conv_state.data) + conv_slot * conv_bytes, conv_bytes);
      copy(base + "-ssm", static_cast<char*>(state.ssm_state.data) + slot * ssm_bytes, ssm_bytes);
    }
    for (size_t layer = 0; layer < kv.size(); ++layer) for (int which = 0; which < 2; ++which)
      copy("attn" + std::to_string(layer) + (which ? "-v" : "-k"),
           static_cast<char*>(kv[layer].data) + (2 * page + which) * 1600 * 1024, (first_context + token) * 1024);
    d.b.Synchronize(d.q);  // One observation drain; every destination remains owned above.
    for (int layer = 0; layer < 48; ++layer) {
      auto& raw = result.at("gdn" + std::to_string(layer) + "-conv");
      raw = Window(raw, spec ? 6 : 3, spec ? token : 0);
    }
    return result;
  }
};
json Compare(const Snapshot& want, const Snapshot& got) {
  Require(want.size() == 128 && got.size() == 128, "incomplete full target snapshot");
  json changed = json::array();
  for (const auto& [name, raw] : want) {
    Require(raw.size() == got.at(name).size(), "full target state byte shape differs");
    if (raw != got.at(name)) changed.push_back(name);
  }
  return {{"compared_states", 128}, {"different_states", changed}, {"all_storage_exact", changed.empty()}};
}
json LogitMetrics(const Bytes& a, const Bytes& b) {
  Require(a.size() == 248320 * 4 && b.size() == a.size(), "wrong full logit row");
  size_t changed = 0, coordinate = 0;
  double square = 0, norm = 0, maximum = 0;
  int first = 0, second = 1;
  const auto value = [&](int i) { return vt::LoadUnaligned<float>(a.data() + i * 4); };
  if (value(second) > value(first)) std::swap(first, second);
  for (int i = 0; i < 248320; ++i) {
    const double x = value(i), y = vt::LoadUnaligned<float>(b.data() + i * 4);
    Require(std::isfinite(x) && std::isfinite(y), "nonfinite target logits");
    changed += std::memcmp(a.data() + i * 4, b.data() + i * 4, 4) != 0;
    const double delta = x - y; square += delta * delta; norm += x * x;
    if (std::abs(delta) > maximum) { maximum = std::abs(delta); coordinate = i; }
    if (i == first || i == second) continue;
    if (x > value(first)) {second = first; first = i;}
    else if (x > value(second)) second = i;
  }
  const auto candidate_value = [&](int i) { return vt::LoadUnaligned<float>(b.data() + i * 4); };
  int candidate_first = 0, candidate_second = 1;
  if (candidate_value(candidate_second) > candidate_value(candidate_first)) std::swap(candidate_first, candidate_second);
  for (int i = 2; i < 248320; ++i) {
    if (candidate_value(i) > candidate_value(candidate_first)) { candidate_second = candidate_first; candidate_first = i; }
    else if (candidate_value(i) > candidate_value(candidate_second)) candidate_second = i;
  }
  return {{"elements", 248320}, {"different_elements", changed}, {"storage_exact", a == b},
          {"max_abs", maximum}, {"max_coordinate", coordinate}, {"rel_l2", std::sqrt(square / norm)},
          {"reference_top2_ids", {first, second}}, {"reference_margin", value(first) - value(second)},
          {"candidate_top2_ids", {candidate_first, candidate_second}},
          {"candidate_margin", candidate_value(candidate_first) - candidate_value(candidate_second)}};
}
Bytes Forward(vllm::Qwen3_5DenseWeights& weights, const vllm::HfConfig& config, Caches& caches,
              const std::vector<int32_t>& tokens, int position = 18, int previous_accepted_tokens = 1) {
  const bool spec = caches.spec;
  vllm::v1::StepInputs step;
  step.input_token_ids = tokens;
  step.positions = spec ? std::vector<int64_t>{position, position + 1, position + 2, position + 3}
                        : std::vector<int64_t>(4, position);
  step.query_start_loc = spec ? std::vector<int32_t>{0, 4} : std::vector<int32_t>{0, 1, 2, 3, 4};
  step.seq_lens = spec ? std::vector<int32_t>{position + 4} : std::vector<int32_t>(4, position + 1);
  std::vector<int64_t> writes;
  for (int i = 0; i < 4; ++i) writes.push_back((spec ? caches.selected_page : caches.pages[i]) * 1600 + step.positions[i]);
  step.slot_mapping = {writes};
  auto am = vllm::v1::MakeCommonAttentionMetadata(step,
      spec ? std::vector<int32_t>{caches.selected_page} : caches.pages, 1, true, 0);
  vllm::v1::GDNAttentionMetadata gm; gm.num_actual_tokens = 4;
  if (spec) {
    gm.num_spec_decodes = 1; gm.num_spec_decode_tokens = 4; gm.spec_state_indices_num_cols = 4;
    gm.spec_state_indices_tensor = std::vector<int32_t>{3, 1, 6, 4};
    gm.spec_query_start_loc = std::vector<int32_t>{0, 4}; gm.spec_sequence_masks = std::vector<uint8_t>{1};
    Require(previous_accepted_tokens >= 1 && previous_accepted_tokens <= 4, "invalid previous accepted length");
    gm.spec_token_indx = std::vector<int32_t>{0, 1, 2, 3};
    gm.num_accepted_tokens = std::vector<int32_t>{previous_accepted_tokens};
  } else {
    gm.num_decodes = 4; gm.num_decode_tokens = 4; gm.non_spec_state_indices_tensor = caches.ordinary_slots;
  }
  std::vector<int32_t> positions(step.positions.begin(), step.positions.end());
  auto output = vllm::Qwen3_5DenseModel::ForwardDevice(tokens, positions, am, gm, caches.kv, caches.gdn, weights, config, caches.d.q);
  Require(output.on_device() && output.device_tensor.dtype == vt::DType::kF32 && output.rows == 4 &&
          output.vocab == 248320 && output.device_tensor.IsContiguous(), "unexpected full target output");
  return caches.Download(output.device_tensor.data, 4 * 248320 * 4);
}
Bytes ForwardOrdinaryLayout(vllm::Qwen3_5DenseWeights& weights, const vllm::HfConfig& config,
                            Caches& caches, const json& layout, bool frozen_alpha = true) {
  const int rows = layout.at("num_reqs"), selected = layout.at("row");
  Require(rows >= 1 && rows <= 4 && selected == caches.ordinary_row && !caches.spec &&
          layout.at("actual_token_rows") == rows && layout.at("own_image_features") == 0 &&
          (layout.value("image_decode",false) || layout.at("mrope_delta") == 0) && layout.at("max_query_len") == 1 && !layout.contains("mtp"),
          "wrong ordinary selected-row layout");
  vllm::v1::StepInputs step;
  step.input_token_ids = layout.at("input_token_ids").get<std::vector<int32_t>>();
  step.positions = layout.at("positions").get<std::vector<int64_t>>();
  step.query_start_loc = layout.at("query_start_loc").get<std::vector<int32_t>>();
  step.seq_lens = layout.at("seq_lens").get<std::vector<int32_t>>();
  Require(step.input_token_ids.size() == size_t(rows) && step.positions.size() == size_t(rows) &&
          step.seq_lens.size() == size_t(rows) && step.query_start_loc.size() == size_t(rows + 1),
          "incomplete ordinary input arrays");
  std::vector<int64_t> writes;
  const int columns = layout.at("kv_block_table_cols");
  Require(columns >= 1 && columns <= 164, "unsupported actual KV table width");
  std::vector<int32_t> table(rows * columns, 0);
  for (int i = 0; i < rows; ++i) {
    Require(step.query_start_loc[i] == i && step.positions[i] >= 0 && step.positions[i] < 1600 &&
            step.seq_lens[i] == step.positions[i] + 1, "not an ordinary one-query decode");
    table[i * columns] = caches.pages[i];
    writes.push_back(caches.pages[i] * 1600 + step.positions[i]);
  }
  Require(step.query_start_loc.back() == rows, "wrong ordinary query extent");
  if (frozen_alpha) Require(step.positions[selected] == 38 &&
          step.input_token_ids[selected] == 2702, "wrong frozen final common query");
  step.slot_mapping = {writes};
  auto am = vllm::v1::MakeCommonAttentionMetadata(step, table, columns, true, 0);
  vllm::v1::GDNAttentionMetadata gm;
  gm.num_actual_tokens = rows; gm.num_decodes = rows; gm.num_decode_tokens = rows;
  gm.non_spec_state_indices_tensor = std::vector<int32_t>(caches.ordinary_slots.begin(), caches.ordinary_slots.begin() + rows);
  std::vector<int32_t> positions(step.positions.begin(), step.positions.end());
  vllm::ForwardLogits output;
  vllm::MmForwardBuffers embeddings;
  if (layout.value("image_decode",false)) {
    std::vector<int32_t> axes;
    for (int axis=0;axis<3;++axis) for (int row=0;row<rows;++row) {
      const int64_t position=step.positions[row]+(row==selected ? layout.at("mrope_delta").get<int>() : 0);
      Require(position>=0 && position<1600,"invalid image completion position");
      axes.push_back(static_cast<int32_t>(position));
    }
    const std::vector<vt::Tensor> visual_slices;
    const std::vector<char> mask(rows,0);
    vllm::MmEmbedInputs embed;
    embed.token_ids=&step.input_token_ids;embed.mm_embeds=&visual_slices;
    embed.is_mm_embed=&mask;embed.mrope_positions=&axes;
    embeddings=vllm::Qwen3_5DenseEmbedMultimodal(weights,config,caches.d.q,embed);
    const std::vector<int32_t> logits_indices;
    vllm::ModelForwardInput input{step.input_token_ids,positions,am,gm,caches.kv,caches.gdn,
                                 config,caches.d.q,logits_indices};
    input.num_reqs=rows;input.gdn_state_slots=8;input.pure_decode=true;input.uniform_query_len=1;
    input.mm=embeddings.mm;
    output=vllm::Qwen3_5DenseForwardEmbeddings(input,weights);
  } else {
    output=vllm::Qwen3_5DenseModel::ForwardDevice(step.input_token_ids,positions,am,gm,
        caches.kv,caches.gdn,weights,config,caches.d.q);
  }
  Require(output.on_device() && output.device_tensor.dtype == vt::DType::kF32 && output.rows == rows &&
          output.vocab == 248320 && output.device_tensor.IsContiguous(), "wrong ordinary complete head output");
  return caches.Download(static_cast<char*>(output.device_tensor.data) + selected * 248320 * 4, 248320 * 4);
}
// Diagnostic only: histories use predetermined tokens, never sampled IDs.
// Same-state controls reset a C4 cache from the C1 incoming state at each step;
// the evolving C4 arms retain their own state normally across the window.
void RunTrajectory(const std::filesystem::path& model, const std::filesystem::path& capture,
                   const std::filesystem::path& plan_path, json& report) {
  const auto plan = ReadJson(plan_path);
  Require(plan.at("schema") == "native-target-trajectory-v1", "unknown trajectory schema");
  const int context = plan.at("initial_context");
  const auto tokens = plan.at("teacher_tokens").get<std::vector<int32_t>>();
  Require(context > 0 && context + tokens.size() < 1600 && !tokens.empty() && tokens.size() <= 32,
          "trajectory window bound exceeded");
  for (int token : tokens) Require(token >= 0 && token < 248320, "teacher token outside vocabulary");
  const auto admit = [&](const char* key) {
    const auto& pin = plan.at(key);
    const auto file = pin.at("file").get<std::string>();
    Require(std::filesystem::path(file).filename() == file &&
            Hash(Read(capture / file, 1024 * 1024)) == pin.at("sha256").get<std::string>(), "trajectory checkpoint pin differs");
    return Capture(capture, file.c_str());
  };
  const auto c1_seed = admit("c1_checkpoint"), c4_seed = admit("c4_checkpoint");
  const bool image_decode=plan.value("image_decode",false);
  const int mrope_delta=c1_seed.metadata.at("mrope_delta");
  Require(c1_seed.metadata.at("seq_len") == context && c4_seed.metadata.at("seq_len") == context &&
          c1_seed.metadata.at("token_prefix") == c4_seed.metadata.at("token_prefix") &&
          c4_seed.metadata.at("mrope_delta") == mrope_delta &&
          (image_decode || mrope_delta==0) && context+mrope_delta>=0 &&
          (!image_decode || (c1_seed.metadata.at("own_image_features")==1 && c4_seed.metadata.at("own_image_features")==1)) &&
          !c1_seed.metadata.contains("mtp") && !c4_seed.metadata.contains("mtp"),
          "trajectory seed histories/position contract differ");
  const auto config = vllm::LoadHfConfig((model / "config.json").string());
  Require(config.hidden_size == 5120 && config.vocab_size == 248320 && config.num_hidden_layers == 64,
          "wrong pinned full target");
  QueueOwner queue; queue.value = vt::CreateQueue({vt::DeviceType::kXPU, 0}); auto& q = *queue.value;
  auto weights = [&] {
    const auto index = vllm::LoadSafetensorsIndex((model / "model.safetensors.index.json").string());
    std::set<std::string> files;
    for (const auto& [name, file] : index) {
      (void)name; Require(std::filesystem::path(file).filename() == file, "unsafe model shard"); files.insert(file);
    }
    std::vector<vllm::SafetensorsFile> shards;
    for (const auto& file : files) shards.push_back(vllm::SafetensorsFile::Open((model / file).string()));
    vllm::MultiModalConfig mm; mm.limit_per_prompt = {{"image", 0}, {"video", 0}};
    return vllm::LoadQwen3_5Dense(shards, config, &q, &mm);
  }();
  Require(weights.exl3_checkpoint && weights.precision.activation == vt::DType::kF16 &&
          weights.precision.gdn_conv_state == vt::DType::kF16 && weights.precision.gdn_recurrent_state == vt::DType::kF32,
          "wrong target trajectory precision");
  vllm::Qwen3_5DenseModel::PrepareLmHeadResident(weights, q);
  vllm::Qwen3_5DenseModel::PrepareGdnFp8Resident(weights, config, q);
  Dev d{vt::GetBackend(q.device), q, vt::DType::kF16};
  const auto layout = [&](int rows, int selected, int step) {
    std::vector<int32_t> ids(rows, 13), starts(rows + 1), lengths(rows, context + step + 1);
    std::vector<int64_t> positions(rows, context + step);
    for (int row = 0; row <= rows; ++row) starts[row] = row;
    ids[selected] = tokens[step];
    return json{{"num_reqs",rows},{"row",selected},{"actual_token_rows",rows},
      {"own_image_features",0},{"mrope_delta",mrope_delta},{"image_decode",image_decode},{"max_query_len",1},
      {"input_token_ids",ids},{"positions",positions},{"query_start_loc",starts},
      {"seq_lens",lengths},{"kv_block_table_cols",1}};
  };
  const auto fingerprints = [&](const Snapshot& state) {
    json hashes = json::object();
    for (const auto& [name, bytes] : state) hashes[name] = Hash(bytes);
    return hashes;
  };
  const auto unused = [&](Caches& cache, int rows) {
    size_t checked = 0;
    for (const auto& state : cache.gdn) {
      for (const auto* tensor : {&state.conv_state, &state.ssm_state}) {
        const size_t bytes = tensor->Bytes() / 8;
        for (int slot = 0; slot < 8; ++slot) {
          if (std::find(cache.ordinary_slots.begin(), cache.ordinary_slots.begin() + rows, slot) !=
              cache.ordinary_slots.begin() + rows) continue;
          const auto raw = cache.Download(static_cast<const char*>(tensor->data) + slot * bytes, bytes);
          Require(std::all_of(raw.begin(), raw.end(), [](unsigned char x) { return x == 0xa5; }),
                  "trajectory wrote an inactive GDN owner");
          checked += bytes;
        }
      }
    }
    for (const auto& kv : cache.kv) for (int page = 0; page < 4; ++page) {
      if (std::find(cache.pages.begin(), cache.pages.begin() + rows, page) != cache.pages.begin() + rows) continue;
      const auto raw = cache.Download(static_cast<const char*>(kv.data) + page * kv.page_size_bytes, kv.page_size_bytes);
      Require(std::all_of(raw.begin(), raw.end(), [](unsigned char x) { return x == 0xa5; }),
              "trajectory wrote an inactive KV owner");
      checked += raw.size();
    }
    return checked;
  };
  // Caches always have the same allocated owner geometry, allowing a complete
  // device clone even when only one row is live. All inactive bytes are poisoned.
  Caches c1(d,c1_seed,false,context,1,0,true), shared(d,c1_seed,false,context,4,0,true),
         own(d,c4_seed,false,context,4,0,true), reset(d,c1_seed,false,context,4,0,true);
  report["steps"] = json::array();
  std::vector<Snapshot> shared_states;
  std::vector<Bytes> shared_logits;
  for (size_t step = 0; step < tokens.size(); ++step) {
    reset.CopyFrom(c1);
    // Clone only the selected state into live sibling rows before the reset
    // control. Siblings remain diagnostic histories, never claimed as serving.
    for (auto& state : reset.gdn) for (const auto* tensor : {&state.conv_state, &state.ssm_state}) {
      const size_t bytes = tensor->Bytes()/8;
      for (int row=1;row<4;++row) d.b.Copy(q,static_cast<char*>(tensor->data)+reset.ordinary_slots[row]*bytes,
          static_cast<const char*>(tensor->data)+reset.ordinary_slots[0]*bytes,bytes);
    }
    for (auto& kv : reset.kv) for (int row=1;row<4;++row)
      d.b.Copy(q,static_cast<char*>(kv.data)+reset.pages[row]*kv.page_size_bytes,
          static_cast<const char*>(kv.data)+reset.pages[0]*kv.page_size_bytes,kv.page_size_bytes);
    const auto a = ForwardOrdinaryLayout(weights,config,c1,layout(1,0,step),false);
    const auto b = ForwardOrdinaryLayout(weights,config,shared,layout(4,0,step),false);
    const auto c = ForwardOrdinaryLayout(weights,config,own,layout(4,0,step),false);
    const auto e = ForwardOrdinaryLayout(weights,config,reset,layout(4,0,step),false);
    auto sa = c1.Observe(0,context+step+1), sb=shared.Observe(0,context+step+1),
         sc=own.Observe(0,context+step+1), se=reset.Observe(0,context+step+1);
    report["steps"].push_back({{"step",step},{"position",context+step},{"mrope_position",context+step+mrope_delta},{"teacher_token",tokens[step]},
      {"identical_incoming_single_step",{{"logits",LogitMetrics(a,e)},{"states",Compare(sa,se)}}},
      {"evolving_shared_seed",{{"logits",LogitMetrics(a,b)},{"states",Compare(sa,sb)}}},
      {"evolving_separate_serving_seeds",{{"logits",LogitMetrics(a,c)},{"states",Compare(sa,sc)}}},
      {"c1_state_sha256",fingerprints(sa)},{"c4_shared_state_sha256",fingerprints(sb)},
      {"c4_own_state_sha256",fingerprints(sc)}});
    shared_states.push_back(std::move(sb)); shared_logits.push_back(b);
  }
  report["inactive_poisoned_bytes_checked"]={{"c1",unused(c1,1)},{"c4_shared",unused(shared,4)},
                                            {"c4_own",unused(own,4)}};
  const auto control = [&](int selected) {
    Caches cache(d,c1_seed,false,context,4,selected,true);
    json result = json::array();
    for (size_t step=0;step<tokens.size();++step) {
      const auto logits=ForwardOrdinaryLayout(weights,config,cache,layout(4,selected,step),false);
      const auto state=cache.Observe(0,context+step+1);
      const auto comparison=Compare(shared_states[step],state);
      Require(comparison.at("all_storage_exact") && logits==shared_logits[step],
              "same-geometry repeat/slot trajectory differs");
      result.push_back({{"step",step},{"states",comparison},{"logits",LogitMetrics(shared_logits[step],logits)}});
    }
    return json{{"steps",result},{"inactive_poisoned_bytes_checked",unused(cache,4)}};
  };
  report["same_geometry_repeat"] = control(0);
  report["same_geometry_moved_row_slot_page"] = control(1);
  report["initial_c1_metadata_sha256"] = Hash(Read(capture / plan.at("c1_checkpoint").at("file").get<std::string>(),1024*1024));
  report["initial_c4_metadata_sha256"] = Hash(Read(capture / plan.at("c4_checkpoint").at("file").get<std::string>(),1024*1024));
  report["seed_state_comparison"] = Compare([&] { Snapshot x;for (auto& [k,v]:c1_seed.blobs) if(k.starts_with("gdn")||k.starts_with("attn"))x.emplace(k,v);return x;}(),
      [&] { Snapshot x;for (auto& [k,v]:c4_seed.blobs) if(k.starts_with("gdn")||k.starts_with("attn"))x.emplace(k,v);return x;}());
  report["image_decode"]=image_decode;
  report["mrope_delta"]=mrope_delta;
  report["teacher_tokens"] = tokens;
  report["initial_context"] = context;
  report["plan_sha256"] = Hash(Read(plan_path,1024*1024));
  report["scope"] = "diagnostic teacher-forced selected target trajectory; actual separately identified C1/C4 seed states, cloned diagnostic siblings, one identical-incoming control per step, same-M4 repeat/slot controls; not scheduler/MTP/graph or free-generation equality";
  report["reference_tier_hits"] = vt::GetReferenceTierHits();
  Require(vt::GetReferenceTierHits()==0,"trajectory reference fallback");
  report["status"] = "DIAGNOSTIC";
}

void RunOrdinaryPair(const std::filesystem::path& model, const std::filesystem::path& capture, json& report) {
  Capture mixed_in(capture, "prefix-0.json"), mixed_out(capture, "prefix-1.json"),
          c1_in(capture, "prefix-2.json"), c1_out(capture, "prefix-3.json");
  const auto validate_pair = [&](const Capture& initial, const Capture& actual) {
    Require(initial.metadata.at("seq_len") == 38 && actual.metadata.at("seq_len") == 39 &&
            actual.metadata.at("num_computed_before_step") == 38 &&
            initial.metadata.at("request_id") == actual.metadata.at("request_id") &&
            initial.metadata.at("gdn_slot") == actual.metadata.at("gdn_slot") &&
            initial.metadata.at("output_prefix").size() == 20 && actual.metadata.at("output_prefix").size() == 21,
            "wrong actual adjacent target pair");
    auto prefix = actual.metadata.at("token_prefix").get<std::vector<int32_t>>();
    Require(prefix.size() == 39 && prefix.back() == 2702, "wrong last common token"); prefix.pop_back();
    Require(initial.metadata.at("token_prefix") == prefix, "adjacent target history differs");
  };
  validate_pair(mixed_in, mixed_out); validate_pair(c1_in, c1_out);
  Require(mixed_in.metadata.at("token_prefix") == c1_in.metadata.at("token_prefix"), "different frozen histories");
  const auto config = vllm::LoadHfConfig((model / "config.json").string());
  Require(config.hidden_size == 5120 && config.vocab_size == 248320 && config.num_hidden_layers == 64,
          "wrong pinned full target");
  QueueOwner queue; queue.value = vt::CreateQueue({vt::DeviceType::kXPU, 0}); auto& q = *queue.value;
  auto weights = [&] {
    const auto index = vllm::LoadSafetensorsIndex((model / "model.safetensors.index.json").string());
    std::set<std::string> files;
    for (const auto& [name, file] : index) { (void)name; Require(std::filesystem::path(file).filename() == file, "unsafe model shard"); files.insert(file); }
    std::vector<vllm::SafetensorsFile> shards;
    for (const auto& file : files) shards.push_back(vllm::SafetensorsFile::Open((model / file).string()));
    vllm::MultiModalConfig mm; mm.limit_per_prompt = {{"image", 0}, {"video", 0}};
    return vllm::LoadQwen3_5Dense(shards, config, &q, &mm);
  }();
  Require(weights.exl3_checkpoint && weights.precision.activation == vt::DType::kF16 &&
          weights.precision.gdn_conv_state == vt::DType::kF16 && weights.precision.gdn_recurrent_state == vt::DType::kF32,
          "wrong actual precision policy");
  vllm::Qwen3_5DenseModel::PrepareLmHeadResident(weights, q);
  vllm::Qwen3_5DenseModel::PrepareGdnFp8Resident(weights, config, q);
  Dev d{vt::GetBackend(q.device), q, vt::DType::kF16};
  const auto run = [&](const Capture& initial, const json& layout) {
    Caches cache(d, initial, false, 38, layout.at("num_reqs"), layout.at("row"));
    auto logits = ForwardOrdinaryLayout(weights, config, cache, layout);
    return std::make_pair(cache.Observe(0, 39), logits);
  };
  const auto captured_states = [&](const Capture& actual) {
    Snapshot result;
    for (const auto& [name, raw] : actual.blobs)
      if (name.starts_with("gdn") || name.starts_with("attn")) result.emplace(name, raw);
    Require(result.size() == 128, "missing complete actual target snapshot"); return result;
  };
  const auto c1 = run(c1_in, c1_out.metadata), mixed = run(mixed_in, mixed_out.metadata);
  auto& anchors = report["actual_ordinary_anchors"];
  anchors["c1"] = Compare(captured_states(c1_out), c1.first);
  anchors["c1"]["logits"] = LogitMetrics(c1_out.State("logits", 0, json::array({248320})), c1.second);
  anchors["mixed_selected"] = Compare(captured_states(mixed_out), mixed.first);
  anchors["mixed_selected"]["logits"] = LogitMetrics(mixed_out.State("logits", 0, json::array({248320})), mixed.second);
  // Shared selected state, real current-token/positions and physical shapes.
  // Sibling rows use diagnostic histories; only the selected row is anchored.
  const auto shared = run(c1_in, mixed_out.metadata);
  report["shared_c1_incoming_new_m4_vs_m1"] = Compare(c1.first, shared.first);
  report["shared_c1_incoming_new_m4_vs_m1"]["logits"] = LogitMetrics(c1.second, shared.second);
  Capture original(capture.parent_path() / "c2-alpha-prefix-v2", "prefix-1.json");
  Require(original.metadata.at("token_prefix") == c1_out.metadata.at("token_prefix") &&
          original.metadata.at("num_reqs") == 4 && original.metadata.at("row") == 1,
          "wrong original failing M4 geometry");
  const auto original_shape = run(c1_in, original.metadata);
  report["shared_c1_incoming_original_m4_vs_m1"] = Compare(c1.first, original_shape.first);
  report["shared_c1_incoming_original_m4_vs_m1"]["logits"] = LogitMetrics(c1.second, original_shape.second);
  report["shared_c1_incoming_new_vs_original_m4"] = Compare(shared.first, original_shape.first);
  report["shared_c1_incoming_new_vs_original_m4"]["logits"] = LogitMetrics(shared.second, original_shape.second);
  report["reference_tier_hits"] = vt::GetReferenceTierHits();
  report["ordinary_recurrence_setting"] = std::getenv("VT_XPU_GDN_DECODE") ? std::getenv("VT_XPU_GDN_DECODE") : "auto";
  report["scope"] = "selected pure-text alpha target only; actual own-state M1/M4 anchors, shared C1 initial state in two real M4 layouts; sibling histories are diagnostic, no whole mixed/MTP/image/reference qualification";
  Require(anchors["c1"]["all_storage_exact"] && anchors["c1"]["logits"]["storage_exact"] &&
          anchors["mixed_selected"]["all_storage_exact"] && anchors["mixed_selected"]["logits"]["storage_exact"] &&
          vt::GetReferenceTierHits() == 0, "actual ordinary selected-row anchor differs");
  report["status"] = "DIAGNOSTIC";
}
void Run(const std::filesystem::path& model, const std::filesystem::path& capture, json& report,
         bool match_attention_reference, bool check_acceptance_states) {
  Capture initial(capture, "prefix-2.json"), actual(capture, "prefix-3.json");
  Require(initial.metadata.at("output_prefix") == json::array() && actual.metadata.at("output_prefix") == json::array({271}) &&
          initial.metadata.at("seq_len") == 18 && actual.metadata.at("seq_len") == 19 &&
          initial.metadata.at("request_id") == actual.metadata.at("request_id") &&
          actual.metadata.at("num_reqs") == 1 && actual.metadata.at("actual_token_rows") == 4 &&
          actual.metadata.at("mtp").at("verification_tokens") == json::array({271, 760, 5089, 314}),
          "wrong frozen actual C1 MTP pair");
  const auto config = vllm::LoadHfConfig((model / "config.json").string());
  Require(config.hidden_size == 5120 && config.vocab_size == 248320 && config.num_hidden_layers == 64,
          "wrong pinned target config");
  QueueOwner queue; queue.value = vt::CreateQueue({vt::DeviceType::kXPU, 0});
  auto& q = *queue.value;
  auto weights = [&] {
    const auto index = vllm::LoadSafetensorsIndex((model / "model.safetensors.index.json").string());
    std::set<std::string> files;
    for (const auto& [name, file] : index) { (void)name; Require(std::filesystem::path(file).filename() == file, "unsafe model shard"); files.insert(file); }
    std::vector<vllm::SafetensorsFile> shards;
    for (const auto& file : files) shards.push_back(vllm::SafetensorsFile::Open((model / file).string()));
    vllm::MultiModalConfig mm; mm.limit_per_prompt = {{"image", 0}, {"video", 0}};
    return vllm::LoadQwen3_5Dense(shards, config, &q, &mm);
  }();
  Require(weights.exl3_checkpoint && weights.precision.activation == vt::DType::kF16 &&
          weights.precision.gdn_conv_state == vt::DType::kF16 && weights.precision.gdn_recurrent_state == vt::DType::kF32,
          "wrong production EXL3 precision policy");
  vllm::Qwen3_5DenseModel::PrepareLmHeadResident(weights, q);
  vllm::Qwen3_5DenseModel::PrepareGdnFp8Resident(weights, config, q);
  Dev d{vt::GetBackend(q.device), q, vt::DType::kF16};
  Caches spec(d, initial, true);
  // The first additional difference after scalar GDN control is across layer3.
  // Existing optional probes observe only that layer, in both forward calls.
  const vllm::actdump::StageLayerSelectionScope layer_dump(3);
  report["optional_substage_capture_layer"] = std::getenv("VT_DUMP_ACT_SUB") ? json(3) : json(nullptr);
  auto spec_logits = Forward(weights, config, spec, {271, 760, 5089, 314});
  const auto spec_states = spec.Observe();
  Snapshot actual_states;
  for (const auto& [name, raw] : spec_states) {
    (void)raw;
    actual_states[name] = name.ends_with("-conv") ? Window(actual.blobs.at(name), 6, 0) : actual.blobs.at(name);
  }
  const auto& expected = actual.State("verify-logits", 0, json::array({4, 248320}));
  report["actual_serving_spec_anchor"] = Compare(actual_states, spec_states);
  report["actual_serving_spec_anchor"]["all_four_logit_rows_storage_exact"] = spec_logits == expected;
  report["actual_serving_spec_anchor"]["first_logit_metrics"] = LogitMetrics(
      Bytes(expected.begin(), expected.begin() + 248320 * 4), Bytes(spec_logits.begin(), spec_logits.begin() + 248320 * 4));
  Caches ordinary(d, initial, false);
  const auto target_logits = Forward(weights, config, ordinary, {271, 271, 271, 271});
  report["same_m_target_control"] = Compare(spec_states, ordinary.Observe());
  report["same_m_target_control"]["first_logit_metrics"] = LogitMetrics(
      Bytes(spec_logits.begin(), spec_logits.begin() + 248320 * 4), Bytes(target_logits.begin(), target_logits.begin() + 248320 * 4));
  if (match_attention_reference) {
    // Test-harness-only attribution: retain the untouched auto-route failure
    // above, then run both shapes through the same existing XPU implementation.
    // Never select this arithmetic as a new serving profile or waive that gate.
    const char* original = std::getenv("VT_XPU_ATTENTION");
    Require(!original || std::string_view(original) == "auto", "diagnostic requires original auto attention");
    Require(setenv("VT_XPU_ATTENTION", "reference", 1) == 0, "cannot select diagnostic attention");
    Caches reference_spec(d, initial, true), reference_ordinary(d, initial, false);
    const auto reference_spec_logits = Forward(weights, config, reference_spec, {271, 760, 5089, 314});
    const auto reference_spec_states = reference_spec.Observe();
    const auto reference_target_logits = Forward(weights, config, reference_ordinary, {271, 271, 271, 271});
    auto& comparison = report["same_m_matched_attention_control"];
    comparison = Compare(reference_spec_states, reference_ordinary.Observe());
    comparison["first_logit_metrics"] = LogitMetrics(
        Bytes(reference_spec_logits.begin(), reference_spec_logits.begin() + 248320 * 4),
        Bytes(reference_target_logits.begin(), reference_target_logits.begin() + 248320 * 4));
    comparison["attention_mode"] = "reference";
    comparison["scope"] = "diagnostic matched XPU attention; not default-route qualification";
    report["forced_spec_vs_actual"] = Compare(actual_states, reference_spec_states);
    report["forced_spec_vs_actual"]["all_four_logit_rows_storage_exact"] = reference_spec_logits == expected;
    report["forced_spec_vs_actual"]["first_logit_metrics"] = LogitMetrics(
        Bytes(expected.begin(), expected.begin() + 248320 * 4),
        Bytes(reference_spec_logits.begin(), reference_spec_logits.begin() + 248320 * 4));
    if (check_acceptance_states) {
      auto& cases = report["matched_acceptance_controls"] = json::array();
      const std::vector<int32_t> verification{271, 760, 5089, 314};
      for (int accepted = 0; accepted < 4; ++accepted) {
        Bytes serial_logits = reference_target_logits;
        if (accepted)
          serial_logits = Forward(weights, config, reference_ordinary,
                                  std::vector<int32_t>(4, verification[accepted]), 18 + accepted);
        const Bytes expected_row(reference_spec_logits.begin() + accepted * 248320 * 4,
                                 reference_spec_logits.begin() + (accepted + 1) * 248320 * 4);
        json item{{"accepted_drafts", accepted}, {"valid_context", 19 + accepted},
                  {"selected_snapshot_slot", std::vector<int32_t>{3, 1, 6, 4}[accepted]}};
        item["prefix_states"] = Compare(reference_spec.Observe(accepted), reference_ordinary.Observe(0, 19 + accepted));
        item["prefix_logits"] = LogitMetrics(expected_row, Bytes(serial_logits.begin(), serial_logits.begin() + 248320 * 4));
        int correction = 0;
        for (int token = 1; token < 248320; ++token)
          if (vt::LoadUnaligned<float>(expected_row.data() + token * 4) >
              vt::LoadUnaligned<float>(expected_row.data() + correction * 4)) correction = token;
        // Clone the same provisional verification for each selector case.
        // Neither the next case nor serial teacher forcing sees rejected suffixes.
        Caches correction_spec(d, initial, true), correction_ordinary(d, initial, false);
        correction_spec.CopyFrom(reference_spec); correction_ordinary.CopyFrom(reference_ordinary);
        const auto spec_correction = Forward(weights, config, correction_spec,
            {correction, 760, 5089, 314}, 19 + accepted, accepted + 1);
        const auto ordinary_correction = Forward(weights, config, correction_ordinary,
            std::vector<int32_t>(4, correction), 19 + accepted);
        item["teacher_forced_correction_token"] = correction;
        item["correction_context"] = 20 + accepted;
        item["correction_states"] = Compare(correction_spec.Observe(0, 20 + accepted),
                                             correction_ordinary.Observe(0, 20 + accepted));
        item["correction_logits"] = LogitMetrics(
            Bytes(spec_correction.begin(), spec_correction.begin() + 248320 * 4),
            Bytes(ordinary_correction.begin(), ordinary_correction.begin() + 248320 * 4));
        cases.push_back(item);
        Require(item["prefix_states"]["all_storage_exact"] && item["prefix_logits"]["storage_exact"] &&
                item["correction_states"]["all_storage_exact"] && item["correction_logits"]["storage_exact"],
                "matched-arithmetic full acceptance/correction state control differs");
      }
      report["matched_acceptance_scope"] = "teacher-forced acceptance lengths of frozen real first-verification tokens; all target states/logits, not actual sampler acceptance or production arithmetic qualification";
    }
    Require(setenv("VT_XPU_ATTENTION", "auto", 1) == 0, "cannot restore original attention");
  }
  report["reference_tier_hits"] = vt::GetReferenceTierHits();
  const char* mode = std::getenv("VT_XPU_GDN_DECODE");
  report["ordinary_recurrence_setting"] = mode ? mode : "auto";
  report["scope"] = "full 64-layer actual target; C1 first MTP verification eager anchor and one ordinary M4 first-query control; dummy ordinary rows share the initial history; no all-acceptance/mixed/image/last-prefix21 qualification";
  if (match_attention_reference)
    Require(report["same_m_matched_attention_control"]["all_storage_exact"] &&
            report["same_m_matched_attention_control"]["first_logit_metrics"]["storage_exact"],
            "matched reference-attention target control differs");
  Require(report["actual_serving_spec_anchor"]["all_storage_exact"] &&
          report["actual_serving_spec_anchor"]["all_four_logit_rows_storage_exact"], "full speculative replay differs from actual serving");
  Require(report["same_m_target_control"]["all_storage_exact"] &&
          report["same_m_target_control"]["first_logit_metrics"]["storage_exact"] && vt::GetReferenceTierHits() == 0,
          "full matched-M4 target state/logit comparison differs");
  report["status"] = "DIAGNOSTIC";
}
}  // namespace
int main(int argc, char** argv) {
  if (argc < 4 || argc > 6 || std::filesystem::exists(argv[3])) return 2;
  if (argc==6 && std::string_view(argv[4])=="--trajectory") {
    json report{{"status","FAIL"}};
    try { RunTrajectory(argv[1],argv[2],argv[5],report); }
    catch (const std::exception& error) { report["error"]=error.what(); }
    std::ofstream file(argv[3]); file<<report.dump(2)<<'\n';
    return file.good() && report.at("status")=="DIAGNOSTIC" ? 0 : 1;
  }
  bool match_attention = false, acceptance = false, ordinary_pair = false;
  for (int i = 4; i < argc; ++i) {
    const std::string_view flag(argv[i]);
    if (flag == "--match-attention-reference" && !match_attention) match_attention = true;
    else if (flag == "--check-acceptance-states" && !acceptance) acceptance = true;
    else if (flag == "--ordinary-prefix-pair" && !ordinary_pair) ordinary_pair = true;
    else return 2;
  }
  if ((acceptance && !match_attention) || (ordinary_pair && (acceptance || match_attention))) return 2;
  json report{{"status", "FAIL"}};
  try {
    if (ordinary_pair) RunOrdinaryPair(argv[1], argv[2], report);
    else Run(argv[1], argv[2], report, match_attention, acceptance);
  }
  catch (const std::exception& error) { report["error"] = error.what(); }
  std::ofstream file(argv[3]); file << report.dump(2) << '\n';
  if (!file.good()) return 2;
  return report.at("status") == "DIAGNOSTIC" ? 0 : 1;
}
