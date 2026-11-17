// Bounded language-model first-layer control for the existing mixed vision case.
// Uses the production FP16 loader, RMSNorm, grouped EXL3 projection and GDN
// block. No full LLM, scheduler, sampler, alternate precision or replacement golden.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_exl3_linear.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_gdn_replay.h"
#include "vllm/model_executor/models/act_dump.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vllm/v1/worker/gpu/prepare_inputs.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vt/exl3_grouped.h"
#include "vt/op_provider.h"
#include "vt/unaligned.h"

namespace {
using json = nlohmann::json;
using Row = std::vector<uint16_t>;
constexpr int64_t kHidden = 5120, kConv = 10240, kGroup = 16384;
const std::vector<int64_t> kRows{1, 4, 18, 246, 509};
void Require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
std::string Hash(const void* data, size_t bytes) {
  const auto raw = vllm::v1::sha256_bytes(std::string(static_cast<const char*>(data), bytes));
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (unsigned char value : raw) { result += hex[value >> 4]; result += hex[value & 15]; }
  return result;
}
std::vector<unsigned char> Read(const std::filesystem::path& path, size_t limit) {
  const auto bytes = std::filesystem::file_size(path);
  Require(bytes > 0 && bytes <= limit, "diagnostic input byte bound exceeded");
  std::ifstream input(path, std::ios::binary);
  Require(input.good(), "cannot open diagnostic input");
  std::vector<unsigned char> result{std::istreambuf_iterator<char>(input), {}};
  Require(result.size() == bytes, "incomplete diagnostic input");
  return result;
}
json ReadJson(const std::filesystem::path& path) {
  const auto bytes = Read(path, 1024 * 1024);
  return json::parse(bytes);
}
struct QueueOwner {
  std::optional<vt::Queue> value;
  ~QueueOwner() { if (value) vt::DestroyQueue(*value); }
};
Row DownloadRow(vllm::dense_attn::Dev d, const vt::Tensor& tensor, int64_t columns, int64_t row = 0) {
  Require(tensor.dtype == vt::DType::kF16 && tensor.rank == 2 && tensor.shape[1] == columns &&
          tensor.stride[1] == 1 && row >= 0 && row < tensor.shape[0], "invalid native row layout");
  Row result(columns);
  d.b.Copy(d.q, result.data(), static_cast<const char*>(tensor.data) + row * tensor.stride[0] * 2,
           result.size() * sizeof(uint16_t));
  d.b.Synchronize(d.q);
  return result;
}
json Compare(const Row& reference, const Row& native) {
  Require(!reference.empty() && reference.size() == native.size(), "different row lengths");
  size_t different = 0, largest = 0;
  double error = 0, norm = 0, maximum = 0;
  for (size_t i = 0; i < reference.size(); ++i) {
    const double a = vt::F16ToF32(reference[i]), b = vt::F16ToF32(native[i]);
    Require(std::isfinite(a) && std::isfinite(b), "nonfinite replay output");
    different += reference[i] != native[i];
    const double delta = std::abs(a - b);
    error += delta * delta; norm += a * a;
    if (delta > maximum) { maximum = delta; largest = i; }
  }
  return {{"elements", reference.size()}, {"different_elements", different},
          {"storage_exact", different == 0}, {"rel_l2", std::sqrt(error / std::max(norm, 1e-30))},
          {"max_abs", maximum}, {"max_coordinate", largest},
          {"reference_at_max", vt::F16ToF32(reference[largest])},
          {"native_at_max", vt::F16ToF32(native[largest])}};
}
Row CapturedConv(const std::filesystem::path& root, int index, json& evidence) {
  const auto metadata = ReadJson(root / ("prefix-" + std::to_string(index) + ".json"));
  Require(metadata.at("phase") == "post_target_before_sampling" && metadata.at("seq_len") == 19 &&
          metadata.at("output_prefix") == json::array({271}) && metadata.at("own_image_features") == 0,
          "wrong actual first-decode boundary");
  const auto& blobs = metadata.at("blobs");
  auto entry = std::find_if(blobs.begin(), blobs.end(), [](const json& b) { return b.at("name") == "gdn0-conv"; });
  Require(entry != blobs.end() && entry->at("dtype") == 1 &&
          entry->at("shape") == json::array({kConv, 3}), "missing first Conv boundary");
  const auto name = entry->at("file").get<std::string>();
  Require(std::filesystem::path(name).filename() == name, "unsafe boundary path");
  const auto bytes = Read(root / name, kConv * 3 * sizeof(uint16_t));
  Require(bytes.size() == kConv * 3 * sizeof(uint16_t) && entry->at("bytes") == bytes.size() &&
          entry->at("sha256") == Hash(bytes.data(), bytes.size()), "actual Conv checksum/size mismatch");
  Row result(kConv);
  for (int64_t i = 0; i < kConv; ++i) result[i] = vt::LoadUnaligned<uint16_t>(bytes.data() + (3 * i + 2) * 2);
  evidence = {{"metadata", "prefix-" + std::to_string(index) + ".json"},
              {"conv_sha256", entry->at("sha256")}, {"selected_temporal_column", 2},
              {"request_id", metadata.at("request_id")}, {"matrix_rows", metadata.at("actual_token_rows")}};
  return result;
}
json Save(const std::filesystem::path& path, const Row& row) {
  Require(!std::filesystem::exists(path), "row evidence already exists");
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(uint16_t));
  Require(output.good(), "cannot write row evidence");
  return {{"file", path.filename().string()}, {"dtype", "F16"}, {"shape", {row.size()}},
          {"sha256", Hash(row.data(), row.size() * sizeof(uint16_t))}};
}
void Run(const std::filesystem::path& model, const std::filesystem::path& capture,
         const std::filesystem::path& output, const std::filesystem::path& other_capture, json& report) {
  // Declare first, create only after host validation, destroy after residents.
  QueueOwner queue;
  const auto config = ReadJson(model / "config.json");
  const auto& text = config.at("text_config");
  Require(config.at("model_type") == "qwen3_5" && text.at("hidden_size") == kHidden &&
          text.at("linear_num_key_heads") == 16 && text.at("linear_num_value_heads") == 48 &&
          text.at("linear_key_head_dim") == 128 && text.at("linear_value_head_dim") == 128 &&
          text.at("linear_conv_kernel_dim") == 4, "wrong pinned model geometry");
  const float eps = text.at("rms_norm_eps").get<float>();
  Require(eps == 1e-6f, "wrong RMSNorm epsilon");
  const auto index = vllm::LoadSafetensorsIndex((model / "model.safetensors.index.json").string());
  std::map<std::string, vllm::SafetensorsFile> shards;
  const vllm::TensorResolver get = [&](const std::string& name) -> const vllm::StTensor& {
    const auto& filename = index.at(name);
    Require(std::filesystem::path(filename).filename() == filename, "unsafe checkpoint shard");
    if (!shards.count(filename)) shards.emplace(filename, vllm::SafetensorsFile::Open((model / filename).string()));
    return shards.at(filename).Get(name);
  };
  const auto has = [&](const std::string& name) { return index.count(name) != 0; };
  const std::string base = "model.language_model.layers.0.";
  const auto& embedding = get("model.language_model.embed_tokens.weight");
  Require(embedding.dtype == "BF16" && embedding.shape == std::vector<int64_t>({248320, kHidden}),
          "wrong embedding storage");
  auto selected = embedding;
  selected.data += 271 * kHidden * 2; selected.shape = {1, kHidden}; selected.nbytes = kHidden * 2;
  auto input = vllm::dense_loaders::LoadF16Direct([&](const std::string&) -> const vllm::StTensor& {
    return selected;
  }, "selected token271 embedding row");
  auto norm = vllm::dense_loaders::LoadF16Direct(get, base + "input_layernorm.weight");
  auto qkv = vllm::dense_loaders::LoadExl3(get, has, base + "linear_attn.in_proj_qkv");
  auto z = vllm::dense_loaders::LoadExl3(get, has, base + "linear_attn.in_proj_z");
  Require(norm.rank == 1 && norm.shape[0] == kHidden && qkv.InFeatures() == kHidden &&
          qkv.OutFeatures() == kConv && z.InFeatures() == kHidden && z.OutFeatures() == kGroup - kConv &&
          qkv.Bits() == 4 && z.Bits() == 4 && qkv.codebook == 2 && z.codebook == 2,
          "wrong actual first QKVZ geometry/codebook");
  auto grouped = vllm::MergeExl3Weights({&qkv, &z}, qkv.name + "+" + z.name);
  report["weights"] = json::object();
  for (const auto& name : {base + "input_layernorm.weight", base + "linear_attn.in_proj_qkv.trellis",
                          base + "linear_attn.in_proj_qkv.suh", base + "linear_attn.in_proj_qkv.svh",
                          base + "linear_attn.in_proj_qkv.mul1",
                          base + "linear_attn.in_proj_z.trellis", base + "linear_attn.in_proj_z.suh",
                          base + "linear_attn.in_proj_z.svh", base + "linear_attn.in_proj_z.mul1"}) {
    const auto& tensor = get(name);
    report["weights"][name] = {{"dtype", tensor.dtype}, {"shape", tensor.shape},
                              {"sha256", Hash(tensor.data, tensor.nbytes)}};
  }
  report["embedding_row"] = {{"token_id", 271}, {"source_dtype", selected.dtype},
      {"source_sha256", Hash(selected.data, selected.nbytes)},
      {"f16_sha256", Hash(input.bytes.data(), input.bytes.size())}};
  json mixed_evidence, c1_evidence;
  const auto captured_mixed = CapturedConv(capture, 1, mixed_evidence);
  const auto captured_c1 = CapturedConv(capture, 3, c1_evidence);
  Require(mixed_evidence.at("matrix_rows") == 509 && c1_evidence.at("matrix_rows") == 1,
          "wrong actual paired shapes");
  report["actual_serving_boundaries"] = {{"mixed", mixed_evidence}, {"c1", c1_evidence}};
  Row captured_m4;
  if (!other_capture.empty()) {
    json evidence;
    captured_m4 = CapturedConv(other_capture, 1, evidence);
    Require(evidence.at("matrix_rows") == 4, "wrong other actual first-decode shape");
    report["actual_serving_boundaries"]["m4"] = evidence;
  }
  for (auto m : kRows) for (const char* stage : {"norm", "chain", "fixed-norm"})
    Require(!std::filesystem::exists(output.parent_path() / (output.stem().string() + "-m" +
                  std::to_string(m) + "-" + stage + ".f16")), "output evidence exists");
  queue.value = vt::CreateQueue({vt::DeviceType::kXPU, 0});
  auto& q = *queue.value;
  vllm::dense_attn::Dev device{vt::GetBackend(q.device), q, vt::DType::kF16};
  vllm::dense_attn::DBuf gamma(device, vt::DType::kF16, {kHidden}, norm.bytes.data());
  Row baseline_norm, baseline_projection;
  report["cases"] = json::array();
  for (const auto m : kRows) {
    Row repeated(m * kHidden), zeros(m * kHidden, 0);
    for (int64_t row = 0; row < m; ++row)
      std::memcpy(repeated.data() + row * kHidden, input.bytes.data(), kHidden * 2);
    vllm::dense_attn::DBuf activation(device, vt::DType::kF16, {m, kHidden}, repeated.data());
    vllm::dense_attn::DBuf residual(device, vt::DType::kF16, {m, kHidden}, zeros.data());
    vllm::dense_attn::DBuf normalized(device, vt::DType::kF16, {m, kHidden});
    vt::RmsNorm(q, normalized.t(), activation.t(), gamma.t(), {eps, true}, &residual.t());
    const auto norm_row = DownloadRow(device, normalized.t(), kHidden);
    auto projection = vllm::dense_exl3::GroupedLinear(device, normalized.t(), {&qkv, &z}, grouped);
    const auto chain_row = DownloadRow(device, projection.t(), kGroup);
    if (m == 1) { baseline_norm = norm_row; baseline_projection = chain_row; }
    for (int64_t row = 0; row < m; ++row)
      std::memcpy(repeated.data() + row * kHidden, baseline_norm.data(), kHidden * 2);
    vllm::dense_attn::DBuf fixed(device, vt::DType::kF16, {m, kHidden}, repeated.data());
    auto isolated = vllm::dense_exl3::GroupedLinear(device, fixed.t(), {&qkv, &z}, grouped);
    const auto isolated_row = DownloadRow(device, isolated.t(), kGroup);
    const auto path = [&](const std::string& stage) { return output.parent_path() /
      (output.stem().string() + "-m" + std::to_string(m) + "-" + stage + ".f16"); };
    json entry{{"m", m}, {"norm_vs_m1", Compare(baseline_norm, norm_row)},
        {"chain_vs_m1", Compare(baseline_projection, chain_row)},
        {"fixed_m1_norm_projection_vs_m1", Compare(baseline_projection, isolated_row)},
        {"norm", Save(path("norm"), norm_row)}, {"chain", Save(path("chain"), chain_row)},
        {"fixed_norm_projection", Save(path("fixed-norm"), isolated_row)}};
    if (m > 1) {
      const auto second = DownloadRow(device, projection.t(), kGroup, 1);
      const auto last = DownloadRow(device, projection.t(), kGroup, m - 1);
      entry["repeated_row1_vs_row0"] = Compare(chain_row, second);
      entry["repeated_last_row_vs_row0"] = Compare(chain_row, last);
      Require(entry["repeated_row1_vs_row0"].at("storage_exact") &&
              entry["repeated_last_row_vs_row0"].at("storage_exact"), "repeated activation rows differ");
    }
    if (m <= 128) {
      const auto plan = vt::PlanExl3SmallM(m, kHidden, kGroup, 4);
      entry["plan"] = {{"route", "SmallM"}, {"vector", plan.vector},
                       {"padded_rows", plan.padded_rows}, {"splits", plan.splits}};
    } else {
      const auto plan = vt::PlanExl3W8A8(m, kHidden, kGroup, 2, 4, vt::Exl3W8A8ModelPanelColumns());
      entry["plan"] = {{"route", "W8A8"}, {"padded_rows", plan.padded_rows},
                       {"panel_columns", vt::Exl3W8A8ModelPanelColumns()}};
    }
    if (m == 1 || m == 509) {
      const Row mixed(chain_row.begin(), chain_row.begin() + kConv);
      entry["chain_vs_actual_new_conv"] = Compare(m == 1 ? captured_c1 : captured_mixed, mixed);
      Require(entry["chain_vs_actual_new_conv"].at("storage_exact"),
              "first-layer replay does not reproduce the actual Conv boundary");
    }
    if (m == 4 && !captured_m4.empty()) {
      const auto actual_row = DownloadRow(device, projection.t(), kGroup, 1);
      const Row mixed(actual_row.begin(), actual_row.begin() + kConv);
      entry["row1_chain_vs_actual_new_conv"] = Compare(captured_m4, mixed);
      Require(entry["row1_chain_vs_actual_new_conv"].at("storage_exact"),
              "M4 replay does not reproduce the actual Conv boundary");
    }
    report["cases"].push_back(std::move(entry));
  }
  report["reference_tier_hits"] = vt::GetReferenceTierHits();
  Require(vt::GetReferenceTierHits() == 0, "reference provider fallback executed");
  report["status"] = "DIAGNOSTIC";
}

std::vector<unsigned char> StateBlob(const std::filesystem::path& root, const json& metadata,
                                     const char* name, int dtype, const json& shape) {
  const auto& blobs = metadata.at("blobs");
  auto entry = std::find_if(blobs.begin(), blobs.end(), [&](const json& b) { return b.at("name") == name; });
  Require(entry != blobs.end() && entry->at("dtype") == dtype && entry->at("shape") == shape,
          "invalid real state boundary");
  const auto filename = entry->at("file").get<std::string>();
  Require(std::filesystem::path(filename).filename() == filename, "unsafe state path");
  const auto raw = Read(root / filename, 4 * 1024 * 1024);
  Require(raw.size() == entry->at("bytes") &&
          Hash(raw.data(), raw.size()) == entry->at("sha256").get<std::string>(),
          "real state checksum differs");
  return raw;
}
json CompareState(const std::vector<unsigned char>& reference,
                  const std::vector<unsigned char>& native, bool fp16) {
  Require(reference.size() == native.size(), "different state byte count");
  if (fp16) {
    Row a(reference.size() / 2), b(a.size());
    std::memcpy(a.data(), reference.data(), reference.size());
    std::memcpy(b.data(), native.data(), native.size());
    return Compare(a, b);
  }
  double error = 0, norm = 0, maximum = 0;
  size_t changed = 0;
  for (size_t i = 0; i < native.size(); i += 4) {
    const double a = vt::LoadUnaligned<float>(reference.data() + i);
    const double b = vt::LoadUnaligned<float>(native.data() + i);
    Require(std::isfinite(a) && std::isfinite(b), "nonfinite state value");
    changed += std::memcmp(reference.data() + i, native.data() + i, 4) != 0;
    const double delta = a - b;
    error += delta * delta; norm += a * a; maximum = std::max(maximum, std::abs(delta));
  }
  return {{"storage_exact", reference == native}, {"different_elements", changed},
          {"elements", reference.size() / 4}, {"max_abs", maximum},
          {"rel_l2", std::sqrt(error / std::max(norm, 1e-30))}};
}
// This is a real-weight, first-block state selection control. Repeating one
// frozen normalized row is deliberate teacher forcing, not a generated draft.
// Four independent ordinary decode rows keep every EXL3 projection at M=4.
void RunSpecControl(vllm::dense_attn::Dev device, const vllm::GdnLayerWeights& weights,
                    const vllm::HfConfig& config, const std::vector<unsigned char>& norm,
                    const std::vector<unsigned char>& initial_conv,
                    const std::vector<unsigned char>& initial_ssm, json& report) {
  const auto setting = [](const char* name, const char* fallback) {
    const char* value = std::getenv(name); return std::string(value ? value : fallback);
  };
  report["recurrence_settings"] = {{"VT_XPU_GDN_DECODE", setting("VT_XPU_GDN_DECODE", "auto")},
      {"VT_XPU_GDN_SPEC_SLM", setting("VT_XPU_GDN_SPEC_SLM", "1")},
      {"VT_XPU_GDN_SPEC_SLM_TYPED", setting("VT_XPU_GDN_SPEC_SLM_TYPED", "1")}};
  report["query_shapes"] = {{"ordinary", {1, 1, 1, 1}}, {"speculative", {4}}};
  constexpr int slots = 8, width = 6, selected = 2, spec_base = 3;
  const std::vector<int32_t> spec_slots{3, 1, 6, 4};
  const size_t ssm_bytes = initial_ssm.size();
  std::vector<unsigned char> input(4 * norm.size());
  for (int row = 0; row < 4; ++row) std::memcpy(input.data() + row * norm.size(), norm.data(), norm.size());
  vllm::dense_attn::DBuf h(device, vt::DType::kF16, {4, kHidden}, input.data());
  const auto download = [&](const vt::Tensor& tensor, size_t bytes) {
    std::vector<unsigned char> result(bytes);
    device.b.Copy(device.q, result.data(), tensor.data, bytes);
    device.b.Synchronize(device.q);
    return result;
  };
  const auto ssm_row = [&](const std::vector<unsigned char>& cache, int slot) {
    return std::vector<unsigned char>(cache.begin() + slot * ssm_bytes,
                                      cache.begin() + (slot + 1) * ssm_bytes);
  };
  const auto conv_window = [&](const std::vector<unsigned char>& cache, int slot, int length, int offset) {
    std::vector<unsigned char> result(initial_conv.size());
    for (int64_t channel = 0; channel < kConv; ++channel)
      std::memcpy(result.data() + channel * 6,
                  cache.data() + ((slot * kConv + channel) * length + offset) * 2, 6);
    return result;
  };
  const auto initial_cache = [&](bool spec) {
    const int length = spec ? width : 3, slot = spec ? spec_base : selected;
    std::vector<unsigned char> conv(slots * kConv * length * 2, 0), ssm(slots * ssm_bytes, 0);
    for (int64_t channel = 0; channel < kConv; ++channel)
      std::memcpy(conv.data() + (slot * kConv + channel) * length * 2,
                  initial_conv.data() + channel * 6, 6);
    std::memcpy(ssm.data() + slot * ssm_bytes, initial_ssm.data(), ssm_bytes);
    return std::make_pair(conv, ssm);
  };
  const auto evaluate = [&](vllm::GdnStateCache state, bool spec, int accepted, int position) {
    vllm::v1::StepInputs step;
    step.input_token_ids = {271, 271, 271, 271};
    step.positions = spec ? std::vector<int64_t>{position, position + 1, position + 2, position + 3}
                          : std::vector<int64_t>(4, position);
    step.query_start_loc = spec ? std::vector<int32_t>{0, 4} : std::vector<int32_t>{0, 1, 2, 3, 4};
    step.seq_lens = spec ? std::vector<int32_t>{position + 4} : std::vector<int32_t>(4, position + 1);
    step.slot_mapping = {{0, 1, 2, 3}};
    const auto attention = vllm::v1::MakeCommonAttentionMetadata(step, spec ? std::vector<int32_t>{0}
                                                                    : std::vector<int32_t>{0, 1, 2, 3}, 1, true, 0);
    vllm::v1::GDNAttentionMetadata gdn;
    gdn.num_actual_tokens = 4;
    if (spec) {
      gdn.num_spec_decodes = 1; gdn.num_spec_decode_tokens = 4;
      gdn.spec_state_indices_num_cols = 4; gdn.spec_state_indices_tensor = spec_slots;
      gdn.spec_query_start_loc = std::vector<int32_t>{0, 4};
      gdn.spec_sequence_masks = std::vector<uint8_t>{1};
      gdn.spec_token_indx = std::vector<int32_t>{0, 1, 2, 3};
      gdn.num_accepted_tokens = std::vector<int32_t>{accepted};
    } else {
      gdn.num_decodes = 4; gdn.num_decode_tokens = 4;
      gdn.non_spec_state_indices_tensor = std::vector<int32_t>{selected, 0, 5, 7};
    }
    return vllm::ReplayQwen3_5GdnBlockFp16ForDiagnostics(device.q, weights, config, h.t(), state,
        attention, gdn, std::vector<int32_t>(step.positions.begin(), step.positions.end()), 0);
  };
  std::vector<std::vector<unsigned char>> ref_conv, ref_ssm;
  std::vector<Row> ref_output;
  {
    auto [conv, ssm] = initial_cache(false);
    vllm::dense_attn::DBuf dc(device, vt::DType::kF16, {slots, kConv, 3}, conv.data());
    vllm::dense_attn::DBuf ds(device, vt::DType::kF32, {slots, 48, 128, 128}, ssm.data());
    vllm::GdnStateCache state; state.conv_state = dc.t(); state.ssm_state = ds.t();
    for (int token = 0; token < 4; ++token) {
      ref_output.push_back(evaluate(state, false, 1, 18 + token));
      ref_conv.push_back(conv_window(download(dc.t(), conv.size()), selected, 3, 0));
      ref_ssm.push_back(ssm_row(download(ds.t(), ssm.size()), selected));
    }
  }
  auto [conv, ssm] = initial_cache(true);
  vllm::dense_attn::DBuf dc(device, vt::DType::kF16, {slots, kConv, width}, conv.data());
  vllm::dense_attn::DBuf ds(device, vt::DType::kF32, {slots, 48, 128, 128}, ssm.data());
  vllm::GdnStateCache state; state.conv_state = dc.t(); state.ssm_state = ds.t();
  const auto first = evaluate(state, true, 1, 18);
  const auto provisional_conv = download(dc.t(), conv.size()), provisional_ssm = download(ds.t(), ssm.size());
  report["initial_output"] = Compare(ref_output[0], first);
  report["cases"] = json::array();
  bool exact = report["initial_output"]["storage_exact"];
  for (int accepted = 1; accepted <= 4; ++accepted) {
    const int index = accepted - 1;
    json result{{"accepted_drafts", index}, {"num_accepted_tokens", accepted},
      {"selected_ssm_slot", spec_slots[index]}, {"physical_m", 4},
      {"snapshot_conv", CompareState(ref_conv[index], conv_window(provisional_conv, spec_base, width, index), true)},
      {"snapshot_ssm", CompareState(ref_ssm[index], ssm_row(provisional_ssm, spec_slots[index]), false)}};
    // Roll back the provisional suffix by the production accepted selector.
    device.b.Copy(device.q, dc.t().data, provisional_conv.data(), provisional_conv.size());
    device.b.Copy(device.q, ds.t().data, provisional_ssm.data(), provisional_ssm.size());
    const auto correction = evaluate(state, true, accepted, 18 + accepted);
    auto [target_conv, target_ssm] = initial_cache(false);
    std::memcpy(target_conv.data() + selected * initial_conv.size(), ref_conv[index].data(), initial_conv.size());
    std::memcpy(target_ssm.data() + selected * ssm_bytes, ref_ssm[index].data(), ssm_bytes);
    vllm::dense_attn::DBuf tc(device, vt::DType::kF16, {slots, kConv, 3}, target_conv.data());
    vllm::dense_attn::DBuf ts(device, vt::DType::kF32, {slots, 48, 128, 128}, target_ssm.data());
    vllm::GdnStateCache target; target.conv_state = tc.t(); target.ssm_state = ts.t();
    const auto expected = evaluate(target, false, 1, 18 + accepted);
    result["correction_output"] = Compare(expected, correction);
    result["correction_conv"] = CompareState(conv_window(download(tc.t(), target_conv.size()), selected, 3, 0),
        conv_window(download(dc.t(), conv.size()), spec_base, width, 0), true);
    result["correction_ssm"] = CompareState(ssm_row(download(ts.t(), target_ssm.size()), selected),
        ssm_row(download(ds.t(), ssm.size()), spec_slots[0]), false);
    for (const char* field : {"snapshot_conv", "snapshot_ssm", "correction_output", "correction_conv", "correction_ssm"})
      exact &= result[field]["storage_exact"].get<bool>();
    report["cases"].push_back(result);
  }
  report["reference_tier_hits"] = vt::GetReferenceTierHits();
  report["scope"] = "first real EXL3 GDN block, repeated frozen token271 norm row, FP16 Conv/F32 SSM, M4 matched projections; teacher-forced accept0/1/2/all control; no whole-model/MTP serving qualification";
  report["all_compared_values_exact"] = exact;
  Require(exact && vt::GetReferenceTierHits() == 0, "typed first-block speculative state control differs");
  report["status"] = "DIAGNOSTIC";
}
void RunImagePrefill(const std::filesystem::path& model,const std::filesystem::path& capture,
                     const std::filesystem::path& output,json& report,bool repaired=false) {
  const auto config=vllm::LoadHfConfig((model/"config.json").string());
  Require(config.hidden_size==kHidden && config.num_hidden_layers==64,"wrong image replay model");
  const auto index=vllm::LoadSafetensorsIndex((model/"model.safetensors.index.json").string());
  std::map<std::string,vllm::SafetensorsFile> shards;
  const vllm::TensorResolver get=[&](const std::string& name)->const vllm::StTensor& {
    const auto& file=index.at(name);
    Require(std::filesystem::path(file).filename()==file,"unsafe replay shard");
    if(!shards.count(file))shards.emplace(file,vllm::SafetensorsFile::Open((model/file).string()));
    const auto& tensor=shards.at(file).Get(name);
    if(name!="model.language_model.embed_tokens.weight")report["weights"][name]={
      {"dtype",tensor.dtype},{"shape",tensor.shape},{"sha256",Hash(tensor.data,tensor.nbytes)}};
    return tensor;
  };
  auto weights=vllm::LoadQwen3_5DenseGdnFp16ForDiagnostics(get,"model.language_model.layers.0.");
  auto norm=vllm::dense_loaders::LoadF16Direct(get,"model.language_model.layers.0.input_layernorm.weight");
  const auto& embedding=get("model.language_model.embed_tokens.weight");
  Require(embedding.dtype=="BF16" && embedding.shape==std::vector<int64_t>({248320,kHidden}),
          "wrong original embedding geometry");
  QueueOwner queue;queue.value=vt::CreateQueue({vt::DeviceType::kXPU,0});auto& q=*queue.value;
  vllm::dense_attn::Dev d{vt::GetBackend(q.device),q,vt::DType::kF16};
  vllm::dense_attn::DBuf norm_weight(d,vt::DType::kF16,{kHidden},norm.bytes.data());
  std::vector<unsigned char> previous_conv,previous_ssm;
  Row previous_normalized,previous_output;
  report["cases"]=json::array();
  for(int arm=0;arm<2;++arm) {
    const auto md=ReadJson(capture/("prefix-"+std::to_string(arm)+".json"));
    Require(md.at("seq_len")==224 && md.at("prompt_tokens")==224 && md.at("own_image_features")==1 &&
            md.at("mrope_delta")==-176 && md.at("output_prefix").empty(),"wrong selected image checkpoint");
    const auto ids=md.at("input_token_ids").get<std::vector<int32_t>>();
    const auto starts=md.at("query_start_loc").get<std::vector<int32_t>>();
    const auto row=md.at("row").get<int>();const auto slot=md.at("gdn_slot").get<int>();
    Require((arm==0 && ids.size()==224 && row==0) || (arm==1 && ids.size()==507 && row==3),
            "wrong frozen prefill geometry");
    const int begin=starts.at(row),end=starts.at(row+1);
    Require(end-begin==224 && std::vector<int32_t>(ids.begin()+begin,ids.begin()+end)==
            md.at("token_prefix").get<std::vector<int32_t>>(),"wrong selected prompt query");
    const auto& feature=md.at("image_features").at(0);
    Require(md.at("image_features").size()==1 && feature.at("request_id")==md.at("request_id") &&
            feature.at("offset")==4 && feature.at("length")==192,"wrong selected visual span");
    const auto image=StateBlob(capture,md,"image0",1,json::array({192,kHidden}));
    Row host(ids.size()*kHidden);
    for(size_t r=0;r<ids.size();++r) {
      Require(ids[r]>=0 && ids[r]<248320,"invalid embedding token ID");
      const auto* source=embedding.data+int64_t(ids[r])*kHidden*2;
      report["embedding_rows"][std::to_string(ids[r])]=Hash(source,kHidden*2);
      for(int64_t col=0;col<kHidden;++col)host[r*kHidden+col]=vt::F32ToF16(
          vt::BF16ToF32(vt::LoadUnaligned<uint16_t>(source+col*2)));
    }
    std::memcpy(host.data()+(begin+4)*kHidden,image.data(),image.size());
    vllm::dense_attn::DBuf input(d,vt::DType::kF16,{int64_t(ids.size()),kHidden},host.data()),
                         normalized(d,vt::DType::kF16,{int64_t(ids.size()),kHidden});
    vt::RmsNorm(q,normalized.t(),input.t(),norm_weight.t(),vt::RmsNormArgs{1e-6f,true});
    Row selected_normalized(224*kHidden);
    d.b.Copy(q,selected_normalized.data(),static_cast<const char*>(normalized.t().data)+begin*kHidden*2,
             selected_normalized.size()*2);d.b.Synchronize(q);
    if(arm)Require(previous_normalized==selected_normalized,"selected normalized rows changed with geometry");
    else previous_normalized=selected_normalized;
    const auto indices=md.at("gdn_indices").get<std::vector<int32_t>>();
    Require(indices.size()==size_t(arm ? 4 : 1) && slot>=0 && slot<4,"wrong state owner geometry");
    std::vector<unsigned char> conv(4*kConv*3*2,0),ssm(4*48*128*128*4,0);
    vllm::dense_attn::DBuf dc(d,vt::DType::kF16,{4,kConv,3},conv.data()),
                         ds(d,vt::DType::kF32,{4,48,128,128},ssm.data());
    vllm::GdnStateCache state;state.conv_state=dc.t();state.ssm_state=ds.t();
    vllm::v1::StepInputs step;step.input_token_ids=ids;step.query_start_loc=starts;
    step.positions=md.at("positions").get<std::vector<int64_t>>();
    step.seq_lens=md.at("seq_lens").get<std::vector<int32_t>>();
    step.slot_mapping={md.at("kv_write_slots").get<std::vector<int64_t>>()};
    auto attention=vllm::v1::MakeCommonAttentionMetadata(step,
      md.at("kv_block_table").get<std::vector<int32_t>>(),md.at("kv_block_table_cols"),true,0);
    auto state_attention=vllm::v1::MakeCommonAttentionMetadata(step,indices,1,true,0);
    vllm::v1::GDNAttentionMetadataBuilder builder;const auto gm=builder.build(0,state_attention);
    const auto prefill_slot=std::find(gm.prefill_state_indices->begin(),gm.prefill_state_indices->end(),slot);
    Require(prefill_slot!=gm.prefill_state_indices->end() &&
            !gm.prefill_has_initial_state->at(prefill_slot-gm.prefill_state_indices->begin()),
            "selected new image prefill must start from zero state");
    std::vector<int32_t> positions(step.positions.begin(),step.positions.end());
    const vllm::actdump::LayerScope observer(vllm::actdump::BeginStep(),0);
    const auto out=vllm::ReplayQwen3_5GdnBlockFp16ForDiagnostics(q,weights,config,normalized.t(),state,
                                                             attention,gm,positions,begin);
    std::vector<unsigned char> got_conv(kConv*3*2),got_ssm(48*128*128*4);
    d.b.Copy(q,got_conv.data(),static_cast<const char*>(dc.t().data)+slot*got_conv.size(),got_conv.size());
    d.b.Copy(q,got_ssm.data(),static_cast<const char*>(ds.t().data)+slot*got_ssm.size(),got_ssm.size());
    d.b.Synchronize(q);
    const auto want_conv=StateBlob(capture,md,"gdn0-conv",1,json::array({kConv,3}));
    const auto want_ssm=StateBlob(capture,md,"gdn0-ssm",0,json::array({48,128,128}));
    const auto write=[&](const char* label,const void* data,size_t bytes) {
      const auto file=output.stem().string()+"-"+std::to_string(arm)+"-"+label+".bin";
      Require(!std::filesystem::exists(output.parent_path()/file),"preserve replay payload");
      std::ofstream f(output.parent_path()/file,std::ios::binary);f.write(static_cast<const char*>(data),bytes);
      Require(f.good(),"cannot write replay payload");return json{{"file",file},{"bytes",bytes},{"sha256",Hash(data,bytes)}};
    };
    json entry{{"arm",arm},{"actual_token_rows",ids.size()},{"selected_begin",begin},{"selected_slot",slot},
      {"selected_initial_state","zero"},{"normalized",write("normalized",selected_normalized.data(),selected_normalized.size()*2)},
      {"ssm",write("ssm",got_ssm.data(),got_ssm.size())},{"conv",write("conv",got_conv.data(),got_conv.size())},
      {"first_image_query_output",write("output",out.data(),out.size()*2)},
      {"actual_conv",CompareState(want_conv,got_conv,true)},{"actual_ssm",CompareState(want_ssm,got_ssm,false)},
      {"decode_requests",gm.num_decodes},{"prefill_requests",gm.num_prefills}};
    if(arm) {
      entry["shared_input_conv_vs_c1"]=CompareState(previous_conv,got_conv,true);
      entry["shared_input_ssm_vs_c1"]=CompareState(previous_ssm,got_ssm,false);
      entry["first_query_output_vs_c1"]=Compare(previous_output,out);
    } else {previous_conv=got_conv;previous_ssm=got_ssm;previous_output=out;}
    report["cases"].push_back(entry);
    if (arm && repaired) {
      // Preserve the old mixed anchor as an observed difference. The repair
      // must compute the independently verified C1/reference state instead.
      Require(previous_conv==got_conv && previous_ssm==got_ssm,
              "repaired mixed prefill does not match the reference-verified C1 state");
    } else {
      Require(want_conv==got_conv && want_ssm==got_ssm,"local first GDN does not reproduce actual selected prefill state");
    }
  }
  report["reference_tier_hits"]=vt::GetReferenceTierHits();
  Require(vt::GetReferenceTierHits()==0,"prefill replay reference fallback");
  report["scope"]="first GDN only, complete identical selected normalized image input, selected prefill starts from zero; sibling caches are zero diagnostic owners, not actual earlier sibling histories; no full LM or quality claim";
  report["computed_mixed_repair_control"]=repaired;
  report["status"]="DIAGNOSTIC";
}

void RunBlock(const std::filesystem::path& model, const std::filesystem::path& capture,
              const std::filesystem::path& norm_row, const std::filesystem::path& output, json& report, bool spec = false) {
  QueueOwner queue;
  const auto norm_bytes = Read(norm_row, kHidden * 2);
  Require(norm_bytes.size() == kHidden * 2 && Hash(norm_bytes.data(), norm_bytes.size()) ==
          "34eb05c8f923ba017ce333f0ce43d40bd623a2e6b0088631e7f20110da2a1b88",
          "actual normalized token271 row differs");
  const auto config = vllm::LoadHfConfig((model / "config.json").string());
  Require(config.hidden_size == kHidden && config.linear_num_value_heads == 48 &&
          config.linear_num_key_heads == 16 && config.linear_key_head_dim == 128 &&
          config.linear_value_head_dim == 128 && config.linear_conv_kernel_dim == 4,
          "wrong actual block configuration");
  const auto index = vllm::LoadSafetensorsIndex((model / "model.safetensors.index.json").string());
  std::map<std::string, vllm::SafetensorsFile> shards;
  const vllm::TensorResolver get = [&](const std::string& name) -> const vllm::StTensor& {
    const auto& file = index.at(name);
    Require(std::filesystem::path(file).filename() == file, "unsafe block checkpoint shard");
    if (!shards.count(file)) shards.emplace(file, vllm::SafetensorsFile::Open((model / file).string()));
    const auto& tensor = shards.at(file).Get(name);
    report["weights"][name] = {{"dtype", tensor.dtype}, {"shape", tensor.shape},
                              {"sha256", Hash(tensor.data, tensor.nbytes)}};
    return tensor;
  };
  auto weights = vllm::LoadQwen3_5DenseGdnFp16ForDiagnostics(get, "model.language_model.layers.0.");
  Require(weights.v_head_perm_key_heads == 0, "unexpected deferred head permutation");
  const auto metadata0 = ReadJson(capture / "prefix-0.json");
  const auto metadata2 = ReadJson(capture / "prefix-2.json");
  const auto initial_conv = StateBlob(capture, metadata0, "gdn0-conv", 1, json::array({kConv, 3}));
  const auto initial_ssm = StateBlob(capture, metadata0, "gdn0-ssm", 0, json::array({48, 128, 128}));
  Require(initial_conv == StateBlob(capture, metadata2, "gdn0-conv", 1, json::array({kConv, 3})) &&
          initial_ssm == StateBlob(capture, metadata2, "gdn0-ssm", 0, json::array({48, 128, 128})),
          "original paired initial states differ");
  if (spec) {
    queue.value = vt::CreateQueue({vt::DeviceType::kXPU, 0});
    auto& q = *queue.value;
    RunSpecControl({vt::GetBackend(q.device), q, vt::DType::kF16}, weights, config,
                   norm_bytes, initial_conv, initial_ssm, report);
    return;
  }
  std::vector<json> metadata;
  std::vector<std::vector<unsigned char>> expected_conv, expected_ssm;
  for (const int number : {3, 1}) {
    metadata.push_back(ReadJson(capture / ("prefix-" + std::to_string(number) + ".json")));
    expected_conv.push_back(StateBlob(capture, metadata.back(), "gdn0-conv", 1, json::array({kConv, 3})));
    expected_ssm.push_back(StateBlob(capture, metadata.back(), "gdn0-ssm", 0, json::array({48, 128, 128})));
  }
  queue.value = vt::CreateQueue({vt::DeviceType::kXPU, 0});
  auto& q = *queue.value;
  vllm::dense_attn::Dev device{vt::GetBackend(q.device), q, vt::DType::kF16};
  std::vector<unsigned char> baseline_conv, baseline_ssm;
  Row baseline_output;
  report["cases"] = json::array();
  for (size_t arm = 0; arm < metadata.size(); ++arm) {
    const auto& entry = metadata[arm];
    const int64_t m = entry.at("actual_token_rows"), slot = entry.at("gdn_slot");
    Require((arm == 0 && m == 1) || (arm == 1 && m == 509), "wrong paired block shape");
    const auto indices = entry.at("gdn_indices").get<std::vector<int32_t>>();
    Require(!indices.empty() && indices.size() <= 4 &&
            std::all_of(indices.begin(), indices.end(), [](int32_t value) { return value >= 0; }),
            "invalid captured GDN indices");
    const int slots = *std::max_element(indices.begin(), indices.end()) + 1;
    Require(slots > slot && slots <= 4, "invalid block state slot bound");
    std::vector<unsigned char> conv(slots * initial_conv.size(), 0), ssm(slots * initial_ssm.size(), 0);
    std::memcpy(conv.data() + slot * initial_conv.size(), initial_conv.data(), initial_conv.size());
    std::memcpy(ssm.data() + slot * initial_ssm.size(), initial_ssm.data(), initial_ssm.size());
    std::vector<unsigned char> input(m * norm_bytes.size());
    for (int64_t row = 0; row < m; ++row) std::memcpy(input.data() + row * norm_bytes.size(), norm_bytes.data(), norm_bytes.size());
    vllm::dense_attn::DBuf h(device, vt::DType::kF16, {m, kHidden}, input.data());
    vllm::dense_attn::DBuf dc(device, vt::DType::kF16, {slots, kConv, 3}, conv.data());
    vllm::dense_attn::DBuf ds(device, vt::DType::kF32, {slots, 48, 128, 128}, ssm.data());
    vllm::GdnStateCache state; state.conv_state = dc.t(); state.ssm_state = ds.t();
    vllm::v1::StepInputs step;
    step.input_token_ids = entry.at("input_token_ids").get<std::vector<int32_t>>();
    step.positions = entry.at("positions").get<std::vector<int64_t>>();
    step.query_start_loc = entry.at("query_start_loc").get<std::vector<int32_t>>();
    step.seq_lens = entry.at("seq_lens").get<std::vector<int32_t>>();
    step.slot_mapping = {entry.at("kv_write_slots").get<std::vector<int64_t>>()};
    auto attention = vllm::v1::MakeCommonAttentionMetadata(step,
        entry.at("kv_block_table").get<std::vector<int32_t>>(), entry.at("kv_block_table_cols"), true, 0);
    auto state_attention = vllm::v1::MakeCommonAttentionMetadata(step, indices, 1, true, 0);
    vllm::v1::GDNAttentionMetadataBuilder builder;
    const auto gdn = builder.build(0, state_attention);
    std::vector<int32_t> positions(step.positions.begin(), step.positions.end());
    const int64_t selected = step.query_start_loc.at(entry.at("row").get<size_t>());
    auto out = vllm::ReplayQwen3_5GdnBlockFp16ForDiagnostics(q, weights, config, h.t(), state,
                                                           attention, gdn, positions, selected);
    std::vector<unsigned char> actual_conv(initial_conv.size()), actual_ssm(initial_ssm.size());
    device.b.Copy(q, actual_conv.data(), static_cast<const char*>(dc.t().data) + slot * actual_conv.size(), actual_conv.size());
    device.b.Copy(q, actual_ssm.data(), static_cast<const char*>(ds.t().data) + slot * actual_ssm.size(), actual_ssm.size());
    device.b.Synchronize(q);
    if (arm == 0) { baseline_conv = actual_conv; baseline_ssm = actual_ssm; baseline_output = out; }
    json value{{"m", m}, {"selected_state_slot", slot}, {"selected_token_row", selected},
        {"decode_requests", gdn.num_decodes}, {"prefill_requests", gdn.num_prefills},
        {"conv_vs_actual_serving", CompareState(expected_conv[arm], actual_conv, true)},
        {"ssm_vs_actual_serving", CompareState(expected_ssm[arm], actual_ssm, false)},
        {"conv_vs_m1", CompareState(baseline_conv, actual_conv, true)},
        {"ssm_vs_m1", CompareState(baseline_ssm, actual_ssm, false)},
        {"output_vs_m1", Compare(baseline_output, out)},
        {"output", Save(output.parent_path() / (output.stem().string() + "-m" + std::to_string(m) + ".f16"), out)}};
    report["cases"].push_back(value);
  }
  report["reference_tier_hits"] = vt::GetReferenceTierHits();
  Require(vt::GetReferenceTierHits() == 0, "block replay used a reference fallback");
  report["scope"] = "first GDN block from identical selected states; sibling norm rows duplicated with fresh zero states; no whole-model or MTP gate";
  report["status"] = "DIAGNOSTIC";
}
}  // namespace
int main(int argc, char** argv) {
  if(argc==5 && (std::string(argv[1])=="--image-prefill" ||
                std::string(argv[1])=="--image-prefill-repaired")) {
    if(std::filesystem::exists(argv[4]))return 2;
    json report{{"status","FAIL"}};
    try {RunImagePrefill(argv[2],argv[3],argv[4],report,std::string(argv[1])=="--image-prefill-repaired");}
    catch(const std::exception& error){report["error"]=error.what();}
    std::ofstream file(argv[4]);file<<report.dump(2)<<'\n';
    return file.good() && report.at("status")=="DIAGNOSTIC" ? 0 : 1;
  }
  if (argc == 6 && (std::string(argv[1]) == "--block" || std::string(argv[1]) == "--spec-block")) {
    const std::filesystem::path output(argv[5]);
    if (std::filesystem::exists(output)) return 2;
    json report{{"status", "FAIL"}};
    try { RunBlock(argv[2], argv[3], argv[4], output, report, std::string(argv[1]) == "--spec-block"); }
    catch (const std::exception& error) { report["error"] = error.what(); }
    std::ofstream file(output); file << report.dump(2) << '\n';
    if (!file.good()) return 2;
    return report.at("status") == "DIAGNOSTIC" ? 0 : 1;
  }
  if (argc != 4 && argc != 5) {
    std::cerr << "usage: native-vision-first-gdn-compare MODEL PAIRED_CAPTURE OUTPUT_JSON [M4_FIRST_DECODE_CAPTURE]\n";
    return 2;
  }
  const std::filesystem::path output(argv[3]);
  if (std::filesystem::exists(output)) { std::cerr << "preserve existing evidence\n"; return 2; }
  json report{{"status", "FAIL"}, {"scope", "same token271 first-layer RMSNorm/QKVZ at actual shapes; no parity waiver"}};
  try { Run(argv[1], argv[2], output, argc == 5 ? argv[4] : "", report); }
  catch (const std::exception& error) { report["error"] = error.what(); }
  std::ofstream file(output);
  file << report.dump(2) << '\n';
  if (!file.good()) return 2;
  return report.at("status") == "DIAGNOSTIC" ? 0 : 1;
}
