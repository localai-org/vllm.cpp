#include <doctest/doctest.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include "vt/xpu_test_helpers.h"
#include "vt/xpu.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"
#include "support/exl3_external_artifacts.h"
#include "support/native_vision_external_artifacts.h"

TEST_CASE("XPU vision MTP: supplied real image embeddings survive repeat and refusal" *
          doctest::test_suite("vision-external") * doctest::skip(
              exl3_test::OptionalExternalEnvironmentMissing({"VT_B70_EXL3_MODEL", "VT_B70_VISION_MTP_CAPTURE"}))) {
  const char* model_path = exl3_test::CaseExternalEnvironment("VT_B70_EXL3_MODEL");
  const char* capture_path = exl3_test::CaseExternalEnvironment("VT_B70_VISION_MTP_CAPTURE");
  const std::filesystem::path directory(capture_path);
  const auto record = native_vision_test::ReferenceDocument(directory / "capture.json");
  REQUIRE(record.at("status") == "PASS");
  auto tensor_bytes = [&](const std::string& name, const std::string& dtype,
                          const std::vector<int64_t>& shape, size_t width) {
    const auto& captures = record.at("worker").at("captures");
    const auto found = std::find_if(captures.begin(), captures.end(), [&](const auto& item) {
      return item.at("name") == name;
    });
    REQUIRE(found != captures.end());
    REQUIRE(found->contains("sha256"));
    REQUIRE(found->at("dtype") == dtype);
    REQUIRE(found->at("shape").template get<std::vector<int64_t>>() == shape);
    const auto count = std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
    std::vector<unsigned char> bytes(static_cast<size_t>(count) * width);
    std::ifstream file(directory / found->at("file").template get<std::string>(), std::ios::binary);
    REQUIRE(file.good());
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    REQUIRE(file.gcount() == static_cast<std::streamsize>(bytes.size()));
    REQUIRE(file.peek() == std::char_traits<char>::eof());
    return bytes;
  };
  constexpr int64_t tokens = 234, hidden_size = 5120;
  const auto ids_bytes = tensor_bytes("1-draft-forward-input_ids", "torch.int32", {tokens}, 4);
  const auto position_bytes = tensor_bytes("1-draft-forward-positions", "torch.int64", {tokens}, 8);
  const auto feedback_bytes = tensor_bytes("1-draft-forward-hidden_states", "torch.float16", {tokens, hidden_size}, 2);
  const auto merged_bytes = tensor_bytes("1-draft-forward-inputs_embeds", "torch.float16", {tokens, hidden_size}, 2);
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  std::vector<int32_t> ids(tokens), positions(tokens);
  std::memcpy(ids.data(), ids_bytes.data(), ids_bytes.size());
  for (size_t row = 0; row < positions.size(); ++row) {
    int64_t position;
    std::memcpy(&position, position_bytes.data() + row * 8, 8);
    REQUIRE(position == static_cast<int64_t>(row));
    positions[row] = static_cast<int32_t>(position);
  }
  const auto config = vllm::LoadHfConfig((std::filesystem::path(model_path) / "config.json").string());
  REQUIRE(config.hidden_size == hidden_size);
  std::vector<vllm::SafetensorsFile> shards;
  for (const auto* file : {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"})
    shards.push_back(vllm::SafetensorsFile::Open((std::filesystem::path(model_path) / file).string()));
  const auto weights = vllm::LoadQwen3_5MTP(shards, config, vllm::Qwen3_5MTPKind::kDense, vt::DeviceType::kXPU);
  REQUIRE(weights.IsExl3());
  vllm::Qwen3_5DenseWeights target;
  const vllm::TensorResolver get = [&](const std::string& name) -> const vllm::StTensor& { return shards[0].Get(name); };
  target.embed_tokens = vllm::dense_loaders::LoadF16Direct(get, "model.language_model.embed_tokens.weight");
  const vllm::Qwen3_5MTPModel draft(weights, target, config);
  xpu_test::Buffer feedback(gpu.q, vt::DType::kF16, {tokens, hidden_size});
  xpu_test::Buffer merged(gpu.q, vt::DType::kF16, {tokens, hidden_size});
  feedback.upload(feedback_bytes.data());
  merged.upload(merged_bytes.data());
  xpu_test::Buffer storage(gpu.q, vt::DType::kI8, {1600 * 4 * 512});
  const std::vector<unsigned char> poison(storage.bytes, 0x7f);
  storage.upload(poison.data());
  vllm::PagedKvCache cache;
  cache.data = storage.tensor.data; cache.dtype = vt::DType::kI8;
  cache.num_blocks = 1; cache.block_size = 1600; cache.num_kv_heads = 4; cache.head_size = 256;
  cache.fp8_kind = vt::Fp8KVCacheDataType::kFp8E4M3;
  vllm::v1::CommonAttentionMetadata metadata;
  metadata.num_reqs = 1; metadata.num_actual_tokens = tokens;
  metadata.query_start_loc = metadata.query_start_loc_cpu = {0, tokens};
  metadata.seq_lens = metadata.seq_lens_cpu = {tokens}; metadata.num_computed_tokens_cpu = {0};
  metadata.max_query_len = tokens; metadata.max_seq_len = tokens;
  metadata.block_table_num_cols = 1; metadata.block_table_tensor = {0};
  metadata.slot_mapping.resize(tokens);
  std::iota(metadata.slot_mapping.begin(), metadata.slot_mapping.end(), int64_t{0});
  auto download = [&](const vt::Tensor& tensor) {
    std::vector<unsigned char> bytes(static_cast<size_t>(tokens * hidden_size * 2));
    auto& backend = vt::GetBackend(gpu.q.device);
    backend.Copy(gpu.q, bytes.data(), tensor.data, bytes.size()); backend.Synchronize(gpu.q);
    return bytes;
  };
  const auto first = draft.ForwardPaged(ids, positions, feedback.tensor, metadata, cache, gpu.q, 0, &merged.tensor);
  REQUIRE(first.tensor.dtype == vt::DType::kF16);
  const auto first_bytes = download(first.tensor);
  bool finite = true;
  for (size_t offset = 0; offset < first_bytes.size(); offset += 2) {
    uint16_t bits;
    std::memcpy(&bits, first_bytes.data() + offset, 2);
    finite = finite && std::isfinite(vt::F16ToF32(bits));
  }
  CHECK(finite);

  const auto first_cache = storage.download();
  storage.upload(poison.data());
  const auto second = draft.ForwardPaged(ids, positions, feedback.tensor, metadata, cache, gpu.q, 0, &merged.tensor);
  CHECK(download(second.tensor) == first_bytes);
  CHECK(storage.download() == first_cache);
  CHECK(download(first.tensor) == first_bytes);  // output owners survive another forward
  CHECK(merged.download() == merged_bytes);
  CHECK(feedback.download() == feedback_bytes);
  auto invalid = merged.tensor;
  invalid.shape[0]--;
  CHECK_THROWS(draft.ForwardPaged(ids, positions, feedback.tensor, metadata, cache, gpu.q, 0, &invalid));
  CHECK(storage.download() == first_cache);
  auto changed = merged_bytes;
  std::fill(changed.begin() + 3 * hidden_size * 2, changed.begin() + 195 * hidden_size * 2, 0);
  merged.upload(changed.data()); storage.upload(poison.data());
  const auto omitted_image = draft.ForwardPaged(ids, positions, feedback.tensor, metadata, cache, gpu.q, 0, &merged.tensor);
  CHECK(download(omitted_image.tensor) != first_bytes);
  CHECK(vt::GetReferenceTierHits() == 0);
}
