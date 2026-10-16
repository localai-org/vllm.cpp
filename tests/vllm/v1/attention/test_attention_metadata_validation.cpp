#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "vllm/v1/attention/metadata_validation.h"

namespace {

using vllm::v1::CommonAttentionMetadata;
using vllm::v1::ValidateAttentionPageSlots;
using vllm::v1::ValidateAttentionUploadMetadata;

CommonAttentionMetadata MakePages(int context, int query, int block_size,
                                  std::vector<int32_t>& positions) {
  const int length = context + query;
  const int pages = (length + block_size - 1) / block_size;
  CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = query;
  am.query_start_loc = {0, query};
  am.seq_lens = {length};
  am.block_table_num_cols = pages;
  for (int page = 0; page < pages; ++page)
    am.block_table_tensor.push_back(pages - 1 - page);
  for (int i = 0; i < query; ++i) {
    const int position = context + i;
    positions.push_back(position);
    am.slot_mapping.push_back(
        am.block_table_tensor[position / block_size] * block_size +
        position % block_size);
  }
  return am;
}

}  // namespace

TEST_CASE("attention metadata upload uses exact host array shapes") {
  std::vector<int32_t> positions;
  auto am = MakePages(128, 1, 128, positions);
  CHECK_NOTHROW(ValidateAttentionUploadMetadata(am, 1));
  am.block_table_tensor.push_back(0);
  CHECK_THROWS_AS(ValidateAttentionUploadMetadata(am, 1), std::invalid_argument);
  am.block_table_tensor.pop_back();
  am.query_start_loc.back() = 2;
  CHECK_THROWS_AS(ValidateAttentionUploadMetadata(am, 1), std::invalid_argument);
  am.query_start_loc.back() = 1;
  am.slot_mapping[0] = -2;
  CHECK_THROWS_AS(ValidateAttentionUploadMetadata(am, 1), std::invalid_argument);
}

TEST_CASE("attention metadata maps reads to writes over page boundaries") {
  constexpr int kBlock = 128;
  const std::vector<int> lengths{1, 63, 64, 65, 127, 128, 129,
                                 255, 256, 257, 4095, 4096, 4097};
  for (const int length : lengths) {
    for (const int context : {0, 1, 63, 64, 127, 128, 129}) {
      if (context >= length) continue;
      std::vector<int32_t> positions;
      const auto am = MakePages(context, length - context, kBlock, positions);
      const int blocks = am.block_table_num_cols;
      REQUIRE_NOTHROW(ValidateAttentionPageSlots(am, positions, kBlock, blocks));
      // Every physical slot carries its own nonrepeating value. The page table
      // must read the exact slot written for each query token.
      std::vector<int64_t> cache(static_cast<size_t>(blocks) * kBlock);
      for (size_t i = 0; i < cache.size(); ++i) cache[i] = static_cast<int64_t>(i) + 17;
      for (size_t i = 0; i < positions.size(); ++i) {
        const int position = positions[i];
        const int read_slot = am.block_table_tensor[position / kBlock] * kBlock +
                              position % kBlock;
        CHECK(cache[read_slot] == cache[am.slot_mapping[i]]);
      }
    }
  }
}

TEST_CASE("attention metadata permits shared prefixes and inert graph rows") {
  CommonAttentionMetadata am;
  am.num_reqs = 3;
  am.num_actual_tokens = 5;
  am.query_start_loc = {0, 2, 4, 5};
  am.seq_lens = {129, 129, 1};
  am.block_table_num_cols = 2;
  am.block_table_tensor = {2, 3, 2, 4, 0, 0};
  am.slot_mapping = {2 * 128 + 127, 3 * 128, 2 * 128 + 127, 4 * 128, -1};
  const std::vector<int32_t> positions{127, 128, 127, 128, 0};
  CHECK_NOTHROW(ValidateAttentionPageSlots(am, positions, 128, 5));
}

TEST_CASE("attention metadata rejects missing and mismatched pages") {
  std::vector<int32_t> positions;
  auto am = MakePages(128, 1, 128, positions);
  am.block_table_num_cols = 1;
  am.block_table_tensor.resize(1);
  CHECK_THROWS_AS(ValidateAttentionPageSlots(am, positions, 128, 2),
                  std::invalid_argument);
  am.block_table_num_cols = 2;
  am.block_table_tensor.push_back(1);
  am.slot_mapping[0] = 0;
  CHECK_THROWS_AS(ValidateAttentionPageSlots(am, positions, 128, 2),
                  std::invalid_argument);
  am.slot_mapping[0] = 128;
  am.block_table_tensor[1] = 2;
  CHECK_THROWS_AS(ValidateAttentionPageSlots(am, positions, 128, 2),
                  std::invalid_argument);
}
