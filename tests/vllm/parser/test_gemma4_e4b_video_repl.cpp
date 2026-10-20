// Tests for Gemma4 video token replacement sequence builder.
// Mirrors Python get_video_repl in gemma4_mm.py:412-447.

#include <doctest/doctest.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "vllm/model_executor/models/gemma4.h"

namespace {

// Simple mock encoder: maps each character to its ASCII value.
// The real tokenizer encodes "00:00 " differently, but the test
// verifies structure (timestamp formatting, prefix spacing, boi/video/eoi
// sequence) without needing the real tokenizer.
std::vector<int32_t> ascii_encode(std::string_view text) {
  std::vector<int32_t> result;
  result.reserve(text.size());
  for (char c : text) result.push_back(static_cast<int32_t>(static_cast<unsigned char>(c)));
  return result;
}

}  // namespace

TEST_CASE("Gemma4 video repl: single frame at t=0") {
  std::vector<double> timestamps = {0.0};
  std::vector<int32_t> soft_tokens = {70};
  const int32_t boi = 255999, eoi = 258882, video = 258884;

  auto result = vllm::Gemma4VideoReplacement(
      timestamps, soft_tokens, boi, eoi, video, ascii_encode);

  // Expected: encode("00:00 ") + [boi, video*70, eoi]
  std::vector<int32_t> expected;
  auto prefix = ascii_encode("00:00 ");
  expected.insert(expected.end(), prefix.begin(), prefix.end());
  expected.push_back(boi);
  expected.insert(expected.end(), 70, video);
  expected.push_back(eoi);

  CHECK(result == expected);
}

TEST_CASE("Gemma4 video repl: two frames at t=0 and t=5") {
  std::vector<double> timestamps = {0.0, 5.0};
  std::vector<int32_t> soft_tokens = {70, 70};
  const int32_t boi = 255999, eoi = 258882, video = 258884;

  auto result = vllm::Gemma4VideoReplacement(
      timestamps, soft_tokens, boi, eoi, video, ascii_encode);

  // Frame 0: encode("00:00 ") + [boi, video*70, eoi]
  // Frame 1: encode(" 00:05 ") + [boi, video*70, eoi]
  std::vector<int32_t> expected;
  auto p0 = ascii_encode("00:00 ");
  expected.insert(expected.end(), p0.begin(), p0.end());
  expected.push_back(boi);
  expected.insert(expected.end(), 70, video);
  expected.push_back(eoi);

  auto p1 = ascii_encode(" 00:05 ");
  expected.insert(expected.end(), p1.begin(), p1.end());
  expected.push_back(boi);
  expected.insert(expected.end(), 70, video);
  expected.push_back(eoi);

  CHECK(result == expected);
}

TEST_CASE("Gemma4 video repl: timestamp formatting minutes rollover") {
  std::vector<double> timestamps = {65.0};
  std::vector<int32_t> soft_tokens = {10};
  const int32_t boi = 255999, eoi = 258882, video = 258884;

  auto result = vllm::Gemma4VideoReplacement(
      timestamps, soft_tokens, boi, eoi, video, ascii_encode);

  // 65 seconds -> 01:05
  std::vector<int32_t> expected;
  auto prefix = ascii_encode("01:05 ");
  expected.insert(expected.end(), prefix.begin(), prefix.end());
  expected.push_back(boi);
  expected.insert(expected.end(), 10, video);
  expected.push_back(eoi);

  CHECK(result == expected);
}

TEST_CASE("Gemma4 video repl: timestamp formatting 125 seconds") {
  std::vector<double> timestamps = {125.0};
  std::vector<int32_t> soft_tokens = {10};
  const int32_t boi = 255999, eoi = 258882, video = 258884;

  auto result = vllm::Gemma4VideoReplacement(
      timestamps, soft_tokens, boi, eoi, video, ascii_encode);

  // 125 seconds -> 02:05
  std::vector<int32_t> expected;
  auto prefix = ascii_encode("02:05 ");
  expected.insert(expected.end(), prefix.begin(), prefix.end());
  expected.push_back(boi);
  expected.insert(expected.end(), 10, video);
  expected.push_back(eoi);

  CHECK(result == expected);
}

TEST_CASE("Gemma4 video repl: empty frames produces empty result") {
  std::vector<double> timestamps;
  std::vector<int32_t> soft_tokens;
  const int32_t boi = 255999, eoi = 258882, video = 258884;

  auto result = vllm::Gemma4VideoReplacement(
      timestamps, soft_tokens, boi, eoi, video, ascii_encode);

  CHECK(result.empty());
}

TEST_CASE("Gemma4 video repl: fractional timestamp truncates") {
  std::vector<double> timestamps = {5.7};
  std::vector<int32_t> soft_tokens = {10};
  const int32_t boi = 255999, eoi = 258882, video = 258884;

  auto result = vllm::Gemma4VideoReplacement(
      timestamps, soft_tokens, boi, eoi, video, ascii_encode);

  // 5.7 seconds -> int(5.7 // 60) = 0, int(5.7 % 60) = 5 -> "00:05"
  std::vector<int32_t> expected;
  auto prefix = ascii_encode("00:05 ");
  expected.insert(expected.end(), prefix.begin(), prefix.end());
  expected.push_back(boi);
  expected.insert(expected.end(), 10, video);
  expected.push_back(eoi);

  CHECK(result == expected);
}

TEST_CASE("Gemma4 video repl: unequal lengths throws") {
  std::vector<double> timestamps = {0.0, 5.0};
  std::vector<int32_t> soft_tokens = {70};  // mismatched length
  const int32_t boi = 255999, eoi = 258882, video = 258884;

  CHECK_THROWS(vllm::Gemma4VideoReplacement(
      timestamps, soft_tokens, boi, eoi, video, ascii_encode));
}
