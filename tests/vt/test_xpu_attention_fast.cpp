#include "xpu_test_helpers.h"
#include "vt/xpu.h"
#include "vt/fp8_kv.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vt/unaligned.h"
#include "vt/paged_attn_route.h"
#include <filesystem>
#include <array>
#include <memory>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string_view>
#include <utility>
#include <nlohmann/json.hpp>

namespace {
using vt::DType;
using xpu_test::Buffer;
using xpu_test::Queue;
std::vector<float> Random(size_t size, uint32_t seed, float scale = 1) {
  std::vector<float> values(size);
  for (auto& v : values) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    v = float(int(seed & 65535) - 32768) / 32768.0f * scale;
  }
  return values;
}
void Accuracy(const std::vector<float>& actual, const std::vector<float>& expected, bool quantized = false, bool matrix = false) {
  REQUIRE(actual.size() == expected.size());
  double error = 0, norm = 0, worst = 0, peak = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i])) FAIL("nonfinite attention output");
    const double delta = actual[i] - expected[i];
    error += delta * delta; norm += double(expected[i]) * expected[i];
    worst = std::max(worst, std::abs(delta)); peak = std::max(peak, std::abs(double(expected[i])));
  }
  const double rms = std::sqrt(error / std::max(1e-30, norm));
  std::cout << "ATTN_ERROR quantized=" << quantized << " rms=" << rms << " max_abs=" << worst << std::endl;
  CHECK(rms <= (quantized ? 0.05 : matrix ? 0.001 : 1e-4));
  CHECK(worst <= 3e-6 + (quantized ? 0.1 : matrix ? 0.002 : 1e-4) * peak);
}
struct Fixture {
  vt::Queue& queue;
  int requests, tokens, context, page = 16, blocks, columns;
  Buffer query, out, cache, table, lens, offsets;
  vt::Tensor kc, vc;
  vt::PagedAttentionArgs args;
  Fixture(vt::Queue& q, int nreq, int chunk, int length, bool fp8, bool unequal = false,
          int block = 16, DType query_type = DType::kF32,
          DType output_type = DType::kF32, DType cache_type = DType::kBF16,
          bool unit_scales = false)
      : queue(q), requests(nreq), tokens(nreq * chunk), context(length), page(block),
        blocks(nreq * ((length + block - 1) / block)), columns((length + block - 1) / block),
        query(q, query_type, {tokens, 24, 256}), out(q, output_type, {tokens, 24, 256}),
        cache(q, fp8 ? DType::kI8 : cache_type, {blocks, 2 * page, 4, 256}),
        table(q, DType::kI32, {requests, 2 * columns + 1}),
        lens(q, DType::kI32, {requests}), offsets(q, DType::kI32, {requests + 1}) {
    std::vector<int32_t> lens_data(requests), qsl{0}, bt_data(requests * (2 * columns + 1), -1);
    for (int r = 0; r < requests; ++r) {
      lens_data[r] = length - (unequal ? r * 17 : 0);
      qsl.push_back(qsl.back() + (unequal && r ? chunk - r : chunk));
      for (int b = 0; b < columns; ++b) bt_data[r * (2 * columns + 1) + 2 * b] = blocks - 1 - r * columns - b;
    }
    tokens = qsl.back();
    auto queries = Random(query.tensor.Numel(), 83175);
    query.put(queries); query.tensor.shape[0] = out.tensor.shape[0] = tokens;
    auto data = Random(cache.tensor.Numel(), 991); // same BF16-rounded source for both cache formats
    for (auto& v : data)
      v = cache_type == DType::kF16 ? vt::F16ToF32(vt::F32ToF16(v))
                                    : vt::BF16ToF32(vt::F32ToBF16(v));
    args.scale = 1.0f / 16;
    if (fp8) {
      args.kv_cache_dtype = vt::Fp8KVCacheDataType::kFp8E4M3;
      args.k_scale = unit_scales ? 1.0f : 0.125f;
      args.v_scale = unit_scales ? 1.0f : 0.0625f;
      std::vector<uint8_t> bytes(data.size());
      for (size_t i = 0; i < data.size(); ++i)
        bytes[i] = vt::StoreKvFp8E4M3(data[i], (i / (page * 4 * 256)) % 2 ? args.v_scale : args.k_scale);
      cache.upload(bytes.data());
    } else cache.put(data);
    kc = vt::Tensor::Contiguous(cache.tensor.data, cache.tensor.dtype, q.device, {blocks, page, 4, 256});
    kc.stride[0] *= 2; vc = kc;
    vc.data = static_cast<char*>(kc.data) + page * 4 * 256 * vt::SizeOf(kc.dtype);
    table.upload(bt_data.data()); table.tensor.shape[1] = columns; table.tensor.stride[1] = 2;
    lens.upload(lens_data.data()); offsets.upload(qsl.data()); args.max_seq_len = length;
  }
  void run(const char* mode) {
    setenv("VT_XPU_ATTENTION", mode, 1);
    vt::PagedAttention(queue, out.tensor, query.tensor, kc, vc, table.tensor, lens.tensor, offsets.tensor, args);
    vt::GetBackend(queue.device).Synchronize(queue);
  }
  std::vector<float> result() { auto data = out.floats(); data.resize(tokens * 24 * 256); return data; }
};
}
#ifdef VLLM_CPP_XPU_ONEDNN
TEST_CASE("XPU EXL3 oneDNN attention: exact page boundaries scales and query padding"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE"))) {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  REQUIRE(std::getenv("VT_XPU_PROFILE") != nullptr);
  (void)vt::xpu::DrainProfileEvents();
  for (int length : {1599, 1600, 1601}) for (int rows : {3, 129}) {
    CAPTURE(length);
    CAPTURE(rows);
    Fixture ref(cpu.q, 1, rows, length, true, false, 1600, DType::kF16, DType::kF32,
                DType::kF16, rows == 129);
    Fixture got(gpu.q, 1, rows, length, true, false, 1600, DType::kF16, DType::kF16,
                DType::kF16, rows == 129);
    // A second attention scale exercises the runtime FP16 divisor input;
    // cache partitions must not reuse the first call's model constant.
    for (float scale : {0.0625f, 0.125f}) {
      ref.args.scale = got.args.scale = scale;
      ref.run("reference"); got.run("exl3_onednn");
      Accuracy(got.result(), ref.result(), false, true);
      int sdpa = 0;
      for (const auto& event : vt::xpu::DrainProfileEvents())
        if (event.stage == "exl3_onednn_sdpa") ++sdpa;
      CHECK(sdpa == 4);
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 oneDNN attention: unequal C4 and strided output copy"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE"))) {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  REQUIRE(std::getenv("VT_XPU_PROFILE") != nullptr);
  (void)vt::xpu::DrainProfileEvents();
  Fixture ref(cpu.q, 4, 8, 1601, true, true, 1600, DType::kF16, DType::kF32);
  Fixture got(gpu.q, 4, 8, 1601, true, true, 1600, DType::kF16, DType::kF16);
  Buffer padded(gpu.q, DType::kF16, {got.tokens, 24, 258});
  padded.put(std::vector<float>(padded.tensor.Numel(), -7));
  auto view = padded.tensor; view.shape[2] = 256;
  ref.run("reference");
  setenv("VT_XPU_ATTENTION", "exl3_onednn", 1);
  vt::PagedAttention(gpu.q, view, got.query.tensor, got.kc, got.vc, got.table.tensor,
                     got.lens.tensor, got.offsets.tensor, got.args);
  vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
  const auto physical = padded.floats();
  std::vector<float> active;
  bool padding_unchanged = true;
  for (int64_t row = 0; row < got.tokens*24; ++row) {
    active.insert(active.end(), physical.begin()+row*258, physical.begin()+row*258+256);
    padding_unchanged &= physical[row*258+256] == -7 && physical[row*258+257] == -7;
  }
  CHECK(padding_unchanged);
  Accuracy(active, ref.result(), false, true);
  int sdpa = 0;
  for (const auto& event : vt::xpu::DrainProfileEvents())
    if (event.stage == "exl3_onednn_sdpa") ++sdpa;
  CHECK(sdpa == 16);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 oneDNN attention: original real operands and physical hybrid strides"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE"))) {
  const char* env = std::getenv("VT_B70_EXL3_EXACT_K_FIXTURE");
  if (!env) std::exit(77);
  const std::filesystem::path path(env);
  auto file = vllm::SafetensorsFile::Open(path.string());
  auto receipt_path = path; receipt_path.replace_extension(".json");
  std::ifstream stream(receipt_path);
  const auto receipt = nlohmann::json::parse(stream);
  Queue gpu(vt::DeviceType::kXPU);
  REQUIRE(std::getenv("VT_XPU_PROFILE") != nullptr);
  for (const auto& item : receipt.at("cases")) {
    const auto label = item.at("label").get<std::string>();
    CAPTURE(label);
    const int rows = item.at("logical_q"), length = item.at("exact_k"), blocks = item.at("blocks");
    Buffer query(gpu.q, DType::kF16, {rows, 24, 256}), output(gpu.q, DType::kF16, {rows, 24, 256});
    Buffer cache(gpu.q, DType::kI8, {blocks, 1600, 4, 512}), pages(gpu.q, DType::kI32, {1, blocks});
    Buffer lengths(gpu.q, DType::kI32, {1}), offsets(gpu.q, DType::kI32, {2});
    query.upload(file.Get(label+"_query").data); cache.upload(file.Get(label+"_cache").data);
    pages.upload(file.Get(label+"_pages").data);
    lengths.upload(&length); const int32_t qsl[]{0, rows}; offsets.upload(qsl);
    auto key = cache.tensor; key.shape[3] = 256;
    auto value = key; value.data = static_cast<uint8_t*>(key.data)+256;
    vt::PagedAttentionArgs args; args.causal = true;
    args.kv_cache_dtype = vt::Fp8KVCacheDataType::kFp8E4M3;
    args.k_scale = item.at("k_scale"); args.v_scale = item.at("v_scale");
    args.scale = item.at("attention_scale");
    // An intentionally loose host maximum must never enter the mask or the
    // compiled key dimensions: only the actual GPU sequence length is valid.
    args.max_seq_len = length+1600;
    setenv("VT_XPU_ATTENTION", rows > 128 ? "auto" : "exl3_onednn", 1);
    (void)vt::xpu::DrainProfileEvents();
    vt::PagedAttention(gpu.q, output.tensor, query.tensor, key, value, pages.tensor,
                       lengths.tensor, offsets.tensor, args);
    const auto raw = output.download();
    const auto& expected = file.Get(label+"_output");
    REQUIRE(expected.dtype == "F16"); REQUIRE(raw.size() == expected.nbytes);
    std::vector<float> gold(raw.size()/2);
    size_t half_differences = 0;
    for (size_t i = 0; i < gold.size(); ++i) {
      const auto bits = vt::LoadUnaligned<uint16_t>(expected.data+2*i);
      gold[i] = vt::F16ToF32(bits);
      half_differences += bits != vt::LoadUnaligned<uint16_t>(raw.data()+2*i);
    }
    std::cout << "ORIGINAL_EXACT_K " << label << " half_differences=" << half_differences << '\n';
    Accuracy(output.floats(), gold, false, true);
    int sdpa = 0;
    for (const auto& event : vt::xpu::DrainProfileEvents())
      if (event.stage == "exl3_onednn_sdpa") ++sdpa;
    CHECK(sdpa == 4);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 oneDNN attention: bounded eviction preserves reused output") {
  Queue gpu(vt::DeviceType::kXPU);
  Fixture first(gpu.q, 1, 3, 513, true, false, 1600, DType::kF16, DType::kF16);
  first.run("exl3_onednn"); const auto initial = first.result();
  for (int length = 514; length < 532; ++length) {
    Fixture current(gpu.q, 1, 3, length, true, false, 1600, DType::kF16, DType::kF16);
    current.run("exl3_onednn");
    const auto values = current.result();
    CHECK(std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); }));
  }
  first.run("exl3_onednn"); CHECK(first.result() == initial);
  CHECK(vt::GetReferenceTierHits() == 0);
}
#endif

TEST_CASE("XPU split-KV: 32k, batch4, M2-5, uneven requests and windowed softcap") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (bool fp8 : {false, true}) for (int chunk : {1, 2, 3, 4, 5}) {
    const int nreq = chunk == 5 ? 4 : 1, length = chunk == 1 || chunk == 5 ? 32768 : 4097;
    CAPTURE(fp8);
    CAPTURE(chunk);
    Fixture ref(cpu.q, nreq, chunk, length, fp8, nreq == 4);
    Fixture got(gpu.q, nreq, chunk, length, fp8, nreq == 4);
    if (fp8 && chunk == 5 && std::getenv("VT_XPU_PROFILE"))
      (void)vt::xpu::DrainProfileEvents();
    ref.run("reference"); got.run("split"); Accuracy(got.result(), ref.result());
    if (fp8 && chunk == 5 && std::getenv("VT_XPU_PROFILE")) {
      int split_events = 0, reduce_events = 0;
      for (const auto& event : vt::xpu::DrainProfileEvents()) {
        split_events += event.stage == "attention_split_partial";
        reduce_events += event.stage == "attention_split_reduce_cooperative";
      }
      CHECK(split_events == 1);
      CHECK(reduce_events == 1);
    }
    if (chunk == 5) {
      ref.args.causal = got.args.causal = false;
      ref.args.window_size = got.args.window_size = vt::AttentionWindow{129, 3};
      ref.args.logits_soft_cap = got.args.logits_soft_cap = 0.7f;
      ref.run("reference"); got.run("split"); Accuracy(got.result(), ref.result());
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == 16 * 1024 * 1024);
}
TEST_CASE("XPU split-KV: E4M3 quantization against BF16 at 32k") {
  Queue gpu(vt::DeviceType::kXPU);
  Fixture bf16(gpu.q, 1, 5, 32768, false), fp8(gpu.q, 1, 5, 32768, true);
  bf16.run("split"); fp8.run("split");
  Accuracy(fp8.result(), bf16.result(), true);
  CHECK(fp8.cache.bytes * 2 == bf16.cache.bytes);
}
TEST_CASE("XPU FP8 split-K: padded block tables retain active-page plan"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE"))) {
  Queue gpu(vt::DeviceType::kXPU);
  for (int page : {1600, 1664}) {
    Fixture f(gpu.q, 1, 1, 4096, true, false, page,
              DType::kF16, DType::kF16, DType::kF16, true);
    for (int cols : {3, 6, 164}) {
      CAPTURE(page);
      CAPTURE(cols);
      Buffer table(gpu.q, DType::kI32, {1, cols});
      std::vector<int32_t> ids(cols, 0);
      ids[0] = 2; ids[1] = 1; ids[2] = 0;
      table.upload(ids.data());
      auto run = [&](const char* mode) {
        setenv("VT_XPU_ATTENTION", mode, 1);
        vt::PagedAttention(gpu.q, f.out.tensor, f.query.tensor, f.kc, f.vc,
                           table.tensor, f.lens.tensor, f.offsets.tensor, f.args);
        vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
      };
      run("reference");
      const auto expected = f.result();
      (void)vt::xpu::DrainProfileEvents();
      run("auto");
      Accuracy(f.result(), expected, false, true);
      int split = 0;
      for (const auto& event : vt::xpu::DrainProfileEvents())
        split += event.stage == "attention_split_partial";
      CHECK(split == 1);
      if (cols == 164) {
        setenv("VT_XPU_ATTN_SPLIT_ACTIVE_PAGE_CAP", "0", 1);
        run("auto");
        Accuracy(f.result(), expected, false, true);
        unsetenv("VT_XPU_ATTN_SPLIT_ACTIVE_PAGE_CAP");
      }
    }
  }
}
TEST_CASE("XPU FP8 attention: 20-query split boundary at 4K context"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE"))) {
  Queue gpu(vt::DeviceType::kXPU);
  for (int queries : {20, 21, 31, 32, 33}) {
    CAPTURE(queries);
    Fixture f(gpu.q, 1, queries, 4096 + queries, true, false, 1600,
              DType::kF16, DType::kF16, DType::kF16, true);
    f.run("reference");
    const auto expected = f.result();
    (void)vt::xpu::DrainProfileEvents();
    f.run("auto");
    Accuracy(f.result(), expected, false, true);
    int split = 0, q32 = 0, fallback = 0;
    for (const auto& event : vt::xpu::DrainProfileEvents()) {
      split += event.stage == "attention_split_partial";
      q32 += event.stage == "attention_prefill_q32";
      fallback += event.stage == "attention_reference";
    }
    CHECK(split + q32 + fallback == 1);
    CHECK((queries <= 31 ? split == 1 : q32 == 1));
    if (queries == 21) {
      setenv("VT_XPU_ATTN_SPLIT_EXTENDED", "0", 1);
      (void)vt::xpu::DrainProfileEvents();
      f.run("auto");
      int opted_out = 0;
      for (const auto& event : vt::xpu::DrainProfileEvents())
        opted_out += event.stage == "attention_reference";
      CHECK(opted_out == 1);
      unsetenv("VT_XPU_ATTN_SPLIT_EXTENDED");
    }
    if (std::getenv("VT_B70_ATTN_BENCH")) {
      std::vector<double> samples;
      for (int repeat = 0; repeat < 5; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        f.run("auto");
        samples.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
      }
      std::sort(samples.begin(), samples.end());
      std::cout << nlohmann::json{{"event", "fp8_attention_boundary"},
          {"query_tokens", queries}, {"context_tokens", 4096 + queries},
          {"route", split ? "split" : q32 ? "q32" : "reference"},
          {"median_ms", samples[2]}, {"samples_ms", samples}}.dump()
                << std::endl;
    }
  }
}
TEST_CASE("XPU attention: timing" * doctest::skip(!std::getenv("VT_B70_ATTN_BENCH"))) {
  Queue gpu(vt::DeviceType::kXPU);
  std::cout << "DEVICE " << vt::xpu::DeviceDescription() << std::endl;
  for (int length : {128, 512, 4096, 32768}) for (int chunk : {1, 5}) for (bool fp8 : {false, true}) {
    Fixture f(gpu.q, 1, chunk, length, fp8);
    for (const char* mode : {"reference", "split"}) {
      f.run(mode);
      auto start = std::chrono::steady_clock::now();
      int warmups = 1;  // includes the initial dispatch above
      do { f.run(mode); ++warmups; } while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(100));
      std::vector<double> ms;
      for (int i = 0; i < 5; ++i) {
        start = std::chrono::steady_clock::now(); f.run(mode);
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
      }
      std::sort(ms.begin(), ms.end());
      std::cout << nlohmann::json{{"context", length}, {"queries", chunk}, {"mode", mode}, {"fp8", fp8},
          {"median_ms", ms[2]}, {"samples_ms", ms}, {"warmups", warmups}, {"cache_bytes", f.cache.bytes},
          {"workspace_bytes", vt::xpu::GetMemoryInfo().attention_workspace_bytes}}.dump() << std::endl;
    }
  }
}

TEST_CASE("XPU XMX attention prefill: tails, batch4, appended 32k context and masks") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (bool fp8 : {false, true}) for (int chunk : {31, 33, 63, 128, 511}) {
    CAPTURE(fp8);
    CAPTURE(chunk);
    const int nreq = chunk == 63 ? 4 : 1, context = chunk == 128 ? 32768 : chunk + 3 + (nreq - 1) * 17;
    Fixture ref(cpu.q, nreq, chunk, context, fp8, nreq == 4);
    Fixture got(gpu.q, nreq, chunk, context, fp8, nreq == 4);
    ref.run("reference"); got.run("prefill"); Accuracy(got.result(), ref.result(), false, true);
    if (chunk == 63) {
      ref.args.causal = got.args.causal = false;
      ref.args.window_size = got.args.window_size = vt::AttentionWindow{13, 7};
      ref.args.logits_soft_cap = got.args.logits_soft_cap = 0.7f;
      ref.run("reference"); got.run("prefill"); Accuracy(got.result(), ref.result(), false, true);
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}
TEST_CASE("XPU short F16 attention prefill selects XMX path"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE"))) {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  Fixture ref(cpu.q, 1, 64, 67, false, false, 16,
              DType::kF16, DType::kF32, DType::kF16);
  Fixture got(gpu.q, 1, 64, 67, false, false, 16,
              DType::kF16, DType::kF16, DType::kF16);
  ref.run("reference");
  (void)vt::xpu::DrainProfileEvents();
  got.run("auto");
  Accuracy(got.result(), ref.result(), false, true);
  const auto records = vt::xpu::DrainProfileEvents();
  size_t prefill = 0, fallback = 0;
  for (const auto& record : records) {
    prefill += record.stage == "attention_prefill_q64";
    fallback += record.stage == "attention_reference";
  }
  CHECK(prefill == 1);
  CHECK(fallback == 0);
}
TEST_CASE("XPU Q32/Q64 F16 attention prefill: page boundaries, continuation and masks") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (int chunk : {31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 512}) {
    CAPTURE(chunk);
    const int page = chunk == 129 || chunk == 257 ? 3 : 16;
    const int context = chunk + 17;
    Fixture ref(gpu.q, 1, chunk, context, false, false, page,
                DType::kF16, DType::kF16, DType::kF16);
    Fixture got(gpu.q, 1, chunk, context, false, false, page,
                DType::kF16, DType::kF16, DType::kF16);
    ref.run("reference");
    if (chunk == 127) {
      setenv("VT_XPU_ATTN_PREFILL_TILE", "q16", 1);
      got.run("prefill");
      Accuracy(got.result(), ref.result(), false, true);
    }
    setenv("VT_XPU_ATTN_PREFILL_TILE", "q32", 1);
    got.run("prefill");
    Accuracy(got.result(), ref.result(), false, true);
    setenv("VT_XPU_ATTN_PREFILL_TILE", "q64", 1);
    got.run("prefill");
    Accuracy(got.result(), ref.result(), false, true);
    if (chunk == 127 || chunk == 257) {
      ref.args.causal = got.args.causal = false;
      ref.args.window_size = got.args.window_size = vt::AttentionWindow{37, 5};
      ref.args.logits_soft_cap = got.args.logits_soft_cap = 0.7f;
      ref.run("reference"); got.run("prefill");
      Accuracy(got.result(), ref.result(), false, true);
    }
    if (chunk == 127) {
      Fixture independent(cpu.q, 1, chunk, context, false, false, page,
                          DType::kF16, DType::kF32, DType::kF16);
      independent.run("reference");
      ref.args.causal = true;
      ref.args.window_size.reset();
      ref.args.logits_soft_cap = 0;
      ref.run("reference");
      Accuracy(ref.result(), independent.result(), false, true);
    }
  }
  unsetenv("VT_XPU_ATTN_PREFILL_TILE");
}
TEST_CASE("XPU Q32/Q64 F16 attention prefill: ragged batch and aliased output") {
  Queue gpu(vt::DeviceType::kXPU);
  Fixture ref(gpu.q, 4, 63, 129, false, true, 3,
              DType::kF16, DType::kF16, DType::kF16);
  Fixture got(gpu.q, 4, 63, 129, false, true, 3,
              DType::kF16, DType::kF16, DType::kF16);
  ref.run("reference");
  setenv("VT_XPU_ATTN_PREFILL_TILE", "q32", 1);
  got.run("prefill");
  Accuracy(got.result(), ref.result(), false, true);
  setenv("VT_XPU_ATTN_PREFILL_TILE", "q64", 1);
  got.run("prefill");
  Accuracy(got.result(), ref.result(), false, true);
  ref.args.causal = got.args.causal = false;
  ref.args.window_size = got.args.window_size = vt::AttentionWindow{33, 7};
  ref.args.logits_soft_cap = got.args.logits_soft_cap = 0.7f;
  ref.run("reference"); got.run("prefill");
  Accuracy(got.result(), ref.result(), false, true);

  Fixture aliased(gpu.q, 1, 33, 49, false, false, 16,
                  DType::kF16, DType::kF16, DType::kF16);
  Fixture alias_ref(gpu.q, 1, 33, 49, false, false, 16,
                    DType::kF16, DType::kF16, DType::kF16);
  alias_ref.run("reference");
  setenv("VT_XPU_ATTN_PREFILL_TILE", "q32", 1);
  setenv("VT_XPU_ATTENTION", "prefill", 1);
  vt::PagedAttention(gpu.q, aliased.query.tensor, aliased.query.tensor,
                     aliased.kc, aliased.vc, aliased.table.tensor,
                     aliased.lens.tensor, aliased.offsets.tensor, aliased.args);
  Accuracy(aliased.query.floats(), alias_ref.result(), false, true);
  unsetenv("VT_XPU_ATTN_PREFILL_TILE");
}
TEST_CASE("XPU Q32/Q64 F16 attention prefill: BF16 and E4M3 cache") {
  Queue gpu(vt::DeviceType::kXPU);
  for (bool fp8 : {false, true}) {
    CAPTURE(fp8);
    Fixture ref(gpu.q, 1, 129, 146, fp8, false, 16,
                DType::kF16, DType::kF16, DType::kBF16);
    Fixture got(gpu.q, 1, 129, 146, fp8, false, 16,
                DType::kF16, DType::kF16, DType::kBF16);
    ref.run("reference");
    setenv("VT_XPU_ATTN_PREFILL_TILE", "q32", 1);
    got.run("prefill");
    Accuracy(got.result(), ref.result(), false, true);
    setenv("VT_XPU_ATTN_PREFILL_TILE", "q64", 1);
    got.run("prefill");
    Accuracy(got.result(), ref.result(), false, true);
  }
  unsetenv("VT_XPU_ATTN_PREFILL_TILE");
}
TEST_CASE("XPU XMX attention prefill: FP16 overflow eligibility fallback") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  Fixture ref(cpu.q, 1, 33, 40, false), got(gpu.q, 1, 33, 40, false);
  auto values = Random(33 * 24 * 256, 828, 100000);
  ref.query.put(values); got.query.put(values);
  ref.run("reference"); got.run("prefill"); Accuracy(got.result(), ref.result());
}
TEST_CASE("XPU XMX attention prefill: empty requests and aliased query output") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  Fixture ref(cpu.q, 4, 17, 256, false), got(gpu.q, 4, 17, 256, false);
  const int32_t qsl[] = {0, 0, 17, 17, 66};
  for (auto* f : {&ref, &got}) {
    f->tokens = 66; f->query.tensor.shape[0] = f->out.tensor.shape[0] = 66;
    f->offsets.upload(qsl);
  }
  ref.run("reference");
  setenv("VT_XPU_ATTENTION", "prefill", 1);
  vt::PagedAttention(gpu.q, got.query.tensor, got.query.tensor, got.kc, got.vc,
                      got.table.tensor, got.lens.tensor, got.offsets.tensor, got.args);
  auto actual = got.query.floats(); actual.resize(66 * 24 * 256);
  Accuracy(actual, ref.result(), false, true);
}
TEST_CASE("XPU split-KV: insufficient workspace budget uses native reference"
          * doctest::skip(!std::getenv("VT_B70_LOW_MEMORY_TEST"))) {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  Fixture ref(cpu.q, 1, 5, 128, false), got(gpu.q, 1, 5, 128, false);
  ref.run("reference"); got.run("split"); Accuracy(got.result(), ref.result());
  CHECK(vt::xpu::GetMemoryInfo().attention_workspace_bytes == 0);
  CHECK(vt::GetReferenceTierHits() == 0);
}
TEST_CASE("XPU XMX attention prefill: non-power-of-two pages") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (bool fp8 : {false, true}) {
    Fixture ref(cpu.q, 2, 33, 97, fp8, false, 3), got(gpu.q, 2, 33, 97, fp8, false, 3);
    ref.run("reference"); got.run("prefill"); Accuracy(got.result(), ref.result(), false, true);
  }
}
TEST_CASE("XPU split-KV: workspace survives queue replacement") {
  std::vector<float> first;
  for (int repeat = 0; repeat < 3; ++repeat) {
    Queue gpu(vt::DeviceType::kXPU);
    {
      Fixture got(gpu.q, 1, 5, 513, false); got.run("split");
      if (!repeat) first = got.result(); else CHECK(got.result() == first);
    }
    CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == 16 * 1024 * 1024);
  }
}
TEST_CASE("XPU attention prefill: timing" * doctest::skip(!std::getenv("VT_B70_ATTN_BENCH"))) {
  Queue gpu(vt::DeviceType::kXPU);
  std::cout << "DEVICE " << vt::xpu::DeviceDescription() << std::endl;
  for (int length : {128, 512, 4096}) for (bool fp8 : {false, true}) {
    Fixture f(gpu.q, 1, length, length, fp8);
    std::vector<float> reference;
    for (const char* mode : {"reference", "prefill"}) {
      f.run(mode);
      auto start = std::chrono::steady_clock::now();
      int warmups = 1;  // includes the initial dispatch above
      do { f.run(mode); ++warmups; } while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(100));
      std::vector<double> ms;
      for (int i = 0; i < 5; ++i) {
        start = std::chrono::steady_clock::now(); f.run(mode);
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
      }
      std::sort(ms.begin(), ms.end());
      std::cout << nlohmann::json{{"context", length}, {"queries", length}, {"mode", mode}, {"fp8", fp8},
          {"median_ms", ms[2]}, {"samples_ms", ms}, {"warmups", warmups}, {"cache_bytes", f.cache.bytes}}.dump() << std::endl;
      if (std::string(mode) == "reference") reference = f.result();
      else Accuracy(f.result(), reference, false, true);
    }
  }
}

TEST_CASE("XPU FP8 KV auto selects fast prefill and decode"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE"))) {
  Queue gpu(vt::DeviceType::kXPU);
  Fixture prefill(gpu.q, 1, 64, 67, true, false, 64,
                  DType::kF16, DType::kF16, DType::kF16);
  prefill.run("prefill");
  const auto prefill_expected = prefill.result();
  (void)vt::xpu::DrainProfileEvents();
  prefill.run("auto");
  Accuracy(prefill.result(), prefill_expected, false, true);
  bool saw_prefill = false;
  for (const auto& event : vt::xpu::DrainProfileEvents())
    saw_prefill |= event.stage == "attention_prefill_q64";
  CHECK(saw_prefill);

  Fixture decode(gpu.q, 1, 1, 513, true, false, 64,
                 DType::kF16, DType::kF16, DType::kF16);
  decode.run("split");
  const auto decode_expected = decode.result();
  (void)vt::xpu::DrainProfileEvents();
  decode.run("auto");
  Accuracy(decode.result(), decode_expected);
  bool saw_split = false;
  for (const auto& event : vt::xpu::DrainProfileEvents())
    saw_split |= event.stage == "attention_split_partial";
  CHECK(saw_split);
}

TEST_CASE("XPU FP8 split span candidate"
          * doctest::skip(!std::getenv("VT_B70_ATTENTION_SPLIT_CANDIDATE"))) {
  Queue gpu(vt::DeviceType::kXPU);
  for (int length : {64, 129, 513, 4097}) {
    Fixture f(gpu.q, 1, 1, length, true, false, 64,
              DType::kF16, DType::kF16, DType::kF16);
    std::vector<float> reference;
    for (const char* span : {"256", "128", "64", "32"}) {
      setenv("VT_XPU_ATTN_SPLIT_SPAN", span, 1);
      f.run("split");
      std::vector<double> ms;
      for (int i = 0; i < 15; ++i) {
        const auto start = std::chrono::steady_clock::now();
        f.run("split");
        ms.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
      }
      std::sort(ms.begin(), ms.end());
      std::cout << nlohmann::json{{"length", length}, {"span", span},
                                  {"median_ms", ms[7]},
                                  {"samples_ms", ms}}.dump() << std::endl;
      if (reference.empty()) reference = f.result();
      else Accuracy(f.result(), reference, false, true);
    }
  }
}


// Focused measurement scaffold for the d35295682f FP8 prefill review.
// This is deliberately separate from the existing F32/page16 timing matrix.
// No model, generic path, tolerance or production dispatch is changed.
TEST_CASE("XPU FP8 prefill: focused F16 P64 baseline"
          * doctest::skip(!std::getenv("VT_B70_FP8_PREFILL_FOCUS"))) {
  const std::string length_text = std::getenv("VT_B70_FP8_PREFILL_FOCUS");
  REQUIRE((length_text == "512" || length_text == "4096"));
  const int length = length_text == "512" ? 512 : 4096;
  // Run in an isolated test process. Set these before creating the queue.
  const char* tile = std::getenv("VT_XPU_ATTN_PREFILL_TILE");
  const char* probability = std::getenv("VT_XPU_ATTN_PROBABILITY");
  REQUIRE(tile != nullptr);
  REQUIRE(probability != nullptr);
  REQUIRE(std::string(tile) == "q64");
  REQUIRE(std::string(probability) == "single");
  const char* profile_setting = std::getenv("VT_XPU_PROFILE");
  const bool profiling = profile_setting && std::string(profile_setting) == "1";
  Queue gpu(vt::DeviceType::kXPU);
  Fixture f(gpu.q, 1, length, length, true, false, 64,
            DType::kF16, DType::kF16, DType::kF16);

  // Dense table columns, reversed physical pages. Buffer::upload copies the
  // allocation size, so keep a full-sized host buffer despite the smaller view.
  std::vector<int32_t> table_data(f.table.bytes / sizeof(int32_t), -1);
  for (int i = 0; i < f.columns; ++i) table_data[i] = f.blocks - 1 - i;
  f.table.upload(table_data.data());
  f.table.tensor.stride[0] = f.columns;
  f.table.tensor.stride[1] = 1;
  // Match the current model benchmark's unit scales. Re-encode once, outside
  // timing, rather than changing the interpretation of already encoded bytes.
  f.args.k_scale = f.args.v_scale = 1.0f;
  auto values = Random(f.cache.tensor.Numel(), 991);
  std::vector<uint8_t> encoded(values.size());
  for (size_t i = 0; i < values.size(); ++i)
    encoded[i] = vt::StoreKvFp8E4M3(vt::F16ToF32(vt::F32ToF16(values[i])), 1.0f);
  f.cache.upload(encoded.data());
  std::cout << "DEVICE " << vt::xpu::DeviceDescription() << std::endl;

  // One generic GPU reference, not repeated generic-reference warmups.
  f.run("reference");
  const auto reference = f.result();
  (void)vt::xpu::DrainProfileEvents();
  f.run("prefill"); // initial dispatch/JIT, outside samples
  Accuracy(f.result(), reference, false, true); // same-cache strict matrix gate
  auto events = vt::xpu::DrainProfileEvents();
  if (profiling) {
    int count = 0;
    for (const auto& event : events) count += event.stage == "attention_prefill_q64";
    REQUIRE(count == 1); // never silently benchmark a generic fallback
  }
  const auto warm_start = std::chrono::steady_clock::now();
  int warmups = 1;
  // Bounded screen, not a claim of clocks/thermal stabilization.
  while (warmups < 32 && std::chrono::steady_clock::now() - warm_start <
                             std::chrono::milliseconds(100)) {
    f.run("prefill");
    ++warmups;
    (void)vt::xpu::DrainProfileEvents();
  }
  std::vector<double> wall_ms, kernel_ms;
  for (int sample = 0; sample < 5; ++sample) {
    const auto start = std::chrono::steady_clock::now();
    f.run("prefill");
    wall_ms.push_back(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count());
    events = vt::xpu::DrainProfileEvents(); // outside the wall-clock sample
    if (profiling) {
      int count = 0;
      double elapsed = 0;
      for (const auto& event : events) if (event.stage == "attention_prefill_q64") {
        REQUIRE(event.end_ns >= event.start_ns);
        elapsed += double(event.end_ns - event.start_ns) / 1.0e6;
        ++count;
      }
      REQUIRE(count == 1);
      kernel_ms.push_back(elapsed);
    }
  }
  auto sorted = wall_ms;
  std::sort(sorted.begin(), sorted.end());
  std::cout << nlohmann::json{
      {"event", "fp8_prefill_focus"}, {"context", length}, {"queries", length},
      {"query_dtype", "float16"}, {"output_dtype", "float16"},
      {"kv_dtype", "fp8_e4m3"}, {"page_size", 64},
      {"table_column_stride", f.table.tensor.stride[1]},
      {"k_page_stride_elements", f.kc.stride[0]},
      {"k_token_stride_elements", f.kc.stride[1]},
      {"k_head_stride_elements", f.kc.stride[2]},
      {"v_page_stride_elements", f.vc.stride[0]},
      {"k_scale", f.args.k_scale}, {"v_scale", f.args.v_scale},
      {"tile", tile}, {"probability", probability}, {"mode", "prefill"},
      {"profiling", profiling}, {"dispatch_verified", profiling},
      {"scope", "one operator; wall includes synchronization/status, kernel is event time"},
      {"warmups", warmups}, {"wall_samples_ms", wall_ms},
      {"wall_median_ms", sorted[2]}, {"kernel_samples_ms", kernel_ms},
      {"cache_bytes", f.cache.bytes}}.dump() << std::endl;
  Accuracy(f.result(), reference, false, true);
  CHECK(vt::GetReferenceTierHits() == 0); // generic reference above is a GPU kernel
}

TEST_CASE("XPU Xe2 FP8 prefill: default dispatch with 1600/1664-token pages"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE") ||
                          std::getenv("VT_XPU_XE2_PREFILL"))) {
  Queue gpu(vt::DeviceType::kXPU);
  for (int page : {1600, 1664}) {
    CAPTURE(page);
    Fixture reference(gpu.q, 1, 4096, 4096, true, false, page,
                      DType::kF16, DType::kF16, DType::kF16, true);
    Fixture selected(gpu.q, 1, 4096, 4096, true, false, page,
                     DType::kF16, DType::kF16, DType::kF16, true);
    Buffer contiguous_table(gpu.q, DType::kI32, {1, 3});
    const int32_t page_ids[] = {2, 1, 0};
    contiguous_table.upload(page_ids);
    setenv("VT_XPU_XE2_PREFILL", "0", 1);
    reference.run("prefill");
    unsetenv("VT_XPU_XE2_PREFILL");
    (void)vt::xpu::DrainProfileEvents();
    setenv("VT_XPU_ATTENTION", "auto", 1);
    vt::PagedAttention(gpu.q, selected.out.tensor, selected.query.tensor,
                       selected.kc, selected.vc, contiguous_table.tensor,
                       selected.lens.tensor, selected.offsets.tensor, selected.args);
    vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
    Accuracy(selected.result(), reference.result(), false, true);
    const auto events = vt::xpu::DrainProfileEvents();
    size_t xe2 = 0, fallback = 0;
    for (const auto& event : events) {
      xe2 += event.stage == "attention_prefill_xe2";
      fallback += event.stage == "attention_prefill_q64";
    }
    CHECK(xe2 == 1);
    CHECK(fallback == 0);
  }
}

TEST_CASE("XPU Xe2 FP8 prefill: 2K-8K ragged initial prompt with paged KV"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE") ||
                          std::getenv("VT_XPU_XE2_PREFILL"))) {
  Queue gpu(vt::DeviceType::kXPU);
  for (int tokens : {2048, 2049, 2050, 3071, 3072, 3073,
                     4095, 4096, 4097, 6143, 6144, 6145, 8191, 8192})
      for (int page : {64, 1600}) {
    CAPTURE(tokens);
    CAPTURE(page);
    Fixture f(gpu.q, 1, tokens, tokens, true, false, page,
              DType::kF16, DType::kF16, DType::kF16, true);
    const int pages = (tokens + page - 1) / page;
    Buffer contiguous_table(gpu.q, DType::kI32, {1, pages});
    std::vector<int32_t> page_ids(pages);
    for (int b = 0; b < pages; ++b) page_ids[b] = pages - 1 - b;
    contiguous_table.upload(page_ids.data());
    auto run = [&](const char* mode) {
      setenv("VT_XPU_ATTENTION", mode, 1);
      vt::PagedAttention(gpu.q, f.out.tensor, f.query.tensor, f.kc, f.vc,
                         contiguous_table.tensor, f.lens.tensor, f.offsets.tensor,
                         f.args);
      vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
    };
    setenv("VT_XPU_XE2_PREFILL", "0", 1);
    run("prefill");
    const auto expected = f.result();
    unsetenv("VT_XPU_XE2_PREFILL");
    (void)vt::xpu::DrainProfileEvents();
    run("auto");
    Accuracy(f.result(), expected, false, true);
    int xe2 = 0, fallback = 0;
    for (const auto& event : vt::xpu::DrainProfileEvents()) {
      xe2 += event.stage == "attention_prefill_xe2";
      fallback += event.stage == "attention_prefill_q64";
    }
    CHECK(xe2 == 1);
    CHECK(fallback == 0);
  }
}

TEST_CASE("XPU Xe2 FP8 prefill: ragged continuation through 8K KV"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE") ||
                          std::getenv("VT_XPU_XE2_PREFILL"))) {
  Queue gpu(vt::DeviceType::kXPU);
  for (auto [queries, prefix] : {std::pair{2048, 63},
                                 std::pair{2048, 2048},
                                 std::pair{2048, 4096},
                                 std::pair{2049, 4096},
                                 std::pair{3071, 4096},
                                 std::pair{3072, 1600},
                                 std::pair{3072, 4096},
                                 std::pair{3073, 4096},
                                 std::pair{4095, 4096},
                                 std::pair{4096, 4096}})
      for (int page : {64, 1600}) {
    const int context = prefix + queries;
    CAPTURE(queries);
    CAPTURE(prefix);
    CAPTURE(page);
    Fixture f(gpu.q, 1, queries, context, true, false, page,
              DType::kF16, DType::kF16, DType::kF16, true);
    const int pages = (context + page - 1) / page;
    Buffer contiguous_table(gpu.q, DType::kI32, {1, pages});
    std::vector<int32_t> page_ids(pages);
    for (int b = 0; b < pages; ++b) page_ids[b] = pages - 1 - b;
    contiguous_table.upload(page_ids.data());
    auto run = [&](const char* mode) {
      setenv("VT_XPU_ATTENTION", mode, 1);
      vt::PagedAttention(gpu.q, f.out.tensor, f.query.tensor, f.kc, f.vc,
                         contiguous_table.tensor, f.lens.tensor, f.offsets.tensor,
                         f.args);
      vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
    };
    setenv("VT_XPU_XE2_PREFILL", "0", 1);
    run("prefill");
    const auto expected = f.result();
    unsetenv("VT_XPU_XE2_PREFILL");
    (void)vt::xpu::DrainProfileEvents();
    run("auto");
    Accuracy(f.result(), expected, false, true);
    int xe2 = 0, fallback = 0;
    for (const auto& event : vt::xpu::DrainProfileEvents()) {
      xe2 += event.stage == "attention_prefill_xe2";
      fallback += event.stage == "attention_prefill_q64";
    }
    CHECK(xe2 == 1);
    CHECK(fallback == 0);
    if (page == 1600) {
      setenv("VT_XPU_XE2_CONTINUATION", "0", 1);
      (void)vt::xpu::DrainProfileEvents();
      run("auto");
      int opted_out = 0;
      for (const auto& event : vt::xpu::DrainProfileEvents())
        opted_out += event.stage == "attention_prefill_q64";
      CHECK(opted_out == 1);
      unsetenv("VT_XPU_XE2_CONTINUATION");
    }
    if (std::getenv("VT_B70_ATTN_BENCH") &&
        ((queries == 3072 && prefix == 4096) ||
         (queries == 2048 && prefix == 4096) ||
         (queries == 4096 && prefix == 4096))) {
      std::vector<double> old_ms, xe2_ms;
      for (int repeat = 0; repeat < 5; ++repeat) {
        setenv("VT_XPU_XE2_PREFILL", "0", 1);
        auto start = std::chrono::steady_clock::now();
        run("prefill");
        old_ms.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
        unsetenv("VT_XPU_XE2_PREFILL");
        start = std::chrono::steady_clock::now();
        run("auto");
        xe2_ms.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
      }
      std::sort(old_ms.begin(), old_ms.end());
      std::sort(xe2_ms.begin(), xe2_ms.end());
      std::cout << nlohmann::json{{"event", "xe2_continuation_benchmark"},
          {"query_tokens", queries}, {"context_tokens", context}, {"page", page},
          {"q64_median_ms", old_ms[2]}, {"xe2_median_ms", xe2_ms[2]},
          {"q64_samples_ms", old_ms}, {"xe2_samples_ms", xe2_ms}}.dump()
                << std::endl;
    }
  }
}

TEST_CASE("XPU FP8 prefill: captured 4096-token Python operator replay"
          * doctest::skip(!std::getenv("VT_B70_FP8_PREFILL_REPLAY_DIR"))) {
  const std::string directory = std::getenv("VT_B70_FP8_PREFILL_REPLAY_DIR");
  auto read = [&](const std::string& name, size_t size) {
    std::ifstream file(directory + "/" + name, std::ios::binary);
    REQUIRE(file.good());
    std::vector<unsigned char> bytes(size);
    file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    REQUIRE(file.gcount() == static_cast<std::streamsize>(size));
    REQUIRE(file.peek() == std::char_traits<char>::eof());
    return bytes;
  };
  const auto metadata_bytes = [&] {
    std::ifstream file(directory + "/metadata.json", std::ios::binary);
    REQUIRE(file.good());
    return nlohmann::json::parse(file);
  }();
  const int tokens = metadata_bytes.at("q_original_shape").at(0).get<int>();
  const int heads = metadata_bytes.at("q_original_shape").at(1).get<int>();
  const int dim = metadata_bytes.at("q_original_shape").at(2).get<int>();
  const int source_pages = metadata_bytes.at("files").at("k").at("shape").at(0).get<int>();
  const int source_page = metadata_bytes.at("files").at("k").at("shape").at(1).get<int>();
  const char* xe2_setting = std::getenv("VT_XPU_XE2_PREFILL");
  const bool xe2 = !xe2_setting || std::string_view(xe2_setting) == "1";
  const int page = xe2 ? 64 : source_page;
  const int pages = (tokens + page - 1) / page;
  const int physical_pages = xe2 ? pages + 7 : pages;
  const int kv_heads = metadata_bytes.at("files").at("k").at("shape").at(2).get<int>();
  REQUIRE(tokens == 4096);
  REQUIRE(heads == 24);
  REQUIRE(dim == 256);
  REQUIRE(kv_heads == 4);
  REQUIRE(source_pages * source_page >= tokens);
  REQUIRE(metadata_bytes.at("seqused_k").get<int>() == tokens);
  REQUIRE(metadata_bytes.at("causal").get<bool>());
  REQUIRE(metadata_bytes.at("softcap").get<int>() == 0);
  REQUIRE(metadata_bytes.at("window_size") == nlohmann::json::array({-1, -1}));
  REQUIRE(metadata_bytes.at("k_original_stride").at(1).get<int>() == 2 * kv_heads * dim);
  REQUIRE(metadata_bytes.at("k_original_stride").at(2).get<int>() == 2 * dim);
  REQUIRE(metadata_bytes.at("v_original_stride") == metadata_bytes.at("k_original_stride"));
  for (const char* name : {"k_descale", "v_descale"})
    for (const auto& row : metadata_bytes.at(name))
      for (const auto& scale : row) REQUIRE(scale.get<float>() == 1.0f);

  const size_t q_bytes = size_t(tokens) * heads * dim * sizeof(uint16_t);
  const size_t source_kv_bytes = size_t(source_pages) * source_page * kv_heads * dim;
  const size_t kv_bytes = size_t(physical_pages) * page * kv_heads * dim;
  const auto q_host = read("q.bin", q_bytes);
  const auto k_host = read("k.bin", source_kv_bytes);
  const auto v_host = read("v.bin", source_kv_bytes);
  const auto expected_host = read("output.bin", q_bytes);
  // VT's public paged-attention contract requires head-contiguous K/V views.
  // Repack the same FP8 values into VT's [K page, V page] allocation layout.
  // This checks arithmetic on identical values; it does not make the memory
  // transaction pattern identical to Python's LBNHC layout.
  std::vector<unsigned char> kv_host(kv_bytes * 2);
  const size_t page_bytes = size_t(page) * kv_heads * dim;
  const size_t token_bytes = size_t(kv_heads) * dim;
  for (int token = 0; token < tokens; ++token) {
    const int physical = xe2 ? physical_pages - 1 - token / page : token / page;
    const size_t source_offset = size_t(token / source_page) * source_page * token_bytes +
                                 size_t(token % source_page) * token_bytes;
    const size_t dest_offset = size_t(physical) * 2 * page_bytes +
                               size_t(token % page) * token_bytes;
    std::memcpy(kv_host.data() + dest_offset, k_host.data() + source_offset, token_bytes);
    std::memcpy(kv_host.data() + dest_offset + page_bytes,
                v_host.data() + source_offset, token_bytes);
  }

  Queue gpu(vt::DeviceType::kXPU);
  Buffer query(gpu.q, DType::kF16, {tokens, heads, dim});
  Buffer output(gpu.q, DType::kF16, {tokens, heads, dim});
  Buffer cache(gpu.q, DType::kI8, {physical_pages, 2 * page, kv_heads, dim});
  Buffer table(gpu.q, DType::kI32, {1, pages});
  Buffer lengths(gpu.q, DType::kI32, {1});
  Buffer offsets(gpu.q, DType::kI32, {2});
  query.upload(q_host.data());
  cache.upload(kv_host.data());
  std::vector<int32_t> page_ids(pages);
  std::iota(page_ids.begin(), page_ids.end(), 0);
  if (xe2) for (int& id : page_ids) id = physical_pages - 1 - id;
  table.upload(page_ids.data());
  const int32_t length_data[] = {tokens}, offset_data[] = {0, tokens};
  lengths.upload(length_data);
  offsets.upload(offset_data);
  auto kc = vt::Tensor::Contiguous(cache.tensor.data, DType::kI8, gpu.q.device,
                                    {physical_pages, page, kv_heads, dim});
  kc.stride[0] *= 2;
  auto vc = kc;
  vc.data = static_cast<char*>(kc.data) + page * kv_heads * dim;
  vt::PagedAttentionArgs args;
  args.kv_cache_dtype = vt::Fp8KVCacheDataType::kFp8E4M3;
  args.k_scale = args.v_scale = 1.0f;
  args.scale = metadata_bytes.at("softmax_scale").get<float>();
  args.max_seq_len = tokens;
  args.causal = true;
  auto run = [&](const char* mode) {
    setenv("VT_XPU_ATTENTION", mode, 1);
    vt::PagedAttention(gpu.q, output.tensor, query.tensor, kc, vc,
                       table.tensor, lengths.tensor, offsets.tensor, args);
    vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
  };
  std::vector<float> expected(q_bytes / 2);
  for (size_t i = 0; i < expected.size(); ++i) {
    uint16_t bits;
    std::memcpy(&bits, expected_host.data() + 2 * i, 2);
    expected[i] = vt::F16ToF32(bits);
  }
  run("prefill");
  if (const char* path = std::getenv("VT_B70_FP8_PREFILL_OUTPUT_DUMP")) {
    const auto bytes = output.download();
    std::ofstream dump(path, std::ios::binary);
    REQUIRE(dump.good());
    dump.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    REQUIRE(dump.good());
  }
  Accuracy(output.floats(), expected, false, true);
  const auto events = vt::xpu::DrainProfileEvents();
  if (std::getenv("VT_XPU_PROFILE")) {
    int selected = 0;
    for (const auto& event : events)
      selected += event.stage == (xe2 ? "attention_prefill_xe2" : "attention_prefill_q64");
    REQUIRE(selected == 1);
  }
  std::vector<double> wall_ms;
  for (int sample = 0; sample < 5; ++sample) {
    const auto start = std::chrono::steady_clock::now();
    run("prefill");
    wall_ms.push_back(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count());
  }
  auto sorted = wall_ms;
  std::sort(sorted.begin(), sorted.end());
  std::cout << nlohmann::json{{"event", "fp8_prefill_python_replay"},
      {"tokens", tokens}, {"page", page}, {"pages", pages},
      {"physical_pages", physical_pages},
      {"kv_row_stride", kc.stride[1]}, {"wall_samples_ms", wall_ms},
      {"wall_median_ms", sorted[2]}}.dump() << std::endl;
  Accuracy(output.floats(), expected, false, true);
  if (xe2 && std::getenv("VT_XPU_PROFILE")) {
    (void)vt::xpu::DrainProfileEvents();
    args.k_scale = 0.1f;
    args.v_scale = 0.3f;
    run("prefill");
    int fallback = 0;
    for (const auto& event : vt::xpu::DrainProfileEvents())
      fallback += event.stage == "attention_prefill_q64";
    CHECK(fallback == 1);  // Nonunit scales retain the existing GPU implementation.
  }
}

TEST_CASE("XPU FP8 attention P1: captured original C1 page1600 Q4 verifier"
          * doctest::skip(!std::getenv("VT_B70_EXL3_VERIFY_FIXTURE"))) {
  const std::filesystem::path path = std::getenv("VT_B70_EXL3_VERIFY_FIXTURE");
  const auto fixture = vllm::SafetensorsFile::Open(path.string());
  auto metadata_path = path;
  metadata_path.replace_extension(".json");
  std::ifstream metadata(metadata_path);
  REQUIRE(metadata.good());
  const auto meta = nlohmann::json::parse(metadata);
  REQUIRE(meta.at("route").at("operator") ==
          "b70_exl3_attention.shared_kv_verify_out.default");
  REQUIRE(meta.at("route").at("splits") == 16);
  REQUIRE(meta.at("route").at("tile") == 8);
  REQUIRE(meta.at("active_length") == 4100);
  const auto& query_ref = fixture.Get("query");
  const auto& output_ref = fixture.Get("output");
  REQUIRE(query_ref.dtype == "F16");
  REQUIRE(output_ref.dtype == "F16");
  REQUIRE((query_ref.shape == std::vector<int64_t>{4, 24, 256}));
  REQUIRE(output_ref.shape == query_ref.shape);
  REQUIRE(meta.at("layouts").at("k").at("shape") ==
          nlohmann::json::array({180, 1600, 4, 256}));
  REQUIRE(meta.at("layouts").at("k").at("strides_elements") ==
          nlohmann::json::array({3276800, 2048, 512, 1}));
  REQUIRE(meta.at("layouts").at("v").at("strides_elements") ==
          meta.at("layouts").at("k").at("strides_elements"));
  REQUIRE(meta.at("layouts").at("v").at("storage_offset_elements") == 256);
  REQUIRE(meta.at("scale") == 0.0625f);
  REQUIRE(meta.at("causal") == true);
  for (const auto* name : {"k_descale", "v_descale"}) {
    const auto& scale = fixture.Get(name);
    REQUIRE(scale.dtype == "F32");
    REQUIRE(scale.nbytes == sizeof(float));
    REQUIRE(vt::LoadUnaligned<float>(scale.data) == 1.0f);
  }
  const auto& ids = fixture.Get("block_table");
  REQUIRE(ids.dtype == "I32");
  REQUIRE((ids.shape == std::vector<int64_t>{1, 3}));
  const auto& offsets_ref = fixture.Get("query_offsets");
  REQUIRE(offsets_ref.dtype == "I32");
  REQUIRE(offsets_ref.nbytes == 2 * sizeof(int32_t));
  REQUIRE(vt::LoadUnaligned<int32_t>(offsets_ref.data) == 0);
  REQUIRE(vt::LoadUnaligned<int32_t>(offsets_ref.data + 4) == 4);
  const auto& lengths_ref = fixture.Get("lengths");
  REQUIRE(lengths_ref.dtype == "I32");
  REQUIRE(lengths_ref.nbytes == sizeof(int32_t));
  REQUIRE(vt::LoadUnaligned<int32_t>(lengths_ref.data) == 4100);

  Queue gpu(vt::DeviceType::kXPU);
  Buffer query(gpu.q, DType::kF16, {4, 24, 256});
  Buffer output(gpu.q, DType::kF16, {4, 24, 256});
  // Rebuild the actual head-interleaved layout, including physical block IDs,
  // with NaN poison everywhere the original capture deliberately did not read.
  Buffer cache(gpu.q, DType::kI8, {180, 1600, 4, 512});
  std::vector<unsigned char> cache_host(cache.bytes, 0x7f);
  for (int p = 0; p < 3; ++p) {
    const int block = vt::LoadUnaligned<int32_t>(ids.data + p * 4);
    const int rows = p == 2 ? 900 : 1600;
    REQUIRE(block == meta.at("pages").at(p).at("physical_block"));
    REQUIRE(block >= 0);
    REQUIRE(block < 180);
    for (const auto* kind : {"k", "v"}) {
      const auto& source = fixture.Get(std::string(kind) + "_page" + std::to_string(p));
      REQUIRE(source.dtype == "U8");
      REQUIRE((source.shape == std::vector<int64_t>{rows, 4, 256}));
      for (int row = 0; row < rows; ++row) for (int head = 0; head < 4; ++head) {
        const size_t destination = size_t(block) * 3276800 + row * 2048 +
                                   head * 512 + (kind[0] == 'v' ? 256 : 0);
        std::memcpy(cache_host.data() + destination,
                    source.data + size_t(row * 4 + head) * 256, 256);
      }
    }
  }
  Buffer table(gpu.q, DType::kI32, {1, 164});
  Buffer lengths(gpu.q, DType::kI32, {1});
  Buffer offsets(gpu.q, DType::kI32, {2});
  std::vector<int32_t> table_host(164, -1);
  std::memcpy(table_host.data(), ids.data, ids.nbytes);
  query.upload(query_ref.data); cache.upload(cache_host.data());
  table.upload(table_host.data()); lengths.upload(lengths_ref.data);
  offsets.upload(offsets_ref.data);
  auto key = vt::Tensor::Contiguous(cache.tensor.data, DType::kI8, gpu.q.device,
                                     {180, 1600, 4, 256});
  key.stride[0] = 3276800; key.stride[1] = 2048; key.stride[2] = 512;
  auto value = key;
  value.data = static_cast<unsigned char*>(key.data) + 256;
  vt::PagedAttentionArgs args;
  args.kv_cache_dtype = vt::Fp8KVCacheDataType::kFp8E4M3;
  args.max_seq_len = 4100;
  args.scale = 0.0625f;
  args.causal = true;
  struct RestoreMode {
    std::string old;
    bool had;
    ~RestoreMode() {
      if (had) setenv("VT_XPU_ATTENTION", old.c_str(), 1);
      else unsetenv("VT_XPU_ATTENTION");
    }
  } restore{std::getenv("VT_XPU_ATTENTION") ? std::getenv("VT_XPU_ATTENTION") : "",
            std::getenv("VT_XPU_ATTENTION") != nullptr};
  REQUIRE(setenv("VT_XPU_ATTENTION", "verify", 1) == 0);
  if (std::getenv("VT_XPU_PROFILE")) (void)vt::xpu::DrainProfileEvents();
  const auto start = std::chrono::steady_clock::now();
  vt::PagedAttention(gpu.q, output.tensor, query.tensor, key, value,
                     table.tensor, lengths.tensor, offsets.tensor, args);
  vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
  const double cold_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  const auto actual = output.download();
  size_t differences = 0, nonfinite = 0;
  double error2 = 0, norm2 = 0, max_error = 0;
  for (size_t i = 0; i < actual.size(); i += 2) {
    const auto got_bits = vt::LoadUnaligned<uint16_t>(actual.data() + i);
    const auto ref_bits = vt::LoadUnaligned<uint16_t>(output_ref.data + i);
    const double got = vt::F16ToF32(got_bits), ref = vt::F16ToF32(ref_bits);
    differences += got_bits != ref_bits; nonfinite += !std::isfinite(got);
    error2 += (got - ref) * (got - ref); norm2 += ref * ref;
    max_error = std::max(max_error, std::abs(got - ref));
  }
  std::cout << nlohmann::json{{"event", "p1_original_C1_page1600_Q4"},
      {"half_differences", differences}, {"nonfinite", nonfinite},
      {"relative_l2", std::sqrt(error2 / std::max(norm2, 1e-30))},
      {"max_abs", max_error}, {"cold_complete_operator_ms", cold_ms}}.dump() << '\n';
  CHECK(nonfinite == 0);
  CHECK(differences == 0);
  if (std::getenv("VT_XPU_PROFILE")) {
    int packs = 0, unpacks = 0, generic = 0;
    for (const auto& event : vt::xpu::DrainProfileEvents()) {
      packs += event.stage == "attention_verify_pack";
      unpacks += event.stage == "attention_verify_unpack";
      generic += event.stage == "attention_split_partial";
    }
    CHECK(packs == 1);
    CHECK(unpacks == 1);
    CHECK(generic == 0);
  }
  xpu_test::SameBytes(cache.download(), cache_host);
  xpu_test::SameBytes(query.download(),
      std::vector<unsigned char>(query_ref.data, query_ref.data + query_ref.nbytes));
  xpu_test::SameBytes(table.download(),
      std::vector<unsigned char>(reinterpret_cast<const unsigned char*>(table_host.data()),
                                reinterpret_cast<const unsigned char*>(table_host.data() + 164)));
  xpu_test::SameBytes(lengths.download(),
      std::vector<unsigned char>(lengths_ref.data, lengths_ref.data + lengths_ref.nbytes));
  xpu_test::SameBytes(offsets.download(),
      std::vector<unsigned char>(offsets_ref.data, offsets_ref.data + offsets_ref.nbytes));
  const std::vector<unsigned char> expected(output_ref.data,
                                            output_ref.data + output_ref.nbytes);
  // Public WithOutput must retire unpack and copy before the caller's query
  // storage becomes the result. No captured input is substituted in inference.
  vt::PagedAttention(gpu.q, query.tensor, query.tensor, key, value,
                     table.tensor, lengths.tensor, offsets.tensor, args);
  vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
  xpu_test::SameBytes(query.download(), expected);
  query.upload(query_ref.data);
  if (std::getenv("VT_B70_EXL3_VERIFY_BENCH")) {
    REQUIRE(!std::getenv("VT_XPU_PROFILE"));  // Score outside the profiler.
    for (const auto* mode : {"split", "verify"}) {
      REQUIRE(setenv("VT_XPU_ATTENTION", mode, 1) == 0);
      auto execute = [&] {
        const auto begin = std::chrono::steady_clock::now();
        vt::PagedAttention(gpu.q, output.tensor, query.tensor, key, value,
                           table.tensor, lengths.tensor, offsets.tensor, args);
        vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
      };
      const double first = execute();
      std::vector<double> samples{execute(), execute(), execute()};
      auto sorted = samples;
      std::sort(sorted.begin(), sorted.end());
      std::cout << nlohmann::json{{"event", "p1_complete_operator_timing"},
          {"mode", mode}, {"first_call_ms", first}, {"warm_samples_ms", samples},
          {"warm_median_ms", sorted[1]}}.dump() << '\n';
    }
    xpu_test::SameBytes(output.download(), expected);
  }
}

TEST_CASE("XPU FP8 attention P1: original true C4 verifier isolation and output copies"
          * doctest::skip(!std::getenv("VT_B70_EXL3_VERIFY_BATCH"))) {
  const std::filesystem::path path = std::getenv("VT_B70_EXL3_VERIFY_BATCH");
  const auto fixture = vllm::SafetensorsFile::Open(path.string());
  auto sidecar = path; sidecar.replace_extension(".json");
  std::ifstream input(sidecar); REQUIRE(input.good());
  const auto meta = nlohmann::json::parse(input);
  REQUIRE(meta.at("schema") == "b70-exl3-P1-C4-verifier-batch-v1");
  REQUIRE(meta.at("source_sha256") ==
          "5b203e2397f7db9c87a2fba041eac31975d96d13d3287244c09186aaedab2a38");
  REQUIRE(meta.at("physical_blocks") == 19); REQUIRE(meta.at("page") == 1600);
  REQUIRE(meta.at("base_rows") == 4100); REQUIRE(meta.at("cases").size() == 9);
  const auto request_blocks = meta.at("request_blocks").get<std::vector<std::vector<int32_t>>>();
  REQUIRE((request_blocks == std::vector<std::vector<int32_t>>{{7}, {12, 3}, {17, 0, 9}, {15, 4, 11, 2}}));
  const int base_lengths[] = {1599, 1601, 4100, 4801};
  const auto& keys = fixture.Get("base_keys"); const auto& values = fixture.Get("base_values");
  REQUIRE(keys.dtype == "U8"); REQUIRE(values.dtype == "U8");
  REQUIRE((keys.shape == std::vector<int64_t>{4100, 4, 256})); REQUIRE(values.shape == keys.shape);
  Queue gpu(vt::DeviceType::kXPU); auto& backend = vt::GetBackend(gpu.q.device);
  struct RestoreMode {
    std::string old; bool had;
    ~RestoreMode() { if (had) setenv("VT_XPU_ATTENTION", old.c_str(), 1); else unsetenv("VT_XPU_ATTENTION"); }
  } restore{std::getenv("VT_XPU_ATTENTION") ? std::getenv("VT_XPU_ATTENTION") : "",
            std::getenv("VT_XPU_ATTENTION") != nullptr};
  REQUIRE(setenv("VT_XPU_ATTENTION", "verify", 1) == 0);
  for (const auto& record : meta.at("cases")) {
    const auto id = record.at("id").get<std::string>(); CAPTURE(id);
    const auto order = record.at("request_ids").get<std::vector<int32_t>>();
    auto sorted = order; std::sort(sorted.begin(), sorted.end());
    REQUIRE((sorted == std::vector<int32_t>{0, 1, 2, 3}));
    const auto lens = record.at("lengths").get<std::vector<int32_t>>();
    REQUIRE(lens.size() == 4); for (int r = 0; r < 4; ++r) REQUIRE(lens[r] == base_lengths[order[r]]);
    const auto layout = record.at("layout").get<std::string>();
    REQUIRE((layout == "planar" || layout == "interleaved" || layout == "padded_interleaved"));
    const bool planar = layout == "planar", padded = layout == "padded_interleaved";
    const int64_t row_stride = planar ? 1024 : 2048;
    const int64_t page_stride = (padded ? 1664 : 1600) * row_stride;
    const int64_t v_offset = planar ? 19 * page_stride : 256;
    const int64_t storage_bytes = 19 * page_stride * (planar ? 2 : 1);
    REQUIRE(record.at("KV_strides") == nlohmann::json::array({page_stride, row_stride, row_stride / 4, int64_t{1}}));
    REQUIRE(record.at("V_offset") == v_offset); REQUIRE(record.at("storage_bytes") == storage_bytes);
    REQUIRE(record.at("route").at("packed_query_shape") == nlohmann::json::array({4, 96, 256}));
    REQUIRE(record.at("route").at("physical_offsets") == nlohmann::json::array({0, 1, 2, 3, 4}));
    REQUIRE(record.at("route").at("splits") == 16); REQUIRE(record.at("route").at("tile") == 8);
    const auto prefix = record.at("prefix").get<std::string>();
    const auto& qref = fixture.Get(prefix + "_query"); const auto& yref = fixture.Get(prefix + "_output");
    REQUIRE(qref.dtype == "F16"); REQUIRE(yref.dtype == "F16");
    REQUIRE((qref.shape == std::vector<int64_t>{16, 24, 256})); REQUIRE(yref.shape == qref.shape);
    const std::vector<unsigned char> expected(yref.data, yref.data + yref.nbytes);
    for (uint8_t poison : {uint8_t{0}, uint8_t{0x7f}}) {
      CAPTURE(int(poison));
      Buffer cache(gpu.q, DType::kI8, {storage_bytes}), query(gpu.q, DType::kF16, {16, 24, 256});
      Buffer out(gpu.q, DType::kF16, {16, 24, 256}), table(gpu.q, DType::kI32, {4, 164});
      Buffer lengths(gpu.q, DType::kI32, {4}), offsets(gpu.q, DType::kI32, {5});
      std::vector<unsigned char> state(storage_bytes, poison);
      for (int r = 0; r < 4; ++r) for (int row = 0; row < base_lengths[r]; ++row) for (int h = 0; h < 4; ++h) {
        const size_t src = size_t(((row + r * 337) % 4100) * 4 + h) * 256;
        const size_t dst = request_blocks[r][row / 1600] * page_stride + (row % 1600) * row_stride + h * (row_stride / 4);
        std::memcpy(state.data() + dst, keys.data + src, 256);
        if (!record.at("mutate_request").is_null() && record.at("mutate_request") == r && row >= base_lengths[r] - 3)
          std::memset(state.data() + dst + v_offset, 0x38, 256);
        else std::memcpy(state.data() + dst + v_offset, values.data + src, 256);
      }
      std::vector<int32_t> table_host(4 * 164, -1);
      for (int r = 0; r < 4; ++r) std::copy(request_blocks[order[r]].begin(), request_blocks[order[r]].end(), table_host.begin() + r * 164);
      const int32_t logical[] = {0, 4, 8, 12, 16};
      cache.upload(state.data()); query.upload(qref.data); table.upload(table_host.data());
      lengths.upload(lens.data()); offsets.upload(logical);
      auto key = vt::Tensor::Contiguous(cache.tensor.data, DType::kI8, gpu.q.device, {19, 1600, 4, 256});
      key.stride[0] = page_stride; key.stride[1] = row_stride; key.stride[2] = row_stride / 4;
      auto value = key; value.data = static_cast<unsigned char*>(key.data) + v_offset;
      vt::PagedAttentionArgs args;
      args.scale = 0.0625f; args.max_seq_len = 4801; args.kv_cache_dtype = vt::Fp8KVCacheDataType::kFp8E4M3;
      args.query_start_loc_host = logical;
      const auto run = [&](vt::Tensor& target) {
        vt::PagedAttention(gpu.q, target, query.tensor, key, value, table.tensor, lengths.tensor, offsets.tensor, args);
        backend.Synchronize(gpu.q);
      };
      if (std::getenv("VT_XPU_PROFILE")) (void)vt::xpu::DrainProfileEvents();
      run(out.tensor);
      const auto raw = out.download(); size_t differences = 0, nonfinite = 0;
      for (size_t i = 0; i < raw.size(); i += 2) {
        differences += std::memcmp(raw.data() + i, yref.data + i, 2) != 0;
        nonfinite += !std::isfinite(vt::F16ToF32(vt::LoadUnaligned<uint16_t>(raw.data() + i)));
      }
      std::cout << nlohmann::json{{"event", "p1_C4_family"}, {"case", id}, {"poison", int(poison)},
          {"half_differences", differences}, {"nonfinite", nonfinite}}.dump() << '\n';
      CHECK(differences == 0); CHECK(nonfinite == 0);
      if (std::getenv("VT_XPU_PROFILE")) {
        int packs = 0, unpacks = 0, generic = 0;
        for (const auto& event : vt::xpu::DrainProfileEvents()) {
          packs += event.stage == "attention_verify_pack"; unpacks += event.stage == "attention_verify_unpack";
          generic += event.stage == "attention_split_partial";
        }
        CHECK(packs == 1); CHECK(unpacks == 1); CHECK(generic == 0);
      }
      xpu_test::SameBytes(query.download(), std::vector<unsigned char>(qref.data, qref.data + qref.nbytes));
      if (std::getenv("VT_B70_EXL3_VERIFY_BATCH_BENCH") &&
          id == "c4-q4-interleaved-ordinary" && poison == 0) {
        REQUIRE(std::getenv("VT_XPU_PROFILE") == nullptr);
        for (const char* mode : {"split", "verify"}) {
          REQUIRE(setenv("VT_XPU_ATTENTION", mode, 1) == 0);
          run(out.tensor);  // compile/cache and warm setup excluded from scores
          std::vector<double> samples;
          for (int repeat = 0; repeat < 3; ++repeat) {
            const auto start = std::chrono::steady_clock::now();
            run(out.tensor);
            samples.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());
          }
          auto sorted = samples; std::sort(sorted.begin(), sorted.end());
          std::cout << nlohmann::json{{"event", "p1_C4_complete_operator"},
              {"case", id}, {"mode", mode}, {"samples_ms", samples},
              {"median_ms", sorted[1]}, {"scope", "warm complete batched operator; changed original arithmetic; no model gain"}}.dump() << '\n';
        }
        xpu_test::SameBytes(out.download(), expected);
      }
      // Public strided-output seam materializes and retires one contiguous result.
      Buffer padded_out(gpu.q, DType::kF16, {16, 24, 264});
      std::vector<unsigned char> padded_bytes(padded_out.bytes, 0xa5);
      padded_out.upload(padded_bytes.data()); auto view = padded_out.tensor; view.shape[2] = 256;
      run(view); const auto copied = padded_out.download();
      bool active_exact = true, guards_exact = true;
      for (int h = 0; h < 16 * 24; ++h) {
        active_exact &= std::memcmp(copied.data() + h * 528, expected.data() + h * 512, 512) == 0;
        guards_exact &= std::memcmp(copied.data() + h * 528 + 512, padded_bytes.data() + h * 528 + 512, 16) == 0;
      }
      CHECK(active_exact); CHECK(guards_exact);
      run(query.tensor); xpu_test::SameBytes(query.download(), expected);
      xpu_test::SameBytes(cache.download(), state);
      xpu_test::SameBytes(table.download(), std::vector<unsigned char>(reinterpret_cast<const unsigned char*>(table_host.data()), reinterpret_cast<const unsigned char*>(table_host.data() + table_host.size())));
      xpu_test::SameBytes(lengths.download(), std::vector<unsigned char>(reinterpret_cast<const unsigned char*>(lens.data()), reinterpret_cast<const unsigned char*>(lens.data() + lens.size())));
      xpu_test::SameBytes(offsets.download(), std::vector<unsigned char>(reinterpret_cast<const unsigned char*>(logical), reinterpret_cast<const unsigned char*>(logical + 5)));
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU FP8 attention P7: original C2 C3 Q4 partial batches and graph guards"
          * doctest::skip(!std::getenv("VT_B70_EXL3_VERIFY_PARTIAL"))) {
  const std::filesystem::path path = std::getenv("VT_B70_EXL3_VERIFY_PARTIAL");
  const auto fixture = vllm::SafetensorsFile::Open(path.string());
  auto sidecar = path; sidecar.replace_extension(".json");
  std::ifstream input(sidecar); REQUIRE(input.good());
  const auto meta = nlohmann::json::parse(input);
  REQUIRE(meta.at("schema") == "b70-exl3-C2-C3-Q4-original-v1");
  REQUIRE(meta.at("source_sha256") == "5b203e2397f7db9c87a2fba041eac31975d96d13d3287244c09186aaedab2a38");
  REQUIRE(meta.at("physical_blocks") == 63); REQUIRE(meta.at("page") == 1600);
  REQUIRE(meta.at("columns") == 164); REQUIRE(meta.at("cases").size() == 12);
  const auto& keys = fixture.Get("base_keys"); const auto& values = fixture.Get("base_values");
  REQUIRE(keys.dtype == "U8"); REQUIRE(values.dtype == "U8");
  REQUIRE((keys.shape == std::vector<int64_t>{4100, 4, 256})); REQUIRE(values.shape == keys.shape);
  Queue gpu(vt::DeviceType::kXPU); auto& backend = vt::GetBackend(gpu.q.device);
  struct RestoreMode {
    std::string old; bool had;
    ~RestoreMode() { if (had) setenv("VT_XPU_ATTENTION", old.c_str(), 1); else unsetenv("VT_XPU_ATTENTION"); }
  } restore{std::getenv("VT_XPU_ATTENTION") ? std::getenv("VT_XPU_ATTENTION") : "",
            std::getenv("VT_XPU_ATTENTION") != nullptr};
  struct Graph {
    vt::Backend& backend; void* handle = nullptr;
    ~Graph() { if (handle) { try { backend.DestroyGraph(handle); } catch (...) {} } }
  } graph{backend};
  for (const auto& record : meta.at("cases")) {
    const auto id = record.at("id").get<std::string>(); CAPTURE(id);
    const auto order = record.at("request_ids").get<std::vector<int32_t>>();
    const int n = int(order.size()), rows = n * 4;
    REQUIRE((n == 2 || n == 3)); auto sorted = order; std::sort(sorted.begin(), sorted.end());
    for (int r = 0; r < n; ++r) REQUIRE(sorted[r] == r);
    const auto base = record.at("base_lengths").get<std::vector<int32_t>>();
    const auto lens = record.at("lengths").get<std::vector<int32_t>>();
    REQUIRE(base.size() == 3); REQUIRE(lens.size() == size_t(n));
    const auto logical = record.at("logical_offsets").get<std::vector<int32_t>>();
    REQUIRE(logical.size() == size_t(n + 1));
    for (int r = 0; r <= n; ++r) REQUIRE(logical[r] == r * 4);
    REQUIRE(record.at("route").at("packed_query_shape") == nlohmann::json::array({n, 96, 256}));
    REQUIRE(record.at("route").at("splits") == 16); REQUIRE(record.at("route").at("tile") == 8);
    const auto prefix = record.at("prefix").get<std::string>();
    const auto& qr = fixture.Get(prefix + "_query"); const auto& yr = fixture.Get(prefix + "_output");
    REQUIRE(qr.dtype == "F16"); REQUIRE(yr.dtype == "F16");
    REQUIRE((qr.shape == std::vector<int64_t>{rows, 24, 256})); REQUIRE(yr.shape == qr.shape);
    const std::vector<unsigned char> expected(yr.data, yr.data + yr.nbytes);
    for (uint8_t poison : {uint8_t{0}, uint8_t{0x7f}}) {
      CAPTURE(int(poison));
      Buffer cache(gpu.q, DType::kI8, {63 * 1600 * 2048});
      Buffer query(gpu.q, DType::kF16, {rows, 24, 256}), out(gpu.q, DType::kF16, {rows, 24, 256});
      Buffer table(gpu.q, DType::kI32, {n, 164}), lengths(gpu.q, DType::kI32, {n});
      Buffer offsets(gpu.q, DType::kI32, {n + 1});
      std::vector<unsigned char> state(cache.bytes, poison);
      for (int r = 0; r < 3; ++r) for (int row = 0; row < base[r]; ++row) for (int h = 0; h < 4; ++h) {
        const size_t src = size_t(((row + r * 337) % 4100) * 4 + h) * 256;
        const size_t dst = size_t(r * 21 + row / 1600) * 1600 * 2048 + (row % 1600) * 2048 + h * 512;
        std::memcpy(state.data() + dst, keys.data + src, 256);
        if (!record.at("mutate_request").is_null() && record.at("mutate_request") == r && row >= base[r] - 3)
          std::memset(state.data() + dst + 256, 0x38, 256);
        else std::memcpy(state.data() + dst + 256, values.data + src, 256);
      }
      std::vector<int32_t> bt(n * 164, -1);
      for (int r = 0; r < n; ++r) {
        REQUIRE(lens[r] == base[order[r]]);
        for (int b = 0; b < 21; ++b) bt[r * 164 + b] = order[r] * 21 + b;
      }
      cache.upload(state.data()); query.upload(qr.data); table.upload(bt.data());
      lengths.upload(lens.data()); offsets.upload(logical.data());
      auto key = vt::Tensor::Contiguous(cache.tensor.data, DType::kI8, gpu.q.device, {63, 1600, 4, 256});
      key.stride[0] = 1600 * 2048; key.stride[1] = 2048; key.stride[2] = 512;
      auto value = key; value.data = static_cast<unsigned char*>(key.data) + 256;
      vt::PagedAttentionArgs args;
      args.scale = 0.0625f; args.max_seq_len = record.at("max_keys").get<int32_t>();
      REQUIRE(record.at("route").at("max_keys") == args.max_seq_len);
      args.kv_cache_dtype = vt::Fp8KVCacheDataType::kFp8E4M3; args.query_start_loc_host = logical.data();
      const auto submit = [&](vt::Tensor& target) {
        vt::PagedAttention(gpu.q, target, query.tensor, key, value, table.tensor, lengths.tensor, offsets.tensor, args);
      };
      const auto run = [&](const char* mode) {
        REQUIRE(setenv("VT_XPU_ATTENTION", mode, 1) == 0);
        submit(out.tensor); backend.Synchronize(gpu.q);
      };
      if (std::getenv("VT_XPU_PROFILE")) (void)vt::xpu::DrainProfileEvents();
      run("verify"); xpu_test::SameBytes(out.download(), expected);
      if (std::getenv("VT_XPU_PROFILE")) {
        int packs = 0, generic = 0;
        for (const auto& e : vt::xpu::DrainProfileEvents()) {
          packs += e.stage == "attention_verify_pack"; generic += e.stage == "attention_split_partial";
        }
        CHECK(packs == 1); CHECK(generic == 0);
      }
      // With the stable model bound, capture shorter metadata then grow to the
      // exact original full operands within the page, without stale host lengths.
      const bool rounded = id.ends_with("rounded");
      if (rounded) {
        auto shorter = lens; for (auto& length : shorter) --length;
        lengths.upload(shorter.data()); run("verify");
      }
      backend.BeginCapture(gpu.q); submit(out.tensor); graph.handle = backend.EndCaptureGraph(gpu.q);
      lengths.upload(lens.data());
      backend.ReplayGraph(gpu.q, graph.handle); backend.Synchronize(gpu.q);
      xpu_test::SameBytes(out.download(), expected);
      auto disagreement = logical; disagreement[1] = 3; offsets.upload(disagreement.data());
      const auto before = out.download();
      CHECK_THROWS(backend.ReplayGraph(gpu.q, graph.handle)); xpu_test::SameBytes(out.download(), before);
      CHECK_THROWS(run("verify")); xpu_test::SameBytes(out.download(), before);
      offsets.upload(logical.data()); backend.ReplayGraph(gpu.q, graph.handle); backend.Synchronize(gpu.q);
      xpu_test::SameBytes(out.download(), expected);
      backend.DestroyGraph(graph.handle); graph.handle = nullptr;
      CHECK(vt::xpu::GetMemoryInfo().graph_device_bytes == 0);
      const auto bound = args.max_seq_len;
      args.max_seq_len = *std::max_element(lens.begin(), lens.end()) - 1;
      CHECK_THROWS(run("verify")); xpu_test::SameBytes(out.download(), expected);
      args.max_seq_len = bound;
      // No shape-only admission; ragged and missing hints retain the generic control.
      args.query_start_loc_host = nullptr; args.uniform_spec_query_len = 4;
      run("split"); const auto generic = out.download(); run("verify");
      xpu_test::SameBytes(out.download(), generic);
      auto ragged = logical; ragged[1] = 3; args.query_start_loc_host = ragged.data(); offsets.upload(ragged.data());
      run("split"); const auto generic_ragged = out.download(); run("verify");
      xpu_test::SameBytes(out.download(), generic_ragged);
      args.query_start_loc_host = logical.data(); offsets.upload(logical.data()); run("verify");
      xpu_test::SameBytes(out.download(), expected);
      Buffer padded(gpu.q, DType::kF16, {rows, 24, 264});
      std::vector<unsigned char> guards(padded.bytes, 0xa5); padded.upload(guards.data());
      auto view = padded.tensor; view.shape[2] = 256; submit(view); backend.Synchronize(gpu.q);
      const auto copied = padded.download(); bool exact = true, untouched = true;
      for (int row = 0; row < rows * 24; ++row) {
        exact &= std::memcmp(copied.data() + row * 528, expected.data() + row * 512, 512) == 0;
        untouched &= std::memcmp(copied.data() + row * 528 + 512, guards.data() + row * 528 + 512, 16) == 0;
      }
      CHECK(exact); CHECK(untouched);
      submit(query.tensor); backend.Synchronize(gpu.q); xpu_test::SameBytes(query.download(), expected);
      query.upload(qr.data);
      xpu_test::SameBytes(cache.download(), state);
      if (std::getenv("VT_B70_EXL3_VERIFY_PARTIAL_BENCH") && poison == 0 && rounded) {
        REQUIRE(std::getenv("VT_XPU_PROFILE") == nullptr);
        for (const char* mode : {"split", "verify"}) {
          run(mode); std::vector<double> samples;
          for (int repeat = 0; repeat < 3; ++repeat) {
            const auto start = std::chrono::steady_clock::now(); run(mode);
            samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
          }
          auto sorted_samples = samples; std::sort(sorted_samples.begin(), sorted_samples.end());
          std::cout << nlohmann::json{{"event", "partial_Q4_complete_operator"}, {"case", id},
              {"mode", mode}, {"samples_ms", samples}, {"median_ms", sorted_samples[1]}}.dump() << '\n';
        }
        xpu_test::SameBytes(out.download(), expected);
      }
      std::cout << nlohmann::json{{"event", "partial_Q4_original_exact"}, {"case", id}, {"poison", int(poison)}}.dump() << '\n';
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU FP8 attention P1: C4 uniform metadata proof and ragged generic remainder"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE"))) {
  Queue gpu(vt::DeviceType::kXPU); auto& backend = vt::GetBackend(gpu.q.device);
  Fixture f(gpu.q, 4, 4, 1601, true, false, 1600, DType::kF16, DType::kF16, DType::kF16, true);
  Buffer table(gpu.q, DType::kI32, {4, 2});
  const int32_t blocks[] = {7, 6, 5, 4, 3, 2, 1, 0};
  const int32_t lens[] = {1601, 1600, 1599, 1598};
  const int32_t uniform[] = {0, 4, 8, 12, 16};
  table.upload(blocks); f.lens.upload(lens);
  f.args.query_start_loc_host = uniform;
  struct RestoreMode {
    std::string old; bool had;
    ~RestoreMode() { if (had) setenv("VT_XPU_ATTENTION", old.c_str(), 1); else unsetenv("VT_XPU_ATTENTION"); }
  } restore{std::getenv("VT_XPU_ATTENTION") ? std::getenv("VT_XPU_ATTENTION") : "",
            std::getenv("VT_XPU_ATTENTION") != nullptr};
  const auto run = [&](const char* mode) {
    REQUIRE(setenv("VT_XPU_ATTENTION", mode, 1) == 0);
    vt::PagedAttention(gpu.q, f.out.tensor, f.query.tensor, f.kc, f.vc, table.tensor, f.lens.tensor, f.offsets.tensor, f.args);
    backend.Synchronize(gpu.q);
  };
  (void)vt::xpu::DrainProfileEvents(); run("verify");
  int packs = 0;
  for (const auto& e : vt::xpu::DrainProfileEvents()) packs += e.stage == "attention_verify_pack";
  CHECK(packs == 1);
  const auto cache_before = f.cache.download(), query_before = f.query.download();
  const auto table_before = table.download(), lens_before = f.lens.download();
  // Uniform shape alone or the speculative routing hint is insufficient.
  f.args.query_start_loc_host = nullptr; f.args.uniform_spec_query_len = 4;
  run("split"); const auto generic_uniform = f.out.download();
  (void)vt::xpu::DrainProfileEvents(); run("verify");
  xpu_test::SameBytes(f.out.download(), generic_uniform);
  packs = 0; for (const auto& e : vt::xpu::DrainProfileEvents()) packs += e.stage == "attention_verify_pack";
  CHECK(packs == 0);
  // A true ragged host/device contract uses the generic remainder, not four
  // serial C1 launches or unchecked total/requests division. Output tail stays untouched.
  const int32_t ragged[] = {0, 2, 6, 10, 14};
  f.args.query_start_loc_host = ragged; f.offsets.upload(ragged);
  f.query.tensor.shape[0] = f.out.tensor.shape[0] = 14;
  f.out.put(std::vector<float>(16 * 24 * 256, -7));
  run("split"); const auto generic_ragged = f.out.download();
  (void)vt::xpu::DrainProfileEvents(); run("verify");
  xpu_test::SameBytes(f.out.download(), generic_ragged);
  packs = 0; for (const auto& e : vt::xpu::DrainProfileEvents()) packs += e.stage == "attention_verify_pack";
  CHECK(packs == 0);
  bool tail_untouched = true;
  for (size_t i = size_t(14 * 24 * 256) * 2; i < generic_ragged.size(); i += 2)
    tail_untouched &= vt::F16ToF32(vt::LoadUnaligned<uint16_t>(generic_ragged.data() + i)) == -7;
  CHECK(tail_untouched);
  // Valid ragged device offsets with total16 must not be packed when the host
  // claims four uniform rows. This mismatch is rejected before any writes.
  f.query.tensor.shape[0] = f.out.tensor.shape[0] = 16;
  f.args.query_start_loc_host = uniform;
  const int32_t disagreement[] = {0, 3, 8, 12, 16};
  f.offsets.upload(disagreement); const auto rejected_out = f.out.download();
  CHECK_THROWS(run("verify")); xpu_test::SameBytes(f.out.download(), rejected_out);
  f.offsets.upload(uniform); f.args.max_seq_len = 1600;
  CHECK_THROWS(run("verify")); xpu_test::SameBytes(f.out.download(), rejected_out);
  f.args.max_seq_len = 1601; run("verify");
  xpu_test::SameBytes(f.cache.download(), cache_before); xpu_test::SameBytes(f.query.download(), query_before);
  xpu_test::SameBytes(table.download(), table_before); xpu_test::SameBytes(f.lens.download(), lens_before);
  xpu_test::SameBytes(f.offsets.download(), std::vector<unsigned char>(reinterpret_cast<const unsigned char*>(uniform), reinterpret_cast<const unsigned char*>(uniform + 5)));
  // A generic C4 graph has no packed-layout proof or exact verifier bound.
  // Increasing fresh device lengths with unchanged32-part arithmetic must
  // preserve the existing generic replay, even when the host has uniform Q4.
  const int32_t short_lens[] = {1600, 1600, 1599, 1598};
  f.lens.upload(short_lens); f.args.max_seq_len = 1600; run("split");
  struct GenericGraph {
    vt::Backend& b; void* handle = nullptr;
    ~GenericGraph() { if (handle) { try { b.DestroyGraph(handle); } catch (...) {} } }
  } graph{backend};
  backend.BeginCapture(gpu.q);
  vt::PagedAttention(gpu.q, f.out.tensor, f.query.tensor, f.kc, f.vc,
                     table.tensor, f.lens.tensor, f.offsets.tensor, f.args);
  graph.handle = backend.EndCaptureGraph(gpu.q);
  f.lens.upload(lens); f.args.max_seq_len = 1601; run("split");
  const auto fresh_generic = f.out.download();
  backend.ReplayGraph(gpu.q, graph.handle);
  xpu_test::SameBytes(f.out.download(), fresh_generic);
  backend.DestroyGraph(graph.handle); graph.handle = nullptr;
  CHECK(vt::xpu::GetMemoryInfo().graph_device_bytes == 0);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU FP8 attention P1: original C1 verifier family boundaries and layouts"
          * doctest::skip(!std::getenv("VT_B70_EXL3_VERIFY_FAMILY"))) {
  const std::filesystem::path path = std::getenv("VT_B70_EXL3_VERIFY_FAMILY");
  const auto fixture = vllm::SafetensorsFile::Open(path.string());
  auto sidecar = path;
  sidecar.replace_extension(".json");
  std::ifstream input(sidecar);
  REQUIRE(input.good());
  const auto meta = nlohmann::json::parse(input);
  REQUIRE(meta.at("schema") == "b70-exl3-P1-C1-verifier-family-v1");
  REQUIRE(meta.at("source_sha256") ==
          "5b203e2397f7db9c87a2fba041eac31975d96d13d3287244c09186aaedab2a38");
  REQUIRE(meta.at("physical_blocks") == 29);
  REQUIRE(meta.at("page") == 1600);
  REQUIRE(meta.at("base_rows") == 4100);
  REQUIRE(meta.at("cases").size() == 39);
  auto blocks = meta.at("block_ids").get<std::vector<int32_t>>();
  REQUIRE(blocks.size() == 29);
  auto sorted_blocks = blocks;
  std::sort(sorted_blocks.begin(), sorted_blocks.end());
  for (int i = 0; i < 29; ++i) REQUIRE(sorted_blocks[i] == i);
  const auto& keys = fixture.Get("base_keys");
  const auto& values = fixture.Get("base_values");
  REQUIRE(keys.dtype == "U8"); REQUIRE(values.dtype == "U8");
  REQUIRE((keys.shape == std::vector<int64_t>{4100, 4, 256}));
  REQUIRE(values.shape == keys.shape);
  Queue gpu(vt::DeviceType::kXPU);
  struct RestoreMode {
    std::string old;
    bool had;
    ~RestoreMode() {
      if (had) setenv("VT_XPU_ATTENTION", old.c_str(), 1);
      else unsetenv("VT_XPU_ATTENTION");
    }
  } restore{std::getenv("VT_XPU_ATTENTION") ? std::getenv("VT_XPU_ATTENTION") : "",
            std::getenv("VT_XPU_ATTENTION") != nullptr};
  REQUIRE(setenv("VT_XPU_ATTENTION", "verify", 1) == 0);
  for (const auto& record : meta.at("cases")) {
    const auto id = record.at("id").get<std::string>();
    const int rows = record.at("queries").get<int>();
    const int length = record.at("length").get<int>();
    CAPTURE(id);
    REQUIRE(rows >= 2); REQUIRE(rows <= 5);
    REQUIRE((length == 1599 || length == 1600 || length == 1601 || length == 4096 ||
             length == 4100 || length == 4799 || length == 4800 || length == 4801 ||
             length == 32768));
    const auto layout = record.at("layout").get<std::string>();
    REQUIRE((layout == "planar" || layout == "interleaved" || layout == "padded_interleaved"));
    const bool planar = layout == "planar", padded = layout == "padded_interleaved";
    const int64_t row_stride = planar ? 1024 : 2048;
    const int64_t page_stride = (padded ? 1664 : 1600) * row_stride;
    const int64_t v_offset = planar ? 29 * page_stride : 256;
    const int64_t storage_bytes = 29 * page_stride * (planar ? 2 : 1);
    REQUIRE(record.at("KV_strides") ==
            nlohmann::json::array({page_stride, row_stride, row_stride / 4, int64_t{1}}));
    REQUIRE(record.at("V_offset") == v_offset);
    REQUIRE(record.at("storage_bytes") == storage_bytes);
    REQUIRE(record.at("ordinary_observed_repeat_exact") == true);
    REQUIRE(record.at("route").at("splits") == (rows == 2 ? 32 : rows == 3 ? 8 : 16));
    REQUIRE(record.at("route").at("tile") == 8);
    const auto prefix = record.at("prefix").get<std::string>();
    const auto& qref = fixture.Get(prefix + "_query");
    const auto& yref = fixture.Get(prefix + "_output");
    REQUIRE(qref.dtype == "F16"); REQUIRE(yref.dtype == "F16");
    REQUIRE((qref.shape == std::vector<int64_t>{rows, 24, 256}));
    REQUIRE(yref.shape == qref.shape);
    const std::vector<unsigned char> expected(yref.data, yref.data + yref.nbytes);
    for (const uint8_t poison : {uint8_t{0}, uint8_t{0x7f}}) {
      CAPTURE(int(poison));
      Buffer cache(gpu.q, DType::kI8, {storage_bytes});
      Buffer query(gpu.q, DType::kF16, {rows, 24, 256});
      Buffer output(gpu.q, DType::kF16, {rows, 24, 256});
      Buffer table(gpu.q, DType::kI32, {1, 164});
      Buffer lengths(gpu.q, DType::kI32, {1});
      Buffer offsets(gpu.q, DType::kI32, {2});
      std::vector<unsigned char> state(storage_bytes, poison);
      for (int row = 0; row < length; ++row) for (int head = 0; head < 4; ++head) {
        const size_t src = size_t((row % 4100) * 4 + head) * 256;
        const size_t dst = blocks[row / 1600] * page_stride + (row % 1600) * row_stride +
                           head * (row_stride / 4);
        std::memcpy(state.data() + dst, keys.data + src, 256);
        if (record.at("mutate_tail_values").get<bool>() && row >= length - 3)
          std::memset(state.data() + dst + v_offset, 0x38, 256);
        else std::memcpy(state.data() + dst + v_offset, values.data + src, 256);
      }
      std::vector<int32_t> table_host(164, -1);
      std::copy_n(blocks.begin(), (length + 1599) / 1600, table_host.begin());
      const int32_t lens_host[] = {length}, offsets_host[] = {0, rows};
      cache.upload(state.data()); query.upload(qref.data);
      table.upload(table_host.data()); lengths.upload(lens_host); offsets.upload(offsets_host);
      auto key = vt::Tensor::Contiguous(cache.tensor.data, DType::kI8, gpu.q.device,
                                       {29, 1600, 4, 256});
      key.stride[0] = page_stride; key.stride[1] = row_stride; key.stride[2] = row_stride / 4;
      auto value = key;
      value.data = static_cast<unsigned char*>(key.data) + v_offset;
      vt::PagedAttentionArgs args;
      args.kv_cache_dtype = vt::Fp8KVCacheDataType::kFp8E4M3;
      args.max_seq_len = length; args.scale = 0.0625f; args.causal = true;
      if (std::getenv("VT_XPU_PROFILE")) (void)vt::xpu::DrainProfileEvents();
      vt::PagedAttention(gpu.q, output.tensor, query.tensor, key, value,
                         table.tensor, lengths.tensor, offsets.tensor, args);
      vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
      const auto raw = output.download();
      size_t differences = 0, nonfinite = 0;
      for (size_t i = 0; i < raw.size(); i += 2) {
        differences += std::memcmp(raw.data() + i, yref.data + i, 2) != 0;
        nonfinite += !std::isfinite(vt::F16ToF32(vt::LoadUnaligned<uint16_t>(raw.data() + i)));
      }
      std::cout << nlohmann::json{{"event", "p1_C1_family"}, {"case", id},
          {"poison", int(poison)}, {"half_differences", differences},
          {"nonfinite", nonfinite}}.dump() << '\n';
      CHECK(differences == 0); CHECK(nonfinite == 0);
      if (std::getenv("VT_XPU_PROFILE")) {
        int packs = 0, unpacks = 0, generic = 0;
        for (const auto& event : vt::xpu::DrainProfileEvents()) {
          packs += event.stage == "attention_verify_pack";
          unpacks += event.stage == "attention_verify_unpack";
          generic += event.stage == "attention_split_partial";
        }
        CHECK(packs == 1); CHECK(unpacks == 1); CHECK(generic == 0);
      }
      xpu_test::SameBytes(query.download(),
          std::vector<unsigned char>(qref.data, qref.data + qref.nbytes));
      vt::PagedAttention(gpu.q, query.tensor, query.tensor, key, value,
                         table.tensor, lengths.tensor, offsets.tensor, args);
      vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
      xpu_test::SameBytes(query.download(), expected);
      xpu_test::SameBytes(cache.download(), state);
      xpu_test::SameBytes(table.download(),
          std::vector<unsigned char>(reinterpret_cast<const unsigned char*>(table_host.data()),
                                    reinterpret_cast<const unsigned char*>(table_host.data() + 164)));
      xpu_test::SameBytes(lengths.download(),
          std::vector<unsigned char>(reinterpret_cast<const unsigned char*>(lens_host),
                                    reinterpret_cast<const unsigned char*>(lens_host + 1)));
      xpu_test::SameBytes(offsets.download(),
          std::vector<unsigned char>(reinterpret_cast<const unsigned char*>(offsets_host),
                                    reinterpret_cast<const unsigned char*>(offsets_host + 2)));
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU FP8 attention P1: verifier declines unsupported forms before metadata writes"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE"))) {
  Queue gpu(vt::DeviceType::kXPU);
  Fixture f(gpu.q, 1, 4, 4100, true, false, 1600,
            DType::kF16, DType::kF16, DType::kF16, true);
  auto& backend = vt::GetBackend(gpu.q.device);
  Buffer table(gpu.q, DType::kI32, {1, 3});
  const int32_t table_data[] = {2, 1, 0};
  table.upload(table_data);
  const auto cache_before = f.cache.download(), query_before = f.query.download();
  Buffer padded_query(gpu.q, DType::kF16, {4, 24 * 256 + 16});
  std::vector<unsigned char> padded_host(padded_query.bytes, 0xff);
  for (int row = 0; row < 4; ++row)
    std::memcpy(padded_host.data() + row * (24 * 256 + 16) * 2,
                query_before.data() + row * 24 * 256 * 2, 24 * 256 * 2);
  padded_query.upload(padded_host.data());
  auto padded = vt::Tensor::Contiguous(padded_query.tensor.data, DType::kF16,
                                      gpu.q.device, {4, 24, 256});
  padded.stride[0] = 24 * 256 + 16;
  struct RestoreMode {
    std::string old;
    bool had;
    ~RestoreMode() {
      if (had) setenv("VT_XPU_ATTENTION", old.c_str(), 1);
      else unsetenv("VT_XPU_ATTENTION");
    }
  } restore{std::getenv("VT_XPU_ATTENTION") ? std::getenv("VT_XPU_ATTENTION") : "",
            std::getenv("VT_XPU_ATTENTION") != nullptr};
  auto execute = [&](const char* mode, const vt::Tensor& q, const vt::Tensor& bt,
                     const vt::PagedAttentionArgs& args) {
    REQUIRE(setenv("VT_XPU_ATTENTION", mode, 1) == 0);
    vt::PagedAttention(gpu.q, f.out.tensor, q, f.kc, f.vc,
                       bt, f.lens.tensor, f.offsets.tensor, args);
    backend.Synchronize(gpu.q);
  };
  (void)vt::xpu::DrainProfileEvents();
  execute("verify", f.query.tensor, table.tensor, f.args);
  int admitted = 0;
  for (const auto& event : vt::xpu::DrainProfileEvents())
    admitted += event.stage == "attention_verify_pack";
  REQUIRE(admitted == 1);  // The control really reaches the guarded route.
  for (const int variant : {0, 1, 2, 3, 4, 5, 7}) {
    CAPTURE(variant);
    auto args = f.args;
    if (variant == 0) args.causal = false;
    if (variant == 1) args.k_scale = 0.5f;
    if (variant == 2) args.v_scale = 0.3f;
    if (variant == 3) args.scale = 0.03125f;
    if (variant == 4) args.window_size = vt::AttentionWindow{13, 3};
    if (variant == 5) args.logits_soft_cap = 0.7f;
    const auto& q = f.query.tensor;
    const auto& bt = variant == 7 ? f.table.tensor : table.tensor;
    execute("split", q, bt, args);
    const auto expected = f.out.download();
    (void)vt::xpu::DrainProfileEvents();
    execute("verify", q, bt, args);
    int packs = 0, unpacks = 0;
    for (const auto& event : vt::xpu::DrainProfileEvents()) {
      packs += event.stage == "attention_verify_pack";
      unpacks += event.stage == "attention_verify_unpack";
    }
    CHECK(packs == 0); CHECK(unpacks == 0);
    xpu_test::SameBytes(f.out.download(), expected);
    xpu_test::SameBytes(f.cache.download(), cache_before);
    xpu_test::SameBytes(f.query.download(), query_before);
    xpu_test::SameBytes(padded_query.download(), padded_host);
  }
  execute("verify", f.query.tensor, table.tensor, f.args);
  const auto output_before = f.out.download();
  // Contiguous query/out is a public API requirement, so an unsupported
  // query stride is rejected before the XPU route chooser rather than copied.
  CHECK_THROWS_AS(execute("verify", padded, table.tensor, f.args), std::runtime_error);
  xpu_test::SameBytes(f.out.download(), output_before);
  xpu_test::SameBytes(padded_query.download(), padded_host);
  xpu_test::SameBytes(f.cache.download(), cache_before);
  const int32_t valid_offsets[] = {0, 4}, valid_length[] = {4100};
  for (int invalid = 0; invalid < 3; ++invalid) {
    CAPTURE(invalid);
    table.upload(table_data); f.offsets.upload(valid_offsets); f.lens.upload(valid_length);
    const int32_t bad_table[] = {-1, 1, 0}, bad_offsets[] = {0, 3}, bad_length[] = {3};
    if (invalid == 0) table.upload(bad_table);
    if (invalid == 1) f.offsets.upload(bad_offsets);
    if (invalid == 2) f.lens.upload(bad_length);
    const auto table_before = table.download(), offsets_before = f.offsets.download();
    const auto lengths_before = f.lens.download();
    CHECK_THROWS_AS(execute("verify", f.query.tensor, table.tensor, f.args), std::runtime_error);
    xpu_test::SameBytes(f.out.download(), output_before);
    xpu_test::SameBytes(f.cache.download(), cache_before);
    xpu_test::SameBytes(table.download(), table_before);
    xpu_test::SameBytes(f.offsets.download(), offsets_before);
    xpu_test::SameBytes(f.lens.download(), lengths_before);
  }
}

TEST_CASE("XPU FP8 attention P1: C4 and C1 graphs share scratch with fresh request metadata"
          * doctest::skip(!std::getenv("VT_B70_EXL3_VERIFY_BATCH"))) {
  const std::filesystem::path path = std::getenv("VT_B70_EXL3_VERIFY_BATCH");
  const auto fixture = vllm::SafetensorsFile::Open(path.string());
  auto sidecar = path; sidecar.replace_extension(".json");
  std::ifstream input(sidecar); REQUIRE(input.good());
  const auto meta = nlohmann::json::parse(input);
  REQUIRE(meta.at("schema") == "b70-exl3-P1-C4-verifier-batch-v1");
  REQUIRE(meta.at("physical_blocks") == 19); REQUIRE(meta.at("page") == 1600);
  const auto request_blocks = meta.at("request_blocks").get<std::vector<std::vector<int32_t>>>();
  REQUIRE((request_blocks == std::vector<std::vector<int32_t>>{{7}, {12, 3}, {17, 0, 9}, {15, 4, 11, 2}}));
  const auto& keys = fixture.Get("base_keys"); const auto& values = fixture.Get("base_values");
  REQUIRE(keys.dtype == "U8"); REQUIRE(values.dtype == "U8");
  REQUIRE((keys.shape == std::vector<int64_t>{4100, 4, 256})); REQUIRE(values.shape == keys.shape);
  const auto record = [&](const std::string& suffix) -> nlohmann::json {
    for (const auto& c : meta.at("cases"))
      if (c.at("id") == "c4-q4-interleaved-" + suffix) return c;
    throw std::runtime_error("missing original C4 graph case " + suffix);
  };
  Queue first(vt::DeviceType::kXPU), second(vt::DeviceType::kXPU), eager(vt::DeviceType::kXPU);
  auto& backend = vt::GetBackend(first.q.device); REQUIRE(backend.SupportsGraphCapture());
  const auto before = vt::xpu::GetMemoryInfo();
  const auto captures = backend.GraphsCaptured(), replays = backend.GraphReplays();
  Buffer cache(eager.q, DType::kI8, {19, 1600, 4, 512});
  Buffer table4(eager.q, DType::kI32, {4, 164}), table1(eager.q, DType::kI32, {1, 164});
  Buffer lens4(eager.q, DType::kI32, {4}), lens1(eager.q, DType::kI32, {1});
  Buffer offsets4(eager.q, DType::kI32, {5}), offsets1(eager.q, DType::kI32, {2});
  Buffer query4(first.q, DType::kF16, {16, 24, 256}), query1(second.q, DType::kF16, {4, 24, 256});
  Buffer out4(first.q, DType::kF16, {16, 24, 256}), out1(second.q, DType::kF16, {4, 24, 256});
  Buffer ref4(eager.q, DType::kF16, {16, 24, 256}), ref1(eager.q, DType::kF16, {4, 24, 256});
  // Handles die before the last baked input/output/cache address.
  struct Graphs {
    vt::Backend& backend; std::array<void*, 2> handles{};
    void reset() { for (auto& h : handles) if (h) { backend.DestroyGraph(h); h = nullptr; } }
    ~Graphs() { try { reset(); } catch (...) {} }
  } graphs{backend};
  struct RestoreMode {
    std::string old; bool had;
    ~RestoreMode() { if (had) setenv("VT_XPU_ATTENTION", old.c_str(), 1); else unsetenv("VT_XPU_ATTENTION"); }
  } restore{std::getenv("VT_XPU_ATTENTION") ? std::getenv("VT_XPU_ATTENTION") : "",
            std::getenv("VT_XPU_ATTENTION") != nullptr};
  REQUIRE(setenv("VT_XPU_ATTENTION", "verify", 1) == 0);
  const int32_t logical4[] = {0, 4, 8, 12, 16}, logical1[] = {0, 4}, length1[] = {4100};
  offsets4.upload(logical4); offsets1.upload(logical1); lens1.upload(length1);
  std::vector<int32_t> single_table(164, -1);
  std::copy(request_blocks[2].begin(), request_blocks[2].end(), single_table.begin());
  table1.upload(single_table.data());
  auto key = vt::Tensor::Contiguous(cache.tensor.data, DType::kI8, eager.q.device, {19, 1600, 4, 256});
  key.stride[0] = 1600 * 2048; key.stride[1] = 2048; key.stride[2] = 512;
  auto value = key; value.data = static_cast<unsigned char*>(key.data) + 256;
  vt::PagedAttentionArgs args4;
  args4.scale = 0.0625f; args4.max_seq_len = vt::PagedAttnXpuPackedVerifyBound(4801, 1600); args4.kv_cache_dtype = vt::Fp8KVCacheDataType::kFp8E4M3;
  args4.query_start_loc_host = logical4;
  auto args1 = args4; args1.query_start_loc_host = logical1;
  const auto run = [&](vt::Queue& q, int slot, vt::Tensor& dest) {
    vt::PagedAttention(q, dest, slot == 0 ? query4.tensor : query1.tensor, key, value,
        slot == 0 ? table4.tensor : table1.tensor, slot == 0 ? lens4.tensor : lens1.tensor,
        slot == 0 ? offsets4.tensor : offsets1.tensor, slot == 0 ? args4 : args1);
  };
  const int lengths[] = {1599, 1601, 4100, 4801};
  std::vector<unsigned char> state;
  std::vector<int32_t> active_table, active_lens;
  std::vector<unsigned char> expected4, expected1, query4_bytes, query1_bytes;
  const auto int_bytes = [](const auto& entries) {
    const auto* begin = reinterpret_cast<const unsigned char*>(std::data(entries));
    return std::vector<unsigned char>(begin, begin + std::size(entries) * sizeof(int32_t));
  };
  const auto populate = [&](const std::string& suffix, bool zero_single_query) {
    const auto c = record(suffix); const auto prefix = c.at("prefix").get<std::string>();
    const auto& qref = fixture.Get(prefix + "_query"); const auto& yref = fixture.Get(prefix + "_output");
    REQUIRE((qref.shape == std::vector<int64_t>{16, 24, 256})); REQUIRE(yref.shape == qref.shape);
    auto order = c.at("request_ids").get<std::vector<int32_t>>();
    REQUIRE(order.size() == 4);
    const auto where = std::find(order.begin(), order.end(), 2); REQUIRE(where != order.end());
    const size_t slot2 = size_t(where - order.begin()), one_bytes = 4 * 24 * 256 * 2;
    query1_bytes.assign(qref.data + slot2 * one_bytes, qref.data + (slot2 + 1) * one_bytes);
    expected1.assign(yref.data + slot2 * one_bytes, yref.data + (slot2 + 1) * one_bytes);
    expected4.assign(yref.data, yref.data + yref.nbytes);
    query4_bytes.assign(qref.data, qref.data + qref.nbytes);
    state.assign(cache.bytes, 0x7f);
    for (int r = 0; r < 4; ++r) for (int row = 0; row < lengths[r]; ++row) for (int head = 0; head < 4; ++head) {
      const size_t src = size_t(((row + r * 337) % 4100) * 4 + head) * 256;
      const size_t dst = request_blocks[r][row / 1600] * size_t(1600 * 2048) + (row % 1600) * 2048 + head * 512;
      std::memcpy(state.data() + dst, keys.data + src, 256);
      if (suffix == "tail-values-r2" && r == 2 && row >= lengths[r] - 3) std::memset(state.data() + dst + 256, 0x38, 256);
      else std::memcpy(state.data() + dst + 256, values.data + src, 256);
    }
    cache.upload(state.data()); active_table.assign(4 * 164, -1); active_lens.clear();
    for (int r = 0; r < 4; ++r) {
      std::copy(request_blocks[order[r]].begin(), request_blocks[order[r]].end(), active_table.begin() + r * 164);
      active_lens.push_back(lengths[order[r]]);
    }
    table4.upload(active_table.data()); lens4.upload(active_lens.data()); query4.upload(qref.data);
    if (zero_single_query) { std::fill(query1_bytes.begin(), query1_bytes.end(), 0); query1.upload(query1_bytes.data()); }
    else query1.upload(query1_bytes.data());
  };
  populate("ordinary", false);
  std::array<vt::Queue*, 2> queues{&first.q, &second.q};
  std::array<Buffer*, 2> output{&out4, &out1}, reference{&ref4, &ref1};
  for (int slot = 0; slot < 2; ++slot) {
    run(*queues[slot], slot, output[slot]->tensor); backend.Synchronize(*queues[slot]);
    REQUIRE(vt::xpu::GetMemoryInfo().attention_workspace_bytes == 16 * 1024 * 1024);
    backend.Memset(*queues[slot], output[slot]->tensor.data, 0xcd, output[slot]->bytes); backend.Synchronize(*queues[slot]);
    const auto untouched = output[slot]->download();
    backend.BeginCapture(*queues[slot]); run(*queues[slot], slot, output[slot]->tensor);
    graphs.handles[slot] = backend.EndCaptureGraph(*queues[slot]);
    xpu_test::SameBytes(output[slot]->download(), untouched);
  }
  const auto stable = vt::xpu::GetMemoryInfo(); REQUIRE(stable.graph_count == before.graph_count + 2);
  int step = 0;
  for (const auto* suffix : {"ordinary", "permuted", "tail-values-r2", "permuted", "ordinary"}) {
    CAPTURE(suffix); const bool zero_single = step++ % 2 != 0; populate(suffix, zero_single);
    run(eager.q, 0, ref4.tensor); run(eager.q, 1, ref1.tensor); backend.Synchronize(eager.q);
    backend.ReplayGraph(first.q, graphs.handles[0]); backend.ReplayGraph(second.q, graphs.handles[1]);
    run(eager.q, 0, ref4.tensor);  // Private offsets5/2 share one completion-owned pool.
    for (int slot = 0; slot < 2; ++slot) xpu_test::SameBytes(output[slot]->download(), reference[slot]->download());
    xpu_test::SameBytes(out4.download(), expected4);
    if (!zero_single) xpu_test::SameBytes(out1.download(), expected1);
    xpu_test::SameBytes(query1.download(), query1_bytes); xpu_test::SameBytes(query4.download(), query4_bytes);
    xpu_test::SameBytes(cache.download(), state);
    xpu_test::SameBytes(table4.download(), int_bytes(active_table)); xpu_test::SameBytes(table1.download(), int_bytes(single_table));
    xpu_test::SameBytes(lens4.download(), int_bytes(active_lens)); xpu_test::SameBytes(lens1.download(), int_bytes(length1));
    xpu_test::SameBytes(offsets4.download(), int_bytes(logical4)); xpu_test::SameBytes(offsets1.download(), int_bytes(logical1));
    CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == stable.allocated_bytes);
    CHECK(vt::xpu::GetMemoryInfo().graph_device_bytes == stable.graph_device_bytes);
    CHECK(backend.GraphsCaptured() == captures + 2);
  }
  const auto last4 = out4.download(), last1 = out1.download();
  const int32_t ragged_claim[] = {0, 3, 8, 12, 16}; offsets4.upload(ragged_claim);
  CHECK_THROWS_AS(backend.ReplayGraph(first.q, graphs.handles[0]), std::runtime_error);
  xpu_test::SameBytes(out4.download(), last4); offsets4.upload(logical4);
  auto bad_lens = active_lens; bad_lens[3] = args4.max_seq_len + 1; lens4.upload(bad_lens.data());
  CHECK_THROWS_AS(backend.ReplayGraph(first.q, graphs.handles[0]), std::runtime_error);
  xpu_test::SameBytes(out4.download(), last4); lens4.upload(active_lens.data());
  auto bad_table = active_table; bad_table[2 * 164] = -1; table4.upload(bad_table.data());
  CHECK_THROWS_AS(backend.ReplayGraph(first.q, graphs.handles[0]), std::runtime_error);
  xpu_test::SameBytes(out4.download(), last4); table4.upload(active_table.data());
  xpu_test::SameBytes(out1.download(), last1); xpu_test::SameBytes(cache.download(), state);
  xpu_test::SameBytes(query4.download(), query4_bytes); xpu_test::SameBytes(query1.download(), query1_bytes);
  xpu_test::SameBytes(table4.download(), int_bytes(active_table)); xpu_test::SameBytes(table1.download(), int_bytes(single_table));
  xpu_test::SameBytes(lens4.download(), int_bytes(active_lens)); xpu_test::SameBytes(lens1.download(), int_bytes(length1));
  xpu_test::SameBytes(offsets4.download(), int_bytes(logical4)); xpu_test::SameBytes(offsets1.download(), int_bytes(logical1));
  backend.ReplayGraph(second.q, graphs.handles[0]); backend.ReplayGraph(first.q, graphs.handles[1]);
  graphs.reset();  // Destruction waits for both final cross-queue submissions.
  xpu_test::SameBytes(out4.download(), last4); xpu_test::SameBytes(out1.download(), last1);
  CHECK(vt::xpu::GetMemoryInfo().graph_count == before.graph_count);
  CHECK(vt::xpu::GetMemoryInfo().graph_device_bytes == before.graph_device_bytes);
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == stable.allocated_bytes);
  CHECK(backend.GraphReplays() == replays + 12);
  std::cout << nlohmann::json{{"event", "p1_C4_C1_graph_ownership"}, {"captures", backend.GraphsCaptured() - captures},
      {"successful_replays", backend.GraphReplays() - replays}, {"shared_attention_bytes", vt::xpu::GetMemoryInfo().attention_workspace_bytes},
      {"peak_device_bytes", vt::xpu::GetMemoryInfo().peak_allocated_bytes},
      {"graph_device_bytes_after_retirement", vt::xpu::GetMemoryInfo().graph_device_bytes}}.dump() << '\n';
}

TEST_CASE("XPU FP8 attention P1: C1 graphs share verifier scratch and revalidate metadata"
          * doctest::skip(!std::getenv("VT_B70_EXL3_VERIFY_FAMILY"))) {
  const std::filesystem::path path = std::getenv("VT_B70_EXL3_VERIFY_FAMILY");
  const auto fixture = vllm::SafetensorsFile::Open(path.string());
  auto sidecar = path; sidecar.replace_extension(".json");
  std::ifstream file(sidecar); REQUIRE(file.good());
  const auto meta = nlohmann::json::parse(file);
  REQUIRE(meta.at("physical_blocks") == 29);
  auto record = [&](const std::string& id) -> nlohmann::json {
    for (const auto& item : meta.at("cases")) if (item.at("id") == id) return item;
    throw std::runtime_error("missing original graph fixture case " + id);
  };
  const auto& keys = fixture.Get("base_keys");
  const auto& values = fixture.Get("base_values");
  REQUIRE(keys.dtype == "U8"); REQUIRE(values.dtype == "U8");
  REQUIRE((keys.shape == std::vector<int64_t>{4100, 4, 256}));
  REQUIRE(values.shape == keys.shape);
  auto blocks = meta.at("block_ids").get<std::vector<int32_t>>();
  REQUIRE(blocks.size() == 29);
  Queue first(vt::DeviceType::kXPU), second(vt::DeviceType::kXPU), eager(vt::DeviceType::kXPU);
  auto& backend = vt::GetBackend(first.q.device);
  REQUIRE(backend.SupportsGraphCapture());
  const auto before = vt::xpu::GetMemoryInfo();
  const auto captures = backend.GraphsCaptured(), replays = backend.GraphReplays();
  Buffer cache(eager.q, DType::kI8, {29, 1600, 4, 512});
  Buffer table(eager.q, DType::kI32, {1, 164});
  std::vector<int32_t> table_host(164, -1);
  std::copy_n(blocks.begin(), (32768 + 1599) / 1600, table_host.begin());
  table.upload(table_host.data());
  auto key = vt::Tensor::Contiguous(cache.tensor.data, DType::kI8, eager.q.device, {29, 1600, 4, 256});
  key.stride[0] = 1600 * 2048; key.stride[1] = 2048; key.stride[2] = 512;
  auto value = key; value.data = static_cast<unsigned char*>(key.data) + 256;
  std::vector<unsigned char> state;
  auto populate = [&](int length) {
    state.assign(cache.bytes, 0x7f);
    for (int row = 0; row < length; ++row) for (int head = 0; head < 4; ++head) {
      const size_t src = size_t((row % 4100) * 4 + head) * 256;
      const size_t dst = size_t(blocks[row / 1600]) * 1600 * 2048 +
                         (row % 1600) * 2048 + head * 512;
      std::memcpy(state.data() + dst, keys.data + src, 256);
      std::memcpy(state.data() + dst + 256, values.data + src, 256);
    }
    cache.upload(state.data());
  };
  populate(4100);
  std::array<vt::Queue*, 2> queues{&first.q, &second.q};
  std::array<std::unique_ptr<Buffer>, 2> queries, output, reference, lengths, offsets;
  struct Graphs {
    vt::Backend& backend;
    std::array<void*, 2> handles{};
    void reset() { for (auto& h : handles) if (h) { backend.DestroyGraph(h); h = nullptr; } }
    ~Graphs() { try { reset(); } catch (...) {} }
  } graphs{backend};  // Destroy graphs before any baked buffer address dies.
  struct RestoreMode {
    std::string old; bool had;
    ~RestoreMode() {
      if (had) setenv("VT_XPU_ATTENTION", old.c_str(), 1);
      else unsetenv("VT_XPU_ATTENTION");
    }
  } restore{std::getenv("VT_XPU_ATTENTION") ? std::getenv("VT_XPU_ATTENTION") : "",
            std::getenv("VT_XPU_ATTENTION") != nullptr};
  REQUIRE(setenv("VT_XPU_ATTENTION", "verify", 1) == 0);
  vt::PagedAttentionArgs args;
  args.kv_cache_dtype = vt::Fp8KVCacheDataType::kFp8E4M3;
  args.scale = 0.0625f; args.causal = true;
  args.max_seq_len = 32768;  // Fixed upper bound; active device lengths change.
  auto run = [&](vt::Queue& q, int slot, vt::Tensor& destination) {
    vt::PagedAttention(q, destination, queries[slot]->tensor, key, value,
                       table.tensor, lengths[slot]->tensor, offsets[slot]->tensor, args);
  };
  for (int slot = 0; slot < 2; ++slot) {
    const int qrows = slot == 0 ? 4 : 3;
    const auto& qref = fixture.Get(record("q" + std::to_string(qrows) + "-l4100-interleaved").at("prefix").get<std::string>() + "_query");
    for (auto* buffers : {&queries, &output, &reference})
      (*buffers)[slot] = std::make_unique<Buffer>(*queues[slot], DType::kF16,
                                                 std::initializer_list<int64_t>{qrows, 24, 256});
    lengths[slot] = std::make_unique<Buffer>(*queues[slot], DType::kI32, std::initializer_list<int64_t>{1});
    offsets[slot] = std::make_unique<Buffer>(*queues[slot], DType::kI32, std::initializer_list<int64_t>{2});
    const int32_t active[] = {4100}, qsl[] = {0, qrows};
    queries[slot]->upload(qref.data); lengths[slot]->upload(active); offsets[slot]->upload(qsl);
    run(*queues[slot], slot, output[slot]->tensor); backend.Synchronize(*queues[slot]);
    REQUIRE(vt::xpu::GetMemoryInfo().attention_workspace_bytes == 16 * 1024 * 1024);
    backend.Memset(*queues[slot], output[slot]->tensor.data, 0xcd, output[slot]->bytes);
    backend.Synchronize(*queues[slot]);
    const auto untouched = output[slot]->download();
    backend.BeginCapture(*queues[slot]); run(*queues[slot], slot, output[slot]->tensor);
    graphs.handles[slot] = backend.EndCaptureGraph(*queues[slot]);
    xpu_test::SameBytes(output[slot]->download(), untouched);
  }
  const auto stable = vt::xpu::GetMemoryInfo();
  REQUIRE(stable.graph_count == before.graph_count + 2);
  for (const int length : {4100, 1599, 1600, 1601, 4799, 4800, 4801, 32768, 4096}) {
    CAPTURE(length);
    populate(std::max(length, 4100));  // Two C1 graphs read the same isolated KV fixture.
    const int32_t current[] = {length}; lengths[0]->upload(current);
    const auto& q3 = fixture.Get(record("q3-l4100-interleaved").at("prefix").get<std::string>() + "_query");
    if (length & 1) backend.Memset(second.q, queries[1]->tensor.data, 0, queries[1]->bytes);
    else queries[1]->upload(q3.data);
    backend.Synchronize(second.q);  // Publish the changed query to the eager queue.
    for (int slot = 0; slot < 2; ++slot) run(eager.q, slot, reference[slot]->tensor);
    backend.Synchronize(eager.q);
    backend.ReplayGraph(first.q, graphs.handles[0]);
    backend.ReplayGraph(second.q, graphs.handles[1]);
    // Interleave an eager use of the same private packed scratch while graph
    // consumers may still be in flight on other queues.
    run(eager.q, 0, reference[0]->tensor);
    for (int slot = 0; slot < 2; ++slot)
      xpu_test::SameBytes(output[slot]->download(), reference[slot]->download());
    const auto& expected = fixture.Get(record("q4-l" + std::to_string(length) + "-interleaved").at("prefix").get<std::string>() + "_output");
    xpu_test::SameBytes(output[0]->download(),
        std::vector<unsigned char>(expected.data, expected.data + expected.nbytes));
    CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == stable.allocated_bytes);
    CHECK(vt::xpu::GetMemoryInfo().graph_device_bytes == stable.graph_device_bytes);
    CHECK(backend.GraphsCaptured() == captures + 2);
  }
  const auto previous = output[0]->download();
  const int32_t invalid[] = {3}; lengths[0]->upload(invalid);
  CHECK_THROWS_AS(backend.ReplayGraph(first.q, graphs.handles[0]), std::runtime_error);
  xpu_test::SameBytes(output[0]->download(), previous);
  xpu_test::SameBytes(cache.download(), state);
  const int32_t good[] = {4096}; lengths[0]->upload(good);
  backend.ReplayGraph(second.q, graphs.handles[0]);
  backend.ReplayGraph(first.q, graphs.handles[1]);
  graphs.reset();  // Retire last submissions before releasing baked addresses.
  xpu_test::SameBytes(output[0]->download(), previous);
  CHECK(vt::xpu::GetMemoryInfo().graph_count == before.graph_count);
  CHECK(vt::xpu::GetMemoryInfo().graph_device_bytes == before.graph_device_bytes);
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == stable.allocated_bytes);
  CHECK(backend.GraphReplays() == replays + 20);
  const auto retired = vt::xpu::GetMemoryInfo();
  std::cout << nlohmann::json{{"event", "p1_C1_graph_ownership"},
      {"captures", backend.GraphsCaptured() - captures},
      {"successful_replays", backend.GraphReplays() - replays},
      {"shared_attention_bytes", retired.attention_workspace_bytes},
      {"live_device_bytes", retired.allocated_bytes},
      {"peak_device_bytes", retired.peak_allocated_bytes},
      {"graph_device_bytes_after_retirement", retired.graph_device_bytes}}.dump() << '\n';
}

TEST_CASE("XPU FP8 attention: captured Python M04 Q5 operator replay"
          * doctest::skip(!std::getenv("VT_B70_M04_REPLAY_DIR"))) {
  const std::string directory = std::getenv("VT_B70_M04_REPLAY_DIR");
  auto read = [&](const std::string& name, size_t bytes) {
    std::ifstream file(directory + "/" + name, std::ios::binary);
    REQUIRE(file.good());
    std::vector<unsigned char> data(bytes);
    file.read(reinterpret_cast<char*>(data.data()), data.size());
    REQUIRE(file.gcount() == static_cast<std::streamsize>(bytes));
    REQUIRE(file.peek() == std::char_traits<char>::eof());
    return data;
  };
  std::ifstream manifest(directory + "/metadata.json");
  REQUIRE(manifest.good());
  const auto meta = nlohmann::json::parse(manifest);
  const int tokens = meta.at("q_shape").at(0).get<int>();
  const int heads = meta.at("q_shape").at(1).get<int>();
  const int dim = meta.at("q_shape").at(2).get<int>();
  const int pages = meta.at("kv_shape").at(0).get<int>();
  const int page = meta.at("kv_shape").at(1).get<int>();
  const int kv_heads = meta.at("kv_shape").at(2).get<int>();
  const int used = meta.at("used_k").get<int>();
  REQUIRE(tokens == 5);
  REQUIRE(heads == 24);
  REQUIRE(dim == 256);
  REQUIRE(page == 1664);
  REQUIRE(kv_heads == 4);
  REQUIRE(meta.at("kv_shape").at(3).get<int>() == dim);
  REQUIRE(pages == (used + page - 1) / page);
  REQUIRE(used >= 4096 + tokens);
  REQUIRE(meta.at("cu_seqlens_q") == nlohmann::json::array({0, tokens}));
  REQUIRE(meta.at("softmax_scale").get<float>() == 0.0625f);
  REQUIRE(meta.at("causal").get<bool>());

  const size_t q_bytes = size_t(tokens) * heads * dim * sizeof(uint16_t);
  const size_t page_bytes = size_t(page) * kv_heads * dim;
  const auto q_host = read("q.bin", q_bytes);
  const auto k_host = read("k.bin", size_t(pages) * page_bytes);
  const auto v_host = read("v.bin", size_t(pages) * page_bytes);
  const auto expected_host = read("output.bin", q_bytes);
  std::vector<unsigned char> cache_host(size_t(pages) * 2 * page_bytes);
  for (int p = 0; p < pages; ++p) {
    std::memcpy(cache_host.data() + size_t(2 * p) * page_bytes,
                k_host.data() + size_t(p) * page_bytes, page_bytes);
    std::memcpy(cache_host.data() + size_t(2 * p + 1) * page_bytes,
                v_host.data() + size_t(p) * page_bytes, page_bytes);
  }

  Queue gpu(vt::DeviceType::kXPU);
  Buffer query(gpu.q, DType::kF16, {tokens, heads, dim});
  Buffer output(gpu.q, DType::kF16, {tokens, heads, dim});
  Buffer cache(gpu.q, DType::kI8, {pages, 2 * page, kv_heads, dim});
  Buffer table(gpu.q, DType::kI32, {1, pages});
  Buffer lengths(gpu.q, DType::kI32, {1});
  Buffer offsets(gpu.q, DType::kI32, {2});
  query.upload(q_host.data());
  cache.upload(cache_host.data());
  std::vector<int32_t> ids(pages);
  std::iota(ids.begin(), ids.end(), 0);
  table.upload(ids.data());
  const int32_t seq_len[] = {used}, q_offsets[] = {0, tokens};
  lengths.upload(seq_len);
  offsets.upload(q_offsets);
  auto kc = vt::Tensor::Contiguous(cache.tensor.data, DType::kI8, gpu.q.device,
                                    {pages, page, kv_heads, dim});
  kc.stride[0] *= 2;
  auto vc = kc;
  vc.data = static_cast<char*>(kc.data) + page_bytes;
  vt::PagedAttentionArgs args;
  args.kv_cache_dtype = vt::Fp8KVCacheDataType::kFp8E4M3;
  args.k_scale = meta.at("k_scale").get<float>();
  args.v_scale = meta.at("v_scale").get<float>();
  args.scale = meta.at("softmax_scale").get<float>();
  args.max_seq_len = used;
  args.causal = true;
  if (std::getenv("VT_XPU_PROFILE")) (void)vt::xpu::DrainProfileEvents();
  setenv("VT_XPU_ATTENTION", "split", 1);
  vt::PagedAttention(gpu.q, output.tensor, query.tensor, kc, vc,
                     table.tensor, lengths.tensor, offsets.tensor, args);
  vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
  if (std::getenv("VT_XPU_PROFILE")) {
    int split = 0;
    for (const auto& event : vt::xpu::DrainProfileEvents())
      split += event.stage == "attention_split_partial";
    CHECK(split == 1);
  }
  std::vector<float> expected(q_bytes / sizeof(uint16_t));
  for (size_t i = 0; i < expected.size(); ++i) {
    uint16_t bits;
    std::memcpy(&bits, expected_host.data() + i * sizeof(bits), sizeof(bits));
    expected[i] = vt::F16ToF32(bits);
  }
  const auto actual = output.floats();
  for (int row = 0; row < tokens; ++row) {
    double delta2 = 0, norm2 = 0, maximum = 0;
    for (int i = 0; i < heads * dim; ++i) {
      const size_t index = size_t(row) * heads * dim + i;
      const double delta = double(actual[index]) - expected[index];
      delta2 += delta * delta;
      norm2 += double(expected[index]) * expected[index];
      maximum = std::max(maximum, std::abs(delta));
    }
    std::cout << "M04_REPLAY row=" << row
              << " relative_rms=" << std::sqrt(delta2 / norm2)
              << " max_abs=" << maximum << std::endl;
  }
  Accuracy(actual, expected, false, true);
  if (std::getenv("VT_B70_M04_BENCH")) {
    auto timed_run = [&] {
      const auto start = std::chrono::steady_clock::now();
      vt::PagedAttention(gpu.q, output.tensor, query.tensor, kc, vc,
                         table.tensor, lengths.tensor, offsets.tensor, args);
      vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
      return std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - start).count();
    };
    for (int i = 0; i < 5; ++i) timed_run();
    std::vector<double> samples;
    for (int i = 0; i < 20; ++i) samples.push_back(timed_run());
    std::sort(samples.begin(), samples.end());
    std::cout << "M04_SPLITK_WALL_MEDIAN_MS " <<
        (samples[9] + samples[10]) / 2 << " SAMPLES_MS "
              << nlohmann::json(samples).dump() << std::endl;
    if (std::getenv("VT_XPU_PROFILE")) {
      std::vector<double> partial_ms, reduce_ms;
      for (const auto& event : vt::xpu::DrainProfileEvents()) {
        const double ms = double(event.end_ns - event.start_ns) / 1e6;
        if (event.stage == "attention_split_partial") partial_ms.push_back(ms);
        if (event.stage == "attention_split_reduce_cooperative") reduce_ms.push_back(ms);
      }
      REQUIRE(partial_ms.size() == 25);
      REQUIRE(reduce_ms.size() == 25);
      partial_ms.erase(partial_ms.begin(), partial_ms.begin() + 5);
      reduce_ms.erase(reduce_ms.begin(), reduce_ms.begin() + 5);
      std::sort(partial_ms.begin(), partial_ms.end());
      std::sort(reduce_ms.begin(), reduce_ms.end());
      std::cout << "M04_SPLITK_GPU_PARTIAL_MEDIAN_MS " <<
          (partial_ms[9] + partial_ms[10]) / 2
                << " REDUCE_MEDIAN_MS " << (reduce_ms[9] + reduce_ms[10]) / 2
                << std::endl;
    }
  }
  if (std::getenv("VT_B70_M04_VERIFY")) {
    setenv("VT_XPU_ATTENTION", "verify", 1);
    if (std::getenv("VT_XPU_PROFILE")) (void)vt::xpu::DrainProfileEvents();
    auto verify_run = [&](const vt::Tensor& verify_key, const vt::Tensor& verify_value) {
      const auto start = std::chrono::steady_clock::now();
      vt::PagedAttention(gpu.q, output.tensor, query.tensor, verify_key, verify_value,
                         table.tensor, lengths.tensor, offsets.tensor, args);
      vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
      return std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - start).count();
    };
    verify_run(kc, vc);
    const auto verify_output = output.floats();
    int mismatches = 0, nonfinite = 0;
    for (size_t i = 0; i < expected.size(); ++i)
      mismatches += verify_output[i] != expected[i],
      nonfinite += !std::isfinite(verify_output[i]);
    std::cout << "M04_NATIVE_VERIFY_F16_MISMATCHES " << mismatches
              << " NONFINITE " << nonfinite << " FIRST ";
    for (int i = 0; i < 8; ++i) std::cout << verify_output[i] << " ";
    std::cout << std::endl;
    Accuracy(verify_output, expected, false, true);
    if (std::getenv("VT_XPU_PROFILE")) {
      int packs = 0, unpacks = 0;
      for (const auto& event : vt::xpu::DrainProfileEvents()) {
        packs += event.stage == "attention_verify_pack";
        unpacks += event.stage == "attention_verify_unpack";
      }
      CHECK(packs == 1);
      CHECK(unpacks == 1);
    }
    for (int i = 0; i < 5; ++i) verify_run(kc, vc);
    std::vector<double> samples;
    for (int i = 0; i < 20; ++i) samples.push_back(verify_run(kc, vc));
    std::sort(samples.begin(), samples.end());
    std::cout << "M04_NATIVE_VERIFY_WALL_MEDIAN_MS " <<
        (samples[9] + samples[10]) / 2 << " SAMPLES_MS "
              << nlohmann::json(samples).dump() << std::endl;
  }
  if (const char* cpp_directory = std::getenv("VT_B70_M04_CPP_REPLAY_DIR")) {
    auto read_cpp = [&](const std::string& name, size_t bytes) {
      std::ifstream file(std::string(cpp_directory) + "/" + name, std::ios::binary);
      REQUIRE(file.good());
      std::vector<unsigned char> data(bytes);
      file.read(reinterpret_cast<char*>(data.data()), data.size());
      REQUIRE(file.gcount() == static_cast<std::streamsize>(bytes));
      REQUIRE(file.peek() == std::char_traits<char>::eof());
      return data;
    };
    const auto cpp_q = read_cpp("q.bin", q_bytes);
    const auto cpp_k = read_cpp("k.bin", size_t(pages) * page_bytes);
    const auto cpp_v = read_cpp("v.bin", size_t(pages) * page_bytes);
    const auto cpp_out = read_cpp("output.bin", q_bytes);
    std::vector<unsigned char> cpp_cache(cache_host.size());
    for (int p = 0; p < pages; ++p) {
      std::memcpy(cpp_cache.data() + size_t(2 * p) * page_bytes,
                  cpp_k.data() + size_t(p) * page_bytes, page_bytes);
      std::memcpy(cpp_cache.data() + size_t(2 * p + 1) * page_bytes,
                  cpp_v.data() + size_t(p) * page_bytes, page_bytes);
    }
    auto rerun = [&](const char* label, const std::vector<unsigned char>& q_bytes_in,
                     const std::vector<unsigned char>& cache_bytes_in) {
      query.upload(q_bytes_in.data());
      cache.upload(cache_bytes_in.data());
      vt::PagedAttention(gpu.q, output.tensor, query.tensor, kc, vc,
                         table.tensor, lengths.tensor, offsets.tensor, args);
      vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
      const auto values = output.floats();
      double delta2 = 0, norm2 = 0;
      for (size_t i = 0; i < values.size(); ++i) {
        const double delta = double(values[i]) - expected[i];
        delta2 += delta * delta;
        norm2 += double(expected[i]) * expected[i];
      }
      std::cout << "M04_CROSS_REPLAY " << label
                << " relative_rms_vs_python=" << std::sqrt(delta2 / norm2)
                << std::endl;
      return values;
    };
    rerun("cpp_q_python_kv", cpp_q, cache_host);
    auto cpp_k_python_v = cache_host;
    auto python_k_cpp_v = cache_host;
    for (int p = 0; p < pages; ++p) {
      std::memcpy(cpp_k_python_v.data() + size_t(2 * p) * page_bytes,
                  cpp_cache.data() + size_t(2 * p) * page_bytes, page_bytes);
      std::memcpy(python_k_cpp_v.data() + size_t(2 * p + 1) * page_bytes,
                  cpp_cache.data() + size_t(2 * p + 1) * page_bytes, page_bytes);
    }
    rerun("python_q_cpp_k_python_v", q_host, cpp_k_python_v);
    rerun("python_q_python_k_cpp_v", q_host, python_k_cpp_v);
    rerun("python_q_cpp_kv", q_host, cpp_cache);
    const auto self = rerun("cpp_q_cpp_kv", cpp_q, cpp_cache);
    std::vector<float> cpp_expected(q_bytes / sizeof(uint16_t));
    for (size_t i = 0; i < cpp_expected.size(); ++i) {
      uint16_t bits;
      std::memcpy(&bits, cpp_out.data() + i * sizeof(bits), sizeof(bits));
      cpp_expected[i] = vt::F16ToF32(bits);
    }
    Accuracy(self, cpp_expected, false, true);
  }
  if (const char* prequant_directory = std::getenv("VT_B70_M04_PREQUANT_DIR")) {
    auto read_prequant = [&](const char* name) {
      std::ifstream file(std::string(prequant_directory) + "/" + name + ".bin",
                         std::ios::binary);
      REQUIRE(file.good());
      std::vector<unsigned char> data(size_t(4096) * kv_heads * dim * sizeof(uint16_t));
      file.read(reinterpret_cast<char*>(data.data()), data.size());
      REQUIRE(file.gcount() == static_cast<std::streamsize>(data.size()));
      REQUIRE(file.peek() == std::char_traits<char>::eof());
      return data;
    };
    Buffer source_k(gpu.q, DType::kF16, {4096, kv_heads, dim});
    Buffer source_v(gpu.q, DType::kF16, {4096, kv_heads, dim});
    Buffer fresh_cache(gpu.q, DType::kI8, {pages, 2 * page, kv_heads, dim});
    Buffer slots(gpu.q, DType::kI64, {4096});
    const auto prequant_k = read_prequant("k");
    const auto prequant_v = read_prequant("v");
    source_k.upload(prequant_k.data());
    source_v.upload(prequant_v.data());
    std::vector<int64_t> slot_ids(4096);
    std::iota(slot_ids.begin(), slot_ids.end(), 0);
    slots.upload(slot_ids.data());
    auto fresh_k = vt::Tensor::Contiguous(fresh_cache.tensor.data, DType::kI8,
                                          gpu.q.device, {pages, page, kv_heads, dim});
    fresh_k.stride[0] *= 2;
    auto fresh_v = fresh_k;
    fresh_v.data = static_cast<char*>(fresh_k.data) + page_bytes;
    vt::ReshapeAndCacheFp8(gpu.q, source_k.tensor, source_v.tensor, fresh_k,
                           fresh_v, slots.tensor, vt::Fp8KVCacheDataType::kFp8E4M3,
                           1.0f, 1.0f);
    vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
    const auto written = fresh_cache.download();
    int64_t k_mismatches = 0, v_mismatches = 0;
    for (int token = 0; token < 4096; ++token) {
      const size_t row = size_t(token) * kv_heads * dim;
      const size_t block = size_t(token / page) * 2 * page_bytes +
                           size_t(token % page) * kv_heads * dim;
      for (size_t i = 0; i < size_t(kv_heads * dim); ++i) {
        k_mismatches += written[block + i] != k_host[row + i];
        v_mismatches += written[block + page_bytes + i] != v_host[row + i];
      }
    }
    std::cout << "M04_PREQUANT_REPLAY k_mismatches=" << k_mismatches
              << " v_mismatches=" << v_mismatches << std::endl;
    CHECK(k_mismatches == 0);
    CHECK(v_mismatches == 0);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}
