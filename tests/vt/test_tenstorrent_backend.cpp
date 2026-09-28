// Tenstorrent backend skeleton unit gates (BACKEND-TENSTORRENT, W0). Newly
// authored — vLLM has no Tenstorrent tests to port. Mirrors the shape of
// tests/vt/test_vulkan_backend.cpp / test_metal_backend.cpp so the three are
// read side by side.
//
// This TU is COMPILED ONLY in a Tenstorrent build (tests/CMakeLists.txt gates
// it on VLLM_CPP_TENSTORRENT) and every assertion goes through the public
// vt:: seam — if the skeleton needed ttnn headers in a test to be checkable,
// the seam would be leaking. (This is also why this file needs none of the
// object-library include isolation tenstorrent_ops.cpp needed — it never
// touches ttnn/tt-metal headers at all.)
//
// Every case is SKIPPED, not failed, when no Blackhole card is present — the
// registrars stay silent by design (tenstorrent_backend.cpp/tenstorrent.cpp),
// and a Tenstorrent-enabled build legitimately runs in CI containers with no
// card. The skip is REPORTED so a silently-unregistered backend on a box that
// DOES have one cannot masquerade as a pass.
#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/quant.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::Queue;
using vt::Tensor;

namespace vt::tenstorrent {
// Tenstorrent residency probe (tenstorrent_internal.h); declared here to keep
// the test TU free of internal includes.
bool DeviceShadowExact(const Tensor& t, uint32_t rows, uint32_t cols);
// ISSUE-LOCAL-01M3JXEFQKSZP23PP2HWY9G0VQ surfaces (tenstorrent_residency.cpp).
void CommitRank3DeviceLogicalForTest(Tensor& out, uint32_t b, uint32_t h, uint32_t d);
void EnsureDevice2DForTest(Tensor& t);
bool TraceCaptureActive();
}  // namespace vt::tenstorrent

namespace {

bool TenstorrentPresent() { return vt::TryGetBackend(DeviceType::kTENSTORRENT) != nullptr; }

}  // namespace

TEST_CASE("kTENSTORRENT backend registers iff a device is present") {
  Backend* b = vt::TryGetBackend(DeviceType::kTENSTORRENT);
  if (b == nullptr) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  CHECK_FALSE(b->UnifiedMemory());  // discrete PCIe card — see backend.h's SCOPE note
}

TEST_CASE("kTENSTORRENT Platform mirrors the registered Backend") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vllm::platforms::HasPlatform(DeviceType::kTENSTORRENT));
  vllm::platforms::Platform& p = vllm::platforms::GetPlatform(DeviceType::kTENSTORRENT);
  CHECK(p.device_type() == DeviceType::kTENSTORRENT);
  CHECK(&p.backend() == vt::TryGetBackend(DeviceType::kTENSTORRENT));
  // OPT-125m path: BF16 weights/activations + F32 logits — see tenstorrent_ops.cpp.
  CHECK(p.supported_dtypes() ==
        std::vector<vt::DType>{vt::DType::kBF16, vt::DType::kF32});
  // FLASH_ATTN is registered against the NHD layout our kPagedAttention uses.
  CHECK(p.get_attn_backend_priority({}) == std::vector<std::string>{"FLASH_ATTN"});
  CHECK(p.supports_model_architecture("OPTForCausalLM"));
  // Qwen3-dense after RmsNorm / SiluAndMul / Cast / RoPE landed (Metal M3b twin).
  CHECK(p.supports_model_architecture("Qwen3ForCausalLM"));
  // Mistral-7B-v0.3 reuses the Qwen3-dense forward verbatim (qk-norm skipped,
  // plain rope, untied lm_head) — every op already registered, no new kernel.
  CHECK(p.supports_model_architecture("MistralForCausalLM"));
  CHECK_FALSE(p.supports_model_architecture("LlamaForCausalLM"));
  // Capture decline (tenstorrent.cpp's support_static_graph_mode): the
  // conjunction host-free decode AND an explicit VT_TT_DECODE_CAPTURE
  // opt-in — capture hangs deterministically on multi-request decode
  // (#1625), so it is declined by default. This case pins the FULL truth
  // table of that conjunction and is ambient-immune: every cell sets BOTH
  // envs before evaluating, so an ambient VT_TT_HOST_FREE_DECODE=0 (the
  // documented pre-flip opt-out) can no longer make the opt-in cell
  // vacuous. The host-free-OFF cells matter for exactly that reason
  // (#1688's lesson: the flag is read live on every call) — they are what
  // catches a dropped HostFreeDecodeEnabled conjunct, and the capture=0
  // cell catches a dropped VT_TT_DECODE_CAPTURE opt-out parse. The env is
  // read live per call (HostFreeDecodeEnabled's no-caching contract,
  // tenstorrent_device.h), so re-resolving after each setenv proves that.
  const char* const prev_host_free = std::getenv("VT_TT_HOST_FREE_DECODE");
  const bool had_host_free = prev_host_free != nullptr;
  const std::string saved_host_free = had_host_free ? std::string(prev_host_free) : std::string();
  const char* const prev_capture = std::getenv("VT_TT_DECODE_CAPTURE");
  const bool had_capture = prev_capture != nullptr;
  const std::string saved_capture = had_capture ? std::string(prev_capture) : std::string();
  const auto static_graph_mode = [] {
    return vllm::platforms::GetPlatform(DeviceType::kTENSTORRENT).support_static_graph_mode();
  };
  // Cell 1: host-free OFF, capture unset → declined (the host-free conjunct).
  ::setenv("VT_TT_HOST_FREE_DECODE", "0", 1);
  ::unsetenv("VT_TT_DECODE_CAPTURE");
  CHECK_FALSE(static_graph_mode());
  // Cell 2: host-free ON, capture unset → ACCEPTED. The #1625 flip: capture
  // is the DEFAULT, so the bare default cell asserts the flip itself.
  ::setenv("VT_TT_HOST_FREE_DECODE", "1", 1);
  CHECK(static_graph_mode());
  // Cell 3: host-free ON, capture explicitly "1" → accepted.
  ::setenv("VT_TT_DECODE_CAPTURE", "1", 1);
  CHECK(static_graph_mode());
  // Cell 4: host-free ON, capture "0" → declined (the opt-out parse).
  ::setenv("VT_TT_DECODE_CAPTURE", "0", 1);
  CHECK_FALSE(static_graph_mode());
  // Cell 5: host-free OFF, capture "1" → declined (the host-free conjunct).
  ::setenv("VT_TT_HOST_FREE_DECODE", "0", 1);
  ::setenv("VT_TT_DECODE_CAPTURE", "1", 1);
  CHECK_FALSE(static_graph_mode());
  // Restore the ORIGINAL ambient state of both envs (unset if it was unset).
  if (had_host_free) {
    ::setenv("VT_TT_HOST_FREE_DECODE", saved_host_free.c_str(), 1);
  } else {
    ::unsetenv("VT_TT_HOST_FREE_DECODE");
  }
  if (had_capture) {
    ::setenv("VT_TT_DECODE_CAPTURE", saved_capture.c_str(), 1);
  } else {
    ::unsetenv("VT_TT_DECODE_CAPTURE");
  }
}

TEST_CASE("kTENSTORRENT kMatmul matches a host F32 reference") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmul, DeviceType::kTENSTORRENT));

  constexpr int64_t M = 32, K = 32, N = 32;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_a(M * K), host_b(K * N), host_out(M * N, 0.0f);
  for (size_t i = 0; i < host_a.size(); ++i) host_a[i] = static_cast<float>(i % 7) * 0.1f;
  for (size_t i = 0; i < host_b.size(); ++i) host_b[i] = static_cast<float>(i % 5) * 0.2f;

  void* mem_a = backend.Alloc(host_a.size() * sizeof(float));
  void* mem_b = backend.Alloc(host_b.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_a, host_a.data(), host_a.size() * sizeof(float));
  backend.Copy(q, mem_b, host_b.data(), host_b.size() * sizeof(float));

  Tensor a = Tensor::Contiguous(mem_a, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, K});
  Tensor b = Tensor::Contiguous(mem_b, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {K, N});
  Tensor out =
      Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, N});

  auto matmul = reinterpret_cast<vt::MatmulFn>(vt::GetOp(vt::OpId::kMatmul, DeviceType::kTENSTORRENT));
  matmul(q, out, a, b);

  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_a);
  backend.Free(mem_b);
  backend.Free(mem_out);

  std::vector<float> ref(M * N, 0.0f);
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) acc += host_a[i * K + k] * host_b[k * N + j];
      ref[i * N + j] = acc;
    }
  }

  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i) max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[i] - ref[i]));
  // bf16 accumulation over K=32 on-device — same tolerance the hands-on spike
  // measured (.agents/specs/tenstorrent-backend.md), not a rubber stamp.
  CHECK(max_abs_diff < 0.5f);
}

TEST_CASE("kTENSTORRENT kMatmulBT matches a host F32 reference (a @ b^T)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));

  // `a` is [M,K] activations; `b` is [N,K] nn.Linear weight (torch layout).
  constexpr int64_t M = 32, K = 32, N = 32;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_a(M * K), host_b(N * K), host_out(M * N, 0.0f);
  for (size_t i = 0; i < host_a.size(); ++i) host_a[i] = static_cast<float>(i % 7) * 0.1f;
  for (size_t i = 0; i < host_b.size(); ++i) host_b[i] = static_cast<float>(i % 5) * 0.2f;

  void* mem_a = backend.Alloc(host_a.size() * sizeof(float));
  void* mem_b = backend.Alloc(host_b.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_a, host_a.data(), host_a.size() * sizeof(float));
  backend.Copy(q, mem_b, host_b.data(), host_b.size() * sizeof(float));

  Tensor a = Tensor::Contiguous(mem_a, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, K});
  Tensor b = Tensor::Contiguous(mem_b, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {N, K});
  Tensor out =
      Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, N});

  auto matmul_bt =
      reinterpret_cast<vt::MatmulFn>(vt::GetOp(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));
  matmul_bt(q, out, a, b);

  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_a);
  backend.Free(mem_b);
  backend.Free(mem_out);

  std::vector<float> ref(M * N, 0.0f);
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) acc += host_a[i * K + k] * host_b[j * K + k];
      ref[i * N + j] = acc;
    }
  }

  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i) max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[i] - ref[i]));
  CHECK(max_abs_diff < 0.5f);
}

TEST_CASE("kTENSTORRENT kAdd matches a host F32 reference (elementwise + bias broadcast)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kAdd, DeviceType::kTENSTORRENT));

  constexpr int64_t Rows = 32, D = 32;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  auto add = reinterpret_cast<vt::AddFn>(vt::GetOp(vt::OpId::kAdd, DeviceType::kTENSTORRENT));

  SUBCASE("elementwise, same rank") {
    std::vector<float> host_a(Rows * D), host_b(Rows * D), host_out(Rows * D, 0.0f);
    for (size_t i = 0; i < host_a.size(); ++i) host_a[i] = static_cast<float>(i % 7) * 0.1f;
    for (size_t i = 0; i < host_b.size(); ++i) host_b[i] = static_cast<float>(i % 5) * 0.2f;

    void* mem_a = backend.Alloc(host_a.size() * sizeof(float));
    void* mem_b = backend.Alloc(host_b.size() * sizeof(float));
    void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
    Queue q = backend.CreateQueue();
    backend.Copy(q, mem_a, host_a.data(), host_a.size() * sizeof(float));
    backend.Copy(q, mem_b, host_b.data(), host_b.size() * sizeof(float));

    Tensor a =
        Tensor::Contiguous(mem_a, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
    Tensor b =
        Tensor::Contiguous(mem_b, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
    Tensor out =
        Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
    add(q, out, a, b);
    backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
    backend.Free(mem_a);
    backend.Free(mem_b);
    backend.Free(mem_out);

    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < host_out.size(); ++i)
      max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[i] - (host_a[i] + host_b[i])));
    CHECK(max_abs_diff < 0.1f);
  }

  SUBCASE("rank-1 bias broadcast over the last dim") {
    std::vector<float> host_a(Rows * D), host_bias(D), host_out(Rows * D, 0.0f);
    for (size_t i = 0; i < host_a.size(); ++i) host_a[i] = static_cast<float>(i % 7) * 0.1f;
    for (size_t i = 0; i < host_bias.size(); ++i) host_bias[i] = static_cast<float>(i) * 0.05f;

    void* mem_a = backend.Alloc(host_a.size() * sizeof(float));
    void* mem_bias = backend.Alloc(host_bias.size() * sizeof(float));
    void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
    Queue q = backend.CreateQueue();
    backend.Copy(q, mem_a, host_a.data(), host_a.size() * sizeof(float));
    backend.Copy(q, mem_bias, host_bias.data(), host_bias.size() * sizeof(float));

    Tensor a =
        Tensor::Contiguous(mem_a, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
    Tensor bias =
        Tensor::Contiguous(mem_bias, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {D});
    Tensor out =
        Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
    add(q, out, a, bias);
    backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
    backend.Free(mem_a);
    backend.Free(mem_bias);
    backend.Free(mem_out);

    float max_abs_diff = 0.0f;
    for (int64_t r = 0; r < Rows; ++r)
      for (int64_t c = 0; c < D; ++c)
        max_abs_diff = std::max(max_abs_diff,
                                 std::fabs(host_out[r * D + c] - (host_a[r * D + c] + host_bias[c])));
    CHECK(max_abs_diff < 0.1f);
  }
}

TEST_CASE("kTENSTORRENT kRelu matches a host F32 reference") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kRelu, DeviceType::kTENSTORRENT));

  constexpr int64_t Rows = 32, D = 32;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_x(Rows * D), host_out(Rows * D, 0.0f);
  for (size_t i = 0; i < host_x.size(); ++i)
    host_x[i] = (static_cast<float>(i % 11) - 5.0f) * 0.3f;  // mix of signs

  void* mem_x = backend.Alloc(host_x.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_x, host_x.data(), host_x.size() * sizeof(float));

  Tensor x = Tensor::Contiguous(mem_x, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
  Tensor out =
      Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});

  auto relu = reinterpret_cast<vt::ReluFn>(vt::GetOp(vt::OpId::kRelu, DeviceType::kTENSTORRENT));
  relu(q, out, x);

  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_x);
  backend.Free(mem_out);

  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < host_x.size(); ++i)
    max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[i] - std::max(0.0f, host_x[i])));
  CHECK(max_abs_diff < 0.1f);
}

TEST_CASE("kTENSTORRENT kEmbedding matches a host F32 reference (row gather)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kEmbedding, DeviceType::kTENSTORRENT));

  // Non-tile-aligned (t, h) on purpose: forces the ROW_MAJOR path and
  // proves download is dense without TILE padding. Vocab is modest so the
  // host oracle stays trivial.
  constexpr int64_t Vocab = 17, H = 24, T = 7;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_table(Vocab * H);
  for (size_t i = 0; i < host_table.size(); ++i)
    host_table[i] = static_cast<float>(i % 13) * 0.1f - 0.5f;
  // i32 ids covering edges: first, last, and middle rows; repeats allowed.
  std::vector<int32_t> host_ids = {0, 3, 16, 1, 3, 8, 16};
  REQUIRE(static_cast<int64_t>(host_ids.size()) == T);

  std::vector<float> host_out(T * H, 0.0f);
  void* mem_table = backend.Alloc(host_table.size() * sizeof(float));
  void* mem_ids = backend.Alloc(host_ids.size() * sizeof(int32_t));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_table, host_table.data(), host_table.size() * sizeof(float));
  backend.Copy(q, mem_ids, host_ids.data(), host_ids.size() * sizeof(int32_t));

  Tensor table = Tensor::Contiguous(mem_table, vt::DType::kF32,
                                    Device{DeviceType::kTENSTORRENT, 0}, {Vocab, H});
  Tensor ids = Tensor::Contiguous(mem_ids, vt::DType::kI32,
                                  Device{DeviceType::kTENSTORRENT, 0}, {T});
  Tensor out = Tensor::Contiguous(mem_out, vt::DType::kF32,
                                  Device{DeviceType::kTENSTORRENT, 0}, {T, H});

  auto embedding =
      reinterpret_cast<vt::EmbeddingFn>(vt::GetOp(vt::OpId::kEmbedding, DeviceType::kTENSTORRENT));
  embedding(q, out, table, ids);

  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_table);
  backend.Free(mem_ids);
  backend.Free(mem_out);

  // Host oracle: pure row gather. bf16 table storage means a modest abs tol.
  float max_abs_diff = 0.0f;
  for (int64_t i = 0; i < T; ++i) {
    const int32_t id = host_ids[static_cast<size_t>(i)];
    for (int64_t j = 0; j < H; ++j) {
      const float ref = host_table[static_cast<size_t>(id) * H + j];
      max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[i * H + j] - ref));
    }
  }
  CHECK(max_abs_diff < 0.1f);
}

TEST_CASE("kTENSTORRENT kLayerNorm matches a host F32 reference (affine + plain)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kLayerNorm, DeviceType::kTENSTORRENT));

  // Tile-aligned so the TILE upload path is exercised cleanly (same as the
  // linear ops). Host oracle is the ATen/cpu_layernorm biased-variance form.
  constexpr int64_t Rows = 32, D = 32;
  constexpr float Eps = 1e-5f;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  auto layer_norm = reinterpret_cast<vt::LayerNormFn>(
      vt::GetOp(vt::OpId::kLayerNorm, DeviceType::kTENSTORRENT));

  auto run_case = [&](bool with_affine) {
    std::vector<float> host_x(Rows * D), host_w(D), host_b(D), host_out(Rows * D, 0.0f);
    for (size_t i = 0; i < host_x.size(); ++i)
      host_x[i] = (static_cast<float>(i % 17) - 8.0f) * 0.15f;
    for (int64_t j = 0; j < D; ++j) {
      host_w[static_cast<size_t>(j)] = 0.5f + static_cast<float>(j % 5) * 0.1f;
      host_b[static_cast<size_t>(j)] = static_cast<float>(j % 7) * 0.05f - 0.15f;
    }

    void* mem_x = backend.Alloc(host_x.size() * sizeof(float));
    void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
    void* mem_w = with_affine ? backend.Alloc(host_w.size() * sizeof(float)) : nullptr;
    void* mem_b = with_affine ? backend.Alloc(host_b.size() * sizeof(float)) : nullptr;
    Queue q = backend.CreateQueue();
    backend.Copy(q, mem_x, host_x.data(), host_x.size() * sizeof(float));
    if (with_affine) {
      backend.Copy(q, mem_w, host_w.data(), host_w.size() * sizeof(float));
      backend.Copy(q, mem_b, host_b.data(), host_b.size() * sizeof(float));
    }

    Tensor x =
        Tensor::Contiguous(mem_x, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
    Tensor out =
        Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
    Tensor w_t, b_t;
    const Tensor* w_ptr = nullptr;
    const Tensor* b_ptr = nullptr;
    if (with_affine) {
      w_t = Tensor::Contiguous(mem_w, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {D});
      b_t = Tensor::Contiguous(mem_b, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {D});
      w_ptr = &w_t;
      b_ptr = &b_t;
    }

    vt::LayerNormArgs args;
    args.eps = Eps;
    layer_norm(q, out, x, w_ptr, b_ptr, args);

    backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
    backend.Free(mem_x);
    backend.Free(mem_out);
    if (with_affine) {
      backend.Free(mem_w);
      backend.Free(mem_b);
    }

    float max_abs_diff = 0.0f;
    for (int64_t r = 0; r < Rows; ++r) {
      float sum = 0.0f;
      for (int64_t j = 0; j < D; ++j) sum += host_x[r * D + j];
      const float mean = sum / static_cast<float>(D);
      float sq = 0.0f;
      for (int64_t j = 0; j < D; ++j) {
        const float dv = host_x[r * D + j] - mean;
        sq += dv * dv;
      }
      const float rstd = 1.0f / std::sqrt(sq / static_cast<float>(D) + Eps);
      for (int64_t j = 0; j < D; ++j) {
        float ref = (host_x[r * D + j] - mean) * rstd;
        if (with_affine) ref = ref * host_w[static_cast<size_t>(j)] + host_b[static_cast<size_t>(j)];
        max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[r * D + j] - ref));
      }
    }
    // bf16 storage + device reduction; not bit-exact vs f32 host mean/var.
    CHECK(max_abs_diff < 0.5f);
  };

  SUBCASE("elementwise_affine=True (weight + bias)") { run_case(true); }
  SUBCASE("elementwise_affine=False (no weight/bias)") { run_case(false); }
}

// First Qwen3-dense (`Qwen3ForCausalLM`) op beyond OPT's set. Host oracle is
// cpu_ops RmsNormKernel (no residual, gemma=false) — the Qwen3 default.
TEST_CASE("kTENSTORRENT kRmsNorm matches a host F32 reference (weight, no residual)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kRmsNorm, DeviceType::kTENSTORRENT));

  constexpr int64_t Rows = 32, D = 32;
  constexpr float Eps = 1e-6f;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  auto rms_norm = reinterpret_cast<vt::RmsNormFn>(
      vt::GetOp(vt::OpId::kRmsNorm, DeviceType::kTENSTORRENT));

  std::vector<float> host_x(Rows * D), host_w(D), host_out(Rows * D, 0.0f);
  for (size_t i = 0; i < host_x.size(); ++i)
    host_x[i] = (static_cast<float>(i % 17) - 8.0f) * 0.15f;
  for (int64_t j = 0; j < D; ++j)
    host_w[static_cast<size_t>(j)] = 0.5f + static_cast<float>(j % 5) * 0.1f;

  void* mem_x = backend.Alloc(host_x.size() * sizeof(float));
  void* mem_w = backend.Alloc(host_w.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_x, host_x.data(), host_x.size() * sizeof(float));
  backend.Copy(q, mem_w, host_w.data(), host_w.size() * sizeof(float));

  Tensor x =
      Tensor::Contiguous(mem_x, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
  Tensor w =
      Tensor::Contiguous(mem_w, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {D});
  Tensor out =
      Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});

  vt::RmsNormArgs args;
  args.eps = Eps;
  args.gemma = false;
  rms_norm(q, out, x, w, args, /*residual=*/nullptr);

  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_x);
  backend.Free(mem_w);
  backend.Free(mem_out);

  float max_abs_diff = 0.0f;
  for (int64_t r = 0; r < Rows; ++r) {
    float sumsq = 0.0f;
    for (int64_t j = 0; j < D; ++j) {
      const float v = host_x[r * D + j];
      sumsq += v * v;
    }
    const float inv = 1.0f / std::sqrt(sumsq / static_cast<float>(D) + Eps);
    for (int64_t j = 0; j < D; ++j) {
      const float ref = host_x[r * D + j] * inv * host_w[static_cast<size_t>(j)];
      max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[r * D + j] - ref));
    }
  }
  // bf16 storage + device reduction; same envelope as kLayerNorm.
  CHECK(max_abs_diff < 0.5f);
}

// Gemma style (w+1) — the Qwen3.5 norm (every RmsNorm site passes gemma=true).
// The device arm must bake +1 into the gamma it hands ttnn::rms_norm; dropping
// it collapsed ambient host-free prefill to `,` (BACKEND-TENSTORRENT-QWEN35
// W2b). Reference computes in f32 with the SAME wj = w+1 order as the kernel's
// host arm.
TEST_CASE("kTENSTORRENT kRmsNorm gemma matches a host F32 reference (w+1)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kRmsNorm, DeviceType::kTENSTORRENT));

  constexpr int64_t Rows = 32, D = 32;
  constexpr float Eps = 1e-6f;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  auto rms_norm = reinterpret_cast<vt::RmsNormFn>(
      vt::GetOp(vt::OpId::kRmsNorm, DeviceType::kTENSTORRENT));

  std::vector<float> host_x(Rows * D), host_w(D), host_out(Rows * D, 0.0f);
  for (size_t i = 0; i < host_x.size(); ++i)
    host_x[i] = (static_cast<float>(i % 17) - 8.0f) * 0.15f;
  for (int64_t j = 0; j < D; ++j)
    // Small weights make the +1 dominant: a dropped bake fails by ~1.0 per
    // element, far outside any storage envelope.
    host_w[static_cast<size_t>(j)] = 0.05f * static_cast<float>(j % 3);

  void* mem_x = backend.Alloc(host_x.size() * sizeof(float));
  void* mem_w = backend.Alloc(host_w.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_x, host_x.data(), host_x.size() * sizeof(float));
  backend.Copy(q, mem_w, host_w.data(), host_w.size() * sizeof(float));

  Tensor x =
      Tensor::Contiguous(mem_x, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
  Tensor w =
      Tensor::Contiguous(mem_w, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {D});
  Tensor out =
      Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});

  vt::RmsNormArgs args;
  args.eps = Eps;
  args.gemma = true;
  rms_norm(q, out, x, w, args, /*residual=*/nullptr);

  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_x);
  backend.Free(mem_w);
  backend.Free(mem_out);

  float max_abs_diff = 0.0f;
  for (int64_t r = 0; r < Rows; ++r) {
    float sumsq = 0.0f;
    for (int64_t j = 0; j < D; ++j) {
      const float v = host_x[r * D + j];
      sumsq += v * v;
    }
    const float inv = 1.0f / std::sqrt(sumsq / static_cast<float>(D) + Eps);
    for (int64_t j = 0; j < D; ++j) {
      const float ref =
          host_x[r * D + j] * inv * (host_w[static_cast<size_t>(j)] + 1.0f);
      max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[r * D + j] - ref));
    }
  }
  CHECK(max_abs_diff < 0.5f);
}

// Qwen3-dense MLP SwiGLU half. Device path (slice + silu + mul) via BF16 tiles.
TEST_CASE("kTENSTORRENT kSiluAndMul matches host F32 within BF16 envelope") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kSiluAndMul, DeviceType::kTENSTORRENT));

  constexpr int64_t T = 7, D = 16;  // x is [T, 2D]
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  auto silu_mul = reinterpret_cast<vt::SiluAndMulFn>(
      vt::GetOp(vt::OpId::kSiluAndMul, DeviceType::kTENSTORRENT));

  std::vector<float> host_x(T * 2 * D), host_out(T * D, 0.0f);
  for (size_t i = 0; i < host_x.size(); ++i)
    host_x[i] = (static_cast<float>(i % 13) - 6.0f) * 0.2f;

  void* mem_x = backend.Alloc(host_x.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_x, host_x.data(), host_x.size() * sizeof(float));

  Tensor x = Tensor::Contiguous(mem_x, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                {T, 2 * D});
  Tensor out =
      Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, D});
  silu_mul(q, out, x);
  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_x);
  backend.Free(mem_out);

  float max_abs_diff = 0.0f;
  for (int64_t i = 0; i < T; ++i) {
    for (int64_t j = 0; j < D; ++j) {
      const float gate = host_x[static_cast<size_t>(i * 2 * D + j)];
      const float up = host_x[static_cast<size_t>(i * 2 * D + D + j)];
      const float ref = (gate / (1.0f + std::exp(-gate))) * up;
      max_abs_diff =
          std::max(max_abs_diff, std::fabs(host_out[static_cast<size_t>(i * D + j)] - ref));
    }
  }
  // BF16 tile storage + silu; same envelope as kRelu / kRmsNorm.
  CHECK(max_abs_diff < 0.05f);
}

// Split-operand sibling of the case above: the GGUF dense MLP arm reaches
// vt::MoeSiluMul (qwen3_5.cpp DenseMlpBlock tail) with gate/up as TWO separate
// [T,I] GEMM outputs, so this op must be registered for the Qwen3.5 GGUF
// vehicle to run at all — the captured e2e previously fatalled on it
// ("no kernel for op MoeSiluMul ... on device tenstorrent").
TEST_CASE("kTENSTORRENT kMoeSiluMul matches host F32 within BF16 envelope") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMoeSiluMul, DeviceType::kTENSTORRENT));

  constexpr int64_t T = 7, D = 16;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  auto moe_silu_mul = reinterpret_cast<vt::MoeSiluMulFn>(
      vt::GetOp(vt::OpId::kMoeSiluMul, DeviceType::kTENSTORRENT));

  std::vector<float> host_g(T * D), host_u(T * D), host_out(T * D, 0.0f);
  for (size_t i = 0; i < host_g.size(); ++i)
    host_g[i] = (static_cast<float>(i % 13) - 6.0f) * 0.2f;
  for (size_t i = 0; i < host_u.size(); ++i)
    host_u[i] = (static_cast<float>(i % 7) - 3.0f) * 0.3f;

  void* mem_g = backend.Alloc(host_g.size() * sizeof(float));
  void* mem_u = backend.Alloc(host_u.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_g, host_g.data(), host_g.size() * sizeof(float));
  backend.Copy(q, mem_u, host_u.data(), host_u.size() * sizeof(float));

  Tensor g = Tensor::Contiguous(mem_g, vt::DType::kF32,
                                Device{DeviceType::kTENSTORRENT, 0}, {T, D});
  Tensor u = Tensor::Contiguous(mem_u, vt::DType::kF32,
                                Device{DeviceType::kTENSTORRENT, 0}, {T, D});
  Tensor out = Tensor::Contiguous(mem_out, vt::DType::kF32,
                                  Device{DeviceType::kTENSTORRENT, 0}, {T, D});
  moe_silu_mul(q, out, g, u);
  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_g);
  backend.Free(mem_u);
  backend.Free(mem_out);

  float max_abs_diff = 0.0f;
  for (int64_t i = 0; i < T; ++i) {
    for (int64_t j = 0; j < D; ++j) {
      const float gate = host_g[static_cast<size_t>(i * D + j)];
      const float up = host_u[static_cast<size_t>(i * D + j)];
      const float ref = (gate / (1.0f + std::exp(-gate))) * up;
      max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[static_cast<size_t>(i * D + j)] - ref));
    }
  }
  // BF16 tile storage + silu; same envelope as the kSiluAndMul sibling.
  CHECK(max_abs_diff < 0.05f);
}

// Cast pair used by Qwen3 K/V cache dtype and logits paths.
TEST_CASE("kTENSTORRENT kCastBf16 / kCastF32 round-trip F32 values") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kCastBf16, DeviceType::kTENSTORRENT));
  REQUIRE(vt::OpRegistered(vt::OpId::kCastF32, DeviceType::kTENSTORRENT));

  constexpr int64_t N = 64;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  auto cast_bf16 = reinterpret_cast<vt::CastBf16Fn>(
      vt::GetOp(vt::OpId::kCastBf16, DeviceType::kTENSTORRENT));
  auto cast_f32 = reinterpret_cast<vt::CastF32Fn>(
      vt::GetOp(vt::OpId::kCastF32, DeviceType::kTENSTORRENT));

  std::vector<float> host_f(N), host_back(N, 0.0f);
  for (int64_t i = 0; i < N; ++i) host_f[static_cast<size_t>(i)] = static_cast<float>(i) * 0.125f;

  void* mem_f = backend.Alloc(N * sizeof(float));
  void* mem_bf = backend.Alloc(N * sizeof(uint16_t));
  void* mem_back = backend.Alloc(N * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_f, host_f.data(), N * sizeof(float));

  Tensor tf =
      Tensor::Contiguous(mem_f, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {N});
  Tensor tbf =
      Tensor::Contiguous(mem_bf, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {N});
  Tensor tback =
      Tensor::Contiguous(mem_back, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {N});
  cast_bf16(q, tbf, tf);
  cast_f32(q, tback, tbf);
  backend.Copy(q, host_back.data(), mem_back, N * sizeof(float));
  backend.Free(mem_f);
  backend.Free(mem_bf);
  backend.Free(mem_back);

  for (int64_t i = 0; i < N; ++i) {
    // Exact for values representable in bf16 (i * 0.125).
    REQUIRE(host_back[static_cast<size_t>(i)] == host_f[static_cast<size_t>(i)]);
  }
}

// RED-first for ISSUE-LOCAL-01M2K8WWA0X09VJF55NTS16VTV: a producer shadow
// natively shaped [128, 6144] (the APEX gated-norm activation) reaches
// kCastBf16, which serves it raw and hands NormalizeDevF32Tile(1, n). The
// (1, n) arguments are the FLATTENED shape, so the logical-shape guard fires
// and the free ttnn::reshape launched reshape_rm — whose staging CBs scale
// with the 3,145,728 B f32 row, 2x the L1 cap, fatal at program allocation.
// The shadow is device-authoritative (committed by a device Matmul, the same
// CommitDevice2D path the production driver uses) and the oracle is the host
// single-round RNE cast.
static uint16_t RneF32ToBf16(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, 4);
  const uint32_t lsb = (bits >> 16) & 1u;
  const uint32_t rounded = bits + 0x7fffu + lsb;
  return static_cast<uint16_t>(rounded >> 16);
}

TEST_CASE("kTENSTORRENT kCastBf16 serves a [128,6144] f32 shadow without the "
          "L1-fatal reshape") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kCastBf16, DeviceType::kTENSTORRENT));
  constexpr int64_t M = 128, K = 16, C = 6144;
  const Device dev{DeviceType::kTENSTORRENT, 0};
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  auto cast_bf16 = reinterpret_cast<vt::CastBf16Fn>(
      vt::GetOp(vt::OpId::kCastBf16, DeviceType::kTENSTORRENT));

  // Producer: a device Matmul commits an f32 [M, C] shadow on mo. The b leg
  // is all ones and every a value is exact, so the shadow holds known f32
  // values without any readback of the committed slot.
  std::vector<float> ha(static_cast<size_t>(M * K));
  for (size_t i = 0; i < ha.size(); ++i)
    ha[i] = static_cast<float>(static_cast<int>(i % 15) - 7) * 0.25f;
  std::vector<float> hb(static_cast<size_t>(K * C), 1.0f);
  std::vector<float> hout(static_cast<size_t>(M * C), -1.0f);
  void* ma = backend.Alloc(ha.size() * sizeof(float));
  void* mb = backend.Alloc(hb.size() * sizeof(float));
  void* mo = backend.Alloc(hout.size() * sizeof(float));
  void* mo_bf = backend.Alloc(hout.size() * sizeof(uint16_t));
  Queue q = backend.CreateQueue();
  backend.Copy(q, ma, ha.data(), ha.size() * sizeof(float));
  backend.Copy(q, mb, hb.data(), hb.size() * sizeof(float));
  Tensor ta = Tensor::Contiguous(ma, vt::DType::kF32, dev, {M, K});
  Tensor tb = Tensor::Contiguous(mb, vt::DType::kF32, dev, {K, C});
  Tensor to = Tensor::Contiguous(mo, vt::DType::kF32, dev, {M, C});
  reinterpret_cast<vt::MatmulFn>(
      vt::GetOp(vt::OpId::kMatmul, DeviceType::kTENSTORRENT))(q, to, ta, tb);

  // The cast input rides the committed [M, C] shadow (device-authoritative:
  // device_current, host_current=false, dtype f32, numel == 786,432). The
  // f32 wide-row shadow is built through the production kCastF32 arm on the
  // committed matmul shadow, so the case exercises the 3,145,728 B f32 row
  // the APEX e2e dies on, not just the bf16 one.
  Tensor in2d = Tensor::Contiguous(mo, vt::DType::kF32, dev, {M, C});
  void* mo_f32 = backend.Alloc(hout.size() * sizeof(float));
  Tensor tf32 = Tensor::Contiguous(mo_f32, vt::DType::kF32, dev, {M * C});
  auto cast_f32 = reinterpret_cast<vt::CastF32Fn>(
      vt::GetOp(vt::OpId::kCastF32, DeviceType::kTENSTORRENT));
  cast_f32(q, tf32, in2d);
  Tensor in_f32 = Tensor::Contiguous(mo_f32, vt::DType::kF32, dev, {M, C});
  Tensor out = Tensor::Contiguous(mo_bf, vt::DType::kBF16, dev, {M * C});
  cast_bf16(q, out, in_f32);

  // Readback and bit-exact compare against the host single-round RNE cast.
  std::vector<uint16_t> got(static_cast<size_t>(M * C), 0xbeef);
  backend.Copy(q, got.data(), mo_bf, got.size() * sizeof(uint16_t));
  size_t bad = 0, first_bad = 0;
  for (int64_t i = 0; i < M; ++i) {
    const uint16_t want = RneF32ToBf16(ha[static_cast<size_t>(i * K)]);
    for (int64_t j = 0; j < C; ++j) {
      const size_t idx = static_cast<size_t>(i * C + j);
      if (got[idx] != want) {
        if (bad == 0) first_bad = idx;
        ++bad;
      }
    }
  }
  backend.Free(ma);
  backend.Free(mb);
  backend.Free(mo);
  backend.Free(mo_f32);
  backend.Free(mo_bf);
  CHECK_MESSAGE(bad == 0, "RNE mismatch: " << bad << " elements, first at "
                                           << first_bad);
}

// Default Qwen3-dense RoPE (Metal M3b). Small T*H uses host apply (bit-exact);
// large prefill uses device NeoX (BF16) — covered by PreferDeviceRope threshold.
TEST_CASE("kTENSTORRENT kRopeNeox is BIT-EXACT vs a host F32 reference (small)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  // This case asserts the HOST apply path (bit-exact). An ambient
  // VT_TT_HOST_FREE_DECODE (e.g. a suite run under the host-free gate) flips
  // PreferDeviceRope to the device BF16 path even at small T*H and reds the
  // bit-exact checks — so the case owns its own default-path env, mirroring
  // the inertness-guard case below. The opt-out is restored afterwards: a
  // leaked "0" silently disabled the host-free lane for EVERY later case in
  // the suite (found via ISSUE-LOCAL-01M3JXEFQKSZP23PP2HWY9G0VQ's fresh-slot
  // memset case, which needs the default host-free warmup).
  const bool had_hf = std::getenv("VT_TT_HOST_FREE_DECODE") != nullptr;
  const std::string saved_hf =
      had_hf ? std::string(std::getenv("VT_TT_HOST_FREE_DECODE")) : std::string();
  struct RestoreHf {
    bool had;
    std::string saved;
    ~RestoreHf() {
      if (had) ::setenv("VT_TT_HOST_FREE_DECODE", saved.c_str(), 1);
      else ::unsetenv("VT_TT_HOST_FREE_DECODE");
    }
  } restore_hf{had_hf, saved_hf};
  ::setenv("VT_TT_HOST_FREE_DECODE", "0", 1);  // opt-out path
  REQUIRE(vt::OpRegistered(vt::OpId::kRopeNeox, DeviceType::kTENSTORRENT));

  constexpr int64_t T = 4, Hq = 2, Hk = 1, Dh = 8, Rot = 8;
  constexpr float Base = 10000.0f;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  auto rope = reinterpret_cast<vt::RopeFn>(
      vt::GetOp(vt::OpId::kRopeNeox, DeviceType::kTENSTORRENT));

  std::vector<float> hq(T * Hq * Dh), hk(T * Hk * Dh), hq_ref, hk_ref;
  std::vector<int32_t> pos(T);
  for (int64_t i = 0; i < T * Hq * Dh; ++i)
    hq[static_cast<size_t>(i)] = (static_cast<float>(i % 11) - 5.0f) * 0.1f;
  for (int64_t i = 0; i < T * Hk * Dh; ++i)
    hk[static_cast<size_t>(i)] = (static_cast<float>(i % 7) - 3.0f) * 0.1f;
  for (int64_t i = 0; i < T; ++i) pos[static_cast<size_t>(i)] = static_cast<int32_t>(i + 1);
  hq_ref = hq;
  hk_ref = hk;

  auto rotate_head = [&](std::vector<float>& t, int64_t head_off, int64_t p) {
    const int half = Rot / 2;
    for (int i = 0; i < half; ++i) {
      const double freq =
          std::pow(static_cast<double>(Base), -2.0 * i / static_cast<double>(Rot));
      const double angle = static_cast<double>(p) * freq;
      const float c = static_cast<float>(std::cos(angle));
      const float s = static_cast<float>(std::sin(angle));
      const float x = t[static_cast<size_t>(head_off + i)];
      const float y = t[static_cast<size_t>(head_off + i + half)];
      t[static_cast<size_t>(head_off + i)] = x * c - y * s;
      t[static_cast<size_t>(head_off + i + half)] = x * s + y * c;
    }
  };
  for (int64_t i = 0; i < T; ++i) {
    for (int64_t h = 0; h < Hq; ++h) rotate_head(hq_ref, (i * Hq + h) * Dh, pos[static_cast<size_t>(i)]);
    for (int64_t h = 0; h < Hk; ++h) rotate_head(hk_ref, (i * Hk + h) * Dh, pos[static_cast<size_t>(i)]);
  }

  void* mem_q = backend.Alloc(hq.size() * sizeof(float));
  void* mem_k = backend.Alloc(hk.size() * sizeof(float));
  void* mem_p = backend.Alloc(pos.size() * sizeof(int32_t));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_q, hq.data(), hq.size() * sizeof(float));
  backend.Copy(q, mem_k, hk.data(), hk.size() * sizeof(float));
  backend.Copy(q, mem_p, pos.data(), pos.size() * sizeof(int32_t));

  Tensor tq = Tensor::Contiguous(mem_q, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                 {T, Hq, Dh});
  Tensor tk = Tensor::Contiguous(mem_k, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                 {T, Hk, Dh});
  Tensor tp = Tensor::Contiguous(mem_p, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {T});
  vt::RopeArgs args;
  args.base = Base;
  args.rotary_dim = Rot;
  rope(q, tq, tk, tp, args);

  backend.Copy(q, hq.data(), mem_q, hq.size() * sizeof(float));
  backend.Copy(q, hk.data(), mem_k, hk.size() * sizeof(float));
  backend.Free(mem_q);
  backend.Free(mem_k);
  backend.Free(mem_p);

  for (size_t i = 0; i < hq.size(); ++i) REQUIRE(hq[i] == hq_ref[i]);
  for (size_t i = 0; i < hk.size(); ++i) REQUIRE(hk[i] == hk_ref[i]);
}

TEST_CASE("kTENSTORRENT kRopeCosSinCache + kRopeFromCache match kRopeNeox") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kRopeCosSinCache, DeviceType::kTENSTORRENT));
  REQUIRE(vt::OpRegistered(vt::OpId::kRopeFromCache, DeviceType::kTENSTORRENT));
  REQUIRE(vt::OpRegistered(vt::OpId::kRopeNeox, DeviceType::kTENSTORRENT));

  constexpr int64_t T = 3, Hq = 2, Hk = 1, Dh = 8, Rot = 8;
  constexpr float Base = 10000.0f;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  auto rope_neox = reinterpret_cast<vt::RopeFn>(
      vt::GetOp(vt::OpId::kRopeNeox, DeviceType::kTENSTORRENT));
  auto rope_cache = reinterpret_cast<vt::RopeCosSinCacheFn>(
      vt::GetOp(vt::OpId::kRopeCosSinCache, DeviceType::kTENSTORRENT));
  auto rope_from = reinterpret_cast<vt::RopeFromCacheFn>(
      vt::GetOp(vt::OpId::kRopeFromCache, DeviceType::kTENSTORRENT));

  std::vector<float> q0(T * Hq * Dh), k0(T * Hk * Dh);
  std::vector<int32_t> pos(T);
  for (size_t i = 0; i < q0.size(); ++i) q0[i] = (static_cast<float>(i % 9) - 4.0f) * 0.05f;
  for (size_t i = 0; i < k0.size(); ++i) k0[i] = (static_cast<float>(i % 5) - 2.0f) * 0.05f;
  for (int64_t i = 0; i < T; ++i) pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);

  auto alloc_copy = [&](const void* src, size_t bytes) {
    void* p = backend.Alloc(bytes);
    Queue q = backend.CreateQueue();
    backend.Copy(q, p, src, bytes);
    return p;
  };

  void* mq1 = alloc_copy(q0.data(), q0.size() * sizeof(float));
  void* mk1 = alloc_copy(k0.data(), k0.size() * sizeof(float));
  void* mq2 = alloc_copy(q0.data(), q0.size() * sizeof(float));
  void* mk2 = alloc_copy(k0.data(), k0.size() * sizeof(float));
  void* mp = alloc_copy(pos.data(), pos.size() * sizeof(int32_t));
  void* mcs = backend.Alloc(static_cast<size_t>(T * Rot) * sizeof(float));
  // Identity row index into a T-row cache built from positions 0..T-1.
  std::vector<int32_t> rows(T);
  for (int64_t i = 0; i < T; ++i) rows[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  void* mrow = alloc_copy(rows.data(), rows.size() * sizeof(int32_t));

  Queue q = backend.CreateQueue();
  Tensor tq1 = Tensor::Contiguous(mq1, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {T, Hq, Dh});
  Tensor tk1 = Tensor::Contiguous(mk1, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {T, Hk, Dh});
  Tensor tq2 = Tensor::Contiguous(mq2, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {T, Hq, Dh});
  Tensor tk2 = Tensor::Contiguous(mk2, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {T, Hk, Dh});
  Tensor tp = Tensor::Contiguous(mp, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {T});
  Tensor tcs =
      Tensor::Contiguous(mcs, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, Rot});
  Tensor trow =
      Tensor::Contiguous(mrow, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {T});

  vt::RopeArgs args;
  args.base = Base;
  args.rotary_dim = Rot;
  args.is_neox_style = true;

  rope_neox(q, tq1, tk1, tp, args);
  rope_cache(q, tcs, tp, args);
  rope_from(q, tq2, &tk2, trow, tcs, args);

  std::vector<float> qn(q0.size()), kn(k0.size()), qc(q0.size()), kc(k0.size());
  backend.Copy(q, qn.data(), mq1, qn.size() * sizeof(float));
  backend.Copy(q, kn.data(), mk1, kn.size() * sizeof(float));
  backend.Copy(q, qc.data(), mq2, qc.size() * sizeof(float));
  backend.Copy(q, kc.data(), mk2, kc.size() * sizeof(float));
  for (void* p : {mq1, mk1, mq2, mk2, mp, mcs, mrow}) backend.Free(p);

  // Small T*H → host apply on both paths → bit-identical.
  for (size_t i = 0; i < qn.size(); ++i) REQUIRE(qn[i] == qc[i]);
  for (size_t i = 0; i < kn.size(); ++i) REQUIRE(kn[i] == kc[i]);
}

TEST_CASE("kTENSTORRENT kQkvSplit is BIT-EXACT vs a host reference (unequal widths)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kQkvSplit, DeviceType::kTENSTORRENT));

  // Independent q/k/v widths (kernel contract); not equal-width MHA only.
  constexpr int64_t T = 11, Qd = 24, Kd = 12, Vd = 12;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_qkv(static_cast<size_t>(T * (Qd + Kd + Vd)));
  for (size_t i = 0; i < host_qkv.size(); ++i)
    host_qkv[i] = static_cast<float>(i % 19) * 0.1f - 0.7f;
  std::vector<float> host_q(static_cast<size_t>(T * Qd), 0.0f);
  std::vector<float> host_k(static_cast<size_t>(T * Kd), 0.0f);
  std::vector<float> host_v(static_cast<size_t>(T * Vd), 0.0f);

  void* mem_qkv = backend.Alloc(host_qkv.size() * sizeof(float));
  void* mem_q = backend.Alloc(host_q.size() * sizeof(float));
  void* mem_k = backend.Alloc(host_k.size() * sizeof(float));
  void* mem_v = backend.Alloc(host_v.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_qkv, host_qkv.data(), host_qkv.size() * sizeof(float));

  Tensor qkv = Tensor::Contiguous(mem_qkv, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {T, Qd + Kd + Vd});
  Tensor tq =
      Tensor::Contiguous(mem_q, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, Qd});
  Tensor tk =
      Tensor::Contiguous(mem_k, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, Kd});
  Tensor tv =
      Tensor::Contiguous(mem_v, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, Vd});

  auto split =
      reinterpret_cast<vt::QkvSplitFn>(vt::GetOp(vt::OpId::kQkvSplit, DeviceType::kTENSTORRENT));
  split(q, tq, tk, tv, qkv);

  backend.Copy(q, host_q.data(), mem_q, host_q.size() * sizeof(float));
  backend.Copy(q, host_k.data(), mem_k, host_k.size() * sizeof(float));
  backend.Copy(q, host_v.data(), mem_v, host_v.size() * sizeof(float));
  backend.Free(mem_qkv);
  backend.Free(mem_q);
  backend.Free(mem_k);
  backend.Free(mem_v);

  // Pure column split — bit-exact.
  for (int64_t i = 0; i < T; ++i) {
    for (int64_t j = 0; j < Qd; ++j)
      CHECK(host_q[i * Qd + j] == host_qkv[i * (Qd + Kd + Vd) + j]);
    for (int64_t j = 0; j < Kd; ++j)
      CHECK(host_k[i * Kd + j] == host_qkv[i * (Qd + Kd + Vd) + Qd + j]);
    for (int64_t j = 0; j < Vd; ++j)
      CHECK(host_v[i * Vd + j] == host_qkv[i * (Qd + Kd + Vd) + Qd + Kd + j]);
  }
}

// Device path: make qkv device-resident (kRelu), then split without host memcpy.
// Matches e2e MatmulBT → QkvSplit residency chain.
TEST_CASE("kTENSTORRENT kQkvSplit device path matches host within BF16 envelope") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kQkvSplit, DeviceType::kTENSTORRENT));
  REQUIRE(vt::OpRegistered(vt::OpId::kRelu, DeviceType::kTENSTORRENT));

  constexpr int64_t T = 4, Qd = 32, Kd = 16, Vd = 16;  // total=64, tile-friendly
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  auto relu = reinterpret_cast<vt::ReluFn>(vt::GetOp(vt::OpId::kRelu, DeviceType::kTENSTORRENT));
  auto split =
      reinterpret_cast<vt::QkvSplitFn>(vt::GetOp(vt::OpId::kQkvSplit, DeviceType::kTENSTORRENT));

  std::vector<float> host_qkv(static_cast<size_t>(T * (Qd + Kd + Vd)));
  for (size_t i = 0; i < host_qkv.size(); ++i)
    host_qkv[i] = std::fabs(static_cast<float>(i % 19) * 0.1f - 0.3f) + 0.05f;  // all > 0

  void* mem_qkv = backend.Alloc(host_qkv.size() * sizeof(float));
  void* mem_q = backend.Alloc(static_cast<size_t>(T * Qd) * sizeof(float));
  void* mem_k = backend.Alloc(static_cast<size_t>(T * Kd) * sizeof(float));
  void* mem_v = backend.Alloc(static_cast<size_t>(T * Vd) * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_qkv, host_qkv.data(), host_qkv.size() * sizeof(float));

  Tensor qkv = Tensor::Contiguous(mem_qkv, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {T, Qd + Kd + Vd});
  Tensor tq =
      Tensor::Contiguous(mem_q, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, Qd});
  Tensor tk =
      Tensor::Contiguous(mem_k, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, Kd});
  Tensor tv =
      Tensor::Contiguous(mem_v, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, Vd});

  // Relu in place → CommitDevice2D leaves qkv device-resident (positive inputs).
  relu(q, qkv, qkv);
  split(q, tq, tk, tv, qkv);

  std::vector<float> host_q(static_cast<size_t>(T * Qd)), host_k(static_cast<size_t>(T * Kd)),
      host_v(static_cast<size_t>(T * Vd));
  backend.Copy(q, host_q.data(), mem_q, host_q.size() * sizeof(float));
  backend.Copy(q, host_k.data(), mem_k, host_k.size() * sizeof(float));
  backend.Copy(q, host_v.data(), mem_v, host_v.size() * sizeof(float));
  backend.Free(mem_qkv);
  backend.Free(mem_q);
  backend.Free(mem_k);
  backend.Free(mem_v);

  float max_abs = 0.0f;
  for (int64_t i = 0; i < T; ++i) {
    for (int64_t j = 0; j < Qd; ++j)
      max_abs = std::max(
          max_abs, std::fabs(host_q[static_cast<size_t>(i * Qd + j)] -
                             host_qkv[static_cast<size_t>(i * (Qd + Kd + Vd) + j)]));
    for (int64_t j = 0; j < Kd; ++j)
      max_abs = std::max(
          max_abs, std::fabs(host_k[static_cast<size_t>(i * Kd + j)] -
                             host_qkv[static_cast<size_t>(i * (Qd + Kd + Vd) + Qd + j)]));
    for (int64_t j = 0; j < Vd; ++j)
      max_abs = std::max(
          max_abs,
          std::fabs(host_v[static_cast<size_t>(i * Vd + j)] -
                    host_qkv[static_cast<size_t>(i * (Qd + Kd + Vd) + Qd + Kd + j)]));
  }
  CHECK(max_abs < 0.05f);
}

TEST_CASE("kTENSTORRENT kReshapeAndCache is BIT-EXACT incl. slot<0 skip") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kReshapeAndCache, DeviceType::kTENSTORRENT));

  constexpr int64_t NBlocks = 6, Bsz = 8, Hkv = 3, Dh = 16, T = 10;
  constexpr int64_t Page = Hkv * Dh;
  const size_t cache_elems = static_cast<size_t>(NBlocks * Bsz * Hkv * Dh);
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_k(static_cast<size_t>(T * Page)), host_v(static_cast<size_t>(T * Page));
  for (size_t i = 0; i < host_k.size(); ++i) {
    host_k[i] = static_cast<float>(i % 11) * 0.1f;
    host_v[i] = static_cast<float>(i % 13) * 0.05f - 0.2f;
  }
  // Scattered slots + one padded (-1) token that must leave its page untouched.
  std::vector<int64_t> slots{0, 9, 17, 3, -1, 40, 25, 8, 33, 11};
  REQUIRE(static_cast<int64_t>(slots.size()) == T);

  std::vector<float> seed(cache_elems);
  for (size_t i = 0; i < seed.size(); ++i) seed[i] = static_cast<float>(i % 977) * 0.001f;
  std::vector<float> host_kc = seed, host_vc = seed;

  void* mem_k = backend.Alloc(host_k.size() * sizeof(float));
  void* mem_v = backend.Alloc(host_v.size() * sizeof(float));
  void* mem_kc = backend.Alloc(cache_elems * sizeof(float));
  void* mem_vc = backend.Alloc(cache_elems * sizeof(float));
  void* mem_slots = backend.Alloc(slots.size() * sizeof(int64_t));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_k, host_k.data(), host_k.size() * sizeof(float));
  backend.Copy(q, mem_v, host_v.data(), host_v.size() * sizeof(float));
  backend.Copy(q, mem_kc, host_kc.data(), host_kc.size() * sizeof(float));
  backend.Copy(q, mem_vc, host_vc.data(), host_vc.size() * sizeof(float));
  backend.Copy(q, mem_slots, slots.data(), slots.size() * sizeof(int64_t));

  Tensor tk = Tensor::Contiguous(mem_k, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                 {T, Hkv, Dh});
  Tensor tv = Tensor::Contiguous(mem_v, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                 {T, Hkv, Dh});
  Tensor tkc = Tensor::Contiguous(mem_kc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, Dh});
  Tensor tvc = Tensor::Contiguous(mem_vc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, Dh});
  Tensor tsl = Tensor::Contiguous(mem_slots, vt::DType::kI64, Device{DeviceType::kTENSTORRENT, 0},
                                  {T});

  auto rac = reinterpret_cast<vt::ReshapeAndCacheFn>(
      vt::GetOp(vt::OpId::kReshapeAndCache, DeviceType::kTENSTORRENT));
  rac(q, tk, tv, tkc, tvc, tsl);

  backend.Copy(q, host_kc.data(), mem_kc, host_kc.size() * sizeof(float));
  backend.Copy(q, host_vc.data(), mem_vc, host_vc.size() * sizeof(float));
  backend.Free(mem_k);
  backend.Free(mem_v);
  backend.Free(mem_kc);
  backend.Free(mem_vc);
  backend.Free(mem_slots);

  // Host oracle: same stride math as cpu_cache.cpp.
  std::vector<float> ref_kc = seed, ref_vc = seed;
  for (int64_t t = 0; t < T; ++t) {
    const int64_t slot = slots[static_cast<size_t>(t)];
    if (slot < 0) continue;
    const int64_t block = slot / Bsz;
    const int64_t offset = slot % Bsz;
    const int64_t dst = (block * Bsz + offset) * Page;
    std::memcpy(ref_kc.data() + dst, host_k.data() + t * Page, static_cast<size_t>(Page) * sizeof(float));
    std::memcpy(ref_vc.data() + dst, host_v.data() + t * Page, static_cast<size_t>(Page) * sizeof(float));
  }
  CHECK(host_kc == ref_kc);
  CHECK(host_vc == ref_vc);
}

TEST_CASE("kTENSTORRENT kPagedAttention matches a host causal GQA reference") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kPagedAttention, DeviceType::kTENSTORRENT));

  // Single-request prefill: T=4 tokens, Hq=4, Hkv=2 (GQA 2:1), D=8, block=4.
  constexpr int64_t T = 4, Hq = 4, Hkv = 2, D = 8, Bsz = 4, NBlocks = 2;
  constexpr int64_t Page = Hkv * D;
  const size_t cache_elems = static_cast<size_t>(NBlocks * Bsz * Page);
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_q(static_cast<size_t>(T * Hq * D));
  std::vector<float> host_k(static_cast<size_t>(T * Page)), host_v(static_cast<size_t>(T * Page));
  for (size_t i = 0; i < host_q.size(); ++i) host_q[i] = static_cast<float>(i % 7) * 0.1f - 0.3f;
  for (size_t i = 0; i < host_k.size(); ++i) {
    host_k[i] = static_cast<float>(i % 5) * 0.15f;
    host_v[i] = static_cast<float>(i % 9) * 0.05f - 0.1f;
  }
  // Write K/V into contiguous slots 0..T-1 of the cache first.
  std::vector<float> host_kc(cache_elems, 0.0f), host_vc(cache_elems, 0.0f);
  for (int64_t t = 0; t < T; ++t) {
    std::memcpy(host_kc.data() + t * Page, host_k.data() + t * Page,
                static_cast<size_t>(Page) * sizeof(float));
    std::memcpy(host_vc.data() + t * Page, host_v.data() + t * Page,
                static_cast<size_t>(Page) * sizeof(float));
  }
  std::vector<int32_t> block_table{0, 1};  // [num_reqs=1, max_blocks=2]
  std::vector<int32_t> seq_lens{static_cast<int32_t>(T)};
  std::vector<int32_t> qsl{0, static_cast<int32_t>(T)};
  std::vector<float> host_out(static_cast<size_t>(T * Hq * D), 0.0f);

  void* mem_q = backend.Alloc(host_q.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  void* mem_kc = backend.Alloc(host_kc.size() * sizeof(float));
  void* mem_vc = backend.Alloc(host_vc.size() * sizeof(float));
  void* mem_bt = backend.Alloc(block_table.size() * sizeof(int32_t));
  void* mem_sl = backend.Alloc(seq_lens.size() * sizeof(int32_t));
  void* mem_qsl = backend.Alloc(qsl.size() * sizeof(int32_t));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_q, host_q.data(), host_q.size() * sizeof(float));
  backend.Copy(q, mem_kc, host_kc.data(), host_kc.size() * sizeof(float));
  backend.Copy(q, mem_vc, host_vc.data(), host_vc.size() * sizeof(float));
  backend.Copy(q, mem_bt, block_table.data(), block_table.size() * sizeof(int32_t));
  backend.Copy(q, mem_sl, seq_lens.data(), seq_lens.size() * sizeof(int32_t));
  backend.Copy(q, mem_qsl, qsl.data(), qsl.size() * sizeof(int32_t));

  Tensor tq =
      Tensor::Contiguous(mem_q, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, Hq, D});
  Tensor tout = Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                   {T, Hq, D});
  Tensor tkc = Tensor::Contiguous(mem_kc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, D});
  Tensor tvc = Tensor::Contiguous(mem_vc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, D});
  Tensor tbt = Tensor::Contiguous(mem_bt, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0},
                                  {1, 2});
  Tensor tsl =
      Tensor::Contiguous(mem_sl, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {1});
  Tensor tqsl =
      Tensor::Contiguous(mem_qsl, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {2});

  vt::PagedAttentionArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(D));
  args.causal = true;
  auto pa = reinterpret_cast<vt::PagedAttentionFn>(
      vt::GetOp(vt::OpId::kPagedAttention, DeviceType::kTENSTORRENT));
  pa(q, tout, tq, tkc, tvc, tbt, tsl, tqsl, args);

  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_q);
  backend.Free(mem_out);
  backend.Free(mem_kc);
  backend.Free(mem_vc);
  backend.Free(mem_bt);
  backend.Free(mem_sl);
  backend.Free(mem_qsl);

  // Host oracle: same two-pass max-subtracted softmax as cpu_paged_attn.cpp.
  std::vector<float> ref(static_cast<size_t>(T * Hq * D), 0.0f);
  const int64_t qpk = Hq / Hkv;
  for (int64_t t = 0; t < T; ++t) {
    const int64_t p = t;  // prefill, context=0
    for (int64_t h = 0; h < Hq; ++h) {
      const int64_t g = h / qpk;
      const int64_t qoff = (t * Hq + h) * D;
      float m = -std::numeric_limits<float>::infinity();
      std::vector<float> scores(static_cast<size_t>(p + 1));
      for (int64_t j = 0; j <= p; ++j) {
        float dot = 0.0f;
        for (int64_t e = 0; e < D; ++e)
          dot += host_q[static_cast<size_t>(qoff + e)] *
                 host_kc[static_cast<size_t>(j * Page + g * D + e)];
        scores[static_cast<size_t>(j)] = dot * args.scale;
        m = std::max(m, scores[static_cast<size_t>(j)]);
      }
      float denom = 0.0f;
      for (int64_t j = 0; j <= p; ++j) {
        scores[static_cast<size_t>(j)] = std::exp(scores[static_cast<size_t>(j)] - m);
        denom += scores[static_cast<size_t>(j)];
      }
      const float inv = 1.0f / denom;
      for (int64_t e = 0; e < D; ++e) {
        float acc = 0.0f;
        for (int64_t j = 0; j <= p; ++j)
          acc += scores[static_cast<size_t>(j)] * inv *
                 host_vc[static_cast<size_t>(j * Page + g * D + e)];
        ref[static_cast<size_t>(qoff + e)] = acc;
      }
    }
  }

  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i)
    max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[i] - ref[i]));
  // Host f32 path — should be essentially bit-exact; allow tiny float noise.
  CHECK(max_abs_diff < 1e-5f);
}

// Optional microbench (TT_PA_BENCH=1): Qwen3-0.6B-ish decode shape at long
// context to measure host PA throughput independent of e2e matmul/PCIe.
TEST_CASE("kTENSTORRENT kPagedAttention host microbench (opt-in)") {
  if (std::getenv("TT_PA_BENCH") == nullptr) {
    MESSAGE("SKIPPED: set TT_PA_BENCH=1 to run host PA microbench");
    return;
  }
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  // Decode-shaped: T=1, Hq=16, Hkv=8, D=128, seqlen=512, block=16.
  constexpr int64_t T = 1, Hq = 16, Hkv = 8, D = 128, Bsz = 16, Seq = 512;
  constexpr int64_t NBlocks = Seq / Bsz;
  constexpr int64_t Page = Hkv * D;
  const size_t cache_elems = static_cast<size_t>(NBlocks * Bsz * Page);
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_q(static_cast<size_t>(T * Hq * D), 0.1f);
  std::vector<float> host_kc(cache_elems, 0.05f), host_vc(cache_elems, 0.02f);
  std::vector<int32_t> block_table(static_cast<size_t>(NBlocks));
  for (int64_t i = 0; i < NBlocks; ++i) block_table[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  std::vector<int32_t> seq_lens{static_cast<int32_t>(Seq)};
  std::vector<int32_t> qsl{0, static_cast<int32_t>(T)};
  std::vector<float> host_out(static_cast<size_t>(T * Hq * D), 0.0f);

  void* mem_q = backend.Alloc(host_q.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  void* mem_kc = backend.Alloc(host_kc.size() * sizeof(float));
  void* mem_vc = backend.Alloc(host_vc.size() * sizeof(float));
  void* mem_bt = backend.Alloc(block_table.size() * sizeof(int32_t));
  void* mem_sl = backend.Alloc(seq_lens.size() * sizeof(int32_t));
  void* mem_qsl = backend.Alloc(qsl.size() * sizeof(int32_t));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_q, host_q.data(), host_q.size() * sizeof(float));
  backend.Copy(q, mem_kc, host_kc.data(), host_kc.size() * sizeof(float));
  backend.Copy(q, mem_vc, host_vc.data(), host_vc.size() * sizeof(float));
  backend.Copy(q, mem_bt, block_table.data(), block_table.size() * sizeof(int32_t));
  backend.Copy(q, mem_sl, seq_lens.data(), seq_lens.size() * sizeof(int32_t));
  backend.Copy(q, mem_qsl, qsl.data(), qsl.size() * sizeof(int32_t));

  Tensor tq =
      Tensor::Contiguous(mem_q, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, Hq, D});
  Tensor tout = Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                   {T, Hq, D});
  Tensor tkc = Tensor::Contiguous(mem_kc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, D});
  Tensor tvc = Tensor::Contiguous(mem_vc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, D});
  Tensor tbt = Tensor::Contiguous(mem_bt, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0},
                                  {1, NBlocks});
  Tensor tsl =
      Tensor::Contiguous(mem_sl, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {1});
  Tensor tqsl =
      Tensor::Contiguous(mem_qsl, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {2});

  vt::PagedAttentionArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(D));
  args.causal = true;
  auto pa = reinterpret_cast<vt::PagedAttentionFn>(
      vt::GetOp(vt::OpId::kPagedAttention, DeviceType::kTENSTORRENT));

  constexpr int kWarm = 3, kIters = 20;
  for (int i = 0; i < kWarm; ++i) pa(q, tout, tq, tkc, tvc, tbt, tsl, tqsl, args);
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kIters; ++i) pa(q, tout, tq, tkc, tvc, tbt, tsl, tqsl, args);
  const auto t1 = std::chrono::steady_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count() / static_cast<double>(kIters);
  MESSAGE("PA host microbench T=1 Hq=16 Hkv=8 D=128 seq=512: ", ms, " ms/call");

  backend.Free(mem_q);
  backend.Free(mem_out);
  backend.Free(mem_kc);
  backend.Free(mem_vc);
  backend.Free(mem_bt);
  backend.Free(mem_sl);
  backend.Free(mem_qsl);
  CHECK(ms > 0.0);
}

// Pure decode with TILE-legal geometry (D=128, block=32) exercises the
// ttnn::paged_scaled_dot_product_attention_decode path (or host fallback).
TEST_CASE("kTENSTORRENT kPagedAttention pure-decode matches host within BF16 envelope") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kPagedAttention, DeviceType::kTENSTORRENT));

  constexpr int64_t Hq = 4, Hkv = 2, D = 128, Bsz = 32, Seq = 64;
  constexpr int64_t NBlocks = Seq / Bsz;  // 2
  constexpr int64_t Page = Hkv * D;
  const size_t cache_elems = static_cast<size_t>(NBlocks * Bsz * Page);
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_q(static_cast<size_t>(Hq * D));
  std::vector<float> host_kc(cache_elems), host_vc(cache_elems);
  for (size_t i = 0; i < host_q.size(); ++i)
    host_q[i] = (static_cast<float>(i % 17) - 8.0f) * 0.05f;
  for (size_t i = 0; i < host_kc.size(); ++i) {
    host_kc[i] = (static_cast<float>(i % 13) - 6.0f) * 0.03f;
    host_vc[i] = (static_cast<float>(i % 11) - 5.0f) * 0.02f;
  }
  std::vector<int32_t> block_table(static_cast<size_t>(NBlocks));
  for (int64_t i = 0; i < NBlocks; ++i) block_table[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  std::vector<int32_t> seq_lens{static_cast<int32_t>(Seq)};
  std::vector<int32_t> qsl{0, 1};
  std::vector<float> host_out(static_cast<size_t>(Hq * D), 0.0f);

  void* mem_q = backend.Alloc(host_q.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  void* mem_kc = backend.Alloc(host_kc.size() * sizeof(float));
  void* mem_vc = backend.Alloc(host_vc.size() * sizeof(float));
  void* mem_bt = backend.Alloc(block_table.size() * sizeof(int32_t));
  void* mem_sl = backend.Alloc(seq_lens.size() * sizeof(int32_t));
  void* mem_qsl = backend.Alloc(qsl.size() * sizeof(int32_t));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_q, host_q.data(), host_q.size() * sizeof(float));
  backend.Copy(q, mem_kc, host_kc.data(), host_kc.size() * sizeof(float));
  backend.Copy(q, mem_vc, host_vc.data(), host_vc.size() * sizeof(float));
  backend.Copy(q, mem_bt, block_table.data(), block_table.size() * sizeof(int32_t));
  backend.Copy(q, mem_sl, seq_lens.data(), seq_lens.size() * sizeof(int32_t));
  backend.Copy(q, mem_qsl, qsl.data(), qsl.size() * sizeof(int32_t));

  Tensor tq =
      Tensor::Contiguous(mem_q, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {1, Hq, D});
  Tensor tout = Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                   {1, Hq, D});
  Tensor tkc = Tensor::Contiguous(mem_kc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, D});
  Tensor tvc = Tensor::Contiguous(mem_vc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, D});
  Tensor tbt = Tensor::Contiguous(mem_bt, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0},
                                  {1, NBlocks});
  Tensor tsl =
      Tensor::Contiguous(mem_sl, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {1});
  Tensor tqsl =
      Tensor::Contiguous(mem_qsl, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {2});

  vt::PagedAttentionArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(D));
  args.causal = true;
  auto pa = reinterpret_cast<vt::PagedAttentionFn>(
      vt::GetOp(vt::OpId::kPagedAttention, DeviceType::kTENSTORRENT));
  pa(q, tout, tq, tkc, tvc, tbt, tsl, tqsl, args);

  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_q);
  backend.Free(mem_out);
  backend.Free(mem_kc);
  backend.Free(mem_vc);
  backend.Free(mem_bt);
  backend.Free(mem_sl);
  backend.Free(mem_qsl);

  // Host NHD oracle (same as unit test / cpu_paged_attn).
  const int64_t qpk = Hq / Hkv;
  const int64_t p = Seq - 1;
  float max_abs = 0.0f;
  for (int64_t h = 0; h < Hq; ++h) {
    const int64_t g = h / qpk;
    const int64_t qoff = h * D;
    float m = -std::numeric_limits<float>::infinity();
    std::vector<float> scores(static_cast<size_t>(p + 1));
    for (int64_t j = 0; j <= p; ++j) {
      float dot = 0.0f;
      const int64_t kbase = j * Page + g * D;
      for (int64_t e = 0; e < D; ++e)
        dot += host_q[static_cast<size_t>(qoff + e)] * host_kc[static_cast<size_t>(kbase + e)];
      scores[static_cast<size_t>(j)] = dot * args.scale;
      m = std::max(m, scores[static_cast<size_t>(j)]);
    }
    float denom = 0.0f;
    for (int64_t j = 0; j <= p; ++j) {
      scores[static_cast<size_t>(j)] = std::exp(scores[static_cast<size_t>(j)] - m);
      denom += scores[static_cast<size_t>(j)];
    }
    const float inv = 1.0f / denom;
    for (int64_t e = 0; e < D; ++e) {
      float acc = 0.0f;
      for (int64_t j = 0; j <= p; ++j)
        acc += scores[static_cast<size_t>(j)] * inv *
               host_vc[static_cast<size_t>(j * Page + g * D + e)];
      max_abs = std::max(max_abs, std::fabs(host_out[static_cast<size_t>(qoff + e)] - acc));
    }
  }
  // Device BF16 SDPA or host path — generous envelope.
  CHECK(max_abs < 0.5f);
}

// Batched (B=2) pure decode. RED-FIRST for
// ISSUE-LOCAL-01M2X9XSS0N7WB5BTJ0326B892: the eager (uncaptured) device
// decode path flattened the sdpa output [1,B,H,D] TILE to [B, H*D] through
// the free ttnn::reshape, which derives padded = logical and computes a
// physical shape up to 16x the buffer — "MeshBuffer must be large enough to
// hold the tensor" on every concurrency>=2 decode step, silently caught, and
// the op fell back to the host oracle. The B=1 case above never saw it: its
// decode step replays a captured graph. The device-path requirement is the
// red signal: the broken reshape forces CommitHost (no device shadow).
TEST_CASE("kTENSTORRENT kPagedAttention pure-decode batched B=2 keeps the device path") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kPagedAttention, DeviceType::kTENSTORRENT));

  constexpr int64_t Hq = 4, Hkv = 2, D = 128, Bsz = 32, Seq = 64, B = 2;
  constexpr int64_t NBlocksPerReq = Seq / Bsz;  // 2
  constexpr int64_t NBlocks = B * NBlocksPerReq;
  constexpr int64_t Page = Hkv * D;
  const size_t cache_elems = static_cast<size_t>(NBlocks * Bsz * Page);
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_q(static_cast<size_t>(B * Hq * D));
  std::vector<float> host_kc(cache_elems), host_vc(cache_elems);
  for (size_t i = 0; i < host_q.size(); ++i)
    host_q[i] = (static_cast<float>((i * 7) % 19) - 9.0f) * 0.05f;
  for (size_t i = 0; i < host_kc.size(); ++i) {
    host_kc[i] = (static_cast<float>((i * 5) % 15) - 7.0f) * 0.03f;
    host_vc[i] = (static_cast<float>((i * 3) % 13) - 6.0f) * 0.02f;
  }
  // Request r owns physical blocks {2r, 2r+1} — distinct pages per request.
  std::vector<int32_t> block_table(static_cast<size_t>(B * NBlocksPerReq));
  for (int64_t r = 0; r < B; ++r)
    for (int64_t c = 0; c < NBlocksPerReq; ++c)
      block_table[static_cast<size_t>(r * NBlocksPerReq + c)] =
          static_cast<int32_t>(r * NBlocksPerReq + c);
  std::vector<int32_t> seq_lens{static_cast<int32_t>(Seq),
                                static_cast<int32_t>(Seq)};
  std::vector<int32_t> qsl{0, 1, 2};
  std::vector<float> host_out(static_cast<size_t>(B * Hq * D), 0.0f);

  void* mem_q = backend.Alloc(host_q.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  void* mem_kc = backend.Alloc(host_kc.size() * sizeof(float));
  void* mem_vc = backend.Alloc(host_vc.size() * sizeof(float));
  void* mem_bt = backend.Alloc(block_table.size() * sizeof(int32_t));
  void* mem_sl = backend.Alloc(seq_lens.size() * sizeof(int32_t));
  void* mem_qsl = backend.Alloc(qsl.size() * sizeof(int32_t));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_q, host_q.data(), host_q.size() * sizeof(float));
  backend.Copy(q, mem_kc, host_kc.data(), host_kc.size() * sizeof(float));
  backend.Copy(q, mem_vc, host_vc.data(), host_vc.size() * sizeof(float));
  backend.Copy(q, mem_bt, block_table.data(), block_table.size() * sizeof(int32_t));
  backend.Copy(q, mem_sl, seq_lens.data(), seq_lens.size() * sizeof(int32_t));
  backend.Copy(q, mem_qsl, qsl.data(), qsl.size() * sizeof(int32_t));

  Tensor tq = Tensor::Contiguous(mem_q, vt::DType::kF32,
                                 Device{DeviceType::kTENSTORRENT, 0}, {B, Hq, D});
  Tensor tout = Tensor::Contiguous(mem_out, vt::DType::kF32,
                                   Device{DeviceType::kTENSTORRENT, 0}, {B, Hq, D});
  Tensor tkc = Tensor::Contiguous(mem_kc, vt::DType::kF32,
                                  Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, D});
  Tensor tvc = Tensor::Contiguous(mem_vc, vt::DType::kF32,
                                  Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, D});
  Tensor tbt = Tensor::Contiguous(mem_bt, vt::DType::kI32,
                                  Device{DeviceType::kTENSTORRENT, 0}, {B, NBlocksPerReq});
  Tensor tsl =
      Tensor::Contiguous(mem_sl, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {B});
  Tensor tqsl = Tensor::Contiguous(mem_qsl, vt::DType::kI32,
                                   Device{DeviceType::kTENSTORRENT, 0}, {B + 1});

  vt::PagedAttentionArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(D));
  args.causal = true;
  auto pa = reinterpret_cast<vt::PagedAttentionFn>(
      vt::GetOp(vt::OpId::kPagedAttention, DeviceType::kTENSTORRENT));
  pa(q, tout, tq, tkc, tvc, tbt, tsl, tqsl, args);

  // THE RED SIGNAL: the device decode path commits its flattened [B, H*D]
  // result into out's slot; the host fallback (the crash's silent landing
  // zone) commits host bytes and drops the device shadow.
  REQUIRE(vt::tenstorrent::DeviceShadowExact(tout, static_cast<uint32_t>(B),
                                             static_cast<uint32_t>(Hq * D)));

  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_q);
  backend.Free(mem_out);
  backend.Free(mem_kc);
  backend.Free(mem_vc);
  backend.Free(mem_bt);
  backend.Free(mem_sl);
  backend.Free(mem_qsl);

  // Host NHD oracle per request (same envelope as the B=1 case above).
  const int64_t qpk = Hq / Hkv;
  float max_abs = 0.0f;
  for (int64_t r = 0; r < B; ++r) {
    for (int64_t h = 0; h < Hq; ++h) {
      const int64_t g = h / qpk;
      const int64_t qoff = (r * Hq + h) * D;
      float m = -std::numeric_limits<float>::infinity();
      std::vector<float> scores(static_cast<size_t>(Seq));
      for (int64_t j = 0; j < Seq; ++j) {
        float dot = 0.0f;
        const int64_t kbase =
            (r * NBlocksPerReq + j / Bsz) * (Bsz * Page) + (j % Bsz) * Page + g * D;
        for (int64_t e = 0; e < D; ++e)
          dot += host_q[static_cast<size_t>(qoff + e)] *
                 host_kc[static_cast<size_t>(kbase + e)];
        scores[static_cast<size_t>(j)] = dot * args.scale;
        m = std::max(m, scores[static_cast<size_t>(j)]);
      }
      float denom = 0.0f;
      for (int64_t j = 0; j < Seq; ++j) {
        scores[static_cast<size_t>(j)] = std::exp(scores[static_cast<size_t>(j)] - m);
        denom += scores[static_cast<size_t>(j)];
      }
      const float inv = 1.0f / denom;
      for (int64_t e = 0; e < D; ++e) {
        float acc = 0.0f;
        for (int64_t j = 0; j < Seq; ++j)
          acc += scores[static_cast<size_t>(j)] * inv *
                 host_vc[static_cast<size_t>(
                     (r * NBlocksPerReq + j / Bsz) * (Bsz * Page) +
                     (j % Bsz) * Page + g * D + e)];
        max_abs = std::max(max_abs, std::fabs(host_out[static_cast<size_t>(qoff + e)] - acc));
      }
    }
  }
  CHECK(max_abs < 0.5f);
}

// Multi-token pure prefill with TILE-legal geometry exercises
// ttnn::chunked_scaled_dot_product_attention (or host fallback).
TEST_CASE("kTENSTORRENT kPagedAttention pure-prefill matches host within BF16 envelope") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kPagedAttention, DeviceType::kTENSTORRENT));

  // T=32 query tokens, full prefill (seq=T), D=128, block=32 → one chunked SDPA call.
  constexpr int64_t T = 32, Hq = 4, Hkv = 2, D = 128, Bsz = 32;
  constexpr int64_t Seq = T;
  constexpr int64_t NBlocks = (Seq + Bsz - 1) / Bsz;  // 1
  constexpr int64_t Page = Hkv * D;
  const size_t cache_elems = static_cast<size_t>(NBlocks * Bsz * Page);
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_q(static_cast<size_t>(T * Hq * D));
  std::vector<float> host_kc(cache_elems), host_vc(cache_elems);
  for (size_t i = 0; i < host_q.size(); ++i)
    host_q[i] = (static_cast<float>(i % 17) - 8.0f) * 0.05f;
  // Dense NHD cache: position j lives at block j/Bsz, offset j%Bsz.
  for (int64_t j = 0; j < Seq; ++j) {
    const int64_t blk = j / Bsz, off = j % Bsz;
    for (int64_t g = 0; g < Hkv; ++g) {
      for (int64_t e = 0; e < D; ++e) {
        const size_t idx =
            static_cast<size_t>(((blk * Bsz + off) * Hkv + g) * D + e);
        host_kc[idx] = (static_cast<float>((j * 3 + g * 5 + e) % 13) - 6.0f) * 0.03f;
        host_vc[idx] = (static_cast<float>((j * 7 + g * 2 + e) % 11) - 5.0f) * 0.02f;
      }
    }
  }
  std::vector<int32_t> block_table(static_cast<size_t>(NBlocks));
  for (int64_t i = 0; i < NBlocks; ++i) block_table[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  std::vector<int32_t> seq_lens{static_cast<int32_t>(Seq)};
  std::vector<int32_t> qsl{0, static_cast<int32_t>(T)};
  std::vector<float> host_out(static_cast<size_t>(T * Hq * D), 0.0f);

  void* mem_q = backend.Alloc(host_q.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  void* mem_kc = backend.Alloc(host_kc.size() * sizeof(float));
  void* mem_vc = backend.Alloc(host_vc.size() * sizeof(float));
  void* mem_bt = backend.Alloc(block_table.size() * sizeof(int32_t));
  void* mem_sl = backend.Alloc(seq_lens.size() * sizeof(int32_t));
  void* mem_qsl = backend.Alloc(qsl.size() * sizeof(int32_t));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_q, host_q.data(), host_q.size() * sizeof(float));
  backend.Copy(q, mem_kc, host_kc.data(), host_kc.size() * sizeof(float));
  backend.Copy(q, mem_vc, host_vc.data(), host_vc.size() * sizeof(float));
  backend.Copy(q, mem_bt, block_table.data(), block_table.size() * sizeof(int32_t));
  backend.Copy(q, mem_sl, seq_lens.data(), seq_lens.size() * sizeof(int32_t));
  backend.Copy(q, mem_qsl, qsl.data(), qsl.size() * sizeof(int32_t));

  Tensor tq =
      Tensor::Contiguous(mem_q, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, Hq, D});
  Tensor tout = Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                   {T, Hq, D});
  Tensor tkc = Tensor::Contiguous(mem_kc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, D});
  Tensor tvc = Tensor::Contiguous(mem_vc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, D});
  Tensor tbt = Tensor::Contiguous(mem_bt, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0},
                                  {1, NBlocks});
  Tensor tsl =
      Tensor::Contiguous(mem_sl, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {1});
  Tensor tqsl =
      Tensor::Contiguous(mem_qsl, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {2});

  vt::PagedAttentionArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(D));
  args.causal = true;
  auto pa = reinterpret_cast<vt::PagedAttentionFn>(
      vt::GetOp(vt::OpId::kPagedAttention, DeviceType::kTENSTORRENT));
  pa(q, tout, tq, tkc, tvc, tbt, tsl, tqsl, args);

  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_q);
  backend.Free(mem_out);
  backend.Free(mem_kc);
  backend.Free(mem_vc);
  backend.Free(mem_bt);
  backend.Free(mem_sl);
  backend.Free(mem_qsl);

  // Host causal GQA oracle over the dense NHD cache.
  const int64_t qpk = Hq / Hkv;
  float max_abs = 0.0f;
  for (int64_t t = 0; t < T; ++t) {
    const int64_t p = t;  // pure prefill: query pos == token index
    for (int64_t h = 0; h < Hq; ++h) {
      const int64_t g = h / qpk;
      const int64_t qoff = (t * Hq + h) * D;
      float m = -std::numeric_limits<float>::infinity();
      std::vector<float> scores(static_cast<size_t>(p + 1));
      for (int64_t j = 0; j <= p; ++j) {
        float dot = 0.0f;
        const int64_t blk = j / Bsz, off = j % Bsz;
        const int64_t kbase = ((blk * Bsz + off) * Hkv + g) * D;
        for (int64_t e = 0; e < D; ++e)
          dot += host_q[static_cast<size_t>(qoff + e)] * host_kc[static_cast<size_t>(kbase + e)];
        scores[static_cast<size_t>(j)] = dot * args.scale;
        m = std::max(m, scores[static_cast<size_t>(j)]);
      }
      float denom = 0.0f;
      for (int64_t j = 0; j <= p; ++j) {
        scores[static_cast<size_t>(j)] = std::exp(scores[static_cast<size_t>(j)] - m);
        denom += scores[static_cast<size_t>(j)];
      }
      const float inv = 1.0f / denom;
      for (int64_t e = 0; e < D; ++e) {
        float acc = 0.0f;
        for (int64_t j = 0; j <= p; ++j) {
          const int64_t blk = j / Bsz, off = j % Bsz;
          const int64_t vbase = ((blk * Bsz + off) * Hkv + g) * D + e;
          acc += scores[static_cast<size_t>(j)] * inv * host_vc[static_cast<size_t>(vbase)];
        }
        max_abs = std::max(max_abs, std::fabs(host_out[static_cast<size_t>(qoff + e)] - acc));
      }
    }
  }
  MESSAGE("pure-prefill max_abs vs host oracle: ", max_abs);
  CHECK(max_abs < 0.5f);
}

// ttnn mesh-trace capture: warm a matmul, capture it, replay, check BF16 envelope.
// Program cache must be warm before BeginCapture (ttnn trace contract).
TEST_CASE("kTENSTORRENT SupportsGraphCapture and matmul capture/replay") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  REQUIRE(backend.SupportsGraphCapture());

  constexpr int64_t M = 32, K = 64, N = 32;
  std::vector<float> ha(static_cast<size_t>(M * K), 0.1f);
  std::vector<float> hb(static_cast<size_t>(K * N), 0.2f);
  std::vector<float> hc(static_cast<size_t>(M * N), 0.0f);
  for (size_t i = 0; i < ha.size(); ++i) ha[i] = static_cast<float>((i % 7) - 3) * 0.05f;
  for (size_t i = 0; i < hb.size(); ++i) hb[i] = static_cast<float>((i % 5) - 2) * 0.04f;

  void* ma = backend.Alloc(ha.size() * sizeof(float));
  void* mb = backend.Alloc(hb.size() * sizeof(float));
  void* mc = backend.Alloc(hc.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, ma, ha.data(), ha.size() * sizeof(float));
  backend.Copy(q, mb, hb.data(), hb.size() * sizeof(float));

  Tensor ta = Tensor::Contiguous(ma, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, K});
  Tensor tb = Tensor::Contiguous(mb, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {K, N});
  Tensor tc = Tensor::Contiguous(mc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, N});

  auto mm = reinterpret_cast<vt::MatmulFn>(vt::GetOp(vt::OpId::kMatmul, DeviceType::kTENSTORRENT));
  // Warm program cache (required before capture).
  mm(q, tc, ta, tb);
  backend.Copy(q, hc.data(), mc, hc.size() * sizeof(float));

  // Capture the same matmul on the already-resident shadows.
  backend.BeginCapture(q);
  mm(q, tc, ta, tb);
  backend.EndCapture(q);

  // Replay several times — should not throw.
  for (int i = 0; i < 3; ++i) backend.Replay(q);

  std::vector<float> after(hc.size(), 0.0f);
  backend.Copy(q, after.data(), mc, after.size() * sizeof(float));
  float max_abs = 0.0f;
  for (size_t i = 0; i < after.size(); ++i)
    max_abs = std::max(max_abs, std::fabs(after[i] - hc[i]));
  MESSAGE("trace replay max_abs vs warm: ", max_abs);
  CHECK(max_abs < 1e-3f);

  // Multi-graph handle API: capture again into an owned handle.
  backend.BeginCapture(q);
  mm(q, tc, ta, tb);
  void* graph = backend.EndCaptureGraph(q);
  REQUIRE(graph != nullptr);
  backend.ReplayGraph(q, graph);
  backend.DestroyGraph(graph);

  backend.Free(ma);
  backend.Free(mb);
  backend.Free(mc);
}

// ISSUE-LOCAL-01M3JXEFQKSZP23PP2HWY9G0VQ: EnsureDevice2D's exact-rows/cols
// arm on a rank-3-committed slot ran a bare ttnn::reshape on the TILED shadow
// during capture — a program the eager pass never warmed (its row-major
// reshape is a free view) — so the 27B decode capture created
// ReshapeViewTiledProgramFactory's program mid-trace and died on its
// to_device write ("Writes are not supported during trace capture"). The fix
// runs one chain in both passes; this case fails red if the capture pass
// diverges again: with the old branch restored, BeginCapture fatals.
TEST_CASE("kTENSTORRENT EnsureDevice2D rank-3 reshape is capture-safe (one chain both passes)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  REQUIRE(backend.SupportsGraphCapture());

  // [2, 48, 32] device result, flat 2D geometry [96, 32]: reshaping the tiled
  // rank-3 shadow to 2D is NOT a metadata view (the 48 second-last dim is not
  // tile-aligned — reshape.cpp's this_is_view), so the divergent capture arm
  // had to create ReshapeViewTiledProgramFactory's program mid-capture.
  constexpr uint32_t B = 2, H = 48, D = 32;
  constexpr uint32_t Rows = B * H, Cols = D;
  std::vector<float> host(static_cast<size_t>(Rows * Cols), 0.25f);

  void* mem = backend.Alloc(host.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem, host.data(), host.size() * sizeof(float));
  Tensor t = Tensor::Contiguous(mem, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                {Rows, Cols});

  // Commit the rank-3 shadow under the 2D slot record, warm the eager chain,
  // re-commit (the warm consumed the rank-3 logical shape), then capture.
  vt::tenstorrent::CommitRank3DeviceLogicalForTest(t, B, H, D);
  vt::tenstorrent::EnsureDevice2DForTest(t);
  vt::tenstorrent::CommitRank3DeviceLogicalForTest(t, B, H, D);

  backend.BeginCapture(q);
  vt::tenstorrent::EnsureDevice2DForTest(t);
  backend.EndCapture(q);
  MESSAGE("shadow exact after capture: ", vt::tenstorrent::DeviceShadowExact(t, Rows, Cols));
  backend.Replay(q);
  backend.Replay(q);

  std::vector<float> after(host.size(), 0.0f);
  backend.Copy(q, after.data(), mem, after.size() * sizeof(float));
  float max_abs = 0.0f;
  for (size_t i = 0; i < after.size(); ++i)
    max_abs = std::max(max_abs, std::fabs(after[i] - host[i]));
  MESSAGE("replay max_abs vs staged: ", max_abs);
  // Diagnostic: a fresh eager EnsureDevice2D must serve the replayed bytes.
  vt::tenstorrent::EnsureDevice2DForTest(t);
  std::vector<float> after2(host.size(), 0.0f);
  backend.Copy(q, after2.data(), mem, after2.size() * sizeof(float));
  float max_abs2 = 0.0f;
  for (size_t i = 0; i < after2.size(); ++i)
    max_abs2 = std::max(max_abs2, std::fabs(after2[i] - host[i]));
  MESSAGE("post-eager max_abs vs staged: ", max_abs2);
  CHECK(max_abs2 < 1e-3f);

  backend.Free(mem);
}

// ISSUE-LOCAL-01M3JXEFQKSZP23PP2HWY9G0VQ, site 2: MemsetDeviceIfCapture's
// fresh-slot lane installed a [1, cols] bf16 shadow ONLY under capture; the
// eager pass primed the zero and kept the host fallback, so the slot ended
// each pass in a different state. The 27B bench: DBuf::Zero of the 20480-B
// residual, then kRmsNorm's EnsureDevice2D at [2, 5120] — the capture step
// hit the same-numel arm with a reshape spec the eager pass never ran, and
// ReshapeViewTiledProgramFactory created its program mid-trace and died on
// its to_device write. The fix installs the same [1, cols] shadow in BOTH
// passes, so the eager consumer warms the reshape and the capture replays it
// as a cache hit. This case fails red if the eager lane diverges again: with
// the capture-only install restored, the EnsureDevice2D inside capture
// creates the program and TT_FATALs ("Writes are not supported during trace
// capture").
TEST_CASE("kTENSTORRENT fresh-slot Memset installs the same shadow in both passes") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  REQUIRE(backend.SupportsGraphCapture());

  // The production geometry: 20480 B = 10240 bf16 elems, consumer [2, 5120]
  // — the same-numel arm's [1, 10240] -> [2, 5120] reshape is not a metadata
  // view (different tile counts), so it needs the warmed program exactly as
  // the bench ran it (bf16 tensor: the shadow's element count is bytes/2,
  // which must equal the consumer's numel for the same-numel arm to fire).
  constexpr uint32_t Rows = 2, Cols = 5120;
  void* mem = backend.Alloc(Rows * Cols * sizeof(uint16_t));
  // The pool recycles blocks across test cases: acquire the block so both
  // passes start from the same W7 fresh-slot state (no stale shadow from a
  // previous tenant can route the eager memset down a different lane).
  backend.OnScratchBlockAcquired(mem);
  Queue q = backend.CreateQueue();

  // The case needs the default host-free decode lane for BOTH passes; own the
  // env explicitly (earlier cases legitimately run under the opt-out).
  const bool had_hf = std::getenv("VT_TT_HOST_FREE_DECODE") != nullptr;
  const std::string saved_hf =
      had_hf ? std::string(std::getenv("VT_TT_HOST_FREE_DECODE")) : std::string();
  struct RestoreHf {
    bool had;
    std::string saved;
    ~RestoreHf() {
      if (had) ::setenv("VT_TT_HOST_FREE_DECODE", saved.c_str(), 1);
      else ::unsetenv("VT_TT_HOST_FREE_DECODE");
    }
  } restore_hf{had_hf, saved_hf};
  ::unsetenv("VT_TT_HOST_FREE_DECODE");

  MESSAGE("capture flag at entry: ", vt::tenstorrent::TraceCaptureActive());

  // Warmup step: fresh slot, DBuf::Zero, then the consumer's stage. With the
  // fix, the Memset installs the [1, 5120] shadow and this EnsureDevice2D
  // runs (and warms) the same-numel reshape in the eager pass.
  backend.Memset(q, mem, 0, Rows * Cols * sizeof(uint16_t));
  Tensor t = Tensor::Contiguous(mem, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0},
                                {Rows, Cols});
  vt::tenstorrent::EnsureDevice2DForTest(t);

  // The production capture step saw this buffer as a FRESH slot (the pool
  // handed the block to a new tensor between steps, W7 semantics). Reproduce
  // the acquisition so the capture memset takes the fresh-slot lane exactly
  // like the bench's trace did.
  backend.OnScratchBlockAcquired(mem);
  // Capture step: the production zero-fill runs INSIDE the captured region
  // (the bench trace's "device zero-fill (fresh slot ...)" fires after
  // BeginCapture), so begin capture first, then repeat the memset on the
  // recycled slot and the consumer's stage.
  backend.BeginCapture(q);
  backend.Memset(q, mem, 0, Rows * Cols * sizeof(uint16_t));
  vt::tenstorrent::EnsureDevice2DForTest(t);
  backend.EndCapture(q);
  backend.Replay(q);

  // The replayed region holds the zeros the memsets wrote.
  std::vector<uint16_t> after(Rows * Cols, 0x3f80);  // f32 1.0 bits in the low half: garbage if read
  backend.Copy(q, after.data(), mem, after.size() * sizeof(uint16_t));
  float max_abs = 0.0f;
  for (uint16_t v : after) max_abs = std::max(max_abs, std::fabs(static_cast<float>(v)));
  MESSAGE("post-replay nonzero bf16 elems: ", max_abs);
  CHECK(max_abs == 0.0f);

  backend.Free(mem);
}

// BACKEND-TENSTORRENT-RESIDUAL-GOLDEN: op-level numerics probe at the
// kDeviceResidualMinRows == 32 boundary. The device path (rows >= 32,
// non-gemma) does ttnn::add + ttnn::rms_norm in bf16; the host/CPU path
// (cpu_ops.cpp:371) accumulates the variance in f32. This measures the
// divergence across the boundary both ways so the accept/raise/force-f32
// decision is grounded in a real number, not a prior. CPU is the oracle.
TEST_CASE("kTENSTORRENT kRmsNorm residual: device vs CPU f32 oracle across the rows=32 boundary") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& tt = *vt::TryGetBackend(DeviceType::kTENSTORRENT);
  Backend& cpu = *vt::TryGetBackend(DeviceType::kCPU);
  REQUIRE(&cpu != nullptr);

  // Deterministic inputs: a simple LCG, independent of platform RNG.
  auto lcg = [](uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return (s >> 8) * (1.0f / 16777216.0f) - 0.5f;  // [-0.5, 0.5)
  };

  // Qwen3-0.6B hidden width is 1024; use it so the reduction length is
  // realistic (the bf16 variance accumulation is length-sensitive).
  constexpr int64_t D = 1024;
  const vt::RmsNormArgs args{1e-6f, /*gemma=*/false};

  std::vector<float> w(D);
  {
    uint32_t s = 999;
    for (int64_t j = 0; j < D; ++j) w[j] = 0.8f + 0.4f * lcg(s);  // [0.6, 1.0)
  }

  // rows that span the boundary both ways: below, at, just above, and larger.
  const std::vector<int64_t> rows_cases = {1, 31, 32, 33, 64, 128};

  for (int64_t rows : rows_cases) {
    std::vector<float> x(static_cast<size_t>(rows * D));
    std::vector<float> res(static_cast<size_t>(rows * D));
    {
      uint32_t sx = 12345, sr = 54321;
      for (size_t i = 0; i < x.size(); ++i) {
        x[i] = 2.0f * lcg(sx);    // [-1, 1)
        res[i] = 2.0f * lcg(sr);  // [-1, 1)
      }
    }

    auto run = [&](Backend& b, DeviceType dt, std::vector<float>& out) -> void {
      void* mx = b.Alloc(x.size() * sizeof(float));
      void* mw = b.Alloc(w.size() * sizeof(float));
      void* mr = b.Alloc(res.size() * sizeof(float));
      void* mo = b.Alloc(out.size() * sizeof(float));
      Queue q = b.CreateQueue();
      b.Copy(q, mx, x.data(), x.size() * sizeof(float));
      b.Copy(q, mw, w.data(), w.size() * sizeof(float));
      b.Copy(q, mr, res.data(), res.size() * sizeof(float));
      Tensor tx = Tensor::Contiguous(mx, vt::DType::kF32, Device{dt, 0}, {rows, D});
      Tensor tw = Tensor::Contiguous(mw, vt::DType::kF32, Device{dt, 0}, {D});
      Tensor tr = Tensor::Contiguous(mr, vt::DType::kF32, Device{dt, 0}, {rows, D});
      Tensor to = Tensor::Contiguous(mo, vt::DType::kF32, Device{dt, 0}, {rows, D});
      // Residual is passed (&tr) so the fused add->rms path
      // (tenstorrent_ops.cpp ttnn::add + ttnn::rms_norm for rows>=32) is
      // exercised on device — NOT plain rms. Dropping &tr would silently
      // skip the residual merge this probe exists to measure.
      vt::RmsNorm(q, to, tx, tw, args, &tr);
      b.Copy(q, out.data(), mo, out.size() * sizeof(float));
      b.Free(mx); b.Free(mw); b.Free(mr); b.Free(mo);
    };

    std::vector<float> out_cpu(x.size()), out_tt(x.size());
    run(cpu, DeviceType::kCPU, out_cpu);
    run(tt, DeviceType::kTENSTORRENT, out_tt);

    float max_abs = 0.0f, max_rel = 0.0f;
    for (size_t i = 0; i < out_cpu.size(); ++i) {
      float d = std::fabs(out_tt[i] - out_cpu[i]);
      if (d > max_abs) max_abs = d;
      float denom = std::fabs(out_cpu[i]);
      if (denom > 1e-3f) {
        float r = d / denom;
        if (r > max_rel) max_rel = r;
      }
    }
    const bool device_path = (rows >= 32);  // kDeviceResidualMinRows
    MESSAGE("rows=", rows, " (device_path=", device_path,
            "): max_abs=", max_abs, " max_rel=", max_rel);
    // Loose envelope: the device bf16 path must stay in bf16 territory. This
    // is NOT the parity verdict — it is the non-vacuous RED hook. A diverging
    // run (e.g. NaN, or >5%) trips it; the real accept/raise decision is
    // recorded from the measured band, not asserted here.
    CHECK(std::isfinite(max_abs));
    CHECK(max_abs < 0.05f);
  }
}

// BACKEND-TENSTORRENT-HOST-FREE-R1: guard the env-gated host-free helpers'
// DEFAULT-PATH INERTNESS. The helpers (CopyDeviceDeviceIfCapture /
// MemsetDeviceIfCapture, vt/tenstorrent/tenstorrent_device.h) must DECLINE
// unless VT_TT_HOST_FREE_DECODE is set (or capture is active). Without this
// case that property is enforced by code review alone: a removed gate flips
// ordinary eager Copy/Memset to device variants silently (review mutation M1)
// and a capture flag stuck true after a failed EndCapture does the same (M4).
// Both buffers below carry CURRENT device shadows with equal byte sizes, so
// the flag gate is the ONLY thing that can make the helpers decline.
#include "../../src/vt/tenstorrent/tenstorrent_device.h"

TEST_CASE("kTENSTORRENT host-free helpers decline by default (inertness guard)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  ::setenv("VT_TT_HOST_FREE_DECODE", "0", 1);  // opt-out path; the guard is about the OPT-OUT case
  Backend& backend = *vt::TryGetBackend(DeviceType::kTENSTORRENT);

  // Two same-shaped outputs, each given a current device shadow by a device
  // Matmul (CommitDevice2D leaves device_current=true, host_current=false).
  constexpr int64_t M = 8, K = 32, N = 8;
  auto shadowed = [&](std::vector<float>& host) {
    std::vector<float> a(M * K, 0.5f), b(K * N, 0.25f);
    host.assign(static_cast<size_t>(M * N), -1.0f);
    void* ma = backend.Alloc(a.size() * sizeof(float));
    void* mb = backend.Alloc(b.size() * sizeof(float));
    void* mo = backend.Alloc(host.size() * sizeof(float));
    Queue q = backend.CreateQueue();
    backend.Copy(q, ma, a.data(), a.size() * sizeof(float));
    backend.Copy(q, mb, b.data(), b.size() * sizeof(float));
    Tensor ta = Tensor::Contiguous(ma, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, K});
    Tensor tb = Tensor::Contiguous(mb, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {K, N});
    Tensor to = Tensor::Contiguous(mo, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, N});
    reinterpret_cast<vt::MatmulFn>(vt::GetOp(vt::OpId::kMatmul, DeviceType::kTENSTORRENT))(q, to, ta, tb);
    return mo;  // caller keeps the allocation; shadow lives in the slot map
  };
  std::vector<float> h1, h2;
  void* m1 = shadowed(h1);
  void* m2 = shadowed(h2);

  // The gate: same bytes, both shadows current -> only the env/capture gate
  // can decline. These CHECKs go RED if the gate is removed (M1) or if the
  // capture flag is stuck true (M4).
  CHECK_FALSE(vt::tenstorrent::CopyDeviceDeviceIfCapture(m2, m1, h1.size() * sizeof(float)));
  CHECK_FALSE(vt::tenstorrent::MemsetDeviceIfCapture(m2, 0, h2.size() * sizeof(float)));
  // value!=0 always declines (host memset is the only path for it).
  CHECK_FALSE(vt::tenstorrent::MemsetDeviceIfCapture(m2, 1, h2.size() * sizeof(float)));

  // And the default host path still works: Copy m1 -> m2 yields identical
  // host bytes once materialized.
  Queue q = backend.CreateQueue();
  std::vector<float> got(h1.size(), -7.0f);
  backend.Copy(q, m2, m1, h1.size() * sizeof(float));
  backend.Copy(q, got.data(), m2, got.size() * sizeof(float));
  // 0.5f * 0.25f summed over K=32 == 4.0f per element (bf16 device acc).
  CHECK(got == std::vector<float>(static_cast<size_t>(M * N), 4.0f));

  backend.Free(m1);
  backend.Free(m2);
}

// ==== BACKEND-TENSTORRENT-GDN W1: the GDN prefill op set vs the CPU f32 oracle
// The CPU arm (cpu_ops.cpp GdnPrefillKernel & friends) is the correctness
// oracle (spec "Upstream chain" #1): identical random inputs, both arms run
// the SAME public vt:: op on their own backend, outputs compared. Every case
// calls through the vt:: facade, so a missing TT kernel REFUSES BY NAME —
// that refusal is this suite's red state before the kernels land.
namespace {

// Deterministic LCG, platform-RNG independent (same doctrine as the
// residual-golden probe above).
float GdnLcg(uint32_t& s) {
  s = s * 1664525u + 1013904223u;
  return (s >> 8) * (1.0f / 16777216.0f) - 0.5f;  // [-0.5, 0.5)
}

struct GdnDiffStats {
  float max_abs = 0.0f;
  float max_rel = 0.0f;  // over |ref| > 1e-3
  bool within = true;    // every element satisfied the envelope
};

// Elementwise envelope: |got - ref| <= rel*|ref| + abs_floor. Returns the
// worst-case stats either way so a MESSAGE can carry the per-T table.
// NaN/Inf-SAFE (W2 fold-in from the W1 fresh review, MEDIUM finding): the old
// `a > envelope` predicate is false when `a` is NaN, so a NaN `got[i]` passed;
// the negated form `!(a <= envelope)` fails on NaN, and any non-finite `got`
// fails outright. std::max(d.max_abs, a) returns d.max_abs for NaN `a`
// ((d.max_abs < NaN) is false), so the stats stay readable.
GdnDiffStats CompareVsOracle(const std::vector<float>& got, const std::vector<float>& ref,
                             float rel, float abs_floor) {
  GdnDiffStats d;
  for (size_t i = 0; i < ref.size(); ++i) {
    const float a = std::fabs(got[i] - ref[i]);
    if (!std::isfinite(got[i])) d.within = false;
    if (!(a <= rel * std::fabs(ref[i]) + abs_floor)) d.within = false;
    d.max_abs = std::max(d.max_abs, a);
    if (std::fabs(ref[i]) > 1e-3f) d.max_rel = std::max(d.max_rel, a / std::fabs(ref[i]));
  }
  return d;
}

}  // namespace

TEST_CASE("kTENSTORRENT kL2Norm matches the CPU f32 oracle (GDN q/k row shape)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& cpu = *vt::TryGetBackend(DeviceType::kCPU);
  constexpr int64_t D = 128;  // Qwen3.8 head k dim
  const vt::L2NormArgs args{1e-6f};

  for (int64_t T : {int64_t{3}, int64_t{64}, int64_t{65}, int64_t{200}}) {
    for (int64_t H : {int64_t{2}, int64_t{8}}) {
      const int64_t rows = T * H;
      std::vector<float> x(static_cast<size_t>(rows * D));
      {
        uint32_t s = 1000u + static_cast<uint32_t>(T * 31 + H);
        for (float& v : x) v = 2.0f * GdnLcg(s);
      }
      std::vector<float> out_cpu(x.size(), 0.0f), out_tt(x.size(), 0.0f);
      auto run = [&](Backend& b, DeviceType dt, std::vector<float>& out) {
        void* mx = b.Alloc(x.size() * sizeof(float));
        void* mo = b.Alloc(out.size() * sizeof(float));
        Queue q = b.CreateQueue();
        b.Copy(q, mx, x.data(), x.size() * sizeof(float));
        Tensor tx = Tensor::Contiguous(mx, vt::DType::kF32, Device{dt, 0}, {T, H, D});
        Tensor to = Tensor::Contiguous(mo, vt::DType::kF32, Device{dt, 0}, {T, H, D});
        vt::L2Norm(q, to, tx, args);
        b.Copy(q, out.data(), mo, out.size() * sizeof(float));
        b.Free(mx);
        b.Free(mo);
      };
      run(cpu, DeviceType::kCPU, out_cpu);
      run(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, out_tt);

      // bf16 tile path: per-T envelope calibrated on the P150 (see the spec's
      // Evidence table; values stated per T, not global).
      const float tol = 0.02f;
      GdnDiffStats d = CompareVsOracle(out_tt, out_cpu, /*rel=*/0.0f, tol);
      MESSAGE("kL2Norm T=", T, " H=", H, " rows=", rows,
              ": max_abs=", d.max_abs, " max_rel=", d.max_rel, " tol=", tol);
      CHECK(std::isfinite(d.max_abs));
      CHECK(d.within);
    }
  }
}

TEST_CASE("kTENSTORRENT kRmsNormGated matches the CPU f32 oracle (silu + sigmoid, padded gate stride)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& cpu = *vt::TryGetBackend(DeviceType::kCPU);
  constexpr int64_t D = 128;  // Qwen3.8 value head dim
  for (int64_t T : {int64_t{3}, int64_t{64}, int64_t{65}, int64_t{200}}) {
    for (int64_t Hv : {int64_t{2}, int64_t{8}}) {
      for (int sigmoid_gate : {0, 1}) {
        vt::RmsNormGatedArgs args;
        args.eps = 1e-6f;
        args.sigmoid_gate = sigmoid_gate != 0;
        const int64_t rows = T * Hv;
        std::vector<float> x(static_cast<size_t>(rows * D));
        std::vector<float> w(static_cast<size_t>(D));
        // Gate as a PADDED-row rank-3 view (the merged qkvz z-slice layout the
        // op contract admits): token stride Hv*D + 8 with garbage in the pad.
        const int64_t gate_row = Hv * D, gate_pad = 8;
        std::vector<float> gate(static_cast<size_t>(T * (gate_row + gate_pad) + gate_row), 0.0f);
        {
          uint32_t sx = 7000u + static_cast<uint32_t>(T * 37 + Hv * 3 + sigmoid_gate);
          uint32_t sw = 777u;
          for (float& v : x) v = 2.0f * GdnLcg(sx);
          for (float& v : w) v = 0.8f + 0.4f * (GdnLcg(sw) + 0.5f);
          for (int64_t t = 0; t < T; ++t)
            for (int64_t e = 0; e < gate_row; ++e)
              gate[static_cast<size_t>(t * (gate_row + gate_pad) + e)] = 2.0f * GdnLcg(sx);
        }
        std::vector<float> out_cpu(x.size(), 0.0f), out_tt(x.size(), 0.0f);
        auto run = [&](Backend& b, DeviceType dt, std::vector<float>& out) {
          void* mx = b.Alloc(x.size() * sizeof(float));
          void* mg = b.Alloc(gate.size() * sizeof(float));
          void* mw = b.Alloc(w.size() * sizeof(float));
          void* mo = b.Alloc(out.size() * sizeof(float));
          Queue q = b.CreateQueue();
          b.Copy(q, mx, x.data(), x.size() * sizeof(float));
          b.Copy(q, mg, gate.data(), gate.size() * sizeof(float));
          b.Copy(q, mw, w.data(), w.size() * sizeof(float));
          Tensor tx = Tensor::Contiguous(mx, vt::DType::kF32, Device{dt, 0}, {T, Hv, D});
          Tensor tw = Tensor::Contiguous(mw, vt::DType::kF32, Device{dt, 0}, {D});
          Tensor to = Tensor::Contiguous(mo, vt::DType::kF32, Device{dt, 0}, {T, Hv, D});
          Tensor tg{};
          tg.data = mg;
          tg.dtype = vt::DType::kF32;
          tg.device = Device{dt, 0};
          tg.rank = 3;
          tg.shape[0] = T;
          tg.shape[1] = Hv;
          tg.shape[2] = D;
          tg.stride[0] = gate_row + gate_pad;
          tg.stride[1] = D;
          tg.stride[2] = 1;
          vt::RmsNormGated(q, to, tx, tg, tw, args);
          b.Copy(q, out.data(), mo, out.size() * sizeof(float));
          b.Free(mx);
          b.Free(mg);
          b.Free(mw);
          b.Free(mo);
        };
        run(cpu, DeviceType::kCPU, out_cpu);
        run(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, out_tt);

        // bf16 tile path (rms_norm + act eltwise): row-wise op, so the
        // envelope is flat in T (no recurrence to amplify). Measured on the
        // P150 over this sweep: max_abs 0.0099-0.0263, max_rel <= 0.0267;
        // 0.035 keeps ~33% headroom under the RESIDUAL-GOLDEN 0.0459 anchor
        // (spec numerics doctrine).
        const float tol = 0.035f;
        GdnDiffStats d = CompareVsOracle(out_tt, out_cpu, /*rel=*/0.0f, tol);
        MESSAGE("kRmsNormGated T=", T, " Hv=", Hv, " sigmoid=", args.sigmoid_gate,
                " rows=", rows, ": max_abs=", d.max_abs, " max_rel=", d.max_rel,
                " tol=", tol);
        CHECK(std::isfinite(d.max_abs));
        CHECK(d.within);
      }
    }
  }
}

// Batched-prefill gated-norm (ISSUE-LOCAL-01M2XSWCP507W0M18EXHGR5TGS): the
// gate rides a committed device shadow whose native geometry is [T, W] (the
// z projection output, [256, 6144]) while x flattens to [T*Hv, D] = [12288,
// 128]. NormalizeDevF32Tile keeps the rank-2 native geometry, so the kernel
// must bridge [256,6144] -> [12288,128] (same numel AND same tile count) —
// the free TILE reshape at the M=2 prefill shape fatals 'MeshBuffer must be
// large enough' and the poisoned view OOMs the step downstream.
TEST_CASE("kTENSTORRENT kRmsNormGated serves a [256,6144] committed gate "
          "shadow at the batched-prefill [12288,128] geometry") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kRmsNormGated, DeviceType::kTENSTORRENT));
  constexpr int64_t T = 256, Hv = 48, D = 128;
  const Device dev{DeviceType::kTENSTORRENT, 0};
  Backend& tt = vt::GetBackend(DeviceType::kTENSTORRENT);
  Backend& cpu = *vt::TryGetBackend(DeviceType::kCPU);

  vt::RmsNormGatedArgs args;
  args.eps = 1e-6f;
  args.sigmoid_gate = 1;

  // Producer: a device Matmul commits the f32 [T, Hv*D] z-projection shadow.
  // b is all ones so the shadow values are the (host-known) row sums of a.
  std::vector<float> ha(static_cast<size_t>(T * 16));
  for (size_t i = 0; i < ha.size(); ++i)
    ha[i] = static_cast<float>(static_cast<int>(i % 15) - 7) * 0.25f;
  std::vector<float> gate_vals(static_cast<size_t>(T), 0.0f);
  for (int64_t t = 0; t < T; ++t) {
    float s = 0.0f;
    for (int64_t k = 0; k < 16; ++k)
      s += ha[static_cast<size_t>(t * 16 + k)];
    gate_vals[static_cast<size_t>(t)] = s;
  }
  void* ma = tt.Alloc(ha.size() * sizeof(float));
  void* mb = tt.Alloc(16 * Hv * D * sizeof(float));
  void* mo = tt.Alloc(T * Hv * D * sizeof(float));
  Queue q = tt.CreateQueue();
  tt.Copy(q, ma, ha.data(), ha.size() * sizeof(float));
  std::vector<float> ones(static_cast<size_t>(16 * Hv * D), 1.0f);
  tt.Copy(q, mb, ones.data(), ones.size() * sizeof(float));
  Tensor ta = Tensor::Contiguous(ma, vt::DType::kF32, dev, {T, 16});
  Tensor tb = Tensor::Contiguous(mb, vt::DType::kF32, dev, {16, Hv * D});
  Tensor to = Tensor::Contiguous(mo, vt::DType::kF32, dev, {T, Hv * D});
  reinterpret_cast<vt::MatmulFn>(
      vt::GetOp(vt::OpId::kMatmul, DeviceType::kTENSTORRENT))(q, to, ta, tb);

  // x: [T, Hv, D] contiguous host f32, committed as its own slot.
  std::vector<float> hx(static_cast<size_t>(T * Hv * D));
  {
    uint32_t s = 9100u;
    for (float& v : hx) v = 2.0f * GdnLcg(s);
  }
  std::vector<float> hw(static_cast<size_t>(D));
  {
    uint32_t s = 777u;
    for (float& v : hw) v = 0.8f + 0.4f * (GdnLcg(s) + 0.5f);
  }
  void* mx = tt.Alloc(hx.size() * sizeof(float));
  void* mw = tt.Alloc(hw.size() * sizeof(float));
  void* mout = tt.Alloc(hx.size() * sizeof(float));
  tt.Copy(q, mx, hx.data(), hx.size() * sizeof(float));
  tt.Copy(q, mw, hw.data(), hw.size() * sizeof(float));
  Tensor tx = Tensor::Contiguous(mx, vt::DType::kF32, dev, {T, Hv, D});
  Tensor tw = Tensor::Contiguous(mw, vt::DType::kF32, dev, {D});
  Tensor tout = Tensor::Contiguous(mout, vt::DType::kF32, dev, {T, Hv, D});
  // The gate view: rank-3 [T, Hv, D] covering the committed owner exactly.
  Tensor tg{};
  tg.data = mo;
  tg.dtype = vt::DType::kF32;
  tg.device = dev;
  tg.rank = 3;
  tg.shape[0] = T;
  tg.shape[1] = Hv;
  tg.shape[2] = D;
  tg.stride[0] = Hv * D;
  tg.stride[1] = D;
  tg.stride[2] = 1;
  vt::RmsNormGated(q, tout, tx, tg, tw, args);
  std::vector<float> out_tt(hx.size(), 0.0f);
  tt.Copy(q, out_tt.data(), mout, out_tt.size() * sizeof(float));

  // CPU oracle: identical geometry, gate bytes = the known producer values
  // (each gate row is the constant row sum of a).
  std::vector<float> hg(static_cast<size_t>(T * Hv * D));
  for (int64_t t = 0; t < T; ++t)
    for (int64_t e = 0; e < Hv * D; ++e)
      hg[static_cast<size_t>(t * Hv * D + e)] = gate_vals[static_cast<size_t>(t)];
  void* cgx = cpu.Alloc(hx.size() * sizeof(float));
  void* cgg = cpu.Alloc(hg.size() * sizeof(float));
  void* cgw = cpu.Alloc(hw.size() * sizeof(float));
  void* cgo = cpu.Alloc(hx.size() * sizeof(float));
  Queue cq = cpu.CreateQueue();
  cpu.Copy(cq, cgx, hx.data(), hx.size() * sizeof(float));
  cpu.Copy(cq, cgg, hg.data(), hg.size() * sizeof(float));
  cpu.Copy(cq, cgw, hw.data(), hw.size() * sizeof(float));
  Tensor ctx = Tensor::Contiguous(cgx, vt::DType::kF32, Device{DeviceType::kCPU, 0}, {T, Hv, D});
  Tensor ctg = Tensor::Contiguous(cgg, vt::DType::kF32, Device{DeviceType::kCPU, 0}, {T, Hv, D});
  Tensor ctw = Tensor::Contiguous(cgw, vt::DType::kF32, Device{DeviceType::kCPU, 0}, {D});
  Tensor cto = Tensor::Contiguous(cgo, vt::DType::kF32, Device{DeviceType::kCPU, 0}, {T, Hv, D});
  vt::RmsNormGated(cq, cto, ctx, ctg, ctw, args);
  std::vector<float> out_cpu(hx.size(), 0.0f);
  cpu.Copy(cq, out_cpu.data(), cgo, out_cpu.size() * sizeof(float));

  const float tol = 0.035f;
  GdnDiffStats d = CompareVsOracle(out_tt, out_cpu, /*rel=*/0.0f, tol);
  MESSAGE("kRmsNormGated committed-shadow [", T, ",", Hv * D, "] -> [",
          T * Hv, ",", D, "]: max_abs=", d.max_abs, " max_rel=", d.max_rel,
          " tol=", tol);
  tt.Free(ma);
  tt.Free(mb);
  tt.Free(mo);
  tt.Free(mx);
  tt.Free(mw);
  tt.Free(mout);
  cpu.Free(cgx);
  cpu.Free(cgg);
  cpu.Free(cgw);
  cpu.Free(cgo);
  CHECK(std::isfinite(d.max_abs));
  CHECK(d.within);
}

TEST_CASE("kTENSTORRENT kCausalConv1dFwd matches the CPU f32 oracle (rolling conv state, varlen)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& cpu = *vt::TryGetBackend(DeviceType::kCPU);
  constexpr int64_t C = 64, K = 4;  // conv dim / kernel width (Qwen GDN: K=4)

  // (qsl, silu_activation) sweep: N=1 at each T, then a ragged N=3 batch with
  // an EMPTY middle sequence.
  struct Case {
    std::vector<int32_t> qsl;
    bool silu;
  };
  std::vector<Case> cases;
  for (int64_t T : {int64_t{3}, int64_t{64}, int64_t{65}, int64_t{200}}) {
    cases.push_back({{0, static_cast<int32_t>(T)}, true});
    cases.push_back({{0, static_cast<int32_t>(T)}, false});
  }
  cases.push_back({{0, 37, 37, 68}, true});  // lengths {37, 0, 31}: ragged + empty

  for (const Case& cs : cases) {
    const int64_t T = cs.qsl.back();
    const int64_t N = static_cast<int64_t>(cs.qsl.size()) - 1;
    vt::CausalConv1dArgs args;
    args.silu_activation = cs.silu;
    std::vector<float> x(static_cast<size_t>(T * C));
    std::vector<float> w(static_cast<size_t>(C * K));
    std::vector<float> bias(static_cast<size_t>(C));
    std::vector<float> state(static_cast<size_t>(N * C * (K - 1)));
    std::vector<int32_t> his(static_cast<size_t>(N));
    {
      uint32_t sx = 31000u + static_cast<uint32_t>(T * 7 + N * 101 + (cs.silu ? 1 : 0));
      for (float& v : x) v = 2.0f * GdnLcg(sx);
      for (float& v : w) v = 0.4f * GdnLcg(sx);
      for (float& v : bias) v = 0.1f * GdnLcg(sx);
      for (float& v : state) v = GdnLcg(sx);
      for (int64_t n = 0; n < N; ++n) his[static_cast<size_t>(n)] = (n % 2 == 0) ? 1 : 0;
    }
    std::vector<float> out_cpu(x.size(), 0.0f), out_tt(x.size(), 0.0f);
    std::vector<float> st_cpu = state, st_tt = state;
    auto run = [&](Backend& b, DeviceType dt, std::vector<float>& out,
                   std::vector<float>& st) {
      void* mx = b.Alloc(x.size() * sizeof(float));
      void* mw = b.Alloc(w.size() * sizeof(float));
      void* mb = b.Alloc(bias.size() * sizeof(float));
      void* ms = b.Alloc(st.size() * sizeof(float));
      void* mq = b.Alloc(cs.qsl.size() * sizeof(int32_t));
      void* mh = b.Alloc(his.size() * sizeof(int32_t));
      void* mo = b.Alloc(out.size() * sizeof(float));
      Queue q = b.CreateQueue();
      b.Copy(q, mx, x.data(), x.size() * sizeof(float));
      b.Copy(q, mw, w.data(), w.size() * sizeof(float));
      b.Copy(q, mb, bias.data(), bias.size() * sizeof(float));
      b.Copy(q, ms, st.data(), st.size() * sizeof(float));
      b.Copy(q, mq, cs.qsl.data(), cs.qsl.size() * sizeof(int32_t));
      b.Copy(q, mh, his.data(), his.size() * sizeof(int32_t));
      Tensor tx = Tensor::Contiguous(mx, vt::DType::kF32, Device{dt, 0}, {T, C});
      Tensor tw = Tensor::Contiguous(mw, vt::DType::kF32, Device{dt, 0}, {C, K});
      Tensor tb = Tensor::Contiguous(mb, vt::DType::kF32, Device{dt, 0}, {C});
      Tensor ts = Tensor::Contiguous(ms, vt::DType::kF32, Device{dt, 0}, {N, C, K - 1});
      Tensor tq = Tensor::Contiguous(mq, vt::DType::kI32, Device{dt, 0},
                                     {static_cast<int64_t>(cs.qsl.size())});
      Tensor th = Tensor::Contiguous(mh, vt::DType::kI32, Device{dt, 0}, {N});
      Tensor to = Tensor::Contiguous(mo, vt::DType::kF32, Device{dt, 0}, {T, C});
      vt::CausalConv1dFwd(q, to, tx, tw, &tb, ts, tq, th, args);
      b.Copy(q, out.data(), mo, out.size() * sizeof(float));
      b.Copy(q, st.data(), ms, st.size() * sizeof(float));
      b.Free(mx);
      b.Free(mw);
      b.Free(mb);
      b.Free(ms);
      b.Free(mq);
      b.Free(mh);
      b.Free(mo);
    };
    run(cpu, DeviceType::kCPU, out_cpu, st_cpu);
    run(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, out_tt, st_tt);

    // Host-staged f32 path: same reduction order as the oracle — tight.
    GdnDiffStats d = CompareVsOracle(out_tt, out_cpu, /*rel=*/1e-4f, /*abs_floor=*/1e-5f);
    MESSAGE("kCausalConv1dFwd T=", T, " N=", N, " silu=", cs.silu,
            ": out max_abs=", d.max_abs, " max_rel=", d.max_rel);
    CHECK(d.within);
    GdnDiffStats ds = CompareVsOracle(st_tt, st_cpu, /*rel=*/1e-4f, /*abs_floor=*/1e-5f);
    MESSAGE("kCausalConv1dFwd T=", T, " N=", N, " silu=", cs.silu,
            ": conv_state max_abs=", ds.max_abs, " max_rel=", ds.max_rel);
    CHECK(ds.within);
  }
}

TEST_CASE("kTENSTORRENT kGdnPrefill matches the CPU f32 oracle (out AND final state, GQA + varlen)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& cpu = *vt::TryGetBackend(DeviceType::kCPU);
  constexpr int64_t Dk = 128, Dv = 128;  // Qwen3.8 head dims
  // (Hk, Hv, qsl): 4:1 GQA and 1:1, N=1 per T sweep, then ragged N=3 with an
  // EMPTY middle sequence. T covers non-multiples of the tt-metal chunk size.
  struct Case {
    int64_t Hk, Hv;
    std::vector<int32_t> qsl;
  };
  std::vector<Case> cases;
  for (int64_t T : {int64_t{3}, int64_t{64}, int64_t{65}, int64_t{200}}) {
    cases.push_back({2, 8, {0, static_cast<int32_t>(T)}});   // 4:1
    cases.push_back({2, 2, {0, static_cast<int32_t>(T)}});   // 1:1
  }
  cases.push_back({2, 8, {0, 37, 37, 68}});  // lengths {37, 0, 31}

  for (const Case& cs : cases) {
    const int64_t T = cs.qsl.back();
    const int64_t N = static_cast<int64_t>(cs.qsl.size()) - 1;
    vt::GdnArgs args;
    args.scale = 1.0f / std::sqrt(static_cast<float>(Dk));
    const int64_t Hk = cs.Hk, Hv = cs.Hv;
    std::vector<float> q(static_cast<size_t>(T * Hk * Dk));
    std::vector<float> k(static_cast<size_t>(T * Hk * Dk));
    std::vector<float> v(static_cast<size_t>(T * Hv * Dv));
    std::vector<float> g(static_cast<size_t>(T * Hv));
    std::vector<float> beta(static_cast<size_t>(T * Hv));
    std::vector<float> state(static_cast<size_t>(N * Hv * Dv * Dk));
    {
      uint32_t s = 52000u + static_cast<uint32_t>(T * 13 + Hk * 7 + Hv);
      for (float& x : q) x = GdnLcg(s);
      for (float& x : k) x = GdnLcg(s);
      for (float& x : v) x = GdnLcg(s);
      for (float& x : g) x = 0.24f * GdnLcg(s);   // log-decay in [-0.12, 0)
      for (float& x : beta) x = 0.75f + 0.5f * GdnLcg(s);  // (0.5, 1.25] gate
      for (float& x : state) x = 0.05f * GdnLcg(s);
      // The real caller pre-normalizes q/k (l2norm over the head dim, eps
      // 1e-6) BEFORE the op — mirror that so the inputs stay on-manifold.
      auto l2rows = [&](std::vector<float>& t, int64_t heads) {
        for (int64_t r = 0; r < T * heads; ++r) {
          float ss = 0.0f;
          for (int64_t j = 0; j < Dk; ++j) {
            const float x = t[static_cast<size_t>(r * Dk + j)];
            ss += x * x;
          }
          const float inv = 1.0f / std::sqrt(ss + 1e-6f);
          for (int64_t j = 0; j < Dk; ++j)
            t[static_cast<size_t>(r * Dk + j)] *= inv;
        }
      };
      l2rows(q, Hk);
      l2rows(k, Hk);
    }
    std::vector<float> out_cpu(v.size(), 0.0f), out_tt(v.size(), 0.0f);
    std::vector<float> st_cpu = state, st_tt = state;
    auto run = [&](Backend& b, DeviceType dt, std::vector<float>& out,
                   std::vector<float>& st) {
      void* mq = b.Alloc(q.size() * sizeof(float));
      void* mk = b.Alloc(k.size() * sizeof(float));
      void* mv = b.Alloc(v.size() * sizeof(float));
      void* mg = b.Alloc(g.size() * sizeof(float));
      void* mb = b.Alloc(beta.size() * sizeof(float));
      void* ms = b.Alloc(st.size() * sizeof(float));
      void* mx = b.Alloc(cs.qsl.size() * sizeof(int32_t));
      void* mo = b.Alloc(out.size() * sizeof(float));
      Queue qd = b.CreateQueue();
      b.Copy(qd, mq, q.data(), q.size() * sizeof(float));
      b.Copy(qd, mk, k.data(), k.size() * sizeof(float));
      b.Copy(qd, mv, v.data(), v.size() * sizeof(float));
      b.Copy(qd, mg, g.data(), g.size() * sizeof(float));
      b.Copy(qd, mb, beta.data(), beta.size() * sizeof(float));
      b.Copy(qd, ms, st.data(), st.size() * sizeof(float));
      b.Copy(qd, mx, cs.qsl.data(), cs.qsl.size() * sizeof(int32_t));
      Tensor tq = Tensor::Contiguous(mq, vt::DType::kF32, Device{dt, 0}, {T, Hk, Dk});
      Tensor tk = Tensor::Contiguous(mk, vt::DType::kF32, Device{dt, 0}, {T, Hk, Dk});
      Tensor tv = Tensor::Contiguous(mv, vt::DType::kF32, Device{dt, 0}, {T, Hv, Dv});
      Tensor tg = Tensor::Contiguous(mg, vt::DType::kF32, Device{dt, 0}, {T, Hv});
      Tensor tb = Tensor::Contiguous(mb, vt::DType::kF32, Device{dt, 0}, {T, Hv});
      Tensor ts = Tensor::Contiguous(ms, vt::DType::kF32, Device{dt, 0}, {N, Hv, Dv, Dk});
      Tensor tx = Tensor::Contiguous(mx, vt::DType::kI32, Device{dt, 0},
                                     {static_cast<int64_t>(cs.qsl.size())});
      Tensor to = Tensor::Contiguous(mo, vt::DType::kF32, Device{dt, 0}, {T, Hv, Dv});
      vt::GdnPrefill(qd, to, tq, tk, tv, tg, tb, ts, tx, args);
      b.Copy(qd, out.data(), mo, out.size() * sizeof(float));
      b.Copy(qd, st.data(), ms, st.size() * sizeof(float));
      b.Free(mq);
      b.Free(mk);
      b.Free(mv);
      b.Free(mg);
      b.Free(mb);
      b.Free(ms);
      b.Free(mx);
      b.Free(mo);
    };
    run(cpu, DeviceType::kCPU, out_cpu, st_cpu);
    run(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, out_tt, st_tt);

    // bf16 q/k/v tiles + fp32 state (HiFi4): per-T envelope, calibrated on the
    // P150 and stated per T (a recurrence amplifies rounding — spec Risk #1).
    const float tol_o = T <= 3 ? 0.05f : (T <= 64 ? 0.05f : 0.08f);
    const float tol_s = 0.05f;
    GdnDiffStats d = CompareVsOracle(out_tt, out_cpu, /*rel=*/0.0f, tol_o);
    MESSAGE("kGdnPrefill T=", T, " N=", N, " Hk=", Hk, " Hv=", Hv,
            ": out max_abs=", d.max_abs, " max_rel=", d.max_rel, " tol=", tol_o);
    CHECK(std::isfinite(d.max_abs));
    CHECK(d.within);
    GdnDiffStats ds = CompareVsOracle(st_tt, st_cpu, /*rel=*/0.0f, tol_s);
    MESSAGE("kGdnPrefill T=", T, " N=", N, " Hk=", Hk, " Hv=", Hv,
            ": final_state max_abs=", ds.max_abs, " max_rel=", ds.max_rel,
            " tol=", tol_s);
    CHECK(std::isfinite(ds.max_abs));
    CHECK(ds.within);
  }
}

// ==== BACKEND-TENSTORRENT-GDN W2: the decode op set vs the CPU f32 oracle.
// Same doctrine as the W1 block above: identical inputs, both arms run the
// public vt:: op on their own backend. Before the kernels land, Resolve
// refuses BY NAME on the TT arm (discrete card, no portable tier) — that
// refusal is this block's red state. The decode state / conv state live in
// DEVICE shadows keyed by the host pointer, so decode must not round-trip the
// state per token: the vt::tenstorrent traffic counters assert that by
// evidence inside the round-trip case (spec Evidence), not by assumption.
#include "../../src/vt/tenstorrent/tenstorrent_device.h"

TEST_CASE("kTENSTORRENT kCausalConv1dUpdate matches the CPU f32 oracle (read-old-then-roll, indexed + NULL + widened)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& cpu = *vt::TryGetBackend(DeviceType::kCPU);
  constexpr int64_t C = 64, K = 4;  // conv dim / kernel width (Qwen GDN: K=4)

  // One update step on a given starting conv_state (host bytes in `st`),
  // one token per row, optional indexed cache. Returns out; `st` is updated.
  auto step = [&](Backend& b, DeviceType dt, int64_t B, bool silu,
                  std::vector<float>& st, const std::vector<float>& x,
                  const std::vector<float>& w, const std::vector<float>& bias,
                  const std::vector<int32_t>* idx, int64_t state_len,
                  std::vector<float>& out) {
    void* mx = b.Alloc(x.size() * sizeof(float));
    void* mw = b.Alloc(w.size() * sizeof(float));
    void* mb = bias.empty() ? nullptr : b.Alloc(bias.size() * sizeof(float));
    void* ms = b.Alloc(st.size() * sizeof(float));
    void* mo = b.Alloc(out.size() * sizeof(float));
    void* mi = idx == nullptr ? nullptr : b.Alloc(idx->size() * sizeof(int32_t));
    Queue q = b.CreateQueue();
    b.Copy(q, mx, x.data(), x.size() * sizeof(float));
    b.Copy(q, mw, w.data(), w.size() * sizeof(float));
    if (mb != nullptr) b.Copy(q, mb, bias.data(), bias.size() * sizeof(float));
    b.Copy(q, ms, st.data(), st.size() * sizeof(float));
    if (mi != nullptr) b.Copy(q, mi, idx->data(), idx->size() * sizeof(int32_t));
    // Seed the out buffer with its CURRENT content: the NULL-row contract
    // ("the kernel leaves the out row untouched") is only observable when the
    // prefill actually reaches the buffer the kernel sees.
    b.Copy(q, mo, out.data(), out.size() * sizeof(float));
    // x is fed as a PADDED-row view (the merged qkvz slice the contract
    // admits): outer stride C+8, garbage in the pad.
    const int64_t x_row = C, pad = 8;
    std::vector<float> xp(static_cast<size_t>(B * (x_row + pad)), 0.0f);
    for (int64_t i = 0; i < B; ++i)
      for (int64_t c = 0; c < C; ++c)
        xp[static_cast<size_t>(i * (x_row + pad) + c)] = x[static_cast<size_t>(i * C + c)];
    void* mxp = b.Alloc(xp.size() * sizeof(float));
    b.Copy(q, mxp, xp.data(), xp.size() * sizeof(float));
    Tensor tx{};
    tx.data = mxp;
    tx.dtype = vt::DType::kF32;
    tx.device = Device{dt, 0};
    tx.rank = 2;
    tx.shape[0] = B;
    tx.shape[1] = C;
    tx.stride[0] = x_row + pad;
    tx.stride[1] = 1;
    Tensor tw = Tensor::Contiguous(mw, vt::DType::kF32, Device{dt, 0}, {C, K});
    Tensor tb{};
    if (mb != nullptr) tb = Tensor::Contiguous(mb, vt::DType::kF32, Device{dt, 0}, {C});
    const int64_t slots = st.size() / static_cast<size_t>(C * state_len);
    Tensor ts = Tensor::Contiguous(ms, vt::DType::kF32, Device{dt, 0},
                                   {slots, C, state_len});
    Tensor ti{};
    if (mi != nullptr)
      ti = Tensor::Contiguous(mi, vt::DType::kI32, Device{dt, 0},
                              {static_cast<int64_t>(idx->size())});
    Tensor to = Tensor::Contiguous(mo, vt::DType::kF32, Device{dt, 0}, {B, C});
    vt::CausalConv1dArgs args;
    args.silu_activation = silu;
    vt::CausalConv1dUpdate(q, to, tx, tw, mb != nullptr ? &tb : nullptr, ts, args,
                           mi != nullptr ? &ti : nullptr);
    b.Copy(q, out.data(), mo, out.size() * sizeof(float));
    b.Copy(q, st.data(), ms, st.size() * sizeof(float));
    b.Free(mx);
    b.Free(mw);
    if (mb != nullptr) b.Free(mb);
    b.Free(ms);
    b.Free(mo);
    if (mi != nullptr) b.Free(mi);
    b.Free(mxp);
  };

  // --- Sweep A: fresh state, B in {1,3} x silu x bias on/off. Tight envelope:
  // f32 device compute of a 4-tap MAC.
  for (int64_t B : {int64_t{1}, int64_t{3}}) {
    for (int silu : {0, 1}) {
      for (int has_bias : {0, 1}) {
        uint32_t s = 61000u + static_cast<uint32_t>(B * 131 + silu * 17 + has_bias);
        std::vector<float> w(static_cast<size_t>(C * K)), bias;
        std::vector<float> st(static_cast<size_t>(B * C * (K - 1))), x(static_cast<size_t>(B * C));
        for (float& v : w) v = 0.4f * GdnLcg(s);
        for (float& v : st) v = GdnLcg(s);
        for (float& v : x) v = 2.0f * GdnLcg(s);
        if (has_bias) {
          bias.resize(static_cast<size_t>(C));
          for (float& v : bias) v = 0.1f * GdnLcg(s);
        }
        std::vector<float> st_cpu = st, st_tt = st;
        std::vector<float> out_cpu(static_cast<size_t>(B * C), 0.0f),
            out_tt(static_cast<size_t>(B * C), 0.0f);
        step(cpu, DeviceType::kCPU, B, silu != 0, st_cpu, x, w, bias, nullptr, K - 1, out_cpu);
        step(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, B,
             silu != 0, st_tt, x, w, bias, nullptr, K - 1, out_tt);
        GdnDiffStats d = CompareVsOracle(out_tt, out_cpu, 1e-4f, 1e-5f);
        MESSAGE("kCausalConv1dUpdate B=", B, " silu=", silu, " bias=", has_bias,
                ": out max_abs=", d.max_abs, " max_rel=", d.max_rel);
        CHECK(std::isfinite(d.max_abs));
        CHECK(d.within);
        GdnDiffStats ds = CompareVsOracle(st_tt, st_cpu, 1e-4f, 1e-5f);
        MESSAGE("kCausalConv1dUpdate B=", B, " silu=", silu, " bias=", has_bias,
                ": conv_state max_abs=", ds.max_abs, " max_rel=", ds.max_rel);
        CHECK(std::isfinite(ds.max_abs));
        CHECK(ds.within);
      }
    }
  }

  // --- Sweep B: ROLLING CONTINUATION from a W1 kCausalConv1dFwd state — the
  // decode step must consume exactly the state prefill leaves behind.
  {
    const int64_t T = 5, B = 1;
    uint32_t s = 62000u;
    std::vector<float> xf(static_cast<size_t>(T * C)), w(static_cast<size_t>(C * K)),
        bias(static_cast<size_t>(C)), st(static_cast<size_t>(B * C * (K - 1))),
        x1(static_cast<size_t>(B * C));
    for (float& v : xf) v = 2.0f * GdnLcg(s);
    for (float& v : w) v = 0.4f * GdnLcg(s);
    for (float& v : bias) v = 0.1f * GdnLcg(s);
    for (float& v : st) v = GdnLcg(s);
    for (float& v : x1) v = 2.0f * GdnLcg(s);
    const std::vector<int32_t> qsl{0, static_cast<int32_t>(T)};
    const std::vector<int32_t> his{1};
    std::vector<float> st_cpu = st, st_tt = st;
    auto fwd = [&](Backend& b, DeviceType dt, std::vector<float>& stt,
                   std::vector<float>& outf) {
      void* mx = b.Alloc(xf.size() * sizeof(float));
      void* mw = b.Alloc(w.size() * sizeof(float));
      void* mb = b.Alloc(bias.size() * sizeof(float));
      void* ms = b.Alloc(stt.size() * sizeof(float));
      void* mq = b.Alloc(qsl.size() * sizeof(int32_t));
      void* mh = b.Alloc(his.size() * sizeof(int32_t));
      void* mo = b.Alloc(outf.size() * sizeof(float));
      Queue q = b.CreateQueue();
      b.Copy(q, mx, xf.data(), xf.size() * sizeof(float));
      b.Copy(q, mw, w.data(), w.size() * sizeof(float));
      b.Copy(q, mb, bias.data(), bias.size() * sizeof(float));
      b.Copy(q, ms, stt.data(), stt.size() * sizeof(float));
      b.Copy(q, mq, qsl.data(), qsl.size() * sizeof(int32_t));
      b.Copy(q, mh, his.data(), his.size() * sizeof(int32_t));
      Tensor tx = Tensor::Contiguous(mx, vt::DType::kF32, Device{dt, 0}, {T, C});
      Tensor tw = Tensor::Contiguous(mw, vt::DType::kF32, Device{dt, 0}, {C, K});
      Tensor tb = Tensor::Contiguous(mb, vt::DType::kF32, Device{dt, 0}, {C});
      Tensor ts = Tensor::Contiguous(ms, vt::DType::kF32, Device{dt, 0}, {B, C, K - 1});
      Tensor tq = Tensor::Contiguous(mq, vt::DType::kI32, Device{dt, 0},
                                     {static_cast<int64_t>(qsl.size())});
      Tensor th = Tensor::Contiguous(mh, vt::DType::kI32, Device{dt, 0}, {B});
      Tensor to = Tensor::Contiguous(mo, vt::DType::kF32, Device{dt, 0}, {T, C});
      vt::CausalConv1dArgs a;
      a.silu_activation = true;
      vt::CausalConv1dFwd(q, to, tx, tw, &tb, ts, tq, th, a);
      b.Copy(q, stt.data(), ms, stt.size() * sizeof(float));
      b.Free(mx);
      b.Free(mw);
      b.Free(mb);
      b.Free(ms);
      b.Free(mq);
      b.Free(mh);
      b.Free(mo);
    };
    std::vector<float> of_cpu(static_cast<size_t>(T * C)), of_tt(static_cast<size_t>(T * C));
    fwd(cpu, DeviceType::kCPU, st_cpu, of_cpu);
    fwd(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, st_tt, of_tt);
    GdnDiffStats df = CompareVsOracle(st_tt, st_cpu, 1e-4f, 1e-5f);
    MESSAGE("kCausalConv1dUpdate continuation: fwd state max_abs=", df.max_abs);
    CHECK(df.within);
    std::vector<float> out_cpu(static_cast<size_t>(B * C)), out_tt(static_cast<size_t>(B * C));
    step(cpu, DeviceType::kCPU, B, true, st_cpu, x1, w, bias, nullptr, K - 1, out_cpu);
    step(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, B, true,
         st_tt, x1, w, bias, nullptr, K - 1, out_tt);
    GdnDiffStats d = CompareVsOracle(out_tt, out_cpu, 1e-4f, 1e-5f);
    GdnDiffStats ds = CompareVsOracle(st_tt, st_cpu, 1e-4f, 1e-5f);
    MESSAGE("kCausalConv1dUpdate continuation: out max_abs=", d.max_abs,
            " state max_abs=", ds.max_abs);
    CHECK(std::isfinite(d.max_abs));
    CHECK(d.within);
    CHECK(std::isfinite(ds.max_abs));
    CHECK(ds.within);
  }

  // --- Indexed form: conv_state is the FULL cache; idx names the slot. NULL
  // (idx<0) rows: the oracle leaves out AND the cache row untouched.
  {
    const int64_t slots = 5, B = 3;
    uint32_t s = 63000u;
    std::vector<float> w(static_cast<size_t>(C * K)), bias(static_cast<size_t>(C));
    std::vector<float> cache(static_cast<size_t>(slots * C * (K - 1))),
        x(static_cast<size_t>(B * C));
    for (float& v : w) v = 0.4f * GdnLcg(s);
    for (float& v : bias) v = 0.1f * GdnLcg(s);
    for (float& v : cache) v = GdnLcg(s);
    for (float& v : x) v = 2.0f * GdnLcg(s);
    const std::vector<int32_t> idx{4, 0, 2};
    std::vector<float> ca_cpu = cache, ca_tt = cache;
    std::vector<float> out_cpu(static_cast<size_t>(B * C), 0.0f),
        out_tt(static_cast<size_t>(B * C), 0.0f);
    step(cpu, DeviceType::kCPU, B, true, ca_cpu, x, w, bias, &idx, K - 1, out_cpu);
    step(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, B, true,
         ca_tt, x, w, bias, &idx, K - 1, out_tt);
    GdnDiffStats d = CompareVsOracle(out_tt, out_cpu, 1e-4f, 1e-5f);
    GdnDiffStats ds = CompareVsOracle(ca_tt, ca_cpu, 1e-4f, 1e-5f);
    MESSAGE("kCausalConv1dUpdate indexed: out max_abs=", d.max_abs,
            " cache max_abs=", ds.max_abs);
    CHECK(std::isfinite(d.max_abs));
    CHECK(d.within);
    CHECK(std::isfinite(ds.max_abs));
    CHECK(ds.within);
    // NULL slot: sentinel-prefilled out must stay UNTOUCHED on the NULL row,
    // and the named-away cache row must stay UNTOUCHED (ops.h: NULL row skip).
    const std::vector<int32_t> idx_null{1, -1, 3};
    std::vector<float> ca2_cpu = cache, ca2_tt = cache;
    for (int64_t i = 0; i < B * C; ++i) {
      out_cpu[static_cast<size_t>(i)] = 7.5f;
      out_tt[static_cast<size_t>(i)] = 7.5f;
    }
    step(cpu, DeviceType::kCPU, B, true, ca2_cpu, x, w, bias, &idx_null, K - 1, out_cpu);
    step(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, B, true,
         ca2_tt, x, w, bias, &idx_null, K - 1, out_tt);
    GdnDiffStats dn = CompareVsOracle(out_tt, out_cpu, 1e-4f, 1e-5f);
    GdnDiffStats dsn = CompareVsOracle(ca2_tt, ca2_cpu, 1e-4f, 1e-5f);
    MESSAGE("kCausalConv1dUpdate indexed NULL: out max_abs=", dn.max_abs,
            " cache max_abs=", dsn.max_abs);
    CHECK(std::isfinite(dn.max_abs));
    CHECK(dn.within);
    CHECK(std::isfinite(dsn.max_abs));
    CHECK(dsn.within);
    CHECK(out_tt[static_cast<size_t>(1 * C + 3)] == 7.5f);  // NULL row untouched
  }

  // --- Widened cache row (spec taps): the update operates on the LEADING K-1
  // window with the physical stride; the tail taps stay untouched.
  {
    const int64_t slots = 3, B = 2, state_len = (K - 1) + 2;
    uint32_t s = 64000u;
    std::vector<float> w(static_cast<size_t>(C * K)), bias(static_cast<size_t>(C));
    std::vector<float> cache(static_cast<size_t>(slots * C * state_len)),
        x(static_cast<size_t>(B * C));
    for (float& v : w) v = 0.4f * GdnLcg(s);
    for (float& v : bias) v = 0.1f * GdnLcg(s);
    for (float& v : cache) v = GdnLcg(s);
    for (float& v : x) v = 2.0f * GdnLcg(s);
    const std::vector<int32_t> idx{2, 0};
    std::vector<float> ca_cpu = cache, ca_tt = cache;
    std::vector<float> out_cpu(static_cast<size_t>(B * C)), out_tt(static_cast<size_t>(B * C));
    step(cpu, DeviceType::kCPU, B, false, ca_cpu, x, w, bias, &idx, state_len, out_cpu);
    step(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, B, false,
         ca_tt, x, w, bias, &idx, state_len, out_tt);
    GdnDiffStats d = CompareVsOracle(out_tt, out_cpu, 1e-4f, 1e-5f);
    GdnDiffStats ds = CompareVsOracle(ca_tt, ca_cpu, 1e-4f, 1e-5f);
    MESSAGE("kCausalConv1dUpdate widened: out max_abs=", d.max_abs,
            " cache max_abs=", ds.max_abs);
    CHECK(std::isfinite(d.max_abs));
    CHECK(d.within);
    CHECK(std::isfinite(ds.max_abs));
    CHECK(ds.within);
  }

  // --- Traffic: three chained update steps on the SAME cache buffer — the
  // conv state must stay device-resident (one upload, zero downloads). The
  // buffer is allocated ONCE (a fresh Alloc per step would be a different
  // host pointer and legitimately re-upload — the residency contract is per
  // buffer, mirroring the decode round-trip below).
  {
    const int64_t B = 2;
    uint32_t s = 65000u;
    std::vector<float> w(static_cast<size_t>(C * K)), bias(static_cast<size_t>(C));
    std::vector<float> st(static_cast<size_t>(B * C * (K - 1)));
    std::vector<float> x(static_cast<size_t>(B * C));
    for (float& v : w) v = 0.4f * GdnLcg(s);
    for (float& v : bias) v = 0.1f * GdnLcg(s);
    for (float& v : st) v = GdnLcg(s);
    for (float& v : x) v = 2.0f * GdnLcg(s);
    Backend& tt = *vt::TryGetBackend(DeviceType::kTENSTORRENT);
    void* mw = tt.Alloc(w.size() * sizeof(float));
    void* mb = tt.Alloc(bias.size() * sizeof(float));
    void* ms = tt.Alloc(st.size() * sizeof(float));
    void* mo = tt.Alloc(B * C * sizeof(float));
    void* mx = tt.Alloc(x.size() * sizeof(float));
    Queue q = tt.CreateQueue();
    tt.Copy(q, mw, w.data(), w.size() * sizeof(float));
    tt.Copy(q, mb, bias.data(), bias.size() * sizeof(float));
    tt.Copy(q, ms, st.data(), st.size() * sizeof(float));  // the ONE upload
    vt::tenstorrent::ResetGdnShadowTraffic();
    std::vector<float> out(static_cast<size_t>(B * C), 0.0f);
    for (int step_i = 0; step_i < 3; ++step_i) {
      for (float& v : x) v = 2.0f * GdnLcg(s);
      tt.Copy(q, mx, x.data(), x.size() * sizeof(float));
      tt.Copy(q, mo, out.data(), out.size() * sizeof(float));
      Tensor tx = Tensor::Contiguous(mx, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {B, C});
      Tensor tw = Tensor::Contiguous(mw, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {C, K});
      Tensor tb = Tensor::Contiguous(mb, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {C});
      Tensor ts = Tensor::Contiguous(ms, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                     {B, C, K - 1});
      Tensor to = Tensor::Contiguous(mo, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {B, C});
      vt::CausalConv1dArgs a;
      a.silu_activation = true;
      vt::CausalConv1dUpdate(q, to, tx, tw, &tb, ts, a, nullptr);
      tt.Copy(q, out.data(), mo, out.size() * sizeof(float));  // out readback only
    }
    const auto tr = vt::tenstorrent::GetGdnShadowTraffic();
    const uint64_t want_up = static_cast<uint64_t>(st.size()) * sizeof(float);
    MESSAGE("kCausalConv1dUpdate traffic: steps=", tr.decode_steps,
            " h2d=", tr.state_h2d_bytes, " d2h=", tr.state_d2h_bytes,
            " (cache bytes=", want_up, ")");
    CHECK(tr.decode_steps == 3);
    CHECK(tr.state_h2d_bytes == want_up);  // exactly ONE upload across 3 steps
    CHECK(tr.state_d2h_bytes == 0);
    tt.Free(mw);
    tt.Free(mb);
    tt.Free(ms);
    tt.Free(mo);
    tt.Free(mx);
  }
  // --- bf16 conv_state (production mamba_cache_dtype default): the
  // SupportsCompressedConvState arm. The oracle is the CUDA bf16-STORAGE
  // emulation — widen bf16->f32, compute the step in f32, round the state
  // back to bf16 EVERY step ("read/written in f32 registers", cuda_gdn.cu) —
  // not a persistent f32 recurrence. x carries a FULL f32 mantissa in the
  // first arm so a shadow that skips the per-step store-rounding diverges in
  // step 2+ (the rounded tap feeds the next MAC); the second arm feeds
  // bf16-representable x (the production activation dtype), where storage
  // rounding is a no-op and both readings agree bit-for-bit.
  {
    Backend& tt = *vt::TryGetBackend(DeviceType::kTENSTORRENT);
    Backend& cpu = *vt::TryGetBackend(DeviceType::kCPU);
    const int64_t slots = 4, B = 2, state_len = K - 1;
    const std::vector<int32_t> idx{3, 1};
    for (int x_full_mantissa : {1, 0}) {
      uint32_t s = 66000u + static_cast<uint32_t>(x_full_mantissa);
      std::vector<float> w(static_cast<size_t>(C * K)), bias(static_cast<size_t>(C));
      std::vector<float> st_f(static_cast<size_t>(slots * C * state_len));
      for (float& v : w) v = 0.4f * GdnLcg(s);
      for (float& v : bias) v = 0.1f * GdnLcg(s);
      for (float& v : st_f) v = GdnLcg(s);
      // The cache holds bf16 bits (production storage); the initial values
      // round ONCE so both sides start from identical storage.
      std::vector<uint16_t> st_bits(st_f.size());
      for (size_t i = 0; i < st_f.size(); ++i) st_bits[i] = vt::F32ToBF16(st_f[i]);

      // Emulation: per step, widen the bf16 cache, run the CPU f32 oracle
      // in place, round the cache back to bf16 (the CUDA store boundary).
      std::vector<uint16_t> emu = st_bits;
      auto emu_step = [&](const std::vector<float>& x, std::vector<float>& out) {
        std::vector<float> st(emu.size());
        for (size_t i = 0; i < emu.size(); ++i) st[i] = vt::BF16ToF32(emu[i]);
        void* mx = cpu.Alloc(x.size() * sizeof(float));
        void* mw = cpu.Alloc(w.size() * sizeof(float));
        void* mb = cpu.Alloc(bias.size() * sizeof(float));
        void* ms = cpu.Alloc(st.size() * sizeof(float));
        void* mo = cpu.Alloc(out.size() * sizeof(float));
        void* mi = cpu.Alloc(idx.size() * sizeof(int32_t));
        Queue q = cpu.CreateQueue();
        cpu.Copy(q, mx, x.data(), x.size() * sizeof(float));
        cpu.Copy(q, mw, w.data(), w.size() * sizeof(float));
        cpu.Copy(q, mb, bias.data(), bias.size() * sizeof(float));
        cpu.Copy(q, ms, st.data(), st.size() * sizeof(float));
        cpu.Copy(q, mi, idx.data(), idx.size() * sizeof(int32_t));
        Tensor tx = Tensor::Contiguous(mx, vt::DType::kF32, Device{DeviceType::kCPU, 0}, {B, C});
        Tensor tw = Tensor::Contiguous(mw, vt::DType::kF32, Device{DeviceType::kCPU, 0}, {C, K});
        Tensor tb = Tensor::Contiguous(mb, vt::DType::kF32, Device{DeviceType::kCPU, 0}, {C});
        Tensor ts = Tensor::Contiguous(ms, vt::DType::kF32, Device{DeviceType::kCPU, 0},
                                       {slots, C, state_len});
        Tensor ti = Tensor::Contiguous(mi, vt::DType::kI32, Device{DeviceType::kCPU, 0},
                                       {static_cast<int64_t>(idx.size())});
        Tensor to = Tensor::Contiguous(mo, vt::DType::kF32, Device{DeviceType::kCPU, 0}, {B, C});
        vt::CausalConv1dArgs a;
        a.silu_activation = true;
        vt::CausalConv1dUpdate(q, to, tx, tw, &tb, ts, a, &ti);
        cpu.Copy(q, out.data(), mo, out.size() * sizeof(float));
        cpu.Copy(q, st.data(), ms, st.size() * sizeof(float));
        for (size_t i = 0; i < emu.size(); ++i) emu[i] = vt::F32ToBF16(st[i]);
        for (void* m : {mx, mw, mb, ms, mo, mi}) cpu.Free(m);
      };

      // TT side: one persistent bf16 cache buffer, chained steps (the
      // transposed-shadow fast path across steps is exactly what must honor
      // the storage rounding).
      void* mw = tt.Alloc(w.size() * sizeof(float));
      void* mb = tt.Alloc(bias.size() * sizeof(float));
      void* ms = tt.Alloc(st_bits.size() * sizeof(uint16_t));
      void* mo = tt.Alloc(static_cast<size_t>(B * C) * sizeof(float));
      void* mx = tt.Alloc(static_cast<size_t>(B * C) * sizeof(float));
      void* mi = tt.Alloc(idx.size() * sizeof(int32_t));
      Queue q = tt.CreateQueue();
      tt.Copy(q, mw, w.data(), w.size() * sizeof(float));
      tt.Copy(q, mb, bias.data(), bias.size() * sizeof(float));
      tt.Copy(q, ms, st_bits.data(), st_bits.size() * sizeof(uint16_t));  // the ONE upload
      tt.Copy(q, mi, idx.data(), idx.size() * sizeof(int32_t));
      for (int step_i = 0; step_i < 4; ++step_i) {
        std::vector<float> x(static_cast<size_t>(B * C));
        for (float& v : x)
          v = x_full_mantissa ? (2.0f * GdnLcg(s) + 0.03125f)  // full f32 mantissa
                              : vt::BF16ToF32(vt::F32ToBF16(2.0f * GdnLcg(s)));
        std::vector<float> out_emu(static_cast<size_t>(B * C), 0.0f);
        emu_step(x, out_emu);
        tt.Copy(q, mx, x.data(), x.size() * sizeof(float));
        Tensor tx = Tensor::Contiguous(mx, vt::DType::kF32,
                                       Device{DeviceType::kTENSTORRENT, 0}, {B, C});
        Tensor tw = Tensor::Contiguous(mw, vt::DType::kF32,
                                       Device{DeviceType::kTENSTORRENT, 0}, {C, K});
        Tensor tb = Tensor::Contiguous(mb, vt::DType::kF32,
                                       Device{DeviceType::kTENSTORRENT, 0}, {C});
        Tensor ts = Tensor::Contiguous(ms, vt::DType::kBF16,
                                       Device{DeviceType::kTENSTORRENT, 0},
                                       {slots, C, state_len});
        Tensor ti = Tensor::Contiguous(mi, vt::DType::kI32,
                                       Device{DeviceType::kTENSTORRENT, 0},
                                       {static_cast<int64_t>(idx.size())});
        Tensor to = Tensor::Contiguous(mo, vt::DType::kF32,
                                       Device{DeviceType::kTENSTORRENT, 0}, {B, C});
        vt::CausalConv1dArgs a;
        a.silu_activation = true;
        vt::CausalConv1dUpdate(q, to, tx, tw, &tb, ts, a, &ti);
        std::vector<float> out_tt(static_cast<size_t>(B * C), 0.0f);
        tt.Copy(q, out_tt.data(), mo, out_tt.size() * sizeof(float));
        GdnDiffStats d = CompareVsOracle(out_tt, out_emu, 1e-4f, 1e-5f);
        MESSAGE("kCausalConv1dUpdate bf16-state x_full=", x_full_mantissa,
                " step=", step_i, ": out max_abs=", d.max_abs,
                " max_rel=", d.max_rel);
        CHECK(std::isfinite(d.max_abs));
        CHECK(d.within);
      }
      // Storage truth: the downloaded bf16 cache must match the emulation's
      // bf16 cache BIT-FOR-BIT (both round the same values through RNE).
      std::vector<uint16_t> got_bits(st_bits.size(), 0);
      tt.Copy(q, got_bits.data(), ms, got_bits.size() * sizeof(uint16_t));
      size_t mism = 0;
      for (size_t i = 0; i < emu.size(); ++i)
        if (got_bits[i] != emu[i]) ++mism;
      MESSAGE("kCausalConv1dUpdate bf16-state x_full=", x_full_mantissa,
              ": cache bit mismatches=", mism, "/", emu.size());
      CHECK(mism == 0);
      for (void* m : {mw, mb, ms, mo, mx, mi}) tt.Free(m);
    }
  }
}

TEST_CASE("kTENSTORRENT kGdnDecode matches the CPU f32 oracle (rank-1 step, both state_idx forms, NULL slot)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& cpu = *vt::TryGetBackend(DeviceType::kCPU);
  constexpr int64_t Dk = 128, Dv = 128;

  // One decode step: B single-token sequences (token t of the stream), state
  // compact [B,Hv,Dv,Dk] or the FULL cache with idx rows. Updates st/out.
  auto step = [&](Backend& b, DeviceType dt, int64_t B, int64_t Hk, int64_t Hv,
                  const std::vector<float>& q, const std::vector<float>& k,
                  const std::vector<float>& v, const std::vector<float>& g,
                  const std::vector<float>& beta, std::vector<float>& st,
                  const std::vector<int32_t>* idx, std::vector<float>& out) {
    void* mq = b.Alloc(q.size() * sizeof(float));
    void* mk = b.Alloc(k.size() * sizeof(float));
    void* mv = b.Alloc(v.size() * sizeof(float));
    void* mg = b.Alloc(g.size() * sizeof(float));
    void* mb = b.Alloc(beta.size() * sizeof(float));
    void* ms = b.Alloc(st.size() * sizeof(float));
    void* mo = b.Alloc(out.size() * sizeof(float));
    void* mi = idx == nullptr ? nullptr : b.Alloc(idx->size() * sizeof(int32_t));
    Queue qq = b.CreateQueue();
    b.Copy(qq, mq, q.data(), q.size() * sizeof(float));
    b.Copy(qq, mk, k.data(), k.size() * sizeof(float));
    b.Copy(qq, mv, v.data(), v.size() * sizeof(float));
    b.Copy(qq, mg, g.data(), g.size() * sizeof(float));
    b.Copy(qq, mb, beta.data(), beta.size() * sizeof(float));
    b.Copy(qq, ms, st.data(), st.size() * sizeof(float));
    if (mi != nullptr) b.Copy(qq, mi, idx->data(), idx->size() * sizeof(int32_t));
    const int64_t slots = st.size() / static_cast<size_t>(Hv * Dv * Dk);
    Tensor tq = Tensor::Contiguous(mq, vt::DType::kF32, Device{dt, 0}, {B, Hk, Dk});
    Tensor tk = Tensor::Contiguous(mk, vt::DType::kF32, Device{dt, 0}, {B, Hk, Dk});
    Tensor tv = Tensor::Contiguous(mv, vt::DType::kF32, Device{dt, 0}, {B, Hv, Dv});
    Tensor tg = Tensor::Contiguous(mg, vt::DType::kF32, Device{dt, 0}, {B, Hv});
    Tensor tb = Tensor::Contiguous(mb, vt::DType::kF32, Device{dt, 0}, {B, Hv});
    Tensor ts = Tensor::Contiguous(ms, vt::DType::kF32, Device{dt, 0},
                                   {slots, Hv, Dv, Dk});
    Tensor ti{};
    if (mi != nullptr)
      ti = Tensor::Contiguous(mi, vt::DType::kI32, Device{dt, 0},
                              {static_cast<int64_t>(idx->size())});
    Tensor to = Tensor::Contiguous(mo, vt::DType::kF32, Device{dt, 0}, {B, Hv, Dv});
    vt::GdnArgs args;
    args.scale = 1.0f / std::sqrt(static_cast<float>(Dk));
    vt::GdnDecode(qq, to, tq, tk, tv, tg, tb, ts, args, mi != nullptr ? &ti : nullptr);
    b.Copy(qq, out.data(), mo, out.size() * sizeof(float));
    b.Copy(qq, st.data(), ms, st.size() * sizeof(float));
    b.Free(mq);
    b.Free(mk);
    b.Free(mv);
    b.Free(mg);
    b.Free(mb);
    b.Free(ms);
    b.Free(mo);
    if (mi != nullptr) b.Free(mi);
  };

  // Inputs on-manifold: l2-normalized q/k, log-decay g, (0.5,1.25] beta.
  auto gen = [&](uint32_t seed, int64_t B, int64_t Hk, int64_t Hv,
                 std::vector<float>& q, std::vector<float>& k, std::vector<float>& v,
                 std::vector<float>& g, std::vector<float>& beta) {
    uint32_t s = seed;
    q.assign(static_cast<size_t>(B * Hk * Dk), 0.0f);
    k.assign(static_cast<size_t>(B * Hk * Dk), 0.0f);
    v.assign(static_cast<size_t>(B * Hv * Dv), 0.0f);
    g.assign(static_cast<size_t>(B * Hv), 0.0f);
    beta.assign(static_cast<size_t>(B * Hv), 0.0f);
    for (float& x : q) x = GdnLcg(s);
    for (float& x : k) x = GdnLcg(s);
    for (float& x : v) x = GdnLcg(s);
    for (float& x : g) x = 0.24f * GdnLcg(s);
    for (float& x : beta) x = 0.75f + 0.5f * GdnLcg(s);
    for (int64_t r = 0; r < B * Hk; ++r) {
      float ss = 0.0f;
      for (int64_t j = 0; j < Dk; ++j) {
        const float x = q[static_cast<size_t>(r * Dk + j)];
        ss += x * x;
      }
      const float inv = 1.0f / std::sqrt(ss + 1e-6f);
      for (int64_t j = 0; j < Dk; ++j) q[static_cast<size_t>(r * Dk + j)] *= inv;
      ss = 0.0f;
      for (int64_t j = 0; j < Dk; ++j) {
        const float x = k[static_cast<size_t>(r * Dk + j)];
        ss += x * x;
      }
      for (int64_t j = 0; j < Dk; ++j) k[static_cast<size_t>(r * Dk + j)] *= inv;
    }
  };

  // --- Compact form: B in {1,3}, GQA 4:1 and 1:1. Out + updated state.
  for (int64_t B : {int64_t{1}, int64_t{3}}) {
    for (auto [Hk, Hv] : {std::pair<int64_t, int64_t>{2, 8}, {2, 2}}) {
      std::vector<float> q, k, v, g, beta;
      gen(71000u + static_cast<uint32_t>(B * 131 + Hk * 7 + Hv), B, Hk, Hv, q, k, v, g, beta);
      std::vector<float> st(static_cast<size_t>(B * Hv * Dv * Dk));
      {
        uint32_t s = 72000u + static_cast<uint32_t>(B);
        for (float& x : st) x = 0.05f * GdnLcg(s);
      }
      std::vector<float> st_cpu = st, st_tt = st;
      std::vector<float> out_cpu(static_cast<size_t>(B * Hv * Dv), 0.0f),
          out_tt(static_cast<size_t>(B * Hv * Dv), 0.0f);
      step(cpu, DeviceType::kCPU, B, Hk, Hv, q, k, v, g, beta, st_cpu, nullptr, out_cpu);
      step(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, B, Hk,
           Hv, q, k, v, g, beta, st_tt, nullptr, out_tt);
      // Device f32 compute path: envelope stated per measurement (Evidence).
      const float tol = 0.02f;
      GdnDiffStats d = CompareVsOracle(out_tt, out_cpu, 0.0f, tol);
      GdnDiffStats ds = CompareVsOracle(st_tt, st_cpu, 0.0f, tol);
      MESSAGE("kGdnDecode B=", B, " Hk=", Hk, " Hv=", Hv,
              ": out max_abs=", d.max_abs, " max_rel=", d.max_rel,
              " state max_abs=", ds.max_abs, " tol=", tol);
      CHECK(std::isfinite(d.max_abs));
      CHECK(d.within);
      CHECK(std::isfinite(ds.max_abs));
      CHECK(ds.within);
    }
  }

  // --- Large-dynamic-range state (fidelity arm): the real APEX prefill-final
  // GDN state carries mixed O(10^2)/O(10^-3) elements, and on that
  // distribution the composed rank-1 step's matmuls (run without an f32
  // ComputeConfig) lose 10-30% per layer. The small-range arm above hides the
  // defect inside its 2% elementwise envelope. Here the state draws
  // ±10^uniform(-3,2) and — critically — v is built ON MANIFOLD: the delta
  // rule's correction v - S@k is deliberately small in the real model (the
  // state has already absorbed the stream), so the reduced-precision error in
  // dot = S@k (relative ~2^-11 under tf32 MACs) stops being a rounding footnote
  // and becomes a first-order error in v' = (v-dot)*beta, and hence in the
  // committed state. The gate is a tight rel-RMS envelope on the state (0.2%;
  // the f32-safe chunked adapter measures ~0.01% on identical inputs).
  {
    const int64_t B = 1, Hk = 2, Hv = 8;
    std::vector<float> q, k, v, g, beta;
    gen(75000u, B, Hk, Hv, q, k, v, g, beta);
    std::vector<float> st(static_cast<size_t>(B * Hv * Dv * Dk));
    {
      uint32_t s = 76000u;
      for (float& x : st) {
        const float mag =
            std::pow(10.0f, -3.0f + 5.0f * (GdnLcg(s) + 0.5f));  // 10^[-3, 2]
        x = (GdnLcg(s) < 0.0f ? -1.0f : 1.0f) * mag;
      }
    }
    // On-manifold v: v[b,h,r] = (S[b,h,r,:] @ k[b,0,:]) * (1 + eps), eps ~ 1%
    // — the stream the state has already absorbed, plus a fresh-token
    // correction. Host-side in double so only the DEVICE's dot precision is
    // under test.
    for (int64_t b = 0; b < B; ++b)
      for (int64_t h = 0; h < Hv; ++h) {
        uint32_t s = static_cast<uint32_t>(77000u + b * 17 + h);
        for (int64_t r = 0; r < Dv; ++r) {
          double dot = 0.0;
          for (int64_t j = 0; j < Dk; ++j)
            dot += static_cast<double>(
                       st[static_cast<size_t>(((b * Hv) + h) * Dv * Dk + r * Dk + j)]) *
                   static_cast<double>(k[static_cast<size_t>((b * Hk + h / (Hv / Hk)) * Dk + j)]);
          v[static_cast<size_t>((b * Hv + h) * Dv + r)] =
              static_cast<float>(dot) * (1.0f + 0.01f * (2.0f * GdnLcg(s)));
        }
      }
    std::vector<float> st_cpu = st, st_tt = st;
    std::vector<float> out_cpu(static_cast<size_t>(B * Hv * Dv), 0.0f),
        out_tt(static_cast<size_t>(B * Hv * Dv), 0.0f);
    step(cpu, DeviceType::kCPU, B, Hk, Hv, q, k, v, g, beta, st_cpu, nullptr, out_cpu);
    step(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, B, Hk,
         Hv, q, k, v, g, beta, st_tt, nullptr, out_tt);
    auto rel_rms = [](const std::vector<float>& got, const std::vector<float>& ref) {
      double num = 0.0, den = 0.0;
      for (size_t i = 0; i < ref.size(); ++i) {
        const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
        num += d * d;
        den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
      }
      return std::sqrt(num) / std::sqrt(den + 1e-30);
    };
    const double st_rr = rel_rms(st_tt, st_cpu);
    const double out_rr = rel_rms(out_tt, out_cpu);
    // Diagnostics: where the reduced-precision error lands (small elements).
    GdnDiffStats d_small = [&] {
      GdnDiffStats d;
      for (size_t i = 0; i < st_cpu.size(); ++i) {
        if (std::fabs(st_cpu[i]) >= 0.1f) continue;
        const float a = std::fabs(st_tt[i] - st_cpu[i]);
        d.max_abs = std::max(d.max_abs, a);
        if (std::fabs(st_cpu[i]) > 1e-5f)
          d.max_rel = std::max(d.max_rel, a / std::fabs(st_cpu[i]));
      }
      return d;
    }();
    // Tight elementwise envelope on the state: 0.2% relative + a 1e-5 abs
    // floor (f32 compute agrees with the scalar f32 oracle at ~1e-6 relative;
    // the chunked f32-safe adapter measures ~0.01% on identical inputs). A
    // whole-state rel-RMS gate would NOT catch the defect: the reduced-
    // precision error lands in the small-magnitude elements (the correction
    // signal), and the large elements dominate the RMS.
    const float tol = 0.002f, abs_floor = 1e-5f;
    GdnDiffStats ds = CompareVsOracle(st_tt, st_cpu, tol, abs_floor);
    MESSAGE("kGdnDecode wide-range state: state rel_rms=", st_rr,
            " out rel_rms=", out_rr, " max_rel(|ref|<0.1)=",
            d_small.max_rel, " state tol=", tol, " abs_floor=", abs_floor,
            " state max_abs=", ds.max_abs, " max_rel=", ds.max_rel);
    CHECK(std::isfinite(ds.max_abs));
    CHECK(ds.within);
  }

  // --- Indexed form: state is the FULL cache; slot idx[bt] per token.
  {
    const int64_t B = 3, Hk = 2, Hv = 8, slots = 5;
    std::vector<float> q, k, v, g, beta;
    gen(73000u, B, Hk, Hv, q, k, v, g, beta);
    std::vector<float> cache(static_cast<size_t>(slots * Hv * Dv * Dk));
    {
      uint32_t s = 74000u;
      for (float& x : cache) x = 0.05f * GdnLcg(s);
    }
    const std::vector<int32_t> idx{4, 0, 2};
    std::vector<float> ca_cpu = cache, ca_tt = cache;
    std::vector<float> out_cpu(static_cast<size_t>(B * Hv * Dv), 0.0f),
        out_tt(static_cast<size_t>(B * Hv * Dv), 0.0f);
    step(cpu, DeviceType::kCPU, B, Hk, Hv, q, k, v, g, beta, ca_cpu, &idx, out_cpu);
    step(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, B, Hk, Hv,
         q, k, v, g, beta, ca_tt, &idx, out_tt);
    const float tol = 0.02f;
    GdnDiffStats d = CompareVsOracle(out_tt, out_cpu, 0.0f, tol);
    GdnDiffStats ds = CompareVsOracle(ca_tt, ca_cpu, 0.0f, tol);
    MESSAGE("kGdnDecode indexed: out max_abs=", d.max_abs, " cache max_abs=", ds.max_abs,
            " tol=", tol);
    CHECK(std::isfinite(d.max_abs));
    CHECK(d.within);
    CHECK(std::isfinite(ds.max_abs));
    CHECK(ds.within);
    // NULL slot (idx<0): the oracle ZEROES that out row and skips the state;
    // rows 0/2/4 of the cache stay untouched.
    const std::vector<int32_t> idx_null{1, -1, 3};
    std::vector<float> ca2_cpu = cache, ca2_tt = cache;
    step(cpu, DeviceType::kCPU, B, Hk, Hv, q, k, v, g, beta, ca2_cpu, &idx_null, out_cpu);
    step(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, B, Hk, Hv,
         q, k, v, g, beta, ca2_tt, &idx_null, out_tt);
    GdnDiffStats dn = CompareVsOracle(out_tt, out_cpu, 0.0f, tol);
    GdnDiffStats dsn = CompareVsOracle(ca2_tt, ca2_cpu, 0.0f, tol);
    MESSAGE("kGdnDecode indexed NULL: out max_abs=", dn.max_abs,
            " cache max_abs=", dsn.max_abs);
    CHECK(std::isfinite(dn.max_abs));
    CHECK(dn.within);
    CHECK(std::isfinite(dsn.max_abs));
    CHECK(dsn.within);
    // The NULL row is explicitly zeroed by the oracle — not left stale.
    for (int64_t e = 0; e < Hv * Dv; ++e)
      CHECK(out_tt[static_cast<size_t>(1 * Hv * Dv + e)] == 0.0f);
  }

  // --- W2a: the PRODUCTION state-row width Hv*Dk*Dv = 16*128*128 = 262144
  // through the exact indexed scatter, WITH a NULL row (so the live-row
  // compaction feeds the column-chunked launch). Without ScatterRowsExact's
  // chunking this throws "dataflow buffers ... beyond max L1 size of
  // 1572864 B"; with it, out AND cache must match the oracle within the same
  // envelope as every other arm.
  {
    const int64_t B = 2, Hk = 16, Hv = 16, slots = 3;
    std::vector<float> q, k, v, g, beta;
    gen(77000u, B, Hk, Hv, q, k, v, g, beta);
    std::vector<float> cache(static_cast<size_t>(slots * Hv * Dv * Dk));
    {
      uint32_t s = 78000u;
      for (float& x : cache) x = 0.05f * GdnLcg(s);
    }
    const std::vector<int32_t> idx{2, -1};
    std::vector<float> ca_cpu = cache, ca_tt = cache;
    std::vector<float> out_cpu(static_cast<size_t>(B * Hv * Dv), 0.0f),
        out_tt(static_cast<size_t>(B * Hv * Dv), 0.0f);
    step(cpu, DeviceType::kCPU, B, Hk, Hv, q, k, v, g, beta, ca_cpu, &idx, out_cpu);
    step(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, B, Hk,
         Hv, q, k, v, g, beta, ca_tt, &idx, out_tt);
    const float tol = 0.02f;
    GdnDiffStats d = CompareVsOracle(out_tt, out_cpu, 0.0f, tol);
    GdnDiffStats ds = CompareVsOracle(ca_tt, ca_cpu, 0.0f, tol);
    MESSAGE("kGdnDecode wide-state (row 262144) + NULL: out max_abs=", d.max_abs,
            " cache max_abs=", ds.max_abs, " tol=", tol);
    CHECK(std::isfinite(d.max_abs));
    CHECK(d.within);
    CHECK(std::isfinite(ds.max_abs));
    CHECK(ds.within);
  }

  // --- Chained steps on the SAME state buffer + traffic counters: the state
  // must stay device-resident across steps (one upload, zero downloads).
  {
    const int64_t B = 2, Hk = 2, Hv = 8;
    std::vector<float> q0, k0, v0, g0, b0;
    gen(75000u, B, Hk, Hv, q0, k0, v0, g0, b0);
    std::vector<float> st(static_cast<size_t>(B * Hv * Dv * Dk));
    {
      uint32_t s = 76000u;
      for (float& x : st) x = 0.05f * GdnLcg(s);
    }
    std::vector<float> out(static_cast<size_t>(B * Hv * Dv));
    Backend& tt = *vt::TryGetBackend(DeviceType::kTENSTORRENT);
    // ONE device state buffer across all steps: the shadow keys on the host
    // pointer, so a fresh Alloc per step would legitimately re-upload.
    void* mq = tt.Alloc(q0.size() * sizeof(float));
    void* mk = tt.Alloc(k0.size() * sizeof(float));
    void* mv = tt.Alloc(v0.size() * sizeof(float));
    void* mg = tt.Alloc(g0.size() * sizeof(float));
    void* mb = tt.Alloc(b0.size() * sizeof(float));
    void* ms = tt.Alloc(st.size() * sizeof(float));
    void* mo = tt.Alloc(out.size() * sizeof(float));
    Queue qq = tt.CreateQueue();
    tt.Copy(qq, ms, st.data(), st.size() * sizeof(float));  // the ONE upload
    Tensor ts = Tensor::Contiguous(ms, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                   {B, Hv, Dv, Dk});
    Tensor to = Tensor::Contiguous(mo, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                   {B, Hv, Dv});
    vt::GdnArgs args;
    args.scale = 1.0f / std::sqrt(static_cast<float>(Dk));
    vt::tenstorrent::ResetGdnShadowTraffic();
    for (int i = 0; i < 3; ++i) {
      // perturb the token inputs per step (g/beta stay on-manifold)
      std::vector<float> q = q0, k = k0, v = v0, g = g0, beta = b0;
      for (float& x : q) x += 0.01f * i;
      for (float& x : k) x += 0.01f * i;
      for (float& x : v) x += 0.01f * i;
      tt.Copy(qq, mq, q.data(), q.size() * sizeof(float));
      tt.Copy(qq, mk, k.data(), k.size() * sizeof(float));
      tt.Copy(qq, mv, v.data(), v.size() * sizeof(float));
      tt.Copy(qq, mg, g.data(), g.size() * sizeof(float));
      tt.Copy(qq, mb, beta.data(), beta.size() * sizeof(float));
      Tensor tq = Tensor::Contiguous(mq, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                     {B, Hk, Dk});
      Tensor tk = Tensor::Contiguous(mk, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                     {B, Hk, Dk});
      Tensor tv = Tensor::Contiguous(mv, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                     {B, Hv, Dv});
      Tensor tg = Tensor::Contiguous(mg, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                     {B, Hv});
      Tensor tb = Tensor::Contiguous(mb, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                     {B, Hv});
      vt::GdnDecode(qq, to, tq, tk, tv, tg, tb, ts, args, nullptr);
    }
    tt.Copy(qq, st.data(), ms, st.size() * sizeof(float));  // explicit readback only
    tt.Free(mq);
    tt.Free(mk);
    tt.Free(mv);
    tt.Free(mg);
    tt.Free(mb);
    tt.Free(ms);
    tt.Free(mo);
    const auto tr = vt::tenstorrent::GetGdnShadowTraffic();
    const uint64_t want_up = static_cast<uint64_t>(st.size()) * sizeof(float);
    MESSAGE("kGdnDecode traffic: steps=", tr.decode_steps, " h2d=", tr.state_h2d_bytes,
            " d2h=", tr.state_d2h_bytes, " (state bytes=", want_up, ")");
    CHECK(tr.decode_steps == 3);
    CHECK(tr.state_h2d_bytes == want_up);  // exactly ONE upload across 3 steps
    CHECK(tr.state_d2h_bytes == 0);
  }

  // --- bf16 ssm_state (production mamba_ssm_dtype default = conv dtype):
  // the SupportsCompressedGdnState arm. The contract is CUDA bf16 STORAGE
  // semantics — each step loads the bf16 state into f32 registers, computes,
  // and STORES back to bf16 ("read/written in f32 registers",
  // cuda_backend.cu) — not a persistent f32 recurrence. The reference here is
  // the TT f32-state path EMULATING that boundary: after every step the f32
  // state is rounded to bf16 bits and widened back before the next step. The
  // bf16-state path must match it — the shadow may keep f32 tiles internally,
  // but the values that RE-ENTER the next step must be the stored bf16 ones.
  {
    const int64_t B = 2, Hk = 2, Hv = 2, slots = 3;
    const std::vector<int32_t> idx{2, 0};
    vt::GdnArgs args;
    args.scale = 1.0f / std::sqrt(static_cast<float>(Dk));
    Backend& tt = *vt::TryGetBackend(DeviceType::kTENSTORRENT);
    const size_t st_n = static_cast<size_t>(slots * Hv * Dv * Dk);
    std::vector<float> st0(st_n);
    {
      uint32_t s = 77000u;
      for (float& x : st0) x = 0.05f * GdnLcg(s);  // full f32 mantissa
    }
    std::vector<uint16_t> bits0(st_n);
    for (size_t i = 0; i < st_n; ++i) bits0[i] = vt::F32ToBF16(st0[i]);

    // Emulation: f32 buffer, per-step round-trip through bf16 bits.
    std::vector<float> emu_st(st_n);
    for (size_t i = 0; i < st_n; ++i) emu_st[i] = vt::BF16ToF32(bits0[i]);
    std::vector<std::vector<float>> out_emu(3);  // emulation out per step
    void* mq = tt.Alloc(static_cast<size_t>(B * Hk * Dk) * sizeof(float));
    void* mk = tt.Alloc(static_cast<size_t>(B * Hk * Dk) * sizeof(float));
    void* mv = tt.Alloc(static_cast<size_t>(B * Hv * Dv) * sizeof(float));
    void* mg = tt.Alloc(static_cast<size_t>(B * Hv) * sizeof(float));
    void* mb = tt.Alloc(static_cast<size_t>(B * Hv) * sizeof(float));
    void* mo = tt.Alloc(static_cast<size_t>(B * Hv * Dv) * sizeof(float));
    void* mi = tt.Alloc(idx.size() * sizeof(int32_t));
    void* ms_f32 = tt.Alloc(st_n * sizeof(float));
    void* ms_bf16 = tt.Alloc(st_n * sizeof(uint16_t));
    Queue qq = tt.CreateQueue();
    tt.Copy(qq, mi, idx.data(), idx.size() * sizeof(int32_t));
    tt.Copy(qq, ms_bf16, bits0.data(), st_n * sizeof(uint16_t));  // the ONE upload
    Tensor tidx = Tensor::Contiguous(mi, vt::DType::kI32,
                                     Device{DeviceType::kTENSTORRENT, 0},
                                     {static_cast<int64_t>(idx.size())});
    auto step_tensors = [&](void* ms, vt::DType sdt, Tensor& tq, Tensor& tk,
                            Tensor& tv, Tensor& tg, Tensor& tb, Tensor& ts,
                            Tensor& to) {
      tq = Tensor::Contiguous(mq, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                              {B, Hk, Dk});
      tk = Tensor::Contiguous(mk, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                              {B, Hk, Dk});
      tv = Tensor::Contiguous(mv, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                              {B, Hv, Dv});
      tg = Tensor::Contiguous(mg, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                              {B, Hv});
      tb = Tensor::Contiguous(mb, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                              {B, Hv});
      ts = Tensor::Contiguous(ms, sdt, Device{DeviceType::kTENSTORRENT, 0},
                              {slots, Hv, Dv, Dk});
      to = Tensor::Contiguous(mo, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                              {B, Hv, Dv});
    };
    for (int step_i = 0; step_i < 3; ++step_i) {
      std::vector<float> q, k, v, g, beta;
      gen(78000u + static_cast<uint32_t>(step_i * 17), B, Hk, Hv, q, k, v, g, beta);
      tt.Copy(qq, mq, q.data(), q.size() * sizeof(float));
      tt.Copy(qq, mk, k.data(), k.size() * sizeof(float));
      tt.Copy(qq, mv, v.data(), v.size() * sizeof(float));
      tt.Copy(qq, mg, g.data(), g.size() * sizeof(float));
      tt.Copy(qq, mb, beta.data(), beta.size() * sizeof(float));
      // Emulation step (f32 buffer + host-side bf16 round of the state).
      tt.Copy(qq, ms_f32, emu_st.data(), st_n * sizeof(float));
      {
        Tensor tq, tk, tv, tg, tb, ts, to;
        step_tensors(ms_f32, vt::DType::kF32, tq, tk, tv, tg, tb, ts, to);
        vt::GdnDecode(qq, to, tq, tk, tv, tg, tb, ts, args, &tidx);
        out_emu[static_cast<size_t>(step_i)].assign(
            static_cast<size_t>(B * Hv * Dv), 0.0f);
        tt.Copy(qq, out_emu[static_cast<size_t>(step_i)].data(), mo,
                out_emu[static_cast<size_t>(step_i)].size() * sizeof(float));
        tt.Copy(qq, emu_st.data(), ms_f32, st_n * sizeof(float));
        for (size_t i = 0; i < st_n; ++i)
          emu_st[i] = vt::BF16ToF32(vt::F32ToBF16(emu_st[i]));  // the store boundary
      }
      // bf16-state step on the PERSISTENT buffer (shadow fast path).
      std::vector<float> out_tt(static_cast<size_t>(B * Hv * Dv), 0.0f);
      {
        Tensor tq, tk, tv, tg, tb, ts, to;
        step_tensors(ms_bf16, vt::DType::kBF16, tq, tk, tv, tg, tb, ts, to);
        vt::GdnDecode(qq, to, tq, tk, tv, tg, tb, ts, args, &tidx);
        tt.Copy(qq, out_tt.data(), mo, out_tt.size() * sizeof(float));
      }
      // Both paths fed IDENTICAL f32 inputs and identical state VALUES (the
      // bf16 path's shadow rounds exactly where the emulation rounds), so the
      // deterministic device kernels must produce bit-identical outs.
      GdnDiffStats d =
          CompareVsOracle(out_tt, out_emu[static_cast<size_t>(step_i)], 0.0f, 0.0f);
      MESSAGE("kGdnDecode bf16-state step=", step_i,
              ": out max_abs=", d.max_abs, " (0 = storage semantics honored)");
      CHECK(d.max_abs == 0.0f);
    }
    // Final stored truth: the bf16 buffer's bits must equal the emulation's
    // rounded bits exactly.
    std::vector<uint16_t> got_bits(st_n, 0);
    tt.Copy(qq, got_bits.data(), ms_bf16, st_n * sizeof(uint16_t));
    size_t mism = 0;
    for (size_t i = 0; i < st_n; ++i)
      if (got_bits[i] != vt::F32ToBF16(emu_st[i])) ++mism;
    MESSAGE("kGdnDecode bf16-state: cache bit mismatches=", mism, "/", st_n);
    CHECK(mism == 0);
    for (void* m : {mq, mk, mv, mg, mb, mo, mi, ms_f32, ms_bf16}) tt.Free(m);
  }
}

TEST_CASE("kTENSTORRENT kGdnPrefill<->kGdnDecode round-trip (final states agree, both arms, traffic)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& cpu = *vt::TryGetBackend(DeviceType::kCPU);
  constexpr int64_t Dk = 128, Dv = 128;
  struct Case {
    int64_t T, Hk, Hv;
  };
  const std::vector<Case> cases{{3, 2, 8},  {64, 2, 8}, {65, 2, 8},
                                {200, 2, 8}, {65, 2, 2}};

  for (const Case& cs : cases) {
    const int64_t T = cs.T, Hk = cs.Hk, Hv = cs.Hv, B = 1;
    vt::GdnArgs args;
    args.scale = 1.0f / std::sqrt(static_cast<float>(Dk));
    std::vector<float> q(static_cast<size_t>(T * Hk * Dk)), k(static_cast<size_t>(T * Hk * Dk)),
        v(static_cast<size_t>(T * Hv * Dv)), g(static_cast<size_t>(T * Hv)),
        beta(static_cast<size_t>(T * Hv)), st0(static_cast<size_t>(B * Hv * Dv * Dk));
    {
      uint32_t s = 81000u + static_cast<uint32_t>(T * 13 + Hk * 7 + Hv);
      for (float& x : q) x = GdnLcg(s);
      for (float& x : k) x = GdnLcg(s);
      for (float& x : v) x = GdnLcg(s);
      for (float& x : g) x = 0.24f * GdnLcg(s);
      for (float& x : beta) x = 0.75f + 0.5f * GdnLcg(s);
      for (float& x : st0) x = 0.05f * GdnLcg(s);
      auto l2rows = [&](std::vector<float>& t, int64_t heads) {
        for (int64_t r = 0; r < T * heads; ++r) {
          float ss = 0.0f;
          for (int64_t j = 0; j < Dk; ++j) {
            const float x = t[static_cast<size_t>(r * Dk + j)];
            ss += x * x;
          }
          const float inv = 1.0f / std::sqrt(ss + 1e-6f);
          for (int64_t j = 0; j < Dk; ++j) t[static_cast<size_t>(r * Dk + j)] *= inv;
        }
      };
      l2rows(q, Hk);
      l2rows(k, Hk);
    }

    // Prefill arm (W1 kernel) → final state.
    auto prefill = [&](Backend& b, DeviceType dt, std::vector<float>& st,
                       std::vector<float>& out) {
      const std::vector<int32_t> qsl{0, static_cast<int32_t>(T)};
      void* mq = b.Alloc(q.size() * sizeof(float));
      void* mk = b.Alloc(k.size() * sizeof(float));
      void* mv = b.Alloc(v.size() * sizeof(float));
      void* mg = b.Alloc(g.size() * sizeof(float));
      void* mb = b.Alloc(beta.size() * sizeof(float));
      void* ms = b.Alloc(st.size() * sizeof(float));
      void* mx = b.Alloc(qsl.size() * sizeof(int32_t));
      void* mo = b.Alloc(out.size() * sizeof(float));
      Queue qq = b.CreateQueue();
      b.Copy(qq, mq, q.data(), q.size() * sizeof(float));
      b.Copy(qq, mk, k.data(), k.size() * sizeof(float));
      b.Copy(qq, mv, v.data(), v.size() * sizeof(float));
      b.Copy(qq, mg, g.data(), g.size() * sizeof(float));
      b.Copy(qq, mb, beta.data(), beta.size() * sizeof(float));
      b.Copy(qq, ms, st.data(), st.size() * sizeof(float));
      b.Copy(qq, mx, qsl.data(), qsl.size() * sizeof(int32_t));
      Tensor tq = Tensor::Contiguous(mq, vt::DType::kF32, Device{dt, 0}, {T, Hk, Dk});
      Tensor tk = Tensor::Contiguous(mk, vt::DType::kF32, Device{dt, 0}, {T, Hk, Dk});
      Tensor tv = Tensor::Contiguous(mv, vt::DType::kF32, Device{dt, 0}, {T, Hv, Dv});
      Tensor tg = Tensor::Contiguous(mg, vt::DType::kF32, Device{dt, 0}, {T, Hv});
      Tensor tb = Tensor::Contiguous(mb, vt::DType::kF32, Device{dt, 0}, {T, Hv});
      Tensor ts = Tensor::Contiguous(ms, vt::DType::kF32, Device{dt, 0}, {B, Hv, Dv, Dk});
      Tensor tx = Tensor::Contiguous(mx, vt::DType::kI32, Device{dt, 0},
                                     {static_cast<int64_t>(qsl.size())});
      Tensor to = Tensor::Contiguous(mo, vt::DType::kF32, Device{dt, 0}, {T, Hv, Dv});
      vt::GdnPrefill(qq, to, tq, tk, tv, tg, tb, ts, tx, args);
      b.Copy(qq, out.data(), mo, out.size() * sizeof(float));
      b.Copy(qq, st.data(), ms, st.size() * sizeof(float));
      b.Free(mq);
      b.Free(mk);
      b.Free(mv);
      b.Free(mg);
      b.Free(mb);
      b.Free(ms);
      b.Free(mx);
      b.Free(mo);
    };

    // Decode arm: replay the SAME tokens one at a time through kGdnDecode.
    auto replay = [&](Backend& b, DeviceType dt, std::vector<float>& st) {
      std::vector<float> out(static_cast<size_t>(Hv * Dv));
      void* mq = b.Alloc(static_cast<size_t>(Hk * Dk) * sizeof(float));
      void* mk = b.Alloc(static_cast<size_t>(Hk * Dk) * sizeof(float));
      void* mv = b.Alloc(static_cast<size_t>(Hv * Dv) * sizeof(float));
      void* mg = b.Alloc(static_cast<size_t>(Hv) * sizeof(float));
      void* mb = b.Alloc(static_cast<size_t>(Hv) * sizeof(float));
      void* ms = b.Alloc(st.size() * sizeof(float));
      void* mo = b.Alloc(out.size() * sizeof(float));
      Queue qq = b.CreateQueue();
      b.Copy(qq, ms, st.data(), st.size() * sizeof(float));
      Tensor ts = Tensor::Contiguous(ms, vt::DType::kF32, Device{dt, 0}, {B, Hv, Dv, Dk});
      Tensor to = Tensor::Contiguous(mo, vt::DType::kF32, Device{dt, 0}, {B, Hv, Dv});
      for (int64_t t = 0; t < T; ++t) {
        b.Copy(qq, mq, q.data() + static_cast<size_t>(t) * Hk * Dk,
               static_cast<size_t>(Hk * Dk) * sizeof(float));
        b.Copy(qq, mk, k.data() + static_cast<size_t>(t) * Hk * Dk,
               static_cast<size_t>(Hk * Dk) * sizeof(float));
        b.Copy(qq, mv, v.data() + static_cast<size_t>(t) * Hv * Dv,
               static_cast<size_t>(Hv * Dv) * sizeof(float));
        b.Copy(qq, mg, g.data() + static_cast<size_t>(t) * Hv,
               static_cast<size_t>(Hv) * sizeof(float));
        b.Copy(qq, mb, beta.data() + static_cast<size_t>(t) * Hv,
               static_cast<size_t>(Hv) * sizeof(float));
        Tensor tq = Tensor::Contiguous(mq, vt::DType::kF32, Device{dt, 0}, {B, Hk, Dk});
        Tensor tk = Tensor::Contiguous(mk, vt::DType::kF32, Device{dt, 0}, {B, Hk, Dk});
        Tensor tv = Tensor::Contiguous(mv, vt::DType::kF32, Device{dt, 0}, {B, Hv, Dv});
        Tensor tg = Tensor::Contiguous(mg, vt::DType::kF32, Device{dt, 0}, {B, Hv});
        Tensor tb = Tensor::Contiguous(mb, vt::DType::kF32, Device{dt, 0}, {B, Hv});
        vt::GdnDecode(qq, to, tq, tk, tv, tg, tb, ts, args, nullptr);
      }
      b.Copy(qq, st.data(), ms, st.size() * sizeof(float));
      b.Free(mq);
      b.Free(mk);
      b.Free(mv);
      b.Free(mg);
      b.Free(mb);
      b.Free(ms);
      b.Free(mo);
    };

    // CPU arm: prefill vs decode replay must agree BIT-EXACTLY (the oracle
    // runs the same GdnHeadTokenStep instruction sequence either way).
    std::vector<float> st_pf_cpu = st0, st_dec_cpu = st0;
    std::vector<float> out_pf(static_cast<size_t>(T * Hv * Dv));
    prefill(cpu, DeviceType::kCPU, st_pf_cpu, out_pf);
    replay(cpu, DeviceType::kCPU, st_dec_cpu);
    GdnDiffStats dcpu = CompareVsOracle(st_dec_cpu, st_pf_cpu, 0.0f, 0.0f);
    MESSAGE("round-trip CPU T=", T, " Hk=", Hk, " Hv=", Hv,
            ": prefill==decode max_abs=", dcpu.max_abs);
    CHECK(dcpu.max_abs == 0.0f);

    // TT arm: per-op oracle checks for both arms, then cross-kernel.
    std::vector<float> st_pf_tt = st0, st_dec_tt = st0;
    std::vector<float> out_pf_tt(static_cast<size_t>(T * Hv * Dv));
    prefill(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, st_pf_tt,
            out_pf_tt);
    vt::tenstorrent::ResetGdnShadowTraffic();
    replay(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, st_dec_tt);
    const auto tr = vt::tenstorrent::GetGdnShadowTraffic();
    const uint64_t want_up = static_cast<uint64_t>(st0.size()) * sizeof(float);
    MESSAGE("round-trip TT T=", T, ": traffic steps=", tr.decode_steps,
            " h2d=", tr.state_h2d_bytes, " d2h=", tr.state_d2h_bytes,
            " (state bytes=", want_up, ")");
    CHECK(tr.decode_steps == static_cast<uint64_t>(T));
    CHECK(tr.state_h2d_bytes == want_up);  // exactly ONE upload across T steps
    CHECK(tr.state_d2h_bytes == 0);

    GdnDiffStats dpf = CompareVsOracle(st_pf_tt, st_pf_cpu, 0.0f, 0.05f);
    MESSAGE("round-trip TT T=", T, ": prefill state vs CPU max_abs=", dpf.max_abs);
    CHECK(std::isfinite(dpf.max_abs));
    CHECK(dpf.within);
    GdnDiffStats ddec = CompareVsOracle(st_dec_tt, st_dec_cpu, 0.0f, 0.05f);
    MESSAGE("round-trip TT T=", T, ": decode state vs CPU max_abs=", ddec.max_abs);
    CHECK(std::isfinite(ddec.max_abs));
    CHECK(ddec.within);
    // Cross-kernel: TT prefill final state vs TT decode replay final state.
    GdnDiffStats dx = CompareVsOracle(st_dec_tt, st_pf_tt, 0.0f, 0.05f);
    MESSAGE("round-trip TT T=", T, ": decode vs prefill (cross-kernel) max_abs=", dx.max_abs);
    CHECK(std::isfinite(dx.max_abs));
    CHECK(dx.within);
  }
}

// Opt-in decode-step microbench for the composition choice (spec
// tenstorrent-gdn.md, kGdnDecode: composed matmul+eltwise vs one T=1
// chunk_gated_delta_rule call). The kernel mode is selected by
// VT_TT_GDN_DECODE (unset = composed, `chunked` = the T=1 fused call), so run
// this binary twice and compare the ms/step lines. The persistent state
// buffer also re-proves residency under load: warmup builds the shadow, and
// the timed window must then move ZERO state bytes in either direction.
TEST_CASE("kTENSTORRENT kGdnDecode step microbench (opt-in)") {
  if (std::getenv("TT_GDN_BENCH") == nullptr) {
    MESSAGE("SKIPPED: set TT_GDN_BENCH=1 to run the GDN decode microbench");
    return;
  }
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  // Decode-shaped: B=8 single-token sequences, GQA 2:8, Dk=Dv=128.
  const int64_t B = 8, Hk = 2, Hv = 8;
  constexpr int64_t Dk = 128, Dv = 128;
  Backend& tt = *vt::TryGetBackend(DeviceType::kTENSTORRENT);
  std::vector<float> q(static_cast<size_t>(B * Hk * Dk), 0.1f);
  std::vector<float> k(static_cast<size_t>(B * Hk * Dk), 0.1f);
  std::vector<float> v(static_cast<size_t>(B * Hv * Dv), 0.1f);
  std::vector<float> g(static_cast<size_t>(B * Hv), -0.1f);
  std::vector<float> beta(static_cast<size_t>(B * Hv), 1.0f);
  std::vector<float> st(static_cast<size_t>(B * Hv * Dv * Dk), 0.0f);
  std::vector<float> out(static_cast<size_t>(B * Hv * Dv), 0.0f);
  void* mq = tt.Alloc(q.size() * sizeof(float));
  void* mk = tt.Alloc(k.size() * sizeof(float));
  void* mv = tt.Alloc(v.size() * sizeof(float));
  void* mg = tt.Alloc(g.size() * sizeof(float));
  void* mb = tt.Alloc(beta.size() * sizeof(float));
  void* ms = tt.Alloc(st.size() * sizeof(float));
  void* mo = tt.Alloc(out.size() * sizeof(float));
  Queue qq = tt.CreateQueue();
  tt.Copy(qq, mq, q.data(), q.size() * sizeof(float));
  tt.Copy(qq, mk, k.data(), k.size() * sizeof(float));
  tt.Copy(qq, mv, v.data(), v.size() * sizeof(float));
  tt.Copy(qq, mg, g.data(), g.size() * sizeof(float));
  tt.Copy(qq, mb, beta.data(), beta.size() * sizeof(float));
  tt.Copy(qq, ms, st.data(), st.size() * sizeof(float));  // the ONE upload
  Tensor ts =
      Tensor::Contiguous(ms, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {B, Hv, Dv, Dk});
  Tensor to = Tensor::Contiguous(mo, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                 {B, Hv, Dv});
  vt::GdnArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(Dk));
  const char* mode = std::getenv("VT_TT_GDN_DECODE");
  const std::string label =
      (mode != nullptr && std::string_view(mode) == "chunked") ? "chunked" : "composed";
  constexpr int kWarm = 5, kIters = 50;
  for (int i = 0; i < kWarm; ++i) {
    Tensor tq = Tensor::Contiguous(mq, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                   {B, Hk, Dk});
    Tensor tk = Tensor::Contiguous(mk, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                   {B, Hk, Dk});
    Tensor tv = Tensor::Contiguous(mv, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                   {B, Hv, Dv});
    Tensor tg = Tensor::Contiguous(mg, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                   {B, Hv});
    Tensor tb = Tensor::Contiguous(mb, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                   {B, Hv});
    vt::GdnDecode(qq, to, tq, tk, tv, tg, tb, ts, args, nullptr);
  }
  vt::tenstorrent::ResetGdnShadowTraffic();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kIters; ++i) {
    Tensor tq = Tensor::Contiguous(mq, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                   {B, Hk, Dk});
    Tensor tk = Tensor::Contiguous(mk, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                   {B, Hk, Dk});
    Tensor tv = Tensor::Contiguous(mv, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                   {B, Hv, Dv});
    Tensor tg = Tensor::Contiguous(mg, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                   {B, Hv});
    Tensor tb = Tensor::Contiguous(mb, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                   {B, Hv});
    vt::GdnDecode(qq, to, tq, tk, tv, tg, tb, ts, args, nullptr);
  }
  const auto t1 = std::chrono::steady_clock::now();
  const double ms_step =
      std::chrono::duration<double, std::milli>(t1 - t0).count() / static_cast<double>(kIters);
  const auto tr = vt::tenstorrent::GetGdnShadowTraffic();
  const uint64_t want_up = static_cast<uint64_t>(st.size()) * sizeof(float);
  tt.Copy(qq, st.data(), ms, st.size() * sizeof(float));
  tt.Copy(qq, out.data(), mo, out.size() * sizeof(float));
  tt.Free(mq);
  tt.Free(mk);
  tt.Free(mv);
  tt.Free(mg);
  tt.Free(mb);
  tt.Free(ms);
  tt.Free(mo);
  MESSAGE("kGdnDecode step microbench ", label, " B=", B, " Hk=", Hk, " Hv=", Hv,
          " Dk=", Dk, " Dv=", Dv, ": ", ms_step, " ms/step over ", kIters,
          " steps; traffic h2d=", tr.state_h2d_bytes, " d2h=", tr.state_d2h_bytes,
          " (state bytes=", want_up, ")");
  CHECK(ms_step > 0.0);
  CHECK(tr.decode_steps == static_cast<uint64_t>(kIters));
  CHECK(tr.state_h2d_bytes == 0);  // shadow already resident: NO state bytes move
  CHECK(tr.state_d2h_bytes == 0);
}

TEST_CASE("kTENSTORRENT kGdnStateGather/Scatter match the CPU f32 oracle (indexed cache I/O, inverse on live slots)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& cpu = *vt::TryGetBackend(DeviceType::kCPU);
  constexpr int64_t Hv = 8, Dv = 128, Dk = 128, C = 64, W = 3;

  // Row-major contiguous Tensor from a runtime shape (Contiguous takes an
  // initializer_list; the sweep shapes are computed).
  auto make_tensor = [](void* mem, vt::DType dt, Device dev,
                        const std::vector<int64_t>& shape) {
    Tensor t{};
    t.data = mem;
    t.dtype = dt;
    t.device = dev;
    t.rank = static_cast<int32_t>(shape.size());
    int64_t acc = 1;
    for (size_t i = shape.size(); i-- > 0;) {
      t.shape[i] = shape[i];
      t.stride[i] = acc;
      acc *= shape[i];
    }
    return t;
  };

  // Gather rows `idx` from `cache` into `working` (optional has_init zeroing),
  // then scatter `working` back. Compares BOTH sides against the CPU oracle.
  // `cache_row` = elements per cache row (may exceed working's row width).
  auto gather_scatter = [&](const std::vector<int64_t>& cache_shape,
                            const std::vector<int64_t>& work_shape,
                            const std::vector<int32_t>& idx,
                            const std::vector<int32_t>* his, const char* label) {
    const int64_t rows = static_cast<int64_t>(idx.size());
    int64_t cache_elems = 1, work_elems = 1;
    for (int64_t d : cache_shape) cache_elems *= d;
    for (int64_t d : work_shape) work_elems *= d;
    std::vector<float> cache(static_cast<size_t>(cache_elems));
    {
      uint32_t s = 91000u + static_cast<uint32_t>(cache_elems % 9973);
      for (float& x : cache) x = 0.3f * GdnLcg(s);
    }
    auto run = [&](Backend& b, DeviceType dt, std::vector<float>& ca,
                   std::vector<float>& work) {
      void* mc = b.Alloc(ca.size() * sizeof(float));
      void* mw = b.Alloc(work.size() * sizeof(float));
      void* mi = b.Alloc(idx.size() * sizeof(int32_t));
      void* mh = his == nullptr ? nullptr : b.Alloc(his->size() * sizeof(int32_t));
      Queue q = b.CreateQueue();
      b.Copy(q, mc, ca.data(), ca.size() * sizeof(float));
      if (mh != nullptr) b.Copy(q, mh, his->data(), his->size() * sizeof(int32_t));
      b.Copy(q, mi, idx.data(), idx.size() * sizeof(int32_t));
      Tensor tc = make_tensor(mc, vt::DType::kF32, Device{dt, 0}, cache_shape);
      Tensor tw = make_tensor(mw, vt::DType::kF32, Device{dt, 0}, work_shape);
      Tensor ti = Tensor::Contiguous(mi, vt::DType::kI32, Device{dt, 0}, {rows});
      Tensor th{};
      if (mh != nullptr)
        th = Tensor::Contiguous(mh, vt::DType::kI32, Device{dt, 0},
                                {static_cast<int64_t>(his->size())});
      vt::GdnStateGather(q, tw, tc, ti, mh != nullptr ? &th : nullptr);
      b.Copy(q, work.data(), mw, work.size() * sizeof(float));
      vt::GdnStateScatter(q, tc, tw, ti);
      b.Copy(q, ca.data(), mc, ca.size() * sizeof(float));
      b.Free(mc);
      b.Free(mw);
      b.Free(mi);
      if (mh != nullptr) b.Free(mh);
    };
    std::vector<float> ca_cpu = cache, ca_tt = cache;
    std::vector<float> wk_cpu(static_cast<size_t>(work_elems), 0.0f),
        wk_tt(static_cast<size_t>(work_elems), 0.0f);
    run(cpu, DeviceType::kCPU, ca_cpu, wk_cpu);
    run(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, ca_tt, wk_tt);
    GdnDiffStats dw = CompareVsOracle(wk_tt, wk_cpu, 1e-5f, 1e-6f);
    GdnDiffStats dc = CompareVsOracle(ca_tt, ca_cpu, 1e-5f, 1e-6f);
    MESSAGE("kGdnStateGather/Scatter ", label, ": working max_abs=", dw.max_abs,
            " cache max_abs=", dc.max_abs);
    CHECK(std::isfinite(dw.max_abs));
    CHECK(dw.within);
    CHECK(std::isfinite(dc.max_abs));
    CHECK(dc.within);
    // Inverse property on live slots: gather(scatter(w)) == w exactly on both
    // arms (a 0/1 one-hot row copy is exact), and the scatter round-trip left
    // the cache rows bit-equal to the oracle (already checked above).
  };

  // SSM state cache (rank 4, never widened).
  gather_scatter({5, Hv, Dv, Dk}, {3, Hv, Dv, Dk}, {4, 0, 2}, nullptr, "ssm rank-4");
  gather_scatter({5, Hv, Dv, Dk}, {3, Hv, Dv, Dk}, {0, 2, 2}, nullptr, "ssm dup-order");
  // has_initial_state zeroing of fresh rows.
  const std::vector<int32_t> his_vec{1, 0, 1};
  gather_scatter({5, Hv, Dv, Dk}, {3, Hv, Dv, Dk}, {4, 0, 2}, &his_vec, "ssm his");
  // Conv cache (rank 3) and the WIDENED row (leading (K-1) sub-window).
  gather_scatter({5, C, W}, {3, C, W}, {1, 3, 0}, nullptr, "conv rank-3");
  gather_scatter({5, C, W + 2}, {3, C, W}, {1, 3, 0}, nullptr, "conv widened");
  // Rank-2 cache.
  gather_scatter({5, Dk}, {3, Dk}, {2, 2, 4}, nullptr, "rank-2");

  // W2a: a row WIDER than indexed_fill's L1 staging budget. The Qwen3.5 GDN
  // ssm_state row is Hv*Dk*Dv = 262144 elems; without the split-shadow
  // geometry (SplitFactor) the generic interleaved indexed_fill stages
  // 2 x 1 MB of dataflow buffer and throws "beyond max L1 size of 1572864 B".
  // One arm pins all three properties at that width: bit-exact rows,
  // per-(slot, block) last-of-duplicates wins (idx repeats slot 1), and the
  // unnamed slot 2 keeps its bytes.
  gather_scatter({3, 262144}, {2, 262144}, {1, 0}, nullptr,
                 "ssm wide-row (262144 > L1 page budget)");

  // --- Untouched rows: scatter must not write rows no index names.
  {
    const int64_t slots = 5, rows = 2;
    std::vector<float> cache(static_cast<size_t>(slots * Hv * Dv * Dk));
    uint32_t s = 95000u;
    for (float& x : cache) x = 0.3f * GdnLcg(s);
    const std::vector<int32_t> idx{1, 3};
    auto scatter_only = [&](Backend& b, DeviceType dt, std::vector<float>& ca) {
      std::vector<float> work(static_cast<size_t>(rows * Hv * Dv * Dk));
      {
        uint32_t sw = 95100u;
        for (float& x : work) x = 0.3f * GdnLcg(sw);
      }
      void* mc = b.Alloc(ca.size() * sizeof(float));
      void* mw = b.Alloc(work.size() * sizeof(float));
      void* mi = b.Alloc(idx.size() * sizeof(int32_t));
      Queue q = b.CreateQueue();
      b.Copy(q, mc, ca.data(), ca.size() * sizeof(float));
      b.Copy(q, mw, work.data(), work.size() * sizeof(float));
      b.Copy(q, mi, idx.data(), idx.size() * sizeof(int32_t));
      Tensor tc = Tensor::Contiguous(mc, vt::DType::kF32, Device{dt, 0},
                                     {slots, Hv, Dv, Dk});
      Tensor tw = Tensor::Contiguous(mw, vt::DType::kF32, Device{dt, 0},
                                     {rows, Hv, Dv, Dk});
      Tensor ti = Tensor::Contiguous(mi, vt::DType::kI32, Device{dt, 0}, {rows});
      vt::GdnStateScatter(q, tc, tw, ti);
      b.Copy(q, ca.data(), mc, ca.size() * sizeof(float));
      b.Free(mc);
      b.Free(mw);
      b.Free(mi);
    };
    std::vector<float> ca_cpu = cache, ca_tt = cache;
    scatter_only(cpu, DeviceType::kCPU, ca_cpu);
    scatter_only(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT,
                 ca_tt);
    GdnDiffStats d = CompareVsOracle(ca_tt, ca_cpu, 1e-5f, 1e-6f);
    MESSAGE("kGdnStateScatter untouched rows: max_abs=", d.max_abs);
    CHECK(std::isfinite(d.max_abs));
    CHECK(d.within);
  }
}

TEST_CASE("kTENSTORRENT GDN edge shapes (all-empty prefill early return, empty decode batch, empty gather)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& cpu = *vt::TryGetBackend(DeviceType::kCPU);
  constexpr int64_t Hk = 2, Hv = 8, Dk = 128, Dv = 128, C = 64, K = 4;

  // Manual contiguous construction that ALLOWS zero dims: Tensor::Contiguous
  // refuses them (tensor.cpp "shape dims must be positive") but the facades'
  // CheckGdnCommon/CheckConvCommon accept a 0-batch and the kernels
  // early-return on it — the empty arms are reachable through the ABI, just
  // not through that one constructor.
  auto mt = [](void* mem, vt::DType dt, Device dev,
               std::initializer_list<int64_t> shape) {
    Tensor t{};
    t.data = mem;
    t.dtype = dt;
    t.device = dev;
    t.rank = static_cast<int32_t>(shape.size());
    std::vector<int64_t> dims(shape);
    int64_t acc = 1;
    for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d) {
      t.shape[d] = dims[static_cast<size_t>(d)];
      t.stride[d] = acc;
      acc *= dims[static_cast<size_t>(d)];
    }
    return t;
  };

  // ALL-EMPTY prefill (total==0): pins the W1 early return — no out rows, the
  // state bytes stay EXACTLY where the caller left them (sentinel-checked).
  {
    const std::vector<int32_t> qsl{0, 0, 0, 0};  // four marks: three empty seqs
    const int64_t N = 3;
    std::vector<float> st(static_cast<size_t>(N * Hv * Dv * Dk));
    for (size_t i = 0; i < st.size(); ++i) st[i] = 0.25f;  // sentinel
    auto run = [&](Backend& b, DeviceType dt, std::vector<float>& s) {
      std::vector<float> qe, ke, ve, ge, be, out;
      void* mq = b.Alloc(1), *mk = b.Alloc(1), *mv = b.Alloc(1), *mg = b.Alloc(1),
          *mb = b.Alloc(1), *mx = b.Alloc(qsl.size() * sizeof(int32_t)),
          *mo = b.Alloc(1);
      Queue qq = b.CreateQueue();
      b.Copy(qq, mx, qsl.data(), qsl.size() * sizeof(int32_t));
      Tensor tq = mt(mq, vt::DType::kF32, Device{dt, 0}, {0, Hk, Dk});
      Tensor tk = mt(mk, vt::DType::kF32, Device{dt, 0}, {0, Hk, Dk});
      Tensor tv = mt(mv, vt::DType::kF32, Device{dt, 0}, {0, Hv, Dv});
      Tensor tg = mt(mg, vt::DType::kF32, Device{dt, 0}, {0, Hv});
      Tensor tb = mt(mb, vt::DType::kF32, Device{dt, 0}, {0, Hv});
      Tensor ts = Tensor::Contiguous(b.Alloc(s.size() * sizeof(float)), vt::DType::kF32,
                                     Device{dt, 0}, {N, Hv, Dv, Dk});
      void* ms = ts.data;
      b.Copy(qq, ms, s.data(), s.size() * sizeof(float));
      Tensor tx = Tensor::Contiguous(mx, vt::DType::kI32, Device{dt, 0},
                                     {static_cast<int64_t>(qsl.size())});
      Tensor to = mt(mo, vt::DType::kF32, Device{dt, 0}, {0, Hv, Dv});
      vt::GdnArgs args;
      args.scale = 0.088f;
      vt::GdnPrefill(qq, to, tq, tk, tv, tg, tb, ts, tx, args);
      b.Copy(qq, s.data(), ms, s.size() * sizeof(float));
      b.Free(mq);
      b.Free(mk);
      b.Free(mv);
      b.Free(mg);
      b.Free(mb);
      b.Free(ms);
      b.Free(mx);
      b.Free(mo);
    };
    std::vector<float> st_cpu = st, st_tt = st;
    run(cpu, DeviceType::kCPU, st_cpu);
    run(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, st_tt);
    GdnDiffStats d = CompareVsOracle(st_tt, st_cpu, 0.0f, 0.0f);
    MESSAGE("edge all-empty prefill: state max_abs=", d.max_abs);
    CHECK(d.max_abs == 0.0f);
    CHECK(st_tt.front() == 0.25f);  // sentinel survived: nothing touched the state
  }

  // EMPTY decode batch (B==0) on the INDEXED form: the facade demands one
  // compact state row per token, so the no-op pins a REAL full cache row left
  // untouched (idx is empty, state stays [1,...]).
  {
    std::vector<float> st(static_cast<size_t>(Hv * Dv * Dk), 0.5f);
    auto run = [&](Backend& b, DeviceType dt, std::vector<float>& s) {
      void* mq = b.Alloc(1), *mk = b.Alloc(1), *mv = b.Alloc(1), *mg = b.Alloc(1),
          *mb = b.Alloc(1), *mo = b.Alloc(1), *mi = b.Alloc(1);
      void* ms = b.Alloc(s.size() * sizeof(float));
      Queue qq = b.CreateQueue();
      b.Copy(qq, ms, s.data(), s.size() * sizeof(float));
      Tensor tq = mt(mq, vt::DType::kF32, Device{dt, 0}, {0, Hk, Dk});
      Tensor tk = mt(mk, vt::DType::kF32, Device{dt, 0}, {0, Hk, Dk});
      Tensor tv = mt(mv, vt::DType::kF32, Device{dt, 0}, {0, Hv, Dv});
      Tensor tg = mt(mg, vt::DType::kF32, Device{dt, 0}, {0, Hv});
      Tensor tb = mt(mb, vt::DType::kF32, Device{dt, 0}, {0, Hv});
      Tensor ts = Tensor::Contiguous(ms, vt::DType::kF32, Device{dt, 0}, {1, Hv, Dv, Dk});
      Tensor ti = mt(mi, vt::DType::kI32, Device{dt, 0}, {0});
      Tensor to = mt(mo, vt::DType::kF32, Device{dt, 0}, {0, Hv, Dv});
      vt::GdnArgs args;
      args.scale = 0.088f;
      vt::GdnDecode(qq, to, tq, tk, tv, tg, tb, ts, args, &ti);
      b.Copy(qq, s.data(), ms, s.size() * sizeof(float));
      b.Free(mq);
      b.Free(mk);
      b.Free(mv);
      b.Free(mg);
      b.Free(mb);
      b.Free(ms);
      b.Free(mo);
      b.Free(mi);
    };
    std::vector<float> st_cpu = st, st_tt = st;
    run(cpu, DeviceType::kCPU, st_cpu);
    run(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, st_tt);
    MESSAGE("edge empty decode: state untouched=", (st_tt == st));
    CHECK(st_tt == st);
    CHECK(st_cpu == st);
  }

  // EMPTY conv update (B==0) and EMPTY gather (N==0): no-ops.
  {
    std::vector<float> cs(static_cast<size_t>(C * (K - 1)), 0.5f);
    auto run_conv = [&](Backend& b, DeviceType dt, std::vector<float>& s) {
      void* mx = b.Alloc(1), *mw = b.Alloc(K * sizeof(float)), *mi = b.Alloc(1),
            *ms = b.Alloc(s.size() * sizeof(float)), *mo = b.Alloc(1);
      Queue qq = b.CreateQueue();
      b.Copy(qq, ms, s.data(), s.size() * sizeof(float));
      Tensor tx = mt(mx, vt::DType::kF32, Device{dt, 0}, {0, C});
      Tensor tw = Tensor::Contiguous(mw, vt::DType::kF32, Device{dt, 0}, {C, K});
      Tensor ts = Tensor::Contiguous(ms, vt::DType::kF32, Device{dt, 0}, {1, C, K - 1});
      Tensor ti = mt(mi, vt::DType::kI32, Device{dt, 0}, {0});
      Tensor to = mt(mo, vt::DType::kF32, Device{dt, 0}, {0, C});
      vt::CausalConv1dArgs a;
      vt::CausalConv1dUpdate(qq, to, tx, tw, nullptr, ts, a, &ti);
      b.Copy(qq, s.data(), ms, s.size() * sizeof(float));
      b.Free(mx);
      b.Free(mw);
      b.Free(ms);
      b.Free(mo);
      b.Free(mi);
    };
    std::vector<float> cs_cpu = cs, cs_tt = cs;
    run_conv(cpu, DeviceType::kCPU, cs_cpu);
    run_conv(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, cs_tt);
    CHECK(cs_tt == cs);
    CHECK(cs_cpu == cs);

    std::vector<float> cache(static_cast<size_t>(3 * C * (K - 1)), 0.25f);
    auto run_gather = [&](Backend& b, DeviceType dt, std::vector<float>& ca) {
      std::vector<float> work;
      void* mc = b.Alloc(ca.size() * sizeof(float));
      void* mw = b.Alloc(1), *mi = b.Alloc(1);
      Queue qq = b.CreateQueue();
      b.Copy(qq, mc, ca.data(), ca.size() * sizeof(float));
      Tensor tc = Tensor::Contiguous(mc, vt::DType::kF32, Device{dt, 0}, {3, C, K - 1});
      Tensor tw = mt(mw, vt::DType::kF32, Device{dt, 0}, {0, C, K - 1});
      Tensor ti = mt(mi, vt::DType::kI32, Device{dt, 0}, {0});
      vt::GdnStateGather(qq, tw, tc, ti, nullptr);
      vt::GdnStateScatter(qq, tc, tw, ti);
      b.Copy(qq, ca.data(), mc, ca.size() * sizeof(float));
      b.Free(mc);
      b.Free(mw);
      b.Free(mi);
    };
    std::vector<float> ca_cpu = cache, ca_tt = cache;
    run_gather(cpu, DeviceType::kCPU, ca_cpu);
    run_gather(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, ca_tt);
    CHECK(ca_tt == cache);
    CHECK(ca_cpu == cache);
    MESSAGE("edge empty conv/gather: no-ops held");
  }
}

// ==== BACKEND-TENSTORRENT-QWEN35 W1: the Qwen3.5 op delta vs the CPU f32 oracle
// Same doctrine as the GDN block above: identical inputs, both arms run the
// SAME public vt:: op on their own backend, outputs compared. A missing TT
// kernel refuses by name — this suite's red state before the kernel lands.

TEST_CASE("kTENSTORRENT kSigmoidGateBf16 matches the CPU f32 oracle (attn dtype arms)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& cpu = *vt::TryGetBackend(DeviceType::kCPU);
  // Production shape is [T, K] with K = Hq*Dh (qwen3_5.cpp:2546). T=1 is the
  // decode step; 65 crosses the 64-row tile boundary. Gate values span +-8 so
  // the sigmoid input precision is load-bearing: at gate ~ -5 the sigmoid is
  // ~7e-3 and a bf16 pre-round of the gate moves the product by >2% relative —
  // exactly the arm that proves the gate stays f32 (ops.cpp:4140).
  for (int64_t T : {int64_t{1}, int64_t{3}, int64_t{64}, int64_t{65}}) {
    for (int64_t K : {int64_t{128}, int64_t{384}}) {
      for (const vt::DType adt : {vt::DType::kF32, vt::DType::kBF16}) {
        const int64_t n = T * K;
        std::vector<float> attn(static_cast<size_t>(n)), gate(static_cast<size_t>(n));
        {
          uint32_t s = 4001u + static_cast<uint32_t>(T * 7 + K + (adt == vt::DType::kF32 ? 0 : 1));
          for (int64_t i = 0; i < n; ++i) {
            attn[static_cast<size_t>(i)] = 4.0f * GdnLcg(s) + 0.125f;   // full f32 mantissa
            gate[static_cast<size_t>(i)] = 16.0f * GdnLcg(s);           // spans +-8
          }
        }
        auto run = [&](Backend& b, DeviceType dt, std::vector<uint16_t>& out_bf) {
          out_bf.assign(static_cast<size_t>(n), 0);
          const size_t a_bytes = adt == vt::DType::kF32 ? sizeof(float) : sizeof(uint16_t);
          void* ma = b.Alloc(n * a_bytes);
          void* mg = b.Alloc(n * sizeof(float));
          void* mo = b.Alloc(n * sizeof(uint16_t));
          Queue q = b.CreateQueue();
          if (adt == vt::DType::kF32) {
            b.Copy(q, ma, attn.data(), attn.size() * sizeof(float));
          } else {  // bf16 arm holds bf16-representable values (upcast exact)
            std::vector<uint16_t> bits(static_cast<size_t>(n));
            for (int64_t i = 0; i < n; ++i)
              bits[static_cast<size_t>(i)] = vt::F32ToBF16(attn[static_cast<size_t>(i)]);
            b.Copy(q, ma, bits.data(), bits.size() * sizeof(uint16_t));
          }
          b.Copy(q, mg, gate.data(), gate.size() * sizeof(float));
          Tensor ta = Tensor::Contiguous(ma, adt, Device{dt, 0}, {T, K});
          Tensor tg = Tensor::Contiguous(mg, vt::DType::kF32, Device{dt, 0}, {T, K});
          Tensor to = Tensor::Contiguous(mo, vt::DType::kBF16, Device{dt, 0}, {T, K});
          vt::SigmoidGateBf16(q, to, ta, tg);
          b.Copy(q, out_bf.data(), mo, out_bf.size() * sizeof(uint16_t));
          b.Free(ma);
          b.Free(mg);
          b.Free(mo);
        };
        std::vector<uint16_t> out_cpu, out_tt;
        run(cpu, DeviceType::kCPU, out_cpu);
        run(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, out_tt);
        std::vector<float> ref(out_cpu.size()), got(out_tt.size());
        for (size_t i = 0; i < out_cpu.size(); ++i) {
          ref[i] = vt::BF16ToF32(out_cpu[i]);
          got[i] = vt::BF16ToF32(out_tt[i]);
        }
        // Both arms round once to bf16 (RNE); the only TT-vs-CPU delta is the
        // SFPU f32 sigmoid (accurate exp + reciprocal_iter<2) vs std::exp —
        // a few f32 ULP, which can flip at most one bf16 rounding. One bf16
        // ULP is <= 2^-7 relative on any binade boundary, so that is the
        // envelope; abs_floor guards the exact-zero ref.
        const float rel = 1.0f / 128.0f, abs_floor = 1e-6f;
        GdnDiffStats d = CompareVsOracle(got, ref, rel, abs_floor);
        // doctest quirk (this build): MESSAGE streams a `const char*` VARIABLE
        // as its pointer→bool ("1"); only literals and std::string print as
        // text (same fix as kAttnQkNormRopeGate below).
        const std::string adt_name = adt == vt::DType::kF32 ? "f32" : "bf16";
        MESSAGE("kSigmoidGateBf16 T=", T, " K=", K,
                " attn=", adt_name,
                ": max_abs=", d.max_abs, " max_rel=", d.max_rel, " within=", d.within);
        CHECK(d.within);
      }
    }
  }
}

TEST_CASE("kTENSTORRENT kGdnPostConv matches the CPU f32 oracle (fused split + l2norm + g/beta)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& cpu = *vt::TryGetBackend(DeviceType::kCPU);
  const vt::L2NormArgs args{1e-6f};
  // Production geometry (0.8B GDN layers): Hk=Hv=32 heads, Dk=Dv=128, so
  // conv cols = 2*Hk*Dk + Hv*Dv (qwen3_5.cpp:4146). Sweep smaller heads plus
  // the production shape; one config exercises Hk != Hv (the contract admits
  // it even though the model does not use it).
  struct Cfg { int64_t T, Hk, Dk, Hv, Dv; };
  const Cfg cfgs[] = {{1, 2, 128, 2, 128},  {3, 8, 128, 8, 128},
                      {65, 2, 128, 2, 128}, {64, 32, 128, 32, 128},
                      {3, 2, 128, 4, 128}};
  for (const Cfg& c : cfgs) {
    // out_bf16: the PRODUCTION arm — VT_GDN_BF16 defaults the matmul-input
    // activations q/k/v to bf16 (qwen3_5.cpp GdnActDType), while g/beta stay
    // f32. Covers the kernel's per-out typecast on commit.
    for (int out_bf16 : {0, 1}) {
    for (int conv_bf16 : {0, 1}) {
      for (int ab_bf16 : {0, 1}) {
        const int64_t key_dim = c.Hk * c.Dk, value_dim = c.Hv * c.Dv;
        const int64_t conv_dim = 2 * key_dim + value_dim;
        const int64_t n = c.T * conv_dim;
        const int64_t a_stride = c.Hv + 8;  // padded row view (production form)
        // f32 values; the bf16 arms round these to bf16 bits ONCE so both
        // backends read identical bf16-representable inputs.
        std::vector<float> conv_f(n), araw_f(c.T * c.Hv), braw_f(c.T * c.Hv),
            a_log(static_cast<size_t>(c.Hv)), dt_bias(static_cast<size_t>(c.Hv));
        {
          uint32_t s = 9001u + static_cast<uint32_t>(c.T * 11 + c.Hk * 5 + c.Hv +
                                                     conv_bf16 * 2 + ab_bf16 +
                                                     out_bf16 * 41);
          for (float& v : conv_f) v = 2.0f * GdnLcg(s) + 0.0625f;
          // araw spans +-25 so x = araw+dt_bias crosses the softplus
          // threshold-20 branch; braw spans +-8 (sigmoid sensitivity).
          for (float& v : araw_f) v = 50.0f * GdnLcg(s);
          for (float& v : braw_f) v = 16.0f * GdnLcg(s);
          for (float& v : a_log) v = -4.0f * (GdnLcg(s) + 0.5f);   // exp in (0,1]
          for (float& v : dt_bias) v = 4.0f * GdnLcg(s);
        }
        const vt::DType cdt = conv_bf16 ? vt::DType::kBF16 : vt::DType::kF32;
        const vt::DType adt = ab_bf16 ? vt::DType::kBF16 : vt::DType::kF32;
        const vt::DType odt = out_bf16 ? vt::DType::kBF16 : vt::DType::kF32;
        const size_t cb = cdt == vt::DType::kF32 ? sizeof(float) : sizeof(uint16_t);
        const size_t ab = adt == vt::DType::kF32 ? sizeof(float) : sizeof(uint16_t);
        // bf16-bit staging helpers (round once, reuse for both backends).
        auto to_bits = [&](const std::vector<float>& v) {
          std::vector<uint16_t> bits(v.size());
          for (size_t i = 0; i < v.size(); ++i) bits[i] = vt::F32ToBF16(v[i]);
          return bits;
        };
        std::vector<uint16_t> conv_bits = conv_bf16 ? to_bits(conv_f) : std::vector<uint16_t>{};
        // Padded a/b backing buffers, garbage in the pad cols (production form);
        // copied whole so both backends see identical bytes.
        auto pad_ab = [&](const std::vector<float>& live) {
          std::vector<uint16_t> bits(static_cast<size_t>(c.T * a_stride), 0xABCDu);
          for (int64_t t = 0; t < c.T; ++t)
            for (int64_t h = 0; h < c.Hv; ++h)
              bits[static_cast<size_t>(t * a_stride + h)] = vt::F32ToBF16(live[static_cast<size_t>(t * c.Hv + h)]);
          return bits;
        };
        std::vector<uint16_t> a_bits, b_bits;
        std::vector<float> a_pad_f, b_pad_f;  // f32-arm padded buffers
        if (ab_bf16) {
          a_bits = pad_ab(araw_f);
          b_bits = pad_ab(braw_f);
        } else {
          a_pad_f.assign(static_cast<size_t>(c.T * a_stride), std::numeric_limits<float>::quiet_NaN());
          b_pad_f.assign(static_cast<size_t>(c.T * a_stride), std::numeric_limits<float>::quiet_NaN());
          for (int64_t t = 0; t < c.T; ++t)
            for (int64_t h = 0; h < c.Hv; ++h) {
              a_pad_f[static_cast<size_t>(t * a_stride + h)] = araw_f[static_cast<size_t>(t * c.Hv + h)];
              b_pad_f[static_cast<size_t>(t * a_stride + h)] = braw_f[static_cast<size_t>(t * c.Hv + h)];
            }
        }

        std::vector<float> q_cpu(static_cast<size_t>(c.T * key_dim)),
            k_cpu(q_cpu.size()), v_cpu(static_cast<size_t>(c.T * value_dim)),
            g_cpu(static_cast<size_t>(c.T * c.Hv)), beta_cpu(g_cpu.size());
        std::vector<float> q_tt(q_cpu.size()), k_tt(k_cpu.size()), v_tt(v_cpu.size()),
            g_tt(g_cpu.size()), beta_tt(beta_cpu.size());
        auto run = [&](Backend& b, DeviceType dt, std::vector<float>& qo,
                       std::vector<float>& ko, std::vector<float>& vo,
                       std::vector<float>& go, std::vector<float>& bo) {
          void* mc = b.Alloc(n * cb);
          // padded a/b backing (garbage in the pad rows' tail columns)
          void* mab = b.Alloc(static_cast<size_t>(c.T * a_stride) * ab);
          void* mbb = b.Alloc(static_cast<size_t>(c.T * a_stride) * ab);
          void* mal = b.Alloc(a_log.size() * sizeof(float));
          void* md = b.Alloc(dt_bias.size() * sizeof(float));
          void* mq = b.Alloc(qo.size() * (out_bf16 ? sizeof(uint16_t) : sizeof(float)));
          void* mk = b.Alloc(ko.size() * (out_bf16 ? sizeof(uint16_t) : sizeof(float)));
          void* mv = b.Alloc(vo.size() * (out_bf16 ? sizeof(uint16_t) : sizeof(float)));
          void* mg = b.Alloc(go.size() * sizeof(float));
          void* mb = b.Alloc(bo.size() * sizeof(float));
          Queue q = b.CreateQueue();
          if (conv_bf16) b.Copy(q, mc, conv_bits.data(), conv_bits.size() * sizeof(uint16_t));
          else b.Copy(q, mc, conv_f.data(), conv_f.size() * sizeof(float));
          if (ab_bf16) {
            b.Copy(q, mab, a_bits.data(), a_bits.size() * sizeof(uint16_t));
            b.Copy(q, mbb, b_bits.data(), b_bits.size() * sizeof(uint16_t));
          } else {
            b.Copy(q, mab, a_pad_f.data(), a_pad_f.size() * sizeof(float));
            b.Copy(q, mbb, b_pad_f.data(), b_pad_f.size() * sizeof(float));
          }
          b.Copy(q, mal, a_log.data(), a_log.size() * sizeof(float));
          b.Copy(q, md, dt_bias.data(), dt_bias.size() * sizeof(float));
          Tensor tc = Tensor::Contiguous(mc, cdt, Device{dt, 0}, {c.T, conv_dim});
          auto view2 = [&](void* m) {  // [T, Hv] inner-contiguous row view
            Tensor t{};
            t.data = m;
            t.dtype = adt;
            t.device = Device{dt, 0};
            t.rank = 2;
            t.shape[0] = c.T;
            t.shape[1] = c.Hv;
            t.stride[0] = a_stride;
            t.stride[1] = 1;
            return t;
          };
          Tensor ta = view2(mab), tb2 = view2(mbb);
          Tensor tal = Tensor::Contiguous(mal, vt::DType::kF32, Device{dt, 0}, {c.Hv});
          Tensor td = Tensor::Contiguous(md, vt::DType::kF32, Device{dt, 0}, {c.Hv});
          Tensor tq = Tensor::Contiguous(mq, odt, Device{dt, 0}, {c.T, c.Hk, c.Dk});
          Tensor tk = Tensor::Contiguous(mk, odt, Device{dt, 0}, {c.T, c.Hk, c.Dk});
          Tensor tv = Tensor::Contiguous(mv, odt, Device{dt, 0}, {c.T, c.Hv, c.Dv});
          Tensor tg = Tensor::Contiguous(mg, vt::DType::kF32, Device{dt, 0}, {c.T, c.Hv});
          Tensor tbe = Tensor::Contiguous(mb, vt::DType::kF32, Device{dt, 0}, {c.T, c.Hv});
          vt::GdnPostConv(q, tq, tk, tv, tg, tbe, tc, ta, tb2, tal, td, args);
          // bf16 outs download as bits and widen once for the comparator.
          auto read_out = [&](void* m, std::vector<float>& o) {
            if (!out_bf16) {
              b.Copy(q, o.data(), m, o.size() * sizeof(float));
              return;
            }
            std::vector<uint16_t> bits(o.size());
            b.Copy(q, bits.data(), m, bits.size() * sizeof(uint16_t));
            for (size_t i = 0; i < o.size(); ++i)
              o[i] = vt::BF16ToF32(bits[i]);
          };
          read_out(mq, qo);
          read_out(mk, ko);
          read_out(mv, vo);
          b.Copy(q, go.data(), mg, go.size() * sizeof(float));
          b.Copy(q, bo.data(), mb, bo.size() * sizeof(float));
          for (void* m : {mc, mab, mbb, mal, md, mq, mk, mv, mg, mb}) b.Free(m);
        };
        run(cpu, DeviceType::kCPU, q_cpu, k_cpu, v_cpu, g_cpu, beta_cpu);
        run(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, q_tt,
            k_tt, v_tt, g_tt, beta_tt);
        // q/k: bf16 tile l2norm — the kL2Norm envelope (0.02 abs, the row's
        // calibrated bound for unit-vector-scale outputs). bf16 outs add one
        // store round on each side (a compute ULP can flip it: <= 2^-7 rel).
        const float o_rel = out_bf16 ? 1.0f / 128.0f : 0.0f;
        GdnDiffStats dq = CompareVsOracle(q_tt, q_cpu, o_rel, 0.02f);
        GdnDiffStats dk = CompareVsOracle(k_tt, k_cpu, o_rel, 0.02f);
        // v: slice copy through the bf16 tile — exact for bf16 conv inputs,
        // one bf16 round for f32 conv inputs (<= 2^-7 rel on any binade),
        // plus the store round when the out itself is bf16.
        const float v_rel = (conv_bf16 ? 0.0f : 1.0f / 128.0f) + o_rel;
        GdnDiffStats dv = CompareVsOracle(v_tt, v_cpu, v_rel, 1e-6f);
        // g/beta: f32 tiles end to end (softplus threshold-20, exp(a_log),
        // sigmoid) — SFPU f32 vs std::exp/log1p is ULP-level.
        GdnDiffStats dg = CompareVsOracle(g_tt, g_cpu, 1e-4f, 1e-6f);
        GdnDiffStats db = CompareVsOracle(beta_tt, beta_cpu, 1e-4f, 1e-6f);
        // doctest quirk (this build): MESSAGE streams a `const char*` VARIABLE
        // as its pointer→bool ("1"); only literals and std::string print as
        // text (same fix as kAttnQkNormRopeGate).
        const std::string conv_name = conv_bf16 ? "bf16" : "f32";
        const std::string ab_name = ab_bf16 ? "bf16" : "f32";
        const std::string out_name = out_bf16 ? "bf16" : "f32";
        MESSAGE("kGdnPostConv T=", c.T, " Hk=", c.Hk, " Hv=", c.Hv,
                " conv=", conv_name, " ab=", ab_name, " out=", out_name,
                ": q[max_abs=", dq.max_abs, "] k[max_abs=", dk.max_abs,
                "] v[max_abs=", dv.max_abs, "] g[max_abs=", dg.max_abs,
                " max_rel=", dg.max_rel, "] beta[max_abs=", db.max_abs,
                " max_rel=", db.max_rel, "]");
        CHECK(dq.within);
        CHECK(dk.within);
        CHECK(dv.within);
        CHECK(dg.within);
        CHECK(db.within);
      }
    }
    }
  }
}

TEST_CASE("kTENSTORRENT kAttnQkNormRopeGate matches the CPU f32 oracle (fused preamble, GQA + partial rope)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& cpu = *vt::TryGetBackend(DeviceType::kCPU);
  // Production call: q/k/gate f32 outs, gemma=true (qwen3_5.cpp:5206-5224),
  // qgate rows [q(Dh)|gate(Dh)] per head with the merged-QKV row stride, kf
  // rows likewise; rot < Dh exercises the partial-rope tail (normed, not
  // rotated). Hq > Hkv covers GQA.
  struct Cfg { int64_t T, Hq, Hkv, Dh, rot; bool gemma; int in_bf16; int out_bf16; };
  const Cfg cfgs[] = {
      {1, 4, 2, 128, 64, true, 0, 0},   {3, 4, 2, 128, 64, true, 1, 0},
      {3, 4, 2, 128, 128, true, 0, 0},  {65, 4, 2, 128, 64, true, 1, 0},
      {3, 32, 8, 128, 64, true, 1, 0},  {3, 4, 2, 128, 64, false, 0, 0},
      // bf16-out arms — the kernel's per-out typecast branches
      // (tenstorrent_ops.cpp AttnQkNormRopeGateKernel commit legs). bf16 in +
      // bf16 out: the gate passthrough stays exact (rounding a bf16 value is
      // identity). f32 in + bf16 out: every leg rounds once at the store.
      {1, 4, 2, 128, 64, true, 1, 1},  {3, 32, 8, 128, 64, true, 0, 1},
  };
  for (const Cfg& c : cfgs) {
    const vt::RmsNormArgs na{1e-6f, c.gemma};
    const vt::RopeArgs ra{10000.0, static_cast<int>(c.rot)};  // base unused: cos_sin is data
    const int64_t qrow = c.Hq * 2 * c.Dh, krow = c.Hkv * c.Dh;
    const int64_t qgate_pad = 16, kf_pad = 16;
    std::vector<float> qgate_f(static_cast<size_t>(c.T * (qrow + qgate_pad)));
    std::vector<float> kf_f(static_cast<size_t>(c.T * (krow + kf_pad)));
    std::vector<float> qw(static_cast<size_t>(c.Dh)), kw(static_cast<size_t>(c.Dh));
    std::vector<float> cs(static_cast<size_t>(c.T * c.rot));
    {
      uint32_t s = 11001u + static_cast<uint32_t>(c.T * 13 + c.Hq * 7 + c.rot + c.gemma * 3 + c.in_bf16 + c.out_bf16 * 43);
      for (float& v : qgate_f) v = 2.0f * GdnLcg(s);
      for (float& v : kf_f) v = 2.0f * GdnLcg(s);
      for (float& v : qw) v = 0.2f * GdnLcg(s);   // gemma adds 1
      for (float& v : kw) v = 0.2f * GdnLcg(s);
      for (float& v : cs) v = 2.0f * GdnLcg(s);   // cos|sin as data in [-1,1)
    }
    const vt::DType idt = c.in_bf16 ? vt::DType::kBF16 : vt::DType::kF32;
    const size_t ib = idt == vt::DType::kF32 ? sizeof(float) : sizeof(uint16_t);
    std::vector<uint16_t> qgate_bits, kf_bits;
    if (c.in_bf16) {
      qgate_bits.resize(qgate_f.size());
      kf_bits.resize(kf_f.size());
      for (size_t i = 0; i < qgate_f.size(); ++i) qgate_bits[i] = vt::F32ToBF16(qgate_f[i]);
      for (size_t i = 0; i < kf_f.size(); ++i) kf_bits[i] = vt::F32ToBF16(kf_f[i]);
    }
    const int64_t qn = c.T * c.Hq * c.Dh, kn = c.T * c.Hkv * c.Dh;
    std::vector<float> q_cpu(static_cast<size_t>(qn)), k_cpu(static_cast<size_t>(kn)),
        g_cpu(static_cast<size_t>(qn));
    std::vector<float> q_tt(q_cpu.size()), k_tt(k_cpu.size()), g_tt(g_cpu.size());
    auto run = [&](Backend& b, DeviceType dt, std::vector<float>& qo,
                   std::vector<float>& ko, std::vector<float>& go) {
      void* mqg = b.Alloc(qgate_f.size() * ib);
      void* mkf = b.Alloc(kf_f.size() * ib);
      void* mqw = b.Alloc(qw.size() * sizeof(float));
      void* mkw = b.Alloc(kw.size() * sizeof(float));
      void* mcs = b.Alloc(cs.size() * sizeof(float));
      void* mq = b.Alloc(qo.size() * (c.out_bf16 ? sizeof(uint16_t) : sizeof(float)));
      void* mk = b.Alloc(ko.size() * (c.out_bf16 ? sizeof(uint16_t) : sizeof(float)));
      void* mg = b.Alloc(go.size() * (c.out_bf16 ? sizeof(uint16_t) : sizeof(float)));
      Queue q = b.CreateQueue();
      if (c.in_bf16) {
        b.Copy(q, mqg, qgate_bits.data(), qgate_bits.size() * sizeof(uint16_t));
        b.Copy(q, mkf, kf_bits.data(), kf_bits.size() * sizeof(uint16_t));
      } else {
        b.Copy(q, mqg, qgate_f.data(), qgate_f.size() * sizeof(float));
        b.Copy(q, mkf, kf_f.data(), kf_f.size() * sizeof(float));
      }
      b.Copy(q, mqw, qw.data(), qw.size() * sizeof(float));
      b.Copy(q, mkw, kw.data(), kw.size() * sizeof(float));
      b.Copy(q, mcs, cs.data(), cs.size() * sizeof(float));
      auto strided2 = [&](void* m, int64_t rows, int64_t cols, int64_t stride) {
        Tensor t{};
        t.data = m;
        t.dtype = idt;
        t.device = Device{dt, 0};
        t.rank = 2;
        t.shape[0] = rows;
        t.shape[1] = cols;
        t.stride[0] = stride;
        t.stride[1] = 1;
        return t;
      };
      Tensor tqg = strided2(mqg, c.T, qrow, qrow + qgate_pad);
      Tensor tkf = strided2(mkf, c.T, krow, krow + kf_pad);
      Tensor tqw = Tensor::Contiguous(mqw, vt::DType::kF32, Device{dt, 0}, {c.Dh});
      Tensor tkw = Tensor::Contiguous(mkw, vt::DType::kF32, Device{dt, 0}, {c.Dh});
      Tensor tcs = Tensor::Contiguous(mcs, vt::DType::kF32, Device{dt, 0}, {c.T, c.rot});
      const vt::DType odt = c.out_bf16 ? vt::DType::kBF16 : vt::DType::kF32;
      Tensor tq = Tensor::Contiguous(mq, odt, Device{dt, 0}, {c.T, c.Hq, c.Dh});
      Tensor tk = Tensor::Contiguous(mk, odt, Device{dt, 0}, {c.T, c.Hkv, c.Dh});
      Tensor tg = Tensor::Contiguous(mg, odt, Device{dt, 0}, {c.T, c.Hq, c.Dh});
      vt::AttnQkNormRopeGate(q, tq, tk, tg, tqg, tkf, tqw, tkw, tcs, na, ra);
      // bf16 outs download as bits and widen once for the comparator.
      auto read_out = [&](void* m, std::vector<float>& o) {
        if (!c.out_bf16) {
          b.Copy(q, o.data(), m, o.size() * sizeof(float));
          return;
        }
        std::vector<uint16_t> bits(o.size());
        b.Copy(q, bits.data(), m, bits.size() * sizeof(uint16_t));
        for (size_t i = 0; i < o.size(); ++i) o[i] = vt::BF16ToF32(bits[i]);
      };
      read_out(mq, qo);
      read_out(mk, ko);
      read_out(mg, go);
      for (void* m : {mqg, mkf, mqw, mkw, mcs, mq, mk, mg}) b.Free(m);
    };
    run(cpu, DeviceType::kCPU, q_cpu, k_cpu, g_cpu);
    run(*vt::TryGetBackend(DeviceType::kTENSTORRENT), DeviceType::kTENSTORRENT, q_tt, k_tt,
        g_tt);
    // q/k: f32 tile norm + rope (inside the row's RopeApplyDeviceNeox
    // envelope); outputs are O(1) (unit-RMS rows, rotation preserves
    // magnitude). bf16 outs add one store round per side (<= 2^-7 rel).
    // f32 outs carry no store round, so the abs floor is pinned at 1e-3:
    // measured green headroom on P150 is <= 5.3e-4, while an unconditional
    // pre-norm bf16 typecast of q (the ops.cpp:4140 branch this guards)
    // lands at 5.4e-3 — 1e-3 separates both sides.
    const float o_rel = c.out_bf16 ? 1.0f / 128.0f : 0.0f;
    const float o_abs = c.out_bf16 ? 0.02f : 1e-3f;
    GdnDiffStats dq = CompareVsOracle(q_tt, q_cpu, o_rel, o_abs);
    GdnDiffStats dk = CompareVsOracle(k_tt, k_cpu, o_rel, o_abs);
    // gate: EXACT passthrough when it stays f32 or when the inputs are
    // already bf16 (rounding identity); one store round when f32 inputs meet
    // a bf16 out (ops.cpp:1660-1662, 4140 — the sigmoid input must not be
    // rounded BEFORE the gate; the out store is the only legal round).
    const float g_rel = (c.out_bf16 && !c.in_bf16) ? 1.0f / 128.0f : 0.0f;
    GdnDiffStats dg = CompareVsOracle(g_tt, g_cpu, g_rel, 0.0f);
    // doctest quirk (this build): MESSAGE streams a `const char*` VARIABLE as
    // its pointer→bool ("1"); only literals and std::string print as text.
    const std::string in = c.in_bf16 ? "bf16" : "f32";
    const std::string out = c.out_bf16 ? "bf16" : "f32";
    MESSAGE("kAttnQkNormRopeGate T=", c.T, " Hq=", c.Hq, " Hkv=", c.Hkv,
            " rot=", c.rot, " gemma=", c.gemma, " in=", in, " out=", out,
            ": q[max_abs=", dq.max_abs, "] k[max_abs=", dk.max_abs,
            "] gate[max_abs=", dg.max_abs, "]");
    CHECK(dq.within);
    CHECK(dk.within);
    CHECK(dg.within);
  }
}


// SCRATCH (2b replay): real captured bytes through kMatmulBT. Reads
// /tmp/w2b_qkvz_cpu/{h0.bin,w_packed.bin}; skips when absent. REMOVE before
// landing.

// Interior slice views of ONE tracked allocation must not share the base's
// staged device tensor. EnsureDevice2D keyed its hit on (base slot, dims)
// only, so ProjectGdnBA's `a` projection consumed the `b` weight rows and
// every Qwen3.5 GDN layer diverged from layer 0 (BACKEND-TENSTORRENT-QWEN35
// W2c: TT-a == CPU-b, corr -0.23). The fix requires t.data == slot host base
// for hits AND stores; this case pins that with the engine's exact sequence:
// one packed [2N,K] resident, then b-view then a-view matmuls.
TEST_CASE("kTENSTORRENT kMatmulBT slice views do not consume the base staging") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));
  constexpr int64_t M = 5, K = 64, N = 16;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  std::vector<uint16_t> hb(M * K), wb(2 * N * K);
  // Activations O(1) and halves far apart (+1 vs -2 weights): consuming the
  // wrong slice's staging moves outputs by ~3x the row sum — far past the
  // threshold. Tiny activations would swallow the poisoning below it.
  for (size_t i = 0; i < hb.size(); ++i) hb[i] = static_cast<uint16_t>(0x3E80 + (i % 9));
  for (size_t i = 0; i < wb.size(); ++i)
    wb[i] = static_cast<uint16_t>(i < wb.size() / 2 ? 0x3F80 : 0xC000);
  void* ma = backend.Alloc(M * K * 2);
  void* mb = backend.Alloc(2 * N * K * 2);
  void* mo = backend.Alloc(M * N * 4);
  Queue q = backend.CreateQueue();
  backend.Copy(q, ma, hb.data(), M * K * 2);
  backend.Copy(q, mb, wb.data(), 2 * N * K * 2);
  Tensor a = Tensor::Contiguous(ma, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {M, K});
  Tensor packed = Tensor::Contiguous(mb, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {2 * N, K});
  auto mm = reinterpret_cast<vt::MatmulFn>(vt::GetOp(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));
  auto widen = [](uint16_t u) {
    uint32_t bits = static_cast<uint32_t>(u) << 16;
    float f; std::memcpy(&f, &bits, 4); return f;
  };
  // Run TWICE so the second view also meets a warm (b-staged) cache.
  for (int rep = 0; rep < 2; ++rep) {
    for (int half = 0; half < 2; ++half) {
      Tensor wview = packed.Slice(0, half * N, (half + 1) * N);
      Tensor o = Tensor::Contiguous(mo, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, N});
      mm(q, o, a, wview);
      std::vector<float> oh(M * N);
      backend.Copy(q, oh.data(), mo, M * N * 4);
      double worst = 0;
      for (int64_t i = 0; i < M; ++i)
        for (int64_t j = 0; j < N; ++j) {
          double acc = 0;
          for (int64_t k = 0; k < K; ++k)
            acc += static_cast<double>(widen(hb[i * K + k])) *
                   widen(wb[(half * N + j) * K + k]);
          worst = std::max(worst, std::fabs(acc - oh[i * N + j]));
        }
      CHECK_MESSAGE(worst < 1.0,
                    "half " << half << " rep " << rep << " worst " << worst
                            << " — a view consumed another slice's staging");
    }
  }
  backend.Free(ma); backend.Free(mb); backend.Free(mo);
}

// bf16 x bf16 -> F32 out through kMatmulBT (the ProjectGdnBA split-arm
// signature): output dtype differs from input dtype.
TEST_CASE("kTENSTORRENT kMatmulBT bf16 inputs to F32 output") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));
  constexpr int64_t M = 5, K = 1024, N = 16;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  std::vector<uint16_t> hb(M * K), wb(N * K);
  for (size_t i = 0; i < hb.size(); ++i) hb[i] = static_cast<uint16_t>(0x3800 + (i % 13));
  for (size_t i = 0; i < wb.size(); ++i) wb[i] = static_cast<uint16_t>(0x3c00 + (i % 7));
  void* ma = backend.Alloc(M * K * 2); void* mb = backend.Alloc(N * K * 2);
  void* mo = backend.Alloc(M * N * 4);
  Queue q = backend.CreateQueue();
  backend.Copy(q, ma, hb.data(), M * K * 2);
  backend.Copy(q, mb, wb.data(), N * K * 2);
  Tensor a = Tensor::Contiguous(ma, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {M, K});
  Tensor b = Tensor::Contiguous(mb, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {N, K});
  Tensor o = Tensor::Contiguous(mo, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, N});
  auto mm = reinterpret_cast<vt::MatmulFn>(vt::GetOp(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));
  mm(q, o, a, b);
  std::vector<float> oh(M * N);
  backend.Copy(q, oh.data(), mo, M * N * 4);
  auto widen = [](uint16_t u) {
    uint32_t bits = static_cast<uint32_t>(u) << 16;
    float f; std::memcpy(&f, &bits, 4); return f;
  };
  double worst = 0;
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j) {
      double acc = 0;
      for (int64_t k = 0; k < K; ++k)
        acc += static_cast<double>(widen(hb[i * K + k])) * widen(wb[j * K + k]);
      worst = std::max(worst, std::fabs(acc - oh[i * N + j]));
    }
  CHECK(worst < 1.0);
  backend.Free(ma); backend.Free(mb); backend.Free(mo);
}

// ==== BACKEND-TENSTORRENT-QWEN35 W4 (#2107): bulk bf16 staging ==============
// The #1715 profile put the 0.104 tok/s eager-decode wall in EnsureDevice2D's
// host staging: a per-element f32 gather of a master that is ALREADY bf16,
// then a second f32→bf16 conversion inside ttnn's from_vector. W4 lever 1
// stages the raw bf16 bytes (one from_span, no f32 intermediate); lever 2
// resolves the slot once per call instead of four locked FindSlot probes.
//
// This case pins the ROUTE (the staging counter — a deleted bulk branch or an
// unwired counter cannot pass it), the BYTES (the device copy equals the
// per-element f32 reference bit-for-bit: exact widen-then-pack round trip,
// ±0, denormals, ±inf included) and the INTERIOR-VIEW class (the W2c
// defect): a view stages from its OWN window bytes, never the base slot's,
// even when another slice of the same base is already staged. The f32 path
// stays the reference arm — an f32 master must NOT take the bulk route (the
// f32 logits GEMM output keeps its declared dtype, neither widened nor
// narrowed by staging).
TEST_CASE("kTENSTORRENT W4 EnsureDevice2D bulk bf16 staging: route, bytes, views") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  using vt::tenstorrent::GetStagingStats;
  using vt::tenstorrent::ResetStagingStats;
  constexpr int64_t M = 5, K = 64, N = 16;
  auto widen = [](uint16_t u) {
    uint32_t bits = static_cast<uint32_t>(u) << 16;
    float f; std::memcpy(&f, &bits, 4); return f;
  };
  auto f32bits = [](float f) {
    uint32_t b; std::memcpy(&b, &f, 4); return b;
  };

  // 1) Route + bytes: a fresh bf16 master stages through the bulk route and
  //    the device copy matches the per-element f32 reference bit-for-bit.
  const uint16_t edges[] = {0x0000, 0x8000, 0x7F80, 0xFF80, 0x0001, 0x8001,
                            0x007F, 0x7F7F, 0x3F80, 0xBF80, 0x3E80, 0xC000};
  std::vector<uint16_t> hb(M * K), wb(N * K);
  for (size_t i = 0; i < hb.size(); ++i) hb[i] = edges[i % 12];
  for (size_t i = 0; i < wb.size(); ++i) wb[i] = static_cast<uint16_t>(0x3C00 + (i % 7));
  void* ma = backend.Alloc(M * K * 2);
  void* mb = backend.Alloc(N * K * 2);
  void* mo = backend.Alloc(M * N * 4);
  Queue q = backend.CreateQueue();
  backend.Copy(q, ma, hb.data(), M * K * 2);
  backend.Copy(q, mb, wb.data(), N * K * 2);
  Tensor a = Tensor::Contiguous(ma, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {M, K});
  Tensor b = Tensor::Contiguous(mb, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {N, K});
  Tensor o = Tensor::Contiguous(mo, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, N});
  auto mm = reinterpret_cast<vt::MatmulFn>(vt::GetOp(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));

  ResetStagingStats();
  mm(q, o, a, b);  // both operands are fresh bf16 masters → both stage
  vt::tenstorrent::StagingStats s = GetStagingStats();
  CHECK_MESSAGE(s.uploads_bulk_bf16 == 2,
                "both bf16 operands must stage through the bulk route, got "
                    << s.uploads_bulk_bf16);
  CHECK_MESSAGE(s.staged_bulk_bf16_bytes ==
                    static_cast<uint64_t>((M * K + N * K) * 2),
                "bulk bytes: got " << s.staged_bulk_bf16_bytes);
  CHECK_MESSAGE(s.staged_f32_elems == 0,
                "bf16 masters must not re-enter the f32 path, got "
                    << s.staged_f32_elems);
  std::vector<float> dev = vt::tenstorrent::DebugDeviceReadbackF32(q, a);
  REQUIRE(static_cast<int64_t>(dev.size()) == M * K);
  for (int64_t i = 0; i < M * K; ++i)
    CHECK_MESSAGE(f32bits(dev[static_cast<size_t>(i)]) == f32bits(widen(hb[static_cast<size_t>(i)])),
                  "device byte " << i << " diverges from the per-element reference");

  // 2) Interior view (W2c class): stage half 0, then half 1 of the same
  //    base. Half 1 must stage ITS OWN window bytes — not the base's, not
  //    half 0's — even with a same-shape staging resident on the base's
  //    behalf.
  std::vector<uint16_t> pb(2 * N * K);
  for (size_t i = 0; i < pb.size(); ++i)
    pb[i] = static_cast<uint16_t>(i < pb.size() / 2 ? 0x3F80 : 0xC000);
  void* mpb = backend.Alloc(2 * N * K * 2);
  backend.Copy(q, mpb, pb.data(), 2 * N * K * 2);
  Tensor packed = Tensor::Contiguous(mpb, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {2 * N, K});
  Tensor v0 = packed.Slice(0, 0, N);
  Tensor v1 = packed.Slice(0, N, 2 * N);
  ResetStagingStats();
  mm(q, o, a, v0);
  mm(q, o, a, v1);
  s = GetStagingStats();
  CHECK_MESSAGE(s.uploads_bulk_bf16 == 2,
                "each view stage is its own bulk upload, got "
                    << s.uploads_bulk_bf16);
  dev = vt::tenstorrent::DebugDeviceReadbackF32(q, v1);
  REQUIRE(static_cast<int64_t>(dev.size()) == N * K);
  for (int64_t i = 0; i < N * K; ++i)
    CHECK_MESSAGE(f32bits(dev[static_cast<size_t>(i)]) ==
                          f32bits(widen(pb[static_cast<size_t>(N * K + i)])),
                  "view staged the wrong window at byte " << i
                  << " — base staging leaked into an interior view");

  // 3) f32 master keeps the f32 reference arm: staged, never bulk-routed.
  std::vector<float> a32(M * K);
  for (size_t i = 0; i < a32.size(); ++i) a32[i] = widen(hb[i]);
  void* ma32 = backend.Alloc(M * K * 4);
  backend.Copy(q, ma32, a32.data(), M * K * 4);
  Tensor a32t = Tensor::Contiguous(ma32, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, K});
  ResetStagingStats();
  mm(q, o, a32t, b);  // b is already staged; only a32 stages
  s = GetStagingStats();
  CHECK_MESSAGE(s.uploads_bulk_bf16 == 0,
                "an f32 master must not take the bulk bf16 route, got "
                    << s.uploads_bulk_bf16);
  CHECK_MESSAGE(s.staged_f32_elems == static_cast<uint64_t>(M * K),
                "f32 master stages through the f32 path, got "
                    << s.staged_f32_elems);
  backend.Free(ma); backend.Free(mb); backend.Free(mo); backend.Free(mpb);
  backend.Free(ma32);
}

// ==== BACKEND-TENSTORRENT-QWEN35 W5 (#2244): allocation-free staging =========
// W4's profile left ~23% of the staging chain inside tt-metal per-upload
// internal work: UploadRowsBf16 built a NEW ttnn::Tensor via from_span on
// every staging upload, paying a fresh MeshBuffer allocation, cluster/chip
// discovery and tensor-attribute creation for identical geometry every step.
// W5 allocates the device buffer once per staging slot (lifecycle tied to the
// slot structures, under the #1486 never-destroy rule for static caches) and
// re-uploads by packing the host bytes (the exact from_span packing) and
// writing them through the mesh command queue into the resident buffer.
//
// This case pins the ROUTE (the persistent counters: cold slot allocates ONCE,
// a re-staged slot must NOT reallocate), the BYTES (the device copy equals the
// window's bf16 bits bit-for-bit after an in-place rewrite, so the persistent
// buffer provably carries the NEW bytes) and the f32 arm (still excluded — a
// genuine conversion never enters the bf16 persistent route). The W4 counters
// keep counting every bulk bf16 staging regardless of sub-route.
TEST_CASE("kTENSTORRENT W5 EnsureDevice2D persistent staging buffer: route, reuse, bytes") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  using vt::tenstorrent::GetStagingStats;
  using vt::tenstorrent::ResetStagingStats;
  constexpr int64_t M = 5, K = 64, N = 16;
  auto widen = [](uint16_t u) {
    uint32_t bits = static_cast<uint32_t>(u) << 16;
    float f; std::memcpy(&f, &bits, 4); return f;
  };
  auto f32bits = [](float f) {
    uint32_t b; std::memcpy(&b, &f, 4); return b;
  };
  // Two distinguishable bf16 bit patterns for the in-place rewrite leg.
  auto pattern = [](std::vector<uint16_t>& v, uint16_t base) {
    for (size_t i = 0; i < v.size(); ++i)
      v[i] = static_cast<uint16_t>(base + (i % 5));
  };

  // 1) Cold slot: the FIRST bulk upload allocates the per-slot persistent
  //    buffer and serves the upload through it.
  std::vector<uint16_t> ha(M * K), hb(N * K);
  pattern(ha, 0x3C00);
  pattern(hb, 0x3F80);
  void* ma = backend.Alloc(M * K * 2);
  void* mb = backend.Alloc(N * K * 2);
  void* mo = backend.Alloc(M * N * 4);
  Queue q = backend.CreateQueue();
  backend.Copy(q, ma, ha.data(), M * K * 2);
  backend.Copy(q, mb, hb.data(), N * K * 2);
  Tensor a = Tensor::Contiguous(ma, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {M, K});
  Tensor b = Tensor::Contiguous(mb, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {N, K});
  Tensor o = Tensor::Contiguous(mo, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, N});
  auto mm = reinterpret_cast<vt::MatmulFn>(vt::GetOp(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));

  ResetStagingStats();
  mm(q, o, a, b);  // a is cold → allocates; b is cold → allocates
  vt::tenstorrent::StagingStats s = GetStagingStats();
  CHECK_MESSAGE(s.uploads_bulk_bf16 == 2,
                "both bf16 operands still stage through the bulk route, got "
                    << s.uploads_bulk_bf16);
  CHECK_MESSAGE(s.uploads_persistent_bf16 == 2,
                "both bulk uploads must be served by the persistent route, got "
                    << s.uploads_persistent_bf16);
  CHECK_MESSAGE(s.uploads_persistent_allocs == 2,
                "two cold slots must allocate one persistent buffer each, got "
                    << s.uploads_persistent_allocs);
  CHECK_MESSAGE(s.staged_persistent_bf16_bytes ==
                    static_cast<uint64_t>((M * K + N * K) * 2),
                "persistent bytes: got " << s.staged_persistent_bf16_bytes);
  {
    std::vector<float> dev = vt::tenstorrent::DebugDeviceReadbackF32(q, a);
    REQUIRE(static_cast<int64_t>(dev.size()) == M * K);
    for (int64_t i = 0; i < M * K; ++i)
      CHECK_MESSAGE(f32bits(dev[static_cast<size_t>(i)]) == f32bits(widen(ha[static_cast<size_t>(i)])),
                    "cold persistent buffer carries the wrong bits at " << i);
  }

  // 2) Rewrite the SAME master in place and restage: the persistent buffer
  //    must be REUSED (zero new allocations) and must carry the NEW bytes —
  //    a stale in-place write cannot pass the readback.
  pattern(ha, 0x3800);  // different bit pattern entirely
  backend.Copy(q, ma, ha.data(), M * K * 2);  // MarkHostWritten drops the shadow
  ResetStagingStats();
  mm(q, o, a, b);  // a restages (shadow dropped); b's shadow is still resident
  s = GetStagingStats();
  CHECK_MESSAGE(s.uploads_bulk_bf16 == 1,
                "only the rewritten master restages, got "
                    << s.uploads_bulk_bf16);
  CHECK_MESSAGE(s.uploads_persistent_allocs == 0,
                "a re-staged slot must REUSE its persistent buffer, got "
                    << s.uploads_persistent_allocs << " new allocations");
  CHECK_MESSAGE(s.uploads_persistent_bf16 == 1,
                "the restage must be one in-place persistent write, got "
                    << s.uploads_persistent_bf16);
  CHECK_MESSAGE(s.staged_persistent_bf16_bytes == static_cast<uint64_t>(M * K * 2),
                "the in-place write must count the rewritten bytes, got "
                    << s.staged_persistent_bf16_bytes);
  {
    std::vector<float> dev = vt::tenstorrent::DebugDeviceReadbackF32(q, a);
    REQUIRE(static_cast<int64_t>(dev.size()) == M * K);
    for (int64_t i = 0; i < M * K; ++i)
      CHECK_MESSAGE(f32bits(dev[static_cast<size_t>(i)]) == f32bits(widen(ha[static_cast<size_t>(i)])),
                    "persistent buffer did not carry the rewritten bytes at " << i);
  }

  // 3) Geometry change on the same slot: the resident buffer cannot serve a
  //    different staging shape — reallocate, and COUNT the reallocation.
  std::vector<uint16_t> ha1(K);
  pattern(ha1, 0x4000);
  void* mo1 = backend.Alloc(N * 4);
  backend.Copy(q, ma, ha1.data(), K * 2);
  Tensor a1 = Tensor::Contiguous(ma, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {1, K});
  Tensor o1 = Tensor::Contiguous(mo1, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {1, N});
  ResetStagingStats();
  mm(q, o1, a1, b);  // a1 is the SAME base slot, staged at a new [1, K] shape
  s = GetStagingStats();
  CHECK_MESSAGE(s.uploads_bulk_bf16 == 1, "geometry change still bulk-stages, got "
                    << s.uploads_bulk_bf16);
  CHECK_MESSAGE(s.uploads_persistent_allocs == 1,
                "a staging-geometry change must reallocate the persistent "
                "buffer exactly once, got " << s.uploads_persistent_allocs);
  CHECK_MESSAGE(s.uploads_persistent_bf16 == 1,
                "the new geometry stages through the persistent route, got "
                    << s.uploads_persistent_bf16);
  {
    std::vector<float> dev = vt::tenstorrent::DebugDeviceReadbackF32(q, a1);
    REQUIRE(static_cast<int64_t>(dev.size()) == K);
    for (int64_t i = 0; i < K; ++i)
      CHECK_MESSAGE(f32bits(dev[static_cast<size_t>(i)]) == f32bits(widen(ha1[static_cast<size_t>(i)])),
                    "reallocated persistent buffer carries wrong bits at " << i);
  }

  // 4) The f32 arm keeps out of the persistent bf16 route entirely (the
  //    f32 logits GEMM output keeps its declared dtype).
  std::vector<float> a32(M * K);
  for (size_t i = 0; i < a32.size(); ++i) a32[i] = widen(ha[static_cast<size_t>(i)]);
  void* ma32 = backend.Alloc(M * K * 4);
  backend.Copy(q, ma32, a32.data(), M * K * 4);
  Tensor a32t = Tensor::Contiguous(ma32, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, K});
  ResetStagingStats();
  mm(q, o, a32t, b);  // only a32 stages; b is resident
  s = GetStagingStats();
  CHECK_MESSAGE(s.uploads_persistent_bf16 == 0,
                "an f32 master must not enter the persistent bf16 route, got "
                    << s.uploads_persistent_bf16);
  CHECK_MESSAGE(s.uploads_persistent_allocs == 0,
                "an f32 master must not allocate a persistent bf16 buffer, got "
                    << s.uploads_persistent_allocs);
  CHECK_MESSAGE(s.staged_f32_elems == static_cast<uint64_t>(M * K),
                "f32 master stages through the f32 path, got "
                    << s.staged_f32_elems);
  backend.Free(ma); backend.Free(mb); backend.Free(mo); backend.Free(mo1);
  backend.Free(ma32);
}

// ==== BACKEND-TENSTORRENT-QWEN35 W7 (#2282): staging-write elimination =======
// The W6 probe read ~7-8 staging writes per decode step off
// uploads_persistent_bf16 and traced each to the residency state dropping a
// shadow a consumer could have served. W7 makes the residency precise. Each
// case below pins ONE eliminated class through its production entry point —
// DevicePool::Get's OnScratchBlockAcquired, Backend::Memset, Backend::Copy —
// asserts the avoided counter AND a zero delta on the W6 write-count
// observable, and checks bit-identity: the bytes a consumer reads are the
// bytes it read before the arm existed.

// A pool-acquired block holds a PREVIOUS tenant's bytes on both sides; a
// pending op that fully overwrites the buffer must not restage that garbage.
namespace {
// The W7 arms pin EAGER semantics: the capture/host-free lane
// (VT_TT_HOST_FREE_DECODE) has its own tests, and under it IfCapture would
// satisfy the copy case without the eager arm. Saved and restored like the
// static-graph-mode cells above.
struct ForceEager {
  const bool had = std::getenv("VT_TT_HOST_FREE_DECODE") != nullptr;
  const std::string saved =
      had ? std::string(std::getenv("VT_TT_HOST_FREE_DECODE")) : std::string();
  ForceEager() { ::setenv("VT_TT_HOST_FREE_DECODE", "0", 1); }
  ~ForceEager() {
    if (had) ::setenv("VT_TT_HOST_FREE_DECODE", saved.c_str(), 1);
    else ::unsetenv("VT_TT_HOST_FREE_DECODE");
  }
};
}  // namespace
TEST_CASE("kTENSTORRENT W7 scratch-acquisition reservation: an acquired block restages nothing") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  ForceEager eager;
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));
  constexpr int64_t M = 5, K = 64, N = 16;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  using vt::tenstorrent::GetStagingStats;
  using vt::tenstorrent::ResetStagingStats;
  std::vector<uint16_t> hb(M * K), wb(N * K);
  for (size_t i = 0; i < hb.size(); ++i) hb[i] = static_cast<uint16_t>(0x3800 + (i % 13));
  for (size_t i = 0; i < wb.size(); ++i) wb[i] = static_cast<uint16_t>(0x3c00 + (i % 7));
  void* ma = backend.Alloc(M * K * 2); void* mb = backend.Alloc(N * K * 2);
  void* mo = backend.Alloc(M * N * 4);
  Queue q = backend.CreateQueue();
  backend.Copy(q, ma, hb.data(), M * K * 2);
  backend.Copy(q, mb, wb.data(), N * K * 2);
  Tensor a = Tensor::Contiguous(ma, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {M, K});
  Tensor b = Tensor::Contiguous(mb, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {N, K});
  Tensor o = Tensor::Contiguous(mo, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, N});
  auto mm = reinterpret_cast<vt::MatmulFn>(vt::GetOp(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));
  mm(q, o, a, b);  // stage both; a's slot now owns a persistent buffer
  std::vector<float> o1(M * N);
  backend.Copy(q, o1.data(), mo, M * N * 4);
  // DevicePool::Get free-list hit calls exactly this (device_pool.h) before the
  // block reaches its new tenant.
  backend.OnScratchBlockAcquired(ma);
  ResetStagingStats();
  mm(q, o, a, b);  // the new tenant's first device touch
  vt::tenstorrent::StagingStats s = GetStagingStats();
  CHECK_MESSAGE(s.stages_avoided_reservation == 1,
                "the acquired block must be served without staging, got "
                    << s.stages_avoided_reservation);
  CHECK_MESSAGE(s.uploads_persistent_bf16 == 0,
                "the acquired block must not restage, got "
                    << s.uploads_persistent_bf16);
  CHECK_MESSAGE(s.uploads_persistent_allocs == 0,
                "serving the resident allocation must not reallocate, got "
                    << s.uploads_persistent_allocs);
  std::vector<float> o2(M * N);
  backend.Copy(q, o2.data(), mo, M * N * 4);
  for (size_t i = 0; i < o1.size(); ++i)
    CHECK_MESSAGE(o1[i] == o2[i], "output mismatch at " << i << ": "
                            << o1[i] << " vs " << o2[i]);
  backend.Free(ma); backend.Free(mb); backend.Free(mo);
}

// An eager full-slot zero-fill of a device-resident slot must keep the shadow:
// the next device read serves zeros from the device instead of restaging them.
TEST_CASE("kTENSTORRENT W7 eager full-slot zero-fill keeps the shadow") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  ForceEager eager;
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));
  constexpr int64_t M = 5, K = 64, N = 16;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  using vt::tenstorrent::GetStagingStats;
  using vt::tenstorrent::ResetStagingStats;
  std::vector<uint16_t> hb(M * K), wb(N * K);
  for (size_t i = 0; i < hb.size(); ++i) hb[i] = static_cast<uint16_t>(0x3800 + (i % 13));
  for (size_t i = 0; i < wb.size(); ++i) wb[i] = static_cast<uint16_t>(0x3c00 + (i % 7));
  void* ma = backend.Alloc(M * K * 2); void* mb = backend.Alloc(N * K * 2);
  void* mo = backend.Alloc(M * N * 4);
  Queue q = backend.CreateQueue();
  backend.Copy(q, ma, hb.data(), M * K * 2);
  backend.Copy(q, mb, wb.data(), N * K * 2);
  Tensor a = Tensor::Contiguous(ma, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {M, K});
  Tensor b = Tensor::Contiguous(mb, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {N, K});
  Tensor o = Tensor::Contiguous(mo, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, N});
  auto mm = reinterpret_cast<vt::MatmulFn>(vt::GetOp(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));
  mm(q, o, a, b);  // stage a; its shadow is live
  ResetStagingStats();
  backend.Memset(q, ma, 0, M * K * 2);  // production entry: eager full-slot zero
  vt::tenstorrent::StagingStats s = GetStagingStats();
  CHECK_MESSAGE(s.stages_avoided_device_memset == 1,
                "the full-slot zero-fill must fill the device shadow, got "
                    << s.stages_avoided_device_memset);
  CHECK_MESSAGE(s.uploads_persistent_bf16 == 0,
                "the zero-fill must not stage, got " << s.uploads_persistent_bf16);
  CHECK_MESSAGE(s.staged_persistent_bf16_bytes == 0,
                "the zero-fill must not push staging bytes, got "
                    << s.staged_persistent_bf16_bytes);
  const auto* ha = static_cast<const uint16_t*>(ma);
  for (int64_t i = 0; i < M * K; ++i)
    CHECK_MESSAGE(ha[i] == 0, "host byte " << i << " not zeroed");
  mm(q, o, a, b);  // the next device read of the zeroed slot
  s = GetStagingStats();
  CHECK_MESSAGE(s.uploads_persistent_bf16 == 0,
                "the zeroed slot must serve from its shadow, got "
                    << s.uploads_persistent_bf16);
  std::vector<float> oh(M * N);
  backend.Copy(q, oh.data(), mo, M * N * 4);
  for (size_t i = 0; i < oh.size(); ++i)
    CHECK_MESSAGE(oh[i] == 0.0f, "output " << i << " = " << oh[i] << ", want 0");
  backend.Free(ma); backend.Free(mb); backend.Free(mo);
}

// A copy whose SOURCE is device-resident and whose dst already owns a
// persistent buffer must go device->device: today it downloads, memcpys, drops
// the shadow, and the next device read re-uploads the same bytes.
TEST_CASE("kTENSTORRENT W7 eager device-resident D2D copy skips the host round trip") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  ForceEager eager;
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));
  constexpr int64_t M = 5, K = 64, N = 16;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  using vt::tenstorrent::GetStagingStats;
  using vt::tenstorrent::ResetStagingStats;
  std::vector<uint16_t> hb(M * K), wb(N * K), db(M * K);
  for (size_t i = 0; i < hb.size(); ++i) hb[i] = static_cast<uint16_t>(0x3800 + (i % 13));
  for (size_t i = 0; i < wb.size(); ++i) wb[i] = static_cast<uint16_t>(0x3c00 + (i % 7));
  for (size_t i = 0; i < db.size(); ++i) db[i] = static_cast<uint16_t>(0x3f00 + (i % 11));
  void* ma = backend.Alloc(M * K * 2); void* mb = backend.Alloc(N * K * 2);
  void* md = backend.Alloc(M * K * 2);
  void* mo = backend.Alloc(M * N * 4);
  Queue q = backend.CreateQueue();
  backend.Copy(q, ma, hb.data(), M * K * 2);
  backend.Copy(q, mb, wb.data(), N * K * 2);
  backend.Copy(q, md, db.data(), M * K * 2);
  Tensor a = Tensor::Contiguous(ma, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {M, K});
  Tensor d = Tensor::Contiguous(md, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {M, K});
  Tensor b = Tensor::Contiguous(mb, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {N, K});
  Tensor o = Tensor::Contiguous(mo, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, N});
  auto mm = reinterpret_cast<vt::MatmulFn>(vt::GetOp(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));
  mm(q, o, a, b);  // stage a — its shadow is live
  std::vector<float> o_a(M * N);
  backend.Copy(q, o_a.data(), mo, M * N * 4);  // the a×b reference
  mm(q, o, d, b);  // stage d — dst owns a persistent buffer
  ResetStagingStats();
  backend.Copy(q, md, ma, M * K * 2);  // src is device-resident
  vt::tenstorrent::StagingStats s = GetStagingStats();
  CHECK_MESSAGE(s.stages_avoided_device_copy == 1,
                "the device-resident copy must go device->device, got "
                    << s.stages_avoided_device_copy);
  CHECK_MESSAGE(s.uploads_persistent_bf16 == 0,
                "the copy must not stage, got " << s.uploads_persistent_bf16);
  mm(q, o, d, b);  // the next device consumer of d
  s = GetStagingStats();
  CHECK_MESSAGE(s.uploads_persistent_bf16 == 0,
                "d must serve from its fresh shadow, got "
                    << s.uploads_persistent_bf16);
  std::vector<float> o_d(M * N);
  backend.Copy(q, o_d.data(), mo, M * N * 4);
  for (size_t i = 0; i < o_a.size(); ++i)
    CHECK_MESSAGE(o_a[i] == o_d[i], "d does not carry a's bytes at " << i
                            << ": " << o_a[i] << " vs " << o_d[i]);
  backend.Free(ma); backend.Free(mb); backend.Free(md); backend.Free(mo);
}

// W7 repair (#2282): the reservation arm serves a resident-or-empty BFLOAT16
// allocation, so a non-bf16 master whose first device use lands on an acquired
// block must NOT take it — the arm would hand a bf16 device tensor to an f32
// master (the W7 invariant: the f32 arms keep their declared dtypes). The arm
// must be gated on the master's dtype and the staging must fall through to the
// normal path, whose f32 arm uploads the declared dtype.
TEST_CASE("kTENSTORRENT W7 reservation arm keeps an f32 master's declared dtype") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  ForceEager eager;
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));
  constexpr int64_t M = 5, K = 64, N = 16;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  using vt::tenstorrent::GetStagingStats;
  using vt::tenstorrent::ResetStagingStats;
  std::vector<uint16_t> wb(N * K);
  for (size_t i = 0; i < wb.size(); ++i) wb[i] = static_cast<uint16_t>(0x3c00 + (i % 7));
  std::vector<float> a32(M * K);
  for (size_t i = 0; i < a32.size(); ++i)
    a32[i] = 0.25f * static_cast<float>(i % 17) - 1.0f;
  void* mb = backend.Alloc(N * K * 2);
  void* ma32 = backend.Alloc(M * K * 4);
  void* mo = backend.Alloc(M * N * 4);
  Queue q = backend.CreateQueue();
  backend.Copy(q, mb, wb.data(), N * K * 2);
  backend.Copy(q, ma32, a32.data(), M * K * 4);
  Tensor b = Tensor::Contiguous(mb, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {N, K});
  Tensor a = Tensor::Contiguous(ma32, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, K});
  Tensor o = Tensor::Contiguous(mo, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, N});
  auto mm = reinterpret_cast<vt::MatmulFn>(vt::GetOp(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));
  mm(q, o, a, b);  // stage the f32 master; its shadow is live
  std::vector<float> o1(M * N);
  backend.Copy(q, o1.data(), mo, M * N * 4);  // the f32 master's own reference
  // DevicePool::Get free-list hit calls exactly this (device_pool.h) before the
  // block reaches its new tenant.
  backend.OnScratchBlockAcquired(ma32);
  ResetStagingStats();
  mm(q, o, a, b);  // the acquired block's first device use is an f32 master
  vt::tenstorrent::StagingStats s = GetStagingStats();
  CHECK_MESSAGE(s.stages_avoided_reservation == 0,
                "the reservation arm must not serve a non-bf16 master, got "
                    << s.stages_avoided_reservation);
  CHECK_MESSAGE(s.staged_f32_elems == static_cast<uint64_t>(M * K),
                "the f32 master must fall through to the f32 staging arm, got "
                    << s.staged_f32_elems);
  std::vector<float> o2(M * N);
  backend.Copy(q, o2.data(), mo, M * N * 4);
  for (size_t i = 0; i < o1.size(); ++i)
    CHECK_MESSAGE(o1[i] == o2[i], "f32 master served wrong bytes at " << i
                            << ": " << o1[i] << " vs " << o2[i]);
  backend.Free(ma32); backend.Free(mb); backend.Free(mo);
}

// W7 repair (#2282): the eager D2D copy installs src's shadow — src's logical
// geometry — into dst's slot. Equal byte size does not mean equal geometry, so
// the slot record must name the SERVED geometry: a stale record would let a
// later stage at dst's recorded geometry hit the exact-shape fast path and be
// handed a wrongly-shaped tensor.
TEST_CASE("kTENSTORRENT W7 D2D copy records the served geometry") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  ForceEager eager;
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));
  // 32 elements on both sides ([4,8] src vs [8,4] dst) — equal 64-byte slots
  // (Alloc registers the 64-rounded size), different geometry.
  constexpr int64_t R = 8, C = 4, N = 16;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  using vt::tenstorrent::GetStagingStats;
  using vt::tenstorrent::ResetStagingStats;
  std::vector<uint16_t> sb(C * R), db(R * C), wb4(N * R), wb5(N * C);
  for (size_t i = 0; i < sb.size(); ++i) sb[i] = static_cast<uint16_t>(0x3800 + (i % 13));
  for (size_t i = 0; i < db.size(); ++i) db[i] = static_cast<uint16_t>(0x3c00 + (i % 7));
  for (size_t i = 0; i < wb4.size(); ++i) wb4[i] = static_cast<uint16_t>(0x3f00 + (i % 5));
  for (size_t i = 0; i < wb5.size(); ++i) wb5[i] = static_cast<uint16_t>(0x4000 + (i % 9));
  void* ma = backend.Alloc(C * R * 2);
  void* md = backend.Alloc(R * C * 2);
  void* mr = backend.Alloc(R * C * 2);
  void* mb4 = backend.Alloc(N * R * 2);
  void* mb5 = backend.Alloc(N * C * 2);
  void* mos = backend.Alloc(C * N * 4);
  void* mo = backend.Alloc(R * N * 4);
  Queue q = backend.CreateQueue();
  backend.Copy(q, ma, sb.data(), C * R * 2);
  backend.Copy(q, md, db.data(), R * C * 2);
  backend.Copy(q, mr, sb.data(), R * C * 2);  // dst's post-copy flat bytes as [R, C]
  backend.Copy(q, mb4, wb4.data(), N * R * 2);
  backend.Copy(q, mb5, wb5.data(), N * C * 2);
  Tensor s = Tensor::Contiguous(ma, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {C, R});
  Tensor d = Tensor::Contiguous(md, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {R, C});
  Tensor r = Tensor::Contiguous(mr, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {R, C});
  Tensor b4 = Tensor::Contiguous(mb4, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {N, R});
  Tensor b5 = Tensor::Contiguous(mb5, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {N, C});
  Tensor os = Tensor::Contiguous(mos, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {C, N});
  Tensor o = Tensor::Contiguous(mo, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {R, N});
  auto mm = reinterpret_cast<vt::MatmulFn>(vt::GetOp(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));
  mm(q, os, s, b4);  // stage the src [C, R] shadow
  mm(q, o, d, b5);   // stage dst — its slot owns a persistent buffer
  mm(q, o, r, b5);   // the reference: the same flat bytes declared [R, C]
  std::vector<float> want(R * N);
  backend.Copy(q, want.data(), mo, R * N * 4);
  ResetStagingStats();
  backend.Copy(q, md, ma, R * C * 2);  // whole-slot D2D copy: dst's shadow is src-shaped
  vt::tenstorrent::StagingStats st = GetStagingStats();
  CHECK_MESSAGE(st.stages_avoided_device_copy == 1,
                "the device-resident copy must go device->device, got "
                    << st.stages_avoided_device_copy);
  mm(q, o, d, b5);  // consume dst at its DECLARED [R, C] geometry
  st = GetStagingStats();
  CHECK_MESSAGE(st.uploads_persistent_bf16 == 0,
                "dst must serve from its copied shadow, got "
                    << st.uploads_persistent_bf16);
  std::vector<float> o2(R * N);
  backend.Copy(q, o2.data(), mo, R * N * 4);
  for (size_t i = 0; i < want.size(); ++i)
    CHECK_MESSAGE(want[i] == o2[i], "dst at its declared geometry differs at "
                            << i << ": " << want[i] << " vs " << o2[i]);
  backend.Free(ma); backend.Free(md); backend.Free(mr);
  backend.Free(mb4); backend.Free(mb5); backend.Free(mos); backend.Free(mo);
}

// The capture-lane twin of the eager case above. #2294: CopyDeviceDeviceIfCapture
// also installs SRC's shadow into dst's slot, and equal byte size does not mean
// equal geometry there either. A stale record lets a later stage at dst's
// recorded exact geometry hit the fast path and be handed a wrongly-shaped
// src-shaped tensor. The whole-slot D2D copy runs INSIDE a real trace capture
// region (BeginCapture/EndCapture, then Replay to execute the captured op):
// under an active capture the eager arm CopyDeviceDeviceIfResident structurally
// declines, so the capture lane is the only path that can serve the copy and
// the avoided-copy counter is lane-exclusive — deleting the capture lane's
// call site cannot stay green (the host fallback would read back inside the
// captured region, violating the ttnn trace contract, and the counter would
// read 0). The consume at dst's declared [R, C] geometry pins the
// served-geometry record: without it the stage false-hits the exact-shape
// fast path and hands the matmul the wrongly-shaped [C, R] tensor, which
// throws. The case arms the lane through the live env read the same way the
// inertness-guard case sets it.
struct ForceHostFreeDecode {
  const bool had = std::getenv("VT_TT_HOST_FREE_DECODE") != nullptr;
  const std::string saved =
      had ? std::string(std::getenv("VT_TT_HOST_FREE_DECODE")) : std::string();
  ForceHostFreeDecode() { ::setenv("VT_TT_HOST_FREE_DECODE", "1", 1); }
  ~ForceHostFreeDecode() {
    if (had) ::setenv("VT_TT_HOST_FREE_DECODE", saved.c_str(), 1);
    else ::unsetenv("VT_TT_HOST_FREE_DECODE");
  }
};
TEST_CASE("kTENSTORRENT #2294 capture-lane D2D copy records the served geometry") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  ForceHostFreeDecode host_free;
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));
  // 32 elements on both sides ([4,8] src vs [8,4] dst) — equal 64-byte slots
  // (Alloc registers the 64-rounded size), different geometry.
  constexpr int64_t R = 8, C = 4, N = 16;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  using vt::tenstorrent::GetStagingStats;
  using vt::tenstorrent::ResetStagingStats;
  std::vector<uint16_t> sb(C * R), db(R * C), wb4(N * R), wb5(N * C);
  for (size_t i = 0; i < sb.size(); ++i) sb[i] = static_cast<uint16_t>(0x3800 + (i % 13));
  for (size_t i = 0; i < db.size(); ++i) db[i] = static_cast<uint16_t>(0x3c00 + (i % 7));
  for (size_t i = 0; i < wb4.size(); ++i) wb4[i] = static_cast<uint16_t>(0x3f00 + (i % 5));
  for (size_t i = 0; i < wb5.size(); ++i) wb5[i] = static_cast<uint16_t>(0x4000 + (i % 9));
  void* ma = backend.Alloc(C * R * 2);
  void* md = backend.Alloc(R * C * 2);
  void* mr = backend.Alloc(R * C * 2);
  void* mb4 = backend.Alloc(N * R * 2);
  void* mb5 = backend.Alloc(N * C * 2);
  void* mos = backend.Alloc(C * N * 4);
  void* mo = backend.Alloc(R * N * 4);
  Queue q = backend.CreateQueue();
  backend.Copy(q, ma, sb.data(), C * R * 2);
  backend.Copy(q, md, db.data(), R * C * 2);
  backend.Copy(q, mr, sb.data(), R * C * 2);  // dst's post-copy flat bytes as [R, C]
  backend.Copy(q, mb4, wb4.data(), N * R * 2);
  backend.Copy(q, mb5, wb5.data(), N * C * 2);
  Tensor s = Tensor::Contiguous(ma, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {C, R});
  Tensor d = Tensor::Contiguous(md, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {R, C});
  Tensor r = Tensor::Contiguous(mr, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {R, C});
  Tensor b4 = Tensor::Contiguous(mb4, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {N, R});
  Tensor b5 = Tensor::Contiguous(mb5, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {N, C});
  Tensor os = Tensor::Contiguous(mos, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {C, N});
  Tensor o = Tensor::Contiguous(mo, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {R, N});
  auto mm = reinterpret_cast<vt::MatmulFn>(vt::GetOp(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));
  mm(q, os, s, b4);  // stage the src [C, R] shadow
  mm(q, o, d, b5);   // stage dst — its slot owns a persistent buffer
  mm(q, o, r, b5);   // the reference: the same flat bytes declared [R, C]
  std::vector<float> want(R * N);
  backend.Copy(q, want.data(), mo, R * N * 4);
  // Program-cache warm contract (ttnn trace fatals on "Cannot load new
  // binaries during trace capture"): run the SAME D2D copy once eagerly,
  // outside capture — the host-free lane arms it and the first use enables
  // the program cache and compiles the captured ttnn::empty + ttnn::copy
  // pair. Drop its counter contribution before the capture region.
  backend.Copy(q, md, ma, R * C * 2);
  ResetStagingStats();
  // Capture region holds only the whole-slot D2D copy: dst's shadow becomes
  // src-shaped. Replay executes it (non-blocking; the readback below
  // synchronizes, mirroring the matmul capture/replay recipe above).
  backend.BeginCapture(q);
  backend.Copy(q, md, ma, R * C * 2);
  backend.EndCapture(q);
  backend.Replay(q);
  vt::tenstorrent::StagingStats st = GetStagingStats();
  CHECK_MESSAGE(st.stages_avoided_device_copy == 1,
                "under an active capture only the capture lane may serve the "
                    "copy, got "
                    << st.stages_avoided_device_copy);
  mm(q, o, d, b5);  // consume dst at its DECLARED [R, C] geometry
  st = GetStagingStats();
  CHECK_MESSAGE(st.uploads_persistent_bf16 == 0,
                "dst must serve from its copied shadow, got "
                    << st.uploads_persistent_bf16);
  std::vector<float> o2(R * N);
  backend.Copy(q, o2.data(), mo, R * N * 4);
  for (size_t i = 0; i < want.size(); ++i)
    CHECK_MESSAGE(want[i] == o2[i], "dst at its declared geometry differs at "
                            << i << ": " << want[i] << " vs " << o2[i]);
  backend.Free(ma); backend.Free(md); backend.Free(mr);
  backend.Free(mb4); backend.Free(mb5); backend.Free(mos); backend.Free(mo);
}

// W7 drift repair (#2282): the reservation must not survive content
// establishment, and the reserved arm must never discard a live shadow.
// Captured on the host-free e2e drift (LEG B, prompt[1] tok=0): a pool
// block handed to a new tenant
// received a device-committed [8,256] result while device_reserved stayed
// armed; the next bf16 consumer staged the block at its earlier [5,1024]
// geometry, took the reserved arm, and was handed the STALE persistent
// buffer — the previous tenant's bytes — while the live [8,256] shadow was
// dropped. The stage must fall through to the normal path (refresh + upload)
// whenever the slot holds a live device shadow. The arming wiring itself
// (OnScratchBlockAcquired -> MarkScratchAcquired) is pinned by the sibling
// acquired-block-restages-nothing test below; this test pins the
// serve-over-live-shadow guard, which passes vacuously if arming is removed.
TEST_CASE("kTENSTORRENT W7 reservation never serves over a live device shadow") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  ForceEager eager;
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));
  constexpr int64_t T = 5, H = 1024, N1 = 16;   // the block's [5,1024] role
  constexpr int64_t M2 = 8, K2 = 64, N2 = 256;  // the new tenant's [8,256] role
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  using vt::tenstorrent::GetStagingStats;
  using vt::tenstorrent::ResetStagingStats;
  auto mm = reinterpret_cast<vt::MatmulFn>(
      vt::GetOp(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));
  Queue q = backend.CreateQueue();
  const Device dev{DeviceType::kTENSTORRENT, 0};

  void* w1b = backend.Alloc(N1 * H * 2);
  std::vector<uint16_t> w1(N1 * H);
  for (size_t i = 0; i < w1.size(); ++i) w1[i] = static_cast<uint16_t>(0x3c00 + (i % 7));
  void* a2b = backend.Alloc(M2 * K2 * 2);
  std::vector<uint16_t> a2(M2 * K2);
  for (size_t i = 0; i < a2.size(); ++i) a2[i] = static_cast<uint16_t>(0x3800 + (i % 13));
  void* b2b = backend.Alloc(N2 * K2 * 2);
  std::vector<uint16_t> b2(N2 * K2);
  for (size_t i = 0; i < b2.size(); ++i) b2[i] = static_cast<uint16_t>(0x3f00 + (i % 5));
  backend.Copy(q, w1b, w1.data(), N1 * H * 2);
  backend.Copy(q, a2b, a2.data(), M2 * K2 * 2);
  backend.Copy(q, b2b, b2.data(), N2 * K2 * 2);
  Tensor W1 = Tensor::Contiguous(w1b, vt::DType::kBF16, dev, {N1, H});
  Tensor a2t = Tensor::Contiguous(a2b, vt::DType::kBF16, dev, {M2, K2});
  Tensor b2t = Tensor::Contiguous(b2b, vt::DType::kBF16, dev, {N2, K2});

  // Tenant 1 stages the block as a [5,1024] bf16 input: its W5 persistent
  // buffer now holds bytes A.
  void* blk = backend.Alloc(T * H * 2);
  std::vector<uint16_t> a(T * H);
  for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<uint16_t>(0x3800 + (i % 13));
  backend.Copy(q, blk, a.data(), T * H * 2);
  Tensor in1 = Tensor::Contiguous(blk, vt::DType::kBF16, dev, {T, H});
  void* o1b = backend.Alloc(T * N1 * 4);
  Tensor o1 = Tensor::Contiguous(o1b, vt::DType::kF32, dev, {T, N1});
  mm(q, o1, in1, W1);
  std::vector<float> r1(T * N1);
  backend.Copy(q, r1.data(), o1b, T * N1 * 4);  // bytes A's own reference

  // DevicePool::Get hands the block to a new tenant (reservation armed).
  backend.OnScratchBlockAcquired(blk);

  // The new tenant's producer commits a DIFFERENT-geometry [8,256] result
  // into the block: the slot's bytes are now REAL (device truth), which must
  // consume the reservation.
  void* o2b = backend.Alloc(M2 * N2 * 4);
  Tensor o2 = Tensor::Contiguous(blk, vt::DType::kF32, dev, {M2, N2});
  mm(q, o2, a2t, b2t);  // CommitDeviceLogical2D: live [8,256] shadow on blk
  std::vector<float> r2(M2 * N2);
  backend.Copy(q, r2.data(), blk, M2 * N2 * 4);  // the live shadow serves

  // The next bf16 consumer stages the block at its earlier [5,1024] geometry.
  Tensor in3 = Tensor::Contiguous(blk, vt::DType::kBF16, dev, {T, H});
  void* o3b = backend.Alloc(T * N1 * 4);
  Tensor o3 = Tensor::Contiguous(o3b, vt::DType::kF32, dev, {T, N1});
  ResetStagingStats();
  mm(q, o3, in3, W1);
  vt::tenstorrent::StagingStats s = GetStagingStats();
  CHECK_MESSAGE(s.stages_avoided_reservation == 0,
                "the reserved arm must not serve a slot holding a live device "
                "shadow, got " << s.stages_avoided_reservation);
  std::vector<float> r3(T * N1);
  backend.Copy(q, r3.data(), o3b, T * N1 * 4);
  bool differs = false;
  for (size_t i = 0; i < r1.size(); ++i) {
    if (r1[i] != r3[i]) {
      differs = true;
      break;
    }
  }
  CHECK_MESSAGE(differs,
                "the stage consumed the PREVIOUS tenant's staged bytes: the "
                "stale persistent buffer was served over a live shadow");
  backend.Free(w1b); backend.Free(a2b); backend.Free(b2b); backend.Free(blk);
  backend.Free(o1b); backend.Free(o2b); backend.Free(o3b);
}

// ==== BACKEND-TENSTORRENT-QWEN35 W3 (#2201): the GDN reviewer leftovers ======
// (a) the state d2h counter must see BOTH remaining download paths — the
// EnsureGdnCacheDevice slow-path refresh and the CommitConvTransposed
// untracked fallback; (b) EnsureGdnCacheDevice must refuse a host pointer
// presented under a different role (a conv_transposed shadow asked for as an
// ssm cache). qwen3_5.cpp uses distinct buffers per role, so (b) hardens.

namespace {

// Row-major contiguous Tensor over an existing TT-backed allocation
// (Contiguous takes an initializer_list; these shapes are computed).
Tensor RowMajorTT(void* mem, vt::DType dt, const std::vector<int64_t>& shape) {
  Tensor t{};
  t.data = mem;
  t.dtype = dt;
  t.device = Device{DeviceType::kTENSTORRENT, 0};
  t.rank = static_cast<int32_t>(shape.size());
  int64_t acc = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    t.shape[i] = shape[i];
    t.stride[i] = acc;
    acc *= shape[i];
  }
  return t;
}

}  // namespace

TEST_CASE("kTENSTORRENT GDN d2h counter sees the EnsureGdnCacheDevice slow-path refresh (#2201)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& tt = *vt::TryGetBackend(DeviceType::kTENSTORRENT);
  const int64_t t = 2, d = 64, rows = 3;  // the shared [t, d] geometry
  // One allocation plays BOTH roles: kSiluAndMul commits a bf16 [t, d]
  // device result over it (device-current, host-stale), then the GDN scatter
  // presents the SAME host pointer as an f32 [t, d] cache. Equal volume,
  // dtype mismatch: the EnsureGdnCacheDevice fast path misses, so the refresh
  // downloads the resident bf16 shadow back to host before re-uploading —
  // a volume change instead would be refused by EnsureHost's size check.
  void* mc = tt.Alloc(static_cast<size_t>(t * d) * sizeof(float));  // f32-sized
  void* mx = tt.Alloc(static_cast<size_t>(t * 2 * d) * 2);          // bf16 master
  void* mw = tt.Alloc(static_cast<size_t>(rows * d) * sizeof(float));
  void* mi = tt.Alloc(static_cast<size_t>(rows) * sizeof(int32_t));
  Queue q = tt.CreateQueue();
  std::vector<uint16_t> xb(static_cast<size_t>(t * 2 * d));
  for (size_t i = 0; i < xb.size(); ++i)
    xb[i] = static_cast<uint16_t>(0x3C00 + (i % 7));  // small bf16-exact values
  tt.Copy(q, mx, xb.data(), xb.size() * 2);
  const std::vector<int32_t> idx{0, 1, 1};
  tt.Copy(q, mi, idx.data(), idx.size() * sizeof(int32_t));
  Tensor ti = RowMajorTT(mi, vt::DType::kI32, {rows});

  Tensor tx = RowMajorTT(mx, vt::DType::kBF16, {t, 2 * d});
  Tensor tob = RowMajorTT(mc, vt::DType::kBF16, {t, d});
  vt::SiluAndMul(q, tob, tx);
  vt::tenstorrent::ResetGdnShadowTraffic();
  Tensor tc = RowMajorTT(mc, vt::DType::kF32, {t, d});
  Tensor twk = RowMajorTT(mw, vt::DType::kF32, {rows, d});
  vt::GdnStateScatter(q, tc, twk, ti);
  const auto tr = vt::tenstorrent::GetGdnShadowTraffic();
  const uint64_t want_d2h = static_cast<uint64_t>(t * d) * sizeof(float);
  MESSAGE("slow-path refresh d2h=", tr.state_d2h_bytes, " (want ", want_d2h,
          ")");
  CHECK(tr.state_d2h_bytes == want_d2h);
  tt.Free(mc);
  tt.Free(mx);
  tt.Free(mw);
  tt.Free(mi);
}

TEST_CASE("kTENSTORRENT GDN d2h counter sees the CommitConvTransposed untracked fallback (#2201)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& tt = *vt::TryGetBackend(DeviceType::kTENSTORRENT);
  const int64_t B = 2, C = 64, K = 4, sl = K - 1;
  uint32_t s = 77001u;
  std::vector<float> w(static_cast<size_t>(C * K)), bias(static_cast<size_t>(C)),
      x(static_cast<size_t>(B * C)), st(static_cast<size_t>(B * C * sl)),
      out(static_cast<size_t>(B * C), 0.0f);
  for (float& v : w) v = 0.4f * GdnLcg(s);
  for (float& v : bias) v = 0.1f * GdnLcg(s);
  for (float& v : st) v = GdnLcg(s);
  for (float& v : x) v = 2.0f * GdnLcg(s);
  // NONE of these pointers is tt.Alloc'd: the conv state is an UNTRACKED
  // buffer, so the step's final commit takes the host-materialization
  // fallback in CommitConvTransposed — a real device→host download of the
  // whole [sl+1, R] shadow that the counter must see.
  Queue q = tt.CreateQueue();
  Tensor to = RowMajorTT(out.data(), vt::DType::kF32, {B, C});
  Tensor tx = RowMajorTT(x.data(), vt::DType::kF32, {B, C});
  Tensor tw = RowMajorTT(w.data(), vt::DType::kF32, {C, K});
  Tensor tb = RowMajorTT(bias.data(), vt::DType::kF32, {C});
  Tensor ts = RowMajorTT(st.data(), vt::DType::kF32, {B, C, sl});
  vt::CausalConv1dArgs a;
  a.silu_activation = true;
  vt::tenstorrent::ResetGdnShadowTraffic();
  vt::CausalConv1dUpdate(q, to, tx, tw, &tb, ts, a, nullptr);
  const auto tr = vt::tenstorrent::GetGdnShadowTraffic();
  const uint64_t want_d2h =
      static_cast<uint64_t>(sl + 1) * static_cast<uint64_t>(B * C) *
      sizeof(float);
  MESSAGE("untracked conv commit d2h=", tr.state_d2h_bytes, " (want ",
          want_d2h, ")");
  CHECK(tr.state_d2h_bytes == want_d2h);
}

TEST_CASE("kTENSTORRENT EnsureGdnCacheDevice refuses a conv_transposed pointer presented as an ssm cache (#2201)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& tt = *vt::TryGetBackend(DeviceType::kTENSTORRENT);
  const int64_t B = 2, C = 64, K = 4, sl = K - 1, R = B * C;
  uint32_t s = 77002u;
  std::vector<float> w(static_cast<size_t>(C * K)), bias(static_cast<size_t>(C)),
      x(static_cast<size_t>(B * C)), st(static_cast<size_t>(B * C * sl)),
      out(static_cast<size_t>(B * C), 0.0f);
  for (float& v : w) v = 0.4f * GdnLcg(s);
  for (float& v : bias) v = 0.1f * GdnLcg(s);
  for (float& v : st) v = GdnLcg(s);
  for (float& v : x) v = 2.0f * GdnLcg(s);
  // One tracked update step leaves the conv-state buffer holding a
  // TRANSPOSED [sl+1, R] device shadow. The allocation covers the LARGER ssm
  // view so only the role can be wrong here, never the extent.
  const size_t ms_floats =
      st.size() > static_cast<size_t>((sl + 1) * R) ? st.size()
                                                    : static_cast<size_t>((sl + 1) * R);
  void* mx = tt.Alloc(x.size() * sizeof(float));
  void* mw = tt.Alloc(w.size() * sizeof(float));
  void* mb = tt.Alloc(bias.size() * sizeof(float));
  void* mo = tt.Alloc(out.size() * sizeof(float));
  void* ms = tt.Alloc(ms_floats * sizeof(float));
  void* mg = tt.Alloc(static_cast<size_t>(3 * ((sl + 1) * R / 2)) * sizeof(float));
  void* mp = tt.Alloc(3 * sizeof(int32_t));
  Queue q = tt.CreateQueue();
  tt.Copy(q, mx, x.data(), x.size() * sizeof(float));
  tt.Copy(q, mw, w.data(), w.size() * sizeof(float));
  tt.Copy(q, mb, bias.data(), bias.size() * sizeof(float));
  tt.Copy(q, mo, out.data(), out.size() * sizeof(float));
  tt.Copy(q, ms, st.data(), st.size() * sizeof(float));
  Tensor to = RowMajorTT(mo, vt::DType::kF32, {B, C});
  Tensor tx = RowMajorTT(mx, vt::DType::kF32, {B, C});
  Tensor tw = RowMajorTT(mw, vt::DType::kF32, {C, K});
  Tensor tb = RowMajorTT(mb, vt::DType::kF32, {C});
  Tensor ts = RowMajorTT(ms, vt::DType::kF32, {B, C, sl});
  vt::CausalConv1dArgs a;
  a.silu_activation = true;
  vt::CausalConv1dUpdate(q, to, tx, tw, &tb, ts, a, nullptr);

  // The SAME host pointer, SSM role, equal volume: the fast path would serve
  // (or volume-reshape) the transposed conv shadow as the ssm cache. The
  // refusal must name the role and the refused geometry.
  const int64_t ssm_rows = 2, ssm_cols = (sl + 1) * R / ssm_rows;
  Tensor twk = RowMajorTT(mg, vt::DType::kF32, {3, ssm_cols});
  Tensor tc = RowMajorTT(ms, vt::DType::kF32, {ssm_rows, ssm_cols});
  const std::vector<int32_t> gidx{0, 1, 0};
  tt.Copy(q, mp, gidx.data(), gidx.size() * sizeof(int32_t));
  Tensor tp = RowMajorTT(mp, vt::DType::kI32, {3});
  bool threw = false;
  std::string what;
  try {
    vt::GdnStateGather(q, twk, tc, tp, nullptr);
  } catch (const std::exception& e) {
    threw = true;
    what = e.what();
  }
  CHECK_MESSAGE(threw, "cross-role pointer must be refused, got: ", what);
  CHECK_MESSAGE(what.find("conv_transposed") != std::string::npos,
                "refusal must name the conv role, got: ", what);
  CHECK_MESSAGE(what.find("2x256") != std::string::npos,
                "refusal must name the refused geometry, got: ", what);
  tt.Free(mx);
  tt.Free(mw);
  tt.Free(mb);
  tt.Free(mo);
  tt.Free(ms);
  tt.Free(mg);
  tt.Free(mp);
}

// (d) The decode→prefill alternation on ONE conv-state buffer. The decode step
// commits the tracked slot TRANSPOSED ([sl+1, R] shadow, host-stale), and a
// later prefill-bearing step gathers the SAME buffer in the ssm/cache view at
// a volume-DIFFERING geometry (slots x C*sl = 384 elements against the
// 512-element shadow). Continuous batching makes that transition ordinary, so
// EnsureGdnCacheDevice must fall through to the slow path — EnsureHost
// transposes the shadow back into the caller's order and the refresh
// re-uploads it in this role's geometry — instead of refusing. What stays
// refused is the EQUAL-VOLUME serve or reshape (case (c) above): same numel,
// different geometry is a silent wrong-geometry serve. The committed state is
// the oracle's read-old-then-roll with width == sl (new taps [old1, old2, x];
// the roll moves bytes only), so the gathered rows compare bit-exact.
TEST_CASE("kTENSTORRENT GDN gather serves a conv_transposed slot across the decode-to-prefill transition (#2201)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& tt = *vt::TryGetBackend(DeviceType::kTENSTORRENT);
  const int64_t B = 2, C = 64, K = 4, sl = K - 1;
  uint32_t s = 77003u;
  std::vector<float> w(static_cast<size_t>(C * K)), bias(static_cast<size_t>(C)),
      x(static_cast<size_t>(B * C)), st(static_cast<size_t>(B * C * sl)),
      out(static_cast<size_t>(B * C), 0.0f);
  for (float& v : w) v = 0.4f * GdnLcg(s);
  for (float& v : bias) v = 0.1f * GdnLcg(s);
  for (float& v : st) v = GdnLcg(s);
  for (float& v : x) v = 2.0f * GdnLcg(s);
  std::vector<float> want(static_cast<size_t>(B * C * sl));
  for (int64_t b = 0; b < B; ++b)
    for (int64_t c = 0; c < C; ++c)
      for (int64_t j = 0; j < sl; ++j)
        want[static_cast<size_t>((b * C + c) * sl + j)] =
            j + 1 < sl ? st[static_cast<size_t>((b * C + c) * sl + j + 1)]
                       : x[static_cast<size_t>(b * C + c)];
  // One tracked allocation plays the conv and cache views (the model's shape:
  // one logical state, two views). The gather's three indexed rows reuse slot
  // 0 so a wrong-geometry serve cannot hide behind distinct rows.
  Queue q = tt.CreateQueue();
  void* mx = tt.Alloc(x.size() * sizeof(float));
  void* mw = tt.Alloc(w.size() * sizeof(float));
  void* mb = tt.Alloc(bias.size() * sizeof(float));
  void* mo = tt.Alloc(out.size() * sizeof(float));
  void* ms = tt.Alloc(st.size() * sizeof(float));
  void* mg = tt.Alloc(static_cast<size_t>(3 * C * sl) * sizeof(float));
  void* mp = tt.Alloc(3 * sizeof(int32_t));
  tt.Copy(q, mx, x.data(), x.size() * sizeof(float));
  tt.Copy(q, mw, w.data(), w.size() * sizeof(float));
  tt.Copy(q, mb, bias.data(), bias.size() * sizeof(float));
  tt.Copy(q, mo, out.data(), out.size() * sizeof(float));
  tt.Copy(q, ms, st.data(), st.size() * sizeof(float));
  Tensor to = RowMajorTT(mo, vt::DType::kF32, {B, C});
  Tensor tx = RowMajorTT(mx, vt::DType::kF32, {B, C});
  Tensor tw = RowMajorTT(mw, vt::DType::kF32, {C, K});
  Tensor tb = RowMajorTT(mb, vt::DType::kF32, {C});
  Tensor ts = RowMajorTT(ms, vt::DType::kF32, {B, C, sl});
  vt::CausalConv1dArgs a;
  a.silu_activation = true;
  // Decode: leaves the tracked slot conv_transposed [4x128], host-stale.
  vt::CausalConv1dUpdate(q, to, tx, tw, &tb, ts, a, nullptr);

  // Prefill-bearing step: the SAME pointer as the [slots=2, C*sl=192] cache —
  // volume 384, which differs from the shadow's 512, so the slow path (not a
  // refusal) is the required outcome.
  Tensor tc = RowMajorTT(ms, vt::DType::kF32, {B, C, sl});
  Tensor twk = RowMajorTT(mg, vt::DType::kF32, {3, C, sl});
  const std::vector<int32_t> gidx{0, 1, 0};
  tt.Copy(q, mp, gidx.data(), gidx.size() * sizeof(int32_t));
  Tensor tp = RowMajorTT(mp, vt::DType::kI32, {3});
  bool threw = false;
  std::string what;
  try {
    vt::GdnStateGather(q, twk, tc, tp, nullptr);
  } catch (const std::exception& e) {
    threw = true;
    what = e.what();
  }
  CHECK_MESSAGE(!threw,
                "volume-differing gather on a conv_transposed slot must take "
                "the slow path, got: ",
                what);
  std::vector<float> got(static_cast<size_t>(3 * C * sl), 0.0f);
  tt.Copy(q, got.data(), mg, got.size() * sizeof(float));
  for (int64_t r = 0; r < 3; ++r)
    for (int64_t e = 0; e < C * sl; ++e) {
      const float g =
          got[static_cast<size_t>(r * C * sl + e)];
      const float v = want[static_cast<size_t>(
          gidx[static_cast<size_t>(r)] * C * sl + e)];
      CHECK_MESSAGE(g == v, "gather row " << r << " element " << e
                                          << " is not the oracle state byte");
    }
  // A repeat gather at the served geometry must keep working: the slow path
  // replaced the shadow with this role's logical layout, so the fast path —
  // not a refusal — is the required outcome.
  threw = false;
  try {
    vt::GdnStateGather(q, twk, tc, tp, nullptr);
  } catch (const std::exception& e) {
    threw = true;
    what = e.what();
  }
  CHECK_MESSAGE(!threw,
                "repeat gather at the served geometry must hit the fast path, "
                "got: ",
                what);
  tt.Free(mx);
  tt.Free(mw);
  tt.Free(mb);
  tt.Free(mo);
  tt.Free(ms);
  tt.Free(mg);
  tt.Free(mp);
}

// kKeepQuantDecode: the Q4_K block-decode device path (BACKEND-TENSTORRENT-
// KEEPQUANT W1). The numerics bar is the spec's stated one: the decode's f32
// output is BIT-EXACT against the CPU decoder `vt::cpu::BlockToFloat` — same
// multiply/subtract order, IEEE f32, no FMA — because the reader side of this
// encoding is already pinned bit-exact vs llama.cpp and a device decode that
// "merely" lands inside a band would move the divergence into the dot where
// nobody can attribute it. bf16 tile storage (the W2 dot's input) applies the
// device RNE conversion ON TOP of this exact f32 and is W2's seam, not this
// test's.
TEST_CASE("kTENSTORRENT kKeepQuantDecode matches vt::cpu::BlockToFloat bit-exactly (Q4_K sweep)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kKeepQuantDecode, vt::DeviceType::kTENSTORRENT));

  // block_q4_K = { f16 d; f16 dmin; u8 scales[12]; u8 qs[128]; } (144 bytes)
  const int64_t kBlockBytes = vt::BlockBytes(vt::DType::kQ4_K);
  const int64_t kBlockElems = vt::BlockElems(vt::DType::kQ4_K);
  REQUIRE(kBlockBytes == 144);
  REQUIRE(kBlockElems == 256);

  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  auto decode = reinterpret_cast<vt::KeepQuantDecodeFn>(
      vt::GetOp(vt::OpId::kKeepQuantDecode, vt::DeviceType::kTENSTORRENT));
  Queue q = backend.CreateQueue();

  // Deterministic packed blocks with structural variety: PRNG nibbles and
  // scale bytes (the j >= 4 scale formulas consume bits 6-7, so the scale
  // bytes must be full bytes), and d/dmin drawn from finite f16 magnitudes
  // with both signs — random BYTES would make d/dmin NaN/Inf and the bit
  // comparison vacuous against a NaN-propagating decode.
  std::mt19937 rng(20260905u);
  auto rand_byte = [&rng]() { return static_cast<uint8_t>(rng() & 0xFF); };

  const int64_t rows_list[] = {1, 3, 17};
  const int64_t nb_list[] = {1, 2, 16};
  for (int64_t rows : rows_list) {
    for (int64_t nb : nb_list) {
      const int64_t k = nb * kBlockElems;
      std::vector<uint8_t> packed(rows * nb * kBlockBytes);
      for (int64_t b = 0; b < rows * nb; ++b) {
        uint8_t* blk = packed.data() + b * kBlockBytes;
        const float d = (0.05f + 0.35f * static_cast<float>(rng() % 64) / 64.0f) *
                        ((rng() % 2) != 0 ? 1.0f : -1.0f);
        const float dmin = (0.005f + 0.02f * static_cast<float>(rng() % 32) / 32.0f) *
                           ((rng() % 2) != 0 ? 1.0f : -1.0f);
        const uint16_t d_bits = vt::F32ToF16(d);
        const uint16_t dmin_bits = vt::F32ToF16(dmin);
        std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
        std::memcpy(blk + 2, &dmin_bits, sizeof(dmin_bits));
        for (int i = 0; i < 12; ++i) blk[4 + i] = rand_byte();
        for (int i = 0; i < 128; ++i) blk[16 + i] = rand_byte();
      }

      std::vector<float> oracle(rows * k);
      vt::cpu::BlockToFloat(vt::DType::kQ4_K)(packed.data(), oracle.data(), rows * k);

      void* mem_packed = backend.Alloc(packed.size());
      void* mem_out = backend.Alloc(oracle.size() * sizeof(float));
      backend.Copy(q, mem_packed, packed.data(), packed.size());

      Tensor packed_t =
          Tensor::Contiguous(mem_packed, vt::DType::kQ4_K,
                             Device{vt::DeviceType::kTENSTORRENT, 0}, {rows, nb});
      Tensor out_t = Tensor::Contiguous(mem_out, vt::DType::kF32,
                                        Device{vt::DeviceType::kTENSTORRENT, 0}, {rows, k});
      decode(q, out_t, packed_t);

      std::vector<float> device_out(rows * k, 0.0f);
      backend.Copy(q, device_out.data(), mem_out, oracle.size() * sizeof(float));
      backend.Free(mem_packed);
      backend.Free(mem_out);

      INFO("rows=", rows, " nb=", nb, " K=", k);
      if (std::memcmp(device_out.data(), oracle.data(),
                      oracle.size() * sizeof(float)) != 0) {
        const float* dev = device_out.data();
        int64_t bad = 0;
        for (int64_t i = 0; i < static_cast<int64_t>(oracle.size()); ++i) {
          if (std::memcmp(&dev[i], &oracle[i], sizeof(float)) != 0) {
            if (bad < 4)
              MESSAGE("diff i=", i, " (blk=", i / 256, " col=", i % 256,
                      ") dev=", dev[i], " oracle=", oracle[i]);
            ++bad;
          }
        }
        MESSAGE("total bad: ", bad, " / ", oracle.size());
      }
      CHECK(std::memcmp(device_out.data(), oracle.data(), oracle.size() * sizeof(float)) == 0);
    }
  }
}

// W2: the keep-quant DOT. Enters through vt::MatmulBT's public dispatch (the
// entry point every model matmul helper already uses — ops.cpp:163 routes a
// block-typed [N,K] weight to kMatmulBTQuant), not through a hand-cast op
// pointer, so the test proves the routing a GGUF load actually takes. The
// oracle is the DECODE-based bf16 reference computed here: the weight decoded
// by vt::cpu::BlockToFloat (bit-exact per W1) and BOTH operands rounded to
// bf16 once (RNE), accumulated in f32 in ascending k — the same convention
// the device arm runs (decode f32 → one bf16 RNE → bf16 tile matmul). The
// envelope is the analytic bf16 operand-rounding bound, not a picked
// tolerance: rounding a and w to bf16 perturbs each product by at most
// 2^-8 relative to |a||w|, and the bf16 matmul output adds one more output
// rounding, so |device - ref| <= 1.05 * 2^-8 * (sum_k |a_k w_k| + |ref|)
// must hold elementwise; a bigger gap is a real defect, not noise.
TEST_CASE("kTENSTORRENT kMatmulBTQuant Q4_K via vt::MatmulBT matches the decode-based bf16 oracle") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuant, vt::DeviceType::kTENSTORRENT));

  const int64_t kBlockBytes = vt::BlockBytes(vt::DType::kQ4_K);
  constexpr int64_t N = 16;
  std::mt19937 rng(20260906u);

  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();

  auto widen = [](uint16_t u) {
    uint32_t bits = static_cast<uint32_t>(u) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
  };

  for (int64_t M : {int64_t{1}, int64_t{5}}) {     // M=1 is the decode GEMV
    for (int64_t nb : {int64_t{1}, int64_t{2}}) {  // K spans 1 and 2 blocks
      const int64_t K = nb * 256;
      std::vector<uint8_t> packed(N * nb * kBlockBytes);
      for (int64_t b = 0; b < N * nb; ++b) {
        uint8_t* blk = packed.data() + b * kBlockBytes;
        const float d = (0.05f + 0.35f * static_cast<float>(rng() % 64) / 64.0f) *
                        ((rng() % 2) != 0 ? 1.0f : -1.0f);
        const float dmin = (0.005f + 0.02f * static_cast<float>(rng() % 32) / 32.0f) *
                           ((rng() % 2) != 0 ? 1.0f : -1.0f);
        const uint16_t d_bits = vt::F32ToF16(d);
        const uint16_t dmin_bits = vt::F32ToF16(dmin);
        std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
        std::memcpy(blk + 2, &dmin_bits, sizeof(dmin_bits));
        for (int i = 0; i < 12; ++i) blk[4 + i] = static_cast<uint8_t>(rng() & 0xFF);
        for (int i = 0; i < 128; ++i) blk[16 + i] = static_cast<uint8_t>(rng() & 0xFF);
      }
      std::vector<float> a_f32(M * K);
      for (auto& v : a_f32) v = (static_cast<float>(rng() % 401) - 200.0f) / 100.0f;

      // Oracle: decode (bit-exact W1 authority) -> round ONCE to bf16 ->
      // f32 accumulate in ascending k; plus the analytic envelope.
      std::vector<float> w_f32(N * K);
      vt::cpu::BlockToFloat(vt::DType::kQ4_K)(packed.data(), w_f32.data(), N * K);
      std::vector<uint16_t> a_bf(M * K), w_bf(N * K);
      for (size_t i = 0; i < a_f32.size(); ++i) a_bf[i] = vt::F32ToBF16(a_f32[i]);
      for (size_t i = 0; i < w_f32.size(); ++i) w_bf[i] = vt::F32ToBF16(w_f32[i]);
      std::vector<float> ref(M * N), bound(M * N);
      for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
          float acc = 0.0f, mag = 0.0f;
          for (int64_t k = 0; k < K; ++k) {
            const float p = widen(a_bf[static_cast<size_t>(m) * K + k]) *
                            widen(w_bf[static_cast<size_t>(n) * K + k]);
            acc += p;
            mag += std::fabs(p);
          }
          ref[static_cast<size_t>(m) * N + n] = acc;
          bound[static_cast<size_t>(m) * N + n] =
              1.05f * std::ldexp(1.0f, -8) * (mag + std::fabs(acc));
        }

      void* mem_a = backend.Alloc(M * K * sizeof(uint16_t));
      void* mem_b = backend.Alloc(packed.size());
      void* mem_o = backend.Alloc(std::max<size_t>(M * N, 16) * sizeof(float));
      backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
      backend.Copy(q, mem_b, packed.data(), packed.size());
      Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                      Device{vt::DeviceType::kTENSTORRENT, 0}, {M, K});
      Tensor b_t = Tensor::Contiguous(mem_b, vt::DType::kQ4_K,
                                      Device{vt::DeviceType::kTENSTORRENT, 0}, {N, K});
      Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                      Device{vt::DeviceType::kTENSTORRENT, 0}, {M, N});
      vt::MatmulBT(q, o_t, a_t, b_t);  // the PUBLIC dispatch: block weight -> kMatmulBTQuant
      std::vector<float> out(std::max<size_t>(M * N, 16), 0.0f);
      backend.Copy(q, out.data(), mem_o, out.size() * sizeof(float));
      backend.Free(mem_a);
      backend.Free(mem_b);
      backend.Free(mem_o);

      float worst = 0.0f, worst_ratio = 0.0f;
      for (int64_t i = 0; i < M * N; ++i) {
        const float diff = std::fabs(out[static_cast<size_t>(i)] -
                                     ref[static_cast<size_t>(i)]);
        worst = std::max(worst, diff);
        worst_ratio = std::max(worst_ratio, diff / bound[static_cast<size_t>(i)]);
        CHECK(std::isfinite(out[static_cast<size_t>(i)]));
        CHECK_MESSAGE(diff <= bound[static_cast<size_t>(i)],
                      "M=" << M << " K=" << K << " i=" << i << " out=" << out[i]
                           << " ref=" << ref[i] << " bound=" << bound[i]);
      }
      MESSAGE("kMatmulBTQuant M=", M, " K=", K,
              ": worst_abs=", worst, " worst bound-ratio=", worst_ratio);
    }
  }
}

// KEEPQUANT W4b (issue #3031), RED-FIRST for the int8-dot lever. The dense
// keep-quant dot today runs in the BF16 domain (decode-to-bf16-then-tile-
// matmul, the W4a wave-3b-1 arm), so it cannot agree BIT-EXACTLY with the
// CPU integer vec_dot the CPU provider runs — this test is RED on the W4a
// code and turns GREEN only when the device dot moves into the quantized
// domain (activation quantized once to the encoding the vec_dot pairs with
// the weight, the upstream 8-lane integer dot, the per-block scale applied
// once). The oracle is the pinned llama.cpp b10451 chain already ported
// bit-exact host-side, PER ENCODING: quantize_row_q8_K_ref
// (cpu_quant_act.cpp:88, via vt::cpu::BlockFromFloat(kQ8_K)) or
// quantize_row_q8_0_ref (cpu_quant_act.cpp:47, via
// vt::cpu::BlockFromFloat(kQ8_0)) — the QuantTraits vec_dot_type pairing,
// the fact QuantActRowBytes derives — then that encoding's own
// ggml_vec_dot_q4_K_q8_K_generic (cpu_quant_dot.cpp:285),
// ggml_vec_dot_q5_K_q8_K_generic (:367), ggml_vec_dot_q6_K_q8_K_generic
// (:457) or ggml_vec_dot_q8_0_q8_0_generic (~170), via
// vt::cpu::BlockVecDot — the K-quants have no ISA tier, the generic scalar
// IS the selected kernel. The sweep pins the WHOLE registered set {Q4_K,
// Q5_K, Q6_K, Q8_0} — a Q4_K-only sweep left the bit-exact claim unpinned
// for three of the four encodings the lever registers — with the W1 shape
// pattern per encoding: weight rows {1,3,17} x blocks {1,2,16}, with
// activation rows {1,3}. An f32-activation case per encoding pins the
// domain contract the lever exists to restore: the CPU provider quantizes
// the F32 master, so the device must not take the bf16 detour the W4a arm
// takes today.
TEST_CASE("kTENSTORRENT kMatmulBTQuant matches the CPU integer vec_dot bit-exactly across the registered set (int8-dot sweep)") {
  // W4b landing decision: the lever is OP-LEVEL and DEFAULT OFF. On default
  // this f32-out dispatch serves the W4a grouped arm (BF16 domain), and this
  // test's oracle is the CPU integer vec_dot that arm does not compute — so
  // skip loudly instead of redding on the wrong arm; run under
  // VT_TT_KEEPQUANT_INT8DOT=1.
  if (const char* lever = std::getenv("VT_TT_KEEPQUANT_INT8DOT");
      lever == nullptr || lever[0] == '\0' || std::strcmp(lever, "0") == 0) {
    MESSAGE("SKIPPED: set VT_TT_KEEPQUANT_INT8DOT=1 — this sweep asserts the "
            "int8-dot lever (bit-exact vs the CPU integer vec_dot); the "
            "default dispatch is the W4a grouped arm");
    return;
  }
  ::setenv("VT_TT_KEEPQUANT_INT8DOT", "1", 1);  // canonical opt-in; the dispatch reads the env live per call
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuant, vt::DeviceType::kTENSTORRENT));

  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();

  // The registered set, in the kernel's ARG_ENC order (0/1/2/3/4/5/6): Q4_K
  // is W1's vehicle, Q5_K/Q6_K/Q8_0 the W3 decode set, IQ3_XXS the
  // QUANT-GGUF-IQ-TENSTORRENT wave-1 addition (enc_sel 4) and IQ2_XXS/IQ2_S
  // the wave-2 addition (enc_sel 5/6), all q8_K pairings, and Q3_K the
  // wave-3 addition (enc_sel 7, the min-term K-quant — not the codebook
  // family), and IQ3_S the tenstorrent-gsq-keepquant wave-1 addition
  // (enc_sel 8, the IQ3_XXS cousin with separate scales, signs and qh
  // high-index bits), and IQ4_XS the tenstorrent-gsq-keepquant wave-2
  // addition (enc_sel 9, the 4-bit non-linear codebook with a 6-bit
  // super-block scale splice).
  const vt::DType encodings[] = {
      vt::DType::kQ4_K,   vt::DType::kQ5_K,
      vt::DType::kQ6_K,   vt::DType::kQ8_0,
      vt::DType::kIQ3_XXS, vt::DType::kIQ2_XXS,
      vt::DType::kIQ2_S,  vt::DType::kQ3_K, vt::DType::kIQ3_S,
      vt::DType::kIQ4_XS, vt::DType::kIQ2_XS,
      vt::DType::kQ2_K, vt::DType::kIQ1_S, vt::DType::kIQ1_M};
  for (const vt::DType enc : encodings) {
    const int64_t kBlockBytes = vt::BlockBytes(enc);
    const int64_t kBlockElems = vt::BlockElems(enc);
    if (enc == vt::DType::kQ4_K) {
      REQUIRE(kBlockBytes == 144);
      REQUIRE(kBlockElems == 256);
    } else if (enc == vt::DType::kQ2_K) {
      // tenstorrent-gsq-keepquant wave 4: the 28 census tensors (ffn + embd)
      // join the sweep — the block-size pin is part of the wave contract.
      REQUIRE(kBlockBytes == 84);
      REQUIRE(kBlockElems == 256);
    } else if (enc == vt::DType::kIQ1_S || enc == vt::DType::kIQ1_M) {
      // tenstorrent-gsq-keepquant wave 5: the IQ1 pair (8 ffn-tail census
      // tensors) joins the sweep — the block-size pins are part of the wave
      // contract.
      REQUIRE(kBlockBytes == (enc == vt::DType::kIQ1_S ? 50 : 56));
      REQUIRE(kBlockElems == 256);
    } else if (enc == vt::DType::kQ5_K) {
      REQUIRE(kBlockBytes == 176);
      REQUIRE(kBlockElems == 256);
    } else if (enc == vt::DType::kQ6_K) {
      REQUIRE(kBlockBytes == 210);
      REQUIRE(kBlockElems == 256);
    } else if (enc == vt::DType::kIQ3_XXS) {
      REQUIRE(kBlockBytes == 98);
      REQUIRE(kBlockElems == 256);
    } else if (enc == vt::DType::kIQ2_XXS) {
      REQUIRE(kBlockBytes == 66);
      REQUIRE(kBlockElems == 256);
    } else if (enc == vt::DType::kIQ2_S) {
      REQUIRE(kBlockBytes == 82);
      REQUIRE(kBlockElems == 256);
    } else if (enc == vt::DType::kQ3_K) {
      REQUIRE(kBlockBytes == 110);
      REQUIRE(kBlockElems == 256);
    } else if (enc == vt::DType::kIQ3_S) {
      REQUIRE(kBlockBytes == 110);
      REQUIRE(kBlockElems == 256);
    } else if (enc == vt::DType::kIQ4_XS) {
      REQUIRE(kBlockBytes == 136);
      REQUIRE(kBlockElems == 256);
    } else if (enc == vt::DType::kIQ2_XS) {
      REQUIRE(kBlockBytes == 74);
      REQUIRE(kBlockElems == 256);
    } else if (enc == vt::DType::kQ8_0) {
      REQUIRE(kBlockBytes == 34);
      REQUIRE(kBlockElems == 32);
    }
    const vt::DType act_enc =
        enc == vt::DType::kQ8_0 ? vt::DType::kQ8_0 : vt::DType::kQ8_K;
    auto quant_act = vt::cpu::BlockFromFloat(act_enc);
    auto vec_dot = vt::cpu::BlockVecDot(enc);
    REQUIRE(quant_act != nullptr);
    REQUIRE(vec_dot != nullptr);

    // Per-encoding PRNG and packed-block generator (the decode sweeps' W1/W3
    // fixtures): PRNG payload bytes and full-byte scale sets where the
    // encoding has them (the K-quant scale formulas consume bits 6-7), d
    // (and dmin where the encoding has one) drawn from finite f16 magnitudes
    // with both signs so the bit comparison never degenerates on NaN/Inf
    // propagation.
    std::mt19937 rng(20260909u);
    auto rand_byte = [&rng]() { return static_cast<uint8_t>(rng() & 0xFF); };
    auto rand_f16_signed = [&rng](float lo, float span) {
      return (lo + span * static_cast<float>(rng() % 64) / 64.0f) *
             ((rng() % 2) != 0 ? 1.0f : -1.0f);
    };
    auto fill_block = [&](uint8_t* blk) {
      if (enc == vt::DType::kQ4_K) {
        // block_q4_K = { f16 d; f16 dmin; u8 scales[12]; u8 qs[128] } (144B)
        const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
        const uint16_t dmin_bits = vt::F32ToF16(rand_f16_signed(0.005f, 0.02f));
        std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
        std::memcpy(blk + 2, &dmin_bits, sizeof(dmin_bits));
        for (int i = 0; i < 12; ++i) blk[4 + i] = rand_byte();
        for (int i = 0; i < 128; ++i) blk[16 + i] = rand_byte();
      } else if (enc == vt::DType::kQ5_K) {
        // block_q5_K = { f16 d; f16 dmin; u8 scales[12]; u8 qh[32];
        //                u8 qs[128] } (176B)
        const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
        const uint16_t dmin_bits = vt::F32ToF16(rand_f16_signed(0.005f, 0.02f));
        std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
        std::memcpy(blk + 2, &dmin_bits, sizeof(dmin_bits));
        for (int i = 0; i < 12; ++i) blk[4 + i] = rand_byte();
        for (int i = 0; i < 32; ++i) blk[16 + i] = rand_byte();   // qh
        for (int i = 0; i < 128; ++i) blk[48 + i] = rand_byte();  // qs
      } else if (enc == vt::DType::kQ6_K) {
        // block_q6_K = { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d }
        // (210B) — the scales are signed bytes both sides read as int8, so
        // random bytes stay in the dot's bit-exact class.
        for (int i = 0; i < 128; ++i) blk[0 + i] = rand_byte();   // ql
        for (int i = 0; i < 64; ++i) blk[128 + i] = rand_byte();  // qh
        for (int i = 0; i < 16; ++i) blk[192 + i] = rand_byte();  // scales
        const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
        std::memcpy(blk + 208, &d_bits, sizeof(d_bits));
      } else if (enc == vt::DType::kIQ3_XXS) {
        // block_iq3_xxs = { f16 d; u8 qs[96] } (98B) — qs[0..63] are grid
        // indices (a full byte indexes kIq3xxsGrid[256]), qs[64..71] are the
        // per-32 scale+sign u32s: the top nibble is the 4-bit scale, bits
        // 0..27 four 7-bit sign selectors — random bytes stay in range.
        const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
        std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
        for (int i = 0; i < 96; ++i) blk[2 + i] = rand_byte();
      } else if (enc == vt::DType::kIQ2_XXS) {
        // block_iq2_xxs = { f16 d; u16 qs[32] } (66B) — the 32 u16s are 8 u32
        // pairs: the pair's low bytes index kIq2xxsGrid[256] (full byte, in
        // range), the second u32's top nibble is the 4-bit scale and bits
        // 0..27 four 7-bit kKsignsIq2xs selectors — random bytes stay in range.
        const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
        std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
        for (int i = 0; i < 64; ++i) blk[2 + i] = rand_byte();
      } else if (enc == vt::DType::kIQ2_S) {
        // block_iq2_s = { f16 d; u8 qs[64]; u8 qh[8]; u8 scales[8] } (82B) —
        // qs bytes are 8-bit kIq2sGrid[1024] indices widened by 2 qh bits (in
        // range), qh any, scales two 4-bit scale nibbles per 32-sub-block.
        const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
        std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
        for (int i = 0; i < 64; ++i) blk[2 + i] = rand_byte();   // qs
        for (int i = 0; i < 8; ++i) blk[66 + i] = rand_byte();   // qh
        for (int i = 0; i < 8; ++i) blk[74 + i] = rand_byte();   // scales
      } else if (enc == vt::DType::kQ3_K) {
        // block_q3_K = { u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d }
        // (110B) — hmask any (the clear-bit subtraction reads single bits),
        // qs the packed low 2 bits, scales the 6-bit splice bytes the
        // kmask1/kmask2 split consumes; random bytes stay in the dot's
        // bit-exact class (the same argument as Q6_K's signed scales).
        for (int i = 0; i < 32; ++i) blk[0 + i] = rand_byte();   // hmask
        for (int i = 0; i < 64; ++i) blk[32 + i] = rand_byte();  // qs
        for (int i = 0; i < 12; ++i) blk[96 + i] = rand_byte();  // scales
        const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
        std::memcpy(blk + 108, &d_bits, sizeof(d_bits));
      } else if (enc == vt::DType::kIQ3_S) {
        // block_iq3_s = { f16 d; u8 qs[64]; u8 qh[8]; u8 signs[32];
        // u8 scales[4] } (110B) — qs bytes are 8-bit kIq3sGrid[512] indices
        // widened by one qh bit (in range), qh any, signs any (single-bit
        // selectors), scales two 4-bit scale nibbles per 32-sub-block pair;
        // random bytes stay in the dot's bit-exact class.
        const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
        std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
        for (int i = 0; i < 64; ++i) blk[2 + i] = rand_byte();    // qs
        for (int i = 0; i < 8; ++i) blk[66 + i] = rand_byte();    // qh
        for (int i = 0; i < 32; ++i) blk[74 + i] = rand_byte();   // signs
        for (int i = 0; i < 4; ++i) blk[106 + i] = rand_byte();   // scales
      } else if (enc == vt::DType::kIQ4_XS) {
        // block_iq4_xs = { f16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128] }
        // (136B) — each qs byte is two kvalues_iq4nl[16] indices (nibbles, in
        // range), scales_l the low 4 bits of six sub-block scales and
        // scales_h their 2-bit-high splices; random bytes stay in the dot's
        // bit-exact class.
        const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
        std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
        for (int i = 0; i < 134; ++i) blk[2 + i] = rand_byte();
      } else if (enc == vt::DType::kIQ2_XS) {
        // block_iq2_xs = { f16 d; u16 qs[32]; u8 scales[8] } (74B) — each qs
        // u16 is a 9-bit kIq2xsGrid[512] index plus a 7-bit kKsignsIq2xs
        // selector (any bytes in range), scales two 4-bit sub-scale nibbles
        // per 32-sub-block; random bytes stay in the dot's bit-exact class.
        const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
        std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
        for (int i = 0; i < 72; ++i) blk[2 + i] = rand_byte();  // qs + scales
      } else if (enc == vt::DType::kQ2_K) {
        // block_q2_K = { u8 scales[16]; u8 qs[64]; f16 d; f16 dmin } (84B) —
        // scales lead, deltas trail; each scale byte is a 4-bit sub-scale d
        // (low) + 4-bit sub-min (high) pair; qs bytes are four 2-bit quants.
        // Random bytes stay in the dot's bit-exact class (all-int accumulation).
        const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
        const uint16_t dmin_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
        for (int i = 0; i < 80; ++i) blk[i] = rand_byte();  // scales + qs
        std::memcpy(blk + 80, &d_bits, sizeof(d_bits));
        std::memcpy(blk + 82, &dmin_bits, sizeof(dmin_bits));
      } else if (enc == vt::DType::kIQ1_S) {
        // block_iq1_s = { f16 d; u8 qs[32]; u16 qh[8] } (50B) — qs the low
        // 8 grid-index bits (any byte, the index is 11 bits), qh any (3-bit
        // high index bits + 3-bit scale + sign bit); random bytes stay in
        // the dot's bit-exact class (all-int accumulation).
        const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
        std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
        for (int i = 0; i < 48; ++i) blk[2 + i] = rand_byte();  // qs + qh
      } else if (enc == vt::DType::kIQ1_M) {
        // block_iq1_m = { u8 qs[32]; u8 qh[16]; u8 scales[8] } (56B) — no
        // f16 field: the scale is SPLICED from the top nibbles of scales;
        // qs/qh any (11-bit index), scales any (3-bit sub-block scales).
        // The splice does NOT stay bit-exact under fully random bytes: the
        // spliced f16's exponent bits (packed 14-10, fed by sb[1] and
        // sb[3]) are then random too, so the scale hits inf/NaN — both the
        // device and the oracle produce them, with different payloads, and
        // the bit-exact class is gone. Constrain the exponent instead.
        for (int i = 0; i < 56; ++i) blk[i] = rand_byte();
        blk[55] = 0x30;     // sb[7]: packed bits 15-12 = 0b0011 (sign 0,
        blk[53] &= 0x03;    // sb[5]: exponent bit 10 = 0 (exp 12, d in [8,16))
      } else if (enc == vt::DType::kQ8_0) {
        const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
        std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
        for (int i = 0; i < 32; ++i) blk[2 + i] = rand_byte();  // full int8 range
      }
    };

    const int64_t n_list[] = {1, 3, 17};
    const int64_t nb_list[] = {1, 2, 16};
    const int64_t m_list[] = {1, 3};
    for (int64_t M : m_list) {
      for (int64_t N : n_list) {
        for (int64_t nb : nb_list) {
          const int64_t K = nb * kBlockElems;
          std::vector<uint8_t> packed(N * nb * kBlockBytes);
          for (int64_t b = 0; b < N * nb; ++b)
            fill_block(packed.data() + b * kBlockBytes);
          // f32 master, rounded ONCE to bf16: the activation the device arm
          // actually receives. The oracle quantizes the bf16 values widened
          // back to f32 (lossless), so the ONLY domain difference left is the
          // dot's.
          std::vector<float> a_f32(M * K);
          for (auto& v : a_f32) v = (static_cast<float>(rng() % 401) - 200.0f) / 100.0f;
          std::vector<uint16_t> a_bf(M * K);
          for (size_t i = 0; i < a_f32.size(); ++i) a_bf[i] = vt::F32ToBF16(a_f32[i]);
          std::vector<float> a_q32(M * K);
          for (size_t i = 0; i < a_f32.size(); ++i) a_q32[i] = vt::BF16ToF32(a_bf[i]);

          // CPU integer oracle: quantize each activation row once, then one
          // nrc==1 vec_dot per (m, n) pair — the exact work the CPU provider
          // does per output element.
          const size_t y_row_bytes = vt::cpu::QuantActRowBytes(enc, K);
          std::vector<uint8_t> y(M * y_row_bytes);
          for (int64_t m = 0; m < M; ++m)
            quant_act(a_q32.data() + m * K, y.data() + m * y_row_bytes, K);
          std::vector<float> oracle(M * N);
          for (int64_t m = 0; m < M; ++m)
            for (int64_t n = 0; n < N; ++n)
              vec_dot(static_cast<int>(K), &oracle[static_cast<size_t>(m) * N + n],
                      /*bs=*/0, packed.data() + static_cast<size_t>(n) * nb * kBlockBytes,
                      /*bx=*/0, y.data() + m * y_row_bytes, /*by=*/0, /*nrc=*/1);

          void* mem_a = backend.Alloc(a_bf.size() * sizeof(uint16_t));
          void* mem_b = backend.Alloc(packed.size());
          void* mem_o = backend.Alloc(std::max<size_t>(M * N, 16) * sizeof(float));
          backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
          backend.Copy(q, mem_b, packed.data(), packed.size());
          Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                          Device{vt::DeviceType::kTENSTORRENT, 0}, {M, K});
          Tensor b_t = Tensor::Contiguous(mem_b, enc,
                                          Device{vt::DeviceType::kTENSTORRENT, 0}, {N, K});
          Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                          Device{vt::DeviceType::kTENSTORRENT, 0}, {M, N});
          vt::MatmulBT(q, o_t, a_t, b_t);
          std::vector<float> out(std::max<size_t>(M * N, 16), 0.0f);
          backend.Copy(q, out.data(), mem_o, out.size() * sizeof(float));
          backend.Free(mem_a);
          backend.Free(mem_b);
          backend.Free(mem_o);

          INFO("enc=", static_cast<int>(enc), " M=", M, " N=", N, " nb=", nb,
               " K=", K);
          if (std::memcmp(out.data(), oracle.data(), oracle.size() * sizeof(float)) != 0) {
            int64_t bad = 0;
            for (int64_t i = 0; i < static_cast<int64_t>(oracle.size()); ++i) {
              if (std::memcmp(&out[static_cast<size_t>(i)], &oracle[static_cast<size_t>(i)],
                              sizeof(float)) != 0) {
                if (bad < 4)
                  MESSAGE("diff m=", i / N, " n=", i % N, " dev=", out[static_cast<size_t>(i)],
                          " oracle=", oracle[static_cast<size_t>(i)]);
                ++bad;
              }
            }
            MESSAGE("total bad: ", bad, " / ", oracle.size(),
                    " (bf16-domain dot vs the integer reference)");
          }
          CHECK(std::memcmp(out.data(), oracle.data(), oracle.size() * sizeof(float)) == 0);
        }
      }
    }

    // The f32-activation domain contract, per encoding: the CPU provider
    // quantizes the f32 master directly (no bf16 detour), so the device must
    // too. One small shape; the same bit-exact bar.
    {
      const int64_t M = 2, N = 5, nb = 2;
      const int64_t K = nb * kBlockElems;
      std::vector<uint8_t> packed(N * nb * kBlockBytes);
      for (int64_t b = 0; b < N * nb; ++b)
        fill_block(packed.data() + b * kBlockBytes);
      std::vector<float> a_f32(M * K);
      for (auto& v : a_f32) v = (static_cast<float>(rng() % 401) - 200.0f) / 100.0f;

      const size_t y_row_bytes = vt::cpu::QuantActRowBytes(enc, K);
      std::vector<uint8_t> y(M * y_row_bytes);
      for (int64_t m = 0; m < M; ++m)
        quant_act(a_f32.data() + m * K, y.data() + m * y_row_bytes, K);
      std::vector<float> oracle(M * N);
      for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n)
          vec_dot(static_cast<int>(K), &oracle[static_cast<size_t>(m) * N + n],
                  /*bs=*/0, packed.data() + static_cast<size_t>(n) * nb * kBlockBytes,
                  /*bx=*/0, y.data() + m * y_row_bytes, /*by=*/0, /*nrc=*/1);

      void* mem_a = backend.Alloc(a_f32.size() * sizeof(float));
      void* mem_b = backend.Alloc(packed.size());
      void* mem_o = backend.Alloc(std::max<size_t>(M * N, 16) * sizeof(float));
      backend.Copy(q, mem_a, a_f32.data(), a_f32.size() * sizeof(float));
      backend.Copy(q, mem_b, packed.data(), packed.size());
      Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kF32,
                                      Device{vt::DeviceType::kTENSTORRENT, 0}, {M, K});
      Tensor b_t = Tensor::Contiguous(mem_b, enc,
                                      Device{vt::DeviceType::kTENSTORRENT, 0}, {N, K});
      Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                      Device{vt::DeviceType::kTENSTORRENT, 0}, {M, N});
      vt::MatmulBT(q, o_t, a_t, b_t);
      std::vector<float> out(M * N, 0.0f);
      backend.Copy(q, out.data(), mem_o, out.size() * sizeof(float));
      backend.Free(mem_a);
      backend.Free(mem_b);
      backend.Free(mem_o);

      INFO("f32 activation: enc=", static_cast<int>(enc), " M=", M, " N=", N,
           " nb=", nb, " K=", K);
      if (std::memcmp(out.data(), oracle.data(), oracle.size() * sizeof(float)) != 0) {
        int64_t bad = 0;
        for (int64_t i = 0; i < static_cast<int64_t>(oracle.size()); ++i) {
          if (std::memcmp(&out[static_cast<size_t>(i)], &oracle[static_cast<size_t>(i)],
                          sizeof(float)) != 0) {
            if (bad < 4)
              MESSAGE("diff m=", i / N, " n=", i % N, " dev=", out[static_cast<size_t>(i)],
                      " oracle=", oracle[static_cast<size_t>(i)]);
            ++bad;
          }
        }
        MESSAGE("total bad: ", bad, " / ", oracle.size(),
                " (f32 activation took the bf16 detour)");
      }
      CHECK(std::memcmp(out.data(), oracle.data(), oracle.size() * sizeof(float)) == 0);
    }
    MESSAGE("int8-dot sweep enc=", static_cast<int>(enc),
            ": the 18-shape bf16-act sweep + the f32-act leg bit-exact vs the CPU vec_dot");
  }
}

// QUANT-GGUF-IQ-TENSTORRENT repair: pin the DEFAULT-path dispatch the row's
// reachability claim names. IQ3_XXS has no W4a grouped arm, so its only serve
// is the int8-dot kernel, dispatched REGARDLESS of VT_TT_KEEPQUANT_INT8DOT
// (tenstorrent_ops.cpp MatmulBTQuantKernel). The sweep above self-gates on the
// env, so deleting the dispatch override alone left every gate green: the
// route pin (OpRegistered) proves admission, not dispatch. This leg runs the
// IQ3_XXS device decode with the lever explicitly UNSET and holds the same
// bit-exact bar vs the CPU integer vec_dot oracle — the dispatch-only
// mutation (override deleted, admission kept) refuses the route here and goes
// RED.
TEST_CASE("kTENSTORRENT kMatmulBTQuant IQ3_XXS serves the int8-dot arm on the DEFAULT path (VT_TT_KEEPQUANT_INT8DOT unset)") {
  // Default-path contract: the lever env must NOT be set for this leg. Save
  // and restore the ambient value whatever happens (the microbench pattern).
  const char* prev_lever = std::getenv("VT_TT_KEEPQUANT_INT8DOT");
  const bool had_lever = prev_lever != nullptr;
  const std::string saved_lever =
      had_lever ? std::string(prev_lever) : std::string();
  struct RestoreLeverEnv {
    const bool had;
    const std::string saved;
    ~RestoreLeverEnv() {
      if (had) ::setenv("VT_TT_KEEPQUANT_INT8DOT", saved.c_str(), 1);
      else ::unsetenv("VT_TT_KEEPQUANT_INT8DOT");
    }
  } restore_lever_env{had_lever, saved_lever};
  ::unsetenv("VT_TT_KEEPQUANT_INT8DOT");

  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuant, vt::DeviceType::kTENSTORRENT));

  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();

  const vt::DType enc = vt::DType::kIQ3_XXS;
  const int64_t kBlockBytes = vt::BlockBytes(enc);
  const int64_t kBlockElems = vt::BlockElems(enc);
  REQUIRE(kBlockBytes == 98);
  REQUIRE(kBlockElems == 256);
  auto quant_act = vt::cpu::BlockFromFloat(vt::DType::kQ8_K);
  auto vec_dot = vt::cpu::BlockVecDot(enc);
  REQUIRE(quant_act != nullptr);
  REQUIRE(vec_dot != nullptr);

  // The sweep's IQ3_XXS packed generator and PRNG style, fixed seed.
  std::mt19937 rng(20260909u);
  auto rand_byte = [&rng]() { return static_cast<uint8_t>(rng() & 0xFF); };
  auto rand_f16_signed = [&rng](float lo, float span) {
    return (lo + span * static_cast<float>(rng() % 64) / 64.0f) *
           ((rng() % 2) != 0 ? 1.0f : -1.0f);
  };
  auto fill_block = [&](uint8_t* blk) {
    // block_iq3_xxs = { f16 d; u8 qs[96] } (98B): qs[0..63] grid indices,
    // qs[64..71] per-32 scale+sign u32s; random bytes stay in range.
    const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
    std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
    for (int i = 0; i < 96; ++i) blk[2 + i] = rand_byte();
  };

  // Decode shapes (M=1 is the GEMV), both one- and multi-block K.
  for (int64_t M : {int64_t{1}, int64_t{3}}) {
    for (int64_t N : {int64_t{1}, int64_t{17}}) {
      for (int64_t nb : {int64_t{1}, int64_t{16}}) {
        const int64_t K = nb * kBlockElems;
        std::vector<uint8_t> packed(N * nb * kBlockBytes);
        for (int64_t b = 0; b < N * nb; ++b)
          fill_block(packed.data() + b * kBlockBytes);
        std::vector<float> a_f32(M * K);
        for (auto& v : a_f32) v = (static_cast<float>(rng() % 401) - 200.0f) / 100.0f;
        std::vector<uint16_t> a_bf(M * K);
        for (size_t i = 0; i < a_f32.size(); ++i) a_bf[i] = vt::F32ToBF16(a_f32[i]);
        std::vector<float> a_q32(M * K);
        for (size_t i = 0; i < a_f32.size(); ++i) a_q32[i] = vt::BF16ToF32(a_bf[i]);

        // CPU integer oracle: quantize each activation row once, one nrc==1
        // vec_dot per (m, n) pair.
        const size_t y_row_bytes = vt::cpu::QuantActRowBytes(enc, K);
        std::vector<uint8_t> y(M * y_row_bytes);
        for (int64_t m = 0; m < M; ++m)
          quant_act(a_q32.data() + m * K, y.data() + m * y_row_bytes, K);
        std::vector<float> oracle(M * N);
        for (int64_t m = 0; m < M; ++m)
          for (int64_t n = 0; n < N; ++n)
            vec_dot(static_cast<int>(K), &oracle[static_cast<size_t>(m) * N + n],
                    /*bs=*/0, packed.data() + static_cast<size_t>(n) * nb * kBlockBytes,
                    /*bx=*/0, y.data() + m * y_row_bytes, /*by=*/0, /*nrc=*/1);

        void* mem_a = backend.Alloc(a_bf.size() * sizeof(uint16_t));
        void* mem_b = backend.Alloc(packed.size());
        void* mem_o = backend.Alloc(std::max<size_t>(M * N, 16) * sizeof(float));
        backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
        backend.Copy(q, mem_b, packed.data(), packed.size());
        Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                        Device{vt::DeviceType::kTENSTORRENT, 0}, {M, K});
        Tensor b_t = Tensor::Contiguous(mem_b, enc,
                                        Device{vt::DeviceType::kTENSTORRENT, 0}, {N, K});
        Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                        Device{vt::DeviceType::kTENSTORRENT, 0}, {M, N});
        vt::MatmulBT(q, o_t, a_t, b_t);  // route refusal under the mutation throws here
        std::vector<float> out(std::max<size_t>(M * N, 16), 0.0f);
        backend.Copy(q, out.data(), mem_o, out.size() * sizeof(float));
        backend.Free(mem_a);
        backend.Free(mem_b);
        backend.Free(mem_o);

        INFO("default path, enc=", static_cast<int>(enc), " M=", M, " N=", N,
             " nb=", nb, " K=", K);
        if (std::memcmp(out.data(), oracle.data(), oracle.size() * sizeof(float)) != 0) {
          int64_t bad = 0;
          for (int64_t i = 0; i < static_cast<int64_t>(oracle.size()); ++i) {
            if (std::memcmp(&out[static_cast<size_t>(i)], &oracle[static_cast<size_t>(i)],
                            sizeof(float)) != 0) {
              if (bad < 4)
                MESSAGE("diff m=", i / N, " n=", i % N, " dev=", out[static_cast<size_t>(i)],
                        " oracle=", oracle[static_cast<size_t>(i)]);
              ++bad;
            }
          }
          MESSAGE("total bad: ", bad, " / ", oracle.size(),
                  " (default-path IQ3_XXS did not take the int8-dot arm)");
        }
        CHECK(std::memcmp(out.data(), oracle.data(), oracle.size() * sizeof(float)) == 0);
      }
    }
  }
  MESSAGE("default-path IQ3_XXS decode: the 8-shape leg bit-exact vs the CPU vec_dot "
          "with VT_TT_KEEPQUANT_INT8DOT unset");
}

// QUANT-GGUF-IQ-TENSTORRENT wave 2: the same default-path pin for the two
// encodings wave 2 registers — IQ2_XXS (enc_sel 5) and IQ2_S (enc_sel 6).
// Neither has a W4a grouped arm, so the int8-dot kernel is their only serve,
// dispatched REGARDLESS of VT_TT_KEEPQUANT_INT8DOT; the lever is explicitly
// UNSET here and the same bit-exact bar holds vs the CPU integer vec_dot
// oracles (VecDotIQ2_XXSQ8_K, VecDotIQ2_SQ8_K). The dispatch-only mutation
// (override deleted, admission kept) refuses the route here and goes RED.
TEST_CASE("kTENSTORRENT kMatmulBTQuant IQ2_XXS and IQ2_S serve the int8-dot arm on the DEFAULT path (VT_TT_KEEPQUANT_INT8DOT unset)") {
  // Default-path contract: the lever env must NOT be set for this leg.
  const char* prev_lever = std::getenv("VT_TT_KEEPQUANT_INT8DOT");
  const bool had_lever = prev_lever != nullptr;
  const std::string saved_lever =
      had_lever ? std::string(prev_lever) : std::string();
  struct RestoreLeverEnv {
    const bool had;
    const std::string saved;
    ~RestoreLeverEnv() {
      if (had) ::setenv("VT_TT_KEEPQUANT_INT8DOT", saved.c_str(), 1);
      else ::unsetenv("VT_TT_KEEPQUANT_INT8DOT");
    }
  } restore_lever_env{had_lever, saved_lever};
  ::unsetenv("VT_TT_KEEPQUANT_INT8DOT");

  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuant, vt::DeviceType::kTENSTORRENT));

  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();

  const vt::DType encodings[] = {vt::DType::kIQ2_XXS, vt::DType::kIQ2_S};
  for (const vt::DType enc : encodings) {
    const int64_t kBlockBytes = vt::BlockBytes(enc);
    const int64_t kBlockElems = vt::BlockElems(enc);
    if (enc == vt::DType::kIQ2_XXS) {
      REQUIRE(kBlockBytes == 66);
      REQUIRE(kBlockElems == 256);
    } else {
      REQUIRE(kBlockBytes == 82);
      REQUIRE(kBlockElems == 256);
    }
    auto quant_act = vt::cpu::BlockFromFloat(vt::DType::kQ8_K);
    auto vec_dot = vt::cpu::BlockVecDot(enc);
    REQUIRE(quant_act != nullptr);
    REQUIRE(vec_dot != nullptr);

    // The sweep's PRNG style and packed generators, fixed seed.
    std::mt19937 rng(20260909u);
    auto rand_byte = [&rng]() { return static_cast<uint8_t>(rng() & 0xFF); };
    auto rand_f16_signed = [&rng](float lo, float span) {
      return (lo + span * static_cast<float>(rng() % 64) / 64.0f) *
             ((rng() % 2) != 0 ? 1.0f : -1.0f);
    };
    auto fill_block = [&](uint8_t* blk) {
      const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
      std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
      if (enc == vt::DType::kIQ2_XXS) {
        for (int i = 0; i < 64; ++i) blk[2 + i] = rand_byte();  // qs (u16[32])
      } else {
        for (int i = 0; i < 64; ++i) blk[2 + i] = rand_byte();   // qs
        for (int i = 0; i < 8; ++i) blk[66 + i] = rand_byte();   // qh
        for (int i = 0; i < 8; ++i) blk[74 + i] = rand_byte();   // scales
      }
    };

    // Decode shapes (M=1 is the GEMV), both one- and multi-block K.
    for (int64_t M : {int64_t{1}, int64_t{3}}) {
      for (int64_t N : {int64_t{1}, int64_t{17}}) {
        for (int64_t nb : {int64_t{1}, int64_t{16}}) {
          const int64_t K = nb * kBlockElems;
          std::vector<uint8_t> packed(N * nb * kBlockBytes);
          for (int64_t b = 0; b < N * nb; ++b)
            fill_block(packed.data() + b * kBlockBytes);
          std::vector<float> a_f32(M * K);
          for (auto& v : a_f32) v = (static_cast<float>(rng() % 401) - 200.0f) / 100.0f;
          std::vector<uint16_t> a_bf(M * K);
          for (size_t i = 0; i < a_f32.size(); ++i) a_bf[i] = vt::F32ToBF16(a_f32[i]);
          std::vector<float> a_q32(M * K);
          for (size_t i = 0; i < a_f32.size(); ++i) a_q32[i] = vt::BF16ToF32(a_bf[i]);

          // CPU integer oracle: quantize each activation row once, one nrc==1
          // vec_dot per (m, n) pair.
          const size_t y_row_bytes = vt::cpu::QuantActRowBytes(enc, K);
          std::vector<uint8_t> y(M * y_row_bytes);
          for (int64_t m = 0; m < M; ++m)
            quant_act(a_q32.data() + m * K, y.data() + m * y_row_bytes, K);
          std::vector<float> oracle(M * N);
          for (int64_t m = 0; m < M; ++m)
            for (int64_t n = 0; n < N; ++n)
              vec_dot(static_cast<int>(K), &oracle[static_cast<size_t>(m) * N + n],
                      /*bs=*/0, packed.data() + static_cast<size_t>(n) * nb * kBlockBytes,
                      /*bx=*/0, y.data() + m * y_row_bytes, /*by=*/0, /*nrc=*/1);

          void* mem_a = backend.Alloc(a_bf.size() * sizeof(uint16_t));
          void* mem_b = backend.Alloc(packed.size());
          void* mem_o = backend.Alloc(std::max<size_t>(M * N, 16) * sizeof(float));
          backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
          backend.Copy(q, mem_b, packed.data(), packed.size());
          Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                          Device{vt::DeviceType::kTENSTORRENT, 0}, {M, K});
          Tensor b_t = Tensor::Contiguous(mem_b, enc,
                                          Device{vt::DeviceType::kTENSTORRENT, 0}, {N, K});
          Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                          Device{vt::DeviceType::kTENSTORRENT, 0}, {M, N});
          vt::MatmulBT(q, o_t, a_t, b_t);  // route refusal under the mutation throws here
          std::vector<float> out(std::max<size_t>(M * N, 16), 0.0f);
          backend.Copy(q, out.data(), mem_o, out.size() * sizeof(float));
          backend.Free(mem_a);
          backend.Free(mem_b);
          backend.Free(mem_o);

          INFO("default path, enc=", static_cast<int>(enc), " M=", M, " N=", N,
               " nb=", nb, " K=", K);
          if (std::memcmp(out.data(), oracle.data(), oracle.size() * sizeof(float)) != 0) {
            int64_t bad = 0;
            for (int64_t i = 0; i < static_cast<int64_t>(oracle.size()); ++i) {
              if (std::memcmp(&out[static_cast<size_t>(i)], &oracle[static_cast<size_t>(i)],
                              sizeof(float)) != 0) {
                if (bad < 4)
                  MESSAGE("diff m=", i / N, " n=", i % N, " dev=", out[static_cast<size_t>(i)],
                          " oracle=", oracle[static_cast<size_t>(i)]);
                ++bad;
              }
            }
            MESSAGE("total bad: ", bad, " / ", oracle.size(),
                    " (default-path ", vt::Name(enc), " did not take the int8-dot arm)");
          }
          CHECK(std::memcmp(out.data(), oracle.data(), oracle.size() * sizeof(float)) == 0);
        }
      }
    }
    MESSAGE("default-path ", vt::Name(enc),
            " decode: the 8-shape leg bit-exact vs the CPU vec_dot "
            "with VT_TT_KEEPQUANT_INT8DOT unset");
  }
}

// QUANT-GGUF-IQ-TENSTORRENT wave 3: the same default-path pin for Q3_K
// (enc_sel 7), the min-term K-quant the census closes with. Q3_K has no W4a
// grouped arm, so the int8-dot kernel is its only serve, dispatched
// REGARDLESS of VT_TT_KEEPQUANT_INT8DOT; the lever is explicitly UNSET here
// and the same bit-exact bar holds vs the CPU integer vec_dot oracle
// (VecDotQ3_KQ8_K). The dispatch-only mutation (override deleted, admission
// kept) refuses the route here and goes RED.
TEST_CASE("kTENSTORRENT kMatmulBTQuant Q3_K serves the int8-dot arm on the DEFAULT path (VT_TT_KEEPQUANT_INT8DOT unset)") {
  // Default-path contract: the lever env must NOT be set for this leg.
  const char* prev_lever = std::getenv("VT_TT_KEEPQUANT_INT8DOT");
  const bool had_lever = prev_lever != nullptr;
  const std::string saved_lever =
      had_lever ? std::string(prev_lever) : std::string();
  struct RestoreLeverEnv {
    const bool had;
    const std::string saved;
    ~RestoreLeverEnv() {
      if (had) ::setenv("VT_TT_KEEPQUANT_INT8DOT", saved.c_str(), 1);
      else ::unsetenv("VT_TT_KEEPQUANT_INT8DOT");
    }
  } restore_lever_env{had_lever, saved_lever};
  ::unsetenv("VT_TT_KEEPQUANT_INT8DOT");

  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuant, vt::DeviceType::kTENSTORRENT));

  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();

  const vt::DType enc = vt::DType::kQ3_K;
  const int64_t kBlockBytes = vt::BlockBytes(enc);
  const int64_t kBlockElems = vt::BlockElems(enc);
  REQUIRE(kBlockBytes == 110);
  REQUIRE(kBlockElems == 256);
  auto quant_act = vt::cpu::BlockFromFloat(vt::DType::kQ8_K);
  auto vec_dot = vt::cpu::BlockVecDot(enc);
  REQUIRE(quant_act != nullptr);
  REQUIRE(vec_dot != nullptr);

  // The sweep's Q3_K packed generator and PRNG style, fixed seed.
  std::mt19937 rng(20260909u);
  auto rand_byte = [&rng]() { return static_cast<uint8_t>(rng() & 0xFF); };
  auto rand_f16_signed = [&rng](float lo, float span) {
    return (lo + span * static_cast<float>(rng() % 64) / 64.0f) *
           ((rng() % 2) != 0 ? 1.0f : -1.0f);
  };
  auto fill_block = [&](uint8_t* blk) {
    // block_q3_K = { u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d } (110B).
    for (int i = 0; i < 32; ++i) blk[0 + i] = rand_byte();   // hmask
    for (int i = 0; i < 64; ++i) blk[32 + i] = rand_byte();  // qs
    for (int i = 0; i < 12; ++i) blk[96 + i] = rand_byte();  // scales
    const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
    std::memcpy(blk + 108, &d_bits, sizeof(d_bits));
  };

  // Decode shapes (M=1 is the GEMV), both one- and multi-block K.
  for (int64_t M : {int64_t{1}, int64_t{3}}) {
    for (int64_t N : {int64_t{1}, int64_t{17}}) {
      for (int64_t nb : {int64_t{1}, int64_t{16}}) {
        const int64_t K = nb * kBlockElems;
        std::vector<uint8_t> packed(N * nb * kBlockBytes);
        for (int64_t b = 0; b < N * nb; ++b)
          fill_block(packed.data() + b * kBlockBytes);
        std::vector<float> a_f32(M * K);
        for (auto& v : a_f32) v = (static_cast<float>(rng() % 401) - 200.0f) / 100.0f;
        std::vector<uint16_t> a_bf(M * K);
        for (size_t i = 0; i < a_f32.size(); ++i) a_bf[i] = vt::F32ToBF16(a_f32[i]);
        std::vector<float> a_q32(M * K);
        for (size_t i = 0; i < a_f32.size(); ++i) a_q32[i] = vt::BF16ToF32(a_bf[i]);

        // CPU integer oracle: quantize each activation row once, one nrc==1
        // vec_dot per (m, n) pair.
        const size_t y_row_bytes = vt::cpu::QuantActRowBytes(enc, K);
        std::vector<uint8_t> y(M * y_row_bytes);
        for (int64_t m = 0; m < M; ++m)
          quant_act(a_q32.data() + m * K, y.data() + m * y_row_bytes, K);
        std::vector<float> oracle(M * N);
        for (int64_t m = 0; m < M; ++m)
          for (int64_t n = 0; n < N; ++n)
            vec_dot(static_cast<int>(K), &oracle[static_cast<size_t>(m) * N + n],
                    /*bs=*/0, packed.data() + static_cast<size_t>(n) * nb * kBlockBytes,
                    /*bx=*/0, y.data() + m * y_row_bytes, /*by=*/0, /*nrc=*/1);

        void* mem_a = backend.Alloc(a_bf.size() * sizeof(uint16_t));
        void* mem_b = backend.Alloc(packed.size());
        void* mem_o = backend.Alloc(std::max<size_t>(M * N, 16) * sizeof(float));
        backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
        backend.Copy(q, mem_b, packed.data(), packed.size());
        Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                        Device{vt::DeviceType::kTENSTORRENT, 0}, {M, K});
        Tensor b_t = Tensor::Contiguous(mem_b, enc,
                                        Device{vt::DeviceType::kTENSTORRENT, 0}, {N, K});
        Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                        Device{vt::DeviceType::kTENSTORRENT, 0}, {M, N});
        vt::MatmulBT(q, o_t, a_t, b_t);  // route refusal under the mutation throws here
        std::vector<float> out(std::max<size_t>(M * N, 16), 0.0f);
        backend.Copy(q, out.data(), mem_o, out.size() * sizeof(float));
        backend.Free(mem_a);
        backend.Free(mem_b);
        backend.Free(mem_o);

        INFO("default path, enc=", static_cast<int>(enc), " M=", M, " N=", N,
             " nb=", nb, " K=", K);
        if (std::memcmp(out.data(), oracle.data(), oracle.size() * sizeof(float)) != 0) {
          int64_t bad = 0;
          for (int64_t i = 0; i < static_cast<int64_t>(oracle.size()); ++i) {
            if (std::memcmp(&out[static_cast<size_t>(i)], &oracle[static_cast<size_t>(i)],
                            sizeof(float)) != 0) {
              if (bad < 4)
                MESSAGE("diff m=", i / N, " n=", i % N, " dev=", out[static_cast<size_t>(i)],
                        " oracle=", oracle[static_cast<size_t>(i)]);
              ++bad;
            }
          }
          MESSAGE("total bad: ", bad, " / ", oracle.size(),
                  " (default-path Q3_K did not take the int8-dot arm)");
        }
        CHECK(std::memcmp(out.data(), oracle.data(), oracle.size() * sizeof(float)) == 0);
      }
    }
  }
  MESSAGE("default-path Q3_K decode: the 8-shape leg bit-exact vs the CPU vec_dot "
          "with VT_TT_KEEPQUANT_INT8DOT unset");
}

// ISSUE-LOCAL-01M2NSDATJQ1YNW1PA9ZBMAAM5: the 27B prefill shapes that broke the
// int8-dot arm. Both encodings the 27B checkpoint stores on this route, the two
// production tile widths (down-proj K=17408 N=5120, gate/up K=5120 N=17408) and
// the M=256 prefill row count, one shape each per encoding, M=256 fixed. The
// bar is the default-path Q3_K pin's: the device result is BIT-EXACT against
// the CPU integer vec_dot oracle per (m, n), and the outputs are finite and
// non-degenerate.
TEST_CASE("kTENSTORRENT int8-dot 27B prefill shapes M=256 (ISSUE-LOCAL-01M2NSDATJQ1YNW1PA9ZBMAAM5)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuant, vt::DeviceType::kTENSTORRENT));

  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();

  auto quant_act = vt::cpu::BlockFromFloat(vt::DType::kQ8_K);
  REQUIRE(quant_act != nullptr);

  const int64_t M = 256;
  struct Shape {
    const char* label;
    int64_t N;
    int64_t nb;
  };
  const Shape shapes[] = {
      {"down-proj K=17408", 5120, 68},
      {"gate/up K=5120", 17408, 20},
  };

  for (const vt::DType enc :
       {vt::DType::kQ3_K, vt::DType::kIQ3_XXS, vt::DType::kIQ3_S,
        vt::DType::kIQ4_XS, vt::DType::kIQ2_XS, vt::DType::kQ2_K,
        vt::DType::kIQ1_S, vt::DType::kIQ1_M}) {
    const int64_t kBlockBytes = vt::BlockBytes(enc);
    const int64_t kBlockElems = vt::BlockElems(enc);
    if (enc == vt::DType::kQ3_K) {
      REQUIRE(kBlockBytes == 110);
      REQUIRE(kBlockElems == 256);
    } else if (enc == vt::DType::kIQ3_S) {
      // tenstorrent-gsq-keepquant wave 1: the largest census gap (97
      // tensors) joins the prefill-shape bar the ISSUE-LOCAL-01M2NSDATJQ1Y
      // repair pinned for the other two.
      REQUIRE(kBlockBytes == 110);
      REQUIRE(kBlockElems == 256);
    } else if (enc == vt::DType::kIQ4_XS) {
      // tenstorrent-gsq-keepquant wave 2: the 33 census tensors (ssm_out +
      // attn) join the prefill-shape bar the ISSUE-LOCAL-01M2NSDATJQ1Y
      // repair pinned for the other three.
      REQUIRE(kBlockBytes == 136);
      REQUIRE(kBlockElems == 256);
    } else if (enc == vt::DType::kIQ2_XS) {
      // tenstorrent-gsq-keepquant wave 3: the 32 census tensors (ffn) join
      // the same prefill-shape bar.
      REQUIRE(kBlockBytes == 74);
      REQUIRE(kBlockElems == 256);
    } else if (enc == vt::DType::kQ2_K) {
      // tenstorrent-gsq-keepquant wave 4: the 28 census tensors (ffn + embd)
      // join the same prefill-shape bar.
      REQUIRE(kBlockBytes == 84);
      REQUIRE(kBlockElems == 256);
    } else if (enc == vt::DType::kIQ1_S || enc == vt::DType::kIQ1_M) {
      // tenstorrent-gsq-keepquant wave 5: the IQ1 pair joins the same
      // prefill-shape bar.
      REQUIRE(kBlockBytes == (enc == vt::DType::kIQ1_S ? 50 : 56));
      REQUIRE(kBlockElems == 256);
    } else {
      REQUIRE(kBlockBytes == 98);
      REQUIRE(kBlockElems == 256);
    }
    auto vec_dot = vt::cpu::BlockVecDot(enc);
    REQUIRE(vec_dot != nullptr);

    // The default-path Q3_K test's packed generator and PRNG style, fixed seed.
    std::mt19937 rng(20260921u);
    auto rand_byte = [&rng]() { return static_cast<uint8_t>(rng() & 0xFF); };
    auto rand_f16_signed = [&rng](float lo, float span) {
      return (lo + span * static_cast<float>(rng() % 64) / 64.0f) *
             ((rng() % 2) != 0 ? 1.0f : -1.0f);
    };
    auto fill_block = [&](uint8_t* blk) {
      if (enc == vt::DType::kQ3_K) {
        // block_q3_K = { u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d } (110B).
        for (int i = 0; i < 32; ++i) blk[0 + i] = rand_byte();   // hmask
        for (int i = 0; i < 64; ++i) blk[32 + i] = rand_byte();  // qs
        for (int i = 0; i < 12; ++i) blk[96 + i] = rand_byte();  // scales
        const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
        std::memcpy(blk + 108, &d_bits, sizeof(d_bits));
      } else if (enc == vt::DType::kIQ3_S) {
        // block_iq3_s = { f16 d; u8 qs[64]; u8 qh[8]; u8 signs[32];
        // u8 scales[4] } (110 bytes).
        const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
        std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
        for (int i = 0; i < 64; ++i) blk[2 + i] = rand_byte();    // qs
        for (int i = 0; i < 8; ++i) blk[66 + i] = rand_byte();    // qh
        for (int i = 0; i < 32; ++i) blk[74 + i] = rand_byte();   // signs
        for (int i = 0; i < 4; ++i) blk[106 + i] = rand_byte();   // scales
      } else if (enc == vt::DType::kIQ4_XS) {
        // block_iq4_xs = { f16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128] }
        // (136 bytes).
        const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
        std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
        for (int i = 0; i < 134; ++i) blk[2 + i] = rand_byte();
      } else if (enc == vt::DType::kIQ2_XS) {
        // block_iq2_xs = { f16 d; u16 qs[32]; u8 scales[8] } (74 bytes).
        const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
        std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
        for (int i = 0; i < 72; ++i) blk[2 + i] = rand_byte();
      } else if (enc == vt::DType::kQ2_K) {
        // block_q2_K = { u8 scales[16]; u8 qs[64]; f16 d; f16 dmin }
        // (84 bytes): scales lead, deltas trail.
        const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
        const uint16_t dmin_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
        for (int i = 0; i < 80; ++i) blk[i] = rand_byte();  // scales + qs
        std::memcpy(blk + 80, &d_bits, sizeof(d_bits));
        std::memcpy(blk + 82, &dmin_bits, sizeof(dmin_bits));
      } else if (enc == vt::DType::kIQ1_S) {
        // block_iq1_s = { f16 d; u8 qs[32]; u16 qh[8] } (50 bytes).
        const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
        std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
        for (int i = 0; i < 48; ++i) blk[2 + i] = rand_byte();  // qs + qh
      } else if (enc == vt::DType::kIQ1_M) {
        // block_iq1_m = { u8 qs[32]; u8 qh[16]; u8 scales[8] } (56 bytes):
        // the spliced f16 scale needs the same finite-exponent constraint
        // as the sweep leg (random top nibbles make random exponents).
        for (int i = 0; i < 56; ++i) blk[i] = rand_byte();
        blk[55] = 0x30;     // sb[7]: packed bits 15-12 = 0b0011 (sign 0,
        blk[53] &= 0x03;    // sb[5]: exponent bit 10 = 0 (exp 12, d in [8,16))
      } else {
        // block_iq3_xxs = { f16 d; u8 qs[96] } (98 bytes).
        const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
        std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
        for (int i = 0; i < 96; ++i) blk[2 + i] = rand_byte();  // qs
      }
    };

    for (const Shape& sh : shapes) {
      const int64_t N = sh.N;
      const int64_t K = sh.nb * kBlockElems;
      std::vector<uint8_t> packed(N * sh.nb * kBlockBytes);
      for (int64_t b = 0; b < N * sh.nb; ++b)
        fill_block(packed.data() + b * kBlockBytes);
      std::vector<float> a_f32(M * K);
      for (auto& v : a_f32) v = (static_cast<float>(rng() % 401) - 200.0f) / 100.0f;
      std::vector<uint16_t> a_bf(M * K);
      for (size_t i = 0; i < a_f32.size(); ++i) a_bf[i] = vt::F32ToBF16(a_f32[i]);
      std::vector<float> a_q32(M * K);
      for (size_t i = 0; i < a_f32.size(); ++i) a_q32[i] = vt::BF16ToF32(a_bf[i]);

      // CPU integer oracle: quantize each activation row once, one nrc==1
      // vec_dot per (m, n) pair.
      const size_t y_row_bytes = vt::cpu::QuantActRowBytes(enc, K);
      std::vector<uint8_t> y(M * y_row_bytes);
      for (int64_t m = 0; m < M; ++m)
        quant_act(a_q32.data() + m * K, y.data() + m * y_row_bytes, K);
      std::vector<float> oracle(M * N);
      for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n)
          vec_dot(static_cast<int>(K), &oracle[static_cast<size_t>(m) * N + n],
                  /*bs=*/0, packed.data() + static_cast<size_t>(n) * sh.nb * kBlockBytes,
                  /*bx=*/0, y.data() + m * y_row_bytes, /*by=*/0, /*nrc=*/1);

      void* mem_a = backend.Alloc(a_bf.size() * sizeof(uint16_t));
      void* mem_b = backend.Alloc(packed.size());
      void* mem_o = backend.Alloc(std::max<size_t>(M * N, 16) * sizeof(float));
      backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
      backend.Copy(q, mem_b, packed.data(), packed.size());
      Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                      Device{vt::DeviceType::kTENSTORRENT, 0}, {M, K});
      Tensor b_t = Tensor::Contiguous(mem_b, enc,
                                      Device{vt::DeviceType::kTENSTORRENT, 0}, {N, K});
      Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                      Device{vt::DeviceType::kTENSTORRENT, 0}, {M, N});
      vt::MatmulBT(q, o_t, a_t, b_t);
      std::vector<float> out(std::max<size_t>(M * N, 16), 0.0f);
      backend.Copy(q, out.data(), mem_o, out.size() * sizeof(float));
      backend.Free(mem_a);
      backend.Free(mem_b);
      backend.Free(mem_o);

      const char* enc_name = (enc == vt::DType::kQ3_K)    ? "Q3_K"
                             : (enc == vt::DType::kIQ3_S)  ? "IQ3_S"
                             : (enc == vt::DType::kIQ4_XS) ? "IQ4_XS"
                             : (enc == vt::DType::kIQ2_XS) ? "IQ2_XS"
                             : (enc == vt::DType::kQ2_K)   ? "Q2_K"
                             : (enc == vt::DType::kIQ1_S)  ? "IQ1_S"
                             : (enc == vt::DType::kIQ1_M)  ? "IQ1_M"
                                                           : "IQ3_XXS";
      INFO(enc_name, " ", sh.label,
           " N=", N, " nb=", sh.nb, " K=", K, " M=", M);
      int64_t bad = 0;
      int64_t zero = 0;
      int64_t infs = 0;
      for (int64_t i = 0; i < static_cast<int64_t>(M * N); ++i) {
        const float dev = out[static_cast<size_t>(i)];
        const float ref = oracle[static_cast<size_t>(i)];
        if (std::memcmp(&dev, &ref, sizeof(float)) != 0) {
          if (bad < 4)
            MESSAGE("diff m=", i / N, " n=", i % N, " dev=", dev, " oracle=", ref);
          ++bad;
        }
        if (dev == 0.0f) ++zero;
        if (std::isinf(dev)) ++infs;
      }
      MESSAGE("enc=", enc_name, " ", sh.label, ": bad=", bad,
              " zero=", zero, " infs=", infs, " / ", M * N);
      CHECK(bad == 0);
      CHECK(zero < (M * N) / 2);
      CHECK(std::memcmp(out.data(), oracle.data(), M * N * sizeof(float)) == 0);
    }
  }
  MESSAGE("27B prefill shapes M=256: Q3_K, IQ3_XXS, IQ3_S, IQ4_XS, IQ2_XS, "
          "Q2_K, IQ1_S and IQ1_M bit-exact vs the CPU vec_dot");
}

// tenstorrent-gsq-keepquant wave 1: the IQ3_S default-path dispatch pin and
// the APEX decode-shape leg (M=1 GEMV at the two 27B production tile
// widths). The Q3_K wave-3 test is the template; the lever is explicitly
// UNSET here and the bit-exact bar holds vs the CPU integer vec_dot oracle
// (VecDotIQ3_SQ8_K). The dispatch-only mutation (override deleted, admission
// kept) refuses the route here and goes RED.
TEST_CASE("kTENSTORRENT kMatmulBTQuant IQ3_S serves the int8-dot arm on the DEFAULT path (VT_TT_KEEPQUANT_INT8DOT unset)") {
  // Default-path contract: the lever env must NOT be set for this leg.
  const char* prev_lever = std::getenv("VT_TT_KEEPQUANT_INT8DOT");
  const bool had_lever = prev_lever != nullptr;
  const std::string saved_lever =
      had_lever ? std::string(prev_lever) : std::string();
  struct RestoreLeverEnv {
    const bool had;
    const std::string saved;
    ~RestoreLeverEnv() {
      if (had) ::setenv("VT_TT_KEEPQUANT_INT8DOT", saved.c_str(), 1);
      else ::unsetenv("VT_TT_KEEPQUANT_INT8DOT");
    }
  } restore_lever_env{had_lever, saved_lever};
  ::unsetenv("VT_TT_KEEPQUANT_INT8DOT");

  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuant, vt::DeviceType::kTENSTORRENT));

  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();

  const vt::DType enc = vt::DType::kIQ3_S;
  const int64_t kBlockBytes = vt::BlockBytes(enc);
  const int64_t kBlockElems = vt::BlockElems(enc);
  REQUIRE(kBlockBytes == 110);
  REQUIRE(kBlockElems == 256);
  auto quant_act = vt::cpu::BlockFromFloat(vt::DType::kQ8_K);
  auto vec_dot = vt::cpu::BlockVecDot(enc);
  REQUIRE(quant_act != nullptr);
  REQUIRE(vec_dot != nullptr);

  // The sweep's PRNG style and packed generator, fixed seed.
  std::mt19937 rng(20260922u);
  auto rand_byte = [&rng]() { return static_cast<uint8_t>(rng() & 0xFF); };
  auto rand_f16_signed = [&rng](float lo, float span) {
    return (lo + span * static_cast<float>(rng() % 64) / 64.0f) *
           ((rng() % 2) != 0 ? 1.0f : -1.0f);
  };
  auto fill_block = [&](uint8_t* blk) {
    const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
    std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
    for (int i = 0; i < 64; ++i) blk[2 + i] = rand_byte();    // qs
    for (int i = 0; i < 8; ++i) blk[66 + i] = rand_byte();    // qh
    for (int i = 0; i < 32; ++i) blk[74 + i] = rand_byte();   // signs
    for (int i = 0; i < 4; ++i) blk[106 + i] = rand_byte();   // scales
  };

  // Decode shapes: small blocks (the f32-exact P==1 floor) plus the two APEX
  // decode shapes M=1 at the production tile widths (down-proj K=17408
  // N=5120, gate/up K=5120 N=17408). M=256 at the same widths is the
  // prefill-shapes leg above.
  struct Shape {
    const char* label;
    int64_t M;
    int64_t N;
    int64_t nb;
  };
  const Shape shapes[] = {
      {"small M=1", 1, 17, 16},
      {"small M=3", 3, 17, 16},
      {"down-proj decode", 1, 5120, 68},
      {"gate/up decode", 1, 17408, 20},
  };
  for (const Shape& sh : shapes) {
    const int64_t M = sh.M;
    const int64_t N = sh.N;
    const int64_t K = sh.nb * kBlockElems;
    std::vector<uint8_t> packed(N * sh.nb * kBlockBytes);
    for (int64_t b = 0; b < N * sh.nb; ++b)
      fill_block(packed.data() + b * kBlockBytes);
    std::vector<float> a_f32(M * K);
    for (auto& v : a_f32) v = (static_cast<float>(rng() % 401) - 200.0f) / 100.0f;
    std::vector<uint16_t> a_bf(M * K);
    for (size_t i = 0; i < a_f32.size(); ++i) a_bf[i] = vt::F32ToBF16(a_f32[i]);
    std::vector<float> a_q32(M * K);
    for (size_t i = 0; i < a_f32.size(); ++i) a_q32[i] = vt::BF16ToF32(a_bf[i]);

    // CPU integer oracle: quantize each activation row once, one nrc==1
    // vec_dot per (m, n) pair.
    const size_t y_row_bytes = vt::cpu::QuantActRowBytes(enc, K);
    std::vector<uint8_t> y(M * y_row_bytes);
    for (int64_t m = 0; m < M; ++m)
      quant_act(a_q32.data() + m * K, y.data() + m * y_row_bytes, K);
    std::vector<float> oracle(M * N);
    for (int64_t m = 0; m < M; ++m)
      for (int64_t n = 0; n < N; ++n)
        vec_dot(static_cast<int>(K), &oracle[static_cast<size_t>(m) * N + n],
                /*bs=*/0, packed.data() + static_cast<size_t>(n) * sh.nb * kBlockBytes,
                /*bx=*/0, y.data() + m * y_row_bytes, /*by=*/0, /*nrc=*/1);

    void* mem_a = backend.Alloc(a_bf.size() * sizeof(uint16_t));
    void* mem_b = backend.Alloc(packed.size());
    void* mem_o = backend.Alloc(std::max<size_t>(M * N, 16) * sizeof(float));
    backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
    backend.Copy(q, mem_b, packed.data(), packed.size());
    Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {M, K});
    Tensor b_t = Tensor::Contiguous(mem_b, enc,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {N, K});
    Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {M, N});
    vt::MatmulBT(q, o_t, a_t, b_t);  // route refusal under the mutation throws here
    std::vector<float> out(std::max<size_t>(M * N, 16), 0.0f);
    backend.Copy(q, out.data(), mem_o, out.size() * sizeof(float));
    backend.Free(mem_a);
    backend.Free(mem_b);
    backend.Free(mem_o);

    INFO("default path IQ3_S ", sh.label, " N=", N, " nb=", sh.nb, " K=", K,
         " M=", M);
    int64_t bad = 0;
    int64_t zero = 0;
    int64_t infs = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(M * N); ++i) {
      const float dev = out[static_cast<size_t>(i)];
      const float ref = oracle[static_cast<size_t>(i)];
      if (std::memcmp(&dev, &ref, sizeof(float)) != 0) {
        if (bad < 4)
          MESSAGE("diff m=", i / N, " n=", i % N, " dev=", dev,
                  " oracle=", ref);
        ++bad;
      }
      if (dev == 0.0f) ++zero;
      if (std::isinf(dev)) ++infs;
    }
    MESSAGE("IQ3_S ", sh.label, ": bad=", bad, " zero=", zero,
            " infs=", infs, " / ", M * N);
    CHECK(bad == 0);
    CHECK(zero < (M * N) / 2);
    CHECK(std::memcmp(out.data(), oracle.data(), M * N * sizeof(float)) == 0);
  }
  MESSAGE("default-path IQ3_S decode: bit-exact vs the CPU vec_dot with "
          "VT_TT_KEEPQUANT_INT8DOT unset");
}

// tenstorrent-gsq-keepquant wave 2: the IQ4_XS default-path dispatch pin and
// the APEX decode-shape leg (M=1 GEMV at the two 27B production tile
// widths), the wave-1 IQ3_S test as template. The lever is explicitly UNSET
// here and the bit-exact bar holds vs the CPU integer vec_dot oracle
// (VecDotIQ4_XSQ8_K, which already existed — the IQ2_XS/IQ4_XS row landed it
// with the CPU dot tests). The dispatch-only mutation (override deleted,
// admission kept) refuses the route here and goes RED.
TEST_CASE("kTENSTORRENT kMatmulBTQuant IQ4_XS serves the int8-dot arm on the DEFAULT path (VT_TT_KEEPQUANT_INT8DOT unset)") {
  // Default-path contract: the lever env must NOT be set for this leg.
  const char* prev_lever = std::getenv("VT_TT_KEEPQUANT_INT8DOT");
  const bool had_lever = prev_lever != nullptr;
  const std::string saved_lever =
      had_lever ? std::string(prev_lever) : std::string();
  struct RestoreLeverEnv {
    const bool had;
    const std::string saved;
    ~RestoreLeverEnv() {
      if (had) ::setenv("VT_TT_KEEPQUANT_INT8DOT", saved.c_str(), 1);
      else ::unsetenv("VT_TT_KEEPQUANT_INT8DOT");
    }
  } restore_lever_env{had_lever, saved_lever};
  ::unsetenv("VT_TT_KEEPQUANT_INT8DOT");

  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuant, vt::DeviceType::kTENSTORRENT));

  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();

  const vt::DType enc = vt::DType::kIQ4_XS;
  const int64_t kBlockBytes = vt::BlockBytes(enc);
  const int64_t kBlockElems = vt::BlockElems(enc);
  REQUIRE(kBlockBytes == 136);
  REQUIRE(kBlockElems == 256);
  auto quant_act = vt::cpu::BlockFromFloat(vt::DType::kQ8_K);
  auto vec_dot = vt::cpu::BlockVecDot(enc);
  REQUIRE(quant_act != nullptr);
  REQUIRE(vec_dot != nullptr);

  // The sweep's PRNG style and packed generator, fixed seed.
  std::mt19937 rng(20260923u);
  auto rand_byte = [&rng]() { return static_cast<uint8_t>(rng() & 0xFF); };
  auto rand_f16_signed = [&rng](float lo, float span) {
    return (lo + span * static_cast<float>(rng() % 64) / 64.0f) *
           ((rng() % 2) != 0 ? 1.0f : -1.0f);
  };
  auto fill_block = [&](uint8_t* blk) {
    const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
    std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
    for (int i = 0; i < 134; ++i) blk[2 + i] = rand_byte();  // scales_h,
                                                             // scales_l, qs
  };

  // Decode shapes: small blocks (the f32-exact P==1 floor) plus the two APEX
  // decode shapes M=1 at the production tile widths (down-proj K=17408
  // N=5120, gate/up K=5120 N=17408). M=256 at the same widths is the
  // prefill-shapes leg above.
  struct Shape {
    const char* label;
    int64_t M;
    int64_t N;
    int64_t nb;
  };
  const Shape shapes[] = {
      {"small M=1", 1, 17, 16},
      {"small M=3", 3, 17, 16},
      {"down-proj decode", 1, 5120, 68},
      {"gate/up decode", 1, 17408, 20},
  };
  for (const Shape& sh : shapes) {
    const int64_t M = sh.M;
    const int64_t N = sh.N;
    const int64_t K = sh.nb * kBlockElems;
    std::vector<uint8_t> packed(N * sh.nb * kBlockBytes);
    for (int64_t b = 0; b < N * sh.nb; ++b)
      fill_block(packed.data() + b * kBlockBytes);
    std::vector<float> a_f32(M * K);
    for (auto& v : a_f32) v = (static_cast<float>(rng() % 401) - 200.0f) / 100.0f;
    std::vector<uint16_t> a_bf(M * K);
    for (size_t i = 0; i < a_f32.size(); ++i) a_bf[i] = vt::F32ToBF16(a_f32[i]);
    std::vector<float> a_q32(M * K);
    for (size_t i = 0; i < a_f32.size(); ++i) a_q32[i] = vt::BF16ToF32(a_bf[i]);

    // CPU integer oracle: quantize each activation row once, one nrc==1
    // vec_dot per (m, n) pair.
    const size_t y_row_bytes = vt::cpu::QuantActRowBytes(enc, K);
    std::vector<uint8_t> y(M * y_row_bytes);
    for (int64_t m = 0; m < M; ++m)
      quant_act(a_q32.data() + m * K, y.data() + m * y_row_bytes, K);
    std::vector<float> oracle(M * N);
    for (int64_t m = 0; m < M; ++m)
      for (int64_t n = 0; n < N; ++n)
        vec_dot(static_cast<int>(K), &oracle[static_cast<size_t>(m) * N + n],
                /*bs=*/0, packed.data() + static_cast<size_t>(n) * sh.nb * kBlockBytes,
                /*bx=*/0, y.data() + m * y_row_bytes, /*by=*/0, /*nrc=*/1);

    void* mem_a = backend.Alloc(a_bf.size() * sizeof(uint16_t));
    void* mem_b = backend.Alloc(packed.size());
    void* mem_o = backend.Alloc(std::max<size_t>(M * N, 16) * sizeof(float));
    backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
    backend.Copy(q, mem_b, packed.data(), packed.size());
    Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {M, K});
    Tensor b_t = Tensor::Contiguous(mem_b, enc,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {N, K});
    Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {M, N});
    vt::MatmulBT(q, o_t, a_t, b_t);  // route refusal under the mutation throws here
    std::vector<float> out(std::max<size_t>(M * N, 16), 0.0f);
    backend.Copy(q, out.data(), mem_o, out.size() * sizeof(float));
    backend.Free(mem_a);
    backend.Free(mem_b);
    backend.Free(mem_o);

    INFO("default path IQ4_XS ", sh.label, " N=", N, " nb=", sh.nb, " K=", K,
         " M=", M);
    {
    }
    int64_t bad = 0;
    int64_t zero = 0;
    int64_t infs = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(M * N); ++i) {
      const float dev = out[static_cast<size_t>(i)];
      const float ref = oracle[static_cast<size_t>(i)];
      if (std::memcmp(&dev, &ref, sizeof(float)) != 0) {
        if (bad < 4)
          MESSAGE("diff m=", i / N, " n=", i % N, " dev=", dev,
                  " oracle=", ref);
        ++bad;
      }
      if (dev == 0.0f) ++zero;
      if (std::isinf(dev)) ++infs;
    }
    MESSAGE("IQ4_XS ", sh.label, ": bad=", bad, " zero=", zero,
            " infs=", infs, " / ", M * N);
    CHECK(bad == 0);
    CHECK(zero < (M * N) / 2);
    CHECK(std::memcmp(out.data(), oracle.data(), M * N * sizeof(float)) == 0);
  }
  MESSAGE("default-path IQ4_XS decode: bit-exact vs the CPU vec_dot with "
          "VT_TT_KEEPQUANT_INT8DOT unset");
}

// tenstorrent-gsq-keepquant wave 3: the IQ2_XS default-path dispatch pin and
// the APEX decode-shape leg (M=1 GEMV at the two 27B production tile
// widths), the wave-2 IQ4_XS test as template. The lever is explicitly UNSET
// here and the bit-exact bar holds vs the CPU integer vec_dot oracle
// (VecDotIQ2_XSQ8_K, which already existed — the IQ2_XS/IQ4_XS row landed it
// with the CPU dot tests). The dispatch-only mutation (override deleted,
// admission kept) refuses the route here and goes RED.
TEST_CASE("kTENSTORRENT kMatmulBTQuant IQ2_XS serves the int8-dot arm on the DEFAULT path (VT_TT_KEEPQUANT_INT8DOT unset)") {
  // Default-path contract: the lever env must NOT be set for this leg.
  const char* prev_lever = std::getenv("VT_TT_KEEPQUANT_INT8DOT");
  const bool had_lever = prev_lever != nullptr;
  const std::string saved_lever =
      had_lever ? std::string(prev_lever) : std::string();
  struct RestoreLeverEnv {
    const bool had;
    const std::string saved;
    ~RestoreLeverEnv() {
      if (had) ::setenv("VT_TT_KEEPQUANT_INT8DOT", saved.c_str(), 1);
      else ::unsetenv("VT_TT_KEEPQUANT_INT8DOT");
    }
  } restore_lever_env{had_lever, saved_lever};
  ::unsetenv("VT_TT_KEEPQUANT_INT8DOT");

  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuant, vt::DeviceType::kTENSTORRENT));

  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();

  const vt::DType enc = vt::DType::kIQ2_XS;
  const int64_t kBlockBytes = vt::BlockBytes(enc);
  const int64_t kBlockElems = vt::BlockElems(enc);
  REQUIRE(kBlockBytes == 74);
  REQUIRE(kBlockElems == 256);
  auto quant_act = vt::cpu::BlockFromFloat(vt::DType::kQ8_K);
  auto vec_dot = vt::cpu::BlockVecDot(enc);
  REQUIRE(quant_act != nullptr);
  REQUIRE(vec_dot != nullptr);

  // The sweep's PRNG style and packed generator, fixed seed.
  std::mt19937 rng(20260924u);
  auto rand_byte = [&rng]() { return static_cast<uint8_t>(rng() & 0xFF); };
  auto rand_f16_signed = [&rng](float lo, float span) {
    return (lo + span * static_cast<float>(rng() % 64) / 64.0f) *
           ((rng() % 2) != 0 ? 1.0f : -1.0f);
  };
  auto fill_block = [&](uint8_t* blk) {
    const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
    std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
    for (int i = 0; i < 72; ++i) blk[2 + i] = rand_byte();  // qs + scales
  };

  // Decode shapes: small blocks (the f32-exact P==1 floor) plus the two APEX
  // decode shapes M=1 at the production tile widths (down-proj K=17408
  // N=5120, gate/up K=5120 N=17408). M=256 at the same widths is the
  // prefill-shapes leg above.
  struct Shape {
    const char* label;
    int64_t M;
    int64_t N;
    int64_t nb;
  };
  const Shape shapes[] = {
      {"small M=1", 1, 17, 16},
      {"small M=3", 3, 17, 16},
      {"down-proj decode", 1, 5120, 68},
      {"gate/up decode", 1, 17408, 20},
  };
  for (const Shape& sh : shapes) {
    const int64_t M = sh.M;
    const int64_t N = sh.N;
    const int64_t K = sh.nb * kBlockElems;
    std::vector<uint8_t> packed(N * sh.nb * kBlockBytes);
    for (int64_t b = 0; b < N * sh.nb; ++b)
      fill_block(packed.data() + b * kBlockBytes);
    std::vector<float> a_f32(M * K);
    for (auto& v : a_f32) v = (static_cast<float>(rng() % 401) - 200.0f) / 100.0f;
    std::vector<uint16_t> a_bf(M * K);
    for (size_t i = 0; i < a_f32.size(); ++i) a_bf[i] = vt::F32ToBF16(a_f32[i]);
    std::vector<float> a_q32(M * K);
    for (size_t i = 0; i < a_f32.size(); ++i) a_q32[i] = vt::BF16ToF32(a_bf[i]);

    // CPU integer oracle: quantize each activation row once, one nrc==1
    // vec_dot per (m, n) pair.
    const size_t y_row_bytes = vt::cpu::QuantActRowBytes(enc, K);
    std::vector<uint8_t> y(M * y_row_bytes);
    for (int64_t m = 0; m < M; ++m)
      quant_act(a_q32.data() + m * K, y.data() + m * y_row_bytes, K);
    std::vector<float> oracle(M * N);
    for (int64_t m = 0; m < M; ++m)
      for (int64_t n = 0; n < N; ++n)
        vec_dot(static_cast<int>(K), &oracle[static_cast<size_t>(m) * N + n],
                /*bs=*/0, packed.data() + static_cast<size_t>(n) * sh.nb * kBlockBytes,
                /*bx=*/0, y.data() + m * y_row_bytes, /*by=*/0, /*nrc=*/1);

    void* mem_a = backend.Alloc(a_bf.size() * sizeof(uint16_t));
    void* mem_b = backend.Alloc(packed.size());
    void* mem_o = backend.Alloc(std::max<size_t>(M * N, 16) * sizeof(float));
    backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
    backend.Copy(q, mem_b, packed.data(), packed.size());
    Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {M, K});
    Tensor b_t = Tensor::Contiguous(mem_b, enc,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {N, K});
    Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {M, N});
    vt::MatmulBT(q, o_t, a_t, b_t);  // route refusal under the mutation throws here
    std::vector<float> out(std::max<size_t>(M * N, 16), 0.0f);
    backend.Copy(q, out.data(), mem_o, out.size() * sizeof(float));
    backend.Free(mem_a);
    backend.Free(mem_b);
    backend.Free(mem_o);

    INFO("default path IQ2_XS ", sh.label, " N=", N, " nb=", sh.nb, " K=", K,
         " M=", M);
    int64_t bad = 0;
    int64_t zero = 0;
    int64_t infs = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(M * N); ++i) {
      const float dev = out[static_cast<size_t>(i)];
      const float ref = oracle[static_cast<size_t>(i)];
      if (std::memcmp(&dev, &ref, sizeof(float)) != 0) {
        if (bad < 4)
          MESSAGE("diff m=", i / N, " n=", i % N, " dev=", dev,
                  " oracle=", ref);
        ++bad;
      }
      if (dev == 0.0f) ++zero;
      if (std::isinf(dev)) ++infs;
    }
    MESSAGE("IQ2_XS ", sh.label, ": bad=", bad, " zero=", zero,
            " infs=", infs, " / ", M * N);
    CHECK(bad == 0);
    CHECK(zero < (M * N) / 2);
    CHECK(std::memcmp(out.data(), oracle.data(), M * N * sizeof(float)) == 0);
  }
  MESSAGE("default-path IQ2_XS decode: bit-exact vs the CPU vec_dot with "
          "VT_TT_KEEPQUANT_INT8DOT unset");
}

// tenstorrent-gsq-keepquant wave 4: the Q2_K default-path dispatch pin and
// the APEX decode-shape leg (M=1 GEMV at the two 27B production tile
// widths), the wave-3 IQ2_XS test as template. The lever is explicitly UNSET
// here and the bit-exact bar holds vs the CPU integer vec_dot oracle
// (VecDotQ2_KQ8_K, the pre-existing CPU arm — Q2_K is one of the original
// k-quants). The dispatch-only mutation (override deleted, admission kept)
// refuses the route here and goes RED.
TEST_CASE("kTENSTORRENT kMatmulBTQuant Q2_K serves the int8-dot arm on the DEFAULT path (VT_TT_KEEPQUANT_INT8DOT unset)") {
  // Default-path contract: the lever env must NOT be set for this leg.
  const char* prev_lever = std::getenv("VT_TT_KEEPQUANT_INT8DOT");
  const bool had_lever = prev_lever != nullptr;
  const std::string saved_lever =
      had_lever ? std::string(prev_lever) : std::string();
  struct RestoreLeverEnv {
    const bool had;
    const std::string saved;
    ~RestoreLeverEnv() {
      if (had) ::setenv("VT_TT_KEEPQUANT_INT8DOT", saved.c_str(), 1);
      else ::unsetenv("VT_TT_KEEPQUANT_INT8DOT");
    }
  } restore_lever_env{had_lever, saved_lever};
  ::unsetenv("VT_TT_KEEPQUANT_INT8DOT");

  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuant, vt::DeviceType::kTENSTORRENT));

  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();

  const vt::DType enc = vt::DType::kQ2_K;
  const int64_t kBlockBytes = vt::BlockBytes(enc);
  const int64_t kBlockElems = vt::BlockElems(enc);
  REQUIRE(kBlockBytes == 84);
  REQUIRE(kBlockElems == 256);
  auto quant_act = vt::cpu::BlockFromFloat(vt::DType::kQ8_K);
  auto vec_dot = vt::cpu::BlockVecDot(enc);
  REQUIRE(quant_act != nullptr);
  REQUIRE(vec_dot != nullptr);

  // The sweep's PRNG style and packed generator, fixed seed.
  std::mt19937 rng(20260925u);
  auto rand_byte = [&rng]() { return static_cast<uint8_t>(rng() & 0xFF); };
  auto rand_f16_signed = [&rng](float lo, float span) {
    return (lo + span * static_cast<float>(rng() % 64) / 64.0f) *
           ((rng() % 2) != 0 ? 1.0f : -1.0f);
  };
  auto fill_block = [&](uint8_t* blk) {
    // block_q2_K = { u8 scales[16]; u8 qs[64]; f16 d; f16 dmin } (84 bytes):
    // scales lead, deltas trail; each scale byte is a 4-bit sub-scale (low)
    // + 4-bit sub-min (high) pair. Random bytes stay in the dot's bit-exact
    // class (the per-block accumulation is all-integer).
    const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
    const uint16_t dmin_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
    for (int i = 0; i < 80; ++i) blk[i] = rand_byte();  // scales + qs
    std::memcpy(blk + 80, &d_bits, sizeof(d_bits));
    std::memcpy(blk + 82, &dmin_bits, sizeof(dmin_bits));
  };

  // Decode shapes: small blocks (the f32-exact P==1 floor) plus the two APEX
  // decode shapes M=1 at the production tile widths (down-proj K=17408
  // N=5120, gate/up K=5120 N=17408). M=256 at the same widths is the
  // prefill-shapes leg above.
  struct Shape {
    const char* label;
    int64_t M;
    int64_t N;
    int64_t nb;
  };
  const Shape shapes[] = {
      {"small M=1", 1, 17, 16},
      {"small M=3", 3, 17, 16},
      {"down-proj decode", 1, 5120, 68},
      {"gate/up decode", 1, 17408, 20},
  };
  for (const Shape& sh : shapes) {
    const int64_t M = sh.M;
    const int64_t N = sh.N;
    const int64_t K = sh.nb * kBlockElems;
    std::vector<uint8_t> packed(N * sh.nb * kBlockBytes);
    for (int64_t b = 0; b < N * sh.nb; ++b)
      fill_block(packed.data() + b * kBlockBytes);
    std::vector<float> a_f32(M * K);
    for (auto& v : a_f32) v = (static_cast<float>(rng() % 401) - 200.0f) / 100.0f;
    std::vector<uint16_t> a_bf(M * K);
    for (size_t i = 0; i < a_f32.size(); ++i) a_bf[i] = vt::F32ToBF16(a_f32[i]);
    std::vector<float> a_q32(M * K);
    for (size_t i = 0; i < a_f32.size(); ++i) a_q32[i] = vt::BF16ToF32(a_bf[i]);

    // CPU integer oracle: quantize each activation row once, one nrc==1
    // vec_dot per (m, n) pair.
    const size_t y_row_bytes = vt::cpu::QuantActRowBytes(enc, K);
    std::vector<uint8_t> y(M * y_row_bytes);
    for (int64_t m = 0; m < M; ++m)
      quant_act(a_q32.data() + m * K, y.data() + m * y_row_bytes, K);
    std::vector<float> oracle(M * N);
    for (int64_t m = 0; m < M; ++m)
      for (int64_t n = 0; n < N; ++n)
        vec_dot(static_cast<int>(K), &oracle[static_cast<size_t>(m) * N + n],
                /*bs=*/0, packed.data() + static_cast<size_t>(n) * sh.nb * kBlockBytes,
                /*bx=*/0, y.data() + m * y_row_bytes, /*by=*/0, /*nrc=*/1);

    void* mem_a = backend.Alloc(a_bf.size() * sizeof(uint16_t));
    void* mem_b = backend.Alloc(packed.size());
    void* mem_o = backend.Alloc(std::max<size_t>(M * N, 16) * sizeof(float));
    backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
    backend.Copy(q, mem_b, packed.data(), packed.size());
    Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {M, K});
    Tensor b_t = Tensor::Contiguous(mem_b, enc,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {N, K});
    Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {M, N});
    vt::MatmulBT(q, o_t, a_t, b_t);  // route refusal under the mutation throws here
    std::vector<float> out(std::max<size_t>(M * N, 16), 0.0f);
    backend.Copy(q, out.data(), mem_o, out.size() * sizeof(float));
    backend.Free(mem_a);
    backend.Free(mem_b);
    backend.Free(mem_o);

    INFO("default path Q2_K ", sh.label, " N=", N, " nb=", sh.nb, " K=", K,
         " M=", M);
    int64_t bad = 0;
    int64_t zero = 0;
    int64_t infs = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(M * N); ++i) {
      const float dev = out[static_cast<size_t>(i)];
      const float ref = oracle[static_cast<size_t>(i)];
      if (std::memcmp(&dev, &ref, sizeof(float)) != 0) {
        if (bad < 4)
          MESSAGE("diff m=", i / N, " n=", i % N, " dev=", dev,
                  " oracle=", ref);
        ++bad;
      }
      if (dev == 0.0f) ++zero;
      if (std::isinf(dev)) ++infs;
    }
    MESSAGE("Q2_K ", sh.label, ": bad=", bad, " zero=", zero,
            " infs=", infs, " / ", M * N);
    CHECK(bad == 0);
    CHECK(zero < (M * N) / 2);
    CHECK(std::memcmp(out.data(), oracle.data(), M * N * sizeof(float)) == 0);
  }
  MESSAGE("default-path Q2_K decode: bit-exact vs the CPU vec_dot with "
          "VT_TT_KEEPQUANT_INT8DOT unset");
}


// tenstorrent-gsq-keepquant wave 5: the IQ1_S default-path dispatch pin and
// the APEX decode-shape leg (M=1 GEMV at the two 27B production tile
// widths), the wave-4 Q2_K test as template. The lever is explicitly UNSET
// here and the bit-exact bar holds vs the CPU integer vec_dot oracle
// (VecDotIQ1_SQ8_K, the #3228 CPU arm). The dispatch-only mutation
// (override deleted, admission kept) refuses the route here and goes RED.
TEST_CASE("kTENSTORRENT kMatmulBTQuant IQ1_S serves the int8-dot arm on the DEFAULT path (VT_TT_KEEPQUANT_INT8DOT unset)") {
  // Default-path contract: the lever env must NOT be set for this leg.
  const char* prev_lever = std::getenv("VT_TT_KEEPQUANT_INT8DOT");
  const bool had_lever = prev_lever != nullptr;
  const std::string saved_lever =
      had_lever ? std::string(prev_lever) : std::string();
  struct RestoreLeverEnv {
    const bool had;
    const std::string saved;
    ~RestoreLeverEnv() {
      if (had) ::setenv("VT_TT_KEEPQUANT_INT8DOT", saved.c_str(), 1);
      else ::unsetenv("VT_TT_KEEPQUANT_INT8DOT");
    }
  } restore_lever_env{had_lever, saved_lever};
  ::unsetenv("VT_TT_KEEPQUANT_INT8DOT");

  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuant, vt::DeviceType::kTENSTORRENT));

  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();

  const vt::DType enc = vt::DType::kIQ1_S;
  const int64_t kBlockBytes = vt::BlockBytes(enc);
  const int64_t kBlockElems = vt::BlockElems(enc);
  REQUIRE(kBlockBytes == 50);
  REQUIRE(kBlockElems == 256);
  auto quant_act = vt::cpu::BlockFromFloat(vt::DType::kQ8_K);
  auto vec_dot = vt::cpu::BlockVecDot(enc);
  REQUIRE(quant_act != nullptr);
  REQUIRE(vec_dot != nullptr);

  // The sweep's PRNG style and packed generator, fixed seed.
  std::mt19937 rng(20260925u);
  auto rand_byte = [&rng]() { return static_cast<uint8_t>(rng() & 0xFF); };
  auto rand_f16_signed = [&rng](float lo, float span) {
    return (lo + span * static_cast<float>(rng() % 64) / 64.0f) *
           ((rng() % 2) != 0 ? 1.0f : -1.0f);
  };
  auto fill_block = [&](uint8_t* blk) {
    // block_iq1_s = { f16 d; u8 qs[32]; u16 qh[8] } (50 bytes) — qs the low
    // 8 grid-index bits, qh the 3-bit high index bits + 3-bit scale + sign
    // bit. Random bytes stay in the dot's bit-exact class (the per-block
    // accumulation is all-integer).
    const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
    std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
    for (int i = 0; i < 48; ++i) blk[2 + i] = rand_byte();  // qs + qh
  };

  // Decode shapes: small blocks (the f32-exact P==1 floor) plus the two APEX
  // decode shapes M=1 at the production tile widths (down-proj K=17408
  // N=5120, gate/up K=5120 N=17408). M=256 at the same widths is the
  // prefill-shapes leg above.
  struct Shape {
    const char* label;
    int64_t M;
    int64_t N;
    int64_t nb;
  };
  const Shape shapes[] = {
      {"small M=1", 1, 17, 16},
      {"small M=3", 3, 17, 16},
      {"down-proj decode", 1, 5120, 68},
      {"gate/up decode", 1, 17408, 20},
  };
  for (const Shape& sh : shapes) {
    const int64_t M = sh.M;
    const int64_t N = sh.N;
    const int64_t K = sh.nb * kBlockElems;
    std::vector<uint8_t> packed(N * sh.nb * kBlockBytes);
    for (int64_t b = 0; b < N * sh.nb; ++b)
      fill_block(packed.data() + b * kBlockBytes);
    std::vector<float> a_f32(M * K);
    for (auto& v : a_f32) v = (static_cast<float>(rng() % 401) - 200.0f) / 100.0f;
    std::vector<uint16_t> a_bf(M * K);
    for (size_t i = 0; i < a_f32.size(); ++i) a_bf[i] = vt::F32ToBF16(a_f32[i]);
    std::vector<float> a_q32(M * K);
    for (size_t i = 0; i < a_f32.size(); ++i) a_q32[i] = vt::BF16ToF32(a_bf[i]);

    // CPU integer oracle: quantize each activation row once, one nrc==1
    // vec_dot per (m, n) pair.
    const size_t y_row_bytes = vt::cpu::QuantActRowBytes(enc, K);
    std::vector<uint8_t> y(M * y_row_bytes);
    for (int64_t m = 0; m < M; ++m)
      quant_act(a_q32.data() + m * K, y.data() + m * y_row_bytes, K);
    std::vector<float> oracle(M * N);
    for (int64_t m = 0; m < M; ++m)
      for (int64_t n = 0; n < N; ++n)
        vec_dot(static_cast<int>(K), &oracle[static_cast<size_t>(m) * N + n],
                /*bs=*/0, packed.data() + static_cast<size_t>(n) * sh.nb * kBlockBytes,
                /*bx=*/0, y.data() + m * y_row_bytes, /*by=*/0, /*nrc=*/1);

    void* mem_a = backend.Alloc(a_bf.size() * sizeof(uint16_t));
    void* mem_b = backend.Alloc(packed.size());
    void* mem_o = backend.Alloc(std::max<size_t>(M * N, 16) * sizeof(float));
    backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
    backend.Copy(q, mem_b, packed.data(), packed.size());
    Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {M, K});
    Tensor b_t = Tensor::Contiguous(mem_b, enc,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {N, K});
    Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {M, N});
    vt::MatmulBT(q, o_t, a_t, b_t);  // route refusal under the mutation throws here
    std::vector<float> out(std::max<size_t>(M * N, 16), 0.0f);
    backend.Copy(q, out.data(), mem_o, out.size() * sizeof(float));
    backend.Free(mem_a);
    backend.Free(mem_b);
    backend.Free(mem_o);

    INFO("default path IQ1_S ", sh.label, " N=", N, " nb=", sh.nb, " K=", K,
         " M=", M);
    int64_t bad = 0;
    int64_t zero = 0;
    int64_t infs = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(M * N); ++i) {
      const float dev = out[static_cast<size_t>(i)];
      const float ref = oracle[static_cast<size_t>(i)];
      if (std::memcmp(&dev, &ref, sizeof(float)) != 0) {
        if (bad < 4)
          MESSAGE("diff m=", i / N, " n=", i % N, " dev=", dev,
                  " oracle=", ref);
        ++bad;
      }
      if (dev == 0.0f) ++zero;
      if (std::isinf(dev)) ++infs;
    }
    MESSAGE("IQ1_S ", sh.label, ": bad=", bad, " zero=", zero,
            " infs=", infs, " / ", M * N);
    CHECK(bad == 0);
    CHECK(zero < (M * N) / 2);
    CHECK(std::memcmp(out.data(), oracle.data(), M * N * sizeof(float)) == 0);
  }
  MESSAGE("default-path IQ1_S decode: bit-exact vs the CPU vec_dot with "
          "VT_TT_KEEPQUANT_INT8DOT unset");
}

// tenstorrent-gsq-keepquant wave 5: the IQ1_M default-path dispatch pin and
// the APEX decode-shape leg (M=1 GEMV at the two 27B production tile
// widths), the wave-4 Q2_K test as template. The lever is explicitly UNSET
// here and the bit-exact bar holds vs the CPU integer vec_dot oracle
// (VecDotIQ1_MQ8_K, the #3228 CPU arm). The dispatch-only mutation
// (override deleted, admission kept) refuses the route here and goes RED.
TEST_CASE("kTENSTORRENT kMatmulBTQuant IQ1_M serves the int8-dot arm on the DEFAULT path (VT_TT_KEEPQUANT_INT8DOT unset)") {
  // Default-path contract: the lever env must NOT be set for this leg.
  const char* prev_lever = std::getenv("VT_TT_KEEPQUANT_INT8DOT");
  const bool had_lever = prev_lever != nullptr;
  const std::string saved_lever =
      had_lever ? std::string(prev_lever) : std::string();
  struct RestoreLeverEnv {
    const bool had;
    const std::string saved;
    ~RestoreLeverEnv() {
      if (had) ::setenv("VT_TT_KEEPQUANT_INT8DOT", saved.c_str(), 1);
      else ::unsetenv("VT_TT_KEEPQUANT_INT8DOT");
    }
  } restore_lever_env{had_lever, saved_lever};
  ::unsetenv("VT_TT_KEEPQUANT_INT8DOT");

  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuant, vt::DeviceType::kTENSTORRENT));

  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();

  const vt::DType enc = vt::DType::kIQ1_M;
  const int64_t kBlockBytes = vt::BlockBytes(enc);
  const int64_t kBlockElems = vt::BlockElems(enc);
  REQUIRE(kBlockBytes == 56);
  REQUIRE(kBlockElems == 256);
  auto quant_act = vt::cpu::BlockFromFloat(vt::DType::kQ8_K);
  auto vec_dot = vt::cpu::BlockVecDot(enc);
  REQUIRE(quant_act != nullptr);
  REQUIRE(vec_dot != nullptr);

  // The sweep's PRNG style and packed generator, fixed seed.
  std::mt19937 rng(20260925u);
  auto rand_byte = [&rng]() { return static_cast<uint8_t>(rng() & 0xFF); };
  auto fill_block = [&](uint8_t* blk) {
    // block_iq1_m = { u8 qs[32]; u8 qh[16]; u8 scales[8] } (56 bytes) — no
    // f16 field: the super-block scale is spliced from the top nibbles of
    // scales. The splice does NOT stay bit-exact under fully random bytes:
    // a random exponent produces inf/NaN scales (both sides, different
    // payloads), so the exponent is constrained to a finite 2^3-scale.
    for (int i = 0; i < 56; ++i) blk[i] = rand_byte();
    blk[55] = 0x30;     // sb[7]: packed bits 15-12 = 0b0011 (sign 0,
    blk[53] &= 0x03;    // sb[5]: exponent bit 10 = 0 (exp 12, d in [8,16))
  };

  // Decode shapes: small blocks (the f32-exact P==1 floor) plus the two APEX
  // decode shapes M=1 at the production tile widths (down-proj K=17408
  // N=5120, gate/up K=5120 N=17408). M=256 at the same widths is the
  // prefill-shapes leg above.
  struct Shape {
    const char* label;
    int64_t M;
    int64_t N;
    int64_t nb;
  };
  const Shape shapes[] = {
      {"small M=1", 1, 17, 16},
      {"small M=3", 3, 17, 16},
      {"down-proj decode", 1, 5120, 68},
      {"gate/up decode", 1, 17408, 20},
  };
  for (const Shape& sh : shapes) {
    const int64_t M = sh.M;
    const int64_t N = sh.N;
    const int64_t K = sh.nb * kBlockElems;
    std::vector<uint8_t> packed(N * sh.nb * kBlockBytes);
    for (int64_t b = 0; b < N * sh.nb; ++b)
      fill_block(packed.data() + b * kBlockBytes);
    std::vector<float> a_f32(M * K);
    for (auto& v : a_f32) v = (static_cast<float>(rng() % 401) - 200.0f) / 100.0f;
    std::vector<uint16_t> a_bf(M * K);
    for (size_t i = 0; i < a_f32.size(); ++i) a_bf[i] = vt::F32ToBF16(a_f32[i]);
    std::vector<float> a_q32(M * K);
    for (size_t i = 0; i < a_f32.size(); ++i) a_q32[i] = vt::BF16ToF32(a_bf[i]);

    // CPU integer oracle: quantize each activation row once, one nrc==1
    // vec_dot per (m, n) pair.
    const size_t y_row_bytes = vt::cpu::QuantActRowBytes(enc, K);
    std::vector<uint8_t> y(M * y_row_bytes);
    for (int64_t m = 0; m < M; ++m)
      quant_act(a_q32.data() + m * K, y.data() + m * y_row_bytes, K);
    std::vector<float> oracle(M * N);
    for (int64_t m = 0; m < M; ++m)
      for (int64_t n = 0; n < N; ++n)
        vec_dot(static_cast<int>(K), &oracle[static_cast<size_t>(m) * N + n],
                /*bs=*/0, packed.data() + static_cast<size_t>(n) * sh.nb * kBlockBytes,
                /*bx=*/0, y.data() + m * y_row_bytes, /*by=*/0, /*nrc=*/1);

    void* mem_a = backend.Alloc(a_bf.size() * sizeof(uint16_t));
    void* mem_b = backend.Alloc(packed.size());
    void* mem_o = backend.Alloc(std::max<size_t>(M * N, 16) * sizeof(float));
    backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
    backend.Copy(q, mem_b, packed.data(), packed.size());
    Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {M, K});
    Tensor b_t = Tensor::Contiguous(mem_b, enc,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {N, K});
    Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {M, N});
    vt::MatmulBT(q, o_t, a_t, b_t);  // route refusal under the mutation throws here
    std::vector<float> out(std::max<size_t>(M * N, 16), 0.0f);
    backend.Copy(q, out.data(), mem_o, out.size() * sizeof(float));
    backend.Free(mem_a);
    backend.Free(mem_b);
    backend.Free(mem_o);

    INFO("default path IQ1_M ", sh.label, " N=", N, " nb=", sh.nb, " K=", K,
         " M=", M);
    int64_t bad = 0;
    int64_t zero = 0;
    int64_t infs = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(M * N); ++i) {
      const float dev = out[static_cast<size_t>(i)];
      const float ref = oracle[static_cast<size_t>(i)];
      if (std::memcmp(&dev, &ref, sizeof(float)) != 0) {
        if (bad < 4)
          MESSAGE("diff m=", i / N, " n=", i % N, " dev=", dev,
                  " oracle=", ref);
        ++bad;
      }
      if (dev == 0.0f) ++zero;
      if (std::isinf(dev)) ++infs;
    }
    MESSAGE("IQ1_M ", sh.label, ": bad=", bad, " zero=", zero,
            " infs=", infs, " / ", M * N);
    CHECK(bad == 0);
    CHECK(zero < (M * N) / 2);
    CHECK(std::memcmp(out.data(), oracle.data(), M * N * sizeof(float)) == 0);
  }
  MESSAGE("default-path IQ1_M decode: bit-exact vs the CPU vec_dot with "
          "VT_TT_KEEPQUANT_INT8DOT unset");
}

// KEEPQUANT W3 (issue #2959): the decode set generalizes to the other GGUF
// k-quants the q4km vehicle actually stores — Q5_K (attn_qkv/ssm_out), Q6_K
// (token_embd, tied LM head, half of ffn_down/attn_v) and Q8_0 (ssm_alpha/
// ssm_beta). The numerics bar is W1's, unchanged: the f32 decode is BIT-EXACT
// against the CPU decoder `vt::cpu::BlockToFloat` (cpu_quant_dequant.cpp —
// dequantize_row_q5_K:1673, q6_K:1881, q8_0:495 at the pinned llama.cpp
// b10451), because these are the blocks the dot consumes and a band here is
// unattributable downstream. Same sweep shape as W1: rows {1,3,17} x blocks
// {1,2,16}, deterministic weights, both signs, plus the zero corners the
// device decode must repair (signed-zero products from a zero scale byte or
// a +/-0 d; q6's -32 bias crossing zero; q8_0's full int8 range).
TEST_CASE("kTENSTORRENT kKeepQuantDecode matches vt::cpu::BlockToFloat bit-exactly (Q5_K/Q6_K/Q8_0 sweep)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kKeepQuantDecode, vt::DeviceType::kTENSTORRENT));

  const vt::DType encodings[] = {vt::DType::kQ5_K, vt::DType::kQ6_K,
                                 vt::DType::kQ8_0};
  for (const vt::DType enc : encodings) {
    const int64_t kBlockBytes = vt::BlockBytes(enc);
    const int64_t kBlockElems = vt::BlockElems(enc);
    REQUIRE(kBlockElems * kBlockBytes > 0);
    if (enc == vt::DType::kQ5_K) {
      REQUIRE(kBlockBytes == 176);
      REQUIRE(kBlockElems == 256);
    } else if (enc == vt::DType::kQ6_K) {
      REQUIRE(kBlockBytes == 210);
      REQUIRE(kBlockElems == 256);
    } else {
      REQUIRE(kBlockBytes == 34);
      REQUIRE(kBlockElems == 32);
    }

    Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
    auto decode = reinterpret_cast<vt::KeepQuantDecodeFn>(
        vt::GetOp(vt::OpId::kKeepQuantDecode, vt::DeviceType::kTENSTORRENT));
    Queue q = backend.CreateQueue();

    // Deterministic packed blocks with structural variety (the W1 generator,
    // generalized): PRNG bytes for quants and scales, d (and dmin where the
    // encoding has one) drawn from finite f16 magnitudes with both signs —
    // random BYTES would make the f16 scales NaN/Inf and the bit comparison
    // vacuous against a NaN-propagating decode. Every 7th block pins a
    // signed-zero corner (d = +/-0, dmin = the other zero), every 11th an
    // all-zero scale byte set (d*sc = +/-0, the signed-zero repair path).
    std::mt19937 rng(20260907u);
    auto rand_byte = [&rng]() { return static_cast<uint8_t>(rng() & 0xFF); };

    const int64_t rows_list[] = {1, 3, 17};
    const int64_t nb_list[] = {1, 2, 16};
    for (int64_t rows : rows_list) {
      for (int64_t nb : nb_list) {
        const int64_t k = nb * kBlockElems;
        std::vector<uint8_t> packed(rows * nb * kBlockBytes);
        for (int64_t b = 0; b < rows * nb; ++b) {
          uint8_t* blk = packed.data() + b * kBlockBytes;
          const bool zero_d = (b % 7) == 3;
          const bool zero_scales = (b % 11) == 5;
          auto put_f16 = [&](int64_t off, float v) {
            const uint16_t bits = vt::F32ToF16(v);
            std::memcpy(blk + off, &bits, sizeof(bits));
          };
          auto put_scales = [&](int64_t off, int64_t n) {
            for (int64_t i = 0; i < n; ++i)
              blk[off + i] = zero_scales ? 0u : rand_byte();
          };
          if (enc == vt::DType::kQ5_K) {
            // block_q5_K = { f16 d; f16 dmin; u8 scales[12]; u8 qh[32];
            //                u8 qs[128]; } (176 bytes)
            const float d = zero_d ? ((b % 2) ? -0.0f : 0.0f)
                                   : (0.05f + 0.35f * static_cast<float>(rng() % 64) / 64.0f) *
                                         ((rng() % 2) != 0 ? 1.0f : -1.0f);
            const float dmin = zero_d ? ((b % 2) ? 0.0f : -0.0f)
                                      : (0.005f + 0.02f * static_cast<float>(rng() % 32) / 32.0f) *
                                            ((rng() % 2) != 0 ? 1.0f : -1.0f);
            put_f16(0, d);
            put_f16(2, dmin);
            put_scales(4, 12);
            for (int i = 0; i < 32; ++i) blk[16 + i] = rand_byte();  // qh
            for (int i = 0; i < 128; ++i) blk[48 + i] = rand_byte();  // qs
          } else if (enc == vt::DType::kQ6_K) {
            // block_q6_K = { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d; }
            // (210 bytes)
            for (int i = 0; i < 128; ++i) blk[0 + i] = rand_byte();    // ql
            for (int i = 0; i < 64; ++i) blk[128 + i] = rand_byte();   // qh
            put_scales(192, 16);
            const float d = zero_d ? ((b % 2) ? -0.0f : 0.0f)
                                   : (0.05f + 0.35f * static_cast<float>(rng() % 64) / 64.0f) *
                                         ((rng() % 2) != 0 ? 1.0f : -1.0f);
            put_f16(208, d);
          } else {
            // block_q8_0 = { f16 d; i8 qs[32]; } (34 bytes)
            const float d = zero_d ? ((b % 2) ? -0.0f : 0.0f)
                                   : (0.05f + 0.35f * static_cast<float>(rng() % 64) / 64.0f) *
                                         ((rng() % 2) != 0 ? 1.0f : -1.0f);
            put_f16(0, d);
            for (int i = 0; i < 32; ++i)
              blk[2 + i] = zero_scales ? 0u : rand_byte();  // full int8 range
          }
        }

        std::vector<float> oracle(rows * k);
        vt::cpu::BlockToFloat(enc)(packed.data(), oracle.data(), rows * k);

        void* mem_packed = backend.Alloc(packed.size());
        void* mem_out = backend.Alloc(oracle.size() * sizeof(float));
        backend.Copy(q, mem_packed, packed.data(), packed.size());

        Tensor packed_t =
            Tensor::Contiguous(mem_packed, enc,
                               Device{vt::DeviceType::kTENSTORRENT, 0}, {rows, nb});
        Tensor out_t = Tensor::Contiguous(mem_out, vt::DType::kF32,
                                          Device{vt::DeviceType::kTENSTORRENT, 0}, {rows, k});
        decode(q, out_t, packed_t);

        std::vector<float> device_out(rows * k, 0.0f);
        backend.Copy(q, device_out.data(), mem_out, oracle.size() * sizeof(float));
        backend.Free(mem_packed);
        backend.Free(mem_out);

        INFO("enc=", static_cast<int>(enc), " rows=", rows, " nb=", nb, " K=", k);
        if (std::memcmp(device_out.data(), oracle.data(),
                        oracle.size() * sizeof(float)) != 0) {
          const float* dev = device_out.data();
          int64_t bad = 0;
          for (int64_t i = 0; i < static_cast<int64_t>(oracle.size()); ++i) {
            if (std::memcmp(&dev[i], &oracle[i], sizeof(float)) != 0) {
              if (bad < 4)
                MESSAGE("diff i=", i, " (blk=", i / kBlockElems, " col=", i % kBlockElems,
                        ") dev=", dev[i], " oracle=", oracle[i]);
              ++bad;
            }
          }
          MESSAGE("total bad: ", bad, " / ", oracle.size());
        }
        CHECK(std::memcmp(device_out.data(), oracle.data(), oracle.size() * sizeof(float)) == 0);
      }
    }
  }
}

// KEEPQUANT W3: the DOT admits the whole registered set. Enters through
// vt::MatmulBT's public dispatch (the entry a GGUF load actually takes), one
// shape per encoding — the per-encoding numerics authority is the bit-exact
// decode proven above, so the dot needs one envelope check each, not a sweep.
// Also pins the REFUSE side: an encoding outside the registered set must
// throw naming itself and the registered four, never fall through to a
// misread.
TEST_CASE("kTENSTORRENT kMatmulBTQuant admits the registered keep-quant set via vt::MatmulBT") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuant, vt::DeviceType::kTENSTORRENT));

  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();
  auto widen = [](uint16_t u) {
    uint32_t bits = static_cast<uint32_t>(u) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
  };

  constexpr int64_t M = 4, N = 8;
  std::mt19937 rng(20260908u);
  const vt::DType encodings[] = {vt::DType::kQ4_K, vt::DType::kQ5_K,
                                 vt::DType::kQ6_K, vt::DType::kQ8_0};
  for (const vt::DType enc : encodings) {
    const int64_t kBlockBytes = vt::BlockBytes(enc);
    const int64_t kBlockElems = vt::BlockElems(enc);
    const int64_t K = 2 * kBlockElems;  // two whole blocks per row
    std::vector<uint8_t> packed(N * 2 * kBlockBytes);
    for (int64_t b = 0; b < N * 2; ++b) {
      uint8_t* blk = packed.data() + b * kBlockBytes;
      auto put_f16 = [&](int64_t off, float v) {
        const uint16_t bits = vt::F32ToF16(v);
        std::memcpy(blk + off, &bits, sizeof(bits));
      };
      if (enc == vt::DType::kQ8_0) {
        put_f16(0, 0.1f + 0.2f * static_cast<float>(rng() % 16) / 16.0f);
        for (int i = 0; i < 32; ++i) blk[2 + i] = static_cast<uint8_t>(rng() & 0xFF);
      } else if (enc == vt::DType::kQ6_K) {
        for (int i = 0; i < 128; ++i) blk[0 + i] = static_cast<uint8_t>(rng() & 0xFF);
        for (int i = 0; i < 64; ++i) blk[128 + i] = static_cast<uint8_t>(rng() & 0xFF);
        for (int i = 0; i < 16; ++i) blk[192 + i] = static_cast<uint8_t>(rng() & 0xFF);
        put_f16(208, 0.1f + 0.2f * static_cast<float>(rng() % 16) / 16.0f);
      } else {
        put_f16(0, 0.05f + 0.35f * static_cast<float>(rng() % 64) / 64.0f);
        if (enc == vt::DType::kQ5_K) {
          put_f16(2, 0.005f + 0.02f * static_cast<float>(rng() % 32) / 32.0f);
          for (int i = 0; i < 32; ++i) blk[16 + i] = static_cast<uint8_t>(rng() & 0xFF);
          for (int i = 0; i < 128; ++i) blk[48 + i] = static_cast<uint8_t>(rng() & 0xFF);
        } else {
          put_f16(2, 0.005f + 0.02f * static_cast<float>(rng() % 32) / 32.0f);
          for (int i = 0; i < 128; ++i) blk[16 + i] = static_cast<uint8_t>(rng() & 0xFF);
        }
        for (int i = 0; i < 12; ++i) blk[4 + i] = static_cast<uint8_t>(rng() & 0xFF);
      }
    }
    std::vector<float> a_f32(M * K);
    for (auto& v : a_f32) v = (static_cast<float>(rng() % 401) - 200.0f) / 100.0f;

    // Oracle: bit-exact decode -> round ONCE to bf16 -> f32 accumulate in
    // ascending k; plus the analytic bf16 operand-rounding envelope (the W2
    // bound, unchanged).
    std::vector<float> w_f32(N * K);
    vt::cpu::BlockToFloat(enc)(packed.data(), w_f32.data(), N * K);
    std::vector<uint16_t> a_bf(M * K), w_bf(N * K);
    for (size_t i = 0; i < a_f32.size(); ++i) a_bf[i] = vt::F32ToBF16(a_f32[i]);
    for (size_t i = 0; i < w_f32.size(); ++i) w_bf[i] = vt::F32ToBF16(w_f32[i]);
    std::vector<float> ref(M * N), bound(M * N);
    for (int64_t m = 0; m < M; ++m)
      for (int64_t n = 0; n < N; ++n) {
        float acc = 0.0f, mag = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          const float p = widen(a_bf[static_cast<size_t>(m) * K + k]) *
                          widen(w_bf[static_cast<size_t>(n) * K + k]);
          acc += p;
          mag += std::fabs(p);
        }
        ref[static_cast<size_t>(m) * N + n] = acc;
        bound[static_cast<size_t>(m) * N + n] =
            1.05f * std::ldexp(1.0f, -8) * (mag + std::fabs(acc));
      }

    void* mem_a = backend.Alloc(M * K * sizeof(uint16_t));
    void* mem_b = backend.Alloc(packed.size());
    void* mem_o = backend.Alloc(M * N * sizeof(float));
    backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
    backend.Copy(q, mem_b, packed.data(), packed.size());
    Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {M, K});
    Tensor b_t = Tensor::Contiguous(mem_b, enc,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {N, K});
    Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {M, N});
    vt::MatmulBT(q, o_t, a_t, b_t);
    std::vector<float> out(M * N, 0.0f);
    backend.Copy(q, out.data(), mem_o, out.size() * sizeof(float));
    backend.Free(mem_a);
    backend.Free(mem_b);
    backend.Free(mem_o);

    float worst = 0.0f, worst_ratio = 0.0f;
    for (int64_t i = 0; i < M * N; ++i) {
      const float diff = std::fabs(out[static_cast<size_t>(i)] - ref[static_cast<size_t>(i)]);
      worst = std::max(worst, diff);
      worst_ratio = std::max(worst_ratio, diff / bound[static_cast<size_t>(i)]);
      CHECK(std::isfinite(out[static_cast<size_t>(i)]));
      CHECK_MESSAGE(diff <= bound[static_cast<size_t>(i)],
                    "enc=" << static_cast<int>(enc) << " i=" << i
                           << " out=" << out[i] << " ref=" << ref[i]
                           << " bound=" << bound[i]);
    }
    MESSAGE("kMatmulBTQuant enc=", static_cast<int>(enc),
            ": worst_abs=", worst, " worst bound-ratio=", worst_ratio);
  }

  // THE REFUSE SIDE: an unregistered encoding must throw naming ITSELF (and
  // the registered four), never fall through to a misread. kQ4_0 has no TT
  // arm anywhere.
  {
    const int64_t kBlockBytes = vt::BlockBytes(vt::DType::kQ4_0);
    const int64_t kBlockElems = vt::BlockElems(vt::DType::kQ4_0);
    const int64_t K = 2 * kBlockElems;
    std::vector<uint8_t> packed(N * 2 * kBlockBytes, 0u);
    void* mem_a = backend.Alloc(M * K * sizeof(uint16_t));
    void* mem_b = backend.Alloc(packed.size());
    void* mem_o = backend.Alloc(M * N * sizeof(float));
    Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {M, K});
    Tensor b_t = Tensor::Contiguous(mem_b, vt::DType::kQ4_0,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {N, K});
    Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {M, N});
    bool threw = false;
    std::string what;
    try {
      vt::MatmulBT(q, o_t, a_t, b_t);
    } catch (const std::exception& e) {
      threw = true;
      what = e.what();
    }
    backend.Free(mem_a);
    backend.Free(mem_b);
    backend.Free(mem_o);
    CHECK_MESSAGE(threw, "kQ4_0 must refuse on TENSTORRENT, not misread");
    CHECK_MESSAGE(what.find("kQ4_0") != std::string::npos,
                  "the refusal must name the encoding, got: ", what);
    CHECK_MESSAGE(what.find("kQ8_0") != std::string::npos,
                  "the refusal must name the registered set, got: ", what);
  }
}

// KEEPQUANT W3, THE CAPTURE LEG (red-first): the decode's staging must not
// write during a trace capture. Pre-staging, every decoded call EnsureHosts
// the packed bytes, repacks them to i32 words on the host and from_vector-
// uploads them — a per-call host round trip that inside a capture is the
// #2812 class (the captured graph pins capture-time bytes its replay cannot
// refresh). The counter (KeepQuantCaptureStagingWrites) is the observable:
// warm eagerly, reset, capture + replay, require ZERO staging writes and
// replay bytes identical to the eager run.
TEST_CASE("kTENSTORRENT keep-quant decode stages zero words during capture") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  REQUIRE(backend.SupportsGraphCapture());
  auto decode = reinterpret_cast<vt::KeepQuantDecodeFn>(
      vt::GetOp(vt::OpId::kKeepQuantDecode, vt::DeviceType::kTENSTORRENT));
  Queue q = backend.CreateQueue();

  const vt::DType encodings[] = {vt::DType::kQ4_K, vt::DType::kQ8_0};
  for (const vt::DType enc : encodings) {
    const int64_t kBlockBytes = vt::BlockBytes(enc);
    const int64_t kBlockElems = vt::BlockElems(enc);
    constexpr int64_t kRows = 3, kNb = 2;
    const int64_t k = kNb * kBlockElems;
    std::mt19937 rng(20260909u);
    std::vector<uint8_t> packed(kRows * kNb * kBlockBytes);
    for (int64_t b = 0; b < kRows * kNb; ++b) {
      uint8_t* blk = packed.data() + b * kBlockBytes;
      const float d = (0.05f + 0.35f * static_cast<float>(rng() % 64) / 64.0f) *
                      ((rng() % 2) != 0 ? 1.0f : -1.0f);
      const uint16_t d_bits = vt::F32ToF16(d);
      std::memcpy(blk, &d_bits, sizeof(d_bits));
      if (enc == vt::DType::kQ4_K) {
        // dmin must stay a FINITE f16 (the W1 generator's rule): random f16
        // bits go NaN and the bit-exact comparison dies on the device's
        // NaN-payload canonicalization, not on a real decode defect.
        const float dmin = (0.005f + 0.02f * static_cast<float>(rng() % 32) / 32.0f) *
                           ((rng() % 2) != 0 ? 1.0f : -1.0f);
        const uint16_t dmin_bits = vt::F32ToF16(dmin);
        std::memcpy(blk + 2, &dmin_bits, sizeof(dmin_bits));
        for (int i = 4; i < kBlockBytes; ++i) blk[i] = static_cast<uint8_t>(rng() & 0xFF);
      } else {
        for (int i = 2; i < kBlockBytes; ++i) blk[i] = static_cast<uint8_t>(rng() & 0xFF);
      }
    }
    std::vector<float> oracle(kRows * k);
    vt::cpu::BlockToFloat(enc)(packed.data(), oracle.data(), kRows * k);

    void* mem_packed = backend.Alloc(packed.size());
    void* mem_out = backend.Alloc(oracle.size() * sizeof(float));
    backend.Copy(q, mem_packed, packed.data(), packed.size());
    Tensor packed_t =
        Tensor::Contiguous(mem_packed, enc, Device{vt::DeviceType::kTENSTORRENT, 0}, {kRows, kNb});
    Tensor out_t = Tensor::Contiguous(mem_out, vt::DType::kF32,
                                      Device{vt::DeviceType::kTENSTORRENT, 0}, {kRows, k});

    // Eager warm: populates the word shadow + program cache, and proves the
    // decode before it is captured.
    decode(q, out_t, packed_t);
    std::vector<float> eager(oracle.size(), 0.0f);
    backend.Copy(q, eager.data(), mem_out, eager.size() * sizeof(float));
    CHECK(std::memcmp(eager.data(), oracle.data(), oracle.size() * sizeof(float)) == 0);

    vt::tenstorrent::ResetKeepQuantCaptureStagingWritesForTest();
    backend.BeginCapture(q);
    decode(q, out_t, packed_t);
    backend.EndCapture(q);
    const int64_t writes = vt::tenstorrent::KeepQuantCaptureStagingWrites();
    CHECK_MESSAGE(writes == 0,
                  "keep-quant decode staged ", writes,
                  " word uploads DURING capture (the #2812 class) for enc=",
                  static_cast<int>(enc));

    backend.Replay(q);
    std::vector<float> after(oracle.size(), 0.0f);
    backend.Copy(q, after.data(), mem_out, after.size() * sizeof(float));
    INFO("enc=", static_cast<int>(enc));
    CHECK(std::memcmp(after.data(), oracle.data(), oracle.size() * sizeof(float)) == 0);
    backend.Free(mem_packed);
    backend.Free(mem_out);
  }
}

// KEEPQUANT W4a wave-2 (#3030): the GROUPED keep-quant GEMM on the P150 —
// vt::MatmulBTQuantGrouped (ops.cpp:220) with the packed [E*N,K] tower resident
// in its i32 word form, the P selected [N,K] row-slices decoded on-core per
// call (the W3 chains on a word slice), and the kMatmulBT tile matmul per
// group. No bf16 twin, never a full-tower decode. This case is the ROUTING
// leg, red-first: before the provider was registered the first REQUIRE reded,
// and the seam refused through GetOp's no-provider path ("no kernel for op
// MatmulBTQuantGrouped ... on device TENSTORRENT") — never a silent wrong
// answer. The registered encoding set is EXACTLY {Q4_K, Q8_0}: Q5_K and Q6_K
// must refuse BY NAME (the owed grouped extension, recorded in the spec's W4
// plan), so a wrongly-widened kernel reds the refusal legs below.
TEST_CASE("kTENSTORRENT kMatmulBTQuantGrouped registers {Q4_K,Q5_K,Q6_K,Q8_0} and refuses the rest") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuantGrouped,
                           vt::DeviceType::kTENSTORRENT));

  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();

  // One Q4_K block per row: K = 256. The block generator is the W3 one:
  // d/dmin are finite f16 (random f16 bits go NaN and the device's
  // NaN-payload canonicalization would mask a real defect as a decode
  // failure), payload bytes are random.
  auto fill_block = [](uint8_t* blk, vt::DType enc, std::mt19937& rng) {
    auto put_f16 = [&](int64_t off, float v) {
      const uint16_t bits = vt::F32ToF16(v);
      std::memcpy(blk + off, &bits, sizeof(bits));
    };
    if (enc == vt::DType::kQ8_0) {
      put_f16(0, 0.1f + 0.2f * static_cast<float>(rng() % 16) / 16.0f);
      for (int i = 2; i < 34; ++i) blk[i] = static_cast<uint8_t>(rng() & 0xFF);
    } else if (enc == vt::DType::kQ4_K) {
      put_f16(0, 0.05f + 0.35f * static_cast<float>(rng() % 64) / 64.0f);
      put_f16(2, 0.005f + 0.02f * static_cast<float>(rng() % 32) / 32.0f);
      for (int i = 4; i < 144; ++i) blk[i] = static_cast<uint8_t>(rng() & 0xFF);
    } else if (enc == vt::DType::kQ5_K) {
      // block_q5_k: f16 d, f16 dmin, u8 scales[12], qh[32], ql[128]. All
      // payload bytes are integers the decode reads exactly; only the two
      // f16 scales need finite values.
      put_f16(0, 0.05f + 0.35f * static_cast<float>(rng() % 64) / 64.0f);
      put_f16(2, 0.005f + 0.02f * static_cast<float>(rng() % 32) / 32.0f);
      for (int i = 4; i < 176; ++i) blk[i] = static_cast<uint8_t>(rng() & 0xFF);
    } else {  // Q6_K
      // block_q6_k: ql[128], qh[64], i8 scales[16], f16 d. Same reasoning.
      for (int i = 0; i < 208; ++i) blk[i] = static_cast<uint8_t>(rng() & 0xFF);
      put_f16(208, 0.05f + 0.35f * static_cast<float>(rng() % 64) / 64.0f);
    }
  };

  // THE ADMITTED ARMS: one call each, E=2 tower, all four encodings. Only
  // reachability is pinned here — the numerics have their own sweep below.
  for (const vt::DType enc :
       {vt::DType::kQ4_K, vt::DType::kQ5_K, vt::DType::kQ6_K,
        vt::DType::kQ8_0}) {
    const int64_t kBlockBytes = vt::BlockBytes(enc);
    const int64_t kBlockElems = vt::BlockElems(enc);
    constexpr int64_t kE = 2, kN = 8, kP = 3;
    const int64_t k = 1 * kBlockElems;
    std::mt19937 rng(20260910u);
    std::vector<uint8_t> packed(kE * kN * (k / kBlockElems) * kBlockBytes);
    for (size_t b = 0; b < packed.size() / kBlockBytes; ++b)
      fill_block(packed.data() + b * kBlockBytes, enc, rng);
    std::vector<uint16_t> a_bf(kP * k);
    for (auto& v : a_bf) v = vt::F32ToBF16((static_cast<float>(rng() % 401) - 200.0f) / 100.0f);
    std::vector<int32_t> ids(kP);
    for (auto& e : ids) e = static_cast<int32_t>(rng() % kE);

    void* mem_a = backend.Alloc(a_bf.size() * sizeof(uint16_t));
    void* mem_w = backend.Alloc(packed.size());
    void* mem_o = backend.Alloc(kP * kN * sizeof(float));
    void* mem_i = backend.Alloc(ids.size() * sizeof(int32_t));
    backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
    backend.Copy(q, mem_w, packed.data(), packed.size());
    backend.Copy(q, mem_i, ids.data(), ids.size() * sizeof(int32_t));
    Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {kP, k});
    Tensor w_t = Tensor::Contiguous(mem_w, enc,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {kE * kN, k});
    Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {kP, kN});
    Tensor i_t = Tensor::Contiguous(mem_i, vt::DType::kI32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {kP});
    bool threw = false;
    std::string what;
    try {
      vt::MatmulBTQuantGrouped(q, o_t, a_t, w_t, i_t);
    } catch (const std::exception& e) {
      threw = true;
      what = e.what();
    }
    backend.Free(mem_a);
    backend.Free(mem_w);
    backend.Free(mem_o);
    backend.Free(mem_i);
    const std::string enc_ok = vt::Name(enc);
    CHECK_MESSAGE(!threw, "the registered encoding ", enc_ok,
                  " must answer on the TENSTORRENT grouped arm, threw: ", what);
  }

  // THE REFUSE SIDE: kQ4_0 (no TT arm anywhere) must throw naming ITSELF and
  // the four-encoding registered set — never fall through to a misread, never
  // silently widen past the set.
  for (const vt::DType enc : {vt::DType::kQ4_0}) {
    const int64_t kBlockBytes = vt::BlockBytes(enc);
    const int64_t kBlockElems = vt::BlockElems(enc);
    constexpr int64_t kE = 2, kN = 8, kP = 3;
    const int64_t k = 1 * kBlockElems;
    std::vector<uint8_t> packed(kE * kN * (k / kBlockElems) * kBlockBytes, 0u);
    std::vector<uint16_t> a_bf(kP * k, 0u);
    std::vector<int32_t> ids(kP, 0);
    void* mem_a = backend.Alloc(a_bf.size() * sizeof(uint16_t));
    void* mem_w = backend.Alloc(packed.size());
    void* mem_o = backend.Alloc(kP * kN * sizeof(float));
    void* mem_i = backend.Alloc(ids.size() * sizeof(int32_t));
    Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {kP, k});
    Tensor w_t = Tensor::Contiguous(mem_w, enc,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {kE * kN, k});
    Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {kP, kN});
    Tensor i_t = Tensor::Contiguous(mem_i, vt::DType::kI32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {kP});
    bool threw = false;
    std::string what;
    try {
      vt::MatmulBTQuantGrouped(q, o_t, a_t, w_t, i_t);
    } catch (const std::exception& e) {
      threw = true;
      what = e.what();
    }
    backend.Free(mem_a);
    backend.Free(mem_w);
    backend.Free(mem_o);
    backend.Free(mem_i);
    CHECK_MESSAGE(threw, "unregistered encoding ", static_cast<int>(enc),
                  " must refuse on the TENSTORRENT grouped arm, not misread");
    const std::string enc_lower = vt::Name(enc);
    const std::string enc_name = std::string("k") +
        static_cast<char>(enc_lower[0] - 'a' + 'A') + enc_lower.substr(1);
    CHECK_MESSAGE(what.find(enc_name) != std::string::npos,
                  "the refusal must name the encoding (", enc_name,
                  "), got: ", what);
    CHECK_MESSAGE(what.find("kQ4_K/kQ5_K/kQ6_K/kQ8_0") != std::string::npos,
                  "the refusal must name the registered set, got: ", what);
  }
}

// THE NUMERICS LEG (W4a wave-2, red-first): the grouped op against the CPU
// grouped provider (cpu_quant_gemm.cpp MatmulBTQuantGroupedKernel — the SAME
// kMatmulBTQuant integer-dot core once per group) across shapes sweeping P, N,
// K, E: the E=1 dense arm (ids all zero), the E=N expert tower arm, P=1, a
// broadcast activation ([1,K]), a non-tile-multiple N, all four registered
// encodings, and a 27B-mirroring slice (K = the Qwen3.8-27B hidden_size,
// N a production-like per-group intermediate). The bar is the W2-ratified
// analytic operand-rounding envelope — bf16-round-once both operands, f32
// ascending-k accumulate, bound = 1.05 * 2^-8 * (mag + |acc|) — unchanged
// from the dense W2/W3 pin; the reported bound ratios are the wave evidence.
// Decode bit-exactness on the SLICE path is pinned separately below: a Q8_0
// tower with power-of-two scales and one-hot activations makes every dot term
// exact, so the grouped output must be BIT-EQUAL to the dequantized weight
// element the slice selection picked.
TEST_CASE("kTENSTORRENT kMatmulBTQuantGrouped matches the CPU grouped provider inside the ratified envelope") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuantGrouped,
                           vt::DeviceType::kTENSTORRENT));

  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();
  vt::Queue qcpu{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  // W4a wave-3a: force the E=1 arms through MANY chunks per call so the
  // envelope ratios below measure the chunked slice-decode (policy default
  // chunks would cover these small N in one pass). Reset before every
  // return path — the guard covers the whole case body.
  vt::tenstorrent::KeepQuantChunkRowsOverrideForTest(3);
  struct ChunkReset {
    ~ChunkReset() { vt::tenstorrent::KeepQuantChunkRowsOverrideForTest(0); }
  } chunk_reset;
  auto widen = [](uint16_t u) {
    uint32_t bits = static_cast<uint32_t>(u) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
  };
  auto fill_block = [](uint8_t* blk, vt::DType enc, std::mt19937& rng) {
    auto put_f16 = [&](int64_t off, float v) {
      const uint16_t bits = vt::F32ToF16(v);
      std::memcpy(blk + off, &bits, sizeof(bits));
    };
    if (enc == vt::DType::kQ8_0) {
      put_f16(0, 0.1f + 0.2f * static_cast<float>(rng() % 16) / 16.0f);
      for (int i = 2; i < 34; ++i) blk[i] = static_cast<uint8_t>(rng() & 0xFF);
    } else if (enc == vt::DType::kQ4_K) {
      put_f16(0, 0.05f + 0.35f * static_cast<float>(rng() % 64) / 64.0f);
      put_f16(2, 0.005f + 0.02f * static_cast<float>(rng() % 32) / 32.0f);
      for (int i = 4; i < 144; ++i) blk[i] = static_cast<uint8_t>(rng() & 0xFF);
    } else if (enc == vt::DType::kQ5_K) {
      // block_q5_k: f16 d, f16 dmin, u8 scales[12], qh[32], ql[128]. The
      // payload bytes are integers the decode reads exactly; only the two
      // f16 scales need finite values.
      put_f16(0, 0.05f + 0.35f * static_cast<float>(rng() % 64) / 64.0f);
      put_f16(2, 0.005f + 0.02f * static_cast<float>(rng() % 32) / 32.0f);
      for (int i = 4; i < 176; ++i) blk[i] = static_cast<uint8_t>(rng() & 0xFF);
    } else {  // Q6_K
      // block_q6_k: ql[128], qh[64], i8 scales[16], f16 d. Same reasoning.
      for (int i = 0; i < 208; ++i) blk[i] = static_cast<uint8_t>(rng() & 0xFF);
      put_f16(208, 0.05f + 0.35f * static_cast<float>(rng() % 64) / 64.0f);
    }
  };

  struct Shape {
    int64_t p, n, k_blocks, e, act_rows;
    vt::DType enc;
    const char* note;
  };
  const Shape shapes[] = {
      {3, 8, 1, 1, -1, vt::DType::kQ4_K, "E=1 dense arm, ids all zero"},
      {2, 8, 1, 1, 1, vt::DType::kQ4_K,
       "E=1 broadcast [1,K] act, P=2 (wave-3a replication)"},
      {4, 8, 2, 4, -1, vt::DType::kQ4_K, "E=N expert tower arm, permuted ids"},
      {1, 16, 1, 2, -1, vt::DType::kQ4_K, "P=1"},
      {4, 33, 1, 4, 1, vt::DType::kQ4_K, "broadcast [1,K] act, non-tile N=33"},
      {3, 8, 2, 2, -1, vt::DType::kQ8_0, "Q8_0 tower arm"},
      {4, 33, 1, 4, -1, vt::DType::kQ8_0, "Q8_0, non-tile N=33"},
      {2, 1024, 20, 1, -1, vt::DType::kQ4_K,
       "27B mirror: K=5120 (Qwen3.8-27B hidden_size), N=1024 per-group slice"},
      // W4a wave-2b: the Q5_K/Q6_K grouped extension. No ROCm grouped
      // reference exists for Q5_K (rocm_grouped_gemm.hip admits
      // Q8_0/Q4_K/Q6_K); the decode is the W3 dense chain, bit-exact vs
      // vt::cpu::BlockToFloat — the same numerics authority for both arms.
      {3, 8, 1, 1, -1, vt::DType::kQ5_K, "Q5_K E=1 dense arm, ids all zero"},
      {4, 8, 2, 4, -1, vt::DType::kQ5_K,
       "Q5_K E=N expert tower arm, permuted ids"},
      {4, 33, 1, 4, 1, vt::DType::kQ5_K,
       "Q5_K broadcast [1,K] act, non-tile N=33"},
      {2, 1024, 20, 1, -1, vt::DType::kQ5_K,
       "Q5_K 27B mirror: K=5120, N=1024 per-group slice (the 27B pin carries "
       "48 Q5_K tensors)"},
      {3, 8, 1, 1, -1, vt::DType::kQ6_K, "Q6_K E=1 dense arm, ids all zero"},
      {4, 8, 2, 4, -1, vt::DType::kQ6_K,
       "Q6_K E=N expert tower arm, permuted ids"},
      {4, 33, 1, 4, 1, vt::DType::kQ6_K,
       "Q6_K broadcast [1,K] act, non-tile N=33"},
      {2, 1024, 20, 1, -1, vt::DType::kQ6_K,
       "Q6_K 27B mirror: K=5120, N=1024 per-group slice (the 27B pin carries "
       "67 Q6_K tensors)"},
  };
  for (const Shape& s : shapes) {
    const int64_t kBlockBytes = vt::BlockBytes(s.enc);
    const int64_t kBlockElems = vt::BlockElems(s.enc);
    const int64_t P = s.p, N = s.n, K = s.k_blocks * kBlockElems, E = s.e;
    const int64_t Pa = s.act_rows == -1 ? P : s.act_rows;
    const int64_t nb = K / kBlockElems;
    std::mt19937 rng(static_cast<uint32_t>(20260911u + P * 7 + N * 13 + E * 3));

    std::vector<uint8_t> packed(E * N * nb * kBlockBytes);
    for (size_t b = 0; b < packed.size() / kBlockBytes; ++b)
      fill_block(packed.data() + b * kBlockBytes, s.enc, rng);
    std::vector<float> a_f32(Pa * K);
    for (auto& v : a_f32) v = (static_cast<float>(rng() % 401) - 200.0f) / 100.0f;
    std::vector<uint16_t> a_bf(a_f32.size());
    for (size_t i = 0; i < a_f32.size(); ++i) a_bf[i] = vt::F32ToBF16(a_f32[i]);
    std::vector<int32_t> ids(P);
    for (auto& e : ids) e = static_cast<int32_t>(rng() % E);
    if (E == 1) std::fill(ids.begin(), ids.end(), 0);

    // ---- the CPU grouped provider on the IDENTICAL bytes ----
    std::vector<float> cpu_out(P * N, 0.0f);
    {
      Tensor at = Tensor::Contiguous(a_bf.data(), vt::DType::kBF16, qcpu.device, {Pa, K});
      Tensor ot = Tensor::Contiguous(cpu_out.data(), vt::DType::kF32, qcpu.device, {P, N});
      Tensor it = Tensor::Contiguous(ids.data(), vt::DType::kI32, qcpu.device, {P});
      Tensor wt = Tensor::Contiguous(packed.data(), vt::DType::kF32, qcpu.device, {E * N, K});
      wt.dtype = s.enc;  // block dtype: elementwise strides are inert
      vt::MatmulBTQuantGrouped(qcpu, ot, at, wt, it);
    }

    // ---- the analytic bf16-operand reference + the W2 bound ----
    std::vector<float> w_f32(E * N * K);
    vt::cpu::BlockToFloat(s.enc)(packed.data(), w_f32.data(), E * N * K);
    std::vector<uint16_t> w_bf(w_f32.size());
    for (size_t i = 0; i < w_f32.size(); ++i) w_bf[i] = vt::F32ToBF16(w_f32[i]);
    std::vector<float> ref(P * N), bound(P * N);
    for (int64_t p = 0; p < P; ++p) {
      const int64_t e = ids[p];
      const int64_t pa = Pa == 1 ? 0 : p;
      for (int64_t n = 0; n < N; ++n) {
        float acc = 0.0f, mag = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          const float prod = widen(a_bf[pa * K + k]) *
                             widen(w_bf[(e * N + n) * K + k]);
          acc += prod;
          mag += std::fabs(prod);
        }
        ref[p * N + n] = acc;
        bound[p * N + n] = 1.05f * std::ldexp(1.0f, -8) * (mag + std::fabs(acc));
      }
    }

    // ---- the device grouped call ----
    void* mem_a = backend.Alloc(a_bf.size() * sizeof(uint16_t));
    void* mem_w = backend.Alloc(packed.size());
    void* mem_o = backend.Alloc(P * N * sizeof(float));
    void* mem_i = backend.Alloc(ids.size() * sizeof(int32_t));
    backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
    backend.Copy(q, mem_w, packed.data(), packed.size());
    backend.Copy(q, mem_i, ids.data(), ids.size() * sizeof(int32_t));
    Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {Pa, K});
    Tensor w_t = Tensor::Contiguous(mem_w, s.enc,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {E * N, K});
    Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {P, N});
    Tensor i_t = Tensor::Contiguous(mem_i, vt::DType::kI32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {P});
    vt::MatmulBTQuantGrouped(q, o_t, a_t, w_t, i_t);
    std::vector<float> tt_out(P * N, 0.0f);
    backend.Copy(q, tt_out.data(), mem_o, tt_out.size() * sizeof(float));
    backend.Free(mem_a);
    backend.Free(mem_w);
    backend.Free(mem_o);
    backend.Free(mem_i);

    float worst_cpu = 0.0f, worst_ref = 0.0f;
    for (int64_t i = 0; i < P * N; ++i) {
      const float d_cpu = std::fabs(tt_out[i] - cpu_out[i]);
      const float d_ref = std::fabs(tt_out[i] - ref[i]);
      worst_cpu = std::max(worst_cpu, d_cpu / bound[i]);
      worst_ref = std::max(worst_ref, d_ref / bound[i]);
      CHECK(std::isfinite(tt_out[i]));
      CHECK_MESSAGE(d_cpu <= bound[i],
                    "P=" << P << " N=" << N << " K=" << K << " E=" << E
                         << " enc=" << static_cast<int>(s.enc) << " i=" << i
                         << " tt=" << tt_out[i] << " cpu=" << cpu_out[i]
                         << " bound=" << bound[i]);
      CHECK_MESSAGE(d_ref <= bound[i],
                    "P=" << P << " N=" << N << " K=" << K << " E=" << E
                         << " enc=" << static_cast<int>(s.enc) << " i=" << i
                         << " tt=" << tt_out[i] << " ref=" << ref[i]
                         << " bound=" << bound[i]);
    }
    MESSAGE("grouped P=", P, " N=", N, " K=", K, " E=", E, " Pa=", Pa,
            " enc=", static_cast<int>(s.enc), " (", s.note,
            "): worst bound-ratio vs cpu=", worst_cpu,
            " vs analytic-bf16-ref=", worst_ref);
  }

  // ---- THE SLICE-DECODE BIT-EXACT LEG ----
  // Q8_0, d = +2^-6, qs = full int8: every dequantized element qs*2^-6 is
  // exactly representable in bf16, so the on-core decode -> one bf16 RNE ->
  // tile matmul (bf16 output) chain is LOSSLESS for one-hot activations, and
  // the grouped output must be BIT-EQUAL to the dequant of the row-slice the
  // routing picked — any slice-selection defect (wrong expert row-range,
  // whole-tower misindex, bf16-twin aliasing) lands on a DIFFERENT grid value
  // and cannot hide. The CPU grouped provider is NOT the bit oracle here: its
  // dot is the INTEGER-dot core (the activation is quantized to q8 on the CPU
  // side), so TT-vs-CPU agreement is the envelope's job — the sweep above.
  {
    constexpr int64_t kE = 3, kN = 8, kP = 5, kNb = 2;
    const int64_t K = kNb * 32;
    const uint16_t d_bits = vt::F32ToF16(std::ldexp(1.0f, -6));
    std::mt19937 rng(20260912u);
    std::vector<uint8_t> packed(kE * kN * kNb * 34);
    std::vector<int8_t> qs(kE * kN * kNb * 32);
    for (int64_t b = 0; b < kE * kN * kNb; ++b) {
      uint8_t* blk = packed.data() + b * 34;
      std::memcpy(blk, &d_bits, sizeof(d_bits));
      for (int i = 0; i < 32; ++i) {
        const int8_t q = static_cast<int8_t>(rng() % 255 - 127);  // [-127,127]
        blk[2 + i] = static_cast<uint8_t>(q);
        qs[b * 32 + i] = q;
      }
    }
    // One-hot activations: row p selects column (p % K); the group may repeat
    // (two rows route to the same expert) and the ids are NOT ascending, so
    // every selected slice is exercised independently.
    std::vector<uint16_t> a_bf(kP * K, 0u);
    for (int64_t p = 0; p < kP; ++p) a_bf[p * K + (p % K)] = vt::F32ToBF16(1.0f);
    std::vector<int32_t> ids = {2, 0, 2, 1, 0};

    // The true dequant of the selected slice: out[p,n] = w_dec[ids[p], n,
    // p%K]. Block b's qs live at qs[b*32 + i]; bf16(w_dec) == w_dec exactly,
    // and the device commits the bf16 matmul output widened to f32, so the
    // comparison is memcmp over the widened bits.
    std::vector<float> dequant(kP * kN);
    for (int64_t p = 0; p < kP; ++p)
      for (int64_t n = 0; n < kN; ++n) {
        const int64_t c = p % K;
        const int64_t b = (ids[p] * kN + n) * kNb + c / 32;
        dequant[p * kN + n] =
            static_cast<float>(qs[b * 32 + c % 32]) * std::ldexp(1.0f, -6);
      }

    void* mem_a = backend.Alloc(a_bf.size() * sizeof(uint16_t));
    void* mem_w = backend.Alloc(packed.size());
    void* mem_o = backend.Alloc(kP * kN * sizeof(float));
    void* mem_i = backend.Alloc(ids.size() * sizeof(int32_t));
    backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
    backend.Copy(q, mem_w, packed.data(), packed.size());
    backend.Copy(q, mem_i, ids.data(), ids.size() * sizeof(int32_t));
    Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {kP, K});
    Tensor w_t = Tensor::Contiguous(mem_w, vt::DType::kQ8_0,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {kE * kN, K});
    Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {kP, kN});
    Tensor i_t = Tensor::Contiguous(mem_i, vt::DType::kI32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {kP});
    vt::MatmulBTQuantGrouped(q, o_t, a_t, w_t, i_t);
    std::vector<float> tt_out(kP * kN, 0.0f);
    backend.Copy(q, tt_out.data(), mem_o, tt_out.size() * sizeof(float));
    backend.Free(mem_a);
    backend.Free(mem_w);
    backend.Free(mem_o);
    backend.Free(mem_i);
    for (int64_t i = 0; i < kP * kN; ++i) {
      const uint16_t got_bf = vt::F32ToBF16(tt_out[i]);
      const uint16_t want_bf = vt::F32ToBF16(dequant[i]);
      CHECK_MESSAGE(got_bf == want_bf,
                    "slice-decode bit-exact: p=" << i / kN << " n=" << i % kN
                                                 << " tt=" << tt_out[i]
                                                 << " dequant=" << dequant[i]);
    }
  }

  // ---- THE Q5_K/Q6_K SLICE-DECODE BIT-EXACT LEGS (W4a wave-2b) ----
  // The Q8_0 leg above pins slice selection on a bf16-exact grid. These two
  // pin the W3 dense chains — the only Q5_K/Q6_K decode there is: the ROCm
  // grouped kernel has no Q5_K arm to mirror, so the decode derives from the
  // W3 chain and its bit-exactness vs vt::cpu::BlockToFloat — through the
  // SLICE path: with one-hot activations the grouped output is exactly
  // bf16(decode(selected row, col)), and decode is W3-pinned bit-exact vs
  // BlockToFloat, so the expected bits are bf16(BlockToFloat) at the routed
  // element. A slice-selection or staging defect (wrong expert row-range,
  // word-lane misindex, a read past the 210-byte block into the pad) lands on
  // a different value and cannot hide.
  for (const vt::DType enc : {vt::DType::kQ5_K, vt::DType::kQ6_K}) {
    constexpr int64_t kE = 3, kN = 8, kP = 5, kNb = 1;
    const int64_t elems = vt::BlockElems(enc);  // 256 per K-quant block
    const int64_t bb = vt::BlockBytes(enc);     // 176 (Q5_K) / 210 (Q6_K)
    const int64_t K = kNb * elems;
    std::mt19937 rng(static_cast<uint32_t>(20260913u));
    std::vector<uint8_t> packed(kE * kN * kNb * bb);
    for (int64_t b = 0; b < kE * kN * kNb; ++b) {
      uint8_t* blk = packed.data() + b * bb;
      fill_block(blk, enc, rng);
      if (enc == vt::DType::kQ6_K) {
        // Q6_K dequant y = (d*sc)*(q-32): a NEGATIVE or zero scale meeting a
        // zero q makes the true dequant -0, which any dot then flattens to
        // +0 in the f32 accumulate (IEEE (+0)+(-0) = +0) — the DECODE keeps
        // the -0 (the W3 pin, or_sign repair at the Q6_K arm), but a matmul
        // cannot carry it through a sum. The bit-exact leg, exactly like the
        // Q8_0 leg above, therefore constrains its data to the -0-free
        // class: strictly positive scales make sign(y) = sign(q-32) with
        // q = 32 giving +0.
        for (int i = 192; i < 208; ++i)
          blk[i] = static_cast<uint8_t>(1 + rng() % 127);
      }
    }
    // One-hot activations with repeated, non-ascending ids — every selected
    // slice exercised independently, exactly as the Q8_0 leg above.
    std::vector<uint16_t> a_bf(kP * K, 0u);
    for (int64_t p = 0; p < kP; ++p) a_bf[p * K + (p % K)] = vt::F32ToBF16(1.0f);
    std::vector<int32_t> ids = {2, 0, 2, 1, 0};

    // The decode oracle: BlockToFloat over the IDENTICAL bytes (the W3 bit
    // authority), then the ONE bf16 RNE the device applies after the slice
    // decode. Every payload is an exact f32 integer product under finite
    // positive scales, so no -0/NaN ambiguity survives the chain.
    std::vector<float> w_f32(kE * kN * K);
    vt::cpu::BlockToFloat(enc)(packed.data(), w_f32.data(), kE * kN * K);
    std::vector<float> dequant(kP * kN);
    for (int64_t p = 0; p < kP; ++p)
      for (int64_t n = 0; n < kN; ++n)
        dequant[p * kN + n] = w_f32[(ids[p] * kN + n) * K + (p % K)];

    void* mem_a = backend.Alloc(a_bf.size() * sizeof(uint16_t));
    void* mem_w = backend.Alloc(packed.size());
    void* mem_o = backend.Alloc(kP * kN * sizeof(float));
    void* mem_i = backend.Alloc(ids.size() * sizeof(int32_t));
    backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
    backend.Copy(q, mem_w, packed.data(), packed.size());
    backend.Copy(q, mem_i, ids.data(), ids.size() * sizeof(int32_t));
    Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {kP, K});
    Tensor w_t = Tensor::Contiguous(mem_w, enc,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {kE * kN, K});
    Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {kP, kN});
    Tensor i_t = Tensor::Contiguous(mem_i, vt::DType::kI32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {kP});
    vt::MatmulBTQuantGrouped(q, o_t, a_t, w_t, i_t);
    std::vector<float> tt_out(kP * kN, 0.0f);
    backend.Copy(q, tt_out.data(), mem_o, tt_out.size() * sizeof(float));
    backend.Free(mem_a);
    backend.Free(mem_w);
    backend.Free(mem_o);
    backend.Free(mem_i);
    for (int64_t i = 0; i < kP * kN; ++i) {
      const uint16_t got_bf = vt::F32ToBF16(tt_out[i]);
      const uint16_t want_bf = vt::F32ToBF16(dequant[i]);
      const std::string msg = std::string("slice-decode bit-exact (") +
                              vt::Name(enc) + "): p=" +
                              std::to_string(i / kN) + " n=" +
                              std::to_string(i % kN) + " tt=" +
                              std::to_string(tt_out[i]) + " dequant=" +
                              std::to_string(dequant[i]);
      CHECK_MESSAGE(got_bf == want_bf, msg);
    }
  }
}

// THE DECODE f32-EXACT LEG: M=1 (decode) wide-range Q4_K grouped GEMV vs a
// double-accumulated f32-domain CPU oracle over the exact BlockToFloat
// weight decode and the SAME bf16-widened activation the device stages. The
// pre-fix decode branch typecast BOTH the decoded f32 weight tile and the
// f32 activation to bf16 and ran a full bf16 TILE matmul; the oracle
// accumulates in f64 over exact f32 weights. Wide-range activations
// (state-like ±10^uniform(-3,2)) make that operand rounding first-order:
// small-magnitude outputs — the elements the elementwise envelope exists to
// catch — carry a relative error far past the 0.002 gate. Shape mirrors a
// 27B decode slice (N=1024 per-group intermediate, K=5120 = Qwen3.8-27B
// hidden_size).
TEST_CASE("kTENSTORRENT kMatmulBTQuantGrouped decode (P=1) matches the CPU f32 quantized dot within the elementwise envelope") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuantGrouped,
                           vt::DeviceType::kTENSTORRENT));

  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();

  const vt::DType enc = vt::DType::kQ4_K;
  const int64_t kBlockBytes = vt::BlockBytes(enc);
  const int64_t kBlockElems = vt::BlockElems(enc);
  constexpr int64_t kE = 1, kN = 1024, kNb = 20;  // K = 5120, 27B hidden_size
  const int64_t K = kNb * kBlockElems;

  std::mt19937 rng(20260913u);
  std::vector<uint8_t> packed(kE * kN * kNb * kBlockBytes);
  for (size_t b = 0; b < packed.size() / kBlockBytes; ++b) {
    uint8_t* blk = packed.data() + b * kBlockBytes;
    const uint16_t d_bits =
        vt::F32ToF16((0.05f + 0.35f * static_cast<float>(rng() % 64) / 64.0f) *
                     ((rng() % 2) != 0 ? 1.0f : -1.0f));
    const uint16_t dmin_bits =
        vt::F32ToF16(0.005f + 0.02f * static_cast<float>(rng() % 32) / 32.0f);
    std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
    std::memcpy(blk + 2, &dmin_bits, sizeof(dmin_bits));
    for (int i = 4; i < kBlockBytes; ++i) blk[i] = static_cast<uint8_t>(rng() & 0xFF);
  }
  // Wide-range state-like activation: ±10^uniform(-3,2), one row (M=1).
  std::vector<float> a_f32(K);
  for (float& x : a_f32) {
    const float mag = std::pow(10.0f, -3.0f + 5.0f * (static_cast<float>(rng() % 1024) / 1024.0f));
    x = ((rng() % 2) != 0 ? -1.0f : 1.0f) * mag;
  }
  std::vector<uint16_t> a_bf(a_f32.size());
  for (size_t i = 0; i < a_f32.size(); ++i) a_bf[i] = vt::F32ToBF16(a_f32[i]);
  std::vector<float> a_q32(a_f32.size());
  for (size_t i = 0; i < a_f32.size(); ++i) a_q32[i] = vt::BF16ToF32(a_bf[i]);
  std::vector<int32_t> ids(1, 0);

  // CPU f32-domain oracle: the exact weight decode the device must mirror
  // (BlockToFloat, W3-pinned bit-exact) dotted against the SAME bf16-widened
  // activation the device stages, accumulated in double. NOT the q8_K
  // vec_dot: quantizing the activation to q8_K is the int8-dot arm's domain
  // (bit-exact, pinned by its own sweep); this arm keeps activations
  // unquantized, so the q8_K rounding (~0.4-2% elementwise) would swamp the
  // gate. What the envelope proves is that no bf16 ROUNDING detour remains
  // on the device dot.
  auto block_to_float = vt::cpu::BlockToFloat(enc);
  REQUIRE(block_to_float != nullptr);
  std::vector<float> w_f32(kE * kN * K);
  block_to_float(packed.data(), w_f32.data(), kE * kN * K);
  std::vector<float> oracle(kN, 0.0f);
  for (int64_t n = 0; n < kN; ++n) {
    double acc = 0.0;
    for (int64_t c = 0; c < K; ++c)
      acc += static_cast<double>(w_f32[static_cast<size_t>(n) * K + c]) *
             static_cast<double>(a_q32[static_cast<size_t>(c)]);
    oracle[static_cast<size_t>(n)] = static_cast<float>(acc);
  }

  void* mem_a = backend.Alloc(a_bf.size() * sizeof(uint16_t));
  void* mem_w = backend.Alloc(packed.size());
  void* mem_o = backend.Alloc(kN * sizeof(float));
  void* mem_i = backend.Alloc(ids.size() * sizeof(int32_t));
  backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
  backend.Copy(q, mem_w, packed.data(), packed.size());
  backend.Copy(q, mem_i, ids.data(), ids.size() * sizeof(int32_t));
  Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                  Device{vt::DeviceType::kTENSTORRENT, 0}, {1, K});
  Tensor w_t = Tensor::Contiguous(mem_w, enc,
                                  Device{vt::DeviceType::kTENSTORRENT, 0}, {kE * kN, K});
  Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                  Device{vt::DeviceType::kTENSTORRENT, 0}, {1, kN});
  Tensor i_t = Tensor::Contiguous(mem_i, vt::DType::kI32,
                                  Device{vt::DeviceType::kTENSTORRENT, 0}, {1});
  vt::MatmulBTQuantGrouped(q, o_t, a_t, w_t, i_t);
  std::vector<float> out(kN, 0.0f);
  backend.Copy(q, out.data(), mem_o, out.size() * sizeof(float));
  backend.Free(mem_a);
  backend.Free(mem_w);
  backend.Free(mem_o);
  backend.Free(mem_i);

  // Elementwise envelope: 0.002 relative + a 1e-5 abs floor (the f32 device
  // compute agrees with the f32 scalar oracle at ~1e-6 relative; a bf16
  // operand rounding detour measures far past 0.002 on the small outputs).
  const double rel_tol = 0.002, abs_floor = 1e-5;
  double worst_rel = 0.0, worst_abs = 0.0;
  int64_t bad = 0;
  for (int64_t i = 0; i < kN; ++i) {
    const double ref = static_cast<double>(oracle[static_cast<size_t>(i)]);
    const double d = std::fabs(static_cast<double>(out[static_cast<size_t>(i)]) - ref);
    const double lim = rel_tol * std::fabs(ref) + abs_floor;
    worst_rel = std::max(worst_rel, d / (std::fabs(ref) + 1e-30));
    worst_abs = std::max(worst_abs, d);
    if (d > lim) {
      if (bad < 4)
        MESSAGE("diff n=", i, " dev=", out[static_cast<size_t>(i)],
                " oracle=", oracle[static_cast<size_t>(i)], " |d|=", d,
                " lim=", lim);
      ++bad;
    }
  }
  MESSAGE("grouped decode P=1 Q4_K N=", kN, " K=", K, ": worst_rel=", worst_rel,
          " worst_abs=", worst_abs, " bad=", bad, "/", kN,
          " (rel_tol=", rel_tol, " abs_floor=", abs_floor, ")");
  CHECK_MESSAGE(bad == 0, "grouped decode drift past the elementwise envelope: ",
                bad, " of ", kN, " outputs (worst_rel=", worst_rel, ")");
}
// ---------------------------------------------------------------------------
// KEEPQUANT W4a wave-3a (#3030): the E=1 (dense) grouped arm becomes
// capture-compatible and memory-bounded. Three legs:
//   1. THE TRACE-BOUND CASE (red-first, the wave-1b falsification class):
//      a head-shaped [248320, 1024] Q6_K E=1 op CAPTURED whole — before the
//      chunked slice-decode the per-call whole-[N,K] decode persists as a
//      bf16 tile inside the captured graph and end_trace_capture demands
//      425,754,624 B-class trace demand against the 52,428,800 B region
//      (TT_FATAL, mesh_trace.cpp:81). After: bounded chunk tiles, capture
//      succeeds, and the device-reported demand is recorded.
//   2. Capture-clean ids: E=1 ids are statically all zero (the only
//      in-range expert), so the op must not EnsureHost/read them; the
//      E=N tower arm keeps today's range check.
//   3. Bit-exactness: the chunked decode itself must match BlockToFloat —
//      decode math unchanged, only the loop bounds move.
// ---------------------------------------------------------------------------

TEST_CASE("kTENSTORRENT E=1 grouped keep-quant capture survives the 50 MiB trace region") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  REQUIRE(backend.SupportsGraphCapture());
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuantGrouped,
                           vt::DeviceType::kTENSTORRENT));
  Queue q = backend.CreateQueue();
  // The production chunk policy (no override): this leg measures it. The env
  // knob exists so one build can measure the trace-demand curve across chunk
  // counts (the wave-3a trace-region survey); it is a test-only lever.
  int64_t chunk_rows_override = 0;
  if (const char* env_rows = std::getenv("VT_KEEPQUANT_TEST_CHUNK_ROWS")) {
    const long long parsed = std::atoll(env_rows);
    if (parsed > 0) chunk_rows_override = static_cast<int64_t>(parsed);
  }
  vt::tenstorrent::KeepQuantChunkRowsOverrideForTest(chunk_rows_override);
  MESSAGE("chunk rows override: " << chunk_rows_override);

  struct ChunkReset {
    ~ChunkReset() { vt::tenstorrent::KeepQuantChunkRowsOverrideForTest(0); }
  } chunk_reset;

  // The 0.8B tied head: [248320, 1024] Q6_K. Decoded once as one tile that
  // is 254,274,560 elems ≈ 485 MiB of bf16 — ~9× the 52,428,800 B trace
  // region, and the f32 chain planes are ~2× that again.
  constexpr int64_t kN = 248320, kK = 1024, kP = 1, kPa = 1;
  const int64_t kElems = vt::BlockElems(vt::DType::kQ6_K);  // 256
  const int64_t kBB = vt::BlockBytes(vt::DType::kQ6_K);     // 210
  const int64_t kNb = kK / kElems;                          // 4

  std::mt19937 rng(20260914u);
  std::vector<uint8_t> packed(static_cast<size_t>(kN) * kNb * kBB);
  for (size_t b = 0; b < packed.size() / static_cast<size_t>(kBB); ++b) {
    uint8_t* blk = packed.data() + b * kBB;
    for (int i = 0; i < 208; ++i) blk[i] = static_cast<uint8_t>(rng() & 0xFF);
    const uint16_t d_bits =
        vt::F32ToF16(0.05f + 0.35f * static_cast<float>(rng() % 64) / 64.0f);
    std::memcpy(blk + 208, &d_bits, sizeof(d_bits));
  }
  std::vector<float> a_f32(static_cast<size_t>(kPa * kK));
  for (auto& v : a_f32) v = (static_cast<float>(rng() % 401) - 200.0f) / 100.0f;
  std::vector<uint16_t> a_bf(a_f32.size());
  for (size_t i = 0; i < a_f32.size(); ++i) a_bf[i] = vt::F32ToBF16(a_f32[i]);
  std::vector<int32_t> ids(kP, 0);

  // ---- the CPU grouped provider on the IDENTICAL bytes + the W2 bound ----
  std::vector<float> cpu_out(static_cast<size_t>(kP * kN), 0.0f);
  std::vector<float> bound(static_cast<size_t>(kP * kN), 0.0f);
  {
    vt::Queue qcpu{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
    Tensor at = Tensor::Contiguous(a_bf.data(), vt::DType::kBF16, qcpu.device,
                                   {kPa, kK});
    Tensor ot = Tensor::Contiguous(cpu_out.data(), vt::DType::kF32, qcpu.device,
                                   {kP, kN});
    Tensor it = Tensor::Contiguous(ids.data(), vt::DType::kI32, qcpu.device, {kP});
    Tensor wt = Tensor::Contiguous(packed.data(), vt::DType::kF32, qcpu.device,
                                   {kN, kK});
    wt.dtype = vt::DType::kQ6_K;  // block dtype: elementwise strides are inert
    vt::MatmulBTQuantGrouped(qcpu, ot, at, wt, it);

    auto widen = [](uint16_t u) {
      uint32_t bits = static_cast<uint32_t>(u) << 16;
      float f;
      std::memcpy(&f, &bits, 4);
      return f;
    };
    std::vector<float> w_f32(static_cast<size_t>(kN) * kK);
    vt::cpu::BlockToFloat(vt::DType::kQ6_K)(packed.data(), w_f32.data(),
                                            kN * kK);
    std::vector<uint16_t> w_bf(w_f32.size());
    for (size_t i = 0; i < w_f32.size(); ++i) w_bf[i] = vt::F32ToBF16(w_f32[i]);
    for (int64_t n = 0; n < kN; ++n) {
      float acc = 0.0f, mag = 0.0f;
      for (int64_t k = 0; k < kK; ++k) {
        const float prod =
            widen(a_bf[static_cast<size_t>(k)]) *
            widen(w_bf[static_cast<size_t>(n * kK + k)]);
        acc += prod;
        mag += std::fabs(prod);
      }
      bound[static_cast<size_t>(n)] =
          1.05f * std::ldexp(1.0f, -8) * (mag + std::fabs(acc));
    }
  }

  // ---- the device grouped call, warmed eagerly ----
  void* mem_a = backend.Alloc(a_bf.size() * sizeof(uint16_t));
  void* mem_w = backend.Alloc(packed.size());
  void* mem_o = backend.Alloc(static_cast<size_t>(kP * kN) * sizeof(float));
  void* mem_i = backend.Alloc(ids.size() * sizeof(int32_t));
  backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
  backend.Copy(q, mem_w, packed.data(), packed.size());
  backend.Copy(q, mem_i, ids.data(), ids.size() * sizeof(int32_t));
  Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                  Device{vt::DeviceType::kTENSTORRENT, 0},
                                  {kPa, kK});
  Tensor w_t = Tensor::Contiguous(mem_w, vt::DType::kQ6_K,
                                  Device{vt::DeviceType::kTENSTORRENT, 0},
                                  {kN, kK});
  Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                  Device{vt::DeviceType::kTENSTORRENT, 0},
                                  {kP, kN});
  Tensor i_t = Tensor::Contiguous(mem_i, vt::DType::kI32,
                                  Device{vt::DeviceType::kTENSTORRENT, 0}, {kP});
  vt::MatmulBTQuantGrouped(q, o_t, a_t, w_t, i_t);
  std::vector<float> eager(static_cast<size_t>(kP * kN), 0.0f);
  backend.Copy(q, eager.data(), mem_o, eager.size() * sizeof(float));

  float worst_cpu = 0.0f;
  for (int64_t n = 0; n < kN; ++n) {
    const float d =
        std::fabs(eager[static_cast<size_t>(n)] - cpu_out[static_cast<size_t>(n)]);
    worst_cpu = std::max(worst_cpu, d / bound[static_cast<size_t>(n)]);
    CHECK(std::isfinite(eager[static_cast<size_t>(n)]));
    CHECK_MESSAGE(d <= bound[static_cast<size_t>(n)],
                  "head-shape envelope: n=" << n << " tt="
                                            << eager[static_cast<size_t>(n)]
                                            << " cpu="
                                            << cpu_out[static_cast<size_t>(n)]
                                            << " bound="
                                            << bound[static_cast<size_t>(n)]);
  }
  MESSAGE("head-shape [", kN, ",", kK, "] Q6_K E=1: worst bound-ratio vs cpu=",
          worst_cpu);

  // ---- capture ×2 byte-identity (the #2907 discipline) ----
  std::vector<float> dumps[2];
  int64_t demand[2] = {0, 0};
  for (int pass = 0; pass < 2; ++pass) {
    vt::tenstorrent::ResetKeepQuantCaptureStagingWritesForTest();
    void* graph = nullptr;
    std::string what;
    bool threw = false;
    try {
      backend.BeginCapture(q);
      vt::MatmulBTQuantGrouped(q, o_t, a_t, w_t, i_t);
      graph = backend.EndCaptureGraph(q);
    } catch (const std::exception& ex) {
      threw = true;
      what = ex.what();
    }
    REQUIRE_MESSAGE(!threw, "capture pass " << pass
                                            << " threw (the wave-1b trace "
                                               "region fatal lives here): "
                                            << what);
    REQUIRE(graph != nullptr);
    demand[pass] = vt::tenstorrent::LastTraceBytesForTest();
    backend.ReplayGraph(q, graph);
    dumps[pass].resize(static_cast<size_t>(kP * kN), 0.0f);
    backend.Copy(q, dumps[pass].data(), mem_o,
                 dumps[pass].size() * sizeof(float));
    backend.DestroyGraph(graph);  // after the blocking readback above
    MESSAGE("capture pass ", pass, ": device trace demand ", demand[pass],
            " B (region 52428800 B), staging writes during capture=",
            vt::tenstorrent::KeepQuantCaptureStagingWrites());
    CHECK_MESSAGE(vt::tenstorrent::KeepQuantCaptureStagingWrites() == 0,
                  "E=1 chunked capture staged ",
                  vt::tenstorrent::KeepQuantCaptureStagingWrites(),
                  " word uploads DURING capture (the #2812 class)");
  }
  for (int pass = 0; pass < 2; ++pass) {
    REQUIRE_MESSAGE(demand[pass] <= 52428800,
                    "capture pass " << pass << " demanded " << demand[pass]
                                    << " B of trace region against 52428800 B");
    CHECK(std::memcmp(dumps[static_cast<size_t>(pass)].data(), eager.data(),
                      eager.size() * sizeof(float)) == 0);
  }
  CHECK(std::memcmp(dumps[1].data(), dumps[0].data(),
                    eager.size() * sizeof(float)) == 0);
  MESSAGE("capture x2 byte-identity: PASS; trace demand pass0=", demand[0],
          " B pass1=", demand[1], " B (region 52428800 B)");
  // Cleanup, once (the teardown double-free: an earlier revision indented
  // backend.Free(mem_i) INSIDE this pass loop, freeing the same buffer twice
  // per run before the trailing free — a plain triple free that glibc catches
  // nondeterministically, depending on the heap layout the run happens to
  // produce, which is why the abort looked flaky and "at tt-metal teardown").
  backend.Free(mem_a);
  backend.Free(mem_w);
  backend.Free(mem_o);
  backend.Free(mem_i);
}

// KEEPQUANT W4b (issue #3031), RED-FIRST for the int8-dot arm's capture
// safety. The F32-out dispatch (vt::MatmulBT with an f32 out and a keep-quant
// packed weight — the same entry the int8-dot sweep test uses) builds a fresh
// MeshWorkload per call, and EnqueueMeshWorkload always runs load_binaries on
// a fresh object; tt-metal fatals there whenever program_binary_status_ is
// empty while a trace is being captured (mesh_workload.cpp "Cannot load new
// binaries during trace capture"). This test is RED on that fatal and turns
// GREEN only when the arm persists and reuses the workload across calls (the
// ttnn program-cache reuse contract). Numerics are the sweep test's job; this
// leg owns the capture mechanics: warm eager run, capture x2 byte-identity
// (the #2907 discipline), zero staging writes during capture (the #2812
// class), and the trace region fit.
TEST_CASE("kTENSTORRENT E=1 int8-dot keep-quant capture survives the 50 MiB trace region (F32-out dispatch)") {
  // W4b landing decision: the lever is OP-LEVEL and DEFAULT OFF. On default
  // this f32-out dispatch serves the W4a grouped arm, so a capture run here
  // would capture the wrong arm and prove nothing about the lever — skip
  // loudly; run under VT_TT_KEEPQUANT_INT8DOT=1.
  if (const char* lever = std::getenv("VT_TT_KEEPQUANT_INT8DOT");
      lever == nullptr || lever[0] == '\0' || std::strcmp(lever, "0") == 0) {
    MESSAGE("SKIPPED: set VT_TT_KEEPQUANT_INT8DOT=1 — this capture asserts the "
            "int8-dot lever through the F32-out dispatch; the default "
            "dispatch is the W4a grouped arm");
    return;
  }
  ::setenv("VT_TT_KEEPQUANT_INT8DOT", "1", 1);  // canonical opt-in; the dispatch reads the env live per call
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  REQUIRE(backend.SupportsGraphCapture());
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuant, vt::DeviceType::kTENSTORRENT));
  Queue q = backend.CreateQueue();

  // The 0.8B tied head: [248320, 1024] Q6_K, one activation row — the same
  // production shape the grouped capture test above owns, entered here through
  // the F32-out dispatch so the captured arm is the int8-dot kernel.
  constexpr int64_t kN = 248320, kK = 1024, kM = 1;
  const int64_t kElems = vt::BlockElems(vt::DType::kQ6_K);  // 256
  const int64_t kBB = vt::BlockBytes(vt::DType::kQ6_K);     // 210
  const int64_t kNb = kK / kElems;                          // 4

  std::mt19937 rng(20260916u);
  std::vector<uint8_t> packed(static_cast<size_t>(kN) * kNb * kBB);
  for (size_t b = 0; b < packed.size() / static_cast<size_t>(kBB); ++b) {
    uint8_t* blk = packed.data() + b * kBB;
    for (int i = 0; i < 208; ++i) blk[i] = static_cast<uint8_t>(rng() & 0xFF);
    const uint16_t d_bits =
        vt::F32ToF16(0.05f + 0.35f * static_cast<float>(rng() % 64) / 64.0f);
    std::memcpy(blk + 208, &d_bits, sizeof(d_bits));
  }
  std::vector<uint16_t> a_bf(static_cast<size_t>(kM * kK));
  for (auto& v : a_bf) v = vt::F32ToBF16((static_cast<float>(rng() % 401) - 200.0f) / 100.0f);

  // ---- the device call, warmed eagerly: the words shadow stages here; a
  // capture-time miss refuses by name (the warm-first contract) ----
  void* mem_a = backend.Alloc(a_bf.size() * sizeof(uint16_t));
  void* mem_w = backend.Alloc(packed.size());
  void* mem_o = backend.Alloc(static_cast<size_t>(kM * kN) * sizeof(float));
  backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
  backend.Copy(q, mem_w, packed.data(), packed.size());
  Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                  Device{vt::DeviceType::kTENSTORRENT, 0},
                                  {kM, kK});
  Tensor w_t = Tensor::Contiguous(mem_w, vt::DType::kQ6_K,
                                  Device{vt::DeviceType::kTENSTORRENT, 0},
                                  {kN, kK});
  Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                  Device{vt::DeviceType::kTENSTORRENT, 0},
                                  {kM, kN});
  vt::MatmulBT(q, o_t, a_t, w_t);  // the PUBLIC dispatch: f32 out -> int8-dot
  std::vector<float> eager(static_cast<size_t>(kM * kN), 0.0f);
  backend.Copy(q, eager.data(), mem_o, eager.size() * sizeof(float));
  for (float v : eager) CHECK(std::isfinite(v));

  // ---- capture x2 byte-identity (the #2907 discipline) ----
  std::vector<float> dumps[2];
  int64_t demand[2] = {0, 0};
  for (int pass = 0; pass < 2; ++pass) {
    vt::tenstorrent::ResetKeepQuantCaptureStagingWritesForTest();
    void* graph = nullptr;
    std::string what;
    bool threw = false;
    try {
      backend.BeginCapture(q);
      vt::MatmulBT(q, o_t, a_t, w_t);
      graph = backend.EndCaptureGraph(q);
    } catch (const std::exception& ex) {
      threw = true;
      what = ex.what();
    }
    REQUIRE_MESSAGE(!threw, "capture pass " << pass
                                            << " threw (the int8-dot workload "
                                               "capture fatal lives here): "
                                            << what);
    REQUIRE(graph != nullptr);
    demand[pass] = vt::tenstorrent::LastTraceBytesForTest();
    backend.ReplayGraph(q, graph);
    dumps[pass].resize(static_cast<size_t>(kM * kN), 0.0f);
    backend.Copy(q, dumps[pass].data(), mem_o,
                 dumps[pass].size() * sizeof(float));
    backend.DestroyGraph(graph);  // after the blocking readback above
    MESSAGE("capture pass ", pass, ": device trace demand ", demand[pass],
            " B (region 52428800 B), staging writes during capture=",
            vt::tenstorrent::KeepQuantCaptureStagingWrites());
    CHECK_MESSAGE(vt::tenstorrent::KeepQuantCaptureStagingWrites() == 0,
                  "int8-dot capture staged ",
                  vt::tenstorrent::KeepQuantCaptureStagingWrites(),
                  " word uploads DURING capture (the #2812 class)");
  }
  for (int pass = 0; pass < 2; ++pass) {
    REQUIRE_MESSAGE(demand[pass] <= 52428800,
                    "capture pass " << pass << " demanded " << demand[pass]
                                    << " B of trace region against 52428800 B");
    CHECK_MESSAGE(std::memcmp(dumps[static_cast<size_t>(pass)].data(),
                              eager.data(),
                              eager.size() * sizeof(float)) == 0,
                  "replay pass " << pass
                                 << " diverged from the warm eager run");
  }
  CHECK(std::memcmp(dumps[1].data(), dumps[0].data(),
                    eager.size() * sizeof(float)) == 0);
  MESSAGE("capture x2 byte-identity: PASS; trace demand pass0=", demand[0],
          " B pass1=", demand[1], " B (region 52428800 B)");
  backend.Free(mem_a);
  backend.Free(mem_w);
  backend.Free(mem_o);
}

// W4d W6: the BF16-OUT dispatch joined the int8-dot lever. The W4b landing
// decision refused bf16-out because committing the kernel's f32 dev_out into
// a bf16 slot left the slot holding f32 bytes at an f32 page geometry — the
// next bf16 reader got word-halved garbage (the ROW_MAJOR chained leg's NaN
// signature). The fix is an explicit f32->bf16 cast before the commit; this
// test is its red-first lock (RED on the pre-cast tree: the replay output
// word-halves against the eager reference).
TEST_CASE("kTENSTORRENT E=1 int8-dot keep-quant capture survives the 50 MiB trace region (BF16-out dispatch)") {
  // Same lever gate as the F32-out sibling: on default the bf16-out
  // dispatch serves the W4a grouped arm and this test would prove nothing.
  if (const char* lever = std::getenv("VT_TT_KEEPQUANT_INT8DOT");
      lever == nullptr || lever[0] == '\0' || std::strcmp(lever, "0") == 0) {
    MESSAGE("SKIPPED: set VT_TT_KEEPQUANT_INT8DOT=1 — this capture asserts the "
            "int8-dot lever through the BF16-out dispatch");
    return;
  }
  ::setenv("VT_TT_KEEPQUANT_INT8DOT", "1", 1);
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  REQUIRE(backend.SupportsGraphCapture());
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuant, vt::DeviceType::kTENSTORRENT));
  Queue q = backend.CreateQueue();

  // Two chained keep-quant matmuls: m1 (bf16 out) feeds m2 (f32 out) as its
  // ACTIVATION. The W4b store-geometry bug lived in exactly that handoff:
  // the f32 dev_out committed into the bf16 slot, and the next DEVICE
  // reader (m2's activation load) word-halved it. A host download converts
  // and hides the poison, so the probe is the device-side consumer.
  constexpr int64_t kN1 = 4096, kK = 1024, kN2 = 1024, kM = 1;
  const int64_t kElems = vt::BlockElems(vt::DType::kQ6_K);
  const int64_t kBB = vt::BlockBytes(vt::DType::kQ6_K);
  const int64_t kNb1 = kK / kElems, kNb2 = kN1 / kElems;

  std::mt19937 rng(20260913u);
  auto fill_q6k = [&](std::vector<uint8_t>& packed, uint32_t seed) {
    std::mt19937 r(seed);
    packed.resize(static_cast<size_t>(packed.size()));
    for (size_t b = 0; b < packed.size() / static_cast<size_t>(kBB); ++b) {
      uint8_t* blk = packed.data() + b * kBB;
      for (int i = 0; i < 208; ++i) blk[i] = static_cast<uint8_t>(r() & 0xFF);
      const uint16_t d_bits =
          vt::F32ToF16(0.05f + 0.35f * static_cast<float>(r() % 64) / 64.0f);
      std::memcpy(blk + 208, &d_bits, sizeof(d_bits));
    }
  };
  std::vector<uint8_t> p1(static_cast<size_t>(kN1) * kNb1 * kBB);
  std::vector<uint8_t> p2(static_cast<size_t>(kN2) * kNb2 * kBB);
  fill_q6k(p1, 20260913u);
  fill_q6k(p2, 20260914u);
  std::vector<uint16_t> a_bf(static_cast<size_t>(kM * kK));
  for (auto& v : a_bf) v = vt::F32ToBF16((static_cast<float>(rng() % 401) - 200.0f) / 100.0f);

  void* mem_a = backend.Alloc(a_bf.size() * sizeof(uint16_t));
  void* mem_w1 = backend.Alloc(p1.size());
  void* mem_w2 = backend.Alloc(p2.size());
  void* mem_o16 = backend.Alloc(static_cast<size_t>(kM * kN1) * sizeof(uint16_t));
  void* mem_o32 = backend.Alloc(static_cast<size_t>(kM * kN1) * sizeof(float));
  void* mem_o2 = backend.Alloc(static_cast<size_t>(kM * kN2) * sizeof(float));
  backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
  backend.Copy(q, mem_w1, p1.data(), p1.size());
  backend.Copy(q, mem_w2, p2.data(), p2.size());
  Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                  Device{vt::DeviceType::kTENSTORRENT, 0},
                                  {kM, kK});
  Tensor w1_t = Tensor::Contiguous(mem_w1, vt::DType::kQ6_K,
                                   Device{vt::DeviceType::kTENSTORRENT, 0},
                                   {kN1, kK});
  Tensor w2_t = Tensor::Contiguous(mem_w2, vt::DType::kQ6_K,
                                   Device{vt::DeviceType::kTENSTORRENT, 0},
                                   {kN2, kN1});
  Tensor o16_t = Tensor::Contiguous(mem_o16, vt::DType::kBF16,
                                    Device{vt::DeviceType::kTENSTORRENT, 0},
                                    {kM, kN1});
  Tensor o32_t = Tensor::Contiguous(mem_o32, vt::DType::kF32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0},
                                    {kM, kN1});
  Tensor o2_t = Tensor::Contiguous(mem_o2, vt::DType::kF32,
                                   Device{vt::DeviceType::kTENSTORRENT, 0},
                                   {kM, kN2});

  // The reference chain, all f32-out (no bf16 slot handoff): m1 ref, cast
  // to bf16, m2 ref, cast to bf16 — what the bf16-out arm MUST produce.
  vt::MatmulBT(q, o32_t, a_t, w1_t);
  std::vector<float> ref32(static_cast<size_t>(kM * kN1), 0.0f);
  backend.Copy(q, ref32.data(), mem_o32, ref32.size() * sizeof(float));
  std::vector<uint16_t> ref16(static_cast<size_t>(kM * kN1));
  for (size_t i = 0; i < ref16.size(); ++i) ref16[i] = vt::F32ToBF16(ref32[i]);
  Tensor ref16_t = Tensor::Contiguous(ref16.data(), vt::DType::kBF16,
                                      Device{vt::DeviceType::kTENSTORRENT, 0},
                                      {kM, kN1});
  // Warm the bf16-out arm EAGERLY once: the int8-dot launch AND the W6
  // f32->bf16 cast op must be in the program cache before the capture (a
  // new binary during capture refuses: mesh_workload.cpp:196).
  vt::MatmulBT(q, o16_t, a_t, w1_t);
  vt::MatmulBT(q, o2_t, ref16_t, w2_t);
  std::vector<float> ref2(static_cast<size_t>(kM * kN2), 0.0f);
  backend.Copy(q, ref2.data(), mem_o2, ref2.size() * sizeof(float));

  // The bf16-out m1 under capture x2 (the #2907 discipline), then the
  // DEVICE consumer: m2 reads the committed bf16 slot as its activation.
  std::vector<uint16_t> dumps[2];
  int64_t demand[2] = {0, 0};
  for (int pass = 0; pass < 2; ++pass) {
    void* graph = nullptr;
    std::string what;
    bool threw = false;
    try {
      backend.BeginCapture(q);
      vt::MatmulBT(q, o16_t, a_t, w1_t);  // bf16 out: the W6 cast fires
      graph = backend.EndCaptureGraph(q);
    } catch (const std::exception& ex) {
      threw = true;
      what = ex.what();
    }
    REQUIRE_MESSAGE(!threw, "capture pass " << pass << " threw: " << what);
    REQUIRE(graph != nullptr);
    demand[pass] = vt::tenstorrent::LastTraceBytesForTest();
    backend.ReplayGraph(q, graph);
    dumps[pass].resize(static_cast<size_t>(kM * kN1), 0);
    backend.Copy(q, dumps[pass].data(), mem_o16,
                 dumps[pass].size() * sizeof(uint16_t));
    backend.DestroyGraph(graph);
    // The device-side consumer probe: m2 over the committed slot.
    vt::MatmulBT(q, o2_t, o16_t, w2_t);
    std::vector<float> got2(static_cast<size_t>(kM * kN2), 0.0f);
    backend.Copy(q, got2.data(), mem_o2, got2.size() * sizeof(float));
    for (size_t i = 0; i < got2.size(); ++i)
      CHECK(std::isfinite(got2[i]));  // the word-halved-garbage signature
  }
  for (int pass = 0; pass < 2; ++pass) {
    REQUIRE_MESSAGE(demand[pass] <= 52428800,
                    "capture pass " << pass << " demanded " << demand[pass]
                                    << " B of trace region against 52428800 B");
    CHECK_MESSAGE(std::memcmp(dumps[static_cast<size_t>(pass)].data(),
                              ref16.data(),
                              ref16.size() * sizeof(uint16_t)) == 0,
                  "replay pass " << pass
                                 << " diverged from the f32-out reference");
  }
  CHECK(std::memcmp(dumps[1].data(), dumps[0].data(),
                    ref16.size() * sizeof(uint16_t)) == 0);
  MESSAGE("bf16-out capture x2 byte-identity + device-consumer probe: PASS");
  backend.Free(mem_a);
  backend.Free(mem_w1);
  backend.Free(mem_w2);
  backend.Free(mem_o16);
  backend.Free(mem_o32);
  backend.Free(mem_o2);
}

// KEEPQUANT W4b (issue #3031) C4 profile, spec ## W4b "Profile first": the
// KEEPQUANT W4b (issue #3031) C4 profile, spec ## W4b "Profile first": the
// packed arm vs the int8-dot lever per call, on ONE build, both through the
// public vt::MatmulBT dispatch — bf16-out for the W4a grouped packed arm, the
// same call f32-out with VT_TT_KEEPQUANT_INT8DOT=1 for the lever. Numbers are
// RECORDED ONLY (the spec floor is "llama.cpp-comparable, recorded only"): no
// performance assertion lives here and the ratio is motivation, never a
// claim. Opt-in like the kGdnDecode step microbench, whose TT_GDN_BENCH flag
// this case shares.
TEST_CASE("kTENSTORRENT keep-quant dense matmul packed-vs-int8dot microbench (opt-in)") {
  if (std::getenv("TT_GDN_BENCH") == nullptr) {
    MESSAGE("SKIPPED: set TT_GDN_BENCH=1 to run the opt-in microbenches "
            "(shared with the kGdnDecode step microbench)");
    return;
  }
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  // Both arms must run in one process on one build, so the gate env flips
  // per arm below; restore the ambient value whatever happens.
  const char* prev_lever = std::getenv("VT_TT_KEEPQUANT_INT8DOT");
  const bool had_lever = prev_lever != nullptr;
  const std::string saved_lever =
      had_lever ? std::string(prev_lever) : std::string();
  struct RestoreLeverEnv {
    const bool had;
    const std::string saved;
    ~RestoreLeverEnv() {
      if (had) ::setenv("VT_TT_KEEPQUANT_INT8DOT", saved.c_str(), 1);
      else ::unsetenv("VT_TT_KEEPQUANT_INT8DOT");
    }
  } restore_lever_env{had_lever, saved_lever};

  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();

  // The int8-dot sweep's packed generator and PRNG style, fixed seed.
  std::mt19937 rng(20260917u);
  auto rand_byte = [&rng]() { return static_cast<uint8_t>(rng() & 0xFF); };
  auto rand_f16_signed = [&rng](float lo, float span) {
    return (lo + span * static_cast<float>(rng() % 64) / 64.0f) *
           ((rng() % 2) != 0 ? 1.0f : -1.0f);
  };

  constexpr int kWarm = 5;
  constexpr int kIters = 200;  // big cells land ~1-2 s per arm; recorded only

  const int64_t m_list[] = {1, 4};
  const int64_t k_list[] = {1024, 4096};
  const int64_t n_list[] = {1024, 4096};
  const vt::DType encodings[] = {vt::DType::kQ4_K, vt::DType::kQ6_K};

  for (const vt::DType enc : encodings) {
    const int64_t bb = vt::BlockBytes(enc);
    const int64_t bel = vt::BlockElems(enc);
    for (int64_t M : m_list) {
      for (int64_t K : k_list) {
        const int64_t nb = K / bel;
        REQUIRE(K % bel == 0);
        for (int64_t N : n_list) {
          std::vector<uint8_t> packed(static_cast<size_t>(N) * nb * bb);
          for (int64_t b = 0; b < N * nb; ++b) {
            uint8_t* blk = packed.data() + static_cast<size_t>(b) * bb;
            if (enc == vt::DType::kQ6_K) {
              for (int i = 0; i < 208; ++i) blk[i] = rand_byte();
              const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
              std::memcpy(blk + 208, &d_bits, sizeof(d_bits));
            } else {
              const uint16_t d_bits = vt::F32ToF16(rand_f16_signed(0.05f, 0.35f));
              const uint16_t dmin_bits = vt::F32ToF16(rand_f16_signed(0.005f, 0.02f));
              std::memcpy(blk + 0, &d_bits, sizeof(d_bits));
              std::memcpy(blk + 2, &dmin_bits, sizeof(dmin_bits));
              for (int i = 0; i < 12; ++i) blk[4 + i] = rand_byte();
              for (int i = 0; i < 128; ++i) blk[16 + i] = rand_byte();
            }
          }
          std::vector<uint16_t> a_bf(static_cast<size_t>(M) * K);
          for (auto& v : a_bf)
            v = vt::F32ToBF16((static_cast<float>(rng() % 401) - 200.0f) / 100.0f);

          void* mem_a = backend.Alloc(a_bf.size() * sizeof(uint16_t));
          void* mem_b = backend.Alloc(packed.size());
          void* mem_ob = backend.Alloc(std::max<size_t>(M * N, 16) * sizeof(uint16_t));
          void* mem_of = backend.Alloc(std::max<size_t>(M * N, 16) * sizeof(float));
          backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
          backend.Copy(q, mem_b, packed.data(), packed.size());
          Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                          Device{vt::DeviceType::kTENSTORRENT, 0},
                                          {M, K});
          Tensor b_t = Tensor::Contiguous(mem_b, enc,
                                          Device{vt::DeviceType::kTENSTORRENT, 0},
                                          {N, K});
          Tensor ob_t = Tensor::Contiguous(mem_ob, vt::DType::kBF16,
                                           Device{vt::DeviceType::kTENSTORRENT, 0},
                                           {M, N});
          Tensor of_t = Tensor::Contiguous(mem_of, vt::DType::kF32,
                                           Device{vt::DeviceType::kTENSTORRENT, 0},
                                           {M, N});

          // One arm: warmup, then a fixed iteration count timed end-to-end,
          // with a 4-byte blocking probe inside the window so the queue drain
          // is included and identical for both arms.
          auto time_arm = [&](Tensor& o_t, void* mem_o, bool lever_on) -> double {
            if (lever_on) ::setenv("VT_TT_KEEPQUANT_INT8DOT", "1", 1);
            else ::unsetenv("VT_TT_KEEPQUANT_INT8DOT");
            for (int i = 0; i < kWarm; ++i) vt::MatmulBT(q, o_t, a_t, b_t);
            uint32_t probe = 0;
            backend.Copy(q, &probe, mem_o, sizeof(probe));  // drain the warmup
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < kIters; ++i) vt::MatmulBT(q, o_t, a_t, b_t);
            backend.Copy(q, &probe, mem_o, sizeof(probe));
            const auto t1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double>(t1 - t0).count() /
                   static_cast<double>(kIters);
          };

          const double s_packed = time_arm(ob_t, mem_ob, /*lever_on=*/false);
          const double s_int8 = time_arm(of_t, mem_of, /*lever_on=*/true);
          MESSAGE("packed-vs-int8dot enc=", static_cast<int>(enc),
                  " M=", M, " K=", K, " N=", N, ": packed ", s_packed,
                  " s/call | int8dot ", s_int8, " s/call | ratio int8/packed ",
                  (s_packed > 0.0 ? s_int8 / s_packed : 0.0));
          CHECK(s_packed > 0.0);
          CHECK(s_int8 > 0.0);

          backend.Free(mem_a);
          backend.Free(mem_b);
          backend.Free(mem_ob);
          backend.Free(mem_of);
        }
      }
    }
  }
  MESSAGE("packed-vs-int8dot microbench: 16 shapes x 2 arms, kIters=", kIters,
          " (recorded only; no performance claim)");
}

TEST_CASE("kTENSTORRENT E=1 grouped keep-quant never reads the routing ids; E=N still range-checks") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuantGrouped,
                           vt::DeviceType::kTENSTORRENT));
  REQUIRE(backend.SupportsGraphCapture());
  Queue q = backend.CreateQueue();
  vt::tenstorrent::KeepQuantChunkRowsOverrideForTest(0);
  struct ChunkReset {
    ~ChunkReset() { vt::tenstorrent::KeepQuantChunkRowsOverrideForTest(0); }
  } chunk_reset;

  // Q4_K, E=1 (dense [N,K]), P=3, Pa=1 broadcast. The ids bytes are GARBAGE
  // (out of range for E=1): the E=1 arm must treat the routing ids as
  // statically all zero — the only in-range expert — so it neither
  // EnsureHosts nor reads them, which is exactly what makes the arm
  // capture-clean. Red-first: with the W2 host readback the eager call
  // below CHECK-fails on "expert id out of range".
  constexpr int64_t kN = 8, kK = 256, kP = 3, kPa = 1;
  const int64_t kBB = vt::BlockBytes(vt::DType::kQ4_K);
  std::mt19937 rng(20260915u);
  std::vector<uint8_t> packed(kN * kBB);
  for (int64_t b = 0; b < kN; ++b) {
    uint8_t* blk = packed.data() + b * kBB;
    const uint16_t d_bits = vt::F32ToF16(0.05f + 0.35f * static_cast<float>(rng() % 64) / 64.0f);
    const uint16_t dmin_bits = vt::F32ToF16(0.005f + 0.02f * static_cast<float>(rng() % 32) / 32.0f);
    std::memcpy(blk, &d_bits, sizeof(d_bits));
    std::memcpy(blk + 2, &dmin_bits, sizeof(dmin_bits));
    for (int i = 4; i < kBB; ++i) blk[i] = static_cast<uint8_t>(rng() & 0xFF);
  }
  std::vector<float> a_f32(static_cast<size_t>(kPa * kK));
  for (auto& v : a_f32) v = (static_cast<float>(rng() % 401) - 200.0f) / 100.0f;
  std::vector<uint16_t> a_bf(a_f32.size());
  for (size_t i = 0; i < a_f32.size(); ++i) a_bf[i] = vt::F32ToBF16(a_f32[i]);
  std::vector<int32_t> garbage_ids{7, -5, 999};  // inert for E=1

  // The oracle: the CPU grouped provider fed ids {0,0,0} (the semantics the
  // E=1 arm must implement), inside the W2 analytic bf16-operand bound.
  std::vector<float> cpu_out(static_cast<size_t>(kP * kN), 0.0f);
  {
    vt::Queue qcpu{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
    Tensor at = Tensor::Contiguous(a_bf.data(), vt::DType::kBF16, qcpu.device,
                                   {kPa, kK});
    Tensor ot = Tensor::Contiguous(cpu_out.data(), vt::DType::kF32, qcpu.device,
                                   {kP, kN});
    std::vector<int32_t> zero_ids(kP, 0);
    Tensor it = Tensor::Contiguous(zero_ids.data(), vt::DType::kI32,
                                   qcpu.device, {kP});
    Tensor wt = Tensor::Contiguous(packed.data(), vt::DType::kF32, qcpu.device,
                                   {kN, kK});
    wt.dtype = vt::DType::kQ4_K;
    vt::MatmulBTQuantGrouped(qcpu, ot, at, wt, it);
  }
  auto widen = [](uint16_t u) {
    uint32_t bits = static_cast<uint32_t>(u) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
  };
  std::vector<float> w_f32(static_cast<size_t>(kN) * kK);
  vt::cpu::BlockToFloat(vt::DType::kQ4_K)(packed.data(), w_f32.data(), kN * kK);
  std::vector<uint16_t> w_bf(w_f32.size());
  for (size_t i = 0; i < w_f32.size(); ++i) w_bf[i] = vt::F32ToBF16(w_f32[i]);
  std::vector<float> bound(static_cast<size_t>(kP * kN), 0.0f);
  for (int64_t n = 0; n < kN; ++n) {
    float acc = 0.0f, mag = 0.0f;
    for (int64_t k = 0; k < kK; ++k) {
      const float prod = widen(a_bf[static_cast<size_t>(k)]) *
                         widen(w_bf[static_cast<size_t>(n * kK + k)]);
      acc += prod;
      mag += std::fabs(prod);
    }
    for (int64_t p = 0; p < kP; ++p)
      bound[static_cast<size_t>(p * kN + n)] =
          1.05f * std::ldexp(1.0f, -8) * (mag + std::fabs(acc));
  }

  void* mem_a = backend.Alloc(a_bf.size() * sizeof(uint16_t));
  void* mem_w = backend.Alloc(packed.size());
  void* mem_o = backend.Alloc(static_cast<size_t>(kP * kN) * sizeof(float));
  void* mem_i = backend.Alloc(garbage_ids.size() * sizeof(int32_t));
  backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
  backend.Copy(q, mem_w, packed.data(), packed.size());
  backend.Copy(q, mem_i, garbage_ids.data(), garbage_ids.size() * sizeof(int32_t));
  Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                  Device{vt::DeviceType::kTENSTORRENT, 0},
                                  {kPa, kK});
  Tensor w_t = Tensor::Contiguous(mem_w, vt::DType::kQ4_K,
                                  Device{vt::DeviceType::kTENSTORRENT, 0},
                                  {kN, kK});
  Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                  Device{vt::DeviceType::kTENSTORRENT, 0},
                                  {kP, kN});
  Tensor i_t = Tensor::Contiguous(mem_i, vt::DType::kI32,
                                  Device{vt::DeviceType::kTENSTORRENT, 0}, {kP});
  vt::MatmulBTQuantGrouped(q, o_t, a_t, w_t, i_t);  // must not read the ids
  std::vector<float> eager(static_cast<size_t>(kP * kN), 0.0f);
  backend.Copy(q, eager.data(), mem_o, eager.size() * sizeof(float));
  for (int64_t i = 0; i < kP * kN; ++i) {
    const float d = std::fabs(eager[static_cast<size_t>(i)] -
                              cpu_out[static_cast<size_t>(i)]);
    CHECK_MESSAGE(d <= bound[static_cast<size_t>(i)],
                  "E=1 ids-inert output diverges at " << i << ": tt="
                                                      << eager[static_cast<size_t>(i)]
                                                      << " cpu="
                                                      << cpu_out[static_cast<size_t>(i)]);
  }

  // The same op CAPTURED: no readback may exist to run inside the region.
  {
    backend.BeginCapture(q);
    vt::MatmulBTQuantGrouped(q, o_t, a_t, w_t, i_t);
    backend.EndCapture(q);
    backend.Replay(q);
    std::vector<float> after(static_cast<size_t>(kP * kN), 0.0f);
    backend.Copy(q, after.data(), mem_o, after.size() * sizeof(float));
    CHECK(std::memcmp(after.data(), eager.data(), eager.size() * sizeof(float)) == 0);
  }

  // NO WEAKENING on the tower arm: E=N ids are dynamic and stay range-
  // checked on the host (its capture indirection is a later wave).
  {
    constexpr int64_t kE2 = 4;
    std::vector<uint8_t> tower(kE2 * kN * kBB);
    for (size_t b = 0; b < tower.size() / static_cast<size_t>(kBB); ++b) {
      uint8_t* blk = tower.data() + b * kBB;
      const uint16_t d_bits = vt::F32ToF16(0.05f);
      const uint16_t dmin_bits = vt::F32ToF16(0.005f);
      std::memcpy(blk, &d_bits, sizeof(d_bits));
      std::memcpy(blk + 2, &dmin_bits, sizeof(dmin_bits));
      for (int i = 4; i < kBB; ++i) blk[i] = static_cast<uint8_t>(rng() & 0xFF);
    }
    void* mem_w2 = backend.Alloc(tower.size());
    backend.Copy(q, mem_w2, tower.data(), tower.size());
    Tensor w2 = Tensor::Contiguous(mem_w2, vt::DType::kQ4_K,
                                   Device{vt::DeviceType::kTENSTORRENT, 0},
                                   {kE2 * kN, kK});
    std::vector<int32_t> bad_ids{99, 0, 0};
    void* mem_i2 = backend.Alloc(bad_ids.size() * sizeof(int32_t));
    backend.Copy(q, mem_i2, bad_ids.data(), bad_ids.size() * sizeof(int32_t));
    Tensor i2 = Tensor::Contiguous(mem_i2, vt::DType::kI32,
                                   Device{vt::DeviceType::kTENSTORRENT, 0}, {kP});
    bool threw = false;
    std::string what;
    try {
      vt::MatmulBTQuantGrouped(q, o_t, a_t, w2, i2);
    } catch (const std::exception& ex) {
      threw = true;
      what = ex.what();
    }
    CHECK_MESSAGE(threw, "E=N out-of-range id must still be refused");
    CHECK_MESSAGE(what.find("out of range") != std::string::npos,
                  "the refusal must name the range check, got: ", what);
    backend.Free(mem_w2);
    backend.Free(mem_i2);
  }

  backend.Free(mem_a);
  backend.Free(mem_w);
  backend.Free(mem_o);
  backend.Free(mem_i);
}

TEST_CASE("kTENSTORRENT E=1 grouped keep-quant chunked slice-decode is bit-exact vs BlockToFloat") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuantGrouped,
                           vt::DeviceType::kTENSTORRENT));
  Queue q = backend.CreateQueue();

  // One-hot activations on a bf16-exact dequant grid, the wave-2 legs'
  // construction at E=1, with the chunk rows FORCED below the N of the case
  // so every call executes MANY chunks. Chunks cover DISJOINT weight rows,
  // so each output element is one chunk's dot over the full K and the
  // expected bits are exactly bf16(BlockToFloat(element)) — a chunk offset
  // or loop-bound defect lands on a different grid value and cannot hide.
  // (Pre-chunking this case is trivially green — the whole slice is one
  // "chunk"; it pins the chunked loop bounds that replace it.)
  auto one_hot_leg = [&](vt::DType enc, int64_t chunk_rows, int64_t nb) {
    vt::tenstorrent::KeepQuantChunkRowsOverrideForTest(chunk_rows);
    struct ChunkReset {
      ~ChunkReset() { vt::tenstorrent::KeepQuantChunkRowsOverrideForTest(0); }
    } chunk_reset;
    constexpr int64_t kN = 8, kP = 5;
    const int64_t elems = vt::BlockElems(enc);
    const int64_t bb = vt::BlockBytes(enc);
    const int64_t kK = nb * elems;
    const int64_t chunks = (kN + chunk_rows - 1) / chunk_rows;
    std::mt19937 rng(static_cast<uint32_t>(20260916u + nb));
    std::vector<uint8_t> packed(kN * nb * bb);
    const uint16_t d_bits = vt::F32ToF16(std::ldexp(1.0f, -6));
    for (int64_t b = 0; b < kN * nb; ++b) {
      uint8_t* blk = packed.data() + b * bb;
      if (enc == vt::DType::kQ8_0) {
        std::memcpy(blk, &d_bits, sizeof(d_bits));
        for (int i = 0; i < 32; ++i) {
          blk[2 + i] = static_cast<uint8_t>(rng() % 255 - 127);
        }
      } else {
        // Q6_K: ql[128] @0, qh[64] @128, i8 scales[16] @192, d f16 @208 —
        // positive scales keep the dequant in the -0-free class.
        for (int i = 0; i < 128; ++i) blk[i] = static_cast<uint8_t>(rng() & 0xFF);
        for (int i = 128; i < 192; ++i) blk[i] = static_cast<uint8_t>(rng() & 0xFF);
        for (int i = 192; i < 208; ++i)
          blk[i] = static_cast<uint8_t>(1 + rng() % 127);
        std::memcpy(blk + 208, &d_bits, sizeof(d_bits));
      }
    }
    std::vector<uint16_t> a_bf(static_cast<size_t>(kP * kK), 0u);
    for (int64_t p = 0; p < kP; ++p)
      a_bf[static_cast<size_t>(p * kK + (p % kK))] = vt::F32ToBF16(1.0f);
    std::vector<int32_t> ids(kP, 0);  // E=1: statically all zero

    // The decode oracle: BlockToFloat over the IDENTICAL bytes, expert 0.
    std::vector<float> w_f32(static_cast<size_t>(kN) * kK);
    vt::cpu::BlockToFloat(enc)(packed.data(), w_f32.data(), kN * kK);
    std::vector<float> dequant(static_cast<size_t>(kP * kN));
    for (int64_t p = 0; p < kP; ++p)
      for (int64_t n = 0; n < kN; ++n)
        dequant[static_cast<size_t>(p * kN + n)] =
            w_f32[static_cast<size_t>(n * kK + (p % kK))];

    void* mem_a = backend.Alloc(a_bf.size() * sizeof(uint16_t));
    void* mem_w = backend.Alloc(packed.size());
    void* mem_o = backend.Alloc(static_cast<size_t>(kP * kN) * sizeof(float));
    void* mem_i = backend.Alloc(ids.size() * sizeof(int32_t));
    backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
    backend.Copy(q, mem_w, packed.data(), packed.size());
    backend.Copy(q, mem_i, ids.data(), ids.size() * sizeof(int32_t));
    Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                    Device{vt::DeviceType::kTENSTORRENT, 0},
                                    {kP, kK});
    Tensor w_t = Tensor::Contiguous(mem_w, enc,
                                    Device{vt::DeviceType::kTENSTORRENT, 0},
                                    {kN, kK});
    Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0},
                                    {kP, kN});
    Tensor i_t = Tensor::Contiguous(mem_i, vt::DType::kI32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {kP});
    vt::MatmulBTQuantGrouped(q, o_t, a_t, w_t, i_t);
    std::vector<float> tt_out(static_cast<size_t>(kP * kN), 0.0f);
    backend.Copy(q, tt_out.data(), mem_o, tt_out.size() * sizeof(float));
    backend.Free(mem_a);
    backend.Free(mem_w);
    backend.Free(mem_o);
    backend.Free(mem_i);
    for (int64_t i = 0; i < kP * kN; ++i) {
      const uint16_t got_bf = vt::F32ToBF16(tt_out[static_cast<size_t>(i)]);
      const uint16_t want_bf =
          vt::F32ToBF16(dequant[static_cast<size_t>(i)]);
      CHECK_MESSAGE(got_bf == want_bf,
                    "chunked slice-decode bit-exact (" << vt::Name(enc)
                                                       << ", chunk="
                                                       << chunk_rows
                                                       << ", chunks="
                                                       << chunks
                                                       << "): i=" << i
                                                       << " tt=" << tt_out[static_cast<size_t>(i)]
                                                       << " dequant="
                                                       << dequant[static_cast<size_t>(i)]);
    }
    MESSAGE("chunked bit leg ", vt::Name(enc), " chunk=", chunk_rows,
            " chunks=", chunks, ": bits match BlockToFloat");
  };
  one_hot_leg(vt::DType::kQ8_0, 2, 2);   // K=64,  4 chunks of 2 rows
  one_hot_leg(vt::DType::kQ6_K, 3, 1);   // K=256, 3 chunks (last of 2)
}

// W4a wave-3b-1 (#3030): THE RESIDENCY SWITCH. The dense keep-quant matmul
// routes through the wave-3a chunked E=1 grouped arm — the PACKED words stage
// once per weight (EnsureKeepQuantWords) and every call chunk-decodes +
// accumulates from them — so the decoded bf16 TWIN is gone from the matmul
// path. RED-first at d614aa4f3: the twin build made the twin probe read TRUE
// and the word probe FALSE (the twin path decoded host-side and staged no
// words); the same warm-matmul pair must flip both after the switch. All four
// registered encodings, entering through vt::MatmulBT's public dispatch (the
// entry a GGUF load actually takes, ops.cpp:163). The gather class keeps its
// own embed-table twin — that survivor leg is the next case.
TEST_CASE("kTENSTORRENT dense keep-quant matmul stages PACKED words and builds NO twin (all four encodings)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuant, vt::DeviceType::kTENSTORRENT));

  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();

  constexpr int64_t kN = 8;   // weight rows
  constexpr int64_t kNb = 1;  // one block per row -> K spans a single block
  std::mt19937 rng(20260921u);
  auto rand_byte = [&rng]() { return static_cast<uint8_t>(rng() & 0xFF); };
  auto put_f16 = [](uint8_t* blk, int64_t off, float v) {
    const uint16_t bits = vt::F32ToF16(v);
    std::memcpy(blk + off, &bits, sizeof(bits));
  };

  const vt::DType encodings[] = {vt::DType::kQ4_K, vt::DType::kQ5_K,
                                 vt::DType::kQ6_K, vt::DType::kQ8_0};
  for (const vt::DType enc : encodings) {
    const int64_t bb = vt::BlockBytes(enc);
    const int64_t be = vt::BlockElems(enc);
    REQUIRE(be * bb > 0);
    const int64_t K = kNb * be;

    // Deterministic packed blocks (the W3 sweep generator): PRNG bytes
    // everywhere, then finite f16 scales (d, and dmin where the encoding has
    // one) placed at the layout's offsets, so the decode never sees NaN/Inf.
    std::vector<uint8_t> packed(kN * kNb * bb);
    for (int64_t b = 0; b < kN * kNb; ++b) {
      uint8_t* blk = packed.data() + b * bb;
      for (int i = 0; i < bb; ++i) blk[i] = rand_byte();
      const float d = (0.05f + 0.35f * static_cast<float>(rng() % 64) / 64.0f) *
                      ((rng() % 2) != 0 ? 1.0f : -1.0f);
      if (enc == vt::DType::kQ6_K) {
        put_f16(blk, 208, d);
      } else {
        put_f16(blk, 0, d);
        if (enc == vt::DType::kQ4_K || enc == vt::DType::kQ5_K)
          put_f16(blk, 2, 0.005f + 0.02f * static_cast<float>(rng() % 32) / 32.0f);
      }
    }
    std::vector<uint16_t> a_bf(K);
    for (auto& v : a_bf)
      v = vt::F32ToBF16((static_cast<float>(rng() % 401) - 200.0f) / 100.0f);

    // Envelope oracle (the W3 dot's shape, M=1): decode -> ONE bf16 RNE ->
    // f32 ascending accumulation; 1.05 * 2^-8 bf16-operand bound.
    std::vector<float> w_f32(kN * K);
    vt::cpu::BlockToFloat(enc)(packed.data(), w_f32.data(), kN * K);
    auto widen = [](uint16_t u) {
      uint32_t bits = static_cast<uint32_t>(u) << 16;
      float f;
      std::memcpy(&f, &bits, 4);
      return f;
    };
    std::vector<float> ref(kN), bound(kN);
    for (int64_t n = 0; n < kN; ++n) {
      float acc = 0.0f, mag = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        const float p = widen(a_bf[static_cast<size_t>(k)]) *
                        widen(vt::F32ToBF16(w_f32[static_cast<size_t>(n) * K + k]));
        acc += p;
        mag += std::fabs(p);
      }
      ref[static_cast<size_t>(n)] = acc;
      bound[static_cast<size_t>(n)] = 1.05f * std::ldexp(1.0f, -8) * (mag + std::fabs(acc));
    }

    void* mem_a = backend.Alloc(a_bf.size() * sizeof(uint16_t));
    void* mem_b = backend.Alloc(packed.size());
    void* mem_o = backend.Alloc(kN * sizeof(float));
    backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
    backend.Copy(q, mem_b, packed.data(), packed.size());
    Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {1, K});
    Tensor b_t = Tensor::Contiguous(mem_b, enc,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {kN, K});
    Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {1, kN});

    // The warm pair: an eager warm matmul + a second call. A twin, if the
    // policy still built one, exists after call one; the second call proves
    // no lazy build either. Then the residency probes read the policy.
    vt::MatmulBT(q, o_t, a_t, b_t);
    vt::MatmulBT(q, o_t, a_t, b_t);

    CHECK_FALSE_MESSAGE(
        vt::tenstorrent::DecodedWeightShadowPresentForTest(b_t.data),
        "TWIN-ABSENCE (" << vt::Name(enc) << "): the dense keep-quant matmul "
                         "built a decoded bf16 twin — the wave-2 twin "
                         "residency survived the wave-3b-1 switch");
    REQUIRE_MESSAGE(
        vt::tenstorrent::KeepQuantWordShadowPresentForTest(b_t.data),
        "PACKED-WORDS (" << vt::Name(enc) << "): the dense keep-quant matmul "
                         "staged no resident i32 word shadow — the chunked "
                         "E=1 arm's residency did not engage");

    std::vector<float> out(kN, 0.0f);
    backend.Copy(q, out.data(), mem_o, out.size() * sizeof(float));
    backend.Free(mem_a);
    backend.Free(mem_b);
    backend.Free(mem_o);
    for (int64_t n = 0; n < kN; ++n) {
      const float diff = std::fabs(out[static_cast<size_t>(n)] -
                                   ref[static_cast<size_t>(n)]);
      CHECK(std::isfinite(out[static_cast<size_t>(n)]));
      CHECK_MESSAGE(diff <= bound[static_cast<size_t>(n)],
                    vt::Name(enc) << " out[" << n <<"]=" << out[n]
                                  << " ref=" << ref[n]
                                  << " bound=" << bound[n]);
    }
    MESSAGE("dense keep-quant residency switch ", vt::Name(enc),
            ": twin absent, words staged, envelope ok");
  }
}

// W4a wave-3b-1 (#3030) survivor leg: the GATHER class keeps its twin. The
// embedding table's bf16 device twin (EnsureEmbedTableDevice -> the
// EmbedTableShadows map) is the residency the twin policy deliberately keeps;
// the switch must not have over-removed it. Mutation guard for the same
// commit's switch: delete the embed map's insert and this stays red.
TEST_CASE("kTENSTORRENT embedding gather twin survives the wave-3b-1 switch (gather class)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kEmbedding, vt::DeviceType::kTENSTORRENT));

  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();

  constexpr int64_t Vocab = 17, H = 24, T = 3;
  std::vector<float> host_table(Vocab * H);
  for (size_t i = 0; i < host_table.size(); ++i)
    host_table[i] = static_cast<float>(i % 13) * 0.1f - 0.5f;
  const std::vector<int32_t> host_ids = {0, 16, 3};

  std::vector<float> host_out(T * H, 0.0f);
  void* mem_table = backend.Alloc(host_table.size() * sizeof(float));
  void* mem_ids = backend.Alloc(host_ids.size() * sizeof(int32_t));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  backend.Copy(q, mem_table, host_table.data(), host_table.size() * sizeof(float));
  backend.Copy(q, mem_ids, host_ids.data(), host_ids.size() * sizeof(int32_t));

  Tensor table = Tensor::Contiguous(mem_table, vt::DType::kF32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {Vocab, H});
  Tensor ids = Tensor::Contiguous(mem_ids, vt::DType::kI32,
                                  Device{vt::DeviceType::kTENSTORRENT, 0}, {T});
  Tensor out = Tensor::Contiguous(mem_out, vt::DType::kF32,
                                  Device{vt::DeviceType::kTENSTORRENT, 0}, {T, H});

  auto embedding = reinterpret_cast<vt::EmbeddingFn>(
      vt::GetOp(vt::OpId::kEmbedding, vt::DeviceType::kTENSTORRENT));
  embedding(q, out, table, ids);
  embedding(q, out, table, ids);  // warm + second call, the twin pair

  REQUIRE_MESSAGE(
      vt::tenstorrent::EmbedTableShadowPresentForTest(table.data),
      "GATHER-SURVIVOR: the embedding table's device twin is gone after a "
      "warm gather pair — the switch over-removed the gather class");
  REQUIRE_FALSE_MESSAGE(
      vt::tenstorrent::DecodedWeightShadowPresentForTest(table.data),
      "the gather must serve from the embed-table map, never from the "
      "matmul twin map");

  backend.Free(mem_table);
  backend.Free(mem_ids);
  backend.Free(mem_out);
  MESSAGE("embedding gather twin survivor leg: embed twin present, matmul twin map clean");
}

// W4a wave-3b-1 (#3030) decode-shape leg: the ROW_MAJOR chained activation.
// The vehicle's MLP down projection consumes the silu-mul output, whose slot
// carries a ROW_MAJOR device staging (CommitDeviceLogical2D binds whatever
// layout the chain produced), and EnsureDevice2D's exact-shape hit hands that
// ROW_MAJOR tensor straight to the matmul. A ROW_MAJOR [M<32, K] operand
// drives ttnn's auto program config to per_core_M = M / 32 == 0
// (matmul_program_config.cpp get_mcast_1d_config) — the TT_FATAL the vehicle
// AFTER leg hit on the first non-tile-aligned decode shape. RED-first: the
// chain below fatals while the E=1 arm passes the activation through
// untouched; the arm's TILE conversion makes it green and pins the fix.
// Chained through public calls only: an E=1 grouped matmul produces the
// bf16 activation slot state, then vt::MatmulBT consumes it.
TEST_CASE("kTENSTORRENT dense keep-quant matmul converts a ROW_MAJOR chained activation to TILE (decode shape)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuant, vt::DeviceType::kTENSTORRENT));

  Backend& backend = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();

  constexpr int64_t kM = 5;   // the vehicle's decode M: < 32, non-tile-aligned
  constexpr int64_t kK = 256; // one Q6_K block per row
  std::mt19937 rng(20260922u);
  auto rand_byte = [&rng]() { return static_cast<uint8_t>(rng() & 0xFF); };
  auto put_f16 = [](uint8_t* blk, int64_t off, float v) {
    const uint16_t bits = vt::F32ToF16(v);
    std::memcpy(blk + off, &bits, sizeof(bits));
  };
  auto pack_q6 = [&rand_byte, &put_f16, &rng](int64_t rows) {
    std::vector<uint8_t> packed(rows * vt::BlockBytes(vt::DType::kQ6_K));
    for (int64_t b = 0; b < rows; ++b) {
      uint8_t* blk = packed.data() + b * vt::BlockBytes(vt::DType::kQ6_K);
      for (int i = 0; i < vt::BlockBytes(vt::DType::kQ6_K); ++i) blk[i] = rand_byte();
      const float d = (0.05f + 0.35f * static_cast<float>(rng() % 64) / 64.0f) *
                      ((rng() % 2) != 0 ? 1.0f : -1.0f);
      put_f16(blk, 208, d);
    }
    return packed;
  };
  auto widen = [](uint16_t u) {
    uint32_t bits = static_cast<uint32_t>(u) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
  };

  // Producer: [kM, kK] bf16 activation against a [kK, kK] Q6_K weight. The
  // E=1 arm's assembly commits ROW_MAJOR f32 -> typecast bf16, so out1's slot
  // ends ROW_MAJOR — the state the vehicle's elementwise chain leaves.
  const std::vector<uint8_t> w1 = pack_q6(kK);
  std::vector<uint16_t> a1(kM * kK);
  for (auto& v : a1)
    v = vt::F32ToBF16((static_cast<float>(rng() % 401) - 200.0f) / 100.0f);
  void* mem_a1 = backend.Alloc(a1.size() * sizeof(uint16_t));
  void* mem_w1 = backend.Alloc(w1.size());
  void* mem_o1 = backend.Alloc(static_cast<size_t>(kM * kK) * sizeof(uint16_t));
  backend.Copy(q, mem_a1, a1.data(), a1.size() * sizeof(uint16_t));
  backend.Copy(q, mem_w1, w1.data(), w1.size());
  Tensor a1_t = Tensor::Contiguous(mem_a1, vt::DType::kBF16,
                                   Device{vt::DeviceType::kTENSTORRENT, 0}, {kM, kK});
  Tensor w1_t = Tensor::Contiguous(mem_w1, vt::DType::kQ6_K,
                                   Device{vt::DeviceType::kTENSTORRENT, 0}, {kK, kK});
  Tensor o1_t = Tensor::Contiguous(mem_o1, vt::DType::kBF16,
                                   Device{vt::DeviceType::kTENSTORRENT, 0}, {kM, kK});
  vt::MatmulBT(q, o1_t, a1_t, w1_t);
  vt::MatmulBT(q, o1_t, a1_t, w1_t);  // warm pair; slot state is what matters

  // Consumer: out1 (bf16 [kM, kK], ROW_MAJOR slot) feeds a second keep-quant
  // matmul — the vehicle's down-projection shape. Pre-fix this call fatals
  // inside ttnn's program-config search.
  const std::vector<uint8_t> w2 = pack_q6(8);
  void* mem_w2 = backend.Alloc(w2.size());
  void* mem_o2 = backend.Alloc(static_cast<size_t>(kM * 8) * sizeof(float));
  backend.Copy(q, mem_w2, w2.data(), w2.size());
  Tensor w2_t = Tensor::Contiguous(mem_w2, vt::DType::kQ6_K,
                                   Device{vt::DeviceType::kTENSTORRENT, 0}, {8, kK});
  Tensor o2_t = Tensor::Contiguous(mem_o2, vt::DType::kF32,
                                   Device{vt::DeviceType::kTENSTORRENT, 0}, {kM, 8});
  vt::MatmulBT(q, o2_t, o1_t, w2_t);
  vt::MatmulBT(q, o2_t, o1_t, w2_t);  // second call: no lazy rebuild either

  std::vector<uint16_t> o1_bits(kM * kK, 0);
  backend.Copy(q, o1_bits.data(), mem_o1, o1_bits.size() * sizeof(uint16_t));
  std::vector<float> out2(kM * 8, 0.0f);
  backend.Copy(q, out2.data(), mem_o2, out2.size() * sizeof(float));
  backend.Free(mem_a1);
  backend.Free(mem_w1);
  backend.Free(mem_o1);
  backend.Free(mem_w2);
  backend.Free(mem_o2);

  // Envelope oracle for the chained call: the activation operand is EXACTLY
  // the bf16 bits call one committed (read back above), one more bf16 RNE on
  // the w2 decode, f32 ascending accumulation; 1.05 * 2^-8 bound.
  std::vector<float> w2_f32(8 * kK);
  vt::cpu::BlockToFloat(vt::DType::kQ6_K)(w2.data(), w2_f32.data(), 8 * kK);
  for (int64_t m = 0; m < kM; ++m) {
    for (int64_t n = 0; n < 8; ++n) {
      float acc = 0.0f, mag = 0.0f;
      for (int64_t k = 0; k < kK; ++k) {
        const float p =
            widen(o1_bits[static_cast<size_t>(m) * kK + k]) *
            widen(vt::F32ToBF16(w2_f32[static_cast<size_t>(n) * kK + k]));
        acc += p;
        mag += std::fabs(p);
      }
      const float bound = 1.05f * std::ldexp(1.0f, -8) * (mag + std::fabs(acc));
      const float got = out2[static_cast<size_t>(m) * 8 + n];
      CHECK(std::isfinite(got));
      CHECK_MESSAGE(std::fabs(got - acc) <= bound,
                    "ROW-MAJOR chained activation: m=" << m << " n=" << n
                                                       << " got=" << got
                                                       << " ref=" << acc
                                                       << " bound=" << bound);
    }
  }
  MESSAGE("ROW_MAJOR chained activation leg: TILE conversion engaged, envelope ok");
}

// W4d W0 (#3042): the allocation-trace tool's red-first observables.
//
// The trace is gated by VT_TT_ALLOC_TRACE (read live per call, never cached):
// without the env the snapshots are no-ops and the count stays zero even after
// a keep-quant matmul. With the env set, the count goes positive after a matmul
// (EnsureKeepQuantWords + the grouped chunk loop both interleave snapshots) and
// the max-allocation-delta goes positive — proving GetMemoryView ran between
// real device allocations. The env gate is the red-first cut: a missing or
// broken gate would record snapshots unconditionally and fail Phase 1.
TEST_CASE("kTENSTORRENT alloc-trace: zero without env, positive with it (#3042)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuant, vt::DeviceType::kTENSTORRENT));

  const char* const prev = std::getenv("VT_TT_ALLOC_TRACE");
  const bool had = prev != nullptr;
  const std::string saved = had ? std::string(prev) : std::string();

  Backend& backend = *vt::TryGetBackend(DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();

  constexpr int64_t M = 4, N = 8;
  const int64_t kBlockBytes = vt::BlockBytes(vt::DType::kQ4_K);
  const int64_t kBlockElems = vt::BlockElems(vt::DType::kQ4_K);
  const int64_t K = 2 * kBlockElems;

  std::mt19937 rng(20260911u);
  auto make_weight = [&]() {
    std::vector<uint8_t> packed(static_cast<size_t>(N * 2 * kBlockBytes));
    for (int64_t b = 0; b < N * 2; ++b) {
      uint8_t* blk = packed.data() + b * kBlockBytes;
      const uint16_t d = vt::F32ToF16(
          0.05f + 0.35f * static_cast<float>(rng() % 64) / 64.0f);
      std::memcpy(blk + 0, &d, sizeof(d));
      const uint16_t ls = vt::F32ToF16(
          0.005f + 0.02f * static_cast<float>(rng() % 32) / 32.0f);
      std::memcpy(blk + 2, &ls, sizeof(ls));
      for (int i = 0; i < 12; ++i) blk[4 + i] = static_cast<uint8_t>(rng() & 0xFF);
      for (int i = 0; i < 128; ++i) blk[16 + i] = static_cast<uint8_t>(rng() & 0xFF);
    }
    return packed;
  };
  auto make_activation = [&]() {
    std::vector<uint16_t> a_bf(static_cast<size_t>(M * K));
    for (auto& v : a_bf)
      v = vt::F32ToBF16((static_cast<float>(rng() % 401) - 200.0f) / 100.0f);
    return a_bf;
  };

  auto run_matmul = [&](std::vector<uint8_t>& packed,
                        std::vector<uint16_t>& a_bf) {
    void* mem_a = backend.Alloc(M * K * sizeof(uint16_t));
    void* mem_b = backend.Alloc(packed.size());
    void* mem_o = backend.Alloc(M * N * sizeof(float));
    backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
    backend.Copy(q, mem_b, packed.data(), packed.size());
    Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {M, K});
    Tensor b_t = Tensor::Contiguous(mem_b, vt::DType::kQ4_K,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {N, K});
    Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {M, N});
    vt::MatmulBT(q, o_t, a_t, b_t);
    std::vector<float> out(static_cast<size_t>(M * N), 0.0f);
    backend.Copy(q, out.data(), mem_o, out.size() * sizeof(float));
    backend.Free(mem_a);
    backend.Free(mem_b);
    backend.Free(mem_o);
  };

  // --- Phase 1: trace is OFF (env unset) ---
  ::unsetenv("VT_TT_ALLOC_TRACE");
  vt::tenstorrent::ResetAllocTraceForTest();
  CHECK(vt::tenstorrent::AllocTraceSnapshotCountForTest() == 0);
  CHECK(vt::tenstorrent::AllocTraceMaxDeltaForTest() == 0);
  {
    auto packed = make_weight();
    auto a_bf = make_activation();
    run_matmul(packed, a_bf);
  }
  CHECK_MESSAGE(vt::tenstorrent::AllocTraceSnapshotCountForTest() == 0,
                "trace recorded snapshots with VT_TT_ALLOC_TRACE unset");

  // --- Phase 2: trace is ON (env set) ---
  ::setenv("VT_TT_ALLOC_TRACE", "1", 1);
  vt::tenstorrent::ResetAllocTraceForTest();
  CHECK(vt::tenstorrent::AllocTraceSnapshotCountForTest() == 0);
  CHECK(vt::tenstorrent::AllocTraceMaxDeltaForTest() == 0);
  {
    auto packed = make_weight();
    auto a_bf = make_activation();
    run_matmul(packed, a_bf);
  }
  CHECK_MESSAGE(vt::tenstorrent::AllocTraceSnapshotCountForTest() > 0,
                "trace recorded no snapshots with VT_TT_ALLOC_TRACE set");
  CHECK_MESSAGE(vt::tenstorrent::AllocTraceMaxDeltaForTest() > 0,
                "max allocation delta stayed zero after a matmul with trace on");

  if (had) {
    ::setenv("VT_TT_ALLOC_TRACE", saved.c_str(), 1);
  } else {
    ::unsetenv("VT_TT_ALLOC_TRACE");
  }
  vt::tenstorrent::ResetAllocTraceForTest();
}

// W4d W2 (#3042) red-first: the chunked keep-quant decode's f32 planes must
// return to the allocator once the call ends. The W1 trace (2026-09-11, 601
// snapshots) proved they do not — the eager dispatch holds TensorAttributes
// refs past every plane's scope exit, ttnn::Tensor::~Tensor()'s
// use_count()==1 gate skips the free, and 4 wide-weight first decodes
// orphaned 7.31 GB with alloc_per_bank never decreasing in 600 transitions.
// This test books the same shape on a small scale: a 2-chunk Q4_K weight
// whose decode planes sum to several GB, one call, then every host-side
// surface dropped. The reclaim bar is 256 MiB; the pre-fix orphan set is
// gigabytes — the gap between the two is the leak this wave closes.
TEST_CASE("kTENSTORRENT keep-quant decode planes return to the allocator (#3042)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuant, vt::DeviceType::kTENSTORRENT));

  // The reclaim path this wave adds runs in EAGER mode only; the int8-dot
  // env would route the f32-out dense arm away from the chunk decode, and
  // the alloc trace would spam stderr behind the numbers under test.
  const char* const trace_prev = std::getenv("VT_TT_ALLOC_TRACE");
  const bool trace_had = trace_prev != nullptr;
  const std::string trace_saved = trace_had ? std::string(trace_prev) : std::string();
  const char* const int8dot_prev = std::getenv("VT_TT_KEEPQUANT_INT8DOT");
  const bool int8dot_had = int8dot_prev != nullptr;
  const std::string int8dot_saved = int8dot_had ? std::string(int8dot_prev) : std::string();
  ::unsetenv("VT_TT_KEEPQUANT_INT8DOT");
  ::unsetenv("VT_TT_ALLOC_TRACE");

  Backend& backend = *vt::TryGetBackend(DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();

  // Shape: plane budget 256 MiB / (K*4) = 8192 rows per chunk against
  // N=16384 -> 2 chunk iterations, each decoding ~256 MiB f32 planes through
  // the Q4_K chain (the W1 leak shape, scaled to fit the box with room to
  // fail red without OOM-aborting the binary).
  constexpr int64_t M = 1, N = 16384;
  const int64_t kBlockBytes = vt::BlockBytes(vt::DType::kQ4_K);
  const int64_t kBlockElems = vt::BlockElems(vt::DType::kQ4_K);
  const int64_t K = 256 * kBlockElems;  // 8192; 256 blocks per row

  std::mt19937 rng(20260911u);
  std::vector<uint8_t> packed(static_cast<size_t>(N * (K / kBlockElems) * kBlockBytes));
  for (int64_t b = 0; b < N * (K / kBlockElems); ++b) {
    uint8_t* blk = packed.data() + b * kBlockBytes;
    const uint16_t d =
        vt::F32ToF16(0.05f + 0.35f * static_cast<float>(rng() % 64) / 64.0f);
    std::memcpy(blk + 0, &d, sizeof(d));
    const uint16_t ls =
        vt::F32ToF16(0.005f + 0.02f * static_cast<float>(rng() % 32) / 32.0f);
    std::memcpy(blk + 2, &ls, sizeof(ls));
    for (int i = 0; i < 12; ++i) blk[4 + i] = static_cast<uint8_t>(rng() & 0xFF);
    for (int i = 0; i < 128; ++i) blk[16 + i] = static_cast<uint8_t>(rng() & 0xFF);
  }
  std::vector<uint16_t> a_bf(static_cast<size_t>(M * K));
  for (auto& v : a_bf)
    v = vt::F32ToBF16((static_cast<float>(rng() % 401) - 200.0f) / 100.0f);

  auto run_once = [&]() {
    void* mem_a = backend.Alloc(M * K * sizeof(uint16_t));
    void* mem_b = backend.Alloc(packed.size());
    void* mem_o = backend.Alloc(M * N * sizeof(float));
    backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
    backend.Copy(q, mem_b, packed.data(), packed.size());
    Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {M, K});
    Tensor b_t = Tensor::Contiguous(mem_b, vt::DType::kQ4_K,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {N, K});
    Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {M, N});
    vt::MatmulBT(q, o_t, a_t, b_t);
    std::vector<float> out(static_cast<size_t>(M * N), 0.0f);
    backend.Copy(q, out.data(), mem_o, out.size() * sizeof(float));
    // UnregisterHostBuffer drops the word shadow, the output slot twin and
    // the activation slot with the host buffers — everything the row owns
    // BY DESIGN. Whatever free bytes are still missing afterwards are the
    // decode planes, which no Free path can reach.
    backend.Free(mem_a);
    backend.Free(mem_b);
    backend.Free(mem_o);
  };

  // Warm-up: a small matmul settles the device/JIT init allocations so the
  // baseline below measures residency, not first-touch.
  {
    constexpr int64_t sM = 4, sN = 8;
    const int64_t sK = 2 * kBlockElems;
    std::vector<uint8_t> s_packed(static_cast<size_t>(sN * 2 * kBlockBytes));
    for (int64_t b = 0; b < sN * 2; ++b) {
      uint8_t* blk = s_packed.data() + b * kBlockBytes;
      const uint16_t d = vt::F32ToF16(0.2f);
      std::memcpy(blk + 0, &d, sizeof(d));
      const uint16_t ls = vt::F32ToF16(0.01f);
      std::memcpy(blk + 2, &ls, sizeof(ls));
    }
    std::vector<uint16_t> s_a(static_cast<size_t>(sM * sK),
                              vt::F32ToBF16(0.5f));
    void* mem_a = backend.Alloc(sM * sK * sizeof(uint16_t));
    void* mem_b = backend.Alloc(s_packed.size());
    void* mem_o = backend.Alloc(sM * sN * sizeof(float));
    backend.Copy(q, mem_a, s_a.data(), s_a.size() * sizeof(uint16_t));
    backend.Copy(q, mem_b, s_packed.data(), s_packed.size());
    Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {sM, sK});
    Tensor b_t = Tensor::Contiguous(mem_b, vt::DType::kQ4_K,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {sN, sK});
    Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {sM, sN});
    vt::MatmulBT(q, o_t, a_t, b_t);
    backend.Free(mem_a);
    backend.Free(mem_b);
    backend.Free(mem_o);
  }

  const int64_t free0 = vt::tenstorrent::FreeDeviceDramBytesForTest();
  run_once();
  const int64_t free1 = vt::tenstorrent::FreeDeviceDramBytesForTest();
  run_once();
  const int64_t free2 = vt::tenstorrent::FreeDeviceDramBytesForTest();

  // Every owned surface (word shadow, slots) rode UnregisterHostBuffer away;
  // both calls' decode planes are the only thing that can hold free1/free2
  // below free0. 256 MiB slack covers allocator alignment and kernel-cache
  // residue; the W1-measured orphan set for this shape is gigabytes.
  constexpr int64_t kSlack = 256ll << 20;
  const std::string leak1_msg =
      "first wide keep-quant call leaked its decode planes: free fell from " +
      std::to_string(free0) + " to " + std::to_string(free1) + " bytes (drop " +
      std::to_string(free0 - free1) + ")";
  const std::string leak2_msg =
      "second wide keep-quant call leaked its decode planes: free fell from " +
      std::to_string(free0) + " to " + std::to_string(free2) + " bytes (drop " +
      std::to_string(free0 - free2) + ")";
  CHECK_MESSAGE(free1 >= free0 - kSlack, leak1_msg);
  CHECK_MESSAGE(free2 >= free0 - kSlack, leak2_msg);

  if (trace_had) {
    ::setenv("VT_TT_ALLOC_TRACE", trace_saved.c_str(), 1);
  } else {
    ::unsetenv("VT_TT_ALLOC_TRACE");
  }
  if (int8dot_had) {
    ::setenv("VT_TT_KEEPQUANT_INT8DOT", int8dot_saved.c_str(), 1);
  } else {
    ::unsetenv("VT_TT_KEEPQUANT_INT8DOT");
  }
}

// W4d W2 (#3042) red-first, the chunk-loop sl_alias guard: a SINGLE-chunk
// weight runs the dense chunk loop exactly once with c0 == 0 && c1 == N, and
// that full-extent window is the one case where ttnn::slice returns its INPUT
// (tt-metal slice.cpp:182) — the staged slice IS the resident word shadow,
// so the forced reclaim must skip it or the cache entry dies mid-call.
// Mutation (red-first): sl_alias=false at
// src/vt/tenstorrent/tenstorrent_ops.cpp:2901 force-frees the shadow, the
// second call's cache hit returns the dead tensor and must throw "Tensor is
// not allocated" (or lose residency, the free-memory bar below) — the
// mutation evidence is captured in the queued device phase (the GPU is
// occupied by the 27B gate run).
TEST_CASE("kTENSTORRENT single-chunk keep-quant decode keeps the word shadow resident (#3042)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBTQuant, vt::DeviceType::kTENSTORRENT));

  // The guard under test lives in the W4a chunk decode, which the f32-out
  // dense arm reaches only with the int8-dot env unset; the alloc trace
  // would spam stderr behind the numbers under test.
  const char* const trace_prev = std::getenv("VT_TT_ALLOC_TRACE");
  const bool trace_had = trace_prev != nullptr;
  const std::string trace_saved = trace_had ? std::string(trace_prev) : std::string();
  const char* const int8dot_prev = std::getenv("VT_TT_KEEPQUANT_INT8DOT");
  const bool int8dot_had = int8dot_prev != nullptr;
  const std::string int8dot_saved = int8dot_had ? std::string(int8dot_prev) : std::string();
  ::unsetenv("VT_TT_KEEPQUANT_INT8DOT");
  ::unsetenv("VT_TT_ALLOC_TRACE");

  Backend& backend = *vt::TryGetBackend(DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();

  // Pin the production chunk policy: chunk == N is the single-iteration
  // shape under test, and a leaked non-zero override from another case
  // would silently re-chunk it (and un-exercise the alias).
  vt::tenstorrent::KeepQuantChunkRowsOverrideForTest(0);
  struct ChunkReset {
    ~ChunkReset() { vt::tenstorrent::KeepQuantChunkRowsOverrideForTest(0); }
  } chunk_reset;

  // Shape: plane budget 256 MiB / (K*4) = 32768 rows per chunk against
  // N=8192 -> chunk = min(N, max(32768, ceil(N/8))) = N — ONE chunk
  // iteration, c0 == 0 && c1 == N, the exact window where the slice is its
  // own input.
  constexpr int64_t M = 1, N = 8192;
  const int64_t kBlockBytes = vt::BlockBytes(vt::DType::kQ4_K);
  const int64_t kBlockElems = vt::BlockElems(vt::DType::kQ4_K);
  const int64_t K = 8 * kBlockElems;  // 2048; 8 blocks per row

  std::mt19937 rng(20260912u);
  std::vector<uint8_t> packed(static_cast<size_t>(N * (K / kBlockElems) * kBlockBytes));
  for (int64_t b = 0; b < N * (K / kBlockElems); ++b) {
    uint8_t* blk = packed.data() + b * kBlockBytes;
    const uint16_t d =
        vt::F32ToF16(0.05f + 0.35f * static_cast<float>(rng() % 64) / 64.0f);
    std::memcpy(blk + 0, &d, sizeof(d));
    const uint16_t ls =
        vt::F32ToF16(0.005f + 0.02f * static_cast<float>(rng() % 32) / 32.0f);
    std::memcpy(blk + 2, &ls, sizeof(ls));
    for (int i = 0; i < 12; ++i) blk[4 + i] = static_cast<uint8_t>(rng() & 0xFF);
    for (int i = 0; i < 128; ++i) blk[16 + i] = static_cast<uint8_t>(rng() & 0xFF);
  }
  std::vector<uint16_t> a_bf(static_cast<size_t>(M * K));
  for (auto& v : a_bf)
    v = vt::F32ToBF16((static_cast<float>(rng() % 401) - 200.0f) / 100.0f);

  // Warm-up: a small matmul settles the device/JIT init allocations so the
  // baseline below measures residency, not first-touch.
  {
    constexpr int64_t sM = 4, sN = 8;
    const int64_t sK = 2 * kBlockElems;
    std::vector<uint8_t> s_packed(static_cast<size_t>(sN * 2 * kBlockBytes));
    for (int64_t b = 0; b < sN * 2; ++b) {
      uint8_t* blk = s_packed.data() + b * kBlockBytes;
      const uint16_t d = vt::F32ToF16(0.2f);
      std::memcpy(blk + 0, &d, sizeof(d));
      const uint16_t ls = vt::F32ToF16(0.01f);
      std::memcpy(blk + 2, &ls, sizeof(ls));
    }
    std::vector<uint16_t> s_a(static_cast<size_t>(sM * sK),
                              vt::F32ToBF16(0.5f));
    void* mem_a = backend.Alloc(sM * sK * sizeof(uint16_t));
    void* mem_b = backend.Alloc(s_packed.size());
    void* mem_o = backend.Alloc(sM * sN * sizeof(float));
    backend.Copy(q, mem_a, s_a.data(), s_a.size() * sizeof(uint16_t));
    backend.Copy(q, mem_b, s_packed.data(), s_packed.size());
    Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {sM, sK});
    Tensor b_t = Tensor::Contiguous(mem_b, vt::DType::kQ4_K,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {sN, sK});
    Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                    Device{vt::DeviceType::kTENSTORRENT, 0}, {sM, sN});
    vt::MatmulBT(q, o_t, a_t, b_t);
    backend.Free(mem_a);
    backend.Free(mem_b);
    backend.Free(mem_o);
  }

  // ONE host weight buffer held across BOTH calls: the word shadow is keyed
  // by the host pointer, so the second call must hit the cache and decode
  // from the SAME resident words the first call staged — the exact state
  // the sl_alias guard protects.
  void* mem_a = backend.Alloc(M * K * sizeof(uint16_t));
  void* mem_b = backend.Alloc(packed.size());
  void* mem_o = backend.Alloc(M * N * sizeof(float));
  backend.Copy(q, mem_a, a_bf.data(), a_bf.size() * sizeof(uint16_t));
  backend.Copy(q, mem_b, packed.data(), packed.size());
  Tensor a_t = Tensor::Contiguous(mem_a, vt::DType::kBF16,
                                  Device{vt::DeviceType::kTENSTORRENT, 0}, {M, K});
  Tensor b_t = Tensor::Contiguous(mem_b, vt::DType::kQ4_K,
                                  Device{vt::DeviceType::kTENSTORRENT, 0}, {N, K});
  Tensor o_t = Tensor::Contiguous(mem_o, vt::DType::kF32,
                                  Device{vt::DeviceType::kTENSTORRENT, 0}, {M, N});

  const int64_t free0 = vt::tenstorrent::FreeDeviceDramBytesForTest();
  vt::MatmulBT(q, o_t, a_t, b_t);
  const int64_t free1 = vt::tenstorrent::FreeDeviceDramBytesForTest();
  vt::MatmulBT(q, o_t, a_t, b_t);
  const int64_t free2 = vt::tenstorrent::FreeDeviceDramBytesForTest();
  std::vector<float> out(static_cast<size_t>(M * N), 0.0f);
  backend.Copy(q, out.data(), mem_o, out.size() * sizeof(float));

  // The shadow is ~1 MiB of i32 words and both calls' decode planes are
  // reclaimed in-call; 256 MiB slack covers allocator alignment and
  // kernel-cache residue. Under the mutation the shadow is force-freed and
  // the second call throws before this bar is even reached — the uncaught
  // exception IS the red.
  constexpr int64_t kSlack = 256ll << 20;
  const std::string drop1_msg =
      "first single-chunk keep-quant call lost residency: free fell from " +
      std::to_string(free0) + " to " + std::to_string(free1) + " bytes (drop " +
      std::to_string(free0 - free1) + ")";
  const std::string drop2_msg =
      "second single-chunk keep-quant call lost residency: free fell from " +
      std::to_string(free0) + " to " + std::to_string(free2) + " bytes (drop " +
      std::to_string(free0 - free2) + ")";
  CHECK_MESSAGE(free1 >= free0 - kSlack, drop1_msg);
  CHECK_MESSAGE(free2 >= free0 - kSlack, drop2_msg);

  // UnregisterHostBuffer drops the word shadow with the host weight buffer —
  // the designed teardown, run only after both calls survived on it.
  backend.Free(mem_a);
  backend.Free(mem_b);
  backend.Free(mem_o);

  if (trace_had) {
    ::setenv("VT_TT_ALLOC_TRACE", trace_saved.c_str(), 1);
  } else {
    ::unsetenv("VT_TT_ALLOC_TRACE");
  }
  if (int8dot_had) {
    ::setenv("VT_TT_KEEPQUANT_INT8DOT", int8dot_saved.c_str(), 1);
  } else {
    ::unsetenv("VT_TT_KEEPQUANT_INT8DOT");
  }
}

// W4d W3 (#3042): the loader stages every weight's bf16 TILE form as the
// slot's persistent buffer. For a k-quant weight that form is dead the
// moment the keep-quant word shadow exists — the matmul reads only the
// words — but it stayed resident forever, and at 27B the staged bf16 forms
// filled the banks to 93 percent during the warm pass and fragmented them
// into the init OOM (attn_qkv alone: 97 x [10240,5120] bf16 = 9.7 GiB
// beside ~21 MiB q6_K each). EnsureKeepQuantWords now releases the slot's
// bf16 forms when it stores the shadow. RED on the pre-fix tree: the same
// assertion measured 268 MB of bf16 staging still held after the shadow
// existed (the census ledger: /tmp/census-run2.log, pers 9.7 GiB at
// block/54 with the words already resident).
// The premise-broken bf16-staging test was removed (W4d W6): its premise —
// that the loader stages a bf16 twin of packed weights — was falsified by
// the slot census (the `pers` bf16 forms are the GDN projections' LEGITIMATE
// expand-arm residency, not a double-hold), and its EnsureDevice2D-based
// setup cannot stage a packed tensor ("unsupported float dtype"). The
// equivalence it aimed at is proven TT-free by the W4d W4 standalone check
// (q6_K + q4_K, bit-for-bit vs the element-level reorder through the real
// dequantizer) and locked by the bf16-out capture test above.


// ── BFP8 weight residency (spec .agents/specs/tenstorrent-bfp-weight-
// residency.md) ─────────────────────────────────────────────────────────────
// VT_TT_WEIGHT_RESIDENCY=1 converts a bf16 matmul WEIGHT to a device-resident
// BFLOAT8_B operand at first staging and the NATIVE ttnn::matmul consumes it
// (bf16 activation x BFP8 weight — ttnn/operations/matmul/matmul.cpp:521-532).
// The f32-exact SFPU floor is not involved. BFLOAT8_B packs 1 sign + 7
// shared-group mantissa bits per element with one 8-bit exponent per
// 16-element group (tt_metal/impl/data_format/blockfloat_common.cpp,
// `convert_bfp_to_u32`, Bfp8_b arm) — the rounding IS the precision contract.
namespace vt::tenstorrent {
// Residency probes (tenstorrent_internal.h); declared here so the test TU
// never includes internal headers.
uint64_t Bfp8ResidentWeights();
uint64_t Bfp8MatmulUses();
uint64_t Bfp8Refusals();
const char* Bfp8LastRefusal();
}  // namespace vt::tenstorrent

namespace {

uint16_t F32ToBf16Bits(float v) {
  uint32_t b;
  std::memcpy(&b, &v, 4);
  const uint32_t lsb = (b >> 16) & 1u;
  b += 0x7fffu + lsb;
  return static_cast<uint16_t>(b >> 16);
}
float Bf16BitsToFloat(uint16_t h) {
  const uint32_t b = static_cast<uint32_t>(h) << 16;
  float v;
  std::memcpy(&v, &b, 4);
  return v;
}

// CPU test-ONLY mirror of the BFLOAT8_B quantization: per 16-element group,
// shared exponent = biased exponent of the group max; each element keeps
// 1 sign + 7 mantissa bits relative to that shared exponent. This is the
// DEQUANTIZED reference operand for the quantize-then-compare gate — never a
// runtime path.
float Bfp8MirrorDequant(float v, uint8_t shared_exp) {
  if (v == 0.0f) return 0.0f;
  const float scale = std::ldexp(1.0f, static_cast<int>(shared_exp) - 127);
  float x = std::fabs(v) / scale;  // in (2^-8, 2)
  if (x >= 2.0f) x = 1.9999999f;
  float q = std::nearbyintf(x * 128.0f);  // RNE on 7 mantissa bits
  if (q > 255.0f) q = 255.0f;
  return (v < 0 ? -1.0f : 1.0f) * (q / 128.0f) * scale;
}

// Runs `a @ b^T` (bf16 operands, bf16 out) through the production kMatmulBT
// op and returns the device output widened to f32.
std::vector<float> RunMatmulBTBF16(Backend& backend, Queue& q, uint32_t M,
                                   uint32_t K, uint32_t N,
                                   const std::vector<float>& host_a,
                                   const std::vector<float>& host_b) {
  std::vector<uint16_t> a_bits(host_a.size()), b_bits(host_b.size());
  for (size_t i = 0; i < host_a.size(); ++i) a_bits[i] = F32ToBf16Bits(host_a[i]);
  for (size_t i = 0; i < host_b.size(); ++i) b_bits[i] = F32ToBf16Bits(host_b[i]);

  void* mem_a = backend.Alloc(a_bits.size() * 2);
  void* mem_b = backend.Alloc(b_bits.size() * 2);
  void* mem_out = backend.Alloc(static_cast<size_t>(M) * N * 2);
  backend.Copy(q, mem_a, a_bits.data(), a_bits.size() * 2);
  backend.Copy(q, mem_b, b_bits.data(), b_bits.size() * 2);
  Tensor a = Tensor::Contiguous(mem_a, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {M, K});
  Tensor b = Tensor::Contiguous(mem_b, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {N, K});
  Tensor out = Tensor::Contiguous(mem_out, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {M, N});
  auto matmul_bt =
      reinterpret_cast<vt::MatmulFn>(vt::GetOp(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));
  matmul_bt(q, out, a, b);
  std::vector<uint16_t> out_bits(static_cast<size_t>(M) * N);
  backend.Copy(q, out_bits.data(), mem_out, out_bits.size() * 2);
  backend.Free(mem_a);
  backend.Free(mem_b);
  backend.Free(mem_out);
  std::vector<float> out_f(out_bits.size());
  for (size_t i = 0; i < out_bits.size(); ++i) out_f[i] = Bf16BitsToFloat(out_bits[i]);
  return out_f;
}

struct ScopedEnv {
  explicit ScopedEnv(const char* v) {
    const char* old = std::getenv("VT_TT_WEIGHT_RESIDENCY");
    had_ = old != nullptr;
    if (had_) old_ = old;
    if (v != nullptr) ::setenv("VT_TT_WEIGHT_RESIDENCY", v, 1);
    else ::unsetenv("VT_TT_WEIGHT_RESIDENCY");
  }
  ~ScopedEnv() {
    if (had_) ::setenv("VT_TT_WEIGHT_RESIDENCY", old_.c_str(), 1);
    else ::unsetenv("VT_TT_WEIGHT_RESIDENCY");
  }
  bool had_ = false;
  std::string old_;
};

}  // namespace

TEST_CASE("kTENSTORRENT VT_TT_WEIGHT_RESIDENCY default OFF leaves kMatmulBT "
          "inert (no BFP8 resident, no use)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  // Both the unset default AND the explicit "off" opt-out must leave every
  // existing path untouched: no conversion, no BFP8 operand consumed, and a
  // result inside the plain bf16 matmul envelope.
  for (const char* env_val : {(const char*)nullptr, "off"}) {
    ScopedEnv guard(env_val);
    Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
    Queue q = backend.CreateQueue();
    constexpr uint32_t M = 32, K = 64, N = 32;
    std::vector<float> host_a(static_cast<size_t>(M) * K),
        host_b(static_cast<size_t>(N) * K);
    for (size_t i = 0; i < host_a.size(); ++i) host_a[i] = 0.25f * static_cast<float>(i % 7);
    for (size_t i = 0; i < host_b.size(); ++i) host_b[i] = 0.125f * static_cast<float>(i % 13);
    const uint64_t resident_before = vt::tenstorrent::Bfp8ResidentWeights();
    const uint64_t uses_before = vt::tenstorrent::Bfp8MatmulUses();
    const std::vector<float> out =
        RunMatmulBTBF16(backend, q, M, K, N, host_a, host_b);
    CHECK(vt::tenstorrent::Bfp8ResidentWeights() == resident_before);
    CHECK(vt::tenstorrent::Bfp8MatmulUses() == uses_before);
    // Plain bf16-matmul envelope vs the f32 reference.
    float max_abs = 0.0f;
    for (uint32_t i = 0; i < M; ++i)
      for (uint32_t j = 0; j < N; ++j) {
        float ref = 0.0f;
        for (uint32_t k = 0; k < K; ++k)
          ref += Bf16BitsToFloat(F32ToBf16Bits(host_a[i * K + k])) *
                 Bf16BitsToFloat(F32ToBf16Bits(host_b[j * K + k]));
        max_abs = std::max(max_abs, std::fabs(out[i * N + j] - ref));
      }
    CHECK(max_abs < 0.5f);
  }
}

TEST_CASE("kTENSTORRENT VT_TT_WEIGHT_RESIDENCY=bfp8 stages the weight as a "
          "device-resident BFLOAT8_B operand consumed by the native matmul "
          "(quantize-then-compare gate)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  ScopedEnv guard("bfp8");
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();
  constexpr uint32_t M = 32, K = 64, N = 32;
  std::vector<float> host_a(static_cast<size_t>(M) * K),
      host_b(static_cast<size_t>(N) * K);
  for (size_t i = 0; i < host_a.size(); ++i) host_a[i] = 0.25f * static_cast<float>(i % 7);
  for (size_t i = 0; i < host_b.size(); ++i) host_b[i] = 0.125f * static_cast<float>(i % 13);

  const uint64_t resident_before = vt::tenstorrent::Bfp8ResidentWeights();
  const uint64_t uses_before = vt::tenstorrent::Bfp8MatmulUses();
  const std::vector<float> out = RunMatmulBTBF16(backend, q, M, K, N, host_a, host_b);
  // PATH PIN: the weight became a device-resident BFP8 operand and a matmul
  // consumed it. Without the arm these counters never move — this is what reds
  // on the wrong path (the bf16 resident arm computes a similar number).
  CHECK(vt::tenstorrent::Bfp8ResidentWeights() > resident_before);
  CHECK(vt::tenstorrent::Bfp8MatmulUses() > uses_before);
  // A second call allocates a FRESH weight staging buffer (a new host
  // pointer), so it stages its own BFP8 tensor — the shadow cache is keyed by
  // the host pointer, which at model load is the stable weight allocation.
  (void)RunMatmulBTBF16(backend, q, M, K, N, host_a, host_b);
  CHECK(vt::tenstorrent::Bfp8ResidentWeights() == resident_before + 2);
  CHECK(vt::tenstorrent::Bfp8MatmulUses() > uses_before + 1);

  // QUANTIZE-THEN-COMPARE: dequantize `b` through the CPU BFP8 mirror (groups
  // of 16 along K) and take the f32 matmul of THAT as the reference. The
  // contract is "computes what BFP8 defines", so the envelope is the BFP
  // quantization step — never a raw-f32 tolerance.
  std::vector<float> b_deq(host_b.size());
  float max_b = 0.0f;
  for (float v : host_b) max_b = std::max(max_b, std::fabs(v));
  for (size_t row = 0; row < host_b.size(); row += 16) {
    uint8_t shared = 0;
    for (size_t e = 0; e < 16 && row + e < host_b.size(); ++e) {
      int exp = 0;
      std::frexp(host_b[row + e], &exp);
      shared = std::max(shared, static_cast<uint8_t>(exp - 1 + 127));
    }
    for (size_t e = 0; e < 16 && row + e < host_b.size(); ++e)
      b_deq[row + e] = Bfp8MirrorDequant(host_b[row + e], shared);
  }
  for (uint32_t i = 0; i < M; ++i)
    for (uint32_t j = 0; j < N; ++j) {
      float ref = 0.0f;
      float sum_abs_a = 0.0f;
      for (uint32_t k = 0; k < K; ++k) {
        const float a = Bf16BitsToFloat(F32ToBf16Bits(host_a[i * K + k]));
        ref += a * b_deq[j * K + k];
        sum_abs_a += std::fabs(a);
      }
      // |Δb| ≤ max|b|/128 (half ulp of the 7-bit mantissa) doubled for the
      // device-side rounding-mode difference, plus the bf16 output store.
      const float bound = (max_b / 64.0f) * sum_abs_a +
                          std::fabs(ref) * (1.0f / 256.0f) + 1e-3f;
      const float diff = std::fabs(out[i * N + j] - ref);
      CHECK_MESSAGE(diff <= bound,
                    "BFP8 matmul output outside the quantization envelope at ("
                        << i << "," << j << "): " << diff << " > " << bound);
    }
}

TEST_CASE("kTENSTORRENT VT_TT_WEIGHT_RESIDENCY=bfp8 refuses a non-TILE-aligned "
          "weight BY NAME and falls through to the bf16 arm") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  ScopedEnv guard("bfp8");
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();
  constexpr uint32_t M = 32, K = 33, N = 32;  // K=33 is not TILE-aligned
  std::vector<float> host_a(static_cast<size_t>(M) * K),
      host_b(static_cast<size_t>(N) * K);
  for (size_t i = 0; i < host_a.size(); ++i) host_a[i] = 0.25f * static_cast<float>(i % 5);
  for (size_t i = 0; i < host_b.size(); ++i) host_b[i] = 0.5f * static_cast<float>(i % 3);

  const uint64_t refusals_before = vt::tenstorrent::Bfp8Refusals();
  const std::vector<float> out = RunMatmulBTBF16(backend, q, M, K, N, host_a, host_b);
  CHECK(vt::tenstorrent::Bfp8Refusals() > refusals_before);
  CHECK(std::string(vt::tenstorrent::Bfp8LastRefusal())
            .find("TILE-aligned") != std::string::npos);
  // The fall-through is still CORRECT (bf16 envelope vs the f32 reference).
  float max_abs = 0.0f;
  for (uint32_t i = 0; i < M; ++i)
    for (uint32_t j = 0; j < N; ++j) {
      float ref = 0.0f;
      for (uint32_t k = 0; k < K; ++k)
        ref += Bf16BitsToFloat(F32ToBf16Bits(host_a[i * K + k])) *
               Bf16BitsToFloat(F32ToBf16Bits(host_b[j * K + k]));
      max_abs = std::max(max_abs, std::fabs(out[i * N + j] - ref));
    }
  CHECK(max_abs < 0.5f);
}

TEST_CASE("kTENSTORRENT VT_TT_WEIGHT_RESIDENCY=bfp4 refuses BY NAME "
          "(not implemented) and falls through to the bf16 arm") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  ScopedEnv guard("bfp4");
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();
  constexpr uint32_t M = 32, K = 64, N = 32;
  std::vector<float> host_a(static_cast<size_t>(M) * K),
      host_b(static_cast<size_t>(N) * K);
  for (size_t i = 0; i < host_a.size(); ++i) host_a[i] = 0.25f * static_cast<float>(i % 5);
  for (size_t i = 0; i < host_b.size(); ++i) host_b[i] = 0.5f * static_cast<float>(i % 3);

  const uint64_t refusals_before = vt::tenstorrent::Bfp8Refusals();
  const std::vector<float> out = RunMatmulBTBF16(backend, q, M, K, N, host_a, host_b);
  // bfp4 is a RESERVED lever value: the staging seam refuses it by name, and
  // no BFP8 conversion happens (the resident counter must not move).
  CHECK(vt::tenstorrent::Bfp8Refusals() > refusals_before);
  CHECK(std::string(vt::tenstorrent::Bfp8LastRefusal())
            .find("bfp4") != std::string::npos);
  CHECK(std::string(vt::tenstorrent::Bfp8LastRefusal())
            .find("not implemented") != std::string::npos);
  // The fall-through is still CORRECT (bf16 envelope vs the f32 reference).
  float max_abs = 0.0f;
  for (uint32_t i = 0; i < M; ++i)
    for (uint32_t j = 0; j < N; ++j) {
      float ref = 0.0f;
      for (uint32_t k = 0; k < K; ++k)
        ref += Bf16BitsToFloat(F32ToBf16Bits(host_a[i * K + k])) *
               Bf16BitsToFloat(F32ToBf16Bits(host_b[j * K + k]));
      max_abs = std::max(max_abs, std::fabs(out[i * N + j] - ref));
    }
  CHECK(max_abs < 0.5f);
}

TEST_CASE("kTENSTORRENT BFP8 vs bf16 resident-weight matmul timing (VT_TT_BFP8_BENCH=1)") {
  if (std::getenv("VT_TT_BFP8_BENCH") == nullptr) {
    MESSAGE("SKIPPED: set VT_TT_BFP8_BENCH=1");
    return;
  }
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  Queue q = backend.CreateQueue();
  // 27B-class MLP gate/up shape family: [N,K] = 17408x5120, decode M=1.
  constexpr uint32_t M = 1, K = 5120, N = 17408;
  std::vector<float> host_a(static_cast<size_t>(M) * K, 0.5f),
      host_b(static_cast<size_t>(N) * K);
  for (size_t i = 0; i < host_b.size(); ++i)
    host_b[i] = 0.125f * static_cast<float>(i % 13);
  for (const char* env_val : {(const char*)nullptr, "bfp8"}) {
    ScopedEnv guard(env_val);
    // Warm the staging (first call converts/uploads), then time eager calls.
    (void)RunMatmulBTBF16(backend, q, M, K, N, host_a, host_b);
    const auto t0 = std::chrono::steady_clock::now();
    constexpr int kIters = 10;
    for (int it = 0; it < kIters; ++it)
      (void)RunMatmulBTBF16(backend, q, M, K, N, host_a, host_b);
    const auto t1 = std::chrono::steady_clock::now();
    // RunMatmulBTBF16 includes host<->device copies; report the whole-call cost
    // as an upper bound on the GEMM delta.
    const double us_per_call =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / kIters;
    MESSAGE("VT_TT_WEIGHT_RESIDENCY=", env_val == nullptr ? "unset(bf16)" : "bfp8",
            ": ", us_per_call, " us/call whole-op (M=", M, ",K=", K, ",N=", N,
            "), Bfp8MatmulUses=", vt::tenstorrent::Bfp8MatmulUses());
  }
}
