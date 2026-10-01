// VT_HOST_EMBEDDING: the HOST arm of the token-table gather.
//
// The arm exists to keep `[vocab, H]` out of device memory: the table stays in
// host RAM, the requested rows are gathered through the CPU `vt::Embedding`
// kernel (ONE ROW per id, the `ggml_get_rows` discipline), and only the `[T,H]`
// result is copied to the device. The gate therefore has two halves, and both
// matter:
//
//   1. the rows are RIGHT, byte-compared against the same pinned IQ4_NL/Q5_0
//      oracle vectors the device-side `vt::Embedding` gate uses
//      (`tests/vt/iq4nl_q5_0_golden_vectors.h`), and
//   2. the table never reached the device — `OwnedTensor::d_dev` stays null. A
//      gather that uploads the table first passes (1) and fails the entire point.
//
// The flag is read ONCE per process (`HostEmbedInto`'s function-static), so this
// binary turns it on in a global initializer, before main and thus before any
// TEST_CASE. The flag-OFF arm is a plain early return, and every other model
// suite in the tree exercises it (none sets the variable); what is pinned here for
// the boundary between the arms is the DECLINE, which is the released-host-bytes
// case below.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#define VLLM_HOST_EMBED_CAPTURE 1
#endif

#include "vllm/model_executor/models/dense_device_glue.h"  // Dev, DBuf
#include "vllm/model_executor/models/host_embedding.h"  // HostEmbedInto, EmbedGather
#include "vllm/model_executor/models/owned_bytes.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"  // detail::DeviceTokenIdsScope
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor
#include "vt/dtype.h"
#include "vt/ops.h"

#include "support/test_env.h"
#include "vt/iq4nl_q5_0_golden_vectors.h"

namespace {

struct EnableHostEmbedding {
  EnableHostEmbedding() {
    // `support/test_env.h` is the portable setter (`_putenv_s` on MSVC): the
    // unconditional POSIX `::setenv` this used to call does not compile in a
    // CPU-only MSVC build, and this target is unconditional.
    vllm_test::SetEnv("VT_HOST_EMBEDDING", "1");
    // The arm prints ONE banner per process (`logged`), so the first case is the
    // one that can read it. It is what proves the arm RAN: the CPU device arm
    // aliases the host bytes and sets no `d_dev`, so `d_dev == nullptr` alone
    // cannot tell the two arms apart on this backend.
    vllm_test::SetEnv("VT_HOST_EMBED_TRACE", "1");
  }
};
const EnableHostEmbedding g_enable_host_embedding;

#ifdef VLLM_HOST_EMBED_CAPTURE
// Everything written to stderr while `body` runs, as a string. Same shape as the
// qwen4_exp MoE tap's capture: the arm's banner is the observable, so a test that
// cannot read it must not report that the arm ran.
template <typename F>
std::string CaptureStderr(F&& body) {
  std::fflush(stderr);
  int saved = ::dup(2);
  char path[] = "/tmp/vllm_hostembed_XXXXXX";
  int fd = ::mkstemp(path);
  REQUIRE(saved >= 0);
  REQUIRE(fd >= 0);
  ::dup2(fd, 2);
  body();
  std::fflush(stderr);
  ::dup2(saved, 2);
  ::close(saved);
  ::lseek(fd, 0, SEEK_SET);
  std::string out;
  char buf[4096];
  ssize_t n = 0;
  while ((n = ::read(fd, buf, sizeof(buf))) > 0) out.append(buf, static_cast<size_t>(n));
  ::close(fd);
  ::unlink(path);
  return out;
}
#endif

uint32_t BitsToF32(float f) {
  uint32_t bits = 0;
  std::memcpy(&bits, &f, sizeof(bits));
  return bits;
}
float BitsToFloat(uint32_t bits) {
  float f = 0.0F;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// The loader's residency for a gather table: an OWNED host buffer with the
// descriptor a forward reads. No device handles — creating them is the one thing
// the host arm must not do.
vllm::OwnedTensor HostTable(vt::DType dt, int64_t rows, int64_t k,
                            const void* bytes, size_t nbytes) {
  std::vector<uint8_t> copy(nbytes);
  std::memcpy(copy.data(), bytes, nbytes);
  vllm::OwnedTensor t;
  t.dtype = dt;
  t.rank = 2;
  t.shape[0] = rows;
  t.shape[1] = k;
  t.bytes = vllm::OwnedBytes(std::move(copy));
  return t;
}

}  // namespace

TEST_CASE("host embedding: IQ4_NL rows match the pinned oracle, table stays off-device") {
  constexpr int64_t k = 160;  // 5 whole IQ4_NL blocks per row
  constexpr int64_t rows = 2;
  vllm::OwnedTensor table =
      HostTable(vt::DType::kIQ4_NL, rows, k, vllm_test::kIq4nlGoldenBlocks,
                sizeof(vllm_test::kIq4nlGoldenBlocks));
  // Repeats and a backwards step, so a gather that ignored the id or walked the
  // table in order cannot pass (the same ids the device-side op test uses).
  const std::vector<int32_t> ids{1, 0, 1, 1, 0};
  const int64_t T = static_cast<int64_t>(ids.size());

  vt::Queue q = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};
  vllm::dense_attn::DBuf out(d, vt::DType::kBF16, {T, k});
#ifdef VLLM_HOST_EMBED_CAPTURE
  // (0) THE ARM RAN. This is the assertion that separates the host arm from the
  // device arm on a CPU backend, where `ResidentWeight` aliases the host bytes
  // and the `d_dev` check below is trivially true either way. With the flag off
  // this capture is empty and the case fails.
  const std::string trace = CaptureStderr([&] {
    vllm::dense_attn::EmbedGather(d, out, ids, table, rows, k,
                                  "test_host_embedding");
  });
  CHECK(trace.find("[host-embed]") != std::string::npos);
  CHECK(trace.find("CPU row gather") != std::string::npos);
#else
  vllm::dense_attn::EmbedGather(d, out, ids, table, rows, k,
                                "test_host_embedding");
#endif

  // (2) THE POINT OF THE ARM on a backend that stages: no device resident was
  // created. On the CPU backend the device arm aliases instead of staging, so
  // this half is not the discriminating one here — (0) is.
  CHECK(table.d_dev == nullptr);
  CHECK(table.HasHostBytes());

  // (1) The rows, against the pinned oracle, rounded once into bf16 — the same
  // comparison `test_ops_embedding_quant.cpp` makes on the device arm.
  const auto* got = static_cast<const uint16_t*>(out.ptr());
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t j = 0; j < k; ++j) {
      CAPTURE(t);
      CAPTURE(j);
      const float want = BitsToFloat(
          vllm_test::kIq4nlGoldenBits[static_cast<size_t>(ids[static_cast<size_t>(t)]) *
                                          static_cast<size_t>(k) +
                                      static_cast<size_t>(j)]);
      CHECK(got[static_cast<size_t>(t) * static_cast<size_t>(k) +
                static_cast<size_t>(j)] == vt::F32ToBF16(want));
    }
  }
}

TEST_CASE("host embedding: a bf16 table takes the byte-copy fast path") {
  constexpr int64_t k = 8;
  constexpr int64_t rows = 3;
  std::vector<uint16_t> values(static_cast<size_t>(rows * k));
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = vt::F32ToBF16(static_cast<float>(i) * 0.5F - 2.0F);
  }
  vllm::OwnedTensor table =
      HostTable(vt::DType::kBF16, rows, k, values.data(), values.size() * 2);
  const std::vector<int32_t> ids{2, 0, 1, 2};

  vt::Queue q = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};
  vllm::dense_attn::DBuf out(d, vt::DType::kBF16,
                             {static_cast<int64_t>(ids.size()), k});
  // The arm ACCEPTS (the process-wide banner already printed in the first case,
  // so the return value is the assertion here rather than the trace).
  CHECK(vllm::dense_attn::HostEmbedInto(d, out, ids, table, rows, k));
  vllm::dense_attn::EmbedGather(d, out, ids, table, rows, k,
                                "test_host_embedding");

  CHECK(table.d_dev == nullptr);
  const auto* got = static_cast<const uint16_t*>(out.ptr());
  for (size_t t = 0; t < ids.size(); ++t) {
    for (int64_t j = 0; j < k; ++j) {
      CAPTURE(t);
      CAPTURE(j);
      CHECK(got[t * static_cast<size_t>(k) + static_cast<size_t>(j)] ==
            values[static_cast<size_t>(ids[t]) * static_cast<size_t>(k) +
                   static_cast<size_t>(j)]);
    }
  }
}

TEST_CASE("host embedding: an f16 table converts through the CPU kernel") {
  constexpr int64_t k = 4;
  constexpr int64_t rows = 2;
  std::vector<uint16_t> halved(static_cast<size_t>(rows * k));
  for (size_t i = 0; i < halved.size(); ++i) {
    halved[i] = vt::F32ToF16(static_cast<float>(i) + 0.25F);
  }
  vllm::OwnedTensor table =
      HostTable(vt::DType::kF16, rows, k, halved.data(), halved.size() * 2);
  const std::vector<int32_t> ids{1, 0};

  vt::Queue q = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};
  vllm::dense_attn::DBuf out(d, vt::DType::kF32,
                             {static_cast<int64_t>(ids.size()), k});
  CHECK(vllm::dense_attn::HostEmbedInto(d, out, ids, table, rows, k));
  vllm::dense_attn::EmbedGather(d, out, ids, table, rows, k,
                                "test_host_embedding");

  CHECK(table.d_dev == nullptr);
  const auto* got = static_cast<const float*>(out.ptr());
  for (size_t t = 0; t < ids.size(); ++t) {
    for (int64_t j = 0; j < k; ++j) {
      CAPTURE(t);
      CAPTURE(j);
      const float want =
          vt::F16ToF32(halved[static_cast<size_t>(ids[t]) * static_cast<size_t>(k) +
                              static_cast<size_t>(j)]);
      CHECK(BitsToF32(got[t * static_cast<size_t>(k) + static_cast<size_t>(j)]) ==
            BitsToF32(want));
    }
  }
}

TEST_CASE("host embedding: an out-of-range id is refused") {
  constexpr int64_t k = 4;
  constexpr int64_t rows = 2;
  const std::vector<uint16_t> values(static_cast<size_t>(rows * k),
                                     vt::F32ToBF16(1.0F));
  vllm::OwnedTensor table =
      HostTable(vt::DType::kBF16, rows, k, values.data(), values.size() * 2);
  const std::vector<int32_t> ids{2};  // rows == 2, so 2 is out of range

  vt::Queue q = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};
  vllm::dense_attn::DBuf out(d, vt::DType::kBF16, {1, k});
  CHECK_THROWS_AS(vllm::dense_attn::EmbedGather(d, out, ids, table, rows, k,
                                                "test_host_embedding"),
                  std::runtime_error);
}

TEST_CASE("host embedding: a released host table declines the host arm") {
  constexpr int64_t k = 4;
  constexpr int64_t rows = 2;
  const std::vector<uint16_t> values(static_cast<size_t>(rows * k),
                                     vt::F32ToBF16(1.0F));
  vllm::OwnedTensor table =
      HostTable(vt::DType::kBF16, rows, k, values.data(), values.size() * 2);
  table.host_released = true;

  vt::Queue q = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};
  vllm::dense_attn::DBuf out(d, vt::DType::kBF16, {1, k});
  const std::vector<int32_t> ids{0};

  // The boundary between the arms: the host arm must DECLINE, so `EmbedGather`
  // falls through to the device arm rather than gathering bytes that are gone.
  CHECK_FALSE(vllm::dense_attn::HostEmbedInto(d, out, ids, table, rows, k));
}

TEST_CASE("host embedding: a device override splices its rows over the host upload") {
  constexpr int64_t k = 4;
  constexpr int64_t rows = 3;
  // Row r is the constant bf16(r + 1), so a resolved id is readable from the
  // output bytes alone, with no second gather to trust.
  std::vector<uint16_t> values(static_cast<size_t>(rows * k));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t j = 0; j < k; ++j) {
      values[static_cast<size_t>(r * k + j)] =
          vt::F32ToBF16(static_cast<float>(r + 1));
    }
  }
  vllm::OwnedTensor table =
      HostTable(vt::DType::kBF16, rows, k, values.data(), values.size() * 2);

  const std::vector<int32_t> host_ids{0, 1, 2};
  // ONE id for the first row: the shared splice replaces exactly `count` rows,
  // so the output must read [1, 1, 2]. The tail keeps the host upload's rows,
  // which is the padded case a "replace everything" copy would break.
  const std::vector<int32_t> override_ids{1};

  vt::Queue q = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};
  vllm::dense_attn::DBuf out(d, vt::DType::kBF16, {3, k});

  {
    vllm::detail::DeviceTokenIdsScope scope(
        override_ids.data(), static_cast<int64_t>(override_ids.size()));
    CHECK(vllm::dense_attn::HostEmbedInto(d, out, host_ids, table, rows, k));
  }

  const auto* got = static_cast<const uint16_t*>(out.ptr());
  const int32_t want[3] = {1, 1, 2};
  for (int64_t t = 0; t < 3; ++t) {
    for (int64_t j = 0; j < k; ++j) {
      CAPTURE(t);
      CAPTURE(j);
      CHECK(got[static_cast<size_t>(t) * static_cast<size_t>(k) +
                static_cast<size_t>(j)] ==
            values[static_cast<size_t>(want[static_cast<size_t>(t)]) *
                       static_cast<size_t>(k) +
                   static_cast<size_t>(j)]);
    }
  }
}

TEST_CASE("host embedding: an override longer than the embed input is refused") {
  constexpr int64_t k = 4;
  constexpr int64_t rows = 2;
  const std::vector<uint16_t> values(static_cast<size_t>(rows * k),
                                     vt::F32ToBF16(1.0F));
  vllm::OwnedTensor table =
      HostTable(vt::DType::kBF16, rows, k, values.data(), values.size() * 2);

  const std::vector<int32_t> host_ids{0};         // T = 1 row
  const std::vector<int32_t> override_ids{1, 0};  // 2 > T

  vt::Queue q = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};
  vllm::dense_attn::DBuf out(d, vt::DType::kBF16, {1, k});

  vllm::detail::DeviceTokenIdsScope scope(
      override_ids.data(), static_cast<int64_t>(override_ids.size()));
  // The host upload is one row; splicing two would write past it. The device
  // arm's shared `ApplyDeviceTokenIds` bounds the override and throws with the
  // caller's name, and the host arm now routes through that same check instead
  // of issuing its own unchecked Copy.
  CHECK_THROWS_AS(
      vllm::dense_attn::HostEmbedInto(d, out, host_ids, table, rows, k),
      std::runtime_error);
}
