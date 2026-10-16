#include <doctest/doctest.h>
#include "vt/ops.h"
#include <array>

TEST_CASE("copy: CPU transposed overlap uses snapshot semantics") {
  vt::Queue q;
  std::array<float, 9> data{0,1,2,3,4,5,6,7,8};
  auto in = vt::Tensor::Contiguous(data.data(), vt::DType::kF32, q.device, {3,3});
  auto out = in; out.stride[0] = 1; out.stride[1] = 3;
  vt::Copy(q, out, in);
  CHECK(data == std::array<float, 9>{0,3,6,1,4,7,2,5,8});
}
TEST_CASE("copy: CPU casts strided rows and rejects overlapping destination elements") {
  vt::Queue q;
  std::array<float, 8> data{1,2,3,99,4,5,6,99};
  std::array<uint16_t, 6> bits{};
  auto in = vt::Tensor::Contiguous(data.data(), vt::DType::kF32, q.device, {2,4});
  in.shape[1] = 3;
  auto out = vt::Tensor::Contiguous(bits.data(), vt::DType::kBF16, q.device, {2,3});
  vt::Copy(q, out, in);
  for (int i = 0; i < 6; ++i) CHECK(vt::BF16ToF32(bits[i]) == i + 1);
  out.stride[0] = 1;
  CHECK_THROWS(vt::Copy(q, out, in));
}
