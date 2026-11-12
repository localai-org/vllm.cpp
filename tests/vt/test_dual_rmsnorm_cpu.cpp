#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include "vt/fused_ops.h"
#include "vt/ops.h"
#include "vt/op_provider.h"

namespace {
bool oversized_dispatched = false;
void RejectOversizedDispatch(vt::Queue&, vt::Tensor&, const vt::Tensor&,
                             const vt::Tensor&, const vt::RmsNormArgs&, vt::Tensor*) {
  oversized_dispatched = true;
  throw std::runtime_error("oversized RMSNorm reached provider");
}
}


int main() {
  using namespace vt;
  Queue q{{DeviceType::kCPU, 0}, nullptr};
  for (DType dtype : {DType::kBF16, DType::kF32}) {
    for (bool gemma : {false, true}) {
      constexpr int T = 2, H = 7;
      const size_t bytes = T * H * SizeOf(dtype);
      std::vector<uint32_t> a(T * H), b(T * H), residual(T * H),
          output(T * H), n1(T * H), n2(T * H), sum(T * H),
          normalized(T * H), expected(T * H), w1(H), w2(H), w3(H);
      auto set = [&](std::vector<uint32_t>& v, int i, float f) {
        if (dtype == DType::kF32) std::memcpy(reinterpret_cast<char*>(v.data()) + i * 4, &f, 4);
        else { const uint16_t h = F32ToBF16(f); std::memcpy(reinterpret_cast<char*>(v.data()) + i * 2, &h, 2); }
      };
      for (int i = 0; i < T * H; ++i) {
        set(a, i, float(i % 5 - 2) / 3);
        set(b, i, float(i % 7 - 1) / 5);
        set(residual, i, float(i % 3 + 1) / 7);
      }
      for (int i = 0; i < H; ++i) {
        set(w1, i, float(i + 1) / 9); set(w2, i, float(8 - i) / 7); set(w3, i, float(i + 3) / 11);
      }
      auto matrix = [&](std::vector<uint32_t>& v) { return Tensor::Contiguous(v.data(), dtype, q.device, {T, H}); };
      auto weight = [&](std::vector<uint32_t>& v) { return Tensor::Contiguous(v.data(), dtype, q.device, {H}); };
      auto x1 = matrix(a), x2 = matrix(b), r = matrix(residual), out = matrix(output);
      auto p1 = matrix(n1), p2 = matrix(n2), s = matrix(sum), norm = matrix(normalized), ref = matrix(expected);
      auto z1 = weight(w1), z2 = weight(w2), z3 = weight(w3);
      RmsNormArgs args{1e-5, gemma};
      RmsNorm(q, p1, x1, z1, args); RmsNorm(q, p2, x2, z2, args);
      Add(q, s, p1, p2); RmsNorm(q, norm, s, z3, args); Add(q, ref, norm, r);
      for (int alias = 0; alias < 4; ++alias) {
        auto ac = a, bc = b, rc = residual;
        auto ax1 = matrix(ac), ax2 = matrix(bc), ar = matrix(rc);
        Tensor destination = alias == 1 ? ax1 : alias == 2 ? ax2 : alias == 3 ? ar : out;
        DualRmsNormPlusRes(q, destination, ax1, z1, ax2, z2, z3, ar, args);
        if (std::memcmp(destination.data, expected.data(), bytes)) throw std::runtime_error("dual RMS norm result or alias mismatch");
      }
      bool rejected = false; auto bad = x2; bad.shape[1] = H - 1;
      try { DualRmsNormPlusRes(q, out, x1, z1, bad, z2, z3, r, args); } catch (const std::exception&) { rejected = true; }
      if (!rejected) throw std::runtime_error("malformed shape accepted");
    }
  }
  // Keep the broken byte-wrap path safe: its zero-byte allocation reaches this
  // provider, which records dispatch and throws before touching tensor memory.
  RegisterOpProvider(OpId::kRmsNorm, DeviceType::kCPU,
                     {"overflow-test", 1000, nullptr,
                      reinterpret_cast<void*>(static_cast<RmsNormFn>(&RejectOversizedDispatch))});
  float scalar = 1.0F;
  Tensor weight = Tensor::Contiguous(&scalar, DType::kF32, q.device, {4});
  for (int64_t rows : {int64_t{1} << 60, int64_t{1} << 62}) {
    // Set metadata directly so constructing the negative fixture cannot overflow.
    Tensor huge{};
    huge.data = &scalar; huge.dtype = DType::kF32; huge.device = q.device;
    huge.rank = 2; huge.shape[0] = rows; huge.shape[1] = 4;
    huge.stride[0] = 4; huge.stride[1] = 1;
    auto destination = huge;
    oversized_dispatched = false;
    bool rejected = false;
    try {
      DualRmsNormPlusRes(q, destination, huge, weight, huge, weight,
                         weight, huge, RmsNormArgs{1e-5, false});
    } catch (const std::exception&) { rejected = true; }
    if (!rejected || oversized_dispatched)
      throw std::runtime_error("oversized dual RMS norm was not rejected before dispatch");
  }
  std::cout << "BF16/F32 dual RMS norm, Gemma offset, aliases, malformed shape PASS\n";
}
