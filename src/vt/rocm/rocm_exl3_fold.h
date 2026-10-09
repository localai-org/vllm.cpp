#pragma once
// The m=1 GEMV arms' output-transform fold, shared by rocm_exl3.hip (the
// caller) and rocm_exl3_gemv.hip (the kernels).
//
// When the caller passes one and the arm accepts it, the GEMV's last block to
// finish each 128-column group runs step 3 of the chain — had_r_128(raw,
// post_scale = post) into a bf16 `out` — itself, with the same arithmetic as
// HadK<0, 2>, and sets `applied`; the caller then skips its separate output
// HadK launch. `cnt` is n/128 zeroed u32 counters the kernel leaves zeroed.
#include <cstdint>

namespace vt::rocm {

struct Exl3GemvOutFold {
  void* out = nullptr;             // bf16 [n]
  const uint16_t* post = nullptr;  // svh, fp16 [n]
  uint32_t* cnt = nullptr;         // [n / 128], zero on entry and on exit
  float r_scale = 0.0f;
  bool applied = false;
};

}  // namespace vt::rocm
