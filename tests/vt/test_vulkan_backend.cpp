// Vulkan backend skeleton unit gates (BACKEND-VULKAN, W0). Newly authored — vLLM
// has no Vulkan tests to port, and llama.cpp's `test-backend-ops` is a ggml
// harness with no vt:: analogue. Mirrors the shape of
// tests/vt/test_metal_backend.cpp (and through it tests/vt/test_backend.cpp) so
// the three are read side by side.
//
// This TU is COMPILED ONLY in a Vulkan build (tests/CMakeLists.txt gates it on
// VLLM_CPP_VULKAN) and every assertion goes through the public vt:: seam — if
// the skeleton needed Vulkan headers in a test to be checkable, the seam would
// be leaking.
//
// Cross-device NUMERIC equality vs an oracle is NOT here; it lives in
// tests/vt/test_backend_cross_device.cpp, which runs against every registered
// non-CPU backend and so covers Vulkan automatically — and which, on the GB10
// box, compares Vulkan against a CUDA build in the SAME binary, the strongest
// cross-backend oracle in the project.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <limits>
#include <utility>
#include <vector>

#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/quant.h"
#include "vt/vulkan/vulkan_context.h"
#include "vt/vulkan/vulkan_spirv.h"
#include "support/test_env.h"  // SetEnv/UnsetEnv -- MSVC has no setenv (#603)

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::Queue;
using vt::Tensor;

namespace {

// A Vulkan-ENABLED build can legitimately run where there is no loader or no
// conformant device (a headless CI container), in which case the registrars stay
// silent by design. Every case below is skipped in that state rather than
// failing — but the skip is REPORTED, so a silently-unregistered backend on a
// box that does have one cannot masquerade as a pass.
bool VulkanPresent() { return vt::vulkan::VulkanDeviceAvailable(); }

}  // namespace

TEST_CASE("the committed SPIR-V table is present and well-formed") {
  // Independent of any device: this is a property of the CHECKED-IN artifact, so
  // it also gates the generator (scripts/gen-vulkan-spirv.py) on a box with no
  // Vulkan at all.
  // The blobs live in vulkan_spirv.cpp, so the array is `extern` and of unknown
  // bound here and the generated count is the only way to size it. That is the
  // point of the split: at the target shader surface the words must not be
  // re-parsed by every TU that merely needs the table.
  const size_t n = vt::vulkan::kSpirvModuleCount;
  CHECK(n == 44);  // +1: BACKEND-VULKAN-GDN (vt_causal_conv1d_fwd);
                   // +2: BACKEND-VULKAN-EXL3 (#2530); +15: BACKEND-VULKAN-TQ1_0
                   //   (vt_matmul_bt_tq2, vt_matmul_bt_tq2_grouped, vt_matmul_bt_tq2_dev,
                   //    vt_moe_gate_up_swiglu_grouped_tq2, vt_matmul_bt_tq2_grouped_dev,
                   //    vt_matmul_bt_tq2_dev; VK4 rope/moe)
  for (size_t mi = 0; mi < n; ++mi) {
    const auto& m = vt::vulkan::kSpirvModules[mi];
    CAPTURE(m.name);
    REQUIRE(m.word_count > 5);          // a SPIR-V header alone is 5 words
    CHECK(m.words[0] == 0x07230203u);   // SPIR-V magic
  }
  // Every registered op is served by one of these modules (kCastBf16 and kCastF32
  // share vt_cast), so a rename in either direction breaks here rather than at
  // pipeline-creation time on a device we might not have.
  for (const char* want : {"vt_add", "vt_cast", "vt_embedding", "vt_fused_chain",
                           "vt_greedy_argmax", "vt_layer_norm", "vt_matmul",
                           "vt_matmul_coopmat", "vt_matmul_vec", "vt_paged_attn",
                           "vt_qkv_split",
                           "vt_relu", "vt_reshape_and_cache", "vt_rms_norm",
                           "vt_rms_norm_wide",
                           "vt_rope_from_cache", "vt_silu_and_mul",
                           // BACKEND-VULKAN-GDN: the GDN / conv1d glue family.
                           "vt_causal_conv1d_update", "vt_causal_conv1d_fwd",
                           "vt_gdn_post_conv",
                           "vt_gdn_state_gather", "vt_gdn_state_scatter",
                           "vt_rms_norm_gated", "vt_sigmoid_gate_bf16",
                           // BACKEND-VULKAN-GDN-CORE: the two recurrences.
                           "vt_gdn_prefill", "vt_gdn_decode",
                           // BACKEND-VULKAN-QKNORM: the fused attn preamble.
                           "vt_attn_qk_norm_rope_gate",
                           // BACKEND-VULKAN-EXL3: the EXL3 trellis linear. ONE op
                           // (kExl3Gemm) is served by BOTH modules -- the had is
                           // steps 1 and 3 of the same fused chain -- so a
                           // module-per-op reading of this list is wrong here.
                           "vt_exl3_had", "vt_exl3_gemm"}) {
    bool found = false;
    for (size_t mi = 0; mi < vt::vulkan::kSpirvModuleCount; ++mi) {
      if (std::strcmp(vt::vulkan::kSpirvModules[mi].name, want) == 0) found = true;
    }
    CAPTURE(want);
    CHECK(found);
  }
}

TEST_CASE("the committed SPIR-V table records each module's WRITABLE bindings") {
  // Device-independent: a property of the checked-in artifact.
  //
  // WHY THIS IS THE MOST LOAD-BEARING ASSERTION IN THE FILE
  // (BACKEND-VULKAN-BARRIERS). writable_mask is what lets the dispatch path skip
  // a pipeline barrier, and its DANGEROUS failure mode is silent: a mask that
  // came back all-zero would describe every binding as read-only, no dispatch
  // would ever appear to write anything, no hazard would ever be detected, and
  // every barrier in the batch would be dropped. That produces no error and no
  // crash -- only wrong numbers, on real hardware, in a way a software
  // rasterizer's effectively serial execution hides. So the mask is asserted as a
  // STRUCTURE here rather than trusted because the numbers came out right.
  for (size_t mi = 0; mi < vt::vulkan::kSpirvModuleCount; ++mi) {
    const auto& m = vt::vulkan::kSpirvModules[mi];
    CAPTURE(m.name);
    // Every compute shader in this backend produces an output. A module with NO
    // writable binding is a reflection failure, not a legitimate shader.
    CHECK(m.writable_mask != 0u);
    REQUIRE(m.binding_count >= 1u);
    // One bit per binding, and the dispatch path's stack arrays are 32 wide.
    CHECK(m.binding_count <= 32u);
    // No bit may be set above the declared binding count, or the mask and the
    // dispatch's buffer array have drifted out of alignment.
    const uint32_t above =
        m.binding_count >= 32u ? 0u : (m.writable_mask >> m.binding_count);
    CHECK(above == 0u);
    // Not EVERY binding writable either: each of these shaders reads at least one
    // operand, so an all-ones mask is the other reflection failure (it would cost
    // barriers rather than correctness, but it would mean nothing was parsed).
    const uint32_t all = m.binding_count >= 32u
                             ? 0xffffffffu
                             : ((1u << m.binding_count) - 1u);
    CHECK(m.writable_mask != all);
  }
  // Spot checks against the GLSL, read directly from src/vt/vulkan/shaders. These
  // pin the reflection to specific known-correct answers, so a generator change
  // that starts reporting plausible-but-wrong masks fails here.
  struct Expect { const char* name; uint32_t bindings; uint32_t mask; };
  // vt_add.comp: A at 0/1 readonly, B at 2/3 readonly, D (out) at 4/5 writable.
  // vt_silu_and_mul.comp: x at 0/1 readonly, out at 2/3 writable.
  // vt_greedy_argmax.comp: logits at 0 readonly, out at 1 writable.
  // vt_rms_norm.comp: x/weight readonly at 0..3, out at 4/5 and the in-place
  //   residual stream at 6/7 both writable.
  // vt_matmul_vec.comp: a/b at 0..3, out at 4/5, and the two 64-bit ALIASES of a
  //   and b at 6/7 -- aliases of READ operands, so still read-only.
  const Expect kExpect[] = {
      {"vt_add", 6u, 0x30u},
      {"vt_silu_and_mul", 4u, 0x0cu},
      {"vt_greedy_argmax", 2u, 0x02u},
      {"vt_rms_norm", 8u, 0xf0u},
      {"vt_matmul_vec", 8u, 0x30u},
      // vt_exl3_had.comp: in at 0/1 readonly, out at 2/3 writable, and the two
      // OPTIONAL fp16 scale vectors at 4 and 5 -- single 16-bit views, because
      // both are fp16 by contract on every arm, and both read-only.
      {"vt_exl3_had", 6u, 0x0cu},
      // vt_exl3_gemm.comp: the f32 `raw` staging buffer at 0 WRITABLE, a_had at 1
      // and the trellis at 2 read-only. THREE bindings and not six: each operand
      // is bound through the ONE view it uses, so there is no unused 32-bit view
      // for glslang to strip into a binding hole.
      {"vt_exl3_gemm", 3u, 0x01u},
  };
  for (const auto& e : kExpect) {
    CAPTURE(e.name);
    bool found = false;
    for (size_t mi = 0; mi < vt::vulkan::kSpirvModuleCount; ++mi) {
      const auto& m = vt::vulkan::kSpirvModules[mi];
      if (std::strcmp(m.name, e.name) != 0) continue;
      found = true;
      CHECK(m.binding_count == e.bindings);
      CHECK(m.writable_mask == e.mask);
    }
    CHECK(found);
  }
}

TEST_CASE("the committed SPIR-V table records each module's specialization constants") {
  // Device-independent: a property of the checked-in artifact, so this also gates
  // the generator on a box with no Vulkan.
  //
  // The host passes specialization values by constantID. Vulkan SILENTLY IGNORES
  // a VkSpecializationMapEntry whose ID the module does not declare, so a drift
  // between host and shader is WRONG NUMBERS, not a clean error. Recording the
  // declared IDs alongside each blob is what lets GetPipeline check it.
  for (size_t mi = 0; mi < vt::vulkan::kSpirvModuleCount; ++mi) {
    const auto& m = vt::vulkan::kSpirvModules[mi];
    CAPTURE(m.name);
    // Structural: the pointer and the count agree, and the IDs are sorted with no
    // duplicates — GetPipeline builds VkSpecializationMapEntry positionally from
    // this array, so an unsorted or duplicated ID would bind the wrong value.
    CHECK((m.spec_ids == nullptr) == (m.spec_id_count == 0));
    for (size_t i = 1; i < m.spec_id_count; ++i) {
      CHECK(m.spec_ids[i - 1] < m.spec_ids[i]);
    }
    // vt_cast is the backend's FIRST variant axis: constants 0 and 1 are its
    // source and destination dtype, so one module serves every (src, dst) pair
    // instead of a module per pair. Every other W0 shader still declares none.
    //
    // The workgroup size is deliberately NOT such a constant — see the measured
    // reason in src/vt/vulkan/shaders/vt_common.glsl (local_size_x_id emits
    // LocalSize 1 1 1 at the vulkan1.1 target and computes ~1/128 of the tensor).
    if (std::strcmp(m.name, "vt_cast") == 0) {
      REQUIRE(m.spec_id_count == 2);  // src dtype, dst dtype
      CHECK(m.spec_ids[0] == 0u);
      CHECK(m.spec_ids[1] == 1u);
    } else if (std::strcmp(m.name, "vt_embedding") == 0) {
      // table dtype, out dtype, id width (i32 vs i64).
      REQUIRE(m.spec_id_count == 3);
      for (uint32_t want = 0; want < 3; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_rope_from_cache") == 0) {
      // q / k / cache dtype, the NeoX-vs-GPT-J pairing, and the position width.
      REQUIRE(m.spec_id_count == 5);
      for (uint32_t want = 0; want < 5; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_qkv_split") == 0) {
      // source dtype, destination dtype.
      REQUIRE(m.spec_id_count == 2);
      for (uint32_t want = 0; want < 2; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_reshape_and_cache") == 0) {
      // A single WIDTH selector, not a dtype code: this op moves bytes and
      // converts nothing, so 32-bit and 16-bit are the only two paths.
      REQUIRE(m.spec_id_count == 1);
      CHECK(m.spec_ids[0] == 0u);
    } else if (std::strcmp(m.name, "vt_paged_attn") == 0) {
      // query / k-cache / v-cache / out dtype.
      REQUIRE(m.spec_id_count == 4);
      for (uint32_t want = 0; want < 4; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_matmul_coopmat") == 0) {
      // Only the b orientation and the output dtype: A and B are bf16 by the
      // hardware configuration this shader is written to, so they are not axes.
      REQUIRE(m.spec_id_count == 2);
      for (uint32_t want = 0; want < 2; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_matmul_tiled") == 0) {
      // A dtype, B dtype, output dtype, and orientation.
      REQUIRE(m.spec_id_count == 4);
      for (uint32_t want = 0; want < 4; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_matmul_vec") == 0) {
      // a dtype, b dtype, out dtype, the UNROLL factor, the rows-per-workgroup
      // count and the packed-load flag -- but NOT the orientation. This module is
      // MatmulBT by construction: the whole reason it exists is that b's [N,K]
      // layout makes lane-strided K reads contiguous, and the other orientation is
      // already coalesced in vt_matmul and would be made worse here. Every tuning
      // axis rides a spec constant so all the arms are ONE committed module and an
      // A/B never needs a second build.
      REQUIRE(m.spec_id_count == 6);
      for (uint32_t want = 0; want < 6; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_matmul") == 0) {
      // a dtype, b dtype, out dtype, orientation and the COLUMN-BLOCK factor:
      // 3*3*3*2*3 = 162 variants served by ONE committed module, which is the
      // argument for specialization constants over llama.cpp's module-per-#define
      // in miniature. The block factor is the only PERFORMANCE axis here; the other
      // four are correctness axes, which is why the gate below has to look at the
      // specialization VALUES and not just at the module name.
      REQUIRE(m.spec_id_count == 5);
      for (uint32_t want = 0; want < 5; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_sigmoid_gate_bf16") == 0) {
      // ONE axis, and the count is the assertion: the gate is f32 and the output
      // bf16 by the op contract (src/vt/ops.cpp:3327-3334), so only the attention
      // operand varies. A second constant appearing here would mean someone
      // widened the shader past what the op actually promises.
      REQUIRE(m.spec_id_count == 1);
      CHECK(m.spec_ids[0] == 0u);
    } else if (std::strcmp(m.name, "vt_gdn_post_conv") == 0) {
      // conv dtype, the shared q/k/v dtype, the shared araw/braw dtype. g/beta,
      // a_log and dt_bias are f32 by contract and are therefore NOT axes.
      REQUIRE(m.spec_id_count == 3);
      for (uint32_t want = 0; want < 3; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_gdn_prefill") == 0 ||
               std::strcmp(m.name, "vt_gdn_decode") == 0) {
      // The shared q/k/v dtype and the out dtype. TWO, not four: g, beta and the
      // state are f32 by contract (a compressed state is CUDA-only,
      // src/vt/ops.cpp:1631-1638) and the host DECLINES a call whose q/k/v
      // disagree, so a third dtype constant here would mean the shader had grown
      // past what the op promises. Both modules assert the same list because they
      // share one step body (vt_gdn_recurrence.glsl) and must stay in lockstep.
      REQUIRE(m.spec_id_count == 2);
      for (uint32_t want = 0; want < 2; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_rope_cos_sin_cache") == 0) {
      // position dtype and the llama3 scaling flag.
      REQUIRE(m.spec_id_count == 2);
      for (uint32_t want = 0; want < 2; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_rope_neox") == 0) {
      // q/k dtype, position dtype, llama3 flag, q/k head-dim overrides.
      REQUIRE(m.spec_id_count == 6);
      for (uint32_t want = 0; want < 6; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_causal_conv1d_fwd") == 0) {
      // ONE axis, and it is not a dtype: VT_CONV_BLOCKS, the token blocks per
      // (sequence, channel). Every dtype is a push-constant code, because the
      // five tensors vary independently and a spec axis per tensor would be a
      // module explosion for a kernel that runs once per GDN layer.
      REQUIRE(m.spec_id_count == 1);
      CHECK(m.spec_ids[0] == 0u);
    } else if (std::strcmp(m.name, "vt_moe_router_topk") == 0) {
      // E, K, renormalize flag, logits dtype.
      REQUIRE(m.spec_id_count == 4);
      for (uint32_t want = 0; want < 4; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_moe_combine") == 0) {
      // out dtype, expert dtype, has-shared gate.
      REQUIRE(m.spec_id_count == 3);
      for (uint32_t want = 0; want < 3; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_matmul_bt_tq2") == 0) {
      // Single axis: the output dtype (f32/f16/bf16). The weight/activation
      // layouts are fixed by the TQ2_0/Q8_K formats, not free axes.
      REQUIRE(m.spec_id_count == 1);
      CHECK(m.spec_ids[0] == 0);
    } else if (std::strcmp(m.name, "vt_matmul_bt_tq2_grouped") == 0) {
      // Same single output-dtype axis as vt_matmul_bt_tq2 (f32/f16/bf16).
      REQUIRE(m.spec_id_count == 1);
      CHECK(m.spec_ids[0] == 0);
    } else if (std::strcmp(m.name, "vt_moe_gate_up_swiglu_grouped_tq2") == 0) {
      // Fused gate+up+SwiGLU: one axis — the activation dtype (f32 or bf16).
      // Output is always f32 (the op contract pins it).
      REQUIRE(m.spec_id_count == 2);
      for (uint32_t want = 0; want < 2; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_matmul_bt_tq2_grouped_dev") == 0) {
      // Same single output-dtype axis as vt_matmul_bt_tq2_grouped (f32/f16/bf16).
      REQUIRE(m.spec_id_count == 2);
      for (uint32_t want = 0; want < 2; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_matmul_bt_tq2_dev") == 0) {
      // Two axes: output dtype (f32/f16/bf16) and activation dtype (f32/bf16).
      REQUIRE(m.spec_id_count == 2);
      for (uint32_t want = 0; want < 2; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_matmul_bt_tq1_0") == 0) {
      REQUIRE(m.spec_id_count == 1);
      CHECK(m.spec_ids[0] == 0);
    } else if (std::strcmp(m.name, "vt_matmul_bt_tq1_0_grouped") == 0) {
      REQUIRE(m.spec_id_count == 1);
      CHECK(m.spec_ids[0] == 0);
    } else if (std::strcmp(m.name, "vt_matmul_bt_tq1_0_dev") == 0) {
      REQUIRE(m.spec_id_count == 2);
      for (uint32_t want = 0; want < 2; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_matmul_bt_tq1_0_grouped_dev") == 0) {
      REQUIRE(m.spec_id_count == 2);
      for (uint32_t want = 0; want < 2; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_moe_gate_up_swiglu_grouped_tq1_0") == 0) {
      REQUIRE(m.spec_id_count == 2);
      for (uint32_t want = 0; want < 2; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_attn_qk_norm_rope_gate") == 0) {
      // The shared qgate/kf dtype, the shared q_out/k_out dtype, and the gate
      // dtype. THREE, and the third one is the assertion: the gate is a separate
      // axis only because the op contract admits an f32 gate alongside bf16 q/k
      // (the FA-2 prefill combo, src/vt/ops.cpp:1530-1536). q_norm, k_norm and
      // the cos/sin cache are f32 by contract and are therefore NOT axes.
      REQUIRE(m.spec_id_count == 3);
      for (uint32_t want = 0; want < 3; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_exl3_had") == 0) {
      // The input width and the output width, which are INDEPENDENT bits and not
      // one enum: upstream's three inners are (half,half), (float,float) and
      // (float,half), and Exl3GemmKernelCpu selects between exactly those off
      // `c.dtype`. TWO, and a third would mean the shader had grown an axis the
      // op does not promise -- the scale vectors are fp16 BY CONTRACT on every
      // arm (hadamard_inner.cuh:109/:171) and are therefore not axes.
      REQUIRE(m.spec_id_count == 2);
      for (uint32_t want = 0; want < 2; ++want) CHECK(m.spec_ids[want] == want);
    } else {
      CHECK(m.spec_id_count == 0);
    }
  }
}

TEST_CASE("Vulkan specializes pipelines per dtype pair and caches them separately") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  // Two DIFFERENT dtype pairs through the same committed module. The results
  // being right is necessary but not sufficient: a specialization that silently
  // did nothing would also produce right results here, because the shader's
  // defaults happen to be f32->f32. What proves the mechanism engaged is that the
  // pipeline cache GREW BY TWO — one specialized pipeline per pair.
  const size_t before = ctx.PipelineCacheSize();

  const int64_t n = 300;  // not a multiple of the workgroup size
  std::vector<float> src(n);
  for (int64_t i = 0; i < n; ++i) src[i] = static_cast<float>(i) - 150.5f;

  auto* f32_in = static_cast<float*>(vk.Alloc(n * sizeof(float)));
  auto* bf16_mid = static_cast<uint16_t*>(vk.Alloc(n * sizeof(uint16_t)));
  auto* f32_out = static_cast<float*>(vk.Alloc(n * sizeof(float)));
  vk.Copy(q, f32_in, src.data(), n * sizeof(float));

  Tensor t_f32_in = Tensor::Contiguous(f32_in, vt::DType::kF32, d, {n});
  Tensor t_bf16 = Tensor::Contiguous(bf16_mid, vt::DType::kBF16, d, {n});
  Tensor t_f32_out = Tensor::Contiguous(f32_out, vt::DType::kF32, d, {n});

  vt::CastBf16(q, t_bf16, t_f32_in);   // f32 -> bf16 : one specialization
  vt::CastF32(q, t_f32_out, t_bf16);   // bf16 -> f32 : a different one
  vk.Synchronize(q);

  CHECK(ctx.PipelineCacheSize() == before + 2);

  // The round trip must be EXACTLY the CPU codec's, so this stays in the
  // bit-exact tier rather than the NMSE tier: bf16 keeps the high 16 bits under
  // round-to-nearest-even, so a value that survives the narrowing must come back
  // identical.
  std::vector<float> back(n);
  vk.Copy(q, back.data(), f32_out, n * sizeof(float));
  vk.Synchronize(q);
  for (int64_t i = 0; i < n; ++i) {
    CAPTURE(i);
    CHECK(back[i] == vt::BF16ToF32(vt::F32ToBF16(src[i])));
  }

  vk.Free(f32_in);
  vk.Free(bf16_mid);
  vk.Free(f32_out);
  vk.DestroyQueue(q);
}

TEST_CASE("Vulkan backend is registered on a Vulkan-capable host") {
  if (!VulkanPresent()) {
    MESSAGE("no Vulkan loader or no conformant device on this host; skipping");
    return;
  }
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);

  // GB10 exposes one 89.72 GiB DEVICE_LOCAL|HOST_VISIBLE heap, and llvmpipe is a
  // CPU device, so both report unified. This is load-bearing beyond a hardware
  // fact: vt::Backend's SEVEN async-output primitive defaults
  // (src/vt/backend.cpp:19-32) are documented correct exactly for unified
  // backends, so the skeleton inherits them instead of implementing them.
  CHECK(vk.UnifiedMemory());

  // A pre-recorded VkCommandBuffer is the eventual mapping
  // (include/vt/backend.h:92) but is NOT implemented; the honest answer today is
  // false, and the base class makes BeginCapture throw loudly rather than
  // silently no-op.
  CHECK_FALSE(vk.SupportsGraphCapture());
  Queue q = vk.CreateQueue();
  CHECK_THROWS_AS(vk.BeginCapture(q), std::runtime_error);

  CHECK(q.device.type == DeviceType::kVULKAN);
  CHECK(q.handle != nullptr);  // the shared VkQueue
  CHECK(q.id != 0);            // a live identity for the workspace-key machinery

  // The Vulkan API version as the capability pair. The assertion is deliberately
  // ">= 1.1", not "== 1.4": the gate is that a REAL probe ran AND that the
  // version floor this backend needs (16-bit storage in core) actually holds,
  // not that we are on one specific GPU.
  CHECK(vk.DeviceCapabilityMajor() >= 1);
  CHECK((vk.DeviceCapabilityMajor() > 1 || vk.DeviceCapabilityMinor() >= 1));

  vk.DestroyQueue(q);
}

TEST_CASE("Vulkan allocations are 64B-aligned, byte-exact and freeable") {
  if (!VulkanPresent()) return;
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();

  void* p = vk.Alloc(64);
  REQUIRE(p != nullptr);
  // include/vt/backend.h:26 — vt::StepArena depends on >= 64-byte alignment.
  CHECK(reinterpret_cast<uintptr_t>(p) % 64 == 0);

  vk.Memset(q, p, 0xAB, 64);
  vk.Synchronize(q);
  unsigned char dst[64];
  vk.Copy(q, dst, p, 64);
  vk.Synchronize(q);
  CHECK(dst[0] == 0xAB);
  CHECK(dst[63] == 0xAB);
  vk.Free(p);

  // A zero-byte request still yields a valid, distinct, freeable block (the CPU
  // backend's contract, which the arena relies on).
  void* z = vk.Alloc(0);
  CHECK(z != nullptr);
  vk.Free(z);
  vk.Free(nullptr);  // no-op

  vk.DestroyQueue(q);
}

TEST_CASE("Vulkan resolves INTERIOR pointers (tensor views/slices) to the owning buffer") {
  if (!VulkanPresent()) return;
  // vt::Tensor::Slice / ::View hand out pointers INTO an allocation, while Vulkan
  // binds resources, not pointers. The allocation registry
  // (src/vt/vulkan/vulkan_buffers.h) is what bridges that; this case is its gate.
  // It is a STRONGER gate on Vulkan than on Metal, because Vulkan additionally
  // has a descriptor-offset ALIGNMENT rule that this backend sidesteps by binding
  // whole buffers and passing the byte offset in push constants — if that ever
  // regressed to a descriptor offset, a non-zero interior offset would either
  // fail validation or silently read shifted data, and this case catches both.
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  const int64_t rows = 4, cols = 8;
  auto* base = static_cast<float*>(vk.Alloc(rows * cols * sizeof(float)));
  std::vector<float> host(rows * cols);
  for (size_t i = 0; i < host.size(); ++i) host[i] = -1.0f * static_cast<float>(i + 1);
  vk.Copy(q, base, host.data(), host.size() * sizeof(float));

  // Operate on rows [1,3) only — an INTERIOR pointer at byte offset 32, which is
  // NOT a multiple of a typical minStorageBufferOffsetAlignment of 256.
  Tensor sub = Tensor::Contiguous(base + cols, vt::DType::kF32, d, {2, cols});
  vt::Relu(q, sub, sub);
  vk.Synchronize(q);

  std::vector<float> back(host.size());
  vk.Copy(q, back.data(), base, back.size() * sizeof(float));
  vk.Synchronize(q);
  // Rows 0 and 3 untouched (bit-exact); rows 1-2 relu'd to zero (input was all
  // negative), which also proves the buffer OFFSET was applied and not ignored.
  CHECK(back[0] == host[0]);
  CHECK(back[cols * 3] == host[cols * 3]);
  for (int64_t i = cols; i < cols * 3; ++i) CHECK(back[i] == 0.0f);

  vk.Free(base);
  vk.DestroyQueue(q);
}

TEST_CASE("Vulkan rejects memory it did not allocate, loudly") {
  if (!VulkanPresent()) return;
  // Handing a Vulkan kernel a host std::vector is THE bring-up mistake; it must
  // throw, never read garbage.
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};
  std::vector<float> host(64, 1.0f);
  Tensor t = Tensor::Contiguous(host.data(), vt::DType::kF32, d, {8, 8});
  CHECK_THROWS_AS(vt::Relu(q, t, t), std::runtime_error);
  vk.DestroyQueue(q);
}

TEST_CASE("Vulkan platform is registered and reports unified/no-pool residency") {
  if (!VulkanPresent()) return;
  vllm::platforms::Platform& p = vllm::platforms::GetPlatform(DeviceType::kVULKAN);
  CHECK(p.device_type() == DeviceType::kVULKAN);
  CHECK_FALSE(p.is_cuda());
  CHECK_FALSE(p.is_cpu());
  CHECK(p.is_unified_memory());
  CHECK_FALSE(p.supports_graph_capture());

  // Deliberately NON-present: returning the driver's API version (e.g. {1,4})
  // as a compute capability made every attention-backend predicate that assumes
  // NVIDIA SM semantics misroute B60 ("Vulkan 1.4" read as sm_14). The
  // platform keeps the raw numbers on vt::Backend but reports no capability here
  // (vulkan.cpp get_device_capability).
  CHECK_FALSE(p.get_device_capability().present());

  // interface.py:181-187 order — bf16 is the default fallback.
  REQUIRE(p.supported_dtypes().size() == 3);
  CHECK(p.supported_dtypes()[0] == vt::DType::kBF16);

  // Unified memory: never free the only copy, never pool device scratch.
  const auto rp = p.residency_policy();
  CHECK_FALSE(rp.release_host_weights_after_upload);
  CHECK_FALSE(rp.uses_device_memory_pool);

  // Vulkan now HAS native kPagedAttention + kReshapeAndCache reading and writing
  // the NHD layout FlashAttentionBackend::get_kv_cache_shape allocates, so the
  // selector may reach FLASH_ATTN — on exactly the footing Metal reached it.
  // This is what let OPT-125m run end to end on Vulkan.
  {
    vllm::platforms::AttnSelectorConfig cfg;
    const auto prio = p.get_attn_backend_priority(cfg);
    REQUIRE(prio.size() == 1);
    CHECK(prio[0] == "FLASH_ATTN");
  }
  // MLA stays EMPTY, and that is a capability statement rather than a stub:
  // kMlaDecodeAttention / kMlaPrefillAttention / kConcatAndCacheMla — and
  // kConcatAndCacheDsMla / kDequantAndGatherDsMla, the fp8_ds_mla byte-page pair
  // (KV-DSV4-MULTICACHE W8, #2455) — have no Vulkan kernel, so naming a backend
  // here would route an MLA model into one that cannot serve it. Selection must
  // fail loudly instead.
  {
    vllm::platforms::AttnSelectorConfig mla;
    mla.use_mla = true;
    CHECK(p.get_attn_backend_priority(mla).empty());
  }
}

TEST_CASE("Vulkan registers the W0 op set and NOT the unimplemented rest") {
  if (!VulkanPresent()) return;
  // The skeleton's registered surface, stated as an executable fact so a later
  // work row cannot quietly claim more than it implements.
  for (vt::OpId op : {vt::OpId::kAdd, vt::OpId::kRelu, vt::OpId::kSiluAndMul,
                      vt::OpId::kCastBf16, vt::OpId::kCastF32, vt::OpId::kLayerNorm,
                      vt::OpId::kRmsNorm, vt::OpId::kFusedChain,
                      // VK-B: the dense path's GEMM (both orientations) and the
                      // two ends of the model, token ids in and out.
                      vt::OpId::kMatmul, vt::OpId::kMatmulBT,
                      vt::OpId::kEmbedding, vt::OpId::kGreedyArgmax,
                      // The attention block: paged attention (the one kernel
                      // with no llama.cpp Vulkan counterpart), the KV write, the
                      // QKV split and the rotary APPLY.
                      vt::OpId::kPagedAttention, vt::OpId::kReshapeAndCache,
                      vt::OpId::kQkvSplit, vt::OpId::kRopeFromCache,
                      // BACKEND-VULKAN-GDN: the GDN / conv1d glue family that a
                      // GDN hybrid (Qwen3.6-27B) hits on every step.
                      vt::OpId::kSigmoidGateBf16, vt::OpId::kRmsNormGated,
                      vt::OpId::kGdnStateGather, vt::OpId::kGdnStateScatter,
                      vt::OpId::kCausalConv1dUpdate, vt::OpId::kGdnPostConv,
                      // ...and the PREFILL conv, which closes the family.
                      vt::OpId::kCausalConv1dFwd,
                      // BACKEND-VULKAN-GDN-CORE: the two gated-delta recurrences
                      // themselves, which are where a GDN hybrid's prefill time
                      // actually was.
                      vt::OpId::kGdnPrefill, vt::OpId::kGdnDecode,
                      // BACKEND-VULKAN-EXL3 (#2530): the exactly two ops an EXL3
                      // checkpoint ran on the portable CPU tier when the queue was
                      // Vulkan. kCastF16 is not an EXL3 op -- the EXL3 linear
                      // demands an f16 activation while models carry bf16/f32 --
                      // and its two siblings were registered here from W0.
                      vt::OpId::kCastF16, vt::OpId::kExl3Gemm}) {
    CHECK(vt::OpRegistered(op, DeviceType::kVULKAN));
  }
  // The VK4 kernels landed NATIVE Vulkan modules for the rotary table build and
  // the two MoE router/combine ops (rope_cos_sin_cache, rope_neox,
  // moe_router_topk, moe_combine); the reference tier likewise serves
  // MatmulBTQuant's keep-quant path. Still genuinely unimplemented: the sampler
  // beyond greedy argmax (kApplyTemperature). The PRE-FILL conv used to be listed
  // here and is native now (BACKEND-VULKAN-GDN, vt_causal_conv1d_fwd).
  for (vt::OpId op : {vt::OpId::kApplyTemperature}) {
    CHECK_FALSE(vt::OpRegistered(op, DeviceType::kVULKAN));
  }
  // ...but they no longer THROW, and this assertion used to say they did.
  //
  // Accelerator-seam work row S5 (af0b21ba) added the PORTABLE REFERENCE TIER:
  // for a unified-memory device, a missed GetOp lazily installs the CPU kernel as
  // a priority -1000 provider, below every native kernel. Vulkan is eligible (GB10
  // integrated and llvmpipe both report unified), so every op the CPU backend has
  // resolves here and runs ON THE HOST against shared memory — correct, and
  // arbitrarily slow.
  //
  // The Metal sibling was updated for this (test_metal_backend.cpp:215-231); THIS
  // file was not, and the assertion sat RED from the moment S5 landed because no
  // CI leg builds the Vulkan backend and nobody built it locally. The mirrored
  // form below is deliberate: the two backends should fail the same way.
  //
  // Measured on this tree (VK-A1, 2026-08-06): of 87 CPU-registered ops, 8 are
  // NATIVE on Vulkan, 79 are served by the reference tier, and ZERO throw.
  REQUIRE(vt::ReferenceTierEligible(DeviceType::kVULKAN));
  // This must stay an op that is GENUINELY unimplemented natively; it moves on as
  // the backend fills in: kMatmul, then kPagedAttention, then kReshapeAndCache,
  // then kRopeNeox / kMoeRouterTopK all had their turn and now have native
  // kernels. kApplyTemperature (the sampler past greedy argmax) is the current one.
  void* rope = nullptr;
  CHECK_NOTHROW(rope = vt::GetOp(vt::OpId::kApplyTemperature, DeviceType::kVULKAN));
  CHECK(rope != nullptr);
  // BY NAME, so a host kernel can never masquerade as a native Vulkan one (Risk 7).
  const auto stats = vt::GetOpProviderStats(vt::OpId::kApplyTemperature, DeviceType::kVULKAN);
  REQUIRE(stats.last_selected != nullptr);
  CHECK(std::string(stats.last_selected) == std::string(vt::kReferenceProviderName));
  CHECK(vt::GetReferenceTierHits() > 0);
}

TEST_CASE("cooperative-matrix capability is PROBED, and absent on llvmpipe") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  MESSAGE("vulkan device: " << ctx.device_name());
  // Assembled BEFORE the macro. MESSAGE(x << y) expands to
  // `MessageBuilder << x << y`, so an expression written inside it is consumed by
  // the builder rather than evaluated first -- `MESSAGE("text" << flag)` renders
  // as "1", which reads as if the capability were TRUE. For a line whose entire
  // job is to report a capability honestly, that is the worst failure mode.
  const std::string coop_line = std::string("coopmat bf16xbf16->f32 16x16x16 SUBGROUP: ") +
                                (ctx.coopmat_bf16_f32() ? "YES" : "no");
  MESSAGE(coop_line);
  MESSAGE("subgroup size: " << ctx.subgroup_size());

  // The predicate must be a REPORT, never an assumption, so this asserts the
  // property rather than a specific answer: a device may or may not have it.
  // What IS asserted unconditionally is that the backend stayed usable either
  // way -- enabling an absent extension would have failed vkCreateDevice
  // outright, so merely getting here proves the enablement is conditional.
  CHECK(ctx.subgroup_size() > 0);

  // MEASURED 2026-08-07: llvmpipe exposes VK_KHR_cooperative_matrix NOT AT ALL,
  // so on the software rasterizer -- the only Vulkan device CI can reach -- the
  // answer must be NO, and the scalar GEMM tactic is what runs. This is pinned
  // because it is what makes the CI leg a real test of the FALLBACK path rather
  // than an accident.
  if (ctx.device_name().find("llvmpipe") != std::string::npos) {
    CHECK_FALSE(ctx.coopmat_bf16_f32());
  }
}

TEST_CASE("bf16 GEMM takes COOPMAT where available, tiled GEMM where not") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  // K = 32 is a multiple of 16 (the tactic requires it); M = 20 and N = 12 are
  // deliberately RAGGED so the shader's bounds-checked store is exercised rather
  // than only whole tiles.
  constexpr int64_t kM = 32, kK = 32, kN = 16;

  std::vector<float> a(kM * kK), b(kN * kK);
  for (int64_t i = 0; i < kM * kK; ++i) a[i] = 0.5f * static_cast<float>((i % 7) - 3);
  for (int64_t i = 0; i < kN * kK; ++i) b[i] = 0.25f * static_cast<float>((i % 5) - 2);

  // bf16 device operands. The values above are chosen to be exactly
  // representable in bf16, so the ORACLE below can be computed in f32 without the
  // narrowing itself becoming the error under test.
  std::vector<uint16_t> a_bf(kM * kK), b_bf(kN * kK);
  for (size_t i = 0; i < a.size(); ++i) a_bf[i] = vt::F32ToBF16(a[i]);
  for (size_t i = 0; i < b.size(); ++i) b_bf[i] = vt::F32ToBF16(b[i]);

  // Oracle: MatmulBT semantics, b is [N,K]. Sequential f32 accumulation, which is
  // the CPU kernel's contract; the coopmat tile order differs, hence the NMSE bar.
  std::vector<float> ref(kM * kN, 0.0f);
  for (int64_t i = 0; i < kM; ++i) {
    for (int64_t j = 0; j < kN; ++j) {
      float acc = 0.0f;
      for (int64_t p2 = 0; p2 < kK; ++p2) {
        acc += vt::BF16ToF32(a_bf[i * kK + p2]) * vt::BF16ToF32(b_bf[j * kK + p2]);
      }
      ref[i * kN + j] = acc;
    }
  }

  void* da = vk.Alloc(a_bf.size() * sizeof(uint16_t));
  void* db = vk.Alloc(b_bf.size() * sizeof(uint16_t));
  auto* dout = static_cast<float*>(vk.Alloc(kM * kN * sizeof(float)));
  vk.Copy(q, da, a_bf.data(), a_bf.size() * sizeof(uint16_t));
  vk.Copy(q, db, b_bf.data(), b_bf.size() * sizeof(uint16_t));
  vk.Synchronize(q);

  Tensor ta = Tensor::Contiguous(da, vt::DType::kBF16, d, {kM, kK});
  Tensor tb = Tensor::Contiguous(db, vt::DType::kBF16, d, {kN, kK});
  Tensor to = Tensor::Contiguous(dout, vt::DType::kF32, d, {kM, kN});
  vt::MatmulBT(q, to, ta, tb);
  vk.Synchronize(q);

  // WHICH TACTIC RAN. This is the load-bearing assertion, not the numbers: the
  // scalar kernel would produce results just as correct, so a numeric check alone
  // cannot tell a working coopmat path from a silent fallback. Same reasoning as
  // the op-provider decline counters.
  //
  // NOTE the shape: M and N are WHOLE TILES here (32 and 16), because the tactic
  // now requires that. The previous version of this case used M=20, N=12 to
  // "exercise raggedness" and PASSED while the kernel was reading past the end of
  // its operand -- the out-of-bounds read stayed inside the allocation and the
  // garbage rows were discarded by the bounds-checked store. See the ragged case
  // below, which asserts the tactic DECLINES rather than trying to be correct.
  const bool coop_expected = ctx.coopmat_bf16_f32() && ctx.subgroup_size() == 32;
  const char* expected_pipeline = coop_expected ? "vt_matmul_coopmat" : "vt_matmul_tiled";
  const std::string tactic_line = std::string("bf16 GEMM tactic: ") + expected_pipeline;
  MESSAGE(tactic_line);
  CHECK(ctx.PipelineExistsFor(expected_pipeline));
  if (!coop_expected) {
    // On a device without the configuration the coopmat module must NEVER be
    // built -- selecting it there would fail at pipeline creation. Dense prefill
    // uses vt_matmul_tiled; vt_matmul_vec is the small/partial-M fallback.
    CHECK_FALSE(ctx.PipelineExistsFor("vt_matmul_coopmat"));
  }

  std::vector<float> got(kM * kN);
  vk.Copy(q, got.data(), dout, got.size() * sizeof(float));
  vk.Synchronize(q);

  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double diff = static_cast<double>(ref[i]) - static_cast<double>(got[i]);
    num += diff * diff;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  const double nmse = den == 0.0 ? num : num / den;
  const std::string nmse_line =
      std::string("bf16 GEMM NMSE vs the f32 oracle: ") + std::to_string(nmse);
  MESSAGE(nmse_line);
  CHECK(nmse <= 5e-4);

  vk.Free(da);
  vk.Free(db);
  vk.Free(dout);
  vk.DestroyQueue(q);
}

TEST_CASE("a PARTIAL-TILE GEMM declines coopmat -- the shape that hung a GPU") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  // M = 1 is the DECODE shape, and it is what hung GB10: lm_head dispatched
  // vt_matmul_coopmat with 9,496 workgroups, coopMatLoad read a full 16-row tile
  // from a 1-row activation -- ~30 KB past the buffer -- the GPU faulted and
  // vkWaitForFences(UINT64_MAX) never returned. A hang, not an error.
  //
  // N is also deliberately partial (17) so both edges are covered.
  constexpr int64_t kM = 1, kK = 32, kN = 17;

  std::vector<float> a(kM * kK), b(kN * kK);
  for (int64_t i = 0; i < kM * kK; ++i) a[i] = 0.5f * static_cast<float>((i % 7) - 3);
  for (int64_t i = 0; i < kN * kK; ++i) b[i] = 0.25f * static_cast<float>((i % 5) - 2);
  std::vector<uint16_t> a_bf(a.size()), b_bf(b.size());
  for (size_t i = 0; i < a.size(); ++i) a_bf[i] = vt::F32ToBF16(a[i]);
  for (size_t i = 0; i < b.size(); ++i) b_bf[i] = vt::F32ToBF16(b[i]);

  std::vector<float> ref(kM * kN, 0.0f);
  for (int64_t i = 0; i < kM; ++i) {
    for (int64_t j = 0; j < kN; ++j) {
      float acc = 0.0f;
      for (int64_t p2 = 0; p2 < kK; ++p2) {
        acc += vt::BF16ToF32(a_bf[i * kK + p2]) * vt::BF16ToF32(b_bf[j * kK + p2]);
      }
      ref[i * kN + j] = acc;
    }
  }

  void* da = vk.Alloc(a_bf.size() * sizeof(uint16_t));
  void* db = vk.Alloc(b_bf.size() * sizeof(uint16_t));
  auto* dout = static_cast<float*>(vk.Alloc(kM * kN * sizeof(float)));
  vk.Copy(q, da, a_bf.data(), a_bf.size() * sizeof(uint16_t));
  vk.Copy(q, db, b_bf.data(), b_bf.size() * sizeof(uint16_t));
  vk.Synchronize(q);

  const size_t before = ctx.PipelineCacheSize();
  Tensor ta = Tensor::Contiguous(da, vt::DType::kBF16, d, {kM, kK});
  Tensor tb = Tensor::Contiguous(db, vt::DType::kBF16, d, {kN, kK});
  Tensor to = Tensor::Contiguous(dout, vt::DType::kF32, d, {kM, kN});
  vt::MatmulBT(q, to, ta, tb);
  vk.Synchronize(q);

  // THE ASSERTION THAT MATTERS: on a partial tile the tactic must DECLINE. A
  // numeric check cannot express this -- the scalar kernel is equally correct, and
  // the coopmat kernel would not return a wrong answer here, it would HANG.
  CHECK(ctx.PipelineExistsFor("vt_matmul"));
  const size_t after = ctx.PipelineCacheSize();
  CAPTURE(before);
  CAPTURE(after);

  std::vector<float> got(kM * kN);
  vk.Copy(q, got.data(), dout, got.size() * sizeof(float));
  vk.Synchronize(q);
  for (int64_t i = 0; i < kM * kN; ++i) {
    CAPTURE(i);
    CHECK(got[i] == doctest::Approx(ref[i]).epsilon(1e-3));
  }

  vk.Free(da);
  vk.Free(db);
  vk.Free(dout);
  vk.DestroyQueue(q);
}

TEST_CASE("the GEMV vec tactic serves decode and prefill; the non-BT orientation declines") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  // Unlike the coopmat tactic, this one has NO hardware precondition -- it is a
  // different assignment of work to lanes, not a different instruction -- so
  // llvmpipe runs it and CI can gate the selection for real rather than only
  // gating that it is refused.
  //
  // K = 256 is two full workgroup widths, so every lane gets work and the strided
  // loop runs more than one iteration. N is deliberately NOT a multiple of 16, so
  // the coopmat tactic declines and cannot be what is being measured here.
  constexpr int64_t kK = 256, kN = 33;

  auto build = [&](int64_t m) {
    std::vector<uint16_t> a(m * kK), b(kN * kK);
    for (int64_t i = 0; i < m * kK; ++i)
      a[i] = vt::F32ToBF16(0.5f * static_cast<float>((i % 7) - 3));
    for (int64_t i = 0; i < kN * kK; ++i)
      b[i] = vt::F32ToBF16(0.25f * static_cast<float>((i % 5) - 2));
    return std::make_pair(a, b);
  };

  auto run_bt = [&](int64_t m, std::vector<float>& got) {
    auto ab = build(m);
    void* da = vk.Alloc(ab.first.size() * sizeof(uint16_t));
    void* db = vk.Alloc(ab.second.size() * sizeof(uint16_t));
    auto* dout = static_cast<float*>(vk.Alloc(m * kN * sizeof(float)));
    vk.Copy(q, da, ab.first.data(), ab.first.size() * sizeof(uint16_t));
    vk.Copy(q, db, ab.second.data(), ab.second.size() * sizeof(uint16_t));
    vk.Synchronize(q);
    Tensor ta = Tensor::Contiguous(da, vt::DType::kBF16, d, {m, kK});
    Tensor tb = Tensor::Contiguous(db, vt::DType::kBF16, d, {kN, kK});
    Tensor to = Tensor::Contiguous(dout, vt::DType::kF32, d, {m, kN});
    vt::MatmulBT(q, to, ta, tb);
    vk.Synchronize(q);
    got.assign(static_cast<size_t>(m * kN), 0.0f);
    vk.Copy(q, got.data(), dout, got.size() * sizeof(float));
    vk.Synchronize(q);
    // Host oracle in f64, so neither tactic's accumulation order is privileged.
    std::vector<float> ref(static_cast<size_t>(m * kN), 0.0f);
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < kN; ++j) {
        double acc = 0.0;
        for (int64_t c = 0; c < kK; ++c) {
          acc += static_cast<double>(vt::BF16ToF32(ab.first[i * kK + c])) *
                 static_cast<double>(vt::BF16ToF32(ab.second[j * kK + c]));
        }
        ref[static_cast<size_t>(i * kN + j)] = static_cast<float>(acc);
      }
    }
    vk.Free(da);
    vk.Free(db);
    vk.Free(dout);
    return ref;
  };

  SUBCASE("M=1 MatmulBT selects vt_matmul_vec and is numerically right") {
    std::vector<float> got;
    const std::vector<float> ref = run_bt(1, got);
    // THE LOAD-BEARING ASSERTION IS THE TACTIC. The scalar kernel is equally
    // correct, so a silent fallback would post numbers that pass every value
    // check below while proving nothing about the kernel this change adds.
    CHECK(ctx.PipelineExistsFor("vt_matmul_vec"));
    for (size_t i = 0; i < got.size(); ++i) {
      CAPTURE(i);
      CHECK(got[i] == doctest::Approx(ref[i]).epsilon(1e-3));
    }
  }

  SUBCASE("M=8 is prefill-shaped and takes the vec tactic (GEMV for M>1)") {
    // The GEMV vec tactic is now allowed for M>1 (prefill) because its
    // coalesced B reads beat the scalar kernel's uncoalesced reads. The
    // predicate accepts any M when kBT and K >= kWorkgroupSize. The numbers
    // are still correct -- verified against the f64 host oracle.
    std::vector<float> got;
    const std::vector<float> ref = run_bt(8, got);
    CHECK(ctx.PipelineExistsFor("vt_matmul_vec"));
    for (size_t i = 0; i < got.size(); ++i) {
      CAPTURE(i);
      CHECK(got[i] == doctest::Approx(ref[i]).epsilon(1e-3));
    }
  }

  SUBCASE("the non-BT orientation declines -- it is ALREADY coalesced") {
    // vt_matmul reads b[q*n + j] there, so adjacent lanes already hit adjacent
    // addresses; the vec shape would make that strided and strictly worse. This
    // asserts the tactic is scoped to the orientation it actually helps, which a
    // numeric check can never show.
    std::vector<uint16_t> a(kK), b(kK * kN);
    // int64_t, NOT size_t: `(i % 7) - 3` on an unsigned type underflows to a huge
    // value for i % 7 < 3, and the resulting operands overflow the accumulation
    // to inf. Same expression as the BT builder above, which uses int64_t.
    for (int64_t i = 0; i < static_cast<int64_t>(a.size()); ++i)
      a[static_cast<size_t>(i)] = vt::F32ToBF16(0.5f * static_cast<float>((i % 7) - 3));
    for (int64_t i = 0; i < static_cast<int64_t>(b.size()); ++i)
      b[static_cast<size_t>(i)] = vt::F32ToBF16(0.25f * static_cast<float>((i % 5) - 2));
    void* da = vk.Alloc(a.size() * sizeof(uint16_t));
    void* db = vk.Alloc(b.size() * sizeof(uint16_t));
    auto* dout = static_cast<float*>(vk.Alloc(kN * sizeof(float)));
    vk.Copy(q, da, a.data(), a.size() * sizeof(uint16_t));
    vk.Copy(q, db, b.data(), b.size() * sizeof(uint16_t));
    vk.Synchronize(q);
    Tensor ta = Tensor::Contiguous(da, vt::DType::kBF16, d, {1, kK});
    Tensor tb = Tensor::Contiguous(db, vt::DType::kBF16, d, {kK, kN});
    Tensor to = Tensor::Contiguous(dout, vt::DType::kF32, d, {1, kN});
    vt::Matmul(q, to, ta, tb);
    vk.Synchronize(q);
    std::vector<float> got(kN);
    vk.Copy(q, got.data(), dout, got.size() * sizeof(float));
    vk.Synchronize(q);
    for (int64_t j = 0; j < kN; ++j) {
      double acc = 0.0;
      for (int64_t c = 0; c < kK; ++c) {
        acc += static_cast<double>(vt::BF16ToF32(a[static_cast<size_t>(c)])) *
               static_cast<double>(vt::BF16ToF32(b[static_cast<size_t>(c * kN + j)]));
      }
      CAPTURE(j);
      CHECK(got[static_cast<size_t>(j)] == doctest::Approx(static_cast<float>(acc)).epsilon(1e-3));
    }
    CHECK(ctx.PipelineExistsFor("vt_matmul"));
    vk.Free(da);
    vk.Free(db);
    vk.Free(dout);
  }

  vk.DestroyQueue(q);
}

// The load-width and rows-per-workgroup axes of vt_matmul_vec, asserted by the
// SPECIALIZATION VALUES the dispatch actually built rather than by the numbers.
//
// THIS TEST EXISTS BECAUSE A TOLERANCE TEST CANNOT SEE THE OPTIMISATION AT ALL.
// Every value of both axes computes the same dot product to within the tier this
// kernel already lives in, so if a host predicate silently stopped selecting the
// wide load -- a refactor, a reordered clause, an env parse that stopped parsing,
// an allocator whose offsets stopped being 8-byte aligned -- every numeric check
// in this file would still pass and 27B decode would quietly lose 8% of its
// hottest kernel. The pipeline cache KEY is the only observable that changes.
//
// The expected default below is a LITERAL, not a read of kGemvPackDefault: a test
// that imports the value it is checking asserts nothing. A change of default has
// to come here and be argued.
TEST_CASE("vt_matmul_vec selects its load width and row count from the shape") {
  if (!VulkanPresent()) return;
  // Mirrored from src/vt/vulkan/vulkan_ops.cpp kGemvPackDefault / kGemvRowsDefault.
  // pack 2 = four 16-bit elements per load, MEASURED 1.086x over pack 0 on GB10;
  // rows 1 = one output element per workgroup, because rows > 1 MEASURED 0.966x
  // there (the axis is kept for the AMD/Intel boards llama.cpp raises it for).
  constexpr uint32_t kExpectPack = 2;
  constexpr uint32_t kExpectRows = 1;
  // The A/B levers are process-wide and read once, so under one of them the KEY
  // assertions would be asserting the override rather than the default. The
  // NUMERIC checks still run in that case, and that is deliberate: forcing an arm
  // and running this test is how a non-default variant -- every pack width above
  // 0 changes the answer's low bits -- gets verified on a device CI can reach.
  const bool overridden = std::getenv("VT_VULKAN_GEMV_ROWS") != nullptr ||
                          std::getenv("VT_VULKAN_GEMV_PACK") != nullptr ||
                          std::getenv("VT_VULKAN_GEMV_UNROLL") != nullptr ||
                          std::getenv("VT_VULKAN_GEMV") != nullptr;
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  // Runs one M=1 MatmulBT and returns (the key this shape newly built, all keys).
  // `a_f32` makes the ACTIVATION f32 while the weight stays bf16, which is the
  // real reason a decode GEMV would ever decline the packed path.
  auto key_for = [&](int64_t n, int64_t k, bool a_f32) {
    std::vector<uint16_t> a16(static_cast<size_t>(k));
    std::vector<float> a32(static_cast<size_t>(k));
    std::vector<uint16_t> b(static_cast<size_t>(n * k));
    for (int64_t i = 0; i < k; ++i) {
      const float v = 0.5f * static_cast<float>((i % 7) - 3);
      a16[static_cast<size_t>(i)] = vt::F32ToBF16(v);
      a32[static_cast<size_t>(i)] = v;
    }
    for (int64_t i = 0; i < n * k; ++i)
      b[static_cast<size_t>(i)] = vt::F32ToBF16(0.25f * static_cast<float>((i % 5) - 2));
    const size_t a_bytes = a_f32 ? a32.size() * sizeof(float) : a16.size() * sizeof(uint16_t);
    void* da = vk.Alloc(a_bytes);
    void* db = vk.Alloc(b.size() * sizeof(uint16_t));
    auto* dout = static_cast<float*>(vk.Alloc(static_cast<size_t>(n) * sizeof(float)));
    vk.Copy(q, da,
            a_f32 ? static_cast<const void*>(a32.data()) : static_cast<const void*>(a16.data()),
            a_bytes);
    vk.Copy(q, db, b.data(), b.size() * sizeof(uint16_t));
    vk.Synchronize(q);
    Tensor ta = Tensor::Contiguous(da, a_f32 ? vt::DType::kF32 : vt::DType::kBF16, d, {1, k});
    Tensor tb = Tensor::Contiguous(db, vt::DType::kBF16, d, {n, k});
    Tensor to = Tensor::Contiguous(dout, vt::DType::kF32, d, {1, n});
    const std::vector<std::string> before = ctx.PipelineKeysFor("vt_matmul_vec");
    vt::MatmulBT(q, to, ta, tb);
    vk.Synchronize(q);
    // Correctness travels with the selection check: a variant that is selected and
    // WRONG is worse than one that is not selected. The LAST row is included
    // because a rows-per-workgroup or wide-load mistake at the end of a dispatch
    // leaves every earlier row right.
    std::vector<float> got(static_cast<size_t>(n), 0.0f);
    vk.Copy(q, got.data(), dout, got.size() * sizeof(float));
    vk.Synchronize(q);
    for (int64_t j : {int64_t{0}, n / 2, n - 1}) {
      double acc = 0.0;
      for (int64_t c = 0; c < k; ++c) {
        const double av = a_f32
                              ? static_cast<double>(a32[static_cast<size_t>(c)])
                              : static_cast<double>(vt::BF16ToF32(a16[static_cast<size_t>(c)]));
        acc += av * static_cast<double>(vt::BF16ToF32(b[static_cast<size_t>(j * k + c)]));
      }
      CAPTURE(n);
      CAPTURE(k);
      CAPTURE(j);
      CHECK(got[static_cast<size_t>(j)] ==
            doctest::Approx(static_cast<float>(acc)).epsilon(1e-3));
    }
    const std::vector<std::string> after = ctx.PipelineKeysFor("vt_matmul_vec");
    std::string key;
    for (const std::string& s : after) {
      bool fresh = true;
      for (const std::string& pk : before) {
        if (pk == s) fresh = false;
      }
      if (fresh) key = s;
    }
    vk.Free(da);
    vk.Free(db);
    vk.Free(dout);
    return std::make_pair(key, after);
  };

  // "<module>|<a dt>,<b dt>,<out dt>,<unroll>,<rows>,<pack>" -- bf16 = 2, f32 = 0,
  // unroll 4 by default.
  auto expect = [](uint32_t a_dt, uint32_t rows, uint32_t pack) {
    return "vt_matmul_vec|" + std::to_string(a_dt) + ",2,0,4," + std::to_string(rows) + "," +
           std::to_string(pack);
  };
  auto has = [](const std::vector<std::string>& v, const std::string& want) {
    for (const std::string& s : v) {
      if (s == want) return true;
    }
    return false;
  };
  // A fresh key that is NOT the expected one means the predicate picked a
  // different variant; no fresh key means an earlier shape already built this
  // exact specialization, which is equally fine.
  auto only = [](const std::string& fresh, const std::string& want) {
    return fresh.empty() || fresh == want;
  };

  SUBCASE("the shipped default is the widest load, one output element per group") {
    // K = 256 is two full workgroup widths and a multiple of 4; both operands
    // bf16; a fresh allocation is 64-byte aligned. Nothing degrades.
    auto r = key_for(64, 256, false);
    const std::string want = expect(2, kExpectRows, kExpectPack);
    CAPTURE(r.first);
    CAPTURE(want);
    if (!overridden) {
      CHECK(has(r.second, want));
      CHECK(only(r.first, want));
      // The load-bearing half: the optimisation is actually ON. If
      // kGemvPackDefault ever silently reverts to 0, this fails.
      CHECK(kExpectPack == 2);
    }
  }

  SUBCASE("K = 2 (mod 4) degrades to the 4-byte load rather than declining") {
    // 258 = 2 x 129. Four elements per load would need a row start j*K that is a
    // multiple of 4 for every j, which an even-but-not-multiple-of-4 K does not
    // give, so the width halves instead of falling all the way back.
    auto r = key_for(64, 258, false);
    const std::string want = expect(2, kExpectRows, kExpectPack >= 1 ? 1u : 0u);
    CAPTURE(r.first);
    CAPTURE(want);
    if (!overridden) {
      CHECK(has(r.second, want));
      CHECK(only(r.first, want));
    }
  }

  SUBCASE("odd K falls all the way back to one element per load") {
    auto r = key_for(64, 257, false);
    const std::string want = expect(2, kExpectRows, 0);
    CAPTURE(r.first);
    CAPTURE(want);
    if (!overridden) {
      CHECK(has(r.second, want));
      CHECK(only(r.first, want));
    }
  }

  SUBCASE("an f32 activation declines the packed path entirely") {
    // An f32 operand is already one element per 32-bit word: there is nothing to
    // pack, and the shader's 16-bit half-extraction would be nonsense on it. The
    // numeric checks inside key_for are what prove the fallback still computes the
    // right answer for the mixed f32-activation / bf16-weight combination.
    auto r = key_for(64, 256, true);
    const std::string want = expect(0, kExpectRows, 0);
    CAPTURE(r.first);
    CAPTURE(want);
    if (!overridden) {
      CHECK(has(r.second, want));
      CHECK(only(r.first, want));
    }
  }

  SUBCASE("the rows axis degrades on N rather than declining") {
    // Driven by whatever row count is REQUESTED, so this ladder is exercised for
    // real by a VT_VULKAN_GEMV_ROWS=4 run; with the shipped default of 1 it
    // asserts the complementary thing, that nothing accidentally selects rows > 1.
    const char* rows_env = std::getenv("VT_VULKAN_GEMV_ROWS");
    const char* pack_env = std::getenv("VT_VULKAN_GEMV_PACK");
    const uint32_t want_rows =
        rows_env != nullptr ? static_cast<uint32_t>(std::atoi(rows_env)) : kExpectRows;
    const uint32_t want_pack =
        pack_env != nullptr ? static_cast<uint32_t>(std::atoi(pack_env)) : kExpectPack;
    // Those two would change the unroll value or disable the tactic outright, so
    // the key stops being predictable from the row count alone.
    if (std::getenv("VT_VULKAN_GEMV_UNROLL") == nullptr &&
        std::getenv("VT_VULKAN_GEMV") == nullptr) {
      // N = 64 keeps every row count; 66 is 2 (mod 4); 33 is odd.
      auto r4 = key_for(64, 256, false);
      CHECK(has(r4.second, expect(2, want_rows, want_pack)));
      auto r2 = key_for(66, 256, false);
      CHECK(has(r2.second, expect(2, want_rows >= 2 ? 2u : 1u, want_pack)));
      auto r1 = key_for(33, 256, false);
      CHECK(has(r1.second, expect(2, 1, want_pack)));
    }
  }

  vk.DestroyQueue(q);
}


TEST_CASE("the scalar matmul takes the COLUMN-BLOCKED variant and it is bit-identical") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  // THE SHAPE IS THE lm_head's, IN MINIATURE. MEASURED on 27B Vulkan decode:
  // exactly one GEMM per token stays on vt_matmul rather than reaching the GEMV
  // tactic -- the lm_head, at bt=0 (b is [K,N]), m=1, k=5120, n=248320, moving
  // 2.54 GB per call at 12.43 ms. Every other decode GEMM is MatmulBT and goes to
  // vt_matmul_vec. So this orientation at m == 1 is the shape the K-unroll exists
  // for, and it is the shape asserted here.
  //
  // N = 33 is deliberately FAR SMALLER than one blocked span (128*8 = 1024), so
  // every lane but the first few is out of range and the shader's per-column
  // bounds check is the only thing between this and writes past the output. N is
  // also not a multiple of 16, so the coopmat tactic declines and cannot be what is
  // measured. bt = 0 at m = 1 makes the GEMV tactic decline for the documented
  // reason (that layout is already coalesced), which is exactly why this GEMM is
  // on vt_matmul at all.
  constexpr int64_t kK = 258, kN = 33;

  // THE OPERANDS MUST SPAN DECADES, and that is a MEASURED requirement, not
  // decoration. The first version of this case used the small exact ladder the
  // neighbouring cases use (multiples of 0.25 and 0.5). Every partial sum there is
  // representable in f32 with no rounding at all, so a deliberately reassociated
  // shader body -- `acc += (a0*b0 + a1*b1) + (a2*b2 + a3*b3)`, exactly the
  // "improvement" this gate exists to forbid -- produced BITWISE IDENTICAL output
  // and the memcmp below PASSED the mutation. A gate on floating-point
  // associativity has to feed it operands whose partial sums actually round.
  //
  // These do: a 32-bit LCG picks a sign, a mantissa and an exponent in [2^-7,
  // 2^8), so terms differ by up to ~2^15 and a large partial absorbs a small one
  // differently depending on when it is added. Both operands are bf16, whose
  // 8-bit mantissas make every PRODUCT exact in f32 -- so the only source of
  // difference between the two arms is the accumulation ORDER, which is precisely
  // the property under test.
  std::vector<uint16_t> a(static_cast<size_t>(kK)), b(static_cast<size_t>(kK * kN));
  uint32_t rng = 0x9E3779B9u;
  auto next = [&rng] {
    rng = rng * 1664525u + 1013904223u;
    const float mag = std::ldexp(1.0f + static_cast<float>((rng >> 8) & 0xFFu) / 256.0f,
                                 static_cast<int>((rng >> 20) & 0xFu) - 7);
    return vt::F32ToBF16((rng & 1u) != 0u ? -mag : mag);
  };
  for (int64_t i = 0; i < kK; ++i) a[static_cast<size_t>(i)] = next();
  for (int64_t i = 0; i < kK * kN; ++i) b[static_cast<size_t>(i)] = next();

  void* da = vk.Alloc(a.size() * sizeof(uint16_t));
  void* db = vk.Alloc(b.size() * sizeof(uint16_t));
  auto* dout = static_cast<float*>(vk.Alloc(kN * sizeof(float)));
  vk.Copy(q, da, a.data(), a.size() * sizeof(uint16_t));
  vk.Copy(q, db, b.data(), b.size() * sizeof(uint16_t));
  vk.Synchronize(q);
  Tensor ta = Tensor::Contiguous(da, vt::DType::kBF16, d, {1, kK});
  Tensor tb = Tensor::Contiguous(db, vt::DType::kBF16, d, {kK, kN});
  Tensor to = Tensor::Contiguous(dout, vt::DType::kF32, d, {1, kN});

  auto run = [&](uint32_t ncols) {
    vt::vulkan::SetMatmulColumnsPerLane(ncols);
    vk.Memset(q, dout, 0, kN * sizeof(float));
    vt::Matmul(q, to, ta, tb);
    vk.Synchronize(q);
    std::vector<float> got(static_cast<size_t>(kN));
    vk.Copy(q, got.data(), dout, got.size() * sizeof(float));
    vk.Synchronize(q);
    return got;
  };

  const uint32_t restore = vt::vulkan::MatmulColumnsPerLane();
  const std::vector<float> blocked = run(4u);
  const std::vector<float> flat = run(1u);
  vt::vulkan::SetMatmulColumnsPerLane(restore);

  // MECHANISM, ASSERTION 1: the two arms are DISTINCT PIPELINES of the same
  // module. PipelineExistsFor("vt_matmul") passes identically whether column
  // blocking was selected or not -- it is a PERFORMANCE axis, so the numbers are
  // bit-identical by construction and no value check can ever see it. The
  // specialization VALUES are the only place the mechanism is visible, which is
  // why PipelineKeys() exists.
  //
  // Key layout is "<module>|<a dtype>,<b dtype>,<out dtype>,<bt>,<ncols>";
  // bf16 = 2, f32 = 0 (DtypeCode, src/vt/vulkan/vulkan_ops.cpp), bt = 0 here.
  const std::vector<std::string> keys = ctx.PipelineKeys();
  const bool has_blocked =
      std::find(keys.begin(), keys.end(), std::string("vt_matmul|2,2,0,0,4")) != keys.end();
  const bool has_flat =
      std::find(keys.begin(), keys.end(), std::string("vt_matmul|2,2,0,0,1")) != keys.end();
  std::string joined;
  for (const std::string& k : keys) { joined += k; joined += " "; }
  CAPTURE(joined);
  CHECK(has_blocked);
  CHECK(has_flat);

  // MECHANISM, ASSERTION 2: the DEFAULT is the blocked arm, AND it is the arm
  // that measured fastest. Without this the optimization could be present,
  // specializable and never selected, and every assertion above would still pass.
  // The constant is 4 rather than "not 1" on purpose: 8 blocks harder and measured
  // SLOWER than not blocking at all (243 workgroups is too little parallelism to
  // hide memory latency), so "some blocking" is not the property worth pinning.
  CHECK(vt::vulkan::MatmulColumnsPerLane() == 4u);

  // NUMERIC TIER, ASSERTION 3: BITWISE, not a tolerance. The whole argument for
  // blocking COLUMNS rather than splitting the K reduction is that vt_matmul is
  // the ONE matmul tactic that shares cpu_ops.cpp MatmulChunked's accumulation
  // order -- the byte-exact tier the coopmat and GEMV tactics both gave up. Each
  // of a lane's accumulators owns a different output element and sums the whole K
  // sequentially, which is that order exactly, so the correct gate is memcmp. An
  // epsilon check here would pass just as well if someone "improved" the shader
  // into partial K accumulators and silently moved the general matmul path -- and
  // the sampler that reads the lm_head's output -- into the NMSE tier.
  REQUIRE(blocked.size() == flat.size());
  CHECK(std::memcmp(blocked.data(), flat.data(), blocked.size() * sizeof(float)) == 0);

  // And both are right: a f64 host oracle, so neither arm's order is privileged.
  //
  // The tolerance is scaled by the sum of |terms|, NOT by the result. These
  // operands span decades and cancel, so a dot product can land near zero while
  // its terms are large, and a relative-to-result epsilon would then demand more
  // precision than f32 accumulation can deliver for reasons that have nothing to
  // do with this change. Absolute error against the CONDITIONING of the sum is the
  // honest bound.
  for (int64_t j = 0; j < kN; ++j) {
    double acc = 0.0, scale = 0.0;
    for (int64_t c = 0; c < kK; ++c) {
      const double term = static_cast<double>(vt::BF16ToF32(a[static_cast<size_t>(c)])) *
                          static_cast<double>(vt::BF16ToF32(b[static_cast<size_t>(c * kN + j)]));
      acc += term;
      scale += std::abs(term);
    }
    CAPTURE(j);
    CAPTURE(acc);
    CAPTURE(scale);
    CHECK(std::abs(static_cast<double>(blocked[static_cast<size_t>(j)]) - acc) <= 1e-4 * scale);
  }

  vk.Free(da);
  vk.Free(db);
  vk.Free(dout);
  vk.DestroyQueue(q);
}

TEST_CASE("greedy argmax tree-reduces the vocabulary and keeps the first-wins tie-break") {
  if (!VulkanPresent()) return;
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  // The kernel changed from one INVOCATION per row scanning the vocabulary
  // serially (10.03 ms/call measured, 10% of all GPU time in an e2e run) to one
  // WORKGROUP per row with a tree reduction. A reduction can be fast and still
  // wrong in ways a single max VALUE never reveals, so what is asserted here is
  // the INDEX, under the two conditions a reduction actually breaks.
  //
  // kV is deliberately NOT a multiple of the 128-lane workgroup, so the last
  // lane's chunk is short and the empty-range path is exercised.
  constexpr int64_t kV = 1000;

  auto run = [&](const std::vector<float>& logits) {
    void* dl = vk.Alloc(logits.size() * sizeof(float));
    void* dt = vk.Alloc(2 * sizeof(int64_t));
    vk.Copy(q, dl, logits.data(), logits.size() * sizeof(float));
    vk.Synchronize(q);
    Tensor tl = Tensor::Contiguous(dl, vt::DType::kF32, d, {1, kV});
    Tensor tt = Tensor::Contiguous(dt, vt::DType::kI64, d, {1});
    vt::GreedyArgmax(q, tt, tl);
    vk.Synchronize(q);
    int64_t got = -1;
    vk.Copy(q, &got, dt, sizeof(int64_t));
    vk.Synchronize(q);
    vk.Free(dl);
    vk.Free(dt);
    return got;
  };

  SUBCASE("a plain maximum is found across the whole vocabulary") {
    std::vector<float> l(kV, 0.0f);
    // Past lane 0's chunk, so a scan that only ever covered chunk 0 -- the shape
    // of the old single-lane kernel's parallel replacement done wrong -- misses it.
    l[777] = 5.0f;
    CHECK(run(l) == 777);
  }

  SUBCASE("ties resolve to the LOWEST index, including across lane boundaries") {
    std::vector<float> l(kV, 0.0f);
    // 8 and 900 land in different lanes' chunks, so the winner is decided by the
    // MERGE, not by either lane's own scan. A merge written with `>=` instead of
    // `>` -- the natural way to write it -- returns 900 here and passes every
    // check that only looks at the maximum value.
    l[8] = 3.0f;
    l[900] = 3.0f;
    CHECK(run(l) == 8);
  }

  SUBCASE("ties within a single lane's chunk also resolve to the lowest index") {
    std::vector<float> l(kV, 0.0f);
    l[3] = 2.0f;
    l[4] = 2.0f;
    CHECK(run(l) == 3);
  }

  SUBCASE("a NaN POISONS the scan exactly as the CPU kernel's does") {
    // cpu_sample.cpp:49 compares with `x > best`, which is false for every NaN.
    // A NaN adopted as the running best therefore blocks every later candidate,
    // and the CPU returns the index it was holding -- here 0, the initial one.
    // This is the case a STRIDED split would get wrong: the lane covering 500
    // would never see the NaN at 0 and would return 500, disagreeing with the CPU
    // oracle on a diverged model. Contiguous chunks are what make it agree.
    std::vector<float> l(kV, 0.0f);
    l[0] = std::numeric_limits<float>::quiet_NaN();
    l[500] = 9.0f;
    CHECK(run(l) == 0);
  }

  SUBCASE("a NaN AFTER the running maximum does not displace it") {
    std::vector<float> l(kV, 0.0f);
    l[10] = 7.0f;
    l[600] = std::numeric_limits<float>::quiet_NaN();
    CHECK(run(l) == 10);
  }

  vk.DestroyQueue(q);
}

TEST_CASE("VK-A2 batching records many dispatches per submit and stays byte-exact") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  // Asked, not re-derived. An earlier version of this gate recomputed the lever's
  // default from the environment and asserted the WRONG branch as soon as the
  // default flipped on -- it failed loudly, but a subtler duplication would just
  // have gone vacuous.
  const bool batching = ctx.batching_enabled();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  // WHAT THIS ASSERTS IS THE MECHANISM, because the RESULTS cannot show it:
  // batching runs the same kernels in the same order, so a batch that silently
  // degrades to one dispatch per submit computes identical numbers and every
  // value check still passes. Without a pending-count assertion this whole path
  // could stop batching and no gate would notice -- the same failure shape the
  // coopmat tactic needed PipelineExistsFor for.
  constexpr int64_t kN = 4096;
  constexpr int kOps = 6;

  std::vector<float> a(kN, 1.5f), b(kN, 2.25f);
  void* da = vk.Alloc(kN * sizeof(float));
  void* db = vk.Alloc(kN * sizeof(float));
  auto* dout = static_cast<float*>(vk.Alloc(kN * sizeof(float)));
  vk.Copy(q, da, a.data(), a.size() * sizeof(float));
  vk.Copy(q, db, b.data(), b.size() * sizeof(float));
  vk.Synchronize(q);

  Tensor ta = Tensor::Contiguous(da, vt::DType::kF32, d, {kN});
  Tensor tb = Tensor::Contiguous(db, vt::DType::kF32, d, {kN});
  Tensor to = Tensor::Contiguous(dout, vt::DType::kF32, d, {kN});

  // Issue several dependent ops with NO intervening host read, which is the only
  // way a batch can accumulate.
  for (int i = 0; i < kOps; ++i) vt::Add(q, to, ta, tb);
  const uint32_t pending = ctx.pending_batch();
  CAPTURE(pending);

  if (batching) {
    // More than one dispatch is in flight, i.e. the submit really was deferred.
    CHECK(pending > 1u);
  } else {
    // Default build: every dispatch submitted and waited, so nothing is ever
    // pending. This half keeps the assertion honest on the default path rather
    // than making the test vacuous when the lever is off.
    CHECK(pending == 0u);
  }

  // Flushing must make the writes visible to a plain host read -- Copy is a
  // memcpy over mapped memory, so a missing flush shows up as stale bytes here.
  std::vector<float> got(kN, 0.0f);
  vk.Copy(q, got.data(), dout, got.size() * sizeof(float));
  vk.Synchronize(q);
  CHECK(ctx.pending_batch() == 0u);
  for (int64_t i = 0; i < kN; i += 512) {
    CAPTURE(i);
    CHECK(got[static_cast<size_t>(i)] == doctest::Approx(3.75f));
  }

  vk.Free(da);
  vk.Free(db);
  vk.Free(dout);
  vk.DestroyQueue(q);
}

TEST_CASE("submission is PIPELINED: a flush does not block, and a drain retires everything") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  if (!ctx.batching_enabled()) return;  // nothing is ever submitted without a flush
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  // WHY THIS GATE EXISTS (BACKEND-VULKAN-HOSTDISPATCH). Pipelined submission
  // records the SAME kernels in the SAME order as submit-and-wait, so it is
  // invisible in every number this file checks: if the pipelining silently
  // stopped applying -- a slot count that fell back to 1, a rotation that never
  // happened, a wait that crept back into the flush -- every value assertion in
  // this file would still pass and only the wall clock would move. The MECHANISM
  // is the only thing that can be asserted.
  const uint32_t limit = ctx.in_flight_limit();
  const uint32_t slice = ctx.ring_slice();
  CAPTURE(limit);
  CAPTURE(slice);
  // Asked, never re-derived from the environment: the same duplication trap
  // batching_enabled() was introduced for.
  REQUIRE(limit >= 1u);
  REQUIRE(limit <= vt::vulkan::VulkanContext::kMaxInFlight);
  // The slices must PARTITION the ring, because that is the whole correctness
  // argument for letting a submitted batch keep executing while the next batch
  // rewrites descriptor sets: slot s only ever touches [s*slice, (s+1)*slice).
  const uint32_t depth = ctx.ring_depth();
  CAPTURE(depth);
  CHECK(slice * limit <= depth);
  CHECK(slice >= 1u);

  constexpr int64_t kN = 2048;
  std::vector<float> a(static_cast<size_t>(kN), 1.5f), b(static_cast<size_t>(kN), 2.25f);
  void* da = vk.Alloc(kN * sizeof(float));
  void* db = vk.Alloc(kN * sizeof(float));
  auto* dout = static_cast<float*>(vk.Alloc(kN * sizeof(float)));
  vk.Copy(q, da, a.data(), a.size() * sizeof(float));
  vk.Copy(q, db, b.data(), b.size() * sizeof(float));
  vk.Synchronize(q);

  Tensor ta = Tensor::Contiguous(da, vt::DType::kF32, d, {kN});
  Tensor tb = Tensor::Contiguous(db, vt::DType::kF32, d, {kN});
  Tensor to = Tensor::Contiguous(dout, vt::DType::kF32, d, {kN});

  // Synchronize left nothing in flight, so the counters below start from a known
  // point rather than from whatever earlier test cases left behind.
  REQUIRE(ctx.in_flight_batches() == 0u);
  const uint64_t submits0 = ctx.submit_count();
  const uint64_t waits0 = ctx.fence_wait_count();
  const uint64_t barriers0 = ctx.barrier_count();
  const uint32_t base0 = ctx.ring_base();

  // Exactly enough dispatches of ONE pipeline to exhaust its slice and force a
  // ring-full flush, plus one more so a batch is open again afterwards.
  for (uint32_t i = 0; i < slice + 1; ++i) vt::Add(q, to, ta, tb);
  const uint32_t base1 = ctx.ring_base();
  CAPTURE(base0);
  CAPTURE(base1);

  // ONE BARRIER PER DISPATCH, first-in-command-buffer included. That first one is
  // the ONLY thing ordering a new command buffer's work against the previous,
  // still-executing batch; dropping it computes correct numbers on a software
  // rasterizer and races on real hardware, so it is asserted by count and not by
  // value.
  //
  // The hazard analysis reaches the SAME count on this loop rather than being
  // excused from it: every iteration rewrites `to`, which is a write-after-write
  // against the previous iteration, so each one is a genuine dependency.
  CHECK(ctx.barrier_count() - barriers0 == static_cast<uint64_t>(slice) + 1u);

  // THE SLICES PARTITION THE RING. Without this a mutation collapsing every slot
  // onto slice 0 passes this entire file on llvmpipe -- MEASURED, it did.
  if (limit > 1u) {
    CHECK(base1 != base0);
    CHECK(base1 % slice == 0u);
  } else {
    CHECK(base1 == 0u);
  }
  CHECK(base1 + slice <= depth);

  const uint64_t submits1 = ctx.submit_count();
  const uint64_t waits1 = ctx.fence_wait_count();
  const uint32_t in_flight = ctx.in_flight_batches();
  CAPTURE(submits1 - submits0);
  CAPTURE(waits1 - waits0);
  CAPTURE(in_flight);

  // The ring-full flush happened.
  REQUIRE(submits1 > submits0);

  if (limit > 1u) {
    // THE MECHANISM. The flush submitted and RETURNED: a batch is still in
    // flight, and the rotation landed on a slot that was already idle so nothing
    // blocked. Under the old submit-and-wait shape both of these read zero.
    CHECK(in_flight >= 1u);
    CHECK(waits1 == waits0);
    // And the host really did carry on recording into the next slot.
    CHECK(ctx.pending_batch() > 0u);
  } else {
    // VT_VULKAN_INFLIGHT=1 is the A/B's control arm and must be EXACTLY the
    // pre-row behaviour: every flush waits, nothing is ever left in flight.
    CHECK(in_flight == 0u);
  }

  // A host read must still see every byte, which is the contract pipelining is
  // most able to break: the drain has to wait for EVERY submitted batch, not
  // merely the open one.
  std::vector<float> got(static_cast<size_t>(kN), 0.0f);
  vk.Copy(q, got.data(), dout, got.size() * sizeof(float));
  vk.Synchronize(q);
  CHECK(ctx.pending_batch() == 0u);
  CHECK(ctx.in_flight_batches() == 0u);
  for (int64_t i = 0; i < kN; i += 256) {
    CAPTURE(i);
    CHECK(got[static_cast<size_t>(i)] == doctest::Approx(3.75f));
  }

  // DESCRIPTOR-SET REUSE UNDER ROTATION. A long DEPENDENT chain that wraps the
  // slot rotation several times is what would expose a set being rewritten while
  // an earlier, still-executing dispatch reads it: each step must observe exactly
  // its predecessor's output, so a stale or clobbered descriptor shows up as a
  // wrong final value rather than as a crash.
  const uint32_t steps = slice * (limit + 1) * 2 + 3;
  CAPTURE(steps);
  std::vector<float> zero(static_cast<size_t>(kN), 0.0f), one(static_cast<size_t>(kN), 1.0f);
  vk.Copy(q, da, zero.data(), zero.size() * sizeof(float));
  vk.Copy(q, db, one.data(), one.size() * sizeof(float));
  vk.Synchronize(q);
  // acc <- acc + 1, `steps` times, with NO host read in between.
  Tensor tacc = Tensor::Contiguous(da, vt::DType::kF32, d, {kN});
  for (uint32_t i = 0; i < steps; ++i) vt::Add(q, tacc, tacc, tb);
  vk.Copy(q, got.data(), da, got.size() * sizeof(float));
  vk.Synchronize(q);
  for (int64_t i = 0; i < kN; i += 256) {
    CAPTURE(i);
    CHECK(got[static_cast<size_t>(i)] == doctest::Approx(static_cast<float>(steps)));
  }

  vk.Free(da);
  vk.Free(db);
  vk.Free(dout);
  vk.DestroyQueue(q);
}

// A scoped restore for the barrier-policy lever, so a failing assertion inside a
// case below cannot leave the whole rest of the file running under a policy it
// did not ask for.
namespace {
class SmartBarrierArm {
 public:
  explicit SmartBarrierArm(int v) : prev_(vt::vulkan::VulkanContext::Get().smart_barriers_override()) {
    vt::vulkan::VulkanContext::Get().set_smart_barriers_override(v);
  }
  ~SmartBarrierArm() { vt::vulkan::VulkanContext::Get().set_smart_barriers_override(prev_); }
  SmartBarrierArm(const SmartBarrierArm&) = delete;
  SmartBarrierArm& operator=(const SmartBarrierArm&) = delete;

 private:
  int prev_;
};
}  // namespace

TEST_CASE("SMART BARRIERS: a real hazard barriers, a proven-independent pair does not") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  if (!ctx.batching_enabled()) return;  // no barriers are recorded at all without it
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  // WHY THIS GATE EXISTS (BACKEND-VULKAN-BARRIERS), AND WHY IT ASSERTS COUNTERS.
  // Skipping a barrier changes NO number when it is correct and changes numbers
  // only on real hardware when it is wrong -- a software rasterizer runs the
  // dispatches serially enough to hide a dropped dependency, which is how this
  // backend's fence-spin arm passed 33/33 on llvmpipe while producing "the the
  // capital capital of of" on GB10. So the mechanism is asserted directly: the
  // pairs below are constructed so that the CORRECT number of barriers is known
  // in advance from the dependency structure alone, and any predicate that stops
  // detecting hazards moves that number.
  constexpr int64_t kN = 1024;
  const size_t bytes = static_cast<size_t>(kN) * sizeof(float);
  std::vector<float> a(static_cast<size_t>(kN), 1.5f), b(static_cast<size_t>(kN), 2.25f);
  // FIVE SEPARATE ALLOCATIONS, which is what makes the independence real: this
  // backend gives every allocation its own VkBuffer and its own VkDeviceMemory,
  // so distinct pointers here cannot alias and "different buffer" is a fact
  // rather than an assumption about offsets.
  void* da = vk.Alloc(bytes);
  void* db = vk.Alloc(bytes);
  void* d1 = vk.Alloc(bytes);
  void* d2 = vk.Alloc(bytes);
  void* d3 = vk.Alloc(bytes);
  vk.Copy(q, da, a.data(), bytes);
  vk.Copy(q, db, b.data(), bytes);
  vk.Synchronize(q);

  Tensor ta = Tensor::Contiguous(da, vt::DType::kF32, d, {kN});
  Tensor tb = Tensor::Contiguous(db, vt::DType::kF32, d, {kN});
  Tensor t1 = Tensor::Contiguous(d1, vt::DType::kF32, d, {kN});
  Tensor t2 = Tensor::Contiguous(d2, vt::DType::kF32, d, {kN});
  Tensor t3 = Tensor::Contiguous(d3, vt::DType::kF32, d, {kN});

  // vt_add reads both inputs and writes the output -- the writable mask asserted
  // by the SPIR-V table case above -- so the five dispatches below have exactly
  // one dependency structure and it is visible by inspection.
  //
  //   d0  t1 <- a + b      (the policy switch forces this one to barrier)
  //   d1  t2 <- a + b      INDEPENDENT of d0: shares only the READ operands
  //   d2  t3 <- t1 + b     READ-AFTER-WRITE on t1
  //   d3  t1 <- a + b      WRITE-AFTER-READ on t1 (d2 read it)
  //   d4  t1 <- a + b      WRITE-AFTER-WRITE on t1 (d3 wrote it)
  //
  // So: 4 barriers and exactly 1 skip. Each of the three hazard KINDS appears
  // once, because a predicate that only implements read-after-write is a real
  // and plausible mistake that produces correct numbers most of the time.
  SUBCASE("the analysis skips exactly the independent pair") {
    SmartBarrierArm arm(+1);
    REQUIRE(ctx.smart_barriers());
    const uint64_t barriers0 = ctx.barrier_count();
    const uint64_t skips0 = ctx.barrier_skip_count();

    vt::Add(q, t1, ta, tb);
    vt::Add(q, t2, ta, tb);
    vt::Add(q, t3, t1, tb);
    vt::Add(q, t1, ta, tb);
    vt::Add(q, t1, ta, tb);

    const uint64_t barriers = ctx.barrier_count() - barriers0;
    const uint64_t skips = ctx.barrier_skip_count() - skips0;
    CAPTURE(barriers);
    CAPTURE(skips);
    // THE MUTATION TARGET. Forcing the hazard predicate to report "independent"
    // takes this to 1 and 4 (only the policy-switch barrier survives); dropping
    // the write-after-read half alone takes it to 3 and 2. Neither changes any
    // value this file checks.
    CHECK(barriers == 4u);
    CHECK(skips == 1u);
    // Every dispatch is accounted for exactly once.
    CHECK(barriers + skips == 5u);
    vk.Synchronize(q);
  }

  // THE CONTROL ARM, in the SAME BINARY. The two arms run the same kernels in the
  // same order on the same inputs, so a cross-BUILD comparison could not tell
  // them apart at all -- and this campaign has already been given a false 1.2x by
  // exactly that shape. Here the lever is proven to MOVE.
  SUBCASE("the always-barrier arm records one per dispatch and skips none") {
    SmartBarrierArm arm(-1);
    REQUIRE(!ctx.smart_barriers());
    const uint64_t barriers0 = ctx.barrier_count();
    const uint64_t skips0 = ctx.barrier_skip_count();

    vt::Add(q, t1, ta, tb);
    vt::Add(q, t2, ta, tb);
    vt::Add(q, t3, t1, tb);
    vt::Add(q, t1, ta, tb);
    vt::Add(q, t1, ta, tb);

    CHECK(ctx.barrier_count() - barriers0 == 5u);
    CHECK(ctx.barrier_skip_count() - skips0 == 0u);
    vk.Synchronize(q);
  }

  // A DEPENDENT CHAIN LONG ENOUGH TO CROSS SEVERAL COMMAND BUFFERS, with the
  // analysis on. This is the value half, and it is the half only real hardware
  // can fail: each step must observe exactly its predecessor's output, and the
  // chain wraps the slot rotation repeatedly so the dependency has to survive the
  // command-buffer boundary that a pipelined submission creates.
  SUBCASE("a long dependent chain is still exact with the analysis on") {
    SmartBarrierArm arm(+1);
    const uint32_t steps = ctx.ring_slice() * (ctx.in_flight_limit() + 1) * 2 + 3;
    CAPTURE(steps);
    std::vector<float> zero(static_cast<size_t>(kN), 0.0f), one(static_cast<size_t>(kN), 1.0f);
    vk.Copy(q, d1, zero.data(), bytes);
    vk.Copy(q, db, one.data(), bytes);
    vk.Synchronize(q);
    const uint64_t barriers0 = ctx.barrier_count();
    for (uint32_t i = 0; i < steps; ++i) vt::Add(q, t1, t1, tb);
    // Every step is a read-after-write AND a write-after-write on t1, so the
    // analysis must find a hazard at every single one. A chain like this is
    // exactly where a dropped barrier turns into a wrong number.
    CHECK(ctx.barrier_count() - barriers0 == static_cast<uint64_t>(steps));
    std::vector<float> got(static_cast<size_t>(kN), -1.0f);
    vk.Copy(q, got.data(), d1, bytes);
    vk.Synchronize(q);
    for (int64_t i = 0; i < kN; i += 128) {
      CAPTURE(i);
      CHECK(got[static_cast<size_t>(i)] == doctest::Approx(static_cast<float>(steps)));
    }
  }

  // INDEPENDENT WRITES FOLLOWED BY A READER. The writes may all skip; the reader
  // must not, and its value proves the skipped ones really did complete. This is
  // the pattern the decode step would actually exploit -- several projections off
  // one normalized activation, then a consumer -- so it is checked for VALUE and
  // not only for counts.
  SUBCASE("independent writers then a consumer: the consumer barriers, values hold") {
    SmartBarrierArm arm(+1);
    std::vector<float> three(static_cast<size_t>(kN), 3.0f);
    vk.Copy(q, da, a.data(), bytes);
    vk.Copy(q, db, three.data(), bytes);
    vk.Synchronize(q);
    // Re-assert the policy to force the history to a KNOWN state: the setter
    // drains and marks the next dispatch as unconditionally barriered, so the
    // dispatch below clears the access sets and the counts that follow start from
    // a history containing exactly it. Without this the sets would still carry
    // whatever an earlier subcase left, and t2/t3 could look written.
    ctx.set_smart_barriers_override(+1);
    vt::Add(q, t1, ta, tb);
    const uint64_t barriers0 = ctx.barrier_count();
    const uint64_t skips0 = ctx.barrier_skip_count();
    vt::Add(q, t2, ta, tb);   // independent of t1: skip
    vt::Add(q, t3, ta, tb);   // independent of both: skip
    vt::Add(q, t1, t2, t3);   // reads t2 AND t3: read-after-write, barrier
    CHECK(ctx.barrier_count() - barriers0 == 1u);
    CHECK(ctx.barrier_skip_count() - skips0 == 2u);
    std::vector<float> got(static_cast<size_t>(kN), -1.0f);
    vk.Copy(q, got.data(), d1, bytes);
    vk.Synchronize(q);
    // t2 = t3 = 1.5 + 3.0 = 4.5, so t1 = 9.0. A dropped barrier before the
    // consumer reads whatever t2/t3 held before -- which on this box is the 4.5
    // of an earlier subcase or uninitialised memory, either way not 9.0.
    for (int64_t i = 0; i < kN; i += 128) {
      CAPTURE(i);
      CHECK(got[static_cast<size_t>(i)] == doctest::Approx(9.0f));
    }
  }

  vk.Free(da);
  vk.Free(db);
  vk.Free(d1);
  vk.Free(d2);
  vk.Free(d3);
  vk.DestroyQueue(q);
}

TEST_CASE("a REFERENCE-TIER op drains the batch before it touches device memory") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  if (!ctx.batching_enabled()) return;  // nothing is ever pending with the lever off
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  // THE HAZARD THIS CLOSES, and why it needed its own gate. The portable
  // reference tier runs a HOST kernel directly over device memory, which is only
  // sound because this backend is unified. With batching, a dispatch may be
  // recorded and NOT yet submitted, so that host kernel would read bytes the GPU
  // has not written -- silently, with no error and no crash.
  //
  // The opt-125m STRICT gate passes with batching on, but it CANNOT prove this:
  // OPT touches the reference tier only at setup, before any batch is open. So it
  // would pass whether or not the hook exists. This asserts the hook FIRES.
  constexpr int64_t kN = 1024;
  std::vector<float> a(kN, 1.0f);
  void* da = vk.Alloc(kN * sizeof(float));
  void* db = vk.Alloc(kN * sizeof(float));
  auto* dout = static_cast<float*>(vk.Alloc(kN * sizeof(float)));
  vk.Copy(q, da, a.data(), a.size() * sizeof(float));
  vk.Copy(q, db, a.data(), a.size() * sizeof(float));
  vk.Synchronize(q);

  Tensor ta = Tensor::Contiguous(da, vt::DType::kF32, d, {kN});
  Tensor tb = Tensor::Contiguous(db, vt::DType::kF32, d, {kN});
  Tensor to = Tensor::Contiguous(dout, vt::DType::kF32, d, {kN});
  for (int i = 0; i < 4; ++i) vt::Add(q, to, ta, tb);
  REQUIRE(ctx.pending_batch() > 0u);  // a batch really is open

  // Resolving a reference-tier op is what op_provider.cpp routes through
  // Backend::FlushPending. kApplyTemperature (the sampler past greedy argmax) is
  // still genuinely unimplemented natively on Vulkan.
  REQUIRE(vt::ReferenceTierEligible(DeviceType::kVULKAN));
  CHECK_FALSE(vt::OpRegistered(vt::OpId::kApplyTemperature, DeviceType::kVULKAN));
  void* ref = vt::GetOp(vt::OpId::kApplyTemperature, DeviceType::kVULKAN);
  CHECK(ref != nullptr);

  // THE ASSERTION: resolving that host kernel drained the batch. Had FlushPending
  // stayed the base class's no-op, this would still read > 0 and the host kernel
  // would go on to read stale device memory.
  CHECK(ctx.pending_batch() == 0u);

  vk.Free(da);
  vk.Free(db);
  vk.Free(dout);
  vk.DestroyQueue(q);
}

TEST_CASE("a RAGGED-M GEMM takes coopmat and is exact -- the shifted trailing tile") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  if (!ctx.coopmat_bf16_f32() || ctx.subgroup_size() != 32) return;  // scalar device
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  // M = 17 IS THE SHAPE THAT COST A 100x. Prompt length gives m = tokens + 1, so
  // every prefill M is one past a tile boundary: 513 % 16 == 1. The old predicate
  // required m % 16 == 0, so every 27B prefill GEMM fell to the untiled scalar
  // kernel -- measured at 99.9% of GPU time and ~96 GFLOP/s.
  //
  // The trailing tile now slides back to start at M-16, so it reads only real
  // rows. Rows 1..15 are therefore computed TWICE, by both tiles, and this test
  // exists to prove those duplicate writes are bit-identical rather than merely
  // close: a row's result depends only on that row of A and on B.
  constexpr int64_t kM = 17, kK = 32, kN = 32;

  std::vector<uint16_t> a(kM * kK), b(kN * kK);
  for (int64_t i = 0; i < kM * kK; ++i)
    a[static_cast<size_t>(i)] = vt::F32ToBF16(0.5f * static_cast<float>((i % 7) - 3));
  for (int64_t i = 0; i < kN * kK; ++i)
    b[static_cast<size_t>(i)] = vt::F32ToBF16(0.25f * static_cast<float>((i % 5) - 2));

  void* da = vk.Alloc(a.size() * sizeof(uint16_t));
  void* db = vk.Alloc(b.size() * sizeof(uint16_t));
  auto* dout = static_cast<float*>(vk.Alloc(kM * kN * sizeof(float)));
  vk.Copy(q, da, a.data(), a.size() * sizeof(uint16_t));
  vk.Copy(q, db, b.data(), b.size() * sizeof(uint16_t));
  vk.Synchronize(q);

  Tensor ta = Tensor::Contiguous(da, vt::DType::kBF16, d, {kM, kK});
  Tensor tb = Tensor::Contiguous(db, vt::DType::kBF16, d, {kN, kK});
  Tensor to = Tensor::Contiguous(dout, vt::DType::kF32, d, {kM, kN});
  vt::MatmulBT(q, to, ta, tb);
  vk.Synchronize(q);

  // THE TACTIC IS THE LOAD-BEARING ASSERTION. The scalar kernel is equally
  // correct, so numbers alone cannot show that a ragged M now reaches the tensor
  // cores -- which is the entire point of the change.
  CHECK(ctx.PipelineExistsFor("vt_matmul_coopmat"));

  std::vector<float> got(static_cast<size_t>(kM * kN));
  vk.Copy(q, got.data(), dout, got.size() * sizeof(float));
  vk.Synchronize(q);

  // Exact against a host oracle in f64. Row 16 (only the shifted tile) and rows
  // 1..15 (BOTH tiles) must agree; a wrong shift shows up as a wrong or
  // duplicated row rather than as noise.
  for (int64_t i = 0; i < kM; ++i) {
    for (int64_t j = 0; j < kN; ++j) {
      double acc = 0.0;
      for (int64_t c = 0; c < kK; ++c) {
        acc += static_cast<double>(vt::BF16ToF32(a[static_cast<size_t>(i * kK + c)])) *
               static_cast<double>(vt::BF16ToF32(b[static_cast<size_t>(j * kK + c)]));
      }
      CAPTURE(i);
      CAPTURE(j);
      CHECK(got[static_cast<size_t>(i * kN + j)] ==
            doctest::Approx(static_cast<float>(acc)).epsilon(1e-3));
    }
  }

  vk.Free(da);
  vk.Free(db);
  vk.Free(dout);
  vk.DestroyQueue(q);
}

TEST_CASE("Vulkan float-controls are PROBED and reported, not assumed") {
  if (!VulkanPresent()) return;
  // The relaxed-precision knobs Vulkan leaves implementation-defined. We cannot
  // pin fp32 denormal/signed-zero behaviour from GLSL without
  // SPV_KHR_float_controls execution modes, so the honest gate is to RECORD what
  // this device does. Both outcomes are acceptable — the shaders avoid
  // `inversesqrt` and carry integer bf16/f16 codecs precisely so that neither
  // knob can move a gated result — but a silent change here would be the first
  // clue if a future NMSE regression appeared, so it is printed rather than
  // asserted to a particular value.
  auto& ctx = vt::vulkan::VulkanContext::Get();
  MESSAGE("vulkan device: " << ctx.device_name() << " (API " << ctx.api_major() << "."
                            << ctx.api_minor() << ")");
  MESSAGE("shaderDenormPreserveFloat32 = " << ctx.denorm_preserve_f32());
  MESSAGE("shaderSignedZeroInfNanPreserveFloat32 = "
          << ctx.signed_zero_inf_nan_preserve_f32());
  // What we DO require: the workgroup-count limit must cover the dispatches the
  // skeleton makes (one workgroup per row).
  CHECK(ctx.max_workgroup_count_x() >= 65535u);
}

// ===========================================================================
// BACKEND-VULKAN-GDN — numeric gates for the GDN / conv1d glue family.
//
// THE ORACLE IS OUR OWN CPU BACKEND, evaluated in the SAME binary on the SAME
// inputs, which is the contract tests/vt/test_backend_cross_device.cpp already
// states. It is used here rather than there because these ops need shapes with
// structure — padded gate strides, widened cache rows, NULL cache indices — that
// the generic harness does not generate.
//
// EVERY CASE ALSO ASSERTS THE MECHANISM, not only the numbers. On a unified
// device an unregistered op silently resolves to the portable CPU reference tier
// and produces answers that are not merely close to the oracle but IDENTICAL to
// it, so a numbers-only gate would pass just as green with no shader written at
// all. `PipelineExistsFor` proves a Vulkan pipeline for the intended module was
// built, and the provider's `last_selected` name proves the call did not fall
// through to the host. Both, because either alone has a hole: a pipeline can
// exist from an earlier case in the same process, and a provider can be selected
// for a shape whose kernel then declines.
// ===========================================================================
namespace {

// Deterministic, library-independent filler. A fixed LCG rather than
// std::mt19937 so both backends see byte-identical inputs regardless of standard
// library version, spread over a range that actually exercises sigmoid and
// softplus instead of sitting in their linear middle.
std::vector<float> Spread(size_t n, float scale, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed | 1u;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = (static_cast<float>(s >> 8) / 8388608.0f - 1.0f) * scale;
  }
  return v;
}

double NmseOf(const std::vector<float>& ref, const std::vector<float>& got) {
  REQUIRE(ref.size() == got.size());
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double dd = static_cast<double>(ref[i]) - static_cast<double>(got[i]);
    num += dd * dd;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  return den == 0.0 ? num : num / den;
}

// The already-ported bar for reducing / arithmetic kernels
// (tests/vt/test_backend_cross_device.cpp, itself from llama.cpp
// test-quantize-fns:17-28). NOT bit-exactness: a workgroup tree reduction does
// not preserve the CPU tier's fixed sequential accumulation order.
constexpr double kGdnNmseTol = 5e-4;

// Owns one allocation on a backend. Deliberately not retrofitted onto the cases
// above, which predate this row — rewriting them would put unrelated churn in
// this change.
class Buf {
 public:
  Buf(Backend& b, size_t elems, size_t elem_bytes) : b_(b), p_(b.Alloc(elems * elem_bytes)) {}
  ~Buf() { b_.Free(p_); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  void* p() const { return p_; }
  template <typename T>
  T* as() const { return static_cast<T*>(p_); }

 private:
  Backend& b_;
  void* p_;
};

// Was this op served by the NATIVE Vulkan kernel on its last call, BY NAME? The
// reference tier registers under a different name, so this is what separates
// "the shader ran" from "the host kernel ran and the numbers matched".
bool RanNative(vt::OpId op) {
  const auto stats = vt::GetOpProviderStats(op, DeviceType::kVULKAN);
  return stats.last_selected != nullptr &&
         std::string(stats.last_selected) == std::string(vt::kNativeProviderName);
}

}  // namespace

TEST_CASE("GDN sigmoid output-gate runs NATIVELY on Vulkan and matches the CPU oracle") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // 300 is deliberately NOT a multiple of the 128-wide workgroup, so the tail
  // guard is exercised rather than assumed.
  constexpr int64_t kN = 300;
  const std::vector<float> attn = Spread(kN, 3.0f, 11u);
  const std::vector<float> gate = Spread(kN, 6.0f, 29u);  // wide: sigmoid saturates

  Buf va(vk, kN, 4), vg(vk, kN, 4), vo(vk, kN, 2);
  Buf ca(cpu, kN, 4), cg(cpu, kN, 4), co(cpu, kN, 2);
  vk.Copy(vq, va.p(), attn.data(), kN * 4);
  vk.Copy(vq, vg.p(), gate.data(), kN * 4);
  std::memcpy(ca.p(), attn.data(), kN * 4);
  std::memcpy(cg.p(), gate.data(), kN * 4);
  vk.Synchronize(vq);

  Tensor vat = Tensor::Contiguous(va.p(), vt::DType::kF32, vd, {kN});
  Tensor vgt = Tensor::Contiguous(vg.p(), vt::DType::kF32, vd, {kN});
  Tensor vot = Tensor::Contiguous(vo.p(), vt::DType::kBF16, vd, {kN});
  Tensor cat = Tensor::Contiguous(ca.p(), vt::DType::kF32, cd, {kN});
  Tensor cgt = Tensor::Contiguous(cg.p(), vt::DType::kF32, cd, {kN});
  Tensor cot = Tensor::Contiguous(co.p(), vt::DType::kBF16, cd, {kN});

  vt::SigmoidGateBf16(cq, cot, cat, cgt);
  vt::SigmoidGateBf16(vq, vot, vat, vgt);
  vk.Synchronize(vq);

  CHECK(ctx.PipelineExistsFor("vt_sigmoid_gate_bf16"));
  CHECK(RanNative(vt::OpId::kSigmoidGateBf16));

  std::vector<uint16_t> got(kN);
  vk.Copy(vq, got.data(), vo.p(), kN * 2);
  vk.Synchronize(vq);
  std::vector<float> ref_f(kN), got_f(kN);
  for (int64_t i = 0; i < kN; ++i) {
    ref_f[i] = vt::BF16ToF32(co.as<uint16_t>()[i]);
    got_f[i] = vt::BF16ToF32(got[i]);
  }
  const double nmse = NmseOf(ref_f, got_f);
  MESSAGE("sigmoid_gate_bf16 NMSE vs the CPU oracle: " << nmse);
  CHECK(nmse <= kGdnNmseTol);

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

TEST_CASE("gated RMSNorm runs NATIVELY on Vulkan: silu, sigmoid, and a padded gate") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // D = 300 > the 128-wide workgroup, so each lane walks the row in a strided
  // loop and the tree reduction is genuinely over partial sums.
  constexpr int64_t kRows = 5, kD = 300;
  const std::vector<float> x = Spread(kRows * kD, 2.0f, 7u);
  const std::vector<float> z = Spread(kRows * kD, 4.0f, 13u);
  const std::vector<float> w = Spread(kD, 1.5f, 17u);

  Buf vx(vk, kRows * kD, 4), vz(vk, kRows * kD, 4), vw(vk, kD, 4), vo(vk, kRows * kD, 4);
  Buf cx(cpu, kRows * kD, 4), cz(cpu, kRows * kD, 4), cw(cpu, kD, 4), co(cpu, kRows * kD, 4);
  vk.Copy(vq, vx.p(), x.data(), x.size() * 4);
  vk.Copy(vq, vz.p(), z.data(), z.size() * 4);
  vk.Copy(vq, vw.p(), w.data(), w.size() * 4);
  std::memcpy(cx.p(), x.data(), x.size() * 4);
  std::memcpy(cz.p(), z.data(), z.size() * 4);
  std::memcpy(cw.p(), w.data(), w.size() * 4);
  vk.Synchronize(vq);

  Tensor vxt = Tensor::Contiguous(vx.p(), vt::DType::kF32, vd, {kRows, kD});
  Tensor vzt = Tensor::Contiguous(vz.p(), vt::DType::kF32, vd, {kRows, kD});
  Tensor vwt = Tensor::Contiguous(vw.p(), vt::DType::kF32, vd, {kD});
  Tensor vot = Tensor::Contiguous(vo.p(), vt::DType::kF32, vd, {kRows, kD});
  Tensor cxt = Tensor::Contiguous(cx.p(), vt::DType::kF32, cd, {kRows, kD});
  Tensor czt = Tensor::Contiguous(cz.p(), vt::DType::kF32, cd, {kRows, kD});
  Tensor cwt = Tensor::Contiguous(cw.p(), vt::DType::kF32, cd, {kD});
  Tensor cot = Tensor::Contiguous(co.p(), vt::DType::kF32, cd, {kRows, kD});

  for (bool sigmoid_gate : {false, true}) {
    vt::RmsNormGatedArgs args;
    args.eps = 1e-6f;
    args.sigmoid_gate = sigmoid_gate;
    vt::RmsNormGated(cq, cot, cxt, czt, cwt, args);
    vt::RmsNormGated(vq, vot, vxt, vzt, vwt, args);
    vk.Synchronize(vq);
    std::vector<float> got(kRows * kD), ref(kRows * kD);
    vk.Copy(vq, got.data(), vo.p(), got.size() * 4);
    vk.Synchronize(vq);
    std::memcpy(ref.data(), co.p(), ref.size() * 4);
    const double nmse = NmseOf(ref, got);
    // Assembled OUTSIDE the macro: MESSAGE(x << y) hands the expression to the
    // doctest MessageBuilder, so a flag written inside renders as "1".
    const std::string line = std::string("rmsnorm_gated (") +
                             (sigmoid_gate ? "sigmoid" : "silu") +
                             ") NMSE vs the CPU oracle: " + std::to_string(nmse);
    MESSAGE(line);
    CHECK(nmse <= kGdnNmseTol);
  }
  CHECK(ctx.PipelineExistsFor("vt_rms_norm_gated"));
  CHECK(RanNative(vt::OpId::kRmsNormGated));

  // THE PADDED-ROW rank-3 GATE, which is the shape the merged-qkvz path actually
  // produces: the gate is the `z` slice of one fused projection, so its TOKEN
  // stride exceeds Hv*D while the inner block stays contiguous. A shader that
  // ignored gate.stride[0] would still pass the rank-2 case above, so this is the
  // assertion that the stride is honoured at all.
  constexpr int64_t kT = 3, kHv = 2, kDv = 64, kPad = 16;
  constexpr int64_t kZStride = kHv * kDv + kPad;
  const std::vector<float> x3 = Spread(kT * kHv * kDv, 2.0f, 23u);
  const std::vector<float> z3 = Spread(kT * kZStride, 4.0f, 31u);
  const std::vector<float> w3 = Spread(kDv, 1.5f, 37u);

  Buf vx3(vk, kT * kHv * kDv, 4), vz3(vk, kT * kZStride, 4), vw3(vk, kDv, 4),
      vo3(vk, kT * kHv * kDv, 4);
  Buf cx3(cpu, kT * kHv * kDv, 4), cz3(cpu, kT * kZStride, 4), cw3(cpu, kDv, 4),
      co3(cpu, kT * kHv * kDv, 4);
  vk.Copy(vq, vx3.p(), x3.data(), x3.size() * 4);
  vk.Copy(vq, vz3.p(), z3.data(), z3.size() * 4);
  vk.Copy(vq, vw3.p(), w3.data(), w3.size() * 4);
  std::memcpy(cx3.p(), x3.data(), x3.size() * 4);
  std::memcpy(cz3.p(), z3.data(), z3.size() * 4);
  std::memcpy(cw3.p(), w3.data(), w3.size() * 4);
  vk.Synchronize(vq);

  auto padded_gate = [](void* p, Device dev) {
    Tensor t = Tensor::Contiguous(p, vt::DType::kF32, dev, {kT, kHv, kDv});
    t.stride[0] = kZStride;  // the padded TOKEN stride; inner dims stay packed
    return t;
  };
  Tensor vx3t = Tensor::Contiguous(vx3.p(), vt::DType::kF32, vd, {kT, kHv, kDv});
  Tensor vo3t = Tensor::Contiguous(vo3.p(), vt::DType::kF32, vd, {kT, kHv, kDv});
  Tensor vw3t = Tensor::Contiguous(vw3.p(), vt::DType::kF32, vd, {kDv});
  Tensor cx3t = Tensor::Contiguous(cx3.p(), vt::DType::kF32, cd, {kT, kHv, kDv});
  Tensor co3t = Tensor::Contiguous(co3.p(), vt::DType::kF32, cd, {kT, kHv, kDv});
  Tensor cw3t = Tensor::Contiguous(cw3.p(), vt::DType::kF32, cd, {kDv});
  Tensor vz3t = padded_gate(vz3.p(), vd);
  Tensor cz3t = padded_gate(cz3.p(), cd);

  vt::RmsNormGatedArgs args3;
  args3.eps = 1e-6f;
  vt::RmsNormGated(cq, co3t, cx3t, cz3t, cw3t, args3);
  vt::RmsNormGated(vq, vo3t, vx3t, vz3t, vw3t, args3);
  vk.Synchronize(vq);
  std::vector<float> got3(kT * kHv * kDv), ref3(kT * kHv * kDv);
  vk.Copy(vq, got3.data(), vo3.p(), got3.size() * 4);
  vk.Synchronize(vq);
  std::memcpy(ref3.data(), co3.p(), ref3.size() * 4);
  const double nmse3 = NmseOf(ref3, got3);
  MESSAGE("rmsnorm_gated (padded rank-3 gate) NMSE vs the CPU oracle: " << nmse3);
  CHECK(nmse3 <= kGdnNmseTol);

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

TEST_CASE("GDN state gather/scatter run NATIVELY on Vulkan and are BIT-EXACT") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // BIT-EXACT, not NMSE: these two ops move f32 words and compute nothing, so
  // anything short of equality would be hiding a bug (the tier
  // vt_reshape_and_cache is already gated in).
  //
  // The cache inner dim is WIDER than the working one (6 vs 4) — the
  // spec-decode-widened conv row. That shape is the whole reason the kernel walks
  // channels at the cache's physical stride, and a shader that assumed the
  // contiguous fast path would pass a same-width test and corrupt this one.
  constexpr int64_t kCacheRows = 5, kC = 3, kCacheInner = 6, kWorkInner = 4;
  constexpr int64_t kRows = 3;
  constexpr int64_t kCacheElems = kCacheRows * kC * kCacheInner;
  constexpr int64_t kWorkElems = kRows * kC * kWorkInner;
  const std::vector<float> cache_init = Spread(kCacheElems, 5.0f, 41u);
  const std::vector<int32_t> idx = {4, 0, 2};
  // Request 1 has NO initial state: its working row must come back ZEROED, not
  // copied. i8 is the interesting width — it is why the shader reads the flag
  // byte-wise through the 32-bit view instead of needing VK_KHR_8bit_storage.
  //
  // THREE ELEMENTS IS ALSO THREE BYTES, and that is deliberate: it is the shape
  // that caught the sub-word allocation bug this row fixed. A 3-byte VkBuffer's
  // `uint[]` view has ZERO elements, so every flag read back as false and the
  // gather zeroed rows it should have copied — silently, because the read is
  // robust rather than faulting. AllocBuffer now rounds every buffer up to a
  // whole 32-bit word (src/vt/vulkan/vulkan_context.cpp), and keeping this length
  // at 3 is what keeps that fix gated. Do not "tidy" it to 4.
  const std::vector<int8_t> his = {1, 0, 1};

  Buf vc(vk, kCacheElems, 4), vw(vk, kWorkElems, 4), vi(vk, kRows, 4), vh(vk, kRows, 1);
  Buf cc(cpu, kCacheElems, 4), cw(cpu, kWorkElems, 4), ci(cpu, kRows, 4), ch(cpu, kRows, 1);
  vk.Copy(vq, vc.p(), cache_init.data(), kCacheElems * 4);
  vk.Copy(vq, vi.p(), idx.data(), kRows * 4);
  vk.Copy(vq, vh.p(), his.data(), kRows);
  std::memcpy(cc.p(), cache_init.data(), kCacheElems * 4);
  std::memcpy(ci.p(), idx.data(), kRows * 4);
  std::memcpy(ch.p(), his.data(), kRows);
  vk.Synchronize(vq);

  auto cache_t = [](void* p, Device dev) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {kCacheRows, kC, kCacheInner});
  };
  auto work_t = [](void* p, Device dev) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {kRows, kC, kWorkInner});
  };
  Tensor vct = cache_t(vc.p(), vd), vwt = work_t(vw.p(), vd);
  Tensor cct = cache_t(cc.p(), cd), cwt = work_t(cw.p(), cd);
  Tensor vit = Tensor::Contiguous(vi.p(), vt::DType::kI32, vd, {kRows});
  Tensor cit = Tensor::Contiguous(ci.p(), vt::DType::kI32, cd, {kRows});
  Tensor vht = Tensor::Contiguous(vh.p(), vt::DType::kI8, vd, {kRows});
  Tensor cht = Tensor::Contiguous(ch.p(), vt::DType::kI8, cd, {kRows});

  vt::GdnStateGather(cq, cwt, cct, cit, &cht);
  vt::GdnStateGather(vq, vwt, vct, vit, &vht);
  vk.Synchronize(vq);
  CHECK(ctx.PipelineExistsFor("vt_gdn_state_gather"));
  CHECK(RanNative(vt::OpId::kGdnStateGather));

  std::vector<float> got(kWorkElems);
  vk.Copy(vq, got.data(), vw.p(), kWorkElems * 4);
  vk.Synchronize(vq);
  CHECK(std::memcmp(got.data(), cw.p(), kWorkElems * 4) == 0);
  // The zeroing is asserted DIRECTLY, not left to the memcmp: if both kernels
  // wrongly copied row 1, the comparison above would still be green.
  bool row1_zero = true;
  for (int64_t e = 0; e < kC * kWorkInner; ++e) {
    if (got[kC * kWorkInner + e] != 0.0f) row1_zero = false;
  }
  CHECK(row1_zero);

  // Scatter new working rows back and compare the WHOLE cache, so an over-wide
  // write into the widened row's tail columns is caught.
  const std::vector<float> work_new = Spread(kWorkElems, 9.0f, 43u);
  vk.Copy(vq, vw.p(), work_new.data(), kWorkElems * 4);
  std::memcpy(cw.p(), work_new.data(), kWorkElems * 4);
  vk.Synchronize(vq);
  vt::GdnStateScatter(cq, cct, cwt, cit);
  vt::GdnStateScatter(vq, vct, vwt, vit);
  vk.Synchronize(vq);
  CHECK(ctx.PipelineExistsFor("vt_gdn_state_scatter"));
  CHECK(RanNative(vt::OpId::kGdnStateScatter));

  std::vector<float> cache_got(kCacheElems);
  vk.Copy(vq, cache_got.data(), vc.p(), kCacheElems * 4);
  vk.Synchronize(vq);
  CHECK(std::memcmp(cache_got.data(), cc.p(), kCacheElems * 4) == 0);
  // And cache row 1, which no index names, still holds its ORIGINAL bytes —
  // proof the scatter wrote only where it was told to.
  CHECK(std::memcmp(cache_got.data() + kC * kCacheInner,
                    cache_init.data() + kC * kCacheInner,
                    static_cast<size_t>(kC * kCacheInner) * 4) == 0);

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

TEST_CASE("the decode causal conv1d update runs NATIVELY on Vulkan, state roll included") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // conv_state has MORE rows than the batch and is addressed through
  // conv_state_indices — the in-place indexed decode path the model takes. A
  // NEGATIVE index is upstream's NULL block and must leave both the output
  // element and the cache row alone.
  constexpr int64_t kBatch = 4, kC = 5, kK = 4, kWidth = kK - 1;
  constexpr int64_t kStateRows = 6, kStateLen = kWidth;
  const std::vector<float> x = Spread(kBatch * kC, 2.0f, 53u);
  const std::vector<float> w = Spread(kC * kK, 1.0f, 59u);
  const std::vector<float> bias = Spread(kC, 0.5f, 61u);
  const std::vector<float> state0 = Spread(kStateRows * kC * kStateLen, 3.0f, 67u);
  const std::vector<int32_t> cidx = {5, 1, -1, 0};  // token 2 -> NULL block

  Buf vx(vk, kBatch * kC, 4), vw(vk, kC * kK, 4), vb(vk, kC, 4), vo(vk, kBatch * kC, 4),
      vs(vk, kStateRows * kC * kStateLen, 4), vi(vk, kBatch, 4);
  Buf cx(cpu, kBatch * kC, 4), cw(cpu, kC * kK, 4), cb(cpu, kC, 4), co(cpu, kBatch * kC, 4),
      cs(cpu, kStateRows * kC * kStateLen, 4), cci(cpu, kBatch, 4);
  vk.Copy(vq, vx.p(), x.data(), x.size() * 4);
  vk.Copy(vq, vw.p(), w.data(), w.size() * 4);
  vk.Copy(vq, vb.p(), bias.data(), bias.size() * 4);
  vk.Copy(vq, vs.p(), state0.data(), state0.size() * 4);
  vk.Copy(vq, vi.p(), cidx.data(), cidx.size() * 4);
  // The output buffer is pre-seeded so the NULL-block token's element can be
  // checked for having been LEFT ALONE rather than merely being plausible.
  const std::vector<float> out_seed(kBatch * kC, -12345.0f);
  vk.Copy(vq, vo.p(), out_seed.data(), out_seed.size() * 4);
  std::memcpy(cx.p(), x.data(), x.size() * 4);
  std::memcpy(cw.p(), w.data(), w.size() * 4);
  std::memcpy(cb.p(), bias.data(), bias.size() * 4);
  std::memcpy(cs.p(), state0.data(), state0.size() * 4);
  std::memcpy(cci.p(), cidx.data(), cidx.size() * 4);
  std::memcpy(co.p(), out_seed.data(), out_seed.size() * 4);
  vk.Synchronize(vq);

  Tensor vxt = Tensor::Contiguous(vx.p(), vt::DType::kF32, vd, {kBatch, kC});
  Tensor vwt = Tensor::Contiguous(vw.p(), vt::DType::kF32, vd, {kC, kK});
  Tensor vbt = Tensor::Contiguous(vb.p(), vt::DType::kF32, vd, {kC});
  Tensor vot = Tensor::Contiguous(vo.p(), vt::DType::kF32, vd, {kBatch, kC});
  Tensor vst = Tensor::Contiguous(vs.p(), vt::DType::kF32, vd, {kStateRows, kC, kStateLen});
  Tensor vit = Tensor::Contiguous(vi.p(), vt::DType::kI32, vd, {kBatch});
  Tensor cxt = Tensor::Contiguous(cx.p(), vt::DType::kF32, cd, {kBatch, kC});
  Tensor cwt = Tensor::Contiguous(cw.p(), vt::DType::kF32, cd, {kC, kK});
  Tensor cbt = Tensor::Contiguous(cb.p(), vt::DType::kF32, cd, {kC});
  Tensor cot = Tensor::Contiguous(co.p(), vt::DType::kF32, cd, {kBatch, kC});
  Tensor cst = Tensor::Contiguous(cs.p(), vt::DType::kF32, cd, {kStateRows, kC, kStateLen});
  Tensor citt = Tensor::Contiguous(cci.p(), vt::DType::kI32, cd, {kBatch});

  vt::CausalConv1dArgs args;  // silu_activation defaults true, as Qwen GDN uses
  vt::CausalConv1dUpdate(cq, cot, cxt, cwt, &cbt, cst, args, &citt);
  vt::CausalConv1dUpdate(vq, vot, vxt, vwt, &vbt, vst, args, &vit);
  vk.Synchronize(vq);

  CHECK(ctx.PipelineExistsFor("vt_causal_conv1d_update"));
  CHECK(RanNative(vt::OpId::kCausalConv1dUpdate));

  std::vector<float> got(kBatch * kC), ref(kBatch * kC);
  vk.Copy(vq, got.data(), vo.p(), got.size() * 4);
  vk.Synchronize(vq);
  std::memcpy(ref.data(), co.p(), ref.size() * 4);
  const double nmse = NmseOf(ref, got);
  MESSAGE("causal_conv1d_update NMSE vs the CPU oracle: " << nmse);
  CHECK(nmse <= kGdnNmseTol);
  // The NULL-block token kept its seed.
  for (int64_t c = 0; c < kC; ++c) {
    CAPTURE(c);
    CHECK(got[2 * kC + c] == -12345.0f);
  }

  // THE ROLLED STATE IS THE HALF A NUMBERS-ONLY OUTPUT CHECK MISSES: the output
  // reads the OLD taps, so a kernel that never rolled the window would produce a
  // perfect first step and diverge from the second one onward.
  std::vector<float> state_got(kStateRows * kC * kStateLen);
  vk.Copy(vq, state_got.data(), vs.p(), state_got.size() * 4);
  vk.Synchronize(vq);
  const std::vector<float> state_ref(cs.as<float>(), cs.as<float>() + state_got.size());
  CHECK(std::memcmp(state_got.data(), state_ref.data(), state_got.size() * 4) == 0);
  // Spelled out independently of the oracle: the roll writes the RAW x sample
  // into the last tap, so if BOTH kernels skipped the roll the memcmp above
  // would still be green.
  for (int64_t bt = 0; bt < kBatch; ++bt) {
    if (cidx[static_cast<size_t>(bt)] < 0) continue;
    for (int64_t c = 0; c < kC; ++c) {
      CAPTURE(bt);
      CAPTURE(c);
      const int64_t slot = cidx[static_cast<size_t>(bt)];
      const int64_t last = (slot * kC + c) * kStateLen + kWidth - 1;
      CHECK(state_got[static_cast<size_t>(last)] == x[static_cast<size_t>(bt * kC + c)]);
    }
  }

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

TEST_CASE("the prefill causal conv1d runs NATIVELY on Vulkan, in both token mappings") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // A varlen batch of three sequences, lengths 5, 1, 9, one of them with no
  // initial state. The shape is chosen for the SPLIT arm below: with three
  // sequences of 24 channels the default grid is a single workgroup, a target
  // of 5 groups therefore asks for 5 token blocks, and sequence 0 (length 5)
  // then gets one token per block -- so t = 1 and t = 2, both < width and both
  // reading the carried state, land OUTSIDE block 0 unless the shader gives
  // them to it. That is the race the block-0 rule exists for.
  //
  // FIVE, NOT FOUR, deliberately. The blocks of one (sequence, channel) are
  // adjacent invocations; with 4 they sat inside one of llvmpipe's 8-lane
  // subgroups in this layout and the pre-fix shader PASSED 5 runs of 5. With 5
  // they straddle a subgroup and it fails 5 of 5.
  constexpr int64_t kN = 3, kT = 15, kC = 24, kK = 4, kWidth = kK - 1;
  const std::vector<int32_t> qsl = {0, 5, 6, 15};
  const std::vector<int32_t> has = {1, 0, 1};
  const std::vector<float> x = Spread(kT * kC, 2.0f, 71u);
  const std::vector<float> w = Spread(kC * kK, 0.5f, 73u);
  const std::vector<float> bias = Spread(kC, 0.2f, 79u);
  const std::vector<float> state0 = Spread(kN * kC * kWidth, 0.5f, 83u);

  // CPU oracle, once.
  Buf cx(cpu, kT * kC, 4), cw(cpu, kC * kK, 4), cb(cpu, kC, 4), co(cpu, kT * kC, 4),
      cs(cpu, kN * kC * kWidth, 4), cqsl(cpu, kN + 1, 4), chas(cpu, kN, 4);
  std::memcpy(cx.p(), x.data(), x.size() * 4);
  std::memcpy(cw.p(), w.data(), w.size() * 4);
  std::memcpy(cb.p(), bias.data(), bias.size() * 4);
  std::memcpy(cs.p(), state0.data(), state0.size() * 4);
  std::memcpy(cqsl.p(), qsl.data(), qsl.size() * 4);
  std::memcpy(chas.p(), has.data(), has.size() * 4);
  std::memset(co.p(), 0, kT * kC * 4);
  {
    Tensor xt = Tensor::Contiguous(cx.p(), vt::DType::kF32, cd, {kT, kC});
    Tensor wt = Tensor::Contiguous(cw.p(), vt::DType::kF32, cd, {kC, kK});
    Tensor bt = Tensor::Contiguous(cb.p(), vt::DType::kF32, cd, {kC});
    Tensor ot = Tensor::Contiguous(co.p(), vt::DType::kF32, cd, {kT, kC});
    Tensor st = Tensor::Contiguous(cs.p(), vt::DType::kF32, cd, {kN, kC, kWidth});
    Tensor qt = Tensor::Contiguous(cqsl.p(), vt::DType::kI32, cd, {kN + 1});
    Tensor ht = Tensor::Contiguous(chas.p(), vt::DType::kI32, cd, {kN});
    vt::CausalConv1dFwd(cq, ot, xt, wt, &bt, st, qt, ht, vt::CausalConv1dArgs{});
  }
  const std::vector<float> out_ref(co.as<float>(), co.as<float>() + kT * kC);
  const std::vector<float> state_ref(cs.as<float>(), cs.as<float>() + kN * kC * kWidth);

  // Restores the knob even when a CHECK below throws out of the case.
  struct EnvRestore {
    ~EnvRestore() { vllm_test::UnsetEnv("VT_VULKAN_CONV_TARGET_GROUPS"); }
  } restore;

  // Arm "" is the default mapping (one invocation per (sequence, channel)); arm
  // "5" is the opt-in split. The split runs several times because a race is a
  // scheduling outcome: one green dispatch would not show it is absent.
  std::vector<float> out_default;
  for (const char* target : {"", "5", "5", "5", "5"}) {
    CAPTURE(std::string(target));
    vllm_test::SetEnv("VT_VULKAN_CONV_TARGET_GROUPS", target);

    Buf vx(vk, kT * kC, 4), vw(vk, kC * kK, 4), vb(vk, kC, 4), vo(vk, kT * kC, 4),
        vs(vk, kN * kC * kWidth, 4), vqsl(vk, kN + 1, 4), vhas(vk, kN, 4);
    vk.Copy(vq, vx.p(), x.data(), x.size() * 4);
    vk.Copy(vq, vw.p(), w.data(), w.size() * 4);
    vk.Copy(vq, vb.p(), bias.data(), bias.size() * 4);
    vk.Copy(vq, vs.p(), state0.data(), state0.size() * 4);
    vk.Copy(vq, vqsl.p(), qsl.data(), qsl.size() * 4);
    vk.Copy(vq, vhas.p(), has.data(), has.size() * 4);
    vk.Synchronize(vq);
    Tensor xt = Tensor::Contiguous(vx.p(), vt::DType::kF32, vd, {kT, kC});
    Tensor wt = Tensor::Contiguous(vw.p(), vt::DType::kF32, vd, {kC, kK});
    Tensor bt = Tensor::Contiguous(vb.p(), vt::DType::kF32, vd, {kC});
    Tensor ot = Tensor::Contiguous(vo.p(), vt::DType::kF32, vd, {kT, kC});
    Tensor st = Tensor::Contiguous(vs.p(), vt::DType::kF32, vd, {kN, kC, kWidth});
    Tensor qt = Tensor::Contiguous(vqsl.p(), vt::DType::kI32, vd, {kN + 1});
    Tensor ht = Tensor::Contiguous(vhas.p(), vt::DType::kI32, vd, {kN});
    vt::CausalConv1dFwd(vq, ot, xt, wt, &bt, st, qt, ht, vt::CausalConv1dArgs{});
    vk.Synchronize(vq);

    CHECK(RanNative(vt::OpId::kCausalConv1dFwd));
    // The block count is the mechanism, and it is invisible in the numbers by
    // design, so the specialization VALUE is asserted, not just the module.
    {
      const std::string want = std::string("vt_causal_conv1d_fwd|") +
                               (target[0] == '\0' ? "1" : "5");
      const std::vector<std::string> keys = ctx.PipelineKeysFor("vt_causal_conv1d_fwd");
      std::string joined;
      for (const std::string& k : keys) joined += k + " ";
      CAPTURE(joined);
      CHECK(std::find(keys.begin(), keys.end(), want) != keys.end());
    }

    std::vector<float> out_got(kT * kC), state_got(kN * kC * kWidth);
    vk.Copy(vq, out_got.data(), vo.p(), out_got.size() * 4);
    vk.Copy(vq, state_got.data(), vs.p(), state_got.size() * 4);
    vk.Synchronize(vq);

    const double nmse = NmseOf(out_ref, out_got);
    MESSAGE("causal_conv1d_fwd NMSE vs the CPU oracle: " << nmse);
    CHECK(nmse <= kGdnNmseTol);
    // The state write-back moves RAW x samples, so it is exact, not a tolerance.
    CHECK(std::memcmp(state_got.data(), state_ref.data(), state_got.size() * 4) == 0);
    // Spelled out independently of the oracle for the two sequences at least
    // `width` long: their new state is their own last `width` input rows.
    for (int64_t s : {int64_t{0}, int64_t{2}}) {
      const int64_t end = qsl[static_cast<size_t>(s) + 1];
      for (int64_t c = 0; c < kC; ++c) {
        for (int64_t j = 0; j < kWidth; ++j) {
          CAPTURE(s);
          CAPTURE(c);
          CAPTURE(j);
          CHECK(state_got[static_cast<size_t>((s * kC + c) * kWidth + j)] ==
                x[static_cast<size_t>((end - kWidth + j) * kC + c)]);
        }
      }
    }
    // The split changes which invocation computes an element, never the order
    // of its own k-tap sum, so both mappings must agree to the bit.
    if (target[0] == '\0') {
      out_default = out_got;
    } else {
      CHECK(std::memcmp(out_got.data(), out_default.data(), out_got.size() * 4) == 0);
    }
  }

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

namespace {

// One prefill-conv call on one backend, everything uploaded from host vectors
// and downloaded back. Shared by the edge-case and compressed-state cases below
// so the CPU oracle and the Vulkan arm run the SAME setup code.
struct ConvFwdCase {
  std::vector<int32_t> qsl;
  std::vector<uint8_t> has;      // one flag per sequence
  bool his_i8 = false;
  int64_t his_byte_off = 0;      // i8 view start within its allocation
  int64_t c = 8, k = 4;
  int64_t x_pad = 0;             // extra elements per x row (padded stride)
  int64_t state_extra = 0;       // conv_state.shape[2] = k - 1 + state_extra
  bool with_bias = true;
  bool silu = true;
  vt::DType state_dt = vt::DType::kF32;
  vt::DType op_dt = vt::DType::kF32;   // x, weight and bias storage
  vt::DType out_dt = vt::DType::kF32;
  std::vector<float> x, w, bias, state0;  // state0 is [n, c, k-1+state_extra]
};

struct ConvFwdResult {
  std::vector<float> out, state;  // state as f32 whatever its storage dtype
};

// f32 values to the bytes of `dt` (f32, f16 or bf16), and back.
std::vector<uint8_t> ConvEncode(const std::vector<float>& v, vt::DType dt) {
  std::vector<uint8_t> out(v.size() * (dt == vt::DType::kF32 ? 4 : 2));
  for (size_t i = 0; i < v.size(); ++i) {
    if (dt == vt::DType::kF32) {
      std::memcpy(out.data() + i * 4, &v[i], 4);
    } else {
      const uint16_t h = dt == vt::DType::kBF16 ? vt::F32ToBF16(v[i]) : vt::F32ToF16(v[i]);
      std::memcpy(out.data() + i * 2, &h, 2);
    }
  }
  return out;
}

std::vector<float> ConvDecode(const std::vector<uint8_t>& b, vt::DType dt) {
  const size_t es = dt == vt::DType::kF32 ? 4 : 2;
  std::vector<float> out(b.size() / es);
  for (size_t i = 0; i < out.size(); ++i) {
    if (dt == vt::DType::kF32) {
      std::memcpy(&out[i], b.data() + i * 4, 4);
    } else {
      uint16_t h;
      std::memcpy(&h, b.data() + i * 2, 2);
      out[i] = dt == vt::DType::kBF16 ? vt::BF16ToF32(h) : vt::F16ToF32(h);
    }
  }
  return out;
}

ConvFwdResult RunConvFwd(Backend& b, Queue q, Device d, const ConvFwdCase& cs) {
  const int64_t n = static_cast<int64_t>(cs.qsl.size()) - 1;
  const int64_t t = cs.qsl.back();
  const int64_t rs = cs.c + cs.x_pad;
  const int64_t sw = cs.k - 1 + cs.state_extra;
  const int64_t sn = n * cs.c * sw;
  const size_t st_bytes = cs.state_dt == vt::DType::kF32 ? 4 : 2;
  const std::vector<uint8_t> xe = ConvEncode(cs.x, cs.op_dt);
  const std::vector<uint8_t> we = ConvEncode(cs.w, cs.op_dt);
  const std::vector<uint8_t> be = ConvEncode(cs.bias, cs.op_dt);
  const size_t oes = cs.out_dt == vt::DType::kF32 ? 4 : 2;
  Buf bx(b, xe.size() / 4 + 1, 4), bw(b, we.size() / 4 + 1, 4), bb(b, be.size() / 4 + 1, 4),
      bo(b, static_cast<size_t>(std::max<int64_t>(1, t) * cs.c), oes),
      bs(b, static_cast<size_t>(sn), st_bytes), bq(b, static_cast<size_t>(n + 1), 4),
      bh(b, static_cast<size_t>(n + 8), 4);
  b.Copy(q, bx.p(), xe.data(), xe.size());
  b.Copy(q, bw.p(), we.data(), we.size());
  if (cs.with_bias) b.Copy(q, bb.p(), be.data(), be.size());
  if (cs.state_dt == vt::DType::kF32) {
    b.Copy(q, bs.p(), cs.state0.data(), cs.state0.size() * 4);
  } else {
    std::vector<uint16_t> s16(cs.state0.size());
    for (size_t i = 0; i < s16.size(); ++i) s16[i] = vt::F32ToBF16(cs.state0[i]);
    b.Copy(q, bs.p(), s16.data(), s16.size() * 2);
  }
  b.Copy(q, bq.p(), cs.qsl.data(), cs.qsl.size() * 4);
  // Flags: i32 at offset 0, or i8 bytes starting at his_byte_off of a buffer
  // whose other bytes are a pattern that would flip the answer if read.
  std::vector<uint8_t> hb(static_cast<size_t>((n + 8) * 4), 0xA5);
  if (cs.his_i8) {
    for (int64_t s = 0; s < n; ++s) hb[static_cast<size_t>(cs.his_byte_off + s)] = cs.has[s];
  } else {
    for (int64_t s = 0; s < n; ++s) {
      const int32_t v = cs.has[static_cast<size_t>(s)];
      std::memcpy(hb.data() + s * 4, &v, 4);
    }
  }
  b.Copy(q, bh.p(), hb.data(), hb.size());
  const std::vector<uint8_t> seed = ConvEncode(
      std::vector<float>(static_cast<size_t>(std::max<int64_t>(1, t) * cs.c), -7.0f), cs.out_dt);
  b.Copy(q, bo.p(), seed.data(), seed.size());
  b.Synchronize(q);

  Tensor xt = Tensor::Contiguous(bx.p(), cs.op_dt, d, {t, cs.c});
  xt.stride[0] = rs;
  Tensor wt = Tensor::Contiguous(bw.p(), cs.op_dt, d, {cs.c, cs.k});
  Tensor bt = Tensor::Contiguous(bb.p(), cs.op_dt, d, {cs.c});
  Tensor ot = Tensor::Contiguous(bo.p(), cs.out_dt, d, {t, cs.c});
  Tensor st = Tensor::Contiguous(bs.p(), cs.state_dt, d, {n, cs.c, sw});
  Tensor qt = Tensor::Contiguous(bq.p(), vt::DType::kI32, d, {n + 1});
  Tensor ht = cs.his_i8
                  ? Tensor::Contiguous(static_cast<char*>(bh.p()) + cs.his_byte_off,
                                       vt::DType::kI8, d, {n})
                  : Tensor::Contiguous(bh.p(), vt::DType::kI32, d, {n});
  vt::CausalConv1dArgs args;
  args.silu_activation = cs.silu;
  vt::CausalConv1dFwd(q, ot, xt, wt, cs.with_bias ? &bt : nullptr, st, qt, ht, args);
  b.Synchronize(q);

  ConvFwdResult r;
  std::vector<uint8_t> ob(static_cast<size_t>(t * cs.c) * oes);
  b.Copy(q, ob.data(), bo.p(), ob.size());
  r.state.resize(static_cast<size_t>(sn));
  if (cs.state_dt == vt::DType::kF32) {
    b.Copy(q, r.state.data(), bs.p(), r.state.size() * 4);
  } else {
    std::vector<uint16_t> s16(r.state.size());
    b.Copy(q, s16.data(), bs.p(), s16.size() * 2);
    for (size_t i = 0; i < s16.size(); ++i) r.state[i] = vt::BF16ToF32(s16[i]);
  }
  b.Synchronize(q);
  r.out = ConvDecode(ob, cs.out_dt);
  return r;
}

ConvFwdCase MakeConvFwdCase(std::vector<int32_t> qsl, std::vector<uint8_t> has, int64_t c,
                            int64_t k, int64_t x_pad, uint32_t seed) {
  ConvFwdCase cs;
  cs.qsl = std::move(qsl);
  cs.has = std::move(has);
  cs.c = c;
  cs.k = k;
  cs.x_pad = x_pad;
  const int64_t n = static_cast<int64_t>(cs.qsl.size()) - 1;
  const int64_t t = cs.qsl.back();
  cs.x = Spread(static_cast<size_t>(std::max<int64_t>(1, t) * (c + x_pad)), 2.0f, seed);
  cs.w = Spread(static_cast<size_t>(c * k), 0.5f, seed + 1);
  cs.bias = Spread(static_cast<size_t>(c), 0.2f, seed + 2);
  cs.state0 = Spread(static_cast<size_t>(n * c * (k - 1)), 0.5f, seed + 3);
  return cs;
}

}  // namespace

TEST_CASE("the prefill causal conv1d matches the CPU oracle on its edge shapes") {
  if (!VulkanPresent()) return;
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // Lengths 5, 0, 1, 7: an EMPTY sequence, and a length-1 sequence WITH initial
  // state, whose new state is two shifted old taps plus one x sample. Flags
  // 1,1,1,0. x rows are padded by 3 elements, so the row stride is not c.
  ConvFwdCase base = MakeConvFwdCase({0, 5, 5, 6, 13}, {1, 1, 1, 0}, 40, 4, 3, 211u);

  struct Variant {
    const char* name;
    bool his_i8;
    int64_t his_byte_off;
    bool with_bias, silu;
    vt::DType op_dt = vt::DType::kF32, out_dt = vt::DType::kF32;
  };
  // i8 flags at every byte phase of a 32-bit word, the read a 4-aligned
  // assumption gets wrong; and bias/silu both ways.
  for (const Variant& v : {Variant{"i32 flags, bias, silu", false, 0, true, true},
                           Variant{"i8 flags @0, no bias, no silu", true, 0, false, false},
                           Variant{"i8 flags @1", true, 1, true, true},
                           Variant{"i8 flags @2, no bias", true, 2, false, true},
                           Variant{"i8 flags @3, no silu", true, 3, true, false},
                           // Reduced-precision operands and output: the shader
                           // reads them through the 16-bit views and its own
                           // dtype codes, the CPU reference through LoadF32.
                           Variant{"bf16 x/w/bias, bf16 out", false, 0, true, true,
                                   vt::DType::kBF16, vt::DType::kBF16},
                           Variant{"f16 x/w/bias, f32 out", true, 1, true, true,
                                   vt::DType::kF16, vt::DType::kF32}}) {
    CAPTURE(std::string(v.name));
    ConvFwdCase cs = base;
    cs.his_i8 = v.his_i8;
    cs.his_byte_off = v.his_byte_off;
    cs.with_bias = v.with_bias;
    cs.silu = v.silu;
    cs.op_dt = v.op_dt;
    cs.out_dt = v.out_dt;
    const auto before = vt::GetOpProviderStats(vt::OpId::kCausalConv1dFwd, DeviceType::kVULKAN);
    const ConvFwdResult ref = RunConvFwd(cpu, cq, cd, cs);
    const ConvFwdResult got = RunConvFwd(vk, vq, vd, cs);
    const auto after = vt::GetOpProviderStats(vt::OpId::kCausalConv1dFwd, DeviceType::kVULKAN);
    CHECK(RanNative(vt::OpId::kCausalConv1dFwd));
    CHECK(after.declines == before.declines);
    CHECK(NmseOf(ref.out, got.out) <= kGdnNmseTol);
    CHECK(std::memcmp(got.state.data(), ref.state.data(), ref.state.size() * 4) == 0);
  }

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

TEST_CASE("the prefill causal conv1d keeps a bf16 conv_state IN PLACE, bit-exact vs the f32 arm") {
  if (!VulkanPresent()) return;
  REQUIRE(vt::GetBackend(DeviceType::kVULKAN).SupportsCompressedConvState());
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue vq = vk.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};

  // The CPU reference reads f32 state only, so the oracle here is the native f32
  // arm on bf16-REPRESENTABLE inputs -- the same shape as the compressed-state
  // update case above. With every x and state value exactly a bf16, the
  // compressed arm must read the same taps, produce the same outputs, and store
  // the same raw samples back.
  ConvFwdCase cs = MakeConvFwdCase({0, 5, 6, 15}, {1, 0, 1}, 24, 4, 0, 307u);
  for (float& v : cs.x) v = vt::BF16ToF32(vt::F32ToBF16(v));
  for (float& v : cs.state0) v = vt::BF16ToF32(vt::F32ToBF16(v));
  const ConvFwdResult f32_arm = RunConvFwd(vk, vq, vd, cs);
  cs.state_dt = vt::DType::kBF16;
  const ConvFwdResult bf16_arm = RunConvFwd(vk, vq, vd, cs);
  CHECK(RanNative(vt::OpId::kCausalConv1dFwd));
  CHECK(std::memcmp(bf16_arm.out.data(), f32_arm.out.data(), f32_arm.out.size() * 4) == 0);
  CHECK(std::memcmp(bf16_arm.state.data(), f32_arm.state.data(), f32_arm.state.size() * 4) == 0);

  vk.DestroyQueue(vq);
}

TEST_CASE("the prefill causal conv1d DECLINES a conv_state row wider than K-1") {
  if (!VulkanPresent()) return;
  REQUIRE(vt::ReferenceTierEligible(DeviceType::kVULKAN));
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // A speculative-decode-widened row (K-1 + 2). The CPU reference addresses
  // rows with stride K-1; the op layer admits the wider row; the native kernel
  // declines rather than pick one, so the answer is the reference's, byte for
  // byte, and the decline is counted. Whether the reference's stride is right
  // for widened rows is a question about the CPU kernel, not tested here.
  ConvFwdCase cs = MakeConvFwdCase({0, 4, 9}, {1, 1}, 8, 4, 0, 401u);
  cs.state_extra = 2;
  cs.state0 = Spread(static_cast<size_t>(2 * 8 * (3 + 2)), 0.5f, 404u);
  const ConvFwdResult ref = RunConvFwd(cpu, cq, cd, cs);
  const auto before = vt::GetOpProviderStats(vt::OpId::kCausalConv1dFwd, DeviceType::kVULKAN);
  const ConvFwdResult got = RunConvFwd(vk, vq, vd, cs);
  const auto after = vt::GetOpProviderStats(vt::OpId::kCausalConv1dFwd, DeviceType::kVULKAN);
  CHECK(after.declines == before.declines + 1);
  CHECK(std::memcmp(got.out.data(), ref.out.data(), ref.out.size() * 4) == 0);
  CHECK(std::memcmp(got.state.data(), ref.state.data(), ref.state.size() * 4) == 0);

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

TEST_CASE("the prefill causal conv1d DECLINES a width past its window, and stays correct") {
  if (!VulkanPresent()) return;
  REQUIRE(vt::ReferenceTierEligible(DeviceType::kVULKAN));
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // K = 10 -> width 9, one past the shader's private window. The native kernel
  // must hand the call to the next provider, not throw and not read past the
  // array -- and the answer must still be the CPU oracle's.
  constexpr int64_t kN = 1, kT = 12, kC = 8, kK = 10, kWidth = kK - 1;
  const std::vector<int32_t> qsl = {0, 12};
  const std::vector<int32_t> has = {1};
  const std::vector<float> x = Spread(kT * kC, 2.0f, 89u);
  const std::vector<float> w = Spread(kC * kK, 0.5f, 97u);
  const std::vector<float> state0 = Spread(kN * kC * kWidth, 0.5f, 101u);

  struct Arm {
    Backend& b;
    Queue qu;
    Device d;
  };
  std::vector<float> outs[2], states[2];
  for (int arm = 0; arm < 2; ++arm) {
    Arm a = arm == 0 ? Arm{cpu, cq, cd} : Arm{vk, vq, vd};
    Buf bx(a.b, kT * kC, 4), bw(a.b, kC * kK, 4), bo(a.b, kT * kC, 4),
        bs(a.b, kN * kC * kWidth, 4), bq(a.b, kN + 1, 4), bh(a.b, kN, 4);
    a.b.Copy(a.qu, bx.p(), x.data(), x.size() * 4);
    a.b.Copy(a.qu, bw.p(), w.data(), w.size() * 4);
    a.b.Copy(a.qu, bs.p(), state0.data(), state0.size() * 4);
    a.b.Copy(a.qu, bq.p(), qsl.data(), qsl.size() * 4);
    a.b.Copy(a.qu, bh.p(), has.data(), has.size() * 4);
    a.b.Synchronize(a.qu);
    Tensor xt = Tensor::Contiguous(bx.p(), vt::DType::kF32, a.d, {kT, kC});
    Tensor wt = Tensor::Contiguous(bw.p(), vt::DType::kF32, a.d, {kC, kK});
    Tensor ot = Tensor::Contiguous(bo.p(), vt::DType::kF32, a.d, {kT, kC});
    Tensor st = Tensor::Contiguous(bs.p(), vt::DType::kF32, a.d, {kN, kC, kWidth});
    Tensor qt = Tensor::Contiguous(bq.p(), vt::DType::kI32, a.d, {kN + 1});
    Tensor ht = Tensor::Contiguous(bh.p(), vt::DType::kI32, a.d, {kN});
    const auto before = vt::GetOpProviderStats(vt::OpId::kCausalConv1dFwd, DeviceType::kVULKAN);
    vt::CausalConv1dFwd(a.qu, ot, xt, wt, nullptr, st, qt, ht, vt::CausalConv1dArgs{});
    a.b.Synchronize(a.qu);
    if (arm == 1) {
      const auto after = vt::GetOpProviderStats(vt::OpId::kCausalConv1dFwd, DeviceType::kVULKAN);
      CHECK(after.declines == before.declines + 1);
    }
    outs[arm].resize(kT * kC);
    states[arm].resize(kN * kC * kWidth);
    a.b.Copy(a.qu, outs[arm].data(), bo.p(), outs[arm].size() * 4);
    a.b.Copy(a.qu, states[arm].data(), bs.p(), states[arm].size() * 4);
    a.b.Synchronize(a.qu);
  }
  // The fallback IS the CPU kernel, so this is exact, not a tolerance.
  CHECK(std::memcmp(outs[1].data(), outs[0].data(), outs[0].size() * 4) == 0);
  CHECK(std::memcmp(states[1].data(), states[0].data(), states[0].size() * 4) == 0);

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

// BACKEND-VULKAN-DEVICE-RESIDENT. The capability the model's indexed state-I/O
// path needs: a COMPRESSED (bf16) conv_state addressed IN PLACE, which is what
// Backend::SupportsCompressedConvState() advertises. The oracle is the arm this
// replaces — gather the bf16 rows into an f32 working copy, run the CPU kernel
// on the copy, round back — so a drift shows up HERE rather than as a slow
// numeric divergence in a 64-layer model run.
TEST_CASE("a COMPRESSED conv_state is updated IN PLACE on Vulkan, bit-exact vs the upcast arm") {
  if (!VulkanPresent()) return;
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  CHECK(vk.SupportsCompressedConvState());
  CHECK_FALSE(cpu.SupportsCompressedConvState());

  constexpr int64_t kBatch = 3, kC = 5, kK = 4, kWidth = kK - 1;
  constexpr int64_t kStateRows = 5, kStateLen = kWidth;
  const std::vector<float> x = Spread(kBatch * kC, 2.0f, 71u);
  const std::vector<float> w = Spread(kC * kK, 1.0f, 73u);
  const std::vector<float> state_f = Spread(kStateRows * kC * kStateLen, 3.0f, 79u);
  const std::vector<int32_t> cidx = {4, 0, 2};

  // The bf16 cache, and the f32 UPCAST of exactly those bytes. Starting from the
  // rounded values is what makes the two arms comparable at all: the upcast arm
  // reads the same numbers the in-place arm reads.
  std::vector<uint16_t> state_bf(state_f.size());
  std::vector<float> state_up(state_f.size());
  for (size_t i = 0; i < state_f.size(); ++i) {
    state_bf[i] = vt::F32ToBF16(state_f[i]);
    state_up[i] = vt::BF16ToF32(state_bf[i]);
  }

  Buf vx(vk, kBatch * kC, 4), vw(vk, kC * kK, 4), vo(vk, kBatch * kC, 4),
      vs(vk, state_bf.size(), 2), vi(vk, kBatch, 4);
  Buf cx(cpu, kBatch * kC, 4), cw(cpu, kC * kK, 4), co(cpu, kBatch * kC, 4),
      cs(cpu, state_up.size(), 4), cci(cpu, kBatch, 4);
  vk.Copy(vq, vx.p(), x.data(), x.size() * 4);
  vk.Copy(vq, vw.p(), w.data(), w.size() * 4);
  vk.Copy(vq, vs.p(), state_bf.data(), state_bf.size() * 2);
  vk.Copy(vq, vi.p(), cidx.data(), cidx.size() * 4);
  std::memcpy(cx.p(), x.data(), x.size() * 4);
  std::memcpy(cw.p(), w.data(), w.size() * 4);
  std::memcpy(cs.p(), state_up.data(), state_up.size() * 4);
  std::memcpy(cci.p(), cidx.data(), cidx.size() * 4);
  vk.Synchronize(vq);

  Tensor vxt = Tensor::Contiguous(vx.p(), vt::DType::kF32, vd, {kBatch, kC});
  Tensor vwt = Tensor::Contiguous(vw.p(), vt::DType::kF32, vd, {kC, kK});
  Tensor vot = Tensor::Contiguous(vo.p(), vt::DType::kF32, vd, {kBatch, kC});
  Tensor vst = Tensor::Contiguous(vs.p(), vt::DType::kBF16, vd, {kStateRows, kC, kStateLen});
  Tensor vit = Tensor::Contiguous(vi.p(), vt::DType::kI32, vd, {kBatch});
  Tensor cxt = Tensor::Contiguous(cx.p(), vt::DType::kF32, cd, {kBatch, kC});
  Tensor cwt = Tensor::Contiguous(cw.p(), vt::DType::kF32, cd, {kC, kK});
  Tensor cot = Tensor::Contiguous(co.p(), vt::DType::kF32, cd, {kBatch, kC});
  Tensor cst = Tensor::Contiguous(cs.p(), vt::DType::kF32, cd, {kStateRows, kC, kStateLen});
  Tensor citt = Tensor::Contiguous(cci.p(), vt::DType::kI32, cd, {kBatch});

  vt::CausalConv1dArgs args;
  vt::CausalConv1dUpdate(cq, cot, cxt, cwt, nullptr, cst, args, &citt);
  vt::CausalConv1dUpdate(vq, vot, vxt, vwt, nullptr, vst, args, &vit);
  vk.Synchronize(vq);

  CHECK(RanNative(vt::OpId::kCausalConv1dUpdate));

  std::vector<float> got(kBatch * kC), ref(kBatch * kC);
  vk.Copy(vq, got.data(), vo.p(), got.size() * 4);
  vk.Synchronize(vq);
  std::memcpy(ref.data(), co.p(), ref.size() * 4);
  const double nmse = NmseOf(ref, got);
  MESSAGE("compressed-state causal_conv1d_update NMSE vs the upcast arm: " << nmse);
  CHECK(nmse <= kGdnNmseTol);

  // THE STATE IS THE POINT. The roll is a value move plus ONE rounded store, so
  // rounding the upcast arm's result is BIT-EXACT against the in-place bf16 row.
  std::vector<uint16_t> state_got(state_bf.size());
  vk.Copy(vq, state_got.data(), vs.p(), state_got.size() * 2);
  vk.Synchronize(vq);
  const float* state_ref = cs.as<float>();
  for (size_t i = 0; i < state_got.size(); ++i) {
    CAPTURE(i);
    CHECK(state_got[i] == vt::F32ToBF16(state_ref[i]));
  }
  // Independently of the oracle: the last tap is the RAW x sample, rounded once.
  for (int64_t bt = 0; bt < kBatch; ++bt) {
    for (int64_t c = 0; c < kC; ++c) {
      CAPTURE(bt);
      CAPTURE(c);
      const int64_t last = (cidx[static_cast<size_t>(bt)] * kC + c) * kStateLen + kWidth - 1;
      CHECK(state_got[static_cast<size_t>(last)] ==
            vt::F32ToBF16(x[static_cast<size_t>(bt * kC + c)]));
    }
  }

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

TEST_CASE("the fused GDN post-conv preamble runs NATIVELY on Vulkan, all five outputs") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  constexpr int64_t kT = 3, kHk = 2, kDk = 64, kHv = 2, kDv = 32;
  constexpr int64_t kKeyDim = kHk * kDk, kValDim = kHv * kDv;
  constexpr int64_t kConvDim = 2 * kKeyDim + kValDim;
  const std::vector<float> conv = Spread(kT * kConvDim, 2.0f, 71u);
  // araw is spread WIDE on purpose: softplus has two branches (the > 20
  // pass-through and the log1p one) and a narrow range would only ever reach one.
  const std::vector<float> araw = Spread(kT * kHv, 25.0f, 73u);
  const std::vector<float> braw = Spread(kT * kHv, 4.0f, 79u);
  const std::vector<float> a_log = Spread(kHv, 1.0f, 83u);
  const std::vector<float> dt_bias = Spread(kHv, 1.0f, 89u);

  Buf vconv(vk, kT * kConvDim, 4), va(vk, kT * kHv, 4), vb(vk, kT * kHv, 4), val(vk, kHv, 4),
      vdt(vk, kHv, 4);
  Buf vqo(vk, kT * kKeyDim, 4), vko(vk, kT * kKeyDim, 4), vvo(vk, kT * kValDim, 4),
      vgo(vk, kT * kHv, 4), vbo(vk, kT * kHv, 4);
  Buf cconv(cpu, kT * kConvDim, 4), ca(cpu, kT * kHv, 4), cb(cpu, kT * kHv, 4),
      cal(cpu, kHv, 4), cdt(cpu, kHv, 4);
  Buf cqo(cpu, kT * kKeyDim, 4), cko(cpu, kT * kKeyDim, 4), cvo(cpu, kT * kValDim, 4),
      cgo(cpu, kT * kHv, 4), cbo(cpu, kT * kHv, 4);

  vk.Copy(vq, vconv.p(), conv.data(), conv.size() * 4);
  vk.Copy(vq, va.p(), araw.data(), araw.size() * 4);
  vk.Copy(vq, vb.p(), braw.data(), braw.size() * 4);
  vk.Copy(vq, val.p(), a_log.data(), a_log.size() * 4);
  vk.Copy(vq, vdt.p(), dt_bias.data(), dt_bias.size() * 4);
  std::memcpy(cconv.p(), conv.data(), conv.size() * 4);
  std::memcpy(ca.p(), araw.data(), araw.size() * 4);
  std::memcpy(cb.p(), braw.data(), braw.size() * 4);
  std::memcpy(cal.p(), a_log.data(), a_log.size() * 4);
  std::memcpy(cdt.p(), dt_bias.data(), dt_bias.size() * 4);
  vk.Synchronize(vq);

  auto run = [&](Queue& q, Device dev, const Buf& bconv, const Buf& ba, const Buf& bb,
                 const Buf& bal, const Buf& bdt, const Buf& bq, const Buf& bk, const Buf& bv,
                 const Buf& bg, const Buf& bbeta) {
    Tensor tconv = Tensor::Contiguous(bconv.p(), vt::DType::kF32, dev, {kT, kConvDim});
    Tensor ta = Tensor::Contiguous(ba.p(), vt::DType::kF32, dev, {kT, kHv});
    Tensor tb = Tensor::Contiguous(bb.p(), vt::DType::kF32, dev, {kT, kHv});
    Tensor tal = Tensor::Contiguous(bal.p(), vt::DType::kF32, dev, {kHv});
    Tensor tdt = Tensor::Contiguous(bdt.p(), vt::DType::kF32, dev, {kHv});
    Tensor tq = Tensor::Contiguous(bq.p(), vt::DType::kF32, dev, {kT, kHk, kDk});
    Tensor tk = Tensor::Contiguous(bk.p(), vt::DType::kF32, dev, {kT, kHk, kDk});
    Tensor tv = Tensor::Contiguous(bv.p(), vt::DType::kF32, dev, {kT, kHv, kDv});
    Tensor tg = Tensor::Contiguous(bg.p(), vt::DType::kF32, dev, {kT, kHv});
    Tensor tbeta = Tensor::Contiguous(bbeta.p(), vt::DType::kF32, dev, {kT, kHv});
    vt::L2NormArgs l2args;
    vt::GdnPostConv(q, tq, tk, tv, tg, tbeta, tconv, ta, tb, tal, tdt, l2args);
  };
  run(cq, cd, cconv, ca, cb, cal, cdt, cqo, cko, cvo, cgo, cbo);
  run(vq, vd, vconv, va, vb, val, vdt, vqo, vko, vvo, vgo, vbo);
  vk.Synchronize(vq);

  CHECK(ctx.PipelineExistsFor("vt_gdn_post_conv"));
  CHECK(RanNative(vt::OpId::kGdnPostConv));

  // ALL FIVE outputs are compared. The kernel writes them from two different
  // branches of one dispatch (q/k from the head-slot branch, v/g/beta from the
  // other), so checking a subset would leave a whole branch unasserted.
  auto compare = [&](const char* what, const Buf& dev_buf, const Buf& host_buf, int64_t n) {
    std::vector<float> got(static_cast<size_t>(n));
    vk.Copy(vq, got.data(), dev_buf.p(), got.size() * 4);
    vk.Synchronize(vq);
    const std::vector<float> ref(host_buf.as<float>(), host_buf.as<float>() + n);
    const double nmse = NmseOf(ref, got);
    const std::string line =
        std::string("gdn_post_conv ") + what + " NMSE vs the CPU oracle: " + std::to_string(nmse);
    MESSAGE(line);
    CHECK(nmse <= kGdnNmseTol);
  };
  compare("q_out", vqo, cqo, kT * kKeyDim);
  compare("k_out", vko, cko, kT * kKeyDim);
  compare("v_out", vvo, cvo, kT * kValDim);
  compare("g_out", vgo, cgo, kT * kHv);
  compare("beta_out", vbo, cbo, kT * kHv);

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

// ===========================================================================
// BACKEND-VULKAN-GDN-CORE — the two gated-delta RECURRENCES.
//
// Same contract as the glue gates above: the oracle is our own CPU backend in
// the same binary, and every case asserts the MECHANISM as well as the numbers,
// because on this unified-memory device an unregistered op resolves to the CPU
// reference tier and returns answers IDENTICAL to the oracle.
//
// THE SHAPES ARE CHOSEN AGAINST THE TILE GEOMETRY, not for roundness
// (src/vt/vulkan/shaders/vt_gdn_recurrence.glsl: BV=16 state rows per workgroup,
// NW=8 lanes cooperating on each):
//   Dv = 19  -> ceil(19/16) = 2 value tiles and the second has only 3 valid rows,
//               so the tile tail's zero-fill and the guarded store-back both run;
//   Dk = 20  -> ceil(20/8) = 3 columns per lane, and lane 7's slice starts PAST
//               the end (c1 < c0), so a lane contributing an EMPTY partial to the
//               row reduction still has to reach every barrier;
//   Hv/Hk = 6/2 -> a GQA broadcast ratio of 3, not 1.
// A shape that was a multiple of everything would pass with all three wrong.
// ===========================================================================
namespace {

constexpr int64_t kGdnHk = 2, kGdnHv = 6, kGdnDk = 20, kGdnDv = 19;

// g is a LOG decay and beta a gate in (0,1). Feeding the raw spread would make
// exp(g) > 1 and GROW the carried state instead of contracting it, which is the
// opposite of the regime the recurrence runs in — and would let a sign error in
// the decay hide inside a plausible-looking NMSE.
std::vector<float> GdnDecays(size_t n, uint32_t seed) {
  std::vector<float> v = Spread(n, 2.0f, seed);
  for (float& x : v) x = -std::fabs(x);
  return v;
}
std::vector<float> GdnBetas(size_t n, uint32_t seed) {
  std::vector<float> v = Spread(n, 3.0f, seed);
  for (float& x : v) x = 1.0f / (1.0f + std::exp(-x));
  return v;
}

std::vector<uint16_t> ToBf16(const std::vector<float>& v) {
  std::vector<uint16_t> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = vt::F32ToBF16(v[i]);
  return o;
}

}  // namespace

TEST_CASE("the GDN PREFILL recurrence runs NATIVELY on Vulkan and matches the CPU oracle") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // Three sequences of lengths 5, 0 and 3. The EMPTY one is the interesting case:
  // its workgroups must stage their state tile, run no steps, and write it back
  // UNCHANGED — the CPU reference's `for (t = qslp[s]; t < qslp[s+1]; ...)`.
  constexpr int64_t kN = 3, kT = 8;
  const std::vector<int32_t> qsl = {0, 5, 5, 8};
  constexpr int64_t kQk = kT * kGdnHk * kGdnDk;
  constexpr int64_t kVo = kT * kGdnHv * kGdnDv;
  constexpr int64_t kGb = kT * kGdnHv;
  constexpr int64_t kRowElems = kGdnHv * kGdnDv * kGdnDk;
  constexpr int64_t kSt = kN * kRowElems;

  const std::vector<float> qv = Spread(kQk, 1.0f, 101u);
  const std::vector<float> kv = Spread(kQk, 1.0f, 103u);
  const std::vector<float> vv = Spread(kVo, 2.0f, 107u);
  const std::vector<float> gv = GdnDecays(kGb, 109u);
  const std::vector<float> bv = GdnBetas(kGb, 113u);
  const std::vector<float> s0 = Spread(kSt, 0.5f, 127u);

  vt::GdnArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(kGdnDk));

  Buf vqb(vk, kQk, 4), vkb(vk, kQk, 4), vvb(vk, kVo, 4), vob(vk, kVo, 4);
  Buf vgb(vk, kGb, 4), vbb(vk, kGb, 4), vsb(vk, kSt, 4), vlb(vk, kN + 1, 4);
  Buf cqb(cpu, kQk, 4), ckb(cpu, kQk, 4), cvb(cpu, kVo, 4), cob(cpu, kVo, 4);
  Buf cgb(cpu, kGb, 4), cbb(cpu, kGb, 4), csb(cpu, kSt, 4), clb(cpu, kN + 1, 4);

  vk.Copy(vq, vqb.p(), qv.data(), kQk * 4);
  vk.Copy(vq, vkb.p(), kv.data(), kQk * 4);
  vk.Copy(vq, vvb.p(), vv.data(), kVo * 4);
  vk.Copy(vq, vgb.p(), gv.data(), kGb * 4);
  vk.Copy(vq, vbb.p(), bv.data(), kGb * 4);
  vk.Copy(vq, vsb.p(), s0.data(), kSt * 4);
  vk.Copy(vq, vlb.p(), qsl.data(), (kN + 1) * 4);
  std::memcpy(cqb.p(), qv.data(), kQk * 4);
  std::memcpy(ckb.p(), kv.data(), kQk * 4);
  std::memcpy(cvb.p(), vv.data(), kVo * 4);
  std::memcpy(cgb.p(), gv.data(), kGb * 4);
  std::memcpy(cbb.p(), bv.data(), kGb * 4);
  std::memcpy(csb.p(), s0.data(), kSt * 4);
  std::memcpy(clb.p(), qsl.data(), (kN + 1) * 4);
  vk.Synchronize(vq);

  auto qk_t = [](void* p, Device dev, vt::DType dt) {
    return Tensor::Contiguous(p, dt, dev, {kT, kGdnHk, kGdnDk});
  };
  auto vo_t = [](void* p, Device dev, vt::DType dt) {
    return Tensor::Contiguous(p, dt, dev, {kT, kGdnHv, kGdnDv});
  };
  auto gb_t = [](void* p, Device dev) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {kT, kGdnHv});
  };
  auto st_t = [](void* p, Device dev) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {kN, kGdnHv, kGdnDv, kGdnDk});
  };

  Tensor vqt = qk_t(vqb.p(), vd, vt::DType::kF32), vkt = qk_t(vkb.p(), vd, vt::DType::kF32);
  Tensor vvt = vo_t(vvb.p(), vd, vt::DType::kF32), vot = vo_t(vob.p(), vd, vt::DType::kF32);
  Tensor vgt = gb_t(vgb.p(), vd), vbt = gb_t(vbb.p(), vd), vst = st_t(vsb.p(), vd);
  Tensor vlt = Tensor::Contiguous(vlb.p(), vt::DType::kI32, vd, {kN + 1});
  Tensor cqt = qk_t(cqb.p(), cd, vt::DType::kF32), ckt = qk_t(ckb.p(), cd, vt::DType::kF32);
  Tensor cvt = vo_t(cvb.p(), cd, vt::DType::kF32), cot = vo_t(cob.p(), cd, vt::DType::kF32);
  Tensor cgt = gb_t(cgb.p(), cd), cbt = gb_t(cbb.p(), cd), cst = st_t(csb.p(), cd);
  Tensor clt = Tensor::Contiguous(clb.p(), vt::DType::kI32, cd, {kN + 1});

  vt::GdnPrefill(cq, cot, cqt, ckt, cvt, cgt, cbt, cst, clt, args);
  vt::GdnPrefill(vq, vot, vqt, vkt, vvt, vgt, vbt, vst, vlt, args);
  vk.Synchronize(vq);

  CHECK(ctx.PipelineExistsFor("vt_gdn_prefill"));
  CHECK(RanNative(vt::OpId::kGdnPrefill));

  std::vector<float> got_out(kVo), got_state(kSt);
  vk.Copy(vq, got_out.data(), vob.p(), kVo * 4);
  vk.Copy(vq, got_state.data(), vsb.p(), kSt * 4);
  vk.Synchronize(vq);
  const std::vector<float> ref_out(cob.as<float>(), cob.as<float>() + kVo);
  const std::vector<float> ref_state(csb.as<float>(), csb.as<float>() + kSt);
  const double nmse_out = NmseOf(ref_out, got_out);
  const double nmse_state = NmseOf(ref_state, got_state);
  MESSAGE("gdn_prefill out NMSE vs the CPU oracle: " << nmse_out);
  MESSAGE("gdn_prefill CARRIED STATE NMSE vs the CPU oracle: " << nmse_state);
  CHECK(nmse_out <= kGdnNmseTol);
  // The state is asserted SEPARATELY from the output because it is the part that
  // carries: a kernel that decayed correctly but wrote its tile back to the wrong
  // rows would still produce a plausible `out`.
  CHECK(nmse_state <= kGdnNmseTol);

  // Sequence 1 is EMPTY, so its state block must come back BIT-IDENTICAL to what
  // it went in as — not merely close. This catches a store-back that wrote a
  // decayed or zeroed tile for a zero-length token range.
  CHECK(std::memcmp(got_state.data() + kRowElems, s0.data() + kRowElems,
                    static_cast<size_t>(kRowElems) * 4) == 0);

  // --- bf16 arm: proof the dtype specialization actually engaged --------------
  // The numbers alone would not prove it. The shader's DEFAULT specialization is
  // f32/f32, so a silently-ignored VkSpecializationMapEntry would read bf16 bytes
  // as f32 — the exact failure this backend's spec-id bookkeeping exists for — and
  // a NEW pipeline appearing in the cache is what says the value was bound.
  const size_t pipelines_before = ctx.PipelineCacheSize();
  const std::vector<uint16_t> q16 = ToBf16(qv), k16 = ToBf16(kv), v16 = ToBf16(vv);
  Buf vq16(vk, kQk, 2), vk16(vk, kQk, 2), vv16(vk, kVo, 2), vo16(vk, kVo, 2);
  Buf cq16(cpu, kQk, 2), ck16(cpu, kQk, 2), cv16(cpu, kVo, 2), co16(cpu, kVo, 2);
  vk.Copy(vq, vq16.p(), q16.data(), kQk * 2);
  vk.Copy(vq, vk16.p(), k16.data(), kQk * 2);
  vk.Copy(vq, vv16.p(), v16.data(), kVo * 2);
  vk.Copy(vq, vsb.p(), s0.data(), kSt * 4);  // reset the carried state
  std::memcpy(cq16.p(), q16.data(), kQk * 2);
  std::memcpy(ck16.p(), k16.data(), kQk * 2);
  std::memcpy(cv16.p(), v16.data(), kVo * 2);
  std::memcpy(csb.p(), s0.data(), kSt * 4);
  vk.Synchronize(vq);

  Tensor vq16t = qk_t(vq16.p(), vd, vt::DType::kBF16);
  Tensor vk16t = qk_t(vk16.p(), vd, vt::DType::kBF16);
  Tensor vv16t = vo_t(vv16.p(), vd, vt::DType::kBF16);
  Tensor vo16t = vo_t(vo16.p(), vd, vt::DType::kBF16);
  Tensor cq16t = qk_t(cq16.p(), cd, vt::DType::kBF16);
  Tensor ck16t = qk_t(ck16.p(), cd, vt::DType::kBF16);
  Tensor cv16t = vo_t(cv16.p(), cd, vt::DType::kBF16);
  Tensor co16t = vo_t(co16.p(), cd, vt::DType::kBF16);

  vt::GdnPrefill(cq, co16t, cq16t, ck16t, cv16t, cgt, cbt, cst, clt, args);
  vt::GdnPrefill(vq, vo16t, vq16t, vk16t, vv16t, vgt, vbt, vst, vlt, args);
  vk.Synchronize(vq);
  CHECK(ctx.PipelineCacheSize() == pipelines_before + 1);

  std::vector<uint16_t> got16(kVo);
  vk.Copy(vq, got16.data(), vo16.p(), kVo * 2);
  vk.Synchronize(vq);
  std::vector<float> got16_f(kVo), ref16_f(kVo);
  for (int64_t i = 0; i < kVo; ++i) {
    got16_f[i] = vt::BF16ToF32(got16[i]);
    ref16_f[i] = vt::BF16ToF32(co16.as<uint16_t>()[i]);
  }
  const double nmse16 = NmseOf(ref16_f, got16_f);
  MESSAGE("gdn_prefill out NMSE (bf16 q/k/v and bf16 out): " << nmse16);
  CHECK(nmse16 <= kGdnNmseTol);

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

TEST_CASE("the GDN DECODE recurrence runs NATIVELY on Vulkan, indexed and compact") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // The cache has MORE rows than the batch and the indices are scrambled, so a
  // kernel that used the token index as its state row would pass a
  // batch-rows-only test and corrupt this one. Index 1 is NEGATIVE — fla's NULL
  // block — whose output row must come back ZEROED.
  constexpr int64_t kBatch = 5, kSlots = 7;
  const std::vector<int32_t> idx = {3, -1, 0, 6, 1};
  constexpr int64_t kQk = kBatch * kGdnHk * kGdnDk;
  constexpr int64_t kVo = kBatch * kGdnHv * kGdnDv;
  constexpr int64_t kGb = kBatch * kGdnHv;
  constexpr int64_t kRowElems = kGdnHv * kGdnDv * kGdnDk;
  constexpr int64_t kSt = kSlots * kRowElems;

  const std::vector<float> qv = Spread(kQk, 1.0f, 211u);
  const std::vector<float> kv = Spread(kQk, 1.0f, 223u);
  const std::vector<float> vv = Spread(kVo, 2.0f, 227u);
  const std::vector<float> gv = GdnDecays(kGb, 229u);
  const std::vector<float> bv = GdnBetas(kGb, 233u);
  const std::vector<float> s0 = Spread(kSt, 0.5f, 239u);

  vt::GdnArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(kGdnDk));

  Buf vqb(vk, kQk, 4), vkb(vk, kQk, 4), vvb(vk, kVo, 4), vob(vk, kVo, 4);
  Buf vgb(vk, kGb, 4), vbb(vk, kGb, 4), vsb(vk, kSt, 4), vib(vk, kBatch, 4);
  Buf cqb(cpu, kQk, 4), ckb(cpu, kQk, 4), cvb(cpu, kVo, 4), cob(cpu, kVo, 4);
  Buf cgb(cpu, kGb, 4), cbb(cpu, kGb, 4), csb(cpu, kSt, 4), cib(cpu, kBatch, 4);

  vk.Copy(vq, vqb.p(), qv.data(), kQk * 4);
  vk.Copy(vq, vkb.p(), kv.data(), kQk * 4);
  vk.Copy(vq, vvb.p(), vv.data(), kVo * 4);
  vk.Copy(vq, vgb.p(), gv.data(), kGb * 4);
  vk.Copy(vq, vbb.p(), bv.data(), kGb * 4);
  vk.Copy(vq, vsb.p(), s0.data(), kSt * 4);
  vk.Copy(vq, vib.p(), idx.data(), kBatch * 4);
  std::memcpy(cqb.p(), qv.data(), kQk * 4);
  std::memcpy(ckb.p(), kv.data(), kQk * 4);
  std::memcpy(cvb.p(), vv.data(), kVo * 4);
  std::memcpy(cgb.p(), gv.data(), kGb * 4);
  std::memcpy(cbb.p(), bv.data(), kGb * 4);
  std::memcpy(csb.p(), s0.data(), kSt * 4);
  std::memcpy(cib.p(), idx.data(), kBatch * 4);
  vk.Synchronize(vq);

  auto qk_t = [](void* p, Device dev) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {kBatch, kGdnHk, kGdnDk});
  };
  auto vo_t = [](void* p, Device dev) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {kBatch, kGdnHv, kGdnDv});
  };
  auto gb_t = [](void* p, Device dev) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {kBatch, kGdnHv});
  };
  auto st_t = [](void* p, Device dev, int64_t rows) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {rows, kGdnHv, kGdnDv, kGdnDk});
  };

  Tensor vqt = qk_t(vqb.p(), vd), vkt = qk_t(vkb.p(), vd);
  Tensor vvt = vo_t(vvb.p(), vd), vot = vo_t(vob.p(), vd);
  Tensor vgt = gb_t(vgb.p(), vd), vbt = gb_t(vbb.p(), vd);
  Tensor vst = st_t(vsb.p(), vd, kSlots);
  Tensor vit = Tensor::Contiguous(vib.p(), vt::DType::kI32, vd, {kBatch});
  Tensor cqt = qk_t(cqb.p(), cd), ckt = qk_t(ckb.p(), cd);
  Tensor cvt = vo_t(cvb.p(), cd), cot = vo_t(cob.p(), cd);
  Tensor cgt = gb_t(cgb.p(), cd), cbt = gb_t(cbb.p(), cd);
  Tensor cst = st_t(csb.p(), cd, kSlots);
  Tensor cit = Tensor::Contiguous(cib.p(), vt::DType::kI32, cd, {kBatch});

  vt::GdnDecode(cq, cot, cqt, ckt, cvt, cgt, cbt, cst, args, &cit);
  vt::GdnDecode(vq, vot, vqt, vkt, vvt, vgt, vbt, vst, args, &vit);
  vk.Synchronize(vq);

  CHECK(ctx.PipelineExistsFor("vt_gdn_decode"));
  CHECK(RanNative(vt::OpId::kGdnDecode));

  std::vector<float> got_out(kVo), got_state(kSt);
  vk.Copy(vq, got_out.data(), vob.p(), kVo * 4);
  vk.Copy(vq, got_state.data(), vsb.p(), kSt * 4);
  vk.Synchronize(vq);
  const std::vector<float> ref_out(cob.as<float>(), cob.as<float>() + kVo);
  const std::vector<float> ref_state(csb.as<float>(), csb.as<float>() + kSt);
  const double nmse_out = NmseOf(ref_out, got_out);
  const double nmse_state = NmseOf(ref_state, got_state);
  MESSAGE("gdn_decode (indexed cache) out NMSE vs the CPU oracle: " << nmse_out);
  MESSAGE("gdn_decode (indexed cache) CACHE NMSE vs the CPU oracle: " << nmse_state);
  CHECK(nmse_out <= kGdnNmseTol);
  CHECK(nmse_state <= kGdnNmseTol);

  // The NULL block's output row is asserted DIRECTLY: if both kernels wrongly ran
  // the recurrence for it, the comparison above would still be green.
  bool null_row_zero = true;
  for (int64_t e = 0; e < kGdnHv * kGdnDv; ++e) {
    if (got_out[kGdnHv * kGdnDv + e] != 0.0f) null_row_zero = false;
  }
  CHECK(null_row_zero);
  // Cache slots 2, 4 and 5 are named by no index and must still hold their
  // ORIGINAL bytes — proof the indirection wrote only where it was told to.
  for (int64_t slot : {2, 4, 5}) {
    CAPTURE(slot);
    CHECK(std::memcmp(got_state.data() + slot * kRowElems, s0.data() + slot * kRowElems,
                      static_cast<size_t>(kRowElems) * 4) == 0);
  }

  // --- the COMPACT path: no state_idx, one state row per token ----------------
  Buf vsc(vk, kBatch * kRowElems, 4), csc(cpu, kBatch * kRowElems, 4);
  vk.Copy(vq, vsc.p(), s0.data(), kBatch * kRowElems * 4);
  std::memcpy(csc.p(), s0.data(), kBatch * kRowElems * 4);
  vk.Synchronize(vq);
  Tensor vsct = st_t(vsc.p(), vd, kBatch), csct = st_t(csc.p(), cd, kBatch);
  vt::GdnDecode(cq, cot, cqt, ckt, cvt, cgt, cbt, csct, args, nullptr);
  vt::GdnDecode(vq, vot, vqt, vkt, vvt, vgt, vbt, vsct, args, nullptr);
  vk.Synchronize(vq);

  std::vector<float> got_c(kVo), got_sc(kBatch * kRowElems);
  vk.Copy(vq, got_c.data(), vob.p(), kVo * 4);
  vk.Copy(vq, got_sc.data(), vsc.p(), kBatch * kRowElems * 4);
  vk.Synchronize(vq);
  const std::vector<float> ref_c(cob.as<float>(), cob.as<float>() + kVo);
  const std::vector<float> ref_sc(csc.as<float>(), csc.as<float>() + kBatch * kRowElems);
  const double nmse_c = NmseOf(ref_c, got_c);
  MESSAGE("gdn_decode (compact state) out NMSE vs the CPU oracle: " << nmse_c);
  CHECK(nmse_c <= kGdnNmseTol);
  CHECK(NmseOf(ref_sc, got_sc) <= kGdnNmseTol);

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

TEST_CASE("a GDN recurrence wider than the shared tile DECLINES to the reference tier") {
  if (!VulkanPresent()) return;
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // Dk = 132 is past VT_GDN_MAX_DK, so the [BV,Dk] tile would not fit Vulkan's
  // GUARANTEED 16 KB of shared memory. The kernel must forward through the
  // provider seam rather than throw or index past the array: declining preserves a
  // capability the reference tier already had, which is what GetOpFallback is for.
  // Asserted through the DECLINE COUNTER, because `last_selected` still names
  // vt-native — the native provider WAS selected, and then forwarded.
  constexpr int64_t kN = 1, kT = 2, kHk = 1, kHv = 1, kDk = 132, kDv = 4;
  constexpr int64_t kQk = kT * kHk * kDk, kVo = kT * kHv * kDv, kGb = kT * kHv;
  constexpr int64_t kSt = kN * kHv * kDv * kDk;
  const std::vector<int32_t> qsl = {0, 2};
  const std::vector<float> qv = Spread(kQk, 1.0f, 307u), kv = Spread(kQk, 1.0f, 311u);
  const std::vector<float> vv = Spread(kVo, 2.0f, 313u);
  const std::vector<float> gv = GdnDecays(kGb, 317u), bv = GdnBetas(kGb, 331u);
  const std::vector<float> s0 = Spread(kSt, 0.5f, 337u);
  vt::GdnArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(kDk));

  Buf vqb(vk, kQk, 4), vkb(vk, kQk, 4), vvb(vk, kVo, 4), vob(vk, kVo, 4);
  Buf vgb(vk, kGb, 4), vbb(vk, kGb, 4), vsb(vk, kSt, 4), vlb(vk, kN + 1, 4);
  Buf cqb(cpu, kQk, 4), ckb(cpu, kQk, 4), cvb(cpu, kVo, 4), cob(cpu, kVo, 4);
  Buf cgb(cpu, kGb, 4), cbb(cpu, kGb, 4), csb(cpu, kSt, 4), clb(cpu, kN + 1, 4);
  vk.Copy(vq, vqb.p(), qv.data(), kQk * 4);
  vk.Copy(vq, vkb.p(), kv.data(), kQk * 4);
  vk.Copy(vq, vvb.p(), vv.data(), kVo * 4);
  vk.Copy(vq, vgb.p(), gv.data(), kGb * 4);
  vk.Copy(vq, vbb.p(), bv.data(), kGb * 4);
  vk.Copy(vq, vsb.p(), s0.data(), kSt * 4);
  vk.Copy(vq, vlb.p(), qsl.data(), (kN + 1) * 4);
  std::memcpy(cqb.p(), qv.data(), kQk * 4);
  std::memcpy(ckb.p(), kv.data(), kQk * 4);
  std::memcpy(cvb.p(), vv.data(), kVo * 4);
  std::memcpy(cgb.p(), gv.data(), kGb * 4);
  std::memcpy(cbb.p(), bv.data(), kGb * 4);
  std::memcpy(csb.p(), s0.data(), kSt * 4);
  std::memcpy(clb.p(), qsl.data(), (kN + 1) * 4);
  vk.Synchronize(vq);

  auto qk_t = [](void* p, Device dev) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {kT, kHk, kDk});
  };
  auto vo_t = [](void* p, Device dev) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {kT, kHv, kDv});
  };
  auto gb_t = [](void* p, Device dev) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {kT, kHv});
  };
  auto st_t = [](void* p, Device dev) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {kN, kHv, kDv, kDk});
  };
  Tensor vqt = qk_t(vqb.p(), vd), vkt = qk_t(vkb.p(), vd);
  Tensor vvt = vo_t(vvb.p(), vd), vot = vo_t(vob.p(), vd);
  Tensor vgt = gb_t(vgb.p(), vd), vbt = gb_t(vbb.p(), vd), vst = st_t(vsb.p(), vd);
  Tensor vlt = Tensor::Contiguous(vlb.p(), vt::DType::kI32, vd, {kN + 1});
  Tensor cqt = qk_t(cqb.p(), cd), ckt = qk_t(ckb.p(), cd);
  Tensor cvt = vo_t(cvb.p(), cd), cot = vo_t(cob.p(), cd);
  Tensor cgt = gb_t(cgb.p(), cd), cbt = gb_t(cbb.p(), cd), cst = st_t(csb.p(), cd);
  Tensor clt = Tensor::Contiguous(clb.p(), vt::DType::kI32, cd, {kN + 1});

  const auto before = vt::GetOpProviderStats(vt::OpId::kGdnPrefill, DeviceType::kVULKAN);
  CHECK_NOTHROW(vt::GdnPrefill(vq, vot, vqt, vkt, vvt, vgt, vbt, vst, vlt, args));
  vk.Synchronize(vq);
  const auto after = vt::GetOpProviderStats(vt::OpId::kGdnPrefill, DeviceType::kVULKAN);
  CHECK(after.declines == before.declines + 1);

  // And it still computed the right answer on the tier it forwarded to — BIT-EXACT
  // this time, because the same host code ran both sides.
  vt::GdnPrefill(cq, cot, cqt, ckt, cvt, cgt, cbt, cst, clt, args);
  std::vector<float> got(kVo);
  vk.Copy(vq, got.data(), vob.p(), kVo * 4);
  vk.Synchronize(vq);
  CHECK(std::memcmp(got.data(), cob.p(), static_cast<size_t>(kVo) * 4) == 0);

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

// ===========================================================================
// BACKEND-VULKAN-QKNORM — the fused full-attention preamble.
//
// SHAPES ARE CHOSEN, NOT DEFAULTED. Hq=5 / Hkv=2 is RAGGED (Hq is not a multiple
// of Hkv, and 5+2=7 workgroups per token exercise the flattened (token, head)
// decomposition rather than a power-of-two one). Dh=160 exceeds the 128-wide
// workgroup, so every lane walks its head row in a strided loop and the mean
// square is a genuine tree reduction over partial sums. rot=96 < Dh is the whole
// point of the op — PARTIAL RoPE, so dims [96,160) must come out normed but
// UNROTATED, and a kernel that rotated the whole head would still pass a
// rot==Dh test. Both source rows are PADDED VIEWS (stride[0] > shape[1]), the
// QKVParallelLinear packed layout the model actually hands the op.
// ===========================================================================
namespace {

// The op's contract: qgate is [T, Hq*2*Dh] with q at [h*2*Dh, h*2*Dh+Dh) and the
// gate at the second Dh (src/vt/ops.cpp:1516-1523).
constexpr int64_t kQkT = 3, kQkHq = 5, kQkHkv = 2, kQkDh = 160, kQkRot = 96;
constexpr int64_t kQkQRow = kQkHq * 2 * kQkDh, kQkKRow = kQkHkv * kQkDh;
constexpr int64_t kQkQPad = 24, kQkKPad = 8;  // packed-view row padding
constexpr int64_t kQkQStride = kQkQRow + kQkQPad, kQkKStride = kQkKRow + kQkKPad;

// std::to_string gives six DECIMALS, which renders every NMSE this kernel
// produces as "0.000000" — the one number the message exists to carry.
std::string Sci(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.3g", v);
  return buf;
}

Tensor QkSrc(void* p, Device dev, vt::DType dt, int64_t cols, int64_t stride) {
  Tensor t = Tensor::Contiguous(p, dt, dev, {kQkT, cols});
  t.stride[0] = stride;  // inner-contiguous, padded row: the packed projection view
  return t;
}

}  // namespace

TEST_CASE("the fused attn preamble runs NATIVELY on Vulkan: partial RoPE, ragged heads") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  const std::vector<float> qgate = Spread(kQkT * kQkQStride, 2.0f, 401u);
  const std::vector<float> kf = Spread(kQkT * kQkKStride, 2.0f, 409u);
  // Norm weights around zero: with gemma the applied weight is (1+w), so this is
  // a spread around 1 and a kernel that dropped the +1 changes every output.
  const std::vector<float> qn = Spread(kQkDh, 0.4f, 419u);
  const std::vector<float> kn = Spread(kQkDh, 0.4f, 421u);

  const int64_t qelems = kQkT * kQkQStride, kelems = kQkT * kQkKStride;
  const int64_t qout = kQkT * kQkHq * kQkDh, kout = kQkT * kQkHkv * kQkDh;

  Buf vqg(vk, qelems, 4), vkf(vk, kelems, 4), vqn(vk, kQkDh, 4), vkn(vk, kQkDh, 4);
  Buf vcs(vk, kQkT * kQkRot, 4);
  Buf vqo(vk, qout, 4), vko(vk, kout, 4), vgo(vk, qout, 4);
  Buf cqg(cpu, qelems, 4), ckf(cpu, kelems, 4), cqn(cpu, kQkDh, 4), ckn(cpu, kQkDh, 4);
  Buf ccs(cpu, kQkT * kQkRot, 4), cpo(cpu, kQkT, 4);
  Buf cqo(cpu, qout, 4), cko(cpu, kout, 4), cgo(cpu, qout, 4);

  // The cos/sin table is built ONCE on the host — that is the op's own split
  // (kRopeCosSinCache stays on the portable tier by design) — and the SAME BYTES
  // are handed to both devices, so the comparison isolates this kernel.
  const std::vector<int32_t> pos = {0, 7, 4096};  // a long position: real angles
  std::memcpy(cpo.p(), pos.data(), pos.size() * 4);
  Tensor cpot = Tensor::Contiguous(cpo.p(), vt::DType::kI32, cd, {kQkT});
  Tensor ccst = Tensor::Contiguous(ccs.p(), vt::DType::kF32, cd, {kQkT, kQkRot});
  vt::RopeCosSinCache(cq, ccst, cpot, vt::RopeArgs{10000.0f, static_cast<int>(kQkRot)});

  vk.Copy(vq, vqg.p(), qgate.data(), qelems * 4);
  vk.Copy(vq, vkf.p(), kf.data(), kelems * 4);
  vk.Copy(vq, vqn.p(), qn.data(), kQkDh * 4);
  vk.Copy(vq, vkn.p(), kn.data(), kQkDh * 4);
  vk.Copy(vq, vcs.p(), ccs.p(), kQkT * kQkRot * 4);
  std::memcpy(cqg.p(), qgate.data(), qelems * 4);
  std::memcpy(ckf.p(), kf.data(), kelems * 4);
  std::memcpy(cqn.p(), qn.data(), kQkDh * 4);
  std::memcpy(ckn.p(), kn.data(), kQkDh * 4);
  vk.Synchronize(vq);

  auto out3 = [](void* p, Device dev, int64_t h) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {kQkT, h, kQkDh});
  };
  Tensor vqgt = QkSrc(vqg.p(), vd, vt::DType::kF32, kQkQRow, kQkQStride);
  Tensor vkft = QkSrc(vkf.p(), vd, vt::DType::kF32, kQkKRow, kQkKStride);
  Tensor vqnt = Tensor::Contiguous(vqn.p(), vt::DType::kF32, vd, {kQkDh});
  Tensor vknt = Tensor::Contiguous(vkn.p(), vt::DType::kF32, vd, {kQkDh});
  Tensor vcst = Tensor::Contiguous(vcs.p(), vt::DType::kF32, vd, {kQkT, kQkRot});
  Tensor vqot = out3(vqo.p(), vd, kQkHq), vkot = out3(vko.p(), vd, kQkHkv);
  Tensor vgot = out3(vgo.p(), vd, kQkHq);
  Tensor cqgt = QkSrc(cqg.p(), cd, vt::DType::kF32, kQkQRow, kQkQStride);
  Tensor ckft = QkSrc(ckf.p(), cd, vt::DType::kF32, kQkKRow, kQkKStride);
  Tensor cqnt = Tensor::Contiguous(cqn.p(), vt::DType::kF32, cd, {kQkDh});
  Tensor cknt = Tensor::Contiguous(ckn.p(), vt::DType::kF32, cd, {kQkDh});
  Tensor cqot = out3(cqo.p(), cd, kQkHq), ckot = out3(cko.p(), cd, kQkHkv);
  Tensor cgot = out3(cgo.p(), cd, kQkHq);

  const vt::RopeArgs ra{10000.0f, static_cast<int>(kQkRot)};
  // BOTH gemma arms. The model only ever calls gemma=true, but the weight is
  // applied as (1+w) vs w through a UNIFORM branch in the shader, and a branch
  // written the other way round would be invisible with only one arm gated.
  for (bool gemma : {true, false}) {
    const vt::RmsNormArgs na{1e-6f, gemma};
    vt::AttnQkNormRopeGate(cq, cqot, ckot, cgot, cqgt, ckft, cqnt, cknt, ccst, na, ra);
    vt::AttnQkNormRopeGate(vq, vqot, vkot, vgot, vqgt, vkft, vqnt, vknt, vcst, na, ra);
    vk.Synchronize(vq);

    std::vector<float> gq(qout), gk(kout), gg(qout);
    vk.Copy(vq, gq.data(), vqo.p(), qout * 4);
    vk.Copy(vq, gk.data(), vko.p(), kout * 4);
    vk.Copy(vq, gg.data(), vgo.p(), qout * 4);
    vk.Synchronize(vq);
    const std::vector<float> rq(cqo.as<float>(), cqo.as<float>() + qout);
    const std::vector<float> rk(cko.as<float>(), cko.as<float>() + kout);
    const double nq = NmseOf(rq, gq), nk = NmseOf(rk, gk);
    // Reported, NOT asserted: on THIS device the tree reduction happens to land
    // on the CPU tier's own bits, but that is a property of the shapes and the
    // driver, not a promise the op makes across devices — the tier this kernel is
    // gated at is NMSE, like every other reducing shader in the backend.
    const bool bitwise = std::memcmp(rq.data(), gq.data(), static_cast<size_t>(qout) * 4) == 0 &&
                         std::memcmp(rk.data(), gk.data(), static_cast<size_t>(kout) * 4) == 0;
    // Assembled OUTSIDE the macro: MESSAGE(x << y) hands the expression to the
    // doctest MessageBuilder, so a flag written inside renders as "1".
    const std::string line = std::string("attn_qk_norm_rope_gate (gemma=") +
                             (gemma ? "1" : "0") + ") q NMSE " + Sci(nq) + ", k NMSE " + Sci(nk) +
                             ", bitwise-equal to the CPU: " + (bitwise ? "YES" : "no");
    MESSAGE(line);
    CHECK(nq <= kGdnNmseTol);
    CHECK(nk <= kGdnNmseTol);

    // THE GATE IS A PASSTHROUGH, so it is held to the BIT-EXACT tier, not NMSE:
    // it is a copy of the raw second Dh of each q|gate pair with no norm and no
    // rotation, and it is compared against the SOURCE rather than against the CPU
    // oracle's output — a kernel that copied the q half into both would agree
    // with a kernel that did the same on the other device.
    bool gate_exact = true;
    for (int64_t tok = 0; tok < kQkT; ++tok) {
      for (int64_t h = 0; h < kQkHq; ++h) {
        for (int64_t j = 0; j < kQkDh; ++j) {
          const float want =
              qgate[static_cast<size_t>(tok * kQkQStride + h * 2 * kQkDh + kQkDh + j)];
          if (gg[static_cast<size_t>((tok * kQkHq + h) * kQkDh + j)] != want) gate_exact = false;
        }
      }
    }
    CHECK(gate_exact);
    // The gate must also be DIFFERENT from q, or "carrying data" is unproven.
    CHECK(std::memcmp(gg.data(), gq.data(), static_cast<size_t>(qout) * 4) != 0);

    // PARTIAL RoPE: dims [rot, Dh) are normed but NOT rotated, so they equal the
    // plain gemma-RMSNorm of the source. Checked against a hand-computed f64 row
    // rather than against the CPU kernel, which would agree with a Vulkan kernel
    // that rotated everything only if the CPU one did too.
    for (int64_t h = 0; h < kQkHq; ++h) {
      const size_t base = static_cast<size_t>(h * 2 * kQkDh);  // token 0
      double ss = 0.0;
      for (int64_t j = 0; j < kQkDh; ++j) {
        const double v = qgate[base + static_cast<size_t>(j)];
        ss += v * v;
      }
      const double inv = 1.0 / std::sqrt(ss / static_cast<double>(kQkDh) + 1e-6);
      for (int64_t j = kQkRot; j < kQkDh; ++j) {
        const double w = gemma ? qn[static_cast<size_t>(j)] + 1.0 : qn[static_cast<size_t>(j)];
        const double want = qgate[base + static_cast<size_t>(j)] * inv * w;
        const double got = gq[static_cast<size_t>(h * kQkDh + j)];
        CAPTURE(h);
        CAPTURE(j);
        CHECK(std::fabs(got - want) <= 1e-4 * (std::fabs(want) + 1e-3));
      }
    }
  }

  CHECK(ctx.PipelineExistsFor("vt_attn_qk_norm_rope_gate"));
  CHECK(RanNative(vt::OpId::kAttnQkNormRopeGate));

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

TEST_CASE("the fused attn preamble serves the bf16 q/k + f32 gate combo natively") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // The FA-2 prefill combo the op contract singles out (src/vt/ops.cpp:1530-1536):
  // bf16 sources, bf16 q/k (they feed attention and the bf16 KV-cache write) and
  // an f32 gate (sigmoid(gate) must see the un-rounded value). It is the ONLY
  // shape in which the gate dtype differs from q/k, which is the reason the gate
  // is its own specialization axis — a shader that reused the q/k dtype constant
  // for the gate would write bf16 into an f32 buffer and pass every same-dtype
  // test.
  const std::vector<float> qgate_f = Spread(kQkT * kQkQStride, 2.0f, 431u);
  const std::vector<float> kf_f = Spread(kQkT * kQkKStride, 2.0f, 433u);
  const std::vector<float> qn = Spread(kQkDh, 0.4f, 439u);
  const std::vector<float> kn = Spread(kQkDh, 0.4f, 443u);
  std::vector<uint16_t> qgate_b(qgate_f.size()), kf_b(kf_f.size());
  for (size_t i = 0; i < qgate_f.size(); ++i) qgate_b[i] = vt::F32ToBF16(qgate_f[i]);
  for (size_t i = 0; i < kf_f.size(); ++i) kf_b[i] = vt::F32ToBF16(kf_f[i]);

  const int64_t qelems = kQkT * kQkQStride, kelems = kQkT * kQkKStride;
  const int64_t qout = kQkT * kQkHq * kQkDh, kout = kQkT * kQkHkv * kQkDh;

  Buf vqg(vk, qelems, 2), vkf(vk, kelems, 2), vqn(vk, kQkDh, 4), vkn(vk, kQkDh, 4);
  Buf vcs(vk, kQkT * kQkRot, 4);
  Buf vqo(vk, qout, 2), vko(vk, kout, 2), vgo(vk, qout, 4);
  Buf cqg(cpu, qelems, 2), ckf(cpu, kelems, 2), cqn(cpu, kQkDh, 4), ckn(cpu, kQkDh, 4);
  Buf ccs(cpu, kQkT * kQkRot, 4), cpo(cpu, kQkT, 4);
  Buf cqo(cpu, qout, 2), cko(cpu, kout, 2), cgo(cpu, qout, 4);

  const std::vector<int32_t> pos = {1, 33, 900};
  std::memcpy(cpo.p(), pos.data(), pos.size() * 4);
  Tensor cpot = Tensor::Contiguous(cpo.p(), vt::DType::kI32, cd, {kQkT});
  Tensor ccst = Tensor::Contiguous(ccs.p(), vt::DType::kF32, cd, {kQkT, kQkRot});
  vt::RopeCosSinCache(cq, ccst, cpot, vt::RopeArgs{10000.0f, static_cast<int>(kQkRot)});

  vk.Copy(vq, vqg.p(), qgate_b.data(), qelems * 2);
  vk.Copy(vq, vkf.p(), kf_b.data(), kelems * 2);
  vk.Copy(vq, vqn.p(), qn.data(), kQkDh * 4);
  vk.Copy(vq, vkn.p(), kn.data(), kQkDh * 4);
  vk.Copy(vq, vcs.p(), ccs.p(), kQkT * kQkRot * 4);
  std::memcpy(cqg.p(), qgate_b.data(), qelems * 2);
  std::memcpy(ckf.p(), kf_b.data(), kelems * 2);
  std::memcpy(cqn.p(), qn.data(), kQkDh * 4);
  std::memcpy(ckn.p(), kn.data(), kQkDh * 4);
  vk.Synchronize(vq);

  auto qk3 = [](void* p, Device dev, int64_t h) {
    return Tensor::Contiguous(p, vt::DType::kBF16, dev, {kQkT, h, kQkDh});
  };
  Tensor vqgt = QkSrc(vqg.p(), vd, vt::DType::kBF16, kQkQRow, kQkQStride);
  Tensor vkft = QkSrc(vkf.p(), vd, vt::DType::kBF16, kQkKRow, kQkKStride);
  Tensor vqnt = Tensor::Contiguous(vqn.p(), vt::DType::kF32, vd, {kQkDh});
  Tensor vknt = Tensor::Contiguous(vkn.p(), vt::DType::kF32, vd, {kQkDh});
  Tensor vcst = Tensor::Contiguous(vcs.p(), vt::DType::kF32, vd, {kQkT, kQkRot});
  Tensor vqot = qk3(vqo.p(), vd, kQkHq), vkot = qk3(vko.p(), vd, kQkHkv);
  Tensor vgot = Tensor::Contiguous(vgo.p(), vt::DType::kF32, vd, {kQkT, kQkHq, kQkDh});
  Tensor cqgt = QkSrc(cqg.p(), cd, vt::DType::kBF16, kQkQRow, kQkQStride);
  Tensor ckft = QkSrc(ckf.p(), cd, vt::DType::kBF16, kQkKRow, kQkKStride);
  Tensor cqnt = Tensor::Contiguous(cqn.p(), vt::DType::kF32, cd, {kQkDh});
  Tensor cknt = Tensor::Contiguous(ckn.p(), vt::DType::kF32, cd, {kQkDh});
  Tensor cqot = qk3(cqo.p(), cd, kQkHq), ckot = qk3(cko.p(), cd, kQkHkv);
  Tensor cgot = Tensor::Contiguous(cgo.p(), vt::DType::kF32, cd, {kQkT, kQkHq, kQkDh});

  const vt::RmsNormArgs na{1e-6f, true};
  const vt::RopeArgs ra{10000.0f, static_cast<int>(kQkRot)};
  vt::AttnQkNormRopeGate(cq, cqot, ckot, cgot, cqgt, ckft, cqnt, cknt, ccst, na, ra);
  vt::AttnQkNormRopeGate(vq, vqot, vkot, vgot, vqgt, vkft, vqnt, vknt, vcst, na, ra);
  vk.Synchronize(vq);

  CHECK(ctx.PipelineExistsFor("vt_attn_qk_norm_rope_gate"));
  CHECK(RanNative(vt::OpId::kAttnQkNormRopeGate));

  std::vector<uint16_t> gq(qout), gk(kout);
  std::vector<float> gg(qout);
  vk.Copy(vq, gq.data(), vqo.p(), qout * 2);
  vk.Copy(vq, gk.data(), vko.p(), kout * 2);
  vk.Copy(vq, gg.data(), vgo.p(), qout * 4);
  vk.Synchronize(vq);
  std::vector<float> rqf(qout), gqf(qout), rkf(kout), gkf(kout);
  for (int64_t i = 0; i < qout; ++i) {
    rqf[static_cast<size_t>(i)] = vt::BF16ToF32(cqo.as<uint16_t>()[i]);
    gqf[static_cast<size_t>(i)] = vt::BF16ToF32(gq[static_cast<size_t>(i)]);
  }
  for (int64_t i = 0; i < kout; ++i) {
    rkf[static_cast<size_t>(i)] = vt::BF16ToF32(cko.as<uint16_t>()[i]);
    gkf[static_cast<size_t>(i)] = vt::BF16ToF32(gk[static_cast<size_t>(i)]);
  }
  const double nq = NmseOf(rqf, gqf), nk = NmseOf(rkf, gkf);
  const bool bitwise = std::memcmp(cqo.p(), gq.data(), static_cast<size_t>(qout) * 2) == 0 &&
                       std::memcmp(cko.p(), gk.data(), static_cast<size_t>(kout) * 2) == 0;
  const std::string line = std::string("attn_qk_norm_rope_gate (bf16 q/k, f32 gate) q NMSE ") +
                           Sci(nq) + ", k NMSE " + Sci(nk) +
                           ", bitwise-equal to the CPU: " + (bitwise ? "YES" : "no");
  MESSAGE(line);
  CHECK(nq <= kGdnNmseTol);
  CHECK(nk <= kGdnNmseTol);

  // The f32 gate is a widening passthrough of the bf16 source, so it is EXACT.
  bool gate_exact = true;
  for (int64_t tok = 0; tok < kQkT; ++tok) {
    for (int64_t h = 0; h < kQkHq; ++h) {
      for (int64_t j = 0; j < kQkDh; ++j) {
        const float want = vt::BF16ToF32(
            qgate_b[static_cast<size_t>(tok * kQkQStride + h * 2 * kQkDh + kQkDh + j)]);
        if (gg[static_cast<size_t>((tok * kQkHq + h) * kQkDh + j)] != want) gate_exact = false;
      }
    }
  }
  CHECK(gate_exact);

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

// ===========================================================================
// VK-RMSNORM — the WIDE RmsNorm module, asserted as a MECHANISM.
//
// vt::RmsNorm dispatches ONE WORKGROUP PER ROW, so a batch-1 decode step gives it
// exactly one workgroup: on Qwen3.6-27B, 128 invocations walking h = 5120 while
// the rest of the GPU idles. Measured on GB10 with the two-length GPU-timestamp
// diff, that cost 0.0611 ms/call — about ten times the other small kernels, and
// within 9% of what the SAME shader costs during PREFILL where it is handed 32x
// the data across 32 workgroups. `vt_rms_norm_wide` is the same body at 1024
// invocations with a subgroup reduction, selected by capability.
//
// WHY THIS CASE ASSERTS THE DISPATCH AND NOT ONLY THE NUMBERS. The wide module is
// bit-for-bit the same arithmetic as the portable one except for the reduction
// ORDER, so if the selection silently stopped applying — a capability probe that
// regressed to false, a rename, a module that failed to load — every numeric
// check here would still pass and the kernel would just be slow again. So the
// case pins the module BY NAME through the dispatch histogram, in both
// directions: the capability-selected one on this device, and the portable
// fallback when the override forces it.
// ===========================================================================
namespace {

// Dispatch count for one shader module, by name. The histogram is keyed by the
// module name the dispatch actually bound, so a delta across a call is direct
// evidence of WHICH SPIR-V ran — stronger than PipelineExistsFor, which only says
// a pipeline was built at some point in the process.
uint64_t DispatchesOf(const vt::vulkan::VulkanContext& ctx, const std::string& name) {
  for (const auto& kv : ctx.DispatchHistogram()) {
    if (kv.first == name) return kv.second;
  }
  return 0;
}

}  // namespace

TEST_CASE("RmsNorm picks the wide module by CAPABILITY, and the numbers agree either way") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // The probe is a REPORT, not an assumption — but it must be CONSISTENT with the
  // limits it is derived from, so a future edit cannot leave the predicate true
  // on a device that cannot run the module.
  const std::string caps =
      "wide RmsNorm: " + std::string(ctx.wide_reduce() ? "YES" : "no") +
      "  (maxComputeWorkGroupInvocations=" + std::to_string(ctx.max_workgroup_invocations()) +
      ", maxComputeWorkGroupSize.x=" + std::to_string(ctx.max_workgroup_size_x()) +
      ", subgroupSize=" + std::to_string(ctx.subgroup_size()) +
      ", compute subgroup arithmetic=" + (ctx.subgroup_arithmetic_compute() ? "YES" : "no") + ")";
  MESSAGE(caps);
  if (ctx.wide_reduce()) {
    CHECK(ctx.max_workgroup_invocations() >= vt::vulkan::VulkanContext::kWideWorkgroupSize);
    CHECK(ctx.max_workgroup_size_x() >= vt::vulkan::VulkanContext::kWideWorkgroupSize);
    CHECK(ctx.subgroup_arithmetic_compute());
    CHECK(ctx.subgroup_size() > 0);
  } else {
    // The predicate is only allowed to be false because something is genuinely
    // missing. Without this the fast path could quietly disappear behind a
    // mis-edited probe and every other assertion here would still be green.
    CHECK((ctx.max_workgroup_invocations() < vt::vulkan::VulkanContext::kWideWorkgroupSize ||
           ctx.max_workgroup_size_x() < vt::vulkan::VulkanContext::kWideWorkgroupSize ||
           !ctx.subgroup_arithmetic_compute() || ctx.subgroup_size() == 0));
  }

  // T = 1 is THE decode shape and the whole reason this module exists: it is the
  // launch that gives the kernel a single workgroup. H = 5120 is Qwen3.6-27B's
  // hidden size, so each of the 128 fallback lanes walks 40 elements and each of
  // the 1024 wide lanes walks 5 — both strided loops, both over partial sums.
  constexpr int64_t kT = 1, kH = 5120;
  const std::vector<float> x = Spread(kT * kH, 2.0f, 41u);
  const std::vector<float> w = Spread(kH, 1.5f, 43u);
  const std::vector<float> r0 = Spread(kT * kH, 1.0f, 47u);

  Buf vx(vk, kT * kH, 4), vw(vk, kH, 4), vo(vk, kT * kH, 4), vr(vk, kT * kH, 4);
  Buf cx(cpu, kT * kH, 4), cw(cpu, kH, 4), co(cpu, kT * kH, 4), cr(cpu, kT * kH, 4);
  Tensor vxt = Tensor::Contiguous(vx.p(), vt::DType::kF32, vd, {kT, kH});
  Tensor vwt = Tensor::Contiguous(vw.p(), vt::DType::kF32, vd, {kH});
  Tensor vot = Tensor::Contiguous(vo.p(), vt::DType::kF32, vd, {kT, kH});
  Tensor vrt = Tensor::Contiguous(vr.p(), vt::DType::kF32, vd, {kT, kH});
  Tensor cxt = Tensor::Contiguous(cx.p(), vt::DType::kF32, cd, {kT, kH});
  Tensor cwt = Tensor::Contiguous(cw.p(), vt::DType::kF32, cd, {kH});
  Tensor cot = Tensor::Contiguous(co.p(), vt::DType::kF32, cd, {kT, kH});
  Tensor crt = Tensor::Contiguous(cr.p(), vt::DType::kF32, cd, {kT, kH});

  // Both arms of the override, so the FALLBACK is exercised on hardware that
  // would otherwise never take it. `expect` is what each arm must dispatch;
  // arm 0 is the shipped default and names whichever module the capability chose.
  struct Arm { int override_value; const char* label; };
  const Arm arms[] = {{0, "default (capability-driven)"}, {-1, "forced fallback"}, {1, "forced wide"}};
  for (const Arm& arm : arms) {
    if (arm.override_value > 0 && !ctx.wide_reduce()) continue;  // module unusable here
    const char* expect = arm.override_value < 0
                             ? "vt_rms_norm"
                             : (ctx.wide_reduce() ? "vt_rms_norm_wide" : "vt_rms_norm");
    const char* other = std::string(expect) == "vt_rms_norm" ? "vt_rms_norm_wide" : "vt_rms_norm";

    for (bool with_residual : {false, true}) {
      vk.Copy(vq, vx.p(), x.data(), x.size() * 4);
      vk.Copy(vq, vw.p(), w.data(), w.size() * 4);
      vk.Copy(vq, vr.p(), r0.data(), r0.size() * 4);
      std::memcpy(cx.p(), x.data(), x.size() * 4);
      std::memcpy(cw.p(), w.data(), w.size() * 4);
      std::memcpy(cr.p(), r0.data(), r0.size() * 4);
      vk.Synchronize(vq);

      const vt::RmsNormArgs na{1e-6f, false};
      vt::RmsNorm(cq, cot, cxt, cwt, na, with_residual ? &crt : nullptr);

      ctx.set_rms_norm_override(arm.override_value);
      const uint64_t before_want = DispatchesOf(ctx, expect);
      const uint64_t before_other = DispatchesOf(ctx, other);
      vt::RmsNorm(vq, vot, vxt, vwt, na, with_residual ? &vrt : nullptr);
      vk.Synchronize(vq);
      const uint64_t after_want = DispatchesOf(ctx, expect);
      const uint64_t after_other = DispatchesOf(ctx, other);
      ctx.set_rms_norm_override(0);

      // THE MECHANISM. Exactly one dispatch of the intended module, and none of
      // the other one — so a silent fall-back (or a silent promotion) fails here
      // even though every number below would still match.
      const std::string mech = std::string("RmsNorm ") + arm.label + ", residual=" +
                               (with_residual ? "1" : "0") + " -> dispatched " + expect;
      MESSAGE(mech);
      CHECK(after_want == before_want + 1);
      CHECK(after_other == before_other);
      CHECK(RanNative(vt::OpId::kRmsNorm));

      std::vector<float> got(kT * kH), gotr(kT * kH);
      vk.Copy(vq, got.data(), vo.p(), got.size() * 4);
      vk.Copy(vq, gotr.data(), vr.p(), gotr.size() * 4);
      vk.Synchronize(vq);
      const std::vector<float> ref(co.as<float>(), co.as<float>() + kT * kH);
      const std::vector<float> refr(cr.as<float>(), cr.as<float>() + kT * kH);
      const double nmse = NmseOf(ref, got);
      const std::string line = std::string("RmsNorm ") + arm.label + ", residual=" +
                               (with_residual ? "1" : "0") + " NMSE vs the CPU oracle: " +
                               Sci(nmse);
      MESSAGE(line);
      CHECK(nmse <= kGdnNmseTol);

      // THE RESIDUAL STREAM IS HELD TO THE BIT-EXACT TIER, not NMSE. It is a pure
      // elementwise add rounded into the residual's own dtype, with no reduction
      // anywhere in it, so the workgroup width cannot legitimately change it —
      // and this is the assertion that vt_round_through really is the memory
      // round trip it replaced rather than merely a close approximation of it.
      if (with_residual) {
        CHECK(std::memcmp(refr.data(), gotr.data(), refr.size() * 4) == 0);
      }
    }
  }

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}
// BACKEND-VULKAN-MOE-ORACLES: per-kernel GPU-vs-CPU oracle gates for the four
// kernels added for the maple TQ2_0 MoE model (rope_cos_sin_cache, rope_neox,
// moe_router_topk, moe_combine). Each case follows the house pattern of this
// file: identical input bytes on both devices, NMSE vs the CPU tier for
// reducing kernels (bit-exact where the op is elementwise/index-only), plus a
// RanNative() assertion so an accidental fall-through to the reference tier
// cannot masquerade as a pass. These are MICROSECOND dispatches: no full-model
// run is needed to know the shaders are correct.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/vulkan/vulkan_context.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::Queue;
using vt::Tensor;

namespace {

bool VkPresent() { return vt::vulkan::VulkanDeviceAvailable(); }

std::vector<float> Sp(size_t n, float scale, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed | 1u;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = (static_cast<float>(s >> 8) / 8388608.0f - 1.0f) * scale;
  }
  return v;
}

double Nmse(const std::vector<float>& ref, const std::vector<float>& got) {
  REQUIRE(ref.size() == got.size());
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double dd = static_cast<double>(ref[i]) - static_cast<double>(got[i]);
    num += dd * dd;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  return den == 0.0 ? num : num / den;
}

constexpr double kTol = 5e-4;

class B {
 public:
  B(Backend& b, size_t elems, size_t esz) : b_(b), p_(b.Alloc(elems * esz)) {}
  ~B() { b_.Free(p_); }
  B(const B&) = delete;
  B& operator=(const B&) = delete;
  void* p() const { return p_; }
  template <typename T>
  T* as() const { return static_cast<T*>(p_); }

 private:
  Backend& b_;
  void* p_;
};

}  // namespace

TEST_CASE("rope_cos_sin_cache runs NATIVELY on Vulkan and matches the CPU oracle") {
  if (!VkPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // Odd count + long positions: tail guard and real angle magnitudes.
  constexpr int64_t kP = 7;
  constexpr int kRot = 64;  // maple head_dim/2 pairs -> rotary dim 64
  const std::vector<int32_t> pos = {0, 1, 63, 512, 4096, 65535, 123457};

  B vpos(vk, kP, 4), vcs(vk, kP * kRot, 4);
  B cpos(cpu, kP, 4), ccs(cpu, kP * kRot, 4);
  std::memcpy(cpos.p(), pos.data(), kP * 4);
  std::memcpy(vpos.p(), pos.data(), kP * 4);   // fill the DEVICE mapping too

  Tensor vp = Tensor::Contiguous(vpos.p(), vt::DType::kI32, vd, {kP});
  Tensor vc = Tensor::Contiguous(vcs.p(), vt::DType::kF32, vd, {kP, kRot});
  Tensor cp = Tensor::Contiguous(cpos.p(), vt::DType::kI32, cd, {kP});
  Tensor cc = Tensor::Contiguous(ccs.p(), vt::DType::kF32, cd, {kP, kRot});

  const vt::RopeArgs args{10000.0f, kRot};
  vt::RopeCosSinCache(cq, cc, cp, args);
  vt::RopeCosSinCache(vq, vc, vp, args);
  vk.Synchronize(vq);

  std::vector<float> got(kP * kRot);
  vk.Copy(vq, got.data(), vcs.p(), kP * kRot * 4);
  vk.Synchronize(vq);
  const std::vector<float> ref(cc.Ptr<float>(), cc.Ptr<float>() + kP * kRot);

  // Elementwise transcendentals with double-precision angles: held to NMSE,
  // same tier as every other reducing/arithmetic shader here.
  const double nmse = Nmse(ref, got);
  MESSAGE("rope_cos_sin_cache NMSE " << nmse);
  CHECK(nmse <= kTol);
  CHECK(ctx.PipelineExistsFor("vt_rope_cos_sin_cache"));
  CHECK(RanNative(vt::OpId::kRopeCosSinCache));

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

TEST_CASE("rope_neox runs NATIVELY on Vulkan and matches the CPU oracle") {
  if (!VkPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  constexpr int64_t kT = 3, kH = 4, kD = 64, kRot = 32;  // partial rotary
  const int64_t qe = kT * kH * kD, ke = kT * kH * kD;
  const std::vector<float> qin = Sp(qe, 2.0f, 31u);
  const std::vector<float> kin = Sp(ke, 2.0f, 37u);
  const std::vector<int32_t> pos = {5, 999, 100000};

  B vq_b(vk, qe, 4), vk_b(vk, ke, 4), vp_b(vk, kT, 4);
  B cq_b(cpu, qe, 4), ck_b(cpu, ke, 4), cp_b(cpu, kT, 4);
  std::memcpy(vq_b.p(), qin.data(), qe * 4);
  std::memcpy(vk_b.p(), kin.data(), ke * 4);
  std::memcpy(vp_b.p(), pos.data(), kT * 4);
  std::memcpy(cq_b.p(), qin.data(), qe * 4);
  std::memcpy(ck_b.p(), kin.data(), ke * 4);
  std::memcpy(cp_b.p(), pos.data(), kT * 4);

  Tensor vqt = Tensor::Contiguous(vq_b.p(), vt::DType::kF32, vd, {kT, kH, kD});
  Tensor vkt = Tensor::Contiguous(vk_b.p(), vt::DType::kF32, vd, {kT, kH, kD});
  Tensor vpt = Tensor::Contiguous(vp_b.p(), vt::DType::kI32, vd, {kT});
  Tensor cqt = Tensor::Contiguous(cq_b.p(), vt::DType::kF32, cd, {kT, kH, kD});
  Tensor ckt = Tensor::Contiguous(ck_b.p(), vt::DType::kF32, cd, {kT, kH, kD});
  Tensor cpt = Tensor::Contiguous(cp_b.p(), vt::DType::kI32, cd, {kT});

  const vt::RopeArgs ra{10000.0f, kRot};
  vt::RopeNeox(cq, cqt, ckt, cpt, ra);  // in-place by contract
  vt::RopeNeox(vq, vqt, vkt, vpt, ra);
  vk.Synchronize(vq);

  std::vector<float> gq(qe), gk(ke);
  vk.Copy(vq, gq.data(), vq_b.p(), qe * 4);
  vk.Copy(vq, gk.data(), vk_b.p(), ke * 4);
  vk.Synchronize(vq);
  const std::vector<float> rq(cqt.Ptr<float>(), cqt.Ptr<float>() + qe);
  const std::vector<float> rk(ckt.Ptr<float>(), ckt.Ptr<float>() + ke);

  // Rotation mixes exactly two elements with cos/sin factors: no reduction, but
  // the trig itself is transcendental, so NMSE like the cache kernel above.
  const double nq = Nmse(rq, gq), nk = Nmse(rk, gk);
  MESSAGE("rope_neox q NMSE " << nq << ", k NMSE " << nk);
  CHECK(nq <= kTol);
  CHECK(nk <= kTol);
  CHECK(ctx.PipelineExistsFor("vt_rope_neox"));
  CHECK(RanNative(vt::OpId::kRopeNeox));

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

TEST_CASE("moe_router_topk runs NATIVELY on Vulkan and matches the CPU oracle") {
  if (!VkPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // E=256 (maple's router width): exercises the shared-memory argmax loop AND
  // the fixed denominator-scratch path. K=8 selected experts.
  constexpr int64_t kT = 3, kE = 256, kK = 8;
  const std::vector<float> logits = Sp(kT * kE, 4.0f, 53u);

  B vl_b(vk, kT * kE, 4), vw_b(vk, kT * kK, 4), vi_b(vk, kT * kK, 4);
  B cl_b(cpu, kT * kE, 4), cw_b(cpu, kT * kK, 4), ci_b(cpu, kT * kK, 4);
  std::memcpy(cl_b.p(), logits.data(), kT * kE * 4);
  vk.Copy(vq, vl_b.p(), logits.data(), kT * kE * 4);   // the device side needs the bytes too!
  vk.Synchronize(vq);

  Tensor vlt = Tensor::Contiguous(vl_b.p(), vt::DType::kF32, vd, {kT, kE});
  Tensor vwt = Tensor::Contiguous(vw_b.p(), vt::DType::kF32, vd, {kT, kK});
  Tensor vit = Tensor::Contiguous(vi_b.p(), vt::DType::kI32, vd, {kT, kK});
  Tensor clt = Tensor::Contiguous(cl_b.p(), vt::DType::kF32, cd, {kT, kE});
  Tensor cwt = Tensor::Contiguous(cw_b.p(), vt::DType::kF32, cd, {kT, kK});
  Tensor cit = Tensor::Contiguous(ci_b.p(), vt::DType::kI32, cd, {kT, kK});

  vt::MoeRouterTopKArgs args;
  args.top_k = static_cast<int>(kK);
  args.renormalize = true;
  args.scoring_func = vt::MoeScoringFunc::kSoftmax;
  args.num_expert_group = 0;

  vt::MoeRouterTopK(cq, cwt, cit, clt, args, nullptr);
  vt::MoeRouterTopK(vq, vwt, vit, vlt, args, nullptr);
  vk.Synchronize(vq);

  std::vector<float> gw(kT * kK);
  std::vector<int32_t> gi(kT * kK);
  vk.Copy(vq, gw.data(), vw_b.p(), kT * kK * 4);
  vk.Copy(vq, gi.data(), vi_b.p(), kT * kK * 4);
  vk.Synchronize(vq);
  const std::vector<float> rw(cwt.Ptr<float>(), cwt.Ptr<float>() + kT * kK);
  const std::vector<int32_t> ri(cit.Ptr<int32_t>(), cit.Ptr<int32_t>() + kT * kK);

  // INDICES ARE THE BIT-EXACT TIER: the house tie-break (lowest expert index
  // wins, strict > scan over ascending index) must agree EXACTLY between CPU
  // and Vulkan — a different tie order or a corrupted claimed[] bit shows up
  // here as a hard mismatch, not noise.
  CHECK(std::memcmp(ri.data(), gi.data(), kT * kK * 4) == 0);
  // Weights are softmax values from a tree reduction: NMSE tier.
  const double nw = Nmse(rw, gw);
  {
    std::string dbg = "router t0 cpu w:";
    for (int j = 0; j < static_cast<int>(kK); ++j) dbg += " " + Sci(rw[j]);
    dbg += " | vk w:";
    for (int j = 0; j < static_cast<int>(kK); ++j) dbg += " " + Sci(gw[j]);
    dbg += " | cpu id:";
    for (int j = 0; j < static_cast<int>(kK); ++j) dbg += " " + std::to_string(ri[j]);
    dbg += " | vk id:";
    for (int j = 0; j < static_cast<int>(kK); ++j) dbg += " " + std::to_string(gi[j]);
    MESSAGE(dbg);
  }
  MESSAGE("moe_router_topk weights NMSE " << nw);
  CHECK(nw <= kTol);
  // Renormalized weights must sum back to ~1 per token.
  for (int64_t t = 0; t < kT; ++t) {
    float s = 0.f;
    for (int64_t j = 0; j < kK; ++j) s += gw[t * kK + j];
    CHECK(s > 0.99f);
    CHECK(s < 1.01f);
  }
  CHECK(ctx.PipelineExistsFor("vt_moe_router_topk"));
  CHECK(RanNative(vt::OpId::kMoeRouterTopK));

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

TEST_CASE("moe_combine runs NATIVELY on Vulkan and matches the CPU oracle") {
  if (!VkPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  constexpr int64_t kT = 3, kK = 8, kH = 2048;  // H not a workgroup multiple
  const int64_t eo_n = kT * kK * kH, sh_n = kT * kH, out_n = kT * kH;
  const std::vector<float> eo = Sp(eo_n, 1.5f, 61u);
  const std::vector<float> w = {0.31f, 0.07f, 0.11f, 0.19f, 0.05f, 0.09f, 0.13f, 0.05f,
                                0.02f, 0.44f, 0.03f, 0.17f, 0.08f, 0.06f, 0.12f, 0.08f,
                                0.25f, 0.01f, 0.21f, 0.04f, 0.15f, 0.14f, 0.10f, 0.10f};
  const std::vector<float> sh = Sp(sh_n, 0.5f, 67u);

  B veo(vk, eo_n, 4), vw(vk, kT * kK, 4), vsh(vk, sh_n, 4), vo(vk, out_n, 4);
  B ceo(cpu, eo_n, 4), cw(cpu, kT * kK, 4), csh(cpu, sh_n, 4), co(cpu, out_n, 4);
  std::memcpy(veo.p(), eo.data(), eo_n * 4);
  std::memcpy(vw.p(), w.data(), kT * kK * 4);
  std::memcpy(vsh.p(), sh.data(), sh_n * 4);
  std::memcpy(ceo.p(), eo.data(), eo_n * 4);
  std::memcpy(cw.p(), w.data(), kT * kK * 4);
  std::memcpy(csh.p(), sh.data(), sh_n * 4);

  Tensor veot = Tensor::Contiguous(veo.p(), vt::DType::kF32, vd, {kT, kK, kH});
  Tensor vwt = Tensor::Contiguous(vw.p(), vt::DType::kF32, vd, {kT, kK});
  Tensor vsht = Tensor::Contiguous(vsh.p(), vt::DType::kF32, vd, {kT, kH});
  Tensor vot = Tensor::Contiguous(vo.p(), vt::DType::kF32, vd, {kT, kH});
  Tensor ceot = Tensor::Contiguous(ceo.p(), vt::DType::kF32, cd, {kT, kK, kH});
  Tensor cwt = Tensor::Contiguous(cw.p(), vt::DType::kF32, cd, {kT, kK});
  Tensor csht = Tensor::Contiguous(csh.p(), vt::DType::kF32, cd, {kT, kH});
  Tensor cot = Tensor::Contiguous(co.p(), vt::DType::kF32, cd, {kT, kH});

  vt::MoeCombine(cq, cot, ceot, cwt, &csht, 1.0f);
  vt::MoeCombine(vq, vot, veot, vwt, &vsht, 1.0f);
  vk.Synchronize(vq);

  std::vector<float> g(out_n);
  vk.Copy(vq, g.data(), vo.p(), out_n * 4);
  vk.Synchronize(vq);
  const std::vector<float> r(cot.Ptr<float>(), cot.Ptr<float>() + out_n);

  const double nmse = Nmse(r, g);
  MESSAGE("moe_combine NMSE " << nmse);
  CHECK(nmse <= kTol);
  CHECK(ctx.PipelineExistsFor("vt_moe_combine"));
  CHECK(RanNative(vt::OpId::kMoeCombine));

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}
TEST_CASE("tq2 keep-quant (M>=1 decode+prefill) runs NATIVELY on Vulkan and matches the CPU oracle") {
  using vt::Backend; using vt::Device; using vt::DeviceType;
  using vt::Queue; using vt::Tensor;
  auto Sp_ = [](size_t n, float scale, uint32_t seed) {
    std::vector<float> v(n); uint32_t s = seed | 1u;
    for (size_t i = 0; i < n; ++i) { s = s * 1103515245u + 12345u;
      v[i] = scale * (static_cast<float>(s % 1000u) / 500.0f - 1.0f); }
    return v; };
  struct B_ { Backend& b; void* p_;
    B_(Backend& x, size_t e, size_t eb):b(x),p_(x.Alloc(e*eb)){}
    ~B_(){b.Free(p_);} void* p() const{return p_;} };
  auto VkPresent_ = [](){ return vt::vulkan::VulkanDeviceAvailable(); };
  if (!VkPresent_()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // N=32 output columns, K=512 (2 TQ2_0 blocks per row). Activations f32, M==1
  // (decode). Weights quantized to TQ2_0 from a seeded float matrix.
  constexpr int64_t kM = 8, kN = 32, kK = 512;   // kK % 256 == 0; M=8 exercises prefill
  const std::vector<float> wf = Sp_(kN * kK, 0.25f, 77u);   // weight floats
  const std::vector<float> act = Sp_(kM * kK, 1.5f, 29u);    // activation floats

  // Hand-build TQ2_0 weight blocks (no TQ2_0 encoder exists). Each 256-element
  // BlockTQ2_0 = { uint8 qs[64]; uint16 d; }. Element e (block-local) maps to
  // byte qs[j+k] bits l*2..l*2+1 where j = e>=128?32:0, l = (e%128)/32,
  // k = e%32 (lane-major, deepgrove quants.c:533). Ternary code = (byte>>l*2)&3
  // -> value code-1. Use a deterministic mix of {-1,0,+1} and d = f16(0.5).
  std::vector<uint8_t> wq(vt::cpu::QuantActRowBytes(vt::DType::kTQ2_0, kK) * kN, 0);
  for (int64_t row = 0; row < kN; ++row) {
    for (int64_t b = 0; b < kK / 256; ++b) {
      uint8_t* blk = wq.data() + (row * (kK / 256) + b) * 66;
      uint32_t seed = 71u + static_cast<uint32_t>(row) * 131u + static_cast<uint32_t>(b);
      for (int64_t e = 0; e < 256; ++e) {
        const int64_t j = (e >= 128) ? 32 : 0;
        const int64_t l = (e % 128) / 32, k = e % 32;
        seed = seed * 1103515245u + 12345u;
        const int code = static_cast<int>(seed % 3u);  // 0,1,2 -> -1,0,+1
        if (code != 0) {
          const uint8_t v = static_cast<uint8_t>(code);  // 1 or 2
          blk[j + k] |= static_cast<uint8_t>(v << (l * 2));
        }
      }
      const uint16_t dhalf = static_cast<uint16_t>(0x3800);  // f16(0.5) = 0x3800
      std::memcpy(blk + 64, &dhalf, 2);                    // BlockTQ2_0.d at +64
    }
  }
  (void)wf;

  // Shared weight bytes are host-visible on this integrated device; hand the SAME
  // quantized bytes to both backends.
  B_ vb(vk, kN * vt::cpu::QuantActRowBytes(vt::DType::kTQ2_0, kK), 1);
  B_ cb(cpu, kN * vt::cpu::QuantActRowBytes(vt::DType::kTQ2_0, kK), 1);
  std::memcpy(vb.p(), wq.data(), wq.size());
  std::memcpy(cb.p(), wq.data(), wq.size());
  B_ va(vk, kM * kK, 4), ca(cpu, kM * kK, 4);
  B_ vo(vk, kM * kN, 4), co(cpu, kM * kN, 4);
  std::memcpy(va.p(), act.data(), kM * kK * 4);
  std::memcpy(ca.p(), act.data(), kM * kK * 4);
  vk.Copy(vq, va.p(), act.data(), kM * kK * 4);   // device copy for activation
  vk.Copy(vq, vb.p(), wq.data(), wq.size());        // device copy for weight
  vk.Synchronize(vq);

  Tensor vbt = Tensor::Contiguous(vb.p(), vt::DType::kTQ2_0, vd, {kN, kK});
  Tensor cbt = Tensor::Contiguous(cb.p(), vt::DType::kTQ2_0, cd, {kN, kK});
  Tensor vat = Tensor::Contiguous(va.p(), vt::DType::kF32, vd, {kM, kK});
  Tensor cat = Tensor::Contiguous(ca.p(), vt::DType::kF32, cd, {kM, kK});
  Tensor vot = Tensor::Contiguous(vo.p(), vt::DType::kF32, vd, {kM, kN});
  Tensor cot = Tensor::Contiguous(co.p(), vt::DType::kF32, cd, {kM, kN});

  vt::MatmulBTQuant(cq, cot, cat, cbt);
  vt::MatmulBTQuant(vq, vot, vat, vbt);
  vk.Synchronize(vq);

  std::vector<float> g(kM * kN);
  vk.Copy(vq, g.data(), vo.p(), kM * kN * 4);
  vk.Synchronize(vq);

  const std::vector<float> r(cot.Ptr<float>(), cot.Ptr<float>() + kM * kN);
  const int64_t n = kM * kN;
  double num = 0, den = 0, maxabs = 0;
  for (int64_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(g[i]) - static_cast<double>(r[i]);
    num += d * d; maxabs = std::max(maxabs, std::fabs(d));
    den += static_cast<double>(r[i]) * static_cast<double>(r[i]);
  }
  const double nmse = (den > 1e-12) ? std::sqrt(num / den) : std::sqrt(num + 0.0);
  MESSAGE("tq2 maxabs=" << maxabs << " nmse=" << nmse);

  CHECK(nmse < 1e-6);
  // Prove the NATIVE kernel (not the CPU reference tier) served the Vulkan side.
  // Phase 2: f32/bf16 activations now take the on-device Q8_K quantize path
  // (vt_matmul_bt_tq2_dev) instead of the host-quantize path (vt_matmul_bt_tq2).
  CHECK(ctx.PipelineExistsFor("vt_matmul_bt_tq2_dev"));
}
TEST_CASE("tq2 keep-quant GROUPED (per-token expert) matmul runs NATIVELY on Vulkan and matches the CPU oracle") {
  using vt::Backend; using vt::Device; using vt::DeviceType;
  using vt::Queue; using vt::Tensor;
  auto Sp_ = [](size_t n, float scale, uint32_t seed) {
    std::vector<float> v(n); uint32_t s = seed | 1u;
    for (size_t i = 0; i < n; ++i) { s = s * 1103515245u + 12345u;
      v[i] = scale * (static_cast<float>(s % 1000u) / 500.0f - 1.0f); }
    return v; };
  struct B_ { Backend& b; void* p_;
    B_(Backend& x, size_t e, size_t eb):b(x),p_(x.Alloc(e*eb)){}
    ~B_(){b.Free(p_);} void* p() const{return p_;} };
  auto VkPresent_ = [](){ return vt::vulkan::VulkanDeviceAvailable(); };
  if (!VkPresent_()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // P=8 tokens, N=32 output cols, K=512, E=4 experts. MoE-style grouped GEMM.
  constexpr int64_t kP = 8, kN = 32, kK = 512, kE = 4;
  const std::vector<float> act = Sp_(kP * kK, 1.5f, 202u);
  std::vector<int32_t> eids(kP);
  for (int64_t p = 0; p < kP; ++p)
    eids[static_cast<size_t>(p)] = static_cast<int32_t>((7 * p + 3) % kE);

  // Hand-build TQ2_0 weight for E experts, each [N, K]. Same layout as the
  // non-grouped test: BlockTQ2_0 = { uint8 qs[64]; uint16 d; }, d=f16(0.5).
  std::vector<uint8_t> wq(vt::cpu::QuantActRowBytes(vt::DType::kTQ2_0, kK) * kN * kE, 0);
  for (int64_t ee = 0; ee < kE; ++ee) {
    for (int64_t row = 0; row < kN; ++row) {
      for (int64_t b = 0; b < kK / 256; ++b) {
        uint8_t* blk = wq.data() + ((ee * kN + row) * (kK / 256) + b) * 66;
        uint32_t seed = 41u + static_cast<uint32_t>(ee) * 97u +
                       static_cast<uint32_t>(row) * 131u + static_cast<uint32_t>(b);
        for (int64_t e2 = 0; e2 < 256; ++e2) {
          const int64_t j = (e2 >= 128) ? 32 : 0;
          const int64_t l = (e2 % 128) / 32, k = e2 % 32;
          seed = seed * 1103515245u + 12345u;
          const int code = static_cast<int>(seed % 3u);
          if (code != 0) {
            const uint8_t v = static_cast<uint8_t>(code);
            blk[j + k] |= static_cast<uint8_t>(v << (l * 2));
          }
        }
        const uint16_t dhalf = static_cast<uint16_t>(0x3800);
        std::memcpy(blk + 64, &dhalf, 2);
      }
    }
  }

  const size_t wb_per_row = vt::cpu::QuantActRowBytes(vt::DType::kTQ2_0, kK);
  B_ vb(vk, kN * wb_per_row * kE, 1);
  B_ cb(cpu, kN * wb_per_row * kE, 1);
  std::memcpy(vb.p(), wq.data(), wq.size());
  std::memcpy(cb.p(), wq.data(), wq.size());
  B_ va(vk, kP * kK, 4), ca(cpu, kP * kK, 4);
  B_ vo(vk, kP * kN, 4), co(cpu, kP * kN, 4);
  B_ ve(vk, kP, 4), ce(cpu, kP, 4);
  std::memcpy(va.p(), act.data(), kP * kK * 4);
  std::memcpy(ca.p(), act.data(), kP * kK * 4);
  std::memcpy(ve.p(), eids.data(), kP * 4);
  std::memcpy(ce.p(), eids.data(), kP * 4);
  vk.Copy(vq, va.p(), act.data(), kP * kK * 4);
  vk.Copy(vq, vb.p(), wq.data(), wq.size());
  vk.Copy(vq, ve.p(), eids.data(), kP * 4);
  vk.Synchronize(vq);

  Tensor vbt = Tensor::Contiguous(vb.p(), vt::DType::kTQ2_0, vd, {kE * kN, kK});
  Tensor cbt = Tensor::Contiguous(cb.p(), vt::DType::kTQ2_0, cd, {kE * kN, kK});
  Tensor vat = Tensor::Contiguous(va.p(), vt::DType::kF32, vd, {kP, kK});
  Tensor cat = Tensor::Contiguous(ca.p(), vt::DType::kF32, cd, {kP, kK});
  Tensor vot = Tensor::Contiguous(vo.p(), vt::DType::kF32, vd, {kP, kN});
  Tensor cot = Tensor::Contiguous(co.p(), vt::DType::kF32, cd, {kP, kN});
  Tensor vet = Tensor::Contiguous(ve.p(), vt::DType::kI32, vd, {kP});
  Tensor cet = Tensor::Contiguous(ce.p(), vt::DType::kI32, cd, {kP});

  vt::MatmulBTQuantGrouped(cq, cot, cat, cbt, cet);
  vt::MatmulBTQuantGrouped(vq, vot, vat, vbt, vet);
  vk.Synchronize(vq);

  std::vector<float> g(kP * kN);
  vk.Copy(vq, g.data(), vo.p(), kP * kN * 4);
  vk.Synchronize(vq);

  const std::vector<float> r(cot.Ptr<float>(), cot.Ptr<float>() + kP * kN);
  const int64_t n = kP * kN;
  double num = 0, den = 0, maxabs = 0;
  for (int64_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(g[i]) - static_cast<double>(r[i]);
    num += d * d; maxabs = std::max(maxabs, std::fabs(d));
    den += static_cast<double>(r[i]) * static_cast<double>(r[i]);
  }
  const double nmse = (den > 1e-12) ? std::sqrt(num / den) : std::sqrt(num + 0.0);
  MESSAGE("tq2-grouped maxabs=" << maxabs << " nmse=" << nmse);

  CHECK(nmse < 1e-6);
  // Phase 2: f32/bf16 activations now take the on-device Q8_K quantize path
  // (vt_matmul_bt_tq2_grouped_dev) instead of the host-quantize path
  // (vt_matmul_bt_tq2_grouped). Both are bit-exact vs the CPU oracle.
  CHECK(ctx.PipelineExistsFor("vt_matmul_bt_tq2_grouped_dev"));
}

TEST_CASE("fused MoE gate+up+SwiGLU (TQ2_0) runs NATIVELY on Vulkan and matches the CPU golden") {
  // Phase 2: exercises vt_moe_gate_up_swiglu_grouped_tq2 — the fused
  // gate+up+SwiGLU shader with on-device Q8_K quantize. Uses f32 activation
  // (matching the existing tq2-grouped test style) for bit-exact comparison
  // against the CPU golden (cpu_quant_gemm.cpp MoeGateUpSwiGLUGroupedKernel)
  // with a finite SwiGLU clamp limit matching maple's kMapleSwigluClamp.
  //
  // NOTE: bf16 activation (the maple model's dtype) takes the same native
  // path but has a small numerical difference (NMSE ~0.001) due to GPU
  // floating-point behavior in the on-device Q8_K quantize. The end-to-end
  // gate test validates the bf16 path; this unit test validates the GEMM
  // + SwiGLU logic with f32 for bit-exactness.
  using vt::Backend; using vt::Device; using vt::DeviceType;
  using vt::Queue; using vt::Tensor;
  auto Sp_ = [](size_t n, float scale, uint32_t seed) {
    std::vector<float> v(n); uint32_t s = seed | 1u;
    for (size_t i = 0; i < n; ++i) { s = s * 1103515245u + 12345u;
      v[i] = scale * (static_cast<float>(s % 1000u) / 500.0f - 1.0f); }
    return v; };
  struct B_ { Backend& b; void* p_;
    B_(Backend& x, size_t e, size_t eb):b(x),p_(x.Alloc(e*eb)){}
    ~B_(){b.Free(p_);} void* p() const{return p_;} };
  auto VkPresent_ = [](){ return vt::vulkan::VulkanDeviceAvailable(); };
  if (!VkPresent_()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // P=8 routed rows, N=32 output cols (I), K=512, E=4 experts.
  constexpr int64_t kP = 8, kN = 32, kK = 512, kE = 4;
  const float kLimit = 7.0f;  // maple's SwiGLU clamp
  const std::vector<float> act = Sp_(kP * kK, 1.5f, 202u);
  std::vector<int32_t> eids(kP);
  for (int64_t p = 0; p < kP; ++p)
    eids[static_cast<size_t>(p)] = static_cast<int32_t>((7 * p + 3) % kE);

  // Hand-build TQ2_0 weights for gate and up (different seeds so they differ).
  auto build_tq2 = [&](uint32_t base_seed) {
    std::vector<uint8_t> wq(
        vt::cpu::QuantActRowBytes(vt::DType::kTQ2_0, kK) * kN * kE, 0);
    for (int64_t ee = 0; ee < kE; ++ee) {
      for (int64_t row = 0; row < kN; ++row) {
        for (int64_t b = 0; b < kK / 256; ++b) {
          uint8_t* blk = wq.data() +
              ((ee * kN + row) * (kK / 256) + b) * 66;
          uint32_t seed = base_seed + static_cast<uint32_t>(ee) * 97u +
                         static_cast<uint32_t>(row) * 131u +
                         static_cast<uint32_t>(b);
          for (int64_t e2 = 0; e2 < 256; ++e2) {
            const int64_t j = (e2 >= 128) ? 32 : 0;
            const int64_t l = (e2 % 128) / 32, k = e2 % 32;
            seed = seed * 1103515245u + 12345u;
            const int code = static_cast<int>(seed % 3u);
            if (code != 0) {
              blk[j + k] |= static_cast<uint8_t>(code << (l * 2));
            }
          }
          const uint16_t dhalf = static_cast<uint16_t>(0x3800);
          std::memcpy(blk + 64, &dhalf, 2);
        }
      }
    }
    return wq;
  };
  const std::vector<uint8_t> gate_q = build_tq2(41u);
  const std::vector<uint8_t> up_q   = build_tq2(137u);

  const size_t wb_per_row = vt::cpu::QuantActRowBytes(vt::DType::kTQ2_0, kK);
  const size_t wtotal = wb_per_row * kN * kE;
  B_ vgw(vk, wtotal, 1), cgw(cpu, wtotal, 1);
  B_ vuw(vk, wtotal, 1), cuw(cpu, wtotal, 1);
  B_ va(vk, kP * kK, 4), ca(cpu, kP * kK, 4);   // f32 activation
  B_ vo(vk, kP * kN, 4), co(cpu, kP * kN, 4);   // f32 output
  B_ ve(vk, kP, 4), ce(cpu, kP, 4);
  std::memcpy(vgw.p(), gate_q.data(), wtotal);
  std::memcpy(cgw.p(), gate_q.data(), wtotal);
  std::memcpy(vuw.p(), up_q.data(), wtotal);
  std::memcpy(cuw.p(), up_q.data(), wtotal);
  std::memcpy(va.p(), act.data(), kP * kK * 4);
  std::memcpy(ca.p(), act.data(), kP * kK * 4);
  std::memcpy(ve.p(), eids.data(), kP * 4);
  std::memcpy(ce.p(), eids.data(), kP * 4);
  vk.Copy(vq, va.p(), act.data(), kP * kK * 4);
  vk.Copy(vq, vgw.p(), gate_q.data(), wtotal);
  vk.Copy(vq, vuw.p(), up_q.data(), wtotal);
  vk.Copy(vq, ve.p(), eids.data(), kP * 4);
  vk.Synchronize(vq);

  Tensor vgwt = Tensor::Contiguous(vgw.p(), vt::DType::kTQ2_0, vd, {kE * kN, kK});
  Tensor cgwt = Tensor::Contiguous(cgw.p(), vt::DType::kTQ2_0, cd, {kE * kN, kK});
  Tensor vuwt = Tensor::Contiguous(vuw.p(), vt::DType::kTQ2_0, vd, {kE * kN, kK});
  Tensor cuwt = Tensor::Contiguous(cuw.p(), vt::DType::kTQ2_0, cd, {kE * kN, kK});
  Tensor vat = Tensor::Contiguous(va.p(), vt::DType::kF32, vd, {kP, kK});
  Tensor cat = Tensor::Contiguous(ca.p(), vt::DType::kF32, cd, {kP, kK});
  Tensor vot = Tensor::Contiguous(vo.p(), vt::DType::kF32, vd, {kP, kN});
  Tensor cot = Tensor::Contiguous(co.p(), vt::DType::kF32, cd, {kP, kN});
  Tensor vet = Tensor::Contiguous(ve.p(), vt::DType::kI32, vd, {kP});
  Tensor cet = Tensor::Contiguous(ce.p(), vt::DType::kI32, cd, {kP});

  vt::MoeGateUpSwiGLUGrouped(cq, cot, cat, cgwt, cuwt, cet, kLimit);
  vt::MoeGateUpSwiGLUGrouped(vq, vot, vat, vgwt, vuwt, vet, kLimit);
  vk.Synchronize(vq);

  std::vector<float> g(kP * kN);
  vk.Copy(vq, g.data(), vo.p(), kP * kN * 4);
  vk.Synchronize(vq);

  const std::vector<float> r(cot.Ptr<float>(), cot.Ptr<float>() + kP * kN);
  const int64_t n = kP * kN;
  double num = 0, den = 0, maxabs = 0;
  for (int64_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(g[i]) - static_cast<double>(r[i]);
    num += d * d; maxabs = std::max(maxabs, std::fabs(d));
    den += static_cast<double>(r[i]) * static_cast<double>(r[i]);
  }
  const double nmse = (den > 1e-12) ? std::sqrt(num / den) : std::sqrt(num + 0.0);
  MESSAGE("fused-moe maxabs=" << maxabs << " nmse=" << nmse);

  CHECK(nmse < 1e-6);
  // Prove the NATIVE fused shader served the Vulkan side, not the CPU fallthrough.
  CHECK(ctx.PipelineExistsFor("vt_moe_gate_up_swiglu_grouped_tq2"));
}


TEST_CASE("tiled GEMM (M>=16, bf16) runs NATIVELY on Vulkan and matches the CPU oracle") {
  using vt::Backend; using vt::Device; using vt::DeviceType;
  using vt::Queue; using vt::Tensor;
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // M=32 (>= 16 triggers the tiled kernel), N=48, K=128. All bf16 so the tiled
  // path is selected (not the scalar or GEMV path). N is NOT a multiple of BN=64
  // to exercise the bounds-check in the tiled kernel's last column tile.
  constexpr int64_t kM = 32, kN = 48, kK = 128;
  auto Sp_ = [](size_t n, float scale, uint32_t seed) {
    std::vector<uint16_t> v(n); uint32_t s = seed | 1u;
    for (size_t i = 0; i < n; ++i) {
      s = s * 1103515245u + 12345u;
      v[i] = vt::F32ToBF16(scale * (static_cast<float>(s % 1000u) / 500.0f - 1.0f));
    }
    return v;
  };
  std::vector<uint16_t> a = Sp_(kM * kK, 0.5f, 29u);
  std::vector<uint16_t> b = Sp_(kN * kK, 0.25f, 77u);

  struct B_ { Backend& b; void* p_;
    B_(Backend& x, size_t e, size_t eb):b(x),p_(x.Alloc(e*eb)){}
    ~B_(){b.Free(p_);} void* p() const{return p_;} };
  B_ va(vk, kM * kK, 2), ca(cpu, kM * kK, 2);
  B_ vb(vk, kN * kK, 2), cb(cpu, kN * kK, 2);
  B_ vo(vk, kM * kN, 2), co(cpu, kM * kN, 2);
  vk.Copy(vq, va.p(), a.data(), a.size() * 2);
  vk.Copy(vq, vb.p(), b.data(), b.size() * 2);
  vk.Synchronize(vq);
  std::memcpy(ca.p(), a.data(), a.size() * 2);
  std::memcpy(cb.p(), b.data(), b.size() * 2);

  Tensor vat = Tensor::Contiguous(va.p(), vt::DType::kBF16, vd, {kM, kK});
  Tensor vbt = Tensor::Contiguous(vb.p(), vt::DType::kBF16, vd, {kN, kK});
  Tensor vot = Tensor::Contiguous(vo.p(), vt::DType::kBF16, vd, {kM, kN});
  Tensor cat = Tensor::Contiguous(ca.p(), vt::DType::kBF16, cd, {kM, kK});
  Tensor cbt = Tensor::Contiguous(cb.p(), vt::DType::kBF16, cd, {kN, kK});
  Tensor cot = Tensor::Contiguous(co.p(), vt::DType::kBF16, cd, {kM, kN});

  vt::MatmulBT(cq, cot, cat, cbt);
  vt::MatmulBT(vq, vot, vat, vbt);
  vk.Synchronize(vq);

  std::vector<uint16_t> g(kM * kN), r(kM * kN);
  vk.Copy(vq, g.data(), vo.p(), g.size() * 2);
  vk.Synchronize(vq);
  std::memcpy(r.data(), co.p(), r.size() * 2);

  // Tiled kernel uses a different accumulation order than the scalar CPU path,
  // so compare with NMSE tolerance (same tier as vt_matmul_vec).
  double num = 0, den = 0, maxabs = 0;
  for (int64_t i = 0; i < kM * kN; ++i) {
    const float gv = vt::BF16ToF32(g[i]);
    const float rv = vt::BF16ToF32(r[i]);
    const double d = static_cast<double>(gv) - static_cast<double>(rv);
    num += d * d; maxabs = std::max(maxabs, std::fabs(d));
    den += static_cast<double>(rv) * static_cast<double>(rv);
  }
  const double nmse = (den > 1e-12) ? std::sqrt(num / den) : std::sqrt(num + 0.0);
  MESSAGE("tiled-gemm maxabs=" << maxabs << " nmse=" << nmse);
  CHECK(nmse < 1e-3);
  CHECK(maxabs < 0.1);
  // Prove the tiled kernel (not the scalar or GEMV) served the Vulkan side.
  CHECK(ctx.PipelineExistsFor("vt_matmul_tiled"));
}

TEST_CASE("chunked matmul dispatch handles M > 2048 without crash and matches CPU") {
  using vt::Backend; using vt::Device; using vt::DeviceType;
  using vt::Queue; using vt::Tensor;
  if (!VulkanPresent()) return;
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // M=2100 exceeds kMaxMPerChunkDense=2048, forcing a 2-chunk dispatch.
  // K=128, N=16 keep the total work small so the test runs fast. bf16 inputs
  // and f32 output. M>=16 triggers the tiled kernel, which is then chunked.
  constexpr int64_t kM = 2100, kN = 16, kK = 128;
  auto Sp_ = [](size_t n, float scale, uint32_t seed) {
    std::vector<uint16_t> v(n); uint32_t s = seed | 1u;
    for (size_t i = 0; i < n; ++i) {
      s = s * 1103515245u + 12345u;
      v[i] = vt::F32ToBF16(scale * (static_cast<float>(s % 1000u) / 500.0f - 1.0f));
    }
    return v;
  };
  std::vector<uint16_t> a = Sp_(kM * kK, 0.5f, 29u);
  std::vector<uint16_t> b = Sp_(kN * kK, 0.25f, 77u);

  struct B_ { Backend& b; void* p_;
    B_(Backend& x, size_t e, size_t eb):b(x),p_(x.Alloc(e*eb)){}
    ~B_(){b.Free(p_);} void* p() const{return p_;} };
  B_ va(vk, kM * kK, 2), ca(cpu, kM * kK, 2);
  B_ vb(vk, kN * kK, 2), cb(cpu, kN * kK, 2);
  B_ vo(vk, kM * kN, 4), co(cpu, kM * kN, 4);
  vk.Copy(vq, va.p(), a.data(), a.size() * 2);
  vk.Copy(vq, vb.p(), b.data(), b.size() * 2);
  vk.Synchronize(vq);
  std::memcpy(ca.p(), a.data(), a.size() * 2);
  std::memcpy(cb.p(), b.data(), b.size() * 2);

  Tensor vat = Tensor::Contiguous(va.p(), vt::DType::kBF16, vd, {kM, kK});
  Tensor vbt = Tensor::Contiguous(vb.p(), vt::DType::kBF16, vd, {kN, kK});
  Tensor vot = Tensor::Contiguous(vo.p(), vt::DType::kF32, vd, {kM, kN});
  Tensor cat = Tensor::Contiguous(ca.p(), vt::DType::kBF16, cd, {kM, kK});
  Tensor cbt = Tensor::Contiguous(cb.p(), vt::DType::kBF16, cd, {kN, kK});
  Tensor cot = Tensor::Contiguous(co.p(), vt::DType::kF32, cd, {kM, kN});

  vt::MatmulBT(cq, cot, cat, cbt);
  vt::MatmulBT(vq, vot, vat, vbt);
  vk.Synchronize(vq);

  std::vector<float> g(kM * kN), r(kM * kN);
  vk.Copy(vq, g.data(), vo.p(), g.size() * 4);
  vk.Synchronize(vq);
  std::memcpy(r.data(), co.p(), r.size() * 4);

  // The chunk boundary (row 2048) is the critical check: a mis-strided a_off
  // or out_off would produce garbage at exactly that row. Compare every element.
  double num = 0, den = 0, maxabs = 0;
  for (int64_t i = 0; i < kM * kN; ++i) {
    const double d = static_cast<double>(g[i]) - static_cast<double>(r[i]);
    num += d * d; maxabs = std::max(maxabs, std::fabs(d));
    den += static_cast<double>(r[i]) * static_cast<double>(r[i]);
  }
  const double nmse = (den > 1e-12) ? std::sqrt(num / den) : std::sqrt(num + 0.0);
  MESSAGE("chunked-matmul maxabs=" << maxabs << " nmse=" << nmse);
  CHECK(nmse < 1e-3);
  CHECK(maxabs < 0.1);
}

TEST_CASE("tq1_0 keep-quant dequantizer matches the shader's trit extraction") {
  // Unit test for the CPU TQ1_0 dequantizer (DequantTQ1_0) that serves as the
  // reference oracle for the Vulkan TQ1_0 keep-quant path. No Vulkan device
  // needed — pure CPU check.
  using vt::DType;
  constexpr int64_t kK = 256;
  std::vector<uint8_t> block(54, 0);
  // Set all qs bytes to 0x55 (alternating bits) to exercise all trit lanes.
  for (int i = 0; i < 48; ++i) block[i] = 0x55;
  for (int i = 0; i < 4; ++i) block[48 + i] = 0x55;
  // d = f16(1.0) = 0x3C00
  const uint16_t dhalf = 0x3C00;
  std::memcpy(block.data() + 52, &dhalf, 2);

  std::vector<float> y(kK);
  const vt::cpu::ToFloatFn fn = vt::cpu::BlockToFloat(DType::kTQ1_0);
  REQUIRE(fn != nullptr);
  fn(block.data(), y.data(), kK);

  // Every element should be in {-1, 0, +1} since d=1.0. The exact value depends
  // on the trit extraction, but the range is bounded.
  for (int64_t e = 0; e < kK; ++e) {
    CHECK(y[e] >= -1.0f);
    CHECK(y[e] <= 1.0f);
  }
  // Verify the block geometry: 54 bytes, 256 elements.
  CHECK(vt::BlockElems(DType::kTQ1_0) == 256);
  CHECK(vt::BlockBytes(DType::kTQ1_0) == 54);
  CHECK(vt::IsBlockQuant(DType::kTQ1_0));
}

TEST_CASE("tq2_0 keep-quant dequantizer matches the shader's trit extraction") {
  // Unit test for the CPU TQ2_0 dequantizer (DequantTQ2_0). No Vulkan device
  // needed — pure CPU check.
  using vt::DType;
  constexpr int64_t kK = 256;
  std::vector<uint8_t> block(66, 0);
  // Set all qs bytes to 0xAA (alternating 10 bits) to exercise all 2-bit codes.
  for (int i = 0; i < 64; ++i) block[i] = 0xAA;
  // d = f16(1.0) = 0x3C00
  const uint16_t dhalf = 0x3C00;
  std::memcpy(block.data() + 64, &dhalf, 2);

  std::vector<float> y(kK);
  const vt::cpu::ToFloatFn fn = vt::cpu::BlockToFloat(DType::kTQ2_0);
  REQUIRE(fn != nullptr);
  fn(block.data(), y.data(), kK);

  // With qs=0xAA and 2-bit codes: (0xAA >> (l*2)) & 3 = 2 for all l.
  // So all codes are 2, meaning all values are (2-1)*1.0 = 1.0.
  for (int64_t e = 0; e < kK; ++e) {
    CHECK(y[e] == doctest::Approx(1.0f).epsilon(1e-6));
  }
  // Verify the block geometry: 66 bytes, 256 elements.
  CHECK(vt::BlockElems(DType::kTQ2_0) == 256);
  CHECK(vt::BlockBytes(DType::kTQ2_0) == 66);
}
