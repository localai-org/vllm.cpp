// Ported from: tests/kernels/attention/test_attention_selector.py @ pin
// 5559679229 — the executable spec of
// `vllm/v1/attention/backend.py::AttentionBackend.validate_configuration`
// (`:320-393`) and the `supports_*` predicates it collects (`:154-317`), as
// applied by `vllm/platforms/cuda.py::get_valid_backends` (`:359-394`) and
// `get_attn_backend_cls` (`:397-492`).
//
// Every upstream case in that file monkeypatches the device capability and then
// asserts a chosen backend NAME, so the whole file is CPU-runnable: the only
// thing it needs from a GPU is a number, and `FakeCudaPlatform` below takes that
// number as a constructor argument.
//
// WHAT THIS FILE DOES NOT MEASURE (issue #1332, and the reason that issue
// exists). These predicates describe what a backend CLAIMS. They cannot see what
// the shipped binary contains, and they cannot see which kernel actually runs:
// `dense_attn::AttnBlock` calls `vt::PagedAttention` unconditionally, so the
// name selected here dispatches nothing. Upstream's
// `FlashAttentionBackend.supports_compute_capability` is `>= (8,0)`; it passed on
// a GB10 (12,1) whose FA2 fatbin holds `sm_80` SASS and `compute_80` PTX alone,
// and every launch then failed a driver JIT. A green run of this file is
// necessary and NOT sufficient. See `.agents/specs/attn-validate-configuration.md`.
#include <doctest/doctest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/platforms/cuda_attn_priority.h"
#include "vllm/platforms/interface.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/registry.h"
#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"

using vllm::platforms::AttnSelectorConfig;
using vllm::platforms::DeviceCapability;
using vllm::platforms::Platform;
using vllm::platforms::ResidencyPolicy;
using vllm::v1::AttentionBackend;
using vllm::v1::MakeAttentionBackend;
using vllm::v1::SelectAttentionBackendName;
using vt::DeviceType;
using vt::DType;

namespace {

// The capability-injecting platform of the upstream test's
// `patch("vllm.platforms.current_platform", CudaPlatform())` plus
// `monkeypatch.setattr(torch.cuda, "get_device_capability", ...)`. It delegates
// to the REAL shared priority table so no list is hand-copied here.
class FakeCudaPlatform final : public Platform {
 public:
  explicit FakeCudaPlatform(int major, int minor) : cap_{major, minor} {}
  DeviceType device_type() const override { return DeviceType::kCUDA; }
  vt::Backend& backend() const override { return vt::GetBackend(DeviceType::kCPU); }
  DeviceCapability get_device_capability() const override { return cap_; }
  std::vector<DType> supported_dtypes() const override { return {DType::kBF16}; }
  ResidencyPolicy residency_policy() const override { return {}; }
  std::vector<std::string> get_attn_backend_priority(
      const AttnSelectorConfig& cfg) const override {
    return vllm::platforms::LookupAttnPriority(cap_.major, cfg);
  }

 private:
  DeviceCapability cap_;
};

// The request the gate models actually make on GB10: Qwen3-dense geometry,
// bf16, framework block size 16, plain causal decoder attention. Every case
// below starts from this and changes ONE field, so a refusal is attributable.
AttnSelectorConfig GateModelConfig() {
  AttnSelectorConfig cfg;
  cfg.head_size = 128;
  cfg.dtype = DType::kBF16;
  cfg.kv_cache_dtype = "auto";
  cfg.block_size = 16;
  return cfg;
}

}  // namespace

TEST_CASE("the gate-model request still selects FLASH_ATTN (behavior control)") {
  // THE control for every refusal below: a fully populated, valid config
  // resolves exactly what the runner resolves today. If this case ever goes red,
  // the capability layer has changed production selection, which is a behavior
  // change and not a capability port.
  FakeCudaPlatform sm121(12, 1);
  CHECK(SelectAttentionBackendName(sm121, "", GateModelConfig()) == "FLASH_ATTN");

  // And the all-defaults config, which is what every pre-existing call site
  // passes, is unchanged too.
  CHECK(SelectAttentionBackendName(sm121, "", AttnSelectorConfig{}) == "FLASH_ATTN");
}

TEST_CASE("test_flash_attn: an unsupported compute capability refuses FLASH_ATTN") {
  // test_attention_selector.py:277-280 — `get_device_capability -> (7, 5)` then
  // `assert backend.get_name() != "FLASH_ATTN"`. Upstream's predicate is
  // `capability >= DeviceCapability(8, 0)` (flash_attn.py:200-202). sm_75 is
  // Turing, the last generation before FlashAttention's floor.
  FakeCudaPlatform sm75(7, 5);
  CHECK_THROWS_AS(SelectAttentionBackendName(sm75, "", GateModelConfig()),
                  std::runtime_error);
}

TEST_CASE("test_flash_attn: an unsupported model dtype refuses FLASH_ATTN") {
  // test_attention_selector.py:285-287 (`torch.float8_e4m3fn`) and
  // test_fp32_fallback:242-247. `supported_dtypes` is `[float16, bfloat16]`
  // (backend.py:59, flash_attn.py:73). Upstream's fp32 request lands on
  // FLEX_ATTENTION; this tree registers no FLEX_ATTENTION, so the walk is
  // exhausted and refuses — which is the correct answer, not a fallback.
  FakeCudaPlatform sm121(12, 1);
  AttnSelectorConfig f32 = GateModelConfig();
  f32.dtype = DType::kF32;
  CHECK_THROWS_AS(SelectAttentionBackendName(sm121, "", f32), std::runtime_error);

  // HARNESS ADAPTATION: upstream's second parameter here is
  // `torch.float8_e4m3fn`, and `vt::DType` carries no fp8 member, so the second
  // "outside the supported list" value is `kI8`. The predicate under test is a
  // list membership either way, and keeping a second value proves the check is
  // the list rather than an `== kF32` special case.
  AttnSelectorConfig i8 = GateModelConfig();
  i8.dtype = DType::kI8;
  CHECK_THROWS_AS(SelectAttentionBackendName(sm121, "", i8), std::runtime_error);
}

TEST_CASE("test_flash_attn: an unsupported kv-cache dtype refuses FLASH_ATTN") {
  // test_attention_selector.py:289-291. FLASH_ATTN's
  // `supported_kv_cache_dtypes` is ["auto","float16","bfloat16","fp8","fp8_e4m3"]
  // (flash_attn.py:74-80), so `fp8_e5m2` is outside it.
  FakeCudaPlatform sm121(12, 1);
  AttnSelectorConfig cfg = GateModelConfig();
  cfg.kv_cache_dtype = "fp8_e5m2";
  CHECK_THROWS_AS(SelectAttentionBackendName(sm121, "", cfg), std::runtime_error);
}

TEST_CASE("test_flash_attn: an unsupported block size refuses FLASH_ATTN") {
  // test_attention_selector.py:293-296 (`block_size = 8`). FLASH_ATTN declares
  // `MultipleOf(16)` (flash_attn.py:82-84), and `supports_block_size` accepts any
  // MULTIPLE of a declared size (backend.py:184-192), so 32 is fine and 8 is not.
  FakeCudaPlatform sm121(12, 1);
  AttnSelectorConfig bad = GateModelConfig();
  bad.block_size = 8;
  CHECK_THROWS_AS(SelectAttentionBackendName(sm121, "", bad), std::runtime_error);

  AttnSelectorConfig ok = GateModelConfig();
  ok.block_size = 32;
  CHECK(SelectAttentionBackendName(sm121, "", ok) == "FLASH_ATTN");
}

TEST_CASE("test_flash_attn: an unsupported head size refuses FLASH_ATTN") {
  // test_attention_selector.py:313-315 (`head_size = 17`). Upstream:
  // `head_size % 8 != 0 -> False`, `<= 256 -> True` (flash_attn.py:170-178).
  FakeCudaPlatform sm121(12, 1);
  AttnSelectorConfig odd = GateModelConfig();
  odd.head_size = 17;
  CHECK_THROWS_AS(SelectAttentionBackendName(sm121, "", odd), std::runtime_error);

  // 264 is a multiple of 8 and ABOVE the 256 ceiling — the second half of the
  // predicate, which a `% 8` test alone would not reach. Upstream raises the
  // ceiling to 512 only when FlashAttention v4 resolves, and this tree ships FA2.
  AttnSelectorConfig big = GateModelConfig();
  big.head_size = 264;
  CHECK_THROWS_AS(SelectAttentionBackendName(sm121, "", big), std::runtime_error);

  // 256 itself is inside it.
  AttnSelectorConfig edge = GateModelConfig();
  edge.head_size = 256;
  CHECK(SelectAttentionBackendName(sm121, "", edge) == "FLASH_ATTN");
}

TEST_CASE("test_per_head_quant_scales_backend_selection: FA2 refuses per-head scales") {
  // test_attention_selector.py:369-427, the `("FLASH_ATTN", 2, False)` row:
  // FA2 does not support per-head quant scales. Upstream's predicate is
  // `get_flash_attn_version() >= 3` (flash_attn.py:120-123); this tree ships FA2
  // only, so the answer is upstream's own answer for FA2.
  FakeCudaPlatform sm121(12, 1);
  AttnSelectorConfig cfg = GateModelConfig();
  cfg.kv_cache_dtype = "fp8";
  cfg.use_per_head_quant_scales = true;
  CHECK_THROWS_AS(SelectAttentionBackendName(sm121, "", cfg), std::runtime_error);
}

TEST_CASE("test_non_causal_backend_selection: FLASH_ATTN serves a non-causal request") {
  // test_attention_selector.py:430-487, the two FLASH_ATTN rows: FlashAttention
  // supports non-causal (flash_attn.py:106-108), so a drafter's non-causal
  // request selects it, and so does an ordinary causal one.
  FakeCudaPlatform sm121(12, 1);
  AttnSelectorConfig non_causal = GateModelConfig();
  non_causal.use_non_causal = true;
  CHECK(SelectAttentionBackendName(sm121, "", non_causal) == "FLASH_ATTN");
}

TEST_CASE("a sliding-window request refuses a backend that does not declare it") {
  // backend.py:370-371 `has_sliding_window and not supports_sliding_window()`.
  // FLASH_ATTN declares it (flash_attn.py:98-100), so the positive control
  // selects; GDN_ATTN inherits the base `false`, so a platform preferring GDN
  // for a sliding-window request finds nothing valid.
  FakeCudaPlatform sm121(12, 1);
  AttnSelectorConfig sw = GateModelConfig();
  sw.has_sliding_window = true;
  CHECK(SelectAttentionBackendName(sm121, "", sw) == "FLASH_ATTN");

  class GdnOnlyPlatform final : public Platform {
   public:
    DeviceType device_type() const override { return DeviceType::kCPU; }
    vt::Backend& backend() const override { return vt::GetBackend(DeviceType::kCPU); }
    DeviceCapability get_device_capability() const override { return {}; }
    std::vector<DType> supported_dtypes() const override { return {DType::kBF16}; }
    ResidencyPolicy residency_policy() const override { return {}; }
    std::vector<std::string> get_attn_backend_priority(
        const AttnSelectorConfig& cfg) const override {
      (void)cfg;
      return {"GDN_ATTN"};
    }
  } gdn;
  CHECK(SelectAttentionBackendName(gdn, "", GateModelConfig()) == "GDN_ATTN");
  CHECK_THROWS_AS(SelectAttentionBackendName(gdn, "", sw), std::runtime_error);
}

TEST_CASE("the refusal names every failed predicate, not only the first") {
  // Upstream builds a LIST (`invalid_reasons`) and prints all of it
  // (cuda.py:416-420, :432-446). A message that collapsed to the first failure
  // would send a reader chasing one cause of four. Three violations at once:
  // head size, block size and kv-cache dtype.
  FakeCudaPlatform sm121(12, 1);
  AttnSelectorConfig bad = GateModelConfig();
  bad.head_size = 17;
  bad.block_size = 8;
  bad.kv_cache_dtype = "fp8_e5m2";

  std::string message;
  try {
    SelectAttentionBackendName(sm121, "", bad);
  } catch (const std::exception& e) {
    message = e.what();
  }
  REQUIRE_FALSE(message.empty());
  CHECK(message.find("head_size not supported") != std::string::npos);
  CHECK(message.find("block_size not supported") != std::string::npos);
  CHECK(message.find("kv_cache_dtype not supported") != std::string::npos);
}

TEST_CASE("an explicit override that fails validation is refused by reason") {
  // cuda.py:407-420 — a `selected_backend` is validated too, and the error names
  // the reasons rather than silently falling through to the priority walk.
  FakeCudaPlatform sm121(12, 1);
  AttnSelectorConfig odd = GateModelConfig();
  odd.head_size = 17;

  std::string message;
  try {
    SelectAttentionBackendName(sm121, "FLASH_ATTN", odd);
  } catch (const std::exception& e) {
    message = e.what();
  }
  REQUIRE_FALSE(message.empty());
  CHECK(message.find("FLASH_ATTN") != std::string::npos);
  CHECK(message.find("head_size not supported") != std::string::npos);
}

TEST_CASE("#1333: sm_100 prefers FLASH_ATTN for a non-causal request") {
  // cuda.py:144-163 @ pin 5559679229. The sm_100 non-MLA arm is guarded by
  // `and not use_non_causal`, because SM100f's non-causal cutlass path — the one
  // DFlash attention uses — is known-bad (`:145-147`). Causal keeps FlashInfer
  // first; non-causal falls to the `else` arm, FlashAttention first.
  FakeCudaPlatform sm100(10, 0);
  const std::vector<std::string> causal{"FLASHINFER", "FLASH_ATTN", "TRITON_ATTN",
                                        "FLEX_ATTENTION", "TURBOQUANT"};
  const std::vector<std::string> non_causal{"FLASH_ATTN", "FLASHINFER", "TRITON_ATTN",
                                            "FLEX_ATTENTION", "TURBOQUANT"};
  AttnSelectorConfig plain;
  CHECK(sm100.get_attn_backend_priority(plain) == causal);

  AttnSelectorConfig nc;
  nc.use_non_causal = true;
  CHECK(sm100.get_attn_backend_priority(nc) == non_causal);

  // sm_121 is in the `else` arm either way, so the guard moves nothing there.
  FakeCudaPlatform sm121(12, 1);
  CHECK(sm121.get_attn_backend_priority(plain) == non_causal);
  CHECK(sm121.get_attn_backend_priority(nc) == non_causal);
}

// ─── The predicate-level upstream cases, asked of the backend directly ───────

TEST_CASE("test_flash_attn_rejects_unhandled_kv_cache_dtypes") {
  // test_attention_selector.py:524-540, all six upstream parameters. The point
  // upstream makes in its own docstring: FlashAttentionBackend must not CLAIM
  // support for kv_cache dtypes it cannot handle. Our list is upstream's
  // (flash_attn.py:74-80), so the same six fall outside it.
  const std::unique_ptr<AttentionBackend> fa =
      MakeAttentionBackend(DeviceType::kCUDA, "FLASH_ATTN");
  REQUIRE(fa != nullptr);
  for (const char* name : {"fp8_e5m2", "fp8_ds_mla", "fp8_inc", "nvfp4",
                           "fp8_per_token_head", "int8_per_token_head"}) {
    CAPTURE(name);
    CHECK_FALSE(fa->supports_kv_cache_dtype(name));
  }
}

TEST_CASE("test_flash_attn_accepts_handled_fp8_variants") {
  // test_attention_selector.py:543-556, both upstream parameters: "fp8" (the
  // alias for fp8_e4m3fn) and "fp8_e4m3" are the two FlashAttention can handle.
  // Upstream needs an `is_xpu` monkeypatch to reach the answer through
  // fa_utils; our predicate is the plain list membership it resolves to.
  const std::unique_ptr<AttentionBackend> fa =
      MakeAttentionBackend(DeviceType::kCUDA, "FLASH_ATTN");
  REQUIRE(fa != nullptr);
  CHECK(fa->supports_kv_cache_dtype("fp8"));
  CHECK(fa->supports_kv_cache_dtype("fp8_e4m3"));
  // backend.py:169-170 — upstream's `None`, which our empty string models.
  CHECK(fa->supports_kv_cache_dtype(""));
}

TEST_CASE("validate_configuration returns an EMPTY list for a valid request") {
  // backend.py:339,393 — the empty list IS the "valid" answer, and the selector
  // reads emptiness rather than a bool. A predicate that returned a one-element
  // list of "" would select nothing while looking correct here, so the size is
  // asserted, not just the truthiness.
  const std::unique_ptr<AttentionBackend> fa =
      MakeAttentionBackend(DeviceType::kCUDA, "FLASH_ATTN");
  REQUIRE(fa != nullptr);
  const std::vector<std::string> reasons =
      fa->validate_configuration(GateModelConfig(), DeviceCapability{12, 1});
  CHECK(reasons.empty());
}

TEST_CASE("validate_configuration collects ONE reason per failed predicate") {
  // backend.py:339-393. Four independent violations produce four reasons, in
  // upstream's order, with upstream's strings. A short-circuiting implementation
  // returns one; a mis-ordered one fails the sequence.
  const std::unique_ptr<AttentionBackend> fa =
      MakeAttentionBackend(DeviceType::kCUDA, "FLASH_ATTN");
  REQUIRE(fa != nullptr);
  AttnSelectorConfig bad = GateModelConfig();
  bad.head_size = 17;             // :340-341
  bad.dtype = DType::kF32;        // :342-343
  bad.kv_cache_dtype = "nvfp4";   // :344-345
  bad.block_size = 8;             // :346-347
  const std::vector<std::string> reasons =
      fa->validate_configuration(bad, DeviceCapability{12, 1});
  const std::vector<std::string> expected{
      "head_size not supported", "dtype not supported",
      "kv_cache_dtype not supported", "block_size not supported"};
  CHECK(reasons == expected);
}

TEST_CASE("validate_configuration keys MLA and sparse on equality, both ways") {
  // backend.py:352-363 — the check is `use_mla != is_mla()`, not "backend is
  // MLA", so a dense backend asked for MLA and an MLA backend asked for dense
  // BOTH fail, with different strings. This is the pair the pre-#1332 selector
  // asked inline, so it is the behavior that must not have moved.
  const std::unique_ptr<AttentionBackend> fa =
      MakeAttentionBackend(DeviceType::kCUDA, "FLASH_ATTN");
  const std::unique_ptr<AttentionBackend> mla =
      MakeAttentionBackend(DeviceType::kCUDA, "TRITON_MLA");
  REQUIRE(fa != nullptr);
  REQUIRE(mla != nullptr);

  AttnSelectorConfig want_mla = GateModelConfig();
  want_mla.use_mla = true;
  const std::vector<std::string> fa_reasons =
      fa->validate_configuration(want_mla, DeviceCapability{12, 1});
  CHECK(fa_reasons == std::vector<std::string>{"MLA not supported"});

  // TRITON_MLA asked for a DENSE request says the opposite thing. Its head size
  // and block size are the MLA geometry, so the request is otherwise valid.
  AttnSelectorConfig dense = GateModelConfig();
  dense.head_size = 576;
  const std::vector<std::string> mla_reasons =
      mla->validate_configuration(dense, DeviceCapability{12, 1});
  CHECK(mla_reasons == std::vector<std::string>{"non-MLA not supported"});

  AttnSelectorConfig sparse = want_mla;
  sparse.head_size = 576;
  sparse.use_sparse = true;
  const std::vector<std::string> sparse_reasons =
      mla->validate_configuration(sparse, DeviceCapability{12, 1});
  CHECK(sparse_reasons == std::vector<std::string>{"sparse not supported"});
}

TEST_CASE("the compute-capability predicate is skipped when there is none") {
  // backend.py:366-367 applies `supports_compute_capability` unconditionally,
  // and CudaPlatform.get_attn_backend_cls guarantees a capability before calling
  // (cuda.py:403-404) while CpuPlatform never calls it at all (cpu.py:75-87).
  // Our selector is shared, and this tree registers FLASH_ATTN for kCPU too, so
  // a default-constructed (absent) DeviceCapability must NOT refuse it — that
  // would refuse the CPU tier on a rule about NVIDIA compute capability.
  const std::unique_ptr<AttentionBackend> fa =
      MakeAttentionBackend(DeviceType::kCPU, "FLASH_ATTN");
  REQUIRE(fa != nullptr);
  CHECK_FALSE(DeviceCapability{}.present());
  CHECK(fa->validate_configuration(GateModelConfig(), DeviceCapability{}).empty());

  // And a capability that IS present is still enforced.
  const std::vector<std::string> turing =
      fa->validate_configuration(GateModelConfig(), DeviceCapability{7, 5});
  CHECK(turing == std::vector<std::string>{"compute capability not supported"});
  // 8.0 is the floor itself, so it passes (flash_attn.py:200-202 is `>=`).
  CHECK(fa->validate_configuration(GateModelConfig(), DeviceCapability{8, 0}).empty());
}

TEST_CASE("attn_type is upstream's STRING, and the base serves decoder alone") {
  // backend.py:291-298 + :38-46. FLASH_ATTN overrides to all four
  // (flash_attn.py:110-118); GDN_ATTN inherits the base, which is decoder-only.
  CHECK(std::string(vllm::v1::AttentionTypeName(vllm::v1::AttentionType::kDecoder)) ==
        "decoder");
  CHECK(std::string(vllm::v1::AttentionTypeName(
            vllm::v1::AttentionType::kEncoderOnly)) == "encoder_only");

  const std::unique_ptr<AttentionBackend> fa =
      MakeAttentionBackend(DeviceType::kCUDA, "FLASH_ATTN");
  const std::unique_ptr<AttentionBackend> gdn =
      MakeAttentionBackend(DeviceType::kCPU, "GDN_ATTN");
  REQUIRE(fa != nullptr);
  REQUIRE(gdn != nullptr);
  for (const char* type : {"decoder", "encoder", "encoder_only", "encoder_decoder"}) {
    CAPTURE(type);
    CHECK(fa->supports_attn_type(type));
  }
  CHECK(gdn->supports_attn_type("decoder"));
  CHECK_FALSE(gdn->supports_attn_type("encoder_only"));

  AttnSelectorConfig encoder = GateModelConfig();
  encoder.attn_type = "encoder_only";
  const std::vector<std::string> reasons =
      gdn->validate_configuration(encoder, DeviceCapability{});
  CHECK(reasons == std::vector<std::string>{"attention type encoder_only not supported"});
}

TEST_CASE("supports_combination is asked, and FLASH_ATTN refuses a pre-Hopper sink") {
  // flash_attn.py:204-218 — the cross-field rule no per-field predicate can
  // express: sinks need compute capability >= 9.0. supports_sink() is false in
  // this tree (no sink kernel), so the sink flag is refused by the per-field
  // predicate first and BOTH reasons appear — which is the list shape, again.
  const std::unique_ptr<AttentionBackend> fa =
      MakeAttentionBackend(DeviceType::kCUDA, "FLASH_ATTN");
  REQUIRE(fa != nullptr);
  AttnSelectorConfig sink = GateModelConfig();
  sink.has_sink = true;
  const std::vector<std::string> reasons =
      fa->validate_configuration(sink, DeviceCapability{8, 6});
  const std::vector<std::string> expected{
      "attention sinks not supported",
      "sink not supported on compute capability < 9.0"};
  CHECK(reasons == expected);

  // On Blackwell the combination rule is silent, so only the per-field one
  // remains — proving the combination hook is genuinely consulted rather than
  // being a constant.
  const std::vector<std::string> blackwell =
      fa->validate_configuration(sink, DeviceCapability{10, 0});
  CHECK(blackwell == std::vector<std::string>{"attention sinks not supported"});
}

TEST_CASE("block_size accepts any MULTIPLE of a declared size (hybrid blocks)") {
  // backend.py:184-192 — the rule is `block_size % supported == 0`, NOT
  // equality, because the framework block size only has to be a multiple of the
  // kernel's requirement. A `== 16` implementation passes 16 and fails 32.
  const std::unique_ptr<AttentionBackend> fa =
      MakeAttentionBackend(DeviceType::kCUDA, "FLASH_ATTN");
  REQUIRE(fa != nullptr);
  CHECK(fa->supports_block_size(16));
  CHECK(fa->supports_block_size(32));
  CHECK(fa->supports_block_size(64));
  CHECK_FALSE(fa->supports_block_size(8));
  CHECK_FALSE(fa->supports_block_size(24));
  // backend.py:177-178 — upstream's `None`.
  CHECK(fa->supports_block_size(0));

  // TRITON_MLA declares the same MultipleOf(16) through the LIST, and it is now
  // the base virtual rather than a static that hid it: the answer must be the
  // same through an AttentionBackend& as it was through the concrete type.
  const std::unique_ptr<AttentionBackend> mla =
      MakeAttentionBackend(DeviceType::kCUDA, "TRITON_MLA");
  REQUIRE(mla != nullptr);
  CHECK(mla->supports_block_size(16));
  CHECK_FALSE(mla->supports_block_size(24));

  // ROCM_ATTN declares MultipleOf(16) upstream (rocm_attn.py:181-190), for the
  // same reason its get_kv_cache_shape refuses block_size % 16 != 0
  // (rocm_attn.py:249-256): the native paged-attn kernel is LDS-bound to 16/32.
  // Declaring it is what lets the REGISTRY refuse an unsupported block size
  // while selecting a backend, instead of a selected backend throwing at
  // allocation time. Without this the two halves disagree -- ROCM_ATTN
  // advertises every block size and then refuses most of them (#1608).
  const std::unique_ptr<AttentionBackend> rocm =
      MakeAttentionBackend(DeviceType::kROCM, "ROCM_ATTN");
  REQUIRE(rocm != nullptr);
  CHECK(rocm->supports_block_size(16));
  CHECK(rocm->supports_block_size(32));
  CHECK_FALSE(rocm->supports_block_size(8));
  CHECK_FALSE(rocm->supports_block_size(24));
  CHECK(rocm->supports_block_size(0));
}

TEST_CASE("legacy CUDA selects portable attention without changing FLASH_ATTN") {
  for (const auto cap : {DeviceCapability{6, 0}, DeviceCapability{6, 1}, DeviceCapability{7, 0}}) {
    FakeCudaPlatform platform(cap.major, cap.minor);
    CHECK(SelectAttentionBackendName(platform, "", GateModelConfig()) == "CUDA_LEGACY");
    CHECK_THROWS_AS(SelectAttentionBackendName(platform, "FLASH_ATTN", GateModelConfig()),
                    std::invalid_argument);
  }
  const auto legacy = MakeAttentionBackend(DeviceType::kCUDA, "CUDA_LEGACY");
  for (const auto cap : {DeviceCapability{5, 2}, DeviceCapability{6, 2}, DeviceCapability{7, 5}, DeviceCapability{8, 0}, DeviceCapability{12, 1}})
    CHECK_FALSE(legacy->supports_compute_capability(cap));
  for (const auto dtype : {DType::kF16, DType::kI8}) CHECK_FALSE(legacy->supports_dtype(dtype));
  for (const auto dtype : {DType::kF32, DType::kBF16}) CHECK(legacy->supports_dtype(dtype));
  for (const auto name : {"fp8", "float16"}) CHECK_FALSE(legacy->supports_kv_cache_dtype(name));
  CHECK_FALSE(legacy->supports_attn_type("encoder"));
  CHECK_FALSE(legacy->supports_sink());
  CHECK_FALSE(legacy->is_mla());
  CHECK_FALSE(legacy->is_sparse());
  CHECK(legacy->get_kv_cache_shape(4, 16, 1, 64) == std::vector<int64_t>{4, 2, 16, 1, 64});
}

TEST_CASE("legacy CUDA pins head and block geometry through the registered selector") {
  FakeCudaPlatform pascal(6, 1);
  for (const int head : {8, 16, 64, 128, 192, 256}) {
    CAPTURE(head);
    auto cfg = GateModelConfig();
    cfg.head_size = head;
    CHECK(SelectAttentionBackendName(pascal, "CUDA_LEGACY", cfg) == "CUDA_LEGACY");
  }
  for (const int head : {-8, 0, 7, 17, 255, 264}) {
    CAPTURE(head);
    auto cfg = GateModelConfig();
    cfg.head_size = head;
    CHECK_THROWS_AS(SelectAttentionBackendName(pascal, "CUDA_LEGACY", cfg), std::invalid_argument);
  }
  // Match the base predicate's MultipleOf(16) and zero-as-unspecified rules.
  for (const int block : {0, 16, 32, 64}) {
    CAPTURE(block);
    auto cfg = GateModelConfig();
    cfg.block_size = block;
    CHECK(SelectAttentionBackendName(pascal, "CUDA_LEGACY", cfg) == "CUDA_LEGACY");
  }
  for (const int block : {8, 24}) {
    CAPTURE(block);
    auto cfg = GateModelConfig();
    cfg.block_size = block;
    CHECK_THROWS_AS(SelectAttentionBackendName(pascal, "CUDA_LEGACY", cfg), std::invalid_argument);
  }
  const auto legacy = MakeAttentionBackend(DeviceType::kCUDA, "CUDA_LEGACY");
  CHECK(legacy->get_name() == "CUDA_LEGACY");
  CHECK(legacy->get_kv_cache_shape(4, 32, 1, 256) == std::vector<int64_t>{4, 2, 32, 1, 256});
  CHECK_THROWS_AS(legacy->get_kv_cache_shape(4, 24, 1, 256), std::invalid_argument);
}

TEST_CASE("legacy CUDA pins supported storage through the registered selector") {
  FakeCudaPlatform volta(7, 0);
  for (const auto dtype : {DType::kBF16, DType::kF32}) {
    for (const auto kv : {"", "auto", "bfloat16"}) {
      CAPTURE(kv);
      auto cfg = GateModelConfig();
      cfg.dtype = dtype;
      cfg.kv_cache_dtype = kv;
      CHECK(SelectAttentionBackendName(volta, "CUDA_LEGACY", cfg) == "CUDA_LEGACY");
    }
  }
  for (const auto kv : {"float16", "fp8", "fp8_e4m3", "fp8_e5m2"}) {
    CAPTURE(kv);
    auto cfg = GateModelConfig();
    cfg.kv_cache_dtype = kv;
    CHECK_THROWS_AS(SelectAttentionBackendName(volta, "CUDA_LEGACY", cfg), std::invalid_argument);
  }
}

TEST_CASE("legacy CUDA refuses unsupported request modes through the registered selector") {
  FakeCudaPlatform pascal(6, 1);
  using Flag = bool AttnSelectorConfig::*;
  const Flag unsupported[] = {
      &AttnSelectorConfig::use_mla, &AttnSelectorConfig::use_sparse,
      &AttnSelectorConfig::has_sink, &AttnSelectorConfig::use_mm_prefix,
      &AttnSelectorConfig::use_per_head_quant_scales,
      &AttnSelectorConfig::has_sliding_window, &AttnSelectorConfig::use_non_causal,
      &AttnSelectorConfig::use_batch_invariant, &AttnSelectorConfig::use_kv_connector,
      &AttnSelectorConfig::use_pcp};
  for (const auto flag : unsupported) {
    auto cfg = GateModelConfig();
    cfg.*flag = true;
    CHECK_THROWS_AS(SelectAttentionBackendName(pascal, "CUDA_LEGACY", cfg), std::invalid_argument);
  }
  for (const auto type : {"encoder", "encoder_only", "encoder_decoder"}) {
    auto cfg = GateModelConfig();
    cfg.attn_type = type;
    CHECK_THROWS_AS(SelectAttentionBackendName(pascal, "CUDA_LEGACY", cfg), std::invalid_argument);
  }
}
