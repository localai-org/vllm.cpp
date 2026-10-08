// Kolibri-1 Tenstorrent B2b-i first slice (MODEL-TEXT-kolibri-1-tenstorrent,
// spec .agents/specs/kolibri-tt.md ### B2 scope — B2b addendum, slice i;
// issue ISSUE-LOCAL-01M4D2KFZ8H7B39XE75XYE0AYY).
//
// The B2b-i bring-up's gate TU. Two halves:
//
//  - HOST-SIDE (runs under ctest with NO card and NO checkpoint mount):
//    the resident staging plan (kolibri1_tt.h, PlanKolibri1TTResidentStaging)
//    over the tiny synthetic fixture and over the REAL checkpoint manifest
//    (kolibri1_manifest.inc) — dtype decisions, per-component byte
//    accounting against the spec's byte-math table, the refusals, and the
//    property the slice exists for: the resident slice FITS one P150
//    (~3.09 GiB) where the full model refuses (~73.6 GiB, wave A).
//
//  - DEVICE LEG (runs only with a Blackhole card AND
//    VT_KOLIBRI1_TT_B2I_MODEL=<real checkpoint dir>; operator-run under the
//    GPU lock, raw logs under /tmp, quoted in
//    docs/bench-evidence/kolibri1-tt-b2i-smoke-<date>.md): load the REAL
//    checkpoint through the production registry path, stage the resident
//    slice on device (fp8-block projections native FP8_E4M3 row-major,
//    packed bytes verbatim; router/norms/embed/head bf16; bias f32),
//    verify EVERY staged operand by readback, and run ONE verified device
//    op end-to-end — the embedding lookup + the first resident projection
//    (layer 0 q_proj) — against the CPU row's output for the same input.
//
//    The comparison method, stated before running: the embedding gather
//    must be BIT-EXACT (a row gather, no arithmetic). The q_proj GEMM
//    consumes the bf16 dequant of the STAGED fp8 bytes (the CPU row's
//    documented R1 disposition: dequant-to-bf16, then a plain bf16 GEMM —
//    the device fp8-block GEMM is the B2b compute wave's, not this
//    slice's); both sides consume identical bf16 operands. The CPU side
//    accumulates in f32 with a single rounding to bf16 at the store (the
//    CPU row's contract). The device side is ttnn's matmul, whose MEASURED
//    error for this tiny M=6 fallback shape (3.0 s for 94 MFLOP — not the
//    tensor-core path) is bf16-unit-roundoff scale relative to the
//    term-magnitude sum sum_k |a_ik w_jk|: 2026-10-08 max_abs 2.441e-3
//    against max terms 1.49, i.e. <= 2^-9.2 per unit terms — NOT
//    f32-accumulation-order scale (<= 2(K-1)2^-24 * terms = 4.5e-4 worst),
//    so an earlier 2-bf16-ulp premise was falsified by that measurement.
//    The envelope, stated before running: per element,
//    |dev - cpu| <= 8 * 2^-8 * sum_k |a_ik w_jk| + max(ulp(|cpu|),
//    ulp(|dev|)) — at most 8 bf16 unit roundoffs per unit term-magnitude
//    sum plus the store rounding, 5.8x above the measured worst
//    per-element ratio (max_err_ratio = 1.375 on 2026-10-08).
//    The CHECK asserts it per element and reports the bit-exact fraction,
//    max ulp distance, max abs/rel diff, and the max per-element error
//    ratio. The staged fp8 bytes themselves are verified byte-exact
//    against the packed bytes (raw readback, not a dtype pivot — the
//    ttnn host FP8->F32 decode flushes subnormals to zero).
#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/kolibri1_fp8_dequant.h"
#include "vllm/model_executor/models/kolibri1_tt.h"
#include "vllm/model_executor/models/kolibri1_weights.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/fp8_kv.h"  // vt::F8E4M3ToF32 (the scalar reference decode)
#include "vt/ops.h"
#include "vt/tenstorrent/tenstorrent_device.h"
#include "vt/tensor.h"

#include "kolibri1_manifest.inc"

#ifndef KOLIBRI1_GOLDENS
#define KOLIBRI1_GOLDENS "kolibri1_goldens.json"
#endif

using namespace vllm;  // NOLINT

namespace {

// ---- the synthetic tiny checkpoint (the test_kolibri1_tt.cpp pattern) ----

struct FixtureTensor {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (const int64_t d : shape) n *= d;
  return n;
}

int64_t CDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

std::string U64Le(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) s[i] = static_cast<char>((v >> (8 * i)) & 0xff);
  return s;
}

std::string BuildSafetensors(const std::vector<FixtureTensor>& tensors) {
  nlohmann::json header = nlohmann::json::object();
  std::string payload;
  for (const FixtureTensor& t : tensors) {
    const size_t begin = payload.size();
    payload.append(reinterpret_cast<const char*>(t.bytes.data()),
                   t.bytes.size());
    nlohmann::json entry = nlohmann::json::object();
    entry["dtype"] = t.dtype;
    entry["shape"] = t.shape;
    entry["data_offsets"] = nlohmann::json::array({begin, payload.size()});
    header[t.name] = std::move(entry);
  }
  const std::string head = header.dump();
  return U64Le(head.size()) + head + payload;
}

class TempCheckpoint {
 public:
  explicit TempCheckpoint(const std::vector<FixtureTensor>& tensors) {
    static std::atomic<uint64_t> counter{0};
    static const uint64_t nonce = [] {
      std::random_device rd;
      return (static_cast<uint64_t>(rd()) << 32) ^ rd();
    }();
    dir_ = std::filesystem::temp_directory_path() /
           ("vllm_kolibri1_tt_b2i_" + std::to_string(nonce) + "_" +
            std::to_string(counter.fetch_add(1)));
    std::filesystem::create_directories(dir_);
    path_ = dir_ / "model.safetensors";
    const std::string bytes = BuildSafetensors(tensors);
    std::ofstream out(path_, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("failed to write fixture checkpoint");
  }
  ~TempCheckpoint() {
    std::error_code ignored;
    std::filesystem::remove_all(dir_, ignored);
  }
  TempCheckpoint(const TempCheckpoint&) = delete;
  TempCheckpoint& operator=(const TempCheckpoint&) = delete;
  std::string path() const { return path_.string(); }

 private:
  std::filesystem::path dir_;
  std::filesystem::path path_;
};

std::vector<uint8_t> Fp8Bytes(const std::vector<int64_t>& shape) {
  const size_t n = static_cast<size_t>(Numel(shape));
  std::vector<uint8_t> bytes(n);
  for (size_t i = 0; i < n; ++i) bytes[i] = static_cast<uint8_t>((i * 7) & 0x7f);
  return bytes;
}

std::vector<uint8_t> Bf16Filled(const std::vector<int64_t>& shape,
                                uint16_t pattern) {
  std::vector<uint8_t> bytes(static_cast<size_t>(Numel(shape)) * 2);
  for (size_t i = 0; i < bytes.size(); i += 2) {
    bytes[i] = static_cast<uint8_t>(pattern & 0xff);
    bytes[i + 1] = static_cast<uint8_t>(pattern >> 8);
  }
  return bytes;
}

void AppendProjection(std::vector<FixtureTensor>& out, const std::string& proj,
                      int64_t n, int64_t k, bool fp8 = true) {
  if (!fp8) {
    out.push_back({proj + ".weight", "BF16", {n, k},
                   Bf16Filled({n, k}, 0x3F80)});
    return;
  }
  out.push_back({proj + ".weight", "F8_E4M3", {n, k}, Fp8Bytes({n, k})});
  const std::vector<int64_t> sshape = {CDiv(n, 128), CDiv(k, 128)};
  out.push_back({proj + ".weight_scale_inv", "BF16", sshape,
                 Bf16Filled(sshape, 0x3E00)});  // 0.125 exactly
}

// Tiny kolibri1 geometry: 2 layers (swa, full), hidden 64, 4 q heads /
// 2 kv heads, head_dim 16, 4 experts, intermediate 32.
struct TinyShape {
  int64_t hidden = 64;
  int64_t vocab = 32;
  int64_t heads = 4;
  int64_t kv_heads = 2;
  int64_t head_dim = 16;
  int64_t experts = 4;
  int64_t inter = 32;
};

HfConfig MakeTinyConfig() {
  HfConfig config;
  config.model_type = "kolibri1";
  config.architectures = {"Kolibri1ForCausalLM"};
  config.hidden_size = 64;
  config.num_hidden_layers = 2;
  config.vocab_size = 32;
  config.num_attention_heads = 4;
  config.num_key_value_heads = 2;
  config.head_dim = 16;

  nlohmann::json j;
  j["head_dim"] = 16;
  j["sliding_window"] = 8;
  j["use_sliding_window"] = true;
  j["rope_theta"] = 10000.0;
  j["num_experts"] = 4;
  j["num_experts_per_tok"] = 2;
  j["moe_intermediate_size"] = 32;
  j["shared_expert_intermediate_size"] = 32;
  j["norm_topk_prob"] = false;
  j["rms_norm_eps"] = 1e-6;
  j["hidden_act"] = "silu";
  j["tie_word_embeddings"] = false;
  j["layer_types"] =
      nlohmann::json::array({"sliding_attention", "full_attention"});

  nlohmann::json quant;
  quant["quant_method"] = "fp8";
  quant["activation_scheme"] = "dynamic";
  quant["weight_block_size"] = nlohmann::json::array({128, 128});
  j["quantization_config"] = quant;

  config.raw = j;
  return config;
}

std::vector<FixtureTensor> TinyFixture(const TinyShape& s = {}) {
  std::vector<FixtureTensor> t;
  t.push_back({"model.embed_tokens.weight", "BF16", {s.vocab, s.hidden},
               Bf16Filled({s.vocab, s.hidden}, 0x3F80)});
  t.push_back({"lm_head.weight", "BF16", {s.vocab, s.hidden},
               Bf16Filled({s.vocab, s.hidden}, 0x3F80)});
  t.push_back({"model.norm.weight", "BF16", {s.hidden},
               Bf16Filled({s.hidden}, 0x3F80)});
  for (int64_t l = 0; l < 2; ++l) {
    const std::string base = "model.layers." + std::to_string(l) + ".";
    for (const char* norm : {"input_layernorm", "post_attn_norm",
                             "post_attention_layernorm", "post_ffn_norm"}) {
      t.push_back({base + norm + ".weight", "BF16", {s.hidden},
                   Bf16Filled({s.hidden}, 0x3F80)});
    }
    t.push_back({base + "self_attn.q_norm.weight", "BF16", {s.head_dim},
                 Bf16Filled({s.head_dim}, 0x3F80)});
    t.push_back({base + "self_attn.k_norm.weight", "BF16", {s.head_dim},
                 Bf16Filled({s.head_dim}, 0x3F80)});
    AppendProjection(t, base + "self_attn.q_proj", s.heads * s.head_dim,
                     s.hidden);
    AppendProjection(t, base + "self_attn.k_proj", s.kv_heads * s.head_dim,
                     s.hidden);
    AppendProjection(t, base + "self_attn.v_proj", s.kv_heads * s.head_dim,
                     s.hidden);
    AppendProjection(t, base + "self_attn.o_proj", s.hidden,
                     s.heads * s.head_dim);
    AppendProjection(t, base + "mlp.gate", s.experts, s.hidden,
                     /*fp8=*/false);
    t.push_back({base + "moe.router.expert_bias", "BF16", {s.experts},
                 Bf16Filled({s.experts}, 0x3F80)});
    for (int64_t e = 0; e < s.experts; ++e) {
      const std::string expert = base + "mlp.experts." + std::to_string(e);
      AppendProjection(t, expert + ".gate_proj", s.inter, s.hidden);
      AppendProjection(t, expert + ".up_proj", s.inter, s.hidden);
      AppendProjection(t, expert + ".down_proj", s.hidden, s.inter);
    }
    AppendProjection(t, base + "mlp.shared_experts.gate_proj", s.inter,
                     s.hidden);
    AppendProjection(t, base + "mlp.shared_experts.up_proj", s.inter,
                     s.hidden);
    AppendProjection(t, base + "mlp.shared_experts.down_proj", s.hidden,
                     s.inter);
  }
  return t;
}

Kolibri1Weights LoadTiny(const TempCheckpoint& ckpt, const HfConfig& config) {
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(ckpt.path()));
  return vllm::LoadKolibri1Weights(shards, config);
}

std::string ResidentFailure(const Kolibri1Weights& w,
                            const Kolibri1TTStagingOptions& o) {
  try {
    const Kolibri1TTResidentStagingPlan p = PlanKolibri1TTResidentStaging(w, o);
    (void)p;
    return "";
  } catch (const std::exception& e) {
    return e.what();
  }
}

// ---- real-manifest byte math (the resident slice, from measured data) ----

int64_t ManifestBytes(const vllm_test::Kolibri1ManifestTensor& t) {
  int64_t elems = 1;
  for (int i = 0; i < t.rank; ++i) elems *= t.shape[i];
  const std::string d = t.dtype;
  if (d == "BF16") return elems * 2;
  if (d == "F8_E4M3") return elems;   // one fp8-e4m3 byte per element
  if (d == "F32") return elems * 4;
  throw std::runtime_error(std::string("unknown manifest dtype ") + d);
}

// Classifies every NON-EXPERT manifest tensor into the resident
// components, with the STAGED dtype (the router bias stages f32 — the
// loader widens the on-disk bf16 losslessly).
struct ManifestResidentBytes {
  int64_t attention_fp8 = 0;
  int64_t attention_scale = 0;
  int64_t shared_fp8 = 0;
  int64_t shared_scale = 0;
  int64_t router_gate = 0;
  int64_t router_bias = 0;  // staged f32
  int64_t norms = 0;
  int64_t embed_head = 0;
  int64_t total = 0;
  int64_t expert_tensors = 0;
};

ManifestResidentBytes ManifestResident() {
  ManifestResidentBytes r;
  for (const auto& t : vllm_test::kKolibri1Tensors) {
    const std::string name = t.name;
    if (name.find("mlp.experts.") != std::string::npos) {
      ++r.expert_tensors;
      continue;
    }
    int64_t bytes = ManifestBytes(t);
    if (name == "model.embed_tokens.weight" || name == "lm_head.weight") {
      r.embed_head += bytes;
    } else if (name == "model.norm.weight" ||
               name.find("input_layernorm.weight") != std::string::npos ||
               name.find("post_attn_norm.weight") != std::string::npos ||
               name.find("post_attention_layernorm.weight") !=
                   std::string::npos ||
               name.find("post_ffn_norm.weight") != std::string::npos ||
               name.find("self_attn.q_norm.weight") != std::string::npos ||
               name.find("self_attn.k_norm.weight") != std::string::npos) {
      r.norms += bytes;
    } else if (name.find(".self_attn.") != std::string::npos &&
               name.find("weight_scale_inv") == std::string::npos) {
      r.attention_fp8 += bytes;
    } else if (name.find(".self_attn.") != std::string::npos) {
      r.attention_scale += bytes;
    } else if (name.find(".mlp.gate.weight") != std::string::npos) {
      r.router_gate += bytes;
    } else if (name.find(".moe.router.expert_bias") != std::string::npos) {
      int64_t elems = 1;
      for (int i = 0; i < t.rank; ++i) elems *= t.shape[i];
      r.router_bias += elems * 4;  // staged f32 (the loader widens)
    } else if (name.find(".mlp.shared_experts.") != std::string::npos &&
               name.find("weight_scale_inv") == std::string::npos) {
      r.shared_fp8 += bytes;
    } else if (name.find(".mlp.shared_experts.") != std::string::npos) {
      r.shared_scale += bytes;
    } else {
      throw std::runtime_error("unclassified resident manifest tensor " +
                               name);
    }
  }
  r.total = r.attention_fp8 + r.attention_scale + r.shared_fp8 +
            r.shared_scale + r.router_gate + r.router_bias + r.norms +
            r.embed_head;
  return r;
}

double NowSec() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace

// ---- Host-side: the resident staging plan over the tiny fixture ----

TEST_CASE("kolibri1 TT B2b-i: resident plan dtype decisions and byte "
          "accounting (tiny)") {
  const HfConfig config = MakeTinyConfig();
  TempCheckpoint ckpt(TinyFixture());
  const Kolibri1Weights w = LoadTiny(ckpt, config);

  const Kolibri1TTResidentStagingPlan plan = PlanKolibri1TTResidentStaging(w);
  CHECK(plan.full_layers == 1);
  CHECK(plan.swa_layers == 1);

  int fp8_tensors = 0;
  int bf16_tensors = 0;
  int f32_tensors = 0;
  int expert_tensors = 0;
  int64_t sum = 0;
  for (const auto& t : plan.tensors) {
    if (t.name.find("mlp.experts.") != std::string::npos) ++expert_tensors;
    sum += t.bytes + t.scale_bytes;
    CHECK(t.host != nullptr);
    switch (t.dtype) {
      case Kolibri1TTDType::kFp8E4M3:
        // n*k fp8 bytes plus an f32 grid of cdiv(n,128)*cdiv(k,128)*4.
        CHECK(t.bytes == t.rows * t.cols);
        CHECK(t.scale_bytes > 0);
        CHECK(t.scale_host != nullptr);
        CHECK(t.scale_rows == CDiv(t.rows, 128));
        CHECK(t.scale_cols == CDiv(t.cols, 128));
        ++fp8_tensors;
        break;
      case Kolibri1TTDType::kBf16:
        CHECK(t.bytes % 2 == 0);
        CHECK(t.scale_bytes == 0);
        CHECK(t.scale_host == nullptr);
        ++bf16_tensors;
        break;
      case Kolibri1TTDType::kF32:
        CHECK(t.scale_bytes == 0);
        ++f32_tensors;
        break;
    }
  }
  // The routed-expert tier is ABSENT: B2b-i carries no expert slots.
  CHECK(expert_tensors == 0);
  CHECK(sum == plan.total_bytes);
  CHECK(plan.total_bytes == plan.components.total_bytes);
  CHECK(plan.fp8_bytes + plan.scale_bytes + plan.bf16_bytes +
            plan.f32_bytes ==
        plan.total_bytes);

  // Independent fixture math (NOT recomputed from the plan): per layer
  // attention fp8 = q[64,64] + k[32,64] + v[32,64] + o[64,64] = 12,288 B
  // + 4 one-element f32 grids = 16 B; shared fp8 = 3 x [32,64]/[64,32] =
  // 6,144 B + 3 grids = 12 B; router gate [4,64] bf16 = 512 B + bias
  // [4] f32 = 16 B; norms = 4 x [64] bf16 + 2 x [16] bf16 = 576 B.
  const Kolibri1TTResidentComponentBytes& c = plan.components;
  CHECK(c.attention_fp8_bytes == 2 * 12288);
  CHECK(c.attention_scale_bytes == 2 * 16);
  CHECK(c.shared_expert_fp8_bytes == 2 * 6144);
  CHECK(c.shared_expert_scale_bytes == 2 * 12);
  CHECK(c.router_gate_bytes == 2 * 512);
  CHECK(c.router_bias_bytes == 2 * 16);
  CHECK(c.norm_bytes == 2 * 576 + 128);  // + the final norm [64] bf16
  CHECK(c.embed_head_bytes == 2 * 32 * 64 * 2);
  CHECK(plan.total_bytes == 47448);

  // 2 layers x (4 attention + 3 shared) fp8 projections; bf16 modules:
  // embed + head + final norm + per layer (4 sandwich + 2 qk + 1 router
  // gate); f32: the router bias per layer.
  CHECK(fp8_tensors == 2 * (4 + 3));
  CHECK(bf16_tensors == 3 + 2 * 7);
  CHECK(f32_tensors == 2);
  CHECK(plan.tensor_count == 33);
}

TEST_CASE("kolibri1 TT B2b-i: the resident slice fits one P150 where the "
          "full model refuses") {
  // Tiny: the resident plan must NOT fire the single-device full-model
  // refusal — it plans the resident slice only.
  const HfConfig config = MakeTinyConfig();
  TempCheckpoint ckpt(TinyFixture());
  const Kolibri1Weights w = LoadTiny(ckpt, config);
  const Kolibri1TTResidentStagingPlan resident =
      PlanKolibri1TTResidentStaging(w);
  CHECK(resident.total_bytes < (int64_t(32) << 30));

  // Real manifest: the resident slice (~3.09 GiB) fits one P150 while the
  // FULL model (~73.3 GiB on disk) does not — the wave-A single-device
  // full-model refusal fires for the full plan, never for the resident
  // slice.
  const ManifestResidentBytes r = ManifestResident();
  CHECK(r.total < (int64_t(32) << 30));
  CHECK(r.total > (int64_t(3) << 30));
  CHECK(vllm_test::kKolibri1TotalSize > (int64_t(32) << 30));
  CHECK(r.expert_tensors == vllm_test::kKolibri1ExpertTensorCount);
}

TEST_CASE("kolibri1 TT B2b-i: resident-plan refusals name the deficit, the "
          "row, and the owed mesh layout") {
  const HfConfig config = MakeTinyConfig();
  TempCheckpoint ckpt(TinyFixture());
  const Kolibri1Weights w = LoadTiny(ckpt, config);

  const Kolibri1TTResidentStagingPlan fitted = PlanKolibri1TTResidentStaging(w);
  // A budget one byte short of the resident slice refuses by name.
  Kolibri1TTStagingOptions o;
  o.device_budget_bytes = fitted.total_bytes - 1;
  const std::string err = ResidentFailure(w, o);
  REQUIRE(!err.empty());
  CHECK(err.find("resident") != std::string::npos);
  CHECK(err.find("deficit") != std::string::npos);
  CHECK(err.find(std::to_string(fitted.total_bytes - 1)) !=
        std::string::npos);
  CHECK(err.find("MODEL-TEXT-kolibri-1-tenstorrent") != std::string::npos);
  // A mesh plan is refused: the mesh / expert-parallel staging layout is
  // owed, not decided here.
  Kolibri1TTStagingOptions mesh;
  mesh.mesh_chips = 4;
  const std::string mesh_err = ResidentFailure(w, mesh);
  REQUIRE(!mesh_err.empty());
  CHECK(mesh_err.find("mesh") != std::string::npos);
  CHECK(mesh_err.find("OWED") != std::string::npos);
}

TEST_CASE("kolibri1 TT B2b-i: manifest-driven resident accounting matches "
          "the byte-math table") {
  const ManifestResidentBytes r = ManifestResident();
  // Exact component totals from the REAL checkpoint manifest (the B1
  // streaming shape's committed numbers where they overlap: attention
  // 1,703,936,000; shared 196,608,000 fp8 + 48,000 grids = the B1
  // shared_expert_bytes 196,656,000; router gate 98,304,000; embed+head
  // 1,310,720,000).
  CHECK(r.attention_fp8 == 1703936000);
  CHECK(r.attention_scale == 416000);
  CHECK(r.shared_fp8 == 196608000);
  CHECK(r.shared_scale == 48000);
  CHECK(r.router_gate == 98304000);
  CHECK(r.router_bias == 76800);   // staged f32, 50 x [384] x 4
  CHECK(r.norms == 1054720);       // incl. the final norm
  CHECK(r.embed_head == 1310720000);
  CHECK(r.total == 3311163520);    // ~3.09 GiB resident
  // The B1 shape cross-check (the streaming plan's committed accounting).
  CHECK(r.attention_fp8 == 50 * 34078720);
  CHECK(r.shared_fp8 + r.shared_scale == 50 * 3933120);
  CHECK(r.router_gate == 50 * (384 * 2560 * 2));
  CHECK(r.embed_head == 2 * (128000 * 2560 * 2));
  // The resident slice is what fits: routed experts + resident > 32 GiB.
  const int64_t routed = int64_t(50) * 384 * 3933120;
  CHECK(routed + r.total > (int64_t(32) << 30));
}

TEST_CASE("kolibri1 TT B2b-i: resident plan asserts the hybrid geometry") {
  HfConfig config = MakeTinyConfig();
  // All-sliding: no full-attention (RNoPE) group — refused like wave A.
  config.raw["layer_types"] =
      nlohmann::json::array({"sliding_attention", "sliding_attention"});
  TempCheckpoint ckpt(TinyFixture());
  const Kolibri1Weights w = LoadTiny(ckpt, config);
  const std::string err = ResidentFailure(w, Kolibri1TTStagingOptions{});
  REQUIRE(!err.empty());
  CHECK(err.find("hybrid geometry") != std::string::npos);
}

// ---- Device leg (operator-run: card + VT_KOLIBRI1_TT_B2I_MODEL) ----

namespace {

bool TenstorrentPresent() {
  return vt::TryGetBackend(vt::DeviceType::kTENSTORRENT) != nullptr;
}

// Builds the staging request list from the resident plan: one request per
// planned tensor, PLUS one f32 request per fp8 tensor's scale grid. The
// handles are the request indices (0..n-1 in request order).
std::vector<vt::tenstorrent::TtStageOperand> BuildStageRequests(
    const Kolibri1TTResidentStagingPlan& plan) {
  std::vector<vt::tenstorrent::TtStageOperand> ops;
  ops.reserve(plan.tensors.size() * 2);
  for (const Kolibri1TTTensor& t : plan.tensors) {
    vt::tenstorrent::TtStageOperand op;
    op.host = t.host;
    op.rows = t.rows;
    op.cols = t.cols;
    if (t.dtype == Kolibri1TTDType::kFp8E4M3) {
      op.kind = vt::tenstorrent::TtStageOperand::Kind::kFp8E4M3;
      ops.push_back(op);
      // The f32 scale grid stages beside the packed operand (the plan
      // accounts it; the wave-A decision stages it with the operand
      // when the compute wave needs it — B2b is that wave).
      vt::tenstorrent::TtStageOperand grid;
      grid.host = t.scale_host;
      grid.rows = t.scale_rows;
      grid.cols = t.scale_cols;
      grid.kind = vt::tenstorrent::TtStageOperand::Kind::kF32;
      ops.push_back(grid);
    } else if (t.dtype == Kolibri1TTDType::kBf16) {
      op.kind = vt::tenstorrent::TtStageOperand::Kind::kBf16;
      ops.push_back(op);
    } else {
      op.kind = vt::tenstorrent::TtStageOperand::Kind::kF32;
      ops.push_back(op);
    }
  }
  return ops;
}

// Maps a plan tensor to its request handle (the fp8 scale grid rides in
// the NEXT slot).
int64_t HandleOf(const Kolibri1TTResidentStagingPlan& plan, size_t tensor) {
  int64_t handle = 0;
  for (size_t i = 0; i < tensor; ++i) {
    handle += plan.tensors[i].dtype == Kolibri1TTDType::kFp8E4M3 ? 2 : 1;
  }
  return handle;
}

// The bf16 ulp at magnitude x (the spacing of representable neighbors),
// clamped at the smallest bf16 subnormal (2^-133).
double UlpBF16(double x) {
  if (x == 0.0) return 0x1p-133;
  int e = 0;
  std::frexp(x, &e);  // x = m * 2^e with m in [0.5, 1): x in [2^(e-1), 2^e)
  return std::max(std::ldexp(1.0, e - 8), 0x1p-133);
}

}  // namespace

TEST_CASE("kolibri1 TT B2b-i: device bring-up — resident staging on the "
          "card + one verified device op") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  const char* model_dir = std::getenv("VT_KOLIBRI1_TT_B2I_MODEL");
  if (model_dir == nullptr || *model_dir == '\0') {
    MESSAGE("SKIPPED: VT_KOLIBRI1_TT_B2I_MODEL not set (no real checkpoint "
            "requested)");
    return;
  }
  const std::string dir = model_dir;
  if (!std::filesystem::exists(dir + "/model.safetensors.index.json")) {
    MESSAGE("SKIPPED: " << dir << " is not a Kolibri-1 checkpoint");
    return;
  }
  const double t_start = NowSec();
  std::fprintf(stderr, "[kolibri1-tt-b2i] device leg start: model=%s\n",
               dir.c_str());

  // ---- 1. Load the REAL checkpoint through the production registry path.
  const HfConfig config = vllm::LoadHfConfig(dir + "/config.json");
  const ModelRegistration& reg = ModelRegistry::Resolve(config);
  REQUIRE(reg.architecture == "Kolibri1ForCausalLM");
  reg.factory->parse_config(config);  // the production config hook
  const auto index = nlohmann::json::parse(
      std::ifstream(dir + "/model.safetensors.index.json"));
  // ONE open per DISTINCT shard (the W3 pattern: iterating the weight_map
  // itself would try 116303 opens).
  std::set<std::string> shard_names;
  for (const auto& [name, shard] : index.at("weight_map").items()) {
    (void)name;
    shard_names.insert(shard.get<std::string>());
  }
  std::vector<SafetensorsFile> shards;
  for (const std::string& shard : shard_names)
    shards.push_back(SafetensorsFile::Open(dir + "/" + shard));
  const ModelSource source = ModelSource::FromSafetensors(shards);
  std::fprintf(stderr, "[kolibri1-tt-b2i] loading the real fp8 checkpoint "
                       "(%zu shards) through ModelRegistry::Load...\n",
               shards.size());
  const double t_load0 = NowSec();
  std::unique_ptr<LoadedModel> model = ModelRegistry::Load(config, source);
  REQUIRE(model != nullptr);
  const Kolibri1Weights& w = Kolibri1LoadedModelWeights(*model);
  const double t_load = NowSec() - t_load0;
  std::fprintf(stderr,
               "[kolibri1-tt-b2i] load: %.1f s (hidden=%lld layers=%lld "
               "experts=%lld)\n",
               t_load, static_cast<long long>(w.params.hidden_size),
               static_cast<long long>(w.params.num_hidden_layers),
               static_cast<long long>(w.params.num_experts));

  // ---- 2. Plan the resident slice; the full-model refusal must NOT fire.
  const double t_plan0 = NowSec();
  const Kolibri1TTResidentStagingPlan plan = PlanKolibri1TTResidentStaging(w);
  const double t_plan = NowSec() - t_plan0;
  const Kolibri1TTResidentComponentBytes& c = plan.components;
  std::fprintf(stderr,
               "[kolibri1-tt-b2i] resident plan: %.3f s\n"
               "[kolibri1-tt-b2i]   attention fp8   %15lld B\n"
               "[kolibri1-tt-b2i]   attention grids %15lld B\n"
               "[kolibri1-tt-b2i]   shared fp8      %15lld B\n"
               "[kolibri1-tt-b2i]   shared grids    %15lld B\n"
               "[kolibri1-tt-b2i]   router gate     %15lld B\n"
               "[kolibri1-tt-b2i]   router bias     %15lld B\n"
               "[kolibri1-tt-b2i]   norms           %15lld B\n"
               "[kolibri1-tt-b2i]   embed + head    %15lld B\n"
               "[kolibri1-tt-b2i]   TOTAL           %15lld B (%.3f GiB, "
               "%lld tensors)\n",
               t_plan, static_cast<long long>(c.attention_fp8_bytes),
               static_cast<long long>(c.attention_scale_bytes),
               static_cast<long long>(c.shared_expert_fp8_bytes),
               static_cast<long long>(c.shared_expert_scale_bytes),
               static_cast<long long>(c.router_gate_bytes),
               static_cast<long long>(c.router_bias_bytes),
               static_cast<long long>(c.norm_bytes),
               static_cast<long long>(c.embed_head_bytes),
               static_cast<long long>(plan.total_bytes),
               double(plan.total_bytes) / (1 << 30),
               static_cast<long long>(plan.tensor_count));
  // The wave-A single-device FULL-model refusal DOES fire for the full
  // tree (the resident slice is what does not refuse).
  {
    bool full_refused = false;
    try {
      (void)PlanKolibri1TTStaging(w);
    } catch (const std::exception& e) {
      full_refused = true;
      const std::string what = e.what();
      CHECK(what.find("deficit") != std::string::npos);
    }
    CHECK(full_refused);
    std::fprintf(stderr, "[kolibri1-tt-b2i] full-model single-device "
                         "refusal (expected): fires\n");
  }

  // ---- 3. Stage the resident slice on device; verify EVERY operand.
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  vt::Queue q = be.CreateQueue();
  const std::vector<vt::tenstorrent::TtStageOperand> ops =
      BuildStageRequests(plan);
  const double t_stage0 = NowSec();
  const vt::tenstorrent::TtStagingReport report =
      vt::tenstorrent::StageResidentOperands(q, ops);
  const double t_stage = NowSec() - t_stage0;
  std::fprintf(stderr,
               "[kolibri1-tt-b2i] staged %lld operands (%lld fp8 / %lld bf16 "
               "/ %lld f32) = %lld B in %.2f s\n"
               "[kolibri1-tt-b2i]   chips=%lld first_chip_id=%lld "
               "dram_total=%lld B dram_free_after=%lld B\n",
               static_cast<long long>(report.tensors),
               static_cast<long long>(report.fp8_tensors),
               static_cast<long long>(report.bf16_tensors),
               static_cast<long long>(report.f32_tensors),
               static_cast<long long>(report.bytes_staged), t_stage,
               static_cast<long long>(report.chips),
               static_cast<long long>(report.first_chip_id),
               static_cast<long long>(report.dram_total_bytes),
               static_cast<long long>(report.dram_free_after_bytes));
  // The staged bytes match the plan EXACTLY (fp8 operands + grids + bf16
  // modules + router bias).
  CHECK(report.bytes_staged == plan.total_bytes);
  CHECK(report.tensors ==
        plan.tensor_count +
            static_cast<int64_t>(std::count_if(
                plan.tensors.begin(), plan.tensors.end(),
                [](const Kolibri1TTTensor& t) {
                  return t.dtype == Kolibri1TTDType::kFp8E4M3;
                })));

  // Readback verification of EVERY staged operand: the fp8 operands
  // BYTE-EXACT (raw host bytes vs the packed bytes; the ttnn f32 pivot
  // flushes subnormals and cannot verify fp8), the bf16/f32 operands
  // through the exact f32 pivot. Mismatch COUNTS, not per-element
  // assertions — a per-element CHECK over ~3.3 GiB of staged bytes would
  // dominate the runtime.
  const double t_verify0 = NowSec();
  int64_t fp8_checked = 0;
  int64_t bf16_checked = 0;
  int64_t f32_checked = 0;
  int64_t mismatches = 0;
  for (size_t i = 0; i < plan.tensors.size(); ++i) {
    const Kolibri1TTTensor& t = plan.tensors[i];
    const int64_t handle = HandleOf(plan, i);
    const std::vector<float> got =
        vt::tenstorrent::ReadbackStagedOperandF32(q, handle);
    if (got.size() != static_cast<size_t>(t.rows * t.cols)) {
      ++mismatches;
      continue;
    }
    if (t.dtype == Kolibri1TTDType::kFp8E4M3) {
      // Byte-exact verification: the staged operand must hold the packed
      // bytes VERBATIM. (The f32 pivot readback above is not byte-exact for
      // fp8: tt-metal's host FP8→F32 decode flushes subnormals to zero —
      // tt_metal/impl/data_format/float8.cpp, float8_e4m3::operator float —
      // and fp8 weight tensors contain subnormal-range bytes, so every
      // tensor would mismatch through the pivot. The fp8 check therefore
      // goes through the raw host bytes, which are dtype-agnostic.)
      const uint8_t* packed = static_cast<const uint8_t*>(t.host);
      const std::vector<uint8_t> raw =
          vt::tenstorrent::ReadbackStagedOperandBytes(q, handle);
      if (raw.size() != static_cast<size_t>(t.rows * t.cols) ||
          std::memcmp(raw.data(), packed,
                      static_cast<size_t>(t.rows * t.cols)) != 0) {
        ++mismatches;
      }
      // The scale grid rides in the NEXT request slot (f32, exact pivot).
      const std::vector<float> grid =
          vt::tenstorrent::ReadbackStagedOperandF32(q, handle + 1);
      if (grid.size() != static_cast<size_t>(t.scale_rows * t.scale_cols)) {
        ++mismatches;
      } else {
        const float* scale = static_cast<const float*>(t.scale_host);
        for (size_t e = 0; e < grid.size(); ++e) {
          if (grid[e] != scale[e]) {
            ++mismatches;
            break;
          }
        }
      }
      ++fp8_checked;
    } else if (t.dtype == Kolibri1TTDType::kBf16) {
      const uint16_t* src = static_cast<const uint16_t*>(t.host);
      for (int64_t e = 0; e < t.rows * t.cols; ++e) {
        if (got[static_cast<size_t>(e)] != vt::BF16ToF32(src[e])) {
          ++mismatches;
          break;
        }
      }
      ++bf16_checked;
    } else {
      const float* src = static_cast<const float*>(t.host);
      for (int64_t e = 0; e < t.rows * t.cols; ++e) {
        if (got[static_cast<size_t>(e)] != src[e]) {
          ++mismatches;
          break;
        }
      }
      ++f32_checked;
    }
  }
  const double t_verify = NowSec() - t_verify0;
  std::fprintf(stderr,
               "[kolibri1-tt-b2i] readback verified %lld fp8 operands (+grids) "
               "/ %lld bf16 / %lld f32 in %.1f s — mismatches=%lld\n",
               static_cast<long long>(fp8_checked),
               static_cast<long long>(bf16_checked),
               static_cast<long long>(f32_checked), t_verify,
               static_cast<long long>(mismatches));
  CHECK(mismatches == 0);

  // ---- 4. ONE verified device op: embedding lookup + the first resident
  //         projection (layer 0 q_proj), vs the CPU row's output.
  const int64_t vocab = w.params.vocab_size;
  const int64_t hidden = w.params.hidden_size;
  const int64_t q_rows = w.params.num_attention_heads * w.params.head_dim;
  // The first golden prompt's first tokens (a real workload's ids).
  const nlohmann::json goldens =
      nlohmann::json::parse(std::ifstream(KOLIBRI1_GOLDENS));
  std::vector<int32_t> ids;
  for (const auto& id : goldens.at("prompts").at(0).at("input_ids")) {
    ids.push_back(id.get<int32_t>());
    if (static_cast<int64_t>(ids.size()) == 8) break;
  }
  REQUIRE(!ids.empty());
  const int64_t T = static_cast<int64_t>(ids.size());
  std::fprintf(stderr, "[kolibri1-tt-b2i] device op: embedding + layer0 q_proj "
                       "over %lld golden-prompt tokens (hidden=%lld q=%lld)\n",
               static_cast<long long>(T), static_cast<long long>(hidden),
               static_cast<long long>(q_rows));

  // CPU reference embedding: the bf16 rows for the same ids (the CPU row's
  // gather source, read straight from the loaded table).
  const uint16_t* embed_src =
      reinterpret_cast<const uint16_t*>(w.embed_tokens.bytes.data());
  std::vector<uint16_t> cpu_embed(static_cast<size_t>(T * hidden));
  for (int64_t i = 0; i < T; ++i) {
    std::memcpy(&cpu_embed[static_cast<size_t>(i * hidden)],
                &embed_src[static_cast<size_t>(ids[static_cast<size_t>(i)]) *
                               hidden],
                static_cast<size_t>(hidden) * 2);
  }

  // Device embedding through the production vt op (the TT backend stages
  // the table and computes on the card).
  const vt::Device dev{vt::DeviceType::kTENSTORRENT, 0};
  void* mem_ids = be.Alloc(ids.size() * sizeof(int32_t));
  void* mem_emb = be.Alloc(static_cast<size_t>(T * hidden) * 2);
  be.Copy(q, mem_ids, ids.data(), ids.size() * sizeof(int32_t));
  // The op only READS the table bytes (the TT embedding kernel stages its
  // own device shadow from the host mirror).
  vt::Tensor table = vt::Tensor::Contiguous(
      const_cast<uint8_t*>(w.embed_tokens.bytes.data()), vt::DType::kBF16,
      dev, {vocab, hidden});
  vt::Tensor ids_t =
      vt::Tensor::Contiguous(mem_ids, vt::DType::kI32, dev, {T});
  vt::Tensor emb_t = vt::Tensor::Contiguous(mem_emb, vt::DType::kBF16, dev,
                                            {T, hidden});
  const double t_op0 = NowSec();
  vt::Embedding(q, emb_t, table, ids_t);
  std::vector<uint16_t> dev_embed(static_cast<size_t>(T * hidden));
  be.Copy(q, dev_embed.data(), mem_emb,
          static_cast<size_t>(T * hidden) * 2);
  be.Synchronize(q);
  // The embedding gather is BIT-EXACT (a row gather, no arithmetic).
  int64_t emb_mismatch = 0;
  for (size_t i = 0; i < dev_embed.size(); ++i) {
    if (dev_embed[i] != cpu_embed[i]) ++emb_mismatch;
  }
  std::fprintf(stderr,
               "[kolibri1-tt-b2i] embedding: %zu elements, mismatches=%lld "
               "(%.1f ms)\n",
               dev_embed.size(), static_cast<long long>(emb_mismatch),
               (NowSec() - t_op0) * 1e3);
  CHECK(emb_mismatch == 0);

  // The first resident projection: layer 0 q_proj. The device GEMM
  // consumes the bf16 dequant of the STAGED fp8 bytes (read back raw —
  // byte-exact verified above — and decoded with the scalar reference
  // vt::F8E4M3ToF32; the ttnn f32 pivot is NOT usable here because its
  // host FP8→F32 decode flushes subnormals to zero), scaled by the plan's
  // f32 grid: the CPU row's documented R1 disposition.
  const Fp8BlockWeight& q_proj = w.layers[0].attn.q_proj.fp8_block;
  const int64_t q_n = q_proj.n;
  const int64_t q_k = q_proj.k;
  REQUIRE(q_n == q_rows);
  REQUIRE(q_k == hidden);
  size_t q_tensor = plan.tensors.size();
  for (size_t i = 0; i < plan.tensors.size(); ++i) {
    if (plan.tensors[i].name == "model.layers.0.self_attn.q_proj") {
      q_tensor = i;
      break;
    }
  }
  REQUIRE(q_tensor < plan.tensors.size());
  const std::vector<uint8_t> q_staged =
      vt::tenstorrent::ReadbackStagedOperandBytes(q, HandleOf(plan, q_tensor));
  REQUIRE(q_staged.size() == static_cast<size_t>(q_n * q_k));
  const float* q_scale =
      reinterpret_cast<const float*>(q_proj.scale.bytes.data());
  const int64_t scale_cols = q_proj.scale.shape[1];
  std::vector<uint16_t> q_w_dequant(static_cast<size_t>(q_n * q_k));
  for (int64_t n = 0; n < q_n; ++n) {
    for (int64_t k = 0; k < q_k; ++k) {
      const float s =
          q_scale[(n / q_proj.block_n) * scale_cols + (k / q_proj.block_k)];
      q_w_dequant[static_cast<size_t>(n * q_k + k)] =
          vt::F32ToBF16(vt::F8E4M3ToF32(q_staged[static_cast<size_t>(n * q_k + k)]) * s);
    }
  }

  // Device GEMM through the production vt op (bf16 out, like the CPU
  // row's LinearBT).
  void* mem_qw = be.Alloc(static_cast<size_t>(q_n * q_k) * 2);
  be.Copy(q, mem_qw, q_w_dequant.data(),
          static_cast<size_t>(q_n * q_k) * 2);
  void* mem_qout = be.Alloc(static_cast<size_t>(T * q_n) * 2);
  vt::Tensor q_w_t = vt::Tensor::Contiguous(mem_qw, vt::DType::kBF16, dev,
                                            {q_n, q_k});
  vt::Tensor q_out_t = vt::Tensor::Contiguous(mem_qout, vt::DType::kBF16,
                                              dev, {T, q_n});
  const double t_gemm0 = NowSec();
  vt::MatmulBT(q, q_out_t, emb_t, q_w_t);
  std::vector<uint16_t> dev_q(static_cast<size_t>(T * q_n));
  be.Copy(q, dev_q.data(), mem_qout, static_cast<size_t>(T * q_n) * 2);
  be.Synchronize(q);
  const double t_gemm = NowSec() - t_gemm0;

  // CPU reference: the CPU row's own path — DequantRowsBf16 over the same
  // packed bytes (bit-identical values to the pivot, verified above), then
  // the CPU vt::MatmulBT over the same bf16 activation (the device's
  // embedding readback was verified bit-exact, so the inputs are the
  // same).
  vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue cq = cpu.CreateQueue();
  std::vector<uint16_t> cpu_qw(static_cast<size_t>(q_n * q_k));
  kolibri1_fp8::DequantRowsBf16(
      q_proj.packed.bytes.data(), q_scale, scale_cols, 0, q_n, q_k,
      q_proj.block_n, q_proj.block_k, cpu_qw.data());
  // The dequant from the staged-bytes pivot is bit-identical to the CPU
  // row's dequant from the packed bytes.
  CHECK(std::memcmp(cpu_qw.data(), q_w_dequant.data(),
                    static_cast<size_t>(q_n * q_k) * 2) == 0);
  const vt::Device cpu_dev{vt::DeviceType::kCPU, 0};
  std::vector<uint16_t> cpu_act = cpu_embed;  // same input both sides
  std::vector<uint16_t> cpu_qout(static_cast<size_t>(T * q_n));
  vt::Tensor a_t = vt::Tensor::Contiguous(cpu_act.data(), vt::DType::kBF16,
                                          cpu_dev, {T, hidden});
  vt::Tensor w_t = vt::Tensor::Contiguous(cpu_qw.data(), vt::DType::kBF16,
                                          cpu_dev, {q_n, q_k});
  vt::Tensor o_t = vt::Tensor::Contiguous(cpu_qout.data(), vt::DType::kBF16,
                                          cpu_dev, {T, q_n});
  const double t_cgemm0 = NowSec();
  vt::MatmulBT(cq, o_t, a_t, w_t);
  const double t_cgemm = NowSec() - t_cgemm0;

  // Comparison: identical bf16 operands. The CPU reference accumulates in
  // f32 with a single bf16 rounding at the store (the CPU row's contract);
  // the device ttnn matmul's internal precision is bf16-granularity for
  // this tiny-M fallback shape (measured 2026-10-08 — see the header
  // note). Envelope per element: |dev - cpu| <= 8 * 2^-8 * sum_k
  // |a_ik w_jk| + max(ulp(|cpu|), ulp(|dev|)) — at most 8 bf16 unit
  // roundoffs per unit term-magnitude sum plus the store rounding.
  // Measured: bit-exact fraction, max ulp distance, max abs/rel diff,
  // max per-element error ratio.
  //
  // Term-magnitude sums over the SAME operands both sides consume
  // (bf16 -> f32 is exact; accumulated in f64 so the bound itself
  // carries no rounding).
  std::vector<double> abs_a(static_cast<size_t>(T) * hidden);
  std::vector<double> abs_w(static_cast<size_t>(q_n) * q_k);
  for (int64_t i = 0; i < T; ++i)
    for (int64_t k = 0; k < hidden; ++k)
      abs_a[static_cast<size_t>(i) * hidden + k] = std::fabs(
          double(vt::BF16ToF32(cpu_act[static_cast<size_t>(i) * hidden + k])));
  for (int64_t j = 0; j < q_n; ++j)
    for (int64_t k = 0; k < q_k; ++k)
      abs_w[static_cast<size_t>(j) * q_k + k] = std::fabs(
          double(vt::BF16ToF32(cpu_qw[static_cast<size_t>(j) * q_k + k])));
  std::vector<double> terms(static_cast<size_t>(T) * q_n, 0.0);
  for (int64_t i = 0; i < T; ++i)
    for (int64_t j = 0; j < q_n; ++j) {
      double s = 0.0;
      for (int64_t k = 0; k < q_k; ++k)
        s += abs_a[static_cast<size_t>(i) * hidden + k] *
             abs_w[static_cast<size_t>(j) * q_k + k];
      terms[static_cast<size_t>(i) * q_n + j] = s;
    }
  int64_t exact = 0;
  int64_t max_ulp = 0;
  int64_t envelope_violations = 0;
  double max_abs = 0.0;
  double max_rel = 0.0;
  double max_ratio = 0.0;  // diff / (2^-8 * terms)
  const bool dbg = std::getenv("VT_KOLIBRI1_TT_B2I_DEBUG") != nullptr;
  int64_t shown = 0;
  std::vector<int64_t> bad_per_row(static_cast<size_t>(T), 0);
  for (int64_t i = 0; i < T; ++i) {
    for (int64_t j = 0; j < q_n; ++j) {
      const size_t e = static_cast<size_t>(i * q_n + j);
      if (dev_q[e] == cpu_qout[e]) {
        ++exact;
        continue;
      }
      ++bad_per_row[static_cast<size_t>(i)];
      const float a = vt::BF16ToF32(dev_q[e]);
      const float b = vt::BF16ToF32(cpu_qout[e]);
      const double diff = std::fabs(double(a) - double(b));
      max_abs = std::max(max_abs, diff);
      const double denom = std::max(std::fabs(double(b)), 1e-30);
      max_rel = std::max(max_rel, diff / denom);
      // bf16 ulp distance at the reference magnitude (magnitude bits only).
      const uint32_t ua = static_cast<uint32_t>(dev_q[e]) & 0x7fffu;
      const uint32_t ub = static_cast<uint32_t>(cpu_qout[e]) & 0x7fffu;
      const int64_t ulp =
          static_cast<int64_t>(ua > ub ? ua - ub : ub - ua);
      max_ulp = std::max(max_ulp, ulp);
      // The stated envelope: 8 bf16 unit roundoffs per unit term-magnitude
      // sum plus the store rounding (the larger of the two magnitudes'
      // bf16 ulps bounds both sides' round-at-store contributions).
      const double bound = 8.0 * 0x1p-8 * terms[e] +
                           std::max(UlpBF16(std::fabs(double(a))),
                                    UlpBF16(std::fabs(double(b))));
      if (diff > bound) ++envelope_violations;
      const double ratio_denom = std::max(0x1p-8 * terms[e], 1e-300);
      max_ratio = std::max(max_ratio, diff / ratio_denom);
      if (dbg && shown < 8) {
        ++shown;
        std::fprintf(stderr,
                     "[b2i-debug] GEMM mismatch i=%lld j=%lld dev=%.9g "
                     "cpu=%.9g ulp=%lld terms=%.4f bound=%.4e\n",
                     static_cast<long long>(i), static_cast<long long>(j),
                     static_cast<double>(a), static_cast<double>(b),
                     static_cast<long long>(ulp), terms[e], bound);
      }
    }
  }
  if (dbg) {
    std::fprintf(stderr, "[b2i-debug] GEMM bad-per-row:");
    for (int64_t i = 0; i < T; ++i)
      std::fprintf(stderr, " %lld:%lld", static_cast<long long>(i),
                   static_cast<long long>(bad_per_row[static_cast<size_t>(i)]));
    std::fprintf(stderr, " (of %lld cols)\n", static_cast<long long>(q_n));
  }
  std::fprintf(stderr,
               "[kolibri1-tt-b2i] q_proj GEMM: device %.1f ms vs CPU %.1f "
               "ms; bit-exact %lld/%zu; max_ulp=%lld max_abs=%.3e "
               "max_rel=%.3e max_err_ratio=%.3e (per 2^-8*terms); "
               "envelope_violations=%lld\n",
               t_gemm * 1e3, t_cgemm * 1e3, static_cast<long long>(exact),
               dev_q.size(), static_cast<long long>(max_ulp), max_abs,
               max_rel, max_ratio, static_cast<long long>(envelope_violations));
  // Stated envelope: per element, |dev - cpu| <= 8 bf16 unit roundoffs per
  // unit term-magnitude sum plus the store rounding (see the header note),
  // and the outputs are finite and non-constant.
  CHECK(envelope_violations == 0);
  double lo = vt::BF16ToF32(dev_q[0]), hi = lo;
  for (const uint16_t v : dev_q) {
    const float f = vt::BF16ToF32(v);
    CHECK(std::isfinite(f));
    lo = std::min(lo, double(f));
    hi = std::max(hi, double(f));
  }
  CHECK(hi - lo > 1e-6);

  be.Free(mem_ids);
  be.Free(mem_emb);
  be.Free(mem_qw);
  be.Free(mem_qout);
  std::fprintf(stderr,
               "[kolibri1-tt-b2i] device leg PASS in %.1f s total "
               "(load %.1f / plan %.3f / stage %.2f / verify %.1f / op %.1f)\n",
               NowSec() - t_start, t_load, t_plan, t_stage, t_verify,
               NowSec() - t_op0);
}
