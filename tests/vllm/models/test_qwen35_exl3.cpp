// MODEL-QWEN35-EXL3 / MODEL-QWEN35-GDN-EXL3 — the EXL3 (exllamav3 trellis) arm
// of the Qwen3.5 text model: the DENSE half (#2495 items 3 and 5) and the GDN
// linear-attention tower (#2495 item 4).
//
// `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` declares
// `architectures: ["Qwen3_5ForConditionalGeneration"]`, which this tree already
// registers and already runs in bf16, FP8 and NVFP4. Before this row
// `grep -c exl3 src/vllm/model_executor/models/qwen3_5_dense_weights.cpp`
// returned 0: an EXL3 checkpoint died on a missing-tensor lookup, because an
// EXL3 projection ships NO `.weight` at all and every probe in the resolver
// reads one.
//
// FOUR QUESTIONS, and only the first is about the arm that was added.
//
//   G1  Does an EXL3 checkpoint LOAD, with the right codebook, the right
//       per-tensor width and the right geometry? The codebook is the half a
//       shape check cannot see: the wrong multiplier decodes to the correct RMS
//       and uncorrelated values (`test_exl3_native_loader.cpp` records the
//       measurement), so it is asserted directly.
//
//   G2  Does a PRODUCTION entry point REACH the arm? The case enters at
//       `ModelRegistry::Forward` over the weights the LOADER produced, and
//       compares against the SAME trellis bytes decoded into the bf16 fields.
//       Deleting an EXL3 call site does not merely move the numbers: the arm
//       falls through to a bf16 field an EXL3 load leaves EMPTY and
//       `dense_attn::ResidentWeight` refuses it by name, so the case reds
//       either way. A unit test that constructs an `Exl3Weight` by hand would
//       prove the class works and never that anything reaches it
//       (`.agents/reachability.md`).
//
//   G3  Does the GDN half LOAD AND FORWARD? 48 of the real model's 64 layers
//       are `linear_attention`, so this half is most of the checkpoint. This
//       case REPLACES a refusal: until #2495 item 4 the three GDN projections
//       were refused by name because nothing in `ProjectGdnQkvz`/`ProjectGdnOut`
//       consumed an `Exl3Weight`, and refusing beat half-loading. The case
//       enters at `ModelRegistry::Forward` for the reason G2 does.
//
//   G3b Does the ARTIFACT'S PATTERN load? The published checkpoint is 64 layers
//       of three `linear_attention` then one `full_attention`. A loader that
//       keyed the arm off the layer INDEX rather than off the tensors still
//       passes at 1/1, so the ratio is exercised directly.
//
//   G4  Are the bf16, per-tensor FP8 and NVFP4 arms UNCHANGED, in the dense
//       towers AND in the GDN one? A fourth rung in a presence-based resolver is
//       exactly where an existing arm gets mis-selected, and a mis-selection is
//       silent: every one of these checkpoints still loads and still produces
//       plausible numbers.
//
// No checkpoint download, no GPU, no snapshot. The fixture is a complete but
// tiny `Qwen3_5ForConditionalGeneration` dense checkpoint written to a temp
// directory, in the safetensors byte layout pinned by
// `tests/vllm/test_safetensors.cpp`.
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/model_executor/models/qwen3_dflash.h"
#include "vllm/model_executor/models/qwen3_dflash2.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace {

using vllm::Exl3Weight;
using vllm::ForwardLogits;
using vllm::GdnStateCache;
using vllm::HfConfig;
using vllm::ModelForwardInput;
using vllm::ModelRegistry;
using vllm::ModelSource;
using vllm::PagedKvCache;
using vllm::Qwen3_5DenseWeights;
using vllm::Qwen3_5MTPKind;
using vllm::Qwen3_5MTPModel;
using vllm::Qwen3_5MTPWeights;
using vllm::SafetensorsFile;
using vllm::v1::CommonAttentionMetadata;
using vllm::v1::GDNAttentionMetadata;
using vt::DType;

constexpr const char* kArch = "Qwen3_5ForConditionalGeneration";

// The published artifact's width and codebook. 270 of the 272 quantized tensors
// of `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` are FOUR-bit, and every one of them
// carries a `mul1` marker, which is codebook 2 (`exl3.py:74-77`).
constexpr int kBits = 4;
constexpr int kCodebook = 2;
// `torch.tensor(0x83DCD12D, uint32).view(torch.int)` (`quantize.py:1421-1424`):
// one I32 holding the codebook's own multiplier, exactly as `mcg` is written.
constexpr uint32_t kMul1Multiplier = 0x83DCD12DU;

// ---------------------------------------------------------------------------
// Deterministic bytes
// ---------------------------------------------------------------------------

struct Rng {
  uint32_t s = 1u;
  uint32_t next() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  }
  float unit() { return static_cast<float>(next() % 2000) / 1000.0F - 1.0F; }
};

int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (const int64_t d : shape) n *= d;
  return n;
}

struct FixtureTensor {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

std::vector<uint8_t> Bytes16(const std::vector<uint16_t>& values) {
  std::vector<uint8_t> b(values.size() * 2);
  for (size_t i = 0; i < values.size(); ++i) {
    b[2 * i] = static_cast<uint8_t>(values[i] & 0xff);
    b[2 * i + 1] = static_cast<uint8_t>(values[i] >> 8);
  }
  return b;
}

std::vector<uint8_t> Bytes32F(const std::vector<float>& values) {
  std::vector<uint8_t> b(values.size() * 4);
  for (size_t i = 0; i < values.size(); ++i)
    std::memcpy(b.data() + 4 * i, &values[i], 4);
  return b;
}

std::vector<uint8_t> Bytes32U(uint32_t v) {
  std::vector<uint8_t> b(4);
  std::memcpy(b.data(), &v, 4);
  return b;
}

// The UNQUANTIZED REMAINDER, written at either dtype from ONE set of values.
//
// An EXL3 artifact stores its remainder at F16, because exllamav3 runs the
// linear in fp16; the config's own `torch_dtype` is bfloat16, so the loader
// materializes it to bf16 -- the MODEL dtype every layer inherits. The BF16 arm
// here therefore writes `F32ToBF16(F16ToF32(F32ToF16(v)))` rather than
// `F32ToBF16(v)`: after loading, the two checkpoints hold BIT-IDENTICAL norms
// and embeddings, so the forward comparison in G2 is measuring the projection
// arms and nothing else. Writing the obvious `F32ToBF16(v)` would put a second
// difference into the comparison and quietly loosen it.
std::vector<uint8_t> RemainderBytes(const std::vector<int64_t>& shape,
                                    uint32_t seed, float scale, bool as_f16) {
  Rng r;
  r.s = seed | 1u;
  const int64_t n = Numel(shape);
  std::vector<uint16_t> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    const uint16_t half = vt::F32ToF16(r.unit() * scale);
    v[static_cast<size_t>(i)] =
        as_f16 ? half : vt::F32ToBF16(vt::F16ToF32(half));
  }
  return Bytes16(v);
}

// ---------------------------------------------------------------------------
// One EXL3 projection, and the SAME weights decoded
// ---------------------------------------------------------------------------

// The trellis bytes plus the two sign vectors, held once so the EXL3 fixture
// and its decoded twin are the same bytes read two ways. That is what makes the
// forward comparison an EQUIVALENCE rather than a tolerance between two
// different models.
struct Exl3Proj {
  int64_t k = 0;
  int64_t n = 0;
  std::vector<uint16_t> trellis;
  std::vector<uint16_t> suh;
  std::vector<uint16_t> svh;
};

Exl3Proj MakeProj(int64_t k, int64_t n, uint32_t seed) {
  Rng r;
  r.s = seed | 1u;
  Exl3Proj p;
  p.k = k;
  p.n = n;
  p.trellis.resize(static_cast<size_t>(k / 16) * static_cast<size_t>(n / 16) *
                   16 * kBits);
  for (auto& w : p.trellis) w = static_cast<uint16_t>(r.next() & 0xffffu);
  p.suh.resize(static_cast<size_t>(k));
  p.svh.resize(static_cast<size_t>(n));
  // Sign vectors, which is what they are: +-1 in fp16 (`exl3.py:48-49`).
  for (auto& v : p.suh) v = vt::F32ToF16((r.next() & 1u) != 0u ? 1.0F : -1.0F);
  for (auto& v : p.svh) v = vt::F32ToF16((r.next() & 1u) != 0u ? 1.0F : -1.0F);
  return p;
}

void AppendExl3(std::vector<FixtureTensor>& out, const std::string& proj,
                const Exl3Proj& p) {
  out.push_back({proj + ".trellis", "I16",
                 {p.k / 16, p.n / 16, 16 * kBits}, Bytes16(p.trellis)});
  out.push_back({proj + ".suh", "F16", {p.k}, Bytes16(p.suh)});
  out.push_back({proj + ".svh", "F16", {p.n}, Bytes16(p.svh)});
  out.push_back({proj + ".mul1", "I32", {1}, Bytes32U(kMul1Multiplier)});
}

// `Exl3DequantLinear` yields [k, n]; a torch Linear stores [out, in] = [n, k],
// which is what the bf16 loader rung reads. This transpose is the ONE place the
// two arms' orientations are reconciled, and getting it wrong shows up
// immediately as a failed equivalence rather than as a plausible wrong number.
std::vector<uint8_t> DecodedTorchWeight(const Exl3Proj& p) {
  std::vector<float> w(static_cast<size_t>(p.k) * static_cast<size_t>(p.n), 0.0F);
  vt::Exl3DequantLinear(p.trellis.data(), p.suh.data(), p.svh.data(), p.k, p.n,
                        kBits, kCodebook, w.data());
  std::vector<uint16_t> bf(static_cast<size_t>(p.k) * static_cast<size_t>(p.n));
  for (int64_t i = 0; i < p.n; ++i)
    for (int64_t j = 0; j < p.k; ++j)
      bf[static_cast<size_t>(i) * static_cast<size_t>(p.k) + static_cast<size_t>(j)] =
          vt::F32ToBF16(w[static_cast<size_t>(j) * static_cast<size_t>(p.n) +
                          static_cast<size_t>(i)]);
  return Bytes16(bf);
}

void AppendDecodedBf16(std::vector<FixtureTensor>& out, const std::string& proj,
                       const Exl3Proj& p) {
  out.push_back({proj + ".weight", "BF16", {p.n, p.k}, DecodedTorchWeight(p)});
}

// ---------------------------------------------------------------------------
// The checkpoint container
// ---------------------------------------------------------------------------

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
    payload.append(reinterpret_cast<const char*>(t.bytes.data()), t.bytes.size());
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
           ("vllm_qwen35_exl3_" + std::to_string(nonce) + "_" +
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

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

// EVERY EXL3 projection's k AND n is a multiple of 128, because each side was
// Hadamard-128 transformed at quantization time (`exl3_lib/quantize.py:15`) and
// the reference dequant refuses anything else. That constrains this model more
// than a bf16 one: `num_key_value_heads * head_dim` is a projection width and
// cannot be the usual small GQA number, and the vocabulary is a projection
// width too because the head is quantized.
// The artifact's own `layer_types`: three `linear_attention` then one
// `full_attention`, 64 entries. Indices 0,1,2 are linear and 3 is full, which is
// `i % 4 == 3` — read from `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw`'s `config.json`.
std::vector<std::string> Ratio3To1LayerTypes(int layers) {
  std::vector<std::string> t;
  t.reserve(static_cast<size_t>(layers));
  for (int i = 0; i < layers; ++i)
    t.emplace_back(i % 4 == 3 ? "full_attention" : "linear_attention");
  return t;
}

HfConfig DenseConfig(bool with_exl3_quant_config,
                     const std::vector<std::string>& layer_types = {
                         "linear_attention", "full_attention"}) {
  HfConfig c;
  c.model_type = "qwen3_5_text";
  c.architectures = {kArch};
  c.hidden_size = 128;
  c.num_hidden_layers = static_cast<int>(layer_types.size());
  c.vocab_size = 128;          // lm_head's N
  c.num_attention_heads = 2;   // q_proj N = 2*Hq*Dh = 256 (output gate doubled)
  c.num_key_value_heads = 2;   // k/v_proj N = Hkv*Dh = 128
  c.head_dim = 64;             // o_proj K = Hq*Dh = 128
  c.layer_types = layer_types;
  c.intermediate_size = 128;
  c.num_experts = 0;
  c.linear_num_key_heads = 1;
  c.linear_num_value_heads = 2;
  // MODEL-QWEN35-GDN-EXL3 (#2495 item 4): 64, not the 32 this fixture carried
  // while the GDN tower could only be refused. `conv_dim = 2*Hk*Dk + Hv*Dv` is
  // an EXL3 projection WIDTH once the tower is quantized, and every EXL3 K and
  // N must be a multiple of 128 because each side carries a blockwise
  // Hadamard-128 (`exl3_lib/quantize.py:15`; `src/vt/exl3_policy.cpp:150`
  // declines anything else). At 32 the width was 192 and could not be an EXL3
  // projection at all. At 64 it is 256, and `value_dim` stays 128.
  c.linear_key_head_dim = 64;    // conv_dim == 256
  c.linear_value_head_dim = 64;  // value_dim == 128
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 32;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = 64;
  nlohmann::json doc = nlohmann::json::object();
  doc["architectures"] = nlohmann::json::array({kArch});
  if (with_exl3_quant_config) {
    // What the artifact actually declares. It is written here NOT because the
    // loader reads it -- the scheme is resolved per Linear from the tensors, as
    // `Linear.is_exl3_storage` does upstream -- but because a config this tree
    // has never seen must not derail the compressed-tensors, ModelOpt and
    // block-FP8 readers that run over `config.raw` before any weight loads.
    nlohmann::json q = nlohmann::json::object();
    q["quant_method"] = "exl3";
    q["bits"] = 3.5;
    doc["quantization_config"] = q;
  }
  c.raw = std::move(doc);
  return c;
}

// How the four self_attn projections, the three MLP projections and the head
// are written. Every arm shares one geometry, so a mis-selection is the only
// difference a case can be reading.
enum class Arm {
  kExl3,        // .trellis + .suh + .svh + .mul1
  kBf16,        // .weight BF16, the SAME weights decoded
  kFp8,         // .weight F8_E4M3 + .weight_scale + .input_scale (per-tensor)
  kNvfp4,       // .weight U8 + .weight_scale F8 + .weight_scale_2 (ModelOpt)
};

// The projections of the model, in one place so every arm writes the same set.
struct Geometry {
  int64_t hidden = 128;
  int64_t vocab = 128;
  int64_t q_n = 256;
  int64_t kv_n = 128;
  int64_t inter = 128;
  int64_t head_dim = 64;
  int64_t attn_out_k = 128;  // Hq * Dh
  // 2*Hk*Dk + Hv*Dv = 2*1*64 + 2*64. A multiple of 128 so the GDN tower can be
  // an EXL3 projection; see `DenseConfig`.
  int64_t conv_dim = 256;
  int64_t value_dim = 128;
  int64_t num_v_heads = 2;
  int64_t v_head_dim = 64;
  int64_t conv_k = 4;
};

std::vector<uint8_t> Fp8Walk(int64_t n) {
  std::vector<uint8_t> b(static_cast<size_t>(n));
  for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<uint8_t>((i * 7) & 0x7f);
  return b;
}

void AppendProjection(std::vector<FixtureTensor>& out, const std::string& proj,
                      int64_t n, int64_t k, Arm arm, uint32_t seed) {
  if (arm == Arm::kExl3 || arm == Arm::kBf16) {
    const Exl3Proj p = MakeProj(k, n, seed);
    if (arm == Arm::kExl3) {
      AppendExl3(out, proj, p);
    } else {
      AppendDecodedBf16(out, proj, p);
    }
    return;
  }
  if (arm == Arm::kFp8) {
    out.push_back({proj + ".weight", "F8_E4M3", {n, k}, Fp8Walk(n * k)});
    out.push_back({proj + ".weight_scale", "F32", {}, Bytes32F({0.25F})});
    out.push_back({proj + ".input_scale", "F32", {}, Bytes32F({0.5F})});
    return;
  }
  out.push_back({proj + ".weight", "U8", {n, k / 2}, Fp8Walk(n * k / 2)});
  out.push_back({proj + ".weight_scale", "F8_E4M3", {n, k / 16},
                 Fp8Walk(n * k / 16)});
  out.push_back({proj + ".weight_scale_2", "F32", {}, Bytes32F({0.5F})});
}

// How the GDN linear-attention tower is written. THREE forms, not two —
// MODEL-QWEN35-GDN-EXL3 (#2495 item 4).
enum class GdnArm {
  // Independent random BF16. What every non-EXL3 case wants: a tower that is
  // simply not quantized, so a case about the dense arms is not also a case
  // about this one.
  kPlainBf16,
  // `.trellis` + `.suh` + `.svh` + `.mul1`, which is what
  // `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` ships on all 48 of its
  // `linear_attention` layers.
  kExl3,
  // The SAME trellis bytes as `kExl3`, decoded into BF16 `.weight` tensors.
  // This is the comparison twin: it makes the forward check an EQUIVALENCE
  // between two readings of one set of bytes rather than a tolerance between
  // two different models.
  kDecodedBf16,
};

// `f16_remainder` is the artifact's own asymmetry and NOT a free choice.
// `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` stores `in_proj_a` and `in_proj_b` at
// **F16**, while storing `conv1d`, `norm`, `A_log` and `dt_bias` beside them at
// **BF16** — read from its safetensors headers by range request. exllamav3
// keeps the unquantized LINEAR remainder at fp16 because it runs the linear in
// fp16, and leaves the rest alone. `LoadMergedBf16RawNK` refused F16 by name,
// so without this the tower is blocked a second time on a tensor nothing
// quantized. Writing these BF16 here would leave that path unreached.
//
// The decoded twin writes `F32ToBF16(F16ToF32(v))` of the SAME values
// (`RemainderBytes`'s `as_f16=false` arm), so after loading the two checkpoints
// hold bit-identical `in_proj_ba` and the forward comparison is measuring the
// three projections and nothing else.
void AppendGdnLayer(std::vector<FixtureTensor>& out, const std::string& base,
                    const Geometry& g, GdnArm arm, bool f16_remainder,
                    uint32_t seed = 4000) {
  const std::string la = base + "linear_attn.";
  const Exl3Proj qkv = MakeProj(g.hidden, g.conv_dim, seed + 1);
  const Exl3Proj z = MakeProj(g.hidden, g.value_dim, seed + 2);
  const Exl3Proj o = MakeProj(g.value_dim, g.hidden, seed + 3);
  if (arm == GdnArm::kExl3) {
    AppendExl3(out, la + "in_proj_qkv", qkv);
    AppendExl3(out, la + "in_proj_z", z);
    AppendExl3(out, la + "out_proj", o);
  } else if (arm == GdnArm::kDecodedBf16) {
    AppendDecodedBf16(out, la + "in_proj_qkv", qkv);
    AppendDecodedBf16(out, la + "in_proj_z", z);
    AppendDecodedBf16(out, la + "out_proj", o);
  } else {
    out.push_back({la + "in_proj_qkv.weight", "BF16", {g.conv_dim, g.hidden},
                   RemainderBytes({g.conv_dim, g.hidden}, seed + 1, 0.1F, false)});
    out.push_back({la + "in_proj_z.weight", "BF16", {g.value_dim, g.hidden},
                   RemainderBytes({g.value_dim, g.hidden}, seed + 2, 0.1F, false)});
    out.push_back({la + "out_proj.weight", "BF16", {g.hidden, g.value_dim},
                   RemainderBytes({g.hidden, g.value_dim}, seed + 3, 0.1F, false)});
  }
  // KEYED ON THE GDN ARM, not on the dense one. F16 `in_proj_a`/`in_proj_b` is
  // what accompanies a QUANTIZED tower in the artifact; a checkpoint whose
  // dense half is EXL3 while this tower is plain bf16 stores them BF16 like
  // every other bf16 tower, and the loader's non-EXL3 rung refuses F16 there by
  // name. The decoded twin also writes BF16, and `RemainderBytes` derives it
  // from the SAME fp16 values, so the two hold bit-identical `in_proj_ba` after
  // loading.
  const bool ba_f16 = f16_remainder && arm == GdnArm::kExl3;
  const char* ba_dt = ba_f16 ? "F16" : "BF16";
  out.push_back({la + "in_proj_b.weight", ba_dt, {g.num_v_heads, g.hidden},
                 RemainderBytes({g.num_v_heads, g.hidden}, seed + 4, 0.1F,
                                ba_f16)});
  out.push_back({la + "in_proj_a.weight", ba_dt, {g.num_v_heads, g.hidden},
                 RemainderBytes({g.num_v_heads, g.hidden}, seed + 5, 0.1F,
                                ba_f16)});
  out.push_back({la + "conv1d.weight", "BF16", {g.conv_dim, 1, g.conv_k},
                 RemainderBytes({g.conv_dim, 1, g.conv_k}, seed + 6, 0.1F, false)});
  out.push_back({la + "A_log", "BF16", {g.num_v_heads},
                 RemainderBytes({g.num_v_heads}, seed + 7, 0.5F, false)});
  out.push_back({la + "dt_bias", "BF16", {g.num_v_heads},
                 RemainderBytes({g.num_v_heads}, seed + 8, 0.5F, false)});
  out.push_back({la + "norm.weight", "BF16", {g.v_head_dim},
                 RemainderBytes({g.v_head_dim}, seed + 9, 0.5F, false)});
}

// The whole checkpoint. Layer 0 is `linear_attention`, layer 1 is
// `full_attention`, and BOTH carry a dense MLP -- the same split the real model
// has, at 48/16 instead of 1/1.
std::vector<FixtureTensor> DenseCheckpoint(
    Arm arm, const Geometry& g = {}, GdnArm gdn = GdnArm::kPlainBf16,
    const std::vector<std::string>& layer_types = {"linear_attention",
                                                   "full_attention"}) {
  // `f16` is a property of the checkpoint FAMILY, not of a single tensor: an
  // EXL3 artifact stores its whole unquantized remainder at F16.
  const bool f16 = (arm == Arm::kExl3);
  const char* rdt = f16 ? "F16" : "BF16";
  std::vector<FixtureTensor> t;
  t.push_back({"model.embed_tokens.weight", rdt, {g.vocab, g.hidden},
               RemainderBytes({g.vocab, g.hidden}, 11, 0.5F, f16)});
  t.push_back({"model.norm.weight", rdt, {g.hidden},
               RemainderBytes({g.hidden}, 12, 0.5F, f16)});

  for (int layer = 0; layer < static_cast<int>(layer_types.size()); ++layer) {
    const std::string base = "model.layers." + std::to_string(layer) + ".";
    const uint32_t seed = 1000 + static_cast<uint32_t>(layer) * 500;
    t.push_back({base + "input_layernorm.weight", rdt, {g.hidden},
                 RemainderBytes({g.hidden}, seed + 1, 0.5F, f16)});
    t.push_back({base + "post_attention_layernorm.weight", rdt, {g.hidden},
                 RemainderBytes({g.hidden}, seed + 2, 0.5F, f16)});
    if (layer_types[static_cast<size_t>(layer)] == "linear_attention") {
      AppendGdnLayer(t, base, g, gdn, f16, 4000 + static_cast<uint32_t>(layer) * 20);
    } else {
      const std::string sa = base + "self_attn.";
      AppendProjection(t, sa + "q_proj", g.q_n, g.hidden, arm, seed + 10);
      AppendProjection(t, sa + "k_proj", g.kv_n, g.hidden, arm, seed + 20);
      AppendProjection(t, sa + "v_proj", g.kv_n, g.hidden, arm, seed + 30);
      AppendProjection(t, sa + "o_proj", g.hidden, g.attn_out_k, arm, seed + 40);
      t.push_back({sa + "q_norm.weight", rdt, {g.head_dim},
                   RemainderBytes({g.head_dim}, seed + 50, 0.5F, f16)});
      t.push_back({sa + "k_norm.weight", rdt, {g.head_dim},
                   RemainderBytes({g.head_dim}, seed + 60, 0.5F, f16)});
    }
    const std::string mlp = base + "mlp.";
    AppendProjection(t, mlp + "gate_proj", g.inter, g.hidden, arm, seed + 101);
    AppendProjection(t, mlp + "up_proj", g.inter, g.hidden, arm, seed + 102);
    AppendProjection(t, mlp + "down_proj", g.hidden, g.inter, arm, seed + 103);
  }

  // The output head. An EXL3 artifact ships a REAL quantized head, and it is
  // preferred over a tied embedding table even when the config declares
  // `tie_word_embeddings: true` (`llama_weights.cpp:152-161`).
  AppendProjection(t, "lm_head", g.vocab, g.hidden, arm, 9001);
  return t;
}

// ---------------------------------------------------------------------------
// The `mtp.*` draft head — MODEL-QWEN35-EXL3-HEAD (#2495 item 5)
// ---------------------------------------------------------------------------
//
// NINE tensors of `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` live under `mtp.` and are
// quantized: `fc`, the four `self_attn` projections and the three `mlp` ones.
// Read from `model-00002-of-00002.safetensors`'s own header by HTTP range
// request (no payload downloaded): each carries `.trellis` / `.suh` / `.svh` /
// `.mul1`, and each trellis last dim is 64 -- FOUR bits, which is what
// `quantization_config.mtp_bits` also says and which the loader reads from the
// tensor rather than from that key.
//
// TWO asymmetries with the body, and neither is a free choice here.
//
// `mtp.fc` is [K=2H, N=H] -- the fc-cat projection consumes the concatenated
// [embed_norm | target_norm] row. The bf16 twin therefore writes a torch Linear
// `.weight` of [N=H, K=2H], which is the orientation `AppendDecodedBf16`
// already produces from the same `Exl3Proj`.
//
// The `mtp.*` REMAINDER is BF16, not F16. The body's remainder is F16 because
// exllamav3 runs its linears in fp16, and the parent row had to widen the
// readers for it; the artifact stores `mtp.norm`, `mtp.pre_fc_norm_*`, the two
// layernorms and the two per-head norms as BF16 regardless. Writing them F16
// here would test a widening this row does not do and the artifact does not
// need.
void AppendMtp(std::vector<FixtureTensor>& t, const Geometry& g, Arm arm) {
  const Exl3Proj fc = MakeProj(2 * g.hidden, g.hidden, 7101);
  if (arm == Arm::kExl3) {
    AppendExl3(t, "mtp.fc", fc);
  } else {
    AppendDecodedBf16(t, "mtp.fc", fc);
  }
  // BF16 in BOTH arms, and from ONE set of values, so the two checkpoints hold
  // bit-identical norms and the forward comparison measures the projections.
  t.push_back({"mtp.pre_fc_norm_embedding.weight", "BF16", {g.hidden},
               RemainderBytes({g.hidden}, 7110, 0.5F, false)});
  t.push_back({"mtp.pre_fc_norm_hidden.weight", "BF16", {g.hidden},
               RemainderBytes({g.hidden}, 7111, 0.5F, false)});
  t.push_back({"mtp.norm.weight", "BF16", {g.hidden},
               RemainderBytes({g.hidden}, 7112, 0.5F, false)});

  const std::string base = "mtp.layers.0.";
  t.push_back({base + "input_layernorm.weight", "BF16", {g.hidden},
               RemainderBytes({g.hidden}, 7120, 0.5F, false)});
  t.push_back({base + "post_attention_layernorm.weight", "BF16", {g.hidden},
               RemainderBytes({g.hidden}, 7121, 0.5F, false)});
  const std::string sa = base + "self_attn.";
  AppendProjection(t, sa + "q_proj", g.q_n, g.hidden, arm, 7130);
  AppendProjection(t, sa + "k_proj", g.kv_n, g.hidden, arm, 7140);
  AppendProjection(t, sa + "v_proj", g.kv_n, g.hidden, arm, 7150);
  AppendProjection(t, sa + "o_proj", g.hidden, g.attn_out_k, arm, 7160);
  t.push_back({sa + "q_norm.weight", "BF16", {g.head_dim},
               RemainderBytes({g.head_dim}, 7170, 0.5F, false)});
  t.push_back({sa + "k_norm.weight", "BF16", {g.head_dim},
               RemainderBytes({g.head_dim}, 7171, 0.5F, false)});
  const std::string mlp = base + "mlp.";
  AppendProjection(t, mlp + "gate_proj", g.inter, g.hidden, arm, 7180);
  AppendProjection(t, mlp + "up_proj", g.inter, g.hidden, arm, 7181);
  AppendProjection(t, mlp + "down_proj", g.hidden, g.inter, arm, 7182);
}

// A whole checkpoint that ALSO carries the draft head, which is what the
// published artifact is.
std::vector<FixtureTensor> DenseCheckpointWithMtp(Arm arm, const Geometry& g) {
  std::vector<FixtureTensor> t = DenseCheckpoint(arm, g);
  AppendMtp(t, g, arm);
  return t;
}

// ---------------------------------------------------------------------------
// Driving the production paths
// ---------------------------------------------------------------------------

Qwen3_5DenseWeights LoadDense(const TempCheckpoint& ckpt,
                              const HfConfig& config) {
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(ckpt.path()));
  return vllm::LoadQwen3_5Dense(shards, config, /*load_queue=*/nullptr);
}

std::string LoadFailure(const TempCheckpoint& ckpt, const HfConfig& config) {
  try {
    const Qwen3_5DenseWeights weights = LoadDense(ckpt, config);
    (void)weights;
    return "";
  } catch (const std::exception& e) {
    return e.what();
  }
}

// The REGISTRY seam, the entry point a user actually arrives through.
std::string RegistryLoadFailure(const TempCheckpoint& ckpt,
                                const HfConfig& config) {
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(ckpt.path()));
  const ModelSource source = ModelSource::FromSafetensors(shards);
  try {
    std::unique_ptr<vllm::LoadedModel> model =
        vllm::ModelRegistry::Load(config, source);
    (void)model;
    return "";
  } catch (const std::exception& e) {
    return e.what();
  }
}

bool Names(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

vt::Queue Cpu() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

// The caches one prefill needs. Sized in floats throughout; a bf16 KV cache
// simply uses half of each buffer.
struct CachePool {
  std::vector<std::vector<float>> attn_buf;
  std::vector<std::vector<float>> ssm_buf;
  std::vector<std::vector<float>> conv_buf;
  std::vector<PagedKvCache> attn_kv;
  std::vector<GdnStateCache> gdn_state;

  // ONE cache per layer of each kind. The 3:1 fixture has six `linear_attention`
  // layers and two `full_attention` ones, and a pool sized for the 1/1 fixture
  // would index past its own storage on the seventh.
  CachePool(const HfConfig& c, const Geometry& g, int64_t nb, int64_t bs,
            int n_gdn = 1, int n_attn = 1) {
    // Every buffer is emplaced BEFORE any `.data()` is taken below: a later
    // emplace reallocates the outer vector and would dangle the pointers.
    for (int i = 0; i < n_gdn; ++i) {
      ssm_buf.emplace_back(
          static_cast<size_t>(nb * g.num_v_heads * g.v_head_dim *
                              c.linear_key_head_dim),
          0.0F);
      conv_buf.emplace_back(
          static_cast<size_t>(nb * g.conv_dim * (g.conv_k - 1)), 0.0F);
    }
    for (int i = 0; i < n_attn; ++i) {
      attn_buf.emplace_back(
          static_cast<size_t>(nb * 2 * bs * c.num_key_value_heads * c.head_dim),
          0.0F);
    }
    const vt::Device cpu{vt::DeviceType::kCPU, 0};
    for (auto& b : attn_buf) {
      PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = DType::kBF16;
      kv.num_blocks = nb;
      kv.block_size = bs;
      kv.num_kv_heads = c.num_key_value_heads;
      kv.head_size = c.head_dim;
      attn_kv.push_back(kv);
    }
    for (size_t i = 0; i < ssm_buf.size(); ++i) {
      GdnStateCache gs;
      gs.ssm_state = vt::Tensor::Contiguous(
          ssm_buf[i].data(), DType::kF32, cpu,
          {nb, g.num_v_heads, g.v_head_dim, c.linear_key_head_dim});
      gs.conv_state = vt::Tensor::Contiguous(conv_buf[i].data(), DType::kF32,
                                             cpu, {nb, g.conv_dim, g.conv_k - 1});
      gdn_state.push_back(gs);
    }
  }
};

CommonAttentionMetadata PrefillAttnMeta(int64_t T, int64_t block_size) {
  CommonAttentionMetadata m;
  m.num_reqs = 1;
  m.num_actual_tokens = static_cast<int>(T);
  m.query_start_loc = {0, static_cast<int32_t>(T)};
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens = {static_cast<int32_t>(T)};
  m.seq_lens_cpu = m.seq_lens;
  m.max_query_len = static_cast<int>(T);
  m.max_seq_len = static_cast<int>(T);
  m.block_table_num_cols = 1;
  m.block_table_tensor = {0};
  for (int64_t t = 0; t < T; ++t)
    m.slot_mapping.push_back(t % block_size);
  m.causal = true;
  return m;
}

GDNAttentionMetadata PrefillGdnMeta(int64_t T) {
  GDNAttentionMetadata g;
  g.num_prefills = 1;
  g.num_prefill_tokens = static_cast<int>(T);
  g.num_decodes = 0;
  g.num_decode_tokens = 0;
  g.num_actual_tokens = static_cast<int>(T);
  g.has_initial_state = std::vector<uint8_t>{0};
  g.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  g.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  g.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  g.prefill_state_indices = std::vector<int32_t>{0};
  g.prefill_has_initial_state = std::vector<uint8_t>{0};
  const auto conv =
      vllm::v1::ComputeCausalConv1dMetadata(*g.non_spec_query_start_loc);
  g.batch_ptr = conv.batch_ptr;
  g.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  return g;
}

// THE PRODUCTION ENTRY POINT. `ModelRegistry::Forward` over the weights the
// LOADER produced -- not a hand-constructed container, and not
// `Qwen3_5DenseModel::Forward` -- so deleting an EXL3 call site in the dense
// forward reds this rather than leaving a class-level gate green.
std::vector<float> RegistryForward(const HfConfig& c, const Geometry& g,
                                   const Qwen3_5DenseWeights& w) {
  const int64_t T = 4;
  const std::vector<int32_t> ids = {5, 9, 2, 17};
  const std::vector<int32_t> pos = {0, 1, 2, 3};
  const std::vector<int32_t> logits_indices;
  int n_gdn = 0, n_attn = 0;
  for (const std::string& lt : c.layer_types)
    (lt == "linear_attention" ? n_gdn : n_attn) += 1;
  CachePool pool(c, g, /*nb=*/4, /*bs=*/8, n_gdn, n_attn);
  const CommonAttentionMetadata am = PrefillAttnMeta(T, 8);
  const GDNAttentionMetadata gm = PrefillGdnMeta(T);
  vt::Queue q = Cpu();
  std::unique_ptr<vllm::LoadedModel> model =
      vllm::BorrowQwen3_5DenseLoadedModel(w);
  ModelForwardInput in{ids, pos, am,       gm, pool.attn_kv,
                       pool.gdn_state,     c,  q,  logits_indices};
  in.num_reqs = 1;
  in.gather_logits = false;
  const ForwardLogits out = ModelRegistry::Forward(*model, in);
  REQUIRE(out.host.size() == static_cast<size_t>(T * c.vocab_size));
  return out.host;
}

// ---------------------------------------------------------------------------
// Driving the MTP draft — MODEL-QWEN35-EXL3-HEAD (#2495 item 5)
// ---------------------------------------------------------------------------

// The draft's KV, one block, sized for the ONE full-attention layer the MTP head
// is (`qwen3_5_mtp.py:105-112`).
struct DraftKv {
  std::vector<uint16_t> buf;
  PagedKvCache kv;
  DraftKv(const HfConfig& c, int64_t num_blocks, int64_t block_size) {
    buf.assign(static_cast<size_t>(num_blocks * 2 * block_size *
                                   c.num_key_value_heads * c.head_dim),
               0);
    kv.data = buf.data();
    kv.dtype = DType::kBF16;
    kv.num_blocks = num_blocks;
    kv.block_size = block_size;
    kv.num_kv_heads = c.num_key_value_heads;
    kv.head_size = c.head_dim;
  }
};

// THE PRODUCTION DRAFT PATH. `ForwardPaged` is what the spec-decode runner calls
// (`Qwen3_5MTPModel::ForwardPaged`, reached from the MTP speculator);
// `ComputeLogits` applies the TARGET's shared head to its output. Both are
// entered over weights the LOADER produced from a checkpoint on disk, so a
// deleted call site reds this rather than leaving a class-level gate green.
std::vector<float> MtpPagedLogits(const HfConfig& c, const Geometry& g,
                                  const Qwen3_5MTPWeights& mtp,
                                  const Qwen3_5DenseWeights& target,
                                  vt::Backend& backend, vt::Queue& queue) {
  const int64_t T = 4;
  const std::vector<int32_t> ids = {5, 9, 2, 17};
  const std::vector<int32_t> pos = {0, 1, 2, 3};
  const Qwen3_5MTPModel model(mtp, target, c);
  // The target's hidden-state tap, bf16 [T,H]. Deterministic and identical for
  // both arms, so the comparison below measures the head and nothing else.
  std::vector<uint16_t> hidden(static_cast<size_t>(T * g.hidden));
  Rng r;
  r.s = 8181u;
  for (auto& v : hidden) v = vt::F32ToBF16(r.unit() * 0.25F);
  const vt::Tensor th = vt::Tensor::Contiguous(
      hidden.data(), DType::kBF16, queue.device, {T, g.hidden});

  DraftKv pool(c, /*num_blocks=*/1, /*block_size=*/8);
  CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = static_cast<int>(T);
  am.query_start_loc = {0, static_cast<int32_t>(T)};
  am.query_start_loc_cpu = am.query_start_loc;
  am.seq_lens = {static_cast<int32_t>(T)};
  am.seq_lens_cpu = am.seq_lens;
  am.max_query_len = static_cast<int>(T);
  am.max_seq_len = static_cast<int>(T);
  am.block_table_num_cols = 1;
  am.block_table_tensor = {0};
  for (int64_t t = 0; t < T; ++t) am.slot_mapping.push_back(t);
  am.causal = true;

  const vllm::Qwen3_5MTPHiddenStates hs =
      model.ForwardPaged(ids, pos, th, am, pool.kv, queue);
  const ForwardLogits logits = model.ComputeLogits(hs.tensor, queue);
  std::vector<float> host(static_cast<size_t>(logits.rows) * logits.vocab);
  backend.Copy(queue, host.data(), logits.device_tensor.data,
               host.size() * sizeof(float));
  backend.Synchronize(queue);
  return host;
}

double RelRms(const std::vector<float>& got, const std::vector<float>& ref) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = static_cast<double>(got[i]) - ref[i];
    num += d * d;
    den += static_cast<double>(ref[i]) * ref[i];
  }
  REQUIRE(den > 0.0);  // a zeroed forward satisfies any relative bound
  return std::sqrt(num / den);
}

}  // namespace

// G1 -------------------------------------------------------------------------
TEST_CASE("qwen3_5 exl3: an EXL3 dense checkpoint LOADS, with the codebook and the width it stores") {
  const Geometry g;
  const TempCheckpoint ckpt(DenseCheckpoint(Arm::kExl3, g));
  const HfConfig c = DenseConfig(/*with_exl3_quant_config=*/true);

  const Qwen3_5DenseWeights w = LoadDense(ckpt, c);
  REQUIRE(w.layers.size() == 2);
  CHECK(w.precision.activation == DType::kBF16);
  CHECK(w.embed_tokens.dtype == DType::kBF16);

  // The full-attention tower.
  const vllm::FullAttnLayerWeights& a = w.layers[1].attn;
  REQUIRE(a.IsExl3());
  const Exl3Weight* attn[4] = {&a.q_proj_exl3, &a.k_proj_exl3, &a.v_proj_exl3,
                               &a.o_proj_exl3};
  for (const Exl3Weight* e : attn) {
    // THE ASSERTION A SHAPE CHECK CANNOT MAKE. The wrong multiplier yields a
    // codebook with the SAME DISTRIBUTION and no relation to the right one, so
    // the weight decodes to the correct RMS and uncorrelated values and every
    // geometry check below still passes.
    CHECK(e->codebook == kCodebook);
    CHECK(e->Bits() == kBits);
  }
  CHECK(a.q_proj_exl3.InFeatures() == g.hidden);
  CHECK(a.q_proj_exl3.OutFeatures() == g.q_n);
  CHECK(a.k_proj_exl3.OutFeatures() == g.kv_n);
  CHECK(a.v_proj_exl3.OutFeatures() == g.kv_n);
  CHECK(a.o_proj_exl3.InFeatures() == g.attn_out_k);
  CHECK(a.o_proj_exl3.OutFeatures() == g.hidden);

  // The dense MLP, on BOTH layers -- the GDN layer has one too.
  for (const auto& layer : w.layers) {
    REQUIRE(layer.mlp.IsExl3());
    CHECK(layer.mlp.gate_proj_exl3.codebook == kCodebook);
    CHECK(layer.mlp.gate_proj_exl3.InFeatures() == g.hidden);
    CHECK(layer.mlp.gate_proj_exl3.OutFeatures() == g.inter);
    CHECK(layer.mlp.up_proj_exl3.OutFeatures() == g.inter);
    CHECK(layer.mlp.down_proj_exl3.InFeatures() == g.inter);
    CHECK(layer.mlp.down_proj_exl3.OutFeatures() == g.hidden);
  }

  // The head (#2495 item 5), preferred over the tied embedding table.
  REQUIRE_FALSE(w.lm_head_exl3.Empty());
  CHECK(w.lm_head_exl3.codebook == kCodebook);
  CHECK(w.lm_head_exl3.Bits() == kBits);
  CHECK(w.lm_head_exl3.InFeatures() == g.hidden);
  CHECK(w.lm_head_exl3.OutFeatures() == g.vocab);
  CHECK_FALSE(w.tied_lm_head);

  // EXACTLY ONE representation is populated. Without this the case passes for a
  // loader that filled the trellis fields AND dequantized into the bf16 ones,
  // which is numerically better and therefore invisible to every value check.
  CHECK(w.lm_head.Empty());
  CHECK(w.lm_head_fp4.Empty());
  CHECK(a.q_proj.Empty());
  CHECK(a.k_proj.Empty());
  CHECK(a.v_proj.Empty());
  CHECK(a.o_proj.Empty());
  CHECK(a.q_proj_fp8.Empty());
  CHECK(a.q_proj_fp4.Empty());
  CHECK(a.q_proj_fp8_block.Empty());
  CHECK(w.layers[1].mlp.gate_up_proj.Empty());
  CHECK(w.layers[1].mlp.down_proj.Empty());

  // The trellis is still trellis BYTES. A silent dequant-at-load would be
  // numerically better and invisible above; only the byte count sees it.
  const int64_t k = g.hidden, n = g.inter;
  CHECK(w.layers[1].mlp.gate_proj_exl3.trellis.bytes.size() ==
        static_cast<size_t>(k / 16) * static_cast<size_t>(n / 16) * 32 * kBits);
  CHECK(w.layers[1].mlp.gate_proj_exl3.trellis.dtype == DType::kI8);

  // The GDN tower loaded BF16 beside it: the dense half being quantized does
  // not disturb the half that is not. This fixture writes an UNQUANTIZED GDN
  // tower, so the trellis fields must stay empty even though the checkpoint is
  // an EXL3 one -- the arm is resolved per projection from the tensors, never
  // from the checkpoint family.
  CHECK(w.layers[0].is_linear_attention);
  CHECK_FALSE(w.layers[0].gdn.in_proj_qkvz.Empty());
  CHECK_FALSE(w.layers[0].gdn.out_proj.Empty());
  CHECK_FALSE(w.layers[0].gdn.IsExl3());
  CHECK(w.layers[0].gdn.in_proj_qkv_exl3.Empty());
  CHECK(w.layers[0].gdn.in_proj_z_exl3.Empty());
  CHECK(w.layers[0].gdn.out_proj_exl3.Empty());

  // And the direct-device staging path must not claim this model.
  CHECK_FALSE(vllm::IsPlainBf16Qwen3_5Dense(w));

  // The registry seam accepts it too, which is the entry a user arrives through.
  CHECK(RegistryLoadFailure(ckpt, c).empty());
}

// G2 -------------------------------------------------------------------------
TEST_CASE("qwen3_5 exl3: ModelRegistry::Forward REACHES the arm and agrees with the decoded twin") {
  const Geometry g;
  const HfConfig c = DenseConfig(/*with_exl3_quant_config=*/true);
  const HfConfig c_plain = DenseConfig(/*with_exl3_quant_config=*/false);

  const TempCheckpoint exl3_ckpt(DenseCheckpoint(Arm::kExl3, g));
  const TempCheckpoint bf16_ckpt(DenseCheckpoint(Arm::kBf16, g));
  const Qwen3_5DenseWeights we = LoadDense(exl3_ckpt, c);
  const Qwen3_5DenseWeights wb = LoadDense(bf16_ckpt, c_plain);
  REQUIRE(we.layers[1].attn.IsExl3());
  REQUIRE_FALSE(wb.layers[1].attn.IsExl3());

  const std::vector<float> le = RegistryForward(c, g, we);
  const std::vector<float> lb = RegistryForward(c_plain, g, wb);
  REQUIRE(le.size() == lb.size());
  REQUIRE(!le.empty());

  for (const float x : le) REQUIRE(std::isfinite(x));
  for (const float x : lb) REQUIRE(std::isfinite(x));

  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < le.size(); ++i) {
    const double d = static_cast<double>(le[i]) - lb[i];
    num += d * d;
    den += static_cast<double>(lb[i]) * lb[i];
  }
  // NOT VACUOUS: a forward returning zeros satisfies any relative bound whose
  // reference is also zero. Asserted BEFORE the ratio is formed.
  REQUIRE(den > 0.0);
  const double rel = std::sqrt(num / den);
  MESSAGE("exl3 forward vs decoded-bf16 forward: rel_rms = ", rel);
  // A BOUND, not an equality: the two arms are the same weights through
  // different kernels. EXL3 rides the Hadamards on the activations while the
  // decoded twin has them baked in, and the twin rounds the decode to bf16.
  CHECK(rel <= 5.0e-2);
}

// G3 -------------------------------------------------------------------------
TEST_CASE("qwen3_5 exl3: an EXL3 GDN tower LOADS, and ModelRegistry::Forward REACHES it") {
  // THE CASE THIS ROW EXISTS FOR, and it replaces a refusal. Until #2495 item 4
  // the three GDN projections were refused by name, because nothing in
  // `ProjectGdnQkvz`/`ProjectGdnOut` consumed an `Exl3Weight` and the
  // alternative was half-loading. 48 of the real model's 64 layers are
  // `linear_attention`, so the refusal was the whole checkpoint.
  const Geometry g;
  const HfConfig c = DenseConfig(/*with_exl3_quant_config=*/true);
  const TempCheckpoint ckpt(DenseCheckpoint(Arm::kExl3, g, GdnArm::kExl3));

  const Qwen3_5DenseWeights w = LoadDense(ckpt, c);
  REQUIRE(w.layers.size() == 2);
  REQUIRE(w.layers[0].is_linear_attention);
  const vllm::GdnLayerWeights& gdn = w.layers[0].gdn;
  REQUIRE(gdn.IsExl3());

  // The codebook is the half a shape check cannot see: the wrong multiplier
  // decodes to the correct RMS and uncorrelated values.
  const Exl3Weight* projs[3] = {&gdn.in_proj_qkv_exl3, &gdn.in_proj_z_exl3,
                                &gdn.out_proj_exl3};
  for (const Exl3Weight* e : projs) {
    CHECK(e->codebook == kCodebook);
    CHECK(e->Bits() == kBits);
  }
  // THE ARTIFACT'S OWN GEOMETRY, not the bf16 convention. The checkpoint ships
  // `in_proj_qkv` and `in_proj_z` SEPARATELY, so their N are conv_dim and
  // value_dim rather than one merged conv_dim+value_dim owner.
  CHECK(gdn.in_proj_qkv_exl3.InFeatures() == g.hidden);
  CHECK(gdn.in_proj_qkv_exl3.OutFeatures() == g.conv_dim);
  CHECK(gdn.in_proj_z_exl3.InFeatures() == g.hidden);
  CHECK(gdn.in_proj_z_exl3.OutFeatures() == g.value_dim);
  CHECK(gdn.out_proj_exl3.InFeatures() == g.value_dim);
  CHECK(gdn.out_proj_exl3.OutFeatures() == g.hidden);

  // EXACTLY ONE representation. Without this the case passes for a loader that
  // filled the trellis fields AND dequantized into the bf16 ones, which is
  // numerically better and therefore invisible to every value check below.
  CHECK(gdn.in_proj_qkvz.Empty());
  CHECK(gdn.in_proj_qkv.Empty());
  CHECK(gdn.in_proj_z.Empty());
  CHECK(gdn.out_proj.Empty());
  CHECK(gdn.in_proj_qkv_fp8.Empty());
  CHECK(gdn.in_proj_z_fp8.Empty());
  CHECK(gdn.out_proj_fp8.Empty());
  CHECK(gdn.out_proj_fp4.Empty());
  CHECK(gdn.in_proj_qkv_fp8_block.Empty());
  CHECK(gdn.out_proj_fp8_block.Empty());

  // The trellis is still trellis BYTES. A silent dequant-at-load would be
  // numerically better and invisible above; only the byte count sees it.
  CHECK(gdn.in_proj_qkv_exl3.trellis.dtype == DType::kI8);
  CHECK(gdn.in_proj_qkv_exl3.trellis.bytes.size() ==
        static_cast<size_t>(g.hidden / 16) *
            static_cast<size_t>(g.conv_dim / 16) * 32 * kBits);

  // The tensors NOTHING quantizes still loaded beside the trellis. `in_proj_ba`
  // is the one that mattered: the artifact stores `in_proj_a`/`in_proj_b` at
  // F16, which `LoadMergedBf16RawNK` refused by name.
  CHECK_FALSE(gdn.in_proj_ba.Empty());
  CHECK(gdn.in_proj_ba.nk);
  CHECK(gdn.in_proj_ba.shape[0] == 2 * g.num_v_heads);
  CHECK(gdn.in_proj_ba.shape[1] == g.hidden);
  CHECK_FALSE(gdn.conv1d_weight.Empty());
  CHECK_FALSE(gdn.a_log.Empty());
  CHECK_FALSE(gdn.dt_bias.Empty());
  CHECK_FALSE(gdn.norm_weight.Empty());

  // The registry seam accepts it, which is the entry a user arrives through.
  CHECK(RegistryLoadFailure(ckpt, c).empty());

  // The DIRECT-DEVICE staging path must not claim this model. That path stages
  // `OwnedTensor` members only, so it would walk past the trellis, `suh` and
  // `svh` and leave the tower unstaged -- silently, because every shape check
  // still passes and the bf16 fields it does stage are empty.
  CHECK_FALSE(vllm::IsPlainBf16Qwen3_5Dense(w));

  // THE PRODUCTION ENTRY POINT, over the weights the LOADER produced. Deleting
  // an EXL3 call site in `ProjectGdnQkvz` or `ProjectGdnOut` does not merely
  // move these numbers: the arm falls through to a bf16 field an EXL3 load
  // leaves EMPTY and `dense_attn::ResidentWeight` refuses it by name, so the
  // case reds either way. A unit test that built a `GdnLayerWeights` by hand
  // would prove the struct holds an `Exl3Weight` and never that anything
  // reaches it (`.agents/reachability.md`).
  const HfConfig c_plain = DenseConfig(/*with_exl3_quant_config=*/false);
  const TempCheckpoint twin(
      DenseCheckpoint(Arm::kBf16, g, GdnArm::kDecodedBf16));
  const Qwen3_5DenseWeights wt = LoadDense(twin, c_plain);
  REQUIRE_FALSE(wt.layers[0].gdn.IsExl3());
  REQUIRE_FALSE(wt.layers[0].gdn.in_proj_qkvz.Empty());

  const std::vector<float> le = RegistryForward(c, g, w);
  const std::vector<float> lb = RegistryForward(c_plain, g, wt);
  REQUIRE(le.size() == lb.size());
  REQUIRE(!le.empty());
  for (const float x : le) REQUIRE(std::isfinite(x));
  for (const float x : lb) REQUIRE(std::isfinite(x));

  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < le.size(); ++i) {
    const double d = static_cast<double>(le[i]) - lb[i];
    num += d * d;
    den += static_cast<double>(lb[i]) * lb[i];
  }
  // NOT VACUOUS: a forward returning zeros satisfies any relative bound whose
  // reference is also zero. Asserted BEFORE the ratio is formed.
  REQUIRE(den > 0.0);
  const double rel = std::sqrt(num / den);
  MESSAGE("exl3 GDN forward vs decoded-bf16 twin: rel_rms = ", rel);
  // A BOUND, not an equality: the two arms are the same weights through
  // different kernels. EXL3 rides the Hadamards on the activations while the
  // twin has them baked in, and the twin rounds the decode to bf16. The GDN
  // recurrence then carries that difference through a scan, so this is looser
  // than the dense arms' bound in G2 and deliberately so.
  CHECK(rel <= 1.0e-1);
}

// G3b ------------------------------------------------------------------------
TEST_CASE("qwen3_5 exl3: the artifact's 3:1 layer_types pattern loads and forwards") {
  // The 1/1 fixture proves the ARM. It does not prove the PATTERN, and the
  // pattern is where a per-layer resolver goes wrong: `Mia-AiLab/Qwen3.8-27B-
  // EXL3-3.5bpw` is 64 layers of three `linear_attention` then one
  // `full_attention`, so a loader that keyed the arm off the layer INDEX rather
  // than off the tensors would still pass at 1/1.
  const Geometry g;
  const std::vector<std::string> types = Ratio3To1LayerTypes(8);
  REQUIRE(types.size() == 8);
  CHECK(types[0] == "linear_attention");
  CHECK(types[2] == "linear_attention");
  CHECK(types[3] == "full_attention");
  CHECK(types[7] == "full_attention");

  const HfConfig c = DenseConfig(/*with_exl3_quant_config=*/true, types);
  const TempCheckpoint ckpt(
      DenseCheckpoint(Arm::kExl3, g, GdnArm::kExl3, types));

  const Qwen3_5DenseWeights w = LoadDense(ckpt, c);
  REQUIRE(w.layers.size() == 8);
  int gdn_exl3 = 0, attn_exl3 = 0;
  for (size_t i = 0; i < w.layers.size(); ++i) {
    if (types[i] == "linear_attention") {
      REQUIRE(w.layers[i].is_linear_attention);
      CHECK(w.layers[i].gdn.IsExl3());
      CHECK(w.layers[i].gdn.in_proj_qkv_exl3.codebook == kCodebook);
      gdn_exl3 += 1;
    } else {
      REQUIRE_FALSE(w.layers[i].is_linear_attention);
      CHECK(w.layers[i].attn.IsExl3());
      attn_exl3 += 1;
    }
    CHECK(w.layers[i].mlp.IsExl3());
  }
  // The 3:1 ratio itself, so a fixture that silently became all-GDN or
  // all-attention cannot read as a pass.
  CHECK(gdn_exl3 == 6);
  CHECK(attn_exl3 == 2);

  CHECK(RegistryLoadFailure(ckpt, c).empty());
  const std::vector<float> logits = RegistryForward(c, g, w);
  REQUIRE(!logits.empty());
  for (const float x : logits) REQUIRE(std::isfinite(x));
  // NOT VACUOUS: an all-zero forward is finite too.
  double energy = 0.0;
  for (const float x : logits) energy += static_cast<double>(x) * x;
  CHECK(energy > 0.0);
}

// G3c ------------------------------------------------------------------------
TEST_CASE("qwen3_5 exl3: an EXL3 GDN tower whose geometry disagrees is REFUSED by name") {
  // The rung asserts its own geometry rather than trusting it, and this case is
  // what stops that assertion being decoration. A trellis is self-consistent at
  // more than one reading of its own shape -- the bytes are there either way and
  // only the values come out wrong -- so a projection whose width disagrees with
  // the rest of the layer is invisible to every check that reads the trellis
  // alone. `conv1d.weight`'s channel count is the checkpoint's OWN statement of
  // `conv_dim`, so the two are compared against each other and not against the
  // config.
  const Geometry g;
  const HfConfig c = DenseConfig(/*with_exl3_quant_config=*/true);

  std::vector<FixtureTensor> tensors =
      DenseCheckpoint(Arm::kExl3, g, GdnArm::kExl3);
  // Widen conv1d ALONE, leaving `in_proj_qkv.trellis` at conv_dim. Nothing else
  // in the checkpoint moves.
  const std::string conv = "model.layers.0.linear_attn.conv1d.weight";
  bool patched = false;
  for (FixtureTensor& t : tensors) {
    if (t.name != conv) continue;
    t.shape = {g.conv_dim + 128, 1, g.conv_k};
    t.bytes = RemainderBytes(t.shape, 77, 0.1F, false);
    patched = true;
  }
  // The fixture must actually carry the tensor this case patches; a renamed
  // tensor would otherwise make the case pass by testing nothing.
  REQUIRE(patched);

  const TempCheckpoint ckpt(tensors);
  const std::string message = LoadFailure(ckpt, c);
  REQUIRE_FALSE(message.empty());
  CHECK(Names(message, "conv_dim"));
  CHECK(Names(message, "linear_attn"));
  // NOT a missing-tensor accident: every tensor is present and only one width
  // disagrees.
  CHECK_FALSE(Names(message, "tensor not found"));

  // AND IT IS NOT UNCONDITIONAL: the same fixture unpatched loads, so the case
  // measures the geometry check rather than the presence of an EXL3 GDN tower.
  const TempCheckpoint ok(DenseCheckpoint(Arm::kExl3, g, GdnArm::kExl3));
  CHECK(LoadFailure(ok, c).empty());
}

// G4 -------------------------------------------------------------------------
TEST_CASE("qwen3_5 exl3: the bf16, per-tensor FP8 and NVFP4 arms are UNCHANGED") {
  // A fourth rung in a presence-based resolver is exactly where an existing arm
  // gets mis-selected, and the mis-selection is silent: each of these
  // checkpoints still loads and still produces plausible numbers. So the arm
  // each one lands in is asserted directly.
  const Geometry g;
  const HfConfig c = DenseConfig(/*with_exl3_quant_config=*/false);

  SUBCASE("bf16 stays bf16") {
    const TempCheckpoint ckpt(DenseCheckpoint(Arm::kBf16, g));
    const Qwen3_5DenseWeights w = LoadDense(ckpt, c);
    const vllm::FullAttnLayerWeights& a = w.layers[1].attn;
    CHECK_FALSE(a.IsExl3());
    CHECK_FALSE(w.layers[1].mlp.IsExl3());
    CHECK_FALSE(w.layers[0].mlp.IsExl3());
    CHECK(w.lm_head_exl3.Empty());
    CHECK_FALSE(a.q_proj.Empty());
    CHECK(a.q_proj.nk);
    CHECK_FALSE(a.o_proj.Empty());
    CHECK_FALSE(w.layers[1].mlp.gate_up_proj.Empty());
    CHECK_FALSE(w.layers[1].mlp.down_proj.Empty());
    CHECK_FALSE(w.lm_head.Empty());
    // The direct-device staging path still claims a plain bf16 model, which is
    // the predicate the EXL3 rung had to be added to WITHOUT disturbing.
    CHECK(vllm::IsPlainBf16Qwen3_5Dense(w));
  }

  SUBCASE("per-tensor FP8 stays per-tensor FP8") {
    const TempCheckpoint ckpt(DenseCheckpoint(Arm::kFp8, g));
    const Qwen3_5DenseWeights w = LoadDense(ckpt, c);
    const vllm::FullAttnLayerWeights& a = w.layers[1].attn;
    CHECK_FALSE(a.IsExl3());
    CHECK_FALSE(a.q_proj_fp8.Empty());
    CHECK(a.q_proj_fp8.n == g.q_n);
    CHECK(a.q_proj_fp8.k == g.hidden);
    CHECK(a.q_proj.Empty());
    CHECK(a.q_proj_fp8_block.Empty());
    CHECK(a.q_proj_fp4.Empty());
    CHECK_FALSE(vllm::IsPlainBf16Qwen3_5Dense(w));
  }

  SUBCASE("ModelOpt NVFP4 stays NVFP4") {
    const TempCheckpoint ckpt(DenseCheckpoint(Arm::kNvfp4, g));
    const Qwen3_5DenseWeights w = LoadDense(ckpt, c);
    const vllm::FullAttnLayerWeights& a = w.layers[1].attn;
    CHECK_FALSE(a.IsExl3());
    CHECK_FALSE(a.q_proj_fp4.Empty());
    CHECK(a.q_proj_fp4.n == g.q_n);
    CHECK(a.q_proj_fp4.k == g.hidden);
    CHECK(a.q_proj.Empty());
    CHECK(a.q_proj_fp8.Empty());
    CHECK_FALSE(w.layers[1].mlp.gate_proj_fp4.Empty());
    CHECK_FALSE(vllm::IsPlainBf16Qwen3_5Dense(w));
  }

  // MODEL-QWEN35-GDN-EXL3 (#2495 item 4). THE GDN TOWER, asserted for the same
  // reason the dense arms are: a fourth rung in a presence-based resolver is
  // exactly where an existing arm gets mis-selected, and a mis-selection here is
  // silent -- every one of these checkpoints still loads and still produces
  // plausible numbers. Each case names the field the tower MUST land in and
  // checks the trellis fields are empty.
  SUBCASE("the bf16 GDN tower stays bf16") {
    const TempCheckpoint ckpt(DenseCheckpoint(Arm::kBf16, g));
    const Qwen3_5DenseWeights w = LoadDense(ckpt, c);
    const vllm::GdnLayerWeights& gdn = w.layers[0].gdn;
    CHECK_FALSE(gdn.IsExl3());
    CHECK(gdn.in_proj_qkv_exl3.Empty());
    CHECK(gdn.in_proj_z_exl3.Empty());
    CHECK(gdn.out_proj_exl3.Empty());
    // The MERGED bf16 owner, which is the shape vLLM's single in_proj_qkvz GEMM
    // needs and the one the EXL3 arm deliberately does not build.
    CHECK_FALSE(gdn.in_proj_qkvz.Empty());
    CHECK(gdn.in_proj_qkvz.nk);
    CHECK(gdn.in_proj_qkvz.shape[0] == g.conv_dim + g.value_dim);
    CHECK_FALSE(gdn.out_proj.Empty());
    CHECK_FALSE(gdn.in_proj_ba.Empty());
  }

  SUBCASE("an EXL3 GDN tower alone is not a plain bf16 model") {
    // THE TERM THE OTHER CASES CANNOT ISOLATE. `IsPlainBf16Qwen3_5Dense`
    // returns false for the published artifact because its MLP is EXL3, so a
    // missing GDN term is invisible on every checkpoint that quantizes both
    // halves. This checkpoint quantizes ONLY the GDN tower, so the GDN term is
    // the single thing that can answer.
    const TempCheckpoint ckpt(DenseCheckpoint(Arm::kBf16, g, GdnArm::kExl3));
    const Qwen3_5DenseWeights w = LoadDense(ckpt, c);
    REQUIRE(w.layers[0].gdn.IsExl3());
    REQUIRE_FALSE(w.layers[1].mlp.IsExl3());
    REQUIRE_FALSE(w.layers[1].attn.IsExl3());
    REQUIRE(w.lm_head_exl3.Empty());
    CHECK_FALSE(vllm::IsPlainBf16Qwen3_5Dense(w));
  }

  SUBCASE("an EXL3 dense half does not quantize an unquantized GDN tower") {
    // The separability item 3 was built on, asserted from the other side: an
    // EXL3 CHECKPOINT whose GDN tower ships bf16 must still load that tower
    // bf16. A resolver that keyed the arm off the checkpoint family rather than
    // off the tensors would fail here and nowhere else.
    const HfConfig ce = DenseConfig(/*with_exl3_quant_config=*/true);
    const TempCheckpoint ckpt(
        DenseCheckpoint(Arm::kExl3, g, GdnArm::kPlainBf16));
    const Qwen3_5DenseWeights w = LoadDense(ckpt, ce);
    CHECK(w.layers[1].attn.IsExl3());
    CHECK_FALSE(w.layers[0].gdn.IsExl3());
    CHECK_FALSE(w.layers[0].gdn.in_proj_qkvz.Empty());
  }
}

// ===========================================================================
// MODEL-QWEN35-EXL3-HEAD (#2495 item 5 remainder, item 6, #2569)
//
// ONE representation decision, three readers. The target's own logits already
// compute against the trellis (the parent row). These cases are the other two:
// the MTP draft, which SHARES the target's head, and the DFlash/DSpark draft's
// shared-head read. Plus the `mtp.*` half of item 5, which is nine quantized
// tensors the loader refused as "expected BF16".
// ===========================================================================

// H1 + H2 + H3 ---------------------------------------------------------------
TEST_CASE("qwen3_5 exl3: XPU MTP loads all seven norms at FP16 boundaries") {
  const Geometry g;
  const HfConfig c = DenseConfig(/*with_exl3_quant_config=*/true);
  auto tensors = DenseCheckpointWithMtp(Arm::kExl3, g);
  SUBCASE("checkpoint BF16 remainders") {}
  SUBCASE("checkpoint F16 and F32 remainders") {
    for (auto& tensor : tensors) {
      if (tensor.name.rfind("mtp.", 0) != 0 || tensor.dtype != "BF16") continue;
      const bool half = tensor.name.find("self_attn.") != std::string::npos;
      const size_t count = tensor.bytes.size() / 2;
      std::vector<uint8_t> converted(count * (half ? 2 : 4));
      for (size_t i = 0; i < count; ++i) {
        uint16_t bits;
        std::memcpy(&bits, tensor.bytes.data() + 2 * i, 2);
        const float value = vt::BF16ToF32(bits);
        if (half) {
          bits = vt::F32ToF16(value);
          std::memcpy(converted.data() + 2 * i, &bits, 2);
        } else {
          std::memcpy(converted.data() + 4 * i, &value, 4);
        }
      }
      tensor.dtype = half ? "F16" : "F32";
      tensor.bytes = std::move(converted);
    }
  }
  const TempCheckpoint ckpt(tensors);
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(ckpt.path()));
  const auto weights = vllm::LoadQwen3_5MTP(
      shards, c, Qwen3_5MTPKind::kDense, vt::DeviceType::kXPU);
  REQUIRE(weights.IsExl3());
  REQUIRE(weights.dense_layers.size() == 1);
  const auto& layer = weights.dense_layers[0];
  const std::vector<std::pair<std::string, const vllm::OwnedTensor*>> norms = {
      {"mtp.pre_fc_norm_embedding.weight", &weights.pre_fc_norm_embedding},
      {"mtp.pre_fc_norm_hidden.weight", &weights.pre_fc_norm_hidden},
      {"mtp.norm.weight", &weights.final_norm},
      {"mtp.layers.0.input_layernorm.weight", &layer.input_layernorm},
      {"mtp.layers.0.post_attention_layernorm.weight", &layer.post_attention_layernorm},
      {"mtp.layers.0.self_attn.q_norm.weight", &layer.attn.q_norm},
      {"mtp.layers.0.self_attn.k_norm.weight", &layer.attn.k_norm}};
  for (const auto& [name, owned] : norms) {
    CAPTURE(name);
    REQUIRE(owned->dtype == DType::kF16);
    const auto& source = shards[0].Get(name);
    const size_t count = owned->bytes.size() / 2;
    bool exact = true;
    for (size_t i = 0; i < count; ++i) {
      uint16_t got, bits;
      std::memcpy(&got, owned->bytes.data() + 2 * i, 2);
      uint16_t expected;
      if (source.dtype == "F32") {
        float value;
        std::memcpy(&value, source.data + 4 * i, 4);
        expected = vt::F32ToF16(value);
      } else {
        std::memcpy(&bits, source.data + 2 * i, 2);
        expected = source.dtype == "F16" ? bits
                                        : vt::F32ToF16(vt::BF16ToF32(bits));
      }
      exact &= got == expected;
    }
    CHECK(exact);
  }
  for (const auto* packed : {&weights.fc_exl3, &layer.attn.q_proj_exl3,
       &layer.attn.k_proj_exl3, &layer.attn.v_proj_exl3, &layer.attn.o_proj_exl3,
       &layer.mlp.gate_proj_exl3, &layer.mlp.up_proj_exl3, &layer.mlp.down_proj_exl3}) {
    CHECK_FALSE(packed->Empty());
  }
  CHECK(weights.fc.Empty());
  CHECK(layer.attn.q_proj.Empty());
  CHECK(layer.mlp.gate_proj.Empty());
}

TEST_CASE("qwen3_5 exl3: the mtp.* draft head LOADS, and its PAGED forward "
          "agrees with the decoded twin") {
  const Geometry g;
  HfConfig c = DenseConfig(/*with_exl3_quant_config=*/true);
  // The EXL3 target's explicit precision policy requires the checkpoint's
  // declared FP32 recurrence even when this test only exercises the draft.
  c.mamba_ssm_dtype = "float32";
  const HfConfig c_plain = DenseConfig(/*with_exl3_quant_config=*/false);

  const TempCheckpoint exl3_ckpt(DenseCheckpointWithMtp(Arm::kExl3, g));
  const TempCheckpoint bf16_ckpt(DenseCheckpointWithMtp(Arm::kBf16, g));

  std::vector<SafetensorsFile> se;
  se.push_back(SafetensorsFile::Open(exl3_ckpt.path()));
  std::vector<SafetensorsFile> sb;
  sb.push_back(SafetensorsFile::Open(bf16_ckpt.path()));

  // THE PRODUCTION LOADER ENTRY. `model_loader.cpp`'s `maybe_attach_mtp` calls
  // exactly this overload on exactly these shards.
  const Qwen3_5MTPWeights me =
      vllm::LoadQwen3_5MTP(se, c, Qwen3_5MTPKind::kDense);
  const Qwen3_5MTPWeights mb =
      vllm::LoadQwen3_5MTP(sb, c_plain, Qwen3_5MTPKind::kDense);

  // H1. The arm is active, and it is keyed on the TRELLIS.
  REQUIRE(me.IsExl3());
  REQUIRE_FALSE(mb.IsExl3());
  CHECK(me.fc.Empty());
  CHECK_FALSE(mb.fc.Empty());
  CHECK(mb.fc.nk);

  // The codebook is the half a shape check cannot see: the wrong multiplier
  // decodes to the right RMS and uncorrelated values.
  CHECK(me.fc_exl3.codebook == kCodebook);
  CHECK(me.fc_exl3.Bits() == kBits);
  // [K=2H, N=H] -- the fc-cat orientation, which the torch Linear stores the
  // other way round. Both directions are asserted so a transposed read cannot
  // pass on a square-ish fixture.
  CHECK(me.fc_exl3.InFeatures() == 2 * g.hidden);
  CHECK(me.fc_exl3.OutFeatures() == g.hidden);

  REQUIRE(me.dense_layers.size() == 1);
  CHECK(me.dense_layers[0].attn.IsExl3());
  CHECK(me.dense_layers[0].mlp.IsExl3());
  CHECK(me.dense_layers[0].attn.q_proj_exl3.codebook == kCodebook);
  CHECK(me.dense_layers[0].mlp.down_proj_exl3.Bits() == kBits);
  CHECK_FALSE(mb.dense_layers[0].attn.IsExl3());
  CHECK_FALSE(mb.dense_layers[0].mlp.IsExl3());

  // The remainder is BF16 on BOTH arms, which is what the artifact stores under
  // `mtp.` even though its BODY remainder is F16.
  CHECK(me.final_norm.dtype == DType::kBF16);
  CHECK(std::memcmp(me.final_norm.bytes.data(), mb.final_norm.bytes.data(),
                    me.final_norm.bytes.size()) == 0);

  // H3. The TARGET's head is a trellis too, and the draft SHARES it. On this
  // artifact there is no `lm_head.weight` at all, so the trellis is the only
  // head there is.
  const Qwen3_5DenseWeights te = LoadDense(exl3_ckpt, c);
  const Qwen3_5DenseWeights tb = LoadDense(bf16_ckpt, c_plain);
  REQUIRE_FALSE(te.lm_head_exl3.Empty());
  REQUIRE(te.lm_head.Empty());
  REQUIRE(tb.lm_head_exl3.Empty());
  REQUIRE_FALSE(tb.lm_head.Empty());

  // H2. Both drafts run their PAGED forward and apply their target's head.
  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();
  const std::vector<float> le = MtpPagedLogits(c, g, me, te, backend, queue);
  const std::vector<float> lb =
      MtpPagedLogits(c_plain, g, mb, tb, backend, queue);
  backend.DestroyQueue(queue);

  REQUIRE(le.size() == lb.size());
  REQUIRE(!le.empty());
  for (const float x : le) REQUIRE(std::isfinite(x));
  for (const float x : lb) REQUIRE(std::isfinite(x));

  const double rel = RelRms(le, lb);
  MESSAGE("mtp exl3 paged logits vs decoded-bf16 twin: rel_rms = ", rel);
  // A BOUND, not an equality, for the reason the dense case gives: EXL3 rides
  // the Hadamards on the activations while the decoded twin has them baked in,
  // and the twin rounds the decode to bf16. The head is 6 bits in the artifact
  // and 4 here, so this bound is the fixture's and not the artifact's.
  CHECK(rel <= 5.0e-2);
}

// H4 -------------------------------------------------------------------------
TEST_CASE("qwen3_5 exl3: the DFlash draft's SHARED head read takes the trellis "
          "arm, and ONLY it") {
  const Geometry g;
  const HfConfig c = DenseConfig(/*with_exl3_quant_config=*/true);
  const TempCheckpoint ckpt(DenseCheckpointWithMtp(Arm::kExl3, g));
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(ckpt.path()));

  vllm::Qwen3DFlashWeights w;
  vllm::LoadDflashSharedLmHead(shards, &w.lm_head, &w.lm_head_fp4,
                               &w.lm_head_exl3);

  // EXACTLY ONE owner, which is the rule the target's own head already obeys.
  REQUIRE_FALSE(w.lm_head_exl3.Empty());
  CHECK(w.lm_head.Empty());
  CHECK(w.lm_head_fp4.Empty());
  CHECK(w.lm_head_exl3.codebook == kCodebook);
  CHECK(w.lm_head_exl3.Bits() == kBits);
  CHECK(w.lm_head_exl3.InFeatures() == g.hidden);
  CHECK(w.lm_head_exl3.OutFeatures() == g.vocab);

  // The SAME tensor the target loads for itself, byte for byte. Two readings of
  // one head that disagreed would be the "two descriptions of one rule" defect.
  const Qwen3_5DenseWeights target = LoadDense(ckpt, c);
  REQUIRE(target.lm_head_exl3.trellis.bytes.size() ==
          w.lm_head_exl3.trellis.bytes.size());
  CHECK(std::memcmp(target.lm_head_exl3.trellis.bytes.data(),
                    w.lm_head_exl3.trellis.bytes.data(),
                    w.lm_head_exl3.trellis.bytes.size()) == 0);

  // THE REFUSAL DOES NOT MOVE. `lm_head_dequantized` means "this head was
  // WIDENED on the way in", which only the GGUF arm can do; a packed trellis is
  // the opposite state and must not set it. Deleting the guard's trigger is what
  // #1314's fresh review caught, so the guard is exercised in both directions.
  w.conv_taps = 2;
  REQUIRE(w.IsDflash2());
  CHECK_FALSE(w.lm_head_dequantized);
  CHECK_NOTHROW(vllm::RefuseQuantizedDflash2LmHead(w));
  vllm::Qwen3DFlashWeights widened = w;
  widened.lm_head_dequantized = true;
  CHECK_THROWS(vllm::RefuseQuantizedDflash2LmHead(widened));
}

// H5 -------------------------------------------------------------------------
TEST_CASE("qwen3_5 exl3: a lane with no trellis owner REFUSES an EXL3 target "
          "head by name") {
  // The DSpark shape: its backbone holds ONE bf16 `lm_head`, so it passes
  // `head_fp4 = nullptr` and `head_exl3 = nullptr`. On this artifact there is no
  // `lm_head.weight` to fall back to.
  //
  // THIS IS #2569's CASE. Before this row the read fell off the end of its bf16
  // loop and RETURNED, leaving every owner empty and no sentence anywhere; the
  // refusal was left to whichever caller happened to test emptiness, and the one
  // that does names bf16 tensors -- the wrong reason for a head that is present
  // and simply not dense. Deleting the throw makes this case pass silently.
  const Geometry g;
  const TempCheckpoint ckpt(DenseCheckpointWithMtp(Arm::kExl3, g));
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(ckpt.path()));

  vllm::OwnedTensor bf16_only;
  std::string message;
  try {
    vllm::LoadDflashSharedLmHead(shards, &bf16_only, /*head_fp4=*/nullptr,
                                 /*head_exl3=*/nullptr);
  } catch (const std::exception& e) {
    message = e.what();
  }
  REQUIRE_FALSE(message.empty());
  CHECK(Names(message, "matched NO arm"));
  CHECK(Names(message, "NOT OFFERED"));
  CHECK(Names(message, "2569"));
  CHECK(bf16_only.Empty());
}

// H5b ------------------------------------------------------------------------
TEST_CASE("qwen3_5 exl3: a head that matches no arm at all REFUSES, with every "
          "owner offered") {
  // The pure #2569 case: all three owners available and none of them selectable.
  // A half-written EXL3 head is the shape -- `.suh` present, `.trellis` absent
  // -- which `Linear.is_exl3_storage` correctly declines, leaving nothing.
  const Geometry g;
  std::vector<FixtureTensor> t = DenseCheckpoint(Arm::kExl3, g);
  t.erase(std::remove_if(t.begin(), t.end(),
                         [](const FixtureTensor& f) {
                           return f.name.rfind("lm_head.", 0) == 0;
                         }),
          t.end());
  t.push_back({"lm_head.suh", "F16", {g.hidden},
               RemainderBytes({g.hidden}, 9101, 1.0F, true)});
  const TempCheckpoint ckpt(t);
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(ckpt.path()));

  vllm::OwnedTensor bf16;
  vllm::Nvfp4Weight packed;
  Exl3Weight trellis;
  std::string message;
  try {
    vllm::LoadDflashSharedLmHead(shards, &bf16, &packed, &trellis);
  } catch (const std::exception& e) {
    message = e.what();
  }
  REQUIRE_FALSE(message.empty());
  CHECK(Names(message, "matched NO arm"));
  CHECK(bf16.Empty());
  CHECK(packed.Empty());
  CHECK(trellis.Empty());
}

TEST_CASE("qwen3_5 exl3: FP16 remainders require the explicit layer policy") {
  const Geometry g;
  const TempCheckpoint ckpt(DenseCheckpoint(Arm::kExl3, g, GdnArm::kExl3));
  const auto shard = SafetensorsFile::Open(ckpt.path());
  const auto get = [&](const std::string& name) -> const vllm::StTensor& {
    return shard.Get(name);
  };
  const auto has = [&](const std::string& name) {
    const auto& names = shard.Names();
    return std::find(names.begin(), names.end(), name) != names.end();
  };
  for (const bool fp16 : {false, true}) {
    const auto dtype = fp16 ? DType::kF16 : DType::kBF16;
    const auto gdn = vllm::LoadQwen3_5DenseLayer(
        get, has, "linear_attention", 0, "model.",
        vllm::Fp8BlockQuantConfig{}, fp16);
    const auto attn = vllm::LoadQwen3_5DenseLayer(
        get, has, "full_attention", 1, "model.",
        vllm::Fp8BlockQuantConfig{}, fp16);
    CHECK(gdn.input_layernorm.dtype == dtype);
    CHECK(gdn.post_attention_layernorm.dtype == dtype);
    CHECK(gdn.gdn.in_proj_ba.dtype == dtype);
    CHECK(gdn.gdn.conv1d_weight.dtype == dtype);
    CHECK(gdn.gdn.norm_weight.dtype == dtype);
    CHECK(attn.input_layernorm.dtype == dtype);
    CHECK(attn.attn.q_norm.dtype == dtype);
    CHECK(attn.attn.k_norm.dtype == dtype);
    CHECK(gdn.gdn.IsExl3());
    CHECK(attn.attn.IsExl3());
  }
}
