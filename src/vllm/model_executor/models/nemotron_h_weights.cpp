// Nemotron-H: config descent, the on-disk weight name map, and the WEIGHT LOAD
// that materializes it. See nemotron_h.h for the port anchors and
// nemotron_h_loader.h for the MIXED_PRECISION scheme table, why each weight is
// held in the format it SHIPS in, and what is deferred by name.
//
// The loader lives in this TU rather than a `nemotron_h_loader.cpp` of its own
// for two reasons. It is where every other architecture puts its weight load —
// `llama_weights.cpp`, `laguna_weights.cpp`, `deepseek_v2_weights.cpp` are all
// loaders — and the enumeration it must agree with tensor-for-tensor is right
// here, so the two cannot drift into separate files with separate ideas of what
// the checkpoint ships. A new TU would also have meant editing the top-level
// `CMakeLists.txt`, which `check-doc-checkpoint.py:83-93` classifies as a
// user-facing build entrypoint owing a `docs/USAGE.md` change; nothing here is
// user-facing yet (the forward is the host reference, W6 owns the runner path),
// so that edit would have documented nothing.
#include "vllm/model_executor/models/nemotron_h.h"
#include "vllm/model_executor/models/nemotron_h_loader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vt/unaligned.h"  // LoadUnaligned — safetensors offsets carry no alignment

namespace vllm {
namespace {

using nlohmann::json;

[[noreturn]] void Refuse(const std::string& detail) {
  throw std::runtime_error("NemotronHForCausalLM: " + detail);
}

// `raw` is the whole config document; every NemotronH-specific key is read from
// it because HfConfig types only the shared subset.
const json& Raw(const HfConfig& config) { return config.raw; }

bool Has(const json& doc, const char* key) {
  return doc.contains(key) && !doc.at(key).is_null();
}

int64_t GetInt(const json& doc, const char* key, int64_t fallback) {
  if (!Has(doc, key)) return fallback;
  const json& v = doc.at(key);
  if (!v.is_number_integer() && !v.is_number_unsigned()) {
    Refuse(std::string("config key '") + key + "' must be an integer");
  }
  return v.get<int64_t>();
}

double GetDouble(const json& doc, const char* key, double fallback) {
  if (!Has(doc, key)) return fallback;
  const json& v = doc.at(key);
  if (!v.is_number()) {
    Refuse(std::string("config key '") + key + "' must be a number");
  }
  return v.get<double>();
}

bool GetBool(const json& doc, const char* key, bool fallback) {
  if (!Has(doc, key)) return fallback;
  const json& v = doc.at(key);
  if (!v.is_boolean()) {
    Refuse(std::string("config key '") + key + "' must be a boolean");
  }
  return v.get<bool>();
}

std::string GetString(const json& doc, const char* key,
                      const std::string& fallback) {
  if (!Has(doc, key)) return fallback;
  const json& v = doc.at(key);
  if (!v.is_string()) {
    Refuse(std::string("config key '") + key + "' must be a string");
  }
  return v.get<std::string>();
}

// The legacy-alias reads of the `mamba_*` SCALARS
// (configuration_nemotron_h.py:145-155). Upstream is:
//
//   self.n_groups = kwargs.pop("mamba_n_groups") if "mamba_n_groups" in kwargs
//                   else self.n_groups
//
// The dataclass field already holds the modern value (or the class default) by
// the time `__post_init__` runs, and the legacy alias OVERWRITES it whenever it
// is present. So the precedence is LEGACY > modern > class default — the
// opposite of what reads naturally, and the opposite of the SCHEDULE pair below.
//
// Verified by RUNNING transformers @ 7d06b1a5 rather than by reading it:
//   NemotronHConfig(n_groups=8, mamba_n_groups=4,
//                   conv_kernel=4, mamba_d_conv=7) -> n_groups=4, conv_kernel=7
//
// A checkpoint carrying only the alias must likewise not silently deserialize
// to the class default. No released checkpoint ships both spellings of one
// field, so this is a mirroring obligation, not a live defect — pinned by
// "when BOTH spellings ship, the precedence is upstream's and it is PER-FAMILY"
// so it cannot drift back.
int64_t GetIntAliased(const json& doc, const char* key, const char* legacy,
                      int64_t fallback) {
  if (Has(doc, legacy)) return GetInt(doc, legacy, fallback);
  return GetInt(doc, key, fallback);
}

double GetDoubleAliased(const json& doc, const char* key, const char* legacy,
                        double fallback) {
  if (Has(doc, legacy)) return GetDouble(doc, legacy, fallback);
  return GetDouble(doc, key, fallback);
}

bool GetBoolAliased(const json& doc, const char* key, const char* legacy,
                    bool fallback) {
  if (Has(doc, legacy)) return GetBool(doc, legacy, fallback);
  return GetBool(doc, key, fallback);
}

// The four block spellings, in enum order. The single source of truth for both
// directions of the name<->enum map, so a fifth block kind cannot be added with
// a refusal message that still lists four.
constexpr NemotronHBlock kAllBlocks[] = {
    NemotronHBlock::kMamba, NemotronHBlock::kAttention, NemotronHBlock::kMoe,
    NemotronHBlock::kMlp};

NemotronHBlock BlockFromName(const std::string& name) {
  for (NemotronHBlock block : kAllBlocks) {
    if (name == NemotronHBlockName(block)) return block;
  }
  // Mirror of validate_layer_type (configuration_nemotron_h.py:195-204).
  std::string expected;
  for (NemotronHBlock block : kAllBlocks) {
    if (!expected.empty()) expected += ", ";
    expected += NemotronHBlockName(block);
  }
  Refuse("layers_block_type contains the unsupported block type '" + name +
         "' (expected one of " + expected + ")");
}

NemotronHBlock BlockFromPatternChar(char c) {
  // configuration_nemotron_h.py:265-268.
  switch (c) {
    case 'M':
      return NemotronHBlock::kMamba;
    case 'E':
      return NemotronHBlock::kMoe;
    case '*':
      return NemotronHBlock::kAttention;
    case '-':
      return NemotronHBlock::kMlp;
    default:
      Refuse(std::string("hybrid_override_pattern contains '") + c +
             "' (expected one of M, E, *, -)");
  }
}

// One layer schedule, resolved with upstream's precedence: the explicit list,
// else the legacy pattern string, else the class default. `list_key` /
// `pattern_key` are the modern/legacy pair, `fallback` the class default.
//
// NOTE the polarity, which is the OPPOSITE of the `mamba_*` scalars above and
// is deliberate on both sides. configuration_nemotron_h.py:158-165:
//
//   if "hybrid_override_pattern" in kwargs:
//       pattern = kwargs.pop("hybrid_override_pattern")
//       if self.layer_types is None:
//           self.layer_types = self._pattern_to_list(pattern)
//
// the legacy pattern is consulted ONLY when the modern list is absent, so here
// MODERN wins; :176-184 does the same for the MTP pair. Verified by running
// transformers @ 7d06b1a5: `NemotronHConfig(layer_types=['mamba','mamba'],
// hybrid_override_pattern='*-')` -> `['mamba','mamba']`. Do not "unify" the two
// families — upstream genuinely disagrees with itself here.
std::vector<NemotronHBlock> ResolveSchedule(
    const json& doc, const char* list_key, const char* pattern_key,
    const std::vector<NemotronHBlock>& fallback) {
  if (Has(doc, list_key)) {
    const json& v = doc.at(list_key);
    if (!v.is_array()) {
      Refuse(std::string("config key '") + list_key + "' must be a list");
    }
    std::vector<NemotronHBlock> out;
    out.reserve(v.size());
    for (const json& entry : v) {
      if (!entry.is_string()) {
        Refuse(std::string("config key '") + list_key +
               "' must be a list of strings");
      }
      out.push_back(BlockFromName(entry.get<std::string>()));
    }
    return out;
  }
  if (Has(doc, pattern_key)) {
    std::vector<NemotronHBlock> out;
    for (char c : GetString(doc, pattern_key, "")) {
      out.push_back(BlockFromPatternChar(c));
    }
    return out;
  }
  return fallback;
}

NemotronHQuantSurface ResolveQuantSurface(const json& doc) {
  NemotronHQuantSurface q;
  if (!Has(doc, "quantization_config")) return q;
  const json& qc = doc.at("quantization_config");
  if (!qc.is_object()) Refuse("quantization_config must be an object");
  q.present = true;
  q.quant_method = GetString(qc, "quant_method", "");
  q.quant_algo = GetString(qc, "quant_algo", "");
  // W3 resolves NO per-module algorithm — that is W1 (#517 W1). It only refuses
  // a producer whose on-disk companion-tensor layout we have not read, because
  // enumerating the wrong companions is exactly the silent-wrong-bytes failure
  // a token gate cannot see.
  if (q.quant_method != "modelopt") {
    Refuse("quantization_config.quant_method '" + q.quant_method +
           "' is not implemented (this row ports ModelOpt MIXED_PRECISION; see "
           ".agents/specs/nemotron-h-model.md W1)");
  }
  if (Has(qc, "kv_cache_scheme")) {
    const json& kv = qc.at("kv_cache_scheme");
    const bool fp8 = kv.is_object() && GetInt(kv, "num_bits", 0) == 8 &&
                     GetString(kv, "type", "") == "float";
    if (!fp8) {
      Refuse("quantization_config.kv_cache_scheme is not the fp8 scheme this "
             "row implements (expected num_bits 8, type float)");
    }
    q.fp8_kv_cache = true;
  }
  if (Has(qc, "ignore")) {
    const json& ignore = qc.at("ignore");
    if (!ignore.is_array()) Refuse("quantization_config.ignore must be a list");
    for (const json& entry : ignore) {
      if (!entry.is_string()) continue;
      const std::string name = entry.get<std::string>();
      // The released list carries the wildcard `mtp*`, which is what leaves the
      // whole MTP tower bf16 with no scale companions.
      if (name.rfind("mtp", 0) == 0) q.mtp_ignored = true;
    }
  }
  return q;
}

// ─── enumeration helpers ─────────────────────────────────────────────────────

void Claim(std::vector<NemotronHTensor>& out, std::string name,
           std::string consumer) {
  out.push_back(NemotronHTensor{std::move(name), std::move(consumer)});
}

// An NVFP4 W4A16 group-16 weight: the packed nibbles, the per-16-block e4m3
// scale, and the fp32 global scale (spec §1; config_groups group_1 on the
// released checkpoint). WHICH kernel consumes them is W1/W4, not W3.
void ClaimNvfp4(std::vector<NemotronHTensor>& out, const std::string& prefix,
                const std::string& consumer, bool quantized) {
  Claim(out, prefix + ".weight", consumer);
  if (!quantized) return;
  Claim(out, prefix + ".weight_scale", consumer + ".weight_scale[nvfp4]");
  Claim(out, prefix + ".weight_scale_2", consumer + ".weight_scale_2[nvfp4]");
}

// An FP8 W8A8 static-scaled projection: the e4m3 weight, its fp32 weight scale
// and the fp32 static input scale (config_groups group_0, 46 targets).
// `quantized` gates the companions exactly as `ClaimNvfp4` does: an UNQUANTIZED
// producer ships the bare bf16 weight and no scales at all.
void ClaimFp8(std::vector<NemotronHTensor>& out, const std::string& prefix,
              const std::string& consumer, bool quantized) {
  Claim(out, prefix + ".weight", consumer);
  if (!quantized) return;
  Claim(out, prefix + ".weight_scale", consumer + ".weight_scale[fp8]");
  Claim(out, prefix + ".input_scale", consumer + ".input_scale[fp8]");
}

// One Mamba2 mixer (MambaMixer2, nemotron_h.py:373-389). `use_conv_bias` and
// `mamba_proj_bias` gate the two optional biases exactly as upstream does, and
// `quantized` gates the in/out projection scale companions. Released bf16
// NemotronH safetensors checkpoints ship NO `quantization_config` and no
// `mixer.{in,out}_proj.{weight_scale,input_scale}`; hard-coding the FP8 pair
// here enumerated 92 tensors such a checkpoint does not have (23 mamba blocks x
// 2 projections x 2 companions).
void ClaimMamba(std::vector<NemotronHTensor>& out, const NemotronHParams& p,
                const std::string& mixer, bool quantized) {
  ClaimFp8(out, mixer + ".in_proj", "mamba2.in_proj", quantized);
  ClaimFp8(out, mixer + ".out_proj", "mamba2.out_proj", quantized);
  if (p.mamba_proj_bias) {
    Claim(out, mixer + ".in_proj.bias", "mamba2.in_proj.bias");
    Claim(out, mixer + ".out_proj.bias", "mamba2.out_proj.bias");
  }
  Claim(out, mixer + ".conv1d.weight", "mamba2.conv1d");
  if (p.use_conv_bias) Claim(out, mixer + ".conv1d.bias", "mamba2.conv1d.bias");
  // `A_log` on disk; the mapper renames it to `A` in-module
  // (nemotron_h.py:719).
  Claim(out, mixer + ".A_log", "mamba2.A_log");
  Claim(out, mixer + ".D", "mamba2.D");
  Claim(out, mixer + ".dt_bias", "mamba2.dt_bias");
  Claim(out, mixer + ".norm.weight", "mamba2.gated_rmsnorm");
}

// One GQA attention mixer (NemotronHAttention, nemotron_h.py:503). q/k/v ship
// SEPARATE on disk; upstream stacks them into `qkv_proj` at load
// (hf_to_vllm_mapper orig_to_new_stacked, :719-723).
void ClaimAttention(std::vector<NemotronHTensor>& out,
                    const NemotronHParams& p, const std::string& mixer,
                    bool fp8_kv) {
  for (const char* proj : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
    Claim(out, mixer + "." + proj + ".weight", std::string("attn.") + proj);
    if (p.attention_bias) {
      Claim(out, mixer + "." + proj + ".bias",
            std::string("attn.") + proj + ".bias");
    }
  }
  if (fp8_kv) {
    Claim(out, mixer + ".k_proj.k_scale", "attn.k_scale[fp8-kv]");
    Claim(out, mixer + ".v_proj.v_scale", "attn.v_scale[fp8-kv]");
  }
}

// One non-gated relu² MoE block (NemotronHMoE, nemotron_h.py:126-256). There is
// NO gate_proj anywhere: FusedMoE is built with
// ckpt_names=("up_proj","down_proj","") (:220).
void ClaimMoe(std::vector<NemotronHTensor>& out, const NemotronHParams& p,
              const std::string& mixer, bool quantized) {
  Claim(out, mixer + ".gate.weight", "moe.router");
  // The noaux_tc score-correction bias registered on the gate (:158).
  Claim(out, mixer + ".gate.e_score_correction_bias", "moe.router.bias");
  for (int64_t e = 0; e < p.n_routed_experts; ++e) {
    const std::string expert = mixer + ".experts." + std::to_string(e);
    ClaimNvfp4(out, expert + ".up_proj", "moe.expert.up_proj", quantized);
    ClaimNvfp4(out, expert + ".down_proj", "moe.expert.down_proj", quantized);
  }
  if (p.n_shared_experts > 0) {
    const std::string shared = mixer + ".shared_experts";
    ClaimNvfp4(out, shared + ".up_proj", "moe.shared.up_proj", quantized);
    ClaimNvfp4(out, shared + ".down_proj", "moe.shared.down_proj", quantized);
  }
}

// One dense relu² MLP block (NemotronHMLP, nemotron_h.py:86-123). Absent from
// the driver checkpoint (no `-` in its schedule) but reachable through the
// class default schedule, so it is enumerated rather than left to be discovered.
//
// HONEST DEBT: the quantized companion layout here is DERIVED, not verified.
// `up_proj`/`down_proj` are the same ColumnParallelLinear/RowParallelLinear
// pair the experts use under the same ModelOpt config, so the NVFP4 triple is
// the only consistent reading — but no in-scope released NemotronH checkpoint
// ships an `mlp` block, so the enumeration gate cannot confirm it. It is
// recorded here rather than discovered later.
void ClaimMlp(std::vector<NemotronHTensor>& out, const NemotronHParams& p,
              const std::string& mixer, bool quantized) {
  ClaimNvfp4(out, mixer + ".up_proj", "mlp.up_proj", quantized);
  ClaimNvfp4(out, mixer + ".down_proj", "mlp.down_proj", quantized);
  if (p.mlp_bias) {
    Claim(out, mixer + ".up_proj.bias", "mlp.up_proj.bias");
    Claim(out, mixer + ".down_proj.bias", "mlp.down_proj.bias");
  }
}

// ─── the WEIGHT LOADER (see nemotron_h_loader.h) ─────────────────────────────


// The loader's own refusal. Distinct from the config descent's `Refuse`
// above, and deliberately so: the two name different phases, and a message
// reading "NemotronHForCausalLM:" for a TENSOR problem would send a reader to
// the config.
[[noreturn]] void RefuseLoad(const std::string& detail) {
  throw std::runtime_error("NemotronHForCausalLM weight load: " + detail);
}

// One tensor's on-disk view, keyed by the name the checkpoint ships.
using TensorIndex = std::map<std::string, const StTensor*>;

int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

std::string ShapeStr(const std::vector<int64_t>& shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) s += ", ";
    s += std::to_string(shape[i]);
  }
  return s + "]";
}

// The loader's accumulating state: the index it reads from, the names it has
// consumed (so the accounting is BOTH ways), and the report it fills.
struct Loader {
  const TensorIndex& index;
  NemotronHLoadReport& report;
  std::set<std::string> consumed;

  const StTensor& Need(const std::string& name) {
    const auto it = index.find(name);
    if (it == index.end()) {
      RefuseLoad("the checkpoint does not ship '" + name +
             "', which this architecture's enumeration names");
    }
    if (!consumed.insert(name).second) {
      RefuseLoad("'" + name + "' was claimed twice");
    }
    report.source_bytes += static_cast<int64_t>(it->second->nbytes);
    return *it->second;
  }

  void Expect(const StTensor& t, const std::string& name, const char* dtype,
              const std::vector<int64_t>& shape) {
    if (t.dtype != dtype) {
      RefuseLoad("'" + name + "' ships dtype " + t.dtype + ", not the " + dtype +
             " its scheme declares");
    }
    if (t.shape != shape) {
      RefuseLoad("'" + name + "' ships shape " + ShapeStr(t.shape) + ", not " +
             ShapeStr(shape));
    }
  }
};

// Release the source pages of a range the loader has finished with. The
// mappings stay open (the caller owns them), so without this the whole 20.1 GiB
// checkpoint stays resident alongside the owned mirror.
void Consumed(const StTensor& t) { MaybeReleaseSourcePages(t.data, t.nbytes); }

// ─── dense copies ───────────────────────────────────────────────────────────

// Copy a plain (unquantized) tensor into `want`. `disk_shape` is what the
// checkpoint ships; `logical_shape` is what the forward indexes it as — they
// differ only for `conv1d.weight`, which ships [Cd, 1, K] and is consumed as the
// squeezed [Cd, K] (mamba_mixer2.py's `self.conv1d.weight.view(conv_dim, K)`).
NemotronHOwned CopyDense(Loader& ld, const std::string& name, vt::DType want,
                         const std::vector<int64_t>& disk_shape,
                         std::vector<int64_t> logical_shape) {
  const StTensor& t = ld.Need(name);
  if (t.shape != disk_shape) {
    RefuseLoad("'" + name + "' ships shape " + ShapeStr(t.shape) + ", not " +
           ShapeStr(disk_shape));
  }
  const int64_t n = Numel(disk_shape);
  NemotronHOwned w;
  w.dtype = want;
  w.shape = std::move(logical_shape);
  w.bytes.assign(static_cast<size_t>(n) * vt::SizeOf(want), 0);
  if (t.dtype == "BF16") {
    ld.report.bf16_tensors += 1;
    if (t.nbytes != static_cast<size_t>(n) * 2) {
      RefuseLoad("'" + name + "' is BF16 but its byte count does not match its shape");
    }
    // NOT `reinterpret_cast<const uint16_t*>(t.data)`. `t.data` addresses the
    // safetensors mmap at `8 + <JSON header length> + <sum of the preceding
    // tensors' sizes>` (safetensors_reader.h) and NONE of those three terms is
    // required to be even, so that pointer is one this loader may not form —
    // undefined regardless of whether the access happens to fault, and a real
    // fault on the strict-alignment targets this builds for
    // (build-test-cpu-arm64, Jetson/Orin). Reads go through the shared
    // `vt::LoadUnaligned` seam (a plain memcpy; #301 left it behind, #627 tracks
    // the class, and fc903b8dd/#674 took the same repair in the LTX-2.5 VAE
    // loader after `main` sat RED on `sanitize-cpu (address,undefined)` for it).
    // The bulk `memcpy` below needs no typed pointer at all.
    if (want == vt::DType::kBF16) {
      std::memcpy(w.bytes.data(), t.data, static_cast<size_t>(n) * 2);
    } else if (want == vt::DType::kF32) {
      // A WIDENING. Counted, because the only ones the released checkpoint asks
      // for are the three f32-by-contract SSM scalars; a widening anywhere else
      // is a defect that a token gate would absorb.
      ld.report.widened_tensors += 1;
      auto* dst = reinterpret_cast<float*>(w.bytes.data());
      for (int64_t i = 0; i < n; ++i) {
        dst[i] = vt::BF16ToF32(
            vt::LoadUnaligned<uint16_t>(t.data + static_cast<size_t>(i) * 2));
      }
    } else {
      RefuseLoad("'" + name + "' cannot be materialized at the requested dtype");
    }
  } else if (t.dtype == "F32") {
    ld.report.f32_tensors += 1;
    if (t.nbytes != static_cast<size_t>(n) * 4) {
      RefuseLoad("'" + name + "' is F32 but its byte count does not match its shape");
    }
    // Same seam, same reason as the BF16 arm above: a 4-byte-aligned
    // `const float*` into the mapping is not a pointer this loader may form.
    if (want == vt::DType::kF32) {
      std::memcpy(w.bytes.data(), t.data, static_cast<size_t>(n) * 4);
    } else if (want == vt::DType::kBF16) {
      // A NARROWING, which is what a bf16 model dtype asks for on a tensor the
      // producer happened to store wide. Never silent: it is the model dtype
      // every other layer inherits, and the router (the one f32 consumer) asks
      // for f32 explicitly.
      auto* dst = reinterpret_cast<uint16_t*>(w.bytes.data());
      for (int64_t i = 0; i < n; ++i) {
        dst[i] = vt::F32ToBF16(
            vt::LoadUnaligned<float>(t.data + static_cast<size_t>(i) * 4));
      }
    } else {
      RefuseLoad("'" + name + "' cannot be materialized at the requested dtype");
    }
  } else {
    RefuseLoad("'" + name + "' ships dtype " + t.dtype +
           ", which is not an unquantized dtype this loader reads");
  }
  Consumed(t);
  return w;
}

NemotronHOwned CopyDense(Loader& ld, const std::string& name, vt::DType want,
                         const std::vector<int64_t>& shape) {
  return CopyDense(ld, name, want, shape, shape);
}

// A2-R (#810): the same dense copy, delivered in the tree's SHARED residency
// type so the weight can be uploaded once by `dense_attn::ResidentWeight`.
//
// It DELEGATES to CopyDense rather than duplicating it. That is deliberate:
// CopyDense owns the unaligned-read seam, the dtype widening/narrowing arms and
// every `report` counter, and a second copy of that logic would be a second
// place for the load composition to drift from what the gate asserts. The bytes
// are MOVED into the OwnedBytes, so this costs one vector move and the reported
// `host_bytes` total is byte-identical to before the conversion.
//
// `nk` records the matmul orientation for the four attention projections: the
// checkpoint ships raw torch Linear [out, in] and BOTH arms consume them via
// vt::MatmulBT, which is what `nk = true` means (qwen3_5_weights.h:60-63). It
// is left false for the embedding table and the norms, which are not matmul
// weights and for which the field has no meaning.
OwnedTensor CopyDenseOwned(Loader& ld, const std::string& name, vt::DType want,
                           const std::vector<int64_t>& shape, bool nk) {
  NemotronHOwned w = CopyDense(ld, name, want, shape, shape);
  OwnedTensor t;
  t.dtype = w.dtype;
  t.nk = nk;
  VT_CHECK(!w.shape.empty() && w.shape.size() <= static_cast<size_t>(vt::kMaxRank),
           "NemotronH loader: '" + name + "' rank is out of range for OwnedTensor");
  t.rank = static_cast<int>(w.shape.size());
  for (int i = 0; i < t.rank; ++i) t.shape[i] = w.shape[static_cast<size_t>(i)];
  t.bytes = OwnedBytes(std::move(w.bytes));
  return t;
}

// Bytes resident for an OwnedTensor-held weight. The NemotronHOwned twin is
// `HostBytes()` (payload + scales); a dense OwnedTensor has no scales, so this
// is just the payload — exactly what HostBytes() returned for the same weight
// before A2-R converted it, which is why the load report's total is unchanged.
int64_t OwnedHostBytes(const OwnedTensor& t) {
  return static_cast<int64_t>(t.bytes.size());
}

// A per-tensor f32 scalar companion. ModelOpt writes `weight_scale_2` and
// `weight_scale` as rank-0 and `input_scale` / `k_scale` / `v_scale` as [1];
// both spellings are one number and both are accepted, nothing else is.
float ReadF32Scalar(Loader& ld, const std::string& name) {
  const StTensor& t = ld.Need(name);
  if (t.dtype != "F32") {
    RefuseLoad("'" + name + "' ships dtype " + t.dtype + ", not the F32 a scale is");
  }
  if (!(t.shape.empty() || t.shape == std::vector<int64_t>{1})) {
    RefuseLoad("'" + name + "' ships shape " + ShapeStr(t.shape) +
           ", not the scalar a per-tensor scale is");
  }
  if (t.nbytes != sizeof(float)) {
    RefuseLoad("'" + name + "' is a scalar F32 but does not carry 4 bytes");
  }
  float v = 0.0F;
  std::memcpy(&v, t.data, sizeof(float));
  Consumed(t);
  return v;
}

// ─── the two quantized schemes ──────────────────────────────────────────────

// `W4A16_NVFP4`, `group_size=16`. `rows`/`cols` are the LOGICAL [out, in].
NemotronHOwned LoadNvfp4(Loader& ld, const std::string& prefix, vt::DType logical,
                         int64_t rows, int64_t cols) {
  if (cols % kNvfp4GroupSize != 0) {
    RefuseLoad("'" + prefix + "' has in_features " + std::to_string(cols) +
           ", which is not a multiple of the NVFP4 group size 16");
  }
  NemotronHOwned w;
  w.form = NemotronHWeightForm::kNvfp4W4A16G16;
  w.dtype = logical;
  w.shape = {rows, cols};

  const std::string wname = prefix + ".weight";
  const StTensor& packed = ld.Need(wname);
  ld.Expect(packed, wname, "U8", {rows, cols / 2});
  w.bytes.assign(packed.data, packed.data + packed.nbytes);
  Consumed(packed);

  const std::string sname = prefix + ".weight_scale";
  const StTensor& gs = ld.Need(sname);
  // The GROUP scale, not a per-tensor one. Binding a per-tensor scale here (or
  // this one to the wrong projection) produces a finite, correctly-shaped,
  // wrongly-scaled matrix — the x1.10-class error a token gate absorbs.
  ld.Expect(gs, sname, "F8_E4M3", {rows, cols / kNvfp4GroupSize});
  w.scale.assign(gs.data, gs.data + gs.nbytes);
  Consumed(gs);

  w.global_scale = ReadF32Scalar(ld, prefix + ".weight_scale_2");

  ld.report.nvfp4_weights += 1;
  ld.report.nvfp4_tensors += 3;
  return w;
}

// FP8 W8A8 static. Weight-only on this path: `input_scale` is carried, not
// applied (nemotron_h_forward.h records why).
NemotronHOwned LoadFp8(Loader& ld, const std::string& prefix, vt::DType logical,
                       int64_t rows, int64_t cols) {
  NemotronHOwned w;
  w.form = NemotronHWeightForm::kFp8W8A8Static;
  w.dtype = logical;
  w.shape = {rows, cols};

  const std::string wname = prefix + ".weight";
  const StTensor& q = ld.Need(wname);
  ld.Expect(q, wname, "F8_E4M3", {rows, cols});
  w.bytes.assign(q.data, q.data + q.nbytes);
  Consumed(q);

  w.global_scale = ReadF32Scalar(ld, prefix + ".weight_scale");
  w.input_scale = ReadF32Scalar(ld, prefix + ".input_scale");
  w.has_input_scale = true;

  ld.report.fp8_weights += 1;
  ld.report.fp8_tensors += 3;
  return w;
}

// ─── the blocks ─────────────────────────────────────────────────────────────

void LoadMamba(Loader& ld, const NemotronHParams& p, const std::string& mixer,
               vt::DType adt, bool quantized, NemotronHMambaWeights& out) {
  const int64_t H = p.hidden_size;
  const int64_t I = p.mamba_intermediate_size();
  const int64_t Cd = p.conv_dim();
  const int64_t K = p.conv_kernel;
  const int64_t Hh = p.mamba_num_heads;

  // BRANCH ON `quantized`, exactly as `LoadExpert`/`LoadMlp` and the
  // `ClaimFp8`/`ClaimNvfp4` enumeration do. `ClaimMamba` deliberately supports
  // the unquantized case — hard-coding the FP8 companions there enumerated 92
  // tensors a released bf16 checkpoint does not ship — so loading these two
  // projections as FP8 unconditionally made the loader disagree with its own
  // enumeration: a bf16 NemotronH refused with `'…in_proj.weight' ships dtype
  // BF16, not the F8_E4M3 its scheme declares`, a DTYPE message for what is
  // really the scheme the config declared.
  if (quantized) {
    out.in_proj = LoadFp8(ld, mixer + ".in_proj", adt, p.in_proj_out_features(), H);
    out.out_proj = LoadFp8(ld, mixer + ".out_proj", adt, H, I);
  } else {
    out.in_proj =
        CopyDense(ld, mixer + ".in_proj.weight", adt, {p.in_proj_out_features(), H});
    out.out_proj = CopyDense(ld, mixer + ".out_proj.weight", adt, {H, I});
  }
  // The conv weight ships [Cd, 1, K] and is consumed squeezed.
  out.conv1d_weight =
      CopyDense(ld, mixer + ".conv1d.weight", adt, {Cd, 1, K}, {Cd, K});
  if (p.use_conv_bias) {
    out.conv1d_bias = CopyDense(ld, mixer + ".conv1d.bias", adt, {Cd});
  }
  // f32 BY CONTRACT, and bf16 on disk. Upstream keeps these f32 whatever the
  // model dtype (`self.A = -torch.exp(self.A_log.float())`, and D/dt_bias feed
  // the same f32 scan); `vt::Mamba2ChunkScan` validates all three as f32. This
  // is the annotated f32 escape AGENTS.md allows, it is upstream's own polarity,
  // and `report.widened_tensors` counts it so it stays exactly these three.
  out.A_log = CopyDense(ld, mixer + ".A_log", vt::DType::kF32, {Hh});
  out.D = CopyDense(ld, mixer + ".D", vt::DType::kF32, {Hh});
  out.dt_bias = CopyDense(ld, mixer + ".dt_bias", vt::DType::kF32, {Hh});
  // Mixer2RMSNormGated over the SSM intermediate width, NOT hidden_size.
  out.norm_weight = CopyDense(ld, mixer + ".norm.weight", adt, {I});
}

void LoadAttention(Loader& ld, const NemotronHParams& p, const std::string& mixer,
                   vt::DType adt, bool fp8_kv, NemotronHAttentionWeights& out) {
  const int64_t H = p.hidden_size;
  const int64_t qd = p.q_proj_out_features();
  const int64_t kvd = p.kv_proj_out_features();
  // A2-R: OwnedTensor, nk=true — raw torch Linear [out, in], consumed via
  // vt::MatmulBT by both the host mixer and the device NemotronHAttnBlock.
  out.q_proj = CopyDenseOwned(ld, mixer + ".q_proj.weight", adt, {qd, H}, true);
  out.k_proj = CopyDenseOwned(ld, mixer + ".k_proj.weight", adt, {kvd, H}, true);
  out.v_proj = CopyDenseOwned(ld, mixer + ".v_proj.weight", adt, {kvd, H}, true);
  out.o_proj = CopyDenseOwned(ld, mixer + ".o_proj.weight", adt, {H, qd}, true);
  if (fp8_kv) {
    out.k_scale = ReadF32Scalar(ld, mixer + ".k_proj.k_scale");
    out.v_scale = ReadF32Scalar(ld, mixer + ".v_proj.v_scale");
    out.has_kv_scales = true;
    ld.report.fp8_kv_scale_tensors += 2;
  }
}

void LoadExpert(Loader& ld, const std::string& prefix, vt::DType adt, int64_t H,
                int64_t I, bool quantized, NemotronHExpertWeights& out) {
  // `ckpt_names=("up_proj","down_proj","")` (nemotron_h.py:220): there is no
  // gate_proj anywhere in this checkpoint.
  if (quantized) {
    out.up_proj = LoadNvfp4(ld, prefix + ".up_proj", adt, I, H);
    out.down_proj = LoadNvfp4(ld, prefix + ".down_proj", adt, H, I);
  } else {
    out.up_proj = CopyDense(ld, prefix + ".up_proj.weight", adt, {I, H});
    out.down_proj = CopyDense(ld, prefix + ".down_proj.weight", adt, {H, I});
  }
}

void LoadMoe(Loader& ld, const NemotronHParams& p, const std::string& mixer,
             vt::DType adt, bool quantized, NemotronHMoeWeights& out) {
  const int64_t H = p.hidden_size;
  const int64_t E = p.n_routed_experts;
  // The ROUTER IS F32 AND THAT IS MIRRORED, NOT INHERITED:
  // `GateLinear(..., out_dtype=torch.float32, force_fp32_compute=True)`
  // (nemotron_h.py:150-156). The released backbone even ships it F32 on disk;
  // the MTP tower's twin ships BF16, which is exactly why the dtype is REQUESTED
  // here rather than taken from whatever the producer wrote.
  out.gate = CopyDense(ld, mixer + ".gate.weight", vt::DType::kF32, {E, H});
  out.e_score_correction_bias = CopyDense(
      ld, mixer + ".gate.e_score_correction_bias", vt::DType::kF32, {E});
  out.experts.resize(static_cast<size_t>(E));
  for (int64_t e = 0; e < E; ++e) {
    LoadExpert(ld, mixer + ".experts." + std::to_string(e), adt, H,
               p.moe_intermediate_size, quantized,
               out.experts[static_cast<size_t>(e)]);
  }
  if (p.n_shared_experts > 0) {
    LoadExpert(ld, mixer + ".shared_experts", adt, H,
               p.moe_shared_expert_intermediate_size * p.n_shared_experts,
               quantized, out.shared);
    out.has_shared = true;
  }
}

void LoadMlp(Loader& ld, const NemotronHParams& p, const std::string& mixer,
             vt::DType adt, bool quantized, NemotronHMlpWeights& out) {
  NemotronHExpertWeights e;
  LoadExpert(ld, mixer, adt, p.hidden_size, p.intermediate_size, quantized, e);
  out.up_proj = std::move(e.up_proj);
  out.down_proj = std::move(e.down_proj);
}

int64_t HostBytesOf(const NemotronHHostWeights& h) {
  // A2-R: embeddings / norm_f / per-layer norm / attention q,k,v,o are now
  // OwnedTensor, so they are counted through OwnedHostBytes. Same bytes, same
  // total — the conversion changed the holder, not the payload.
  int64_t n = OwnedHostBytes(h.embeddings) + OwnedHostBytes(h.norm_f) +
              h.lm_head.HostBytes();
  for (const NemotronHLayerWeights& l : h.layers) {
    n += OwnedHostBytes(l.norm);
    n += l.mamba.in_proj.HostBytes() + l.mamba.out_proj.HostBytes() +
         l.mamba.conv1d_weight.HostBytes() + l.mamba.conv1d_bias.HostBytes() +
         l.mamba.A_log.HostBytes() + l.mamba.D.HostBytes() +
         l.mamba.dt_bias.HostBytes() + l.mamba.norm_weight.HostBytes();
    n += OwnedHostBytes(l.attn.q_proj) + OwnedHostBytes(l.attn.k_proj) +
         OwnedHostBytes(l.attn.v_proj) + OwnedHostBytes(l.attn.o_proj);
    n += l.moe.gate.HostBytes() + l.moe.e_score_correction_bias.HostBytes();
    for (const NemotronHExpertWeights& e : l.moe.experts) {
      n += e.up_proj.HostBytes() + e.down_proj.HostBytes();
    }
    n += l.moe.shared.up_proj.HostBytes() + l.moe.shared.down_proj.HostBytes();
    n += l.mlp.up_proj.HostBytes() + l.mlp.down_proj.HostBytes();
  }
  return n;
}

}  // namespace

std::string_view NemotronHBlockName(NemotronHBlock block) {
  switch (block) {
    case NemotronHBlock::kMamba:
      return "mamba";
    case NemotronHBlock::kAttention:
      return "attention";
    case NemotronHBlock::kMoe:
      return "moe";
    case NemotronHBlock::kMlp:
      return "mlp";
  }
  return "unknown";
}

std::vector<int64_t> NemotronHParams::LayerIndices(NemotronHBlock block) const {
  std::vector<int64_t> out;
  for (size_t i = 0; i < layers_block_type.size(); ++i) {
    if (layers_block_type[i] == block) out.push_back(static_cast<int64_t>(i));
  }
  return out;
}

NemotronHParams ParseNemotronHParams(const HfConfig& config) {
  const json& doc = Raw(config);
  NemotronHParams p;

  // --- the schedule, which IS the depth ---
  p.layers_block_type = ResolveSchedule(
      doc, "layers_block_type", "hybrid_override_pattern",
      // configuration_nemotron_h.py:165.
      {NemotronHBlock::kMamba, NemotronHBlock::kMoe, NemotronHBlock::kAttention,
       NemotronHBlock::kMlp});
  if (p.layers_block_type.empty()) {
    RefuseLoad("layers_block_type resolved to an empty schedule");
  }
  p.num_nextn_predict_layers = GetInt(doc, "num_nextn_predict_layers", 0);
  p.mtp_layers_block_type =
      ResolveSchedule(doc, "mtp_layers_block_type",
                      "mtp_hybrid_override_pattern",
                      // configuration_nemotron_h.py:180.
                      {NemotronHBlock::kAttention, NemotronHBlock::kMoe});
  if (p.num_nextn_predict_layers > 0 && p.mtp_layers_block_type.empty()) {
    // Mirror of validate_layer_type (configuration_nemotron_h.py:206-212).
    RefuseLoad(
        "mtp_layers_block_type is required when num_nextn_predict_layers > 0");
  }

  // --- shared geometry ---
  p.hidden_size = GetInt(doc, "hidden_size", 4096);
  p.vocab_size = GetInt(doc, "vocab_size", 131072);
  p.max_position_embeddings = GetInt(doc, "max_position_embeddings", 4096);
  p.layer_norm_epsilon = GetDouble(doc, "layer_norm_epsilon", 1e-5);
  p.tie_word_embeddings = GetBool(doc, "tie_word_embeddings", false);

  // --- attention ---
  p.num_attention_heads = GetInt(doc, "num_attention_heads", 32);
  // Three states, not two (configuration_nemotron_h.py:97 default 8, :188-189
  // `if self.num_key_value_heads is None: = num_attention_heads`): ABSENT takes
  // the class default 8, an explicit `null` takes the query-head count, a value
  // is taken as-is.
  if (Has(doc, "num_key_value_heads")) {
    p.num_key_value_heads = GetInt(doc, "num_key_value_heads", 8);
  } else if (doc.contains("num_key_value_heads")) {
    p.num_key_value_heads = p.num_attention_heads;
  } else {
    p.num_key_value_heads = 8;
  }
  p.head_dim = GetInt(doc, "head_dim", 128);
  p.rope_theta = GetDouble(doc, "rope_theta", 10000.0);
  p.partial_rotary_factor = GetDouble(doc, "partial_rotary_factor", 1.0);
  p.attention_bias = GetBool(doc, "attention_bias", false);
  if (Has(doc, "sliding_window")) {
    p.sliding_window = GetInt(doc, "sliding_window", 0);
  }

  // --- Mamba2 (legacy aliases WIN, configuration_nemotron_h.py:145-155) ---
  p.mamba_num_heads = GetInt(doc, "mamba_num_heads", 128);
  p.mamba_head_dim = GetInt(doc, "mamba_head_dim", 64);
  p.n_groups = GetIntAliased(doc, "n_groups", "mamba_n_groups", 8);
  p.ssm_state_size = GetInt(doc, "ssm_state_size", 128);
  p.conv_kernel = GetIntAliased(doc, "conv_kernel", "mamba_d_conv", 4);
  p.chunk_size = GetIntAliased(doc, "chunk_size", "mamba_chunk_size", 128);
  p.expand = GetIntAliased(doc, "expand", "mamba_expand", 2);
  p.mamba_hidden_act = GetString(doc, "mamba_hidden_act", "silu");
  p.mamba_ssm_cache_dtype = GetString(doc, "mamba_ssm_cache_dtype", "float32");
  p.use_conv_bias = GetBoolAliased(doc, "use_conv_bias", "mamba_conv_bias", true);
  p.use_bias = GetBool(doc, "use_bias", false);
  p.mamba_proj_bias = GetBool(doc, "mamba_proj_bias", false);
  p.time_step_min = GetDoubleAliased(doc, "time_step_min", "mamba_dt_min", 1e-3);
  p.time_step_max = GetDoubleAliased(doc, "time_step_max", "mamba_dt_max", 1e-1);
  p.time_step_floor =
      GetDoubleAliased(doc, "time_step_floor", "mamba_dt_init_floor", 1e-4);
  if (p.mamba_num_heads <= 0 || p.mamba_head_dim <= 0 || p.n_groups <= 0 ||
      p.ssm_state_size <= 0 || p.conv_kernel <= 1) {
    RefuseLoad("the Mamba2 geometry is degenerate (mamba_num_heads, mamba_head_dim, "
           "n_groups, ssm_state_size must be positive and conv_kernel > 1)");
  }

  // --- MoE ---
  p.n_routed_experts = GetInt(doc, "n_routed_experts", 8);
  p.num_experts_per_tok = GetInt(doc, "num_experts_per_tok", 2);
  p.moe_intermediate_size = GetInt(doc, "moe_intermediate_size", 7688);
  p.n_shared_experts = GetInt(doc, "n_shared_experts", 1);
  p.moe_shared_expert_intermediate_size =
      GetInt(doc, "moe_shared_expert_intermediate_size", 7688);
  p.n_group = GetInt(doc, "n_group", 1);
  p.topk_group = GetInt(doc, "topk_group", 1);
  p.routed_scaling_factor = GetDouble(doc, "routed_scaling_factor", 1.0);
  p.norm_topk_prob = GetBool(doc, "norm_topk_prob", true);
  p.moe_shared_expert_overlap = GetBool(doc, "moe_shared_expert_overlap", true);
  p.mlp_hidden_act = GetString(doc, "mlp_hidden_act", "relu2");
  if (Has(doc, "moe_latent_size")) {
    p.moe_latent_size = GetInt(doc, "moe_latent_size", 0);
    // The `fc1_latent_proj`/`fc2_latent_proj` pair (nemotron_h.py:191-207) is
    // out of scope for this row (spec §0), and silently ignoring the key would
    // build a differently-shaped MoE with no error.
    RefuseLoad("moe_latent_size is set, but the latent MoE "
           "(fc1_latent_proj/fc2_latent_proj) is out of scope for this row "
           "(see .agents/specs/nemotron-h-model.md §0)");
  }
  if (p.mlp_hidden_act != "relu2") {
    RefuseLoad("mlp_hidden_act '" + p.mlp_hidden_act +
           "' is not implemented (this architecture is the non-gated relu2 "
           "expert; see .agents/specs/nemotron-h-model.md W2)");
  }

  // --- dense MLP block ---
  if (doc.contains("intermediate_size") && doc.at("intermediate_size").is_array()) {
    // `get_nemotron_h_config_for_layer` / NemotronHPuzzleForCausalLM
    // (nemotron_h.py:283-288) is explicitly out of scope (spec §0).
    RefuseLoad("a per-layer intermediate_size list (NemotronHPuzzleForCausalLM) is "
           "out of scope for this row (see .agents/specs/nemotron-h-model.md §0)");
  }
  p.intermediate_size = GetInt(doc, "intermediate_size", 21504);
  p.mlp_bias = GetBool(doc, "mlp_bias", false);

  // The recurrent-cache dtype is resolved independently of the conv dtype
  // (mamba_utils.py:99-104); refuse an alias we cannot represent HERE rather
  // than at cache-allocation time.
  const std::string& ssm = p.mamba_ssm_cache_dtype;
  if (!(ssm.empty() || ssm == "auto" || ssm == "float32" || ssm == "float" ||
        ssm == "float16" || ssm == "half" || ssm == "bfloat16")) {
    RefuseLoad("mamba_ssm_cache_dtype '" + ssm +
           "' is not supported (expected auto, float32/float, float16/half or "
           "bfloat16)");
  }

  p.quant = ResolveQuantSurface(doc);

  // `num_hidden_layers` is DEPRECATED, and its setter ignores whatever the
  // checkpoint says (configuration_nemotron_h.py:233-238) — depth is the
  // schedule's LENGTH. Upstream only WARNS on a conflicting scalar
  // (:167-175), so this mirrors that rather than refusing: the released
  // config.json ships `num_hidden_layers: 52` alongside a 52-entry
  // `layers_block_type`, and a checkpoint that ships only the scalar becomes a
  // 4-block model upstream too. The behavior is pinned by a test so it stays a
  // deliberate mirror rather than an accident.
  (void)config.num_hidden_layers;
  return p;
}

void ParseNemotronHConfig(const HfConfig& config) {
  (void)ParseNemotronHParams(config);
}

vt::DType NemotronHSsmCacheDType(const NemotronHParams& params,
                                 vt::DType conv_dtype) {
  const std::string& dtype = params.mamba_ssm_cache_dtype;
  if (dtype.empty() || dtype == "auto") return conv_dtype;
  if (dtype == "float32" || dtype == "float") return vt::DType::kF32;
  if (dtype == "float16" || dtype == "half") return vt::DType::kF16;
  if (dtype == "bfloat16") return vt::DType::kBF16;
  // ParseNemotronHParams already refused anything else by name; this is the
  // unreachable arm kept so the mapping cannot silently widen.
  RefuseLoad("mamba_ssm_cache_dtype '" + dtype + "' is not supported");
}

std::vector<NemotronHTensor> EnumerateNemotronHTensors(
    const NemotronHParams& params) {
  const NemotronHParams& p = params;
  // The backbone is quantized where the config groups say so; the MTP tower is
  // covered by the `mtp*` entry in `ignore` and ships bf16 with no companions.
  const bool quantized = p.quant.present;
  const bool mtp_quantized = quantized && !p.quant.mtp_ignored;

  std::vector<NemotronHTensor> out;
  out.reserve(4096);

  // --- root ---
  Claim(out, "backbone.embeddings.weight", "embed_tokens");
  Claim(out, "backbone.norm_f.weight", "final_norm");
  if (!p.tie_word_embeddings) {
    ClaimNvfp4(out, "lm_head", "lm_head", quantized);
  }

  // --- the 52 backbone layers ---
  for (size_t i = 0; i < p.layers_block_type.size(); ++i) {
    const std::string layer = "backbone.layers." + std::to_string(i);
    const std::string mixer = layer + ".mixer";
    // Every block kind carries the pre-mixer RMSNorm (nemotron_h.py:299, :339,
    // :391, :521).
    Claim(out, layer + ".norm.weight", "layer_norm");
    switch (p.layers_block_type[i]) {
      case NemotronHBlock::kMamba:
        ClaimMamba(out, p, mixer, quantized);
        break;
      case NemotronHBlock::kAttention:
        ClaimAttention(out, p, mixer, quantized && p.quant.fp8_kv_cache);
        break;
      case NemotronHBlock::kMoe:
        ClaimMoe(out, p, mixer, quantized);
        break;
      case NemotronHBlock::kMlp:
        ClaimMlp(out, p, mixer, quantized);
        break;
    }
  }

  // --- the MTP tower (nemotron_h_mtp.py:241-275) ---
  // total_layers = num_nextn_predict_layers * pattern_len; the FIRST block of
  // every step carries enorm/hnorm/eh_proj and the LAST carries
  // final_layernorm (:65-86, :152-173).
  const int64_t pattern_len =
      static_cast<int64_t>(p.mtp_layers_block_type.size());
  const int64_t mtp_layers = p.num_nextn_predict_layers * pattern_len;
  for (int64_t i = 0; i < mtp_layers; ++i) {
    const int64_t rel = i % pattern_len;
    const std::string layer = "mtp.layers." + std::to_string(i);
    const std::string mixer = layer + ".mixer";
    Claim(out, layer + ".norm.weight", "mtp.layer_norm");
    if (rel == 0) {
      Claim(out, layer + ".enorm.weight", "mtp.enorm");
      Claim(out, layer + ".hnorm.weight", "mtp.hnorm");
      Claim(out, layer + ".eh_proj.weight", "mtp.eh_proj");
    }
    if (rel == pattern_len - 1) {
      Claim(out, layer + ".final_layernorm.weight", "mtp.final_layernorm");
    }
    switch (p.mtp_layers_block_type[static_cast<size_t>(rel)]) {
      case NemotronHBlock::kAttention:
        // The MTP attention block is unquantized: no k_scale/v_scale ship for
        // it, which is what `ignore: [... "mtp*"]` buys.
        ClaimAttention(out, p, mixer, mtp_quantized && p.quant.fp8_kv_cache);
        break;
      case NemotronHBlock::kMoe:
        ClaimMoe(out, p, mixer, mtp_quantized);
        break;
      case NemotronHBlock::kMamba:
        ClaimMamba(out, p, mixer, mtp_quantized);
        break;
      case NemotronHBlock::kMlp:
        ClaimMlp(out, p, mixer, mtp_quantized);
        break;
    }
  }
  return out;
}


vt::DType ResolveNemotronHModelDType(const HfConfig& config) {
  // transformers serializes the model dtype as `dtype` (the released
  // NemotronH config.json ships `"dtype": "bfloat16"`); `torch_dtype` is the
  // legacy spelling and is the fallback, not the other way round. HfConfig
  // parses only the legacy key, so the modern one is read from the raw
  // document here.
  std::string name;
  if (config.raw.contains("dtype") && config.raw.at("dtype").is_string()) {
    name = config.raw.at("dtype").get<std::string>();
  }
  if (name.empty()) name = config.torch_dtype;
  if (name.empty() || name == "auto") name = "bfloat16";
  if (name == "bfloat16") return vt::DType::kBF16;
  if (name == "float32" || name == "float") return vt::DType::kF32;
  // f16 is deliberately refused rather than substituted: `vt::MoeRelu2` has no
  // f16 output arm (spec §6a), so a silent widen-to-bf16 would run a different
  // model than the checkpoint declares.
  throw std::runtime_error(
      "NemotronHForCausalLM weight load: model dtype '" + name +
      "' is not supported (this forward composes ops whose output dtypes are "
      "bf16 — the released checkpoint's — and f32)");
}

NemotronHHostWeights LoadNemotronHHostWeights(
    const std::vector<SafetensorsFile>& shards, const NemotronHParams& params,
    vt::DType act_dtype, NemotronHLoadReport* report,
    const std::string& name_prefix) {
  NemotronHLoadReport local;
  NemotronHLoadReport& rep = report != nullptr ? *report : local;
  rep = NemotronHLoadReport{};

  TensorIndex index;
  for (const SafetensorsFile& shard : shards) {
    for (const std::string& name : shard.Names()) {
      if (name.compare(0, name_prefix.size(), name_prefix) != 0) continue;
      if (!index.emplace(name.substr(name_prefix.size()), &shard.Get(name)).second) {
        RefuseLoad("'" + name + "' appears in more than one shard");
      }
    }
  }
  rep.in_index = static_cast<int64_t>(index.size());

  const std::vector<NemotronHTensor> enumerated =
      EnumerateNemotronHTensors(params);
  rep.enumerated = static_cast<int64_t>(enumerated.size());

  NemotronHHostWeights host;
  host.act_dtype = act_dtype;
  const NemotronHParams& p = params;
  const bool quantized = p.quant.present;

  Loader ld{index, rep, {}};

  // --- root ---
  host.embeddings = CopyDenseOwned(ld, "backbone.embeddings.weight", act_dtype,
                                   {p.vocab_size, p.hidden_size}, false);
  host.norm_f =
      CopyDenseOwned(ld, "backbone.norm_f.weight", act_dtype, {p.hidden_size}, false);
  if (!p.tie_word_embeddings) {
    host.lm_head = quantized
                       ? LoadNvfp4(ld, "lm_head", act_dtype, p.vocab_size,
                                   p.hidden_size)
                       : CopyDense(ld, "lm_head.weight", act_dtype,
                                   {p.vocab_size, p.hidden_size});
  }

  // --- the 52 backbone layers ---
  const int64_t L = p.num_hidden_layers();
  host.layers.resize(static_cast<size_t>(L));
  for (int64_t i = 0; i < L; ++i) {
    NemotronHLayerWeights& lw = host.layers[static_cast<size_t>(i)];
    const std::string layer = "backbone.layers." + std::to_string(i);
    const std::string mixer = layer + ".mixer";
    lw.block = p.layers_block_type[static_cast<size_t>(i)];
    lw.norm =
        CopyDenseOwned(ld, layer + ".norm.weight", act_dtype, {p.hidden_size}, false);
    switch (lw.block) {
      case NemotronHBlock::kMamba:
        LoadMamba(ld, p, mixer, act_dtype, quantized, lw.mamba);
        break;
      case NemotronHBlock::kAttention:
        LoadAttention(ld, p, mixer, act_dtype,
                      quantized && p.quant.fp8_kv_cache, lw.attn);
        break;
      case NemotronHBlock::kMoe:
        LoadMoe(ld, p, mixer, act_dtype, quantized, lw.moe);
        break;
      case NemotronHBlock::kMlp:
        LoadMlp(ld, p, mixer, act_dtype, quantized, lw.mlp);
        break;
    }
  }

  rep.materialized = static_cast<int64_t>(ld.consumed.size());

  // --- the MTP tower: DEFERRED BY NAME (W5) ---------------------------------
  //
  // 270 unquantized bf16 tensors (`ignore` carries `mtp*`). W5 owns the head;
  // nothing here can consume it, and materializing it would cost 2.65 GB of
  // host memory nothing reads. It is counted and NAMED rather than skipped —
  // the whole point of the enumeration is that no tensor is in the "nobody
  // thought of it" state.
  std::set<std::string> deferred_tags;
  for (const NemotronHTensor& t : enumerated) {
    if (ld.consumed.count(t.name) != 0) continue;
    if (t.name.rfind("mtp.", 0) != 0) {
      RefuseLoad("'" + t.name + "' (consumer '" + t.consumer +
             "') is enumerated but no host slot claimed it; every enumerated "
             "tensor must be materialized or deferred by name");
    }
    rep.deferred += 1;
    deferred_tags.insert(t.consumer);
  }
  for (const std::string& tag : deferred_tags) {
    rep.deferred_by_name.push_back(
        tag + " (MTP head, W5 of .agents/specs/nemotron-h-model.md)");
  }

  // The other direction: a tensor the checkpoint ships that nothing named.
  if (rep.materialized + rep.deferred != rep.enumerated) {
    RefuseLoad("accounting mismatch: " + std::to_string(rep.materialized) +
           " materialized + " + std::to_string(rep.deferred) + " deferred != " +
           std::to_string(rep.enumerated) + " enumerated");
  }
  if (rep.enumerated != rep.in_index) {
    RefuseLoad("the enumeration names " + std::to_string(rep.enumerated) +
           " tensors but the checkpoint ships " + std::to_string(rep.in_index));
  }

  rep.host_bytes = HostBytesOf(host);
  host.materialized = true;
  return host;
}

}  // namespace vllm
