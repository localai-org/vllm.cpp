// vllm.cpp original. GGUF wire semantics follow the llama.cpp format; pinned
// vLLM e24d1b24 has no GGUF load format.
#include "vllm/model_executor/model_loader/gguf_reader.h"

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <utility>
#include <variant>

namespace vllm {

namespace {

std::filesystem::path Utf8Path(const std::string& path) {
  return std::filesystem::path(std::u8string(
      reinterpret_cast<const char8_t*>(path.data()), path.size()));
}

[[noreturn]] void Fail(const std::string& path, const std::string& what) {
  throw std::runtime_error("gguf: " + what + " in " + path);
}

// Sanity cap on kv and tensor counts (UNTRUSTED header): real checkpoints
// have thousands of entries, so 1e6 rejects hostile counts before any
// allocation is sized from them.
constexpr uint64_t kMaxCount = 1000000;
// Arrays may nest (an array element can itself be an array); bound the
// recursion so a hostile file cannot overflow the stack.
constexpr int kMaxArrayDepth = 16;

// Bounds-checked little-endian cursor over the mmap'd file. Every Need()
// call validates against the real file size BEFORE the bytes are touched.
struct Cursor {
  const uint8_t* base;
  size_t size;
  size_t pos = 0;
  const std::string& path;

  void Need(size_t n, const char* what) {
    // pos <= size always holds, so size - pos cannot underflow.
    if (n > size - pos)
      Fail(path, std::string("truncated file: ") + what + " needs " +
                     std::to_string(n) + " bytes at offset " +
                     std::to_string(pos) + " but only " +
                     std::to_string(size - pos) + " remain");
  }
  uint8_t U8(const char* what) {
    Need(1, what);
    return base[pos++];
  }
  uint16_t U16(const char* what) {
    Need(2, what);
    uint16_t v = 0;
    for (int i = 0; i < 2; ++i)
      v = static_cast<uint16_t>(v | static_cast<uint16_t>(base[pos + i])
                                        << (8 * i));
    pos += 2;
    return v;
  }
  uint32_t U32(const char* what) {
    Need(4, what);
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i)
      v |= static_cast<uint32_t>(base[pos + i]) << (8 * i);
    pos += 4;
    return v;
  }
  uint64_t U64(const char* what) {
    Need(8, what);
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i)
      v |= static_cast<uint64_t>(base[pos + i]) << (8 * i);
    pos += 8;
    return v;
  }
  std::string Str(const char* what) {
    const uint64_t len = U64(what);
    // UNTRUSTED length: bound against the remaining file before use (this
    // also caps it at the file size).
    if (len > size - pos)
      Fail(path, std::string(what) + " string length " + std::to_string(len) +
                     " exceeds remaining file size " +
                     std::to_string(size - pos));
    std::string s(reinterpret_cast<const char*>(base + pos),
                  static_cast<size_t>(len));
    pos += static_cast<size_t>(len);
    return s;
  }
};

// `elems_parsed` is the running RECURSIVE total of array elements parsed so
// far in this file; it is bounded by kMaxCount so nested arrays cannot
// multiply the per-array count checks into an amplified allocation.
GgufValue ReadValue(Cursor& cur, uint32_t type, int depth,
                    uint64_t& elems_parsed) {
  GgufValue out;
  switch (type) {
    case kGgufU8:
      out.v = cur.U8("u8 kv");
      break;
    case kGgufI8:
      out.v = static_cast<int8_t>(cur.U8("i8 kv"));
      break;
    case kGgufU16:
      out.v = cur.U16("u16 kv");
      break;
    case kGgufI16:
      out.v = static_cast<int16_t>(cur.U16("i16 kv"));
      break;
    case kGgufU32:
      out.v = cur.U32("u32 kv");
      break;
    case kGgufI32:
      out.v = static_cast<int32_t>(cur.U32("i32 kv"));
      break;
    case kGgufF32: {
      const uint32_t bits = cur.U32("f32 kv");
      float f;
      std::memcpy(&f, &bits, sizeof(f));
      out.v = f;
      break;
    }
    case kGgufBool:
      out.v = cur.U8("bool kv") != 0;
      break;
    case kGgufString:
      out.v = cur.Str("kv");
      break;
    case kGgufArray: {
      if (depth >= kMaxArrayDepth)
        Fail(cur.path, "kv array nesting exceeds depth limit " +
                           std::to_string(kMaxArrayDepth));
      GgufArray arr;
      arr.elem_type = cur.U32("array elem type");
      // Validate the element type even when the array is empty (the loop
      // below would otherwise never see an unknown type for count == 0).
      if (arr.elem_type > kGgufF64)
        Fail(cur.path, "unknown kv array element type " +
                           std::to_string(arr.elem_type));
      const uint64_t count = cur.U64("array count");
      // UNTRUSTED count: every element consumes at least 1 byte, so a count
      // beyond the remaining bytes is malformed; reject before allocating.
      if (count > cur.size - cur.pos)
        Fail(cur.path, "kv array count " + std::to_string(count) +
                           " exceeds remaining file size " +
                           std::to_string(cur.size - cur.pos));
      // Bound the recursive TOTAL number of array elements in the file: each
      // parsed element costs sizeof(GgufValue) >> 1 byte of file, so without
      // this budget a small file could amplify into ~40x its size in memory.
      // elems_parsed <= kMaxCount holds, so the subtraction cannot underflow.
      if (count > kMaxCount - elems_parsed)
        Fail(cur.path, "array element budget exceeded (more than " +
                           std::to_string(kMaxCount) +
                           " total array elements)");
      elems_parsed += count;
      arr.elems.reserve(static_cast<size_t>(count));
      for (uint64_t i = 0; i < count; ++i)
        arr.elems.push_back(
            ReadValue(cur, arr.elem_type, depth + 1, elems_parsed));
      out.v = std::move(arr);
      break;
    }
    case kGgufU64:
      out.v = cur.U64("u64 kv");
      break;
    case kGgufI64:
      out.v = static_cast<int64_t>(cur.U64("i64 kv"));
      break;
    case kGgufF64: {
      const uint64_t bits = cur.U64("f64 kv");
      double d;
      std::memcpy(&d, &bits, sizeof(d));
      out.v = d;
      break;
    }
    default:
      Fail(cur.path, "unknown kv value type " + std::to_string(type));
  }
  return out;
}

// Standard ggml type traits. Ids and block geometry mirror ggml.h's
// enum ggml_type / type_traits table (llama.cpp); recorded here so the
// reader has no ggml dependency.
//
// Ids 39-41 follow mudler's killgate llama.cpp fork
// (~/llama-phase84-attn-only-source on dgx.casa), which appends
// GGML_TYPE_NVFP4 = 40 and GGML_TYPE_Q1_0 = 41 after mainline's
// GGML_TYPE_MXFP4 = 39 (ggml/include/ggml.h:429-431). Block geometry from
// ggml/src/ggml-common.h and gguf-py/gguf/constants.py GGML_QUANT_SIZES.
// See .agents/specs/gguf-nvfp4-notes.md for the full layout writeup.
const GgmlTypeTraits* FindGgmlTraits(uint32_t type) {
  switch (type) {
    case 0: {
      static constexpr GgmlTypeTraits t{1, 4, "F32"};
      return &t;
    }
    case 1: {
      static constexpr GgmlTypeTraits t{1, 2, "F16"};
      return &t;
    }
    case 2: {
      static constexpr GgmlTypeTraits t{32, 18, "Q4_0"};
      return &t;
    }
    case 3: {
      // block_q4_1 (llama.cpp ggml-common.h): f16 d + f16 m + QK4_1/2 u8 qs
      // = 2 + 2 + 16 = 20, QK4_1 = 32. A STANDARD ggml id (not a fork
      // extension): tabulated here for completeness alongside Q5_1.
      static constexpr GgmlTypeTraits t{32, 20, "Q4_1"};
      return &t;
    }
    case 6: {
      // block_q5_0 (llama.cpp @ b10451 ggml-common.h:229-235): f16 d
      // + u8 qh[4] + QK5_0/2 u8 qs = 2 + 4 + 16 = 22, QK5_0 = 32. Reachable
      // because llama.cpp's `tensor_type_fallback` maps Q4_K -> Q5_0 for a
      // tensor whose row is not a multiple of 256, which is every
      // `qwen4exp` expert row (640) and its per-layer table row (160).
      static constexpr GgmlTypeTraits t{32, 22, "Q5_0"};
      return &t;
    }
    case 7: {
      // block_q5_1 (llama.cpp ggml-common.h:237-249): f16 d + f16 m
      // + u8 qh[4] + QK5_1/2 u8 qs = 2 + 2 + 4 + 16 = 24, QK5_1 = 32.
      // qh is FOUR bytes, not one: each of the 32 quants contributes its
      // fifth high bit to one bit of qh. static_assert in ggml-common.h:
      // sizeof(block_q5_1) == 2*sizeof(ggml_half) + sizeof(uint32_t) + QK5_1/2.
      // The pinned mudler/Inkling-2-Mini-APEX-GGUF Balanced artifact stores
      // 14 `blk.N.ffn_down_exps.weight` tensors in it; without this row
      // GgufFile::Open refused the whole 20 GB file over those 14 tensors.
      static constexpr GgmlTypeTraits t{32, 24, "Q5_1"};
      return &t;
    }
    case 8: {
      static constexpr GgmlTypeTraits t{32, 34, "Q8_0"};
      return &t;
    }
    case 10: {
      static constexpr GgmlTypeTraits t{256, 84, "Q2_K"};
      return &t;
    }
    case 11: {
      static constexpr GgmlTypeTraits t{256, 110, "Q3_K"};
      return &t;
    }
    case 12: {
      static constexpr GgmlTypeTraits t{256, 144, "Q4_K"};
      return &t;
    }
    case 13: {
      static constexpr GgmlTypeTraits t{256, 176, "Q5_K"};
      return &t;
    }
    case 14: {
      static constexpr GgmlTypeTraits t{256, 210, "Q6_K"};
      return &t;
    }
    case 15: {
      // Q8_K is decoder-only for embedding tables. Stock llama.cpp 10bf611e,
      // ggml-common.h:370-376: f32 delta + 256 i8 values + 16 i16 sums.
      static constexpr GgmlTypeTraits t{256, 292, "Q8_K"};
      return &t;
    }
    case 16: {
      // block_iq2_xxs (ggml-common.h:371-374): f16 d + QK_K/8 u16 qs
      // = 2 + 32*2 = 66. The Unsloth-Dynamic `UD-IQ2_XXS` ~2-bit encoding
      // (codebook dequant in cpu_quant_dequant.cpp / vt DType kIQ2_XXS).
      static constexpr GgmlTypeTraits t{256, 66, "IQ2_XXS"};
      return &t;
    }
    case 17: {
      // block_iq2_xs (llama.cpp @ b10451 ggml-common.h:388-392): f16 d
      // + QK_K/8 u16 qs + QK_K/32 u8 scales = 2 + 64 + 8 = 74, i.e. 2.3125 bpw.
      // The `unsloth/GLM-5.3-Flash-GGUF UD-Q2_K_XL` arm stores 82 of its 1412
      // tensors in it — the `ffn_gate_exps`/`ffn_up_exps` routed experts — and
      // its absence stopped `LoadedEngine::FromModelDir` at
      // `blk.3.ffn_gate_exps.weight` before any dequant code ran (#2240).
      // Codebook dequant in cpu_quant_dequant.cpp / vt DType kIQ2_XS.
      static constexpr GgmlTypeTraits t{256, 74, "IQ2_XS"};
      return &t;
    }
    case 18: {
      // block_iq3_xxs (ggml-common.h:385-400): f16 d + 3*QK_K/8 u8 qs
      // = 2 + 96 = 98. The Unsloth-Dynamic `UD-IQ2_XXS` down-projection
      // routed experts (`ffn_down_exps`) are IQ3_XXS (codebook dequant in
      // cpu_quant_dequant.cpp / vt DType kIQ3_XXS). DeepSeek-V4 W8.
      static constexpr GgmlTypeTraits t{256, 98, "IQ3_XXS"};
      return &t;
    }
    case 19: {
      // block_iq1_s (ggml-common.h:414-419): f16 d + QK_K/8 u8 qs
      // + QK_K/32 u16 qh = 2 + 32 + 16 = 50, i.e. 1.5625 bpw. Carries the
      // routed experts (ffn_down/gate/up_exps) of the Qwen3.8-2.4T-A95B
      // UD-IQ1_S checkpoint, which is 96.92 % of that model's parameters
      // (codebook dequant in cpu_quant_dequant.cpp / vt DType kIQ1_S).
      static constexpr GgmlTypeTraits t{256, 50, "IQ1_S"};
      return &t;
    }
    case 29: {
      // block_iq1_m (llama.cpp @ b10451 ggml-common.h:429-437): QK_K/8 u8 qs
      // + QK_K/16 u8 qh + QK_K/32 u8 scales = 32 + 16 + 8 = 56, i.e. 1.75 bpw.
      // ggml type id 29. The one IQ1 block with NO f16 field: the super-block
      // scale is an f16 spliced out of the four top nibbles of scales[0..3],
      // and the eight 3-bit sub-block scales share those same bytes. The four
      // attention-gate tensors of the
      // `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` IQ3_XXS artifact are IQ1_M;
      // this case is what stops `GgufFile::Open` refusing the whole file.
      static constexpr GgmlTypeTraits t{256, 56, "IQ1_M"};
      return &t;
    }
    case 66: {
      // block_iq1_xxxs, from the PINNED FORK oracle `llama-cpp-unsloth`
      // (.agents/oracles/llama-cpp-unsloth.md, ggml-common.h:478-483):
      // f16 d + QK_K/8 u8 qs + QK_K/64 u8 sc = 2 + 32 + 4 = 38, i.e.
      // 1.1875 bpw. NO upstream llama.cpp defines type 66. It carries the
      // routed experts of the Qwen3.8-2.4T-A95B UD-Q1_0 checkpoint, 96.92 % of
      // that model's parameters.
      static constexpr GgmlTypeTraits t{256, 38, "IQ1_XXXS"};
      return &t;
    }
    case 20: {
      // block_iq4_nl (llama.cpp @ b10451 ggml-common.h:447-452): f16 d
      // + QK4_NL/2 u8 qs = 2 + 16 = 18, QK4_NL = 32. Q4_0's geometry with a
      // 16-entry NON-LINEAR codebook (ggml-common.h:1120 kvalues_iq4nl) in
      // place of the affine step. This is the encoding
      // `unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ1_S` uses for all 48
      // `ffn_down_exps` (K = 640) and for the 20M-entry n-gram table
      // `per_layer_token_embd.weight` (row = 160): neither dimension is a
      // multiple of 256, so no K-quant can encode them and llama.cpp's
      // `tensor_type_fallback` drops IQ4_XS -> IQ4_NL.
      static constexpr GgmlTypeTraits t{32, 18, "IQ4_NL"};
      return &t;
    }
    case 21: {
      // block_iq3_s (llama.cpp @ b10451 ggml-common.h:413-422): f16 d
      // + QK_K/4 u8 qs + QK_K/32 u8 qh + QK_K/8 u8 signs + IQ3S_N_SCALE u8
      // scales = 2 + 64 + 8 + 32 + 4 = 110, i.e. 3.4375 bpw. The 110 is the
      // ORACLE's own `sizeof(block_iq3_s)`, printed by the harness in
      // tests/vt/iq3s_golden_vectors.h, not read off the struct.
      //
      // NOT a variant of IQ3_XXS (18) despite the family name: it reads the
      // 512-entry `iq3s_grid` where IQ3_XXS reads the 256-entry `iq3xxs_grid`,
      // it splices the ninth index bit out of `qh`, and it carries DIRECT sign
      // bytes like IQ2_S rather than IQ3_XXS's packed `ksigns_iq2xs` selector.
      // Codebook dequant in cpu_quant_dequant.cpp / vt DType kIQ3_S.
      //
      // This was the ONE hole in the i-quant run 16..23, and it cost whole
      // artifacts: `unsloth/Qwen3.8-27B-GGUF UD-Q4_K_M` stores 4 of its 866
      // tensors in it (UD-Q4_K_XL stores 1 of 866), and every other id in
      // those files this reader already handled, so `GgufFile::Open` refused
      // 16.4 GB of loadable weights over 146 MiB of them (#2510).
      static constexpr GgmlTypeTraits t{256, 110, "IQ3_S"};
      return &t;
    }
    case 22: {
      // block_iq2_s: f16 d + QK_K/4 qs + QK_K/16 qh = 2 + 64 + 16.
      // Used by the APEX "Mini" GGUFs for expert weights.
      static constexpr GgmlTypeTraits t{256, 82, "IQ2_S"};
      return &t;
    }
    case 23: {
      // block_iq4_xs (llama.cpp @ b10451 ggml-common.h:454-459): f16 d
      // + u16 scales_h + QK_K/64 scales_l + QK_K/2 qs = 2 + 2 + 4 + 128 = 136.
      // Used by the APEX "Quality" GGUFs, and by 3 tensors of the
      // `unsloth/GLM-5.3-Flash-GGUF UD-Q2_K_XL` arm. Same `kValuesIq4nl`
      // codebook as IQ4_NL; the delta is the super-block scale layout
      // (cpu_quant_dequant.cpp / vt DType kIQ4_XS).
      static constexpr GgmlTypeTraits t{256, 136, "IQ4_XS"};
      return &t;
    }
    case 24: {
      static constexpr GgmlTypeTraits t{1, 1, "I8"};
      return &t;
    }
    case 25: {
      static constexpr GgmlTypeTraits t{1, 2, "I16"};
      return &t;
    }
    case 26: {
      static constexpr GgmlTypeTraits t{1, 4, "I32"};
      return &t;
    }
    case 27: {
      static constexpr GgmlTypeTraits t{1, 8, "I64"};
      return &t;
    }
    case 28: {
      static constexpr GgmlTypeTraits t{1, 8, "F64"};
      return &t;
    }
    case 30: {
      static constexpr GgmlTypeTraits t{1, 2, "BF16"};
      return &t;
    }
    case 39: {
      // block_mxfp4: u8 E8M0 scale + 16 bytes packed 4-bit e2m1
      // (fork ggml-common.h:205-210; same id/geometry as mainline).
      static constexpr GgmlTypeTraits t{32, 17, "MXFP4"};
      return &t;
    }
    case 40: {
      // Killgate fork extension: block_nvfp4 = 4 u8 UE4M3 scales (one per
      // 16-element sub-block) + 32 bytes packed 4-bit e2m1 => 64 elems in
      // 36 bytes. No per-tensor scale tensor; blocks are self-contained.
      // Fork ggml-common.h:211-217, ggml.h:430, gguf-py constants.py
      // GGML_QUANT_SIZES: (64, 4 + 32).
      static constexpr GgmlTypeTraits t{64, 36, "NVFP4"};
      return &t;
    }
    case 41: {
      // Killgate fork extension: block_q1_0 = f16 d + QK1_0/8 bit-packed
      // quants => 128 elems in 18 bytes (fork ggml-common.h:177-182,
      // ggml.h:431).
      static constexpr GgmlTypeTraits t{128, 18, "Q1_0"};
      return &t;
    }
    default:
      return nullptr;
  }
}

}  // namespace

const GgmlTypeTraits& GgmlTraits(uint32_t type) {
  const GgmlTypeTraits* t = FindGgmlTraits(type);
  if (t == nullptr)
    throw std::runtime_error("gguf: unknown ggml type id " +
                             std::to_string(type));
  return *t;
}

GgufFile GgufFile::OpenOne(const std::string& path) {
  GgufFile f;  // fully constructed: dtor cleans up on any throw below
  f.path_ = path;

  // The mapping is refcounted from the moment it exists, so every early Fail()
  // below unmaps through the same one owner (GgufMapping's destructor).
  auto mapping = std::make_shared<GgufMapping>();
  f.map_ = mapping;

  try {
    mapping->file = detail::ReadOnlyFileMapping::Open(Utf8Path(path));
  } catch (const std::runtime_error& e) {
    Fail(path, e.what());
  }
  const size_t file_size = mapping->file->size();
  Cursor cur{mapping->file->data(), file_size, 0, path};

  // Header: magic "GGUF", u32 version, u64 tensor_count, u64 kv_count.
  cur.Need(4, "magic");
  if (std::memcmp(cur.base, "GGUF", 4) != 0) Fail(path, "bad magic (not GGUF)");
  cur.pos = 4;
  const uint32_t version = cur.U32("version");
  if (version != 2 && version != 3) {
    // A byte-swapped version field means a big-endian GGUFv3 file.
    if (version == 0x02000000u || version == 0x03000000u)
      Fail(path, "big-endian GGUF is not supported (version field is "
                 "byte-swapped)");
    Fail(path, "unsupported GGUF version " + std::to_string(version) +
                   " (v2 and v3 little-endian are supported)");
  }
  // v2 and v3 share the little-endian layout below (v1 used u32 lengths and
  // is rejected above; v3 only added the big-endian variant over v2).

  const uint64_t tensor_count = cur.U64("tensor count");
  const uint64_t kv_count = cur.U64("kv count");
  // UNTRUSTED counts: cap before any allocation is sized from them.
  if (tensor_count > kMaxCount)
    Fail(path, "tensor count " + std::to_string(tensor_count) +
                   " exceeds sanity cap " + std::to_string(kMaxCount));
  if (kv_count > kMaxCount)
    Fail(path, "kv count " + std::to_string(kv_count) +
                   " exceeds sanity cap " + std::to_string(kMaxCount));

  // Metadata kvs. `array_elems` is the file-wide recursive total of array
  // elements parsed, budgeted at kMaxCount inside ReadValue.
  uint64_t array_elems = 0;
  for (uint64_t i = 0; i < kv_count; ++i) {
    std::string key = cur.Str("kv key");
    const uint32_t type = cur.U32("kv value type");
    GgufValue value = ReadValue(cur, type, 0, array_elems);
    auto [it, inserted] = f.kvs_.emplace(std::move(key), std::move(value));
    if (!inserted) Fail(path, "duplicate kv key \"" + it->first + "\"");
  }

  // Data-section alignment: kv "general.alignment" (u32, power of two),
  // default 32.
  uint64_t alignment = 32;
  if (auto it = f.kvs_.find("general.alignment"); it != f.kvs_.end()) {
    const uint32_t* a = std::get_if<uint32_t>(&it->second.v);
    if (a == nullptr) Fail(path, "general.alignment kv is not a u32");
    if (*a == 0 || (*a & (*a - 1)) != 0)
      Fail(path, "general.alignment " + std::to_string(*a) +
                     " is not a power of two");
    alignment = *a;
  }

  // Tensor infos: name, u32 n_dims, u64 dims (ggml order), u32 type, u64
  // offset relative to the data section start. Offsets are stashed and
  // bounds-checked after the loop, once the section start (which depends on
  // the end of this table) is known.
  // (No reserve from the UNTRUSTED count: a tiny hostile file could claim
  // the full cap; growth stays proportional to bytes actually parsed.)
  std::vector<uint64_t> offsets;
  for (uint64_t i = 0; i < tensor_count; ++i) {
    GgufTensorInfo t;
    t.name = cur.Str("tensor name");
    const uint32_t n_dims = cur.U32("tensor n_dims");
    if (n_dims > 4)  // mirrors GGML_MAX_DIMS
      Fail(path, "tensor \"" + t.name + "\" has " + std::to_string(n_dims) +
                     " dims, exceeding GGML_MAX_DIMS (4)");
    uint64_t numel = 1;
    std::vector<uint64_t> ggml_dims(n_dims);
    for (uint32_t d = 0; d < n_dims; ++d) {
      const uint64_t dim = cur.U64("tensor dim");
      if (dim > static_cast<uint64_t>(INT64_MAX))
        Fail(path, "tensor \"" + t.name + "\" dim does not fit in int64");
      // UNTRUSTED dims: division-check before each multiply so a huge
      // declared shape throws instead of wrapping (same guard pattern as
      // vt::StepArena / the safetensors reader).
      if (dim != 0 && numel > UINT64_MAX / dim)
        Fail(path, "tensor \"" + t.name + "\" element count overflows");
      numel *= dim;
      ggml_dims[d] = dim;
    }
    // ggml stores the fastest-varying dim first; reverse into torch
    // row-major order.
    t.shape.reserve(n_dims);
    for (uint32_t d = n_dims; d > 0; --d)
      t.shape.push_back(static_cast<int64_t>(ggml_dims[d - 1]));

    t.ggml_type = cur.U32("tensor ggml type");
    const GgmlTypeTraits* traits = FindGgmlTraits(t.ggml_type);
    if (traits == nullptr)
      Fail(path, "tensor \"" + t.name + "\" has unknown ggml type id " +
                     std::to_string(t.ggml_type));
    const uint64_t block_elems = static_cast<uint64_t>(traits->block_elems);
    const uint64_t block_bytes = static_cast<uint64_t>(traits->block_bytes);
    if (numel % block_elems != 0)
      Fail(path, "tensor \"" + t.name + "\" element count " +
                     std::to_string(numel) + " is not divisible by the " +
                     traits->name + " block size " +
                     std::to_string(block_elems));
    const uint64_t blocks = numel / block_elems;
    if (blocks != 0 && block_bytes > UINT64_MAX / blocks)
      Fail(path, "tensor \"" + t.name + "\" byte size overflows");
    const uint64_t nbytes = blocks * block_bytes;

    const uint64_t offset = cur.U64("tensor offset");
    if (offset % alignment != 0)
      Fail(path, "tensor \"" + t.name + "\" offset " + std::to_string(offset) +
                     " is not a multiple of the alignment " +
                     std::to_string(alignment));
    t.nbytes = static_cast<size_t>(nbytes);
    if (!f.index_.emplace(t.name, f.tensors_.size()).second)
      Fail(path, "duplicate tensor name \"" + t.name + "\"");
    offsets.push_back(offset);
    f.tensors_.push_back(std::move(t));
  }

  // Data section starts at the alignment boundary after the tensor-info
  // table. cur.pos <= file_size and alignment <= 2^32, so this cannot
  // overflow size_t.
  const size_t data_start =
      (cur.pos + static_cast<size_t>(alignment) - 1) /
      static_cast<size_t>(alignment) * static_cast<size_t>(alignment);
  if (data_start > file_size && !f.tensors_.empty())
    Fail(path, "data section start " + std::to_string(data_start) +
                   " is beyond the file size " + std::to_string(file_size));
  const size_t data_section =
      data_start <= file_size ? file_size - data_start : 0;

  // Bind tensor spans, bounds-checking each (UNTRUSTED) offset + nbytes
  // against the data section.
  for (size_t i = 0; i < f.tensors_.size(); ++i) {
    GgufTensorInfo& t = f.tensors_[i];
    const uint64_t offset = offsets[i];
    if (offset > data_section || t.nbytes > data_section - offset)
      Fail(path, "tensor \"" + t.name + "\" span [" + std::to_string(offset) +
                     ", " + std::to_string(offset + t.nbytes) +
                     ") exceeds the data section size " +
                     std::to_string(data_section));
    t.data = cur.base + data_start + static_cast<size_t>(offset);
  }

  return f;
}

namespace {

// llama.cpp split-GGUF naming: a shard path ends with "-NNNNN-of-MMMMM.gguf"
// (5-digit, 1-based). On a match, fills `total` (MMMMM) and returns a builder
// that yields the path of the i-th shard (0-based): the "-NNNNN-" run is
// rewritten to i+1, everything else preserved. Returns false for a plain name.
struct SplitNaming {
  std::string prefix;  // up to and including the leading '-' before NNNNN
  std::string suffix;  // "-of-MMMMM.gguf"
  int total = 0;
  std::string Shard(int i0) const {
    char num[16];  // wide enough for any int (silences -Wformat-truncation)
    std::snprintf(num, sizeof(num), "%05d", i0 + 1);
    return prefix + num + suffix;
  }
};

bool DetectSplit(const std::string& path, SplitNaming* out) {
  static const std::string kExt = ".gguf";
  if (path.size() < kExt.size() || path.compare(path.size() - kExt.size(),
                                                kExt.size(), kExt) != 0)
    return false;
  // Locate the LAST "-of-" and require 5 digits on each side: "-NNNNN-of-MMMMM".
  const std::string mark = "-of-";
  const size_t body = path.size() - kExt.size();  // index just past MMMMM
  const size_t of = path.rfind(mark, body);
  if (of == std::string::npos) return false;
  const size_t m_begin = of + mark.size();
  if (body != m_begin + 5) return false;  // MMMMM must be exactly 5 digits
  const size_t n_begin = of >= 5 ? of - 5 : std::string::npos;
  if (n_begin == std::string::npos || of < 5) return false;
  if (n_begin == 0 || path[n_begin - 1] != '-') return false;  // "-NNNNN-of-"
  auto all_digits = [&](size_t b, size_t n) {
    for (size_t i = 0; i < n; ++i)
      if (path[b + i] < '0' || path[b + i] > '9') return false;
    return true;
  };
  if (!all_digits(n_begin, 5) || !all_digits(m_begin, 5)) return false;
  const int total = std::atoi(path.substr(m_begin, 5).c_str());
  if (total <= 1) return false;  // single shard: nothing to merge
  out->total = total;
  out->prefix = path.substr(0, n_begin);              // "...-"
  out->suffix = path.substr(of);                       // "-of-MMMMM.gguf"
  return true;
}

}  // namespace

GgufFile GgufFile::Open(const std::string& path) {
  SplitNaming sn;
  if (std::getenv("VT_GGUF_NO_SPLIT") != nullptr || !DetectSplit(path, &sn))
    return OpenOne(path);

  // Shard 00001 carries the full KV header + its own tensors; open it as the
  // primary. Every additional shard contributes only its tensor table (its KVs
  // are just split.no/count and would collide), and its mapping is kept alive
  // under the primary so borrowed spans in any shard stay valid.
  GgufFile f = OpenOne(sn.Shard(0));
  // A merged file is addressed by its shard-00001 name in error messages.
  f.path_ = path;
  auto* primary = const_cast<GgufMapping*>(f.map_.get());  // just made non-const
  for (int i = 1; i < sn.total; ++i) {
    GgufFile s = OpenOne(sn.Shard(i));
    primary->siblings.push_back(s.map_);  // pin the shard mapping to the primary
    for (GgufTensorInfo& t : s.tensors_) {
      // Each tensor's `data` already points into shard i's mapping (now pinned).
      if (!f.index_.emplace(t.name, f.tensors_.size()).second)
        Fail(path, "tensor \"" + t.name + "\" appears in more than one shard");
      f.tensors_.push_back(std::move(t));
    }
  }
  // Soft cross-check against the file's own split.count, when present.
  if (const GgufValue* c = f.FindKv("split.count")) {
    const auto* u = std::get_if<uint16_t>(&c->v);
    const auto* w = std::get_if<uint32_t>(&c->v);
    const int64_t declared = u != nullptr ? *u : (w != nullptr ? *w : sn.total);
    if (declared != sn.total)
      Fail(path, "split.count " + std::to_string(declared) +
                     " disagrees with the filename's -of-" +
                     std::to_string(sn.total));
  }
  return f;
}

const GgufValue* GgufFile::FindKv(const std::string& key) const {
  auto it = kvs_.find(key);
  return it == kvs_.end() ? nullptr : &it->second;
}

const GgufTensorInfo& GgufFile::Get(const std::string& name) const {
  auto it = index_.find(name);
  if (it == index_.end()) Fail(path_, "no tensor named \"" + name + "\"");
  return tensors_[it->second];
}

bool GgufFile::OwnsSpan(const uint8_t* data, size_t nbytes) const {
  const auto in = [&](const GgufMapping* m) {
    if (m == nullptr || m->file == nullptr) return false;
    const auto* base = m->file->data();
    const size_t size = m->file->size();
    return data >= base && nbytes <= size &&
           static_cast<size_t>(data - base) <= size - nbytes;
  };
  if (in(map_.get())) return true;
  // A merged split GGUF: the span may live in any sibling shard's mapping.
  if (map_ != nullptr)
    for (const auto& s : map_->siblings)
      if (in(s.get())) return true;
  return false;
}

GgufFile::SpanSource GgufFile::SourceOfSpan(const uint8_t* data,
                                            size_t nbytes) const {
  SpanSource out;
#if !defined(_WIN32)
  const auto try_map = [&](const GgufMapping* m) {
    if (m == nullptr || m->file == nullptr) return false;
    const uint8_t* base = m->file->data();
    const size_t size = m->file->size();
    if (base == nullptr || data < base) return false;
    const size_t off = static_cast<size_t>(data - base);
    if (nbytes > size || off > size - nbytes) return false;
    out.fd = m->file->fd();
    out.offset = off;
    return out.fd >= 0;
  };
  if (try_map(map_.get())) return out;
  if (map_ != nullptr)
    for (const auto& s : map_->siblings)
      if (try_map(s.get())) return out;
#else
  (void)data;
  (void)nbytes;
#endif
  return out;
}

void GgufFile::DropSpanResidency(const uint8_t* data, size_t nbytes) const {
#if defined(__unix__)
  if (!release_expanded_ || !OwnsSpan(data, nbytes)) return;
  const long ps_l = ::sysconf(_SC_PAGESIZE);
  const auto ps = static_cast<uintptr_t>(ps_l > 0 ? ps_l : 4096);
  const auto begin = reinterpret_cast<uintptr_t>(data);
  const uintptr_t end = begin + nbytes;
  // INTERIOR whole pages only: a boundary page may also hold the first/last
  // bytes of a neighbouring tensor that IS being kept in place.
  const uintptr_t page_begin = (begin + ps - 1) & ~(ps - 1);
  const uintptr_t page_end = end & ~(ps - 1);
  if (page_end > page_begin) {
    // Best-effort by contract: a failure costs resident pages, never
    // correctness, so there is nothing to report or recover.
    (void)::madvise(reinterpret_cast<void*>(page_begin),
                    static_cast<size_t>(page_end - page_begin), MADV_DONTNEED);
  }
#else
  (void)data;
  (void)nbytes;
#endif
}

// Drops THIS object's reference. The mapping itself survives while any borrowing
// weight still holds one (GgufMapping's destructor does the munmap/close).
void GgufFile::Release() noexcept { map_.reset(); }

GgufFile::~GgufFile() = default;

GgufFile::GgufFile(GgufFile&& other) noexcept
    : path_(std::move(other.path_)),
      map_(std::move(other.map_)),
      kvs_(std::move(other.kvs_)),
      tensors_(std::move(other.tensors_)),
      index_(std::move(other.index_)) {}

GgufFile& GgufFile::operator=(GgufFile&& other) noexcept {
  if (this != &other) {
    path_ = std::move(other.path_);
    map_ = std::move(other.map_);
    kvs_ = std::move(other.kvs_);
    tensors_ = std::move(other.tensors_);
    index_ = std::move(other.index_);
  }
  return *this;
}

}  // namespace vllm
