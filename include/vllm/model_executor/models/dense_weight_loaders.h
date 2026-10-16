// Shared BF16 safetensors loader helpers for dense-transformer weight loaders.
//
// Extracted VERBATIM (behavior-preserving) from the anonymous namespace of
// qwen3_5_dense_weights.cpp so a second dense arch (Qwen3 `Qwen3ForCausalLM`,
// the first additive-model bring-up — qwen3_weights.cpp) reuses the exact same
// BF16 copy/transpose/merge routines rather than re-deriving them. The only
// change vs the qwen3_5-local originals is the diagnostic message prefix, which
// is generalized from "qwen3_5 dense:" to "dense loader:" (a shared helper must
// not name one arch); the LOADED bytes are byte-identical, so the qwen3_5 load
// result is unchanged (.agents/specs/first-additive-model-qwen3-dense.md §3b
// SEAM GAP #3).
//
// Helpers (all in `vllm::dense_loaders`):
//   MakeOwned            — allocate a zero-filled OwnedTensor of dtype+shape.
//   ReadF32Scalar        — one per-tensor F32 scale, count and dtype CHECKED
//                          (#1181): the six local copies of this disagreed.
//   TransposeBf16        — bf16 [rows,cols] -> bf16 [cols,rows].
//   LoadBf16Direct       — copy a BF16 tensor verbatim (optionally reshaped).
//   LoadBf16Transposed   — BF16 [out,in] -> owned bf16 [in,out] (Matmul-B).
//   LoadMergedBf16RawNK  — concat BF16 torch-Linear shards [N_i,K] along output
//                          rows, kept RAW [N,K] with nk=true for vt::MatmulBT.
//   ProbeThroughResolver — a tensor-presence probe built from a resolver.
//   CheckProbeCanAnswerNo — refuse a presence probe that cannot answer
//                          `false`.
#pragma once

#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"  // StTensor, MaybeReleaseSourcePages
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"           // OwnedTensor, TensorResolver
#include "vllm/model_executor/models/tensor_parallel.h"           // TensorParallel/TpShard (W2)
#include "vt/dtype.h"
#include "vt/unaligned.h"

namespace vllm {
namespace dense_loaders {

// Allocate a zero-filled owned host tensor of the given dtype and shape.
inline OwnedTensor MakeOwned(vt::DType dt, const std::vector<int64_t>& shape) {
  OwnedTensor o;
  o.dtype = dt;
  o.rank = static_cast<int>(shape.size());
  VT_CHECK(o.rank <= vt::kMaxRank, "dense loader: rank exceeds kMaxRank");
  int64_t n = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  o.bytes.resize(static_cast<size_t>(n) * vt::SizeOf(dt));
  return o;
}

// `[96, 40]`, `[8]`, `[]` -- the shape as a reader can compare it against a
// checkpoint's own header.
inline std::string ShapeString(const std::vector<int64_t>& shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) s += ", ";
    s += std::to_string(shape[i]);
  }
  return s + "]";
}

// --- Tensor-presence probes (FIX-PROBE-CANNOT-SAY-NO, issue #1258) -----------
//
// A name no checkpoint carries. Its ONLY use is asking a presence probe whether
// it is capable of its negative answer.
inline constexpr const char* kAbsentProbeSentinel =
    "__vllm_cpp__a_tensor_no_checkpoint_carries__";

// "Does this checkpoint carry `name`?", built from the ONE thing a resolver-only
// loader seam is given.
//
// `TensorResolver` returns a REFERENCE, so absence has exactly one
// representation in its contract: it throws. `SafetensorsFile::Get` documents
// that (`safetensors_reader.h`), and every fixture resolver in the tree mirrors
// it. Probing the resolver is therefore the honest answer AND the only one
// available where no name index is in scope.
//
// ONE implementation on purpose. Issue #1256 was two copies of an always-true
// stub; the repair for the first (#1257) was a hand-written `try`/`catch`, and a
// second hand-written copy of that is the same mistake pointed the other way.
//
// The resolver is captured BY VALUE. A returned lambda holding a reference to a
// caller's `std::function` is the next subtle lifetime bug in a helper that
// exists because a subtle bug got shipped twice; a `std::function` copy costs
// one allocation, once per seam call, against a load that reads gigabytes.
inline std::function<bool(const std::string&)> ProbeThroughResolver(
    const TensorResolver& get) {
  return [get](const std::string& name) {
    try {
      (void)get(name);
      return true;
    } catch (const std::exception&) {
      return false;
    }
  };
}

// A presence predicate that cannot answer `false` is a DEFECT, not a
// conservative default: it reports every optional tensor as present, and the
// first guard to ask about one refuses a checkpoint over a tensor that does not
// exist. That is issue #1256, twice, in one file.
//
// This is deliberately independent of HOW the probe was built, which is what a
// stronger type could not be: it also catches a name index populated from the
// wrong shard list and a probe a refactor detached from its data.
//
// `VT_CHECK` and not `assert`: Release defines `NDEBUG`, and a guard that
// evaporates in the configuration everything ships in is not a guard.
inline void CheckProbeCanAnswerNo(
    const std::function<bool(const std::string&)>& has, const char* seam) {
  VT_CHECK(!has(kAbsentProbeSentinel),
           std::string("dense loader: ") + seam +
               " was given a tensor-presence probe that answered YES for '" +
               kAbsentProbeSentinel +
               "', a name no checkpoint carries. A probe that cannot answer NO "
               "reports every optional tensor as present, so the next guard to "
               "ask about one refuses the checkpoint over a tensor that is not "
               "there (issues 1256, 1258)");
}

// THE per-tensor f32 scale read. One implementation, because six hand-written
// copies of it disagreed about what a scale is (issue #1181).
//
// UPSTREAM, at pin `555967922`. A per-tensor scale is not a raw tensor read
// there, it is a parameter TYPE: `PerTensorScaleParameter`
// (`vllm/model_executor/parameter.py:260-272`) asserts
// `loaded_weight.shape[0] == 1` for any non-rank-0 scale before it copies
// (`:304-309`), with the sibling shape assert in `_assert_and_load` at
// `:93-96`. The slot is allocated `torch.float32`
// (`layers/quantization/utils/fp8_utils.py:1276`), so `copy_` VALUE-converts a
// narrower on-disk dtype and never reinterprets its bytes. And the declared
// strategy -- TENSOR, CHANNEL or BLOCK -- picks the parameter type before a
// byte is read (`compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py:63,128`),
// so upstream never infers a scale's shape from its byte count. Both checks
// below are that type's whole job on this side.
//
// WHY AN EXACT COUNT AND AN EXACT DTYPE, NOT A FLOOR. The previous bound was
// `t.nbytes >= sizeof(float)`, and a floor admits every wrong answer that is
// large enough. A block-wise grid `[ceil(N/128), ceil(K/128)]` passed and was
// read as block (0, 0), which then stood in for the whole matrix. A
// per-output-channel `[out] BF16` scale passed at two bytes an element and was
// read as one float built from the first two entries. Neither raised anything,
// because both produce a finite, plausible number, and a plausible wrong scale
// yields fluent wrong tokens rather than a failure. That is the one defect
// class a token gate cannot see.
//
// A NARROW DTYPE IS REFUSED RATHER THAN CONVERTED, deliberately. Upstream
// converts, so conversion would be defensible, but no caller here can be shown
// to need it: a one-element BF16 scale has never been read correctly on this
// path, since the four-byte copy took two bytes of the scale and two bytes of
// whatever followed it. The BF16 scale layout that IS published is
// per-output-channel, which the count check refuses first whatever the dtype
// rule says. Converting would be untested code on a path no checkpoint reaches.
inline float ReadF32Scalar(const TensorResolver& get, const std::string& name) {
  const StTensor& t = get(name);
  int64_t numel = 1;
  for (const int64_t d : t.shape) numel *= d;
  VT_CHECK(numel == 1,
           "dense loader: '" + name + "' ships shape " + ShapeString(t.shape) +
               " (" + std::to_string(numel) +
               " elements), not the ONE element a per-tensor scale is");
  VT_CHECK(t.dtype == "F32",
           "dense loader: '" + name + "' ships dtype " + t.dtype +
               ", not the F32 a per-tensor scale is");
  VT_CHECK(t.data != nullptr && t.nbytes == sizeof(float),
           "dense loader: '" + name +
               "' is a one-element F32 scale but does not carry 4 readable bytes");
  float v = 0.0F;
  std::memcpy(&v, t.data, sizeof(float));
  return v;
}

// Block-wise (fine-grained) FP8 projection: `<proj>.weight` F8_E4M3 [N, K]
// beside `<proj>.weight_scale_inv` [cdiv(N, block_n), cdiv(K, block_k)] ->
// `Fp8BlockWeight`. MODEL-FP8-BLOCK-WEIGHT, #1189 M3, spec
// `.agents/specs/model-fp8-block-weight.md`.
//
// The fp8 bytes are kept RAW in the on-disk [N=out, K=in] orientation, as
// `LoadFp8Raw` does for the per-tensor arm: no dequant and no transpose, so the
// projection costs one byte per element and every scale decision stays inside
// the GEMM where upstream applies it (per K-block, in the mainloop -- see
// `.agents/specs/vt-matmul-fp8-block-ref.md`).
//
// THE SCALE IS WIDENED TO F32, NOT REINTERPRETED. Upstream allocates the
// parameter `torch.float32` (`utils/fp8_utils.py:1276,1283-1296`) and loads the
// checkpoint tensor into it with `self.data.copy_()`
// (`vllm/model_executor/parameter.py:97`), which CONVERTS. `Qwen/Qwen3.8-27B-FP8`
// ships the tensor `BF16`, so the resident f32 is the mirror rather than a
// widening: it is the dtype upstream carries, `vt::MatmulFp8BlockScaled` refuses
// anything else, and bf16 -> f32 is exact. The switch below has NO default
// branch that memcpy's bytes, because #1181 landed a guard for exactly that.
// `vt::LoadUnaligned` because a safetensors tensor's offset is the running byte
// total of everything ahead of it and can be odd (#627).
//
// The shape check is upstream's own: the allocation at `fp8_utils.py:1283-1296`
// uses `cdiv` on BOTH axes and `parameter.py:95-98` then asserts the loaded
// tensor matches it exactly. A short final block is legal and must work.
inline Fp8BlockWeight LoadFp8BlockRaw(const TensorResolver& get,
                                      const std::string& proj, int64_t block_n,
                                      int64_t block_k) {
  VT_CHECK(block_n > 0 && block_k > 0,
           "dense loader: '" + proj +
               "' block-wise FP8 needs positive block dimensions, got [" +
               std::to_string(block_n) + ", " + std::to_string(block_k) + "]");
  const StTensor& w = get(proj + ".weight");
  VT_CHECK(w.dtype == "F8_E4M3",
           "dense loader: '" + proj + ".weight' ships dtype " + w.dtype +
               ", not the F8_E4M3 a block-wise FP8 weight is");
  VT_CHECK(w.shape.size() == 2,
           "dense loader: '" + proj + ".weight' ships shape " +
               ShapeString(w.shape) +
               ", not the 2-D [out_features, in_features] a block-wise FP8 "
               "weight is");
  Fp8BlockWeight r;
  r.n = w.shape[0];
  r.k = w.shape[1];
  r.block_n = block_n;
  r.block_k = block_k;

  const std::string scale_name = proj + ".weight_scale_inv";
  const StTensor& s = get(scale_name);
  const int64_t rows = (r.n + block_n - 1) / block_n;
  const int64_t cols = (r.k + block_k - 1) / block_k;
  VT_CHECK(
      s.shape.size() == 2 && s.shape[0] == rows && s.shape[1] == cols,
      "dense loader: '" + scale_name + "' ships shape " +
          ShapeString(s.shape) + ", not the " +
          ShapeString(std::vector<int64_t>{rows, cols}) +
          " a [" + std::to_string(r.n) + ", " + std::to_string(r.k) +
          "] weight quantized in [" + std::to_string(block_n) + ", " +
          std::to_string(block_k) +
          "] blocks needs. Both dimensions round UP (ceil), so a short final "
          "block still owns a scale");
  const int64_t count = rows * cols;
  r.scale = MakeOwned(vt::DType::kF32, {rows, cols});
  auto* dst = reinterpret_cast<float*>(r.scale.bytes.data());
  if (s.dtype == "BF16") {
    VT_CHECK(s.data != nullptr &&
                 s.nbytes == static_cast<size_t>(count) * sizeof(uint16_t),
             "dense loader: '" + scale_name +
                 "' is a BF16 block scale but does not carry " +
                 std::to_string(count * 2) + " readable bytes");
    for (int64_t i = 0; i < count; ++i)
      dst[i] = vt::BF16ToF32(vt::LoadUnaligned<uint16_t>(s.data + i * 2));
  } else if (s.dtype == "F32") {
    VT_CHECK(s.data != nullptr &&
                 s.nbytes == static_cast<size_t>(count) * sizeof(float),
             "dense loader: '" + scale_name +
                 "' is an F32 block scale but does not carry " +
                 std::to_string(count * 4) + " readable bytes");
    for (int64_t i = 0; i < count; ++i)
      dst[i] = vt::LoadUnaligned<float>(s.data + i * 4);
  } else {
    VT_CHECK(false,
             "dense loader: '" + scale_name + "' ships dtype " + s.dtype +
                 ", and a block-wise FP8 scale is read as BF16 or F32 only. "
                 "Upstream loads it into an F32 parameter with a CONVERTING "
                 "copy, so a narrower dtype is widened by VALUE; reading its "
                 "bytes as another dtype is the defect issue #1181 fixed");
  }
  MaybeReleaseSourcePages(s.data, s.nbytes);

  r.packed = MakeOwned(vt::DType::kI8, {r.n, r.k});
  VT_CHECK(w.nbytes == r.packed.bytes.size(),
           "dense loader: '" + proj +
               ".weight' block-wise FP8 byte-size mismatch");
  std::memcpy(r.packed.bytes.data(), w.data, w.nbytes);
  MaybeReleaseSourcePages(w.data, w.nbytes);
  return r;
}

// src bf16 [rows, cols] -> dst bf16 [cols, rows].
inline void TransposeBf16(const void* src, int64_t rows, int64_t cols,
                          uint16_t* dst) {
  const auto* bytes = static_cast<const uint8_t*>(src);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      const int64_t source_index = r * cols + c;
      dst[c * rows + r] =
          vt::LoadUnaligned<uint16_t>(bytes + source_index * 2);
    }
  }
}

// --- FP8 shard materialization (shared by every BF16 loader below) -----------
// The 2026-08 Qwen3.6-27B NVFP4 republishes quantize parts of the tower to
// per-tensor or per-output-channel FP8 while leaving the rest BF16, and they do
// not agree on WHICH parts: nvidia/Qwen3.6-27B-NVFP4 ships FP8 `linear_attn`
// in_proj_qkv/in_proj_z/out_proj with scalar F32 scales next to NVFP4 attention
// and MLP, while unsloth @ccdaab7e went FP8 across the whole tower with BF16
// per-output-channel scales. Rather than teach each loader its own dtype rules,
// every BF16 entry point routes its source bytes through this one materializer.
//
// Returns a pointer to BF16 [rows, cols] bytes: the mmap'd source itself when the
// tensor is already BF16 (zero copy, unchanged behavior), or `staging` after
// dequantization. A per-output-channel scale read as per-tensor would be
// silently WRONG rather than loud, so the element count decides and anything
// else is rejected.
inline const uint8_t* MaterializeBf16Source(const TensorResolver& get,
                                            const std::string& name,
                                            const StTensor& t,
                                            std::vector<uint16_t>* staging) {
  if (t.dtype == "BF16") return static_cast<const uint8_t*>(t.data);
  VT_CHECK(t.dtype == "F8_E4M3",
           "dense loader: unsupported dtype '" + t.dtype + "' for " + name +
               "; supported: BF16, F8_E4M3 (+ <name>_scale)");
  VT_CHECK(t.shape.size() == 2,
           "dense loader: expected 2-D weight for FP8 " + name);
  const int64_t rows = t.shape[0];
  const int64_t cols = t.shape[1];
  const StTensor& sc = get(name + "_scale");
  const int64_t n_scale =
      static_cast<int64_t>(sc.nbytes) / (sc.dtype == "BF16" ? 2 : 4);
  VT_CHECK(n_scale == 1 || n_scale == rows,
           "dense loader: " + name +
               "_scale must be per-tensor or one value per output row");
  staging->resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
  for (int64_t r = 0; r < rows; ++r) {
    const int64_t si = (n_scale == 1) ? 0 : r;
    float scale = 1.0F;
    if (sc.dtype == "BF16") {
      uint16_t h = 0;
      std::memcpy(&h, static_cast<const uint8_t*>(sc.data) + si * 2, 2);
      const uint32_t bits = static_cast<uint32_t>(h) << 16;
      std::memcpy(&scale, &bits, sizeof(scale));
    } else {
      std::memcpy(&scale, static_cast<const uint8_t*>(sc.data) + si * 4,
                  sizeof(scale));
    }
    DequantFp8ToBf16(static_cast<const uint8_t*>(t.data) + r * cols, scale, cols,
                     staging->data() + static_cast<size_t>(r) * cols);
  }
  // Deliberately NOT releasing here: every caller already calls
  // MaybeReleaseSourcePages(t.data, t.nbytes) exactly once on the same range.
  return reinterpret_cast<const uint8_t*>(staging->data());
}

// BF16 tensor copied verbatim (optionally reshaped).
//
// ENG-LOAD-DIRECT-UPLOAD (issue #150): this is a whole-range verbatim copy into
// a same-size destination -- a reshape changes no byte -- so it is one of the
// call sites that QUALIFIES for the borrow-the-mapping path. When it takes it,
// no owned buffer is allocated and the device upload reads the file mapping
// directly; the copy below is the unchanged fallback for every other case.
inline OwnedTensor LoadBf16Direct(const TensorResolver& get,
                                  const std::string& name,
                                  const std::vector<int64_t>& shape_override = {}) {
  const StTensor& t = get(name);
  std::vector<uint16_t> staging;
  const uint8_t* src = MaterializeBf16Source(get, name, t, &staging);
  std::vector<int64_t> shape = shape_override.empty() ? t.shape : shape_override;
  OwnedTensor borrowed;
  if (BorrowStTensorBytes(borrowed, t, vt::DType::kBF16, shape)) return borrowed;
  OwnedTensor o = MakeOwned(vt::DType::kBF16, shape);
  const size_t src_bytes =
      staging.empty() ? t.nbytes : staging.size() * sizeof(uint16_t);
  VT_CHECK(src_bytes == o.bytes.size(),
           "dense loader: byte-size mismatch for " + name);
  std::memcpy(o.bytes.data(), src, src_bytes);
  // LOAD-SAFETENSORS: source range now copied-then-dead; drop its resident pages
  // so the owned mirror never double-resides with the mmap (spec §page-lifetime).
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return o;
}

// ENG-LOAD-DIRECT-UPLOAD: release the resident pages behind a weight that
// BORROWED the mapping and whose bytes a merged loader has just copied into its
// own buffer. Without this the per-shard borrow would keep the source pages
// resident until the temporary shard dies, which is the residency the windowed
// release exists to avoid. A no-op for a copied (non-borrowing) shard, whose
// LoadBf16Direct/LoadCt* already released it.
inline void ReleaseBorrowedShardSource(const OwnedTensor& shard) {
  if (shard.mmap_src != nullptr)
    MaybeReleaseSourcePages(shard.mmap_src, shard.mmap_src_bytes);
}

// BF16 [out, in] -> owned bf16 [in, out] (Matmul-B layout).
inline OwnedTensor LoadBf16Transposed(const TensorResolver& get,
                                      const std::string& name) {
  const StTensor& t = get(name);
  VT_CHECK(t.shape.size() == 2, "dense loader: expected 2-D weight for " + name);
  std::vector<uint16_t> staging;
  const uint8_t* src = MaterializeBf16Source(get, name, t, &staging);
  const int64_t out_dim = t.shape[0];
  const int64_t in_dim = t.shape[1];
  OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
  TransposeBf16(src, out_dim, in_dim,
                reinterpret_cast<uint16_t*>(o.bytes.data()));
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return o;
}

// Load and concatenate raw BF16 torch-Linear weights `[N_i,K]` along their
// output rows, preserving the exact listed order and setting `nk=true` for
// vt::MatmulBT (the cuBLASLt TN fast path). This is vLLM's physical ownership
// rule for MergedColumnParallelLinear/QKVParallelLinear (one merged param).
//
// MODEL-QWEN35-GDN-EXL3 (#2495 item 4): `allow_f16` is OPT-IN and scoped to an
// EXL3 load, the same polarity `LoadModelVectorForScheme` uses for the same
// reason. `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` stores `linear_attn.in_proj_a` and
// `linear_attn.in_proj_b` at F16 -- exllamav3 keeps the unquantized linear
// remainder at fp16 because it runs the linear in fp16 -- while storing
// `conv1d`, `norm`, `A_log` and `dt_bias` beside them at BF16. Without this the
// EXL3 GDN tower is refused a second time, on a tensor that is not quantized at
// all. Teaching this loader F16 UNCONDITIONALLY would widen acceptance for every
// dense model through a conversion that drops three mantissa bits, which is the
// argument `LoadF16AsBf16Direct` already makes at its own declaration.
inline OwnedTensor LoadMergedBf16RawNK(const TensorResolver& get,
                                       const std::vector<std::string>& names,
                                       const TensorParallel* tp = nullptr,
                                       bool allow_f16 = false) {
  VT_CHECK(!names.empty(),
           "dense loader: merged BF16 projection requires at least one shard");
  int64_t in_dim = -1;
  int64_t out_dim = 0;
  std::vector<const StTensor*> shards;
  shards.reserve(names.size());
  for (const std::string& name : names) {
    const StTensor& tensor = get(name);
    // Both new Qwen3.6-27B NVFP4 publishers quantize the GDN in-projections to
    // per-tensor FP8 while leaving in_proj_a/b BF16 (nvidia/Qwen3.6-27B-NVFP4:
    // in_proj_qkv F8_E4M3 [10240,5120] + scalar weight_scale/input_scale). The
    // shard is materialized to BF16 below so the merge, the TP row split and the
    // nk=true MatmulBT orientation stay exactly as they were for a BF16 shard.
    VT_CHECK(tensor.dtype == "BF16" || tensor.dtype == "F8_E4M3" ||
                 (allow_f16 && tensor.dtype == "F16"),
             "dense loader: unsupported dtype '" + tensor.dtype + "' for " + name +
                 "; supported: BF16, F8_E4M3 (+ <name>_scale)" +
                 (allow_f16 ? ", F16" : ""));
    VT_CHECK(tensor.shape.size() == 2,
             "dense loader: expected 2-D weight for " + name);
    VT_CHECK(tensor.shape[0] > 0 && tensor.shape[1] > 0,
             "dense loader: merged BF16 shard has an empty dimension: " + name);
    VT_CHECK(tensor.data != nullptr,
             "dense loader: merged BF16 shard has null data: " + name);
    if (in_dim < 0) in_dim = tensor.shape[1];
    VT_CHECK(tensor.shape[1] == in_dim,
             "dense loader: merged BF16 shards must share input width");
    VT_CHECK(out_dim <= std::numeric_limits<int64_t>::max() - tensor.shape[0],
             "dense loader: merged BF16 output width overflow");
    out_dim += tensor.shape[0];
    shards.push_back(&tensor);
  }

  // MergedColumnParallelLinear column shard (linear.py:832-833): each named
  // constituent (gate|up, or q|k|v) is sharded INDEPENDENTLY along its output
  // rows (dim 0), then concatenated. `TpShard` returns the whole row range at
  // tp_size==1, so `sharded_out_dim == out_dim` and the copy below is
  // byte-identical to the single-GPU merge. RESIDUAL (HW-gated): QKVParallel
  // head-aware KV replication when tp_size > num_kv_heads (linear.py:1076-1079)
  // is NOT modeled here — this even split matches MergedColumnParallel and
  // QKVParallel where tp divides every head group.
  int64_t sharded_out_dim = 0;
  std::vector<ShardRange> ranges;
  ranges.reserve(shards.size());
  for (const StTensor* shard : shards) {
    const ShardRange r = TpShard(tp, shard->shape[0]);
    ranges.push_back(r);
    sharded_out_dim += r.size();
  }

  // Materialize any FP8 shard to BF16 before the merge. The scale is either a
  // single per-tensor F32 scalar (nvidia/Qwen3.6-27B-NVFP4 in_proj_qkv) or one
  // value per output row (unsloth @ccdaab7e, stored BF16). Reading a per-channel
  // scale as per-tensor would be silently WRONG rather than loud, so the row
  // count decides and anything else is rejected.
  std::vector<std::vector<uint16_t>> staged(shards.size());
  std::vector<const uint8_t*> src_ptr(shards.size());
  std::vector<size_t> src_bytes(shards.size());
  for (size_t i = 0; i < shards.size(); ++i) {
    const StTensor& shard = *shards[i];
    if (shard.dtype == "BF16") {
      src_ptr[i] = static_cast<const uint8_t*>(shard.data);
      src_bytes[i] = shard.nbytes;
      continue;
    }
    const int64_t rows = shard.shape[0];
    const int64_t cols = shard.shape[1];
    // MODEL-QWEN35-GDN-EXL3: the F16 shard is CONVERTED, never reinterpreted.
    // The two formats share a width and nothing else, so a byte copy would read
    // an exponent field as a mantissa and produce a correctly shaped, entirely
    // wrong weight.
    if (shard.dtype == "F16") {
      staged[i].resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
      const auto* src = static_cast<const uint8_t*>(shard.data);
      for (size_t e = 0; e < staged[i].size(); ++e) {
        uint16_t half = 0;
        std::memcpy(&half, src + e * 2, 2);
        staged[i][e] = vt::F32ToBF16(vt::F16ToF32(half));
      }
      MaybeReleaseSourcePages(shard.data, shard.nbytes);
      src_ptr[i] = reinterpret_cast<const uint8_t*>(staged[i].data());
      src_bytes[i] = staged[i].size() * sizeof(uint16_t);
      continue;
    }
    const StTensor& sc = get(names[i] + "_scale");
    const int64_t n_scale =
        static_cast<int64_t>(sc.nbytes) / (sc.dtype == "BF16" ? 2 : 4);
    VT_CHECK(n_scale == 1 || n_scale == rows,
             "dense loader: " + names[i] + "_scale must be per-tensor or one "
             "value per output row");
    staged[i].resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    for (int64_t r = 0; r < rows; ++r) {
      const int64_t si = (n_scale == 1) ? 0 : r;
      float scale = 1.0F;
      if (sc.dtype == "BF16") {
        uint16_t h = 0;
        std::memcpy(&h, static_cast<const uint8_t*>(sc.data) + si * 2, 2);
        const uint32_t bits = static_cast<uint32_t>(h) << 16;
        std::memcpy(&scale, &bits, sizeof(scale));
      } else {
        std::memcpy(&scale, static_cast<const uint8_t*>(sc.data) + si * 4,
                    sizeof(scale));
      }
      DequantFp8ToBf16(static_cast<const uint8_t*>(shard.data) + r * cols, scale,
                       cols, staged[i].data() + static_cast<size_t>(r) * cols);
    }
    MaybeReleaseSourcePages(shard.data, shard.nbytes);
    src_ptr[i] = reinterpret_cast<const uint8_t*>(staged[i].data());
    src_bytes[i] = staged[i].size() * sizeof(uint16_t);
  }

  VT_CHECK(sharded_out_dim <= std::numeric_limits<int64_t>::max() / in_dim,
           "dense loader: merged BF16 element count overflow");
  const auto elements =
      static_cast<uint64_t>(sharded_out_dim) * static_cast<uint64_t>(in_dim);
  VT_CHECK(elements <= std::numeric_limits<size_t>::max() / sizeof(uint16_t),
           "dense loader: merged BF16 byte count overflow");
  OwnedTensor merged = MakeOwned(vt::DType::kBF16, {sharded_out_dim, in_dim});
  const size_t row_bytes = static_cast<size_t>(in_dim) * sizeof(uint16_t);
  size_t offset = 0;
  for (size_t i = 0; i < shards.size(); ++i) {
    const StTensor& shard = *shards[i];
    const size_t full = static_cast<size_t>(shard.shape[0]) * row_bytes;
    VT_CHECK(src_bytes[i] == full,
             "dense loader: byte-size mismatch for " + names[i]);
    const ShardRange& r = ranges[i];
    const size_t src_off = static_cast<size_t>(r.begin) * row_bytes;
    const size_t copied = static_cast<size_t>(r.size()) * row_bytes;
    std::memcpy(merged.bytes.data() + offset, src_ptr[i] + src_off, copied);
    if (shard.dtype == "BF16") MaybeReleaseSourcePages(shard.data, full);
    offset += copied;
  }
  VT_CHECK(offset == merged.bytes.size(),
           "dense loader: merged BF16 byte accounting mismatch");
  merged.nk = true;
  return merged;
}

// Load and concatenate rank-1 BF16 tensors in the listed order — the vector
// analog of LoadMergedBf16RawNK, for merging the per-shard BIAS terms of a
// MergedColumnParallelLinear/QKVParallelLinear whose weights were merged by it.
// ADDED (append-only, no existing helper touched) by the OPT (`OPTForCausalLM`)
// bring-up: OPT is the first family we port whose projections carry bias
// (`config.enable_bias`, opt.py:90-104), so its merged qkv needs a merged
// [3*H] bias to go with the merged [3*H, H] weight. Generic — every future
// biased family (GPT-2, BLOOM, Falcon, ...) reuses it.
inline OwnedTensor LoadMergedBf16Vector(const TensorResolver& get,
                                        const std::vector<std::string>& names) {
  VT_CHECK(!names.empty(),
           "dense loader: merged BF16 vector requires at least one shard");
  int64_t total = 0;
  std::vector<const StTensor*> shards;
  shards.reserve(names.size());
  for (const std::string& name : names) {
    const StTensor& tensor = get(name);
    VT_CHECK(tensor.dtype == "BF16", "dense loader: expected BF16 for " + name);
    VT_CHECK(tensor.shape.size() == 1,
             "dense loader: expected 1-D vector for " + name);
    VT_CHECK(tensor.shape[0] > 0, "dense loader: merged BF16 vector shard is empty: " + name);
    VT_CHECK(tensor.data != nullptr,
             "dense loader: merged BF16 vector shard has null data: " + name);
    VT_CHECK(total <= std::numeric_limits<int64_t>::max() - tensor.shape[0],
             "dense loader: merged BF16 vector length overflow");
    total += tensor.shape[0];
    shards.push_back(&tensor);
  }
  OwnedTensor merged = MakeOwned(vt::DType::kBF16, {total});
  size_t offset = 0;
  for (size_t i = 0; i < shards.size(); ++i) {
    const StTensor& shard = *shards[i];
    const size_t expected = static_cast<size_t>(shard.shape[0]) * sizeof(uint16_t);
    VT_CHECK(shard.nbytes == expected,
             "dense loader: byte-size mismatch for " + names[i]);
    std::memcpy(merged.bytes.data() + offset, shard.data, expected);
    MaybeReleaseSourcePages(shard.data, expected);
    offset += expected;
  }
  VT_CHECK(offset == merged.bytes.size(),
           "dense loader: merged BF16 vector byte accounting mismatch");
  return merged;
}

// --- compressed-tensors NVFP4 **W4A16** (`nvfp4-pack-quantized`) ------------
// ADDED (append-only; no existing helper touched) by the Qwen3-32B-NVFP4A16
// bring-up — the QUANT-SCHEME additivity row. These are the quantized analogs of
// LoadBf16Direct / LoadMergedBf16RawNK: same [N=out, K=in] raw orientation, same
// merged-shard ownership rule, but the payload is NVFP4 instead of BF16.
//
// ON-DISK LAYOUT (verified on RedHatAI/Qwen3-32B-NVFP4A16, and the format
// `compressed-tensors` emits for `nvfp4-pack-quantized`, group_size 16,
// num_bits 4, type float, strategy tensor_group, symmetric):
//   <proj>.weight_packed        U8       [N, K/2]    two E2M1 nibbles per byte
//   <proj>.weight_scale         F8_E4M3  [N, K/16]   one fp8 scale per 16 elems
//   <proj>.weight_global_scale  F32      [1]         per-tensor DIVISOR
// and — the discriminator that makes this W4A16 rather than W4A4 — there is NO
// `<proj>.input_global_scale`. That mirrors vLLM exactly: `input_activations`
// null in the config group selects `CompressedTensorsW4A4Fp4(use_a16=True)`
// (compressed_tensors.py:696-698), whose `create_weights` registers
// `input_global_scale` ONLY when `not use_a16`
// (compressed_tensors_w4a4_nvfp4.py:86-91).
//
// SCALE CONVENTION: compressed-tensors stores the global scale as a DIVISOR, so
// scale2 is its RECIPROCAL — vLLM takes it at
// compressed_tensors_w4a4_nvfp4.py:111-114
// (`weight_global_scale = 1.0 / layer.weight_global_scale.max()`). The exact
// on-disk divisor is retained in `weight_global_scale_inv` because a merged
// (qkv / gate_up) linear must take `max()` over the shards' DIVISORS *before*
// reciprocating — reconstructing a divisor from scale2 can lose a float ULP.
// `alpha` is deliberately left 0 so `Nvfp4Weight::IsTrueW4A4()` is false and the
// weight routes to the W4A16 (Marlin, bf16-activation) dispatcher.

// True when `proj` is stored as a compressed-tensors NVFP4 linear. This is the
// per-layer scheme probe: presence of `.weight_packed` means the config group
// matched this Linear (vLLM resolves the same thing through `find_matched_target`
// + the `ignore` list, compressed_tensors.py:868-880).
// Explicit FP16 model storage. F16 is retained verbatim; BF16/F32 checkpoint
// remainders are cast straight to F16, without an intermediate BF16 rounding.
// Callers choose this policy explicitly; existing BF16 readers stay strict.
inline OwnedTensor LoadF16Direct(const TensorResolver& get, const std::string& name,
                                 const std::vector<int64_t>& shape_override = {}) {
  const StTensor& source = get(name);
  VT_CHECK(source.dtype == "F16" || source.dtype == "BF16" || source.dtype == "F32",
           "dense loader: expected F16/BF16/F32 source for FP16 " + name);
  const auto numel = [&name](const std::vector<int64_t>& shape) {
    VT_CHECK(!shape.empty() && shape.size() <= vt::kMaxRank,
             "dense loader: invalid FP16 rank for " + name);
    int64_t count = 1;
    for (const int64_t dim : shape) {
      VT_CHECK(dim > 0 && count <= std::numeric_limits<int64_t>::max() / dim,
               "dense loader: invalid FP16 shape for " + name);
      count *= dim;
    }
    return static_cast<size_t>(count);
  };
  const auto& shape = shape_override.empty() ? source.shape : shape_override;
  const size_t count = numel(source.shape);
  const size_t source_width = source.dtype == "F32" ? 4 : 2;
  VT_CHECK(numel(shape) == count && count <= SIZE_MAX / source_width &&
               source.nbytes == count * source_width && source.data != nullptr,
           "dense loader: invalid FP16 source span/reshape for " + name);
  OwnedTensor result;
  if (source.dtype == "F16" && BorrowStTensorBytes(result, source, vt::DType::kF16, shape))
    return result;
  result = MakeOwned(vt::DType::kF16, shape);
  if (source.dtype == "F16") {
    std::memcpy(result.bytes.data(), source.data, source.nbytes);
  } else {
    for (size_t i = 0; i < count; ++i) {
      const float value = source.dtype == "BF16"
          ? vt::BF16ToF32(vt::LoadUnaligned<uint16_t>(source.data + 2 * i))
          : vt::LoadUnaligned<float>(source.data + 4 * i);
      const uint16_t half = vt::F32ToF16(value);
      std::memcpy(result.bytes.data() + 2 * i, &half, sizeof(half));
    }
  }
  MaybeReleaseSourcePages(source.data, source.nbytes);
  return result;
}

// Physical merged dense parameter, in caller-specified row order. Each source
// is converted according to the explicit FP16 policy before concatenation.
inline OwnedTensor LoadMergedF16RawNK(const TensorResolver& get,
                                     const std::vector<std::string>& names) {
  VT_CHECK(!names.empty(), "dense loader: merged FP16 projection needs shards");
  int64_t rows = 0, cols = -1;
  for (const auto& name : names) {
    const auto& source = get(name);
    VT_CHECK(source.shape.size() == 2 && source.shape[0] > 0 && source.shape[1] > 0,
             "dense loader: expected positive [N,K] FP16 source for " + name);
    if (cols == -1) cols = source.shape[1];
    VT_CHECK(source.shape[1] == cols && rows <= INT64_MAX - source.shape[0],
             "dense loader: merged FP16 geometry mismatch for " + name);
    rows += source.shape[0];
  }
  VT_CHECK(rows <= INT64_MAX / cols && static_cast<uint64_t>(rows * cols) <= SIZE_MAX / 2,
           "dense loader: merged FP16 size overflows");
  OwnedTensor merged = MakeOwned(vt::DType::kF16, {rows, cols});
  merged.nk = true;
  size_t offset = 0;
  for (const auto& name : names) {
    const OwnedTensor shard = LoadF16Direct(get, name);
    std::memcpy(merged.bytes.data() + offset, shard.bytes.data(), shard.bytes.size());
    offset += shard.bytes.size();
  }
  return merged;
}

// F16 -> BF16, for the unquantized remainder of an EXL3 checkpoint.
//
// WHY THIS IS SCOPED TO EXL3 RATHER THAN ADDED TO `MaterializeBf16Source`.
// That helper accepts BF16 and F8_E4M3 and REFUSES anything else by name.
// Teaching it F16 would silently widen acceptance for every dense model: a
// checkpoint that refuses today would start loading, through a conversion that
// DROPS THREE MANTISSA BITS (F16 keeps 10, BF16 keeps 7). That is a change to
// other rows' models made as a side effect of this one, so it is not made.
//
// Inside an EXL3 load the conversion is the right polarity rather than a
// compromise. exllamav3 runs its linear in fp16 and stores the unquantized
// remainder to match, but the config's own `torch_dtype` is `bfloat16` -- so
// bf16 is the MODEL dtype every layer inherits (AGENTS.md "Inherit vLLM
// defaults"), and materializing the remainder at bf16 is what loading this
// checkpoint at its declared dtype means.
inline OwnedTensor LoadF16AsBf16Direct(const TensorResolver& get, const std::string& name,
                                       const std::vector<int64_t>& want_shape = {}) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "F16",
           "dense loader: expected F16 for " + name + " (the unquantized remainder of an "
           "EXL3 checkpoint), got " + t.dtype);
  std::vector<int64_t> shape(t.shape.begin(), t.shape.end());
  if (!want_shape.empty()) {
    VT_CHECK(shape == want_shape,
             "dense loader: " + name + " is " + ShapeString(shape) + ", expected " +
                 ShapeString(want_shape));
  }
  int64_t numel = 1;
  for (int64_t d : shape) numel *= d;
  VT_CHECK(static_cast<size_t>(numel) * 2 == t.nbytes,
           "dense loader: " + name + " byte size does not match its F16 shape");
  OwnedTensor r = MakeOwned(vt::DType::kBF16, shape);
  auto* dst = reinterpret_cast<uint16_t*>(r.bytes.data());
  // `vt::LoadUnaligned`, NOT a `reinterpret_cast<const uint16_t*>`: a
  // safetensors tensor's offset is the running byte total of everything before
  // it and carries NO alignment guarantee, so a widened load at an odd offset
  // is undefined behaviour. The BF16 and F32 arms above already read their
  // payloads this way for exactly that reason; this one did not, and UBSan
  // caught it on the first fixture whose payload is deliberately misaligned:
  // "load of misaligned address ... requires 2 byte alignment".
  for (int64_t i = 0; i < numel; ++i)
    dst[i] = vt::F32ToBF16(vt::F16ToF32(vt::LoadUnaligned<uint16_t>(t.data + i * 2)));
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return r;
}

// ── EXL3 (exllamav3 trellis) — QUANT-EXL3 W1b (#2181) ────────────────────────
//
// The predicate is upstream's own: `Linear.is_exl3_storage` requires
// `{key}.trellis` TOGETHER WITH `{key}.suh|.su` and `{key}.svh|.sv`
// (`exllamav3/modules/linear.py:385-389`). Requiring all three rather than the
// trellis alone is what makes a half-written or differently-quantized
// projection fall through to the dense loader instead of being read as EXL3.
inline bool IsExl3Projection(const std::function<bool(const std::string&)>& has,
                             const std::string& proj) {
  return has(proj + ".trellis") && has(proj + ".suh") && has(proj + ".svh");
}

// One EXL3 Linear -> `Exl3Weight`, in the NATIVE exllamav3 layout: no `.rank{r}`
// segment and no coalescing. That segment belongs to SparkInfer's
// `rank-sliced-deepseek-v4-v1` variant, which `MODEL-DSV4-EXL3` reads; the
// stock `turboderp/*-exl3` artifacts are a single unsliced tensor per
// projection, which is why this reader is the simpler of the two.
inline Exl3Weight LoadExl3(const TensorResolver& get,
                           const std::function<bool(const std::string&)>& has,
                           const std::string& proj) {
  const StTensor& tr = get(proj + ".trellis");
  VT_CHECK(tr.dtype == "I16",
           "dense loader: expected I16 trellis for " + proj + " (exl3.py:47), got " + tr.dtype);
  VT_CHECK(tr.shape.size() == 3,
           "dense loader: expected 3-D trellis [k/16, n/16, 16*bits] for " + proj +
               ", got " + ShapeString(std::vector<int64_t>(tr.shape.begin(), tr.shape.end())));
  const int64_t k = tr.shape[0] * 16;
  const int64_t n = tr.shape[1] * 16;
  const int64_t words = tr.shape[2];
  VT_CHECK(words > 0 && words % 16 == 0,
           "dense loader: trellis last dim must be 16*bits words for " + proj + ", got " +
               std::to_string(words));
  const int64_t bits = words / 16;
  VT_CHECK(bits >= 1 && bits <= 8,
           "dense loader: exl3 bits must be in [1, 8] for " + proj + "; the trellis last dim " +
               std::to_string(words) + " implies " + std::to_string(bits));

  const StTensor& suh = get(proj + ".suh");
  const StTensor& svh = get(proj + ".svh");
  VT_CHECK(suh.dtype == "F16" && svh.dtype == "F16",
           "dense loader: expected F16 suh/svh for " + proj + " (exl3.py:48-49)");
  // suh is the INPUT side and svh the OUTPUT side. Checking both against the
  // trellis geometry is what catches a transposed projection, which otherwise
  // loads, runs, and returns a confidently wrong answer on a square linear.
  VT_CHECK(suh.shape.size() == 1 && suh.shape[0] == k,
           "dense loader: " + proj + ".suh must be [k=" + std::to_string(k) + "], got [" +
               std::to_string(suh.shape.empty() ? -1 : suh.shape[0]) + "]");
  VT_CHECK(svh.shape.size() == 1 && svh.shape[0] == n,
           "dense loader: " + proj + ".svh must be [n=" + std::to_string(n) + "], got [" +
               std::to_string(svh.shape.empty() ? -1 : svh.shape[0]) + "]");

  Exl3Weight r;
  r.name = proj;
  // THE CODEBOOK IS SELECTED BY TENSOR PRESENCE, and the polarity is the
  // opposite of the obvious guess. `LinearEXL3` sets
  // `self.mcg = (self.mcg_tensor is not None)` and likewise for `mul1`
  // (`exl3.py:74-77`), then passes those BOOLEANS to `ext.reconstruct`
  // (`:197,223`). So a checkpoint that ships NO marker is NOT MCG: it is cb 0,
  // the original QTIP 3INST. The SparkInfer DeepSeek-V4 artifact ships an `mcg`
  // marker and is cb 1; every stock `turboderp/*-exl3` artifact ships neither
  // and is cb 0.
  //
  // Getting this backwards is not a loud failure. The wrong multiplier produces
  // a codebook with the SAME DISTRIBUTION and no relation to the right one, so
  // the weight decodes to the correct RMS and uncorrelated values, every shape
  // check passes, and the model emits fluent nonsense. MEASURED on
  // `turboderp/Llama-3.2-1B-Instruct-exl3` layer 0 `q_proj` against the
  // unquantized tensor: cb 1 gives RMS 0.038454 and cosine -0.0006, cb 0 gives
  // RMS 0.035941 and cosine +0.9896, reference RMS 0.036056.
  const bool has_mcg = has(proj + ".mcg");
  const bool has_mul1 = has(proj + ".mul1");
  VT_CHECK(!(has_mcg && has_mul1),
           "dense loader: " + proj + " carries BOTH an mcg and a mul1 marker, which "
           "selects two codebooks at once (QUANT-EXL3, #2181)");
  if (has_mcg) {
    const StTensor& mcg = get(proj + ".mcg");
    VT_CHECK(mcg.dtype == "I32",
             "dense loader: expected I32 mcg marker for " + proj + ", got " + mcg.dtype);
  }
  if (has_mul1) {
    // `quantize.py:1421-1424` writes it as
    // `torch.tensor(0x83DCD12D, uint32).view(torch.int)` -- one I32 holding the
    // codebook's own multiplier -- exactly as it writes `mcg`.
    const StTensor& mul1 = get(proj + ".mul1");
    VT_CHECK(mul1.dtype == "I32",
             "dense loader: expected I32 mul1 marker for " + proj + ", got " + mul1.dtype);
    // The native codebook-2 decoder uses this fixed multiplier. Accept the
    // producer's scalar and the historical one-element spelling, but never
    // silently decode another codebook from a tensor wearing the same name.
    VT_CHECK((mul1.shape.empty() || mul1.shape == std::vector<int64_t>{1}) &&
                 mul1.nbytes == sizeof(uint32_t) && mul1.data != nullptr,
             "dense loader: expected one I32 mul1 multiplier for " + proj);
    VT_CHECK(vt::LoadUnaligned<uint32_t>(mul1.data) == 0x83DCD12DU,
             "dense loader: unsupported mul1 multiplier for " + proj);
  }
  VT_CHECK(!has(proj + ".had"),
           "dense loader: " + proj +
               " carries a `had` tensor, which is exllamav3's EXPLICIT Hadamard "
               "storage rather than the suh/svh sign-vector form this reader "
               "implements (QUANT-EXL3, #2181)");
  // `su`/`sv` are the PACKED-BITFIELD form of the sign vectors (`unpack_bf`,
  // `exl3.py:142-158`), which this reader does not unpack.
  VT_CHECK(!has(proj + ".su") && !has(proj + ".sv"),
           "dense loader: " + proj +
               " carries packed `su`/`sv` sign vectors, which this reader does not "
               "unpack (QUANT-EXL3, #2181)");
  // The three-way form of the SAME presence rule. `LinearEXL3` sets `self.mcg`
  // and `self.mul1` from two independent tensor lookups and passes both booleans
  // to `ext.reconstruct` (`exl3.py:74-77,197,223`); with both false the codebook
  // is 0. `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` marks every quantized linear
  // `mul1` and is cb 2 (QUANT-EXL3-MUL1, #2495).
  //
  // A `mul1` marker was REFUSED here until cb 2 was ported, and the refusal was
  // right for as long as it stood: the wrong multiplier yields a correctly
  // distributed and completely wrong weight, which no shape check can see.
  // Lifting it is safe now only because `Exl3DecodeCodeword(cw, 2)` implements
  // upstream's byte-sum decode and is gated against hand-computed values.
  r.codebook = has_mul1 ? 2 : (has_mcg ? 1 : 0);

  // ENG-LOAD-DIRECT-UPLOAD (#150): all three are taken VERBATIM into a
  // same-size destination, so all three qualify for the borrow path. The
  // trellis is BORROWED AS BYTES at 32*bits: identical bytes, and the dtype
  // differs from disk only because `vt::DType` has no 16-bit integer and
  // `vt::Exl3Gemm` reads the operand as `kI8` anyway.
  if (!BorrowStTensorBytes(r.trellis, tr, vt::DType::kI8, {k / 16, n / 16, 32 * bits})) {
    r.trellis = MakeOwned(vt::DType::kI8, {k / 16, n / 16, 32 * bits});
    VT_CHECK(tr.nbytes == r.trellis.bytes.size(),
             "dense loader: trellis byte-size mismatch for " + proj);
    std::memcpy(r.trellis.bytes.data(), tr.data, tr.nbytes);
    MaybeReleaseSourcePages(tr.data, tr.nbytes);
  }
  if (!BorrowStTensorBytes(r.suh, suh, vt::DType::kF16, {k})) {
    r.suh = MakeOwned(vt::DType::kF16, {k});
    std::memcpy(r.suh.bytes.data(), suh.data, suh.nbytes);
    MaybeReleaseSourcePages(suh.data, suh.nbytes);
  }
  if (!BorrowStTensorBytes(r.svh, svh, vt::DType::kF16, {n})) {
    r.svh = MakeOwned(vt::DType::kF16, {n});
    std::memcpy(r.svh.bytes.data(), svh.data, svh.nbytes);
    MaybeReleaseSourcePages(svh.data, svh.nbytes);
  }
  return r;
}

inline bool IsCtNvfp4Projection(
    const std::function<bool(const std::string&)>& has, const std::string& proj) {
  return has(proj + ".weight_packed");
}

// One compressed-tensors NVFP4 W4A16 Linear -> raw fp4-resident Nvfp4Weight in
// the on-disk [N=out, K=in] orientation the Marlin/naive W4A16 GEMMs read
// directly (no bf16 materialization). Rejects a W4A4 checkpoint outright: an
// `input_global_scale` here means the scheme is NOT weight-only and belongs to
// the fp4-activation path, which this dense helper does not implement.
inline Nvfp4Weight LoadCtNvfp4W4A16(
    const TensorResolver& get, const std::function<bool(const std::string&)>& has,
    const std::string& proj) {
  const StTensor& packed = get(proj + ".weight_packed");
  VT_CHECK(packed.dtype == "U8",
           "dense loader: expected U8 weight_packed for " + proj);
  VT_CHECK(packed.shape.size() == 2,
           "dense loader: expected 2-D weight_packed for " + proj);
  const int64_t out_dim = packed.shape[0];
  const int64_t in_dim = packed.shape[1] * 2;
  VT_CHECK(in_dim % 16 == 0,
           "dense loader: NVFP4 in_dim must be a multiple of 16 for " + proj);
  const StTensor& ws = get(proj + ".weight_scale");
  VT_CHECK(ws.dtype == "F8_E4M3",
           "dense loader: expected F8_E4M3 weight_scale for " + proj);
  VT_CHECK(ws.shape.size() == 2 && ws.shape[0] == out_dim &&
               ws.shape[1] == in_dim / 16,
           "dense loader: weight_scale shape must be [N, K/16] for " + proj);
  VT_CHECK(!has(proj + ".input_global_scale"),
           "dense loader: " + proj +
               " carries input_global_scale (W4A4); the dense NVFP4 loader "
               "implements the WEIGHT-ONLY W4A16 scheme only");
  const float wgs_disk =
      ReadF32Scalar(get, proj + ".weight_global_scale");
  VT_CHECK(wgs_disk != 0.0F,
           "dense loader: zero weight_global_scale (divisor) for " + proj);

  Nvfp4Weight r;
  r.n = out_dim;
  r.k = in_dim;
  r.weight_global_scale_inv = wgs_disk;  // exact divisor, for merged linears
  r.scale2 = 1.0F / wgs_disk;            // CT stores a divisor -> reciprocate
  r.alpha = 0.0F;                        // W4A16: no activation quant
  // ENG-LOAD-DIRECT-UPLOAD (issue #150): weight_packed and weight_scale are each
  // taken VERBATIM into their own same-size destination, so both qualify for the
  // borrow path; the memcpys are the unchanged fallback.
  if (!BorrowStTensorBytes(r.packed, packed, vt::DType::kI8,
                           {out_dim, in_dim / 2})) {
    r.packed = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 2});
    VT_CHECK(packed.nbytes == r.packed.bytes.size(),
             "dense loader: packed byte-size mismatch for " + proj);
    std::memcpy(r.packed.bytes.data(), packed.data, packed.nbytes);
    MaybeReleaseSourcePages(packed.data, packed.nbytes);
  }
  if (!BorrowStTensorBytes(r.scale, ws, vt::DType::kI8,
                           {out_dim, in_dim / 16})) {
    r.scale = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 16});
    VT_CHECK(ws.nbytes == r.scale.bytes.size(),
             "dense loader: scale byte-size mismatch for " + proj);
    std::memcpy(r.scale.bytes.data(), ws.data, ws.nbytes);
    MaybeReleaseSourcePages(ws.data, ws.nbytes);
  }
  return r;
}

// Load and concatenate compressed-tensors NVFP4 W4A16 shards `[N_i, K]` along
// their output rows — the NVFP4 analog of LoadMergedBf16RawNK, and vLLM's
// physical ownership rule for QKVParallelLinear / MergedColumnParallelLinear
// (ONE merged parameter, qwen3.py:271-274).
//
// `weight_packed` [N_i,K/2] and `weight_scale` [N_i,K/16] are both row-major
// over N, so both concatenate by plain row-stack (packing/grouping runs along K
// and is therefore untouched) — vLLM likewise fuses them by concat along
// output_dim=0 (compressed_tensors_w4a4_nvfp4.py:53-62,73-84).
//
// The GLOBAL scale is the one lossy part, and we mirror vLLM exactly: it keeps a
// PerTensorScaleParameter with one entry per shard and then collapses them with
// `.max()` before reciprocating (compressed_tensors_w4a4_nvfp4.py:111-114,
// warning at :101-108 when the shards disagree). We take the max over the on-disk
// DIVISORS and reciprocate once, which is the identical arithmetic without an
// intermediate rounding. (On RedHatAI/Qwen3-32B-NVFP4A16 the shards' divisors are
// bit-identical within every fused group, so the collapse is exactly lossless
// there; the max is kept for faithfulness on checkpoints where they differ.)
inline Nvfp4Weight LoadMergedCtNvfp4W4A16(
    const TensorResolver& get, const std::function<bool(const std::string&)>& has,
    const std::vector<std::string>& projs) {
  VT_CHECK(!projs.empty(),
           "dense loader: merged NVFP4 projection requires at least one shard");
  std::vector<Nvfp4Weight> shards;
  shards.reserve(projs.size());
  int64_t in_dim = -1;
  int64_t out_dim = 0;
  for (const std::string& proj : projs) {
    Nvfp4Weight s = LoadCtNvfp4W4A16(get, has, proj);
    if (in_dim < 0) in_dim = s.k;
    VT_CHECK(s.k == in_dim,
             "dense loader: merged NVFP4 shards must share input width");
    VT_CHECK(out_dim <= std::numeric_limits<int64_t>::max() - s.n,
             "dense loader: merged NVFP4 output width overflow");
    out_dim += s.n;
    shards.push_back(std::move(s));
  }
  if (shards.size() == 1) return std::move(shards[0]);

  Nvfp4Weight merged;
  merged.n = out_dim;
  merged.k = in_dim;
  merged.alpha = 0.0F;
  // vLLM: weight_global_scale = 1.0 / max(per-shard on-disk global scales).
  float max_divisor = 0.0F;
  for (const Nvfp4Weight& s : shards)
    if (s.weight_global_scale_inv > max_divisor)
      max_divisor = s.weight_global_scale_inv;
  VT_CHECK(max_divisor != 0.0F,
           "dense loader: merged NVFP4 max weight_global_scale is zero");
  merged.weight_global_scale_inv = max_divisor;
  merged.scale2 = 1.0F / max_divisor;

  merged.packed = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 2});
  merged.scale = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 16});
  size_t p_off = 0;
  size_t s_off = 0;
  for (const Nvfp4Weight& s : shards) {
    std::memcpy(merged.packed.bytes.data() + p_off, s.packed.bytes.data(),
                s.packed.bytes.size());
    p_off += s.packed.bytes.size();
    std::memcpy(merged.scale.bytes.data() + s_off, s.scale.bytes.data(),
                s.scale.bytes.size());
    s_off += s.scale.bytes.size();
    // A concatenation is NOT a verbatim view, so the merged buffer is owned and
    // any per-shard borrow is now consumed-and-dead.
    ReleaseBorrowedShardSource(s.packed);
    ReleaseBorrowedShardSource(s.scale);
  }
  VT_CHECK(p_off == merged.packed.bytes.size() &&
               s_off == merged.scale.bytes.size(),
           "dense loader: merged NVFP4 byte accounting mismatch");
  return merged;
}

// --- compressed-tensors MXFP4 **W4A16** (`mxfp4-pack-quantized`) -------------
// ADDED (append-only; no existing helper touched) by the QUANT-CT-MXFP4 dense-Qwen
// bring-up. MXFP4 analog of LoadCtNvfp4W4A16: same [N=out, K=in] raw orientation
// and merged-shard ownership rule, but the block scale is E8M0/UE8M0 at
// group_size 32 with NO global scale.
//
// ON-DISK LAYOUT (verified on Yi30/Qwen3-8B-MXFP4, the format compressed-tensors
// emits for `mxfp4-pack-quantized`, group_size 32, num_bits 4, type float):
//   <proj>.weight_packed  U8  [N, K/2]    two E2M1 nibbles per byte
//   <proj>.weight_scale   U8  [N, K/32]   one E8M0 (biased exponent) per 32 elems
// and NO `<proj>.weight_global_scale` / `<proj>.input_global_scale` (MXFP4 has no
// global, and the FlashInfer/Marlin W4A16 path folds the E8M0 scale directly). On
// GB10 the runnable oracle path is Marlin W4A16 (FlashInfer cute-dsl mxf4 rejects
// sm_121), which is exactly what this routes to. The result is a raw fp4-resident
// Nvfp4Weight with is_mxfp4=true, group_size=32, scale2 unused.

// True when `proj` is stored as a compressed-tensors MXFP4 linear: `.weight_packed`
// present AND `.weight_scale` is U8 (E8M0). The U8 scale is the discriminator vs
// NVFP4 (whose weight_scale is F8_E4M3), so this never matches an NVFP4 checkpoint.
inline bool IsCtMxfp4Projection(
    const TensorResolver& get,
    const std::function<bool(const std::string&)>& has, const std::string& proj) {
  if (!has(proj + ".weight_packed") || !has(proj + ".weight_scale")) return false;
  return get(proj + ".weight_scale").dtype == "U8";
}

// One compressed-tensors MXFP4 W4A16 Linear -> raw fp4-resident Nvfp4Weight in the
// on-disk [N=out, K=in] orientation the Marlin W4A16 GEMM reads directly.
// `has` is unused (MXFP4 carries no optional global/input-scale tensors to probe)
// but kept for signature parity with LoadCtNvfp4W4A16 so the merged loader and the
// per-projection call sites are uniform.
inline Nvfp4Weight LoadCtMxfp4W4A16(
    const TensorResolver& get,
    [[maybe_unused]] const std::function<bool(const std::string&)>& has,
    const std::string& proj) {
  const StTensor& packed = get(proj + ".weight_packed");
  VT_CHECK(packed.dtype == "U8",
           "dense loader: expected U8 weight_packed for " + proj);
  VT_CHECK(packed.shape.size() == 2,
           "dense loader: expected 2-D weight_packed for " + proj);
  const int64_t out_dim = packed.shape[0];
  const int64_t in_dim = packed.shape[1] * 2;
  VT_CHECK(in_dim % 32 == 0,
           "dense loader: MXFP4 in_dim must be a multiple of 32 for " + proj);
  const StTensor& ws = get(proj + ".weight_scale");
  VT_CHECK(ws.dtype == "U8",
           "dense loader: expected U8 (E8M0) weight_scale for " + proj);
  VT_CHECK(ws.shape.size() == 2 && ws.shape[0] == out_dim &&
               ws.shape[1] == in_dim / 32,
           "dense loader: weight_scale shape must be [N, K/32] for " + proj);

  Nvfp4Weight r;
  r.n = out_dim;
  r.k = in_dim;
  r.group_size = 32;
  r.is_mxfp4 = true;
  r.scale2 = 0.0F;  // MXFP4 has no global scale (unused)
  r.alpha = 0.0F;   // W4A16: no activation quant
  // ENG-LOAD-DIRECT-UPLOAD (issue #150): both payloads are verbatim; see
  // LoadCtNvfp4W4A16 above.
  if (!BorrowStTensorBytes(r.packed, packed, vt::DType::kI8,
                           {out_dim, in_dim / 2})) {
    r.packed = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 2});
    VT_CHECK(packed.nbytes == r.packed.bytes.size(),
             "dense loader: packed byte-size mismatch for " + proj);
    std::memcpy(r.packed.bytes.data(), packed.data, packed.nbytes);
    MaybeReleaseSourcePages(packed.data, packed.nbytes);
  }
  if (!BorrowStTensorBytes(r.scale, ws, vt::DType::kI8,
                           {out_dim, in_dim / 32})) {
    r.scale = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 32});
    VT_CHECK(ws.nbytes == r.scale.bytes.size(),
             "dense loader: scale byte-size mismatch for " + proj);
    std::memcpy(r.scale.bytes.data(), ws.data, ws.nbytes);
    MaybeReleaseSourcePages(ws.data, ws.nbytes);
  }
  return r;
}

// Load and concatenate compressed-tensors MXFP4 W4A16 shards `[N_i, K]` along
// output rows (the MXFP4 analog of LoadMergedCtNvfp4W4A16). Both weight_packed
// [N_i,K/2] and weight_scale [N_i,K/32] are row-major over N, so both concat by
// plain row-stack (grouping runs along K, untouched). No global scale to collapse.
inline Nvfp4Weight LoadMergedCtMxfp4W4A16(
    const TensorResolver& get, const std::function<bool(const std::string&)>& has,
    const std::vector<std::string>& projs) {
  VT_CHECK(!projs.empty(),
           "dense loader: merged MXFP4 projection requires at least one shard");
  std::vector<Nvfp4Weight> shards;
  shards.reserve(projs.size());
  int64_t in_dim = -1;
  int64_t out_dim = 0;
  for (const std::string& proj : projs) {
    Nvfp4Weight s = LoadCtMxfp4W4A16(get, has, proj);
    if (in_dim < 0) in_dim = s.k;
    VT_CHECK(s.k == in_dim,
             "dense loader: merged MXFP4 shards must share input width");
    VT_CHECK(out_dim <= std::numeric_limits<int64_t>::max() - s.n,
             "dense loader: merged MXFP4 output width overflow");
    out_dim += s.n;
    shards.push_back(std::move(s));
  }
  if (shards.size() == 1) return std::move(shards[0]);

  Nvfp4Weight merged;
  merged.n = out_dim;
  merged.k = in_dim;
  merged.group_size = 32;
  merged.is_mxfp4 = true;
  merged.scale2 = 0.0F;
  merged.alpha = 0.0F;
  merged.packed = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 2});
  merged.scale = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 32});
  size_t p_off = 0;
  size_t s_off = 0;
  for (const Nvfp4Weight& s : shards) {
    std::memcpy(merged.packed.bytes.data() + p_off, s.packed.bytes.data(),
                s.packed.bytes.size());
    p_off += s.packed.bytes.size();
    std::memcpy(merged.scale.bytes.data() + s_off, s.scale.bytes.data(),
                s.scale.bytes.size());
    s_off += s.scale.bytes.size();
    ReleaseBorrowedShardSource(s.packed);
    ReleaseBorrowedShardSource(s.scale);
  }
  VT_CHECK(p_off == merged.packed.bytes.size() &&
               s_off == merged.scale.bytes.size(),
           "dense loader: merged MXFP4 byte accounting mismatch");
  return merged;
}

}  // namespace dense_loaders
}  // namespace vllm
