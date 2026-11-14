// vllm.cpp ORIGINAL; see qwen3_5_gguf_weights.h. GGUF tensor names + metadata
// keys mirror llama.cpp @ 237ad9b (qwen35moe/qwen3next: src/llama-arch.cpp,
// src/models/qwen35moe.cpp); the convert-time value transforms this file
// INVERTS to recover raw-HF weights are in conversion/qwen.py.
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#if defined(__unix__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "vllm/config/weight_residency.h"
#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vt/dtype.h"
#include "vt/quant.h"
#if defined(VLLM_CPP_TENSTORRENT)
#include "vt/tenstorrent/tenstorrent_device.h"
#endif

namespace vllm {
// W4d W0 (#3042) attribution: DRAM bank snapshots in the host load path.
#if defined(VLLM_CPP_TENSTORRENT)
static void TtAllocTrace(const char* label) {
  vt::tenstorrent::AllocTraceSnapshot(vt::tenstorrent::SharedMeshDevice(), label);
}
static void TtAllocTraceLayer(const char* what, int64_t il) {
  char lbl[64];
  std::snprintf(lbl, sizeof(lbl), "load/%s/%lld", what, (long long)il);
  TtAllocTrace(lbl);
}
#else
static inline void TtAllocTrace(const char*) {}
static inline void TtAllocTraceLayer(const char*, int64_t) {}
#endif


// Load-time PREFAULT of a mmap-borrowed weight span (VT_GGUF_PREFAULT or
// `--offload-config`'s `vllm_cpp.mmap.prefault`, default ON with mmap residency). A weight left BORROWED in the read-only mapping is not
// resident until first touch; without a prefault those page faults land in the
// TIMED prefill (the L6 keep-f16 TTFT regression). Faulting them in at load —
// off the timed path — mirrors llama.cpp, which prefetches its mmap at load with
// posix_madvise(POSIX_MADV_WILLNEED) (`src/llama-mmap.cpp:451`, the mmap prefetch
// under `use_mmap`). We use MADV_WILLNEED as the async hint plus a synchronous
// one-byte-per-page read so the pages are guaranteed resident before prefill,
// not merely queued. Read-only, so it changes no bytes. No-op on the copy arm.
inline void PrefaultBorrowedSpan(const uint8_t* src, size_t bytes) {
#if defined(__unix__)
  // ENG-RESIDENCY-CONFIG (#1110): the decision now comes from
  // `ResolveGgufPrefault()`, which holds `VT_GGUF_PREFAULT` >
  // `--offload-config`'s `vllm_cpp.mmap.prefault` > ON, and which is the SOLE
  // reader of that variable.
  //
  // THE `static` IS GONE ON PURPOSE. Caching the answer cost one `getenv` per
  // prefaulted span, which is nothing beside the megabytes of pages this function
  // then reads, and it bought two defects: a config installed at load could be
  // ignored by whichever caller happened to ask first, and the existing A/B case
  // (tests/vllm/test_gguf_keep_quant.cpp, "L7 load-time prefault is
  // byte-transparent") was silently vacuous, because its second `setenv` could not
  // change an already-latched value and both of its arms ran identically.
  if (!ResolveGgufPrefault() || bytes == 0) return;
  // LOAD-IO: timed, because this loop IS the load's file I/O. Every weight page
  // of a keep-quant GGUF arrives here and nowhere else, so its elapsed time
  // separates "the artifact was still arriving" from "we were computing", which
  // a single weights-phase number cannot.
  const auto t_pf = std::chrono::steady_clock::now();
  (void)::madvise(const_cast<uint8_t*>(src), bytes, MADV_WILLNEED);
  const long ps_l = ::sysconf(_SC_PAGESIZE);
  const size_t ps = static_cast<size_t>(ps_l > 0 ? ps_l : 4096);
  volatile uint8_t sink = 0;
  for (size_t off = 0; off < bytes; off += ps) sink = sink ^ src[off];
  (void)sink;
  NoteGgufPrefaultedBytes(
      static_cast<uint64_t>(bytes),
      static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - t_pf)
              .count()));
  // Count the span AFTER faulting it, so the counter means "this actually
  // happened". It is the only way a test can tell a prefault from a skip: the
  // prefault reads pages and changes no byte, which is why the pre-existing
  // byte-transparency case could not see the difference and a site mutated to
  // never consult its resolver left the whole GGUF suite green.
  NoteGgufPrefaultedSpan();
#else
  (void)src;
  (void)bytes;
#endif
}

OwnedTensor OwnGgufQuantBlocks(const GgufTensorInfo& tensor, int64_t n,
                               int64_t k, int64_t row_offset,
                               const GgufFile* mmap_src, bool repack,
                               bool cuda_align, bool prefault, GgufTensorRole role) {
  vt::DType dt = vt::DType::kF32;
  const bool gather = role == GgufTensorRole::kEmbeddingTable;
  VT_CHECK(gather ? KeepQuantGatherDType(tensor.ggml_type, &dt)
                  : KeepQuantDType(tensor.ggml_type, &dt),
           "qwen3_5 gguf: keep-quant on a non-keep-quant encoding for " +
               tensor.name);
  VT_CHECK(!gather || (!repack && !cuda_align),
             "qwen3_5 gguf: embedding blocks cannot use a matrix repack");
  VT_CHECK(n > 0 && k > 0 && row_offset >= 0,
           "qwen3_5 gguf: bad keep-quant slice for " + tensor.name);
  // Throws when k is not a whole number of blocks (ggml_row_size contract).
  const size_t row_bytes = vt::RowSizeBytes(dt, k);
  const size_t begin = static_cast<size_t>(row_offset) * row_bytes;
  const size_t bytes = static_cast<size_t>(n) * row_bytes;
  VT_CHECK(begin + bytes <= tensor.nbytes,
           "qwen3_5 gguf: keep-quant slice exceeds the tensor span for " +
               tensor.name);

  OwnedTensor o;
  o.dtype = dt;
  o.rank = 2;
  o.shape[0] = n;  // N = out features (ggml src0 rows)
  o.shape[1] = k;  // K = in features
  // GGUF disk order [out, in] IS the MatmulBT [N, K] orientation: no transpose.
  o.nk = true;
  const uint8_t* src = tensor.data + begin;

  // Brick 4 (DeepSeek-V4 last-mile): repack the Q8_0 slice into the CUDA
  // coalesced-load layout. Like the CIQ G7 branch below it always COPIES (the
  // transform rewrites the buffer; an mmap borrow is read-only), marks
  // o.q8_0_aligned, and drops the now-dead source file pages so the Q8_0 mass is
  // not counted twice (owned repack + resident file page). Same total bytes; the
  // integer dot is unchanged (byte permutation only). Precedes the CPU i8mm
  // branch — the two are mutually exclusive and cuda_align wins where both apply.
  if (cuda_align && vt::cpu::RepackQ8_0CudaEligible(dt, n, k)) {
    o.bytes.assign(src, src + bytes);
    vt::cpu::RepackQ8_0Cuda(o.bytes.data(), n, k);
    o.q8_0_aligned = true;
    if (mmap_src != nullptr) mmap_src->DropSpanResidency(src, bytes);
    return o;
  }

  // CIQ G7: repack the slice into the i8mm interleave. It rewrites the buffer,
  // so it always COPIES (an mmap borrow is read-only) and marks o.repacked; the
  // total byte count is unchanged (block_q8_0x4 == 4 * block_q8_0). A slice that
  // is not repack-eligible (wrong dtype/shape, or no i8mm) falls through to the
  // normal copy/borrow below and stays a plain block weight.
  if (repack && vt::cpu::QuantRepackEligible(dt, n, k)) {
    o.bytes.assign(src, src + bytes);
    vt::cpu::QuantRepackWeight(dt, o.bytes.data(), n, k);
    o.repacked = true;
    // The repacked OWNED (anonymous) buffer is now the authority for this weight;
    // the source mmap blocks were read once by the copy above and are dead. When
    // the weight was borrowed from a mapping (mmap residency), those file pages
    // otherwise stay RESIDENT next to the repacked copy — the q8_0 mass counted
    // TWICE (once file-backed source, once anonymous repack). Drop the source's
    // resident file pages (L5 read-once release, port of llama.cpp unmap_fragment
    // `src/llama-mmap.cpp:490`; gated by VT_GGUF_RELEASE_PAGES). Byte-identical:
    // a later read simply re-faults the same bytes. No-op on the copy arm
    // (mmap_src == nullptr) and on non-repack CPUs (this branch never runs).
    if (mmap_src != nullptr) mmap_src->DropSpanResidency(src, bytes);
    return o;
  }

  if (mmap_src == nullptr) {
    o.bytes.assign(src, src + bytes);  // L2: copy into an owned buffer
    return o;
  }
  // L5 mmap residency. Re-validate the span against the MAPPING (not just the
  // tensor record) so a mismatched file can never hand out a view outside it,
  // and require the block structs' natural alignment — every keep-quant block
  // begins with a 2-byte scale field, and GGUF's own tensor-data alignment
  // (`general.alignment`, >= 8 in practice) already guarantees it, so a failure
  // here means a malformed file rather than a limitation of this path.
  VT_CHECK(mmap_src->OwnsSpan(src, bytes),
           "qwen3_5 gguf: keep-quant mmap span is outside the file mapping for " +
               tensor.name);
  VT_CHECK(reinterpret_cast<uintptr_t>(src) % alignof(uint16_t) == 0,
           "qwen3_5 gguf: keep-quant mmap span is not block-aligned for " +
               tensor.name);
  o.bytes = OwnedBytes::Borrow(src, bytes, mmap_src->Mapping());
  // Remember WHERE these bytes live, so a later consumer can pread them instead
  // of faulting them through the mapping. Two fields and one lookup at load;
  // saves a page trap per 4 KiB at decode.
  {
    const GgufFile::SpanSource ss = mmap_src->SourceOfSpan(src, bytes);
    o.mmap_fd = ss.fd;
    o.mmap_file_offset = ss.offset;
  }
  // Not for a span the expert-streaming slot lane will serve: see the `prefault`
  // note on the declaration. The lane preads each slice into its own slot, so a
  // prefaulted tower is bytes read off disk into pages nothing reads.
  if (prefault) PrefaultBorrowedSpan(src, bytes);
  return o;
}

OwnedTensor OwnGgufF16(const GgufTensorInfo& tensor, int64_t n, int64_t k,
                       int64_t row_offset, const GgufFile* mmap_src, bool nk,
                       bool elem_kn_repack, bool prefault,
                       std::optional<vt::DType> weight_value_dtype) {
  VT_CHECK(KeepF16DType(tensor.ggml_type),
           "qwen3_5 gguf: keep-f16 on a non-f16 encoding for " + tensor.name);
  VT_CHECK(n > 0 && k > 0 && row_offset >= 0,
           "qwen3_5 gguf: bad keep-f16 slice for " + tensor.name);
  const size_t row_bytes = vt::RowSizeBytes(vt::DType::kF16, k);  // 2 * k
  const size_t begin = static_cast<size_t>(row_offset) * row_bytes;
  const size_t bytes = static_cast<size_t>(n) * row_bytes;
  VT_CHECK(begin + bytes <= tensor.nbytes,
           "qwen3_5 gguf: keep-f16 slice exceeds the tensor span for " +
               tensor.name);

  OwnedTensor o;
  o.dtype = vt::DType::kF16;
  o.weight_value_dtype = weight_value_dtype;
  o.rank = 2;
  o.shape[0] = n;  // N = out features (ggml src0 rows), or vocab for a gather
  o.shape[1] = k;  // K = in features, or H for a gather
  // nk = true: the file's [N, K] MatmulBT orientation, no transpose. nk = false:
  // a [vocab, H] gather table (embedding), read row-wise by EmbeddingKernel.
  o.nk = nk;
  const uint8_t* src = tensor.data + begin;

  // KERNEL-GEMM-CPU-TILED lever 2: transpose [N,K] -> [K,N] so vt::MatmulBT
  // reaches the transpose-free nk/nkm kernels (1.16x to 1.30x on dgx,
  // byte-identical). Only for the MatmulBT orientation: an `nk == false` tensor
  // is an embedding gather table read row-wise by EmbeddingKernel, which would
  // read transposed bytes as garbage. Rewrites the buffer, so like the quant
  // repack branches it always COPIES and drops the now-dead source pages.
  if (elem_kn_repack && nk && vt::cpu::ElemRepackEligible(o.dtype, n, k)) {
    o.bytes.assign(src, src + bytes);
    vt::cpu::ElemRepackWeight(o.dtype, o.bytes.data(), n, k);
    o.elem_kn_repacked = true;
    if (mmap_src != nullptr) mmap_src->DropSpanResidency(src, bytes);
    return o;
  }

  if (mmap_src == nullptr) {
    o.bytes.assign(src, src + bytes);  // copy into an owned buffer
    return o;
  }
  // Borrow in place out of the read-only mapping (L5 residency, L6 dtype). The
  // f16 bytes need only 2-byte alignment, which GGUF's tensor-data alignment
  // guarantees; re-validate against the MAPPING so a mismatched file can never
  // hand out a view outside it.
  VT_CHECK(mmap_src->OwnsSpan(src, bytes),
           "qwen3_5 gguf: keep-f16 mmap span is outside the file mapping for " +
               tensor.name);
  VT_CHECK(reinterpret_cast<uintptr_t>(src) % alignof(uint16_t) == 0,
           "qwen3_5 gguf: keep-f16 mmap span is not 2-byte aligned for " +
               tensor.name);
  o.bytes = OwnedBytes::Borrow(src, bytes, mmap_src->Mapping());
  // Remember WHERE these bytes live, so a later consumer can pread them instead
  // of faulting them through the mapping. Two fields and one lookup at load;
  // saves a page trap per 4 KiB at decode.
  {
    const GgufFile::SpanSource ss = mmap_src->SourceOfSpan(src, bytes);
    o.mmap_fd = ss.fd;
    o.mmap_file_offset = ss.offset;
  }
  // Not for a span the expert-streaming slot lane will serve: see the `prefault`
  // note on the declaration. The lane preads each slice into its own slot, so a
  // prefaulted tower is bytes read off disk into pages nothing reads.
  if (prefault) PrefaultBorrowedSpan(src, bytes);
  return o;
}

namespace {

// --- small helpers -------------------------------------------------------

// Independent opt-out for the read-once page release (MADV_DONTNEED on the file
// pages of EXPANDED tensors). Default ON with mmap residency. Kept separate so
// the page-drop can be A/B'd against the mmap borrow alone.
bool EnvReleaseExpandedPages() {
  const char* v = std::getenv("VT_GGUF_RELEASE_PAGES");
  if (v == nullptr) return true;
  return !(std::strcmp(v, "") == 0 || std::strcmp(v, "0") == 0 ||
           std::strcmp(v, "false") == 0 || std::strcmp(v, "off") == 0);
}

// Scopes the file's read-once-page release to ONE load, so a GgufFile that
// outlives the loader (or is loaded from twice, as the tests do) never carries
// a stale residency policy.
class GgufPageReleaseScope {
 public:
  GgufPageReleaseScope(const GgufFile& g, bool on)
      : g_(g), prev_(g.releases_expanded_pages()) {
    g_.ReleaseExpandedPages(on);
  }
  ~GgufPageReleaseScope() { g_.ReleaseExpandedPages(prev_); }
  GgufPageReleaseScope(const GgufPageReleaseScope&) = delete;
  GgufPageReleaseScope& operator=(const GgufPageReleaseScope&) = delete;

 private:
  const GgufFile& g_;
  bool prev_;
};

// L5: the file to borrow kept blocks from, or null to copy them (L2's arm,
// which `VT_GGUF_MMAP=0` and `VT_CPU_REF=1` both select).
const GgufFile* MmapSrc(const GgufFile& g, const GgufLoadPolicy& pol) {
  return pol.mmap_residency ? &g : nullptr;
}

// L6: keep a verbatim 2-D / stacked-expert weight slice resident — quant blocks
// or native f16 — in [N, K] nk=true order. Precondition: the policy already
// routed this (tensor, role) to a KEEP residency (`r != kExpandBf16`).
// Centralizes the block-vs-f16 dispatch so every keep call site is one line.
OwnedTensor OwnGgufKeptSlice(const GgufFile& g, const GgufLoadPolicy& pol,
                             const GgufTensorInfo& t, GgufResidency r, int64_t n,
                             int64_t k, int64_t row_offset) {
  if (r == GgufResidency::kKeepQuant) {
    return OwnGgufQuantBlocks(t, n, k, row_offset, MmapSrc(g, pol),
                              pol.quant_repack, /*cuda_align=*/false,
                              pol.prefault);
  }
  VT_CHECK(r == GgufResidency::kKeepF16,
           "qwen3_5 gguf: OwnGgufKeptSlice called for a non-keep residency on " +
               t.name);
  return OwnGgufF16(t, n, k, row_offset, MmapSrc(g, pol), /*nk=*/true,
                    pol.elem_kn_repack, pol.prefault, pol.weight_value_dtype);
}

bool HasTensor(const GgufFile& g, const std::string& name) {
  for (const GgufTensorInfo& t : g.Tensors()) {
    if (t.name == name) return true;
  }
  return false;
}

int64_t ShapeNumel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

OwnedTensor MakeOwned(vt::DType dt, const std::vector<int64_t>& shape) {
  OwnedTensor o;
  o.dtype = dt;
  o.rank = static_cast<int>(shape.size());
  VT_CHECK(o.rank <= vt::kMaxRank, "qwen3_5 gguf: rank exceeds kMaxRank");
  int64_t n = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[i];
    n *= shape[i];
  }
  o.bytes.resize(static_cast<size_t>(n) * vt::SizeOf(dt));
  return o;
}

// src bf16 [rows, cols] -> dst bf16 [cols, rows].
void TransposeBf16(const uint16_t* src, int64_t rows, int64_t cols,
                   uint16_t* dst) {
  for (int64_t r = 0; r < rows; ++r) {
    const uint16_t* src_row = src + r * cols;
    for (int64_t c = 0; c < cols; ++c) dst[c * rows + r] = src_row[c];
  }
}

// Materialize an expanded GDN projection from its dequantized [out, in] = [N, K]
// row-major bf16 buffer `dq`. With `nk` (GgufLoadPolicy::gdn_expand_nk) the
// weight is KEPT in that native [N, K] order (nk = true), so it reaches the
// M-blocked vt::MatmulBT instead of the N-striding vt::Matmul — the same
// orientation win GgufLoadPolicy::expand_nk already gives every other expanded
// weight, and bit-identical for the same reason (the CPU kernels differ only in
// the weight offset, MatmulOneChunk keeps its sequential f32 K reduction).
// Without it, TODAY's transpose to Matmul-B [K, N] is preserved (nk = false),
// which VT_CPU_REF=1 and VT_GGUF_GDN_NK=0 select for the A/B oracle. Any V-head
// reorder is applied to `dq` (rows or cols) by the caller BEFORE this, so it is
// orthogonal to the orientation chosen here.
OwnedTensor MakeGdnProj(const std::vector<uint16_t>& dq, int64_t out_dim,
                        int64_t in_dim, bool nk) {
  if (nk) {
    OwnedTensor o = MakeOwned(vt::DType::kBF16, {out_dim, in_dim});  // [N, K]
    std::memcpy(o.bytes.data(), dq.data(), dq.size() * sizeof(uint16_t));
    o.nk = true;
    return o;
  }
  OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});  // [K, N]
  TransposeBf16(dq.data(), out_dim, in_dim,
                reinterpret_cast<uint16_t*>(o.bytes.data()));
  return o;
}

// Reorder the V-head rows in [row_off, row_off + num_v*head_rows) of a
// [rows, cols] row-major buffer from GGUF tiled order back to HF grouped order
// (inverse of conversion/qwen.py _reorder_v_heads). grouped head g=k*R+r reads
// tiled head t=r*K+k (R = num_v_per_k, K = num_k_heads).
template <typename T>
void ReorderVRows(std::vector<T>& buf, int64_t cols, int64_t row_off,
                  int64_t num_k, int64_t num_v_per_k, int64_t head_rows) {
  const int64_t num_v = num_k * num_v_per_k;
  const int64_t head_stride = head_rows * cols;
  std::vector<T> seg(static_cast<size_t>(num_v) * head_stride);
  T* base = buf.data() + row_off * cols;
  for (int64_t k = 0; k < num_k; ++k) {
    for (int64_t r = 0; r < num_v_per_k; ++r) {
      const int64_t g = k * num_v_per_k + r;
      const int64_t t = r * num_k + k;
      std::memcpy(seg.data() + g * head_stride, base + t * head_stride,
                  static_cast<size_t>(head_stride) * sizeof(T));
    }
  }
  std::memcpy(base, seg.data(), seg.size() * sizeof(T));
}



// Reorder the full column range [0, cols) of a [rows, cols] row-major buffer
// (cols = num_v * head_cols) from GGUF tiled to HF grouped order (out_proj).
template <typename T>
void ReorderVCols(std::vector<T>& buf, int64_t rows, int64_t cols,
                  int64_t num_k, int64_t num_v_per_k, int64_t head_cols) {
  std::vector<int64_t> src_col(static_cast<size_t>(cols));
  for (int64_t k = 0; k < num_k; ++k) {
    for (int64_t r = 0; r < num_v_per_k; ++r) {
      const int64_t g = k * num_v_per_k + r;  // grouped head (dst)
      const int64_t t = r * num_k + k;        // tiled head (src)
      for (int64_t h = 0; h < head_cols; ++h) {
        src_col[static_cast<size_t>(g * head_cols + h)] = t * head_cols + h;
      }
    }
  }
  std::vector<T> out(buf.size());
  for (int64_t rr = 0; rr < rows; ++rr) {
    const T* in_row = buf.data() + rr * cols;
    T* out_row = out.data() + rr * cols;
    for (int64_t c = 0; c < cols; ++c) out_row[c] = in_row[src_col[c]];
  }
  buf.swap(out);
}

// The per-tensor scalars an encoding keeps OUTSIDE its blocks.
//
// Every ggml encoding we read is self-contained except NVFP4 (type 40): its
// blocks carry the per-16 fp8 scales, but the tensor value ALSO needs the
// `<stem>.scale` f32 sidecar TENSOR the converter emits beside the weight (the
// reciprocal of the compressed-tensors `weight_global_scale` divisor). Without
// it the weights are off by a per-tensor factor of ~1e4 — finite, plausible and
// wrong — so this resolves it explicitly and fails loudly when it is missing.
//
// A 2-D weight has ONE scalar; a stacked expert weight (GGUF [E, out, in]) has
// ONE PER EXPERT in expert order, exactly like vLLM's `w13_weight_scale_2`. The
// result is therefore size 1 or size E, and {1.0f} for a self-contained
// encoding. `<stem>.input_scale` is the W4A4 ACTIVATION scale; it is read by the
// SAME helper (GgufNvfp4SidecarScalars below) for the fp4-resident path and
// stays unread on this weight-only dequant path.
std::vector<float> GgufNvfp4SidecarScalars(const GgufFile& g,
                                           const std::string& name,
                                           const GgufTensorInfo& t,
                                           const std::string& suffix) {
  const std::string kSuffix = ".weight";
  VT_CHECK(name.size() > kSuffix.size() &&
               name.compare(name.size() - kSuffix.size(), kSuffix.size(),
                            kSuffix) == 0,
           "qwen3_5 gguf: NVFP4 tensor without a .weight name: " + name);
  const std::string sidecar =
      name.substr(0, name.size() - kSuffix.size()) + suffix;
  VT_CHECK(HasTensor(g, sidecar),
           "qwen3_5 gguf: " + name + " is NVFP4 but the file has no " +
               sidecar + " sidecar; its blocks alone do not determine the "
                         "weight");
  const GgufTensorInfo& s = g.Get(sidecar);
  VT_CHECK(s.ggml_type == 0, "qwen3_5 gguf: " + sidecar + " must be F32");
  const int64_t n = ShapeNumel(s.shape);
  VT_CHECK(n == 1 || (t.shape.size() == 3 && n == t.shape[0]),
           "qwen3_5 gguf: " + sidecar +
               " must hold one scalar, or one per expert");
  std::vector<float> out(static_cast<size_t>(n));
  std::memcpy(out.data(), s.data, static_cast<size_t>(n) * sizeof(float));
  for (float v : out) {
    VT_CHECK(std::isfinite(v) && v > 0.0F,
             "qwen3_5 gguf: non-positive or non-finite scale in " + sidecar);
  }
  return out;
}

std::vector<float> GgufGlobalScales(const GgufFile& g, const std::string& name,
                                    const GgufTensorInfo& t) {
  if (!GgmlTypeNeedsGlobalScale(t.ggml_type)) return {1.0F};
  return GgufNvfp4SidecarScalars(g, name, t, ".scale");
}

// Dequant a named GGUF tensor (natural [out, in] row-major order, ne0 = in
// fastest). Returns the flat buffer; `info` is filled in.
//
// Templated only so the per-expert slab loop — one dequant call per sidecar
// scale — is not written twice for the f32 and bf16 element types.
template <typename T, typename Fn>
std::vector<T> DqSlabs(const GgufFile& g, const std::string& name,
                       const GgufTensorInfo** info, Fn dequant) {
  const GgufTensorInfo& t = g.Get(name);
  *info = &t;
  const int64_t numel = ShapeNumel(t.shape);
  const std::vector<float> scales = GgufGlobalScales(g, name, t);

  std::vector<T> out;
  if (scales.size() == 1) {
    out = dequant(t.ggml_type, t.data, numel, scales[0]);
  } else {
    // One contiguous slab per expert. The expert axis is the SLOWEST, so a slab
    // is a whole number of rows and no block is ever straddled.
    const int64_t slabs = static_cast<int64_t>(scales.size());
    VT_CHECK(numel % slabs == 0,
             "qwen3_5 gguf: expert count does not divide " + name);
    const int64_t per = numel / slabs;
    const GgmlTypeTraits& traits = GgmlTraits(t.ggml_type);
    VT_CHECK(per % traits.block_elems == 0,
             "qwen3_5 gguf: expert slab of " + name + " is not whole blocks");
    const int64_t slab_bytes = per / traits.block_elems * traits.block_bytes;
    out.resize(static_cast<size_t>(numel));
    for (int64_t e = 0; e < slabs; ++e) {
      const std::vector<T> part =
          dequant(t.ggml_type, t.data + e * slab_bytes, per,
                  scales[static_cast<size_t>(e)]);
      std::memcpy(out.data() + e * per, part.data(),
                  static_cast<size_t>(per) * sizeof(T));
    }
  }
  // The expansion is now the authority for this tensor; its file pages are
  // read-once and must not sit in RSS next to it (L5, no-op unless the load
  // enabled it). Re-reading the tensor later is still correct — the pages just
  // fault back in.
  g.DropSpanResidency(t.data, t.nbytes);
  return out;
}

std::vector<uint16_t> DqBf16(const GgufFile& g, const std::string& name,
                             const GgufTensorInfo** info) {
  return DqSlabs<uint16_t>(g, name, info,
                           [](uint32_t ty, const uint8_t* d, int64_t n,
                              float s) {
                             return DequantGgufRowToBf16(ty, d, n, s);
                           });
}

std::vector<float> DqF32(const GgufFile& g, const std::string& name,
                         const GgufTensorInfo** info) {
  return DqSlabs<float>(g, name, info,
                        [](uint32_t ty, const uint8_t* d, int64_t n, float s) {
                          return DequantGgufRowToF32(ty, d, n, s);
                        });
}

// bf16 tensor copied verbatim with `shape` (dequant, then own the bytes).
OwnedTensor OwnBf16(const GgufFile& g, const std::string& name,
                    const std::vector<int64_t>& shape) {
  const GgufTensorInfo* t = nullptr;
  std::vector<uint16_t> dq = DqBf16(g, name, &t);
  VT_CHECK(ShapeNumel(shape) == static_cast<int64_t>(dq.size()),
           "qwen3_5 gguf: element-count mismatch for " + name);
  OwnedTensor o = MakeOwned(vt::DType::kBF16, shape);
  std::memcpy(o.bytes.data(), dq.data(), dq.size() * sizeof(uint16_t));
  return o;
}

// bf16 [out, in] -> owned bf16 [in, out] (Matmul-B layout).
OwnedTensor OwnBf16T(const GgufFile& g, const std::string& name) {
  const GgufTensorInfo* t = nullptr;
  std::vector<uint16_t> dq = DqBf16(g, name, &t);
  VT_CHECK(t->shape.size() == 2, "qwen3_5 gguf: expected 2-D weight " + name);
  const int64_t out_dim = t->shape[0];
  const int64_t in_dim = t->shape[1];
  OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
  TransposeBf16(dq.data(), out_dim, in_dim,
                reinterpret_cast<uint16_t*>(o.bytes.data()));
  return o;
}

// RMSNorm weight stored (w + 1) by the convert -> owned bf16 [n] holding raw w.
OwnedTensor OwnNormMinus1(const GgufFile& g, const std::string& name) {
  const GgufTensorInfo* t = nullptr;
  std::vector<float> dq = DqF32(g, name, &t);
  OwnedTensor o = MakeOwned(vt::DType::kBF16,
                            {static_cast<int64_t>(dq.size())});
  auto* dst = reinterpret_cast<uint16_t*>(o.bytes.data());
  for (size_t i = 0; i < dq.size(); ++i) dst[i] = vt::F32ToBF16(dq[i] - 1.0F);
  return o;
}

// --- residency routing (L3) ----------------------------------------------
//
// EVERY tensor this loader touches passes through GgufLoadPolicy::Route with
// an explicit role, so the policy's audit hook observes the complete tensor
// list and no tensor can reach a residency by omission.

// A 2-D GEMM weight taken verbatim from the file. Three outcomes, in order:
//
//  1. keep-quant       -> raw blocks in the file's [N, K] order (nk = true);
//  2. expand, nk       -> bf16 in the file's OWN [N, K] order (nk = true), so
//                         the GEMM is the contiguous-weight-row kMatmulBT
//                         instead of the N-striding kMatmul, with NO transpose
//                         at load. Bit-identical to (3) on the CPU kernels
//                         (cpu_ops.cpp MatmulOneChunk<kBT> differs only in the
//                         weight offset) — see GgufLoadPolicy::expand_nk;
//  3. expand, Matmul-B -> TODAY'S EXACT path, dequant to bf16 and transpose to
//                         [K, N], byte for byte unchanged. This is what
//                         VT_CPU_REF=1 and every non-CPU device get.
//
// (2) matters because a "mixed" GGUF is the normal case, not the exception:
// the Unsloth Qwen3.5-2B-UD-Q8_K_XL bench file is 103 q8_0 tensors but ALSO 56
// f16 ones, including the tied token_embd/lm_head and whole layers of ffn — no
// encoding keeps those blocks, so without (2) the biggest GEMM in the model
// would still be paying the slow orientation.
OwnedTensor OwnMatmulWeight(const GgufFile& g, const std::string& name,
                            const GgufLoadPolicy& pol) {
  const GgufTensorInfo& t = g.Get(name);
  const GgufResidency r = pol.Route(t, GgufTensorRole::kMatmulWeight);
  // The fp4 residency does not produce an OwnedTensor at all (its product is an
  // Nvfp4Weight), so reaching here with it means a call site that CAN take fp4
  // did not, or one that CANNOT was not given NoNvfp4(). Loud, never silent.
  VT_CHECK(r != GgufResidency::kNvfp4Fp4,
           "qwen3_5 gguf: " + name +
               " routed to the fp4 residency but this call site builds a bf16 "
               "OwnedTensor; use OwnGgufNvfp4 or NoNvfp4()");
  if (GgufResidencyKeepsBlockWeights(r)) {
    // keep-quant blocks OR keep-f16, both in the file's own [N, K] order.
    VT_CHECK(t.shape.size() == 2, "qwen3_5 gguf: expected 2-D weight " + name);
    return OwnGgufKeptSlice(g, pol, t, r, t.shape[0], t.shape[1], 0);
  }
  if (pol.expand_nk) {
    VT_CHECK(t.shape.size() == 2, "qwen3_5 gguf: expected 2-D weight " + name);
    OwnedTensor o = OwnBf16(g, name, t.shape);
    o.nk = true;
    return o;
  }
  return OwnBf16T(g, name);
}

// --- native fp4 residency (`QUANT-GGUF-NVFP4` column C) --------------------
//
// Build the fp4-resident `Nvfp4Weight` for ONE row-slice of an NVFP4 GGUF
// tensor: the repacked operand pair plus the per-tensor scalars. The result is
// EXACTLY what LoadCtNvfp4Raw produces from the compressed-tensors container of
// the same quantization run (its `weight_packed` / `weight_scale` bytes are
// byte-identical to the repack — measured, see the leaf spec Sec B), so the
// forward's fp4 dispatch is reached with operands it has already been gated on.
//
// `scale2` is the `<stem>.scale` sidecar: the multiply-not-divide
// `weight_scale_2` form, i.e. the RECIPROCAL of the compressed-tensors
// `weight_global_scale` DIVISOR, bit-identical to it on the real files. The
// divisor itself is NOT in the GGUF, so `weight_global_scale_inv` (which holds
// the divisor, for the merged-linear max-then-reciprocate rule vLLM uses) is
// RECONSTRUCTED as 1/scale2 and may differ from the safetensors divisor by one
// ULP; recorded in the leaf spec's Risks/decisions rather than hidden.
//
// `input_scale2` is the `<stem>.input_scale` sidecar in the same reciprocal
// form. With W4A4 on, the ACTIVATION divisor `input_global_scale_inv` (used
// DIRECTLY by vt::ScaledFp4Quant) is likewise 1/input_scale2, and `alpha` is
// formed by the SAME expression LoadCtNvfp4Raw uses so the two containers'
// loaders cannot drift. With W4A4 off, both stay 0 and `IsTrueW4A4()` is false,
// which is precisely how the forward selects vLLM's `use_a16` (W4A16) arm.
Nvfp4Weight OwnGgufNvfp4Slice(const GgufTensorInfo& t, int64_t n, int64_t k,
                              int64_t row_offset, float scale2,
                              float input_scale2, bool w4a4) {
  VT_CHECK(t.ggml_type == 40,
           "qwen3_5 gguf: fp4 residency asked for a non-NVFP4 tensor " + t.name);
  VT_CHECK(k % 64 == 0,
           "qwen3_5 gguf: NVFP4 K must be whole 64-element blocks for " +
               t.name);
  VT_CHECK(std::isfinite(scale2) && scale2 > 0.0F,
           "qwen3_5 gguf: bad <stem>.scale for " + t.name);
  Nvfp4Weight w;
  w.n = n;
  w.k = k;
  w.scale2 = scale2;
  w.weight_global_scale_inv = 1.0F / scale2;  // reconstructed disk divisor
  if (w4a4) {
    VT_CHECK(std::isfinite(input_scale2) && input_scale2 > 0.0F,
             "qwen3_5 gguf: bad <stem>.input_scale for " + t.name);
    w.input_global_scale_inv = 1.0F / input_scale2;
    // Same expression as LoadCtNvfp4Raw: scale2 * (1/input_divisor).
    w.alpha = w.scale2 * (1.0F / w.input_global_scale_inv);
  }
  w.packed = MakeOwned(vt::DType::kI8, {n, k / 2});
  w.scale = MakeOwned(vt::DType::kI8, {n, k / 16});
  const int64_t blocks_per_row = k / 64;
  const uint8_t* src = t.data + row_offset * blocks_per_row * 36;
  RepackGgufNvfp4Rows(src, n, k,
                      reinterpret_cast<uint8_t*>(w.packed.bytes.data()),
                      reinterpret_cast<uint8_t*>(w.scale.bytes.data()));
  return w;
}

// A 2-D NVFP4 matmul weight, taken verbatim in the file's own [N=out, K=in]
// orientation — which is already the orientation vt::MatmulNvfp4 reads, so
// unlike the bf16 path there is no transpose question here.
Nvfp4Weight OwnGgufNvfp4(const GgufFile& g, const std::string& name,
                         const GgufLoadPolicy& pol) {
  const GgufTensorInfo& t = g.Get(name);
  VT_CHECK(t.shape.size() == 2, "qwen3_5 gguf: expected 2-D NVFP4 weight " + name);
  const std::vector<float> ws = GgufNvfp4SidecarScalars(g, name, t, ".scale");
  VT_CHECK(ws.size() == 1,
           "qwen3_5 gguf: a 2-D NVFP4 weight needs exactly one .scale: " + name);
  float is2 = 0.0F;
  if (pol.nvfp4_w4a4) {
    const std::vector<float> in =
        GgufNvfp4SidecarScalars(g, name, t, ".input_scale");
    VT_CHECK(in.size() == 1,
             "qwen3_5 gguf: a 2-D NVFP4 weight needs exactly one .input_scale: " +
                 name);
    is2 = in[0];
  }
  Nvfp4Weight w = OwnGgufNvfp4Slice(t, t.shape[0], t.shape[1], 0, ws[0], is2,
                                    pol.nvfp4_w4a4);
  // The blocks are read once and the repack is now the authority for this
  // tensor; drop their file pages exactly as the bf16 expansion path does.
  g.DropSpanResidency(t.data, t.nbytes);
  return w;
}

// A stacked expert tensor (GGUF [E, out, in]) -> one Nvfp4Weight per expert.
// The expert axis is the SLOWEST, so an expert slab is a whole number of ROWS
// and therefore of 64-element blocks; the split is a byte range and no block is
// ever cut. The sidecars hold ONE scalar PER EXPERT in expert order, matching
// vLLM's `w13_weight_scale_2`.
std::vector<Nvfp4Weight> OwnGgufNvfp4Experts(const GgufFile& g,
                                             const std::string& name,
                                             int64_t num_experts,
                                             const GgufLoadPolicy& pol) {
  const GgufTensorInfo& t = g.Get(name);
  VT_CHECK(t.shape.size() == 3 && t.shape[0] == num_experts,
           "qwen3_5 gguf: expected [E,out,in] NVFP4 expert tensor " + name);
  const int64_t out_dim = t.shape[1];
  const int64_t in_dim = t.shape[2];
  const std::vector<float> ws = GgufNvfp4SidecarScalars(g, name, t, ".scale");
  VT_CHECK(ws.size() == 1 || static_cast<int64_t>(ws.size()) == num_experts,
           "qwen3_5 gguf: .scale must hold one scalar or one per expert: " + name);
  std::vector<float> in(1, 0.0F);
  if (pol.nvfp4_w4a4) {
    in = GgufNvfp4SidecarScalars(g, name, t, ".input_scale");
    VT_CHECK(in.size() == 1 || static_cast<int64_t>(in.size()) == num_experts,
             "qwen3_5 gguf: .input_scale must hold one scalar or one per "
             "expert: " + name);
  }
  std::vector<Nvfp4Weight> out;
  out.reserve(static_cast<size_t>(num_experts));
  for (int64_t e = 0; e < num_experts; ++e) {
    const float s2 = ws.size() == 1 ? ws[0] : ws[static_cast<size_t>(e)];
    const float i2 = in.size() == 1 ? in[0] : in[static_cast<size_t>(e)];
    out.push_back(OwnGgufNvfp4Slice(t, out_dim, in_dim, e * out_dim, s2, i2,
                                    pol.nvfp4_w4a4));
  }
  g.DropSpanResidency(t.data, t.nbytes);
  return out;
}

// `PeekRoute` — the pure routing decision without the audit hook — used to live
// here, next to its first caller (the tied-head probe in LoadEmbedAndHead uses
// the same pattern). It now lives in `gguf_keep_quant.h` beside the routing
// function it delegates to, because the load-time device-fit lane asks the same
// question of the same expert towers and one decision may have only one
// description. These call sites bind to `vllm::PeekRoute` unchanged.

// A policy copy with the fp4 residency DISABLED, for the call sites whose
// consumer is an OwnedTensor and has no `Nvfp4Weight` field to fill: the
// lm_head/embedding pair, the MoE router gate, and the GDN in_proj family
// (whose 27B V-head reorder makes them kTransformedWeight anyway, but which
// would otherwise become fp4-eligible on a model without that reorder). Stating
// it as a policy rather than a silent fallthrough keeps the audit hook truthful
// about what each tensor actually got.
GgufLoadPolicy NoNvfp4(const GgufLoadPolicy& pol) {
  GgufLoadPolicy p = pol;
  p.nvfp4_fp4 = false;
  p.nvfp4_w4a4 = false;
  return p;
}

// Route a tensor that can NEVER keep its blocks (a value/layout rewrite, a
// gather table, a conv filter or a 1-D vector) and assert the policy agrees.
// This is the runtime half of the totality contract: the audit hook still sees
// the tensor, and a policy that ever tried to keep such a tensor quantized
// would fail LOUDLY here rather than silently skip its transform.
void RequireExpand(const GgufLoadPolicy& pol, const GgufFile& g,
                   const std::string& name, GgufTensorRole role) {
  VT_CHECK(pol.Route(g.Get(name), role) == GgufResidency::kExpandBf16,
           std::string("qwen3_5 gguf: a ") + Name(role) +
               " tensor must not keep quant blocks: " + name);
}

// --- embedding + lm_head, with tied-head sharing (L5 bf16 / L6 f16) --------
//
// A tied-embedding GGUF omits `output.weight`: the head IS `token_embd.weight`,
// and llama.cpp materializes no second copy — it aliases the head onto the
// embedding (TENSOR_DUPLICATED). Materialized twice, that is a whole extra
// vocab matrix (248320 x 2048 = 0.947 GiB on the 2B bench file).
//
// The two uses want the SAME bytes in the SAME order and differ only in
// metadata: the gather table is [vocab, H] with nk = false, the GEMM weight is
// [N = vocab, K = H] with nk = true. They coincide — so ONE resident matrix can
// serve both — in exactly two configurations, which is what this function picks:
//
//   * L6 keep-f16: token_embd is F16 and stays F16 for both uses. The gather
//     table borrows the file's f16 bytes (or a copy) and the head borrows the
//     SAME bytes with nk = true. This is the cheapest — no bf16 re-expansion at
//     all — and matches llama.cpp, which keeps token_embd f16 and gathers/GEMMs
//     on it directly.
//   * L5 expand-bf16: both expand to bf16 in the file's own [N, K] order, which
//     is the `expand_nk` case; the transposing Matmul-B path (VT_CPU_REF, or a
//     non-CPU device) builds genuinely different bytes and shares nothing.
//
// Either way both tensors BORROW one shared, refcounted buffer (the mmap, or one
// bf16/f16 expansion via OwnedBytes::KeepAlive); neither owns it, so there is no
// "which one frees it" question and no double free. If the two do not coincide,
// the head falls through to OwnMatmulWeight — its own residency, a second buffer
// — exactly as before L5. token_embd is routed TWICE when tied (once as the
// gather table, once as the GEMM head), which is the correct answer for both.
void LoadEmbedAndHead(const GgufFile& g, const GgufLoadPolicy& pol,
                      OwnedTensor* embed, OwnedTensor* head) {
  const std::string kEmbed = "token_embd.weight";
  const GgufTensorInfo& et = g.Get(kEmbed);
  // The embedding table is a gather and the head a bf16/keep-quant GEMM weight;
  // neither has an Nvfp4Weight field, so the fp4 residency is off for both (the
  // table is ineligible by ROLE already; the head is disabled explicitly below).
  const GgufResidency embed_r =
      pol.Route(et, GgufTensorRole::kEmbeddingTable);  // the embed's audit event

  // Build the gather table in its chosen residency: keep-quant blocks, keep-f16
  // in place, else the historical bf16 expansion.
  //
  // The keep-quant arm is MODEL-MM-QWEN4-EXP W6a. Until the dequantizing gather
  // existed this branch was a hard refusal, and the refusal was RIGHT: a kept
  // table would have been bytes no op in the tree could read. `vt::Embedding`
  // now decodes one row per gathered id, so the table's residency follows the
  // same policy every other tensor's does.
  if (embed_r == GgufResidency::kKeepQuant) {
    VT_CHECK(et.shape.size() == 2, "qwen3_5 gguf: token_embd must be 2-D");
    // NEITHER repack applies, and both would be silent corruption rather than a
    // slowdown: the i8mm interleave and the CUDA coalesced-load permutation
    // rewrite the bytes INSIDE a row for a vec_dot to consume, and the gather
    // decodes the row in its ggml order. `nk` stays false for the same reason
    // it does on the f16 arm — this tensor is a table, not a [N,K] GEMM weight.
    *embed = OwnGgufQuantBlocks(et, et.shape[0], et.shape[1], /*row_offset=*/0,
                                MmapSrc(g, pol), /*repack=*/false,
                                /*cuda_align=*/false, pol.prefault,
                                GgufTensorRole::kEmbeddingTable);
    embed->nk = false;
  } else if (embed_r == GgufResidency::kKeepF16) {
    VT_CHECK(et.shape.size() == 2, "qwen3_5 gguf: token_embd must be 2-D");
    // Embedding gather table: never repacked (EmbeddingKernel reads it row-wise).
    *embed = OwnGgufF16(et, et.shape[0], et.shape[1], 0, MmapSrc(g, pol),
                        /*nk=*/false, /*elem_kn_repack=*/false, pol.prefault,
                        pol.weight_value_dtype);
  } else {
    VT_CHECK(embed_r == GgufResidency::kExpandBf16,
             "qwen3_5 gguf: unexpected embedding-table residency " +
                 std::string(Name(embed_r)));
    *embed = OwnBf16(g, kEmbed, et.shape);
  }

  const bool tied = !HasTensor(g, "output.weight");
  const std::string head_name = tied ? kEmbed : "output.weight";
  const GgufTensorInfo& ht = g.Get(head_name);
  // Ask the PURE decision function, not `pol.Route`, so probing the head here
  // does not fire the audit hook a second time: the head is routed EXACTLY ONCE
  // on both branches below.
  const GgufResidency head_r =
      RouteGgufTensor(pol.keep_quant, pol.keep_f16, /*nvfp4_fp4=*/false,
                      pol.cpu_ref, GgufTensorRole::kMatmulWeight, ht.ggml_type,
                      ht.shape, pol.device, pol.weight_value_dtype);

  // The two coincide (share one buffer) iff both kept f16, OR both expanded in
  // the file's own [N, K] order (expand_nk). share_tied_head already implies
  // expand_nk (FromEnv), which is why the bf16 arm needs no extra check.
  //
  // A tied KEEP-QUANT pair deliberately does NOT share, even though the file
  // bytes are the same bytes: the head goes through `OwnMatmulWeight`, which
  // may REPACK the slice for i8mm or for the CUDA coalesced load, and a table
  // sharing a repacked buffer would gather permuted rows. The cost is one
  // duplicated block image on a tied quantized file; the alternative is a
  // corruption that only appears on the hosts where a repack is live. Making
  // them share correctly needs the repack decision hoisted, which is a change to
  // `OwnMatmulWeight` and not to this function.
  const bool f16_share = embed_r == GgufResidency::kKeepF16 &&
                         head_r == GgufResidency::kKeepF16;
  const bool bf16_share = embed_r == GgufResidency::kExpandBf16 &&
                          head_r == GgufResidency::kExpandBf16 && pol.expand_nk;
  // TT-EMBED-DUAL-RESIDENCY: a tied KEEP-QUANT pair shares when the device
  // cannot repack. The original refusal (:871-878) names the head's
  // OwnMatmulWeight repack; QuantRepackForDevice (gguf_keep_quant.cpp:96-100)
  // answers that only on kCPU, so on kTENSTORRENT the head's words are the
  // file's own GGML block order — the SAME bytes the gather decodes. Sharing
  // is REQUIRED there, not just cheaper: the TT gather reads the shared word
  // shadow, and an unshared table would stage a SECOND ~vocab-sized word
  // image beside the head's. CPU keeps the refusal (repack live).
  const bool kq_share = embed_r == GgufResidency::kKeepQuant &&
                        head_r == GgufResidency::kKeepQuant &&
                        !pol.quant_repack &&
                        pol.device == vt::DeviceType::kTENSTORRENT;
  if (tied && pol.share_tied_head &&
      (f16_share || bf16_share || kq_share)) {
    (void)NoNvfp4(pol).Route(ht,
                             GgufTensorRole::kMatmulWeight);  // head's 1 audit
    std::shared_ptr<const void> owner = embed->bytes.KeepAlive();
    OwnedTensor h;
    h.dtype = embed->dtype;
    h.weight_value_dtype = embed->weight_value_dtype;
    h.rank = embed->rank;
    for (int i = 0; i < embed->rank; ++i) h.shape[i] = embed->shape[i];
    h.nk = true;  // [N = vocab, K = H], the file's own order
    // The keep-quant borrow carries the same metadata OwnMatmulWeight would
    // have built (nk = true, GGUF disk order IS the MatmulBT [N, K]
    // orientation; pol.quant_repack is false on this device) — minus the
    // second copy of the bytes. The gather keeps the EMBED tensor's own
    // nk = false view of the same bytes.
    h.bytes = OwnedBytes::Borrow(embed->bytes.data(), embed->bytes.size(), owner);
    *head = std::move(h);
    return;
  }
  *head = OwnMatmulWeight(g, head_name, NoNvfp4(pol));
}

// --- config --------------------------------------------------------------

int64_t KvInt(const GgufValue& v, const std::string& key) {
  switch (v.TypeId()) {
    case kGgufU8: return std::get<uint8_t>(v.v);
    case kGgufI8: return std::get<int8_t>(v.v);
    case kGgufU16: return std::get<uint16_t>(v.v);
    case kGgufI16: return std::get<int16_t>(v.v);
    case kGgufU32: return std::get<uint32_t>(v.v);
    case kGgufI32: return std::get<int32_t>(v.v);
    case kGgufU64: return static_cast<int64_t>(std::get<uint64_t>(v.v));
    case kGgufI64: return std::get<int64_t>(v.v);
    case kGgufBool: return std::get<bool>(v.v) ? 1 : 0;
    default:
      throw std::runtime_error("qwen3_5 gguf: key " + key +
                               " is not an integer");
  }
}

double KvFloat(const GgufValue& v, const std::string& key) {
  if (v.TypeId() == kGgufF32) return std::get<float>(v.v);
  if (v.TypeId() == kGgufF64) return std::get<double>(v.v);
  return static_cast<double>(KvInt(v, key));
}

int64_t ReqInt(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  VT_CHECK(v != nullptr, "qwen3_5 gguf: missing metadata key " + key);
  return KvInt(*v, key);
}

double ReqFloat(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  VT_CHECK(v != nullptr, "qwen3_5 gguf: missing metadata key " + key);
  return KvFloat(*v, key);
}

int64_t OptInt(const GgufFile& g, const std::string& key, int64_t dflt) {
  const GgufValue* v = g.FindKv(key);
  return v ? KvInt(*v, key) : dflt;
}

}  // namespace

HfConfig HfConfigFromGguf(const GgufFile& gguf) {
  const GgufValue* arch_v = gguf.FindKv("general.architecture");
  VT_CHECK(arch_v != nullptr && arch_v->TypeId() == kGgufString,
           "qwen3_5 gguf: general.architecture must be a string");
  const std::string arch = std::get<std::string>(arch_v->v);
  VT_CHECK(arch == "qwen35moe" || arch == "qwen3next" || arch == "qwen35",
           "qwen3_5 gguf: unexpected architecture '" + arch + "'");
  const std::string p = arch + ".";

  HfConfig c;
  c.model_type = arch;
  // `general.architecture` is llama.cpp's GGUF family key, not the HuggingFace
  // model-class ID consumed by vLLM's ModelRegistry. Map it onto the canonical
  // registered architecture (dense `qwen35` -> the 27B-family dense wrapper,
  // the MoE keys -> the MoE wrapper) while retaining the GGUF key in
  // model_type for metadata lookup.
  c.architectures = {arch == "qwen35" ? "Qwen3_5ForConditionalGeneration"
                                      : "Qwen3_5MoeForConditionalGeneration"};

  c.hidden_size = ReqInt(gguf, p + "embedding_length");
  const int64_t block_count = ReqInt(gguf, p + "block_count");
  const int64_t nextn = OptInt(gguf, p + "nextn_predict_layers", 0);
  c.num_hidden_layers = block_count - nextn;
  // The MTP head depth, republished under the HF spelling the rest of the
  // engine reads (ResolveSpecConfig -> NumMtpLayers -> raw
  // "mtp_num_hidden_layers"). llama.cpp's converter writes it as
  // `<arch>.nextn_predict_layers` and folds the head blocks into block_count,
  // which is why the trunk layer count above subtracts it. Without this the
  // spec resolver would silently fall back to its depth-1 default on a GGUF
  // whose head is deeper. Absent key => 0 => no head, and we publish nothing.
  if (nextn > 0) c.raw["mtp_num_hidden_layers"] = nextn;

  c.num_attention_heads = ReqInt(gguf, p + "attention.head_count");
  c.num_key_value_heads =
      OptInt(gguf, p + "attention.head_count_kv", c.num_attention_heads);
  c.head_dim = OptInt(gguf, p + "attention.key_length",
                      c.num_attention_heads > 0
                          ? c.hidden_size / c.num_attention_heads
                          : 0);

  // vocab_size: prefer the kv, else read token_embd's leading (out) dim.
  const GgufValue* vocab_kv = gguf.FindKv(p + "vocab_size");
  c.vocab_size = vocab_kv ? KvInt(*vocab_kv, p + "vocab_size")
                          : gguf.Get("token_embd.weight").shape[0];

  // MoE.
  c.num_experts = OptInt(gguf, p + "expert_count", 0);
  c.num_experts_per_tok = OptInt(gguf, p + "expert_used_count", 0);
  c.moe_intermediate_size =
      OptInt(gguf, p + "expert_feed_forward_length", 0);
  c.shared_expert_intermediate_size =
      OptInt(gguf, p + "expert_shared_feed_forward_length", 0);
  c.intermediate_size = OptInt(gguf, p + "feed_forward_length", 0);

  // Gated DeltaNet (llama.cpp qwen35moe: head_k/v_dim = ssm.state_size,
  // n_k_heads = ssm.group_count, n_v_heads = ssm.time_step_rank).
  c.linear_num_key_heads = OptInt(gguf, p + "ssm.group_count", 0);
  c.linear_num_value_heads = OptInt(gguf, p + "ssm.time_step_rank", 0);
  const int64_t ssm_state = OptInt(gguf, p + "ssm.state_size", 0);
  c.linear_key_head_dim = ssm_state;
  c.linear_value_head_dim = ssm_state;
  c.linear_conv_kernel_dim = OptInt(gguf, p + "ssm.conv_kernel", 0);
  // The GGUF metadata does not currently carry HF's mamba_ssm_dtype, but this
  // loader is specific to the Qwen3.5/3.6 family whose recurrent cache contract
  // is FP32. Preserve it when reconstructing HfConfig so safetensors and GGUF
  // allocate the same BF16-conv/FP32-SSM state pair.
  c.mamba_ssm_dtype = "float32";

  // RoPE / norm / context.
  const GgufValue* freq = gguf.FindKv(p + "rope.freq_base");
  c.rope_theta = freq ? KvFloat(*freq, p + "rope.freq_base") : 10000.0;
  // llama.cpp omits rope.dimension_count when it equals head_dim (full rotary).
  // Qwen3.5 has no partial_rotary_factor, so default to head_dim.
  c.rotary_dim = OptInt(gguf, p + "rope.dimension_count", c.head_dim);
  c.rms_norm_eps = ReqFloat(gguf, p + "attention.layer_norm_rms_epsilon");
  c.max_position_embeddings = OptInt(gguf, p + "context_length", 0);
  c.torch_dtype = "bfloat16";

  // layer_types: explicit recurrent-layers bool array, else the
  // full_attention_interval pattern (every interval-th layer is full attn).
  c.layer_types.reserve(static_cast<size_t>(c.num_hidden_layers));
  const GgufValue* recr = gguf.FindKv(p + "attention.recurrent_layers");
  if (recr != nullptr && recr->TypeId() == kGgufArray) {
    const GgufArray& arr = std::get<GgufArray>(recr->v);
    for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
      const bool is_recr =
          l < static_cast<int64_t>(arr.elems.size()) &&
          KvInt(arr.elems[static_cast<size_t>(l)], "recurrent_layers") != 0;
      c.layer_types.emplace_back(is_recr ? "linear_attention"
                                         : "full_attention");
    }
  } else {
    const int64_t interval =
        OptInt(gguf, p + "full_attention_interval", 4);
    for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
      const bool is_recr = interval <= 0 || ((l + 1) % interval != 0);
      c.layer_types.emplace_back(is_recr ? "linear_attention"
                                         : "full_attention");
    }
  }
  return c;
}

// `SPEC-DFLASH-GGUF` B1. Contract + rationale in the header; the two things
// worth restating at the code are why this is a plain bf16 read and what it
// reuses.
//
// PLAIN BF16, on purpose, to match the safetensors side byte for byte: the
// draft's `ResidentWeight` views are `{vocab, H}` gather + `[N, K]` MatmulBT,
// so a keep-F16 or block-quantized residency would be a different tensor for
// the same weight depending on the target's container. `OwnBf16` also carries
// the `<stem>.scale` sidecar resolution (`DqSlabs` -> `GgufGlobalScales`), so
// an NVFP4-stored head is dequantized with its per-tensor factor rather than
// silently off by ~1e4.
//
// TIED FALLBACK, reusing the trunk's rule verbatim rather than a second one:
// a GGUF that omits `output.weight` has the head aliased onto the embedding
// (llama.cpp TENSOR_DUPLICATED), and `LoadEmbedAndHead` above resolves it the
// same way. The two copies here are NOT buffer-shared the way that function's
// L5 tied-head sharing is: these are the draft's own host tensors, one of them
// transposed in meaning (nk) though not in bytes, and the draft is loaded once
// at startup. Correctness first; the sharing is an RSS optimization that only
// pays on a tied file, which neither gate model is.
void LoadGgufSharedEmbedAndHeadBf16(const GgufFile& g, OwnedTensor* embed,
                                    OwnedTensor* head, bool* head_was_quantized) {
  const std::string kEmbed = "token_embd.weight";
  VT_CHECK(HasTensor(g, kEmbed),
           "gguf shared head: the target file has no " + kEmbed +
               " (a DFlash draft has no embedding of its own and reads the "
               "target's)");
  const GgufTensorInfo& et = g.Get(kEmbed);
  VT_CHECK(et.shape.size() == 2, "gguf shared head: " + kEmbed + " must be 2-D");
  *embed = OwnBf16(g, kEmbed, et.shape);  // [vocab, H]
  embed->nk = false;                      // a gather table, not a GEMM weight.

  const bool tied = !HasTensor(g, "output.weight");
  const std::string head_name = tied ? kEmbed : "output.weight";
  const GgufTensorInfo& ht = g.Get(head_name);
  VT_CHECK(ht.shape.size() == 2,
           "gguf shared head: " + head_name + " must be 2-D");
  *head = OwnBf16(g, head_name, ht.shape);  // [N = vocab, K = H]
  head->nk = true;
  // A scalar ggml type packs ONE element per "block"; anything else is
  // block-quantized and `OwnBf16` above dequantized it. SPEC-DFLASH2 W3 (#1314).
  if (head_was_quantized != nullptr)
    *head_was_quantized = GgmlTraits(ht.ggml_type).block_elems != 1;
}

// --- weights -------------------------------------------------------------

namespace {

std::string Blk(int64_t il, const std::string& suffix) {
  return "blk." + std::to_string(il) + "." + suffix;
}

GdnLayerWeights LoadGdnGguf(const GgufFile& g, int64_t il, const HfConfig& c,
                            const GgufLoadPolicy& pol_in) {
  const int64_t num_k = c.linear_num_key_heads;
  const int64_t num_v = c.linear_num_value_heads;
  const int64_t dv = c.linear_value_head_dim;
  const int64_t key_dim = num_k * c.linear_key_head_dim;
  const bool reorder = num_v != num_k && num_k > 0 && (num_v % num_k) == 0;
  const int64_t rpk = num_k > 0 ? num_v / num_k : 1;  // num_v_per_k
  // When the V-head reorder is active, the projections are LAYOUT-rewritten at
  // load. For COLUMN-permuted tensors (out_proj/ssm_out) the reorder cuts across
  // quantization block boundaries, so they are kTransformedWeight and must
  // expand to bf16. For ROW-permuted tensors (in_proj_qkv, in_proj_z) the
  // reorder only changes row order — quantization blocks are along the K
  // (column) dimension and are self-contained per row — so the blocks can be
  // kept and the permutation applied to the block rows at load time (T21).
  // Without reorder they are ordinary verbatim GEMM weights. Column-permuted
  // tensors (out_proj/ssm_out) stay kTransformedWeight and expand to bf16.
  // T21 env gate: VT_GDN_ROWPERM_KEEP_QUANT=0 forces the row-permuted tensors
  // back to kTransformedWeight (bf16 expansion) for A/B isolation.
  // Opt-in, by VALUE like its VT_GDN_COLPERM_KEEP_QUANT sibling: unset keeps
  // the load path byte-identical to the trunk (reorder, then expand to bf16).
  // A default-on flip here silently changed the trunk numerics of every
  // row-permuted V-head projection and broke the token-exact gate -- caught on
  // the gfx1100 engine leg, reverted to opt-in.
  const char* rpkq = std::getenv("VT_GDN_ROWPERM_KEEP_QUANT");
  const bool rowperm_keep =
      rpkq != nullptr && rpkq[0] == '1' && rpkq[1] == '\0';
  const GgufTensorRole proj_role = reorder
                                       ? GgufTensorRole::kTransformedWeight
                                       : GgufTensorRole::kMatmulWeight;
  // T25: keep the COLUMN-permuted tensor (ssm_out/out_proj) as K-quant in tiled
  // order (no ReorderVCols) and permute the GEMV input at runtime instead. The
  // column reorder cuts across Q5_K block boundaries, so the weight cannot be
  // permuted in place. But keeping the tiled-order weight and permuting the
  // 4096-element activation gather before the K-quant GEMV saves ~4x weight
  // bandwidth (Q5_K ~5 MB vs bf16 20 MB per call).
  const char* cpkq = std::getenv("VT_GDN_COLPERM_KEEP_QUANT");
  const bool colperm_keep =
      cpkq != nullptr && cpkq[0] == '1' && cpkq[1] == '\0';
  const GgufTensorRole colperm_role =
      (reorder && colperm_keep) ? GgufTensorRole::kMatmulWeight : proj_role;
  // GdnLayerWeights carries an Nvfp4Weight ONLY for out_proj, and even that is
  // unreachable on the 27B because the V-column reorder makes ssm_out
  // kTransformedWeight. The in_proj family has no fp4 field at all. So the GDN
  // block loads against a policy with the fp4 residency off: an NVFP4 GDN
  // projection (which the 27B GGUF does carry, unlike its safetensors sibling —
  // see the leaf spec Sec A) expands to bf16, correctly but unquantized. This is
  // the ONE documented gap left in the C column.
  const GgufLoadPolicy pol = NoNvfp4(pol_in);

  GdnLayerWeights gdn;

  // in_proj_qkv <- attn_qkv [conv_dim, H]; only the trailing V rows reorder.
  // T21: ReorderVRows is a row permutation (block-safe for K-quant). Route as
  // kMatmulWeight to allow keep-quant, then permute the block rows in place.
  // Saves ~661 MB/tok of bf16 read amplification (24 Q5_K tensors × 2.9x).
  // The forward pass already dispatches quantized nk=true weights through
  // vt::MatmulBT → matmul_bt_quant, so no forward-pass change is needed.
  {
    const std::string nm = Blk(il, "attn_qkv.weight");
    // W4d W4 (tenstorrent-27b-gdn-keepquant.md): with the V-head reorder
    // active the permutation acts on whole weight rows, so the PACKED
    // blocks survive a byte-level row reorder and the projection stays
    // keep-quant (kMatmulWeight) instead of expanding to bf16 — which
    // cost the P150 ~9.7 GiB of attn_qkv staging at 27B
    // (ISSUE-LOCAL-01M2AA4ZVD9EWJG5NNQD5DWZXS).
    // T21: VT_GDN_ROWPERM_KEEP_QUANT=1 gates the row-permuted keep-quant
    // path (opt-in, by value). Without it, the packed_reorder check below
    // is false and the projections expand to bf16 as before.
    vt::DType kq_dt = vt::DType::kF32;
    const bool packed_reorder =
        reorder && pol.keep_quant && rowperm_keep &&
        KeepQuantDType(g.Get(nm).ggml_type, &kq_dt) &&
        g.Get(nm).shape[1] % vt::BlockElems(kq_dt) == 0 &&
        DeviceKeepQuantSupported(kq_dt, pol.device);
    const GgufResidency r = pol.Route(
        g.Get(nm),
        packed_reorder ? GgufTensorRole::kMatmulWeight : proj_role);
    if (GgufResidencyKeepsBlockWeights(r)) {
      const GgufTensorInfo& ti = g.Get(nm);
      gdn.in_proj_qkv =
          OwnGgufKeptSlice(g, pol, ti, r, ti.shape[0], ti.shape[1], 0);
      if (packed_reorder) {
        // The kept bytes may be a BORROWED mmap view — the reorder
        // materializes them into an owned buffer before permuting.
        std::vector<uint8_t> tmp(gdn.in_proj_qkv.bytes.data(),
                                 gdn.in_proj_qkv.bytes.data() +
                                     gdn.in_proj_qkv.bytes.size());
        ReorderVRows<uint8_t>(tmp,
                              tmp.size() / ti.shape[0],
                              /*row_off=*/2 * key_dim, num_k, rpk, dv);
        gdn.in_proj_qkv.bytes = OwnedBytes(std::move(tmp));
      }
    } else {
      const GgufTensorInfo* t = nullptr;
      std::vector<uint16_t> dq = DqBf16(g, nm, &t);
      const int64_t out_dim = t->shape[0];
      const int64_t in_dim = t->shape[1];
      if (reorder) ReorderVRows(dq, in_dim, /*row_off=*/2 * key_dim, num_k, rpk, dv);
      gdn.in_proj_qkv = MakeGdnProj(dq, out_dim, in_dim, pol.gdn_expand_nk);
    }
  }
  // in_proj_z <- attn_gate [value_dim, H]; all rows are V.
  // T21: Same row-permutation keep-quant path as in_proj_qkv above.
  // Saves ~360 MB/tok of bf16 read amplification (24 Q4_K tensors × 2.9x).
  {
    const std::string nm = Blk(il, "attn_gate.weight");
    vt::DType z_dt = vt::DType::kF32;
    const bool packed_reorder_z =
        reorder && pol.keep_quant && rowperm_keep &&
        KeepQuantDType(g.Get(nm).ggml_type, &z_dt) &&
        g.Get(nm).shape[1] % vt::BlockElems(z_dt) == 0 &&
        DeviceKeepQuantSupported(z_dt, pol.device);
    const GgufResidency r = pol.Route(
        g.Get(nm),
        packed_reorder_z ? GgufTensorRole::kMatmulWeight : proj_role);
    if (GgufResidencyKeepsBlockWeights(r)) {
      const GgufTensorInfo& ti = g.Get(nm);
      gdn.in_proj_z =
          OwnGgufKeptSlice(g, pol, ti, r, ti.shape[0], ti.shape[1], 0);
      if (packed_reorder_z) {
        std::vector<uint8_t> tmp(gdn.in_proj_z.bytes.data(),
                                 gdn.in_proj_z.bytes.data() +
                                     gdn.in_proj_z.bytes.size());
        ReorderVRows<uint8_t>(tmp,
                              tmp.size() / ti.shape[0],
                              /*row_off=*/0, num_k, rpk, dv);
        gdn.in_proj_z.bytes = OwnedBytes(std::move(tmp));
      }
    } else {
      const GgufTensorInfo* t = nullptr;
      std::vector<uint16_t> dq = DqBf16(g, nm, &t);
      const int64_t out_dim = t->shape[0];
      const int64_t in_dim = t->shape[1];
      if (reorder) ReorderVRows(dq, in_dim, 0, num_k, rpk, dv);
      gdn.in_proj_z = MakeGdnProj(dq, out_dim, in_dim, pol.gdn_expand_nk);
    }
  }
  // in_proj_b <- ssm_beta, in_proj_a <- ssm_alpha [num_v, H]; rows are V heads.
  for (auto* pr : {&gdn.in_proj_b, &gdn.in_proj_a}) {
    const std::string nm =
        Blk(il, pr == &gdn.in_proj_b ? "ssm_beta.weight" : "ssm_alpha.weight");
    const GgufResidency r = pol.Route(g.Get(nm), proj_role);
    if (GgufResidencyKeepsBlockWeights(r)) {
      const GgufTensorInfo& ti = g.Get(nm);
      *pr = OwnGgufKeptSlice(g, pol, ti, r, ti.shape[0], ti.shape[1], 0);
      continue;
    }
    const GgufTensorInfo* t = nullptr;
    std::vector<uint16_t> dq = DqBf16(g, nm, &t);
    const int64_t out_dim = t->shape[0];
    const int64_t in_dim = t->shape[1];
    if (reorder) ReorderVRows(dq, in_dim, 0, num_k, rpk, 1);
    *pr = MakeGdnProj(dq, out_dim, in_dim, pol.gdn_expand_nk);
  }
  // conv1d <- ssm_conv1d [conv_dim, K]; only V channels reorder. NOT transposed.
  {
    RequireExpand(pol, g, Blk(il, "ssm_conv1d.weight"),
                  GgufTensorRole::kConvWeight);
    const GgufTensorInfo* t = nullptr;
    std::vector<uint16_t> dq = DqBf16(g, Blk(il, "ssm_conv1d.weight"), &t);
    VT_CHECK(t->shape.size() == 2, "qwen3_5 gguf: ssm_conv1d must be 2-D");
    const int64_t conv_dim = t->shape[0];
    const int64_t kk = t->shape[1];
    if (reorder) ReorderVRows(dq, kk, /*row_off=*/2 * key_dim, num_k, rpk, dv);
    gdn.conv1d_weight = MakeOwned(vt::DType::kBF16, {conv_dim, kk});
    std::memcpy(gdn.conv1d_weight.bytes.data(), dq.data(),
                dq.size() * sizeof(uint16_t));
  }
  // out_proj <- ssm_out [H, value_dim]; reorder V columns, then transpose.
  // The COLUMN reorder cuts across block boundaries, so when it is active this
  // tensor is kTransformedWeight and must expand — UNLESS T25
  // (VT_GDN_COLPERM_KEEP_QUANT=1) keeps the tiled-order Q5_K weight and
  // permutes the GEMV input at runtime instead.
  {
    const std::string nm = Blk(il, "ssm_out.weight");
    const GgufResidency r = pol.Route(g.Get(nm), colperm_role);
    if (r == GgufResidency::kKeepQuant && colperm_keep) {
      // T25: keep Q5_K in tiled order (no ReorderVCols). The forward pass
      // permutes the 4096-element activation from grouped→tiled before the
      // K-quant GEMV, saving ~4x weight bandwidth.
      OwnedTensor qk =
          OwnGgufQuantBlocks(g.Get(nm), g.Get(nm).shape[0], g.Get(nm).shape[1],
                             0, /*mmap_src=*/nullptr);
      gdn.out_proj = std::move(qk);
      gdn.out_proj_tiled = true;
    } else if (r != GgufResidency::kExpandBf16) {
      const GgufTensorInfo& ti = g.Get(nm);
      gdn.out_proj =
          OwnGgufKeptSlice(g, pol, ti, r, ti.shape[0], ti.shape[1], 0);
    } else {
      const GgufTensorInfo* t = nullptr;
      std::vector<uint16_t> dq = DqBf16(g, nm, &t);
      const int64_t out_dim = t->shape[0];
      const int64_t in_dim = t->shape[1];
      if (reorder) ReorderVCols(dq, out_dim, in_dim, num_k, rpk, dv);
      gdn.out_proj = MakeGdnProj(dq, out_dim, in_dim, pol.gdn_expand_nk);
    }
  }
  // a_log <- ssm_a = -exp(A_log): recover A_log = log(-value). f32 [num_v].
  {
    RequireExpand(pol, g, Blk(il, "ssm_a"),
                  GgufTensorRole::kTransformedWeight);
    const GgufTensorInfo* t = nullptr;
    std::vector<float> dq = DqF32(g, Blk(il, "ssm_a"), &t);
    if (reorder) ReorderVRows(dq, 1, 0, num_k, rpk, 1);
    gdn.a_log = MakeOwned(vt::DType::kF32, {static_cast<int64_t>(dq.size())});
    auto* dst = reinterpret_cast<float*>(gdn.a_log.bytes.data());
    for (size_t i = 0; i < dq.size(); ++i) dst[i] = std::log(-dq[i]);
  }
  // dt_bias <- ssm_dt.bias (unchanged value). f32 [num_v].
  {
    RequireExpand(pol, g, Blk(il, "ssm_dt.bias"), GgufTensorRole::kVector);
    const GgufTensorInfo* t = nullptr;
    std::vector<float> dq = DqF32(g, Blk(il, "ssm_dt.bias"), &t);
    if (reorder) ReorderVRows(dq, 1, 0, num_k, rpk, 1);
    gdn.dt_bias = MakeOwned(vt::DType::kF32, {static_cast<int64_t>(dq.size())});
    std::memcpy(gdn.dt_bias.bytes.data(), dq.data(), dq.size() * sizeof(float));
  }
  // norm_weight <- ssm_norm: raw HF (convert does NOT add 1 here). bf16 [Dv].
  RequireExpand(pol, g, Blk(il, "ssm_norm.weight"), GgufTensorRole::kVector);
  gdn.norm_weight = OwnBf16(g, Blk(il, "ssm_norm.weight"), {dv});
  return gdn;
}

// One 2-D matmul weight into EITHER the fp4-resident field or the bf16 one,
// with the other left EMPTY — the invariant the forward's `!*.Empty()` dispatch
// relies on. This is the whole wiring of the C column at a call site.
void LoadMatmulWeightOrNvfp4(const GgufFile& g, const std::string& name,
                             const GgufLoadPolicy& pol, OwnedTensor* bf16,
                             Nvfp4Weight* fp4) {
  const GgufTensorInfo& t = g.Get(name);
  if (PeekRoute(pol, t, GgufTensorRole::kMatmulWeight) ==
      GgufResidency::kNvfp4Fp4) {
    (void)pol.Route(t, GgufTensorRole::kMatmulWeight);  // its ONE audit event
    *fp4 = OwnGgufNvfp4(g, name, pol);
    return;
  }
  *bf16 = OwnMatmulWeight(g, name, pol);  // routes (and audits) once itself
}

FullAttnLayerWeights LoadAttnGguf(const GgufFile& g, int64_t il,
                                  const GgufLoadPolicy& pol) {
  FullAttnLayerWeights a;
  LoadMatmulWeightOrNvfp4(g, Blk(il, "attn_q.weight"), pol, &a.q_proj,
                          &a.q_proj_fp4);
  LoadMatmulWeightOrNvfp4(g, Blk(il, "attn_k.weight"), pol, &a.k_proj,
                          &a.k_proj_fp4);
  LoadMatmulWeightOrNvfp4(g, Blk(il, "attn_v.weight"), pol, &a.v_proj,
                          &a.v_proj_fp4);
  LoadMatmulWeightOrNvfp4(g, Blk(il, "attn_output.weight"), pol, &a.o_proj,
                          &a.o_proj_fp4);
  // (w - 1) rewrite: a VALUE transform, so never keep-quant.
  RequireExpand(pol, g, Blk(il, "attn_q_norm.weight"),
                GgufTensorRole::kTransformedWeight);
  a.q_norm = OwnNormMinus1(g, Blk(il, "attn_q_norm.weight"));
  RequireExpand(pol, g, Blk(il, "attn_k_norm.weight"),
                GgufTensorRole::kTransformedWeight);
  a.k_norm = OwnNormMinus1(g, Blk(il, "attn_k_norm.weight"));
  return a;
}

// Split a stacked expert tensor "blk.N.<stem>" (GGUF torch shape [E, out, in])
// into E owned bf16 [in, out] (transposed to Matmul-B layout).
std::vector<OwnedTensor> LoadExpertsT(const GgufFile& g, int64_t il,
                                      const std::string& stem,
                                      int64_t num_experts,
                                      const GgufLoadPolicy& pol) {
  const std::string name = Blk(il, stem);
  const GgufTensorInfo& ti = g.Get(name);
  VT_CHECK(ti.shape.size() == 3 && ti.shape[0] == num_experts,
           "qwen3_5 gguf: expected [E,out,in] expert tensor " + stem);
  const int64_t out_dim = ti.shape[1];
  const int64_t in_dim = ti.shape[2];

  std::vector<OwnedTensor> experts;
  experts.reserve(static_cast<size_t>(num_experts));

  const GgufResidency r = pol.Route(ti, GgufTensorRole::kStackedExpertWeight);
  if (GgufResidencyKeepsBlockWeights(r)) {
    // keep-quant blocks OR keep-f16. Each expert occupies `out_dim` WHOLE rows,
    // i.e. a whole number of blocks (and any whole number of f16 rows), so the
    // split is a byte range and no block is ever cut.
    for (int64_t e = 0; e < num_experts; ++e) {
      experts.push_back(OwnGgufKeptSlice(g, pol, ti, r, out_dim, in_dim,
                                         /*row_offset=*/e * out_dim));
    }
    return experts;
  }

  const GgufTensorInfo* t = nullptr;
  std::vector<uint16_t> dq = DqBf16(g, name, &t);
  const int64_t per = out_dim * in_dim;
  for (int64_t e = 0; e < num_experts; ++e) {
    OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
    TransposeBf16(dq.data() + e * per, out_dim, in_dim,
                  reinterpret_cast<uint16_t*>(o.bytes.data()));
    experts.push_back(std::move(o));
  }
  return experts;
}

// A3: load the WHOLE keep-quant expert tower as ONE stacked [E*out, in] block tensor
// (row_offset=0, all E*out rows) — the [E*N,K] layout vt::MatmulBTQuantGrouped consumes.
// The keep-quant arm of LoadExpertsT slices per-expert WITHOUT transpose, so the whole
// tower IS this layout; loading it whole REPLACES the E per-expert copies (memory-neutral,
// no duplicate ~30GB on the 35B). Called only when the tower routes keep-quant.
OwnedTensor LoadExpertsStackedKq(const GgufFile& g, int64_t il, const std::string& stem,
                                 int64_t num_experts, const GgufLoadPolicy& pol) {
  const std::string name = Blk(il, stem);
  const GgufTensorInfo& ti = g.Get(name);
  VT_CHECK(ti.shape.size() == 3 && ti.shape[0] == num_experts,
           "qwen3_5 gguf A3: expected [E,out,in] expert tensor " + stem);
  const int64_t out_dim = ti.shape[1];
  const int64_t in_dim = ti.shape[2];
  const GgufResidency r =
      pol.Route(ti, GgufTensorRole::kStackedExpertWeight);  // ONE audit event
  VT_CHECK(GgufResidencyKeepsBlockWeights(r),
           "qwen3_5 gguf A3: LoadExpertsStackedKq for a non-keep residency on " + name);
  // Each expert = out_dim WHOLE rows, so E*out_dim rows never cuts a block/f16 row.
  return OwnGgufKeptSlice(g, pol, ti, r, num_experts * out_dim, in_dim,
                          /*row_offset=*/0);
}

// Route the stacked routed-expert tower to EXACTLY ONE of: fp4-resident (*fp4), the
// bf16-EXPAND per-expert transposed vector (*bf16), or the A3 keep-quant WHOLE stacked
// tensor (*kq). The fp4 arm keeps the GGUF's own [N=out,K=in] orientation (what
// vt::MatmulNvfp4 reads). The MoE forward dispatch keys on which one is non-empty.
void LoadExpertsOrNvfp4(const GgufFile& g, int64_t il, const std::string& stem,
                        int64_t num_experts, const GgufLoadPolicy& pol,
                        std::vector<OwnedTensor>* bf16, std::vector<Nvfp4Weight>* fp4,
                        OwnedTensor* kq) {
  const std::string name = Blk(il, stem);
  const GgufTensorInfo& t = g.Get(name);
  const GgufResidency peek =
      PeekRoute(pol, t, GgufTensorRole::kStackedExpertWeight);
  if (peek == GgufResidency::kNvfp4Fp4) {
    (void)pol.Route(t, GgufTensorRole::kStackedExpertWeight);  // ONE audit event
    *fp4 = OwnGgufNvfp4Experts(g, name, num_experts, pol);
    return;
  }
  if (GgufResidencyExpandsBf16(peek)) {
    *bf16 = LoadExpertsT(g, il, stem, num_experts, pol);  // per-expert transposed; audits
    return;
  }
  *kq = LoadExpertsStackedKq(g, il, stem, num_experts, pol);  // keep-quant whole; audits
}

MoeBlockWeights LoadMoeGguf(const GgufFile& g, int64_t il, const HfConfig& c,
                            const GgufLoadPolicy& pol) {
  MoeBlockWeights m;
  // The router gate is F32 in every published file and has no fp4 field.
  m.router_gate =
      OwnMatmulWeight(g, Blk(il, "ffn_gate_inp.weight"), NoNvfp4(pol));
  // shared gate: 1-D [H] in GGUF -> [H, 1] (matches the safetensors [H,1]).
  RequireExpand(pol, g, Blk(il, "ffn_gate_inp_shexp.weight"),
                GgufTensorRole::kVector);
  m.shared_gate =
      OwnBf16(g, Blk(il, "ffn_gate_inp_shexp.weight"), {c.hidden_size, 1});
  // Stacked routed experts: fp4-resident per expert when routed there, else the
  // historical transposed bf16 slabs. Exactly one set is filled, which is the
  // invariant the MoE forward's `!expert_*_fp4.empty()` dispatch relies on.
  LoadExpertsOrNvfp4(g, il, "ffn_gate_exps.weight", c.num_experts, pol,
                     &m.expert_gate, &m.expert_gate_fp4, &m.expert_gate_kq);
  LoadExpertsOrNvfp4(g, il, "ffn_up_exps.weight", c.num_experts, pol,
                     &m.expert_up, &m.expert_up_fp4, &m.expert_up_kq);
  LoadExpertsOrNvfp4(g, il, "ffn_down_exps.weight", c.num_experts, pol,
                     &m.expert_down, &m.expert_down_fp4, &m.expert_down_kq);
  LoadMatmulWeightOrNvfp4(g, Blk(il, "ffn_gate_shexp.weight"), pol,
                          &m.shared_gate_proj, &m.shared_gate_proj_fp4);
  LoadMatmulWeightOrNvfp4(g, Blk(il, "ffn_up_shexp.weight"), pol,
                          &m.shared_up_proj, &m.shared_up_proj_fp4);
  LoadMatmulWeightOrNvfp4(g, Blk(il, "ffn_down_shexp.weight"), pol,
                          &m.shared_down_proj, &m.shared_down_proj_fp4);
  return m;
}

}  // namespace

Qwen3_5MoeWeights LoadQwen3_5MoeFromGguf(const GgufFile& gguf,
                                         const HfConfig& config,
                                         const GgufLoadPolicy* policy) {
  // Null policy => the process environment, whose defaults reproduce the
  // historical all-bf16 expansion byte for byte.
  // A null `policy` means NO ENGINE RESOLVED A DEVICE: every production GGUF
  // entry point now builds one from `ModelSource::device` in its registry hook.
  // `kCPU` is STATED rather than probed — see `GgufLoadPolicy::FromEnv`.
  const GgufLoadPolicy env_policy = GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU);
  const GgufLoadPolicy& pol = policy != nullptr ? *policy : env_policy;
  // Read-once file pages are dropped as the load goes, but only when the kept
  // weights are the ones staying resident (mmap residency). With the copy arm or
  // the oracle there is nothing in place to protect and the historical
  // page-residency behavior is preserved exactly.
  const GgufPageReleaseScope page_release(
      gguf, pol.mmap_residency && EnvReleaseExpandedPages());
  VT_CHECK(config.num_hidden_layers > 0 &&
               static_cast<int64_t>(config.layer_types.size()) ==
                   config.num_hidden_layers,
           "qwen3_5 gguf: layer_types size must equal num_hidden_layers");
  VT_CHECK(config.num_experts > 0,
           "qwen3_5 gguf: num_experts must be > 0 for the MoE model");

  Qwen3_5MoeWeights w;
  // embed_tokens [vocab, H] (NOT transposed): a GATHER table, never a GEMM.
  // lm_head [H, vocab] (+T). Tied-embedding GGUFs omit output.weight; then the
  // head IS token_embd, as llama.cpp has it (TENSOR_DUPLICATED). token_embd is
  // routed TWICE when tied — once as the gather table, once as this GEMM weight
  // — which is the correct answer for both uses; L5 makes the two SHARE one
  // expansion where their byte images coincide. See LoadEmbedAndHead.
  LoadEmbedAndHead(gguf, pol, &w.embed_tokens, &w.lm_head);
  RequireExpand(pol, gguf, "output_norm.weight",
                GgufTensorRole::kTransformedWeight);
  w.final_norm = OwnNormMinus1(gguf, "output_norm.weight");

  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t il = 0; il < config.num_hidden_layers; ++il) {
    TtAllocTraceLayer("layer-moe", il);
    Qwen3_5MoeLayerWeights layer;
    RequireExpand(pol, gguf, Blk(il, "attn_norm.weight"),
                  GgufTensorRole::kTransformedWeight);
    layer.input_layernorm = OwnNormMinus1(gguf, Blk(il, "attn_norm.weight"));
    RequireExpand(pol, gguf, Blk(il, "post_attention_norm.weight"),
                  GgufTensorRole::kTransformedWeight);
    layer.post_attention_layernorm =
        OwnNormMinus1(gguf, Blk(il, "post_attention_norm.weight"));
    const std::string& lt = config.layer_types[static_cast<size_t>(il)];
    if (lt == "linear_attention") {
      layer.is_linear_attention = true;
      layer.gdn = LoadGdnGguf(gguf, il, config, pol);
    } else if (lt == "full_attention") {
      layer.is_linear_attention = false;
      layer.attn = LoadAttnGguf(gguf, il, pol);
    } else {
      VT_CHECK(false, "qwen3_5 gguf: unknown layer_type " + lt);
    }
    layer.moe = LoadMoeGguf(gguf, il, config, pol);
    w.layers.push_back(std::move(layer));
  }
  return w;
}

// ─── MTP head from GGUF (SPEC-MTP-GGUF) ──────────────────────────────────────
//
// llama.cpp's Qwen3.5 converter (conversion/qwen.py `_Qwen35MtpMixin`) folds the
// HF `mtp.*` block into the ordinary block list: it extends `block_count` by
// `mtp_num_hidden_layers`, emits the `nextn_predict_layers` key, and remaps the
// four scalar tensors onto block L (= the ORIGINAL num_hidden_layers) under the
// DeepSeek-style `nextn` names, with `mtp.layers.{i}.*` landing on block L+i as
// ordinary layer tensors:
//
//   mtp.fc                     -> blk.{L}.nextn.eh_proj
//   mtp.pre_fc_norm_embedding  -> blk.{L}.nextn.enorm
//   mtp.pre_fc_norm_hidden     -> blk.{L}.nextn.hnorm
//   mtp.norm                   -> blk.{L}.nextn.shared_head_norm
//   mtp.layers.{i}.<rest>      -> blk.{L+i}.<rest>
//
// This is deliberately NOT routed through the TensorResolver seam that
// LoadQwen3_5MTP's safetensors entry point uses. That seam hands out raw tensor
// views, but the GGUF conventions live one level up in THIS file's helpers: norm
// weights are stored as (w + 1) and must be un-shifted (OwnNormMinus1), matmul
// weights carry the file's quantization and residency routing
// (OwnMatmulWeight), and shapes arrive in torch [N, K] order. Reusing those
// helpers is what makes the head obey exactly the same conventions as the trunk
// it speculates for; going through the resolver would mean re-deriving all three
// in a second place.
//
// The head SHARES the target's embed_tokens and lm_head (Qwen3.5 ships no
// dedicated ones, which is why the converter emits no nextn.embed_tokens /
// nextn.shared_head_head for it), so nothing here loads them.
Qwen3_5MTPWeights LoadQwen3_5MTPFromGguf(const GgufFile& gguf,
                                         const HfConfig& config,
                                         Qwen3_5MTPKind kind,
                                         const GgufLoadPolicy& pol) {
  const int64_t n_mtp = NumMtpLayers(config);
  VT_CHECK(n_mtp > 0,
           "qwen3_5 gguf MTP: mtp_num_hidden_layers must be > 0 (is the GGUF "
           "missing <arch>.nextn_predict_layers?)");
  VT_CHECK(!UsesDedicatedEmbeddings(config),
           "qwen3_5 gguf MTP: dedicated MTP embeddings are not supported");
  // Block index of the FIRST head block. HfConfigFromGguf already subtracted the
  // head depth from block_count, so num_hidden_layers is exactly that index.
  const int64_t L = config.num_hidden_layers;

  Qwen3_5MTPWeights out;
  out.kind = kind;

  // fc is kept in raw [N=out, K=in] (LoadBf16RawNK on the safetensors path), and
  // GgufTensorInfo::shape is already torch [N, K], so this is the verbatim path
  // rather than the transposing one.
  {
    const std::string name = Blk(L, "nextn.eh_proj.weight");
    RequireExpand(pol, gguf, name, GgufTensorRole::kTransformedWeight);
    out.fc = OwnBf16(gguf, name, gguf.Get(name).shape);
    // The draft forward requires fc in RAW [N, K] with the nk flag SET
    // (qwen3_5.cpp "fc must be raw bf16 [H,2H]"); the safetensors path gets it
    // from LoadBf16RawNK. GGUF already stores this orientation, so only the
    // flag is missing - and without it the shape check passes while the forward
    // refuses, which is exactly how the G4 gate caught this.
    out.fc.nk = true;
  }
  for (const auto& [field, stem] :
       std::initializer_list<std::pair<OwnedTensor*, const char*>>{
           {&out.pre_fc_norm_embedding, "nextn.enorm.weight"},
           {&out.pre_fc_norm_hidden, "nextn.hnorm.weight"},
           {&out.final_norm, "nextn.shared_head_norm.weight"}}) {
    const std::string name = Blk(L, stem);
    RequireExpand(pol, gguf, name, GgufTensorRole::kTransformedWeight);
    *field = OwnNormMinus1(gguf, name);
  }

  for (int64_t i = 0; i < n_mtp; ++i) {
    const int64_t il = L + i;
    RequireExpand(pol, gguf, Blk(il, "attn_norm.weight"),
                  GgufTensorRole::kTransformedWeight);
    RequireExpand(pol, gguf, Blk(il, "post_attention_norm.weight"),
                  GgufTensorRole::kTransformedWeight);
    OwnedTensor input_ln = OwnNormMinus1(gguf, Blk(il, "attn_norm.weight"));
    OwnedTensor post_ln =
        OwnNormMinus1(gguf, Blk(il, "post_attention_norm.weight"));
    // The MTP block is always full-attention (it is a transformer block over the
    // fused [embedding; hidden] state), never GDN - so it is not looked up in
    // config.layer_types, which only covers the trunk.
    FullAttnLayerWeights attn = LoadAttnGguf(gguf, il, pol);

    if (kind == Qwen3_5MTPKind::kDense) {
      Qwen3_5DenseLayerWeights layer;
      layer.is_linear_attention = false;
      layer.input_layernorm = std::move(input_ln);
      layer.post_attention_layernorm = std::move(post_ln);
      layer.attn = std::move(attn);
      layer.mlp.gate_proj = OwnMatmulWeight(gguf, Blk(il, "ffn_gate.weight"), pol);
      layer.mlp.up_proj = OwnMatmulWeight(gguf, Blk(il, "ffn_up.weight"), pol);
      layer.mlp.down_proj = OwnMatmulWeight(gguf, Blk(il, "ffn_down.weight"), pol);
      out.dense_layers.push_back(std::move(layer));
    } else {
      Qwen3_5MoeLayerWeights layer;
      layer.is_linear_attention = false;
      layer.input_layernorm = std::move(input_ln);
      layer.post_attention_layernorm = std::move(post_ln);
      layer.attn = std::move(attn);
      layer.moe = LoadMoeGguf(gguf, il, config, pol);
      out.moe_layers.push_back(std::move(layer));
    }
  }
  return out;
}

Qwen3_5DenseWeights LoadQwen3_5DenseFromGguf(const GgufFile& gguf,
                                             const HfConfig& config,
                                             const GgufLoadPolicy* policy) {
  // Null policy => the process environment, whose defaults reproduce the
  // historical all-bf16 expansion byte for byte.
  // A null `policy` means NO ENGINE RESOLVED A DEVICE: every production GGUF
  // entry point now builds one from `ModelSource::device` in its registry hook.
  // `kCPU` is STATED rather than probed — see `GgufLoadPolicy::FromEnv`.
  const GgufLoadPolicy env_policy = GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU);
  const GgufLoadPolicy& pol = policy != nullptr ? *policy : env_policy;
  // Read-once file pages are dropped as the load goes, but only when the kept
  // weights are the ones staying resident (mmap residency). With the copy arm or
  // the oracle there is nothing in place to protect and the historical
  // page-residency behavior is preserved exactly.
  const GgufPageReleaseScope page_release(
      gguf, pol.mmap_residency && EnvReleaseExpandedPages());
  VT_CHECK(config.num_hidden_layers > 0 &&
               static_cast<int64_t>(config.layer_types.size()) ==
                   config.num_hidden_layers,
           "qwen3_5 gguf: layer_types size must equal num_hidden_layers");
  VT_CHECK(config.num_experts == 0,
           "qwen3_5 gguf: num_experts must be 0 for the dense model");

  Qwen3_5DenseWeights w;
  TtAllocTrace("load/embed-and-head/pre");
  // Tied-embedding GGUFs (the 2B bench file) omit output.weight; the head is
  // then token_embd itself, as llama.cpp has it (TENSOR_DUPLICATED), and L5
  // lets the two SHARE one expansion. See LoadEmbedAndHead.
  LoadEmbedAndHead(gguf, pol, &w.embed_tokens, &w.lm_head);
  RequireExpand(pol, gguf, "output_norm.weight",
                GgufTensorRole::kTransformedWeight);
  w.final_norm = OwnNormMinus1(gguf, "output_norm.weight");
  TtAllocTrace("load/embed-and-head/post");

  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t il = 0; il < config.num_hidden_layers; ++il) {
    TtAllocTraceLayer("layer-dense", il);
    Qwen3_5DenseLayerWeights layer;
    RequireExpand(pol, gguf, Blk(il, "attn_norm.weight"),
                  GgufTensorRole::kTransformedWeight);
    layer.input_layernorm = OwnNormMinus1(gguf, Blk(il, "attn_norm.weight"));
    RequireExpand(pol, gguf, Blk(il, "post_attention_norm.weight"),
                  GgufTensorRole::kTransformedWeight);
    layer.post_attention_layernorm =
        OwnNormMinus1(gguf, Blk(il, "post_attention_norm.weight"));
    const std::string& lt = config.layer_types[static_cast<size_t>(il)];
    if (lt == "linear_attention") {
      layer.is_linear_attention = true;
      layer.gdn = LoadGdnGguf(gguf, il, config, pol);
    } else if (lt == "full_attention") {
      layer.is_linear_attention = false;
      layer.attn = LoadAttnGguf(gguf, il, pol);
    } else {
      VT_CHECK(false, "qwen3_5 gguf: unknown layer_type " + lt);
    }
    // Dense SwiGLU MLP (bf16 fields; the fp4 variants stay empty).
    LoadMatmulWeightOrNvfp4(gguf, Blk(il, "ffn_gate.weight"), pol,
                            &layer.mlp.gate_proj, &layer.mlp.gate_proj_fp4);
    LoadMatmulWeightOrNvfp4(gguf, Blk(il, "ffn_up.weight"), pol,
                            &layer.mlp.up_proj, &layer.mlp.up_proj_fp4);
    LoadMatmulWeightOrNvfp4(gguf, Blk(il, "ffn_down.weight"), pol,
                            &layer.mlp.down_proj, &layer.mlp.down_proj_fp4);
    w.layers.push_back(std::move(layer));
  }
  return w;
}

// ─── Tensor accounting (QUANT-QWEN38-27B-GGUF-ARM, issue #821) ──────────────

bool IsQwen3_5Gguf(const GgufFile& gguf) {
  const GgufValue* arch_v = gguf.FindKv("general.architecture");
  if (arch_v == nullptr || arch_v->TypeId() != kGgufString) return false;
  const std::string& arch = std::get<std::string>(arch_v->v);
  return arch == "qwen35" || arch == "qwen35moe" || arch == "qwen3next";
}

namespace {

// The MTP head depth this CHECKPOINT declares, which is ZERO when it declares
// none.
//
// `NumMtpLayers` answers 1 for an absent key, because it answers "how deep is
// the head" for a spec resolver that already knows there is one. Accounting is
// asking a different question — "does this file ship one at all" — and reusing
// the first answer would enumerate a drafter block on every head-less GGUF and
// then report its eleven tensors as missing. `HfConfigFromGguf` publishes the
// key only when `<arch>.nextn_predict_layers` is set and positive, so its
// PRESENCE is the checkpoint's own statement.
int64_t DeclaredMtpDepth(const HfConfig& config) {
  const nlohmann::json& text =
      (config.raw.is_object() && config.raw.contains("text_config") &&
       config.raw.at("text_config").is_object())
          ? config.raw.at("text_config")
          : config.raw;
  if (!text.is_object() || !text.contains("mtp_num_hidden_layers")) return 0;
  const int64_t n = NumMtpLayers(config);
  return n > 0 ? n : 0;
}

// One block's names, exactly as the loaders above read them: the two norms,
// the attention family the layer's KIND selects, and the MLP the architecture
// selects.
void AppendBlockTensors(int64_t il, bool linear_attention, bool moe,
                        std::vector<std::string>* out) {
  out->push_back(Blk(il, "attn_norm.weight"));
  out->push_back(Blk(il, "post_attention_norm.weight"));
  if (linear_attention) {
    // LoadGdnGguf, in its own order.
    for (const char* stem :
         {"attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight",
          "ssm_alpha.weight", "ssm_conv1d.weight", "ssm_out.weight", "ssm_a",
          "ssm_dt.bias", "ssm_norm.weight"}) {
      out->push_back(Blk(il, stem));
    }
  } else {
    // LoadAttnGguf.
    for (const char* stem :
         {"attn_q.weight", "attn_k.weight", "attn_v.weight",
          "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight"}) {
      out->push_back(Blk(il, stem));
    }
  }
  if (moe) {
    // LoadMoeGguf.
    for (const char* stem :
         {"ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight",
          "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
          "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
          "ffn_down_shexp.weight"}) {
      out->push_back(Blk(il, stem));
    }
  } else {
    for (const char* stem :
         {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}) {
      out->push_back(Blk(il, stem));
    }
  }
}

}  // namespace

std::vector<std::string> Qwen3_5GgufExpectedTensors(const HfConfig& config,
                                                    bool has_output_weight) {
  const bool moe = config.num_experts > 0;
  std::vector<std::string> out;

  // LoadEmbedAndHead + the trunk's final norm. `output.weight` is absent on a
  // tied-embedding export (llama.cpp TENSOR_DUPLICATED), where the head IS
  // token_embd, so it is expected only when the file ships it.
  out.emplace_back("token_embd.weight");
  if (has_output_weight) out.emplace_back("output.weight");
  out.emplace_back("output_norm.weight");

  for (int64_t il = 0; il < config.num_hidden_layers; ++il) {
    const bool linear =
        il < static_cast<int64_t>(config.layer_types.size()) &&
        config.layer_types[static_cast<size_t>(il)] == "linear_attention";
    AppendBlockTensors(il, linear, moe, &out);
  }

  // LoadQwen3_5MTPFromGguf. llama.cpp's converter folds the head into the block
  // list at `blk.{num_hidden_layers + i}`, with the four scalar `nextn.*`
  // tensors on the FIRST head block. The head block is always FULL ATTENTION —
  // it is a transformer block over the fused [embedding; hidden] state — so it
  // is not looked up in `layer_types`, which covers only the trunk.
  const int64_t n_mtp = DeclaredMtpDepth(config);
  const int64_t L = config.num_hidden_layers;
  if (n_mtp > 0) {
    for (const char* stem :
         {"nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight",
          "nextn.shared_head_norm.weight"}) {
      out.push_back(Blk(L, stem));
    }
    for (int64_t i = 0; i < n_mtp; ++i) {
      AppendBlockTensors(L + i, /*linear_attention=*/false, moe, &out);
    }
  }
  return out;
}

Qwen3_5GgufAccounting Qwen3_5GgufAccountTensors(
    const HfConfig& config, const std::vector<std::string>& present) {
  const std::set<std::string> have(present.begin(), present.end());
  const std::vector<std::string> want =
      Qwen3_5GgufExpectedTensors(config, have.count("output.weight") != 0);
  const std::set<std::string> wanted(want.begin(), want.end());

  Qwen3_5GgufAccounting out;
  out.enumerated = static_cast<int64_t>(wanted.size());
  out.present = static_cast<int64_t>(have.size());
  for (const std::string& name : wanted) {
    if (have.count(name) == 0) out.missing.push_back(name);
  }
  for (const std::string& name : have) {
    if (wanted.count(name) != 0) continue;
    // An NVFP4 sidecar belongs to the weight it scales, and whether it exists
    // is a property of that weight's ENCODING rather than of the architecture,
    // so it is accounted by its stem instead of being enumerated.
    bool sidecar = false;
    for (const char* suffix : {".scale", ".input_scale"}) {
      const std::string s(suffix);
      if (name.size() <= s.size()) continue;
      if (name.compare(name.size() - s.size(), s.size(), s) != 0) continue;
      if (wanted.count(name.substr(0, name.size() - s.size()) + ".weight") != 0) {
        sidecar = true;
        break;
      }
    }
    if (!sidecar) out.unaccounted.push_back(name);
  }
  return out;
}

void RefuseUnaccountedQwen3_5Gguf(const GgufFile& gguf,
                                  const HfConfig& config) {
  std::vector<std::string> present;
  present.reserve(gguf.Tensors().size());
  for (const GgufTensorInfo& t : gguf.Tensors()) present.push_back(t.name);
  const Qwen3_5GgufAccounting acct = Qwen3_5GgufAccountTensors(config, present);
  if (acct.unaccounted.empty()) return;

  // Name them, up to a bound: a 65th-block defect produces ten and a
  // wholly-misread file produces hundreds, and a message that scrolls is a
  // message nobody reads.
  constexpr size_t kMaxNamed = 12;
  std::string names;
  for (size_t i = 0; i < acct.unaccounted.size() && i < kMaxNamed; ++i) {
    names += (i == 0 ? "" : ", ") + acct.unaccounted[i];
  }
  if (acct.unaccounted.size() > kMaxNamed) {
    names += ", ... (" + std::to_string(acct.unaccounted.size() - kMaxNamed) +
             " more)";
  }
  VT_CHECK(false,
           "qwen3_5 gguf: this file carries " +
               std::to_string(acct.unaccounted.size()) +
               " tensor(s) that NOTHING in this loader reads, out of " +
               std::to_string(acct.present) + " present against " +
               std::to_string(acct.enumerated) + " enumerated for " +
               std::to_string(config.num_hidden_layers) + " trunk layers + " +
               std::to_string(DeclaredMtpDepth(config)) +
               " MTP block(s): " + names +
               ". Loading it would leave those weights out of the graph "
               "SILENTLY. Check <arch>.block_count against "
               "<arch>.nextn_predict_layers, and see "
               ".agents/specs/qwen38-27b-quant-arms.md");
}

Qwen3_5GgufMtpHeadSkip Qwen3_5GgufMtpHeadSkipTensors(const GgufFile& gguf,
                                                     const HfConfig& config) {
  Qwen3_5GgufMtpHeadSkip out;
  const int64_t n_mtp = DeclaredMtpDepth(config);
  if (n_mtp <= 0) return out;

  // The same names the expected-tensor enumeration lists for the head block:
  // the four scalar `nextn.*` tensors on the FIRST head block, then one
  // ordinary (always full-attention) block per head layer. Kept in lockstep
  // with `Qwen3_5GgufExpectedTensors` — its accounting is what refuses a name
  // this list forgets, so the two cannot drift silently.
  std::vector<std::string> want;
  const int64_t L = config.num_hidden_layers;
  for (const char* stem :
       {"nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight",
        "nextn.shared_head_norm.weight"}) {
    want.push_back(Blk(L, stem));
  }
  for (int64_t i = 0; i < n_mtp; ++i) {
    AppendBlockTensors(L + i, /*linear_attention=*/false,
                       config.num_experts > 0, &want);
  }

  // Intersect with the file, in file order, and total the bytes: the numbers
  // the loud skip line reports.
  std::set<std::string> want_set(want.begin(), want.end());
  for (const GgufTensorInfo& t : gguf.Tensors()) {
    if (want_set.count(t.name) == 0) continue;
    out.names.push_back(t.name);
    out.bytes += static_cast<int64_t>(t.nbytes);
  }
  out.tensor_count = static_cast<int64_t>(out.names.size());
  out.present = out.tensor_count > 0;
  return out;
}

void LogQwen3_5GgufMtpHeadSkip(const GgufFile& gguf, const HfConfig& config) {
  const Qwen3_5GgufMtpHeadSkip skip =
      Qwen3_5GgufMtpHeadSkipTensors(gguf, config);
  if (!skip.present) return;
  std::string names;
  for (size_t i = 0; i < skip.names.size(); ++i) {
    names += (i == 0 ? "" : ", ") + skip.names[i];
  }
  std::cerr << "engine: qwen3.5 gguf: SKIPPING the MTP drafter head — "
            << skip.tensor_count << " tensor(s), " << skip.bytes
            << " B, none of them read by this trunk-only load: " << names
            << ". They are the multi-token-prediction head at blk."
            << config.num_hidden_layers << " (block_count "
            << (config.num_hidden_layers + DeclaredMtpDepth(config))
            << " = " << config.num_hidden_layers << " trunk + "
            << DeclaredMtpDepth(config)
            << " head); they load only when speculative decoding is "
               "configured (speculative-config method \"mtp\"). Denominator "
               "parity: the pinned llama.cpp b10451 oracle IGNORES these same "
               "tensors too (64 trunk layers, no MTP head; "
               ".agents/oracles/llama-cpp.md), so a gate against it is "
               "matched work only with this skip loud."
            << std::endl;
}


void ReorderVPackedForTest(std::vector<uint8_t>& packed, int64_t row_bytes,
                           int64_t row_off, int64_t num_k,
                           int64_t num_v_per_k, int64_t head_rows) {
  ReorderVRows<uint8_t>(packed, row_bytes, row_off, num_k, num_v_per_k,
                        head_rows);
}

}  // namespace vllm
