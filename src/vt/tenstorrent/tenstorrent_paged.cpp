// Tenstorrent paged-attention / paged-KV ops — split stage 4 of
// tenstorrent_ops.cpp (.agents/specs/tenstorrent-ops-split.md,
// ISSUE-LOCAL-01M2M2PZC0E998B88C9XKJT8JJ). Mechanical move from
// tenstorrent_ops.cpp: the PagedKvShadow mirror + device push helpers,
// kReshapeAndCache / kPagedAttention (CPU-oracle f32 softmax over the
// host-resident paged cache), and the WarmPagedKvShadow / WarmRacIdx /
// WarmPaMeta / WarmDecodePos / CaptureDecodePosAdvance warm paths.
// Registrations stay in tenstorrent_ops.cpp's Registrar.

#include "vt/tenstorrent/tenstorrent_internal.h"

namespace vt::tenstorrent {
namespace {
// ---- Paged KV device shadows (ttnn layout) ---------------------------------
// Host keeps vLLM NHD [nb, block, nkv, d] (LMCache/plane). Device PA needs
// ttnn order [nb, nkv, block, d] TILE DRAM.
//
// Incremental path:
//  1) Host-side ttnn-order *mirror* patched on each ReshapeAndCache write.
//  2) When a device shadow already covers the written blocks, push them on-device:
//       - paged_fill_cache for long sequential prefill (interleaved [1,nkv,T,d])
//       - else batched paged_update_cache (height-sharded [1,B,nkv_pad,d], B tokens
//         per call, chunked to the device core count)
//     On failure, leave device dirty → Ensure re-uploads from the mirror only.

struct PagedKvShadow {
  std::optional<ttnn::Tensor> device;  // [nb, nkv, bs, d] TILE BF16 DRAM
  std::vector<float> mirror;           // ttnn-order f32, size nb*nkv*bs*d
  uint32_t nb = 0, nkv = 0, bs = 0, d = 0;
  bool mirror_valid = false;   // mirror matches host NHD for [0, nb)
  bool device_current = false; // device matches mirror
};

std::mutex& PagedKvMutex() {
  static std::mutex m;
  return m;
}
std::map<uintptr_t, PagedKvShadow>& PagedKvShadows() {
  static std::map<uintptr_t, PagedKvShadow>* m = new std::map<uintptr_t, PagedKvShadow>(); // never destroyed (#1486)
  return *m;
}


}  // namespace
// Drop device + mirror (geometry change or Free).
void DropPagedKvShadow(void* host) {
  if (host == nullptr) return;
  std::lock_guard<std::mutex> g(PagedKvMutex());
  PagedKvShadows().erase(reinterpret_cast<uintptr_t>(host));
}
namespace {


// The flash-KV unbind(1) slice of the combined [nb,2,bs,nkv,d] paged store
// (dense_attn_block.h KvSlice): dense inner dims, block stride 2*bs*nkv*d.
// WarmPagedKvShadow stages exactly this view; nothing else may relax the
// contiguity requirement.
bool IsFlashKvUnbindView(const Tensor& t) {
  return t.rank == 4 && t.stride[1] == t.shape[2] * t.shape[3] &&
         t.stride[2] == t.shape[3] && t.stride[3] == 1;
}

// Convert host NHD blocks [0, used_nb) → ttnn [used_nb, nkv, bs, d] f32.
// `accept_unbind_view` also admits the flash-KV unbind view above — vetted,
// never assumed: the caller must pass the flag and the inner strides must
// still be dense, so a genuinely misdescribed cache still fails the check
// instead of silently reading the wrong slab. Default false keeps every
// existing caller strict.
std::vector<float> NhdToTtnnLayoutPrefix(const Tensor& cache, uint32_t used_nb,
                                         bool accept_unbind_view = false) {
  EnsureHost(cache);
  VT_CHECK(cache.rank == 4, "NhdToTtnn: rank-4");
  const int64_t nb = cache.shape[0], bs = cache.shape[1], nkv = cache.shape[2],
                d = cache.shape[3];
  VT_CHECK(cache.IsContiguous() || (accept_unbind_view && IsFlashKvUnbindView(cache)),
           "NhdToTtnn: rank-4 contiguous (or vetted unbind view)");
  VT_CHECK(used_nb > 0 && static_cast<int64_t>(used_nb) <= nb, "NhdToTtnn: used_nb");
  std::vector<float> out(static_cast<size_t>(used_nb) * static_cast<size_t>(nkv * bs * d));
  for (int64_t b = 0; b < static_cast<int64_t>(used_nb); ++b) {
    for (int64_t g = 0; g < nkv; ++g) {
      for (int64_t off = 0; off < bs; ++off) {
        for (int64_t e = 0; e < d; ++e) {
          // General stride form: for a contiguous cache this is exactly the
          // previous dense formula; for the unbind view it is the only form
          // that reads block b's own slab instead of its neighbor's.
          const int64_t src = b * cache.stride[0] + off * cache.stride[1] +
                              g * cache.stride[2] + e;
          const int64_t dst = ((b * nkv + g) * bs + off) * d + e;
          out[static_cast<size_t>(dst)] = LoadElemF32(cache, src);
        }
      }
    }
  }
  return out;
}

// Grow mirror to cover at least `need_nb` blocks (zero-fill new blocks).
void EnsureMirrorCapacity(PagedKvShadow& s, uint32_t need_nb, uint32_t nkv, uint32_t bs,
                          uint32_t d) {
  if (s.mirror_valid && s.nkv == nkv && s.bs == bs && s.d == d && s.nb >= need_nb) return;
  if (s.mirror_valid && s.nkv == nkv && s.bs == bs && s.d == d && s.nb < need_nb) {
    // Grow: keep existing prefix, zero the new blocks.
    const size_t old_n = s.mirror.size();
    s.mirror.resize(static_cast<size_t>(need_nb) * nkv * bs * d, 0.0f);
    (void)old_n;
    s.nb = need_nb;
    s.device_current = false;
    return;
  }
  // Geometry mismatch or cold: allocate zeros; caller may fill from NHD.
  s.mirror.assign(static_cast<size_t>(need_nb) * nkv * bs * d, 0.0f);
  s.nb = need_nb;
  s.nkv = nkv;
  s.bs = bs;
  s.d = d;
  s.mirror_valid = true;
  s.device_current = false;
  s.device = std::nullopt;
}

// Patch one token into the ttnn-order mirror (and mark device stale).
// `tok` is contiguous [nkv, d] for that cache plane (K or V).
void PatchMirrorToken(PagedKvShadow& s, uint32_t block, uint32_t offset, const float* tok,
                      uint32_t nkv, uint32_t d) {
  VT_CHECK(s.mirror_valid && s.nkv == nkv && s.d == d && block < s.nb && offset < s.bs,
           "PatchMirrorToken: mirror geometry");
  for (uint32_t g = 0; g < nkv; ++g) {
    const size_t dst =
        (static_cast<size_t>(block) * nkv + g) * s.bs * d + static_cast<size_t>(offset) * d;
    std::memcpy(s.mirror.data() + dst, tok + static_cast<size_t>(g) * d,
                static_cast<size_t>(d) * sizeof(float));
  }
  s.device_current = false;
}

// True when tokens form a sequential fill from logical position 0 of a page
// table: offset[i] == i % bs and block is constant per logical page. Enables
// paged_fill_cache (one interleaved write for the whole prefill chunk).
bool IsSequentialFillEligible(const std::vector<uint32_t>& blocks,
                              const std::vector<uint32_t>& offsets, uint32_t bs) {
  const size_t T = blocks.size();
  if (T == 0 || offsets.size() != T || bs == 0) return false;
  if (offsets[0] != 0) return false;
  for (size_t i = 0; i < T; ++i) {
    if (offsets[i] != static_cast<uint32_t>(i % bs)) return false;
    const size_t group0 = (i / bs) * bs;
    if (blocks[i] != blocks[group0]) return false;
  }
  return true;
}

// Prefill fill via paged_fill_cache. Input layout [1, nkv, T_pad, d] INTERLEAVED
// TILE. The kernel walks padded_shape[2] in TILE rows, so T is rounded up to a
// multiple of 32 and the pad region is zero-filled (safe for unused tail slots
// in a fresh prefill block). `toks` is packed [T, nkv, d] token-major.
bool TryDevicePagedFill(ttnn::Tensor& cache_dev, MeshDevice& device,
                        const std::vector<uint32_t>& blocks, const float* toks, uint32_t T,
                        uint32_t nkv, uint32_t d, uint32_t bs) {
  if (T == 0 || blocks.size() < T) return false;
  try {
    const uint32_t T_pad = ((T + 31u) / 32u) * 32u;
    // Pack [1, nkv, T_pad, d] from token-major [T, nkv, d]; pad tail with zeros.
    std::vector<float> x(static_cast<size_t>(nkv) * T_pad * d, 0.0f);
    for (uint32_t t = 0; t < T; ++t) {
      for (uint32_t g = 0; g < nkv; ++g) {
        const float* src = toks + (static_cast<size_t>(t) * nkv + g) * d;
        float* dst = x.data() + (static_cast<size_t>(g) * T_pad + t) * d;
        std::memcpy(dst, src, static_cast<size_t>(d) * sizeof(float));
      }
    }
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
      std::fprintf(stderr, "[TT-UP] TryDevicePagedFill from_vector WRITE during capture\n");
    ttnn::Tensor xt = ttnn::Tensor::from_vector<float>(
        x, SpecOf(tt::tt_metal::Shape({1u, nkv, T_pad, d}), ttnn::DataType::BFLOAT16,
                  ttnn::Layout::TILE),
        &device);

    const uint32_t n_logical = (T_pad + bs - 1u) / bs;
    std::vector<int32_t> pt(static_cast<size_t>(n_logical));
    for (uint32_t j = 0; j < n_logical; ++j) {
      const uint32_t tok_i = std::min(j * bs, T - 1u);
      pt[static_cast<size_t>(j)] = static_cast<int32_t>(blocks[static_cast<size_t>(tok_i)]);
    }
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
      std::fprintf(stderr, "[TT-UP] TryDevicePagedFill from_vector WRITE during capture\n");
    ttnn::Tensor page_table = ttnn::Tensor::from_vector<int32_t>(
        pt, SpecOf(tt::tt_metal::Shape({1u, n_logical}), ttnn::DataType::INT32,
                   ttnn::Layout::ROW_MAJOR),
        &device);

    cache_dev = ttnn::experimental::paged_fill_cache(
        cache_dev, xt, page_table, /*batch_idx_tensor=*/std::nullopt, /*batch_idx=*/0,
        /*compute_kernel_config=*/std::nullopt, /*mesh_coords=*/std::nullopt);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

// Pack token-major [C, nkv, d] → height-sharded [1, C, nkv_pad, d] L1 TILE.
ttnn::Tensor MakeHeightShardedUpdateInput(MeshDevice& device, const float* toks, uint32_t base,
                                          uint32_t C, uint32_t nkv, uint32_t nkv_pad, uint32_t d,
                                          const tt::tt_metal::CoreCoord& grid) {
  std::vector<float> x(static_cast<size_t>(C) * nkv_pad * d, 0.0f);
  for (uint32_t b = 0; b < C; ++b) {
    const float* src = toks + (static_cast<size_t>(base + b) * nkv * d);
    float* dst = x.data() + static_cast<size_t>(b) * nkv_pad * d;
    for (uint32_t g = 0; g < nkv; ++g) {
      std::memcpy(dst + static_cast<size_t>(g) * d, src + static_cast<size_t>(g) * d,
                  static_cast<size_t>(d) * sizeof(float));
    }
  }
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-UP] MakeHeightShardedUpdateInput from_vector WRITE during capture\n");
  ttnn::Tensor xt = ttnn::Tensor::from_vector<float>(
      x, SpecOf(tt::tt_metal::Shape({1u, C, nkv_pad, d}), ttnn::DataType::BFLOAT16,
                ttnn::Layout::TILE),
      &device);
  const tt::tt_metal::CoreRangeSet core_set =
      tt::tt_metal::num_cores_to_corerangeset(C, grid, /*row_wise=*/true);
  tt::tt_metal::ShardSpec shard_spec(core_set, {nkv_pad, d},
                                     tt::tt_metal::ShardOrientation::ROW_MAJOR);
  tt::tt_metal::MemoryConfig sharded_mem(tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                                         tt::tt_metal::BufferType::L1, shard_spec);
  return ttnn::to_memory_config(xt, sharded_mem);
}

// Batched in-place device write via paged_update_cache. Treats each token as a
// "batch user" with a 1-entry synthetic page table (phys_block) and
// update_idx=offset. Height-shards onto B cores; chunks to the device grid when
// B exceeds available cores. `toks` is packed [B, nkv, d].
bool TryDevicePagedUpdateBatch(ttnn::Tensor& cache_dev, MeshDevice& device,
                               const std::vector<uint32_t>& phys_blocks,
                               const std::vector<uint32_t>& offsets, const float* toks,
                               uint32_t nkv, uint32_t d, uint32_t /*bs*/) {
  const uint32_t B = static_cast<uint32_t>(phys_blocks.size());
  if (B == 0 || offsets.size() != phys_blocks.size()) return false;
  try {
    const uint32_t nkv_pad = std::max(32u, ((nkv + 31u) / 32u) * 32u);
    const auto grid = device.compute_with_storage_grid_size();
    const uint32_t max_cores =
        std::max(1u, static_cast<uint32_t>(grid.x) * static_cast<uint32_t>(grid.y));

    for (uint32_t base = 0; base < B; base += max_cores) {
      const uint32_t C = std::min(max_cores, B - base);
      ttnn::Tensor xt =
          MakeHeightShardedUpdateInput(device, toks, base, C, nkv, nkv_pad, d, grid);

      std::vector<int32_t> pt(static_cast<size_t>(C));
      std::vector<int32_t> idxs(static_cast<size_t>(C));
      for (uint32_t b = 0; b < C; ++b) {
        pt[static_cast<size_t>(b)] = static_cast<int32_t>(phys_blocks[static_cast<size_t>(base + b)]);
        idxs[static_cast<size_t>(b)] = static_cast<int32_t>(offsets[static_cast<size_t>(base + b)]);
      }
      if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
        std::fprintf(stderr, "[TT-UP] TryDevicePagedUpdateBatch from_vector WRITE during capture\n");
      ttnn::Tensor page_table = ttnn::Tensor::from_vector<int32_t>(
          pt, SpecOf(tt::tt_metal::Shape({C, 1u}), ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR),
          &device);
      // Paged mode requires update_idxs as a DEVICE tensor (the vector form
      // alone is rejected: "Paged cache requires update_idxs tensor").
      ttnn::Tensor update_idxs_tensor = ttnn::Tensor::from_vector<int32_t>(
          idxs, SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(C)}),
                      ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR), &device);

      cache_dev = ttnn::experimental::paged_update_cache(
          cache_dev, xt, /*update_idxs=*/{}, update_idxs_tensor,
          /*share_cache=*/false, page_table, /*batch_offset=*/0,
          /*compute_kernel_config=*/std::nullopt, /*mesh_coords=*/std::nullopt);
    }
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

// Fused K+V paged_update — one device op for both caches (half the launches).
bool TryDevicePagedFusedUpdateBatch(ttnn::Tensor& k_dev, ttnn::Tensor& v_dev, MeshDevice& device,
                                    const std::vector<uint32_t>& phys_blocks,
                                    const std::vector<uint32_t>& offsets, const float* k_toks,
                                    const float* v_toks, uint32_t nkv, uint32_t d,
                                     uint32_t /*bs*/) {
  const uint32_t B = static_cast<uint32_t>(phys_blocks.size());
  if (B == 0 || offsets.size() != phys_blocks.size()) return false;
  // The fused op rejects overlapping K/V sharded input grids
  // ("input_tensor1 and input_tensor2 must not overlap"). Both K and V
  // shards land on the same C cores via MakeHeightShardedUpdateInput.
  // Fall back to two separate TryDevicePagedUpdateBatch calls (the paired
  // path at TryDevicePagedPushPair handles this).
  return false;
  (void)k_dev; (void)v_dev; (void)device; (void)k_toks; (void)v_toks;
  (void)nkv; (void)d;  // suppress unused-param warnings
}

// Sequential prefills always take paged_fill_cache, at any chunk length. The
// batched update is per token: one batch user per token over a synthetic
// one-entry page-table stick, so a short chunk sends several users at the SAME
// physical block, and paged_update_cache's page-granular concurrent
// read-modify-write turns that into last-writer-wins — the block keeps the
// previous occupant's rows while the push publishes the shadow as current
// (#2669). Decode pushes never satisfy the eligibility predicate (their first
// offset is mid-block), so they stay on the height-sharded update path.
constexpr uint32_t kPagedFillMinTokens = 1;

bool TryDevicePagedPush(ttnn::Tensor& cache_dev, MeshDevice& device,
                        const std::vector<uint32_t>& blocks, const std::vector<uint32_t>& offsets,
                        const float* toks, uint32_t nkv, uint32_t d, uint32_t bs) {
  const uint32_t T = static_cast<uint32_t>(blocks.size());
  if (T >= kPagedFillMinTokens && IsSequentialFillEligible(blocks, offsets, bs)) {
    if (TryDevicePagedFill(cache_dev, device, blocks, toks, T, nkv, d, bs)) return true;
    // Fall through to batched update on fill failure.
  }
  return TryDevicePagedUpdateBatch(cache_dev, device, blocks, offsets, toks, nkv, d, bs);
}

// Push K and V together: fill (2 calls) or fused update (1 call). Falls back to
// independent pushes if the paired path fails.
bool TryDevicePagedPushPair(ttnn::Tensor& k_dev, ttnn::Tensor& v_dev, MeshDevice& device,
                            const std::vector<uint32_t>& blocks,
                            const std::vector<uint32_t>& offsets, const float* k_toks,
                            const float* v_toks, uint32_t nkv, uint32_t d, uint32_t bs) {
  const uint32_t T = static_cast<uint32_t>(blocks.size());
  if (T >= kPagedFillMinTokens && IsSequentialFillEligible(blocks, offsets, bs)) {
    if (TryDevicePagedFill(k_dev, device, blocks, k_toks, T, nkv, d, bs) &&
        TryDevicePagedFill(v_dev, device, blocks, v_toks, T, nkv, d, bs)) {
      return true;
    }
    // Fall through.
  }
  if (TryDevicePagedFusedUpdateBatch(k_dev, v_dev, device, blocks, offsets, k_toks, v_toks, nkv, d,
                                     bs)) {
    return true;
  }
  // Last resort: independent updates.
  return TryDevicePagedUpdateBatch(k_dev, device, blocks, offsets, k_toks, nkv, d, bs) &&
         TryDevicePagedUpdateBatch(v_dev, device, blocks, offsets, v_toks, nkv, d, bs);
}

// After host NHD RAC writes: patch ttnn mirrors for all valid slots, then push
// to device in one (or few) paged_fill / paged_update call(s) when a shadow
// already covers every written block.
// `blocks`/`offsets` length B; `k_toks`/`v_toks` packed [B, nkv, d].
void NotePagedKvRacWrites(Tensor& k_cache, Tensor& v_cache, const std::vector<uint32_t>& blocks,
                          const std::vector<uint32_t>& offsets, const std::vector<float>& k_toks,
                          const std::vector<float>& v_toks) {
  if (k_cache.rank != 4 || v_cache.rank != 4) return;
  const uint32_t B = static_cast<uint32_t>(blocks.size());
  if (B == 0 || offsets.size() != blocks.size()) return;
  const uint32_t bs = static_cast<uint32_t>(k_cache.shape[1]);
  const uint32_t nkv = static_cast<uint32_t>(k_cache.shape[2]);
  const uint32_t d = static_cast<uint32_t>(k_cache.shape[3]);
  if ((d % 32u) != 0 || (bs % 32u) != 0) return;  // device PA won't run
  if (k_toks.size() < static_cast<size_t>(B) * nkv * d ||
      v_toks.size() < static_cast<size_t>(B) * nkv * d) {
    return;
  }

  uint32_t max_block = 0;
  for (uint32_t b : blocks) {
    if (b >= static_cast<uint32_t>(k_cache.shape[0])) return;
    if (b > max_block) max_block = b;
  }
  for (uint32_t o : offsets) {
    if (o >= bs) return;
  }
  const uint32_t need = max_block + 1u;

  // Phase 1: patch host mirrors under lock; snapshot device tensors if present.
  std::optional<ttnn::Tensor> k_dev, v_dev;
  bool k_can_update = false, v_can_update = false;
  {
    std::lock_guard<std::mutex> g(PagedKvMutex());
    auto prepare = [&](void* host, const std::vector<float>& toks,
                       std::optional<ttnn::Tensor>& dev_out, bool& can_update) {
      PagedKvShadow& s = PagedKvShadows()[reinterpret_cast<uintptr_t>(host)];
      EnsureMirrorCapacity(s, need, nkv, bs, d);
      for (uint32_t i = 0; i < B; ++i) {
        PatchMirrorToken(s, blocks[static_cast<size_t>(i)], offsets[static_cast<size_t>(i)],
                         toks.data() + static_cast<size_t>(i) * nkv * d, nkv, d);
      }
      if (s.device.has_value() && s.nb > max_block && s.nkv == nkv && s.bs == bs && s.d == d) {
        dev_out = s.device;
        can_update = true;
      }
    };
    prepare(k_cache.data, k_toks, k_dev, k_can_update);
    prepare(v_cache.data, v_toks, v_dev, v_can_update);
  }

  // Phase 2: optional on-device push (outside lock). Prefer fused K+V.
  MeshDevice* device = nullptr;
  try {
    device = &SharedMeshDevice();
  } catch (...) {
    return;
  }
  auto mark_dirty = [&](void* host) {
    std::lock_guard<std::mutex> g(PagedKvMutex());
    PagedKvShadow& s = PagedKvShadows()[reinterpret_cast<uintptr_t>(host)];
    if (s.device.has_value()) s.device_current = false;
  };
  auto publish = [&](void* host, ttnn::Tensor dev) {
    std::lock_guard<std::mutex> g(PagedKvMutex());
    PagedKvShadow& s = PagedKvShadows()[reinterpret_cast<uintptr_t>(host)];
    s.device = std::move(dev);
    s.device_current = true;
  };

  if (k_can_update && v_can_update && k_dev.has_value() && v_dev.has_value()) {
    if (TryDevicePagedPushPair(*k_dev, *v_dev, *device, blocks, offsets, k_toks.data(),
                               v_toks.data(), nkv, d, bs)) {
      publish(k_cache.data, std::move(*k_dev));
      publish(v_cache.data, std::move(*v_dev));
    } else {
      mark_dirty(k_cache.data);
      mark_dirty(v_cache.data);
    }
    return;
  }
  auto try_one = [&](void* host, std::optional<ttnn::Tensor>& dev, bool can, const float* toks) {
    if (!can || !dev.has_value()) return;
    if (!TryDevicePagedPush(*dev, *device, blocks, offsets, toks, nkv, d, bs)) {
      mark_dirty(host);
      return;
    }
    publish(host, std::move(*dev));
  };
  try_one(k_cache.data, k_dev, k_can_update, k_toks.data());
  try_one(v_cache.data, v_dev, v_can_update, v_toks.data());
}

// Return a current ttnn-layout device tensor covering physical blocks [0, used_nb).
// HOST-FREE-DECODE: the device shadow is allocated ONCE at the FULL pool size
// (all num_blocks blocks, tail zero-filled). Sizing it to the used prefix — as
// this function originally did — reallocates and FREES the previous device
// buffer whenever `used_nb` crosses a block boundary. While a mesh trace is
// alive the captured RAC/PA commands still reference the old address; the
// allocator hands it to a new allocation and tt-metal's warning fires
// ("buffers may be corrupted once a trace is executed"). Observed as every
// replay collapsing to all-zero logits at the first block-table growth
// (slot 63→64 with block_size 32). A pool-sized shadow never moves.
ttnn::Tensor EnsurePagedKvTtnn(const Tensor& cache_nhd, MeshDevice& device, uint32_t used_nb,
                               bool accept_unbind_view = false) {
  VT_CHECK(cache_nhd.rank == 4, "EnsurePagedKvTtnn: rank-4 NHD cache");
  VT_CHECK(cache_nhd.IsContiguous() ||
               (accept_unbind_view && IsFlashKvUnbindView(cache_nhd)),
           "EnsurePagedKvTtnn: contiguous rank-4 NHD cache (or vetted unbind view)");
  const uint32_t pool_nb = static_cast<uint32_t>(cache_nhd.shape[0]);
  const uint32_t bs = static_cast<uint32_t>(cache_nhd.shape[1]);
  const uint32_t nkv = static_cast<uint32_t>(cache_nhd.shape[2]);
  const uint32_t d = static_cast<uint32_t>(cache_nhd.shape[3]);
  VT_CHECK(used_nb > 0 && used_nb <= pool_nb, "EnsurePagedKvTtnn: used_nb out of range");

  {
    std::lock_guard<std::mutex> g(PagedKvMutex());
    PagedKvShadow& s = PagedKvShadows()[reinterpret_cast<uintptr_t>(cache_nhd.data)];
    if (s.device_current && s.device.has_value() && s.nb >= used_nb && s.nkv == nkv &&
        s.bs == bs && s.d == d) {
      return *s.device;
    }
    // Mirror grows to the full pool (zero-filled tail) so both mirror and
    // device shadow stay at one stable size for the cache's lifetime.
    EnsureMirrorCapacity(s, pool_nb, nkv, bs, d);
  }

  // Cold / short / geometry change: rebuild the used prefix from the host NHD
  // cache into the (full-size) mirror; the tail stays zero.
  std::vector<float> used;
  {
    std::lock_guard<std::mutex> g(PagedKvMutex());
    PagedKvShadow& s = PagedKvShadows()[reinterpret_cast<uintptr_t>(cache_nhd.data)];
    if (!s.mirror_valid || s.nkv != nkv || s.bs != bs || s.d != d || s.nb < used_nb) {
      s.mirror_valid = false;  // content below [0,used) not trustworthy yet
    }
  }
  used = NhdToTtnnLayoutPrefix(cache_nhd, used_nb, accept_unbind_view);
  {
    std::lock_guard<std::mutex> g(PagedKvMutex());
    PagedKvShadow& s = PagedKvShadows()[reinterpret_cast<uintptr_t>(cache_nhd.data)];
    EnsureMirrorCapacity(s, pool_nb, nkv, bs, d);
    VT_CHECK(s.mirror.size() >= used.size(), "EnsurePagedKvTtnn: mirror shorter than used");
    std::memcpy(s.mirror.data(), used.data(), used.size() * sizeof(float));
    s.mirror_valid = true;
  }

  // Build the pool-sized device shadow WITHOUT a full-pool from_vector: the
  // TILE-layout host transform is per-element and a 256-block vector costs
  // seconds per cache (56 caches stalled cold for minutes). Upload only the
  // used prefix, allocate the zero tail with ttnn::zeros (host std::fill +
  // straight DMA — a constant fill is layout-order-agnostic), and stitch with
  // a device-side concat. paged_update_cache / sdpa_decode never read the
  // zero tail (page-table entries only cover allocated blocks).
  const auto used_spec = SpecOf(tt::tt_metal::Shape({used_nb, nkv, bs, d}),
                                ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-UP] EnsurePagedKvTtnn from_vector WRITE during capture\n");
  const auto dbg = std::getenv("VT_TT_TRACE_DEBUG") != nullptr;
  const auto t0 = std::chrono::steady_clock::now();
  ttnn::Tensor dev = ttnn::Tensor::from_vector<float>(used, used_spec, &device);
  if (dbg) {
    const auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "[TT-KVTIM] %p from_vector used_nb=%u %.0fms\n",
                 cache_nhd.data, used_nb,
                 std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  if (used_nb < pool_nb) {
    ttnn::Tensor tail = ttnn::zeros(
        tt::tt_metal::Shape({pool_nb - used_nb, nkv, bs, d}), ttnn::DataType::BFLOAT16,
        ttnn::Layout::TILE, std::ref(device));
    if (dbg) {
      const auto t2 = std::chrono::steady_clock::now();
      std::fprintf(stderr, "[TT-KVTIM] %p zeros tail_nb=%u %.0fms\n",
                   cache_nhd.data, pool_nb - used_nb,
                   std::chrono::duration<double, std::milli>(t2 - t0).count());
    }
    dev = ttnn::concat(std::vector<ttnn::Tensor>{dev, tail}, 0);
    if (dbg) {
      const auto t3 = std::chrono::steady_clock::now();
      std::fprintf(stderr, "[TT-KVTIM] %p concat pool_nb=%u %.0fms\n",
                   cache_nhd.data, pool_nb,
                   std::chrono::duration<double, std::milli>(t3 - t0).count());
    }
  }

  std::lock_guard<std::mutex> g(PagedKvMutex());
  PagedKvShadow& s = PagedKvShadows()[reinterpret_cast<uintptr_t>(cache_nhd.data)];
  s.device = dev;
  s.device_current = true;
  return dev;
}

}  // namespace
namespace {
// kReshapeAndCache: write per-token K/V into paged NHD cache slots
// (cpu_cache.cpp ReshapeAndCacheKernel). Stride-driven so unbind-style
// [num_blocks,2,bs,H,D] views work; slot < 0 is a padded-token skip.
// Host-staged pure element copy for F32.
// ---- ITEM 5 (PA): persistent page_table + cur_pos device tensors -------------
namespace {
struct PaMetaEntry {
  ttnn::Tensor page_table;  // int32 [B, max_blocks] device
  ttnn::Tensor cur_pos;     // int32 [B] device
  std::vector<int32_t> pt_host;
  std::vector<int32_t> cp_host;
  bool allocated = false;  // ttnn::Tensor::is_allocated() crashes on default-constructed tensors in this build
};
std::mutex& PaMetaMutex() { static std::mutex m; return m; }
std::map<std::pair<int64_t, int64_t>, PaMetaEntry>& PaMetaCache() {
  static std::map<std::pair<int64_t, int64_t>, PaMetaEntry>* c = new std::map<std::pair<int64_t, int64_t>, PaMetaEntry>(); // never destroyed (#1486)
  return *c;
}
}  // namespace

// ---- R2: persistent cur_pos advanced on-device via plus_one ----------------
// The PaMeta cur_pos tensor (read by sdpa_decode) and the RAC update_idxs
// tensor (read by paged_update_cache) both hold `seq_lens - 1` for decode.
// R2 aliases them: WarmDecodePos seeds the single persistent cur_pos tensor;
// CaptureDecodePosAdvance does plus_one on it inside the trace; WarmRacIdx's
// update_idxs copy_to_device is skipped (it reuses this tensor).
namespace {
struct DecodePosEntry {
  ttnn::Tensor cur_pos;  // int32 [num_reqs] device — advanced in-trace
  bool allocated = false;
  // #2469: host mirror of what cur_pos holds on device. The seed branch
  // records seq_lens-1 (its copy_to_device lands); the replay branch adds 1
  // per step it declines seeding for (each launched trace plus_one'd the
  // device). WarmPaMeta echoes this into PaMetaEntry.cp_host so the
  // TryPagedAttentionDeviceDecode guard reads what the DEVICE holds — a
  // stale device tensor can no longer be masked by echoing the step's
  // freshly computed seq_lens-1 into the entry.
  std::vector<int32_t> host_val;
  // #2469: the plus_one program-cache warm runs its op on a scratch tensor.
  // Allocate it ONCE, at this entry's first (always trace-free) seed, and
  // reuse it after: a fresh device allocation per seed is what the allocator
  // warns about under a live trace ("these buffers may be corrupted once a
  // trace is executed"), and the boundary seed is the one seeding step that
  // runs with THIS slot's trace still live.
  ttnn::Tensor plus_one_scratch;
  bool scratch_ready = false;
};
std::mutex& DecodePosMutex() { static std::mutex m; return m; }
std::map<int64_t, DecodePosEntry>& DecodePosCache() {
  static std::map<int64_t, DecodePosEntry>* c = new std::map<int64_t, DecodePosEntry>(); // never destroyed (#1486)
  return *c;
}
}  // namespace

// ---- ITEM 5 (RAC): persistent update-idx / page-table device tensors -------
// Refreshed by WarmRacIdx (driver Refresh slot, outside capture) so the
// captured paged_update_cache replays against stable addresses. Keyed by the
// slot-mapping HOST buffer (the decode-graph slot's persistent buffer), so a
// different graph size gets its own entries.
namespace {
struct RacIdxEntry {
  ttnn::Tensor update_idxs;  // int32 [C] device (persistent, content refreshed)
  ttnn::Tensor page_table;   // int32 [C,pt_width] device (persistent, content refreshed)
  std::vector<int32_t> pt_host;   // last page-table content copied to device
  int64_t pt_width = 0;           // columns of the allocated page_table
  // Retired page-table tensors from width growth, kept ALIVE deliberately:
  // a freed device buffer can hand its address to a new allocation while a
  // (doomed, never-replayed-again) trace still records it. Bounded by the
  // number of block boundaries crossed (~context/block_size).
  std::vector<ttnn::Tensor> retired_pts;
  // Persistent height-sharded RAC input: logical [1,1,nkv,d], padded
  // [1,1,nkv_pad,d] (shard [nkv_pad,d] on one core). The in-region RAC
  // ttnn::copy's the rope output into it; paged_update_cache reads only the
  // first nkv rows (num_heads loop bound), so the padded tail rows are never
  // read and may hold garbage — no zeros tail, no concat, no allocation.
  ttnn::Tensor sharded_in;   // K input (height-sharded)
  ttnn::Tensor sharded_in_v;  // V input (separate — K and V must NOT share the same buffer)
  uint32_t nkv = 0;
  uint32_t d = 0;
  bool allocated = false;        // ttnn::Tensor::is_allocated() crashes on default-constructed tensors in this build
  bool sharded_in_is_alloc = false;
};
std::mutex& RacIdxMutex() { static std::mutex m; return m; }
// Keyed by (num_slots, block_size) shape — idx tensors depend on slot values + block_size.
std::map<std::pair<int64_t, int64_t>, RacIdxEntry>& RacIdxCache() {
  static std::map<std::pair<int64_t, int64_t>, RacIdxEntry>* c = new std::map<std::pair<int64_t, int64_t>, RacIdxEntry>(); // never destroyed (#1486)
  return *c;
}
}  // namespace

// Warm hook: stage persistent idx tensors for THIS slot mapping. Host reads
// here are legal (called outside capture). Idempotent per content change.

// Host-free decode RAC: device shadows in, paged_update_cache out. Returns
// false (host path) unless every precondition holds.
bool TryReshapeAndCacheDeviceDecode(const Tensor& k, const Tensor& v,
                                    Tensor& k_cache, Tensor& v_cache,
                                    const Tensor& slot_mapping) {
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] TryRACDevice called capturing=%d\n",
                 (int)tt_capture_active());
  const int64_t T = k.shape[0];
  const int64_t nkv = k.shape[1];
  const int64_t d = k.shape[2];
  const int64_t bs = k_cache.shape[1];
  const int64_t num_slots = slot_mapping.shape[0];
  if (T < 1 || num_slots < 1) return false;
  if ((d % 32u) != 0u || (bs % 32u) != 0u) return false;
  if (num_slots > 1) return false;  // decode T=1 only for now

  // k/v must carry CURRENT device shadows ([T*nkv, d] TILE bf16 from rope).
  std::optional<ttnn::Tensor> k_dev, v_dev;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* sk = FindSlot(k.data);
    BufferSlot* sv = FindSlot(v.data);
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-TRACE] RAC kv shadow k_ptr=%p sk=%p dc=%d val=%d | v_ptr=%p sv=%p dc=%d val=%d\n",
                   k.data, (void*)sk, sk?sk->device_current:0, sk?(int)sk->device.has_value():0,
                   v.data, (void*)sv, sv?sv->device_current:0, sv?(int)sv->device.has_value():0);
    if (sk == nullptr || !sk->device_current || !sk->device.has_value()) return false;
    if (sv == nullptr || !sv->device_current || !sv->device.has_value()) return false;
    k_dev = sk->device;
    v_dev = sv->device;
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && !tt_capture_active()) {
      auto dump_src = [](const char* tag, const ttnn::Tensor& t) {
        const auto ls = t.logical_shape();
        const auto ps = t.padded_shape();
        std::fprintf(stderr,
                     "[TT-TRACE] RAC src %s: logical=[", tag);
        for (size_t i = 0; i < ls.rank(); ++i)
          std::fprintf(stderr, "%s%u", i ? "," : "", ls[i]);
        std::fprintf(stderr, "] padded=[");
        for (size_t i = 0; i < ps.rank(); ++i)
          std::fprintf(stderr, "%s%u", i ? "," : "", ps[i]);
        std::fprintf(stderr,
                     "] dtype=%d layout=%d pages=%u strides0123=[%u,%u,%u,%u]\n",
                     (int)t.dtype(), (int)t.layout(),
                     t.buffer() ? t.buffer()->num_pages() : 0u,
                     ls.rank() > 0 ? t.strides()[0] : 0, ls.rank() > 1 ? t.strides()[1] : 0,
                     ls.rank() > 2 ? t.strides()[2] : 0, ls.rank() > 3 ? t.strides()[3] : 0);
      };
      dump_src("k", *k_dev);
      dump_src("v", *v_dev);
    }
  }

  // Paged-KV shadows must exist and cover the target block.
  const int64_t slot = slot_mapping.Ptr<int64_t>()[0];
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] RAC slot=%lld cap=%d\n",
                 (long long)slot, (int)tt_capture_active());
  if (slot < 0) return true;  // nothing to write; treat as handled
  const uint32_t block = static_cast<uint32_t>(slot / bs);
  const uint32_t offset = static_cast<uint32_t>(slot % bs);

  std::optional<ttnn::Tensor> kc_dev, vc_dev;
  {
    std::lock_guard<std::mutex> g(PagedKvMutex());
    PagedKvShadow* skc = &PagedKvShadows()[reinterpret_cast<uintptr_t>(k_cache.data)];
    PagedKvShadow* svc = &PagedKvShadows()[reinterpret_cast<uintptr_t>(v_cache.data)];
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
      std::fprintf(stderr, "[TT-TRACE] RAC paged-kv shadow k=%d v=%d k_nb=%u\n",
                   skc->device.has_value(), svc->device.has_value(), skc->nb);
    if (!skc->device.has_value() || !svc->device.has_value()) return false;
    if (skc->nb <= block || skc->nkv != static_cast<uint32_t>(nkv) ||
        skc->bs != static_cast<uint32_t>(bs) || skc->d != static_cast<uint32_t>(d)) return false;
    if (svc->nb <= block || svc->nkv != static_cast<uint32_t>(nkv) ||
        svc->bs != static_cast<uint32_t>(bs) || svc->d != static_cast<uint32_t>(d)) return false;
    kc_dev = skc->device;
    vc_dev = svc->device;
  }

  // Persistent idx tensors for THIS slot-mapping buffer (warmed outside
  // capture). Both must exist; content refresh happens at warm time.
  {
    std::lock_guard<std::mutex> g(RacIdxMutex());
    const auto key = std::make_pair(num_slots, static_cast<int64_t>(bs));
    auto it = RacIdxCache().find(key);
    const int64_t slot0 = slot_mapping.Ptr<int64_t>()[0];
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-TRACE] RAC idx-check slot0=%lld cap=%d key=(%lld,%lld)\n",
                   (long long)slot0, (int)tt_capture_active(),
                   (long long)num_slots, (long long)bs);
    // WarmRacIdx (driver Refresh slot) refreshes update_idxs/page_table content
    // every step via copy_to_device; here we just verify the tensors exist.
    if (it == RacIdxCache().end() || !it->second.allocated) {
      VT_CHECK(!tt_capture_active(),
               "tenstorrent: RAC idx tensors not warmed — call WarmRacIdx "
               "outside capture (driver Refresh slot) first");
      return false;
    }
    // sharded_in must exist (WarmRacIdx needs the paged-KV shadow geometry).
    if (!it->second.sharded_in_is_alloc) {
      VT_CHECK(!tt_capture_active(),
               "tenstorrent: RAC sharded input not warmed — call WarmRacIdx "
               "outside capture after WarmPagedKvShadow");
      return false;
    }
  }

  // Eager (cold) step AND capture: the identical op sequence (see
  // build_input below). Running it eagerly first compiles the programs;
  // capture then hits the program cache and replays against the persistent
  // addresses.
  RacIdxEntry rac_entry = [&] {
    std::lock_guard<std::mutex> g(RacIdxMutex());
    return RacIdxCache().at(std::make_pair(num_slots, static_cast<int64_t>(bs)));
  }();

  // Single code path for the eager (cold) step AND capture. The cold step must
  // compile the exact programs the captured region will replay, so the op
  // sequence and every TensorSpec must be identical in both phases:
  //   1. reshape (metadata-only) to logical [1,1,nkv,d]
  //   2. scalar multiply by 1.0 — eltwise ops allocate a FRESH output with a
  //      native 4D spec. Feeding the bare 2D→4D reshape view straight into
  //      ttnn::copy only writes head0 (the view's 2D-allocated storage
  //      confuses the tilized copy program), and a host-side to_vector →
  //      from_vector round-trip is illegal during capture and hashes
  //      differently (program-cache miss → binary load during capture).
  //   3. ttnn::copy into the persistent sharded_in (preallocated output; the
  //      interleaved-TILE→height-sharded copy program uses only CBs)
  // paged_fused_update_cache (in-place, has override_runtime_arguments) then
  // ingests the sharded input against persistent addresses.
  auto build_input = [&](const ttnn::Tensor& src, ttnn::Tensor& sharded_dst,
                         const RacIdxEntry& entry) -> ttnn::Tensor {
      // Materialize a NATIVE [1,1,nkv,d] TILE tensor on device with a single
      // code path on cold and capture (host round-trips are illegal during
      // capture and compile differently-hashed programs).
      //
      // TILE readers map a logical element of a 4D [1,1,nkv,d] tensor to
      // in-page ROW h of the (d/32)-page grid — so only [nkv,d]-SHAPED
      // storage (heads at in-page rows 0..nkv-1) can be viewed; a [1,N]
      // native (data at in-page row 0 of N/32 pages) mis-maps (head0-only
      // or stale garbage — verified by per-head value dumps). Therefore:
      //   * [nkv,d] source (rope K output): explicit tile-padded 4D view —
      //     its spec is byte-identical to a native 4D's; multiply reads
      //     per-head exact (verified).
      //   * [1,N] source (QkvSplit V slice): materialize [nkv,d] storage
      //     first — per-head [1,d] slices concatenated on dim 0. Only ops
      //     already proven capture-safe in-region (slice/concat/eltwise).
      // The scalar multiply materializes a fresh native 4D allocation;
      // ttnn::copy moves it into the persistent sharded input.
      const uint32_t nkv_pad = ((entry.nkv + 31u) / 32u) * 32u;
      ttnn::Tensor laid_out = src;
      if (src.logical_shape().rank() == 2 && src.logical_shape()[0] == 1) {
        std::vector<ttnn::Tensor> heads;
        heads.reserve(entry.nkv);
        for (uint32_t h = 0; h < entry.nkv; ++h) {
          heads.push_back(ttnn::slice(
              src, ttsl::SmallVector<uint32_t>{0u, h * entry.d},
              ttsl::SmallVector<uint32_t>{1u, (h + 1u) * entry.d},
              ttsl::SmallVector<uint32_t>{1u, 1u}));
        }
        laid_out = ttnn::concat(heads, /*dim=*/0);
      }
      ttnn::Tensor native4 = ttnn::multiply(
          ttnn::experimental::view(
              laid_out, ttnn::Shape({1u, 1u, entry.nkv, entry.d}),
              ttnn::Shape({1u, 1u, nkv_pad, entry.d})),
          1.0f);
      auto head_maxima = [](const ttnn::Tensor& t, uint32_t nkv, uint32_t d) {
        auto v = t.to_vector<float>();
        std::string s;
        for (uint32_t h = 0; h < nkv; ++h) {
          float mx = 0;
          for (uint32_t e = 0; e < d; ++e) {
            const size_t i = static_cast<size_t>(h) * d + e;
            if (i < v.size()) mx = std::max(mx, std::abs(v[i]));
          }
          s += std::to_string(mx) + ",";
        }
        return s;
      };
      if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && !tt_capture_active()) {
        auto chk = native4.to_vector<float>();
        int nonzero = 0;
        for (uint32_t h = 0; h < entry.nkv; ++h) {
          float mx = 0;
          for (uint32_t e = 0; e < entry.d; ++e) {
            const size_t i = static_cast<size_t>(h) * entry.d + e;
            if (i < chk.size()) mx = std::max(mx, std::abs(chk[i]));
          }
          if (mx > 1e-6f) ++nonzero;
        }
        std::fprintf(stderr, "[TT-TRACE] RAC native4 nonzero_heads=%d/%u "
                     "src_headmax=[%s] laid_headmax=[%s] out_headmax=[%s]\n",
                     nonzero, entry.nkv,
                     head_maxima(src, entry.nkv, entry.d).c_str(),
                     head_maxima(laid_out, entry.nkv, entry.d).c_str(),
                     head_maxima(native4, entry.nkv, entry.d).c_str());
      }
    ttnn::copy(native4, sharded_dst);
    return sharded_dst;
  };
  // V first, then K
  // Debug: dump v_dev properties before sharding
  ttnn::Tensor v_in = build_input(*v_dev, rac_entry.sharded_in_v, rac_entry);
  ttnn::Tensor k_in = build_input(*k_dev, rac_entry.sharded_in, rac_entry);
  // Debug: check v_in for all heads
  // num_kv_heads_override pins the kernel's head loop to nkv rows: the input
  // shard is tile-padded (nkv_pad rows) but only the first nkv rows hold data
  // (upstream decode pattern, test_paged_cache_flexible_geometry.py).
  // Use paged_fused_update_cache (single call for K+V) instead of two separate
  // paged_update_cache calls. The fused op has override_runtime_arguments
  // (the non-fused doesn't), so it works correctly with program cache enabled.
  // The second separate call would reuse the first's cached program with the
  // first's buffer addresses (program cache collision).
  auto [new_kc, new_vc] = ttnn::experimental::paged_fused_update_cache(
      *kc_dev, k_in, *vc_dev, v_in,
      /*update_idxs=*/{}, rac_entry.update_idxs,
      /*share_cache=*/false, rac_entry.page_table,
      /*batch_offset=*/0, /*compute_kernel_config=*/std::nullopt,
      /*mesh_coords=*/std::nullopt);
  {
    std::lock_guard<std::mutex> g(PagedKvMutex());
    PagedKvShadows()[reinterpret_cast<uintptr_t>(k_cache.data)].device = std::move(new_kc);
    PagedKvShadows()[reinterpret_cast<uintptr_t>(k_cache.data)].device_current = true;
    PagedKvShadows()[reinterpret_cast<uintptr_t>(v_cache.data)].device = std::move(new_vc);
    PagedKvShadows()[reinterpret_cast<uintptr_t>(v_cache.data)].device_current = true;
  }
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] RAC device update (copy+paged_update_cache)\n");
  (void)offset; (void)block;
  return true;
}


}  // namespace
void ReshapeAndCacheKernel(Queue&, const Tensor& k, const Tensor& v, Tensor& k_cache,
                           Tensor& v_cache, const Tensor& slot_mapping) {
  TT_OP_TRACE("ReshapeAndCache");
  VT_CHECK(k.rank == 3 && v.rank == 3 && k_cache.rank == 4 && v_cache.rank == 4,
           "tenstorrent kReshapeAndCache: k/v rank-3, caches rank-4");
  VT_CHECK(IsFloatDType(k.dtype) && k.dtype == v.dtype && k_cache.dtype == k.dtype &&
               v_cache.dtype == k.dtype,
           "tenstorrent kReshapeAndCache: k/v/caches must share one float dtype");
  VT_CHECK(slot_mapping.rank == 1 && slot_mapping.dtype == DType::kI64,
           "tenstorrent kReshapeAndCache: slot_mapping rank-1 i64");

  // ITEM 5 (RAC): host-free decode branch. The host path below downloads k/v
  // (rope output shadows) and re-uploads via from_vector in the device push —
  // both fatal during capture. This branch instead feeds the DEVICE shadows
  // straight into paged_update_cache with persistent idx/page-table tensors.
  // Conditions: capturing (or host-free flag), all inputs device-shadowed,
  // TILE-legal dims, and the warm hook already staged the idx tensors.
  // Live read, NOT a function-local static: a latch here would cache the
  // now-default-ON value and silently strip VT_TT_HOST_FREE_DECODE=0 of its
  // effect on this path for the rest of the process (#1688).
  const bool host_free_rac = HostFreeDecodeEnabled();
  if (host_free_rac || tt_capture_active()) {
    if (TryReshapeAndCacheDeviceDecode(k, v, k_cache, v_cache, slot_mapping)) {
      return;
    }
  }

  EnsureHost(k);
  EnsureHost(v);
  EnsureHost(k_cache);
  EnsureHost(v_cache);
  EnsureHost(slot_mapping);
  const int64_t num_slots = slot_mapping.shape[0];
  const int64_t block_size = k_cache.shape[1];
  const int64_t num_kv_heads = k_cache.shape[2];
  const int64_t head_size = k_cache.shape[3];
  const int64_t n_elems = num_kv_heads * head_size;
  VT_CHECK(k.shape[1] == num_kv_heads && k.shape[2] == head_size && v.shape[1] == num_kv_heads &&
               v.shape[2] == head_size,
           "tenstorrent kReshapeAndCache: k/v head shape must match cache");
  VT_CHECK(k.shape[0] >= num_slots && v.shape[0] >= num_slots,
           "tenstorrent kReshapeAndCache: token count must cover slots");
  // Contiguous NHD page: head stride == head_size (ops.cpp contract).
  VT_CHECK(k_cache.stride[2] == head_size && v_cache.stride[2] == head_size &&
               k_cache.stride[3] == 1 && v_cache.stride[3] == 1,
           "tenstorrent kReshapeAndCache: cache pages must be dense NHD");
  VT_CHECK(k.stride[2] == 1 && v.stride[2] == 1,
           "tenstorrent kReshapeAndCache: k/v innermost stride must be 1");

  const int64_t k_block_stride = k_cache.stride[0];
  const int64_t k_page_stride = k_cache.stride[1];
  const int64_t v_block_stride = v_cache.stride[0];
  const int64_t v_page_stride = v_cache.stride[1];
  const int64_t k_tok_stride = k.stride[0];
  const int64_t v_tok_stride = v.stride[0];
  const int64_t* slots = slot_mapping.Ptr<int64_t>();
  const size_t esz = SizeOf(k.dtype);
  const auto* ksrc = static_cast<const uint8_t*>(k.data);
  const auto* vsrc = static_cast<const uint8_t*>(v.data);
  auto* kdst = static_cast<uint8_t*>(k_cache.data);
  auto* vdst = static_cast<uint8_t*>(v_cache.data);
  const size_t bytes = static_cast<size_t>(n_elems) * esz;

  // Optional float staging for ttnn-mirror incremental patches (TILE-legal only).
  // Collect all valid slots then one batched device push (fill or multi-token update).
  const bool patch_mirror = (head_size % 32) == 0 && (block_size % 32) == 0;
  std::vector<uint32_t> rac_blocks, rac_offsets;
  std::vector<float> k_toks, v_toks;
  if (patch_mirror) {
    rac_blocks.reserve(static_cast<size_t>(num_slots));
    rac_offsets.reserve(static_cast<size_t>(num_slots));
    k_toks.reserve(static_cast<size_t>(num_slots) * static_cast<size_t>(n_elems));
    v_toks.reserve(static_cast<size_t>(num_slots) * static_cast<size_t>(n_elems));
  }

  for (int64_t t = 0; t < num_slots; ++t) {
    const int64_t slot = slots[t];
    if (slot < 0) continue;
    const int64_t block = slot / block_size;
    const int64_t offset = slot % block_size;
    const int64_t kdst_off = block * k_block_stride + offset * k_page_stride;
    const int64_t vdst_off = block * v_block_stride + offset * v_page_stride;
    std::memcpy(kdst + static_cast<size_t>(kdst_off) * esz,
                ksrc + static_cast<size_t>(t * k_tok_stride) * esz, bytes);
    std::memcpy(vdst + static_cast<size_t>(vdst_off) * esz,
                vsrc + static_cast<size_t>(t * v_tok_stride) * esz, bytes);
    if (patch_mirror) {
      rac_blocks.push_back(static_cast<uint32_t>(block));
      rac_offsets.push_back(static_cast<uint32_t>(offset));
      const size_t base = k_toks.size();
      k_toks.resize(base + static_cast<size_t>(n_elems));
      v_toks.resize(base + static_cast<size_t>(n_elems));
      for (int64_t i = 0; i < n_elems; ++i) {
        k_toks[base + static_cast<size_t>(i)] = LoadElemF32(k, t * k_tok_stride + i);
        v_toks[base + static_cast<size_t>(i)] = LoadElemF32(v, t * v_tok_stride + i);
      }
    }
  }
  CommitHost(k_cache);
  CommitHost(v_cache);
  if (patch_mirror) {
    if (!rac_blocks.empty()) {
      NotePagedKvRacWrites(k_cache, v_cache, rac_blocks, rac_offsets, k_toks, v_toks);
    }
  } else {
    DropPagedKvShadow(k_cache.data);
    DropPagedKvShadow(v_cache.data);
  }
}
namespace {


// Try pure-decode device PA via ttnn::paged_scaled_dot_product_attention_decode.
// Host keeps NHD; we upload a ttnn-layout [nb,nkv,bs,d] shadow (rebuilt when
// ReshapeAndCache dirties it). Returns true if `out` was written.
bool TryPagedAttentionDeviceDecode(Tensor& out, const Tensor& query, const Tensor& k_cache,
                                   const Tensor& v_cache, const Tensor& block_table,
                                   const Tensor& seq_lens, const Tensor& query_start_loc,
                                   const PagedAttentionArgs& args) {
  TT_OP_TRACE("TryPagedAttentionDeviceDecode");
  if (!args.causal || args.logits_soft_cap > 0.0f) return false;
  if (args.window_size.has_value()) return false;
  if (args.kv_cache_dtype != Fp8KVCacheDataType::kAuto) return false;
  if (query.rank != 3 || out.rank != 3 || k_cache.rank != 4 || v_cache.rank != 4) return false;
  if (!query.IsContiguous() || !out.IsContiguous()) return false;
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] TryPADecode entered cap=%d\n", (int)tt_capture_active());

  const int64_t total_q = query.shape[0];
  const int64_t hq = query.shape[1];
  const int64_t d = query.shape[2];
  const int64_t block_size = k_cache.shape[1];
  const int64_t nkv = k_cache.shape[2];
  if (d != k_cache.shape[3] || d != v_cache.shape[3]) return false;
  if (hq % nkv != 0) return false;
  // TILE constraints used by sdpa_decode validation.
  if ((d % 32) != 0 || (block_size % 32) != 0) return false;
  if (block_table.rank != 2 || seq_lens.rank != 1 || query_start_loc.rank != 1) return false;

  // Query may stay device-resident after rope — do not force EnsureHost(query).
  EnsureHost(block_table);
  EnsureHost(seq_lens);
  EnsureHost(query_start_loc);

  const int64_t num_reqs = seq_lens.shape[0];
  const int32_t* qsl = query_start_loc.Ptr<int32_t>();
  const int32_t* slens = seq_lens.Ptr<int32_t>();
  // Pure decode batch: one query token per request.
  if (total_q != num_reqs) return false;
  for (int64_t r = 0; r < num_reqs; ++r) {
    if (qsl[r + 1] - qsl[r] != 1) return false;
    if (slens[r] <= 0) return false;
  }

  const int64_t max_blocks = block_table.shape[1];
  const int32_t* btab = block_table.Ptr<int32_t>();
  const int64_t bt_row = block_table.stride[0], bt_col = block_table.stride[1];

  // page_table [B, max_blocks] + highest physical block id we must cover.
  std::vector<int32_t> pt(static_cast<size_t>(num_reqs * max_blocks));
  int32_t max_phys = -1;
  for (int64_t r = 0; r < num_reqs; ++r) {
    for (int64_t c = 0; c < max_blocks; ++c) {
      const int32_t id = btab[r * bt_row + c * bt_col];
      pt[static_cast<size_t>(r * max_blocks + c)] = id;
      if (id > max_phys) max_phys = id;
    }
  }
  if (max_phys < 0) return false;
  const uint32_t used_nb = static_cast<uint32_t>(max_phys) + 1u;
  if (static_cast<int64_t>(used_nb) > k_cache.shape[0]) return false;

  try {
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-TRACE] PA reached EnsurePagedKvTtnn cap=%d used_nb=%u\n", (int)tt_capture_active(), used_nb);

    MeshDevice& device = SharedMeshDevice();
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
      std::fprintf(stderr, "[TT-TRACE] PA EnsurePagedKvTtnn k used_nb=%u\n", used_nb);
    // Use the cached shadow when it exists (primed by WarmPagedKvShadow).
    // This skips EnsurePagedKvTtnn's from_vector upload AND its contiguous
    // check (KvSlice returns a non-contiguous strided view that the VT_CHECK
    // rejects). Needed on BOTH cold and capture steps so sdpa_decode compiles.
    ttnn::Tensor dev_k, dev_v;
    {
  std::lock_guard<std::mutex> g(PagedKvMutex());
      auto& sk = PagedKvShadows()[reinterpret_cast<uintptr_t>(k_cache.data)];
      auto& sv = PagedKvShadows()[reinterpret_cast<uintptr_t>(v_cache.data)];
      if (sk.device_current && sk.device.has_value() && sk.nb >= used_nb &&
          sv.device_current && sv.device.has_value() && sv.nb >= used_nb) {
        dev_k = *sk.device;
        dev_v = *sv.device;
        if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
          std::fprintf(stderr, "[TT-TRACE] PA using cached KV shadows (k_nb=%u v_nb=%u) cap=%d\n",
                       sk.nb, sv.nb, (int)tt_capture_active());
      } else if (tt_capture_active()) {
        throw std::runtime_error("PA: no KV shadow during capture");
      } else {
        if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
          std::fprintf(stderr, "[TT-TRACE] PA shadow miss: k_dc=%d k_dev=%d k_nb=%u/%u v_dc=%d v_dev=%d v_nb=%u/%u\n",
                       (int)sk.device_current, (int)sk.device.has_value(), sk.nb, used_nb,
                       (int)sv.device_current, (int)sv.device.has_value(), sv.nb, used_nb);
        // Cold step without shadow: fall through to EnsurePagedKvTtnn
        // (may fail on non-contiguous KvSlice; that's OK — the host path runs).
        g.~lock_guard();  // release before EnsurePagedKvTtnn
        dev_k = EnsurePagedKvTtnn(k_cache, device, used_nb);
        dev_v = EnsurePagedKvTtnn(v_cache, device, used_nb);
      }
    }
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
      std::fprintf(stderr, "[TT-TRACE] PA KV shadows OK, building page_table\n");

    const uint32_t Bu = static_cast<uint32_t>(num_reqs);
    const uint32_t hu = static_cast<uint32_t>(hq);
    const uint32_t du = static_cast<uint32_t>(d);

    // Q: [1, B, H, D]. Prefer reshape of a resident [B*H, D] / [B, H*D] shadow
    // (post device rope) so we never download then re-upload.
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-TRACE] PA before identity_q cap=%d total_q=%lld num_reqs=%lld qsl0=%d qsl1=%d\n",
                   (int)tt_capture_active(), (long long)total_q, (long long)num_reqs,
                   qsl[0], num_reqs > 0 ? qsl[1] : -1);
    bool identity_q = true;
    for (int64_t r = 0; r < num_reqs; ++r) {
      if (qsl[r] != r) {
        identity_q = false;
        break;
      }
    }
    ttnn::Tensor dev_q;
    bool q_from_device = false;
      if (identity_q) {
      try {
        // SINGLE code path (cold compiles exactly what capture replays):
        // resident [B*H, D] rope shadow → padding-correct 4D view → scalar
        // multiply materializes a native [1, B, H, D] TILE tensor. The old
        // Tensor::reshape view carried an UNPADDED spec (padded=logical) so
        // sdpa_decode mis-mapped the storage (capture read head0-only while
        // the cold host round-trip was correct); the explicit tile-padded
        // view's spec is identical to a native 4D tensor's.
        {
          Tensor q_flat = query.View({total_q * hq, d});
          ttnn::Tensor dev_q_2d = EnsureDevice2D(q_flat, device);
          // The 4D view's H dimension must NOT be tile-padded per batch:
          // the 2D tensor [B*H, D] packs heads contiguously (B*H rows in
          // one tile column), but padding H→32 per batch element would
          // require B*32 rows (2× the buffer for H=16, B=2). Instead,
          // use the 2D tensor's padded [rows, D] as a 4D padded shape
          // [1, 1, rows, D] — the total padded volume is the same
          // (rows*D = B*H*D), the last two dims are tile-aligned (rows
          // is B*H padded to 32, D is already aligned), and the buffer
          // is large enough. The logical [1, B, H, D] reinterprets the
          // same flat data; sdpa_decode reads per-batch head slices.
          //
          // That trick is only legal at B == 1: its padded dim 1 (1) sits
          // below the logical dim 1 (B) for any B > 1, and a padded shape
          // that covers [1, B, pad32(H), D] exceeds the buffer —
          // tt::tt_metal::view then fatals "MeshBuffer must be large
          // enough to hold the tensor" on every EAGER batched decode step
          // (concurrency >= 2; ISSUE-LOCAL-01M2X9XSS0N7WB5BTJ0326B892 —
          // the fatal was silently caught and the op fell to the host Q
          // path). No metadata view of the [rows, D] buffer represents
          // the per-batch head tiling at B > 1, so materialize the
          // correct [1, B, H, D] TILE tensor through the free reshape's
          // device program. That program calls to_device — forbidden
          // during trace capture — so a captured batched step takes the
          // host Q path (the contract the old fatal enforced, loudly).
          const auto ps2d = dev_q_2d.padded_shape();
          (void)ps2d;
          if (tt_capture_active() && Bu > 1)
            throw std::runtime_error(
                "tenstorrent PA: batched (B>1) Q 4D materialization is "
                "not capture-safe; the host Q path must serve this step");
          if (Bu == 1) {
            dev_q = ttnn::multiply(
                ttnn::experimental::view(
                    dev_q_2d, ttnn::Shape({1u, Bu, hu, du}),
                    ttnn::Shape({1u, 1u, ps2d[0], ps2d[1]})),
                1.0f);
          } else {
            dev_q = ttnn::multiply(
                ttnn::reshape(dev_q_2d, ttnn::Shape({1u, Bu, hu, du})),
                1.0f);
          }
        }
        // sdpa_decode requires bf16 (the host arm below builds bf16 too).
        // The rope shadow behind this view is f32 for heads outside every
        // FA2 decode lane (e.g. the 0.8B head ratio), so cast the
        // materialized tile in place of the host round-trip.
        if (dev_q.dtype() != ttnn::DataType::BFLOAT16) {
          dev_q = ttnn::typecast(dev_q, ttnn::DataType::BFLOAT16);
        }
        // Shard if needed
        if (std::getenv("VT_TT_SHARD_Q") != nullptr) {
          const uint32_t padded_hq = std::max(32u, hu);
          const auto q_grid = device.compute_with_storage_grid_size();
          const tt::tt_metal::CoreRangeSet q_core_set =
              tt::tt_metal::num_cores_to_corerangeset(Bu, q_grid, true);
          tt::tt_metal::ShardSpec q_ss(q_core_set, {padded_hq, du},
                                       tt::tt_metal::ShardOrientation::ROW_MAJOR);
          tt::tt_metal::MemoryConfig q_mc(
              tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
              tt::tt_metal::BufferType::L1, q_ss);
          dev_q = ttnn::to_memory_config(dev_q, q_mc);
        }
        q_from_device = true;
        if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
          std::fprintf(stderr, "[TT-TRACE] PA q_from_device OK cap=%d\n", (int)tt_capture_active());
      } catch (const std::exception& e) {
        if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
          std::fprintf(stderr, "[TT-TRACE] PA q_from_device FAILED: %s\n", e.what());
        q_from_device = false;
      }
    }
    if (!q_from_device) {
      if (tt_capture_active()) {
        VT_CHECK(false, "tenstorrent: PA Q host path is not capture-safe "
                        "(from_vector readback); the resident rope shadow "
                        "must be used during capture");
      }
      EnsureHost(query);
      std::vector<float> q_host(static_cast<size_t>(num_reqs * hq * d));
      for (int64_t r = 0; r < num_reqs; ++r) {
        const int64_t t = qsl[r];
        for (int64_t h = 0; h < hq; ++h) {
          for (int64_t e = 0; e < d; ++e) {
            q_host[static_cast<size_t>((r * hq + h) * d + e)] =
                LoadElemF32(query, (t * hq + h) * d + e);
          }
        }
      }
      if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
        std::fprintf(stderr, "[TT-UP] TryPagedAttentionDeviceDecode from_vector WRITE during capture\n");
      dev_q = ttnn::Tensor::from_vector<float>(
          q_host, SpecOf(tt::tt_metal::Shape({1u, Bu, hu, du}), ttnn::DataType::BFLOAT16,
                         ttnn::Layout::TILE),
          &device);
    }
    ttnn::Tensor dev_pt, dev_pos;
    bool use_warm_meta = false;
    {
      std::lock_guard<std::mutex> g(PaMetaMutex());
      const auto pkey = std::make_pair(static_cast<int64_t>(num_reqs),
                                       static_cast<int64_t>(max_blocks));
      auto it = PaMetaCache().find(pkey);
      if (it != PaMetaCache().end() && it->second.allocated) {
        dev_pt = it->second.page_table;
        dev_pos = it->second.cur_pos;
        const int32_t expect_cp = slens[0] - 1;  // WarmPaMeta stores seq_lens - 1
        VT_CHECK(it->second.cp_host.size() >= 1 && it->second.cp_host[0] == expect_cp,
                 "tenstorrent: PA meta not warmed for this step");
        use_warm_meta = true;
        if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
          std::fprintf(stderr, "[TT-TRACE] PA using cached meta (pt+cp) cap=%d\n",
                       (int)tt_capture_active());
      }
    }
    if (!use_warm_meta) {
      // Trimmed inline page_table ([B, max_blocks_per_seq] like the
      // upstream test): sdpa_decode mis-executes with a wide [B, 256]
      // table when only a few blocks are allocated.
      VT_CHECK(!tt_capture_active(),
               "tenstorrent: PA meta not warmed for this step");
      const int64_t max_vblk = (num_reqs > 0 && slens[0] > 0)
          ? (slens[0] - 1) / block_size + 1 : 1;
      const int64_t pt_cols = std::min(max_blocks, std::max<int64_t>(2, max_vblk));
      std::vector<int32_t> pt_trim(static_cast<size_t>(Bu * pt_cols));
      for (int64_t r = 0; r < num_reqs; ++r) {
        for (int64_t c = 0; c < pt_cols; ++c) {
          pt_trim[static_cast<size_t>(r * pt_cols + c)] =
              pt[static_cast<size_t>(r * max_blocks + c)];
        }
      }
      dev_pt = ttnn::Tensor::from_vector<int32_t>(
          pt_trim, SpecOf(tt::tt_metal::Shape({Bu, static_cast<uint32_t>(pt_cols)}),
                      ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR),
          &device);
      std::vector<int32_t> cpos(static_cast<size_t>(num_reqs));
      for (int64_t r = 0; r < num_reqs; ++r) cpos[static_cast<size_t>(r)] = slens[r] - 1;
      dev_pos = ttnn::Tensor::from_vector<int32_t>(
          cpos, SpecOf(tt::tt_metal::Shape({Bu}), ttnn::DataType::INT32,
                       ttnn::Layout::ROW_MAJOR),
          &device);
    }

    // DON'T pass program_config — let sdpa_decode use its default.
    // Our explicit config may interact badly with the program cache when
    // called after other ops in the model forward.

    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-TRACE] PA calling sdpa_decode cap=%d\n", (int)tt_capture_active());
    // Don't pass compute_kernel_config — let sdpa_decode use its default (HiFi2).
    // Our HiFi4 was needed for nkv>1 correctness, but the 2-head issue is separate.
     // Debug: when VT_TT_SDPA_TEST is set, create FRESH Q/KV/pt/pos from
    // scratch (random data, all heads populated) and call sdpa_decode.
    // This tests whether sdpa_decode works inside the model forward context
    // with tensors created the same way as the Python standalone test.
    if (std::getenv("VT_TT_SDPA_TEST") != nullptr && !tt_capture_active()) {
      static bool tested = false;
      if (!tested) {
        tested = true;
        fprintf(stderr, "[TT-SDPA-TEST] Running standalone sdpa_decode test inside model forward...\n");
        // Create fresh Q: [1,1,16,128] with random data in ALL heads
        std::vector<float> test_q(16 * 128);
        for (auto& v : test_q) v = static_cast<float>(rand()) / 2147483647.0f * 2.0f - 1.0f;
        ttnn::Tensor test_q_rm = ttnn::Tensor::from_vector<float>(test_q,
            SpecOf(tt::tt_metal::Shape({1u, 1u, 16u, 128u}), ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR),
            nullptr);
        ttnn::Tensor test_q_dev = test_q_rm.to_device(&device, tt::tt_metal::MemoryConfig{});
        ttnn::Tensor test_q_tile = ttnn::to_layout(test_q_dev, ttnn::Layout::TILE);
        // Create fresh KV: [2,8,32,128] with data in block 1, offset 0
        std::vector<float> test_k(2*8*32*128, 0.0f), test_v(2*8*32*128, 0.0f);
        for (uint32_t h = 0; h < 8; h++)
          for (uint32_t e = 0; e < 128; e++) {
            size_t off = (1*8*32 + h*32 + 0) * 128 + e;
            test_k[off] = static_cast<float>(rand()) / 2147483647.0f * 2.0f - 1.0f;
            test_v[off] = static_cast<float>(rand()) / 2147483647.0f * 2.0f - 1.0f;
          }
        ttnn::Tensor test_k_rm = ttnn::Tensor::from_vector<float>(test_k,
            SpecOf(tt::tt_metal::Shape({2u,8u,32u,128u}), ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR), nullptr);
        ttnn::Tensor test_k_dev = test_k_rm.to_device(&device, tt::tt_metal::MemoryConfig{});
        ttnn::Tensor test_k_tile = ttnn::to_layout(test_k_dev, ttnn::Layout::TILE);
        ttnn::Tensor test_v_rm = ttnn::Tensor::from_vector<float>(test_v,
            SpecOf(tt::tt_metal::Shape({2u,8u,32u,128u}), ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR), nullptr);
        ttnn::Tensor test_v_dev = test_v_rm.to_device(&device, tt::tt_metal::MemoryConfig{});
        ttnn::Tensor test_v_tile = ttnn::to_layout(test_v_dev, ttnn::Layout::TILE);
        // page_table: [1,2] = [1,0]
        std::vector<int32_t> test_pt = {1, 0};
        ttnn::Tensor test_pt_dev = ttnn::Tensor::from_vector<int32_t>(test_pt,
            SpecOf(tt::tt_metal::Shape({1u,2u}), ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR), &device);
        // cur_pos: [1] = [0]
        std::vector<int32_t> test_pos = {0};
        ttnn::Tensor test_pos_dev = ttnn::Tensor::from_vector<int32_t>(test_pos,
            SpecOf(tt::tt_metal::Shape({1u}), ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR), &device);
        // Call sdpa_decode with FRESH Q + model KV/pt/pos
        // But first check if dev_k has data at block 1
        {
          auto k_check = dev_k.to_vector<float>();
          const uint32_t k_nkv = dev_k.logical_shape()[1];
          const uint32_t k_bs = dev_k.logical_shape()[2];
          const uint32_t k_d = dev_k.logical_shape()[3];
          size_t b1_off = (1 * k_nkv * k_bs + 0 * k_bs + 0) * k_d;
          fprintf(stderr, "[TT-SDPA-TEST] dev_k block1: [%f,%f,%f,%f] (off=%zu/%zu)\n",
                  k_check.size()>b1_off?k_check[b1_off]:0,
                  k_check.size()>b1_off+1?k_check[b1_off+1]:0,
                  k_check.size()>b1_off+2?k_check[b1_off+2]:0,
                  k_check.size()>b1_off+3?k_check[b1_off+3]:0,
                  b1_off, k_check.size());
          // Check V at block 1 for ALL 8 KV heads
          auto v_check = dev_v.to_vector<float>();
          const uint32_t v_nkv = dev_v.logical_shape()[1];
          const uint32_t v_bs = dev_v.logical_shape()[2];
          const uint32_t v_d = dev_v.logical_shape()[3];
          for (uint32_t h = 0; h < v_nkv; h++) {
            size_t voff = (1 * v_nkv * v_bs + h * v_bs + 0) * v_d;
            fprintf(stderr, "[TT-SDPA-TEST] dev_v block1 head%u: [%f,%f,%f,%f]\n", h,
                    v_check.size()>voff?v_check[voff]:0,
                    v_check.size()>voff+1?v_check[voff+1]:0,
                    v_check.size()>voff+2?v_check[voff+2]:0,
                    v_check.size()>voff+3?v_check[voff+3]:0);
          }
          // Also check if model KV WITHOUT RAC works: create fresh KV via
          // to_layout(TILE) with the SAME data as dev_k
          auto k_vec = dev_k.to_vector<float>();
          auto v_vec = dev_v.to_vector<float>();
          ttnn::Tensor fresh_k_rm = ttnn::Tensor::from_vector<float>(k_vec,
              SpecOf(tt::tt_metal::Shape({2u,8u,32u,128u}), ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR), nullptr);
          ttnn::Tensor fresh_k_dev = fresh_k_rm.to_device(&device, tt::tt_metal::MemoryConfig{});
          ttnn::Tensor fresh_k_tile = ttnn::to_layout(fresh_k_dev, ttnn::Layout::TILE);
          ttnn::Tensor fresh_v_rm = ttnn::Tensor::from_vector<float>(v_vec,
              SpecOf(tt::tt_metal::Shape({2u,8u,32u,128u}), ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR), nullptr);
          ttnn::Tensor fresh_v_dev = fresh_v_rm.to_device(&device, tt::tt_metal::MemoryConfig{});
          ttnn::Tensor fresh_v_tile = ttnn::to_layout(fresh_v_dev, ttnn::Layout::TILE);
          // Call sdpa_decode with fresh Q + fresh-KV-from-model-data
          ttnn::Tensor fresh_out = ttnn::transformer::paged_scaled_dot_product_attention_decode(
              test_q_tile, fresh_k_tile, fresh_v_tile, dev_pt,
              true, std::nullopt, dev_pos, std::nullopt,
              1.0f/std::sqrt(128.0f), std::nullopt, std::nullopt,
              std::nullopt, std::nullopt, std::nullopt, std::nullopt);
          auto fresh_out_vec = fresh_out.to_vector<float>();
          int fresh_nonzero = 0;
          for (uint32_t h = 0; h < 16; h++) {
            size_t off = static_cast<size_t>(h) * 128;
            float maxval = 0;
            for (size_t i = off; i < off + 128 && i < fresh_out_vec.size(); i++)
              maxval = std::max(maxval, std::abs(fresh_out_vec[i]));
            if (maxval > 0.001f) fresh_nonzero++;
          }
          fprintf(stderr, "[TT-SDPA-TEST] fresh-KV-from-model-data: %d/16 heads\n", fresh_nonzero);
        }
        ttnn::Tensor test_out = ttnn::transformer::paged_scaled_dot_product_attention_decode(
            test_q_tile, dev_k, dev_v, dev_pt,
            true, std::nullopt, test_pos_dev, std::nullopt,
            1.0f/std::sqrt(128.0f), std::nullopt, std::nullopt,
            std::nullopt, std::nullopt, std::nullopt, std::nullopt);
        auto test_out_vec = test_out.to_vector<float>();
        int non_zero = 0;
        for (uint32_t h = 0; h < 16; h++) {
          size_t off = static_cast<size_t>(h) * 128;
          float maxval = 0;
          for (size_t i = off; i < off + 128 && i < test_out_vec.size(); i++)
            maxval = std::max(maxval, std::abs(test_out_vec[i]));
          if (maxval > 0.001f) non_zero++;
          if (h % 2 == 0)
            fprintf(stderr, "[TT-SDPA-TEST] head%u: max=%.4f %s\n", h, maxval, maxval > 0.001f ? "OK" : "ZERO");
        }
        fprintf(stderr, "[TT-SDPA-TEST] Non-zero heads: %d/16\n", non_zero);
      }
    }
    // Dump all shapes right before sdpa_decode
    ttnn::Tensor dev_out = ttnn::transformer::paged_scaled_dot_product_attention_decode(
        dev_q, dev_k, dev_v, dev_pt,
        /*is_causal=*/true,
        /*attn_mask=*/std::nullopt,
        /*cur_pos_tensor=*/dev_pos,
        /*attention_sink=*/std::nullopt,
        /*scale=*/args.scale,
        /*sliding_window_size=*/std::nullopt,
        /*memory_config=*/tt::tt_metal::MemoryConfig{},
        /*program_config=*/std::nullopt,
        /*compute_kernel_config=*/std::nullopt,
        /*paged_cache_geometry=*/std::nullopt,
        /*cache_position_modulo=*/std::nullopt);

    // Dump PA output for comparison (first layer, cold step)
    {
    }

    // Prefer keeping activations on device for o_proj: flatten to [B, H*D].
    // Pure-decode with identity token order (qsl[r]==r) matches out's storage
    // layout [T,H,D] == [B,H,D] so Reshape→MatmulBT hits EnsureDevice2D.
    bool identity_order = true;
    for (int64_t r = 0; r < num_reqs; ++r) {
      if (qsl[r] != r) {
        identity_order = false;
        break;
      }
    }
    if (identity_order && total_q == num_reqs) {
      try {
        const uint32_t flat_cols = static_cast<uint32_t>(hq * d);
        // B == 1: CaptureSafeReshape's metadata view (padded volume equals
        // the buffer) keeps the captured decode replay capture-safe. B > 1:
        // no metadata view of the [1, B, pad32(H), D] buffer can represent
        // [B, H*D] — a covering padded shape exceeds the buffer, the
        // "MeshBuffer must be large enough" fatal of
        // ISSUE-LOCAL-01M2X9XSS0N7WB5BTJ0326B892 — so materialize the
        // flattened tensor through the free reshape's device program; a
        // captured batched step falls back to the host scatter below.
        ttnn::Tensor flat =
            tt_capture_active()
                ? CaptureSafeReshape(dev_out, ttnn::Shape({Bu, flat_cols}))
                : ttnn::reshape(dev_out, ttnn::Shape({Bu, flat_cols}));
        CommitDeviceLogical2D(out, std::move(flat), Bu, flat_cols);
        // Verify the committed output matches the PA output
        {
        }
        return true;
      } catch (const std::exception&) {
        // Fall through to host materialization.
      }
    }

    // Output ~ [1, B, H, D] → host [B, H, D] in request order, then scatter to
    // global query token indices.
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-TRACE] PA decode to_vector\n");
    std::vector<float> result = dev_out.to_vector<float>();
    VT_CHECK(static_cast<int64_t>(result.size()) >= num_reqs * hq * d,
             "tenstorrent device PA: unexpected output size");
    for (int64_t r = 0; r < num_reqs; ++r) {
      const int64_t t = qsl[r];
      for (int64_t h = 0; h < hq; ++h) {
        for (int64_t e = 0; e < d; ++e) {
          const float v = result[static_cast<size_t>((r * hq + h) * d + e)];
          StoreElemF32(out, (t * hq + h) * d + e, v);
        }
      }
    }
    CommitHost(out);
    return true;
  } catch (const std::exception& e) {
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-TRACE] PA device decode FAILED: %s\n", e.what());
    // Fall back to host oracle (shape/grid/dtype edge cases).
    return false;
  }
}

// Multi-token (prefill) device PA via ttnn::chunked_scaled_dot_product_attention.
// Each request is processed independently (B=1) so seq lengths need not match.
// Query tokens for request r cover absolute positions [seq_len-q_len, seq_len);
// chunk_start must be a multiple of the program q/k chunk size (32).
// Pads the last Q chunk with zeros (causal → pad queries do not affect earlier
// real positions). Returns true if every request was written.
bool TryPagedAttentionDevicePrefill(Tensor& out, const Tensor& query, const Tensor& k_cache,
                                    const Tensor& v_cache, const Tensor& block_table,
                                    const Tensor& seq_lens, const Tensor& query_start_loc,
                                    const PagedAttentionArgs& args) {
  if (!args.causal || args.logits_soft_cap > 0.0f) return false;
  if (args.window_size.has_value()) return false;
  if (args.kv_cache_dtype != Fp8KVCacheDataType::kAuto) return false;
  if (query.rank != 3 || out.rank != 3 || k_cache.rank != 4 || v_cache.rank != 4) return false;
  if (!query.IsContiguous() || !out.IsContiguous()) return false;

  const int64_t total_q = query.shape[0];
  const int64_t hq = query.shape[1];
  const int64_t d = query.shape[2];
  const int64_t block_size = k_cache.shape[1];
  const int64_t nkv = k_cache.shape[2];
  if (d != k_cache.shape[3] || d != v_cache.shape[3]) return false;
  if (hq % nkv != 0) return false;
  if ((d % 32) != 0 || (block_size % 32) != 0) return false;
  if (block_table.rank != 2 || seq_lens.rank != 1 || query_start_loc.rank != 1) return false;

  EnsureHost(query);
  EnsureHost(k_cache);
  EnsureHost(v_cache);
  EnsureHost(block_table);
  EnsureHost(seq_lens);
  EnsureHost(query_start_loc);

  const int64_t num_reqs = seq_lens.shape[0];
  const int32_t* qsl = query_start_loc.Ptr<int32_t>();
  const int32_t* slens = seq_lens.Ptr<int32_t>();
  if (qsl[num_reqs] != total_q) return false;

  // Chunk size must divide TILE and match program_config (decode uses 32 too).
  constexpr int64_t kChunk = 32;
  bool any_multi = false;
  for (int64_t r = 0; r < num_reqs; ++r) {
    const int64_t q_len = static_cast<int64_t>(qsl[r + 1] - qsl[r]);
    const int64_t seq = static_cast<int64_t>(slens[r]);
    if (q_len <= 0 || seq < q_len) return false;
    const int64_t chunk_start = seq - q_len;
    if ((chunk_start % kChunk) != 0) return false;
    if (q_len > 1) any_multi = true;
  }
  // Pure-decode batches stay on the decode SDPA path.
  if (!any_multi) return false;

  const int64_t max_blocks = block_table.shape[1];
  const int32_t* btab = block_table.Ptr<int32_t>();
  const int64_t bt_row = block_table.stride[0], bt_col = block_table.stride[1];

  int32_t max_phys = -1;
  for (int64_t r = 0; r < num_reqs; ++r) {
    for (int64_t c = 0; c < max_blocks; ++c) {
      const int32_t id = btab[r * bt_row + c * bt_col];
      if (id > max_phys) max_phys = id;
    }
  }
  if (max_phys < 0) return false;
  const uint32_t used_nb = static_cast<uint32_t>(max_phys) + 1u;
  if (static_cast<int64_t>(used_nb) > k_cache.shape[0]) return false;

  try {
    MeshDevice& device = SharedMeshDevice();
    ttnn::Tensor dev_k = EnsurePagedKvTtnn(k_cache, device, used_nb);
    ttnn::Tensor dev_v = EnsurePagedKvTtnn(v_cache, device, used_nb);

    const auto grid = device.compute_with_storage_grid_size();
    ttnn::operations::transformer::SDPAProgramConfig prog{
        grid,
        std::nullopt,
        /*q_chunk_size=*/static_cast<uint32_t>(kChunk),
        /*k_chunk_size=*/static_cast<uint32_t>(kChunk),
        /*exp_approx_mode=*/false,
        /*max_cores_per_head_batch=*/16};

    const uint32_t hu = static_cast<uint32_t>(hq);
    const uint32_t du = static_cast<uint32_t>(d);

    // Single-request contiguous prefill (qsl[0]==0, q_len==total_q): keep
    // result on device as [T, H*D] for o_proj. Multi-chunk uses permute+concat
    // so we never host-materialize the full sequence.
    const bool device_prefill_out =
        num_reqs == 1 && qsl[0] == 0 &&
        static_cast<int64_t>(qsl[1] - qsl[0]) == total_q && total_q > 0;

    std::vector<ttnn::Tensor> out_pieces;  // each [n_real, H*D]
    out_pieces.reserve(static_cast<size_t>((total_q + kChunk - 1) / kChunk));

    for (int64_t r = 0; r < num_reqs; ++r) {
      const int64_t q_begin = static_cast<int64_t>(qsl[r]);
      const int64_t q_len = static_cast<int64_t>(qsl[r + 1] - qsl[r]);
      const int64_t seq = static_cast<int64_t>(slens[r]);
      const int64_t chunk_start0 = seq - q_len;

      // Page table for this request: [1, max_blocks].
      std::vector<int32_t> pt(static_cast<size_t>(max_blocks));
      for (int64_t c = 0; c < max_blocks; ++c) {
        pt[static_cast<size_t>(c)] = btab[r * bt_row + c * bt_col];
      }
      // KV address space must cover seq (and any Q pad to kChunk).
      const int64_t q_pad = ((q_len + kChunk - 1) / kChunk) * kChunk;
      const int64_t need_kv = chunk_start0 + q_pad;
      if (max_blocks * block_size < need_kv) return false;

      if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
        std::fprintf(stderr, "[TT-UP] TryPagedAttentionDevicePrefill from_vector WRITE during capture\n");
      ttnn::Tensor dev_pt = ttnn::Tensor::from_vector<int32_t>(
          pt, SpecOf(tt::tt_metal::Shape({1u, static_cast<uint32_t>(max_blocks)}),
                     ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR),
          &device);

      for (int64_t local = 0; local < q_len; local += kChunk) {
        const int64_t n_real = std::min(kChunk, q_len - local);
        const int64_t chunk_start = chunk_start0 + local;
        // Pack Q chunk [1, H, kChunk, D] (pad tail with zeros).
        std::vector<float> q_host(static_cast<size_t>(hq) * static_cast<size_t>(kChunk) * d, 0.0f);
        for (int64_t i = 0; i < n_real; ++i) {
          const int64_t t = q_begin + local + i;
          for (int64_t h = 0; h < hq; ++h) {
            for (int64_t e = 0; e < d; ++e) {
              q_host[static_cast<size_t>((h * kChunk + i) * d + e)] =
                  LoadElemF32(query, (t * hq + h) * d + e);
            }
          }
        }
        if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
          std::fprintf(stderr, "[TT-UP] TryPagedAttentionDevicePrefill from_vector WRITE during capture\n");
        ttnn::Tensor dev_q = ttnn::Tensor::from_vector<float>(
            q_host,
            SpecOf(tt::tt_metal::Shape({1u, hu, static_cast<uint32_t>(kChunk), du}),
                   ttnn::DataType::BFLOAT16, ttnn::Layout::TILE),
            &device);

        ttnn::Tensor dev_out = ttnn::transformer::chunked_scaled_dot_product_attention(
            dev_q, dev_k, dev_v, dev_pt, chunk_start, /*scale=*/args.scale,
            /*memory_config=*/std::nullopt, /*program_config=*/prog,
            /*compute_kernel_config=*/std::nullopt, /*paged_cache_geometry=*/std::nullopt);

        if (device_prefill_out) {
          // [1, H, S, D] → [1, S, H, D] → slice real tokens → [n_real, H*D].
          ttnn::Tensor perm =
              ttnn::permute(dev_out, ttsl::SmallVector<int64_t>{0, 2, 1, 3});
          if (n_real < kChunk) {
            const uint32_t nr = static_cast<uint32_t>(n_real);
            perm = ttnn::slice(perm, ttsl::SmallVector<uint32_t>{0, 0, 0, 0},
                               ttsl::SmallVector<uint32_t>{1u, nr, hu, du},
                               ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
          }
          const uint32_t nr = static_cast<uint32_t>(n_real);
          out_pieces.push_back(
              ttnn::reshape(perm, ttnn::Shape({nr, static_cast<uint32_t>(hq * d)})));
          continue;
        }

        if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
          std::fprintf(stderr, "[TT-TRACE] PA prefill to_vector\n");
        std::vector<float> result = dev_out.to_vector<float>();
        // Expected dense logical [1, H, kChunk, D].
        VT_CHECK(static_cast<int64_t>(result.size()) >= hq * kChunk * d,
                 "tenstorrent device prefill PA: unexpected output size");
        for (int64_t i = 0; i < n_real; ++i) {
          const int64_t t = q_begin + local + i;
          for (int64_t h = 0; h < hq; ++h) {
            for (int64_t e = 0; e < d; ++e) {
              const float v = result[static_cast<size_t>((h * kChunk + i) * d + e)];
              StoreElemF32(out, (t * hq + h) * d + e, v);
            }
          }
        }
      }
    }

    if (device_prefill_out && !out_pieces.empty()) {
      ttnn::Tensor flat = out_pieces.size() == 1
                              ? std::move(out_pieces[0])
                              : ttnn::concat(out_pieces, /*dim=*/0);
      CommitDeviceLogical2D(out, std::move(flat), static_cast<uint32_t>(total_q),
                            static_cast<uint32_t>(hq * d));
      return true;
    }
    CommitHost(out);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

// kPagedAttention: causal/non-causal GQA softmax over the paged NHD cache
// (cpu_paged_attn.cpp PagedAttentionKernel). Host-staged f32 oracle matching
// the CPU reference while Alloc is host memory.
//
// Device paths (ttnn-layout K/V shadows [nb,nkv,bs,d]):
//   * pure decode → paged_scaled_dot_product_attention_decode
//   * multi-token prefill → chunked_scaled_dot_product_attention (per request)
// Host NHD stays the source of truth for ReshapeAndCache / LMCache plane layout.
//
// Host perf levers:
//   * NEON FMA dots/axpy + Q hoist + specialized f32/bf16 loads
//   * Short single-req sequences: gather pages → dense [seq,nkv,d] (capped) for
//     sequential inner loops / GQA reuse
//   * Prefill (T*H ≥ 64): small dedicated 4–16 thread pool (not 128-core global)
//
// Correctness: each (t,h) writes a disjoint out row with the same j-order and
// max-subtracted softmax as the serial path → bit-identical to n_threads==1.
//
// This step supports: F32/BF16/F16 query/cache, f32/bf16 out, kAuto KV (no
// fp8), optional softcap and window_size (same math as CPU). OPT-125m uses
// causal + full window + no softcap.

}  // namespace
void PagedAttentionKernel(Queue&, Tensor& out, const Tensor& query, const Tensor& k_cache,
                          const Tensor& v_cache, const Tensor& block_table,
                          const Tensor& seq_lens, const Tensor& query_start_loc,
                          const PagedAttentionArgs& args) {
  // Device SDPA (ttnn-layout shadows). Softcap / window / odd TILE dims /
  // misaligned chunk starts fall through to the host oracle.
  if (TryPagedAttentionDeviceDecode(out, query, k_cache, v_cache, block_table, seq_lens,
                                    query_start_loc, args)) {
    return;
  }
  if (TryPagedAttentionDevicePrefill(out, query, k_cache, v_cache, block_table, seq_lens,
                                     query_start_loc, args)) {
    return;
  }

  EnsureHost(query);
  EnsureHost(k_cache);
  EnsureHost(v_cache);
  EnsureHost(block_table);
  EnsureHost(seq_lens);
  EnsureHost(query_start_loc);
  VT_CHECK(query.rank == 3 && out.rank == 3 && k_cache.rank == 4 && v_cache.rank == 4,
           "tenstorrent kPagedAttention: query/out rank-3, caches rank-4");
  VT_CHECK(IsFloatDType(query.dtype) && (out.dtype == DType::kF32 || out.dtype == DType::kBF16) &&
               IsFloatDType(k_cache.dtype) && k_cache.dtype == v_cache.dtype,
           "tenstorrent kPagedAttention: float query/cache, f32/bf16 out");
  VT_CHECK(args.kv_cache_dtype == Fp8KVCacheDataType::kAuto,
           "tenstorrent kPagedAttention: fp8 KV not supported in this step");
  VT_CHECK(args.scale > 0.0f, "tenstorrent kPagedAttention: scale must be > 0");
  VT_CHECK(query.IsContiguous() && out.IsContiguous() && seq_lens.IsContiguous() &&
               query_start_loc.IsContiguous(),
           "tenstorrent kPagedAttention: query/out/seq_lens/query_start_loc contiguous");

  const int64_t total_q = query.shape[0];
  const int64_t hq = query.shape[1];
  const int64_t d = query.shape[2];
  const int64_t block_size = k_cache.shape[1];
  const int64_t num_kv_heads = k_cache.shape[2];
  VT_CHECK(d == k_cache.shape[3], "tenstorrent kPagedAttention: head_size mismatch");
  VT_CHECK(hq % num_kv_heads == 0, "tenstorrent kPagedAttention: GQA ratio");
  const int64_t qpk = hq / num_kv_heads;
  const float scale = args.scale;
  const float softcap = args.logits_soft_cap;
  const int64_t window_left = args.window_size.has_value() ? args.window_size->left : -1;
  const int64_t window_right = args.window_size.has_value() ? args.window_size->right : -1;

  const int64_t num_reqs = seq_lens.shape[0];
  const int32_t* qsl = query_start_loc.Ptr<int32_t>();
  const int32_t* slens = seq_lens.Ptr<int32_t>();
  const int32_t* btab = block_table.Ptr<int32_t>();
  const int64_t bt_row = block_table.stride[0], bt_col = block_table.stride[1];
  const int64_t kc_blk = k_cache.stride[0], kc_pg = k_cache.stride[1], kc_hd = k_cache.stride[2];
  const int64_t vc_blk = v_cache.stride[0], vc_pg = v_cache.stride[1], vc_hd = v_cache.stride[2];

  std::vector<int32_t> tok_pos(static_cast<size_t>(total_q));
  std::vector<int32_t> tok_slen(static_cast<size_t>(total_q));
  std::vector<int32_t> tok_req(static_cast<size_t>(total_q));
  for (int64_t r = 0; r < num_reqs; ++r) {
    const int64_t q0 = qsl[r], q1 = qsl[r + 1];
    const int64_t query_len = q1 - q0;
    if (query_len <= 0) continue;
    const int64_t seqlen = slens[r];
    const int64_t context = seqlen - query_len;
    for (int64_t local = 0; local < query_len; ++local) {
      tok_pos[static_cast<size_t>(q0 + local)] = static_cast<int32_t>(context + local);
      tok_slen[static_cast<size_t>(q0 + local)] = static_cast<int32_t>(seqlen);
      tok_req[static_cast<size_t>(q0 + local)] = static_cast<int32_t>(r);
    }
  }

  // F32 contiguous inner kernels (NEON on aarch64). Bit-identical to scalar
  // for normal finite scores (same j order / max-subtracted softmax).
  auto dot_f32 = [](const float* a, const float* b, int64_t n) -> float {
#if defined(__aarch64__)
    float32x4_t vacc = vdupq_n_f32(0.0f);
    int64_t e = 0;
    for (; e + 4 <= n; e += 4) {
      vacc = vfmaq_f32(vacc, vld1q_f32(a + e), vld1q_f32(b + e));
    }
    float sum = vaddvq_f32(vacc);
    for (; e < n; ++e) sum += a[e] * b[e];
    return sum;
#else
    float sum = 0.0f;
    for (int64_t e = 0; e < n; ++e) sum += a[e] * b[e];
    return sum;
#endif
  };
  auto axpy_f32 = [](float* acc, const float* v, float pw, int64_t n) {
#if defined(__aarch64__)
    const float32x4_t vp = vdupq_n_f32(pw);
    int64_t e = 0;
    for (; e + 4 <= n; e += 4) {
      float32x4_t a = vld1q_f32(acc + e);
      a = vfmaq_f32(a, vp, vld1q_f32(v + e));
      vst1q_f32(acc + e, a);
    }
    for (; e < n; ++e) acc[e] += pw * v[e];
#else
    for (int64_t e = 0; e < n; ++e) acc[e] += pw * v[e];
#endif
  };

  // Contiguous float loaders — dtype is fixed for the whole kernel call.
  // Captured bases avoid per-element Tensor::Ptr + switch in the hot loop.
  const DType q_dt = query.dtype, k_dt = k_cache.dtype, v_dt = v_cache.dtype,
              o_dt = out.dtype;
  const float* q_f = (q_dt == DType::kF32) ? query.Ptr<float>() : nullptr;
  const uint16_t* q_h =
      (q_dt == DType::kBF16 || q_dt == DType::kF16) ? query.Ptr<uint16_t>() : nullptr;
  const float* k_f = (k_dt == DType::kF32) ? k_cache.Ptr<float>() : nullptr;
  const uint16_t* k_h =
      (k_dt == DType::kBF16 || k_dt == DType::kF16) ? k_cache.Ptr<uint16_t>() : nullptr;
  const float* v_f = (v_dt == DType::kF32) ? v_cache.Ptr<float>() : nullptr;
  const uint16_t* v_h =
      (v_dt == DType::kBF16 || v_dt == DType::kF16) ? v_cache.Ptr<uint16_t>() : nullptr;
  float* o_f = (o_dt == DType::kF32) ? out.Ptr<float>() : nullptr;
  uint16_t* o_h = (o_dt == DType::kBF16) ? out.Ptr<uint16_t>() : nullptr;

  auto load_half = [](DType dt, const uint16_t* base, int64_t i) -> float {
    return dt == DType::kBF16 ? BF16ToF32(base[i]) : F16ToF32(base[i]);
  };

  // Work unit = (token, head). Decode has total_q==1 so the head axis is the
// useful one. Small dedicated pool (not the 128-thread global) for prefill
// fan-out; decode (nwork=hq≈16) stays serial — ParallelForRows barriers still
// lose to the NEON body at that size (measured).
  const int64_t nwork = total_q * hq;
  auto& pa_pool = []() -> vt::cpu::Threadpool& {
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    const int n = std::clamp(hw > 0 ? hw / 8 : 8, 4, 16);
    static vt::cpu::Threadpool pool(n);
    return pool;
  }();
  // Prefill (T*H large) parallelizes; pure decode (T=1,H=16) stays serial.
  constexpr int64_t kPaParallelMinWork = 64;

  // Single-request causal, full-window, no softcap: gather paged K/V into
  // dense [seq, nkv, d] once so the inner j-loop is sequential (better for
  // NEON + GQA reuse). Cap size — long-seq gather doubles traffic and loses
  // (measured: seq=512 gather ~20ms vs paged walk ~7ms). Short decode/prefill
  // stays under the cap and benefits.
  constexpr int64_t kGatherMaxElems = 32 * 1024;  // floats per of K/V (~128 KiB)
  std::vector<float> k_dense, v_dense;
  int64_t gather_seq = 0;
  if (num_reqs == 1 && args.causal && softcap <= 0.0f && window_left < 0 &&
      window_right < 0 && total_q > 0) {
    gather_seq = tok_slen[0];
    const int64_t gather_elems = gather_seq * num_kv_heads * d;
    if (gather_seq > 0 && gather_elems <= kGatherMaxElems) {
      k_dense.resize(static_cast<size_t>(gather_elems));
      v_dense.resize(static_cast<size_t>(gather_elems));
      auto load_k_i = [&](int64_t i) -> float {
        return k_f != nullptr ? k_f[i] : load_half(k_dt, k_h, i);
      };
      auto load_v_i = [&](int64_t i) -> float {
        return v_f != nullptr ? v_f[i] : load_half(v_dt, v_h, i);
      };
      for (int64_t j = 0; j < gather_seq; ++j) {
        const int64_t blk = btab[(j / block_size) * bt_col];
        const int64_t off = j % block_size;
        for (int64_t g = 0; g < num_kv_heads; ++g) {
          const int64_t kbase = blk * kc_blk + off * kc_pg + g * kc_hd;
          const int64_t vbase = blk * vc_blk + off * vc_pg + g * vc_hd;
          const int64_t dst = (j * num_kv_heads + g) * d;
          for (int64_t e = 0; e < d; ++e) {
            k_dense[static_cast<size_t>(dst + e)] = load_k_i(kbase + e);
            v_dense[static_cast<size_t>(dst + e)] = load_v_i(vbase + e);
          }
        }
      }
    }
  }
  const float* k_dense_p = k_dense.empty() ? nullptr : k_dense.data();
  const float* v_dense_p = v_dense.empty() ? nullptr : v_dense.data();

  auto process_range = [&](int64_t w0, int64_t w1) {
    std::vector<float> probs;
    std::vector<float> acc(static_cast<size_t>(d));
    std::vector<float> qloc(static_cast<size_t>(d));
    std::vector<float> kvloc;  // bf16/f16 K/V page staging (paged path)
    const bool k_is_f32 = k_f != nullptr;
    const bool v_is_f32 = v_f != nullptr;
    if (k_dense_p == nullptr && (!k_is_f32 || !v_is_f32))
      kvloc.resize(static_cast<size_t>(d));

    for (int64_t w = w0; w < w1; ++w) {
      const int64_t t = w / hq;
      const int64_t h = w % hq;
      const int64_t r = tok_req[static_cast<size_t>(t)];
      const int64_t p = tok_pos[static_cast<size_t>(t)];
      const int64_t seqlen = tok_slen[static_cast<size_t>(t)];
      const int64_t jmin = window_left >= 0 ? std::max<int64_t>(0, p - window_left) : 0;
      int64_t jmax = args.causal ? p : seqlen - 1;
      if (window_right >= 0) jmax = std::min(jmax, p + window_right);
      jmax = std::min(jmax, seqlen - 1);
      if (jmax < jmin) continue;

      const int64_t g = h / qpk;
      const int64_t qoff = (t * hq + h) * d;
      // Hoist Q head once (was re-loaded for every key position).
      if (q_f != nullptr) {
        std::memcpy(qloc.data(), q_f + qoff, static_cast<size_t>(d) * sizeof(float));
      } else {
        for (int64_t e = 0; e < d; ++e) qloc[static_cast<size_t>(e)] = load_half(q_dt, q_h, qoff + e);
      }

      probs.assign(static_cast<size_t>(jmax - jmin + 1), 0.0f);
      float m = -std::numeric_limits<float>::infinity();
      for (int64_t j = jmin; j <= jmax; ++j) {
        float dot;
        if (k_dense_p != nullptr) {
          const int64_t kbase = (j * num_kv_heads + g) * d;
          dot = dot_f32(qloc.data(), k_dense_p + kbase, d);
        } else {
          const int64_t blk = btab[r * bt_row + (j / block_size) * bt_col];
          const int64_t off = j % block_size;
          const int64_t kbase = blk * kc_blk + off * kc_pg + g * kc_hd;
          if (k_is_f32) {
            dot = dot_f32(qloc.data(), k_f + kbase, d);
          } else {
            for (int64_t e = 0; e < d; ++e)
              kvloc[static_cast<size_t>(e)] = load_half(k_dt, k_h, kbase + e);
            dot = dot_f32(qloc.data(), kvloc.data(), d);
          }
        }
        dot *= scale;
        if (softcap > 0.0f) dot = softcap * std::tanh(dot / softcap);
        probs[static_cast<size_t>(j - jmin)] = dot;
        if (dot > m) m = dot;
      }
      float denom = 0.0f;
      for (int64_t j = jmin; j <= jmax; ++j) {
        const float e = std::exp(probs[static_cast<size_t>(j - jmin)] - m);
        probs[static_cast<size_t>(j - jmin)] = e;
        denom += e;
      }
      const float inv = 1.0f / denom;
      std::fill(acc.begin(), acc.end(), 0.0f);
      for (int64_t j = jmin; j <= jmax; ++j) {
        const float pw = probs[static_cast<size_t>(j - jmin)] * inv;
        if (v_dense_p != nullptr) {
          const int64_t vbase = (j * num_kv_heads + g) * d;
          axpy_f32(acc.data(), v_dense_p + vbase, pw, d);
        } else {
          const int64_t blk = btab[r * bt_row + (j / block_size) * bt_col];
          const int64_t off = j % block_size;
          const int64_t vbase = blk * vc_blk + off * vc_pg + g * vc_hd;
          if (v_is_f32) {
            axpy_f32(acc.data(), v_f + vbase, pw, d);
          } else {
            for (int64_t e = 0; e < d; ++e)
              kvloc[static_cast<size_t>(e)] = load_half(v_dt, v_h, vbase + e);
            axpy_f32(acc.data(), kvloc.data(), pw, d);
          }
        }
      }
      if (o_f != nullptr) {
        std::memcpy(o_f + qoff, acc.data(), static_cast<size_t>(d) * sizeof(float));
      } else {
        for (int64_t e = 0; e < d; ++e) o_h[qoff + e] = F32ToBF16(acc[static_cast<size_t>(e)]);
      }
    }
  };

  if (nwork >= kPaParallelMinWork && pa_pool.NThreads() > 1) {
    vt::cpu::ParallelForRows(pa_pool, nwork, process_range);
  } else {
    process_range(0, nwork);
  }
  CommitHost(out);
}
namespace {

}  // namespace
void WarmPagedKvShadow(void* k_cache_data, void* v_cache_data,
                      int64_t num_blocks, int64_t block_size,
                      int64_t num_kv_heads, int64_t head_size,
                      int64_t used_blocks) {
  if (!HostFreeDecodeEnabled()) return;
  if (num_blocks < 1 || block_size < 1 || used_blocks < 1) return;
  MeshDevice& device = SharedMeshDevice();
  auto warm_one = [&](void* data) {
    // The flash KV store is one combined [nb,2,bs,nkv,d] buffer and the driver
    // hands us its dim-1 unbind slices (dense_attn_block.h KvSlice): rank-4
    // views whose block stride is 2*bs*nkv*d. Describing that view as
    // Tensor::Contiguous fabricated dense strides, so the shadow prefix upload
    // read block b at b*bs*nkv*d — one slab early inside the combined store.
    // For every block >= 1 the K shadow received the previous block's V slab
    // (zeros at prefill positions) and the V shadow the next block's K slab;
    // block 0 stayed correct only because offset 0 is each view's own base.
    // Describe the view with its real strides instead, and let
    // EnsurePagedKvTtnn admit it explicitly.
    Tensor cache;
    cache.data = data;
    cache.dtype = DType::kBF16;
    cache.device = Device{DeviceType::kTENSTORRENT, 0};
    cache.rank = 4;
    cache.shape[0] = num_blocks;
    cache.shape[1] = block_size;
    cache.shape[2] = num_kv_heads;
    cache.shape[3] = head_size;
    cache.stride[0] = 2 * block_size * num_kv_heads * head_size;
    cache.stride[1] = num_kv_heads * head_size;
    cache.stride[2] = head_size;
    cache.stride[3] = 1;
    const uint32_t used = static_cast<uint32_t>(
        std::min(used_blocks, num_blocks));
    EnsurePagedKvTtnn(cache, device, used, /*accept_unbind_view=*/true);
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr) {
      std::lock_guard<std::mutex> pg(PagedKvMutex());
      auto& sh = PagedKvShadows();
      std::fprintf(stderr, "[TT-TRACE] WarmPagedKvShadow ptr=%p nb=%lld used=%u shadows=%zu dev=%d\n",
                   data, (long long)num_blocks, used, sh.size(),
                   sh.count(reinterpret_cast<uintptr_t>(data)) ?
                       (int)sh[reinterpret_cast<uintptr_t>(data)].device.has_value() : -1);
    }
  };
  warm_one(k_cache_data);
  warm_one(v_cache_data);
}

void WarmRacIdx(const void* /*slot_mapping_owner*/, const int64_t* slots,
                int64_t num_slots, int64_t block_size,
                const int32_t* block_table, int64_t block_table_cols,
                const int32_t* seq_lens) {
  if (!HostFreeDecodeEnabled()) return;
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] WarmRacIdx n=%lld bs=%lld slot0=%lld sl0=%d\n",
                 (long long)num_slots, (long long)block_size, (long long)slots[0],
                 seq_lens ? seq_lens[0] : -1);
  // VT_TT_NO_IDX_WARM: stall bisection only — skip the per-step H2D copies
  // after the first capture (stale idx/page-table on device, numerically
  // wrong, mechanics test only).
  if (ReplayRegimeBisectSkip("VT_TT_NO_IDX_WARM")) return;
  const bool r2_steady = HostFreeDecodeEnabled()
                         && GraphCapturesDone() > 0;
  if (num_slots < 1) return;
  MeshDevice& device = SharedMeshDevice();
  // paged_update_cache needs:
  //   update_idxs[t] = the sequence position of the token being written
  //     (= seq_lens[t] - 1, the current decode position for user t)
  //   page_table       = the block table (virtual→physical block mapping)
  // The PA reads KV up to cur_pos = seq_lens - 1, so the RAC must write at
  // exactly that position for the PA to see the current token's KV.
  std::vector<int32_t> ptv;
  std::vector<int32_t> idxv;
  // The kernel maps update_idx -> page_table_ptr[update_idx / block_size]
  // (reader_{,paged_fused_}update_cache: virtual_block_id indexes the
  // page-table STICK), so the device page_table must carry the user's WHOLE
  // block-table row, not just the current virtual block (#1476: the old
  // [C,1] tensor made every write past the first block land in a garbage
  // physical block the moment cur_pos crossed block_size).
  ptv.reserve(static_cast<size_t>(num_slots * block_table_cols));
  idxv.reserve(static_cast<size_t>(num_slots));
  for (int64_t t = 0; t < num_slots; ++t) {
    const int64_t slot = slots[t];
    if (slot < 0 || seq_lens == nullptr || block_table == nullptr) {
      // Padding slot: paged_update_cache skips when update_idx == -1.
      for (int64_t c = 0; c < block_table_cols; ++c) ptv.push_back(0);
      idxv.push_back(-1);
      continue;
    }
    // update_idx = the 0-indexed position of the token being decoded this step
    // (= seq_lens[t] - 1, since seq_lens is the length BEFORE this token).
    // paged_update_cache writes to page_table[vblk] * block_size + update_idx % block_size,
    // which must equal the slot_mapping from the scheduler.
    const int32_t cur_pos = seq_lens[t] - 1;
    idxv.push_back(cur_pos);
    // Full row: every virtual block the kernel may resolve this step and
    // later in this width regime (steady state refreshes on content change).
    for (int64_t c = 0; c < block_table_cols; ++c) {
      ptv.push_back(block_table[t * block_table_cols + c]);
    }
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr) {
      const int32_t vblk = cur_pos / static_cast<int32_t>(block_size);
      const int32_t pblk = block_table[t * block_table_cols + vblk];
      std::fprintf(stderr, "[TT-TRACE] WarmRacIdx user=%lld slot=%lld cur_pos=%d vblk=%d pblk=%d bt_cols=%lld (expect slot=%d)\n",
                   (long long)t, (long long)slot, cur_pos, vblk, pblk, (long long)block_table_cols,
                   pblk * static_cast<int32_t>(block_size) + cur_pos % static_cast<int32_t>(block_size));
    }
  }
  const auto key = std::make_pair(num_slots, block_size);
  std::lock_guard<std::mutex> g(RacIdxMutex());
  RacIdxEntry& e = RacIdxCache()[key];
  // idx/page-table tensors are allocated ONCE per key and their CONTENT is
  // refreshed in place each step (copy_to_device, outside capture). The
  // captured paged_update_cache replays against the stable address and reads
  // the fresh values device-side.
  if (!e.allocated || block_table_cols != e.pt_width) {
    // ANY width change (block boundary growth, or the shrink when the longest
    // request of a multi-request batch finishes and block_table_num_cols drops)
    // reallocates: the else-branch copy_to_device would TT_FATAL on a shape
    // mismatch, and the driver resets + re-captures on any column-count change
    // (`cols_changed` compares with `!=`), so the new address is what the next
    // capture records. The retired tensor stays alive (see the field) — never
    // free a buffer a recorded trace addresses.
    if (e.allocated) e.retired_pts.push_back(std::move(e.page_table));
    e.page_table = ttnn::Tensor::from_vector<int32_t>(
        ptv, SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(num_slots),
                    static_cast<uint32_t>(block_table_cols)}),
                    ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR),
        &device);
    // R2: alias update_idxs to the on-device-advanced cur_pos (DecodePosCache)
    // when VT_TT_HOST_FREE_DECODE and the shapes match (decode T=1:
    // num_slots == num_reqs). plus_one on cur_pos then advances update_idxs
    // too, eliminating the per-replay update_idxs copy_to_device (toxic class).
    bool aliased = false;
    if (HostFreeDecodeEnabled()) {
      std::lock_guard<std::mutex> dg(DecodePosMutex());
      auto dit = DecodePosCache().find(num_slots);
      if (dit != DecodePosCache().end() && dit->second.allocated) {
        e.update_idxs = dit->second.cur_pos;  // share the same device buffer
        aliased = true;
      }
    }
    // After the first capture, a standalone update_idxs is never plus_one'd.
    // Refuse rather than freeze the write index and emit fluent wrong tokens.
    VT_CHECK(!r2_steady || aliased,
             "tenstorrent: WarmRacIdx allocated a standalone update_idxs after "
             "capture — plus_one will not advance it. Seed DecodePos per "
             "cache entry; recapture does NOT clear this (#1105).");
    if (!aliased) {
      e.update_idxs = ttnn::Tensor::from_vector<int32_t>(
          idxv, SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(num_slots)}),
                       ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR),
          &device);
    }
    e.allocated = true;
    e.pt_width = block_table_cols;
  } else {
    // Steady state within one width: update_idxs is aliased to the
    // on-device-advanced cur_pos (plus_one on replay steps, WarmDecodePos
    // re-seed on cold/capture steps) — never copied here. The RAC page_table
    // refreshes ONLY when its content changed (a new block was mapped):
    // zero copies inside a block, so the toxic every-step-interleaved-write
    // class stays out of the steady state; the copy that does fire rides the
    // same step as the boundary re-capture (#1476 — the "Phase 2 full"
    // refresh the old comment owed but never implemented).
    if (ptv != e.pt_host) {
      ttnn::Tensor pt_host = ttnn::Tensor::from_vector<int32_t>(
          ptv, SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(num_slots),
                      static_cast<uint32_t>(block_table_cols)}),
                      ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR));
      ttnn::copy_to_device(pt_host, e.page_table);
    }
  }
  e.pt_host = ptv;
  // Build the persistent sharded RAC input ONCE from the first available
  // paged-KV shadow's geometry (same nkv/d as the cache): logical
  // [1,1,nkv,d], padded [1,1,nkv_pad,d], HEIGHT_SHARDED L1, shard
  // [nkv_pad,d] on one core. paged_update_cache never reads the padded tail
  // rows (num_heads loop bound), so it is left uninitialized — no zeros, no
  // concat.
  if (!e.sharded_in_is_alloc) {
    std::lock_guard<std::mutex> pg(PagedKvMutex());
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-TRACE] WarmRacIdx shadow loop: %zu shadows\n",
                   PagedKvShadows().size());
    for (auto& [ptr, shadow] : PagedKvShadows()) {
      if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
        std::fprintf(stderr, "[TT-TRACE] shadow ptr=%p nkv=%u d=%u dc=%d\n",
                     (void*)ptr, shadow.nkv, shadow.d, shadow.device_current);
      if (shadow.nkv > 0 && shadow.d > 0 && (shadow.d % 32u) == 0u) {
        const uint32_t np = std::max(32u, ((shadow.nkv + 31u) / 32u) * 32u);
        const auto grid = device.compute_with_storage_grid_size();
        const tt::tt_metal::CoreRangeSet core_set =
            tt::tt_metal::num_cores_to_corerangeset(1u, grid, true);
        tt::tt_metal::ShardSpec ss(core_set, {np, shadow.d},
                                   tt::tt_metal::ShardOrientation::ROW_MAJOR);
        tt::tt_metal::MemoryConfig sm(
            tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            tt::tt_metal::BufferType::L1, ss);
        // Logical [1,1,nkv,d]; TILE alignment derives the physical
        // [1,1,nkv_pad,d] and the [nkv_pad,d] shard covers it on one core.
        e.sharded_in = ttnn::create_device_tensor(
            tt::tt_metal::TensorSpec(
                tt::tt_metal::Shape({1u, 1u, shadow.nkv, shadow.d}),
                tt::tt_metal::TensorLayout(
                    ttnn::DataType::BFLOAT16,
                    tt::tt_metal::PageConfig(ttnn::Layout::TILE), sm)),
            &device);
        // V needs a SEPARATE sharded buffer on a DIFFERENT core (forces a
        // program cache miss so interleaved_to_sharded compiles a fresh
        // program with V's buffer. Without different cores, the second call
        // reuses K's cached program and writes to K's buffer).
        const tt::tt_metal::CoreRangeSet core_set_v =
            tt::tt_metal::CoreRangeSet({
                tt::tt_metal::CoreRange(
                    tt::tt_metal::CoreCoord(1, 0),
                    tt::tt_metal::CoreCoord(1, 0))
            });
        tt::tt_metal::ShardSpec ss_v(core_set_v, {np, shadow.d},
                                     tt::tt_metal::ShardOrientation::ROW_MAJOR);
        tt::tt_metal::MemoryConfig sm_v(
            tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            tt::tt_metal::BufferType::L1, ss_v);
        e.sharded_in_v = ttnn::create_device_tensor(
            tt::tt_metal::TensorSpec(
                tt::tt_metal::Shape({1u, 1u, shadow.nkv, shadow.d}),
                tt::tt_metal::TensorLayout(
                    ttnn::DataType::BFLOAT16,
                    tt::tt_metal::PageConfig(ttnn::Layout::TILE), sm_v)),
            &device);
        e.nkv = shadow.nkv;
        e.d = shadow.d;
        e.sharded_in_is_alloc = true;
        break;
      }
    }
  }
  // paged_update_cache + the interleaved-TILE→sharded ttnn::copy are warmed
  // naturally: WarmPagedKvShadow (called by the driver BEFORE WarmRacIdx)
  // primes the shadows, and the cold step's eager ForwardLayers runs
  // TryReshapeAndCacheDeviceDecode (host_free is set, capturing is false)
  // which runs the identical copy+update sequence, compiling both programs.
}

void WarmPaMeta(const int32_t* block_table, int64_t num_reqs, int64_t max_blocks,
                int64_t bt_row_stride, int64_t bt_col_stride,
                const int32_t* seq_lens) {
  if (!HostFreeDecodeEnabled()) return;
  if (num_reqs < 1) return;
  MeshDevice& device = SharedMeshDevice();
  std::vector<int32_t> pt(static_cast<size_t>(num_reqs * max_blocks));
  for (int64_t r = 0; r < num_reqs; ++r) {
    for (int64_t c = 0; c < max_blocks; ++c) {
      const int32_t id = block_table[r * bt_row_stride + c * bt_col_stride];
      pt[static_cast<size_t>(r * max_blocks + c)] = id;
    }
  }
  std::vector<int32_t> cpos(static_cast<size_t>(num_reqs));
  for (int64_t r = 0; r < num_reqs; ++r) cpos[static_cast<size_t>(r)] = seq_lens[r] - 1;
  // Allocate ONCE per key; refresh CONTENT in place (copy_to_device, outside
  // capture) so the captured sdpa_decode replays against a stable address
  // while reading the fresh block-table/cur-pos values.
  const auto key = std::make_pair(num_reqs, max_blocks);
  // VT_TT_NO_IDX_WARM: legacy bisection override (skip ALL per-step copies).
  if (ReplayRegimeBisectSkip("VT_TT_NO_IDX_WARM")) return;
  const bool r2_steady = HostFreeDecodeEnabled()
                         && GraphCapturesDone() > 0;
  // R2 steady state: cur_pos/update_idxs advance on-device (plus_one); only
  // page_table needs a host refresh, and only when it actually changed (block
  // boundary crossed). The capture step (r2_steady==false) seeds everything.
  ttnn::Tensor pt_host = ttnn::Tensor::from_vector<int32_t>(
      pt, SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(num_reqs),
                static_cast<uint32_t>(max_blocks)}),
                ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR));
  std::lock_guard<std::mutex> g(PaMetaMutex());
  PaMetaEntry& e = PaMetaCache()[key];
  bool cur_pos_aliased = false;  // e.cur_pos shares the DecodePos device buffer
  bool pt_changed = true;        // a fresh allocation is a content change
  if (!e.allocated) {
    e.page_table = ttnn::Tensor::from_vector<int32_t>(
        pt, SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(num_reqs),
                  static_cast<uint32_t>(max_blocks)}),
                  ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR), &device);
    // R2: alias cur_pos to the on-device-advanced DecodePos cur_pos (advanced
    // by plus_one in the trace) when VT_TT_HOST_FREE_DECODE and it exists.
    // sdpa_decode reads this tensor; plus_one advances it → no per-replay
    // copy_to_device (the toxic ~38-replay hang class).
    bool aliased = false;
    if (HostFreeDecodeEnabled()) {
      std::lock_guard<std::mutex> dg(DecodePosMutex());
      auto dit = DecodePosCache().find(num_reqs);
      if (dit != DecodePosCache().end() && dit->second.allocated) {
        e.cur_pos = dit->second.cur_pos;  // share the same device buffer
        aliased = true;
        cur_pos_aliased = true;
      }
    }
    // After the first capture, a standalone cur_pos is never plus_one'd.
    // Refuse rather than freeze KV length and emit fluent wrong tokens.
    VT_CHECK(!r2_steady || aliased,
             "tenstorrent: WarmPaMeta allocated a standalone cur_pos after "
             "capture — plus_one will not advance it. Seed DecodePos per "
             "cache entry; recapture does NOT clear this (#1105).");
    if (!aliased) {
      e.cur_pos = ttnn::Tensor::from_vector<int32_t>(
          cpos, SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(num_reqs)}),
                       ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR), &device);
    }
    e.allocated = true;
  } else {
    // R2 steady state: page_table refreshes ONLY when content changed (block
    // boundary crossed). cur_pos/update_idxs advance on-device via plus_one.
    pt_changed = (e.pt_host != pt);
    if (pt_changed || !r2_steady) {
      ttnn::copy_to_device(pt_host, e.page_table);
    }
    if (!r2_steady) {
      ttnn::Tensor cp_host = ttnn::Tensor::from_vector<int32_t>(
          cpos, SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(num_reqs)}),
                       ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR));
      ttnn::copy_to_device(cp_host, e.cur_pos);
    }
    // #2469: the alias was recorded at allocation; it still holds. Detect it
    // here too — flagging only the alloc branch left cp_host frozen at the
    // last seeding step's value for every steady step, so the guard compared
    // a stale host value against the step's expectation.
    if (HostFreeDecodeEnabled()) {
      std::lock_guard<std::mutex> dg(DecodePosMutex());
      auto dit = DecodePosCache().find(num_reqs);
      if (dit != DecodePosCache().end() && dit->second.allocated) {
        cur_pos_aliased = true;
      }
    }
  }
  e.pt_host = pt;
  // #2469: cp_host must reflect what the DEVICE holds, not what this step
  // computes. For an aliased cur_pos the device side is the DecodePos buffer,
  // advanced in trace; at a request boundary this step's cpos is exactly the
  // value the device does NOT hold, and echoing it unconditionally made
  // TryPagedAttentionDeviceDecode's `cp_host[0] == seq_lens[0]-1` guard
  // compare the host against itself — it could never fire on a stale device
  // tensor.
  if (cur_pos_aliased) {
    std::lock_guard<std::mutex> dg(DecodePosMutex());
    auto dit = DecodePosCache().find(num_reqs);
    if (dit != DecodePosCache().end() && !dit->second.host_val.empty()) {
      e.cp_host = dit->second.host_val;  // the seeded/advanced device value
    } else if (!r2_steady) {
      e.cp_host = cpos;  // this call seeded the buffer with cpos content
    }
    // else: no mirror to read — leave the entry's last recorded value.
  } else if (!r2_steady) {
    e.cp_host = cpos;  // the copy/allocation above made the device hold cpos
  }
  // else: standalone in steady state — the copy was skipped, so the device
  // still holds the value cp_host already records. Leave it stale, honestly.
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] WarmPaMeta n=%lld mb=%lld cp0=%d host0=%d r2=%d pt_chg=%d\n",
                 (long long)num_reqs, (long long)max_blocks, (int)cpos[0],
                 e.cp_host.empty() ? -1 : (int)e.cp_host[0],
                 (int)r2_steady, (int)pt_changed);
}

// R2: seed the persistent cur_pos device tensor (= seq_lens - 1) and warm the
// plus_one program (program cache) so CaptureDecodePosAdvance can run inside
// the trace. Called on the capture/warm step (re-seed), NOT every replay.
void WarmDecodePos(const int32_t* seq_lens, int64_t num_reqs, bool replay_regime) {
  if (!HostFreeDecodeEnabled()) return;
  if (num_reqs < 1 || seq_lens == nullptr) return;
  // Replay regime: cur_pos advances on-device by the captured plus_one —
  // re-seeding here would overwrite the advance and break correctness.
  // A new num_reqs that was never seeded is refused rather than left frozen.
  if (replay_regime) {
    std::lock_guard<std::mutex> g(DecodePosMutex());
    auto it = DecodePosCache().find(num_reqs);
    VT_CHECK(it != DecodePosCache().end() && it->second.allocated,
             "tenstorrent: WarmDecodePos after capture for a num_reqs that "
             "was never seeded — cur_pos would freeze. Seed DecodePos per "
             "cache entry; recapture does NOT clear this (#1105).");
    // No device update here: the previous step's trace plus_one'd it once.
    // Keep the host mirror exactly one plus_one ahead of the last recorded
    // seed so WarmPaMeta's cp_host stays honest (#2469).
    if (!it->second.host_val.empty()) {
      for (auto& v : it->second.host_val) v += 1;
    }
    return;
  }
  // Cold/warm/capture step: (re-)seed cur_pos = seq_lens - 1 for THIS step.
  // The regime flag comes from the driver (graph captured?), NOT from
  // GraphCapturesDone(): Reset() releases the trace without clearing that
  // process-global counter, and the cold eager step that follows a Reset
  // runs no plus_one — so after a re-capture the on-device cur_pos is one
  // position behind unless it is re-seeded here (#1476). This also keeps the
  // FIRST capture correct (its capture step re-seeds, which is why the bug
  // only surfaced at the first block boundary, where Reset+re-capture runs).
  MeshDevice& device = SharedMeshDevice();
  std::vector<int32_t> cpos(static_cast<size_t>(num_reqs));
  for (int64_t r = 0; r < num_reqs; ++r)
    cpos[static_cast<size_t>(r)] = seq_lens[r] - 1;

  std::lock_guard<std::mutex> g(DecodePosMutex());
  DecodePosEntry& e = DecodePosCache()[num_reqs];
  if (!e.allocated) {
    e.cur_pos = ttnn::Tensor::from_vector<int32_t>(
        cpos, SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(num_reqs)}),
                     ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR), &device);
    e.allocated = true;
  } else {
    ttnn::Tensor cp_host = ttnn::Tensor::from_vector<int32_t>(
        cpos, SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(num_reqs)}),
                     ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR));
    ttnn::copy_to_device(cp_host, e.cur_pos);
  }
  e.host_val = cpos;  // the copy/allocation above made the device hold cpos
  // Warm plus_one (program cache) on a SCRATCH tensor so the in-trace call
  // doesn't trigger "Cannot load new binaries during trace capture" — but
  // leave e.cur_pos at its seeded value (the warm must NOT advance it, or the
  // captured body reads cur_pos+1). The scratch is allocated ONCE per entry
  // (the first seed always runs before any trace is live) and only REUSED
  // after: the boundary seed runs with this slot's trace live, and a fresh
  // device allocation then is the corruption the allocator warns about — it
  // made the first post-boundary replay read garbage KV rows and emit flaky
  // near-tie tokens (#2469).
  {
    if (!e.scratch_ready) {
      e.plus_one_scratch = ttnn::Tensor::from_vector<int32_t>(
          cpos, SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(num_reqs)}),
                       ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR), &device);
      e.scratch_ready = true;
    }
    ttnn::operations::experimental::plus_one(e.plus_one_scratch,
        /*sub_core_grids=*/std::nullopt, /*skip_negative_entries=*/true);
  }
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] WarmDecodePos n=%lld cp0=%d (seeded+warmed plus_one)\n",
                 (long long)num_reqs, (int)cpos[0]);
}

// R2: capture ttnn::plus_one(cur_pos) at the END of the trace body. The NEXT
// replay sees cur_pos+1. Must be called INSIDE BeginCapture/EndCapture, after
// all reads of cur_pos (sdpa_decode / paged_update_cache) in the body.
void CaptureDecodePosAdvance(int64_t num_reqs) {
  if (!HostFreeDecodeEnabled()) return;
  std::lock_guard<std::mutex> g(DecodePosMutex());
  auto it = DecodePosCache().find(num_reqs);
  if (it == DecodePosCache().end() || !it->second.allocated) {
    std::fprintf(stderr, "[TT-TRACE] CaptureDecodePosAdvance: no seeded cur_pos for n=%lld\n",
                 (long long)num_reqs);
    return;
  }
  ttnn::operations::experimental::plus_one(it->second.cur_pos,
      /*sub_core_grids=*/std::nullopt, /*skip_negative_entries=*/true);
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] CaptureDecodePosAdvance n=%lld (plus_one captured)\n",
                 (long long)num_reqs);
}
}  // namespace vt::tenstorrent
