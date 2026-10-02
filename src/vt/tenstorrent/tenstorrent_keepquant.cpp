// Tenstorrent keep-quant GEMM kernels (BACKEND-TENSTORRENT-SPLIT stage 2).
// Definitions moved verbatim from tenstorrent_ops.cpp; the shared shadow
// state and the cross-TU declarations live in tenstorrent_internal.h.
#include "vt/tenstorrent/tenstorrent_internal.h"

#include <optional>

namespace vt::tenstorrent {

// ---- KEEPQUANT W3: the resident i32 word shadow -----------------------------
// Words per staged packed block: the GGML block bytes (Q4_K 144 B -> 36
// words, Q5_K 176 -> 44, Q6_K 210 -> 53, Q8_0 32 -> 9; the 2-byte tails of
// Q6_K/Q8_0 zero-fill) ZERO-PADDED UP so the staged row is a 64-B multiple:
// the tensor's logical page size then EQUALS the device accessor's aligned
// page size, and the host write and the kernel read share one stride (leg33:
// a 144-B logical page inside 192-B DRAM slots read back as stream bytes
// 16 + 152*slot — the tensor write and the accessor disagreed on the
// layout; padding removes the disagreement by construction). The pad never
// reaches an output: every byte position the decoders read is inside the
// true block bytes, and the dot strides the padded width.
int KeepQuantWordsPerBlock(DType enc) {
  switch (enc) {
    case DType::kQ4_K: return 48;
    case DType::kQ5_K: return 48;
    case DType::kQ6_K: return 64;
    case DType::kQ8_0: return 16;
    case DType::kIQ3_XXS: return 32;  // 98 B zero-padded to 128 B
    case DType::kIQ2_XXS: return 32;  // 66 B zero-padded to 128 B
    case DType::kIQ2_S: return 32;    // 82 B zero-padded to 128 B
    // tenstorrent-gsq-keepquant wave 3: block_iq2_xs is 74 B (d 2 + qs 32
    // u16 + scales 8, ggml-common.h:388-392). Same 64-B-multiple word-grid
    // rule as every registered encoding — pad to 32 words = 128 B (the
    // IQ2_XXS/IQ2_S footprint), not the bare 19 words = 74 B. The decode
    // reads only the first 74 bytes. Derived from the traits block size
    // (vt::BlockBytes(kIQ2_XS) == 74), not copied from the wave comments.
    case DType::kIQ2_XS: return 32;
    // tenstorrent-gsq-keepquant wave 4: block_q2_K is 84 B (scales 16 +
    // qs 64 + d 2 + dmin 2, ggml-common.h:301-308). Same 64-B-multiple
    // word-grid rule as every registered encoding — pad to 32 words = 128 B
    // (the IQ2 family footprint), not the bare 21 words = 84 B. The decode
    // reads only the first 84 bytes. Derived from the traits block size
    // (vt::BlockBytes(kQ2_K) == 84), not copied from the wave comments.
    case DType::kQ2_K: return 32;
    // tenstorrent-gsq-keepquant wave 5: block_iq1_s is 50 B (d 2 + qs 32 +
    // qh 16, ggml-common.h:426-430) and block_iq1_m is 56 B (qs 32 + qh 16 +
    // scales 8, ggml-common.h:433-438). Same 64-B-multiple word-grid rule as
    // every registered encoding — pad to 16 words = 64 B (the Q8_0
    // footprint), not the bare 13/14 words. The decodes read only the true
    // 50/56 block bytes. Derived from the traits block sizes
    // (vt::BlockBytes(kIQ1_S) == 50, vt::BlockBytes(kIQ1_M) == 56), not
    // copied from the wave comments.
    case DType::kIQ1_S: return 16;
    case DType::kIQ1_M: return 16;
    case DType::kQ3_K: return 32;     // 110 B zero-padded to 128 B
    // tenstorrent-gsq-keepquant wave 1: block_iq3_s is 110 B (d 2 + qs 64 +
    // qh 8 + signs 32 + scales 4, ggml-common.h:413-422) — same 128 B pad as
    // Q3_K, derived from the traits block size, not copied from the wave
    // comments (the spec's risk note).
    case DType::kIQ3_S: return 32;
    // tenstorrent-gsq-keepquant wave 2: block_iq4_xs is 136 B (d 2 +
    // scales_h 2 + scales_l 4 + qs 128, ggml-common.h:455-459). The staged
    // word row must match the geometry the other six encodings use — every
    // registered encoding stages 64-B-multiple rows (64/128/192/256 B;
    // 16/32/48/64 words) — so pad to 48 words = 192 B (the Q4_K/Q5_K
    // footprint), not the bare 34 words = 136 B. The decode reads only the
    // first 136 bytes. Derived from the traits block size, not copied from
    // the wave comments (the spec's risk note).
    case DType::kIQ4_XS: return 48;
    default: return 0;
  }
}

// Free path hook: a freed host weight must drop its word shadow, so a
// recycled address can never alias a stale PACKED stage (UnregisterHostBuffer,
// the DropDecodedWeightShadow pattern). The immutable-post-load assumption
// covers the bytes, never the ADDRESS: std::aligned_alloc recycles chunks, and
// a later test/tensor reusing the address with matching (rows, nb, wpb) would
// otherwise be served another weight's words (leg41: the Q5_K decode sweep
// read Q4_K words — 256/256 wrong from a silent (1,1,48) key hit).
void DropKeepQuantWordShadow(void* host) {
  if (host == nullptr) return;
  std::lock_guard<std::mutex> g(KeepQuantWordMutex());
  KeepQuantWordShadows().erase(host);
}

// Stage the packed keep-quant stream as a resident i32 word tensor ONCE per
// weight — the eager pre-capture step or the first eager call pays it — and
// serve it forever after: the per-call decode runs entirely on-core from the
// resident words (no host repack, no from_vector upload). The packed master is
// immutable post-load, so a serve can never go stale while the master lives
// (same assumption as the BufferSlot/WeightViewShadow resident shadows); the
// UnregisterHostBuffer drop above covers the recycle case.
void TTReclaimPlanes(MeshDevice& device, std::vector<ttnn::Tensor>& planes);
ttnn::Tensor EnsureKeepQuantWords(const Tensor& packed, DType enc, int64_t rows,
                                  int64_t nb, MeshDevice& device) {
  VT_CHECK(packed.rank == 2 && packed.IsContiguous(),
           "tenstorrent EnsureKeepQuantWords: contiguous rank-2 packed");
  const int wpb = KeepQuantWordsPerBlock(enc);
  VT_CHECK(wpb > 0, "tenstorrent EnsureKeepQuantWords: unsupported encoding");
  {
    std::lock_guard<std::mutex> g(KeepQuantWordMutex());
    auto it = KeepQuantWordShadows().find(packed.data);
    if (it != KeepQuantWordShadows().end() && it->second.rows == rows &&
        it->second.nb == nb && it->second.wpb == wpb) {
      return it->second.words;
    }
  }
  // MISS = the one staging write this weight ever pays. A capture-time arrival
  // refuses by name (the ServeActF32 / zero-cache precedent): a captured graph
  // pins capture-time bytes its replay cannot refresh (the #2812 class), and
  // the engine's pre-capture eager step ("run one EAGER step", qwen3_5.cpp)
  // warms the shadow, so every later call — captured or not — hits it.
  if (tt_capture_active()) {
    KeepQuantCaptureStagingWritesCounter()++;
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr,
                   "[TT-KQ] EnsureKeepQuantWords word staging during capture\n");
  }
  VT_CHECK(!tt_capture_active(),
           "tenstorrent: keep-quant word-shadow miss during trace capture (" +
               std::string(Name(enc)) +
               ") — warm the keep-quant arm eagerly first");
  EnsureHost(packed);
  const uint8_t* bytes = packed.Ptr<uint8_t>();
  const int64_t b64 = rows * nb;
  const int64_t block_bytes = BlockBytes(enc);
  std::vector<int32_t> words;
  words.reserve(static_cast<size_t>(b64) * static_cast<size_t>(wpb));
  std::vector<uint8_t> padded(static_cast<size_t>(wpb) * 4u, 0u);
  for (int64_t b = 0; b < b64; ++b) {
    std::memcpy(padded.data(), bytes + b * block_bytes,
                static_cast<size_t>(block_bytes));
    for (int wj = 0; wj < wpb; ++wj) {
      const uint8_t* p = padded.data() + wj * 4;
      words.push_back(static_cast<int32_t>(
          static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
          (static_cast<uint32_t>(p[2]) << 16) |
          (static_cast<uint32_t>(p[3]) << 24)));
    }
  }
  AllocTraceSnapshot(device, "EnsureKeepQuantWords/pre");
  ttnn::Tensor staged = ttnn::Tensor::from_vector<int32_t>(
      std::move(words),
      SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(b64),
                                  static_cast<uint32_t>(wpb)}),
             ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR),
      &device);
  {
    std::lock_guard<std::mutex> g(KeepQuantWordMutex());
    KeepQuantWordShadows()[packed.data] =
        KeepQuantWordShadow{staged, rows, nb, wpb};
  }
  AllocTraceSnapshot(device, "EnsureKeepQuantWords/post");
  // W4d W3 (#3042): the weight's bf16 TILE staging is dead weight the moment
  // its word shadow exists — the keep-quant matmul reads only the words, and
  // a bf16-arm decline re-stages on demand from the slot's live host bytes.
  // At 27B the staged bf16 form of every k-quant weight sat beside the words
  // (attn_qkv alone: 97 x [10240,5120] bf16 = 9.7 GiB beside ~21 MiB q6_K
  // each), filled the banks to 93 percent during the warm pass, and
  // fragmented them into the init OOM. Release the slot's bf16 forms here;
  // the word shadows, embed twin and persistent non-quant stagings stay.
  // Scoped AFTER the KeepQuantWordMutex guard so the SlotMutex nesting order
  // (Slot -> KeepQuant elsewhere, sequential here) is never inverted.
  if (!tt_capture_active()) {
    std::vector<ttnn::Tensor> dead;
    {
      std::lock_guard<std::mutex> g_slot(SlotMutex());
      if (BufferSlot* s = FindSlot(packed.data);
          s != nullptr && packed.data == s->host &&
          (s->device.has_value() || s->persistent.has_value())) {
        if (s->device) dead.push_back(*s->device);
        if (s->persistent) dead.push_back(*s->persistent);
        s->device.reset();
        s->persistent.reset();
        s->device_current = false;
      }
    }
    if (!dead.empty()) TTReclaimPlanes(device, dead);
  }
  return staged;
}

// W4d W2 (#3042): force-free consumed decode planes. A decode plane's
// scope-exit free never runs: the eager dispatch copies the plane's
// TensorAttributes into every op it records, so the refcount is still above
// one when the C++ object dies and ~Tensor()'s use_count()==1 gate
// (ttnn/core/tensor/tensor.cpp:120) skips the free — the plane is orphaned
// for the process lifetime. The W1 vehicle trace booked 7.6 GB of orphans
// across 4 wide-weight first decodes with alloc_per_bank never decreasing in
// 600 transitions, and the focused red-first test books 4.3 GB on ONE
// 2-chunk weight. Draining the queue alone does not release the refs (a
// drain-only build leaked the same 4.3 GB), so every consumed plane is
// deallocated by force at its last use: finish() first (the device must not
// read a freed buffer), then ttnn::deallocate(force=true), which frees
// regardless of the refcount (the MeshTensorHolder swaps its Allocated state
// for a tombstone, so a repeated call on an aliased view is a no-op). Every
// plane in a reclaim list is consumed BEFORE the call — no alias is read
// afterwards — and no plane is a direct slice of the resident word shadow
// (a slice materializes its own buffer; the shadow is untouched). Capture
// skips the reclaim: finish() records a mesh event, which ttnn forbids
// mid-capture (the int8-dot staging precedent); the recorded ops keep the
// ordering and the replay's addresses.
void TTReclaimPlanes(MeshDevice& device,
                     std::initializer_list<ttnn::Tensor*> planes) {
  if (tt_capture_active()) return;
  device.mesh_command_queue().finish();
  for (ttnn::Tensor* plane : planes)
    ttnn::deallocate(*plane, /*force=*/true);
}
void TTReclaimPlanes(MeshDevice& device, std::vector<ttnn::Tensor>& planes) {
  if (tt_capture_active()) return;
  device.mesh_command_queue().finish();
  for (ttnn::Tensor& plane : planes)
    ttnn::deallocate(plane, /*force=*/true);
}


// Stream bytes [first, last) of the word-staged block as u8 {B, last-first}
// (little-endian lanes). concat stacks the four lane tensors, so the
// (lane, word)->(word, lane) permute restores stream order — the same trick
// the Q4_K scale extraction uses, generalized to any byte range.
ttnn::Tensor KeepQuantByteRange(const ttnn::Tensor& w, uint32_t B, int first,
                                int last, MeshDevice& device) {
  const int w0 = first / 4;
  const int nwords = (last + 3) / 4 - w0;
  ttnn::Tensor sw = ttnn::slice(
      w, ttsl::SmallVector<uint32_t>{0u, static_cast<uint32_t>(w0)},
      ttsl::SmallVector<uint32_t>{B,
                                  static_cast<uint32_t>(w0 + nwords)},
      ttsl::SmallVector<uint32_t>{1u, 1u});
  auto byte_lane = [](const ttnn::Tensor& t, int shift) {
    return ttnn::bitwise_and(ttnn::bitwise_right_shift(t, shift), 0xFF);
  };
  // W4d W2 (#3042): the lane web is ~25 word-plane sizes for the 128-byte
  // ql range — reclaimed below once the returned slice consumed the stream.
  ttnn::Tensor bl0 = byte_lane(sw, 0);
  ttnn::Tensor bl1 = byte_lane(sw, 8);
  ttnn::Tensor bl2 = byte_lane(sw, 16);
  ttnn::Tensor bl3 = byte_lane(sw, 24);
  ttnn::Tensor lanes_cat = ttnn::concat(
      std::vector<ttnn::Tensor>{bl0, bl1, bl2, bl3}, /*dim=*/1);
  ttnn::Tensor lanes_pm = ttnn::permute(
      ttnn::reshape(lanes_cat,
                    ttnn::Shape({B, 4u, static_cast<uint32_t>(nwords)})),
      ttsl::SmallVector<int64_t>{0, 2, 1});
  ttnn::Tensor stream =
      ttnn::reshape(lanes_pm,
                    ttnn::Shape({B, static_cast<uint32_t>(4 * nwords)}));
  const int off = first - 4 * w0;
  // W4d W2 (#3042): slice() returns its INPUT for a full-extent step-1
  // window (tt-metal slice.cpp:182), and two identities hide here. sw is
  // the word shadow itself when the range starts at word 0 and covers the
  // whole row (Q8_0's qb: [2,34) over a 9-word row), and the tail slice is
  // stream itself when off == 0 and the range is word-aligned (sb, ql, qh,
  // sc all land there). A forced reclaim of either alias kills the plane
  // out from under its owner — the resident shadow, or this frame's
  // permute buffer handed to the caller — so the aliased tensor stays with
  // its owner: sw dies with the frame (the view keeps the shadow alive),
  // and the aligned tail returns stream, whose buffer the caller's own
  // reclaim list frees at its last use.
  const int wpb = static_cast<int>(w.logical_shape()[1]);
  const bool sw_alias = w0 == 0 && nwords == wpb;
  const bool out_alias = off == 0 && 4 * nwords == last - first;
  if (out_alias) {
    if (sw_alias)
      TTReclaimPlanes(device, {&bl0, &bl1, &bl2, &bl3, &lanes_cat});
    else
      TTReclaimPlanes(device, {&sw, &bl0, &bl1, &bl2, &bl3, &lanes_cat});
    return stream;
  }
  ttnn::Tensor out = ttnn::slice(
      stream, ttsl::SmallVector<uint32_t>{0u, static_cast<uint32_t>(off)},
      ttsl::SmallVector<uint32_t>{B,
                                  static_cast<uint32_t>(off + last - first)},
      ttsl::SmallVector<uint32_t>{1u, 1u});
  if (sw_alias)
    TTReclaimPlanes(device, {&bl0, &bl1, &bl2, &bl3, &lanes_cat, &lanes_pm});
  else
    TTReclaimPlanes(device,
                    {&sw, &bl0, &bl1, &bl2, &bl3, &lanes_cat, &lanes_pm});
  return out;
}

// The -0.0f signed-zero repair constant, one cached device tensor per shape.
// It cannot be BUILT on device: the i32->f32 bitcast maps -0 to +0 (the W1
// comment was literal — bit-31-setting scalars also die in the scalar
// binding, and multiply/subtract canonicalize what survives), so every
// device-side construction arrived +0 and a forced-true ternary wrote +0.
// Host bytes survive the whole chain: -0.0f uploaded once per shape — a
// one-time staging fill, warmed by the same eager pre-pass as the word
// shadows and refused on a capture-time miss — then selected by where(),
// pure data movement on both ends.
ttnn::Tensor Neg0CacheGet(const ttnn::Shape& shape, MeshDevice& device) {
  const std::string key = "neg0/" + ZeroCacheKey(shape,
                                                 ttnn::DataType::FLOAT32,
                                                 ttnn::Layout::TILE);
  std::lock_guard<std::mutex> g(ZeroCacheMutex());
  auto& c = ZeroCache();
  auto it = c.find(key);
  if (it == c.end()) {
    VT_CHECK(!tt_capture_active(),
             "tenstorrent: -0 cache miss during capture — warm the keep-quant "
             "decode eagerly first");
    uint64_t n = 1;
    for (const auto d : shape.view()) n *= d;
    std::vector<float> z(static_cast<size_t>(n), -0.0f);
    it = c.emplace(key,
                   ttnn::to_layout(
                       ttnn::Tensor::from_vector<float>(
                           std::move(z),
                           SpecOf(shape, ttnn::DataType::FLOAT32,
                                  ttnn::Layout::ROW_MAJOR),
                           &device),
                       ttnn::Layout::TILE))
             .first;
  }
  return it->second;
}

// Decode `slice_rows` packed rows (nb blocks each) ALREADY staged as the
// resident i32 word tensor w ([slice_rows*nb, wpb]) into the repaired f32
// {slice_rows, nb*elems} in ROW_MAJOR — the W3 chains, one encoding each, run
// on a word RANGE instead of a whole packed tensor. DecodeKeepQuantBlocksF32
// below is the whole-tensor form; the grouped keep-quant matmul
// (MatmulBTQuantGroupedKernel) slices the tower's words to the P selected
// [N,K] row-ranges per call and runs THIS. Same chain, same numerics, so the
// W1/W3 bit-exact pins carry over unchanged.
std::optional<ttnn::Tensor> DecodeKeepQuantWordsFusedQK(
    const ttnn::Tensor& w, DType enc, int64_t slice_rows, int64_t nb,
    MeshDevice& device);
bool KeepQuantFusedDecodeEnabled();

ttnn::Tensor DecodeKeepQuantWordsF32(const ttnn::Tensor& w, DType enc,
                                     int64_t slice_rows, int64_t nb,
                                     MeshDevice& device) {
  const uint32_t B = static_cast<uint32_t>(slice_rows * nb);
  // TT-DECODE-FUSION: the fused single-program arm. Q6_K is the wave-1
  // served encoding, Q4_K the wave-3 one; anything else declines to the
  // chain below by falling through. VT_TT_KEEPQUANT_FUSED=0 forces the
  // proven chain (the named fallback arm); an unserved shape declines BY
  // NAME inside the fused dispatcher and also falls through — never
  // silently.
  if (enc == DType::kQ6_K || enc == DType::kQ4_K) {
    if (std::optional<ttnn::Tensor> fused =
            DecodeKeepQuantWordsFusedQK(w, enc, slice_rows, nb, device))
      return std::move(*fused);
  }
  {  // W4d W0 (#3042) attribution: label carries the slice size.
    char lbl[64];
    std::snprintf(lbl, sizeof(lbl), "kq-decode/rows=%lld/nb=%lld",
                  (long long)slice_rows, (long long)nb);
    AllocTraceSnapshot(device, lbl);
  }

  // W4d W2 (#3042): every plane this decode builds is reclaimed by force at
  // its last use — TTReclaimPlanes above (the W1 trace booked 7.6 GB of
  // orphans here; a drain-only build leaked the same 4.3 GB as the red test,
  // so the refs must be dropped, not waited on). The per-case batches below
  // each list planes consumed before the call; the helpers repair,
  // sign_bit_f32, zero_mask_f32, signed_byte_f32 and f16_bits_to_f32
  // reclaim their own webs internally. neg0 is the cached -0 constant and is
  // never listed.

  // f16 bit pattern (held in INT32) -> f32 value, the integer chain of
  auto f16_bits_to_f32 = [&](ttnn::Tensor t) {
    t = ttnn::to_layout(t, ttnn::Layout::TILE);
    ttnn::Tensor sign_b = ttnn::bitwise_left_shift(
        ttnn::bitwise_and(ttnn::bitwise_right_shift(t, 15), 1), 31);
    ttnn::Tensor mant = ttnn::bitwise_and(t, 0x3FF);
    ttnn::Tensor exp =
        ttnn::bitwise_and(ttnn::bitwise_right_shift(t, 10), 0x1F);
    ttnn::Tensor normal_bits = ttnn::bitwise_or(
        sign_b,
        ttnn::bitwise_or(ttnn::bitwise_left_shift(ttnn::add(exp, 112), 23),
                         ttnn::bitwise_left_shift(mant, 13)));
    ttnn::Tensor denorm_val =
        ttnn::multiply(ttnn::typecast(mant, ttnn::DataType::FLOAT32),
                       std::ldexp(1.0f, -24));
    ttnn::Tensor denorm_bits =
        ttnn::bitwise_or(sign_b, ttnn::bitcast(denorm_val, ttnn::DataType::INT32));
    ttnn::Tensor sub_bits =
        ttnn::where(ttnn::gt(mant, 0), denorm_bits, sign_b);  // +/- zero
    ttnn::Tensor out_bits =
        ttnn::where(ttnn::gt(exp, 0), normal_bits, sub_bits);
    const ttnn::Tensor out_f = ttnn::to_layout(
        ttnn::bitcast(out_bits, ttnn::DataType::FLOAT32),
        ttnn::Layout::ROW_MAJOR);
    // W4d W2 (#3042): the whole web is consumed by out_f — reclaim it,
    // including the TILE copy the to_layout above rebound into `t` (the
    // caller's bits tensor keeps its own buffer and is reclaimed by the
    // caller at its own last use).
    TTReclaimPlanes(device, {&t, &sign_b, &mant, &exp, &normal_bits,
                             &denorm_val, &denorm_bits, &sub_bits, &out_bits});
    return out_f;
  };
  // u8 byte -> SIGNED i8 value as exact f32: v - 256*bit7 (the int8 sign
  // extension; every operand and step is exact in the integer domain).
  // Q6_K's scales and Q8_0's qs are the signed-byte consumers.
  auto signed_byte_f32 = [&](const ttnn::Tensor& v) {
    ttnn::Tensor u = ttnn::bitwise_and(v, 0xFF);
    ttnn::Tensor bit7 = ttnn::bitwise_right_shift(u, 7);
    ttnn::Tensor u_f = ttnn::typecast(u, ttnn::DataType::FLOAT32);
    ttnn::Tensor b7_f = ttnn::typecast(bit7, ttnn::DataType::FLOAT32);
    ttnn::Tensor out = ttnn::subtract(u_f, ttnn::multiply(b7_f, 256.0f));
    TTReclaimPlanes(device, {&u, &bit7, &u_f, &b7_f});
    return out;
  };
  // Signed-zero repair, third mechanism, after two device-falsified drafts.
  // Draft one selected a -0 constant through where() over an arithmetic-f32
  // mask: the ternary never took its true branch (every sweep mismatch was
  // dev +0 where the oracle keeps -0, none the reverse). Draft two OR-ed the
  // sign bit into the product's own bits — bitwise work over a BITCAST of an
  // F32 TILE tensor, which mangled every bit pattern through a
  // reduced-precision path and landed the whole sweep on the f16 grid. This
  // one keeps to primitives the working f16 decode above proves:
  //   - the zero test is a FLOAT compare (IEEE eq() is true for both zeros)
  //     — never a bitcast of the product;
  //   - the -0 constant is born-int: cached INT32 zeros OR the sign bit
  //     (scalar), then the chain's own bitcast(int -> float);
  //   - the where() predicate is a COMPARISON output (gt(mask, 0)) — the
  //     one predicate form the f16 decode exercises (its two where() calls);
  //   - the {0,1} masks are broadcast f32 arithmetic over sign bits
  //     typecast OUT of born-int patterns (the proven direction).
  // W4d W2 (#3042): the constant is a {1,1,1} broadcast scalar. where()
  // broadcasts it, so ONE tiny cached tensor serves every repair shape; the
  // per-shape cache materialized a TILE-padded constant per distinct shape
  // ({B,8,32} pads to {B,32,32} — 2 GiB each at the test shape, and a fresh
  // 2 GiB entry per layer width at 27B). The cached entry is never
  // reclaimed.
  auto neg0_scalar = [&device]() {
    return Neg0CacheGet(ttnn::Shape({1u, 1u, 1u}), device);
  };
  // {0,1} f32 masks, built in ROW_MAJOR elementwise ops end to end: the
  // sign bit travels as an integer (shift/and over the BIT pattern — never
  // a device float, whose zero sign is already canonicalized), and the
  // zero test is a float compare. Callers combine masks with broadcast f32
  // arithmetic and reshape ROW_MAJOR only.
  auto sign_bit_f32 = [&](const ttnn::Tensor& bits, int shift) {
    ttnn::Tensor s = ttnn::bitwise_right_shift(bits, shift);
    ttnn::Tensor a = ttnn::bitwise_and(s, 1);
    ttnn::Tensor f = ttnn::typecast(a, ttnn::DataType::FLOAT32);
    TTReclaimPlanes(device, {&s, &a});
    return f;
  };
  auto zero_mask_f32 = [&](const ttnn::Tensor& v) {
    ttnn::Tensor eq = ttnn::eq(v, 0.0f);
    ttnn::Tensor eqr = ttnn::to_layout(eq, ttnn::Layout::ROW_MAJOR);
    ttnn::Tensor f = ttnn::typecast(eqr, ttnn::DataType::FLOAT32);
    TTReclaimPlanes(device, {&eq, &eqr});
    return f;
  };
  // The repair itself: where(gt(mask, 0), -0, value), every input TILE (a
  // TILE predicate mixed with ROW_MAJOR branches wrote only its first 16
  // output elements once already). Callers pass the value BY VALUE (a
  // shared-handle copy): moving it in beside an argument expression that
  // still reads it is unspecified-order evaluation, and the moved-from read
  // segfaults inside to_layout. W4d W2 (#3042): the repair owns its
  // operands' last reads, so it reclaims the incoming value and mask (the
  // caller passes fresh expression results and reassigns the value target
  // from the return), its own TILE copies and the where output. neg0 is the
  // cached -0 constant — never reclaimed.
  auto repair = [&](ttnn::Tensor value, ttnn::Tensor mask_f32,
                    const ttnn::Tensor& neg0) {
    AllocTraceSnapshot(device, "kq-decode/repair");
    ttnn::Tensor pred = ttnn::to_layout(ttnn::gt(mask_f32, 0.0f),
                                        ttnn::Layout::TILE);
    ttnn::Tensor value_t = ttnn::to_layout(value, ttnn::Layout::TILE);
    ttnn::Tensor where_out = ttnn::where(pred, neg0, value_t);
    ttnn::Tensor out =
        ttnn::to_layout(where_out, ttnn::Layout::ROW_MAJOR);
    TTReclaimPlanes(device,
                    {&value, &mask_f32, &pred, &value_t, &where_out});
    return out;
  };

  switch (enc) {
    case DType::kQ4_K:
    case DType::kQ5_K: {
      // Word 0: d | dmin. Words 1..3: scales[12]. Q5_K moves qs to words
      // 12..43 (ql[128] @ byte 48) and inserts qh[32] @ byte 16 (words
      // 4..11); the scale unpack and the 8x32 nibble planes are Q4_K's
      // verbatim, and Q5_K adds the 5th bit (+16) before the same epilogue.
      const bool q5 = enc == DType::kQ5_K;
      ttnn::Tensor w0 = ttnn::slice(
          w, ttsl::SmallVector<uint32_t>{0u, 0u},
          ttsl::SmallVector<uint32_t>{B, 1u},
          ttsl::SmallVector<uint32_t>{1u, 1u});
      ttnn::Tensor d_bits = ttnn::bitwise_and(w0, 0xFFFF);
      ttnn::Tensor dmin_bits =
          ttnn::bitwise_and(ttnn::bitwise_right_shift(w0, 16), 0xFFFF);
      ttnn::Tensor d = f16_bits_to_f32(d_bits);
      ttnn::Tensor dmin = f16_bits_to_f32(dmin_bits);

      ttnn::Tensor sb = KeepQuantByteRange(w, B, 4, 16, device);  // scales[12]
      // GetScaleMinK4(is, scales) for is = 0..7, group-major: is<4 low pair from
      // scales[is]/scales[is+4]; is>=4 high pair from scales[is+4] low bits and
      // scales[is-4]/scales[is] top bits.
      auto top6l4 = [](const ttnn::Tensor& t) {
        return ttnn::bitwise_left_shift(ttnn::bitwise_right_shift(t, 6), 4);
      };
      ttnn::Tensor sa = ttnn::slice(
          sb, ttsl::SmallVector<uint32_t>{0u, 0u},
          ttsl::SmallVector<uint32_t>{B, 4u},
          ttsl::SmallVector<uint32_t>{1u, 1u});
      ttnn::Tensor sm = ttnn::slice(
          sb, ttsl::SmallVector<uint32_t>{0u, 4u},
          ttsl::SmallVector<uint32_t>{B, 8u},
          ttsl::SmallVector<uint32_t>{1u, 1u});
      ttnn::Tensor sc = ttnn::slice(
          sb, ttsl::SmallVector<uint32_t>{0u, 8u},
          ttsl::SmallVector<uint32_t>{B, 12u},
          ttsl::SmallVector<uint32_t>{1u, 1u});
      ttnn::Tensor sc_f = ttnn::typecast(
          ttnn::concat(std::vector<ttnn::Tensor>{
              ttnn::bitwise_and(sa, 63),
              ttnn::bitwise_or(ttnn::bitwise_and(sc, 0xF), top6l4(sa))},
              /*dim=*/1),
          ttnn::DataType::FLOAT32);  // {B, 8}, values <= 63: exact
      ttnn::Tensor mm_f = ttnn::typecast(
          ttnn::concat(std::vector<ttnn::Tensor>{
              ttnn::bitwise_and(sm, 63),
              ttnn::bitwise_or(ttnn::bitwise_right_shift(sc, 4), top6l4(sm))},
              /*dim=*/1),
          ttnn::DataType::FLOAT32);  // {B, 8}, values <= 63: exact
      // W4d W2 (#3042): the scale web is consumed — w0 by the d/dmin bit
      // extractions, sb/sa/sm/sc by the sc_f/mm_f concats above.
      TTReclaimPlanes(device, {&w0, &sb, &sa, &sm, &sc});

      // Nibbles: 32 words are qs[128]; 8 nibble-lane shifts -> {B,256} with
      // idx = 8*lane + word. Flat idx = 16l + 8h + 8q + r over (byte lane l,
      // nibble half h, quarter q, word-in-quarter r); permute (l,h,q,r) ->
      // (q,h,r,l) so the flat order is the output order q*64 + h*32 + 4r + l —
      // scale group 2q+h covers the same 32-value run.
      const uint32_t qs_w0 = q5 ? 12u : 4u;
      ttnn::Tensor qw = ttnn::slice(
          w, ttsl::SmallVector<uint32_t>{0u, qs_w0},
          ttsl::SmallVector<uint32_t>{B, qs_w0 + 32u},
          ttsl::SmallVector<uint32_t>{1u, 1u});
      auto nib = [](const ttnn::Tensor& t, int shift) {
        return ttnn::bitwise_and(ttnn::bitwise_right_shift(t, shift), 0xF);
      };
      // W4d W2 (#3042): the eight nibble lanes and their concat/permute
      // chain are ~12 word-plane sizes — named so they can be reclaimed once
      // the typecast lands x5 (the reshapes are contiguous views, so the
      // concat and permute bases carry the memory).
      std::vector<ttnn::Tensor> lanes{nib(qw, 0), nib(qw, 4), nib(qw, 8),
                                      nib(qw, 12), nib(qw, 16), nib(qw, 20),
                                      nib(qw, 24), nib(qw, 28)};
      ttnn::Tensor lanes_cat = ttnn::concat(lanes, /*dim=*/1);
      ttnn::Tensor lanes_pm = ttnn::permute(
          ttnn::reshape(lanes_cat, ttnn::Shape({B, 4u, 2u, 4u, 8u})),
          ttsl::SmallVector<int64_t>{0, 3, 2, 4, 1});
      ttnn::Tensor x5 = ttnn::typecast(
          ttnn::reshape(lanes_pm, ttnn::Shape({B, 8u, 32u})),
          ttnn::DataType::FLOAT32);  // {B, 8, 32}, values <= 15: exact
      {
        std::vector<ttnn::Tensor> dead{qw, lanes_cat, lanes_pm};
        dead.insert(dead.end(), lanes.begin(), lanes.end());
        TTReclaimPlanes(device, dead);
      }
      if (q5) {
        // The 5th bit: output col c reads bit c/32 — the group index g — of
        // qh[c%32] (host u1 = 1<<2q for the low plane and u2 = 2<<2q for the
        // high plane of quarter q; g = 2q+h is exactly that bit). qh bytes
        // broadcast {B,1,32} against one shift per group; +16*bit is exact.
        ttnn::Tensor qh = KeepQuantByteRange(w, B, 16, 48, device);  // qh[32]
        ttnn::Tensor qh3 = ttnn::reshape(qh, ttnn::Shape({B, 1u, 32u}));
        std::vector<ttnn::Tensor> planes;
        planes.reserve(8);
        for (int g = 0; g < 8; ++g)
          planes.push_back(
              ttnn::bitwise_and(ttnn::bitwise_right_shift(qh3, g), 1));
        ttnn::Tensor q5_cat = ttnn::concat(planes, /*dim=*/1);
        ttnn::Tensor q5_f = ttnn::typecast(
            ttnn::reshape(q5_cat, ttnn::Shape({B, 8u, 32u})),
            ttnn::DataType::FLOAT32);
        // values <= 31: exact
        ttnn::Tensor x5b =
            ttnn::add(x5, ttnn::multiply(q5_f, 16.0f));
        // W4d W2 (#3042): the bit-plane web and the pre-add x5 are consumed.
        {
          std::vector<ttnn::Tensor> dead{qh3, q5_cat, q5_f, qh, x5};
          dead.insert(dead.end(), planes.begin(), planes.end());
          TTReclaimPlanes(device, dead);
        }
        x5 = std::move(x5b);
      }

      // y = (d*sc)*x - (dmin*mm): the host's exact f32 order, as separate ops,
      // every operand held at {B,8,32} — one 32x32 tile per block, the shape
      // every op handles (a TILE {B,8,1} operand physically pads to 32
      // columns and a TILE broadcast reads those padding columns as data,
      // which corrupted every non-zero logical column at B=1).
      // Signed-zero repair as bit work (or_sign above): the device multiply
      // AND subtract canonicalize a zero result's sign, while the host chain
      // is IEEE-exact — prod keeps d's sign through a zero product and m1
      // keeps dmin's through a zero product (mm and x are unsigned, so the
      // IEEE sign of both intermediates is the f16 sign bit), and a zero y
      // keeps a sign only when the m1 term's bit is clear ((-0) - (+0) is
      // -0). All three masks are broadcast f32 arithmetic on {0,1} values
      // read from the f16 BIT patterns — never from a device float, whose
      // zero sign is already canonicalized.
      // Signed-zero repair (repair/neg0_scalar above): the device multiply AND
      // subtract canonicalize a zero result's sign, while the host chain is
      // IEEE-exact — prod keeps d's sign through a zero product and m1 keeps
      // dmin's through a zero product (mm and x are unsigned, so the IEEE
      // sign of both intermediates is the f16 sign bit), and a zero y keeps a
      // sign only when the m1 term's bit is clear ((-0) - (+0) is -0).
      ttnn::Tensor dsign =
          ttnn::reshape(sign_bit_f32(d_bits, 15), ttnn::Shape({B, 1u, 1u}));
      ttnn::Tensor prod =
          ttnn::multiply(ttnn::reshape(ttnn::multiply(d, sc_f),
                                       ttnn::Shape({B, 8u, 1u})),
                         x5);
      prod = repair(prod, ttnn::multiply(dsign, zero_mask_f32(prod)),
                    neg0_scalar());
      // W4d W2 (#3042): d, sc_f, x5 and d_bits are consumed; the pre-repair
      // prod buffer was freed inside repair (the by-value value operand).
      TTReclaimPlanes(device, {&d, &sc_f, &x5, &d_bits});
      ttnn::Tensor dminsign =
          ttnn::reshape(sign_bit_f32(dmin_bits, 15), ttnn::Shape({B, 1u, 1u}));
      // Repair m1 at its FINAL {B,8,1} shape: the subtract must consume the
      // repaired tensor directly. Reshaping a to_layout(ROW_MAJOR) round-trip
      // output before the subtract made the broadcast read block 0's m1 row
      // for every block (constant y offset per element beyond block 0).
      ttnn::Tensor m1 = ttnn::reshape(ttnn::multiply(dmin, mm_f),
                                      ttnn::Shape({B, 8u, 1u}));
      m1 = repair(m1, ttnn::multiply(dminsign, zero_mask_f32(m1)),
                  neg0_scalar());
      TTReclaimPlanes(device, {&dmin, &mm_f, &dmin_bits});
      ttnn::Tensor y = ttnn::subtract(prod, m1);
      TTReclaimPlanes(device, {&prod, &m1});
      y = repair(
          y,
          ttnn::multiply(
              ttnn::reshape(ttnn::subtract(dsign,
                                           ttnn::multiply(dsign, dminsign)),
                            ttnn::Shape({B, 1u, 1u})),
              zero_mask_f32(y)),
          neg0_scalar());
      TTReclaimPlanes(device, {&dsign, &dminsign});
      return ttnn::reshape(
          std::move(y),
          ttnn::Shape({static_cast<uint32_t>(slice_rows),
                       static_cast<uint32_t>(nb) * 256u}));
    }
    case DType::kQ6_K: {
      // Word 52 holds d in its low half (bytes 208..209). 16 sub-blocks of
      // 16: output col c sits in sub-block s = c/16 at i = c%16, and
      // s = 8h + 2r + l/16 over (half h, run r, l in 0..32): per (h, r) the
      // host loop reads ql byte 64h + 32*(r&1) + l, low nibble for r<2 / high
      // otherwise, ORs the 2 high bits (qh[32h + l] >> 2r) & 3 into bit 4,
      // subtracts the 32 bias, and multiplies (d*sc)*q with the SIGNED scale
      // sc[8h + 2r + l/16] — no min term. Eight {B,2,16} pieces concat in
      // (h, r, l/16, l%16) order straight into [s][i].
      ttnn::Tensor w52 = ttnn::slice(
          w, ttsl::SmallVector<uint32_t>{0u, 52u},
          ttsl::SmallVector<uint32_t>{B, 53u},
          ttsl::SmallVector<uint32_t>{1u, 1u});
      ttnn::Tensor d_bits = ttnn::bitwise_and(w52, 0xFFFF);
      ttnn::Tensor d3 = ttnn::reshape(f16_bits_to_f32(d_bits),
                                      ttnn::Shape({B, 1u, 1u}));
      ttnn::Tensor ql = KeepQuantByteRange(w, B, 0, 128, device);
      ttnn::Tensor qh = KeepQuantByteRange(w, B, 128, 192, device);
      ttnn::Tensor sc_bytes = KeepQuantByteRange(w, B, 192, 208, device);
      ttnn::Tensor sc_f = signed_byte_f32(sc_bytes);  // {B,16}: exact ints
      ttnn::Tensor sc_sign = ttnn::typecast(
          ttnn::bitwise_and(ttnn::bitwise_right_shift(sc_bytes, 7), 1),
          ttnn::DataType::FLOAT32);  // {B,16} in {0,1}: the i8 sign bit
      std::vector<ttnn::Tensor> halves;
      halves.reserve(2);
      std::vector<ttnn::Tensor> sign_halves;
      sign_halves.reserve(2);
      for (int h = 0; h < 2; ++h) {
        std::vector<ttnn::Tensor> runs;
        runs.reserve(4);
        std::vector<ttnn::Tensor> sign_runs;
        sign_runs.reserve(4);
        // W4d W2 (#3042): every read plane of a run dies after its two
        // pushes — batched per half so the reclaim is one finish() per half.
        std::vector<ttnn::Tensor> run_dead;
        run_dead.reserve(4 * 7 + 8);
        for (int r = 0; r < 4; ++r) {
          const int qoff = 64 * h + 32 * (r % 2);
          ttnn::Tensor qb = ttnn::slice(
              ql,
              ttsl::SmallVector<uint32_t>{0u,
                                          static_cast<uint32_t>(qoff)},
              ttsl::SmallVector<uint32_t>{
                  B, static_cast<uint32_t>(qoff + 32)},
              ttsl::SmallVector<uint32_t>{1u, 1u});
          ttnn::Tensor nib = (r < 2) ? ttnn::bitwise_and(qb, 0xF)
                                     : ttnn::bitwise_right_shift(qb, 4);
          ttnn::Tensor hb = ttnn::slice(
              qh,
              ttsl::SmallVector<uint32_t>{0u,
                                          static_cast<uint32_t>(32 * h)},
              ttsl::SmallVector<uint32_t>{
                  B, static_cast<uint32_t>(32 * h + 32)},
              ttsl::SmallVector<uint32_t>{1u, 1u});
          ttnn::Tensor hi2 = ttnn::bitwise_and(
              ttnn::bitwise_right_shift(hb, 2 * r), 3);
          ttnn::Tensor nib6 =
              ttnn::bitwise_or(nib, ttnn::bitwise_left_shift(hi2, 4));
          ttnn::Tensor q6 = ttnn::subtract(
              ttnn::typecast(nib6, ttnn::DataType::FLOAT32),
              32.0f);  // {B,32}, values in [-32, 31]: exact
          const int soff = 8 * h + 2 * r;
          ttnn::Tensor s2 = ttnn::slice(
              sc_f,
              ttsl::SmallVector<uint32_t>{0u,
                                          static_cast<uint32_t>(soff)},
              ttsl::SmallVector<uint32_t>{
                  B, static_cast<uint32_t>(soff + 2)},
              ttsl::SmallVector<uint32_t>{1u, 1u});
          // (d*sc) first — the host's left-to-right association — then *q.
          runs.push_back(ttnn::multiply(
              ttnn::multiply(d3,
                             ttnn::reshape(s2, ttnn::Shape({B, 2u, 1u}))),
              ttnn::reshape(q6, ttnn::Shape({B, 2u, 16u}))));
          // sign(q): the nibble sits below bit 5 before the -32 bias, so
          // q < 0 is exactly nib6 < 32 — an INT32 compare on the raw bits,
          // never a device float's canonicalized zero sign.
          sign_runs.push_back(
              ttnn::reshape(ttnn::typecast(ttnn::lt(nib6, 32),
                                           ttnn::DataType::FLOAT32),
                            ttnn::Shape({B, 2u, 16u})));
          for (ttnn::Tensor* dead_plane :
               {&qb, &hb, &nib, &hi2, &nib6, &q6, &s2})
            run_dead.push_back(*dead_plane);
        }
        halves.push_back(ttnn::concat(runs, /*dim=*/1));
        sign_halves.push_back(
            ttnn::concat(sign_runs, /*dim=*/1));
        run_dead.insert(run_dead.end(), runs.begin(), runs.end());
        run_dead.insert(run_dead.end(), sign_runs.begin(), sign_runs.end());
        TTReclaimPlanes(device, run_dead);
      }
      // W4d W2 (#3042): the per-half pieces are consumed by these concats;
      // the reshapes are contiguous views over them.
      ttnn::Tensor halves_cat = ttnn::concat(halves, /*dim=*/1);
      ttnn::Tensor signs_cat = ttnn::concat(sign_halves, /*dim=*/1);
      TTReclaimPlanes(device, halves);
      TTReclaimPlanes(device, sign_halves);
      ttnn::Tensor prod = ttnn::reshape(
          halves_cat,
          ttnn::Shape({B, 16u, 16u}));  // [s][i] with s = 8h + 2r + l/16
      ttnn::Tensor qsign = ttnn::reshape(
          signs_cat,
          ttnn::Shape({B, 16u, 16u}));
      // Zero-product sign: the IEEE sign of (d*sc)*q is the XOR of all
      // three operand signs — d's f16 bit, sc's i8 bit7, and sign(q) above
      // (q and the typecast zeros are +0, so they contribute their bit
      // directly). The XOR of three {0,1} masks is
      // a+b+c-2(ab+ac+bc)+4abc, exact in f32 broadcast arithmetic; the
      // product is repaired where that XOR lands on a zero (or_sign).
      ttnn::Tensor ds = ttnn::reshape(sign_bit_f32(d_bits, 15),
                                      ttnn::Shape({B, 1u, 1u}));
      ttnn::Tensor ss = ttnn::reshape(sc_sign, ttnn::Shape({B, 16u, 1u}));
      ttnn::Tensor ds_ss = ttnn::multiply(ds, ss);  // {B,16,1}
      ttnn::Tensor pairs = ttnn::add(
          ttnn::add(ds_ss, ttnn::multiply(ds, qsign)),
          ttnn::multiply(ss, qsign));  // ab + ac + bc
      ttnn::Tensor quad = ttnn::multiply(ds_ss, qsign);  // abc
      ttnn::Tensor pred =
          ttnn::add(ttnn::subtract(ttnn::add(ttnn::add(ds, ss), qsign),
                                   ttnn::multiply(pairs, 2.0f)),
                    ttnn::multiply(quad, 4.0f));
      prod = repair(prod, ttnn::multiply(pred, zero_mask_f32(prod)),
                    neg0_scalar());
      // W4d W2 (#3042): the sign-algebra planes and the concat bases are
      // consumed; the word ranges and d/sc planes were consumed in the runs.
      TTReclaimPlanes(device, {&ds, &ss, &ds_ss, &pairs, &quad, &pred, &qsign,
                               &halves_cat, &signs_cat});
      TTReclaimPlanes(device, {&w52, &d_bits, &d3, &ql, &qh, &sc_bytes, &sc_f});
      return ttnn::reshape(
          prod,
          ttnn::Shape({static_cast<uint32_t>(slice_rows),
                       static_cast<uint32_t>(nb) * 256u}));
    }
    case DType::kQ8_0: {
      // Word 0: d | first two qs bytes. qs[32] = stream bytes 2..33; host:
      // y = qs * d — IEEE multiply is commutative in value AND zero sign, so
      // the device's d*q is the same bit pattern.
      ttnn::Tensor w0 = ttnn::slice(
          w, ttsl::SmallVector<uint32_t>{0u, 0u},
          ttsl::SmallVector<uint32_t>{B, 1u},
          ttsl::SmallVector<uint32_t>{1u, 1u});
      ttnn::Tensor d_bits = ttnn::bitwise_and(w0, 0xFFFF);
      ttnn::Tensor d = f16_bits_to_f32(d_bits);  // {B,1}
      ttnn::Tensor qb = KeepQuantByteRange(w, B, 2, 34, device);  // {B,32} raw bytes
      ttnn::Tensor qf = signed_byte_f32(qb);
      ttnn::Tensor prod = ttnn::multiply(d, qf);  // {B,32}
      // Zero-product sign: the IEEE product's sign is sign(d) XOR sign(q) —
      // sign(d) is the f16 bit, sign(q) the raw byte's bit7 (a zero byte
      // typecasts to +0 and contributes a clear bit, exactly its sign). The
      // XOR of two {0,1} masks is a+b-2ab, exact in f32 broadcast
      // arithmetic; the product is repaired where the XOR lands on a zero.
      ttnn::Tensor dsign = sign_bit_f32(d_bits, 15);  // {B,1}
      ttnn::Tensor qsign = sign_bit_f32(qb, 7);       // {B,32}
      ttnn::Tensor pred = ttnn::subtract(
          ttnn::add(dsign, qsign),
          ttnn::multiply(ttnn::multiply(dsign, qsign), 2.0f));
      prod = repair(prod, ttnn::multiply(pred, zero_mask_f32(prod)),
                    neg0_scalar());
      // W4d W2 (#3042): the whole web is consumed — d/dsign by prod, qb/qf
      // by prod and qsign, the sign algebra by the repair mask.
      TTReclaimPlanes(device, {&w0, &d_bits, &d, &qb, &qf, &dsign, &qsign,
                               &pred});
      return ttnn::reshape(
          prod,
          ttnn::Shape({static_cast<uint32_t>(slice_rows),
                       static_cast<uint32_t>(nb) * 32u}));
    }
    default:
      // EnsureKeepQuantWords refuses anything else before this point.
      VT_CHECK(false, "tenstorrent keep-quant decode: unsupported encoding");
      return w;
  }
}

// The whole-tensor form: stage the packed [rows, nb] tensor once (EnsureKeepQuantWords)
// and run the chains on all of it — the W1/W3 keep-quant decode entry.
ttnn::Tensor DecodeKeepQuantBlocksF32(const Tensor& packed, DType enc,
                                      int64_t rows, int64_t nb,
                                      MeshDevice& device) {
  const ttnn::Tensor w = EnsureKeepQuantWords(packed, enc, rows, nb, device);
  return DecodeKeepQuantWordsF32(w, enc, rows, nb, device);
}

void KeepQuantDecodeKernel(Queue&, Tensor& out, const Tensor& packed) {
  TT_OP_TRACE("KeepQuantDecode");
  VT_CHECK(packed.rank == 2 && out.rank == 2,
           "tenstorrent kKeepQuantDecode: packed rank-2 [rows, nb], out "
           "rank-2 [rows, nb*elems]");
  const DType enc = packed.dtype;
  VT_CHECK(enc == DType::kQ4_K || enc == DType::kQ5_K || enc == DType::kQ6_K ||
               enc == DType::kQ8_0,
           std::string("tenstorrent kKeepQuantDecode: packed dtype ") +
               Name(enc) +
               " has no registered keep-quant decode (registered set: kQ4_K/"
               "kQ5_K/kQ6_K/kQ8_0)");
  const int64_t elems = BlockElems(enc);
  VT_CHECK(out.dtype == DType::kF32,
           "tenstorrent kKeepQuantDecode: out must be f32");
  VT_CHECK(packed.IsContiguous() && out.IsContiguous(),
           "tenstorrent kKeepQuantDecode: contiguous required");
  const int64_t rows = packed.shape[0];
  const int64_t nb = packed.shape[1];
  VT_CHECK(out.shape[0] == rows && out.shape[1] == nb * elems,
           "tenstorrent kKeepQuantDecode: out shape mismatch");
  ttnn::Tensor y =
      DecodeKeepQuantBlocksF32(packed, enc, rows, nb, SharedMeshDevice());
  CommitDeviceLogical2D(out, std::move(y), static_cast<uint32_t>(rows),
                        static_cast<uint32_t>(nb * elems));
}

// kMatmulBTQuant (KEEPQUANT W2, generalized W3, switched W4a wave-3b-1):
// `a` is [M,K] float, `b` is [N,K] packed keep-quant blocks — out = a @ b^T,
// reached through vt::MatmulBT's block-weight dispatch (ops.cpp:163), the
// entry every model matmul helper already uses. W4a wave-3b-1 (#3030): the
// dense keep-quant matmul IS the wave-3a chunked E=1 grouped arm — the
// production dense path consumes PACKED words. The PACKED words stage once
// per weight (EnsureKeepQuantWords) and each call slice-decodes + accumulates
// in capture-safe chunks (256 MiB plane policy, see the E=1 arm); the decoded
// bf16 TWIN of wave-2/W3 is GONE from this path. The fourth spec amendment
// makes this the row's production surface: the whole keep-quant set beyond
// the gather class is served PACKED, and a captured graph must never hold a
// whole-weight tile. The gather class keeps its own embed-table twin
// (EnsureEmbedTableDevice); the vehicle's tied head shares the GGUF tensor
// with the embedding — its GATHER keeps that twin, its MATMUL stages only the
// packed words. Non-keep-quant (bf16/f32) weights never enter this kernel
// (vt::MatmulBT routes them to kMatmulBT), and non-TT devices are untouched.
// Exactly the encodings DeviceKeepQuantSupported admits on kTENSTORRENT
// (gguf_keep_quant.cpp). Refusing here BY NAME keeps an admitted-but-
// unimplemented encoding from reaching the device.
void MatmulBTQuantGroupedKernel(Queue&, Tensor& out, const Tensor& act,
                                const Tensor& weight, const Tensor& expert_ids);
// W4b (#3031): the below-ttnn int8-dot device kernel — the dense arm's
// quantized-domain dot (definition below, after the grouped kernel).
void MatmulBTQuantInt8DotKernel(Queue& q, Tensor& out, const Tensor& a,
                                const Tensor& b);
void MatmulBTQuantKernel(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  TT_OP_TRACE("MatmulBTQuant");
  VT_CHECK(a.rank == 2 && b.rank == 2 && out.rank == 2,
           "tenstorrent kMatmulBTQuant: rank-2 a/b/out required");
  const DType enc = b.dtype;
  // vt::Name() emits the lowercase storage name ("q4_0"); the refusal must
  // name the ENUM the caller passed, so the k-prefix and capital go on here.
  const std::string enc_lower = Name(enc);
  const std::string enc_name =
      std::string("k") + static_cast<char>(enc_lower[0] - 'a' + 'A') +
      enc_lower.substr(1);
  VT_CHECK(enc == DType::kQ4_K || enc == DType::kQ5_K || enc == DType::kQ6_K ||
               enc == DType::kQ8_0 || enc == DType::kIQ3_XXS ||
               enc == DType::kIQ2_XXS || enc == DType::kIQ2_S ||
               enc == DType::kQ3_K || enc == DType::kIQ3_S ||
               enc == DType::kIQ4_XS || enc == DType::kIQ2_XS ||
               enc == DType::kQ2_K || enc == DType::kIQ1_S ||
               enc == DType::kIQ1_M,
           std::string("tenstorrent kMatmulBTQuant: ") + enc_name +
               " has no keep-quant decode on TENSTORRENT; the registered set "
               "is kQ4_K/kQ5_K/kQ6_K/kQ8_0/kIQ3_XXS/kIQ2_XXS/kIQ2_S/kQ3_K/"
               "kIQ3_S/kIQ4_XS/kIQ2_XS/kQ2_K/kIQ1_S/kIQ1_M "
               "(BACKEND-TENSTORRENT-KEEPQUANT, QUANT-GGUF-IQ-TENSTORRENT)");
  const int64_t elems = BlockElems(enc);
  VT_CHECK(b.shape[1] % elems == 0,
           std::string("tenstorrent kMatmulBTQuant: K must be a whole number "
                       "of ") +
               Name(enc) + " blocks (" + std::to_string(elems) + " elems)");
  {  // W4d W0 (#3042) attribution: which matmul weight is being staged.
    char lbl[64];
    std::snprintf(lbl, sizeof(lbl), "matmulbtq/N=%lld/K=%lld",
                  (long long)b.shape[0], (long long)b.shape[1]);
    AllocTraceSnapshot(SharedMeshDevice(), lbl);
  }
  VT_CHECK(IsFloatDType(a.dtype) &&
               (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kMatmulBTQuant: float activation, f32/bf16 out");
  const uint32_t M = static_cast<uint32_t>(a.shape[0]);
  const uint32_t K = static_cast<uint32_t>(a.shape[1]);
  const uint32_t N = static_cast<uint32_t>(b.shape[0]);
  VT_CHECK(b.shape[1] == K, "tenstorrent kMatmulBTQuant: a/b inner dim mismatch");
  VT_CHECK(out.shape[0] == M && out.shape[1] == N,
           "tenstorrent kMatmulBTQuant: out shape mismatch");
  VT_CHECK(a.IsContiguous() && b.IsContiguous() && out.IsContiguous(),
           "tenstorrent kMatmulBTQuant: strided tensors are not supported in W2");

  // W4b (#3031, path decided in the spec amendment): the int8-dot device
  // kernel behind the F32-OUT dense arm — the quantized-domain dot,
  // activation quantized once on-core to the vec_dot pairing encoding, the
  // upstream 8-lane lane split preserved verbatim, bit-exact vs
  // vt::cpu::BlockVecDot by the red-first sweep. One captured launch replaces
  // the per-chunk E=1 chain (the capture-demand gate this row owes).
  //
  // LANDING DECISION: the lever lands OP-LEVEL and DEFAULT OFF — the e2e
  // anchor band (<= 500 mnat, spec ## W4b) fails on the quantized domain's
  // one non-tie flip (vehicle p5 tok7, 1125 mnats; determinism-proven by
  // byte-identical capture dumps x2), so the production vehicle keeps the W4a
  // path this wave. With VT_TT_KEEPQUANT_INT8DOT unset or "0" the f32-out
  // dense arm falls through to the W4a E=1 grouped arm below, which served
  // exactly these calls before W4b; the env opts the lever in, the op suite
  // opts in explicitly, and e2e reach is owed (row spec ## Owed + follow-up).
  //
  // A BF16-OUT call keeps the W4a E=1 grouped arm: the int8-dot kernel
  // computes f32 cells only, and committing its f32 dev_out into a bf16
  // slot left the slot holding f32 bytes at an f32 page geometry — the next
  // consumer that reads the slot as bf16 got word-halved garbage (the
  // ROW_MAJOR chained leg's NaN signature: the vehicle's mid-layer bf16
  // keep-quant matmul fed a down-projection whose activation read the
  // poisoned slot).
  // W4d W6: int8-dot serves BOTH out dtypes when opted in — one captured
  // launch per matmul replaces the per-chunk E=1 chain, whose capture-time
  // transients pinned 3.8 GiB into the trace region at 27B
  // (tenstorrent-27b-int8dot-capture.md). bf16-out carries an explicit
  // f32->bf16 cast before the commit (the W4b store-geometry bug, fixed).
  // QUANT-GGUF-IQ-TENSTORRENT wave 1: IQ3_XXS has NO W4a grouped arm (the
  // grouped decode set is the four W3 encodings), so its only serve is the
  // int8-dot kernel — dispatch it there REGARDLESS of the env, which makes
  // the capability reachable on the default configuration. The other four
  // encodings keep the env-gated lever exactly as W4d left it. Wave 3
  // (QUANT-GGUF-IQ-TENSTORRENT): kQ3_K (enc_sel 7) joins the unconditional
  // set the same way — the grouped arm has no Q3_K decode to fall through
  // to either. tenstorrent-gsq-keepquant wave 1: kIQ3_S (enc_sel 8, the
  // largest GSQ-RCO census gap at 97 tensors) joins the unconditional set
  // the same way — no grouped decode exists for it either.
  // tenstorrent-gsq-keepquant wave 2: kIQ4_XS (enc_sel 9, 33 census tensors
  // on ssm_out + attn) joins the unconditional set the same way — no
  // grouped decode exists for it either.
  // tenstorrent-gsq-keepquant wave 3: kIQ2_XS (enc_sel 10, 32 census tensors
  // on ffn) joins the unconditional set the same way — no grouped decode
  // exists for it either.
  // tenstorrent-gsq-keepquant wave 4: kQ2_K (enc_sel 11, 28 census tensors on
  // ffn + embd) joins the unconditional set too. It IS a k-quant, but its
  // vec_dot pairing is q8_K — the same pairing Q3_K (already unconditional
  // here) uses — so the int8-dot kernel shape serves it; the grouped W4a arm
  // has no Q2_K decode to fall through to (and is E=1 anyway).
  // tenstorrent-gsq-keepquant wave 5: kIQ1_S (enc_sel 12) and kIQ1_M
  // (enc_sel 13), the 8 ffn-tail census tensors, join the unconditional set
  // the same way — no grouped decode exists for either (and the grouped arm
  // is E=1 anyway). Sub-bit accuracy: the decodes are BIT-EXACT against the
  // CPU vec_dot arms, so the arms reproduce whatever the CPU tier computes;
  // the near-tie greedy-decode disposition of the spec's e2e gate is a
  // separate, model-level question and stays with the row.
  if (enc == DType::kIQ3_XXS || enc == DType::kIQ2_XXS ||
      enc == DType::kIQ2_S || enc == DType::kQ3_K ||
      enc == DType::kIQ3_S || enc == DType::kIQ4_XS ||
      enc == DType::kIQ2_XS || enc == DType::kQ2_K ||
      enc == DType::kIQ1_S || enc == DType::kIQ1_M) {
    MatmulBTQuantInt8DotKernel(q, out, a, b);
    return;
  }
  if (const char* int8dot_env = std::getenv("VT_TT_KEEPQUANT_INT8DOT");
      int8dot_env != nullptr && int8dot_env[0] != '\0' &&
      std::strcmp(int8dot_env, "0") != 0) {
    MatmulBTQuantInt8DotKernel(q, out, a, b);
    return;
  }
  // W4a wave-3b-1 (#3030): the bf16-out dispatch. P = M output rows against
  // expert 0; the ids are statically all zero and the E=1 arm never reads
  // them (the capture contract proven by the garbage-ids leg), so a host
  // zeros tensor is all the grouped contract needs. Capture-safe by
  // construction: the eager warm step stages every word shadow before
  // capture (a capture-time miss refuses inside EnsureKeepQuantWords) and
  // every chunk offset is a capture-time constant replayed verbatim.
  std::vector<int32_t> zero_ids(static_cast<size_t>(a.shape[0]), 0);
  const Tensor ids = Tensor::Contiguous(zero_ids.data(), DType::kI32,
                                        Device{DeviceType::kCPU, 0},
                                        {a.shape[0]});
  MatmulBTQuantGroupedKernel(q, out, a, b, ids);
}

// The decoded bf16 twin map, retained after the wave-3b-1 switch as the
// TWIN-ABSENCE probe's surface and the recycled-address hygiene hook: the
// Free path hook: a freed host weight must drop its twin, so a recycled
// address can never alias a stale decode (UnregisterHostBuffer).
void DropDecodedWeightShadow(void* host) {
  if (host == nullptr) return;
  std::lock_guard<std::mutex> g(DecodedWeightMutex());
  DecodedWeightShadows().erase(reinterpret_cast<uintptr_t>(host));
}

void DropGroupedActShadow(void* host) {
  if (host == nullptr) return;
  std::lock_guard<std::mutex> g(GroupedActMutex());
  GroupedActShadows().erase(reinterpret_cast<uintptr_t>(host));
}

// kMatmulBTQuantGrouped (KEEPQUANT W4a wave-2, #3030): out[P,N], act[Pa,K]
// (Pa==1 broadcast), weight[E*N,K] PACKED block-quant, expert_ids[P] i32 —
// the expert-batched analog of MatmulBTQuantKernel above, mirroring the ROCm
// reference's native packed-weight grouped GEMM
// (rocm_grouped_gemm.hip MatmulBTQuantGroupedKernelRocm). The tower sits on
// device in its i32 WORD form (EnsureKeepQuantWords — the PACKED residency,
// never a bf16 twin), and each call decodes ONLY the P selected [N,K]
// row-slices (per group p: word rows [e*N*nb, (e+1)*N*nb)) through the
// bit-exact W3 chain on the slice (DecodeKeepQuantWordsF32), rounds the
// decoded weight ONCE to bf16 (the device's round-once convention), and runs
// the kMatmulBT tile matmul per group — the CPU provider's per-group
// structure (cpu_quant_gemm.cpp, the comparison oracle) with the decode
// on-core. The whole tower is NEVER decoded; no twin is built.
//
// REGISTERED SET: exactly {Q4_K, Q5_K, Q6_K, Q8_0} on kTENSTORRENT (W4a
// wave-2b widened the wave-2 pair: the 27B pin carries 67 Q6_K + 48 Q5_K
// tensors, so the tower is not servable without them). The two K-quant
// decodes are the W3 dense chains verbatim, run on the selected word slice —
// Q6_K mirrors the ROCm grouped kernel's native dequant
// (rocm_grouped_gemm.hip:1456), and Q5_K has NO ROCm grouped reference (the
// ROCm set is Q8_0/Q4_K/Q6_K), so its decode derives from the W3 dense Q5_K
// chain, bit-exact vs vt::cpu::BlockToFloat. Any other encoding refuses BY
// NAME. Never a silent wrong answer — the ROCm refusal precedent.
//
// W4a wave-3a (#3030) split the two arms by capture compatibility:
//  - E=1 (dense, the 27B path) is CHUNKED slice-decode + f32 assembly and
//    never reads the routing ids (statically all zero) — capture-clean and
//    trace-bounded, see the CHUNK POLICY comment in the arm;
//  - E=N (experts) keeps the wave-2 whole-slice decode and its dynamic-id
//    host readback; its capture indirection is staged-owed behind a MoE
//    artifact (spec ## W4), and it stays UNREACHED (no production entry
//    point until wave-3b wires the model).
constexpr int64_t kKeepQuantChunkPlaneBytes = 256 << 20;  // 256 MiB f32 plane
void MatmulBTQuantGroupedKernel(Queue&, Tensor& out, const Tensor& act,
                                const Tensor& weight,
                                const Tensor& expert_ids) {
  TT_OP_TRACE("MatmulBTQuantGrouped");
  VT_CHECK(act.rank == 2 && weight.rank == 2 && out.rank == 2,
           "tenstorrent kMatmulBTQuantGrouped: rank-2 act/weight/out required");
  const DType enc = weight.dtype;
  // vt::Name() emits the lowercase storage name ("q4_0"); the refusal must
  // name the ENUM the caller passed, so the k-prefix and capital go on here
  // (the MatmulBTQuantKernel convention).
  const std::string enc_lower = Name(enc);
  const std::string enc_name =
      std::string("k") + static_cast<char>(enc_lower[0] - 'a' + 'A') +
      enc_lower.substr(1);
  VT_CHECK(enc == DType::kQ4_K || enc == DType::kQ5_K ||
               enc == DType::kQ6_K || enc == DType::kQ8_0,
           std::string("tenstorrent kMatmulBTQuantGrouped: ") + enc_name +
               " has no GROUPED keep-quant decode on TENSTORRENT; the "
               "registered set is kQ4_K/kQ5_K/kQ6_K/kQ8_0 (BACKEND-"
               "TENSTORRENT-KEEPQUANT W4a)");
  const int64_t elems = BlockElems(enc);
  VT_CHECK(weight.shape[1] % elems == 0,
           std::string("tenstorrent kMatmulBTQuantGrouped: K must be a whole "
                       "number of ") +
               Name(enc) + " blocks (" + std::to_string(elems) + " elems)");
  VT_CHECK(IsFloatDType(act.dtype) &&
               (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kMatmulBTQuantGrouped: float activation, f32/bf16 out");
  VT_CHECK(act.IsContiguous() && weight.IsContiguous() && out.IsContiguous(),
           "tenstorrent kMatmulBTQuantGrouped: strided tensors are not "
           "supported in W4a wave-2");
  const int64_t P = out.shape[0];
  const int64_t N = out.shape[1];
  const int64_t K = act.shape[1];
  const int64_t Pa = act.shape[0];
  VT_CHECK(Pa == P || Pa == 1,
           "tenstorrent kMatmulBTQuantGrouped: act rows must be P (per-expert) "
           "or 1 (broadcast)");
  VT_CHECK(weight.shape[1] == K,
           "tenstorrent kMatmulBTQuantGrouped: act/weight inner dim mismatch");
  VT_CHECK(weight.shape[0] % N == 0,
           "tenstorrent kMatmulBTQuantGrouped: weight rows must be a whole "
           "multiple of N");
  VT_CHECK(out.shape[0] == P && out.shape[1] == N,
           "tenstorrent kMatmulBTQuantGrouped: out shape mismatch");
  const int64_t E = weight.shape[0] / N;
  const int64_t nb = K / elems;
  if (P == 0 || N == 0) return;

  MeshDevice& device = SharedMeshDevice();
  {  // W4d W0 (#3042) attribution: grouped arm shape.
    char lbl[80];
    std::snprintf(lbl, sizeof(lbl), "matmulbtq-grouped/E=%lld/N=%lld/K=%lld",
                  (long long)E, (long long)N, (long long)K);
    AllocTraceSnapshot(device, lbl);
  }

  // Stage the PACKED tower once — the resident i32 word shadow keyed by the
  // host weight pointer, served forever after (the dense arm's pattern). The
  // per-call decode reads word ROW-RANGES of it; a capture-time miss refuses
  // inside EnsureKeepQuantWords, as on the dense arm.
  const ttnn::Tensor words = EnsureKeepQuantWords(weight, enc, E * N, nb, device);

  // Activation serving (W3, capture-warmup redesign): device-resident
  // in-region. The engine's producers commit their outputs device-side in
  // every phase (the projections; MoeSiluMul's CommitDevice2D), so the
  // activation's CURRENT device shadow IS the value this pass computed —
  // serve it through the SAME op chain in the eager and the captured pass,
  // so the in-region typecast/layout programs the serve runs are warmed by
  // the eager step and hit the program cache under capture.
  //
  // This REPLACES the #3042 pointer-keyed GroupedActShadows staging: that
  // cache was keyed by the HOST activation pointer, and the engine's warmup
  // and capture steps allocate their activation scratch at DIFFERENT pool
  // addresses, so the capture lookup structurally missed (the "activation
  // staging miss during trace capture" refusal,
  // ISSUE-LOCAL-01M3918KQ580Z3NHVRNXVF15FZ). Serving the resident shadow
  // instead reads the address THIS pass's producer wrote — correct by
  // construction in both passes. The serve is value-identical to the host
  // staging it replaces: EnsureHost's download and the serve read the same
  // device bytes, the f32->bf16 typecast is the same RNE both places, and
  // the layout conversion moves no bits. A shadow-less arrival keeps the
  // from_span host staging (the op-level tests' hand-built activations), and
  // under capture that arm refuses by name — the producer must commit
  // device-side before the captured region (the GDN serve doctrine).
  ttnn::Tensor dev_a;
  {
    ttnn::Tensor raw;
    const uint32_t ua = static_cast<uint32_t>(Pa);
    const uint32_t uk = static_cast<uint32_t>(K);
    if (ServeDeviceShadowRaw(act, ua, uk, raw) ||
        ServeDeviceWindow(act, ua, uk, raw)) {
      // The volume-equal shadow may carry its producer's NATIVE logical
      // geometry — the GDN chain commits the gated-norm output at the
      // head-row form [T*Hv, Dv] over the same flat bytes the projection
      // consumes as [T, Hv*Dv] (the host staging this replaces reinterpreted
      // them implicitly through the host round-trip). Reinterpret at the
      // consumer's geometry: same numel, and for the TILE forms the same
      // tile count, so CaptureSafeReshape is the metadata view both passes
      // share (W4 makes its two branches one free reshape).
      const auto ls = raw.logical_shape();
      if (ls.rank() != 2 || ls[0] != ua || ls[1] != uk)
        raw = CaptureSafeReshape(std::move(raw), ttnn::Shape({ua, uk}));
      if (raw.dtype() == ttnn::DataType::FLOAT32)
        raw = ttnn::typecast(std::move(raw), ttnn::DataType::BFLOAT16);
      if (raw.layout() != ttnn::Layout::TILE)
        raw = ttnn::to_layout(std::move(raw), ttnn::Layout::TILE);
      dev_a = std::move(raw);
    } else {
      // The HOST-STAGED fallback (narrowed W3): an activation this kernel
      // itself staged from host bytes earlier in the SAME buffer is served
      // from the staged-tensor cache — pointer identity holds by
      // construction for that pattern (one buffer, staged eagerly then
      // captured, no pool recycling in between: the op-level capture
      // tests). The engine's pool-recycled activations structurally miss
      // here and reach the refusal below, which is the design.
      if (tt_capture_active()) {
        std::lock_guard<std::mutex> g(GroupedActMutex());
        auto it = GroupedActShadows().find(reinterpret_cast<uintptr_t>(act.data));
        VT_CHECK(it != GroupedActShadows().end() &&
                     it->second.rows == ua && it->second.cols == uk &&
                     it->second.dtype == act.dtype,
                 "tenstorrent grouped-quant: activation arrived without a "
                 "servable device shadow during trace capture — the producer "
                 "must commit device-side before the captured region");
        raw = it->second.device;
        const auto ls = raw.logical_shape();
        if (ls.rank() != 2 || ls[0] != ua || ls[1] != uk)
          raw = CaptureSafeReshape(std::move(raw), ttnn::Shape({ua, uk}));
        if (raw.dtype() == ttnn::DataType::FLOAT32)
          raw = ttnn::typecast(std::move(raw), ttnn::DataType::BFLOAT16);
        if (raw.layout() != ttnn::Layout::TILE)
          raw = ttnn::to_layout(std::move(raw), ttnn::Layout::TILE);
        dev_a = std::move(raw);
      } else {
      EnsureHost(act);
      if (act.dtype == DType::kF32) {
        ttnn::Tensor dev_f32 = ttnn::Tensor::from_span(
            ttsl::Span<const float>(
                act.Ptr<float>(),
                static_cast<size_t>(Pa) * static_cast<size_t>(K)),
            SpecOf(tt::tt_metal::Shape({ua, uk}),
                   ttnn::DataType::FLOAT32, ttnn::Layout::ROW_MAJOR),
            &device);
        dev_a = ttnn::to_layout(
            ttnn::typecast(std::move(dev_f32), ttnn::DataType::BFLOAT16),
            ttnn::Layout::TILE);
      } else {
        ttnn::Tensor dev_bf16 = ttnn::Tensor::from_span(
            ttsl::Span<const bfloat16>(
                act.Ptr<bfloat16>(),
                static_cast<size_t>(Pa) * static_cast<size_t>(K)),
            SpecOf(tt::tt_metal::Shape({ua, uk}),
                   ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR),
            &device);
        dev_a = ttnn::to_layout(std::move(dev_bf16), ttnn::Layout::TILE);
      }
      // Register the staged tensor under THIS buffer's host pointer so a
      // later capture of the same host-staged buffer can serve it (the
      // narrowed warm-first contract above).
      std::lock_guard<std::mutex> g(GroupedActMutex());
      GroupedActShadow& sh =
          GroupedActShadows()[reinterpret_cast<uintptr_t>(act.data)];
      sh.device = dev_a;
      sh.rows = ua;
      sh.cols = uk;
      sh.dtype = act.dtype;
      }
    }
  }
  ttnn::Tensor a_rows;
  if (E > 1 && Pa > 1)
    a_rows = ttnn::to_layout(std::move(dev_a), ttnn::Layout::ROW_MAJOR);

  const uint32_t wpb = static_cast<uint32_t>(KeepQuantWordsPerBlock(enc));

  if (E == 1) {
    // == W4a wave-3a (#3030): the DENSE arm — CHUNKED slice-decode + f32
    // assembly, capture-clean. The fourth spec amendment (b40907ee2) makes
    // this arm the row's production surface: the whole keep-quant set beyond
    // the gather class is served PACKED through E=1, and a captured graph
    // must never hold a whole-weight tile. Wave-1b falsified that shape on
    // the vehicle; the chunk-count survey below (P150, 2026-09-07) measured
    // BOTH capture-time failure modes on the head shape [248320, 1024] Q6_K:
    // one whole-weight chunk dies on DEVICE DRAM (bank_manager.cpp:462 — a
    // 1.02 GiB f32 decode plane), while many small chunks die on the TRACE
    // REGION (mesh_trace.cpp:81): the captured command stream costs ~3.3 MB
    // per chunk (485 chunks = 1,566,662,656 B; 16 chunks = 53,764,096 B;
    // 2-4 chunks capture clean). CHUNK POLICY: decode [chunk, K] word
    // ranges — one tile matmul per chunk — so the live working set is the
    // i32 word slice, the chain's f32 planes and the bf16 tile of ONE chunk
    // plus the tiny f32 partials, never a whole-weight tile; each chunk's
    // decoded tile dies before the next chunk allocates. The budget bounds
    // the chain's largest live tensor — one [chunk, K] f32 plane — at
    // 256 MiB (chunk = 256 MiB / (4 B . K), and a whole decode for any
    // weight whose plane fits), while ceil(N / 8) keeps the command stream
    // under the 52,428,800 B trace region for weights large enough to
    // chunk. Chunks cover DISJOINT weight rows, so every output element is
    // still ONE dot over the full K and chunks concatenate in f32; the
    // decode itself is unchanged (the bit-exact leg pins it). Capture-clean:
    // E=1 ids are statically all zero — the only in-range expert — so the
    // routing ids are never EnsureHosted or read, and every chunk offset is
    // a capture-time constant replayed verbatim.
    const int64_t chunk_override =
        KeepQuantChunkRowsOverride().load(std::memory_order_relaxed);
    // W4a wave-3b-2 (#3030): the plane budget is env-tunable the way
    // VT_TT_TRACE_REGION_MB is. The 256 MiB default was surveyed on the 0.8B
    // vehicle; at 27B the first forward's ffn_down chunk died with 244 MB
    // free and a 105 MB largest block, so the gate recipe can shrink the
    // plane without a rebuild. Empty/unset keeps the surveyed default.
    int64_t plane_bytes = kKeepQuantChunkPlaneBytes;
    bool plane_env_set = false;
    if (const char* plane_env = std::getenv("VT_TT_KEEPQUANT_CHUNK_BYTES");
        plane_env != nullptr && plane_env[0] != '\0') {
      const long long parsed = std::atoll(plane_env);
      if (parsed > 0) {
        plane_bytes = static_cast<int64_t>(parsed);
        plane_env_set = true;
      }
    }
    int64_t chunk =
        chunk_override > 0
            ? std::min(chunk_override, N)
            : std::min(N, std::max<int64_t>(
                              plane_bytes / (K * 4),
                              (N + 7) / 8));
    // W4a wave-3b-2 (#3030): the env knob is a HARD CAP — the ceil(N/8)
    // trace term above forces N/8-row chunks for wide-N weights (the head
    // [248320, 5120] would decode 31040-row planes, 606+ MB, whatever the
    // plane budget says), which is exactly the alloc that died at 27B. A
    // set knob trades command-stream length for live memory, the same
    // trade VT_TT_TRACE_REGION_MB records on its axis.
    if (plane_env_set)
      chunk = std::min(chunk, std::max<int64_t>(plane_bytes / (K * 4), 1));
    // The repair lambda (DecodeKeepQuantWordsF32) tiles the {B,16,16}
    // product planes to {B,32,32} — 4x the f32 plane the budget above
    // counts, the 27B-head OOM (ISSUE-LOCAL-01M2N8DKVKM2J03FCVBSKYVHXK:
    // 2.5 GB demanded, 238 MB largest free block). Cap eager chunks at the
    // tiled plane — plane_bytes/(K*16) reproduces the proven 3,276-row
    // chunk on the head.
    //
    // The policy is STICKY PER WEIGHT, not per call: a captured replay must
    // hash-match the eager warm-up's stream, and a per-call capture flag
    // makes warm-up and capture diverge (the slice extent differs, the
    // capture-time SliceDeviceOperation misses the program cache, and every
    // later test poisons — the suite regression this replaced). The first
    // decode of a weight is always eager — a capture-time arrival with a
    // cold shadow refuses at EnsureKeepQuantWords — so the flag is recorded
    // eagerly and replayed verbatim under capture.
    bool tile_cap;
    {
      std::lock_guard<std::mutex> g(KeepQuantWordMutex());
      static auto* tile_cap_policy =
          new std::unordered_map<intptr_t, bool>();
      auto key = reinterpret_cast<intptr_t>(weight.data);
      auto it = tile_cap_policy->find(key);
      if (it != tile_cap_policy->end()) {
        tile_cap = it->second;
      } else {
        tile_cap = !tt_capture_active();
        tile_cap_policy->emplace(key, tile_cap);
      }
    }
    if (tile_cap)
      chunk = std::min(chunk, std::max<int64_t>(plane_bytes / (K * 16), 1));
    // == TT-DECODE-FUSION wave 2 (TT-MATMUL-FUSION): the whole-decode fused
    // arm. When the served encoding is Q6_K and the WHOLE weight's decoded
    // f32 plane fits the same budget the chunk loop already enforces (the
    // prod plane counted too on the exact-f32 decode arm), the chunk loop is
    // bypassed: ONE fused-kernel launch decodes the entire word shadow (the
    // wave-1 kernel, whole-extent rows — no slice, no per-chunk programs),
    // then the chain's own single-chunk tail runs unchanged (typecast ->
    // TILE -> ONE matmul / exact-f32 broadcast-dot -> the ROW_MAJOR partial).
    // A launch that enqueued ~20 tt-metal programs (the per-chunk
    // slice/decode/typecast/layout/matmul chains plus the concat) now
    // enqueues ~6 (the fused decode, the typecast/layout pair, the matmul,
    // the partial's typecast/layout pair) — the class-split verdict's
    // program-count lever, the safe route: the matmul itself stays stock.
    // Bit-exact by construction: the decode is the same kernel on the same
    // word rows (the whole-extent window the chain's sl_alias case names),
    // and every downstream op is the chain's own single-chunk iteration at
    // the identical shape (chunking splits OUTPUT columns only, so a
    // whole-weight dot equals the concatenated chunk dots elementwise).
    // Declines: an unserved decode shape (env off, L1 over budget) falls to
    // the chunk chain BY NAME — and identically in the eager and the
    // captured pass (the decline is shape-determined), so the two passes
    // never diverge in arm.
    AllocTraceSnapshot(device, "KQuantGrouped/chunk-loop/pre");
    AllocTraceSnapshot(device, "KQuantGrouped/chunk-loop/pre");
    // Decode (P == 1) keeps the decoded f32 weight tile and widens the
    // staged activation to f32; the chunk dot runs in exact f32 SFPU (see
    // below). Prefill (P > 1) keeps the bf16 activation tile for the bf16
    // TILE matmul. The widen is a device-side typecast of the ALREADY
    // STAGED activation shadow — never a fresh host upload, which would be
    // a write inside trace capture — so captured replays see the identical
    // stream, and the bf16-staged values widen exactly to f32.
    ttnn::Tensor dev_a_f32exact;
    const bool f32exact = P == 1;
    if (f32exact)
      dev_a_f32exact = ttnn::typecast(dev_a, ttnn::DataType::FLOAT32);
    // The wave-2/wave-3 arm gate (see the comment above the chunk loop):
    // Q6_K or Q4_K on the fused decode kernel, the WHOLE decoded plane (the
    // exact-f32 broadcast-dot's prod plane counted too) inside the same
    // budget the chunk loop enforces. VT_TT_KEEPQUANT_MM_CHAIN=1 forces the
    // chunk chain even where the fused arm serves — the named fallback kill
    // switch, and the doctest's chain baseline leg.
    bool force_chain = false;
    if (const char* mm_chain = std::getenv("VT_TT_KEEPQUANT_MM_CHAIN");
        mm_chain != nullptr && mm_chain[0] != '\0' &&
        std::strcmp(mm_chain, "0") != 0)
      force_chain = true;
    bool fused_tile_tail = false;
    bool fused_mm = !force_chain &&
                    (enc == DType::kQ6_K || enc == DType::kQ4_K) &&
                    KeepQuantFusedDecodeEnabled() &&
                    (f32exact ? 2 : 1) * N * K * 4 <= plane_bytes;
    std::optional<ttnn::Tensor> fused_wf;
    if (fused_mm) {
      fused_wf = DecodeKeepQuantWordsFusedQK(words, enc, N, nb, device);
      if (!fused_wf) {
        std::fprintf(stderr,
                     "[TT-KQ-FUSED-MM] whole-decode fused arm declined "
                     "(rows=%lld nb=%lld) — the chunk chain serves\n",
                     (long long)N, (long long)nb);
        fused_mm = false;
      }
    }
    std::vector<ttnn::Tensor> partials;
    partials.reserve(static_cast<size_t>(
        fused_mm ? 1 : (N + chunk - 1) / chunk));
    if (fused_mm) {
      // The fused arm's single "chunk": the whole [N, K] decoded plane from
      // the ONE launch above, then the chain's own single-chunk tail verbatim
      // (sl_alias holds — the decode consumed the full-extent window).
      ttnn::Tensor wf = std::move(*fused_wf);
      ttnn::Tensor part;
      if (f32exact) {
        ttnn::Tensor wft = ttnn::to_layout(wf, ttnn::Layout::TILE);
        TTReclaimPlanes(device, {&wf});
        ttnn::Tensor prod = ttnn::multiply(wft, dev_a_f32exact);
        part = ttnn::permute(
            ttnn::sum(prod, ttsl::SmallVector<int>{1}, /*keep_dim=*/true),
            ttsl::SmallVector<int64_t>{1, 0});  // [rows,1] -> [1,rows]
        TTReclaimPlanes(device, {&wft, &prod});
      } else {
        ttnn::Tensor wbf = ttnn::typecast(wf, ttnn::DataType::BFLOAT16);
        TTReclaimPlanes(device, {&wf});
        ttnn::Tensor wb = ttnn::to_layout(wbf, ttnn::Layout::TILE);
        TTReclaimPlanes(device, {&wbf});
        part = ttnn::operations::matmul::matmul(
            dev_a, wb, /*transpose_a=*/false, /*transpose_b=*/true);
        TTReclaimPlanes(device, {&wb});
      }
      // The prefill arm's TILE fast tail: the matmul output typecasts f32
      // IN the TILE domain (an elementwise op — every value bit-identical
      // to the chain's ROW_MAJOR round-trip, which only reorders), and the
      // assembly below commits it as-is — two fewer programs per launch,
      // which is what puts the whole launch under the 32 KiB gate. The
      // exact-f32 arm keeps the chain's permuted tail (its partial is the
      // [1, rows] column, not a TILE).
      fused_tile_tail = !f32exact;
      ttnn::Tensor partf = ttnn::typecast(part, ttnn::DataType::FLOAT32);
      TTReclaimPlanes(device, {&part});
      partials.push_back(std::move(partf));
    } else {
    for (int64_t c0 = 0; c0 < N; c0 += chunk) {
      const int64_t c1 = std::min(N, c0 + chunk);
      ttnn::Tensor sl = ttnn::slice(
          words,
          ttsl::SmallVector<uint32_t>{static_cast<uint32_t>(c0 * nb), 0u},
          ttsl::SmallVector<uint32_t>{static_cast<uint32_t>(c1 * nb), wpb},
          ttsl::SmallVector<uint32_t>{1u, 1u});
      ttnn::Tensor wf = DecodeKeepQuantWordsF32(sl, enc, c1 - c0, nb, device);
      // W4d W2 (#3042): the chunk's staged slice and decode output are
      // consumed by the typecast/to_layout chain below — reclaimed as soon
      // as each is read. A slice materializes its own buffer EXCEPT the
      // full-extent one: slice() returns its input for a whole-tensor
      // window (tt-metal slice.cpp:182), so the single-chunk case
      // (c0 == 0 && c1 == N) makes sl the resident word shadow, and the
      // forced reclaim would kill the cache entry for every later call
      // (the full-suite regression: a dead shadow came back as an
      // unallocated tensor inside slice). Leave sl to the shadow there.
      const bool sl_alias = c0 == 0 && c1 == N;
      // Exact f32 dot for the single-activation-row (decode) case. The
      // bf16 arm rounds BOTH the decoded weight tile and the f32 activation
      // to bf16 before ttnn::matmul — a full bf16 GEMM where the CPU oracle
      // computes the quantized dot in f32 — and that rounding is the dominant
      // remaining decode drift (measured: grouped-arm rel_rms 2-12% vs
      // int8dot 0.3%; ffn_down max_abs 5.33; lm_head logit max_abs 0.375
      // against a 0.25-0.34 nat oracle tie gap).
      ttnn::Tensor part;
      ttnn::Tensor prod;
      ttnn::Tensor wft;
      if (f32exact) {
        wft = ttnn::to_layout(wf, ttnn::Layout::TILE);
        if (sl_alias)
          TTReclaimPlanes(device, {&wf});
        else
          TTReclaimPlanes(device, {&sl, &wf});
        // Exact f32 dot: 2D broadcast multiply (rows,K)x(1,K) — exact f32
        // SFPU — then an f32 ttnn::sum over the K dim (the accumulation the
        // CPU oracle computes), rotated back to [1, rows]. prod (the one
        // large plane in this chain) is reclaimed at the loop bottom; part,
        // the [rows,1] permute column, may alias prod's buffer and is left
        // to the refcounted free (it is 4 bytes per row — an orphan is
        // bounded), so a forced double free of the shared storage cannot
        // happen.
        prod = ttnn::multiply(wft, dev_a_f32exact);
        part = ttnn::permute(
            ttnn::sum(prod, ttsl::SmallVector<int>{1}, /*keep_dim=*/true),
            ttsl::SmallVector<int64_t>{1, 0});  // [rows,1] -> [1,rows]
      } else {
        ttnn::Tensor wbf = ttnn::typecast(wf, ttnn::DataType::BFLOAT16);
        ttnn::Tensor wb = ttnn::to_layout(wbf, ttnn::Layout::TILE);
        if (sl_alias)
          TTReclaimPlanes(device, {&wf, &wbf});
        else
          TTReclaimPlanes(device, {&sl, &wf, &wbf});
        part = ttnn::operations::matmul::matmul(
            dev_a, wb, /*transpose_a=*/false, /*transpose_b=*/true);
        TTReclaimPlanes(device, {&wb});
      }
      ttnn::Tensor partf = ttnn::typecast(part, ttnn::DataType::FLOAT32);
      ttnn::Tensor partl = ttnn::to_layout(partf, ttnn::Layout::ROW_MAJOR);
      if (f32exact)
        TTReclaimPlanes(device, {&wft, &prod});
      else
        TTReclaimPlanes(device, {&part, &partf});
      partials.push_back(std::move(partl));
    }
    }  // !fused_mm
    AllocTraceSnapshot(device, "KQuantGrouped/chunk-loop/post");
    ttnn::Tensor assembled;
    if (fused_tile_tail) {
      // The TILE fast tail: the single f32 TILE partial IS the commit form;
      // no concat, no ROW_MAJOR round-trip.
      assembled = std::move(partials[0]);
    } else {
    assembled =
        partials.size() == 1
            ? std::move(partials[0])
            : ttnn::concat(partials, /*dim=*/1);
    // W4d W2 (#3042): the concat read every partial — reclaim them. The
    // single-partial form moved the only entry into assembled; its buffer
    // IS the output and is never freed here.
    if (partials.size() > 1) TTReclaimPlanes(device, partials);
    }
    if (Pa == 1 && P > 1) {
      // Broadcast contract: every output row is the SAME [1, K] activation
      // against expert 0 — replicate the assembled row. Bit-identical to the
      // wave-2 per-group decode (identical operands, identical programs).
      std::vector<ttnn::Tensor> rows(static_cast<size_t>(P), assembled);
      ttnn::Tensor bred = ttnn::concat(rows, /*dim=*/0);
      // The replicate inputs share assembled's buffer (handle copies); one
      // forced free removes it for every alias (the holder tombstone makes
      // the repeats no-ops).
      TTReclaimPlanes(device, {&assembled});
      TTReclaimPlanes(device, rows);
      assembled = std::move(bred);
    }
    if (out.dtype == DType::kBF16) {
      ttnn::Tensor abf = ttnn::typecast(assembled, ttnn::DataType::BFLOAT16);
      TTReclaimPlanes(device, {&assembled});
      assembled = std::move(abf);
    }
    // Commit form: TILE — the layout the twin path's matmul output carried.
    // The ROW_MAJOR assembly above is an internal concat domain only. A
    // ROW_MAJOR commit leaves a ROW_MAJOR slot for the next consumer, and
    // downstream consumers build tile-padded views over that slot's buffer
    // (the vehicle's rope -> paged-KV RAC reshape view exceeded the buffer:
    // mesh_tensor_impl.hpp packed-size fatal on the replay step). Values are
    // unchanged; only the committed slot's layout lands as the twin's did.
    if (assembled.layout() != ttnn::Layout::TILE) {
      ttnn::Tensor atl = ttnn::to_layout(assembled, ttnn::Layout::TILE);
      TTReclaimPlanes(device, {&assembled});
      assembled = std::move(atl);
    }
    CommitDeviceLogical2D(out, std::move(assembled), static_cast<uint32_t>(P),
                          static_cast<uint32_t>(N));
    // The committed slot owns assembled's buffer through the moved handle
    // (the W1 trace books the committed slots as the legit residency), so
    // nothing is reclaimed past the commit.
    return;
  }

  // E=N EXPERT TOWER arm: the wave-2 path unchanged. The routing ids are
  // dynamic here; the established TT index-tensor contract is EnsureHost + a
  // host read (EmbeddingKernel), range-checked like the embedding gather.
  // That host readback is the arm's eager construct, and its capture
  // indirection is staged-owed behind a MoE artifact (spec ## W4).
  EnsureHost(expert_ids);
  const int32_t* eids = expert_ids.Ptr<int32_t>();
  for (int64_t p = 0; p < P; ++p)
    VT_CHECK(eids[p] >= 0 && eids[p] < E,
             "tenstorrent kMatmulBTQuantGrouped: expert id out of range (id " +
                 std::to_string(eids[p]) + ", E " + std::to_string(E) + ")");

  // The selected [N,K] slice for group p: word rows [e*N*nb, (e+1)*N*nb) —
  // decode, one bf16 RNE, TILE — the dense dot's exact weight convention.
  auto slice_decode = [&](int64_t p) {
    const int64_t w0 = eids[p] * N * nb;
    ttnn::Tensor sl = ttnn::slice(
        words, ttsl::SmallVector<uint32_t>{static_cast<uint32_t>(w0), 0u},
        ttsl::SmallVector<uint32_t>{static_cast<uint32_t>(w0 + N * nb), wpb},
        ttsl::SmallVector<uint32_t>{1u, 1u});
    ttnn::Tensor wf = DecodeKeepQuantWordsF32(sl, enc, N, nb, device);
    // W4d W2 (#3042): the same reclaim as the chunk loop — the staged slice
    // and decode output die into the typecast/to_layout chain. The sl_alias
    // guard below is DEFENSIVE ONLY: this lambda runs in the E > 1 arm,
    // where the word shadow always holds E*N*nb rows, so the identity's
    // second term — words.logical_shape()[0] == N*nb, the full-extent window
    // where slice() returns its input — can never hold (it would demand
    // E == 1, and the E == 1 arm returned long before this lambda). The
    // reclaim never actually skips sl today; the guard stays so the shadow
    // is checked, not assumed, if this lambda is ever shared with the
    // single-expert arm.
    const bool sl_alias =
        w0 == 0 &&
        static_cast<int64_t>(words.logical_shape()[0]) == N * nb;
    ttnn::Tensor wbf = ttnn::typecast(wf, ttnn::DataType::BFLOAT16);
    ttnn::Tensor out_t = ttnn::to_layout(wbf, ttnn::Layout::TILE);
    if (sl_alias)
      TTReclaimPlanes(device, {&wf, &wbf});
    else
      TTReclaimPlanes(device, {&sl, &wf, &wbf});
    return out_t;
  };

  std::vector<ttnn::Tensor> outs;
  outs.reserve(static_cast<size_t>(P));
  for (int64_t p = 0; p < P; ++p) {
    ttnn::Tensor a_p;
    if (Pa == 1) {
      a_p = dev_a;
    } else {
      a_p = ttnn::to_layout(
          ttnn::slice(a_rows,
                      ttsl::SmallVector<uint32_t>{static_cast<uint32_t>(p), 0u},
                      ttsl::SmallVector<uint32_t>{static_cast<uint32_t>(p + 1),
                                                  static_cast<uint32_t>(K)},
                      ttsl::SmallVector<uint32_t>{1u, 1u}),
          ttnn::Layout::TILE);
    }
    outs.push_back(ttnn::operations::matmul::matmul(
        std::move(a_p), slice_decode(p), /*transpose_a=*/false,
        /*transpose_b=*/true));
  }
  // Assemble [P,N] and commit ONCE (the slot is per host pointer, so the
  // commit must be a single whole-output store). The tile matmul output is
  // bf16, committed as the dense arm commits it; the P > 1 assembly goes
  // through ROW_MAJOR f32 (the chains' commit form) and lands out.dtype.
  ttnn::Tensor assembled;
  if (P == 1) {
    assembled = std::move(outs[0]);
  } else {
    std::vector<ttnn::Tensor> rows_f;
    rows_f.reserve(outs.size());
    for (auto& t : outs)
      rows_f.push_back(ttnn::to_layout(
          ttnn::typecast(std::move(t), ttnn::DataType::FLOAT32),
          ttnn::Layout::ROW_MAJOR));
    assembled = ttnn::concat(std::move(rows_f), /*dim=*/0);
    if (out.dtype == DType::kBF16)
      assembled =
          ttnn::typecast(std::move(assembled), ttnn::DataType::BFLOAT16);
  }
  CommitDeviceLogical2D(out, std::move(assembled), static_cast<uint32_t>(P),
                        static_cast<uint32_t>(N));
}

// == W4b int8-dot (BACKEND-TENSTORRENT-KEEPQUANT, #3031): the custom device
// kernel below ttnn — the decided path (spec ## W4b, amendment 4f95e6fe4).
//
// The dense keep-quant matmul runs in the QUANTIZED DOMAIN: the packed words
// are never decoded to f32 and the activation is quantized ONCE per row to
// the encoding the CPU vec_dot pairs with the weight (q8_K for the K-quants,
// q8_0 for Q8_0 — ggml-cpu.c:230-326), then every output element is the
// upstream generic vec_dot itself, on-core, in the upstream 8-lane split
// (aux32[8] each <= 2^24, so every f32 `sums[l] += d*aux32[l]` is exact; the
// whole-block sum is NOT f32-exact, which is why the lane split and the
// lane-sum/min-correction interleave are preserved verbatim). The algorithm
// lives in kernels/keepquant_kernel_code.h — the SAME file the red-first
// sweep test pins on the DEVICE-COMPILED path (the header reaches the core
// through compiler_include_paths; no host test includes it) against
// vt::cpu::BlockVecDot/BlockFromFloat, across every registered encoding
// {Q4_K, Q5_K, Q6_K, Q8_0} — one numerics source of truth, not a device
// twin.
//
// CAPTURE (the shrink-or-flat gate): the per-call program enqueues ONE
// MeshWorkload on the trace cq (QueueId 0, the cq ttnn traces), so the
// captured graph holds one launch instead of the W4a chunk chain's hundreds
// of ttnn programs per chunk. Everything the kernel reads is a stable address
// at capture time: the word shadow (EnsureKeepQuantWords, warmed eagerly),
// the ROW_MAJOR activation view (slot-hit or a to_layout conversion recorded
// ahead of the launch in the same trace), and the freshly allocated out
// tensor whose buffer the capture bakes. The activation quantization runs
// ON-CORE inside the launch, so replay re-quantizes fresh activations —
// nothing capture-time is frozen into the graph (the #2812 class).
//
// SHAPE VARIANCE IS RUNTIME: every size/count/address reaches the kernel as
// launch arguments, so tt-metal's kernel cache compiles the source exactly
// once per (accessor page geometry) — the eager warm step pays it before any
// capture.
namespace {

// The device kernel source. keepquant_kernel_code.h comes from the repo
// kernels/ dir via compiler_include_paths (resolved from __FILE__ below), so
// the shipped tree — not a copy — is what runs on-core (wave 2).
constexpr const char* kKeepQuantInt8DotKernelSrc = R"TTKQ(
#include "api/dataflow/dataflow_api.h"
#include "keepquant_kernel_code.h"

// Shape-global runtime args (the SetCommonRuntimeArgs stream). The 27B
// trace-fit fix (tt-launch-record-attribution-20260928): every word here is
// identical across cores, so the program records ONE common-args page per
// captured launch instead of one per-core Unique RTA UNICAST page per core —
// 11,040 recorded pages ≈ 2.97 MB per launch is what blew the 27B whole-graph
// trace out of DRAM. Indices 0-2 are the three bank bases the original
// common-args set carried; the shape words follow. The only per-core-varying
// words (row0 = c*tcols, rowc = its clamp) are derived in-kernel from the
// core coordinate.
constexpr uint32_t CARG_W_ADDR = 0;
constexpr uint32_t CARG_A_ADDR = 1;
constexpr uint32_t CARG_O_ADDR = 2;
constexpr uint32_t CARG_M = 3;
constexpr uint32_t CARG_K = 4;
constexpr uint32_t CARG_N = 5;
constexpr uint32_t CARG_NB = 6;        // weight blocks per row (K / elems)
constexpr uint32_t CARG_WPB = 7;       // staged i32 words per block
constexpr uint32_t CARG_ACT_F32 = 8;   // 1: f32 activation bytes, 0: bf16
constexpr uint32_t CARG_MTILE = 9;     // activation rows per quantize tile
constexpr uint32_t CARG_QB_PAD = 10;   // 16B-aligned activation-quant row bytes
constexpr uint32_t CARG_ENC = 11;      // 0..13, the enc-select dispatch below
constexpr uint32_t CARG_TCOLS = 12;    // padded tile width: groups_per_core*4
constexpr uint32_t CARG_GRID_X = 13;   // core-grid width: c = y*grid_x + x
constexpr uint32_t kNumCommonArgs = 14;

// CB scratch (self-cycled: reserve -> use -> push -> pop; no consumer core).
constexpr uint32_t CB_F32 = 0;  // one activation row widened to f32 (K*4 B)
                                //   plus the raw bf16 tail (K*2 B)
constexpr uint32_t CB_Q8 = 1;   // mtile quantized activation rows
constexpr uint32_t CB_W = 2;    // one weight row's packed words (nb*wpb*4 B)
constexpr uint32_t CB_O = 3;    // out staging, mtile x tcols f32 (tcols % 4 == 0)

void kernel_main() {
  // Three interleaved DRAM tensors: packed words, activation, out. Interleaved
  // accessors consume two ct args each (config word + aligned page size) and
  // zero crta words; the bank bases are the three words SetCommonRuntimeArgs
  // pushes (address, then accessor's own words — none — then the next).
  constexpr auto args_w = TensorAccessorArgs<0, 0>();
  constexpr uint32_t cta_1 = args_w.next_compile_time_args_offset();
  constexpr auto args_a = TensorAccessorArgs<cta_1, 0>();
  constexpr uint32_t cta_2 = args_a.next_compile_time_args_offset();
  constexpr auto args_o = TensorAccessorArgs<cta_2, 0>();
  const auto acc_w =
      TensorAccessor(args_w, get_common_arg_val<uint32_t>(0));
  const auto acc_a =
      TensorAccessor(args_a, get_common_arg_val<uint32_t>(1));
  const auto acc_o =
      TensorAccessor(args_o, get_common_arg_val<uint32_t>(2));

  const uint32_t M = get_common_arg_val<uint32_t>(CARG_M);
  const uint32_t K = get_common_arg_val<uint32_t>(CARG_K);
  const uint32_t N = get_common_arg_val<uint32_t>(CARG_N);
  const uint32_t nb = get_common_arg_val<uint32_t>(CARG_NB);
  const uint32_t wpb = get_common_arg_val<uint32_t>(CARG_WPB);
  const uint32_t act_f32 = get_common_arg_val<uint32_t>(CARG_ACT_F32);
  const uint32_t mtile = get_common_arg_val<uint32_t>(CARG_MTILE);
  const uint32_t qb_pad = get_common_arg_val<uint32_t>(CARG_QB_PAD);
  const uint32_t enc = get_common_arg_val<uint32_t>(CARG_ENC);
  const uint32_t tcols = get_common_arg_val<uint32_t>(CARG_TCOLS);  // padded tile width
  const uint32_t grid_x = get_common_arg_val<uint32_t>(CARG_GRID_X);
  // The per-core slice, derived from the core coordinate instead of a
  // per-core SetRuntimeArgs word (the 27B trace-fit fix): c enumerates the
  // grid row-major, exactly the host's {c % grid.x, c / grid.x} mapping.
  const uint32_t c =
      static_cast<uint32_t>(get_relative_logical_y()) * grid_x +
      static_cast<uint32_t>(get_relative_logical_x());
  const uint32_t row0 = c * tcols;
  const uint32_t rowc =
      row0 >= N ? 0u
                : ((tcols < N - row0) ? tcols : (N - row0));
  if (rowc == 0 || M == 0) return;

  const uint32_t word_bytes = wpb * 4;
  const uint32_t act_bytes = K * (act_f32 ? 4u : 2u);

  for (uint32_t m0 = 0; m0 < M; m0 += mtile) {
    const uint32_t mr = (m0 + mtile <= M) ? mtile : (M - m0);
    // Quantize this tile's activation rows into the core-private q8 scratch.
    cb_reserve_back(CB_Q8, mr);
    const uint32_t q8_base = get_write_ptr(CB_Q8);
    for (uint32_t r = 0; r < mr; ++r) {
      cb_reserve_back(CB_F32, 1);
      const uint32_t fp = get_write_ptr(CB_F32);
      if (act_f32) {
        noc_async_read(acc_a.get_noc_addr(m0 + r), fp, act_bytes);
        noc_async_read_barrier();
      } else {
        // bf16 -> f32 is exact (f32 bits = u16 << 16). Read the raw u16 row
        // into the tail PAST the widened row (fp is a byte address, so the
        // tail starts at fp + K*4), then convert backward.
        noc_async_read(acc_a.get_noc_addr(m0 + r), fp + K * 4, act_bytes);
        noc_async_read_barrier();
        uint32_t* dst = reinterpret_cast<uint32_t*>(fp);
        const uint16_t* src =
            reinterpret_cast<const uint16_t*>(fp + K * 4);
        for (int32_t j = static_cast<int32_t>(K) - 1; j >= 0; --j)
          dst[j] = static_cast<uint32_t>(src[j]) << 16;
      }
      cb_push_back(CB_F32, 1);
      cb_pop_front(CB_F32, 1);
      uint8_t* q8row = reinterpret_cast<uint8_t*>(q8_base) + r * qb_pad;
      if (enc == 3)
        kq_quantize_row_q8_0(reinterpret_cast<const uint8_t*>(fp), q8row, K);
      else
        kq_quantize_row_q8_K(reinterpret_cast<const uint8_t*>(fp), q8row, K);
    }
    cb_push_back(CB_Q8, mr);

    cb_reserve_back(CB_W, 1);
    const uint32_t wp = get_write_ptr(CB_W);
    cb_reserve_back(CB_O, mr);
    const uint32_t op = get_write_ptr(CB_O);
    // Pad columns [rowc, tcols) of every tile row carry ZEROS: the last core's
    // partial 4-column group writes the whole 16-B sector, and the pad floats
    // land in the out page's alignment padding, never in a logical cell.
    for (uint32_t z = 0; z < mr * tcols; ++z)
      reinterpret_cast<uint32_t*>(op)[z] = 0u;

    // Each assigned weight row streams in once per tile and dots against
    // every quantized activation row — the m-tile loop bounds the q8 L1
    // residency, not the math.
    for (uint32_t n = 0; n < rowc; ++n) {
      const uint32_t grow = row0 + n;
      for (uint32_t b = 0; b < nb; ++b)
        noc_async_read(acc_w.get_noc_addr(grow * nb + b),
                       wp + b * word_bytes, word_bytes);
      noc_async_read_barrier();
      float* out_tile = reinterpret_cast<float*>(op);
      for (uint32_t r = 0; r < mr; ++r) {
        const uint8_t* xw = reinterpret_cast<const uint8_t*>(wp);
        const uint8_t* yq =
            reinterpret_cast<const uint8_t*>(q8_base) + r * qb_pad;
        float v = 0.0f;
        if (enc == 0)
          v = kq_vec_dot_q4_K_q8_K(xw, word_bytes, yq, nb);
        else if (enc == 1)
          v = kq_vec_dot_q5_K_q8_K(xw, word_bytes, yq, nb);
        else if (enc == 2)
          v = kq_vec_dot_q6_K_q8_K(xw, word_bytes, yq, nb);
        else if (enc == 3)
          v = kq_vec_dot_q8_0_q8_0(xw, word_bytes, yq, nb);
        else if (enc == 4)
          v = kq_vec_dot_iq3_xxs_q8_K(xw, word_bytes, yq, nb);
        else if (enc == 5)
          v = kq_vec_dot_iq2_xxs_q8_K(xw, word_bytes, yq, nb);
        else if (enc == 6)
          v = kq_vec_dot_iq2_s_q8_K(xw, word_bytes, yq, nb);
        else if (enc == 7)
          v = kq_vec_dot_q3_k_q8_K(xw, word_bytes, yq, nb);
        else if (enc == 8)
          v = kq_vec_dot_iq3_s_q8_K(xw, word_bytes, yq, nb);
        else if (enc == 9)
          v = kq_vec_dot_iq4_xs_q8_K(xw, word_bytes, yq, nb);
        else if (enc == 10)
          v = kq_vec_dot_iq2_xs_q8_K(xw, word_bytes, yq, nb);
        else if (enc == 11)
          v = kq_vec_dot_q2_k_q8_K(xw, word_bytes, yq, nb);
        else if (enc == 12)
          v = kq_vec_dot_iq1_s_q8_K(xw, word_bytes, yq, nb);
        else if (enc == 13)
          v = kq_vec_dot_iq1_m_q8_K(xw, word_bytes, yq, nb);
        else
          v = kq_vec_dot_iq2_s_q8_K(xw, word_bytes, yq, nb);
        out_tile[r * tcols + n] = v;
      }
    }

    // The out tensor is ROW_MAJOR f32 with page = row. Blackhole's NOC moves
    // DRAM writes in 16-byte units (NOC_DRAM_WRITE_ALIGNMENT_BYTES —
    // noc_parameters.h:380), so the old per-cell 4-byte writes at row0*4
    // offsets were off-alignment for most cores, and the misdirected or
    // dropped transactions are exactly the leg36 scattered-cell signature.
    // The write unit is therefore a 4-float GROUP: the column split gives
    // every core whole groups (row0 = 4*group0, tile width tcols = 4*groups),
    // each group is one 16-B write at byte offset row0*4 + g*16 inside page
    // m, the pad floats a partial last group carries are the zeros written
    // above, and groups never straddle cores — one writer per 16-B sector,
    // no read-modify-write race. The last group's bytes land inside the out
    // page's alignment padding (align16(N*4) <= page size, host-checked).
    for (uint32_t r = 0; r < mr; ++r) {
      const uint32_t src = op + r * tcols * 4;
      for (uint32_t g = 0; g < tcols / 4; ++g)
        noc_async_write(src + g * 16,
                        acc_o.get_noc_addr(m0 + r, row0 * 4 + g * 16), 16);
    }
    noc_async_write_barrier();

    cb_push_back(CB_W, 1);
    cb_pop_front(CB_W, 1);
    cb_push_back(CB_O, mr);
    cb_pop_front(CB_O, mr);
    cb_pop_front(CB_Q8, mr);
  }
}
)TTKQ";

// The kernels/ dir next to this translation unit — the include path that lets
// the device kernel #include the same header the device-compiled sweep test
// pins (all four registered encodings).
std::filesystem::path KeepQuantKernelIncludeDir() {
  return std::filesystem::path(__FILE__).parent_path() / "kernels";
}

// KEEPQUANT W4b (#3031): the int8-dot arm's persisted MeshWorkload, keyed by
// program identity — the encoding, the activation dtype, M, K, N and the
// grid, i.e. everything the kernel compile args (buffer page sizes) and the
// CB geometry derive from. The in-tree pattern this mirrors is the ttnn
// program cache: a cache miss builds the program, wraps it in a MeshWorkload
// and enqueues it once eagerly; every later call re-sets the runtime args on
// the SAME workload's program (ttnn's override_runtime_arguments hook,
// device_operation.hpp:283) and re-enqueues that workload object
// (device_operation.hpp:291). Reuse is what makes the arm capture-safe:
// EnqueueMeshWorkload runs load_binaries on EVERY enqueue
// (distributed.cpp:118-121), and load_binaries fatals whenever
// program_binary_status_ is empty while a trace is being captured
// (mesh_workload.cpp:148-153) — a workload enqueued once eagerly is
// Committed (fd_mesh_command_queue.cpp:549) and takes the non-fatal branch,
// the exact contract every cached ttnn op in the decode graph already relies
// on. The entry also holds the kernel handle the runtime-arg setters address
// inside the persisted program.
struct Int8DotWorkloadEntry {
  tt::tt_metal::distributed::MeshWorkload workload;
  tt::tt_metal::KernelHandle kernel;
};
std::mutex& Int8DotWorkloadMutex() {
  static std::mutex m;
  return m;
}
std::map<std::string, Int8DotWorkloadEntry>& Int8DotWorkloadCache() {
  // Heap-allocated, deliberately never destroyed (#1486 — every cache
  // accessor in this file): the workload owns device-backed Program state,
  // and a static-storage destructor would unwind after tt-metal's own
  // teardown (tenstorrent_device.cpp:36).
  static std::map<std::string, Int8DotWorkloadEntry>* c =
      new std::map<std::string, Int8DotWorkloadEntry>();
  return *c;
}

}  // namespace

// ---------------------------------------------------------------------------
// TT-DECODE-FUSION: the fused single-program decode arm.
//
// ONE full-grid CreateKernelFromString data-movement kernel (RISCV_0,
// DM_DEDICATED_NOC, O2, compiled against keepquant_kernel_code.h — the same
// include path the int8dot kernel uses) replaces the ~85-program eltwise
// chain DecodeKeepQuantWordsF32's Q6_K case runs per chunk. The kernel
// unpacks exact integer nibbles and executes TWO IEEE f32 multiplies in the
// host's left-to-right order, (d*sc)*q — no FMA contraction is possible (no
// add consumes the first product) and no division exists, so the result is
// bit-identical to the chain by construction. The zero-product sign algebra
// the chain performs (the three-sign XOR polynomial + repair) DELETES: an
// IEEE multiply carries the sign of a zero product through naturally, and
// both operands' signs are the raw bits the chain XORs.
//
// Capture-safety is the int8dot shape verbatim (the precedent at
// kKeepQuantInt8DotKernelSrc): ONE SetCommonRuntimeArgs vector, uniform
// self-cycled CBs, per-core row0/rowc derived in-kernel from the core
// coordinate (the a1661114b doctrine), workload cached and warmed eagerly —
// a capture-time miss refuses by name (mesh_workload.cpp:153).
constexpr const char* kKeepQuantDecodeFusedKernelSrc = R"TTDQ(
#include "api/dataflow/dataflow_api.h"
#include "keepquant_kernel_code.h"

// The ONE common-args vector (see the int8dot kernel's CARG_* table). Every
// word is identical across cores; the only per-core-varying values
// (row0, rowc) are derived from the core coordinate below.
constexpr uint32_t CARG_W_ADDR = 0;   // packed words bank base
constexpr uint32_t CARG_O_ADDR = 1;   // f32 out bank base
constexpr uint32_t CARG_ROWS = 2;     // decode rows (weight rows) in the slice
constexpr uint32_t CARG_NB = 3;       // blocks per row
constexpr uint32_t CARG_WPB = 4;      // staged i32 words per block (64, Q6_K)
constexpr uint32_t CARG_GRID_X = 5;   // core-grid width: c = y*grid_x + x
constexpr uint32_t CARG_RPC = 6;      // uniform rows per core
constexpr uint32_t kNumCommonArgs = 7;

// CB scratch (self-cycled: reserve -> use -> push -> pop; no consumer core).
constexpr uint32_t CB_W = 0;  // one decode row's packed words (wpb*4*nb B)
constexpr uint32_t CB_O = 1;  // one block's 256 f32 outputs (1024 B)

void kernel_main() {
  constexpr auto args_w = TensorAccessorArgs<0, 0>();
  constexpr uint32_t cta_1 = args_w.next_compile_time_args_offset();
  constexpr auto args_o = TensorAccessorArgs<cta_1, 0>();
  const auto acc_w =
      TensorAccessor(args_w, get_common_arg_val<uint32_t>(CARG_W_ADDR));
  const auto acc_o =
      TensorAccessor(args_o, get_common_arg_val<uint32_t>(CARG_O_ADDR));
  const uint32_t rows = get_common_arg_val<uint32_t>(CARG_ROWS);
  const uint32_t nb = get_common_arg_val<uint32_t>(CARG_NB);
  const uint32_t wpb = get_common_arg_val<uint32_t>(CARG_WPB);
  const uint32_t grid_x = get_common_arg_val<uint32_t>(CARG_GRID_X);
  const uint32_t rpc = get_common_arg_val<uint32_t>(CARG_RPC);
  // The per-core row slice, derived from the core coordinate (the int8dot
  // kernel's derivation verbatim): c enumerates the grid row-major.
  const uint32_t c =
      static_cast<uint32_t>(get_relative_logical_y()) * grid_x +
      static_cast<uint32_t>(get_relative_logical_x());
  const uint32_t row0 = c * rpc;
  const uint32_t rowc =
      row0 >= rows ? 0u : ((rpc < rows - row0) ? rpc : (rows - row0));
  if (rowc == 0 || rows == 0) return;  // idle tail: touches no DRAM

  const uint32_t word_bytes = wpb * 4;
  for (uint32_t r = 0; r < rowc; ++r) {
    cb_reserve_back(CB_W, 1);
    const uint32_t wp = get_write_ptr(CB_W);
    // One page per block — the interleaved accessor maps page (row*nb + b)
    // to its bank address (the int8dot kernel's per-block read).
    for (uint32_t b = 0; b < nb; ++b)
      noc_async_read(acc_w.get_noc_addr((row0 + r) * nb + b),
                     wp + b * word_bytes, word_bytes);
    noc_async_read_barrier();
    const uint8_t* x = reinterpret_cast<const uint8_t*>(wp);
    for (uint32_t b = 0; b < nb; ++b, x += word_bytes) {
      // Q6_K block: ql[128] qh[64] sc[16] d(f16) — bytes 0..128, 128..192,
      // 192..208, 208..210 (tenstorrent_keepquant.cpp:630-637 comment).
      const float d = kq_f16_bits_to_f32(kq_load16(x + 208));
      float scf[16];
      for (uint32_t j = 0; j < 16; ++j)
        scf[j] = static_cast<float>(static_cast<int8_t>(x[192 + j]));
      cb_reserve_back(CB_O, 1);
      float* outp = reinterpret_cast<float*>(get_write_ptr(CB_O));
      // The host (h, r) nibble loop verbatim: per (half h, run r, l in
      // 0..32) — ql byte 64h + 32*(r&1) + l, low nibble for r<2 / high
      // otherwise, the 2 high bits (qh[32h + l] >> 2r) & 3 into bit 4,
      // minus the 32 bias, scale s = 8h + 2r + l/16, column i = l%16.
      // TWO IEEE f32 multiplies in the host's left-to-right order:
      // (d*sc) FIRST, then *q. The IEEE sign of a zero product is the XOR
      // of the operand signs — the chain's sign-algebra repair for free.
      for (uint32_t h = 0; h < 2; ++h) {
        for (uint32_t rn = 0; rn < 4; ++rn) {
          const uint32_t qoff = 64 * h + 32 * (rn % 2);
          const uint32_t soff = 8 * h + 2 * rn;
          for (uint32_t l = 0; l < 32; ++l) {
            const uint8_t qb = x[qoff + l];
            const uint32_t nib = (rn < 2) ? (qb & 0xFu) : (qb >> 4);
            const uint8_t hb = x[128 + 32 * h + l];
            const uint32_t nib6 = nib | ((((hb >> (2 * rn)) & 3u) << 4));
            const float q = static_cast<float>(static_cast<int32_t>(nib6) - 32);
            const float t = d * scf[soff + l / 16];
            outp[(soff + l / 16) * 16 + (l % 16)] = t * q;
          }
        }
      }
      noc_async_write(reinterpret_cast<uint32_t>(outp),
                      acc_o.get_noc_addr(row0 + r, b * 1024), 1024);
      cb_push_back(CB_O, 1);
      cb_pop_front(CB_O, 1);
    }
    cb_push_back(CB_W, 1);
    cb_pop_front(CB_W, 1);
  }
  noc_async_write_barrier();
}
)TTDQ";

// The wave-3 Q4_K member of the same family. One common-args vector, the
// same CB geometry — only the block decode differs. The staged Q4_K block
// (KeepQuantWordsPerBlock(kQ4_K) = 48 words = 192 B, the 144-B GGML block
// padded) lays out: d | dmin @ 0..4 (two f16 halves of word 0), scales[12]
// @ 4..16, qs[128] @ 16..144, padding 144..192 ignored (the host chain's
// own slice geometry, tenstorrent_keepquant.cpp:476-492).
constexpr const char* kKeepQuantDecodeFusedKernelQ4KSrc = R"TTDQ(
#include "api/dataflow/dataflow_api.h"
#include "keepquant_kernel_code.h"

constexpr uint32_t CARG_W_ADDR = 0;   // packed words bank base
constexpr uint32_t CARG_O_ADDR = 1;   // f32 out bank base
constexpr uint32_t CARG_ROWS = 2;     // decode rows (weight rows) in the slice
constexpr uint32_t CARG_NB = 3;       // blocks per row
constexpr uint32_t CARG_WPB = 4;      // staged i32 words per block (48, Q4_K)
constexpr uint32_t CARG_GRID_X = 5;   // core-grid width: c = y*grid_x + x
constexpr uint32_t CARG_RPC = 6;      // uniform rows per core
constexpr uint32_t kNumCommonArgs = 7;

constexpr uint32_t CB_W = 0;  // one decode row's packed words (192*nb B)
constexpr uint32_t CB_O = 1;  // one block's 256 f32 outputs (1024 B)

void kernel_main() {
  constexpr auto args_w = TensorAccessorArgs<0, 0>();
  constexpr uint32_t cta_1 = args_w.next_compile_time_args_offset();
  constexpr auto args_o = TensorAccessorArgs<cta_1, 0>();
  const auto acc_w =
      TensorAccessor(args_w, get_common_arg_val<uint32_t>(CARG_W_ADDR));
  const auto acc_o =
      TensorAccessor(args_o, get_common_arg_val<uint32_t>(CARG_O_ADDR));
  const uint32_t rows = get_common_arg_val<uint32_t>(CARG_ROWS);
  const uint32_t nb = get_common_arg_val<uint32_t>(CARG_NB);
  const uint32_t wpb = get_common_arg_val<uint32_t>(CARG_WPB);
  const uint32_t grid_x = get_common_arg_val<uint32_t>(CARG_GRID_X);
  const uint32_t rpc = get_common_arg_val<uint32_t>(CARG_RPC);
  const uint32_t c =
      static_cast<uint32_t>(get_relative_logical_y()) * grid_x +
      static_cast<uint32_t>(get_relative_logical_x());
  const uint32_t row0 = c * rpc;
  const uint32_t rowc =
      row0 >= rows ? 0u : ((rpc < rows - row0) ? rpc : (rows - row0));
  if (rowc == 0 || rows == 0) return;  // idle tail: touches no DRAM

  const uint32_t word_bytes = wpb * 4;
  for (uint32_t r = 0; r < rowc; ++r) {
    cb_reserve_back(CB_W, 1);
    const uint32_t wp = get_write_ptr(CB_W);
    for (uint32_t b = 0; b < nb; ++b)
      noc_async_read(acc_w.get_noc_addr((row0 + r) * nb + b),
                     wp + b * word_bytes, word_bytes);
    noc_async_read_barrier();
    const uint8_t* x = reinterpret_cast<const uint8_t*>(wp);
    for (uint32_t b = 0; b < nb; ++b, x += word_bytes) {
      // Q4_K block: d(f16) dmin(f16) scales[12] qs[128]. The host's
      // GetScaleMinK4 (tenstorrent_keepquant.cpp:493-524): groups is<4 take
      // the low pair scales[is]&63/scales[is+4]&63; groups is>=4 take
      // sc = scales[is+4]&0xF | scales[is-4] top-2-bits and
      // m = scales[is+4]>>4 | scales[is] top-2-bits.
      const float d = kq_f16_bits_to_f32(kq_load16(x));
      const float dmin = kq_f16_bits_to_f32(kq_load16(x + 2));
      float scf[8];
      float mmf[8];
      for (uint32_t g = 0; g < 4; ++g) {
        scf[g] = static_cast<float>(x[4 + g] & 63);
        mmf[g] = static_cast<float>(x[4 + g + 4] & 63);
        scf[g + 4] = static_cast<float>((x[4 + g + 8] & 0xF) |
                                        ((x[4 + g] >> 6) << 4));
        mmf[g + 4] = static_cast<float>((x[4 + g + 8] >> 4) |
                                        ((x[4 + g + 4] >> 6) << 4));
      }
      cb_reserve_back(CB_O, 1);
      float* outp = reinterpret_cast<float*>(get_write_ptr(CB_O));
      // The host nibble loop verbatim (the chain's (d*sc)*x - dmin*mm order,
      // tenstorrent_keepquant.cpp:587-629): per quarter c (64 values) the
      // 32 qs bytes feed group 2c with their LOW nibble and group 2c+1 with
      // their HIGH nibble, output column l within each 32-run. TWO IEEE f32
      // multiplies then ONE subtract, each operand a rounded value: fl(d*sc)
      // keeps d's sign through a zero product (sc unsigned), fl(dmin*mm)
      // keeps dmin's, and the IEEE subtract preserves an exact-zero sign —
      // the chain's signed-zero repair algebra, exact on raw RISC-V floats.
      // The volatile staging defeats fma contraction (GCC -ffp-contract
      // would fuse t*q - m and change every rounding).
      const uint8_t* q = x + 16;
      for (uint32_t cq = 0; cq < 4; ++cq, q += 32) {
        for (uint32_t l = 0; l < 32; ++l) {
          const uint8_t qb = q[l];
          volatile float p0 = d * scf[2 * cq] * static_cast<float>(qb & 0xFu);
          volatile float p1 = d * scf[2 * cq + 1] * static_cast<float>(qb >> 4);
          volatile float m0 = dmin * mmf[2 * cq];
          volatile float m1 = dmin * mmf[2 * cq + 1];
          // p0 is fl(fl(d*sc)*q): the LEFT-TO-RIGHT grouping above is the
          // host's order, and the volatile stores pin both roundings — no
          // fma contraction (GCC would fuse p*m - m and change every
          // rounding and zero sign).
          outp[(2 * cq) * 32 + l] = p0 - m0;
          outp[(2 * cq + 1) * 32 + l] = p1 - m1;
        }
      }
      noc_async_write(reinterpret_cast<uint32_t>(outp),
                      acc_o.get_noc_addr(row0 + r, b * 1024), 1024);
      cb_push_back(CB_O, 1);
      cb_pop_front(CB_O, 1);
    }
    cb_push_back(CB_W, 1);
    cb_pop_front(CB_W, 1);
  }
  noc_async_write_barrier();
}
)TTDQ";

struct FusedDecodeWorkloadEntry {
  tt::tt_metal::distributed::MeshWorkload workload;
  tt::tt_metal::KernelHandle kernel;
};
std::mutex& FusedDecodeWorkloadMutex() {
  static std::mutex m;
  return m;
}
std::map<std::string, FusedDecodeWorkloadEntry>& FusedDecodeWorkloadCache() {
  // Heap-allocated, never destroyed (#1486 — the int8dot cache's rationale).
  static std::map<std::string, FusedDecodeWorkloadEntry>* c =
      new std::map<std::string, FusedDecodeWorkloadEntry>();
  return *c;
}

// The arm gate: fused is the DEFAULT on Q6_K; VT_TT_KEEPQUANT_FUSED=0 is the
// named fallback to the proven chain (the spec's env-gated decline arm).
bool KeepQuantFusedDecodeEnabled() {
  const char* e = std::getenv("VT_TT_KEEPQUANT_FUSED");
  return e == nullptr || e[0] == '\0' || e[0] != '0';
}

// The fused dispatcher for the whole-decode family — Q6_K (wave 2) and Q4_K
// (wave 3). Returns nullopt when the shape is not served — the caller falls
// through to the chain, and the decline is NAMED (a stderr line naming the
// missing part) whenever the arm was requested, never silent.
std::optional<ttnn::Tensor> DecodeKeepQuantWordsFusedQK(
    const ttnn::Tensor& w, DType enc, int64_t slice_rows, int64_t nb,
    MeshDevice& device) {
  VT_CHECK(enc == DType::kQ4_K || enc == DType::kQ6_K,
           "tenstorrent keep-quant fused decode: encoding not in the served "
           "set (kQ4_K/kQ6_K)");
  if (slice_rows <= 0 || nb <= 0) return std::nullopt;
  if (!KeepQuantFusedDecodeEnabled()) return std::nullopt;
  const uint32_t rows = static_cast<uint32_t>(slice_rows);
  const uint32_t wpb = static_cast<uint32_t>(KeepQuantWordsPerBlock(enc));
  const uint32_t word_bytes = wpb * 4;
  const uint32_t row_bytes = word_bytes * static_cast<uint32_t>(nb);
  const auto grid = device.compute_with_storage_grid_size();
  const uint32_t grid_cores =
      static_cast<uint32_t>(grid.x) * static_cast<uint32_t>(grid.y);
  const uint32_t rpc = (rows + grid_cores - 1) / grid_cores;
  const auto align16 = [](uint32_t v) { return (v + 15u) & ~15u; };
  // L1 budget: the words-row CB page + the 1 KB out CB page, with the
  // int8dot kernel's 768 KiB per-core envelope. Over budget -> decline BY
  // NAME; the chain serves the shape.
  const uint32_t cb_w_page = align16(row_bytes);
  const uint64_t l1_bytes = static_cast<uint64_t>(cb_w_page) + 1024u;
  if (l1_bytes > (700u << 10)) {
    std::fprintf(stderr,
                 "[TT-KQ-FUSED] decline: rows=%lld nb=%lld needs %llu B of "
                 "per-core L1 over the 700 KiB budget — chain serves\n",
                 (long long)slice_rows, (long long)nb,
                 (unsigned long long)l1_bytes);
    return std::nullopt;
  }
  if (rpc == 0) return std::nullopt;

  // The out tensor: ROW_MAJOR f32 [slice_rows, nb*256] — the chain's own
  // return geometry (the caller's typecast/to_layout consume it unchanged).
  ttnn::Tensor dev_out = ttnn::empty(
      ttnn::Shape({rows, static_cast<uint32_t>(nb) * 256u}),
      ttnn::DataType::FLOAT32, ttnn::Layout::ROW_MAJOR, &device,
      ttnn::MemoryConfig{});

  // Program identity: rows, nb and the grid — everything the CB geometry and
  // the compile args derive from. Addresses are runtime args (common args,
  // re-set per call below).
  const std::string workload_key =
      (enc == DType::kQ6_K ? std::string("fusedq6k/") : std::string("fusedq4k/")) +
      std::to_string(rows) + "x" + std::to_string(nb) + "x" +
      std::to_string(grid.x) + "x" + std::to_string(grid.y);

  std::lock_guard<std::mutex> workload_guard(FusedDecodeWorkloadMutex());
  const auto common_args = [&] {
    return std::vector<uint32_t>{
        static_cast<uint32_t>(w.mesh_buffer().address()),
        static_cast<uint32_t>(dev_out.mesh_buffer().address()),
        rows, static_cast<uint32_t>(nb), wpb,
        static_cast<uint32_t>(grid.x), rpc};
  };
  auto& workload_cache = FusedDecodeWorkloadCache();
  auto workload_it = workload_cache.find(workload_key);
  const bool workload_miss = workload_it == workload_cache.end();
  if (workload_miss) {
    // Warm-first: a workload never enqueued eagerly cannot enter a trace
    // (load_binaries fatals mid-capture, mesh_workload.cpp:148-153).
    VT_CHECK(!tt_capture_active(),
             "tenstorrent keep-quant fused decode: shape not warmed before "
             "trace capture — run the shape eagerly once first (tt-metal "
             "refuses new binaries mid-capture, mesh_workload.cpp:153)");
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    {
      tt::tt_metal::CircularBufferConfig cfg(
          cb_w_page, {{tt::CBIndex::c_0, tt::DataFormat::Float32}});
      cfg.set_page_size(tt::CBIndex::c_0, cb_w_page);
      tt::tt_metal::CreateCircularBuffer(
          program,
          tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord{0, 0},
                                  tt::tt_metal::CoreCoord{grid.x - 1, grid.y - 1}),
          cfg);
    }
    {
      tt::tt_metal::CircularBufferConfig cfg(
          1024u, {{tt::CBIndex::c_1, tt::DataFormat::Float32}});
      cfg.set_page_size(tt::CBIndex::c_1, 1024u);
      tt::tt_metal::CreateCircularBuffer(
          program,
          tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord{0, 0},
                                  tt::tt_metal::CoreCoord{grid.x - 1, grid.y - 1}),
          cfg);
    }
    std::vector<uint32_t> compile_args;
    tt::tt_metal::TensorAccessorArgs(w.mesh_buffer()).append_to(compile_args);
    tt::tt_metal::TensorAccessorArgs(dev_out.mesh_buffer()).append_to(compile_args);
    tt::tt_metal::KernelHandle kernel = tt::tt_metal::CreateKernelFromString(
        program,
        enc == DType::kQ6_K ? kKeepQuantDecodeFusedKernelSrc
                            : kKeepQuantDecodeFusedKernelQ4KSrc,
        tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord{0, 0},
                                tt::tt_metal::CoreCoord{grid.x - 1, grid.y - 1}),
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
            .compile_args = compile_args,
            .defines = {},
            .named_compile_args = {},
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O2,
            .compiler_include_paths = {KeepQuantKernelIncludeDir()}});
    // The ONE legal initial common-args set (kernel.cpp:786 — later calls
    // update in place, the int8dot dispatch's pattern).
    tt::tt_metal::SetCommonRuntimeArgs(program, kernel, common_args());
    tt::tt_metal::distributed::MeshWorkload workload;
    workload.add_program(
        tt::tt_metal::distributed::MeshCoordinateRange(device.shape()),
        std::move(program));
    workload_it =
        workload_cache
            .emplace(workload_key,
                     FusedDecodeWorkloadEntry{std::move(workload), kernel})
            .first;
  }
  tt::tt_metal::Program& program =
      workload_it->second.workload.get_programs().begin()->second;
  if (!workload_miss) {
    auto& crta =
        tt::tt_metal::GetCommonRuntimeArgs(program, workload_it->second.kernel);
    const std::vector<uint32_t> next = common_args();
    for (uint32_t i = 0; i < next.size(); ++i) crta[i] = next[i];
  }
  program.set_runtime_id(static_cast<uint64_t>(
      ttnn::CoreIDs::instance().fetch_and_increment_device_operation_id()));
  // The eager drain (the int8dot dispatch's ordering contract); skipped
  // under capture — finish() records a mesh event ttnn forbids mid-capture.
  if (!tt_capture_active()) {
    device.mesh_command_queue().finish();
  }
  tt::tt_metal::distributed::EnqueueMeshWorkload(device.mesh_command_queue(),
                                                 workload_it->second.workload,
                                                 /*blocking=*/false);
  return dev_out;
}

// kMatmulBTQuant's W4b body: out[M,N] = a[M,K] @ b[N,K]^T with b PACKED
// keep-quant blocks, computed entirely by the device kernel above. M, N, K,
// nb, wpb and the per-core row split are launch arguments; the activation is
// quantized on-core per m-tile into core-private L1 (redundant across the
// grid, never shared, so there is no inter-core sync and no DRAM q8 scratch).
void MatmulBTQuantInt8DotKernel(Queue& q, Tensor& out, const Tensor& a,
                                const Tensor& b) {
  MeshDevice& device = SharedMeshDevice();
  (void)q;  // the workload enqueues on the device's mesh command queue
  const DType enc = b.dtype;
  const int64_t elems = BlockElems(enc);
  const int64_t M = a.shape[0];
  const int64_t K = a.shape[1];
  const int64_t N = b.shape[0];
  const int64_t nb = K / elems;
  if (M == 0 || N == 0) return;

  // PACKED words: the resident i32 shadow ([N*nb, wpb]), staged once per
  // weight and refused on a capture-time miss (the warm-first contract).
  const ttnn::Tensor words = EnsureKeepQuantWords(b, enc, N, nb, device);

  // Activation: ROW_MAJOR flat bytes in the declared dtype (f32 stays f32 —
  // the quantizer consumes the master's own values; bf16 stays bf16, widened
  // exactly on-core). A slot hit returns the producer's layout; a TILE view
  // converts through a recorded to_layout ahead of the launch in the trace.
  // Activation: ROW_MAJOR flat bytes in the declared dtype. A bf16 master
  // rides the proven EnsureDevice2D staging (persistent-buffer in-place
  // writes under capture) and a TILE view converts through a recorded
  // to_layout ahead of the launch. An F32 MASTER STAGES AS F32: the old
  // route (EnsureDevice2D's f32 reference arm) stages every non-bf16 master
  // as a BFLOAT16 TILE, while the kernel's act_f32 flag then read the bf16
  // bytes as f32 words — the leg41 act_f32 sweep failing 10/10 including
  // (0,0). from_span FLOAT32 ROW_MAJOR is the same direct staging the words
  // shadow uses; capture-time arrival refuses by name (the words-shadow
  // discipline — no captured consumer exists for an f32 activation today).
  // W4 (capture-warmup redesign): the activation serves device-resident
  // in-region FIRST — the same treatment the grouped kernel got in W3. The
  // engine's producers commit their outputs device-side in every phase, and
  // the 27B decode reaches this kernel inside the captured region (the
  // APEX anchor leg fatalled on the refusal below before this serve
  // existed). The serve lands the MASTER'S OWN dtype ("an f32 master
  // stages as f32") through the shadow's — an f32 shadow over a bf16
  // buffer typecasts with the same RNE the host round-trip applied. The
  // program reads dev_a's
  // raw bytes, so the serve normalizes to ROW_MAJOR at [M, K]: a TILE shadow
  // converts through the recorded to_layout ahead of the launch (the comment
  // above already named that conversion as the design), and the volume-equal
  // native geometry (the GDN head-row commits) reinterprets through
  // CaptureSafeReshape — same numel, and both passes run the identical chain
  // so the programs the serve records are warmed by the eager step.
  ttnn::Tensor dev_a;
  bool act_served = false;
  {
    ttnn::Tensor raw;
    const uint32_t um = static_cast<uint32_t>(M);
    const uint32_t uk = static_cast<uint32_t>(K);
    if (ServeDeviceShadowRaw(a, um, uk, raw) ||
        ServeDeviceWindow(a, um, uk, raw)) {
      // The master's own dtype, reached through the shadow's: the norm
      // chain commits an F32 shadow over a bf16-declared buffer (rms_norm
      // computes f32), and the host staging this replaces rounded that
      // shadow to the master's dtype through the host round-trip — the
      // device typecast is the same RNE (the SigmoidGateBf16 doctrine),
      // so the serve is value-identical either way.
      const ttnn::DataType want =
          a.dtype == DType::kF32 ? ttnn::DataType::FLOAT32
                                : ttnn::DataType::BFLOAT16;
      const auto ls = raw.logical_shape();
      if (ls.rank() != 2 || ls[0] != um || ls[1] != uk)
        raw = CaptureSafeReshape(std::move(raw), ttnn::Shape({um, uk}));
      if (raw.dtype() != want)
        raw = ttnn::typecast(std::move(raw), want);
      if (raw.layout() != ttnn::Layout::ROW_MAJOR)
        raw = ttnn::to_layout(std::move(raw), ttnn::Layout::ROW_MAJOR);
      dev_a = std::move(raw);
      act_served = true;
    }
  }
  if (!act_served) {
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr) {
      std::lock_guard<std::mutex> g(SlotMutex());
      BufferSlot* s = FindSlot(a.data);
      std::fprintf(stderr,
                   "[TT-I8DOT] act-serve miss ptr=%p M=%lld K=%lld dtype=%d "
                   "slot=%d devcur=%d hostcur=%d hasdev=%d devr=%u devc=%u "
                   "devdt=%d cap=%d\n",
                   a.data, (long long)M, (long long)K, (int)a.dtype,
                   s != nullptr, s ? (int)s->device_current : -1,
                   s ? (int)s->host_current : -1, s && s->device.has_value(),
                   s ? s->dev_rows : 0u, s ? s->dev_cols : 0u,
                   s && s->device.has_value()
                       ? static_cast<int>(s->device->dtype())
                       : -1,
                   (int)tt_capture_active());
    }
  if (a.dtype == DType::kF32) {
    VT_CHECK(!tt_capture_active(),
             "tenstorrent kMatmulBTQuant int8-dot: f32 activation staging "
             "during trace capture — stage the f32 master eagerly first");
    EnsureHost(a);
    dev_a = ttnn::Tensor::from_span(
        ttsl::Span<const float>(a.Ptr<float>(),
                                static_cast<size_t>(M) * static_cast<size_t>(K)),
        SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(M),
                                    static_cast<uint32_t>(K)}),
               ttnn::DataType::FLOAT32, ttnn::Layout::ROW_MAJOR),
        &device);
  } else {
    VT_CHECK(!tt_capture_active(),
             "tenstorrent kMatmulBTQuant int8-dot: bf16 activation staging "
             "during trace capture — stage the bf16 master eagerly first");
    EnsureHost(a);
    dev_a = ttnn::Tensor::from_span(
        ttsl::Span<const bfloat16>(
            a.Ptr<bfloat16>(),
            static_cast<size_t>(M) * static_cast<size_t>(K)),
        SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(M),
                                     static_cast<uint32_t>(K)}),
               ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR),
        &device);
  }
  }

  // The out pages must be 16-B multiples: Blackhole moves DRAM writes in
  // 16-B units (NOC_DRAM_WRITE_ALIGNMENT_BYTES — noc_parameters.h:380), and
  // ttnn sizes a ROW_MAJOR page at N*4 exactly (a {3,3} f32 page is 12 B), so
  // a narrow-N out cannot legally receive the 16-B group writes at all — the
  // leg36 scattered-cell signature. The width is therefore padded to
  // align4(N) here and sliced back to N before the commit; N % 4 == 0 (every
  // model shape) takes no slice.
  const uint32_t n4 = (static_cast<uint32_t>(N) + 3u) & ~3u;
  ttnn::Tensor dev_out =
      ttnn::empty(ttnn::Shape({static_cast<uint32_t>(M), n4}),
                  ttnn::DataType::FLOAT32, ttnn::Layout::ROW_MAJOR, &device,
                  ttnn::MemoryConfig{});

  // Grid: one core per 4-column GROUP slice. The write unit the out page
  // gets is one 16-B group (see the kernel's write loop), so the split is in
  // groups — row0 = 4*group0, a core's real column count rowc <= tcols, and
  // tcols is uniform so the staging CB geometry is per-program constant.
  const auto grid = device.compute_with_storage_grid_size();
  const uint32_t grid_cores =
      static_cast<uint32_t>(grid.x) * static_cast<uint32_t>(grid.y);
  const uint32_t groups_total =
      (static_cast<uint32_t>(N) + 3u) / 4u;
  const uint32_t groups_per_core =
      (groups_total + grid_cores - 1) / grid_cores;
  const uint32_t tcols = groups_per_core * 4u;  // padded tile width, uniform
  const uint32_t qb_pad =
      (enc == DType::kQ8_0
           ? (static_cast<uint32_t>((K / 32) * 34) + 15u) & ~15u
           : (static_cast<uint32_t>(nb * 292) + 15u) & ~15u);
  // m-tile: keep the per-core q8 residency inside the L1 budget.
  const uint32_t mtile =
      std::max(1u, std::min(8u, (160u << 10) / std::max(qb_pad, 1u)));
  const uint32_t weight_row_bytes = static_cast<uint32_t>(
      KeepQuantWordsPerBlock(enc) * 4 * nb);
  const uint64_t l1_bytes = static_cast<uint64_t>(K) * 6 +
                            static_cast<uint64_t>(mtile) * qb_pad +
                            weight_row_bytes +
                            static_cast<uint64_t>(mtile) * tcols * 4;
  VT_CHECK(l1_bytes <= (768u << 10),
           "tenstorrent kMatmulBTQuant int8-dot: shape needs " +
               std::to_string(l1_bytes) +
               " B of per-core L1, over the 768 KiB budget — refuse by name "
               "rather than corrupt a neighboring buffer");
  {
    // Tripwire, not a fix: dev_out's width is padded to align4(N) above, so
    // the page the group writes target is a 16-B multiple by construction.
    // This fires if ttnn changes page sizing under us.
    const uint32_t out_page =
        dev_out.mesh_buffer().page_size();
    VT_CHECK(out_page % 16u == 0u && out_page >= (tcols * 4u),
             "tenstorrent kMatmulBTQuant int8-dot: out page " +
                 std::to_string(out_page) +
                 " B is not a 16-B multiple wide enough for a group tile row (" +
                 std::to_string(tcols * 4u) +
                 " B) — the group write would be misaligned or leave the page");
  }

  // Program identity: the encoding, activation dtype, M, K, N and grid —
  // everything the compile args (buffer page sizes) and CB geometry derive
  // from; the shape-spec identity the ttnn program cache hashes. Per-call
  // variation (buffer addresses, the per-core shape/slice words) is runtime
  // args, re-set on EVERY call below.
  const uint32_t enc_sel = enc == DType::kQ4_K    ? 0
                           : enc == DType::kQ5_K  ? 1
                           : enc == DType::kQ6_K  ? 2
                           : enc == DType::kQ8_0  ? 3
                           : enc == DType::kIQ3_XXS ? 4
                           : enc == DType::kIQ2_XXS ? 5
                           : enc == DType::kIQ2_S   ? 6
                           : enc == DType::kQ3_K    ? 7
                           : enc == DType::kIQ3_S   ? 8
                           : enc == DType::kIQ4_XS  ? 9
                           : enc == DType::kIQ2_XS  ? 10
                           : enc == DType::kQ2_K   ? 11
                           : enc == DType::kIQ1_S  ? 12
                           : enc == DType::kIQ1_M  ? 13
                                                  : 0;
  const uint32_t wpb = static_cast<uint32_t>(KeepQuantWordsPerBlock(enc));
  const uint32_t act_f32 = a.dtype == DType::kF32 ? 1u : 0u;
  const std::string workload_key =
      std::to_string(static_cast<int>(enc)) + "x" + std::to_string(act_f32) +
      "x" + std::to_string(M) + "x" + std::to_string(K) + "x" +
      std::to_string(N) + "x" + std::to_string(grid.x) + "x" +
      std::to_string(grid.y);

  std::lock_guard<std::mutex> workload_guard(Int8DotWorkloadMutex());
  // The ONE common-args vector the program runs (see the kernel's CARG_*
  // table): the three bank bases followed by the shape-global words. Built per
  // call, set once on a miss and updated in place on a hit.
  const auto common_args = [&] {
    return std::vector<uint32_t>{
        static_cast<uint32_t>(words.mesh_buffer().address()),
        static_cast<uint32_t>(dev_a.mesh_buffer().address()),
        static_cast<uint32_t>(dev_out.mesh_buffer().address()),
        static_cast<uint32_t>(M), static_cast<uint32_t>(K),
        static_cast<uint32_t>(N), static_cast<uint32_t>(nb), wpb, act_f32,
        mtile, qb_pad, enc_sel, tcols, static_cast<uint32_t>(grid.x)};
  };
  auto& workload_cache = Int8DotWorkloadCache();
  auto workload_it = workload_cache.find(workload_key);
  const bool workload_miss = workload_it == workload_cache.end();
  if (workload_miss) {
    // A workload never enqueued eagerly cannot enter a trace: load_binaries
    // fatals on an empty program_binary_status_ mid-capture
    // (mesh_workload.cpp:148-153). Refuse by name — the warm-first contract
    // the words shadow and the f32 staging already bind.
    VT_CHECK(!tt_capture_active(),
             "tenstorrent kMatmulBTQuant int8-dot: shape not warmed before "
             "trace capture — run the shape eagerly once first (tt-metal "
             "refuses new binaries mid-capture, mesh_workload.cpp:153)");
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const auto align16 = [](uint32_t x) { return (x + 15u) & ~15u; };
    // Widened f32 row (K*4 B) plus, for the bf16 arm, the raw u16 tail (K*2 B)
    // the kernel reads before widening in place.
    const uint32_t cb_f32_page = align16(static_cast<uint32_t>(K) * 4 +
                                         static_cast<uint32_t>(K) * 2);
    {
      tt::tt_metal::CircularBufferConfig cfg(
          cb_f32_page,
          {{tt::CBIndex::c_0, tt::DataFormat::Float32}});
      cfg.set_page_size(tt::CBIndex::c_0, cb_f32_page);
      tt::tt_metal::CreateCircularBuffer(
          program,
          tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord{0, 0},
                                  tt::tt_metal::CoreCoord{grid.x - 1, grid.y - 1}),
          cfg);
    }
    {
      tt::tt_metal::CircularBufferConfig cfg(
          mtile * qb_pad, {{tt::CBIndex::c_1, tt::DataFormat::Float32}});
      cfg.set_page_size(tt::CBIndex::c_1, qb_pad);
      tt::tt_metal::CreateCircularBuffer(
          program,
          tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord{0, 0},
                                  tt::tt_metal::CoreCoord{grid.x - 1, grid.y - 1}),
          cfg);
    }
    {
      const uint32_t page = align16(weight_row_bytes);
      tt::tt_metal::CircularBufferConfig cfg(
          page, {{tt::CBIndex::c_2, tt::DataFormat::Float32}});
      cfg.set_page_size(tt::CBIndex::c_2, page);
      tt::tt_metal::CreateCircularBuffer(
          program,
          tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord{0, 0},
                                  tt::tt_metal::CoreCoord{grid.x - 1, grid.y - 1}),
          cfg);
    }
    {
      const uint32_t page = tcols * 4u;  // already a 16-B multiple
      tt::tt_metal::CircularBufferConfig cfg(
          mtile * page, {{tt::CBIndex::c_3, tt::DataFormat::Float32}});
      cfg.set_page_size(tt::CBIndex::c_3, page);
      tt::tt_metal::CreateCircularBuffer(
          program,
          tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord{0, 0},
                                  tt::tt_metal::CoreCoord{grid.x - 1, grid.y - 1}),
          cfg);
    }

    std::vector<uint32_t> compile_args;
    tt::tt_metal::TensorAccessorArgs(words.mesh_buffer()).append_to(compile_args);
    tt::tt_metal::TensorAccessorArgs(dev_a.mesh_buffer()).append_to(compile_args);
    tt::tt_metal::TensorAccessorArgs(dev_out.mesh_buffer()).append_to(compile_args);

    tt::tt_metal::KernelHandle kernel = tt::tt_metal::CreateKernelFromString(
        program, kKeepQuantInt8DotKernelSrc,
        tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord{0, 0},
                                tt::tt_metal::CoreCoord{grid.x - 1, grid.y - 1}),
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
            .compile_args = compile_args,
            .defines = {},
            .named_compile_args = {},
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O2,
            .compiler_include_paths = {KeepQuantKernelIncludeDir()}});
    // The ONE legal initial common-args set (kernel.cpp:786: common runtime
    // args can only be set once; later calls update them in place). ALL the
    // runtime words live here since the 27B trace-fit fix: per-core
    // SetRuntimeArgs recorded one Unique RTA UNICAST page per core (~2.97 MB
    // per captured launch), and the two words that varied per core (row0,
    // rowc) are derived in-kernel from the core coordinate.
    tt::tt_metal::SetCommonRuntimeArgs(program, kernel, common_args());

    tt::tt_metal::distributed::MeshWorkload workload;
    workload.add_program(
        tt::tt_metal::distributed::MeshCoordinateRange(device.shape()),
        std::move(program));
    workload_it =
        workload_cache
            .emplace(workload_key,
                     Int8DotWorkloadEntry{std::move(workload), kernel})
            .first;
  }

  // The program lives INSIDE the persisted workload from here on — ttnn
  // reaches its cached program the same way (workload.get_programs(),
  // device_operation.hpp:184). Per-call runtime args on it (the ttnn
  // override_runtime_arguments contract): the common args carry this call's
  // buffer addresses plus the shape/encoding words. Dispatch commands
  // regenerate from these on every enqueue (mesh_workload.cpp:210), so a
  // re-enqueued workload always runs this call's values. NO per-core
  // SetRuntimeArgs remains on this program: the 27B trace-fit fix derives
  // row0/rowc in-kernel from the core coordinate (grid is part of the
  // workload key, so grid_x is shape-global per program), and every other
  // word is identical across cores. This call is the only call site of the
  // program — nothing else needs per-core args on it.
  tt::tt_metal::Program& program =
      workload_it->second.workload.get_programs().begin()->second;
  if (!workload_miss) {
    // Update the common args IN PLACE on a reused program (kernel.cpp:786
    // forbids a second set) — the GetCommonRuntimeArgs pattern (ttnn
    // unary_program_factory.cpp:647-652). A miss just set them with this
    // call's values.
    auto& crta =
        tt::tt_metal::GetCommonRuntimeArgs(program, workload_it->second.kernel);
    const std::vector<uint32_t> next = common_args();
    for (uint32_t i = 0; i < next.size(); ++i) crta[i] = next[i];
  }
  // A fresh runtime id before EVERY enqueue, hit or miss — what the ttnn
  // dispatch does unconditionally (device_operation.hpp:181-186).
  program.set_runtime_id(static_cast<uint64_t>(
      ttnn::CoreIDs::instance().fetch_and_increment_device_operation_id()));

  // Eager mode keeps the drain: the staging writes (from_vector words,
  // to_layout activation) must be visible to the raw workload below, and
  // that ordering contract is otherwise unproven. During capture the drain
  // is skipped — finish() records a mesh event, which ttnn forbids
  // mid-capture; the recorded ops carry the ordering (capture/replay tests
  // and the vehicle's token-exact gates hold without it).
  if (!tt_capture_active()) {
    device.mesh_command_queue().finish();
  }
  tt::tt_metal::distributed::EnqueueMeshWorkload(device.mesh_command_queue(),
                                                 workload_it->second.workload,
                                                 /*blocking=*/false);

  if (n4 != static_cast<uint32_t>(N)) {
    // Drop the pad columns the pages carry — the commit volume must match
    // the caller's [M, N] exactly.
    dev_out = ttnn::slice(
        dev_out, ttsl::SmallVector<uint32_t>{0u, 0u},
        ttsl::SmallVector<uint32_t>{static_cast<uint32_t>(M),
                                    static_cast<uint32_t>(N)},
        ttsl::SmallVector<uint32_t>{1u, 1u});
  }
  // Commit form: TILE (the twin path's layout — a ROW_MAJOR commit leaves a
  // slot the next consumer's tile view can overflow, the wave-3b lesson).
  ttnn::Tensor committed =
      ttnn::to_layout(std::move(dev_out), ttnn::Layout::TILE);
  // W4d W6: a BF16-OUT call takes an explicit f32->bf16 cast BEFORE the
  // commit. Committing the f32 dev_out into a bf16 slot left the slot
  // holding f32 bytes at an f32 page geometry — the next bf16 reader got
  // word-halved garbage (the ROW_MAJOR chained leg's NaN signature, the
  // W4b landing decision). One recorded typecast launch: capture-safe.
  if (out.dtype == DType::kBF16)
    committed = ttnn::typecast(std::move(committed),
                               ttnn::DataType::BFLOAT16);

  CommitDeviceLogical2D(out, std::move(committed), static_cast<uint32_t>(M),
                        static_cast<uint32_t>(N));
}

int64_t KeepQuantCaptureStagingWrites() {
  return KeepQuantCaptureStagingWritesCounter().load(std::memory_order_relaxed);
}
void ResetKeepQuantCaptureStagingWritesForTest() {
  KeepQuantCaptureStagingWritesCounter().store(0, std::memory_order_relaxed);
}

// W4a wave-3a test hooks — the contract lives in tenstorrent_device.h.
void KeepQuantChunkRowsOverrideForTest(int64_t rows) {
  KeepQuantChunkRowsOverride().store(rows, std::memory_order_relaxed);
}

bool KeepQuantWordShadowPresentForTest(const void* host) {
  if (host == nullptr) return false;
  std::lock_guard<std::mutex> g(KeepQuantWordMutex());
  return KeepQuantWordShadows().find(host) != KeepQuantWordShadows().end();
}
bool DecodedWeightShadowPresentForTest(const void* host) {
  if (host == nullptr) return false;
  std::lock_guard<std::mutex> g(DecodedWeightMutex());
  return DecodedWeightShadows().find(reinterpret_cast<uintptr_t>(host)) !=
         DecodedWeightShadows().end();
}

}  // namespace vt::tenstorrent
