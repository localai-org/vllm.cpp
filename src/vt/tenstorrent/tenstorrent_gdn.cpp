// Tenstorrent GDN op kernels (BACKEND-TENSTORRENT-SPLIT stage 3).
// Definitions moved verbatim from tenstorrent_ops.cpp; the shared counters,
// idx/conv-tile caches and cross-TU declarations live in
// tenstorrent_internal.h.
#include "vt/tenstorrent/tenstorrent_internal.h"

namespace vt::tenstorrent {

// kGdnPostConv: fused GDN post-conv prep = GdnConvSplit + per-head L2Norm(q/k)
// + GdnGBeta in one pass (cpu_ops.cpp GdnPostConvKernel:3472-3516; wrapper
// ops.cpp:4255-4287; the fla fused_recurrent_gated_delta_rule prefill preamble).
// q/k: conv slices [0,key_dim)/[key_dim,2*key_dim) reshape to [T*Hk, Dk] rows
// and run the kL2Norm device math (square -> sum -> +eps -> rsqrt -> mul) in
// bf16 tiles — the row's calibrated l2norm envelope. v: plain slice copy.
// g/beta: FLOAT32 tiles end to end (softplus threshold-20, exp(a_log) and
// sigmoid are f32 in the oracle); araw/braw are row-strided views, gathered
// on host exactly like the kRmsNormGated gate view.
void GdnPostConvKernel(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out,
                       Tensor& g_out, Tensor& beta_out, const Tensor& conv,
                       const Tensor& araw, const Tensor& braw,
                       const Tensor& a_log, const Tensor& dt_bias,
                       const GdnPostConvArgs& args) {
  TT_OP_TRACE("GdnPostConv");
  VT_CHECK((q_out.dtype == DType::kF32 || q_out.dtype == DType::kBF16) &&
               k_out.dtype == q_out.dtype && v_out.dtype == q_out.dtype,
           "tenstorrent kGdnPostConv: q_out/k_out/v_out must be f32 or bf16, same dtype");
  VT_CHECK(conv.dtype == DType::kF32 || conv.dtype == DType::kBF16,
           "tenstorrent kGdnPostConv: conv must be f32 or bf16");
  VT_CHECK(g_out.dtype == DType::kF32 && beta_out.dtype == DType::kF32 &&
               (araw.dtype == DType::kF32 || araw.dtype == DType::kBF16) &&
               braw.dtype == araw.dtype && a_log.dtype == DType::kF32 &&
               dt_bias.dtype == DType::kF32,
           "tenstorrent kGdnPostConv: g/beta/a_log/dt_bias f32; a/b share f32 or bf16");
  VT_CHECK(q_out.IsContiguous() && k_out.IsContiguous() && v_out.IsContiguous() &&
               g_out.IsContiguous() && beta_out.IsContiguous() &&
               conv.IsContiguous() && araw.stride[1] == 1 && braw.stride[1] == 1 &&
               a_log.IsContiguous() && dt_bias.IsContiguous(),
           "tenstorrent kGdnPostConv: contiguous required (a/b row views excepted)");
  const uint32_t t = static_cast<uint32_t>(conv.shape[0]);
  const uint32_t hk = static_cast<uint32_t>(q_out.shape[1]);
  const uint32_t dk = static_cast<uint32_t>(q_out.shape[2]);
  const uint32_t hv = static_cast<uint32_t>(v_out.shape[1]);
  const uint32_t dv = static_cast<uint32_t>(v_out.shape[2]);
  const uint32_t key_dim = hk * dk, value_dim = hv * dv;
  const uint32_t conv_dim = 2 * key_dim + value_dim;
  VT_CHECK(conv.shape[0] == q_out.shape[0] && conv.shape[1] == conv_dim &&
               g_out.shape[0] == t && g_out.shape[1] == hv,
           "tenstorrent kGdnPostConv: shape mismatch");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_conv = DeviceRows(conv, t, conv_dim, device);
  ttnn::Tensor q2 = ttnn::slice(dev_conv, ttsl::SmallVector<uint32_t>{0, 0},
                                ttsl::SmallVector<uint32_t>{t, key_dim},
                                ttsl::SmallVector<uint32_t>{1, 1});
  ttnn::Tensor k2 = ttnn::slice(dev_conv, ttsl::SmallVector<uint32_t>{0, key_dim},
                                ttsl::SmallVector<uint32_t>{t, 2 * key_dim},
                                ttsl::SmallVector<uint32_t>{1, 1});
  ttnn::Tensor v2 = ttnn::slice(dev_conv, ttsl::SmallVector<uint32_t>{0, 2 * key_dim},
                                ttsl::SmallVector<uint32_t>{t, conv_dim},
                                ttsl::SmallVector<uint32_t>{1, 1});
  // Heads are laid contiguously along the sliced cols, so [t, hk*dk] ->
  // [t*hk, dk] is a pure logical re-view; each row is one head's Dk vector.
  auto l2 = [&](const ttnn::Tensor& cols) {
    ttnn::Tensor rows =
        ttnn::reshape(cols, ttnn::Shape({t * hk, dk}));
    ttnn::Tensor sq = ttnn::multiply(rows, rows);
    ttnn::Tensor s = ttnn::sum(sq, ttsl::SmallVector<int>{1}, true);
    ttnn::Tensor denom = ttnn::add(s, args.eps);
    ttnn::Tensor inv = ttnn::rsqrt(denom);
    return ttnn::multiply(rows, inv);
  };
  CommitDeviceLogical2D(q_out, l2(q2), t * hk, dk);
  CommitDeviceLogical2D(k_out, l2(k2), t * hk, dk);
  CommitDeviceLogical2D(v_out, ttnn::reshape(v2, ttnn::Shape({t * hv, dv})), t * hv, dv);

  // g/beta in f32 (the row's "f32 intermediates" doctrine: softplus(x) with
  // threshold 20, exp(a_log) and sigmoid must not round their inputs).
  // a_log/dt_bias are model constants — host bytes stay current for the
  // model's lifetime, so EnsureHost is a no-op. a/b are per-step producer
  // outputs: device-authoritative in the captured arm, so they are served
  // from their current shadows (an EnsureHost here enqueues exactly the
  // readback the trace refuses); the eager arm keeps the host-staging
  // fallback for shapes with no committed shadow.
  EnsureHost(a_log);
  EnsureHost(dt_bias);
  ttnn::Tensor dev_a, dev_b;
  if (!(ServePostConvAB(araw, t, hv, dev_a) && ServePostConvAB(braw, t, hv, dev_b))) {
    VT_CHECK(!tt_capture_active(),
             "tenstorrent kGdnPostConv: a/b arrived without a servable "
             "device shadow during trace capture — the producer must commit "
             "device-side before the captured region");
    EnsureHost(araw);
    EnsureHost(braw);
  if (const char* td = std::getenv("VT_DUMP_TRUST")) {
    static std::atomic<int> gb_seq{0};
    if (gb_seq.fetch_add(1, std::memory_order_relaxed) == 0) {
      // Raw HOST bytes of the a/b windows exactly as LoadElemF32 sees them,
      // after EnsureHost. Compare against ba_a_win_dev/ba_b_win_dev.
      std::FILE* fp = std::fopen(
          (std::string(td) + "/0_ab_host_raw.f32").c_str(), "wb");
      if (fp) {
        const float* ap = static_cast<const float*>(
            static_cast<const void*>(araw.data));
        const float* bp = static_cast<const float*>(
            static_cast<const void*>(braw.data));
        for (uint32_t i = 0; i < t; ++i) {
          std::fwrite(ap + i * araw.stride[0], 4, hv, fp);
          std::fwrite(bp + i * braw.stride[0], 4, hv, fp);
        }
        std::fclose(fp);
      }
    }
  }
  std::vector<float> a(static_cast<size_t>(t) * hv), b(a.size());
  for (uint32_t i = 0; i < t; ++i)
    for (uint32_t h = 0; h < hv; ++h) {
      const int64_t aidx = i * araw.stride[0] + h;
      const int64_t bidx = i * braw.stride[0] + h;
      a[static_cast<size_t>(i) * hv + h] = LoadElemF32(araw, aidx);
      b[static_cast<size_t>(i) * hv + h] = LoadElemF32(braw, bidx);
    }
  dev_a = UploadTensor(std::move(a), ttnn::Shape({t, hv}),
                       ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
  dev_b = UploadTensor(std::move(b), ttnn::Shape({t, hv}),
                       ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
  }
  // a_log/dt_bias are immutable model constants — cached host-content tiles
  // (the conv weight discipline): the warm step uploads them once and the
  // captured region serves the resident tile. The per-call UploadTensor this
  // replaces is a write_shard, which the trace refuses — the captured arm
  // fatalled exactly here once a/b stopped EnsureHost-ing first.
  ttnn::Tensor dev_al = CachedTile(
      a_log.data, hv, 0, 0,
      [&] {
        std::vector<float> v(hv);
        for (uint32_t h = 0; h < hv; ++h) v[h] = LoadElemF32(a_log, h);
        return v;
      },
      ttnn::Shape({1, hv}), device);
  ttnn::Tensor dev_dt = CachedTile(
      dt_bias.data, hv, 0, 0,
      [&] {
        std::vector<float> v(hv);
        for (uint32_t h = 0; h < hv; ++h) v[h] = LoadElemF32(dt_bias, h);
        return v;
      },
      ttnn::Shape({1, hv}), device);

  // x = araw + dt_bias (row-vector broadcast); sp = relu(x) +
  // log1p(exp(-|x|)) — the oracle's log1p(exp(x)) written in a form that is
  // stable for every x (the exp argument is always <= 0). For x > 20 the
  // log1p term sits below x's half-ULP, so the sum rounds to x EXACTLY and
  // reproduces the oracle's threshold branch. Used instead of the one-shot
  // SFPU softplus poly, whose ~1e-6 ABSOLUTE fit error becomes ~1e-3
  // RELATIVE on small softplus values (measured on the P150, first green
  // run: g max_rel 4.4e-4 over |g|~0.04). g = -exp(a_log)*sp.
  ttnn::Tensor x = ttnn::add(dev_a, dev_dt);
  ttnn::Tensor sp = ttnn::add(
      ttnn::relu(x), ttnn::log1p(ttnn::exp(ttnn::neg(ttnn::abs(x)))));
  ttnn::Tensor neg_ea = ttnn::multiply(ttnn::exp(dev_al), -1.0f);
  ttnn::Tensor g = ttnn::multiply(neg_ea, sp);
  ttnn::Tensor beta = ttnn::sigmoid(dev_b);
  CommitDeviceLogical2D(g_out, std::move(g), t, hv);
  CommitDeviceLogical2D(beta_out, std::move(beta), t, hv);
  if (const char* td = std::getenv("VT_DUMP_TRUST")) {
    // Kernel-side trusted captures at the COMMIT SITE: same tensor objects,
    // same TU — removes every cross-function identity assumption.
    static std::atomic<int> gpc_seq{0};
    const int call = gpc_seq.fetch_add(1, std::memory_order_relaxed);
    if (call == 0) {
      TrustDump(q, td, "k_conv_in", conv);
      TrustDump(q, td, "k_q", q_out);
      TrustDump(q, td, "k_k", k_out);
      TrustDump(q, td, "k_v", v_out);
      TrustDump(q, td, "k_g", g_out);
      TrustDump(q, td, "k_beta", beta_out);
      // Intermediates of the g chain (device truth via to_vector).
      auto tdv = [&](const char* n, const ttnn::Tensor& t) {
        auto v = t.to_vector<float>();
        std::FILE* f = std::fopen(
            (std::string(td) + "/" + std::to_string(call) + "_" + n + ".f32")
                .c_str(), "wb");
        if (f) { std::fwrite(v.data(), 4, v.size(), f); std::fclose(f); }
      };
      tdv("k_dev_a", dev_a);
      tdv("k_dev_b", dev_b);
      tdv("k_dev_al", dev_al);
      tdv("k_dev_dt", dev_dt);
      tdv("k_x", x);
      tdv("k_sp", sp);
      tdv("k_neg_ea", neg_ea);
    }
  }
}

// kGdnPrefill: the chunked gated-delta-rule scan over a varlen batch, behind
// the tt-metal fused kernel (ttnn::transformer::chunk_gated_delta_rule — one
// Tensix core per (B·HV) head, recurrent state on-core, fp32 state / HiFi4).
// Adapter duties (spec "Design"):
//   - varlen [T,...] + query_start_loc -> ONE dense padded [N,L,...] batch:
//     per-sequence initial states ride the batch's [B,HV,K,V] initial_state,
//     so N sequences cost one op call, not N. L is padded to the 64-token
//     chunk multiple so the op's own time-pad path stays idle; padded rows
//     carry zero q/k/v/g/beta, and g=0/beta=0 is an IDENTITY state update
//     (exp(0)=1, v'=0), so empty sequences and short tails leave the state
//     exactly where the oracle leaves it.
//   - q/k arrive PRE-normalized and GdnArgs::scale multiplies q (the op folds
//     scale into q itself), so use_qk_l2norm stays FALSE (the op fatal-errors
//     on true; FLA scale semantics match the CPU GdnHeadTokenStep).
//   - state layout: ours [N,Hv,Dv,Dk], the op's [B,HV,K,V] — one transpose of
//     the trailing dims on upload and download, inside the adapter.
// Host-staged upload/download in W1 (the decode shadow that keeps the state
// resident is W2's); outputs are written to the host bytes and committed.
void GdnPrefillKernel(Queue&, Tensor& out, const Tensor& q_in, const Tensor& k_in,
                      const Tensor& v_in, const Tensor& g, const Tensor& beta,
                      Tensor& state, const Tensor& qsl, const GdnArgs& args) {
  TT_OP_TRACE("GdnPrefill");
  const int64_t n = state.shape[0], hv = state.shape[1], dv = state.shape[2],
                dk = state.shape[3];
  const int64_t hk = q_in.shape[1];
  const int64_t total = q_in.shape[0];
  // chunk_gated_delta_rule device constraints (validated TILE dims): refuse
  // by name rather than silently reshape into a wrong answer (spec Risk #4).
  VT_CHECK(dk % 32 == 0 && dv % 32 == 0,
           "tenstorrent gdn_prefill: tt-metal chunk_gated_delta_rule needs "
           "Dk/Dv multiples of 32 (tile), got Dk=" +
               std::to_string(dk) + " Dv=" + std::to_string(dv));
  EnsureHost(qsl);
  const int32_t* qslp = qsl.Ptr<int32_t>();
  VT_CHECK(qslp[0] == 0 && qslp[n] == total,
           "tenstorrent gdn_prefill: bad query_start_loc bounds");
  int64_t max_len = 0;
  for (int64_t s = 0; s < n; ++s) {
    VT_CHECK(qslp[s + 1] >= qslp[s],
             "tenstorrent gdn_prefill: query_start_loc not monotonic");
    max_len = std::max(max_len, static_cast<int64_t>(qslp[s + 1] - qslp[s]));
  }
  if (total == 0) return;  // all sequences empty: no out rows, state as-is

  constexpr int64_t kChunk = 64;
  const int64_t len = ((max_len + kChunk - 1) / kChunk) * kChunk;

  EnsureHost(q_in);
  EnsureHost(k_in);
  EnsureHost(v_in);
  EnsureHost(g);
  EnsureHost(beta);
  EnsureHost(state);
  EnsureHost(out);

  // Dense padded batch [N, len, ...]: zero-filled tails are identity updates.
  const size_t qk_elems = static_cast<size_t>(n) * len * hk * dk;
  const size_t v_elems = static_cast<size_t>(n) * len * hv * dv;
  std::vector<float> qp(qk_elems, 0.0f), kp(qk_elems, 0.0f), vp(v_elems, 0.0f);
  std::vector<float> gp(static_cast<size_t>(n) * len * hv, 0.0f);
  std::vector<float> bp(static_cast<size_t>(n) * len * hv, 0.0f);
  // initial state [N,Hv,Dk,Dv] — the trailing-dims transpose of ours.
  std::vector<float> s0(static_cast<size_t>(n) * hv * dk * dv, 0.0f);
  for (int64_t s = 0; s < n; ++s) {
    const int64_t t0 = qslp[s], t_len = qslp[s + 1] - t0;
    for (int64_t t = 0; t < t_len; ++t) {
      const int64_t src = t0 + t, dst = s * len + t;
      for (int64_t e = 0; e < hk * dk; ++e) {
        qp[static_cast<size_t>(dst * hk * dk + e)] =
            LoadElemF32(q_in, src * hk * dk + e);
        kp[static_cast<size_t>(dst * hk * dk + e)] =
            LoadElemF32(k_in, src * hk * dk + e);
      }
      for (int64_t e = 0; e < hv * dv; ++e)
        vp[static_cast<size_t>(dst * hv * dv + e)] =
            LoadElemF32(v_in, src * hv * dv + e);
      for (int64_t e = 0; e < hv; ++e) {
        gp[static_cast<size_t>(dst * hv + e)] = LoadElemF32(g, src * hv + e);
        bp[static_cast<size_t>(dst * hv + e)] = LoadElemF32(beta, src * hv + e);
      }
    }
    for (int64_t h = 0; h < hv; ++h) {
      const float* ours =
          state.Ptr<float>() + ((s * hv + h) * dv) * dk;  // [Dv,Dk] rows
      float* theirs = &s0[static_cast<size_t>((s * hv + h) * dk) * dv];  // [Dk,Dv]
      for (int64_t i = 0; i < dv; ++i)
        for (int64_t j = 0; j < dk; ++j)
          theirs[static_cast<size_t>(j * dv + i)] =
              ours[static_cast<size_t>(i * dk + j)];
    }
  }

  MeshDevice& device = SharedMeshDevice();
  const auto grid = device.compute_with_storage_grid_size();
  const uint32_t cores =
      static_cast<uint32_t>(grid.x) * static_cast<uint32_t>(grid.y);
  VT_CHECK(hv <= static_cast<int64_t>(cores),
           "tenstorrent gdn_prefill: Hv=" + std::to_string(hv) +
               " alone exceeds the device compute-core count " +
               std::to_string(cores) +
               " — tt-metal chunk_gated_delta_rule distributes BH rows over "
               "cores and cannot run at any batch size on this device");
  // BH = batch * value heads is the flattened row count tt-metal's scan
  // distributes over cores (chunk_gdn_phased_program_factory requires BH <=
  // ncores). Above that bound, decompose into sub-batch calls: the op
  // factorizes exactly over the (batch, head) dim — the scan recurrence is
  // per-row, the final state is one row per (batch, head), and per-sequence
  // initial states slice along dim 0 — so per-sub-batch outputs and states
  // concatenate back to the whole-batch result bit-for-bit. sub_b >= n
  // (hence every c=1 call) stays a single call.
  int64_t sub_b = n;
  if (n * hv > static_cast<int64_t>(cores))
    sub_b = std::min(n, std::max<int64_t>(1, static_cast<int64_t>(cores) / hv));
  std::vector<float> ov, fv;  // [N,len,Hv,Dv] / [N,Hv,Dk,Dv] slices, in order
  const size_t qk_stride = static_cast<size_t>(len) * hk * dk;
  const size_t v_stride = static_cast<size_t>(len) * hv * dv;
  const size_t gb_stride = static_cast<size_t>(len) * hv;
  const size_t s0_stride = static_cast<size_t>(hv) * dk * dv;
  for (int64_t b0 = 0; b0 < n; b0 += sub_b) {
    const int64_t bn = std::min(sub_b, n - b0);
    const uint32_t un = static_cast<uint32_t>(bn), ul = static_cast<uint32_t>(len),
                   uhk = static_cast<uint32_t>(hk), uhv = static_cast<uint32_t>(hv),
                   udk = static_cast<uint32_t>(dk), udv = static_cast<uint32_t>(dv);
    // A single full-batch call (sub_b >= n) owns the whole packed vector:
    // move it into the upload instead of staging a full copy (the pre-fix
    // single-call path). The split path always slices.
    auto slice = [&](std::vector<float>& src,
                     size_t stride) -> std::vector<float> {
      if (b0 == 0 && bn == n)
        return std::move(src);
      return std::vector<float>(src.begin() + b0 * stride,
                                src.begin() + (b0 + bn) * stride);
    };
    ttnn::Tensor dev_q =
        UploadTensor(slice(qp, qk_stride), ttnn::Shape({un, ul, uhk, udk}),
                     ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, device);
    ttnn::Tensor dev_k =
        UploadTensor(slice(kp, qk_stride), ttnn::Shape({un, ul, uhk, udk}),
                     ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, device);
    ttnn::Tensor dev_v =
        UploadTensor(slice(vp, v_stride), ttnn::Shape({un, ul, uhv, udv}),
                     ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, device);
    ttnn::Tensor dev_g =
        UploadTensor(slice(gp, gb_stride), ttnn::Shape({un, ul, uhv}),
                     ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
    ttnn::Tensor dev_b =
        UploadTensor(slice(bp, gb_stride), ttnn::Shape({un, ul, uhv}),
                     ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
    ttnn::Tensor dev_s0 =
        UploadTensor(slice(s0, s0_stride), ttnn::Shape({un, uhv, udk, udv}),
                     ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
    auto [o, final_state] = ttnn::transformer::chunk_gated_delta_rule(
        dev_q, dev_k, dev_v, dev_g, dev_b, args.scale, dev_s0,
        /*output_final_state=*/true, static_cast<uint32_t>(kChunk),
        /*use_qk_l2norm=*/false,  // caller pre-normalized q/k
        /*output_head_major=*/false);
    VT_CHECK(final_state.has_value(),
             "tenstorrent gdn_prefill: chunk_gated_delta_rule returned no final state");
    std::vector<float> ovec = o.to_vector<float>();             // [bn,len,Hv,Dv]
    std::vector<float> fvec = final_state->to_vector<float>();  // [bn,Hv,Dk,Dv]
    ov.insert(ov.end(), ovec.begin(), ovec.end());
    fv.insert(fv.end(), fvec.begin(), fvec.end());
  }

  // Scatter token-major outputs back into the packed varlen rows.
  for (int64_t s = 0; s < n; ++s) {
    const int64_t t_len = qslp[s + 1] - qslp[s];
    for (int64_t t = 0; t < t_len; ++t)
      for (int64_t e = 0; e < hv * dv; ++e)
        StoreElemF32(out, (qslp[s] + t) * hv * dv + e,
                     ov[static_cast<size_t>(((s * len + t) * hv) * dv + e)]);
  }
  // Final state: transpose the trailing dims back into [N,Hv,Dv,Dk].
  for (int64_t s = 0; s < n; ++s)
    for (int64_t h = 0; h < hv; ++h) {
      const float* theirs =
          &fv[static_cast<size_t>((s * hv + h) * dk) * dv];  // [Dk,Dv]
      float* ours = state.Ptr<float>() + ((s * hv + h) * dv) * dk;  // [Dv,Dk]
      for (int64_t i = 0; i < dv; ++i)
        for (int64_t j = 0; j < dk; ++j)
          ours[static_cast<size_t>(i * dk + j)] =
              theirs[static_cast<size_t>(j * dv + i)];
    }
  CommitHost(out);
  CommitHost(state);
}

namespace {

// The shadow-row split factor for a wide cache: indexed_fill's generic
// interleaved path stages TWO full pages of the LAST dim through its
// dataflow buffer (indexed_fill_program_factory.cpp: page_size =
// padded_shape[-1] * elem_size, data DFB num_entries = 2), so a row wider
// than kStageBytes/(2*esz) elems overflows L1 — the Qwen3.5 GDN ssm_state
// row Hv*Dk*Dv = 16*128*128 = 262144 f32 elems stages 2 x 1 MB and threw
// exactly that ("grow to 2208704 B beyond max L1 size of 1572864 B", first
// W2a e2e bootstrap, /tmp/w2a_e2e_bootstrap.log). The fix is NOT a
// last-dim-changing view: tt::tt_metal::view moves no bytes, but device
// pages are the INTERLEAVE UNIT (buffer.cpp Buffer::page_address:
// bank_offset = aligned_page_size * (page_index / num_banks)) — reinterpreting
// [rows, cols] as [rows*nb, cb] changes every page index's bank mapping and
// scrambles the flat order (measured in W2a: one block landed correctly,
// three came back scrambled). Instead the SHADOW IS BORN SPLIT:
// EnsureGdnCacheDevice uploads the SAME flat host bytes as
// [rows*F, cols/F], so every later op sees narrow rows natively and the
// fill stays ONE launch inside the budget. F is the smallest divisor of
// cols with 2*(cols/F)*esz <= 1 MiB; F == 1 (cols at or under the budget —
// every small-cache model) keeps the W1-verified form byte-identical.
int64_t SplitFactor(int64_t cols, int64_t esz) {
  constexpr int64_t kStageBytes = 1 << 20;
  int64_t f = std::max<int64_t>(1, (2 * cols * esz + kStageBytes - 1) / kStageBytes);
  while (cols % f != 0) ++f;
  return f;
}

// Ensure the f32 device shadow of a GDN state/conv CACHE tensor, viewed as
// [rows, cols]. Uploads (counted) only when the shadow is missing, stale
// (host wrote), or the wrong shape/dtype. The device tensor is FLOAT32 TILE
// regardless of the host cache dtype (bf16 caches round on upload/download —
// EnsureHostBytes already does the f32→bf16 rounding on the way out).
ttnn::Tensor EnsureGdnCacheDevice(const Tensor& t, int64_t rows, int64_t cols,
                                  MeshDevice& device,
                                  ttnn::Layout layout = ttnn::Layout::TILE) {
  // The shadow is stored SPLIT: [rows*F, cols/F] with F = SplitFactor(cols)
  // (see the helper above — wide rows must not become indexed_fill pages).
  // dev_rows/dev_cols stay LOGICAL; the flat bytes are identical either way,
  // so every download and every volume check is unaffected.
  const uint32_t sf = static_cast<uint32_t>(SplitFactor(cols, /*esz=*/4));
  const uint32_t ublk = static_cast<uint32_t>(cols / sf);
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(t.data);
    if (s != nullptr && s->conv_transposed) {
      // W3 #2201, scoped to the corrupting case by the HIGH review finding:
      // ONE conv-state buffer legitimately reaches this path under two views —
      // the decode step commits it time-major ([sl+1, R], conv_transposed,
      // host-stale) and a later prefill-bearing step gathers it in the
      // ssm/cache view at a DIFFERING volume (qwen3_5.cpp GdnStateGather on
      // state.conv_state). That transition must fall through to the slow path
      // below: EnsureHost transposes the shadow back into the caller's order
      // and the refresh re-uploads it in this role's geometry (pinned by the
      // decode-to-prefill alternation test). What must never happen is the
      // fast path serving — or, at equal volume, reshaping — the transposed
      // shadow under this role's shape: same numel, different geometry is a
      // silent wrong-geometry serve. Refuse exactly that, naming both
      // geometries.
      VT_CHECK(static_cast<uint64_t>(s->dev_rows) * s->dev_cols !=
                   static_cast<uint64_t>(rows) * cols,
               "tenstorrent: GDN cache role mismatch: host pointer already "
               "holds a conv_transposed shadow [" + std::to_string(s->dev_rows) +
                   "x" + std::to_string(s->dev_cols) +
                   "]; refusing to serve or reshape it at equal volume as [" +
                   std::to_string(rows) + "x" + std::to_string(cols) + "]");
    }
    if (s != nullptr && s->device_current && s->device.has_value() &&
        s->device->dtype() == ttnn::DataType::FLOAT32 &&
        s->device->layout() == layout &&
        static_cast<uint64_t>(s->dev_rows) * s->dev_cols ==
            static_cast<uint64_t>(rows) * cols) {
      if (s->dev_rows == rows && s->dev_cols == cols) return *s->device;
      // Same buffer re-served at new logical dims: reshape to the NEW dims'
      // split geometry (equal volume). Always use the member reshape (pure
      // view) to produce the same padded shape in both eager and capture,
      // so downstream ops (untilize in ScatterRowsDevice) see the same input
      // and hit the program cache.
      const auto old_padded = s->device->padded_shape();
      ttnn::Tensor reshaped = s->device->reshape(
          ttnn::Shape({static_cast<uint32_t>(rows) * sf, ublk}), old_padded);
      s->device = reshaped;
      s->dev_rows = static_cast<uint32_t>(rows);
      s->dev_cols = static_cast<uint32_t>(cols);
      return reshaped;
    }
    // W3 #2201: the refresh below calls EnsureHost, which downloads a
    // device-current host-stale shadow (wrong dtype/layout for this role —
    // a foreign commit over the same buffer) before the re-upload. Count
    // that d2h traffic; the download itself is the shadow's logical volume.
    if (s != nullptr && !s->host_current) {
      GdnStateD2hBytes().fetch_add(
          static_cast<uint64_t>(s->dev_rows) * s->dev_cols * sizeof(float),
          std::memory_order_relaxed);
    }
  }
  // (Cold arm only — the fast path above serves the warmed f32 shadow
  // in-region. Refuse a capture-time miss BEFORE the EnsureHost for the same
  // reason as the conv shadow: across a prefill role transition the slot is
  // device-current host-stale, and downloading during capture is the
  // trace-refused read that aborts the process instead of naming the
  // condition.)
  VT_CHECK(!tt_capture_active(),
           "tenstorrent: GDN cache shadow not warmed during capture — the "
           "cold step must establish it first");
  EnsureHost(t);  // host truth (downloads a stale foreign shadow if any)
  const int64_t n = t.Numel();
  std::vector<float> host(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) host[static_cast<size_t>(i)] = LoadElemF32(t, i);
  GdnStateH2dBytes().fetch_add(static_cast<uint64_t>(n) * sizeof(float),
                               std::memory_order_relaxed);
  ttnn::Tensor dev = UploadTensor(std::move(host),
                                  ttnn::Shape({static_cast<uint32_t>(rows) * sf,
                                               ublk}),
                                  ttnn::DataType::FLOAT32, layout, device);
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(t.data);
  if (s != nullptr) {
    s->device = dev;
    s->dev_rows = static_cast<uint32_t>(rows);
    s->dev_cols = static_cast<uint32_t>(cols);
    s->device_current = true;
    s->host_current = true;
    s->device_reserved = false;  // real bytes staged — the reservation is spent
    // The re-upload replaces the transposed shadow with this role's logical
    // [rows, cols] split layout — the slot is no longer transposed, and a
    // stale flag would make the equal-volume refusal above fire on the next
    // ordinary serve (BufferSlot: every shadow-replacing commit restates the
    // layout).
    s->conv_transposed = false;
  }
  return dev;
}

// The [rows, slots] one-hot f32 matrix for `idx` (idx<0 = NULL row → zero
// row): G@X gathers rows — each row independently reads its own slot, so
// duplicates are fine. Only the DECODE kernel still uses it (a [batch, slots]
// matrix is far smaller than GatherRowsExact's [batch, row-width] index on
// the wide state rows); its matmul rounds the gathered state to tf32, which
// sits inside the decode step's own tf32 envelope. Exact paths use
// GatherRowsExact/ScatterRowsExact above.
//
// `factor` adapts the matrix to the SPLIT shadow geometry
// (EnsureGdnCacheDevice): the result is [rows*factor, slots*factor] with
// gmat[(b*F+j)][(s*F+j')] = 1 iff s == idx[b] and j == j', so each gathered
// block-row lands under its own batch row. Same values, same tf32 envelope.
ttnn::Tensor UploadOneHot(const std::vector<int32_t>& idx, int64_t slots,
                          MeshDevice& device, int64_t factor = 1) {
  const int64_t rows = static_cast<int64_t>(idx.size());
  const int64_t scols = slots * factor;
  std::vector<float> gmat(static_cast<size_t>(rows * factor) *
                              static_cast<size_t>(scols),
                          0.0f);
  for (int64_t r = 0; r < rows; ++r) {
    const int32_t ix = idx[static_cast<size_t>(r)];
    if (ix < 0) continue;  // NULL row: gathers zeros
    for (int64_t j = 0; j < factor; ++j)
      gmat[static_cast<size_t>(r * factor + j) * static_cast<size_t>(scols) +
           static_cast<size_t>(ix * factor + j)] = 1.0f;
  }
  return UploadTensor(std::move(gmat),
                      ttnn::Shape({static_cast<uint32_t>(rows * factor),
                                   static_cast<uint32_t>(scols)}),
                      ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
}

// The gdn_decode-side entry: the ssm one-hot gather and the split-block
// scatter ids, in the same content-keyed discipline.
GdnIdxCacheEntry GdnSsmIdxEntry(const std::vector<int32_t>& idxv,
                                uint32_t slots, uint32_t sf,
                                MeshDevice& device) {
  const std::array<uint64_t, 3> key{slots, sf, idxv.size()};
  std::lock_guard<std::mutex> g(GdnIdxCacheMutex());
  GdnIdxCacheEntry& e = GdnIdxCacheMap()[key];
  if (e.warmed_gdn && e.idx == idxv) return e;
  VT_CHECK(!tt_capture_active(),
           "tenstorrent gdn_decode: state-slot indices changed during trace "
           "capture — decode slots must be stable across a sequence's steps; "
           "the recapture cadence owns a slot change (reset the graph, then "
           "re-warm on the eager boundary step)");
  GdnIdxCacheEntry n;
  n.idx = idxv;
  n.warmed_gdn = true;
  n.oh = UploadOneHot(idxv, slots, device, sf);
  std::vector<uint32_t> blocks;
  blocks.reserve(idxv.size() * static_cast<size_t>(sf));
  for (int32_t ix : idxv)
    for (uint32_t j = 0; j < sf; ++j)
      blocks.push_back(static_cast<uint32_t>(ix) * sf + j);
  n.scatter_bid = UploadIdxU32(
      std::move(blocks), ttnn::Shape({static_cast<uint32_t>(idxv.size()) * sf}),
      ttnn::Layout::ROW_MAJOR, device);
  e = std::move(n);
  return e;
}

// The GQA head-map gather index for gdn_decode: rows[(b*Hv + h)*Dk + j] =
// b*Hk + h/ratio — the full-width u32 TILE form the pin's dim-0 gather takes
// (the same shape rule as GatherRowsExact's index and the conv out-select's
// idx_rows). Step-constant per (batch, hv, hk, dk), so the conv-tile cache
// discipline applies; the content is a pure function of the key, so no host
// bytes are kept, and the upload belongs to the eager cold step.
ttnn::Tensor CachedHeadMapIdx(uint64_t batch, uint64_t hv, uint64_t hk,
                              uint64_t dk, MeshDevice& device) {
  const std::array<uint64_t, 4> key{batch, hv, hk, dk};
  std::lock_guard<std::mutex> g(ConvTileCacheMutex());
  auto it = ConvTileCacheMap().find(key);
  if (it != ConvTileCacheMap().end()) return it->second.dev;
  VT_CHECK(!tt_capture_active(),
           "tenstorrent gdn_decode: head-map gather index not warmed — the "
           "cold step must build it before the captured region");
  const uint32_t rows_n = static_cast<uint32_t>(batch * hv);
  const uint64_t ratio = hv / hk;
  std::vector<uint32_t> rows(static_cast<size_t>(rows_n) * dk);
  for (uint64_t b = 0; b < batch; ++b)
    for (uint64_t h = 0; h < hv; ++h)
      for (uint64_t j = 0; j < dk; ++j)
        rows[static_cast<size_t>((b * hv + h) * dk + j)] =
            static_cast<uint32_t>(b * hk + h / ratio);
  ConvTileEntry e;
  e.dev = UploadIdxU32(std::move(rows),
                       ttnn::Shape({rows_n, static_cast<uint32_t>(dk)}),
                       ttnn::Layout::TILE, device);
  ConvTileCacheMap().emplace(key, std::move(e));
  return ConvTileCacheMap()[key].dev;
}

// The rank-1 gated-delta step for a gathered batch, state [B*Hv, Dv, Dk] f32
// on device: S *= exp(g); dot = S@k; v' = (v-dot)*beta; S += v'k^T;
// o = S@(q*scale). Returns {o [B*Hv, Dv, 1], S_new [B*Hv, Dv, Dk]} — the
// state stays on the card; only q/k/v/g/beta cross PCIe per step.
struct DecodeStepResult {
  ttnn::Tensor o;
  ttnn::Tensor state;
};

DecodeStepResult GdnDecodeStepComposed(const ttnn::Tensor& S, const ttnn::Tensor& dev_q,
                                       const ttnn::Tensor& dev_k, const ttnn::Tensor& dev_v,
                                       const ttnn::Tensor& dev_g, const ttnn::Tensor& dev_b,
                                       float scale, uint32_t bh, uint32_t dk, uint32_t dv) {
  // During capture, use the free ttnn::reshape: the program IS cached from
  // the eager warmup (GdnDecodeStepComposed runs the same reshapes in both
  // passes). create_program_artifacts only runs on cache miss, so the cache
  // hit path doesn't call to_device.
  ttnn::Tensor decay = ttnn::exp(ttnn::reshape(dev_g, ttnn::Shape({bh, 1, 1})));
  ttnn::Tensor Sd = ttnn::multiply(S, decay);
  ttnn::Tensor krow = ttnn::reshape(dev_k, ttnn::Shape({bh, 1, dk}));
  // f32-safe reductions: ttnn::matmul on this pin TRUNCATES f32 operands to
  // tf32 in its MACs on Blackhole, and the ComputeConfig does NOT lift it —
  // HiFi4 + fp32_dest_acc_en + math_approx off leaves the result bit-identical
  // to the plain call (probed on the wide-range fidelity arm's distribution:
  // matmul max_abs 0.53 vs a double reference, broadcast multiply + ttnn::sum
  // 4e-5). The truncated dot error lands in the small-magnitude elements of
  // the delta-rule correction (v - S@k) and compounds across the GDN layers —
  // pinned by the wide-range arm in tests/vt/test_tenstorrent_backend.cpp.
  // The eltwise multiply is exact f32 SFPU and ttnn::sum accumulates f32, so
  // the three GEMMs route through broadcast multiply + sum (the outer product
  // has no reduction at all — one exact product per element):
  ttnn::Tensor dot = ttnn::sum(ttnn::multiply(Sd, krow), ttsl::SmallVector<int>{2},
                               /*keep_dim=*/true);
  ttnn::Tensor vcol = ttnn::reshape(dev_v, ttnn::Shape({bh, dv, 1}));
  ttnn::Tensor beta = ttnn::reshape(dev_b, ttnn::Shape({bh, 1, 1}));
  ttnn::Tensor vp = ttnn::multiply(ttnn::subtract(vcol, dot), beta);
  ttnn::Tensor S2 = ttnn::add(Sd, ttnn::multiply(vp, krow));
  ttnn::Tensor qs = ttnn::multiply(dev_q, scale);
  ttnn::Tensor o = ttnn::sum(
      ttnn::multiply(S2, ttnn::reshape(qs, ttnn::Shape({bh, 1, dk}))),
      ttsl::SmallVector<int>{2}, /*keep_dim=*/true);
  return {std::move(o), std::move(S2)};
}

}  // namespace

// kGdnDecode (§7; cpu_ops.cpp GdnDecodeKernel). Both forms: compact
// state [B,Hv,Dv,Dk] (row == token) and the indexed FULL cache (slot per
// token, NULL idx<0 ⇒ out row zeroed, slot untouched). State lives in the
// device shadow across steps — the traffic counters prove no per-token
// round-trip. Two adapter choices, spec Design: the composed rank-1 step
// (default) and one T=1 chunk_gated_delta_rule call per step
// (VT_TT_GDN_DECODE=chunked diagnostic; the op pads T to its chunk multiple
// internally, so it pays the full-chunk cost for one token).
void GdnDecodeKernel(Queue&, Tensor& out, const Tensor& q_in, const Tensor& k,
                     const Tensor& v, const Tensor& g, const Tensor& beta,
                     Tensor& state, const Tensor* state_idx, const GdnArgs& args) {
  TT_OP_TRACE("GdnDecode");
  GdnDecodeSteps().fetch_add(1, std::memory_order_relaxed);
  const int64_t batch = q_in.shape[0];
  const int64_t hv = state.shape[1], dv = state.shape[2], dk = state.shape[3];
  const int64_t hk = q_in.shape[1];
  const int64_t ratio = hv / hk;
  const int64_t slots = state.shape[0];
  if (batch == 0) return;  // empty batch: no-op, state as-is
  bool has_null = false;
  std::vector<int32_t> idxv;
  if (state_idx != nullptr) {
    idxv = ReadIdxHost(*state_idx, slots, "gdn_decode", true);
    for (int32_t ix : idxv)
      if (ix < 0) has_null = true;
  }
  // NULL rows zero their out row and skip the state update — supported on
  // the eager arm through the host-staged validity mask. The captured arm
  // refuses them by name: the gated captured battery runs unpadded batches
  // (PadToCaptureSize(1) == 1), so the padded-capture arm stays a named owed
  // item rather than a silent divergence from the oracle's skip semantics.
  VT_CHECK(!has_null || !tt_capture_active(),
           "tenstorrent gdn_decode: NULL state rows (idx<0) under trace "
           "capture are the padded-capture arm this row owes — refusing "
           "rather than diverging from the oracle's skip semantics");

  MeshDevice& device = SharedMeshDevice();
  const uint32_t ub = static_cast<uint32_t>(batch), uhv = static_cast<uint32_t>(hv),
                  udk = static_cast<uint32_t>(dk), udv = static_cast<uint32_t>(dv),
                  bh = static_cast<uint32_t>(batch * hv);
  // Device-pure staging: the projections commit their outputs device-side in
  // every phase, so the activations are served from their current shadows
  // (ServeActF32; the eager arm keeps its host-staging fallback). q/k expand
  // to Hv heads — the hk head each value head maps to, exactly the CPU
  // GdnHeadTokenStep head map — through the proven dim-0 gather (pure data
  // movement, bit-identical, unlike the one-hot matmul's tf32 rounding); the
  // full-width u32 index is step-constant per geometry and warmed outside
  // capture, and ratio==1 needs no movement at all.
  ttnn::Tensor q2d = ServeActF32(q_in, static_cast<uint32_t>(batch * hk), udk,
                                 "q", device);
  ttnn::Tensor k2d =
      ServeActF32(k, static_cast<uint32_t>(batch * hk), udk, "k", device);
  ttnn::Tensor dev_q, dev_k;
  if (ratio == 1) {
    dev_q = std::move(q2d);
    dev_k = std::move(k2d);
  } else {
    ttnn::Tensor head_map =
        CachedHeadMapIdx(ub, uhv, static_cast<uint32_t>(hk), udk, device);
    dev_q =
        ttnn::gather(q2d, /*dim=*/0, head_map, /*sparse_grad=*/false, std::nullopt);
    dev_k =
        ttnn::gather(k2d, /*dim=*/0, head_map, /*sparse_grad=*/false, std::nullopt);
  }
  ttnn::Tensor dev_v = ServeActF32(v, bh, udv, "v", device);
  ttnn::Tensor dev_g = ServeActF32(g, bh, 1, "g", device);
  ttnn::Tensor dev_b = ServeActF32(beta, bh, 1, "beta", device);

  // The shadow is born split ([slots*F, blk], F = SplitFactor of the state
  // row) — see EnsureGdnCacheDevice / SplitFactor.
  const uint32_t sf =
      static_cast<uint32_t>(SplitFactor(hv * dv * dk, /*esz=*/4));
  ttnn::Tensor cache2d =
      EnsureGdnCacheDevice(state, slots, hv * dv * dk, device);
  std::optional<ttnn::Tensor> oh;
  GdnIdxCacheEntry sentry;
  ttnn::Tensor S;  // [B*Hv, Dv, Dk] — the batch's state rows, on device
  if (state_idx != nullptr) {
    // The one-hot gather matrix and the split-block scatter ids are warmed
    // outside capture (content-keyed: a slot change mid-capture is the
    // recapture cadence's, and the VT_CHECK inside names it).
    sentry = GdnSsmIdxEntry(idxv, slots, sf, device);
    oh = sentry.oh;
    S = CaptureSafeReshape(ttnn::matmul(*oh, cache2d),
                      ttnn::Shape({bh, udv, udk}));
  } else {
    S = CaptureSafeReshape(cache2d, ttnn::Shape({bh, udv, udk}));
  }

  // W2 (capture-warmup redesign) instrumented state probe. VT_TT_GDN_STATE_PROBE
  // checksums the step's INITIAL state rows as served from the cache (one
  // line per eager decode call, outside capture only — a download is the
  // read the trace refuses, and a replay runs no host code at all). The
  // frozen-recurrence defect prints ONE IDENTICAL checksum at every step of
  // a sequence (the pre-warmup state rolled back over each step's commit);
  // an advancing recurrence prints changing checksums. The gate leg that
  // adjudicates it runs VLLM_CPP_CUDAGRAPH=0 so every step is eager.
  if (const char* st_probe = std::getenv("VT_TT_GDN_STATE_PROBE");
      st_probe != nullptr && st_probe[0] != '\0' && !tt_capture_active()) {
    std::vector<float> sv = S.to_vector<float>();
    uint64_t fnv = 1469598103934665603ULL;
    double acc = 0.0;
    for (float f : sv) {
      uint32_t bits;
      std::memcpy(&bits, &f, sizeof(bits));
      fnv = (fnv ^ bits) * 1099511628211ULL;
      acc += f > 0.0f ? f : -f;
    }
    std::fprintf(stderr,
                 "[GDN-STATE-PROBE] step=%llu slots=%lld bh=%u n=%zu "
                 "fnv=%016llx asum=%a\n",
                 static_cast<unsigned long long>(GdnDecodeSteps().load(
                     std::memory_order_relaxed)),
                 static_cast<long long>(slots), bh, sv.size(),
                 static_cast<unsigned long long>(fnv), acc);
    std::fflush(stderr);
  }

  const char* mode = std::getenv("VT_TT_GDN_DECODE");
  const bool chunked = mode != nullptr && std::string_view(mode) == "chunked";
  VT_CHECK(!chunked || !tt_capture_active(),
           "tenstorrent gdn_decode: the VT_TT_GDN_DECODE=chunked diagnostic "
           "stages its inputs host-side and cannot run inside a trace "
           "capture");
  ttnn::Tensor o;      // [B*Hv, Dv, 1] (composed) or [B, 1, Hv, Dv] (chunked)
  ttnn::Tensor S_new;  // [B*Hv, Dv, Dk] (composed) or [B, Hv, Dk, Dv] (chunked)
  if (chunked) {
    // One T=1 fused call on the whole batch (the alternative adapter, kept as
    // a diagnostic per the spec: pad/permute duties mirror GdnPrefillKernel).
    VT_CHECK(dk % 32 == 0 && dv % 32 == 0,
             "tenstorrent gdn_decode: the chunked adapter (VT_TT_GDN_DECODE="
             "chunked) needs Dk/Dv multiples of 32");
    ttnn::Tensor s_b =
        ttnn::reshape(S, ttnn::Shape({ub, uhv, udv, udk}));
    ttnn::Tensor s0 = ttnn::permute(s_b, ttsl::SmallVector<int64_t>{0, 1, 3, 2});
    // q/k [B,1,Hk,Dk] from the compact staged rows; v [B,1,Hv,Dv].
    std::vector<float> qc(static_cast<size_t>(batch) * hk * dk),
        kc(static_cast<size_t>(batch) * hk * dk), vc(static_cast<size_t>(batch) * hv * dv),
        gc(static_cast<size_t>(batch) * hv), bc(static_cast<size_t>(batch) * hv);
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t h = 0; h < hv; ++h) {
        for (int64_t j = 0; j < dk; ++j) {
          qc[static_cast<size_t>((b * hk + h / ratio) * dk + j)] =
              LoadElemF32(q_in, (b * hk + h / ratio) * dk + j);
          kc[static_cast<size_t>((b * hk + h / ratio) * dk + j)] =
              LoadElemF32(k, (b * hk + h / ratio) * dk + j);
        }
        for (int64_t j = 0; j < dv; ++j)
          vc[static_cast<size_t>((b * hv + h) * dv + j)] =
              LoadElemF32(v, (b * hv + h) * dv + j);
        gc[static_cast<size_t>(b * hv + h)] = LoadElemF32(g, b * hv + h);
        bc[static_cast<size_t>(b * hv + h)] = LoadElemF32(beta, b * hv + h);
      }
    }
    // Same core bound as prefill: BH = batch * hv is the scan's flattened
    // row count. Decompose above the bound; sub_b >= batch (hence c=1)
    // stays a single call: the staged host vectors move whole into the
    // uploads and the device s0 takes a full-extent slice (a no-op).
    const auto chunked_grid = device.compute_with_storage_grid_size();
    const uint32_t chunked_cores = static_cast<uint32_t>(chunked_grid.x) *
                                   static_cast<uint32_t>(chunked_grid.y);
    VT_CHECK(hv <= static_cast<int64_t>(chunked_cores),
             "tenstorrent gdn_decode: Hv=" + std::to_string(hv) +
                 " alone exceeds the device compute-core count " +
                 std::to_string(chunked_cores) +
                 " — tt-metal chunk_gated_delta_rule distributes BH rows over "
                 "cores and cannot run at any batch size on this device");
    int64_t sub_b = batch;
    if (batch * hv > static_cast<int64_t>(chunked_cores))
      sub_b = std::min(batch, std::max<int64_t>(
                                  1, static_cast<int64_t>(chunked_cores) / hv));
    const size_t qk_stride = static_cast<size_t>(hk) * dk;
    const size_t v_stride = static_cast<size_t>(hv) * dv;
    const size_t gb_stride = static_cast<size_t>(hv);
    std::vector<ttnn::Tensor> os, fss;  // per-sub-batch o ([bn,1,Hv,Dv]) and state
    for (int64_t b0 = 0; b0 < batch; b0 += sub_b) {
      const int64_t bn = std::min(sub_b, batch - b0);
      const uint32_t ubn = static_cast<uint32_t>(bn);
      // Same single-call rule as prefill: sub_b >= batch moves the whole
      // staged vector into the upload; the split path always slices.
      auto slice = [&](std::vector<float>& src,
                       size_t stride) -> std::vector<float> {
        if (b0 == 0 && bn == batch)
          return std::move(src);
        return std::vector<float>(src.begin() + b0 * stride,
                                  src.begin() + (b0 + bn) * stride);
      };
      ttnn::Tensor s0_sl = ttnn::slice(
          s0, ttsl::SmallVector<uint32_t>{static_cast<uint32_t>(b0), 0u, 0u, 0u},
          ttsl::SmallVector<uint32_t>{static_cast<uint32_t>(b0 + bn), uhv, udk,
                                      udv},
          ttsl::SmallVector<uint32_t>{1u, 1u, 1u, 1u});
      ttnn::Tensor dq = UploadTensor(
          slice(qc, qk_stride), ttnn::Shape({ubn, 1, static_cast<uint32_t>(hk), udk}),
          ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
      ttnn::Tensor dk4 = UploadTensor(
          slice(kc, qk_stride), ttnn::Shape({ubn, 1, static_cast<uint32_t>(hk), udk}),
          ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
      ttnn::Tensor dv4 = UploadTensor(slice(vc, v_stride),
                                      ttnn::Shape({ubn, 1, uhv, udv}),
                                      ttnn::DataType::FLOAT32, ttnn::Layout::TILE,
                                      device);
      ttnn::Tensor dg = UploadTensor(slice(gc, gb_stride), ttnn::Shape({ubn, 1, uhv}),
                                     ttnn::DataType::FLOAT32, ttnn::Layout::TILE,
                                     device);
      ttnn::Tensor db = UploadTensor(slice(bc, gb_stride), ttnn::Shape({ubn, 1, uhv}),
                                     ttnn::DataType::FLOAT32, ttnn::Layout::TILE,
                                     device);
      auto [oc, fs] = ttnn::transformer::chunk_gated_delta_rule(
          dq, dk4, dv4, dg, db, args.scale, s0_sl,
          /*output_final_state=*/true, /*chunk_size=*/64,
          /*use_qk_l2norm=*/false, /*output_head_major=*/false);
      VT_CHECK(fs.has_value(),
               "tenstorrent gdn_decode: chunk_gated_delta_rule returned no final state");
      os.push_back(std::move(oc));  // [bn,1,Hv,Dv]
      fss.push_back(
          ttnn::permute(*fs, ttsl::SmallVector<int64_t>{0, 1, 3, 2}));  // [bn,Hv,Dv,Dk]
    }
    o = os.size() == 1 ? std::move(os[0]) : ttnn::concat(os, /*dim=*/0);
    S_new = fss.size() == 1 ? std::move(fss[0]) : ttnn::concat(fss, /*dim=*/0);
    o = ttnn::reshape(o, ttnn::Shape({bh, udv, 1}));  // [B,1,Hv,Dv] -> [B*Hv,Dv,1]
    S_new = ttnn::reshape(S_new, ttnn::Shape({bh, udv, udk}));
  } else {
    DecodeStepResult r =
        GdnDecodeStepComposed(S, dev_q, dev_k, dev_v, dev_g, dev_b, args.scale, bh,
                              udk, udv);
    o = std::move(r.o);
    S_new = std::move(r.state);
  }

  if (has_null) {
    // NULL rows: the oracle zeroes the out row and skips the state update.
    std::vector<float> valid(static_cast<size_t>(bh));
    for (int64_t b = 0; b < batch; ++b) {
      const float live = idxv[static_cast<size_t>(b)] >= 0 ? 1.0f : 0.0f;
      for (int64_t h = 0; h < hv; ++h) valid[static_cast<size_t>(b * hv + h)] = live;
    }
    o = ttnn::multiply(o, UploadTensor(std::move(valid), ttnn::Shape({bh, 1, 1}),
                                       ttnn::DataType::FLOAT32, ttnn::Layout::TILE,
                                       device));
  }

  CommitDeviceLogical2D(
      out,
      CaptureSafeReshape(o, ttnn::Shape({ub, uhv, udv})),
      static_cast<uint32_t>(batch * hv), udv);

  // rows2d in the SPLIT shadow geometry ([B*F, blk]) — same flat bytes as
  // [B, Hv*Dv*Dk]; the regroup across the head boundary is one exact
  // data-movement program with per-tile circular buffers (only when F > 1).
  ttnn::Tensor rows2d = CaptureSafeReshape(
      S_new, ttnn::Shape({static_cast<uint32_t>(ub * sf),
                          static_cast<uint32_t>((uhv * udv * udk) / sf)}));
  // bf16 STORAGE semantics (SupportsCompressedGdnState, cuda_backend.cu): a
  // bf16 ssm_state is "read/written in f32 registers" — the state a step
  // STORES rounds through bf16, and the next step reads those bits. The
  // device shadow is f32, so round it in place on commit (RNE typecast
  // round-trip, zero PCIe). With a persistent f32 shadow instead, a
  // full-mantissa state value would re-enter unrounded and drift from the
  // CUDA contract within a few steps — pinned by the bf16-state arm of the
  // kGdnDecode oracle test (per-step outs must match the f32 path with
  // host-side bf16 round-trips bit-for-bit).
  if (state.dtype == DType::kBF16) {
    rows2d = ttnn::typecast(ttnn::typecast(rows2d, ttnn::DataType::BFLOAT16),
                            ttnn::DataType::FLOAT32);
  }
  // Exact scatter (last-writer-wins on duplicate slots, NULL rows write
  // nothing) — the state bytes never round through tf32 on the way back.
  // In-region the block ids come from the warmed entry (the per-call
  // upload would be the enqueue_write the trace refuses; the entry's
  // content match plus the NULL refusal above make it the all-live form
  // ScatterRowsExact would build); eager keeps the per-call path, NULL
  // compaction included. Use ScatterRowsDevice for BOTH eager and capture:
  // the committed state must have the same properties in both passes, so
  // EnsureGdnCacheDevice takes the same branch in the next layer. The eager
  // warmup warms the untilize/indexed_fill programs that capture needs.
  //
  // TT-GDN-REGION-REPLAY: commit the state IN PLACE — into the SAME device
  // shadow the captured trace reads (ttnn::copy into the served buffer: the
  // preallocated-destination in-region write the RAC lane proves
  // capture-safe, tenstorrent_paged.cpp:826) — and publish THAT buffer as
  // the slot's shadow. The commit this replaces installed a FRESH device
  // tensor every step, so the address a captured trace baked at capture time
  // was freed by the very commit that produced it — the allocator recycled
  // the block — and every replay re-read that capture-time buffer (recycled
  // garbage: the anchor's first served replay wrote 248,320/248,320 zero
  // logits; the bf16 9B's coherent-wrong stream) and re-scattered the result
  // over the LIVE slot, corrupting the committed state process-permanently.
  // In place, the trace's next gather reads the state its own scatter
  // wrote: the recurrence advances at every replay. The identical op
  // sequence runs in the eager warmup and the captured pass (the W4
  // discipline), so the copy's program is program-cache-warm before the
  // capture; a pure copy keeps the eager values byte-identical. The shape
  // guard falls back to the replacing commit on any geometry the update did
  // not reproduce exactly.
  const auto same_spec = [](const ttnn::Tensor& a, const ttnn::Tensor& b) {
    if (a.dtype() != b.dtype() || a.layout() != b.layout()) return false;
    const auto sa = a.logical_shape(), sb = b.logical_shape();
    if (sa.rank() != sb.rank()) return false;
    for (uint32_t r = 0; r < sa.rank(); ++r)
      if (sa[r] != sb[r]) return false;
    return true;
  };
  ttnn::Tensor next = oh.has_value()
                           ? ScatterRowsDevice(cache2d, sentry.scatter_bid,
                                               rows2d, slots, hv * dv * dk, sf)
                           : rows2d;
  if (same_spec(next, cache2d)) {
    ttnn::copy(next, cache2d);
    CommitDeviceLogical2D(
        state, std::move(cache2d),
        static_cast<uint32_t>(oh.has_value() ? slots : ub),
        static_cast<uint32_t>(hv * dv * dk));
  } else {
    CommitDeviceLogical2D(
        state, std::move(next),
        static_cast<uint32_t>(oh.has_value() ? slots : ub),
        static_cast<uint32_t>(hv * dv * dk));
  }
}

// TT-GDN-REGION-REPLAY attribution instrument: the state-slot probe, the
// driver-side extension of the VT_TT_GDN_STATE_PROBE pattern to the REPLAY
// step. The in-kernel probe cannot print at replay (a replay runs no host
// code, and a download is the read a trace capture refuses), so the decode
// graph driver calls this BETWEEN steps — outside capture, with the trace
// merely instantiated — to checksum what a GDN state cache's CURRENT device
// shadow holds. The captured trace's state reads are baked to the address
// served at capture time; the run's log of this probe therefore answers the
// spec's discriminating question directly: slots that ADVANCE across replays
// print changing checksums, slots frozen at the capture-time values print one
// identical checksum at every replay (the stale-binding signature), and slots
// collapsed to recycled memory print garbage or zeros. Read-only; unset runs
// never reach the call site.
void GdnShadowProbe(const void* host_ptr, const char* what) {
  if (host_ptr == nullptr) return;
  if (tt_capture_active()) return;  // the read the trace refuses
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(const_cast<void*>(host_ptr));
  if (s == nullptr || !s->device_current || !s->device.has_value()) {
    std::fprintf(stderr, "[GDN-SHADOW-PROBE] %s no-device-shadow\n", what);
    return;
  }
  // The structural half of the probe lives in the tree, not the log: the
  // out-of-place state commits (CommitDeviceLogical2D / CommitConvTransposed)
  // install a FRESH device tensor every step, so the address a captured trace
  // baked at capture time is freed by the very commit that produced it. The
  // CONTENT columns below carry the observable: slots whose step-N checksum
  // was sane and whose step-N+1 replay produced zero logits were not read —
  // the trace read the freed pre-capture buffer instead.
  std::vector<float> v = s->device->to_vector<float>();
  uint64_t fnv = 1469598103934665603ULL;
  double acc = 0.0;
  double tmin = 0.0, tmax = 0.0;
  bool first = true;
  int64_t zeros = 0;
  for (float f : v) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    fnv = (fnv ^ bits) * 1099511628211ULL;
    acc += f > 0.0 ? f : -static_cast<double>(f);
    if (f == 0.0f) ++zeros;
    if (first || static_cast<double>(f) < tmin) tmin = f;
    if (first || static_cast<double>(f) > tmax) tmax = f;
    first = false;
  }
  std::fprintf(stderr,
               "[GDN-SHADOW-PROBE] %s rows=%ux%u n=%zu fnv=%016llx asum=%a "
               "tmin=%a tmax=%a zeros=%lld/%zu\n",
               what, s->dev_rows, s->dev_cols, v.size(),
               static_cast<unsigned long long>(fnv), acc, tmin, tmax,
               static_cast<long long>(zeros), v.size());
  std::fflush(stderr);
}

// kGdnStateGather (cpu_ops.cpp GdnStateGatherKernel): the indexed
// persistent-cache boundary for GDN mixed prefill. Exact-dim caches gather
// through the one-hot matmul on the device shadow (the cache never fully
// downloads); a WIDENED cache row (spec taps) takes the host path on the
// shadowed bytes — a spec-decode-only arm, counted as d2h traffic.
void GdnStateGatherKernel(Queue&, Tensor& working, const Tensor& cache,
                          const Tensor& state_idx, const Tensor* has_initial_state) {
  TT_OP_TRACE("GdnStateGather");
  const int64_t rows = state_idx.shape[0];
  if (rows == 0) return;
  const int64_t slots = cache.shape[0];
  const int64_t work_inner = working.shape[working.rank - 1];
  const int64_t cache_inner = cache.shape[cache.rank - 1];
  const int64_t work_row = working.Numel() / rows;
  const int64_t mid = work_row / work_inner;
  const int64_t cache_row = mid * cache_inner;
  const std::vector<int32_t> idxv = ReadIdxHost(state_idx, slots, "gdn_state_gather", false);
  bool has_his = false;
  std::vector<float> keepv(static_cast<size_t>(rows), 1.0f);
  if (has_initial_state != nullptr) {
    EnsureHost(*has_initial_state);
    has_his = true;
    for (int64_t r = 0; r < rows; ++r) {
      const bool live = has_initial_state->dtype == DType::kI8
                            ? has_initial_state->Ptr<int8_t>()[r] != 0
                            : has_initial_state->Ptr<int32_t>()[r] != 0;
      keepv[static_cast<size_t>(r)] = live ? 1.0f : 0.0f;
    }
  }

  if (cache_inner != work_inner) {
    // Widened row: per-channel LEADING work_inner columns with the physical
    // stride — host path on the shadowed bytes (spec-decode arm).
    EnsureHost(cache);
    GdnStateD2hBytes().fetch_add(static_cast<uint64_t>(cache.Numel()) * sizeof(float),
                                 std::memory_order_relaxed);
    for (int64_t r = 0; r < rows; ++r)
      for (int64_t m = 0; m < mid; ++m) {
        const float* src = cache.Ptr<float>() +
                           static_cast<int64_t>(idxv[static_cast<size_t>(r)]) * cache_row +
                           m * cache_inner;
        float* dst = working.Ptr<float>() + r * work_row + m * work_inner;
        const float mul = has_his ? keepv[static_cast<size_t>(r)] : 1.0f;
        for (int64_t e = 0; e < work_inner; ++e) dst[e] = mul * src[e];
      }
    CommitHost(working);
    return;
  }

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor cache2d = EnsureGdnCacheDevice(cache, slots, cache_row, device);
  // Split-shadow geometry: expand the (NULL-free) row indices to their
  // block-rows — contiguous, in order — and the has_initial_state mask the
  // same way, so every gathered block-row keeps its own keep factor.
  const uint32_t sgf =
      static_cast<uint32_t>(SplitFactor(cache_row, /*esz=*/4));
  std::vector<int32_t> idx_blk;
  const std::vector<int32_t>* idxp = &idxv;
  if (sgf > 1) {
    idx_blk.reserve(idxv.size() * static_cast<size_t>(sgf));
    for (int32_t ix : idxv)
      for (uint32_t j = 0; j < sgf; ++j)
        idx_blk.push_back(ix * static_cast<int32_t>(sgf) +
                          static_cast<int32_t>(j));  // BLOCK-row id, not slot id
    idxp = &idx_blk;
    std::vector<float> keep_exp;
    keep_exp.reserve(keepv.size() * static_cast<size_t>(sgf));
    for (float kv : keepv) keep_exp.insert(keep_exp.end(), sgf, kv);
    keepv.swap(keep_exp);
  }
  ttnn::Tensor gathered = GatherRowsExact(cache2d, *idxp,
                                          cache_row / sgf, device);  // exact, no NULLs
  if (has_his) {
    gathered = ttnn::multiply(
        gathered,
        UploadTensor(std::move(keepv),
                     ttnn::Shape({static_cast<uint32_t>(rows) * sgf, 1}),
                     ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device));
  }
  CommitDeviceLogical2D(working, std::move(gathered), static_cast<uint32_t>(rows),
                        static_cast<uint32_t>(work_row));
}

// kGdnStateScatter (cpu_ops.cpp GdnStateScatterKernel): the inverse indexed
// store into the persistent cache. Exact-dim caches scatter through the
// exact indexed_fill on the device shadow; unnamed rows keep their bytes and
// the LAST row wins a duplicate slot (the oracle's loop order). A widened row
// takes the host path (spec-decode arm, counted).
void GdnStateScatterKernel(Queue&, Tensor& cache, const Tensor& working,
                           const Tensor& state_idx) {
  TT_OP_TRACE("GdnStateScatter");
  const int64_t rows = state_idx.shape[0];
  if (rows == 0) return;
  const int64_t slots = cache.shape[0];
  const int64_t work_inner = working.shape[working.rank - 1];
  const int64_t cache_inner = cache.shape[cache.rank - 1];
  const int64_t work_row = working.Numel() / rows;
  const int64_t mid = work_row / work_inner;
  const int64_t cache_row = mid * cache_inner;
  const std::vector<int32_t> idxv = ReadIdxHost(state_idx, slots, "gdn_state_scatter", false);

  if (cache_inner != work_inner) {
    EnsureHost(working);
    EnsureHost(cache);
    GdnStateD2hBytes().fetch_add(static_cast<uint64_t>(cache.Numel()) * sizeof(float),
                                 std::memory_order_relaxed);
    for (int64_t r = 0; r < rows; ++r)
      for (int64_t m = 0; m < mid; ++m) {
        const float* src = working.Ptr<float>() + r * work_row + m * work_inner;
        float* dst = cache.Ptr<float>() +
                     static_cast<int64_t>(idxv[static_cast<size_t>(r)]) * cache_row +
                     m * cache_inner;
        for (int64_t e = 0; e < work_inner; ++e) dst[e] = src[e];
      }
    CommitHost(cache);
    return;
  }

  MeshDevice& device = SharedMeshDevice();
  const uint32_t ssf =
      static_cast<uint32_t>(SplitFactor(cache_row, /*esz=*/4));
  ttnn::Tensor cache2d = EnsureGdnCacheDevice(cache, slots, cache_row, device);
  // working rows: reuse a resident f32 shadow when its element count and
  // dims match, else upload (and leave it unregistered — transient rows).
  // The upload carries the SPLIT shadow geometry ([rows*F, blk]); the flat
  // host bytes are unchanged.
  bool rows_resident = false;
  ttnn::Tensor rows2d;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(working.data);
    if (s != nullptr && s->device_current && s->device.has_value() &&
        s->device->dtype() == ttnn::DataType::FLOAT32 &&
        s->dev_rows == rows && s->dev_cols == work_row) {
      rows2d = *s->device;
      rows_resident = true;
    }
  }
  if (!rows_resident) {
    EnsureHost(working);
    std::vector<float> wh(static_cast<size_t>(working.Numel()));
    for (int64_t i = 0; i < working.Numel(); ++i)
      wh[static_cast<size_t>(i)] = LoadElemF32(working, i);
    rows2d = UploadTensor(
        std::move(wh),
        ttnn::Shape({static_cast<uint32_t>(rows) * ssf,
                     static_cast<uint32_t>(work_row / ssf)}),
        ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
  }
  ttnn::Tensor newc = ScatterRowsExact(cache2d, idxv, rows2d, slots,
                                       cache_row, ssf, device);
  CommitDeviceLogical2D(cache, std::move(newc), static_cast<uint32_t>(slots),
                        static_cast<uint32_t>(cache_row));
}

// (W2 of the capture-warmup redesign removed the GDN-shadow
// snapshot/restore pair that lived here: the warmup's scatter commit IS
// the geometry the capture's first read serves, and restoring pre-warmup
// shadows only discarded the warmup's state update — the frozen GDN
// recurrence. ISSUE-LOCAL-01M3918KQ580Z3NHVRNXVF15FZ.)

// ---- BACKEND-TENSTORRENT-GDN W2: GDN state/conv-shadow traffic counters ----
// Readers for the tenstorrent_device.h API; the counters themselves live in
// tenstorrent_internal.h.
GdnShadowTraffic GetGdnShadowTraffic() {
  GdnShadowTraffic t;
  t.state_h2d_bytes = GdnStateH2dBytes().load(std::memory_order_relaxed);
  t.state_d2h_bytes = GdnStateD2hBytes().load(std::memory_order_relaxed);
  t.decode_steps = GdnDecodeSteps().load(std::memory_order_relaxed);
  return t;
}

void ResetGdnShadowTraffic() {
  GdnStateH2dBytes().store(0, std::memory_order_relaxed);
  GdnStateD2hBytes().store(0, std::memory_order_relaxed);
  GdnDecodeSteps().store(0, std::memory_order_relaxed);
}


// ---- GDN prefill op set (BACKEND-TENSTORRENT-GDN W1) -------------------------
// kL2Norm / kRmsNormGated / kCausalConv1dFwd / kGdnPrefill: the op chain the
// Qwen3.5-family GDN layer issues in prefill. The CPU f32 arm (cpu_ops.cpp
// GdnPrefillKernel / CausalConv1dFwdKernel / L2NormKernel / RmsNormGatedKernel)
// is the correctness oracle; the tt-metal substrate is the implementation.
// Device-composed: kL2Norm (square → row-sum → rsqrt → scale, bf16 tiles),
// kRmsNormGated (ttnn::rms_norm + silu/sigmoid gate eltwise), kGdnPrefill
// (ttnn::transformer::chunk_gated_delta_rule behind a varlen→dense adapter).
// Host-staged in W1: kCausalConv1dFwd (the varlen window build + rolling
// conv_state writeback is pure data movement at these shapes; the conv-state
// device shadow that would make a composed path win is W2's decode work).

// Rank-flexible rows view of a contiguous float tensor as [rows, cols] TILE
// bf16 on device: reuses a resident same-numel shadow without a host
// round-trip (the EnsureDevice2D residency win, reached from rank-3 GDN
// shapes [T,H,D]).
ttnn::Tensor DeviceRows(const Tensor& t, uint32_t rows, uint32_t cols,
                        MeshDevice& device) {
  if (t.rank == 2 && t.IsContiguous()) return EnsureDevice2D(t, device);
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(t.data);
    if (s != nullptr && s->device_current && s->device.has_value() &&
        static_cast<uint64_t>(s->dev_rows) * s->dev_cols ==
            static_cast<uint64_t>(rows) * cols) {
      // Same re-layout discipline as EnsureDevice2D's exact hit: round-trip
      // ROW_MAJOR so a non-tile-aligned row count cannot reinterpret the
      // owner's padded tile rows.
      ttnn::Tensor reshaped = ttnn::to_layout(
          ttnn::reshape(ttnn::to_layout(*s->device, ttnn::Layout::ROW_MAJOR),
                        ttnn::Shape({rows, cols})),
          s->device->layout());
      s->device = reshaped;
      s->dev_rows = rows;
      s->dev_cols = cols;
      return reshaped;
    }
  }
  EnsureHost(t);
  const auto host = ToHostF32(t);
  return UploadRows(host.data(), rows, cols, device);
}

// Upload an arbitrary-rank host f32 buffer as a device tensor of `dtype` /
// `layout` (from_vector handles tile padding of logical dims).
ttnn::Tensor UploadTensor(std::vector<float> host, const ttnn::Shape& shape,
                          ttnn::DataType dtype, ttnn::Layout layout,
                          MeshDevice& device) {
  return ttnn::Tensor::from_vector<float>(
      std::move(host),
      tt::tt_metal::TensorSpec(tt::tt_metal::Shape(shape),
                               tt::tt_metal::TensorLayout(
                                   dtype, tt::tt_metal::PageConfig(layout),
                                   tt::tt_metal::MemoryConfig{})),
      &device);
}


// kL2Norm: y = x * rsqrt(sum(x^2) + eps) over the last dim (cpu_ops.cpp
// L2NormKernel; gdn-semantics.md §4). GDN callers run it on q/k [T,H,D] rows.
// Device path: square → row sum → +eps → rsqrt → broadcast multiply, bf16
// tiles (same envelope as kRmsNorm).
void L2NormKernel(Queue&, Tensor& out, const Tensor& x, const L2NormArgs& args) {
  TT_OP_TRACE("L2Norm");
  VT_CHECK(x.rank == 2 || x.rank == 3,
           "tenstorrent kL2Norm: rank 2 or 3 required");
  VT_CHECK(out.rank == x.rank, "tenstorrent kL2Norm: out rank must match x");
  VT_CHECK(IsFloatDType(x.dtype) &&
               (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kL2Norm: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(),
           "tenstorrent kL2Norm: contiguous required");
  const uint32_t d = static_cast<uint32_t>(x.shape[x.rank - 1]);
  const uint32_t rows = static_cast<uint32_t>(x.Numel() / d);

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_x = DeviceRows(x, rows, d, device);
  ttnn::Tensor sq = ttnn::multiply(dev_x, dev_x);
  ttnn::Tensor s = ttnn::sum(sq, ttsl::SmallVector<int>{1}, true);
  ttnn::Tensor denom = ttnn::add(s, args.eps);
  ttnn::Tensor inv = ttnn::rsqrt(denom);
  ttnn::Tensor dev_y = ttnn::multiply(dev_x, inv);
  CommitDeviceLogical2D(out, std::move(dev_y), rows, d);
}

// kRmsNormGated: out = x * rsqrt(mean(x^2)+eps) * w * act(gate) with
// norm_before_gate=True semantics baked in (cpu_ops.cpp RmsNormGatedKernel;
// gdn-semantics.md §5). Device path reuses the kRmsNorm machinery
// (ttnn::rms_norm with the [1,D] affine upload) + a silu/sigmoid gate pass.
// The gate may be a padded-row rank-3 view of the merged qkvz z-slice: its
// rows are gathered honoring the token stride and uploaded compactly.
void RmsNormGatedKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& gate,
                        const Tensor& weight, const RmsNormGatedArgs& args) {
  TT_OP_TRACE("RmsNormGated");
  VT_CHECK((x.rank == 2 || x.rank == 3) && gate.rank == x.rank &&
               out.rank == x.rank && weight.rank == 1,
           "tenstorrent kRmsNormGated: x/gate/out rank-2 or rank-3, weight rank-1");
  VT_CHECK(IsFloatDType(x.dtype) && IsFloatDType(gate.dtype) &&
               IsFloatDType(weight.dtype) &&
               (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kRmsNormGated: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && weight.IsContiguous() && out.IsContiguous(),
           "tenstorrent kRmsNormGated: x/out/weight contiguous required");
  const uint32_t d = static_cast<uint32_t>(x.shape[x.rank - 1]);
  const uint32_t rows = static_cast<uint32_t>(x.Numel() / d);
  VT_CHECK(weight.shape[0] == d,
           "tenstorrent kRmsNormGated: weight size mismatch");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_x = DeviceRows(x, rows, d, device);
  ttnn::Tensor dev_w = EnsureAffine1D(weight, d, device);
  // The gate rides its device shadow in the captured arm (a committed
  // projection output, or a packed slice of one via ServeDeviceWindow).
  // EnsureHost on a committed gate is exactly the readback the trace
  // refuses, and the gathered re-upload below is a capture-time write —
  // both stay in the eager arm only.
  ttnn::Tensor dev_g;
  if (ServeDeviceShadowRaw(gate, rows, d, dev_g) ||
      ServeDeviceWindow(gate, rows, d, dev_g)) {
    dev_g = NormalizeDevF32Tile(std::move(dev_g), rows, d);
  } else {
    VT_CHECK(!tt_capture_active(),
             "tenstorrent kRmsNormGated: gate arrived without a servable "
             "device shadow during trace capture — the producer must commit "
             "device-side before the captured region");
    EnsureHost(gate);
    const int64_t gate_group = gate.rank == 3 ? gate.shape[1] : 1;
    const int64_t gate_outer = gate.stride[0];
    std::vector<float> gh(static_cast<size_t>(rows) * d);
    for (uint32_t i = 0; i < rows; ++i) {
      const int64_t gbase =
          (i / gate_group) * gate_outer + (i % gate_group) * d;
      for (uint32_t j = 0; j < d; ++j)
        gh[static_cast<size_t>(i) * d + j] = LoadElemF32(gate, gbase + j);
    }
    dev_g = UploadRows(gh.data(), rows, d, device);
  }
  ttnn::Tensor act =
      args.sigmoid_gate ? ttnn::sigmoid(dev_g) : ttnn::silu(dev_g);
  // Reshape act to [rows, d] to match rms_norm(dev_x) output. The gate
  // shadow's native geometry (e.g. [256, 6144] for APEX's gated-norm
  // activation) is preserved by NormalizeDevF32Tile (the L1 overflow fix:
  // a free reshape to [1, n] overflows L1, and view_device on ROW_MAJOR
  // with a changed last dim corrupts data). act is TILE layout here
  // (NormalizeDevF32Tile's final to_layout(TILE), preserved by the
  // elementwise sigmoid/silu), so CaptureSafeReshape's member view_device
  // takes the TILE safe path (tensor_ops.cpp:398: layout != ROW_MAJOR
  // short-circuits before the page_size rebuild that corrupts ROW_MAJOR).
  // Same numel (256*6144 == 12288*128), same tile count (1536), so the
  // reshape is a pure metadata view — no bytes move, no device program.
  {
    const auto target_shape = ttnn::Shape({rows, d});
    if (act.logical_shape() != target_shape) {
      act = CaptureSafeReshape(act, target_shape);
    }
  }
  if (std::getenv("VT_TT_SLOT_TRACE") != nullptr) {
    std::fprintf(stderr,
                 "[TT-RNG] rows=%u d=%u x=%s g=%s act=%s\n", rows, d,
                 DevShapeStr(dev_x).c_str(), DevShapeStr(dev_g).c_str(),
                 DevShapeStr(act).c_str());
    std::fflush(stderr);
  }
  ttnn::Tensor dev_y = ttnn::multiply(ttnn::rms_norm(dev_x, args.eps, dev_w), act);
  CommitDeviceLogical2D(out, std::move(dev_y), rows, d);
}

// kCausalConv1dFwd: depthwise causal conv over time with the rolling
// conv_state writeback (cpu_ops.cpp CausalConv1dFwdKernel; gdn-semantics.md
// §2). W1 HOST-STAGED: this backend's Alloc is host memory
// (tenstorrent_backend.cpp), so the scalar port below runs the exact oracle
// instruction order on the host bytes — outputs read the OLD state row
// (buffered), the new row carries the last K-1 RAW x tokens (left-shifted
// from the old state when T < K-1). A composed slice/concat+MAC path pays a
// full [T*K,C] window materialization to build what this loop streams; the
// conv-state device shadow that would flip that trade is W2's decode work.
void CausalConv1dFwdKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                           const Tensor* bias, Tensor& conv_state,
                           const Tensor& qsl, const Tensor& his,
                           const CausalConv1dArgs& args) {
  TT_OP_TRACE("CausalConv1dFwd");
  const int64_t total = x.shape[0], c_dim = x.shape[1], k = w.shape[1];
  const int64_t width = k - 1;
  const int64_t n = conv_state.shape[0];
  const int64_t x_rs = x.stride[0];
  EnsureHost(x);
  EnsureHost(w);
  if (bias != nullptr) EnsureHost(*bias);
  EnsureHost(conv_state);
  EnsureHost(qsl);
  EnsureHost(his);
  const int32_t* qslp = qsl.Ptr<int32_t>();
  VT_CHECK(qslp[0] == 0 && qslp[n] == total,
           "tenstorrent causal_conv1d_fwd: bad query_start_loc bounds");
  for (int64_t s = 0; s < n; ++s) {
    VT_CHECK(qslp[s + 1] >= qslp[s] && qslp[s] >= 0,
             "tenstorrent causal_conv1d_fwd: query_start_loc not monotonic");
  }
  // Per (sequence, channel): mirrors the oracle's row-chunked decomposition —
  // disjoint out columns / conv_state rows, so the loop order is free.
  std::vector<float> old_row(static_cast<size_t>(width));
  for (int64_t s = 0; s < n; ++s) {
    const bool init = his.dtype == DType::kI8 ? his.Ptr<int8_t>()[s] != 0
                                              : his.Ptr<int32_t>()[s] != 0;
    const int64_t begin = qslp[s], t_len = qslp[s + 1] - begin;
    for (int64_t c = 0; c < c_dim; ++c) {
      float* srow = conv_state.Ptr<float>() + (s * c_dim + c) * width;
      for (int64_t j = 0; j < width; ++j)
        old_row[static_cast<size_t>(j)] = srow[j];
      const float b = bias != nullptr ? LoadElemF32(*bias, c) : 0.0f;
      for (int64_t t = 0; t < t_len; ++t) {
        float acc = b;
        for (int64_t j = 0; j < k; ++j) {
          const int64_t ti = t - (k - 1 - j);  // token index of window[j]
          float v = 0.0f;
          if (ti >= 0) {
            v = LoadElemF32(x, (begin + ti) * x_rs + c);
          } else if (init) {
            v = old_row[static_cast<size_t>(width + ti)];  // state col (K-1)+(t-i)
          }
          acc += LoadElemF32(w, c * k + j) * v;
        }
        const float y = args.silu_activation
                            ? acc / (1.0f + std::exp(-acc))
                            : acc;
        StoreElemF32(out, (begin + t) * c_dim + c, y);
      }
      for (int64_t j = 0; j < width; ++j) {
        const int64_t tj = t_len - width + j;  // new state col j holds token tj
        float v = 0.0f;
        if (tj >= 0) {
          v = LoadElemF32(x, (begin + tj) * x_rs + c);
        } else if (init) {
          v = old_row[static_cast<size_t>(width + tj)];  // shifted old state
        }
        srow[j] = v;
      }
    }
  }
  CommitHost(out);
  CommitHost(conv_state);
}

// ==== W2: the decode set =====================================================
// kCausalConv1dUpdate / kGdnDecode (both state_idx forms) / kGdnStateGather /
// kGdnStateScatter. The decode state and the conv state live in DEVICE
// shadows keyed by the host pointer — the same BufferSlot residency every
// other op uses (the PagedKvShadow discipline, but the GDN caches keep the
// CALLER's layout on device so EnsureHostBytes stays byte-correct): a host
// write via Backend::Copy drops the shadow (MarkHostWritten), a host read
// downloads it once (EnsureHostBytes), and a decode step updates it in place
// with ZERO PCIe state traffic. Indexed forms gather/scatter rows on-device
// through exact 0/1 one-hot matmuls (0*x + v == v in f32), so the full cache
// never moves either.


// Upload a small u32 index vector as a device tensor (ROW_MAJOR — the
// indexed_fill batch_id contract; gather takes TILE, reshaped at the call).
ttnn::Tensor UploadIdxU32(std::vector<uint32_t> host, const ttnn::Shape& shape,
                          ttnn::Layout layout, MeshDevice& device) {
  return ttnn::Tensor::from_vector<uint32_t>(
      std::move(host),
      tt::tt_metal::TensorSpec(tt::tt_metal::Shape(shape),
                               tt::tt_metal::TensorLayout(
                                   ttnn::DataType::UINT32,
                                   tt::tt_metal::PageConfig(layout),
                                   tt::tt_metal::MemoryConfig{})),
      &device);
}

// EXACT row gather from a [slots, cols] cache: ttnn::gather is pure data
// movement, so the bytes land bit-identical (the one-hot matmul alternative
// rounds its inputs to tf32 on this hardware — measured 2.4e-4 — and breaks
// the conv/cache-I/O 1e-4 envelopes). idx<0 = NULL row: gathered as a ZERO
// row (index clamped to 0, then masked) — the decode/conv NULL semantics.
ttnn::Tensor GatherRowsExact(const ttnn::Tensor& cache2d,
                             const std::vector<int32_t>& idx, int64_t cols,
                             MeshDevice& device) {
  const int64_t rows = static_cast<int64_t>(idx.size());
  const ttnn::Layout lay = cache2d.layout();
  // gather demands TILE on both tensors; the layout conversion is a pure
  // copy (f32 bytes unchanged).
  ttnn::Tensor tin = lay == ttnn::Layout::TILE
                         ? cache2d
                         : ttnn::to_layout(cache2d, ttnn::Layout::TILE);
  bool has_null = false;
  std::vector<uint32_t> flat(static_cast<size_t>(rows) * cols);
  for (int64_t r = 0; r < rows; ++r) {
    const int32_t ix = idx[static_cast<size_t>(r)];
    if (ix < 0) has_null = true;
    const uint32_t u = ix < 0 ? 0u : static_cast<uint32_t>(ix);
    for (int64_t c = 0; c < cols; ++c)
      flat[static_cast<size_t>(r) * cols + c] = u;
  }
  ttnn::Tensor index = UploadIdxU32(
      std::move(flat),
      ttnn::Shape({static_cast<uint32_t>(rows), static_cast<uint32_t>(cols)}),
      ttnn::Layout::TILE, device);
  ttnn::Tensor out = ttnn::gather(tin, /*dim=*/0, index,
                                  /*sparse_grad=*/false, std::nullopt);
  if (has_null) {
    std::vector<float> valid(static_cast<size_t>(rows));
    for (int64_t r = 0; r < rows; ++r)
      valid[static_cast<size_t>(r)] = idx[static_cast<size_t>(r)] >= 0 ? 1.0f : 0.0f;
    out = ttnn::multiply(
        out, UploadTensor(std::move(valid),
                          ttnn::Shape({static_cast<uint32_t>(rows), 1}),
                          ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device));
  }
  return lay == ttnn::Layout::TILE ? out : ttnn::to_layout(out, lay);
}

// EXACT scatter of rows2d [rows*factor, cols/factor] into cache2d
// [slots*factor, cols/factor] (both shadows are born split by SplitFactor;
// the LOGICAL geometry both kernels serve is [rows, cols]) via indexed_fill
// (torch.index_copy_ semantics): pure data movement, unnamed slots keep
// their bytes, and a slot named by SEVERAL rows keeps the LAST row — the
// CPU oracle's sequential loop order. idx<0 = NULL row: writes nothing (its
// row is compacted away before the fill). Each live slot s expands into its
// `factor` block ids s*F+j, so source block-row (e, j) lands in slot
// idx[e]'s j-th block; relative entry order is preserved inside every block
// group, so per-(slot, block) last-of-duplicates wins exactly like the
// whole-row form.
// The scatter core on a PREBUILT [rows*factor] u32 block-id tensor — the
// in-region (captured) form, where the ids were warmed outside capture and
// the per-call upload would be the enqueue_write the trace refuses. `rows2d`
// may be either layout; the ROW_MAJOR staging view keeps the last dim
// (metadata-only rank change — see the geometry note below).
ttnn::Tensor ScatterRowsDevice(const ttnn::Tensor& cache2d,
                               const ttnn::Tensor& bid_dev,
                               const ttnn::Tensor& rows2d, int64_t slots,
                               int64_t cols, int64_t factor) {
  const uint32_t blk = static_cast<uint32_t>(cols / factor);
  const uint32_t unb = static_cast<uint32_t>(bid_dev.logical_shape()[0]);
  const ttnn::Tensor& src = rows2d;
  const ttnn::Layout lay = cache2d.layout();
  ttnn::Tensor cache_rm =
      lay == ttnn::Layout::ROW_MAJOR ? cache2d : ttnn::to_layout(cache2d, ttnn::Layout::ROW_MAJOR);
  ttnn::Tensor src_rm =
      lay == ttnn::Layout::ROW_MAJOR ? src : ttnn::to_layout(src, ttnn::Layout::ROW_MAJOR);
  ttnn::Tensor out4 = ttnn::indexed_fill(
      bid_dev,
      ttnn::experimental::view(
          cache_rm, ttnn::Shape({static_cast<uint32_t>(slots * factor), 1, 1, blk})),
      ttnn::experimental::view(src_rm, ttnn::Shape({unb, 1, 1, blk})),
      std::nullopt, /*dim=*/0);
  ttnn::Tensor out2 = ttnn::experimental::view(
      out4, ttnn::Shape({static_cast<uint32_t>(slots * factor), blk}));
  return lay == ttnn::Layout::ROW_MAJOR ? out2 : ttnn::to_layout(out2, lay);
}

ttnn::Tensor ScatterRowsExact(const ttnn::Tensor& cache2d,
                              const std::vector<int32_t>& idx,
                              const ttnn::Tensor& rows2d, int64_t slots,
                              int64_t cols, int64_t factor, MeshDevice& device) {
  const uint32_t blk = static_cast<uint32_t>(cols / factor);
  std::vector<uint32_t> bid;
  bid.reserve(idx.size());
  for (int32_t ix : idx)
    if (ix >= 0) bid.push_back(static_cast<uint32_t>(ix));
  if (bid.empty()) return cache2d;  // all NULL: cache untouched
  ttnn::Tensor src = rows2d;
  if (rows2d.layout() != cache2d.layout())
    src = ttnn::to_layout(rows2d, cache2d.layout());  // pure copy, f32 exact
  if (bid.size() != idx.size()) {
    // Compact the live rows through the same exact gather: row_of[e] is the
    // ORIGINAL row of the e-th live entry — expanded to its `factor`
    // block-rows, which sit contiguous and in order.
    std::vector<int32_t> row_of;
    row_of.reserve(bid.size());
    for (int64_t r = 0; r < static_cast<int64_t>(idx.size()); ++r)
      if (idx[static_cast<size_t>(r)] >= 0) row_of.push_back(static_cast<int32_t>(r));
    std::vector<int32_t> row_of_exp;
    row_of_exp.reserve(row_of.size() * static_cast<size_t>(factor));
    for (int32_t r : row_of)
      for (int64_t j = 0; j < factor; ++j)
        row_of_exp.push_back(static_cast<int32_t>(r * factor + j));
    src = GatherRowsExact(src, row_of_exp, blk, device);
  }
  // Expand each live slot into its `factor` block ids before the upload.
  std::vector<uint32_t> blocks;
  blocks.reserve(bid.size() * static_cast<size_t>(factor));
  for (uint32_t s : bid)
    for (int64_t j = 0; j < factor; ++j)
      blocks.push_back(s * static_cast<uint32_t>(factor) + static_cast<uint32_t>(j));
  const uint32_t unb = static_cast<uint32_t>(blocks.size());
  ttnn::Tensor bid_dev = UploadIdxU32(
      std::move(blocks), ttnn::Shape({unb}), ttnn::Layout::ROW_MAJOR, device);
  // indexed_fill wants rank-4 on dim 0, and the rank change must stay a
  // ZERO-COPY view, never ttnn::reshape: a TILE rank-4 reshape of
  // [rows, cols] pads the trailing 1-dims to the 32-wide tile (physical
  // x1024 — the Qwen3.5 GDN ssm cache asked for 32 GiB and OOM'd the P150),
  // and a ROW_MAJOR reshape launches a data-movement program whose circular
  // buffers scale with the tensor (4.3 MB > 1.5 MB L1 for the 4 GiB cache) —
  // both found by the W0 sweep (BACKEND-TENSTORRENT-QWEN35, runs 2-5). So:
  // convert to ROW_MAJOR once (to_layout — a pure copy, bytes unchanged),
  // then view rank-4. The views keep the LAST dim (the block width), which
  // is the metadata-only case that preserves flat order; a last-dim-changing
  // view would scramble the bank interleave (see SplitFactor above). The
  // staging per launch is 2 x blk x elem_size, inside L1 by construction.
  // Semantics are unchanged: pure data movement, unnamed slots keep their
  // bytes, last-of-duplicates wins (the oracle's loop order).
  return ScatterRowsDevice(cache2d, bid_dev, src, slots, cols, factor);
}


// The conv-side entry: scatter/gather ids, the scatter base, the roll masks.
// Refreshes eagerly when the idx content changed; refuses to refresh during
// capture (a content change in-region means a slot change the recapture
// cadence owns — the eager boundary lane re-warms after the graph reset).
GdnIdxCacheEntry GdnConvIdxEntry(const std::vector<int32_t>& idxv,
                                 uint32_t slots, uint32_t C,
                                 MeshDevice& device) {
  const uint32_t R = slots * C;
  const std::array<uint64_t, 3> key{slots, C, idxv.size()};
  std::lock_guard<std::mutex> g(GdnIdxCacheMutex());
  GdnIdxCacheEntry& e = GdnIdxCacheMap()[key];
  if (e.warmed_conv && e.idx == idxv) return e;  // copy under the lock
  VT_CHECK(!tt_capture_active(),
           "tenstorrent causal_conv1d_update: state-slot indices changed "
           "during trace capture — decode slots must be stable across a "
           "sequence's steps; the recapture cadence owns a slot change "
           "(reset the graph, then re-warm on the eager boundary step)");
  GdnIdxCacheEntry n;
  n.idx = idxv;
  n.warmed_conv = true;
  std::vector<uint32_t> ids;
  ids.reserve(idxv.size());
  for (int32_t ix : idxv) ids.push_back(static_cast<uint32_t>(ix));
  n.idx_scatter = UploadIdxU32(std::move(ids), ttnn::Shape({static_cast<uint32_t>(idxv.size())}),
                               ttnn::Layout::ROW_MAJOR, device);
  std::vector<uint32_t> rows(static_cast<size_t>(idxv.size()) * C);
  for (size_t i = 0; i < idxv.size(); ++i)
    for (uint32_t c = 0; c < C; ++c)
      rows[i * C + c] = static_cast<uint32_t>(idxv[i]);
  n.idx_rows = UploadIdxU32(std::move(rows), ttnn::Shape({static_cast<uint32_t>(idxv.size()), C}),
                            ttnn::Layout::TILE, device);
  n.zeros_rc = ttnn::zeros(tt::tt_metal::Shape({slots, C}),
                           ttnn::DataType::FLOAT32, ttnn::Layout::ROW_MAJOR,
                           std::ref(device));
  std::vector<float> mv(R, 0.0f), mk(R, 1.0f);
  for (int32_t s : idxv)
    if (s >= 0)
      for (uint32_t c = 0; c < C; ++c) {
        mv[static_cast<size_t>(s) * C + c] = 1.0f;
        mk[static_cast<size_t>(s) * C + c] = 0.0f;
      }
  n.mv = UploadTensor(std::move(mv), ttnn::Shape({1, R}),
                      ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
  n.mk = UploadTensor(std::move(mk), ttnn::Shape({1, R}),
                      ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
  e = std::move(n);
  return e;
}

// Serve the tensor's current device shadow without touching host bytes — the
// capture-pure read (an EnsureHost here would enqueue the readback the trace
// refuses). Returns false when no current shadow exists; the caller then
// takes its eager staging path, and a capture-time miss is refused there.
// `rows`/`cols` are the caller's logical 2D geometry; the shadow must match
// that volume (the recorded commit geometry is the producer's logical shape,
// and a mismatch means a shape the device path has no evidence for).
bool ServeDeviceShadowRaw(const Tensor& t, uint32_t rows, uint32_t cols,
                          ttnn::Tensor& out) {
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(t.data);
  if (s == nullptr || !s->device_current || !s->device.has_value()) return false;
  // Device-authoritative only: the last tracked write must have been a
  // COMMIT (host bytes known-stale), not a plain stage. A both-current slot
  // says nothing about the device bytes matching the buffer's latest value —
  // a raw backend Copy (the op tests' per-step input refresh; any
  // non-tracked producer) bypasses slot tracking, and serving such a shadow
  // reads the previous tenant's bytes.
  if (s->host_current) return false;
  if (static_cast<uint64_t>(s->dev_rows) * s->dev_cols !=
      static_cast<uint64_t>(rows) * cols)
    return false;
  if (s->device->dtype() != ttnn::DataType::FLOAT32 &&
      s->device->dtype() != ttnn::DataType::BFLOAT16)
    return false;  // the device path covers the production activation dtypes
  out = *s->device;
  return true;
}


// Serve the activation as an f32 TILE device tensor for the decode step: the
// current device shadow when the producer committed one (the production
// driver commits its outputs device-side in every phase — the traffic
// counters prove no per-token round-trip), reshaped/typecast in-region when
// the shadow's split or dtype differs (both pure device ops, exact bytes;
// TILE is what every consumer at this pin demands — a non-TILE shadow is
// converted, a pure copy); else the eager arm's existing host staging, which
// a capture-time arrival refuses by name (an EnsureHost here would enqueue
// exactly the readback the trace forbids).
ttnn::Tensor ServeActF32(const Tensor& t, uint32_t rows, uint32_t cols,
                         const char* what, MeshDevice& device) {
  ttnn::Tensor raw;
  if (ServeDeviceShadowRaw(t, rows, cols, raw)) {
    const auto ls = raw.logical_shape();
    if (ls.rank() != 2 || ls[0] != rows || ls[1] != cols) {
      if (tt_capture_active() && ls.rank() == 2 &&
          ls[0] * ls[1] == static_cast<uint32_t>(rows) * cols) {
        // CaptureSafeReshape: member reshape (pure view, no device program)
        raw = CaptureSafeReshape(raw, ttnn::Shape({rows, cols}));
      } else {
        raw = CaptureSafeReshape(raw, ttnn::Shape({rows, cols}));
      }
    }
    if (raw.dtype() != ttnn::DataType::FLOAT32)
      raw = ttnn::typecast(raw, ttnn::DataType::FLOAT32);
    if (raw.layout() != ttnn::Layout::TILE)
      raw = ttnn::to_layout(raw, ttnn::Layout::TILE);
    return raw;
  }
  // The exact-shadow serve demands owner-numel equality, so an interior
  // window of a packed producer output (the q/k expand row views at a batch
  // the owner padded wider) falls through here. ServeDeviceWindow splits the
  // interior pointer offset into the owner's row/column geometry and rides a
  // device column slice — the same serve ServePostConvAB takes. Without it
  // the host fallback ran EnsureHost on the OWNER slot and downloaded the
  // whole plane against the window's numel (ISSUE-LOCAL-01M2E5F69CMWDERKXG32YY9P8N:
  // EnsureHost 1x16x128 vs dev 1024x128, vllm-bench chunked decode).
  if (ServeDeviceWindow(t, rows, cols, raw)) return raw;
  VT_CHECK(!tt_capture_active(),
           std::string("tenstorrent gdn_decode: ") + what +
               " arrived without a current device shadow during trace "
               "capture — the captured arm reads device-resident "
               "activations only");
  EnsureHost(t);
  std::vector<float> h(static_cast<size_t>(rows) * cols);
  for (uint64_t i = 0; i < static_cast<uint64_t>(rows) * cols; ++i)
    h[static_cast<size_t>(i)] = LoadElemF32(t, static_cast<int64_t>(i));
  return UploadTensor(std::move(h), ttnn::Shape({rows, cols}),
                      ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
}


// Both serve paths normalize through here. The ROW_MAJOR round-trip is the
// load-bearing step: a committed shadow can carry any logical shape the
// producer chose, a TILE reshape of such a tensor is not a safe reshape
// input at this pin (the AttnQkNormRopeGate doctrine), and a column slice
// of a TILE owner keeps the owner's tile padding — every one of those feeds
// a downstream elementwise op a misaligned physical layout (the prefill
// "Invalid subtile broadcast type" fatal). Materializing in ROW_MAJOR gives
// a fresh contiguous buffer; the elementwise chain on it is shape-agnostic,
// and the final to_layout(TILE) rebuilds the padding from the tensor's own
// (served) geometry. A rank-2 shadow keeps its native logical shape — the
// callers commit that served geometry (see NormalizeDevF32Tile's comment).
ttnn::Tensor NormalizeDevF32Tile(ttnn::Tensor x, uint32_t rows, uint32_t cols) {
  x = ttnn::to_layout(x, ttnn::Layout::ROW_MAJOR);
  const auto ls = x.logical_shape();
  if (ls.rank() != 2 || ls[0] != rows || ls[1] != cols) {
    // ISSUE-LOCAL-01M2K8WWA0X09VJF55NTS16VTV: a producer shadow natively
    // shaped [T, W] (the APEX gated-norm activation [128, 6144]) must not
    // take the free ttnn::reshape to (1, n) — reshape_rm's statically
    // allocated staging CBs scale with the row width, and the 3,145,728 B
    // f32 row is 2x the L1 cap (fatal at program allocation). The
    // metadata view cannot serve it either: view_device on ROW_MAJOR with
    // a CHANGED last dim rebuilds the buffer page mapping and the data
    // comes back wrong (the focused cast test's bit-exact memcmp caught
    // it) — which is why upstream routes this case through reshape_rm.
    // The elementwise ops below (typecast, to_layout(TILE)) are
    // shape-agnostic on the contiguous buffer, so a rank-2 shadow runs at
    // its native geometry and the CALLER commits the served geometry (the
    // #2282 served-geometry doctrine). The free reshape stays for the
    // non-rank-2 cases the elementwise chain cannot carry as-is.
    if (ls.rank() != 2) x = ttnn::reshape(x, ttnn::Shape({rows, cols}));
  }
  if (x.dtype() != ttnn::DataType::FLOAT32)
    x = ttnn::typecast(x, ttnn::DataType::FLOAT32);
  return ttnn::to_layout(x, ttnn::Layout::TILE);
}

// Serve an interior window of a device-authoritative OWNER slot. araw/braw
// are interior column slices of ONE packed producer output (merged-BA,
// VT_GDN_MERGED_BA default-on since #1168; qwen3_5.cpp ProjectGdnBA hands
// strided row views into the BA matmul result), and the gated-norm gate leg
// can be the z slice of a fused projection. FindSlot resolves an interior
// pointer to the owner slot; the element offset splits into a row and a
// column of the owner's logical geometry, and the window rides a device
// column slice — the same slice form GdnPostConvKernel already runs on
// dev_conv. A stride that does not walk the owner's rows exactly, a window
// past the owner's bounds, or a byte offset that is not a whole number of
// device elements is not served (the caller refuses under capture). Both
// this and the exact ServeDeviceShadowRaw path normalize to an f32 TILE
// [rows, cols] via `normalize` so a served shadow never changes the
// arithmetic the eager host staging defined.
bool ServeDeviceWindow(const Tensor& t, uint32_t rows, uint32_t cols,
                       ttnn::Tensor& out) {
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(t.data);
  if (s == nullptr || !s->device_current || s->host_current || !s->device)
    return false;
  if (s->device->dtype() != ttnn::DataType::FLOAT32 &&
      s->device->dtype() != ttnn::DataType::BFLOAT16)
    return false;
  const int64_t elem_bytes =
      s->device->dtype() == ttnn::DataType::FLOAT32 ? 4 : 2;
  // The window's own geometry: rank-2 views stride [dev_cols, 1] by
  // construction; higher-rank views are accepted only when CONTIGUOUS
  // row-major (every stride the product of the inner extents), which makes
  // them a plain flattened [rows*cols] span — the v/g/beta expands of the
  // decode step arrive as rank-3 [1, bh, ud] views of the packed producer
  // output (the vllm-bench chunked path; ISSUE-LOCAL-01M2E5F69CMWDERKXG32YY9P8N).
  // A non-contiguous multi-rank view is not served.
  const int64_t dev_cols = static_cast<int64_t>(s->dev_cols);
  if (t.rank == 2) {
    if (t.stride[0] != dev_cols || t.stride[1] != 1) return false;
  } else {
    if (t.shape[t.rank - 1] != cols) return false;
    int64_t expect = 1;
    for (int i = t.rank - 1; i >= 0; --i) {
      if (t.shape[i] != 0 && t.stride[i] != expect) return false;
      expect *= t.shape[i];
    }
    if (rows * cols != expect) return false;
  }
  const int64_t delta = static_cast<const char*>(t.data) -
                        static_cast<const char*>(s->host);
  if (delta < 0 || delta % elem_bytes != 0) return false;
  const int64_t off = delta / elem_bytes;
  // The slot's registered allocation must COVER the whole window (the slot
  // map is never erased, #1486 — a freed tensor's range can host a later,
  // unrelated allocation; the vllm-bench prefill buffers hit exactly that).
  if (delta + rows * cols * elem_bytes >
      static_cast<int64_t>(s->bytes))
    return false;
  const int64_t off_row = off / dev_cols, off_col = off % dev_cols;
  if (off_row + rows > s->dev_rows || off_col + cols > dev_cols)
    return false;
  ttnn::Tensor win = ttnn::slice(
      *s->device, ttsl::SmallVector<uint32_t>{static_cast<uint32_t>(off_row),
                                              static_cast<uint32_t>(off_col)},
      ttsl::SmallVector<uint32_t>{static_cast<uint32_t>(off_row + rows),
                                  static_cast<uint32_t>(off_col + cols)},
      ttsl::SmallVector<uint32_t>{1, 1});
  if (std::getenv("VT_TT_SLOT_TRACE") != nullptr) {
    std::fprintf(stderr, "[TT-WINSERVE] serve %ux%u off=%" PRId64
                         " owner=%s\n",
                 rows, cols, off, DevShapeStr(*s->device).c_str());
    std::fflush(stderr);
  }
  out = NormalizeDevF32Tile(std::move(win), rows, cols);
  return true;
}


// Serve the kGdnPostConv a/b operand: exact shadow first, then the packed
// owner window. Returns false when neither exists (the caller then stages
// from host in the eager arm and refuses under capture).
bool ServePostConvAB(const Tensor& t, uint32_t rows, uint32_t cols,
                     ttnn::Tensor& out) {
  ttnn::Tensor raw;
  if (ServeDeviceShadowRaw(t, rows, cols, raw)) {
    out = NormalizeDevF32Tile(std::move(raw), rows, cols);
    return true;
  }
  return ServeDeviceWindow(t, rows, cols, out);
}

ttnn::Tensor CachedTile(const void* owner, uint64_t g0, uint64_t g1, uint64_t g2,
                        const std::function<std::vector<float>()>& build,
                        const ttnn::Shape& shape, MeshDevice& device) {
  const std::array<uint64_t, 4> key{reinterpret_cast<uintptr_t>(owner), g0, g1, g2};
  std::lock_guard<std::mutex> g(ConvTileCacheMutex());
  auto it = ConvTileCacheMap().find(key);
  if (it != ConvTileCacheMap().end()) {
    // The content identity below is the point of `host` — the key is the
    // OWNER POINTER, and a test allocator hands the next case the same
    // address with different weight bytes (the rope cos/sin discipline).
    // Rebuild the host tile every call (the eager arm always did) and
    // re-upload only when the bytes moved; in-region they are the model
    // constants the warm step baked — match, no upload, no PCIe.
    std::vector<float> now = build();
    if (now == it->second.host) return it->second.dev;
    VT_CHECK(!tt_capture_active(),
             "tenstorrent constant tile: content changed during trace "
             "capture — the warm step baked different bytes than the "
             "captured region reads");
    it->second.host = std::move(now);
    it->second.dev = UploadTensor(it->second.host, shape,
                                  ttnn::DataType::FLOAT32, ttnn::Layout::TILE,
                                  device);
    return it->second.dev;
  }
  VT_CHECK(!tt_capture_active(),
           "tenstorrent constant tile not warmed — the cold step must build "
           "it before the captured region");
  ConvTileEntry e;
  e.host = build();
  e.dev = UploadTensor(e.host, shape, ttnn::DataType::FLOAT32,
                       ttnn::Layout::TILE, device);
  ConvTileCacheMap().emplace(key, std::move(e));
  return ConvTileCacheMap()[key].dev;
}


// The per-(token,head) repeat index for the rope cos|sin expansion:
// rows[(i*H + h)*half + j] = i. Content is a pure function of the key, so
// no host bytes are kept; the key's (0, 3, ...) prefix cannot collide with
// the scratch rows ((0, 1/2, ...)) or the head map (batch >= 1).
ttnn::Tensor CachedRepeatIdx(uint64_t t, uint64_t heads, uint64_t half,
                             MeshDevice& device) {
  const std::array<uint64_t, 4> key{0, 3, t * heads, half};
  std::lock_guard<std::mutex> g(ConvTileCacheMutex());
  auto it = ConvTileCacheMap().find(key);
  if (it != ConvTileCacheMap().end()) return it->second.dev;
  VT_CHECK(!tt_capture_active(),
           "tenstorrent attn rope: repeat index not warmed — the cold step "
           "must build it before the captured region");
  const uint32_t rows_n = static_cast<uint32_t>(t * heads);
  std::vector<uint32_t> rows(static_cast<size_t>(rows_n) * half);
  for (uint64_t r = 0; r < t * heads; ++r)
    for (uint64_t j = 0; j < half; ++j)
      rows[static_cast<size_t>(r) * half + j] =
          static_cast<uint32_t>(r / heads);
  ConvTileEntry e;
  e.dev = UploadIdxU32(
      std::move(rows),
      ttnn::Shape({rows_n, static_cast<uint32_t>(half)}),
      ttnn::Layout::TILE, device);
  ConvTileCacheMap().emplace(key, std::move(e));
  return ConvTileCacheMap()[key].dev;
}


ttnn::Tensor CachedAccBase(uint32_t R, MeshDevice& device) {
  const std::array<uint64_t, 4> key{0, 2, R, 0};
  std::lock_guard<std::mutex> g(ConvTileCacheMutex());
  auto it = ConvTileCacheMap().find(key);
  if (it != ConvTileCacheMap().end()) return it->second.dev;
  VT_CHECK(!tt_capture_active(),
           "tenstorrent causal_conv1d_update: accumulation base not warmed — "
           "the cold step must build it before the captured region");
  ConvTileEntry e;
  e.dev = ttnn::zeros(tt::tt_metal::Shape({1, R}), ttnn::DataType::FLOAT32,
                      ttnn::Layout::TILE, std::ref(device));
  ConvTileCacheMap().emplace(key, std::move(e));
  return ConvTileCacheMap()[key].dev;
}



// Read the tiny i32 index metadata on the host (NOT state traffic) and refuse
// out-of-range slots by name — the CPU oracle dereferences blindly, the TT
// kernel owns a one-hot and must not silently read a wrong row. The decode
// ops allow idx<0 (the NULL slot), gather/scatter refuse it like the oracle.
std::vector<int32_t> ReadIdxHost(const Tensor& idx, int64_t slots, const char* op,
                                 bool allow_null) {
  EnsureHost(idx);
  const int32_t* p = idx.Ptr<int32_t>();
  std::vector<int32_t> v(p, p + idx.shape[0]);
  for (int32_t ix : v)
    VT_CHECK(ix < slots && (allow_null || ix >= 0),
             std::string("tenstorrent ") + op +
                 ": state index out of range");
  return v;
}


// ---- Conv-state transposed shadow (BACKEND-TENSTORRENT-GDN W2) --------------
// ttnn movement exactness at this pin is dim-0 only (W2 micro-probe evidence):
// slice/concat/gather/indexed_fill on ROWS are byte-exact, while any
// last-dim (width) slice/concat at sub-tile offsets is broken or tf32-rounds.
// The conv roll shifts the window along the width axis, so the shadow is
// stored TRANSPOSED: [sl+1, slots*C] time-major (row j = timestep j of every
// (slot, channel); row sl = scratch for the incoming x). Every per-step op is
// then row-based and byte-exact: x lands via indexed_fill, the window+scratch
// come out via one gather with CONSTANT indices, the MAC is exact multiply +
// sequential single-row adds (bit-identical to the CPU oracle's accumulation
// order), and the roll is one gather whose index is the same rotation every
// step (rows [1..width-1, scratch, tail..., scratch]).
//
// Constant gather indices are cached per geometry; uploads cross PCIe only
// for x / w / bias / tiny per-step masks — never the state.

struct ConvGatherIdxCache {
  std::mutex mu;
  // key: (kind, R = slots*C, width, sl) -> u32 index tensor [rows, R] TILE.
  // `kind` separates the mac and roll lists — they share every geometry
  // field whenever state_len == width (rows == width+1 == sl+1), and the
  // roll list differs from the mac list exactly then.
  std::map<std::array<uint32_t, 4>, ttnn::Tensor> cache;
};

ConvGatherIdxCache& ConvGatherIdxCacheInstance() {
  static ConvGatherIdxCache* inst = new ConvGatherIdxCache(); // never destroyed (#1486)
  return *inst;
}

ttnn::Tensor ConvGatherIdx(const std::vector<uint32_t>& rows_of_src, uint32_t R,
                           uint32_t width, uint32_t sl, MeshDevice& device,
                           uint32_t kind) {
  ConvGatherIdxCache& c = ConvGatherIdxCacheInstance();
  std::lock_guard<std::mutex> g(c.mu);
  const std::array<uint32_t, 4> key{kind, R, width, sl};
  auto it = c.cache.find(key);
  if (it != c.cache.end()) return it->second;
  // The captured region serves the entry the cold step warmed; a miss
  // in-region would upload (enqueue_write) and the trace refuses it.
  VT_CHECK(!tt_capture_active(),
           "tenstorrent: conv gather-index cache miss during capture — the "
           "cold step must warm the constant index first");
  // The full [rows, R] broadcast index gather demands (same as
  // GatherRowsExact); both callers pass the list that is constant for
  // (kind, geometry), so first-writer-wins stores the right tensor.
  std::vector<uint32_t> flat(rows_of_src.size() * R);
  for (size_t r = 0; r < rows_of_src.size(); ++r)
    for (uint32_t j = 0; j < R; ++j)
      flat[r * R + j] = rows_of_src[r];
  ttnn::Tensor idx = UploadIdxU32(
      std::move(flat), ttnn::Shape({static_cast<uint32_t>(rows_of_src.size()), R}),
      ttnn::Layout::TILE, device);
  // First-caller-wins under the same geometry key: both callers pass the
  // constant list for that geometry, so the stored tensor is correct.
  if (!c.cache.emplace(key, idx).second) {
    idx = c.cache[key];
  }
  return idx;
}

// Ensure the conv-state shadow in TRANSPOSED [sl+1, R] TILE f32 form. Uploads
// (counted) only when missing/stale/wrong geometry. The scratch row is built
// on device (zeros + row concat — exact), so exactly `state bytes` cross h2d.
ttnn::Tensor EnsureConvStateTransposed(const Tensor& t, uint32_t slots,
                                       uint32_t Cc, uint32_t sl,
                                       MeshDevice& device) {
  const uint32_t R = slots * Cc;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(t.data);
    if (s != nullptr && s->device_current && s->device.has_value() &&
        s->conv_transposed && s->conv_slots == slots && s->conv_c == Cc &&
        s->conv_sl == sl && s->device->dtype() == ttnn::DataType::FLOAT32 &&
        s->device->layout() == ttnn::Layout::TILE &&
        s->device->logical_shape()[0] == sl + 1 && s->device->logical_shape()[1] == R) {
      return *s->device;
    }
  }
  // (Reachable only on the cold arm — the fast path above serves the warmed
  // shadow in-region. A capture-time miss must refuse BEFORE the EnsureHost:
  // across a prefill role transition the slot is device-current host-stale,
  // and the download would be the trace-refused read that aborts the process
  // instead of naming the condition. The driver's ConvShadowServeable query
  // keeps captures off this path; this CHECK is the backstop.)
  VT_CHECK(!tt_capture_active(),
           "tenstorrent: conv-state transposed shadow not warmed during "
           "capture — the cold step must establish it first");
  EnsureHost(const_cast<Tensor&>(t));  // host truth (oracle [slots, C, sl] order)
  const int64_t numel = static_cast<int64_t>(slots) * Cc * sl;
  VT_CHECK(t.Numel() == numel, "tenstorrent: conv state numel mismatch");
  std::vector<float> host(static_cast<size_t>(numel));
  for (int64_t i = 0; i < numel; ++i) host[static_cast<size_t>(i)] = LoadElemF32(t, i);
  std::vector<float> trans(static_cast<size_t>(numel));
  for (uint32_t s = 0; s < slots; ++s)
    for (uint32_t c = 0; c < Cc; ++c)
      for (uint32_t j = 0; j < sl; ++j)
        trans[static_cast<size_t>(j) * R + s * Cc + c] =
            host[static_cast<size_t>((s * Cc + c) * sl + j)];
  GdnStateH2dBytes().fetch_add(static_cast<uint64_t>(numel) * sizeof(float),
                               std::memory_order_relaxed);
  ttnn::Tensor body = UploadTensor(
      std::move(trans), ttnn::Shape({sl, R}), ttnn::DataType::FLOAT32,
      ttnn::Layout::TILE, device);
  ttnn::Tensor scratch = ttnn::zeros(tt::tt_metal::Shape({1, R}),
                                     ttnn::DataType::FLOAT32, ttnn::Layout::TILE,
                                     std::ref(device));
  ttnn::Tensor dev = ttnn::concat(std::vector<ttnn::Tensor>{body, scratch}, 0);
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(t.data);
    if (s != nullptr) {
      s->device = dev;
      s->dev_rows = sl + 1;
      s->dev_cols = R;
      s->device_current = true;
      s->host_current = true;
      s->device_reserved = false;  // real bytes staged — reservation spent
      s->conv_transposed = true;
      s->conv_slots = slots;
      s->conv_c = Cc;
      s->conv_sl = sl;
    }
  }
  return dev;
}

// Publish a new transposed shadow as the conv state's current value.
void CommitConvTransposed(Tensor& state, ttnn::Tensor dev, uint32_t slots,
                          uint32_t Cc, uint32_t sl) {
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(state.data);
  if (s == nullptr) {
    // Untracked buffer: materialize the transposition on host directly.
    std::vector<float> v = dev.to_vector<float>();
    // W3 #2201: the readback is a real device→host download of the whole
    // [sl+1, R] shadow — count it like every other GDN state download.
    GdnStateD2hBytes().fetch_add(static_cast<uint64_t>(v.size()) * sizeof(float),
                                 std::memory_order_relaxed);
    const uint32_t R = slots * Cc;
    for (uint32_t sc = 0; sc < slots * Cc; ++sc)
      for (uint32_t j = 0; j < sl; ++j)
        StoreElemF32(state, static_cast<int64_t>(sc) * sl + j,
                     v[static_cast<size_t>(j) * R + sc]);
    return;
  }
  s->device = std::move(dev);
  s->dev_rows = sl + 1;
  s->dev_cols = slots * Cc;
  s->device_current = true;
  s->host_current = false;
  s->conv_transposed = true;
  s->conv_slots = slots;
  s->conv_c = Cc;
  s->conv_sl = sl;
  s->device_reserved = false;  // real bytes committed — the reservation is spent
}

// kCausalConv1dUpdate (§3, seqlen==1 read-old-then-roll; cpu_ops.cpp
// CausalConv1dUpdateKernel at :1341). Transposed-shadow device path: every op
// is a proven-exact primitive (micro-probe table in the W2 evidence): x is
// written into the scratch row by indexed_fill, the [window | x] MAC source
// is one constant-index gather, the MAC multiplies exactly and accumulates by
// sequential single-row adds in the ORACLE's order (bias, taps 0..width-1,
// x tap last) so out is bit-identical up to silu, and the roll is one
// constant-index gather (shift rows, scratch in at width-1, tail identity).
// The indexed form masks with exact 0/1 multiplies: NULL rows (idx<0) keep
// both the out row's previous bytes and the cache row untouched (ops.h NULL
// skip), and non-batch cache columns never move. The state never crosses
// PCIe per step: only x / w / bias / the tiny 0/1 masks upload.
void CausalConv1dUpdateKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                              const Tensor* bias, Tensor& conv_state,
                              const Tensor* conv_state_indices,
                              const CausalConv1dArgs& args) {
  TT_OP_TRACE("CausalConv1dUpdate");
  GdnDecodeSteps().fetch_add(1, std::memory_order_relaxed);
  const int64_t batch = x.shape[0], c_dim = x.shape[1], k = w.shape[1];
  const int64_t width = k - 1;
  const int64_t state_len = conv_state.shape[2];
  const int64_t slots = conv_state.shape[0];
  const int64_t x_rs = x.stride[0];
  if (batch == 0) return;  // empty batch: no-op, cache as-is
  std::vector<int32_t> idxv;
  const bool indexed = conv_state_indices != nullptr;
  if (indexed) {
    idxv = ReadIdxHost(*conv_state_indices, slots, "causal_conv1d_update", true);
  } else {
    idxv.resize(static_cast<size_t>(batch));
    for (int64_t b = 0; b < batch; ++b) idxv[static_cast<size_t>(b)] = static_cast<int32_t>(b);
  }

  bool has_null = false;
  for (int32_t s : idxv)
    if (s < 0) has_null = true;
  // The captured arm is the pure-decode indexed path: every batch row names a
  // live slot, so the NULL-row arms (keep the previous out bytes, skip the
  // state row) never engage in-region. NULL handling stays on the eager path
  // (pinned by the op-level oracle suite); a capture with a NULL row is a
  // refused shape, not a silent skip.
  VT_CHECK(!has_null || !tt_capture_active(),
           "tenstorrent causal_conv1d_update: NULL state rows during trace "
           "capture — the captured arm is the pure-decode indexed path");

  MeshDevice& device = SharedMeshDevice();
  const uint32_t ub = static_cast<uint32_t>(batch), uc = static_cast<uint32_t>(c_dim),
                  uw = static_cast<uint32_t>(width), usl = static_cast<uint32_t>(state_len),
                  uslots = static_cast<uint32_t>(slots), R = uslots * uc,
                  utaps = uw + 1;

  // Weights are immutable across steps: the [taps, R] tap tile and the [1, R]
  // bias tile are step-constant caches built once on the cold step and served
  // in-region (a capture-time upload is an enqueue_write the trace refuses).
  // The host weight reads run only inside the cold-step builders.
  ttnn::Tensor Wt = CachedTile(
      w.data, utaps, R, 0,
      [&w, utaps, R, uslots, uc, k] {
        EnsureHost(w);
        std::vector<float> wh(static_cast<size_t>(utaps) * R);
        for (uint32_t j = 0; j < utaps; ++j)
          for (uint32_t s = 0; s < uslots; ++s)
            for (uint32_t c = 0; c < uc; ++c)
              wh[static_cast<size_t>(j) * R + s * uc + c] =
                  LoadElemF32(w, static_cast<int64_t>(c) * k + j);
        return wh;
      },
      ttnn::Shape({utaps, R}), device);
  ttnn::Tensor biasT;
  if (bias != nullptr) {
    biasT = CachedTile(
        bias->data, R, 0, 0,
        [bias, R, uslots, uc] {
          EnsureHost(*bias);
          std::vector<float> bh(static_cast<size_t>(R));
          for (uint32_t s = 0; s < uslots; ++s)
            for (uint32_t c = 0; c < uc; ++c)
              bh[static_cast<size_t>(s) * uc + c] = LoadElemF32(*bias, c);
          return bh;
        },
        ttnn::Shape({1, R}), device);
  }

  // TT-GDN-REGION-REPLAY: hold the PERSISTENT shadow's handle. `T` is
  // reassigned below (the [window | x] concat), but the transposed shadow
  // `EnsureConvStateTransposed` served is the buffer the captured trace's
  // slice reads, and the tail commit below writes the rolled window back
  // INTO it (the same in-place fix GdnDecodeKernel's state commit carries)
  // so the trace's next slice reads the advanced window instead of the
  // freed capture-time buffer.
  ttnn::Tensor T0 = EnsureConvStateTransposed(conv_state, uslots, uc, usl, device);
  ttnn::Tensor T = T0;

  // x into the scratch row sl: indexed_fill (exact row write). Non-batch
  // columns are zero — they are masked out of the roll and never read.
  // Device arm: the producer commits x device-side, so the shadow serves
  // in-region and the scatter is one dim-0 indexed_fill into the warmed zeros
  // base (duplicates last-writer-wins, exactly the host loop). The eager
  // fallback covers shapes with no current shadow (first use, op-level tests).
  ttnn::Tensor xT;
  GdnIdxCacheEntry ice;
  const bool ice_ok = indexed && !has_null;
  if (ice_ok) ice = GdnConvIdxEntry(idxv, uslots, uc, device);
  {
    ttnn::Tensor x_raw;
    if (ServeDeviceShadowRaw(x, ub, uc, x_raw)) {
      ttnn::Tensor x_f32 = x_raw.dtype() == ttnn::DataType::FLOAT32
                               ? x_raw
                               : ttnn::typecast(x_raw, ttnn::DataType::FLOAT32);
      if (ice_ok) {
        // The rank-4 indexed_fill rides on ROW_MAJOR copies with the LAST
        // dim preserved — the exact ScatterRowsExact discipline. A rank-4
        // view on a TILE tensor re-pads the trailing 1-dims (the W0-sweep
        // x1024 physical blowup), and a view's span check fires on the
        // producer's tight slice buffers — found by the first driver run of
        // this path (MeshBuffer fatal from CausalConv1dUpdateKernel).
        auto to_rm = [](ttnn::Tensor t) {
          return t.layout() == ttnn::Layout::ROW_MAJOR
                     ? t
                     : ttnn::to_layout(t, ttnn::Layout::ROW_MAJOR);
        };
        ttnn::Tensor filled = ttnn::indexed_fill(
            ice.idx_scatter,
            ttnn::experimental::view(to_rm(ice.zeros_rc),
                                     ttnn::Shape({uslots, 1, 1, uc})),
            ttnn::experimental::view(to_rm(x_f32),
                                     ttnn::Shape({ub, 1, 1, uc})),
            std::nullopt, /*dim=*/0);
        xT = ttnn::to_layout(ttnn::reshape(std::move(filled),
                                           ttnn::Shape({1, R})),
                             ttnn::Layout::TILE);
      } else {
        // Non-indexed (compact) form: the batch rows ARE the cache rows.
        xT = ttnn::reshape(x_f32, ttnn::Shape({1, R}));
      }
    } else {
      VT_CHECK(!tt_capture_active(),
               "tenstorrent causal_conv1d_update: x arrived without a current "
               "device shadow during trace capture — the producer must commit "
               "device-side before the captured region");
      EnsureHost(x);
      std::vector<float> xrow(static_cast<size_t>(R), 0.0f);
      for (int64_t b = 0; b < batch; ++b) {
        const int32_t s = idxv[static_cast<size_t>(b)];
        if (s < 0) continue;  // NULL row: no x lands anywhere
        for (int64_t c = 0; c < c_dim; ++c)
          xrow[static_cast<size_t>(static_cast<int64_t>(s) * c_dim + c)] =
              LoadElemF32(x, b * x_rs + c);
      }
      xT = UploadTensor(std::move(xrow), ttnn::Shape({1, R}),
                        ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
    }
  }
  // The scratch row is the LAST state row (index usl), and the fill never
  // reads its old content — "write x's row into the scratch row" is therefore
  // exactly concat(history rows, xT), rank-2 TILE end to end. The rank-4
  // indexed_fill this replaces was the W4d 27B wall twice over: a rank-4 view
  // on the TILE state re-pads the trailing 1-dims to 32x32 per element (the
  // x1024 blowup — ~1.6 GB per decode step at R = 102400), and the ROW_MAJOR
  // rescue dies inside tt-metal's all-RM indexed_fill at op scale (conv
  // oracle max_abs 0.17 under every view/reshape variant, against a bit-clean
  // baseline). concat touches only {usl+1, R} planes — 13 MB at 27B — and
  // reinterprets nothing.
  if (xT.dtype() != T.dtype()) xT = ttnn::typecast(std::move(xT), T.dtype());
  T = usl == 0
          ? xT
          : ttnn::concat(
                std::vector<ttnn::Tensor>{
                    ttnn::slice(T, ttsl::SmallVector<uint32_t>{0u, 0u},
                                ttsl::SmallVector<uint32_t>{
                                    static_cast<uint32_t>(usl),
                                    static_cast<uint32_t>(R)},
                                ttsl::SmallVector<uint32_t>{1u, 1u}),
                    xT},
                /*dim=*/0);

  // MAC source: rows [0..width-1, scratch sl] — constant per geometry.
  std::vector<uint32_t> mac_rows;
  mac_rows.reserve(utaps);
  for (uint32_t j = 0; j < uw; ++j) mac_rows.push_back(j);
  mac_rows.push_back(usl);
  ttnn::Tensor Msrc = ttnn::gather(
      T, /*dim=*/0, ConvGatherIdx(mac_rows, R, uw, usl, device, /*kind=*/0),
      /*sparse_grad=*/false, std::nullopt);

  // Tap weights as [taps, R] (row j, column s*C+c = w[c, j]) — the step-
  // constant Wt tile warmed above; no per-step build or upload.
  ttnn::Tensor prod = ttnn::multiply(Msrc, Wt);  // exact (one ulp)

  // Sequential accumulation in the oracle's order: bias first, taps 0..width.
  // Both bases are step-constant caches (bias tile / warmed zeros).
  ttnn::Tensor acc = bias != nullptr ? biasT : CachedAccBase(R, device);
  for (uint32_t j = 0; j < utaps; ++j) {
    acc = ttnn::add(
        acc, ttnn::slice(prod, ttsl::SmallVector<uint32_t>{j, 0},
                         ttsl::SmallVector<uint32_t>{j + 1, R},
                         ttsl::SmallVector<uint32_t>{1, 1}));
  }
  ttnn::Tensor y = args.silu_activation ? ttnn::silu(acc) : acc;

  if (indexed && !has_null) {
    // Every batch row names a live slot: select each out row straight from
    // silu(y) viewed [slots, C] by the warmed broadcast row ids (one exact
    // dim-0 gather — the index's last dim spans the input's, the
    // ConvGatherIdx shape rule). No previous-bytes read: with no NULL rows
    // the oracle's prev branch never engages, so the fresh out buffer needs
    // nothing staged and the whole select stays device-side in-region.
    ttnn::Tensor yv = ttnn::reshape(y, ttnn::Shape({uslots, uc}));
    ttnn::Tensor ysel = ttnn::gather(yv, /*dim=*/0, ice.idx_rows,
                                     /*sparse_grad=*/false, std::nullopt);
    CommitDeviceLogical2D(out, std::move(ysel), ub, uc);
  } else if (indexed) {
    // NULL rows (idx<0) keep the out row's PREVIOUS bytes (the oracle skips
    // them entirely). Select each out element from one two-block row source:
    // block 0 = silu(y)^T [R, 1], block 1 = prev^T [R, 1] (prev[b, c] stored
    // at row b*C+c, independent of the slot map — no aliasing with named
    // slots). One per-step gather index; never cached (idx varies).
    EnsureHost(out);
    std::vector<float> prev(static_cast<size_t>(R), 0.0f);
    for (int64_t b = 0; b < batch; ++b)
      for (int64_t c = 0; c < c_dim; ++c)
        prev[static_cast<size_t>(b * c_dim + c)] = LoadElemF32(out, b * c_dim + c);
    ttnn::Tensor prevT = UploadTensor(std::move(prev), ttnn::Shape({1, R}),
                                      ttnn::DataType::FLOAT32,
                                      ttnn::Layout::TILE, device);
    ttnn::Tensor src = ttnn::concat(
        std::vector<ttnn::Tensor>{ttnn::transpose(y, 0, 1),
                                  ttnn::transpose(prevT, 0, 1)},
        0);  // [2R, 1]
    std::vector<uint32_t> orows(static_cast<size_t>(ub) * uc);
    for (int64_t b = 0; b < batch; ++b)
      for (int64_t c = 0; c < c_dim; ++c) {
        const int32_t s = idxv[static_cast<size_t>(b)];
        orows[static_cast<size_t>(b * c_dim + c)] =
            s >= 0 ? static_cast<uint32_t>(s) * uc + static_cast<uint32_t>(c)
                   : R + static_cast<uint32_t>(b * c_dim + c);
      }
    ttnn::Tensor oidx = UploadIdxU32(
        std::move(orows), ttnn::Shape({ub * uc, 1}), ttnn::Layout::TILE, device);
    ttnn::Tensor ysel = ttnn::gather(src, /*dim=*/0, oidx,
                                     /*sparse_grad=*/false, std::nullopt);
    ttnn::Tensor yout = ttnn::reshape(ysel, ttnn::Shape({ub, uc}));
    CommitDeviceLogical2D(out, std::move(yout), ub, uc);
  } else {
    // Compact form: acc's columns ARE the [B, C] out order.
    CommitDeviceLogical2D(out, ttnn::reshape(y, ttnn::Shape({ub, uc})), ub, uc);
  }

  // Roll: one constant-index gather — rows [1..width-1], scratch in at
  // width-1, tail rows identity, scratch row stays last.
  std::vector<uint32_t> roll_rows;
  roll_rows.reserve(usl + 1);
  for (uint32_t j = 1; j < uw; ++j) roll_rows.push_back(j);
  roll_rows.push_back(usl);
  for (uint32_t j = uw; j < usl; ++j) roll_rows.push_back(j);
  roll_rows.push_back(usl);
  ttnn::Tensor rolled = ttnn::gather(
      T, /*dim=*/0, ConvGatherIdx(roll_rows, R, uw, usl, device, /*kind=*/1),
      /*sparse_grad=*/false, std::nullopt);
  if (indexed && !has_null) {
    // Non-batch columns must not move (oracle leaves their rows untouched):
    // exact 0/1 blend back to the pre-step shadow — the warmed masks, no
    // per-step build or upload.
    rolled = ttnn::add(ttnn::multiply(rolled, ice.mv),
                        ttnn::multiply(T, ice.mk));
  } else if (indexed) {
    // NULL rows also pin their slot's row: exact 0/1 blend, built inline
    // (eager-only shape — NULL rows never capture).
    std::vector<float> mv(static_cast<size_t>(R), 0.0f),
        mk(static_cast<size_t>(R), 1.0f);
    for (int32_t s : idxv)
      if (s >= 0)
        for (uint32_t c = 0; c < uc; ++c) {
          mv[static_cast<size_t>(static_cast<int64_t>(s) * uc + c)] = 1.0f;
          mk[static_cast<size_t>(static_cast<int64_t>(s) * uc + c)] = 0.0f;
        }
    auto up1 = [&](std::vector<float>&& v) {
      return UploadTensor(std::move(v), ttnn::Shape({1, R}),
                          ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
    };
    rolled = ttnn::add(ttnn::multiply(rolled, up1(std::move(mv))),
                        ttnn::multiply(T, up1(std::move(mk))));
  }
  // bf16 STORAGE semantics (SupportsCompressedConvState, cuda_backend.cu:119):
  // a bf16 conv_state is "read/written in f32 registers" — every value that
  // enters the cache rounds through bf16 at the store boundary, so the next
  // step's window sees the STORED bits, not the unrounded f32 tap. The device
  // shadow is f32, so honor that boundary here: round the committed shadow
  // through bf16 on device (RNE, zero PCIe — the shadow stays resident). With
  // bf16-representable inputs (the production activation dtype) this is a
  // no-op; with f32-mantissa taps it is what keeps TT on the CUDA contract —
  // pinned by the bf16-state arm in tests/vt/test_tenstorrent_backend.cpp
  // (steps 1+: out max_rel 2-6.5% without this round).
  if (conv_state.dtype == DType::kBF16) {
    rolled = ttnn::typecast(ttnn::typecast(rolled, ttnn::DataType::BFLOAT16),
                            ttnn::DataType::FLOAT32);
  }
  // TT-GDN-REGION-REPLAY: commit the rolled window back INTO the persistent
  // transposed shadow (ttnn::copy — the RAC lane's proven capture-safe
  // preallocated-destination write) and publish THAT buffer, so the captured
  // trace's next slice reads the advanced window at the SAME address. The
  // replacing commit this fix removed churned the shadow's device buffer
  // every step, which freed the address the trace's slice had baked at
  // capture time — the recycled-garbage read and the clobbered live slot of
  // the served-replay defect (see GdnDecodeKernel's state commit for the
  // full chain). The shape guard keeps the replacing commit on any geometry
  // the roll did not reproduce exactly (the degenerate usl==0 arm).
  if (rolled.dtype() == T0.dtype() && rolled.layout() == T0.layout()) {
    const auto sr = rolled.logical_shape(), s0 = T0.logical_shape();
    bool same = sr.rank() == s0.rank();
    for (uint32_t r = 0; same && r < sr.rank(); ++r)
      same = sr[r] == s0[r];
    if (same) {
      ttnn::copy(rolled, T0);
      CommitConvTransposed(conv_state, std::move(T0), uslots, uc, usl);
      return;
    }
  }
  CommitConvTransposed(conv_state, std::move(rolled), uslots, uc, usl);
}



}  // namespace vt::tenstorrent
