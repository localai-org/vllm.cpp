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
                       const L2NormArgs& args) {
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
  // VT_TT_GDN_SPEC_DEBUG: the conv stream AS THE POST-CONV READS IT — pairs
  // with the conv-spec kernel's y readback to attribute a NaN/zero q/v to the
  // conv's commit vs this kernel's read.
  if (const char* dbg = std::getenv("VT_TT_GDN_SPEC_DEBUG");
      dbg != nullptr && dbg[0] != '\0' && !tt_capture_active()) {
    std::vector<float> cv = dev_conv.to_vector<float>();
    double c_acc = 0.0;
    uint64_t c_nan = 0, c_zero = 0;
    for (float f : cv) {
      c_acc += f > 0.0f ? f : -f;
      if (std::isnan(f)) ++c_nan;
      if (f == 0.0f) ++c_zero;
    }
    std::fprintf(stderr,
                 "[POSTCONV-IN] T=%u conv_dim=%u conv_asum=%.6g nan=%llu "
                 "zero=%llu/%zu conv_dt=%d\n",
                 t, conv_dim, c_acc, static_cast<unsigned long long>(c_nan),
                 static_cast<unsigned long long>(c_zero), cv.size(),
                 static_cast<int>(conv.dtype));
    std::fflush(stderr);
  }
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
  ttnn::Tensor q_dev = l2(q2), k_dev = l2(k2), v_dev = ttnn::reshape(v2, ttnn::Shape({t * hv, dv}));
  CommitDeviceLogical2D(q_out, q_dev, t * hk, dk);
  CommitDeviceLogical2D(k_out, k_dev, t * hk, dk);
  CommitDeviceLogical2D(v_out, std::move(v_dev), t * hv, dv);
  // VT_TT_GDN_SPEC_DEBUG: the q this kernel committed — pairs with the
  // GdnSpecDecode serve's q_asum (a zero there with a nonzero here is a
  // serve-side miss, not a producer defect).
  if (const char* dbg = std::getenv("VT_TT_GDN_SPEC_DEBUG");
      dbg != nullptr && dbg[0] != '\0' && !tt_capture_active()) {
    std::vector<float> qv = q_dev.to_vector<float>();
    double q_acc = 0.0;
    uint64_t q_nan = 0;
    for (float f : qv) {
      q_acc += f > 0.0f ? f : -f;
      if (std::isnan(f)) ++q_nan;
    }
    std::fprintf(stderr, "[POSTCONV-OUT] T=%u q_asum=%.6g q_nan=%llu\n", t,
                 q_acc, static_cast<unsigned long long>(q_nan));
    std::fflush(stderr);
  }

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
  // W21 (PROBE 2): the committed g/beta at the SPEC geometry (T == 9 — the
  // prefill runs T=32), first two spec calls, as raw f32. Pairs with the
  // GdnSpecDecode serve-side dumps (gs*_g/gs*_beta).
  if (const char* sd = std::getenv("VT_TT_SPEC_DUMP"); sd != nullptr && sd[0] != '\0') {
    static std::atomic<int64_t> gb_seq{0};
    if (t == 9) {
      const int64_t dc = gb_seq.fetch_add(1, std::memory_order_relaxed);
      if (dc < 2) {
        auto dump1 = [&](const char* name, const Tensor& tt) {
          auto v = vt::tenstorrent::DebugDeviceReadbackF32(q, tt);
          std::FILE* f = std::fopen(
              (std::string(sd) + "/gb" + std::to_string(dc) + "_" + name +
               ".f32").c_str(), "wb");
          if (f) { std::fwrite(v.data(), 4, v.size(), f); std::fclose(f); }
        };
        dump1("g", g_out);
        dump1("beta", beta_out);
        if (araw.rank == 2 && araw.IsContiguous()) dump1("a", araw);
        if (braw.rank == 2 && braw.IsContiguous()) dump1("b", braw);
      }
    }
  }
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
        /*output_head_major=*/false,
        /*use_mcast=*/true);  // the tt-metal pin's overload (default)
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
  // The eltwise multiply is exact f32 SFPU, so the three GEMMs route through
  // broadcast multiply + reduce (the outer product has no reduction at all —
  // one exact product per element). The reduction is ttnn::mean · Dk, NOT
  // ttnn::sum: on the tt-metal pin this backend builds against
  // (tt_umd 0.9.12 era, /tmp/umdtrial-install), f32 ttnn::sum lands on the
  // tf32 FPU tile-reduce (the header documents the split: mean's default
  // fast_and_approximate_mode=false is "the accurate SFPU path (full fp32)",
  // true "the faster tf32 FPU path", and sum has no SFPU arm), which
  // truncates the products exactly like the matmul MACs the wide-range arm
  // exists to exclude. mean over the K dim divides by an exact f32 scalar and
  // the rescale multiplies it back — two ulp of rounding against tf32's
  // 2^-11 truncation.
  const float dkf = static_cast<float>(dk);
  ttnn::Tensor dot = ttnn::multiply(
      ttnn::mean(ttnn::multiply(Sd, krow), ttsl::SmallVector<int>{2},
                 /*keep_dim=*/true),
      dkf);
  ttnn::Tensor vcol = ttnn::reshape(dev_v, ttnn::Shape({bh, dv, 1}));
  ttnn::Tensor beta = ttnn::reshape(dev_b, ttnn::Shape({bh, 1, 1}));
  ttnn::Tensor vp = ttnn::multiply(ttnn::subtract(vcol, dot), beta);
  ttnn::Tensor S2 = ttnn::add(Sd, ttnn::multiply(vp, krow));
  ttnn::Tensor qs = ttnn::multiply(dev_q, scale);
  ttnn::Tensor o = ttnn::multiply(
      ttnn::mean(ttnn::multiply(S2, ttnn::reshape(qs, ttnn::Shape({bh, 1, dk}))),
                 ttsl::SmallVector<int>{2}, /*keep_dim=*/true),
      dkf);
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
          /*use_qk_l2norm=*/false,
          /*output_head_major=*/false,
          /*use_mcast=*/true);  // the tt-metal pin's overload (default)
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
    // W20: the decisive instrument's SERVE side for the z gate. The
    // read-only trusted dual-read of the SERVED shadow + both addresses,
    // BEFORE the normalize (the bytes the gate arithmetic will consume).
    // Pairs with the commit-side [TRUST-SLOT] qkvz-z read: commit clean +
    // serve garbage = the alias; both garbage = the commit.
    if (std::getenv("VT_DUMP_TRUST") != nullptr && !tt_capture_active()) {
      static std::atomic<int64_t> gs_seq{0};
      if (gs_seq.fetch_add(1, std::memory_order_relaxed) < 96) {
        std::vector<float> p1 = dev_g.to_vector<float>();
        std::vector<float> p2 = dev_g.to_vector<float>();
        const bool agree = (p1 == p2);
        double acc = 0.0;
        uint64_t nan = 0, inf = 0;
        for (float f : p1) {
          if (std::isnan(f)) {
            ++nan;
            continue;
          }
          if (std::isinf(f)) {
            ++inf;
            continue;
          }
          acc += f > 0.0f ? f : -f;
        }
        auto os = DebugSlotStateOf(gate);
        std::fprintf(stderr,
                     "[GSERVE-TRUST] agree=%d n=%zu asum=%.6g nan=%llu "
                     "inf=%llu serve_addr=%llx owner_addr=%llx owndc=%d "
                     "ownhc=%d\n",
                     agree ? 1 : 0, p1.size(), acc,
                     static_cast<unsigned long long>(nan),
                     static_cast<unsigned long long>(inf),
                     static_cast<unsigned long long>(
                         dev_g.mesh_buffer().address()),
                     static_cast<unsigned long long>(os.dev_buf_addr),
                     os.device_current ? 1 : 0, os.host_current ? 1 : 0);
        std::fflush(stderr);
      }
    }
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
  // VT_TT_GDN_SPEC_DEBUG: the gated norm's own operands, read back at the
  // exact tensors the compute consumes. The model-side taps read the slot
  // shadows; if those are clean while these are poisoned, the serve handed a
  // different tensor than the slot holds.
  if (const char* dbg = std::getenv("VT_TT_GDN_SPEC_DEBUG");
      dbg != nullptr && dbg[0] != '\0' && !tt_capture_active()) {
    static std::atomic<int64_t> rng_seq{0};
    const int64_t call = rng_seq.fetch_add(1, std::memory_order_relaxed);
    if (call < 400) {
      auto probe = [](const char* tag, const ttnn::Tensor& t) {
        std::vector<float> v = t.to_vector<float>();
        double acc = 0.0;
        uint64_t nan = 0, inf = 0;
        for (float f : v) {
          if (std::isnan(f)) ++nan;
          else if (std::isinf(f)) ++inf;
          else acc += f > 0.0f ? f : -f;
        }
        std::fprintf(stderr, "[RNG-IN] %s n=%zu asum=%.6g nan=%llu inf=%llu\n",
                     tag, v.size(), acc, static_cast<unsigned long long>(nan),
                     static_cast<unsigned long long>(inf));
      };
      probe("x", dev_x);
      probe("g", dev_g);
      std::fflush(stderr);
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
  // W21: the CONV-SPEC fill's miswrite (the k-th replacement row lands at
  // id[k]+k) was measured at STRIDED ids [0,9,...,72]; the consecutive-id
  // scatters (the state commits, [0..F)) are the form the decode path has
  // always run and the W21 concat-side dumps confirm faithful (out == src ++
  // tail). The identity fast path was tried here and broke the scatter test's
  // volume contract — reverted; the fill stays for this form.
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

// ---- kCausalConv1dSpecUpdate (QWEN38-DFLASH2-STAGE-C W3) --------------------
//
// The SPECULATIVE conv window update (SPEC-MTP I4; cpu_ops.cpp
// CausalConv1dSpecUpdateKernel at :1863, ported from vllm causal_conv1d.py
// @ e24d1b24 IS_SPEC_DECODING + IS_VARLEN). The kCausalConv1dUpdate above is
// the decode step: read-old-then-roll at window offset 0, one token per
// request. The spec verify advances the WIDENED conv row (state_len =
// (K-1) + (max_query_len-1) taps) by the ACCEPTED prefix: per request i,
// with off = num_accepted_tokens[i]-1, the window stream is
//   S = old_state[off .. off+K-2] ++ x[lo .. hi-1]
// out[t] = silu(bias + sum_j w[j]*S[t+j]) over the request's tokens, and the
// committed row is new_state = old_state[off+1 .. off+keep] ++ x[lo..hi-1]
// (keep = state_len - seqlen; the one-tap left shift IS the conv rollback).
//
// PROVENANCE (never a mirror source; oracle registry `vllm-tt-plugin`,
// gateable = no): the bundle's gdn/tp.py `spec_conv_sel` rebuilds each user's
// conv window for the next verify by SELECTING rows out of
// cat([E_prev, qkv_new]) at the accepted index — the same
// select-by-accepted-prefix semantics, implemented here against our
// persistent transposed shadow instead of a per-verify one-hot matmul.
//
// Composition = CausalConv1dUpdateKernel's proven primitives, dim-0 only
// (the W2 micro-probe doctrine). The shadow T0 is [sl+1, R] time-major
// (row j, col s*C+c = state[s][c][j]). The stream the MAC reads is one
// concat of the per-request state-window rows and the x token rows; the MAC
// is exact tap-major multiply + sequential adds in the oracle's accumulation
// order (bias, taps 0..K-1); the commit is one gather out of
// SRC = [T0 | zeros | xT] plus the exact 0/1 blend that leaves
// non-participating slots byte-identical.
//
// The gather indices split by CONTENT STABILITY, which is what makes the op
// capture-safe AND acceptance-varying-safe at once:
//   * everything derived from the state slots and cu_seqlens (the owner map,
//     the x scatter ids, the out column select, the roll's x-part rows and
//     its keep mask) is STABLE across a sequence's steps — host-built once
//     per content, cached, and refused on change during capture exactly like
//     GdnConvIdxEntry;
//   * everything derived from num_accepted_tokens (the window read offset
//     and the roll's state-part rows) CHANGES EVERY VERIFY STEP — those two
//     indices are derived ON DEVICE from the persistent num_accepted_tokens
//     buffer (integer-exact f32 arithmetic: every value is < 2^24, so the
//     f32 mul/add is exact and the f32->u32 typecast is the identity on
//     integral values). A host-built index would bake one step's acceptance
//     into the captured graph and silently corrupt every later replay.
struct ConvSpecIdxEntry {
  std::vector<int32_t> idx;  // content identity: slot per request
  std::vector<int32_t> cs;   // content identity: cu_seqlens
  bool warmed = false;
  // slot/cu_seqlens-derived (stable per capture):
  ttnn::Tensor owner_idx;   // [R, 1] u32 TILE — column -> owning request (0
                            // when no live request owns it)
  ttnn::Tensor x_scatter;   // [T] u32 RM — indexed_fill bid: token t -> its
                            // slot's block of row t
  ttnn::Tensor zeros_ts;    // [T*slots, C] f32 RM — the x-scatter base
  ttnn::Tensor mac_idx;     // [K*T, R] u32 TILE — tap-major stream rows
                            // (tap j's slice is rows [j*T, (j+1)*T))
  ttnn::Tensor osel_idx;    // [C, T] u32 TILE — the out column select on the
                            // transposed y (row ids s_i*C + c)
  ttnn::Tensor roll_x;      // [sl, R] f32 TILE — the roll's x-part SRC rows
  ttnn::Tensor roll_m;      // [sl, R] f32 TILE — 1 where the roll reads a
                            // state row (j < keep), 0 on the x part
  ttnn::Tensor mv, mk;      // [1, R] f32 TILE — the non-owned-slot restore /
                            // keep masks for the state commit
  ttnn::Tensor zeros_blk;   // [sl-width+1, R] f32 TILE — SRC's out-of-range
                            // zeros (the oracle's `src < state_len` guard)
  ttnn::Tensor x_mask;      // [T, slots] f32 TILE — W21: the broadcast-mask
                            // form of the x scatter (the indexed_fill
                            // replacement): mask[t,s] = (s == slot(t))
};
std::mutex& ConvSpecIdxMutex() {
  static std::mutex* m = new std::mutex();  // never destroyed (#1486)
  return *m;
}
std::map<std::array<uint64_t, 5>, ConvSpecIdxEntry>& ConvSpecIdxMap() {
  static std::map<std::array<uint64_t, 5>, ConvSpecIdxEntry>* c =
      new std::map<std::array<uint64_t, 5>, ConvSpecIdxEntry>();  // #1486
  return *c;
}

void CausalConv1dSpecUpdateKernel(Queue& q_unused_probe, Tensor& out, const Tensor& x, const Tensor& w,
                                  const Tensor* bias, Tensor& conv_state,
                                  const Tensor& conv_state_indices,
                                  const Tensor& num_accepted_tokens,
                                  const Tensor& cu_seqlens,
                                  const CausalConv1dArgs& args) {
  TT_OP_TRACE("CausalConv1dSpecUpdate");
  GdnDecodeSteps().fetch_add(1, std::memory_order_relaxed);
  const int64_t tokens = x.shape[0], c_dim = x.shape[1], k = w.shape[1];
  const int64_t width = k - 1;
  const int64_t state_len = conv_state.shape[2];
  const int64_t slots = conv_state.shape[0];
  const int64_t num_reqs = cu_seqlens.shape[0] - 1;
  const int64_t x_rs = x.stride[0];
  if (tokens == 0) return;  // empty stream: no-op, cache as-is
  MeshDevice& device = SharedMeshDevice();
  const uint32_t R = static_cast<uint32_t>(slots * c_dim);
  const uint32_t uc = static_cast<uint32_t>(c_dim);
  const uint32_t usl = static_cast<uint32_t>(state_len);
  const uint32_t uslots = static_cast<uint32_t>(slots);
  const uint32_t utok = static_cast<uint32_t>(tokens);

  // Metadata: the slot ids and cu_seqlens are stable across a sequence's
  // steps (the decode discipline), so the host masters are read for the
  // content check and the eager builds. NULL rows never capture (the decode
  // discipline); the eager arm handles them by skipping the request
  // entirely (the oracle's :114-115 continue) — its out rows keep their
  // previous bytes and its slot is not owned by the roll.
  std::vector<int32_t> idxv =
      ReadIdxHost(conv_state_indices, slots, "causal_conv1d_spec_update", true);
  EnsureHost(cu_seqlens);
  const int32_t* csp = cu_seqlens.Ptr<int32_t>();
  std::vector<int32_t> csv(csp, csp + num_reqs + 1);
  bool has_null = false;
  for (int32_t s : idxv)
    if (s < 0) has_null = true;
  VT_CHECK(!has_null || !tt_capture_active(),
           "tenstorrent causal_conv1d_spec_update: NULL state rows during "
           "trace capture — the captured arm is the pure-spec indexed path");

  // Per-request geometry from cu_seqlens (host, stable content).
  int64_t total = 0;
  std::vector<int64_t> lo(static_cast<size_t>(num_reqs)), keep(static_cast<size_t>(num_reqs));
  for (int64_t i = 0; i < num_reqs; ++i) {
    lo[static_cast<size_t>(i)] = csv[static_cast<size_t>(i)];
    const int64_t seqlen = csv[static_cast<size_t>(i + 1)] - csv[static_cast<size_t>(i)];
    keep[static_cast<size_t>(i)] = state_len - seqlen;  // may exceed width
    total += seqlen;
  }
  VT_CHECK(total == tokens,
           "tenstorrent causal_conv1d_spec_update: cu_seqlens does not cover "
           "the x stream");

  // ---- the content-keyed stable index set --------------------------------
  const std::array<uint64_t, 5> key{
      static_cast<uint64_t>(slots), static_cast<uint64_t>(c_dim),
      static_cast<uint64_t>(state_len), static_cast<uint64_t>(tokens),
      static_cast<uint64_t>(num_reqs)};
  ConvSpecIdxEntry e;
  {
    std::lock_guard<std::mutex> g(ConvSpecIdxMutex());
    auto it = ConvSpecIdxMap().find(key);
    if (it != ConvSpecIdxMap().end() && it->second.warmed && it->second.idx == idxv &&
        it->second.cs == csv) {
      e = it->second;  // copy the handles under the lock
    } else {
      VT_CHECK(!tt_capture_active(),
               "tenstorrent causal_conv1d_spec_update: slot/seqlen metadata "
               "changed during trace capture — the recapture cadence owns a "
               "metadata change (reset the graph, then re-warm on the eager "
               "boundary step)");
      ConvSpecIdxEntry n;
      n.warmed = true;
      // owner map: column (s*C+c) -> the request that names slot s (0 when
      // none does — request 0's off is read there, a finite garbage value
      // the roll blend and the out select never propagate).
      std::vector<uint32_t> owner(static_cast<size_t>(R), 0u);
      for (int64_t i = 0; i < num_reqs; ++i) {
        const int32_t s = idxv[static_cast<size_t>(i)];
        if (s < 0) continue;
        for (int64_t c = 0; c < c_dim; ++c)
          owner[static_cast<size_t>(s * c_dim + c)] = static_cast<uint32_t>(i);
      }
      n.owner_idx = UploadIdxU32(
          std::move(owner), ttnn::Shape({R, 1}), ttnn::Layout::TILE, device);
      // x scatter: token t's row lands at block (t*slots + slot_t) of the
      // [T*slots, C] base — flat (t*slots + s)*C + c == t*R + s*C + c, so
      // the [T*slots, C] result views as the [T, R] slot-ordered x.
      std::vector<uint32_t> xbid(static_cast<size_t>(tokens));
      for (int64_t t = 0; t < tokens; ++t) {
        int64_t req = num_reqs - 1;
        for (int64_t i = 0; i < num_reqs; ++i) {
          if (t >= csv[static_cast<size_t>(i)] && t < csv[static_cast<size_t>(i + 1)]) {
            req = i;
            break;
          }
        }
        const int32_t s = idxv[static_cast<size_t>(req)];
        xbid[static_cast<size_t>(t)] = static_cast<uint32_t>(t * slots + (s < 0 ? 0 : s));
      }
      n.x_scatter = UploadIdxU32(std::move(xbid), ttnn::Shape({utok}),
                                 ttnn::Layout::ROW_MAJOR, device);
      n.zeros_ts = ttnn::zeros(tt::tt_metal::Shape({utok * uslots, uc}),
                               ttnn::DataType::FLOAT32, ttnn::Layout::ROW_MAJOR,
                               std::ref(device));
      // VT_TT_GDN_SPEC_DEBUG W7: the scatter base's resident DRAM address —
      // the aliasing check against the DENSE-LAYER probe's slot addresses.
      if (std::getenv("VT_TT_GDN_SPEC_DEBUG") != nullptr)
        std::fprintf(stderr,
                     "[CONV-SPEC-ENTRY] zeros_ts %ux%u addr=%llx "
                     "x_scatter=%u rows\n",
                     static_cast<unsigned>(utok * uslots),
                     static_cast<unsigned>(uc),
                     static_cast<unsigned long long>(
                         n.zeros_ts.mesh_buffer().address()),
                     static_cast<unsigned>(utok));
      // MAC stream rows: token t (request i, position p = t - lo_i), tap j:
      //   p + j < width ? p + j (the state window part)
      //     : width + (lo_i + p + j - width) = t + j (the x part)
      // — the per-request offsets are baked into the stream's state rows, so
      // the tap row is uniform across the columns and the oracle's
      // win[t+j] addressing falls out exactly. The gather index is the FULL
      // [K*T, R] broadcast form (ConvGatherIdx's shape rule: the index's
      // last dim spans the input's), stored TAP-major so tap j's slice is
      // the contiguous rows [j*T, (j+1)*T).
      std::vector<uint32_t> mac(static_cast<size_t>(k * tokens) * R, 0u);
      for (int64_t t = 0; t < tokens; ++t) {
        int64_t req = num_reqs - 1;
        for (int64_t i = 0; i < num_reqs; ++i) {
          if (t >= csv[static_cast<size_t>(i)] && t < csv[static_cast<size_t>(i + 1)]) {
            req = i;
            break;
          }
        }
        const int64_t p = t - lo[static_cast<size_t>(req)];
        for (int64_t j = 0; j < k; ++j) {
          const int64_t m = p + j;
          const uint32_t row = static_cast<uint32_t>(
              m < width ? m : width + (t + j - width));
          for (int64_t c = 0; c < R; ++c)
            mac[static_cast<size_t>(j * tokens + t) * R + c] = row;
        }
      }
      n.mac_idx = UploadIdxU32(
          std::move(mac),
          ttnn::Shape({static_cast<uint32_t>(k * utok), R}), ttnn::Layout::TILE,
          device);
      // out column select on the transposed y: osel[c, t] = slot_i*C + c.
      std::vector<uint32_t> osel(static_cast<size_t>(c_dim * tokens));
      for (int64_t t = 0; t < tokens; ++t) {
        int64_t req = num_reqs - 1;
        for (int64_t i = 0; i < num_reqs; ++i) {
          if (t >= csv[static_cast<size_t>(i)] && t < csv[static_cast<size_t>(i + 1)]) {
            req = i;
            break;
          }
        }
        const int32_t s = idxv[static_cast<size_t>(req)];
        for (int64_t c = 0; c < c_dim; ++c)
          osel[static_cast<size_t>(c * tokens + t)] =
              static_cast<uint32_t>((s < 0 ? 0 : s) * c_dim + c);
      }
      n.osel_idx = UploadIdxU32(
          std::move(osel), ttnn::Shape({uc, utok}), ttnn::Layout::TILE, device);
      // The roll's x-part rows and keep mask (cs/stable-derived; the mux is
      // exact 0/1 arithmetic, never a `where`): roll_x[j][col] = SRC row of
      // the x part when j >= keep_i (else 0 — masked out), roll_m[j][col] =
      // 1 when j < keep_i (read the state part).
      const uint32_t x_base = 2 * usl - static_cast<uint32_t>(width) + 2;
      std::vector<float> rx(static_cast<size_t>(state_len) * R, 0.0f);
      std::vector<float> rm(static_cast<size_t>(state_len) * R, 0.0f);
      for (int64_t i = 0; i < num_reqs; ++i) {
        const int32_t s = idxv[static_cast<size_t>(i)];
        if (s < 0) continue;
        for (int64_t c = 0; c < c_dim; ++c) {
          const size_t base = static_cast<size_t>(s * c_dim + c);
          for (int64_t j = 0; j < state_len; ++j) {
            rm[base + static_cast<size_t>(j) * R] = j < keep[static_cast<size_t>(i)] ? 1.0f : 0.0f;
            if (j >= keep[static_cast<size_t>(i)]) {
              rx[base + static_cast<size_t>(j) * R] = static_cast<float>(
                  x_base + lo[static_cast<size_t>(i)] + (j - keep[static_cast<size_t>(i)]));
            }
          }
        }
      }
      n.roll_x = UploadTensor(std::move(rx), ttnn::Shape({usl, R}),
                              ttnn::DataType::FLOAT32, ttnn::Layout::TILE,
                              device);
      n.roll_m = UploadTensor(std::move(rm), ttnn::Shape({usl, R}),
                              ttnn::DataType::FLOAT32, ttnn::Layout::TILE,
                              device);
      // the non-owned-slot restore/keep masks (the decode kernel's mv/mk)
      std::vector<float> mv(static_cast<size_t>(R), 0.0f), mk(static_cast<size_t>(R), 1.0f);
      for (int32_t s : idxv) {
        if (s < 0) continue;
        for (int64_t c = 0; c < c_dim; ++c) {
          mv[static_cast<size_t>(s * c_dim + c)] = 1.0f;
          mk[static_cast<size_t>(s * c_dim + c)] = 0.0f;
        }
      }
      n.mv = UploadTensor(std::move(mv), ttnn::Shape({1, R}),
                          ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
      n.mk = UploadTensor(std::move(mk), ttnn::Shape({1, R}),
                          ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
      // W21: the x scatter's broadcast mask — mask[t, s] = (s == slot(t)).
      // Exact 0/1 f32; content-keyed with the rest of the entry.
      {
        std::vector<float> xm(static_cast<size_t>(tokens) * uslots, 0.0f);
        for (int64_t t = 0; t < tokens; ++t) {
          int64_t req = num_reqs - 1;
          for (int64_t i = 0; i < num_reqs; ++i) {
            if (t >= csv[static_cast<size_t>(i)] && t < csv[static_cast<size_t>(i + 1)]) {
              req = i;
              break;
            }
          }
          const int32_t s = idxv[static_cast<size_t>(req)];
          if (s >= 0)
            xm[static_cast<size_t>(t) * uslots + s] = 1.0f;
        }
        n.x_mask = UploadTensor(std::move(xm), ttnn::Shape({utok, uslots}),
                                ttnn::DataType::FLOAT32, ttnn::Layout::TILE,
                                device);
      }
      // SRC's out-of-range zeros: the oracle's `src < state_len ? srow[src] :
      // 0` guard — the roll's state part can read past the row when a short
      // request's keep exceeds the window; those reads land in this block.
      const uint32_t zrows = usl - static_cast<uint32_t>(width) + 1;
      n.zeros_blk = ttnn::zeros(tt::tt_metal::Shape({zrows, R}),
                                ttnn::DataType::FLOAT32, ttnn::Layout::TILE,
                                std::ref(device));
      n.idx = idxv;
      n.cs = csv;
      ConvSpecIdxMap()[key] = std::move(n);
      e = ConvSpecIdxMap()[key];
    }
  }
  // ---- the transposed shadow ----------------------------------------------
  ttnn::Tensor T0 = EnsureConvStateTransposed(conv_state, uslots, uc, usl, device);
  ttnn::Tensor T = T0;
  // ---- x into slot-ordered rows [T, R] ------------------------------------
  ttnn::Tensor xT;
  bool x_from_shadow = false;  // VT_TT_GDN_SPEC_DEBUG: which serve the x took
  // VT_TT_GDN_SPEC_DEBUG stage probe: read back EACH scatter-chain stage
  // (as-served shadow -> post-to_layout -> post-indexed_fill) and NaN-count
  // it. One run attributes the NaN to the serve, the layout conversion, or
  // the fill itself. Gated to the first 48 calls (step 1 covers the NaN
  // onset and the frozen-x onset); each readback is a blocking D2H, so this
  // is a diagnostic-only lane.
  static std::atomic<int64_t> conv_spec_probe_seq{0};
  static const bool conv_spec_dbg_on = [] {
    const char* d = std::getenv("VT_TT_GDN_SPEC_DEBUG");
    return d != nullptr && d[0] != '\0';
  }();
  const bool conv_spec_probe =
      conv_spec_dbg_on && !tt_capture_active() &&
      conv_spec_probe_seq.fetch_add(1, std::memory_order_relaxed) < 48;
  // W21 (the conv-spec zero-row probe): raw f32 dumps of the scatter-chain
  // stages at the first spec calls. The W20 dump forensics showed the
  // conv-spec OUTPUT carries exact-zero token rows at every spec call while
  // the served x reads clean — these dumps split which stage introduces the
  // zeros (the fill's clone, the window gather, the MAC, or the out select).
  static const char* spec_dump_dir = std::getenv("VT_TT_SPEC_DUMP");
  static std::atomic<int64_t> spec_dump_seq{0};
  const bool spec_dump_on =
      spec_dump_dir != nullptr && spec_dump_dir[0] != '\0' && !tt_capture_active();
  const int64_t spec_dump_call =
      spec_dump_on ? spec_dump_seq.fetch_add(1, std::memory_order_relaxed) : -1;
  auto spec_dump = [&](const char* name, const ttnn::Tensor& t) {
    if (!spec_dump_on || spec_dump_call >= 2) return;
    auto v = t.to_vector<float>();
    std::FILE* f = std::fopen(
        (std::string(spec_dump_dir) + "/cs" + std::to_string(spec_dump_call) +
         "_" + name + ".f32").c_str(), "wb");
    if (f) { std::fwrite(v.data(), 4, v.size(), f); std::fclose(f); }
  };
  auto probe_stage = [conv_spec_probe](const char* tag,
                                       const ttnn::Tensor& t) {
    if (!conv_spec_probe) return;
    std::vector<float> v = t.to_vector<float>();
    double acc = 0.0;
    uint64_t nan = 0;
    for (float f : v) {
      if (std::isnan(f)) {
        ++nan;
        continue;
      }
      acc += f > 0.0f ? f : -f;
    }
    const auto ls = t.logical_shape();
    std::fprintf(stderr,
                 "[CONV-SPEC-STAGE] %s n=%zu asum=%.6g nan=%llu shape=%ux%u "
                 "dt=%d layout=%d\n",
                 tag, v.size(), acc, static_cast<unsigned long long>(nan),
                 static_cast<unsigned>(ls[0]), static_cast<unsigned>(ls[1]),
                 static_cast<int>(t.dtype()), static_cast<int>(t.layout()));
    std::fflush(stderr);
  };
  {
    ttnn::Tensor x_raw;
    if ((x_from_shadow = ServeDeviceShadowRaw(x, utok, uc, x_raw))) {
      probe_stage("serve", x_raw);
      // W20: the decisive instrument's SERVE side for x (the qkvz mixed
      // output as this kernel consumes it). The read-only trusted dual-read
      // of the SERVED shadow + both addresses: if the served buffer's DRAM
      // address differs from the owner slot's, the serve handed a different
      // buffer than the commit wrote (the alias); if the two passes disagree,
      // the residency state changed under the read.
      if (conv_spec_dbg_on && !tt_capture_active()) {
        static std::atomic<int64_t> xs_seq{0};
        if (xs_seq.fetch_add(1, std::memory_order_relaxed) < 48) {
          std::vector<float> p1 = x_raw.to_vector<float>();
          std::vector<float> p2 = x_raw.to_vector<float>();
          const bool agree = (p1 == p2);
          double acc = 0.0;
          uint64_t nan = 0, inf = 0;
          for (float f : p1) {
            if (std::isnan(f)) {
              ++nan;
              continue;
            }
            if (std::isinf(f)) {
              ++inf;
              continue;
            }
            acc += f > 0.0f ? f : -f;
          }
          auto os = vt::tenstorrent::DebugSlotStateOf(x);
          std::fprintf(stderr,
                       "[XSERVE-TRUST] agree=%d n=%zu asum=%.6g nan=%llu "
                       "inf=%llu serve_addr=%llx owner_addr=%llx "
                       "owndc=%d ownhc=%d\n",
                       agree ? 1 : 0, p1.size(), acc,
                       static_cast<unsigned long long>(nan),
                       static_cast<unsigned long long>(inf),
                       static_cast<unsigned long long>(
                           x_raw.mesh_buffer().address()),
                       static_cast<unsigned long long>(os.dev_buf_addr),
                       os.device_current ? 1 : 0, os.host_current ? 1 : 0);
          std::fflush(stderr);
        }
      }
      ttnn::Tensor x_f32 = x_raw.dtype() == ttnn::DataType::FLOAT32
                               ? x_raw
                               : ttnn::typecast(x_raw, ttnn::DataType::FLOAT32);
      auto to_rm = [](ttnn::Tensor t) {
        return t.layout() == ttnn::Layout::ROW_MAJOR
                   ? t
                   : ttnn::to_layout(t, ttnn::Layout::ROW_MAJOR);
      };
      ttnn::Tensor x_rm = to_rm(x_f32);
      probe_stage("rm", x_rm);
      spec_dump("x_served", x_rm);
      // W7b: the fill's BASE and the scatter INDICES, read at fill time. The
      // filled output = zeros_ts.clone() with x's rows written at the scatter
      // positions, so garbage in the non-written region IS the base's live
      // content — this read separates "the cached base was stomped" from
      // "the clone never copied it". The indices print raw: an out-of-range
      // batch_id makes the fill write out of bounds (the stomp candidate).
      if (conv_spec_probe) {
        probe_stage("base", e.zeros_ts);
        std::vector<uint32_t> sv = e.x_scatter.to_vector<uint32_t>();
        std::fprintf(stderr, "[CONV-SPEC-STAGE] scatter n=%zu:", sv.size());
        for (size_t i = 0; i < sv.size() && i < 12; ++i)
          std::fprintf(stderr, " %u", sv[i]);
        std::fprintf(stderr, "\n");
        std::fflush(stderr);
      }
      // W21 Experiment C (MASK): build xT as a broadcast multiply —
      // xT[t, s*C+c] = x[t, c] * (s == slot(t)) — instead of the indexed_fill
      // scatter. Exact f32 elementwise (bit-exact vs the oracle's row write);
      // bypasses the fill whose k-th replacement row lands at id[k]+k in the
      // engine (the W21 finding). Same traffic class as the fill (one [T,R]
      // materialization).
      static const bool fill_mask = [] {
        const char* v = std::getenv("VT_TT_CONV_SPEC_FILL_MASK");
        return v == nullptr || v[0] != '0';  // default ON
      }();
      if (fill_mask) {
        ttnn::Tensor xb = ttnn::reshape(
            x_f32, ttnn::Shape({utok, 1u, uc}));  // [T,1,C] broadcast source
        ttnn::Tensor xm = ttnn::reshape(
            e.x_mask, ttnn::Shape({utok, uslots, 1u}));  // [T,slots,1]
        ttnn::Tensor xsc = ttnn::multiply(xb, xm);       // [T,slots,C] exact
        xT = ttnn::to_layout(
            ttnn::reshape(std::move(xsc), ttnn::Shape({utok, R})),
            ttnn::Layout::TILE);
        probe_stage("masked", xT);
        spec_dump("masked", xT);
      } else {
        ttnn::Tensor fill_base = e.zeros_ts;
        ttnn::Tensor fill_ids = e.x_scatter;
      // W21 Experiment A (FRESH): rebuild the base + ids tensors per call —
      // fresh device tensors, same values. Distinguishes a stale TENSOR
      // binding on the program-cache-hit path from a stale RUN-ARGS path.
      static const bool fill_fresh = [] {
        const char* v = std::getenv("VT_TT_CONV_SPEC_FILL_FRESH");
        return v != nullptr && v[0] == '1';
      }();
      // W21 Experiment B (BUST): a UNIQUE batch-ids shape per call (extra ids
      // pointing at a harmless non-slot-0 block, x padded to match) — every
      // call is a program-cache MISS, so a cache-hit run-args/binding
      // staleness (the d1be4cf5b layernorm class) cannot serve.
      static const bool fill_bust = [] {
        const char* v = std::getenv("VT_TT_CONV_SPEC_FILL_BUST");
        return v != nullptr && v[0] == '1';
      }();
      static std::atomic<int64_t> fill_bust_seq{0};
      const int64_t bust_ix = fill_bust ? fill_bust_seq.fetch_add(1) : 0;
      uint32_t base_rows = utok * uslots;
      if (fill_fresh || fill_bust) {
        fill_base = ttnn::zeros(tt::tt_metal::Shape({base_rows, uc}),
                                ttnn::DataType::FLOAT32, ttnn::Layout::ROW_MAJOR,
                                std::ref(device));
      }
      if (fill_fresh) {
        fill_ids = UploadIdxU32(e.x_scatter.to_vector<uint32_t>(),
                                ttnn::Shape({utok}), ttnn::Layout::ROW_MAJOR,
                                device);
      }
      if (fill_bust) {
        const uint32_t extra = static_cast<uint32_t>(bust_ix % 7) + 1;
        std::vector<uint32_t> ids = e.x_scatter.to_vector<uint32_t>();
        for (uint32_t j = 0; j < extra; ++j) ids.push_back(1u);  // harmless block
        fill_ids = UploadIdxU32(std::move(ids),
                                ttnn::Shape({utok + extra}),
                                ttnn::Layout::ROW_MAJOR, device);
        std::vector<float> xp(x_rm.to_vector<float>());
        xp.resize(static_cast<size_t>(utok + extra) * uc, 0.0f);
        x_rm = ttnn::Tensor::from_vector<float>(
            std::move(xp),
            tt::tt_metal::TensorSpec(
                tt::tt_metal::Shape({utok + extra, uc}),
                tt::tt_metal::TensorLayout(
                    ttnn::DataType::FLOAT32, tt::tt_metal::PageConfig(ttnn::Layout::ROW_MAJOR),
                    tt::tt_metal::MemoryConfig{})),
            &device);
      }
      ttnn::Tensor filled = ttnn::indexed_fill(
          fill_ids,
          ttnn::experimental::view(fill_base,
                                   ttnn::Shape({base_rows, 1, 1, uc})),
          ttnn::experimental::view(x_rm,
                                   ttnn::Shape({x_rm.logical_shape()[0], 1, 1, uc})),
          std::nullopt, /*dim=*/0);
      probe_stage("filled", ttnn::experimental::view(
                                filled, ttnn::Shape({utok, R})));
      spec_dump("filled", ttnn::experimental::view(
                              filled, ttnn::Shape({utok, R})));
      xT = ttnn::to_layout(
          ttnn::experimental::view(std::move(filled),
                                   ttnn::Shape({utok, R})),
          ttnn::Layout::TILE);
      probe_stage("tile", xT);
      }
    } else {
      VT_CHECK(!tt_capture_active(),
               "tenstorrent causal_conv1d_spec_update: x arrived without a "
               "current device shadow during trace capture — the captured "
               "arm reads device-resident activations only");
      EnsureHost(x);
      std::vector<float> xr(static_cast<size_t>(tokens) * R, 0.0f);
      for (int64_t t = 0; t < tokens; ++t) {
        int64_t req = num_reqs - 1;
        for (int64_t i = 0; i < num_reqs; ++i) {
          if (t >= csv[static_cast<size_t>(i)] && t < csv[static_cast<size_t>(i + 1)]) {
            req = i;
            break;
          }
        }
        const int32_t s = idxv[static_cast<size_t>(req)];
        if (s < 0) continue;  // NULL request: no x lands anywhere
        for (int64_t c = 0; c < c_dim; ++c)
          xr[static_cast<size_t>(t * R + s * c_dim + c)] =
              LoadElemF32(x, t * x_rs + c);
      }
      xT = UploadTensor(std::move(xr), ttnn::Shape({utok, R}),
                        ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
    }
  }
  // ---- the acceptance-derived indices (device side) ------------------------
  // off_col[c] = num_accepted[owner(c)] - 1. All values are small integers,
  // so the f32 arithmetic is exact and the f32->u32 typecast back is the
  // identity on integral values.
  //
  // num_accepted_tokens is the ONE metadata whose content changes every
  // verify step (the previous step's acceptance). Its host master is fresh
  // (the driver's StageSpecStepInputs memcpy), so the eager arm reads it and
  // re-uploads the tiny [nreq, 1] tile per call. A CAPTURED call refuses
  // here, by name: tt-metal records no writes in-region ("Writes are not
  // supported during trace capture"), and a capture-time upload would replay
  // one step's acceptance forever — silently advancing every later verify by
  // a stale prefix. The named gap is the driver-side persistent-device
  // refresh (the WarmRopeCosSin copy_to_device pattern) this op is shaped to
  // consume the moment it lands.
  VT_CHECK(!tt_capture_active(),
           "tenstorrent causal_conv1d_spec_update: the acceptance metadata "
           "has no persistent device refresh on this backend — the captured "
           "arm would replay one step's num_accepted forever. Named gap: the "
           "driver's TT spec-metadata in-place device refresh (the "
           "WarmRopeCosSin copy_to_device pattern); until it lands the spec "
           "verify runs on the eager arm (VLLM_CPP_CUDAGRAPH=0).");
  EnsureHost(num_accepted_tokens);
  const int32_t* np = num_accepted_tokens.Ptr<int32_t>();
  std::vector<float> nat_host(static_cast<size_t>(num_reqs));
  for (int64_t i = 0; i < num_reqs; ++i)
    nat_host[static_cast<size_t>(i)] = static_cast<float>(np[i]);
  ttnn::Tensor nat_f32 = UploadTensor(
      std::move(nat_host), ttnn::Shape({static_cast<uint32_t>(num_reqs), 1}),
      ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
  ttnn::Tensor off_col =
      ttnn::gather(nat_f32, /*dim=*/0, e.owner_idx, /*sparse_grad=*/false, std::nullopt);
  off_col = ttnn::subtract(std::move(off_col), 1.0f);  // [R, 1] f32
  // state-window gather index: row j reads shadow row off_c + j — the
  // gather's per-column semantics carries the column, so the index VALUE is
  // the ROW number alone (flat row*R+col addressing would read rows the
  // shadow does not have). Built as width blocks [R, 1] -> [1, R],
  // concatenated j-major.
  ttnn::Tensor stat_idx;
  {
    std::vector<ttnn::Tensor> rows;
    rows.reserve(static_cast<size_t>(width));
    for (int64_t j = 0; j < width; ++j) {
      ttnn::Tensor rj = ttnn::add(off_col, static_cast<float>(j));    // [R, 1]
      rows.push_back(ttnn::transpose(std::move(rj), 0, 1));           // [1, R]
    }
    stat_idx = ttnn::typecast(ttnn::concat(std::move(rows), /*dim=*/0),
                              ttnn::DataType::UINT32);  // [width, R]
  }
  ttnn::Tensor statG =
      ttnn::gather(T, /*dim=*/0, stat_idx, /*sparse_grad=*/false, std::nullopt);
  spec_dump("statG", statG);
  // ---- the stream + the MAC ----------------------------------------------
  // xT stays alive: the state commit's SRC concat reads it again below (the
  // handles are refcounted — pass it by value here).
  ttnn::Tensor S = ttnn::concat(std::vector<ttnn::Tensor>{std::move(statG), xT},
                                /*dim=*/0);  // [width + T, R]
  const uint32_t utaps = static_cast<uint32_t>(k);
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
  ttnn::Tensor acc = bias != nullptr
                         ? CachedTile(
                               bias->data, R, 0, 0,
                               [bias, R, uslots, uc] {
                                 EnsureHost(*bias);
                                 std::vector<float> bh(static_cast<size_t>(R));
                                 for (uint32_t s = 0; s < uslots; ++s)
                                   for (uint32_t c = 0; c < uc; ++c)
                                     bh[static_cast<size_t>(s) * uc + c] =
                                         LoadElemF32(*bias, c);
                                 return bh;
                               },
                               ttnn::Shape({1, R}), device)
                         : CachedAccBase(R, device);
  // Exact accumulation in the oracle's order: bias first, then taps 0..K-1
  // (one exact multiply + one add per tap — the decode kernel's discipline).
  for (int64_t j = 0; j < k; ++j) {
    ttnn::Tensor mac_j = ttnn::slice(
        e.mac_idx, ttsl::SmallVector<uint32_t>{static_cast<uint32_t>(j * tokens), 0u},
        ttsl::SmallVector<uint32_t>{static_cast<uint32_t>((j + 1) * tokens),
                                    static_cast<uint32_t>(R)},
        ttsl::SmallVector<uint32_t>{1u, 1u});  // [T, R] u32
    ttnn::Tensor src_j =
        ttnn::gather(S, /*dim=*/0, mac_j, /*sparse_grad=*/false, std::nullopt);
    ttnn::Tensor wj = ttnn::slice(
        Wt, ttsl::SmallVector<uint32_t>{static_cast<uint32_t>(j), 0u},
        ttsl::SmallVector<uint32_t>{static_cast<uint32_t>(j + 1),
                                    static_cast<uint32_t>(R)},
        ttsl::SmallVector<uint32_t>{1u, 1u});  // [1, R]
    acc = ttnn::add(std::move(acc), ttnn::multiply(std::move(src_j), std::move(wj)));
  }
  spec_dump("acc", acc);
  ttnn::Tensor y = args.silu_activation ? ttnn::silu(acc) : acc;  // [T, R]
  spec_dump("y", y);
  // VT_TT_GDN_SPEC_DEBUG (the GdnSpecDecode probe's twin): the conv spec
  // step's x stream + MAC result checksums — distinguishes a zero x stream
  // from a zero state window from a zero MAC.
  if (const char* dbg = std::getenv("VT_TT_GDN_SPEC_DEBUG");
      dbg != nullptr && dbg[0] != '\0' && !tt_capture_active()) {
    std::vector<float> yv = y.to_vector<float>();
    double y_acc = 0.0;
    uint64_t y_nan = 0;
    for (float f : yv) {
      y_acc += f > 0.0f ? f : -f;
      if (std::isnan(f)) ++y_nan;
    }
    std::vector<float> xv = xT.to_vector<float>();
    double x_acc = 0.0;
    uint64_t x_nan = 0;
    for (float f : xv) {
      x_acc += f > 0.0f ? f : -f;
      if (std::isnan(f)) ++x_nan;
    }
    std::vector<float> tv_ = T.to_vector<float>();
    double t_acc = 0.0;
    for (float f : tv_) t_acc += f > 0.0f ? f : -f;
    std::fprintf(stderr,
                 "[CONV-SPEC] T=%lld nreq=%lld conv_state=%p y_n=%zu "
                 "y_asum=%.6g x_asum=%.6g T_asum=%.6g x_nan=%llu y_nan=%llu "
                 "x_shadow=%d x_dt=%d x_ptr=%p\n",
                 static_cast<long long>(tokens),
                 static_cast<long long>(num_reqs), conv_state.data, yv.size(),
                 y_acc, x_acc, t_acc,
                 static_cast<unsigned long long>(x_nan),
                 static_cast<unsigned long long>(y_nan),
                 x_from_shadow ? 1 : 0, static_cast<int>(x.dtype),
                 static_cast<const void*>(x.data));
    // The NaN positions (row, col) of xT and y — one line per call while any
    // NaN is present; the row maps to the token, the col to slot/channel.
    if (x_nan > 0 || y_nan > 0) {
      std::string xp, yp;
      int shown = 0;
      for (size_t i = 0; i < xv.size() && shown < 6; ++i)
        if (std::isnan(xv[i])) {
          xp += " [" + std::to_string(i / R) + "," + std::to_string(i % R) + "]";
          ++shown;
        }
      shown = 0;
      for (size_t i = 0; i < yv.size() && shown < 6; ++i)
        if (std::isnan(yv[i])) {
          yp += " [" + std::to_string(i / R) + "," + std::to_string(i % R) + "]";
          ++shown;
        }
      std::fprintf(stderr, "[CONV-SPEC-NAN] xT:%s y:%s\n", xp.c_str(),
                   yp.c_str());
      // The three-way comparison: the SCATTERED xT vs the RAW served shadow
      // (pre-scatter) vs a direct EnsureDevice2D readback of x — attributes
      // the NaN to the serve, the shadow, or the scatter itself.
      {
        uint64_t raw_nan = 0;
        double raw_acc = 0.0;
        {
          std::lock_guard<std::mutex> g(SlotMutex());
          BufferSlot* s = FindSlot(x.data);
          if (s != nullptr && s->device_current && !s->host_current &&
              s->device.has_value()) {
            std::fprintf(stderr,
                         "[CONV-SPEC-NAN-SLOT] ptr=%p dev=%ux%u bytes=%zu "
                         "dt=%d reserved=%d persist=%d\n",
                         static_cast<const void*>(x.data), s->dev_rows,
                         s->dev_cols, s->bytes,
                         static_cast<int>(s->device->dtype()),
                         s->device_reserved ? 1 : 0,
                         s->persistent.has_value() ? 1 : 0);
            std::vector<float> rv = s->device->to_vector<float>();
            for (float f : rv) {
              raw_acc += f > 0.0f ? f : -f;
              if (std::isnan(f)) ++raw_nan;
            }
            std::fprintf(stderr,
                         "[CONV-SPEC-NAN-RAW] raw_n=%zu raw_nan=%llu "
                         "raw_asum=%.6g\n",
                         rv.size(), static_cast<unsigned long long>(raw_nan),
                         raw_acc);
          }
        }
        std::vector<float> xv2 =
            vt::tenstorrent::DebugDeviceReadbackF32(q_unused_probe, x);
        double x2_acc = 0.0;
        uint64_t x2_nan = 0;
        for (float f : xv2) {
          x2_acc += f > 0.0f ? f : -f;
          if (std::isnan(f)) ++x2_nan;
        }
        std::fprintf(stderr,
                     "[CONV-SPEC-NAN-X2] x2_n=%zu x2_nan=%llu x2_asum=%.6g\n",
                     xv2.size(), static_cast<unsigned long long>(x2_nan),
                     x2_acc);
      }
    }
    std::fflush(stderr);
  }
  // ---- the out select ------------------------------------------------------
  if (!has_null) {
    // yT[r, t] = y[t, r]; the select reads row s_i*C + c per (c, t) — one
    // exact dim-0 gather with the cached per-column index, then the
    // transpose back to the out's [T, C] order.
    ttnn::Tensor yT = ttnn::transpose(y, 0, 1);  // [R, T]
    ttnn::Tensor sel = ttnn::gather(yT, /*dim=*/0, e.osel_idx,
                                    /*sparse_grad=*/false, std::nullopt);
    ttnn::Tensor o = ttnn::transpose(sel, 0, 1);  // [T, C]
    spec_dump("o_precommit", o);
    CommitDeviceLogical2D(out, std::move(o), utok, uc);
    if (spec_dump_on && spec_dump_call < 2) {
      auto ov = vt::tenstorrent::DebugDeviceReadbackF32(q_unused_probe, out);
      std::FILE* f = std::fopen(
          (std::string(spec_dump_dir) + "/cs" +
           std::to_string(spec_dump_call) + "_out.f32").c_str(), "wb");
      if (f) { std::fwrite(ov.data(), 4, ov.size(), f); std::fclose(f); }
    }
  } else {
    // Eager-only (NULL rows never capture): the oracle leaves a NULL
    // request's out rows untouched, so the select runs on the host against
    // the out buffer's current bytes.
    EnsureHost(out);
    std::vector<float> yv = y.to_vector<float>();
    for (int64_t t = 0; t < tokens; ++t) {
      int64_t req = num_reqs - 1;
      for (int64_t i = 0; i < num_reqs; ++i) {
        if (t >= csv[static_cast<size_t>(i)] && t < csv[static_cast<size_t>(i + 1)]) {
          req = i;
          break;
        }
      }
      const int32_t s = idxv[static_cast<size_t>(req)];
      if (s < 0) continue;  // NULL request: out keeps its previous bytes
      for (int64_t c = 0; c < c_dim; ++c)
        StoreElemF32(out, t * c_dim + c,
                     yv[static_cast<size_t>(t * R + s * c_dim + c)]);
    }
    CommitHost(out);
  }
  // ---- the state commit ----------------------------------------------------
  // SRC = [T0 | zeros | xT]: the zeros block absorbs the out-of-range state
  // reads (the oracle's `src < state_len` guard), the x part follows it.
  ttnn::Tensor SRC = ttnn::concat(
      std::vector<ttnn::Tensor>{T, e.zeros_blk, xT}, /*dim=*/0);
  // The roll's state-part rows (device-derived from num_accepted — the
  // value that changes every verify step): row j reads SRC row off_c + j + 1
  // (a row number; SRC = [T0 | zeros | xT], so a read past the shadow's taps
  // lands in the zeros block — the oracle's `src < state_len ? srow[src] :
  // 0` guard, no clamp op needed).
  ttnn::Tensor roll_state;
  {
    std::vector<ttnn::Tensor> rows;
    rows.reserve(static_cast<size_t>(state_len));
    for (int64_t j = 0; j < state_len; ++j) {
      ttnn::Tensor rj = ttnn::add(off_col, static_cast<float>(j + 1));  // [R, 1]
      rows.push_back(ttnn::transpose(std::move(rj), 0, 1));  // [1, R]
    }
    roll_state = ttnn::concat(std::move(rows), /*dim=*/0);  // [sl, R] f32
  }
  // The exact 0/1 mux: (state - x) * m + x — m = 1 reads the state part,
  // m = 0 the x part; integral-valued f32 makes every product and sum exact.
  ttnn::Tensor roll_f = ttnn::add(
      ttnn::multiply(ttnn::subtract(std::move(roll_state), e.roll_x), e.roll_m),
      e.roll_x);
  ttnn::Tensor rolled = ttnn::gather(
      SRC, /*dim=*/0, ttnn::typecast(std::move(roll_f), ttnn::DataType::UINT32),
      /*sparse_grad=*/false, std::nullopt);  // [sl, R]
  // The non-participating slots keep their bytes (the exact 0/1 blend back
  // to the pre-step shadow's state rows).
  ttnn::Tensor T_state = ttnn::slice(
      T, ttsl::SmallVector<uint32_t>{0u, 0u},
      ttsl::SmallVector<uint32_t>{usl, static_cast<uint32_t>(R)},
      ttsl::SmallVector<uint32_t>{1u, 1u});  // [sl, R]
  rolled = ttnn::add(ttnn::multiply(std::move(rolled), e.mv),
                     ttnn::multiply(std::move(T_state), e.mk));
  // bf16 STORAGE semantics (SupportsCompressedConvState, cuda_backend.cu:119):
  // every value that enters the cache rounds through bf16 at the store
  // boundary (the decode kernel's discipline, tenstorrent_gdn.cpp:2437).
  if (conv_state.dtype == DType::kBF16) {
    rolled = ttnn::typecast(ttnn::typecast(std::move(rolled), ttnn::DataType::BFLOAT16),
                            ttnn::DataType::FLOAT32);
  }
  // Rebuild the [sl+1, R] shadow: the committed rows + a fresh scratch row
  // (never read before the next step overwrites it).
  ttnn::Tensor new_shadow = ttnn::concat(
      std::vector<ttnn::Tensor>{std::move(rolled), CachedAccBase(R, device)},
      /*dim=*/0);
  // TT-GDN-REGION-REPLAY: commit INTO the persistent shadow (ttnn::copy —
  // the capture-safe preallocated-destination write) so the trace's next
  // slice reads the advanced window at the SAME address.
  if (new_shadow.dtype() == T0.dtype() && new_shadow.layout() == T0.layout()) {
    const auto sr = new_shadow.logical_shape(), s0 = T0.logical_shape();
    bool same = sr.rank() == s0.rank();
    for (uint32_t r = 0; same && r < sr.rank(); ++r) same = sr[r] == s0[r];
    if (same) {
      ttnn::copy(new_shadow, T0);
      CommitConvTransposed(conv_state, std::move(T0), uslots, uc, usl);
      return;
    }
  }
  CommitConvTransposed(conv_state, std::move(new_shadow), uslots, uc, usl);
}

// kGdnSpecDecode (QWEN38-DFLASH2-STAGE-C W4; cpu_ops.cpp GdnSpecDecodeKernel,
// the fused_sigmoid_gating.py IS_SPEC_DECODING + INPLACE_FINAL_STATE port).
// The verify's GDN segment: every request carries its K+1-row spec block, the
// initial state comes from column num_accepted-1 (the snapshot taken after the
// last ACCEPTED token of the previous step), and the recurrence runs T
// sequential composed steps — the SAME GdnDecodeStepComposed the single-row
// kGdnDecode kernel runs — with each timestep's post-token state snapshotted
// into its own slot column (the SSM rollback the stage-B commit-or-restore
// composes with: a rejected verify tail restores the pre-step shadow this
// kernel's commit replaced; the accepted tail keeps it).
//
// Device composition per request: the initial state row is ONE exact gather
// from the cache shadow (GatherRowsExact — the one-hot matmul alternative
// rounds its inputs to tf32, measured 2.4e-4, and the initial state's error
// compounds through the delta-rule correction the wide-range arm pins); the
// per-timestep q/k/v/g/beta are CONTIGUOUS row slices of the served streams
// (request i's tokens are rows [cs_i, cs_i+1) in batch order — the pure-spec
// contract), so the recurrence needs no per-timestep index build at all; the
// per-timestep snapshots accumulate as device rows and land in ONE
// split-block scatter at the end (ScatterRowsDevice, last-writer-wins — the
// oracle's sequential store order).
//
// Metadata discipline (the conv spec kernel's, tenstorrent_gdn.cpp:2799): the
// slot columns are step-stable (the engine's fixed per-request slot window)
// but the INIT selection is acceptance-derived — num_accepted_tokens is the
// one metadata whose content changes every verify step, so the eager arm
// reads the host masters and the CAPTURED arm refuses by name: a capture-time
// upload would replay one step's acceptance forever. The named gap is the
// driver-side persistent-device refresh; until it lands the spec verify runs
// on the eager arm (VLLM_CPP_CUDAGRAPH=0) — the arm the token-identity gate
// needs.
void GdnSpecDecodeKernel(Queue&, Tensor& out, const Tensor& q_in, const Tensor& k,
                         const Tensor& v, const Tensor& g, const Tensor& beta,
                         Tensor& state, const Tensor& cu_seqlens,
                         const Tensor& state_indices,
                         const Tensor& num_accepted_tokens, const GdnArgs& args) {
  TT_OP_TRACE("GdnSpecDecode");
  GdnDecodeSteps().fetch_add(1, std::memory_order_relaxed);
  const int64_t T = q_in.shape[0];
  const int64_t hv = state.shape[1], dv = state.shape[2], dk = state.shape[3];
  const int64_t hk = q_in.shape[1];
  const int64_t slots = state.shape[0];
  const int64_t num_reqs = state_indices.shape[0];
  const int64_t num_cols = state_indices.shape[1];
  if (T == 0) return;  // empty stream: no-op, cache as-is

  VT_CHECK(!tt_capture_active(),
           "tenstorrent gdn_spec_decode: the acceptance metadata has no "
           "persistent device refresh on this backend — the captured arm "
           "would replay one step's num_accepted forever. Named gap: the "
           "driver's TT spec-metadata in-place device refresh (the "
           "WarmRopeCosSin copy_to_device pattern); until it lands the spec "
           "verify runs on the eager arm (VLLM_CPP_CUDAGRAPH=0).");
  EnsureHost(cu_seqlens);
  EnsureHost(num_accepted_tokens);
  EnsureHost(state_indices);
  const int32_t* csp = cu_seqlens.Ptr<int32_t>();
  const int32_t* natp = num_accepted_tokens.Ptr<int32_t>();
  const int32_t* sidx = state_indices.Ptr<int32_t>();
  std::vector<int32_t> csv(csp, csp + num_reqs + 1);
  VT_CHECK(csv[0] == 0 && csv[num_reqs] == T,
           "tenstorrent gdn_spec_decode: bad cu_seqlens bounds");
  std::vector<int64_t> lo(static_cast<size_t>(num_reqs)),
      tlen(static_cast<size_t>(num_reqs));
  std::vector<int32_t> init(static_cast<size_t>(num_reqs));
  for (int64_t i = 0; i < num_reqs; ++i) {
    lo[static_cast<size_t>(i)] = csv[static_cast<size_t>(i)];
    tlen[static_cast<size_t>(i)] =
        csv[static_cast<size_t>(i + 1)] - csv[static_cast<size_t>(i)];
    VT_CHECK(tlen[static_cast<size_t>(i)] >= 0 &&
                 tlen[static_cast<size_t>(i)] <= num_cols,
             "tenstorrent gdn_spec_decode: query length exceeds the spec "
             "slot count");
    VT_CHECK(natp[static_cast<size_t>(i)] >= 1 &&
                 natp[static_cast<size_t>(i)] <= num_cols,
             "tenstorrent gdn_spec_decode: num_accepted_tokens out of range");
    const int32_t s = sidx[static_cast<size_t>(i * num_cols +
                                              (natp[static_cast<size_t>(i)] - 1))];
    VT_CHECK(s < slots, "tenstorrent gdn_spec_decode: state index out of range");
    init[static_cast<size_t>(i)] = s;
  }

  MeshDevice& device = SharedMeshDevice();
  const uint32_t uhv = static_cast<uint32_t>(hv), udv = static_cast<uint32_t>(dv),
                  udk = static_cast<uint32_t>(dk);
  const int64_t row_elems = hv * dv * dk;
  const uint32_t sf = static_cast<uint32_t>(SplitFactor(row_elems, /*esz=*/4));
  const uint32_t blk = static_cast<uint32_t>(row_elems / sf);

  // Device-pure staging (the decode kernel's ServeActF32 discipline: the
  // projections commit their outputs device-side, so the activations are
  // served from their current shadows; the eager arm keeps its host-staging
  // fallback). q/k expand to Hv heads through the proven dim-0 gather — the
  // head map is step-constant per geometry and content-keyed.
  ttnn::Tensor q2d = ServeActF32(q_in, static_cast<uint32_t>(T * hk), udk, "q",
                                 device);
  ttnn::Tensor k2d =
      ServeActF32(k, static_cast<uint32_t>(T * hk), udk, "k", device);
  ttnn::Tensor v2d = ServeActF32(v, static_cast<uint32_t>(T * hv), udv, "v", device);
  ttnn::Tensor g2d = ServeActF32(g, static_cast<uint32_t>(T * hv), 1, "g", device);
  ttnn::Tensor b2d =
      ServeActF32(beta, static_cast<uint32_t>(T * hv), 1, "beta", device);
  ttnn::Tensor q_exp = q2d, k_exp = k2d;
  if (hv != hk) {
    ttnn::Tensor head_map =
        CachedHeadMapIdx(static_cast<uint64_t>(T), static_cast<uint64_t>(hv),
                         static_cast<uint64_t>(hk), static_cast<uint64_t>(dk),
                         device);
    q_exp = ttnn::gather(q2d, /*dim=*/0, head_map, /*sparse_grad=*/false,
                         std::nullopt);
    k_exp = ttnn::gather(k2d, /*dim=*/0, head_map, /*sparse_grad=*/false,
                         std::nullopt);
  }

  // The split shadow ([slots*F, blk]) and the initial-state gather: one exact
  // row gather per request (NULL rows gather zeros + the mask — the oracle's
  // :114-115 skip). The acceptance-derived selection is why this is a
  // per-call gather, not a content-keyed entry. The gather runs in the
  // shadow's SPLIT geometry — each request's slot expands into its sf block
  // rows (flat bytes identical either way), and the result views as
  // [ns*Hv, Dv, Dk] (the decode kernel's one-hot matmul produces the same
  // [B*sf, blk] -> [bh, dv, dk] reshape).
  ttnn::Tensor cache2d = EnsureGdnCacheDevice(state, slots, row_elems, device);
  std::vector<int32_t> init_blocks;
  init_blocks.reserve(init.size() * static_cast<size_t>(sf));
  for (int32_t ix : init)
    for (uint32_t j = 0; j < sf; ++j)
      init_blocks.push_back(ix < 0 ? -1
                                   : ix * static_cast<int32_t>(sf) +
                                         static_cast<int32_t>(j));
  ttnn::Tensor S_all =
      GatherRowsExact(cache2d, init_blocks, blk, device);
  S_all = CaptureSafeReshape(
      S_all, ttnn::Shape({static_cast<uint32_t>(init.size() * hv), udv, udk}));
  // W21 (PROBE 1 + 2): the recurrence's unprobed inputs as raw f32 — the
  // gathered initial state (S_all) and the served g/beta (the BA outputs).
  // The composed CPU reference consumes exactly these files.
  static const char* spec_dump_dir2 = std::getenv("VT_TT_SPEC_DUMP");
  static std::atomic<int64_t> spec_dump_seq2{0};
  if (spec_dump_dir2 != nullptr && spec_dump_dir2[0] != '\0' &&
      !tt_capture_active()) {
    const int64_t dc = spec_dump_seq2.fetch_add(1, std::memory_order_relaxed);
    if (dc < 2) {
      auto dump1 = [&](const char* name, const ttnn::Tensor& t) {
        auto v = t.to_vector<float>();
        std::FILE* f = std::fopen(
            (std::string(spec_dump_dir2) + "/gs" + std::to_string(dc) + "_" +
             name + ".f32").c_str(), "wb");
        if (f) { std::fwrite(v.data(), 4, v.size(), f); std::fclose(f); }
      };
      dump1("S_all", S_all);
      dump1("g", g2d);
      dump1("beta", b2d);
      dump1("q", q_exp);
      dump1("k", k_exp);
      dump1("v", v2d);
      // The cache shadow's slot-0 split rows, read DIRECTLY (no gather) —
      // splits "the scatter/commit wrote wrong" from "the gather read wrong".
      dump1("slot0_direct", ttnn::slice(
          cache2d, ttsl::SmallVector<uint32_t>{0u, 0u},
          ttsl::SmallVector<uint32_t>{static_cast<uint32_t>(sf),
                                      static_cast<uint32_t>(blk)},
          ttsl::SmallVector<uint32_t>{1u, 1u}));
    }
  }

  // The recurrence: per request, T sequential composed steps over that
  // request's [Hv, Dv, Dk] state block; per-timestep activations are
  // contiguous row slices (request i's tokens are rows [cs_i, cs_i+1) in
  // batch order, each token's Hv head rows contiguous after the expand).
  std::vector<ttnn::Tensor> o_parts;    // per (request, timestep): [Hv, Dv, 1]
  std::vector<ttnn::Tensor> snap_rows;  // per snapshot: [sf, blk]
  std::vector<uint32_t> snap_bids;      // expanded split-block ids
  for (int64_t i = 0; i < num_reqs; ++i) {
    const int64_t Ti = tlen[static_cast<size_t>(i)];
    if (Ti == 0) continue;  // T == 0 early-out (the oracle's :76-78)
    if (init[static_cast<size_t>(i)] < 0) {
      // NULL init: the oracle zeroes the request's out rows and skips the
      // whole request — no snapshots even from live columns (:114-115).
      o_parts.push_back(ttnn::zeros(
          tt::tt_metal::Shape({static_cast<uint32_t>(Ti * hv), udv, 1u}),
          ttnn::DataType::FLOAT32, ttnn::Layout::TILE, std::ref(device)));
      continue;
    }
    ttnn::Tensor S = ttnn::slice(
        S_all,
        ttsl::SmallVector<uint32_t>{static_cast<uint32_t>(i * hv), 0u, 0u},
        ttsl::SmallVector<uint32_t>{static_cast<uint32_t>((i + 1) * hv), udv,
                                    udk},
        ttsl::SmallVector<uint32_t>{1u, 1u, 1u});  // [Hv, Dv, Dk]
    for (int64_t p = 0; p < Ti; ++p) {
      const int64_t t = lo[static_cast<size_t>(i)] + p;
      const uint32_t r0 = static_cast<uint32_t>(t * hv);
      const uint32_t r1 = static_cast<uint32_t>((t + 1) * hv);
      ttnn::Tensor qp = ttnn::slice(
          q_exp, ttsl::SmallVector<uint32_t>{r0, 0u},
          ttsl::SmallVector<uint32_t>{r1, udk}, ttsl::SmallVector<uint32_t>{1u, 1u});
      ttnn::Tensor kp = ttnn::slice(
          k_exp, ttsl::SmallVector<uint32_t>{r0, 0u},
          ttsl::SmallVector<uint32_t>{r1, udk}, ttsl::SmallVector<uint32_t>{1u, 1u});
      ttnn::Tensor vp = ttnn::slice(
          v2d, ttsl::SmallVector<uint32_t>{r0, 0u},
          ttsl::SmallVector<uint32_t>{r1, udv}, ttsl::SmallVector<uint32_t>{1u, 1u});
      ttnn::Tensor gp = ttnn::slice(
          g2d, ttsl::SmallVector<uint32_t>{r0, 0u},
          ttsl::SmallVector<uint32_t>{r1, 1u}, ttsl::SmallVector<uint32_t>{1u, 1u});
      ttnn::Tensor bp = ttnn::slice(
          b2d, ttsl::SmallVector<uint32_t>{r0, 0u},
          ttsl::SmallVector<uint32_t>{r1, 1u}, ttsl::SmallVector<uint32_t>{1u, 1u});
      DecodeStepResult r = GdnDecodeStepComposed(S, qp, kp, vp, gp, bp, args.scale,
                                                 uhv, udk, udv);
      o_parts.push_back(std::move(r.o));
      // Snapshot AFTER this timestep into its own slot column (the oracle's
      // :156-166). A NULL column skips only its store.
      S = std::move(r.state);
      const int32_t slot =
          sidx[static_cast<size_t>(i * num_cols + p)];
      if (slot >= 0) {
        snap_rows.push_back(
            CaptureSafeReshape(S, ttnn::Shape({sf, blk})));
        for (uint32_t j = 0; j < sf; ++j)
          snap_bids.push_back(static_cast<uint32_t>(slot) * sf + j);
      }
    }
  }

  // The out: the (request, timestep) order IS token order (cs is the prefix
  // sum) — one concat, then the logical-2D commit the decode kernel uses.
  ttnn::Tensor o_all = o_parts.size() == 1 ? std::move(o_parts[0])
                                           : ttnn::concat(o_parts, /*dim=*/0);
  // W21: the recurrence's OUTPUT as raw f32 (the [GDN-SPEC] asum's bytes).
  if (spec_dump_dir2 != nullptr && spec_dump_dir2[0] != '\0' &&
      !tt_capture_active()) {
    static std::atomic<int64_t> o_seq{0};
    const int64_t dc = o_seq.fetch_add(1, std::memory_order_relaxed);
    if (dc < 2) {
      auto v = o_all.to_vector<float>();
      std::FILE* f = std::fopen(
          (std::string(spec_dump_dir2) + "/gs" + std::to_string(dc) + "_o.f32")
              .c_str(), "wb");
      if (f) { std::fwrite(v.data(), 4, v.size(), f); std::fclose(f); }
    }
  }
  // VT_TT_GDN_SPEC_DEBUG: the spec step's metadata + result checksums, one
  // line per call (eager only — a download is the read the trace refuses).
  // The frozen-recurrence signature is the GDN-STATE-PROBE's: one IDENTICAL
  // out checksum at every step means the recurrence never advanced.
  if (const char* dbg = std::getenv("VT_TT_GDN_SPEC_DEBUG");
      dbg != nullptr && dbg[0] != '\0' && !tt_capture_active()) {
    std::vector<float> ov = o_all.to_vector<float>();
    double acc = 0.0;
    uint64_t fnv = 1469598103934665603ULL;
    for (float f : ov) {
      uint32_t bits;
      std::memcpy(&bits, &f, sizeof(bits));
      fnv = (fnv ^ bits) * 1099511628211ULL;
      acc += f > 0.0f ? f : -f;
    }
    // The gathered initial state + the q activation: distinguishes a zero
    // STATE gather from zero ACTIVATIONS from a zero result.
    std::vector<float> sv = S_all.to_vector<float>();
    double s_acc = 0.0;
    for (float f : sv) s_acc += f > 0.0f ? f : -f;
    std::vector<float> qv = q_exp.to_vector<float>();
    double q_acc = 0.0;
    uint64_t q_nan = 0;
    for (float f : qv) {
      q_acc += f > 0.0f ? f : -f;
      if (std::isnan(f)) ++q_nan;
    }
    std::vector<float> vv = v2d.to_vector<float>();
    double v_acc = 0.0;
    uint64_t v_nan = 0;
    for (float f : vv) {
      v_acc += f > 0.0f ? f : -f;
      if (std::isnan(f)) ++v_nan;
    }
    std::fprintf(stderr,
                 "[GDN-SPEC] T=%lld nreq=%lld nc=%lld nat0=%d init0=%d "
                 "slots=%lld cache=%p out_n=%zu fnv=%016llx asum=%a "
                 "s_asum=%.6g q_asum=%.6g v_asum=%.6g q_nan=%llu v_nan=%llu\n",
                 static_cast<long long>(T), static_cast<long long>(num_reqs),
                 static_cast<long long>(num_cols),
                 num_reqs > 0 ? natp[0] : -1,
                 num_reqs > 0 ? init[static_cast<size_t>(num_reqs - 1)] : -1,
                 static_cast<long long>(slots), state.data, ov.size(),
                 static_cast<unsigned long long>(fnv), acc, s_acc, q_acc, v_acc,
                 static_cast<unsigned long long>(q_nan),
                 static_cast<unsigned long long>(v_nan));
    // W21: the shadow's residency state for the state tensor — the fast path
    // (a resident f32 shadow) vs the cold path (a fresh upload of the host
    // bytes). A cold path per call with a changing host pointer = the state
    // never persists in the shadow registry.
    {
      std::lock_guard<std::mutex> g(SlotMutex());
      BufferSlot* s = FindSlot(state.data);
      std::fprintf(stderr,
                   "[GDN-SPEC-SLOT] ptr=%p found=%d devc=%d hostc=%d "
                   "dev=%ux%u persist=%d addr=%llx\n",
                   state.data, s != nullptr ? 1 : 0,
                   s != nullptr && s->device_current ? 1 : 0,
                   s != nullptr && s->host_current ? 1 : 0,
                   s != nullptr ? s->dev_rows : 0u,
                   s != nullptr ? s->dev_cols : 0u,
                   s != nullptr && s->persistent.has_value() ? 1 : 0,
                   s != nullptr && s->device.has_value()
                       ? static_cast<unsigned long long>(
                             s->device->mesh_buffer().address())
                       : 0ULL);
      std::fflush(stderr);
    }
    std::fflush(stderr);
  }
  CommitDeviceLogical2D(
      out,
      CaptureSafeReshape(o_all, ttnn::Shape({static_cast<uint32_t>(T), uhv, udv})),
      static_cast<uint32_t>(T * hv), udv);

  // ONE split-block scatter: every snapshot row lands in its slot in the
  // oracle's store order (last-writer-wins on duplicates). No snapshots —
  // all requests NULL/T=0 — leaves the cache untouched.
  if (!snap_bids.empty()) {
    ttnn::Tensor rows_all = snap_rows.size() == 1
                                ? std::move(snap_rows[0])
                                : ttnn::concat(snap_rows, /*dim=*/0);
    // bf16 STORAGE semantics (the decode kernel's discipline): the state a
    // step STORES rounds through bf16; the in-step register chain stays f32 —
    // exactly the oracle's StoreF32 boundary.
    if (state.dtype == DType::kBF16) {
      rows_all = ttnn::typecast(ttnn::typecast(std::move(rows_all),
                                               ttnn::DataType::BFLOAT16),
                                ttnn::DataType::FLOAT32);
    }
    const uint32_t unb = static_cast<uint32_t>(snap_bids.size());
    // W21 IDENTITY FAST PATH: the snapshots' block ids are the identity
    // permutation in the gated configuration (the per-timestep slots are the
    // request's consecutive [base, base+k] range, expanded by the split
    // factor) — the scatter is then a pure copy of rows_all, and the fill
    // (whose multi-replacement miswrite is the W21 finding) is bypassed.
    bool snap_identity = unb > 0;
    for (uint32_t i = 0; i < unb && snap_identity; ++i)
      if (snap_bids[static_cast<size_t>(i)] != i) snap_identity = false;
    const uint32_t total_block_rows = static_cast<uint32_t>(slots) * sf;
    ttnn::Tensor next =
        snap_identity && unb == total_block_rows
            ? (rows_all.dtype() == cache2d.dtype() &&
                       rows_all.layout() == cache2d.layout()
                   ? rows_all
                   : ttnn::typecast(std::move(rows_all), cache2d.dtype()))
            : snap_identity
                  ? [&] {
                      // Partial identity (the leading block rows): the rows
                      // land at cache rows [0, unb) and the tail keeps its
                      // bytes — one exact concat (the W21 concat-side dumps
                      // pin it faithful), no fill.
                      ttnn::Tensor tail = ttnn::slice(
                          cache2d, ttsl::SmallVector<uint32_t>{unb, 0u},
                          ttsl::SmallVector<uint32_t>{total_block_rows,
                                                      static_cast<uint32_t>(row_elems / sf)},
                          ttsl::SmallVector<uint32_t>{1u, 1u});
                      return ttnn::concat(
                          std::vector<ttnn::Tensor>{rows_all, tail}, /*dim=*/0);
                    }()
                  : [&] {
                      ttnn::Tensor bid_dev =
                          UploadIdxU32(std::move(snap_bids), ttnn::Shape({unb}),
                                       ttnn::Layout::ROW_MAJOR, device);
                      return ScatterRowsDevice(cache2d, bid_dev, rows_all, slots,
                                               row_elems, sf);
                    }();
    // W21: the scatter's two sides at the first calls — the snapshot rows as
    // produced, and the cache shadow AFTER the commit copy.
    if (spec_dump_dir2 != nullptr && spec_dump_dir2[0] != '\0' &&
        !tt_capture_active()) {
      static std::atomic<int64_t> sc_seq{0};
      const int64_t dc = sc_seq.fetch_add(1, std::memory_order_relaxed);
      if (dc < 2) {
        auto rv = next.to_vector<float>();
        std::FILE* f = std::fopen(
            (std::string(spec_dump_dir2) + "/gs" + std::to_string(dc) +
             "_snap_next.f32").c_str(), "wb");
        if (f) { std::fwrite(rv.data(), 4, rv.size(), f); std::fclose(f); }
        auto cv = cache2d.to_vector<float>();
        f = std::fopen((std::string(spec_dump_dir2) + "/gs" +
                        std::to_string(dc) + "_cache_post.f32").c_str(), "wb");
        if (f) { std::fwrite(cv.data(), 4, cv.size(), f); std::fclose(f); }
      }
    }
    // TT-GDN-REGION-REPLAY: commit INTO the persistent shadow (ttnn::copy —
    // the capture-safe preallocated-destination write) so the next step's
    // gather reads the advanced slots at the SAME address.
    const auto same_spec = [](const ttnn::Tensor& a, const ttnn::Tensor& b) {
      if (a.dtype() != b.dtype() || a.layout() != b.layout()) return false;
      const auto sa = a.logical_shape(), sb = b.logical_shape();
      if (sa.rank() != sb.rank()) return false;
      for (uint32_t r = 0; r < sa.rank(); ++r)
        if (sa[r] != sb[r]) return false;
      return true;
    };
    if (same_spec(next, cache2d)) {
      // W21: the spec kernel is EAGER-ONLY (the capture refusal above), so the
      // TT-GDN-REGION-REPLAY in-place copy is not required here — and the W21
      // evidence shows the copy DOES NOT LAND in the engine (the cache shadow
      // reads its pre-copy content after ttnn::copy; the snapshots' bytes are
      // lost). Install the committed tensor as the slot's shadow directly:
      // CommitDeviceLogical2D replaces s->device, so the next call's
      // EnsureGdnCacheDevice fast path serves exactly these bytes.
      CommitDeviceLogical2D(state, std::move(next),
                            static_cast<uint32_t>(slots),
                            static_cast<uint32_t>(row_elems));
    } else {
      CommitDeviceLogical2D(state, std::move(next),
                            static_cast<uint32_t>(slots),
                            static_cast<uint32_t>(row_elems));
    }
  }
}



// W21 (the interleaved-op harness): the engine's EXACT multi-replacement
// indexed_fill form (the conv-spec x scatter's geometry) as a test surface —
// the caller passes the base, the values, and the ids; the fill runs with the
// same rank-4 views / dim-0 / RM discipline the kernel uses, and the flat
// result comes back for byte comparison. Defined in the TT-gated gdn TU,
// inline no-op otherwise (tenstorrent_device.h).
std::vector<float> ConvSpecFillForTest(const std::vector<float>& base,
                                       const std::vector<float>& vals,
                                       const std::vector<uint32_t>& ids,
                                       int64_t base_rows, int64_t cols) {
  MeshDevice& device = SharedMeshDevice();
  const uint32_t rows_b = static_cast<uint32_t>(base_rows);
  const uint32_t uc = static_cast<uint32_t>(cols);
  const uint32_t nv = static_cast<uint32_t>(vals.size() / cols);
  ttnn::Tensor tb = ttnn::Tensor::from_vector<float>(
      base, tt::tt_metal::TensorSpec(
                tt::tt_metal::Shape({rows_b, uc}),
                tt::tt_metal::TensorLayout(
                    ttnn::DataType::FLOAT32,
                    tt::tt_metal::PageConfig(ttnn::Layout::ROW_MAJOR),
                    tt::tt_metal::MemoryConfig{})),
      &device);
  ttnn::Tensor tv = ttnn::Tensor::from_vector<float>(
      vals, tt::tt_metal::TensorSpec(
                tt::tt_metal::Shape({nv, uc}),
                tt::tt_metal::TensorLayout(
                    ttnn::DataType::FLOAT32,
                    tt::tt_metal::PageConfig(ttnn::Layout::ROW_MAJOR),
                    tt::tt_metal::MemoryConfig{})),
      &device);
  ttnn::Tensor ti = ttnn::Tensor::from_vector<uint32_t>(
      ids, tt::tt_metal::TensorSpec(
               tt::tt_metal::Shape({nv}),
               tt::tt_metal::TensorLayout(
                   ttnn::DataType::UINT32,
                   tt::tt_metal::PageConfig(ttnn::Layout::ROW_MAJOR),
                   tt::tt_metal::MemoryConfig{})),
      &device);
  ttnn::Tensor filled = ttnn::indexed_fill(
      ti,
      ttnn::experimental::view(tb, tt::tt_metal::Shape({rows_b, 1, 1, uc})),
      ttnn::experimental::view(tv, tt::tt_metal::Shape({nv, 1, 1, uc})),
      std::nullopt, /*dim=*/0);
  return filled.to_vector<float>();
}

// W21: the exact GatherRowsExact form (the ttnn::gather with the [rows, cols]
// TILE index) as a test surface.
std::vector<float> GatherRowsForTest(const std::vector<float>& cache,
                                     const std::vector<uint32_t>& row_ids,
                                     int64_t cache_rows, int64_t cols) {
  MeshDevice& device = SharedMeshDevice();
  const uint32_t rows_c = static_cast<uint32_t>(cache_rows);
  const uint32_t uc = static_cast<uint32_t>(cols);
  const uint32_t npick = static_cast<uint32_t>(row_ids.size());
  ttnn::Tensor tc = ttnn::Tensor::from_vector<float>(
      cache, tt::tt_metal::TensorSpec(
                 tt::tt_metal::Shape({rows_c, uc}),
                 tt::tt_metal::TensorLayout(
                     ttnn::DataType::FLOAT32,
                     tt::tt_metal::PageConfig(ttnn::Layout::TILE),
                     tt::tt_metal::MemoryConfig{})),
      &device);
  std::vector<uint32_t> flat(static_cast<size_t>(npick) * cols);
  for (uint32_t r = 0; r < npick; ++r)
    for (int64_t c = 0; c < cols; ++c) flat[r * cols + c] = row_ids[r];
  ttnn::Tensor tidx = ttnn::Tensor::from_vector<uint32_t>(
      flat, tt::tt_metal::TensorSpec(
                tt::tt_metal::Shape({npick, uc}),
                tt::tt_metal::TensorLayout(
                    ttnn::DataType::UINT32,
                    tt::tt_metal::PageConfig(ttnn::Layout::TILE),
                    tt::tt_metal::MemoryConfig{})),
      &device);
  ttnn::Tensor out = ttnn::gather(tc, /*dim=*/0, tidx, /*sparse_grad=*/false,
                                  std::nullopt);
  return out.to_vector<float>();
}

}  // namespace vt::tenstorrent
