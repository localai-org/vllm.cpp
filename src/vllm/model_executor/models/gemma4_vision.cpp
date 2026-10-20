// Gemma-4 NaFlex SigLIP2 vision tower forward — MODEL-GEMMA4 G2-impl.
//
// Ported 1:1 from transformers/models/gemma4/modeling_gemma4.py (see the header
// for the class:line map). Composed from the public vt:: ops
// (MatmulBT/Add/RmsNorm/RopeFromCache/AttentionDenseFlash/GeluAndMul). All GEMMs
// run bf16; RMSNorm + softmax accumulate f32; the pooler sqrt(hidden) scale runs
// in host f32 (upstream keeps it f32 because it can exceed the fp16 range).
//
// The multidimensional vision RoPE (head_dim 64 = two 32-wide axis parts, each a
// NeoX-32 rope over its (x|y) axis) is applied with TWO RopeFromCache calls that
// share ONE cos|sin cache (both axes use identical inv_freq) but different
// position arrays: the first rotates channels [0:32] with the x-positions, the
// second rotates channels [32:64] (a +32-channel-offset head view) with the
// y-positions. This is bit-faithful to apply_multidimensional_rope's
// per-part apply_rotary_pos_emb (rotate_half over each 32-wide part).
#include "vllm/model_executor/models/gemma4_vision.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace vllm::multimodal {
namespace {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;

// RAII device buffer (mirror of qwen3_vl_vision's Buf).
struct Buf {
  Backend& b;
  void* p = nullptr;
  size_t bytes = 0;
  Tensor t;
  Buf(Backend& backend, Queue& q, DType dt, std::vector<int64_t> shape, const void* host = nullptr)
      : b(backend) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p = b.Alloc(bytes == 0 ? 1 : bytes);
    t.data = p;
    t.dtype = dt;
    t.device = q.device;
    t.rank = static_cast<int>(shape.size());
    int64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      t.shape[i] = shape[static_cast<size_t>(i)];
      t.stride[i] = stride;
      stride *= shape[static_cast<size_t>(i)];
    }
    if (host != nullptr) b.Copy(q, p, host, bytes);
  }
  ~Buf() { b.Free(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  Tensor& tensor() { return t; }
  void Download(Queue& q, void* dst) {
    b.Copy(q, dst, p, bytes);
    b.Synchronize(q);
  }
};

std::vector<uint16_t> ToBf16(const std::vector<float>& f) {
  std::vector<uint16_t> o(f.size());
  for (size_t i = 0; i < f.size(); ++i) o[i] = vt::F32ToBF16(f[i]);
  return o;
}

// A device-resident bf16 weight (owns one buffer).
struct DevW {
  Backend* b = nullptr;
  void* p = nullptr;
  Tensor t{};
  DevW() = default;
  DevW(const DevW&) = delete;
  DevW& operator=(const DevW&) = delete;
  DevW(DevW&& o) noexcept { *this = std::move(o); }
  DevW& operator=(DevW&& o) noexcept {
    if (this != &o) {
      Reset();
      b = o.b;
      p = o.p;
      t = o.t;
      o.b = nullptr;
      o.p = nullptr;
    }
    return *this;
  }
  void Reset() {
    if (b != nullptr && p != nullptr) b->Free(p);
    b = nullptr;
    p = nullptr;
  }
  ~DevW() { Reset(); }
  const Tensor& tensor() const { return t; }
};

// Upload host bf16 bits once as a device-resident buffer. Since #1359 the host
// store IS bf16, so this is a straight `Copy` of the checkpoint's own bytes
// rather than an N-element `F32ToBF16` pass per weight per forward.
DevW MakeDevBf16(Backend& b, Queue& q, const std::vector<uint16_t>& bf,
                 std::vector<int64_t> shape) {
  DevW d;
  d.b = &b;
  int64_t numel = 1;
  for (auto s : shape) numel *= s;
  const size_t bytes = static_cast<size_t>(numel) * vt::SizeOf(DType::kBF16);
  d.p = b.Alloc(bytes == 0 ? 1 : bytes);
  d.t.data = d.p;
  d.t.dtype = DType::kBF16;
  d.t.device = q.device;
  d.t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    d.t.shape[i] = shape[static_cast<size_t>(i)];
    d.t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  // The copy length is the ALLOCATED length, not the store's, and the store is
  // checked against it first. Copying `bf.size() * sizeof(uint16_t)` into a
  // `bytes`-sized allocation overruns it whenever a weight's shape and its store
  // disagree — a loader bug would land as heap corruption rather than as a
  // named refusal. The twin `qwen3_vl_vision.cpp` MakeDevBf16 grew this guard in
  // #1359; this one did not, and #2174 is the issue that names the asymmetry.
  // Behaviour is unchanged on every shape the loaders actually produce, where
  // the two lengths are equal.
  VT_CHECK(bf.size() * sizeof(uint16_t) >= bytes,
           "gemma-4 vision: weight store is smaller than its declared shape");
  if (bytes != 0) b.Copy(q, d.p, bf.data(), bytes);
  return d;
}

// out[M,N] = x[M,K] @ W[N,K]^T  (bias-free — every Gemma-4 vision Linear).
void Linear(Queue& q, Buf& out, Tensor x, Tensor w) { vt::MatmulBT(q, out.tensor(), x, w); }

struct DevBlock {
  DevW input_ln, post_attn_ln, pre_ff_ln, post_ff_ln;
  DevW q_proj, k_proj, v_proj;
  DevW o_proj;
  DevW q_norm, k_norm;
  DevW gate_proj, up_proj;
  DevW down_proj;
};

// Download a device bf16 [rows,cols] tensor into a host f32 vector.
std::vector<float> DownloadBf16(Queue& q, Buf& buf, int64_t rows, int64_t cols) {
  std::vector<uint16_t> tmp(static_cast<size_t>(rows) * cols);
  buf.Download(q, tmp.data());
  std::vector<float> f(tmp.size());
  for (size_t i = 0; i < tmp.size(); ++i) f[i] = vt::BF16ToF32(tmp[i]);
  return f;
}

// Gemma4ClippableLinear QAT clamp, host-side (correctness-first for the one-shot
// gate; a fused device clamp is the perf follow-on). torch clamps bf16 x against
// bf16 buffers, so round the bounds to bf16 first, then clamp the bf16-widened
// activation and round back. no-op when the bound is +/-inf (default).
void ClampDevice(Backend& b, Queue& q, Buf& buf, int64_t numel, float lo, float hi) {
  if (lo <= -3.0e38f && hi >= 3.0e38f) return;  // both defaults -> nothing to do
  const float lob = vt::BF16ToF32(vt::F32ToBF16(lo));
  const float hib = vt::BF16ToF32(vt::F32ToBF16(hi));
  std::vector<uint16_t> tmp(static_cast<size_t>(numel));
  b.Copy(q, tmp.data(), buf.tensor().data, tmp.size() * sizeof(uint16_t));
  b.Synchronize(q);
  for (size_t i = 0; i < tmp.size(); ++i) {
    float v = vt::BF16ToF32(tmp[i]);
    v = std::min(std::max(v, lob), hib);
    tmp[i] = vt::F32ToBF16(v);
  }
  b.Copy(q, buf.tensor().data, tmp.data(), tmp.size() * sizeof(uint16_t));
}

}  // namespace

std::vector<float> Gemma4VisionForward(const std::vector<float>& pixel_values,
                                       const std::vector<int64_t>& position_ids, int64_t n_total,
                                       const Gemma4VisionWeights& w, const Gemma4VisionConfig& cfg,
                                       Backend& b, Gemma4VisionCapture* cap) {
  const int64_t L = ValidateGemma4VisionInput(pixel_values, position_ids, n_total, cfg);
  VT_CHECK(cfg.hidden_size > 0 && cfg.num_heads > 0 && cfg.head_dim > 0 &&
               cfg.num_heads <= INT64_MAX / cfg.head_dim &&
               cfg.num_heads * cfg.head_dim == cfg.hidden_size &&
               cfg.num_kv_heads == cfg.num_heads && cfg.head_dim % 4 == 0 &&
               cfg.depth > 0 && cfg.intermediate_size > 0 && cfg.text_hidden_size > 0 &&
               std::isfinite(cfg.rms_norm_eps) && cfg.rms_norm_eps > 0 &&
               std::isfinite(cfg.rope_theta) && cfg.rope_theta > 0,
           "gemma4 vision: unsupported execution config (MHA required)");
  auto size = [](int64_t rows, int64_t cols) {
    VT_CHECK(rows > 0 && cols > 0 && rows <= INT64_MAX / cols / 2 &&
                 static_cast<uint64_t>(rows) <= SIZE_MAX / sizeof(float) / cols,
             "gemma4 vision: weight size overflow");
    return static_cast<size_t>(rows * cols);
  };
  const auto hid = cfg.hidden_size, imid = cfg.intermediate_size;
  VT_CHECK(w.input_proj.size() == size(hid, cfg.patch_dim()) &&
               w.position_embedding_table.size() == size(cfg.position_embedding_size, hid) * 2 &&
               w.embed_projection.size() == size(cfg.text_hidden_size, hid) &&
               w.blocks.size() == static_cast<size_t>(cfg.depth), "gemma4 vision: weight shape");
  for (const auto& block : w.blocks) {
    for (const auto* v : {&block.q_proj, &block.k_proj, &block.v_proj, &block.o_proj})
      VT_CHECK(v->size() == size(hid, hid), "gemma4 vision: attention weight shape");
    for (const auto* v : {&block.gate_proj, &block.up_proj, &block.down_proj})
      VT_CHECK(v->size() == size(hid, imid), "gemma4 vision: MLP weight shape");
    for (const auto* v : {&block.input_ln, &block.post_attn_ln, &block.pre_ff_ln, &block.post_ff_ln})
      VT_CHECK(v->size() == static_cast<size_t>(hid), "gemma4 vision: norm weight shape");
    VT_CHECK(block.q_norm.size() == static_cast<size_t>(cfg.head_dim) &&
                 block.k_norm.size() == static_cast<size_t>(cfg.head_dim), "gemma4 vision: head norm shape");
  }
  Queue q = b.CreateQueue();
  struct QueueGuard {
    Backend& backend;
    Queue& queue;
    ~QueueGuard() { backend.DestroyQueue(queue); }
  } queue_guard{b, q};
  const int64_t H = cfg.hidden_size;
  const int64_t nh = cfg.num_heads;
  const int64_t hd = cfg.head_dim;
  const int64_t I = cfg.intermediate_size;
  const int64_t PD = cfg.patch_dim();
  const int64_t k = cfg.pooling_kernel_size;
  const float eps = cfg.rms_norm_eps;

  // --- host precompute: 2-D pos-embed lookup (x_emb + y_emb), valid rows -------
  // position_embedding_table [2, pos_embed_size, H]; table[0]=x, table[1]=y.
  const int64_t PE = cfg.position_embedding_size;
  std::vector<float> pos_embed(static_cast<size_t>(L) * H);
  {
    const float* tx = w.position_embedding_table.data();               // [PE,H]
    const float* ty = w.position_embedding_table.data() + PE * H;      // [PE,H]
    for (int64_t r = 0; r < L; ++r) {
      const int64_t x = position_ids[static_cast<size_t>(r) * 2];
      const int64_t y = position_ids[static_cast<size_t>(r) * 2 + 1];
      const float* ex = tx + x * H;
      const float* ey = ty + y * H;
      float* dst = &pos_embed[static_cast<size_t>(r) * H];
      for (int64_t d = 0; d < H; ++d) dst[d] = ex[d] + ey[d];
    }
  }

  // --- host precompute: rope cos|sin cache [Ncache, rotary_dim=32] --------------
  // spatial_dim = head_dim/2 = 32; inv_freq[i] = theta^(-(2i)/spatial_dim),
  // i in [0, spatial_dim/2) = 16 freqs. cache[p] = [cos(p*inv_freq)(16) | sin(16)].
  const int64_t spatial_dim = hd / 2;      // 32
  const int64_t rot = spatial_dim;         // per-part rotary_dim = 32
  const int64_t nfreq = spatial_dim / 2;   // 16
  int64_t max_pos = 0;
  for (int64_t r = 0; r < L; ++r) {
    max_pos = std::max(max_pos, position_ids[static_cast<size_t>(r) * 2]);
    max_pos = std::max(max_pos, position_ids[static_cast<size_t>(r) * 2 + 1]);
  }
  const int64_t ncache = max_pos + 1;
  std::vector<double> inv_freq(static_cast<size_t>(nfreq));
  for (int64_t i = 0; i < nfreq; ++i)
    inv_freq[static_cast<size_t>(i)] =
        1.0 / std::pow(cfg.rope_theta, static_cast<double>(2 * i) / static_cast<double>(spatial_dim));
  std::vector<float> cache_f(static_cast<size_t>(ncache) * rot);
  for (int64_t p = 0; p < ncache; ++p) {
    float* row = &cache_f[static_cast<size_t>(p) * rot];
    for (int64_t i = 0; i < nfreq; ++i) {
      const double a = static_cast<double>(p) * inv_freq[static_cast<size_t>(i)];
      row[i] = static_cast<float>(std::cos(a));
      row[nfreq + i] = static_cast<float>(std::sin(a));
    }
  }
  Buf cache(b, q, DType::kBF16, {ncache, rot});
  {
    const auto cb = ToBf16(cache_f);
    b.Copy(q, cache.tensor().data, cb.data(), cb.size() * sizeof(uint16_t));
  }
  std::vector<int32_t> px(static_cast<size_t>(L)), py(static_cast<size_t>(L));
  for (int64_t r = 0; r < L; ++r) {
    px[static_cast<size_t>(r)] = static_cast<int32_t>(position_ids[static_cast<size_t>(r) * 2]);
    py[static_cast<size_t>(r)] = static_cast<int32_t>(position_ids[static_cast<size_t>(r) * 2 + 1]);
  }
  Buf posx(b, q, DType::kI32, {L}, px.data());
  Buf posy(b, q, DType::kI32, {L}, py.data());

  // --- device weights (converted + uploaded once; per-image gate, cost is nil) -
  DevW input_proj = MakeDevBf16(b, q, w.input_proj, {H, PD});
  DevW embed_proj = MakeDevBf16(b, q, w.embed_projection, {cfg.text_hidden_size, H});
  std::vector<DevBlock> blk(static_cast<size_t>(cfg.depth));
  for (int64_t l = 0; l < cfg.depth; ++l) {
    const Gemma4VisionBlockWeights& bw = w.blocks[static_cast<size_t>(l)];
    DevBlock& d = blk[static_cast<size_t>(l)];
    d.input_ln = MakeDevBf16(b, q, bw.input_ln, {H});
    d.post_attn_ln = MakeDevBf16(b, q, bw.post_attn_ln, {H});
    d.pre_ff_ln = MakeDevBf16(b, q, bw.pre_ff_ln, {H});
    d.post_ff_ln = MakeDevBf16(b, q, bw.post_ff_ln, {H});
    d.q_proj = MakeDevBf16(b, q, bw.q_proj, {H, H});
    d.k_proj = MakeDevBf16(b, q, bw.k_proj, {H, H});
    d.v_proj = MakeDevBf16(b, q, bw.v_proj, {H, H});
    d.o_proj = MakeDevBf16(b, q, bw.o_proj, {H, H});
    d.q_norm = MakeDevBf16(b, q, bw.q_norm, {hd});
    d.k_norm = MakeDevBf16(b, q, bw.k_norm, {hd});
    d.gate_proj = MakeDevBf16(b, q, bw.gate_proj, {I, H});
    d.up_proj = MakeDevBf16(b, q, bw.up_proj, {I, H});
    d.down_proj = MakeDevBf16(b, q, bw.down_proj, {H, I});
  }
  // weight-less norms (v_norm on head_dim, projector pre-norm on H): ones vectors.
  const std::vector<uint16_t> ones_hd(static_cast<size_t>(hd), vt::F32ToBF16(1.0f));
  const std::vector<uint16_t> ones_h(static_cast<size_t>(H), vt::F32ToBF16(1.0f));
  DevW v_norm = MakeDevBf16(b, q, ones_hd, {hd});
  DevW proj_norm = MakeDevBf16(b, q, ones_h, {H});

  // --- patch embed: 2*(pv-0.5), input_proj, + pos_embed -----------------------
  std::vector<float> pv_scaled(static_cast<size_t>(L) * PD);
  for (int64_t i = 0; i < L * PD; ++i)
    pv_scaled[static_cast<size_t>(i)] = 2.0f * (pixel_values[static_cast<size_t>(i)] - 0.5f);
  Buf pv(b, q, DType::kBF16, {L, PD});
  {
    const auto pb = ToBf16(pv_scaled);
    b.Copy(q, pv.tensor().data, pb.data(), pb.size() * sizeof(uint16_t));
  }
  Buf hidden(b, q, DType::kBF16, {L, H});
  Linear(q, hidden, pv.tensor(), input_proj.tensor());
  {
    Buf pe(b, q, DType::kBF16, {L, H});
    const auto pe_bf = ToBf16(pos_embed);
    b.Copy(q, pe.tensor().data, pe_bf.data(), pe_bf.size() * sizeof(uint16_t));
    vt::Add(q, hidden.tensor(), hidden.tensor(), pe.tensor());
  }
  if (cap != nullptr) cap->patch_embed_out = DownloadBf16(q, hidden, L, H);

  // --- encoder: 16 sandwich-norm blocks ---------------------------------------
  const vt::RmsNormArgs rms{eps, /*gemma=*/false};
  vt::RopeArgs ra;
  ra.rotary_dim = static_cast<int>(rot);  // 32 per part
  ra.is_neox_style = true;
  const float attn_scale = 1.0f;  // Gemma4VisionAttention.scaling == 1.0

  Buf n1(b, q, DType::kBF16, {L, H});
  Buf qb(b, q, DType::kBF16, {L, H}), kb(b, q, DType::kBF16, {L, H}), vb(b, q, DType::kBF16, {L, H});
  Buf ao(b, q, DType::kBF16, {L, nh, hd});
  Buf attn(b, q, DType::kBF16, {L, H});
  Buf n2(b, q, DType::kBF16, {L, H});
  Buf gu(b, q, DType::kBF16, {L, 2 * I});
  Buf act(b, q, DType::kBF16, {L, I});
  Buf mlp(b, q, DType::kBF16, {L, H});

  for (int64_t l = 0; l < cfg.depth; ++l) {
    const DevBlock& d = blk[static_cast<size_t>(l)];

    const Gemma4VisionBlockWeights& cw = w.blocks[static_cast<size_t>(l)];
    // residual = h ; h = input_layernorm(h)
    vt::RmsNorm(q, n1.tensor(), hidden.tensor(), d.input_ln.tensor(), rms);
    // Each source ClippableLinear owns independent bounds. Never mutate the
    // shared normalized input, even when one checkpoint happens to match them.
    auto clipped_linear = [&](Buf& out, Buf& input, const DevW& weight,
                              const Clip& clip, int64_t in_width, int64_t out_width) {
      Buf clamped(b, q, DType::kBF16, {L, in_width});
      b.Copy(q, clamped.tensor().data, input.tensor().data, clamped.bytes);
      ClampDevice(b, q, clamped, L * in_width, clip.in_min, clip.in_max);
      Linear(q, out, clamped.tensor(), weight.tensor());
      ClampDevice(b, q, out, L * out_width, clip.out_min, clip.out_max);
    };
    clipped_linear(qb, n1, d.q_proj, cw.q_clip, H, H);
    clipped_linear(kb, n1, d.k_proj, cw.k_clip, H, H);
    clipped_linear(vb, n1, d.v_proj, cw.v_clip, H, H);

    // view [L, nh, hd]
    auto view3 = [&](Buf& buf) {
      Tensor t = buf.tensor();
      t.rank = 3;
      t.shape[0] = L;
      t.shape[1] = nh;
      t.shape[2] = hd;
      t.stride[0] = nh * hd;
      t.stride[1] = hd;
      t.stride[2] = 1;
      return t;
    };
    Tensor q3 = view3(qb), k3 = view3(kb), v3 = view3(vb);

    // per-head q/k RMSNorm (over head_dim) + weight-less v RMSNorm. RmsNorm reads
    // only (shape[0]=rows, shape[1]=h), so normalize over head_dim via a flat
    // [L*nh, hd] view (contiguous == the [L,nh,hd] layout).
    auto view2 = [&](Buf& buf) {
      Tensor t = buf.tensor();
      t.rank = 2;
      t.shape[0] = L * nh;
      t.shape[1] = hd;
      t.stride[0] = hd;
      t.stride[1] = 1;
      return t;
    };
    Tensor q2h = view2(qb), k2h = view2(kb), v2h = view2(vb);
    vt::RmsNorm(q, q2h, q2h, d.q_norm.tensor(), rms);
    vt::RmsNorm(q, k2h, k2h, d.k_norm.tensor(), rms);
    vt::RmsNorm(q, v2h, v2h, v_norm.tensor(), rms);

    // multidim rope: part0 (ch 0:32) with x-positions, part1 (ch 32:64) with y.
    vt::RopeFromCache(q, q3, &k3, posx.tensor(), cache.tensor(), ra);
    Tensor q3b = q3, k3b = k3;
    const size_t elt = vt::SizeOf(DType::kBF16);
    q3b.shape[2] = spatial_dim;  // 32-wide offset view onto channels [32:64]
    k3b.shape[2] = spatial_dim;
    q3b.data = static_cast<char*>(q3.data) + static_cast<size_t>(spatial_dim) * elt;
    k3b.data = static_cast<char*>(k3.data) + static_cast<size_t>(spatial_dim) * elt;
    vt::RopeFromCache(q, q3b, &k3b, posy.tensor(), cache.tensor(), ra);

    // full non-causal attention over the L valid patches, scale 1.0.
    const vt::AttentionArgs aargs{attn_scale, /*causal=*/false};
    vt::AttentionDenseFlash(q, ao.tensor(), q3, k3, v3, aargs);
    if (cap != nullptr) cap->attention_out = DownloadBf16(q, ao, L, H);

    // o_proj + post_attention_layernorm + residual add.
    Tensor ao2 = ao.tensor();
    ao2.rank = 2;
    ao2.shape[0] = L;
    ao2.shape[1] = H;
    ao2.stride[0] = H;
    ao2.stride[1] = 1;
    // o_proj: clamp(input=attn output) -> linear -> clamp(output).
    ClampDevice(b, q, ao, L * H, cw.o_clip.in_min, cw.o_clip.in_max);
    Linear(q, attn, ao2, d.o_proj.tensor());
    ClampDevice(b, q, attn, L * H, cw.o_clip.out_min, cw.o_clip.out_max);
    vt::RmsNorm(q, attn.tensor(), attn.tensor(), d.post_attn_ln.tensor(), rms);
    vt::Add(q, hidden.tensor(), hidden.tensor(), attn.tensor());

    vt::RmsNorm(q, n2.tensor(), hidden.tensor(), d.pre_ff_ln.tensor(), rms);
    Buf gate(b, q, DType::kBF16, {L, I}), up(b, q, DType::kBF16, {L, I});
    clipped_linear(gate, n2, d.gate_proj, cw.gate_clip, H, I);
    clipped_linear(up, n2, d.up_proj, cw.up_clip, H, I);
    // Preserve the public fused activation layout, but not shared clipping.
    for (int64_t row = 0; row < L; ++row) {
      auto* dst = gu.tensor().Ptr<uint16_t>() + row * 2 * I;
      b.Copy(q, dst, gate.tensor().Ptr<uint16_t>() + row * I, static_cast<size_t>(I) * 2);
      b.Copy(q, dst + I, up.tensor().Ptr<uint16_t>() + row * I, static_cast<size_t>(I) * 2);
    }
    vt::GeluAndMul(q, act.tensor(), gu.tensor());
    ClampDevice(b, q, act, L * I, cw.down_clip.in_min, cw.down_clip.in_max);
    Linear(q, mlp, act.tensor(), d.down_proj.tensor());
    ClampDevice(b, q, mlp, L * H, cw.down_clip.out_min, cw.down_clip.out_max);
    vt::RmsNorm(q, mlp.tensor(), mlp.tensor(), d.post_ff_ln.tensor(), rms);
    if (cap != nullptr) cap->mlp_out = DownloadBf16(q, mlp, L, H);
    vt::Add(q, hidden.tensor(), hidden.tensor(), mlp.tensor());
  }
  std::vector<float> enc = DownloadBf16(q, hidden, L, H);
  if (cap != nullptr) cap->encoder_out = enc;

  // --- pooler: avg-pool-by-position over a k^2 grid, * sqrt(hidden), fp32 ------
  // soft_idx = (x//k) + (max_x//k) * (y//k); one_hot weight 1/k^2 per patch.
  int64_t max_x = 0;
  for (int64_t r = 0; r < L; ++r) max_x = std::max(max_x, position_ids[static_cast<size_t>(r) * 2]);
  const int64_t grid_x = (max_x + 1) / k;  // == max_x//k in upstream (max_x already +1)
  const double root_h = std::sqrt(static_cast<double>(H));
  const double inv_k2 = 1.0 / static_cast<double>(k * k);
  // number of soft tokens = distinct populated kernel idxs (contiguous 0..n_soft-1).
  int64_t n_soft = 0;
  for (int64_t r = 0; r < L; ++r) {
    const int64_t x = position_ids[static_cast<size_t>(r) * 2];
    const int64_t y = position_ids[static_cast<size_t>(r) * 2 + 1];
    n_soft = std::max(n_soft, (x / k) + grid_x * (y / k) + 1);
  }
  std::vector<double> pool(static_cast<size_t>(n_soft) * H, 0.0);
  for (int64_t r = 0; r < L; ++r) {
    const int64_t x = position_ids[static_cast<size_t>(r) * 2];
    const int64_t y = position_ids[static_cast<size_t>(r) * 2 + 1];
    const int64_t idx = (x / k) + grid_x * (y / k);
    const float* h = &enc[static_cast<size_t>(r) * H];
    double* o = &pool[static_cast<size_t>(idx) * H];
    for (int64_t d = 0; d < H; ++d) o[d] += static_cast<double>(h[d]) * inv_k2;
  }
  std::vector<float> pooled_f(static_cast<size_t>(n_soft) * H);
  std::vector<uint16_t> pooled_bf(static_cast<size_t>(n_soft) * H);
  for (int64_t i = 0; i < n_soft * H; ++i) {
    const float average = vt::BF16ToF32(vt::F32ToBF16(static_cast<float>(pool[static_cast<size_t>(i)])));
    const float v = average * static_cast<float>(root_h);
    pooled_bf[static_cast<size_t>(i)] = vt::F32ToBF16(v);
    pooled_f[static_cast<size_t>(i)] = vt::BF16ToF32(pooled_bf[static_cast<size_t>(i)]);
  }
  if (cap != nullptr) cap->pooled = pooled_f;

  // --- projector: RMSNorm(no-weight) -> Linear(H -> text_hidden) ---------------
  Buf pooled(b, q, DType::kBF16, {n_soft, H}, pooled_bf.data());
  Buf normed(b, q, DType::kBF16, {n_soft, H});
  vt::RmsNorm(q, normed.tensor(), pooled.tensor(), proj_norm.tensor(), rms);
  Buf projected(b, q, DType::kBF16, {n_soft, cfg.text_hidden_size});
  Linear(q, projected, normed.tensor(), embed_proj.tensor());
  std::vector<float> out = DownloadBf16(q, projected, n_soft, cfg.text_hidden_size);
  if (cap != nullptr) cap->projected = out;

  return out;
}

}  // namespace vllm::multimodal
