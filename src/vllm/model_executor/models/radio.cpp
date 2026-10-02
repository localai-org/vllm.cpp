// RADIO vision tower and the Nemotron Nano VL projector. See
// include/vllm/model_executor/models/radio.h for the port map and the silent
// traps this file is written around.
//
// Ported from vLLM `e126687a9a`:
//   vllm/model_executor/models/radio.py:109-421, :470-644, :745-774
//   vllm/model_executor/models/intern_vit.py:145-350
//   vllm/model_executor/models/nano_nemotron_vl.py:955-976, :1012-1058
// plus torch `F.interpolate(mode="bilinear", align_corners=False)` for the CPE
// table (aten/src/ATen/native/UpSample.h `area_pixel_compute_source_index`,
// `guard_index_and_lambda`, `compute_source_index_and_lambda`, and the
// `Interpolate<n, ..., 2>` accumulation order in
// aten/src/ATen/native/cpu/UpSampleKernel.cpp, torch v2.11.0).
//
// Composed from the public vt:: ops (MatmulBT / Add / LayerNorm / GeluErf /
// AttentionDenseFlash / RmsNorm / MoeRelu2) and the shared merged-QKV seam
// vllm::models::FusedMergedQkvBiasSplit, exactly as the Muse Glimmer and
// Qwen3-VL towers are. The weights are uploaded ONCE, in the constructor, in
// the compute dtype; the per-image host precomputes (the positional table, the
// patch rows) are f32.
#include "vllm/model_executor/models/radio.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/merged_qkv_fold.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace vllm::multimodal {
namespace {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;

// RAII device buffer (movable, so blocks of them live in a vector).
struct Buf {
  Backend* b = nullptr;
  void* p = nullptr;
  size_t bytes = 0;
  Tensor t{};

  Buf() = default;
  Buf(Backend& backend, Queue& q, DType dt, const std::vector<int64_t>& shape,
      const void* host = nullptr)
      : b(&backend) {
    int64_t numel = 1;
    for (int64_t s : shape) numel *= s;
    bytes = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p = b->Alloc(bytes == 0 ? 1 : bytes);
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
    if (host != nullptr && bytes != 0) b->Copy(q, p, host, bytes);
  }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  Buf(Buf&& o) noexcept { *this = std::move(o); }
  Buf& operator=(Buf&& o) noexcept {
    if (this != &o) {
      if (b != nullptr && p != nullptr) b->Free(p);
      b = o.b;
      p = o.p;
      bytes = o.bytes;
      t = o.t;
      o.b = nullptr;
      o.p = nullptr;
      o.bytes = 0;
    }
    return *this;
  }
  ~Buf() {
    if (b != nullptr && p != nullptr) b->Free(p);
  }
  Tensor& tensor() { return t; }
  const Tensor& tensor() const { return t; }
  void Download(Queue& q, void* dst) const {
    b->Copy(q, dst, p, bytes);
    b->Synchronize(q);
  }
};

// Upload host bf16 bits in `dt`: verbatim for bf16, widened for f32.
Buf UploadBf16(Backend& b, Queue& q, DType dt, const std::vector<int64_t>& shape,
               const std::vector<uint16_t>& host, const char* what) {
  int64_t numel = 1;
  for (int64_t s : shape) numel *= s;
  VT_CHECK(static_cast<int64_t>(host.size()) == numel,
           std::string("RADIO/mlp1 weight '") + what + "' has " +
               std::to_string(host.size()) + " elements, expected " +
               std::to_string(numel));
  if (dt == DType::kBF16) return Buf(b, q, dt, shape, host.data());
  VT_CHECK(dt == DType::kF32, "RADIO/mlp1: compute dtype must be bf16 or f32");
  std::vector<float> f(host.size());
  for (size_t i = 0; i < host.size(); ++i) f[i] = vt::BF16ToF32(host[i]);
  return Buf(b, q, dt, shape, f.data());
}

// Upload host f32 in `dt` (rounded to nearest even for bf16).
Buf UploadF32(Backend& b, Queue& q, DType dt, const std::vector<int64_t>& shape,
              const std::vector<float>& host) {
  if (dt == DType::kBF16) {
    std::vector<uint16_t> bits(host.size());
    for (size_t i = 0; i < host.size(); ++i) bits[i] = vt::F32ToBF16(host[i]);
    return Buf(b, q, dt, shape, bits.data());
  }
  return Buf(b, q, dt, shape, host.data());
}

std::vector<float> DownloadF32(const Buf& buf, Queue& q, DType dt, size_t numel) {
  std::vector<float> out(numel);
  if (dt == DType::kBF16) {
    std::vector<uint16_t> tmp(numel);
    buf.Download(q, tmp.data());
    for (size_t i = 0; i < numel; ++i) out[i] = vt::BF16ToF32(tmp[i]);
  } else {
    buf.Download(q, out.data());
  }
  return out;
}

float RoundToDType(float v, DType dt) {
  return dt == DType::kBF16 ? vt::BF16ToF32(vt::F32ToBF16(v)) : v;
}

Tensor RowSlice(const Tensor& src, int64_t row, int64_t rows) {
  Tensor s = src;
  s.shape[0] = rows;
  int64_t inner = 1;
  for (int i = 1; i < src.rank; ++i) inner *= src.shape[i];
  s.data = static_cast<char*>(src.data) +
           static_cast<size_t>(row * inner) * vt::SizeOf(src.dtype);
  return s;
}

Tensor Reshape2(const Tensor& src, int64_t d0, int64_t d1) {
  Tensor s = src;
  s.rank = 2;
  s.shape[0] = d0;
  s.shape[1] = d1;
  s.stride[0] = d1;
  s.stride[1] = 1;
  return s;
}

Tensor Reshape3(const Tensor& src, int64_t d0, int64_t d1, int64_t d2) {
  Tensor s = src;
  s.rank = 3;
  s.shape[0] = d0;
  s.shape[1] = d1;
  s.shape[2] = d2;
  s.stride[0] = d1 * d2;
  s.stride[1] = d2;
  s.stride[2] = 1;
  return s;
}

void LinearBias(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                const Tensor* bias) {
  vt::MatmulBT(q, out, x, w);
  if (bias != nullptr) vt::Add(q, out, out, *bias);
}

// One axis of torch's non-antialiased bilinear resample, align_corners=False
// (UpSample.h `compute_source_index_and_lambda`): the two source taps and
// their weights for output index `i`. All arithmetic is f32, as `opmath_t` is
// for an f32 input.
struct LinearTap {
  int64_t i0 = 0, i1 = 0;
  float l0 = 1.0f, l1 = 0.0f;
};

LinearTap BilinearTap(int64_t in, int64_t out, int64_t i) {
  LinearTap t;
  if (in == out) {  // "scale_factor = 1, simply copy"
    t.i0 = t.i1 = i;
    return t;
  }
  const float scale = static_cast<float>(in) / static_cast<float>(out);
  float src = scale * (static_cast<float>(i) + 0.5f) - 0.5f;
  if (src < 0.0f) src = 0.0f;  // linear modes clamp at 0 (UpSample.h:296-305)
  t.i0 = std::min(static_cast<int64_t>(std::floor(src)), in - 1);
  t.l1 = std::min(std::max(src - static_cast<float>(t.i0), 0.0f), 1.0f);
  t.i1 = t.i0 + ((t.i0 < in - 1) ? 1 : 0);
  t.l0 = 1.0f - t.l1;
  return t;
}

// `F.interpolate(x.float(), size=(oh, ow), mode="bilinear",
// align_corners=False)` of a channels-last [ih, iw, C] table into
// [oh, ow, C]. The accumulation order is `Interpolate<2, ..., 2>`:
// t0 = in[h0,w0]*ww0 + in[h0,w1]*ww1; t1 likewise at h1; out = t0*wh0 + t1*wh1.
std::vector<float> Bilinear(const std::vector<float>& in, int64_t ih, int64_t iw,
                            int64_t c, int64_t oh, int64_t ow) {
  std::vector<float> out(static_cast<size_t>(oh * ow * c));
  std::vector<LinearTap> tw(static_cast<size_t>(ow));
  for (int64_t x = 0; x < ow; ++x) tw[static_cast<size_t>(x)] = BilinearTap(iw, ow, x);
  for (int64_t y = 0; y < oh; ++y) {
    const LinearTap th = BilinearTap(ih, oh, y);
    for (int64_t x = 0; x < ow; ++x) {
      const LinearTap& w = tw[static_cast<size_t>(x)];
      const float* a00 = &in[static_cast<size_t>((th.i0 * iw + w.i0) * c)];
      const float* a01 = &in[static_cast<size_t>((th.i0 * iw + w.i1) * c)];
      const float* a10 = &in[static_cast<size_t>((th.i1 * iw + w.i0) * c)];
      const float* a11 = &in[static_cast<size_t>((th.i1 * iw + w.i1) * c)];
      float* o = &out[static_cast<size_t>((y * ow + x) * c)];
      for (int64_t k = 0; k < c; ++k) {
        const float t0 = a00[k] * w.l0 + a01[k] * w.l1;
        const float t1 = a10[k] * w.l0 + a11[k] * w.l1;
        o[k] = t0 * th.l0 + t1 * th.l1;
      }
    }
  }
  return out;
}

struct DevBlock {
  Buf norm1_w, norm1_b, norm2_w, norm2_b;
  Buf qkv_w, qkv_b, proj_w, proj_b;
  Buf fc1_w, fc1_b, fc2_w, fc2_b;
};

}  // namespace

// ─── _get_pos_embeddings (radio.py:386-421) ─────────────────────────────────
std::vector<float> RadioPosEmbedForGrid(const std::vector<uint16_t>& pos_embed,
                                        int64_t grid_h, int64_t grid_w,
                                        const RadioVisionConfig& cfg,
                                        bool round_to_bf16) {
  const int64_t H = cfg.hidden_size;
  const int64_t rows = cfg.pos_rows;
  const int64_t cols = cfg.pos_cols;
  VT_CHECK(static_cast<int64_t>(pos_embed.size()) == rows * cols * H,
           "RADIO pos_embed must be [pos_rows * pos_cols, hidden]");
  VT_CHECK(grid_h > 0 && grid_w > 0, "RADIO positional grid must be non-empty");
  std::vector<float> table(pos_embed.size());
  for (size_t i = 0; i < pos_embed.size(); ++i) table[i] = vt::BF16ToF32(pos_embed[i]);

  // (num_rows, num_cols) == input_dims returns the table untouched (:387-388).
  if (rows == grid_h && cols == grid_w) return table;

  std::vector<float> pe;
  int64_t ph = rows;
  int64_t pw = cols;
  if (cfg.cpe_mode) {
    // TRAP 1: interpolate to a SQUARE max(h, w) grid first (:401-408) ...
    const int64_t max_dim = std::max(grid_h, grid_w);
    pe = Bilinear(table, rows, cols, H, max_dim, max_dim);
    // ... cast back to the parameter dtype (TRAP 4) ...
    if (round_to_bf16)
      for (float& v : pe) v = RoundToDType(v, DType::kBF16);
    ph = pw = max_dim;
  } else {
    pe = std::move(table);
  }
  // ... then window_select crops the top-left grid_h x grid_w (:394-399, :410).
  const int64_t wh = std::min(grid_h, ph);
  const int64_t ww = std::min(grid_w, pw);
  std::vector<float> cropped(static_cast<size_t>(wh * ww * H));
  for (int64_t y = 0; y < wh; ++y)
    std::memcpy(&cropped[static_cast<size_t>(y * ww * H)],
                &pe[static_cast<size_t>(y * pw * H)],
                static_cast<size_t>(ww * H) * sizeof(float));
  // A crop that is still not the grid (only possible off the CPE path, where
  // the table is smaller than the grid) is interpolated to it (:414-417).
  if (wh != grid_h || ww != grid_w) {
    cropped = Bilinear(cropped, wh, ww, H, grid_h, grid_w);
    if (round_to_bf16)
      for (float& v : cropped) v = RoundToDType(v, DType::kBF16);
  }
  return cropped;
}

// ─── the tower ───────────────────────────────────────────────────────────────
struct RadioVisionTower::Impl {
  RadioVisionConfig cfg;
  Backend* backend = nullptr;
  std::vector<uint16_t> pos_embed;  // host: interpolated per grid
  Buf embedder_w;                   // [H, 3p^2]
  Buf cls_token;                    // [num_skip, H]
  std::vector<DevBlock> blocks;
};

RadioVisionTower::RadioVisionTower(const RadioVisionWeights& w,
                                   const RadioVisionConfig& cfg,
                                   Backend& backend)
    : impl_(std::make_unique<Impl>()) {
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  VT_CHECK(H % cfg.num_attention_heads == 0,
           "RADIO hidden size must divide the number of heads");
  VT_CHECK(static_cast<int64_t>(w.blocks.size()) == cfg.num_hidden_layers,
           "RADIO weights carry " + std::to_string(w.blocks.size()) +
               " blocks, the config asks for " +
               std::to_string(cfg.num_hidden_layers));
  impl_->cfg = cfg;
  impl_->backend = &backend;
  impl_->pos_embed = w.pos_embed;
  VT_CHECK(static_cast<int64_t>(w.pos_embed.size()) ==
               cfg.pos_rows * cfg.pos_cols * H,
           "RADIO pos_embed must be [pos_rows * pos_cols, hidden]");
  const DType dt = cfg.compute_dtype;
  Queue q = backend.CreateQueue();
  impl_->embedder_w =
      UploadBf16(backend, q, dt, {H, cfg.patch_dim()}, w.embedder_w, "embedder");
  impl_->cls_token =
      UploadBf16(backend, q, dt, {cfg.num_skip(), H}, w.cls_token, "cls_token");
  impl_->blocks.resize(w.blocks.size());
  for (size_t l = 0; l < w.blocks.size(); ++l) {
    const RadioBlockWeights& bw = w.blocks[l];
    DevBlock& d = impl_->blocks[l];
    d.norm1_w = UploadBf16(backend, q, dt, {H}, bw.norm1_w, "norm1.weight");
    d.norm1_b = UploadBf16(backend, q, dt, {H}, bw.norm1_b, "norm1.bias");
    d.norm2_w = UploadBf16(backend, q, dt, {H}, bw.norm2_w, "norm2.weight");
    d.norm2_b = UploadBf16(backend, q, dt, {H}, bw.norm2_b, "norm2.bias");
    d.qkv_w = UploadBf16(backend, q, dt, {3 * H, H}, bw.qkv_w, "attn.qkv.weight");
    d.qkv_b = UploadBf16(backend, q, dt, {3 * H}, bw.qkv_b, "attn.qkv.bias");
    d.proj_w = UploadBf16(backend, q, dt, {H, H}, bw.proj_w, "attn.proj.weight");
    d.proj_b = UploadBf16(backend, q, dt, {H}, bw.proj_b, "attn.proj.bias");
    d.fc1_w = UploadBf16(backend, q, dt, {I, H}, bw.fc1_w, "mlp.fc1.weight");
    d.fc1_b = UploadBf16(backend, q, dt, {I}, bw.fc1_b, "mlp.fc1.bias");
    d.fc2_w = UploadBf16(backend, q, dt, {H, I}, bw.fc2_w, "mlp.fc2.weight");
    d.fc2_b = UploadBf16(backend, q, dt, {H}, bw.fc2_b, "mlp.fc2.bias");
  }
  backend.Synchronize(q);
  backend.DestroyQueue(q);
}

RadioVisionTower::~RadioVisionTower() = default;

const RadioVisionConfig& RadioVisionTower::config() const { return impl_->cfg; }

std::vector<float> RadioVisionTower::Forward(
    const std::vector<RadioImage>& images) const {
  const RadioVisionConfig& cfg = impl_->cfg;
  Backend& backend = *impl_->backend;
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  const int64_t nh = cfg.num_attention_heads;
  const int64_t hd = cfg.head_dim();
  const int64_t pd = cfg.patch_dim();
  const int64_t skip = cfg.num_skip();
  const DType dt = cfg.compute_dtype;
  VT_CHECK(!images.empty(), "RADIO forward needs at least one image");

  // The packed sequence: per image, [num_skip CLS/register rows | patches].
  int64_t L = 0;
  int64_t n_patches = 0;
  std::vector<int64_t> seg_lens;
  for (const RadioImage& im : images) {
    VT_CHECK(im.grid_h > 0 && im.grid_w > 0, "RADIO image grid must be non-empty");
    VT_CHECK(static_cast<int64_t>(im.patches.size()) == im.grid_h * im.grid_w * pd,
             "RADIO image patches must be [grid_h * grid_w, 3 * patch^2]");
    seg_lens.push_back(skip + im.grid_h * im.grid_w);
    L += seg_lens.back();
    n_patches += im.grid_h * im.grid_w;
  }

  Queue q = backend.CreateQueue();
  Buf hidden(backend, q, dt, {L, H});
  {
    // embedder over every patch row at once (the Linear is row-local), then
    // per image: + pos_enc, and the CLS/register rows in front (:201-206,
    // :265-312).
    std::vector<float> all_patches;
    all_patches.reserve(static_cast<size_t>(n_patches * pd));
    for (const RadioImage& im : images)
      all_patches.insert(all_patches.end(), im.patches.begin(), im.patches.end());
    const Buf pix = UploadF32(backend, q, dt, {n_patches, pd}, all_patches);
    Buf emb(backend, q, dt, {n_patches, H});
    vt::MatmulBT(q, emb.tensor(), pix.tensor(), impl_->embedder_w.tensor());

    int64_t in_off = 0;
    int64_t out_off = 0;
    const size_t esz = vt::SizeOf(dt);
    for (const RadioImage& im : images) {
      const int64_t n = im.grid_h * im.grid_w;
      // TRAP 2: every image gets its own CLS/register rows.
      backend.Copy(q, static_cast<char*>(hidden.p) + static_cast<size_t>(out_off * H) * esz,
                   impl_->cls_token.p, static_cast<size_t>(skip * H) * esz);
      const std::vector<float> pe = RadioPosEmbedForGrid(
          impl_->pos_embed, im.grid_h, im.grid_w, cfg, dt == DType::kBF16);
      const Buf pe_dev = UploadF32(backend, q, dt, {n, H}, pe);
      Tensor dst = RowSlice(hidden.tensor(), out_off + skip, n);
      const Tensor src = RowSlice(emb.tensor(), in_off, n);
      vt::Add(q, dst, src, pe_dev.tensor());
      backend.Synchronize(q);  // pe_dev dies at scope end
      in_off += n;
      out_off += skip + n;
    }
  }

  Buf n1(backend, q, dt, {L, H});
  Buf qkv(backend, q, dt, {L, 3 * H});
  Buf qb(backend, q, dt, {L, H}), kb(backend, q, dt, {L, H}), vb(backend, q, dt, {L, H});
  Buf ao(backend, q, dt, {L, nh, hd});
  Buf attn(backend, q, dt, {L, H});
  Buf n2(backend, q, dt, {L, H});
  Buf f1(backend, q, dt, {L, I});
  Buf f2(backend, q, dt, {L, H});
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));  // intern_vit.py:185
  const vt::AttentionArgs aargs{scale, /*causal=*/false};

  for (const DevBlock& d : impl_->blocks) {
    // RadioVisionEncoderLayer.forward (:493-505), ls1 = ls2 = 1.
    vt::LayerNorm(q, n1.tensor(), hidden.tensor(), &d.norm1_w.tensor(),
                  &d.norm1_b.tensor(), vt::LayerNormArgs{cfg.layer_norm_eps});
    {
      // qkv is one merged [3H, H] Linear WITH bias; `qkv.chunk(3, dim=-1)`
      // (:474-475) is exactly the contiguous q|k|v thirds of this fold.
      Tensor qkv_bias = d.qkv_b.tensor();
      vllm::models::FusedMergedQkvBiasSplit(q, qkv.tensor(), qb.tensor(), kb.tensor(),
                                            vb.tensor(), n1.tensor(), d.qkv_w.tensor(),
                                            &qkv_bias);
    }
    const Tensor q3 = Reshape3(qb.tensor(), L, nh, hd);
    const Tensor k3 = Reshape3(kb.tensor(), L, nh, hd);
    const Tensor v3 = Reshape3(vb.tensor(), L, nh, hd);
    // One non-causal segment per image: the block-diagonal mask cu_seqlens
    // builds (:579-604). A single image is one segment, as upstream passes no
    // mask at all in that case (:630-634).
    //
    // KERNEL CHOICE: `vt::AttentionDenseFlash`, never `vt::Attention` (#1545):
    // the same rung the Muse Glimmer and Qwen3-VL towers use for non-causal
    // vision attention; on CPU both resolve to the same kernel.
    int64_t off = 0;
    for (int64_t seg : seg_lens) {
      Tensor qs = RowSlice(q3, off, seg);
      const Tensor ks = RowSlice(k3, off, seg);
      const Tensor vs = RowSlice(v3, off, seg);
      Tensor os = RowSlice(ao.tensor(), off, seg);
      vt::AttentionDenseFlash(q, os, qs, ks, vs, aargs);
      off += seg;
    }
    const Tensor ao2 = Reshape2(ao.tensor(), L, H);
    LinearBias(q, attn.tensor(), ao2, d.proj_w.tensor(), &d.proj_b.tensor());
    vt::Add(q, hidden.tensor(), hidden.tensor(), attn.tensor());

    vt::LayerNorm(q, n2.tensor(), hidden.tensor(), &d.norm2_w.tensor(),
                  &d.norm2_b.tensor(), vt::LayerNormArgs{cfg.layer_norm_eps});
    LinearBias(q, f1.tensor(), n2.tensor(), d.fc1_w.tensor(), &d.fc1_b.tensor());
    vt::GeluErf(q, f1.tensor(), f1.tensor());  // get_act_fn("gelu") is erf GELU
    LinearBias(q, f2.tensor(), f1.tensor(), d.fc2_w.tensor(), &d.fc2_b.tensor());
    vt::Add(q, hidden.tensor(), hidden.tensor(), f2.tensor());
  }

  const std::vector<float> all = DownloadF32(hidden, q, dt, static_cast<size_t>(L * H));
  backend.DestroyQueue(q);

  // _extract_final (:745-774): drop each image's num_skip leading rows.
  std::vector<float> out;
  out.reserve(static_cast<size_t>(n_patches * H));
  int64_t off = 0;
  for (size_t i = 0; i < images.size(); ++i) {
    const int64_t n = images[i].grid_h * images[i].grid_w;
    const auto first = all.begin() + static_cast<std::ptrdiff_t>((off + skip) * H);
    out.insert(out.end(), first, first + static_cast<std::ptrdiff_t>(n * H));
    off += seg_lens[i];
  }
  return out;
}

// ─── pixel_shuffle v2 (nano_nemotron_vl.py:1012-1029) ───────────────────────
// x.view(n, h/r, r, w/r, r, c).permute(0, 1, 3, 2, 4, 5)
//  .reshape(n, h/r, w/r, c*r*r): output (oy, ox) gathers, in (ry, rx, c) order,
// the input rows (oy*r + ry, ox*r + rx).
std::vector<float> NanoNemotronVLPixelShuffle(const std::vector<float>& features,
                                              int64_t grid_h, int64_t grid_w,
                                              int64_t dim, int64_t r) {
  VT_CHECK(r > 0 && grid_h % r == 0 && grid_w % r == 0,
           "Nano Nemotron VL pixel shuffle: the patch grid must divide the "
           "downsample factor (the tiler rounds it to even, "
           "processors/nano_nemotron_vl.py:434-460)");
  VT_CHECK(static_cast<int64_t>(features.size()) == grid_h * grid_w * dim,
           "Nano Nemotron VL pixel shuffle: features must be [grid_h * grid_w, dim]");
  const int64_t oh = grid_h / r;
  const int64_t ow = grid_w / r;
  const int64_t od = dim * r * r;
  std::vector<float> out(static_cast<size_t>(oh * ow * od));
  for (int64_t oy = 0; oy < oh; ++oy)
    for (int64_t ox = 0; ox < ow; ++ox)
      for (int64_t ry = 0; ry < r; ++ry)
        for (int64_t rx = 0; rx < r; ++rx) {
          const int64_t src_row = (oy * r + ry) * grid_w + (ox * r + rx);
          float* dst = &out[static_cast<size_t>((oy * ow + ox) * od + (ry * r + rx) * dim)];
          std::memcpy(dst, &features[static_cast<size_t>(src_row * dim)],
                      static_cast<size_t>(dim) * sizeof(float));
        }
  return out;
}

// ─── mlp1 (nano_nemotron_vl.py:962-976) ─────────────────────────────────────
struct NanoNemotronVLProjector::Impl {
  NanoNemotronVLProjectorConfig cfg;
  Backend* backend = nullptr;
  Buf norm_w, fc1_w, fc2_w;
};

NanoNemotronVLProjector::NanoNemotronVLProjector(
    const NanoNemotronVLProjectorWeights& w, const NanoNemotronVLProjectorConfig& cfg,
    Backend& backend)
    : impl_(std::make_unique<Impl>()) {
  impl_->cfg = cfg;
  impl_->backend = &backend;
  const DType dt = cfg.compute_dtype;
  Queue q = backend.CreateQueue();
  impl_->norm_w = UploadBf16(backend, q, dt, {cfg.in_dim}, w.norm_w, "mlp1.0.weight");
  impl_->fc1_w =
      UploadBf16(backend, q, dt, {cfg.hidden_dim, cfg.in_dim}, w.fc1_w, "mlp1.1.weight");
  impl_->fc2_w =
      UploadBf16(backend, q, dt, {cfg.out_dim, cfg.hidden_dim}, w.fc2_w, "mlp1.3.weight");
  backend.Synchronize(q);
  backend.DestroyQueue(q);
}

NanoNemotronVLProjector::~NanoNemotronVLProjector() = default;

const NanoNemotronVLProjectorConfig& NanoNemotronVLProjector::config() const {
  return impl_->cfg;
}

std::vector<float> NanoNemotronVLProjector::Forward(const std::vector<float>& x,
                                                    int64_t rows) const {
  const NanoNemotronVLProjectorConfig& cfg = impl_->cfg;
  Backend& backend = *impl_->backend;
  VT_CHECK(rows > 0 && static_cast<int64_t>(x.size()) == rows * cfg.in_dim,
           "Nano Nemotron VL projector input must be [rows, in_dim]");
  const DType dt = cfg.compute_dtype;
  Queue q = backend.CreateQueue();
  const Buf xin = UploadF32(backend, q, dt, {rows, cfg.in_dim}, x);
  Buf xn(backend, q, dt, {rows, cfg.in_dim});
  Buf h(backend, q, dt, {rows, cfg.hidden_dim});
  Buf out(backend, q, dt, {rows, cfg.out_dim});
  // RMSNorm (vllm/ir/ops/layernorm.py rms_norm: f32 statistics, weight applied
  // in the weight dtype).
  vt::RmsNorm(q, xn.tensor(), xin.tensor(), impl_->norm_w.tensor(),
              vt::RmsNormArgs{cfg.rms_norm_eps, false});
  vt::MatmulBT(q, h.tensor(), xn.tensor(), impl_->fc1_w.tensor());
  // ReLUSquaredActivation (activation.py:648-662): square(relu(x)), which is
  // exactly vt::MoeRelu2's elementwise contract.
  vt::MoeRelu2(q, h.tensor(), h.tensor());
  vt::MatmulBT(q, out.tensor(), h.tensor(), impl_->fc2_w.tensor());
  std::vector<float> result =
      DownloadF32(out, q, dt, static_cast<size_t>(rows * cfg.out_dim));
  backend.DestroyQueue(q);
  return result;
}

}  // namespace vllm::multimodal
