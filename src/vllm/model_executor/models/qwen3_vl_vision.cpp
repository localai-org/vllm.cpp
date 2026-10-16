// Qwen3-VL vision tower (`Qwen3_VisionTransformer`) forward — M2a + speed pass.
//
// Ported 1:1 from vllm/model_executor/models/qwen3_vl.py @ e24d1b24:
//   forward (:800-841), Qwen3_VisionPatchEmbed (:347-373),
//   Qwen3_VisionBlock (:413-464), Qwen3_VisionMLP (:376-410),
//   Qwen3_VisionPatchMerger (:467-516), pos_embed_interpolate_native (:277-344),
//   rot_pos_ids (:640-665) + rot_pos_emb (:667-683),
//   vision attention Qwen2_5_VisionAttention.forward (qwen2_5_vl.py:397-460),
//   ApplyRotaryEmb.forward_static (rotary_embedding/common.py:151-186).
//
// Composed from the public vt:: ops (Matmul/Add/LayerNorm/RopeFromCache/
// Attention/GeluTanh/GeluErf). All GEMMs run in the production model dtype bf16;
// softmax/norm accumulate in f32. The pos-embed bilinear interp and the vision
// rope cos|sin are deterministic host precomputes (f32) consumed on device — vLLM
// computes them on GPU (a Triton bilinear kernel + a rope cache), gated within a
// stated bf16 tolerance in the M2a unit test.
//
// SPEED PASS (CLAIM-MULTIMODAL-SPEED-TOWER): the tower weights are converted to
// bf16 + uploaded ONCE via PrepareVisionDeviceWeights and kept device-resident
// (mirroring vLLM's already-loaded nn.Linears); the per-image forward then does
// only the tiny pixel/pos-embed/rope uploads + the ViT GEMMs/attention. The old
// host-weights overload is preserved as a thin prepare-then-forward wrapper
// (BIT-IDENTICAL: same bf16 weight bytes, same GEMM order) so every unit gate is
// unchanged. Set VLLM_MM_TOWER_PROFILE=1 to print the prepare(marshal) vs
// forward(compute) split on stderr.
#include "vllm/model_executor/models/qwen3_vl_vision.h"
#include "vllm/model_executor/models/qwen3_vl_vision_attention.h"

#include <algorithm>
#include <array>
#include <set>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <map>
#include <mutex>
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

// --- RAII device buffer (mirror of the tests' DeviceTensor helper). ----------
struct Buf {
  Backend& b;
  void* p = nullptr;
  size_t bytes = 0;
  Tensor t;
  Buf(Backend& backend, Queue& q, DType dt, std::vector<int64_t> shape,
      const void* host = nullptr)
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

// out[M,N] = x[M,K] @ W[N,K]^T + bias[N]  (bias optional). All bf16.
void LinearBias(Queue& q, Buf& out, Tensor x, Tensor w, const Tensor* bias) {
  vt::MatmulBT(q, out.tensor(), x, w);
  if (bias != nullptr) vt::Add(q, out.tensor(), out.tensor(), *bias);
}

}  // namespace

// --- device-resident weight holder (owns one bf16 buffer, device-global) ------
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

namespace {

// Source bits already have the declared dtype. The caller retains their owner
// until the prepare queue drains.
DevW MakeDevRaw16(Backend& b, Queue& q, const std::vector<uint16_t>& bits,
                  std::vector<int64_t> shape, DType dtype) {
  DevW d;
  d.b = &b;
  int64_t numel = 1;
  for (auto s : shape) numel *= s;
  const size_t bytes = static_cast<size_t>(numel) * vt::SizeOf(DType::kBF16);
  d.p = b.Alloc(bytes == 0 ? 1 : bytes);
  d.t.data = d.p;
  d.t.dtype = dtype;
  d.t.device = q.device;
  d.t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    d.t.shape[i] = shape[static_cast<size_t>(i)];
    d.t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  VT_CHECK(bits.size() * sizeof(uint16_t) >= bytes,
           "qwen3-vl vision: weight store is smaller than its declared shape");
  if (bytes != 0) b.Copy(q, d.p, bits.data(), bytes);
  return d;
}

DevW MakeDevWeight(Backend& b, Queue& q, const std::vector<uint16_t>& bf,
                   std::vector<int64_t> shape, DType dtype,
                   std::vector<std::vector<uint16_t>>& staging) {
  if (dtype == DType::kBF16) return MakeDevRaw16(b, q, bf, std::move(shape), dtype);
  auto& converted = staging.emplace_back(bf.size());
  for (size_t i = 0; i < converted.size(); ++i)
    converted[i] = vt::F32ToF16(vt::BF16ToF32(bf[i]));
  return MakeDevRaw16(b, q, converted, std::move(shape), dtype);
}

// (Former SubRows/SubVec qkv row-slicers removed: the qkv projection now folds
// to ONE MatmulBT over the resident merged qkv_w [3H,H] + merged-bias epilogue
// + contiguous QkvSplit — see models::FusedMergedQkvBiasSplit, Tier C2.)

struct DevBlock {
  DevW norm1_w, norm1_b, norm2_w, norm2_b;
  DevW qkv_w, qkv_b;  // qkv_w [3H,H], qkv_b [3H]
  DevW proj_w, proj_b;
  DevW fc1_w, fc1_b, fc2_w, fc2_b;
};

struct DevMerger {
  bool use_postshuffle_norm = false;
  DevW norm_w, norm_b, fc1_w, fc1_b, fc2_w, fc2_b;
};

void UploadMerger(DevMerger& dm, Backend& b, Queue& q, const VisionMergerWeights& mw,
                  const Qwen3VLVisionConfig& cfg, DType dtype,
                  std::vector<std::vector<uint16_t>>& staging) {
  const int64_t H = cfg.hidden_size;
  const int64_t ctx4 = H * cfg.merge_unit();
  const int64_t D = cfg.out_hidden_size;
  dm.use_postshuffle_norm = mw.use_postshuffle_norm;
  const int64_t nd = mw.use_postshuffle_norm ? ctx4 : H;
  dm.norm_w = MakeDevWeight(b, q, mw.norm_w, {nd}, dtype, staging);
  dm.norm_b = MakeDevWeight(b, q, mw.norm_b, {nd}, dtype, staging);
  dm.fc1_w = MakeDevWeight(b, q, mw.fc1_w, {ctx4, ctx4}, dtype, staging);
  dm.fc1_b = MakeDevWeight(b, q, mw.fc1_b, {ctx4}, dtype, staging);
  dm.fc2_w = MakeDevWeight(b, q, mw.fc2_w, {D, ctx4}, dtype, staging);
  dm.fc2_b = MakeDevWeight(b, q, mw.fc2_b, {D}, dtype, staging);
}

}  // namespace

// The device-resident tower weights (opaque to callers; built once).
struct Qwen3VLVisionDeviceWeights {
  Backend* backend = nullptr;
  DType execution_dtype = DType::kBF16;
  DevW patch_proj_w, patch_proj_b;
  std::vector<float> pos_embed_w;  // legacy BF16 host interpolation only
  DevW device_pos_embed_w;        // explicit FP16 execution, device interpolation
  std::vector<DevBlock> blocks;
  DevMerger merger;
  std::vector<DevMerger> deepstack_mergers;
};

// --- host precompute: pos-embed bilinear interp + spatial-merge reorder -------
// pos_embed_interpolate_native (qwen3_vl.py:277-344) for a single (t,h,w).
std::vector<float> VisionPosEmbedInterpolate(const std::vector<float>& pos_embed_w,
                                             const std::array<int64_t, 3>& grid_thw,
                                             const Qwen3VLVisionConfig& cfg) {
  const int64_t t = grid_thw[0], h = grid_thw[1], w = grid_thw[2];
  const int64_t H = cfg.hidden_size;
  const int64_t G = cfg.num_grid_per_side();
  const int64_t m = cfg.spatial_merge_size;
  // linspace(0, G-1, n) in f32.
  auto linspace = [](int64_t n, int64_t g) {
    std::vector<float> v(static_cast<size_t>(n));
    if (n == 1) {
      v[0] = 0.0f;
      return v;
    }
    for (int64_t i = 0; i < n; ++i)
      v[static_cast<size_t>(i)] =
          static_cast<float>(i) * static_cast<float>(g - 1) / static_cast<float>(n - 1);
    return v;
  };
  std::vector<float> h_idx = linspace(h, G), w_idx = linspace(w, G);
  const int64_t hm = h / m, wm = w / m;
  // one frame [h*w, H] in spatial-merge order, then repeated t times.
  std::vector<float> frame(static_cast<size_t>(h) * w * H);
  // r enumerates (bi,bj,li,lj) C-order; source (i=bi*m+li, j=bj*m+lj).
  for (int64_t bi = 0; bi < hm; ++bi)
    for (int64_t bj = 0; bj < wm; ++bj)
      for (int64_t li = 0; li < m; ++li)
        for (int64_t lj = 0; lj < m; ++lj) {
          const int64_t i = bi * m + li, j = bj * m + lj;
          const int64_t r = ((bi * wm + bj) * m + li) * m + lj;
          const float hf = h_idx[static_cast<size_t>(i)], wf = w_idx[static_cast<size_t>(j)];
          const int64_t h_floor = static_cast<int64_t>(std::floor(hf));
          const int64_t w_floor = static_cast<int64_t>(std::floor(wf));
          const int64_t h_ceil = std::min(h_floor + 1, G - 1);
          const int64_t w_ceil = std::min(w_floor + 1, G - 1);
          const float dh = hf - static_cast<float>(h_floor);
          const float dw = wf - static_cast<float>(w_floor);
          const float w11 = dh * dw, w10 = dh - w11, w01 = dw - w11, w00 = 1.0f - dh - w01;
          const int64_t i00 = h_floor * G + w_floor, i01 = h_floor * G + w_ceil;
          const int64_t i10 = h_ceil * G + w_floor, i11 = h_ceil * G + w_ceil;
          float* dst = &frame[static_cast<size_t>(r) * H];
          const float* e00 = &pos_embed_w[static_cast<size_t>(i00) * H];
          const float* e01 = &pos_embed_w[static_cast<size_t>(i01) * H];
          const float* e10 = &pos_embed_w[static_cast<size_t>(i10) * H];
          const float* e11 = &pos_embed_w[static_cast<size_t>(i11) * H];
          for (int64_t d = 0; d < H; ++d)
            dst[d] = w00 * e00[d] + w01 * e01[d] + w10 * e10[d] + w11 * e11[d];
        }
  std::vector<float> out(static_cast<size_t>(t) * h * w * H);
  for (int64_t f = 0; f < t; ++f)
    std::memcpy(&out[static_cast<size_t>(f) * h * w * H], frame.data(),
                frame.size() * sizeof(float));
  return out;
}

// --- host precompute: vision rope cos|sin ([L, head_dim/2] each) --------------
// rot_pos_ids (:640-665) + rot_pos_emb (:667-683). partial_rotary_factor 0.5:
// rotary_dim = head_dim/2; inv_freq over rotary_dim/2 = head_dim/4 freqs, each
// spatial axis (h,w) contributes head_dim/4 → cos|sin width = head_dim/2.
void VisionRopeCosSin(const std::array<int64_t, 3>& grid_thw, const Qwen3VLVisionConfig& cfg,
                      std::vector<float>* cos, std::vector<float>* sin) {
  const int64_t t = grid_thw[0], h = grid_thw[1], w = grid_thw[2];
  const int64_t m = cfg.spatial_merge_size;
  const int64_t head_dim = cfg.head_dim();
  const int64_t rotary_dim = head_dim / 2;   // partial 0.5
  const int64_t nfreq = rotary_dim / 2;      // per-axis frequency count (=head_dim/4)
  const int64_t half = head_dim / 2;         // cos|sin width
  const double base = 10000.0;
  std::vector<double> inv_freq(static_cast<size_t>(nfreq));
  for (int64_t i = 0; i < nfreq; ++i)
    inv_freq[static_cast<size_t>(i)] =
        1.0 / std::pow(base, static_cast<double>(2 * i) / static_cast<double>(rotary_dim));
  const int64_t hm = h / m, wm = w / m;
  const int64_t L = t * h * w;
  cos->assign(static_cast<size_t>(L) * half, 0.0f);
  sin->assign(static_cast<size_t>(L) * half, 0.0f);
  // per-frame pos_ids [(bi,bj,li,lj)] -> hpos=bi*m+li, wpos=bj*m+lj; cos[r] =
  // concat(cos(hpos*inv_freq), cos(wpos*inv_freq)).
  for (int64_t f = 0; f < t; ++f)
    for (int64_t bi = 0; bi < hm; ++bi)
      for (int64_t bj = 0; bj < wm; ++bj)
        for (int64_t li = 0; li < m; ++li)
          for (int64_t lj = 0; lj < m; ++lj) {
            const int64_t hpos = bi * m + li, wpos = bj * m + lj;
            const int64_t rframe = ((bi * wm + bj) * m + li) * m + lj;
            const int64_t r = f * (h * w) + rframe;
            float* cr = &(*cos)[static_cast<size_t>(r) * half];
            float* sr = &(*sin)[static_cast<size_t>(r) * half];
            for (int64_t i = 0; i < nfreq; ++i) {
              const double ah = static_cast<double>(hpos) * inv_freq[static_cast<size_t>(i)];
              const double aw = static_cast<double>(wpos) * inv_freq[static_cast<size_t>(i)];
              cr[i] = static_cast<float>(std::cos(ah));
              sr[i] = static_cast<float>(std::sin(ah));
              cr[nfreq + i] = static_cast<float>(std::cos(aw));
              sr[nfreq + i] = static_cast<float>(std::sin(aw));
            }
          }
}

// --- build the resident device weights (the ONE-TIME conversion + upload) -----
std::shared_ptr<Qwen3VLVisionDeviceWeights> PrepareVisionDeviceWeights(
    const Qwen3VLVisionWeights& w, const Qwen3VLVisionConfig& cfg, Backend& b,
    DType execution_dtype) {
  VT_CHECK(execution_dtype == DType::kBF16 || execution_dtype == DType::kF16,
           "qwen3-vl vision: execution dtype must be BF16 or FP16");
  // Destruction order matters: drain/destroy the queue before staging or device
  // weights are released, including exceptions during conversion/allocation.
  auto dw = std::make_shared<Qwen3VLVisionDeviceWeights>();
  std::vector<std::vector<uint16_t>> staging;
  Queue q = b.CreateQueue();
  struct PrepareQueue {
    Backend& b;
    Queue& q;
    bool finished = false;
    void Finish() { b.Synchronize(q); b.DestroyQueue(q); finished = true; }
    ~PrepareQueue() {
      if (!finished) {
        try { b.Synchronize(q); } catch (...) {}
        try { b.DestroyQueue(q); } catch (...) {}
      }
    }
  } queue_owner{b, q};
  VT_CHECK(execution_dtype != DType::kF16 || q.device.type == vt::DeviceType::kXPU,
           "qwen3-vl vision: explicit FP16 preparation requires XPU");
  dw->backend = &b;
  dw->execution_dtype = execution_dtype;
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  const int64_t patch_dim =
      cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
  auto upload = [&](const std::vector<uint16_t>& source, std::vector<int64_t> shape) {
    return MakeDevWeight(b, q, source, std::move(shape), execution_dtype, staging);
  };
  dw->patch_proj_w = upload(w.patch_proj_w, {H, patch_dim});
  dw->patch_proj_b = upload(w.patch_proj_b, {H});
  if (execution_dtype == DType::kF16) {
    std::vector<uint16_t> bits(w.pos_embed_w.size());
    for (size_t i = 0; i < bits.size(); ++i) bits[i] = vt::F32ToF16(w.pos_embed_w[i]);
    // These are already FP16 bits; upload without converting them as BF16.
    staging.push_back(std::move(bits));
    dw->device_pos_embed_w = MakeDevRaw16(
        b, q, staging.back(), {cfg.num_position_embeddings, H}, DType::kF16);
  } else {
    dw->pos_embed_w = w.pos_embed_w;
  }
  dw->blocks.resize(w.blocks.size());
  for (size_t l = 0; l < w.blocks.size(); ++l) {
    const VisionBlockWeights& bw = w.blocks[l];
    DevBlock& db = dw->blocks[l];
    db.norm1_w = upload(bw.norm1_w, {H});
    db.norm1_b = upload(bw.norm1_b, {H});
    db.norm2_w = upload(bw.norm2_w, {H});
    db.norm2_b = upload(bw.norm2_b, {H});
    db.qkv_w = upload(bw.qkv_w, {3 * H, H});  // fused, sliced at forward
    db.qkv_b = upload(bw.qkv_b, {3 * H});
    db.proj_w = upload(bw.proj_w, {H, H});
    db.proj_b = upload(bw.proj_b, {H});
    db.fc1_w = upload(bw.fc1_w, {I, H});
    db.fc1_b = upload(bw.fc1_b, {I});
    db.fc2_w = upload(bw.fc2_w, {H, I});
    db.fc2_b = upload(bw.fc2_b, {H});
  }
  UploadMerger(dw->merger, b, q, w.merger, cfg, execution_dtype, staging);
  dw->deepstack_mergers.resize(w.deepstack_mergers.size());
  for (size_t i = 0; i < w.deepstack_mergers.size(); ++i)
    UploadMerger(dw->deepstack_mergers[i], b, q, w.deepstack_mergers[i], cfg,
                 execution_dtype, staging);
  queue_owner.Finish();  // resident + ready for any later queue; release staging
  return dw;
}

namespace {

// One patch-merger (main or deepstack) on resident weights. in = current hidden
// [L, hidden] device bf16; returns [Nmerge, out_hidden] device bf16 into `out`.
void RunMerger(Backend& b, Queue& q, const DevMerger& mw, const Qwen3VLVisionConfig& cfg,
               Tensor hidden, int64_t L, Buf& out) {
  const int64_t H = cfg.hidden_size;
  const int64_t ctx4 = H * cfg.merge_unit();  // 4*context
  const int64_t Nm = L / cfg.merge_unit();
  const float eps = cfg.norm_eps;

  Buf normed(b, q, DType::kBF16, {L, H});
  Buf fc1(b, q, DType::kBF16, {Nm, ctx4});

  if (mw.use_postshuffle_norm) {
    // x.view(-1, ctx4) THEN norm over ctx4.
    Tensor xv = hidden;  // [L,H] contiguous == [Nm,ctx4] reinterpret
    xv.rank = 2; xv.shape[0] = Nm; xv.shape[1] = ctx4; xv.stride[0] = ctx4; xv.stride[1] = 1;
    Buf nrm(b, q, DType::kBF16, {Nm, ctx4});
    vt::LayerNorm(q, nrm.tensor(), xv, &mw.norm_w.tensor(), &mw.norm_b.tensor(),
                  vt::LayerNormArgs{eps});
    LinearBias(q, fc1, nrm.tensor(), mw.fc1_w.tensor(), &mw.fc1_b.tensor());
  } else {
    // norm over context_dim (H) THEN view(-1, ctx4).
    vt::LayerNorm(q, normed.tensor(), hidden, &mw.norm_w.tensor(), &mw.norm_b.tensor(),
                  vt::LayerNormArgs{eps});
    Tensor nv = normed.tensor();  // [L,H] -> [Nm,ctx4]
    nv.rank = 2; nv.shape[0] = Nm; nv.shape[1] = ctx4; nv.stride[0] = ctx4; nv.stride[1] = 1;
    LinearBias(q, fc1, nv, mw.fc1_w.tensor(), &mw.fc1_b.tensor());
  }
  vt::GeluErf(q, fc1.tensor(), fc1.tensor());
  LinearBias(q, out, fc1.tensor(), mw.fc2_w.tensor(), &mw.fc2_b.tensor());
}

}  // namespace

// The typed XPU route reuses the same block/merger structure with reference
// FP16 projection/bias ordering. The legacy BF16 forward below is unchanged.
namespace {
struct VisionEvent {
  Backend& backend;
  vt::Event event;
  explicit VisionEvent(Backend& b) : backend(b), event(b.CreateEvent(true)) {}
  ~VisionEvent() { try { backend.DestroyEvent(event); } catch (...) {} }
};
void CheckVisionQueue(Backend& b, Queue& q, Backend* owner, const vt::Device& device) {
  VT_CHECK(&b==owner && q.device==device,
           "qwen3-vl vision: workspace/output belongs to another backend/device");
}
void CaptureF16(Buf& buffer, Queue& q, std::vector<float>& out) {
  std::vector<uint16_t> bits(buffer.bytes/2);
  buffer.Download(q,bits.data());out.resize(bits.size());
  for (size_t i=0;i<bits.size();++i) out[i]=vt::F16ToF32(bits[i]);
}
void LinearF16(Queue& q, Tensor& out, const Tensor& x, const DevW& w, const DevW& bias) {
  vt::MatmulDenseF16(q,out,x,w.tensor(),&bias.tensor());
}
}

struct Qwen3VLVisionWorkspace {
  Backend& backend;
  vt::Device device;
  uint64_t queue_id;
  Qwen3VLVisionConfig cfg;
  std::array<int64_t,3> grid;
  int64_t length;
  VisionEvent last_encode;
  std::mutex submission_mutex;
  std::unique_ptr<Buf> hidden,pos,base32,base16,ids,rope,n1,qb,kb,vb,qkv,ao,
                       projected,n2,fc1,fc2,merger_norm,merger_fc1;
  Qwen3VLVisionWorkspace(Backend& b, Queue& q, const Qwen3VLVisionConfig& c,
                         const std::array<int64_t,3>& g)
      : backend(b),device(q.device),queue_id(q.id),cfg(c),grid(g),
        length(g[1]*g[2]),last_encode(b) {}
  ~Qwen3VLVisionWorkspace() {
    // Drain before member buffers are freed, including abandoned requests.
    try { backend.SynchronizeEvent(last_encode.event); } catch (...) {}
  }
};

std::shared_ptr<Qwen3VLVisionWorkspace> PrepareVisionWorkspace(
    const std::array<int64_t,3>& grid, const Qwen3VLVisionConfig& cfg,
    Backend& b, Queue& q) {
  VT_CHECK(q.device.type==vt::DeviceType::kXPU && cfg.hidden_size==1152 &&
               cfg.num_heads==16 && cfg.head_dim()==72,
           "qwen3-vl vision: FP16 workspace requires XPU tower width 1152/16x72");
  VT_CHECK(grid[0]==1 && grid[1]>0 && grid[2]>0 && grid[1]<=32768 && grid[2]<=32768 &&
               grid[1]*grid[2]<=16384 && cfg.spatial_merge_size>0 &&
               cfg.spatial_merge_size<=128 && grid[1]%cfg.spatial_merge_size==0 &&
               grid[2]%cfg.spatial_merge_size==0,
           "qwen3-vl vision: invalid or oversized image grid");
  VT_CHECK(cfg.depth>0 && cfg.depth<=27 && cfg.intermediate_size>0 &&
               cfg.intermediate_size<=16384 && cfg.out_hidden_size>0 &&
               cfg.out_hidden_size<=16384 && cfg.num_position_embeddings>0 &&
               cfg.num_position_embeddings<=1048576 && cfg.patch_size>0 &&
               cfg.patch_size<=64 && cfg.temporal_patch_size>0 &&
               cfg.temporal_patch_size<=8 && cfg.in_channels>0 && cfg.in_channels<=4 &&
               std::isfinite(cfg.norm_eps) && cfg.norm_eps>0 && cfg.deepstack_visual_indexes.empty(),
           "qwen3-vl vision: unsupported FP16 image tower configuration");
  auto ws=std::make_shared<Qwen3VLVisionWorkspace>(b,q,cfg,grid);
  const int64_t L=ws->length,H=cfg.hidden_size,I=cfg.intermediate_size,
      merge=cfg.merge_unit(),Nm=L/merge,ctx=H*merge,P=std::max(grid[1],grid[2]);
  const auto make=[&](DType dt,std::vector<int64_t> shape) {
    return std::make_unique<Buf>(b,q,dt,std::move(shape));
  };
  ws->hidden=make(DType::kF16,{L,H});ws->pos=make(DType::kF16,{L,H});
  ws->base32=make(DType::kF32,{P,36});ws->base16=make(DType::kF16,{P,36});
  ws->ids=make(DType::kI32,{P});ws->rope=make(DType::kF16,{L,72});
  ws->n1=make(DType::kF16,{L,H});ws->qb=make(DType::kF16,{L,H});
  ws->kb=make(DType::kF16,{L,H});ws->vb=make(DType::kF16,{L,H});
  ws->qkv=make(DType::kF16,{L,3*H});ws->ao=make(DType::kF16,{L,16,72});
  ws->projected=make(DType::kF16,{L,H});ws->n2=make(DType::kF16,{L,H});
  ws->fc1=make(DType::kF16,{L,I});ws->fc2=make(DType::kF16,{L,H});
  ws->merger_norm=make(DType::kF16,{L,H});ws->merger_fc1=make(DType::kF16,{Nm,ctx});
  // Initialization only: the small static base positions never travel back to
  // the host. All per-image spatial ordering/rotation remains on the GPU.
  std::vector<int32_t> ids(static_cast<size_t>(P));
  for (int64_t i=0;i<P;++i) ids[static_cast<size_t>(i)]=static_cast<int32_t>(i);
  try {
    b.Copy(q,ws->ids->t.data,ids.data(),ids.size()*sizeof(int32_t));
    vt::RopeArgs args;args.rotary_dim=36;args.linear_scaling_factor=1;args.fp16_intermediates=true;
    vt::RopeCosSinCache(q,ws->base32->t,ws->ids->t,args);
    vt::CastF16(q,ws->base16->t,ws->base32->t);
    vt::VisionRopeGrid(q,ws->rope->t,ws->base16->t,{1,grid[1],grid[2],cfg.spatial_merge_size});
    b.RecordEvent(ws->last_encode.event,q);
  } catch (...) { b.Synchronize(q);throw; }
  return ws;
}

namespace {
void RunTypedVisionBlock(Qwen3VLVisionWorkspace& ws, const DevBlock& db,
                         Queue& q, int64_t block, Qwen3VLVisionCapture* cap,
                         const std::map<std::string,Tensor>* reference_outputs = nullptr) {
  const auto& cfg=ws.cfg;
  const int64_t L=ws.length,H=cfg.hidden_size;
  const bool selected=cap && std::find(cap->selected_blocks.begin(),cap->selected_blocks.end(),block)
      !=cap->selected_blocks.end();
  const auto capture=[&](const char* name,Buf& buffer) {
    if (selected) CaptureF16(buffer,q,cap->block_boundaries[block][name]);
    if (reference_outputs) {
      const auto it=reference_outputs->find(name);
      if (it!=reference_outputs->end()) {
        const auto& input=it->second;
        VT_CHECK(input.data && input.device==q.device && input.dtype==DType::kF16 &&
                     input.IsContiguous() && input.Bytes()==buffer.bytes,
                 "qwen3-vl vision: invalid diagnostic reference output");
        ws.backend.Copy(q,buffer.t.data,input.data,input.Bytes());
      }
    }
  };
  capture("input",*ws.hidden);
  vt::LayerNorm(q,ws.n1->t,ws.hidden->t,&db.norm1_w.t,&db.norm1_b.t,{cfg.norm_eps});
  capture("norm1",*ws.n1);
  LinearF16(q,ws.qkv->t,ws.n1->t,db.qkv_w,db.qkv_b);
  capture("qkv",*ws.qkv);
  vt::QkvSplit(q,ws.qb->t,ws.kb->t,ws.vb->t,ws.qkv->t);
  capture("q",*ws.qb);capture("k",*ws.kb);capture("v",*ws.vb);
  auto queries=ws.qb->t.View({L,16,72}),keys=ws.kb->t.View({L,16,72}),values=ws.vb->t.View({L,16,72});
  vt::VisionRopeApply(q,queries,keys,ws.rope->t);
  capture("rotated_q",*ws.qb);capture("rotated_k",*ws.kb);
  vt::AttentionDenseFlash(q,ws.ao->t,queries,keys,values,TypedVisionAttentionArgs(cfg.head_dim()));
  capture("attention",*ws.ao);
  LinearF16(q,ws.projected->t,ws.ao->t.View({L,H}),db.proj_w,db.proj_b);
  capture("projection",*ws.projected);
  vt::Add(q,ws.hidden->t,ws.hidden->t,ws.projected->t);
  capture("residual1",*ws.hidden);
  vt::LayerNorm(q,ws.n2->t,ws.hidden->t,&db.norm2_w.t,&db.norm2_b.t,{cfg.norm_eps});
  capture("norm2",*ws.n2);
  LinearF16(q,ws.fc1->t,ws.n2->t,db.fc1_w,db.fc1_b);
  capture("fc1",*ws.fc1);
  vt::GeluTanh(q,ws.fc1->t,ws.fc1->t);
  capture("gelu",*ws.fc1);
  LinearF16(q,ws.fc2->t,ws.fc1->t,db.fc2_w,db.fc2_b);
  capture("fc2",*ws.fc2);
  vt::Add(q,ws.hidden->t,ws.hidden->t,ws.fc2->t);
  capture("output",*ws.hidden);
}

void RunTypedVisionMerger(Qwen3VLVisionWorkspace& ws, const DevMerger& mw,
                          Buf& output, Queue& q, Qwen3VLVisionCapture* cap,
                          const std::map<std::string,Tensor>* reference_outputs = nullptr) {
  const int64_t Nm=ws.length/ws.cfg.merge_unit(),ctx=ws.cfg.hidden_size*ws.cfg.merge_unit();
  const auto capture=[&](const char* name,Buf& buffer,std::vector<float>* destination) {
    if (destination) CaptureF16(buffer,q,*destination);
    if (reference_outputs) {
      const auto it=reference_outputs->find(name);
      if (it!=reference_outputs->end()) {
        const auto& input=it->second;
        VT_CHECK(input.data && input.device==q.device && input.dtype==DType::kF16 &&
                     input.IsContiguous() && input.Bytes()==buffer.bytes,
                 "qwen3-vl vision: invalid diagnostic merger reference output");
        ws.backend.Copy(q,buffer.t.data,input.data,input.Bytes());
      }
    }
  };
  capture("input",*ws.hidden,cap ? &cap->merger_input : nullptr);
  vt::LayerNorm(q,ws.merger_norm->t,ws.hidden->t,&mw.norm_w.t,&mw.norm_b.t,{ws.cfg.norm_eps});
  capture("norm",*ws.merger_norm,cap ? &cap->merger_norm_out : nullptr);
  LinearF16(q,ws.merger_fc1->t,ws.merger_norm->t.View({Nm,ctx}),mw.fc1_w,mw.fc1_b);
  capture("fc1",*ws.merger_fc1,cap ? &cap->merger_fc1_out : nullptr);
  vt::GeluErf(q,ws.merger_fc1->t,ws.merger_fc1->t);
  capture("gelu",*ws.merger_fc1,cap ? &cap->merger_gelu_out : nullptr);
  LinearF16(q,output.t,ws.merger_fc1->t,mw.fc2_w,mw.fc2_b);
  capture("output",output,cap ? &cap->merger_out : nullptr);
}
}

struct Qwen3VLVisionDeviceOutput::State {
  Backend& backend;
  vt::Device device;
  std::shared_ptr<const Qwen3VLVisionDeviceWeights> weights;
  std::unique_ptr<Buf> output;
  VisionEvent ready;
  std::mutex mutex;
  std::map<uint64_t,std::unique_ptr<VisionEvent>> consumers;
  State(Backend& b,Queue& q,std::shared_ptr<const Qwen3VLVisionDeviceWeights> w,
        int64_t rows,int64_t width)
      : backend(b),device(q.device),weights(std::move(w)),
        output(std::make_unique<Buf>(b,q,DType::kF16,std::vector<int64_t>{rows,width})),ready(b) {}
  ~State() {
    try { backend.SynchronizeEvent(ready.event); } catch (...) {}
    for (auto& [_,event] : consumers)
      try { backend.SynchronizeEvent(event->event); } catch (...) {}
  }
};
Qwen3VLVisionDeviceOutput::Qwen3VLVisionDeviceOutput(std::shared_ptr<State> state)
    : state_(std::move(state)) {
  VT_CHECK(state_,"qwen3-vl vision: output owner must not be null");
}
const Tensor& Qwen3VLVisionDeviceOutput::tensor() const { return state_->output->t; }
void Qwen3VLVisionDeviceOutput::WaitOn(Queue& q) const {
  VT_CHECK(q.device==state_->device,"qwen3-vl vision: output consumer on another device");
  state_->backend.QueueWaitEvent(q,state_->ready.event);
}
void Qwen3VLVisionDeviceOutput::RecordUse(Queue& q) const {
  VT_CHECK(q.device==state_->device,"qwen3-vl vision: output consumer on another device");
  std::lock_guard<std::mutex> lock(state_->mutex);
  for (auto it=state_->consumers.begin();it!=state_->consumers.end();) {
    if (state_->backend.QueryEvent(it->second->event)) it=state_->consumers.erase(it);
    else ++it;
  }
  auto& event=state_->consumers[q.id];
  if (!event) event=std::make_unique<VisionEvent>(state_->backend);
  state_->backend.RecordEvent(event->event,q);
}

Qwen3VLVisionDeviceOutput Qwen3VLVisionForwardDevice(const Tensor& pixels,
    std::shared_ptr<const Qwen3VLVisionDeviceWeights> dw,
    Qwen3VLVisionWorkspace& ws, Backend& b, Queue& q, Qwen3VLVisionCapture* cap) {
  // Serialize host submission to shared scratch, without waiting for the GPU.
  std::lock_guard<std::mutex> lock(ws.submission_mutex);
  CheckVisionQueue(b,q,&ws.backend,ws.device);
  VT_CHECK(q.id==ws.queue_id,"qwen3-vl vision: scratch is bound to its encode queue");
  VT_CHECK(dw && dw->backend==&b && dw->execution_dtype==DType::kF16,
           "qwen3-vl vision: device forward requires owned FP16 weights");
  const auto& cfg=ws.cfg;
  const int64_t L=ws.length,H=cfg.hidden_size,I=cfg.intermediate_size,
      Nm=L/cfg.merge_unit(),ctx=H*cfg.merge_unit(),D=cfg.out_hidden_size,
      patch_dim=cfg.in_channels*cfg.temporal_patch_size*cfg.patch_size*cfg.patch_size;
  VT_CHECK(pixels.rank==2 && pixels.shape[0]==L && pixels.shape[1]==patch_dim &&
               pixels.dtype==DType::kF16 && pixels.device==q.device &&
               pixels.IsContiguous() && pixels.data!=nullptr,
           "qwen3-vl vision: invalid typed input patches");
  const auto shape=[&](const DevW& w,std::initializer_list<int64_t> dims) {
    VT_CHECK(w.t.data && w.t.dtype==DType::kF16 && w.t.device==q.device &&
                 w.t.IsContiguous() && w.t.rank==static_cast<int>(dims.size()),
             "qwen3-vl vision: invalid prepared weight metadata");
    int axis=0;for (auto d : dims)
      VT_CHECK(w.t.shape[axis++]==d,"qwen3-vl vision: prepared weight geometry differs from workspace");
  };
  shape(dw->patch_proj_w,{H,patch_dim});shape(dw->patch_proj_b,{H});
  shape(dw->device_pos_embed_w,{cfg.num_position_embeddings,H});
  VT_CHECK(dw->blocks.size()>=static_cast<size_t>(cfg.depth) && dw->deepstack_mergers.empty() &&
               !dw->merger.use_postshuffle_norm,"qwen3-vl vision: unsupported typed tower/merger");
  for (int64_t l=0;l<cfg.depth;++l) {
    const auto& db=dw->blocks[static_cast<size_t>(l)];
    shape(db.norm1_w,{H});shape(db.norm1_b,{H});shape(db.norm2_w,{H});shape(db.norm2_b,{H});
    shape(db.qkv_w,{3*H,H});shape(db.qkv_b,{3*H});shape(db.proj_w,{H,H});shape(db.proj_b,{H});
    shape(db.fc1_w,{I,H});shape(db.fc1_b,{I});shape(db.fc2_w,{H,I});shape(db.fc2_b,{H});
  }
  const auto& mw=dw->merger;
  shape(mw.norm_w,{H});shape(mw.norm_b,{H});shape(mw.fc1_w,{ctx,ctx});
  shape(mw.fc1_b,{ctx});shape(mw.fc2_w,{D,ctx});shape(mw.fc2_b,{D});
  if (cap) for (auto block : cap->selected_blocks)
    VT_CHECK(block>=0 && block<cfg.depth,"qwen3-vl vision: invalid selected capture block");
  auto result=std::make_shared<Qwen3VLVisionDeviceOutput::State>(b,q,dw,Nm,D);
  try {
    LinearF16(q,ws.hidden->t,pixels,dw->patch_proj_w,dw->patch_proj_b);
    if (cap) CaptureF16(*ws.hidden,q,cap->patch_embed_out);
    vt::VisionPosEmbedInterpolate(q,ws.pos->t,dw->device_pos_embed_w.t,
        {1,ws.grid[1],ws.grid[2],cfg.num_grid_per_side(),cfg.spatial_merge_size});
    vt::Add(q,ws.hidden->t,ws.hidden->t,ws.pos->t);
    if (cap) {
      CaptureF16(*ws.pos,q,cap->pos_embeds);
      std::vector<float> rope;CaptureF16(*ws.rope,q,rope);
      cap->rotary_cos.resize(static_cast<size_t>(L)*36);cap->rotary_sin.resize(static_cast<size_t>(L)*36);
      for (int64_t i=0;i<L;++i) {
        std::copy_n(rope.data()+i*72,36,cap->rotary_cos.data()+i*36);
        std::copy_n(rope.data()+i*72+36,36,cap->rotary_sin.data()+i*36);
      }
      cap->deepstack_out.clear();
      cap->block_outputs.clear();
      cap->block_outputs.resize(static_cast<size_t>(cfg.depth));
      cap->block_boundaries.clear();
    }
    for (int64_t l=0;l<cfg.depth;++l) {
      RunTypedVisionBlock(ws,dw->blocks[static_cast<size_t>(l)],q,l,cap);
      if (cap && l==0) CaptureF16(*ws.hidden,q,cap->block0_out);
      if (cap && (cap->selected_blocks.empty() ||
          std::find(cap->selected_blocks.begin(),cap->selected_blocks.end(),l)!=cap->selected_blocks.end()))
        CaptureF16(*ws.hidden,q,cap->block_outputs[static_cast<size_t>(l)]);
    }
    RunTypedVisionMerger(ws,mw,*result->output,q,cap);
    b.RecordEvent(result->ready.event,q);b.RecordEvent(ws.last_encode.event,q);
  } catch (...) { b.Synchronize(q);throw; }
  return Qwen3VLVisionDeviceOutput(std::move(result));
}

void Qwen3VLVisionReplayBlockDevice(const Tensor& input,
    std::shared_ptr<const Qwen3VLVisionDeviceWeights> dw,
    Qwen3VLVisionWorkspace& ws, Backend& b, Queue& q,
    int64_t block, Qwen3VLVisionCapture& cap,
    const std::map<std::string,Tensor>* reference_outputs) {
  std::lock_guard<std::mutex> lock(ws.submission_mutex);
  CheckVisionQueue(b,q,&ws.backend,ws.device);
  VT_CHECK(q.id==ws.queue_id && dw && dw->backend==&b && dw->execution_dtype==DType::kF16,
           "qwen3-vl vision: replay requires matching FP16 weights/encode queue");
  VT_CHECK(block>=0 && block<ws.cfg.depth && dw->blocks.size()>=static_cast<size_t>(ws.cfg.depth),
           "qwen3-vl vision: invalid replay block");
  VT_CHECK(input.rank==2 && input.shape[0]==ws.length && input.shape[1]==ws.cfg.hidden_size &&
               input.dtype==DType::kF16 && input.device==q.device && input.IsContiguous() && input.data,
           "qwen3-vl vision: invalid replay input");
  if (reference_outputs) {
    const std::set<std::string> stages{"norm1","qkv","q","k","v","rotated_q","rotated_k",
                                     "attention","projection","residual1","norm2","fc1","gelu","fc2"};
    for (const auto& [name,_] : *reference_outputs)
      VT_CHECK(stages.contains(name),"qwen3-vl vision: unknown diagnostic reference stage");
  }
  cap.selected_blocks={block};cap.block_boundaries.clear();cap.block_outputs.clear();
  try {
    b.Copy(q,ws.hidden->t.data,input.data,input.Bytes());
    RunTypedVisionBlock(ws,dw->blocks[static_cast<size_t>(block)],q,block,&cap,reference_outputs);
    b.RecordEvent(ws.last_encode.event,q);
    b.Synchronize(q); // Diagnostic returns all host captures; never called by serving.
  } catch (...) { b.Synchronize(q);throw; }
}

void Qwen3VLVisionReplayMergerDevice(const Tensor& input,
    std::shared_ptr<const Qwen3VLVisionDeviceWeights> dw,
    Qwen3VLVisionWorkspace& ws, Backend& b, Queue& q, Qwen3VLVisionCapture& cap,
    const std::map<std::string,Tensor>* reference_outputs) {
  std::lock_guard<std::mutex> lock(ws.submission_mutex);
  CheckVisionQueue(b,q,&ws.backend,ws.device);
  VT_CHECK(q.id==ws.queue_id && dw && dw->backend==&b && dw->execution_dtype==DType::kF16 &&
               !dw->merger.use_postshuffle_norm,
           "qwen3-vl vision: merger replay requires matching FP16 weights/encode queue");
  VT_CHECK(input.rank==2 && input.shape[0]==ws.length && input.shape[1]==ws.cfg.hidden_size &&
               input.dtype==DType::kF16 && input.device==q.device && input.IsContiguous() && input.data,
           "qwen3-vl vision: invalid merger replay input");
  const auto& mw=dw->merger;
  const int64_t H=ws.cfg.hidden_size,ctx=H*ws.cfg.merge_unit(),D=ws.cfg.out_hidden_size;
  const auto shape=[&](const DevW& w,std::initializer_list<int64_t> dims) {
    VT_CHECK(w.t.data && w.t.dtype==DType::kF16 && w.t.device==q.device &&
                 w.t.IsContiguous() && w.t.rank==static_cast<int>(dims.size()),
             "qwen3-vl vision: invalid replay merger weight metadata");
    int axis=0;for (auto d:dims)
      VT_CHECK(w.t.shape[axis++]==d,"qwen3-vl vision: replay merger geometry differs");
  };
  shape(mw.norm_w,{H});shape(mw.norm_b,{H});shape(mw.fc1_w,{ctx,ctx});
  shape(mw.fc1_b,{ctx});shape(mw.fc2_w,{D,ctx});shape(mw.fc2_b,{D});
  if (reference_outputs) for (const auto& [name,_]:*reference_outputs)
    VT_CHECK(name=="norm" || name=="fc1" || name=="gelu",
             "qwen3-vl vision: unknown diagnostic merger reference stage");
  Buf output(b,q,DType::kF16,{ws.length/ws.cfg.merge_unit(),D});
  try {
    b.Copy(q,ws.hidden->t.data,input.data,input.Bytes());
    RunTypedVisionMerger(ws,mw,output,q,&cap,reference_outputs);
    b.RecordEvent(ws.last_encode.event,q);
    b.Synchronize(q); // Diagnostic host captures only; never called by serving.
  } catch (...) { b.Synchronize(q);throw; }
}

// --- the resident-weights forward (the fast/production path) ------------------
std::vector<float> Qwen3VLVisionForward(const std::vector<uint16_t>& pixel_values_bf16,
                                        const std::array<int64_t, 3>& grid_thw,
                                        const Qwen3VLVisionDeviceWeights& dw,
                                        const Qwen3VLVisionConfig& cfg, Backend& b,
                                        Qwen3VLVisionCapture* cap) {
  VT_CHECK(dw.execution_dtype == DType::kBF16,
           "qwen3-vl vision: legacy BF16 forward cannot consume FP16 weights");
  VT_CHECK(dw.backend == &b,
           "qwen3-vl vision: prepared weights belong to another backend");
  Queue q = b.CreateQueue();
  const int64_t H = cfg.hidden_size;
  const int64_t nh = cfg.num_heads;
  const int64_t hd = cfg.head_dim();
  const int64_t I = cfg.intermediate_size;
  const int64_t patch_dim =
      cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
  const int64_t L = grid_thw[0] * grid_thw[1] * grid_thw[2];
  const int64_t half = hd / 2;
  const float eps = cfg.norm_eps;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

  // --- inputs (the only per-image uploads) ------------------------------------
  Buf pix(b, q, DType::kBF16, {L, patch_dim}, pixel_values_bf16.data());

  // patch_embed: [L,patch_dim] @ proj_w[H,patch_dim]^T + bias -> [L,H].
  Buf hidden(b, q, DType::kBF16, {L, H});
  LinearBias(q, hidden, pix.tensor(), dw.patch_proj_w.tensor(), &dw.patch_proj_b.tensor());
  if (cap != nullptr) {
    cap->patch_embed_out.resize(static_cast<size_t>(L) * H);
    std::vector<uint16_t> tmp(static_cast<size_t>(L) * H);
    hidden.Download(q, tmp.data());
    for (size_t i = 0; i < tmp.size(); ++i) cap->patch_embed_out[i] = vt::BF16ToF32(tmp[i]);
  }

  // + pos_embeds (host interp, uploaded bf16).
  std::vector<float> pos = VisionPosEmbedInterpolate(dw.pos_embed_w, grid_thw, cfg);
  {
    Buf pe(b, q, DType::kBF16, {L, H});
    const auto pe_bf = ToBf16(pos);
    b.Copy(q, pe.tensor().data, pe_bf.data(), pe_bf.size() * sizeof(uint16_t));
    vt::Add(q, hidden.tensor(), hidden.tensor(), pe.tensor());
  }

  // vision rope cache [L,hd] bf16 = [cos(half)|sin(half)]; positions [0..L-1].
  std::vector<float> rcos, rsin;
  VisionRopeCosSin(grid_thw, cfg, &rcos, &rsin);
  std::vector<float> cache_f(static_cast<size_t>(L) * hd);
  for (int64_t r = 0; r < L; ++r) {
    std::memcpy(&cache_f[static_cast<size_t>(r) * hd], &rcos[static_cast<size_t>(r) * half],
                static_cast<size_t>(half) * sizeof(float));
    std::memcpy(&cache_f[static_cast<size_t>(r) * hd + half],
                &rsin[static_cast<size_t>(r) * half], static_cast<size_t>(half) * sizeof(float));
  }
  Buf cache(b, q, DType::kBF16, {L, hd});
  {
    const auto cache_bf = ToBf16(cache_f);
    b.Copy(q, cache.tensor().data, cache_bf.data(), cache_bf.size() * sizeof(uint16_t));
  }
  std::vector<int32_t> pos_ids(static_cast<size_t>(L));
  for (int64_t i = 0; i < L; ++i) pos_ids[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  Buf posb(b, q, DType::kI32, {L}, pos_ids.data());
  if (cap != nullptr) {
    cap->rotary_cos = rcos;
    cap->rotary_sin = rsin;
    cap->pos_embeds = pos;
  }

  // --- ViT blocks -------------------------------------------------------------
  vt::RopeArgs ra;
  ra.rotary_dim = static_cast<int>(hd);
  ra.is_neox_style = true;
  if (cap != nullptr) cap->deepstack_out.clear();
  std::vector<std::vector<float>> ds_features(cfg.deepstack_visual_indexes.size());

  // Per-block SCRATCH buffers hoisted OUT of the loop and reused every block —
  // every block writes them fresh (LayerNorm/GEMM outputs), so reuse is
  // BIT-IDENTICAL. This is THE tower lever: nsys attributed ~100% of the ~1.6 s
  // forward to per-op cudaMalloc/cudaFree (each Buf alloc/free synchronizes the
  // device on GB10; the ViT kernels themselves are ~1.6 ms total). Allocating
  // once collapses ~250 allocs/forward to a handful (mirror torch's caching
  // allocator, which makes vLLM's per-op tensors free).
  Buf n1(b, q, DType::kBF16, {L, H});
  Buf qb(b, q, DType::kBF16, {L, H}), kb(b, q, DType::kBF16, {L, H}),
      vb(b, q, DType::kBF16, {L, H});
  // C2 fold scratch: merged qkv GEMM output [L, 3H] (view-split into qb/kb/vb).
  Buf qkv(b, q, DType::kBF16, {L, 3 * H});
  Buf ao(b, q, DType::kBF16, {L, nh, hd});
  Buf attn(b, q, DType::kBF16, {L, H});
  Buf n2(b, q, DType::kBF16, {L, H});
  Buf f1(b, q, DType::kBF16, {L, I});
  Buf f2(b, q, DType::kBF16, {L, H});

  // Vision self-attention kernel selection (per-forward; the forward runs once per
  // image so getenv cost is nil). DEFAULT: vt::AttentionDenseFlash — the
  // SHARED-MEMORY-TILED flash kernel (spec §14/§16). Its per-warp online-softmax
  // recurrence is BYTE-FOR-BYTE the AttentionDenseFast (AttentionWarpKernel) math —
  // identical per-lane head_dim grouping, identical butterfly reduction, identical
  // sequential j-order — only K/V are read from shared-memory tiles reused across a
  // block of query-warps instead of re-streamed from global per (query,head). So the
  // output is BIT-IDENTICAL to the warp kernel (token-identical, goldens unchanged),
  // while killing AttentionDenseFast's O(t^2) redundant global K/V reads over the 784
  // non-causal patches. Same-binary A/B knobs: VT_QWEN3VL_ATTN_WARP=1 → warp
  // AttentionDenseFast (the pre-§16 default); VT_QWEN3VL_ATTN_EAGER=1 → naive
  // kAttention. kAttention (text/audio decode) is untouched by all three.
  const int vis_attn = [] {
    const char* e = std::getenv("VT_QWEN3VL_ATTN_EAGER");
    if (e != nullptr && e[0] == '1') return 0;  // naive kAttention
    const char* w = std::getenv("VT_QWEN3VL_ATTN_WARP");
    if (w != nullptr && w[0] == '1') return 1;  // warp AttentionDenseFast
    return 2;                                   // flash-tiled AttentionDenseFlash (default)
  }();

  for (int64_t l = 0; l < cfg.depth; ++l) {
    const DevBlock& db = dw.blocks[static_cast<size_t>(l)];
    // norm1
    vt::LayerNorm(q, n1.tensor(), hidden.tensor(), &db.norm1_w.tensor(), &db.norm1_b.tensor(),
                  vt::LayerNormArgs{eps});
    // qkv (C2 fold): ONE MatmulBT over the resident merged qkv_w [3H,H] + a fused
    // merged [3H] BIAS epilogue, then a contiguous QkvSplit into qb/kb/vb [L,H].
    // BIT-identical to the prior 3x {LinearBias(row-slice)} (merged GEMM math ==
    // per-slice GEMM math; [3H] bias add broadcasts per column == 3x [H] adds;
    // QkvSplit is a pure contiguous copy). The merged-bias epilogue is the NEW
    // C2 piece vs the text bf16 merged-QKV (which carries no qkv bias).
    {
      Tensor qbias = db.qkv_b.tensor();
      models::FusedMergedQkvBiasSplit(q, qkv.tensor(), qb.tensor(), kb.tensor(),
                                      vb.tensor(), n1.tensor(), db.qkv_w.tensor(),
                                      &qbias);
    }
    // rope on q,k viewed [L,nh,hd].
    Tensor q3 = qb.tensor(); q3.rank = 3; q3.shape[0] = L; q3.shape[1] = nh; q3.shape[2] = hd;
    q3.stride[0] = nh * hd; q3.stride[1] = hd; q3.stride[2] = 1;
    Tensor k3 = kb.tensor(); k3.rank = 3; k3.shape[0] = L; k3.shape[1] = nh; k3.shape[2] = hd;
    k3.stride[0] = nh * hd; k3.stride[1] = hd; k3.stride[2] = 1;
    vt::RopeFromCache(q, q3, &k3, posb.tensor(), cache.tensor(), ra);
    // Non-causal attention, WINDOWED PER FRAME (image grid_t==1 == single window).
    Tensor v3 = vb.tensor(); v3.rank = 3; v3.shape[0] = L; v3.shape[1] = nh; v3.shape[2] = hd;
    v3.stride[0] = nh * hd; v3.stride[1] = hd; v3.stride[2] = 1;
    {
      const int64_t nframes = grid_thw[0];
      const int64_t hw = grid_thw[1] * grid_thw[2];  // patches per frame
      const size_t elt = vt::SizeOf(DType::kBF16);
      auto frame_slice = [&](const Tensor& src, int64_t f) -> Tensor {
        Tensor s = src;
        s.shape[0] = hw;
        s.data = static_cast<char*>(src.data) +
                 static_cast<size_t>(f * hw * nh * hd) * elt;
        return s;
      };
      for (int64_t f = 0; f < nframes; ++f) {
        Tensor qf = frame_slice(q3, f), kf = frame_slice(k3, f),
               vf = frame_slice(v3, f), aof = frame_slice(ao.tensor(), f);
        // Dense non-causal attention, head_dim 72, windowed per frame (image
        // grid_t==1 == single 784-patch window). Default flash-tiled (byte-identical
        // to warp; see vis_attn above). kAttention (text/audio) is untouched.
        const vt::AttentionArgs aargs{scale, /*causal=*/false};
        // VT-ATTN-NAIVE: the EAGER rung of the same-binary A/B above, reachable
        // only with VT_QWEN3VL_ATTN_EAGER=1, which nothing in the tree sets. The
        // default is the flash-tiled rung two branches down. Rerouting this arm
        // would delete the baseline the other two are measured against (#1544).
        if (vis_attn == 0)
          vt::Attention(q, aof, qf, kf, vf, aargs);
        else if (vis_attn == 1)
          vt::AttentionDenseFast(q, aof, qf, kf, vf, aargs);
        else
          vt::AttentionDenseFlash(q, aof, qf, kf, vf, aargs);
      }
    }
    // proj + residual.
    Tensor ao2 = ao.tensor(); ao2.rank = 2; ao2.shape[0] = L; ao2.shape[1] = H;
    ao2.stride[0] = H; ao2.stride[1] = 1;
    LinearBias(q, attn, ao2, db.proj_w.tensor(), &db.proj_b.tensor());
    vt::Add(q, hidden.tensor(), hidden.tensor(), attn.tensor());
    // norm2 + MLP + residual.
    vt::LayerNorm(q, n2.tensor(), hidden.tensor(), &db.norm2_w.tensor(), &db.norm2_b.tensor(),
                  vt::LayerNormArgs{eps});
    LinearBias(q, f1, n2.tensor(), db.fc1_w.tensor(), &db.fc1_b.tensor());
    vt::GeluTanh(q, f1.tensor(), f1.tensor());
    LinearBias(q, f2, f1.tensor(), db.fc2_w.tensor(), &db.fc2_b.tensor());
    vt::Add(q, hidden.tensor(), hidden.tensor(), f2.tensor());

    if (cap != nullptr && l == 0) {
      cap->block0_out.resize(static_cast<size_t>(L) * H);
      std::vector<uint16_t> tmp(static_cast<size_t>(L) * H);
      hidden.Download(q, tmp.data());
      for (size_t i = 0; i < tmp.size(); ++i) cap->block0_out[i] = vt::BF16ToF32(tmp[i]);
    }
    // deepstack tap after this block?
    for (size_t di = 0; di < cfg.deepstack_visual_indexes.size(); ++di) {
      if (cfg.deepstack_visual_indexes[di] == static_cast<int>(l)) {
        const int64_t Nm = L / cfg.merge_unit();
        Buf dsout(b, q, DType::kBF16, {Nm, cfg.out_hidden_size});
        RunMerger(b, q, dw.deepstack_mergers[di], cfg, hidden.tensor(), L, dsout);
        std::vector<uint16_t> tmp(static_cast<size_t>(Nm) * cfg.out_hidden_size);
        dsout.Download(q, tmp.data());
        std::vector<float> f(tmp.size());
        for (size_t i = 0; i < tmp.size(); ++i) f[i] = vt::BF16ToF32(tmp[i]);
        ds_features[di] = f;
        if (cap != nullptr) {
          if (cap->deepstack_out.size() <= di) cap->deepstack_out.resize(di + 1);
          cap->deepstack_out[di] = std::move(f);
        }
      }
    }
  }

  // --- merger + deepstack concat -> [Nm, out_hidden*(1+ndeep)] -----------------
  const int64_t Nm = L / cfg.merge_unit();
  const int64_t D = cfg.out_hidden_size;
  const int64_t ndeep = static_cast<int64_t>(cfg.deepstack_visual_indexes.size());
  Buf mout(b, q, DType::kBF16, {Nm, D});
  RunMerger(b, q, dw.merger, cfg, hidden.tensor(), L, mout);
  std::vector<float> merger_f(static_cast<size_t>(Nm) * D);
  {
    std::vector<uint16_t> tmp(static_cast<size_t>(Nm) * D);
    mout.Download(q, tmp.data());
    for (size_t i = 0; i < tmp.size(); ++i) merger_f[i] = vt::BF16ToF32(tmp[i]);
  }
  if (cap != nullptr) cap->merger_out = merger_f;

  std::vector<std::vector<float>>& ds = ds_features;

  // concat: [merger | ds0 | ds1 | ds2] along dim1.
  const int64_t W = D * (1 + ndeep);
  std::vector<float> tower(static_cast<size_t>(Nm) * W);
  for (int64_t r = 0; r < Nm; ++r) {
    std::memcpy(&tower[static_cast<size_t>(r) * W], &merger_f[static_cast<size_t>(r) * D],
                static_cast<size_t>(D) * sizeof(float));
    for (int64_t di = 0; di < ndeep; ++di)
      std::memcpy(&tower[static_cast<size_t>(r) * W + (di + 1) * D],
                  &ds[static_cast<size_t>(di)][static_cast<size_t>(r) * D],
                  static_cast<size_t>(D) * sizeof(float));
  }

  b.DestroyQueue(q);
  return tower;
}

// --- host-weights overload: prepare-then-forward (BIT-IDENTICAL wrapper) -------
std::vector<float> Qwen3VLVisionForward(const std::vector<uint16_t>& pixel_values_bf16,
                                        const std::array<int64_t, 3>& grid_thw,
                                        const Qwen3VLVisionWeights& w,
                                        const Qwen3VLVisionConfig& cfg, Backend& b,
                                        Qwen3VLVisionCapture* cap) {
  using clock = std::chrono::steady_clock;
  const bool prof = std::getenv("VLLM_MM_TOWER_PROFILE") != nullptr;
  const auto t0 = clock::now();
  const std::shared_ptr<Qwen3VLVisionDeviceWeights> dw =
      PrepareVisionDeviceWeights(w, cfg, b);  // synchronizes internally
  const auto t1 = clock::now();
  std::vector<float> out = Qwen3VLVisionForward(pixel_values_bf16, grid_thw, *dw, cfg, b, cap);
  const auto t2 = clock::now();
  if (prof) {
    const double marshal_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double compute_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::fprintf(stderr,
                 "[VLLM_MM_TOWER_PROFILE] prepare(marshal)=%.1f ms  forward(compute)=%.1f ms  "
                 "total=%.1f ms\n",
                 marshal_ms, compute_ms, marshal_ms + compute_ms);
  }
  return out;
}

}  // namespace vllm::multimodal
