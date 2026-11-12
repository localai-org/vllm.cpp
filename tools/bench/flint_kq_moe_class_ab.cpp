// FLINT-KQ-CPU-PERF — the routed-expert compute-class A/B micro harness
// (spec .agents/specs/flint-kq-cpu-perf.md, stage 1).
//
// WHAT IT MEASURES. The MoE routed-expert phase of the flint keep-quant
// paged arm — gather -> {gate, up} -> SiLU-mul -> down -> scatter — timed at
// batch 1 and batch 8 for three compute classes on the SAME loaded weights:
//
//   (a) the current CIQ dispatch: per-expert `vt::MatmulBT` on the Q4_K
//       block towers, exactly the flint.cpp FlintMoeBlock routed loop;
//   (b) dequant-at-load: the same layer's experts decoded to bf16 Matmul-B
//       weights once, then the per-expert `vt::Matmul` loop the bf16 arm
//       runs (the tree has no grouped bf16 GEMM op — this IS the bf16
//       arm's routed phase, which is what its 3.25 tok/s record measured);
//   (c) NOT a separate leg: the Q8_K-style class the row brief names is
//       already the current dispatch — `cpu_quant_traits.cpp` maps Q4_K to
//       the Q8_K activation encoding, and `cpu_quant_gemm.cpp` quantizes
//       the activation once per call and runs one integer `vec_dot` per
//       weight row. There is no second kernel to reuse, so instead of a
//       fake third arm this harness measures the PROFILE SPLIT of the
//       existing kernel: activation-quantize time, raw scalar `vec_dot`
//       row rate, the weight-byte memcpy floor, and one bf16 GEMV.
//
// The weights come from the PRODUCTION loader (ModelRegistry::Load with
// VT_FLINT_KEEP_QUANT=1) — one real MoE layer of the published Q4_K_M
// container, never a hand-built tower. The routing is a fixed
// deterministic top-k pattern shared by every leg, so the legs differ only
// in the compute class.
//
// CI NEVER RUNS THIS: it loads a ~21 GB tree and spends seconds per leg,
// the same trade its tools/bench siblings record. It is compiled by the
// ordinary build so the one file a reader has to compile to re-derive the
// stage-1 table cannot rot behind an op-signature change.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"
#include "vllm/model_executor/models/flint.h"
#include "vllm/model_executor/models/flint_gguf_weights.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/quant.h"

#ifndef FLINT_GGUF_PATH
#define FLINT_GGUF_PATH "/mnt/models/flint-GGUF/flint-Q4_K_M.gguf"
#endif

namespace {

namespace dense_attn = vllm::dense_attn;

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::ResidentWeight;
using vt::DType;
using vt::Tensor;

struct Timer {
  std::chrono::steady_clock::time_point t0;
  void Start() { t0 = std::chrono::steady_clock::now(); }
  double Ms() const {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
        .count();
  }
};

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

long RssKb() {
  FILE* f = fopen("/proc/self/status", "r");
  if (!f) return -1;
  char line[256];
  long kb = -1;
  while (fgets(line, sizeof line, f))
    if (strncmp(line, "VmRSS:", 6) == 0) {
      sscanf(line + 6, "%ld", &kb);
      break;
    }
  fclose(f);
  return kb;
}

// GatherRows: the flint.cpp:712 helper (row-indexed byte gather).
void GatherRows(Dev d, void* dst, const Tensor& src,
                const std::vector<int32_t>& idx, int64_t row_elems) {
  const size_t rb = static_cast<size_t>(row_elems) * vt::SizeOf(src.dtype);
  auto* dp = static_cast<char*>(dst);
  const auto* sp = static_cast<const char*>(src.data);
  for (size_t s = 0; s < idx.size(); ++s)
    d.b.Copy(d.q, dp + s * rb, sp + static_cast<size_t>(idx[s]) * rb, rb);
}

// Matmul-B over a FlintLinear (the flint.cpp:735 dispatch seam, verbatim).
DBuf FlintLinearMatmul(Dev d, const Tensor& x, const vllm::FlintLinear& w,
                       DType out_dtype) {
  Tensor wt = ResidentWeight(d, w.w);
  if (vt::IsBlockQuant(wt.dtype)) {
    DBuf out(d, out_dtype, {x.shape[0], wt.shape[0]});
    vt::MatmulBT(d.q, out.t(), x, wt);
    return out;
  }
  DBuf out(d, out_dtype, {x.shape[0], wt.shape[1]});
  vt::Matmul(d.q, out.t(), x, wt);
  return out;
}

// One routed-expert phase in class (a): the flint.cpp FlintMoeBlock routed
// loop with the router already resolved (fixed ids/slots, shared by all
// legs). Per selected expert: gather -> gate/up MatmulBT -> MoeSiluMul ->
// down MatmulBT -> scatter.
double RoutedCiq(Dev d, const vllm::FlintMoeWeights& w,
                 const vllm::FlintParams& p, int64_t t,
                 const std::vector<int32_t>& hid,
                 const std::vector<int32_t>& /*slot*/, const Tensor& x) {
  const int64_t top_k = p.num_experts_per_tok;
  const int64_t e_count = p.num_experts;
  DBuf expert_out(d, DType::kBF16, {t, top_k, p.hidden_size});
  expert_out.Zero(d);
  Timer tm;
  tm.Start();
  for (int64_t ex = 0; ex < e_count; ++ex) {
    std::vector<int32_t> rows;
    std::vector<int32_t> slots;
    for (int64_t ti = 0; ti < t; ++ti)
      for (int64_t kk = 0; kk < top_k; ++kk) {
        const int64_t idx = ti * top_k + kk;
        if (hid[static_cast<size_t>(idx)] == static_cast<int32_t>(ex)) {
          rows.push_back(static_cast<int32_t>(ti));
          slots.push_back(static_cast<int32_t>(idx));
          break;
        }
      }
    if (rows.empty()) continue;
    const int64_t ne = static_cast<int64_t>(rows.size());
    DBuf gathered(d, DType::kBF16, {ne, p.hidden_size});
    GatherRows(d, gathered.ptr(), x, rows, p.hidden_size);
    DBuf g = FlintLinearMatmul(d, gathered.t(),
                               w.experts[static_cast<size_t>(ex)].gate_proj,
                               DType::kBF16);
    DBuf u = FlintLinearMatmul(d, gathered.t(),
                               w.experts[static_cast<size_t>(ex)].up_proj,
                               DType::kBF16);
    DBuf a(d, DType::kBF16, {ne, p.moe_intermediate_size});
    vt::MoeSiluMul(d.q, a.t(), g.t(), u.t());
    DBuf o = FlintLinearMatmul(d, a.t(),
                               w.experts[static_cast<size_t>(ex)].down_proj,
                               DType::kBF16);
    for (int64_t i = 0; i < ne; ++i) {
      d.b.Copy(d.q,
               static_cast<char*>(expert_out.ptr()) +
                   static_cast<size_t>(slots[static_cast<size_t>(i)]) *
                       static_cast<size_t>(p.hidden_size) * sizeof(uint16_t),
               static_cast<const char*>(o.ptr()) +
                   static_cast<size_t>(i) * static_cast<size_t>(p.hidden_size) *
                       sizeof(uint16_t),
               static_cast<size_t>(p.hidden_size) * sizeof(uint16_t));
    }
  }
  return tm.Ms();
}

// Dequant one block weight [N, K] to a bf16 Matmul-B [K, N] buffer.
std::vector<uint16_t> DequantToBf16(const Tensor& w, int64_t n, int64_t k) {
    const auto& qt = vt::cpu::QuantTraits(w.dtype);
    const size_t row_bytes = vt::RowSizeBytes(w.dtype, k);
  const auto* blocks = static_cast<const uint8_t*>(w.data);
  std::vector<uint16_t> out(static_cast<size_t>(n * k));
  std::vector<float> row(static_cast<size_t>(k));
  for (int64_t j = 0; j < n; ++j) {
    qt.to_float(blocks + static_cast<size_t>(j) * row_bytes, row.data(), k);
    for (int64_t i = 0; i < k; ++i)
      out[static_cast<size_t>(i * n + j)] =
          vt::F32ToBF16(row[static_cast<size_t>(i)]);
  }
  return out;
}

struct Bf16Experts {
  // Per expert: bf16 Matmul-B [K, N] gate, up, down.
  std::vector<std::vector<uint16_t>> gate, up, down;
};

Bf16Experts DequantLayer(Dev d, const vllm::FlintMoeWeights& w,
                         const vllm::FlintParams& p, double* dequant_ms,
                         long* rss_before, long* rss_after) {
  const int64_t h = p.hidden_size;
  const int64_t i_dim = p.moe_intermediate_size;
  Bf16Experts b;
  b.gate.resize(p.num_experts);
  b.up.resize(p.num_experts);
  b.down.resize(p.num_experts);
  *rss_before = RssKb();
  Timer tm;
  tm.Start();
  for (int64_t e = 0; e < p.num_experts; ++e) {
    const auto& ex = w.experts[static_cast<size_t>(e)];
    b.gate[static_cast<size_t>(e)] =
        DequantToBf16(ResidentWeight(d, ex.gate_proj.w), i_dim, h);
    b.up[static_cast<size_t>(e)] =
        DequantToBf16(ResidentWeight(d, ex.up_proj.w), i_dim, h);
    b.down[static_cast<size_t>(e)] =
        DequantToBf16(ResidentWeight(d, ex.down_proj.w), h, i_dim);
  }
  *dequant_ms = tm.Ms();
  *rss_after = RssKb();
  return b;
}

// The float-weight twin of FlintLinearMatmul above (its float path, with the
// weight as a raw bf16 [K, N] Matmul-B buffer).
DBuf Bf16LinearMatmul(Dev d, const Tensor& x, const std::vector<uint16_t>& wt,
                      int64_t k, int64_t n) {
  DBuf out(d, DType::kBF16, {x.shape[0], n});
  Tensor wv = Tensor::Contiguous(const_cast<uint16_t*>(wt.data()), DType::kBF16,
                                 d.q.device, {k, n});
  vt::Matmul(d.q, out.t(), x, wv);
  return out;
}

// One routed-expert phase in class (b): the bf16 arm's routed loop — the same
// gather/silu/scatter, the weights dequantized to bf16 Matmul-B [K, N].
double RoutedBf16(Dev d, const Bf16Experts& b, const vllm::FlintParams& p,
                  int64_t t, const std::vector<int32_t>& hid,
                  const std::vector<int32_t>& /*slot*/, const Tensor& x) {
  const int64_t top_k = p.num_experts_per_tok;
  const int64_t e_count = p.num_experts;
  DBuf expert_out(d, DType::kBF16, {t, top_k, p.hidden_size});
  expert_out.Zero(d);
  Timer tm;
  tm.Start();
  for (int64_t ex = 0; ex < e_count; ++ex) {
    std::vector<int32_t> rows;
    std::vector<int32_t> slots;
    for (int64_t ti = 0; ti < t; ++ti)
      for (int64_t kk = 0; kk < top_k; ++kk) {
        const int64_t idx = ti * top_k + kk;
        if (hid[static_cast<size_t>(idx)] == static_cast<int32_t>(ex)) {
          rows.push_back(static_cast<int32_t>(ti));
          slots.push_back(static_cast<int32_t>(idx));
          break;
        }
      }
    if (rows.empty()) continue;
    const int64_t ne = static_cast<int64_t>(rows.size());
    DBuf gathered(d, DType::kBF16, {ne, p.hidden_size});
    GatherRows(d, gathered.ptr(), x, rows, p.hidden_size);
    DBuf g = Bf16LinearMatmul(d, gathered.t(), b.gate[static_cast<size_t>(ex)],
                              p.hidden_size, p.moe_intermediate_size);
    DBuf u = Bf16LinearMatmul(d, gathered.t(), b.up[static_cast<size_t>(ex)],
                              p.hidden_size, p.moe_intermediate_size);
    DBuf a(d, DType::kBF16, {ne, p.moe_intermediate_size});
    vt::MoeSiluMul(d.q, a.t(), g.t(), u.t());
    DBuf o = Bf16LinearMatmul(d, a.t(), b.down[static_cast<size_t>(ex)],
                              p.moe_intermediate_size, p.hidden_size);
    for (int64_t i = 0; i < ne; ++i) {
      d.b.Copy(d.q,
               static_cast<char*>(expert_out.ptr()) +
                   static_cast<size_t>(slots[static_cast<size_t>(i)]) *
                       static_cast<size_t>(p.hidden_size) * sizeof(uint16_t),
               static_cast<const char*>(o.ptr()) +
                   static_cast<size_t>(i) * static_cast<size_t>(p.hidden_size) *
                       sizeof(uint16_t),
               static_cast<size_t>(p.hidden_size) * sizeof(uint16_t));
    }
  }
  return tm.Ms();
}

}  // namespace

int main() {
  setenv("VT_FLINT_KEEP_QUANT", "1", 1);
  const char* path = FLINT_GGUF_PATH;
  printf("loading %s (keep-quant, production loader)\n", path);
  long rss0 = RssKb();
  vllm::GgufFile gguf = vllm::GgufFile::Open(path);
  const vllm::HfConfig config = vllm::FlintHfConfigFromGguf(gguf);
  const vllm::FlintParams p = vllm::ParseFlintParams(config);
  vllm::ModelSource source = vllm::ModelSource::FromGguf(gguf, vt::DeviceType::kCPU);
  std::unique_ptr<vllm::LoadedModel> model = vllm::ModelRegistry::Load(config, source);
  const vllm::FlintWeights& w = vllm::FlintLoadedModelWeights(*model);
  long rss1 = RssKb();

  int64_t moe_layer = -1;
  for (int64_t l = 0; l < p.num_hidden_layers; ++l)
    if (!w.layers[static_cast<size_t>(l)].moe.experts.empty()) {
      moe_layer = l;
      break;
    }
  const vllm::FlintMoeWeights& mw = w.layers[static_cast<size_t>(moe_layer)].moe;
  printf("tree: %ld -> %ld KiB RSS; first MoE layer %ld, H=%ld E=%ld topk=%ld I=%ld\n",
         rss0, rss1, moe_layer, p.hidden_size, p.num_experts,
         p.num_experts_per_tok, p.moe_intermediate_size);

  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t h = p.hidden_size;

  // Deterministic routing: token i selects experts (7i+1 .. 7i+topk) mod E.
  std::vector<int32_t> hid, slot;
  auto build_routing = [&](int64_t t) {
    hid.clear();
    slot.clear();
    for (int64_t i = 0; i < t; ++i)
      for (int64_t kk = 0; kk < p.num_experts_per_tok; ++kk) {
        hid.push_back(static_cast<int32_t>((7 * (i + 1) + kk) % p.num_experts));
        slot.push_back(static_cast<int32_t>(i * p.num_experts_per_tok + kk));
      }
  };

  // The activation: bf16 gaussian-ish rows (values only shift the quantize
  // scale; the class timings are value-independent except denormals).
  std::vector<uint16_t> xbytes;
  auto build_x = [&](int64_t t) {
    xbytes.resize(static_cast<size_t>(t * h));
    uint64_t st = 0x9e3779b97f4a7c15ull;
    for (size_t i = 0; i < xbytes.size(); ++i) {
      st = st * 6364136223846793005ull + 1442695040888963407ull;
      float v = static_cast<float>(static_cast<double>(st >> 11) / 9007199254740992.0) - 0.5f;
      xbytes[i] = vt::F32ToBF16(v);
    }
  };

  // Class (b): dequant this one layer's experts once.
  double dequant_ms = 0;
  long rb = 0, ra = 0;
  Bf16Experts bf = DequantLayer(d, mw, p, &dequant_ms, &rb, &ra);
  const double layer_packed_mb =
      static_cast<double>(p.num_experts) * 3.0 *
      static_cast<double>(vt::RowSizeBytes(DType::kQ4_K, h)) *
      static_cast<double>(p.moe_intermediate_size) / (1024.0 * 1024.0);
  printf("\n== class (b) dequant cost ==\n");
  printf("one layer dequant: %.1f ms, RSS %+ld KiB (packed layer ~%.1f MB, bf16 layer ~%.1f MB)\n",
         dequant_ms, ra - rb, layer_packed_mb, layer_packed_mb * (2048.0 / 547.0));

  for (int64_t t : {1, 8}) {
    build_routing(t);
    build_x(t);
    DBuf x(d, DType::kBF16, {t, h});
    d.b.Copy(d.q, x.ptr(), xbytes.data(),
             xbytes.size() * sizeof(uint16_t));

    const int reps = 3;
    std::vector<double> a_ms, b_ms;
    // warmup
    RoutedCiq(d, mw, p, t, hid, slot, x.t());
    RoutedBf16(d, bf, p, t, hid, slot, x.t());
    for (int r = 0; r < reps; ++r) a_ms.push_back(RoutedCiq(d, mw, p, t, hid, slot, x.t()));
    for (int r = 0; r < reps; ++r) b_ms.push_back(RoutedBf16(d, bf, p, t, hid, slot, x.t()));

    // Profile split of class (a) at this batch: selected rows ne = t*topk.
    const int64_t ne = t * p.num_experts_per_tok;
    // (i) activation quantize: widen + from_float per selected row (what
    // every MatmulBTQuant call redoes).
    const Tensor gate0 =
        ResidentWeight(d, mw.experts[static_cast<size_t>(0)].gate_proj.w);
    const int64_t kg = gate0.shape[1];
    const auto& qtraits = vt::cpu::QuantTraits(gate0.dtype);
    const auto& act_traits = vt::cpu::QuantTraits(qtraits.vec_dot_type);
    const size_t act_row = vt::cpu::QuantActRowBytes(gate0.dtype, kg);
    std::vector<uint8_t> scratch(act_row);
    std::vector<float> rowv(static_cast<size_t>(kg));
    double q_ms;
    {
      Timer tm;
      tm.Start();
      for (int64_t r = 0; r < ne; ++r) {
        for (int64_t c = 0; c < kg; ++c) rowv[static_cast<size_t>(c)] = vt::BF16ToF32(xbytes[static_cast<size_t>(r * kg + c)]);
        act_traits.from_float(rowv.data(), scratch.data(), kg);
      }
      q_ms = tm.Ms();
    }
    // (ii) raw scalar vec_dot over ALL gate rows of ONE expert (the kernel's
    // inner loop, isolated).
    double vd_ms;
    {
      std::vector<uint8_t> act_all(static_cast<size_t>(act_row));
      for (int64_t c = 0; c < kg; ++c) rowv[static_cast<size_t>(c)] = vt::BF16ToF32(xbytes[c]);
      act_traits.from_float(rowv.data(), act_all.data(), kg);
      const size_t wrow = vt::RowSizeBytes(gate0.dtype, kg);
      const int64_t nrows = gate0.shape[0];
      Timer tm;
      tm.Start();
      for (int64_t j = 0; j < nrows; ++j) {
        float s = 0;
        qtraits.vec_dot(static_cast<int>(kg), &s, 0,
                        static_cast<const uint8_t*>(gate0.data) + static_cast<size_t>(j) * wrow,
                        0, act_all.data(), 0, 1);
      }
      vd_ms = tm.Ms();
    }
    // (iii) weight-byte memcpy floor for one gate tower.
    double mc_ms;
    {
      const size_t wrow = vt::RowSizeBytes(gate0.dtype, kg);
      const int64_t nrows = gate0.shape[0];
      std::vector<uint8_t> sink(static_cast<size_t>(nrows) * wrow);
      Timer tm;
      tm.Start();
      std::memcpy(sink.data(), gate0.data, sink.size());
      mc_ms = tm.Ms();
    }
    // (iv) one bf16 GEMV (class (b)'s inner op) at the gate shape.
    double gemv_ms;
    {
      DBuf x1 = DBuf(d, DType::kBF16, {1, h});
      d.b.Copy(d.q, x1.ptr(), xbytes.data(), static_cast<size_t>(h) * 2);
      Tensor wv = Tensor::Contiguous(
          const_cast<uint16_t*>(bf.gate[0].data()), DType::kBF16, d.q.device,
          {h, p.moe_intermediate_size});
      DBuf out = DBuf(d, DType::kBF16, {1, p.moe_intermediate_size});
      vt::Matmul(d.q, out.t(), x1.t(), wv);  // warm
      Timer tm;
      tm.Start();
      vt::Matmul(d.q, out.t(), x1.t(), wv);
      gemv_ms = tm.Ms();
    }

    printf("\n== routed-expert phase, batch %ld (%d reps, median) ==\n", t, reps);
    printf("(a) CIQ per-expert dispatch : %8.3f ms  (median of %zu)\n", Median(a_ms), a_ms.size());
    printf("(b) dequant bf16 per-expert : %8.3f ms  (median of %zu)\n", Median(b_ms), b_ms.size());
    printf("profile split of (a):\n");
    printf("  activation quantize x%ld rows (per-call cost, x3 projections x%ld experts): %.3f ms\n",
           ne, p.num_experts_per_tok, q_ms);
    printf("  scalar vec_dot, one gate tower (%ld rows): %.3f ms -> x3 towers x%ld experts = %.1f ms/token-layer\n",
           gate0.shape[0], vd_ms, p.num_experts_per_tok, vd_ms * 3 * p.num_experts_per_tok);
    printf("  weight memcpy floor, one gate tower: %.3f ms (%.1f MB)\n", mc_ms,
           static_cast<double>(gate0.shape[0]) * vt::RowSizeBytes(gate0.dtype, kg) / (1024.0 * 1024.0));
    printf("  bf16 GEMV one gate tower: %.3f ms\n", gemv_ms);
  }
  return 0;
}
