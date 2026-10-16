#include "xpu_common.h"
#include "xpu_kernels.h"
#include "vt/sample_common.h"
#include "vt/xpu_sampling.h"
#include <cstdio>
#include <cstdlib>
#include <limits>
namespace vt::xpu {
namespace {
constexpr float NegInf = -std::numeric_limits<float>::infinity();
constexpr size_t SamplingLanes = 128;

// Row reductions stay on the GPU even for the 248,320-token vocabulary.
// Mode 0: softmax, 1: log-softmax, 2: in-place min-p masking.
template<int Mode>
void Normalize(Queue& q, Tensor& out, const Tensor& logits, const Tensor* min_p = nullptr) {
  const int64_t rows = logits.shape[0], vocab = logits.shape[1];
  if (!rows || !vocab) return;
  const auto* src = static_cast<const float*>(logits.data);
  auto* dst = static_cast<float*>(out.data);
  const auto* mp = min_p ? static_cast<const float*>(min_p->data) : nullptr;
  NativeQueue(q).parallel_for(sycl::nd_range<1>(rows * SamplingLanes, SamplingLanes), [=](sycl::nd_item<1> item) {
    const int64_t row = item.get_group(0), lane = item.get_local_id(0);
    if constexpr (Mode == 2) { if (mp[row] <= 0) return; }
    float maximum = NegInf;
    for (int64_t j = lane; j < vocab; j += SamplingLanes) maximum = sycl::fmax(maximum, src[row * vocab + j]);
    maximum = sycl::reduce_over_group(item.get_group(), maximum, sycl::maximum<float>());
    float sum = 0;
    for (int64_t j = lane; j < vocab; j += SamplingLanes) sum += sycl::exp(src[row * vocab + j] - maximum);
    sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>());
    const float lse = maximum + sycl::log(sum);
    for (int64_t j = lane; j < vocab; j += SamplingLanes) {
      const auto index = row * vocab + j;
      if constexpr (Mode == 0) dst[index] = sycl::exp(src[index] - maximum) / sum;
      if constexpr (Mode == 1) dst[index] = src[index] - lse;
      if constexpr (Mode == 2)
        if (sycl::exp(src[index] - maximum) / sum < mp[row] * (1.0f / sum)) dst[index] = NegInf;
    }
  });
}
template<bool Bias>
void SparseMask(Queue& q, Tensor& logits, const Tensor& rows, const Tensor& cols, const Tensor* biases) {
  const auto count = rows.Numel(), n = logits.shape[0], v = logits.shape[1];
  if (!count) return;
  const auto* rr = static_cast<const int32_t*>(rows.data);
  const auto* cc = static_cast<const int32_t*>(cols.data);
  const auto* bb = biases ? static_cast<const float*>(biases->data) : nullptr;
  auto* values = static_cast<float*>(logits.data);
  CheckDeviceMetadata(q, [=] {
    for (int64_t i = 0; i < count; ++i) if (rr[i] < 0 || rr[i] >= n || cc[i] < 0 || cc[i] >= v) return false;
    return true;
  }, "XPU sampling sparse index outside logits", {&rows, &cols});
  NativeQueue(q).parallel_for(sycl::range<1>(count), [=](sycl::id<1> item) {
    const int64_t i = item[0], index = int64_t(rr[i]) * v + cc[i];
    // Preserve the reference's addition order for repeated bias coordinates;
    // duplicate masks also have just one writer, without atomics or races.
    for (int64_t j = 0; j < i; ++j) if (rr[j] == rr[i] && cc[j] == cc[i]) return;
    if constexpr (Bias) {
      float value = values[index];
      for (int64_t j = i; j < count; ++j) if (rr[j] == rr[i] && cc[j] == cc[i]) value += bb[j];
      values[index] = value;
    } else values[index] = NegInf;
  });
}
}
void ApplyTemperatureKernel(Queue& q, Tensor& logits, const Tensor& temp, bool all_random) {
  TraceXpuOp(OpId::kApplyTemperature, q, {&logits, &temp});
  if (!logits.Numel()) return;
  const auto vocab = logits.shape[1];
  auto* values = static_cast<float*>(logits.data);
  const auto* temps = static_cast<const float*>(temp.data);
  NativeQueue(q).parallel_for(sycl::range<1>(logits.Numel()), [=](sycl::id<1> i) {
    float t = temps[i[0] / vocab];
    if (!all_random && t < kSamplingEps) t = 1;
    values[i] /= t;
  });
}
void ComputeProbsKernel(Queue& q, Tensor& probs, const Tensor& logits) {
  TraceXpuOp(OpId::kComputeProbs, q, {&probs, &logits});
  WithOutput(q, probs, {&logits}, [&](Tensor& target) { Normalize<0>(q, target, logits); });
}
void ComputeLogprobsKernel(Queue& q, Tensor& probs, const Tensor& logits) {
  TraceXpuOp(OpId::kComputeLogprobs, q, {&probs, &logits});
  WithOutput(q, probs, {&logits}, [&](Tensor& target) { Normalize<1>(q, target, logits); });
}
void ApplyMinPKernel(Queue& q, Tensor& logits, const Tensor& min_p) {
  TraceXpuOp(OpId::kApplyMinP, q, {&logits, &min_p});
  Normalize<2>(q, logits, logits, &min_p);
}

bool ApplyTopK20TopP(Queue& q, Tensor& logits, const Tensor& p) {
  constexpr int64_t K = 20, Lanes = 128, Tile = 256;
  const int64_t rows = logits.shape[0], vocab = logits.shape[1];
  const int64_t tiles = (vocab + Tile - 1) / Tile;
  if (rows < 1 || rows > 20 || vocab < K || tiles > Lanes * 16) return false;
  VT_CHECK(logits.rank == 2 && logits.dtype == DType::kF32 &&
               logits.IsContiguous() && logits.device == q.device &&
               p.rank == 1 && p.shape[0] == rows && p.dtype == DType::kF32 &&
               p.IsContiguous() && p.device == q.device,
           "XPU top-20 requires contiguous F32 logits and per-row top-p");
  auto* values = static_cast<float*>(logits.data);
  const auto* ps = static_cast<const float*>(p.data);
  const size_t candidate_count = size_t(rows * tiles * K);
  Scratch scratch(q.device, (candidate_count + size_t(rows * K + 2 * rows)) * sizeof(int32_t));
  auto* candidates = static_cast<int32_t*>(scratch.data);
  auto* selected = candidates + candidate_count;
  auto* fallback = selected + rows * K;
  auto* cutoffs = fallback + rows;

  const auto tile_event = NativeQueue(q).submit([&](sycl::handler& h) {
    sycl::local_accessor<float> work_values(Lanes, h);
    sycl::local_accessor<int32_t> work_ids(Lanes, h);
    h.parallel_for(sycl::nd_range<1>(rows * tiles * Lanes, Lanes),
                   [=](sycl::nd_item<1> item) {
      const int64_t group = item.get_group(0), row = group / tiles;
      const int64_t tile = group % tiles, lane = item.get_local_id(0);
      const int32_t a = int32_t(tile * Tile + lane);
      const int32_t b = int32_t(a + Lanes);
      float av = a < vocab ? values[row * vocab + a] : NegInf;
      float bv = b < vocab ? values[row * vocab + b] : NegInf;
      for (int rank = 0; rank < K; ++rank) {
        const bool first = av > bv || (av == bv && a < b);
        work_values[lane] = first ? av : bv;
        work_ids[lane] = first ? (a < vocab ? a : INT_MAX)
                               : (b < vocab ? b : INT_MAX);
        item.barrier(sycl::access::fence_space::local_space);
        for (int step = Lanes / 2; step; step /= 2) {
          if (lane < step &&
              (work_values[lane + step] > work_values[lane] ||
               (work_values[lane + step] == work_values[lane] &&
                work_ids[lane + step] < work_ids[lane]))) {
            work_values[lane] = work_values[lane + step];
            work_ids[lane] = work_ids[lane + step];
          }
          item.barrier(sycl::access::fence_space::local_space);
        }
        const int32_t winner = work_ids[0];
        if (lane == 0) candidates[group * K + rank] = winner;
        item.barrier(sycl::access::fence_space::local_space);
        if (winner == a) av = NegInf;
        if (winner == b) bv = NegInf;
      }
    });
  });
  RecordProfileEvent(q, "topk20_tile", tile_event);

  const auto merge_event = NativeQueue(q).submit([&](sycl::handler& h) {
    sycl::local_accessor<float> work_values(Lanes, h);
    sycl::local_accessor<int32_t> work_ids(Lanes, h);
    h.parallel_for(sycl::nd_range<1>(rows * Lanes, Lanes),
                   [=](sycl::nd_item<1> item) {
      const int64_t row = item.get_group(0), lane = item.get_local_id(0);
      int cursor[16] = {};
      for (int rank = 0; rank < K; ++rank) {
        float best = NegInf;
        int32_t id = INT_MAX;
        for (int slot = 0; slot < 16; ++slot) {
          const int64_t tile = lane + slot * Lanes;
          if (tile >= tiles || cursor[slot] >= K) continue;
          const int32_t candidate = candidates[(row * tiles + tile) * K + cursor[slot]];
          if (candidate == INT_MAX) continue;
          const float value = values[row * vocab + candidate];
          if (value > best || (value == best && candidate < id)) {
            best = value;
            id = candidate;
          }
        }
        work_values[lane] = best;
        work_ids[lane] = id;
        item.barrier(sycl::access::fence_space::local_space);
        for (int step = Lanes / 2; step; step /= 2) {
          if (lane < step &&
              (work_values[lane + step] > work_values[lane] ||
               (work_values[lane + step] == work_values[lane] &&
                work_ids[lane + step] < work_ids[lane]))) {
            work_values[lane] = work_values[lane + step];
            work_ids[lane] = work_ids[lane + step];
          }
          item.barrier(sycl::access::fence_space::local_space);
        }
        const int32_t winner = work_ids[0];
        if (lane == 0) selected[row * K + rank] = winner;
        item.barrier(sycl::access::fence_space::local_space);
        if (winner != INT_MAX && (winner / Tile) % Lanes == lane)
          ++cursor[(winner / Tile) / Lanes];
      }
    });
  });
  RecordProfileEvent(q, "topk20_merge", merge_event);

  const auto probe_event = NativeQueue(q).submit([&](sycl::handler& h) {
    sycl::local_accessor<int32_t> counts(Lanes, h);
    sycl::local_accessor<int32_t> invalids(Lanes, h);
    h.parallel_for(sycl::nd_range<1>(rows * Lanes, Lanes),
                   [=](sycl::nd_item<1> item) {
      const int64_t row = item.get_group(0), lane = item.get_local_id(0);
      const int32_t kth = selected[row * K + K - 1];
      const float threshold = kth == INT_MAX ? NegInf : values[row * vocab + kth];
      int32_t count = 0, invalid = kth == INT_MAX;
      for (int64_t col = lane; col < vocab; col += Lanes) {
        const float value = values[row * vocab + col];
        count += value >= threshold;
        invalid |= !sycl::isfinite(value);
      }
      counts[lane] = count;
      invalids[lane] = invalid;
      item.barrier(sycl::access::fence_space::local_space);
      for (int step = Lanes / 2; step; step /= 2) {
        if (lane < step) {
          counts[lane] += counts[lane + step];
          invalids[lane] |= invalids[lane + step];
        }
        item.barrier(sycl::access::fence_space::local_space);
      }
      if (lane == 0) {
        const float top_p = ps[row];
        float weight[K], sum = 0;
        const float maximum = values[row * vocab + selected[row * K]];
        for (int rank = 0; rank < K; ++rank) {
          weight[rank] = sycl::exp(values[row * vocab + selected[row * K + rank]] - maximum);
          sum += weight[rank];
        }
        int keep = 0;
        float before = 0;
        bool boundary = false;
        for (int rank = 0; rank < K; ++rank) {
          if (rank && sycl::fabs(before / sum - top_p) < 1e-5f) boundary = true;
          if (rank == 0 || before / sum < top_p) ++keep;
          before += weight[rank];
        }
        fallback[row] = (invalids[0] ? 1 : 0) | (counts[0] > K ? 2 : 0) |
                        (!(top_p > 0 && top_p <= 1) ? 4 : 0) |
                        (boundary ? 8 : 0);
        cutoffs[row] = selected[row * K + keep - 1];
      }
    });
  });
  RecordProfileEvent(q, "topk20_probe", probe_event);
  std::vector<int32_t> host_fallback(static_cast<size_t>(rows), 0);
  auto& backend = GetBackend(q.device.type);
  backend.Copy(q, host_fallback.data(), fallback, size_t(rows) * sizeof(int32_t));
  backend.Synchronize(q);
  if (std::getenv("VT_B70_FAST_TOPK_TRACE")) {
    std::fprintf(stderr, "topk20 fallback:");
    for (int32_t value : host_fallback) std::fprintf(stderr, " %d", value);
    std::fprintf(stderr, "\n");
  }
  if (std::any_of(host_fallback.begin(), host_fallback.end(),
                  [](int32_t value) { return value != 0; })) return false;

  const auto mask_event = NativeQueue(q).parallel_for(
      sycl::range<1>(rows * vocab), [=](sycl::id<1> item) {
    const int64_t row = item[0] / vocab, col = item[0] % vocab;
    const int32_t cutoff_id = cutoffs[row];
    const float cutoff = values[row * vocab + cutoff_id];
    const float value = values[item[0]];
    if (value < cutoff || (value == cutoff && col > cutoff_id))
      values[item[0]] = NegInf;
  });
  RecordProfileEvent(q, "topk20_mask", mask_event);
  backend.Synchronize(q);
  return true;
}
void ApplyPenaltiesKernel(Queue& q, Tensor& logits, const Tensor& prompt_mask,
                           const Tensor& counts, const Tensor& output_mask,
                           const Tensor& frequency, const Tensor& presence, const Tensor& repetition) {
  TraceXpuOp(OpId::kApplyPenalties, q, {&logits, &prompt_mask, &counts, &output_mask, &frequency, &presence, &repetition});
  if (!logits.Numel()) return;
  const auto vocab = logits.shape[1];
  auto* values = static_cast<float*>(logits.data);
  const auto* pm = static_cast<const int8_t*>(prompt_mask.data);
  const auto* om = static_cast<const int8_t*>(output_mask.data);
  const auto* oc = static_cast<const int32_t*>(counts.data);
  const auto* freq = static_cast<const float*>(frequency.data);
  const auto* pres = static_cast<const float*>(presence.data);
  const auto* rep = static_cast<const float*>(repetition.data);
  NativeQueue(q).parallel_for(sycl::range<1>(logits.Numel()), [=](sycl::id<1> item) {
    const auto i = item[0], row = i / vocab;
    float value = values[i];
    if (pm[i] || om[i]) value = value > 0 ? value / rep[row] : value * rep[row];
    value -= freq[row] * float(oc[i]); value -= pres[row] * float(om[i]);
    values[i] = value;
  });
}
void ApplyLogitBiasKernel(Queue& q, Tensor& logits, const Tensor& rows, const Tensor& cols, const Tensor& biases) {
  TraceXpuOp(OpId::kApplyLogitBias, q, {&logits, &rows, &cols, &biases});
  SparseMask<true>(q, logits, rows, cols, &biases);
}
void ApplyTokenMaskKernel(Queue& q, Tensor& logits, const Tensor& rows, const Tensor& cols) {
  TraceXpuOp(OpId::kApplyTokenMask, q, {&logits, &rows, &cols});
  SparseMask<false>(q, logits, rows, cols, nullptr);
}
void ApplyAllowedTokenIdsKernel(Queue& q, Tensor& logits, const Tensor& mask) {
  TraceXpuOp(OpId::kApplyAllowedTokenIds, q, {&logits, &mask});
  if (!logits.Numel()) return;
  auto* values = static_cast<float*>(logits.data);
  const auto* m = static_cast<const int8_t*>(mask.data);
  NativeQueue(q).parallel_for(sycl::range<1>(logits.Numel()), [=](sycl::id<1> i) { if (m[i]) values[i] = NegInf; });
}
void ApplyTopKTopPKernel(Queue& q, Tensor& logits, const Tensor* k, const Tensor* p) {
  TraceXpuOp(OpId::kApplyTopKTopP, q, {&logits, k, p});
  const int64_t rows = logits.shape[0], vocab = logits.shape[1];
  if (!rows || !vocab) return;
  // Process at most four rows at once, with a fixed 16 MiB context workspace.
  // Merge permutations instead of moving logits: masking only starts after
  // all ranks and cumulative probabilities have been computed.
  constexpr int64_t Tile = 128, Workspace = 16 * 1024 * 1024;
  const int64_t tiles = (vocab + Tile - 1) / Tile;
  const int64_t row_bytes = (2 * vocab + tiles + 2) * sizeof(int32_t);
  VT_CHECK(row_bytes <= Workspace, "XPU top-k/top-p vocabulary exceeds bounded workspace");
  const int64_t batch = std::min(int64_t{4}, Workspace / row_bytes);
  auto* values = static_cast<float*>(logits.data);
  const auto* ks = k ? static_cast<const int32_t*>(k->data) : nullptr;
  const auto* ps = p ? static_cast<const float*>(p->data) : nullptr;
  const bool available = WithSamplingWorkspace(q, Workspace, [&](void* storage) {
    for (int64_t start = 0; start < rows; start += batch) {
      const int64_t n = std::min(batch, rows - start);
      auto* first = static_cast<int32_t*>(storage);
      auto* second = first + n * vocab;
      auto* totals = reinterpret_cast<float*>(second + n * vocab);
      auto* stats = totals + n * tiles;
      auto* data = values + start * vocab;
      NativeQueue(q).parallel_for(sycl::range<1>(n * vocab), [=](sycl::id<1> i) { first[i] = i[0] % vocab; });
      // Stable bottom-up merge by binary rank. Each item has exactly one
      // destination, including equal logits (lowest original index first).
      for (int64_t width = 1; width < vocab; width *= 2) {
        const auto* src = first; auto* dst = second;
        NativeQueue(q).parallel_for(sycl::range<1>(n * vocab), [=](sycl::id<1> item) {
          const int64_t row = item[0] / vocab, col = item[0] % vocab;
          const int64_t base = (col / (2 * width)) * 2 * width;
          const int64_t middle = sycl::min(base + width, vocab), end = sycl::min(base + 2 * width, vocab);
          const bool right = col >= middle;
          const int64_t other = right ? base : middle, stop = right ? middle : end;
          const int32_t index = src[row * vocab + col];
          const float value = data[row * vocab + index];
          int64_t lo = other, hi = stop;
          while (lo < hi) {
            const auto mid = (lo + hi) / 2;
            const int32_t candidate = src[row * vocab + mid];
            const float c = data[row * vocab + candidate];
            const bool equal = c == value || (sycl::isnan(c) && sycl::isnan(value));
            if ((sycl::isnan(value) && !sycl::isnan(c)) || c < value || (equal && candidate < index)) lo = mid + 1; else hi = mid;
          }
          const auto rank = base + col - (right ? middle : base) + lo - other;
          dst[row * vocab + rank] = index;
        });
        std::swap(first, second);
      }
      const auto* order = first;
      // The unused permutation buffer becomes per-tile probability prefixes.
      auto* cdf = reinterpret_cast<float*>(second);
      NativeQueue(q).parallel_for(sycl::nd_range<1>(n * Tile, Tile), [=](sycl::nd_item<1> item) {
        const auto row = item.get_group(0), lane = item.get_local_id(0);
        const int64_t keep = ks ? ks[start + row] : vocab;
        const float threshold = keep >= 1 && keep < vocab ? data[row * vocab + order[row * vocab + vocab - keep]] : NegInf;
        const float maximum = data[row * vocab + order[row * vocab + vocab - 1]];
        float sum = 0;
        for (int64_t j = lane; j < vocab; j += Tile) {
          const float value = data[row * vocab + order[row * vocab + j]];
          sum += value >= threshold && value != NegInf ? sycl::exp(value - maximum) : 0;
        }
        sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>());
        if (!lane) { stats[row * 2] = threshold; stats[row * 2 + 1] = sum; }
      });
      if (ps) {
        NativeQueue(q).parallel_for(sycl::nd_range<1>(n * tiles * Tile, Tile), [=](sycl::nd_item<1> item) {
          const int64_t row = item.get_group(0) / tiles, tile = item.get_group(0) % tiles, lane = item.get_local_id(0);
          const int64_t j = tile * Tile + lane;
          const float maximum = data[row * vocab + order[row * vocab + vocab - 1]];
          const float value = j < vocab ? data[row * vocab + order[row * vocab + j]] : NegInf;
          const float probability = value >= stats[row * 2] && value != NegInf ? sycl::exp(value - maximum) / stats[row * 2 + 1] : 0;
          const float prefix = sycl::inclusive_scan_over_group(item.get_group(), probability, sycl::plus<float>());
          if (j < vocab) cdf[row * vocab + j] = prefix;
          if (lane == Tile - 1) totals[row * tiles + tile] = prefix;
        });
        NativeQueue(q).parallel_for(sycl::nd_range<1>(n * Tile, Tile), [=](sycl::nd_item<1> item) {
          const int64_t row = item.get_group(0), lane = item.get_local_id(0);
          float carry = 0;
          for (int64_t tile = 0; tile < tiles; tile += Tile) {
            const float value = tile + lane < tiles ? totals[row * tiles + tile + lane] : 0;
            const float prefix = sycl::exclusive_scan_over_group(item.get_group(), value, sycl::plus<float>());
            if (tile + lane < tiles) totals[row * tiles + tile + lane] = carry + prefix;
            carry += sycl::reduce_over_group(item.get_group(), value, sycl::plus<float>());
          }
        });
      }
      NativeQueue(q).parallel_for(sycl::range<1>(n * vocab), [=](sycl::id<1> item) {
        const int64_t row = item[0] / vocab, j = item[0] % vocab, index = row * vocab + order[item];
        const bool topk = data[index] < stats[row * 2];
        const bool topp = ps && j != vocab - 1 && cdf[item] + totals[row * tiles + j / Tile] <= 1 - ps[start + row];
        if (topk || topp) data[index] = NegInf;
      });
    }
  });
  VT_CHECK(available, "XPU top-k/top-p cannot reserve its 16 MiB sampling workspace");
}
void RandomSampleKernel(Queue& q, Tensor& token_ids, const Tensor& probs, const Tensor& seeds) {
  TraceXpuOp(OpId::kRandomSample, q, {&token_ids, &probs, &seeds});
  const auto rows = probs.shape[0], vocab = probs.shape[1];
  if (!rows) return;
  VT_CHECK(vocab > 0, "XPU random sampling requires a nonempty vocabulary");
  const auto* pp = static_cast<const float*>(probs.data);
  const auto* sp = static_cast<const int64_t*>(seeds.data);
  auto* ids = static_cast<int64_t*>(token_ids.data);
  NativeQueue(q).submit([&](sycl::handler& h) {
    sycl::local_accessor<float> scores(SamplingLanes, h);
    sycl::local_accessor<int64_t> indices(SamplingLanes, h);
    h.parallel_for(sycl::nd_range<1>(rows * SamplingLanes, SamplingLanes), [=](sycl::nd_item<1> item) {
      const int64_t row = item.get_group(0), lane = item.get_local_id(0);
      const auto key = sample::SplitMix64(uint64_t(sp[row]) + 0x9E3779B97F4A7C15ULL * uint64_t(row));
      float best = NegInf; int64_t index = sample::kArgSentinel;
      for (int64_t j = lane; j < vocab; j += SamplingLanes) {
        const uint64_t u = (sample::SplitMix64(key + uint64_t(j)) >> 11) + 1;
        // Same integer draws as VT's reference, with native F32 transcendental
        // math on Xe2. log1p preserves the tail near U=1 without requiring FP64.
        const float noise = u > (1ULL << 52) ? -sycl::log1p(-float((1ULL << 53) - u) * 0x1p-53f)
                                            : -sycl::log(float(u) * 0x1p-53f);
        const float score = pp[row * vocab + j] / noise;
        sample::ArgReduce(best, index, score, j);
      }
      scores[lane] = best; indices[lane] = index;
      item.barrier(sycl::access::fence_space::local_space);
      for (int step = SamplingLanes / 2; step; step /= 2) {
        if (lane < step) {
          float v = scores[lane]; int64_t i = indices[lane];
          sample::ArgReduce(v, i, scores[lane + step], indices[lane + step]);
          scores[lane] = v; indices[lane] = i;
        }
        item.barrier(sycl::access::fence_space::local_space);
      }
      if (!lane) ids[row] = indices[0] == sample::kArgSentinel ? 0 : indices[0];
    });
  });
}
void GreedyArgmaxKernel(Queue& q, Tensor& token_ids, const Tensor& logits) {
  TraceXpuOp(OpId::kGreedyArgmax, q, {&token_ids, &logits});
  if (logits.shape[0] == 0) return;
  const auto rows = static_cast<size_t>(logits.shape[0]);
  const auto vocab = logits.shape[1];
  const auto* values = static_cast<const float*>(logits.data);
  auto* ids = static_cast<int64_t*>(token_ids.data);
  constexpr size_t lanes = 128;
  const auto event = NativeQueue(q).submit([&](sycl::handler& h) {
    sycl::local_accessor<float> best_values(lanes, h);
    sycl::local_accessor<int64_t> best_ids(lanes, h);
    h.parallel_for(sycl::nd_range<1>(rows * lanes, lanes), [=](sycl::nd_item<1> item) {
      const auto row = item.get_group(0), lane = item.get_local_id(0);
      // Match VT's strict '>' scan: NaN at index 0 wins; later NaNs are ignored.
      if (sycl::isnan(values[row * vocab])) { if (lane == 0) ids[row] = 0; return; }
      float value = -std::numeric_limits<float>::infinity();
      int64_t index = std::numeric_limits<int64_t>::max();
      for (int64_t i = lane; i < vocab; i += lanes) {
        const float v = values[row * vocab + i];
        if (v > value || (v == value && i < index)) { value = v; index = i; }
      }
      best_values[lane] = value; best_ids[lane] = index;
      item.barrier(sycl::access::fence_space::local_space);
      for (size_t step = lanes / 2; step; step /= 2) {
        if (lane < step) {
          const float v = best_values[lane + step]; const auto i = best_ids[lane + step];
          if (v > best_values[lane] || (v == best_values[lane] && i < best_ids[lane])) {
            best_values[lane] = v; best_ids[lane] = i;
          }
        }
        item.barrier(sycl::access::fence_space::local_space);
      }
      if (lane == 0) ids[row] = best_ids[0];
    });
  });
  RecordProfileEvent(q, "greedy_argmax", event);
}
void MappedGreedyArgmaxKernel(Queue& q, Tensor& token_ids, const Tensor& logits,
                              const Tensor& global_ids, int64_t target_vocab) {
  TraceXpuOp(OpId::kMappedGreedyArgmax, q, {&token_ids, &logits, &global_ids});
  const auto rows = static_cast<size_t>(logits.shape[0]);
  if (rows == 0) return;
  const int64_t columns = logits.shape[1];
  const auto* values = static_cast<const float*>(logits.data);
  const auto* mapping = static_cast<const int32_t*>(global_ids.data);
  auto* output = static_cast<int32_t*>(token_ids.data);
  constexpr size_t lanes = 128;
  const auto event = NativeQueue(q).submit([&](sycl::handler& h) {
    sycl::local_accessor<float> best_values(lanes, h);
    sycl::local_accessor<int32_t> best_ids(lanes, h);
    sycl::local_accessor<int> invalid(lanes, h);
    h.parallel_for(sycl::nd_range<1>(rows * lanes, lanes), [=](sycl::nd_item<1> item) {
      const auto row = item.get_group(0), lane = item.get_local_id(0);
      float value = -std::numeric_limits<float>::infinity();
      int32_t id = std::numeric_limits<int32_t>::max();
      int bad = 0;
      for (int64_t column = lane; column < columns; column += lanes) {
        const float candidate = values[row * columns + column];
        const int32_t global = mapping[column];
        bad |= !sycl::isfinite(candidate) || global < 0 || global >= target_vocab;
        if (candidate > value || (candidate == value && global < id)) {
          value = candidate; id = global;
        }
      }
      best_values[lane] = value; best_ids[lane] = id; invalid[lane] = bad;
      item.barrier(sycl::access::fence_space::local_space);
      for (size_t step = lanes / 2; step; step /= 2) {
        if (lane < step) {
          const float candidate = best_values[lane + step];
          const int32_t global = best_ids[lane + step];
          if (candidate > best_values[lane] ||
              (candidate == best_values[lane] && global < best_ids[lane])) {
            best_values[lane] = candidate; best_ids[lane] = global;
          }
          invalid[lane] |= invalid[lane + step];
        }
        item.barrier(sycl::access::fence_space::local_space);
      }
      if (lane == 0) output[row] = invalid[0] ? -1 : best_ids[0];
    });
  });
  RecordProfileEvent(q, "mapped_greedy_argmax", event);
}

void GreedyRejectionSampleKernel(Queue& q, Tensor& sampled,
                                 Tensor& num_sampled, Tensor& target_argmax,
                                 const Tensor& logits,
                                 const Tensor& draft_sampled,
                                 const Tensor& cu_num_logits) {
  TraceXpuOp(OpId::kGreedyRejectionSample, q,
             {&sampled, &num_sampled, &target_argmax, &logits,
              &draft_sampled, &cu_num_logits});
  const int64_t rows = logits.shape[0], vocab = logits.shape[1];
  const int64_t requests = cu_num_logits.shape[0] - 1;
  const int64_t width = sampled.shape[1];
  const auto* offsets = static_cast<const int32_t*>(cu_num_logits.data);
  const auto* draft = static_cast<const int32_t*>(draft_sampled.data);
  const auto* values = static_cast<const float*>(logits.data);
  auto* argmax = static_cast<int32_t*>(target_argmax.data);
  auto* tokens = static_cast<int32_t*>(sampled.data);
  auto* counts = static_cast<int32_t*>(num_sampled.data);
  CheckDeviceMetadata(q, [=] {
    if (offsets[0] != 0 || offsets[requests] != rows) return false;
    for (int64_t r = 0; r < requests; ++r)
      if (offsets[r + 1] <= offsets[r] ||
          offsets[r + 1] - offsets[r] > width) return false;
    return true;
  }, "XPU greedy rejection invalid logits offsets or output width",
      {&cu_num_logits});

  // Target decisions for all expanded rows. The lowest token id wins ties,
  // including an all -inf row; a NaN in column zero follows the CPU oracle.
  constexpr size_t lanes = 128;
  const auto argmax_event = NativeQueue(q).submit([&](sycl::handler& h) {
    sycl::local_accessor<float> best_values(lanes, h);
    sycl::local_accessor<int32_t> best_ids(lanes, h);
    h.parallel_for(sycl::nd_range<1>(rows * lanes, lanes),
                   [=](sycl::nd_item<1> item) {
      const int64_t row = item.get_group(0), lane = item.get_local_id(0);
      if (sycl::isnan(values[row * vocab])) {
        if (lane == 0) argmax[row] = 0;
        return;
      }
      float best = NegInf;
      int32_t index = std::numeric_limits<int32_t>::max();
      for (int64_t col = lane; col < vocab; col += lanes) {
        const float value = values[row * vocab + col];
        if (value > best || (value == best && col < index)) {
          best = value;
          index = static_cast<int32_t>(col);
        }
      }
      best_values[lane] = best;
      best_ids[lane] = index;
      item.barrier(sycl::access::fence_space::local_space);
      for (int64_t step = lanes / 2; step; step /= 2) {
        if (lane < step) {
          const float value = best_values[lane + step];
          const int32_t id = best_ids[lane + step];
          if (value > best_values[lane] ||
              (value == best_values[lane] && id < best_ids[lane])) {
            best_values[lane] = value;
            best_ids[lane] = id;
          }
        }
        item.barrier(sycl::access::fence_space::local_space);
      }
      if (lane == 0) argmax[row] = best_ids[0];
    });
  });
  RecordProfileEvent(q, "greedy_rejection_argmax", argmax_event);

  // One independent sequential acceptance walk per request. The caller owns
  // argmax scratch until this queued kernel completes.
  const auto accept_event = NativeQueue(q).parallel_for(
      sycl::range<1>(requests), [=](sycl::id<1> item) {
    const int64_t req = item[0], begin = offsets[req];
    const int64_t drafts = offsets[req + 1] - begin - 1;
    const int64_t base = req * width;
    for (int64_t j = 0; j < width; ++j) tokens[base + j] = -1;
    int64_t accepted = 0;
    for (int64_t j = 0; j < drafts; ++j) {
      const int32_t target = argmax[begin + j];
      const int32_t proposal = draft[begin + j + 1];
      if (target != proposal) {
        tokens[base + j] = target;
        break;
      }
      tokens[base + j] = proposal;
      ++accepted;
    }
    if (accepted == drafts) tokens[base + accepted] = argmax[begin + accepted];
    counts[req] = static_cast<int32_t>(accepted + 1);
  });
  RecordProfileEvent(q, "greedy_rejection_accept", accept_event);
}

void SampleOneHotRejection(Queue& q, Tensor& sampled, Tensor& num_sampled,
                           Tensor& choices, Tensor& accepted,
                           const Tensor& probs, const Tensor& proposal,
                           const Tensor& cu_num_logits, const Tensor& seeds,
                           const Tensor& greedy) {
  const int64_t rows = probs.shape[0], vocab = probs.shape[1];
  const int64_t requests = cu_num_logits.shape[0] - 1;
  const int64_t width = sampled.shape[1];
  VT_CHECK(probs.rank == 2 && probs.dtype == DType::kF32 && probs.IsContiguous() &&
               probs.device == q.device && vocab > 0,
           "XPU sampled rejection requires contiguous f32 probabilities");
  const auto vector_ok = [&](const Tensor& t, DType dtype) {
    return t.rank == 1 && t.shape[0] == rows && t.dtype == dtype &&
           t.IsContiguous() && t.device == q.device;
  };
  VT_CHECK(vector_ok(proposal, DType::kI32) &&
               vector_ok(seeds, DType::kI64) &&
               vector_ok(greedy, DType::kI8) &&
               vector_ok(choices, DType::kI32) &&
               vector_ok(accepted, DType::kI32),
           "XPU sampled rejection requires one proposal, seed and scratch pair per row");
  VT_CHECK(cu_num_logits.rank == 1 && requests >= 0 &&
               cu_num_logits.dtype == DType::kI32 && cu_num_logits.IsContiguous() &&
               cu_num_logits.device == q.device &&
               sampled.rank == 2 && sampled.shape[0] == requests &&
               sampled.dtype == DType::kI32 && sampled.IsContiguous() &&
               sampled.device == q.device &&
               num_sampled.rank == 1 && num_sampled.shape[0] == requests &&
               num_sampled.dtype == DType::kI32 && num_sampled.IsContiguous() &&
               num_sampled.device == q.device,
           "XPU sampled rejection offsets and outputs have incompatible shapes");
  if (!requests) return;
  const auto* offsets = static_cast<const int32_t*>(cu_num_logits.data);
  const auto* proposed = static_cast<const int32_t*>(proposal.data);
  const auto* keys = static_cast<const int64_t*>(seeds.data);
  const auto* greedy_rows = static_cast<const int8_t*>(greedy.data);
  const auto* distribution = static_cast<const float*>(probs.data);
  auto* chosen = static_cast<int32_t*>(choices.data);
  auto* accept = static_cast<int32_t*>(accepted.data);
  auto* tokens = static_cast<int32_t*>(sampled.data);
  auto* counts = static_cast<int32_t*>(num_sampled.data);
  CheckDeviceMetadata(q, [=] {
    if (offsets[0] != 0 || offsets[requests] != rows) return false;
    for (int64_t r = 0; r < requests; ++r) {
      if (offsets[r + 1] <= offsets[r] || offsets[r + 1] - offsets[r] > width)
        return false;
      for (int64_t row = offsets[r]; row < offsets[r + 1]; ++row) {
        const int32_t id = proposed[row];
        if (id < -1 || id >= vocab || (row == offsets[r + 1] - 1) != (id == -1))
          return false;
      }
    }
    return true;
  }, "XPU sampled rejection invalid offsets or proposal", {&cu_num_logits, &proposal});

  // Each verification row independently draws its corrected token. The
  // sequential per-request accept walk below uses only the prefix through the
  // first rejection; generating unused later-row draws does not change their
  // distributions because each row owns a separate seed.
  constexpr size_t lanes = 128;
  const auto row_event = NativeQueue(q).submit([&](sycl::handler& h) {
    sycl::local_accessor<float> best_values(lanes, h);
    sycl::local_accessor<int32_t> best_ids(lanes, h);
    h.parallel_for(sycl::nd_range<1>(rows * lanes, lanes),
                   [=](sycl::nd_item<1> item) {
      const int64_t row = item.get_group(0), lane = item.get_local_id(0);
      const int32_t draft_id = proposed[row];
      const bool deterministic = greedy_rows[row] != 0;
      const uint64_t key = static_cast<uint64_t>(keys[row]);
      if (lane == 0 && !deterministic) {
        const uint64_t bits = sample::SplitMix64(key ^ 0xA0761D6478BD642FULL);
        const double u = static_cast<double>(bits >> 11) * 0x1p-53;
        accept[row] = draft_id >= 0 &&
                      u < static_cast<double>(distribution[row * vocab + draft_id]);
      }
      float best = NegInf;
      int32_t best_id = std::numeric_limits<int32_t>::max();
      const uint64_t draw_key = sample::SplitMix64(key ^ 0xE7037ED1A0B428DBULL);
      for (int64_t col = lane; col < vocab; col += lanes) {
        if (!deterministic && col == draft_id) continue;
        const float p = distribution[row * vocab + col];
        if (!(p > 0)) continue;
        float score = p;
        if (!deterministic) {
          const uint64_t bits = (sample::SplitMix64(draw_key + uint64_t(col)) >> 11) + 1;
          const float noise = bits > (1ULL << 52)
              ? -sycl::log1p(-float((1ULL << 53) - bits) * 0x1p-53f)
              : -sycl::log(float(bits) * 0x1p-53f);
          score = p / noise;
        }
        if (score > best || (score == best && col < best_id)) {
          best = score;
          best_id = static_cast<int32_t>(col);
        }
      }
      best_values[lane] = best;
      best_ids[lane] = best_id;
      item.barrier(sycl::access::fence_space::local_space);
      for (int64_t step = lanes / 2; step; step /= 2) {
        if (lane < step &&
            (best_values[lane + step] > best_values[lane] ||
             (best_values[lane + step] == best_values[lane] &&
              best_ids[lane + step] < best_ids[lane]))) {
          best_values[lane] = best_values[lane + step];
          best_ids[lane] = best_ids[lane + step];
        }
        item.barrier(sycl::access::fence_space::local_space);
      }
      if (lane == 0) {
        chosen[row] = best_ids[0];
        if (deterministic) accept[row] = draft_id >= 0 && draft_id == best_ids[0];
      }
    });
  });
  RecordProfileEvent(q, "sampled_rejection_rows", row_event);

  const auto walk_event = NativeQueue(q).parallel_for(
      sycl::range<1>(requests), [=](sycl::id<1> item) {
    const int64_t req = item[0], begin = offsets[req];
    const int64_t drafts = offsets[req + 1] - begin - 1;
    const int64_t base = req * width;
    for (int64_t i = 0; i < width; ++i) tokens[base + i] = -1;
    int64_t n_accepted = 0;
    for (int64_t i = 0; i < drafts; ++i) {
      if (!accept[begin + i]) {
        tokens[base + i] = chosen[begin + i];
        break;
      }
      tokens[base + i] = proposed[begin + i];
      ++n_accepted;
    }
    if (n_accepted == drafts) tokens[base + n_accepted] = chosen[begin + drafts];
    counts[req] = static_cast<int32_t>(n_accepted + 1);
  });
  RecordProfileEvent(q, "sampled_rejection_walk", walk_event);
}
}  // namespace vt::xpu
