// Why the MiniMax-Music3 vocoder's parallel decomposition does not scale
// (#672, #1334) -- the instrument, not the fix.
//
// WHAT THIS ANSWERS. `.agents/specs/minimax-music3.md` §18.8b measured the
// vocoder window at 2.16x per core against 1.37x on 14 threads and attributed
// the difference to "a shared resource ... memory bandwidth remains the leading
// candidate and not a finding". That is an INFERENCE. This probe separates the
// candidates by ablation, because `ncu` is refused on this fleet
// (`ERR_NVGPUCTRPERM`, .agents/environment.md) and no bandwidth counter is
// readable from a lease.
//
// THE FOUR CANDIDATES AND THE ABLATION THAT SEPARATES EACH ONE.
//
//   1. MEMORY HIERARCHY. `Conv1dKernel` partitions OUTPUT rows only, so every
//      thread sweeps the whole input tensor for its own slice of output
//      channels. `--control` runs the SAME geometry at the SAME total
//      multiply-accumulate count with the time axis cut into pieces small
//      enough that one piece's activations sit in cache. Identical arithmetic,
//      identical instruction mix, different residency. If the curve separates,
//      the limit is residency; if it does not, residency is not the limit.
//
//   2. DISPATCH AND BARRIER COST. `--dispatch` prices one empty
//      `vt::cpu::ParallelForRows` at each thread count. The decode window makes
//      62 of them (31 convolutions x 2 streams), so this converts the
//      hypothesis into arithmetic rather than leaving it as a worry. NOTE the
//      control arm in (1) makes R TIMES MORE dispatches than the production
//      arm, so it is biased AGAINST itself on this axis: a control that still
//      scales better has refuted this candidate twice.
//
//   3. WORK-PARTITION GRANULARITY. Every row prints the chunk grid
//      `ParallelForRows` will build (4x oversubscription, ggml-cpu/ops.cpp:9078)
//      next to the thread count, so a geometry whose rows cannot fill the grid
//      is visible rather than suspected. `conv_out` has ONE output row and is
//      the extreme case: it runs inline on the caller at every thread count.
//
//   4. THE THREADS ARE NOT RUNNING. Every leg reports `getrusage` user CPU
//      seconds over the wall clock of the same window. A pool that is idle
//      reads near 1.0; a pool that is busy but unproductive reads near the
//      thread count while the speedup does not follow, which separates "not
//      dispatched" from "dispatched and stalled".
//
// The fifth candidate -- the SM/CPU clock falling as cores light up -- is not
// measurable from inside this process and is sampled by the job script from
// `/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq` around each leg.
//
// WHAT IT IS NOT. It calls `vt::Conv1d` at the vocoder's own geometries with
// synthetic weights, so it is the OP and not the window: spec §18.4 measures a
// factor of ~2 between the two quantities and §18.9 records a 1.80x instrument
// disagreement between a kernel bench and an e2e bucket. A number from here is
// never multiplied onto a `vocoder.decode_window` figure. The window's own
// scaling is measured with `vllm_music3_vocoder_conv_ab`, which drives
// `VocoderDecode` itself.
//
// CI NEVER RUNS THIS. It is an executable and no test, like its sibling
// `tools/bench/music3_vocoder_conv_ab.cpp`; the build compiles it so the one
// file a reader has to compile to reproduce the measurement cannot rot.
#include "vllm/support/platform_compat.h"  // NOMINMAX before windows.h; GetProcessTimes on Windows
#if !defined(_WIN32)
#include <sys/resource.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vt/cpu/cpu_threadpool.h"
#include "vt/ops.h"

namespace {

// The same LCG `tools/bench/music3_vocoder_conv_ab.cpp` uses, for the same
// reason: identical on every box and every arm, and never exactly 0.0.
struct Lcg {
  uint32_t s;
  float Next() {
    s = s * 1664525U + 1013904223U;
    const double mantissa = static_cast<double>(s >> 8) / 16777216.0 - 0.5;
    const int exponent = static_cast<int>((s >> 4) & 0x7U) - 6;
    double v = mantissa * std::ldexp(1.0, exponent);
    if (v == 0.0) v = 1.0 / 4096.0;
    return static_cast<float>(v);
  }
};

std::vector<float> Fill(Lcg& rng, size_t n) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = rng.Next();
  return v;
}

double UserSeconds() {
#if defined(_WIN32)
  FILETIME created{}, exited{}, kernel{}, user{};
  GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user);
  // FILETIME counts 100 ns ticks. user + kernel matches RUSAGE_SELF's sum below.
  const auto seconds = [](const FILETIME& t) {
    ULARGE_INTEGER ticks{};
    ticks.LowPart = t.dwLowDateTime;
    ticks.HighPart = t.dwHighDateTime;
    return static_cast<double>(ticks.QuadPart) * 1e-7;
  };
  return seconds(user) + seconds(kernel);
#else
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  return static_cast<double>(ru.ru_utime.tv_sec) + 1e-6 * static_cast<double>(ru.ru_utime.tv_usec) +
         static_cast<double>(ru.ru_stime.tv_sec) + 1e-6 * static_cast<double>(ru.ru_stime.tv_usec);
#endif
}

double Now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// A `vt::Conv1d` shape the MiniMax-Music3 vocoder actually runs, at
// `stride == 1`, `groups == 1`, `padding == 0` -- which is how the vocoder calls
// it, because `vocoder1d::Pad1d` materialises the pad before the call
// (minimax_music3_acoustic.cpp:705-711). `out_len` output positions therefore
// need `out_len + kernel - 1` input positions.
struct Geom {
  const char* name;
  int64_t in_ch;
  int64_t out_ch;
  int64_t kernel;
  int64_t out_len;
};

// FNV-1a over the raw output bytes: two arms that differ by a scheduling
// decision must print the same value, and a sum would let them disagree
// silently.
uint64_t Fingerprint(const float* p, size_t n) {
  uint64_t h = 1469598103934665603ULL;
  const unsigned char* b = reinterpret_cast<const unsigned char*>(p);
  for (size_t i = 0; i < n * sizeof(float); ++i) {
    h ^= static_cast<uint64_t>(b[i]);
    h *= 1099511628211ULL;
  }
  return h;
}

struct Leg {
  double wall = 0.0;
  double user = 0.0;
  uint64_t fingerprint = 0;
};

// One timed call of `vt::Conv1d` over `slices` independent pieces of the time
// axis. `slices == 1` is the production shape. `slices == R` is the residency
// control: R calls of `out_len / R` positions each, the SAME weights, the SAME
// total multiply-accumulate count, and R times as many pool dispatches.
Leg RunConv(const Geom& g, int64_t slices, int repeats) {
  const int64_t out_len = g.out_len / slices * slices;  // exact partition only
  const int64_t piece = out_len / slices;
  const int64_t in_len = piece + g.kernel - 1;
  Lcg rng{0xC0FFEEu + static_cast<uint32_t>(g.in_ch * 31 + g.kernel)};
  // One input buffer per piece so the control does not read one hot buffer R
  // times -- that would measure a residency the production arm never gets.
  std::vector<std::vector<float>> xs;
  std::vector<std::vector<float>> outs;
  xs.reserve(static_cast<size_t>(slices));
  outs.reserve(static_cast<size_t>(slices));
  for (int64_t s = 0; s < slices; ++s) {
    xs.push_back(Fill(rng, static_cast<size_t>(g.in_ch * in_len)));
    // Allocated OUTSIDE the timed region. An allocation per piece would charge
    // the control R mallocs the production arm does not pay, and the control is
    // the arm this probe is trying not to flatter.
    outs.emplace_back(static_cast<size_t>(g.out_ch * piece));
  }
  const std::vector<float> w = Fill(rng, static_cast<size_t>(g.out_ch * g.in_ch * g.kernel));
  const std::vector<float> b = Fill(rng, static_cast<size_t>(g.out_ch));

  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const vt::Device dev{vt::DeviceType::kCPU, 0};
  vt::Tensor wt = vt::Tensor::Contiguous(const_cast<float*>(w.data()), vt::DType::kF32, dev,
                                         {g.out_ch, g.in_ch, g.kernel});
  vt::Tensor bt = vt::Tensor::Contiguous(const_cast<float*>(b.data()), vt::DType::kF32, dev,
                                         {g.out_ch});
  vt::Conv1dArgs args;

  Leg best;
  best.wall = 1e30;
  for (int r = 0; r < repeats; ++r) {
    const double u0 = UserSeconds();
    const double t0 = Now();
    for (int64_t s = 0; s < slices; ++s) {
      vt::Tensor xt = vt::Tensor::Contiguous(const_cast<float*>(xs[static_cast<size_t>(s)].data()),
                                             vt::DType::kF32, dev, {1, g.in_ch, in_len});
      vt::Tensor ot = vt::Tensor::Contiguous(outs[static_cast<size_t>(s)].data(), vt::DType::kF32,
                                             dev, {1, g.out_ch, piece});
      vt::Conv1d(q, ot, xt, wt, &bt, args);
    }
    const double dt = Now() - t0;
    if (dt < best.wall) {
      best.wall = dt;
      best.user = UserSeconds() - u0;
    }
  }
  // Folded AFTER the timed region. Each piece writes its own contiguous block
  // of every output row, so the concatenation is NOT the production arm's
  // buffer layout and the two arms' fingerprints are not compared to each
  // other. What a fingerprint pins is ONE arm against itself across thread
  // counts, which is the determinism claim this file makes.
  uint64_t h = 1469598103934665603ULL;
  for (int64_t s = 0; s < slices; ++s) {
    const std::vector<float>& o = outs[static_cast<size_t>(s)];
    const uint64_t part = Fingerprint(o.data(), o.size());
    h ^= part;
    h *= 1099511628211ULL;
  }
  best.fingerprint = h;
  return best;
}

// The chunk grid `ParallelForRows` builds for `nr` rows on `nth` threads --
// transcribed from cpu_threadpool.cpp:413-458 rather than guessed, so the row
// counts that cannot fill the grid are visible in the output.
int64_t ChunkCount(int64_t nr, int nth) {
  if (nr == 1 || nth == 1) return 1;
  const int nth_scaled = nth * 4;
  const int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
  int64_t nchunk = (nr + chunk_size - 1) / chunk_size;
  if (nchunk < nth) nchunk = nth;
  return nchunk;
}

void DispatchCost(int repeats) {
  vt::cpu::Threadpool& tp = vt::cpu::CurrentThreadpool();
  const int64_t nr = 1024;  // large enough that the grid is real, small enough
                            // that the body costs nothing
  volatile int64_t sink = 0;
  const auto body = [&sink](int64_t r0, int64_t r1) { sink = sink + r1 - r0; };
  // Warm the pool: the first dispatch after a park pays a condition-variable
  // wake that no later one does, and pricing that as the steady-state cost
  // would overstate the candidate this probe exists to refute or confirm.
  for (int i = 0; i < 100; ++i) vt::cpu::ParallelForRows(tp, nr, body);
  double best = 1e30;
  for (int r = 0; r < 3; ++r) {
    const double t0 = Now();
    for (int i = 0; i < repeats; ++i) vt::cpu::ParallelForRows(tp, nr, body);
    const double dt = (Now() - t0) / static_cast<double>(repeats);
    if (dt < best) best = dt;
  }
  std::printf("dispatch  threads=%d  per_dispatch_us=%.3f  window_62_dispatches_ms=%.4f\n",
              tp.NThreads(), best * 1e6, best * 62.0 * 1e3);
}

}  // namespace

int main(int argc, char** argv) {
  int64_t latents = 344;
  int repeats = 3;
  bool control = true;
  bool dispatch = true;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a.rfind("--latents=", 0) == 0) {
      latents = std::strtoll(a.substr(10).c_str(), nullptr, 10);
    } else if (a.rfind("--repeats=", 0) == 0) {
      repeats = static_cast<int>(std::strtol(a.substr(10).c_str(), nullptr, 10));
    } else if (a == "--no-control") {
      control = false;
    } else if (a == "--no-dispatch") {
      dispatch = false;
    } else {
      std::fprintf(stderr, "usage: %s [--latents=N] [--repeats=N] [--no-control] [--no-dispatch]\n",
                   argv[0]);
      return 2;
    }
  }

  const int nth = vt::cpu::CurrentThreadpool().NThreads();
  // ASSERTED, not assumed: `VLLM_CPP_CPU_THREADS` is read once when the lazy
  // global pool is built, so a leg that failed to set it would otherwise report
  // a curve point at the wrong x.
  std::printf("# vt::Conv1d decomposition probe (#1334) -- threads=%d latents=%lld repeats=%d\n",
              nth, static_cast<long long>(latents), repeats);
  const char* env = std::getenv("VLLM_CPP_CPU_THREADS");
  std::printf("# VLLM_CPP_CPU_THREADS=%s  pool.NThreads()=%d\n", env != nullptr ? env : "(unset)",
              nth);

  // The MiniMax-Music3 vocoder's `vt::Conv1d` calls, one per distinct shape, at
  // `MiniMaxMusic3VocoderConfig`'s shipped geometry (latent_channels 128 ->
  // stream 64, decoder_input_dim 1024, decoder_hidden_dim 1536, upsampling
  // ratios 8/8/4/2 -- minimax_music3_loader.h:253-265). Residual units repeat
  // three times per block and are timed once.
  const std::vector<Geom> geoms = {
      {"dec_in_proj  k1", 64, 1024, 1, latents},
      {"conv_in      k7", 1024, 1536, 7, latents},
      {"b0_res_conv1 k7", 768, 768, 7, latents * 8},
      {"b0_res_conv2 k1", 768, 768, 1, latents * 8},
      {"b1_res_conv1 k7", 384, 384, 7, latents * 64},
      {"b1_res_conv2 k1", 384, 384, 1, latents * 64},
      {"b2_res_conv1 k7", 192, 192, 7, latents * 256},
      {"b2_res_conv2 k1", 192, 192, 1, latents * 256},
      {"b3_res_conv1 k7", 96, 96, 7, latents * 512},
      {"b3_res_conv2 k1", 96, 96, 1, latents * 512},
      {"conv_out     k7", 96, 1, 7, latents * 512},
  };

  std::printf("%-16s %8s %8s %6s %10s %8s %7s %12s %10s %9s %20s\n", "geom", "in_ch", "out_ch",
              "k", "out_len", "rows", "chunks", "wall_s", "GMAC/s", "user/wall", "fingerprint");
  double total = 0.0;
  for (const Geom& g : geoms) {
    const Leg leg = RunConv(g, 1, repeats);
    const double macs = static_cast<double>(g.out_ch) * static_cast<double>(g.in_ch) *
                        static_cast<double>(g.kernel) * static_cast<double>(g.out_len);
    total += leg.wall;
    std::printf("%-16s %8lld %8lld %6lld %10lld %8lld %7lld %12.5f %10.3f %9.2f %#20llx\n", g.name,
                static_cast<long long>(g.in_ch), static_cast<long long>(g.out_ch),
                static_cast<long long>(g.kernel), static_cast<long long>(g.out_len),
                static_cast<long long>(g.out_ch), static_cast<long long>(ChunkCount(g.out_ch, nth)),
                leg.wall, macs / leg.wall / 1e9, leg.user / leg.wall,
                static_cast<unsigned long long>(leg.fingerprint));
    std::fflush(stdout);
  }
  std::printf("%-16s %8s %8s %6s %10s %8s %7s %12.5f\n", "TOTAL(one of each)", "", "", "", "", "",
              "", total);

  if (control) {
    // RESIDENCY SWEEP -- the ablation that separates "the limit is the memory
    // hierarchy" from "the limit is something else", without a counter.
    //
    // Each row runs the SAME geometry with the time axis cut into pieces of a
    // declared length, all output channels of a piece before the next piece.
    // The arithmetic, the instruction mix, the code path and the kernel's own
    // position tile are identical across the row -- only the ACTIVATION
    // FOOTPRINT that one piece of work touches changes. A flat row means
    // residency is not the limit at this geometry. A row that rises as the
    // footprint falls locates the limit AT a cache level, and the knee names
    // which one.
    //
    // Rates, not walls. A piece length that does not divide `out_len` leaves a
    // remainder, so the arms do different amounts of work; GMAC/s is the
    // quantity that is comparable regardless, and every cell prints the
    // multiply-accumulate count it was computed from.
    //
    // The sweep is also biased AGAINST the small footprints, which is why a win
    // there is worth something: 128 positions per piece is 172x more pool
    // dispatches than one whole-length call at the b3 geometry, and the
    // dispatch cost that buys is measured at the bottom of this output.
    std::printf("\n# residency sweep: same geometry, all output channels of one time piece before the next\n");
    std::printf("%-16s %10s %8s %12s %12s %12s %9s\n", "geom", "piece_len", "pieces", "act_KiB",
                "wall_s", "GMAC/s", "vs_whole");
    // The whole length FIRST: every later row is reported as a ratio against it,
    // so the baseline has to exist before they are printed.
    const int64_t pieces[] = {0, 128, 512, 2048, 8192};  // 0 = the whole length
    for (const Geom& g : geoms) {
      double whole_rate = 0.0;
      for (const int64_t want : pieces) {
        int64_t piece = want == 0 ? g.out_len : want;
        if (piece > g.out_len) continue;
        const int64_t r = g.out_len / piece;
        if (r < 1) continue;
        Geom sub = g;
        sub.out_len = r * piece;
        const Leg leg = RunConv(sub, r, repeats);
        const double macs = static_cast<double>(g.out_ch) * static_cast<double>(g.in_ch) *
                            static_cast<double>(g.kernel) * static_cast<double>(sub.out_len);
        const double rate = macs / leg.wall / 1e9;
        if (want == 0) whole_rate = rate;
        const double act_kib =
            static_cast<double>(g.in_ch * (piece + g.kernel - 1) * 4) / 1024.0;
        std::printf("%-16s %10lld %8lld %12.1f %12.5f %12.3f %9.3f\n", g.name,
                    static_cast<long long>(piece), static_cast<long long>(r), act_kib, leg.wall,
                    rate, whole_rate > 0.0 ? rate / whole_rate : 0.0);
        std::fflush(stdout);
      }
    }
  }

  if (dispatch) {
    std::printf("\n");
    DispatchCost(20000);
  }
  return 0;
}
