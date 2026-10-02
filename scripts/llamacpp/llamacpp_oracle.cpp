// The llama.cpp oracle side of a GGUF token gate: free-running greedy ids and
// the teacher-forced near-tie gap, in the golden-directory layout the paged
// engine gates already read (tests/parity/goldens/<name>/):
//
//   p<i>_prompt.i32        the prompt ids, INPUT (little-endian int32)
//   greedy_ids.npy         [N, T] int32, OUTPUT: llama.cpp's greedy decode
//   our_ids.i32            [N*T] int32, INPUT (optional): our engine's ids
//   our_ids.npy            [N, T] int32, OUTPUT when our_ids.i32 is present
//   neartie_gap_mnats.npy  [N, T] int32, OUTPUT when our_ids.i32 is present:
//                          teacher-forced on OUR prefix, the gap at (i, j) is
//                          max(0, logp(oracle argmax) - logp(our id)) at the
//                          position that predicts our token j, in mnats. The
//                          semantics of scripts/qwen3-neartie-gap-llamacpp-oracle.py,
//                          computed from the oracle's own f32 logits.
//
// Built against a llama.cpp checkout at the pin `.agents/oracles/llama-cpp.md`
// names (tag b10451), CPU only:
//   g++ -O2 -std=c++17 -I$LLAMA/include -I$LLAMA/ggml/include
//       scripts/llamacpp/llamacpp_oracle.cpp -L$LLAMA/build/bin
//       -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o llamacpp_oracle
// Usage: llamacpp_oracle <model.gguf> <golden_dir> <N> <T> [n_threads]
//
// No sampling, no chat template, no BOS insertion: the prompt ids are fed
// exactly as the golden stores them, which is what makes the comparison one of
// the forward and not of two tokenizers.
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::vector<int32_t> ReadI32(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  f.seekg(0, std::ios::end);
  const std::streamoff n = f.tellg();
  f.seekg(0);
  std::vector<int32_t> v(static_cast<size_t>(n) / 4);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
  return v;
}

void WriteNpyI32(const std::string& path, const std::vector<int32_t>& v, int n, int t) {
  std::string hdr = "{'descr': '<i4', 'fortran_order': False, 'shape': (" + std::to_string(n) +
                    ", " + std::to_string(t) + "), }";
  const size_t base = 10 + hdr.size() + 1;
  hdr.append((64 - base % 64) % 64, ' ');
  hdr.push_back('\n');
  std::ofstream f(path, std::ios::binary);
  const uint16_t hl = static_cast<uint16_t>(hdr.size());
  f.write("\x93NUMPY\x01\x00", 8);
  f.write(reinterpret_cast<const char*>(&hl), 2);
  f.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
  f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
}

// logit of the argmax minus the logit of `id` (>= 0), and the argmax, in f64.
void Gap(const float* lg, int nv, int32_t id, int32_t* argmax, double* gap_nats) {
  int best = 0;
  for (int k = 1; k < nv; ++k)
    if (lg[k] > lg[best]) best = k;
  *argmax = best;
  *gap_nats = std::max(0.0, static_cast<double>(lg[best]) - static_cast<double>(lg[id]));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s <model.gguf> <golden_dir> <N> <T> [n_threads]\n", argv[0]);
    return 2;
  }
  const std::string model_path = argv[1], dir = argv[2];
  const int N = std::atoi(argv[3]), T = std::atoi(argv[4]);
  const int threads = argc > 5 ? std::atoi(argv[5]) : 16;

  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = 0;
  llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
  if (model == nullptr) return 3;
  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = 1024;
  cp.n_batch = 1024;
  cp.n_ubatch = 512;
  cp.n_threads = threads;
  cp.n_threads_batch = threads;
  cp.no_perf = true;
  llama_context* ctx = llama_init_from_model(model, cp);
  if (ctx == nullptr) return 4;
  const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));

  std::vector<int32_t> greedy(static_cast<size_t>(N * T), -1);
  const std::vector<int32_t> ours = ReadI32(dir + "/our_ids.i32");
  const bool force = ours.size() == static_cast<size_t>(N * T);
  std::vector<int32_t> gap_mnats(static_cast<size_t>(N * T), 0);

  for (int i = 0; i < N; ++i) {
    const std::vector<int32_t> prompt = ReadI32(dir + "/p" + std::to_string(i) + "_prompt.i32");
    if (prompt.empty()) return 5;
    // Free-running greedy.
    llama_memory_clear(llama_get_memory(ctx), true);
    std::vector<llama_token> p(prompt.begin(), prompt.end());
    llama_batch b = llama_batch_get_one(p.data(), static_cast<int32_t>(p.size()));
    if (llama_decode(ctx, b) != 0) return 6;
    for (int j = 0; j < T; ++j) {
      const float* lg = llama_get_logits_ith(ctx, -1);
      int32_t a = 0;
      double g = 0;
      Gap(lg, nv, 0, &a, &g);
      greedy[static_cast<size_t>(i * T + j)] = a;
      llama_token tok = a;
      b = llama_batch_get_one(&tok, 1);
      if (llama_decode(ctx, b) != 0) return 7;
    }
    std::fprintf(stderr, "prompt %d greedy done\n", i);
    if (!force) continue;
    // Teacher-forced on OUR prefix: one pass over prompt + our ids, logits
    // at every position from the last prompt token on.
    llama_memory_clear(llama_get_memory(ctx), true);
    const int P = static_cast<int>(prompt.size());
    llama_batch bb = llama_batch_init(P + T, 0, 1);
    for (int k = 0; k < P + T; ++k) {
      bb.token[k] = k < P ? prompt[static_cast<size_t>(k)] : ours[static_cast<size_t>(i * T + k - P)];
      bb.pos[k] = k;
      bb.n_seq_id[k] = 1;
      bb.seq_id[k][0] = 0;
      bb.logits[k] = k >= P - 1 ? 1 : 0;
    }
    bb.n_tokens = P + T;
    if (llama_decode(ctx, bb) != 0) return 8;
    for (int j = 0; j < T; ++j) {
      // A difference of log-probabilities is a difference of logits: the
      // log-normalizer cancels, so no softmax is formed.
      const float* lg = llama_get_logits_ith(ctx, P - 1 + j);
      int32_t a = 0;
      double g = 0;
      Gap(lg, nv, ours[static_cast<size_t>(i * T + j)], &a, &g);
      gap_mnats[static_cast<size_t>(i * T + j)] = static_cast<int32_t>(std::lround(g * 1000.0));
    }
    llama_batch_free(bb);
    std::fprintf(stderr, "prompt %d teacher-forced done\n", i);
  }
  WriteNpyI32(dir + "/greedy_ids.npy", greedy, N, T);
  if (force) {
    WriteNpyI32(dir + "/our_ids.npy", ours, N, T);
    WriteNpyI32(dir + "/neartie_gap_mnats.npy", gap_mnats, N, T);
  }
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
