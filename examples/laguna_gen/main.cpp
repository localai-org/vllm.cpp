// laguna-gen — a minimal greedy generation driver for the single-GB10
// Laguna-S-2.1 vehicle. Two checkpoint formats, auto-detected from --model:
//
//   * a .gguf FILE  → keep-quant GGUF (unsloth UD-Q4_K_XL, 3 shards). Experts stay
//     Q4_K/Q5_K blocks, borrowed in place by the mmap residency (no ~69 GiB copy).
//   * a DIRECTORY   → NVFP4 safetensors (poolside/Laguna-S-2.1-NVFP4, W4A4). bf16
//     attn/dense/router/shared-expert + per-expert fp4 routed experts. This is the
//     apples-to-apple-vs-vLLM arm (N3): same tensor-core regime as vLLM's NVFP4.
//
// Both formats resolve to `LagunaWeights` and run the SAME greedy loop:
// `LagunaForwardGguf`/`...Cached` (the forward branches on the resident quant).
// `LagunaForwardGguf` is a STATELESS full-sequence recompute; the KV-cache
// variant (default) attends each new token over cached K/V. --gpu routes the
// block-quant / fp4 GEMMs to the GB10 off the unified-memory blocks.
//
//   laguna-gen --model <shard-1.gguf | nvfp4-dir> [--prompt "..."]
//              [--token-ids 1,2,3] [--max-tokens N] [--load-only] [--gpu]
//              [--stateless]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/laguna.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"  // vt::GetBackend / CreateQueue (--gpu: GEMMs on the GB10)
#include "vt/device.h"

namespace fs = std::filesystem;

namespace {

double CurResidentGiB() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("VmRSS:", 0) == 0) {
      long kb = 0;
      std::sscanf(line.c_str() + 6, "%ld", &kb);
      return static_cast<double>(kb) / (1024.0 * 1024.0);
    }
  return 0.0;
}
double PeakResidentGiB() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("VmHWM:", 0) == 0) {
      long kb = 0;
      std::sscanf(line.c_str() + 6, "%ld", &kb);
      return static_cast<double>(kb) / (1024.0 * 1024.0);
    }
  return 0.0;
}

int64_t KvInt(const vllm::GgufFile& g, const char* key, int64_t dflt) {
  const vllm::GgufValue* v = g.FindKv(key);
  if (v == nullptr) return dflt;
  switch (v->TypeId()) {
    case vllm::kGgufU8: return std::get<uint8_t>(v->v);
    case vllm::kGgufI8: return std::get<int8_t>(v->v);
    case vllm::kGgufU16: return std::get<uint16_t>(v->v);   // split.count is U16
    case vllm::kGgufI16: return std::get<int16_t>(v->v);
    case vllm::kGgufU32: return std::get<uint32_t>(v->v);
    case vllm::kGgufI32: return std::get<int32_t>(v->v);
    case vllm::kGgufU64: return static_cast<int64_t>(std::get<uint64_t>(v->v));
    case vllm::kGgufI64: return std::get<int64_t>(v->v);
    case vllm::kGgufBool: return std::get<bool>(v->v) ? 1 : 0;
    default: return dflt;
  }
}

// eos/eot from the HF config.json `raw` doc (scalar or first element of a list);
// falls back to `dflt` when the key is absent or non-numeric.
int64_t HfInt(const nlohmann::json& raw, const char* key, int64_t dflt) {
  auto it = raw.find(key);
  if (it == raw.end()) return dflt;
  if (it->is_number_integer()) return it->get<int64_t>();
  if (it->is_array() && !it->empty() && it->front().is_number_integer())
    return it->front().get<int64_t>();
  return dflt;
}

int ArgmaxLastRow(const std::vector<float>& logits, int64_t vocab) {
  const int64_t rows = static_cast<int64_t>(logits.size()) / vocab;
  const float* row = logits.data() + (rows - 1) * vocab;
  int best = 0;
  float bv = row[0];
  for (int64_t i = 1; i < vocab; ++i)
    if (row[i] > bv) { bv = row[i]; best = static_cast<int>(i); }
  return best;
}

std::vector<int32_t> ParseIds(const std::string& s) {
  std::vector<int32_t> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
  return out;
}

// Derive the split-sibling shard paths from the shard-1 path + a split count.
// The llama.cpp split naming is "<stem>-NNNNN-of-MMMMM.gguf"; replace the 5-digit
// shard index before "-of-". Returns {path} for a single-file (non-split) model.
std::vector<std::string> ShardPaths(const std::string& shard1, int64_t count) {
  if (count <= 1) return {shard1};
  const std::string tag = "-of-";
  const size_t of = shard1.rfind(tag);
  std::vector<std::string> out;
  if (of == std::string::npos || of < 5) { out.push_back(shard1); return out; }
  const std::string prefix = shard1.substr(0, of - 5);  // up to the NNNNN index
  const std::string suffix = shard1.substr(of);         // "-of-MMMMM.gguf"
  for (int64_t s = 1; s <= count; ++s) {
    char idx[24];
    std::snprintf(idx, sizeof(idx), "%05d", static_cast<int>(s));
    out.push_back(prefix + idx + suffix);
  }
  return out;
}

// Enumerate a dir's *.safetensors shards, sorted (mirror model_loader.cpp LoadShards).
std::vector<vllm::SafetensorsFile> OpenSafetensorsDir(const std::string& dir) {
  std::vector<std::string> paths;
  for (const auto& e : fs::directory_iterator(dir))
    if (e.is_regular_file() && e.path().extension() == ".safetensors")
      paths.push_back(e.path().string());
  if (paths.empty()) throw std::runtime_error("no *.safetensors shards in " + dir);
  std::sort(paths.begin(), paths.end());
  std::vector<vllm::SafetensorsFile> shards;
  shards.reserve(paths.size());
  for (const std::string& p : paths) shards.push_back(vllm::SafetensorsFile::Open(p));
  return shards;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, prompt = "The capital of France is", token_ids_arg;
  int max_tokens = 24;
  bool load_only = false, use_gpu = false, stateless = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--model") model = next();
    else if (a == "--prompt") prompt = next();
    else if (a == "--token-ids") token_ids_arg = next();
    else if (a == "--max-tokens") max_tokens = std::atoi(next());
    else if (a == "--load-only") load_only = true;
    else if (a == "--gpu") use_gpu = true;
    else if (a == "--stateless") stateless = true;  // W5 O(n^2) full-recompute (A/B gate)
    else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  if (model.empty()) {
    std::fprintf(stderr, "usage: --model <shard-1.gguf | nvfp4-dir> [--prompt ...] "
                         "[--token-ids ...] [--max-tokens N] [--load-only] [--gpu]\n");
    return 2;
  }

  // Auto-detect the checkpoint format: a directory = NVFP4 safetensors, a file = GGUF.
  const bool is_nvfp4 = fs::is_directory(model);

  vllm::LagunaWeights w;
  int64_t eos = 2, eot = 2, bos_id = 2;
  std::optional<vllm::tok::Tokenizer> tkz;   // built once; used for encode + decode
  bool add_bos = false;

  // Create the CUDA context BEFORE loading weights. On the GB10's 119 GiB UNIFIED
  // pool, reading the ~67 GiB checkpoint leaves ~67 GiB of reclaimable page cache on
  // top of the ~67 GiB of owned copies; a CUDA context reserved AFTER the load fails
  // (cudaStreamCreate OOM) because the driver can't grab its reservation against that
  // pressure. Reserving the context while the pool is empty avoids it — the forward's
  // later device allocations then grow into the headroom (the kernel evicts the clean
  // page cache under pressure).
  vt::Backend* gpu_backend = nullptr;
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  if (use_gpu) {
    gpu_backend = &vt::GetBackend(vt::DeviceType::kCUDA);
    q = gpu_backend->CreateQueue();
    std::fprintf(stderr, "[gen] GPU context reserved on CUDA device %d (before load)\n",
                 q.device.index);
  } else {
    std::fprintf(stderr, "[gen] CPU queue\n");
  }

  if (is_nvfp4) {
    // ── NVFP4 safetensors arm (N3) ──────────────────────────────────────────
    // N5 lever #1: enable the native sm120a fp4 tensor-core MMA for the routed-expert
    // GEMMs. It reads the SAME linear scale layout LqGemmNvfp4Fp4 produces, so no
    // swizzle is needed; it just defaults OFF. Measured on GB10: decode 0.39 → 0.20
    // s/tok (2×; 2.56 → 5.0 tok/s), coherent + near-tie. `setenv(...,0)` respects an
    // explicit `VT_NVFP4_FP4_NATIVE=0` override. Scoped to this Laguna driver (the
    // 27B/35B use the separate DirectD cutlass path, untouched).
#if defined(_WIN32)
    // setenv(..., 0) semantics: set only when unset, so an explicit override wins.
    if (std::getenv("VT_NVFP4_FP4_NATIVE") == nullptr) {
      _putenv_s("VT_NVFP4_FP4_NATIVE", "1");
    }
#else
    setenv("VT_NVFP4_FP4_NATIVE", "1", 0);
#endif
    const std::string config_path = (fs::path(model) / "config.json").string();
    const std::string tok_path = (fs::path(model) / "tokenizer.json").string();
    std::fprintf(stderr, "[gen] NVFP4 safetensors dir %s\n", model.c_str());
    const vllm::HfConfig config = vllm::LoadHfConfig(config_path);
    std::vector<vllm::SafetensorsFile> shards = OpenSafetensorsDir(model);
    std::fprintf(stderr, "[gen] %zu safetensors shard(s); loading NVFP4 W4A4 tower "
                 "(RSS before %.1f GiB)...\n", shards.size(), CurResidentGiB());
    const auto t0 = std::chrono::steady_clock::now();
    w = vllm::LoadLagunaForCausalLMWeights(shards, config);
    const auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr,
        "[gen] LOADED: layers=%lld experts=%lld vocab=%lld has_nvfp4=%d | load %.1fs | "
        "RSS %.1f GiB PEAK %.1f GiB\n",
        (long long)w.params.num_hidden_layers, (long long)w.params.num_experts,
        (long long)w.params.vocab_size, (int)w.has_nvfp4_weights,
        std::chrono::duration<double>(t1 - t0).count(), CurResidentGiB(), PeakResidentGiB());
    if (!w.has_nvfp4_weights) { std::fprintf(stderr, "[gen] ERROR: no NVFP4 tower\n"); return 1; }
    // VT_LAGUNA_SHARED_FP4 (default OFF): keep the per-layer shared expert fp4-resident
    // (XS-NVFP4 tower) instead of the bf16 dequant, and free the now-dead bf16 shared +
    // fused router-shared copies. MUST run BEFORE the shards are released (it copies the
    // on-disk fp4 bytes out). No-op unless the flag is set and the tower quantizes the
    // shared expert. ADDITIVE — does not touch the SACRED laguna_weights.cpp loader.
    vllm::LagunaLoadSharedExpertFp4(shards, w);
    // The loader COPIES every tensor into owned buffers (LoadBf16Direct / the fp4
    // LnLoadCtNvfp4Raw both memcpy), so the mmap'd shards are dead weight now.
    // On the GB10's 119 GiB UNIFIED pool holding the mmap alongside the ~67 GiB of
    // owned copies pushes RSS to ~114 GiB and starves the CUDA context
    // (cudaStreamCreate OOM). Drop the shards -> RSS falls to the owned-copy size.
    const size_t n_shards_freed = shards.size();
    shards.clear();
    shards.shrink_to_fit();
    std::fprintf(stderr, "[gen] released %zu mmap'd shard(s); RSS %.1f GiB\n",
                 n_shards_freed, CurResidentGiB());
    // N5 campaign-B: build the routed-expert MARLIN residents NOW (at load), not
    // lazily on the first forward — so the 48L×256E repack is a one-time load cost,
    // not a first-token TTFT spike (mirrors vLLM process_weights_after_loading).
    // No-op unless VT_LAGUNA_MARLIN_MOE=1 + GPU (+ built with VT_MARLIN_NVFP4).
    if (use_gpu) {
      const auto tmb0 = std::chrono::steady_clock::now();
      vllm::LagunaBuildMarlinResidents(q, w);
      const auto tmb1 = std::chrono::steady_clock::now();
      const double mb_s = std::chrono::duration<double>(tmb1 - tmb0).count();
      if (mb_s > 0.5)
        std::fprintf(stderr, "[gen] MARLIN residents built at load in %.1fs (RSS %.1f GiB)\n",
                     mb_s, CurResidentGiB());
    }
    eos = HfInt(config.raw, "eos_token_id", 2);
    eot = eos;
    if (fs::exists(tok_path)) {
      try { tkz.emplace(vllm::tok::Tokenizer::FromHfJson(tok_path)); }
      catch (const std::exception& e) {
        std::fprintf(stderr, "[gen] WARN: tokenizer.json load failed (%s); "
                     "text decode disabled, use --token-ids\n", e.what());
      }
    }
  } else {
    // ── keep-quant GGUF arm (W5/W6) ─────────────────────────────────────────
    std::fprintf(stderr, "[gen] opening %s\n", model.c_str());
    vllm::GgufFile meta = vllm::GgufFile::Open(model);
    const int64_t split_count = KvInt(meta, "split.count", 1);
    const std::vector<std::string> paths = ShardPaths(model, split_count);
    std::vector<vllm::GgufFile> shards;
    shards.reserve(paths.size());
    shards.push_back(std::move(meta));  // shard-1 stays index 0 (has the KV)
    for (size_t s = 1; s < paths.size(); ++s) {
      std::fprintf(stderr, "[gen] opening shard %zu: %s\n", s + 1, paths[s].c_str());
      shards.push_back(vllm::GgufFile::Open(paths[s]));
    }
    std::vector<const vllm::GgufFile*> shard_ptrs;
    for (const vllm::GgufFile& s : shards) shard_ptrs.push_back(&s);

    eos = KvInt(shards[0], "tokenizer.ggml.eos_token_id", 2);
    eot = KvInt(shards[0], "tokenizer.ggml.eot_token_id", 24);
    add_bos = KvInt(shards[0], "tokenizer.ggml.add_bos_token", 0) != 0;
    bos_id = KvInt(shards[0], "tokenizer.ggml.bos_token_id", 2);

    // KEEP-QUANT load: routed experts stay Q4_K/Q5_K blocks; mmap borrows in place.
    vllm::GgufLoadPolicy pol;
    pol.keep_quant = true;
    pol.mmap_residency = true;
    std::fprintf(stderr, "[gen] loading keep-quant tower (RSS before %.1f GiB)...\n",
                 CurResidentGiB());
    const auto t0 = std::chrono::steady_clock::now();
    w = vllm::LoadLagunaFromGgufShards(shard_ptrs, &pol);
    const auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr,
        "[gen] LOADED: layers=%lld experts=%lld vocab=%lld has_gguf=%d | load %.1fs | "
        "RSS %.1f GiB PEAK %.1f GiB\n",
        (long long)w.params.num_hidden_layers, (long long)w.params.num_experts,
        (long long)w.params.vocab_size, (int)w.has_gguf_weights,
        std::chrono::duration<double>(t1 - t0).count(), CurResidentGiB(), PeakResidentGiB());
    if (!w.has_gguf_weights) { std::fprintf(stderr, "[gen] ERROR: no keep-quant tower\n"); return 1; }
    try { tkz.emplace(vllm::tok::Tokenizer::FromGguf(shards[0])); }
    catch (const std::exception& e) {
      std::fprintf(stderr, "[gen] WARN: gguf tokenizer load failed (%s)\n", e.what());
    }
  }

  const int64_t vocab = w.params.vocab_size;
  if (load_only) { std::fprintf(stderr, "[gen] --load-only done.\n"); return 0; }

  // Prompt -> token ids. Prefer injected ids (exact cross-check vs the vLLM golden);
  // else encode with the loaded tokenizer.
  std::vector<int32_t> tokens;
  if (!token_ids_arg.empty()) {
    tokens = ParseIds(token_ids_arg);
    std::fprintf(stderr, "[gen] using %zu injected token ids\n", tokens.size());
  } else if (tkz.has_value()) {
    tokens = tkz->Encode(prompt);
    // GGUF path prepends the GGUF bos when add_bos_token is set (the NVFP4
    // tokenizer.json applies its own bos/eos via the post-processor).
    if (add_bos && !is_nvfp4)
      tokens.insert(tokens.begin(), static_cast<int32_t>(bos_id));
    std::fprintf(stderr, "[gen] encoded prompt (%zu ids):", tokens.size());
    for (int32_t t : tokens) std::fprintf(stderr, " %d", t);
    std::fprintf(stderr, "\n");
  } else {
    std::fprintf(stderr, "[gen] ERROR: no tokenizer and no --token-ids\n");
    return 2;
  }
  if (tokens.empty()) { std::fprintf(stderr, "[gen] ERROR: empty token list\n"); return 2; }
  const size_t n_prompt = tokens.size();

  std::fprintf(stderr, "[gen] %s GEMMs (%s)\n", use_gpu ? "GPU" : "CPU",
               is_nvfp4 ? "NVFP4 W4A4" : "keep-quant");

  std::fprintf(stderr, "[gen] decode path: %s\n",
               stateless ? "STATELESS O(n^2) full-recompute (W5)"
                         : "KV-CACHE incremental (W6)");
  std::vector<int32_t> generated;
  double prefill_s = 0.0, decode_s = 0.0;
  vllm::LagunaKvCache kv;  // W6 path only
  // FRONT 1 (measurement honesty): the per-step trace line calls CurResidentGiB()
  // (a /proc/self/status read) + an unbuffered stderr write EVERY decode step, in the
  // GPU-idle gap between two graph replays — host overhead a production engine never
  // pays. Default OFF (set VT_LAGUNA_STEP_LOG=1 to restore the per-step trace); the
  // final summary below is always printed, so the gates (grep decode_hp / generated ids)
  // are unaffected. `decode_hp` (sum of the s0→s1 step timers) EXCLUDES this gap already;
  // `decode_wall` below is the TRUE end-to-end decode throughput INCLUDING every gap, so
  // the FRONT-1 win is visible there (paired A/B: VT_LAGUNA_STEP_LOG=1 vs unset).
  const bool step_log = std::getenv("VT_LAGUNA_STEP_LOG") != nullptr;
  std::chrono::steady_clock::time_point dec_wall_start{};
  for (int step = 0; step < max_tokens; ++step) {
    if (step == 1) dec_wall_start = std::chrono::steady_clock::now();  // decode phase begins
    std::vector<int32_t> step_tokens, positions;
    std::vector<int32_t> logits_idx;
    if (stateless || step == 0) {
      step_tokens = tokens;  // stateless: whole context; step 0: prefill the prompt
      positions.resize(tokens.size());
      for (size_t i = 0; i < tokens.size(); ++i) positions[i] = static_cast<int32_t>(i);
      logits_idx = {static_cast<int32_t>(tokens.size() - 1)};
    } else {
      step_tokens = {tokens.back()};  // one new token
      positions = {static_cast<int32_t>(kv.len)};
      logits_idx = {0};
    }
    const auto s0 = std::chrono::steady_clock::now();
    const std::vector<float> logits =
        stateless ? vllm::LagunaForwardGguf(w, q, step_tokens, positions, logits_idx)
                  : vllm::LagunaForwardGgufCached(w, q, kv, step_tokens, positions, logits_idx);
    // VT_LAGUNA_ONDEV_SAMPLE: the resident decode graph already argmaxed the logits ON-
    // DEVICE and left the token in kv.last_sampled (>=0), so skip the host argmax over the
    // (empty) logits. -1 on prefill / when the lever is off ⇒ host-argmax the returned row.
    // NEXT-TOKEN SELECTION IS INSIDE THE TIMED STEP: the host argmax over the 100352-vocab
    // is real per-step wall (it sits between two graph replays, GPU idle) — attributing it
    // to the step is what makes the on-device-sample win visible in decode wall tok/s.
    const int next = (!stateless && kv.last_sampled >= 0) ? kv.last_sampled
                                                          : ArgmaxLastRow(logits, vocab);
    const auto s1 = std::chrono::steady_clock::now();
    const double step_s = std::chrono::duration<double>(s1 - s0).count();
    if (step == 0) prefill_s = step_s; else decode_s += step_s;
    if (step_log)
      std::fprintf(stderr, "[gen] step %d (ctx=%zu): next=%d  %.2fs  (RSS %.1f GiB)\n",
                   step, tokens.size(), next, step_s, CurResidentGiB());
    generated.push_back(next);
    tokens.push_back(next);
    if (next == eos || next == eot) { std::fprintf(stderr, "[gen] EOS/EOT\n"); break; }
  }
  const auto dec_wall_end = std::chrono::steady_clock::now();

  std::string text;
  if (tkz.has_value()) {
    try { text = tkz->Decode(generated); }
    catch (const std::exception& e) { text = std::string("<decode failed: ") + e.what() + ">"; }
  } else {
    text = "<no tokenizer; ids only>";
  }

  std::printf("\n===== LAGUNA-S-2.1 GENERATION (%s) =====\n", is_nvfp4 ? "NVFP4" : "GGUF");
  std::printf("prompt: %s\n", prompt.c_str());
  std::printf("generated ids:");
  for (int32_t t : generated) std::printf(" %d", t);
  std::printf("\ngenerated text: %s\n", text.c_str());
  const int n_dec = static_cast<int>(generated.size()) - 1;
  std::printf("prompt_tokens=%zu gen_tokens=%zu\n", n_prompt, generated.size());
  std::printf("prefill: %.2fs  decode: %.2fs  TPOT %.2fs/tok over %d steps\n",
              prefill_s, decode_s, n_dec > 0 ? decode_s / n_dec : 0.0, n_dec);
  std::printf("decode_hp: %.5fs steps=%d tok/s=%.4f\n", decode_s, n_dec,
              (n_dec > 0 && decode_s > 0.0) ? n_dec / decode_s : 0.0);
  // TRUE end-to-end decode wall (step 1's start → last step's end), INCLUDING every
  // per-step host gap — the honest throughput a caller sees. decode_hp above is the
  // gap-free device-busy timer; the (decode_wall < decode_hp) delta is the host tax.
  const double decode_wall_s =
      (n_dec >= 1) ? std::chrono::duration<double>(dec_wall_end - dec_wall_start).count() : 0.0;
  std::printf("decode_wall: %.5fs steps=%d tok/s=%.4f\n", decode_wall_s, n_dec,
              (n_dec > 0 && decode_wall_s > 0.0) ? n_dec / decode_wall_s : 0.0);
  std::printf("PEAK RESIDENT: %.2f GiB\n", PeakResidentGiB());
  std::fflush(stdout);
  if (gpu_backend != nullptr) gpu_backend->DestroyQueue(q);
  return 0;
}
