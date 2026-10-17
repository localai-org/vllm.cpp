// Kolibri-1 Tenstorrent decode throughput bench (production shape)
// (MODEL-TEXT-kolibri-1-tenstorrent; spec .agents/specs/kolibri-tt.md).
//
// This is the TT-device mirror of tests/vllm/models/
// test_kolibri1_decode_bench.cpp (the CPU production-shape gate): load once,
// ONE 128-token prefill, then 64 free-running greedy decode steps through the
// production registry seam (ModelRegistry::Forward, pure_decode), timed as
// wall / prefill / decode with the full greedy chain printed. NO per-op
// dumps, NO per-step CPU comparison, NO teacher forcing, NO token anchor —
// the numerics adjudication is a separate question; this TU measures.
//
// Skip convention (the b2bi device-leg gate): loud MESSAGE + return when no
// Tenstorrent device, when VT_KOLIBRI1_TT_B2II_MODEL is unset, or when the
// named directory is not a Kolibri-1 checkpoint. Operator-run under the GPU
// lease; evidence lands in docs/bench-evidence/.
//
// Sampling: the per-step minimum the greedy loop needs is the TOKEN ID, not
// the logits row — the argmax runs on-device (vt::GreedyArgmax, the
// on-device sampling seam) and only the 8-byte id crosses back. Set
// VT_KOLIBRI1_TT_DECODE_BENCH_LOGITS_COPY=1 to also copy the full f32
// logits row per step (diagnostic only; it is NOT the default measurement).
#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/kolibri1_shared.h"
#include "vllm/model_executor/models/kolibri1_tt.h"
#include "vllm/model_executor/models/kolibri1_tt_forward.h"
#include "vllm/model_executor/models/kolibri1_weights.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

using namespace vllm;  // NOLINT

namespace {

double NowSec() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

v1::CommonAttentionMetadata OneReqMeta(int64_t t, int64_t ctx_before,
                                       int64_t block_table_num_cols) {
  v1::CommonAttentionMetadata m;
  m.num_reqs = 1;
  m.num_actual_tokens = static_cast<int>(t);
  m.max_query_len = static_cast<int>(t);
  m.max_seq_len = static_cast<int>(ctx_before + t);
  m.query_start_loc = {0, static_cast<int32_t>(t)};
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens = {static_cast<int32_t>(ctx_before + t)};
  m.seq_lens_cpu = m.seq_lens;
  m.num_computed_tokens_cpu = {static_cast<int32_t>(ctx_before)};
  m.block_table_tensor.assign(static_cast<size_t>(block_table_num_cols), 0);
  for (int64_t i = 0; i < block_table_num_cols; ++i)
    m.block_table_tensor[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  m.block_table_num_cols = static_cast<int>(block_table_num_cols);
  for (int64_t i = 0; i < t; ++i) m.slot_mapping.push_back(ctx_before + i);
  m.causal = true;
  return m;
}

// Prefill shape: the engine CANNOT prefill a whole 128-token prompt in one
// forward — one whole-prompt routed dispatch streams every selected expert's
// slot bytes in a single charge and trips the B1 per-step stream bound
// (observed here: 1404123840 B against the 1179936000 B bound at layer 3;
// the b2ii gate records the same refusal, 1242865920 B at layer 15).
//
// CHUNKED PREFILL WAS TRIED (2026-10-11) AND IS NOT ADOPTED AS THE
// DEFAULT: with the per-token stream envelope generalized to multi-token
// forwards (kolibri1_tt_forward.cpp scales the guard by the token count)
// and an 8-token chunk walk, prefill dropped 156.3 s -> 72.1 s with the
// decode split unchanged. But the greedy chain could NOT be proven
// equivalent: this instrument is NOT bit-reproducible run to run — three
// runs of the SAME one-token walk produced three different chains (first
// token 346, 30686, and 10 under chunking), and the b2ii golden gate's
// HARD flips move between runs too (it is a band gate, 141/145 in the
// 2.5-nat band, not a bit gate). The chunked walk additionally runs the
// prefill-path kernels (the one-token walk forces pure_decode=true,
// decode-path kernels), a further drift source. Adopting chunked prefill
// therefore owes its own adjudicated equivalence evidence (a band-class
// TF comparison, not a chain identity check). The faithful one-token
// incremental walk over the persistent KV cache stays
// (test_kolibri1_tt_b2ii.cpp device gate, which cites
// test_kolibri1_w3.cpp:298-310), so the prefill walks the prompt one
// token per forward and the walk is what the prefill timer covers.
constexpr int kPromptLen = 128;
constexpr int kDecodeSteps = 64;

}  // namespace

TEST_CASE("kolibri1 TT decode bench: 128-token prefill, 64 greedy steps, "
          "wall/prefill/decode split") {
  if (vt::TryGetBackend(vt::DeviceType::kTENSTORRENT) == nullptr) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  const char* model_dir = std::getenv("VT_KOLIBRI1_TT_B2II_MODEL");
  if (model_dir == nullptr || *model_dir == '\0') {
    MESSAGE("SKIPPED: VT_KOLIBRI1_TT_B2II_MODEL not set (no real checkpoint "
            "requested)");
    return;
  }
  const std::string dir = model_dir;
  if (!std::filesystem::exists(dir + "/model.safetensors.index.json")) {
    MESSAGE("SKIPPED: " << dir << " is not a Kolibri-1 checkpoint");
    return;
  }

  // ---- 1. Load the REAL checkpoint through the production registry path.
  const HfConfig config = vllm::LoadHfConfig(dir + "/config.json");
  const ModelRegistration& reg = ModelRegistry::Resolve(config);
  REQUIRE(reg.architecture == "Kolibri1ForCausalLM");
  reg.factory->parse_config(config);
  const auto index = nlohmann::json::parse(
      std::ifstream(dir + "/model.safetensors.index.json"));
  std::set<std::string> shard_names;
  for (const auto& [name, shard] : index.at("weight_map").items()) {
    (void)name;
    shard_names.insert(shard.get<std::string>());
  }
  std::vector<SafetensorsFile> shards;
  for (const std::string& shard : shard_names)
    shards.push_back(SafetensorsFile::Open(dir + "/" + shard));
  const ModelSource source = ModelSource::FromSafetensors(shards);
  const double t_load0 = NowSec();
  std::unique_ptr<LoadedModel> model = ModelRegistry::Load(config, source);
  REQUIRE(model != nullptr);
  const double load_s = NowSec() - t_load0;
  MESSAGE("load " << load_s << " s");

  // ---- 2. Prepare on the TT queue (the production residency the decode
  // runs) and build the device KV topology.
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  vt::Queue q = be.CreateQueue();
  const double t_prep0 = NowSec();
  ModelRegistry::Prepare(*model, config, q);
  const Kolibri1TTResidentDeviceContext& ctx =
      Kolibri1LoadedModelTTContext(*model, q);
  REQUIRE(ctx.built);
  MESSAGE("prepare " << NowSec() - t_prep0 << " s, "
                     << ctx.projections << " projections, "
                     << double(ctx.uploaded_bytes) / (1 << 30) << " GiB bf16");

  const Kolibri1Weights& w = Kolibri1LoadedModelWeights(*model);
  const Kolibri1Params& p = w.params;
  const int64_t hkv = p.num_key_value_heads;
  const int64_t hd = p.head_dim;
  const int64_t block_size = 16;
  // 128-in + 64-out = 192 tokens -> 12 blocks; carry 32 for headroom.
  const int64_t num_blocks = 32;
  const int64_t block_table_cols = num_blocks / 2;
  const int64_t kv_bytes = num_blocks * 2 * block_size * hkv * hd *
                           static_cast<int64_t>(vt::SizeOf(vt::DType::kBF16));
  std::vector<std::shared_ptr<void>> kv_keep;
  std::vector<PagedKvCache> kv;
  kv.reserve(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    void* buf = be.Alloc(kv_bytes);
    kv_keep.emplace_back(buf, [&be](void* ptr) { be.Free(ptr); });
    std::memset(buf, 0, static_cast<size_t>(kv_bytes));
    PagedKvCache c;
    c.data = buf;
    c.dtype = vt::DType::kBF16;
    c.num_blocks = num_blocks;
    c.block_size = block_size;
    c.num_kv_heads = hkv;
    c.head_size = hd;
    kv.push_back(c);
  }

  v1::GDNAttentionMetadata gdn_meta;
  std::vector<GdnStateCache> gdn_state;
  // Deterministic prompt, same recipe as the CPU bench: (i*7919+13)%128000.
  std::vector<int32_t> seq;
  seq.reserve(kPromptLen + kDecodeSteps);
  for (int i = 0; i < kPromptLen; ++i)
    seq.push_back(static_cast<int32_t>((i * 7919 + 13) % 128000));

  // Device-side greedy sampler scratch: the ids tensor lives on device, only
  // the 8-byte token id crosses back per step.
  int64_t* ids_host = static_cast<int64_t*>(be.AllocPinned(sizeof(int64_t)));
  void* ids_dev = be.Alloc(sizeof(int64_t));
  std::shared_ptr<void> ids_keep(ids_host, [&be](void* ptr) {
    be.FreePinned(static_cast<int64_t*>(ptr));
  });
  std::shared_ptr<void> ids_dev_keep(ids_dev, [&be](void* ptr) {
    be.Free(ptr);
  });
  const bool copy_logits =
      std::getenv("VT_KOLIBRI1_TT_DECODE_BENCH_LOGITS_COPY") != nullptr;

  auto GreedyNext = [&](const ForwardLogits& fl) -> int32_t {
    vt::Tensor ids =
        vt::Tensor::Contiguous(ids_dev, vt::DType::kI64, q.device, {1});
    vt::GreedyArgmax(q, ids, fl.device_tensor);
    be.Copy(q, ids_host, ids_dev, sizeof(int64_t));
    be.Synchronize(q);
    return static_cast<int32_t>(ids_host[0]);
  };

  auto Forward = [&](const std::vector<int32_t>& tokens, int64_t ctx_before,
                     bool pure_decode) -> ForwardLogits {
    const int64_t t = static_cast<int64_t>(tokens.size());
    // The b2ii walk convention: a one-token forward IS a decode step,
    // including the prompt walk (same math over the persistent KV cache).
    pure_decode = (t == 1);
    const v1::CommonAttentionMetadata meta =
        OneReqMeta(t, ctx_before, block_table_cols);
    std::vector<int32_t> positions(static_cast<size_t>(t));
    std::iota(positions.begin(), positions.end(),
              static_cast<int32_t>(ctx_before));
    ModelForwardInput in{tokens,
                         positions,
                         meta,
                         gdn_meta,
                         kv,
                         gdn_state,
                         config,
                         q,
                         /*logits_indices=*/{},
                         /*num_reqs=*/1};
    in.pure_decode = pure_decode;
    in.uniform_query_len = pure_decode ? 1 : 0;
    return ModelRegistry::Forward(*model, in);
  };

  // ---- 3. PREFILL: walk the 128 prompt tokens one forward per step over
  // the persistent KV cache (see the prefill-shape note above), timed; the
  // last walk step yields the first greedy token.
  // ITEM-3 opt-in (VT_KOLIBRI1_TT_DECODE_BENCH_PREFILL_CHUNK=N, default 1 =
  // the committed serial walk): drive the 128-token prefill in N-token
  // forwards over the same persistent KV cache. The last forward's row is
  // the first-token row either way.
  int64_t prefill_chunk = 1;
  if (const char* cs =
          std::getenv("VT_KOLIBRI1_TT_DECODE_BENCH_PREFILL_CHUNK"))
    prefill_chunk = std::max<int64_t>(1, std::atoi(cs));
  const double t_pre0 = NowSec();
  ForwardLogits fl;
  for (int i = 0; i < kPromptLen;) {
    const int take = static_cast<int>(std::min<int64_t>(
        prefill_chunk, kPromptLen - static_cast<int64_t>(i)));
    std::vector<int32_t> chunk(seq.begin() + i, seq.begin() + i + take);
    fl = Forward(chunk, /*ctx_before=*/i, /*pure_decode=*/false);
    REQUIRE(fl.rows == 1);
    i += take;
  }
  int32_t next = GreedyNext(fl);
  seq.push_back(next);
  const double prefill_s = NowSec() - t_pre0;
  MESSAGE("prefill walk " << kPromptLen << " tokens in " << prefill_s
                          << " s; first token " << next);

  // ---- 4. DECODE: 64 free-running greedy steps, timed from before the
  // first step to after the last sync.
  std::vector<int32_t> chain{next};
  const double t_dec0 = NowSec();
  for (int step = 0; step < kDecodeSteps; ++step) {
    const int64_t ctx_before = static_cast<int64_t>(seq.size()) - 1;
    fl = Forward({seq.back()}, ctx_before, /*pure_decode=*/true);
    REQUIRE(fl.rows == 1);
    if (copy_logits) {
      // Diagnostic arm only: bring the full f32 row back and argmax on host.
      std::vector<float> logits(static_cast<size_t>(fl.vocab));
      be.Copy(q, logits.data(), fl.device_tensor.data,
              static_cast<size_t>(fl.vocab) * sizeof(float));
      be.Synchronize(q);
      next = static_cast<int32_t>(std::max_element(logits.begin(),
                                                   logits.end()) -
                                  logits.begin());
    } else {
      next = GreedyNext(fl);
    }
    seq.push_back(next);
    chain.push_back(next);
  }
  const double decode_s = NowSec() - t_dec0;
  const double wall_s = prefill_s + decode_s;
  REQUIRE(chain.size() == static_cast<size_t>(kDecodeSteps) + 1);

  std::string chain_str;
  for (int32_t t : chain) chain_str += std::to_string(t) + ",";
  MESSAGE("CHAIN: " << chain_str);
  MESSAGE("wall " << wall_s << " s, prefill " << prefill_s << " s, decode "
                  << decode_s << " s, tok/s "
                  << (static_cast<double>(kDecodeSteps) / decode_s));
  MESSAGE("load " << load_s << " s");
}
