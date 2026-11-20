// SPEC-DFLASH2 W2 (#1314) — the DFlash2 draft: its config, its grouped dynamic
// convolution weights, and the forward that runs them.
//
// BEYOND-PIN throughout. Upstream is
// `vllm/model_executor/models/qwen3_dflash2.py` @ vllm-project/vllm#52816 head
// `19c9351904df4c63042671bc67a866ca48dc7d6f`; the parity pin `555967922` does
// not carry the architecture and this row does not advance it.
//
// PART 1 — the config builder, which is `## Owed` O3 and O4 of the row's spec.
// `MakeQwen3DFlashDraftConfig` could not parse EITHER published DFlash2
// `config.json` at all: it did `c.at("rope_theta")` and `c.at("block_size")`
// while `z-lab/Qwen3.8-27B-DFlash2` nests them as `rope_parameters.rope_theta`
// and `dflash_config.block_size` and declares neither at the top level, so both
// `at` calls threw before any DFlash2 mechanism could be reached. It also did
// `c.at("layer_types")`, which `XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash` does not
// declare, while upstream reads `getattr(config, "layer_types", None)`
// (`qwen3_dflash.py:134` and `:66` @ that head). The configs embedded below are
// the PUBLISHED files BYTE-FOR-BYTE, each with the sha256 OF THE EMBEDDED
// LITERAL recorded beside it, so the gate does not depend on a checkout being
// present and the recorded hash describes what the compiler actually sees.
//
// The `attention_sink_bias` refusal is the other half of O4 and is not
// bookkeeping. Upstream reads `dflash_config.attention_sink_bias` and passes a
// per-head sink into its `Attention` (`qwen3_dflash.py:309-313` and `:240-257` @
// that head); this lane has no attention sink at all. Landing the `layer_types`
// fallback ALONE would let MiMo's draft parse and load with the sinks silently
// absent -- acceptance-only, token-invisible, which is the exact class this row
// exists to remove. So the key is refused BY NAME.
//
// PART 2 — the grouped convolution inside the draft forward. The op itself is
// gated in tests/vt/test_ops_dflash2_grouped_conv.cpp against upstream's own
// sequential reference; what is gated here is that the DRAFT MODEL runs it, on
// weights the PRODUCTION loader read off a real on-disk checkpoint, and that
// deleting the production call site turns this suite red.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <utility>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_dflash.h"
// SPEC-DFLASH2 W8 (#1837): the capture-capable CPU backend + static-graph
// platform swap, so the DFlash2 propose forward's graph admission is testable
// on a box with no GPU. Same harness, same reasons, as the DFlash1 gate in
// test_qwen3_dflash_decode_graph_seam.cpp.
#include "decode_graph_seam_harness.h"
#include "vt/breakable_graph.h"
#include "vllm/v1/worker/gpu/spec_decode/dflash/speculator.h"
#include "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.h"
#include "vllm/model_executor/models/qwen3_dflash2.h"
// SPEC-DFLASH2-QUANT-LMHEAD (#1628): the shared NVFP4 head, and the TARGET's own
// shared-head logits path the selector's top-K is gated against.
#include "vllm/model_executor/models/dense_device_glue.h"
#include "vllm/model_executor/models/dense_nvfp4_gemm.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "../gguf_builder.h"
#include "vllm/transformers_utils/hf_config.h"
// The PRODUCTION loader entry point, for the reachability cases at the end of
// this file (SPEC-DFLASH2-QUANT-LMHEAD, #1628).
#include "vllm/config/speculative.h"
#include "vllm/sampling_params.h"
#include "vllm/entrypoints/model_loader.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"  // #2495 item 6: vt::Exl3DequantLinear, for the decoded twin

// SPEC-DFLASH2 (#1919): the draft context store's capacity is a REQUIRED
// argument now, resolved in production from the engine's own `max_model_len`
// (`Qwen3DFlashModel::ResolveCtxStoreSizing`). These unit stores hold single-
// digit context rows, so they name a small capacity directly; before #1919 each
// of them silently allocated the fixed 4096-slot pool.
constexpr int64_t kUnitCtxSlots = 256;

namespace fs = std::filesystem;
using nlohmann::json;
using vllm::HfConfig;
using vllm::MakeQwen3DFlashDraftConfig;
using vllm::OwnedTensor;
using vllm::Qwen3DFlashLayerAttnMode;
using vllm::Qwen3DFlashModel;
using vllm::Qwen3DFlashWeights;
using vllm::ResolveQwen3DFlashAttnModes;

namespace {

// `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`,
// config.json VERBATIM (1239 bytes, sha256
// 873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980, read on
// 2026-08-19). Kept whole rather than reduced: what this case gates is that the
// PUBLISHED document parses, and a reduced copy would only prove that a document
// this test wrote parses.
//
// The sha256 hashes THE LITERAL BELOW, trailing newline included, so the claim
// is checkable from this file alone rather than only against a copy on a share.
// The three literals in this file were all re-fetched and re-hashed on
// 2026-08-20; one of them (`kMimoDflashConfig`) had been re-indented and had
// lost a key, and it is now the published bytes.
constexpr const char* kQwen38Dflash2Config = R"JSON({
  "architectures": [
    "DFlash2DraftModel"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": null,
  "is_causal": false,
  "dflash_config": {
    "block_size": 8,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "mask_token_id": 248070,
    "selector_rank": 256,
    "selector_top_k": 16,
    "target_layer_ids": [
      5,
      19,
      33,
      47,
      61
    ]
  },
  "dtype": "bfloat16",
  "eos_token_id": 248044,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 5120,
  "initializer_range": 0.02,
  "intermediate_size": 17408,
  "layer_types": [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention"
  ],
  "max_position_embeddings": 262144,
  "max_window_layers": 5,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 5,
  "num_key_value_heads": 8,
  "num_target_layers": 64,
  "pad_token_id": 248044,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {
    "rope_theta": 10000000,
    "rope_type": "default"
  },
  "sliding_window": 2048,
  "tie_word_embeddings": false,
  "transformers_version": "5.15.0",
  "use_cache": true,
  "use_sliding_window": true,
  "vocab_size": 248320
}
)JSON";

// `z-lab/Muse-Glimmer-30B-DFlash2` @ `b54ffdd11fa9cfe2af370012e5763d492c904128`,
// config.json VERBATIM (1326 bytes, sha256
// cb684d6f688a22619a63ea1debe7d30c139c195bf3141fd86a763763ab34b5d9 over the
// literal below, read on 2026-08-19 and re-verified 2026-08-20). The SECOND
// published DFlash2 checkpoint, and the one that makes
// #1327 a correction rather than a note: `block_size` 16 against the 27B's 8,
// and `output_multiplier`/`final_logit_softcapping` SET rather than defaulted.
constexpr const char* kMuseGlimmerDflash2Config = R"JSON({
  "architectures": [
    "DFlash2DraftModel"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 200000,
  "is_causal": false,
  "dflash_config": {
    "block_size": 16,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "final_logit_softcapping": 20.0,
    "mask_token_id": 201818,
    "output_multiplier": 0.19611613513818404,
    "selector_rank": 256,
    "selector_top_k": 16,
    "target_layer_ids": [
      1,
      13,
      25,
      37,
      49
    ]
  },
  "dtype": "bfloat16",
  "eos_token_id": 200001,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 6656,
  "initializer_range": 0.02,
  "intermediate_size": 19968,
  "layer_types": [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention"
  ],
  "max_position_embeddings": 131072,
  "max_window_layers": 5,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 5,
  "num_key_value_heads": 8,
  "num_target_layers": 52,
  "pad_token_id": 200018,
  "rms_norm_eps": 1e-05,
  "rope_parameters": {
    "rope_theta": 500000.0,
    "rope_type": "default"
  },
  "sliding_window": 2048,
  "tie_word_embeddings": false,
  "transformers_version": "5.15.0",
  "use_cache": false,
  "use_sliding_window": true,
  "vocab_size": 202048
}
)JSON";

// `XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash` @
// `b754e6c86008bdb5cc901308dda5a38173ec7276`, `dflash/config.json` VERBATIM
// (1251 bytes, sha256
// 2ed5a998f5f57e00a9fe14d2b3e767f06e49462a97eb09d80c927e112a585c9e over the
// literal below, re-fetched 2026-08-20). A DFlash1 draft, present here for O4:
// it is the ONLY published draft that declares no `layer_types`, and it is also
// the only one that declares `attention_sink_bias`.
//
// This copy was NOT verbatim when W2 landed. It had been re-indented to two
// spaces and had dropped `auto_map`, so the recorded sha256 hashed a file that
// was not in this repository and nothing here hashed what the test parsed. The
// published bytes are in, and the two facts the case turns on -- no
// `layer_types`, `attention_sink_bias` present -- are unchanged by the repair.
constexpr const char* kMimoDflashConfig = R"JSON({
    "architectures": [
        "DFlashDraftModel"
    ],
    "model_type": "qwen3",
    "auto_map": {
        "AutoModel": "dflash.DFlashDraftModel"
    },
    "hidden_size": 6144,
    "intermediate_size": 16384,
    "num_hidden_layers": 5,
    "num_attention_heads": 128,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "v_head_dim": 128,
    "partial_rotary_factor": 0.5,
    "block_size": 8,
    "dflash_config": {
        "target_layer_ids": [
            0,
            15,
            31,
            47,
            69
        ],
        "mask_token_id": 151669,
        "num_anchors": 4096,
        "block_size": 8,
        "loss_decay_gamma": 7.0,
        "use_swa": true,
        "swa_window_size": 1024,
        "backbone_rotary_base": 5000000,
        "attention_value_scale": 0.612,
        "attention_sink_bias": true
    },
    "num_target_layers": 70,
    "vocab_size": 152064,
    "max_position_embeddings": 262144,
    "rope_theta": 10000,
    "sliding_window": 1024,
    "rms_norm_eps": 1e-05,
    "torch_dtype": "bfloat16",
    "hidden_act": "silu",
    "attention_bias": false,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "tie_word_embeddings": false,
    "use_cache": true
})JSON";

}  // namespace

TEST_CASE("dflash2 config: the published Qwen3.8-27B DFlash2 config.json PARSES (O3)") {
  const HfConfig c = MakeQwen3DFlashDraftConfig(json::parse(kQwen38Dflash2Config));
  CHECK(c.hidden_size == 5120);
  CHECK(c.num_hidden_layers == 5);
  CHECK(c.vocab_size == 248320);
  // The nested spellings, which the flat reads could not see. `transformers`
  // moved RoPE settings under `rope_parameters`, so this is a FALLBACK and not a
  // replacement: DFlash1 checkpoints still carry the flat `rope_theta`.
  CHECK(c.rope_theta == doctest::Approx(1e7));
  REQUIRE(c.raw.contains("block_size"));
  CHECK(c.raw.at("block_size").get<int64_t>() == 8);
  // The DFlash2 conv geometry the W2 forward needs, carried through dflash_config.
  REQUIRE(c.raw.contains("dflash_config"));
  CHECK(c.raw.at("dflash_config").at("conv_kernel_size").get<int64_t>() == 2);
  CHECK(c.raw.at("dflash_config").at("conv_group_size").get<int64_t>() == 16);
  // W1's rule, now reachable for the first time on a checkpoint that declares it.
  REQUIRE(c.raw.contains("is_causal"));
  const std::vector<Qwen3DFlashLayerAttnMode> modes = ResolveQwen3DFlashAttnModes(c);
  REQUIRE(modes.size() == 5);
  for (size_t i = 0; i < modes.size(); ++i) {
    CAPTURE(i);
    CHECK_FALSE(modes[i].causal);
    CHECK(modes[i].sliding_window == 2048);
  }
}

TEST_CASE("dflash2 config: the published Muse-Glimmer-30B DFlash2 config.json PARSES (#1327)") {
  const HfConfig c = MakeQwen3DFlashDraftConfig(json::parse(kMuseGlimmerDflash2Config));
  CHECK(c.hidden_size == 6656);
  CHECK(c.rope_theta == doctest::Approx(5e5));
  REQUIRE(c.raw.contains("block_size"));
  CHECK(c.raw.at("block_size").get<int64_t>() == 16);
  // hidden 6656 / conv_group_size 16 = 416 groups, so kernel_projection is
  // 2*taps*num_groups = 1664 wide. The 27B's is 1280. The two shapes are what
  // makes "gate at BOTH published blocks" a real requirement rather than a
  // parameter sweep.
  CHECK(c.raw.at("dflash_config").at("conv_group_size").get<int64_t>() == 16);
  CHECK(c.hidden_size % c.raw.at("dflash_config").at("conv_group_size").get<int64_t>() == 0);
  // The two scalars #1327 corrects the spec about: they are CHECKPOINT-exercised,
  // not synthetic, and both are applied to candidate VALUES before the selector
  // scores them, so a wrong one reorders the top-K and moves acceptance without
  // raising. W3 consumes them; W2 asserts they survive the parse.
  CHECK(c.raw.at("dflash_config").at("output_multiplier").get<double>() ==
        doctest::Approx(0.19611613513818404));
  CHECK(c.raw.at("dflash_config").at("final_logit_softcapping").get<double>() ==
        doctest::Approx(20.0));
}

TEST_CASE("dflash draft config: an ABSENT layer_types is upstream's None (O4)") {
  // `getattr(config, "layer_types", None)` (qwen3_dflash.py:134 and :66 @ the PR
  // head). As shipped this threw `[json.exception.out_of_range.403] key
  // 'layer_types' not found` before any causality could be resolved, which is why
  // #1366's `use_swa` repair was UNREACHED at its own merge commit.
  json doc = json::parse(kMimoDflashConfig);
  doc["dflash_config"].erase("attention_sink_bias");  // refused separately, below
  const HfConfig c = MakeQwen3DFlashDraftConfig(doc);
  CHECK(c.layer_types.empty());
  // Upstream's own `_resolve_layer_attention` docstring row: `layer_types=None`
  // + `use_swa=True` -> SWA window, causal FALSE. Before #1366 this engine
  // answered causal TRUE on every layer.
  const std::vector<Qwen3DFlashLayerAttnMode> modes = ResolveQwen3DFlashAttnModes(c);
  REQUIRE(modes.size() == 5);
  for (size_t i = 0; i < modes.size(); ++i) {
    CAPTURE(i);
    CHECK_FALSE(modes[i].causal);
    CHECK(modes[i].sliding_window == 1024);
  }
}

TEST_CASE("dflash draft config: attention_sink_bias is REFUSED BY NAME (O4)") {
  // Upstream reads it and passes a per-head sink into its Attention; this lane
  // has none. Parsing the config WITHOUT refusing would convert a loud
  // `key 'layer_types' not found` into a draft that loads with the sinks
  // silently absent -- acceptance-only and invisible to a token gate.
  const json doc = json::parse(kMimoDflashConfig);
  CHECK_THROWS_WITH_AS(MakeQwen3DFlashDraftConfig(doc),
                       doctest::Contains("attention_sink_bias"), std::exception);
  // A FALSE value is upstream's default and is not refused: `dflash_config.get(
  // "attention_sink_bias", ...)` falsy means no sink parameter is created at all.
  json off = doc;
  off["dflash_config"]["attention_sink_bias"] = false;
  CHECK_NOTHROW(MakeQwen3DFlashDraftConfig(off));
}

TEST_CASE("dflash draft config: the DFlash1 flat spellings are UNCHANGED") {
  // The fallbacks must not move a published DFlash1 draft. `z-lab/Qwen3.6-27B-DFlash`
  // declares a flat `rope_theta`, a flat `block_size` and a full `layer_types`,
  // and every one of them must still win.
  json doc = json::object();
  doc["hidden_size"] = 5120;
  doc["num_attention_heads"] = 32;
  doc["num_key_value_heads"] = 8;
  doc["head_dim"] = 128;
  doc["rope_theta"] = 1e7;
  doc["intermediate_size"] = 25600;
  doc["vocab_size"] = 248320;
  doc["num_hidden_layers"] = 2;
  doc["rms_norm_eps"] = 1e-6;
  doc["sliding_window"] = 2048;
  doc["layer_types"] = json::array({"sliding_attention", "full_attention"});
  doc["block_size"] = 17;
  doc["dflash_config"] = json::object();
  doc["dflash_config"]["mask_token_id"] = 248070;
  doc["dflash_config"]["target_layer_ids"] = json::array({5, 19});
  // A nested rope_parameters must NOT override a declared flat rope_theta on a
  // document that carries both: upstream reads one resolved `config.rope_theta`.
  const HfConfig c = MakeQwen3DFlashDraftConfig(doc);
  CHECK(c.rope_theta == doctest::Approx(1e7));
  CHECK(c.raw.at("block_size").get<int64_t>() == 17);
  REQUIRE(c.layer_types.size() == 2);
  CHECK(c.layer_types[0] == "sliding_attention");
  CHECK(c.layer_types[1] == "full_attention");
  const std::vector<Qwen3DFlashLayerAttnMode> modes = ResolveQwen3DFlashAttnModes(c);
  REQUIRE(modes.size() == 2);
  CHECK(modes[0].causal);
  CHECK_FALSE(modes[1].causal);
}

// ===========================================================================
// PART 2 — the conv INSIDE the draft, loaded by the production weight loader
// off a real on-disk safetensors checkpoint and run by the production forward.
//
// REACHABILITY (.agents/reachability.md). The chain a user arrives through is
//   LoadedEngine::FromModelDir / ResolveSpecConfig
//     -> LoadDflashDraft            (src/vllm/entrypoints/model_loader.cpp)
//       -> vllm::LoadQwen3DFlash    (reads attention_conv/mlp_conv, sets
//                                    conv_taps/conv_group_size; the loader then
//                                    overwrites conv_block_size with 1 + k)
//     -> GPUModelRunner::propose_drafts_block
//       -> Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV
//         -> the layer body -> DflashConvPrepare / DflashConvFinish
//                           -> vt::DFlashGroupedConv
//       -> vllm::v1::Dflash2SelectCandidates (W3)
//       -> vllm::v1::Dflash2WalkPath -> vt::Dflash2PathWalk (W4) -> the drafts
//
// The cases below enter at `vllm::LoadQwen3DFlash` over a real safetensors file
// -- the same function the loader calls, on the same tensor names the published
// checkpoint uses -- and then at the model forward, which is what
// `propose_drafts_block` calls. The reachability MUTATION deletes the
// `DflashConvPrepare`/`DflashConvFinish` call sites in the layer body and this
// suite must redden.
//
// WHY AN IDENTITY CONV IS THE RIGHT PROBE. With `taps = 1`, `base_kernel` all
// ones and `kernel_projection` all zeros, the conv is EXACTLY the identity in
// bf16: `bf16(1 + 0) = 1` and `bf16(1 * x) = x`, with no tap to mask. So a
// DFlash2 draft carrying that conv must produce BIT-IDENTICAL logits to the same
// draft with no conv at all. That separates "the conv is wired into the right
// places and perturbs nothing it should not" from "the conv is wired somewhere",
// which a tolerance-based comparison cannot.

namespace {

// Minimal safetensors writer: header length (u64 LE) + header JSON + payload,
// which is the whole format. Mirrors the one in
// tests/vllm/multimodal/ltx2_video_fixture.h rather than adding a dependency on
// it, because that header carries an entire LTX-2.5 fixture with it.
struct StEntry {
  // SPEC-DFLASH2-QUANT-LMHEAD (#1628): a NON-BF16 tensor, which every case
  // before that row could do without. When `dtype` is anything but "BF16" the
  // payload is `raw` and `bf16` is unused, so the ~40 existing three-field
  // brace initializers below stay byte-for-byte what they were. The two
  // constructors are what keeps that true under `-Werror=missing-field-
  // initializers`, which rejects an aggregate initializer that leaves a member
  // to its default.
  StEntry() = default;
  StEntry(std::string n, std::vector<int64_t> s, std::vector<uint16_t> b)
      : name(std::move(n)), shape(std::move(s)), bf16(std::move(b)) {}
  StEntry(std::string n, std::vector<int64_t> s, std::string dt,
          std::vector<uint8_t> r)
      : name(std::move(n)), shape(std::move(s)), dtype(std::move(dt)),
        raw(std::move(r)) {}

  std::string name;
  std::vector<int64_t> shape;
  std::vector<uint16_t> bf16;
  std::string dtype = "BF16";
  std::vector<uint8_t> raw;

  const void* Data() const {
    return dtype == "BF16" ? static_cast<const void*>(bf16.data())
                           : static_cast<const void*>(raw.data());
  }
  size_t Bytes() const {
    return dtype == "BF16" ? bf16.size() * sizeof(uint16_t) : raw.size();
  }
};

void WriteSafetensors(const std::vector<StEntry>& entries, const std::string& path) {
  json header = json::object();
  size_t offset = 0;
  for (const StEntry& e : entries) {
    const size_t nbytes = e.Bytes();
    header[e.name] = {{"dtype", e.dtype},
                      {"shape", e.shape},
                      {"data_offsets", json::array({offset, offset + nbytes})}};
    offset += nbytes;
  }
  // PAD the header to an 8-byte boundary, which is what the reference
  // `safetensors` writer does and therefore what every published checkpoint
  // looks like. It is not cosmetic here: the dense loader BORROWS the mapping
  // for a bf16 weight rather than copying it, and `WeightF32`
  // (`qwen3_5.cpp:1054`) reads that borrow through a `const uint16_t*`. An
  // unpadded header leaves `8 + header_len` odd, every tensor lands on an odd
  // offset, and the read is a misaligned load -- undefined behaviour that
  // `-fsanitize=undefined` halts on and that a strict-alignment target would
  // fault on. Measured: without this padding the loader-seam cases below exit 1
  // under the `address,undefined` lane with `load of misaligned address ... for
  // type 'const short unsigned int'`. That the loader ACCEPTS an unpadded
  // producer and then reads it as UB is a separate defect, filed as
  // [#1654](https://github.com/mudler/vllm.cpp/issues/1654) against
  // `ENG-LOAD-DIRECT-UPLOAD`; this fixture's job is to feed the loader what a
  // reference writer produces, not to be the gate for that.
  std::string hs = header.dump();
  while ((8 + hs.size()) % 8 != 0) hs.push_back(' ');
  std::ofstream out(path, std::ios::binary);
  uint64_t hlen = hs.size();
  out.write(reinterpret_cast<const char*>(&hlen), sizeof(hlen));
  out.write(hs.data(), static_cast<std::streamsize>(hs.size()));
  for (const StEntry& e : entries)
    out.write(static_cast<const char*>(e.Data()),
              static_cast<std::streamsize>(e.Bytes()));
}

// Deterministic bf16 fill: value(i) = amp * sin(seed + 0.7*i), the same shape of
// generator tests/vllm/models/test_qwen3_dflash_forward.cpp uses.
std::vector<uint16_t> Fill(int64_t n, double seed, double amp) {
  std::vector<uint16_t> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    v[static_cast<size_t>(i)] =
        vt::F32ToBF16(static_cast<float>(amp * std::sin(seed + 0.7 * static_cast<double>(i))));
  return v;
}

std::vector<uint16_t> Const(int64_t n, float value) {
  return std::vector<uint16_t>(static_cast<size_t>(n), vt::F32ToBF16(value));
}

struct Dims {
  int64_t H = 8, Hq = 2, Hkv = 1, Dh = 4, I = 6, vocab = 8, layers = 2, taps_fc = 2;
  int64_t conv_taps = 0;      // 0 = a DFlash1 checkpoint (no conv tensors at all)
  int64_t conv_group = 4;
  int64_t block = 8;          // the conv's query block, 1 + k
  // Which conv is non-identity, and on which side. An identity conv is
  // taps=1/base=1/projection=0; a non-identity one gets a real projection.
  bool attn_conv_active = false;
  bool mlp_conv_active = false;
  int active_side = -1;       // -1 = both sides active; 0 = prepare only; 1 = finish only
  // `base_kernel[side]`, per side. Both 1.0 with a zero projection is the exact
  // bf16 identity; making the two DIFFER is what separates "the model passes
  // side 0 to prepare and side 1 to finish" from "the model passes a side".
  float base_side0 = 1.0f;
  float base_side1 = 1.0f;
  // SPEC-DFLASH2 W3 (#1314): the candidate selector. Present whenever
  // `conv_taps > 0`, because a DFlash2 checkpoint carries BOTH mechanisms and
  // the loader refuses a draft that declares only one.
  int64_t sel_rank = 4;
  int64_t sel_top_k = 3;
  // The two OUTPUT SCALARS. `z-lab/Qwen3.8-27B-DFlash2` declares NEITHER and
  // takes 1.0 / disabled; `z-lab/Muse-Glimmer-30B-DFlash2` declares
  // 0.19611613513818404 and 20.0 (#1327, `## Risks/decisions` D9). A negative
  // value here means "do not declare the key at all", which is the 27B's shape
  // and the one a defaults-only gate silently measures.
  double output_multiplier = -1.0;
  double final_logit_softcapping = -1.0;
};

// A scratch directory holding one model.safetensors, removed on scope exit.
class ScratchCkpt {
 public:
  explicit ScratchCkpt(const std::vector<StEntry>& entries) {
    static int counter = 0;
    dir_ = fs::temp_directory_path() /
           ("vllmcpp_dflash2_ckpt_" + std::to_string(counter++) + "_" +
            std::to_string(static_cast<long long>(::getpid())));
    fs::create_directories(dir_);
    WriteSafetensors(entries, (dir_ / "model.safetensors").string());
  }
  ~ScratchCkpt() {
    std::error_code ec;
    fs::remove_all(dir_, ec);
  }
  ScratchCkpt(const ScratchCkpt&) = delete;
  ScratchCkpt& operator=(const ScratchCkpt&) = delete;
  std::string shard() const { return (dir_ / "model.safetensors").string(); }

 private:
  fs::path dir_;
};

std::vector<StEntry> DraftEntries(const Dims& dm) {
  const int64_t qdim = dm.Hq * dm.Dh, kdim = dm.Hkv * dm.Dh;
  std::vector<StEntry> e;
  e.push_back({"embed_tokens.weight", {dm.vocab, dm.H}, Fill(dm.vocab * dm.H, 0.1, 0.3)});
  e.push_back({"fc.weight", {dm.H, dm.H * dm.taps_fc}, Fill(dm.H * dm.H * dm.taps_fc, 0.2, 0.2)});
  e.push_back({"hidden_norm.weight", {dm.H}, Fill(dm.H, 0.3, 0.5)});
  e.push_back({"norm.weight", {dm.H}, Fill(dm.H, 0.4, 0.5)});
  e.push_back({"lm_head.weight", {dm.vocab, dm.H}, Fill(dm.vocab * dm.H, 0.5, 0.3)});
  for (int64_t l = 0; l < dm.layers; ++l) {
    const std::string b = "layers." + std::to_string(l) + ".";
    const double s = 1.0 + static_cast<double>(l);
    e.push_back({b + "input_layernorm.weight", {dm.H}, Fill(dm.H, s + 0.1, 0.6)});
    e.push_back({b + "post_attention_layernorm.weight", {dm.H}, Fill(dm.H, s + 0.2, 0.6)});
    e.push_back({b + "self_attn.q_proj.weight", {qdim, dm.H}, Fill(qdim * dm.H, s + 0.3, 0.25)});
    e.push_back({b + "self_attn.k_proj.weight", {kdim, dm.H}, Fill(kdim * dm.H, s + 0.4, 0.25)});
    e.push_back({b + "self_attn.v_proj.weight", {kdim, dm.H}, Fill(kdim * dm.H, s + 0.5, 0.25)});
    e.push_back({b + "self_attn.o_proj.weight", {dm.H, qdim}, Fill(dm.H * qdim, s + 0.6, 0.25)});
    e.push_back({b + "self_attn.q_norm.weight", {dm.Dh}, Fill(dm.Dh, s + 0.7, 0.7)});
    e.push_back({b + "self_attn.k_norm.weight", {dm.Dh}, Fill(dm.Dh, s + 0.8, 0.7)});
    e.push_back({b + "mlp.gate_proj.weight", {dm.I, dm.H}, Fill(dm.I * dm.H, s + 0.9, 0.25)});
    e.push_back({b + "mlp.up_proj.weight", {dm.I, dm.H}, Fill(dm.I * dm.H, s + 1.1, 0.25)});
    e.push_back({b + "mlp.down_proj.weight", {dm.H, dm.I}, Fill(dm.H * dm.I, s + 1.2, 0.25)});
    if (dm.conv_taps > 0) {
      const int64_t groups = dm.H / dm.conv_group;
      const int64_t proj_out = 2 * dm.conv_taps * groups;
      for (int which = 0; which < 2; ++which) {
        const std::string cp = b + (which == 0 ? "attention_conv." : "mlp_conv.");
        const bool active = which == 0 ? dm.attn_conv_active : dm.mlp_conv_active;
        // base_kernel is [2 SIDES, taps, H] and is ALL ONES: with a zero
        // projection that is the exact bf16 identity at taps=1, and with a real
        // projection it is upstream's `base + delta` with base 1.
        // base_kernel[side][tap][channel]. Tap 0 carries the side's scale; every
        // HIGHER tap is 1 when this conv is active and ZERO when it is not.
        //
        // The zero matters, and getting it wrong once already produced a gate
        // that could not tell one missing call site from none: with base 1 on
        // every tap and a zero projection, a taps=2 conv is `x[i] + x[i-1]` and
        // NOT the identity, so an "inactive" conv still moved the logits and an
        // "attention_conv only" arm was quietly exercising both convs.
        std::vector<uint16_t> base;
        for (int side = 0; side < 2; ++side) {
          const float scale = side == 0 ? dm.base_side0 : dm.base_side1;
          for (int64_t t = 0; t < dm.conv_taps; ++t) {
            const std::vector<uint16_t> row =
                Const(dm.H, t == 0 ? scale : (active ? 1.0f : 0.0f));
            base.insert(base.end(), row.begin(), row.end());
          }
        }
        e.push_back({cp + "base_kernel", {2, dm.conv_taps, dm.H}, base});
        std::vector<uint16_t> proj(static_cast<size_t>(proj_out * dm.H), vt::F32ToBF16(0.0f));
        if (active) {
          const std::vector<uint16_t> live = Fill(proj_out * dm.H, s + 2.0 + which, 0.4);
          for (int64_t r = 0; r < proj_out; ++r) {
            // Row r of the projection produces coefficient
            // [side = r / (taps*groups)][tap][group]. Zeroing the rows of the
            // side this case does not exercise is what makes "prepare only" and
            // "finish only" separable.
            const int64_t side = r / (dm.conv_taps * groups);
            if (dm.active_side >= 0 && side != dm.active_side) continue;
            for (int64_t c = 0; c < dm.H; ++c)
              proj[static_cast<size_t>(r * dm.H + c)] = live[static_cast<size_t>(r * dm.H + c)];
          }
        }
        e.push_back({cp + "kernel_projection.weight", {proj_out, dm.H}, proj});
      }
    }
  }
  if (dm.conv_taps > 0) {
    // SPEC-DFLASH2 W3 (#1314): the selector's three tensors, under the exact
    // names the published checkpoint stores them under. The two codebooks are
    // filled with DIFFERENT generators, so a load that swapped them (they share
    // a shape) produces different scores rather than the same ones.
    e.push_back({"candidate_selector.hidden_projection.weight", {dm.sel_rank, dm.H},
                 Fill(dm.sel_rank * dm.H, 3.3, 0.35)});
    e.push_back({"candidate_selector.predecessor_codebook", {dm.vocab, dm.sel_rank},
                 Fill(dm.vocab * dm.sel_rank, 4.4, 0.45)});
    e.push_back({"candidate_selector.successor_codebook", {dm.vocab, dm.sel_rank},
                 Fill(dm.vocab * dm.sel_rank, 5.5, 0.55)});
  }
  return e;
}

HfConfig DraftConfig(const Dims& dm) {
  HfConfig c;
  c.hidden_size = dm.H;
  c.num_attention_heads = dm.Hq;
  c.num_key_value_heads = dm.Hkv;
  c.head_dim = dm.Dh;
  c.rotary_dim = dm.Dh;
  c.rope_theta = 1e7;
  c.intermediate_size = dm.I;
  c.vocab_size = dm.vocab;
  c.num_hidden_layers = dm.layers;
  c.rms_norm_eps = 1e-6;
  c.sliding_window = 2048;
  c.layer_types = std::vector<std::string>(static_cast<size_t>(dm.layers), "sliding_attention");
  c.raw = nlohmann::json::object();
  c.raw["dflash_config"] = json::object();
  c.raw["dflash_config"]["mask_token_id"] = 7;
  if (dm.conv_taps > 0) {
    c.raw["dflash_config"]["conv_kernel_size"] = dm.conv_taps;
    c.raw["dflash_config"]["conv_group_size"] = dm.conv_group;
    c.raw["dflash_config"]["block_size"] = dm.block;
    c.raw["dflash_config"]["selector_rank"] = dm.sel_rank;
    c.raw["dflash_config"]["selector_top_k"] = dm.sel_top_k;
    // Declared only when the case asks for it: the 27B draft declares neither,
    // and a gate built from that draft alone measures the default path.
    if (dm.output_multiplier >= 0.0)
      c.raw["dflash_config"]["output_multiplier"] = dm.output_multiplier;
    if (dm.final_logit_softcapping >= 0.0)
      c.raw["dflash_config"]["final_logit_softcapping"] = dm.final_logit_softcapping;
  }
  c.raw["block_size"] = dm.block;
  c.raw["is_causal"] = false;
  return c;
}

// Load through the PRODUCTION weight loader, off a real safetensors file.
Qwen3DFlashWeights LoadDraft(const ScratchCkpt& ck, const Dims& dm, const HfConfig& c) {
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(ck.shard()));
  Qwen3DFlashWeights w = vllm::LoadQwen3DFlash(shards, c, dm.taps_fc, /*mask_token_id=*/7);
  // What the loader does with the resolved k (src/vllm/entrypoints/model_loader.cpp,
  // LoadDflashDraft): the conv's block is 1 + k and not the checkpoint key.
  if (w.IsDflash2()) w.conv_block_size = dm.block;
  return w;
}

// One draft forward over a single uniform (1+k) query block, through the
// context-free body. `T` is the block, so `cu` is {0, T}.
std::vector<float> Forward(const Qwen3DFlashWeights& w, const HfConfig& c, int64_t T) {
  std::vector<int32_t> ids(static_cast<size_t>(T));
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) {
    ids[static_cast<size_t>(i)] = static_cast<int32_t>(i % c.vocab_size);
    pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  }
  const std::vector<int32_t> cu = {0, static_cast<int32_t>(T)};
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  return Qwen3DFlashModel::ForwardBlockLogits(ids, pos, cu, w, c, q);
}

// The same block through the CONTEXT-AWARE body -- the one
// `DflashProposeBlock` and the runner's gathered path call -- with an empty
// context, so the two bodies are exercised on identical inputs.
std::vector<float> ForwardWithContext(const Qwen3DFlashWeights& w, const HfConfig& c, int64_t T) {
  std::vector<int32_t> ids(static_cast<size_t>(T));
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) {
    ids[static_cast<size_t>(i)] = static_cast<int32_t>(i % c.vocab_size);
    pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  }
  const std::vector<int32_t> cu = {0, static_cast<int32_t>(T)};
  const std::vector<int32_t> ctx_cu = {0, 0};
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  return Qwen3DFlashModel::ForwardBlockLogitsWithContext({}, {}, ctx_cu, ids, pos, cu, w, c, q);
}

bool BitEqual(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return false;
  return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace

TEST_CASE("dflash2 weights: the production loader reads the conv tensors off a real shard") {
  Dims dm;
  dm.conv_taps = 2;
  const ScratchCkpt ck(DraftEntries(dm));
  const HfConfig c = DraftConfig(dm);
  const Qwen3DFlashWeights w = LoadDraft(ck, dm, c);
  CHECK(w.IsDflash2());
  CHECK(w.conv_taps == 2);
  CHECK(w.conv_group_size == 4);
  REQUIRE(w.layers.size() == 2);
  for (size_t l = 0; l < w.layers.size(); ++l) {
    CAPTURE(l);
    CHECK_FALSE(w.layers[l].attention_conv.Empty());
    CHECK_FALSE(w.layers[l].mlp_conv.Empty());
    // [2 SIDES, taps, H] -- dim 0 is prepare/finish, NOT a tap.
    CHECK(w.layers[l].attention_conv.base_kernel.rank == 3);
    CHECK(w.layers[l].attention_conv.base_kernel.shape[0] == 2);
    CHECK(w.layers[l].attention_conv.base_kernel.shape[1] == dm.conv_taps);
    CHECK(w.layers[l].attention_conv.base_kernel.shape[2] == dm.H);
    // 2 * taps * num_groups = 2*2*(8/4) = 8, the shape upstream's
    // kernel_projection produces (1280 at the published 27B's 5120/16).
    CHECK(w.layers[l].mlp_conv.kernel_projection.shape[0] == 2 * dm.conv_taps * (dm.H / dm.conv_group));
    CHECK(w.layers[l].mlp_conv.kernel_projection.shape[1] == dm.H);
  }
}

TEST_CASE("dflash1 weights: a draft with no conv keys stays a DFlash1 draft") {
  // The inertness half. A DFlash1 checkpoint declares no conv key and ships no
  // conv tensor, so `IsDflash2()` is false and every layer body runs the
  // pre-W2 op sequence -- which the identity case below turns into a
  // BIT-IDENTICAL assertion rather than an argument.
  Dims dm;  // conv_taps 0
  const ScratchCkpt ck(DraftEntries(dm));
  const HfConfig c = DraftConfig(dm);
  const Qwen3DFlashWeights w = LoadDraft(ck, dm, c);
  CHECK_FALSE(w.IsDflash2());
  CHECK(w.conv_taps == 0);
  CHECK(w.layers[0].attention_conv.Empty());
  CHECK(w.layers[0].mlp_conv.Empty());
}

TEST_CASE("dflash2 forward: an IDENTITY conv is BIT-IDENTICAL to no conv at all") {
  // taps=1, base_kernel all ones, kernel_projection all zeros. In bf16 that is
  // exactly `out = 1 * x`, with no tap to mask -- so a DFlash2 draft carrying it
  // must reproduce the DFlash1 logits BIT-FOR-BIT. This is what separates
  // "correctly placed and transparent" from "applied somewhere".
  for (int64_t block : {int64_t{8}, int64_t{16}}) {
    CAPTURE(block);
    Dims base;
    base.block = block;
    Dims ident = base;
    ident.conv_taps = 1;
    const ScratchCkpt ck0(DraftEntries(base));
    const ScratchCkpt ck1(DraftEntries(ident));
    const std::vector<float> without = Forward(LoadDraft(ck0, base, DraftConfig(base)),
                                               DraftConfig(base), block);
    const std::vector<float> with = Forward(LoadDraft(ck1, ident, DraftConfig(ident)),
                                            DraftConfig(ident), block);
    REQUIRE(without.size() == with.size());
    CHECK(BitEqual(without, with));
  }
}

TEST_CASE("dflash2 forward: the conv is LOAD-BEARING at block 8 and at block 16") {
  // The reachability assertion. A real (non-identity) conv must move the draft
  // logits at BOTH published block shapes: `z-lab/Qwen3.8-27B-DFlash2` ships
  // block 8 and `z-lab/Muse-Glimmer-30B-DFlash2` ships block 16. Deleting the
  // DflashConvPrepare/DflashConvFinish call sites in the layer body makes each
  // comparison below EQUAL and this case red.
  for (int64_t block : {int64_t{8}, int64_t{16}}) {
    CAPTURE(block);
    Dims base;
    base.block = block;
    Dims live = base;
    live.conv_taps = 2;
    live.attn_conv_active = true;
    live.mlp_conv_active = true;
    const ScratchCkpt ck0(DraftEntries(base));
    const ScratchCkpt ck1(DraftEntries(live));
    const std::vector<float> without = Forward(LoadDraft(ck0, base, DraftConfig(base)),
                                               DraftConfig(base), block);
    const std::vector<float> with = Forward(LoadDraft(ck1, live, DraftConfig(live)),
                                            DraftConfig(live), block);
    REQUIRE(without.size() == with.size());
    CHECK_FALSE(BitEqual(without, with));
    // And the CONTEXT-AWARE body -- a separate layer body, separately wired --
    // must move too. Its DFlash1 output is its own baseline, because the two
    // bodies build the attention differently.
    const std::vector<float> ctx_without =
        ForwardWithContext(LoadDraft(ck0, base, DraftConfig(base)), DraftConfig(base), block);
    const std::vector<float> ctx_with =
        ForwardWithContext(LoadDraft(ck1, live, DraftConfig(live)), DraftConfig(live), block);
    REQUIRE(ctx_without.size() == ctx_with.size());
    CHECK_FALSE(BitEqual(ctx_without, ctx_with));
  }
}

TEST_CASE("dflash2 forward: the prepare side and the finish side land in DIFFERENT places") {
  // `base_kernel` dim 0 is prepare/finish. Driving only side 0 convolves the
  // sublayer INPUT; driving only side 1 convolves its OUTPUT. If the side index
  // were ignored -- or if both calls read the same half -- these two would agree.
  Dims prep;
  prep.conv_taps = 2;
  prep.attn_conv_active = true;
  prep.mlp_conv_active = true;
  prep.active_side = 0;
  Dims fin = prep;
  fin.active_side = 1;
  const ScratchCkpt ck0(DraftEntries(prep));
  const ScratchCkpt ck1(DraftEntries(fin));
  const std::vector<float> a = Forward(LoadDraft(ck0, prep, DraftConfig(prep)),
                                       DraftConfig(prep), prep.block);
  const std::vector<float> b = Forward(LoadDraft(ck1, fin, DraftConfig(fin)),
                                       DraftConfig(fin), fin.block);
  REQUIRE(a.size() == b.size());
  CHECK_FALSE(BitEqual(a, b));
}

TEST_CASE("dflash2 forward: attention_conv and mlp_conv wrap DIFFERENT sublayers") {
  // Two convs, one per sublayer, and upstream builds them with identical
  // arguments -- so nothing but the CALL SITE distinguishes them. Wiring both to
  // the attention (or both to the MLP) would make these two runs agree.
  Dims attn;
  attn.conv_taps = 2;
  attn.attn_conv_active = true;
  Dims mlp = attn;
  mlp.attn_conv_active = false;
  mlp.mlp_conv_active = true;
  const ScratchCkpt ck0(DraftEntries(attn));
  const ScratchCkpt ck1(DraftEntries(mlp));
  const std::vector<float> a = Forward(LoadDraft(ck0, attn, DraftConfig(attn)),
                                       DraftConfig(attn), attn.block);
  const std::vector<float> b = Forward(LoadDraft(ck1, mlp, DraftConfig(mlp)),
                                       DraftConfig(mlp), mlp.block);
  REQUIRE(a.size() == b.size());
  CHECK_FALSE(BitEqual(a, b));
}

TEST_CASE("dflash2 forward: a ragged query block is REFUSED rather than mis-masked") {
  // The conv masks its taps by `row index mod conv_block_size`, which is the
  // intra-block offset only while every request block is contiguous and
  // block-aligned. A ragged batch would mask the WRONG taps and be invisible:
  // the verify is lossless, so only acceptance would move.
  Dims dm;
  dm.conv_taps = 2;
  const ScratchCkpt ck(DraftEntries(dm));
  const HfConfig c = DraftConfig(dm);
  const Qwen3DFlashWeights w = LoadDraft(ck, dm, c);
  std::vector<int32_t> ids(12), pos(12);
  for (int i = 0; i < 12; ++i) {
    ids[static_cast<size_t>(i)] = i % 8;
    pos[static_cast<size_t>(i)] = i;
  }
  const std::vector<int32_t> ragged = {0, 5, 12};  // neither block is 8 rows
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  CHECK_THROWS_WITH_AS(Qwen3DFlashModel::ForwardBlockLogits(ids, pos, ragged, w, c, q),
                       doctest::Contains("conv_block_size"), std::exception);
}

TEST_CASE("dflash1 propose: a DFlash1 draft still proposes through the same entry") {
  // The instrument's precondition again: the refusal above must be about DFlash2
  // and not about `DflashProposeBlock`. The same call on a DFlash1 draft returns
  // k tokens per request.
  Dims dm;  // conv_taps 0
  const ScratchCkpt ck(DraftEntries(dm));
  const HfConfig c = DraftConfig(dm);
  const Qwen3DFlashWeights w = LoadDraft(ck, dm, c);
  REQUIRE_FALSE(w.IsDflash2());
  const int64_t T = dm.block;
  std::vector<int32_t> ids(static_cast<size_t>(T)), pos(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) {
    ids[static_cast<size_t>(i)] = static_cast<int32_t>(i % c.vocab_size);
    pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  }
  const std::vector<int32_t> cu = {0, static_cast<int32_t>(T)};
  const std::vector<int32_t> ctx_cu = {0, 0};
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const vllm::v1::DflashProposeResult r = vllm::v1::DflashProposeBlock(
      w, c, {}, {}, ctx_cu, ids, pos, cu, 1, static_cast<int>(T - 1), q);
  REQUIRE(r.draft_token_ids.size() == 1);
  CHECK(r.draft_token_ids[0].size() == static_cast<size_t>(T - 1));
}

TEST_CASE("dflash2 forward: prepare reads base_kernel[0] and finish reads base_kernel[1]") {
  // `base_kernel` dim 0 is the SIDE. With taps=1 and a ZERO projection the conv
  // is exactly `out = base[side] * x` in bf16, so the two sides are separable by
  // a single scalar each and nothing else in the model changes.
  //
  //   A  base = (1, 1)  -> the exact identity
  //   B  base = (1, 2)  -> only the FINISH side scales
  //   C  base = (2, 1)  -> only the PREPARE side scales
  //
  // A model that passed side 0 to both calls would make B the identity and B == A;
  // one that passed side 1 to both would make C the identity. Comparing B and C
  // to A pins BOTH directions, which comparing them only to each other does not.
  Dims a;
  a.conv_taps = 1;
  Dims b = a;
  b.base_side1 = 2.0f;
  Dims cc = a;
  cc.base_side0 = 2.0f;
  const ScratchCkpt cka(DraftEntries(a));
  const ScratchCkpt ckb(DraftEntries(b));
  const ScratchCkpt ckc(DraftEntries(cc));
  const std::vector<float> la = Forward(LoadDraft(cka, a, DraftConfig(a)), DraftConfig(a), a.block);
  const std::vector<float> lb = Forward(LoadDraft(ckb, b, DraftConfig(b)), DraftConfig(b), b.block);
  const std::vector<float> lc = Forward(LoadDraft(ckc, cc, DraftConfig(cc)), DraftConfig(cc), cc.block);
  CHECK_FALSE(BitEqual(la, lb));  // the FINISH side must reach the model
  CHECK_FALSE(BitEqual(la, lc));  // the PREPARE side must reach the model
  CHECK_FALSE(BitEqual(lb, lc));  // and they must land in different places
}

namespace {

// The DEVICE-KV body -- `ForwardBlockLogitsWithDeviceKV`, which is what
// `GPUModelRunner::propose_drafts_block` calls in production -- over one request
// with a two-row context appended through the production append path.
std::vector<float> ForwardDeviceKV(const Qwen3DFlashWeights& w, const HfConfig& c, int64_t T) {
  std::vector<int32_t> ids(static_cast<size_t>(T)), pos(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) {
    ids[static_cast<size_t>(i)] = static_cast<int32_t>(i % c.vocab_size);
    pos[static_cast<size_t>(i)] = static_cast<int32_t>(2 + i);
  }
  const std::vector<int32_t> cu = {0, static_cast<int32_t>(T)};
  const std::vector<int32_t> ctx_cu = {0, 2};
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  std::vector<float> ctx(static_cast<size_t>(2 * c.hidden_size));
  for (size_t i = 0; i < ctx.size(); ++i)
    ctx[i] = 0.2f * static_cast<float>(std::sin(0.13 * static_cast<double>(i) + 0.4));
  auto store = Qwen3DFlashModel::MakeDeviceKVStore(c, q, kUnitCtxSlots);
  Qwen3DFlashModel::AppendContextKVDevice(*store, ctx, {0, 1}, w, c, q);
  std::vector<vllm::DflashDeviceKVStore*> stores = {store.get()};
  return Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV(stores, ctx_cu, ids, pos, cu, w, c, q);
}

}  // namespace

TEST_CASE("dflash2 forward: EACH conv reaches EACH layer body separately") {
  // Three bodies carry the DFlash draft's layer loop --
  // `ForwardBlockLogits` (context-free), `ForwardWithCtxKVDev` (context-aware,
  // what `DflashProposeBlock` calls) and `ForwardPagedBody` (the paged store,
  // what the runner's production decode path reaches through
  // `ForwardBlockLogitsWithDeviceKV`) -- and each has its OWN four call sites.
  //
  // A case that activates BOTH convs cannot tell one missing call site from
  // none: the other conv still moves the logits and the comparison still passes.
  // That is exactly what a mutation deleting only the context-aware body's
  // attention_conv proved. So each conv is driven ALONE, through each body.
  struct Arm {
    const char* name;
    bool attn;
    bool mlp;
  };
  const Arm arms[] = {{"attention_conv only", true, false}, {"mlp_conv only", false, true}};
  for (const Arm& arm : arms) {
    CAPTURE(arm.name);
    Dims base;
    Dims live = base;
    live.conv_taps = 2;
    live.attn_conv_active = arm.attn;
    live.mlp_conv_active = arm.mlp;
    const ScratchCkpt ck0(DraftEntries(base));
    const ScratchCkpt ck1(DraftEntries(live));
    const Qwen3DFlashWeights w0 = LoadDraft(ck0, base, DraftConfig(base));
    const Qwen3DFlashWeights w1 = LoadDraft(ck1, live, DraftConfig(live));
    const HfConfig c0 = DraftConfig(base);
    const HfConfig c1 = DraftConfig(live);
    CHECK_FALSE(BitEqual(Forward(w0, c0, base.block), Forward(w1, c1, live.block)));
    CHECK_FALSE(BitEqual(ForwardWithContext(w0, c0, base.block),
                         ForwardWithContext(w1, c1, live.block)));
    CHECK_FALSE(BitEqual(ForwardDeviceKV(w0, c0, base.block),
                         ForwardDeviceKV(w1, c1, live.block)));
  }
}

// ===========================================================================
// PART 3 — the CANDIDATE SELECTOR (SPEC-DFLASH2 W3, #1314).
//
// REACHABILITY (.agents/reachability.md). The chain a user arrives through is
//   LoadedEngine::FromModelDir / ResolveSpecConfig
//     -> LoadDflashDraft            (src/vllm/entrypoints/model_loader.cpp)
//       -> vllm::LoadQwen3DFlash    (reads candidate_selector.*, sets
//                                    selector_rank/top_k and the two scalars)
//     -> GPUModelRunner::propose_drafts_block
//       -> Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV (+ final_out)
//       -> vllm::v1::Dflash2SelectCandidates
//            -> Qwen3DFlash2Model::ComputeCandidates  -> vt::TopKValuesIndices
//            -> Qwen3DFlash2Model::SelectorEdgeScores -> vt::MatmulBT,
//                                                        vt::Dflash2SelectorEdges
//       -> vllm::v1::Dflash2WalkPath              (W4, and it IS the draft)
//
// The cases below enter at `vllm::LoadQwen3DFlash` over a real safetensors file
// and at `Dflash2SelectCandidates`, which is the SAME function the runner calls
// -- W3 collapses W2's two copies of that step into one, so there is no
// test-only duplicate to gate by mistake.
namespace {

// A DFlash2 draft with a selector, at the two scalar settings the two published
// checkpoints ship.
Dims Dflash2Dims(bool muse_glimmer_scalars) {
  Dims dm;
  dm.conv_taps = 2;
  dm.attn_conv_active = true;
  dm.mlp_conv_active = true;
  if (muse_glimmer_scalars) {
    // `z-lab/Muse-Glimmer-30B-DFlash2`'s own values (#1327).
    dm.output_multiplier = 0.19611613513818404;
    dm.final_logit_softcapping = 20.0;
  }
  return dm;
}

// SPEC-DFLASH2-QUANT-LMHEAD (#1628): a SCOPED `VT_LMHEAD_FP4`.
//
// `VT_LMHEAD_FP4=0` is the documented in-binary rollback for the packed head:
// it turns `DenseLmHeadTakesNvfp4` off, so `LoadDflashSharedLmHead` falls
// through to the bf16 read and a target whose `lm_head.weight` is NVFP4 refuses
// by name. That is CORRECT and it fails closed -- but the cases below gate the
// PACKED ARM, and running them under the rollback makes four correct refusals
// read as four broken tests. So each one names the arm it measures instead of
// inheriting the ambient value, and the rollback gets a case of its OWN
// (`the VT_LMHEAD_FP4 rollback REFUSES ...`) rather than being a mode in which
// this suite is simply red.
//
// `DenseLmHeadFp4Enabled` reads `getenv` on every call and latches nothing, so
// a scoped set is enough and there is no process-order hazard.
class ScopedLmHeadFp4 {
 public:
  explicit ScopedLmHeadFp4(bool on) {
    const char* prev = std::getenv("VT_LMHEAD_FP4");
    had_ = prev != nullptr;
    if (had_) prev_ = prev;
    ::setenv("VT_LMHEAD_FP4", on ? "1" : "0", /*overwrite=*/1);
  }
  ~ScopedLmHeadFp4() {
    if (had_) {
      ::setenv("VT_LMHEAD_FP4", prev_.c_str(), /*overwrite=*/1);
    } else {
      ::unsetenv("VT_LMHEAD_FP4");
    }
  }
  ScopedLmHeadFp4(const ScopedLmHeadFp4&) = delete;
  ScopedLmHeadFp4& operator=(const ScopedLmHeadFp4&) = delete;

 private:
  bool had_ = false;
  std::string prev_;
};

std::string VecStr(const std::vector<int32_t>& v) {
  std::string out;
  for (int32_t x : v) {
    out += std::to_string(x);
    out += ' ';
  }
  return out;
}

// A deterministic synthetic block, so the selector's inputs are chosen by this
// file rather than by whatever the forward happens to produce. `[P*(1+k), W]`.
std::vector<float> Ramp(int64_t rows, int64_t width, double seed) {
  std::vector<float> v(static_cast<size_t>(rows * width));
  for (int64_t i = 0; i < rows * width; ++i)
    v[static_cast<size_t>(i)] =
        static_cast<float>(2.0 * std::sin(seed + 0.37 * static_cast<double>(i)));
  return v;
}

// ─── SPEC-DFLASH2-QUANT-LMHEAD (#1628): a QUANTIZED target lm_head ──────────
//
// The draft owns no head; it runs the TARGET's. Every fixture above hands it a
// bf16 one, which is the only shape `SharedHeadSource` could read until this
// row. `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` stores `lm_head.weight` as
// ModelOpt NVFP4, and that is the checkpoint the campaign's only speculative
// arm has.
//
// WHY THESE CONSTANTS. The fixture is built so the dequant is EXACT and the
// gate can therefore be bitwise rather than tolerance-based: every fp8-e4m3
// group scale is a power of two (0.5 / 1 / 2, bytes 0x30 / 0x38 / 0x40), the
// global `weight_scale_2` is 0.25, and the E2M1 code set is
// {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}. Every product of those is representable
// in bf16 with no rounding, so `values` below is not an approximation of what
// the loader will produce — it IS what it produces, and a load that lost the
// group scale or the global scale changes it.
constexpr int64_t kNvfp4Group = 16;

// H must be a multiple of 16 for the NVFP4 group grid, so this widens the
// default Dims (H 8) rather than reusing it.
Dims Nvfp4Dims() {
  Dims dm = Dflash2Dims(/*muse_glimmer_scalars=*/false);
  dm.H = 16;
  dm.Hq = 2;
  dm.Hkv = 1;
  dm.Dh = 8;
  dm.I = 6;
  dm.vocab = 16;
  dm.conv_group = 4;
  return dm;
}

struct Fp4Head {
  std::vector<StEntry> entries;  // lm_head.weight / _scale / _scale_2
  std::vector<float> values;     // [V,H] the EXACT f32 the dequant yields
};

// `variant` picks a different code stream, which is how a case shows the gate is
// sensitive to the head's CONTENT rather than to its presence.
Fp4Head MakeNvfp4LmHead(int64_t V, int64_t H, int variant) {
  REQUIRE(H % kNvfp4Group == 0);
  static const float kLut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  static const uint8_t kScaleByte[3] = {0x30, 0x38, 0x40};  // 0.5, 1.0, 2.0
  static const float kScaleVal[3] = {0.5f, 1.0f, 2.0f};
  const float scale2 = 0.25f;

  Fp4Head h;
  std::vector<uint8_t> packed(static_cast<size_t>(V * H / 2), 0);
  std::vector<uint8_t> scales(static_cast<size_t>(V * H / kNvfp4Group), 0);
  h.values.assign(static_cast<size_t>(V * H), 0.0f);
  for (int64_t o = 0; o < V; ++o) {
    for (int64_t g = 0; g < H / kNvfp4Group; ++g) {
      scales[static_cast<size_t>(o * (H / kNvfp4Group) + g)] =
          kScaleByte[(o + g + variant) % 3];
    }
    for (int64_t i = 0; i < H; ++i) {
      const int64_t code =
          (o * 7 + i * 5 + static_cast<int64_t>(variant) * 3) % 16;
      const uint8_t nib = static_cast<uint8_t>(code);
      uint8_t& byte = packed[static_cast<size_t>(o * (H / 2) + i / 2)];
      // ModelOpt / compressed-tensors nibble order: element 2j LOW, 2j+1 HIGH
      // (Nvfp4NibbleOrder::kLowFirst).
      if (i % 2 == 0) {
        byte = static_cast<uint8_t>((byte & 0xF0) | (nib & 0x0F));
      } else {
        byte = static_cast<uint8_t>((byte & 0x0F) | ((nib & 0x0F) << 4));
      }
      const float mag = kLut[nib & 0x7];
      const float sign = (nib & 0x8) != 0 ? -1.0f : 1.0f;
      const float gs = kScaleVal[(o + i / kNvfp4Group + variant) % 3];
      h.values[static_cast<size_t>(o * H + i)] = sign * mag * gs * scale2;
    }
  }
  StEntry w;
  w.name = "lm_head.weight";
  w.shape = {V, H / 2};
  w.dtype = "U8";
  w.raw = std::move(packed);
  StEntry ws;
  ws.name = "lm_head.weight_scale";
  ws.shape = {V, H / kNvfp4Group};
  ws.dtype = "F8_E4M3";
  ws.raw = std::move(scales);
  StEntry ws2;
  ws2.name = "lm_head.weight_scale_2";
  ws2.shape = {};
  ws2.dtype = "F32";
  ws2.raw.resize(sizeof(float));
  std::memcpy(ws2.raw.data(), &scale2, sizeof(float));
  h.entries = {std::move(w), std::move(ws), std::move(ws2)};
  return h;
}

// A BF16 target head under the same name, which is what every target before this
// row shipped and what the DFlash1 lane has run since SPEC-DFLASH-GGUF.
std::vector<StEntry> Bf16LmHeadEntries(int64_t V, int64_t H, double seed) {
  return {StEntry{"lm_head.weight", {V, H}, Fill(V * H, seed, 0.3)}};
}

// ─── The EXL3 SHARED HEAD — MODEL-QWEN35-EXL3-HEAD (#2495 item 6) ────────────
//
// `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` ships `lm_head.{trellis,suh,svh,mul1}`
// and NO `lm_head.weight`: k=5120, n=248320, bits 6, codebook 2, read from its
// safetensors header by range request (#2569). The draft owns no head and runs
// the target's, so this is the third storage form its shared read has to take.
//
// Held once and written TWO ways -- as the trellis, and as the SAME bytes
// decoded into a bf16 `.weight` -- so the forward comparison below is an
// equivalence between two readings of one head rather than a tolerance between
// two heads.
constexpr int kExl3HeadBits = 4;
constexpr int kExl3HeadCodebook = 2;  // `mul1`, the artifact's own
constexpr uint32_t kExl3Mul1Multiplier = 0x83DCD12DU;

struct Exl3Head {
  int64_t k = 0, n = 0;
  std::vector<uint16_t> trellis, suh, svh;
};

Exl3Head MakeExl3Head(int64_t V, int64_t H) {
  Exl3Head p;
  p.k = H;
  p.n = V;
  uint32_t s = 0x51ED270Fu;
  auto next = [&s] {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
  };
  p.trellis.resize(static_cast<size_t>(H / 16) * static_cast<size_t>(V / 16) *
                   16 * kExl3HeadBits);
  for (auto& w : p.trellis) w = static_cast<uint16_t>(next() & 0xffffu);
  p.suh.resize(static_cast<size_t>(H));
  p.svh.resize(static_cast<size_t>(V));
  // Sign vectors, which is what they are: +-1 in fp16 (`exl3.py:48-49`).
  for (auto& v : p.suh) v = vt::F32ToF16((next() & 1u) != 0u ? 1.0F : -1.0F);
  for (auto& v : p.svh) v = vt::F32ToF16((next() & 1u) != 0u ? 1.0F : -1.0F);
  return p;
}

std::vector<uint8_t> RawBytes16(const std::vector<uint16_t>& v) {
  std::vector<uint8_t> b(v.size() * 2);
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}

std::vector<StEntry> Exl3LmHeadEntries(const Exl3Head& p) {
  std::vector<uint8_t> mul1(4);
  std::memcpy(mul1.data(), &kExl3Mul1Multiplier, 4);
  return {
      StEntry{"lm_head.trellis", {p.k / 16, p.n / 16, 16 * kExl3HeadBits}, "I16",
              RawBytes16(p.trellis)},
      StEntry{"lm_head.suh", {p.k}, "F16", RawBytes16(p.suh)},
      StEntry{"lm_head.svh", {p.n}, "F16", RawBytes16(p.svh)},
      StEntry{"lm_head.mul1", {1}, "I32", mul1},
  };
}

// The decoded twin. `Exl3DequantLinear` yields [k, n]; a torch Linear stores
// [out, in] = [n, k], which is the raw-NK layout the bf16 arm reads.
std::vector<StEntry> Exl3DecodedLmHeadEntries(const Exl3Head& p) {
  std::vector<float> w(static_cast<size_t>(p.k) * static_cast<size_t>(p.n), 0.0F);
  vt::Exl3DequantLinear(p.trellis.data(), p.suh.data(), p.svh.data(), p.k, p.n,
                        kExl3HeadBits, kExl3HeadCodebook, w.data());
  std::vector<uint16_t> bf(static_cast<size_t>(p.k) * static_cast<size_t>(p.n));
  for (int64_t i = 0; i < p.n; ++i)
    for (int64_t j = 0; j < p.k; ++j)
      bf[static_cast<size_t>(i) * static_cast<size_t>(p.k) + static_cast<size_t>(j)] =
          vt::F32ToBF16(w[static_cast<size_t>(j) * static_cast<size_t>(p.n) +
                          static_cast<size_t>(i)]);
  return {StEntry{"lm_head.weight", {p.n, p.k}, bf}};
}

// EVERY EXL3 K and N is a multiple of 128: each side carries a blockwise
// Hadamard-128 at quantization time (`exl3_lib/quantize.py:15`) and the
// reference dequant declines anything else. That is what sizes `H` and `vocab`
// here, and it is the artifact's constraint rather than this fixture's taste.
Dims Exl3Dims() {
  Dims dm = Dflash2Dims(/*muse_glimmer_scalars=*/false);
  dm.H = 128;
  dm.Hq = 8;
  dm.Hkv = 2;
  dm.Dh = 16;
  dm.I = 64;
  dm.vocab = 128;
  dm.conv_group = 4;
  return dm;
}

std::vector<vllm::SafetensorsFile> OpenShards(const ScratchCkpt& ck) {
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(ck.shard()));
  return shards;
}

// The block forward, returning the post-final-norm hidden states as well — the
// exact rows the target's own head would be applied to.
std::vector<float> ForwardCapturingHidden(const Qwen3DFlashWeights& w,
                                          const HfConfig& c, int64_t T,
                                          std::vector<float>* hidden) {
  std::vector<int32_t> ids(static_cast<size_t>(T));
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) {
    ids[static_cast<size_t>(i)] = static_cast<int32_t>(i % c.vocab_size);
    pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  }
  const std::vector<int32_t> cu = {0, static_cast<int32_t>(T)};
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  return Qwen3DFlashModel::ForwardBlockLogits(ids, pos, cu, w, c, q,
                                              /*per_layer_out=*/nullptr, hidden);
}

// THE TARGET'S OWN shared-head logits path, over the same hidden states.
//
// `Qwen3_5MTPModel::ComputeLogits` is not a stand-in for it: the MTP drafter
// shares the TARGET's `lm_head` exactly as the DFlash draft does, and its body
// is the one line `DenseLogitsF32D` runs for a packed head
// (`MatmulNvfp4F32D(device, hidden_states, *lm_head_fp4_)`, qwen3_5.cpp). So
// this is the reference the row's gate is stated against: the top-K the selector
// consumes must be the top-K the target's own logits path produces.
std::vector<float> TargetSharedHeadLogits(const vllm::Nvfp4Weight& head,
                                          const std::vector<float>& hidden,
                                          int64_t T, int64_t H, int64_t V) {
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};
  std::vector<uint16_t> hb(static_cast<size_t>(T * H));
  for (size_t i = 0; i < hb.size(); ++i) hb[i] = vt::F32ToBF16(hidden[i]);
  vllm::dense_attn::DBuf hbuf(d, vt::DType::kBF16, {T, H}, hb.data());

  vllm::Qwen3_5DenseWeights target;
  target.lm_head_fp4 = head;
  vllm::Qwen3_5MTPWeights mtp_weights;  // kind defaults to kDense
  HfConfig tcfg;
  tcfg.hidden_size = H;
  tcfg.vocab_size = V;
  const vllm::Qwen3_5MTPModel mtp(mtp_weights, target, tcfg);
  const vllm::ForwardLogits fl = mtp.ComputeLogits(hbuf.t(), q);
  REQUIRE(fl.on_device());
  std::vector<float> out(static_cast<size_t>(T * V));
  vt::GetBackend(q.device.type).Copy(q, out.data(), fl.device_tensor.data,
                                     out.size() * sizeof(float));
  return out;
}

}  // namespace

TEST_CASE("dflash2 weights: the loader does NOT seed conv_block_size from the checkpoint") {
  // SPEC-DFLASH2 W4 (#1314), spec `## Owed` O5 item 1. Through W3
  // `LoadQwen3DFlash` copied `dflash_config.block_size` into `conv_block_size`
  // so a direct caller had "a usable value". That is what made
  // `LoadDflashDraft`'s own `draft->weights.conv_block_size = draft->k + 1;`
  // ungateable: deleting it left the checkpoint's PLAUSIBLE default behind, the
  // conv masked its taps against the wrong block, and the symptom is
  // acceptance-only and token-invisible -- W2 measured every suite staying green
  // under exactly that mutation.
  //
  // The seed is gone. A DFlash2 draft loaded and forwarded WITHOUT the resolved
  // `k` now refuses BY NAME instead of running against the wrong block, so the
  // silent-wrong has no shape left rather than a gate that has to reach an
  // anonymous-namespace function. This case pins that polarity in both
  // directions: unset refuses, set runs.
  const Dims dm = Dflash2Dims(/*muse_glimmer_scalars=*/false);
  const ScratchCkpt ck(DraftEntries(dm));
  const HfConfig c = DraftConfig(dm);
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(ck.shard()));
  Qwen3DFlashWeights raw = vllm::LoadQwen3DFlash(shards, c, dm.taps_fc,
                                                 /*mask_token_id=*/7);
  REQUIRE(raw.IsDflash2());
  // The precondition that makes this case non-vacuous: the checkpoint DOES carry
  // a block_size, and it is a value the conv could plausibly have used.
  REQUIRE(c.raw.contains("block_size"));
  CHECK(c.raw.at("block_size").get<int64_t>() == dm.block);
  CHECK(raw.conv_block_size == 0);

  const int64_t T = dm.block;
  std::vector<int32_t> ids(static_cast<size_t>(T)), pos(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) {
    ids[static_cast<size_t>(i)] = static_cast<int32_t>(i % c.vocab_size);
    pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  }
  const std::vector<int32_t> cu = {0, static_cast<int32_t>(T)};
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  CHECK_THROWS_WITH_AS(Qwen3DFlashModel::ForwardBlockLogits(ids, pos, cu, raw, c, q),
                       doctest::Contains("conv_block_size"), std::exception);
  // And with the resolved block -- what `LoadDflashDraft` writes -- it runs.
  raw.conv_block_size = dm.block;
  CHECK_NOTHROW(
      (void)Qwen3DFlashModel::ForwardBlockLogits(ids, pos, cu, raw, c, q));
  // A DFlash1 draft is untouched: it has no conv, so the field is meaningless
  // and the forward never reads it.
  const Dims one;  // conv_taps 0
  const ScratchCkpt ck1(DraftEntries(one));
  const HfConfig c1 = DraftConfig(one);
  std::vector<vllm::SafetensorsFile> shards1;
  shards1.push_back(vllm::SafetensorsFile::Open(ck1.shard()));
  const Qwen3DFlashWeights w1 = vllm::LoadQwen3DFlash(shards1, c1, one.taps_fc,
                                                      /*mask_token_id=*/7);
  REQUIRE_FALSE(w1.IsDflash2());
  CHECK(w1.conv_block_size == 0);
  CHECK_NOTHROW(
      (void)Qwen3DFlashModel::ForwardBlockLogits(ids, pos, cu, w1, c1, q));
}

TEST_CASE("dflash2 propose: the conv, the SELECTOR and the WALK all run, and it DRAFTS") {
  // The ORDER is the claim, and as of W4 the END of it is a draft rather than a
  // refusal. `DflashProposeBlock` runs the draft block forward -- grouped
  // convolution and all -- then `Dflash2SelectCandidates`, then
  // `Dflash2WalkPath`, whose output IS the returned draft. A DFlash2 draft
  // reaching here executes every line of W2, W3 and W4.
  //
  // WHY "IT RETURNED TOKENS" IS NOT THE ASSERTION. Deleting the DFlash2 branch
  // entirely also returns tokens -- the DFlash1 per-slot argmax over the block
  // logits -- and they are well-formed, the verify is lossless, and only
  // acceptance falls. That is the whole reason W1, W2 and W3 refused rather than
  // fell back. So this case asserts two things an argmax fallback cannot do:
  //
  //   1. every drafted id is one of the SELECTOR's own candidates for its step.
  //      The candidate set is `selector_top_k` = 3 wide over a vocabulary of 8,
  //      so a full-vocab argmax lands outside it most of the time. (This is a
  //      consistency check against the selector, and its value is exactly that
  //      discrimination, not correctness of the selector itself -- which
  //      test_ops_dflash2_selector_edges.cpp and the D9 cases below carry.)
  //   2. the drafts MOVE when D9's output scalars move. Those scalars touch the
  //      candidate VALUES and nothing else in the engine, so the block forward,
  //      the convolution and the block logits are identical between the two arms
  //      and the DFlash1 argmax would answer identically for both.
  //
  // And structurally: the argmax arm is entered on EMPTINESS and guarded by
  // `RefuseDflash1ArgmaxOnDflash2Block`, so deleting the walk's call site throws
  // by name instead of drafting quietly.
  const Dims plain = Dflash2Dims(false);
  const Dims muse = Dflash2Dims(true);
  const ScratchCkpt ckp(DraftEntries(plain));
  const ScratchCkpt ckm(DraftEntries(muse));
  const HfConfig cp = DraftConfig(plain);
  const HfConfig cm = DraftConfig(muse);
  const Qwen3DFlashWeights wp = LoadDraft(ckp, plain, cp);
  const Qwen3DFlashWeights wm = LoadDraft(ckm, muse, cm);
  REQUIRE(wp.IsDflash2());
  REQUIRE(wm.IsDflash2());

  const int64_t T = plain.block;  // one request, one (1+k) query block
  const int k = static_cast<int>(T - 1);
  std::vector<int32_t> ids(static_cast<size_t>(T)), pos(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) {
    ids[static_cast<size_t>(i)] = static_cast<int32_t>(i % cp.vocab_size);
    pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  }
  const std::vector<int32_t> cu = {0, static_cast<int32_t>(T)};
  const std::vector<int32_t> ctx_cu = {0, 0};
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  const vllm::v1::DflashProposeResult got =
      vllm::v1::DflashProposeBlock(wp, cp, {}, {}, ctx_cu, ids, pos, cu,
                                   /*num_reqs=*/1, k, q);
  REQUIRE(got.draft_token_ids.size() == 1);
  REQUIRE(got.draft_token_ids[0].size() == static_cast<size_t>(k));
  for (int32_t id : got.draft_token_ids[0]) {
    CHECK(id >= 0);
    CHECK(id < static_cast<int32_t>(cp.vocab_size));
  }

  // (1) MEMBERSHIP in the selector's candidate set, step by step. The block
  // logits and hidden come from the SAME forward the propose ran, so the
  // candidate set below is the one the walk chose from.
  std::vector<float> block_hidden;
  const std::vector<float> block_logits = Qwen3DFlashModel::ForwardBlockLogits(
      ids, pos, cu, wp, cp, q, nullptr, &block_hidden);
  const std::vector<int32_t> anchors = {ids[0]};
  const vllm::v1::Dflash2ProposeState sel = vllm::v1::Dflash2SelectCandidates(
      block_logits, block_hidden, anchors, /*num_reqs=*/1, k, wp, cp, q);
  REQUIRE(sel.top_k == 3);
  int outside_top_k_ids = 0;
  for (int step = 0; step < k; ++step) {
    bool member = false;
    for (int64_t c = 0; c < sel.top_k; ++c) {
      if (sel.candidates.ids[static_cast<size_t>(step * sel.top_k + c)] ==
          got.draft_token_ids[0][static_cast<size_t>(step)])
        member = true;
    }
    INFO("step ", step, " drafted ", got.draft_token_ids[0][static_cast<size_t>(step)]);
    CHECK(member);
    // The instrument's precondition: the candidate set must be a PROPER subset
    // of the vocabulary, or "member" is satisfied by everything.
    CHECK(sel.top_k < cp.vocab_size);
    if (got.draft_token_ids[0][static_cast<size_t>(step)] >= sel.top_k)
      ++outside_top_k_ids;
  }
  // ...and the ids are not the identity permutation of the slots, so membership
  // is a statement about token ids rather than about small integers.
  INFO("drafted ids at or above top_k: ", outside_top_k_ids, " of ", k);
  CHECK(outside_top_k_ids > 0);

  // (2) THE DRAFT IS NOT THE DFlash1 ARGMAX'S DRAFT, over the SAME block logits.
  // This is the discriminator the whole case exists for, and it is direct: the
  // fallback the guard protects against is exactly this function, on exactly
  // this input, and its answer is computed here rather than argued about.
  const std::vector<std::vector<int32_t>> argmax_drafts =
      vllm::v1::SampleDflashBlockDrafts(block_logits, /*num_reqs=*/1, k,
                                        wp.draft_vocab_size);
  REQUIRE(argmax_drafts.size() == 1);
  int argmax_agreements = 0;
  for (int step = 0; step < k; ++step)
    if (argmax_drafts[0][static_cast<size_t>(step)] ==
        got.draft_token_ids[0][static_cast<size_t>(step)])
      ++argmax_agreements;
  // MEASURED on 2026-08-20 at this fixture: walk `3 2 3 7 7 7 7` against argmax
  // `2 2 2 7 7 7 7` -- 5 of the 7 slots agree, 2 do not. The margin is logged
  // rather than assumed, because a fixture whose two answers coincided would
  // make this assertion pass for the wrong reason and look identical from here.
  INFO("walk ", VecStr(got.draft_token_ids[0]), " argmax ",
       VecStr(argmax_drafts[0]), " agreeing slots ", argmax_agreements, " of ", k);
  CHECK(argmax_drafts[0] != got.draft_token_ids[0]);

  // (3) The scalars are wired into this entry too. D9's own gate below measures
  // how often they flip the lattice's ORDER (8 of 45 predecessor slots over a
  // sweep of blocks); the walk visits only `k` of those slots per block, so
  // whether a given block's DRAFT moves is a property of the block. Asserted as
  // a MEASURED sweep with the count logged rather than as a single block that
  // might coincide.
  int blocks_whose_draft_moved = 0, blocks_run = 0;
  for (int shift = 0; shift < 6; ++shift) {
    std::vector<int32_t> sids(static_cast<size_t>(T));
    for (int64_t i = 0; i < T; ++i)
      sids[static_cast<size_t>(i)] =
          static_cast<int32_t>((i * 3 + shift) % cp.vocab_size);
    const vllm::v1::DflashProposeResult a = vllm::v1::DflashProposeBlock(
        wp, cp, {}, {}, ctx_cu, sids, pos, cu, /*num_reqs=*/1, k, q);
    const vllm::v1::DflashProposeResult bmuse = vllm::v1::DflashProposeBlock(
        wm, cm, {}, {}, ctx_cu, sids, pos, cu, /*num_reqs=*/1, k, q);
    ++blocks_run;
    if (a.draft_token_ids[0] != bmuse.draft_token_ids[0]) ++blocks_whose_draft_moved;
    // The instrument's own precondition, per block: the same arm against itself
    // must never differ, or the counter above measures nondeterminism.
    const vllm::v1::DflashProposeResult again = vllm::v1::DflashProposeBlock(
        wp, cp, {}, {}, ctx_cu, sids, pos, cu, /*num_reqs=*/1, k, q);
    CHECK(again.draft_token_ids[0] == a.draft_token_ids[0]);
  }
  // MEASURED on 2026-08-20 at this fixture: 5 of 6 blocks moved.
  INFO("blocks whose draft moved with the scalars: ", blocks_whose_draft_moved,
       " of ", blocks_run);
  CHECK(blocks_whose_draft_moved > 0);
  // The precondition for THAT comparison: the block LOGITS are identical between
  // the two arms, so any difference cannot come from the forward. The scalars
  // live only in the config's `dflash_config`; the weight fixtures are generated
  // from the same seeds.
  std::vector<float> muse_hidden;
  const std::vector<float> muse_logits = Qwen3DFlashModel::ForwardBlockLogits(
      ids, pos, cu, wm, cm, q, nullptr, &muse_hidden);
  CHECK(muse_logits == block_logits);
  CHECK(muse_hidden == block_hidden);
}

TEST_CASE("dflash2 weights: the production loader reads the SELECTOR off a real shard") {
  const Dims dm = Dflash2Dims(/*muse_glimmer_scalars=*/false);
  const ScratchCkpt ck(DraftEntries(dm));
  const HfConfig c = DraftConfig(dm);
  const Qwen3DFlashWeights w = LoadDraft(ck, dm, c);
  REQUIRE(w.IsDflash2());
  const vllm::Dflash2SelectorWeights& sel = w.candidate_selector;
  CHECK_FALSE(sel.Empty());
  CHECK(sel.rank == dm.sel_rank);
  CHECK(sel.top_k == dm.sel_top_k);
  // [rank, H] and NOT [H, rank]: on the published 27B rank 256 differs from
  // every other axis, so a transposed load is caught by the shape and not by a
  // wrong answer.
  CHECK(sel.hidden_projection.rank == 2);
  CHECK(sel.hidden_projection.shape[0] == dm.sel_rank);
  CHECK(sel.hidden_projection.shape[1] == dm.H);
  for (const vllm::OwnedTensor* book :
       {&sel.predecessor_codebook, &sel.successor_codebook}) {
    CHECK(book->rank == 2);
    CHECK(book->shape[0] == dm.vocab);
    CHECK(book->shape[1] == dm.sel_rank);
  }
  // WHICH codebook is which, asserted against the fixture's own bytes. The two
  // share a shape, so a loader that swapped them loads cleanly and scores every
  // transition with the roles reversed -- acceptance-only and token-invisible,
  // this row's signature defect. Swapping the two `LoadBf16Direct` calls left
  // every suite in this repository GREEN until this assertion existed; it was
  // found by running the mutation, not by reading the loader.
  {
    const std::vector<uint16_t> want_pred = Fill(dm.vocab * dm.sel_rank, 4.4, 0.45);
    const std::vector<uint16_t> want_succ = Fill(dm.vocab * dm.sel_rank, 5.5, 0.55);
    REQUIRE(want_pred != want_succ);  // the fixture must be able to tell them apart
    REQUIRE(sel.predecessor_codebook.bytes.size() == want_pred.size() * sizeof(uint16_t));
    CHECK(std::memcmp(sel.predecessor_codebook.bytes.data(), want_pred.data(),
                      want_pred.size() * sizeof(uint16_t)) == 0);
    CHECK(std::memcmp(sel.successor_codebook.bytes.data(), want_succ.data(),
                      want_succ.size() * sizeof(uint16_t)) == 0);
  }
  // The 27B draft declares NEITHER scalar, so both take upstream's default and
  // the softcap is DISABLED (0 means "no cap", never "cap at zero").
  CHECK(sel.output_multiplier == 1.0f);
  CHECK(sel.final_logit_softcapping == 0.0f);

  // A DFlash1 draft has no selector at all.
  Dims d1;  // conv_taps 0
  const ScratchCkpt ck1(DraftEntries(d1));
  const Qwen3DFlashWeights w1 = LoadDraft(ck1, d1, DraftConfig(d1));
  CHECK(w1.candidate_selector.Empty());
}

TEST_CASE("dflash2 config: a draft declaring the conv keys and NO selector is refused") {
  // Both mechanisms or neither. A checkpoint that declared only the conv would
  // otherwise load with a selector-shaped hole and be discovered at the first
  // propose.
  //
  // THE MESSAGE IS ASSERTED, not just the throw. A bare CHECK_THROWS here passed
  // with the guard REMOVED -- found by the mutation pass, not by reading -- because
  // an absent `selector_rank` leaves `rank` at 0 and the hidden_projection shape
  // check then throws for an unrelated reason. A gate that cannot tell the
  // refusal it means from an incidental one measures nothing.
  Dims dm = Dflash2Dims(false);
  const ScratchCkpt ck(DraftEntries(dm));
  HfConfig c = DraftConfig(dm);
  c.raw["dflash_config"].erase("selector_rank");
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(ck.shard()));
  std::string what;
  try {
    (void)vllm::LoadQwen3DFlash(shards, c, dm.taps_fc, /*mask_token_id=*/7);
    FAIL("expected a refusal for conv keys without selector keys");
  } catch (const std::exception& e) {
    what = e.what();
  }
  INFO("what: ", what);
  CHECK(what.find("selector_rank") != std::string::npos);
  CHECK(what.find("both mechanisms") != std::string::npos);
}

TEST_CASE("dflash2 weights: a MIS-SHAPED selector tensor is refused at load") {
  // The shape assertions are the only thing separating a correct load from a
  // transposed one, and nothing exercised them until the mutation pass removed
  // them and every suite stayed green. On the published 27B the codebooks are
  // [248320, 256] and the projection [256, 5120], so a transposed codebook is a
  // 254 MB tensor that still loads and still produces finite scores.
  Dims dm = Dflash2Dims(false);
  std::vector<StEntry> entries = DraftEntries(dm);
  for (StEntry& e : entries) {
    if (e.name == "candidate_selector.predecessor_codebook") {
      std::swap(e.shape[0], e.shape[1]);  // [rank, vocab] instead of [vocab, rank]
    }
  }
  const ScratchCkpt ck(entries);
  const HfConfig c = DraftConfig(dm);
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(ck.shard()));
  std::string what;
  try {
    (void)vllm::LoadQwen3DFlash(shards, c, dm.taps_fc, /*mask_token_id=*/7);
    FAIL("expected a refusal for a transposed codebook");
  } catch (const std::exception& e) {
    what = e.what();
  }
  INFO("what: ", what);
  CHECK(what.find("codebook must be") != std::string::npos);
}

TEST_CASE("dflash2 selector: Muse Glimmer's OUTPUT SCALARS are read and applied") {
  // `## Risks/decisions` D9. `z-lab/Qwen3.8-27B-DFlash2` sets NEITHER scalar, so
  // a port that reads them with a default passes every gate built from that
  // draft alone -- such a gate measures the default path and reports it as
  // coverage. `z-lab/Muse-Glimmer-30B-DFlash2` sets output_multiplier
  // 0.19611613513818404 and final_logit_softcapping 20.0, and BOTH are applied
  // to the candidate VALUES before the selector scores them, so a wrong value
  // reweights the lattice against the codebook term and moves acceptance without
  // raising anything.
  const Dims plain = Dflash2Dims(false);
  const Dims muse = Dflash2Dims(true);
  const ScratchCkpt ckp(DraftEntries(plain));
  const ScratchCkpt ckm(DraftEntries(muse));
  const HfConfig cp = DraftConfig(plain);
  const HfConfig cm = DraftConfig(muse);
  const Qwen3DFlashWeights wp = LoadDraft(ckp, plain, cp);
  const Qwen3DFlashWeights wm = LoadDraft(ckm, muse, cm);
  CHECK(wp.candidate_selector.output_multiplier == 1.0f);
  CHECK(wp.candidate_selector.final_logit_softcapping == 0.0f);
  CHECK(wm.candidate_selector.output_multiplier == doctest::Approx(0.19611613513818404));
  CHECK(wm.candidate_selector.final_logit_softcapping == 20.0f);

  // The same logits through both. `ComputeCandidates` must return the SAME ids
  // (a positive multiplier and tanh are both monotone, so the top-K SET does not
  // move) and DIFFERENT values.
  const int64_t rows = 4;
  const std::vector<float> logits = Ramp(rows, plain.vocab, 0.9);
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const vllm::Dflash2CandidateSet a =
      vllm::Qwen3DFlash2Model::ComputeCandidates(logits, rows, plain.vocab, wp, q);
  const vllm::Dflash2CandidateSet b =
      vllm::Qwen3DFlash2Model::ComputeCandidates(logits, rows, muse.vocab, wm, q);
  REQUIRE(a.ids.size() == b.ids.size());
  CHECK(a.ids == b.ids);
  bool any_value_moved = false;
  for (size_t i = 0; i < a.values.size(); ++i) {
    // Upstream's order, and it is load-bearing: multiply FIRST, then cap the
    // SCALED value. Capping first would cap a differently-scaled number.
    const float want = static_cast<float>(
        std::tanh(static_cast<double>(a.values[i]) * 0.19611613513818404 / 20.0) * 20.0);
    INFO("slot ", i, " default ", a.values[i], " muse ", b.values[i]);
    CHECK(b.values[i] == doctest::Approx(want).epsilon(1e-6));
    any_value_moved = any_value_moved || b.values[i] != a.values[i];
  }
  CHECK(any_value_moved);
  // The default arm is the RAW top-k logit, unscaled and uncapped -- which is
  // what makes "the scalars are applied" separable from "some scalar is applied".
  for (size_t i = 0; i < a.values.size(); ++i) CHECK(std::isfinite(a.values[i]));
  CHECK(a.values != b.values);
}

TEST_CASE("dflash2 selector: the SCALARS reweight the lattice against the codebook term") {
  // The acceptance-moving half of D9, made executable. `output_multiplier` and
  // the softcap scale the UNARY term only; the codebook contraction is
  // untouched. So the two arms do not differ by a monotone rescale of the edge
  // scores -- the ORDER of the children under some predecessor changes, which is
  // precisely what moves which path the W4 walk takes.
  const Dims plain = Dflash2Dims(false);
  const Dims muse = Dflash2Dims(true);
  const ScratchCkpt ckp(DraftEntries(plain));
  const ScratchCkpt ckm(DraftEntries(muse));
  const Qwen3DFlashWeights wp = LoadDraft(ckp, plain, DraftConfig(plain));
  const Qwen3DFlashWeights wm = LoadDraft(ckm, muse, DraftConfig(muse));
  const HfConfig cp = DraftConfig(plain);
  const HfConfig cm = DraftConfig(muse);

  const int P = 1, k = 3;
  const int64_t nq = k + 1;
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  const auto count_flips = [](const vllm::v1::Dflash2ProposeState& x,
                              const vllm::v1::Dflash2ProposeState& y) {
    const int64_t K = x.top_k;
    int flips = 0;
    for (int64_t l = 0; l < x.num_steps; ++l) {
      for (int64_t p = 0; p < K; ++p) {
        const size_t base = static_cast<size_t>((l * K + p) * K);
        int64_t ax = 0, ay = 0;
        for (int64_t c = 1; c < K; ++c) {
          if (x.edge_scores[base + static_cast<size_t>(c)] >
              x.edge_scores[base + static_cast<size_t>(ax)])
            ax = c;
          if (y.edge_scores[base + static_cast<size_t>(c)] >
              y.edge_scores[base + static_cast<size_t>(ay)])
            ay = c;
        }
        if (ax != ay) ++flips;
      }
    }
    return flips;
  };
  // SWEEP, and the reason it is a sweep. A single block gives `num_steps * K` =
  // 3 * 3 = 9 predecessor slots to compare, and W3's fresh review measured that
  // exactly ONE of those nine flips. `> 0` on a margin of 1-of-9 is non-vacuous
  // but thin: it is one arithmetic accident away from a gate that passes for the
  // wrong reason, and a reader has no way to see how thin it is. So the same
  // comparison runs over several blocks -- different anchors, different logit and
  // hidden ramps -- and the assertion is on the TOTAL, with the per-block counts
  // logged. The floor is on the aggregate rather than on any one block, because
  // which block flips is a property of the synthetic ramp and not of the port.
  struct Block {
    int32_t anchor;
    double logit_seed, hidden_seed;
  };
  const Block blocks[] = {{2, 0.9, 1.7}, {0, 0.31, 2.9}, {5, 1.55, 0.44},
                          {7, 2.2, 3.1}, {3, 0.05, 1.05}};
  int total_flips = 0, total_slots = 0, blocks_that_flipped = 0;
  for (const Block& blk : blocks) {
    const std::vector<float> bl = Ramp(P * nq, plain.vocab, blk.logit_seed);
    const std::vector<float> bh = Ramp(P * nq, plain.H, blk.hidden_seed);
    const std::vector<int32_t> anchors = {blk.anchor};
    const vllm::v1::Dflash2ProposeState sp = vllm::v1::Dflash2SelectCandidates(
        bl, bh, anchors, P, k, wp, cp, q);
    const vllm::v1::Dflash2ProposeState sm = vllm::v1::Dflash2SelectCandidates(
        bl, bh, anchors, P, k, wm, cm, q);
    REQUIRE(sp.edge_scores.size() == sm.edge_scores.size());
    // The SCORES always move; whether the ORDER moves is the question below.
    CHECK(sp.edge_scores != sm.edge_scores);
    const int flips = count_flips(sp, sm);
    // The per-block INFO stays live for the precondition CHECKs below, which are
    // in this same scope. It used to be followed by `CHECK(flips >= 0)`, which
    // asserted nothing an `int` counter could ever violate and existed only to
    // give the INFO something to attach to; the assertion that carries the block
    // is `count_flips(sp, sp) == 0` below, and the ones that carry the sweep are
    // the aggregate floors after the loop.
    INFO("anchor ", blk.anchor, " argmax flips ", flips, " of ",
         sp.num_steps * sp.top_k);
    total_flips += flips;
    total_slots += static_cast<int>(sp.num_steps * sp.top_k);
    if (flips > 0) ++blocks_that_flipped;
    // THE INSTRUMENT'S OWN PRECONDITION, per block. The counter has to be able
    // to read zero, or a floor on it measures nothing: comparing an arm with
    // ITSELF must flip nothing at all.
    CHECK(count_flips(sp, sp) == 0);
    CHECK(count_flips(sm, sm) == 0);
  }
  INFO("total argmax flips ", total_flips, " over ", total_slots, " slots in ",
       blocks_that_flipped, " of 5 blocks");
  // MEASURED 2026-08-20 on this fixture: 8 flips over 45 slots, in 4 of the 5
  // blocks -- against 1 of 9 in one block before the sweep. The slot count is
  // pinned exactly, because it is a shape and a change to it means the sweep
  // stopped measuring what this comment says. The flip floors are stated as
  // floors well under the measured 8 and 4, so a rounding-level change does not
  // red the suite for a reason that is not a defect, while a port that stopped
  // reordering children altogether still cannot pass.
  CHECK(total_slots == 45);
  CHECK(total_flips >= 3);
  CHECK(blocks_that_flipped >= 2);
}

TEST_CASE("dflash2 selector: the org-vocab shard indices rebase the candidates") {
  // Upstream reads both off `lm_head.shard_indices`: `num_org_vocab_padding`
  // forces that many trailing columns to -inf BEFORE the top-k, and
  // `org_vocab_start_index` is added to every surviving id AFTER it, rebasing a
  // shard's column space into the full vocabulary. BOTH are 0 on every path this
  // engine ships -- the DFlash lane's lm_head is the raw unpadded checkpoint
  // tensor and there is no vocab-parallel sharding -- so they are gated
  // SYNTHETICALLY here rather than claimed as checkpoint coverage, which is the
  // posture `## Upstream chain` already records for the output scalars. An id
  // that is not rebased indexes the wrong codebook row and moves acceptance
  // without raising anything, so the arithmetic has to be right the day a
  // sharded head arrives.
  const Dims dm = Dflash2Dims(false);
  const ScratchCkpt ck(DraftEntries(dm));
  const Qwen3DFlashWeights w = LoadDraft(ck, dm, DraftConfig(dm));
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const int64_t rows = 3;
  const std::vector<float> logits = Ramp(rows, dm.vocab, 0.9);

  const vllm::Dflash2CandidateSet base =
      vllm::Qwen3DFlash2Model::ComputeCandidates(logits, rows, dm.vocab, w, q);
  vllm::Dflash2CandidateArgs shifted;
  shifted.org_vocab_start_index = 5;
  const vllm::Dflash2CandidateSet moved = vllm::Qwen3DFlash2Model::ComputeCandidates(
      logits, rows, dm.vocab, w, q, shifted);
  REQUIRE(base.ids.size() == moved.ids.size());
  for (size_t i = 0; i < base.ids.size(); ++i) {
    INFO("slot ", i);
    CHECK(moved.ids[i] == base.ids[i] + 5);
  }
  // The VALUES are untouched by a rebase, which is what separates "the offset is
  // added to the ids" from "the offset moved the search".
  CHECK(moved.values == base.values);

  // And the padding: the columns it excludes must actually change the answer, or
  // the parameter is being exercised without being tested. `dm.vocab` is 8, so
  // padding 4 leaves columns 0..3.
  vllm::Dflash2CandidateArgs padded;
  padded.num_org_vocab_padding = 4;
  const vllm::Dflash2CandidateSet capped = vllm::Qwen3DFlash2Model::ComputeCandidates(
      logits, rows, dm.vocab, w, q, padded);
  for (int64_t id : capped.ids) CHECK(id < dm.vocab - 4);
  CHECK(capped.ids != base.ids);
}

TEST_CASE("dflash2 selector: the SAMPLE rows are +1..+k, never the anchor row") {
  // Upstream gathers `last_hidden_states[sample_indices]` -- the k MASK
  // positions. Row +0 carries a verified token and predicts nothing. An
  // off-by-one that started at the anchor row would still produce k steps of
  // finite scores, and no token gate could see it.
  //
  // The probe is a block whose rows have DISJOINT argmax columns, so the
  // candidate id of each step names the row it came from.
  const Dims dm = Dflash2Dims(false);
  const ScratchCkpt ck(DraftEntries(dm));
  const Qwen3DFlashWeights w = LoadDraft(ck, dm, DraftConfig(dm));
  const HfConfig c = DraftConfig(dm);
  const int P = 1, k = 3;
  const int64_t nq = k + 1, V = dm.vocab;
  std::vector<float> block_logits(static_cast<size_t>(P * nq * V), 0.0f);
  for (int64_t r = 0; r < nq; ++r)
    block_logits[static_cast<size_t>(r * V + r)] = 10.0f;  // row r peaks at column r
  const std::vector<float> block_hidden = Ramp(P * nq, dm.H, 1.7);
  const std::vector<int32_t> anchors = {5};
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const vllm::v1::Dflash2ProposeState st = vllm::v1::Dflash2SelectCandidates(
      block_logits, block_hidden, anchors, P, k, w, c, q);
  REQUIRE(st.num_steps == k);
  const int64_t K = st.top_k;
  for (int64_t j = 0; j < k; ++j) {
    INFO("step ", j);
    // Step j's TOP candidate is column 1+j: the mask row, not the anchor row.
    CHECK(st.candidates.ids[static_cast<size_t>(j * K)] == 1 + j);
  }
  // And column 0 -- the anchor row's peak -- is never a step's top candidate.
  for (int64_t j = 0; j < k; ++j)
    CHECK(st.candidates.ids[static_cast<size_t>(j * K)] != 0);
}

TEST_CASE("dflash2 selector: a DEQUANTIZED target lm_head is REFUSED by name") {
  // Upstream's LM-head guard, in the WIDE form vllm#52883 landed. The selector's
  // whole input is the target head's EXACT top-K, and a GGUF target's
  // `output.weight` reaches this lane DEQUANTIZED to bf16
  // (`LoadGgufSharedEmbedAndHeadBf16`), which is a different candidate set with
  // no visible symptom.
  const Dims dm = Dflash2Dims(false);
  const ScratchCkpt ck(DraftEntries(dm));
  Qwen3DFlashWeights w = LoadDraft(ck, dm, DraftConfig(dm));
  CHECK_FALSE(w.lm_head_dequantized);
  CHECK_NOTHROW(vllm::RefuseQuantizedDflash2LmHead(w));

  w.lm_head_dequantized = true;
  std::string what;
  try {
    vllm::RefuseQuantizedDflash2LmHead(w);
    FAIL("expected a refusal for a dequantized target lm_head");
  } catch (const std::exception& e) {
    what = e.what();
  }
  INFO("what: ", what);
  // SPEC-DFLASH2-QUANT-LMHEAD (#1628) narrowed WHAT this refuses, so the message
  // has to name the state rather than the dtype. It used to quote upstream's own
  // sentence, "DFlash2 requires an unquantized target LM head for candidate
  // TopK", which was live at vllm-project/vllm#52816 head `66e5414c`; that guard
  // is GONE at the MERGED head `b389ac29`, where `compute_candidates` carries no
  // quant-method check at all. Quoting a refusal upstream no longer makes would
  // be a claim about the oracle that reading it disproves, so the assertion moved
  // onto the three things that are true and discriminating: the state refused,
  // the state ADMITTED, and the one container that still produces the refused
  // one.
  CHECK(what.find("DEQUANTIZED") != std::string::npos);
  CHECK(what.find("KEPT PACKED") != std::string::npos);
  CHECK(what.find("LoadGgufSharedEmbedAndHeadBf16") != std::string::npos);
  CHECK(what.find("UnquantizedLinearMethod") != std::string::npos);
  CHECK(what.find("acceptance falls") != std::string::npos);

  // The guard is on the DFlash2 path only: a DFlash1 draft off a GGUF target is
  // unchanged, and this lane has shipped that combination since SPEC-DFLASH-GGUF.
  Dims d1;  // conv_taps 0
  const ScratchCkpt ck1(DraftEntries(d1));
  Qwen3DFlashWeights w1 = LoadDraft(ck1, d1, DraftConfig(d1));
  w1.lm_head_dequantized = true;
  CHECK_NOTHROW(vllm::RefuseQuantizedDflash2LmHead(w1));

  // And it fires from the production entry, not only when called directly.
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const std::vector<float> logits = Ramp(2, dm.vocab, 0.9);
  CHECK_THROWS(vllm::Qwen3DFlash2Model::ComputeCandidates(logits, 2, dm.vocab, w, q));
}

TEST_CASE("dflash2 selector: input_embedding_scale is REFUSED BY NAME") {
  // The third scalar. Upstream applies it inside `embed_input_ids`; NEITHER
  // published DFlash2 draft declares it, so implementing it would land three
  // unreachable call sites and ignoring it would run a quietly different model
  // on the first checkpoint that sets it. The polarity is W2's for
  // `attention_sink_bias`: upstream's own default (1.0) is a no-op and is not
  // refused. Spec `## Owed` O9.
  Dims dm = Dflash2Dims(false);
  const ScratchCkpt ck(DraftEntries(dm));
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(ck.shard()));

  HfConfig ok = DraftConfig(dm);
  ok.raw["dflash_config"]["input_embedding_scale"] = 1.0;
  CHECK_NOTHROW(vllm::LoadQwen3DFlash(shards, ok, dm.taps_fc, /*mask_token_id=*/7));

  HfConfig bad = DraftConfig(dm);
  bad.raw["dflash_config"]["input_embedding_scale"] = 0.5;
  std::string what;
  try {
    (void)vllm::LoadQwen3DFlash(shards, bad, dm.taps_fc, /*mask_token_id=*/7);
    FAIL("expected a refusal for a declared input_embedding_scale != 1.0");
  } catch (const std::exception& e) {
    what = e.what();
  }
  INFO("what: ", what);
  CHECK(what.find("input_embedding_scale") != std::string::npos);
  CHECK(what.find("O9") != std::string::npos);
}

TEST_CASE("dflash2 selector: a GGUF target's block-quant lm_head is kept PACKED (W3 contract)") {
  // QWEN38-DFLASH2-STAGE-C W3 (cbe58628b) flipped this contract: a GGUF
  // target's block-quantized `output.weight` is NO LONGER dequantized to a
  // bf16 copy — it is kept PACKED in the exact owner the target's own logits
  // GEMM reads (OwnGgufQuantBlocks, dtype = the block dtype, nk = true), which
  // also removes the ~2.44 GB duplicate device residency that OOM'd the 27B
  // k-quant arms, and lets the selector's top-K come off the target's exact
  // quantized-head logits (#1628's ADMITTED state). `head_was_quantized` is
  // false in BOTH arms now (packed head = not dequantized; scalar head was
  // never block-quantized); the D12 refusal survives only for the safetensors
  // NVFP4 path, where the head cannot be computed natively.
  constexpr int64_t V = 32, H = 32;  // Q8_0 blocks are 32 elements / 34 bytes
  const std::string bf16_rows(static_cast<size_t>(V * H) * 2, '\x3c');
  const auto build = [&](uint32_t head_type, const std::string& head_data) {
    gguf_test::GgufModelBuilder b;
    b.AddKv(gguf_test::StrKv("general.architecture", "qwen3"));
    b.AddTensor("token_embd.weight", {static_cast<uint64_t>(H), static_cast<uint64_t>(V)},
                /*BF16=*/30, bf16_rows);
    b.AddTensor("output.weight", {static_cast<uint64_t>(H), static_cast<uint64_t>(V)},
                head_type, head_data);
    return b.Build();
  };

  {  // BF16 head: read as dense bf16, nothing dequantized, flag false.
    const gguf_test::TempFile f(build(30, bf16_rows));
    const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
    vllm::OwnedTensor embed, head;
    bool quantized = true;  // deliberately the WRONG default, so a loader that
                            // never writes the flag cannot pass this arm
    vllm::LoadGgufSharedEmbedAndHeadBf16(g, &embed, &head, &quantized);
    CHECK_FALSE(quantized);
    CHECK(head.dtype == vt::DType::kBF16);
  }
  {  // Q8_0 head: 34 bytes per 32-element block, kept PACKED — not widened —
     // and the flag reports false because nothing was dequantized.
    const int64_t blocks = V * H / 32;
    std::string q8(static_cast<size_t>(blocks) * 34, '\0');
    for (int64_t i = 0; i < blocks; ++i) {
      q8[static_cast<size_t>(i * 34)] = '\x00';      // scale, fp16 low byte
      q8[static_cast<size_t>(i * 34 + 1)] = '\x3c';  // fp16 1.0
    }
    const gguf_test::TempFile f(build(/*Q8_0=*/8, q8));
    const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
    vllm::OwnedTensor embed, head;
    bool quantized = true;  // deliberately the WRONG default (see BF16 arm)
    vllm::LoadGgufSharedEmbedAndHeadBf16(g, &embed, &head, &quantized);
    CHECK_FALSE(quantized);
    CHECK(head.dtype == vt::DType::kQ8_0);
    CHECK(head.nk);
    CHECK(head.rank == 2);
    CHECK(head.shape[0] == V);
    CHECK(head.shape[1] == H);
  }
}

TEST_CASE("dflash2 shared lm_head: a NVFP4 target head is loaded PACKED, never widened") {
  const ScopedLmHeadFp4 fp4_on(true);
  // SPEC-DFLASH2-QUANT-LMHEAD (#1628). Before this row the shared-head read was
  // a single `LoadNamedBf16("lm_head.weight")` and refused on the STORED DTYPE:
  //   vllm_engine_load: dflash: target tensor lm_head.weight is not BF16 (got U8)
  // which is the message #1628 reports from `dgx:gpu0`. The predicate cannot
  // separate a head DEQUANTIZED into something the target does not compute with
  // -- the case `## Risks/decisions` D12 refuses -- from a head kept PACKED and
  // computed with natively, which is what upstream does at the MERGED
  // vllm-project/vllm#52816 head `b389ac29`: `compute_candidates` carries no
  // quant-method check at all and goes through `LogitsProcessor._apply_head` ->
  // `lm_head.quant_method.apply` (logits_processor.py:132-142,241-286).
  //
  // The MEMORY FORMAT is asserted, not just the success: a load that widened the
  // head to bf16 would satisfy every value check below and move 4x the bytes,
  // which AGENTS.md `## vLLM is the reference` names as the defect a token gate
  // cannot see.
  const int64_t V = 16, H = 16;
  const Fp4Head head = MakeNvfp4LmHead(V, H, /*variant=*/0);
  const ScratchCkpt target(head.entries);
  const std::vector<vllm::SafetensorsFile> shards = OpenShards(target);

  vllm::OwnedTensor bf16;
  vllm::Nvfp4Weight packed;
  vllm::Exl3Weight trellis;  // #2495 item 6: the third owner, always offered
  vllm::LoadDflashSharedLmHead(shards, &bf16, &packed, &trellis);

  CHECK(bf16.Empty());  // exactly ONE owner, as the target's own head has
  REQUIRE_FALSE(packed.Empty());
  CHECK(packed.n == V);
  CHECK(packed.k == H);
  CHECK(packed.scale2 == doctest::Approx(0.25f));
  CHECK_FALSE(packed.IsTrueW4A4());
  // K*N/2 packed bytes + K*N/16 scale bytes, NOT 2*K*N.
  CHECK(packed.packed.bytes.size() == static_cast<size_t>(V * H / 2));
  CHECK(packed.scale.bytes.size() == static_cast<size_t>(V * H / kNvfp4Group));

  // And the values are the fixture's, exactly -- so the group scale and the
  // global scale both survived the load. A dequant that dropped either is a
  // different number here, not a rounding difference.
  const std::vector<uint16_t> b_layout =
      vllm::dense_nvfp4::DequantNvfp4ToBLayout(packed);
  REQUIRE(b_layout.size() == static_cast<size_t>(V * H));
  int mismatches = 0;
  for (int64_t o = 0; o < V; ++o) {
    for (int64_t i = 0; i < H; ++i) {
      // DequantNvfp4ToBLayout returns [K=in, N=out].
      const float got = vt::BF16ToF32(b_layout[static_cast<size_t>(i * V + o)]);
      if (got != head.values[static_cast<size_t>(o * H + i)]) ++mismatches;
    }
  }
  CHECK(mismatches == 0);
}

TEST_CASE("dflash2 shared lm_head: the BF16 arm is byte-unchanged (the DFlash1 lane)") {
  // The polarity that must NOT move. DFlash1 has no selector and has shipped the
  // shared-bf16-head combination since SPEC-DFLASH-GGUF, so the bf16 arm has to
  // produce the same OwnedTensor it produced before the routing decision was
  // added: raw-NK `[vocab, H]`, `nk = true`, and the shard's bytes verbatim.
  const int64_t V = 16, H = 16;
  const std::vector<StEntry> entries = Bf16LmHeadEntries(V, H, /*seed=*/0.5);
  const ScratchCkpt target(entries);
  const std::vector<vllm::SafetensorsFile> shards = OpenShards(target);

  vllm::OwnedTensor bf16;
  vllm::Nvfp4Weight packed;
  vllm::Exl3Weight trellis;  // #2495 item 6: the third owner, always offered
  vllm::LoadDflashSharedLmHead(shards, &bf16, &packed, &trellis);

  CHECK(packed.Empty());
  REQUIRE_FALSE(bf16.Empty());
  CHECK(bf16.nk);
  CHECK(bf16.rank == 2);
  CHECK(bf16.shape[0] == V);
  CHECK(bf16.shape[1] == H);
  REQUIRE(bf16.bytes.size() == entries[0].bf16.size() * sizeof(uint16_t));
  CHECK(std::memcmp(bf16.bytes.data(), entries[0].bf16.data(),
                    bf16.bytes.size()) == 0);

  // A lane that CANNOT hold a packed head (DSpark passes `head_fp4 = nullptr`)
  // still gets the bf16 arm unchanged.
  vllm::OwnedTensor bf16_only;
  vllm::LoadDflashSharedLmHead(shards, &bf16_only, /*head_fp4=*/nullptr,
                               /*head_exl3=*/nullptr);
  REQUIRE(bf16_only.bytes.size() == bf16.bytes.size());
  CHECK(std::memcmp(bf16_only.bytes.data(), bf16.bytes.data(),
                    bf16.bytes.size()) == 0);
}

// ===========================================================================
// SPEC-DFLASH2 W9 (#1849) — the draft's SHARED bf16 tensors BORROW the target's
// file mapping instead of copying it.
//
// On a real target the head and the embedding table are each ~2.54 GB
// (248320x5120 bf16), and through W8 the draft `memcpy`d BOTH into anonymous
// buffers whose content is byte-identical to the mapping the TARGET's own
// loader already borrows (`BorrowStTensorBytes`, ENG-LOAD-DIRECT-UPLOAD). On
// the r0b0tlab arm the packed head already borrows and the EMBED copy is the
// live one. These cases pin the borrow at the exact functions production
// calls, on both arms of the existing fail-closed seam.

namespace {

// Both arms of the direct-upload decision in one binary, restored on scope
// exit (the seam is process-cached behind `LoadDirectUploadEnabled`).
struct ScopedDirectUpload {
  explicit ScopedDirectUpload(bool on) {
    vllm::detail::SetLoadDirectUploadOverrideForTesting(on);
  }
  ~ScopedDirectUpload() {
    vllm::detail::SetLoadDirectUploadOverrideForTesting(std::nullopt);
  }
  ScopedDirectUpload(const ScopedDirectUpload&) = delete;
  ScopedDirectUpload& operator=(const ScopedDirectUpload&) = delete;
};

// Whether `o.bytes` is a VIEW INSIDE the mapped span of `t` (not merely equal
// content). Pointer identity is the property under test: a copy that happens
// to be byte-equal must fail this.
bool AliasesMapping(const vllm::OwnedTensor& o, const vllm::StTensor& t) {
  return static_cast<const void*>(o.bytes.data()) ==
         static_cast<const void*>(t.data);
}

}  // namespace

TEST_CASE("dflash2 shared lm_head (W9): the bf16 arm BORROWS the shard mapping") {
  const int64_t V = 16, H = 16;
  const std::vector<StEntry> entries = Bf16LmHeadEntries(V, H, /*seed=*/0.5);
  const ScratchCkpt target(entries);
  const std::vector<vllm::SafetensorsFile> shards = OpenShards(target);
  const vllm::StTensor& src = shards[0].Get("lm_head.weight");

  {  // The default arm: the head VIEWS the mapping, no owned buffer exists.
    const ScopedDirectUpload on(true);
    vllm::OwnedTensor bf16;
    vllm::Nvfp4Weight packed;
    vllm::Exl3Weight trellis;  // #2495 item 6: the third owner, always offered
    vllm::LoadDflashSharedLmHead(shards, &bf16, &packed, &trellis);
    CHECK(packed.Empty());
    REQUIRE_FALSE(bf16.Empty());
    CHECK(bf16.nk);
    CHECK(bf16.rank == 2);
    CHECK(bf16.shape[0] == V);
    CHECK(bf16.shape[1] == H);
    REQUIRE(bf16.bytes.size() == src.nbytes);
    CHECK(AliasesMapping(bf16, src));
    CHECK(std::memcmp(bf16.bytes.data(), src.data, src.nbytes) == 0);
  }
  {  // The rollback arm (`VT_LOAD_DIRECT_UPLOAD=0`): the copy, byte-equal.
    const ScopedDirectUpload off(false);
    vllm::OwnedTensor bf16;
    vllm::Nvfp4Weight packed;
    vllm::Exl3Weight trellis;  // #2495 item 6: the third owner, always offered
    vllm::LoadDflashSharedLmHead(shards, &bf16, &packed, &trellis);
    REQUIRE_FALSE(bf16.Empty());
    CHECK(bf16.nk);
    CHECK_FALSE(AliasesMapping(bf16, src));
    REQUIRE(bf16.bytes.size() == src.nbytes);
    CHECK(std::memcmp(bf16.bytes.data(), src.data, src.nbytes) == 0);
  }
}

TEST_CASE("dflash2 shared embed (W9): LoadDflashSharedEmbedBf16 borrows, falls back, refuses") {
  const int64_t V = 16, H = 16;
  const std::string name = "model.language_model.embed_tokens.weight";
  const std::vector<StEntry> entries = {StEntry{name, {V, H}, Fill(V * H, 3.7, 0.4)}};
  const ScratchCkpt target(entries);
  const std::vector<vllm::SafetensorsFile> shards = OpenShards(target);
  const vllm::StTensor& src = shards[0].Get(name);

  {  // The default arm: the gather table VIEWS the mapping.
    const ScopedDirectUpload on(true);
    const vllm::OwnedTensor embed = vllm::LoadDflashSharedEmbedBf16(shards, name);
    REQUIRE_FALSE(embed.Empty());
    CHECK_FALSE(embed.nk);  // [vocab, H] gather-table orientation, as before
    CHECK(embed.rank == 2);
    CHECK(embed.shape[0] == V);
    CHECK(embed.shape[1] == H);
    REQUIRE(embed.bytes.size() == src.nbytes);
    CHECK(AliasesMapping(embed, src));
    CHECK(std::memcmp(embed.bytes.data(), src.data, src.nbytes) == 0);
  }
  {  // The rollback arm: the copy, byte-equal, not aliasing.
    const ScopedDirectUpload off(false);
    const vllm::OwnedTensor embed = vllm::LoadDflashSharedEmbedBf16(shards, name);
    REQUIRE_FALSE(embed.Empty());
    CHECK_FALSE(AliasesMapping(embed, src));
    REQUIRE(embed.bytes.size() == src.nbytes);
    CHECK(std::memcmp(embed.bytes.data(), src.data, src.nbytes) == 0);
  }
  {  // An absent tensor stays the caller's EMPTY, exactly as the read it
     // replaces (`LoadNamedBf16`) answered; `SharedHeadSource::LoadInto` owns
     // the throw that names the source.
    const vllm::OwnedTensor none =
        vllm::LoadDflashSharedEmbedBf16(shards, "model.embed_tokens.weight");
    CHECK(none.Empty());
  }
  {  // A non-BF16 table is refused with the same text the old read threw.
    std::vector<uint8_t> raw(static_cast<size_t>(V * H) * sizeof(float), 0);
    const std::vector<StEntry> f32 = {StEntry{name, {V, H}, "F32", std::move(raw)}};
    const ScratchCkpt bad(f32);
    const std::vector<vllm::SafetensorsFile> bad_shards = OpenShards(bad);
    std::string what;
    try {
      (void)vllm::LoadDflashSharedEmbedBf16(bad_shards, name);
    } catch (const std::exception& e) {
      what = e.what();
    }
    CHECK(what.find("is not BF16 (got F32)") != std::string::npos);
  }
}

TEST_CASE("dflash2 shared lm_head: a DEQUANTIZED-ONLY storage form is STILL refused") {
  // D12's argument survives this row and is what draws the line. An FP8 head has
  // no native arm here: `LoadLmHeadAnyDtype` would WIDEN it to bf16, which is
  // exactly "a head read through a dequantization" and produces a different
  // candidate set with no visible symptom. So it keeps refusing, by name, at
  // startup -- and the refusal now means "this storage form cannot be computed
  // with", not "this head is quantized".
  const int64_t V = 16, H = 16;
  std::vector<StEntry> entries;
  StEntry w;
  w.name = "lm_head.weight";
  w.shape = {V, H};
  w.dtype = "F8_E4M3";
  w.raw.assign(static_cast<size_t>(V * H), 0x38);  // fp8-e4m3 1.0
  entries.push_back(std::move(w));
  StEntry ws;
  ws.name = "lm_head.weight_scale";
  ws.shape = {V, 1};
  ws.dtype = "F32";
  ws.raw.assign(static_cast<size_t>(V) * sizeof(float), 0);
  entries.push_back(std::move(ws));
  const ScratchCkpt target(entries);
  const std::vector<vllm::SafetensorsFile> shards = OpenShards(target);

  vllm::OwnedTensor bf16;
  vllm::Nvfp4Weight packed;
  vllm::Exl3Weight trellis;  // #2495 item 6: the third owner, always offered
  std::string what;
  try {
    vllm::LoadDflashSharedLmHead(shards, &bf16, &packed, &trellis);
    FAIL("expected a refusal for an FP8 target lm_head");
  } catch (const std::exception& e) {
    what = e.what();
  }
  INFO("what: ", what);
  CHECK(what.find("lm_head.weight") != std::string::npos);
  CHECK(what.find("F8_E4M3") != std::string::npos);
}

TEST_CASE("dflash2 selector: a PACKED target lm_head gives the TARGET'S OWN top-K") {
  const ScopedLmHeadFp4 fp4_on(true);
  // THE PROPERTY THIS ROW IS GATED ON, and the one a "it loaded" case would
  // pass while the candidate set was subtly wrong -- the silent failure D12
  // names: the verify is lossless, the emitted tokens stay the target's, and
  // only acceptance falls.
  //
  // The reference is the TARGET'S OWN shared-head logits path
  // (`Qwen3_5MTPModel::ComputeLogits`, whose body for a packed head is the one
  // line `DenseLogitsF32D` runs). The draft's block forward must produce
  // BITWISE-IDENTICAL logits over the same post-final-norm hidden states, and
  // therefore the identical top-K ids and values out of the selector.
  const Dims dm = Nvfp4Dims();
  const ScratchCkpt draft_ck(DraftEntries(dm));
  const HfConfig c = DraftConfig(dm);
  Qwen3DFlashWeights w = LoadDraft(draft_ck, dm, c);
  REQUIRE(w.IsDflash2());

  const Fp4Head head = MakeNvfp4LmHead(dm.vocab, dm.H, /*variant=*/0);
  const ScratchCkpt target(head.entries);
  // What `LoadDflashDraft` does: the draft's own `lm_head` is REPLACED by the
  // target's, and the draft vocabulary comes from whichever owner holds it.
  vllm::LoadDflashSharedLmHead(OpenShards(target), &w.lm_head, &w.lm_head_fp4,
                               &w.lm_head_exl3);
  REQUIRE_FALSE(w.lm_head_fp4.Empty());
  REQUIRE(w.lm_head.Empty());
  w.draft_vocab_size = w.lm_head_fp4.n;

  std::vector<float> hidden;
  const std::vector<float> logits =
      ForwardCapturingHidden(w, c, dm.block, &hidden);
  REQUIRE(logits.size() == static_cast<size_t>(dm.block * dm.vocab));
  REQUIRE(hidden.size() == static_cast<size_t>(dm.block * dm.H));

  const std::vector<float> reference = TargetSharedHeadLogits(
      w.lm_head_fp4, hidden, dm.block, dm.H, dm.vocab);
  CHECK(BitEqual(logits, reference));

  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const vllm::Dflash2CandidateSet mine = vllm::Qwen3DFlash2Model::ComputeCandidates(
      logits, dm.block, dm.vocab, w, q);
  const vllm::Dflash2CandidateSet theirs =
      vllm::Qwen3DFlash2Model::ComputeCandidates(reference, dm.block, dm.vocab, w, q);
  CHECK(mine.ids == theirs.ids);
  CHECK(BitEqual(mine.values, theirs.values));

  // NON-VACUITY. A gate that compared two constants would pass the lines above.
  // A DIFFERENT packed head -- same shapes, different codes and group scales --
  // must move the candidate ids, so the assertion is measuring the head.
  Qwen3DFlashWeights other = LoadDraft(draft_ck, dm, c);
  const Fp4Head head2 = MakeNvfp4LmHead(dm.vocab, dm.H, /*variant=*/1);
  const ScratchCkpt target2(head2.entries);
  vllm::LoadDflashSharedLmHead(OpenShards(target2), &other.lm_head,
                               &other.lm_head_fp4, &other.lm_head_exl3);
  other.draft_vocab_size = other.lm_head_fp4.n;
  std::vector<float> hidden2;
  const std::vector<float> logits2 =
      ForwardCapturingHidden(other, c, dm.block, &hidden2);
  const vllm::Dflash2CandidateSet moved = vllm::Qwen3DFlash2Model::ComputeCandidates(
      logits2, dm.block, dm.vocab, other, q);
  CHECK(moved.ids != mine.ids);

  // And the SHARED head is what the forward reads: the draft checkpoint ships a
  // bf16 `lm_head.weight` of its own, and a forward that kept it would answer
  // with those candidates instead.
  const Qwen3DFlashWeights draft_own = LoadDraft(draft_ck, dm, c);
  REQUIRE_FALSE(draft_own.lm_head.Empty());
  const std::vector<float> own_logits = Forward(draft_own, c, dm.block);
  const vllm::Dflash2CandidateSet own = vllm::Qwen3DFlash2Model::ComputeCandidates(
      own_logits, dm.block, dm.vocab, draft_own, q);
  CHECK(own.ids != mine.ids);
}

TEST_CASE("dflash2 selector: a PACKED head is ADMITTED and a DEQUANTIZED one is not") {
  const ScopedLmHeadFp4 fp4_on(true);
  // The two states the old predicate could not separate, asserted side by side
  // on the SAME guard. `RefuseQuantizedDflash2LmHead` reads
  // `lm_head_dequantized`, which only the GGUF shared-head loader can set, so a
  // packed head passes it and a widened one does not.
  const Dims dm = Nvfp4Dims();
  const ScratchCkpt draft_ck(DraftEntries(dm));
  const HfConfig c = DraftConfig(dm);
  Qwen3DFlashWeights w = LoadDraft(draft_ck, dm, c);
  const Fp4Head head = MakeNvfp4LmHead(dm.vocab, dm.H, /*variant=*/0);
  const ScratchCkpt target(head.entries);
  vllm::LoadDflashSharedLmHead(OpenShards(target), &w.lm_head, &w.lm_head_fp4,
                               &w.lm_head_exl3);
  w.draft_vocab_size = w.lm_head_fp4.n;

  CHECK_FALSE(w.lm_head_dequantized);
  CHECK_NOTHROW(vllm::RefuseQuantizedDflash2LmHead(w));

  Qwen3DFlashWeights widened = w;
  widened.lm_head_dequantized = true;
  CHECK_THROWS(vllm::RefuseQuantizedDflash2LmHead(widened));
}

TEST_CASE("dflash2 draft logits: a W4A4 shared head is REFUSED, not silently rerouted") {
  const ScopedLmHeadFp4 fp4_on(true);
  // The one NVFP4 spelling this draft's logits GEMM cannot take. An lm_head is
  // W4A16 under both namings -- `LoadDenseLmHead` drops the activation divisor
  // because vLLM's `ModelOptNvFp4W4A16LinearMethod` deletes it
  // (modelopt.py:1365) -- unless `VT_MODELOPT_W4A4=1` puts it back. The draft's
  // GEMM is the shared W4A16 dispatcher, so a true-W4A4 head must refuse BY NAME
  // rather than fall into a different kernel than the target's.
  const Dims dm = Nvfp4Dims();
  const ScratchCkpt draft_ck(DraftEntries(dm));
  const HfConfig c = DraftConfig(dm);
  Qwen3DFlashWeights w = LoadDraft(draft_ck, dm, c);
  const Fp4Head head = MakeNvfp4LmHead(dm.vocab, dm.H, /*variant=*/0);
  const ScratchCkpt target(head.entries);
  vllm::LoadDflashSharedLmHead(OpenShards(target), &w.lm_head, &w.lm_head_fp4,
                               &w.lm_head_exl3);
  w.draft_vocab_size = w.lm_head_fp4.n;
  REQUIRE_FALSE(w.lm_head_fp4.IsTrueW4A4());

  w.lm_head_fp4.input_global_scale_inv = 2.0f;
  w.lm_head_fp4.alpha = w.lm_head_fp4.scale2 * 0.5f;
  REQUIRE(w.lm_head_fp4.IsTrueW4A4());
  std::string what;
  try {
    (void)Forward(w, c, dm.block);
    FAIL("expected a refusal for a true-W4A4 shared lm_head");
  } catch (const std::exception& e) {
    what = e.what();
  }
  INFO("what: ", what);
  // Assert on what THIS guard uniquely says, not on "W4A4". The W4A16
  // dispatcher refuses the same weight one layer down with its own message, and
  // that message also contains "W4A4" -- so a case that greps for it passes with
  // this guard deleted, which the row's mutation pass measured. What only this
  // guard can tell a user is the variable to unset.
  CHECK(what.find("VT_MODELOPT_W4A4") != std::string::npos);
  CHECK(what.find("1628") != std::string::npos);
}

// ─── The LOADER SEAM: an NVFP4 target head through the PRODUCTION entry point ──
//
// WHY THIS SECTION EXISTS. Every case above enters at `LoadDflashSharedLmHead`
// or at the draft forward, and the row's fresh review measured what that costs:
// restoring the pre-row read at `src/vllm/entrypoints/model_loader.cpp`
// (`*head = LoadNamedBf16(*shards_, "lm_head.weight", true);`) -- the exact line
// #1628 reports as the defect -- compiled clean and left all three DFlash2
// suites GREEN. A gate nobody can turn red by putting the bug back is measuring
// a function, not a capability, which is what `AGENTS.md` "Nothing lands dead"
// and `.agents/reachability.md` forbid.
//
// So this drives `LoadedEngine::FromModelDir` -- the entry a server, the C ABI
// and the bench client all arrive through -- against a real on-disk target whose
// `lm_head.weight` is ModelOpt NVFP4, with a real on-disk DFlash2 draft named by
// `--speculative-config`. That is #1628's own reproduction: the reported failure
// was `vllm_engine_load: dflash: target tensor lm_head.weight is not BF16 (got
// U8)` on exactly this shape.
//
// WHAT IS ASSERTED, and why it is the loader's own line rather than a return
// value. `LoadDflashDraft` prints `... shared head from the target safetensors
// shards` AFTER `SharedHeadSource::LoadInto` returns, so that line is printed if
// and only if the shared-head read completed. Restoring the bf16 read throws
// before it, and the line is absent -- the mutation this file could not detect
// before now turns this case RED. The refusal text is asserted ABSENT as well,
// which alone would be the "never assert absence from a failed grep" trap; the
// PRESENT line is what carries the case.
namespace {

// The tiny BPE fixture shape `test_dflash2_runner_reach.cpp` uses: ids 0..V-1
// with no holes. `FromModelDir` builds a tokenizer before it reaches any weight,
// so a target directory without one never gets to the seam under test.
std::string TinyTokenizerJson(int64_t vocab) {
  REQUIRE(vocab >= 2);
  json v = json::object();
  // The Metaspace pre-tokenizer prepends U+2581, so it has to BE a token: a
  // vocabulary without it refuses every prompt with "no vocab token,
  // byte-fallback unavailable".
  v["\u2581"] = 0;
  for (int64_t i = 1; i < vocab; ++i)
    v[std::string(1, static_cast<char>('a' + i - 1))] = i;
  json t;
  t["version"] = "1.0";
  t["pre_tokenizer"] = {{"type", "Metaspace"},
                        {"replacement", "\u2581"},
                        {"prepend_scheme", "always"},
                        {"split", true}};
  t["decoder"] = {{"type", "Metaspace"},
                  {"replacement", "\u2581"},
                  {"prepend_scheme", "always"},
                  {"split", true}};
  t["model"] = {{"type", "BPE"},
                {"unk_token", nullptr},
                {"vocab", v},
                {"merges", json::array()}};
  t["added_tokens"] = json::array();
  return t.dump();
}

// A Qwen3.5 DENSE text target on disk, under the `model.language_model.` prefix
// `ResolveQwen3_5BackbonePrefix` selects and `SharedHeadSource::LoadInto` reads
// its embedding table from. Every layer is `full_attention`, which is the
// smallest complete dense layer this loader accepts.
struct TargetDims {
  int64_t H = 16, Hq = 2, Hkv = 1, Dh = 8, I = 6, vocab = 16, layers = 2;
  // The GDN half of the hybrid. Layer 0 is `linear_attention` and layer 1 is
  // `full_attention`, which is the shape `test_dflash2_runner_reach.cpp` runs a
  // DFlash2 propose on: an all-full-attention Qwen3.5 is not a configuration
  // this runner accepts, because it sizes a Mamba state spec either way.
  int64_t Hk = 2, Hv = 2, Dk = 8, Dv = 8, Kw = 4;
  int64_t key_dim() const { return Hk * Dk; }
  int64_t value_dim() const { return Hv * Dv; }
  int64_t conv_dim() const { return 2 * key_dim() + value_dim(); }
};

// `vt::CausalConv1dSpecUpdate` rejects a bf16 conv state off CUDA, and a DFlash2
// verify takes the GDN speculative rollback path. Set once per process, before any
// engine is built -- the same escape `test_dflash2_runner_reach.cpp` takes, and
// for the same reason.
struct F32GdnState {
  F32GdnState() { ::setenv("VT_GDN_STATE_BF16", "0", /*overwrite=*/1); }
};
const F32GdnState kSeamF32GdnState;

// An F32 tensor entry (`A_log` and `dt_bias` are read with `LoadToF32`).
StEntry F32Entry(const std::string& name, const std::vector<int64_t>& shape,
                 float value) {
  int64_t n = 1;
  for (const int64_t d : shape) n *= d;
  std::vector<uint8_t> raw(static_cast<size_t>(n) * sizeof(float));
  for (int64_t i = 0; i < n; ++i) {
    const float v = value + 0.0625f * static_cast<float>(i % 4);
    std::memcpy(raw.data() + static_cast<size_t>(i) * sizeof(float), &v,
                sizeof(float));
  }
  return StEntry(name, shape, "F32", std::move(raw));
}

std::vector<StEntry> TargetEntries(const TargetDims& t, bool nvfp4_head) {
  const std::string bb = "model.language_model.";
  const int64_t qdim = t.Hq * t.Dh, kdim = t.Hkv * t.Dh;
  std::vector<StEntry> e;
  e.push_back({bb + "embed_tokens.weight", {t.vocab, t.H},
               Fill(t.vocab * t.H, 6.1, 0.3)});
  e.push_back({bb + "norm.weight", {t.H}, Fill(t.H, 6.2, 0.5)});
  for (int64_t l = 0; l < t.layers; ++l) {
    const std::string b = bb + "layers." + std::to_string(l) + ".";
    const double s = 7.0 + static_cast<double>(l);
    e.push_back({b + "input_layernorm.weight", {t.H}, Fill(t.H, s + 0.1, 0.6)});
    e.push_back({b + "post_attention_layernorm.weight", {t.H},
                 Fill(t.H, s + 0.2, 0.6)});
    e.push_back({b + "mlp.gate_proj.weight", {t.I, t.H},
                 Fill(t.I * t.H, s + 0.9, 0.25)});
    e.push_back({b + "mlp.up_proj.weight", {t.I, t.H},
                 Fill(t.I * t.H, s + 1.1, 0.25)});
    e.push_back({b + "mlp.down_proj.weight", {t.H, t.I},
                 Fill(t.H * t.I, s + 1.2, 0.25)});
    if (l == 0) {
      const std::string la = b + "linear_attn.";
      e.push_back({la + "in_proj_qkv.weight", {t.conv_dim(), t.H},
                   Fill(t.conv_dim() * t.H, s + 2.1, 0.2)});
      e.push_back({la + "in_proj_z.weight", {t.value_dim(), t.H},
                   Fill(t.value_dim() * t.H, s + 2.2, 0.2)});
      e.push_back({la + "in_proj_b.weight", {t.Hv, t.H}, Fill(t.Hv * t.H, s + 2.3, 0.2)});
      e.push_back({la + "in_proj_a.weight", {t.Hv, t.H}, Fill(t.Hv * t.H, s + 2.4, 0.2)});
      e.push_back({la + "out_proj.weight", {t.H, t.value_dim()},
                   Fill(t.H * t.value_dim(), s + 2.5, 0.2)});
      e.push_back({la + "conv1d.weight", {t.conv_dim(), 1, t.Kw},
                   Fill(t.conv_dim() * t.Kw, s + 2.6, 0.2)});
      e.push_back(F32Entry(la + "A_log", {t.Hv}, 0.5f));
      e.push_back(F32Entry(la + "dt_bias", {t.Hv}, 0.25f));
      e.push_back({la + "norm.weight", {t.Dv}, Fill(t.Dv, s + 2.7, 0.6)});
      continue;
    }
    // Qwen3.5 dense fuses the attention GATE into q_proj: [Hq*2*Dh, H].
    e.push_back({b + "self_attn.q_proj.weight", {2 * qdim, t.H},
                 Fill(2 * qdim * t.H, s + 0.3, 0.25)});
    e.push_back({b + "self_attn.k_proj.weight", {kdim, t.H},
                 Fill(kdim * t.H, s + 0.4, 0.25)});
    e.push_back({b + "self_attn.v_proj.weight", {kdim, t.H},
                 Fill(kdim * t.H, s + 0.5, 0.25)});
    e.push_back({b + "self_attn.o_proj.weight", {t.H, qdim},
                 Fill(t.H * qdim, s + 0.6, 0.25)});
    e.push_back({b + "self_attn.q_norm.weight", {t.Dh}, Fill(t.Dh, s + 0.7, 0.7)});
    e.push_back({b + "self_attn.k_norm.weight", {t.Dh}, Fill(t.Dh, s + 0.8, 0.7)});
  }
  if (nvfp4_head) {
    const Fp4Head head = MakeNvfp4LmHead(t.vocab, t.H, /*variant=*/0);
    for (const StEntry& en : head.entries) e.push_back(en);
  } else {
    for (StEntry& en : Bf16LmHeadEntries(t.vocab, t.H, 6.3))
      e.push_back(std::move(en));
  }
  return e;
}

std::string TargetConfigJson(const TargetDims& t) {
  json c;
  c["architectures"] = json::array({"Qwen3_5ForConditionalGeneration"});
  c["model_type"] = "qwen3_5_text";
  c["hidden_size"] = t.H;
  c["num_hidden_layers"] = t.layers;
  c["vocab_size"] = t.vocab;
  c["num_attention_heads"] = t.Hq;
  c["num_key_value_heads"] = t.Hkv;
  c["head_dim"] = t.Dh;
  c["intermediate_size"] = t.I;
  std::vector<std::string> types(static_cast<size_t>(t.layers), "full_attention");
  types[0] = "linear_attention";
  c["layer_types"] = types;
  c["rope_theta"] = 10000.0;
  c["rms_norm_eps"] = 1e-6;
  // Declared even though every layer here is `full_attention`: the runner sizes
  // a Mamba state spec off these four keys before it looks at `layer_types`,
  // and zeros there refuse with "MambaSpec state shapes must be positive".
  c["linear_num_key_heads"] = t.Hk;
  c["linear_num_value_heads"] = t.Hv;
  c["linear_key_head_dim"] = t.Dk;
  c["linear_value_head_dim"] = t.Dv;
  c["linear_conv_kernel_dim"] = t.Kw;
  c["max_position_embeddings"] = 32;
  c["torch_dtype"] = "bfloat16";
  c["tie_word_embeddings"] = false;
  return c.dump();
}

// The DFlash2 draft's own `config.json`, in the published shape
// `MakeQwen3DFlashDraftConfig` parses. `target_layer_ids` sizes the draft's `fc`
// input, so it must agree with `Dims::taps_fc`.
std::string DraftConfigJson(const Dims& dm, int64_t num_taps) {
  json c;
  c["architectures"] = json::array({"DFlash2DraftModel"});
  c["model_type"] = "qwen3";
  c["hidden_size"] = dm.H;
  c["num_attention_heads"] = dm.Hq;
  c["num_key_value_heads"] = dm.Hkv;
  c["head_dim"] = dm.Dh;
  c["intermediate_size"] = dm.I;
  c["vocab_size"] = dm.vocab;
  c["num_hidden_layers"] = dm.layers;
  c["rms_norm_eps"] = 1e-6;
  c["rope_theta"] = 1e7;
  c["sliding_window"] = 2048;
  c["layer_types"] = std::vector<std::string>(static_cast<size_t>(dm.layers),
                                              std::string("sliding_attention"));
  json d;
  d["mask_token_id"] = dm.vocab - 1;
  // Strictly ascending, which the target's aux-tap capture requires (capture
  // order == concat order, qwen3_5.cpp).
  std::vector<int64_t> taps(static_cast<size_t>(num_taps));
  for (size_t i = 0; i < taps.size(); ++i) taps[i] = static_cast<int64_t>(i);
  d["target_layer_ids"] = taps;
  d["conv_kernel_size"] = dm.conv_taps;
  d["conv_group_size"] = dm.conv_group;
  d["block_size"] = dm.block;
  d["selector_rank"] = dm.sel_rank;
  d["selector_top_k"] = dm.sel_top_k;
  c["dflash_config"] = d;
  c["block_size"] = dm.block;
  c["is_causal"] = false;
  return c.dump();
}

// A scratch checkpoint DIRECTORY: one shard plus the json files a production
// load reads beside it.
class ScratchDir {
 public:
  ScratchDir(const std::vector<StEntry>& entries,
             const std::vector<std::pair<std::string, std::string>>& files) {
    static int counter = 0;
    dir_ = fs::temp_directory_path() /
           ("vllmcpp_dflash2_seam_" + std::to_string(counter++) + "_" +
            std::to_string(static_cast<long long>(::getpid())));
    fs::create_directories(dir_);
    WriteSafetensors(entries, (dir_ / "model.safetensors").string());
    for (const auto& nv : files)
      std::ofstream((dir_ / nv.first).string(), std::ios::binary) << nv.second;
  }
  ~ScratchDir() {
    std::error_code ec;
    fs::remove_all(dir_, ec);
  }
  ScratchDir(const ScratchDir&) = delete;
  ScratchDir& operator=(const ScratchDir&) = delete;
  std::string path() const { return dir_.string(); }

 private:
  fs::path dir_;
};

// REAL fd 2, by dup/dup2, and not a `std::cerr` rdbuf swap: the surrounding load
// reports its phases with `std::fprintf(stderr, ...)`, which an rdbuf swap
// cannot see. A capture that could see only part of the stream would report an
// empty string and read as "the loader never got there", which is the instrument
// failing toward a verdict about the code.
std::string CaptureRealStderr(const std::function<void()>& body) {
  // A NAMED file rather than `std::tmpfile()`, and this is the whole reason.
  //
  // While fd 2 is redirected, ANYTHING that writes to it lands here -- including
  // a sanitizer report. The `address,undefined` lane builds with
  // `-fno-sanitize-recover=all` (`CMakeLists.txt:260`), so UBSan does not
  // continue after a finding: it writes its diagnostic to fd 2 and exits 1.
  // With an anonymous temporary that console read as two lines of doctest banner,
  // no summary, and exit 1 -- an empty red that says nothing about the code, which
  // is the "broken instruments fail toward a code verdict" shape.
  //
  // This file is removed on the normal return below, so an abnormal exit LEAVES
  // it, named and findable: `ls /tmp/vllmcpp_dflash2_stderr_*`. The `strace -f -s
  // 2000 -e trace=write` route recovers the same bytes live.
  static int counter = 0;
  const fs::path cap_path =
      fs::temp_directory_path() /
      ("vllmcpp_dflash2_stderr_" + std::to_string(counter++) + "_" +
       std::to_string(static_cast<long long>(::getpid())) + ".log");
  std::FILE* cap = std::fopen(cap_path.string().c_str(), "w+b");
  REQUIRE(cap != nullptr);
  std::cerr.flush();
  std::fflush(stderr);
  const int saved = ::dup(STDERR_FILENO);
  REQUIRE(saved >= 0);
  REQUIRE(::dup2(::fileno(cap), STDERR_FILENO) >= 0);
  body();
  std::cerr.flush();
  std::fflush(stderr);
  const int restored = ::dup2(saved, STDERR_FILENO);
  ::close(saved);
  REQUIRE(restored >= 0);
  std::rewind(cap);
  std::string out;
  char buf[4096];
  size_t n = 0;
  while ((n = std::fread(buf, 1, sizeof(buf), cap)) > 0) out.append(buf, n);
  std::fclose(cap);
  std::error_code ec;
  fs::remove(cap_path, ec);
  return out;
}

struct SeamRun {
  std::string stderr_text;
  std::string threw;
  bool loaded = false;
  std::string generate_threw;
};

// One `FromModelDir` load with a DFlash2 draft attached, against a target whose
// `lm_head.weight` is NVFP4 or BF16. `with_draft=false` loads the SAME target
// with no speculative config at all — the W9 borrow-delta case subtracts that
// load's borrowed bytes so the target's own borrows cancel exactly.
SeamRun RunLoaderSeam(bool nvfp4_head, bool fp4_arm_on, bool with_draft = true) {
  const ScopedLmHeadFp4 arm(fp4_arm_on);
  const TargetDims t;
  Dims dm = Dflash2Dims(/*muse_glimmer_scalars=*/false);
  dm.H = t.H;
  dm.Hq = t.Hq;
  dm.Hkv = t.Hkv;
  dm.Dh = t.Dh;
  dm.I = t.I;
  dm.vocab = t.vocab;
  dm.conv_group = 4;
  const ScratchDir target(TargetEntries(t, nvfp4_head),
                          {{"config.json", TargetConfigJson(t)},
                           {"tokenizer.json", TinyTokenizerJson(t.vocab)}});
  const ScratchDir draft(DraftEntries(dm),
                         {{"config.json", DraftConfigJson(dm, dm.taps_fc)}});

  vllm::entrypoints::EngineParams p;
  if (with_draft) {
    p.speculative_config = vllm::ParseSpeculativeConfigJson(
        R"({"method":"dflash","num_speculative_tokens":)" +
        std::to_string(dm.block - 1) + R"(,"model":")" + draft.path() + R"("})");
  }
  SeamRun r;
  r.stderr_text = CaptureRealStderr([&] {
    try {
      std::unique_ptr<vllm::entrypoints::LoadedEngine> eng =
          vllm::entrypoints::LoadedEngine::FromModelDir(target.path(), p);
      r.loaded = eng != nullptr;
      // And then DRAFT with it. Loading proves the shared head was read;
      // proposing is what reads `draft_vocab_size`, which the loader derives
      // from the PACKED head's `n` when the bf16 owner is empty. Deleting that
      // branch leaves the draft vocabulary at 0, and only a propose can see it.
      vllm::SamplingParams sp;
      sp.temperature = 0.0;
      sp.max_tokens = 4;
      sp.output_kind = vllm::RequestOutputKind::kCumulative;
      (void)eng->engine().generate("abc", sp, "seam");
    } catch (const std::exception& e) {
      if (r.loaded) {
        r.generate_threw = e.what();
      } else {
        r.threw = e.what();
      }
    }
  });
  return r;
}

}  // namespace

TEST_CASE("dflash2 loader seam: FromModelDir READS an NVFP4 target lm_head (#1628)") {
  const SeamRun r = RunLoaderSeam(/*nvfp4_head=*/true, /*fp4_arm_on=*/true);
  INFO("threw: ", r.threw);
  INFO("stderr: ", r.stderr_text);
  // The shared head was read. This line is printed by `LoadDflashDraft` AFTER
  // `SharedHeadSource::LoadInto` returns, so restoring the pre-row bf16 read
  // removes it.
  CHECK(r.stderr_text.find("shared head from the target safetensors shards") !=
        std::string::npos);
  // And the defect #1628 reports is gone from the production path by name.
  CHECK(r.threw.find("lm_head.weight is not BF16") == std::string::npos);
  // The whole load completes: a packed target head is not a partial capability.
  CHECK(r.threw.empty());
  CHECK(r.loaded);
  // ...and the engine DRAFTS off it. This is what reads `draft_vocab_size`.
  INFO("generate threw: ", r.generate_threw);
  CHECK(r.generate_threw.empty());
}

TEST_CASE("dflash2 loader seam: the BF16 target head still loads through FromModelDir") {
  const SeamRun r = RunLoaderSeam(/*nvfp4_head=*/false, /*fp4_arm_on=*/true);
  INFO("threw: ", r.threw);
  INFO("stderr: ", r.stderr_text);
  // The DFlash1-era arm is unchanged: the same entry point, the same line.
  CHECK(r.stderr_text.find("shared head from the target safetensors shards") !=
        std::string::npos);
  CHECK(r.threw.empty());
  CHECK(r.loaded);
  INFO("generate threw: ", r.generate_threw);
  CHECK(r.generate_threw.empty());
}

TEST_CASE("dflash2 loader seam (W9): the SHARED embed+head borrow the target mapping through FromModelDir") {
  // SPEC-DFLASH2 W9 (#1849), mutation M5's gate. T1/T2 above enter at the two
  // exported functions directly, so restoring `SharedHeadSource::LoadInto`'s
  // old copy-based reads would leave them green. This case measures the
  // PRODUCTION loader: the same bf16 target loaded twice through
  // `FromModelDir`, once without any draft and once with the DFlash2 draft
  // attached. The target's own borrows are identical across the two runs and
  // cancel; the remaining delta is EXACTLY the draft's two shared tensors,
  // because nothing else on the draft side borrows (its own weights are copied
  // out of its own checkpoint).
  vllm::load_stats::Reset();
  const SeamRun without = RunLoaderSeam(/*nvfp4_head=*/false, /*fp4_arm_on=*/true,
                                        /*with_draft=*/false);
  const uint64_t borrowed_without = vllm::load_stats::Snapshot().borrowed_bytes;
  CHECK(without.threw.empty());
  CHECK(without.loaded);

  vllm::load_stats::Reset();
  const SeamRun with = RunLoaderSeam(/*nvfp4_head=*/false, /*fp4_arm_on=*/true,
                                     /*with_draft=*/true);
  const uint64_t borrowed_with = vllm::load_stats::Snapshot().borrowed_bytes;
  CHECK(with.threw.empty());
  CHECK(with.loaded);
  INFO("generate threw: ", with.generate_threw);
  CHECK(with.generate_threw.empty());

  const TargetDims t;
  const uint64_t embed_bytes = static_cast<uint64_t>(t.vocab * t.H) * 2;
  const uint64_t head_bytes = embed_bytes;  // same [vocab, H] bf16 shape
  REQUIRE(borrowed_with >= borrowed_without);
  CHECK(borrowed_with - borrowed_without == embed_bytes + head_bytes);
}

TEST_CASE("dflash2 loader seam: the VT_LMHEAD_FP4 rollback REFUSES an NVFP4 target head") {
  // The documented in-binary rollback, gated rather than assumed. With the
  // packed arm off, `LoadDflashSharedLmHead` falls through to the bf16 read and
  // the load refuses BY NAME at startup -- the pre-#1628 behaviour, which is the
  // point of a rollback. Fails CLOSED: nothing drafts off a head this build
  // cannot compute with.
  const SeamRun r = RunLoaderSeam(/*nvfp4_head=*/true, /*fp4_arm_on=*/false);
  INFO("threw: ", r.threw);
  CHECK_FALSE(r.loaded);
  CHECK(r.threw.find("lm_head.weight is not BF16") != std::string::npos);
  CHECK(r.stderr_text.find("shared head from the target safetensors shards") ==
        std::string::npos);
}

// ===========================================================================
// SPEC-DFLASH2 W8 (#1837) — the DFlash2 propose forward RE-ARMS the D13
// paged+graph draft step.
//
// Through W7 the runner's DFlash2 propose asked `ForwardBlockLogitsWithDeviceKV`
// for the post-final-norm hidden through the `final_out` HOST contract, and the
// paged fast path is guarded on `final_out == nullptr` — so every DFlash2 draft
// step took the materialized fallback, re-materialized the whole context, and
// downloaded the full f32 logits. The in-code debt note in
// `runner.cpp::propose_drafts_block` names this cost. W8 moves the hidden and
// the logits to DEVICE handles (`DflashBlockDeviceOut`), which restores the
// paged branch and its CUDA-graph capture for a DFlash2 draft.
//
// This case drives the model API in the RUNNER'S production shape — a DFlash2
// draft, a persistent device KV store, one (1+k) block, the block hidden
// requested for the candidate selector — under the capture-capable CPU harness,
// and asserts the capture/replay ROUTING the way the DFlash1 gate in
// test_qwen3_dflash_decode_graph_seam.cpp does: `segments_captured` and
// `replays` move only through the shared seam, so they are the one observable
// that separates "took the paged+graph lane" from "fell back". RED before W8:
// the DFlash2 shape disqualified the paged branch, and both counters stayed 0.
//
// WHAT THE CPU HARNESS CANNOT SEE, restated from the harness header: a
// "captured" region executes eagerly here and a "replay" recomputes nothing, so
// this gate holds ROUTING and the capture step's numerics. The real-replay
// numerics on device are `## Owed` O2 of the wave spec
// (.agents/specs/dflash2-device-propose.md).
TEST_CASE("dflash2 graph (W8): the propose forward captures + replays through the seam") {
  vllm_test::StaticGraphCpu graph_cpu;

  const Dims dm = Dflash2Dims(/*muse_glimmer_scalars=*/false);
  const ScratchCkpt ck(DraftEntries(dm));
  const HfConfig c = DraftConfig(dm);
  const Qwen3DFlashWeights w = LoadDraft(ck, dm, c);
  REQUIRE(w.IsDflash2());
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  // A persistent per-request store with a 3-row context, exactly as the runner
  // accumulates one before the first propose.
  auto store = Qwen3DFlashModel::MakeDeviceKVStore(c, q, kUnitCtxSlots);
  const int64_t H = dm.H;
  std::vector<float> ctx(static_cast<size_t>(3 * H));
  for (size_t i = 0; i < ctx.size(); ++i)
    ctx[i] = 0.2f * static_cast<float>(std::sin(0.13 * static_cast<double>(i) + 0.4));
  Qwen3DFlashModel::AppendContextKVDevice(*store, ctx, {0, 1, 2}, w, c, q);
  REQUIRE(Qwen3DFlashModel::DeviceKVNumCtx(*store) == 3);

  // One (1+k) block: anchor + k mask rows at positions 3..3+k, the runner's
  // exact layout. Tq must equal conv_block_size for a DFlash2 draft.
  const int64_t T = dm.block;
  std::vector<int32_t> ids(static_cast<size_t>(T), /*mask_token_id=*/7);
  ids[0] = 2;  // the anchor
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) pos[static_cast<size_t>(i)] = static_cast<int32_t>(3 + i);
  const std::vector<int32_t> cu = {0, static_cast<int32_t>(T)};
  const std::vector<int32_t> ctx_cu = {0, 3};
  std::vector<vllm::DflashDeviceKVStore*> stores = {store.get()};

  vt::ResetGraphBreakStats();

  // PROPOSE 1 — the production DFlash2 shape: the selector needs the block
  // hidden off the SAME forward, as DEVICE handles (W8). The reference arm for
  // the values is the materialized fallback below.
  Qwen3DFlashModel::DflashBlockDeviceOut dev1;
  const std::vector<float> host_ret = Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV(
      stores, ctx_cu, ids, pos, cu, w, c, q, nullptr, nullptr, &dev1);
  // Device handles came back, and the host logits vector is deliberately empty:
  // downloading it is the defect #1837 measures.
  CHECK(host_ret.empty());
  REQUIRE(dev1.logits.data != nullptr);
  REQUIRE(dev1.hidden.data != nullptr);
  CHECK(dev1.logits.shape[0] == T);
  CHECK(dev1.logits.shape[1] == dm.vocab);
  CHECK(dev1.hidden.shape[0] == T);
  CHECK(dev1.hidden.shape[1] == H);
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    // ONE full segment closed through the seam. RED before W8: the DFlash2
    // shape took the fallback and this read 0.
    CHECK(s.segments_captured == 1);
    CHECK(s.full_scopes == 1);
    CHECK(s.piecewise_scopes == 0);
    // The capture step hands out the PERSISTENT graph outputs, which hold no
    // computed values until one replay — so the driver replays once, in the
    // same step, before anything reads them (wave spec D2). Under this CPU
    // harness the "capture" executed eagerly, so the replay recomputes nothing
    // and the values are the warm pass's — which is the bit-identity the
    // eager-arm comparison below holds.
    CHECK(s.replays == 1);
  }

  // The capture step's DEVICE values are bit-identical to the materialized
  // fallback over the same store (the wave spec's D1 stop-condition pin,
  // capture lane).
  {
    std::vector<float> got_logits(static_cast<size_t>(T) * dm.vocab);
    std::vector<float> got_hidden_f32(static_cast<size_t>(T) * H);
    vt::Backend& b = vt::GetBackend(q.device.type);
    b.Copy(q, got_logits.data(), dev1.logits.data, got_logits.size() * sizeof(float));
    std::vector<uint16_t> got_hidden(static_cast<size_t>(T) * H);
    b.Copy(q, got_hidden.data(), dev1.hidden.data, got_hidden.size() * sizeof(uint16_t));
    b.Synchronize(q);
    for (size_t i = 0; i < got_hidden.size(); ++i)
      got_hidden_f32[i] = vt::BF16ToF32(got_hidden[i]);

    setenv("VT_DFLASH_PAGED", "0", 1);
    std::vector<float> ref_hidden;
    const std::vector<float> ref_logits = Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV(
        stores, ctx_cu, ids, pos, cu, w, c, q, nullptr, &ref_hidden);
    unsetenv("VT_DFLASH_PAGED");
    CHECK(got_logits == ref_logits);
    CHECK(got_hidden_f32 == ref_hidden);
  }

  // PROPOSE 2 and 3 — replays through the CONTAINER, with the handles reading
  // the persistent graph outputs (empty keeps: the store owns them).
  for (int i = 0; i < 2; ++i) {
    Qwen3DFlashModel::DflashBlockDeviceOut devN;
    const std::vector<float> r = Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV(
        stores, ctx_cu, ids, pos, cu, w, c, q, nullptr, nullptr, &devN);
    CHECK(r.empty());
    REQUIRE(devN.logits.data != nullptr);
    REQUIRE(devN.hidden.data != nullptr);
  }
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.segments_captured == 1);  // captured ONCE for this block width
    CHECK(s.replays == 3);            // the capture step's own + one per replay step
    CHECK(s.full_scopes == 1);
  }
}

// ===========================================================================
// SPEC-DFLASH2 W8 — the IDENTITY chain, link by link. The wave's non-negotiable
// is that the drafted tokens are BIT-IDENTICAL to the pre-W8 path, and rather
// than one cross-binary A/B the chain is held by executable links inside this
// tree: the graph case above pins capture-lane == materialized; the cases below
// pin eager-paged == materialized (the CPU production lane), device selector ==
// host selector (whose own value behaviour the untouched suites in this file
// pin), and the device pre-phase == the host pre-phase.

// T2 — the DEFAULT CPU production lane: eager paged with device handles must be
// bit-identical to the materialized fallback (wave spec D1's stop-condition pin,
// eager lane; the graph case above pins the capture lane).
TEST_CASE("dflash2 W8: eager-paged device handles == materialized fallback, BIT-IDENTICAL") {
  const Dims dm = Dflash2Dims(/*muse_glimmer_scalars=*/false);
  const ScratchCkpt ck(DraftEntries(dm));
  const HfConfig c = DraftConfig(dm);
  const Qwen3DFlashWeights w = LoadDraft(ck, dm, c);
  REQUIRE(w.IsDflash2());
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  auto store = Qwen3DFlashModel::MakeDeviceKVStore(c, q, kUnitCtxSlots);
  const int64_t H = dm.H;
  std::vector<float> ctx(static_cast<size_t>(5 * H));
  for (size_t i = 0; i < ctx.size(); ++i)
    ctx[i] = 0.3f * static_cast<float>(std::sin(0.29 * static_cast<double>(i) + 1.1));
  Qwen3DFlashModel::AppendContextKVDevice(*store, ctx, {0, 1, 2, 3, 4}, w, c, q);

  const int64_t T = dm.block;
  std::vector<int32_t> ids(static_cast<size_t>(T), 7);
  ids[0] = 3;
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) pos[static_cast<size_t>(i)] = static_cast<int32_t>(5 + i);
  const std::vector<int32_t> cu = {0, static_cast<int32_t>(T)};
  const std::vector<int32_t> ctx_cu = {0, 5};
  std::vector<vllm::DflashDeviceKVStore*> stores = {store.get()};

  // The paged eager lane with DEVICE handles (no harness: CPU cannot capture, so
  // graph_ok is false and this is the exact lane the production runner takes on
  // this box).
  Qwen3DFlashModel::DflashBlockDeviceOut dev;
  const std::vector<float> empty_ret = Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV(
      stores, ctx_cu, ids, pos, cu, w, c, q, nullptr, nullptr, &dev);
  CHECK(empty_ret.empty());
  REQUIRE(dev.logits.data != nullptr);
  REQUIRE(dev.hidden.data != nullptr);
  // The eager lane's handles OWN their storage (function-local buffers released
  // to the caller); the graph lane's replay handles are the store's.
  CHECK(dev.keep_logits != nullptr);
  CHECK(dev.keep_hidden != nullptr);
  std::vector<float> got_logits(static_cast<size_t>(T) * dm.vocab);
  std::vector<uint16_t> got_hidden(static_cast<size_t>(T) * H);
  vt::Backend& b = vt::GetBackend(q.device.type);
  b.Copy(q, got_logits.data(), dev.logits.data, got_logits.size() * sizeof(float));
  b.Copy(q, got_hidden.data(), dev.hidden.data, got_hidden.size() * sizeof(uint16_t));
  b.Synchronize(q);

  // The materialized fallback over the SAME store, host contract.
  setenv("VT_DFLASH_PAGED", "0", 1);
  std::vector<float> ref_hidden;
  const std::vector<float> ref_logits = Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV(
      stores, ctx_cu, ids, pos, cu, w, c, q, nullptr, &ref_hidden);
  unsetenv("VT_DFLASH_PAGED");

  CHECK(got_logits == ref_logits);
  REQUIRE(ref_hidden.size() == got_hidden.size());
  int hidden_mismatches = 0;
  for (size_t i = 0; i < got_hidden.size(); ++i)
    if (vt::BF16ToF32(got_hidden[i]) != ref_hidden[i]) ++hidden_mismatches;
  CHECK(hidden_mismatches == 0);
}

// T3 — the device selector+walk drafts the SAME tokens the host selector+walk
// does over the same block outputs. The host lane's own value behaviour is what
// every pre-W8 case in this file pins (membership, argmax disagreement, the
// scalar flips), so this equality closes the chain from the device lane back to
// the pinned behaviour. NOTE what this equality can and cannot see: the two
// lanes share ONE gather/top-k/edges implementation, so a defect in the shared
// core moves both lanes together and stays invisible here — the ABSOLUTE pins
// (the "+1..+k, never the anchor row" case above, the membership and flip
// gates) are what carry that mutation, and the W8 mutation run recorded exactly
// that split (wave spec, `## Mutations` (a)).
TEST_CASE("dflash2 W8: device selector+walk == host selector+walk, draft for draft") {
  const Dims dm = Dflash2Dims(/*muse_glimmer_scalars=*/true);  // scalars ON: the value path
  const ScratchCkpt ck(DraftEntries(dm));
  const HfConfig c = DraftConfig(dm);
  const Qwen3DFlashWeights w = LoadDraft(ck, dm, c);
  REQUIRE(w.IsDflash2());
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  const int64_t T = dm.block;
  const int k = static_cast<int>(T - 1);
  for (int shift = 0; shift < 4; ++shift) {
    std::vector<int32_t> ids(static_cast<size_t>(T)), pos(static_cast<size_t>(T));
    for (int64_t i = 0; i < T; ++i) {
      ids[static_cast<size_t>(i)] = static_cast<int32_t>((i * 5 + shift) % c.vocab_size);
      pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    const std::vector<int32_t> cu = {0, static_cast<int32_t>(T)};
    const std::vector<int32_t> anchors = {ids[0]};

    // The block outputs, once, through the context-free host forward.
    std::vector<float> block_hidden;
    const std::vector<float> block_logits = Qwen3DFlashModel::ForwardBlockLogits(
        ids, pos, cu, w, c, q, nullptr, &block_hidden);

    // HOST lane (the pre-W8 production composition, now a shell over the cores).
    const vllm::v1::Dflash2ProposeState host_state = vllm::v1::Dflash2SelectCandidates(
        block_logits, block_hidden, anchors, 1, k, w, c, q);
    const std::vector<std::vector<int32_t>> host_drafts =
        vllm::v1::Dflash2WalkPath(host_state, q).draft_token_ids;

    // DEVICE lane (the W8 production composition): the same bits, uploaded once.
    vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};
    vllm::dense_attn::DBuf logits_d(d, vt::DType::kF32, {T, dm.vocab}, block_logits.data());
    vllm::dense_attn::DBuf hidden_f32(d, vt::DType::kF32, {T, dm.H}, block_hidden.data());
    vllm::dense_attn::DBuf hidden_bf16(d, vt::DType::kBF16, {T, dm.H});
    vt::CastBf16(d.q, hidden_bf16.t(), hidden_f32.t());
    const vllm::v1::Dflash2ProposeStateDevice dev_state =
        vllm::v1::Dflash2SelectCandidatesDevice(logits_d.t(), hidden_bf16.t(), anchors, 1, k,
                                                w, c, q);
    const std::vector<std::vector<int32_t>> dev_drafts =
        vllm::v1::Dflash2WalkPathDevice(dev_state, q).draft_token_ids;

    REQUIRE(host_drafts.size() == 1);
    REQUIRE(dev_drafts.size() == 1);
    INFO("shift ", shift, " host ", VecStr(host_drafts[0]), " device ", VecStr(dev_drafts[0]));
    CHECK(host_drafts[0] == dev_drafts[0]);
  }
}

// T4 — the device pre-phase (#1838): CombineAuxFeaturesDevice +
// AppendContextKVDeviceRows build a store whose subsequent block forward is
// BIT-IDENTICAL to the host chain's (CombineAuxFeatures + host row gather +
// AppendContextKVDevice). The forward is the read-back instrument: two stores
// agree exactly iff every appended K/V bit agrees.
TEST_CASE("dflash2 W8: the device aux pre-phase == the host aux pre-phase, store for store") {
  const Dims dm = Dflash2Dims(/*muse_glimmer_scalars=*/false);
  const ScratchCkpt ck(DraftEntries(dm));
  const HfConfig c = DraftConfig(dm);
  const Qwen3DFlashWeights w = LoadDraft(ck, dm, c);
  REQUIRE(w.IsDflash2());
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};

  // A synthetic [T_total, H*taps] bf16 aux tap, and an out-of-order accepted-
  // prefix selection (rows 2,0,3 at positions 0,1,2) — the runner sorts rows by
  // committed position, so the gather order and the row order differ, which is
  // exactly what a wrong IndexSelect would corrupt.
  const int64_t T_total = 6, H = dm.H, Fin = H * dm.taps_fc;
  std::vector<uint16_t> aux_bits(static_cast<size_t>(T_total * Fin));
  for (size_t i = 0; i < aux_bits.size(); ++i)
    aux_bits[i] = vt::F32ToBF16(
        0.25f * static_cast<float>(std::sin(0.41 * static_cast<double>(i) + 0.7)));
  const std::vector<int32_t> rows = {2, 0, 3};
  const std::vector<int32_t> new_pos = {0, 1, 2};

  // HOST chain (the pre-W8 runner): bf16 -> f32 on the host, host fc combine,
  // host row gather, f32 append.
  std::vector<float> aux_f32(aux_bits.size());
  for (size_t i = 0; i < aux_bits.size(); ++i) aux_f32[i] = vt::BF16ToF32(aux_bits[i]);
  const std::vector<float> combined_host =
      Qwen3DFlashModel::CombineAuxFeatures(aux_f32, T_total, w, c, q);
  std::vector<float> feats;
  for (const int32_t r : rows) {
    const float* src = combined_host.data() + static_cast<size_t>(r) * static_cast<size_t>(H);
    feats.insert(feats.end(), src, src + H);
  }
  auto store_host = Qwen3DFlashModel::MakeDeviceKVStore(c, q, kUnitCtxSlots);
  Qwen3DFlashModel::AppendContextKVDevice(*store_host, feats, new_pos, w, c, q);

  // DEVICE chain (W8): the tap stays on device end to end.
  vllm::dense_attn::DBuf aux_d(d, vt::DType::kBF16, {T_total, Fin}, aux_bits.data());
  const Qwen3DFlashModel::DflashCombinedDevice combined_dev =
      Qwen3DFlashModel::CombineAuxFeaturesDevice(aux_d.t(), w, c, q);
  auto store_dev = Qwen3DFlashModel::MakeDeviceKVStore(c, q, kUnitCtxSlots);
  Qwen3DFlashModel::AppendContextKVDeviceRows(*store_dev, combined_dev.tensor, rows, new_pos,
                                              w, c, q);

  REQUIRE(Qwen3DFlashModel::DeviceKVNumCtx(*store_host) == 3);
  REQUIRE(Qwen3DFlashModel::DeviceKVNumCtx(*store_dev) == 3);

  // Read both stores back through the SAME block forward.
  const int64_t T = dm.block;
  std::vector<int32_t> ids(static_cast<size_t>(T), 7);
  ids[0] = 1;
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) pos[static_cast<size_t>(i)] = static_cast<int32_t>(3 + i);
  const std::vector<int32_t> cu = {0, static_cast<int32_t>(T)};
  const std::vector<int32_t> ctx_cu = {0, 3};
  std::vector<vllm::DflashDeviceKVStore*> sh = {store_host.get()};
  std::vector<vllm::DflashDeviceKVStore*> sd = {store_dev.get()};
  const std::vector<float> out_host = Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV(
      sh, ctx_cu, ids, pos, cu, w, c, q);
  const std::vector<float> out_dev = Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV(
      sd, ctx_cu, ids, pos, cu, w, c, q);
  REQUIRE_FALSE(out_host.empty());
  CHECK(out_host == out_dev);
}

// ─── The TRELLIS shared head, computed with — #2495 item 6 ──────────────────
//
// The draft owns no head: it runs the TARGET's over its own hidden states. This
// is that read and that GEMM for the third storage form, and it is the case the
// benchmark target actually is -- `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` ships no
// `lm_head.weight` at all, so before this row the shared read had nothing to
// return and said nothing about it (#2569).
//
// THE REFERENCE IS THE SAME BYTES DECODED, not the same helper called twice. A
// case that compared `DflashLogitsF32D`'s trellis arm against a second call to
// `dense_exl3::Linear` would prove the two calls agree and nothing about the
// decode.
TEST_CASE("dflash2 shared lm_head: an EXL3 target head is COMPUTED WITH, packed") {
  const Dims dm = Exl3Dims();
  const ScratchCkpt draft_ck(DraftEntries(dm));
  const HfConfig c = DraftConfig(dm);
  const Exl3Head head = MakeExl3Head(dm.vocab, dm.H);

  Qwen3DFlashWeights w = LoadDraft(draft_ck, dm, c);
  REQUIRE(w.IsDflash2());
  const ScratchCkpt target(Exl3LmHeadEntries(head));
  // What `LoadDflashDraft` does: the draft's own `lm_head` is REPLACED by the
  // target's, and the draft vocabulary comes from whichever owner holds it --
  // `OutFeatures()` on this arm, which is the trellis geometry and not a
  // declared number.
  vllm::LoadDflashSharedLmHead(OpenShards(target), &w.lm_head, &w.lm_head_fp4,
                               &w.lm_head_exl3);
  REQUIRE_FALSE(w.lm_head_exl3.Empty());
  REQUIRE(w.lm_head.Empty());
  REQUIRE(w.lm_head_fp4.Empty());
  CHECK(w.lm_head_exl3.codebook == kExl3HeadCodebook);
  CHECK(w.lm_head_exl3.Bits() == kExl3HeadBits);
  CHECK(w.lm_head_exl3.InFeatures() == dm.H);
  CHECK(w.lm_head_exl3.OutFeatures() == dm.vocab);
  w.draft_vocab_size = w.lm_head_exl3.OutFeatures();
  CHECK(w.draft_vocab_size == dm.vocab);

  // THE FORWARD. `DflashLogitsF32D`'s trellis arm, entered from the draft's own
  // block forward. Deleting that arm does not merely move the numbers: the
  // packed case falls through to a `lm_head` an EXL3 read leaves EMPTY, which
  // the named refusal one line down reports.
  const std::vector<float> logits = Forward(w, c, dm.block);
  REQUIRE(logits.size() == static_cast<size_t>(dm.block * dm.vocab));
  for (const float x : logits) REQUIRE(std::isfinite(x));

  // The twin: the SAME trellis bytes decoded to a dense bf16 head, read through
  // the bf16 arm of the same function, run through the same forward.
  Qwen3DFlashWeights twin = LoadDraft(draft_ck, dm, c);
  const ScratchCkpt target_bf16(Exl3DecodedLmHeadEntries(head));
  vllm::LoadDflashSharedLmHead(OpenShards(target_bf16), &twin.lm_head,
                               &twin.lm_head_fp4, &twin.lm_head_exl3);
  REQUIRE_FALSE(twin.lm_head.Empty());
  REQUIRE(twin.lm_head_exl3.Empty());
  twin.draft_vocab_size = twin.lm_head.shape[0];
  const std::vector<float> reference = Forward(twin, c, dm.block);
  REQUIRE(reference.size() == logits.size());

  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < logits.size(); ++i) {
    const double d = static_cast<double>(logits[i]) - reference[i];
    num += d * d;
    den += static_cast<double>(reference[i]) * reference[i];
  }
  // NOT VACUOUS: a forward returning zeros satisfies any relative bound whose
  // reference is also zero.
  REQUIRE(den > 0.0);
  const double rel = std::sqrt(num / den);
  MESSAGE("dflash exl3 shared head vs decoded twin: rel_rms = ", rel);
  // A BOUND, not an equality: EXL3 rides the Hadamards on the activations while
  // the decoded twin has them baked in, and the twin rounds the decode to bf16.
  CHECK(rel <= 5.0e-2);

  // The D12 guard is untouched. A packed head is COMPUTED WITH, which is the
  // opposite of the widened state `lm_head_dequantized` records.
  CHECK_FALSE(w.lm_head_dequantized);
  CHECK_NOTHROW(vllm::RefuseQuantizedDflash2LmHead(w));
  Qwen3DFlashWeights widened = w;
  widened.lm_head_dequantized = true;
  CHECK_THROWS(vllm::RefuseQuantizedDflash2LmHead(widened));
}
