#include "vllm/v1/core/recurrent_prefix_snapshot.h"
// Ported from: vllm/v1/core/kv_cache_utils.py @ e24d1b24
// See include/vllm/v1/core/kv_cache_utils.h for scope, the BlockHash/Task 2
// coordination note, and recorded deviations.
#include "vllm/v1/core/kv_cache_utils.h"

#include <array>
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/kv_cache_spec_registry.h"
#include "vllm/v1/request.h"

namespace vllm::v1 {

void unify_hybrid_kv_cache_specs(
    std::unordered_map<std::string, std::shared_ptr<KVCacheSpec>>&
        kv_cache_specs) {
  std::vector<const KVCacheSpec*> specs;
  specs.reserve(kv_cache_specs.size());
  for (const auto& [layer_name, spec] : kv_cache_specs) {
    (void)layer_name;
    specs.push_back(spec.get());
  }
  if (are_uniform_kv_cache_specs(specs)) {
    return;
  }

  bool has_full_attention = false;
  bool has_sliding_window = false;
  bool has_chunked_local_attention = false;
  for (const auto& [layer_name, spec] : kv_cache_specs) {
    (void)layer_name;
    has_full_attention =
        has_full_attention ||
        dynamic_cast<const FullAttentionSpec*>(spec.get()) != nullptr;
    has_sliding_window =
        has_sliding_window ||
        dynamic_cast<const SlidingWindowSpec*>(spec.get()) != nullptr;
    has_chunked_local_attention =
        has_chunked_local_attention ||
        dynamic_cast<const ChunkedLocalAttentionSpec*>(spec.get()) != nullptr;
  }

  if (has_full_attention &&
      (has_sliding_window || has_chunked_local_attention)) {
    for (auto& [layer_name, spec] : kv_cache_specs) {
      (void)layer_name;
      const auto* sliding =
          dynamic_cast<const SlidingWindowSpec*>(spec.get());
      if (sliding != nullptr) {
        spec = std::make_shared<FullAttentionSpec>(
            sliding->block_size, sliding->num_kv_heads, sliding->head_size,
            sliding->dtype, sliding->head_size_v, sliding->kv_quant_mode,
            sliding->page_size_padded,
            /*indexes_kv_by_block_stride=*/false, sliding->sliding_window);
        continue;
      }
      const auto* chunked =
          dynamic_cast<const ChunkedLocalAttentionSpec*>(spec.get());
      if (chunked != nullptr) {
        spec = std::make_shared<FullAttentionSpec>(
            chunked->block_size, chunked->num_kv_heads, chunked->head_size,
            chunked->dtype, /*head_size_v=*/std::nullopt,
            KVQuantMode::kNone, chunked->page_size_padded,
            /*indexes_kv_by_block_stride=*/false,
            /*sliding_window=*/std::nullopt,
            chunked->attention_chunk_size);
      }
    }
  }

  specs.clear();
  for (const auto& [layer_name, spec] : kv_cache_specs) {
    (void)layer_name;
    specs.push_back(spec.get());
  }
  if (!are_uniform_kv_cache_specs(specs)) {
    throw std::invalid_argument(
        "Hybrid KV cache manager is disabled but failed to convert the KV "
        "cache specs to one unified type.");
  }
}

namespace {

// --- SHA-256 -----------------------------------------------------------------
// A compact, self-contained SHA-256 (FIPS 180-4). Vendored because the project
// has no crypto dependency; it matches Python's hashlib.sha256 digest so the
// ported sha256_cbor block-hash vectors line up byte-for-byte.

constexpr std::array<uint32_t, 64> kSha256K = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

inline uint32_t Rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

std::string Sha256(const std::string& data) {
  uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                   0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

  // Padding: append 0x80, then zeros, then the 64-bit big-endian bit length.
  std::string msg = data;
  const uint64_t bit_len = static_cast<uint64_t>(data.size()) * 8;
  msg.push_back(static_cast<char>(0x80));
  while (msg.size() % 64 != 56) msg.push_back('\0');
  for (int i = 7; i >= 0; --i) {
    msg.push_back(static_cast<char>((bit_len >> (i * 8)) & 0xff));
  }

  const auto byte = [&](size_t i) {
    return static_cast<uint32_t>(static_cast<unsigned char>(msg[i]));
  };
  for (size_t off = 0; off < msg.size(); off += 64) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (byte(off + i * 4) << 24) | (byte(off + i * 4 + 1) << 16) |
             (byte(off + i * 4 + 2) << 8) | byte(off + i * 4 + 3);
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t s1 = Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = hh + s1 + ch + kSha256K[i] + w[i];
      const uint32_t s0 = Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = s0 + maj;
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }

  std::string digest(32, '\0');
  for (int i = 0; i < 8; ++i) {
    digest[i * 4 + 0] = static_cast<char>((h[i] >> 24) & 0xff);
    digest[i * 4 + 1] = static_cast<char>((h[i] >> 16) & 0xff);
    digest[i * 4 + 2] = static_cast<char>((h[i] >> 8) & 0xff);
    digest[i * 4 + 3] = static_cast<char>(h[i] & 0xff);
  }
  return digest;
}

// Append a CBOR head (major type in the top 3 bits + minimal-width argument).
void CborHead(std::string& out, uint8_t major, uint64_t arg) {
  const uint8_t mt = static_cast<uint8_t>(major << 5);
  if (arg < 24) {
    out.push_back(static_cast<char>(mt | static_cast<uint8_t>(arg)));
  } else if (arg < 0x100ULL) {
    out.push_back(static_cast<char>(mt | 24));
    out.push_back(static_cast<char>(arg & 0xff));
  } else if (arg < 0x10000ULL) {
    out.push_back(static_cast<char>(mt | 25));
    for (int i = 1; i >= 0; --i) out.push_back(static_cast<char>((arg >> (i * 8)) & 0xff));
  } else if (arg < 0x100000000ULL) {
    out.push_back(static_cast<char>(mt | 26));
    for (int i = 3; i >= 0; --i) out.push_back(static_cast<char>((arg >> (i * 8)) & 0xff));
  } else {
    out.push_back(static_cast<char>(mt | 27));
    for (int i = 7; i >= 0; --i) out.push_back(static_cast<char>((arg >> (i * 8)) & 0xff));
  }
}

// Convert one ExtraKey into its CborValue (text, or a [text, int] array).
CborValue ExtraKeyToCbor(const ExtraKey& key) {
  if (const auto* s = std::get_if<std::string>(&key)) {
    return CborValue::Text(*s);
  }
  const auto& pair = std::get<std::pair<std::string, int64_t>>(key);
  return CborValue::Array({CborValue::Text(pair.first), CborValue::Int(pair.second)});
}

}  // namespace

// --- CborValue ---------------------------------------------------------------

CborValue CborValue::UInt(uint64_t value) {
  CborValue v;
  v.type_ = Type::kUInt;
  v.arg_ = value;
  return v;
}

CborValue CborValue::Int(int64_t value) {
  if (value >= 0) return UInt(static_cast<uint64_t>(value));
  CborValue v;
  v.type_ = Type::kNInt;
  // Major type 1 encodes -1 - value; for value < 0 this is a non-negative arg.
  v.arg_ = static_cast<uint64_t>(-(value + 1));
  return v;
}

CborValue CborValue::Bytes(std::string bytes) {
  CborValue v;
  v.type_ = Type::kBytes;
  v.str_ = std::move(bytes);
  return v;
}

CborValue CborValue::Text(std::string text) {
  CborValue v;
  v.type_ = Type::kText;
  v.str_ = std::move(text);
  return v;
}

CborValue CborValue::Array(std::vector<CborValue> items) {
  CborValue v;
  v.type_ = Type::kArray;
  v.items_ = std::move(items);
  return v;
}

CborValue CborValue::Null() {
  CborValue v;
  v.type_ = Type::kNull;
  return v;
}

void CborValue::Encode(std::string& out) const {
  switch (type_) {
    case Type::kUInt:
      CborHead(out, 0, arg_);
      break;
    case Type::kNInt:
      CborHead(out, 1, arg_);
      break;
    case Type::kBytes:
      CborHead(out, 2, str_.size());
      out += str_;
      break;
    case Type::kText:
      CborHead(out, 3, str_.size());
      out += str_;
      break;
    case Type::kArray:
      CborHead(out, 4, items_.size());
      for (const auto& item : items_) item.Encode(out);
      break;
    case Type::kNull:
      out.push_back(static_cast<char>(0xf6));
      break;
  }
}

std::string CborValue::Encode() const {
  std::string out;
  Encode(out);
  return out;
}

BlockHash sha256_cbor(const CborValue& value) { return Sha256(value.Encode()); }

std::string sha256_bytes(const std::string& data) { return Sha256(data); }

// --- Block-hash packing ------------------------------------------------------

BlockHashWithGroupId make_block_hash_with_group_id(const BlockHash& block_hash,
                                                   uint32_t group_id) {
  BlockHashWithGroupId key = block_hash;
  key.push_back(static_cast<char>((group_id >> 24) & 0xff));
  key.push_back(static_cast<char>((group_id >> 16) & 0xff));
  key.push_back(static_cast<char>((group_id >> 8) & 0xff));
  key.push_back(static_cast<char>(group_id & 0xff));
  return key;
}

BlockHash get_block_hash(const BlockHashWithGroupId& key) {
  assert(key.size() >= 4);
  return key.substr(0, key.size() - 4);
}

uint32_t get_group_id(const BlockHashWithGroupId& key) {
  assert(key.size() >= 4);
  const size_t n = key.size();
  const auto b = [&](size_t i) {
    return static_cast<uint32_t>(static_cast<unsigned char>(key[i]));
  };
  return (b(n - 4) << 24) | (b(n - 3) << 16) | (b(n - 2) << 8) | b(n - 1);
}

// --- NONE_HASH ---------------------------------------------------------------

BlockHash NONE_HASH;

namespace {
NoneHashProvenance g_none_hash_provenance;

// getenv, treating an empty value as unset (an exported-but-empty variable is
// the same "not configured" state as an absent one).
std::optional<std::string> GetEnvNonEmpty(const char* name) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') {
    return std::nullopt;
  }
  return std::string(v);
}
}  // namespace

const char* NoneHashProvenance::source_name() const {
  switch (source) {
    case NoneHashSeedSource::kExplicit:
      return "explicit";
    case NoneHashSeedSource::kEnvVllmCpp:
      return "VLLM_PREFIX_CACHING_HASH_SEED";
    case NoneHashSeedSource::kEnvPythonHashSeed:
      return "PYTHONHASHSEED";
    case NoneHashSeedSource::kDefault:
      return "default";
    case NoneHashSeedSource::kRandom:
      return "random";
  }
  return "unknown";
}

const NoneHashProvenance& none_hash_provenance() {
  return g_none_hash_provenance;
}

ExternalBlockHash maybe_convert_block_hash(const BlockHash& hash_bytes) {
  // kv_cache_utils.py:79-82. Upstream parses the env with
  // `bool(int(os.getenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "1")))`
  // (envs.py:1816-1817) — DEFAULT "1" => True => convert to int. Only "0"
  // (or another falsy int) keeps the raw bytes.
  bool use_int = true;  // upstream default.
  if (auto env = GetEnvNonEmpty("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES")) {
    try {
      use_int = std::stoll(*env) != 0;
    } catch (const std::exception&) {
      // Mirror Python: a non-integer value raises; we conservatively keep the
      // default rather than abort a running server on a malformed env.
      use_int = true;
    }
  }
  if (!use_int) {
    return ExternalBlockHash{hash_bytes};
  }
  // Low 64 bits of the big-endian digest == the last 8 bytes, big-endian.
  uint64_t value = 0;
  const size_t n = hash_bytes.size();
  const size_t start = n > 8 ? n - 8 : 0;
  for (size_t i = start; i < n; ++i) {
    value = (value << 8) | static_cast<uint8_t>(hash_bytes[i]);
  }
  return ExternalBlockHash{value};
}

void init_none_hash(const HashFn& hash_fn, std::optional<std::string> seed) {
  NoneHashSeedSource source = NoneHashSeedSource::kExplicit;

  if (!seed.has_value()) {
    // Our escape hatch first: it is the one that can also SELECT upstream's
    // random behaviour, so it must be able to override the default.
    if (auto env = GetEnvNonEmpty("VLLM_PREFIX_CACHING_HASH_SEED")) {
      if (*env == "random") {
        // Opt in to upstream's os.urandom(32) branch
        // (vllm/v1/core/kv_cache_utils.py:111-112). Block hashes then differ
        // across processes and no persisted cache can be shared.
        std::random_device rd;
        std::string bytes(32, '\0');
        for (int i = 0; i < 32; ++i) {
          bytes[i] = static_cast<char>(rd() & 0xff);
        }
        NONE_HASH = bytes;
        g_none_hash_provenance =
            NoneHashProvenance{NoneHashSeedSource::kRandom, std::string()};
        return;
      }
      seed = *env;
      source = NoneHashSeedSource::kEnvVllmCpp;
    } else if (auto py = GetEnvNonEmpty("PYTHONHASHSEED")) {
      // Upstream's escape hatch, mirrored so an operator who already sets
      // PYTHONHASHSEED across a vLLM fleet gets the same block hashes here.
      seed = *py;
      source = NoneHashSeedSource::kEnvPythonHashSeed;
    } else {
      seed = std::string(kDefaultNoneHashSeed);
      source = NoneHashSeedSource::kDefault;
    }
  }

  NONE_HASH = hash_fn(CborValue::Text(*seed));
  g_none_hash_provenance = NoneHashProvenance{source, *seed};
}

// --- Hashing -----------------------------------------------------------------

namespace {

// _gen_mm_extra_hash_keys (vllm/v1/core/kv_cache_utils.py:451-517). For each
// multimodal input that OVERLAPS this block, emit (mm_hash, offset-in-block).
// mm_features are assumed sorted by mm_position.offset. Returns the extra keys
// for this block plus the next mm index to resume from.
std::vector<ExtraKey> GenMmExtraHashKeys(const Request& request,
                                         int start_token_idx, int end_token_idx,
                                         int start_mm_idx, int* next_mm_idx) {
  std::vector<ExtraKey> extra_keys;
  const auto& mm_features = request.mm_features;
  if (mm_features.empty()) {
    *next_mm_idx = start_mm_idx;
    return extra_keys;
  }

  // Skip the whole scan once the block starts past the last mm input (the late
  // prefill / decode phase). Mirrors the last_pos short-circuit.
  const auto& last = mm_features.back();
  if (last.offset + last.length <= start_token_idx) {
    *next_mm_idx = start_mm_idx;
    return extra_keys;
  }

  const int n = static_cast<int>(mm_features.size());
  // start_mm_idx == -1 indicates the last mm input (block completed by
  // generated tokens, so only the final mm input can matter).
  if (start_mm_idx < 0) {
    assert(-start_mm_idx <= n);
    start_mm_idx = n + start_mm_idx;
  }

  int curr_mm_idx = start_mm_idx;
  while (curr_mm_idx < n) {
    const auto& mm_feature = mm_features[curr_mm_idx];
    // Upstream asserts mm_feature.identifier is not None; mm_hash is that id.
    const int offset = mm_feature.offset;
    const int length = mm_feature.length;
    if (end_token_idx > offset) {
      if (start_token_idx >= offset + length) {
        // This block has passed the current mm input.
        ++curr_mm_idx;
        continue;
      }
      // The block contains the current mm input. Include the mm hash and its
      // offset RELATIVE to the start of the block, so identical placeholder
      // blocks at different positions stay distinct.
      extra_keys.emplace_back(std::pair<std::string, int64_t>(
          mm_feature.mm_hash,
          static_cast<int64_t>(offset) - static_cast<int64_t>(start_token_idx)));
      if (end_token_idx >= offset + length) {
        // This block also ends the current mm input; it may contain the next.
        ++curr_mm_idx;
      } else {
        // Otherwise this block is done with mm inputs.
        break;
      }
    } else {
      // This block has not reached the current mm input.
      break;
    }
  }
  *next_mm_idx = curr_mm_idx;
  return extra_keys;
}

}  // namespace

std::pair<ExtraKeys, int> generate_block_hash_extra_keys(const Request& request,
                                                         int start_token_idx,
                                                         int end_token_idx,
                                                         int start_mm_idx) {
  // Mirrors generate_block_hash_extra_keys
  // (vllm/v1/core/kv_cache_utils.py:562-591). Fixed concatenation order:
  //   lora -> mm -> cache_salt -> (prompt_embeds, deferred: no prompt-embeds
  //   path in this tree). An empty result hashes as None (nullopt), which is
  //   DISTINCT from an empty tuple.
  int next_mm_idx = start_mm_idx;
  std::vector<ExtraKey> mm_extra_keys =
      GenMmExtraHashKeys(request, start_token_idx, end_token_idx, start_mm_idx,
                         &next_mm_idx);

  std::vector<ExtraKey> extra_keys;
  // LoRA name applies to every block (_gen_lora_extra_hash_keys, :523-534).
  if (request.lora_name.has_value()) {
    extra_keys.emplace_back(*request.lora_name);
  }
  // mm keys next.
  for (auto& key : mm_extra_keys) {
    extra_keys.push_back(std::move(key));
  }
  // cache_salt applies to the FIRST block only (:579-581); chaining carries it
  // transitively into later blocks' parent hash.
  if (start_token_idx == 0 && request.cache_salt.has_value() &&
      !request.cache_salt->empty()) {
    extra_keys.emplace_back(*request.cache_salt);
  }

  if (extra_keys.empty()) {
    return {std::nullopt, next_mm_idx};
  }
  return {std::move(extra_keys), next_mm_idx};
}

BlockHash hash_block_tokens(const HashFn& hash_function,
                            const std::optional<BlockHash>& parent_block_hash,
                            const std::vector<int32_t>& curr_block_token_ids,
                            const ExtraKeys& extra_keys) {
  // Upstream: `if not parent_block_hash: parent_block_hash = NONE_HASH`. Python
  // treats both None and empty bytes as falsy.
  const BlockHash& parent = (parent_block_hash.has_value() && !parent_block_hash->empty())
                                ? *parent_block_hash
                                : NONE_HASH;

  std::vector<CborValue> token_items;
  token_items.reserve(curr_block_token_ids.size());
  for (int32_t token_id : curr_block_token_ids) {
    token_items.push_back(CborValue::Int(token_id));
  }

  CborValue extra_value = CborValue::Null();
  if (extra_keys.has_value()) {
    std::vector<CborValue> extra_items;
    extra_items.reserve(extra_keys->size());
    for (const ExtraKey& key : *extra_keys) {
      extra_items.push_back(ExtraKeyToCbor(key));
    }
    extra_value = CborValue::Array(std::move(extra_items));
  }

  CborValue input = CborValue::Array({CborValue::Bytes(parent),
                                      CborValue::Array(std::move(token_items)),
                                      std::move(extra_value)});
  return hash_function(input);
}

std::vector<BlockHash> hash_request_tokens(
    const HashFn& hash_function, int block_size,
    const std::vector<int32_t>& token_ids,
    const std::vector<ExtraKeys>& per_block_extra_keys) {
  std::vector<BlockHash> ret;
  const int num_tokens = static_cast<int>(token_ids.size());
  std::optional<BlockHash> prev_block_hash = std::nullopt;
  int start_token_idx = 0;
  size_t block_idx = 0;
  const ExtraKeys no_extra_keys = std::nullopt;
  // Only hash full blocks; a partial trailing block is left unhashed.
  while (start_token_idx + block_size <= num_tokens) {
    const int end_token_idx = start_token_idx + block_size;
    std::vector<int32_t> block_tokens(token_ids.begin() + start_token_idx,
                                      token_ids.begin() + end_token_idx);
    const ExtraKeys& extra_keys = block_idx < per_block_extra_keys.size()
                                      ? per_block_extra_keys[block_idx]
                                      : no_extra_keys;
    BlockHash block_hash =
        hash_block_tokens(hash_function, prev_block_hash, block_tokens, extra_keys);
    ret.push_back(block_hash);
    prev_block_hash = block_hash;
    start_token_idx += block_size;
    ++block_idx;
  }
  return ret;
}

BlockHasher get_request_block_hasher(int hash_block_size,
                                     const HashFn& caching_hash_fn,
                                     bool mtp_shifted_boundary) {
  return [hash_block_size, caching_hash_fn, mtp_shifted_boundary](
             const Request& request) -> std::vector<BlockHash> {
    int start_token_idx =
        static_cast<int>(request.block_hashes.size()) * hash_block_size;
    const int num_tokens = request.NumTokens();

    // Early stop when there are no new full blocks created.
    if (start_token_idx + hash_block_size > num_tokens) {
      return {};
    }

    int curr_mm_idx = 0;
    if (start_token_idx > 0) {
      // curr_mm_idx = -1 indicates the last mm input: we reach this branch only
      // when the block is completed with generated tokens, so only the last mm
      // input matters.
      curr_mm_idx = -1;
    }

    std::optional<BlockHash> prev_block_hash_value = std::nullopt;
    if (!request.block_hashes.empty()) {
      prev_block_hash_value = request.block_hashes.back();
    }

    // all_token_ids = prompt + output (the T0 no-prompt-embeds path). Upstream
    // reads request.all_token_ids[start:end]; we materialize it here.
    std::vector<int32_t> all_token_ids = request.prompt_token_ids;
    all_token_ids.insert(all_token_ids.end(), request.output_token_ids.begin(),
                         request.output_token_ids.end());

    std::vector<BlockHash> new_block_hashes;
    while (true) {
      const int end_token_idx = start_token_idx + hash_block_size;
      if (end_token_idx > num_tokens ||
          (mtp_shifted_boundary && end_token_idx == num_tokens)) {
        // We only hash full blocks.
        break;
      }

      // MM / LoRA / cache_salt requests fold extra keys into the block hash
      // (generate_block_hash_extra_keys). None for the ordinary text path.
      std::pair<ExtraKeys, int> extra = generate_block_hash_extra_keys(
          request, start_token_idx, end_token_idx, curr_mm_idx);
      curr_mm_idx = extra.second;
      if (mtp_shifted_boundary) {
        if (!extra.first) extra.first.emplace();
        extra.first->emplace_back(std::make_pair(
            std::string("mtp-next-token-v1"), int64_t(all_token_ids[end_token_idx])));
      }

      std::vector<int32_t> block_tokens(all_token_ids.begin() + start_token_idx,
                                        all_token_ids.begin() + end_token_idx);
      BlockHash block_hash = hash_block_tokens(
          caching_hash_fn, prev_block_hash_value, block_tokens, extra.first);

      new_block_hashes.push_back(block_hash);
      start_token_idx += hash_block_size;
      prev_block_hash_value = block_hash;
    }

    return new_block_hashes;
  };
}

std::pair<int, int> resolve_kv_cache_block_sizes(
    const KVCacheConfig& kv_cache_config, int cache_block_size,
    std::optional<int> prefix_match_unit, bool enable_prefix_caching,
    bool connector_enabled, int dcp_world_size) {
  const std::vector<KVCacheGroupSpec>& groups = kv_cache_config.kv_cache_groups;

  if (groups.size() <= 1) {
    const int bs = cache_block_size * dcp_world_size;
    return {bs, bs};
  }

  // Attention groups are scaled by DCP; mamba groups keep their full per-rank
  // state and are not scaled. Mirrors kv_cache_utils.py:653-658.
  std::vector<int> group_block_sizes;
  group_block_sizes.reserve(groups.size());
  for (const KVCacheGroupSpec& g : groups) {
    const KVCacheSpec* spec = g.kv_cache_spec.get();
    if (dynamic_cast<const AttentionSpec*>(spec) != nullptr) {
      group_block_sizes.push_back(spec->block_size * dcp_world_size);
    } else {
      group_block_sizes.push_back(spec->block_size);
    }
  }
  int scheduler_block_size = group_block_sizes[0];
  for (size_t i = 1; i < group_block_sizes.size(); ++i) {
    scheduler_block_size = std::lcm(scheduler_block_size, group_block_sizes[i]);
  }

  // Block hashes are only consumed by prefix caching and KV connectors; when
  // neither is active, keep hash_block_size equal to the scheduler block size.
  // (kv_cache_utils.py:661-666)
  if (!(enable_prefix_caching || connector_enabled)) {
    return {scheduler_block_size, scheduler_block_size};
  }

  // Mamba groups with block_size != cache_block_size (mamba_cache_mode !=
  // "align") break divisibility; back off to the scheduler block size.
  // (kv_cache_utils.py:668-676)
  for (const KVCacheGroupSpec& g : groups) {
    const MambaSpec* m = dynamic_cast<const MambaSpec*>(g.kv_cache_spec.get());
    if (m != nullptr && m->block_size != cache_block_size) {
      return {scheduler_block_size, scheduler_block_size};
    }
  }

  int hash_block_size;
  if (prefix_match_unit.has_value()) {
    hash_block_size = *prefix_match_unit;
  } else {
    hash_block_size = group_block_sizes[0];
    for (size_t i = 1; i < group_block_sizes.size(); ++i) {
      hash_block_size = std::gcd(hash_block_size, group_block_sizes[i]);
    }
  }
  for (int bs : group_block_sizes) {
    if (bs % hash_block_size != 0) {
      std::string sizes;
      for (size_t i = 0; i < group_block_sizes.size(); ++i) {
        if (i > 0) sizes += ", ";
        sizes += std::to_string(group_block_sizes[i]);
      }
      throw std::invalid_argument(
          "Invalid prefix_match_unit=" + std::to_string(hash_block_size) +
          "; all KV cache group block sizes must be divisible by "
          "prefix_match_unit. Got group block sizes=[" + sizes + "].");
    }
  }
  return {scheduler_block_size, hash_block_size};
}

void KVCacheBlock::set_block_hash(BlockHashWithGroupId block_hash,
                                  std::optional<int> num_tokens) {
  // "The block already has a hash. This should not happen."
  assert(!block_hash_.has_value() && !block_hash_num_tokens_.has_value());
  block_hash_ = std::move(block_hash);
  block_hash_num_tokens_ = num_tokens;
}

void KVCacheBlock::reset_hash() {
  block_hash_ = std::nullopt;
  block_hash_num_tokens_ = std::nullopt;
}

FreeKVCacheBlockQueue::FreeKVCacheBlockQueue(
    const std::vector<KVCacheBlock*>& blocks)
    : num_free_blocks(static_cast<int>(blocks.size())) {
  // Initialize doubly links of consecutive blocks.
  for (int i = 0; i < num_free_blocks; ++i) {
    if (i > 0) {
      blocks[i]->prev_free_block = blocks[i - 1];
    }
    if (i < num_free_blocks - 1) {
      blocks[i]->next_free_block = blocks[i + 1];
    }
  }

  // The fake head and tail are NEVER popped, so we can safely assume each real
  // block in the queue has prev and next blocks.
  if (num_free_blocks > 0) {
    // Connect fake_head and fake_tail to the first and last block respectively.
    fake_free_list_head.next_free_block = blocks.front();
    blocks.front()->prev_free_block = &fake_free_list_head;
    fake_free_list_tail.prev_free_block = blocks.back();
    blocks.back()->next_free_block = &fake_free_list_tail;
  } else {
    // For empty list, simply connect the fake head and tail.
    fake_free_list_head.next_free_block = &fake_free_list_tail;
    fake_free_list_tail.prev_free_block = &fake_free_list_head;
  }
}

KVCacheBlock* FreeKVCacheBlockQueue::popleft() {
  if (fake_free_list_head.next_free_block == &fake_free_list_tail ||
      fake_free_list_head.next_free_block == nullptr) {
    assert(num_free_blocks == 0 &&
           "num_free_blocks is out of sync with the free list.");
    throw std::runtime_error("No free blocks available");
  }

  KVCacheBlock* first_block = fake_free_list_head.next_free_block;

  if (first_block->next_free_block == nullptr) {
    // Indicates a bug in the caller's logic.
    throw std::runtime_error(
        "Invalid block found in popleft() "
        "which doesn't have a valid next_free_block");
  }

  // Connect fake_head and the next block of first_block (second block or fake
  // tail).
  fake_free_list_head.next_free_block = first_block->next_free_block;
  first_block->next_free_block->prev_free_block = &fake_free_list_head;

  // Remove the block from the linked list.
  first_block->prev_free_block = nullptr;
  first_block->next_free_block = nullptr;

  num_free_blocks -= 1;
  return first_block;
}

std::vector<KVCacheBlock*> FreeKVCacheBlockQueue::popleft_n(int n) {
  if (n == 0) {
    return {};
  }
  assert(num_free_blocks >= n);
  num_free_blocks -= n;

  KVCacheBlock* curr_block = fake_free_list_head.next_free_block;
  // Pop n blocks from the head of the list.
  std::vector<KVCacheBlock*> ret;
  ret.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    assert(curr_block != nullptr);
    ret.push_back(curr_block);
    KVCacheBlock* last_block = curr_block;
    curr_block = curr_block->next_free_block;
    // Reset prev_free_block and next_free_block of all popped blocks.
    last_block->prev_free_block = nullptr;
    last_block->next_free_block = nullptr;
  }

  if (curr_block != nullptr) {
    // The queue is not empty, connect the fake head to the new first block.
    fake_free_list_head.next_free_block = curr_block;
    curr_block->prev_free_block = &fake_free_list_head;
  }
  return ret;
}

void FreeKVCacheBlockQueue::remove(KVCacheBlock* block) {
  if (block->prev_free_block == nullptr || block->next_free_block == nullptr) {
    // Indicates a bug in the caller's logic.
    throw std::runtime_error("remove() called on an invalid block");
  }

  // Link the previous block to the next block.
  block->prev_free_block->next_free_block = block->next_free_block;
  // Link the next block to the previous block.
  block->next_free_block->prev_free_block = block->prev_free_block;

  // Remove the block from the linked list.
  block->prev_free_block = nullptr;
  block->next_free_block = nullptr;
  num_free_blocks -= 1;
}

void FreeKVCacheBlockQueue::append(KVCacheBlock* block) {
  if (fake_free_list_tail.prev_free_block == nullptr) {
    throw std::runtime_error(
        "prev_free_block of fake_free_list_tail should always exist");
  }
  KVCacheBlock* last_block = fake_free_list_tail.prev_free_block;

  // Connect the new block after the last block.
  last_block->next_free_block = block;
  block->prev_free_block = last_block;

  // Connect the fake tail after the new block.
  block->next_free_block = &fake_free_list_tail;
  fake_free_list_tail.prev_free_block = block;

  num_free_blocks += 1;
}

void FreeKVCacheBlockQueue::prepend_n(const std::vector<KVCacheBlock*>& blocks) {
  if (blocks.empty()) {
    return;
  }

  KVCacheBlock* first_block = fake_free_list_head.next_free_block;
  assert(first_block != nullptr &&
         "next_free_block of fake_free_list_head should always exist");

  KVCacheBlock* prev_block = &fake_free_list_head;
  for (KVCacheBlock* block : blocks) {
    block->prev_free_block = prev_block;
    prev_block->next_free_block = block;
    prev_block = block;
  }

  prev_block->next_free_block = first_block;
  first_block->prev_free_block = prev_block;

  num_free_blocks += static_cast<int>(blocks.size());
}

void FreeKVCacheBlockQueue::append_n(const std::vector<KVCacheBlock*>& blocks) {
  if (blocks.empty()) {
    return;
  }

  KVCacheBlock* last_block = fake_free_list_tail.prev_free_block;
  assert(last_block != nullptr &&
         "prev_free_block of fake_free_list_tail should always exist");
  // Add inter-connections between consecutive blocks.
  for (KVCacheBlock* block : blocks) {
    block->prev_free_block = last_block;
    last_block->next_free_block = block;
    last_block = block;
  }

  // Connect the last block of <blocks> to the fake tail.
  last_block->next_free_block = &fake_free_list_tail;
  fake_free_list_tail.prev_free_block = last_block;

  num_free_blocks += static_cast<int>(blocks.size());
}

std::vector<KVCacheBlock*> FreeKVCacheBlockQueue::get_all_free_blocks() const {
  std::vector<KVCacheBlock*> ret;
  if (fake_free_list_head.next_free_block == nullptr) {
    throw std::runtime_error(
        "next_free_block of fake_free_list_head should always exist");
  }
  // Start from the first block.
  KVCacheBlock* curr_block = fake_free_list_head.next_free_block;
  // As long as next_free_block is available, we haven't reached the fake tail.
  while (curr_block->next_free_block != nullptr) {
    ret.push_back(curr_block);
    curr_block = curr_block->next_free_block;
  }
  return ret;
}

namespace {

// vllm/utils/mem_utils.py:30-31 format_gib: `f"{round(b / GiB_bytes, 2)}"`.
// Fixed 2 decimals rather than Python's float repr (which drops a trailing
// zero); the value is identical, only "1.50" vs "1.5" differs.
std::string FormatGiB(int64_t bytes) {
  const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
  std::array<char, 64> buf{};
  std::snprintf(buf.data(), buf.size(), "%.2f", gib);
  return std::string(buf.data());
}

// The vllm.cpp-specific half of the remediation advice. Upstream points at
// `gpu_memory_utilization`, which our loader does not implement yet
// (model_loader.cpp ResolveNumBlocks step 3, TODO ROAD-V1-MEM M3), so name the
// knobs that do size the pool here. Additive: upstream's sentence is kept
// verbatim ahead of it.
constexpr const char* kLocalRemediation =
    " In vllm.cpp the KV pool is sized by `--num-blocks` / `--kv-cache-memory`"
    " (gpu_memory_utilization profiling is not ported yet), so raise one of"
    " those or lower `--max-model-len`.";

constexpr const char* kConservingMemoryDoc =
    " See https://docs.vllm.ai/en/latest/configuration/conserving_memory/"
    " for more details.";

}  // namespace

int64_t host_available_memory_bytes() {
  // The pool the allocation actually draws from. On a unified-memory device
  // (GB10, Jetson Thor) device allocations come out of exactly this, which is why
  // an oversized state does not fail cleanly there: with vm.overcommit_memory=1
  // and no swap the kernel promises pages it cannot back and the BOX dies.
  // MemAvailable (not MemFree) is the kernel's own estimate of what can be handed
  // out without swapping, which is the honest denominator here.
  //
  // Returns 0 when unreadable; callers treat 0 as "unknown" and do not refuse,
  // because an unknown budget must not become a false refusal.
  std::FILE* f = std::fopen("/proc/meminfo", "r");
  if (f == nullptr) return 0;
  char line[256];
  int64_t kb = 0;
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    if (std::sscanf(line, "MemAvailable: %" SCNd64 " kB", &kb) == 1) break;
  }
  std::fclose(f);
  return kb > 0 ? kb * 1024 : 0;
}

int64_t recurrent_state_bytes(const KVCacheConfig& kv_cfg, int max_num_seqs) {
  // Mirrors what the runner allocates (runner.cpp:449-451,670-679): one state
  // row per sequence slot, and under speculation num_spec+1 CONSECUTIVE slots
  // per sequence for the k+1 draft-timestep snapshots the recurrent rollback
  // selects among. A group's page_size_bytes() is exactly one slot's conv+ssm
  // bytes for ONE layer, so the layer count multiplies it.
  if (max_num_seqs <= 0) return 0;
  int64_t total = 0;
  for (const KVCacheGroupSpec& group : kv_cfg.kv_cache_groups) {
    if (group.kv_cache_spec == nullptr) continue;
    if (group.kv_cache_spec->kind() != KVCacheSpecKind::kMamba) continue;
    const auto* mamba = static_cast<const MambaSpec*>(group.kv_cache_spec.get());
    const int64_t slots_per_seq =
        static_cast<int64_t>(1 + mamba->num_speculative_blocks);
    const int64_t layers = static_cast<int64_t>(group.layer_names.size());
    const int64_t prefix_slots = kv_cfg.recurrent_prefix_snapshots ? kv_cfg.recurrent_prefix_snapshots->capacity() : 0;
    total += mamba->page_size_bytes() * layers *
             (static_cast<int64_t>(max_num_seqs) * slots_per_seq + prefix_slots);
  }
  return total;
}

void check_enough_state_memory(int64_t available_memory, int64_t needed_memory,
                               int max_num_seqs, int num_spec) {
  // No recurrent state: nothing to run out of, and checking would refuse a
  // configuration that works (the `if kv_cache_spec:` guard upstream keeps for
  // the same reason, kv_cache_utils.py:872-878).
  if (needed_memory <= 0) return;
  if (needed_memory <= available_memory) return;

  const int slots = num_spec + 1;
  std::string msg =
      "To serve " + std::to_string(max_num_seqs) +
      " concurrent sequences, " + FormatGiB(needed_memory) +
      " GiB of recurrent (Mamba/GDN) state is needed, which is larger than the "
      "available " + FormatGiB(available_memory) + " GiB.";
  if (num_spec > 0) {
    msg += " Speculative decoding widens that state to " + std::to_string(slots) +
           " snapshot slots per sequence (num_speculative_tokens=" +
           std::to_string(num_spec) + " + 1), so it costs " +
           std::to_string(slots) + "x the non-speculative state.";
  }
  msg +=
      " Reduce --max-num-seqs, lower num_speculative_tokens, or run without "
      "speculative decoding.";
  throw std::invalid_argument(msg);
}

void check_enough_kv_cache_memory(int64_t available_memory,
                                  int64_t needed_memory, int64_t max_model_len,
                                  int64_t estimated_max_model_len) {
  // kv_cache_utils.py:757-765.
  if (available_memory <= 0) {
    throw std::invalid_argument(
        std::string(
            "No available memory for the cache blocks. Try increasing "
            "`gpu_memory_utilization` when initializing the engine (this flag "
            "also controls CPU memory reservation on the CPU backend, despite "
            "its name).") +
        kLocalRemediation + kConservingMemoryDoc);
  }

  // kv_cache_utils.py:769-788.
  if (needed_memory > available_memory) {
    std::string estimated_msg;
    if (estimated_max_model_len > 0) {
      estimated_msg = "Based on the available memory, the estimated maximum "
                      "model length is " +
                      std::to_string(estimated_max_model_len) + ". ";
    }
    throw std::invalid_argument(
        "To serve at least one request with the model's max seq len (" +
        std::to_string(max_model_len) + "), (" + FormatGiB(needed_memory) +
        " GiB KV cache is needed, which is larger than the available KV cache "
        "memory (" +
        FormatGiB(available_memory) + " GiB). " + estimated_msg +
        "Try increasing `gpu_memory_utilization` (which also controls CPU "
        "memory on the CPU backend) or decreasing `max_model_len` when "
        "initializing the engine." +
        kLocalRemediation + kConservingMemoryDoc);
  }
}

int64_t max_blocks_per_request(const KVCacheSpec& spec, int64_t max_model_len,
                               int max_num_batched_tokens) {
  // See kv_cache_utils.h for the anchors and the Mamba `none` deviation.
  const int64_t bs = spec.block_size;
  if (max_model_len <= 0 || bs <= 0) return 0;
  const int64_t full_blocks = (max_model_len + bs - 1) / bs;
  switch (spec.kind()) {
    case KVCacheSpecKind::kFullAttention:
    case KVCacheSpecKind::kMlaAttention:
      // kv_cache_interface.py:464-469 (DCP is 1 in this tree).
      return full_blocks;
    case KVCacheSpecKind::kSlidingWindow:
    case KVCacheSpecKind::kSlidingWindowMla:
      // kv_cache_interface.py:704-738, through the function the manager's
      // admission cap calls.
      return static_cast<const SlidingWindowSpec&>(spec)
          .max_admission_blocks_per_request(max_num_batched_tokens,
                                            static_cast<int>(max_model_len));
    case KVCacheSpecKind::kChunkedLocalAttention:
      // kv_cache_interface.py:658-681.
      return static_cast<const ChunkedLocalAttentionSpec&>(spec)
          .max_admission_blocks_per_request(max_num_batched_tokens,
                                            static_cast<int>(max_model_len));
    case KVCacheSpecKind::kMamba: {
      // kv_cache_interface.py:883-894.
      const auto& mamba = static_cast<const MambaSpec&>(spec);
      const int64_t k = mamba.num_speculative_blocks;
      if (mamba.mamba_cache_mode == "align") return 2 + k;
      // "all", and "none" at this tree's block size (see the header).
      return full_blocks + k;
    }
    case KVCacheSpecKind::kEncoderOnlyAttention:
      // kv_cache_interface.py:923-925: encoder-only layers keep no KV.
      return 0;
    case KVCacheSpecKind::kSinkFullAttention:
    case KVCacheSpecKind::kCrossAttention:
    case KVCacheSpecKind::kUnknown:
      break;
  }
  throw std::invalid_argument(
      "max_blocks_per_request: the per-request KV usage of this spec kind is not "
      "ported, so the startup check cannot count it (FIX-KV-POOL-MIN-FIT)");
}

int64_t max_memory_usage_bytes_from_groups(const KVCacheConfig& kv_cfg,
                                           int64_t max_model_len,
                                           int max_num_batched_tokens) {
  // kv_cache_utils.py:2029-2058: each group claims its own blocks from the one
  // shared pool, so a request consumes the SUM of the per-group counts.
  const int64_t bytes_per_block = KVBytesPerBlock(kv_cfg);
  if (bytes_per_block <= 0 || max_model_len <= 0) return 0;
  int64_t total_blocks = 0;
  for (const KVCacheGroupSpec& group : kv_cfg.kv_cache_groups) {
    if (group.kv_cache_spec == nullptr) continue;
    total_blocks += max_blocks_per_request(*group.kv_cache_spec, max_model_len,
                                           max_num_batched_tokens);
  }
  return bytes_per_block * total_blocks;
}

int64_t estimate_max_model_len_from_groups(const KVCacheConfig& kv_cfg,
                                           int64_t available_memory,
                                           int64_t original_max_model_len,
                                           int max_num_batched_tokens) {
  // kv_cache_utils.py:2061-2093.
  const auto fits = [&](int64_t len) {
    return max_memory_usage_bytes_from_groups(kv_cfg, len,
                                              max_num_batched_tokens) <=
           available_memory;
  };
  int64_t left = 1;
  int64_t right = original_max_model_len;
  if (right < left || !fits(left)) return 0;
  int64_t result = 1;
  while (left <= right) {
    const int64_t mid = left + (right - left) / 2;
    if (fits(mid)) {
      result = mid;
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return result;
}

void check_enough_kv_cache_memory(const KVCacheConfig& kv_cfg,
                                  int64_t available_memory,
                                  int64_t max_model_len,
                                  int max_num_batched_tokens) {
  // kv_cache_utils.py:933-967. `if kv_cache_spec:` -- a config with no paged KV
  // (attention-free, or pure Mamba/GDN, where KVBytesPerBlock is 0) has nothing
  // to size here.
  const int64_t bytes_per_block = KVBytesPerBlock(kv_cfg);
  if (bytes_per_block <= 0) return;
  // :953-961: plan against the blocks BlockPool does not hold back as the null
  // block (block_pool.cpp: block 0 is popped at construction).
  const int64_t check_memory = available_memory - bytes_per_block;
  check_enough_kv_cache_memory(
      check_memory,
      max_memory_usage_bytes_from_groups(kv_cfg, max_model_len,
                                         max_num_batched_tokens),
      max_model_len,
      estimate_max_model_len_from_groups(kv_cfg, check_memory, max_model_len,
                                         max_num_batched_tokens));
}

int64_t auto_fit_max_model_len(const KVCacheConfig& kv_cfg,
                               int64_t available_memory,
                               int64_t derived_max_model_len,
                               int max_num_batched_tokens) {
  // kv_cache_utils.py:2113-2120: an attention-free model has no KV to fit, so
  // the derived length stands.
  const int64_t bytes_per_block = KVBytesPerBlock(kv_cfg);
  if (bytes_per_block <= 0) {
    return derived_max_model_len;
  }
  // :2304-2311: the null block is reserved before auto-fit, as before the check.
  const int64_t check_memory = available_memory - bytes_per_block;
  const int64_t auto_fit_max = estimate_max_model_len_from_groups(
      kv_cfg, check_memory, derived_max_model_len, max_num_batched_tokens);
  // :2133-2137.
  if (auto_fit_max <= 0) {
    throw std::invalid_argument(
        "Cannot auto-fit max_model_len: not enough GPU memory available to "
        "serve even a single token. Try increasing `gpu_memory_utilization`." +
        std::string(kLocalRemediation));
  }
  // :2139-2157: keep the full context when it fits, else reduce to what does.
  if (auto_fit_max >= derived_max_model_len) {
    return derived_max_model_len;
  }
  return auto_fit_max;
}

}  // namespace vllm::v1
