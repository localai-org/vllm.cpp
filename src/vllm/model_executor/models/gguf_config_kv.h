// vllm.cpp ORIGINAL — the KV readers shared by the PARSE-ONLY GGUF config
// builders of row NEW-CANDIDATE-PARSERS (olala, hy_v4_flash). The kolibri1 and
// flint TUs each carry private copies; this row's builders share one because
// the carry-verbatim path (KvToJson) is new with this row and two copies of it
// could drift. Model-private under src/.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_GGUF_CONFIG_KV_H_
#define VLLM_MODEL_EXECUTOR_MODELS_GGUF_CONFIG_KV_H_

#include <cstdint>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/gguf_reader.h"

namespace vllm {
namespace gguf_config {

inline int64_t KvInt(const GgufValue& v, const std::string& key) {
  switch (v.TypeId()) {
    case kGgufU8: return std::get<uint8_t>(v.v);
    case kGgufI8: return std::get<int8_t>(v.v);
    case kGgufU16: return std::get<uint16_t>(v.v);
    case kGgufI16: return std::get<int16_t>(v.v);
    case kGgufU32: return std::get<uint32_t>(v.v);
    case kGgufI32: return std::get<int32_t>(v.v);
    case kGgufU64: return static_cast<int64_t>(std::get<uint64_t>(v.v));
    case kGgufI64: return std::get<int64_t>(v.v);
    case kGgufBool: return std::get<bool>(v.v) ? 1 : 0;
    default:
      throw std::runtime_error("gguf config: key " + key +
                               " is not an integer");
  }
}

inline double KvFloat(const GgufValue& v, const std::string& key) {
  if (v.TypeId() == kGgufF32) return std::get<float>(v.v);
  if (v.TypeId() == kGgufF64) return std::get<double>(v.v);
  return static_cast<double>(KvInt(v, key));
}

inline int64_t ReqInt(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  if (v == nullptr) {
    throw std::runtime_error("gguf config: missing metadata key " + key);
  }
  return KvInt(*v, key);
}

inline int64_t OptInt(const GgufFile& g, const std::string& key,
                      int64_t fallback) {
  const GgufValue* v = g.FindKv(key);
  return v == nullptr ? fallback : KvInt(*v, key);
}

inline double ReqFloat(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  if (v == nullptr) {
    throw std::runtime_error("gguf config: missing metadata key " + key);
  }
  return KvFloat(*v, key);
}

// One GGUF metadata value as JSON, arrays included — the carry-verbatim
// representation. F32 carries as double so no bit value is re-rounded.
inline nlohmann::json KvToJson(const GgufValue& v) {
  switch (v.TypeId()) {
    case kGgufU8: return std::get<uint8_t>(v.v);
    case kGgufI8: return std::get<int8_t>(v.v);
    case kGgufU16: return std::get<uint16_t>(v.v);
    case kGgufI16: return std::get<int16_t>(v.v);
    case kGgufU32: return std::get<uint32_t>(v.v);
    case kGgufI32: return std::get<int32_t>(v.v);
    case kGgufF32: return static_cast<double>(std::get<float>(v.v));
    case kGgufBool: return std::get<bool>(v.v);
    case kGgufString: return std::get<std::string>(v.v);
    case kGgufU64: return std::get<uint64_t>(v.v);
    case kGgufI64: return std::get<int64_t>(v.v);
    case kGgufF64: return std::get<double>(v.v);
    case kGgufArray: {
      nlohmann::json arr = nlohmann::json::array();
      for (const GgufValue& e : std::get<GgufArray>(v.v).elems) {
        arr.push_back(KvToJson(e));
      }
      return arr;
    }
    default:
      throw std::runtime_error("gguf config: unsupported value type " +
                               std::to_string(v.TypeId()));
  }
}

// Every KV NOT in `mapped`, under its GGUF key spelling, into
// `raw["gguf_unmapped"]` plus a count. `mapped` is the EXACT set the builder
// consumed; the tests mutate it to prove the carry is real.
inline void CarryUnmapped(const GgufFile& g,
                          std::initializer_list<const char*> mapped,
                          HfConfig& c) {
  nlohmann::json unmapped = nlohmann::json::object();
  int64_t count = 0;
  for (const auto& kv : g.Kvs()) {
    bool is_mapped = false;
    for (const char* k : mapped) {
      if (kv.first == k) {
        is_mapped = true;
        break;
      }
    }
    if (is_mapped) continue;
    unmapped[kv.first] = KvToJson(kv.second);
    ++count;
  }
  c.raw["gguf_unmapped"] = unmapped;
  c.raw["gguf_unmapped_count"] = count;
}

// True when the named tensor exists (per-layer presence checks).
inline bool HasTensor(const GgufFile& g, const std::string& name) {
  for (const GgufTensorInfo& t : g.Tensors()) {
    if (t.name == name) return true;
  }
  return false;
}

}  // namespace gguf_config
}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_GGUF_CONFIG_KV_H_
