#pragma once
#include "exl3_external_artifacts.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include <fstream>
#include <nlohmann/json.hpp>

namespace native_vision_test {
inline void VerifyBoundary(const std::filesystem::path& root,const std::string& name,
                           const std::string& expected) {
  if (std::filesystem::path(name).filename()!=name || expected.size()!=64)
    throw std::runtime_error("invalid vision reference filename/hash");
  const auto path=root/name;
  if (std::filesystem::file_size(path)>256*1024*1024)
    throw std::runtime_error("vision reference exceeds diagnostic budget");
  std::ifstream stream(path,std::ios::binary);
  if (!stream.good()) throw std::runtime_error("missing vision reference: "+path.string());
  const std::string bytes{std::istreambuf_iterator<char>(stream),std::istreambuf_iterator<char>()};
  const auto hash=vllm::v1::sha256_bytes(bytes);
  constexpr char hex[]="0123456789abcdef";
  std::string actual;actual.reserve(64);
  for (unsigned char byte:hash) {actual+=hex[byte>>4];actual+=hex[byte&15];}
  if (actual!=expected) throw std::runtime_error("vision reference SHA-256 differs: "+name);
}
inline void VerifyReferences(const std::filesystem::path& root,const nlohmann::json& value) {
  if (value.is_object()) {
    if (value.contains("file") && value.contains("sha256"))
      VerifyBoundary(root,value.at("file").get<std::string>(),value.at("sha256").get<std::string>());
    if (value.contains("expected") && value.at("expected").is_string() && value.contains("expected_sha256"))
      VerifyBoundary(root,value.at("expected").get<std::string>(),value.at("expected_sha256").get<std::string>());
    for (const auto& child:value.items()) VerifyReferences(root,child.value());
  } else if (value.is_array()) for (const auto& child:value) VerifyReferences(root,child);
}
inline nlohmann::json ReferenceDocument(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream.good()) throw std::runtime_error("missing vision reference manifest: "+path.string());
  auto value=nlohmann::json::parse(stream);
  VerifyReferences(path.parent_path(),value); // Admission before any GPU queue.
  return value;
}
} // namespace native_vision_test
