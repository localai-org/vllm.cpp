#pragma once
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>

namespace exl3_test {
// Test-only admission; never initializes a GPU or changes a service. CTest
// invokes exit-code based external cases separately. Mixed executables use
// the per-case helpers instead. Required qualification opts into failure.
inline bool RequireExternalArtifacts() {
  const char* mode = std::getenv("EXL3_REQUIRE_ARTIFACTS");
  if (!mode || std::string_view(mode) == "0") return false;
  if (std::string_view(mode) == "1") return true;
  throw std::runtime_error("EXL3_REQUIRE_ARTIFACTS must be0 or1");
}
// Mixed doctest executables must skip the CASE, never exit the process and
// suppress unrelated synthetic cases. Invalid modes enter the case and fail.
inline bool OptionalExternalEnvironmentMissing(std::initializer_list<const char*> names) {
  const char* mode=std::getenv("EXL3_REQUIRE_ARTIFACTS");
  if (mode && std::string_view(mode)!="0") return false;
  for (const auto* name:names) {
    const char* value=std::getenv(name);
    if (!value || !*value) return true;
  }
  return false;
}
inline const char* CaseExternalEnvironment(const char* name) {
  (void)RequireExternalArtifacts();
  const char* value=std::getenv(name);
  if (!value || !*value) throw std::runtime_error(std::string("missing EXL3 test artifact: set ")+name);
  return value;
}
[[noreturn]] inline void MissingExternalArtifact(const std::string& reason) {
  const bool required = RequireExternalArtifacts();
  std::cerr << (required ? "ERROR: " : "SKIP: ") << "missing EXL3 test artifact: "
            << reason << '\n';
  std::exit(required ? 1 : 77);
}
inline const char* ExternalEnvironment(const char* name) {
  (void)RequireExternalArtifacts();
  const char* value = std::getenv(name);
  if (!value || !*value) MissingExternalArtifact(std::string("set ") + name);
  return value;
}
inline void ExternalPath(const std::filesystem::path& path, bool directory = false) {
  (void)RequireExternalArtifacts();
  std::error_code error;
  const auto status = std::filesystem::status(path, error);
  if (error == std::errc::no_such_file_or_directory ||
      (!error && !std::filesystem::exists(status)))
    MissingExternalArtifact(path.string());
  if (error) throw std::runtime_error("cannot inspect artifact: " + path.string() + ": " + error.message());
  if (directory ? !std::filesystem::is_directory(status) : !std::filesystem::is_regular_file(status))
    throw std::runtime_error("wrong artifact type: " + path.string());
}
}  // namespace exl3_test
