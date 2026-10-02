// MODEL-MM-nano-nemotron-vl-nemotron-h-nano-vl-v2: DO BOTH PRODUCTION ENTRY
// POINTS HAND THE NEMOTRON NANO VL SEAM THE ENGINE'S max_model_len?
//
// `test_nano_nemotron_vl_mm_chat.cpp` builds the install context field by field
// and proves everything downstream of it. It never touches the two production
// lines that fill the one field this row added to `MultiModalChatContext`:
//
//   server_main.cpp:  mm_ctx.max_model_len = loaded->max_model_len();
//   vllm_c.cpp:       mm_ctx.max_model_len = engine->loaded->max_model_len();
//
// Delete either and that suite stays green, because it fills the context in
// itself. Production then installs a REFUSING seam, because the dynamic tiler's
// token budget is defined against max_model_len
// (processors/nano_nemotron_vl.py:330-331 @ e126687a9a), and every image
// request gets HTTP 400. That is the UNPASSED PARAMETER shape
// `.agents/reachability.md` names. This file is the gate for those two lines,
// and each case enters through the real entry point with a real LoadedEngine
// built from a tiny checkpoint directory.
//
// The two gates are not equally strong. The C-ABI side proves the field
// carries the ENGINE's value: at max_model_len 16 the tiler shrinks the image
// grid, so a constant in vllm_c.cpp reddens it. The server side proves only
// that the field is SUPPLIED (a positive value, so the seam is wired). The
// server exits on the unbindable port before it serves a request, and the
// install announcement does not print the budget, so no server-side effect of
// the value is observable here. Passing --max-model-len would change nothing
// this case can see.
//
// The server case re-execs this binary into `VllmServerMain` exactly as
// `test_serve_deepseek_v4_mm.cpp` does (ParseArgs can `std::exit`, and the
// unbindable port makes the server exit right after the install announcement).
// The C-ABI case calls `vllm_engine_load` and `vllm_chat` in process.
#include <doctest/doctest.h>

#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm.h"
#include "vllm/entrypoints/openai/server_main.h"
#include "vllm/models/nano_nemotron_vl_tiny_checkpoint.h"

namespace {

constexpr const char* kArch = "NemotronH_Nano_Omni_Reasoning_V3";

// Printed by VllmServerMain AFTER ParseArgs returns.
constexpr const char* kPostParseBanner = "server: request logging";
// The two install outcomes, in `mm_chat_registry.cpp`'s own words.
constexpr const char* kWired = "multimodal chat seam wired for architecture";
constexpr const char* kUnavailable = "multimodal chat seam UNAVAILABLE for architecture";
// Binding it needs privileges this test does not have, so the server exits
// after the install instead of accepting connections.
constexpr const char* kUnbindablePort = "1";

bool Contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

// The tiny checkpoint written as a model directory, removed on scope exit.
class ModelDir {
 public:
  ModelDir() {
    static int c = 0;
    path_ = std::filesystem::temp_directory_path() /
            ("nnvl_serve_" + std::to_string(::getpid()) + "_" + std::to_string(c++));
    nnvl_tiny::WriteModelDir(path_);
  }
  ~ModelDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  std::string path() const { return path_.string(); }

 private:
  std::filesystem::path path_;
};

struct ChildRun {
  std::string output;  // stdout + stderr, combined
  int status = -1;
};

ChildRun RunServer(const std::string& serve_args) {
  // Resolve our own path in the PARENT: popen runs under /bin/sh.
  char exe[4096];
  const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  REQUIRE(n > 0);
  exe[n] = '\0';
  const std::string cmd = "VLLM_TEST_SERVE_ARGS='" + serve_args + "' " + std::string(exe) +
                          " --no-skip --test-case='serve_nano_nemotron_vl_mm_child' 2>&1";
  FILE* pipe = ::popen(cmd.c_str(), "r");
  REQUIRE(pipe != nullptr);
  ChildRun run;
  std::array<char, 4096> buf{};
  while (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe) != nullptr) {
    run.output += buf.data();
  }
  const int closed = ::pclose(pipe);
  REQUIRE(closed != -1);
  run.status = WIFEXITED(closed) ? WEXITSTATUS(closed) : -1;
  return run;
}

std::vector<std::string> SplitOnSpaces(const std::string& text) {
  std::vector<std::string> out;
  std::string current;
  for (const char c : text) {
    if (c == ' ') {
      if (!current.empty()) out.push_back(current);
      current.clear();
    } else {
      current.push_back(c);
    }
  }
  if (!current.empty()) out.push_back(current);
  return out;
}

// A square raw-RGB data URI, the one container DefaultImageCodec decodes
// without a codec library (chat_mm.cpp).
std::string RawRgbDataUri(int64_t side) {
  static const char* kB64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::vector<uint8_t> rgb(static_cast<size_t>(side * side * 3));
  for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = static_cast<uint8_t>((i * 7) % 251);
  std::string out;
  for (size_t i = 0; i < rgb.size(); i += 3) {
    const uint32_t v = (static_cast<uint32_t>(rgb[i]) << 16) |
                       (static_cast<uint32_t>(rgb[i + 1]) << 8) |
                       static_cast<uint32_t>(rgb[i + 2]);
    out.push_back(kB64[(v >> 18) & 63]);
    out.push_back(kB64[(v >> 12) & 63]);
    out.push_back(kB64[(v >> 6) & 63]);
    out.push_back(kB64[v & 63]);
  }
  return "data:image/x-raw-rgb;base64," + out;
}

}  // namespace

// The CHILD case, filtered out of a normal run and executed only when a parent
// re-execs it by name.
TEST_CASE("serve_nano_nemotron_vl_mm_child" * doctest::skip()) {
  const char* raw = std::getenv("VLLM_TEST_SERVE_ARGS");
  REQUIRE(raw != nullptr);
  std::vector<std::string> args{"vllm-server"};
  for (std::string& token : SplitOnSpaces(raw)) args.push_back(std::move(token));
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (std::string& arg : args) argv.push_back(arg.data());
  const int rc = vllm::entrypoints::openai::VllmServerMain(static_cast<int>(argv.size()),
                                                           argv.data());
  std::cout << "SERVE_RC=" << rc << "\n" << std::flush;
  std::exit(0);
}

// Delete `mm_ctx.max_model_len = loaded->max_model_len()` in server_main.cpp
// and this reddens: the factory refuses a context without max_model_len, the
// install announces UNAVAILABLE and names the missing field. It proves the
// field is supplied, not which value it carries (see the file header).
TEST_CASE("serve: the Nemotron Nano Omni image seam is wired with a supplied max_model_len") {
  const ModelDir dir;
  const ChildRun run = RunServer("--model " + dir.path() + " --port " + kUnbindablePort);
  INFO("child output:\n" << run.output);

  CHECK(Contains(run.output, kPostParseBanner));
  CHECK(Contains(run.output, kWired));
  CHECK(Contains(run.output, kArch));
  CHECK(Contains(run.output, "Nemotron Nano VL dynamic-resolution processor"));
  CHECK_FALSE(Contains(run.output, kUnavailable));
  CHECK_FALSE(Contains(run.output, "max_model_len was not supplied"));
  CHECK(run.status == 0);
}

// One image request through `vllm_engine_load` + `vllm_chat` on the tiny
// checkpoint. `max_model_len` <= 0 keeps the engine default, which is the
// checkpoint's max_position_embeddings (128).
namespace {

struct CapiChat {
  vllm_status status = VLLM_OK;
  std::string error;
  nlohmann::json out;
};

CapiChat CapiChatOneImage(int32_t max_model_len) {
  const ModelDir dir;
  const std::string path = dir.path();
  vllm_model_params mp = vllm_model_params_default();
  mp.model_path = path.c_str();
  mp.max_model_len = max_model_len;
  vllm_engine* eng = nullptr;
  const vllm_status load = vllm_engine_load(&mp, &eng);
  {
    const char* e = vllm_last_error();
    INFO("load error: " << (e == nullptr ? "" : e));
    REQUIRE(load == VLLM_OK);
  }
  REQUIRE(eng != nullptr);

  nlohmann::json req;
  req["messages"] = nlohmann::json::array(
      {{{"role", "user"},
        {"content", nlohmann::json::array({{{"type", "text"}, {"text", "what is it"}},
                                           {{"type", "image_url"},
                                            {"image_url", {{"url", RawRgbDataUri(48)}}}}})}}});
  req["temperature"] = 0;
  req["max_tokens"] = 2;
  const std::string body = req.dump();
  char* response = nullptr;
  CapiChat run;
  run.status = vllm_chat(eng, body.c_str(), &response);
  run.error = vllm_last_error() == nullptr ? std::string() : vllm_last_error();
  if (response != nullptr) {
    run.out = nlohmann::json::parse(response);
    vllm_string_free(response);
  }
  vllm_engine_free(eng);
  return run;
}

}  // namespace

// Delete `mm_ctx.max_model_len = engine->loaded->max_model_len()` in
// vllm_c.cpp and this reddens: the install wires the refusing seam, and the
// image request comes back VLLM_ERR_INVALID_ARGUMENT instead of a completion.
TEST_CASE("capi: vllm_chat answers an image request on the Nemotron Nano Omni checkpoint") {
  const CapiChat run = CapiChatOneImage(0);
  INFO("chat error: " << run.error);
  CHECK(run.status == VLLM_OK);
  CHECK_FALSE(Contains(run.error, "max_model_len"));
  REQUIRE(run.out.is_object());
  CHECK(run.out.at("object") == "chat.completion");
  // The image was expanded to <img> + 4 x <image> + </img>: a 48x48 image is a
  // 4x4 patch grid under the tiny config, 4 rows after the pixel shuffle. The
  // text part is "<image>\nwhat is it" (11 bytes after the one <image>), so the
  // prompt is 1 + 4 + 1 + 11 tokens.
  CHECK(run.out.at("usage").at("prompt_tokens") == 17);
}

// Setting the field is not enough: it must carry the ENGINE's value. Replace
// the vllm_c.cpp assignment with any constant (say `1 << 20`) and the case
// above stays green, because 128 already leaves the 48x48 image its full 4x4
// grid. Here the engine runs at max_model_len 16, so upstream's budget
// `max_model_len - text_prompt_length - 4` (processors/nano_nemotron_vl.py:
// 330-331 @ e126687a9a) is 16 - 11 - 4 = 1 post-shuffle token, 4 patches: the
// tiler shrinks the grid to 2x2, which is ONE <image> after the pixel shuffle,
// and the prompt is 1 + 1 + 1 + 11 = 14 tokens. With the constant the tiler
// keeps 4x4, the prompt is 17 tokens, and a 16-token engine refuses it.
TEST_CASE("capi: the tiler budgets the image against the engine's max_model_len") {
  const CapiChat run = CapiChatOneImage(16);
  INFO("chat error: " << run.error);
  CHECK(run.status == VLLM_OK);
  REQUIRE(run.out.is_object());
  CHECK(run.out.at("object") == "chat.completion");
  CHECK(run.out.at("usage").at("prompt_tokens") == 14);
}
