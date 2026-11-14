// vllm.cpp original. `vllm-video-studio` — a self-contained console for MiniMax-H3
// video generation.
//
// WHY THIS IS ITS OWN BINARY. `examples/server` is the OpenAI-compatible API
// surface; a browser console does not belong in it, and neither does
// engine-lifecycle plumbing added to the shared library for one consumer's
// benefit. This example owns its UI, its job queue and its HTTP endpoints, and
// touches nothing the API server uses.
//
// ARCH-ONE-SURFACE: a THIN CLIENT of the public C ABI. It includes `vllm.h` and
// nothing from the internal tree. Everything it does -- loading a checkpoint,
// generating, swapping the loaded weights, building the ffmpeg argv -- is an ABI
// call, which is the point: if the console can do it, an FFI consumer can too.
//
// The ONE process spawn (ffmpeg) lives here rather than in the library, matching
// the developer-ratified boundary: `vllm_video_mux_argv` builds the argv and the
// CALLER exec's it.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#if defined(_WIN32)
#include <process.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include "vllm.h"

namespace {

using json = nlohmann::json;

// ── the loaded checkpoint ────────────────────────────────────────────────────
// H3 ships two partitions that each REFUSE the other's tasks, so a studio pinned
// to one checkpoint could only ever reach two of the three modalities. The slot
// makes the loaded weights swappable at runtime; `vllm_video_engine_free` +
// `vllm_video_engine_load` is the whole mechanism, both already on the ABI.
struct Engine {
  std::mutex m;
  vllm_video_engine* handle = nullptr;
  std::string dit, partition, device;
  int dequant_bf16 = 1;
  // Everything except the DiT is shared between partitions, so a swap reuses it.
  vllm_video_model_params base{};
  std::string encoder, tokenizer, vvae, vvae_cfg, avae, avae_cfg, embeds;
  std::vector<std::string> lora_keys, lora_values;
  std::vector<const char*> extra_key_ptrs, extra_val_ptrs;
};
Engine g_engine;

struct Job {
  std::string id, prompt, task, error, mp4;
  int w = 0, h = 0, frames = 0, steps = 0;
  std::string status = "queued";  // queued | running | succeeded | failed
  std::string first_frame, ref_image;  // temp PPM paths, owned by us
};
std::mutex g_jobs_m;
std::deque<Job> g_jobs;              // newest first
std::deque<std::string> g_pending;   // ids, FIFO
std::condition_variable g_wake;
std::string g_workdir = "/tmp/h3studio";
std::string g_ffmpeg = "ffmpeg";
std::string g_ui_dir;
std::string g_models_dir;

std::string Now() {
  static std::atomic<uint64_t> n{0};
  return "job" + std::to_string(n.fetch_add(1));
}

Job* Find(const std::string& id) {  // caller holds g_jobs_m
  for (Job& j : g_jobs)
    if (j.id == id) return &j;
  return nullptr;
}

// Spawn ffmpeg on the argv the LIBRARY built. No shell: the prompt is attacker-
// adjacent text and a shell here would be an injection surface.
bool RunMux(char** argv, int argc, std::string* err) {
  if (argv == nullptr || argc <= 0) {
    *err = "the library returned no mux argv";
    return false;
  }
  std::vector<char*> a(argv, argv + argc);
  std::string ff = g_ffmpeg;
  a[0] = ff.data();
  a.push_back(nullptr);
  int code = -1;
#if defined(_WIN32)
  // The child inherits this process's stderr. The POSIX-only /dev/null redirect
  // below has no MSVCRT spawn equivalent, so ffmpeg diagnostics reach the console.
  code = static_cast<int>(_spawnvp(_P_WAIT, a[0], a.data()));
#else
  const pid_t pid = fork();
  if (pid < 0) {
    *err = "fork failed";
    return false;
  }
  if (pid == 0) {
    if (freopen("/dev/null", "w", stderr) == nullptr) { /* keep the parent tidy */ }
    execvp(a[0], a.data());
    _exit(127);
  }
  int st = 0;
  waitpid(pid, &st, 0);
  code = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
#endif
  if (code == 0) return true;
  *err = "ffmpeg exited " + std::to_string(code) + " (is it installed? --ffmpeg PATH)";
  return false;
}

// ── the worker: ONE render at a time ─────────────────────────────────────────
// Serialised deliberately. These checkpoints are tens of GB resident and this
// class of box OOMs -- and an OOM takes the machine down, not just the request.
void Worker() {
  for (;;) {
    std::string id;
    {
      std::unique_lock<std::mutex> lk(g_jobs_m);
      g_wake.wait(lk, [] { return !g_pending.empty(); });
      id = g_pending.front();
      g_pending.pop_front();
      if (Job* j = Find(id)) j->status = "running";
    }

    Job snap;
    {
      std::lock_guard<std::mutex> lk(g_jobs_m);
      if (Job* j = Find(id)) snap = *j;
    }

    const std::string out_dir = g_workdir + "/" + id;
    std::string err;
    bool ok = false;

    vllm_video_engine* eng = nullptr;
    {
      std::lock_guard<std::mutex> lk(g_engine.m);
      eng = g_engine.handle;
    }
    if (eng == nullptr) {
      err = "no checkpoint is loaded";
    } else {
      vllm_video_params p = vllm_video_params_default();
      p.prompt = snap.prompt.c_str();
      p.width = snap.w;
      p.height = snap.h;
      p.num_frames = snap.frames;
      p.steps = snap.steps;
      p.output_dir = out_dir.c_str();
      if (!snap.first_frame.empty()) p.first_frame = snap.first_frame.c_str();
      if (!snap.ref_image.empty()) p.ref_image = snap.ref_image.c_str();

      vllm_video_result r{};
      if (vllm_video_generate(eng, &p, &r) != VLLM_OK) {
        err = vllm_last_error();
      } else {
        ok = RunMux(r.mux_argv, r.mux_argc, &err);
        vllm_video_result_free(&r);
      }
    }

    {
      std::lock_guard<std::mutex> lk(g_jobs_m);
      if (Job* j = Find(id)) {
        j->status = ok ? "succeeded" : "failed";
        j->error = err;
        if (ok) j->mp4 = out_dir + "/video.mp4";
      }
    }
  }
}

std::string ReadFile(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// A data: URL carrying a binary PPM, written to disk for the engine. The browser
// produces it: the engine vendors no image codec and no resampler, so the page
// converts and resizes before posting.
bool SaveDataUrl(const std::string& url, const std::string& path, std::string* err) {
  const auto comma = url.find(',');
  if (url.rfind("data:", 0) != 0 || comma == std::string::npos) {
    *err = "expected a data: URL";
    return false;
  }
  static const char* T =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  int rev[256];
  for (int i = 0; i < 256; ++i) rev[i] = -1;
  for (int i = 0; i < 64; ++i) rev[static_cast<unsigned char>(T[i])] = i;
  std::string out;
  int val = 0, bits = -8;
  for (size_t i = comma + 1; i < url.size(); ++i) {
    const int c = rev[static_cast<unsigned char>(url[i])];
    if (c < 0) continue;
    val = (val << 6) + c;
    bits += 6;
    if (bits >= 0) {
      out.push_back(static_cast<char>((val >> bits) & 0xFF));
      bits -= 8;
    }
  }
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    *err = "cannot write " + path;
    return false;
  }
  f.write(out.data(), static_cast<std::streamsize>(out.size()));
  return true;
}

// Checkpoints that are actually on disk. Asking a user to type an absolute path
// into a browser is a bad interface AND an unnecessary one: the weights were
// downloaded to a directory, so offer what is there.
//
// The partition is GUESSED from the filename because a community GGUF strips the
// release metadata and the two partitions are byte-structurally identical -- so
// there is nothing in the file to read. A wrong guess is not silently harmful:
// the engine refuses a task its partition does not serve.
std::string ScanModels() {
  json out = json::array();
  std::error_code ec;
  if (g_models_dir.empty()) return out.dump();
  for (const auto& e : std::filesystem::directory_iterator(g_models_dir, ec)) {
    if (ec) break;
    if (!e.is_regular_file()) continue;
    const std::string path = e.path().string();
    const std::string name = e.path().filename().string();
    if (name.size() < 5 || name.substr(name.size() - 5) != ".gguf") continue;
    std::string lower = name;
    for (char& c : lower) c = static_cast<char>(std::tolower(c));
    // The ENCODER is a gguf in the same directory and is not a DiT; naming it as
    // one would offer a checkpoint that cannot render.
    if (lower.find("qwen3vl") != std::string::npos || lower.find("enc") == 0) continue;
    std::string part;
    if (lower.find("ref2va") != std::string::npos) part = "ref2va";
    else if (lower.find("fl2va") != std::string::npos) part = "fl2va";
    std::error_code sec;
    const auto sz = std::filesystem::file_size(e.path(), sec);
    out.push_back({{"path", path},
                   {"name", name},
                   {"partition", part},
                   {"pruned", lower.find("pruned") != std::string::npos},
                   {"gib", sec ? 0.0 : static_cast<double>(sz) / (1024.0 * 1024.0 * 1024.0)}});
  }
  return out.dump();
}

std::string EngineStatus() {
  std::lock_guard<std::mutex> lk(g_engine.m);
  return json{{"loaded", g_engine.handle != nullptr},
              {"dit", g_engine.dit},
              {"partition", g_engine.partition},
              {"device", g_engine.device},
              {"dequant_bf16", g_engine.dequant_bf16 != 0}}
      .dump();
}

// UNLOAD FIRST. ~20 GB quantised, ~66 GB dequantised: holding the old set while
// building the new one is what OOMs the box. The cost is a window with nothing
// loaded, so a failed load leaves the slot EMPTY and says so rather than quietly
// serving the partition the caller just asked to leave.
std::string SwapEngine(const std::string& dit, const std::string& partition) {
  std::lock_guard<std::mutex> lk(g_engine.m);
  if (g_engine.handle != nullptr) {
    vllm_video_engine_free(g_engine.handle);
    g_engine.handle = nullptr;
  }
  vllm_video_model_params mp = g_engine.base;
  mp.dit_path = dit.c_str();
  mp.partition = partition.c_str();
  vllm_video_engine* fresh = nullptr;
  if (vllm_video_engine_load(&mp, &fresh) != VLLM_OK) return vllm_last_error();
  g_engine.handle = fresh;
  g_engine.dit = dit;
  g_engine.partition = partition;
  std::fprintf(stderr, "studio: loaded %s (%s)\n", dit.c_str(), partition.c_str());
  return std::string();
}

const char* Arg(int argc, char** argv, int& i) {
  if (i + 1 >= argc) {
    std::fprintf(stderr, "%s needs a value\n", argv[i]);
    std::exit(2);
  }
  return argv[++i];
}

}  // namespace

int main(int argc, char** argv) {
  int lora_count = 0;
  std::string host = "0.0.0.0";
  int port = 8080;
  g_ui_dir = "examples/video_studio/webui";

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--dit") g_engine.dit = Arg(argc, argv, i);
    else if (a == "--partition") g_engine.partition = Arg(argc, argv, i);
    else if (a == "--encoder") g_engine.encoder = Arg(argc, argv, i);
    else if (a == "--tokenizer") g_engine.tokenizer = Arg(argc, argv, i);
    else if (a == "--video-vae") g_engine.vvae = Arg(argc, argv, i);
    else if (a == "--video-vae-config") g_engine.vvae_cfg = Arg(argc, argv, i);
    else if (a == "--audio-vae") g_engine.avae = Arg(argc, argv, i);
    else if (a == "--audio-vae-config") g_engine.avae_cfg = Arg(argc, argv, i);
    else if (a == "--prompt-embeds") g_engine.embeds = Arg(argc, argv, i);
    else if (a == "--lora") {
      ++lora_count;
      const std::string suffix = lora_count == 1 ? "" : "_" + std::to_string(lora_count);
      g_engine.lora_keys.push_back("lora_path" + suffix);
      g_engine.lora_values.push_back(Arg(argc, argv, i));
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        g_engine.lora_keys.push_back("lora_strength" + suffix);
        g_engine.lora_values.push_back(argv[++i]);
      }
    }
    else if (a == "--lora-dir") {
      g_engine.lora_keys.push_back("lora_dir");
      g_engine.lora_values.push_back(Arg(argc, argv, i));
    }
    else if (a == "--device") g_engine.device = Arg(argc, argv, i);
    else if (a == "--keep-quant") g_engine.dequant_bf16 = 0;
    else if (a == "--dequant-bf16") g_engine.dequant_bf16 = 1;
    else if (a == "--workdir") g_workdir = Arg(argc, argv, i);
    else if (a == "--ffmpeg") g_ffmpeg = Arg(argc, argv, i);
    else if (a == "--ui") g_ui_dir = Arg(argc, argv, i);
    else if (a == "--models-dir") g_models_dir = Arg(argc, argv, i);
    else if (a == "--host") host = Arg(argc, argv, i);
    else if (a == "--port") port = std::atoi(Arg(argc, argv, i));
    else if (a == "--help" || a == "-h") {
      std::printf(
          "usage: %s --dit <f> --partition fl2va|ref2va --video-vae <f> --audio-vae <f>\n"
          "          [--encoder <f> --tokenizer <f>] [--video-vae-config <j>]\n"
          "          [--audio-vae-config <j>] [--prompt-embeds <f>]\n"
          "          [--device cpu|cuda] [--keep-quant|--dequant-bf16]\n"
          "          [--workdir DIR] [--ffmpeg PATH] [--ui DIR]\n"
          "          [--models-dir DIR]  offer every .gguf here in the picker\n"
          "          [--lora <adapter.safetensors> [STRENGTH]]  fused at load; 1.0\n"
          "          [--lora-dir DIR]  resolve <lora:name:strength> prompt tags\n"
          "          [--host H] [--port P]\n\n"
          "A browser console for MiniMax-H3 video generation: all three tasks,\n"
          "and the loaded checkpoint can be swapped without restarting.\n"
          "Open http://<host>:<port>/ once it is up.\n",
          argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "unknown flag: %s (try --help)\n", argv[i]);
      return 2;
    }
  }
  if (g_engine.device.empty()) g_engine.device = "cuda";
  if (g_models_dir.empty() && !g_engine.dit.empty())
    g_models_dir = std::filesystem::path(g_engine.dit).parent_path().string();

  g_engine.base = vllm_video_model_params_default();
  g_engine.base.encoder_path = g_engine.encoder.empty() ? nullptr : g_engine.encoder.c_str();
  g_engine.base.tokenizer_path = g_engine.tokenizer.empty() ? nullptr : g_engine.tokenizer.c_str();
  g_engine.base.video_vae_path = g_engine.vvae.c_str();
  g_engine.base.video_vae_config_path =
      g_engine.vvae_cfg.empty() ? nullptr : g_engine.vvae_cfg.c_str();
  g_engine.base.audio_vae_path = g_engine.avae.c_str();
  g_engine.base.audio_vae_config_path =
      g_engine.avae_cfg.empty() ? nullptr : g_engine.avae_cfg.c_str();
  g_engine.base.prompt_embeds_path = g_engine.embeds.empty() ? nullptr : g_engine.embeds.c_str();
  g_engine.base.device = g_engine.device == "cuda" ? 1 : 0;
  g_engine.base.dequant_bf16 = g_engine.dequant_bf16;
  if (!g_engine.lora_keys.empty()) {
    g_engine.extra_key_ptrs.clear();
    g_engine.extra_val_ptrs.clear();
    for (size_t i = 0; i < g_engine.lora_keys.size(); ++i) {
      g_engine.extra_key_ptrs.push_back(g_engine.lora_keys[i].c_str());
      g_engine.extra_val_ptrs.push_back(g_engine.lora_values[i].c_str());
    }
    g_engine.base.extra_keys = g_engine.extra_key_ptrs.data();
    g_engine.base.extra_values = g_engine.extra_val_ptrs.data();
    g_engine.base.n_extras = static_cast<int32_t>(g_engine.lora_keys.size());
  }

  if (!g_engine.dit.empty()) {
    std::fprintf(stderr, "studio: loading %s ...\n", g_engine.dit.c_str());
    const std::string err = SwapEngine(g_engine.dit, g_engine.partition);
    if (!err.empty()) {
      std::fprintf(stderr, "studio: %s\n", err.c_str());
      return 1;
    }
  } else {
    std::fprintf(stderr, "studio: no --dit; load one from the console\n");
  }

  std::thread(Worker).detach();

  httplib::Server srv;
  srv.set_payload_max_length(64ull * 1024 * 1024);  // data: URLs carry raw PPM

  srv.Get("/", [](const httplib::Request&, httplib::Response& res) {
    const std::string body = ReadFile(g_ui_dir + "/index.html");
    if (body.empty()) {
      res.status = 500;
      res.set_content("cannot read " + g_ui_dir + "/index.html (pass --ui DIR)", "text/plain");
      return;
    }
    res.set_content(body, "text/html; charset=utf-8");
  });

  srv.Get("/api/models", [](const httplib::Request&, httplib::Response& res) {
    res.set_content(ScanModels(), "application/json");
  });

  srv.Get("/api/engine", [](const httplib::Request&, httplib::Response& res) {
    res.set_content(EngineStatus(), "application/json");
  });

  srv.Post("/api/engine", [](const httplib::Request& req, httplib::Response& res) {
    std::string dit, part;
    try {
      const json b = json::parse(req.body);
      dit = b.value("dit", std::string());
      part = b.value("partition", std::string());
    } catch (const std::exception& e) {
      res.status = 400;
      res.set_content(json{{"error", std::string("bad JSON: ") + e.what()}}.dump(),
                      "application/json");
      return;
    }
    if (dit.empty() || part.empty()) {
      res.status = 400;
      res.set_content(json{{"error", "dit and partition are required"}}.dump(),
                      "application/json");
      return;
    }
    // Synchronous: minutes long, and returning early would leave the OLD
    // partition answering requests the caller believes went to the new one.
    const std::string err = SwapEngine(dit, part);
    if (!err.empty()) {
      res.status = 500;
      res.set_content(json{{"error", err}}.dump(), "application/json");
      return;
    }
    res.set_content(EngineStatus(), "application/json");
  });

  srv.Post("/api/render", [](const httplib::Request& req, httplib::Response& res) {
    json b;
    try {
      b = json::parse(req.body);
    } catch (const std::exception& e) {
      res.status = 400;
      res.set_content(json{{"error", std::string("bad JSON: ") + e.what()}}.dump(),
                      "application/json");
      return;
    }
    Job j;
    j.id = Now();
    j.prompt = b.value("prompt", std::string());
    j.task = b.value("task", std::string("t2va"));
    j.w = b.value("width", 512);
    j.h = b.value("height", 512);
    j.frames = b.value("num_frames", 124);
    j.steps = b.value("steps", 50);

    const std::string dir = g_workdir + "/" + j.id;
    std::error_code mkec;
    std::filesystem::create_directories(dir, mkec);  // no shell: the id is ours,
                                                     // but a shell here would be
                                                     // a standing injection risk
    if (mkec) {
      res.status = 500;
      res.set_content(json{{"error", "cannot create " + dir + ": " + mkec.message()}}.dump(),
                      "application/json");
      return;
    }
    const std::string img = b.value("image", std::string());
    if (!img.empty()) {
      const std::string path = dir + "/input.ppm";
      std::string err;
      if (!SaveDataUrl(img, path, &err)) {
        res.status = 400;
        res.set_content(json{{"error", err}}.dump(), "application/json");
        return;
      }
      if (j.task == "fl2va") j.first_frame = path;
      else if (j.task == "ref2va") j.ref_image = path;
    }
    {
      std::lock_guard<std::mutex> lk(g_jobs_m);
      g_jobs.push_front(j);
      while (g_jobs.size() > 60) g_jobs.pop_back();
      g_pending.push_back(j.id);
    }
    g_wake.notify_one();
    res.set_content(json{{"id", j.id}, {"status", "queued"}}.dump(), "application/json");
  });

  srv.Get("/api/jobs", [](const httplib::Request&, httplib::Response& res) {
    json out = json::array();
    std::lock_guard<std::mutex> lk(g_jobs_m);
    for (const Job& j : g_jobs)
      out.push_back({{"id", j.id},
                     {"prompt", j.prompt},
                     {"task", j.task},
                     {"width", j.w},
                     {"height", j.h},
                     {"frames", j.frames},
                     {"steps", j.steps},
                     {"status", j.status},
                     {"error", j.error}});
    res.set_content(out.dump(), "application/json");
  });

  srv.Get(R"(/api/video/(job\d+))", [](const httplib::Request& req, httplib::Response& res) {
    std::string path;
    {
      std::lock_guard<std::mutex> lk(g_jobs_m);
      const Job* j = Find(req.matches[1].str());
      if (j == nullptr) {
        res.status = 404;
        return;
      }
      if (j->status != "succeeded") {
        res.status = 409;  // not a truncated file
        res.set_content(json{{"status", j->status}}.dump(), "application/json");
        return;
      }
      path = j->mp4;
    }
    const std::string body = ReadFile(path);
    if (body.empty()) {
      res.status = 500;
      res.set_content("the mp4 is gone from disk", "text/plain");
      return;
    }
    res.set_content(body, "video/mp4");
  });

  std::fprintf(stderr, "studio: http://%s:%d/\n", host.c_str(), port);
  if (!srv.listen(host.c_str(), port)) {
    std::fprintf(stderr, "studio: cannot bind %s:%d\n", host.c_str(), port);
    return 1;
  }
  return 0;
}
