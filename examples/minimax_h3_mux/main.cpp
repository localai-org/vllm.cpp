// minimax-h3-mux: muxes a MiniMax-H3 clip (PPM frames + WAV) into an MP4 by
// INVOKING ffmpeg — a THIN CLIENT of the public C ABI (include/vllm.h) and
// nothing else, per the ONE SURFACE directive (ARCH-ONE-SURFACE ROW 2).
//
// THIS FILE IS THE RATIFIED HOME OF THE PROCESS SPAWN. The developer's
// decision (2026-08-03): "re: ffmpeg invocation, correct - let's keep in the
// examples only". So the split is deliberate and load-bearing:
//
//   the LIBRARY  writes the artifacts and composes the ARGV — reachable here
//                through vllm_video_mux_argv (and, for a whole generation,
//                vllm_video_generate's result) — and spawns NOTHING.
//   examples/    (here) performs the invocation.
//
// The argv printed below is byte-identical to the pre-fold binary's
// (--print-only golden, tests/vllm/models/fixtures/minimax_h3_video_fold).
//
// Usage:
//   minimax-h3-mux --frames <pattern> --out <out.mp4> [--audio <in.wav>]
//                  [--fps N] [--crf N] [--ffmpeg <path>] [--print-only]
//
//   --frames  printf-style pattern the library's PPM writer filled in,
//             e.g. /tmp/h3/frame_%06d.ppm
//   --audio   omitted => a silent clip
//   --print-only  print the argv and exit WITHOUT spawning (lets the argv be
//                 inspected, diffed or run by hand on a box with no ffmpeg).
#if defined(_WIN32)
#include <process.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vllm.h"

namespace {

// Run argv to completion and return its exit status. The ONLY process spawn in
// the MiniMax-H3 path, and it is in examples/ by project decision.
int RunFfmpeg(const std::vector<std::string>& args) {
  std::vector<char*> c_args;
  c_args.reserve(args.size() + 1);
  for (const std::string& arg : args) {
    c_args.push_back(const_cast<char*>(arg.c_str()));
  }
  c_args.push_back(nullptr);

#if defined(_WIN32)
  // _P_WAIT returns the child's exit status, or -1 when the program cannot be
  // started. MSVCRT quotes arguments that contain spaces but does not escape
  // embedded double quotes.
  const intptr_t status = _spawnvp(_P_WAIT, c_args[0], c_args.data());
  return static_cast<int>(status);
#else
  const pid_t pid = fork();
  if (pid < 0) {
    std::fprintf(stderr, "error: fork failed\n");
    return -1;
  }
  if (pid == 0) {
    execvp(c_args[0], c_args.data());
    // Only reached if exec failed; _exit (not exit) so the child never runs
    // the parent's atexit handlers or flushes its buffers a second time.
    _exit(127);
  }
  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    std::fprintf(stderr, "error: waitpid failed\n");
    return -1;
  }
  if (WIFSIGNALED(status)) {
    std::fprintf(stderr, "error: ffmpeg died on signal %d\n", WTERMSIG(status));
    return -1;
  }
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

const char* Need(int argc, char** argv, int i, const char* flag) {
  if (i >= argc) {
    std::fprintf(stderr, "error: missing value for %s\n", flag);
    std::exit(2);
  }
  return argv[i];
}

}  // namespace

int main(int argc, char** argv) {
  vllm_video_mux_params request = vllm_video_mux_params_default();
  std::string ffmpeg = "ffmpeg";
  bool print_only = false;

  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--frames") {
      request.frames = Need(argc, argv, ++i, "--frames");
    } else if (flag == "--audio") {
      request.audio_path = Need(argc, argv, ++i, "--audio");
    } else if (flag == "--out") {
      request.output_path = Need(argc, argv, ++i, "--out");
    } else if (flag == "--fps") {
      request.fps = std::atoi(Need(argc, argv, ++i, "--fps"));
    } else if (flag == "--crf") {
      request.crf = std::atoi(Need(argc, argv, ++i, "--crf"));
    } else if (flag == "--ffmpeg") {
      ffmpeg = Need(argc, argv, ++i, "--ffmpeg");
    } else if (flag == "--print-only") {
      print_only = true;
    } else {
      std::fprintf(stderr, "error: unknown argument: %s\n", flag.c_str());
      return 2;
    }
  }
  if (request.frames == nullptr || request.output_path == nullptr) {
    std::fprintf(stderr,
                 "usage: minimax-h3-mux --frames <pattern> --out <out.mp4> "
                 "[--audio <in.wav>] [--fps N] [--crf N] [--ffmpeg <path>] "
                 "[--print-only]\n");
    return 2;
  }

  // The LIBRARY decides the encoding contract (h264/yuv420p + AAC, -shortest,
  // +faststart); this file only runs it.
  char** mux_argv = nullptr;
  int32_t mux_argc = 0;
  if (vllm_video_mux_argv(&request, &mux_argv, &mux_argc) != VLLM_OK) {
    std::fprintf(stderr, "error: %s\n", vllm_last_error());
    return 1;
  }
  std::vector<std::string> args(mux_argv, mux_argv + mux_argc);
  vllm_video_mux_argv_free(mux_argv, mux_argc);
  if (!args.empty()) args[0] = ffmpeg;

  for (size_t i = 0; i < args.size(); ++i) {
    std::printf("%s%s", i == 0 ? "" : " ", args[i].c_str());
  }
  std::printf("\n");
  if (print_only) return 0;

  const int status = RunFfmpeg(args);
  if (status == 127) {
    std::fprintf(stderr, "failed to exec '%s' — is ffmpeg installed and on PATH?\n",
                 ffmpeg.c_str());
    return 127;
  }
  if (status != 0) {
    std::fprintf(stderr, "ffmpeg exited %d\n", status);
    return status;
  }
  std::printf("wrote %s\n", request.output_path);
  return 0;
}
