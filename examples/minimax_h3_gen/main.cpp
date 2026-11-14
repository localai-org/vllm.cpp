// minimax-h3-gen — MiniMax-H3 video+audio generation end to end, as a THIN
// CLIENT of the public C ABI (include/vllm.h) and NOTHING else, per the ONE
// SURFACE directive (ARCH-ONE-SURFACE ROW 2).
//
// The pre-fold version of this example owned the whole assembly pipeline
// privately (1293 lines: loader-arm dispatch, encoder conditioning, reference
// encoding, the noise streams, artifact writing, the ffmpeg mux). All of that
// now lives in the library behind vllm_video_engine_load + vllm_video_generate
// — the SAME entry points any embedder gets — and this file keeps exactly what
// an example may own: argv parsing, printing, and the ONE process spawn (the
// ffmpeg invocation, ratified into examples/ 2026-08-03; the library only
// composes the argv). The rendered frames + WAV are byte-identical to the
// pre-fold binary (gated by tests/vllm/models/test_minimax_h3_video_fold.cpp
// and the committed goldens in tests/vllm/models/fixtures/minimax_h3_video_fold).
//
// Usage:
//   minimax-h3-gen --dit <dit.gguf|dit.safetensors|shard-dir/>
//                  --video-vae <f.safetensors> [--video-vae-config <config.json>]
//                  --audio-vae <f.safetensors> [--audio-vae-config <config.json>]
//                  --partition fl2va|ref2va    (REQUIRED for a full render: the
//                    served checkpoint partition — community GGUF/NVFP4 files
//                    strip it and the FL2VA/Ref2VA DiTs are indistinguishable)
//                  (--prompt-embeds <f32.bin> | --encoder <gguf|shard-dir/>
//                    --prompt <text> [--tokenizer <tokenizer.json>])
//                  --workdir DIR [--out <out.mp4>] [--ffmpeg PATH]
//                  [--steps N] [--frames N] [--height N] [--width N] [--seed N]
//                  [--device cpu|cuda] [--dequant-bf16 | --keep-quant] [--fp4-resident]
//                  [--first-frame f.ppm] [--last-frame f.ppm] [--noise-aug A]
//                  [--ref-image f.ppm] [--ref-video DIR] [--ref-audio f.wav]
//
// The pre-fold binary's DIAGNOSTIC modes (--dry-run, --denoise-only,
// --dump-params, --encoder-only/--save-embeds, --decode-latent, --roundtrip,
// --prompt-image, --cond-image, multiple --ref-image) were part of the private
// pipeline and are gone with it; the capabilities they probed are gated by
// test_minimax_h3 / test_minimax_h3_video_fold, and multi-image ref2va remains
// reachable through the C++ seam (a named residual of the ABI's first slice).
#if defined(_WIN32)
#include <process.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vllm.h"

namespace {

// The ONE process spawn, and it is in examples/ by decision: the library
// composed `args`; this runs it.
int RunFfmpeg(const std::vector<std::string>& args) {
  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (const std::string& a : args) argv.push_back(const_cast<char*>(a.c_str()));
  argv.push_back(nullptr);
#if defined(_WIN32)
  // _P_WAIT returns the child's exit status, or -1 when the program cannot be
  // started. MSVCRT quotes arguments that contain spaces but does not escape
  // embedded double quotes.
  const intptr_t status = _spawnvp(_P_WAIT, argv[0], argv.data());
  return static_cast<int>(status);
#else
  const pid_t pid = fork();
  if (pid < 0) {
    std::fprintf(stderr, "error: fork failed\n");
    return -1;
  }
  if (pid == 0) {
    execvp(argv[0], argv.data());
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

[[noreturn]] void Usage(int code) {
  std::fprintf(
      stderr,
      "usage: minimax-h3-gen --dit <f|shard-dir> --video-vae <f> --audio-vae <f> "
      "--partition fl2va|ref2va\n"
      "                      (--prompt-embeds <f32.bin> | --encoder <f|shard-dir> "
      "--prompt <text> [--tokenizer <f>])\n"
      "                      --workdir DIR [--out <out.mp4>] [--ffmpeg PATH]\n"
      "                      [--video-vae-config <j>] [--audio-vae-config <j>]\n"
      "                      [--steps N] [--frames N] [--height N] [--width N] [--seed N]\n"
      "                      [--device cpu|cuda] [--dequant-bf16 | --keep-quant] "
      "[--fp4-resident]\n"
      "                      [--first-frame f.ppm] [--last-frame f.ppm] [--noise-aug A]\n"
      "                      [--ref-image f.ppm] [--ref-video DIR] [--ref-audio f.wav]\n"
      "                      [--lora <adapter.safetensors> [STRENGTH]]  fused at load; 1.0\n"
      "                      [--lora-dir DIR]  resolve <lora:name:strength> prompt tags\n");
  std::exit(code);
}

}  // namespace

int main(int argc, char** argv) {
  vllm_video_model_params mp = vllm_video_model_params_default();
  vllm_video_params vp = vllm_video_params_default();
  std::string workdir = "/tmp/minimax_h3_gen", out_path, ffmpeg = "ffmpeg", device = "cpu";

  std::vector<std::string> extra_keys, extra_values;
  int lora_count = 0;
  auto SetExtra = [&](const std::string& key, std::string value) {
    for (size_t i = 0; i < extra_keys.size(); ++i) {
      if (extra_keys[i] == key) {
        extra_values[i] = std::move(value);
        return;
      }
    }
    extra_keys.emplace_back(key);
    extra_values.push_back(std::move(value));
  };

  for (int i = 1; i < argc; ++i) {
    const std::string f = argv[i];
    if (f == "--dit") mp.dit_path = Need(argc, argv, ++i, "--dit");
    else if (f == "--encoder") mp.encoder_path = Need(argc, argv, ++i, "--encoder");
    else if (f == "--tokenizer") mp.tokenizer_path = Need(argc, argv, ++i, "--tokenizer");
    else if (f == "--video-vae") mp.video_vae_path = Need(argc, argv, ++i, "--video-vae");
    else if (f == "--video-vae-config") mp.video_vae_config_path = Need(argc, argv, ++i, f.c_str());
    else if (f == "--audio-vae") mp.audio_vae_path = Need(argc, argv, ++i, "--audio-vae");
    else if (f == "--audio-vae-config") mp.audio_vae_config_path = Need(argc, argv, ++i, f.c_str());
    else if (f == "--prompt-embeds") mp.prompt_embeds_path = Need(argc, argv, ++i, f.c_str());
    else if (f == "--partition") mp.partition = Need(argc, argv, ++i, "--partition");
    else if (f == "--device") device = Need(argc, argv, ++i, "--device");
    else if (f == "--keep-quant") mp.dequant_bf16 = 0;  // the default arm, named
    else if (f == "--dequant-bf16") mp.dequant_bf16 = 1;
    else if (f == "--fp4-resident") mp.fp4_resident = 1;
    else if (f == "--prompt") vp.prompt = Need(argc, argv, ++i, "--prompt");
    else if (f == "--steps") vp.steps = std::atoi(Need(argc, argv, ++i, "--steps"));
    else if (f == "--frames") vp.num_frames = std::atoi(Need(argc, argv, ++i, "--frames"));
    else if (f == "--height") vp.height = std::atoi(Need(argc, argv, ++i, "--height"));
    else if (f == "--width") vp.width = std::atoi(Need(argc, argv, ++i, "--width"));
    else if (f == "--seed") {
      vp.seed = static_cast<uint64_t>(std::strtoull(Need(argc, argv, ++i, "--seed"), nullptr, 10));
      vp.has_seed = 1;
    } else if (f == "--first-frame") vp.first_frame = Need(argc, argv, ++i, "--first-frame");
    else if (f == "--last-frame") vp.last_frame = Need(argc, argv, ++i, "--last-frame");
    else if (f == "--ref-image") vp.ref_image = Need(argc, argv, ++i, "--ref-image");
    else if (f == "--ref-video") vp.ref_video = Need(argc, argv, ++i, "--ref-video");
    else if (f == "--ref-audio") vp.ref_audio = Need(argc, argv, ++i, "--ref-audio");
    else if (f == "--noise-aug") vp.noise_aug = std::strtof(Need(argc, argv, ++i, "--noise-aug"), nullptr);
    else if (f == "--lora") {
      ++lora_count;
      const std::string suffix = lora_count == 1 ? "" : "_" + std::to_string(lora_count);
      SetExtra("lora_path" + suffix, Need(argc, argv, ++i, f.c_str()));
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        SetExtra("lora_strength" + suffix, argv[++i]);
      }
    }
    else if (f == "--lora-dir") SetExtra("lora_dir", Need(argc, argv, ++i, "--lora-dir"));
    else if (f == "--workdir") workdir = Need(argc, argv, ++i, "--workdir");
    else if (f == "--out") out_path = Need(argc, argv, ++i, "--out");
    else if (f == "--ffmpeg") ffmpeg = Need(argc, argv, ++i, "--ffmpeg");
    else if (f == "--help" || f == "-h") Usage(0);
    else {
      std::fprintf(stderr, "error: unknown argument: %s\n", f.c_str());
      Usage(2);
    }
  }
  std::vector<const char*> keys, values;
  keys.reserve(extra_keys.size());
  values.reserve(extra_values.size());
  for (size_t i = 0; i < extra_keys.size(); ++i) {
    keys.push_back(extra_keys[i].c_str());
    values.push_back(extra_values[i].c_str());
  }
  if (!keys.empty()) {
    mp.extra_keys = keys.data();
    mp.extra_values = values.data();
    mp.n_extras = static_cast<int32_t>(keys.size());
  }

  if (mp.dit_path == nullptr) Usage(2);
  if (device == "cuda") mp.device = 1;
  else if (device != "cpu") {
    std::fprintf(stderr, "error: --device must be cpu or cuda\n");
    return 2;
  }
  vp.output_dir = workdir.c_str();

  vllm_video_engine* engine = nullptr;
  if (vllm_video_engine_load(&mp, &engine) != VLLM_OK) {
    std::fprintf(stderr, "error: %s\n", vllm_last_error());
    return 1;
  }

  vllm_video_result out;
  if (vllm_video_generate(engine, &vp, &out) != VLLM_OK) {
    std::fprintf(stderr, "error: %s\n", vllm_last_error());
    vllm_video_engine_free(engine);
    return 1;
  }
  std::fprintf(stderr, "  wrote %d frames (%dx%d @ %d fps) + %s (%d Hz)\n", out.frame_count,
               out.width, out.height, out.fps, out.audio_path, out.sample_rate);

  int status = 0;
  if (!out_path.empty()) {
    // Mux to the REQUESTED path: same composer the result's own argv used,
    // pointed at --out, with --ffmpeg substituted for argv[0].
    const std::string pattern = std::string(out.frame_dir) + "/frame_%06d.ppm";
    vllm_video_mux_params mx = vllm_video_mux_params_default();
    mx.frames = pattern.c_str();
    mx.audio_path = out.audio_path;
    mx.output_path = out_path.c_str();
    char** mux_argv = nullptr;
    int32_t mux_argc = 0;
    if (vllm_video_mux_argv(&mx, &mux_argv, &mux_argc) != VLLM_OK) {
      std::fprintf(stderr, "error: %s\n", vllm_last_error());
      vllm_video_result_free(&out);
      vllm_video_engine_free(engine);
      return 1;
    }
    std::vector<std::string> args(mux_argv, mux_argv + mux_argc);
    if (!args.empty()) args[0] = ffmpeg;
    vllm_video_mux_argv_free(mux_argv, mux_argc);
    status = RunFfmpeg(args);
    if (status == 0) {
      std::printf("wrote %s\n", out_path.c_str());
    } else {
      std::fprintf(stderr, "ffmpeg exited %d\n", status);
    }
  }

  vllm_video_result_free(&out);
  vllm_video_engine_free(engine);
  return status;
}
