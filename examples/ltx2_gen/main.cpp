// ltx2-gen — an LTX-2.5 video+audio render, end to end, as a THIN CLIENT of the
// public C ABI (include/vllm.h) and NOTHING else, per the ONE SURFACE directive
// (ARCH-ONE-SURFACE ROW 2). Row MODEL-DIFFUSION-LTX25 phase L9B, issue #435.
//
// WHY A SECOND GENERATION EXAMPLE AND NOT A FLAG ON minimax-h3-gen. The ABI's
// video slice went family-generic at v18: `family` selects the model family and
// two parallel string arrays carry the FAMILY-SPECIFIC load knobs
// (vllm.h:735-747). `minimax-h3-gen` predates that and drives neither, and
// LTX-2.5 CANNOT LOAD without three of them — the audio-stream prompt embeds,
// the DiT config the shipped FP8 checkpoint does not carry, and (for the second
// distilled phase) the latent spatial upsampler. This file exists to name those
// knobs as flags rather than to hand a user a `--extra key=value` grab bag, and
// it is the smallest thing that can drive a real render.
//
// WHAT IT DOES NOT DO. It composes no ffmpeg command line of its own and encodes
// nothing: `vllm_video_mux_argv` builds the argv and this file exec's it, which
// is the ratified process boundary (2026-08-03). It carries no model logic — no
// noise stream, no schedule, no dtype choice — because all of that is the
// library's and a second copy here would be a parallel path.
//
// CONDITIONING HAS TWO SOURCES, and as of phase L13 the first of them is a
// typed prompt. `--encoder` names the Gemma-4 12B text tower and `--prompt`
// carries the words; the tower tokenizes them with its OWN embedded tokenizer,
// runs, projects all 49 hidden states to the two stream widths and hands the
// result to the embeddings connector and then to cross-attention. Before L13
// the tower had no route to the DiT and `--prompt` did not exist here at all.
//
// `--prompt-embeds` + `--audio-prompt-embeds` remain, and remain the only
// conditioning without a tower. Both streams are conditioned or neither:
// LTX-2.5 cross-attends at TWO widths (4096 video, 2048 audio) and one of them
// alone leaves a stream unconditioned, which renders instead of failing.
//
// `--encoder-config` is not optional paperwork. The only shipped LTX-2.5 text
// encoder carries no `__metadata__` at all, so its Gemma config cannot come out
// of the file; the engine refuses rather than defaulting one, because
// `layer_types`, `global_head_dim` and `attention_k_eq_v` resolve a DIFFERENT
// tower out of a byte-identical tensor set.
#if defined(_WIN32)
#include <process.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "vllm.h"

namespace {

// The ONE process spawn, in examples/ by decision: the library composed `args`;
// this runs it. No shell — the argv is exec'd directly.
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
      "usage: ltx2-gen --dit <transformer.safetensors> --video-vae <f> --audio-vae <f>\n"
      "                (--encoder <gemma4-te.safetensors> --prompt \"...\"\n"
      "                 | --prompt-embeds <video.f32.bin> --audio-prompt-embeds <audio.f32.bin>)\n"
      "                --workdir DIR [--out <out.mp4>] [--ffmpeg PATH]\n"
      "                [--encoder-config <gemma_config.json>]     REQUIRED when the text\n"
      "                                                           encoder carries no metadata\n"
      "                [--dit-config <transformer-config.json>]   REQUIRED when the DiT\n"
      "                                                           carries no __metadata__\n"
      "                [--model-version 2.5] [--pipeline-kind distilled_two_stage]\n"
      "                --checkpoint-class full|distilled|keyframe_slot_sft\n"
      "                                                           REQUIRED: which class of\n"
      "                                                           transformer --dit points at\n"
      "                [--upsampler <latent-spatial-x2.safetensors>]  phase 2 needs it\n"
      "                [--temporal-upsampler <latent-temporal-x2.safetensors>]\n"
      "                [--temporal-upsample-rounds 0|1|2]         dfr only; each round\n"
      "                                                           doubles the frame count\n"
      "                [--max-phase N] [--allow-unported]\n"
      "                [--lora <adapter.safetensors> [STRENGTH]]  fused at load; 1.0\n"
      "                                                           REQUIRED on ti2vid_two_stage,\n"
      "                                                           keyframe_interpolation,\n"
      "                                                           a2vid_two_stage,\n"
      "                                                           res2s_two_stage and dfr\n"
      "                [--prompt-valid-rows N]   how many embed rows are real tokens\n"
      "                [--frames N] [--width N] [--height N] [--seed N] [--steps N]\n"
      "                [--first-frame <image.ppm>] [--last-frame <image.ppm>]\n"
      "                [--image-crf 0]\n"
      "                [--audio-path <in.wav>] [--audio-start-time S]\n"
      "                [--audio-max-duration S]\n"
      "                [--device cpu|cuda]\n\n"
      "Renders LTX-2.5 (family \"ltx-2.5\") through vllm_video_engine_load +\n"
      "vllm_video_generate.\n\n"
      "CONDITIONING, two ways. With --encoder the Gemma-4 12B text tower is loaded and\n"
      "--prompt is tokenized by the tower's OWN embedded tokenizer, run, projected to\n"
      "the two stream widths and passed through the embeddings connector. The tower is\n"
      "~24 GB of host bf16 and stays resident, because a prompt arrives per request.\n"
      "--encoder-config supplies the Gemma config when the encoder declares none, which\n"
      "the only shipped one does not; without either the load is refused rather than\n"
      "defaulted, since the config decides which layers are full-attention and a wrong\n"
      "one resolves a different tower from the same tensors.\n\n"
      "Without --encoder, conditioning is PROMPT-EMBEDS: both files are rows of\n"
      "little-endian f32, the video one 4096 wide and the audio one 2048, with the\n"
      "SAME row count. Passing --prompt without --encoder is refused rather than\n"
      "silently rendering those embeddings as if they were it.\n\n"
      "Those rows are the EMBEDDINGS CONNECTOR's input, not the DiT's: when the\n"
      "checkpoint carries the two *_embeddings_connector families (both shipped\n"
      "LTX-2.5 DiTs do) they run through it first, with the checkpoint's own\n"
      "weights. The row count must then be a multiple of the connector's learnable\n"
      "register count (128 on the shipped files), and --prompt-valid-rows says how\n"
      "many of them are real: the rest are padding, and padding is REPLACED by the\n"
      "learnable register table rather than ignored.\n\n"
      "--steps N sets the DENOISE STEP COUNT (#2130). Omitting it, or passing 0 or\n"
      "less, keeps the ABI's own contract for this field: the resolved recipe\n"
      "decides, which for --pipeline-kind one_stage on model version 2.5 is 30\n"
      "(upstream utils/constants.py:83-88). The flag exists because the engine has\n"
      "always honoured a request value and nothing shipped could set one, so every\n"
      "LTX-2.5 render taken here ran one step count and no render could be matched\n"
      "to a reference taken at another. #1864's upstream reference render is 8\n"
      "steps, so an absolute comparison against it needs --steps 8. On a recipe\n"
      "whose schedule is distilled into the weights the engine REFUSES the override\n"
      "by name (ltx2_video.cpp, `allow_request_sigmas`) rather than sampling a\n"
      "trajectory the weights were never distilled for; this flag does not change\n"
      "that.\n\n"
      "IMAGE CONDITIONING (image-to-video). --first-frame takes a binary PPM (P6,\n"
      "maxval 255) and pins latent frame 0 to it: it is decoded, aspect-filled and\n"
      "centre-cropped to each phase's own resolution, VAE-encoded, and written into\n"
      "the clean latent. It needs --image-crf 0, and that is DELIBERATELY not the\n"
      "default. Upstream re-compresses a conditioning image through H.264 at the CRF\n"
      "the checkpoint's generation was trained with, which for LTX-2.5 is 18; that\n"
      "round trip needs libx264 and none is vendored here, so leaving --image-crf out\n"
      "resolves 18 and REFUSES by name. --image-crf 0 is upstream-legal and OUT OF\n"
      "DISTRIBUTION: the model sees pixels it was not trained on. That is a quality\n"
      "cost, and this tool states it rather than turning it on quietly.\n"
      "--last-frame takes a second PPM and pins the CLOSING frame. It is a KEYFRAME\n"
      "rather than a replacement: its tokens are APPENDED to the sequence, carrying\n"
      "the temporal position of pixel frame N-1, and are trimmed off again before the\n"
      "clip is decoded. Both slots share one --image-crf and one strength, and a\n"
      "keyframe at an INTERIOR frame is not requestable (#1187).\n\n"
      "AUDIO-TO-VIDEO. --audio-path takes a 16-bit PCM WAV and CONDITIONS the render\n"
      "on it: the take is decoded, encoded through the audio VAE\'s encoder, truncated\n"
      "to the clip\'s duration, and then held FROZEN through every denoise phase, so\n"
      "the video is generated around a soundtrack you supplied rather than one the\n"
      "model invented. The CONDITIONING MECHANISM is upstream\'s A2VidPipelineTwoStage:\n"
      "decode, encode, truncate to the clip, freeze through both stages. The SCHEDULE\n"
      "is NOT. Upstream\'s stage 1 is a caller-configured GUIDED one whose\n"
      "--a2v-guidance-scale is the guider\'s modality_scale; a take here rides whichever\n"
      "recipe the checkpoint resolves, in practice distilled_two_stage, whose sigmas are\n"
      "fixed. So no claim is made that this reproduces upstream\'s A2Vid output.\n\n"
      "ANY sample rate is accepted: a take whose rate differs from the audio VAE\'s\n"
      "mel front-end rate is RESAMPLED to it, with upstream\'s own filter — the\n"
      "gcd-reduced rational-ratio hann-windowed sinc `torchaudio.functional.resample`\n"
      "builds at its defaults, which is what ops.py:40 calls. The CHANNEL COUNT must\n"
      "still equal the encoder\'s in_channels and a mismatch is refused with both\n"
      "numbers: upstream feeds the file\'s own channel count into a fixed-width\n"
      "convolution, so up- or downmixing here would render a finished clip conditioned\n"
      "on audio you never supplied. The take must also be at least as long as the clip:\n"
      "upstream truncates a long one and asserts on a short one, so a short one is\n"
      "refused.\n\n"
      "--audio-start-time seeks into the file (default 0) and --audio-max-duration caps\n"
      "how much is read (default: the clip\'s own duration). Either without\n"
      "--audio-path is refused rather than ignored. The rendered audio.wav is your own\n"
      "input, not a VAE round trip, which is upstream\'s deliberate choice.\n\n"
      "IC-LoRA REFERENCE CONDITIONING conditions the whole render on a reference clip\n"
      "(depth, pose, edges) and needs --pipeline-kind ic_lora, which is upstream's own\n"
      "ICLoraPipeline: the adapter rides stage 1 and stage 2 runs bare. --ref-video is\n"
      "the clip, as a DIRECTORY of frame_%%06d.ppm, and --ref-video-strength is the\n"
      "second half of upstream's --video-conditioning PATH STRENGTH. Supply the\n"
      "IC-LoRA itself with --lora: its metadata carries the downscale and temporal\n"
      "factors the reference is read at, and without one both default to 1.\n"
      "--conditioning-attention-mask names a grayscale frame directory whose pixels\n"
      "attenuate the reference per region, and --conditioning-attention-strength\n"
      "scales it. A strength below 1 with no mask is REFUSED rather than served: it is\n"
      "upstream's Python-API-only branch, which its own CLI cannot reach either.\n\n"
      "RETAKE regenerates a time window of an existing clip and keeps the rest.\n"
      "--ref-video names a DIRECTORY of frame_%%06d.ppm (the layout minimax-h3-gen\n"
      "writes), not a container file: upstream opens one with PyAV and no demuxer is\n"
      "vendored here, so a .mp4 is refused. --retake-start-time and --retake-end-time\n"
      "are seconds, the end exclusive, and --retake-frame-rate is required because a\n"
      "frame folder carries no container frame rate. Needs --pipeline-kind retake: a\n"
      "retake is ONE stage at the clip\'s own resolution, and the distilled two-stage\n"
      "recipe renders its first stage at half. The geometry comes from the clip, so\n"
      "--width, --height and --frames are refused alongside it. --regenerate-video 0\n"
      "freezes the clip instead; --regenerate-audio has no effect while the source is\n"
      "a frame folder, because a folder carries no audio and both of upstream\'s audio\n"
      "predicates test for one.\n\n"
      "WHICH TRANSFORMER --dit POINTS AT is something this tool cannot work out, so\n"
      "--checkpoint-class is REQUIRED on every pipeline kind but dmd2 (#1137).\n"
      "Upstream keys a checkpoint class per pipeline and most read Full, while the\n"
      "distilled arms need the distilled file. The two 22B bf16 transformers are the\n"
      "same 42,018,190,584 bytes, carry a byte-identical embedded config and both\n"
      "declare model_version 2.5.0, so no header field separates them. Passing the\n"
      "wrong one used to RENDER: right size, right frame count, right sample rate,\n"
      "in a sampling regime the weights were never trained for. It is refused now.\n"
      "\n"
      "TEXT-TO-AUDIO renders a soundtrack and NO PICTURE. --pipeline-kind\n"
      "t2a_one_stage selects it; the result carries an audio.wav, zero frames and no\n"
      "ffmpeg argv, because there is nothing to mux. --video-vae is not needed and\n"
      "--width/--height are refused: upstream passes a 512x512 placeholder whose\n"
      "height and width it documents as unused, and only --frames and the recipe\'s\n"
      "frame rate are read, to derive the DURATION. Unlike the distilled video\n"
      "recipes this one is GUIDED: it runs three DiT forwards per step by default\n"
      "(conditional, unconditional, and one with the audio self-attention perturbed),\n"
      "so it needs a text tower for the negative prompt. --negative-prompt,\n"
      "--audio-cfg-guidance-scale, --audio-stg-guidance-scale, --audio-rescale-scale,\n"
      "--audio-skip-step and --audio-stg-blocks are upstream\'s own flags; absent, each\n"
      "takes the checkpoint generation\'s own value. --audio-stg-blocks is comma\n"
      "separated and a block index outside the DiT\'s layer count is refused rather\n"
      "than clamped. The accelerator is REFUSED by name on this pipeline.\n\n"
      "GUIDANCE ON A VIDEO RENDER. --pipeline-kind one_stage runs upstream's guided\n"
      "denoiser: FOUR DiT forwards per step (conditional, unconditional, perturbed,\n"
      "and one with the audio<->video cross attention off), combined per modality in\n"
      "x0 space. --video-cfg-guidance-scale, --video-stg-guidance-scale,\n"
      "--video-rescale-scale, --video-skip-step, --video-stg-blocks,\n"
      "--a2v-guidance-scale and --v2a-guidance-scale are upstream's own flags and\n"
      "take the checkpoint generation's value when absent. The unconditional forward\n"
      "needs a NEGATIVE conditioning: either a text tower plus --negative-prompt, or\n"
      "--negative-prompt-embeds with --negative-audio-prompt-embeds. Absent both, a\n"
      "cfg scale other than 1.0 is refused by name. --pipeline-kind\n"
      "distilled_two_stage and retake distil their guidance INTO the weights and\n"
      "refuse every one of these flags rather than applying it.\n\n"
      "AUDIO-TO-VIDEO renders a clip AROUND a soundtrack you supply.\n"
      "--pipeline-kind a2vid_two_stage selects it: stage 1 denoises video at half\n"
      "resolution, guided by the checkpoint generation's own scales on a schedule\n"
      "derived from the step count, and stage 2 upsamples 2x and refines with the\n"
      "distilled three-sigma schedule. The take is encoded once and frozen at both\n"
      "stages, and the audio.wav you get back is your own file. --audio-path is\n"
      "REQUIRED on every request and --lora is REQUIRED at load, because upstream\n"
      "makes both required and stage 2 is a refinement the base weights were never\n"
      "distilled for; --upsampler is needed for stage 2 as on any two-stage recipe.\n"
      "The guidance flags above reach stage 1 and are IGNORED by stage 2, which runs\n"
      "no guider at all -- unlike distilled_two_stage, which refuses them. The\n"
      "distilled adapter rides stage 2 ALONE, as upstream does: stage 1 runs the base\n"
      "weights and the engine rebinds the DiT at the phase boundary.\n\n"
      "TWO-STAGE TEXT/IMAGE-TO-VIDEO is the plain two-stage arm.\n"
      "--pipeline-kind ti2vid_two_stage selects it: stage 1 generates at HALF the\n"
      "requested resolution under full CFG on the UNADAPTED model, on a schedule\n"
      "derived from the step count, and stage 2 upsamples 2x and refines with the\n"
      "distilled three-sigma schedule and no guider. --lora is REQUIRED at load, as\n"
      "upstream's --distilled-lora is, and --upsampler is needed for stage 2. There\n"
      "is NO --audio-path here: the soundtrack is generated, and the audio.wav you get\n"
      "back is STAGE 1's, because upstream refines the picture only and discards\n"
      "stage 2's audio. --width and --height describe the FINAL output and must\n"
      "divide 64, since stage 1 halves them. Upstream runs this on the FULL\n"
      "(non-distilled) transformer; pointing it at a distilled checkpoint renders a\n"
      "plausible clip on a trajectory those weights were never trained for, and\n"
      "nothing in the output says so.\n\n"
      "KEYFRAME INTERPOLATION generates the motion BETWEEN keyframes you pin.\n"
      "--pipeline-kind keyframe_interpolation selects it. Its two stages are the\n"
      "ti2vid_two_stage ones -- guided half-res stage 1 on the UNADAPTED model, then a\n"
      "distilled three-sigma refinement -- and it needs the same --lora and\n"
      "--upsampler for the same reasons. TWO fields differ and both of them render\n"
      "either way. First, --first-frame is a KEYFRAME here rather than a frame that\n"
      "overwrites the opening latent: upstream drops the frame-0 special case, so the\n"
      "image is appended as guidance the model interpolates FROM instead of replacing\n"
      "what it would otherwise generate. Second, the audio.wav you get back is STAGE\n"
      "2\'s, not stage 1\'s as on ti2vid_two_stage. Use --first-frame and --last-frame\n"
      "together to pin both ends of the clip; a keyframe at an INTERIOR frame is not\n"
      "requestable yet.\n");
  std::exit(code);
}

}  // namespace

int main(int argc, char** argv) {
  vllm_video_model_params mp = vllm_video_model_params_default();
  vllm_video_params vp = vllm_video_params_default();
  std::string workdir = "/tmp/ltx2_gen", out_path, ffmpeg = "ffmpeg", device = "cuda";
  // BORROWED by `vllm_video_generate`, like the extras below, so it is owned
  // here and pointed at only after parsing.
  std::string prompt, first_frame, last_frame, image_crf;
  std::string audio_path, audio_start_time, audio_max_duration;
  // RETAKE (row LTX25-RETAKE, #924): a source clip DIRECTORY and the window to
  // regenerate. `--ref-video` is a directory of frame_%06d.ppm, not a container.
  std::string ref_video, retake_start, retake_end, retake_fps, regen_video, regen_audio;
  // IC-LoRA reference conditioning (#3020): the strength half of
  // `--video-conditioning`, and both halves of `--conditioning-attention-mask`.
  std::string ref_video_strength, cond_mask_dir, cond_mask_strength;
  // TEXT-TO-AUDIO (row LTX25-T2A-ONE-STAGE, #1005): one flag per argument of
  // upstream's `default_1_stage_t2a_arg_parser` (utils/args.py:1070-1120).
  std::string negative_prompt, audio_cfg_scale, audio_stg_scale, audio_rescale;
  std::string audio_skip_step, audio_stg_blocks;

  // THE VIDEO GUIDER (row LTX25-GUIDED-VIDEO, #1092): the other half of the same
  // parser, `default_1_stage_arg_parser` (utils/args.py:947-1066). `--negative-
  // prompt` above is shared by both, which is why it is not repeated here.
  std::string video_cfg_scale, video_stg_scale, video_rescale, video_skip_step;
  std::string video_stg_blocks, a2v_scale, v2a_scale;
  std::string temporal_rounds;  // DFR's temporal x2/x4 refinement rounds (#986)
  std::string negative_embeds, negative_audio_embeds;

  // The extras are BORROWED by the load call, so the strings must outlive it.
  // Kept as two parallel vectors of owned strings plus the char* views the ABI
  // takes, built once after parsing.
  std::vector<std::string> extra_keys, extra_values;
  // How many `--lora` flags have been seen, which is how the engine's indexed
  // load extras are numbered. See the flag below.
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
    else if (f == "--video-vae") mp.video_vae_path = Need(argc, argv, ++i, "--video-vae");
    else if (f == "--video-vae-config") mp.video_vae_config_path = Need(argc, argv, ++i, f.c_str());
    else if (f == "--audio-vae") mp.audio_vae_path = Need(argc, argv, ++i, "--audio-vae");
    else if (f == "--audio-vae-config") mp.audio_vae_config_path = Need(argc, argv, ++i, f.c_str());
    else if (f == "--encoder") mp.encoder_path = Need(argc, argv, ++i, "--encoder");
    else if (f == "--encoder-config")
      SetExtra("encoder_config_path", Need(argc, argv, ++i, f.c_str()));
    else if (f == "--prompt") prompt = Need(argc, argv, ++i, "--prompt");
    else if (f == "--prompt-embeds") mp.prompt_embeds_path = Need(argc, argv, ++i, f.c_str());
    else if (f == "--audio-prompt-embeds")
      SetExtra("audio_prompt_embeds_path", Need(argc, argv, ++i, f.c_str()));
    else if (f == "--dit-config") SetExtra("dit_config_path", Need(argc, argv, ++i, f.c_str()));
    else if (f == "--model-version") SetExtra("model_version", Need(argc, argv, ++i, f.c_str()));
    else if (f == "--pipeline-kind") SetExtra("pipeline_kind", Need(argc, argv, ++i, f.c_str()));
    // WHICH CLASS OF TRANSFORMER `--dit` points at: `full`, `distilled` or
    // `keyframe_slot_sft` (row LTX25-CHECKPOINT-CLASS, issue #1137). REQUIRED on
    // every pipeline kind whose upstream row states a class, which is every kind
    // but `dmd2`. The library refuses the load without it and says why; this CLI
    // forwards the flag rather than defaulting one, because every default here
    // would pick a sampling regime for the caller.
    else if (f == "--checkpoint-class")
      SetExtra("checkpoint_class", Need(argc, argv, ++i, f.c_str()));
    else if (f == "--upsampler") SetExtra("upsampler_path", Need(argc, argv, ++i, f.c_str()));
    // DFR's SECOND upsampler, and its round count (#986). Upstream splits these
    // the same way: `--temporal-upsampler-path` is a constructor argument
    // (dfr_pipeline.py:578-583, :177) and `--temporal-upsample-rounds` is a
    // `__call__` argument (:584-590, :277), so one is a LOAD extra here and the
    // other rides the per-generation array.
    else if (f == "--temporal-upsampler")
      SetExtra("temporal_upsampler_path", Need(argc, argv, ++i, f.c_str()));
    else if (f == "--temporal-upsample-rounds")
      temporal_rounds = Need(argc, argv, ++i, f.c_str());
    else if (f == "--negative-prompt-embeds") {
      negative_embeds = Need(argc, argv, ++i, f.c_str());
      SetExtra("negative_prompt_embeds_path", negative_embeds);
    } else if (f == "--negative-audio-prompt-embeds") {
      negative_audio_embeds = Need(argc, argv, ++i, f.c_str());
      SetExtra("negative_audio_prompt_embeds_path", negative_audio_embeds);
    }
    // Kept although the library REFUSES this extra by name (#611): the duration
    // head is unported, and forwarding the flag gets the caller that named
    // refusal instead of "unknown option", which says nothing about why.
    else if (f == "--duration-head")
      SetExtra("duration_head_path", Need(argc, argv, ++i, f.c_str()));
    else if (f == "--max-phase") SetExtra("max_phase", Need(argc, argv, ++i, f.c_str()));
    else if (f == "--prompt-valid-rows")
      SetExtra("prompt_embeds_valid_rows", Need(argc, argv, ++i, f.c_str()));
    else if (f == "--allow-unported") SetExtra("allow_unported_modules", "1");
    // IC-LoRA (row LTX25-IC-LORA, issue #923), mirroring upstream's
    // `--lora PATH [STRENGTH]` (ltx-pipelines utils/args.py:600-611): the
    // strength is optional and defaults to upstream's DEFAULT_LORA_STRENGTH.
    // LOAD extras, because upstream fuses the adapter into the weights at
    // construction (ic_lora.py:104-114) and it cannot vary per request.
    //
    // REPEATABLE since row LTX25-LORA-FUSION (#932), which is what upstream's
    // own help text has always said: "Can be specified multiple times. Example:
    // --lora path/to/lora1.safetensors 0.8 --lora path/to/lora2.safetensors"
    // (`utils/args.py:608-609`). The engine numbers adapters from 1 with the
    // first unindexed, so this appends `_2`, `_3`, ... in the order given —
    // which is the order they fuse in, and that order is observable.
    else if (f == "--lora") {
      ++lora_count;
      const std::string suffix = lora_count == 1 ? "" : "_" + std::to_string(lora_count);
      SetExtra("lora_path" + suffix, Need(argc, argv, ++i, f.c_str()));
      // The optional second word. Consumed only when it does not look like the
      // next flag, which is how upstream's `nargs="+"` LoraAction disambiguates.
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        SetExtra("lora_strength" + suffix, argv[++i]);
      }
    }
    // Image conditioning (row LTX25-IMAGE-COND, issue #644). `--first-frame` is
    // a binary PPM; `--image-crf` is the PER-GENERATION extra, so it rides
    // vp.extra_* rather than mp.extra_*. Only 0 is served, and it is NOT
    // defaulted here — leaving it out lets the engine resolve the checkpoint's
    // own 18 and refuse, which is the point: this CLI must not be the thing that
    // quietly turns an out-of-distribution render on.
    else if (f == "--first-frame") first_frame = Need(argc, argv, ++i, "--first-frame");
    // `--last-frame` pins the CLOSING keyframe, at pixel frame `frames - 1`. The
    // ABI has carried `last_frame` and the engine has served it since row
    // LTX25-TOKEN-APPEND (#930); this CLI simply never read the field, which
    // #1191 records. It matters from `keyframe_interpolation` on, because a
    // pipeline whose whole job is the motion BETWEEN two pinned frames could
    // otherwise only be asked for one of them. Same `--image-crf` and the same
    // strength as the first frame, because the request surface carries one of
    // each (#1187).
    else if (f == "--last-frame") last_frame = Need(argc, argv, ++i, "--last-frame");
    else if (f == "--image-crf") image_crf = Need(argc, argv, ++i, "--image-crf");
    // AUDIO-TO-VIDEO (#922). Upstream's `--audio-path` is REQUIRED because that
    // CLI drives the A2V pipeline and nothing else (a2vid_two_stage.py:312-317);
    // here one binary serves every LTX-2.5 path, so supplying it is what selects
    // the audio-conditioned one. Per-generation, so it rides vp.extra_* too.
    else if (f == "--audio-path") audio_path = Need(argc, argv, ++i, "--audio-path");
    else if (f == "--ref-video") ref_video = Need(argc, argv, ++i, "--ref-video");
    else if (f == "--retake-start-time")
      retake_start = Need(argc, argv, ++i, "--retake-start-time");
    else if (f == "--retake-end-time")
      retake_end = Need(argc, argv, ++i, "--retake-end-time");
    else if (f == "--retake-frame-rate")
      retake_fps = Need(argc, argv, ++i, "--retake-frame-rate");
    // IC-LoRA REFERENCE CONDITIONING (#3020). `--ref-video` above names the
    // clip; these three are the rest of upstream's two flags —
    // `--video-conditioning PATH STRENGTH` and
    // `--conditioning-attention-mask MASK_PATH STRENGTH` (ic_lora.py:416-441).
    // Per-generation, so they ride vp.extra_* like the retake knobs.
    else if (f == "--ref-video-strength")
      ref_video_strength = Need(argc, argv, ++i, "--ref-video-strength");
    else if (f == "--conditioning-attention-mask")
      cond_mask_dir = Need(argc, argv, ++i, "--conditioning-attention-mask");
    else if (f == "--conditioning-attention-strength")
      cond_mask_strength = Need(argc, argv, ++i, "--conditioning-attention-strength");
    // TEXT-TO-AUDIO (#1005). Selected by `--pipeline-kind t2a_one_stage`, which
    // is a LOAD extra; these six are per-generation and are refused by name on
    // any other pipeline rather than accepted and ignored.
    else if (f == "--negative-prompt")
      negative_prompt = Need(argc, argv, ++i, "--negative-prompt");
    else if (f == "--audio-cfg-guidance-scale")
      audio_cfg_scale = Need(argc, argv, ++i, "--audio-cfg-guidance-scale");
    else if (f == "--audio-stg-guidance-scale")
      audio_stg_scale = Need(argc, argv, ++i, "--audio-stg-guidance-scale");
    else if (f == "--audio-rescale-scale")
      audio_rescale = Need(argc, argv, ++i, "--audio-rescale-scale");
    else if (f == "--audio-skip-step")
      audio_skip_step = Need(argc, argv, ++i, "--audio-skip-step");
    else if (f == "--audio-stg-blocks")
      audio_stg_blocks = Need(argc, argv, ++i, "--audio-stg-blocks");
    else if (f == "--video-cfg-guidance-scale")
      video_cfg_scale = Need(argc, argv, ++i, "--video-cfg-guidance-scale");
    else if (f == "--video-stg-guidance-scale")
      video_stg_scale = Need(argc, argv, ++i, "--video-stg-guidance-scale");
    else if (f == "--video-rescale-scale")
      video_rescale = Need(argc, argv, ++i, "--video-rescale-scale");
    else if (f == "--video-skip-step")
      video_skip_step = Need(argc, argv, ++i, "--video-skip-step");
    else if (f == "--video-stg-blocks")
      video_stg_blocks = Need(argc, argv, ++i, "--video-stg-blocks");
    else if (f == "--a2v-guidance-scale")
      a2v_scale = Need(argc, argv, ++i, "--a2v-guidance-scale");
    else if (f == "--v2a-guidance-scale")
      v2a_scale = Need(argc, argv, ++i, "--v2a-guidance-scale");
    else if (f == "--regenerate-video")
      regen_video = Need(argc, argv, ++i, "--regenerate-video");
    else if (f == "--regenerate-audio")
      regen_audio = Need(argc, argv, ++i, "--regenerate-audio");
    else if (f == "--audio-start-time")
      audio_start_time = Need(argc, argv, ++i, "--audio-start-time");
    else if (f == "--audio-max-duration")
      audio_max_duration = Need(argc, argv, ++i, "--audio-max-duration");
    else if (f == "--device") device = Need(argc, argv, ++i, "--device");
    else if (f == "--frames") vp.num_frames = std::atoi(Need(argc, argv, ++i, "--frames"));
    // THE DENOISE STEP COUNT (#2130). `vllm_video_params.steps` has shipped in
    // the ABI since the H3 lane (include/vllm.h), `vllm_c.cpp` forwards it and
    // `ltx2_video.cpp` reads it -- `steps = gen.steps > 0 ? gen.steps :
    // recipe.num_inference_steps` -- so the engine has always honoured a request
    // value. Nothing shipped could SET one. Every LTX-2.5 render this repository
    // has taken therefore ran the recipe default, which for `one_stage` at model
    // version 2.5 is 30, and #1864's upstream reference render was taken at 8.
    // An absolute comparison against it would have carried a 3.75x denoise-budget
    // confound on the one axis the command line could not reach, in the direction
    // that flatters us. `<= 0` keeps meaning "the recipe decides", which is the
    // ABI's own contract for this field and not a new one.
    else if (f == "--steps") vp.steps = std::atoi(Need(argc, argv, ++i, "--steps"));
    else if (f == "--width") vp.width = std::atoi(Need(argc, argv, ++i, "--width"));
    else if (f == "--height") vp.height = std::atoi(Need(argc, argv, ++i, "--height"));
    else if (f == "--seed") {
      vp.seed = static_cast<uint64_t>(std::strtoull(Need(argc, argv, ++i, "--seed"), nullptr, 10));
      vp.has_seed = 1;
    }
    else if (f == "--workdir") workdir = Need(argc, argv, ++i, "--workdir");
    else if (f == "--out") out_path = Need(argc, argv, ++i, "--out");
    else if (f == "--ffmpeg") ffmpeg = Need(argc, argv, ++i, "--ffmpeg");
    else if (f == "--help" || f == "-h") Usage(0);
    else {
      std::fprintf(stderr, "error: unknown argument: %s\n", f.c_str());
      Usage(2);
    }
  }
  if (mp.dit_path == nullptr) Usage(2);
  if (device == "cuda") mp.device = 1;
  else if (device != "cpu") {
    std::fprintf(stderr, "error: --device must be cpu or cuda\n");
    return 2;
  }
  // DECLARED, never detected. Detection would also resolve this checkpoint, but
  // an explicit family is what makes an FP8-vs-NVFP4 comparison a statement
  // about the two files rather than about what a detector happened to claim.
  mp.family = "ltx-2.5";
  vp.output_dir = workdir.c_str();
  if (!prompt.empty()) vp.prompt = prompt.c_str();
  if (!first_frame.empty()) vp.first_frame = first_frame.c_str();
  if (!last_frame.empty()) vp.last_frame = last_frame.c_str();
  if (!ref_video.empty()) vp.ref_video = ref_video.c_str();

  // The PER-GENERATION extras are a SEPARATE array from the load-time ones, and
  // conflating them is the whole failure this keeps apart: `image_crf` handed to
  // the load call is an unknown LOAD extra and is refused there, which would
  // read as "the flag does not work" rather than as "it goes on the other call".
  std::vector<std::string> gen_keys, gen_values;
  if (!image_crf.empty()) {
    gen_keys.emplace_back("image_crf");
    gen_values.push_back(image_crf);
  }
  if (!audio_path.empty()) {
    gen_keys.emplace_back("audio_path");
    gen_values.push_back(audio_path);
  }
  if (!ref_video_strength.empty()) {
    gen_keys.emplace_back("ref_video_strength");
    gen_values.push_back(ref_video_strength);
  }
  if (!cond_mask_dir.empty()) {
    gen_keys.emplace_back("conditioning_attention_mask_dir");
    gen_values.push_back(cond_mask_dir);
  }
  if (!cond_mask_strength.empty()) {
    gen_keys.emplace_back("conditioning_attention_strength");
    gen_values.push_back(cond_mask_strength);
  }
  if (!audio_start_time.empty()) {
    gen_keys.emplace_back("audio_start_time");
    gen_values.push_back(audio_start_time);
  }
  if (!audio_max_duration.empty()) {
    gen_keys.emplace_back("audio_max_duration");
    gen_values.push_back(audio_max_duration);
  }
  // Each retake knob rides the SAME per-generation array. Supplying one without
  // the window is refused by the engine rather than ignored, so a half-typed
  // retake reports what is missing instead of rendering the ordinary path.
  // The knobs are a NAMED array rather than a braced-init-list iterated in
  // place. Both are correct -- a braced-init-list bound to the range variable
  // has its backing array lifetime-extended for the whole loop -- but the
  // in-place form draws -Wdangling-reference from gcc 16, which
  // `build-newest-gcc` found on its first run. The warning is a false positive
  // and it is not silenced: the range now has automatic storage and a name, so
  // no reference binds to a temporary and the question does not arise. Making
  // the loop variable a copy does NOT help; the diagnostic is about the range
  // reference, not the element.
  const std::pair<const char*, std::string*> retake_knobs[] = {
      std::make_pair("retake_start_time", &retake_start),
                         std::make_pair("retake_end_time", &retake_end),
                         std::make_pair("retake_frame_rate", &retake_fps),
                         std::make_pair("regenerate_video", &regen_video),
                         std::make_pair("regenerate_audio", &regen_audio),
                         // TEXT-TO-AUDIO (#1005). One flag per upstream CLI
                         // argument (`default_1_stage_t2a_arg_parser`,
                         // ltx-pipelines utils/args.py:1070-1120). They are
                         // per-generation, so they ride this array rather than
                         // the load one; `--pipeline-kind t2a_one_stage` is the
                         // LOAD knob that selects the pipeline, and supplying
                         // these without it is refused by name.
                         std::make_pair("negative_prompt", &negative_prompt),
                         std::make_pair("audio_cfg_guidance_scale", &audio_cfg_scale),
                         std::make_pair("audio_stg_guidance_scale", &audio_stg_scale),
                         std::make_pair("audio_rescale_scale", &audio_rescale),
                         std::make_pair("audio_skip_step", &audio_skip_step),
                         std::make_pair("audio_stg_blocks", &audio_stg_blocks),
                         // THE VIDEO GUIDER (#1092). Per-generation for the same
                         // reason the audio row is, and refused whole on a recipe
                         // whose guidance is distilled into the weights.
                         std::make_pair("video_cfg_guidance_scale", &video_cfg_scale),
                         std::make_pair("video_stg_guidance_scale", &video_stg_scale),
                         std::make_pair("video_rescale_scale", &video_rescale),
                         std::make_pair("video_skip_step", &video_skip_step),
                         std::make_pair("video_stg_blocks", &video_stg_blocks),
                         std::make_pair("a2v_guidance_scale", &a2v_scale),
                         std::make_pair("v2a_guidance_scale", &v2a_scale),
                         // DFR'S TEMPORAL ROUNDS (#986), mirroring upstream's own
                         // `--temporal-upsample-rounds` (dfr_pipeline.py:584-590).
                         // Per-generation because upstream takes it on `__call__`
                         // (:277) rather than on the constructor, unlike the
                         // temporal upsampler PATH beside it, which is a load knob
                         // for the same reason (:177).
                         std::make_pair("temporal_upsample_rounds", &temporal_rounds)};
  for (const auto& kv : retake_knobs) {
    if (kv.second->empty()) continue;
    gen_keys.emplace_back(kv.first);
    gen_values.push_back(*kv.second);
  }
  std::vector<const char*> gkeys, gvalues;
  for (size_t i = 0; i < gen_keys.size(); ++i) {
    gkeys.push_back(gen_keys[i].c_str());
    gvalues.push_back(gen_values[i].c_str());
  }
  if (!gkeys.empty()) {
    vp.extra_keys = gkeys.data();
    vp.extra_values = gvalues.data();
    vp.n_extras = static_cast<int32_t>(gkeys.size());
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

  vllm_video_engine* engine = nullptr;
  if (vllm_video_engine_load(&mp, &engine) != VLLM_OK) {
    std::fprintf(stderr, "error: %s\n", vllm_last_error());
    return 1;
  }
  // Which family actually loaded, from the handle rather than from the request:
  // spec §3.1 requires every artifact to name what produced it.
  std::fprintf(stderr, "ltx2-gen: family=%s dit=%s\n", vllm_video_engine_family(engine),
               mp.dit_path);

  vllm_video_result out;
  if (vllm_video_generate(engine, &vp, &out) != VLLM_OK) {
    std::fprintf(stderr, "error: %s\n", vllm_last_error());
    vllm_video_engine_free(engine);
    return 1;
  }
  std::fprintf(stderr, "  wrote %d frames (%dx%d @ %d fps) + %s (%d Hz)\n", out.frame_count,
               out.width, out.height, out.fps, out.audio_path, out.sample_rate);
  // WHERE THE RENDER SPENT ITS WALL (ABI v23, issue #1010). This example is a
  // client of `vllm.h` and nothing else, so it names the table by asking the
  // handle rather than by guessing a filename beside the frames. Printed on the
  // shipped default: a render long enough to need the table is one nobody knew
  // to instrument in advance, and a path printed after the fact is what makes
  // the evidence retrievable at all.
  const char* phase_log = vllm_video_last_phase_log(engine);
  if (phase_log != nullptr) {
    std::fprintf(stderr, "  phase table: %s\n", phase_log);
  } else {
    std::fprintf(stderr, "  phase table: none (this family emits no phase log)\n");
  }

  int status = 0;
  if (!out_path.empty()) {
    const std::string pattern = std::string(out.frame_dir) + "/frame_%06d.ppm";
    vllm_video_mux_params mx = vllm_video_mux_params_default();
    mx.frames = pattern.c_str();
    mx.audio_path = out.audio_path;
    mx.output_path = out_path.c_str();
    mx.fps = out.fps;  // the RECIPE's frame rate, not the mux default
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
