// Multimodal input containers — C++ mirror of vllm/multimodal/inputs.py.
//
// Ported from: vllm/multimodal/inputs.py (MultiModalKwargs / MultiModalInputs)
// @ vLLM e24d1b24. This is the M1 input-pipeline surface: the processed,
// per-modality feature tensors + grid metadata that the (M2) vision tower will
// consume, the placeholder-EXPANDED prompt ids, and the per-item mm-hashes.
//
// INERT-WHEN-OFF: every field defaults empty; a text-only request carries an
// empty MultiModalInputs and every downstream mm hook is a no-op.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vllm::multimodal {

// Processed features for ONE image item — the Qwen3-VL image branch of
// MultiModalKwargs (vllm/multimodal/inputs.py). `pixel_values` is the flattened
// patch matrix [num_patches, patch_feature_dim]; `image_grid_thw` = [t, h, w]
// (post-merge grid is h/merge x w/merge). vLLM casts mm_kwargs to the model
// dtype in processing/context.py::call_hf_processor -> _postprocess_output.
// Existing callers retain pre-cast float32 and BF16. The native EXL3 vision
// processor selects FP16 directly from float32, without a BF16 intermediate.
// Only the selected production vector is populated; float32 retention is
// optional for diagnostic comparisons. Consumers must check pixel_dtype.
enum class ImagePixelDType { kBF16, kF16 };

struct ImageKwargs {
  std::vector<float> pixel_values_f32;      // [num_patches * patch_feature_dim]
  std::vector<uint16_t> pixel_values_bf16;  // round-to-nearest-even of the above
  int64_t num_patches = 0;
  int64_t patch_feature_dim = 0;
  std::array<int64_t, 3> image_grid_thw{0, 0, 0};  // [grid_t, grid_h, grid_w]
  ImagePixelDType pixel_dtype = ImagePixelDType::kBF16;
  std::vector<uint16_t> pixel_values_f16;  // direct FP32 -> FP16, nearest even

  bool empty() const { return num_patches == 0; }
};

// Processed features for ONE video item — the Qwen3-VL video branch of
// MultiModalKwargs (vllm/multimodal/inputs.py, _process_video_input). Mirrors
// ImageKwargs but the temporal dim T>1: `pixel_values` is the flattened patch
// matrix [num_patches, patch_feature_dim] where num_patches = grid_t*grid_h*
// grid_w and each patch-row fuses temporal_patch_size REAL consecutive frames
// (source frame = grid_t_index*temporal_patch_size + t, NOT duplicated as an
// image is). Same bf16 model-dtype cast contract as ImageKwargs (verified:
// precast-bf16 == production golden on the fixture video).
struct VideoKwargs {
  std::vector<float> pixel_values_f32;      // [num_patches * patch_feature_dim]
  std::vector<uint16_t> pixel_values_bf16;  // round-to-nearest-even of the above
  int64_t num_patches = 0;
  int64_t patch_feature_dim = 0;
  std::array<int64_t, 3> video_grid_thw{0, 0, 0};  // [grid_t, grid_h, grid_w]
  std::vector<double> timestamps;                  // per temporal-group (len grid_t)

  bool empty() const { return num_patches == 0; }
};

// Processed features for ONE audio item — the AUDIO branch of MultiModalKwargs
// (audio-track A1; mirror of vllm/multimodal/inputs.py + the Whisper
// `input_features` field, whisper.py:103,738). `input_features` is the flattened
// log-mel spectrogram [n_mels, n_frames] (row-major), produced by the C++
// WhisperFeatureExtractor-equivalent pipeline. Unlike the image/video branch there
// is no bf16 pre-cast golden: the parity gate is a stated rel-L2 vs the oracle
// log-mel (FFT float ops make bit-exact infeasible; ids/hash stay exact).
struct AudioKwargs {
  std::vector<float> input_features;  // [n_mels * n_frames], row-major [n_mels, n_frames]
  int64_t n_mels = 0;
  int64_t n_frames = 0;

  // ── ADDED BY dots3-note W7a (#2703), additively ────────────────────────────
  //
  // Two lengths a mel matrix cannot carry, both DEFAULTED to 0 so every
  // existing producer and consumer is unchanged: Whisper's `RouteAudioWav`
  // fills neither and reads neither, and the runner and scheduler never look
  // inside this struct at all.
  //
  // `num_samples` is the waveform's OWN length BEFORE the front end padded it
  // to a whole chunk. dots3-note's stem masks the padded tail to zero at four
  // stages from `valid_mel_lens = num_samples // hop_length`
  // (`nvidia/audio_encoder.py:570-574` @ `9035151d6`), and it MUST: the mel of a
  // zero-padded tail is not zero, it is the `-8` global-max floor pushed
  // through `(x + 4) / 4`, a nonzero constant that would otherwise leak through
  // the 3x3 receptive fields into the last VALID tokens.
  //
  // `num_tokens` is the placeholder run length and the tower's output row
  // count, `ceil(num_samples / token_stride)` (`common/processor.py:771`).
  // It is NOT derivable from `num_samples // hop_length` halved three times:
  // at 1281 samples that gives 1 and this gives 2, and the difference is a
  // stem row the mask zeroed that the span still covers. Two numbers, carried
  // separately, because upstream computes them separately.
  int64_t num_samples = 0;
  int64_t num_tokens = 0;

  // ── ADDED BY dots3-note W7b (#2797), additively ────────────────────────────
  //
  // The CHUNK axis. `input_features` is `[num_chunks, n_mels, n_frames]`, one
  // `chunk_mel_frames`-wide padded mel per `chunk_seconds` segment, which is
  // upstream's `torch.stack(mel_features, dim=0)` (`nvidia/audio.py:220` @
  // `9035151d6`). At `num_chunks == 1` that layout is BYTE-IDENTICAL to the one
  // W7a produced, which is why the default is 1 and why every pre-W7b producer
  // and consumer is unchanged: Whisper's `RouteAudioWav` fills one mel and
  // leaves the two vectors empty.
  //
  // The two vectors are upstream's `audio_sample_lens` and `token_lens`
  // (`audio.py:217-218`) and they are NOT derivable from `num_samples` and
  // `num_tokens` above, which stay the WHOLE waveform's numbers: `num_tokens`
  // is the placeholder run length the prompt was expanded with, and
  // `chunk_num_tokens[i]` is how many rows chunk `i` contributes to it. The
  // last chunk is short, so its two entries are smaller than the others'.
  int64_t num_chunks = 1;
  std::vector<int64_t> chunk_num_samples;
  std::vector<int64_t> chunk_num_tokens;

  bool empty() const { return n_frames == 0; }
};

// One multimodal placeholder occupied in the prompt id stream — the C++ analogue
// of MultiModalFeatureSpec (vllm/multimodal/inputs.py). Carried on Request so the
// scheduler/encoder-cache seam can budget/allocate/free per item WITHOUT the
// text path ever seeing a non-empty vector.
struct MultiModalFeatureSpec {
  std::string mm_hash;                     // MultiModalHasher digest (hex)
  std::string modality = "image";          // "image" (M1); "video"/"audio" later
  int offset = 0;                          // start index in the expanded prompt
  int length = 0;                          // number of placeholder tokens (N)
  std::shared_ptr<ImageKwargs> data;       // image/video encoder input (opaque until M2)
  std::shared_ptr<AudioKwargs> audio_data; // audio encoder input (null unless modality=="audio")
};

// The processed bundle returned by the mm input pipeline for a single prompt —
// mirror of MultiModalInputs (prompt_token_ids + mm_kwargs + mm_hashes +
// mm_placeholders). Empty for text-only prompts.
struct MultiModalInputs {
  std::vector<int32_t> prompt_token_ids;             // placeholder-EXPANDED ids
  std::vector<MultiModalFeatureSpec> mm_features;    // one per placeholder item

  bool empty() const { return mm_features.empty(); }
};

}  // namespace vllm::multimodal
