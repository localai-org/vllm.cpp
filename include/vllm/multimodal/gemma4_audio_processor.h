// Pinned Google E4B waveform prerequisite; not a served processor registration.
#pragma once
#include "vllm/multimodal/inputs.h"
namespace vllm::multimodal {
class Gemma4AudioProcessor {
 public:
  // Mono F32 at 16kHz only. Reject >30s rather than silently truncate.
  // Empty/short inputs may have no valid tokens; encoder admission rejects them.
  AudioKwargs ProcessWaveform(const float* samples, int64_t count, int rate) const;
};
}  // namespace vllm::multimodal
