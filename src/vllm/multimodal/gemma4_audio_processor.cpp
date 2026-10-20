// Transformers a08ace4bbd97e721c98751deec37d87b026acadc:
// Gemma4AudioFeatureExtractor._extract_spectrogram and __call__.
#include "vllm/multimodal/gemma4_audio_processor.h"
#include "vllm/multimodal/audio_fft.h"
#include "vllm/multimodal/mel_filter_bank.h"

namespace vllm::multimodal {
AudioKwargs Gemma4AudioProcessor::ProcessWaveform(const float* samples,
                                                  int64_t count, int rate) const {
  if (rate != 16000 || count < 0 || count > 480000 || (count && !samples))
    throw std::invalid_argument("Gemma4 audio: require mono 16kHz, 0..480000 samples");
  for (int64_t i = 0; i < count; ++i)
    if (!std::isfinite(samples[i])) throw std::invalid_argument("Gemma4 audio: nonfinite waveform");
  AudioKwargs out;
  out.n_mels = 128;
  out.num_samples = count;
  const int64_t padded = (count + 127) / 128 * 128;
  out.n_frames = padded < 161 ? 0 : (padded - 161) / 160 + 1;
  const int64_t valid = count < 161 ? 0 : (count - 161) / 160 + 1;
  out.num_tokens = (valid + 3) / 4;
  out.feature_mask.resize(out.n_frames, 0);
  out.input_features.resize(128 * out.n_frames, 0);
  static const auto filters = MelFilterBankHtk(257, 128, 0, 8000, 16000);
  constexpr double pi = 3.14159265358979323846;
  std::vector<float> window(320);
  for (int j = 0; j < 320; ++j) window[j] = static_cast<float>(.5 - .5 * std::cos(2 * pi * j / 320));
  for (int64_t t = 0; t < valid; ++t) {
    out.feature_mask[t] = 1;
    std::vector<std::complex<double>> frame(512);
    for (int j = 0; j < 320; ++j) {
      const int64_t i = t * 160 + j - 160;
      // Explicit F32 product, then double FFT, like NumPy's float32 frames.
      const float value = i >= 0 && i < count ? samples[i] * window[j] : 0;
      frame[j] = value;
    }
    AudioFft(frame);
    for (int m = 0; m < 128; ++m) {
      double mel = 0;
      for (int k = 0; k < 257; ++k) mel += std::abs(frame[k]) * filters[k * 128 + m];
      out.input_features[m * out.n_frames + t] = static_cast<float>(std::log(mel + .001));
    }
  }
  return out;
}
}  // namespace vllm::multimodal
