// SeamlessM4T feature extraction. See w2v_fbank.h for the upstream anchors.
#include "vllm/model_executor/models/w2v_fbank.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "vt/dtype.h"
#include "vllm/multimodal/audio_fft.h"

namespace vllm {
namespace models {
namespace w2v_fbank {
std::vector<double> KaldiMelFilterbankDouble(int64_t freq_bins, int64_t mel_bins,
                                            double min_hz, double max_hz,
                                            double sample_rate);

namespace {

constexpr double kPi = 3.14159265358979323846;

// mel <-> hz on the KALDI scale (1127 ln(1 + f/700)), not slaney.
// Only the forward direction is needed: the triangles are laid out in MEL
// space, so nothing ever converts back.
double HzToMel(double hz) { return 1127.0 * std::log(1.0 + hz / 700.0); }

// Iterative radix-2 FFT. `n` is a power of two; the caller only needs the first
// n/2 + 1 bins of a real transform, but computing the full complex one keeps
// this short and it is not on a hot path.

}  // namespace

std::vector<float> PoveyWindow(int64_t length) {
  VT_CHECK(length > 1, "w2v_fbank: window length must exceed 1");
  std::vector<float> w(static_cast<size_t>(length));
  for (int64_t i = 0; i < length; ++i) {
    // SYMMETRIC hann (periodic=False), then raised to 0.85.
    const double hann =
        0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) /
                             static_cast<double>(length - 1));
    w[static_cast<size_t>(i)] = static_cast<float>(std::pow(hann, 0.85));
  }
  return w;
}

std::vector<double> KaldiMelFilterbankDouble(int64_t freq_bins, int64_t mel_bins,
                                            double min_hz, double max_hz,
                                            double sample_rate) {
  VT_CHECK(freq_bins > 1 && mel_bins > 0, "w2v_fbank: bad filterbank shape");
  const double mel_min = HzToMel(min_hz);
  const double mel_max = HzToMel(max_hz);

  // `triangularize_in_mel_space`: the frequency axis is converted to mel and the
  // triangles are linear THERE, rather than linear in hz.
  std::vector<double> fft_mel(static_cast<size_t>(freq_bins));
  for (int64_t k = 0; k < freq_bins; ++k) {
    const double hz = sample_rate * 0.5 * static_cast<double>(k) /
                      static_cast<double>(freq_bins - 1);
    fft_mel[static_cast<size_t>(k)] = HzToMel(hz);
  }
  std::vector<double> points(static_cast<size_t>(mel_bins + 2));
  for (int64_t m = 0; m < mel_bins + 2; ++m) {
    points[static_cast<size_t>(m)] =
        mel_min + (mel_max - mel_min) * static_cast<double>(m) /
                      static_cast<double>(mel_bins + 1);
  }

  std::vector<double> fb(static_cast<size_t>(freq_bins * mel_bins), 0.0);
  for (int64_t m = 0; m < mel_bins; ++m) {
    const double left = points[static_cast<size_t>(m)];
    const double centre = points[static_cast<size_t>(m + 1)];
    const double right = points[static_cast<size_t>(m + 2)];
    for (int64_t k = 0; k < freq_bins; ++k) {
      const double f = fft_mel[static_cast<size_t>(k)];
      const double up = (f - left) / (centre - left);
      const double down = (right - f) / (right - centre);
      fb[static_cast<size_t>(k * mel_bins + m)] = std::max(0.0, std::min(up, down));
    }
  }
  return fb;
}

std::vector<float> KaldiMelFilterbank(int64_t freq_bins, int64_t mel_bins, double min_hz,
                                      double max_hz, double sample_rate) {
  const std::vector<double> d =
      KaldiMelFilterbankDouble(freq_bins, mel_bins, min_hz, max_hz, sample_rate);
  std::vector<float> fb(d.size());
  for (size_t i = 0; i < d.size(); ++i) {
    fb[i] = static_cast<float>(d[i]);
  }
  return fb;
}

std::vector<float> Extract(const Config& cfg, const std::vector<float>& audio,
                           double sample_rate, int64_t* frames,
                           std::vector<float>* raw_log_mel) {
  VT_CHECK(frames != nullptr, "w2v_fbank: frames out-parameter is required");
  VT_CHECK(!audio.empty(), "w2v_fbank: empty waveform");

  const int64_t n = static_cast<int64_t>(audio.size());
  const int64_t freq_bins = cfg.fft_length / 2 + 1;
  const std::vector<float> window = PoveyWindow(cfg.frame_length);
  // The filterbank is rebuilt in DOUBLE here. `KaldiMelFilterbank` returns
  // float for the public gate, but rounding the triangles to float before the
  // matmul costs ~1% on the low-energy mel bins, which the log then keeps and
  // the per-utterance normalisation amplifies.
  const std::vector<double> fb =
      KaldiMelFilterbankDouble(freq_bins, cfg.mel_bins, 20.0, sample_rate / 2.0,
                               sample_rate);

  // center=False, so frames stop when a full window no longer fits.
  const int64_t raw_frames = (n - cfg.frame_length) / cfg.hop + 1;
  VT_CHECK(raw_frames > 0, "w2v_fbank: waveform shorter than one frame");

  std::vector<float> mel(static_cast<size_t>(raw_frames * cfg.mel_bins));
  std::vector<std::complex<double>> buf(static_cast<size_t>(cfg.fft_length));
  std::vector<double> frame(static_cast<size_t>(cfg.frame_length));

  for (int64_t t = 0; t < raw_frames; ++t) {
    const int64_t off = t * cfg.hop;
    // Kaldi compliance: the waveform is scaled to 16-bit integer range first.
    double mean = 0.0;
    for (int64_t i = 0; i < cfg.frame_length; ++i) {
      frame[static_cast<size_t>(i)] =
          static_cast<double>(audio[static_cast<size_t>(off + i)]) * 32768.0;
      mean += frame[static_cast<size_t>(i)];
    }
    // remove_dc_offset: PER FRAME, and BEFORE preemphasis.
    mean /= static_cast<double>(cfg.frame_length);
    for (double& v : frame) {
      v -= mean;
    }
    // preemphasis, with the first sample using itself as its predecessor.
    for (int64_t i = cfg.frame_length - 1; i > 0; --i) {
      frame[static_cast<size_t>(i)] -=
          cfg.preemphasis * frame[static_cast<size_t>(i - 1)];
    }
    frame[0] -= cfg.preemphasis * frame[0];

    for (int64_t i = 0; i < cfg.fft_length; ++i) {
      const double x = (i < cfg.frame_length)
                           ? frame[static_cast<size_t>(i)] *
                                 static_cast<double>(window[static_cast<size_t>(i)])
                           : 0.0;
      buf[static_cast<size_t>(i)] = std::complex<double>(x, 0.0);
    }
    multimodal::AudioFft(buf);

    for (int64_t m = 0; m < cfg.mel_bins; ++m) {
      double acc = 0.0;
      for (int64_t k = 0; k < freq_bins; ++k) {
        const double re = buf[static_cast<size_t>(k)].real();
        const double im = buf[static_cast<size_t>(k)].imag();
        acc += (re * re + im * im) *
               fb[static_cast<size_t>(k * cfg.mel_bins + m)];
      }
      mel[static_cast<size_t>(t * cfg.mel_bins + m)] =
          static_cast<float>(std::log(std::max(acc, cfg.mel_floor)));
    }
  }

  if (raw_log_mel != nullptr) {
    *raw_log_mel = mel;
  }

  // Per-utterance normalization, with an UNBIASED variance (ddof=1).
  for (int64_t m = 0; m < cfg.mel_bins; ++m) {
    double sum = 0.0;
    for (int64_t t = 0; t < raw_frames; ++t) {
      sum += mel[static_cast<size_t>(t * cfg.mel_bins + m)];
    }
    const double mean = sum / static_cast<double>(raw_frames);
    double sq = 0.0;
    for (int64_t t = 0; t < raw_frames; ++t) {
      const double d = mel[static_cast<size_t>(t * cfg.mel_bins + m)] - mean;
      sq += d * d;
    }
    const double var =
        raw_frames > 1 ? sq / static_cast<double>(raw_frames - 1) : 0.0;
    const double inv = 1.0 / std::sqrt(var + cfg.norm_eps);
    for (int64_t t = 0; t < raw_frames; ++t) {
      float& v = mel[static_cast<size_t>(t * cfg.mel_bins + m)];
      v = static_cast<float>((static_cast<double>(v) - mean) * inv);
    }
  }

  // Pad the frame count UP to a multiple of `stride` (upstream's
  // `pad_to_multiple_of=2`), filling with `padding_value`. This happens AFTER
  // the normalization, so the pad value is not normalized with the rest.
  int64_t padded_frames = raw_frames;
  if (raw_frames % cfg.stride != 0) {
    padded_frames = raw_frames + (cfg.stride - raw_frames % cfg.stride);
    mel.resize(static_cast<size_t>(padded_frames * cfg.mel_bins), cfg.padding_value);
  }

  const int64_t out_frames = padded_frames / cfg.stride;
  const int64_t width = cfg.mel_bins * cfg.stride;
  std::vector<float> out(static_cast<size_t>(out_frames * width));
  for (int64_t t = 0; t < out_frames; ++t) {
    for (int64_t s = 0; s < cfg.stride; ++s) {
      for (int64_t m = 0; m < cfg.mel_bins; ++m) {
        out[static_cast<size_t>(t * width + s * cfg.mel_bins + m)] =
            mel[static_cast<size_t>((t * cfg.stride + s) * cfg.mel_bins + m)];
      }
    }
  }
  *frames = out_frames;
  return out;
}

}  // namespace w2v_fbank
}  // namespace models
}  // namespace vllm
