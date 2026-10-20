// See include/vllm/multimodal/mel_filter_bank.h for the full provenance and for
// why this is a shared seam rather than a fourth copy.
//
// EXTRACTED from `parakeet_audio_processor.cpp:43-106` (the `ParakeetMelFilterBank`
// this file replaces) with two changes and no third: `min_frequency` and
// `max_frequency` became arguments instead of `0.0` and `sampling_rate / 2.0`,
// and the result is emitted in upstream's `[num_frequency_bins,
// num_mel_filters]` orientation with the transpose offered separately. Every
// arithmetic line is the one that was there, so Parakeet's two gated tolerances
// are unmoved.
#include "vllm/multimodal/mel_filter_bank.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace vllm::multimodal {
namespace {

// np.linspace(start, stop, num): step = (stop-start)/(num-1), and the LAST
// element is set to `stop` exactly.
std::vector<double> Linspace(double start, double stop, int num) {
  std::vector<double> out(static_cast<size_t>(num));
  if (num == 1) {
    out[0] = start;
    return out;
  }
  const double step = (stop - start) / static_cast<double>(num - 1);
  for (int i = 0; i < num; ++i) out[static_cast<size_t>(i)] = start + step * i;
  out[static_cast<size_t>(num - 1)] = stop;
  return out;
}

}  // namespace

double HertzToMelSlaney(double hertz) {
  // audio_utils.py:285-296.
  constexpr double kMinLogHertz = 1000.0;
  constexpr double kMinLogMel = 15.0;
  const double logstep = 27.0 / std::log(6.4);
  if (hertz >= kMinLogHertz) {
    return kMinLogMel + std::log(hertz / kMinLogHertz) * logstep;
  }
  return 3.0 * hertz / 200.0;
}

double MelToHertzSlaney(double mels) {
  // audio_utils.py:321-331.
  constexpr double kMinLogHertz = 1000.0;
  constexpr double kMinLogMel = 15.0;
  const double logstep = std::log(6.4) / 27.0;
  if (mels >= kMinLogMel) {
    return kMinLogHertz * std::exp(logstep * (mels - kMinLogMel));
  }
  return 200.0 * mels / 3.0;
}

std::vector<float> MelFilterBankSlaney(int num_frequency_bins,
                                       int num_mel_filters,
                                       double min_frequency,
                                       double max_frequency,
                                       int sampling_rate) {
  if (num_frequency_bins < 2) {
    throw std::runtime_error("MelFilterBankSlaney: num_frequency_bins < 2");
  }
  if (num_mel_filters < 1) {
    throw std::runtime_error("MelFilterBankSlaney: num_mel_filters < 1");
  }
  const int n_freq = num_frequency_bins;
  const int n_mels = num_mel_filters;

  // audio_utils.py:516-519 — the filter centres, equally spaced in MEL space.
  const double mel_min = HertzToMelSlaney(min_frequency);
  const double mel_max = HertzToMelSlaney(max_frequency);
  const std::vector<double> mel_freqs = Linspace(mel_min, mel_max, n_mels + 2);
  std::vector<double> filter_freqs(mel_freqs.size());
  for (size_t i = 0; i < mel_freqs.size(); ++i) {
    filter_freqs[i] = MelToHertzSlaney(mel_freqs[i]);
  }
  // :528 — `np.linspace(0, sampling_rate // 2, num_frequency_bins)`, so the top
  // is the INTEGER-divided Nyquist. At an ODD sampling rate that is NOT
  // `max_frequency`, and the two are therefore separate inputs rather than one
  // value used twice.
  const std::vector<double> fft_freqs =
      Linspace(0.0, static_cast<double>(sampling_rate / 2), n_freq);

  // :371-375 `_create_triangular_filter_bank`, then :532-535 the slaney area
  // normalisation. Emitted as [n_freq, n_mels], upstream's own orientation.
  std::vector<float> out(static_cast<size_t>(n_freq) * n_mels, 0.0f);
  for (int m = 0; m < n_mels; ++m) {
    const double diff_lo = filter_freqs[static_cast<size_t>(m) + 1] -
                           filter_freqs[static_cast<size_t>(m)];
    const double diff_hi = filter_freqs[static_cast<size_t>(m) + 2] -
                           filter_freqs[static_cast<size_t>(m) + 1];
    const double enorm = 2.0 / (filter_freqs[static_cast<size_t>(m) + 2] -
                                filter_freqs[static_cast<size_t>(m)]);
    for (int k = 0; k < n_freq; ++k) {
      const double down =
          -(filter_freqs[static_cast<size_t>(m)] -
            fft_freqs[static_cast<size_t>(k)]) /
          diff_lo;
      const double up = (filter_freqs[static_cast<size_t>(m) + 2] -
                         fft_freqs[static_cast<size_t>(k)]) /
                        diff_hi;
      const double tri = std::max(0.0, std::min(down, up));
      out[static_cast<size_t>(k) * n_mels + m] = static_cast<float>(tri * enorm);
    }
  }
  return out;
}

std::vector<float> MelFilterBankSlaneyTransposed(int num_frequency_bins,
                                                 int num_mel_filters,
                                                 double min_frequency,
                                                 double max_frequency,
                                                 int sampling_rate) {
  const std::vector<float> kn = MelFilterBankSlaney(
      num_frequency_bins, num_mel_filters, min_frequency, max_frequency,
      sampling_rate);
  std::vector<float> out(kn.size());
  for (int k = 0; k < num_frequency_bins; ++k) {
    for (int m = 0; m < num_mel_filters; ++m) {
      out[static_cast<size_t>(m) * num_frequency_bins + k] =
          kn[static_cast<size_t>(k) * num_mel_filters + m];
    }
  }
  return out;
}

std::vector<double> MelFilterBankHtk(int frequency_bins, int mel_bins,
                                     double min_hz, double max_hz, int sample_rate) {
  if (frequency_bins < 2 || mel_bins < 1 || mel_bins > 4096 ||
      sample_rate <= 0 || !std::isfinite(min_hz) || !std::isfinite(max_hz) ||
      min_hz < 0 || max_hz <= min_hz || max_hz > sample_rate / 2)
    throw std::invalid_argument("MelFilterBankHtk: invalid configuration");
  const auto mel = Linspace(2595 * std::log10(1 + min_hz / 700),
                            2595 * std::log10(1 + max_hz / 700), mel_bins + 2);
  std::vector<double> hz(mel.size());
  for (size_t i = 0; i < hz.size(); ++i) hz[i] = 700 * (std::pow(10., mel[i] / 2595) - 1);
  const auto freq = Linspace(0, sample_rate / 2, frequency_bins);
  std::vector<double> out(static_cast<size_t>(frequency_bins) * mel_bins);
  for (int k = 0; k < frequency_bins; ++k)
    for (int m = 0; m < mel_bins; ++m)
      out[static_cast<size_t>(k) * mel_bins + m] = std::max(0., std::min(
          (freq[k] - hz[m]) / (hz[m+1] - hz[m]),
          (hz[m+2] - freq[k]) / (hz[m+2] - hz[m+1])));
  return out;
}

}  // namespace vllm::multimodal
