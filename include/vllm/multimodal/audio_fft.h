// Shared double radix-2 FFT extracted unchanged from w2v_fbank.cpp.
#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>
namespace vllm::multimodal {
inline void AudioFft(std::vector<std::complex<double>>& a) {
  constexpr double kPi = 3.14159265358979323846;
  const size_t n = a.size();
  if (n == 0 || (n & (n - 1))) throw std::invalid_argument("AudioFft: power-of-two size required");
  for (size_t i = 1, j = 0; i < n; ++i) {
    size_t bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      std::swap(a[i], a[j]);
    }
  }
  for (size_t len = 2; len <= n; len <<= 1) {
    const double ang = -2.0 * kPi / static_cast<double>(len);
    const std::complex<double> wl(std::cos(ang), std::sin(ang));
    for (size_t i = 0; i < n; i += len) {
      std::complex<double> w(1.0, 0.0);
      for (size_t k = 0; k < len / 2; ++k) {
        const std::complex<double> u = a[i + k];
        const std::complex<double> v = a[i + k + len / 2] * w;
        a[i + k] = u + v;
        a[i + k + len / 2] = u - v;
        w *= wl;
      }
    }
  }
}

}  // namespace vllm::multimodal
