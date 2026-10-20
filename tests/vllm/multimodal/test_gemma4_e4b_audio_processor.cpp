#include <doctest/doctest.h>
#include <cmath>
#include <complex>
#include <limits>
#include "vllm/multimodal/gemma4_audio_processor.h"
#include "vllm/multimodal/audio_fft.h"
using vllm::multimodal::Gemma4AudioProcessor;
namespace {
// Independent scalar DFT and triangles, no production FFT/filter helpers.
// Source equations: HF a08ace4 audio_utils HTK and Gemma4 _extract_spectrogram.
double reference(const std::vector<float>& x, int t, int m) {
  const double pi = std::acos(-1.);
  double sum = 0;
  auto edge = [](int n) { return 700 * (std::exp(std::log(1 + 8000./700) * n / 129) - 1); };
  const double lo=edge(m), mid=edge(m+1), hi=edge(m+2);
  for (int k=0;k<=256;++k) {
    const double hz=k*31.25;
    if (hz <= lo || hz >= hi) continue;
    double re=0, im=0;
    for (int j=0;j<320;++j) {
      const int i=t*160+j-160;
      const float win=static_cast<float>(std::pow(std::sin(pi*j/320),2));
      const float v=i>=0 && i<static_cast<int>(x.size()) ? x[i]*win : 0;
      re+=v*std::cos(2*pi*k*j/512); im-=v*std::sin(2*pi*k*j/512);
    }
    sum+=std::hypot(re,im)*(hz<mid ? (hz-lo)/(mid-lo) : (hi-hz)/(hi-mid));
  }
  return std::log(sum+.001);
}
}
TEST_CASE("Gemma4 frontend independent scalar DFT impulse and multitone") {
  for(int kind=0;kind<2;++kind) {
    std::vector<float> x(801,0);
    if(kind==0) x[0]=1;
    else for(size_t i=0;i<x.size();++i) x[i]=static_cast<float>(.2*std::sin(i*.173)+.1*std::cos(i*.431));
    const auto a=Gemma4AudioProcessor().ProcessWaveform(x.data(),x.size(),16000);
    REQUIRE(a.n_frames==5); CHECK(a.num_tokens==2);
    for(int t=0;t<5;++t) for(int m=0;m<128;++m)
      CHECK(a.input_features[m*5+t]==doctest::Approx(reference(x,t,m)).epsilon(2e-6));
  }
}
TEST_CASE("Gemma4 frontend source framing boundary table and silence") {
  // Derived independently from end-sample test: 160+t*160 < raw length.
  const int lengths[]={0,1,128,129,160,161,256,320,321,480,481,640,641};
  const int frames[]={0,0,0,1,1,1,1,2,2,3,3,3,4};
  const int valid[]={0,0,0,0,0,1,1,1,2,2,3,3,4};
  for(int i=0;i<13;++i) {
    std::vector<float> x(lengths[i],0);
    const auto a=Gemma4AudioProcessor().ProcessWaveform(x.data(),x.size(),16000);
    CHECK(a.n_frames==frames[i]); CHECK(a.num_samples==lengths[i]);
    CHECK(a.num_tokens==(valid[i]+3)/4); REQUIRE(a.feature_mask.size()==static_cast<size_t>(frames[i]));
    for(int t=0;t<frames[i];++t) {
      CHECK(a.feature_mask[t]==(t<valid[i]));
      for(int m=0;m<128;++m) CHECK(a.input_features[m*frames[i]+t]==doctest::Approx(t<valid[i] ? -6.907755278982137 : 0));
    }
  }
}
TEST_CASE("Gemma4 frontend rejects malformed inputs and enforces source duration cap") {
  Gemma4AudioProcessor p; float x=0;
  CHECK_THROWS(p.ProcessWaveform(&x,-1,16000));
  CHECK_THROWS(p.ProcessWaveform(nullptr,1,16000));
  CHECK_THROWS(p.ProcessWaveform(&x,480001,16000));
  CHECK_THROWS(p.ProcessWaveform(&x,std::numeric_limits<int64_t>::max(),16000));
  for(int rate:{0,8000,44100,48000}) CHECK_THROWS(p.ProcessWaveform(&x,1,rate));
  for(float bad:{std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) CHECK_THROWS(p.ProcessWaveform(&bad,1,16000));
  std::vector<float> max(480000,0);
  const auto a=p.ProcessWaveform(max.data(),max.size(),16000);
  CHECK(a.n_frames==2999); CHECK(a.num_tokens==750); CHECK(a.feature_mask.back()==1);
}
TEST_CASE("shared FFT scalar DFT and invalid sizes") {
  for(int n:{1,2,8,512}) {
    std::vector<std::complex<double>> a(n);
    for(int i=0;i<n;++i) a[i]={std::sin(i*.17),std::cos(i*.31)};
    const auto input=a; vllm::multimodal::AudioFft(a);
    for(int k=0;k<n;++k) {
      std::complex<double> expected{};
      for(int j=0;j<n;++j) expected+=input[j]*std::polar(1.,-2*std::acos(-1.)*j*k/n);
      CHECK(std::abs(a[k]-expected)<1e-9);
    }
  }
  std::vector<std::complex<double>> empty, odd(3);
  CHECK_THROWS(vllm::multimodal::AudioFft(empty)); CHECK_THROWS(vllm::multimodal::AudioFft(odd));
}
#include "gemma4_frontend_goldens.inc"
TEST_CASE("Gemma4 frontend pinned source NumPy fixture") {
  std::vector<float> x(800);
  for(int i=0;i<800;++i) x[i]=static_cast<float>(((i*37)%101)-50)/64;
  const auto a=Gemma4AudioProcessor().ProcessWaveform(x.data(),800,16000);
  REQUIRE(a.input_features.size()==640);
  for(size_t i=0;i<640;++i) CHECK(a.input_features[i]==doctest::Approx(kSourceMel[i]).epsilon(2e-6));
}
