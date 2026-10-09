// #785 P1 host package: frozen fixture hashes, oracle invariants,
// exclusive trace classifier, CTest-registration lock. CPU-only.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "vt/sharedk_wmma_p1_fixture.h"

using namespace vt_785_p1;

TEST_CASE("P1 fixture freeze: shapes, seeds, hashes") {
  CHECK(kT >= 64);
  CHECK(kHq / kHk == 2);
  CHECK(kD == 256);
  const auto f = MakeFixture();
  CHECK(f.q_bf16.size() == static_cast<size_t>(kT * kHq * kD));
  CHECK(f.k_bf16.size() == static_cast<size_t>(f.num_blocks * kBlock * kHk * kD));
  CHECK(f.seq_lens.size() == 1);
  CHECK(f.seq_lens[0] == kT);
  CHECK(f.qsl.back() == kT);
  CHECK(Sha256U16Le(f.q_bf16) == kQHash);
  CHECK(Sha256U16Le(f.k_bf16) == kKHash);
  CHECK(Sha256U16Le(f.v_bf16) == kVHash);
}

TEST_CASE("P1 host f32 oracle freeze") {
  const auto f = MakeFixture();
  const auto ref = Oracle(f);
  CHECK(ref.size() == f.q_f32.size());
  double max_abs = 0, mean = 0;
  for (float x : ref) {
    CHECK(std::isfinite(x));
    max_abs = std::max(max_abs, static_cast<double>(std::abs(x)));
    mean += x;
  }
  mean /= static_cast<double>(ref.size());
  CHECK(max_abs > 1.0);
  CHECK(max_abs < 4.0);
  CHECK(std::abs(mean) < 0.05);
}

TEST_CASE("P1 BF16 tolerance: rounded-ref is GREEN; swapped is RED") {
  const auto f = MakeFixture();
  const auto ref = Oracle(f);
  CHECK(Score(ref, ref).ok);
  auto swapped = ref;
  if (!swapped.empty()) swapped[0] = -swapped[0] - 10.0f;
  CHECK_FALSE(Score(swapped, ref).ok);
}

TEST_CASE("P1 zero-variance / empty fail closed") {
  CHECK_FALSE(Score({}, {1.0f}).ok);
  CHECK_FALSE(Score({1.0f}, {}).ok);
  const std::vector<float> z(8, 0.0f);
  auto s = Score(z, z);
  CHECK(s.nonfinite == 0);
  CHECK(s.corr == doctest::Approx(0.0));
  CHECK_FALSE(s.ok);
}

TEST_CASE("P1 exclusive classifier: exact spec XOR; wrong family UNKNOWN") {
  const std::string a =
      "Dispatch,Kernel_Name\n"
      "0,\"void vt::rocm::(anonymous namespace)::"
      "PagedAttnPrefillSharedKWmma<2, 8, 16, 32, false>(__hip_bfloat16*)\"\n";
  const std::string b =
      "Dispatch,Kernel_Name\n"
      "0,\"void vt::rocm::(anonymous namespace)::"
      "PagedAttnPrefillSharedK<2, 8, 32, 32>(__hip_bfloat16*)\"\n";
  const std::string mangled_a =
      "_ZN2vt4rocm12_GLOBAL__N_127PagedAttnPrefillSharedKWmmaILi2ELi8ELi16ELi32ELb0EEEv";
  const std::string mangled_b =
      "_ZN2vt4rocm12_GLOBAL__N_124PagedAttnPrefillSharedKILi2ELi8ELi32ELi32EEEv";
  const std::string empty = "Dispatch,Kernel_Name\n";
  const std::string wrong_scalar =
      "PagedAttnPrefillSharedK<2, 8, 16, 32>";
  const std::string d512_scalar =
      "PagedAttnPrefillSharedK<2, 16, 16, 16>";
  const std::string wrong_wmma =
      "PagedAttnPrefillSharedKWmma<2, 8, 16, 16, false>";
  CHECK(ClassifyArm(a) == 'A');
  CHECK(ClassifyArm(b) == 'B');
  CHECK(ClassifyArm(mangled_a) == 'A');
  CHECK(ClassifyArm(mangled_b) == 'B');
  CHECK(ClassifyArm(empty) == '?');
  CHECK(ClassifyArm(a + b) == '?');
  CHECK(ClassifyArm(wrong_scalar) == '?');
  CHECK(ClassifyArm(d512_scalar) == '?');
  CHECK(ClassifyArm(wrong_wmma) == '?');
  CHECK(ClassifyArm(std::string("PagedAttnPrefillSharedKILi2ELi8ELi16ELi32EE")) == '?');
}

TEST_CASE("P1 eligibility pins SharedK d=256 qg=2 T>=64 one request") {
  CHECK(kT >= 64);
  CHECK(kD == 256);
  CHECK(kHq == 2);
  CHECK(kHk == 1);
  CHECK(kCausal);
  CHECK(kScale == doctest::Approx(1.0f / std::sqrt(256.0f)));
  CHECK(kWindowLeft == 32);
  CHECK(kWindowRight == 0);
}

TEST_CASE("P1 GPU binary is not registered as ordinary CTest") {
  // ABSOLUTE at configure time (tests/CMakeLists.txt): __FILE__ is
  // source-root-RELATIVE under -ffile-prefix-map, so a __FILE__-relative
  // probe path would resolve against the process CWD and miss under ctest.
#ifdef VLLM_CPP_TESTS_CMAKELISTS
  const std::string cmake = VLLM_CPP_TESTS_CMAKELISTS;
#else
  std::string path = __FILE__;
  const auto slash = path.find_last_of('/');
  REQUIRE(slash != std::string::npos);
  const std::string cmake = path.substr(0, slash) + "/../CMakeLists.txt";
#endif
  std::ifstream in(cmake);
  REQUIRE(in);
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  CHECK(text.find("test_ops_paged_attn_sharedk_wmma_p1_gpu") != std::string::npos);
  CHECK(text.find("add_test(NAME test_ops_paged_attn_sharedk_wmma_p1_gpu") ==
        std::string::npos);
  CHECK(text.find("vllm_cpp_add_test(test_ops_paged_attn_sharedk_wmma_p1_gpu") ==
        std::string::npos);
}
