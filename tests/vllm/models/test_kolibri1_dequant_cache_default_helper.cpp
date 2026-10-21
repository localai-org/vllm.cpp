// The default-budget probe for test_kolibri1_dequant_cache (cases 5, 8 and
// 9). A separate binary so the ProcessCache() static budget is initialized in
// a process whose environment PROVABLY lacks VT_KOLIBRI1_DEQUANT_CACHE_MB (or
// carries exactly the value the parent asked for): the parent test exec's
// this helper, and the helper exits 0 iff the configuration it observes is the
// one the case expects.
//
// MODES (argv[1]):
//   (absent)  the DEFAULT. The parent exec's this helper with an EMPTY
//             environment, so the variable is provably unset here, and the
//             cache must be ENABLED with the DERIVED budget.
//   "0"       DISABLED: budget 0, not enabled.
//   "<N>"     EXACT: budget N MiB, enabled iff N > 0.
//   anything  else  MALFORMED: the knob does not parse, so the DEFAULT
//             policy applies — an unparseable value must not silently
//             switch a default-ON lever off.
//
// The observed budget is printed on stdout so the parent can record it. No
// doctest main: the exit status is the contract.
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "vllm/model_executor/models/kolibri1_dequant_cache.h"

namespace {

using vllm::kolibri1_dequant_cache::DefaultBudgetBytesFor;
using vllm::kolibri1_dequant_cache::HonestAvailableBytes;
using vllm::kolibri1_dequant_cache::kDefaultBudgetKneeBytes;
using vllm::kolibri1_dequant_cache::kDefaultBudgetMaxBytes;

bool IsDigits(const char* s) {
  if (s == nullptr || *s == '\0') return false;
  for (const char* p = s; *p != '\0'; ++p) {
    if (*p < '0' || *p > '9') return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : nullptr;
  if (mode == nullptr) {
    // The DEFAULT mode: the parent exec's this helper with an EMPTY
    // environment, so the variable is provably unset here.
    if (std::getenv("VT_KOLIBRI1_DEQUANT_CACHE_MB") != nullptr) return 2;
  } else {
#if defined(_WIN32)
    if (_putenv_s("VT_KOLIBRI1_DEQUANT_CACHE_MB", mode) != 0) return 3;
#else
    if (setenv("VT_KOLIBRI1_DEQUANT_CACHE_MB", mode, 1) != 0) return 3;
#endif
  }
  const auto& cache = vllm::kolibri1_dequant_cache::ProcessCache();
  const unsigned long long budget = cache.budget_bytes();
  const unsigned long long source = cache.budget_source_bytes();
  const unsigned long long start = cache.budget_start_bytes();
  const unsigned long long plan = cache.budget_plan_bytes();
  std::printf("budget_bytes=%llu budget_source_bytes=%llu "
              "budget_start_bytes=%llu budget_plan_bytes=%llu enabled=%d\n",
              budget, source, start, plan, cache.enabled() ? 1 : 0);
  std::fflush(stdout);

  bool ok = false;
  if (mode == nullptr) {
    // The DEFAULT: the policy's answer for this host, and nothing else. The
    // budget is exactly the policy applied to the HONEST reading the cache
    // derived it from (the maximum of the process-start reading and the
    // reading at plan time).
    ok = source == HonestAvailableBytes(start, plan) &&
         budget == DefaultBudgetBytesFor(source) &&
         cache.enabled() == (budget > 0);
    // The KNEE: a budget is either DISABLED (a host that cannot afford the
    // knee) or at or above it, so the default never ships the thrashing
    // shape.
    ok = ok && (budget == 0 || (budget >= kDefaultBudgetKneeBytes &&
                                budget <= kDefaultBudgetMaxBytes));
  } else if (std::strcmp(mode, "0") == 0) {
    // The explicit OFF switch.
    ok = !cache.enabled() && budget == 0;
  } else if (IsDigits(mode)) {
    // The explicit budget, exactly.
    const unsigned long long mib = std::strtoull(mode, nullptr, 10);
    ok = budget == mib * 1024ull * 1024ull && cache.enabled() == (mib > 0);
  } else {
    // MALFORMED: the DEFAULT policy applies — the knob does not parse, so it
    // is not the explicit "0" (the off switch) and it is not an exact
    // budget either: it is the default policy's answer, byte for byte.
    ok = budget == DefaultBudgetBytesFor(source) &&
         cache.enabled() == (budget > 0);
  }
  return ok ? 0 : 1;
}
