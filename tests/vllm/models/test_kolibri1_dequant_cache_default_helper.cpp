// The default-off probe for test_kolibri1_dequant_cache (case 5). A separate
// binary so the ProcessCache() static budget is initialized in a process whose
// environment PROVABLY lacks VT_KOLIBRI1_DEQUANT_CACHE_MB: the parent test
// exec's this helper with an empty environment, and this helper exits 0 iff
// the default configuration is a disabled cache (budget 0). Any default flip
// (e.g. a nonzero fallback MiB) makes this exit nonzero and the parent case
// fail. No doctest main: plain exit status is the contract.
#include <cstdlib>

#include "vllm/model_executor/models/kolibri1_dequant_cache.h"

int main() {
  if (std::getenv("VT_KOLIBRI1_DEQUANT_CACHE_MB") != nullptr) return 2;
  const auto& cache = vllm::kolibri1_dequant_cache::ProcessCache();
  return (cache.budget_bytes() == 0 && !cache.enabled()) ? 0 : 1;
}
