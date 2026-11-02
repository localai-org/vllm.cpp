#include <lumenengine.h>
int main() {
  return vllm_abi_version() == VLLM_ABI_VERSION && vllm_version()[0] ? 0 : 1;
}
