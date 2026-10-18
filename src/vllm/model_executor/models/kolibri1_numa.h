// Kolibri-1 in-process NUMA interleaving for the weight arenas
// (ISSUE-LOCAL-01M4GQ82X99D6JNBWDPX7ATJPY).
//
// WHY: the kolibri1 CPU decode is bandwidth-bound on weight streaming, and
// the default first-touch allocation parks the loader's and the dequant
// cache's pages on the loading thread's NUMA node. On this host (128 cores,
// 4 nodes) a 32-thread decode then streams most weight bytes REMOTE; the
// external wrapper `numactl --interleave=all` measured +47% decode tok/s
// (3.17-3.21 → 4.69; docs/bench-evidence/kolibri1-perf-next-lever-20261008.md
// and the PR #3423 repair section). A wrapper cannot be part of a production
// entry point, so the same policy is applied here, in-process, over the two
// weight-arena allocation sites:
//
//   * `LoadKolibri1Weights` (kolibri1_weights.cpp) — the fp8/bf16 weight
//     bytes materialized at load;
//   * `kolibri1_dequant_cache::Cache::GetOrDequant` (kolibri1_dequant_cache.h)
//     — the resident bf16 blocks the decode streams (16 GiB at
//     VT_KOLIBRI1_DEQUANT_CACHE_MB=16384).
//
// MECHANISM: `set_mempolicy(MPOL_INTERLEAVE, allowed_nodes)` is per-task
// state; while it holds, every page this task faults lands striped
// round-robin over the node mask, whichever allocator produced the pointer —
// so no allocation site or custom allocator has to change. `PolicyGuard`
// scopes the policy and restores the previous one (get_mempolicy round-trip,
// so a nested guard and an enclosing numactl both survive). It is allocation
// policy ONLY: pages move, bytes do not, and the numerics gates (W3 anchor
// 109726, chain md5 8de1463a) hold unchanged on every A/B leg.
//
// SYSINFO: the raw `set_mempolicy`/`get_mempolicy` syscalls are used, not
// libnuma — the policy needs no cpuset bookkeeping, and a syscall-only
// surface adds no link-time dependency to the shared seam for every host
// that does not have libnuma installed.
//
// KNOB: `VT_KOLIBRI1_NUMA_INTERLEAVE=0` disables the guard entirely (the
// off legs of the A/B); unset or any other value enables. Default ON with
// the measured evidence recorded in the issue and the bench-evidence file.

#pragma once

#include <cstdlib>
#include <string>

#if defined(__linux__) && !defined(_WIN32)
#include <cstddef>
#include <cstdint>

#include <linux/mempolicy.h>  // MPOL_INTERLEAVE, MPOL_F_*
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "vt/dtype.h"  // VT_CHECK

namespace vllm {
namespace kolibri1_numa {

// Knob: enabled unless `VT_KOLIBRI1_NUMA_INTERLEAVE` is exactly "0". Read on
// every guard construction (uncached) so a test binary can flip the env —
// the call rate is per load and per cache miss, never per element.
inline bool Enabled() {
#if defined(__linux__) && !defined(_WIN32)
  const char* v = std::getenv("VT_KOLIBRI1_NUMA_INTERLEAVE");
  return v == nullptr || std::string(v) != "0";
#else
  return false;
#endif
}

#if defined(__linux__) && !defined(_WIN32)

namespace detail {

// Wide enough for any current Linux: CONFIG_NODES_SHIFT caps at 10, so 1024
// possible nodes (128 unsigned longs) bound the kernel's own MAX_NUMNODES.
constexpr unsigned kMaxNodes = 1024;
using NodeMask = unsigned long[kMaxNodes / 64];

inline int GetMempolicy(int* mode, unsigned long* nodemask, unsigned long maxnode,
                        void* addr, int flags) {
  return static_cast<int>(
      syscall(SYS_get_mempolicy, mode, nodemask, maxnode, addr, flags));
}
inline int SetMempolicy(int mode, const unsigned long* nodemask,
                        unsigned long maxnode) {
  return static_cast<int>(
      syscall(SYS_set_mempolicy, mode, nodemask, maxnode));
}

// Query the allowed-node mask and its bit width. False when the kernel
// refuses (a non-NUMA build).
inline bool AllowedNodes(unsigned long* mask, unsigned long* maxnode) {
  int mode = 0;
  if (GetMempolicy(&mode, mask, kMaxNodes, nullptr, MPOL_F_MEMS_ALLOWED) != 0) {
    return false;
  }
  // Highest set bit + 1, so the mask carries no stale bits past the last
  // allowed node (the kernel rejects a nodemask that names a node outside
  // its maxnode range).
  unsigned bits = 1;
  for (unsigned w = 0; w < kMaxNodes / 64; ++w) {
    for (unsigned b = 0; b < 64; ++b) {
      if (mask[w] & (1UL << b)) bits = w * 64 + b + 1;
    }
  }
  *maxnode = bits;
  return true;
}

}  // namespace detail

// Query the NUMA node backing one page (get_mempolicy MPOL_F_NODE |
// MPOL_F_ADDR). False when the kernel refuses (e.g. non-page memory).
inline bool PageNode(const void* addr, int* node) {
  int mode = 0;
  if (detail::GetMempolicy(&mode, nullptr, 0, const_cast<void*>(addr),
                           MPOL_F_NODE | MPOL_F_ADDR) != 0) {
    return false;
  }
  *node = mode;
  return true;
}

inline bool CurrentPolicyIsInterleave() {
  int mode = 0;
  return detail::GetMempolicy(&mode, nullptr, 0, nullptr, 0) == 0 &&
         mode == MPOL_INTERLEAVE;
}

// Scoped MPOL_INTERLEAVE over every allowed node. Restores the previous
// policy on destruction.
class PolicyGuard {
 public:
  PolicyGuard() : active_(Enabled()) {
    if (!active_) return;
    VT_CHECK(detail::GetMempolicy(&saved_mode_, saved_mask_, detail::kMaxNodes,
                                  nullptr, 0) == 0,
             "kolibri1_numa: get_mempolicy failed");
    unsigned long maxnode = 0;
    unsigned long allowed[detail::kMaxNodes / 64] = {0};
    VT_CHECK(detail::AllowedNodes(allowed, &maxnode),
             "kolibri1_numa: MPOL_F_MEMS_ALLOWED query failed");
    VT_CHECK(detail::SetMempolicy(MPOL_INTERLEAVE, allowed, maxnode) == 0,
             "kolibri1_numa: set_mempolicy(MPOL_INTERLEAVE) failed");
  }
  ~PolicyGuard() {
    if (!active_) return;
    detail::SetMempolicy(saved_mode_,
                         saved_mode_ == MPOL_DEFAULT ? nullptr : saved_mask_,
                         detail::kMaxNodes);
  }
  PolicyGuard(const PolicyGuard&) = delete;
  PolicyGuard& operator=(const PolicyGuard&) = delete;

 private:
  bool active_;
  int saved_mode_ = MPOL_DEFAULT;
  detail::NodeMask saved_mask_ = {0};
};

#else  // non-Linux: the knob reads false and the guard is a no-op.

inline bool PageNode(const void*, int*) { return false; }
inline bool CurrentPolicyIsInterleave() { return false; }
class PolicyGuard {};

#endif

}  // namespace kolibri1_numa
}  // namespace vllm
