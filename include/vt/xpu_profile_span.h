#pragma once
#include <memory>

namespace vt {
struct Queue;
namespace xpu {
struct ProfileSpanState;
// Optional diagnostic bracket. Disabled collection allocates nothing and
// submits no commands. Device intervals are queue spans, including host gaps;
// they are not the sum of kernel execution durations. No synchronization here.
struct ProfileSpan {
  ProfileSpan();
  ~ProfileSpan();
  ProfileSpan(ProfileSpan&&) noexcept;
  ProfileSpan& operator=(ProfileSpan&&) noexcept;
  std::unique_ptr<ProfileSpanState> state;
};
ProfileSpan BeginProfileSpan(Queue& queue);
// Consumes the bracket. stage must outlive DrainProfileEvents (use a literal).
void EndProfileSpan(Queue& queue, const char* stage, ProfileSpan span);
}  // namespace xpu
}  // namespace vt
