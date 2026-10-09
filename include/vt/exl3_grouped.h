#pragma once
#include <cstddef>
#include <array>
#include <memory>
#include "vt/ops.h"
#include "vt/shared_ptr_cache.h"
#include "vt/exl3_w8a8_panel_plan.h"

namespace vt {
class Exl3W8A8ModelMap;
// Pinned EXL3 mul1 SmallM launch geometry. M is the physical row count,
// including graph padding; M>128 needs the separate W8A8 implementation.
struct Exl3SmallMPlan {
  bool vector;
  int row_block, padded_rows, tiles_per_thread, splits, tile_rows_per_split;
};
Exl3SmallMPlan PlanExl3SmallM(int64_t m, int64_t k, int64_t n, int bits);

struct Exl3GroupedLinearArgs {
  int bits = 4;
  int codebook = 2;
  const char* debug_name = nullptr;
  // Internal large-M A/B control; SmallM arithmetic is unaffected.
  int w8a8_panel_columns = 128;
  // Private model dispatch payload. All public grouped overloads discard it;
  // caller-provided tensors always undergo their actual map readback/check.
  const Exl3W8A8ModelMap* model_map = nullptr;
};
using Exl3GroupedLinearFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, const Tensor&, Tensor&, Tensor&,
    const Exl3GroupedLinearArgs&);

// F16[M,K] -> F16[M,N], opaque trellis bytes [K/16,N/16,32*bits].
// suh is F16[S,K]; shard_of_nb is I32[N/128], selecting an independent input
// transform for each whole 128-column output block; svh is F16[N].
// Caller-owned scratch must remain alive until the queue completes:
//   in_had: F16[S,K/16,plan.padded_rows,16], blocked by input tile/row;
//   partials: F32[plan.splits,M,N], ordered split-K accumulation.
// Scratch/output may not overlap any operand or each other. This initial
// native XPU implementation supports mul1 4/6bpw and physical M<=128 only;
// unsupported arithmetic is refused rather than replaced by Packed/BF16.
void Exl3GroupedLinear(Queue&, Tensor& out, const Tensor& in, const Tensor& trellis,
    const Tensor& suh, const Tensor& svh, const Tensor& shard_of_nb,
    Tensor& in_had, Tensor& partials, const Exl3GroupedLinearArgs&);

// Independent large-M arithmetic, with the producer's 256-row GEMM padding.
// Offsets are bytes in one caller-owned I8 workspace. All regions start on a
// 64-byte boundary: I8[S,Ms,K], F32[S,Ms], F16[Ms,N], F32[1]. No weight cache:
// a separate bounded I8[K,weight_panel_columns] panel is overwritten on the
// same in-order queue. Tail panels use a compact prefix of that capacity.
struct Exl3W8A8Plan {
  int padded_rows;
  size_t activation_offset, row_scale_offset, intermediate_offset;
  size_t weight_scale_offset, workspace_bytes, weight_panel_bytes;
  int weight_panel_columns;
};
Exl3W8A8Plan PlanExl3W8A8(int64_t m, int64_t k, int64_t n,
                         int64_t groups, int bits, int panel_columns = 128);

// Internal model A/B selection; caller-owned arithmetic defaults remain128.
// Validate this setting at Large-M admission, before uploading model operands.
int Exl3W8A8ModelPanelColumns();

// F16 input/output and packed/group metadata have the same layouts as SmallM.
// M must be in [129,4096]; dispatch is explicit, never an FP16/GPTQ fallback.
// Finite model operands are required. Zero rows use scale1 and signed INT8
// zero, including padded rows. Workspace/panel/output must be disjoint from
// all operands and each other, and live until queue completion. Initial route
// is eager, without graph qualification. oneDNN user scratch is queue-owned
// and accounted separately from these caller-owned buffers.
void Exl3GroupedW8A8(Queue&, Tensor& out, const Tensor& in, const Tensor& trellis,
    const Tensor& suh, const Tensor& svh, const Tensor& shard_of_nb,
    Tensor& workspace, Tensor& weight_panel, const Exl3GroupedLinearArgs&);

// Eager model overload: one backend-owned workspace/panel shared by all model
// projections and queues on the device. Completes consumers before releasing
// the lease; growth and reuse cannot invalidate captured storage (this overload
// refuses capture). Explicit caller-owned overload remains the diagnostic A/B.
void Exl3GroupedW8A8(Queue&, Tensor& out, const Tensor& in, const Tensor& trellis,
    const Tensor& suh, const Tensor& svh, const Tensor& shard_of_nb,
    const Exl3GroupedLinearArgs&);

// Immutable model-only panel metadata. Allocation control-block identity is
// its residency generation, not just the data address. A replacement upload
// or device/shape/group change must build a new certificate. In-place edits
// require a new residency generation; untrusted tensors use the public APIs.
class Exl3W8A8ModelMap {
 public:
  Exl3W8A8ModelMap(Queue&, const Tensor& map, int groups,
                  const std::shared_ptr<void>& resident_owner);
  bool Matches(const Tensor& map, int groups,
               const std::shared_ptr<void>& resident_owner) const;
  const std::vector<Exl3W8A8Panel>& Panels(int columns) const;
  const std::shared_ptr<void>& ResidentOwner() const { return owner_; }
 private:
  Tensor map_;
  int groups_;
  std::shared_ptr<void> owner_;
  std::array<std::vector<Exl3W8A8Panel>, 3> panels_;
};

namespace detail {
// Same SmallM operand/scratch checks as the public operator, with versioned
// immutable model routing. Cold/stale capture uses ordinary replay guards;
// warmed capture pins the map owner and retains graph read-only protection.
void Exl3GroupedLinearModel(Queue&, Tensor& out, const Tensor& in,
    const Tensor& trellis, const Tensor& suh, const Tensor& svh, const Tensor& map,
    Tensor& in_had, Tensor& partials, const Exl3GroupedLinearArgs&,
    const std::shared_ptr<void>& resident_owner,
    SharedPtrCache<const Exl3W8A8ModelMap>& cache);
// Only model-resident immutable maps may use this seam. All ordinary operand,
// activation, scale, alias and workspace checks still execute. The cache owns
// its map allocation, is bounded by its projection owner and is never global.
void Exl3GroupedW8A8Model(Queue&, Tensor& out, const Tensor& in,
    const Tensor& trellis, const Tensor& suh, const Tensor& svh, const Tensor& map,
    const Exl3GroupedLinearArgs&, const std::shared_ptr<void>& resident_owner,
    SharedPtrCache<const Exl3W8A8ModelMap>& cache);
}
}  // namespace vt
