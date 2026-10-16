#include "vt/exl3_grouped.h"
#include "vt/backend.h"
#ifdef VLLM_CPP_XPU
#include "vt/xpu_graph_metadata.h"
#endif
#include "vt/exl3_w8a8_panel_plan.h"
#include <algorithm>
#include <atomic>
#include <limits>
#include <cstdlib>
#include <string_view>

namespace vt {
int Exl3W8A8ModelPanelColumns() {
  const char* value = std::getenv("VT_XPU_EXL3_W8A8_PANEL_COLUMNS");
  // Smallest bounded candidate with a measured complete-prefill gain. The
  // explicit128 route remains available for identical-input diagnostics.
  if (!value) return 1024;
  const std::string_view width(value);
  if (width == "128") return 128;
  if (width == "1024") return 1024;
  if (width == "2048") return 2048;
  VT_CHECK(false, "VT_XPU_EXL3_W8A8_PANEL_COLUMNS must be128/1024/2048");
  return 1024;
}

Exl3W8A8Plan PlanExl3W8A8(int64_t m, int64_t k, int64_t n,
                         int64_t groups, int bits, int panel_columns) {
  VT_CHECK(m > 128 && m <= 4096, "EXL3 W8A8 requires physical M in [129,4096]");
  VT_CHECK(k > 0 && n > 0 && k % 128 == 0 && n % 128 == 0 &&
               k <= std::numeric_limits<int>::max() &&
               n <= std::numeric_limits<int>::max() && groups > 0 &&
               groups <= std::numeric_limits<int>::max() / m / 16,
           "EXL3 W8A8 invalid geometry or I32 input launch overflow");
  VT_CHECK(bits == 4 || bits == 6, "EXL3 W8A8 supports 4/6bpw only");
  VT_CHECK(k <= std::numeric_limits<int32_t>::max() / (127 * 127),
           "EXL3 W8A8 K exceeds bounded signed INT32 accumulation");
  VT_CHECK(m * (n / 128) <= std::numeric_limits<int>::max(),
           "EXL3 W8A8 output launch exceeds I32 indexing");
  const int ms = int((m + 255) / 256 * 256);
  auto multiply = [](size_t a, size_t b) {
    VT_CHECK(b == 0 || a <= (std::numeric_limits<size_t>::max() - 63) / b,
             "EXL3 W8A8 workspace size overflow");
    return a * b;
  };
  size_t cursor = 0;
  auto region = [&](size_t bytes) {
    const size_t start = cursor;
    VT_CHECK(bytes <= std::numeric_limits<size_t>::max() - cursor - 63,
             "EXL3 W8A8 workspace size overflow");
    cursor += (bytes + 63) / 64 * 64;
    return start;
  };
  const size_t rows = multiply(size_t(groups), size_t(ms));
  const size_t act = region(multiply(rows, size_t(k)));
  const size_t sx = region(multiply(rows, sizeof(float)));
  const size_t y = region(multiply(multiply(size_t(ms), size_t(n)), size_t{2}));
  const size_t sw = region(sizeof(float));
  const auto panel = PlanExl3W8A8PanelCapacity(k, n, panel_columns);
  return {ms, act, sx, y, sw, cursor, panel.bytes, panel.columns};
}

namespace {
Exl3W8A8Plan ValidateW8A8Operands(Queue& q, const Tensor& out, const Tensor& in,
    const Tensor& tr, const Tensor& suh, const Tensor& svh, const Tensor& shard,
    const Exl3GroupedLinearArgs& args) {
  VT_CHECK(in.rank == 2 && out.rank == 2 && suh.rank == 2,
           "EXL3 W8A8 requires rank-2 input/output/suh");
  const int64_t m = in.shape[0], k = in.shape[1], n = out.shape[1];
  const auto plan = PlanExl3W8A8(m, k, n, suh.shape[0], args.bits, args.w8a8_panel_columns);
  VT_CHECK(args.codebook == 2 && out.shape[0] == m &&
               in.dtype == DType::kF16 && out.dtype == DType::kF16,
           "EXL3 W8A8 requires mul1 and matching F16 input/output");
  VT_CHECK(tr.dtype == DType::kI8 && tr.rank == 3 && tr.shape[0] == k / 16 &&
               tr.shape[1] == n / 16 && tr.shape[2] == 32 * args.bits,
           "EXL3 W8A8 packed trellis layout mismatch");
  VT_CHECK(suh.dtype == DType::kF16 && suh.shape[1] == k &&
               svh.dtype == DType::kF16 && svh.rank == 1 && svh.shape[0] == n,
           "EXL3 W8A8 requires F16 suh[S,K] and svh[N]");
  VT_CHECK(shard.dtype == DType::kI32 && shard.rank == 1 && shard.shape[0] == n / 128,
           "EXL3 W8A8 requires I32 shard_of_nb[N/128]");
  for (const Tensor* t : {&out, &in, &tr, &suh, &svh, &shard}) {
    VT_CHECK(t->IsContiguous(), "EXL3 W8A8 requires contiguous tensors");
    VT_CHECK(t->device == q.device, "EXL3 W8A8 device mismatch");
  }
  return plan;
}
}  // namespace

void Exl3GroupedW8A8(Queue& q, Tensor& out, const Tensor& in, const Tensor& tr,
    const Tensor& suh, const Tensor& svh, const Tensor& shard,
    Tensor& workspace, Tensor& panel, const Exl3GroupedLinearArgs& args) {
  const auto plan = ValidateW8A8Operands(q, out, in, tr, suh, svh, shard, args);
  const int64_t k = in.shape[1];
  VT_CHECK(workspace.dtype == DType::kI8 && workspace.rank == 1 &&
               workspace.shape[0] >= 0 && size_t(workspace.shape[0]) >= plan.workspace_bytes,
           "EXL3 W8A8 byte workspace too small or wrong layout");
  VT_CHECK(panel.dtype == DType::kI8 && panel.rank == 2 &&
               panel.shape[0] == k && panel.shape[1] == plan.weight_panel_columns,
           "EXL3 W8A8 weight panel does not match bounded planned capacity");
  for (const Tensor* t : {&workspace, &panel}) {
    VT_CHECK(t->IsContiguous(), "EXL3 W8A8 requires contiguous tensors");
    VT_CHECK(t->device == q.device, "EXL3 W8A8 device mismatch");
  }
  reinterpret_cast<Exl3GroupedLinearFn>(GetOp(OpId::kExl3GroupedW8A8, q.device.type))(
      q, out, in, tr, suh, svh, shard, workspace, panel,
      {args.bits, args.codebook, args.debug_name, args.w8a8_panel_columns});
}

void Exl3GroupedW8A8(Queue& q, Tensor& out, const Tensor& in, const Tensor& tr,
    const Tensor& suh, const Tensor& svh, const Tensor& shard,
    const Exl3GroupedLinearArgs& args) {
  (void)ValidateW8A8Operands(q, out, in, tr, suh, svh, shard, args);
  VT_CHECK(q.device.type == DeviceType::kXPU, "EXL3 shared W8A8 requires XPU");
  // Rank-zero private scratch signals the managed overload to the registered
  // XPU operator. The explicit public overload never admits rank-zero scratch.
  Tensor workspace, panel;
  reinterpret_cast<Exl3GroupedLinearFn>(GetOp(OpId::kExl3GroupedW8A8, q.device.type))(
      q, out, in, tr, suh, svh, shard, workspace, panel,
      {args.bits, args.codebook, args.debug_name, args.w8a8_panel_columns});
}

Exl3W8A8ModelMap::Exl3W8A8ModelMap(Queue& q, const Tensor& map, int groups,
    const std::shared_ptr<void>& resident_owner)
    : map_(map), groups_(groups), owner_(resident_owner) {
  VT_CHECK(q.device.type == DeviceType::kXPU && map.device == q.device &&
               map.rank == 1 && map.dtype == DType::kI32 && map.IsContiguous() &&
               map.shape[0] > 0 && map.shape[0] <= std::numeric_limits<int>::max() / 128 &&
               groups > 0 && owner_ && owner_.get() == map.data,
           "EXL3 W8A8 model map requires matching resident owner/device/layout");
  std::vector<int32_t> values(size_t(map.shape[0]));
  auto& backend = GetBackend(q.device);
  backend.Copy(q, values.data(), map.data, values.size() * sizeof(int32_t));
  backend.Synchronize(q);
  // Use the same complete-map validator/decomposition as the public route.
  // Publish nothing until every routing value has passed it.
  for (int i = 0; i < 3; ++i)
    panels_[i] = PlanExl3W8A8Panels(values, groups, std::array{128, 1024, 2048}[i]);
}

bool Exl3W8A8ModelMap::Matches(const Tensor& map, int groups,
    const std::shared_ptr<void>& resident_owner) const {
  return resident_owner && owner_.get() == resident_owner.get() &&
      !owner_.owner_before(resident_owner) && !resident_owner.owner_before(owner_) &&
      map.data == map_.data && map.device == map_.device && map.dtype == map_.dtype &&
      map.rank == map_.rank && map.shape[0] == map_.shape[0] &&
      map.stride[0] == map_.stride[0] && groups == groups_;
}

const std::vector<Exl3W8A8Panel>& Exl3W8A8ModelMap::Panels(int columns) const {
  VT_CHECK(columns == 128 || columns == 1024 || columns == 2048,
           "EXL3 W8A8 panel width must be128/1024/2048");
  return panels_[columns == 128 ? 0 : columns == 1024 ? 1 : 2];
}

void detail::Exl3GroupedW8A8Model(Queue& q, Tensor& out, const Tensor& in,
    const Tensor& tr, const Tensor& suh, const Tensor& svh, const Tensor& map,
    const Exl3GroupedLinearArgs& args, const std::shared_ptr<void>& resident_owner,
    SharedPtrCache<const Exl3W8A8ModelMap>& cache) {
  (void)ValidateW8A8Operands(q, out, in, tr, suh, svh, map, args);
  VT_CHECK(q.device.type == DeviceType::kXPU, "EXL3 shared W8A8 requires XPU");
  const char* setting = std::getenv("VT_XPU_W8A8_MODEL_MAP");
  // Model maps are immutable for their allocation generation. The public
  // tensor entry points still validate their actual device contents each call.
  const std::string_view mode = setting ? setting : "1";
  VT_CHECK(mode == "0" || mode == "1", "Invalid VT_XPU_W8A8_MODEL_MAP");
  if (mode == "0") {
    Exl3GroupedW8A8(q, out, in, tr, suh, svh, map, args);
    return;
  }
  auto metadata = cache.Load();
  if (!metadata || !metadata->Matches(map, int(suh.shape[0]), resident_owner)) {
    metadata = std::make_shared<const Exl3W8A8ModelMap>(q, map, int(suh.shape[0]), resident_owner);
    cache.Store(metadata);
  }
  auto owned_args = args;
  // A call retains its exact immutable generation even if another queue
  // publishes a replacement in the model slot while this call is executing.
  owned_args.model_map = metadata.get();
  Tensor workspace, panel;
  reinterpret_cast<Exl3GroupedLinearFn>(GetOp(OpId::kExl3GroupedW8A8, q.device.type))(
      q, out, in, tr, suh, svh, map, workspace, panel, owned_args);
}

Exl3SmallMPlan PlanExl3SmallM(int64_t m, int64_t k, int64_t n, int bits) {
  VT_CHECK(m > 0 && m <= 128, "EXL3 SmallM requires physical M in [1,128]");
  VT_CHECK(k > 0 && n > 0 && k % 128 == 0 && n % 128 == 0 &&
               k <= std::numeric_limits<int>::max() &&
               n <= std::numeric_limits<int>::max(),
           "EXL3 SmallM requires positive I32 K/N multiples of 128");
  VT_CHECK(bits == 4 || bits == 6, "EXL3 SmallM supports 4/6bpw only");
  // c59d944 exl3_ops.sycl defaults: vector M<=2, DPAS MB24/40/48 enabled,
  // max MB64, NT2 at MB>=24, and thread targets 1024/1408/2048.
  const bool vector = m <= 2;
  const int mb = vector ? int(m) : m <= 8 ? 8 : m <= 16 ? 16 :
                 m <= 24 ? 24 : m <= 32 ? 32 : m <= 40 ? 40 : m <= 48 ? 48 : 64;
  const int blocks = vector ? 1 : int((m + mb - 1) / mb);
  const int nt = vector ? (bits == 4 && m == 1 ? 8 : 4) : mb <= 16 ? 4 : 2;
  const int64_t units = (n / 16 / nt) * blocks;
  const int target = vector ? 1024 : mb <= 16 ? 1408 : mb >= 40 ? 2048 : 1024;
  const int tk = int(k / 16);
  const int requested = int(std::max(int64_t{1}, std::min(int64_t(tk),
                                         (target + units - 1) / units)));
  const int rps = (tk + requested - 1) / requested;
  return {vector, mb, blocks * mb, nt, (tk + rps - 1) / rps, rps};
}

namespace {
void ValidateSmallMOperands(Queue& q, const Tensor& out, const Tensor& in, const Tensor& trellis,
    const Tensor& suh, const Tensor& svh, const Tensor& shard,
    const Tensor& in_had, const Tensor& partials, const Exl3GroupedLinearArgs& args) {
  VT_CHECK(in.rank == 2 && out.rank == 2, "EXL3 grouped linear requires rank-2 input/output");
  const int64_t m = in.shape[0], k = in.shape[1], n = out.shape[1];
  const auto plan = PlanExl3SmallM(m, k, n, args.bits);
  VT_CHECK(args.codebook == 2, "EXL3 grouped linear requires mul1 codebook");
  VT_CHECK(out.shape[0] == m, "EXL3 grouped linear output row mismatch");
  VT_CHECK(in.dtype == DType::kF16 && out.dtype == DType::kF16,
           "EXL3 grouped linear requires F16 model input/output");
  VT_CHECK(trellis.dtype == DType::kI8 && trellis.rank == 3 &&
               trellis.shape[0] == k / 16 && trellis.shape[1] == n / 16 &&
               trellis.shape[2] == 32 * args.bits,
           "EXL3 grouped linear requires opaque packed trellis [K/16,N/16,32*bits]");
  VT_CHECK(suh.dtype == DType::kF16 && suh.rank == 2 && suh.shape[0] > 0 &&
               suh.shape[0] <= std::numeric_limits<int>::max() && suh.shape[1] == k,
           "EXL3 grouped linear requires F16 suh[S,K]");
  VT_CHECK(suh.shape[0] * m * (k / 128) <= std::numeric_limits<int>::max(),
           "EXL3 grouped linear input launch exceeds I32 indexing");
  VT_CHECK(svh.dtype == DType::kF16 && svh.rank == 1 && svh.shape[0] == n,
           "EXL3 grouped linear requires F16 svh[N]");
  VT_CHECK(shard.dtype == DType::kI32 && shard.rank == 1 && shard.shape[0] == n / 128,
           "EXL3 grouped linear requires I32 shard_of_nb[N/128]");
  VT_CHECK(in_had.dtype == DType::kF16 && in_had.rank == 4 &&
               in_had.shape[0] == suh.shape[0] && in_had.shape[1] == k / 16 &&
               in_had.shape[2] == plan.padded_rows && in_had.shape[3] == 16,
           "EXL3 grouped linear input scratch layout mismatch");
  VT_CHECK(partials.dtype == DType::kF32 && partials.rank == 3 &&
               partials.shape[0] == plan.splits && partials.shape[1] == m &&
               partials.shape[2] == n, "EXL3 grouped linear partial scratch layout mismatch");
  for (const Tensor* t : std::initializer_list<const Tensor*>{
           &out, &in, &trellis, &suh, &svh, &shard, &in_had, &partials}) {
    VT_CHECK(t->IsContiguous(), "EXL3 grouped linear requires contiguous tensors");
    VT_CHECK(t->device == q.device, "EXL3 grouped linear device mismatch");
  }
}
}  // namespace

void Exl3GroupedLinear(Queue& q, Tensor& out, const Tensor& in, const Tensor& trellis,
    const Tensor& suh, const Tensor& svh, const Tensor& shard,
    Tensor& in_had, Tensor& partials, const Exl3GroupedLinearArgs& args) {
  ValidateSmallMOperands(q, out, in, trellis, suh, svh, shard, in_had, partials, args);
  reinterpret_cast<Exl3GroupedLinearFn>(GetOp(OpId::kExl3GroupedLinear, q.device.type))(
      q, out, in, trellis, suh, svh, shard, in_had, partials,
      {args.bits, args.codebook, args.debug_name, args.w8a8_panel_columns});
}

void detail::Exl3GroupedLinearModel(Queue& q, Tensor& out, const Tensor& in,
    const Tensor& trellis, const Tensor& suh, const Tensor& svh, const Tensor& shard,
    Tensor& in_had, Tensor& partials, const Exl3GroupedLinearArgs& args,
    const std::shared_ptr<void>& resident_owner,
    SharedPtrCache<const Exl3W8A8ModelMap>& cache) {
  ValidateSmallMOperands(q, out, in, trellis, suh, svh, shard, in_had, partials, args);
  VT_CHECK(q.device.type == DeviceType::kXPU, "EXL3 model SmallM requires XPU");
  const char* setting = std::getenv("VT_XPU_SMALLM_MODEL_MAP");
  const std::string_view mode = setting ? setting : "1";
  VT_CHECK(mode == "0" || mode == "1", "Invalid VT_XPU_SMALLM_MODEL_MAP");
  auto owned_args = args;
  owned_args.model_map = nullptr;
  std::shared_ptr<const Exl3W8A8ModelMap> metadata;
  if (mode == "1") {
    metadata = cache.Load();
    if (!metadata || !metadata->Matches(shard, int(suh.shape[0]), resident_owner)) {
      bool capturing = false;
#ifdef VLLM_CPP_XPU
      capturing = xpu::IsGraphCapturing(q);
#endif
      if (capturing) {
        metadata.reset();  // No readback/certificate construction inside capture.
#ifdef VLLM_CPP_XPU
        xpu::RecordGraphImmutableRead(q, shard.data, shard.Bytes(), resident_owner,
                                     "EXL3 SmallM cold model source map");
#endif
      } else {
        metadata = std::make_shared<const Exl3W8A8ModelMap>(q, shard, int(suh.shape[0]), resident_owner);
        cache.Store(metadata);
      }
    }
    owned_args.model_map = metadata.get();
  }
  reinterpret_cast<Exl3GroupedLinearFn>(GetOp(OpId::kExl3GroupedLinear, q.device.type))(
      q, out, in, trellis, suh, svh, shard, in_had, partials, owned_args);
}
}  // namespace vt
