// The VT_HOST_EMBEDDING gather. See host_embedding.h for the contract.
#include "vllm/model_executor/models/host_embedding.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vllm/model_executor/models/qwen3_5_internal.h"  // detail::TakeDeviceTokenIds
#include "vllm/model_executor/models/dense_attn_block.h"  // ResidentWeight (device arm)

namespace vllm {
namespace dense_attn {
namespace {

vt::Queue& HostEmbedQueue() {
  // Process-lifetime singleton: backend registries may outlive this TU's static
  // destructors, so this queue is deliberately never destroyed.
  static vt::Queue* q =
      new vt::Queue(vt::GetBackend(vt::DeviceType::kCPU).CreateQueue());
  return *q;
}

}  // namespace

bool HostEmbedInto(Dev d, DBuf& hidden, const std::vector<int32_t>& token_ids,
                   const OwnedTensor& table, int64_t vocab, int64_t H) {
  static const bool on = [] {
    const char* e = std::getenv("VT_HOST_EMBEDDING");
    return e != nullptr && e[0] == '1';
  }();
  static bool logged = false;
  if (!on) return false;
  const int64_t T = static_cast<int64_t>(token_ids.size());
  if (T == 0) return true;
  if (table.bytes.empty() || table.host_released) {
    if (!logged) {
      logged = true;
      std::fprintf(stderr,
                   "[host-embed] DISABLED: token table host bytes unavailable "
                   "(dtype=%s)\n",
                   vt::Name(table.dtype));
    }
    return false;
  }
  // ENG-ASYNC-SCHED W4: the async runner may have spliced this step's sampled
  // token into a device-resident ids buffer, making `token_ids` stale for
  // decode rows. Take that override (if any) and read the ids back; otherwise
  // the host vector is authoritative and no device round-trip runs.
  std::vector<int32_t> resolved = token_ids;
  const detail::DeviceTokenIds override_ids = detail::TakeDeviceTokenIds();
  if (override_ids.ids != nullptr) {
    // The device arm below splices through `detail::ApplyDeviceTokenIds`, which
    // bounds the override against the embed input and enqueues the Copy on the
    // queue. The host arm must not keep a second, unchecked copy of that rule:
    // a count larger than `T` means the runner and the model disagree about
    // this step, and the raw Copy that used to sit here wrote past the T-row
    // buffer instead of refusing. The helper also tolerates a SHORTER override,
    // which is the padded case: the first `count` rows are replaced and the
    // host upload's tail is preserved.
    DBuf dids(d, DType::kI32, {T}, token_ids.data());
    detail::ApplyDeviceTokenIds(d.b, d.q, dids.ptr(), T, override_ids,
                                "host embedding");
    dids.Download(d, resolved.data());
  }
  if (std::getenv("VT_HOST_EMBED_TRACE") != nullptr) {
    std::fprintf(stderr, "[host-embed] T=%lld override=%d\n",
                 static_cast<long long>(T), override_ids.ids != nullptr ? 1 : 0);
  }
  const vt::DType out_dtype = hidden.t().dtype;
  if (table.dtype == DType::kBF16 && out_dtype == DType::kBF16) {
    // Byte-copy fast path: the gathered rows are already in the output dtype.
    std::vector<uint16_t> rows(static_cast<size_t>(T) * static_cast<size_t>(H));
    const uint16_t* src = reinterpret_cast<const uint16_t*>(table.bytes.data());
    for (int64_t t = 0; t < T; ++t) {
      const int32_t id = resolved[static_cast<size_t>(t)];
      VT_CHECK(id >= 0 && id < vocab, "host embedding: token id out of range");
      std::memcpy(rows.data() + static_cast<size_t>(t) * static_cast<size_t>(H),
                  src + static_cast<size_t>(id) * static_cast<size_t>(H),
                  static_cast<size_t>(H) * sizeof(uint16_t));
    }
    d.b.Copy(d.q, hidden.ptr(), rows.data(), rows.size() * sizeof(uint16_t));
    if (!logged) {
      logged = true;
      std::fprintf(stderr, "[host-embed] bf16 row copy (T=%lld)\n",
                   static_cast<long long>(T));
    }
    return true;
  }
  const vt::Device cpu{};
  vt::Tensor table_view =
      table.ViewOn(const_cast<uint8_t*>(table.bytes.data()), cpu, {vocab, H});
  std::vector<uint8_t> staging(static_cast<size_t>(T) *
                               static_cast<size_t>(H) * vt::SizeOf(out_dtype));
  vt::Tensor out = MakeTensor(staging.data(), out_dtype, cpu, {T, H});
  vt::Tensor ids = MakeTensor(resolved.data(), DType::kI32, cpu, {T});
  vt::Embedding(HostEmbedQueue(), out, table_view, ids);
  if (!logged) {
    logged = true;
    std::fprintf(stderr, "[host-embed] CPU row gather (T=%lld dtype=%s out=%s)\n",
                 static_cast<long long>(T), vt::Name(table.dtype),
                 vt::Name(out_dtype));
  }
  d.b.Copy(d.q, hidden.ptr(), staging.data(), staging.size());
  return true;
}

void EmbedGather(Dev d, DBuf& out, const std::vector<int32_t>& token_ids,
                 const OwnedTensor& table, int64_t vocab, int64_t H,
                 const char* what) {
  if (HostEmbedInto(d, out, token_ids, table, vocab, H)) return;
  const int64_t T = static_cast<int64_t>(token_ids.size());
  Tensor dtab = ResidentWeight(d, table, {vocab, H});
  // ROW-SERVE-ASYNC-DENSE-MIRROR (ENG-ASYNC-SCHED W4, #1305): on the async
  // serving loop the runner's device combine splices each decode row's sampled
  // token into a device-resident id buffer on the MAIN QUEUE while the host
  // `token_ids` vector stays stale BY DESIGN — materializing it on the host is
  // the synchronize that path removes. Splicing the override over the upload here
  // is ordered AFTER the combine rather than racing it, and it is CLEARED on
  // first use so a second, unrelated embed cannot be handed this step's rows.
  // Null on every other path, where the copy is a no-op and the caller is
  // byte-identical to its pre-#1305 self.
  DBuf dids(d, DType::kI32, {T}, token_ids.data());
  detail::ApplyDeviceTokenIds(d.b, d.q, dids.ptr(), T, what);
  vt::Embedding(d.q, out.t(), dtab, dids.t());
}

}  // namespace dense_attn
}  // namespace vllm
