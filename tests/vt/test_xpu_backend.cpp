#include "vt/backend.h"
#include "vt/xpu.h"
#include "vt/xpu/xpu_common.h"
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <sys/mman.h>
#include <unistd.h>

int main(int argc, char**) {
  try {
    const vt::Device device{vt::DeviceType::kXPU, 0};
    if (argc > 1) {
      VT_CHECK(vt::xpu::DeviceCount() == 0, "no-GPU test unexpectedly sees a GPU");
      try { auto q = vt::CreateQueue(device); vt::DestroyQueue(q); }
      catch (const std::exception& e) {
        VT_CHECK(std::string(e.what()).find("no Level Zero GPU") != std::string::npos,
                 "missing-device diagnostic not explicit");
        std::cout << "PASS: explicit XPU request refuses absent GPU\n";
        return 0;
      }
      VT_CHECK(false, "XPU silently used another device");
    }
    VT_CHECK(vt::xpu::DeviceCount() > 0, "XPU hardware test requires a Level Zero GPU");
    std::cout << vt::xpu::DeviceDescription() << '\n';
    auto& b = vt::GetBackend(device);
    VT_CHECK(!b.DeviceMemoryIsHostAddressable() && !b.UnifiedMemory(), "device USM contract");
    VT_CHECK(b.SupportsGraphCapture() && !b.SupportsAuxStream()
                 && !b.SupportsCompressedGdnState() && b.SupportsCompressedConvState(),
             "native XPU capability contract");
    auto q1 = vt::CreateQueue(device), q2 = vt::CreateQueue(device);
    VT_CHECK(q1.handle != q2.handle && q1.id != q2.id, "queues are not independent");
    VT_CHECK(vt::xpu::NativeQueue(q1).is_in_order() && vt::xpu::NativeQueue(q2).is_in_order(),
             "reference queues must be in order");
    const bool profiling = std::getenv("VT_XPU_PROFILE") &&
                           std::string(std::getenv("VT_XPU_PROFILE")) == "1";
    VT_CHECK(vt::xpu::NativeQueue(q1).has_property<sycl::property::queue::enable_profiling>() == profiling,
             "XPU queue profiling mode mismatch");
    VT_CHECK(vt::xpu::NativeQueue(q1).get_context() == vt::xpu::NativeQueue(q2).get_context(),
             "queues on one GPU must share a context");
    const auto before = vt::xpu::GetMemoryInfo();
    constexpr size_t bytes = 1027;
    auto* p = static_cast<unsigned char*>(vt::Alloc(device, bytes));
    auto* p2 = static_cast<unsigned char*>(vt::Alloc(device, bytes));
    auto* host = static_cast<unsigned char*>(b.AllocPinned(bytes));
    VT_CHECK(reinterpret_cast<uintptr_t>(p) % 64 == 0, "device alignment");
    VT_CHECK(sycl::get_pointer_type(p, vt::xpu::NativeQueue(q1).get_context()) == sycl::usm::alloc::device,
             "allocation is not device USM");
    auto during = vt::xpu::GetMemoryInfo();
    VT_CHECK(during.allocated_bytes == before.allocated_bytes + 2 * bytes
                 && during.pinned_bytes == before.pinned_bytes + bytes
                 && during.total_bytes == before.total_bytes
                 && during.budget_bytes <= during.total_bytes, "memory accounting");
    // Page-locked host -> device -> another device buffer -> host, crossing queues.
    for (size_t i = 0; i < bytes; ++i) host[i] = static_cast<unsigned char>(i);
    b.Copy(q1, p, host, bytes);
    auto event = b.CreateEvent();
    b.RecordEvent(event, q1);
    b.QueueWaitEvent(q2, event);
    b.Copy(q2, p2, p, bytes);
    b.Memset(q2, p2 + 1, 0x42, bytes - 2);
    b.Copy(q2, host, p2, bytes);
    b.RecordEvent(event, q2);
    vt::Free(device, p);  // Must wait for the final use on q2, not merely q1.
    VT_CHECK(b.QueryEvent(event), "Free did not drain dependent work");
    VT_CHECK(host[0] == 0 && host[bytes - 1] == static_cast<unsigned char>(bytes - 1), "copy tails");
    for (size_t i = 1; i + 1 < bytes; ++i) VT_CHECK(host[i] == 0x42, "memset/copy mismatch");
    b.SynchronizeEvent(event);
    b.DestroyEvent(event);
    vt::Free(device, p2);
    b.FreePinned(host);
    // A small, non-square matrix kernel with K/N tails (PR01 queue/matrix probe).
    constexpr int m = 3, n = 9, k = 13;
    std::array<float, m * k> a;
    std::array<float, k * n> w;
    std::array<float, m * n> result{};
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<float>(i % 7) - 3;
    for (size_t i = 0; i < w.size(); ++i) w[i] = static_cast<float>(i % 5) - 2;
    auto* da = static_cast<float*>(vt::Alloc(device, sizeof(a)));
    auto* dw = static_cast<float*>(vt::Alloc(device, sizeof(w)));
    auto* dc = static_cast<float*>(vt::Alloc(device, sizeof(result)));
    b.Copy(q1, da, a.data(), sizeof(a)); b.Copy(q1, dw, w.data(), sizeof(w));
    auto matrix_event = vt::xpu::NativeQueue(q1).parallel_for(sycl::range<1>(m * n), [=](sycl::id<1> id) {
      const int row = id[0] / n, col = id[0] % n;
      float sum = 0;
      for (int inner = 0; inner < k; ++inner) sum += da[row * k + inner] * dw[inner * n + col];
      dc[id[0]] = sum;
    });
    vt::xpu::RecordProfileEvent(q1, "probe_matrix", matrix_event);
    VT_CHECK(vt::xpu::PendingProfileEventCount() == (profiling ? 1u : 0u),
             "XPU pending profile event count");
    b.Copy(q1, result.data(), dc, sizeof(result));
    vt::Free(device, dc); vt::Free(device, da); vt::Free(device, dw);
    const auto profile_records = vt::xpu::DrainProfileEvents();
    VT_CHECK(profile_records.size() == (profiling ? 1u : 0u), "XPU eager profile event count");
    VT_CHECK(vt::xpu::PendingProfileEventCount() == 0, "XPU profile drain did not clear pending events");
    if (profiling) {
      const auto anchor = vt::xpu::CaptureProfileClockAnchor();
      VT_CHECK(anchor.host_before_ns > 0 && anchor.host_after_ns >= anchor.host_before_ns &&
                   anchor.device_start_ns > 0 && anchor.device_end_ns >= anchor.device_start_ns,
               "XPU profile clock anchor unavailable or out of order");
      const auto& record = profile_records.front();
      VT_CHECK(record.stage == "probe_matrix" && record.queue_id == q1.id &&
                   record.submit_ns > 0 && record.start_ns > 0 &&
                   record.end_ns >= record.start_ns,
               "XPU kernel profiling timestamps unavailable or out of order");
    }
    for (int row = 0; row < m; ++row) for (int col = 0; col < n; ++col) {
      float sum = 0;
      for (int inner = 0; inner < k; ++inner) sum += a[row * k + inner] * w[inner * n + col];
      VT_CHECK(result[row * n + col] == sum, "matrix probe mismatch");
    }
    // Regression: read-only file mappings used to fault the Level Zero copy
    // engine before the first real EXL3 kernel. Cross the bounded staging size.
    constexpr size_t mapped_bytes = 6 * 1024 * 1024 + 13;
    std::vector<unsigned char> source(mapped_bytes + 1), downloaded(mapped_bytes);
    for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<unsigned char>(i * 13 + i / 251);
    FILE* file = std::tmpfile();
    VT_CHECK(file != nullptr, "mapping fixture file failed");
    VT_CHECK(std::fwrite(source.data(), 1, source.size(), file) == source.size(), "mapping fixture write failed");
    VT_CHECK(std::fflush(file) == 0, "mapping fixture flush failed");
    void* mapping = mmap(nullptr, source.size(), PROT_READ, MAP_PRIVATE, fileno(file), 0);
    VT_CHECK(mapping != MAP_FAILED, "read-only mapping failed");
    void* mapped_device = b.Alloc(mapped_bytes);
    b.Copy(q1, mapped_device, static_cast<const char*>(mapping) + 1, mapped_bytes);
    b.Copy(q1, downloaded.data(), mapped_device, mapped_bytes);
    b.Synchronize(q1);
    VT_CHECK(std::memcmp(downloaded.data(), source.data() + 1, mapped_bytes) == 0,
             "read-only unaligned mmap transfer differs");
    b.Free(mapped_device);
    munmap(mapping, source.size()); std::fclose(file);
    const auto after = vt::xpu::GetMemoryInfo();
    VT_CHECK(after.allocated_bytes == before.allocated_bytes && after.pinned_bytes == before.pinned_bytes,
             "memory not released");
    try { (void)b.Alloc(after.budget_bytes + 1); VT_CHECK(false, "budget was ignored"); }
    catch (const std::exception& e) {
      VT_CHECK(std::string(e.what()).find("exceeds memory budget") != std::string::npos, "wrong budget error");
    }
    vt::DestroyQueue(q1); vt::DestroyQueue(q2);
    std::cout << "PASS: USM alloc/copy/memset/free, pinned memory, two queues/events, budget, F32 [3,13]x[13,9], read-only mmap >4MiB\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
