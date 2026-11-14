// Vulkan backend — runtime loader implementation. See vulkan_loader.h for the
// port map and for why this is dlopen rather than a link-time dependency.
// BACKEND-VULKAN, W0 skeleton.
#include "vulkan_loader.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <mutex>
#include <string>

#include "vt/dtype.h"  // VT_CHECK

namespace vt::vulkan {
namespace {

VulkanApi g_api;
#if defined(_WIN32)
HMODULE g_handle = nullptr;
#else
void* g_handle = nullptr;
#endif
bool g_loaded = false;

#if !defined(_WIN32)
// The SONAMEs a Vulkan loader is installed under. `libvulkan.so.1` is the only
// one present on either of our boxes; the others keep a macOS/MoltenVK or
// bare-dev-package host working without a code change.
constexpr const char* kLibraryNames[] = {
    "libvulkan.so.1",
    "libvulkan.so",
    "libvulkan.1.dylib",
    "libMoltenVK.dylib",
};
#endif

#if defined(_WIN32)
void* Win32Open(void*, const wchar_t* name) {
  return reinterpret_cast<void*>(LoadLibraryW(name));
}

void* Win32Lookup(void*, void* handle, const char* name) {
  return reinterpret_cast<void*>(
      GetProcAddress(reinterpret_cast<HMODULE>(handle), name));
}

void Win32Close(void*, void* handle) {
  FreeLibrary(reinterpret_cast<HMODULE>(handle));
}

constexpr VulkanLibraryOps kWin32Ops{nullptr, Win32Open, Win32Lookup, Win32Close};
#endif

void CloseLibrary() {
#if defined(_WIN32)
  FreeLibrary(g_handle);
#else
  dlclose(g_handle);
#endif
  g_handle = nullptr;
}

#if defined(_WIN32)
struct Win32LibraryShutdown {
  ~Win32LibraryShutdown() {
    if (g_handle != nullptr) CloseLibrary();
  }
};
Win32LibraryShutdown g_win32_library_shutdown;
#endif

#if !defined(_WIN32)
void* LookupSymbol(const char* name) {
  return dlsym(g_handle, name);
}
#endif

bool ProbeWithOps(const VulkanLibraryOps& ops, bool close_success,
                  VulkanApi* api = nullptr, void** retained_handle = nullptr) {
  void* handle = ops.open(ops.context, L"vulkan-1.dll");
  if (handle == nullptr) return false;
  auto close = [&] { ops.close(ops.context, handle); };
  auto get_instance_proc_addr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
      ops.lookup(ops.context, handle, "vkGetInstanceProcAddr"));
  if (get_instance_proc_addr == nullptr ||
      get_instance_proc_addr(VK_NULL_HANDLE, "vkCreateInstance") == nullptr) {
    close();
    return false;
  }
  if (api != nullptr) {
    api->vkGetInstanceProcAddr = get_instance_proc_addr;
    api->vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(
        get_instance_proc_addr(VK_NULL_HANDLE, "vkCreateInstance"));
  }
  if (retained_handle != nullptr) *retained_handle = handle;
  if (close_success) close();
  return true;
}

}  // namespace

bool ProbeVulkanLibraryForTesting(const VulkanLibraryOps& ops) {
  if (ops.open == nullptr || ops.lookup == nullptr || ops.close == nullptr) return false;
  return ProbeWithOps(ops, true);
}

bool LoadVulkanLibrary() {
  static std::once_flag once;
  std::call_once(once, [] {
#if defined(_WIN32)
    void* retained = nullptr;
    if (!ProbeWithOps(kWin32Ops, false, &g_api, &retained)) return;
    g_handle = reinterpret_cast<HMODULE>(retained);
#else
    for (const char* name : kLibraryNames) {
      g_handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
      if (g_handle != nullptr) break;
    }
    if (g_handle == nullptr) return;

    // vkGetInstanceProcAddr is the ONE symbol a conformant loader must export
    // directly; everything else is resolved through it.
    g_api.vkGetInstanceProcAddr =
        reinterpret_cast<PFN_vkGetInstanceProcAddr>(LookupSymbol("vkGetInstanceProcAddr"));
    if (g_api.vkGetInstanceProcAddr == nullptr) {
      CloseLibrary();
      return;
    }
#endif

#if !defined(_WIN32)
#define VT_VK_LOAD_GLOBAL(name)                                          \
  g_api.name = reinterpret_cast<PFN_##name>(                             \
      g_api.vkGetInstanceProcAddr(VK_NULL_HANDLE, #name));
    VT_VK_GLOBAL_FUNCS(VT_VK_LOAD_GLOBAL)
#undef VT_VK_LOAD_GLOBAL
#else
    g_api.vkEnumerateInstanceVersion =
        reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
            g_api.vkGetInstanceProcAddr(VK_NULL_HANDLE,
                                        "vkEnumerateInstanceVersion"));
#endif

    // vkCreateInstance is mandatory even on a Vulkan 1.0 loader.
    // vkEnumerateInstanceVersion is 1.1 and may legitimately be null; the
    // context treats a null as "the implementation is 1.0" and refuses to
    // register, since this backend needs 1.1 for VK_KHR_16bit_storage.
    if (g_api.vkCreateInstance == nullptr) {
      CloseLibrary();
      return;
    }
    g_loaded = true;
  });
  return g_loaded;
}

const VulkanApi& Api() { return g_api; }

void LoadInstanceFunctions(VkInstance instance) {
#define VT_VK_LOAD_INSTANCE(name)                                              \
  g_api.name =                                                                 \
      reinterpret_cast<PFN_##name>(g_api.vkGetInstanceProcAddr(instance, #name)); \
  VT_CHECK(g_api.name != nullptr,                                              \
           std::string("vulkan: instance entry point missing: ") + #name);
  VT_VK_INSTANCE_FUNCS(VT_VK_LOAD_INSTANCE)
#undef VT_VK_LOAD_INSTANCE
  // Optional tier: resolve, but do NOT check. A null here means the device does
  // not have the extension, which is a supported state the caller branches on.
#define VT_VK_LOAD_INSTANCE_OPT(name)                                          \
  g_api.name =                                                                 \
      reinterpret_cast<PFN_##name>(g_api.vkGetInstanceProcAddr(instance, #name));
  VT_VK_INSTANCE_FUNCS_OPTIONAL(VT_VK_LOAD_INSTANCE_OPT)
#undef VT_VK_LOAD_INSTANCE_OPT
}

void LoadDeviceFunctions(VkDevice device) {
#define VT_VK_LOAD_DEVICE(name)                                             \
  g_api.name =                                                              \
      reinterpret_cast<PFN_##name>(g_api.vkGetDeviceProcAddr(device, #name)); \
  VT_CHECK(g_api.name != nullptr,                                           \
           std::string("vulkan: device entry point missing: ") + #name);
  VT_VK_DEVICE_FUNCS(VT_VK_LOAD_DEVICE)
#undef VT_VK_LOAD_DEVICE
}

}  // namespace vt::vulkan
