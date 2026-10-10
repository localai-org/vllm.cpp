// GENERATED FILE - DO NOT EDIT BY HAND.
// Regenerate with: scripts/gen-vulkan-spirv.py
//
// SPIR-V for the Vulkan backend's compute kernels (BACKEND-VULKAN),
// compiled from the GLSL in src/vt/vulkan/shaders/*.comp.
//
// WHY THE SPIR-V IS COMMITTED rather than compiled by the build, as
// llama.cpp's vulkan-shaders-gen does at build time: the build then needs NO
// shader toolchain on any machine, which is hermetic everywhere including CI
// and the aarch64 gate box. libshaderc would also be a compiled third-party
// dependency, which .agents/discipline.md forbids. The cost is an obligation
// to re-run this generator when a .comp changes, which the
// vulkan-spirv-freshness CI job enforces.
//
// WHY THE WORDS LIVE IN THE .cpp AND NOT THE HEADER (VK-A1): at the target
// shader surface the blobs are megabytes of array initializer, and anything in
// the header is re-parsed by every including TU. The header carries the struct
// and extern declarations; vulkan_spirv.cpp carries the data, so adding shaders
// costs one TU's compile time rather than all of them.
//
// Produced by: Glslang Version: 11:16.5.0
// Target environment: vulkan1.1
#ifndef VT_VULKAN_VULKAN_SPIRV_H_
#define VT_VULKAN_VULKAN_SPIRV_H_

#include <cstddef>
#include <cstdint>

namespace vt::vulkan {

// Name -> SPIR-V module. The NAME is the shader's file stem and is also the
// key the pipeline cache uses (src/vt/vulkan/vulkan_context.cpp).
//
// spec_ids lists the SPECIALIZATION CONSTANT IDs the module declares, parsed
// from its OpDecorate SpecId instructions and sorted ascending. The host
// passes specialization values BY ID, and Vulkan SILENTLY IGNORES a map entry
// whose ID the module does not declare - so without this table a host/shader
// drift produces WRONG NUMBERS instead of a clean failure.
//
// binding_count is the number of descriptor bindings the module declares, and
// writable_mask has bit i set iff binding i is NOT decorated NonWritable, i.e.
// iff the shader is permitted to WRITE that buffer. Both are parsed out of the
// compiled module rather than written down beside it.
//
// This is the READ/WRITE SET the dispatch path needs to decide whether two
// dispatches are genuinely independent. Dispatch() is handed one flat array of
// VkBuffers with no roles attached, and inferring a role from the binding index
// would be an unverified assumption whose failure mode is a MISSING BARRIER and
// therefore silently wrong numbers. `readonly` in the GLSL is enforced by
// glslang (writing such a block is a compile error) and recorded as NonWritable
// in the SPIR-V, so this mask is a machine-checked fact about the shader.
//
// A bit is SET unless read-only is positively proven, so anything the
// reflection does not understand degrades to an extra barrier, never a
// missing one.
struct SpirvModule {
  const char* name;
  const uint32_t* words;
  size_t word_count;
  const uint32_t* spec_ids;
  size_t spec_id_count;
  uint32_t binding_count;
  uint32_t writable_mask;
};

// DEFINED IN vulkan_spirv.cpp. The array is `extern` and therefore of unknown
// bound here, so kSpirvModuleCount is the ONLY way to size it - use it rather
// than sizeof/sizeof.
extern const SpirvModule kSpirvModules[];
extern const size_t kSpirvModuleCount;

}  // namespace vt::vulkan

#endif  // VT_VULKAN_VULKAN_SPIRV_H_
