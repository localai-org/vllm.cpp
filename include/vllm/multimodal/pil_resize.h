// Pillow's `Image.Resampling.BICUBIC` resize, ported for the uint8 RGB images a
// multimodal chat request carries.
//
// Ported from Pillow `src/libImaging/Resample.c` at tag `12.1.1`:
//   bicubic_filter (a = -0.5, support = 2.0)          :46-61
//   PRECISION_BITS / _clip8_lookups / clip8           :88-96, :735-741
//   precompute_coeffs                                 :98-165 (numbering as read)
//   normalize_coeffs_8bpc                             :167-181
//   ImagingResampleHorizontal_8bpc (bands == 3)       :183-...
//   ImagingResampleVertical_8bpc  (bands == 3)
//   ImagingResampleInner                              (two-pass driver)
//
// WHY A FILE OF ITS OWN, AND WHY THE NAME SAYS "PIL". This is NOT "a bicubic
// kernel", and it is NOT torchvision's bicubic. Two things separate it from
// both, and they only matter on a DOWNSCALE, which is what a chat request
// almost always asks for:
//
//   * the filter SUPPORT is scaled by `max(1, in/out)`, so a 5x downscale reads
//     a 21-tap window and behaves as a weighted area average rather than as a
//     four-tap interpolation, and
//   * the weights are normalized PER OUTPUT PIXEL, and the whole thing runs in
//     22-bit fixed point over a uint8 INTERMEDIATE image between the two passes.
//
// A naive four-tap cubic disagrees with this badly on any real downscale, and
// nothing about the disagreement is visible to a shape check or to a token
// count. The pinned torchvision v2 CPU uint8 processor instead uses adaptive
// signed-int16 coefficient precision. `torchvision_resize.h` implements that
// distinct boundary; this fixed-22-bit Pillow routine is not substituted there.
#ifndef VLLM_MULTIMODAL_PIL_RESIZE_H_
#define VLLM_MULTIMODAL_PIL_RESIZE_H_

#include <cstdint>
#include <vector>

namespace vllm::multimodal {

// Resample an HWC uint8 RGB image to `out_h` x `out_w` exactly as
// `PIL.Image.Image.resize((out_w, out_h), Image.Resampling.BICUBIC)` does for a
// mode-"RGB" image with the default (whole-image) box.
//
// `src` is `in_h * in_w * 3` bytes, row-major, three interleaved channels. The
// result is `out_h * out_w * 3` bytes in the same layout. When both axes are
// unchanged the input is returned verbatim, which is what PIL's own
// `size == self.size` short circuit does.
//
// Throws `std::runtime_error` on a non-positive extent, which is the one input
// PIL cannot be asked for (`Image.resize` rejects it upstream of the resampler).
std::vector<uint8_t> PilResizeBicubicRgb(const uint8_t* src, int64_t in_h,
                                         int64_t in_w, int64_t out_h,
                                         int64_t out_w);

}  // namespace vllm::multimodal

#endif  // VLLM_MULTIMODAL_PIL_RESIZE_H_
