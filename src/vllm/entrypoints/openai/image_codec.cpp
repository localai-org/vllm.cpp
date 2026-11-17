// Native container decoding is independent of learned inference. The optional
// system codecs are selected at build time; no URL fetcher or Python is used.
#include "vllm/entrypoints/openai/chat_mm.h"

#include <cmath>
#include <csetjmp>
#include <cstdio>
#include <cstring>
#include <memory>

#include "vllm/v1/engine/input_processor.h"

#if defined(VLLM_CPP_IMAGE_CODECS)
#include <png.h>
#include <jpeglib.h>
#endif

namespace vllm::entrypoints::openai {
namespace {
using vllm::v1::InputValidationError;

void CheckExtents(uint64_t height, uint64_t width) {
  if (height == 0 || width == 0 || height > kMaxDecodedImageExtent ||
      width > kMaxDecodedImageExtent || width > kMaxDecodedImagePixels / height) {
    throw InputValidationError("multimodal image: decoded extent/pixel limit exceeded");
  }
}

#if defined(VLLM_CPP_IMAGE_CODECS)
uint32_t ReadBe32(const uint8_t* p) {
  return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
         (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

// Only the bounded TIFF IFD0 orientation entry is needed. Invalid EXIF is
// ignored, matching normalize_image's suppressed metadata exceptions.
int ExifOrientation(const uint8_t* bytes, std::size_t size) {
  if (size < 8) return 1;
  const bool little = bytes[0] == 'I' && bytes[1] == 'I';
  if (!little && !(bytes[0] == 'M' && bytes[1] == 'M')) return 1;
  const auto u16 = [little](const uint8_t* p) -> uint16_t {
    return little ? uint16_t(p[0]) | (uint16_t(p[1]) << 8)
                  : (uint16_t(p[0]) << 8) | uint16_t(p[1]);
  };
  const auto u32 = [little](const uint8_t* p) -> uint32_t {
    return little ? uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
                        (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24)
                  : ReadBe32(p);
  };
  if (u16(bytes + 2) != 42) return 1;
  const std::size_t offset = u32(bytes + 4);
  if (offset < 8 || offset > size - 2) return 1;
  const std::size_t count = u16(bytes + offset);
  if (count > (size - offset - 2) / 12) return 1;
  for (std::size_t n = 0; n < count; ++n) {
    const auto* entry = bytes + offset + 2 + n * 12;
    if (u16(entry) == 274 && u16(entry + 2) == 3 && u32(entry + 4) == 1) {
      const int orientation = u16(entry + 8);
      return orientation >= 1 && orientation <= 8 ? orientation : 1;
    }
  }
  return 1;
}

DecodedImageRgb OrientRgb(DecodedImageRgb image, int orientation) {
  if (orientation == 1) return image;
  DecodedImageRgb out;
  out.width = orientation >= 5 ? image.height : image.width;
  out.height = orientation >= 5 ? image.width : image.height;
  out.rgb.resize(image.rgb.size());
  for (int64_t y = 0; y < out.height; ++y) {
    for (int64_t x = 0; x < out.width; ++x) {
      int64_t sx = x, sy = y;
      switch (orientation) {
        case 2: sx = image.width - 1 - x; break;
        case 3: sx = image.width - 1 - x; sy = image.height - 1 - y; break;
        case 4: sy = image.height - 1 - y; break;
        case 5: sx = y; sy = x; break;
        case 6: sx = y; sy = image.height - 1 - x; break;
        case 7: sx = image.width - 1 - y; sy = image.height - 1 - x; break;
        case 8: sx = image.width - 1 - y; sy = x; break;
        default: break;
      }
      std::memcpy(out.rgb.data() + static_cast<std::size_t>(y * out.width + x) * 3,
                  image.rgb.data() + static_cast<std::size_t>(sy * image.width + sx) * 3, 3);
    }
  }
  return out;
}

int ValidatePngContainer(const std::vector<uint8_t>& bytes) {
  if (bytes.size() < 8 || png_sig_cmp(bytes.data(), 0, 8) != 0) {
    throw InputValidationError("multimodal image: PNG MIME/container mismatch");
  }
  std::size_t at = 8;
  int orientation = 1;
  bool ended = false;
  while (at < bytes.size()) {
    if (bytes.size() - at < 12) {
      throw InputValidationError("multimodal image: truncated PNG chunk");
    }
    const uint32_t length = ReadBe32(bytes.data() + at);
    if (length > bytes.size() - at - 12) {
      throw InputValidationError("multimodal image: truncated PNG chunk");
    }
    if (std::memcmp(bytes.data() + at + 4, "acTL", 4) == 0) {
      throw InputValidationError("multimodal image: animated PNG is unsupported");
    }
    if (std::memcmp(bytes.data() + at + 4, "eXIf", 4) == 0) {
      orientation = ExifOrientation(bytes.data() + at + 8, length);
    }
    if (std::memcmp(bytes.data() + at + 4, "IEND", 4) == 0) {
      if (length != 0 || at + 12 != bytes.size()) {
        throw InputValidationError("multimodal image: invalid PNG end/trailing data");
      }
      ended = true;
    }
    at += static_cast<std::size_t>(length) + 12;
  }
  if (!ended) throw InputValidationError("multimodal image: missing PNG end");
  return orientation;
}

// All state modified after setjmp lives on the heap. No C++ object with an
// active destructor is constructed in a scope skipped by a libpng longjmp.
struct PngReadState {
  png_structp png = nullptr;
  png_infop info = nullptr;
  const std::vector<uint8_t>* input = nullptr;
  std::size_t offset = 0;
  int orientation = 1;
  char error[256] = {};
  DecodedImageRgb output;
  std::vector<uint8_t> pixels;
  std::vector<png_bytep> rows;
  ~PngReadState() {
    if (png != nullptr) png_destroy_read_struct(&png, &info, nullptr);
  }
};

void PngError(png_structp png, const char* message) {
  auto* state = static_cast<PngReadState*>(png_get_error_ptr(png));
  std::snprintf(state->error, sizeof(state->error), "%s", message);
  png_longjmp(png, 1);
}
void PngWarning(png_structp png, const char* message) {
  PngError(png, message);  // Malformed containers are not repaired silently.
}
void PngRead(png_structp png, png_bytep out, png_size_t count) {
  auto* state = static_cast<PngReadState*>(png_get_io_ptr(png));
  if (count > state->input->size() - state->offset) {
    png_error(png, "truncated PNG input");
  }
  std::memcpy(out, state->input->data() + state->offset, count);
  state->offset += count;
}

DecodedImageRgb DecodePng(const std::vector<uint8_t>& bytes) {
  const auto state = std::make_unique<PngReadState>();
  state->orientation = ValidatePngContainer(bytes);
  state->input = &bytes;
  state->png = png_create_read_struct(PNG_LIBPNG_VER_STRING, state.get(), PngError, PngWarning);
  if (state->png == nullptr) throw std::bad_alloc();
  state->info = png_create_info_struct(state->png);
  if (state->info == nullptr) throw std::bad_alloc();
  if (setjmp(png_jmpbuf(state->png)) != 0) {
    throw InputValidationError(std::string("multimodal image: invalid PNG: ") + state->error);
  }
  png_set_read_fn(state->png, state.get(), PngRead);
  png_set_user_limits(state->png, kMaxDecodedImageExtent, kMaxDecodedImageExtent);
  png_set_chunk_malloc_max(state->png, kMaxImageContainerBytes);
  png_set_chunk_cache_max(state->png, 64);
  png_set_crc_action(state->png, PNG_CRC_ERROR_QUIT, PNG_CRC_ERROR_QUIT);
  png_read_info(state->png, state->info);
  const auto width = png_get_image_width(state->png, state->info);
  const auto height = png_get_image_height(state->png, state->info);
  CheckExtents(height, width);
  const auto color = png_get_color_type(state->png, state->info);
  const auto depth = png_get_bit_depth(state->png, state->info);
  if (depth == 16) {
    throw InputValidationError("multimodal image: 16-bit PNG is not yet qualified");
  }
  if (color == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(state->png);
  if (color == PNG_COLOR_TYPE_GRAY && depth < 8) png_set_expand_gray_1_2_4_to_8(state->png);
  if (color == PNG_COLOR_TYPE_GRAY || color == PNG_COLOR_TYPE_GRAY_ALPHA) {
    png_set_gray_to_rgb(state->png);
  }
  // ImageMediaIO returns an already-RGB image unchanged, including RGB tRNS.
  // Palette/grayscale transparency goes through RGBA and white compositing.
  const bool expand_transparency = color != PNG_COLOR_TYPE_RGB &&
      png_get_valid(state->png, state->info, PNG_INFO_tRNS);
  const bool alpha = (color & PNG_COLOR_MASK_ALPHA) != 0 || expand_transparency;
  if (expand_transparency) {
    png_set_tRNS_to_alpha(state->png);
  }
  // ImageMediaIO applies default white alpha compositing after EXIF
  // normalization. RGB conversion commutes with orientation. No gamma
  // transform or color-profile conversion is introduced.
  png_set_interlace_handling(state->png);
  png_read_update_info(state->png, state->info);
  const std::size_t channels = alpha ? 4 : 3;
  if (png_get_channels(state->png, state->info) != channels ||
      png_get_rowbytes(state->png, state->info) != static_cast<std::size_t>(width) * channels) {
    throw InputValidationError("multimodal image: unsupported decoded PNG layout");
  }
  state->output.width = width;
  state->output.height = height;
  state->output.rgb.resize(static_cast<std::size_t>(width) * height * 3);
  if (alpha) state->pixels.resize(static_cast<std::size_t>(width) * height * 4);
  auto* pixels = alpha ? state->pixels.data() : state->output.rgb.data();
  state->rows.resize(height);
  for (std::size_t y = 0; y < height; ++y) {
    state->rows[y] = pixels + y * width * channels;
  }
  png_read_image(state->png, state->rows.data());
  png_read_end(state->png, state->info);
  if (alpha) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(width) * height; ++i) {
      const unsigned a = state->pixels[i * 4 + 3];
      for (std::size_t c = 0; c < 3; ++c) {
        state->output.rgb[i * 3 + c] =
            static_cast<uint8_t>((state->pixels[i * 4 + c] * a + 255 * (255 - a) + 127) / 255);
      }
    }
  }
  return OrientRgb(std::move(state->output), state->orientation);
}

struct JpegErrorState {
  jpeg_error_mgr error;
  std::jmp_buf jump;
  char message[JMSG_LENGTH_MAX] = {};
};
struct JpegReadState {
  jpeg_decompress_struct decoder = {};
  JpegErrorState error = {};
  int orientation = 1;
  DecodedImageRgb output;
  ~JpegReadState() {
    if (decoder.mem != nullptr) jpeg_destroy_decompress(&decoder);
  }
};
void JpegError(j_common_ptr decoder) {
  auto* state = reinterpret_cast<JpegErrorState*>(decoder->err);
  decoder->err->format_message(decoder, state->message);
  std::longjmp(state->jump, 1);
}
void JpegMessage(j_common_ptr decoder, int level) {
  if (level < 0) JpegError(decoder);  // Includes a synthesized EOI on truncation.
}

DecodedImageRgb DecodeJpeg(const std::vector<uint8_t>& bytes) {
  if (bytes.size() < 2 || bytes[0] != 0xff || bytes[1] != 0xd8) {
    throw InputValidationError("multimodal image: JPEG MIME/container mismatch");
  }
  const auto state = std::make_unique<JpegReadState>();
  state->decoder.err = jpeg_std_error(&state->error.error);
  state->error.error.error_exit = JpegError;
  state->error.error.emit_message = JpegMessage;
  if (setjmp(state->error.jump) != 0) {
    throw InputValidationError(std::string("multimodal image: invalid JPEG: ") + state->error.message);
  }
  jpeg_create_decompress(&state->decoder);
  jpeg_mem_src(&state->decoder, bytes.data(), bytes.size());
  jpeg_save_markers(&state->decoder, JPEG_APP0 + 1, 0xffff);
  jpeg_read_header(&state->decoder, TRUE);
  CheckExtents(state->decoder.image_height, state->decoder.image_width);
  if (state->decoder.data_precision != 8 ||
      (state->decoder.jpeg_color_space != JCS_RGB &&
       state->decoder.jpeg_color_space != JCS_YCbCr &&
       state->decoder.jpeg_color_space != JCS_GRAYSCALE)) {
    throw InputValidationError("multimodal image: JPEG color/precision is not yet qualified");
  }
  for (auto* marker = state->decoder.marker_list; marker != nullptr; marker = marker->next) {
    if (marker->data_length >= 6 && std::memcmp(marker->data, "Exif\0\0", 6) == 0) {
      state->orientation = ExifOrientation(marker->data + 6, marker->data_length - 6);
      break;
    }
  }
  state->decoder.out_color_space = JCS_RGB;
  state->decoder.dct_method = JDCT_ISLOW;
  jpeg_start_decompress(&state->decoder);
  const auto width = state->decoder.output_width;
  const auto height = state->decoder.output_height;
  CheckExtents(height, width);
  if (state->decoder.output_components != 3) {
    throw InputValidationError("multimodal image: unsupported decoded JPEG layout");
  }
  state->output.width = width;
  state->output.height = height;
  state->output.rgb.resize(static_cast<std::size_t>(width) * height * 3);
  while (state->decoder.output_scanline < height) {
    JSAMPROW row = state->output.rgb.data() +
                  static_cast<std::size_t>(state->decoder.output_scanline) * width * 3;
    if (jpeg_read_scanlines(&state->decoder, &row, 1) != 1) {
      throw InputValidationError("multimodal image: incomplete JPEG scanline");
    }
  }
  jpeg_finish_decompress(&state->decoder);
  return OrientRgb(std::move(state->output), state->orientation);
}
#endif
}

ImageCodecFn DefaultImageCodec() {
  return [](const DecodedMedia& media) -> DecodedImageRgb {
    if (media.bytes.size() > kMaxImageContainerBytes) {
      throw InputValidationError("multimodal image: container exceeds byte limit");
    }
    if (media.media_type == "image/x-raw-rgb") {
      const std::size_t n = media.bytes.size();
      const auto side = static_cast<uint64_t>(std::llround(std::sqrt(static_cast<double>(n / 3))));
      CheckExtents(side, side);
      if (side * side * 3 != n) {
        throw InputValidationError("image/x-raw-rgb payload is not a square HxWx3 buffer");
      }
      return {media.bytes, static_cast<int64_t>(side), static_cast<int64_t>(side)};
    }
#if defined(VLLM_CPP_IMAGE_CODECS)
    if (media.media_type == "image/png") return DecodePng(media.bytes);
    if (media.media_type == "image/jpeg") return DecodeJpeg(media.bytes);
    throw InputValidationError("multimodal image: unsupported container; only still PNG/JPEG are supported");
#else
    throw InputValidationError("multimodal image: PNG/JPEG codecs unavailable; configure VLLM_CPP_IMAGE_CODECS=ON");
#endif
  };
}
}
