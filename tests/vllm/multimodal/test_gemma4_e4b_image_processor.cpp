#include <doctest/doctest.h>
#include <algorithm>
#include <array>
#include "vllm/model_executor/models/gemma4_vision.h"
#include <limits>
#include <cmath>
#include <fstream>
#include <iterator>
#include <vector>
#include "vllm/multimodal/gemma4_image_processor.h"

using namespace vllm::multimodal;

TEST_CASE("Gemma4 decoded RGB exact raster patches positions mask and count") {
  // Fixed points of pinned source geometry, not arbitrary no-resize overrides.
  // Default square -> 256 valid soft tokens, not the budget 280.
  for (const auto shape : {std::array<int64_t, 3>{280,768,768},
                           {70,384,384}, {140,528,576},
                           {560,1056,1200}, {1120,1536,1632},
                           {280,672,912}, {280,912,672},
                           {70,48,3360}, {70,3360,48}}) {
    const auto budget=shape[0], h=shape[1], w=shape[2];
    std::vector<uint8_t> rgb(static_cast<size_t>(h*w*3));
    for (int64_t y=0;y<h;++y) for (int64_t x=0;x<w;++x)
      for (int c=0;c<3;++c) rgb[(y*w+x)*3+c]=static_cast<uint8_t>((y*7+x*13+c*83)%256);
    const auto a=Gemma4ImageProcessor(budget).ProcessImage(rgb,h,w);
    REQUIRE(a.patches.num_patches==budget*9);
    REQUIRE(a.patches.patch_feature_dim==768);
    REQUIRE(a.patches.pixel_values_f32.size()==static_cast<size_t>(budget*9*768));
    REQUIRE(a.position_ids.size()==static_cast<size_t>(budget*18));
    CHECK(a.patches.pixel_values_bf16.empty());
    CHECK(a.patches.image_grid_thw==std::array<int64_t,3>{1,h/16,w/16});
    CHECK(a.num_soft_tokens==h*w/2304);
    bool pixels=true, positions=true, padding=true;
    for (int64_t row=0;row<budget*9;++row) {
      if(row<h*w/256) {
        positions &= a.position_ids[row*2]==row%(w/16) && a.position_ids[row*2+1]==row/(w/16);
        for(int i=0;i<768;++i) {
          const auto y=(row/(w/16))*16+i/48, x=(row%(w/16))*16+(i/3)%16;
          const float expected=static_cast<float>((y*7+x*13+(i%3)*83)%256)*(1.0f/255.0f);
          pixels &= a.patches.pixel_values_f32[row*768+i]==expected;
        }
      } else {
        padding &= a.position_ids[row*2]==-1 && a.position_ids[row*2+1]==-1;
        for(int i=0;i<768;++i) padding &= a.patches.pixel_values_f32[row*768+i]==0;
      }
    }
    CHECK(pixels); CHECK(positions); CHECK(padding);
    CHECK(ValidateGemma4VisionInput(a.patches.pixel_values_f32, a.position_ids,
                                   a.patches.num_patches, Gemma4VisionConfig{}) == h*w/256);
  }
}

TEST_CASE("Gemma4 pinned geometry and fail closed decoded input") {
  Gemma4ImageProcessor p;
  CHECK(p.TargetSize(100,100)==std::array<int64_t,2>{768,768});
  CHECK(p.TargetSize(1,65536)==std::array<int64_t,2>{48,13440});
  CHECK(p.TargetSize(65536,1)==std::array<int64_t,2>{13440,48});
  CHECK_THROWS(p.TargetSize(65537,1));
  CHECK_THROWS(p.TargetSize(1,std::numeric_limits<int64_t>::max()));
  std::vector<uint8_t> black(768*768*3, 0);
  const auto default_image = p.ProcessImage(black,768,768);
  CHECK(default_image.patches.num_patches==2520);
  CHECK(default_image.num_soft_tokens==256);
  CHECK(default_image.patches.pixel_values_f32[0]==0);
  CHECK_THROWS(p.ProcessImage({},0,1));
  CHECK_THROWS(p.ProcessImage({},-1,1));
  CHECK_THROWS(p.ProcessImage({},std::numeric_limits<int64_t>::max(),2));
  CHECK_THROWS(p.ProcessImage({},65536,65536));
  CHECK_THROWS(p.ProcessImage({},768,768));
  std::vector<uint8_t> wrong(768*768*3+1);
  CHECK_THROWS(p.ProcessImage(wrong,768,768));
  std::vector<uint8_t> resize(48*48*3);
  CHECK(p.ProcessImage(resize,48,48).num_soft_tokens == 256);
  for(int n : {0,-1,69,71,256,281,1121}) CHECK_THROWS(Gemma4ImageProcessor(n));
}

TEST_CASE("Gemma4 tower boundary rejects malformed processed patches") {
  std::vector<uint8_t> rgb(384*384*3, 255);
  const auto a = Gemma4ImageProcessor(70).ProcessImage(rgb,384,384);
  auto pixels = a.patches.pixel_values_f32;
  auto positions = a.position_ids;
  Gemma4VisionConfig cfg;
  auto validate = [&] { return ValidateGemma4VisionInput(pixels,positions,630,cfg); };
  REQUIRE(validate()==576);
  positions[0]=1; CHECK_THROWS(validate()); positions=a.position_ids;
  positions[576*2]=-2; CHECK_THROWS(validate()); positions=a.position_ids;
  positions[576*2+1]=0; CHECK_THROWS(validate()); positions=a.position_ids;
  pixels[576*768]=1; CHECK_THROWS(validate()); pixels=a.patches.pixel_values_f32;
  pixels[0]=std::numeric_limits<float>::quiet_NaN(); CHECK_THROWS(validate());
  pixels[0]=-0.01f; CHECK_THROWS(validate()); pixels[0]=1.01f; CHECK_THROWS(validate());
  pixels.pop_back(); CHECK_THROWS(validate()); pixels=a.patches.pixel_values_f32;
  positions.pop_back(); CHECK_THROWS(validate()); positions=a.position_ids;
  cfg.position_embedding_size=1; CHECK_THROWS(validate()); cfg=Gemma4VisionConfig{};
  cfg.pooling_kernel_size=5; CHECK_THROWS(validate());
  cfg.pooling_kernel_size=0; CHECK_THROWS(validate()); cfg=Gemma4VisionConfig{};
  cfg.patch_size=std::numeric_limits<int64_t>::max(); CHECK_THROWS(validate());
  cfg=Gemma4VisionConfig{};
  CHECK_THROWS(ValidateGemma4VisionInput(pixels,positions,0,cfg));
  CHECK_THROWS(ValidateGemma4VisionInput(pixels,positions,std::numeric_limits<int64_t>::max(),cfg));
  std::fill(positions.begin(),positions.end(),-1); CHECK_THROWS(validate());
}

TEST_CASE("Gemma4 independent CPU Torch uint8 resize goldens") {
  struct Golden { int64_t budget,h,w,oh,ow; uint64_t hash; };
  const Golden cases[] = {
#include "gemma4_resize_golden.inc"
  };
  for (const auto g : cases) {
    CAPTURE(g.budget); CAPTURE(g.h); CAPTURE(g.w);
    std::vector<uint8_t> rgb(g.h*g.w*3);
    for(int64_t y=0;y<g.h;++y) for(int64_t x=0;x<g.w;++x)
      for(int c=0;c<3;++c) rgb[(y*g.w+x)*3+c]=c==0 ? (x<g.w/2?0:255) : (y*7+x*13+c*83)%256;
    const auto a=Gemma4ImageProcessor(g.budget).ProcessImageRGB(rgb,g.h,g.w);
    std::ifstream file(std::string(E4B_RESIZE_FIXTURES)+"/"+std::to_string(g.budget)+"-"+
                       std::to_string(g.h)+"-"+std::to_string(g.w)+".rgb", std::ios::binary);
    REQUIRE(file.is_open());
    const std::vector<uint8_t> expected((std::istreambuf_iterator<char>(file)), {});
    REQUIRE(expected.size()==static_cast<size_t>(g.oh*g.ow*3));
    uint64_t hash=14695981039346656037ULL;
    bool exact=true;
    for(int64_t y=0;y<g.oh;++y) for(int64_t x=0;x<g.ow;++x)
      for(int c=0;c<3;++c) {
        const float f=a.patches.pixel_values_f32[((y/16)*(g.ow/16)+x/16)*768+(y%16)*48+(x%16)*3+c];
        const auto b=static_cast<uint8_t>(std::lround(f*255));
        exact &= b==expected[(y*g.ow+x)*3+c];
        exact &= f==static_cast<float>(expected[(y*g.ow+x)*3+c])*(1.0f/255.0f);
        hash=(hash^b)*1099511628211ULL;
      }
    CHECK(hash==g.hash); CHECK(exact);
    CHECK(a.num_soft_tokens==g.oh*g.ow/2304);
    CHECK(a.patches.image_grid_thw==std::array<int64_t,3>{1,g.oh/16,g.ow/16});
    CHECK(ValidateGemma4VisionInput(a.patches.pixel_values_f32,a.position_ids,a.patches.num_patches,Gemma4VisionConfig{})==g.oh*g.ow/256);
  }
}
