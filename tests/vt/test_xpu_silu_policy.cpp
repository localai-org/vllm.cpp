#include <doctest/doctest.h>
#include "vt/xpu_silu_policy.h"
#include <string>

TEST_CASE("XPU small SiLU policy: qualified default M1/M4 width17408 only") {
  const char* key = "VT_XPU_SILU_SMALL_DECODE";
  const bool present = std::getenv(key) != nullptr;
  const std::string value = present ? std::getenv(key) : "";
  struct Restore {
    const char* key; bool present; std::string value;
    ~Restore() { if (present) setenv(key, value.c_str(), 1); else unsetenv(key); }
  } restore{key, present, value};
  REQUIRE(unsetenv(key) == 0);
  CHECK(vt::XpuSmallSiluDecode(1, 17408));
  REQUIRE(setenv(key, "1", 1) == 0);
  CHECK(vt::XpuSmallSiluDecode(1, 17408));
  CHECK(vt::XpuSmallSiluDecode(4, 17408));
  for (int rows : {0, 2, 3, 5, 63, 64, 65}) CHECK_FALSE(vt::XpuSmallSiluDecode(rows, 17408));
  for (int width : {0, 1, 5120, 17407, 17409}) CHECK_FALSE(vt::XpuSmallSiluDecode(1, width));
  REQUIRE(setenv(key, "0", 1) == 0);
  CHECK_FALSE(vt::XpuSmallSiluDecode(4, 17408));
  for (const char* invalid : {"", "3", "01", " 1", "1x"}) {
    REQUIRE(setenv(key, invalid, 1) == 0);
    CHECK_THROWS(vt::XpuSmallSiluDecode(1, 17408));
  }
}

TEST_CASE("XPU small SiLU policy: mode2 covers measured concurrency shapes") {
  const char* key="VT_XPU_SILU_SMALL_DECODE";
  const bool present=std::getenv(key)!=nullptr;const std::string saved=present?std::getenv(key):"";
  struct Restore {const char* key;bool present;std::string saved;
    ~Restore(){if(present)setenv(key,saved.c_str(),1);else unsetenv(key);}
  } restore{key,present,saved};
  REQUIRE(setenv(key,"1",1)==0);
  for(int rows:{2,3,8,12,16})CHECK_FALSE(vt::XpuSmallSiluDecode(rows,17408));
  REQUIRE(setenv(key,"2",1)==0);
  for(int rows:{1,2,3,4,8,12,16})CHECK(vt::XpuSmallSiluDecode(rows,17408));
  for(int rows:{0,5,7,15,17,63,64})CHECK_FALSE(vt::XpuSmallSiluDecode(rows,17408));
  for(int width:{0,1,5120,17407,17409})CHECK_FALSE(vt::XpuSmallSiluDecode(16,width));
}
