#include "vllm/entrypoints/chat_template.h"
#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/v1/engine/validation_error.h"
#include <doctest/doctest.h>
#include <fstream>
#include <sstream>

namespace {
using namespace vllm::entrypoints;
using Json = nlohmann::json;
std::string Template() {
  std::ifstream f(std::string(VLLM_TEST_FIXTURES_DIR) + "/gemma4_e4b_google/chat_template.jinja");
  REQUIRE(f.good());
  std::ostringstream s; s << f.rdbuf(); return s.str();
}
std::string Render(const std::string& tmpl, const Json& body, bool generation=true) {
  auto request = body.get<openai::ChatCompletionRequest>();
  auto fn = MakeChatTemplatePromptFn(tmpl, "<bos>", "<eos>");
  return fn(request.messages, generation, request.tools.value_or(std::vector<openai::ChatCompletionToolsParam>{}), request.chat_template_kwargs);
}
Json Request(Json content, std::string role="user") {
  return {{"messages", Json::array({{{"role",role},{"content",content}}})}};
}
}
TEST_CASE("E4B original Google JSON requests match independent Jinja bytes") {
  std::ifstream f(std::string(VLLM_TEST_FIXTURES_DIR)+"/gemma4_e4b_template/oracle.json");
  REQUIRE(f.good());
  Json oracle; f >> oracle;
  REQUIRE(oracle["cases"].size() == 180);
  const auto tmpl=Template();
  for (const auto& row:oracle["cases"]) {
    INFO(row["name"].dump());
    CHECK(Render(tmpl,row["request"],row["generation"]) == row["expected"].get<std::string>());
  }
}
TEST_CASE("E4B refuses dropped typed content only for pinned Google template") {
  const auto tmpl=Template();
  const std::vector<Json> bad={42,Json::object(),Json::array({42}),Json::array({Json::object()}),Json::array({{{"type","unknown"}}}),Json::array({{{"type","audio_url"},{"audio_url","opaque"}}}),Json::array({{{"type","video_url"},{"video_url","opaque"}}}),Json::array({{{"type","text"}}}),Json::array({{{"type","text"},{"text",42}}})};
  for(const auto& content:bad) {
    CHECK_THROWS_AS(Render(tmpl,Request(content)),vllm::v1::InputValidationError);
    CHECK_NOTHROW(Render("{{ messages[0].content }}",Request(content)));
  }
  for(const auto& role:{"system","developer"}) {
    CHECK_THROWS_AS(Render(tmpl,Request(Json::array({{{"type","image"}}}),role)),vllm::v1::InputValidationError);
  }
}
TEST_CASE("E4B policy leaves nonidentical templates and flattened strings unchanged") {
  const auto content=Json::array({{{"type","text"},{"text","first"}},{{"type","image_url"},{"image_url",{{"url","opaque"}}}},{{"type","text"},{"text","last"}}});
  const auto decoded = Request(content).get<openai::ChatCompletionRequest>();
  REQUIRE(decoded.messages[0].template_content.has_value());
  CHECK(*decoded.messages[0].template_content == content);
  REQUIRE(decoded.messages[0].content_parts.has_value());
  CHECK(decoded.messages[0].content_parts->size() == 3);
  CHECK(Render("{{ messages[0].content }}",Request(content)) == "first\nlast");
  // Nonidentical source keeps the pre-existing minja syntax error too.
  CHECK_THROWS_AS(Render(Template()+"\n",Request(content)), ChatTemplateError);
}
