#include "doctest/doctest.h"
#include "vllm/parser/engine/tokenizer_bridge.h"
#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"
using vllm::tok::Tokenizer;
using namespace vllm::parser::engine;
namespace oai = vllm::entrypoints::openai;
namespace {
Tokenizer fixture() {
  return Tokenizer::FromHfJsonBytes(R"({"pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true},"model":{"type":"BPE","byte_fallback":true,"vocab":{"▁Sure":0,"<0x41>":2},"merges":[]},"added_tokens":[{"id":10,"content":"<|channel>","special":true},{"id":11,"content":"<channel|>","special":true},{"id":12,"content":"<|turn>","special":true},{"id":13,"content":"<|tool_call>","special":true},{"id":14,"content":"<|tool_response>","special":true},{"id":15,"content":"<drop>","special":true},{"id":16,"content":"<ordinary>","special":true},{"id":16,"content":"<ordinary>","special":false}]})", "synthetic bridge fixture");
}
}
TEST_CASE("production bridge stored vocab decode and final specials") {
  const auto tok=fixture(); TokenizerBridge b(tok);
  CHECK(b.decode(0)=="Sure"); CHECK(b.decode(2)=="A"); CHECK(b.decode(16)=="<ordinary>");
  CHECK_THROWS(b.decode(1)); CHECK_THROWS(b.decode(-1)); CHECK_THROWS(b.decode(999));
  CHECK_FALSE(tok.HasToken(3));
  const auto* map=&b.get_vocab();
  CHECK(map==&b.get_vocab()); CHECK(map->size()==9);
  CHECK(map->count("▁Sure")==1); CHECK(map->count("<|channel>")==1);
  if(map->count("▁Sure")) CHECK(map->at("▁Sure")==0);
  CHECK(map->count("")==0);
  std::vector<std::string> texts{"stale"}; std::vector<int> ids{-1};
  CHECK(b.all_special(texts,ids)); CHECK(ids==std::vector<int>{10,11,12,13,14,15});
  CHECK(texts==std::vector<std::string>{"<|channel>","<channel|>","<|turn>","<|tool_call>","<|tool_response>","<drop>"});
  TokenIDScanner scanner({{15,kDropTerminal}},&b);
  auto parts=scanner.scan("<drop>Sure",{15,0});
  REQUIRE(parts.size()==2); CHECK(parts[0].terminal==kDropTerminal); CHECK(parts[1].text=="Sure");
}
TEST_CASE("production bridge refuses ambiguous inverse map") {
  const auto tok=Tokenizer::FromHfJsonBytes(R"({"pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true},"model":{"type":"BPE","vocab":{"same":0},"merges":[]},"added_tokens":[{"id":1,"content":"same","special":false}]})", "synthetic bridge fixture");
  CHECK(tok.HasToken(0)); CHECK(tok.HasToken(1));
  CHECK_THROWS_AS(TokenizerBridge{tok},std::invalid_argument);
}
TEST_CASE("production bridge Gemma ABC prompt boundaries and reset") {
 const auto tok=fixture(); TokenizerBridge b(tok);
 for(bool thinking:{false,true}) {
  auto p=oai::get_reasoning_parser("gemma4",thinking,&b); REQUIRE(p);
  for(int boundary:{11,12,13,14}) {
   for(bool opened:{false,true}) {
    p=oai::get_reasoning_parser("gemma4",thinking,&b);
    p->adjust_initial_state_from_prompt(opened?std::vector<int>{boundary,10}:std::vector<int>{10,boundary});
    auto d=p->extract_reasoning_streaming("","","Sure",{}); REQUIRE(d);
    CHECK(d->reasoning.value_or("")==(opened?"Sure":""));
    CHECK(d->content.value_or("")==(opened?"":"Sure"));
   }
  }
  p->adjust_initial_state_from_prompt({10});
  auto d=p->extract_reasoning_streaming("","","Sure",{}); REQUIRE(d); CHECK(d->reasoning=="Sure");
  p->extract_reasoning_streaming("","","<channel|>",{});
  d=p->extract_reasoning_streaming("","","Hello",{}); REQUIRE(d); CHECK(d->content=="Hello");
  p->adjust_initial_state_from_prompt({10});
  d=p->extract_reasoning_streaming("","","Again",{}); REQUIRE(d); CHECK(d->reasoning=="Again");
 }
}
