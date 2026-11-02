#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
int main(int n,char**v){if(n!=3)return 2;auto f=vllm::GgufFile::Open(v[1]);auto t=vllm::tok::Tokenizer::FromGguf(f);std::ifstream in(v[2]);nlohmann::json corpus;in>>corpus;for(const auto&x:corpus){auto s=x.get<std::string>();auto a=t.Encode(s);auto b=t.EncodeWithSpecialTokens(s);std::cout<<nlohmann::json{{"text",s},{"ids",a},{"special_ids",b},{"decoded",t.Decode(a)}}.dump()<<'\n';}}
