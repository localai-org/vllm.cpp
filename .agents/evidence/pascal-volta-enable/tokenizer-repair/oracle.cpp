#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include "llama.h"
int main(int n,char**v){if(n!=3)return 2;llama_backend_init();auto p=llama_model_default_params();p.vocab_only=true;p.n_gpu_layers=0;auto*m=llama_model_load_from_file(v[1],p);if(!m)return 3;const auto*tok=llama_model_get_vocab(m);std::ifstream in(v[2]);nlohmann::json corpus;in>>corpus;for(const auto&x:corpus){auto s=x.get<std::string>();std::vector<llama_token>a(20000),b(20000);int an=llama_tokenize(tok,s.data(),s.size(),a.data(),a.size(),false,true);int bn=llama_tokenize(tok,s.data(),s.size(),b.data(),b.size(),true,true);if(an<0||bn<0)return 4;a.resize(an);b.resize(bn);std::string out(20000,'\0');int z=llama_detokenize(tok,a.data(),a.size(),out.data(),out.size(),false,true);if(z<0)return 5;out.resize(z);std::cout<<nlohmann::json{{"text",s},{"ids",a},{"special_ids",b},{"decoded",out}}.dump()<<'\n';}llama_model_free(m);llama_backend_free();}
