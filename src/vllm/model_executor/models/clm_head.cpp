// CLM (Contrastive-LM) heads, text construction and answers (MODEL-CLM).
//
// Ported from Contrastive-LM/CLM @ bb42c6c5bf914fd449bed2f6ca65be80602cb1f7
// (src/clm/heads.py, schema.py, engine.py, embedder.py); see
// .agents/specs/clm.md "Fidelity repair" for what the first port got wrong.

#include "vllm/model_executor/models/clm.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <unordered_map>

#include "vllm/model_executor/model_loader/safetensors_reader.h"

namespace vllm {
namespace clm {

namespace {

using ojson = nlohmann::ordered_json;

float Act(Activation a, float x) {
  switch (a) {
    case Activation::kGelu:  // nn.GELU(): the exact erf form
      return x * 0.5F * (1.0F + std::erf(x * 0.7071067811865475F));
    case Activation::kRelu:
      return x > 0.0F ? x : 0.0F;
    case Activation::kSilu:
      return x / (1.0F + std::exp(-x));
  }
  return x;
}

// y = W x + b, W [out, in] (nn.Linear).
std::vector<float> Linear(const std::vector<float>& x, const std::vector<float>& w,
                          const std::vector<float>& b, int64_t in, int64_t out) {
  std::vector<float> y(static_cast<size_t>(out));
  for (int64_t j = 0; j < out; ++j) {
    const float* wr = w.data() + static_cast<size_t>(j * in);
    double acc = 0.0;
    for (int64_t i = 0; i < in; ++i) {
      acc += static_cast<double>(x[static_cast<size_t>(i)]) * static_cast<double>(wr[i]);
    }
    y[static_cast<size_t>(j)] = static_cast<float>(acc + b[static_cast<size_t>(j)]);
  }
  return y;
}

// nn.LayerNorm(n), eps 1e-5, biased variance.
void LayerNormInPlace(std::vector<float>& x, const std::vector<float>& g,
                      const std::vector<float>& b) {
  const size_t n = x.size();
  double mean = 0.0;
  for (float v : x) mean += v;
  mean /= static_cast<double>(n);
  double var = 0.0;
  for (float v : x) var += (v - mean) * (v - mean);
  var /= static_cast<double>(n);
  const double inv = 1.0 / std::sqrt(var + 1e-5);
  for (size_t i = 0; i < n; ++i) {
    x[i] = static_cast<float>((x[i] - mean) * inv * g[i] + b[i]);
  }
}

double Norm(const std::vector<float>& v) {
  double s = 0.0;
  for (float x : v) s += static_cast<double>(x) * x;
  return std::sqrt(s);
}

bool IsSpace(unsigned char c) {
  // str.isspace for the ASCII range: \t\n\v\f\r, \x1c-\x1f and space.
  return c == ' ' || (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x1f);
}

std::string Strip(const std::string& s) {
  size_t a = 0, e = s.size();
  while (a < e && IsSpace(static_cast<unsigned char>(s[a]))) ++a;
  while (e > a && IsSpace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(a, e - a);
}

// Python repr(float): the shortest round-trip digits, fixed notation for a
// decimal exponent in [-4, 16), scientific otherwise, and ".0" on integers.
std::string PyFloat(double v) {
  if (std::isnan(v)) return "nan";
  if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
  char buf[64];
  auto r = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::scientific);
  std::string sci(buf, r.ptr);
  const size_t epos = sci.find('e');
  std::string mant = sci.substr(0, epos);
  const int exp = std::stoi(sci.substr(epos + 1));
  std::string sign;
  if (!mant.empty() && mant[0] == '-') {
    sign = "-";
    mant.erase(0, 1);
  }
  std::string digits;
  for (char c : mant) {
    if (c != '.') digits += c;
  }
  if (exp < -4 || exp >= 16) {
    std::string m = digits.substr(0, 1);
    if (digits.size() > 1) m += "." + digits.substr(1);
    char e[16];
    std::snprintf(e, sizeof(e), "e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
    return sign + m + e;
  }
  std::string out;
  if (exp < 0) {
    out = "0." + std::string(static_cast<size_t>(-exp - 1), '0') + digits;
  } else if (static_cast<int>(digits.size()) <= exp + 1) {
    out = digits + std::string(static_cast<size_t>(exp + 1) - digits.size(), '0') + ".0";
  } else {
    out = digits.substr(0, static_cast<size_t>(exp + 1)) + "." +
          digits.substr(static_cast<size_t>(exp + 1));
  }
  return sign + out;
}

// Python repr() of a JSON value, for the one error message that prints one.
std::string PyRepr(const ojson& v) {
  if (v.is_null()) return "None";
  if (v.is_boolean()) return v.get<bool>() ? "True" : "False";
  if (v.is_string()) {
    const std::string s = v.get<std::string>();
    const char q = (s.find('\'') != std::string::npos && s.find('"') == std::string::npos) ? '"' : '\'';
    std::string out(1, q);
    for (char c : s) {
      if (c == '\\' || c == q) out += '\\';
      out += c;
    }
    return out + q;
  }
  return ToText(v);
}

bool NonEmptyContainer(const ojson& v) {
  return (v.is_object() || v.is_array()) && !v.empty();
}

// Python truthiness of a JSON value (the `crit or {}` in candidates).
bool Truthy(const ojson& v) {
  if (v.is_null()) return false;
  if (v.is_boolean()) return v.get<bool>();
  if (v.is_number()) return v.get<double>() != 0.0;
  return !v.empty() && !(v.is_string() && v.get<std::string>().empty());
}

// `d in (None, "")`.
bool NoneOrEmptyString(const ojson& v) {
  return v.is_null() || (v.is_string() && v.get<std::string>().empty());
}

const ojson& Field(const ojson& obj, const char* key) {
  static const ojson kNull = nullptr;
  if (!obj.is_object()) return kNull;
  auto it = obj.find(key);
  return it == obj.end() ? kNull : *it;
}

const std::vector<float>& Need(const std::map<std::string, const StTensor*>& all,
                               const std::string& name,
                               const std::vector<int64_t>& shape,
                               std::vector<float>& dst) {
  auto it = all.find(name);
  if (it == all.end()) {
    throw std::runtime_error("clm: head tensor '" + name +
                             "' is missing; convert the checkpoint with "
                             "scripts/convert-clm.py");
  }
  const StTensor& t = *it->second;
  if (t.dtype != "F32" || t.shape != shape) {
    std::string want, got;
    for (int64_t d : shape) want += std::to_string(d) + ",";
    for (int64_t d : t.shape) got += std::to_string(d) + ",";
    throw std::runtime_error("clm: head tensor '" + name + "' is " + t.dtype + " [" +
                             got + "], expected F32 [" + want + "]");
  }
  dst.resize(t.nbytes / sizeof(float));
  std::memcpy(dst.data(), t.data, t.nbytes);
  return dst;
}

}  // namespace

float ClmScale(double logit_scale) {
  return std::min(std::exp(static_cast<float>(logit_scale)), 100.0F);
}

HeadParams ParseHeadParams(const nlohmann::json& raw, int64_t hidden_size) {
  auto need = [&](const char* key) -> const nlohmann::json& {
    if (!raw.contains(key)) {
      throw std::runtime_error(std::string("clm: config.json has no '") + key +
                               "'; convert the checkpoint with scripts/convert-clm.py");
    }
    return raw[key];
  };
  HeadParams p;
  p.hidden_size = need("clm_hidden_size").get<int64_t>();
  if (p.hidden_size != hidden_size) {
    throw std::runtime_error("clm: clm_hidden_size " + std::to_string(p.hidden_size) +
                             " is not the backbone hidden_size " +
                             std::to_string(hidden_size));
  }
  p.width = need("clm_width").get<int64_t>();
  p.depth = need("clm_depth").get<int64_t>();
  p.proj_dim = need("clm_projection_dim").get<int64_t>();
  p.layernorm = need("clm_layernorm").get<bool>();
  p.residual = need("clm_residual").get<bool>();
  p.logit_scale = need("clm_logit_scale").get<double>();
  const std::string act = need("clm_activation").get<std::string>();
  if (act == "gelu") p.activation = Activation::kGelu;
  else if (act == "relu") p.activation = Activation::kRelu;
  else if (act == "silu") p.activation = Activation::kSilu;
  else throw std::runtime_error("clm: unsupported clm_activation '" + act + "'");
  if (p.depth < 2) throw std::runtime_error("clm: clm_depth must be >= 2");
  return p;
}

HeadWeights LoadHeadWeights(const std::vector<SafetensorsFile>& shards,
                            const HeadParams& params) {
  std::map<std::string, const StTensor*> all;
  for (const SafetensorsFile& s : shards) {
    for (const std::string& n : s.Names()) {
      if (n.rfind("state_head.", 0) == 0 || n.rfind("action_head.", 0) == 0) {
        all[n] = &s.Get(n);
      }
    }
  }
  const int64_t H = params.hidden_size, W = params.width, P = params.proj_dim;
  auto load = [&](const std::string& prefix, MlpHeadWeights& m) {
    Need(all, prefix + ".inp.weight", {W, H}, m.inp_w);
    Need(all, prefix + ".inp.bias", {W}, m.inp_b);
    m.hidden.resize(static_cast<size_t>(params.depth - 2));
    for (size_t i = 0; i < m.hidden.size(); ++i) {
      const std::string h = prefix + ".hidden." + std::to_string(i);
      const std::string n = prefix + ".norms." + std::to_string(i);
      Need(all, h + ".weight", {W, W}, m.hidden[i].w);
      Need(all, h + ".bias", {W}, m.hidden[i].b);
      if (params.layernorm) {
        Need(all, n + ".weight", {W}, m.hidden[i].norm_w);
        Need(all, n + ".bias", {W}, m.hidden[i].norm_b);
      }
    }
    Need(all, prefix + ".out.weight", {P, W}, m.out_w);
    Need(all, prefix + ".out.bias", {P}, m.out_b);
  };
  HeadWeights hw;
  load("state_head", hw.state_head);
  load("action_head", hw.action_head);
  return hw;
}

std::vector<float> ClmMlpHeadForward(const MlpHeadWeights& hw, const HeadParams& params,
                                     const std::vector<float>& x_in) {
  const int64_t H = params.hidden_size, W = params.width, P = params.proj_dim;
  std::vector<float> x = Linear(x_in, hw.inp_w, hw.inp_b, H, W);
  for (float& v : x) v = Act(params.activation, v);
  for (const HiddenBlock& blk : hw.hidden) {
    std::vector<float> h = Linear(x, blk.w, blk.b, W, W);
    if (params.layernorm) LayerNormInPlace(h, blk.norm_w, blk.norm_b);
    for (float& v : h) v = Act(params.activation, v);
    if (params.residual) {
      for (size_t i = 0; i < x.size(); ++i) x[i] += h[i];
    } else {
      x = std::move(h);
    }
  }
  std::vector<float> y = Linear(x, hw.out_w, hw.out_b, W, P);
  // F.normalize(dim=-1): x / max(|x|, 1e-12).
  const double n = std::max(Norm(y), 1e-12);
  for (float& v : y) v = static_cast<float>(v / n);
  return y;
}

std::vector<float> PoolEmbedding(const std::vector<float>& last_hidden) {
  // vLLM's pooler normalizes, then embedder.py l2 divides by |x| + 1e-12.
  std::vector<float> v = last_hidden;
  const double n = Norm(v) + 1e-12;
  for (float& x : v) x = static_cast<float>(x / n);
  return v;
}

std::string ToText(const ojson& x, int indent) {
  if (x.is_null()) return "";
  if (x.is_string()) return x.get<std::string>();
  if (x.is_boolean()) return x.get<bool>() ? "true" : "false";
  if (x.is_number_integer() || x.is_number_unsigned()) return x.dump();
  if (x.is_number_float()) return PyFloat(x.get<double>());
  const std::string pad(static_cast<size_t>(indent), ' ');
  std::string out;
  if (x.is_object()) {
    const std::string sep = indent == 0 ? "\n\n" : "\n";
    bool first = true;
    for (auto it = x.begin(); it != x.end(); ++it) {
      if (!first) out += sep;
      first = false;
      if (NonEmptyContainer(it.value())) {
        out += pad + it.key() + ":\n" + ToText(it.value(), indent + 2);
      } else {
        out += pad + it.key() + ": " + ToText(it.value());
      }
    }
    return out;
  }
  bool first = true;
  for (const ojson& v : x) {
    if (!first) out += "\n";
    first = false;
    if (NonEmptyContainer(v)) {
      out += pad + "-\n" + ToText(v, indent + 2);
    } else {
      out += pad + "- " + ToText(v);
    }
  }
  return out;
}

Pair BuildPair(const ojson& state, const ojson& q) {
  if (!q.is_object()) throw RequestError("question must be an object");
  const ojson& t = Field(q, "type");
  const bool known = t.is_string() && (t == "noul" || t == "choice" || t == "score");
  if (!known) {
    throw RequestError("unknown question type " + PyRepr(t) +
                       "; expected one of ('noul', 'choice', 'score')");
  }
  const std::string type = t.get<std::string>();
  const ojson& crit = Field(q, "criteria");
  const std::string ins = Strip(ToText(Field(q, "instructions")));
  Pair p;
  if (type == "choice") {
    if (!crit.is_object() || crit.empty()) {
      throw RequestError("choice question needs a non-empty 'criteria' object");
    }
    for (auto it = crit.begin(); it != crit.end(); ++it) {
      p.keys.push_back(it.key());
      p.candidates.push_back(NoneOrEmptyString(it.value()) ? it.key() : ToText(it.value()));
    }
  } else if (type == "score") {
    if (!crit.is_array() || crit.size() < 2) {
      throw RequestError("score question needs 'criteria' as an ordered list of >= 2 levels");
    }
    for (size_t i = 0; i < crit.size(); ++i) {
      p.keys.push_back(std::to_string(i));
      p.candidates.push_back(ToText(crit[i]));
    }
  } else {
    for (const char* k : {"false", "true"}) {
      ojson d = nullptr;
      if (Truthy(crit) && crit.is_object() && crit.contains(k)) d = crit[k];
      std::string text;
      if (NoneOrEmptyString(d)) {
        if (ins.empty()) {
          text = k;
        } else {
          text = std::string(k[0] == 't' ? "Yes. This is true: " : "No. This is false: ") + ins;
        }
      } else {
        text = ToText(d);
      }
      p.keys.push_back(k);
      p.candidates.push_back(std::string(k) + ": " + text);
    }
  }
  // state_text: context first, question last.
  const std::string s = Strip(ToText(state));
  p.state_text = (!s.empty() && !ins.empty()) ? s + "\n\n" + ins : (s.empty() ? ins : s);
  return p;
}

ojson AnswerFromLogits(const ojson& q, const std::vector<std::string>& keys,
                       const std::vector<double>& logits) {
  const double m = *std::max_element(logits.begin(), logits.end());
  std::vector<double> probs;
  double z = 0.0;
  for (double v : logits) {
    probs.push_back(std::exp(v - m));
    z += probs.back();
  }
  for (double& p : probs) p /= z;
  size_t j = 0;
  for (size_t i = 1; i < probs.size(); ++i) {
    if (probs[i] > probs[j]) j = i;
  }
  double conf = 1.0;
  if (probs.size() >= 2) {
    double rest = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
      if (i != j) rest += probs[i];
    }
    conf = std::max(0.0, std::min(1.0, probs[j] - rest / static_cast<double>(probs.size() - 1)));
  }
  ojson dist = ojson::object();
  for (size_t i = 0; i < keys.size(); ++i) dist[keys[i]] = probs[i];
  const std::string type = q["type"].get<std::string>();
  ojson a = ojson::object();
  a["type"] = type;
  if (type == "noul") {
    a["noul"] = dist["true"];
    return a;
  }
  if (type == "choice") {
    a["choice"] = keys[j];
    a["confidence"] = conf;
    a["probabilities"] = std::move(dist);
    return a;
  }
  double score = 0.0;
  for (size_t i = 0; i < probs.size(); ++i) score += static_cast<double>(i) * probs[i];
  ojson legend = ojson::object();
  const ojson& levels = q["criteria"];
  for (size_t i = 0; i < levels.size(); ++i) {
    legend[std::to_string(i)] = levels[i].is_string() ? levels[i].get<std::string>()
                                                      : ToText(levels[i]);
  }
  a["score"] = score;
  a["confidence"] = conf;
  a["legend"] = std::move(legend);
  a["probabilities"] = std::move(dist);
  return a;
}

ojson Answer(const HeadWeights& heads, const HeadParams& params, const ojson& body,
             const EmbedFn& embed) {
  // server.py systemone: the body shape and the temperature type.
  if (!body.is_object() || !body.contains("state") || !Field(body, "questions").is_object()) {
    throw RequestError("body must be {state, model, questions}");
  }
  double temperature = 1.0;
  if (body.contains("temperature")) {
    const ojson& t = body["temperature"];
    if (!t.is_number()) throw RequestError("temperature must be a number");
    temperature = t.get<double>();
  }
  // engine.py Engine.answer.
  const ojson& questions = body["questions"];
  if (questions.empty()) throw RequestError("questions must not be empty");
  if (!(temperature > 0.0 && temperature <= 100.0)) {
    throw RequestError("temperature must be in (0, 100]");
  }
  std::vector<Pair> pairs;
  for (auto it = questions.begin(); it != questions.end(); ++it) {
    pairs.push_back(BuildPair(body["state"], it.value()));
  }

  // Encode every distinct text once (the reference embedder's dict.fromkeys),
  // states through the state head and candidates through the action head.
  int64_t tokens = 0;
  std::unordered_map<std::string, std::vector<float>> pooled;
  auto pool = [&](const std::string& text) -> const std::vector<float>& {
    auto it = pooled.find(text);
    if (it == pooled.end()) {
      it = pooled.emplace(text, PoolEmbedding(embed(text, &tokens))).first;
    }
    return it->second;
  };
  const float scale = ClmScale(params.logit_scale);
  ojson answers = ojson::object();
  size_t qi = 0;
  for (auto it = questions.begin(); it != questions.end(); ++it, ++qi) {
    const Pair& p = pairs[qi];
    const std::vector<float> zq = ClmMlpHeadForward(heads.state_head, params, pool(p.state_text));
    std::vector<double> logits;
    for (const std::string& c : p.candidates) {
      const std::vector<float> za = ClmMlpHeadForward(heads.action_head, params, pool(c));
      double cos = 0.0;
      for (size_t i = 0; i < za.size(); ++i) cos += static_cast<double>(za[i]) * zq[i];
      // Engine.answer: (scale * cos / temperature) on float32 tensors.
      const float logit = static_cast<float>(scale * static_cast<float>(cos)) /
                          static_cast<float>(temperature);
      logits.push_back(logit);
    }
    answers[it.key()] = AnswerFromLogits(it.value(), p.keys, logits);
  }
  ojson out = ojson::object();
  out["answers"] = std::move(answers);
  out["usage"] = {{"billing_units", static_cast<int64_t>(questions.size())},
                  {"input_tokens", tokens},
                  {"output_tokens", 0}};
  return out;
}

}  // namespace clm
}  // namespace vllm
