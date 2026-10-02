// The shared request-level SystemOne scorer. Host-only and weight-free apart
// from the injected logits source, so every function here is testable against
// reference goldens without a model. See decision_scorer.h for the references.
#include "vllm/model_executor/models/decision_scorer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vllm/tokenizer/tokenizer.h"

namespace vllm::decision_scorer {

namespace {

using ojson = nlohmann::ordered_json;

// json.dumps with Python's default separators. nlohmann's dump() writes ","
// and ":" with no space, so containers are walked here and only scalars are
// delegated. Both escape the same set in a string (", \, \b \f \n \r \t, other
// C0 controls as \u00XX in lowercase hex) and leave non-ASCII raw, which is
// ensure_ascii=False.
void DumpPython(const ojson& v, std::string& out) {
  if (v.is_array()) {
    out += '[';
    for (std::size_t i = 0; i < v.size(); ++i) {
      if (i > 0) out += ", ";
      DumpPython(v[i], out);
    }
    out += ']';
    return;
  }
  if (v.is_object()) {
    out += '{';
    bool first = true;
    for (auto it = v.begin(); it != v.end(); ++it) {
      if (!first) out += ", ";
      first = false;
      out += ojson(it.key()).dump();
      out += ": ";
      DumpPython(it.value(), out);
    }
    out += '}';
    return;
  }
  if (v.is_number_float() && !std::isfinite(v.get<double>())) {
    // allow_nan=False.
    throw RequestError("Non-finite JSON number in the request");
  }
  out += v.dump();
}

// openjev Content = str | dict | list.
bool IsContent(const ojson& v) {
  return v.is_string() || v.is_object() || v.is_array();
}

void RequireKeys(const ojson& obj, std::initializer_list<const char*> allowed,
                 const std::string& where) {
  for (auto it = obj.begin(); it != obj.end(); ++it) {
    bool ok = false;
    for (const char* a : allowed) ok = ok || it.key() == a;
    if (!ok) {
      throw RequestError(where + ": unsupported key '" + it.key() + "'");
    }
  }
}

void RequireWidth(std::size_t n, const std::string& where,
                  const CompileLimits& limits) {
  if (n < 2 || n > static_cast<std::size_t>(kMaxAnswers)) {
    throw RequestError(where + ": criteria must hold 2-64 entries");
  }
  if (n > static_cast<std::size_t>(limits.max_choices)) {
    throw RequestError(where + ": " + std::to_string(n) + " choices; " +
                       limits.too_wide_reason);
  }
}

}  // namespace

// Python str.strip() over the ASCII whitespace str.isspace() accepts.
bool IsBlank(std::string_view s) {
  return std::all_of(s.begin(), s.end(), [](unsigned char c) {
    return c == ' ' || (c >= '\t' && c <= '\r') || (c >= 0x1c && c <= 0x1f);
  });
}

std::string JsonDumps(const ojson& value) {
  std::string out;
  DumpPython(value, out);
  return out;
}

std::string Serialize(const ojson& value) {
  return value.is_string() ? value.get<std::string>() : JsonDumps(value);
}

Request CompileRequest(const ojson& body, const CompileLimits& limits) {
  if (!body.is_object()) throw RequestError("request body must be an object");
  RequireKeys(body, {"model", "state", "questions"}, "request");
  Request r;
  if (body.contains("model")) {
    if (!body["model"].is_string() || body["model"].get<std::string>().empty()) {
      throw RequestError("model must be a non-empty string");
    }
    r.model = body["model"].get<std::string>();
  }
  if (!body.contains("state") || !IsContent(body["state"])) {
    throw RequestError("state is required and must be a string, object or array");
  }
  r.state = body["state"];
  r.context = Serialize(body["state"]);
  // parallel_schema.py prepare_prompts: "Context must be a nonempty string."
  if (IsBlank(r.context)) throw RequestError("Context must be a nonempty string.");

  if (!body.contains("questions") || !body["questions"].is_object() ||
      body["questions"].empty()) {
    throw RequestError("questions is required and must be a non-empty object");
  }
  if (body["questions"].size() > static_cast<std::size_t>(kMaxQuestions)) {
    throw RequestError("questions must hold 1-64 entries");
  }
  for (auto it = body["questions"].begin(); it != body["questions"].end(); ++it) {
    const std::string where = "question '" + it.key() + "'";
    const ojson& q = it.value();
    if (IsBlank(it.key())) {
      throw RequestError("Fields require a nonempty string name and an object definition.");
    }
    if (!q.is_object() || !q.contains("type") || !q["type"].is_string()) {
      throw RequestError(where + " must be an object with a string type");
    }
    RequireKeys(q, {"type", "instructions", "criteria"}, where);
    Field f;
    f.name = it.key();
    f.type = q["type"].get<std::string>();
    if (!q.contains("instructions") || !IsContent(q["instructions"])) {
      throw RequestError(where + ": instructions is required");
    }
    f.description = Serialize(q["instructions"]);
    if (IsBlank(f.description)) {
      throw RequestError(where + ": a nonempty description is required.");
    }
    if (f.type == "noul") {
      // openjev NoulCriteria: optional "true" / "false", default Yes / No.
      std::string yes = "Yes";
      std::string no = "No";
      if (q.contains("criteria")) {
        const ojson& c = q["criteria"];
        if (!c.is_object()) throw RequestError(where + ": criteria must be an object");
        RequireKeys(c, {"true", "false"}, where + " criteria");
        for (const char* k : {"true", "false"}) {
          if (c.contains(k) && !c[k].is_string()) {
            throw RequestError(where + ": criteria." + k + " must be a string");
          }
        }
        if (c.contains("true")) yes = c["true"].get<std::string>();
        if (c.contains("false")) no = c["false"].get<std::string>();
      }
      f.values = {ojson(false), ojson(true)};
      f.value_descriptions = {no, yes};
      f.keys = {"false", "true"};
    } else if (f.type == "choice") {
      if (!q.contains("criteria") || !q["criteria"].is_object()) {
        throw RequestError(where + ": choice requires a criteria object");
      }
      const ojson& c = q["criteria"];
      RequireWidth(c.size(), where, limits);
      for (auto ci = c.begin(); ci != c.end(); ++ci) {
        if (!ci.value().is_string() && !ci.value().is_null()) {
          throw RequestError(where + ": criteria values must be strings or null");
        }
        if (IsBlank(ci.key())) {
          throw RequestError(where + ": enum choices must be 1-26 nonempty strings.");
        }
        f.values.emplace_back(ci.key());
        f.value_descriptions.push_back(ci.value().is_null()
                                           ? ci.key()
                                           : ci.value().get<std::string>());
        f.keys.push_back(ci.key());
      }
    } else if (f.type == "score") {
      if (!q.contains("criteria") || !q["criteria"].is_array()) {
        throw RequestError(where + ": score requires a criteria array");
      }
      const ojson& c = q["criteria"];
      RequireWidth(c.size(), where, limits);
      for (std::size_t i = 0; i < c.size(); ++i) {
        if (!c[i].is_string()) {
          throw RequestError(where + ": score criteria must be strings");
        }
        f.values.emplace_back(std::to_string(i));
        f.value_descriptions.push_back(c[i].get<std::string>());
        f.keys.push_back(std::to_string(i));
      }
    } else {
      throw RequestError(where + " has unknown type: " + f.type);
    }
    r.fields.push_back(std::move(f));
  }
  return r;
}

double EntropyConfidence(const std::vector<double>& p) {
  if (p.size() < 2) return 1.0;
  double entropy = 0.0;
  for (double x : p) {
    if (x > 0.0) entropy -= x * std::log(x);
  }
  const double c = 1.0 - entropy / std::log(static_cast<double>(p.size()));
  return std::min(1.0, std::max(0.0, c));
}

ojson AnswerFromLogits(const Field& field, const std::vector<double>& logits,
                       double temperature) {
  if (logits.size() != field.keys.size() || logits.empty()) {
    throw std::runtime_error("decision: candidate logit count differs from the field");
  }
  if (!(temperature > 0.0) || !std::isfinite(temperature)) {
    throw std::runtime_error("decision: temperature must be a positive finite number");
  }
  // openjev scoring.normalize: exp((v - peak) / T) / sum.
  double peak = logits[0];
  for (double v : logits) {
    if (!std::isfinite(v)) throw std::runtime_error("decision: non-finite candidate logit");
    peak = std::max(peak, v);
  }
  std::vector<double> p(logits.size());
  double total = 0.0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    p[i] = std::exp((logits[i] - peak) / temperature);
    total += p[i];
  }
  for (double& x : p) x /= total;

  ojson dist = ojson::object();
  for (std::size_t i = 0; i < p.size(); ++i) dist[field.keys[i]] = p[i];

  ojson out = ojson::object();
  if (field.type == "noul") {
    out["type"] = "noul";
    out["noul"] = p[1];
    return out;
  }
  if (field.type == "score") {
    double score = 0.0;
    ojson legend = ojson::object();
    for (std::size_t i = 0; i < p.size(); ++i) {
      score += static_cast<double>(i) * p[i];
      legend[field.keys[i]] = field.value_descriptions[i];
    }
    out["type"] = "score";
    out["score"] = score;
    out["legend"] = std::move(legend);
    out["probabilities"] = std::move(dist);
    out["confidence"] = EntropyConfidence(p);
    return out;
  }
  // max(distribution, key=distribution.__getitem__): the first maximum.
  std::size_t best = 0;
  for (std::size_t i = 1; i < p.size(); ++i) {
    if (p[i] > p[best]) best = i;
  }
  out["type"] = "choice";
  out["choice"] = field.keys[best];
  out["probabilities"] = std::move(dist);
  out["confidence"] = EntropyConfidence(p);
  return out;
}

// Every prompt ends with the same assistant boundary and added tokens are hard
// boundaries, so checking the boundary once is the whole-prompt check
// (extended_schema.py candidate_suffix makes the same reduction).
std::vector<int32_t> CandidateIds(const tok::Tokenizer& tokenizer,
                                  std::string_view boundary, size_t count,
                                  std::string_view model) {
  const std::string tail_text(boundary);
  const std::vector<int32_t> tail = tokenizer.Encode(tail_text);
  std::vector<int32_t> ids;
  for (size_t i = 0; i < count; ++i) {
    const std::string code(1, static_cast<char>('A' + i));
    const std::vector<int32_t> combined = tokenizer.Encode(tail_text + code);
    if (combined.size() != tail.size() + 1 ||
        !std::equal(tail.begin(), tail.end(), combined.begin()) ||
        tokenizer.IsSpecial(combined.back())) {
      throw std::runtime_error(std::string(model) + ": choice code " + code +
                               " is not one ordinary token at the answer boundary");
    }
    ids.push_back(combined.back());
  }
  std::vector<int32_t> sorted = ids;
  std::sort(sorted.begin(), sorted.end());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
    throw std::runtime_error(std::string(model) +
                             ": choice codes must have distinct token ids");
  }
  return ids;
}

ScoredRequest ScoreRequest(const tok::Tokenizer& tokenizer,
                           const Request& request,
                           const std::vector<std::string>& prompts,
                           std::string_view boundary, double temperature,
                           int64_t max_prompt_tokens, std::string_view model,
                           const CandidateLogitsFn& logits_fn) {
  if (prompts.size() != request.fields.size()) {
    throw std::runtime_error(std::string(model) + ": one prompt per field is required");
  }
  // Tokenize every prompt first: the reference refuses an over-long request
  // before it scores anything, and names the longest prompt.
  std::vector<std::vector<int32_t>> ids;
  ids.reserve(prompts.size());
  size_t longest = 0;
  for (const std::string& p : prompts) {
    ids.push_back(tokenizer.Encode(p));
    longest = std::max(longest, ids.back().size());
  }
  if (static_cast<int64_t>(longest) > max_prompt_tokens) {
    throw RequestError("Longest prompt has " + std::to_string(longest) +
                       " tokens; limit is " + std::to_string(max_prompt_tokens) +
                       ". Nothing was truncated.");
  }

  size_t widest = 0;
  for (const Field& f : request.fields) widest = std::max(widest, f.keys.size());
  const std::vector<int32_t> codes = CandidateIds(tokenizer, boundary, widest, model);
  std::vector<std::vector<int32_t>> candidates;
  candidates.reserve(request.fields.size());
  for (const Field& f : request.fields) {
    candidates.emplace_back(codes.begin(),
                            codes.begin() + static_cast<std::ptrdiff_t>(f.keys.size()));
  }

  const CandidateLogits got = logits_fn(ids, candidates);
  if (got.logits.size() != request.fields.size()) {
    throw std::runtime_error(std::string(model) +
                             ": the scorer returned the wrong number of fields");
  }
  ScoredRequest out;
  for (size_t i = 0; i < request.fields.size(); ++i) {
    const Field& f = request.fields[i];
    out.answers[f.name] = AnswerFromLogits(f, got.logits[i], temperature);
    out.input_tokens += static_cast<int64_t>(ids[i].size());
  }
  out.output_tokens = got.output_tokens;
  return out;
}

}  // namespace vllm::decision_scorer
