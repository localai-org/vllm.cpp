// SystemOne API helpers — implementation. Extracted from api_server.cpp's
// former anonymous namespace so the C ABI (vllm_c.cpp) can share the same
// parse/build logic without going through the ApiServer callback layer.
#include "vllm/entrypoints/openai/systemone.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm::entrypoints::openai::systemone {

// ── Math helpers ───────────────────────────────────────────────────────────

double R2(double x) { return std::round(x * 100.0) / 100.0; }

double R4(double x) { return std::round(x * 10000.0) / 10000.0; }

double ChoiceConfidence(const std::vector<float>& p) {
  size_t k = p.size();
  if (k <= 1) return 1.0;
  float mx = *std::max_element(p.begin(), p.end());
  return (static_cast<double>(mx) - 1.0 / static_cast<double>(k)) /
         (1.0 - 1.0 / static_cast<double>(k));
}

double ScoreConfidence(const std::vector<float>& p) {
  size_t l = p.size();
  if (l <= 1) return 1.0;
  size_t mode = 0;
  float max_p = p[0];
  for (size_t i = 1; i < l; ++i) {
    if (p[i] > max_p) { max_p = p[i]; mode = i; }
  }
  double s = 0.0;
  for (size_t i = 0; i < l; ++i) {
    s += static_cast<double>(p[i]) *
        std::abs(static_cast<double>(i) - static_cast<double>(mode));
  }
  return 1.0 - s / static_cast<double>(l - 1);
}

std::vector<float> Softmax(const std::vector<float>& scores) {
  if (scores.empty()) return {};
  float mx = *std::max_element(scores.begin(), scores.end());
  std::vector<float> exps;
  float sum = 0.0F;
  for (float s : scores) {
    float e = std::exp(s - mx);
    exps.push_back(e);
    sum += e;
  }
  if (sum <= 0.0F) {
    return std::vector<float>(scores.size(), 1.0F / scores.size());
  }
  for (float& e : exps) e /= sum;
  return exps;
}

// ── JSON rendering ─────────────────────────────────────────────────────────

std::string RenderJson(const nlohmann::ordered_json& v, int indent) {
  if (v.is_null()) return "";
  if (v.is_string()) return v.get<std::string>();
  if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
  if (v.is_number()) return v.dump();
  std::string pad(static_cast<size_t>(2 * indent), ' ');
  if (v.is_array()) {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) {
      std::string rendered = RenderJson(v[i], indent + 1);
      auto s = rendered.find_first_not_of(" \t\n");
      if (s == std::string::npos) s = 0;
      if (i > 0) out += "\n";
      out += pad + "- " + rendered.substr(s);
    }
    return out;
  }
  if (v.is_object()) {
    std::string out;
    size_t i = 0;
    for (auto it = v.begin(); it != v.end(); ++it, ++i) {
      if (i > 0) out += "\n";
      if (it.value().is_object() || it.value().is_array()) {
        out += pad + it.key() + ":\n" + RenderJson(it.value(), indent + 1);
      } else {
        out += pad + it.key() + ": " + RenderJson(it.value());
      }
    }
    return out;
  }
  return v.dump();
}

std::string OptionText(const std::string& name,
                        const nlohmann::ordered_json& desc) {
  if (desc.is_null()) return name;
  std::string rendered = RenderJson(desc);
  if (rendered.empty()) return name;
  return name + ": " + rendered;
}

std::string GetInstructions(const nlohmann::ordered_json& qj) {
  if (qj.contains("instructions")) return RenderJson(qj["instructions"]);
  if (qj.contains("instr")) return RenderJson(qj["instr"]);
  return "";
}

std::vector<std::string> NerLabels(const SystemOneQuestion& q) {
  if (q.type == "noul") return {q.instructions};
  return q.labels;
}

// ── Request parsing ────────────────────────────────────────────────────────

ParsedSystemOne ParseSystemOneBody(const nlohmann::ordered_json& body) {
  ParsedSystemOne r;
  if (!body.is_object()) {
    r.ok = false;
    r.error_msg = "request body must be an object";
    return r;
  }
  if (!body.contains("state")) {
    r.ok = false;
    r.error_msg = "state is required";
    return r;
  }
  r.text = RenderJson(body["state"]);
  if (body.contains("model") && body["model"].is_string()) {
    r.model = body["model"].get<std::string>();
  }
  if (body.contains("threshold") && body["threshold"].is_number()) {
    r.threshold = body["threshold"].get<float>();
  }
  if (body.contains("max_width") && body["max_width"].is_number_integer()) {
    r.max_width = body["max_width"].get<int64_t>();
  }
  if (!body.contains("questions") || !body["questions"].is_object()) {
    r.ok = false;
    r.error_msg = "questions is required and must be an object";
    return r;
  }
  if (body["questions"].empty()) {
    r.ok = false;
    r.error_msg = "questions must contain at least one question";
    return r;
  }
  for (auto it = body["questions"].begin(); it != body["questions"].end();
       ++it) {
    SystemOneQuestion q;
    q.id = it.key();
    const auto& qj = it.value();
    if (!qj.is_object() || !qj.contains("type") || !qj["type"].is_string()) {
      r.ok = false;
      r.error_msg = "question '" + q.id + "' must have a string type";
      return r;
    }
    q.type = qj["type"].get<std::string>();
    std::string instr = GetInstructions(qj);
    q.instructions = instr;
    if (q.type == "noul") {
      nlohmann::ordered_json false_desc = nullptr, true_desc = nullptr;
      if (qj.contains("criteria") && qj["criteria"].is_object()) {
        if (qj["criteria"].contains("false"))
          false_desc = qj["criteria"]["false"];
        if (qj["criteria"].contains("true"))
          true_desc = qj["criteria"]["true"];
      }
      q.labels = {OptionText("no", false_desc), OptionText("yes", true_desc)};
      q.keys = {"false", "true"};
    } else if (q.type == "choice") {
      if (!qj.contains("criteria") || !qj["criteria"].is_object() ||
          qj["criteria"].empty()) {
        r.ok = false;
        r.error_msg = "question '" + q.id +
                      "' (choice) requires a non-empty criteria object";
        return r;
      }
      for (auto cit = qj["criteria"].begin(); cit != qj["criteria"].end();
          ++cit) {
        std::string opt = OptionText(cit.key(), cit.value());
        q.keys.push_back(cit.key());
        q.labels.push_back(opt);
      }
    } else if (q.type == "score") {
      if (!qj.contains("criteria") || !qj["criteria"].is_array() ||
          qj["criteria"].size() < 2) {
        r.ok = false;
        r.error_msg = "question '" + q.id +
                      "' (score) requires a criteria array with >= 2 levels";
        return r;
      }
      for (const auto& level : qj["criteria"]) {
        std::string rendered = RenderJson(level);
        q.keys.push_back(rendered);
        q.labels.push_back(rendered);
      }
    } else {
      r.ok = false;
      r.error_msg = "question '" + q.id + "' has unknown type: " + q.type;
      return r;
    }
    for (const auto& l : NerLabels(q)) r.all_labels.push_back(l);
    r.questions.push_back(std::move(q));
  }
  return r;
}

// ── Answer building ───────────────────────────────────────────────────────

nlohmann::json BuildSystemOneAnswer(const SystemOneQuestion& q,
                                     const NerResult& result) {
  // The NER label set, not the decision option texts: a noul question's
  // entities are the ones its INSTRUCTION extracted (NerLabels), while its
  // q.labels are kev's "no"/"yes" option texts.
  const std::vector<std::string> labels = NerLabels(q);
  std::vector<float> scores;
  for (const auto& label : labels) {
    float max_conf = 0.0F;
    for (const auto& e : result.entities) {
      if (e.label == label) max_conf = std::max(max_conf, e.confidence);
    }
    scores.push_back(max_conf);
  }
  std::vector<float> probs;
  if (q.type == "noul") {
    probs = {1.0F - scores[0], scores[0]};
  } else {
    probs = Softmax(scores);
  }
  if (q.type == "noul") {
    nlohmann::json entities = nlohmann::json::array();
    for (const auto& e : result.entities) {
      if (e.label == labels[0]) {
        entities.push_back(nlohmann::json{
            {"text", e.text}, {"start", e.start}, {"end", e.end},
            {"confidence", e.confidence}});
      }
    }
    return nlohmann::json{{"type", "noul"}, {"noul", R2(probs[1])},
                          {"entities", std::move(entities)}};
  }
  if (q.type == "choice") {
    size_t argmax = 0;
    for (size_t i = 1; i < probs.size(); ++i) {
      if (probs[i] > probs[argmax]) argmax = i;
    }
    nlohmann::json dist = nlohmann::json::object();
    for (size_t i = 0; i < q.keys.size(); ++i) {
      dist[q.keys[i]] = R2(probs[i]);
    }
    return nlohmann::json{{"type", "choice"}, {"choice", q.keys[argmax]},
                          {"confidence", R2(ChoiceConfidence(probs))},
                          {"probabilities", std::move(dist)}};
  }
  // score
  double score = 0.0;
  for (size_t i = 0; i < probs.size(); ++i) {
    score += static_cast<double>(i) * probs[i];
  }
  nlohmann::json legend = nlohmann::json::object();
  nlohmann::json dist = nlohmann::json::object();
  for (size_t i = 0; i < q.keys.size(); ++i) {
    legend[std::to_string(i)] = q.keys[i];
    dist[std::to_string(i)] = R2(probs[i]);
  }
  return nlohmann::json{{"type", "score"}, {"score", R2(score)},
                        {"legend", std::move(legend)},
                        {"probabilities", std::move(dist)},
                        {"confidence", R2(ScoreConfidence(probs))}};
}

nlohmann::json BuildSystemOneAnswerDecision(const SystemOneQuestion& q,
                                             const DecisionResult& result) {
  std::vector<float> probs = Softmax(result.scores);

  std::vector<float> act_probs = Softmax(result.act_logits);
  float act_prob = act_probs.empty() ? 0.0F : act_probs[0];
  nlohmann::json rl_agent = {{"act_probability", R4(act_prob)}};

  if (q.type == "noul") {
    return nlohmann::json{{"type", "noul"}, {"noul", R4(probs[1])},
                          {"rl_agent", std::move(rl_agent)}};
  }
  if (q.type == "choice") {
    size_t argmax = 0;
    for (size_t i = 1; i < probs.size(); ++i) {
      if (probs[i] > probs[argmax]) argmax = i;
    }
    nlohmann::json dist = nlohmann::json::object();
    for (size_t i = 0; i < q.keys.size(); ++i) {
      dist[q.keys[i]] = R4(probs[i]);
    }
    return nlohmann::json{{"type", "choice"}, {"choice", q.keys[argmax]},
                          {"probabilities", std::move(dist)},
                          {"confidence", R4(ChoiceConfidence(probs))},
                          {"rl_agent", std::move(rl_agent)}};
  }
  // score
  double score = 0.0;
  for (size_t i = 0; i < probs.size(); ++i) {
    score += static_cast<double>(i) * probs[i];
  }
  nlohmann::json legend = nlohmann::json::object();
  nlohmann::json dist = nlohmann::json::object();
  for (size_t i = 0; i < q.keys.size(); ++i) {
    legend[std::to_string(i)] = q.keys[i];
    dist[std::to_string(i)] = R4(probs[i]);
  }
  return nlohmann::json{{"type", "score"}, {"score", R4(score)},
                        {"legend", std::move(legend)},
                        {"probabilities", std::move(dist)},
                        {"confidence", R4(ScoreConfidence(probs))},
                        {"rl_agent", std::move(rl_agent)}};
}

std::vector<std::string> RenderDecisionOptions(const SystemOneQuestion& q) {
  return q.labels;
}

}  // namespace vllm::entrypoints::openai::systemone
