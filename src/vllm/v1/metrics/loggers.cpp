// Ported from: vllm/v1/metrics/loggers.py (PrometheusStatLogger) @ 555967922.
#include "vllm/v1/metrics/loggers.h"

#include <cstdio>
#include <string>
#include <vector>

namespace vllm::v1::metrics {

std::vector<double> Build1_2_5Buckets(int64_t max_value) {
  // loggers.py:1284-1300 build_buckets([1,2,5], max_value).
  const int64_t mantissa[3] = {1, 2, 5};
  std::vector<double> buckets;
  int exponent = 0;
  while (true) {
    int64_t pow10 = 1;
    for (int i = 0; i < exponent; ++i) pow10 *= 10;
    for (int64_t m : mantissa) {
      const int64_t value = m * pow10;
      if (value <= max_value) {
        buckets.push_back(static_cast<double>(value));
      } else {
        return buckets;
      }
    }
    ++exponent;
    // Defensive cap so a pathological max_value cannot loop unbounded.
    if (exponent > 18) return buckets;
  }
}

namespace {

// The two timing-histogram bucket schedules used across loggers.py.
const std::vector<double>& TtftBuckets() {
  static const std::vector<double> b = {
      0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08,   0.1,    0.25,  0.5, 0.75,
      1.0,   2.5,   5.0,  7.5,  10.0, 20.0, 40.0,   80.0,   160.0, 640.0,
      2560.0};
  return b;
}

const std::vector<double>& ItlBuckets() {
  static const std::vector<double> b = {0.01, 0.025, 0.05, 0.075, 0.1,  0.15,
                                        0.2,  0.3,   0.4,  0.5,   0.75, 1.0,
                                        2.5,  5.0,   7.5,  10.0,  20.0, 40.0,
                                        80.0};
  return b;
}

const std::vector<double>& RequestLatencyBuckets() {
  static const std::vector<double> b = {
      0.3,  0.5,  0.8,  1.0,  1.5,   2.0,   2.5,   5.0,   10.0,  15.0, 20.0,
      30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0};
  return b;
}

}  // namespace

PrometheusStatLogger::PrometheusStatLogger(std::string served_model_name,
                                           int64_t max_model_len,
                                           int engine_index,
                                           int num_speculative_tokens)
    : model_name_(std::move(served_model_name)),
      engine_str_(std::to_string(engine_index)),
      labelvalues_({model_name_, std::to_string(engine_index)}) {
  const std::vector<std::string> labels = {"model_name", "engine"};
  const std::vector<double> count_buckets = Build1_2_5Buckets(max_model_len);

  // ── Scheduler state (gauges) ──────────────────────────────────────────────
  registry_.RegisterGauge("vllm:num_requests_running",
                          "Number of requests in model execution batches.",
                          labels);
  registry_.RegisterGauge("vllm:num_requests_waiting",
                          "Number of requests waiting to be processed.", labels);
  // Name parity for the per-reason breakdown; recorded at the aggregate.
  registry_.RegisterGauge(
      "vllm:num_requests_waiting_by_reason",
      "Number of requests waiting to be processed, by reason.",
      {"model_name", "engine", "reason"});
  registry_.RegisterGauge("vllm:kv_cache_usage_perc",
                          "KV-cache usage. 1 means 100 percent usage.", labels);

  // ── Cache + preemption + token counters ───────────────────────────────────
  registry_.RegisterCounter(
      "vllm:prefix_cache_queries",
      "Prefix cache queries, in terms of number of queried tokens.", labels);
  registry_.RegisterCounter(
      "vllm:prefix_cache_hits",
      "Prefix cache hits, in terms of number of cached tokens.", labels);
  registry_.RegisterCounter("vllm:num_preemptions",
                            "Cumulative number of preemption from the engine.",
                            labels);
  registry_.RegisterCounter("vllm:prompt_tokens",
                            "Number of prefill tokens processed.", labels);
  registry_.RegisterCounter(
      "vllm:prompt_tokens_cached",
      "Number of cached prompt tokens (local + external).", labels);
  registry_.RegisterCounter("vllm:generation_tokens",
                            "Number of generation tokens processed.", labels);
  registry_.RegisterCounter("vllm:request_success",
                            "Count of successfully processed requests.",
                            {"model_name", "engine", "finished_reason"});

  // ── Histograms of counts ──────────────────────────────────────────────────
  registry_.RegisterHistogram("vllm:request_prompt_tokens",
                              "Number of prefill tokens processed.", labels,
                              count_buckets);
  registry_.RegisterHistogram("vllm:request_generation_tokens",
                              "Number of generation tokens processed.", labels,
                              count_buckets);
  registry_.RegisterHistogram(
      "vllm:iteration_tokens_total",
      "Histogram of number of tokens per engine_step.", labels,
      {1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384});
  registry_.RegisterHistogram(
      "vllm:request_max_num_generation_tokens",
      "Histogram of maximum number of requested generation tokens.", labels,
      count_buckets);
  registry_.RegisterHistogram("vllm:request_params_n",
                              "Histogram of the n request parameter.", labels,
                              {1, 2, 5, 10, 20});
  registry_.RegisterHistogram(
      "vllm:request_params_max_tokens",
      "Histogram of the max_tokens request parameter.", labels, count_buckets);

  // ── Histograms of timing intervals ────────────────────────────────────────
  registry_.RegisterHistogram("vllm:time_to_first_token_seconds",
                              "Histogram of time to first token in seconds.",
                              labels, TtftBuckets());
  registry_.RegisterHistogram("vllm:inter_token_latency_seconds",
                              "Histogram of inter-token latency in seconds.",
                              labels, ItlBuckets());
  registry_.RegisterHistogram(
      "vllm:request_time_per_output_token_seconds",
      "Histogram of time_per_output_token_seconds per request.", labels,
      ItlBuckets());
  registry_.RegisterHistogram("vllm:e2e_request_latency_seconds",
                              "Histogram of e2e request latency in seconds.",
                              labels, RequestLatencyBuckets());
  registry_.RegisterHistogram(
      "vllm:request_queue_time_seconds",
      "Histogram of time spent in WAITING phase for request.", labels,
      RequestLatencyBuckets());
  registry_.RegisterHistogram(
      "vllm:request_inference_time_seconds",
      "Histogram of time spent in RUNNING phase for request.", labels,
      RequestLatencyBuckets());
  registry_.RegisterHistogram(
      "vllm:request_prefill_time_seconds",
      "Histogram of time spent in PREFILL phase for request.", labels,
      RequestLatencyBuckets());
  registry_.RegisterHistogram(
      "vllm:request_decode_time_seconds",
      "Histogram of time spent in DECODE phase for request.", labels,
      RequestLatencyBuckets());

  registry_.RegisterCounter("vllm:prefill_kv_computed_tokens",
      "Prompt tokens actually completed by the runner, excluding cached tokens.", labels);

  // ── Static config Info ────────────────────────────────────────────────────
  registry_.RegisterInfo("vllm:cache_config_info",
                         "Information of the LLMEngine CacheConfig.",
                         {"kv_cache_size_tokens", "kv_cache_max_concurrency"});

  // Prime every family with this engine's label values so the exposition is
  // fully populated at zero before the first request (prometheus_client child
  // metrics exist from construction).
  const std::vector<std::string> single = {
      "vllm:num_requests_running",
      "vllm:num_requests_waiting",
      "vllm:kv_cache_usage_perc",
      "vllm:prefix_cache_queries",
      "vllm:prefix_cache_hits",
      "vllm:num_preemptions",
      "vllm:prompt_tokens",
      "vllm:prefill_kv_computed_tokens",
      "vllm:prompt_tokens_cached",
      "vllm:generation_tokens",
      "vllm:request_prompt_tokens",
      "vllm:request_generation_tokens",
      "vllm:iteration_tokens_total",
      "vllm:request_max_num_generation_tokens",
      "vllm:request_params_n",
      "vllm:request_params_max_tokens",
      "vllm:time_to_first_token_seconds",
      "vllm:inter_token_latency_seconds",
      "vllm:request_time_per_output_token_seconds",
      "vllm:e2e_request_latency_seconds",
      "vllm:request_queue_time_seconds",
      "vllm:request_inference_time_seconds",
      "vllm:request_prefill_time_seconds",
      "vllm:request_decode_time_seconds"};
  for (const std::string& name : single) registry_.Prime(name, labelvalues_);
  registry_.Prime("vllm:num_requests_waiting_by_reason",
                  {model_name_, engine_str_, "capacity"});
  // request_success is only emitted once a request finishes (labeled by
  // reason); prime the common "stop" reason for exposition presence.
  registry_.Prime("vllm:request_success",
                  {model_name_, engine_str_, "stop"});

  // ── Speculative decoding (config-gated) ───────────────────────────────────
  // loggers.py:477-481. Registers the four spec_decode families and primes
  // their series, or does nothing at all when k == 0. Last, so the always-on
  // catalog's registration order — which is the exposition order — is
  // unchanged for a non-speculative server.
  spec_decoding_prom_.Register(&registry_, num_speculative_tokens, labels,
                               labelvalues_);
}

void PrometheusStatLogger::Inc(const std::string& name, double v) {
  registry_.IncCounter(name, labelvalues_, v);
}
void PrometheusStatLogger::Set(const std::string& name, double v) {
  registry_.SetGauge(name, labelvalues_, v);
}
void PrometheusStatLogger::Obs(const std::string& name, double v) {
  registry_.Observe(name, labelvalues_, v);
}

void PrometheusStatLogger::Record(const SchedulerStats& s,
                                  const IterationStats& it) {
  // One recorder thread (the sync step site, or AsyncLLM's output handler)
  // against N scraping readers — see the header's THREAD SAFETY note.
  std::lock_guard<std::mutex> lock(mu_);
  // Scheduler-state gauges (loggers.py:1109-1123).
  Set("vllm:num_requests_running", static_cast<double>(s.num_running_reqs));
  Set("vllm:num_requests_waiting", static_cast<double>(s.num_waiting_reqs));
  registry_.SetGauge("vllm:num_requests_waiting_by_reason",
                     {model_name_, engine_str_, "capacity"},
                     static_cast<double>(s.num_waiting_reqs));
  Set("vllm:kv_cache_usage_perc", s.kv_cache_usage);

  // Prefix-cache token counters (loggers.py:1125-1129).
  Inc("vllm:prefix_cache_queries",
      static_cast<double>(s.prefix_cache_stats.queries));
  Inc("vllm:prefix_cache_hits",
      static_cast<double>(s.prefix_cache_stats.hits));

  // Speculative decoding (loggers.py:1140-1143). Present only on a step that
  // verified a draft; Observe is itself a no-op while the family is
  // unregistered, so the guard mirrors upstream rather than substituting for it.
  if (s.spec_decoding_stats.has_value()) {
    spec_decoding_prom_.Observe(*s.spec_decoding_stats);
  }

  // Iteration counters (loggers.py:1191-1205).
  Inc("vllm:num_preemptions", static_cast<double>(it.num_preempted_reqs));
  Inc("vllm:prompt_tokens", static_cast<double>(it.num_prompt_tokens));
  Inc("vllm:prefill_kv_computed_tokens",
      static_cast<double>(s.num_prefill_kv_computed_tokens));
  Inc("vllm:prompt_tokens_cached",
      static_cast<double>(it.num_prompt_tokens_cached));
  Inc("vllm:generation_tokens", static_cast<double>(it.num_generation_tokens));
  if (it.iteration_tokens > 0) {
    Obs("vllm:iteration_tokens_total",
        static_cast<double>(it.iteration_tokens));
  }

  // Per-request iteration samples (loggers.py:1205-1219).
  for (double ttft : it.time_to_first_tokens_iter) {
    Obs("vllm:time_to_first_token_seconds", ttft);
  }
  for (double itl : it.inter_token_latencies_iter) {
    Obs("vllm:inter_token_latency_seconds", itl);
  }
  for (int64_t n : it.n_params_iter) {
    Obs("vllm:request_params_n", static_cast<double>(n));
  }
  for (int64_t m : it.max_num_generation_tokens_iter) {
    Obs("vllm:request_max_num_generation_tokens", static_cast<double>(m));
  }

  // Finished-request observations (loggers.py:1221-1257).
  for (const FinishedRequestStats& f : it.finished_requests) {
    registry_.IncCounter("vllm:request_success",
                         {model_name_, engine_str_, f.finish_reason}, 1.0);
    Obs("vllm:e2e_request_latency_seconds", f.e2e_latency);
    Obs("vllm:request_queue_time_seconds", f.queued_time);
    Obs("vllm:request_inference_time_seconds", f.inference_time);
    Obs("vllm:request_prefill_time_seconds", f.prefill_time);
    Obs("vllm:request_decode_time_seconds", f.decode_time);
    Obs("vllm:request_time_per_output_token_seconds",
        f.mean_time_per_output_token);
    Obs("vllm:request_prompt_tokens", static_cast<double>(f.num_prompt_tokens));
    Obs("vllm:request_generation_tokens",
        static_cast<double>(f.num_generation_tokens));
    if (f.max_tokens_param >= 0) {
      Obs("vllm:request_params_max_tokens",
          static_cast<double>(f.max_tokens_param));
    }
  }
}

void PrometheusStatLogger::SetCacheConfigInfo(int64_t kv_cache_size_tokens,
                                              double kv_cache_max_concurrency) {
  std::lock_guard<std::mutex> lock(mu_);
  // Info series value is fixed at 1.0; the config is carried in the labels.
  char conc[32];
  std::snprintf(conc, sizeof(conc), "%g", kv_cache_max_concurrency);
  registry_.SetInfo("vllm:cache_config_info",
                    {std::to_string(kv_cache_size_tokens), conc});
}

}  // namespace vllm::v1::metrics
