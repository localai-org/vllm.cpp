import json
from io import BytesIO
from pathlib import Path

import pytest

from tools.bench.qwen38_endpoint_bench import (
    HTTPTransport,
    _sample,
    comparison_verdict,
    load_corpus,
    run_workload,
    summarize,
)


def test_load_corpus_accepts_utf8_text_and_messages(tmp_path):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps([
        {"id": "text", "text": "Caffè 東京", "max_tokens": 3},
        {"id": "chat", "messages": [{"role": "user", "content": "Olá 👋"}], "max_tokens": 2},
    ], ensure_ascii=False), encoding="utf-8")
    corpus = load_corpus(path)
    assert corpus[0]["text"] == "Caffè 東京"
    assert corpus[1]["messages"][0]["content"] == "Olá 👋"


class FakeTransport:
    tokenizer_identity = {"name": "fixture-tokenizer", "revision": "abc123"}

    def __init__(self):
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(payload)
        n = payload["max_tokens"]
        return {
            "events": [
                {"offset_s": 0.01 * (i + 1), "token_ids": [100 + i], "text": chr(97 + i)}
                for i in range(n)
            ],
            "prompt_tokens": 7,
            "prompt_token_ids": [10, 11, 12, 13, 14, 15, 16],
            "generated_tokens": n,
            "cache": {"hits": 2, "queries": 3},
            "draft": {"accepted": 1, "proposed": 2},
            "memory_samples": [{"offset_s": 0, "bytes": 1234}],
        }


def test_workload_is_canonical_closed_loop_and_preserves_raw_repetitions():
    transport = FakeTransport()
    config = {
        "endpoint": "http://example.invalid/v1/completions",
        "model": "runtime-selected-model",
        "corpus": [{"id": "p", "text": "héllo", "max_tokens": 3}],
        "draft": "on",
        "concurrency": 2,
        "waves": 5,
    }
    result = run_workload(config, transport)
    assert len(result["samples"]) == 10
    assert [s["wave"] for s in result["samples"]].count(0) == 2
    assert result["raw_repetitions"] == result["samples"]
    assert result["tokenizer_identity"] == transport.tokenizer_identity
    assert len(result["canonical_payload_hash"]) == 64
    assert all(s["engine_generated_token_fingerprint"] for s in result["samples"])
    assert all(s["prompt_token_fingerprint"] for s in result["samples"])
    assert all(s["ttft_s"] == pytest.approx(0.01) for s in result["samples"])
    assert all(s["tpot_s"] == pytest.approx(0.01) for s in result["samples"])
    assert all(s["itl_s"] == pytest.approx([0.01, 0.01]) for s in result["samples"])
    assert all(s["cache"] == {"hits": 2, "queries": 3} for s in result["samples"])
    payload = transport.payloads[0]
    assert payload["model"] == "runtime-selected-model"
    assert payload["seed"] == 0
    assert payload["temperature"] == 0
    assert payload["ignore_eos"] is True
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert "draft" not in payload


def test_serving_refuses_fewer_than_five_waves():
    with pytest.raises(ValueError, match="five"):
        run_workload({
            "endpoint": "http://localhost/v1/completions", "model": "m",
            "corpus": [{"text": "x", "max_tokens": 1}], "waves": 4,
        }, FakeTransport())


def test_summary_uses_interpolated_percentiles_and_measured_makespan():
    samples = [
        {"ttft_s": x, "tpot_s": x / 10, "e2e_s": x + 1,
         "prompt_tokens": 2, "generated_tokens": 2, "itl_s": [x / 10]}
        for x in (1.0, 2.0, 3.0, 4.0)
    ]
    out = summarize(samples, workload_makespan_s=20)
    assert out["request_count"] == 4
    assert out["metrics"]["ttft_s"]["p50"] == pytest.approx(2.5)
    assert out["metrics"]["ttft_s"]["p99"] == pytest.approx(3.97)
    assert out["workload_makespan_s"] == 20
    assert out["request_rate"] == pytest.approx(4 / 20)
    assert out["token_throughput"] == pytest.approx(8 / 20)
    assert out["raw_repetitions"] == samples


def test_workload_promotes_sample_refusal_and_records_full_run_makespan(monkeypatch):
    clock = iter([10.0, 10.0, 10.2, 10.2, 10.5, 10.5, 11.0,
                  11.0, 11.4, 11.4, 12.0, 12.0])
    monkeypatch.setattr("tools.bench.qwen38_endpoint_bench.time.perf_counter", lambda: next(clock))

    class RefusingTransport:
        tokenizer_identity = {"name": "t", "revision": "1"}
        def __call__(self, payload):
            return {"events": [], "refusal_reason": "usage missing", "e2e_s": 0.1}

    result = run_workload({"endpoint": "x", "model": "m", "corpus": [{"text": "x"}],
                           "waves": 5, "concurrency": 1}, RefusingTransport())
    assert result["workload_makespan_s"] == pytest.approx(1.0)
    assert result["refusal_reason"] == "one or more samples were refused"
    assert result["request_rate"] is None
    assert result["token_throughput"] is None


def test_comparison_fails_closed_on_all_canonical_evidence():
    sample = {"refusal_reason": None, "payload_hash": "payload", "prompt_tokens": 7,
              "prompt_token_fingerprint": "prompt", "reply_text_token_fingerprint": "reply"}
    base = {"refusal_reason": None, "canonical_payload_hash": "run-payload",
            "run_identity": "workload-shape",
            "tokenizer_identity": {"name": "tok", "revision": "1"},
            "samples": [sample], "prompt_token_fingerprints": ["prompt"],
            "reply_text_token_fingerprints": ["reply"]}
    matched = comparison_verdict(base, json.loads(json.dumps(base)))
    assert matched["verdict"] == "MATCHED_INPUT"
    assert matched["no_ratio_reason"] is None
    assert matched["reply_text_token_equality"] == "equal"
    mutations = [
        ("top refusal", lambda x: x.update(refusal_reason="no")),
        ("sample refusal", lambda x: x["samples"][0].update(refusal_reason="no")),
        ("payload", lambda x: x.update(canonical_payload_hash="other")),
        ("run identity", lambda x: x.update(run_identity="other")),
        ("tokenizer", lambda x: x.update(tokenizer_identity=None)),
        ("prompt tokens", lambda x: x["samples"][0].update(prompt_tokens=8)),
        ("prompt fingerprint", lambda x: x.update(prompt_token_fingerprints=["other"])),
    ]
    for _, mutate in mutations:
        right = json.loads(json.dumps(base))
        mutate(right)
        assert comparison_verdict(base, right)["verdict"] == "REFUSED"


def test_comparison_refuses_fabricated_aggregate_evidence():
    sample = {"refusal_reason": None, "payload_hash": "payload", "prompt_tokens": 2,
              "prompt_token_fingerprint": "prompt", "reply_text_token_fingerprint": "reply",
              "engine_generated_token_fingerprint": "engine"}
    base = {"refusal_reason": None, "canonical_payload_hash": "payloads",
            "run_identity": "run", "tokenizer_identity": "tok", "samples": [sample],
            "prompt_token_fingerprints": ["prompt"],
            "reply_text_token_fingerprints": ["reply"],
            "engine_generated_token_fingerprints": ["engine"]}
    mutations = [
        lambda x: x["samples"][0].update(prompt_token_fingerprint=None),
        lambda x: x["samples"][0].update(prompt_token_fingerprint="different"),
        lambda x: x.update(prompt_token_fingerprints=["fabricated"]),
        lambda x: x.update(reply_text_token_fingerprints=["fabricated"]),
        lambda x: x.update(engine_generated_token_fingerprints=["fabricated"]),
    ]
    for mutate in mutations:
        right = json.loads(json.dumps(base))
        mutate(right)
        verdict = comparison_verdict(base, right)
        assert verdict["verdict"] == "REFUSED"
        assert "fingerprint" in verdict["refusal_reason"]


def test_canonical_hashes_exclude_runtime_model_names_but_include_semantic_extras():
    config = {"endpoint": "x", "corpus": [{"text": "same", "max_tokens": 2}],
              "waves": 5}
    left = run_workload({**config, "model": "engine-a/model"}, FakeTransport())
    right = run_workload({**config, "model": "engine-b-alias"}, FakeTransport())
    assert left["canonical_payload_hash"] == right["canonical_payload_hash"]
    assert [s["payload_hash"] for s in left["samples"]] == [
        s["payload_hash"] for s in right["samples"]
    ]
    assert comparison_verdict(left, right)["verdict"] == "MATCHED_INPUT"

    for key, values in (("top_k", (10, 20)), ("stop", (["A"], ["B"])),
                        ("chat_template_kwargs", ({"enable_thinking": False},
                                                  {"enable_thinking": True}))):
        class SemanticFake(FakeTransport):
            def __init__(self, value):
                super().__init__()
                self.http = HTTPTransport("x", "tok", extra_json={key: value})
            def semantic_payload(self, payload):
                return self.http.semantic_payload(payload)
        extra_left = run_workload({**config, "model": "alias"}, SemanticFake(values[0]))
        extra_right = run_workload({**config, "model": "other"}, SemanticFake(values[1]))
        assert extra_left["canonical_payload_hash"] != extra_right["canonical_payload_hash"]
        assert extra_left["samples"][0]["payload_hash"] != extra_right["samples"][0]["payload_hash"]
        assert comparison_verdict(extra_left, extra_right)["verdict"] == "REFUSED"

    tf = HTTPTransport("x", "tok", adapter="tensorfold", draft_mode="off")
    semantic = tf.semantic_payload({"model": "alias", "prompt": "same"})
    assert semantic["draft"] is False
    assert "return_token_ids" not in semantic


def test_workload_refuses_missing_or_comparison_different_tokenizer_identity():
    transport = FakeTransport()
    transport.tokenizer_identity = None
    config = {"endpoint": "x", "model": "m", "corpus": [{"text": "x", "max_tokens": 1}],
              "waves": 5}
    missing = run_workload(config, transport)
    assert "tokenizer identity" in missing["refusal_reason"]

    left = run_workload(config, FakeTransport())
    different_transport = FakeTransport()
    different_transport.tokenizer_identity = {"name": "other", "revision": "abc123"}
    right = run_workload(config, different_transport)
    verdict = comparison_verdict(left, right)
    assert verdict["verdict"] == "REFUSED"
    assert "tokenizer" in verdict["refusal_reason"]


def test_http_payload_extra_json_is_explicit_and_cannot_override_canonical_fields():
    transport = HTTPTransport("http://example.invalid", tokenizer_identity="tok@revision",
                              extra_json={"draft_model": "d"})
    assert transport.prepare_payload({"model": "m", "stream": True}) == {
        "model": "m", "stream": True, "draft_model": "d"
    }
    with pytest.raises(ValueError, match="override"):
        HTTPTransport("x", tokenizer_identity="tok", extra_json={"model": "other"}).prepare_payload({"model": "m"})


def test_usage_without_token_ids_keeps_valid_absolute_profile_measurements():
    class UsageOnlyTransport:
        tokenizer_identity = "tok@rev"
        def __call__(self, payload):
            return {"events": [{"offset_s": .1, "text": "one"},
                               {"offset_s": .2, "text": " chunk"}],
                    "prompt_tokens": 3, "generated_tokens": 2, "e2e_s": .2}

    result = run_workload({"endpoint": "x", "model": "m", "corpus": [{"text": "x"}],
                           "waves": 5}, UsageOnlyTransport())
    assert result["refusal_reason"] is None
    assert result["request_rate"] is not None
    assert result["token_throughput"] is not None
    assert all(s["tpot_s"] == pytest.approx(.1) for s in result["samples"])
    assert all(s["engine_generated_token_fingerprint"] is None for s in result["samples"])
    verdict = comparison_verdict(result, result)
    assert verdict["verdict"] == "PROFILE_COMPARISON"
    assert "prompt token fingerprints" in verdict["no_ratio_reason"]
    assert verdict["refusal_reason"] is None


def test_comparison_refuses_different_workload_shape():
    base = {"endpoint": "x", "model": "m", "corpus": [{"text": "x", "max_tokens": 1}]}
    reference = run_workload({**base, "draft": "off", "concurrency": 1, "waves": 5},
                             FakeTransport())
    for changed in ({"draft": "on", "concurrency": 1, "waves": 5},
                    {"draft": "off", "concurrency": 2, "waves": 5},
                    {"draft": "off", "concurrency": 1, "waves": 6}):
        candidate = run_workload({**base, **changed}, FakeTransport())
        verdict = comparison_verdict(reference, candidate)
        assert verdict["verdict"] == "REFUSED"
        assert "run identities" in verdict["refusal_reason"]


def test_tensorfold_adapter_maps_semantics_and_rejects_overrides():
    transport = HTTPTransport("x", tokenizer_identity="tok", adapter="tensorfold",
                              draft_mode="off")
    prepared = transport.prepare_payload({"model": "m", "stream": True})
    assert prepared["draft"] is False
    assert prepared["return_token_ids"] is True
    assert transport.semantic_options == {"draft_mode": "off", "token_evidence": True}

    drafted = HTTPTransport("x", tokenizer_identity="tok", adapter="tensorfold",
                            draft_mode="on")
    assert drafted.prepare_payload({"model": "m"})["draft"] is True
    with pytest.raises(ValueError, match="semantic"):
        HTTPTransport("x", tokenizer_identity="tok", adapter="tensorfold",
                      draft_mode="off", extra_json={"draft": True})


def test_tensorfold_terminal_fixture_parses_tokens_telemetry_and_terminal_e2e(monkeypatch):
    terminal = {
        "choices": [{"index": 0, "text": "", "finish_reason": "length"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        "tensorfold": {"token_ids": [101, 102, 103], "token_sha": "pinned-sha",
                       "prefill_s": 0.12, "decode_s": 0.34, "rounds": 2,
                       "drafts": True, "cached": 5, "min_rows": 2},
    }
    body = (b'data: {"choices":[{"text":"ab"}]}\n\n' +
            b'data: {"choices":[{"text":"c"}]}\n\n' +
            f"data: {json.dumps(terminal)}\n\n".encode() + b"data: [DONE]\n\n")
    monkeypatch.setattr("tools.bench.qwen38_endpoint_bench.urllib.request.urlopen",
                        lambda request: BytesIO(body))
    clock = iter([10.0, 10.1, 10.2, 10.5])
    monkeypatch.setattr("tools.bench.qwen38_endpoint_bench.time.perf_counter", lambda: next(clock))
    response = HTTPTransport("http://example.invalid", "tok", adapter="tensorfold", draft_mode="on")(
        {"model": "m", "stream": True})
    assert response["e2e_s"] == pytest.approx(.5)
    assert response["generated_tokens"] == 3
    assert response["terminal_token_ids"] == [101, 102, 103]
    tensorfold_sample = _sample(response, 10.)
    assert tensorfold_sample["engine_generated_token_fingerprint"]
    assert tensorfold_sample["prompt_token_fingerprint"] is None
    assert response["token_sha"] == "pinned-sha"
    assert response["cache"] == {"cached": 5}
    assert response["draft"] == {"drafts": True, "rounds": 2, "min_rows": 2}
    assert response["prefill"] == {"prefill_s": .12}
    assert response["decode"] == {"decode_s": .34}


def test_vllm_cpp_tokenize_fixture_matches_tensorfold_token_evidence(monkeypatch):
    sse = (b'data: {"choices":[{"text":"same output"}]}\n\n'
           b'data: {"choices":[{"text":""}],"usage":{"prompt_tokens":4,'
           b'"completion_tokens":3,"total_tokens":7}}\n\n'
           b'data: [DONE]\n\n')
    tokenize = b'{"count":3,"max_model_len":32768,"tokens":[101,102,103],"token_strs":null}'
    requests = []

    def urlopen(request):
        requests.append(request)
        return BytesIO(sse if request.full_url.endswith("/v1/completions") else tokenize)

    monkeypatch.setattr("tools.bench.qwen38_endpoint_bench.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("tools.bench.qwen38_endpoint_bench.time.perf_counter",
                        iter([10., 10.1, 10.2]).__next__)
    transport = HTTPTransport("http://host:8000/v1/completions", "tok",
                              adapter="vllm-cpp", draft_mode="off")
    response = transport({"model": "m", "stream": True, "prompt": "input"})
    assert response["prompt_tokens"] == 4
    assert response["terminal_token_ids"] is None
    evidence = transport.enrich_evidence(_sample(response, 10.),
                                         {"model": "m", "prompt": "input"})
    assert evidence["reply_text_token_ids"] == [101, 102, 103]
    assert evidence["prompt_token_ids"] == [101, 102, 103]
    assert requests[-1].full_url == "http://host:8000/tokenize"
    assert json.loads(requests[-1].data) == {"prompt": "input", "add_special_tokens": False}

    vllm_sample = _sample(response, 10.)
    vllm_sample.update(evidence)
    vllm_sample["reply_text_token_fingerprint"] = _sample(
        {"events": [], "prompt_tokens": 0, "generated_tokens": 3,
         "terminal_token_ids": evidence["reply_text_token_ids"], "e2e_s": 0}, 0
    )["engine_generated_token_fingerprint"]
    tensorfold_sample = _sample({"events": [{"offset_s": .1, "text": "same output"}],
                                 "prompt_tokens": 4, "generated_tokens": 3,
                                 "terminal_token_ids": [101, 102, 103], "e2e_s": .2}, 10.)
    assert vllm_sample["engine_generated_token_fingerprint"] is None
    assert vllm_sample["reply_text_token_fingerprint"]
    assert tensorfold_sample["engine_generated_token_fingerprint"]
    assert tensorfold_sample.get("reply_text_token_fingerprint") is None

    left = {"refusal_reason": None, "canonical_payload_hash": "p", "run_identity": "r",
            "tokenizer_identity": "tok",
            "samples": [{"payload_hash": "s", "prompt_tokens": 4,
                         "prompt_token_fingerprint": None,
                         "reply_text_token_fingerprint": vllm_sample["reply_text_token_fingerprint"],
                         "engine_generated_token_fingerprint": None}],
            "prompt_token_fingerprints": [None],
            "reply_text_token_fingerprints": [vllm_sample["reply_text_token_fingerprint"]],
            "engine_generated_token_fingerprints": [None]}
    right = {**left,
             "samples": [{"payload_hash": "s", "prompt_tokens": 4,
                          "prompt_token_fingerprint": None,
                          "reply_text_token_fingerprint": None,
                          "engine_generated_token_fingerprint": tensorfold_sample["engine_generated_token_fingerprint"]}],
             "reply_text_token_fingerprints": [None],
             "engine_generated_token_fingerprints": [tensorfold_sample["engine_generated_token_fingerprint"]]}
    verdict = comparison_verdict(left, right)
    assert verdict["reply_text_token_equality"] == "unavailable"
    assert verdict["verdict"] == "PROFILE_COMPARISON"
    assert "prompt token fingerprints" in verdict["no_ratio_reason"]
    assert verdict["refusal_reason"] is None


def test_vllm_cpp_accepts_explicit_safe_tokenize_url():
    transport = HTTPTransport("https://generation.example/v1/completions", "tok",
                              adapter="vllm-cpp",
                              tokenize_url="https://tokens.example/tokenize")
    assert transport.tokenize_url == "https://tokens.example/tokenize"
    with pytest.raises(ValueError, match="http"):
        HTTPTransport("https://host/v1/completions", "tok", adapter="vllm-cpp",
                      tokenize_url="file:///tmp/tokenize")


def test_deferred_evidence_does_not_affect_generation_timing_or_pacing(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("tools.bench.qwen38_endpoint_bench.time.perf_counter", lambda: clock[0])

    class DeferredTransport:
        tokenizer_identity = "tok"
        def __init__(self):
            self.starts = []
            self.evidence_starts = []
        def __call__(self, payload):
            self.starts.append(clock[0])
            clock[0] += 1
            return {"events": [{"offset_s": 1, "text": "x"}], "e2e_s": 1,
                    "prompt_tokens": 1, "generated_tokens": 1}
        def enrich_evidence(self, sample, payload):
            self.evidence_starts.append(clock[0])
            clock[0] += 100
            return {"reply_text_token_ids": [2], "prompt_token_ids": [1]}

    transport = DeferredTransport()
    result = run_workload({"endpoint": "x", "model": "m", "corpus": [{"text": "p"}],
                           "waves": 5}, transport)
    assert transport.starts == [0, 1, 2, 3, 4]
    assert transport.evidence_starts == [5, 105, 205, 305, 405]
    assert result["workload_makespan_s"] == 5
    assert all(sample["e2e_s"] == 1 for sample in result["samples"])


def test_vllm_cpp_chat_prompt_evidence_fails_closed():
    transport = HTTPTransport("http://host/v1/chat/completions", "tok", adapter="vllm-cpp")
    transport._tokenize = lambda text: [1, 2]
    evidence = transport.enrich_evidence({"generated_text": "answer"},
                                         {"messages": [{"role": "user", "content": "hi"}]})
    assert evidence["reply_text_token_ids"] == [1, 2]
    assert "prompt_token_ids" not in evidence
    assert "chat prompt fingerprint unavailable" in evidence["evidence_refusal_reason"]


def test_itl_requires_one_token_per_timed_event():
    class BatchedDeltaTransport(FakeTransport):
        def __call__(self, payload):
            return {"events": [{"offset_s": .1, "text": "ab", "token_ids": [1, 2]},
                               {"offset_s": .2, "text": "c", "token_ids": [3]}],
                    "prompt_tokens": 2, "generated_tokens": 3, "e2e_s": .3,
                    "terminal_token_ids": [1, 2, 3]}
    result = run_workload({"endpoint": "x", "model": "m", "corpus": [{"text": "x"}],
                           "waves": 5}, BatchedDeltaTransport())
    assert all(s["itl_s"] is None for s in result["samples"])
    assert all("cardinality" in s["itl_unavailable_reason"] for s in result["samples"])
    assert result["metrics"]["itl_s"]["count"] == 0


def test_sample_refuses_usage_without_reliable_generated_token_accounting():
    class NoUsageTransport:
        tokenizer_identity = {"name": "t"}
        def __call__(self, payload):
            return {"events": [{"offset_s": .1, "text": "one"},
                               {"offset_s": .2, "text": " chunk"}]}
    result = run_workload({"endpoint": "x", "model": "m", "corpus": [{"text": "x"}],
                           "waves": 5}, NoUsageTransport())
    assert all(s["generated_tokens"] is None for s in result["samples"])
    assert all(s["tpot_s"] is None for s in result["samples"])
    assert result["token_throughput"] is None
