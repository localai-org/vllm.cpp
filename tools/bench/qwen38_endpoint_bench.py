#!/usr/bin/env python3
"""Canonical, engine-neutral OpenAI endpoint benchmark for Qwen3.8 campaigns."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any, Callable, Iterable


def load_corpus(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate a UTF-8 JSON corpus (a list or ``{"prompts": [...]}``)."""
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    rows = value.get("prompts") if isinstance(value, dict) else value
    if not isinstance(rows, list) or not rows:
        raise ValueError("corpus must contain a non-empty prompt list")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or (("text" in row) == ("messages" in row)):
            raise ValueError(f"corpus item {index} must have exactly one of text or messages")
        if "text" in row and not isinstance(row["text"], str):
            raise ValueError(f"corpus item {index} text must be UTF-8 text")
        if "messages" in row and not isinstance(row["messages"], list):
            raise ValueError(f"corpus item {index} messages must be a list")
        if int(row.get("max_tokens", 1)) < 1:
            raise ValueError(f"corpus item {index} max_tokens must be positive")
    return rows


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _payload(model: str, row: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "model": model,
        "max_tokens": int(row.get("max_tokens", 128)),
        "seed": 0,
        "temperature": 0,
        "top_p": 1,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if "messages" in row:
        payload["messages"] = row["messages"]
    else:
        payload["prompt"] = row["text"]
    return payload


def _semantic_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return engine-neutral request fields used for cross-endpoint identity."""
    return {key: value for key, value in payload.items() if key != "model"}


def _percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    valid = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return {
        "count": len(valid),
        "mean": sum(valid) / len(valid) if valid else None,
        "p50": _percentile(valid, .50),
        "p90": _percentile(valid, .90),
        "p95": _percentile(valid, .95),
        "p99": _percentile(valid, .99),
        "min": min(valid) if valid else None,
        "max": max(valid) if valid else None,
    }


def summarize(samples: list[dict[str, Any]], workload_makespan_s: float | None = None) -> dict[str, Any]:
    """Return percentile aggregates without discarding any raw repetition."""
    successful = [s for s in samples if not s.get("refusal_reason")]
    elapsed = float(workload_makespan_s) if workload_makespan_s is not None else None
    generated_values = [s.get("generated_tokens") for s in successful]
    generated = sum(int(x) for x in generated_values if x is not None)
    accounting_valid = len(successful) == len(samples) and all(x is not None for x in generated_values)
    itls = [x for s in successful for x in (s.get("itl_s") or [])]
    return {
        "request_count": len(samples),
        "successful_requests": len(successful),
        "workload_makespan_s": elapsed,
        "metrics": {
            "ttft_s": _stats(s.get("ttft_s") for s in successful),
            "tpot_s": _stats(s.get("tpot_s") for s in successful),
            "itl_s": _stats(itls),
            "e2e_s": _stats(s.get("e2e_s") for s in successful),
        },
        "request_rate": len(successful) / elapsed if accounting_valid and elapsed else None,
        "token_throughput": generated / elapsed if accounting_valid and elapsed else None,
        "raw_repetitions": samples,
    }


def _sample(response: dict[str, Any], started: float) -> dict[str, Any]:
    events = response.get("events", [])
    offsets = [float(event["offset_s"]) for event in events]
    engine_generated_token_ids = [token for event in events for token in event.get("token_ids", [])]
    texts = [str(event.get("text", "")) for event in events]
    generated_value = response.get("generated_tokens")
    generated = int(generated_value) if generated_value is not None else None
    if "e2e_s" in response:
        e2e = float(response["e2e_s"])
    else:
        e2e = offsets[-1] if offsets else time.perf_counter() - started
    ttft = offsets[0] if offsets else None
    event_token_counts = [len(event.get("token_ids", [])) for event in events]
    timed_token_events = bool(events) and all(count == 1 for count in event_token_counts)
    if timed_token_events:
        itl: list[float] | None = [b - a for a, b in zip(offsets, offsets[1:])]
        itl_reason = None
    else:
        itl = None
        itl_reason = "event-to-token cardinality is not provably one"
    tpot = ((e2e - ttft) / (generated - 1)) if ttft is not None and generated is not None and generated > 1 else None
    terminal_token_ids = response.get("terminal_token_ids")
    if terminal_token_ids is not None:
        engine_generated_token_ids = [int(token) for token in terminal_token_ids]
    generated_fingerprint = _hash(engine_generated_token_ids) if engine_generated_token_ids else None
    prompt_token_ids = response.get("prompt_token_ids")
    prompt_fingerprint = _hash(prompt_token_ids) if prompt_token_ids is not None else None
    refusal = response.get("refusal_reason")
    if generated is None and not refusal:
        refusal = "endpoint did not expose reliable generated-token usage"
    prompt_value = response.get("prompt_tokens")
    if prompt_value is None and not refusal:
        refusal = "endpoint did not expose prompt-token usage"
    return {
        "prompt_tokens": int(prompt_value) if prompt_value is not None else None,
        "generated_tokens": generated,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "itl_s": itl,
        "itl_unavailable_reason": itl_reason,
        "e2e_s": e2e,
        "engine_generated_token_ids": engine_generated_token_ids,
        "engine_generated_token_fingerprint": generated_fingerprint,
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_fingerprint": prompt_fingerprint,
        "generated_text": "".join(texts),
        "cache": response.get("cache"),
        "draft": response.get("draft"),
        "prefill": response.get("prefill"),
        "decode": response.get("decode"),
        "token_sha": response.get("token_sha"),
        "tensorfold": response.get("tensorfold"),
        "memory_samples": response.get("memory_samples", []),
        "refusal_reason": refusal,
    }


def run_workload(config: dict[str, Any], transport: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    """Run fixed-size closed-loop waves using the supplied endpoint transport."""
    waves = int(config.get("waves", 5))
    concurrency = int(config.get("concurrency", 1))
    if waves < 5:
        raise ValueError("serving benchmarks require at least five closed-loop waves")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    corpus = config["corpus"]
    payloads = [_payload(config["model"], row) for row in corpus]
    samples: list[dict[str, Any]] = []
    workload_started = time.perf_counter()

    def one(wave: int, slot: int) -> dict[str, Any]:
        index = (wave * concurrency + slot) % len(payloads)
        payload = payloads[index]
        started = time.perf_counter()
        try:
            sample = _sample(transport(payload), started)
        except Exception as error:  # preserve failed repetitions too
            sample = {"prompt_tokens": None, "generated_tokens": None, "ttft_s": None,
                      "tpot_s": None, "itl_s": [], "e2e_s": time.perf_counter() - started,
                      "engine_generated_token_ids": [], "engine_generated_token_fingerprint": None,
                      "reply_text_token_ids": None, "reply_text_token_fingerprint": None,
                      "prompt_token_ids": None, "prompt_token_fingerprint": None,
                      "generated_text": "",
                      "cache": None, "draft": None, "memory_samples": [],
                      "refusal_reason": f"transport error: {error}"}
        semantic_payload = (transport.semantic_payload(payload)
                            if hasattr(transport, "semantic_payload") else _semantic_payload(payload))
        sample.update({"wave": wave, "slot": slot, "corpus_index": index,
                       "payload_hash": _hash(semantic_payload)})
        return sample

    for wave in range(waves):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(one, wave, slot) for slot in range(concurrency)]
            samples.extend(f.result() for f in futures)

    workload_makespan_s = time.perf_counter() - workload_started

    # Evidence-only requests happen after all measured closed-loop generation.
    if hasattr(transport, "enrich_evidence"):
        for sample, payload in zip(samples, (payloads[s["corpus_index"]] for s in samples)):
            try:
                evidence = transport.enrich_evidence(sample, payload)
                sample.update(evidence)
                engine_ids = sample.get("engine_generated_token_ids")
                reply_ids = sample.get("reply_text_token_ids")
                prompt_ids = sample.get("prompt_token_ids")
                sample["engine_generated_token_fingerprint"] = _hash(engine_ids) if engine_ids else None
                sample["reply_text_token_fingerprint"] = _hash(reply_ids) if reply_ids is not None else None
                sample["prompt_token_fingerprint"] = _hash(prompt_ids) if prompt_ids is not None else None
            except Exception as error:
                sample["evidence_refusal_reason"] = f"evidence error: {error}"

    aggregate = summarize(samples, workload_makespan_s)
    tokenizer_identity = getattr(transport, "tokenizer_identity", None)
    refused = any(s.get("refusal_reason") for s in samples)
    refusal_reason = "one or more samples were refused" if refused else None
    if not tokenizer_identity:
        refusal_reason = "tokenizer identity was not provided"
        aggregate["request_rate"] = None
        aggregate["token_throughput"] = None
    semantic_payloads = [(transport.semantic_payload(payload)
                          if hasattr(transport, "semantic_payload") else _semantic_payload(payload))
                         for payload in payloads]
    canonical_payload_hash = _hash(semantic_payloads)
    semantic_options = getattr(transport, "semantic_options", {})
    effective_draft_mode = semantic_options.get("draft_mode", config.get("draft", "off"))
    run_identity = _hash({
        "canonical_payload_hash": canonical_payload_hash,
        "draft_mode": effective_draft_mode,
        "concurrency": concurrency,
        "waves": waves,
    })
    aggregate.update({
        "schema_version": 1,
        "refusal_reason": refusal_reason,
        "endpoint": config["endpoint"],
        "model": config["model"],
        "draft_mode": effective_draft_mode,
        "concurrency": concurrency,
        "waves": waves,
        "canonical_payload_hash": canonical_payload_hash,
        "run_identity": run_identity,
        "tokenizer_identity": tokenizer_identity,
        "prompt_token_fingerprints": [s.get("prompt_token_fingerprint") for s in samples],
        "engine_generated_token_fingerprints": [s.get("engine_generated_token_fingerprint") for s in samples],
        "reply_text_token_fingerprints": [s.get("reply_text_token_fingerprint") for s in samples],
        "samples": samples,
    })
    return aggregate


def comparison_verdict(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Classify input comparability and compare only provenance-equivalent reply evidence."""
    reason = None
    left_samples, right_samples = left.get("samples"), right.get("samples")
    if left.get("refusal_reason") or right.get("refusal_reason"):
        reason = "an input run was refused"
    elif not left.get("canonical_payload_hash") or left.get("canonical_payload_hash") != right.get("canonical_payload_hash"):
        reason = "canonical payload hashes are missing or differ"
    elif not left.get("run_identity") or left.get("run_identity") != right.get("run_identity"):
        reason = "run identities are missing or differ"
    elif not left.get("tokenizer_identity") or left.get("tokenizer_identity") != right.get("tokenizer_identity"):
        reason = "tokenizer identities are missing or differ"
    elif not isinstance(left_samples, list) or not isinstance(right_samples, list) or not left_samples:
        reason = "per-sample evidence is missing"
    elif len(left_samples) != len(right_samples):
        reason = "sample counts differ"
    elif any(s.get("refusal_reason") for s in left_samples + right_samples):
        reason = "an input sample was refused"
    else:
        evidence_fields = (
            ("prompt_token_fingerprint", "prompt_token_fingerprints"),
            ("reply_text_token_fingerprint", "reply_text_token_fingerprints"),
            ("engine_generated_token_fingerprint", "engine_generated_token_fingerprints"),
        )
        for run in (left, right):
            for sample_field, aggregate_field in evidence_fields:
                derived = [sample.get(sample_field) for sample in run["samples"]]
                if aggregate_field in run and run[aggregate_field] != derived:
                    reason = f"{aggregate_field} aggregate/sample fingerprint evidence is inconsistent"
                    break
            if reason:
                break

    if reason is None:
        for left_sample, right_sample in zip(left_samples, right_samples):
            if (not left_sample.get("payload_hash") or
                    left_sample.get("payload_hash") != right_sample.get("payload_hash")):
                reason = "per-sample payload hashes are missing or differ"
                break
            if (left_sample.get("prompt_tokens") is None or
                    left_sample.get("prompt_tokens") != right_sample.get("prompt_tokens")):
                reason = "prompt token counts are missing or differ"
                break
            left_prompt_fp = left_sample.get("prompt_token_fingerprint")
            right_prompt_fp = right_sample.get("prompt_token_fingerprint")
            if left_prompt_fp is not None and right_prompt_fp is not None and left_prompt_fp != right_prompt_fp:
                reason = "per-sample prompt token fingerprints differ"
                break

    left_reply = ([sample.get("reply_text_token_fingerprint") for sample in left_samples]
                  if isinstance(left_samples, list) else None)
    right_reply = ([sample.get("reply_text_token_fingerprint") for sample in right_samples]
                   if isinstance(right_samples, list) else None)
    if left_reply and right_reply and all(left_reply + right_reply):
        reply_text_equality = "equal" if left_reply == right_reply else "different"
    else:
        reply_text_equality = "unavailable"

    if reason:
        return {"verdict": "REFUSED", "refusal_reason": reason, "no_ratio_reason": reason,
                "reply_text_token_equality": reply_text_equality}

    left_prompt = [sample.get("prompt_token_fingerprint") for sample in left_samples]
    right_prompt = [sample.get("prompt_token_fingerprint") for sample in right_samples]
    if any(not fingerprint for fingerprint in left_prompt + right_prompt):
        no_ratio_reason = "prompt token fingerprints are unavailable for one or more samples"
        return {"verdict": "PROFILE_COMPARISON", "refusal_reason": None,
                "no_ratio_reason": no_ratio_reason,
                "reply_text_token_equality": reply_text_equality}
    if left_prompt != right_prompt:
        reason = "prompt token fingerprints differ"
        return {"verdict": "REFUSED", "refusal_reason": reason, "no_ratio_reason": reason,
                "reply_text_token_equality": reply_text_equality}
    return {"verdict": "MATCHED_INPUT", "refusal_reason": None, "no_ratio_reason": None,
            "reply_text_token_equality": reply_text_equality}

class HTTPTransport:
    """UTF-8 OpenAI streaming transport with explicit endpoint semantics."""

    def __init__(self, endpoint: str, tokenizer_identity: Any,
                 extra_json: dict[str, Any] | None = None, *, adapter: str = "generic",
                 draft_mode: str = "off", tokenize_url: str | None = None):
        if adapter not in ("generic", "tensorfold", "vllm-cpp"):
            raise ValueError(f"unknown endpoint adapter: {adapter}")
        self.endpoint = endpoint
        self.tokenizer_identity = tokenizer_identity
        self.adapter = adapter
        self.draft_mode = draft_mode
        self.extra_json = dict(extra_json or {})
        self.tokenize_url = self._tokenize_url(tokenize_url) if adapter == "vllm-cpp" else None
        semantic_fields = {"draft", "return_token_ids"} if adapter == "tensorfold" else set()
        overlap = semantic_fields & self.extra_json.keys()
        if overlap:
            raise ValueError(f"endpoint extra JSON cannot override adapter semantic fields: {sorted(overlap)}")
        self.semantic_options = {"draft_mode": draft_mode, "token_evidence": adapter != "generic"}

    def _tokenize_url(self, explicit: str | None) -> str:
        parsed = urlsplit(explicit or self.endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username:
            raise ValueError("tokenize URL must be an absolute http(s) URL without credentials")
        if explicit:
            return explicit
        return f"{parsed.scheme}://{parsed.netloc}/tokenize"

    def semantic_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Normalize effective semantics, omitting aliases and evidence-only fields."""
        prepared = self.prepare_payload(payload)
        transport_only = {"return_token_ids"} if self.adapter == "tensorfold" else set()
        excluded = {"model"} | transport_only
        semantic = {key: value for key, value in prepared.items() if key not in excluded}
        semantic["draft"] = self.draft_mode == "on"
        return semantic

    def prepare_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        adapter_fields: dict[str, Any] = {}
        if self.adapter == "tensorfold":
            adapter_fields = {"draft": self.draft_mode == "on", "return_token_ids": True}
        overlap = self.extra_json.keys() & (payload.keys() | adapter_fields.keys())
        if overlap:
            raise ValueError(f"endpoint extra JSON cannot override canonical or semantic fields: {sorted(overlap)}")
        return {**payload, **adapter_fields, **self.extra_json}

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(self.endpoint, data=_canonical(self.prepare_payload(payload)),
                                         headers={"Content-Type": "application/json; charset=utf-8"})
        started = time.perf_counter()
        events: list[dict[str, Any]] = []
        usage: dict[str, Any] = {}
        tensorfold: dict[str, Any] = {}
        terminal_offset = 0.0
        with urllib.request.urlopen(request) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                chunk = json.loads(line[5:].strip())
                terminal_offset = time.perf_counter() - started
                usage.update(chunk.get("usage") or {})
                tensorfold.update(chunk.get("tensorfold") or {})
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                text = choice.get("text") or delta.get("content") or delta.get("reasoning_content") or ""
                ids = chunk.get("token_ids", choice.get("token_ids", []))
                if text or ids:
                    events.append({"offset_s": terminal_offset, "text": text, "token_ids": ids})
        return {
            "events": events,
            "e2e_s": terminal_offset,
            "prompt_tokens": usage.get("prompt_tokens"),
            "generated_tokens": usage.get("completion_tokens"),
            "terminal_token_ids": tensorfold.get("token_ids"),
            "token_sha": tensorfold.get("token_sha"),
            "cache": ({"cached": tensorfold["cached"]} if "cached" in tensorfold else None),
            "draft": ({key: tensorfold[key] for key in ("drafts", "rounds", "min_rows")
                       if key in tensorfold} or None),
            "prefill": ({"prefill_s": tensorfold["prefill_s"]} if "prefill_s" in tensorfold else None),
            "decode": ({"decode_s": tensorfold["decode_s"]} if "decode_s" in tensorfold else None),
            "tensorfold": tensorfold or None,
        }

    def _tokenize(self, text: str) -> list[int]:
        request = urllib.request.Request(
            self.tokenize_url,
            data=_canonical({"prompt": text, "add_special_tokens": False}),
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        with urllib.request.urlopen(request) as response:
            tokenized = json.load(response)
        tokens = tokenized.get("tokens")
        if not isinstance(tokens, list) or any(not isinstance(token, int) for token in tokens):
            raise ValueError("vllm-cpp /tokenize response did not contain integer tokens")
        return tokens

    def enrich_evidence(self, sample: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        if self.adapter != "vllm-cpp":
            return {}
        evidence = {"reply_text_token_ids": self._tokenize(sample["generated_text"])}
        if "prompt" in payload:
            evidence["prompt_token_ids"] = self._tokenize(payload["prompt"])
        else:
            evidence["evidence_refusal_reason"] = (
                "chat prompt fingerprint unavailable: identical chat-template application is not exposed"
            )
        return evidence


def _extra_json(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid JSON: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("extra JSON must be an object")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer-identity", required=True,
                        help="shared tokenizer name/revision or immutable digest")
    parser.add_argument("--adapter", choices=("generic", "tensorfold", "vllm-cpp"),
                        default="generic",
                        help="endpoint-specific documented request/response schema")
    parser.add_argument("--tokenize-url",
                        help="vllm-cpp /tokenize URL (defaults safely from endpoint origin)")
    parser.add_argument("--draft", choices=("on", "off"), default="off",
                        help="effective draft policy; tensorfold maps it to the wire")
    parser.add_argument("--extra-json", type=_extra_json, default={}, metavar="OBJECT",
                        help="documented endpoint-specific JSON fields to add to each request")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--waves", type=int, default=5)
    args = parser.parse_args()
    config = vars(args)
    config["corpus"] = load_corpus(args.corpus)
    result = run_workload(
        config, HTTPTransport(args.endpoint, args.tokenizer_identity, args.extra_json,
                              adapter=args.adapter, draft_mode=args.draft,
                              tokenize_url=args.tokenize_url)
    )
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
