#!/usr/bin/env python3
"""Fail-closed validator for Qwen3.8 TensorFold baseline evidence."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any

BENCHMARK_ID = "bench-qwen38-tensorfold-gap"
BLOCKED = "BLOCKED_MISSING_ARTIFACTS"
MEASURED = "MEASURED"
MTP_VERDICTS = {"LOADABLE", "BLOCKED_NO_MTP_WEIGHTS", "BLOCKED_CONVERSION_REQUIRED"}
MTP_BLOCKED = "BLOCKED_NO_MTP_WEIGHTS"
WORKLOADS = {"serial_decode", "drafted_decode", "prefill_ladder", "serving_ladder"}
ENGINES = {"tensorfold", "vllm-cpp"}
SEARCH_TERMS = ["mtp", "nextn", "draft", "eh_proj", "enorm", "hnorm"]
TASK_OUTCOMES = {
    "task_5": "NO_PORT_DECISION",
    "task_6": "BLOCKED_NO_MTP_WEIGHTS",
    "task_7": "SYNTHESIS_COMPLETED_NO_PRODUCT_OPTIMIZATION",
}
SHA256 = re.compile(r"[0-9a-f]{64}")
RAW_UUID = re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
SENSITIVE_KEY = re.compile(r"(?i)(?:^|_)(?:job_id|api_key|access_token|auth_token|password|passwd|secret|credential|bearer)(?:$|_)")
SENSITIVE_VALUE = re.compile(r"(?i)(?:bearer\s+[A-Za-z0-9._~+/-]+|(?:api[_-]?key|password|secret|credential)\s*[:=])")


class EvidenceError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def reject_sensitive(value: Any, location: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            require(not SENSITIVE_KEY.search(str(key)), f"prohibited sensitive key at {location}.{key}")
            reject_sensitive(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_sensitive(child, f"{location}[{index}]")
    elif isinstance(value, str):
        require(RAW_UUID.search(value) is None, f"raw lease/job UUID at {location}")
        require(SENSITIVE_VALUE.search(value) is None, f"credential-like value at {location}")


def load_object(path: pathlib.Path) -> dict:
    require(path.is_file(), f"missing file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"invalid JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path} must contain an object")
    reject_sensitive(value, path.name)
    return value


def exact_keys(value: dict, expected: set[str], label: str) -> None:
    require(set(value) == expected, f"{label} keys differ: expected {sorted(expected)}")


def parse_kv_receipt(path: pathlib.Path, required_begin: str | None = None) -> dict[str, str]:
    require(path.is_file(), f"missing raw receipt: {path}")
    text = path.read_text(encoding="utf-8")
    require(RAW_UUID.search(text) is None, f"raw lease/job UUID in {path.name}")
    require(SENSITIVE_VALUE.search(text) is None, f"credential-like value in {path.name}")
    values: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), 1):
        require(re.fullmatch(r"[A-Z][A-Z0-9_]*=.*", line) is not None,
                f"malformed receipt line {path.name}:{number}")
        key, value = line.split("=", 1)
        require(key not in values, f"duplicate receipt marker {key}")
        values[key] = value
    if required_begin:
        require(values.get(required_begin) == "1", f"missing receipt marker {required_begin}")
    return values


def sha256_file(path: pathlib.Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_manifest(path: pathlib.Path, terms: list[str]) -> dict:
    require(path.is_file(), f"committed GGUF manifest is missing: {path}")
    text = path.read_text(encoding="utf-8")
    declared = re.search(r"kQwen4ExpGgufTensorCount\s*=\s*(\d+)", text)
    require(declared is not None, "GGUF manifest lacks its declared tensor count")
    names = re.findall(r'^\s*\{"([^"]+)",', text, flags=re.MULTILINE)
    pattern = re.compile(r"(?:^|[._])(?:" + "|".join(re.escape(x) for x in terms) + r")(?:[._]|$)", re.I)
    matches = [name for name in names if pattern.search(name)]
    blocks = [int(value) for value in re.findall(r'^\s*\{"blk\.(\d+)\.', text, flags=re.MULTILINE)]
    return {
        "declared_tensor_count": int(declared.group(1)),
        "parsed_tensor_count": len(names),
        "mtp_name_matches": matches,
        "min_block": min(blocks) if blocks else None,
        "max_block": max(blocks) if blocks else None,
    }


def validate_leases(summary: dict, lease: dict, directory: pathlib.Path, receipt: dict[str, str], invocation: dict[str, str]) -> None:
    exact_keys(lease, {"device", "jobs", "gpu", "device_ready_after_jobs", "raw_job_ids_retained", "raw_receipt", "invocation_receipt"}, "lease-discovery")
    require(lease["device"] == "dgx:gpu0", "lease-discovery device must be dgx:gpu0")
    require(lease["device_ready_after_jobs"] is True, "device was not ready after discovery")
    require(lease["raw_job_ids_retained"] is False, "raw lease IDs must not be retained")
    require(isinstance(lease["gpu"], dict) and lease["gpu"].get("name") == "NVIDIA GB10",
            "lease-discovery GPU identity is missing")
    jobs = lease["jobs"]
    require(isinstance(jobs, list) and len(jobs) >= 1, "at least one completed lease scan is required")
    for item in jobs:
        exact_keys(item, {"job_fingerprint_sha256", "state"}, "lease job")
        require(item["state"] == "succeeded", "lease discovery job did not succeed")
        require(SHA256.fullmatch(str(item["job_fingerprint_sha256"])) is not None,
                "lease job fingerprint must be SHA-256")
    summary_jobs = summary.get("lease_checks")
    require(isinstance(summary_jobs, list) and len(summary_jobs) == len(jobs),
            "summary lease count disagrees with lease-discovery")
    expected = [{"device": lease["device"], **item} for item in jobs]
    require(summary_jobs == expected, "summary leases disagree with lease-discovery")
    require(lease["raw_receipt"] == "discovery-receipt.stdout" and
            lease["invocation_receipt"] == "discovery-invocation.txt", "lease receipt names differ")
    require(receipt.get("Q38_DEVICE") == lease["device"], "raw receipt device disagrees")
    require(receipt.get("Q38_LEASE_SHA256") == jobs[0]["job_fingerprint_sha256"],
            "raw receipt lease fingerprint disagrees")
    require(receipt.get("Q38_GPU_NAME") == lease["gpu"]["name"] and
            receipt.get("Q38_GPU_DRIVER") == lease["gpu"]["driver"], "raw GPU receipt disagrees")
    require(receipt.get("Q38_RECEIPT_END") == "1", "raw discovery receipt is incomplete")
    require(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", receipt.get("Q38_TIMESTAMP_UTC", "")) is not None,
            "raw discovery timestamp missing")
    require(invocation.get("RC_EXIT_CODE") == "0", "rc discovery command did not exit zero")
    for key in ("HOST_STARTED_UTC", "HOST_FINISHED_UTC"):
        require(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", invocation.get(key, "")) is not None,
                f"missing invocation timestamp {key}")


def validate_artifacts(summary: dict, discovery: dict, command: dict, receipt: dict[str, str], repo: pathlib.Path) -> None:
    exact_keys(discovery, {
        "scan_scope", "scan_completed", "scan_device", "expected_vllm_cpp_directories",
        "tensorfold_artifact_candidates", "tensorfold_source_candidates",
        "raw_receipt", "command_manifest",
    }, "artifact-discovery")
    require(discovery["scan_completed"] is True, "artifact scan did not complete")
    require(discovery["scan_device"] == "dgx:gpu0", "artifact scan device differs from lease")
    directories = discovery["expected_vllm_cpp_directories"]
    require(isinstance(directories, list) and len(directories) == 2, "two expected vllm.cpp artifact directories are required")
    require(all(isinstance(x, dict) and set(x) == {"name", "present"} and x["present"] is False for x in directories),
            "expected vllm.cpp artifact directory was present or malformed")
    require(discovery["tensorfold_artifact_candidates"] == 0, "TensorFold artifact candidate count must be zero")
    require(discovery["tensorfold_source_candidates"] == 0, "TensorFold source candidate count must be zero")
    require(discovery["raw_receipt"] == "discovery-receipt.stdout" and
            discovery["command_manifest"] == "discovery-command.json", "artifact receipt names differ")
    exact_keys(command, {"schema_version", "script", "script_sha256", "remote_command", "device", "max_runtime", "idle_timeout", "receipt", "invocation_receipt", "scope"}, "discovery-command")
    require(command["schema_version"] == 1 and command["device"] == "dgx:gpu0", "command manifest identity differs")
    require(command["script"] == "tools/bench/qwen38_tensorfold_discovery.sh", "unexpected discovery script")
    script = repo / command["script"]
    require(SHA256.fullmatch(command["script_sha256"]) is not None and sha256_file(script) == command["script_sha256"],
            "committed discovery script hash differs")
    require(receipt.get("Q38_SCRIPT_SHA256") == command["script_sha256"], "executed script hash differs")
    require(receipt.get("Q38_SEARCH_ROOT") == command["scope"]["search_root_label"] and
            int(receipt.get("Q38_SEARCH_MAXDEPTH", "-1")) == command["scope"]["search_maxdepth"],
            "raw bounded search scope differs from command manifest")
    marker_by_label = {
        "VLLM_CPP_CKPT_A": "Q38_STAGING_VLLM_CPP_CKPT_A_PRESENT",
        "VLLM_CPP_CKPT_B": "Q38_STAGING_VLLM_CPP_CKPT_B_PRESENT",
    }
    for item in directories:
        require(receipt.get(marker_by_label[item["name"]]) == "0", f"raw receipt says {item['name']} is present")
    raw_tf_artifacts = sum(int(receipt.get(key, "-1")) for key in
                           ("Q38_SEARCH_VONTRA_DIR_COUNT", "Q38_SEARCH_MLX_MTP_FILE_COUNT", "Q38_SEARCH_FLASH_NEXT_MTP_FILE_COUNT"))
    raw_tf_sources = sum(int(receipt.get(key, "-1")) for key in
                         ("Q38_SEARCH_TENSORFOLD_DIR_COUNT", "Q38_SEARCH_MIAAI_DIR_COUNT"))
    require(raw_tf_artifacts == discovery["tensorfold_artifact_candidates"] == 0,
            "artifact count is not derived from raw receipt")
    require(raw_tf_sources == discovery["tensorfold_source_candidates"] == 0,
            "source count is not derived from raw receipt")
    assertions = summary.get("scan_assertions")
    require(assertions == {
        "workspace_tensorfold_artifact_candidates": discovery["tensorfold_artifact_candidates"],
        "workspace_tensorfold_source_candidates": discovery["tensorfold_source_candidates"],
        "workspace_vllm_cpp_artifact_directories_present": sum(bool(x["present"]) for x in directories),
    }, "summary scan assertions disagree with artifact-discovery")


def validate_checks(summary: dict, checks: dict) -> None:
    exact_keys(checks, {
        "approved_runner_source_sha", "focused_tests", "correctness_gate",
        "timing_gate", "profile_gate", "reason",
    }, "checks")
    require(checks["approved_runner_source_sha"] == summary.get("source_sha"),
            "checks runner SHA disagrees with summary")
    tests = checks["focused_tests"]
    exact_keys(tests, {"command", "status", "passed", "failed"}, "focused_tests")
    require(tests["status"] == "PASSED" and isinstance(tests["passed"], int) and tests["passed"] > 0,
            "focused tests did not pass")
    require(tests["failed"] == 0, "focused tests contain failures")
    if summary.get("status") == MEASURED:
        require(checks["correctness_gate"] == "PASSED", "measured correctness gate did not pass")
        require(checks["timing_gate"] == "CAPTURED" and checks["profile_gate"] == "CAPTURED",
                "measured timing/profile gates were not captured")
    else:
        for key in ("correctness_gate", "timing_gate", "profile_gate"):
            require(checks[key] == "NOT_RUN_PREREQUISITE", f"{key} contradicts blocker")
    require(all(value == checks["timing_gate"] for value in summary.get("workloads", {}).values()),
            "workload statuses disagree with checks")
    require(all(value == checks["profile_gate"] for value in summary.get("profiles", {}).values()),
            "profile statuses disagree with checks")


def validate_mtp(summary: dict, repo: pathlib.Path) -> None:
    mtp = summary.get("mtp")
    require(isinstance(mtp, dict) and mtp.get("verdict") == MTP_BLOCKED,
            f"MTP verdict must be {MTP_BLOCKED}")
    require(mtp.get("search_terms") == SEARCH_TERMS, "MTP search terms differ from the validated set")
    manifest_rel = mtp.get("manifest")
    require(isinstance(manifest_rel, str) and manifest_rel and not pathlib.Path(manifest_rel).is_absolute(),
            "MTP manifest must be repository-relative")
    observed = inspect_manifest(repo / manifest_rel, SEARCH_TERMS)
    require(observed["declared_tensor_count"] == mtp.get("tensor_count") == 1224,
            "MTP tensor count disagrees with committed manifest")
    require(observed["parsed_tensor_count"] == 1224, "manifest entry count is not 1224")
    require(not observed["mtp_name_matches"] and mtp.get("mtp_name_matches") == 0,
            "committed GGUF manifest contains an MTP-like tensor")
    require(observed["min_block"] == 0 and observed["max_block"] == mtp.get("max_block") == 47,
            "GGUF trunk block range is not 0 through 47")


def validate(directory: pathlib.Path) -> None:
    summary = load_object(directory / "summary.json")
    lease = load_object(directory / "lease-discovery.json")
    discovery = load_object(directory / "artifact-discovery.json")
    checks = load_object(directory / "checks.json")
    command = load_object(directory / "discovery-command.json")
    receipt = parse_kv_receipt(directory / "discovery-receipt.stdout", "Q38_RECEIPT_BEGIN")
    invocation = parse_kv_receipt(directory / "discovery-invocation.txt")
    require((directory / "README.md").is_file(), "missing raw blocker evidence: README.md")
    require(summary.get("schema_version") == 1, "schema_version must be 1")
    require(summary.get("benchmark_id") == BENCHMARK_ID, "wrong benchmark_id")
    status = summary.get("status")
    require(status in {BLOCKED, MEASURED}, f"status must be {BLOCKED} or {MEASURED}")
    require(re.fullmatch(r"[0-9a-f]{40}", str(summary.get("source_sha", ""))) is not None,
            "source_sha must be full 40-hex")
    repo = pathlib.Path(__file__).resolve().parents[2]
    validate_leases(summary, lease, directory, receipt, invocation)
    validate_checks(summary, checks)

    if status == MEASURED:
        require(summary.get("numbers_published") is True, "measured evidence must publish numbers")
        require(summary.get("cross_engine_ratio") is None, "unlike-artifact evidence must not contain a ratio")
        require((directory / "comparison.json").is_file(), "measured evidence lacks comparison.json")
        for engine in ENGINES:
            item = summary.get("artifacts", {}).get(engine, {})
            require(item.get("present") is True and item.get("hash_manifest_present") is True,
                    f"measured {engine} requires present, hashed artifacts")
            for name in ("result.json", "clocks-summary.json", "provenance.json", "profile.json"):
                require((directory / engine / name).is_file(), f"measured evidence missing {engine}/{name}")
        require(all(x == "CAPTURED" for x in summary.get("workloads", {}).values()), "measured workloads incomplete")
        require(all(x == "CAPTURED" for x in summary.get("profiles", {}).values()), "measured profiles incomplete")
        return

    validate_artifacts(summary, discovery, command, receipt, repo)
    require(summary.get("numbers_published") is False, "blocked evidence must publish no numbers")
    require(summary.get("cross_engine_ratio") is None, "blocked evidence must not contain a ratio")
    artifacts = summary.get("artifacts")
    require(isinstance(artifacts, dict) and set(artifacts) == ENGINES, "both engine artifacts must be recorded")
    for engine in ENGINES:
        item = artifacts[engine]
        require(item == {"required": True, "present": False, "hash_manifest_present": False},
                f"{engine} artifact state contradicts blocker")
    denominator = summary.get("production_vllm_denominator")
    require(isinstance(denominator, dict) and denominator.get("status") == "NOT_RUN",
            "production vLLM must remain a named NOT_RUN denominator")
    require(set(summary.get("workloads", {})) == WORKLOADS, "all workload dispositions are required")
    require(set(summary.get("profiles", {})) == ENGINES, "both profile dispositions are required")
    validate_mtp(summary, repo)
    require("blocked_later_tasks" not in summary, "obsolete blocked_later_tasks is forbidden")
    require(summary.get("task_outcomes") == TASK_OUTCOMES, "task outcomes disagree with execution plan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_dir", type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        validate(args.evidence_dir.resolve())
    except EvidenceError as exc:
        print(f"qwen38 tensorfold evidence: FAIL: {exc}", file=sys.stderr)
        return 2
    print(f"qwen38 tensorfold evidence: PASS ({load_object(args.evidence_dir.resolve() / 'summary.json')['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
