import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools/bench/run_qwen38_tensorfold_gap.sh"
TF_PIN = "191188075bca56a7c71074a79375eb4c1cb22e1c"
RECIPE_PIN = "856bb6be4b58ce6a6727e6d071fb1c52f3f80e6e"


def sh(command, cwd=None):
    return subprocess.run(command, cwd=cwd or ROOT, text=True, capture_output=True)


def make_repo(path: Path, revision: str | None = None):
    path.mkdir(parents=True)
    sh(["git", "init", "-q"], path)
    sh(["git", "config", "user.email", "fixture@example.invalid"], path)
    sh(["git", "config", "user.name", "Fixture"], path)
    (path / "tracked").write_text("clean\n")
    sh(["git", "add", "tracked"], path)
    sh(["git", "commit", "-qm", "fixture"], path)
    actual = sh(["git", "rev-parse", "HEAD"], path).stdout.strip()
    return revision or actual


def fixture(tmp_path: Path):
    source = tmp_path / "source"
    revision = make_repo(source)
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"measured fixture")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "artifacts.sha256"
    manifest.write_text(f"{digest}  {artifact.name}\n")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable:       999999999 kB\n")
    clock = tmp_path / "clock.py"
    clock.write_text("# fixture clock sampler\n")
    env = tmp_path / "arm.env"
    env.write_text(
        f"ENGINE=vllm-cpp\nSOURCE_DIR={source}\nEXPECTED_REVISION={revision}\n"
        f"ARTIFACT_MANIFEST={manifest}\nENDPOINT=http://127.0.0.1:9/v1/completions\n"
        "ENDPOINT_READY_URL=http://127.0.0.1:9/health\nMODEL=fixture\n"
        "TOKENIZER_IDENTITY=fixture@revision\nLAUNCH_COMMAND='printf launched'\n"
        "START_COMMAND='false'\nSTOP_COMMAND='true'\nSTATUS_COMMAND='false'\n"
        f"SERVER_LOG_PATH={tmp_path / 'server.log'}\nREDACTION_COMMAND=cat\n"
        "MIN_FREE_MEMORY_KIB=1\n"
    )
    rc = tmp_path / "rc"
    rc.write_text(
        "#!/bin/sh\nset -eu\n"
        "if test -n \"${RC_CALLS_FILE:-}\"; then n=$(cat \"$RC_CALLS_FILE\" 2>/dev/null || echo 0); n=$((n+1)); echo $n >\"$RC_CALLS_FILE\"; "
        "if test -n \"${RC_REVOKE_AFTER:-}\" && test $n -gt \"$RC_REVOKE_AFTER\"; then echo '[]'; exit 0; fi; fi\n"
        "printf '%s\\n' '[{\"id\":\"lease-1\",\"device\":\"dgx:gpu0\",\"state\":\"running\"}]'\n"
    )
    rc.chmod(0o755)
    process_env = {
        **os.environ,
        "RC_DEVICE": "dgx:gpu0",
        "RC_JOB_ID": "lease-1",
        "QWEN38_RC_COMMAND": str(rc),
        "QWEN38_MEMINFO_PATH": str(meminfo),
        "QWEN38_CLOCK_SAMPLER": str(clock),
        "QWEN38_ENDPOINT_PROBE": "true",
    }
    return env, source, manifest, process_env


def run_check(env_file, process_env):
    return subprocess.run([str(RUNNER), "check", str(env_file)], cwd=ROOT,
                          env=process_env, text=True, capture_output=True)


def test_examples_are_portable_pinned_templates_without_secrets_or_fake_hashes():
    tf = (ROOT / "benchmarks/manifests/qwen38_tensorfold/tensorfold.env.example").read_text()
    cpp = (ROOT / "benchmarks/manifests/qwen38_tensorfold/vllm_cpp.env.example").read_text()
    assert TF_PIN in tf and RECIPE_PIN in tf
    assert "EXPECTED_REVISION=" in cpp
    assert "ARTIFACT_MANIFEST=" in tf and "ARTIFACT_MANIFEST=" in cpp
    for text in (tf, cpp):
        assert "/home/" not in text and "/workspace/" not in text
        assert "RC_TOKEN=" not in text
        assert "ARTIFACT_SHA256=" not in text
        assert "ARTIFACT_ID=" not in text
        for field in ("START_COMMAND=", "STOP_COMMAND=", "STATUS_COMMAND=",
                      "SERVER_LOG_PATH=", "REDACTION_COMMAND="):
            assert field in text
        assert "RC_JOB_ID=" not in text and "RC_DEVICE=" not in text


def test_check_accepts_clean_exact_revision_and_does_not_launch(tmp_path):
    env_file, _, _, process_env = fixture(tmp_path)
    result = run_check(env_file, process_env)
    assert result.returncode == 0, result.stderr
    assert "launched" not in result.stdout
    assert "check PASS" in result.stdout


def test_default_clock_sampler_runs_as_module_without_inherited_pythonpath(tmp_path):
    env_file, _, _, process_env = fixture(tmp_path)
    process_env.pop("QWEN38_CLOCK_SAMPLER")
    process_env["PYTHONPATH"] = ""
    result = run_check(env_file, process_env)
    assert result.returncode == 0, result.stderr
    source = RUNNER.read_text()
    assert "python3 -m tools.bench.gpu_clock_state" in source
    assert "python3 -m py_compile" not in source


def test_check_requires_ready_endpoint(tmp_path):
    env_file, _, _, process_env = fixture(tmp_path)
    process_env["QWEN38_ENDPOINT_PROBE"] = "false"
    result = run_check(env_file, process_env)
    assert result.returncode != 0
    assert "not ready" in result.stderr


def test_dirty_revision_fails_closed(tmp_path):
    env_file, source, _, process_env = fixture(tmp_path)
    (source / "tracked").write_text("dirty\n")
    result = run_check(env_file, process_env)
    assert result.returncode != 0
    assert "dirty" in result.stderr.lower()


def test_wrong_revision_and_missing_artifact_hash_fail_closed(tmp_path):
    env_file, _, manifest, process_env = fixture(tmp_path)
    text = env_file.read_text().replace("EXPECTED_REVISION=", "EXPECTED_REVISION=" + "0" * 40 + " # ")
    env_file.write_text(text)
    assert run_check(env_file, process_env).returncode != 0

    env_file, _, manifest, process_env = fixture(tmp_path / "second")
    manifest.write_text("weights.bin\n")
    result = run_check(env_file, process_env)
    assert result.returncode != 0
    assert "hash" in result.stderr.lower() or "manifest" in result.stderr.lower()


def test_missing_or_fake_lease_cannot_launch(tmp_path):
    env_file, _, _, process_env = fixture(tmp_path)
    for variable in ("RC_DEVICE", "RC_JOB_ID"):
        candidate = dict(process_env)
        candidate.pop(variable)
        result = run_check(env_file, candidate)
        assert result.returncode != 0
        assert variable in result.stderr

    fake = dict(process_env)
    fake["RC_JOB_ID"] = "invented"
    marker = tmp_path / "launched"
    env_file.write_text(env_file.read_text().replace("LAUNCH_COMMAND='printf launched'", f"LAUNCH_COMMAND='touch {marker}'"))
    result = subprocess.run([str(RUNNER), "vllm-cpp", str(env_file)], cwd=ROOT,
                            env=fake, text=True, capture_output=True)
    assert result.returncode != 0
    assert not marker.exists()
    assert "active dgx:gpu0" in result.stderr


def test_capture_runs_arms_serially_tears_down_and_sanitizes_provenance(tmp_path):
    env_file, _, manifest, process_env = fixture(tmp_path)
    events = tmp_path / "events"
    state = tmp_path / "server.state"
    server_log = tmp_path / "server.log"
    lifecycle = tmp_path / "lifecycle.sh"
    lifecycle.write_text(
        "#!/bin/sh\nset -eu\na=$1; arm=$2; state=$3; events=$4; log=$5\n"
        "case $a in start) test ! -e \"$state\"; echo \"$arm:start\" >>\"$events\"; "
        "echo 'SECRET_TOKEN=hidden'; touch \"$state\"; printf 'safe\\nSECRET_TOKEN=hidden\\n' >\"$log\";; "
        "status) test -e \"$state\";; stop) echo \"$arm:stop\" >>\"$events\"; rm -f \"$state\";; esac\n"
    )
    lifecycle.chmod(0o755)
    clock = tmp_path / "clock.py"
    clock_compare = tmp_path / "clock-compare-called"
    clock.write_text(
        "import json,os,pathlib,sys\n"
        "if sys.argv[1]=='--help': raise SystemExit(0)\n"
        f"marker=pathlib.Path({str(clock_compare)!r})\n"
        "if sys.argv[1]=='compare':\n"
        " marker.write_text('called'); print(json.dumps({'reasons':['forced']})); raise SystemExit(1 if os.environ.get('FAIL_CLOCK_COMPARE') else 0)\n"
        "o=pathlib.Path(sys.argv[sys.argv.index('--output')+1]); s=pathlib.Path(sys.argv[sys.argv.index('--summary')+1])\n"
        "o.write_text('\\n'.join('{}' for _ in range(30))+'\\n')\n"
        "s.write_text(json.dumps({'sm_clock_mhz':{'n':30}}))\n"
    )
    harness = tmp_path / "harness.py"
    harness.write_text(
        "import json,pathlib,sys\n"
        "out=pathlib.Path(sys.argv[sys.argv.index('--output')+1])\n"
        "x={'refusal_reason':None,'canonical_payload_hash':'payload','run_identity':'run',"
        "'tokenizer_identity':'fixture@revision','samples':[{'refusal_reason':None,"
        "'payload_hash':'sample','prompt_tokens':1,'prompt_token_fingerprint':'same'}],"
        "'prompt_token_fingerprints':['same']}\n"
        "out.write_text(json.dumps(x))\n"
    )
    gpu = tmp_path / "gpu-state"
    gpu.write_text("#!/bin/sh\necho '0, Fake GPU, uuid, driver, 40, 10, 20, 100, 90'\n")
    gpu.chmod(0o755)

    def arm(name, engine, recipe=False):
        path = tmp_path / f"{name}.env"
        text = env_file.read_text().replace("ENGINE=vllm-cpp", f"ENGINE={engine}")
        text = text.replace("ENDPOINT=http://127.0.0.1:9/v1/completions",
                            "ENDPOINT=http://user:SECRET_TOKEN@127.0.0.1:9/v1/completions?api_key=SECRET_TOKEN#private")
        text = text.replace("START_COMMAND='false'", f"START_COMMAND='{lifecycle} start {name} {state} {events} {server_log}'")
        text = text.replace("STOP_COMMAND='true'", f"STOP_COMMAND='{lifecycle} stop {name} {state} {events} {server_log}'")
        text = text.replace("STATUS_COMMAND='false'", f"STATUS_COMMAND='{lifecycle} status {name} {state} {events} {server_log}'")
        text = text.replace(f"SERVER_LOG_PATH={tmp_path / 'server.log'}", f"SERVER_LOG_PATH={server_log}")
        text = text.replace("REDACTION_COMMAND=cat", "REDACTION_COMMAND='sed /SECRET/d'")
        if recipe:
            source = tmp_path / "recipe"
            revision = make_repo(source)
            text += f"RECIPE_SOURCE_DIR={source}\nEXPECTED_RECIPE_REVISION={revision}\n"
        path.write_text(text)
        return path

    tf = arm("tensorfold", "tensorfold", recipe=True)
    cpp = arm("vllm-cpp", "vllm-cpp")
    cpp_artifact = tmp_path / "cpp-weights.bin"
    cpp_artifact.write_bytes(b"different verified artifact")
    cpp_manifest = tmp_path / "cpp-artifacts.sha256"
    cpp_manifest.write_text(f"{hashlib.sha256(cpp_artifact.read_bytes()).hexdigest()}  {cpp_artifact.name}\n")
    cpp.write_text(cpp.read_text().replace(f"ARTIFACT_MANIFEST={manifest}",
                                           f"ARTIFACT_MANIFEST={cpp_manifest}"))
    output = tmp_path / "evidence"
    probe = tmp_path / "probe"
    probe.write_text(f"#!/bin/sh\ntest -e {state}\n")
    probe.chmod(0o755)
    process_env.update(QWEN38_CLOCK_SAMPLER=str(clock), QWEN38_HARNESS=str(harness),
                       QWEN38_GPU_STATE_COMMAND=str(gpu), QWEN38_ENDPOINT_PROBE=str(probe))
    result = subprocess.run([str(RUNNER), "capture", str(tf), str(cpp), str(output)],
                            cwd=ROOT, env=process_env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    run = next(output.iterdir())
    assert events.read_text().splitlines() == ["tensorfold:start", "tensorfold:stop",
                                               "vllm-cpp:start", "vllm-cpp:stop"]
    assert not state.exists()
    assert clock_compare.read_text() == "called"
    for name in ("tensorfold", "vllm-cpp"):
        arm_out = run / name
        assert json.loads((arm_out / "clocks-summary.json").read_text())["sm_clock_mhz"]["n"] >= 30
        assert (arm_out / "gpu-state-before.csv").is_file()
        assert (arm_out / "gpu-state-after.csv").is_file()
        assert "SECRET" not in (arm_out / "server.log").read_text()
        assert not (arm_out / f"{name}.env").exists()
    tf_provenance = json.loads((run / "tensorfold/provenance.json").read_text())
    cpp_provenance = json.loads((run / "vllm-cpp/provenance.json").read_text())
    assert tf_provenance["recipe_revision"]
    assert cpp_provenance["recipe_revision"] is None  # no state leaked from the first source
    assert tf_provenance["artifact_identity"] != cpp_provenance["artifact_identity"]
    assert tf_provenance["source_dirty"] is False
    assert tf_provenance["runtime"]["lease_fingerprint"] == cpp_provenance["runtime"]["lease_fingerprint"]
    assert len(tf_provenance["runtime"]["lease_fingerprint"]) == 64
    serialized = json.dumps([tf_provenance, cpp_provenance])
    assert "SECRET_TOKEN" not in serialized and "START_COMMAND" not in serialized
    assert "SECRET_TOKEN" not in "".join(p.read_text(errors="ignore") for p in run.rglob("*") if p.is_file())
    comparison = json.loads((run / "comparison.json").read_text())
    assert comparison["left_artifact"] == tf_provenance["artifact_identity"]
    assert comparison["right_artifact"] == cpp_provenance["artifact_identity"]
    assert comparison["verdict"] == "PROFILE_COMPARISON"
    assert "ratio" not in comparison

    failed_output = tmp_path / "failed-evidence"
    failed_env = dict(process_env, FAIL_CLOCK_COMPARE="1")
    failed = subprocess.run([str(RUNNER), "capture", str(tf), str(cpp), str(failed_output)],
                            cwd=ROOT, env=failed_env, text=True, capture_output=True)
    assert failed.returncode != 0
    assert not failed_output.exists() or not any(failed_output.iterdir())
    assert "clock gate refused" in failed.stderr

    redaction_output = tmp_path / "redaction-failed-evidence"
    raw_parent = tmp_path / "raw-tmp"
    raw_parent.mkdir()
    bad_tf = tmp_path / "bad-redactor-tf.env"
    bad_cpp = tmp_path / "bad-redactor-cpp.env"
    bad_tf.write_text(tf.read_text().replace("REDACTION_COMMAND='sed /SECRET/d'", "REDACTION_COMMAND=false"))
    bad_cpp.write_text(cpp.read_text().replace("REDACTION_COMMAND='sed /SECRET/d'", "REDACTION_COMMAND=false"))
    redaction_failed = subprocess.run(
        [str(RUNNER), "capture", str(bad_tf), str(bad_cpp), str(redaction_output)],
        cwd=ROOT, env=dict(process_env, TMPDIR=str(raw_parent)), text=True, capture_output=True,
    )
    assert redaction_failed.returncode != 0
    assert not redaction_output.exists() or not any(redaction_output.iterdir())
    assert not list(raw_parent.iterdir())
    assert "SECRET_TOKEN" not in "".join(
        p.read_text(errors="ignore") for root in (redaction_output, raw_parent)
        if root.exists() for p in root.rglob("*") if p.is_file()
    )

    revoked_output = tmp_path / "revoked-evidence"
    calls = tmp_path / "rc-calls"
    revoked = subprocess.run(
        [str(RUNNER), "capture", str(tf), str(cpp), str(revoked_output)], cwd=ROOT,
        env=dict(process_env, RC_CALLS_FILE=str(calls), RC_REVOKE_AFTER="5"),
        text=True, capture_output=True,
    )
    assert revoked.returncode != 0
    assert "active dgx:gpu0" in revoked.stderr
    assert not revoked_output.exists() or not any(revoked_output.iterdir())


def test_suspicious_env_field_is_refused_before_launch(tmp_path):
    env_file, _, _, process_env = fixture(tmp_path)
    env_file.write_text(env_file.read_text() + "API_SECRET=do-not-serialize\n")
    marker = tmp_path / "launched"
    env_file.write_text(env_file.read_text().replace("LAUNCH_COMMAND='printf launched'", f"LAUNCH_COMMAND='touch {marker}'"))
    result = subprocess.run([str(RUNNER), "vllm-cpp", str(env_file)], cwd=ROOT,
                            env=process_env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "suspicious" in result.stderr
    assert not marker.exists()


def test_data_env_rejects_shell_expansion_without_execution(tmp_path):
    env_file, _, _, process_env = fixture(tmp_path)
    marker = tmp_path / "owned"
    env_file.write_text(env_file.read_text() + f"MODEL=$(touch {marker})\n")
    result = run_check(env_file, process_env)
    assert result.returncode != 0
    assert not marker.exists()
    assert "strict data" in result.stderr


def test_capture_refuses_preexisting_server_without_stopping_it(tmp_path):
    source = RUNNER.read_text()
    refusal = 'was already running; capture will not attach or stop it'
    assert refusal in source
    assert source.index(refusal) < source.index('run_redacted "$arm_out/server-start.log"')


def test_clock_compare_failure_aborts_final_verdict(tmp_path):
    source = RUNNER.read_text()
    assert 'compare --ours "$out/tensorfold/clocks-summary.json"' in source
    assert '|| fail "cross-arm clock gate refused comparison"' in source
    assert source.index('compare --ours "$out/tensorfold/clocks-summary.json"') < source.index('compare_results "$out/tensorfold/result.json"')


def test_lease_revocation_is_rechecked_before_final_verdict(tmp_path):
    source = RUNNER.read_text()
    final = source.index('run_redacted "$out/clock-comparison.json"')
    assert source.rfind("verify_lease", 0, final) > source.index('capture_arm "$cpp"')
    assert source.index("verify_lease", final) < source.index('compare_results "$out/tensorfold/result.json"')
    assert "lease_fingerprint" in source


def test_capture_comparison_refuses_mismatched_token_fingerprints_and_emits_no_ratio(tmp_path):
    left = {
        "refusal_reason": None, "canonical_payload_hash": "payload", "run_identity": "run",
        "tokenizer_identity": "tok", "samples": [{"refusal_reason": None,
        "payload_hash": "sample", "prompt_tokens": 1, "prompt_token_fingerprint": "left"}],
        "prompt_token_fingerprints": ["left"],
    }
    right = json.loads(json.dumps(left))
    right["samples"][0]["prompt_token_fingerprint"] = "right"
    right["prompt_token_fingerprints"] = ["right"]
    left_path, right_path = tmp_path / "left.json", tmp_path / "right.json"
    left_path.write_text(json.dumps(left)); right_path.write_text(json.dumps(right))
    out = tmp_path / "comparison.json"
    result = subprocess.run([str(RUNNER), "compare-results", str(left_path), str(right_path),
                             "artifact-a", "artifact-b", str(out)], cwd=ROOT,
                            text=True, capture_output=True)
    assert result.returncode != 0
    comparison = json.loads(out.read_text())
    assert comparison["verdict"] == "REFUSED"
    assert "ratio" not in comparison


def test_unlike_manifest_identity_forces_profile_comparison_without_ratio(tmp_path):
    sample = {
        "refusal_reason": None, "canonical_payload_hash": "payload", "run_identity": "run",
        "tokenizer_identity": "tok", "samples": [{"refusal_reason": None,
        "payload_hash": "sample", "prompt_tokens": 1, "prompt_token_fingerprint": "same"}],
        "prompt_token_fingerprints": ["same"],
    }
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text(json.dumps(sample)); right.write_text(json.dumps(sample))
    out = tmp_path / "comparison.json"
    result = subprocess.run([str(RUNNER), "compare-results", str(left), str(right),
                             "manifest-derived-a", "manifest-derived-b", str(out)],
                            cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    comparison = json.loads(out.read_text())
    assert comparison["verdict"] == "PROFILE_COMPARISON"
    assert "ratio" not in comparison
    assert "manifest" in comparison["no_ratio_reason"]


def test_capture_source_has_timestamped_evidence_and_harness_contract():
    source = RUNNER.read_text()
    for command in ("check", "tensorfold", "vllm-cpp", "capture"):
        assert command in source
    assert "qwen38_endpoint_bench.py" in source
    assert "date -u +%Y%m%dT%H%M%SZ" in source
    assert "comparison_verdict" in source
    assert "PROFILE_COMPARISON" in source
