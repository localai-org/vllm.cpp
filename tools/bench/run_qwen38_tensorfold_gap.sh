#!/usr/bin/env bash
# Pinned Qwen3.8 TensorFold/vllm.cpp launcher and serial evidence capture.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HARNESS=${QWEN38_HARNESS:-$ROOT/tools/bench/qwen38_endpoint_bench.py}
CORPUS="$ROOT/benchmarks/manifests/qwen38_tensorfold/corpus.json"
CLOCK_SAMPLER=${QWEN38_CLOCK_SAMPLER:-$ROOT/tools/bench/gpu_clock_state.py}
DEFAULT_CLOCK_SAMPLER=$ROOT/tools/bench/gpu_clock_state.py
MEMINFO=${QWEN38_MEMINFO_PATH:-/proc/meminfo}
RC_COMMAND=${QWEN38_RC_COMMAND:-rc}
GPU_STATE_COMMAND=${QWEN38_GPU_STATE_COMMAND:-nvidia-smi --query-gpu=index,name,uuid,driver_version,temperature.gpu,power.draw,power.limit,memory.total,memory.free --format=csv,noheader,nounits}
CONFIG_VARS=(ENGINE SOURCE_DIR EXPECTED_REVISION RECIPE_SOURCE_DIR EXPECTED_RECIPE_REVISION ARTIFACT_MANIFEST ENDPOINT ENDPOINT_READY_URL MODEL TOKENIZER_IDENTITY LAUNCH_COMMAND START_COMMAND STOP_COMMAND STATUS_COMMAND SERVER_LOG_PATH REDACTION_COMMAND MIN_FREE_MEMORY_KIB DRAFT CONCURRENCY WAVES)
fail() { echo "qwen38-tensorfold-gap: $*" >&2; exit 2; }
require_var() { test -n "${!1:-}" || fail "$1 is required"; }

reset_config() { local v; for v in "${CONFIG_VARS[@]}"; do unset "$v"; done; }
load_env() {
  local path=$1 name value i parsed_file
  local -a parsed=()
  test -f "$path" || fail "environment file does not exist: $path"
  reset_config
  parsed_file=$(mktemp); trap 'rm -f "$parsed_file"' RETURN
  if ! python3 - "$path" "${CONFIG_VARS[@]}" >"$parsed_file" <<'PY'
import pathlib,re,sys
path=pathlib.Path(sys.argv[1]); allowed=set(sys.argv[2:]); seen=set()
for number,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
    if not line.strip() or line.lstrip().startswith("#"): continue
    match=re.fullmatch(r"([A-Z][A-Z0-9_]*)=(.*)",line)
    if not match: raise SystemExit(f"invalid data-only assignment at line {number}")
    key,value=match.groups()
    if key not in allowed: raise SystemExit(f"unsupported or suspicious field: {key}")
    if key in seen: raise SystemExit(f"duplicate field: {key}")
    seen.add(key)
    if len(value)>=2 and value[0]==value[-1] and value[0] in "\"'": value=value[1:-1]
    if any(c in value for c in ("$","`","\x00","\r","\n")):
        raise SystemExit(f"expansion or control character forbidden in {key}")
    sys.stdout.buffer.write(key.encode()+b"\0"+value.encode()+b"\0")
PY
  then
    fail "environment file is not strict data: $path"
  fi
  mapfile -d '' -t parsed <"$parsed_file"
  rm -f "$parsed_file"; trap - RETURN
  test $((${#parsed[@]} % 2)) -eq 0 || fail "environment parser returned malformed data"
  for ((i=0; i<${#parsed[@]}; i+=2)); do
    name=${parsed[i]}; value=${parsed[i+1]}; printf -v "$name" '%s' "$value"
    # shellcheck disable=SC2163
    export "$name"
  done
}
check_revision() {
  local d=$1 expected=$2 label=$3 actual
  git -C "$d" rev-parse --git-dir >/dev/null 2>&1 || fail "$label source is not a git checkout: $d"
  [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || fail "$label expected revision must be full 40-hex"
  actual=$(git -C "$d" rev-parse HEAD); test "$actual" = "$expected" || fail "$label revision mismatch"
  test -z "$(git -C "$d" status --porcelain=v1 --untracked-files=normal)" || fail "$label revision is dirty"
}
check_artifacts() {
  local manifest=$1 base line digest relative actual count=0
  test -s "$manifest" || fail "artifact hash manifest is missing or empty: $manifest"
  base=$(cd "$(dirname "$manifest")" && pwd)
  while IFS= read -r line || test -n "$line"; do
    test -z "$line" && continue
    [[ "$line" =~ ^([0-9a-f]{64})[[:space:]][[:space:]]([^/].*)$ ]] || fail "artifact manifest entry lacks a measured sha256: $line"
    digest=${BASH_REMATCH[1]}; relative=${BASH_REMATCH[2]}
    [[ "$relative" != *".."* ]] || fail "artifact path escapes manifest directory"
    test -f "$base/$relative" || fail "artifact is missing: $relative"
    actual=$(sha256sum "$base/$relative" | cut -d' ' -f1); test "$actual" = "$digest" || fail "artifact hash mismatch: $relative"
    count=$((count+1))
  done < "$manifest"
  test "$count" -gt 0 || fail "artifact manifest contains no hashes"
}
artifact_identity() {
  local manifest=$1 base
  base=$(cd "$(dirname "$manifest")" && pwd)
  while read -r digest relative; do printf '%s  %s\n' "$digest" "$relative"; done < "$manifest" | LC_ALL=C sort | sha256sum | cut -d' ' -f1
}
verify_lease() {
  require_var RC_DEVICE; require_var RC_JOB_ID
  test "$RC_DEVICE" = dgx:gpu0 || fail "campaign requires RC_DEVICE=dgx:gpu0"
  command -v "$RC_COMMAND" >/dev/null || fail "repository rc helper is unavailable"
  "$RC_COMMAND" jobs --device dgx:gpu0 --limit 100 --output json | python3 -c '
import json,os,sys
try: x=json.load(sys.stdin)
except Exception: raise SystemExit(2)
rows=x if isinstance(x,list) else x.get("jobs",x.get("items",[]))
j=os.environ["RC_JOB_ID"]
ok=any(str(r.get("id",r.get("job_id","")))==j and r.get("device",r.get("device_id"))=="dgx:gpu0" and str(r.get("state",r.get("status",""))).lower() in {"running","held","active"} for r in rows)
raise SystemExit(0 if ok else 2)' || fail "RC_JOB_ID is not an active dgx:gpu0 repository lease"
}
clock_tool() {
  if test "$CLOCK_SAMPLER" = "$DEFAULT_CLOCK_SAMPLER"; then
    (cd "$ROOT" && PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -m tools.bench.gpu_clock_state "$@")
  else
    python3 "$CLOCK_SAMPLER" "$@"
  fi
}
clock_smoke() { clock_tool --help >/dev/null 2>&1 || fail "clock sampler runtime smoke check failed"; }
lease_fingerprint() { printf '%s\0%s' "$RC_DEVICE" "$RC_JOB_ID" | sha256sum | cut -d' ' -f1; }
mem_available() { awk '/^MemAvailable:/{print $2; exit}' "$MEMINFO"; }
probe_ready() { if test -n "${QWEN38_ENDPOINT_PROBE:-}"; then "$QWEN38_ENDPOINT_PROBE" "$ENDPOINT_READY_URL" >/dev/null 2>&1; else curl --fail --silent --max-time 5 "$ENDPOINT_READY_URL" >/dev/null; fi; }
common_gate() {
  local env_file=$1 policy=$2 available
  load_env "$env_file"
  for v in ENGINE SOURCE_DIR EXPECTED_REVISION ARTIFACT_MANIFEST ENDPOINT ENDPOINT_READY_URL MODEL TOKENIZER_IDENTITY MIN_FREE_MEMORY_KIB; do require_var "$v"; done
  case "$ENGINE" in tensorfold|vllm-cpp) ;; *) fail "invalid ENGINE";; esac
  check_revision "$SOURCE_DIR" "$EXPECTED_REVISION" "$ENGINE"
  if test -n "${RECIPE_SOURCE_DIR:-}${EXPECTED_RECIPE_REVISION:-}"; then require_var RECIPE_SOURCE_DIR; require_var EXPECTED_RECIPE_REVISION; check_revision "$RECIPE_SOURCE_DIR" "$EXPECTED_RECIPE_REVISION" recipe; fi
  check_artifacts "$ARTIFACT_MANIFEST"; verify_lease
  test -r "$MEMINFO" || fail "memory sampler unavailable"
  available=$(mem_available); [[ "$available" =~ ^[0-9]+$ && "$MIN_FREE_MEMORY_KIB" =~ ^[0-9]+$ ]] || fail "invalid free-memory gate"
  test "$available" -ge "$MIN_FREE_MEMORY_KIB" || fail "free unified memory below baseline"
  test -r "$CLOCK_SAMPLER" || fail "clock sampler unavailable"; clock_smoke
  case "$policy" in down) ! probe_ready || fail "endpoint already active before launch";; ready) probe_ready || fail "endpoint is not ready";; either) :;; esac
}
check_one() { common_gate "$1" ready; echo "qwen38-tensorfold-gap: check PASS ($ENGINE)"; }
launch_one() { local wanted=$1 file=$2; common_gate "$file" down; test "$ENGINE" = "$wanted" || fail "wrong engine"; require_var LAUNCH_COMMAND; exec bash -c "$LAUNCH_COMMAND"; }
run_harness() {
  local output=$1 raw status
  raw=$(mktemp "$RAW_DIR/harness.XXXXXX")
  set +e
  python3 "$HARNESS" --endpoint "$ENDPOINT" --model "$MODEL" --corpus "$CORPUS" --output "$raw" --tokenizer-identity "$TOKENIZER_IDENTITY" --adapter "$ENGINE" --draft "${DRAFT:-off}" --concurrency "${CONCURRENCY:-1}" --waves "${WAVES:-5}"
  status=$?
  set -e
  test -f "$raw" && redact_file "$raw" "$output"
  return "$status"
}
write_provenance() {
  local output=$1 identity lease; identity=$(artifact_identity "$ARTIFACT_MANIFEST"); lease=$(lease_fingerprint)
  python3 - "$output" "$ENGINE" "$EXPECTED_REVISION" "${EXPECTED_RECIPE_REVISION:-}" "$identity" "$ENDPOINT" "$MODEL" "$TOKENIZER_IDENTITY" "$ARTIFACT_MANIFEST" "$lease" <<'PY'
import json,os,pathlib,re,sys,urllib.parse
out,engine,rev,recipe,artifact,endpoint,model,tok,manifest,lease=sys.argv[1:]
def safe(label,value):
 if any(ord(c)<32 or ord(c)==127 for c in value): raise SystemExit(f"control character in {label}")
 if re.search(r"(?i)(password|secret|token|api[_-]?key|bearer|credential)",value): raise SystemExit(f"credential-like value in {label}")
 return value
u=urllib.parse.urlsplit(endpoint)
endpoint=urllib.parse.urlunsplit((u.scheme,u.hostname + ((":"+str(u.port)) if u.port else ""),u.path,"",""))
entries=[]; base=pathlib.Path(manifest).parent
for line in pathlib.Path(manifest).read_text().splitlines():
 d,p=line.split(None,1); p=p.strip(); q=base/p; s=q.stat(); entries.append({"name":safe("artifact name",pathlib.Path(p).name),"sha256":d,"size":s.st_size,"mtime_ns":s.st_mtime_ns})
data={"engine":engine,"revision":rev,"recipe_revision":recipe or None,"artifact_identity":artifact,"artifacts":sorted(entries,key=lambda x:(x["name"],x["sha256"])),"endpoint":safe("endpoint",endpoint),"model":safe("model",model),"tokenizer_identity":safe("tokenizer identity",tok),"source_dirty":False,"runtime":{"rc_device":os.environ["RC_DEVICE"],"lease_fingerprint":lease},"server_log":"server.log"}
pathlib.Path(out).write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
PY
}
redact_file() {
  local input=$1 output=$2 remove=${3:-yes} staged
  staged=$(mktemp "$RAW_DIR/redacted.XXXXXX")
  if ! bash -c "$REDACTION_COMMAND" <"$input" >"$staged"; then
    rm -f "$staged"
    return 2
  fi
  mv "$staged" "$output"
  test "$remove" = no || rm -f "$input"
}
run_redacted() {
  local output=$1 status tmp
  tmp=$(mktemp "$RAW_DIR/subprocess.XXXXXX")
  shift
  set +e; "$@" >"$tmp" 2>&1; status=$?; set -e
  redact_file "$tmp" "$output"
  return "$status"
}
wait_ready() {
  local i
  for ((i=0;i<60;i++)); do
    if bash -c "$STATUS_COMMAND" >/dev/null 2>&1 && probe_ready; then return 0; fi
    sleep 1
  done
  return 1
}
capture_arm() (
  local file=$1 arm_out=$2 before after sampler="" sampler_status started=0 raw_samples raw_summary raw_sampler
  # Invoked indirectly by trap.
  # shellcheck disable=SC2329
  cleanup_arm() {
    local status=$?
    if test -n "$sampler"; then kill -INT "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true; fi
    if test "$started" = 1; then run_redacted "$arm_out/server-stop.log" bash -c "$STOP_COMMAND" || status=2; fi
    if bash -c "$STATUS_COMMAND" >/dev/null 2>&1; then echo "endpoint remains active after teardown" >&2; status=2; fi
    exit "$status"
  }
  common_gate "$file" down
  for v in START_COMMAND STOP_COMMAND STATUS_COMMAND SERVER_LOG_PATH REDACTION_COMMAND; do require_var "$v"; done
  before=$(mem_available); printf '%s\n' "$before" >"$arm_out/memory-before-kib.txt"
  if bash -c "$STATUS_COMMAND" >/dev/null 2>&1; then fail "$ENGINE was already running; capture will not attach or stop it"; fi
  started=1; trap cleanup_arm EXIT INT TERM
  run_redacted "$arm_out/server-start.log" bash -c "$START_COMMAND" || fail "$ENGINE start command failed"
  wait_ready || fail "$ENGINE failed to become ready"
  run_redacted "$arm_out/gpu-state-before.csv" bash -c "$GPU_STATE_COMMAND" || fail "GPU runtime identity probe failed"
  raw_samples=$(mktemp "$RAW_DIR/clocks.XXXXXX")
  raw_summary=$(mktemp "$RAW_DIR/clock-summary.XXXXXX")
  raw_sampler=$(mktemp "$RAW_DIR/clock-log.XXXXXX")
  clock_tool sample --output "$raw_samples" --summary "$raw_summary" --interval 1 --max-duration 3600 >"$raw_sampler" 2>&1 & sampler=$!
  run_harness "$arm_out/result.json"
  kill -INT "$sampler" 2>/dev/null || true
  set +e; wait "$sampler"; sampler_status=$?; set -e; sampler=""
  redact_file "$raw_sampler" "$arm_out/clock-sampler.log" || fail "clock sampler log redaction failed"
  redact_file "$raw_samples" "$arm_out/clocks.jsonl" || fail "clock sample redaction failed"
  redact_file "$raw_summary" "$arm_out/clocks-summary.json" || fail "clock summary redaction failed"
  test "$sampler_status" -eq 0 || fail "clock sampler refused window"
  python3 - "$arm_out/clocks-summary.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); n=x.get("sm_clock_mhz",{}).get("n",0)
raise SystemExit(0 if n>=30 else 2)
PY
  run_redacted "$arm_out/gpu-state-after.csv" bash -c "$GPU_STATE_COMMAND" || fail "GPU power/thermal probe failed"
  redact_file "$SERVER_LOG_PATH" "$arm_out/server.log" no
  write_provenance "$arm_out/provenance.json"
  run_redacted "$arm_out/server-stop.log" bash -c "$STOP_COMMAND" || fail "$ENGINE stop command failed"; started=0
  ! bash -c "$STATUS_COMMAND" >/dev/null 2>&1 || fail "$ENGINE teardown failed"
  after=$(mem_available); printf '%s\n' "$after" >"$arm_out/memory-after-kib.txt"
  test "$after" -ge "$MIN_FREE_MEMORY_KIB" || fail "free unified memory did not return to baseline"
  verify_lease
  printf '%s\n' "$(lease_fingerprint)" >"$arm_out/lease-fingerprint.txt"
  trap - EXIT INT TERM
)
compare_results() {
  python3 - "$ROOT" "$1" "$2" "$3" "$4" "$5" <<'PY'
import json,pathlib,sys
sys.path.insert(0,sys.argv[1]); from tools.bench.qwen38_endpoint_bench import comparison_verdict
out=comparison_verdict(json.loads(pathlib.Path(sys.argv[2]).read_text()),json.loads(pathlib.Path(sys.argv[3]).read_text()))
out.update(left_artifact=sys.argv[4],right_artifact=sys.argv[5]); out.pop("ratio",None)
if sys.argv[4]!=sys.argv[5] and out["verdict"]!="REFUSED": out.update(verdict="PROFILE_COMPARISON",no_ratio_reason="verified artifact manifests differ; cross-engine results are absolute only")
pathlib.Path(sys.argv[6]).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); raise SystemExit(2 if out["verdict"]=="REFUSED" else 0)
PY
}
capture() (
  local tf=$1 cpp=$2 base=${3:-$ROOT/.agents/evidence/bench-qwen38-tensorfold-gap} stamp out ta ca expected_lease status=0
  RAW_DIR=$(mktemp -d "${TMPDIR:-/tmp}/qwen38-capture.XXXXXX"); chmod 700 "$RAW_DIR"; export RAW_DIR
  cleanup_capture() {
    local rc=$?
    find "$RAW_DIR" -type f -exec sh -c 'command -v shred >/dev/null 2>&1 && shred -u "$1" || rm -f "$1"' _ {} \; 2>/dev/null || true
    rm -rf "$RAW_DIR"
    if test "$rc" -ne 0 && test -n "${out:-}"; then rm -rf "$out"; fi
    exit "$rc"
  }
  # shellcheck disable=SC2329
  trap cleanup_capture EXIT INT TERM
  verify_lease; expected_lease=$(lease_fingerprint)
  stamp=$(date -u +%Y%m%dT%H%M%SZ); out="$base/$stamp"; test ! -e "$out" || fail "evidence exists"; mkdir -p "$out/tensorfold" "$out/vllm-cpp"
  capture_arm "$tf" "$out/tensorfold"; test "$(cat "$out/tensorfold/lease-fingerprint.txt")" = "$expected_lease" || fail "lease changed after tensorfold arm"
  load_env "$tf"; ta=$(artifact_identity "$ARTIFACT_MANIFEST")
  capture_arm "$cpp" "$out/vllm-cpp"; test "$(cat "$out/vllm-cpp/lease-fingerprint.txt")" = "$expected_lease" || fail "lease changed after vllm-cpp arm"
  load_env "$cpp"; ca=$(artifact_identity "$ARTIFACT_MANIFEST")
  verify_lease; test "$(lease_fingerprint)" = "$expected_lease" || fail "lease changed before final verdict"
  run_redacted "$out/clock-comparison.json" clock_tool compare --ours "$out/tensorfold/clocks-summary.json" --vllm "$out/vllm-cpp/clocks-summary.json" || fail "cross-arm clock gate refused comparison"
  verify_lease; test "$(lease_fingerprint)" = "$expected_lease" || fail "lease changed before final verdict"
  compare_results "$out/tensorfold/result.json" "$out/vllm-cpp/result.json" "$ta" "$ca" "$out/comparison.json"; printf '%s\n' "$out"
  trap - EXIT INT TERM; cleanup_capture
)
usage(){ echo "usage: $0 {check|tensorfold|vllm-cpp|capture} ..."; }
command=${1:-}; shift || true
case "$command" in
 check) test $# = 1 || fail "check requires ENV_FILE"; check_one "$1";;
 tensorfold|vllm-cpp) test $# = 1 || fail "$command requires ENV_FILE"; launch_one "$command" "$1";;
 capture) test $# -ge 2 -a $# -le 3 || fail "capture requires two env files"; capture "$@";;
 compare-results) test $# = 5 || fail "compare-results requires five arguments"; compare_results "$@";;
 *) usage >&2; exit 2;;
esac
