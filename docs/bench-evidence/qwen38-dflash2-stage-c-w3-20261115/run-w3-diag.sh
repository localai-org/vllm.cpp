#!/usr/bin/env bash
# W3 diagnostic probes: (1) drafter-ON direct arm + staging trace, (2) drafter-OFF.
# Same discipline as run-w3-finish-gate.sh.
set -u
EV="$(cd "$(dirname "$0")" && pwd)"
BENCH=/tmp/build-stagec-w3/examples/vllm-bench
REPO=/tmp/q38-stage-c
Q4KM=/mnt/models/unsloth-qwen3.8-27B-gguf/Qwen3.8-27B-Q4_K_M.gguf
DRAFT=/mnt/models/qwen38-dflash2-drafter
SPEC='{"method":"dflash","model":"'"$DRAFT"'","num_speculative_tokens":8}'
export LD_LIBRARY_PATH="$HOME/Sources/tt/tt-metal-pin-durable/lib64:/tmp/umdtrial-install/lib64:/tmp/umdtrial-install/libexec/tt-metalium"
export TT_METAL_RUNTIME_ROOT="$HOME/Sources/tt/tt-metal-pin"
export TT_METAL_HOME="$HOME/Sources/tt/tt-metal-pin"
export VT_TT_KEEPQUANT_INT8DOT=1 INT8DOT=1

leg() {  # $1 tag, $2 blocks, $3 extra env, $4 spec(1/0)
  local name="$1"
  echo "=== $name start $(date -Is) ==="
  exec 9>"${GPU_LOCK:-$HOME/gpu.lock}"; flock -x 9
  "$HOME/Sources/tt/luwen/target/release/reset" >/dev/null 2>&1; sleep 15
  rm -f /dev/shm/TT_UMD_LOCK.* /dev/shm/tt_device_*_memory 2>/dev/null
  while [ "$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)" -ge 16 ]; do sleep 60; done
  {
    echo "date: $(date -Is)"
    echo "head: $(git -C "$REPO" rev-parse HEAD)"
    echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
    echo "load1: $(cut -d' ' -f1 /proc/loadavg)"
    echo "bench: $BENCH"
    echo "model: $Q4KM"
    echo "model sha256: $(sha256sum "$Q4KM" | cut -d' ' -f1)"
    echo "num-blocks: $2"
    echo "extra-env: $3"
    echo "spec: $4"
  } > "$EV/$name-env.txt"
  ps aux --sort=-%cpu | head -12 > "$EV/ps-top-$name.txt"
  local spec_args="" env_extra="$3"
  [ "$4" = "1" ] && spec_args="--speculative-config $SPEC"
  ( cd "$REPO" && env $env_extra taskset -c 32-63 "$BENCH" \
      --model "$Q4KM" $spec_args \
      --num-prompts 1 --input-len 32 --output-len 24 --concurrency 1 \
      --num-blocks "$2" --max-num-batched-tokens 64 --seed 0 \
      --output-token-ids "$EV/$name-token-ids.json" \
      > "$EV/$name.log" 2> "$EV/$name.stderr.log" )
  local rc=$?
  echo "=== $name end $(date -Is) exit=$rc sha256=$(sha256sum "$EV/$name-token-ids.json" 2>/dev/null | cut -d' ' -f1) ==="
}

case "${1:-all}" in
  on-direct) leg w3f-diag-on-direct 64 "VLLM_CPP_CUDAGRAPH=0 VT_DFLASH_DRAFT=1 VT_TT_WEIGHT_RESIDENCY=bfp8 VT_STAGING_TRACE=1 VT_TT_STEP_PHASES=1" 1 ;;
  off)       leg w3f-diag-off 64 "VT_STAGING_TRACE=1 VT_TT_STEP_PHASES=1" 0 ;;
  all)       $0 on-direct; $0 off ;;
esac
