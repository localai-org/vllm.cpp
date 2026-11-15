#!/usr/bin/env bash
# QWEN38-DFLASH2-STAGE-C W3 device gate: the speculative loop's greedy token
# identity + TPOT, drafter-ON vs OFF, on the Q4_K_M KV-64 baseline config.
# Spec: .agents/specs/qwen38-dflash2-stage-c.md, Waves W3/W4.
# Discipline: file mutex, luwen reset + 15 s + /dev/shm cleanup per leg,
# governor performance logged per leg, idle gate (1-min load < 16) per leg,
# VT_TT_KEEPQUANT_INT8DOT=1 on every leg, bench zone taskset -c 32-63,
# builds zone taskset -c 64-127.
set -u
EV="$(cd "$(dirname "$0")" && pwd)"
BENCH=/tmp/build-stagec-w3/examples/vllm-bench
MODEL=/mnt/models/unsloth-qwen3.8-27B-gguf/Qwen3.8-27B-Q4_K_M.gguf
DRAFT=/mnt/models/qwen38-dflash2-drafter
export LD_LIBRARY_PATH="$HOME/Sources/tt/tt-metal-pin-durable/lib64:/tmp/umdtrial-install/lib64:/tmp/umdtrial-install/libexec/tt-metalium"
export TT_METAL_RUNTIME_ROOT="$HOME/Sources/tt/tt-metal-pin"
export TT_METAL_HOME="$HOME/Sources/tt/tt-metal-pin"
export VT_TT_KEEPQUANT_INT8DOT=1

log_env() {  # $1 = leg name
  {
    echo "date: $(date -Is)"
    echo "head: $(git -C /tmp/q38-stage-c rev-parse HEAD)"
    echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
    echo "load1: $(cut -d' ' -f1 /proc/loadavg)"
    echo "bench: $BENCH"
    echo "model sha256: $(sha256sum "$MODEL" | cut -d' ' -f1)"
  } > "$EV/$1-env.txt"
  ps aux --sort=-%cpu | head -12 > "$EV/ps-top-$1.txt"
}

leg() {  # $1 = arm name, $2 = rep, extra env via LEG_ENV, spec config via SPEC
  local name="$1-$2"
  echo "=== $name start $(date -Is) ==="
  exec 9>"${GPU_LOCK:-$HOME/gpu.lock}"; flock -x 9
  "$HOME/Sources/tt/luwen/target/release/reset" >/dev/null 2>&1; sleep 15
  rm -f /dev/shm/TT_UMD_LOCK.* /dev/shm/tt_device_*_memory 2>/dev/null
  # idle gate: wait until 1-min load < 16
  while [ "$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)" -ge 16 ]; do sleep 60; done
  log_env "$name"
  env $LEG_ENV taskset -c 32-63 "$BENCH" \
    --model "$MODEL" \
    ${SPEC:+--speculative-config "$SPEC"} \
    --num-prompts 1 --input-len 32 --output-len 200 --concurrency 1 \
    --num-blocks 64 --max-num-batched-tokens 64 --seed 0 \
    --output-token-ids "$EV/$name-token-ids.json" \
    > "$EV/$name.log" 2> "$EV/$name.stderr.log"
  echo "=== $name end $(date -Is) exit=$? ==="
}

case "${1:-all}" in
  on)     LEG_ENV="VT_DFLASH_DRAFT=1" SPEC='{"method":"dflash","model":"'"$DRAFT"'","num_speculative_tokens":8}'; leg on 1; leg on 2; leg on 3 ;;
  off)    LEG_ENV="" ; SPEC=""; leg off 1; leg off 2; leg off 3 ;;
  probe)  LEG_ENV="VT_DFLASH_DRAFT=1" SPEC='{"method":"dflash","model":"'"$DRAFT"'","num_speculative_tokens":8}'; leg probe 1 ;;
  *)      leg on 1; leg on 2; leg on 3; leg off 1; leg off 2; leg off 3 ;;
esac
