#!/usr/bin/env bash
# Q6_K_XL vs Q4_K_M benchmark on the P150, same binary (build at main).
# Recipe = stage-B gate: batch 1 / input 32 / output 200, seed 0,
# --num-blocks 8 --max-num-batched-tokens 64, INT8DOT=1, mutex + luwen
# reset + shm cleanup + idle gate + governor/env/ps-top logged per leg.
set -u
EV="$(cd "$(dirname "$0")" && pwd)"
BENCH=/tmp/build-q6k/examples/vllm-bench
export LD_LIBRARY_PATH="$HOME/Sources/tt/tt-metal-pin-durable/lib64:/tmp/umdtrial-install/lib64:/tmp/umdtrial-install/libexec/tt-metalium"
export TT_METAL_RUNTIME_ROOT="$HOME/Sources/tt/tt-metal-pin"
export TT_METAL_HOME="$HOME/Sources/tt/tt-metal-pin"
export VT_TT_KEEPQUANT_INT8DOT=1

log_env() {
  {
    echo "date: $(date -Is)"
    echo "head: $(git -C /tmp/bench-q6k rev-parse HEAD)"
    echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
    echo "load1: $(cut -d' ' -f1 /proc/loadavg)"
    echo "bench: $BENCH"
    echo "model: $MODEL"
    echo "model sha256: $(sha256sum "$MODEL" | cut -d' ' -f1)"
  } > "$EV/$1-env.txt"
  ps aux --sort=-%cpu | head -12 > "$EV/ps-top-$1.txt"
}

leg() {  # $1 = arm tag, $2 = rep
  local name="$1-$2"
  echo "=== $name start $(date -Is) ==="
  exec 9>"${GPU_LOCK:-$HOME/gpu.lock}"; flock -x 9
  "$HOME/Sources/tt/luwen/target/release/reset" >/dev/null 2>&1; sleep 15
  rm -f /dev/shm/TT_UMD_LOCK.* /dev/shm/tt_device_*_memory 2>/dev/null
  while [ "$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)" -ge 16 ]; do sleep 60; done
  log_env "$name"
  env $LEG_ENV taskset -c 32-63 "$BENCH" \
    --model "$MODEL" \
    --num-prompts 1 --input-len 32 --output-len 200 --concurrency 1 \
    --num-blocks 8 --max-num-batched-tokens 64 --seed 0 \
    --output-token-ids "$EV/$name-token-ids.json" \
    > "$EV/$name.log" 2> "$EV/$name.stderr.log"
  echo "=== $name end $(date -Is) exit=$? ==="
}

MODEL=/mnt/models/unsloth-qwen3.8-27B-gguf-v2/Qwen3.8-27B-UD-Q6_K_XL.gguf
LEG_ENV="" ; leg q6kl 1; leg q6kl 2; leg q6kl 3
echo "ALL LEGS DONE $(date -Is)"