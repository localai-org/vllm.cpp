#!/usr/bin/env bash
# TT-STAGING-FOOTPRINT gate on the P150: the row-page repack.
# Phase A: Q4_K_M byte identity — one leg vs the recorded pre-change stream
#          (docs/bench-evidence/qwen38-q6k-vs-q4km-20261113/q4km-*-token-ids.json,
#          sha256 d49c53d2..., recorded at de3493a3742, a code-identical
#          keepquant base).
# Phase B: UD-Q6_K_XL serves on one P150 — 3 reps, deterministic token IDs,
#          KV pool 16 blocks (8 is the fallback), TPOT recorded.
# Recipe = stage-B gate: batch 1 / input 32 / output 200, seed 0,
# --num-blocks 16 --max-num-batched-tokens 64, mutex + luwen reset + shm
# cleanup + idle gate + governor/env/ps-top logged per leg.
set -u
EV="$(cd "$(dirname "$0")" && pwd)"
BENCH=/tmp/build-ttfp/examples/vllm-bench
export LD_LIBRARY_PATH="$HOME/Sources/tt/tt-metal-pin-durable/lib64:/tmp/umdtrial-install/lib64:/tmp/umdtrial-install/libexec/tt-metalium"
export TT_METAL_RUNTIME_ROOT="$HOME/Sources/tt/tt-metal-pin"
export TT_METAL_HOME="$HOME/Sources/tt/tt-metal-pin"

log_env() {
  {
    echo "date: $(date -Is)"
    echo "head: $(git -C /tmp/tt-staging rev-parse HEAD)"
    echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
    echo "load1: $(cut -d' ' -f1 /proc/loadavg)"
    echo "bench: $BENCH"
    echo "model: $MODEL"
    echo "model sha256: $(sha256sum "$MODEL" | cut -d' ' -f1)"
    echo "num-blocks: $BLOCKS"
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
  taskset -c 32-63 "$BENCH" \
    --model "$MODEL" \
    --num-prompts 1 --input-len 32 --output-len 200 --concurrency 1 \
    --num-blocks "$BLOCKS" --max-num-batched-tokens 64 --seed 0 \
    --output-token-ids "$EV/$name-token-ids.json" \
    > "$EV/$name.log" 2> "$EV/$name.stderr.log"
  echo "=== $name end $(date -Is) exit=$? ==="
}

if [ "${1:-}" = "q4km" ]; then
  MODEL=/mnt/models/unsloth-qwen3.8-27B-gguf/Qwen3.8-27B-Q4_K_M.gguf
  BLOCKS=8
  leg q4km 1
else
  MODEL=/mnt/models/unsloth-qwen3.8-27B-gguf-v2/Qwen3.8-27B-UD-Q6_K_XL.gguf
  BLOCKS=${2:-16}
  for r in 1 2 3; do leg q6kl "$r"; done
fi
echo "ALL LEGS DONE $(date -Is)"
