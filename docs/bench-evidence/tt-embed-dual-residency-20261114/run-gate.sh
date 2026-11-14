#!/usr/bin/env bash
# W3 unlock gate for TT-EMBED-DUAL-RESIDENCY on the P150.
# Standing recipe: batch 1 / input 32 / output 200, seed 0,
# --max-num-batched-tokens 64, INT8DOT=1, mutex + luwen reset + shm
# cleanup + idle gate + governor/env/ps-top logged per leg.
set -u
EV=/tmp/tt-embed/docs/bench-evidence/tt-embed-dual-residency-20261114
cd /tmp/tt-embed   # kernel-JIT resolves './src/vt/tenstorrent/kernels' from the repo root
mkdir -p "$EV"
BENCH=/tmp/build-embed/examples/vllm-bench
export LD_LIBRARY_PATH="$HOME/Sources/tt/tt-metal-pin-durable/lib64:/tmp/umdtrial-install/lib64:/tmp/umdtrial-install/libexec/tt-metalium"
export TT_METAL_RUNTIME_ROOT="$HOME/Sources/tt/tt-metal-pin"
export TT_METAL_HOME="$HOME/Sources/tt/tt-metal-pin"
export VT_TT_KEEPQUANT_INT8DOT=1
export INT8DOT=1

leg() {  # $1 tag, $2 rep, $3 model, $4 blocks, $5 extra env, $6 alloc-trace(0/1)
  local name="$1-$2"
  echo "=== $name start $(date -Is) ==="
  exec 9>"${GPU_LOCK:-$HOME/gpu.lock}"; flock -x 9
  "$HOME/Sources/tt/luwen/target/release/reset" >/dev/null 2>&1; sleep 15
  rm -f /dev/shm/TT_UMD_LOCK.* /dev/shm/tt_device_*_memory 2>/dev/null
  while [ "$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)" -ge 16 ]; do sleep 60; done
  {
    echo "date: $(date -Is)"
    echo "head: $(git -C /tmp/tt-embed rev-parse HEAD)"
    echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
    echo "load1: $(cut -d' ' -f1 /proc/loadavg)"
    echo "bench: $BENCH"
    echo "model: $3"
    echo "model sha256: $(sha256sum "$3" | cut -d' ' -f1)"
    echo "num-blocks: $4"
    echo "extra-env: $5"
    echo "alloc-trace: $6"
  } > "$EV/$name-env.txt"
  ps aux --sort=-%cpu | head -12 > "$EV/ps-top-$name.txt"
  local trace=""
  [ "$6" = "1" ] && trace="VT_TT_ALLOC_TRACE=1"
  env $trace $5 taskset -c 32-63 "$BENCH" \
    --model "$3" \
    --num-prompts 1 --input-len 32 --output-len 200 --concurrency 1 \
    --num-blocks "$4" --max-num-batched-tokens 64 --seed 0 \
    --output-token-ids "$EV/$name-token-ids.json" \
    > "$EV/$name.log" 2> "$EV/$name.stderr.log"
  local rc=$?
  echo "=== $name end $(date -Is) exit=$rc ==="
}

case "${1:-all}" in
  q6kl-captured)
    leg q6kl-cap 1 /mnt/models/unsloth-qwen3.8-27B-gguf-v2/Qwen3.8-27B-UD-Q6_K_XL.gguf 8 "" 1
    leg q6kl-cap 2 /mnt/models/unsloth-qwen3.8-27B-gguf-v2/Qwen3.8-27B-UD-Q6_K_XL.gguf 8 "" 0
    leg q6kl-cap 3 /mnt/models/unsloth-qwen3.8-27B-gguf-v2/Qwen3.8-27B-UD-Q6_K_XL.gguf 8 "" 0 ;;
  q6kl-eager)
    leg q6kl-eager 1 /mnt/models/unsloth-qwen3.8-27B-gguf-v2/Qwen3.8-27B-UD-Q6_K_XL.gguf 8 "VT_TT_DECODE_CAPTURE=0" 0 ;;
  q6kl-kv16)
    leg q6kl-cap16 1 /mnt/models/unsloth-qwen3.8-27B-gguf-v2/Qwen3.8-27B-UD-Q6_K_XL.gguf 16 "" 1 ;;
  q8kl)
    leg q8kl-cap 1 /mnt/models/unsloth-qwen3.8-27B-gguf-v2/Qwen3.8-27B-UD-Q8_K_XL.gguf 8 "" 1
    leg q8kl-cap 2 /mnt/models/unsloth-qwen3.8-27B-gguf-v2/Qwen3.8-27B-UD-Q8_K_XL.gguf 8 "" 0
    leg q8kl-cap 3 /mnt/models/unsloth-qwen3.8-27B-gguf-v2/Qwen3.8-27B-UD-Q8_K_XL.gguf 8 "" 0 ;;
  q4km)
    leg q4km-cap 1 /mnt/models/unsloth-qwen3.8-27B-gguf/Qwen3.8-27B-Q4_K_M.gguf 64 "" 0 ;;
  *) echo "usage: $0 q6kl-captured|q6kl-eager|q6kl-kv16|q8kl|q4km" ;;
esac
