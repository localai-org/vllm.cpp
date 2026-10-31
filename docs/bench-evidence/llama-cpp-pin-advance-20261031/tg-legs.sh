#!/usr/bin/env bash
# Idle-gated llama.cpp b11765 tg legs on bench zone 32-63 (row LLAMA-CPP-PIN-ADVANCE).
set -u
BIN=/tmp/build-lc-pin/bin
OUT=/tmp/lc-pin-advance/docs/bench-evidence/llama-cpp-pin-advance-20261031
export LD_LIBRARY_PATH=$BIN
cd "$BIN"

echo "== governor $(cat /sys/devices/system/cpu/cpu32/cpufreq/scaling_governor) $(date -u +%FT%TZ)"

wait_idle() {
  while :; do
    load=$(cut -d. -f1 /proc/loadavg)
    foreign=$(pgrep -c -f 'taskset -c 3[2-6]' || true)
    if [ "$load" -lt 16 ] && [ "${foreign:-0}" -eq 0 ]; then break; fi
    sleep 20
  done
}

run_leg() { # name model threads
  wait_idle
  echo "== $NAME t=$t $(date -u +%FT%TZ) load-before=$(cut -d. -f1 /proc/loadavg)"
  taskset -c 32-63 ./llama-bench -m "$2" -t "$3" -p 0 -n 128 -r 3 -ha 0 2>&1
  echo "== load-after=$(cut -d. -f1 /proc/loadavg)"
}

wait_idle
echo "== I-Micro t16"
echo "== load-after=$(cut -d. -f1 /proc/loadavg)"
wait_idle
echo "== I-Micro t32"; taskset -c 32-63 ./llama-bench -m /mnt/models/kolibri1-apex-gguf/Kolibri-1-APEX-I-Micro.gguf -t 32 -p 0 -n 128 -r 3 2>&1
echo "== load-after=$(cut -d. -f1 /proc/loadavg)"
wait_idle
echo "== qwen38-q4km t16"; taskset -c 32-63 ./llama-bench -m /mnt/models/unsloth-qwen3.8-27B-gguf/Qwen3.8-27B-Q4_K_M.gguf -t 16 -p 0 -n 64 -r 2 2>&1
echo "== load-after=$(cut -d. -f1 /proc/loadavg)"
echo "== done $(date -u +%FT%TZ)"
