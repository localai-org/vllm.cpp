#!/bin/bash
# QWEN38-P150-REBENCH: one measured leg (governor, rep) — the 09-14 recipe at current head.
# Usage: leg.sh <governor> <rep>
set -u
GOV="$1"; REP="$2"
EV=/tmp/qwen38-rebench/docs/bench-evidence/qwen38-p150-rebench-20261102

exec 9>$HOME/gpu.lock
flock -x 9

echo "== leg $GOV rep$REP start $(date -Is) ==" | tee -a "$EV/governor-log.txt"
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do cat "$c"; done \
  | sort | uniq -c | tee -a "$EV/governor-log.txt"

# idle gate: 1-min load < 16
while true; do L=$(awk '{print $1}' /proc/loadavg); echo "load1=$L $(date -Is)" >> "$EV/idle-gate.txt"; [ "$(echo "$L < 16" | bc)" = 1 ] && break; sleep 60; done
ps aux --sort=-%cpu | head -8 > "$EV/ps-top-$GOV-$REP.txt"

# device reset
~/Sources/tt/luwen/target/release/reset
sleep 15
rm -f /dev/shm/TT_UMD_LOCK.* /dev/shm/tt_device_*_memory

# pin governor
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo "$GOV" | sudo tee "$c" >/dev/null; done

source ~/Sources/tt/env-tt-common.sh
export TT_METAL_HOME=/home/lu_zero/Sources/tt/tt-metal-pin
export TT_METAL_RUNTIME_ROOT=/tmp/umdtrial-install/libexec/tt-metalium
export LD_LIBRARY_PATH=/tmp/umdtrial-install/lib64:/tmp/umdtrial-install/libexec/tt-metalium
export VT_TT_KEEPQUANT_INT8DOT=1

taskset -c 32-63 /tmp/build-q38rebench/examples/vllm-bench \
  --model /mnt/models/unsloth-qwen3.8-27B-gguf/Qwen3.8-27B-Q4_K_M.gguf \
  --num-prompts 2 --input-len 128 --output-len 32 --concurrency 2 \
  --num-blocks 64 --max-num-batched-tokens 64 --seed 0 \
  > "$EV/$GOV-r$REP.log" 2>&1
echo "RC=$? leg $GOV rep$REP end $(date -Is)" | tee -a "$EV/governor-log.txt"
