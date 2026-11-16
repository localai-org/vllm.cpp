#!/usr/bin/env bash
# QWEN38-DFLASH2-STAGE-C W21 — the conv-spec zero-row probe + the recurrence's
# unprobed inputs. The W20 dump forensics showed the conv-spec OUTPUT carries
# exact-zero token rows (tokens 5,6,7) at EVERY spec call while the served x
# reads clean — this leg dumps the scatter-chain stages (x_served, filled,
# statG, acc, y, out) and the recurrence inputs (S_all, g, beta, q, k, v, o)
# as raw f32 at the first spec calls, for the composed CPU reference.
#
# Discipline per leg: file mutex ${GPU_LOCK:-$HOME/gpu.lock}, luwen reset +
# 15 s + /dev/shm cleanup, idle gate (1-min load < 16), governor/env/ps-top
# logged, bench zone taskset -c 32-63, run FROM THE REPO ROOT (kernel-JIT
# resolves ./src/vt/tenstorrent/kernels), INT8DOT=1, durable-libs-first,
# TT_METAL_RUNTIME_ROOT at the pin.
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

name=w21-ptr
echo "=== $name start $(date -Is) ==="
exec 9>"${GPU_LOCK:-$HOME/gpu.lock}"; flock -x 9
"$HOME/Sources/tt/luwen/target/release/reset" >/dev/null 2>&1; sleep 15
rm -f /dev/shm/TT_UMD_LOCK.* /dev/shm/tt_device_*_memory 2>/dev/null
while [ "$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)" -ge 16 ]; do sleep 60; done
{
  echo "date: $(date -Is)"
  echo "head: $(git -C "$REPO" rev-parse HEAD)"
  echo "worktree: dirty (W21 spec-dump probes, uncommitted)"
  echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
  echo "load1: $(cut -d' ' -f1 /proc/loadavg)"
  echo "bench: $BENCH"
  echo "model: $Q4KM"
  echo "model sha256: $(sha256sum "$Q4KM" | cut -d' ' -f1)"
  echo "num-blocks: 64"
  echo "extra-env: VLLM_CPP_CUDAGRAPH=0 VT_TT_WEIGHT_RESIDENCY=bfp8 VT_DFLASH_DRAFT=1 VT_TT_GDN_SPEC_DEBUG=1 VT_TT_SPEC_DUMP=$EV/w21-ptr"
  echo "output-len: 24"
  echo "spec: 1"
} > "$EV/$name-env.txt"
ps aux --sort=-%cpu | head -12 > "$EV/ps-top-$name.txt"
( cd "$REPO" && env VLLM_CPP_CUDAGRAPH=0 VT_TT_WEIGHT_RESIDENCY=bfp8 VT_DFLASH_DRAFT=1 VT_TT_GDN_SPEC_DEBUG=1 \
    VT_TT_SPEC_DUMP="$EV/w21-ptr" \
    taskset -c 32-63 "$BENCH" \
    --model "$Q4KM" --speculative-config "$SPEC" \
    --num-prompts 1 --input-len 32 --output-len 24 --concurrency 1 \
    --num-blocks 64 --max-num-batched-tokens 64 --seed 0 \
    --output-token-ids "$EV/$name-token-ids.json" \
    > "$EV/$name.log" 2> "$EV/$name.stderr.log" )
rc=$?
echo "=== $name end $(date -Is) exit=$rc sha256=$(sha256sum "$EV/$name-token-ids.json" 2>/dev/null | cut -d' ' -f1) ==="
