#!/usr/bin/env bash
# QWEN38-DFLASH2-STAGE-C W22 — PROBE 3 (the gated norm's own write) + PROBE 4
# (the out_proj's own write). The W21 record's remaining poison: the GDN
# block's output garbage at l >= 1 ([MIXER-OUT] asum 2.00072e+40) with CLEAN
# gated-norm operands ([RNG-IN] x and g both clean) — the norm's commit and
# the out_proj's commit are the two unprobed writes between them. Instruments
# (all env-gated, default-off): [NORM-PRECOMMIT] + [TRUST-SLOT] norm-commit-N
# + the norm*_*.f32 dumps (the TT RmsNormGated kernel), [OUTPROJ-PRECOMMIT] +
# [TRUST-SLOT] outproj-commit-N + the outproj*_*.f32 dumps (MatmulBTKernel at
# the spec out_proj geometry M=9 K=6144 N=5120).
# Discipline per leg: file mutex ${GPU_LOCK:-$HOME/gpu.lock}, luwen reset +
# 15 s + /dev/shm cleanup, idle gate (1-min load < 16), governor/env/ps-top
# logged, bench zone taskset -c 32-63, run FROM THE REPO ROOT (kernel-JIT
# resolves ./src/vt/tenstorrent/kernels), INT8DOT=1, durable-libs-first,
# TT_METAL_RUNTIME_ROOT at the pin. Drafter-ON: VLLM_CPP_CUDAGRAPH=0
# VT_TT_WEIGHT_RESIDENCY=bfp8 VT_DFLASH_DRAFT=1.
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
export VT_DUMP_TRUST=1
export VT_TT_SPEC_DUMP="$EV/w22-probe"
export VT_TT_GDN_SPEC_DEBUG=1

name=w22-probe
echo "=== $name start $(date -Is) ==="
exec 9>"${GPU_LOCK:-$HOME/gpu.lock}"; flock -x 9
"$HOME/Sources/tt/luwen/target/release/reset" >/dev/null 2>&1; sleep 15
rm -f /dev/shm/TT_UMD_LOCK.* /dev/shm/tt_device_*_memory 2>/dev/null
while [ "$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)" -ge 16 ]; do sleep 60; done
{
  echo "date: $(date -Is)"
  echo "head: $(git -C "$REPO" rev-parse HEAD)"
  echo "worktree: dirty (W22 probes 3/4 instrumentation, uncommitted)"
  echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
  echo "load1: $(cut -d' ' -f1 /proc/loadavg)"
  echo "bench: $BENCH"
  echo "model: $Q4KM"
  echo "model sha256: $(sha256sum "$Q4KM" | cut -d' ' -f1)"
  echo "num-blocks: 64"
  echo "extra-env: VLLM_CPP_CUDAGRAPH=0 VT_TT_WEIGHT_RESIDENCY=bfp8 VT_DFLASH_DRAFT=1 VT_DUMP_TRUST=1 VT_TT_SPEC_DUMP=$EV/w22-probe VT_TT_GDN_SPEC_DEBUG=1"
  echo "output-len: 24"
  echo "spec: 1"
} > "$EV/$name-env.txt"
ps aux --sort=-%cpu | head -12 > "$EV/ps-top-$name.txt"
( cd "$REPO" && env VLLM_CPP_CUDAGRAPH=0 VT_TT_WEIGHT_RESIDENCY=bfp8 VT_DFLASH_DRAFT=1 \
    taskset -c 32-63 "$BENCH" \
    --model "$Q4KM" --speculative-config "$SPEC" \
    --num-prompts 1 --input-len 32 --output-len 24 --concurrency 1 \
    --num-blocks 64 --max-num-batched-tokens 64 --seed 0 \
    --output-token-ids "$EV/$name-token-ids.json" \
    > "$EV/$name.log" 2> "$EV/$name.stderr.log" )
rc=$?
echo "=== $name end $(date -Is) exit=$rc sha256=$(sha256sum "$EV/$name-token-ids.json" 2>/dev/null | cut -d' ' -f1) ==="
