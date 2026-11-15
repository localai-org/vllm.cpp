#!/usr/bin/env bash
# QWEN38-DFLASH2-STAGE-C W4 gate: the kGdnSpecDecode TT kernel wave — the
# device gates on the W3-restacked base + the GDN spec kernel.
# Spec: .agents/specs/qwen38-dflash2-stage-c.md (Waves W3/W4); recipe per the
# committed run-gate.sh / run-w3-finish-gate.sh.
#
# Discipline per leg: file mutex ${GPU_LOCK:-$HOME/gpu.lock}, luwen reset +
# 15 s + /dev/shm cleanup, idle gate (1-min load < 16), governor/env/ps-top
# logged, bench zone taskset -c 32-63 (builds 64-127), run FROM THE REPO ROOT
# (kernel-JIT resolves ./src/vt/tenstorrent/kernels), INT8DOT=1 on every leg.
#
# The drafter-ON token-identity arms run VLLM_CPP_CUDAGRAPH=0: the captured
# spec-verify arm is refused by name (tenstorrent_gdn.cpp — the acceptance
# metadata has no persistent device refresh; a captured replay would advance
# every later verify by a stale prefix). The eager arm is the one the W3
# evidence names for the token-identity gate. VT_TT_WEIGHT_RESIDENCY=bfp8 on
# every DRAFTER-ON leg (the W3 measured memory lever: the draft trunk stages
# ~1.55 GiB BFP8_B and the wall closes; the target's packed keep-quant
# weights refuse bfp8 by name and fall back, so the target's numerics are
# bit-identical and the lossless verify keeps the emitted tokens the target's
# greedy stream — only acceptance/TPOT can move). VT_DFLASH_DRAFT=1 restored
# on every drafter-ON leg (the direct-argmax arm; the selector never runs).
set -u
EV="$(cd "$(dirname "$0")" && pwd)"
BENCH=/tmp/build-stagec-w3/examples/vllm-bench
REPO=/tmp/q38-stage-c
Q4KM=/mnt/models/unsloth-qwen3.8-27B-gguf/Qwen3.8-27B-Q4_K_M.gguf
Q6KL=/mnt/models/unsloth-qwen3.8-27B-gguf-v2/Qwen3.8-27B-UD-Q6_K_XL.gguf
DRAFT=/mnt/models/qwen38-dflash2-drafter
SPEC='{"method":"dflash","model":"'"$DRAFT"'","num_speculative_tokens":8}'
export LD_LIBRARY_PATH="$HOME/Sources/tt/tt-metal-pin-durable/lib64:/tmp/umdtrial-install/lib64:/tmp/umdtrial-install/libexec/tt-metalium"
export TT_METAL_RUNTIME_ROOT="$HOME/Sources/tt/tt-metal-pin"
export TT_METAL_HOME="$HOME/Sources/tt/tt-metal-pin"
export VT_TT_KEEPQUANT_INT8DOT=1 INT8DOT=1

leg() {  # $1 tag, $2 model, $3 blocks, $4 extra env, $5 output-len, $6 spec(1/0)
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
    echo "model: $2"
    echo "model sha256: $(sha256sum "$2" | cut -d' ' -f1)"
    echo "num-blocks: $3"
    echo "extra-env: $4"
    echo "output-len: $5"
    echo "spec: $6"
  } > "$EV/$name-env.txt"
  ps aux --sort=-%cpu | head -12 > "$EV/ps-top-$name.txt"
  local spec_args="" env_extra="$4"
  [ "$6" = "1" ] && spec_args="--speculative-config $SPEC"
  ( cd "$REPO" && env $env_extra taskset -c 32-63 "$BENCH" \
      --model "$2" $spec_args \
      --num-prompts 1 --input-len 32 --output-len "$5" --concurrency 1 \
      --num-blocks "$3" --max-num-batched-tokens 64 --seed 0 \
      --output-token-ids "$EV/$name-token-ids.json" \
      > "$EV/$name.log" 2> "$EV/$name.stderr.log" )
  local rc=$?
  echo "=== $name end $(date -Is) exit=$rc sha256=$(sha256sum "$EV/$name-token-ids.json" 2>/dev/null | cut -d' ' -f1) ==="
}

case "${1:-all}" in
  # cheap end-to-end loop check + 24-token prefix identity (eager, drafter ON)
  probe-on)   leg w4-on-probe "$Q4KM" 64 "VLLM_CPP_CUDAGRAPH=0 VT_TT_WEIGHT_RESIDENCY=bfp8 VT_DFLASH_DRAFT=1" 24 1 ;;
  # gate (a): Q4_K_M KV 64 drafter-ON, 3 reps, identity vs d49c53d2...
  on)         leg w4-on-1 "$Q4KM" 64 "VLLM_CPP_CUDAGRAPH=0 VT_TT_WEIGHT_RESIDENCY=bfp8 VT_DFLASH_DRAFT=1" 200 1
              leg w4-on-2 "$Q4KM" 64 "VLLM_CPP_CUDAGRAPH=0 VT_TT_WEIGHT_RESIDENCY=bfp8 VT_DFLASH_DRAFT=1" 200 1
              leg w4-on-3 "$Q4KM" 64 "VLLM_CPP_CUDAGRAPH=0 VT_TT_WEIGHT_RESIDENCY=bfp8 VT_DFLASH_DRAFT=1" 200 1 ;;
  # gate (a)/(c): Q4_K_M drafter-OFF, knob default (captured), 3 reps
  off)        leg w4-off-1 "$Q4KM" 64 "" 200 0
              leg w4-off-2 "$Q4KM" 64 "" 200 0
              leg w4-off-3 "$Q4KM" 64 "" 200 0 ;;
  # gate (b): UD-Q6_K_XL drafter-ON KV 8, 3 reps (identity vs b36bc61a...)
  q6kl-on)    leg w4-q6kl-on-1 "$Q6KL" 8 "VLLM_CPP_CUDAGRAPH=0 VT_TT_WEIGHT_RESIDENCY=bfp8 VT_DFLASH_DRAFT=1" 200 1
              leg w4-q6kl-on-2 "$Q6KL" 8 "VLLM_CPP_CUDAGRAPH=0 VT_TT_WEIGHT_RESIDENCY=bfp8 VT_DFLASH_DRAFT=1" 200 1
              leg w4-q6kl-on-3 "$Q6KL" 8 "VLLM_CPP_CUDAGRAPH=0 VT_TT_WEIGHT_RESIDENCY=bfp8 VT_DFLASH_DRAFT=1" 200 1 ;;
  # gate (c): UD-Q6_K_XL drafter-OFF, knob default (captured), 3 reps
  q6kl-off)   leg w4-q6kl-off-1 "$Q6KL" 8 "" 200 0
              leg w4-q6kl-off-2 "$Q6KL" 8 "" 200 0
              leg w4-q6kl-off-3 "$Q6KL" 8 "" 200 0 ;;
  all)        $0 probe-on; $0 on; $0 off; $0 q6kl-on; $0 q6kl-off ;;
  *) echo "usage: $0 probe-on|on|off|q6kl-on|q6kl-off|all" ;;
esac
