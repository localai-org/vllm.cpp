#!/usr/bin/env bash
# QWEN38-DFLASH2-STAGE-C W5 — the drafter-ON degeneracy bisect (issue
# ISSUE-LOCAL-01M7FY5W1A91XHV1PZDGZEKR7H): the no-bfp8 A/B.
#
# Suspect 1 is the bfp8 weight-staging lever (VT_TT_WEIGHT_RESIDENCY=bfp8,
# drafter-scoped). The W4 no-bfp8 A/B was capacity-blocked at KV 32 (the
# draft's bf16 trunk OOM'd at the 178257920 B ffn_down, ~149 KB/bank short,
# fragmentation-bound — w4-nobfp8-probe.*). This leg bisects the KV pool down
# (KV 24) so the bf16 trunk fits, and runs the identical workload WITHOUT
# bfp8: if the stream is clean ([220, 16, 23066, ...]) the bfp8 staging is
# the culprit; if it still degenerates ([220, 0, 0, ...]) the input staging
# is. VT_TT_GDN_SPEC_DEBUG=1 keeps the conv/gdn sum trail comparable to
# w4-diag-spec.
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

leg() {  # $1 tag, $2 blocks, $3 extra env, $4 output-len, $5 spec(1/0)
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
    echo "output-len: $4"
    echo "spec: $5"
  } > "$EV/$name-env.txt"
  ps aux --sort=-%cpu | head -12 > "$EV/ps-top-$name.txt"
  local spec_args="" env_extra="$3"
  [ "$5" = "1" ] && spec_args="--speculative-config $SPEC"
  ( cd "$REPO" && env $env_extra taskset -c 32-63 "$BENCH" \
      --model "$Q4KM" $spec_args \
      --num-prompts 1 --input-len 32 --output-len "$4" --concurrency 1 \
      --num-blocks "$2" --max-num-batched-tokens 64 --seed 0 \
      --output-token-ids "$EV/$name-token-ids.json" \
      > "$EV/$name.log" 2> "$EV/$name.stderr.log" )
  local rc=$?
  echo "=== $name end $(date -Is) exit=$rc sha256=$(sha256sum "$EV/$name-token-ids.json" 2>/dev/null | cut -d' ' -f1) ==="
}

case "${1:-all}" in
  # THE A/B: Q4_K_M drafter-ON, NO bfp8 (VT_TT_WEIGHT_RESIDENCY unset),
  # KV pool bisected down to 24 blocks so the bf16 trunk fits.
  ab-nobfp8)  leg w5-ab-nobfp8-kv24 24 "VLLM_CPP_CUDAGRAPH=0 VT_DFLASH_DRAFT=1 VT_TT_GDN_SPEC_DEBUG=1" 24 1 ;;
  # The bfp8 twin at the SAME KV 24 (isolates the KV change from the lever):
  ab-bfp8)    leg w5-ab-bfp8-kv24 24 "VLLM_CPP_CUDAGRAPH=0 VT_DFLASH_DRAFT=1 VT_TT_WEIGHT_RESIDENCY=bfp8 VT_TT_GDN_SPEC_DEBUG=1" 24 1 ;;
  all)        $0 ab-nobfp8; $0 ab-bfp8 ;;
  *) echo "usage: $0 ab-nobfp8|ab-bfp8|all" ;;
esac
