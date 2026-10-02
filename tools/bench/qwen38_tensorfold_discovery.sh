#!/usr/bin/env bash
# Bounded, non-secret discovery receipt for BENCH-QWEN38-TENSORFOLD-GAP.
set -euo pipefail

sha256_file() { sha256sum "$1" | awk '{print $1}'; }
count_find() {
  local root=$1 depth=$2 kind=$3 pattern=$4
  if [[ ! -d "$root" ]]; then printf '0'; return; fi
  find "$root" -maxdepth "$depth" -type "$kind" -iname "$pattern" -printf '.' 2>/dev/null | wc -c
}

[[ "${RC_DEVICE:-}" == "dgx:gpu0" ]] || { echo 'Q38_ERROR=WRONG_DEVICE'; exit 2; }
[[ -n "${RC_JOB_ID:-}" ]] || { echo 'Q38_ERROR=MISSING_LEASE_ID'; exit 2; }

printf 'Q38_RECEIPT_BEGIN=1\n'
printf 'Q38_SCHEMA=1\n'
printf 'Q38_TIMESTAMP_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'Q38_DEVICE=%s\n' "$RC_DEVICE"
printf 'Q38_LEASE_SHA256=%s\n' "$(printf '%s' "$RC_JOB_ID" | sha256sum | awk '{print $1}')"
printf 'Q38_SCRIPT_SHA256=%s\n' "$(sha256_file "$0")"
printf 'Q38_GPU_NAME=%s\n' "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d '\r')"
printf 'Q38_GPU_DRIVER=%s\n' "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d '\r')"

# Exact documented staging paths. Emit labels only, never host paths.
for spec in \
  'VLLM_CPP_CKPT_A:/workspace/ckpt/qwen4exp-flash-next-iq1s' \
  'VLLM_CPP_CKPT_B:/workspace/q4exp-bench/UD-IQ1_S'
do
  label=${spec%%:*}; path=${spec#*:}
  [[ -d "$path" ]] && present=1 || present=0
  printf 'Q38_STAGING_%s_PRESENT=%s\n' "$label" "$present"
done

# Bounded /workspace searches. Counts prove only this root/depth/pattern set.
printf 'Q38_SEARCH_ROOT=workspace\n'
printf 'Q38_SEARCH_MAXDEPTH=3\n'
printf 'Q38_SEARCH_TENSORFOLD_DIR_COUNT=%s\n' "$(count_find /workspace 3 d '*tensorfold*')"
printf 'Q38_SEARCH_MIAAI_DIR_COUNT=%s\n' "$(count_find /workspace 3 d '*miaai*')"
printf 'Q38_SEARCH_VONTRA_DIR_COUNT=%s\n' "$(count_find /workspace 3 d '*vontra*')"
printf 'Q38_SEARCH_MLX_MTP_FILE_COUNT=%s\n' "$(count_find /workspace 3 f '*mlx*mtp*')"
printf 'Q38_SEARCH_FLASH_NEXT_MTP_FILE_COUNT=%s\n' "$(count_find /workspace 3 f '*flash-next*mtp*')"
printf 'Q38_RECEIPT_END=1\n'
