ID: ISSUE-LOCAL-01M7ADZAKNZTC45QPEE1WGW2DV
Title: TT keepquant: the Q8_K weights arm — serve UD-Q8_K_XL-class GGUF checkpoints on the P150
Row: TT-KEEPQUANT-Q8K-WEIGHTS
State: CLOSED
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-13
Updated: 2026-11-13
Closed: 2026-11-13

## Problem

The developer pinned unsloth/Qwen3.8-27B-GGUF @ ce331efc48d5722b4b8041f72793e530eae18d9b UD-Q8_K_XL (31,457,991,680 B, sha256 af36ecb6b5db1407953345b746c14ac93f0657dda413910b4348683a2d990377, /mnt/models/unsloth-qwen3.8-27B-gguf-v2/) and the TT path refuses it: kQ8_K is activation-only in include/vt/dtype.h:41 and absent from tenstorrent_keepquant.cpp. The block layout is friendly — block_q8_K = 36 B per 32 elements = exactly 9 keepquant words, zero padding, and kQ8_K is already the K-quant activation vec_dot_type so the INT8DOT machinery exists. Owed: W1 loader acceptance + dtype-comment record, W2 staging (9 words, no pad) + Q8_KxQ8_K INT8 dot arm vs the CPU f32 oracle, W3 e2e device gate on the pinned artifact (KV pool sized to fit beside 31.5 GB weights — the Q6 legs proved 64 blocks OOM; record the largest fitting pool), existing Q4_K/Q6_K arms byte-identical. Spec: .agents/specs/tt-keepquant-q8k-weights.md

## Resolution

W1+W2 shipped green (loader + 80-word staging + dequant-composite serve; red-before captured; 337/337 device assertions; full loader suite 63/63; Q4_K/Q6_K byte-identity holds — the lone subset failure is the pre-existing kGdnPostConv gap, fails identically on build-q6k). W3 STOP CONDITION MET: the pinned UD-Q8_K_XL artifact (sha256 af36ecb6... verified) OOMs deterministically at the 8-block KV floor in all 3 legs (TT_FATAL 178257920 B, 7.5 MB free, ~171 MB short, inside MatmulBTQuantInt8DotKernel in Qwen3_5DenseModel::ForwardDevice) — unservable-on-P150 at the standing recipe; recorded in docs/bench-evidence/tt-q8k-weights-20261113/.
