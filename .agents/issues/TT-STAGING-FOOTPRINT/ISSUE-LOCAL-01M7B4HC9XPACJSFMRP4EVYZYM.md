ID: ISSUE-LOCAL-01M7B4HC9XPACJSFMRP4EVYZYM
Title: TT staging footprint: the 64-B grid inflates Q6_K x1.52 / Q8_0 x1.88 — the new unsloth high-bit artifacts cannot fit one P150
Row: TT-STAGING-FOOTPRINT
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-13
Updated: 2026-11-13
Closed: -

## Problem

UD-Q6_K_XL (23.56 GiB) and UD-Q8_K_XL (29.30 GiB), both byte-verified and metadata-servable, OOM on the P150 at every KV pool (64/16/8 blocks; docs/bench-evidence/qwen38-q6k-vs-q4km-20261113/{oom-64blocks,oom-16blocks}/ + docs/bench-evidence/tt-q8k-weights-20261113/). Root arithmetic (corrected by W1 2026-11-13): the keepquant 64-B staging grid pads Q6_K rows 210->256 B (x1.22, 64 words — not the x1.52/320 B first recorded here) and Q8_0 34->64 B (x1.88), staging UD-Q6_K_XL to ~36.9 GiB against a 34 GB board, Q8_0-dominated (Q8_0 is 53.5% of UD-Q6_K_XL, 83.6% of UD-Q8_K_XL). The tt-model bundle contrast (developer request): it serves bf16 tilized via ttnn.from_torch on four Blackhole chips — no host-block staging; we have one board, so our staging format is the lever. Owed: W1 the leg33 investigation (what page property the 64-B pad actually protects, written as a constraint with citations — it selects or kills the designs), then Design A (32-B grid for Q6_K rows; Q8_0 has no win there) or Design B (plugin-shaped load-time repack into tight device-native quant tiles), gated greedy-exact where comparable and on existing-arm byte-identity (Q4_K_M must stay byte-identical). Q8_0's inflation is only fixable by Design B — check the artifact dtype mix before choosing. Spec: .agents/specs/tt-staging-footprint.md

## Resolution

- 2026-11-13 W1 finding (records-only; spec `.agents/specs/tt-staging-footprint.md` ## W1 finding carries the full derivation): the leg33 constraint is REAL and binds a LIVE consumer. Evidence: the only leg33 record is the code comment introduced with the pad itself, d9298c85c 2026-09-08 (src/vt/tenstorrent/tenstorrent_keepquant.cpp:15-22: "a 144-B logical page inside 192-B DRAM slots read back as stream bytes 16 + 152*slot"); the pre-leg33 grid was TIGHT exact-block-bytes rows (e456c390f: 36/44/53/9 words), so the 64-B pad is leg33's fix. Constraint: staged rows must be 64-B multiples because the int8-dot kernel strides the staged words tensor by acc_w.get_noc_addr(page) — tt-metal's aligned DRAM page stride, 64-B granularity per the recorded 144->192 rounding — while reading word_bytes = wpb*4 per page (tenstorrent_keepquant.cpp:1821-1822, kernels/keepquant_kernel_code.h:267-273). The consumer is the production dense arm (MatmulBTQuantInt8DotKernel, tenstorrent_keepquant.cpp:2333). Design A KILLED twice: 224 B is not a 64-B multiple, and every registered encoding already sits at its ceil-64 minimum, so no legal tightening exists. Design B SELECTED: tile rows ARE accessor pages sized as 64-B multiples, satisfying the constraint by construction; its kernel read-geometry change is M5-adjudicated; greedy-exactness + Q4_K_M byte-identity gates unchanged. Dtype mix measured off the artifacts (raw GGUF metadata read, offsets sum to file sizes): UD-Q8_K_XL = Q8_0 24.49 GiB 83.6% + BF16 4.79 GiB 16.4% (token_embd/output-adjacent Q8_0, attn_q/k/v/gate + output.weight BF16) with NO Q8_K (ggml type 9) tensors — the name is UD-matrix naming and this artifact never exercises the 032203666 Q8_K arm; UD-Q6_K_XL = Q8_0 12.60 GiB 53.5% + Q6_K 9.93 GiB 42.2% + Q5_K 1.00 GiB 4.2%. The Q8_0 share makes the outcome structural: no per-block grid change serves either artifact; only the repack does.
