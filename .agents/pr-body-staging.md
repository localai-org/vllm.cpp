feat(TT-STAGING-FOOTPRINT): the keepquant 64-B staging grid inflates high-bit weights past one P150 — W1 constraint, row-page repack, and the UD-Q6_K_XL serving gate

Both new unsloth artifacts — UD-Q6_K_XL (25.3 GB) and UD-Q8_K_XL
(31.5 GB), byte-verified and metadata-servable — OOM on the P150 at
every KV pool while the raw weights would fit: the keepquant 64-B
staging grid pads Q6_K rows x1.22 and Q8_0 rows x1.88, staging
UD-Q6_K_XL to ~36.9 GiB against a 34 GB board. The tt-model bundle
contrast (bf16 tilized via ttnn.from_torch on four Blackhole chips, no
host-block staging) shows the plugin never faces this; with one board
our staging format is the lever.

W1 (records): the leg33 constraint is ALIVE — the kernel reads staged
pages at 64-B stride (noc_async_read page walk; dot stride
block_word_bytes), only ceil-to-64 explains the leg33 fix (a Q4_K
block at 144 -> 192 B; the Q5_K framing was wrong, and leg41 is a
separate aliasing bug), and every registered encoding already sits at
its ceil-64 minimum — so grid tightening is killed and the row-page
repack is selected.

Design B (the implementation): one weight row per accessor page — tight
nb blocks at their exact byte strides, zero-padded to ceil64 per row
(pad <= 63 B per row instead of per block). Staged UD-Q6_K_XL drops
36.9 -> 23.6 GiB. The kernel walk (CARG_BLK_BYTES) is localized to the
row-packed encodings; the legacy Q4_K/Q5_K path is byte-identical.

Gates, all reproduced by the coordinator and two fresh reviewers:
- Q4_K_M greedy stream BYTE-IDENTICAL to the recorded pre-change
  stream (sha256 d49c53d2..., live leg re-run on the P150, TPOT
  6624.35 ms matching the recorded 6624).
- THE SERVING GATE: UD-Q6_K_XL serves on one P150 — 3/3 deterministic
  token-ID reps at the eager arm (KV 8; the pool caps output at 64
  tokens, recorded), TPOT 5340/5342/5340 ms — ~19% FASTER than the
  Q4_K_M baseline. The captured arm OOMs at trace capture by ~80 MB,
  fragmentation-bound; the tied-embed dual residency
  (qwen3_5_gguf_weights.cpp:883->:897, keepquant.cpp:107 beside the
  bf16 twin) is the recorded follow-up lever.
- Detection is mutation-proven: the ceil64->ceil32 grid corruption and
  a row-word off-by-one both go red on the new divergence-class cases
  (Q8_0 nb=40, Q6_K nb=4 — divergence iff 0 < (nb x bb) mod 64 <= 32);
  an invalid first red (a cwd-driven include-dir environment error)
  was caught by the scoped review and replaced with the genuine
  geometry refusal, and the divergence-rule wording corrected.
- W3 STOP CONDITION recorded from the companion Q8_K row: the
  UD-Q8_K_XL artifact's Q8_0-dominated mix (83.6%) needs this same
  repack; its unservability verdict predates this fix.

Fresh second-pass review PASS (arithmetic, stride mutation, byte
identity, hooks inert); scoped repair review PASS after replacing the
invalid red log and correcting the divergence-rule wording. Evidence:
docs/bench-evidence/tt-staging-footprint-20261113/ (plus the
companion qwen38-q6k-vs-q4km-20261113/ comparison directory committed
on the branch). docs/USAGE.md records the newly servable arm.

FOLLOWING_AGENTS_PROTOCOL

Following-Agents-Protocol: true
AI-Assisted: true
Assisted-by: AGENT:zai/glm-5-3-flash [maki]
