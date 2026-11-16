#!/usr/bin/env bash
# Summarize the W21 gate legs: token-ids sha256 vs the recorded references,
# mean TPOT vs the recorded baselines.
EV="$(cd "$(dirname "$0")" && pwd)"
Q4_REF=d49c53d23ac5bc4180096b30ebe3fb6990ba6f6eff7d6098760685f9035019ff
Q6_REF=b36bc61a59828c1ddf05fbabf62ca0a491c652d50f7ec06bee86d4b17ad18bcc
for n in w21-off-1 w21-off-2 w21-off-3 w21-on-1 w21-on-2 w21-on-3 \
         w21-q6kl-off-1 w21-q6kl-off-2 w21-q6kl-off-3 w21-q6kl-on-1; do
  f="$EV/$n-token-ids.json"
  [ -f "$f" ] || { echo "$n: (no token-ids)"; continue; }
  sha=$(sha256sum "$f" | cut -d' ' -f1)
  tpot=$(grep -m1 "Mean TPOT" "$EV/$n.log" 2>/dev/null | awk '{print $NF}')
  case "$n" in
    w21-off-*|w21-on-*) ref=$Q4_REF ;;
    *) ref=$Q6_REF ;;
  esac
  if [ "$sha" = "$ref" ]; then v=IDENTICAL; else v=DIFFERS; fi
  echo "$n: sha=${sha:0:8} $v tpot=${tpot:-?}"
done
