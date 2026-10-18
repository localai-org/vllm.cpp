#!/usr/bin/env python3
# W0 VEHICLE PREP for the Yi + InternLM3 (Llama-alias) SACRED gates — a reproducible,
# GATE-NEUTRAL tokenizer staging step on the HF snapshot. Generalizes
# scripts/minicpm3-convert-safetensors.py step 3.
#
# WHY: our C++ engine's LoadedEngine::FromModelDir loads the tokenizer from
# `tokenizer.json` (tok::Tokenizer::FromHfJson), and our parser requires a
# non-null pre_tokenizer and rejects a non-empty normalizer Sequence
# (src/vllm/tokenizer/tokenizer.cpp:293,355). Yi (Yi-1.5-*) ships a tokenizer.json
# whose whitespace handling is a normalizer Sequence [Replace " "->"▁"] with a NULL
# pre_tokenizer; Yi-Coder-* and internlm3-8b-instruct ship ONLY a SentencePiece
# `tokenizer.model` with NO tokenizer.json at all. Either way FromModelDir cannot
# construct.
#
# WHAT (idempotent, in place in the snapshot dir):
#   1. If tokenizer.json is absent, build the fast tokenizer via
#      AutoTokenizer(use_fast=True, trust_remote_code=True) and save it (the fast
#      tokenizer serializes its SentencePiece to a tokenizer.json with the SAME
#      vocab + merges).
#   2. Re-express the whitespace normalizer as the equivalent Metaspace
#      pre_tokenizer our parser accepts: normalizer -> empty Sequence,
#      pre_tokenizer -> Metaspace {replacement "▁", prepend_scheme "first",
#      split False}. This is the SentencePiece<->Metaspace identity.
#
# The gate is ID-BASED (feeds the oracle's exact prompt ids via the TokensPrompt
# engine path and compares token ids), so tokenization is UNUSED by the forward —
# this staging only lets the engine construct and detokenize for display. It does
# NOT touch weights or the vLLM oracle path (the oracle uses its own tokenizer).
#
# Run on dgx:
#   YI_IL3_REPO_DIR=models--01-ai--Yi-Coder-1.5B-Chat \
#     ~/venvs/vllm-oracle/bin/python scripts/stage-tokenizer-metaspace.py
#   YI_IL3_REPO_DIR=models--internlm--internlm3-8b-instruct \
#     ~/venvs/vllm-oracle/bin/python scripts/stage-tokenizer-metaspace.py
import glob
import json
import os

REPO_DIR = os.environ["YI_IL3_REPO_DIR"]


def find_snapshot():
    base = os.path.join(os.path.expanduser("~"), ".cache/huggingface/hub",
                        REPO_DIR, "snapshots")
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        if os.path.exists(os.path.join(d, "config.json")):
            return d
    raise SystemExit(f"no snapshot with config.json under {base}")


def main():
    snap = find_snapshot()
    print(f"snapshot: {snap}")
    tokp = os.path.join(snap, "tokenizer.json")

    # 1. Generate tokenizer.json from the SentencePiece tokenizer.model if absent.
    if not os.path.exists(tokp):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(snap, use_fast=True,
                                            trust_remote_code=True)
        if hasattr(tok, "backend_tokenizer"):
            # A genuine fast tokenizer: its Rust backend serializes directly.
            tok.backend_tokenizer.save(tokp)
            print(f"tokenizer.json: generated via fast tokenizer backend "
                  f"({os.path.getsize(tokp)} bytes)")
        else:
            # A custom SLOW SentencePiece tokenizer (e.g. internlm3-8b-instruct's
            # InternLM3Tokenizer) with NO fast backend. Build the equivalent fast
            # BPE tokenizer directly from the sp_model: extract the pieces, derive
            # BPE merges the standard SentencePiece->BPE way (generate_merges), and
            # assemble a tokenizers.Tokenizer with a Metaspace pre_tokenizer +
            # decoder. (transformers 5.13.1's own SpmConverter path is broken here —
            # its LlamaConverter calls SentencePieceExtractor.extract(vocab_scores)
            # against an extract(model_type) signature.) The gate is ID-based, so the
            # exact tokenization is only used for engine construction + detok display.
            from transformers.convert_slow_tokenizer import SentencePieceExtractor
            from transformers.tokenization_utils_base import generate_merges
            from tokenizers import Tokenizer, models, pre_tokenizers, decoders, \
                AddedToken
            ext = SentencePieceExtractor(os.path.join(snap, "tokenizer.model"))
            pieces = [(p.piece, p.score) for p in ext.proto.pieces]
            vocab = {w: i for i, (w, s) in enumerate(pieces)}
            merges = generate_merges(vocab)
            unk = tok.unk_token
            bpe = models.BPE(vocab, merges,
                             unk_token=unk if unk in vocab else None,
                             fuse_unk=True, byte_fallback=True)
            fast = Tokenizer(bpe)
            fast.pre_tokenizer = pre_tokenizers.Metaspace(
                replacement="▁", prepend_scheme="first", split=False)
            fast.decoder = decoders.Metaspace(
                replacement="▁", prepend_scheme="first", split=False)
            # Carry any control/added tokens that live beyond the sp pieces.
            extra = [(t, i) for t, i in sorted(tok.get_vocab().items(),
                                               key=lambda x: x[1]) if i >= len(vocab)]
            if extra:
                fast.add_special_tokens(
                    [AddedToken(t, normalized=False, special=True) for t, _ in extra])
            fast.save(tokp)
            print(f"tokenizer.json: generated from tokenizer.model via SentencePiece "
                  f"BPE extraction ({len(vocab)} pieces, {len(merges)} merges, "
                  f"{len(extra)} added; {os.path.getsize(tokp)} bytes)")
    else:
        print("tokenizer.json: already present")

    # 2. Re-express the whitespace normalizer as the equivalent Metaspace
    #    pre_tokenizer our parser accepts (SentencePiece<->Metaspace identity).
    tok = json.load(open(tokp))
    norm = tok.get("normalizer")
    needs = (isinstance(norm, dict) and norm.get("type") == "Sequence"
             and norm.get("normalizers")) or tok.get("pre_tokenizer") is None
    if needs:
        tok["normalizer"] = {"type": "Sequence", "normalizers": []}
        tok["pre_tokenizer"] = {"type": "Metaspace", "replacement": "▁",
                                "prepend_scheme": "first", "split": False}
        json.dump(tok, open(tokp, "w"), ensure_ascii=False)
        print("tokenizer.json: normalizer -> empty Sequence; "
              "pre_tokenizer -> Metaspace")
    else:
        print("tokenizer.json: normalizer/pre_tokenizer already accepted; no change")


if __name__ == "__main__":
    main()
