#!/usr/bin/env python3
"""Generate tests/vllm/models/nimble_goldens.inc from the Nimble reference code.

MODEL-NIMBLE (.agents/specs/nimble.md). The goldens are produced by running the
model author's own code, not a transcription of it:

  prompts  bespokelabsai/nimble @ 62076b4f2d365b5879dafcf7f6dd072a1fe76df7
           nimble/serving/compiler.py -> nimble/scoring/parallel_schema.py
           (sha256 a0a0f94d..., the same file the checkpoint's
           schema_config.json pins as prompt_code_sha256)
  answers  ekzhang/openjev-sglang @ 7f84bedc169439f03379c2fa8d00ada220af2295
           src/openjev/scoring.py answer()
  requests src/openjev/models.py SystemOneRequest (validates every case)

Usage:
  python3 scripts/gen-nimble-goldens.py \
      --nimble-repo <bespokelabsai/nimble checkout> \
      --openjev-src <openjev-sglang>/src \
      --tokenizer-dir <bespokelabs/Bespoke-Nimble-9B @ bd792f44 (tokenizer files)> \
      --out tests/vllm/models/nimble_goldens.inc

Needs transformers, pydantic, pydantic-settings and orjson.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

PROMPT_CODE_SHA256 = "a0a0f94d0f65e972bc20d088678ad3d595ff1c42b9c0d78f96034526303f63fc"

# Each case is a /v1/systemone body openjev accepts. Together they reach every
# branch of the mapping and of safe_json: default and explicit noul criteria,
# a null choice description, a score rubric, non-string state and
# instructions, "<" / ">", quotes, a newline, and non-ASCII text.
CASES = [
    {
        "model": "nimble",
        "state": "I was charged twice. Please refund the duplicate.",
        "questions": {
            "refund": {"type": "noul", "instructions": "Does the user request a refund?"},
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this?",
                "criteria": {"billing": "Payments and refunds", "technical": "Software bugs"},
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["Routine", "Urgent", "Emergency"],
            },
        },
    },
    {
        "model": "nimble",
        "state": {
            "ticket": 4412,
            "body": "Café <b>closed</b> \"early\"\nagain",
            "tags": ["billing", 2.5, True, None],
        },
        "questions": {
            "closed": {
                "type": "noul",
                "instructions": {"ask": "Was the venue closed?", "strict": True},
                "criteria": {"true": "It was closed", "false": "It was open"},
            },
            "who": {
                "type": "choice",
                "instructions": "Who wrote this?",
                "criteria": {"customer": None, "staff": "An employee <internal>", "bot": None},
            },
        },
    },
]

# Deterministic candidate logits per field width, and the two published T.
LOGITS = {2: [1.25, -0.5], 3: [0.75, 2.0, -1.0]}
TEMPERATURES = [1.0, 2.179078721266035]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nimble-repo", required=True)
    ap.add_argument("--openjev-src", required=True)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.openjev_src)
    sys.path.insert(0, args.nimble_repo)
    from transformers import AutoTokenizer
    from openjev.models import SystemOneRequest
    from openjev.scoring import answer
    import nimble.scoring.parallel_schema as parallel_schema
    from nimble.serving.compiler import NimbleCompiler

    digest = hashlib.sha256(Path(parallel_schema.__file__).read_bytes()).hexdigest()
    if digest != PROMPT_CODE_SHA256:
        raise SystemExit(f"parallel_schema.py sha256 {digest} != pinned {PROMPT_CODE_SHA256}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    compiler = NimbleCompiler(tokenizer, max_prompt_tokens=8192)
    out_cases = []
    for body in CASES:
        request = SystemOneRequest.model_validate(body)
        prepared = compiler.prepare(request)
        prompts, ids, candidates, answers = [], [], [], []
        for branch in prepared.branches:
            text = tokenizer.decode(branch.input_ids, skip_special_tokens=False)
            if tokenizer.encode(text, add_special_tokens=False) != branch.input_ids:
                raise SystemExit("decode/encode round trip is not exact")
            prompts.append(text)
            ids.append(branch.input_ids)
            candidates.append(branch.label_ids)
            logits = LOGITS[len(branch.option_keys)]
            for t in TEMPERATURES:
                answers.append({
                    "field": branch.question_id,
                    "temperature": t,
                    "logits": logits,
                    "expected": answer(branch, logits, t).model_dump(mode="json"),
                })
        out_cases.append({"body": body, "prompts": prompts, "ids": ids,
                          "candidate_ids": candidates, "answers": answers})

    payload = json.dumps({"cases": out_cases}, ensure_ascii=False, indent=1)
    if ")NIMBLE\"" in payload:
        raise SystemExit("payload collides with the raw-string delimiter")
    Path(args.out).write_text(
        "// GENERATED by scripts/gen-nimble-goldens.py. Do not edit.\n"
        "// Reference revisions are listed in that script's docstring.\n"
        "// clang-format off\n"
        "static const char* const kNimbleGoldens = R\"NIMBLE(" + payload + ")NIMBLE\";\n"
        "// clang-format on\n",
        encoding="utf-8")
    print(f"wrote {args.out}: {len(out_cases)} cases")


if __name__ == "__main__":
    main()
