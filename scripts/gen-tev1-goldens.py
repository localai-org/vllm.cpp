#!/usr/bin/env python3
"""Generate tests/vllm/models/tev1_goldens.inc from the Tev1 and Ollama references.

MODEL-TEV1 Phase 6 (.agents/specs/tev1.md). The goldens come from running the
reference code, not from a transcription of it:

  compile  ollama/ollama @ 1abe35e6e6e777e858bbfbba283667ee8d516801
           decision/systemone.go Compile: which choices a question gets, in
           which order, with which code, value and description
  answers  the same file, Answer: probabilities, noul, score, legend,
           confidence, from fixed candidate logits
  prompt   togethercomputer/tev1 @ 1dde7782382c9f49d627153759b8d1deab426ce0
           examples/decide.py payload() (system prompt, JSON user turn, option
           validation), rendered by transformers apply_chat_template with
           add_generation_prompt=True and enable_thinking=False on BOTH
           checkpoints' tokenizers:
             togethercomputer/Tev1-4B-experimental @ 0b7becf017daa0e5eb222f8ce7483c8c8259c52f
             togethercomputer/Tev1-0.8B-experimental @ 6bb2dff14b38fea90ddb14d870166ccaf77374e9

The Ollama part runs as a small Go program inside the Ollama checkout, because
decision.Compile and Answer are the code under reference. The program is written
to a scratch package in that checkout and removed afterwards.

Usage:
  python3 scripts/gen-tev1-goldens.py \
      --ollama-repo <ollama checkout> --tev1-repo <togethercomputer/tev1 checkout> \
      --tokenizer-dir <Tev1-4B-experimental dir> \
      --tokenizer-dir <Tev1-0.8B-experimental dir> \
      --out tests/vllm/models/tev1_goldens.inc

Needs go and transformers.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

OLLAMA_REV = "1abe35e6e6e777e858bbfbba283667ee8d516801"
TEV1_REV = "1dde7782382c9f49d627153759b8d1deab426ce0"

GO_PROGRAM = r'''package main

import (
	"encoding/json"
	"os"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/decision"
	"github.com/ollama/ollama/llm"
)

type input struct {
	Body   decision.Request `json:"body"`
	Logits [][]float32      `json:"logits"`
}

func main() {
	var in input
	if err := json.NewDecoder(os.Stdin).Decode(&in); err != nil {
		panic(err)
	}
	out := map[string]any{}
	c, err := decision.Compile(in.Body)
	if err != nil {
		out["error"] = err.Error()
		json.NewEncoder(os.Stdout).Encode(out)
		return
	}
	var turns []string
	if err := c.Render(func(m []api.Message) (string, error) {
		turns = append(turns, m[0].Content)
		return "", nil
	}); err != nil {
		panic(err)
	}
	out["turns"] = turns
	if in.Logits != nil {
		r, err := c.Answer(in.Body.Model, llm.ScoreResponse{Logits: in.Logits})
		if err != nil {
			panic(err)
		}
		out["answers"] = r.Answers
	}
	json.NewEncoder(os.Stdout).Encode(out)
}
'''

# /v1/systemone bodies. Together they reach every branch of the mapping and of
# the JSON rendering: the model card's own example, default and explicit noul
# criteria, a null choice description, a score rubric, an object state, object
# instructions, quotes, a newline, "<" / ">" (Tev1 does not escape them) and
# non-ASCII text, and the widest field the model documents (24 options).
CASES = [
    {
        "model": "tev1",
        "state": "Returns are allowed within 30 days. This purchase was 12 days ago.",
        "questions": {
            "window": {
                "type": "choice",
                "instructions": "Is this return within the allowed window?",
                "criteria": {"yes": "Yes.", "no": "No.", "unknown": "Not enough information."},
            },
        },
    },
    {
        "model": "tev1",
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
        "model": "tev1",
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
    {
        "model": "tev1",
        "state": "Route the request to the matching queue.",
        "questions": {
            "queue": {
                "type": "choice",
                "instructions": "Which queue?",
                "criteria": {f"q{i:02d}": f"Queue number {i}" for i in range(24)},
            },
        },
    },
]

# Refusals: Ollama's own error for each body. The engine must refuse all of
# them too (its wording is its own). The 25-option body is the one Ollama
# accepts and the lane refuses, because decide.py documents labels A-X only;
# its ollama_error is null, which records the difference.
REFUSALS = [
    {"model": "tev1", "state": "  ", "questions": {"a": {"type": "noul", "instructions": "x"}}},
    {"model": "tev1", "state": "s", "questions": {"a": {"type": "rank", "instructions": "x"}}},
    {"model": "tev1", "state": "s", "questions": {"a": {"type": "choice", "instructions": "x",
                                                         "criteria": {"only": None}}}},
    {"model": "tev1", "state": "s",
     "questions": {f"q{i}": {"type": "noul", "instructions": "x"} for i in range(65)}},
    {"model": "tev1", "state": "s", "questions": {"a": {"type": "choice", "instructions": "x",
                                                         "criteria": {f"k{i}": None for i in range(27)}}}},
    {"model": "tev1", "state": "s", "questions": {"a": {"type": "choice", "instructions": "x",
                                                         "criteria": {f"k{i}": None for i in range(25)}}}},
]

# Candidate logits exactly representable in float32, since Ollama's scorer
# returns float32 and Answer widens them.
LOGITS = {
    2: [1.25, -0.5],
    3: [0.75, 2.0, -1.0],
    24: [((i * 7) % 11) * 0.25 - 1.0 for i in range(24)],
}


def run_ollama(ollama: Path, pkg: Path, body, logits):
    proc = subprocess.run(
        ["go", "run", "./" + str(pkg.relative_to(ollama))],
        cwd=ollama, input=json.dumps({"body": body, "logits": logits}),
        capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def serialize(value):
    # nimble compiler.py serialize(): a string as-is, anything else json.dumps.
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def tev1_record(body, field_name, schema_field):
    # The lane's mapping: the model's own state value, the question text, and
    # one option per Ollama choice, labelled with Ollama's code.
    question = body["questions"][field_name]["instructions"]
    options = []
    for choice in schema_field["choices"]:
        value = choice["value"]
        key = ("true" if value else "false") if isinstance(value, bool) else value
        options.append({"label": choice["code"], "key": key,
                        "description": choice["description"]})
    return {"state": body["state"], "question": serialize(question), "options": options}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama-repo", required=True)
    ap.add_argument("--tev1-repo", required=True)
    ap.add_argument("--tokenizer-dir", action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ollama = Path(args.ollama_repo).resolve()
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ollama, capture_output=True,
                         text=True, check=True).stdout.strip()
    if rev != OLLAMA_REV:
        raise SystemExit(f"ollama checkout is {rev}, pinned {OLLAMA_REV}")
    tev1_rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.tev1_repo,
                              capture_output=True, text=True, check=True).stdout.strip()
    if tev1_rev != TEV1_REV:
        raise SystemExit(f"tev1 checkout is {tev1_rev}, pinned {TEV1_REV}")
    sys.path.insert(0, str(Path(args.tev1_repo) / "examples"))
    import decide  # togethercomputer/tev1 examples/decide.py
    from transformers import AutoTokenizer

    tokenizers = [AutoTokenizer.from_pretrained(d) for d in args.tokenizer_dir]
    pkg = ollama / "internal" / "vllmcpp_tev1_goldens"
    pkg.mkdir(parents=True, exist_ok=False)
    try:
        (pkg / "main.go").write_text(GO_PROGRAM)
        out_cases = []
        for body in CASES:
            widths = [len(q["criteria"]) if q["type"] != "noul" else 2
                      for q in body["questions"].values()]
            logits = [LOGITS[w] for w in widths]
            ref = run_ollama(ollama, pkg, body, logits)
            if "error" in ref:
                raise SystemExit(f"ollama refused a golden case: {ref['error']}")
            prompts, ids, candidates = [], [], []
            for name, turn in zip(body["questions"], ref["turns"]):
                schema = json.loads(turn.split("\n\nRequested field: ")[0])["schema"]
                field = next(f for f in schema if f["name"] == name)
                messages = decide.payload(tev1_record(body, name, field), "tev1")["messages"]
                rendered = [t.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True,
                                                  enable_thinking=False)
                            for t in tokenizers]
                if any(r != rendered[0] for r in rendered):
                    raise SystemExit("the two checkpoints render different prompts")
                prompt = rendered[0]
                tok = tokenizers[0]
                prompt_ids = tok.encode(prompt, add_special_tokens=False)
                for t in tokenizers[1:]:
                    if t.encode(prompt, add_special_tokens=False) != prompt_ids:
                        raise SystemExit("the two checkpoints tokenize differently")
                cand = []
                for choice in field["choices"]:
                    joined = tok.encode(prompt + choice["code"], add_special_tokens=False)
                    literal = tok.encode(choice["code"], add_special_tokens=False)
                    if joined[:-1] != prompt_ids or len(literal) != 1 or literal[0] != joined[-1]:
                        raise SystemExit(f"{choice['code']} is not one token at the boundary")
                    cand.append(joined[-1])
                prompts.append(prompt)
                ids.append(prompt_ids)
                candidates.append(cand)
            out_cases.append({"body": body, "prompts": prompts, "ids": ids,
                              "candidate_ids": candidates, "logits": logits,
                              "answers": ref["answers"]})
        refusals = []
        for body in REFUSALS:
            ref = run_ollama(ollama, pkg, body, None)
            refusals.append({"body": body, "ollama_error": ref.get("error")})
    finally:
        shutil.rmtree(pkg)

    payload = json.dumps({"cases": out_cases, "refusals": refusals},
                         ensure_ascii=False, indent=1)
    if ")TEV1\"" in payload:
        raise SystemExit("payload collides with the raw-string delimiter")
    Path(args.out).write_text(
        "// GENERATED by scripts/gen-tev1-goldens.py. Do not edit.\n"
        "// Reference revisions are listed in that script's docstring.\n"
        "// clang-format off\n"
        "static const char* const kTev1Goldens = R\"TEV1(" + payload + ")TEV1\";\n"
        "// clang-format on\n",
        encoding="utf-8")
    print(f"wrote {args.out}: {len(out_cases)} cases, {len(refusals)} refusals")


if __name__ == "__main__":
    main()
