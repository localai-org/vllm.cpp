#!/usr/bin/env python3
"""Generate tests/vllm/models/clm_goldens.inc from the CLM reference code.

MODEL-CLM (.agents/specs/clm.md, "Fidelity repair"). Every golden comes from
running Contrastive-LM/CLM @ bb42c6c5bf914fd449bed2f6ca65be80602cb1f7, never
from a transcription of it:

  heads     src/clm/heads.py make_head: two small heads (the checkpoint's
            shape, and a residual, no-LayerNorm variant), their state-dict
            tensors by name, and HeadPair's normalize(head(x)) on fixed inputs
  scale     HeadPair._load: exp(logit_scale).clamp(max=100.0) in float32,
            including the published checkpoint's logit_scale
  pairs     src/clm/schema.py build_pairs: the state text and candidate texts
  answers   src/clm/schema.py answer_from_logits
  refusals  the ValueError text Engine.answer raises for a malformed request
  pipeline  src/clm/engine.py Engine.answer end to end, with a tiny head saved
            in the checkpoint format and an embedder stand-in whose raw
            last-token vectors are written to the golden, so the C++ side can
            feed the same raw vectors into its own pooling normalization
  real      the published checkpoint's answers on five requests, produced by
            the reference Engine with Qwen/Qwen3-8B @
            b968826d9c46dd6066d109eabc6255188de91218 run by transformers in
            bfloat16 (last token of the post-norm hidden state, L2-normalized:
            what `vllm serve --runner pooling` returns). This part is copied
            from a JSON file that run is written to (--real-json), because
            the 8B forward is not something a golden generator should repeat.

Usage:
  python3 scripts/gen-clm-goldens.py --clm-repo <Contrastive-LM/CLM checkout> \
      [--real-json <reference run output>] --out tests/vllm/models/clm_goldens.inc

Needs torch and numpy.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

CLM_REV = "bb42c6c5bf914fd449bed2f6ca65be80602cb1f7"
# CLM_v0.1-8B.pt @ Contrastive-LM/CLM-v0.1-8B e939398d4556fcd9400c76fa8c5a513202f42b0a
# (sha256 b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5):
# ck["logit_scale"] is this float32.
CHECKPOINT_LOGIT_SCALE_HEX = "0x1.273f780000000p+2"
DELIM = "CLM"

HEAD_CONFIGS = [
    # The published checkpoint's cfg, scaled down.
    dict(width=6, depth=3, proj=4, hidden=8, activation="gelu", layernorm=True, residual=False),
    # The other branches make_head has.
    dict(width=6, depth=4, proj=4, hidden=8, activation="silu", layernorm=False, residual=True),
]

PAIR_CASES = [
    {"state": "john works at google",
     "questions": {"pick": {"type": "choice", "instructions": "entity type",
                            "criteria": {"person": None, "organization": ""}}}},
    {"state": "  padded state  ",
     "questions": {"a": {"type": "choice", "instructions": "  Which one?  ",
                         "criteria": {"x": "the letter x", "y": {"kind": "letter", "rank": 25}}},
                   "b": {"type": "noul", "instructions": "Is it a letter?"},
                   "c": {"type": "noul", "instructions": "Is it a vowel?",
                         "criteria": {"true": "it is a vowel", "false": ""}},
                   "d": {"type": "noul"},
                   "e": {"type": "score", "instructions": "How late?",
                         "criteria": ["early", {"level": "late", "by": 2.5}, 3, True]}}},
    {"state": {"user": "alice", "age": 31, "ok": False, "none": None, "empty": [],
               "history": ["hi", {"role": "bot", "text": "hello"}, [1, 2]],
               "meta": {"a": 1, "b": {"c": [0.5, "x"]}}},
     "questions": {"q": {"type": "choice", "instructions": None,
                         "criteria": {"k1": None, "k2": 0}}}},
    {"state": ["first", "second", {"third": 3}],
     "questions": {"q": {"type": "noul", "instructions": {"ask": "is it long?"}}}},
    {"state": "", "questions": {"q": {"type": "noul", "instructions": "only the question"}}},
]

ANSWER_CASES = [
    ({"type": "choice", "criteria": {"a": None, "b": None, "c": None}}, [1.5, -0.25, 3.0]),
    ({"type": "choice", "criteria": {"only": None}}, [0.7]),
    ({"type": "noul", "instructions": "x"}, [0.2, 1.1]),
    ({"type": "score", "criteria": ["low", {"level": "mid"}, "high"]}, [0.1, 2.0, -1.0]),
]

REFUSAL_CASES = [
    {"state": "s", "questions": {}},
    {"state": "s", "questions": {"q": {"type": "rank"}}},
    {"state": "s", "questions": {"q": {"type": "choice", "criteria": {}}}},
    {"state": "s", "questions": {"q": {"type": "choice", "criteria": ["a", "b"]}}},
    {"state": "s", "questions": {"q": {"type": "score", "criteria": ["one"]}}},
    {"state": "s", "questions": {"q": {"type": "score", "criteria": {"a": 1, "b": 2}}}},
    {"state": "s", "temperature": 0, "questions": {"q": {"type": "noul", "instructions": "x"}}},
    {"state": "s", "temperature": 100.5, "questions": {"q": {"type": "noul", "instructions": "x"}}},
]

PIPELINE_CASES = [
    {"state": "the cat sat on the mat",
     "questions": {"pick": {"type": "choice", "instructions": "what animal?",
                            "criteria": {"cat": "a small feline", "dog": None, "bird": "it flies"}},
                   "yes": {"type": "noul", "instructions": "Is there an animal?"},
                   "size": {"type": "score", "instructions": "How big is it?",
                            "criteria": ["tiny", "small", "large"]}}},
    {"state": {"room": "kitchen", "items": ["knife", "pan"]}, "temperature": 0.5,
     "questions": {"pick": {"type": "choice", "instructions": "what room?",
                            "criteria": {"kitchen": None, "bath": "the bathroom"}},
                   "dup": {"type": "choice", "instructions": "what room?",
                           "criteria": {"k": "kitchen", "b": "the bathroom"}}}},
]


def load_reference(repo):
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != CLM_REV:
        raise SystemExit(f"{repo} is at {head}, not the pinned {CLM_REV}")
    sys.path.insert(0, str(Path(repo) / "src"))
    import clm.engine as engine
    import clm.heads as heads
    import clm.schema as schema
    return engine, heads, schema


def f32_list(t):
    return [float(v) for v in t.detach().float().reshape(-1).tolist()]


def raw_vector(text, dim):
    """A fixed, non-unit 'last-token hidden' per text, so pooling must normalize."""
    import numpy as np
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")
    rng = np.random.RandomState(seed)
    return (rng.standard_normal(dim) * 3.0).astype(np.float32)


def make_heads(heads_mod, cfg, seed):
    import torch
    torch.manual_seed(seed)
    pair = []
    for _ in range(2):
        h = heads_mod.make_head(**cfg)
        with torch.no_grad():
            for p in h.parameters():
                p.add_(torch.randn_like(p) * 0.3)  # LayerNorm's 1/0 init would hide a swap
        pair.append(h.eval())
    return pair


def gen_heads(heads_mod):
    import torch
    out = []
    for i, cfg in enumerate(HEAD_CONFIGS):
        head, _ = make_heads(heads_mod, cfg, 1000 + i)
        torch.manual_seed(2000 + i)
        x = torch.randn(3, cfg["hidden"])
        with torch.no_grad():
            y = torch.nn.functional.normalize(head(x), dim=-1)
        out.append({"cfg": cfg,
                    "tensors": {k: {"shape": list(v.shape), "data": f32_list(v)}
                                for k, v in head.state_dict().items()},
                    "inputs": [f32_list(r) for r in x], "outputs": [f32_list(r) for r in y]})
    return out


def gen_scale():
    import torch
    values = [float.fromhex(CHECKPOINT_LOGIT_SCALE_HEX), math.log(50.0), 0.0, 10.0]
    # HeadPair._load, verbatim.
    return [{"logit_scale": v,
             "scale": float(torch.as_tensor(v).float().exp().clamp(max=100.0))} for v in values]


def gen_pipeline(engine_mod, heads_mod):
    import numpy as np
    import torch
    from clm.embedder import l2
    cfg = HEAD_CONFIGS[0]
    sh, ah = make_heads(heads_mod, cfg, 3000)
    logit_scale = 3.25
    vectors = {}

    class Embedder:
        def embed(self, texts):
            rows = []
            for t in texts:
                vectors[t] = raw_vector(t, cfg["hidden"])
                rows.append(l2(vectors[t]))  # what the reference Embedder hands the head
            return np.stack(rows), 0

    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "tiny.pt"
        torch.save({"state_head": sh.state_dict(), "action_head": ah.state_dict(),
                    "logit_scale": torch.tensor(logit_scale),
                    "cfg": {"width": cfg["width"], "depth": cfg["depth"],
                            "projection_dim": cfg["proj"], "activation": cfg["activation"],
                            "layernorm": cfg["layernorm"], "residual": cfg["residual"],
                            "hidden_size": cfg["hidden"]}}, ck)
        eng = engine_mod.Engine(embedder=Embedder(), checkpoint=str(ck), device="cpu",
                                action_cache=0)
        cases = []
        for body in PIPELINE_CASES:
            resp = eng.answer(body["state"], body["questions"],
                              temperature=body.get("temperature", 1.0))
            cases.append({"body": body, "answers": resp["answers"]})
    return {"cfg": cfg, "logit_scale": logit_scale,
            "state_head": {k: {"shape": list(v.shape), "data": f32_list(v)}
                           for k, v in sh.state_dict().items()},
            "action_head": {k: {"shape": list(v.shape), "data": f32_list(v)}
                            for k, v in ah.state_dict().items()},
            "vectors": {k: [float(x) for x in v] for k, v in vectors.items()},
            "cases": cases}


def gen_refusals(engine_mod):
    class NoEmbedder:
        def embed(self, texts):
            raise AssertionError("a refused request must not reach the encoder")

    eng = engine_mod.Engine(embedder=NoEmbedder(), checkpoint=None, device="cpu",
                            action_cache=0)
    out = []
    for body in REFUSAL_CASES:
        try:
            eng.answer(body["state"], body["questions"], model="clm-raw",
                       temperature=body.get("temperature", 1.0))
        except ValueError as e:
            out.append({"body": body, "error": str(e)})
            continue
        raise SystemExit(f"the reference accepted {body}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clm-repo", required=True)
    ap.add_argument("--real-json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    engine_mod, heads_mod, schema = load_reference(args.clm_repo)

    pairs = []
    for body in PAIR_CASES:
        got = schema.build_pairs(body["state"], body["questions"])
        pairs.append({"body": body, "pairs": {k: {"state_text": v[0], "keys": v[1], "candidates": v[2]}
                                              for k, v in got.items()}})
    answers = []
    for q, logits in ANSWER_CASES:
        keys, _ = schema.candidates(q)
        answers.append({"question": q, "logits": logits,
                        "answer": schema.answer_from_logits(q, keys, logits)})
    real = None
    if args.real_json:
        run = json.loads(Path(args.real_json).read_text())
        real = {"dtype": run["dtype"], "scale": run["scale"],
                "cases": [{"body": r["request"], "answers": r["response"]["answers"]}
                          for r in run["results"]]}
    golden = {"clm_revision": CLM_REV, "heads": gen_heads(heads_mod), "scale": gen_scale(),
              "pairs": pairs, "answers": answers, "refusals": gen_refusals(engine_mod),
              "pipeline": gen_pipeline(engine_mod, heads_mod), "real": real}
    payload = json.dumps(golden, indent=1, ensure_ascii=False)
    if f"){DELIM}\"" in payload:
        raise SystemExit("payload collides with the raw-string delimiter")
    # MSVC caps one string literal piece at 16380 bytes; adjacent pieces concatenate.
    pieces = [payload[i:i + 8000] for i in range(0, len(payload), 8000)]
    body = "\n".join(f'R"{DELIM}({p}){DELIM}"' for p in pieces)
    Path(args.out).write_text(
        "// GENERATED by scripts/gen-clm-goldens.py. Do not edit.\n"
        "// Reference revisions are listed in that script's docstring.\n"
        "// clang-format off\n"
        f"static const char* const kClmGoldens =\n{body};\n"
        "// clang-format on\n")
    print(f"wrote {args.out} ({len(payload)} bytes)")


if __name__ == "__main__":
    main()
