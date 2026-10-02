#!/usr/bin/env python3
"""The Nimble adapter converter's contract, without the 9B checkpoint.

`scripts/convert-nimble.py` merges bespokelabs/Bespoke-Nimble-9B's LoRA into
Qwen3.5-9B once, offline (MODEL-NIMBLE). A module it fails to pair or skips
produces a model that loads and answers wrong, with no error anywhere. These
cases hold the pairing, the shape checks, the checksum and contract refusals,
and the config it writes, with fakes and no torch. The merge arithmetic runs
too when torch is installed.
"""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/convert-nimble.py"


def _module():
    spec = importlib.util.spec_from_file_location("convert_nimble", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _module()
A = "base_model.model.model.language_model.layers.{}.{}.lora_{}.weight"
W = "model.language_model.layers.{}.{}.weight"


class KeyMapping(unittest.TestCase):
    def test_gdn_and_attention_modules_map_to_the_base_names(self):
        self.assertEqual(M.map_adapter_key(A.format(0, "linear_attn.in_proj_qkv", "A")),
                         (W.format(0, "linear_attn.in_proj_qkv"), "A"))
        self.assertEqual(M.map_adapter_key(A.format(31, "self_attn.o_proj", "B")),
                         (W.format(31, "self_attn.o_proj"), "B"))
        self.assertEqual(M.map_adapter_key(A.format(3, "mlp.down_proj", "A")),
                         (W.format(3, "mlp.down_proj"), "A"))

    def test_kev_prefix_and_foreign_names_do_not_map(self):
        # convert-kev.py's adapter prefix; a Nimble adapter never carries it.
        self.assertIsNone(M.map_adapter_key("base_model.model.layers.0.mlp.up_proj.lora_A.weight"))
        self.assertIsNone(M.map_adapter_key(A.format(0, "mlp.up_proj", "C")))
        self.assertIsNone(M.map_adapter_key("base_model.model.lm_head.weight"))


class Plan(unittest.TestCase):
    def shapes(self, rank=4):
        adapter = {A.format(0, "mlp.up_proj", "A"): [rank, 8],
                   A.format(0, "mlp.up_proj", "B"): [16, rank],
                   A.format(0, "linear_attn.in_proj_a", "A"): [rank, 8],
                   A.format(0, "linear_attn.in_proj_a", "B"): [2, rank]}
        base = {W.format(0, "mlp.up_proj"): [16, 8],
                W.format(0, "linear_attn.in_proj_a"): [2, 8],
                W.format(0, "mlp.down_proj"): [8, 16]}
        return adapter, base

    def test_every_module_is_paired(self):
        adapter, base = self.shapes()
        plan = M.plan_merge(adapter, base, 4)
        self.assertEqual(set(plan), {W.format(0, "mlp.up_proj"),
                                     W.format(0, "linear_attn.in_proj_a")})

    def test_a_module_missing_from_the_base_is_refused(self):
        adapter, base = self.shapes()
        del base[W.format(0, "linear_attn.in_proj_a")]
        with self.assertRaisesRegex(M.ConvertError, "no such tensor"):
            M.plan_merge(adapter, base, 4)

    def test_a_half_module_is_refused(self):
        adapter, base = self.shapes()
        del adapter[A.format(0, "mlp.up_proj", "B")]
        with self.assertRaisesRegex(M.ConvertError, "only lora_A"):
            M.plan_merge(adapter, base, 4)

    def test_a_shape_that_does_not_fit_is_refused(self):
        adapter, base = self.shapes()
        base[W.format(0, "mlp.up_proj")] = [8, 16]  # transposed
        with self.assertRaisesRegex(M.ConvertError, "do not fit"):
            M.plan_merge(adapter, base, 4)

    def test_an_unrecognized_adapter_tensor_is_refused(self):
        adapter, base = self.shapes()
        adapter["base_model.model.lm_head.weight"] = [4, 4]
        with self.assertRaisesRegex(M.ConvertError, "unrecognized"):
            M.plan_merge(adapter, base, 4)


class Contract(unittest.TestCase):
    def write_adapter(self, d, temperature=1.0, prompt=None, task=None):
        d = Path(d)
        (d / "adapter_config.json").write_text(json.dumps(
            {"peft_type": "LORA", "r": 16, "lora_alpha": 32, "bias": "none",
             "use_dora": False, "use_rslora": False}))
        (d / "schema_config.json").write_text(json.dumps(
            {"task": task or "schema_candidate_classification_v2",
             "model": "Qwen/Qwen3.5-9B", "revision": "c" * 40, "max_length": 8192,
             "prompt_code_sha256": prompt or M.PROMPT_CODE_SHA256}))
        (d / "temperature_config.json").write_text(json.dumps({"temperature": temperature}))
        return d

    def test_both_published_temperatures_are_read(self):
        for t in (1.0, 2.179078721266035):
            with tempfile.TemporaryDirectory() as d:
                _, _, got = M.read_contract(self.write_adapter(d, temperature=t))
                self.assertEqual(got, t)

    def test_a_different_prompt_contract_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(M.ConvertError, "prompt contract"):
                M.read_contract(self.write_adapter(d, prompt="0" * 64))

    def test_an_unknown_task_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(M.ConvertError, "training contract"):
                M.read_contract(self.write_adapter(d, task="schema_candidate_classification_v9"))

    def test_sha256sums_mismatch_and_missing_file_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "a.bin").write_bytes(b"abc")
            good = hashlib.sha256(b"abc").hexdigest()
            (p / "SHA256SUMS").write_text(f"{good}  a.bin\n")
            self.assertEqual(M.verify_sha256sums(p), ["a.bin"])
            (p / "a.bin").write_bytes(b"abd")
            with self.assertRaisesRegex(M.ConvertError, "mismatch"):
                M.verify_sha256sums(p)
            (p / "SHA256SUMS").write_text(f"{good}  b.bin\n")
            with self.assertRaisesRegex(M.ConvertError, "missing"):
                M.verify_sha256sums(p)

    def test_config_names_the_architecture_and_temperature(self):
        cfg = M.build_config({"architectures": ["Qwen3_5ForConditionalGeneration"],
                              "text_config": {"hidden_size": 4096}},
                             2.179078721266035, 8192, {"task": "t"})
        self.assertEqual(cfg["architectures"], ["NimbleModel"])
        self.assertEqual(cfg["nimble_temperature"], 2.179078721266035)
        self.assertEqual(cfg["nimble_max_length"], 8192)
        self.assertEqual(cfg["text_config"], {"hidden_size": 4096})


try:
    import torch
except ImportError:  # pragma: no cover - CI without torch
    torch = None


@unittest.skipUnless(torch is not None, "torch is not installed")
class MergeArithmetic(unittest.TestCase):
    def test_merge_is_peft_merge_on_a_bf16_base(self):
        g = torch.Generator().manual_seed(0)
        w = torch.randn(6, 5, generator=g).to(torch.bfloat16)
        a = torch.randn(2, 5, generator=g)
        b = torch.randn(6, 2, generator=g)
        got = M.merge_tensor(w, a, b, 2.0)
        self.assertEqual(got.dtype, torch.bfloat16)
        # PEFT LoraLayer.merge: base.weight.data += (B @ A) * scaling.
        want = w.clone()
        want += (b @ a) * 2.0
        self.assertTrue(torch.equal(got, want))
        # A merge that drops the scaling, or rounds the delta to bf16 first,
        # is a different tensor.
        self.assertFalse(torch.equal(got, (w.float() + b @ a).to(torch.bfloat16)))


if __name__ == "__main__":
    unittest.main()
