#!/usr/bin/env python3
"""The CLM head converter's contract, without the 8B base.

`scripts/convert-clm.py` turns CLM_v0.1-8B.pt (MODEL-CLM) into the directory
the ClmModel loader reads. A tensor it renames, drops or reshapes, or a scale
it pre-applies, produces a model that loads and answers wrong with no error
anywhere; the first port's loader expected names no checkpoint carries. These
cases hold the names, the shape and cfg refusals, the config keys, and a full
conversion of a tiny checkpoint when torch is installed.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/convert-clm.py"


def _module():
    spec = importlib.util.spec_from_file_location("convert_clm", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _module()
# CLM_v0.1-8B.pt cfg, as stored.
PUBLISHED_CFG = {"model": "Qwen/Qwen3-8B", "hidden_size": 4096, "projection_dim": 512,
                 "width": 1536, "depth": 3, "activation": "gelu", "layernorm": True,
                 "residual": False}


class Shape:
    def __init__(self, *shape):
        self.shape = shape


def fake_checkpoint(cfg):
    sd = {k: Shape(*v) for k, v in M.expected_shapes(M.head_config(
        {"state_head": {}, "action_head": {}, "logit_scale": 0.0, "cfg": cfg})).items()}
    return {"state_head": dict(sd), "action_head": dict(sd), "logit_scale": 4.6132,
            "cfg": cfg}


class Names(unittest.TestCase):
    def test_the_published_cfg_maps_to_the_make_head_names(self):
        cfg = M.head_config(fake_checkpoint(PUBLISHED_CFG))
        self.assertEqual(M.expected_shapes(cfg), {
            "inp.weight": (1536, 4096), "inp.bias": (1536,),
            "hidden.0.weight": (1536, 1536), "hidden.0.bias": (1536,),
            "norms.0.weight": (1536,), "norms.0.bias": (1536,),
            "out.weight": (512, 1536), "out.bias": (512,)})

    def test_no_layernorm_means_no_norm_tensors_and_depth_sets_the_blocks(self):
        cfg = M.head_config(fake_checkpoint(dict(PUBLISHED_CFG, layernorm=False, depth=4)))
        names = set(M.expected_shapes(cfg))
        self.assertIn("hidden.1.weight", names)
        self.assertFalse(any(n.startswith("norms.") for n in names))

    def test_the_first_ports_names_are_not_what_the_converter_writes(self):
        cfg = M.head_config(fake_checkpoint(PUBLISHED_CFG))
        self.assertFalse({"0.weight", "2.weight", "4.weight", "6.weight"} & set(M.expected_shapes(cfg)))


class Refusals(unittest.TestCase):
    def test_a_missing_tensor_is_refused(self):
        ck = fake_checkpoint(PUBLISHED_CFG)
        del ck["action_head"]["norms.0.bias"]
        with self.assertRaisesRegex(M.ConvertError, "action_head: missing norms.0.bias"):
            M.head_tensors(ck, M.head_config(ck))

    def test_an_extra_tensor_is_refused(self):
        ck = fake_checkpoint(PUBLISHED_CFG)
        ck["state_head"]["hidden.1.weight"] = Shape(1536, 1536)
        with self.assertRaisesRegex(M.ConvertError, "unexpected tensor hidden.1.weight"):
            M.head_tensors(ck, M.head_config(ck))

    def test_a_transposed_tensor_is_refused(self):
        ck = fake_checkpoint(PUBLISHED_CFG)
        ck["state_head"]["out.weight"] = Shape(1536, 512)
        with self.assertRaisesRegex(M.ConvertError, "out.weight: shape"):
            M.head_tensors(ck, M.head_config(ck))

    def test_an_unknown_activation_and_a_non_checkpoint_are_refused(self):
        with self.assertRaisesRegex(M.ConvertError, "unsupported activation"):
            M.head_config(fake_checkpoint(dict(PUBLISHED_CFG, activation="tanh")))
        with self.assertRaisesRegex(M.ConvertError, "not a CLM checkpoint"):
            M.head_config({"state_dict": {}})

    def test_a_base_of_another_width_or_architecture_is_refused(self):
        cfg = M.head_config(fake_checkpoint(PUBLISHED_CFG))
        with self.assertRaisesRegex(M.ConvertError, "hidden_size is 2560"):
            M.build_config({"architectures": ["Qwen3ForCausalLM"], "hidden_size": 2560},
                           cfg, 4.6, {})
        with self.assertRaisesRegex(M.ConvertError, "Qwen3ForCausalLM"):
            M.build_config({"architectures": ["LlamaForCausalLM"], "hidden_size": 4096},
                           cfg, 4.6, {})


class Config(unittest.TestCase):
    def test_config_names_the_architecture_and_keeps_the_raw_logit_scale(self):
        cfg = M.head_config(fake_checkpoint(PUBLISHED_CFG))
        out = M.build_config({"architectures": ["Qwen3ForCausalLM"], "hidden_size": 4096,
                              "num_hidden_layers": 36}, cfg, 4.613248825073242, {"x": 1})
        self.assertEqual(out["architectures"], ["ClmModel"])
        self.assertEqual(out["num_hidden_layers"], 36)
        # The engine applies exp(.).clamp(max=100); a pre-applied 100.0 here
        # would be exponentiated a second time.
        self.assertEqual(out["clm_logit_scale"], 4.613248825073242)
        self.assertEqual((out["clm_width"], out["clm_depth"], out["clm_projection_dim"],
                          out["clm_activation"], out["clm_layernorm"], out["clm_residual"],
                          out["clm_hidden_size"]),
                         (1536, 3, 512, "gelu", True, False, 4096))


try:
    import torch
    from safetensors.torch import load_file, save_file
except ImportError:  # pragma: no cover - CI without torch
    torch = None


@unittest.skipUnless(torch is not None, "torch is not installed")
class Convert(unittest.TestCase):
    def tiny(self, d):
        cfg = dict(PUBLISHED_CFG, hidden_size=8, width=6, projection_dim=4)
        g = torch.Generator().manual_seed(0)
        shapes = M.expected_shapes(M.head_config(
            {"state_head": {}, "action_head": {}, "logit_scale": 0.0, "cfg": cfg}))
        ck = {"state_head": {k: torch.randn(*v, generator=g) for k, v in shapes.items()},
              "action_head": {k: torch.randn(*v, generator=g) for k, v in shapes.items()},
              "logit_scale": torch.tensor(4.613248825073242), "cfg": cfg}
        torch.save(ck, d / M.CHECKPOINT_NAME)
        base = d / "base"
        base.mkdir()
        (base / "config.json").write_text(json.dumps(
            {"architectures": ["Qwen3ForCausalLM"], "hidden_size": 8}))
        save_file({"model.norm.weight": torch.ones(8)}, str(base / "model.safetensors"))
        (base / "tokenizer.json").write_text("{}")
        return ck, base

    def test_a_tiny_checkpoint_converts_with_the_names_and_values_the_loader_reads(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ck, base = self.tiny(d)
            out = d / "out"
            M.convert(d, base, out, allow_other_checkpoint=True)
            heads = load_file(str(out / "head.safetensors"))
            self.assertEqual(len(heads), 16)
            for head in M.HEADS:
                for name, t in ck[head].items():
                    self.assertTrue(torch.equal(heads[f"{head}.{name}"], t.float()))
                    self.assertEqual(heads[f"{head}.{name}"].dtype, torch.float32)
            cfg = json.loads((out / "config.json").read_text())
            self.assertEqual(cfg["architectures"], ["ClmModel"])
            self.assertAlmostEqual(cfg["clm_logit_scale"], 4.613248825073242, places=6)
            self.assertTrue((out / "model.safetensors").is_file())
            self.assertTrue((out / "tokenizer.json").is_file())

    def test_an_unverified_checkpoint_needs_the_explicit_flag(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _, base = self.tiny(d)
            with self.assertRaisesRegex(M.ConvertError, "not the verified"):
                M.convert(d, base, d / "out")


if __name__ == "__main__":
    unittest.main()
