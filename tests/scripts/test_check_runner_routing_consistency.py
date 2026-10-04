#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-runner-routing-consistency.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-runner-routing-consistency.py"
ALLOWLIST = ROOT / "scripts/runner-routing-allowlist.txt"
BF16_ALLOWLIST = ROOT / "scripts/runner-bf16-activation-allowlist.txt"
SPEC = importlib.util.spec_from_file_location("check_runner_routing_consistency", CHECKER)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod  # register BEFORE exec so the frozen dataclass resolves
SPEC.loader.exec_module(mod)

ModelRoute = mod.ModelRoute
drift_models = mod.drift_models
f32_stream_drift_models = mod.f32_stream_drift_models


def route(name: str, classification: str, private_loop: bool = False,
          activation: str = "BF16_RESIDENT") -> ModelRoute:
    return ModelRoute(name, f"{name}_registry.cpp", f"Forward{name}", classification,
                      private_loop, activation=activation)


class DriftModelTests(unittest.TestCase):
    def test_device_model_passes(self) -> None:
        # device-resident logits on the runner => clean.
        self.assertEqual(drift_models({"qwen3_dense": route("qwen3_dense", "DEVICE")}, set()), [])

    def test_host_model_fails(self) -> None:
        # host logits off the sampler, not allowlisted => drift.
        self.assertEqual(drift_models({"laguna": route("laguna", "HOST")}, set()), ["laguna"])

    def test_allowlisted_host_passes(self) -> None:
        self.assertEqual(drift_models({"laguna": route("laguna", "HOST")}, {"laguna"}), [])

    def test_refuse_stub_never_trips(self) -> None:
        # A REFUSE-by-name stub (VT_CHECK(false)) decodes nothing => never drifts.
        self.assertEqual(drift_models({"kimi_k3": route("kimi_k3", "REFUSE")}, set()), [])

    def test_none_never_trips(self) -> None:
        # No recognizable logit producer is not treated as HOST drift.
        self.assertEqual(drift_models({"x": route("x", "NONE")}, set()), [])

    def test_mixed_reports_only_host_uncovered(self) -> None:
        result = drift_models(
            {
                "qwen3_dense": route("qwen3_dense", "DEVICE"),  # clean
                "laguna": route("laguna", "HOST"),              # drift
                "deepseek_v4": route("deepseek_v4", "HOST"),    # allowlisted
                "kimi_k3": route("kimi_k3", "REFUSE"),          # skip
            },
            allowlisted={"deepseek_v4"},
        )
        self.assertEqual(result, ["laguna"])

    def test_classify_body(self) -> None:
        device = "{ return WrapDeviceLogits(d, std::move(dl), n, v); }"
        view = "{ ForwardLogits fl; fl.device_tensor = t; return fl; }"
        host = "{ return HostLogits(std::move(logits), vocab); }"
        host_field = "{ ForwardLogits out; out.host = std::move(flat); return out; }"
        refuse = "{ VT_CHECK(false, kPending); return {}; }"
        self.assertEqual(mod.classify_body(device), "DEVICE")
        self.assertEqual(mod.classify_body(view), "DEVICE")
        self.assertEqual(mod.classify_body(host), "HOST")
        self.assertEqual(mod.classify_body(host_field), "HOST")
        self.assertEqual(mod.classify_body(refuse), "REFUSE")
        self.assertEqual(mod.classify_body(None), "NONE")

    def test_classify_body_device_wins_over_comment(self) -> None:
        # A comment mentioning HostLogits inside a device forward must not misclassify.
        body = "{ /* returns HostLogits on the opt-out */ return WrapDeviceLogits(x); }"
        self.assertEqual(mod.classify_body(body), "DEVICE")

    def test_extract_fn_body_matches_definition_not_call(self) -> None:
        text = (
            "ForwardLogits ForwardFoo(LoadedModel& m, const ModelForwardInput& in) {\n"
            "  if (in.gather_logits) return FooModel::ForwardDevice(in);\n"
            "  return HostLogits(FooModel::Forward(in), v);\n"
            "}\n"
            "// elsewhere a call: ForwardFoo(model, input);\n"
        )
        body = mod.extract_fn_body(text, "ForwardFoo")
        self.assertIsNotNone(body)
        self.assertIn("FooModel::ForwardDevice", body)
        self.assertIn("HostLogits", body)

    def test_resolve_alias(self) -> None:
        alias = {"LlamaModel": "Qwen3DenseModel", "MistralModel": "Qwen3DenseModel"}
        self.assertEqual(mod.resolve_alias("LlamaModel", alias), "Qwen3DenseModel")
        self.assertEqual(mod.resolve_alias("GemmaModel", alias), "GemmaModel")

    def test_allowlist_parsing(self) -> None:
        text = "# comment\nlaguna  # trailing reason\nqwen3_vl\n\n"
        self.assertEqual(mod.allowlisted_names(text), {"laguna", "qwen3_vl"})

    def test_shipped_tree_is_green(self) -> None:
        # The real repo must pass: every registered model is device-resident or
        # allowlisted (or a refuse stub).
        scanned = mod.scan_registrations(mod.MODELS_DIR, mod.INCLUDE_DIR)
        allowlisted = mod.allowlisted_names(ALLOWLIST.read_text(encoding="utf-8"))
        self.assertEqual(drift_models(scanned, allowlisted), [])
        # the sweep actually found the registrations (>20 registered models)
        self.assertGreater(len(scanned), 20)
        # and it really classified device-resident models (not everything is HOST)
        self.assertGreater(
            sum(1 for r in scanned.values() if r.classification == "DEVICE"), 15
        )

    def test_known_off_framework_are_host(self) -> None:
        # The off-framework models the audit found must classify HOST (so the
        # allowlist is load-bearing, not decorative). deepseek_v4 was the third; it
        # was ROUTED on-framework (device-resident logits on the registry forward)
        # and its allowlist entry retired, so it is asserted DEVICE below instead.
        scanned = mod.scan_registrations(mod.MODELS_DIR, mod.INCLUDE_DIR)
        for name in ("laguna", "qwen3_vl"):
            self.assertIn(name, scanned)
            self.assertEqual(scanned[name].classification, "HOST", name)
        # Qwen3-VL additionally ships the private *GenerateCore host loop (inv. b).
        self.assertTrue(scanned["qwen3_vl"].private_generate_loop)
        # A clean model that keeps a *GenerateCore example helper (qwen3_5) is NOT
        # flagged — DEVICE classification means it never enters the (b) gate.
        self.assertEqual(scanned["qwen3_5_moe"].classification, "DEVICE")

    def test_private_device_wrapper_classifies_device(self) -> None:
        # REGRESSION (the hole this closed): a model whose ForwardDevice builds its
        # device carrier in its OWN ForwardLogits helper rather than the shared
        # WrapDeviceLogits matched NEITHER seam and classified NONE. NONE is not an
        # error state, so the model dropped out of the drift check and the gate went
        # green while silently exempting it. deepseek_v4 (WrapV4DeviceLogits) is the
        # tree's live case.
        scanned = mod.scan_registrations(mod.MODELS_DIR, mod.INCLUDE_DIR)
        self.assertEqual(scanned["deepseek_v4"].classification, "DEVICE")
        # CROSS-TU DELEGATION (the hole this closes): mimo_v2's registry TU DECLARES
        # ForwardMiMoV2Device but DEFINES it in mimo_v2.cpp, where all three lm_head
        # arms return the shared WrapDeviceLogits. The registry-local one-level hop
        # resolves only the declaration, matched NEITHER seam, and the model landed
        # in the silently-exempt NONE bucket. The scanner now follows a free function
        # the hook calls to the sibling models/*.cpp that defines it.
        self.assertEqual(scanned["mimo_v2"].classification, "DEVICE")

        # ARCH-ONE-SURFACE ROW 6: a registry TU declaring `.is_pooling_model =
        # true` is a NON-GENERATIVE registration — a hidden-state producer for
        # the PoolingRunner, classified POOLING explicitly (deleting the
        # checker's pooling arm would drop it into NONE and red the bucket pin
        # below).
        self.assertEqual(scanned["llama_embedding"].classification, "POOLING")
        # No registered model may sit in the silently-exempt NONE bucket at all.
        self.assertEqual(
            sorted(n for n, r in scanned.items() if r.classification == "NONE"), []
        )

    def test_helper_hop_resolves_both_ways(self) -> None:
        # Mutation on synthetic text: the one-level hop must LIFT a private DEVICE
        # wrapper to DEVICE, and must NOT launder a HOST wrapper into DEVICE.
        body = "{ return WrapMine(std::move(flat)); }"
        self.assertEqual(mod.classify_body(body), "NONE")  # RED without the hop
        device_text = (
            "static ForwardLogits WrapMine(std::vector<float>&& f) {\n"
            "  ForwardLogits fl;\n"
            "  fl.device_tensor = vt::Tensor::Contiguous(f.data());\n"
            "  return fl;\n"
            "}\n"
        )
        self.assertEqual(mod.classify_with_helpers(body, device_text), "DEVICE")
        host_text = (
            "static ForwardLogits WrapMine(std::vector<float>&& f) {\n"
            "  ForwardLogits fl;\n"
            "  fl.host = std::move(f);\n"
            "  return fl;\n"
            "}\n"
        )
        self.assertEqual(mod.classify_with_helpers(body, host_text), "HOST")

    def test_hook_cross_tu_hop_resolves_and_not_launders(self) -> None:
        # Mutation on synthetic text: the cross-TU hop must lift a hook whose only
        # local match is a forward DECLARATION to DEVICE via the sibling TU, and
        # must NOT launder a host-producing sibling definition into DEVICE.
        decl = "ForwardLogits ForwardMine(const Input& in);\n"
        hook = "{ return ForwardMine(input); }"
        self.assertEqual(mod.classify_body(hook), "NONE")
        device_files = {
            "registry.cpp": decl,
            "model.cpp": (
                "ForwardLogits ForwardMine(const Input& in) {\n"
                "  return WrapDeviceLogits(d, n, v);\n"
                "}\n"
            ),
        }
        self.assertEqual(
            mod.classify_hook(hook, decl, device_files, {"ForwardMine": "model.cpp"}),
            "DEVICE",
        )
        host_files = {
            "registry.cpp": decl,
            "model.cpp": (
                "ForwardLogits ForwardMine(const Input& in) {\n"
                "  return HostLogits(std::move(flat), v);\n"
                "}\n"
            ),
        }
        self.assertEqual(
            mod.classify_hook(hook, decl, host_files, {"ForwardMine": "model.cpp"}),
            "HOST",
        )
        # A name with no definition anywhere stays NONE (an error state).
        self.assertEqual(
            mod.classify_hook(hook, decl, {"registry.cpp": decl}, {}), "NONE"
        )

    def test_refuse_stub_is_skipped_on_tree(self) -> None:
        # kimi_k3's ForwardDevice is VT_CHECK(false); it must be REFUSE, not HOST.
        scanned = mod.scan_registrations(mod.MODELS_DIR, mod.INCLUDE_DIR)
        self.assertEqual(scanned["kimi_k3"].classification, "REFUSE")
        self.assertNotIn("kimi_k3", drift_models(scanned, set()))

    def test_a_new_off_runner_model_would_fail(self) -> None:
        # Mutation: a NEW model landing with a HostLogits forward and no allowlist
        # entry must trip the gate.
        scanned = dict(mod.scan_registrations(mod.MODELS_DIR, mod.INCLUDE_DIR))
        allowlisted = mod.allowlisted_names(ALLOWLIST.read_text(encoding="utf-8"))
        scanned["brand_new_arch"] = route("brand_new_arch", "HOST")
        self.assertIn("brand_new_arch", drift_models(scanned, allowlisted))

    def test_removing_allowlist_entry_would_fail(self) -> None:
        # Mutation: dropping a known off-framework model from the allowlist WITHOUT
        # routing it through the runner must re-open the gate (the enforcement teeth).
        # The example is DERIVED from the tree, never hardcoded, so routing a model
        # (retiring its entry) closes the gate rather than breaking this test.
        scanned = mod.scan_registrations(mod.MODELS_DIR, mod.INCLUDE_DIR)
        allowlisted = mod.allowlisted_names(ALLOWLIST.read_text(encoding="utf-8"))
        exposed = set(drift_models(scanned, set()))
        host_models = {n for n, r in scanned.items() if r.classification == "HOST"}
        # Emptying the allowlist must expose every HOST model,
        self.assertEqual(host_models - exposed, set())
        # and the allowlist must be load-bearing: it suppresses at least one model
        # the checker really detects (else the green gate is vacuous).
        self.assertTrue(
            exposed & allowlisted,
            "the allowlist suppresses nothing the checker detects; either the "
            "detector regressed or every allowlist entry is now stale",
        )


class Bf16ActivationTests(unittest.TestCase):
    """Invariant (c): bf16-resident activations vs a hand-rolled f32 host stream."""

    # --- pure classifier on synthetic text -----------------------------------
    def test_classify_activation_f32_stream(self) -> None:
        # A decode that declares several private f32 residual-stream buffers and casts
        # to bf16 before every projection, with NO shared AttnBlock and NO bf16-DBuf
        # residual, is F32_STREAM.
        text = (
            "std::vector<float> hidden(T * H);\n"
            "std::vector<float> residual(T * H);\n"
            "std::vector<float> x(T * H);\n"
            "for (int64_t l = 0; l < nlayers; ++l) {\n"
            "  CastBf16(q, dh.t(), hidden.data());\n"
            "  GemmBf16Into(qkv.data(), lw.attn.qkv, hidden.data(), n, H);\n"
            "}\n"
        )
        self.assertEqual(mod.classify_activation_stream(text), "F32_STREAM")
        self.assertGreaterEqual(mod.count_f32_resid_decls(text), mod.MIN_F32_RESID)

    def test_classify_activation_bf16_via_attnblock(self) -> None:
        # A decode that routes the residual through the shared dense_attn::AttnBlock
        # preamble is BF16_RESIDENT even if it keeps an f32 scratch buffer.
        text = (
            "std::vector<float> hidden(T * H);\n"  # a lone scratch, below threshold
            "DBuf x = dense_attn::AttnBlock(d, w.attn, cfg, dh.t(), positions);\n"
        )
        self.assertEqual(mod.classify_activation_stream(text), "BF16_RESIDENT")
        self.assertTrue(mod.is_bf16_resident(text))

    def test_classify_activation_bf16_via_dbuf_residual(self) -> None:
        # A bf16-DBuf residual stream (DBuf hidden ... kBF16) is BF16_RESIDENT even
        # alongside many f32 host vectors (the f32 are mm-prefill helpers, not the
        # decode residual) — the qwen3_vl shape.
        text = (
            "std::vector<float> embeds(T * H);\n"
            "std::vector<float> residual(T * H);\n"
            "std::vector<float> x(T * H);\n"
            "DBuf hidden(d, DType::kBF16, {T, H}, embeds.data());\n"
            "DBuf res(d, DType::kBF16, {T, H});\n"
        )
        self.assertEqual(mod.classify_activation_stream(text), "BF16_RESIDENT")

    def test_classify_activation_bf16_below_threshold(self) -> None:
        # A lone f32 scratch var (below MIN_F32_RESID) is not an f32 stream.
        text = "std::vector<float> x(H);\n"
        self.assertEqual(mod.classify_activation_stream(text), "BF16_RESIDENT")

    def test_scratch_bf16_dbuf_does_not_exempt(self) -> None:
        # NARROWNESS: a per-op bf16 SCRATCH DBuf (laguna's `DBuf dh(... kBF16)`) must
        # NOT exempt an f32-stream decode — only a residual-NAMED bf16 DBuf does.
        text = (
            "std::vector<float> hidden(T * H);\n"
            "std::vector<float> residual(T * H);\n"
            "std::vector<float> hn(T * H);\n"
            "DBuf dh(d, DType::kBF16, {1, H});\n"        # scratch, not residual-named
            "DBuf dact(d, DType::kBF16, {Pk, moe_I});\n"  # scratch
        )
        self.assertFalse(mod.is_bf16_resident(text))
        self.assertEqual(mod.classify_activation_stream(text), "F32_STREAM")

    # --- pure drift function --------------------------------------------------
    def test_f32_drift_fires_unallowlisted(self) -> None:
        s = {"laguna": route("laguna", "HOST", activation="F32_STREAM")}
        self.assertEqual(f32_stream_drift_models(s, set()), ["laguna"])

    def test_f32_drift_allowlisted_passes(self) -> None:
        s = {"laguna": route("laguna", "HOST", activation="F32_STREAM")}
        self.assertEqual(f32_stream_drift_models(s, {"laguna"}), [])

    def test_bf16_resident_never_drifts(self) -> None:
        s = {"qwen3_vl": route("qwen3_vl", "HOST", activation="BF16_RESIDENT")}
        self.assertEqual(f32_stream_drift_models(s, set()), [])

    def test_refuse_stub_never_f32_drifts(self) -> None:
        # Even if a REFUSE stub somehow carried the f32 signal, it decodes nothing.
        s = {"kimi_k3": route("kimi_k3", "REFUSE", activation="F32_STREAM")}
        self.assertEqual(f32_stream_drift_models(s, set()), [])

    # --- on the shipped tree --------------------------------------------------
    def test_tree_f32_stream_membership(self) -> None:
        # Exactly laguna + deepseek_v4 hand-roll an f32 host stream on the tree; every
        # other registered model (incl. qwen3_vl, whose decode residual is a bf16 DBuf)
        # is BF16_RESIDENT. This axis is ORTHOGONAL to (a): deepseek_v4 is DEVICE on (a)
        # yet F32_STREAM here; qwen3_vl is HOST on (a) yet BF16_RESIDENT here.
        scanned = mod.scan_registrations(mod.MODELS_DIR, mod.INCLUDE_DIR)
        f32 = sorted(n for n, r in scanned.items() if r.activation == "F32_STREAM")
        self.assertEqual(f32, ["deepseek_v4", "laguna"])
        self.assertEqual(scanned["deepseek_v4"].classification, "DEVICE")   # clean on (a)
        self.assertEqual(scanned["deepseek_v4"].activation, "F32_STREAM")   # drift on (c)
        self.assertEqual(scanned["qwen3_vl"].classification, "HOST")        # drift on (a)
        self.assertEqual(scanned["qwen3_vl"].activation, "BF16_RESIDENT")   # clean on (c)
        # And it really is load-bearing: laguna casts hn->bf16 per projection (>0 sites),
        # deepseek_v4 keeps everything f32 (0 casts) — both are the same escape.
        self.assertGreater(scanned["laguna"].f32_resid_decls, mod.MIN_F32_RESID)
        self.assertGreater(scanned["deepseek_v4"].f32_resid_decls, mod.MIN_F32_RESID)

    def test_tree_bf16_activation_is_green(self) -> None:
        scanned = mod.scan_registrations(mod.MODELS_DIR, mod.INCLUDE_DIR)
        allow = mod.allowlisted_names(BF16_ALLOWLIST.read_text(encoding="utf-8"))
        self.assertEqual(f32_stream_drift_models(scanned, allow), [])
        # the bf16 allowlist is load-bearing (not decorative / stale).
        exposed = set(f32_stream_drift_models(scanned, set()))
        self.assertTrue(exposed & allow)
        self.assertEqual(exposed, {"laguna", "deepseek_v4"})

    def test_new_f32_stream_model_would_fail(self) -> None:
        # Mutation: a NEW model landing with an f32 host stream and no allowlist entry
        # must trip invariant (c).
        scanned = dict(mod.scan_registrations(mod.MODELS_DIR, mod.INCLUDE_DIR))
        allow = mod.allowlisted_names(BF16_ALLOWLIST.read_text(encoding="utf-8"))
        scanned["brand_new_arch"] = route("brand_new_arch", "DEVICE",
                                           activation="F32_STREAM")
        self.assertIn("brand_new_arch", f32_stream_drift_models(scanned, allow))

    def test_mutating_on_framework_to_f32_would_fail(self) -> None:
        # Mutation: flip a currently-clean on-framework model to F32_STREAM — the gate
        # must fire (proves the invariant is real, not vacuous). Derived from the tree.
        scanned = dict(mod.scan_registrations(mod.MODELS_DIR, mod.INCLUDE_DIR))
        allow = mod.allowlisted_names(BF16_ALLOWLIST.read_text(encoding="utf-8"))
        victim = next(n for n, r in scanned.items()
                      if r.activation == "BF16_RESIDENT"
                      and r.classification == "DEVICE" and n not in allow)
        r = scanned[victim]
        scanned[victim] = mod.ModelRoute(
            r.name, r.reg_file, r.forward_fn, r.classification,
            r.private_generate_loop, r.device_source, "F32_STREAM", 9, 9)
        self.assertIn(victim, f32_stream_drift_models(scanned, allow))


if __name__ == "__main__":
    unittest.main()
