#!/usr/bin/env python3
"""Focused checks of observer delegation and original-repeat comparisons."""
import math
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/exl3_reference"))
from capture_target import compare_repeats, observe_logits, probability_metrics, target_phases, load_trace, validate_prefix_witnesses, selected_block_layers, deterministic_ba_forward, selected_detail_kind
from extract_projection import write_safetensors
from compare_target import compare_row, write_comparison_report, parse_args
from capture_block import select_gdn_layer
from capture_gdn import block_layer_index
from capture_target import validate_gdn_history, selected_boundary_phase
from capture_target import load_prompt, validate_all_gdn_states, all_target_gdn_layers
from capture_target import validate_all_attention_kv


class TargetCaptureTest(unittest.TestCase):
    def test_all_attention_kv_requires_bounded_all_gdn_state_mode(self):
        validate_all_attention_kv(True, True)
        validate_all_attention_kv(True, False)
        validate_all_attention_kv(False, False)
        for gdn, kv in [(False, True), (1, True), (True, 1), (True, "true")]:
            with self.assertRaises(ValueError): validate_all_attention_kv(gdn, kv)

    def test_held_out_prompt_preserves_legacy_and_rejects_unbounded_inputs(self):
        self.assertEqual(load_prompt(None), [1000 + (i * 37) % 4096 for i in range(128)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.json"
            ids = list(range(128))
            path.write_text(json.dumps({"prompt_token_ids": ids}))
            self.assertEqual(load_prompt(path), ids)
            for invalid in (ids[:-1], ids + [128], [-1] + ids[1:], [248320] + ids[1:],
                            [True] + ids[1:], ["1"] + ids[1:], {"count": 128}):
                path.write_text(json.dumps({"prompt_token_ids": invalid}))
                with self.assertRaises(ValueError): load_prompt(path)

    def test_all_gdn_state_scope_and_complete_target_selection(self):
        validate_all_gdn_states(1, -1, -1, False, True)
        validate_all_gdn_states(64, -1, -1, False, False)
        for steps, block, detail, history, enabled in [(64, -1, -1, False, True),
                (29, 29, 21, True, True), (1, 29, -1, False, True),
                (1, -1, 1, False, True), (1, -1, -1, True, True),
                (1, -1, -1, False, 1)]:
            with self.assertRaises(ValueError):
                validate_all_gdn_states(steps, block, detail, history, enabled)
        modules = {f"language_model.model.layers.{i}": SimpleNamespace(
            layer_type="full_attention" if i % 4 == 3 else "linear_attention")
            for i in reversed(range(64))}
        modules["draft.layers.0"] = SimpleNamespace(layer_type="linear_attention")
        self.assertEqual([i for i, _ in all_target_gdn_layers(modules, "language_model.model.layers.0")],
                         [i for i in range(64) if i % 4 != 3])
        modules["language_model.model.layers.63"].layer_type = "linear_attention"
        with self.assertRaises(ValueError):
            all_target_gdn_layers(modules, "language_model.model.layers.0")
        del modules["language_model.model.layers.63"]
        with self.assertRaises(ValueError):
            all_target_gdn_layers(modules, "language_model.model.layers.0")

    def test_early_boundaries_stop_at_gdn21_and_preserve_d29_selection(self):
        for index in range(64):
            self.assertEqual(selected_boundary_phase(29, index, 29, False), "d29")
            self.assertEqual(selected_boundary_phase(29, index, 29, True), "d29")
            self.assertEqual(selected_boundary_phase(0, index, 29, True),
                             "p128" if index <= 21 else None)
            self.assertIsNone(selected_boundary_phase(0, index, 29, False))
            for step in (1, 2, 28, 30):
                self.assertIsNone(selected_boundary_phase(step, index, 29, True))

    def test_gdn_history_requires_selected_d29_gdn21(self):
        validate_gdn_history(-1, -1, False)
        validate_gdn_history(29, 21, True)
        for step, layer, enabled in [(29, 1, True), (29, 3, True), (1, 21, True),
                                     (-1, 21, True), (29, 21, 1)]:
            with self.assertRaises(ValueError): validate_gdn_history(step, layer, enabled)

    def test_selected_detail_requires_d29_and_exact_supported_layer(self):
        self.assertEqual(selected_detail_kind(-1, -1), "none")
        self.assertEqual(selected_detail_kind(29, 1), "gdn")
        self.assertEqual(selected_detail_kind(29, 3), "attention")
        self.assertEqual(selected_detail_kind(29, 21), "gdn")
        for step, layer in [(-1, 1), (-1, 3), (-1, 21), (1, 3), (1, 21),
                            (29, 2), (29, True), (29, "3")]:
            with self.assertRaises(ValueError): selected_detail_kind(step, layer)

    def test_deterministic_ba_scope_delegates_and_restores_success_and_failure(self):
        setting = [False, True]
        changes, calls = [], []
        value, extra, result = object(), object(), object()
        def read(): return tuple(setting)
        def write(state):
            setting[:] = state; changes.append(state)
        def original(*args, **kwargs):
            self.assertEqual(setting, [True, False])
            calls.append((args, kwargs)); return result
        wrapped = deterministic_ba_forward(original, read, write)
        self.assertIs(wrapped(value, option=extra), result)
        self.assertEqual(calls, [((value,), {"option": extra})])
        self.assertEqual(changes, [(True, False), (False, True)])
        self.assertEqual(setting, [False, True])
        def failed(*args):
            self.assertEqual(setting, [True, False]); raise RuntimeError("original failed")
        with self.assertRaisesRegex(RuntimeError, "original failed"):
            deterministic_ba_forward(failed, read, write)(value)
        self.assertEqual(setting, [False, True])

    def test_gdn_replay_keeps_legacy_layer_and_validates_selected_weights(self):
        self.assertEqual(block_layer_index({}), 0)
        self.assertEqual(block_layer_index({"layer_index": 1}), 1)
        self.assertEqual(block_layer_index({"layer_index": 21}), 21)
        for index in (-1, 2, True, "1"):
            with self.assertRaises(ValueError): block_layer_index({"layer_index": index})

    def test_block_selection_rejects_missing_duplicate_or_non_gdn_target(self):
        gdn = SimpleNamespace(layer_type="linear_attention")
        attn = SimpleNamespace(layer_type="full_attention")
        modules = [("language_model.model.layers.0", gdn),
                   ("language_model.model.layers.1", gdn), ("draft.layers.1", attn),
                   ("language_model.model.layers.21", gdn)]
        self.assertEqual(select_gdn_layer(modules, 1), (modules[1]))
        self.assertEqual(select_gdn_layer(modules, 0), (modules[0]))
        self.assertEqual(select_gdn_layer(modules, 21), (modules[3]))
        for entries, index in ((modules[:1], 1), (modules + [modules[1]], 1),
                               ([(modules[1][0], attn)], 1), (modules, 2), (modules, True)):
            with self.assertRaises(ValueError): select_gdn_layer(entries, index)

    def test_d29_selection_requires_complete_target_blocks_and_exact_trace(self):
        modules = {f"language_model.model.layers.{i}": object() for i in reversed(range(64))}
        modules["draft.layers.0"] = object()
        layers = selected_block_layers(modules, "language_model.model.layers.0")
        self.assertEqual([i for i, _ in layers], list(range(64)))
        del modules["language_model.model.layers.63"]
        with self.assertRaises(ValueError): selected_block_layers(modules, "language_model.model.layers.0")
        self.assertEqual(target_phases(29), ["p128"] + [f"d{i}" for i in range(1, 30)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps({"output_ids": [list(range(30))]}))
            self.assertEqual(load_trace(path, 29), list(range(30)))
            with self.assertRaises(ValueError): load_trace(path, 64)

    def test_observer_preserves_arguments_result_and_single_original_call(self):
        hidden, result, extra = object(), object(), object()
        calls, copies = [], []
        def original(value, *args, **kwargs):
            calls.append((value, args, kwargs)); return result
        observer = observe_logits(original, lambda h, r: copies.append((h, r)))
        self.assertIs(observer(hidden, extra, option=extra), result)
        self.assertEqual(calls, [(hidden, (extra,), {"option": extra})])
        self.assertEqual(copies, [(hidden, result)])

    def test_original_failure_does_not_capture_a_fabricated_result(self):
        copies = []
        def original(hidden): raise RuntimeError("original failure")
        with self.assertRaisesRegex(RuntimeError, "original failure"):
            observe_logits(original, lambda *args: copies.append(args))(object())
        self.assertEqual(copies, [])

    def test_distribution_metrics_known_probabilities_and_logit_offset(self):
        r = probability_metrics([math.log(3), 0], [0, 0])
        self.assertAlmostEqual(r["TV"], 0.25)
        self.assertAlmostEqual(r["KL_reference_to_actual"], 0.5 * math.log(4/3))
        self.assertTrue(r["investigation_trigger"])
        r = probability_metrics([1000, 1001, 1002], [0, 1, 2])
        self.assertAlmostEqual(r["TV"], 0)
        self.assertAlmostEqual(r["KL_reference_to_actual"], 0)
        self.assertFalse(r["investigation_trigger"])

    def test_invalid_logit_contracts_are_rejected(self):
        for a,b in [([], []), ([1], [1, 2]), ([math.nan], [1]), ([1], [math.inf])]:
            with self.assertRaises(ValueError): probability_metrics(a, b)

    def test_repeat_comparison_keeps_fixed_state_band_and_all_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i in range(3):
                tensors = {}
                for phase in ("p128", "d1"):
                    for label in ("ba_output", "ssm_state_after", "hidden_out", "logits_hidden", "logits"):
                        values = [0., 1.]
                        if label == "ssm_state_after" and i == 2: values[0] = 2e-5
                        tensors[phase+"_"+label] = ("F32", [1,2], struct.pack("<2f", *values))
                path = Path(directory) / f"{i}.safetensors"
                write_safetensors(path, tensors, {})
                paths.append(path)
            comparisons = compare_repeats(paths)
            self.assertEqual([(r["reference_repeat"], r["actual_repeat"]) for r in comparisons],
                             [(0,1), (0,2), (1,2)])
            for phase in ("p128", "d1"):
                self.assertEqual(comparisons[0]["stages"][phase+"_ssm_state_after"]["fixed_pointwise_failures"], 0)
                self.assertEqual(comparisons[1]["stages"][phase+"_ssm_state_after"]["fixed_pointwise_failures"], 1)
                self.assertFalse(comparisons[1]["stages"][phase+"_logits"]["distribution"]["investigation_trigger"])

    def test_d64_comparison_reads_all_heads_without_fabricated_later_block_states(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i in range(2):
                tensors = {}
                for phase in target_phases(64):
                    labels = ("ba_output", "ssm_state_after", "hidden_out") if phase in ("p128", "d1") else ()
                    for label in labels + ("logits_hidden", "logits"):
                        tensors[phase+"_"+label] = ("F32", [1,2], struct.pack("<2f", 0, 1))
                path = Path(directory) / f"{i}.safetensors"
                write_safetensors(path, tensors, {}); paths.append(path)
            stages = compare_repeats(paths, 64)[0]["stages"]
            self.assertEqual(len(stages), 65 * 2 + 2 * 3)
            self.assertIn("d64_logits", stages)
            self.assertNotIn("d2_ssm_state_after", stages)
            self.assertFalse(stages["d64_logits"]["distribution"]["investigation_trigger"])
        for steps in (0, 2, 65):
            with self.assertRaises(ValueError): target_phases(steps)

    def test_trace_keeps_recorded_prefix_and_rejects_invalid_counts_or_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps({"output_ids": [[13,198]]}))
            self.assertEqual(load_trace(path, 1), [13,198])
            for ids in ([[13]], [[13,-1]], [[13,248320]], [[13,True]], [[13,198],[13,198]]):
                path.write_text(json.dumps({"output_ids": ids}))
                with self.assertRaises(ValueError): load_trace(path, 1)

    def test_native_rows_require_complete_f32_and_compare_both_reference_dtypes(self):
        native = struct.pack("<2f", 0, 1)
        for dtype, expected in (("F32", native), ("F16", struct.pack("<2e", 0, 1))):
            self.assertEqual(compare_row(native, expected, dtype, 2)["TV"], 0)
            with self.assertRaises(ValueError): compare_row(native[:-1], expected, dtype, 2)
            with self.assertRaises(ValueError): compare_row(native, expected[:-1], dtype, 2)
        with self.assertRaises(ValueError): compare_row(struct.pack("<2f", math.nan, 1), native, "F32", 2)

    def test_actual_input_witnesses_check_every_token_position_and_missing_steps(self):
        witnesses = {"p128": {"token_ids": [7,8], "positions": [[0,1]]*3},
                     "d1": {"token_ids": [13], "positions": [[2]]*3}}
        validate_prefix_witnesses(witnesses, [7,8], [13,198])
        for key, bad in (("token_ids", [198]), ("positions", [[1]]*3)):
            changed = json.loads(json.dumps(witnesses)); changed["d1"][key] = bad
            with self.assertRaises(ValueError): validate_prefix_witnesses(changed, [7,8], [13,198])
        with self.assertRaises(ValueError): validate_prefix_witnesses({"p128": witnesses["p128"]}, [7,8], [13,198])

    def test_gate_failure_and_report_only_keep_the_same_failed_metrics(self):
        failed = {"phase": "d29", **probability_metrics([math.log(3), 0], [0, 0])}
        with tempfile.TemporaryDirectory() as directory:
            records = []
            for mode, code in (("gate", 1), ("report_only", 0)):
                report = Path(directory) / (mode + ".json")
                self.assertEqual(write_comparison_report(report, {"comparisons": [failed]}, mode), code)
                record = json.loads(report.read_text()); records.append(record)
                self.assertEqual(record["status"], "investigation_required")
                self.assertFalse(record["logit_gate_pass"])
                self.assertFalse(record["target_parity_pass"])
                self.assertFalse(record["serving_qualified"])
                self.assertEqual(record["gate_exit_code"], 1)
                self.assertEqual(record["process_exit_code"], code)
                self.assertEqual(record["blocking_issue_ids"], ["S1_D64"])
                with self.assertRaises(FileExistsError):
                    write_comparison_report(report, {"comparisons": [failed]}, mode)
            self.assertEqual(records[0]["comparisons"], records[1]["comparisons"])
            self.assertEqual(records[0]["investigation_triggers"], records[1]["investigation_triggers"])

    def test_passing_logits_do_not_claim_state_or_serving_qualification(self):
        passed = probability_metrics([0, 1, 2], [0, 1, 2])
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "pass.json"
            self.assertEqual(write_comparison_report(report, {"comparisons": [passed]}, "gate"), 0)
            record = json.loads(report.read_text())
            self.assertTrue(record["logit_gate_pass"])
            self.assertEqual(record["status"], "passed_bounded_logit_comparison")
            self.assertIsNone(record["target_parity_pass"])
            self.assertIsNone(record["autonomous_smoke_pass"])
            self.assertFalse(record["serving_qualified"])
            with self.assertRaises(ValueError):
                write_comparison_report(Path(directory) / "empty.json", {"comparisons": []}, "gate")
            self.assertFalse((Path(directory) / "empty.json").exists())

    def test_cli_defaults_to_gate_and_refuses_conflicting_modes(self):
        argv = ["--native-prefix", "native", "--native-trace-json", "trace.json",
                "--reference-dir", "reference", "--report", "report.json"]
        self.assertEqual(parse_args(argv).mode, "gate")
        self.assertEqual(parse_args(argv + ["--report-only"]).mode, "report_only")
        self.assertEqual(parse_args(argv + ["--gate"]).mode, "gate")
        with self.assertRaises(SystemExit) as failure:
            parse_args(argv + ["--gate", "--report-only"])
        self.assertEqual(failure.exception.code, 2)


if __name__ == "__main__": unittest.main()
