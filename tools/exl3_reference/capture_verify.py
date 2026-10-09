#!/usr/bin/env python3
"""Observe one executed original C1/page1600/Q4 call; active cache bytes only."""
import argparse
import dataclasses
import functools
import json
import os
from pathlib import Path

from capture_runtime_layout import PROFILE, verify_inputs
from extract_projection import digest, headers, write_safetensors
from runtime_layout import tensor_layout


def active_page_slices(length, page, table):
    if type(length) is not int or length < 4 or page != 1600:
        raise ValueError("requires Q4 and page1600 active geometry")
    count = (length + page - 1) // page
    if len(table) < count or any(type(v) is not int or v < 0 for v in table[:count]):
        raise ValueError("missing or inactive active block")
    return [{"physical_block": table[i], "active_rows": min(page, length - i * page)}
            for i in range(count)]


class VerifyCaptureWorker:
    def install_verify_capture(self, output):
        import torch
        from exl3xpu import shared_kv_verify as verify
        from torch.utils._python_dispatch import TorchDispatchMode
        headers.require(not hasattr(self, "_verify_original"), "observer already installed")
        self._verify_original, self._verify_record = verify.run, None
        path = Path(output)
        headers.require(not path.exists(), "refusing existing fixture")
        worker = self

        @functools.wraps(self._verify_original)
        def observed(d, *args, **kwargs):
            selected = (worker._verify_record is None and verify.eligible(d) and
                        d["q"].shape == (4, 24, 256) and d["k"].shape[1] == 1600 and
                        d["max_seqlen_k"] >= 4096)
            if not selected:
                return worker._verify_original(d, *args, **kwargs)
            calls = []

            class Witness(TorchDispatchMode):
                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    if str(func) == "b70_exl3_attention.shared_kv_verify_out.default":
                        calls.append({"operator": str(func), "max_keys": args[-3],
                                      "splits": args[-2], "tile": args[-1]})
                    return func(*args, **(kwargs or {}))

            with Witness():
                result = worker._verify_original(d, *args, **kwargs)
            torch.xpu.synchronize()
            headers.require(len(calls) == 1, "actual original XPU verifier witness missing")
            tensors = {}

            def save(name, tensor):
                host = tensor.detach().cpu().contiguous()
                dtype = {torch.float16: "F16", torch.float32: "F32",
                         torch.uint8: "U8", torch.int32: "I32"}[host.dtype]
                tensors[name] = (dtype, list(host.shape), host.numpy().tobytes())

            save("query", d["q"])
            save("output", result)
            save("query_offsets", d["cu_seqlens_q"])
            save("lengths", d["seqused_k"])
            length = int(d["seqused_k"].detach().cpu().item())
            active = (length + 1599) // 1600
            table = d["block_table"][:, :active]
            save("block_table", table)
            pages = active_page_slices(length, 1600, table.cpu().tolist()[0])
            for i, item in enumerate(pages):
                block, rows = item["physical_block"], item["active_rows"]
                headers.require(block < d["k"].shape[0], "active block outside cache")
                for key in ("k", "v"):
                    save(f"{key}_page{i}", d[key][block, :rows].view(torch.uint8))
            for key in ("k_descale", "v_descale"):
                save(key, d[key].as_strided((1,), (1,)))
            # A read-only identical-input repeat, not a generic attention replacement.
            repeat = worker._verify_original(d, *args, **kwargs)
            repeat_raw = repeat.detach().cpu().contiguous().numpy().tobytes()
            headers.require(repeat_raw == tensors["output"][2], "original replay changed output")
            write_safetensors(path, tensors, {"scope": __doc__})
            worker._verify_record = {
                "route": calls[0], "worker_pid": os.getpid(), "active_length": length,
                "pages": pages, "scale": d["softmax_scale"], "causal": d["causal"],
                "layouts": {k: tensor_layout(d[k]) for k in
                            ("q", "k", "v", "block_table", "seqused_k", "cu_seqlens_q")},
                "source": {k: digest(Path(v).read_bytes()) for k, v in
                           {"shared_kv_verify": verify.__file__}.items()},
                "m04_library": {"path": os.environ["EXL3_M04_LIBRARY"],
                                "sha256": digest(Path(os.environ["EXL3_M04_LIBRARY"]).read_bytes())},
                "same_input_repeat_bit_exact": True, "capture_sha256": digest(path.read_bytes()),
                "inactive_cache_bytes_read": False,
                "scope": "Executed original Q4 C1; cache tensors include active rows only. "
                         "Preserve described strides when rebuilding poisoned fixture storage. "
                         "No serving latency or global numerical qualification claim.",
            }
            headers.require(worker._verify_record["m04_library"]["sha256"] ==
                            os.environ["EXL3_M04_LIBRARY_SHA256"], "M04 binary hash mismatch")
            return result

        verify.run = observed
        return {"observer": "one delegated original call plus identical-input repeat"}

    def finish_verify_capture(self):
        from exl3xpu import shared_kv_verify as verify
        verify.run = self._verify_original
        headers.require(self._verify_record is not None, "no executed page1600/Q4 call observed")
        return self._verify_record


def capture(args):
    headers.require(not args.output.exists() and not args.output.with_suffix(".json").exists(),
                    "refusing existing capture/report")
    reference = verify_inputs(args.reference_manifest, args.model, args.image_identity)
    import yaml
    profile = yaml.safe_load(PROFILE.read_text())
    for key, value in profile["env"].items():
        if value is not None:
            os.environ[key] = str(value).replace("{model_dir}", str(PROFILE.parent))
    headers.require(digest(Path(os.environ["EXL3_DRAFT_VOCAB"]).read_bytes()) ==
                    digest(args.subset.read_bytes()), "subset hash mismatch")
    os.environ["EXL3_DRAFT_VOCAB"] = str(args.subset)
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.config import ReasoningConfig
    allowed = {f.name for f in dataclasses.fields(EngineArgs)}
    kwargs = {k: v for k, v in profile["vllm"].items() if k in allowed}
    if isinstance(kwargs.get("reasoning_config"), dict):
        kwargs["reasoning_config"] = ReasoningConfig(**kwargs["reasoning_config"])
    overrides = {"max_model_len": 262144, "max_num_seqs": 4,
                 "max_num_batched_tokens": 1600, "num_gpu_blocks_override": 180,
                 "async_scheduling": False, "enforce_eager": True,
                 "compilation_config": {"cudagraph_mode": "NONE"}}
    kwargs.update(overrides, model=str(args.model),
                  worker_extension_cls="capture_verify.VerifyCaptureWorker")
    task = next(c for c in json.loads(args.workload.read_text())["cases"]
                if c["id"] == "c1-p4096-o1024")
    prompt = task["requests"][0]["prompt_ids"]
    headers.require(len(prompt) == 4096, "requires frozen real4096 prompt")
    llm = None
    try:
        llm = LLM(**kwargs)
        params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
        headers.require(llm.llm_engine.reset_prefix_cache(), "cold prefix reset failed")
        ordinary = llm.generate({"prompt_token_ids": prompt}, params, use_tqdm=False)
        headers.require(llm.llm_engine.reset_prefix_cache(), "cold prefix reset failed")
        llm.collective_rpc("install_verify_capture", timeout=60, args=(str(args.output),))
        observed = llm.generate({"prompt_token_ids": prompt}, params, use_tqdm=False)
        ids = lambda output: [list(o.outputs[0].token_ids) for o in output]
        headers.require(ids(ordinary) == ids(observed), "observer changed emitted IDs")
        records = llm.collective_rpc("finish_verify_capture", timeout=60)
        headers.require(len(records) == 1, "requires one original worker")
        report = records[0] | {"checkpoint": reference["checkpoint"]["identity"],
                  "image": args.image_identity, "overrides": overrides,
                  "ordinary_ids": ids(ordinary), "observed_ids": ids(observed),
                  "tool_sha256": digest(Path(__file__).read_bytes())}
        with args.output.with_suffix(".json").open("x") as stream:
            json.dump(report, stream, indent=2); stream.write("\n")
        print("ORIGINAL_VERIFY_CAPTURE_PASS", report["capture_sha256"], flush=True)
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "reference-manifest", "subset", "workload", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--image-identity", required=True)
    capture(parser.parse_args())
