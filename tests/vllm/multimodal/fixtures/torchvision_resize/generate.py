"""Model-free fixture generator; run in the pinned Python reference image."""
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch
import torchvision
import transformers
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor, smart_resize
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import InterpolationMode

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
cases = [
    (37, 53, 32, 64),  # mixed up/down, odd source sizes
    (19, 13, 64, 32),  # upsample both
    (129, 97, 32, 32),  # antialiased downsample both
    (31, 47, 31, 32),  # horizontal only
    (31, 47, 64, 47),  # vertical only
    (32, 64, 32, 64),  # identity
    (1, 7, 9, 13),  # tiny/border supports
    (7, 1, 13, 9),
    (17, 19, 1, 1),  # complete area average
    (81, 99, 16, 32),  # wide filter support
]
records = []
for number, (h, w, rh, rw) in enumerate(cases):
    y, x = np.indices((h, w))
    rgb = np.stack([(x * 17 + y * 3) % 256, (x * 5 + y * 11) % 256,
                    (x * 7 + y * 19) % 256], axis=-1).astype(np.uint8)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    resized = F.resize(tensor, [rh, rw], InterpolationMode.BICUBIC, antialias=True)
    assert resized.dtype == torch.uint8
    data = resized.permute(1, 2, 0).contiguous().numpy().tobytes()
    filename = f"case-{number}.rgb"
    (out / filename).write_bytes(data)
    records.append({"input_hw": [h, w], "output_hw": [rh, rw], "file": filename,
                    "sha256": hashlib.sha256(data).hexdigest()})
processor_records = []
for number, (h, w) in enumerate([(16, 16), (17, 23), (3, 5), (65, 97)]):
    config = {"patch_size": 4, "temporal_patch_size": 2, "merge_size": 2,
              "size": {"shortest_edge": 64, "longest_edge": 1024},
              "rescale_factor": 1.0 / 255.0,
              "image_mean": [0.25, 0.5, 0.75], "image_std": [0.5, 0.25, 0.125]}
    y, x = np.indices((h, w))
    rgb = np.stack([(x * 17 + y * 3) % 256, (x * 5 + y * 11) % 256,
                    (x * 7 + y * 19) % 256], axis=-1).astype(np.uint8)
    kw = Qwen2VLImageProcessor(**config)(images=[Image.fromarray(rgb)], return_tensors="pt")
    pixels = kw["pixel_values"]
    assert pixels.dtype == torch.float32
    files = {}
    for suffix, data in [("f32", pixels.numpy().tobytes()),
                         ("f16", pixels.to(torch.float16).view(torch.int16).numpy().tobytes())]:
        filename = f"processor-{number}.{suffix}"
        (out / filename).write_bytes(data)
        files[suffix] = {"file": filename, "sha256": hashlib.sha256(data).hexdigest()}
    config_filename = f"processor-{number}.json"
    (out / config_filename).write_text(json.dumps(config, indent=2) + "\n")
    processor_records.append({"input_hw": [h, w], "config_file": config_filename,
                              "patch_shape": list(pixels.shape),
                              "grid": kw["image_grid_thw"].tolist()[0], "files": files})
(out / "model-config.json").write_text(json.dumps({
    "image_token_id": 248056, "vision_start_token_id": 248053, "vision_end_token_id": 248054,
    "vision_config": {"patch_size": 4, "temporal_patch_size": 2, "spatial_merge_size": 2},
}, indent=2) + "\n")
smart_cases = [(513, 385), (1, 199), (80, 112), (1600, 3200), (384, 512), (65, 65)]
(out / "manifest.json").write_text(json.dumps({
    "image": "sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918",
    "torch": torch.__version__, "torch_git": torch.version.git_version,
    "torchvision": torchvision.__version__, "transformers": transformers.__version__,
    "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
    "resize_entrypoint": "torchvision.transforms.v2.functional.resize",
    "cases": records, "processor_cases": processor_records,
    "smart_resize_cases": [{"input_hw": [h, w], "output_hw": list(smart_resize(h, w, 32, 65536, 4194304))}
                           for h, w in smart_cases],
}, indent=2) + "\n")
print(f"Generated {len(records)} executed torchvision and {len(processor_records)} processor cases")
