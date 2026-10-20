#!/usr/bin/env python3
"""Independent reduced Gemma4 pool equations; no C++ imports or outputs.

HF a08ace4bbd97e721c98751deec37d87b026acadc modeling_gemma4.py
Gemma4VisionPatchEmbedder and Gemma4VisionPooler. Residual output
projections are zero, so the nonzero patch encoding is the encoder output.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
dtype = torch.bfloat16
width = 12  # three MHA heads, dimension4; sqrt(width) is not a power of two
positions = torch.tensor([[x, y] for y in range(3) for x in range(6)])
pixels = torch.full((18, 768), 0.5)
pixels[:, :width] = (((torch.arange(18 * width) * 7) % 31).reshape(18, width) + 1) / 32
weight = torch.zeros(width, 768)
weight[:, :width] = torch.eye(width)
table = (((torch.arange(2 * 6 * width) * 3) % 17) - 8).reshape(2, 6, width).float() / 64 + 0.25
encoder = F.linear((2 * (pixels - 0.5)).to(dtype), weight.to(dtype))
encoder += (table[0, positions[:, 0]] + table[1, positions[:, 1]]).to(dtype)
indices = positions[:, 0] // 3 + 2 * (positions[:, 1] // 3)
average = (F.one_hot(indices, 2).float().T / 9) @ encoder.float()
scaled = average.to(dtype).float() * math.sqrt(width)
pooled = scaled.to(dtype)
late = (average * math.sqrt(width)).to(dtype)
assert torch.count_nonzero(pooled != late) > 0
out = dict(width=width, heads=3, head_dim=4,
           pixels=pixels.flatten().tolist(), positions=positions.flatten().tolist(),
           position_table=table.flatten().tolist(), encoder=encoder.float().flatten().tolist(),
           source_scaled=scaled.flatten().tolist(), pooled=pooled.float().flatten().tolist(),
           late_rounding=late.float().flatten().tolist())
path = Path(__file__).resolve().parents[2] / 'tests/fixtures/gemma4_e4b_vision/pool_rounding.json'
path.write_text(json.dumps(out, separators=(',', ':')) + '\n')
print(torch.__version__, torch.version.git_version, 'CPU only')
print('discriminating scalars', torch.count_nonzero(pooled != late).item(), 'of', pooled.numel())
