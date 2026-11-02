"""Same-binary registered attention A/B; run under the GPU file mutex.

The wide bound detects restoration of the per-key block-barrier production
path, rather than trying to measure a small speedup under scheduler noise.
"""
import os
import re
import subprocess
import sys


def measure(binary, rollback):
    env = dict(os.environ)
    env["VT_ATTN_DECODE_OPT"] = "0" if rollback else "1"
    result = subprocess.run([binary], env=env, capture_output=True, text=True)
    if result.returncode == 77:
        sys.exit(77)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    result.check_returncode()
    values = {}
    for line in result.stdout.splitlines():
        match = re.fullmatch(r"(d=64 seq=(?:128|256) input=\d+ output=\d+ group=\d+ window=0 causal=1) us=([\d.]+)", line)
        if match:
            values[match[1]] = float(match[2])
    assert len(values) == 16, values
    return values


baseline = measure(sys.argv[1], True)
optimized = measure(sys.argv[1], False)
for shape, us in baseline.items():
    ratio = us / optimized[shape]
    print(f"{shape} baseline_us={us} optimized_us={optimized[shape]} speedup={ratio:.3f}")
    assert ratio > 4, f"legacy registered decode lost its warp-register dispatch: {shape} {ratio}"
