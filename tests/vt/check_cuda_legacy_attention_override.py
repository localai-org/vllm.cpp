"""Check Pascal d128 override reachability with the registered attention gate.

Run this script under the GPU file mutex. Each child also checks numerical
correctness. The wide timing ratio distinguishes the two production kernels.
"""
import os
import re
import subprocess
import sys


def measure(binary, setting, master="1"):
    env = dict(os.environ)
    env.pop("VT_ATTN_DECODE_D128", None)
    if setting is not None:
        env["VT_ATTN_DECODE_D128"] = setting
    env["VT_ATTN_DECODE_OPT"] = master
    result = subprocess.run([binary], env=env, capture_output=True, text=True)
    if result.returncode == 77:
        sys.exit(77)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    result.check_returncode()
    values = {}
    for line in result.stdout.splitlines():
        match = re.fullmatch(
            r"(d=128 seq=128 input=\d+ output=\d+ group=4 window=0 causal=1) us=([\d.]+)", line
        )
        if match:
            values[match[1]] = float(match[2])
    assert len(values) == 4, values
    return values


unset = measure(sys.argv[1], None)
zero = measure(sys.argv[1], "0")
one = measure(sys.argv[1], "1")
masteroff = measure(sys.argv[1], "1", "0")
for shape in unset:
    for enabled in (unset, one):
        for disabled in (zero, masteroff):
            ratio = disabled[shape] / enabled[shape]
            print(f"{shape} disabled_us={disabled[shape]} enabled_us={enabled[shape]} ratio={ratio:.3f}")
            assert ratio > 4, f"Pascal d128 override did not change production dispatch: {shape} {ratio}"
