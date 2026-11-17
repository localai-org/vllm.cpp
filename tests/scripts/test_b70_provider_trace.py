"""Production lookup probe in fresh processes, with and without trace opt-in."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

probe = str(Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    env = {k: v for k, v in os.environ.items() if k != "VT_OP_PROVIDER_TRACE"}
    subprocess.run([probe], cwd=root, env=env, check=True)
    assert list(root.iterdir()) == [], "default path created a trace"
    trace = root / "trace.jsonl"
    env["VT_OP_PROVIDER_TRACE"] = str(trace)
    subprocess.run([probe], cwd=root, env=env, check=True)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len(rows) == 7, rows
    assert [r["sequence"] for r in rows] == list(range(1, 8))
    assert [r["provider_selections"] for r in rows] == [1, 1, 2, 2, 3, 3, 4]
    assert all(r["device"] == "xpu" and r["count_kind"] == "selection" for r in rows)
    assert all(r["provider"] == "vt-native" and not r["cpu_reference"] for r in rows[0:6:2])
    assert all(r["cpu_reference"] for r in rows[1:6:2])
    assert rows[-1]["event"] == "fallback_selection"
    assert rows[-1]["reference_tier_hits"] == 4
    print("PASS: 7 selections, 4 CPU reference selections; default creates no file")
