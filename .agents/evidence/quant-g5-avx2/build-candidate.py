#!/usr/bin/env python3
"""Build the G5 candidate server by swapping two objects into the GSQ archive.

The Flare loader lives on the GSQ worktree, so this row cannot yet build the
server end to end. This reuses the recorded g++-13 compile and link recipes from
/tmp/flare-gsq-repair-build and replaces only the two CPU quant-dot objects.
"""

import json
import pathlib
import shutil
import subprocess

ROW = pathlib.Path("/home/user/lumen-engine-work-avx2")
GSQ_BUILD = pathlib.Path("/tmp/flare-gsq-repair-build")
OUT = pathlib.Path("/tmp/g5-obj")
OUT.mkdir(exist_ok=True)


def compile_one(source, output, wide):
    argv = json.loads((GSQ_BUILD / "gemma4_gguf_weights-compile.json").read_text())
    exact = {
        "/tmp/flare-gsq-repair-build/gemma4_gguf_weights.cpp.o": str(output),
        "/tmp/lumen-flare-gsq-repair/src/vllm/model_executor/models/gemma4_gguf_weights.cpp":
            str(source),
    }
    argv = [exact.get(a, a) for a in argv]
    argv = [
        a.replace("/tmp/lumen-flare-gsq-repair/include", str(ROW / "include"))
         .replace("/tmp/lumen-flare-gsq-repair/src", str(ROW / "src"))
        for a in argv
    ]
    if wide:
        argv[1:1] = ["-mavx2", "-mf16c"]
    print("+", " ".join(argv))
    subprocess.run(argv, check=True)


# The archive member carries CMake's numeric prefix; reusing the exact name
# replaces it instead of adding a second definition.
compile_one(ROW / "src/vt/cpu/cpu_quant_dot.cpp",
            OUT / "588_cpu_quant_dot.cpp.o", False)
compile_one(ROW / "src/vt/cpu/cpu_quant_dot_avx2.cpp",
            OUT / "cpu_quant_dot_avx2.cpp.o", True)

archive = OUT / "libvllm-g5.a"
shutil.copyfile(GSQ_BUILD / "libvllm.a", archive)
subprocess.run(["ar", "r", str(archive),
                str(OUT / "588_cpu_quant_dot.cpp.o"),
                str(OUT / "cpu_quant_dot_avx2.cpp.o")], check=True)

argv = json.loads((GSQ_BUILD / "server-link.json").read_text())
argv = [a.replace(str(GSQ_BUILD / "libvllm.a"), str(archive)) for a in argv]
argv[argv.index("-o") + 1] = str(OUT / "lumenengine-server")
print("+", " ".join(argv))
subprocess.run(argv, check=True, cwd="/home/user/lumen-engine/build")
