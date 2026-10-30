# Flint llama.cpp oracle gate — 2026-10-23

The gateability measurement for the
[`llama-cpp-flint`](../../.agents/oracles/llama-cpp-flint.md) oracle record:
the pinned PR head BUILT and RAN the `flint` architecture and produced
coherent tokens on the verified `unsloth/flint-GGUF` UD-IQ1_M container. This
file proves gateability only. No rate in it is quotable: the run was a
correctness gate on a heavily loaded shared host at 4 threads, and it recorded
no throughput worth citing. The floors are the row's own later gates.

## What ran

- **Revision.** `ggml-org/llama.cpp` head `1b44e30775dcb2f93952116b05f89e5bd797fb00`
  (the model author's own Flint PR, ggml-org/llama.cpp#30928). Cloned fresh to
  `/tmp/llamacpp-flint-oracle` and checked out at that SHA;
  `git rev-parse HEAD` returned the pin, and the binary self-reports
  `build : b11797-1b44e3077`.
- **Artifact.** `/mnt/models/unsloth-flint-GGUF/flint-preview-UD-IQ1_M.gguf`,
  11.35 GB, sha256 verified in the same session:
  `dd45136a8b26e02efa13e8d6020518f3121c213e1865cd8c3f85de8d26fd5149` — the
  container the row's records name.

## Build

CPU-only, native, on the aarch64 host (`aarch64-unknown-linux-gnu`, GCC 16.2,
`-mcpu=ampere1a+crc+dotprod+i8mm+nosve+nosme` per the project's own feature
detect; SVE/SME absent). The new-at-this-head app/cli/server layout requires
`LLAMA_BUILD_TOOLS=ON` AND `LLAMA_BUILD_SERVER=ON` — `tools/cli` and
`tools/server` (which provide the `llama-cli-impl` / `llama-server-impl`
libraries `llama-app` links) sit behind `LLAMA_BUILD_SERVER`, and building the
app with the server OFF fails at link with `cannot find -llama-server-impl`:

```sh
git clone https://github.com/ggml-org/llama.cpp /tmp/llamacpp-flint-oracle
cd /tmp/llamacpp-flint-oracle
git fetch origin 1b44e30775dcb2f93952116b05f89e5bd797fb00   # a PR head: fetch by SHA
git checkout 1b44e30775dcb2f93952116b05f89e5bd797fb00
cmake -B build -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -G Ninja
cmake --build build --target llama-mtmd-cli -j 4   # pulls in bin/llama + libs
```

`build/bin/llama --version`:
`version: 0.7.0-dev (build 11797, commit 1b44e3077)`, `for Linux aarch64`.
Running the binary needs `LD_LIBRARY_PATH=build/bin` (it dynamically links
`libllama-server-impl.so`, which itself needs `libmtmd` from the same tree).

## Run

The container's embedded chat template does not parse (a jinja lexer error on
a `{% replace("\x00", ...) %}` filter at this llama.cpp's jinja bridge), which
refuses the server init. The run disables the jinja engine and pins the
chatml shell explicitly — a template-shell choice, not a model change:

```sh
LD_LIBRARY_PATH=build/bin taskset -c 0-15 nice -n 19 ./build/bin/llama cli \
  -m /mnt/models/unsloth-flint-GGUF/flint-preview-UD-IQ1_M.gguf \
  -p "Count from 1 to 5 in words." -n 24 --temp 0 --no-warmup -c 512 -t 4 \
  --no-display-prompt --no-jinja --chat-template chatml
```

Exit status 0. The generated text, verbatim:

> The numbers from one to five in words are: one, two, three, four, five.
> `<|im_end|>`

The runtime LOADED the artifact and generated coherent, prompt-faithful text
end-to-end: this oracle pin is gateable for flint. The reported
`[ Prompt: 0.2 t/s | Generation: 0.0 t/s ]` is the loaded-host artifact of a
4-thread correctness gate on a box at load ~265 and is deliberately not a
measurement.

## Machine state

| when | load avg | MemAvailable |
|---|---|---|
| before | 263.46, 263.57, 258.69 | 76.6 GB (76,569,024 kB) |
| after | 268.86, 265.78, 262.53 | 173.1 GB (173,147,392 kB) |

Busy shared host throughout; 256 GB RAM, 256 GB swapfile
(`/mnt/models/swapfile`). No GPU, no lease: CPU-only.
