# Quickstart

Run a model without building the project. Use a local checkpoint or let the
server download one from Hugging Face. See the [usage guide](USAGE.md) and
[reference](reference/README.md) for more options.

`--model` accepts a directory, a `.gguf` file, `org/repo` for a Hugging Face
snapshot, or `org/repo:TAG` for one GGUF file. The repository forms download
files that your cache lacks.

> **Container and release availability.** Use the published `main-<lane>`
> container tags. `:latest` does not exist yet because no release includes the
> container lanes. Release `v0.0.2` predates `--model org/repo`, so its archives
> cannot download a checkpoint by repository ID. The release-archive recipe
> remains marked `PENDING(#1281)`.

## Run a model with Docker

This needs no build, no checkout, and no GPU. The entrypoint is `vllm-server`,
so every flag after the image name goes to the server.

Fetch a checkpoint, then mount it:

```sh
hf download Qwen/Qwen3-0.6B --local-dir ./qwen3-0.6b

docker run --rm -p 8000:8000 \
  -v "$PWD/qwen3-0.6b:/models/qwen3-0.6b:ro" \
  ghcr.io/mudler/vllm.cpp:main-cpu \
  --model /models/qwen3-0.6b
```

Answer it from another terminal:

```sh
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3-0.6b", "prompt": "The capital of France is", "max_tokens": 32}'
```

Any OpenAI client works. Point its `base_url` at `http://localhost:8000/v1`.

To let the server download the checkpoint, pass its Hugging Face repository ID:

```sh
docker run --rm -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/cache" \
  ghcr.io/mudler/vllm.cpp:main-cpu \
  --model Qwen/Qwen3-0.6B
```

`main-cpu` selects the CPU lane, which runs anywhere. For an NVIDIA GPU use
`ghcr.io/mudler/vllm.cpp:main-cuda` and add `--gpus all`. On a Jetson or Tegra
host you need `--runtime nvidia --gpus all` together. Each lane is a
`linux/amd64` and `linux/arm64` manifest, so the same tag works on both. The
lanes, the tag meanings, and the two flag families are in
[`docs/guides/container-images.md`](guides/container-images.md).

The container runs as UID `1000`. That user must be able to write to `/cache`,
which is where the Hugging Face cache lives inside the image.

## Run a model from a release archive

For a host that does not run Docker. The archive carries `bin/vllm-server` and
nothing else it needs, so you unpack it and run it.

```sh
# PENDING(#1281): release v0.0.2 predates `--model org/repo`, so this line needs
# a release cut after that feature landed. Until one exists, build the tree as
# docs/BUILD.md describes and run build/examples/vllm-server instead.
tar -xzf vllm.cpp-<version>-linux-x86_64-glibc-cpu.tar.gz
./bin/vllm-server --model Qwen/Qwen3-0.6B --port 8000
```

The same `curl` above answers it. Checksums, provenance sidecars, and the full
list of published archives are in [`docs/RELEASES.md`](RELEASES.md).

## Models that have been run

This table records a completed container run. It covers the mounted-checkpoint
recipe above; the repository-ID recipe has no recorded container run here.

| Model | Lane image | `--model` line | Memory needed | Date run | Host |
|---|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` | `ghcr.io/mudler/vllm.cpp:main-cpu` @ `sha256:7f88301e` | `--model /models/qwen3-0.6b` | 1.5 GiB on disk | 22 August 2026 | `mudler-ubuntu-box`, x86_64, Docker 29.1.2 |

For this run, the host pulled the image and fetched the checkpoint with
`hf download`. The `/v1/completions` endpoint returned
`" Paris. The capital of Italy is Rome. The capital of Spain is Madrid."` for
`"The capital of France is"` at 16 tokens. This result covers only the CPU lane
and the mounted checkpoint.

## Caches, tokens, and hosts with no network

See the [Hugging Face access guide](guides/hugging-face-access.md) for cache
layout, authentication, and offline behavior.

- **Reusing a cache.** A cache that Python `huggingface_hub` already populated
  is read rather than downloaded again, because the layout is the same one. Give
  the container `-v "$HOME/.cache/huggingface:/cache"` and a second run opens no
  socket.
- **A gated repository.** Set `HF_TOKEN` to a token that can read it, or point
  `HF_TOKEN_PATH` at a file holding one.
- **A host with no network.** Set `HF_HUB_OFFLINE=1`. A warm cache then behaves
  exactly as it would online. A cold one refuses and names the directory it
  searched, so you can see which cache root the run chose.

## Where the full reference lives

| You want | Read |
|---|---|
| Every server flag, endpoint, and default | [`docs/reference/server.md`](reference/server.md) |
| How `--model` resolves, and the cache it writes | [`docs/reference/model-loading.md`](reference/model-loading.md) |
| The C API | [`docs/reference/c-api.md`](reference/c-api.md) |
| Runnable workflows end to end | [`docs/USAGE.md`](USAGE.md) |
| Building, and every CMake option | [`docs/BUILD.md`](BUILD.md) |
| What is proven and what is not | [Project status](../README.md#project-status) |
