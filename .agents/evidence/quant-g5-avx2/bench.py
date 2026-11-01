#!/usr/bin/env python3
"""Same-binary A/B runner for the Flare GSQ CPU profile.

Starts one server, waits for /health, sends one 8-token warmup and then N
greedy 32-token requests, and writes the timings plus the emitted text to the
requested JSON file. The server log is captured beside it.
"""

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request


def post(port, path, body, timeout=900):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=11601)
    parser.add_argument("--threads", type=int, default=18)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--prompt", default="Continue the sequence: 1, 2, 3, 4, 5,")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    env = dict(
        os.environ,
        VLLM_CPP_CPU_THREADS=str(args.threads),
        OMP_NUM_THREADS="4",
        VT_GGUF_KEEP_QUANT="1",
        CUDA_VISIBLE_DEVICES="",
    )
    command = [
        args.server,
        "--model", args.model,
        "--served-model-name", "Lumen-3.8-Flare",
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--device", "cpu",
        "--max-model-len", "8192",
        "--block-size", "16",
        "--num-blocks", "513",
        "--max-num-seqs", "1",
        "--max-num-batched-tokens", "512",
        "--enable-thinking",
        "--enable-radix-attention",
        "--enable-metrics",
        "--language-model-only",
        "--disable-log-requests",
    ]

    log = open(args.out + ".server.log", "w")
    started = time.time()
    server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 240
        while time.time() < deadline:
            if server.poll() is not None:
                raise SystemExit(f"server exited with {server.returncode}")
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{args.port}/health", timeout=2
                ).read()
                break
            except Exception:
                time.sleep(1)
        else:
            raise SystemExit("server did not become healthy")

        startup = time.time() - started
        body = {
            "model": "Lumen-3.8-Flare",
            "prompt": args.prompt,
            "max_tokens": args.tokens,
            "temperature": 0.0,
            "stream": False,
        }
        post(args.port, "/v1/completions", dict(body, max_tokens=8))
        runs = []
        for _ in range(args.runs):
            begin = time.time()
            response = post(args.port, "/v1/completions", body)
            elapsed = time.time() - begin
            choice = response["choices"][0]
            runs.append(
                {
                    "elapsed_s": elapsed,
                    "completion_tokens": response["usage"]["completion_tokens"],
                    "text": choice["text"],
                }
            )

        result = {
            "server": args.server,
            "command": command,
            "threads": args.threads,
            "startup_s": startup,
            "prompt": args.prompt,
            "runs": runs,
            "loadavg": open("/proc/loadavg").read().strip(),
        }
        with open(args.out, "w") as output:
            json.dump(result, output, indent=2)
        print(json.dumps(result, indent=2))
    finally:
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()


if __name__ == "__main__":
    main()
