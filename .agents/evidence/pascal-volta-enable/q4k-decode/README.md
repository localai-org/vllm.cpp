# Pascal Q4_K decode and prefill warp mapping evidence

Issue: `ISSUE-LOCAL-01M652EAJ85CNWG65FGPTZ8742`.
Row: `BACKEND-CUDA-SM061`. Base: `ca0b2ff73`.

## Result

The Q4_K decode lane and the prefill warp mapping pass the focused quant suite
in both the default and the `VT_PASCAL_Q4K_DECODE=0` rollback configuration.
The real-model gate stays token-exact on six of eight axes. The two remaining
axes are near-ties, recorded below.

All GPU runs used `flock /home/user/gpu.lock` on the local Quadro P620,
sm_61, 2048 MiB, driver 580.178.04, CUDA 12.4.131.

## Recipe

```sh
cmake -S /tmp/lumen-pascal -B /tmp/lumen-build-legacy -G Ninja \
  -DVLLM_CPP_CUDA=ON '-DVLLM_CPP_CUDA_ARCHITECTURES=60;61;70' \
  -DVLLM_CPP_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXE_LINKER_FLAGS=-L/usr/lib/gcc/x86_64-linux-gnu/15
cmake --build /tmp/lumen-build-legacy --target test_cuda_quant_dot test_cuda_legacy_capi -j18
flock /home/user/gpu.lock /tmp/lumen-build-legacy/tests/test_cuda_quant_dot \
  -tce='CUDA grouped keep-quant at GLM*'
```

The GLM-5.3-Flash published MoE geometry allocates about 2.4 GB of weights and
cannot fit the 2 GB P620. The preceding Q5_0 evidence records the same blocker.
The case is excluded, not weakened.

## Focused suite

`quant-default.log` and `quant-rollback.log` each exit 0 with 21 cases and
255027 / 254813 assertions. The single excluded case is the GLM geometry.

The new case reconstructs the Q8_1 activation independently and checks the
registered `MatmulBTQuant` path for Q4_K at `K = 256, 2048, 5632` and `N = 7`.

## Kernel time

`nsys-kernels.txt` retains the two traces. The Q4_K decode kernel fell from
868.0 ms to 406.4 ms (2.1x). After that change the generic Q4_K prefill is the
largest item at 412.5 ms, followed by the generic Q6_K prefill at 201.0 ms.

## Same-binary end-to-end A/B

64 generated tokens, three repetitions, warm median:

| Model | `VT_PASCAL_Q4K_DECODE=0` | default | Delta |
|---|---|---|---|
| TinyLlama p16 | 21.0 tok/s | 22.0 tok/s | +4.8% |
| SmolLM2 p16 | 40.1 tok/s | 40.3 tok/s | +0.5% |

The prefill warp mapping moved TinyLlama p16/t16 from 12.8 to 13.8 tok/s (+8%).
It changes no per-output arithmetic and is token-neutral on every exact axis.

## Real-model gate

| Axis | Result | Note |
|---|---|---|
| SmolLM2 p128/t16, p128/t64 | exact | |
| TinyLlama p16/t16, p128/t16, p128/t64 | exact | |
| TinyLlama p16/t64 | near-tie at step 28 | oracle `29909`, engine `29903`, gap `< 0.001` nats |
| SmolLM2 p16 | near-tie at step 8 | gap `0.035` nats; CPU oracle agrees with the engine |

The TinyLlama tie is exact to milli-nat precision. The reviewed register
attention lane resolves it the same way by default. `VT_ATTN_DECODE_OPT=0`
restores the portable attention and makes the axis exact. The Q4_K lane reaches
the same tie by a different reduction order; `VT_PASCAL_Q4K_DECODE=0` restores
the byte-identical generic lane.

The SmolLM2 p16 gap is the separately owned cross-runtime divergence: the
llama.cpp CPU oracle and this engine both choose `357`, while the CUDA oracle
chooses `5649`.

## Raw logs

- `quant-default.log`, `quant-rollback.log`
- `ab-tiny-q4k1.log`, `ab-tiny-q4k0.log`, `ab-smol-q4k1.log`, `ab-smol-q4k0.log`
- `gate-tiny-p16-t64-default.log`, `gate-tiny-p16-t64-opt0.log`
- `gate-tiny-p128-t64.log`, `gate-smol-p128-t64.log`
- `nsys-kernels.txt`
