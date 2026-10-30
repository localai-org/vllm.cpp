# Four-way GB10 benchmark on Qwen3.8-27B

| Field | Value |
|---|---|
| Issue | [#979](https://github.com/mudler/vllm.cpp/issues/979) |
| Owning rows | [`BACKEND-GATE-CUDA-VLLM`](../backend-matrix.md), [`BACKEND-GATE-CUDA-SGLANG`](../backend-matrix.md), [`BACKEND-GATE-CUDA-LLAMACPP`](../backend-matrix.md) |
| Roadmap | `ROAD-V1-A` (the perf and SGLang floor lane) |
| Umbrella | [competitive-benchmarks.md](competitive-benchmarks.md) fixes the workload vocabulary and the per-backend leaf-spike contract |
| Sibling leaf | [cuda-sglang-low-concurrency.md](cuda-sglang-low-concurrency.md) owns the three-arm cache-neutral SGLang gate on the Qwen3.6 snapshots and stays authoritative for it |
| Subject | `Qwen/Qwen3.8-27B` @ `1d4bf0f2`, `unsloth/Qwen3.8-27B-NVFP4` @ `a767244d`, `unsloth/Qwen3.8-27B-GGUF` @ `fe1e2a23`. The two `unsloth` revisions are **named, not inspected**, see section 2.6 |
| Host | `dgx.casa`, GB10 sm_121a, one `flock $GPU_LOCK` for the whole series |
| Role | helper, branch `row/BENCH-QWEN38-27B-FOUR-WAY`, base `598226e962ddd4a83292e3d9264bbea9f41603d2` |
| Status | `SPIKE`. Scoping and record reconciliation only. No number is measured by this spec. |

## 0. Why this is a new leaf and not an edit to the SGLang spike

[cuda-sglang-low-concurrency.md](cuda-sglang-low-concurrency.md) is a three-arm
spike over SGLang, vLLM and ours, on the Qwen3.6 27B and 35B NVFP4 snapshots,
pinned to SGLang v0.5.13 for its P1 harness. Three things here fall outside it.

1. **The subject changes.** Qwen3.8-27B is the checkpoint [#915](https://github.com/mudler/vllm.cpp/issues/915)
   gated. Every checkpoint-scoped artifact in the sibling spike is bound to the
   Qwen3.6 snapshots by revision.
2. **A fourth engine joins.** llama.cpp on GB10 CUDA appears in no matrix row and
   in no leaf spike. The sibling spike names three arms and excludes GGUF.
3. **The deliverable is different.** The sibling spike produces a binding
   every-axis gate on one shared checkpoint. This one produces a matrix of pairs,
   each with its own stated common denominator, because at these pins **no single
   quantization is common to all four engines**.

The sibling spike is not superseded and its Qwen3.6 evidence is not reinterpreted
here.

## 1. Scope

| In scope | Out of scope |
|---|---|
| CUDA on `dgx.casa` at the four engines' recorded pins | Any other host, and any comparison across different checkpoints |
| Text-only greedy serving, `POST /v1/completions` | Vision and video inputs, chat templates, tool calling |
| A per-pair common-denominator matrix with an explicit not-comparable verdict | One headline four-way ratio |
| Raw-decode and drafted arms, each declared, never mixed | Any cell that compares a drafted arm against a raw one |
| Record reconciliation of the two stale entries in section 5 | Advancing the SGLang pin, which is separate deliberate work |

## 2. There is no single quantization all four run

Established from each engine's own source at its recorded pin. Nothing in this
table is inferred from a model card or a release note.

| | bf16 | NVFP4 | Q4_K_M GGUF |
|---|---|---|---|
| ours | GATED, [#915](https://github.com/mudler/vllm.cpp/issues/915) | owed, [#821](https://github.com/mudler/vllm.cpp/issues/821) | text-only loader only, no `clip` projector, [#821](https://github.com/mudler/vllm.cpp/issues/821) |
| vLLM `555967922` | yes | yes | **absent from the tree** |
| SGLang `f63458b5` v0.5.15 | yes | yes | stack present, this architecture unreachable |
| llama.cpp `237ad9b96` | convertible, not its representative arm | yes, in the GGUF container, see 2.5 | its native arm |

This table said `no` for llama.cpp NVFP4 until 2026-08-16. That was false at the
recorded pin, and section 2.5 both re-derives it and re-derives the two
not-comparable verdicts of section 2.4 that leaned on it.

### 2.1 vLLM at our pin has no GGUF path at all

`6635279d8` (vllm#39612, 2026-06-13) removed the whole surface and moved it out
of tree. At `555967922`:

- the `LoadFormats` `Literal` at `vllm/model_executor/model_loader/__init__.py:33-49`
  lists every load format and `gguf` is not among them, and
  `_LOAD_FORMAT_TO_MODEL_LOADER` at `:50-66` agrees.
- `vllm/model_executor/layers/quantization/__init__.py:12-46` has no `gguf` entry
  in `QuantizationMethods`.
- `vllm/config/load.py:30-58` documents the same set and ends with "Other custom
  values can be supported via plugins".
- The replacement is declared at `setup.py:1300` as
  `"extra-quant": ["vllm-gguf-plugin>=0.0.2"]`, and documented at
  `docs/features/quantization/gguf.md:6-13`.
- The only two case-insensitive `gguf` hits left in `vllm/` are comments, at
  `model_executor/models/qwen2_moe.py:499` and `lora/layers/utils.py:69`.

That plugin is not pinned by this project and has no `.agents/oracles/` record.
Consequence: **the vLLM-versus-llama.cpp pair has no common quantization and is
recorded not-comparable.** Naming the plugin as a pinned oracle would be a
separate decision with its own gateability measurement, not a step inside this
campaign.

### 2.2 SGLang's GGUF blocker is deeper than the alias table

The alias table is real and is exactly two entries wide. `python/sglang/srt/model_loader/loader.py:2129-2142`:

```python
        if model_type == "cohere":
            model_type = "command-r"
        elif model_type == "qwen3_moe":
            model_type = "qwen3moe"
        arch = None
        for key, value in gguf.MODEL_ARCH_NAMES.items():
            if value == model_type:
                arch = key
                break
        if arch is None:
            raise RuntimeError(f"Unknown gguf model_type: {model_type}")
```

`qwen3_5` misses because `gguf.MODEL_ARCH_NAMES` spells the family `qwen35`,
which is the same missing underscore the `qwen3_moe` entry exists to paper over.
**Adding the alias would not make it load.** Three further blockers. The first
is its own class. The second and third are one class over different parameters
of the same module, which the inverted wording of blocker 3 previously obscured:

1. `GGUFConfig.get_quant_method` (`layers/quantization/gguf.py:105-125`) handles
   `LinearBase`, `VocabParallelEmbedding` and `FusedMoE`, and returns `None` at
   `:125` for everything else. There is no gated-delta-net state path.
2. `Qwen3_5GatedDeltaNet` holds `A_log` and `dt_bias` as bare parameters with
   custom sharded loaders (`models/qwen3_5.py:250,253,257-258`). In
   `_get_gguf_weights_map` (`loader.py:2149-2153`) an unresolvable name yields
   `None`, producing colliding `None.weight` keys, and there is no guard.
3. `conv1d` is declared a `ColumnParallelLinear` and reshaped to three dimensions
   (`models/qwen3_5.py:195-204`), and its weight loader is replaced with
   `mamba_v2_sharded_weight_loader` (`:236-247`), which is written against the HF
   safetensors layout. It has no GGUF counterpart, exactly as in blocker 2.

**Blocker 3 was recorded with its polarity inverted until 2026-08-16**, and the
correction matters because the inverted version was the stated reason a load
report could not be trusted. The old wording said `conv1d`, being a `LinearBase`
subclass, would receive a `GGUFLinearMethod` and be silently wrong rather than
refused. Its own citation refutes it. At `f63458b5`:

- `models/qwen3_5.py:199` passes `quant_config=None` to that
  `ColumnParallelLinear`, inside the very range `:195-204` the claim cited.
- `layers/linear.py:176-179` takes the `quant_config is None` branch, assigns
  `UnquantizedLinearMethod()`, and never calls `quant_config.get_quant_method`.
  The `else` at `:180-181` is the only call site and it is not taken.
- `ColumnParallelLinear.__init__` forwards the argument unchanged to that
  constructor (`layers/linear.py:331-333`).
- Nothing re-wraps the module afterwards. The only other `conv1d` references in
  the file are `qwen3_5.py:204,237,260-261,272`, and none touches `quant_method`.

`GGUFConfig.get_quant_method` is therefore never reached for `conv1d`, no
`GGUFLinearMethod` is ever attached, and the module stays a dense layer of the
model dtype. If it failed it would fail loudly on the weight, not quietly on the
method. The withdrawn wording also reached `.agents/issue-index.md:266`, and that
row is corrected in place because it has not landed, see the note under `## Now`.

**Where the silence actually lives is a property of the load path, and it is not
yet attributed to a named parameter.** Three facts about `f63458b5` are
established here from source, and one question is left open rather than answered
by assertion.

Established:

- `_get_gguf_weights_map` writes `gguf_to_hf_name_map[f"{gguf_name}.{suffix}"]`
  with no check that `gguf_name` resolved (`model_loader/loader.py:2149-2153`).
  Every parameter the name map cannot resolve therefore collapses onto the single
  literal key `None.weight`, and all but the last one are lost from the map.
- `gguf_quant_weights_iterator` is driven by the tensors present in the file and
  gates each one on `tensor_name in gguf_to_hf_name_map`
  (`model_loader/weight_utils.py:1253,1280` and `:1289,1321`). There is no `else`
  and no counter, so an unmapped tensor is skipped without a message.
- `Qwen3_5ForCausalLM.load_weights` accumulates `loaded_params` but never
  compares it against `params_dict` (`models/qwen3_5.py:1359-1412`), and an
  unmatched name only reaches `logger.warning` at `:1405`. A parameter that is
  never yielded keeps its constructor value and nothing refuses the load.

Open, and owed to whoever adds the alias: **which** Qwen3.5 parameters actually
fail to resolve. That depends on `gguf.MODEL_ARCH_NAMES` and
`gguf.get_tensor_name_map`, and on a `transformers` able to build the meta model
at `loader.py:2146`. This project pins neither package, so the question cannot be
settled from the SGLang tree alone. The experiment is small and named:
instrument `loader.py:2152` to count `gguf_name is None`, print the collision set,
and diff `loaded_params` against `params_dict` after the load. Until that runs,
this spec claims the load path **cannot detect** an unloaded parameter, and
claims nothing about which parameter that is.

There is also an earlier gate at
`utils/hf_transformers/config.py:237-239`, which with a newer `transformers`
passes and lets the failure land on `loader.py:2142`.

Consequence: **the ours-versus-SGLang GGUF pair is recorded not-comparable**, and
the SGLang-versus-llama.cpp pair with it. Neither verdict rested on blocker 3's
polarity. The alias gate at `loader.py:2141-2142` alone is sufficient for both.

### 2.3 llama.cpp is the only comparator that runs the GGUF arm

`LLM_ARCH_QWEN35` and `LLM_ARCH_QWEN35MOE` are registered at
`src/llama-arch.cpp:41-42`, with per-architecture handling at `:890-891,912-913`.
The projector side has `PROJECTOR_TYPE_QWEN3VL` at `tools/mtmd/clip-impl.h:330`.

Our side loads the dense `qwen35` language file and maps it onto the registered
wrapper at `src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp:845,856`, and
has no `clip` projector path. Since this campaign is text-only that is sufficient
for the language arm, and the projector stays owed under
[#821](https://github.com/mudler/vllm.cpp/issues/821).

### 2.4 The resulting pair matrix

Each cell states the common denominator, or says not-comparable and why.

| Pair | Common denominator | Verdict |
|---|---|---|
| ours vs vLLM | bf16 `1d4bf0f2` | **comparable now.** c4 already reads 0.963x output throughput and 1.008x median ITL. c1 and c8 withheld under [#931](https://github.com/mudler/vllm.cpp/issues/931) until re-run. |
| ours vs vLLM | NVFP4 `a767244d` | **comparable after [#821](https://github.com/mudler/vllm.cpp/issues/821).** vLLM's ModelOpt and compressed-tensors NVFP4 paths are present. Ours is owed. |
| ours vs SGLang | bf16 `1d4bf0f2` | **comparable after a load preflight.** SGLang registers `Qwen3_5ForConditionalGeneration` at `models/qwen3_5.py:1633,2179`, and Qwen3.8 declares that architecture, but no SGLang load of this snapshot has been recorded. |
| ours vs SGLang | NVFP4 `a767244d` | **comparable after [#821](https://github.com/mudler/vllm.cpp/issues/821) and a load preflight.** |
| ours vs llama.cpp | Q4_K_M `fe1e2a23` | **comparable after [#821](https://github.com/mudler/vllm.cpp/issues/821).** The one pair where GGUF is the shared native arm. |
| vLLM vs SGLang | bf16 and NVFP4 | comparable in principle, and informational here. Neither is our engine, so no gate turns on it. |
| vLLM vs llama.cpp | none | **NOT COMPARABLE**, on the container, not the quantization. Both engines implement NVFP4. Neither can read the other's container, and every candidate shared artifact needs a conversion on one side. Re-derived in section 2.5. |
| SGLang vs llama.cpp | none | **NOT COMPARABLE.** The same container disjunction, plus section 2.2 for the GGUF direction. Re-derived in section 2.5. |

### 2.5 llama.cpp has NVFP4, and the two verdicts survive on a different reason

The table above and both not-comparable cells previously rested in part on
"llama.cpp has no NVFP4". At `237ad9b96` that is false, and section 2's own
preamble claims every cell was established from source at the recorded pin, so
the error falsified the section's stated method on the engine this campaign's
fourth arm is. The verdicts are therefore re-derived from scratch below rather
than patched.

**What is present at the pin.** NVFP4 is a first-class ggml type, not a
scaffold:

- `GGML_TYPE_NVFP4 = 40` (`ggml/include/ggml.h:430`) and
  `GGML_FTYPE_MOSTLY_NVFP4 = 26` (`:474`).
- `LLAMA_FTYPE_MOSTLY_NVFP4` in the loader (`src/llama-model-loader.cpp:46,763`).
- Type traits with `blck_size = QK_NVFP4`, `type_size = sizeof(block_nvfp4)`,
  `dequantize_row_nvfp4` and `quantize_row_nvfp4_ref`
  (`ggml/src/ggml.c:741-747`), the ftype mapping at `:1414`, and `quantize_nvfp4`
  at `:8012`.
- CUDA dispatch at `ggml/src/ggml-cuda/ggml-cuda.cu:1695,1717,1726,5583`, an MMQ
  instantiation at `ggml/src/ggml-cuda/template-instances/mmq-instance-nvfp4.cu`,
  and a Vulkan dequant shader at
  `ggml/src/ggml-vulkan/vulkan-shaders/dequant_nvfp4.comp`.
- A 493-line native W4A4 FP4-MMA GEMM whose header names **Blackwell sm_121a
  (GB10)**, this campaign's exact card, and states that it consumes the same
  e2m1 nibbles and e4m3 scale bytes from the GGUF `block_nvfp4`
  (`ggml/src/ggml-cuda/fp4-gemm.cu:1-16`, `fp4-gemm.cuh:5-37`).
- A Marlin-style W4A16 grouped MoE prefill GEMM over the same NVFP4 weights
  (`ggml/src/ggml-cuda/w4a16-gemm.cuh:5-25`).

**Upstream or fork.** Our llama.cpp pin is a local fork, as
[`../oracles/llama-cpp.md`](../oracles/llama-cpp.md) records, so the split is
stated rather than assumed. The type, the loader ftype, the CPU quants, the CUDA
MMQ instantiation and the ModelOpt-to-GGUF converter are **upstream**: they are
present in `ggml-org/llama.cpp` `master` at `0d9ceae1e` as well
(`GGML_TYPE_NVFP4 = 40` at the same `ggml/include/ggml.h:430`, and
`conversion/base.py:699` there). The two GB10 prefill GEMMs, patches 0034 and
0035, are **fork-local** at our pin. Upstream llama.cpp already runs NVFP4 on
CUDA without them.

**Why the verdicts still hold: the container, not the quantization.** llama.cpp
reads GGUF and only GGUF. vLLM at `555967922` reads safetensors and has no GGUF
reader at all, per section 2.1. So for every quantization, including NVFP4, there
is no single artifact both engines can open. The barrier is the container
format, and the previous wording never made that argument. SGLang is the same
disjunction from the other side: it does have a GGUF loader, but section 2.2
shows that loader cannot reach `qwen3_5`, and on the NVFP4 side it reads
safetensors while llama.cpp reads `block_nvfp4`.

**Could an `us` / vLLM / llama.cpp NVFP4 comparison exist?** The honest answer is
that it is closer than the old wording implied and is still not available, for
two reasons of different weight.

The conversion side is unusually favourable, and this is worth recording because
it is not the ordinary re-quantization case. llama.cpp's converter does not
re-quantize an NVFP4 checkpoint. `_nvfp4_pack` (`conversion/base.py:654-676`)
**repacks** ModelOpt and compressed-tensors NVFP4 tensors into the ggml
super-block layout: it unpacks the same nibble-packed e2m1 codes (`:664-665,669`)
and regroups four 16-element blocks into one 36-byte super-block (`:671-675`).
The per-tensor `scale2` and `input_scale` are written out as separate tensors and
applied at inference (`:657`, `:685-686`). So a conversion-equivalence proof here
is a **layout** proof over identical code points, not a numerical-agreement
argument across two independent quantizers, which is what the umbrella spike's
fallback rule was written for.

Two things stop it from being free, and both are named rather than waved past:

1. `_nvfp4_pack` preserves the original e4m3 scale bits "as UE4M3 (strip sign
   bit)" (`conversion/base.py:656,668`). That is lossless only if every scale is
   non-negative. Nothing in the pinned tree proves that, so a conversion
   equivalence proof owes an assertion over the actual checkpoint that no scale
   byte has its sign bit set.
2. `Qwen3_5Model._transform_nvfp4_weight` (`conversion/qwen.py:378-386`)
   **permutes** five GDN projections before repacking, to reorder value heads
   inside key-head groups. It is a permutation of the same codes, so it is
   value-preserving, but the proof has to cover it explicitly rather than assert
   the file is a byte-for-byte carry.

The blocking reason is simpler and is on our side. Our NVFP4 arm for this
checkpoint is owed under [#821](https://github.com/mudler/vllm.cpp/issues/821),
so no three-way NVFP4 cell can run today regardless of the conversion argument.
When #821 lands, the `ours vs llama.cpp` NVFP4 pair becomes reachable on a
converted GGUF and is `converted-nonbinding` under the umbrella rule until the
two proofs above are produced, at which point it is worth arguing for binding
status in its own change. The `vLLM vs llama.cpp` cell does not benefit at all,
because it needs vLLM to read a GGUF and vLLM cannot.

**Recorded consequence.** Both cells stay NOT COMPARABLE. The reason on the
record is now the container disjunction plus a conversion this spec does not
accept as binding, and no longer a false statement about llama.cpp's
capabilities.

### 2.6 The NVFP4 column presumes a revision it has not inspected

The `unsloth/Qwen3.8-27B-NVFP4` revision `a767244d` and
`unsloth/Qwen3.8-27B-GGUF` revision `fe1e2a23` appear in no other file in this
repository. Nothing here has read either one. The NVFP4 column of section 2 is
built from the four engines' source, and the assumption that `a767244d` actually
carries NVFP4 weights rests on the repository name alone.

That assumption has already failed once for this vendor.
`docs/BENCHMARKS.md:50` records the same `unsloth` repository family silently
re-quantized: revision `ccdaab7e` of `Qwen3.6-27B-NVFP4` is "the same repo name
re-quantized to FP8 W8A8 throughout, not NVFP4", which is why that gate is
revision-pinned rather than name-pinned.

Section 10's first item, the sorted `sha256sum` manifest for every resolved file
of all three artifact families, is what discharges this. It is a precondition of
quoting any number in this campaign, and the resolved quantization from item 2
has to be checked against the name before a cell is filled in.

## 3. Drafted or raw, declared per arm

SGLang published **38.28 tok/s decode on DGX Spark** for Qwen3.8-27B. Its
wording is "our NVFP4 plus DSpark", so it is a **speculative-decoding** result.

**Provenance, so the number is retrievable rather than remembered.**

| Field | Value |
|---|---|
| Primary source | `@sgl_project` on X, post `2088281320422322413`, <https://x.com/sgl_project/status/2088281320422322413> |
| Date | 2026-08-14 UTC. Derived from the status ID, because the post itself returned HTTP 402 to this session and was read only through a search index. |
| Quoted wording | "Day-0 support is live in SGLang: 206.1 tok/s decode on a single RTX 5090, with our NVFP4 plus DSpark, 38.28 tok/s decode on DGX Spark" |
| Corroborating source | NVIDIA Developer Forums thread 380257, "Qwen3.8-27B at 34-38 tok/s on DGX Spark, open-source one-command setup (SGLang + NVFP4 + DSpark)", <https://forums.developer.nvidia.com/t/qwen3-8-27b-at-34-38-tok-s-on-dgx-spark-open-source-one-command-setup-sglang-nvfp4-dspark/380257> |
| Corroborating date and handle | 2026-08-15, `basbunarhasan`. Retrieved and read on 2026-08-16. |
| What that thread says | "~34 tok/s real-world, 38.0 average on eval-style workloads, 46.7 peak (GSM8K-style)", and a reply from `helge` reporting 32.8 tok/s on code against 19.5 tok/s on multilingual mixed prose |

Note that the corroborating thread's own headline figure is 38.0 on eval-style
workloads, not 38.28, and that its highest and lowest quoted values differ by
more than a factor of two on content alone.

**UNVERIFIED and not to be repeated as fact until sourced:** the drafter's
parameter count, whether TTFT is excluded from that figure, and the batch size.
The 46.7 tok/s upper end and the 19.5 tok/s low are the forum thread's, on
declared-but-unspecified workloads, and neither is a reproduction of the 38.28
configuration. None of this appears in any artifact this project holds. Outside
this section the project's own record contains **zero** occurrences of the string
`38.28`, which is exactly why it is written down here with its provenance
attached rather than carried forward as a target.

Our binding quantized-27B cell is 10.756 against vLLM's 11.250 at c1
(`docs/BENCHMARKS.md:97-98`, Qwen3.6-27B NVFP4). The anchor covers both rows on
purpose, because the refusal below turns on the pair rather than on either half:
`:97` is our own row and carries 10.756, `:98` is the vLLM 0.25.0 row and carries
11.250, and `:96` is the table separator. That is a **raw** decode number.
Dividing 38.28 by it compares a drafted arm against a raw one and is refused by
this spec.

**And that contrast on its own leaves a reader with the wrong picture, so the
measured head-to-head belongs here too.** A reader who meets "SGLang published
38.28" beside "ours is 10.756", with drafted-versus-raw as the only objection,
can leave believing SGLang is simply the faster engine. This project has
measured the two engines directly, and it is not what the measurement says.

On 2026-07-28, under `CLAIM-SGLANG-PERF-BENCH`
([sglang-matrix.md](../sglang-matrix.md), "Perf oracle results"), SGLang v0.5.15
and ours ran **byte-identical NVFP4 weights** on the same idle GB10 under one
lock, driving the identical deterministic corpus, both emitting exactly 80 by 128
output tokens with zero errors, three repetitions each:

| Axis | c16 SGLang | c16 ours | c16 | c8 SGLang | c8 ours | c8 |
|---|---:|---:|---|---:|---:|---|
| Output throughput tok/s | 40.8 | **90.3** | **2.21x ours** | 40.8 | **58.8** | **1.44x ours** |
| Mean TTFT ms | 33425 | **2980** | **11.2x ours** | 11289 | **1775** | **6.4x ours** |
| Mean TPOT ms | **104.0** | 154.4 | 0.67x, our gap | **104.0** | 122.9 | 0.85x, our gap |
| Mean ITL ms | **105.6** | 154.4 | 0.68x, our gap | **105.7** | 122.9 | 0.86x, our gap |

Both directions are load-bearing. Ours wins aggregate throughput by 2.21x at c16
and 1.44x at c8 and wins TTFT by 6x to 12x. **SGLang wins the steady-state
per-token decode-latency axis at the configuration this table measured**, and
that TPOT and ITL result is reproduced rather than explained away.

**Its cause is attributed, which this section understated until 2026-08-16.** A
same-day follow-up, `CLAIM-DECODE-LATENCY-EXPLORE`
([sglang-matrix.md](../sglang-matrix.md) lines 210-224, full curve in
[decode-latency-lever.md](decode-latency-lever.md)), confirmed by direct
measurement that the gap is **batch composition, not per-token kernel cost**.
Sweeping our own decode batch over 1, 2, 4, 8 and 16 on the same 27B-NVFP4 arm
gives an ITL of 101.75 ms at batch 1, already at or below SGLang's 104 to 105 ms
operating point, rising monotonically to 158.5 ms at batch 16. nsys shows every
hot decode kernel sublinear in batch, 1.6x to 1.8x wall time for 16 times the
tokens, so per-token GPU cost falls rather than rises. SGLang's effective decode
concurrency at its own operating point is about 4 rather than 16, because its
prefill-first admission queue keeps few requests decoding at once. The throughput
win and the ITL loss are therefore the same lever, and the knob is named and
already exists: `max_num_seqs` and `max_num_batched_tokens`, with a
latency-oriented point at `max_num_seqs` about 8 measured at 21 percent lower ITL
while still holding 1.38x SGLang's throughput.

**What that does and does not settle.** Our shipped default stays
throughput-oriented and is unchanged, so at the configuration the table above
measured the deficit is real and stands as recorded. What changes is its
description: it is a declared trade with an attributed cause and a named knob,
not an unexplained per-token deficiency. No ceiling is declared either. The batch-1
point already sits at SGLang's operating point, so what a latency-oriented default
would cost on the other axes is open and unmeasured.

**This is not this campaign's subject and is not a rebuttal of 38.28.** It is
Qwen3.6-27B-NVFP4 at `890bdef7`, cache-neutral, c8 and c16 only, both arms
**raw**. It says nothing about Qwen3.8-27B, nothing about c1, and nothing about a
drafted arm. It is placed here for one purpose: a published single number from
one engine is not the state of the comparison, and this project's own direct
measurement of the same two engines is the nearest thing it holds to one.

**Rule.** Every arm declares `drafted` or `raw` in its manifest before it runs. A
cell whose two arms disagree on that field is void, not a ratio. We hold the
technique on our side: `SPEC-DFLASH` is `DONE`, and `SPEC-DSPARK` is `ACTIVE` with
W1 through W8 landed and GPU-gated
([dspark-spec-decode.md](dspark-spec-decode.md)).

**No DSpark speculator ships in `python/sglang/srt/speculative/` at our pin.**
`f63458b5` carries DFlash, EAGLE, ngram and frozen-KV MTP there, and nothing
named `dspark`. That is the claim, and it is the same one
[`../oracles/sglang.md`](../oracles/sglang.md) makes.

**Two wider claims that stood here until 2026-08-16 are withdrawn**, because
neither survives a check against the pin. The same overreach reached
`.agents/issue-index.md:266` ("DSpark does not exist at our SGLang pin"),
[`../oracles/sglang.md`](../oracles/sglang.md) ("the pin also predates SGLang's
DSpark speculator"), and the body of commit `17187f134`. Both records are
corrected in place, because neither has landed. The commit body is the one carrier
that cannot be, see the note under `## Now`.

- "A repo-wide search for `dspark` returns nothing" is false.
  `git grep -il dspark f63458b5` returns `docs_new/index.mdx`, tracked at the
  pin, with four hits at `:86,107,108,127` linking
  `lmsys.org/blog/2026-07-06-dspark-sglang/` and naming "DSpark in SGLang:
  Speculative Decoding with Confidence-Driven, Variable-Length Verification".
- "The announcement says Day-0 support, so that code postdates the pin" is
  contradicted by the pin's own documents. That blog post is dated 2026-07-06,
  three days **before** the pinned tree itself, because `f63458b5` is dated
  2026-07-09. The `pinned_on = 2026-07-27` in
  [`../oracles/sglang.md`](../oracles/sglang.md) is when this project recorded
  the pin, not when the pinned commit was written, and the earlier wording here
  compared the blog against the recording date. Day-0 support for the Qwen3.8-27B
  checkpoint is not the same event as DSpark's arrival, and this spec conflated
  them.
- "Not in `speculative/`" is also not the same as "not reachable". `f63458b5`
  ships a plugin registration API for out-of-tree speculative algorithms:
  `SpeculativeAlgorithm.register` (`python/sglang/srt/speculative/spec_info.py:60-70`)
  over the `CustomSpecAlgo` storage in `speculative/spec_registry.py:24-56,189-222`,
  whose own docstring says plugins register through that classmethod. An
  algorithm can therefore be present at runtime while absent from the directory
  listing.

**The operational conclusion is unchanged, and now rests on an argument rather
than on an absence.** A drafted SGLang arm needs a configuration this project
can pin, name and re-run. Three things are missing for that, independently of
where the code lives. No `dspark` implementation is in the pinned tree. If it
arrives as a plugin then the plugin is a second unpinned upstream needing its own
`.agents/oracles/` record, under the same rule that refuses `vllm-gguf-plugin` in
section 2.1. And the pinned v0.5.15 image this project has actually run
(`lmsysorg/sglang:v0.5.15-cu130@sha256:d0a667e`) is what section 5.2 measured
gateability on. Reaching the published configuration therefore means
**advancing the SGLang oracle pin, or pinning a plugin, as its own reconciled
change**, with every affected row reconciled first. It is not a substitution made
inside a measurement run.

Until that happens, the SGLang arms in this campaign are `raw`, and the only
honest statement about 38.28 is that it belongs to a configuration we have not
pinned.

## 4. What changing the subject costs

| Carries over unchanged | Must be re-established for Qwen3.8-27B |
|---|---|
| The `serve-low` workload shape: concurrency 1, 2, 4, 8, 16, 1024 in and 128 out, ignore EOS, seed 0 | Every checkpoint file manifest and `sha256sum` set, for all three artifact families |
| The whole P1 harness under `tools/bench/*serve_low*` and `scripts/dgx-sglang-low-concurrency.sh` | The deterministic token-ID corpus, because the tokenizer snapshot changes with the checkpoint |
| The vLLM parity pin `555967922`, unchanged by this campaign | The four-way tokenizer agreement preflight over every prompt |
| The one-lock, sequential-arm, teardown-verify execution discipline | Any checkpoint-equivalence proof, which was written against the Qwen3.6 snapshots |
| The `require_complete_request_set` refusal and the `failed == 0` precondition | Every SGLang and llama.cpp load and quantization-path classification |
| The clock-pinning and idle-host protocol | The vLLM production-config denominator for this checkpoint |

The token gate itself does **not** need redoing: [#915](https://github.com/mudler/vllm.cpp/issues/915)
closed it for bf16, 4 of 7 prompts strict and all three first divergences exact
fp32 ties against the pinned oracle. The NVFP4 and Q4_K_M token gates are owed
under [#821](https://github.com/mudler/vllm.cpp/issues/821) and are preconditions,
not products, of this campaign.

## 5. Record reconciliation, with evidence

### 5.1 `BACKEND-GATE-CUDA-SGLANG`, `BLOCKED` on a dependency that closed

`backend-matrix.md:260` recorded `BLOCKED on SERVE-ASYNC-LLM` with the evidence
cell "no binding run; HTTP TTFT/ITL cannot be measured honestly yet". Four
independent lines of evidence say that reason no longer holds.

1. **The production path streams incrementally.** `serving_completion.h:9-11` says
   the live pull-based `SseStream` over `AsyncLLM` is the production path and the
   buffered `LLMEngine` constructor is a test seam. `api_server.cpp:971-981` takes
   `result.sse_stream` and drives `set_chunked_content_provider` off
   `SseStream::next`, one chunk at a time.
2. **The harness enforces it rather than assuming it.**
   `tools/bench/run_serve_low.py:296-310` refuses a probe that produced no
   token-bearing event, refuses a chunk count other than the requested completion
   length, refuses `first_chunk_s >= total_s`, and refuses a spread below a
   configured floor.
3. **It was demonstrated on hardware.** The 2026-07-28 floor run
   (`sglang-matrix.md`, `CLAIM-SGLANG-PERF-BENCH`) measured our c16 mean TTFT at
   2980 ms against a mean ITL of 154.4 ms over 128 tokens. First byte therefore
   preceded completion by roughly twenty seconds. A buffered server cannot produce
   that shape.
4. **The record says so elsewhere.** `engine-matrix.md:207` carries
   `SERVE-ASYNC-LLM` as `GATING` with live completion and chat SSE, disconnect
   abort and deterministic c32 capacity, and `async-metrics.md:196` records the
   `SERVE-ASYNC-LLM` frontend plus `ENG-CORE-BUSY-LOOP` as `DONE`.

[#931](https://github.com/mudler/vllm.cpp/issues/931), landed as `638eba27f`,
strengthens this rather than establishing it: it defaults `VT_SERVER_SSE_PING_S`
to 0 (`serving_utils.h:40`, `serving_utils.cpp:254,278`) so vLLM's own bench
client can parse our stream, and it adds
`require_complete_request_set` (`tools/bench/serve_low_common.py:234`) so no rate
is derived from an incomplete request set.

**New state: `PARTIAL`, not `DONE` and not still `BLOCKED`.** The named blocker is
discharged and partial evidence exists. What is still missing, named:

- The P2 exact-equivalence classification of the image, the model and the GPU,
  which the sibling spike's own dependency table carries and which its
  reconciliation addendum lists first among the residuals.
- The c1, c2 and c4 points. Only c8 and c16 ran, and SGLang c1 measured about
  13.3 seconds per iteration, so three-repetition reproduction there is a real
  scheduling problem and not an oversight.
- The vLLM arm in the same series. The 2026-07-28 run was ours against SGLang
  only.
- The 35B-A3B arm.
- The SGLang token-ID cross-check. `SGLANG-ORACLE-CORRECT` is still `INVENTORIED`,
  so SGLang binds as a floor only for a model whose correctness gate passed.
- Paired warmup-excluded nsys traces at c1 and c16.
- Every point on Qwen3.8-27B, which is this campaign's subject and has no SGLang
  evidence at all.

### 5.2 The SGLang oracle recorded `gateable = no` after it had already run here

`.agents/oracles/sglang.md` said "no SGLang run has been recorded on this
project's hardware" and "Source has been read; nothing has been executed".
`.agents/sglang-matrix.md:162` records the opposite: on 2026-07-28 the
`lmsysorg/sglang:v0.5.15-cu130@sha256:d0a667e` arm64 image was pulled and ran
`unsloth/Qwen3.6-27B-NVFP4` @ `890bdef7` on GB10 sm_121a with CUDA graphs
captured, three repetitions at c8 and c16, both arms emitting exactly 80 by 128
output tokens with zero errors. `docs/STATUS.md:180` carries the same
measurement. `docs/BENCHMARKS.md:485` still said "SGLang floor arms | Never ran".

AGENTS.md sets the bar at "demonstrably builds and runs the model". The image
needed no from-source build and it ran the model. **`gateable` moves to `yes`**,
with `.agents/sglang-matrix.md` as the evidence path. This discharges the SGLang
**third** of the three gateability debts
[#647](https://github.com/mudler/vllm.cpp/issues/647) holds open. That row names
them at `.agents/issue-index.md:193`, and they are `sglang`, `diffusers` and
`tt-forge`, so two remain after this change. This spec said "half" until
2026-08-16, which was simply a miscount.

**The 2026-07-28 run predates the #931 fix and is not voided by it.** The
keepalive then fired only after 15 seconds with no output on a request
(`include/vllm/entrypoints/openai/serving_utils.h:40 @ 638eba27f~1`, and the
`AssignSseWaitResult` call sites). That anchor is deliberately historical.
`638eba27f` replaced the 15-second default with 0, so at HEAD the same file
records the default as off at `:40-41`, and a reader who follows a HEAD anchor
finds the opposite value and concludes this argument is wrong. That run's worst
observed p99 TTFT was 7220 ms at c16 and 3589 ms at c8, with ITL near 154 ms, so
no frame could have been emitted, and the harness independently recorded zero
errors on every leg. This is stated because a silent assumption in the other
direction is precisely the failure #931 documents.

### 5.3 The llama.cpp-on-CUDA comparator had no owning row

`BACKEND-GATE-CPU-LLAMACPP` is the CPU floor. `BACKEND-GATE-CUDA-LLAMACPP-LEGACY`
is scoped to Pascal, Volta and Turing, which is where vLLM has no entry at all.
Neither covers llama.cpp CUDA on a current card, which is what this campaign's
fourth arm is. `bench-27b-five-way.md` already hit this and listed the arm as
"building" with no row behind it. `BACKEND-GATE-CUDA-LLAMACPP` is added
`INVENTORIED`, with no run and no number.

## 6. Protocol

Everything here is inherited from [competitive-benchmarks.md](competitive-benchmarks.md)
and [cuda-sglang-low-concurrency.md](cuda-sglang-low-concurrency.md) unless it is
stated as a change.

- One `flock $GPU_LOCK` for the whole multi-arm series. Engines run strictly
  sequentially with a teardown-verify between legs, because a 27B bf16 resident
  set is about 55 GB against a 119 GB unified pool and two engines cannot coexist.
- vLLM's **production graphed** configuration is the denominator. `--enforce-eager`
  never appears in a measured arm.
- `gpu_memory_utilization` reserves host RAM on GB10 and has hard-rebooted this
  box. Keep it low and never run a second GPU consumer alongside.
- Clocks pinned and recorded per leg, with an always-fires reset trap.
- Three repetitions per point, interleaved, with spread and coefficient of
  variation reported alongside every median.
- `failed == 0` asserted on every leg, and every rate derived through
  `require_complete_request_set`.
- Each cell names the quantization on **both** sides and the `drafted` or `raw`
  field on both sides, or it is not a cell.
- Raw per-repetition JSON, server logs, manifests, commands, engine revisions and
  contention state retained under the claim's evidence root.

## 7. Gates

| Gate | Pass condition |
|---|---|
| Pins | Every engine resolves to its recorded pin, and each pin has an `.agents/oracles/` record. A pin advance is a separate reconciled change. |
| Quantization | Every cell states one common denominator on both sides, or carries an explicit not-comparable verdict with the source reason. |
| Correctness | Ours is token-exact against the pinned vLLM oracle on the arm being timed, or falls under the ratified near-tie protocol. A comparator binds as a floor only after its own correctness cross-check passes. |
| Spec-decode declaration | Both arms of every cell declare `drafted` or `raw` and agree. |
| Completeness | `failed == 0` on every leg and no rate derived from an incomplete request set. |
| Streaming validity | First chunk precedes completion and the spread floor holds, per the harness preconditions. |
| Reproduction | Three repetitions, spread reported, unexplained outliers re-run on an idle host. |
| Record | Every accepted or explicitly pending cell reaches `docs/BENCHMARKS.md` and the owning matrix row in the state-changing commit. |

## 8. Risks

| Risk | Handling |
|---|---|
| The 38.28 figure is treated as a target and drives work | Section 3 gives it a retrievable citation, names the unverified parts, and puts this project's own measured head-to-head beside it. It is never a denominator. |
| A reader takes 38.28 versus 10.756 as the state of the comparison | Section 3 records the direct measurement on byte-identical NVFP4 weights: ours 2.21x at c16 and 1.44x at c8 on output throughput, 6x to 12x on TTFT, with TPOT and ITL a reproduced result in SGLang's favour whose cause `CLAIM-DECODE-LATENCY-EXPLORE` attributed to batch composition and whose knob is named. Its narrower scope is stated in the same paragraph. |
| Someone adds the two-line SGLang GGUF alias and reports a load | Section 2.2 names three further blockers, and separately proves that the load path has **no completeness guard** at `loader.py:2149-2153`, `weight_utils.py:1280,1321` and `qwen3_5.py:1359-1412,1405`. A load that prints no error is not evidence that every parameter arrived. |
| A false capability claim survives because it is quoted rather than checked | Two of them did, for weeks: llama.cpp NVFP4 recorded as absent, and `conv1d` recorded as silently wrong by a citation that says the opposite. Both are re-derived at the pin in sections 2.2 and 2.5, with the withdrawn wording kept visible. |
| Someone installs `vllm-gguf-plugin` to create a vLLM GGUF cell | That plugin is unpinned and has no oracle record. Pinning it is a separate decision. |
| The SGLang pin is advanced mid-campaign to reach DSpark | Forbidden here. Advancing it reconciles every affected row first. |
| bf16 GGUF is substituted to manufacture a vLLM-versus-llama.cpp cell | A converted checkpoint is `converted-nonbinding` under the umbrella spike's fallback rule and cannot produce a ratio. |
| A prior Qwen3.6 artifact is reused for the Qwen3.8 subject | Section 4 lists exactly what carries over. Manifests, corpora and equivalence proofs do not. |
| Two engines resident at once on the unified pool | Sequential arms with teardown-verify, one lock, memory returned to a recorded baseline between legs. |

## 9. Tests

This spec adds no product code and therefore no product test. The executable
obligations it depends on already exist and are named so a reviewer can mutate
them:

- `tests/tools/test_request_set_completeness.py` pins the incomplete-request-set
  refusal.
- `tests/vllm/entrypoints/openai/test_sse_keepalive.cpp` pins the #931 frame
  behaviour.
- `tools/bench/run_serve_low.py:296-310` carries the incremental-streaming
  precondition, and `tests/tools/test_serve_low_client.py` covers the client
  contract.
- `tests/scripts/test_check_oracle_pins.py` pins the `gateable` semantics that
  section 5.2 exercises.

## 10. Evidence required before any number is quoted

1. Sorted `sha256sum` manifest for every resolved file of all three Qwen3.8-27B
   artifact families. This is also what discharges section 2.6: the NVFP4 column
   of section 2 presumes `a767244d` is NVFP4 on the strength of its repository
   name, and that presumption is not carried into any number.
2. Startup log, resolved architecture, resolved quantization and weight-loader
   warnings for every engine and every arm. The resolved quantization is compared
   against the artifact name before a cell is filled in, per section 2.6.
3. Four-way tokenizer agreement on every corpus prompt, with stored token IDs and
   hashes.
4. Native output IDs per engine, never a detokenize-and-retokenize round trip.
5. Per-leg clock state, thermal state, idle proof and contention state.
6. Raw per-request arrays for all three repetitions, plus the recomputed
   percentiles and the formula version.
7. The `drafted` or `raw` declaration and, when drafted, the drafter identity,
   revision and k.

## 11. Stop conditions

- Stop and report if any pair's common denominator cannot be established from
  source. Record not-comparable. Do not substitute a nearby quantization.
- Stop if an arm's `failed` count is non-zero. That leg is void.
- Stop if reproducing a point needs a pin advance. Reconcile the pin in its own
  change.
- Stop if the host is not idle, or if a co-tenant appears mid-series. Discard the
  series rather than annotate it.

## Now

`SPIKE`. This spec, the issue, and the three record corrections in section 5 are
the whole deliverable. Nothing is measured. `BACKEND-GATE-CUDA-SGLANG` moves
`BLOCKED` to `PARTIAL` because its named blocker is discharged and partial
evidence exists. `BACKEND-GATE-CUDA-LLAMACPP` is added `INVENTORIED` with no run.
The SGLang oracle moves to `gateable = yes`.

**Corrected 2026-08-16, after a fresh review, before any measurement.** Four
source facts this spec asserted were re-derived at the pins and three of them
were wrong: llama.cpp does have NVFP4 at `237ad9b96` (section 2.5), SGLang's
`conv1d` is not silently quantized and the polarity was inverted (section 2.2),
and `dspark` does appear at `f63458b5`, in the pin's own documents, dated three
days before the pinned tree itself (section 3). The two not-comparable verdicts
are re-derived and survive on the container disjunction. Sections 2.6 and 3 add
the artifact-name caveat, the citation for 38.28, and this project's own measured
SGLang head-to-head. Section 3 also stops understating that head-to-head: the
TPOT and ITL result has an attributed cause and a named knob, from a same-day
follow-up this spec had not reconciled.

**The index row is corrected in place, and the reason the earlier round thought
it could not be is worth recording, because that reason inverted the premise it
acted on.** `.agents/issue-index.md:266` carried both withdrawn claims, and the
previous round recorded the row as uncorrectable on the grounds that the index is
append-only and a landed row is never edited. **That row has not landed.**
`git merge-base --is-ancestor 17187f134 origin/main` is false, `origin/main`
carries 265 index lines and zero `#979` rows, and no other local or remote branch
carries one. The row exists only on this branch, introduced by this branch's own
unmerged commit.

The append-only rule's own rationale is what settles it. `AGENTS.md` and
`scripts/check-issue-index-append-only.py:2-16` both give the same reason: under
`merge=union` an EDITED line is duplicated rather than merged. That hazard needs
the line to exist at the merge base. This one does not, and the checker measures
exactly that, diffing the merge base against the head rather than reading the
tree. It passes on the edit, and `scripts/check-agent-record.py` passes with it.
The duplicate-row refusal the record checker carried while the live index existed
(retired with it by W6, `7dc2ef1ea`) is real and was
correctly described, and it is simply the wrong instrument here, because this
corrects the first row before it lands rather than appending a second.

**The commit bodies are the one carrier that stays wrong.** `17187f134` and
`dadb3d396` both assert the withdrawn wording, and `dadb3d396` additionally
asserts that the index correction is impossible. A commit body cannot be edited
without a rewrite this branch is forbidden to perform. Under
`squash_merge_commit_message = PR_BODY` no individual commit body reaches `main`
on a squash in any case, so the pull request body is the correction of record for
those, and sections 2.2 and 3 here are the correction of record for the substance.

**FIRST MEASURED CELL, 2026-08-19, and one pair that cannot exist.** The
`ours vs vLLM` bf16 pair of section 2.4's matrix has now been attempted on
`Qwen/Qwen3.8-27B` @`1d4bf0f2`, both arms in an `rc` lease on `dgx:gpu0`, and
the two concurrencies resolve differently:

- **c1 produced both absolutes and no ratio.** Ours 4.4040 tok/s, vLLM 4.2835
  tok/s, three reps each, every request completed on both arms.
  `gpu_clock_state compare` returned `PAIRING_VERDICT=DISCARD` on all three
  pairings because the within-run SM-clock spread breached the 5% ceiling on
  both arms, so the ratio is OWED rather than derived.
- **c8's vLLM denominator is NOT MEASURABLE on this box at the recorded
  configuration**, and that is the cell's answer rather than a gap in it. The
  server reached `/health`, then the KV reservation took 48,715 MB in one
  4-second window and the worker died with 6,261 MB left. Every way to create
  the missing headroom is an engine knob that would change the denominator.
  This is a statement about this box, not about vLLM.

The measurement, the clock blocks, the memory trajectory and the two findings
that outlive the campaign are in
[`../benchmark-record.md`](../benchmark-record.md).

**A campaign premise is falsified in passing.** `.agents/environment.md`
recorded that the pinned oracle builds inside a lease but that "nobody has run a
model that way, so no oracle-side MEASUREMENT is unblocked yet"
([#1185](https://github.com/mudler/vllm.cpp/issues/1185)). It ran. Three clean
c1 legs of `vllm serve` on a 52 GiB checkpoint, from a lease, with no `ssh` and
no container image. What is still blocked is `sglang`, which needs the image
path [#1265](https://github.com/mudler/vllm.cpp/issues/1265) forbids, and the
c8 point on this hardware.

## Owed

- [#979](https://github.com/mudler/vllm.cpp/issues/979) owns this campaign and is
  listed here so the index row and this spec agree.
- [#821](https://github.com/mudler/vllm.cpp/issues/821) owes our NVFP4 and Q4_K_M
  arms for this checkpoint, including the separate `clip` projector. Three cells
  in section 2.4 are blocked on it.
- [#915](https://github.com/mudler/vllm.cpp/issues/915) owes the c1 and c8
  re-measure on a binary carrying the #931 fix.
- The SGLang token-ID correctness cross-check, `SGLANG-ORACLE-CORRECT`, is
  `INVENTORIED` in [sglang-matrix.md](../sglang-matrix.md).
- Advancing the SGLang pin past v0.5.15 to reach DSpark, or pinning it as a
  plugin upstream with its own `.agents/oracles/` record, if a drafted SGLang arm
  is ever wanted. Unowned today and deliberately not started here.
- Which Qwen3.5 parameters actually fail to resolve in SGLang's GGUF name map
  (section 2.2). Unowned, and only answerable by instrumenting `loader.py:2152`
  against a pinned `gguf` and `transformers`, neither of which this project pins.
  Recorded as an open question rather than asserted.
- A conversion-equivalence proof for an `ours vs llama.cpp` NVFP4 cell, blocked
  behind [#821](https://github.com/mudler/vllm.cpp/issues/821) and owing the two
  assertions named in section 2.5: no e4m3 scale byte with its sign bit set, and
  the GDN head permutation at `conversion/qwen.py:378-386` shown to be a
  permutation of the same codes.
- [#1354](https://github.com/mudler/vllm.cpp/issues/1354): **clock pinning is
  unavailable inside an `rc` lease**, so every remaining pair in section 2.4 has
  the same exposure this one hit — a within-run spread the clock gate refuses,
  with no lever to reduce it. Recorded in
  [`../environment.md`](../environment.md).
- Whether the HOST rebooted or only the k3s pod was lost when the c8 vLLM worker
  died. Read `/proc/sys/kernel/random/boot_id` inside a later `dgx:gpu0` job and
  compare against `3fd9745a-d25a-426c-ba3c-97c958a85515`.
- A c8 vLLM denominator for this checkpoint from a box with more than 6-7 GB of
  headroom at `--gpu-memory-utilization 0.85 --max-num-batched-tokens 8192`.
  Unowned, and deliberately not obtained by tuning either knob.
