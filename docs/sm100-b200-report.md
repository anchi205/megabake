# Experimental SM100 inference on one B200

This change adds an opt-in SM100 TMA/`tcgen05` GEMM path, shape-based and
model-profiled tile selection, split-K reduction, supported GEMM epilogues,
and wider prefill token tiles. It also adds an optional grouped prefill
attention load and tools to measure and check complete models. The existing
warp-MMA body remains available. The TMA path is enabled only when
`MEGABAKE_SM100_TCGEN=1`; unsupported shapes use other available bodies.

These are experimental measurements, not a production performance or
correctness claim. Some generated sequences differ from native PyTorch eager,
and the PyTorch graph used for comparison can differ as well. SM90 hardware was
not tested after the shared compiler and attention changes. Only one SM100 B200
was tested; the results do not establish support on other Blackwell devices.

## Rebase validation on 2026-11-03

The PR was rebased onto `9af2847ca72b32653d1907bb5f75fc26aae7fd99` and
measured on one B200 with PyTorch 2.14.0+cu130, CUTLASS DSL 4.8.0 and driver
595.71.05. The tuning profiles use source fingerprint
`06db1fd416166920cd55108ecc5281df80b8126d91ee50151941ec569f4bf731`.
All eight models were tuned independently over feature tiles 64 and 128 and
split limits 1 and 8; each tuning run reported zero failed candidates. The
measurements below use those selected configurations for decode. Prefill
instead uses token tile 32, split-K 1, maximum 128 rows, and grouped attention
value block 8. BF16 and batch size one are used throughout.

Complete-call times are synchronized host medians in milliseconds, with 24
samples and five warmups for decode and 12 samples and three warmups for
prefill. Compilation is excluded. The decode benchmark creates its position
tensor inside each timed call and uses the explicit cache wrapper described
below. The wrapper eager and ordinary compiled paths synchronize on host cache
lengths; they are not native Hugging Face serving timings. The graphed PyTorch
decode path uses static masked attention while MegaBake uses its cache body.
The prefill benchmark disables KV caching and uses a 128-token repeated prompt.
The traces contain `cudaGraphLaunch` for the two graphed decode paths and for
the successful PyTorch reduce-overhead and MegaBake graphed prefill paths; the
other variants below run without graph replay.

| Decode model | Wrapper eager | `torch.compile` | Compiled graph | MegaBake | MegaBake graph |
| --- | ---: | ---: | ---: | ---: | ---: |
| SmolLM 135M | 22.727 | 8.862 | 2.603 | 1.883 | 1.406 |
| Gemma 3 270M | 16.185 | 6.629 | 1.978 | 2.930 | 2.415 |
| Qwen2 0.5B | 14.747 | 8.166 | 2.609 | 2.112 | 1.572 |
| SmolLM 1.7B | 17.254 | 7.305 | 2.494 | 2.589 | 2.217 |
| DeepSeek draft 0.6B | 14.254 | 7.860 | 2.227 | 2.090 | 1.550 |
| Qwen2.5 0.5B Instruct | 14.093 | 7.809 | 2.239 | 2.107 | 1.583 |
| Gemma 2B | 11.532 | 6.155 | 2.431 | 3.709 | 3.215 |
| Qwen3.5 0.8B | 17.387 | 5.646 | 2.293 | 5.042 | 4.459 |

| 128-token prefill model | Eager | `torch.compile` | Compile reduce-overhead | MegaBake | MegaBake graph |
| --- | ---: | ---: | ---: | ---: | ---: |
| SmolLM 135M | 13.660 | 5.105 | 1.979 | 4.008 | 3.505 |
| Gemma 3 270M | 13.827 | 2.900 | 1.519 | 5.479 | 4.990 |
| Qwen2 0.5B | 11.861 | 4.467 | 1.949 | 5.796 | 5.286 |
| SmolLM 1.7B | 10.784 | 3.557 | 2.443 | 9.687 | 9.244 |
| DeepSeek draft 0.6B | 12.231 | 4.457 | 1.940 | 5.738 | 5.252 |
| Qwen2.5 0.5B Instruct | 11.971 | 4.573 | 1.947 | 5.798 | 5.285 |
| Gemma 2B | 9.395 | 3.118 | 2.405 | 12.364 | 11.926 |
| Qwen3.5 0.8B | 29.694 | 8.077 | 4.686 | failed | failed |

Qwen3.5 prefill failed in MegaBake semantic lowering because `aten.cumsum`
and `aten.linalg_solve_triangular` are unsupported. PyTorch used reference
implementations without the optional `causal_conv1d` and `fla` packages.

An independent timer prepared device positions before timing and compared
the native Hugging Face eager cache with the explicit-cache paths. It used
five warmups and 24 samples. These times are a separate measurement, not a
direct repeat of the position-including calls in the first table.

| Decode model | Native eager | Wrapper eager | Compiled graph | MegaBake graph |
| --- | ---: | ---: | ---: | ---: |
| SmolLM 135M | 14.213 | 22.402 | 1.920 | 1.381 |
| Gemma 3 270M | 14.281 | 16.031 | 1.302 | 2.441 |
| Qwen2 0.5B | 12.274 | 14.188 | 1.678 | 1.546 |
| SmolLM 1.7B | 11.138 | 17.695 | 2.014 | 2.190 |
| DeepSeek draft 0.6B | 12.244 | 14.215 | 1.663 | 1.540 |
| Qwen2.5 0.5B Instruct | 12.273 | 14.224 | 1.679 | 1.554 |
| Gemma 2B | 9.724 | 11.562 | 2.000 | 3.221 |
| Qwen3.5 0.8B | 19.880 | 17.449 | 1.816 | 4.482 |

The same independent 24-sample timer was used in the earlier 2026-10-29
audit. The later run uses newly generated tuning choices and was measured on a
different day, so small changes should not be attributed to the rebase.

| Model | Earlier MegaBake graph | Rebased MegaBake graph |
| --- | ---: | ---: |
| Qwen2 0.5B | 1.557 | 1.546 |
| DeepSeek draft 0.6B | 1.558 | 1.540 |
| Qwen2.5 0.5B Instruct | 1.565 | 1.554 |
| Qwen3.5 0.8B | 4.520 | 4.482 |
| Gemma 2B | 3.164 | 3.221 |
| Gemma 3 270M | 2.388 | 2.441 |

The next table reports 128 teacher-forced decode positions against wrapper BF16
eager, plus a separate 40-token greedy run. The last column is the number of
five natural prompts for which all 16 generated tokens matched native eager.
The relative L2 values compare full logits, not probabilities; matching top
tokens does not mean the logits match.

| Model | Teacher top matches | Median / max relative L2 | First greedy difference | Natural prompts matching |
| --- | ---: | ---: | ---: | ---: |
| SmolLM 135M | 124/128 | 1.36% / 9.74% | 17 | 4/5 |
| Gemma 3 270M | 125/128 | 2.96% / 10.10% | none in 40 | 4/5 |
| Qwen2 0.5B | 128/128 | 1.24% / 2.18% | 11 | 5/5 |
| SmolLM 1.7B | 126/128 | 0.80% / 2.18% | 29 | 4/5 |
| DeepSeek draft 0.6B | 128/128 | 1.58% / 3.10% | none in 40 | 4/5 |
| Qwen2.5 0.5B Instruct | 127/128 | 2.38% / 6.29% | none in 40 | 3/5 |
| Gemma 2B | 128/128 | 1.45% / 21.01% | none in 40 | 4/5 |
| Qwen3.5 0.8B | 128/128 | 1.48% / 2.93% | none in 40 | 4/5 |

On the five-prompt audit, the compiled PyTorch graph matched all 16 tokens on
5/5, 5/5, 5/5, 4/5, 4/5, 4/5, 4/5 and 5/5 prompts respectively in the
table's model order. Qwen3.5 wrapper eager matched native eager on only 3/5
prompts. For
Qwen2, the compiled graph's maximum prompt logit relative L2 versus native
eager was 102.64%; its exact cause remains unresolved. All checked
logits were finite. MegaBake graph and non-graph runs had identical logits
and KV state over all 128 teacher-forced positions for each model.

For 128-token prefill, compiled PyTorch and MegaBake graph matched eager's
top token at 124/128 and 124/128 positions for SmolLM 135M, 120/128 and
124/128 for Gemma 3, 125/128 and 127/128 for Qwen2, and 126/128 and
126/128 for SmolLM 1.7B, 127/128 and 127/128 for DeepSeek, 126/128 and
127/128 for Qwen2.5, and 126/128 and 127/128 for Gemma 2B. Their full-logit
relative L2 differences were 2.03% and 1.63%, 5.13% and 2.78%, 5.40% and
1.24%, 1.16% and 0.80%, 6.36% and 1.72%, 5.58% and 2.43%, and 2.94% and
2.16%, respectively. For Qwen3.5 prefill, compiled PyTorch matched all
128 top tokens with 1.32% full-logit relative L2; MegaBake did not compile.

For SmolLM 135M, disabling split-K matched the 40-token greedy run, but
matched 123/128 teacher-forced top tokens. In the same no-split configuration,
the original PR commit and rebased source produced identical recorded
per-position logit error values and greedy sequences. Their separately timed
MegaBake graph medians were 1.569 ms and 1.566 ms, respectively. For the tuned
configuration, an FP32 arithmetic reference on BF16-rounded weights matched
the top token at 124/128 positions for BF16 eager, 126/128 for the PyTorch
graph and 126/128 for MegaBake; TF32 was disabled. This reference does not
recover the checkpoint's original precision.

The complete `tests/unit` and `tests/gpu` suites reported **279 passed,
37 skipped** and 291 warnings. The skips include Hopper-only tests. An added
split-K rebinding test was run separately: **4 passed**, including the new
split-K 8 case. A CPU-only import check with CUDA/CUTLASS imports blocked
passed. Ruff 0.17.3 lint and format checks passed for all changed Python files.
Hopper regression tests have not been run on Hopper hardware.

The raw logs, JSON files, profiler traces, and generated code were copied to
two local results archives, SHA-256
`eb99d06623bee65cd45540f5aa4ad61b7e1cd7345aaec213f5208fccfc3d358d` and
`d968e1e7a9a155e663be9701b53ebbc8d98155e1525f8573f8df2984337ebe51`.
They are not included in this PR. The missing Gemma 2B weight shard was
downloaded after authorization, and both shards matched the saved hashes.

## Earlier measurements from 2026-10-29

## Setup and scope

- One NVIDIA B200, compute capability 10.0, 148 SMs; BF16 and batch size one.
- Python 3.12, PyTorch 2.14.0+cu130, CUDA 13, CUTLASS DSL 4.8.0, driver 595.71.05.
- Measured on 2026-10-29 using an experimental checkout based on commit
  `81819a5284df175734b98ffd65ae654c5f59da5d`. The implementation was then
  applied on top of `69dec06872c74939d7af469d727964bb67611557`. The source
  fingerprint used by all six model-specific tuning files remains
  `dac502f0bbe79924b71d44177e5fc389aef8aacf7b8846d700232b529ca4b0b5`.
  These measurements precede the later rebase onto `9af2847` and should not
  be treated as performance measurements of the rebased code.
- Decode uses an explicit cache wrapper with capacity 64. PyTorch's manually
  captured graph uses static masked attention; MegaBake uses its own cache
  implementation. Graph traces recorded `cudaGraphLaunch` for both paths.
- Synchronized latency includes host call overhead and GPU completion;
  compilation is excluded. Original decode medians use 12 samples after five
  warmups; the repeat uses 24 samples with position tensors prepared before
  timing. Cache lengths grow through the measured calls.
- The eager and ordinary `torch.compile` decode variants in the benchmark use
  the explicit cache wrapper. Its eager attention reads cache lengths on the
  host, which causes per-layer synchronization. Those measurements are not
  native Hugging Face or serving-engine baselines.

## Decode results

Values are synchronized milliseconds per call. Each original PyTorch graph
and MegaBake graph pair was measured in the same run. The model-specific tuner
profiled feature tiles 64 and 128 with split limits 1 and 8; this was a limited
search and its choices were checked again on the whole model.

| Model | PyTorch graph, original / repeat | MegaBake graph, original / repeat | MegaBake matching native eager on five prompts |
| --- | ---: | ---: | ---: |
| Qwen2 0.5B | 1.604 / 1.595 | 1.565 / 1.557 | 5/5 |
| DeepSeek draft 0.6B | 1.594 / 1.591 | 1.540 / 1.558 | 4/5 |
| Qwen2.5 0.5B Instruct | 1.600 / 1.597 | 1.557 / 1.565 | 3/5 |
| Qwen3.5 0.8B | 1.807 / 1.768 | 4.528 / 4.520 | 4/5 |
| Gemma 2B | 1.942 / 1.999 | 3.173 / 3.164 | 4/5 |
| Gemma 3 270M | 1.209 / 1.271 | 2.386 / 2.388 | 4/5 |

The last column counts prompts for which all 16 greedy continuation tokens
matched native Hugging Face BF16 eager decoding. Matching tokens do not imply
identical logits. These five prompts are a small test set; timing differences
of a few percent do not establish a general speedup. The DeepSeek checkpoint is
a draft decoder measured alone, not end-to-end speculative decoding. Beam
DFlash2 was not measured because it requires its target model's components.

The PyTorch Qwen3.5 path used reference implementations because
`causal_conv1d` and `flash-linear-attention` were absent. Its custom eager
wrapper also diverged from native eager on two of the five generated sequences.

Earlier checks of the same implementation on SmolLM used a separate procedure:

| Model / configuration | PyTorch graph | MegaBake graph | 40-token greedy comparison to wrapper eager |
| --- | ---: | ---: | --- |
| SmolLM-135M, fused, no split-K | 1.870 | 1.523 | matched |
| SmolLM-1.7B, model-tuned split-K | 1.947 | 2.199 | first difference at token 29 |

Those SmolLM runs used the explicit wrapper and do not share the five-prompt
native comparison above. The 135M split-K experiment diverged at token 17, so
the no-split configuration was retained for that comparison.

## Numerical checks

For the six models, a repeated four-token input was teacher-forced over 128
positions. Tuned MegaBake matched the wrapper eager top token at 128/128 for
Qwen2, DeepSeek, Qwen3.5 and Gemma 2B; 127/128 for Qwen2.5; and 125/128 for
Gemma 3. The largest Gemma 2B relative L2 logit difference was 21.01%, despite
matching the top token at every position. Logit differences and greedy-token
divergences must be evaluated for each workload.

The five text prompts were `The capital of France is`, `Explain why the sky
appears blue.`, `Write a Python function to add two numbers.`, `2 + 2 =`, and
`Once upon a time in a distant village,`. They were processed incrementally,
followed by 16 greedy tokens. MegaBake first differed from native eager on the
last prompt at token 7 for Gemma 2B and token 3 for Gemma 3. It differed on two
Qwen2.5 prompts at tokens 9 and 11. All checked wrapper, PyTorch graph and
MegaBake logits were finite.

For Qwen2, one first prompt token produced a 102.64% relative L2 logit
difference between the PyTorch compiled graph and native eager. On the checked
first-token inputs, eager static masked attention matched native eager; the
large difference already appeared in ordinary `torch.compile` with either
dynamic or static attention, before graph capture. Its exact cause remains
unresolved. This is a result for this wrapper and environment, not a claim
about all uses of PyTorch compilation.

An FP32 arithmetic reference converted the same BF16-rounded weights to FP32
and disabled TF32. On 128 teacher-forced positions, Gemma 3 matched its top
token at 120 positions for BF16 eager, 126 for the PyTorch graph, and 119 for
MegaBake. This does not recover original FP32 checkpoint weights.

## 128-token prompt processing

Prefill measurements disable model caching and therefore do not measure
prompt-to-generation latency. These use the 32-token matrix tile and grouped
attention value loads. Values are synchronized milliseconds; the repeat uses
12 samples and three warmups on the same repeated-token prompt.

| Model | PyTorch graph, original / repeat | MegaBake graph, original / repeat | Repeat MegaBake top matches vs eager |
| --- | ---: | ---: | ---: |
| Qwen2 0.5B | 1.935 / 2.331 | 5.351 / 5.354 | 127/128 |
| Gemma 2B | 2.587 / 2.756 | 12.030 / 12.010 | 127/128 |
| Gemma 3 270M | 1.620 / 1.463 | 5.022 / 5.020 | 124/128 |

Two PyTorch graph samples in the Gemma 2B repeat were 70.7 and 90.6 ms; the
cause was not established. More samples and controlled repetitions are needed
to characterize latency variability. MegaBake Qwen3.5 prefill fails while
lowering `aten.cumsum` in the chunked gated-delta rule; its fallback also uses
triangular solves. The reported decode result does not demonstrate full
prompt-to-generation support for that model.

Instrumented profiles attributed approximately 4.0 ms of an 11.0 ms Gemma 2B
prefill run to attention and 4.7 ms to SM100 matrix phases. For Gemma 3,
attention accounted for about 2.0 ms of a 4.6 ms profiled run. These are
approximate attributions, not independent end-to-end timings.

## Regression checks

The earlier implementation passed the unit suite and targeted GPU checks on
the same B200: **268 passed**, with 290 warnings, in 124 seconds. The checks
cover the SM100 GEMM and attention bodies, split-K, epilogues, token tiles,
graph input updates, and related compiler behavior. They do not replace the
model-level numerical checks above or establish Hopper compatibility.
The five-prompt audit was also rerun from the rebased implementation for Gemma
3: the PyTorch and MegaBake graph medians were 1.270 and 2.378 ms respectively,
and MegaBake again first differed at token 3 on the last prompt.

```bash
PYTHONPATH="$PWD/src:$PWD" python -m pytest -q tests/unit \
  tests/gpu/test_blackwell_gemm.py tests/gpu/test_sm100_tcgen.py \
  tests/gpu/test_sm100_attention.py --disable-warnings
```

## Reproduction

Install the repository's development dependencies and use a CUDA 13 toolkit.
Run from the repository root. The example uses a locally cached model; gated
checkpoints require prior Hugging Face authorization. `MODEL_DIR` is a local
checkpoint directory, and `OUT` is a writable output directory. No tuning
files or raw benchmark artifacts are included in this branch.

```bash
export MODEL_DIR=/path/to/local/model
export OUT=/tmp/megabake-sm100-results
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MEGABAKE_SM100_TCGEN=1 MEGABAKE_SM100_TCGEN_TILE=auto
export MEGABAKE_SM100_EPILOGUE=1 MEGABAKE_SM100_SPLIT_K=8
export MEGABAKE_SM100_TCGEN_MAX_ROWS=128 MEGABAKE_SM100_TOKEN_TILE=8
export MEGABAKE_PREFILL_VALUE_BLOCK=0

python tools/tune_sm100_model.py --model "$MODEL_DIR" \
  --tiles 64 128 --split-limits 1 8 --calls 3 --profile-timeout 480 \
  --out "$OUT/tuning.json"
export MEGABAKE_SM100_TUNING_FILE="$OUT/tuning.json"

python benchmarks/full_model.py --model "$MODEL_DIR" --attention eager \
  --modes decode --variants eager inductor inductor_graphed megabake_graphed \
  --samples 12 --warmup 5 --capacity 64 --out "$OUT/decode"
python tools/check_sm100_accuracy.py "$MODEL_DIR" "$OUT/accuracy.json"
python tools/audit_sm100_models.py "$MODEL_DIR" "$OUT/audit.json"
```

The tuning file is tied to the GPU, software, selected settings, and source
fingerprint; a mismatch is rejected. The tuner uses instrumented phase timings,
so its selections must be checked on the complete model. The audit timer
prepares positions before timing and compares native incremental eager against
the explicit-cache paths on five prompts. It is a separate timer but reuses the
same graph wrappers, so it is not external verification.

For the first-token diagnostic on Qwen2 or Gemma 3, run:

```bash
python tools/probe_sm100_graph.py "$MODEL_DIR" "$OUT/first-token.json"
python tools/check_sm100_fp32.py "$MODEL_DIR" "$OUT/fp32.json"
```

For a separate prefill compilation, unset the decode tuning file and use:

```bash
unset MEGABAKE_SM100_TUNING_FILE
export MEGABAKE_SM100_SPLIT_K=1 MEGABAKE_SM100_TOKEN_TILE=32
export MEGABAKE_PREFILL_VALUE_BLOCK=8
python benchmarks/full_model.py --model "$MODEL_DIR" --attention eager \
  --modes prefill --variants eager inductor_cudagraphs megabake_graphed \
  --prompt-tokens 128 --check-prefill-correctness --samples 12 --warmup 3 \
  --out "$OUT/prefill"
```

Checkpoint revisions used: `Qwen/Qwen2-0.5B` at
`91d2aff3f957f99e4c74c962f2f408dcc88a18d8`,
`jukofyork/DeepSeek-V3-DRAFT-0.6B-v3.0` at
`b9878ad527387571519457f88a96a79c481a24ad`,
`Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775`,
`Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17`,
`google/gemma-2b` at `9cf48e52b224239de00d483ec8eb84fb8d0f3a3a`,
`google/gemma-3-270m` at `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`,
`HuggingFaceTB/SmolLM-135M` at `1d461723eec654e65efdc40cf49301c89c0c92f4`, and
`HuggingFaceTB/SmolLM-1.7B` at `d7449ff7241c863f3e8accc475155f0f97afa011`.
