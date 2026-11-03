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

## Body provider refactor validation on 2026-11-03

The GPU-specific table GEMM definitions, launch capabilities, descriptor policy,
and target mappings are registered under `src/megabake/bodies`. Shared code
uses those contracts to select, schedule, generate, and bind bodies. Event
scheduling preserves the caller's target options and asks providers for bodies
compatible with that scheduling mode. Tuning fingerprints include the new
body contracts and implementation files, so profiles from earlier versions
are rejected rather than reused.

On one B200, the full `tests/unit` and `tests/gpu` suites reported **286
passed, 37 skipped**. The skipped cases include Hopper-only GPU tests. A
separate run of the expanded provider and Hopper source-generation checks
and vendored-source checks reported **14 passed** with CUDA hidden. A CPU-only
import without CUTLASS installed also succeeded. The Hopper table body has
source-generation coverage, but it has not been executed on Hopper hardware
after this refactor.

All eight models were retuned independently with feature tiles 64 and 128 and
split limits 1 and 8; all candidates completed without a reported failure.
The chosen shape configurations were the same as in the preceding run. The
new implementation fingerprint is
`c62f6a5fe1331b4f0292d9a1257806add2682e88078c44e4fc140984ce16e7d7`.
The same BF16, batch-one setup and timing methodology described in the rebase
section were used. The following synchronized medians include host call
overhead, exclude compilation, and are in milliseconds. Decode uses 24 samples
after five warmups; prefill uses 12 after three warmups. The wrapper eager and
ordinary compiled decode paths synchronize on host cache lengths, and the
graphed PyTorch decode path uses static masked attention.

| Decode model | Wrapper eager | `torch.compile` | Compiled graph | MegaBake | MegaBake graph |
| --- | ---: | ---: | ---: | ---: | ---: |
| SmolLM 135M | 22.820 | 8.910 | 2.600 | 1.878 | 1.402 |
| Qwen2 0.5B | 14.480 | 8.059 | 2.245 | 2.115 | 1.582 |
| Qwen2.5 0.5B Instruct | 14.116 | 7.839 | 2.240 | 2.102 | 1.584 |
| DeepSeek draft 0.6B | 14.915 | 8.151 | 2.602 | 2.084 | 1.549 |
| Gemma 3 270M | 16.166 | 6.494 | 1.728 | 2.952 | 2.426 |
| SmolLM 1.7B | 17.376 | 7.254 | 2.502 | 2.574 | 2.218 |
| Gemma 2B | 11.661 | 6.160 | 2.422 | 3.726 | 3.229 |
| Qwen3.5 0.8B | 17.210 | 5.611 | 2.283 | 5.026 | 4.449 |

| 128-token prefill model | Eager | `torch.compile` | Compile reduce-overhead | MegaBake | MegaBake graph |
| --- | ---: | ---: | ---: | ---: | ---: |
| SmolLM 135M | 13.613 | 5.039 | 1.970 | 4.011 | 3.514 |
| Qwen2 0.5B | 11.830 | 4.415 | 1.949 | 5.785 | 5.288 |
| Qwen2.5 0.5B Instruct | 12.331 | 4.744 | 2.169 | 5.794 | 5.285 |
| DeepSeek draft 0.6B | 11.783 | 4.429 | 2.163 | 5.767 | 5.254 |
| Gemma 3 270M | 13.736 | 2.833 | 1.488 | 5.436 | 4.968 |
| SmolLM 1.7B | 10.588 | 3.535 | 2.427 | 9.674 | 9.235 |
| Gemma 2B | 9.416 | 3.081 | 2.403 | 12.387 | 11.933 |
| Qwen3.5 0.8B | 29.927 | 8.131 | 4.687 | failed | failed |

Qwen3.5 prefill still fails in MegaBake semantic lowering on `aten.cumsum` and
`aten.linalg_solve_triangular`. Its PyTorch run used reference implementations
without the optional `causal_conv1d` and `fla` packages.

The independent timer below pre-created device positions and used 24 samples
after five warmups; it is a separate measurement from the preceding table.

| Decode model | Native eager | Wrapper eager | Compiled graph | MegaBake graph |
| --- | ---: | ---: | ---: | ---: |
| SmolLM 135M | 13.954 | 22.105 | 1.920 | 1.374 |
| Qwen2 0.5B | 12.117 | 14.168 | 1.677 | 1.551 |
| Qwen2.5 0.5B Instruct | 12.175 | 14.117 | 1.677 | 1.545 |
| DeepSeek draft 0.6B | 12.132 | 13.991 | 1.664 | 1.538 |
| Gemma 3 270M | 15.155 | 16.915 | 1.304 | 2.437 |
| SmolLM 1.7B | 11.350 | 18.193 | 2.013 | 2.189 |
| Gemma 2B | 10.064 | 11.973 | 1.999 | 3.204 |
| Qwen3.5 0.8B | 20.559 | 18.223 | 1.818 | 4.493 |

The correctness check compares 128 teacher-forced decode positions against
wrapper eager, a 40-token greedy run, and five natural prompts against native
eager. The last column counts selected SM100 TMA phases and those with fused
epilogues in the saved decode launch metadata. Matching top tokens does not
mean that all logits match.

| Model | Teacher top matches | Median / max relative L2 | First greedy difference | Matching prompts | TMA / fused phases |
| --- | ---: | ---: | ---: | ---: | ---: |
| SmolLM 135M | 124/128 | 1.36% / 9.74% | 17 | 4/5 | 211 / 60 |
| Qwen2 0.5B | 128/128 | 1.24% / 2.18% | 11 | 5/5 | 169 / 120 |
| Qwen2.5 0.5B Instruct | 127/128 | 2.38% / 6.29% | none in 40 | 3/5 | 169 / 120 |
| DeepSeek draft 0.6B | 128/128 | 1.58% / 3.10% | none in 40 | 4/5 | 169 / 120 |
| Gemma 3 270M | 125/128 | 2.96% / 10.10% | none in 40 | 4/5 | 127 / 0 |
| SmolLM 1.7B | 126/128 | 0.80% / 2.18% | 29 | 4/5 | 169 / 48 |
| Gemma 2B | 128/128 | 1.45% / 21.01% | none in 40 | 4/5 | 127 / 36 |
| Qwen3.5 0.8B | 128/128 | 1.48% / 2.93% | none in 40 | 4/5 | 151 / 48 |

For each model, the recorded per-position relative errors, top mismatch
positions, and greedy token sequences were identical to the pre-refactor run.
All checked logits were finite; MegaBake's graph and non-graph KV states
matched. This comparison does not establish bitwise identity of all logits.
These are phases inside a single generated kernel, not separate GPU kernel
launches. The Gemma 3 plan also contains 646 generic and 18 attention phases;
the Qwen3.5 plan contains 1,308 generic, 36 warp-MMA, and 6 attention phases.

### CUTLASS origin and fusion audit

The SM90 body vendors an identified CUTLASS persistent GEMM and keeps its
unmodified source for comparison. The current vendor diff reports 972
insertions and 505 deletions, including split-K, fused epilogues, paired GEMMs,
and weight streaming. The SM100 body is a substantially adapted
experimental kernel based on the CUTLASS Blackwell tutorial; it is not a port
of NVIDIA's optimized persistent GEMM. Its main loop structurally matches
[CUTLASS 4.8.0 `fp16_gemm_0.py`](https://github.com/NVIDIA/cutlass/blob/v4.8.0/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py)
(upstream SHA-256 `a15471df9a2a9c80a6a772416210ea256a58b142aa4b6358ca01851f6f02362a`).
Compared with that file, the current SM100 source has 285 insertions and 224
deletions in a `git diff --no-index` comparison. A snapshot is vendored under
`bodies/cuda/sm100/cutlass_upstream` with two leading lint directives; after
those directives, its bytes match the upstream hash. `tools/vendor_cutlass.py`
checks both and shows the diff, including those two directives.

The current SM100 body uses BF16 operands, FP32 accumulation, TMA loads and
`tcgen05` tensor-core operations, with four A/B stages and one accumulator
stage. It uses 128 threads and the same warp issues TMA loads and MMA. The
outer table body loops over assigned tiles, but allocates and frees tensor
memory inside each tile. MegaBake adds dynamic descriptors, small-row tiles,
split-K partial reduction, and generated epilogues.
The TMA path is opt-in, currently requires SM100, a cooperative phase-table
launch, BF16, specific contiguous operand layouts, and rows and dimensions
accepted by its tile rules. Its TMA body does not implement paired GEMMs;
unsupported operations use other bodies where available.

For the refactored SmolLM 135M decode launch, its saved launch metadata lists
211 SM100 TMA GEMM phases out of 338 total phases, including 60 GEMM phases
with fused epilogues and 30 with split-K 8. The generated epilogues perform
residual addition and preserve the original BF16 rounding point. This is a
specific measured graph, not evidence that every model operation is fused.
The GPU test suite also compares a fused residual GEMM against its unfused
equivalent for split-K 1 and 4.

NVIDIA's [CUTLASS 4.8.0 persistent SM100 example](https://github.com/NVIDIA/cutlass/blob/v4.8.0/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent.py)
has upstream SHA-256 `4f3d5fa8c6032c2190933cb43c71d358bb200225fe7c0ec0b54014a9a6235362`.
It uses separate TMA and MMA warps, four epilogue warps and two accumulator
stages. Its documented MMA N tile is at least 32, while the current decode
body uses token tiles as small as 8. It is a standalone kernel with its own
tile scheduler, not a drop-in device body for MegaBake's phase table. It uses
192 threads, while the current generic and attention bodies use 128; the
planner chooses a common block size and the current widening rule requires
an integer multiple. Integration must reconcile participation, barriers, and
resources as well as the tile shapes. A closer SM90-style integration should
vendor a pinned copy, adapt the persistent
mainloop to the enclosing cooperative launch and its barrier and scratch
contracts, retain supported fused epilogues and split-K, and measure small-row
decode and wider prefill separately. Any speedup remains unverified.

The B200 multi-tile and fused-residual checks passed **6 tests**, including
split-K 8, input rebinding, and repeated graph replay. PTX retained from those
tests contains `tcgen05.mma` and TMA tensor loads; disassembly of the compiled
binaries contains `UTCHMMA` and `UTMALDG.2D` instructions.

As a separate experiment, the unmodified CUTLASS 4.8.0 persistent example and
MegaBake were run on the same randomly initialized BF16 operands. The table
reports median profiler-recorded GPU kernel duration in microseconds across
32 calls, with CUDA graphs for the first two columns and eager PyTorch in the
third. Both compared GEMMs used split-K 1 and no fused epilogue; all four
checked outputs (graph and non-graph) matched PyTorch exactly for these
inputs. NVIDIA's kernel used 32-token tiles and one CTA group; MegaBake used
the token and feature tiles shown below.

| Rows × features × K | CUTLASS tile | MegaBake token × feature tile | CUTLASS graph | MegaBake graph | PyTorch eager |
| --- | --- | --- | ---: | ---: | ---: |
| 1 × 576 × 1536 | 64 × 32 | 8 × 64 | 4.160 | 12.704 | 4.688 |
| 8 × 576 × 1536 | 64 × 32 | 8 × 64 | 4.192 | 12.720 | 4.384 |
| 32 × 576 × 1536 | 64 × 32 | 32 × 64 | 4.224 | 12.928 | 4.208 |
| 128 × 576 × 1536 | 64 × 32 | 32 × 64 | 4.287 | 13.008 | 4.319 |
| 1 × 1536 × 576 | 128 × 32 | 8 × 128 | 3.681 | 11.168 | 3.392 |
| 32 × 1536 × 576 | 128 × 32 | 32 × 128 | 3.776 | 11.392 | 3.072 |

This measures a standalone CUTLASS GEMM against a single-operation MegaBake
launch, which includes phase-table and dynamic-descriptor work. It does not
isolate the device body, measure a fused epilogue, or establish end-to-end
performance after integrating the persistent kernel. Both implementations
launched nine CTAs for the first shape; their block sizes were 192 and 128,
respectively. The profiler observed one kernel per call for every table row.

For a separate GEMM-plus-residual experiment, the unfused graph exposed the
intermediate product as an additional output, forcing a materialized GEMM
followed by a generic elementwise phase. The fused graph used one SM100 GEMM
with an epilogue. Both used the same BF16 inputs, split-K 1 and feature tile
64. Kernel times below are medians from 32 profiled graph replays. The extra
output changes the graph contract, so this comparison does not attribute the
entire timing difference solely to fusion.

| Rows × features × K | Fused kernel µs | Materialized kernel µs | Fused elements different from eager | Max absolute difference |
| --- | ---: | ---: | ---: | ---: |
| 1 × 576 × 1536 | 13.376 | 15.104 | 178 / 576 | 0.001953125 |
| 32 × 576 × 1536 | 16.992 | 16.208 | 5,547 / 18,432 | 0.00390625 |
| 128 × 576 × 1536 | 17.024 | 16.256 | 22,398 / 73,728 | 0.00390625 |

The materialized outputs matched eager exactly on these inputs. The fused
one-row output differed from eager, `torch.addmm`, and `torch.compile` by at
most 0.001953125; it matched an FP32 accumulation plus residual followed by a
single BF16 rounding. The materialized result matched a reference that rounds
the GEMM to BF16 before the residual addition. For 32 and 128 rows the fused
outputs matched `torch.addmm` and `torch.compile` on these inputs but differed
from eager. The FP32 references had TF32 disabled. These cases demonstrate a
numerical limitation, not universal bounds or bitwise correctness.

The new raw model results, traces, generated code, PTX, binaries, test logs,
and comparison scripts were copied to a local archive with SHA-256
`b20865d579b4c8b9637d6d39c4a8f1c7f6cc4e445bfddbd793a22f2dbd035941`.
The archive is not included in this PR.

The measurements in the rebase section below were made before this body
provider refactor; they are retained as historical results.

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
