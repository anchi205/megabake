# SM100 GEMM on B200

On SM100, MegaBake selects a 256-thread GEMM adapted from the CUTLASS 4.8.0
persistent kernel for compatible BF16 phase-table workloads. The adaptation
uses dynamic descriptors, split-K and supported fused epilogues. Unsupported
shapes and launch policies use other available bodies. Other Blackwell targets
retain their existing selection.

This remains an experimental path: sampled generated sequences can differ from
PyTorch eager and from the previous MegaBake paths. Only one B200 was measured;
SM90 hardware was not retested after the shared compiler and attention changes.

## Decode measurements

One B200, BF16, batch one, cache capacity 192. Values are synchronized host
medians in milliseconds over 24 CUDA graph replays after five warmups;
compilation is excluded. The compiled PyTorch graph uses static masked attention
and MegaBake uses its own cache implementation, so these paths do different
work. The columns were measured in separate runs; small differences may be
noise. Qwen3.5 used PyTorch reference implementations without its optional
acceleration packages.

| Model | PyTorch compile graph | Previous default graph | Previous 128 graph | Previous 256 graph | CUTLASS fork graph |
| --- | ---: | ---: | ---: | ---: | ---: |
| SmolLM 135M | 1.9971 | 3.9836 | 1.2768 | 1.2656 | 1.2500 |
| Qwen2 0.5B | 1.7129 | 9.1141 | 1.5901 | 1.5678 | 1.5727 |
| Qwen2.5 0.5B Instruct | 1.7094 | 9.1097 | 1.5767 | 1.5670 | 1.5684 |
| DeepSeek draft 0.6B | 1.6960 | 9.0184 | 1.5822 | 1.5544 | 1.5432 |
| Gemma 3 270M | 1.3379 | 5.3468 | 2.3223 | 2.3383 | 2.3349 |
| SmolLM 1.7B | 2.0391 | 16.1409 | 1.9767 | 1.9538 | 1.9776 |
| Gemma 2B | 2.0157 | 21.3165 | 2.7383 | 2.7443 | 2.6499 |
| Qwen3.5 0.8B | 1.8388 | 11.5070 | 4.3415 | 4.6501 | 4.6623 |

The previous default used warp-MMA with no SM100 flags. The 128, 256 and fork
runs used automatic feature tiles, up to eight split-K partitions, supported
fusion and no tuning file. The 256 and fork runs used one accumulator stage.
The fork measurements used the code subsequently committed as `7105b8e` with
explicit flags. The new flag-free default uses those settings but **has not
been benchmarked after the default-selection change**.

## Correctness observed

In a separate teacher-forced audit, all checked fork logits were finite, and
fork graph and non-graph logits and cache states matched at the checked
positions. The table compares fork logits with wrapper eager and the earlier
128-thread path over 128 positions. The last column compares their 40-token
greedy continuations. These samples do not establish general equivalence.

| Model | Fork top tokens matching eager | Fork relative L2 vs eager, median / max | Full logit hashes matching 128 | 40 tokens match 128 |
| --- | ---: | ---: | ---: | --- |
| SmolLM 135M | 122/128 | 1.39% / 7.06% | 2/128 | yes |
| Qwen2 0.5B | 127/128 | 1.37% / 3.17% | 7/128 | yes |
| Qwen2.5 0.5B Instruct | 127/128 | 2.41% / 4.62% | 24/128 | yes |
| DeepSeek draft 0.6B | 127/128 | 1.82% / 4.58% | 16/128 | yes |
| Gemma 3 270M | 123/128 | 2.99% / 8.26% | 2/128 | no |
| SmolLM 1.7B | 125/128 | 0.77% / 2.03% | 17/128 | no |
| Gemma 2B | 127/128 | 1.48% / 16.09% | 3/128 | yes |
| Qwen3.5 0.8B | 128/128 | 1.53% / 2.54% | 3/128 | yes |

The wider launch also changes other phases' thread layout; the cause of the
model output differences has not been established. A five-prompt audit found
that the fork and previous 256-thread body generated the same 16-token
continuations for all eight models, without proving full-logit equivalence.
The B200 unit and GPU suites before the default-selection change reported
375 passed, 37 skipped; skips included Hopper-only cases.

## Reproduce

Install the repository environment and prepare local checkpoint directories
under `$MODEL_ROOT` using the revisions below. Gemma models require Hugging
Face access. The runner loads checkpoints offline, clears inherited
`MEGABAKE_` settings and writes logs, JSON and a CSV summary to a new or empty
output directory.

```bash
python tools/benchmark_sm100_graphs.py --models-root "$MODEL_ROOT" \
  --out /tmp/megabake-default

# Earlier 128-thread persistent body, with the same shape heuristic
python tools/benchmark_sm100_graphs.py --models-root "$MODEL_ROOT" \
  --implementation existing --threads 128 --out /tmp/megabake-128
```

Use `--models smollm-135m` to run a subset. `--implementation existing
--threads 256` selects the previous 256-thread body; `--implementation
cutlass-fork --threads 256` explicitly selects the fork. `--stages 2` selects
its optional two-stage variant. `--implementation warp-mma` selects the
previous fallback.

For application use, `MEGABAKE_SM100_TCGEN=0` restores the warp-MMA fallback;
`MEGABAKE_SM100_PERSISTENT_WIDTH=128` selects the previous 128-thread
persistent variant. `MEGABAKE_SM100_CUTLASS_FORK=0` selects the previous
256-thread body. Tuning files are tied to the device, settings and source
fingerprint and must be regenerated after code changes.

| Directory | Checkpoint | Revision |
| --- | --- | --- |
| `smollm-135m` | `HuggingFaceTB/SmolLM-135M` | `1d461723eec654e65efdc40cf49301c89c0c92f4` |
| `qwen2-0.5b` | `Qwen/Qwen2-0.5B` | `91d2aff3f957f99e4c74c962f2f408dcc88a18d8` |
| `qwen2.5-0.5b-instruct` | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` |
| `deepseek-draft` | `jukofyork/DeepSeek-V3-DRAFT-0.6B-v3.0` | `b9878ad527387571519457f88a96a79c481a24ad` |
| `gemma-3-270m` | `google/gemma-3-270m` | `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1` |
| `smollm-1.7b` | `HuggingFaceTB/SmolLM-1.7B` | `d7449ff7241c863f3e8accc475155f0f97afa011` |
| `gemma-2b` | `google/gemma-2b` | `9cf48e52b224239de00d483ec8eb84fb8d0f3a3a` |
| `qwen3.5-0.8b` | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` |
