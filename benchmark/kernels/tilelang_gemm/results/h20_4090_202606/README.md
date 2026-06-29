# TileLang FP8 GEMM H20 and RTX 4090 Results

This directory contains the full TileLang FP8 GEMM selected configs and
benchmark artifacts used for PR benchmark validation on H20 and RTX 4090.

All artifacts are committed as text, JSON, CSV, TSV, or log files so they can
be inspected directly in the GitHub web UI.

## Run Matrix

- Shapes: 25 `(N, K)` shapes
- M values: `1 2 3 4 5 6 7 8 12 14 16 24 32 48 64 96 128 256 512 1024 2048 4096 8192`
- Rows per GPU: 575
- Benchmark backend: CUDA graph
- Benchmark repetitions: 100
- SGLang commit benchmarked: `1a6b70949961ce6e79078b35daf2869c18381321`

## Summary

| GPU | Comparison | Rows | Allclose failures | TileLang faster | TileLang slower | Median speedup | Worst TileLang / baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 4090 | Triton | 575 | 0 | 489 | 86 | 2.95x | 1.72x |
| H20 | Triton | 575 | 1 | 548 | 27 | 1.78x | 1.16x |
| H20 | DeepGEMM | 575 | 1 | 499 | 76 | 1.27x | 1.44x |

DeepGEMM is not included in the RTX 4090 run because DeepGEMM does not support
SM89.

## Configs

| GPU | Config |
| --- | --- |
| H20 | [tilelang_selected_configs.json](h20/tilelang_selected_configs.json) |
| H20 | [tilelang_benchmark_selected_configs.json](h20/tilelang_benchmark_selected_configs.json) |
| RTX 4090 | [tilelang_selected_configs.json](rtx4090/tilelang_selected_configs.json) |
| RTX 4090 | [tilelang_benchmark_selected_configs.json](rtx4090/tilelang_benchmark_selected_configs.json) |

The H20 `tilelang_selected_configs.json` was verified against the copy in
`H20_19` container `tilelang_gemm_0613`:

```text
a105dc46acbb921ebb36497cac6a36f35c62921be80a7b8f10689c1db87d816d
```

## Full Results

| GPU | Full benchmark CSV | Summary JSON | Benchmark log | Tune log |
| --- | --- | --- | --- | --- |
| H20 | [tilelang_vs_tuned_triton_deepgemm.csv](h20/tilelang_vs_tuned_triton_deepgemm.csv) | [benchmark_summary.json](h20/benchmark_summary.json) | [benchmark.log](h20/benchmark.log) | [tune.log](h20/tune.log) |
| RTX 4090 | [tilelang_vs_tuned_triton_deepgemm.csv](rtx4090/tilelang_vs_tuned_triton_deepgemm.csv) | [benchmark_summary.json](rtx4090/benchmark_summary.json) | [benchmark.log](rtx4090/benchmark.log) | N/A |

Additional diagnostic files:

| GPU | Files |
| --- | --- |
| H20 | [tilelang_slower_than_triton.csv](h20/tilelang_slower_than_triton.csv), [top20_tilelang_slower_than_triton.csv](h20/top20_tilelang_slower_than_triton.csv), [tilelang_slower_than_deepgemm.csv](h20/tilelang_slower_than_deepgemm.csv), [top20_tilelang_slower_than_deepgemm.csv](h20/top20_tilelang_slower_than_deepgemm.csv), [allclose_failures.csv](h20/allclose_failures.csv), [deepgemm_allclose_failures.csv](h20/deepgemm_allclose_failures.csv) |
| RTX 4090 | [tilelang_slower_than_triton.csv](rtx4090/tilelang_slower_than_triton.csv), [top20_tilelang_slower_than_triton.csv](rtx4090/top20_tilelang_slower_than_triton.csv) |

Run metadata and environment captures:

| GPU | Metadata | Environment | Shapes | M values |
| --- | --- | --- | --- | --- |
| H20 | [run_metadata.txt](h20/run_metadata.txt) | [environment.txt](h20/environment.txt) | [shapes.tsv](h20/shapes.tsv) | [m_values.txt](h20/m_values.txt) |
| RTX 4090 | [run_metadata.txt](rtx4090/run_metadata.txt) | [environment.txt](rtx4090/environment.txt) | [shapes.tsv](rtx4090/shapes.tsv) | [m_values.txt](rtx4090/m_values.txt) |

## Original Artifact Locations

- H20: `/goosefsx/x-c60-2k48ac4x-proxy/data/linjunxian/InferScripts/sglang/myscripts/startup/tilelang_fp8_gemm_repro_outputs/20260615_093452/`
- RTX 4090: `/workspace/model/experiments/tilelang_fp8_gemm_repro/tilelang_fp8_gemm_repro_outputs/cu130_4090_10_no_deepgemm_20260629_061522/`
