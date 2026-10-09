# Efficient Transformer Systems Report

Generated: `2026-10-09T18:04:28.619256+00:00`

## Scope

Measured NumPy CPU matrix; declared bytes are managed state/workspace, not process peak memory.

## Comparable workload results

| Workload | Candidate vs baseline | Median latency | Throughput | Persistent state | Workspace | Numerical error |
|---|---|---:|---:|---:|---:|---:|
| prefill_attention_s512 | `blockwise_64` vs `dense` | 7.872 ms vs 6.625 ms (1.19x) | 65,037 vs 77,287 query_tokens/s | 0 B vs 0 B | 64 KiB vs 4 MiB | 2.38419e-07 |
| single_token_decode_s4096 | `grouped_query_2kv` vs `multi_head_8kv` | 0.944 ms vs 0.869 ms (1.09x) | 1,059 vs 1,150 generated_tokens/s | 4 MiB vs 16 MiB | 128 KiB vs 128 KiB | 0 |
| linear_1024x1024_t32 | `int8_per_channel_reference` vs `fp32` | 1.892 ms vs 1.427 ms (1.33x) | 16,909 vs 22,417 input_tokens/s | 1.004 MiB vs 4 MiB | 4 MiB vs 0 B | 0.00766127 |

## Claim boundaries

- Results describe one fixed-seed NumPy CPU microbenchmark run.
- Declared bytes cover managed persistent state and analytical workspace, not process peak memory.
- Throughput is comparable only within workloads sharing the same useful-work unit.
- Numerical error is not a language-model quality, accuracy, or perplexity measurement.
- CPU reference timings do not predict fused accelerator-kernel performance.

## Reproduction

```bash
python -m pip install -e ".[dev]"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/build_systems_report.py
python -m pytest -q
```

The Markdown report and JSON evidence are generated from the same in-memory snapshot.
