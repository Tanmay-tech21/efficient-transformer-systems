# Efficient Transformer Systems

A seven-day systems laboratory for understanding the memory and compute trade-offs
behind transformer inference. The project begins with transparent analytical
accounting, then progresses towards measured profiling, attention and KV-cache
experiments, quantisation, and reproducible efficiency comparisons.

## Day 1: make architectural costs explicit

The first layer provides validated parameter and memory accounting for a dense,
decoder-only, pre-normalised transformer. Its assumptions are declared rather
than hidden inside a model implementation:

- multi-head or grouped-query attention;
- a two-projection GELU feed-forward block;
- two normalisations per layer plus a final normalisation;
- optional biases and learned positional embeddings; and
- tied or untied token/output embeddings.

The memory estimator reports two quantities with different scaling behaviour.
The KV cache covers all layers; the attention-score estimate is per layer:

\[
\text{KV bytes} = B L S \cdot 2 H_{kv} D_h \cdot \text{bytes per element}
\]

\[
\text{attention-score bytes} = B H S^2 \cdot \text{bytes per element}
\]

KV-cache storage grows linearly with sequence length. A conventional materialised
attention-score matrix grows quadratically; fused attention kernels need not store
that full matrix. These estimates describe storage implied by explicit assumptions,
not measured peak memory or latency on a particular device.

## Reproducible reference profile

The included script profiles a decoder with width 4096, 32 layers, 32 query
heads, 8 key/value heads, feed-forward width 11008, and tied embeddings across
context lengths from 128 to 8192 tokens:

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -e ".[dev]"
    python scripts/profile_reference_model.py
    python -m pytest

The JSON output is labelled as analytical. No hardware timing, throughput,
kernel behaviour, allocator overhead, or measured peak memory is implied.

## Day 2: measure before optimising

The timing layer separates warm-up from recorded repetitions, preserves every
sample, and reports median and interquartile range rather than relying on one
favourable run. A caller may provide a synchronisation callback, which is
required when an accelerator launches work asynchronously.

The first empirical workload is a readable NumPy implementation of dense scaled
dot-product attention. It is a reference and profiling fixture, not a claim to
match a fused production kernel. Reproduce the CPU sweep with:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      python scripts/benchmark_attention.py

One measured run in the development environment used Python 3.12.14, NumPy
2.3.5, Linux x86_64, float32 inputs, four heads, head width 64, three warm-up
iterations, and ten recorded repetitions:

| Sequence length | Score elements | Median latency | IQR |
|---:|---:|---:|---:|
| 64 | 16,384 | 0.102 ms | 0.008 ms |
| 128 | 65,536 | 0.462 ms | 0.032 ms |
| 256 | 262,144 | 1.251 ms | 0.060 ms |
| 512 | 1,048,576 | 5.673 ms | 0.330 ms |

These measurements characterise this particular runtime. They do not predict
PyTorch, MPS, CUDA, production batch sizes, or end-to-end model latency. The raw
samples are emitted as JSON so later runs can be compared without discarding
variation.

## Day 3: trade score storage for tiled computation

The second attention implementation computes the same softmax result without
materialising the complete score matrix. It visits query and key tiles, then
maintains three online statistics for each query row:

- the largest score observed so far;
- the rescaled softmax denominator; and
- the rescaled value-weighted numerator.

When a new key tile changes the running maximum, the previous denominator and
numerator are rescaled before the tile contribution is added. This is the
numerical mechanism that makes blockwise exact attention possible; simply
applying softmax independently inside each tile would be incorrect.

Run the paired comparison with:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      python scripts/compare_attention_algorithms.py

One development-environment run used float32 inputs, four heads, head width 64,
`64x64` score tiles, two warm-up iterations and five recorded repetitions:

| Sequence | Dense median | Blockwise median | Blockwise / dense | Score-tile reduction |
|---:|---:|---:|---:|---:|
| 64 | 0.100 ms | 0.121 ms | 1.21x | 1x |
| 128 | 0.351 ms | 0.460 ms | 1.31x | 4x |
| 256 | 1.715 ms | 1.751 ms | 1.02x | 16x |
| 512 | 4.525 ms | 7.258 ms | 1.60x | 64x |

The maximum absolute difference across the sweep was `5.37e-7`. The workspace
column is an analytical count of score elements in the largest tile, not measured
process memory. The latency penalty reflects this readable NumPy prototype and
must not be interpreted as the performance of FlashAttention or another fused
kernel. Its purpose is to expose the online-softmax invariant and the memory-time
trade-off before moving to framework-specific implementations.

## Day 4: separate KV storage savings from decode work

The cache implementation preallocates key and value tensors, appends complete
chunks only after validation, and exposes both logical and allocated storage.
Single-token grouped-query attention reshapes query heads into groups and uses
the corresponding KV head directly; it does not materialise repeated KV tensors.

This distinction matters. Reducing eight KV heads to two cuts persistent cache
storage by four, but all eight query heads still score the entire cached prefix.
The experiment therefore measures storage and latency separately:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      python scripts/benchmark_kv_cache.py

One development-environment run used Python 3.12.14, NumPy 2.3.5, Linux x86_64,
float32 inputs, eight query heads, head width 64, three warm-up iterations, and
ten recorded repetitions. Each cache was filled to its declared capacity.

| Context | MHA cache | GQA cache | Reduction | MHA median | GQA median | GQA / MHA |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.50 MiB | 0.125 MiB | 4x | 0.071 ms | 0.065 ms | 0.92x |
| 512 | 2.00 MiB | 0.50 MiB | 4x | 0.145 ms | 0.142 ms | 0.98x |
| 2,048 | 8.00 MiB | 2.00 MiB | 4x | 0.422 ms | 0.462 ms | 1.10x |
| 4,096 | 16.00 MiB | 4.00 MiB | 4x | 0.800 ms | 0.868 ms | 1.09x |

The storage reduction is exact for this head configuration. Latency was almost
unchanged and GQA became modestly slower at the two longest contexts. Both paths
computed the same number of attention-score elements, and this readable NumPy
prototype has no fused GQA kernel. These timings characterise one CPU run; they
do not establish accelerator or end-to-end generation performance.

## Day 5: quantify compression, error, and execution separately

The weight-only reference uses symmetric INT8 values with either one scale for
the complete matrix or one scale per output channel. It stores the quantised
weights persistently, then dequantises to float32 inside each linear-layer call.
That transparent path isolates numerical error, but it is deliberately not an
optimised integer matrix-multiplication kernel.

Run the comparison with:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      python scripts/benchmark_weight_quantization.py

One measured run used a `1024x1024` weight matrix whose output-channel
magnitudes varied geometrically from `0.1` to `10.0`. Inputs and weights were
generated with a fixed seed. Three warm-up iterations preceded ten recorded
repetitions.

| Representation | Persistent storage | Compression | Weight relative RMSE |
|---|---:|---:|---:|
| FP32 | 4.000 MiB | 1.00x | 0 |
| INT8, per tensor | 1.000 MiB | 4.00x | 2.788% |
| INT8, per output channel | 1.004 MiB | 3.98x | 0.782% |

| Tokens | FP32 median | Per-tensor median | Per-channel median | Per-channel output RMSE |
|---:|---:|---:|---:|---:|
| 1 | 0.054 ms | 0.263 ms | 0.410 ms | 0.762% |
| 32 | 1.531 ms | 1.800 ms | 1.957 ms | 0.784% |

Per-channel metadata cost only 4 KiB more than per-tensor scaling while reducing
weight reconstruction error by roughly 3.6x on this deliberately heterogeneous
fixture. The reference remained slower than FP32 because every call materialised
a dequantised float32 matrix. Storage compression, numerical fidelity, and
kernel speed are therefore reported as distinct outcomes. The error values are
not language-model accuracy or perplexity measurements.

## Day 6: compare efficiency with explicit contracts

The efficiency matrix puts the three reference workloads behind one reporting
contract: median latency, IQR, useful-work throughput, persistent state,
declared workspace, and numerical error. Comparisons are permitted only within
the same workload and useful-work definition. This prevents, for example,
prefill query-token throughput from being compared directly with generated-token
decode throughput.

Run the fixed-seed CPU matrix with:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      python scripts/build_efficiency_matrix.py

One development-environment run used Python 3.12.14, NumPy 2.3.5, Linux x86_64,
three warm-up iterations, and ten recorded repetitions.

| Workload and candidate | Baseline | Candidate median | Baseline median | Candidate throughput | Baseline throughput | Primary state trade-off | Error |
|---|---|---:|---:|---:|---:|---|---:|
| Prefill, blockwise-64 | Dense | 12.943 ms | 8.795 ms | 39,557 query tokens/s | 58,217 query tokens/s | 64 KiB vs 4 MiB workspace | 2.38e-7 max abs. |
| Decode, GQA with 2 KV heads | MHA with 8 KV heads | 1.292 ms | 1.737 ms | 774 generated tokens/s | 576 generated tokens/s | 4 MiB vs 16 MiB persistent cache | 0 max abs. |
| Linear, per-channel INT8 reference | FP32 | 4.945 ms | 2.101 ms | 6,471 input tokens/s | 15,231 input tokens/s | 1.004 MiB vs 4 MiB persistent weights; 4 MiB vs 0 declared workspace | 0.766% relative RMSE |

The blockwise path reduced analytical attention workspace by 64x but was 47.2%
slower in this readable CPU implementation. The equivalent-cache GQA fixture
reduced persistent KV state by 4x and was 25.7% faster. The INT8 reference
reduced persistent weight storage by about 3.98x, yet was 2.35x slower because
it created a float32 dequantised matrix on every call.

The earlier Day 4 fixture used independently generated MHA and GQA caches and
measured GQA 9% slower at context 4,096. Today's fixture repeats two KV heads
to make the eight-head MHA baseline numerically equivalent, and it measured GQA
faster with substantial timing variation. The architectural cache reduction is
exact; neither isolated CPU timing should be treated as a universal kernel
ranking. The byte counts above are declared persistent arrays and analytical
workspace, excluding output arrays, allocator overhead, and process peak memory.

## Day 7: publish one report from one evidence snapshot

The final reporting layer executes the complete efficiency matrix once, validates
its schema, then renders a human-readable systems report and machine-readable JSON
from the same in-memory snapshot. It refuses incomplete records, non-positive
timings, duplicate variants, and comparisons that do not point to measured
workloads. This prevents a polished summary from quietly drifting away from the
evidence it claims to describe.

Generate and validate the portfolio artifacts with:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      python scripts/build_systems_report.py
    python -m pytest -q

The generated outputs are:

- `artifacts/SYSTEMS_REPORT.md`, for human review;
- `artifacts/efficiency_snapshot.json`, containing the configuration,
  environment, raw timing samples, derived comparisons, and claim boundaries.

The report remains deliberately narrow. It describes transparent NumPy CPU
references rather than fused production kernels, records managed arrays rather
than process peak memory, and does not treat numerical agreement as language-model
quality. Accelerator profiling and downstream model evaluation are therefore
extensions, not results implied by this repository.

## Evaluation principles

- State architectural assumptions before presenting a parameter count.
- Keep analytical estimates separate from empirical measurements.
- Report batch size, sequence length, data type, and KV-head count with memory.
- Distinguish model weights, activations, attention workspaces, and KV cache.
- Treat theoretical savings as hypotheses until the implementation is profiled.

## Planned progression

1. Parameter and memory accounting with explicit assumptions (complete)
2. CPU timing harness with warm-up, synchronisation hooks, and robust summaries (complete)
3. Exact blockwise attention and sequence-length scaling study (complete)
4. KV-cache decoding experiment and grouped-query comparison (complete)
5. Weight-only quantisation and accuracy/memory trade-offs (complete)
6. Throughput, latency, and memory benchmark matrix (complete)
7. Reproducible systems report and portfolio integration (complete)
