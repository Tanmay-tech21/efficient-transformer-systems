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
4. KV-cache decoding experiment and grouped-query comparison
5. Weight-only quantisation and accuracy/memory trade-offs
6. Throughput, latency, and memory benchmark matrix
7. Reproducible systems report and portfolio integration
