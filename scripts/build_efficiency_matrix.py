"""Build a cross-workload transformer efficiency matrix on the current CPU."""

from __future__ import annotations

import argparse
import json
import platform
from typing import Callable

import numpy as np

from transformer_systems import (
    EfficiencyMeasurement,
    benchmark_callable,
    blockwise_attention,
    compare_efficiency,
    dense_attention,
    grouped_query_decode,
    quantize_symmetric_int8,
    relative_root_mean_square_error,
    score_workspace_elements,
    weight_only_linear,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-sequence", type=int, default=512)
    parser.add_argument("--decode-context", type=int, default=4096)
    parser.add_argument("--linear-features", type=int, default=1024)
    parser.add_argument("--linear-tokens", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20)
    return parser.parse_args()


def environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "not reported",
        "numpy": np.__version__,
    }


def timed(
    operation: Callable[[], np.ndarray],
    *,
    warmup: int,
    repetitions: int,
):
    return benchmark_callable(
        operation,
        warmup=warmup,
        repetitions=repetitions,
    )


def attention_records(
    generator: np.random.Generator, args: argparse.Namespace
) -> tuple[EfficiencyMeasurement, EfficiencyMeasurement]:
    shape = (1, 4, args.attention_sequence, 64)
    query = generator.standard_normal(shape, dtype=np.float32)
    key = generator.standard_normal(shape, dtype=np.float32)
    value = generator.standard_normal(shape, dtype=np.float32)
    dense_output = dense_attention(query, key, value)
    tiled_output = blockwise_attention(
        query,
        key,
        value,
        query_block_size=args.block_size,
        key_block_size=args.block_size,
    )
    error = float(np.max(np.abs(tiled_output - dense_output)))
    itemsize = query.dtype.itemsize
    common = {
        "workload": f"prefill_attention_s{args.attention_sequence}",
        "useful_work": args.attention_sequence,
        "work_unit": "query_tokens",
        "persistent_bytes": 0,
    }
    dense = EfficiencyMeasurement(
        **common,
        variant="dense",
        timing=timed(
            lambda: dense_attention(query, key, value),
            warmup=args.warmup,
            repetitions=args.repetitions,
        ),
        workspace_bytes=score_workspace_elements(query, key) * itemsize,
        numerical_error=0.0,
    )
    blockwise = EfficiencyMeasurement(
        **common,
        variant=f"blockwise_{args.block_size}",
        timing=timed(
            lambda: blockwise_attention(
                query,
                key,
                value,
                query_block_size=args.block_size,
                key_block_size=args.block_size,
            ),
            warmup=args.warmup,
            repetitions=args.repetitions,
        ),
        workspace_bytes=(
            score_workspace_elements(
                query,
                key,
                query_block_size=args.block_size,
                key_block_size=args.block_size,
            )
            * itemsize
        ),
        numerical_error=error,
    )
    return dense, blockwise


def decode_records(
    generator: np.random.Generator, args: argparse.Namespace
) -> tuple[EfficiencyMeasurement, EfficiencyMeasurement]:
    query_heads = 8
    key_value_heads = 2
    group_size = query_heads // key_value_heads
    query = generator.standard_normal((1, query_heads, 64), dtype=np.float32)
    grouped_shape = (1, key_value_heads, args.decode_context, 64)
    grouped_key = generator.standard_normal(grouped_shape, dtype=np.float32)
    grouped_value = generator.standard_normal(grouped_shape, dtype=np.float32)
    multi_key = np.repeat(grouped_key, group_size, axis=1)
    multi_value = np.repeat(grouped_value, group_size, axis=1)
    expected = grouped_query_decode(query, multi_key, multi_value)
    actual = grouped_query_decode(query, grouped_key, grouped_value)
    error = float(np.max(np.abs(actual - expected)))
    score_bytes = query_heads * args.decode_context * query.dtype.itemsize
    common = {
        "workload": f"single_token_decode_s{args.decode_context}",
        "useful_work": 1,
        "work_unit": "generated_tokens",
        "workspace_bytes": score_bytes,
    }
    multi_head = EfficiencyMeasurement(
        **common,
        variant="multi_head_8kv",
        timing=timed(
            lambda: grouped_query_decode(query, multi_key, multi_value),
            warmup=args.warmup,
            repetitions=args.repetitions,
        ),
        persistent_bytes=multi_key.nbytes + multi_value.nbytes,
        numerical_error=0.0,
    )
    grouped_query = EfficiencyMeasurement(
        **common,
        variant="grouped_query_2kv",
        timing=timed(
            lambda: grouped_query_decode(query, grouped_key, grouped_value),
            warmup=args.warmup,
            repetitions=args.repetitions,
        ),
        persistent_bytes=grouped_key.nbytes + grouped_value.nbytes,
        numerical_error=error,
    )
    return multi_head, grouped_query


def linear_records(
    generator: np.random.Generator, args: argparse.Namespace
) -> tuple[EfficiencyMeasurement, EfficiencyMeasurement]:
    size = args.linear_features
    weights = generator.standard_normal((size, size), dtype=np.float32)
    row_magnitudes = np.geomspace(0.1, 10.0, size).astype(np.float32)
    weights *= row_magnitudes[:, None]
    inputs = generator.standard_normal(
        (args.linear_tokens, size), dtype=np.float32
    )
    quantized = quantize_symmetric_int8(
        weights, granularity="per_output_channel"
    )
    expected = np.matmul(inputs, weights.T)
    actual = weight_only_linear(inputs, quantized)
    error = relative_root_mean_square_error(actual, expected)
    common = {
        "workload": f"linear_{size}x{size}_t{args.linear_tokens}",
        "useful_work": args.linear_tokens,
        "work_unit": "input_tokens",
    }
    fp32 = EfficiencyMeasurement(
        **common,
        variant="fp32",
        timing=timed(
            lambda: np.matmul(inputs, weights.T),
            warmup=args.warmup,
            repetitions=args.repetitions,
        ),
        persistent_bytes=weights.nbytes,
        workspace_bytes=0,
        numerical_error=0.0,
    )
    int8 = EfficiencyMeasurement(
        **common,
        variant="int8_per_channel_reference",
        timing=timed(
            lambda: weight_only_linear(inputs, quantized),
            warmup=args.warmup,
            repetitions=args.repetitions,
        ),
        persistent_bytes=quantized.storage_bytes,
        # The reference reconstructs this float32 matrix for every invocation.
        workspace_bytes=weights.nbytes,
        numerical_error=error,
    )
    return fp32, int8


def main() -> None:
    args = parse_args()
    generator = np.random.default_rng(args.seed)
    pairs = [
        attention_records(generator, args),
        decode_records(generator, args),
        linear_records(generator, args),
    ]
    records = [record for pair in pairs for record in pair]
    comparisons = [compare_efficiency(*pair) for pair in pairs]
    print(
        json.dumps(
            {
                "status": (
                    "measured NumPy CPU matrix; declared bytes are managed "
                    "state/workspace, not process peak memory"
                ),
                "configuration": vars(args),
                "environment": environment(),
                "records": [record.to_dict() for record in records],
                "comparisons": [comparison.to_dict() for comparison in comparisons],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
