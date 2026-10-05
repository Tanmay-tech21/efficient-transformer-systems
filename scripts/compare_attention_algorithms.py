"""Compare dense and exact blockwise attention on the current CPU."""

from __future__ import annotations

import argparse
import json
import platform
from typing import Any, Callable

import numpy as np

from transformer_systems import (
    benchmark_callable,
    blockwise_attention,
    dense_attention,
    score_workspace_elements,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-width", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "not reported",
        "numpy": np.__version__,
    }


def measure(operation: Callable[[], np.ndarray], warmup: int, repetitions: int) -> dict[str, Any]:
    return benchmark_callable(
        operation,
        warmup=warmup,
        repetitions=repetitions,
    ).to_dict()


def main() -> None:
    args = parse_args()
    generator = np.random.default_rng(args.seed)
    results = []

    for sequence_length in args.sequence_lengths:
        shape = (1, args.heads, sequence_length, args.head_width)
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
        maximum_absolute_error = float(np.max(np.abs(dense_output - tiled_output)))
        if not np.allclose(dense_output, tiled_output, rtol=2e-5, atol=2e-6):
            raise RuntimeError("blockwise attention failed numerical validation")

        dense_workspace = score_workspace_elements(query, key)
        tiled_workspace = score_workspace_elements(
            query,
            key,
            query_block_size=args.block_size,
            key_block_size=args.block_size,
        )
        dense_timing = measure(
            lambda: dense_attention(query, key, value),
            args.warmup,
            args.repetitions,
        )
        tiled_timing = measure(
            lambda: blockwise_attention(
                query,
                key,
                value,
                query_block_size=args.block_size,
                key_block_size=args.block_size,
            ),
            args.warmup,
            args.repetitions,
        )
        results.append(
            {
                "sequence_length": sequence_length,
                "shape": list(shape),
                "maximum_absolute_error": maximum_absolute_error,
                "dense": {
                    "score_workspace_elements": dense_workspace,
                    "timing": dense_timing,
                },
                "blockwise": {
                    "score_workspace_elements": tiled_workspace,
                    "timing": tiled_timing,
                },
                "score_workspace_reduction": dense_workspace / tiled_workspace,
                "latency_ratio_blockwise_over_dense": (
                    tiled_timing["median_seconds"] / dense_timing["median_seconds"]
                ),
            }
        )

    print(
        json.dumps(
            {
                "status": "measured NumPy CPU prototype; not a fused-kernel benchmark",
                "configuration": {
                    "block_size": args.block_size,
                    "warmup": args.warmup,
                    "repetitions": args.repetitions,
                    "seed": args.seed,
                    "dtype": "float32",
                },
                "environment": environment(),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
