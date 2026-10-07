"""Measure storage, error, and reference latency for INT8 weight-only linear layers."""

from __future__ import annotations

import argparse
import json
import platform
from typing import Any

import numpy as np

from transformer_systems import (
    benchmark_callable,
    dequantize_weights,
    quantize_symmetric_int8,
    relative_root_mean_square_error,
    weight_only_linear,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-features", type=int, default=1024)
    parser.add_argument("--output-features", type=int, default=1024)
    parser.add_argument("--token-counts", nargs="+", type=int, default=[1, 32])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "not reported",
        "numpy": np.__version__,
    }


def error_summary(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    denominator = float(np.linalg.norm(actual) * np.linalg.norm(expected))
    cosine = float(np.dot(actual.ravel(), expected.ravel()) / denominator)
    return {
        "maximum_absolute_error": float(np.max(np.abs(actual - expected))),
        "relative_rmse": relative_root_mean_square_error(actual, expected),
        "cosine_similarity": cosine,
    }


def main() -> None:
    args = parse_args()
    generator = np.random.default_rng(args.seed)
    weights = generator.standard_normal(
        (args.output_features, args.input_features), dtype=np.float32
    )
    # Varying row magnitudes make the scale-granularity trade-off observable.
    row_magnitudes = np.geomspace(0.1, 10.0, args.output_features).astype(np.float32)
    weights *= row_magnitudes[:, None]

    quantized = {
        granularity: quantize_symmetric_int8(weights, granularity=granularity)
        for granularity in ("per_tensor", "per_output_channel")
    }
    fp32_bytes = weights.nbytes
    reconstruction = {
        name: {
            "persistent_bytes": value.storage_bytes,
            "compression_ratio": fp32_bytes / value.storage_bytes,
            "weight_relative_rmse": relative_root_mean_square_error(
                dequantize_weights(value), weights
            ),
        }
        for name, value in quantized.items()
    }

    results: list[dict[str, Any]] = []
    for token_count in args.token_counts:
        inputs = generator.standard_normal(
            (token_count, args.input_features), dtype=np.float32
        )
        expected = np.matmul(inputs, weights.T)
        fp32_timing = benchmark_callable(
            lambda: np.matmul(inputs, weights.T),
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        schemes = {}
        for name, value in quantized.items():
            actual = weight_only_linear(inputs, value)
            timing = benchmark_callable(
                lambda value=value: weight_only_linear(inputs, value),
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
            schemes[name] = {
                **error_summary(actual, expected),
                "timing": timing.to_dict(),
                "latency_ratio_over_fp32": (
                    timing.median_seconds / fp32_timing.median_seconds
                ),
            }
        results.append(
            {
                "token_count": token_count,
                "fp32_timing": fp32_timing.to_dict(),
                "quantized_schemes": schemes,
            }
        )

    print(
        json.dumps(
            {
                "status": (
                    "measured NumPy reference with per-call dequantisation; "
                    "not an optimised INT8 kernel"
                ),
                "configuration": {
                    "input_features": args.input_features,
                    "output_features": args.output_features,
                    "token_counts": args.token_counts,
                    "warmup": args.warmup,
                    "repetitions": args.repetitions,
                    "seed": args.seed,
                    "source_dtype": "float32",
                },
                "environment": environment(),
                "fp32_persistent_bytes": fp32_bytes,
                "quantization": reconstruction,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
