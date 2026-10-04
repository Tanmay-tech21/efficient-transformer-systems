"""Measure a transparent dense-attention reference on the current CPU."""

from __future__ import annotations

import argparse
import json
import platform
from typing import Any

import numpy as np

from transformer_systems import benchmark_callable, dense_attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-width", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=16)
    return parser.parse_args()


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "not reported",
        "numpy": np.__version__,
    }


def main() -> None:
    args = parse_args()
    if any(length <= 0 for length in args.sequence_lengths):
        raise ValueError("sequence lengths must be positive")

    generator = np.random.default_rng(args.seed)
    results = []
    for sequence_length in args.sequence_lengths:
        shape = (1, args.heads, sequence_length, args.head_width)
        query = generator.standard_normal(shape, dtype=np.float32)
        key = generator.standard_normal(shape, dtype=np.float32)
        value = generator.standard_normal(shape, dtype=np.float32)
        output = dense_attention(query, key, value)
        if output.shape != shape or not np.isfinite(output).all():
            raise RuntimeError("attention workload failed validation")

        timing = benchmark_callable(
            lambda: dense_attention(query, key, value),
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        results.append(
            {
                "sequence_length": sequence_length,
                "shape": list(shape),
                "score_elements": args.heads * sequence_length**2,
                "timing": timing.to_dict(),
            }
        )

    report = {
        "status": "measured CPU microbenchmark; results are environment-specific",
        "workload": "NumPy dense scaled dot-product attention, non-causal, float32",
        "configuration": {
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "seed": args.seed,
        },
        "environment": environment(),
        "results": results,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
