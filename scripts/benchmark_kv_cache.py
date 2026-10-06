"""Compare multi-head and grouped-query KV-cache decoding on this CPU."""

from __future__ import annotations

import argparse
import json
import platform
from typing import Any

import numpy as np

from transformer_systems import KVCache, benchmark_callable, grouped_query_decode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context-lengths", nargs="+", type=int, default=[128, 512, 2048, 4096]
    )
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--grouped-query-kv-heads", type=int, default=2)
    parser.add_argument("--head-width", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=18)
    return parser.parse_args()


def environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "not reported",
        "numpy": np.__version__,
    }


def run_configuration(
    *,
    generator: np.random.Generator,
    query: np.ndarray,
    context_length: int,
    query_heads: int,
    key_value_heads: int,
    head_width: int,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    cache = KVCache(
        batch_size=1,
        key_value_heads=key_value_heads,
        capacity=context_length,
        head_width=head_width,
    )
    shape = (1, key_value_heads, context_length, head_width)
    cache.append(
        generator.standard_normal(shape, dtype=np.float32),
        generator.standard_normal(shape, dtype=np.float32),
    )
    output = grouped_query_decode(query, cache.keys, cache.values)
    if output.shape != query.shape or not np.isfinite(output).all():
        raise RuntimeError("grouped-query decode failed output validation")

    timing = benchmark_callable(
        lambda: grouped_query_decode(query, cache.keys, cache.values),
        warmup=warmup,
        repetitions=repetitions,
    )
    stats = cache.stats()
    return {
        "key_value_heads": key_value_heads,
        "group_size": query_heads // key_value_heads,
        "active_cache_bytes": stats.active_bytes,
        "allocated_cache_bytes": stats.allocated_bytes,
        "score_elements": query_heads * context_length,
        "timing": timing.to_dict(),
    }


def main() -> None:
    args = parse_args()
    if args.query_heads % args.grouped_query_kv_heads != 0:
        raise ValueError("query heads must be divisible by grouped-query KV heads")
    generator = np.random.default_rng(args.seed)
    results = []

    for context_length in args.context_lengths:
        query = generator.standard_normal(
            (1, args.query_heads, args.head_width), dtype=np.float32
        )
        multi_head = run_configuration(
            generator=generator,
            query=query,
            context_length=context_length,
            query_heads=args.query_heads,
            key_value_heads=args.query_heads,
            head_width=args.head_width,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        grouped_query = run_configuration(
            generator=generator,
            query=query,
            context_length=context_length,
            query_heads=args.query_heads,
            key_value_heads=args.grouped_query_kv_heads,
            head_width=args.head_width,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        results.append(
            {
                "context_length": context_length,
                "multi_head": multi_head,
                "grouped_query": grouped_query,
                "cache_reduction": (
                    multi_head["active_cache_bytes"]
                    / grouped_query["active_cache_bytes"]
                ),
                "latency_ratio_grouped_over_multi_head": (
                    grouped_query["timing"]["median_seconds"]
                    / multi_head["timing"]["median_seconds"]
                ),
            }
        )

    print(
        json.dumps(
            {
                "status": "measured NumPy CPU prototype; not a fused-kernel benchmark",
                "configuration": {
                    "query_heads": args.query_heads,
                    "grouped_query_kv_heads": args.grouped_query_kv_heads,
                    "head_width": args.head_width,
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
