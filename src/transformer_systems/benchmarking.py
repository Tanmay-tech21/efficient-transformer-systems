"""Small, explicit timing primitives for reproducible microbenchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter_ns
from typing import Callable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class TimingSummary:
    repetitions: int
    median_seconds: float
    q1_seconds: float
    q3_seconds: float
    iqr_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    samples_seconds: tuple[float, ...]

    def to_dict(self) -> dict[str, int | float | list[float]]:
        values = asdict(self)
        values["samples_seconds"] = list(self.samples_seconds)
        return values


def _positive_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _quantile(sorted_values: tuple[float, ...], probability: float) -> float:
    """Return a linearly interpolated quantile using inclusive endpoints."""
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + fraction * (
        sorted_values[upper] - sorted_values[lower]
    )


def summarise_timings(samples_seconds: tuple[float, ...]) -> TimingSummary:
    """Summarise repeated measurements without hiding the raw samples."""
    if not samples_seconds:
        raise ValueError("at least one timing sample is required")
    if any(sample < 0 for sample in samples_seconds):
        raise ValueError("timing samples must be non-negative")

    ordered = tuple(sorted(float(sample) for sample in samples_seconds))
    q1 = _quantile(ordered, 0.25)
    median = _quantile(ordered, 0.5)
    q3 = _quantile(ordered, 0.75)
    return TimingSummary(
        repetitions=len(ordered),
        median_seconds=median,
        q1_seconds=q1,
        q3_seconds=q3,
        iqr_seconds=q3 - q1,
        minimum_seconds=ordered[0],
        maximum_seconds=ordered[-1],
        samples_seconds=tuple(float(sample) for sample in samples_seconds),
    )


def benchmark_callable(
    operation: Callable[[], T],
    *,
    warmup: int = 5,
    repetitions: int = 20,
    synchronize: Callable[[], None] | None = None,
    clock_ns: Callable[[], int] = perf_counter_ns,
) -> TimingSummary:
    """Time a callable after warm-up, with optional device synchronisation.

    Asynchronous accelerators require a synchronisation callback. Without one,
    the measured interval may capture dispatch rather than completed execution.
    """
    _positive_integer("warmup", warmup)
    _positive_integer("repetitions", repetitions)
    sync = synchronize or (lambda: None)

    for _ in range(warmup):
        operation()
    sync()

    samples = []
    for _ in range(repetitions):
        sync()
        start = clock_ns()
        operation()
        sync()
        elapsed_ns = clock_ns() - start
        if elapsed_ns < 0:
            raise RuntimeError("clock moved backwards during benchmark")
        samples.append(elapsed_ns / 1_000_000_000)
    return summarise_timings(tuple(samples))
