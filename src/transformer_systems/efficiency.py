"""Common records for comparing transformer-system microbenchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

from .benchmarking import TimingSummary


@dataclass(frozen=True)
class EfficiencyMeasurement:
    """One measured variant with an explicit throughput and memory contract."""

    workload: str
    variant: str
    useful_work: int
    work_unit: str
    timing: TimingSummary
    persistent_bytes: int
    workspace_bytes: int
    numerical_error: float

    def __post_init__(self) -> None:
        for name, value in (
            ("workload", self.workload),
            ("variant", self.variant),
            ("work_unit", self.work_unit),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        if (
            not isinstance(self.useful_work, int)
            or isinstance(self.useful_work, bool)
            or self.useful_work <= 0
        ):
            raise ValueError("useful_work must be a positive integer")
        for name, value in (
            ("persistent_bytes", self.persistent_bytes),
            ("workspace_bytes", self.workspace_bytes),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isfinite(self.numerical_error) or self.numerical_error < 0:
            raise ValueError("numerical_error must be finite and non-negative")

    @property
    def throughput(self) -> float:
        return self.useful_work / self.timing.median_seconds

    @property
    def managed_bytes(self) -> int:
        """Return declared persistent plus workspace bytes, not process peak."""
        return self.persistent_bytes + self.workspace_bytes

    def to_dict(self) -> dict[str, object]:
        return {
            "workload": self.workload,
            "variant": self.variant,
            "useful_work": self.useful_work,
            "work_unit": self.work_unit,
            "throughput_per_second": self.throughput,
            "persistent_bytes": self.persistent_bytes,
            "workspace_bytes": self.workspace_bytes,
            "managed_bytes": self.managed_bytes,
            "numerical_error": self.numerical_error,
            "timing": self.timing.to_dict(),
        }


@dataclass(frozen=True)
class EfficiencyComparison:
    """Candidate-to-baseline ratios for one identically shaped workload."""

    workload: str
    baseline: str
    candidate: str
    latency_ratio: float
    throughput_ratio: float
    persistent_bytes_ratio: float | None
    workspace_bytes_ratio: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _ratio(candidate: int, baseline: int) -> float | None:
    if baseline == 0:
        return 1.0 if candidate == 0 else None
    return candidate / baseline


def compare_efficiency(
    baseline: EfficiencyMeasurement,
    candidate: EfficiencyMeasurement,
) -> EfficiencyComparison:
    """Compare records only when their workload and useful work are identical."""
    if (
        baseline.workload != candidate.workload
        or baseline.useful_work != candidate.useful_work
        or baseline.work_unit != candidate.work_unit
    ):
        raise ValueError("measurements must describe the same workload and useful work")
    return EfficiencyComparison(
        workload=baseline.workload,
        baseline=baseline.variant,
        candidate=candidate.variant,
        latency_ratio=(
            candidate.timing.median_seconds / baseline.timing.median_seconds
        ),
        throughput_ratio=candidate.throughput / baseline.throughput,
        persistent_bytes_ratio=_ratio(
            candidate.persistent_bytes, baseline.persistent_bytes
        ),
        workspace_bytes_ratio=_ratio(
            candidate.workspace_bytes, baseline.workspace_bytes
        ),
    )
