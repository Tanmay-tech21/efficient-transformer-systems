import pytest

from transformer_systems import (
    EfficiencyMeasurement,
    TimingSummary,
    compare_efficiency,
)


def timing(median: float) -> TimingSummary:
    return TimingSummary(
        repetitions=3,
        median_seconds=median,
        q1_seconds=median,
        q3_seconds=median,
        iqr_seconds=0.0,
        minimum_seconds=median,
        maximum_seconds=median,
        samples_seconds=(median, median, median),
    )


def measurement(**overrides) -> EfficiencyMeasurement:
    values = dict(
        workload="decode",
        variant="baseline",
        useful_work=2,
        work_unit="tokens",
        timing=timing(0.5),
        persistent_bytes=100,
        workspace_bytes=50,
        numerical_error=0.0,
    )
    values.update(overrides)
    return EfficiencyMeasurement(**values)


def test_measurement_derives_throughput_and_managed_bytes() -> None:
    record = measurement()

    assert record.throughput == 4.0
    assert record.managed_bytes == 150
    assert record.to_dict()["timing"]["repetitions"] == 3


def test_comparison_reports_candidate_to_baseline_ratios() -> None:
    baseline = measurement()
    candidate = measurement(
        variant="candidate",
        timing=timing(0.25),
        persistent_bytes=25,
        workspace_bytes=100,
    )

    result = compare_efficiency(baseline, candidate)

    assert result.latency_ratio == 0.5
    assert result.throughput_ratio == 2.0
    assert result.persistent_bytes_ratio == 0.25
    assert result.workspace_bytes_ratio == 2.0


def test_zero_baseline_memory_has_no_misleading_ratio() -> None:
    baseline = measurement(persistent_bytes=0, workspace_bytes=0)
    candidate = measurement(
        variant="candidate", persistent_bytes=0, workspace_bytes=10
    )

    result = compare_efficiency(baseline, candidate)

    assert result.persistent_bytes_ratio == 1.0
    assert result.workspace_bytes_ratio is None


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"workload": " "}, "workload"),
        ({"useful_work": 0}, "useful_work"),
        ({"persistent_bytes": -1}, "persistent_bytes"),
        ({"workspace_bytes": 1.5}, "workspace_bytes"),
        ({"numerical_error": float("inf")}, "numerical_error"),
    ],
)
def test_measurement_rejects_invalid_contract(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        measurement(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"workload": "prefill"},
        {"useful_work": 3},
        {"work_unit": "sequences"},
    ],
)
def test_comparison_rejects_incompatible_work(overrides) -> None:
    with pytest.raises(ValueError, match="same workload"):
        compare_efficiency(measurement(), measurement(**overrides))
