import numpy as np
import pytest

from transformer_systems import benchmark_callable, dense_attention, summarise_timings


def test_timing_summary_reports_median_and_iqr() -> None:
    summary = summarise_timings((0.4, 0.1, 0.3, 0.2, 0.5))

    assert summary.median_seconds == pytest.approx(0.3)
    assert summary.q1_seconds == pytest.approx(0.2)
    assert summary.q3_seconds == pytest.approx(0.4)
    assert summary.iqr_seconds == pytest.approx(0.2)
    assert summary.samples_seconds == (0.4, 0.1, 0.3, 0.2, 0.5)


def test_benchmark_warms_up_and_synchronises_measurements() -> None:
    calls = {"operation": 0, "sync": 0}
    timestamps = iter((0, 10, 20, 50, 60, 100))

    def operation() -> None:
        calls["operation"] += 1

    def synchronize() -> None:
        calls["sync"] += 1

    summary = benchmark_callable(
        operation,
        warmup=2,
        repetitions=3,
        synchronize=synchronize,
        clock_ns=lambda: next(timestamps),
    )

    assert calls == {"operation": 5, "sync": 7}
    assert summary.samples_seconds == (10e-9, 30e-9, 40e-9)


def test_dense_attention_matches_uniform_weight_case() -> None:
    query = np.zeros((1, 2, 3, 4), dtype=np.float32)
    key = np.zeros_like(query)
    value = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)

    output = dense_attention(query, key, value)
    expected = value.mean(axis=-2, keepdims=True)

    np.testing.assert_allclose(output, np.repeat(expected, 3, axis=-2))


def test_causal_attention_cannot_see_future_values() -> None:
    query = np.zeros((1, 1, 3, 2), dtype=np.float32)
    key = np.zeros_like(query)
    value = np.array([[[[1.0], [3.0], [8.0]]]], dtype=np.float32)

    output = dense_attention(query, key, value, causal=True)

    np.testing.assert_allclose(output[0, 0, :, 0], (1.0, 2.0, 4.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"warmup": 0},
        {"repetitions": 0},
        {"warmup": True},
    ],
)
def test_invalid_benchmark_counts_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        benchmark_callable(lambda: None, **kwargs)


def test_invalid_attention_shapes_are_rejected() -> None:
    array = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="head widths"):
        dense_attention(array, np.ones((2, 4), dtype=np.float32), array)
