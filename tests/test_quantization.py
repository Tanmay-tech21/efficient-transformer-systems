import numpy as np
import pytest

from transformer_systems import (
    SymmetricInt8Weights,
    dequantize_weights,
    quantize_symmetric_int8,
    relative_root_mean_square_error,
    weight_only_linear,
)


@pytest.mark.parametrize("granularity", ["per_tensor", "per_output_channel"])
def test_quantization_round_trip_is_close(granularity) -> None:
    generator = np.random.default_rng(19)
    weights = generator.uniform(-1.0, 1.0, size=(8, 16)).astype(np.float32)

    quantized = quantize_symmetric_int8(weights, granularity=granularity)
    reconstructed = dequantize_weights(quantized)

    assert quantized.values.dtype == np.int8
    assert reconstructed.dtype == np.float32
    assert relative_root_mean_square_error(reconstructed, weights) < 0.01


def test_per_channel_scales_improve_heterogeneous_rows() -> None:
    generator = np.random.default_rng(7)
    weights = generator.standard_normal((4, 64), dtype=np.float32)
    weights *= np.asarray([0.01, 0.1, 1.0, 10.0], dtype=np.float32)[:, None]
    tensor = dequantize_weights(
        quantize_symmetric_int8(weights, granularity="per_tensor")
    )
    channel = dequantize_weights(
        quantize_symmetric_int8(weights, granularity="per_output_channel")
    )

    tensor_error = relative_root_mean_square_error(tensor[0], weights[0])
    channel_error = relative_root_mean_square_error(channel[0], weights[0])
    assert channel_error < tensor_error


def test_zero_rows_remain_zero() -> None:
    weights = np.zeros((3, 5), dtype=np.float32)
    quantized = quantize_symmetric_int8(
        weights, granularity="per_output_channel"
    )

    np.testing.assert_array_equal(quantized.values, 0)
    np.testing.assert_array_equal(dequantize_weights(quantized), weights)


def test_storage_includes_scale_metadata() -> None:
    weights = np.ones((8, 16), dtype=np.float32)
    quantized = quantize_symmetric_int8(
        weights, granularity="per_output_channel"
    )

    assert weights.nbytes == 8 * 16 * 4
    assert quantized.storage_bytes == 8 * 16 + 8 * 4


def test_weight_only_linear_matches_dequantized_reference() -> None:
    generator = np.random.default_rng(11)
    inputs = generator.standard_normal((2, 3, 5), dtype=np.float32)
    weights = generator.standard_normal((7, 5), dtype=np.float32)
    bias = generator.standard_normal(7, dtype=np.float32)
    quantized = quantize_symmetric_int8(weights)

    expected = np.matmul(inputs, dequantize_weights(quantized).T) + bias
    actual = weight_only_linear(inputs, quantized, bias)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_relative_rmse_handles_zero_reference() -> None:
    zeros = np.zeros(4, dtype=np.float32)
    ones = np.ones(4, dtype=np.float32)

    assert relative_root_mean_square_error(zeros, zeros) == 0.0
    assert relative_root_mean_square_error(ones, zeros) == float("inf")


@pytest.mark.parametrize(
    "weights, exception",
    [
        (np.ones(3, dtype=np.float32), ValueError),
        (np.ones((2, 3), dtype=np.int8), TypeError),
        (np.asarray([[np.nan]], dtype=np.float32), ValueError),
    ],
)
def test_quantization_rejects_invalid_weights(weights, exception) -> None:
    with pytest.raises(exception):
        quantize_symmetric_int8(weights)


def test_quantization_rejects_unknown_granularity() -> None:
    with pytest.raises(ValueError, match="granularity"):
        quantize_symmetric_int8(
            np.ones((2, 3), dtype=np.float32), granularity="per_token"
        )


def test_dequantization_validates_scale_count() -> None:
    invalid = SymmetricInt8Weights(
        values=np.ones((2, 3), dtype=np.int8),
        scales=np.ones(1, dtype=np.float32),
        granularity="per_output_channel",
    )
    with pytest.raises(ValueError, match="one scale per output"):
        dequantize_weights(invalid)
