"""Reference symmetric INT8 weight-only quantisation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]
Granularity = Literal["per_tensor", "per_output_channel"]


@dataclass(frozen=True)
class SymmetricInt8Weights:
    """Quantised matrix and the scales required to reconstruct it."""

    values: NDArray[np.int8]
    scales: NDArray[np.float32]
    granularity: Granularity

    @property
    def storage_bytes(self) -> int:
        return self.values.nbytes + self.scales.nbytes


def _validate_weight_matrix(weights: FloatArray) -> None:
    if weights.ndim != 2:
        raise ValueError("weights must have shape [output_features, input_features]")
    if not np.issubdtype(weights.dtype, np.floating):
        raise TypeError("weights must use a floating-point dtype")
    if weights.size == 0:
        raise ValueError("weights must not be empty")
    if not np.isfinite(weights).all():
        raise ValueError("weights must contain only finite values")


def quantize_symmetric_int8(
    weights: FloatArray,
    *,
    granularity: Granularity = "per_output_channel",
) -> SymmetricInt8Weights:
    """Quantise a matrix with zero-centred scales and no zero-point.

    The signed range is deliberately restricted to [-127, 127], keeping the
    positive and negative reconstruction ranges symmetric around zero.
    """
    _validate_weight_matrix(weights)
    if granularity not in ("per_tensor", "per_output_channel"):
        raise ValueError("granularity must be per_tensor or per_output_channel")

    if granularity == "per_tensor":
        maxima = np.asarray([np.max(np.abs(weights))], dtype=np.float32)
        scales = np.where(maxima == 0, 1.0, maxima / 127.0).astype(np.float32)
        scaled = weights / scales[0]
    else:
        maxima = np.max(np.abs(weights), axis=1).astype(np.float32)
        scales = np.where(maxima == 0, 1.0, maxima / 127.0).astype(np.float32)
        scaled = weights / scales[:, None]

    values = np.clip(np.rint(scaled), -127, 127).astype(np.int8)
    return SymmetricInt8Weights(
        values=values,
        scales=scales,
        granularity=granularity,
    )


def dequantize_weights(weights: SymmetricInt8Weights) -> NDArray[np.float32]:
    """Reconstruct float32 weights from the persistent INT8 representation."""
    values = weights.values.astype(np.float32)
    if weights.granularity == "per_tensor":
        if weights.scales.shape != (1,):
            raise ValueError("per-tensor weights require exactly one scale")
        return values * weights.scales[0]
    if weights.granularity == "per_output_channel":
        if weights.scales.shape != (weights.values.shape[0],):
            raise ValueError("per-channel weights require one scale per output")
        return values * weights.scales[:, None]
    raise ValueError("unsupported quantisation granularity")


def weight_only_linear(
    inputs: FloatArray,
    weights: SymmetricInt8Weights,
    bias: FloatArray | None = None,
) -> NDArray[np.float32]:
    """Evaluate a linear layer by dequantising its weights for each call.

    This transparent reference path measures quantisation error correctly, but
    it is not an integer matrix-multiplication kernel and should not imply an
    optimised INT8 latency result.
    """
    if inputs.ndim < 1:
        raise ValueError("inputs must have at least one dimension")
    if not np.issubdtype(inputs.dtype, np.floating):
        raise TypeError("inputs must use a floating-point dtype")
    if inputs.shape[-1] != weights.values.shape[1]:
        raise ValueError("input width must match the quantised weight width")
    if not np.isfinite(inputs).all():
        raise ValueError("inputs must contain only finite values")
    if bias is not None:
        if bias.shape != (weights.values.shape[0],):
            raise ValueError("bias must have one value per output feature")
        if not np.issubdtype(bias.dtype, np.floating):
            raise TypeError("bias must use a floating-point dtype")

    output = np.matmul(inputs, dequantize_weights(weights).T)
    if bias is not None:
        output = output + bias
    return output.astype(np.float32, copy=False)


def relative_root_mean_square_error(
    actual: FloatArray,
    expected: FloatArray,
) -> float:
    """Return RMSE normalised by the root mean square of the reference."""
    if actual.shape != expected.shape or actual.size == 0:
        raise ValueError("actual and expected must have the same non-empty shape")
    if not all(np.issubdtype(x.dtype, np.floating) for x in (actual, expected)):
        raise TypeError("actual and expected must be floating point")
    reference_rms = float(np.sqrt(np.mean(np.square(expected, dtype=np.float64))))
    error_rms = float(
        np.sqrt(np.mean(np.square(actual - expected, dtype=np.float64)))
    )
    if reference_rms == 0:
        return 0.0 if error_rms == 0 else float("inf")
    return error_rms / reference_rms
