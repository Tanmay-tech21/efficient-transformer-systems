"""Readable NumPy attention used as a benchmark workload and reference."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]


def dense_attention(
    query: FloatArray,
    key: FloatArray,
    value: FloatArray,
    *,
    causal: bool = False,
) -> FloatArray:
    """Compute scaled dot-product attention over the penultimate axis."""
    if query.ndim < 2 or key.ndim != query.ndim or value.ndim != query.ndim:
        raise ValueError("query, key and value must have matching ranks of at least 2")
    if query.shape[:-2] != key.shape[:-2] or query.shape[:-2] != value.shape[:-2]:
        raise ValueError("query, key and value must share batch dimensions")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head widths must match")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("key and value sequence lengths must match")
    if causal and query.shape[-2] != key.shape[-2]:
        raise ValueError("causal attention requires equal query and key lengths")
    if not all(np.issubdtype(array.dtype, np.floating) for array in (query, key, value)):
        raise TypeError("attention inputs must use floating-point dtypes")

    scale = np.sqrt(np.asarray(query.shape[-1], dtype=query.dtype))
    scores = np.matmul(query, np.swapaxes(key, -1, -2)) / scale
    if causal:
        mask = np.triu(np.ones(scores.shape[-2:], dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)

    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=-1, keepdims=True)
    return np.matmul(weights, value)
