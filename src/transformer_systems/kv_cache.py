"""Preallocated KV-cache storage and single-token grouped-query attention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import DTypeLike, NDArray


FloatArray = NDArray[np.floating]


def _positive_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class KVCacheStats:
    """Logical and allocated storage for both key and value tensors."""

    length: int
    capacity: int
    active_bytes: int
    allocated_bytes: int


class KVCache:
    """Fixed-capacity KV cache that appends without reallocating prior tokens."""

    def __init__(
        self,
        *,
        batch_size: int,
        key_value_heads: int,
        capacity: int,
        head_width: int,
        dtype: DTypeLike = np.float32,
    ) -> None:
        for name, value in (
            ("batch_size", batch_size),
            ("key_value_heads", key_value_heads),
            ("capacity", capacity),
            ("head_width", head_width),
        ):
            _positive_integer(name, value)
        storage_dtype = np.dtype(dtype)
        if not np.issubdtype(storage_dtype, np.floating):
            raise TypeError("KV cache dtype must be floating point")

        shape = (batch_size, key_value_heads, capacity, head_width)
        self._keys = np.empty(shape, dtype=storage_dtype)
        self._values = np.empty(shape, dtype=storage_dtype)
        self._length = 0

    @property
    def length(self) -> int:
        return self._length

    @property
    def capacity(self) -> int:
        return self._keys.shape[2]

    @property
    def keys(self) -> FloatArray:
        return self._keys[:, :, : self._length, :]

    @property
    def values(self) -> FloatArray:
        return self._values[:, :, : self._length, :]

    def append(self, key: FloatArray, value: FloatArray) -> None:
        """Append one or more positions after validating the whole update."""
        expected_prefix = self._keys.shape[:2]
        expected_width = self._keys.shape[-1]
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("key and value chunks must have rank 4")
        if key.shape != value.shape:
            raise ValueError("key and value chunks must have identical shapes")
        if key.shape[:2] != expected_prefix or key.shape[-1] != expected_width:
            raise ValueError("key and value chunks do not match the cache shape")
        if key.shape[2] == 0:
            raise ValueError("key and value chunks must contain at least one token")
        if not all(np.issubdtype(array.dtype, np.floating) for array in (key, value)):
            raise TypeError("key and value chunks must be floating point")

        stop = self._length + key.shape[2]
        if stop > self.capacity:
            raise ValueError("append would exceed KV cache capacity")

        self._keys[:, :, self._length : stop, :] = key
        self._values[:, :, self._length : stop, :] = value
        self._length = stop

    def reset(self) -> None:
        """Clear the logical sequence while retaining the allocation."""
        self._length = 0

    def stats(self) -> KVCacheStats:
        bytes_per_position = (
            self._keys.shape[0]
            * self._keys.shape[1]
            * self._keys.shape[3]
            * self._keys.dtype.itemsize
            * 2
        )
        return KVCacheStats(
            length=self.length,
            capacity=self.capacity,
            active_bytes=self.length * bytes_per_position,
            allocated_bytes=self.capacity * bytes_per_position,
        )


def grouped_query_decode(
    query: FloatArray,
    key_cache: FloatArray,
    value_cache: FloatArray,
) -> FloatArray:
    """Attend one query token to a cached prefix without repeating KV heads.

    Query heads are assigned contiguously to key/value heads, matching the
    ordering produced by repeating every KV head by the same group factor.
    """
    if query.ndim != 3:
        raise ValueError("query must have shape [batch, query_heads, head_width]")
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("key and value caches must have rank 4")
    if key_cache.shape != value_cache.shape:
        raise ValueError("key and value caches must have identical shapes")
    if query.shape[0] != key_cache.shape[0]:
        raise ValueError("query and cache batch sizes must match")
    if query.shape[-1] != key_cache.shape[-1]:
        raise ValueError("query and cache head widths must match")
    if key_cache.shape[2] == 0:
        raise ValueError("cannot decode with an empty KV cache")
    if not all(
        np.issubdtype(array.dtype, np.floating)
        for array in (query, key_cache, value_cache)
    ):
        raise TypeError("query and caches must be floating point")

    query_heads = query.shape[1]
    key_value_heads = key_cache.shape[1]
    if query_heads % key_value_heads != 0:
        raise ValueError("query heads must be divisible by key/value heads")

    group_size = query_heads // key_value_heads
    grouped_query = query.reshape(
        query.shape[0], key_value_heads, group_size, query.shape[-1]
    )
    scale = np.sqrt(np.asarray(query.shape[-1], dtype=query.dtype))
    scores = np.einsum(
        "bhgd,bhsd->bhgs", grouped_query, key_cache, optimize=True
    ) / scale
    scores -= np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=-1, keepdims=True)
    output = np.einsum("bhgs,bhsv->bhgv", weights, value_cache, optimize=True)
    return output.reshape(query.shape[0], query_heads, value_cache.shape[-1])
