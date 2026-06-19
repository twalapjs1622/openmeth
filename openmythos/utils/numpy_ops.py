"""
Numpy utility operations for OpenMythos.

Optimized linear algebra and tensor operations
that replace PyTorch functionality.
"""

import numpy as np
from typing import Optional, Tuple


def matmul_safe(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Safe matrix multiplication with shape checking."""
    if a.ndim == 1:
        a = a[np.newaxis, :]
    if b.ndim == 1:
        b = b[:, np.newaxis]
    return np.matmul(a, b)


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Apply layer normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation function."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    scale: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scaled dot-product attention.
    
    Args:
        query: (..., seq_q, dim)
        key: (..., seq_k, dim)
        value: (..., seq_k, dim_v)
        scale: Scaling factor (default: 1/sqrt(dim))
        mask: Optional attention mask
    
    Returns:
        output: (..., seq_q, dim_v)
        weights: (..., seq_q, seq_k)
    """
    dim = query.shape[-1]
    if scale is None:
        scale = 1.0 / np.sqrt(dim)
    
    # Transpose last two dimensions: (..., seq_k, dim) -> (..., dim, seq_k)
    key_t = np.swapaxes(key, -2, -1)
    scores = np.matmul(query, key_t) * scale
    
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    weights = softmax(scores, axis=-1)
    output = np.matmul(weights, value)
    
    return output, weights


def positional_encoding(
    length: int,
    dim: int,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Generate sinusoidal positional encoding."""
    pe = np.zeros((length, dim), dtype=dtype)
    position = np.arange(0, length, dtype=dtype)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, dim, 2, dtype=dtype) * -(np.log(10000.0) / dim)
    )
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def truncated_normal(
    shape: Tuple[int, ...],
    mean: float = 0.0,
    std: float = 1.0,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Generate truncated normal distribution samples."""
    samples = np.random.randn(*shape).astype(dtype)
    # Truncate at 2 standard deviations
    samples = np.clip(samples, -2, 2)
    return samples * std + mean


def cosine_similarity(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute cosine similarity between arrays."""
    norm_a = np.linalg.norm(a, axis=axis, keepdims=True) + 1e-8
    norm_b = np.linalg.norm(b, axis=axis, keepdims=True) + 1e-8
    return np.sum(a * b, axis=axis, keepdims=True) / (norm_a * norm_b)