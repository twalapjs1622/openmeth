"""
Multi-Head Attention mechanism implemented in pure numpy.

This is the core attention primitive used by all OpenMythos
recurrent blocks. No PyTorch — all operations are numpy.
"""

import numpy as np
from typing import Optional, Tuple


class MultiHeadAttention:
    """
    Multi-head self-attention using numpy.
    
    Computes:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    With multiple heads projected independently and concatenated.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        dtype: np.dtype = np.float32,
    ):
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = np.sqrt(self.head_dim).astype(dtype)
        self.dropout_rate = dropout
        self.dtype = dtype

        # Initialize projection matrices (Xavier/Glorot)
        limit = np.sqrt(6.0 / (hidden_dim + hidden_dim))
        
        self.W_q = self._init_weight(hidden_dim, hidden_dim, limit)
        self.W_k = self._init_weight(hidden_dim, hidden_dim, limit)
        self.W_v = self._init_weight(hidden_dim, hidden_dim, limit)
        self.W_o = self._init_weight(hidden_dim, hidden_dim, limit)

        self.b_q = np.zeros(hidden_dim, dtype=dtype)
        self.b_k = np.zeros(hidden_dim, dtype=dtype)
        self.b_v = np.zeros(hidden_dim, dtype=dtype)
        self.b_o = np.zeros(hidden_dim, dtype=dtype)

    def _init_weight(self, fan_in: int, fan_out: int, 
                     limit: float) -> np.ndarray:
        """Xavier uniform initialization."""
        return np.random.uniform(
            -limit, limit, (fan_in, fan_out)
        ).astype(self.dtype)

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Shape (batch, seq_len, hidden_dim)
            key: Shape (batch, seq_len, hidden_dim)
            value: Shape (batch, seq_len, hidden_dim)
            mask: Optional attention mask (batch, 1, seq_len, seq_len)
        
        Returns:
            output: Shape (batch, seq_len, hidden_dim)
            attention_weights: Shape (batch, num_heads, seq_len, seq_len)
        """
        batch_size = query.shape[0]
        seq_q = query.shape[1]
        seq_k = key.shape[1]

        # Linear projections
        Q = query @ self.W_q + self.b_q  # (B, S_q, D)
        K = key @ self.W_k + self.b_k    # (B, S_k, D)
        V = value @ self.W_v + self.b_v  # (B, S_k, D)

        # Reshape to multi-head: (B, H, S, d_k)
        Q = Q.reshape(batch_size, seq_q, self.num_heads, self.head_dim)
        Q = Q.transpose(0, 2, 1, 3)  # (B, H, S_q, d_k)
        
        K = K.reshape(batch_size, seq_k, self.num_heads, self.head_dim)
        K = K.transpose(0, 2, 1, 3)  # (B, H, S_k, d_k)
        
        V = V.reshape(batch_size, seq_k, self.num_heads, self.head_dim)
        V = V.transpose(0, 2, 1, 3)  # (B, H, S_k, d_k)

        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / self.scale
        # scores shape: (B, H, S, S)

        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        # Softmax along last dimension
        attn_weights = self._softmax(scores, axis=-1)

        # Apply dropout (training mode indicator handled externally)
        if self.dropout_rate > 0:
            dropout_mask = np.random.random(attn_weights.shape) > self.dropout_rate
            attn_weights = attn_weights * dropout_mask / (1.0 - self.dropout_rate)

        # Weighted sum
        context = np.matmul(attn_weights, V)  # (B, H, S, d_k)

        # Concatenate heads
        context = context.transpose(0, 2, 1, 3)  # (B, S_q, H, d_k)
        context = context.reshape(batch_size, seq_q, self.hidden_dim)

        # Output projection
        output = context @ self.W_o + self.b_o

        return output, attn_weights

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def load_weights(self, W_q: np.ndarray, W_k: np.ndarray,
                     W_v: np.ndarray, W_o: np.ndarray,
                     b_q: np.ndarray, b_k: np.ndarray,
                     b_v: np.ndarray, b_o: np.ndarray):
        """Load pre-trained weight matrices."""
        self.W_q = W_q.astype(self.dtype)
        self.W_k = W_k.astype(self.dtype)
        self.W_v = W_v.astype(self.dtype)
        self.W_o = W_o.astype(self.dtype)
        self.b_q = b_q.astype(self.dtype)
        self.b_k = b_k.astype(self.dtype)
        self.b_v = b_v.astype(self.dtype)
        self.b_o = b_o.astype(self.dtype)


class CausalSelfAttention(MultiHeadAttention):
    """
    Causal (autoregressive) self-attention with triangular mask.
    Used in recurrent blocks where future tokens must be masked.
    """

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Causal self-attention forward.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            mask: Optional additional mask
        
        Returns:
            output: (batch, seq_len, hidden_dim)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        seq_len = x.shape[1]
        
        # Create causal mask
        causal_mask = np.triu(
            np.ones((seq_len, seq_len), dtype=self.dtype), k=1
        )
        causal_mask = 1 - causal_mask  # Lower triangular
        causal_mask = causal_mask[np.newaxis, np.newaxis, :, :]  # (1, 1, S, S)
        
        if mask is not None:
            causal_mask = causal_mask * mask

        return super().forward(x, x, x, mask=causal_mask)


class CrossAttention(MultiHeadAttention):
    """
    Cross-attention between two sequences.
    Used for reasoning path fusion in OpenMythos.
    """

    def forward(
        self,
        query_seq: np.ndarray,
        context_seq: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cross-attention forward.
        
        Args:
            query_seq: Query sequence (batch, seq_q, hidden_dim)
            context_seq: Context to attend to (batch, seq_c, hidden_dim)
            mask: Optional mask
        
        Returns:
            output: (batch, seq_q, hidden_dim)
            attention_weights: (batch, num_heads, seq_q, seq_c)
        """
        return super().forward(query_seq, context_seq, context_seq, mask=mask)