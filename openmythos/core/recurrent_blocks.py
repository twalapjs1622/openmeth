"""
Recurrent Block Stack — the heart of OpenMythos thinking.

Implements the core recurrent processing pipeline using pure numpy.
Each block takes a thinking representation, processes it through
recurrent connections, and produces a refined output.

The key innovation: these blocks don't just pass information forward.
They RECURSE — feeding output back as input until convergence,
creating increasingly refined reasoning representations.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass

from .attention import CausalSelfAttention, CrossAttention


class LayerNorm:
    """Layer normalization in pure numpy."""

    def __init__(self, dim: int, eps: float = 1e-5, dtype: np.dtype = np.float32):
        self.dim = dim
        self.eps = eps
        self.dtype = dtype
        self.gamma = np.ones(dim, dtype=dtype)
        self.beta = np.zeros(dim, dtype=dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class FeedForward:
    """Position-wise feed-forward network."""

    def __init__(self, hidden_dim: int, ff_dim: int = None, 
                 dtype: np.dtype = np.float32):
        ff_dim = ff_dim or 4 * hidden_dim
        self.dtype = dtype
        limit = np.sqrt(6.0 / (hidden_dim + ff_dim))
        self.W1 = np.random.uniform(-limit, limit, (hidden_dim, ff_dim)).astype(dtype)
        self.b1 = np.zeros(ff_dim, dtype=dtype)
        limit2 = np.sqrt(6.0 / (ff_dim + hidden_dim))
        self.W2 = np.random.uniform(-limit2, limit2, (ff_dim, hidden_dim)).astype(dtype)
        self.b2 = np.zeros(hidden_dim, dtype=dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # GELU activation
        h = x @ self.W1 + self.b1
        h = self._gelu(h)
        return h @ self.W2 + self.b2

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        """Gaussian Error Linear Unit."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class RecurrentCell:
    """
    A single recurrent cell using LSTM-style gating,
    but with attention-based input processing.
    
    This is the atomic unit of OpenMythos recurrent thinking.
    """

    def __init__(self, hidden_dim: int, dtype: np.dtype = np.float32):
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        limit = np.sqrt(6.0 / (hidden_dim + hidden_dim))

        # Input gate
        self.W_i = np.random.uniform(-limit, limit, (hidden_dim, hidden_dim)).astype(dtype)
        self.U_i = np.random.uniform(-limit, limit, (hidden_dim, hidden_dim)).astype(dtype)
        self.b_i = np.zeros(hidden_dim, dtype=dtype)

        # Forget gate
        self.W_f = np.random.uniform(-limit, limit, (hidden_dim, hidden_dim)).astype(dtype)
        self.U_f = np.random.uniform(-limit, limit, (hidden_dim, hidden_dim)).astype(dtype)
        self.b_f = np.ones(hidden_dim, dtype=dtype)  # Initialize forget bias to 1

        # Output gate
        self.W_o = np.random.uniform(-limit, limit, (hidden_dim, hidden_dim)).astype(dtype)
        self.U_o = np.random.uniform(-limit, limit, (hidden_dim, hidden_dim)).astype(dtype)
        self.b_o = np.zeros(hidden_dim, dtype=dtype)

        # Cell candidate
        self.W_c = np.random.uniform(-limit, limit, (hidden_dim, hidden_dim)).astype(dtype)
        self.U_c = np.random.uniform(-limit, limit, (hidden_dim, hidden_dim)).astype(dtype)
        self.b_c = np.zeros(hidden_dim, dtype=dtype)

    def forward(
        self, 
        x: np.ndarray, 
        h_prev: np.ndarray, 
        c_prev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        LSTM-style forward pass.
        
        Args:
            x: Input (batch, hidden_dim)
            h_prev: Previous hidden state (batch, hidden_dim)
            c_prev: Previous cell state (batch, hidden_dim)
        
        Returns:
            h_new: New hidden state
            c_new: New cell state
        """
        # Gates
        i = self._sigmoid(x @ self.W_i + h_prev @ self.U_i + self.b_i)
        f = self._sigmoid(x @ self.W_f + h_prev @ self.U_f + self.b_f)
        o = self._sigmoid(x @ self.W_o + h_prev @ self.U_o + self.b_o)
        
        # Cell candidate
        c_tilde = np.tanh(x @ self.W_c + h_prev @ self.U_c + self.b_c)
        
        # Update cell and hidden
        c_new = f * c_prev + i * c_tilde
        h_new = o * np.tanh(c_new)
        
        return h_new, c_new

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )


class RecurrentBlock:
    """
    A full recurrent block combining:
    - Self-attention over the thinking sequence
    - Recurrent cell for temporal state
    - Feed-forward refinement
    - Layer normalization and residual connections
    
    This block can be applied multiple times (recurrent depth),
    with each pass refining the representation further.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        layer_norm: bool = True,
        residual_alpha: float = 0.5,
        dtype: np.dtype = np.float32,
    ):
        self.hidden_dim = hidden_dim
        self.residual_alpha = residual_alpha
        self.dtype = dtype

        # Sub-components
        self.self_attention = CausalSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            dtype=dtype,
        )
        self.recurrent_cell = RecurrentCell(
            hidden_dim=hidden_dim,
            dtype=dtype,
        )
        self.feed_forward = FeedForward(
            hidden_dim=hidden_dim,
            dtype=dtype,
        )
        
        # Layer norms
        self.ln_attn = LayerNorm(hidden_dim, dtype=dtype) if layer_norm else None
        self.ln_recurrent = LayerNorm(hidden_dim, dtype=dtype) if layer_norm else None
        self.ln_ff = LayerNorm(hidden_dim, dtype=dtype) if layer_norm else None

    def forward(
        self,
        x: np.ndarray,
        hidden_state: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Single forward pass through the recurrent block.
        
        Args:
            x: Input representation (batch, seq_len, hidden_dim)
            hidden_state: Dict with 'h' and 'c' tensors
        
        Returns:
            output: Refined representation
            new_hidden: Updated hidden state
        """
        # 1. Self-attention with residual
        attn_out, _ = self.self_attention.forward(x)
        if self.ln_attn:
            attn_out = self.ln_attn.forward(attn_out)
        x = x + self.residual_alpha * attn_out  # Weighted residual

        # 2. Recurrent processing — compress sequence to single vector,
        #    process through recurrent cell, then broadcast back
        seq_rep = np.mean(x, axis=1, keepdims=False)  # (B, D)
        h = hidden_state["h"]  # (B, D)
        c = hidden_state["c"]  # (B, D)
        
        h_new, c_new = self.recurrent_cell.forward(seq_rep, h, c)
        
        # Broadcast recurrent output back to sequence
        h_broadcast = h_new[:, np.newaxis, :]  # (B, 1, D)
        h_broadcast = np.broadcast_to(h_broadcast, x.shape)  # (B, S, D)
        
        if self.ln_recurrent:
            h_broadcast = self.ln_recurrent.forward(h_broadcast)
        x = x + self.residual_alpha * h_broadcast

        # 3. Feed-forward with residual
        ff_out = self.feed_forward.forward(x)
        if self.ln_ff:
            ff_out = self.ln_ff.forward(ff_out)
        x = x + self.residual_alpha * ff_out

        new_hidden = {"h": h_new, "c": c_new}
        return x, new_hidden


class RecurrentBlockStack:
    """
    Stack of recurrent blocks with convergence-based early stopping.
    
    This is the main processing pipeline. It applies recurrent blocks
    repeatedly until either:
    - The representation converges (change < threshold)
    - Maximum recurrence depth is reached
    
    Special named blocks (mythos, reflection, synthesis, recursion)
    are interleaved with standard recurrent blocks.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        depth: int = 4,
        dropout: float = 0.1,
        max_recurrence: int = 8,
        convergence_threshold: float = 1e-4,
        layer_norm: bool = True,
        residual_alpha: float = 0.5,
        dtype: np.dtype = np.float32,
    ):
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.max_recurrence = max_recurrence
        self.convergence_threshold = convergence_threshold
        self.dtype = dtype

        # Create standard recurrent blocks
        self.blocks = [
            RecurrentBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                layer_norm=layer_norm,
                residual_alpha=residual_alpha,
                dtype=dtype,
            )
            for _ in range(depth)
        ]

        # Named special blocks (added via add_block)
        self.special_blocks: Dict[str, Any] = {}

        # Block execution order
        self.execution_order: List[str] = []

    def add_block(self, name: str, block: Any):
        """Register a named special block."""
        self.special_blocks[name] = block
        self.execution_order.append(name)

    def forward(
        self,
        x: np.ndarray,
        hidden_state: Optional[Dict[str, np.ndarray]] = None,
        max_recurrence: Optional[int] = None,
        convergence_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
        """
        Forward pass through the entire recurrent stack.
        
        Applies blocks in order, then recurses (re-applies the whole
        stack) until convergence or max depth.
        
        Args:
            x: Input representation (batch, seq_len, hidden_dim)
            hidden_state: Initial hidden state for recurrent cells
            max_recurrence: Override max recurrence depth
            convergence_threshold: Override convergence threshold
        
        Returns:
            output: Final refined representation
            final_hidden: Final hidden states
            recurrence_depth: How many recurrent passes were used
        """
        max_rec = max_recurrence or self.max_recurrence
        conv_thresh = convergence_threshold or self.convergence_threshold

        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = {
                "h": np.zeros((x.shape[0], self.hidden_dim), dtype=self.dtype),
                "c": np.zeros((x.shape[0], self.hidden_dim), dtype=self.dtype),
            }

        # Initialize per-block hidden states
        block_hiddens = {}
        for i in range(len(self.blocks)):
            block_hiddens[f"block_{i}"] = {
                "h": hidden_state["h"].copy(),
                "c": hidden_state["c"].copy(),
            }
        for name in self.special_blocks:
            block_hiddens[name] = {
                "h": hidden_state["h"].copy(),
                "c": hidden_state["c"].copy(),
            }

        output = x.copy()
        prev_output_norm = np.inf

        for recurrence_step in range(max_rec):
            # ── Pass through standard blocks ──
            for i, block in enumerate(self.blocks):
                output, new_hidden = block.forward(
                    output, block_hiddens[f"block_{i}"]
                )
                block_hiddens[f"block_{i}"] = new_hidden

            # ── Pass through special blocks ──
            for name in self.execution_order:
                special_block = self.special_blocks[name]
                output, new_hidden = special_block.forward(
                    output, block_hiddens[name]
                )
                block_hiddens[name] = new_hidden

            # ── Check convergence ──
            output_norm = np.linalg.norm(output)
            delta = abs(output_norm - prev_output_norm)
            prev_output_norm = output_norm

            if delta < conv_thresh and recurrence_step > 0:
                logger = __import__('logging').getLogger("openmythos")
                logger.debug(
                    f"Converged at recurrence step {recurrence_step + 1} "
                    f"(delta={delta:.6f})"
                )
                break

        # Aggregate final hidden state
        final_hidden = {
            "h": np.mean(
                [bh["h"] for bh in block_hiddens.values()], axis=0
            ),
            "c": np.mean(
                [bh["c"] for bh in block_hiddens.values()], axis=0
            ),
        }

        recurrence_depth = recurrence_step + 1
        return output, final_hidden, recurrence_depth

    def get_block(self, name: str) -> Optional[Any]:
        """Get a named special block."""
        return self.special_blocks.get(name)