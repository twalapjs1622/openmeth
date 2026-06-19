"""
Recursion Block — enables meta-recursion in thinking.

This block can recursively apply itself, creating nested
processing loops. It includes explicit depth tracking and
convergence monitoring to prevent infinite recursion.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from ..core.recurrent_blocks import RecurrentCell, LayerNorm, FeedForward


class RecursionBlock:
    """
    A block that can recursively apply itself.
    
    Key features:
    - Tracks recursion depth explicitly
    - Monitors convergence at each level
    - Implements "deepening" — each recursion level can focus
      on a different aspect of the thinking
    - Has a learned "recurse or return" gate
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        max_recurrence: int = 8,
        dtype: np.dtype = np.float32,
    ):
        self.hidden_dim = hidden_dim
        self.max_recurrence = max_recurrence
        self.dtype = dtype

        # Core recurrent cell
        self.cell = RecurrentCell(hidden_dim=hidden_dim, dtype=dtype)
        
        # Recursion gate: should we recurse deeper?
        limit = np.sqrt(6.0 / (hidden_dim + 1))
        self.W_recurse_gate = np.random.uniform(
            -limit, limit, (hidden_dim, 1)
        ).astype(dtype)
        self.b_recurse_gate = np.zeros(1, dtype=dtype)

        # Depth embedding: each recursion level gets a unique embedding
        self.depth_embeddings = np.random.randn(
            max_recurrence, hidden_dim
        ).astype(dtype) * 0.02

        # Focus projection: at each depth, focus on different aspects
        self.W_focus = np.random.uniform(
            -limit * 2, limit * 2, (max_recurrence, hidden_dim, hidden_dim)
        ).astype(dtype) * 0.1

        self.ff = FeedForward(hidden_dim=hidden_dim, dtype=dtype)
        self.ln = LayerNorm(hidden_dim, dtype=dtype)

    def forward(
        self,
        x: np.ndarray,
        hidden_state: Dict[str, np.ndarray],
        current_depth: int = 0,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Recursive forward pass.
        
        At each level:
        1. Apply depth-specific focus
        2. Process through recurrent cell
        3. Evaluate: recurse deeper or return?
        4. If recurse: apply self to output with depth+1
        5. If return: output the refined representation
        """
        h = hidden_state["h"]
        c = hidden_state["c"]

        output = x.copy()

        # Apply depth-specific focus and embedding
        safe_depth = min(current_depth, self.max_recurrence - 1)
        depth_embed = self.depth_embeddings[safe_depth]
        output = output + depth_embed[np.newaxis, np.newaxis, :] * 0.1

        # Focus projection
        focus_weights = self.W_focus[safe_depth]
        x_pooled = np.mean(output, axis=1)  # (B, D)
        focused = x_pooled @ focus_weights  # (B, D)

        # Recurrent cell update
        h_new, c_new = self.cell.forward(focused, h, c)

        # Broadcast recurrent output back to sequence
        h_broadcast = h_new[:, np.newaxis, :]  # (B, 1, D)
        h_broadcast = np.broadcast_to(h_broadcast, output.shape).copy()

        # Residual
        output = output + 0.3 * h_broadcast

        # Feed-forward
        ff_out = self.ff.forward(output)
        if self.ln:
            ff_out = self.ln.forward(ff_out)
        output = output + 0.2 * ff_out

        # Recursion gate evaluation
        should_recurse = self._evaluate_recursion(
            output, current_depth
        )

        if should_recurse and current_depth < self.max_recurrence - 1:
            # Recurse deeper
            deeper_output, deeper_hidden = self.forward(
                output,
                {"h": h_new, "c": c_new},
                current_depth=current_depth + 1,
            )
            # Blend this level with deeper level
            blend = 0.7 * deeper_output + 0.3 * output
            return blend, deeper_hidden

        return output, {"h": h_new, "c": c_new}

    def _evaluate_recursion(
        self, x: np.ndarray, current_depth: int
    ) -> bool:
        """
        Evaluate whether to recurse deeper.
        
        Recurse if:
        - The representation still has high entropy
        - We haven't reached max depth
        - The gate opens
        """
        if current_depth >= self.max_recurrence - 1:
            return False

        # Compute gate signal
        x_pooled = np.mean(x, axis=1)  # (B, D)
        gate_signal = self._sigmoid(
            x_pooled @ self.W_recurse_gate + self.b_recurse_gate
        )
        
        # Also consider variance as entropy proxy
        variance = float(np.mean(np.var(x, axis=1)))
        entropy_signal = min(variance * 10, 1.0)
        
        # Combined decision
        recurse_prob = float(gate_signal.mean()) * 0.7 + entropy_signal * 0.3
        
        # Bias against deep recursion
        depth_penalty = 1.0 - (current_depth / self.max_recurrence) * 0.5
        recurse_prob *= depth_penalty

        return recurse_prob > 0.5

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )