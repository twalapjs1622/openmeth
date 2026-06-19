"""
Synthesis Block — integrates multiple reasoning threads.

This block takes divergent thinking threads and synthesizes
them into a coherent, unified representation. It implements
a "condensation" operation that reduces dimensionality while
preserving essential information.
"""

import numpy as np
from typing import Dict, Tuple
from ..core.recurrent_blocks import LayerNorm, FeedForward


class SynthesisBlock:
    """
    Synthesis block that:
    1. Accumulates reasoning from multiple passes
    2. Identifies common patterns (eigenvector analysis)
    3. Projects to a coherent synthesis
    4. Generates a "summary" representation
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        dtype: np.dtype = np.float32,
    ):
        self.hidden_dim = hidden_dim
        self.dtype = dtype

        # Accumulation buffer
        self.accumulation: list = []
        self.accumulation_capacity = 6

        # Synthesis projections
        limit = np.sqrt(6.0 / (hidden_dim + hidden_dim))
        self.W_synth = np.random.uniform(
            -limit, limit, (hidden_dim, hidden_dim)
        ).astype(dtype)
        self.b_synth = np.zeros(hidden_dim, dtype=dtype)
        
        # Pattern extraction (PCA-like)
        self.W_pattern = np.random.uniform(
            -limit, limit, (hidden_dim, hidden_dim)
        ).astype(dtype)

        # Gating for accumulation vs synthesis
        self.W_accum_gate = np.random.uniform(
            -limit, limit, (hidden_dim, 1)
        ).astype(dtype)
        
        self.ff = FeedForward(hidden_dim=hidden_dim, dtype=dtype)
        self.ln = LayerNorm(hidden_dim, dtype=dtype)

    def forward(
        self,
        x: np.ndarray,
        hidden_state: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Synthesis forward pass.
        """
        h = hidden_state["h"]
        c = hidden_state["c"]

        # Accumulate current representation
        x_pooled = np.mean(x, axis=1, keepdims=True)  # (B, 1, D)
        self._accumulate(x_pooled)

        # Determine if we should synthesize (based on accumulation)
        x_flat = np.mean(x_pooled, axis=1)  # (B, D)
        accum_gate = self._sigmoid(
            x_flat @ self.W_accum_gate
        )  # (B, 1)
        should_synthesize = float(accum_gate.mean()) > 0.5 or \
                            len(self.accumulation) >= 3

        output = x.copy()

        if should_synthesize and len(self.accumulation) >= 2:
            # Extract dominant patterns from accumulated reasoning
            patterns = self._extract_patterns()
            
            # Apply synthesis projection
            synth_input = np.mean(x, axis=1)  # (B, D)
            synthesized = np.tanh(
                synth_input @ self.W_synth + self.b_synth
            )
            
            # Blend patterns with synthesized representation
            blended = 0.6 * synthesized + 0.4 * patterns
            
            # Broadcast to sequence
            blended_seq = blended[:, np.newaxis, :]
            blended_seq = np.broadcast_to(blended_seq, x.shape).copy()
            
            # Apply with gating - accum_gate is (B, 1), need (B, 1, 1) for broadcast
            gate = accum_gate[:, :, np.newaxis]  # (B, 1, 1)
            gate = np.broadcast_to(gate, x.shape)
            output = (1 - gate) * x + gate * blended_seq

        # Feed-forward refinement
        ff_out = self.ff.forward(output)
        if self.ln:
            ff_out = self.ln.forward(ff_out)
        output = output + 0.3 * ff_out

        # Update hidden state
        new_h = np.mean(output, axis=1)
        new_c = c * 0.85 + new_h * 0.15

        return output, {"h": new_h, "c": new_c}

    def _accumulate(self, representation: np.ndarray):
        """Add to accumulation buffer."""
        self.accumulation.append(representation.copy())
        if len(self.accumulation) > self.accumulation_capacity:
            self.accumulation.pop(0)

    def _extract_patterns(self) -> np.ndarray:
        """
        Extract dominant patterns from accumulated reasoning
        using approximate PCA (power iteration).
        """
        # Stack accumulations: (B, N, D) where N = len(accumulation)
        stacked = np.concatenate(self.accumulation, axis=1)  # (B, N, D)
        batch_size = stacked.shape[0]
        n_accum = stacked.shape[1]
        
        # Compute mean and center
        mean = np.mean(stacked, axis=1, keepdims=True)  # (B, 1, D)
        centered = stacked - mean  # (B, N, D)
        
        # Pool across accumulation dimension to get (B, D)
        pooled = np.mean(centered, axis=1)  # (B, D)
        
        # Simple pattern extraction via the pattern weight matrix
        patterns = pooled @ self.W_pattern  # (B, D)

        return patterns

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )