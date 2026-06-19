"""
Reflection Block — self-evaluative reasoning component.

This block enables the thinking process to reflect on its own
reasoning, detect inconsistencies, and propose corrections.
It implements a "mirror" mechanism where the representation
is compared against a delayed copy.
"""

import numpy as np
from typing import Dict, Tuple
from ..core.attention import CrossAttention
from ..core.recurrent_blocks import LayerNorm, FeedForward


class ReflectionBlock:
    """
    A block that reflects on its own processing by:
    1. Storing a delayed copy of the input (memory)
    2. Cross-attending current input with the memory
    3. Detecting contradictions via distance computation
    4. Generating correction signals
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        dtype: np.dtype = np.float32,
    ):
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        
        # Memory buffer for delayed copy
        self.memory_buffer: list = []
        self.memory_capacity = 4  # Keep last N representations

        # Cross-attention for comparing current vs memory
        self.cross_attention = CrossAttention(
            hidden_dim=hidden_dim,
            num_heads=8,
            dtype=dtype,
        )
        
        # Contradiction detection projection
        limit = np.sqrt(6.0 / (hidden_dim * 2 + hidden_dim))
        self.W_contradict = np.random.uniform(
            -limit, limit, (hidden_dim * 2, hidden_dim)
        ).astype(dtype)
        self.b_contradict = np.zeros(hidden_dim, dtype=dtype)

        # Correction gate
        self.W_gate = np.random.uniform(
            -limit, limit, (hidden_dim, hidden_dim)
        ).astype(dtype)
        self.b_gate = np.zeros(hidden_dim, dtype=dtype)

        # Feed-forward for refinement
        self.ff = FeedForward(hidden_dim=hidden_dim, dtype=dtype)
        self.ln = LayerNorm(hidden_dim, dtype=dtype)

    def forward(
        self,
        x: np.ndarray,
        hidden_state: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Reflective forward pass.
        
        Args:
            x: Current thinking representation (batch, seq_len, hidden_dim)
            hidden_state: Hidden state dict
        
        Returns:
            Refined representation and updated hidden state
        """
        h = hidden_state["h"]
        c = hidden_state["c"]

        # Store current representation in memory
        x_pooled = np.mean(x, axis=1, keepdims=True)  # (B, 1, D)
        self._update_memory(x_pooled)

        output = x.copy()

        # Cross-attend with memory if available
        if self.memory_buffer:
            # Stack memory into context sequence
            memory_seq = np.concatenate(self.memory_buffer, axis=1)  # (B, M, D)
            
            # Cross-attention: current queries, memory provides context
            cross_out, cross_weights = self.cross_attention.forward(
                query_seq=x,
                context_seq=memory_seq,
            )

            # Detect contradictions: compare current with memory
            contradiction = self._detect_contradiction(x, memory_seq)
            
            # Generate correction signal
            correction = self._generate_correction(x, contradiction)
            
            # Apply corrections with gating
            gate = self._sigmoid(
                x_pooled.squeeze(1) @ self.W_gate + self.b_gate
            )
            gate_broadcast = gate[:, np.newaxis, :]  # (B, 1, D)
            gate_broadcast = np.broadcast_to(gate_broadcast, x.shape)
            
            output = output + gate_broadcast * (cross_out + correction)

        # Feed-forward refinement
        ff_out = self.ff.forward(output)
        if self.ln:
            ff_out = self.ln.forward(ff_out)
        output = output + 0.3 * ff_out

        # Update hidden state
        new_h = np.mean(output, axis=1)
        new_c = c * 0.8 + new_h * 0.2

        return output, {"h": new_h, "c": new_c}

    def _update_memory(self, representation: np.ndarray):
        """Update the memory buffer with FIFO strategy."""
        self.memory_buffer.append(representation.copy())
        if len(self.memory_buffer) > self.memory_capacity:
            self.memory_buffer.pop(0)

    def _detect_contradiction(
        self, current: np.ndarray, memory: np.ndarray
    ) -> np.ndarray:
        """
        Detect contradictions between current thinking and memory.
        
        High distance = potential contradiction.
        """
        # Pool to same length
        curr_pooled = np.mean(current, axis=1, keepdims=True)  # (B, 1, D)
        mem_pooled = np.mean(memory, axis=1, keepdims=True)    # (B, 1, D)
        
        # Concatenate for contradiction detection
        combined = np.concatenate(
            [curr_pooled, mem_pooled], axis=-1
        )  # (B, 1, 2D)
        combined = combined.squeeze(1)  # (B, 2D)
        
        # Project to contradiction signal
        contradiction = np.tanh(combined @ self.W_contradict + self.b_contradict)
        
        return contradiction[:, np.newaxis, :]  # (B, 1, D)

    def _generate_correction(
        self, current: np.ndarray, contradiction: np.ndarray
    ) -> np.ndarray:
        """Generate correction signal based on detected contradictions."""
        # Simple correction: subtract contradiction from current
        # This pushes the representation away from contradictions
        correction = -0.1 * contradiction
        
        # Expand to sequence length
        correction = np.broadcast_to(correction, current.shape)
        return correction

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )