"""
Chain-of-Thought Reasoning — sequential step-by-step reasoning.

Each step is processed independently and then connected
via attention, creating a chain of refined reasoning steps.
"""

import numpy as np
from typing import Dict, Optional


class ChainStep:
    """A single step in the chain of thought."""

    def __init__(self, step_dim: int, dtype: np.dtype = np.float32):
        self.step_dim = step_dim
        self.dtype = dtype
        
        limit = np.sqrt(6.0 / (step_dim + step_dim))
        self.W_step = np.random.uniform(-limit, limit, (step_dim, step_dim)).astype(dtype)
        self.b_step = np.zeros(step_dim, dtype=dtype)

    def forward(self, x: np.ndarray, prev_step: Optional[np.ndarray] = None) -> np.ndarray:
        """Process one chain step."""
        h = np.tanh(x @ self.W_step + self.b_step)
        if prev_step is not None:
            h = h + 0.3 * prev_step  # Residual from previous step
        return h


class ChainOfThought:
    """
    Implements chain-of-thought reasoning as a sequential
    processing pipeline over numpy arrays.
    """

    def __init__(self, depth: int = 5, hidden_dim: int = 768, dtype: np.dtype = np.float32):
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.steps = [ChainStep(step_dim=hidden_dim, dtype=dtype) for _ in range(depth)]

        # Inter-step attention weights
        limit = np.sqrt(6.0 / (hidden_dim + 1))
        self.W_step_gate = np.random.uniform(-limit, limit, (hidden_dim, 1)).astype(dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Process representation through chain of thought.
        
        Args:
            x: Input (batch, seq_len, dim)
        
        Returns:
            Chain-processed output (batch, seq_len, dim)
        """
        output = x.copy()
        prev_step = None

        for step in self.steps:
            x_pooled = np.mean(output, axis=1)  # (B, D)
            step_output = step.forward(x_pooled, prev_step)
            
            # Gate: how much of this step to incorporate
            gate = self._sigmoid(x_pooled @ self.W_step_gate)
            gate_broadcast = gate[:, np.newaxis, :]
            
            step_broadcast = step_output[:, np.newaxis, :]
            step_broadcast = np.broadcast_to(step_broadcast, output.shape)
            
            output = output + gate_broadcast * step_broadcast
            prev_step = step_output

        return output

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))