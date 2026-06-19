"""
Tree-of-Thought Reasoning — branching exploration of reasoning paths.

Explores multiple reasoning branches simultaneously and selects
the most promising paths for further exploration.
"""

import numpy as np
from typing import List, Tuple


class ThoughtBranch:
    """A single branch in the tree of thought."""

    def __init__(self, dim: int, dtype: np.dtype = np.float32):
        self.dim = dim
        self.dtype = dtype
        limit = np.sqrt(6.0 / (dim + dim))
        self.W_branch = np.random.uniform(-limit, limit, (dim, dim)).astype(dtype)
        self.b_branch = np.zeros(dim, dtype=dtype)
        self.score = 0.0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Process input through this branch."""
        return np.tanh(x @ self.W_branch + self.b_branch)

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the quality of this branch's output."""
        output = self.forward(x)
        # Score based on coherence (low variance) and information (high magnitude)
        self.score = float(np.mean(np.abs(output))) / (1.0 + float(np.var(output)))
        return self.score


class TreeOfThought:
    """
    Tree-of-thought reasoning with branching and pruning.
    
    At each level:
    1. Generate N branches from current state
    2. Evaluate each branch
    3. Keep top-K branches
    4. Expand those branches further
    """

    def __init__(
        self,
        branching: int = 3,
        max_depth: int = 3,
        keep_top_k: int = 2,
        hidden_dim: int = 768,
        dtype: np.dtype = np.float32,
    ):
        self.branching = branching
        self.max_depth = max_depth
        self.keep_top_k = min(keep_top_k, branching)
        self.hidden_dim = hidden_dim
        self.dtype = dtype

        # Create branches for each depth level
        self.branch_sets = [
            [ThoughtBranch(dim=hidden_dim, dtype=dtype) for _ in range(branching)]
            for _ in range(max_depth)
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Tree-of-thought forward pass.
        
        Args:
            x: Input (batch, seq_len, dim)
        
        Returns:
            Best combined output (batch, seq_len, dim)
        """
        x_pooled = np.mean(x, axis=1)  # (B, D)
        
        current_states = [x_pooled]
        
        for depth in range(self.max_depth):
            branches = self.branch_sets[depth]
            candidates = []
            
            for state in current_states:
                for branch in branches:
                    output = branch.forward(state)
                    score = branch.evaluate(state)
                    candidates.append((output, score))
            
            # Select top-K candidates
            candidates.sort(key=lambda c: c[1], reverse=True)
            current_states = [c[0] for c in candidates[:self.keep_top_k]]
        
        # Combine surviving branches
        combined = np.mean(np.stack(current_states), axis=0)  # (B, D)
        
        # Broadcast back to sequence
        combined_seq = combined[:, np.newaxis, :]  # (B, 1, D)
        combined_seq = np.broadcast_to(combined_seq, x.shape).copy()
        
        return x + 0.2 * combined_seq