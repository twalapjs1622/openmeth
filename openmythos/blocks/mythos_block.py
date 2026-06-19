"""
Mythos Block — the OpenMythos archetypal reasoning block.

This block implements the four archetypal thinking modes
(Explorer, Architect, Critic, Oracle) as parallel processing
paths that are fused through learned attention weights.

Each archetype has its own processing style:
  - Explorer: High temperature, expansion-biased recurrence
  - Architect: Medium temperature, structure-biased recurrence
  - Critic: Low temperature, compression-biased recurrence
  - Oracle: Balanced temperature, integration-biased recurrence

All implemented in pure numpy.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass


class ArchetypePath:
    """
    A single archetypal processing path.
    Each path has its own recurrent weights and temperature.
    """

    def __init__(
        self,
        name: str,
        hidden_dim: int,
        temperature: float,
        bias: str,  # "expansion", "structuring", "compression", "integration"
        weight: float,
        dtype: np.dtype = np.float32,
    ):
        self.name = name
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.bias = bias
        self.weight = weight
        self.dtype = dtype

        # Recurrent weights for this archetype
        limit = np.sqrt(6.0 / (hidden_dim + hidden_dim))
        self.W_in = np.random.uniform(-limit, limit, (hidden_dim, hidden_dim)).astype(dtype)
        self.W_rec = np.random.uniform(-limit, limit, (hidden_dim, hidden_dim)).astype(dtype)
        self.b_in = np.zeros(hidden_dim, dtype=dtype)
        
        # Bias-specific modification
        self.bias_matrix = self._create_bias_matrix(hidden_dim, bias, dtype)

    def _create_bias_matrix(self, dim: int, bias: str, dtype: np.dtype) -> np.ndarray:
        """
        Create a bias matrix that influences recurrence direction.
        
        - expansion: encourages divergent activation patterns
        - structuring: encourages block-diagonal patterns
        - compression: encourages rank-deficient (compressible) patterns
        - integration: encourages smooth, correlated patterns
        """
        if bias == "expansion":
            # Off-diagonal emphasis → spread activation
            mat = np.random.randn(dim, dim).astype(dtype) * 0.1
            np.fill_diagonal(mat, 0.5)
        elif bias == "structuring":
            # Block-diagonal emphasis → organized activation
            mat = np.zeros((dim, dim), dtype=dtype)
            block_size = dim // 4
            for i in range(4):
                start = i * block_size
                end = (i + 1) * block_size
                mat[start:end, start:end] = np.eye(block_size) * 0.8
                mat[start:end, start:end] += np.random.randn(block_size, block_size) * 0.1
        elif bias == "compression":
            # Low-rank structure → compressible activation
            u = np.random.randn(dim, dim // 4).astype(dtype) * 0.1
            v = np.random.randn(dim // 4, dim).astype(dtype) * 0.1
            mat = u @ v
        elif bias == "integration":
            # Smooth correlation → integrated activation
            mat = np.eye(dim, dtype=dtype) * 0.5
            for offset in [-1, 1]:
                diag = np.diag(np.ones(dim - abs(offset)), offset)
                mat += diag * 0.3
        else:
            mat = np.eye(dim, dtype=dtype) * 0.5

        return mat

    def forward(
        self,
        x: np.ndarray,
        h: np.ndarray,
    ) -> np.ndarray:
        """
        Process input through this archetype path.
        
        Args:
            x: Input (batch, hidden_dim)
            h: Previous hidden state (batch, hidden_dim)
        
        Returns:
            New hidden state (batch, hidden_dim)
        """
        # Input transformation with temperature scaling
        input_proj = (x @ self.W_in + self.b_in) / self.temperature
        
        # Recurrent transformation
        rec_proj = h @ self.W_rec
        
        # Apply bias matrix influence
        bias_influence = h @ self.bias_matrix
        
        # Combine with gating
        gate = self._sigmoid(input_proj)
        new_h = gate * np.tanh(input_proj + rec_proj + bias_influence * 0.1)
        
        return new_h

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )


class MythosBlock:
    """
    The OpenMythos archetypal reasoning block.
    
    Processes thinking through four parallel archetype paths,
    then fuses the results using attention-weighted combination.
    The dominant archetype shifts during recurrence, implementing
    the "mythos rotation" that ensures balanced thinking.
    """

    def __init__(
        self,
        config: Dict,
        hidden_dim: int = 768,
        dtype: np.dtype = np.float32,
    ):
        self.config = config
        self.hidden_dim = hidden_dim
        self.dtype = dtype

        # Create archetype paths
        archetypes_cfg = config.get("archetypes", {})
        self.paths: Dict[str, ArchetypePath] = {}
        
        default_archetypes = {
            "explorer": {"weight": 0.25, "temperature": 1.2, "recurrence_bias": "expansion"},
            "architect": {"weight": 0.25, "temperature": 0.8, "recurrence_bias": "structuring"},
            "critic": {"weight": 0.25, "temperature": 0.6, "recurrence_bias": "compression"},
            "oracle": {"weight": 0.25, "temperature": 1.0, "recurrence_bias": "integration"},
        }
        
        for name, acfg in default_archetypes.items():
            override = archetypes_cfg.get(name, {})
            self.paths[name] = ArchetypePath(
                name=name,
                hidden_dim=hidden_dim,
                temperature=override.get("temperature", acfg["temperature"]),
                bias=override.get("recurrence_bias", acfg["recurrence_bias"]),
                weight=override.get("weight", acfg["weight"]),
                dtype=dtype,
            )

        self.fusion_method = config.get("fusion_method", "weighted_attention")
        self.rotate_on_recurrence = config.get("rotate_on_recurrence", True)
        self.min_active = config.get("min_active_archetypes", 2)

        # Fusion attention weights
        limit = np.sqrt(6.0 / (hidden_dim + len(self.paths)))
        self.fusion_query = np.random.uniform(
            -limit, limit, (hidden_dim, hidden_dim)
        ).astype(dtype)
        self.fusion_key = np.random.uniform(
            -limit, limit, (hidden_dim, hidden_dim)
        ).astype(dtype)
        self.fusion_value = np.random.uniform(
            -limit, limit, (hidden_dim, hidden_dim)
        ).astype(dtype)

        # Output projection
        self.output_proj = np.random.uniform(
            -limit, limit, (hidden_dim, hidden_dim)
        ).astype(dtype)
        self.output_bias = np.zeros(hidden_dim, dtype=dtype)

    def forward(
        self,
        x: np.ndarray,
        hidden_state: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Forward pass through all archetype paths and fusion.
        
        Args:
            x: Input representation (batch, seq_len, hidden_dim)
            hidden_state: Dict with 'h' and 'c'
        
        Returns:
            output: Fused output (batch, seq_len, hidden_dim)
            new_hidden: Updated hidden state
        """
        batch_size, seq_len, _ = x.shape
        h = hidden_state["h"]  # (batch, hidden_dim)
        c = hidden_state["c"]

        # Compress sequence to vector for archetype processing
        x_pooled = np.mean(x, axis=1)  # (batch, hidden_dim)

        # Process through each archetype path
        archetype_outputs = {}
        for name, path in self.paths.items():
            archetype_outputs[name] = path.forward(x_pooled, h)

        # Fuse archetype outputs
        if self.fusion_method == "weighted_attention":
            fused = self._fuse_attention(archetype_outputs, x_pooled)
        elif self.fusion_method == "gating":
            fused = self._fuse_gating(archetype_outputs, x_pooled)
        else:  # voting
            fused = self._fuse_voting(archetype_outputs)

        # Update hidden state (LSTM-style with fused output)
        new_h = fused
        new_c = c * 0.9 + new_h * 0.1  # Slow cell update

        # Broadcast fused vector back to sequence shape
        fused_seq = new_h[:, np.newaxis, :]  # (B, 1, D)
        fused_seq = np.broadcast_to(fused_seq, x.shape).copy()

        # Residual connection with input
        output = x + 0.3 * fused_seq

        new_hidden = {"h": new_h, "c": new_c}
        return output, new_hidden

    def _fuse_attention(
        self,
        archetype_outputs: Dict[str, np.ndarray],
        query: np.ndarray,
    ) -> np.ndarray:
        """
        Attention-weighted fusion of archetype outputs.
        
        Each archetype output acts as a key/value,
        the pooled input acts as the query.
        """
        # Stack archetype outputs: (batch, num_archetypes, hidden_dim)
        names = list(archetype_outputs.keys())
        stacked = np.stack(
            [archetype_outputs[n] for n in names], axis=1
        )

        # Compute attention scores
        Q = query @ self.fusion_query  # (B, D)
        K = stacked @ self.fusion_key  # (B, A, D)
        
        # Attention weights
        scores = np.sum(Q[:, np.newaxis, :] * K, axis=-1)  # (B, A)
        scores = scores / np.sqrt(self.hidden_dim)
        
        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # Apply archetype weights
        archetype_weights = np.array(
            [self.paths[n].weight for n in names], dtype=self.dtype
        )
        combined_weights = attn_weights * archetype_weights[np.newaxis, :]
        combined_weights = combined_weights / (
            np.sum(combined_weights, axis=-1, keepdims=True) + 1e-8
        )

        # Weighted sum of values
        V = stacked @ self.fusion_value  # (B, A, D)
        fused = np.sum(
            combined_weights[:, :, np.newaxis] * V, axis=1
        )  # (B, D)

        # Output projection
        fused = fused @ self.output_proj + self.output_bias

        return fused

    def _fuse_gating(
        self,
        archetype_outputs: Dict[str, np.ndarray],
        context: np.ndarray,
    ) -> np.ndarray:
        """Gating-based fusion with learned gates."""
        stacked = np.stack(list(archetype_outputs.values()), axis=1)
        
        # Simple gating: context determines which archetypes to emphasize
        gate_input = context @ self.fusion_query
        gates = self._sigmoid(gate_input)
        
        # Broadcast gates
        fused = np.sum(
            gates[:, np.newaxis, np.newaxis] * stacked, axis=1
        )
        
        return fused @ self.output_proj + self.output_bias

    def _fuse_voting(
        self,
        archetype_outputs: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Simple weighted average fusion."""
        total_weight = sum(p.weight for p in self.paths.values())
        fused = sum(
            archetype_outputs[name] * self.paths[name].weight
            for name in archetype_outputs
        ) / (total_weight + 1e-8)
        return fused

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )

    def rotate_weights(self, step: int):
        """
        Rotate archetype weights to ensure all archetypes
        contribute during deep recurrence.
        """
        if not self.rotate_on_recurrence:
            return

        names = list(self.paths.keys())
        n = len(names)
        rotation = step % n
        
        # Shift weights cyclically
        base_weight = 0.4
        other_weight = (1.0 - base_weight) / (n - 1)
        
        for i, name in enumerate(names):
            shifted_idx = (i + rotation) % n
            if shifted_idx == 0:
                self.paths[name].weight = base_weight
            else:
                self.paths[name].weight = other_weight