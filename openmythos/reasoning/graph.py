"""
Graph-of-Thought Reasoning — interconnected reasoning nodes.

Unlike chain (sequential) or tree (hierarchical), graph reasoning
allows arbitrary connections between thought nodes, enabling
non-linear reasoning patterns.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ThoughtNode:
    """A node in the graph of thought."""
    id: int
    representation: np.ndarray
    connections: List[int]  # Connected node IDs
    score: float = 0.0


class GraphOfThought:
    """
    Graph-of-thought reasoning where nodes can have arbitrary
    connections, enabling cyclic and non-linear reasoning.
    """

    def __init__(
        self,
        max_nodes: int = 32,
        hidden_dim: int = 768,
        dtype: np.dtype = np.float32,
    ):
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        self.dtype = dtype

        # Node processing weights
        limit = np.sqrt(6.0 / (hidden_dim + hidden_dim))
        self.W_node = np.random.uniform(-limit, limit, (hidden_dim, hidden_dim)).astype(dtype)
        self.b_node = np.zeros(hidden_dim, dtype=dtype)

        # Edge weights for message passing
        self.W_message = np.random.uniform(-limit, limit, (hidden_dim, hidden_dim)).astype(dtype)

        # Node scoring
        self.W_score = np.random.uniform(-limit, limit, (hidden_dim, 1)).astype(dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Graph-of-thought forward pass.
        
        Creates a graph from the input, performs message passing,
        and returns the aggregated result.
        """
        x_pooled = np.mean(x, axis=1)  # (B, D)
        batch_size = x_pooled.shape[0]

        # Initialize graph nodes
        # Split the representation into multiple nodes
        num_nodes = min(8, self.max_nodes)
        chunk_size = self.hidden_dim // num_nodes
        
        nodes = []
        for i in range(num_nodes):
            start = i * chunk_size
            end = start + chunk_size
            node_rep = np.zeros((batch_size, self.hidden_dim), dtype=self.dtype)
            node_rep[:, start:end] = x_pooled[:, start:end]
            
            # Connect to nearby nodes (ring + skip connections)
            connections = [(i + 1) % num_nodes, (i - 1) % num_nodes]
            if i + 2 < num_nodes:
                connections.append(i + 2)
            
            nodes.append(ThoughtNode(
                id=i,
                representation=node_rep,
                connections=connections,
            ))

        # Message passing (2 rounds)
        for round_num in range(2):
            new_reps = []
            
            for node in nodes:
                # Aggregate messages from connected nodes
                messages = np.zeros_like(node.representation)
                for conn_id in node.connections:
                    if conn_id < len(nodes):
                        msg = nodes[conn_id].representation @ self.W_message
                        messages += msg
                
                # Update node representation
                new_rep = np.tanh(
                    node.representation @ self.W_node + 
                    messages / max(len(node.connections), 1) +
                    self.b_node
                )
                new_reps.append(new_rep)
            
            # Apply updates
            for i, new_rep in enumerate(new_reps):
                nodes[i].representation = new_rep

        # Score and aggregate nodes
        scores = []
        for node in nodes:
            score = float(np.mean(node.representation @ self.W_score))
            node.score = score
            scores.append(score)

        # Weighted aggregation
        scores_arr = np.array(scores, dtype=self.dtype)
        weights = self._softmax(scores_arr)
        
        aggregated = np.zeros((batch_size, self.hidden_dim), dtype=self.dtype)
        for i, node in enumerate(nodes):
            aggregated += weights[i] * node.representation

        # Broadcast back to sequence
        aggregated_seq = aggregated[:, np.newaxis, :]
        aggregated_seq = np.broadcast_to(aggregated_seq, x.shape).copy()

        return x + 0.2 * aggregated_seq

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        return exp_x / (np.sum(exp_x) + 1e-8)