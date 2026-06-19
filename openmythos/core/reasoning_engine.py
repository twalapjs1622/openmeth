"""
Reasoning Engine — orchestrates OpenMythos reasoning processes.

Combines chain-of-thought, tree-of-thought, and graph-of-thought
reasoning with the recurrent block stack for deep processing.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass


class TextEncoder:
    """
    Encodes text into numpy tensor representations.
    Uses character-level and statistical features to create
    a fixed-dimension representation suitable for the recurrent blocks.
    """

    def __init__(self, hidden_dim: int = 768, dtype: np.dtype = np.float32):
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        # Character embedding matrix
        self.char_embed = np.random.randn(256, hidden_dim // 4).astype(dtype) * 0.02
        # Feature projection
        limit = np.sqrt(6.0 / (hidden_dim // 4 + hidden_dim))
        self.feature_proj = np.random.uniform(
            -limit, limit, (hidden_dim // 4, hidden_dim)
        ).astype(dtype)
        self.feature_bias = np.zeros(hidden_dim, dtype=dtype)

    def encode(self, text: str, max_len: int = 128) -> np.ndarray:
        """
        Encode text string to numpy array.
        
        Args:
            text: Input text
            max_len: Maximum sequence length
        
        Returns:
            Representation array of shape (1, seq_len, hidden_dim)
        """
        # Character-level encoding
        chars = [ord(c) % 256 for c in text[:max_len]]
        if not chars:
            chars = [0]  # Pad empty
        
        # Embed characters
        embedded = self.char_embed[chars]  # (seq_len, hidden_dim//4)
        
        # Compute statistical features for each position
        seq_len = len(chars)
        
        # Add positional encoding
        positions = np.arange(seq_len, dtype=self.dtype)
        pos_enc = self._positional_encoding(positions, self.hidden_dim // 4)
        embedded = embedded + pos_enc

        # Project to hidden dimension
        projected = embedded @ self.feature_proj + self.feature_bias  # (S, D)

        # Add global context features
        text_features = self._extract_text_features(text)
        projected = projected + text_features[np.newaxis, :]

        return projected[np.newaxis, :, :]  # (1, S, D)

    def _positional_encoding(self, positions: np.ndarray, dim: int) -> np.ndarray:
        """Sinusoidal positional encoding."""
        pe = np.zeros((len(positions), dim), dtype=self.dtype)
        div_term = np.exp(
            np.arange(0, dim, 2, dtype=self.dtype) * 
            -(np.log(10000.0) / dim)
        )
        pe[:, 0::2] = np.sin(positions[:, np.newaxis] * div_term)
        pe[:, 1::2] = np.cos(positions[:, np.newaxis] * div_term)
        return pe

    def _extract_text_features(self, text: str) -> np.ndarray:
        """Extract statistical features from text."""
        features = np.zeros(self.hidden_dim, dtype=self.dtype)
        
        if not text:
            return features

        # Length features
        features[0] = min(len(text) / 1000.0, 1.0)
        
        # Character diversity
        unique_ratio = len(set(text)) / max(len(text), 1)
        features[1] = unique_ratio
        
        # Word count proxy (spaces)
        word_count = text.count(' ') + 1
        features[2] = min(word_count / 100.0, 1.0)
        
        # Reasoning indicators
        reasoning_words = [
            "because", "therefore", "however", "since", "thus",
            "implies", "assume", "suppose", "consider", "observe",
            "notice", "deduce", "infer", "conclude", "prove",
            "step", "first", "then", "next", "finally", "but",
            "although", "while", "if", "then", "else",
        ]
        text_lower = text.lower()
        for i, word in enumerate(reasoning_words):
            if i + 3 < self.hidden_dim:
                features[i + 3] = text_lower.count(word) / max(word_count, 1)

        return features


class ReasoningEngine:
    """
    Central reasoning engine that orchestrates all OpenMythos
    thinking and reasoning processes.
    
    Pipeline:
        1. Encode thinking text -> representation
        2. Analyze for archetype dominance
        3. Process through recurrent blocks
        4. Route through reasoning paths (chain/tree/graph)
        5. Synthesize final output
    """

    def __init__(
        self,
        recurrent_stack: Any,
        chain: Any,
        tree: Any,
        graph: Any,
        config: Dict,
    ):
        self.recurrent_stack = recurrent_stack
        self.chain = chain
        self.tree = tree
        self.graph = graph
        self.config = config

        hidden_dim = config.get("recurrent", {}).get("hidden_dim", 768)
        dtype_str = config.get("numpy", {}).get("dtype", "float32")
        self.dtype = getattr(np, dtype_str)
        
        self.encoder = TextEncoder(hidden_dim=hidden_dim, dtype=self.dtype)

    def encode_to_representation(self, text: str) -> np.ndarray:
        """Encode text to a numpy representation for the recurrent blocks."""
        return self.encoder.encode(text)

    def analyze_prompt_archetype(self, prompt: str) -> str:
        """
        Analyze a prompt to determine which OpenMythos archetype
        should dominate the thinking process.
        
        Archetypes:
            - explorer: Creative, divergent thinking
            - architect: Structural, pattern-based thinking
            - critic: Evaluative, critical thinking
            - oracle: Intuitive, synthesis-based thinking
        """
        prompt_lower = prompt.lower()
        
        # Score each archetype based on keyword signals
        scores = {
            "explorer": 0.0,
            "architect": 0.0,
            "critic": 0.0,
            "oracle": 0.0,
        }

        # Explorer signals
        explorer_keywords = [
            "creative", "imagine", "brainstorm", "explore", "possibility",
            "what if", "innovative", "generate", "ideas", "alternative",
            "novel", "diverse", "wonder", "discover",
        ]
        for kw in explorer_keywords:
            scores["explorer"] += prompt_lower.count(kw)

        # Architect signals
        architect_keywords = [
            "design", "structure", "system", "architecture", "organize",
            "pattern", "framework", "model", "plan", "schema",
            "blueprint", "hierarchy", "modular", "compose",
        ]
        for kw in architect_keywords:
            scores["architect"] += prompt_lower.count(kw)

        # Critic signals
        critic_keywords = [
            "evaluate", "critique", "review", "analyze", "compare",
            "assess", "judge", "verify", "validate", "correct",
            "debug", "fix", "improve", "optimize", "refine",
        ]
        for kw in critic_keywords:
            scores["critic"] += prompt_lower.count(kw)

        # Oracle signals
        oracle_keywords = [
            "synthesize", "integrate", "understand", "insight", "meaning",
            "essence", "intuition", "wisdom", "holistic", "connect",
            "grasp", "comprehend", "perceive", "envision", "intuit",
        ]
        for kw in oracle_keywords:
            scores["oracle"] += prompt_lower.count(kw)

        # If no clear signal, default based on question type
        if sum(scores.values()) == 0:
            if "?" in prompt:
                if any(w in prompt_lower for w in ["how", "why", "explain"]):
                    scores["architect"] = 2.0
                    scores["oracle"] = 1.0
                elif any(w in prompt_lower for w in ["what if", "could", "might"]):
                    scores["explorer"] = 2.0
                else:
                    scores["critic"] = 1.5
                    scores["architect"] = 1.0
            else:
                # Default: balanced
                scores = {k: 1.0 for k in scores}

        # Return the dominant archetype
        return max(scores, key=scores.get)

    def process_thinking_chunk(
        self,
        representation: np.ndarray,
        hidden_state: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict:
        """
        Process a chunk of thinking through the recurrent pipeline.
        
        Args:
            representation: Encoded thinking chunk (1, S, D)
            hidden_state: Recurrent hidden state from previous chunk
        
        Returns:
            Dict with processed output, new hidden state, and metadata
        """
        # Run through recurrent block stack
        output, new_hidden, depth_used = self.recurrent_stack.forward(
            representation,
            hidden_state=hidden_state,
        )

        # Extract modification signal
        modification_signal = self.decode_modification_signal(output)

        return {
            "output": output,
            "hidden_state": new_hidden,
            "depth_used": depth_used,
            "modification_signal": modification_signal,
            "archetype_weights": self.extract_archetype_weights(output),
        }

    def decode_modification_signal(self, output: np.ndarray) -> Dict:
        """
        Decode the recurrent output into a modification signal
        that can guide the model's subsequent thinking.
        
        The output representation encodes:
        - Confidence in current reasoning direction
        - Detected contradictions or gaps
        - Suggested direction shifts
        - Recursion depth recommendations
        """
        # Global statistics
        mean_out = np.mean(output, axis=(0, 1))  # (D,)
        std_out = np.std(output, axis=(0, 1))
        
        # Confidence signal (inverse of variance — low variance = high confidence)
        confidence = float(1.0 / (1.0 + np.mean(std_out)))
        
        # Contradiction signal (high magnitude swings)
        diff = np.diff(output, axis=1)
        contradiction = float(np.mean(np.abs(diff)))
        
        # Direction signal (dominant eigenvalue approximation)
        cov = np.cov(mean_out.reshape(-1, 1).T) if len(mean_out) > 1 else np.eye(1)
        try:
            eigenvalues = np.linalg.eigvalsh(cov)
            direction_strength = float(np.max(eigenvalues) / (np.sum(np.abs(eigenvalues)) + 1e-8))
        except np.linalg.LinAlgError:
            direction_strength = 0.5

        return {
            "confidence": confidence,
            "contradiction": contradiction,
            "direction_strength": direction_strength,
            "recommend_recursion": contradiction > 0.5 and confidence < 0.7,
            "recommend_exploration": confidence < 0.3,
            "recommend_synthesis": confidence > 0.7 and contradiction < 0.3,
        }

    def extract_archetype_weights(self, output: np.ndarray) -> Dict[str, float]:
        """
        Extract archetype activation weights from the recurrent output.
        
        Maps regions of the hidden dimension to each archetype.
        """
        mean_out = np.mean(output, axis=(0, 1))  # (D,)
        dim = len(mean_out)
        quarter = dim // 4
        
        # Each quarter of the hidden dim maps to an archetype
        raw_weights = {
            "explorer": float(np.mean(np.abs(mean_out[:quarter]))),
            "architect": float(np.mean(np.abs(mean_out[quarter:2*quarter]))),
            "critic": float(np.mean(np.abs(mean_out[2*quarter:3*quarter]))),
            "oracle": float(np.mean(np.abs(mean_out[3*quarter:]))),
        }

        # Normalize
        total = sum(raw_weights.values()) + 1e-8
        normalized = {k: v / total for k, v in raw_weights.items()}
        
        return normalized

    def synthesize(self, thinking_text: str, session: Dict) -> Dict:
        """
        Final synthesis pass after thinking block completes.
        
        Combines chain, tree, and graph reasoning outputs
        with the recurrent processing results.
        """
        representation = self.encode_to_representation(thinking_text)
        
        # Process through reasoning paths
        chain_output = self.chain.forward(representation)
        tree_output = self.tree.forward(representation)
        graph_output = self.graph.forward(representation)

        # Combine reasoning paths
        combined = self._fuse_reasoning_paths(
            chain_output, tree_output, graph_output
        )

        # Final recurrent pass
        hidden = session.get("recurrent_hidden")
        final_output, final_hidden, final_depth = self.recurrent_stack.forward(
            combined,
            hidden_state=hidden,
        )

        # Extract final signals
        modification = self.decode_modification_signal(final_output)
        archetype_weights = self.extract_archetype_weights(final_output)

        return {
            "final_output": final_output,
            "final_hidden": final_hidden,
            "final_depth": final_depth,
            "modification": modification,
            "archetype_weights": archetype_weights,
            "chain_signal": self._summarize_path(chain_output),
            "tree_signal": self._summarize_path(tree_output),
            "graph_signal": self._summarize_path(graph_output),
        }

    def _fuse_reasoning_paths(
        self,
        chain: np.ndarray,
        tree: np.ndarray,
        graph: np.ndarray,
    ) -> np.ndarray:
        """Fuse outputs from different reasoning paths."""
        # Ensure same shape via mean pooling to (1, 1, D)
        chain_pooled = np.mean(chain, axis=1, keepdims=True)
        tree_pooled = np.mean(tree, axis=1, keepdims=True)
        graph_pooled = np.mean(graph, axis=1, keepdims=True)

        # Weighted combination
        fusion = 0.4 * chain_pooled + 0.3 * tree_pooled + 0.3 * graph_pooled
        return fusion

    def _summarize_path(self, output: np.ndarray) -> Dict:
        """Summarize a reasoning path output."""
        return {
            "magnitude": float(np.linalg.norm(output)),
            "mean": float(np.mean(output)),
            "variance": float(np.var(output)),
            "peak_activation": float(np.max(np.abs(output))),
        }