"""
Token-level operations for OpenMythos.

Utilities for working with text tokens at the boundary
between the AI model's token stream and numpy processing.
"""

import numpy as np
from typing import List, Dict, Optional


def tokens_to_bag_representation(
    tokens: List[str],
    vocab: Optional[Dict[str, int]] = None,
    dim: int = 768,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Convert a list of tokens to a bag-of-tokens representation.
    Uses hash-based feature mapping for vocabulary-free operation.
    """
    representation = np.zeros((1, dim), dtype=dtype)
    
    for token in tokens:
        # Hash-based feature mapping
        features = _hash_features(token, dim, dtype)
        representation += features
    
    # Normalize
    if len(tokens) > 0:
        representation /= len(tokens)
    
    return representation


def _hash_features(token: str, dim: int, dtype: np.dtype) -> np.ndarray:
    """
    Map a token to a dense feature vector using hash functions.
    This avoids needing a fixed vocabulary.
    """
    features = np.zeros(dim, dtype=dtype)
    
    # Use multiple hash functions for better distribution
    for seed in range(4):
        h = hash(f"{token}_{seed}") % dim
        sign = 1 if hash(f"{token}_sign_{seed}") % 2 == 0 else -1
        features[h] += sign * 0.25
    
    # Add character-level features
    for i, c in enumerate(token[:8]):  # First 8 chars
        idx = (ord(c) * (i + 1)) % dim
        features[idx] += 0.1
    
    return features


def extract_reasoning_segments(text: str) -> List[Dict]:
    """
    Extract structured reasoning segments from thinking text.
    
    Detects patterns like:
    - "Step 1:", "First,", "Then,"
    - "Because...", "Therefore..."
    - "However...", "On the other hand..."
    """
    import re
    
    segments = []
    
    # Reasoning step markers
    step_patterns = [
        (r"step\s*\d+[:.]", "step"),
        (r"first[,:]", "step"),
        (r"second[,:]", "step"),
        (r"then[,:]", "step"),
        (r"next[,:]", "step"),
        (r"finally[,:]", "step"),
        (r"because\b", "causal"),
        (r"therefore\b", "causal"),
        (r"thus\b", "causal"),
        (r"however\b", "contrast"),
        (r"but\b", "contrast"),
        (r"although\b", "contrast"),
        (r"on the other hand", "contrast"),
        (r"suppose\b", "hypothesis"),
        (r"assume\b", "hypothesis"),
        (r"let's say", "hypothesis"),
        (r"if\b", "conditional"),
        (r"then\b", "conditional"),
    ]
    
    for pattern, label in step_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            segments.append({
                "type": label,
                "text": match.group(),
                "position": match.start(),
                "confidence": 0.8,
            })
    
    # Sort by position
    segments.sort(key=lambda s: s["position"])
    
    return segments


def estimate_thinking_quality(text: str) -> Dict[str, float]:
    """
    Estimate the quality and depth of thinking in text.
    Returns metrics that can guide the recurrent processing.
    """
    if not text:
        return {"depth": 0.0, "coherence": 0.0, "diversity": 0.0}
    
    # Depth: presence of layered reasoning
    depth_markers = [
        "because", "therefore", "implies", "since",
        "however", "but", "although", "conversely",
        "in other words", "that is", "specifically",
        "in conclusion", "thus", "hence",
    ]
    text_lower = text.lower()
    depth_score = sum(text_lower.count(m) for m in depth_markers)
    depth_score = min(depth_score / 20.0, 1.0)
    
    # Coherence: logical connectors present
    sentences = text.count('.') + text.count('!') + text.count('?')
    connectors = sum(text_lower.count(c) for c in [
        "and", "but", "or", "so", "then", "because", "however"
    ])
    coherence = min(connectors / max(sentences, 1), 1.0)
    
    # Diversity: vocabulary diversity
    words = text_lower.split()
    if words:
        unique_ratio = len(set(words)) / len(words)
        diversity = min(unique_ratio * 2, 1.0)
    else:
        diversity = 0.0
    
    return {
        "depth": depth_score,
        "coherence": coherence,
        "diversity": diversity,
    }