"""
OpenMythos Recurrent Thinking Plugin for OpenCode/Crush
========================================================

FIXED VERSION — Now compatible with OpenCode's shell-based hook system.

The plugin exposes:
  1. `hook_main.sh` — main PreToolUse hook entry point (bash script)
  2. `hook_reasoning.sh` — lightweight reasoning alternative
  3. `OpenMythosPlugin` — the Python engine invoked by the shell hooks

OpenCode/Crush calls hooks by executing shell scripts via an embedded POSIX shell.
This module provides the Python engine that those scripts call.
"""

import sys
import os
import json
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path

# Ensure we can find our modules
PLUGIN_DIR = Path(__file__).parent
sys.path.insert(0, str(PLUGIN_DIR))

from openmythos.core.reasoning_engine import ReasoningEngine
from openmythos.core.recurrent_blocks import (
    RecurrentBlockStack, RecurrentBlock, LayerNorm, FeedForward, RecurrentCell
)
from openmythos.blocks import MythosBlock, ReflectionBlock, SynthesisBlock, RecursionBlock
from openmythos.reasoning import ChainOfThought, TreeOfThought, GraphOfThought


def _build_default_engine():
    """
    Build a fully-constructed ReasoningEngine with all required dependencies.
    This is the factory function that the hook scripts use.
    """
    config = {
        "recurrent": {
            "hidden_dim": 768,
            "depth": 4,
            "num_heads": 12,
        },
        "numpy": {
            "dtype": "float32"
        }
    }

    hidden_dim = config["recurrent"]["hidden_dim"]
    dtype = getattr(np, config["numpy"]["dtype"], np.float32)

    # Build recurrent stack
    rbs = RecurrentBlockStack(
        hidden_dim=hidden_dim,
        num_heads=config["recurrent"]["num_heads"],
        depth=config["recurrent"]["depth"],
        max_recurrence=3,
        dtype=dtype,
    )

    # Add special blocks — using correct constructor signatures
    mythos_config = {
        "archetypes": {},
        "fusion_method": "weighted_attention",
        "rotate_on_recurrence": True,
        "min_active_archetypes": 2
    }

    rbs.add_block("mythos", MythosBlock(config=mythos_config, hidden_dim=hidden_dim))
    rbs.add_block("reflection", ReflectionBlock(hidden_dim=hidden_dim))
    rbs.add_block("synthesis", SynthesisBlock(hidden_dim=hidden_dim))
    rbs.add_block("recursion", RecursionBlock(hidden_dim=hidden_dim))

    # Build reasoning paths
    chain = ChainOfThought(hidden_dim=hidden_dim, dtype=dtype)
    tree = TreeOfThought(hidden_dim=hidden_dim, max_depth=3, dtype=dtype)
    graph = GraphOfThought(hidden_dim=hidden_dim, dtype=dtype)

    return ReasoningEngine(
        recurrent_stack=rbs,
        chain=chain,
        tree=tree,
        graph=graph,
        config=config,
    )


class OpenMythosPlugin:
    """
    Main plugin class. Provides the reasoning engine interface
    that OpenCode/Crush hooks call via subprocess.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.engine = _build_default_engine()

    def _load_config(self, path: Optional[str]) -> Dict[str, Any]:
        """Load config from crush.json or embedded defaults."""
        if path and os.path.exists(path):
            with open(path) as f:
                return json.load(f)

        return {
            "recurrent": {"hidden_dim": 768, "depth": 4, "num_heads": 12},
            "numpy": {"dtype": "float32"}
        }

    def analyze_tool_use(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a tool use request through the OpenMythos reasoning pipeline.

        Called by hook_main.sh when OpenCode fires a PreToolUse event.
        """
        thinking = f"Tool: {tool_name}, Args: {json.dumps(arguments)}"

        # Encode the tool use as a representation
        rep = self.engine.encode_to_representation(thinking)

        # Determine which archetype should handle this
        archetype = self.engine.analyze_prompt_archetype(thinking)

        # Get modification signal (confidence assessment)
        signal = self.engine.decode_modification_signal(rep)

        # Process through chain-of-thought reasoning
        chain_out = self.engine.chain.forward(rep)

        return {
            "tool": tool_name,
            "recurrent_depth": len(self.engine.recurrent_stack.execution_order),
            "confidence": signal.get("confidence", 0.0),
            "contradiction_detected": signal.get("contradiction", False),
            "dominant_archetype": archetype,
            "refinements": [
                {"path": "chain", "shape": list(chain_out.shape)},
                {"signal": "modification", "confidence": signal.get("confidence", 0.0)}
            ],
            "original_arguments": arguments
        }

    def process(self, request_json: str) -> str:
        """
        Process a PreToolUse request. Entry point for hook_main.sh.

        Args:
            request_json: JSON string of the PreToolUse request (read from stdin)

        Returns:
            JSON string of the enriched request
        """
        try:
            request = json.loads(request_json) if request_json.strip() else {}
        except json.JSONDecodeError:
            return request_json

        tool_name = request.get("name", "unknown")
        arguments = request.get("arguments", {})

        analysis = self.analyze_tool_use(tool_name, arguments)

        # Enrich the request with reasoning metadata
        enriched = {**request, "_mythos": analysis}
        return json.dumps(enriched)


def main():
    """CLI entry point for hook_main.sh"""
    # Read from stdin (injection-safe)
    request_json = sys.stdin.read().strip()

    plugin = OpenMythosPlugin()
    result = plugin.process(request_json)
    print(result)


if __name__ == "__main__":
    main()
