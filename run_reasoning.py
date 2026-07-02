#!/usr/bin/env python3
"""
OpenMythos Reasoning Engine — CLI Interface for OpenCode Hooks

Runs the numpy-powered recurrent reasoning engine from the command line,
invoked by the shell hook script. This replaces the broken plugin.py
which tried to use a non-existent Python plugin API.
"""

import sys
import os
import argparse
import json
import logging
import numpy as np

# Add the plugin directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openmythos.core.reasoning_engine import ReasoningEngine, TextEncoder
from openmythos.core.recurrent_blocks import RecurrentBlockStack
from openmythos.blocks.mythos_block import MythosBlock
from openmythos.blocks.reflection_block import ReflectionBlock
from openmythos.blocks.synthesis_block import SynthesisBlock
from openmythos.blocks.recursion_block import RecursionBlock
from openmythos.reasoning.chain import ChainOfThought
from openmythos.reasoning.tree import TreeOfThought
from openmythos.reasoning.graph import GraphOfThought


def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="[OpenMythos] %(levelname)s: %(message)s",
    )


def build_engine(depth, hidden_dim, max_recurrence, convergence, debug):
    """Build the full OpenMythos reasoning engine stack."""
    dtype = np.float32
    np.random.seed(None)

    # Core recurrent stack
    stack = RecurrentBlockStack(
        hidden_dim=hidden_dim,
        num_heads=12,
        depth=depth,
        dropout=0.1,
        max_recurrence=max_recurrence,
        convergence_threshold=convergence,
        layer_norm=True,
        residual_alpha=0.5,
        dtype=dtype,
    )

    # Archetype config
    mythos_cfg = {
        "archetypes": {
            "explorer": {"weight": 0.25, "temperature": 1.2, "recurrence_bias": "expansion"},
            "architect": {"weight": 0.25, "temperature": 0.8, "recurrence_bias": "structuring"},
            "critic": {"weight": 0.25, "temperature": 0.6, "recurrence_bias": "compression"},
            "oracle": {"weight": 0.25, "temperature": 1.0, "recurrence_bias": "integration"},
        },
        "fusion_method": "weighted_attention",
        "rotate_on_recurrence": True,
        "min_active_archetypes": 2,
    }

    stack.add_block("mythos", MythosBlock(config=mythos_cfg, hidden_dim=hidden_dim, dtype=dtype))
    stack.add_block("reflection", ReflectionBlock(hidden_dim=hidden_dim, dtype=dtype))
    stack.add_block("synthesis", SynthesisBlock(hidden_dim=hidden_dim, dtype=dtype))
    stack.add_block("recursion", RecursionBlock(hidden_dim=hidden_dim, max_recurrence=max_recurrence, dtype=dtype))

    # Reasoning paths
    config = {
        "recurrent": {"hidden_dim": hidden_dim},
        "numpy": {"dtype": "float32"},
    }

    return ReasoningEngine(
        recurrent_stack=stack,
        chain=ChainOfThought(depth=5, dtype=dtype),
        tree=TreeOfThought(branching=3, dtype=dtype),
        graph=GraphOfThought(max_nodes=32, dtype=dtype),
        config=config,
    )


def analyze_tool_call(engine, tool_name, tool_input_str, archetype_hint, reasoning_mode):
    """
    Analyze a tool call through the OpenMythos reasoning engine.
    
    Returns a dict with:
      - decision: "allow", "deny"
      - reason: explanation for deny decisions
      - context: notes to inject into model context
      - modified_input: rewritten tool input (if applicable)
      - contradiction: detected contradictions in the tool call
    """
    # Create combined prompt for analysis
    prompt = f"Tool: {tool_name}\nInput: {tool_input_str}"
    
    # Determine archetype
    if archetype_hint == "auto":
        detected_archetype = engine.analyze_prompt_archetype(prompt)
    else:
        detected_archetype = archetype_hint

    # Encode and process through recurrent blocks
    representation = engine.encode_to_representation(prompt)
    
    # Get modification signal
    signal = engine.decode_modification_signal(representation)
    
    # Build context injection
    context_lines = []
    context_lines.append(f"OpenMythos Analysis [archetype={detected_archetype}]:")
    context_lines.append(f"  Contradiction detected: {signal.get('contradiction', 'none')}")
    context_lines.append(f"  Confidence: {signal.get('confidence', 'unknown')}")
    
    result = {
        "decision": "allow",
        "reason": "",
        "context": "\n".join(context_lines),
        "modified_input": "",
        "contradiction": signal.get("contradiction", "none"),
        "archetype": detected_archetype,
    }
    
    # If contradiction is high, consider denying
    contradiction = signal.get("contradiction", "none")
    if contradiction != "none" and contradiction != "low":
        result["decision"] = "deny"
        result["reason"] = f"High contradiction detected: {contradiction}. " \
                          f"The tool call may be inconsistent with the prompt context. " \
                          f"Detected archetype: {detected_archetype}"
    
    return result


def main():
    parser = argparse.ArgumentParser(description="OpenMythos Reasoning Engine CLI")
    parser.add_argument("--tool-name", required=True, help="Name of the tool being called")
    parser.add_argument("--tool-input", required=True, help="JSON tool input")
    parser.add_argument("--session-id", required=True, help="Session ID")
    parser.add_argument("--depth", type=int, default=3, help="Recurrent depth")
    parser.add_argument("--hidden-dim", type=int, default=768, help="Hidden dimension")
    parser.add_argument("--max-recurrence", type=int, default=6, help="Max recurrence depth")
    parser.add_argument("--convergence", type=float, default=0.0001, help="Convergence threshold")
    parser.add_argument("--archetype", default="auto", 
                       choices=["auto", "explorer", "architect", "critic", "oracle"],
                       help="Archetype to use (or 'auto' for detection)")
    parser.add_argument("--reasoning-mode", default="chain",
                       choices=["chain", "tree", "graph"],
                       help="Reasoning path mode")
    parser.add_argument("--debug", default=False, action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    setup_logging(args.debug)
    
    # Build engine
    engine = build_engine(
        depth=args.depth,
        hidden_dim=args.hidden_dim,
        max_recurrence=args.max_recurrence,
        convergence=args.convergence,
        debug=args.debug,
    )
    
    # Analyze
    result = analyze_tool_call(
        engine=engine,
        tool_name=args.tool_name,
        tool_input_str=args.tool_input,
        archetype_hint=args.archetype,
        reasoning_mode=args.reasoning_mode,
    )
    
    # Output in format expected by shell script
    if result["decision"] == "deny":
        print(f"DENY:{result['reason']}")
    else:
        if result["context"]:
            print(f"CONTEXT:{result['context']}")
        # Allow by default (no DENY output = allow)


if __name__ == "__main__":
    main()
