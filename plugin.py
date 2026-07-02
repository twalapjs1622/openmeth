"""
OpenMythos Recurrent Thinking Plugin for OpenCode/Crush
========================================================

FIXED VERSION — Now compatible with OpenCode's shell-based hook system.

The plugin exposes:
  1. `hook_main.sh` — main PreToolUse hook entry point (bash script)
  2. `hook_reasoning.sh` — lightweight reasoning alternative
  3. `ReasoningEngine` — the Python engine invoked by the shell hooks

OpenCode/Crush calls hooks by executing shell scripts via an embedded POSIX shell.
This module provides the Python engine that those scripts call.
"""

import sys
import os
import json
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

# Ensure we can find our modules
PLUGIN_DIR = Path(__file__).parent
sys.path.insert(0, str(PLUGIN_DIR))

from openmythos.core.reasoning_engine import ReasoningEngine
from openmythos.core.recurrent_blocks import RecurrentBlockStack
from openmythos.blocks import MythosBlock, ReflectionBlock, SynthesisBlock


class OpenMythosPlugin:
    """
    Main plugin class. Provides the reasoning engine interface
    that OpenCode/Crush hooks call via subprocess.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.engine = ReasoningEngine()

    def _load_config(self, path: Optional[str]) -> Dict[str, Any]:
        """Load config from crush.json or embedded defaults."""
        if path and os.path.exists(path):
            with open(path) as f:
                return json.load(f)

        # Default config
        return {
            "recurrent": {
                "hidden_dim": 768,
                "depth": 4,
                "num_heads": 12
            },
            "numpy": {
                "dtype": "float32"
            }
        }

    def analyze_tool_use(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a tool use request through the OpenMythos reasoning pipeline.

        Called by hook_main.sh when OpenCode fires a PreToolUse event.
        """
        thinking = f"Tool: {tool_name}, Args: {json.dumps(arguments)}"
        result = self.engine.analyze_request(thinking)

        return {
            "tool": tool_name,
            "recurrent_depth": result.get("recurrent_depth", 0),
            "confidence": result.get("confidence", 0.0),
            "refinements": result.get("refinements", []),
            "archetypes": result.get("archetypes", {}),
            "original_arguments": arguments
        }

    def process(self, request_json: str) -> str:
        """
        Process a PreToolUse request. Entry point for hook_main.sh.

        Args:
            request_json: JSON string of the PreToolUse request

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
    if len(sys.argv) > 1:
        request_json = sys.argv[1]
    else:
        # Read from stdin if no args
        request_json = sys.stdin.read().strip()

    plugin = OpenMythosPlugin()
    result = plugin.process(request_json)
    print(result)


if __name__ == "__main__":
    main()
