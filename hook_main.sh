#!/usr/bin/env bash
# =============================================================================
# OpenMythos Recurrent Thinking Hook for OpenCode/Crush
# =============================================================================
# Entry point for PreToolUse hook. Calls the Python reasoning engine via
# subprocess to intercept and analyze tool use before it executes.
# =============================================================================
set -euo pipefail

HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="${HOOK_DIR}"
LOG="${ENGINE_DIR}/hook.log"

# Read the PreToolUse input from stdin
read -r INPUT

# Extract the request JSON from the input
if command -v jq &>/dev/null; then
    REQUEST="$(echo "$INPUT" | jq -r '.request // empty' 2>/dev/null)"
else
    REQUEST="$INPUT"
fi

# Validate we have a request to process
if [ -z "$REQUEST" ]; then
    echo "$INPUT"
    exit 0
fi

# Run the OpenMythos reasoning engine
RESULT=$(python3 -c "
import sys, json
sys.path.insert(0, '${ENGINE_DIR}')
from plugin import OpenMythosPlugin

plugin = OpenMythosPlugin()
try:
    result = plugin.process('''${REQUEST}''')
    print(result)
except Exception:
    print('''${REQUEST}''')
" 2>>"$LOG")

# Return the enriched result
echo "$RESULT"
