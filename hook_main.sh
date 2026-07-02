#!/usr/bin/env bash
# =============================================================================
# OpenMythos Recurrent Thinking Hook for OpenCode/Crush
# =============================================================================
# Fixed: Shell injection vulnerability patched — uses stdin piping instead
# of string interpolation to avoid code injection from malicious tool requests.
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

# Run the OpenMythos reasoning engine — FIXED: use stdin to avoid injection
RESULT=$(echo "$REQUEST" | python3 -c "
import sys, json
sys.path.insert(0, '${ENGINE_DIR}')
from plugin import OpenMythosPlugin

plugin = OpenMythosPlugin()
# Read from stdin instead of interpolating into code — prevents injection
request_json = sys.stdin.read().strip()
try:
    result = plugin.process(request_json)
    print(result)
except Exception as e:
    # Fail gracefully — return original request
    print(request_json)
" 2>>"$LOG")

# Return the enriched result
echo "$RESULT"
