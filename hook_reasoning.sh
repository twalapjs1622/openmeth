#!/usr/bin/env bash
# =============================================================================
# OpenMythos Lightweight Reasoning Hook
# =============================================================================
# A faster alternative to hook_main.sh that uses heuristic reasoning
# instead of the full numpy engine. Lower latency, less thorough.
# =============================================================================
set -euo pipefail

HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

read -r INPUT

if command -v jq &>/dev/null; then
    REQUEST="$(echo "$INPUT" | jq -r '.request // empty' 2>/dev/null)"
else
    REQUEST="$INPUT"
fi

if [ -z "$REQUEST" ]; then
    echo "$INPUT"
    exit 0
fi

RESULT=$(python3 -c "
import json, sys
req = json.loads(sys.argv[1]) if sys.argv[1] else {}
tool = req.get('name', '')
args = req.get('arguments', {})

confidence = 0.5
if 'read' in tool:
    confidence = 0.8
elif 'write' in tool:
    confidence = 0.7
elif 'shell' in tool:
    confidence = 0.6
elif 'search' in tool:
    confidence = 0.75

enriched = {**req, '_mythos': {
    'confidence': confidence,
    'analysis': f'Heuristic reasoning for {tool}',
    'tool': tool,
    'mode': 'lightweight'
}}
print(json.dumps(enriched))
" "$REQUEST" 2>/dev/null)

echo "$RESULT"
