# OpenMythos Recurrent Thinking Plugin — Installation Guide

## Compatibility

This plugin is designed for **OpenCode** (now **Crush**) CLI using its **shell-based hook system**.

**Minimum versions:**
- OpenCode/Crush with hook support
- Python 3.8+
- numpy >= 1.24.0

## Installation

### 1. Clone the Plugin

```bash
git clone https://github.com/twalapjs1622/openmeth.git ~/.opencode/openmythos
```

### 2. Install Python Dependencies

```bash
pip install numpy
```

### 3. Create the Crush Hook Configuration

Copy `crush.json` to your OpenCode/Crush config directory and update the path:

```bash
mkdir -p ~/.config/crush
cp ~/.opencode/openmythos/crush.json ~/.config/crush/
```

Edit `~/.config/crush/crush.json` and update the hook path:

```json
{
  "hooks": {
    "PreToolUse": "bash /path/to/openmeth/hook_main.sh"
  }
}
```

### 4. Verify Installation

```bash
echo '{"request":{"name":"test_tool","arguments":{"key":"value"}}}' | bash ~/.opencode/openmythos/hook_main.sh
```

You should see JSON output with a `_mythos` field containing analysis results.

## Hook Modes

### Full Reasoning (hook_main.sh)
Uses the complete numpy-powered recurrent engine for deep analysis. Higher latency but more thorough.

```json
{
  "hooks": {
    "PreToolUse": "bash /path/to/openmeth/hook_main.sh"
  }
}
```

### Lightweight (hook_reasoning.sh)
Uses fast heuristics for quick analysis. Lower latency, less thorough.

```json
{
  "hooks": {
    "PreToolUse": "bash /path/to/openmeth/hook_reasoning.sh"
  }
}
```

## What Changed in v2.0.0

| Before (Broken) | After (Fixed) |
|----------------|---------------|
| Python module with fake API | Shell scripts + Python engine |
| `register_hook()` calls | Direct hook via bash scripts |
| `pre_think`, `during_think` hooks | `PreToolUse` hook only |
| `manifest.json` as config | `crush.json` for OpenCode config |
| `config.yaml` settings | Embedded defaults in Python |

## Troubleshooting

- **Hook not firing?** Verify `crush.json` path and that OpenCode/Crush is running with hooks enabled.
- **Python errors?** Check `hook.log` in the plugin directory.
- **jq missing?** The hooks work without jq but parsing is more basic.
