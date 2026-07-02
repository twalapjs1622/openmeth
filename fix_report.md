# OpenMythos Plugin — OpenCode/Crush Compatibility Fix

## Problem Summary
The plugin was written for a **non-existent Python plugin interface**. OpenCode (renamed to Crush) does not have a Python plugin API at all. Its hook system works by **executing shell scripts** via an embedded POSIX shell.

## What's Broken
1. **`plugin.py`** — Tries to use `api.register_hook()` which doesn't exist
2. **`manifest.json`** — Ignored by OpenCode/Crush (uses `crush.json`)
3. **`config.yaml`** — Ignored (config is JSON in `crush.json`)
4. **Hooks `pre_think`, `during_think`, `post_think`, `token_stream`** — None of these exist. Only `PreToolUse` is available
5. **Entire Python architecture** — OpenCode hooks must be executable shell scripts, not Python modules with a lifecycle API

## Fix Applied
- Created `hook_main.sh` — Main entry point that bridges the OpenMythos Python engine to OpenCode's shell hook system
- Created `crush.json` — Proper OpenCode hook configuration  
- Created `README_INSTALL.md` — Installation instructions
- Created `hook_reasoning.sh` — Lightweight reasoning pre-processor
- Kept all original Python engine files for the numpy-based reasoning core
