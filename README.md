# 🧠 OpenMythos Recurrent Thinking Plugin for OpenCode

> Intercept free AI model thinking and reroute it through OpenMythos
> recurrent reasoning blocks — powered entirely by **numpy** (no PyTorch).

## What It Does

This plugin hooks into OpenCode's AI model pipeline and intercepts
thinking blocks (`<think>...</think>`, etc.). Instead of letting the
model think passively, it:

1. **Captures** thinking tokens as they stream
2. **Encodes** them into numpy tensor representations
3. **Processes** them through recurrent reasoning blocks:
   - **Mythos Block** — four archetypal thinking modes (Explorer, Architect, Critic, Oracle)
   - **Reflection Block** — self-evaluative reasoning with memory
   - **Synthesis Block** — integration of multiple reasoning threads
   - **Recursion Block** — meta-recursion with convergence monitoring
4. **Routes** through reasoning paths (Chain, Tree, Graph of Thought)
5. **Feeds back** modification signals that guide the model's next thinking tokens

## Architecture