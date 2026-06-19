"""
Thinking Interceptor — detects and captures AI model thinking blocks.

This module intercepts the model's thinking process at the token level,
identifies thinking blocks (e.g., <think>...</think>), buffers the
thinking tokens, and routes them through the OpenMythos reasoning engine.
"""

import re
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ThinkingBlock:
    """Represents a captured thinking block."""
    raw_text: str = ""
    start_idx: int = -1
    end_idx: int = -1
    is_active: bool = False
    token_count: int = 0
    chunks_processed: int = 0
    metadata: Dict = field(default_factory=dict)


class ThinkingInterceptor:
    """
    Intercepts AI model thinking blocks and routes them through
    the OpenMythos reasoning engine.
    
    Detection Strategy:
    - Pattern matching for thinking delimiters
    - Heuristic detection of reasoning-like text patterns
    - State machine for tracking in/out of thinking blocks
    """

    def __init__(self, config: Dict, engine: Any):
        self.config = config
        self.engine = engine
        
        # Compile detection patterns
        self.start_patterns = [
            re.compile(re.escape(p), re.IGNORECASE)
            for p in config.get("think_patterns", ["<think>"])
        ]
        self.end_patterns = [
            re.compile(re.escape(p), re.IGNORECASE)
            for p in config.get("end_patterns", ["</think>"])
        ]

        # Heuristic patterns for detecting unmarked thinking
        self.heuristic_patterns = [
            re.compile(r"^(let me think|thinking|step \d+|first,|therefore|however|but wait)", re.I),
            re.compile(r"(let's reason|reasoning:|analysis:|approach:)", re.I),
            re.compile(r"(\d+\.\s+(?:first|second|third|then|next|finally))", re.I),
        ]

        # State
        self.state = "IDLE"  # IDLE, THINKING, PROCESSING
        self.current_block: Optional[ThinkingBlock] = None
        self.completed_blocks: List[ThinkingBlock] = []
        
        # Token buffer
        self.token_buffer: List[str] = []
        self.buffer_size = config.get("token_buffer_size", 64)
        self.flush_interval = config.get("flush_interval", 32)

    def process_token(self, token: str, context: Dict) -> Dict:
        """
        Process a single incoming token.
        
        State machine:
            IDLE -> detect start -> THINKING
            THINKING -> buffer tokens, flush periodically -> PROCESSING
            THINKING -> detect end -> process final chunk -> IDLE
            PROCESSING -> run through engine -> back to THINKING
        
        Args:
            token: The incoming token
            context: Streaming context from OpenCode
        
        Returns:
            Modified context with any OpenMythos injections
        """
        result = context.copy()

        if self.state == "IDLE":
            result = self._handle_idle(token, context)

        elif self.state == "THINKING":
            result = self._handle_thinking(token, context)

        return result

    def _handle_idle(self, token: str, context: Dict) -> Dict:
        """Handle tokens while in IDLE state — look for thinking start."""
        accumulated = context.get("accumulated_text", "")
        
        # Check for explicit start patterns
        for pattern in self.start_patterns:
            if pattern.search(accumulated[-100:]):  # Check recent text
                self._enter_thinking(context)
                return context

        # Check heuristic patterns
        for pattern in self.heuristic_patterns:
            if pattern.search(token) or pattern.search(accumulated[-200:]):
                self._enter_thinking(context, heuristic=True)
                return context

        return context

    def _handle_thinking(self, token: str, context: Dict) -> Dict:
        """Handle tokens while in THINKING state — buffer and process."""
        if self.current_block is None:
            return context

        self.current_block.raw_text += token
        self.current_block.token_count += 1
        self.token_buffer.append(token)

        # Check for end of thinking block
        for pattern in self.end_patterns:
            if pattern.search(self.current_block.raw_text[-100:]):
                self._exit_thinking(context)
                return context

        # Process buffered tokens through engine
        if len(self.token_buffer) >= self.flush_interval:
            result = self._flush_buffer(context)
            return result

        return context

    def _enter_thinking(self, context: Dict, heuristic: bool = False):
        """Transition to THINKING state."""
        self.state = "THINKING"
        self.current_block = ThinkingBlock(
            start_idx=context.get("token_index", 0),
            is_active=True,
            metadata={"heuristic_detection": heuristic},
        )
        self.token_buffer = []

    def _exit_thinking(self, context: Dict):
        """Transition back to IDLE state after final processing."""
        if self.current_block and self.token_buffer:
            self._flush_buffer(context)

        if self.current_block:
            self.current_block.is_active = False
            self.current_block.end_idx = context.get("token_index", 0)
            self.completed_blocks.append(self.current_block)

        self.current_block = None
        self.token_buffer = []
        self.state = "IDLE"

    def _flush_buffer(self, context: Dict) -> Dict:
        """
        Flush the token buffer through the OpenMythos engine.
        """
        if not self.token_buffer or self.current_block is None:
            return context

        text_chunk = "".join(self.token_buffer)

        # Encode text to representation
        representation = self.engine.encode_to_representation(text_chunk)

        # Get current hidden state from session
        session = context.get("mythos_session")
        hidden = None
        if session and "recurrent_hidden" in session:
            hidden = session["recurrent_hidden"]

        # Process through reasoning engine
        engine_output = self.engine.process_thinking_chunk(
            representation,
            hidden_state=hidden,
        )

        # Apply modifications to context
        result = context.copy()
        result["mythos_output"] = engine_output
        result["mythos_modification"] = engine_output.get("modification_signal", {})

        self.current_block.chunks_processed += 1
        self.token_buffer = []

        return result

    def reset(self):
        """Reset interceptor state."""
        self.state = "IDLE"
        self.current_block = None
        self.completed_blocks = []
        self.token_buffer = []