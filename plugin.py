"""
OpenMythos Recurrent Thinking Plugin for OpenCode

Intercepts free AI model thinking and reroutes through
OpenMythos recurrent reasoning blocks (numpy-powered).
"""

import os
import sys
import yaml
import logging
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass, field

# Plugin-relative imports
from openmythos.core.thinking_interceptor import ThinkingInterceptor
from openmythos.core.reasoning_engine import ReasoningEngine
from openmythos.core.recurrent_blocks import RecurrentBlockStack
from openmythos.blocks.mythos_block import MythosBlock
from openmythos.blocks.reflection_block import ReflectionBlock
from openmythos.blocks.synthesis_block import SynthesisBlock
from openmythos.blocks.recursion_block import RecursionBlock
from openmythos.reasoning.chain import ChainOfThought
from openmythos.reasoning.tree import TreeOfThought
from openmythos.reasoning.graph import GraphOfThought

logger = logging.getLogger("openmythos")

@dataclass
class PluginState:
    """Mutable plugin runtime state."""
    initialized: bool = False
    interceptor: Optional[ThinkingInterceptor] = None
    engine: Optional[ReasoningEngine] = None
    recurrent_stack: Optional[RecurrentBlockStack] = None
    active_session: Optional[Dict] = None
    think_buffer: list = field(default_factory=list)
    stats: Dict = field(default_factory=lambda: {
        "interceptions": 0,
        "recurrent_passes": 0,
        "tokens_processed": 0,
        "avg_recurrence_depth": 0.0,
    })


class OpenMythosPlugin:
    """
    Main plugin class that hooks into OpenCode's AI model
    thinking pipeline and reroutes through OpenMythos
    recurrent reasoning blocks.
    """

    # Plugin metadata
    NAME = "openmythos-recurrent-thinking"
    VERSION = "1.0.0"

    def __init__(self, opencode_context: Optional[Any] = None):
        self.state = PluginState()
        self.opencode = opencode_context
        self.config = self._load_config()
        self._setup_logging()

    def _load_config(self) -> Dict:
        """Load plugin configuration."""
        config_path = os.path.join(
            os.path.dirname(__file__), "config.yaml"
        )
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info("OpenMythos config loaded successfully")
            return config
        except FileNotFoundError:
            logger.warning("Config not found, using defaults")
            return self._default_config()
        except yaml.YAMLError as e:
            logger.error(f"Config parse error: {e}")
            return self._default_config()

    def _default_config(self) -> Dict:
        """Fallback default configuration."""
        return {
            "engine": {
                "enabled": True,
                "debug": False,
                "device": "cpu",
            },
            "recurrent": {
                "depth": 4,
                "hidden_dim": 768,
                "num_heads": 12,
                "dropout": 0.1,
                "max_recurrence": 8,
                "convergence_threshold": 1e-4,
                "layer_norm": True,
                "residual_alpha": 0.5,
            },
            "mythos": {
                "archetypes": {
                    "explorer": {"weight": 0.25, "temperature": 1.2, "recurrence_bias": "expansion"},
                    "architect": {"weight": 0.25, "temperature": 0.8, "recurrence_bias": "structuring"},
                    "critic": {"weight": 0.25, "temperature": 0.6, "recurrence_bias": "compression"},
                    "oracle": {"weight": 0.25, "temperature": 1.0, "recurrence_bias": "integration"},
                },
                "fusion_method": "weighted_attention",
                "rotate_on_recurrence": True,
                "min_active_archetypes": 2,
            },
            "interceptor": {
                "think_patterns": ["<think>", "<thinking>", "<reasoning>", "[THINK]"],
                "end_patterns": ["</think>", "</thinking>", "</reasoning>", "[/THINK]"],
                "mid_stream_injection": True,
                "token_buffer_size": 64,
                "flush_interval": 32,
            },
        }

    def _setup_logging(self):
        """Configure plugin logging."""
        level = logging.DEBUG if self.config.get("engine", {}).get("debug") else logging.INFO
        logging.basicConfig(
            level=level,
            format="[OpenMythos] %(levelname)s: %(message)s"
        )

    # ── OpenCode Plugin Lifecycle Hooks ──────────────────────

    def on_load(self, opencode_api: Any):
        """
        Called when OpenCode loads the plugin.
        Initialize all OpenMythos components.
        """
        logger.info(f"Loading {self.NAME} v{self.VERSION}...")

        # Initialize numpy backend
        import numpy as np
        seed = self.config.get("numpy", {}).get("random_seed")
        if seed is not None:
            np.random.seed(seed)
        
        dtype_str = self.config.get("numpy", {}).get("dtype", "float32")
        self.numpy_dtype = getattr(np, dtype_str)
        logger.info(f"Numpy backend: dtype={dtype_str}")

        # Build recurrent block stack
        recurrent_cfg = self.config.get("recurrent", {})
        mythos_cfg = self.config.get("mythos", {})

        self.state.recurrent_stack = RecurrentBlockStack(
            hidden_dim=recurrent_cfg.get("hidden_dim", 768),
            num_heads=recurrent_cfg.get("num_heads", 12),
            depth=recurrent_cfg.get("depth", 4),
            dropout=recurrent_cfg.get("dropout", 0.1),
            max_recurrence=recurrent_cfg.get("max_recurrence", 8),
            convergence_threshold=recurrent_cfg.get("convergence_threshold", 1e-4),
            layer_norm=recurrent_cfg.get("layer_norm", True),
            residual_alpha=recurrent_cfg.get("residual_alpha", 0.5),
            dtype=self.numpy_dtype,
        )

        # Build mythos archetypal blocks
        self.state.recurrent_stack.add_block("mythos", MythosBlock(
            config=mythos_cfg,
            hidden_dim=recurrent_cfg.get("hidden_dim", 768),
            dtype=self.numpy_dtype,
        ))
        self.state.recurrent_stack.add_block("reflection", ReflectionBlock(
            hidden_dim=recurrent_cfg.get("hidden_dim", 768),
            dtype=self.numpy_dtype,
        ))
        self.state.recurrent_stack.add_block("synthesis", SynthesisBlock(
            hidden_dim=recurrent_cfg.get("hidden_dim", 768),
            dtype=self.numpy_dtype,
        ))
        self.state.recurrent_stack.add_block("recursion", RecursionBlock(
            hidden_dim=recurrent_cfg.get("hidden_dim", 768),
            max_recurrence=recurrent_cfg.get("max_recurrence", 8),
            dtype=self.numpy_dtype,
        ))

        # Initialize reasoning engine
        self.state.engine = ReasoningEngine(
            recurrent_stack=self.state.recurrent_stack,
            chain=ChainOfThought(
                depth=self.config.get("reasoning", {}).get("chain_depth", 5),
                dtype=self.numpy_dtype,
            ),
            tree=TreeOfThought(
                branching=self.config.get("reasoning", {}).get("tree_branching", 3),
                dtype=self.numpy_dtype,
            ),
            graph=GraphOfThought(
                max_nodes=self.config.get("reasoning", {}).get("max_graph_nodes", 32),
                dtype=self.numpy_dtype,
            ),
            config=self.config,
        )

        # Initialize thinking interceptor
        self.state.interceptor = ThinkingInterceptor(
            config=self.config.get("interceptor", {}),
            engine=self.state.engine,
        )

        # Register hooks with OpenCode
        self._register_hooks(opencode_api)

        self.state.initialized = True
        logger.info(f"{self.NAME} loaded and ready to intercept thinking!")

    def _register_hooks(self, api: Any):
        """
        Register all interception hooks with OpenCode's plugin API.
        """
        # Hook into the model's thinking/streaming pipeline
        api.register_hook("pre_think", self.hook_pre_think)
        api.register_hook("during_think", self.hook_during_think)
        api.register_hook("post_think", self.hook_post_think)
        api.register_hook("token_stream", self.hook_token_stream)
        
        logger.info("All hooks registered with OpenCode")

    # ── Hook Implementations ─────────────────────────────────

    def hook_pre_think(self, context: Dict) -> Dict:
        """
        Called BEFORE the model starts a thinking block.
        Prepares the OpenMythos recurrent processing pipeline.
        
        Args:
            context: Contains 'prompt', 'model', 'params', etc.
        
        Returns:
            Modified context with OpenMythos injection markers.
        """
        if not self.config.get("engine", {}).get("enabled", True):
            return context

        logger.debug("Pre-think hook triggered")

        self.state.stats["interceptions"] += 1

        # Initialize a new thinking session
        self.state.active_session = {
            "session_id": id(context),
            "prompt": context.get("prompt", ""),
            "model": context.get("model", "unknown"),
            "archetype_state": {},
            "recurrent_hidden": None,
            "thinking_tokens": [],
            "recurrent_depth_used": 0,
        }

        # Prepare the recurrent hidden state
        recurrent_cfg = self.config.get("recurrent", {})
        hidden_dim = recurrent_cfg.get("hidden_dim", 768)
        
        self.state.active_session["recurrent_hidden"] = {
            "h": np.zeros((1, hidden_dim), dtype=self.numpy_dtype),
            "c": np.zeros((1, hidden_dim), dtype=self.numpy_dtype),
        }

        # Determine which archetype to start with based on prompt analysis
        dominant_archetype = self.state.engine.analyze_prompt_archetype(
            context.get("prompt", "")
        )
        self.state.active_session["dominant_archetype"] = dominant_archetype
        
        logger.info(f"Session started | dominant archetype: {dominant_archetype}")

        # Inject OpenMythos system instruction into context
        mythos_instruction = self._build_mythos_instruction(dominant_archetype)
        context["system_override"] = mythos_instruction
        context["mythos_active"] = True
        context["mythos_session"] = self.state.active_session["session_id"]

        return context

    def hook_during_think(self, context: Dict) -> Dict:
        """
        Called DURING thinking as tokens stream.
        Intercepts and processes thinking tokens through
        recurrent blocks in real-time.
        
        Args:
            context: Contains 'token', 'accumulated_text', 'state', etc.
        
        Returns:
            Modified context with potentially altered tokens.
        """
        if not self.state.active_session:
            return context

        token = context.get("token", "")
        accumulated = context.get("accumulated_text", "")
        
        # Buffer incoming thinking tokens
        self.state.think_buffer.append(token)
        self.state.stats["tokens_processed"] += 1

        buffer_size = self.config.get("interceptor", {}).get("token_buffer_size", 64)
        flush_interval = self.config.get("interceptor", {}).get("flush_interval", 32)

        # Process through recurrent blocks when buffer is full
        if len(self.state.think_buffer) >= flush_interval or \
           self._is_think_end(token):
            
            # Encode buffered text to representation
            text_chunk = "".join(self.state.think_buffer)
            representation = self.state.engine.encode_to_representation(text_chunk)
            
            # Run through recurrent block stack
            recurrent_cfg = self.config.get("recurrent", {})
            hidden = self.state.active_session["recurrent_hidden"]
            
            output, new_hidden, depth_used = self.state.recurrent_stack.forward(
                representation,
                hidden_state=hidden,
                max_recurrence=recurrent_cfg.get("max_recurrence", 8),
                convergence_threshold=recurrent_cfg.get("convergence_threshold", 1e-4),
            )

            self.state.active_session["recurrent_hidden"] = new_hidden
            self.state.active_session["recurrent_depth_used"] += depth_used
            self.state.stats["recurrent_passes"] += depth_used

            # Decode the refined representation back to text modifications
            modification_signal = self.state.engine.decode_modification_signal(output)
            
            # Apply modification to context
            context["mythos_modification"] = modification_signal
            context["mythos_recurrence_depth"] = depth_used
            
            # Optionally inject guiding tokens
            if self.config.get("interceptor", {}).get("mid_stream_injection", True):
                guided = self._generate_guidance_tokens(
                    output, self.state.active_session
                )
                context["mythos_guidance"] = guided

            # Clear buffer
            self.state.think_buffer = []

        return context

    def hook_post_think(self, context: Dict) -> Dict:
        """
        Called AFTER a thinking block completes.
        Final recurrent processing pass and synthesis.
        """
        if not self.state.active_session:
            return context

        logger.debug("Post-think hook triggered")

        # Process any remaining buffered tokens
        if self.state.think_buffer:
            text_chunk = "".join(self.state.think_buffer)
            representation = self.state.engine.encode_to_representation(text_chunk)
            
            output, new_hidden, depth_used = self.state.recurrent_stack.forward(
                representation,
                hidden_state=self.state.active_session["recurrent_hidden"],
                max_recurrence=self.config.get("recurrent", {}).get("max_recurrence", 8),
                convergence_threshold=self.config.get("recurrent", {}).get("convergence_threshold", 1e-4),
            )
            
            self.state.active_session["recurrent_hidden"] = new_hidden
            self.state.stats["recurrent_passes"] += depth_used

        # Final synthesis pass
        full_thinking = context.get("thinking_text", "")
        final_output = self.state.engine.synthesize(full_thinking, self.state.active_session)

        # Update stats
        if self.state.stats["interceptions"] > 0:
            self.state.stats["avg_recurrence_depth"] = (
                self.state.stats["recurrent_passes"] / 
                self.state.stats["interceptions"]
            )

        context["mythos_final_synthesis"] = final_output
        context["mythos_stats"] = self.state.stats.copy()

        logger.info(
            f"Session complete | recurrence depth: "
            f"{self.state.active_session['recurrent_depth_used']} | "
            f"avg: {self.state.stats['avg_recurrence_depth']:.2f}"
        )

        # Clean up session
        self.state.active_session = None
        self.state.think_buffer = []

        return context

    def hook_token_stream(self, context: Dict) -> Dict:
        """
        Called for every token in the output stream.
        Detects thinking blocks and activates interception.
        """
        token = context.get("token", "")
        
        # Check for thinking block start
        interceptor_cfg = self.config.get("interceptor", {})
        think_patterns = interceptor_cfg.get("think_patterns", [])
        
        for pattern in think_patterns:
            if pattern in token or pattern in context.get("accumulated_text", "")[-50:]:
                context["mythos_thinking_detected"] = True
                logger.debug("Thinking block detected in stream")
                break

        return context

    # ── Helper Methods ───────────────────────────────────────

    def _build_mythos_instruction(self, archetype: str) -> str:
        """Build OpenMythos system instruction for the model."""
        return (
            f"[OpenMythos Recurrent Thinking Active]\n"
            f"Mode: {archetype}\n"
            f"Engage deep recurrent reasoning. Structure your thinking as:\n"
            f"1. EXPLORE: Generate multiple perspectives and possibilities\n"
            f"2. ARCHITECT: Identify structural patterns and relationships\n"
            f"3. CRITIQUE: Evaluate, find contradictions, refine\n"
            f"4. SYNTHESIZE: Integrate insights into coherent conclusion\n"
            f"5. RECURSE: Re-examine from the synthesis, deepen understanding\n"
            f"Each pass refines the reasoning. Embrace productive uncertainty.\n"
        )

    def _is_think_end(self, token: str) -> bool:
        """Check if token signals end of thinking block."""
        end_patterns = self.config.get("interceptor", {}).get("end_patterns", [])
        return any(p in token for p in end_patterns)

    def _generate_guidance_tokens(self, output: "np.ndarray", 
                                   session: Dict) -> Dict:
        """Generate guidance signals from recurrent output."""
        # The output representation encodes what direction the
        # thinking should take next
        archetype_weights = self.state.engine.extract_archetype_weights(output)
        
        return {
            "archetype_weights": archetype_weights,
            "expansion_signal": float(np.mean(np.abs(output))),
            "coherence_signal": float(1.0 - np.var(output) / (np.mean(output**2) + 1e-8)),
            "recursion_signal": float(np.linalg.norm(output, ord=2)),
        }

    # ── Public API ───────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Return plugin runtime statistics."""
        return self.state.stats.copy()

    def reset_stats(self):
        """Reset runtime statistics."""
        self.state.stats = {
            "interceptions": 0,
            "recurrent_passes": 0,
            "tokens_processed": 0,
            "avg_recurrence_depth": 0.0,
        }

    def set_archetype_weight(self, archetype: str, weight: float):
        """Dynamically adjust an archetype's weight."""
        mythos_cfg = self.config.get("mythos", {})
        if archetype in mythos_cfg.get("archetypes", {}):
            mythos_cfg["archetypes"][archetype]["weight"] = max(0.0, min(1.0, weight))
            logger.info(f"Archetype '{archetype}' weight set to {weight}")

    def on_unload(self):
        """Cleanup when plugin is unloaded."""
        logger.info(f"Unloading {self.NAME}...")
        self.state.active_session = None
        self.state.think_buffer = []
        self.state.initialized = False
        logger.info(f"{self.NAME} unloaded")


# ── Plugin Entry Point ───────────────────────────────────────

# OpenCode looks for this function
def create_plugin(opencode_context=None):
    """Factory function called by OpenCode to instantiate the plugin."""
    plugin = OpenMythosPlugin(opencode_context)
    return plugin


# For standalone testing
if __name__ == "__main__":
    plugin = OpenMythosPlugin()
    print(f"OpenMythos Plugin v{plugin.VERSION} initialized")
    print(f"Config: {plugin.config['recurrent']}")