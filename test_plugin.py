"""
Comprehensive test suite for OpenMythos Recurrent Thinking Plugin.
Tests all modules: core, blocks, reasoning, utils.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = 0
FAIL = 0

def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} {detail}")

print("=" * 60)
print("OpenMythos Recurrent Thinking Plugin - Test Suite")
print("=" * 60)

# ============================================================
print("\n[1] Testing openmythos.core.attention")
# ============================================================
from openmythos.core.attention import MultiHeadAttention, CausalSelfAttention, CrossAttention

mha = MultiHeadAttention(hidden_dim=768, num_heads=12, dropout=0.0)
x = np.random.randn(2, 10, 768).astype(np.float32)
out, weights = mha.forward(x, x, x)
test("MultiHeadAttention output shape", out.shape == (2, 10, 768))
test("MultiHeadAttention weights shape", weights.shape == (2, 12, 10, 10))
test("MultiHeadAttention output finite", np.all(np.isfinite(out)))

csa = CausalSelfAttention(hidden_dim=768, num_heads=12, dropout=0.0)
out2, w2 = csa.forward(x)
test("CausalSelfAttention output shape", out2.shape == (2, 10, 768))

ca = CrossAttention(hidden_dim=768, num_heads=12, dropout=0.0)
q = np.random.randn(2, 5, 768).astype(np.float32)
k = np.random.randn(2, 8, 768).astype(np.float32)
out3, w3 = ca.forward(q, k)
test("CrossAttention output shape", out3.shape == (2, 5, 768))
test("CrossAttention weights shape", w3.shape == (2, 12, 5, 8))

# ============================================================
print("\n[2] Testing openmythos.core.recurrent_blocks")
# ============================================================
from openmythos.core.recurrent_blocks import (
    LayerNorm, FeedForward, RecurrentCell, 
    RecurrentBlock, RecurrentBlockStack
)

ln = LayerNorm(768)
x_ln = np.random.randn(2, 10, 768).astype(np.float32)
out_ln = ln.forward(x_ln)
test("LayerNorm output shape", out_ln.shape == (2, 10, 768))

ff = FeedForward(768)
out_ff = ff.forward(x_ln)
test("FeedForward output shape", out_ff.shape == (2, 10, 768))
test("FeedForward output finite", np.all(np.isfinite(out_ff)))

rc = RecurrentCell(768)
h = np.random.randn(2, 768).astype(np.float32)
c = np.random.randn(2, 768).astype(np.float32)
h_new, c_new = rc.forward(h, h, c)
test("RecurrentCell h_new shape", h_new.shape == (2, 768))
test("RecurrentCell c_new shape", c_new.shape == (2, 768))

rb = RecurrentBlock(hidden_dim=768, num_heads=12, dropout=0.0)
hidden = {"h": np.zeros((2, 768), dtype=np.float32), "c": np.zeros((2, 768), dtype=np.float32)}
out_rb, new_h = rb.forward(x_ln, hidden)
test("RecurrentBlock output shape", out_rb.shape == (2, 10, 768))

rbs = RecurrentBlockStack(hidden_dim=768, num_heads=12, depth=2, max_recurrence=3)
out_rbs, final_h, depth = rbs.forward(x_ln)
test("RecurrentBlockStack output shape", out_rbs.shape == (2, 10, 768))
test("RecurrentBlockStack depth > 0", depth > 0)

# ============================================================
print("\n[3] Testing openmythos.core.reasoning_engine")
# ============================================================
from openmythos.core.reasoning_engine import TextEncoder, ReasoningEngine

te = TextEncoder(hidden_dim=768)
enc = te.encode("Hello world, this is a test of the encoding system.")
test("TextEncoder output shape", enc.shape[0] == 1)
test("TextEncoder output has 3 dims", enc.ndim == 3)

# Test archetype analysis
# We need to mock the reasoning engine dependencies
class MockChain:
    def forward(self, x): return x
class MockTree:
    def forward(self, x): return x
class MockGraph:
    def forward(self, x): return x

config = {"recurrent": {"hidden_dim": 768}, "numpy": {"dtype": "float32"}}
engine = ReasoningEngine(
    recurrent_stack=rbs,
    chain=MockChain(),
    tree=MockTree(),
    graph=MockGraph(),
    config=config,
)
test("analyze_prompt_archetype - creative", 
     engine.analyze_prompt_archetype("Imagine creative alternatives") == "explorer")
test("analyze_prompt_archetype - design", 
     engine.analyze_prompt_archetype("Design a system architecture") == "architect")
test("analyze_prompt_archetype - evaluate", 
     engine.analyze_prompt_archetype("Evaluate and critique this code") == "critic")
test("analyze_prompt_archetype - understand", 
     engine.analyze_prompt_archetype("Synthesize the meaning") == "oracle")

rep = engine.encode_to_representation("Test reasoning")
mod = engine.decode_modification_signal(rep)
test("decode_modification_signal has confidence", "confidence" in mod)
test("decode_modification_signal has contradiction", "contradiction" in mod)

# ============================================================
print("\n[4] Testing openmythos.core.thinking_interceptor")
# ============================================================
from openmythos.core.thinking_interceptor import ThinkingInterceptor, ThinkingBlock

ti_config = {
    "think_patterns": ["<think>"],
    "end_patterns": ["</think>"],
    "token_buffer_size": 64,
    "flush_interval": 32,
}
ti = ThinkingInterceptor(config=ti_config, engine=engine)
test("ThinkingInterceptor initial state", ti.state == "IDLE")

ctx = {"accumulated_text": "", "token_index": 0}
result = ti.process_token("Let me think", ctx)
test("process_token returns dict", isinstance(result, dict))

# ============================================================
print("\n[5] Testing openmythos.blocks.mythos_block")
# ============================================================
from openmythos.blocks.mythos_block import MythosBlock, ArchetypePath

ap = ArchetypePath("explorer", 768, 1.2, "expansion", 0.25)
x_ap = np.random.randn(2, 768).astype(np.float32)
h_ap = np.random.randn(2, 768).astype(np.float32)
out_ap = ap.forward(x_ap, h_ap)
test("ArchetypePath output shape", out_ap.shape == (2, 768))

mb_config = {"archetypes": {}, "fusion_method": "weighted_attention", 
             "rotate_on_recurrence": True, "min_active_archetypes": 2}
mb = MythosBlock(config=mb_config, hidden_dim=768)
x_mb = np.random.randn(2, 5, 768).astype(np.float32)
hidden_mb = {"h": np.zeros((2, 768), dtype=np.float32), "c": np.zeros((2, 768), dtype=np.float32)}
out_mb, new_h_mb = mb.forward(x_mb, hidden_mb)
test("MythosBlock output shape", out_mb.shape == (2, 5, 768))

mb.rotate_weights(1)
test("MythosBlock rotate_weights works", True)

# ============================================================
print("\n[6] Testing openmythos.blocks.reflection_block")
# ============================================================
from openmythos.blocks.reflection_block import ReflectionBlock

refb = ReflectionBlock(hidden_dim=768)
x_ref = np.random.randn(2, 5, 768).astype(np.float32)
hidden_ref = {"h": np.zeros((2, 768), dtype=np.float32), "c": np.zeros((2, 768), dtype=np.float32)}
out_ref, new_h_ref = refb.forward(x_ref, hidden_ref)
test("ReflectionBlock output shape", out_ref.shape == (2, 5, 768))
test("ReflectionBlock memory populated", len(refb.memory_buffer) > 0)

# ============================================================
print("\n[7] Testing openmythos.blocks.synthesis_block")
# ============================================================
from openmythos.blocks.synthesis_block import SynthesisBlock

synb = SynthesisBlock(hidden_dim=768)
x_syn = np.random.randn(2, 5, 768).astype(np.float32)
hidden_syn = {"h": np.zeros((2, 768), dtype=np.float32), "c": np.zeros((2, 768), dtype=np.float32)}
out_syn, new_h_syn = synb.forward(x_syn, hidden_syn)
test("SynthesisBlock output shape", out_syn.shape == (2, 5, 768))

# ============================================================
print("\n[8] Testing openmythos.blocks.recursion_block")
# ============================================================
from openmythos.blocks.recursion_block import RecursionBlock

recb = RecursionBlock(hidden_dim=768, max_recurrence=3)
x_rec = np.random.randn(2, 5, 768).astype(np.float32)
hidden_rec = {"h": np.zeros((2, 768), dtype=np.float32), "c": np.zeros((2, 768), dtype=np.float32)}
out_rec, new_h_rec = recb.forward(x_rec, hidden_rec)
test("RecursionBlock output shape", out_rec.shape == (2, 5, 768))

# ============================================================
print("\n[9] Testing openmythos.reasoning modules")
# ============================================================
from openmythos.reasoning.chain import ChainOfThought
from openmythos.reasoning.tree import TreeOfThought
from openmythos.reasoning.graph import GraphOfThought

cot = ChainOfThought(depth=3)
x_cot = np.random.randn(2, 10, 768).astype(np.float32)
out_cot = cot.forward(x_cot)
test("ChainOfThought output shape", out_cot.shape == (2, 10, 768))

tot = TreeOfThought(branching=3, max_depth=2)
out_tot = tot.forward(x_cot)
test("TreeOfThought output shape", out_tot.shape == (2, 10, 768))

got = GraphOfThought(max_nodes=8)
out_got = got.forward(x_cot)
test("GraphOfThought output shape", out_got.shape == (2, 10, 768))

# ============================================================
print("\n[10] Testing openmythos.utils modules")
# ============================================================
from openmythos.utils.numpy_ops import (
    matmul_safe, layer_norm, softmax, gelu, sigmoid,
    scaled_dot_product_attention, positional_encoding,
    truncated_normal, cosine_similarity
)

a = np.random.randn(5, 10).astype(np.float32)
b = np.random.randn(10, 8).astype(np.float32)
out_mm = matmul_safe(a, b)
test("matmul_safe shape", out_mm.shape == (5, 8))

out_ln2 = layer_norm(a)
test("layer_norm shape", out_ln2.shape == (5, 10))

out_sm = softmax(a)
test("softmax sums to ~1", abs(np.sum(out_sm, axis=-1).mean() - 1.0) < 0.01)

out_gelu = gelu(a)
test("gelu shape", out_gelu.shape == (5, 10))

out_sig = sigmoid(a)
test("sigmoid range [0,1]", np.all(out_sig >= 0) and np.all(out_sig <= 1))

q = np.random.randn(2, 5, 64).astype(np.float32)
k = np.random.randn(2, 8, 64).astype(np.float32)
v = np.random.randn(2, 8, 64).astype(np.float32)
out_sda, w_sda = scaled_dot_product_attention(q, k, v)
test("scaled_dot_product_attention output shape", out_sda.shape == (2, 5, 64))

pe = positional_encoding(20, 64)
test("positional_encoding shape", pe.shape == (20, 64))

tn = truncated_normal((5, 5))
test("truncated_normal shape", tn.shape == (5, 5))

cs = cosine_similarity(a, a[:5])
test("cosine_similarity self = ~1", abs(float(cs.mean()) - 1.0) < 0.01)

from openmythos.utils.token_ops import (
    tokens_to_bag_representation, 
    extract_reasoning_segments,
    estimate_thinking_quality
)

tokens = ["hello", "world", "test"]
bag = tokens_to_bag_representation(tokens, dim=768)
test("tokens_to_bag_representation shape", bag.shape == (1, 768))

text = "Step 1: First, we analyze. Then, because of this, therefore we conclude."
segments = extract_reasoning_segments(text)
test("extract_reasoning_segments returns list", isinstance(segments, list))
test("extract_reasoning_segments found segments", len(segments) > 0)

quality = estimate_thinking_quality(text)
test("estimate_thinking_quality has depth", "depth" in quality)
test("estimate_thinking_quality has coherence", "coherence" in quality)

# ============================================================
print("\n[11] Testing full plugin integration")
# ============================================================
from openmythos.blocks import MythosBlock, ReflectionBlock, SynthesisBlock, RecursionBlock
from openmythos.reasoning import ChainOfThought, TreeOfThought, GraphOfThought

# Test that all imports work
test("All block imports", True)
test("All reasoning imports", True)

# Test the full pipeline: RecurrentBlockStack with special blocks
full_stack = RecurrentBlockStack(hidden_dim=768, num_heads=12, depth=2, max_recurrence=3)
full_stack.add_block("mythos", MythosBlock(config=mb_config, hidden_dim=768))
full_stack.add_block("reflection", ReflectionBlock(hidden_dim=768))
full_stack.add_block("synthesis", SynthesisBlock(hidden_dim=768))

x_full = np.random.randn(1, 8, 768).astype(np.float32)
out_full, h_full, d_full = full_stack.forward(x_full)
test("Full stack with special blocks output shape", out_full.shape == (1, 8, 768))

# ============================================================
print("\n" + "=" * 60)
print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
print("=" * 60)

if FAIL > 0:
    sys.exit(1)
else:
    print("All tests passed!")
    sys.exit(0)