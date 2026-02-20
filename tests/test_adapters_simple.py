#!/usr/bin/env python3
"""Quick adapter validation script (no sklearn dependencies)."""
import sys
sys.path.insert(0, 'src')

import torch

# Import adapters directly to avoid sklearn dependencies
import importlib.util
spec = importlib.util.spec_from_file_location("adapters", "src/fmri2img/models/adapters.py")
adapters_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapters_module)

LinearAdapter = adapters_module.LinearAdapter
MLPAdapter = adapters_module.MLPAdapter
ConditionEmbedding = adapters_module.ConditionEmbedding
AdaptedModel = adapters_module.AdaptedModel
create_adapter = adapters_module.create_adapter

print("=" * 60)
print("ADAPTER MODULE VALIDATION")
print("=" * 60)

# Test 1: LinearAdapter
print("\n1. Testing LinearAdapter...")
adapter = LinearAdapter(embed_dim=512)
x = torch.randn(4, 512)
y = adapter(x)
assert y.shape == (4, 512), f"Expected shape (4, 512), got {y.shape}"
assert torch.allclose(torch.norm(y, dim=1), torch.ones(4), atol=1e-5), "Output not normalized"
print(f"   ✓ Forward pass: {x.shape} -> {y.shape}")
print("   ✓ Output L2 normalized")

# Test 2: MLPAdapter
print("\n2. Testing MLPAdapter...")
adapter2 = MLPAdapter(embed_dim=512, hidden_scale=2.0)
y2 = adapter2(x)
assert y2.shape == (4, 512), f"Expected shape (4, 512), got {y2.shape}"
assert torch.allclose(torch.norm(y2, dim=1), torch.ones(4), atol=1e-5), "Output not normalized"
print(f"   ✓ Forward pass: {x.shape} -> {y2.shape}")
print("   ✓ Output L2 normalized")

# Test 3: ConditionEmbedding
print("\n3. Testing ConditionEmbedding...")
cond = ConditionEmbedding(embed_dim=512, n_conditions=2, mode='add')
x_cond = cond(x, condition_idx=torch.tensor([0, 1, 0, 1]))
assert x_cond.shape == (4, 512), f"Expected shape (4, 512), got {x_cond.shape}"
print(f"   ✓ Additive conditioning: {x.shape} -> {x_cond.shape}")

cond_film = ConditionEmbedding(embed_dim=512, n_conditions=2, mode='film')
x_film = cond_film(x, condition_idx=torch.tensor([0, 1, 0, 1]))
assert x_film.shape == (4, 512), f"Expected shape (4, 512), got {x_film.shape}"
print(f"   ✓ FiLM conditioning: {x.shape} -> {x_film.shape}")

# Test 4: create_adapter factory
print("\n4. Testing create_adapter factory...")
adapter_linear = create_adapter('linear', embed_dim=512)
assert isinstance(adapter_linear, LinearAdapter), "Expected LinearAdapter"
print("   ✓ Created LinearAdapter")

adapter_mlp = create_adapter('mlp', embed_dim=512, hidden_scale=2.0)
assert isinstance(adapter_mlp, MLPAdapter), "Expected MLPAdapter"
print("   ✓ Created MLPAdapter")

# Test 5: Identity initialization check
print("\n5. Testing identity initialization...")
adapter_id = LinearAdapter(embed_dim=512)
x_test = torch.randn(8, 512)
x_test_norm = torch.nn.functional.normalize(x_test, dim=1)
y_test = adapter_id(x_test)
# Should be very close to identity at initialization
cosine_sim = torch.nn.functional.cosine_similarity(x_test_norm, y_test, dim=1)
assert cosine_sim.mean() > 0.99, f"Expected high similarity, got {cosine_sim.mean():.4f}"
print(f"   ✓ Identity init: mean cosine similarity = {cosine_sim.mean():.4f}")

print("\n" + "=" * 60)
print("✅ ALL ADAPTER VALIDATION TESTS PASSED!")
print("=" * 60)
print("\nNote: Full integration tests require sklearn, which has a")
print("numpy compatibility issue in this environment. The adapter")
print("code itself is working correctly.")
