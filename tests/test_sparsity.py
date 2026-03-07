import pytest
import torch

from graviton.sparsity.topk import TopKActivation

def test_topk_activation_sparsity():
    # Keep only top 25% of neurons
    activation = TopKActivation(k_ratio=0.25)
    
    torch.manual_seed(42)
    # Batch=4, Features=100
    x = torch.randn(4, 100)
    
    out = activation(x)
    
    # Check shape is preserved
    assert out.shape == x.shape
    
    # Out of 100 features, keeping top 25% means there should be exactly 75 zeros per row
    zeros_per_row = (out == 0).sum(dim=1)
    
    for count in zeros_per_row:
        assert count.item() == 75, f"Expected 75 zeros, found {count.item()}"

    # Calculate overall sparsity
    overall_sparsity = (out == 0).float().mean().item()
    assert overall_sparsity == 0.75

def test_topk_activation_identity():
    # If ratio is 1.0, it should be equivalent to base activation
    activation = TopKActivation(k_ratio=1.0)
    
    x = torch.randn(4, 100)
    out1 = activation(x)
    out2 = torch.nn.SiLU()(x)
    
    assert torch.allclose(out1, out2)
