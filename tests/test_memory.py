import pytest
import torch

from graviton.memory.manager import MemoryManager
from graviton.core.config import MemoryConfig

def test_memory_budget_enforcement():
    # 1GB budget
    config = MemoryConfig(max_memory_gb=1.0)
    manager = MemoryManager(config)
    
    # We should have ~1GB available
    assert manager.available_gb > 0.9
    
    # Register some tensor allocations
    t1 = torch.zeros(int(0.5 * 1024**3 // 4), dtype=torch.float32) # ~0.5GB
    manager.register_layer("layer_1", t1)
    assert manager.used_gb >= 0.49
    
    t2 = torch.zeros(int(0.3 * 1024**3 // 4), dtype=torch.float32) # ~0.3GB
    manager.register_layer("layer_2", t2)
    assert manager.used_gb >= 0.79
    
    # Attempting to allocate another 0.3GB (total 1.1GB) triggers eviction of layer_1
    t3 = torch.zeros(int(0.3 * 1024**3 // 4), dtype=torch.float32)
    assert manager.register_layer("layer_3", t3)
    
    # Layer 1 should be gone, layer 2 and 3 should remain
    assert "layer_1" not in manager._cache
    assert "layer_2" in manager._cache
    assert "layer_3" in manager._cache

def test_lru_cache_eviction():
    config = MemoryConfig(max_memory_gb=1.0)
    manager = MemoryManager(config)
    
    # 1. Add layers
    manager.register_layer("layer_1", torch.randn(10, 10))
    manager.register_layer("layer_2", torch.randn(10, 10))
    manager.register_layer("layer_3", torch.randn(10, 10))
    
    assert "layer_1" in manager._cache
    assert "layer_2" in manager._cache
    
    # 2. Access layer 1 to make it recent
    _ = manager.get_layer("layer_1")
    
    # 3. Force shrink cache to size 2
    # Since layer 2 is now the Least Recently Used, it should be evicted
    while len(manager._cache) > 2:
        manager._evict_lru()
        
    assert "layer_1" in manager._cache # Kept because accessed recently
    assert "layer_3" in manager._cache # Kept because newly added
    assert "layer_2" not in manager._cache # Evicted
