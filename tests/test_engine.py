import pytest
import os
import torch
import warnings

from graviton.core.config import GravitonConfig, QuantMode
from graviton.core.engine import GravitonEngine
from graviton.core.hardware import detect_hardware

# Suppress warnings that clog test output
warnings.filterwarnings("ignore")

def test_hardware_detection():
    profile = detect_hardware()
    assert profile.platform_name in ["Darwin", "Linux", "Windows"]
    assert profile.total_memory_gb > 0
    assert profile.cpu_cores > 0

def test_engine_initialization_presets():
    # Test built-in config preset for minimal memory
    config = GravitonConfig.for_extreme_compression()
    
    # 1.58 bit ternary
    assert config.quantization.mode == QuantMode.TERNARY
    
    # High sparsity
    assert config.sparsity.k_ratio <= 0.5
    
    engine = GravitonEngine(config=config)
    assert engine is not None

def test_engine_benchmark_mock():
    # We can run the embedded benchmark which uses a blank dummy model naturally
    config = GravitonConfig(
        quant_bits=4.0,
        sparsity_ratio=0.5
    )
    
    engine = GravitonEngine(config=config)
    results = engine.benchmark()
    
    assert "quantization_time_ms" in results
    assert "sparsity_time_ms" in results
    assert results["memory_used_gb"] > 0
    assert results["quantization_mode"] == "int4"
