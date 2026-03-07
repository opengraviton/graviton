"""
Benchmark script to measure real-world memory compression
on actual HuggingFace models using Graviton.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import time
from graviton.core.engine import GravitonEngine
from graviton.core.config import GravitonConfig

def run_compression_benchmark(model_id: str):
    print(f"\n{'='*50}")
    print(f"Graviton Memory Compression Benchmark: {model_id}")
    print(f"{'='*50}")
    
    # FP16 (Baseline)
    print("\n[1/3] Loading Baseline (Loading in standard uncompressed format)...")
    config_fp16 = GravitonConfig(quant_bits=16) 
    # Force no quantizer for baseline
    engine_fp16 = GravitonEngine(model_path=model_id, config=config_fp16)
    engine_fp16.quantizer = None
    
    start = time.time()
    engine_fp16.load_model()
    load_time_fp16 = time.time() - start
    
    info_fp16 = engine_fp16.get_model_info()
    baseline_mem = info_fp16["memory_usage_gb"]
    
    print(f"  Params: {info_fp16['total_parameters_billions']:.2f}B")
    print(f"  Memory Footprint: {baseline_mem:.2f} GB")
    print(f"  Load Time: {load_time_fp16:.2f}s")
    
    # 4-Bit Quantization
    print("\n[2/3] Loading with INT4 Quantization...")
    config_int4 = GravitonConfig(quant_bits=4)
    engine_int4 = GravitonEngine(model_path=model_id, config=config_int4)
    
    start = time.time()
    engine_int4.load_model()
    load_time_int4 = time.time() - start
    
    info_int4 = engine_int4.get_model_info()
    mem_int4 = info_int4["memory_usage_gb"]
    
    print(f"  Memory Footprint: {mem_int4:.2f} GB")
    print(f"  Compression: {baseline_mem / mem_int4:.2f}x")
    print(f"  Load Time: {load_time_int4:.2f}s")
    
    # 1.58-Bit Ternary Quantization
    print("\n[3/3] Loading with 1.58-Bit Ternary Quantization...")
    config_ternary = GravitonConfig(quant_bits=1.58)
    engine_ternary = GravitonEngine(model_path=model_id, config=config_ternary)
    
    start = time.time()
    engine_ternary.load_model()
    load_time_ternary = time.time() - start
    
    info_ternary = engine_ternary.get_model_info()
    mem_ternary = info_ternary["memory_usage_gb"]
    
    print(f"  Memory Footprint: {mem_ternary:.2f} GB")
    print(f"  Compression: {baseline_mem / mem_ternary:.2f}x")
    print(f"  Load Time: {load_time_ternary:.2f}s")
    
    print(f"\n{'='*50}")
    print("Summary:")
    print(f"Baseline FP16:   {baseline_mem:.2f} GB")
    print(f"Graviton INT4:   {mem_int4:.2f} GB ({baseline_mem / mem_int4:.1f}x smaller)")
    print(f"Graviton Ternary:{mem_ternary:.2f} GB ({baseline_mem / mem_ternary:.1f}x smaller)")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    # We use a small 1.1B parameter model to keep the test fast
    # A full 7B or 70B model would take much longer to download
    run_compression_benchmark("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
