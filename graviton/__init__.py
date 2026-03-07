"""
Graviton — Ultra-Efficient AI Inference Engine

Defying the gravitational pull of massive AI models.
Run 500B+ parameter models on consumer hardware through
extreme quantization, dynamic sparsity, and intelligent
memory management.
"""

__version__ = "0.1.0"
__author__ = "Graviton Contributors"

from graviton.core.config import GravitonConfig
from graviton.core.engine import GravitonEngine
from graviton.core.hardware import HardwareProfile, detect_hardware

__all__ = [
    "GravitonConfig",
    "GravitonEngine",
    "HardwareProfile",
    "detect_hardware",
]
