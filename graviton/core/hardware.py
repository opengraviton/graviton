"""
Hardware Detection and Profiling

Automatically detects system capabilities (CPU, GPU, memory, storage)
and recommends optimal Graviton configuration for the detected hardware.
"""

from __future__ import annotations

import os
import platform
import subprocess
import logging
from dataclasses import dataclass, field
from typing import Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    """
    System hardware profile containing detected capabilities.

    Attributes:
        platform_name: Operating system name (Darwin, Linux, Windows).
        arch: CPU architecture (arm64, x86_64).
        cpu_name: Human-readable CPU name.
        cpu_cores: Number of CPU cores.
        total_memory_gb: Total system RAM in GB.
        available_memory_gb: Currently available RAM in GB.
        has_apple_silicon: Whether running on Apple Silicon.
        has_unified_memory: Whether system has unified memory architecture.
        has_mps: Whether Metal Performance Shaders are available.
        has_cuda: Whether CUDA GPUs are available.
        cuda_devices: List of detected CUDA device names.
        gpu_memory_gb: Total GPU memory in GB (CUDA only).
        ssd_read_speed_gbps: Estimated SSD read speed in GB/s.
    """

    platform_name: str = ""
    arch: str = ""
    cpu_name: str = ""
    cpu_cores: int = 0
    total_memory_gb: float = 0.0
    available_memory_gb: float = 0.0

    # Apple Silicon
    has_apple_silicon: bool = False
    has_unified_memory: bool = False
    has_mps: bool = False

    # CUDA
    has_cuda: bool = False
    cuda_devices: list = field(default_factory=list)
    gpu_memory_gb: float = 0.0

    # Storage
    ssd_read_speed_gbps: float = 2.0  # Conservative default

    def summary(self) -> str:
        """Return a formatted summary of the hardware profile."""
        lines = [
            "╔══════════════════════════════════════╗",
            "║       Hardware Profile               ║",
            "╠══════════════════════════════════════╣",
            f"║ Platform: {self.platform_name:>24} ║",
            f"║ Architecture: {self.arch:>20} ║",
            f"║ CPU: {self.cpu_name[:29]:>29} ║",
            f"║ CPU Cores: {self.cpu_cores:>23} ║",
            f"║ Total Memory: {self.total_memory_gb:>17.1f} GB ║",
            f"║ Available Memory: {self.available_memory_gb:>13.1f} GB ║",
            f"║ Apple Silicon: {str(self.has_apple_silicon):>19} ║",
            f"║ Unified Memory: {str(self.has_unified_memory):>18} ║",
            f"║ MPS Available: {str(self.has_mps):>19} ║",
            f"║ CUDA Available: {str(self.has_cuda):>18} ║",
        ]

        if self.has_cuda and self.cuda_devices:
            for i, dev in enumerate(self.cuda_devices):
                lines.append(f"║   GPU {i}: {dev[:25]:>25} ║")
            lines.append(f"║   GPU Memory: {self.gpu_memory_gb:>17.1f} GB ║")

        lines.append(f"║ SSD Speed: {self.ssd_read_speed_gbps:>18.1f} GB/s ║")
        lines.append("╚══════════════════════════════════════╝")

        # Model capacity estimates
        lines.append("")
        lines.append("Estimated Model Capacity:")
        usable = self.available_memory_gb * 0.8
        lines.append(f"  FP16:    {usable / 2:>8.1f}B parameters")
        lines.append(f"  INT8:    {usable:>8.1f}B parameters")
        lines.append(f"  INT4:    {usable * 2:>8.1f}B parameters")
        lines.append(f"  INT2:    {usable * 4:>8.1f}B parameters")
        lines.append(f"  Ternary: {usable * (8 / 1.585):>8.1f}B parameters")

        return "\n".join(lines)

    def max_model_params(self, bits: float = 4.0) -> float:
        """
        Estimate maximum model parameters (in billions) that can fit.

        Args:
            bits: Bits per parameter after quantization.

        Returns:
            Maximum number of parameters in billions.
        """
        usable_memory = self.available_memory_gb * 0.8  # Leave 20% headroom
        bytes_per_param = bits / 8
        max_params = (usable_memory * (1024**3)) / bytes_per_param
        return max_params / 1e9  # Convert to billions


def detect_hardware() -> HardwareProfile:
    """
    Detect system hardware capabilities.

    Returns:
        HardwareProfile with detected capabilities.
    """
    profile = HardwareProfile()

    # Basic system info
    profile.platform_name = platform.system()
    profile.arch = platform.machine()
    profile.cpu_cores = os.cpu_count() or 1

    # CPU name
    profile.cpu_name = _detect_cpu_name()

    # Memory
    mem = psutil.virtual_memory()
    profile.total_memory_gb = mem.total / (1024**3)
    profile.available_memory_gb = mem.available / (1024**3)

    # Apple Silicon detection
    profile.has_apple_silicon = (
        profile.platform_name == "Darwin" and profile.arch == "arm64"
    )
    profile.has_unified_memory = profile.has_apple_silicon

    # MPS detection (Apple Metal)
    profile.has_mps = _detect_mps()

    # CUDA detection
    profile.has_cuda, profile.cuda_devices, profile.gpu_memory_gb = _detect_cuda()

    # SSD speed estimation
    profile.ssd_read_speed_gbps = _estimate_ssd_speed()

    logger.info(f"Detected hardware: {profile.platform_name} {profile.arch}")
    logger.info(f"Memory: {profile.total_memory_gb:.1f}GB total, "
                f"{profile.available_memory_gb:.1f}GB available")

    return profile


def _detect_cpu_name() -> str:
    """Detect human-readable CPU name."""
    system = platform.system()

    if system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    elif system == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        except FileNotFoundError:
            pass

    return platform.processor() or "Unknown CPU"


def _detect_mps() -> bool:
    """Check if Apple Metal Performance Shaders are available."""
    try:
        import torch

        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        # Without PyTorch, check if we're on Apple Silicon
        return platform.system() == "Darwin" and platform.machine() == "arm64"


def _detect_cuda() -> tuple:
    """
    Detect CUDA devices.

    Returns:
        Tuple of (has_cuda, device_names, total_gpu_memory_gb).
    """
    try:
        import torch

        if torch.cuda.is_available():
            devices = []
            total_memory = 0.0
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
                devices.append(name)
                total_memory += mem
            return True, devices, total_memory
    except ImportError:
        pass

    return False, [], 0.0


def _estimate_ssd_speed() -> float:
    """
    Estimate SSD read speed.

    Returns a conservative estimate in GB/s.
    """
    system = platform.system()

    if system == "Darwin":
        # Apple Silicon Macs typically have 2.8-7.4 GB/s NVMe
        if platform.machine() == "arm64":
            return 3.0  # Conservative estimate for M-series
        return 2.0  # Older Intel Macs

    elif system == "Linux":
        # Modern NVMe SSDs: 2-7 GB/s
        return 2.0

    # Conservative default
    return 1.5


def recommend_config(profile: Optional[HardwareProfile] = None):
    """
    Recommend a Graviton configuration based on hardware profile.

    Args:
        profile: Hardware profile. If None, will auto-detect.

    Returns:
        Recommended GravitonConfig.
    """
    from graviton.core.config import GravitonConfig, DeviceType

    if profile is None:
        profile = detect_hardware()

    available = profile.available_memory_gb

    # Determine device
    if profile.has_mps:
        device = DeviceType.MPS
    elif profile.has_cuda:
        device = DeviceType.CUDA
    else:
        device = DeviceType.CPU

    # Choose quantization based on available memory
    if available >= 128:
        quant_bits = 8.0
        sparsity_ratio = 0.8
    elif available >= 64:
        quant_bits = 4.0
        sparsity_ratio = 0.6
    elif available >= 32:
        quant_bits = 4.0
        sparsity_ratio = 0.5
    elif available >= 16:
        quant_bits = 2.0
        sparsity_ratio = 0.4
    else:
        quant_bits = 1.58
        sparsity_ratio = 0.3

    config = GravitonConfig(
        quant_bits=quant_bits,
        sparsity_ratio=sparsity_ratio,
        max_memory_gb=available * 0.8,
        use_mmap=True,
        use_speculative=available >= 16,
        device=device,
    )

    logger.info(f"Recommended config: {quant_bits}-bit quantization, "
                f"{sparsity_ratio:.0%} sparsity, {device.value} device")

    return config
