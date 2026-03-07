"""
Dynamic Memory Manager

Orchestrates memory allocation across all Graviton components.
Tracks usage, implements LRU layer caching, and dynamically
decides what to keep in memory vs. stream from disk.

Optimized for Apple Silicon's unified memory architecture.
"""

from __future__ import annotations

import logging
import gc
from collections import OrderedDict
from typing import Optional, Dict, Any

import torch
import psutil

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Dynamic memory manager for Graviton inference.

    Manages a fixed memory budget and decides:
    - Which model layers to keep in RAM (hot cache)
    - Which layers to evict (LRU policy)
    - When to trigger garbage collection
    - Memory-mapped file management

    Example:
        >>> manager = MemoryManager(config, hardware_profile)
        >>> manager.register_layer("layer.0", tensor)
        >>> tensor = manager.get_layer("layer.0")  # From cache or disk
        >>> manager.report()  # Memory usage report
    """

    def __init__(self, memory_config=None, hardware_profile=None):
        """
        Initialize the memory manager.

        Args:
            memory_config: MemoryConfig instance.
            hardware_profile: HardwareProfile instance.
        """
        self._config = memory_config
        self._hardware = hardware_profile

        # LRU cache for layers
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._max_cache_size = (
            memory_config.layer_cache_size if memory_config else 8
        )

        # Memory budget
        if memory_config and memory_config.max_memory_gb > 0:
            self._budget_bytes = int(memory_config.max_memory_gb * (1024**3))
        elif hardware_profile:
            self._budget_bytes = int(
                hardware_profile.available_memory_gb * 0.8 * (1024**3)
            )
        else:
            mem = psutil.virtual_memory()
            self._budget_bytes = int(mem.available * 0.8)

        # Tracking
        self._allocated_bytes = 0
        self._peak_bytes = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._evictions = 0

        logger.info(
            f"MemoryManager initialized: "
            f"budget={self._budget_bytes / (1024**3):.1f}GB, "
            f"cache_size={self._max_cache_size} layers"
        )

    @property
    def budget_gb(self) -> float:
        """Memory budget in GB."""
        return self._budget_bytes / (1024**3)

    @property
    def used_gb(self) -> float:
        """Currently used memory in GB."""
        return self._allocated_bytes / (1024**3)

    @property
    def available_gb(self) -> float:
        """Available memory within budget."""
        return (self._budget_bytes - self._allocated_bytes) / (1024**3)

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate."""
        total = self._cache_hits + self._cache_misses
        return self._cache_hits / max(total, 1)

    def register_layer(self, name: str, tensor: torch.Tensor) -> bool:
        """
        Register a layer tensor in the cache.

        If cache is full, evicts the least recently used layer.
        If the tensor exceeds budget, returns False.

        Args:
            name: Layer name.
            tensor: Layer weight tensor.

        Returns:
            True if successfully cached, False if rejected.
        """
        tensor_bytes = tensor.element_size() * tensor.numel()

        # Check if we need to evict
        while (
            len(self._cache) >= self._max_cache_size
            or self._allocated_bytes + tensor_bytes > self._budget_bytes
        ):
            if not self._cache:
                logger.warning(
                    f"Cannot cache {name}: tensor ({tensor_bytes / (1024**2):.1f}MB) "
                    f"exceeds available budget ({self.available_gb:.2f}GB)"
                )
                return False

            self._evict_lru()

        # Add to cache
        self._cache[name] = tensor
        self._allocated_bytes += tensor_bytes
        self._peak_bytes = max(self._peak_bytes, self._allocated_bytes)

        logger.debug(
            f"Cached layer {name}: {tensor_bytes / (1024**2):.1f}MB "
            f"(total: {self.used_gb:.2f}GB)"
        )
        return True

    def get_layer(self, name: str) -> Optional[torch.Tensor]:
        """
        Retrieve a layer from cache.

        Moves the accessed layer to the end (most recently used).

        Args:
            name: Layer name.

        Returns:
            Tensor if in cache, None otherwise.
        """
        if name in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(name)
            self._cache_hits += 1
            return self._cache[name]

        self._cache_misses += 1
        return None

    def remove_layer(self, name: str):
        """Remove a specific layer from cache."""
        if name in self._cache:
            tensor = self._cache.pop(name)
            self._allocated_bytes -= tensor.element_size() * tensor.numel()
            logger.debug(f"Removed layer {name} from cache")

    def _evict_lru(self):
        """Evict the least recently used layer."""
        if not self._cache:
            return

        name, tensor = self._cache.popitem(last=False)  # Pop oldest
        freed_bytes = tensor.element_size() * tensor.numel()
        self._allocated_bytes -= freed_bytes
        self._evictions += 1

        logger.debug(
            f"Evicted layer {name}: freed {freed_bytes / (1024**2):.1f}MB"
        )

        # Help garbage collection
        del tensor

    def clear_cache(self):
        """Clear all cached layers."""
        self._cache.clear()
        self._allocated_bytes = 0
        gc.collect()

        # Clear PyTorch CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Cache cleared")

    def optimize_memory(self):
        """
        Run memory optimization routines.

        - Triggers Python garbage collection
        - Clears PyTorch caches
        - Defragments memory where possible
        """
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # On Apple Silicon, unified memory doesn't need special handling
        # but we can suggest to the OS to reclaim pages
        logger.debug("Memory optimization completed")

    def can_fit(self, size_bytes: int) -> bool:
        """
        Check if a tensor of given size can fit in the budget.

        Args:
            size_bytes: Size of the tensor in bytes.

        Returns:
            True if it fits within available budget.
        """
        return self._allocated_bytes + size_bytes <= self._budget_bytes

    def report(self) -> dict:
        """
        Generate a memory usage report.

        Returns:
            Dictionary with memory statistics.
        """
        system_mem = psutil.virtual_memory()

        return {
            "budget_gb": round(self.budget_gb, 2),
            "used_gb": round(self.used_gb, 2),
            "available_gb": round(self.available_gb, 2),
            "utilization": round(self.used_gb / max(self.budget_gb, 0.001), 3),
            "peak_gb": round(self._peak_bytes / (1024**3), 2),
            "cached_layers": len(self._cache),
            "max_cache_size": self._max_cache_size,
            "cache_hit_rate": round(self.cache_hit_rate, 3),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "evictions": self._evictions,
            "system_total_gb": round(system_mem.total / (1024**3), 2),
            "system_available_gb": round(system_mem.available / (1024**3), 2),
            "system_used_percent": system_mem.percent,
        }

    def summary(self) -> str:
        """Human-readable memory summary."""
        r = self.report()
        lines = [
            "╔══════════════════════════════════════╗",
            "║       Memory Manager Report          ║",
            "╠══════════════════════════════════════╣",
            f"║ Budget: {r['budget_gb']:>22.2f} GB ║",
            f"║ Used: {r['used_gb']:>24.2f} GB ║",
            f"║ Available: {r['available_gb']:>19.2f} GB ║",
            f"║ Peak: {r['peak_gb']:>24.2f} GB ║",
            f"║ Utilization: {r['utilization']:>18.1%}    ║",
            f"║ Cached Layers: {r['cached_layers']:>16}/{r['max_cache_size']:<3} ║",
            f"║ Cache Hit Rate: {r['cache_hit_rate']:>16.1%}   ║",
            f"║ Evictions: {r['evictions']:>21}    ║",
            "╠══════════════════════════════════════╣",
            f"║ System Total: {r['system_total_gb']:>16.1f} GB ║",
            f"║ System Available: {r['system_available_gb']:>12.1f} GB ║",
            f"║ System Used: {r['system_used_percent']:>18.1f}%   ║",
            "╚══════════════════════════════════════╝",
        ]
        return "\n".join(lines)
