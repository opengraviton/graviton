"""
Universal Model Loader

Provides a unified interface for loading models from various formats:
- HuggingFace Transformers
- GGUF (llama.cpp)
- SafeTensors
- Raw PyTorch bins
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Universal model loader that auto-detects formats and structure.
    """

    @staticmethod
    def inspect(model_path: str) -> dict:
        """
        Inspect a model to determine its format and architecture.

        Args:
            model_path: Path to model file or directory.

        Returns:
            Dictionary with metadata about the model.
        """
        path = Path(model_path)
        metadata = {
            "path": str(path),
            "format": "unknown",
            "architecture": "unknown",
            "num_parameters": 0,
            "size_bytes": 0,
        }

        if not path.exists():
            return metadata

        if path.is_dir():
            if list(path.glob("*.safetensors")):
                metadata["format"] = "safetensors (sharded)"
                metadata["size_bytes"] = sum(f.stat().st_size for f in path.glob("*.safetensors"))
            elif list(path.glob("*.bin")):
                metadata["format"] = "pytorch (sharded)"
                metadata["size_bytes"] = sum(f.stat().st_size for f in path.glob("*.bin"))
                
            # Check for config.json to determine architecture
            config_file = path / "config.json"
            if config_file.exists():
                try:
                    import json
                    with open(config_file) as f:
                        config = json.load(f)
                    metadata["architecture"] = config.get("architectures", ["unknown"])[0]
                    metadata["config"] = config
                except Exception:
                    pass

        elif path.suffix == ".gguf":
            metadata["format"] = "gguf"
            metadata["size_bytes"] = path.stat().st_size
        elif path.suffix == ".safetensors":
            metadata["format"] = "safetensors"
            metadata["size_bytes"] = path.stat().st_size
        elif path.suffix in (".bin", ".pt", ".pth"):
            metadata["format"] = "pytorch"
            metadata["size_bytes"] = path.stat().st_size

        return metadata

    @classmethod
    def load_config(cls, model_path: str) -> dict:
        """Load model configuration."""
        path = Path(model_path)
        
        # HuggingFace standard config.json
        config_path = path / "config.json" if path.is_dir() else path.parent / "config.json"
        
        if config_path.exists():
            import json
            with open(config_path) as f:
                return json.load(f)
                
        # If no config found, return a generic one
        logger.warning(f"No config.json found for {model_path}. Using generic config.")
        return {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
        }

class HuggingFaceLoader:
    """
    Downloads and manages models from the Hugging Face Hub natively.
    """
    
    @staticmethod
    def ensure_local(model_id: str, local_dir: Optional[str] = None) -> str:
        """
        Ensures a model is downloaded from HF Hub to a local cache.
        If it already exists in the cache, it bypasses the download.
        
        Args:
            model_id: The HuggingFace ID (e.g., 'mistralai/Mixtral-8x22B-v0.1')
            local_dir: Optional specific directory. If None, uses HF default cache.
            
        Returns:
            The absolute path to the downloaded model directory.
        """
        # If a local absolute/relative path was provided and exists, just return it.
        if Path(model_id).exists() and Path(model_id).is_dir():
            if list(Path(model_id).glob("*.safetensors")) or list(Path(model_id).glob("*.bin")):
                return str(Path(model_id).absolute())

        try:
            from huggingface_hub import snapshot_download
            logger.info(f"Connecting to HuggingFace Hub for model: {model_id}...")
            
            # We only download safetensors, config, and tokenizer files to save time/bandwidth
            path = snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*"],
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
            )
            
            logger.info(f"Model successfully located/downloaded at: {path}")
            return path
            
        except ImportError:
            logger.error("huggingface_hub is not installed. Run `pip install huggingface_hub`.")
            raise
        except Exception as e:
            logger.error(f"Failed to fetch model from Hugging Face Hub: {e}")
            raise
