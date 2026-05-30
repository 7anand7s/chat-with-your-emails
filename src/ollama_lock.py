"""Global lock for Ollama model calls.

Ensures only one model is loaded at a time on the GPU.
Without this, calling qwen3:8b and bge-m3 simultaneously would
compete for VRAM and cause OOM or extreme slowdown.
"""

import threading

# Single lock for all Ollama API calls
_model_lock = threading.Lock()


def get_model_lock() -> threading.Lock:
    """Get the global model lock. Use 'with get_model_lock():' to acquire."""
    return _model_lock
