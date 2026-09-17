"""
utils.py — Shared utilities for the RTA project: logging, GPU memory
management, config loading, and disk caching.
"""

import os
import gc
import json
import pickle
import logging
from pathlib import Path
from typing import Any, Optional

import torch


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


log = get_logger("rta.utils")


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def setup_environment():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cache_dir = get_project_root() / ".hf_cache"
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
    os.environ.setdefault("HF_HOME", str(cache_dir))
    log.info("Environment configured for RTX 2050")


def save_json(data: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(data: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


setup_environment()
