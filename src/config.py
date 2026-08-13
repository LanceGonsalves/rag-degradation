"""Configuration loading.

One YAML file, read by every entry point, so a run is described entirely by
`config.yaml` plus the command line. Environment variables override individual keys
where CI or a different machine needs to differ without editing tracked config.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"

# Environment overrides. Kept explicit rather than a generic prefix scheme so that
# reading this file tells you exactly what can be changed from outside.
ENV_OVERRIDES = {
    "RAG_SEED": ("seed", int),
    "RAG_MAX_PER_QUERY": ("corpus.max_per_query", int),
    "RAG_CHUNK_SIZE": ("chunking.chunk_size", int),
    "RAG_CHUNK_OVERLAP": ("chunking.chunk_overlap", int),
    "RAG_TOP_K": ("retrieval.top_k", int),
    "RAG_EMBEDDING_MODEL": ("retrieval.embedding_model", str),
}


class Config:
    """Dotted-path access over the parsed YAML."""

    def __init__(self, data: dict[str, Any], path: Path):
        self._data = data
        self.path = path

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        """Like get(), but fails loudly rather than silently returning None.

        Used for values where a typo in config.yaml would otherwise produce a run
        that completes and reports meaningless numbers.
        """
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise KeyError(f"{dotted!r} missing from {self.path}")
        return value

    def path_for(self, dotted: str) -> Path:
        """Resolve a configured path relative to the project root."""
        raw = self.require(dotted)
        p = Path(raw)
        return p if p.is_absolute() else ROOT / p

    @property
    def seed(self) -> int:
        return int(self.get("seed", 42))

    def __repr__(self) -> str:
        return f"Config({self.path})"


def _set_dotted(data: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config at {cfg_path}")

    data = yaml.safe_load(cfg_path.read_text()) or {}

    for env_name, (dotted, caster) in ENV_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw is not None:
            _set_dotted(data, dotted, caster(raw))

    return Config(data, cfg_path)


def set_seed(seed: int) -> None:
    """Seed every source of randomness this project uses.

    numpy and torch are imported lazily: the ingestion stage needs neither, and
    requiring them here would make `python -m src.ingest.fetch` depend on a GPU stack
    it never touches.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
