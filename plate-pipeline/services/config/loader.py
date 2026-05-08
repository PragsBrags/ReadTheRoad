"""
Configuration Loader — Load and cache pipeline config from YAML.

"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

from services.config.models import PipelineConfig

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"


def _resolve_config_path() -> Path:
    """Resolve config path, checking environment override."""
    env_path = os.environ.get("PLATE_PIPELINE_CONFIG")
    if env_path:
        return Path(env_path)
    return _CONFIG_PATH


@lru_cache(maxsize=1)
def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """
    Load and validate the pipeline configuration.

    Uses lru_cache for singleton behavior — config is loaded once
    and shared for the lifetime of the process.

    Args:
        config_path: Optional explicit path. If None, uses default
                     or PLATE_PIPELINE_CONFIG env var.

    Returns:
        Validated PipelineConfig instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        pydantic.ValidationError: If config is malformed.
    """
    path = Path(config_path) if config_path else _resolve_config_path()

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            f"Expected at: {_CONFIG_PATH}\n"
            f"Or set PLATE_PIPELINE_CONFIG env var to override."
        )

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    config = PipelineConfig(**raw)

    # Validate that the active inference mode is enabled
    active_mode = config.inference.mode.value
    mode_config = getattr(config.inference.modes, active_mode, None)
    if mode_config and not mode_config.enabled:
        raise ValueError(
            f"Inference mode '{active_mode}' is set as active but "
            f"is not enabled in config.inference.modes.{active_mode}.enabled"
        )

    return config


def reload_config(config_path: Optional[str] = None) -> PipelineConfig:
    """Force reload config (clears cache)."""
    load_config.cache_clear()
    return load_config(config_path)
