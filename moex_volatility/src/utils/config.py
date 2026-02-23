"""Configuration loader and manager."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Configuration manager for loading and accessing YAML configs."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize config manager.

        Args:
            config_path: Path to YAML config file. If None, loads default.
        """
        self._config: Dict[str, Any] = {}

        if config_path:
            self.load(config_path)

    def load(self, config_path: str) -> None:
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot-notation key.

        Args:
            key: Dot-notation key (e.g., 'models.har.use_log')
            default: Default value if key not found.

        Returns:
            Config value or default.
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Get config value by key."""
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None

    @property
    def config(self) -> Dict[str, Any]:
        """Get full config dictionary."""
        return self._config


def load_model_config() -> Config:
    """Load model configuration from default path."""
    config_path = Path(__file__).parent.parent.parent / "configs" / "model_config.yaml"
    return Config(str(config_path))


def load_training_config() -> Config:
    """Load training configuration from default path."""
    config_path = Path(__file__).parent.parent.parent / "configs" / "training_config.yaml"
    return Config(str(config_path))


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
