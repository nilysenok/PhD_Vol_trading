"""Logging utilities for the volatility forecasting project."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "moex_volatility",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """Set up a logger with console and optional file handlers.

    Args:
        name: Logger name.
        level: Logging level.
        log_file: Path to log file. If None, only console output.
        console: Whether to output to console.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Format
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "moex_volatility") -> logging.Logger:
    """Get existing logger or create new one.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)

    # If logger has no handlers, set up default
    if not logger.handlers:
        return setup_logger(name)

    return logger


class TrainingLogger:
    """Logger specifically for training progress."""

    def __init__(self, name: str = "training", log_file: Optional[str] = None):
        """Initialize training logger.

        Args:
            name: Logger name.
            log_file: Path to log file.
        """
        self.logger = setup_logger(f"moex_volatility.{name}", log_file=log_file)
        self.start_time = None

    def start_training(self, model_name: str, ticker: str, n_samples: int) -> None:
        """Log training start."""
        self.start_time = datetime.now()
        self.logger.info(f"=" * 60)
        self.logger.info(f"Starting training: {model_name} for {ticker}")
        self.logger.info(f"Training samples: {n_samples:,}")
        self.logger.info(f"=" * 60)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        metrics: Optional[dict] = None,
    ) -> None:
        """Log epoch results."""
        msg = f"Epoch {epoch:4d} | Train Loss: {train_loss:.6f}"
        if val_loss is not None:
            msg += f" | Val Loss: {val_loss:.6f}"
        if metrics:
            for k, v in metrics.items():
                msg += f" | {k}: {v:.4f}"
        self.logger.info(msg)

    def log_fold(self, fold: int, total_folds: int, metrics: dict) -> None:
        """Log cross-validation fold results."""
        self.logger.info(f"Fold {fold}/{total_folds} completed:")
        for k, v in metrics.items():
            self.logger.info(f"  {k}: {v:.6f}")

    def end_training(self, final_metrics: dict) -> None:
        """Log training end."""
        duration = datetime.now() - self.start_time if self.start_time else None
        self.logger.info(f"=" * 60)
        self.logger.info("Training completed!")
        if duration:
            self.logger.info(f"Duration: {duration}")
        self.logger.info("Final metrics:")
        for k, v in final_metrics.items():
            self.logger.info(f"  {k}: {v:.6f}")
        self.logger.info(f"=" * 60)
