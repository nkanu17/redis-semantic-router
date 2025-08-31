"""Simple logging for POC."""

import logging
import sys
from typing import Any


def get_logger(name: str = "semantic_router") -> logging.Logger:
    """Get simple logger with basic console output."""
    # Clear any existing handlers to avoid duplicates
    logger = logging.getLogger(name)
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(name)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent propagation to root logger

    return logger


def log_metrics(
    logger: logging.Logger, metrics: dict[str, Any], prefix: str = ""
) -> None:
    """Log basic metrics."""
    if prefix:
        prefix = f"{prefix} - "

    logger.info(f"{prefix}=== METRICS ===")

    # Log key metrics only
    key_metrics = [
        "total_articles",
        "successful_articles",
        "accuracy",
        "total_cost",
        "total_latency",
        "f1_macro",
    ]

    for key in key_metrics:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                if "cost" in key:
                    logger.info(f"{prefix}{key}: ${value:.4f}")
                elif "accuracy" in key or "f1" in key:
                    logger.info(f"{prefix}{key}: {value:.3f}")
                else:
                    logger.info(f"{prefix}{key}: {value:.2f}")
            else:
                logger.info(f"{prefix}{key}: {value}")


def log_classification_report(
    logger: logging.Logger, metrics: dict[str, Any], prefix: str = ""
) -> None:
    """Log classification summary."""
    if "accuracy" in metrics:
        if prefix:
            prefix = f"{prefix} - "
        logger.info(f"{prefix}Classification accuracy: {metrics['accuracy']:.3f}")


def log_classification_sample(
    logger: logging.Logger,
    article_id: int | str,
    prediction: str,
    actual: str | None = None,
) -> None:
    """Log single classification result."""
    if actual:
        correct = "✓" if prediction.lower() == actual.lower() else "✗"
        logger.info(f"Article {article_id}: {prediction} | actual: {actual} {correct}")
    else:
        logger.info(f"Article {article_id}: {prediction}")
