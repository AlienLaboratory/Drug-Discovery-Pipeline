"""Logging configuration and progress tracking utilities."""

import logging
import sys
from pathlib import Path
from typing import Iterable, Optional, TypeVar

from tqdm import tqdm

T = TypeVar("T")

_configured = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
) -> logging.Logger:
    global _configured
    logger = logging.getLogger("drugflow")

    if _configured:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

    _configured = True
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"drugflow.{name}")


def progress_bar(
    iterable: Iterable[T],
    total: Optional[int] = None,
    desc: str = "",
    disable: bool = False,
) -> Iterable[T]:
    return tqdm(iterable, total=total, desc=desc, disable=disable, unit="mol")
