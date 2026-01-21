#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from pathlib import Path

def setup_logger(log_path: Path, logger_name: str = 'default_logger', verbose: bool = True) -> logging.Logger:
    """
    Sets up a logger that writes to a file and optionally to the console.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Stream handler (for console output)
    if verbose:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger
