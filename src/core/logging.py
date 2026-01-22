"""
Lightweight logging helper so every module shares the same formatting.
"""

from __future__ import annotations

import logging
import os
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a configured logger. The log level can be overridden via the
    LOGLEVEL environment variable.
    """

    level_name = os.getenv("LOGLEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root_configured = logging.getLogger().handlers
    if not root_configured:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        )
    else:
        logging.getLogger().setLevel(level)

    return logging.getLogger(name if name else __name__)

