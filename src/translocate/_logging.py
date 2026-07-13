# SPDX-FileCopyrightText: 2026 CoreWeave
#
# SPDX-License-Identifier: MIT

"""Configure the ``translocate`` logger.

Adds a ``DETAIL`` level between ``DEBUG`` and ``INFO`` and maps
``-v`` / ``-q`` repetition counts to logger levels. The configured
handler writes to stderr.
"""

from __future__ import annotations

import logging
import sys

DETAIL = 15

logging.addLevelName(DETAIL, "DETAIL")

_LOGGER_NAME = "translocate"


def configure(verbose: int, quiet: int) -> None:
    """Apply the verbosity-derived level to ``translocate``'s logger."""
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(_level_for(verbose, quiet))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.propagate = False


def _level_for(verbose: int, quiet: int) -> int:
    delta = max(-2, min(2, verbose - quiet))
    return {
        2: logging.DEBUG,
        1: DETAIL,
        0: logging.INFO,
        -1: logging.WARNING,
        -2: logging.ERROR,
    }[delta]
