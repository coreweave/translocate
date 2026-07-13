# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: MIT
# SPDX-PackageName: translocate

"""Project one file from a source path to a destination path.

Picks between hardlink, symlink, and copy. A hardlink request that
crosses filesystems is detected from ``os.link`` raising ``EXDEV`` and
falls back per the caller's ``FallbackPolicy``; symlinks and copies
are always valid.

A hardlink resolves a symlinked source to its target inode, so a
recorded file that happens to be a symlink lands as a durable link to
the real bytes rather than a link to the symlink. Symlink mode points
the destination at the source as given. All three modes carry the
source's mode bits to the destination.
"""

from __future__ import annotations

import errno
import os
import shutil
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional


class LinkMode(Enum):
    HARDLINK = "hardlink"
    SYMLINK = "symlink"
    COPY = "copy"


class FallbackPolicy(Enum):
    ERROR = "error"
    COPY = "copy"


@dataclass(frozen=True, slots=True)
class LinkOutcome:
    mode_used: LinkMode
    fell_back: bool


if sys.version_info >= (3, 14):

    def _hardlink(source: Path, dest: Path) -> None:
        os.link(source, dest, follow_symlinks=True)

elif os.link in os.supports_dir_fd:

    def _hardlink(source: Path, dest: Path) -> None:
        # Below 3.14, ``follow_symlinks=True`` only reaches ``linkat``
        # when a ``dir_fd`` is supplied; without it ``os.link`` links
        # the symlink itself.
        parent_fd = os.open(dest.parent, os.O_RDONLY)
        try:
            os.link(
                source,
                dest.name,
                dst_dir_fd=parent_fd,
                follow_symlinks=True,
            )
        finally:
            os.close(parent_fd)

else:

    def _hardlink(source: Path, dest: Path) -> None:
        os.link(os.path.realpath(source), dest)


def _symlink(source: Path, dest: Path) -> None:
    os.symlink(source, dest)


def _copy(source: Path, dest: Path) -> None:
    shutil.copy2(source, dest)


def _fall_back(
    source: Path, dest: Path, fallback: FallbackPolicy
) -> LinkOutcome:
    if fallback == FallbackPolicy.ERROR:
        raise OSError(
            f"hardlink unavailable across filesystems ({source} -> {dest})"
        )
    _copy(source, dest)
    return LinkOutcome(mode_used=LinkMode.COPY, fell_back=True)


def materialize_file(
    source: Path,
    dest: Path,
    mode: LinkMode,
    fallback: FallbackPolicy = FallbackPolicy.COPY,
    dev_cache: Optional[Dict[Path, int]] = None,
) -> LinkOutcome:
    """Place ``source`` at ``dest`` using ``mode``, falling back per policy.

    Creates any missing parent directories. A hardlink that crosses
    filesystems (``os.link`` raises ``EXDEV``) falls back to copying
    when ``fallback=FallbackPolicy.COPY``, or raises ``OSError`` when
    ``fallback=FallbackPolicy.ERROR``. The returned outcome records
    which mode the call actually used.

    ``dev_cache`` maps a destination parent to its ``st_dev``. Once a
    parent is known to differ from a source's filesystem, later files
    headed there skip the doomed link attempt.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if mode == LinkMode.SYMLINK:
        _symlink(source, dest)
        return LinkOutcome(mode_used=LinkMode.SYMLINK, fell_back=False)
    if mode == LinkMode.COPY:
        _copy(source, dest)
        return LinkOutcome(mode_used=LinkMode.COPY, fell_back=False)

    if dev_cache is not None:
        cached = dev_cache.get(dest.parent)
        if cached is not None and os.stat(source).st_dev != cached:
            return _fall_back(source, dest, fallback)

    try:
        _hardlink(source, dest)
        return LinkOutcome(mode_used=LinkMode.HARDLINK, fell_back=False)
    except OSError as e:
        if e.errno != errno.EXDEV:
            raise
        if dev_cache is not None:
            dev_cache[dest.parent] = os.stat(dest.parent).st_dev
        return _fall_back(source, dest, fallback)
