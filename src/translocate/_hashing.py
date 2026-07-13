# SPDX-FileCopyrightText: 2026 CoreWeave
#
# SPDX-License-Identifier: MIT

"""sha256 helpers in the RECORD wire format.

RECORD lines encode hashes as ``sha256=<urlsafe-b64 stripped of '='>``.
"""

from __future__ import annotations

import base64
import hashlib
import threading
from pathlib import Path
from typing import Tuple

CHUNK_SIZE = 1 << 20

# Each pool worker reuses one read buffer for its lifetime instead of
# allocating a fresh chunk per file. ``threading.local`` keeps the
# buffer private per worker, so no buffer is shared across a
# concurrent read.
_local = threading.local()


def _read_buffer() -> bytearray:
    buf = getattr(_local, "buf", None)
    if buf is None:
        buf = bytearray(CHUNK_SIZE)
        _local.buf = buf
    return buf


def _b64_nopad(digest: bytes) -> str:
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def hash_file_sha256(path: Path) -> Tuple[str, int]:
    """Return (urlsafe-b64-nopad sha256, size in bytes)."""
    h = hashlib.sha256()
    size = 0
    buf = _read_buffer()
    with open(path, "rb") as f, memoryview(buf) as view:
        while n := f.readinto(buf):
            h.update(view[:n])
            size += n
    return _b64_nopad(h.digest()), size


def hash_bytes_sha256(data: bytes) -> Tuple[str, int]:
    return _b64_nopad(hashlib.sha256(data).digest()), len(data)


def format_record_hash(algo: str, digest: str) -> str:
    return f"{algo}={digest}"


def parse_record_hash(value: str) -> Tuple[str, str]:
    """Split ``algo=digest`` into ``(algo, digest)``.

    Empty input yields ``("", "")``; an input with no ``=`` yields
    ``("", value)`` so callers can still see something was there.
    """
    if not value:
        return ("", "")
    if "=" not in value:
        return ("", value)
    algo, _, digest = value.partition("=")
    return (algo, digest)
