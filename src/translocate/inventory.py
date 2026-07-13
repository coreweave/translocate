# SPDX-FileCopyrightText: 2026 CoreWeave
#
# SPDX-License-Identifier: MIT

"""Shared inventory data model.

An ``Inventory`` is the categorized, reconciled view of one installed
distribution. The wheel emitter and the prefix materializer both
consume it. Each entry carries everything either back-end needs to
decide what to do with the file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple


class Category(Enum):
    ROOT_PURELIB = "root-purelib"
    ROOT_PLATLIB = "root-platlib"
    SCRIPTS = "scripts"
    HEADERS = "headers"
    DATA = "data"
    DIST_INFO = "dist-info"
    OUTSIDE_KNOWN_SCHEME = "outside-known-scheme"


CATEGORY_PRIORITY: Tuple[Category, ...] = (
    Category.SCRIPTS,
    Category.HEADERS,
    Category.ROOT_PURELIB,
    Category.ROOT_PLATLIB,
    Category.DATA,
)
# The `data` category root typically resolves to the install prefix,
# which contains `scripts` (e.g. `{prefix}/bin`) and `headers` (e.g.
# `{prefix}/include/...`). Path classification must test specific
# categories before `data` so a script under the prefix isn't
# mis-classified as data.


class ContentKind(Enum):
    ON_DISK = "on-disk"
    REGENERATE_RECORD = "regenerate-record"
    REGENERATE_SCRIPT = "regenerate-script"


@dataclass(frozen=True, slots=True)
class ContentRef:
    kind: ContentKind
    on_disk_path: Optional[Path] = None
    entry_point_name: Optional[str] = None

    @classmethod
    def on_disk(cls, path: Path) -> "ContentRef":
        return cls(kind=ContentKind.ON_DISK, on_disk_path=path)

    @classmethod
    def regenerate_record(
        cls, on_disk_path: Optional[Path] = None
    ) -> "ContentRef":
        return cls(
            kind=ContentKind.REGENERATE_RECORD,
            on_disk_path=on_disk_path,
        )

    @classmethod
    def regenerate_script(
        cls, name: str, on_disk_path: Optional[Path] = None
    ) -> "ContentRef":
        return cls(
            kind=ContentKind.REGENERATE_SCRIPT,
            entry_point_name=name,
            on_disk_path=on_disk_path,
        )


@dataclass(slots=True)
class EntryFlags:
    is_record: bool = False
    is_pyc: bool = False
    is_signature: bool = False
    is_provenance: bool = False
    is_entrypoint_script: bool = False
    hash_mismatch: bool = False
    missing_on_disk: bool = False
    outside_known_scheme: bool = False
    is_unrecorded_residue: bool = False


@dataclass(slots=True)
class Entry:
    category: Category
    rel_path: PurePosixPath
    content_ref: ContentRef
    src_hash: Optional[Tuple[str, str]] = None
    src_size: Optional[int] = None
    disk_hash: Optional[Tuple[str, str]] = None
    disk_size: Optional[int] = None
    src_mode: Optional[int] = None
    flags: EntryFlags = field(default_factory=EntryFlags)


@dataclass(frozen=True, slots=True)
class InstallScheme:
    """The install-scheme directories of one Python interpreter."""

    purelib: Path
    platlib: Path
    scripts: Path
    headers: Path
    data: Path
    prefix: Path
    executable: Path
    soabi: Optional[str]
    ext_suffix: Optional[str]


@dataclass
class Inventory:
    dist_name: str
    dist_version: str
    dist_info_dir: Path
    source_scheme: InstallScheme
    root_is_purelib: bool
    # Which of purelib/platlib the source distribution treats as its
    # archive root, read from its ``WHEEL``. purelib and platlib
    # coincide on most installs and diverge under separate-platlib
    # layouts; the bit tells the wheel emitter what ``Root-Is-Purelib``
    # to write and which root to place the non-root category under, and
    # tells the materializer where the dist-info lands.
    entries: List[Entry] = field(default_factory=list)
    entry_points: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # ``entry_points`` mirrors ``entry_points.txt``: section
    # (``console_scripts``, ``gui_scripts``, ...) -> name -> the
    # ``module:attr [extras]`` target string.
