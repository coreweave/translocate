# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: MIT
# SPDX-PackageName: translocate

"""Parse RECORD into a categorized inventory of entries.

RECORD is CSV. Each row resolves to an absolute path against
the directory holding the ``.dist-info``, then goes through
``scheme.classify`` to land in one of the wheel categories.

Per-entry flags (``is_record``, ``is_pyc``, ``is_signature``,
``is_provenance``, ``is_entrypoint_script``, ``outside_known_scheme``)
are applied once here. Downstream consumers read flags rather than
re-deriving them.
"""

from __future__ import annotations

import configparser
import csv
import io
import os
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

from translocate._hashing import parse_record_hash
from translocate.errors import IncompleteMetadata, TranslocateRefused
from translocate.inventory import (
    Category,
    ContentKind,
    ContentRef,
    Entry,
    EntryFlags,
    InstallScheme,
    Inventory,
)
from translocate.locate import InstallState, LocatedDistribution
from translocate.scheme import classify, distribution_root_is_purelib

_PROVENANCE_NAMES = {"INSTALLER", "REQUESTED", "direct_url.json"}
_SIGNATURE_NAMES = {"RECORD.jws", "RECORD.p7s"}


def parse_entry_points(text: str) -> Dict[str, Dict[str, str]]:
    """Parse the INI-shaped ``entry_points.txt``.

    Returns ``section -> name -> target``, where ``target`` is the
    ``module:attr [extras]`` string on the right of the ``=``. Names
    keep their original case (entry-point names are case-sensitive).
    """
    parser = configparser.ConfigParser(delimiters=("=",))
    parser.optionxform = str  # preserve case in entry-point names
    parser.read_string(text)
    out: Dict[str, Dict[str, str]] = {}
    for section in parser.sections():
        out[section] = {
            key: value.strip() for key, value in parser.items(section)
        }
    return out


def _read_entry_points(dist_info: Path) -> Dict[str, Dict[str, str]]:
    p = dist_info / "entry_points.txt"
    if not p.exists():
        return {}
    try:
        return parse_entry_points(p.read_text(encoding="utf-8"))
    except (OSError, configparser.Error):
        return {}


def _script_entry_point_names(
    entry_points: Dict[str, Dict[str, str]],
) -> set:
    return set(entry_points.get("console_scripts", {}).keys()) | set(
        entry_points.get("gui_scripts", {}).keys()
    )


def _is_pyc(rel_path: PurePosixPath) -> bool:
    s = rel_path.as_posix()
    return s.endswith(".pyc") or "/__pycache__/" in f"/{s}"


def _resolve_record_path(
    dist_info_dir: Path, record_field: str
) -> Optional[Path]:
    """Resolve a RECORD path field to an absolute on-disk path.

    RECORD paths are POSIX, relative to the directory holding the
    ``.dist-info``. ``..`` segments are how scripts, headers, and data
    files (which live outside purelib/platlib) get referenced.

    Normalization is lexical (``os.path.normpath``): ``..`` segments
    collapse, but symlinks are left unresolved so an entry that is a
    symlink keeps its own location rather than collapsing onto its
    target. Two RECORD rows reaching the same inode through different
    paths stay distinct.

    Returns ``None`` when ``record_field`` is absolute; PEP 376 and the
    wheel spec require relative fields.
    """
    if PurePosixPath(record_field).is_absolute():
        return None
    raw = dist_info_dir.parent / record_field
    return Path(os.path.normpath(raw))


def _categorize(
    absolute: Path,
    record_field: str,
    dist_info_dir: Path,
    scheme: InstallScheme,
    root_is_purelib: bool,
) -> Tuple[Category, PurePosixPath]:
    try:
        rel_to_dist_info = absolute.relative_to(dist_info_dir)
        return Category.DIST_INFO, PurePosixPath(rel_to_dist_info.as_posix())
    except ValueError:
        pass
    category, rel = classify(absolute, scheme, root_is_purelib)
    if category == Category.OUTSIDE_KNOWN_SCHEME:
        return category, PurePosixPath(record_field)
    return category, rel


def _build_entry(
    record_field: str,
    hash_field: str,
    size_field: str,
    dist_info_dir: Path,
    scheme: InstallScheme,
    root_is_purelib: bool,
    script_ep_names: set,
) -> Entry:
    absolute = _resolve_record_path(dist_info_dir, record_field)
    if absolute is None:
        return Entry(
            category=Category.OUTSIDE_KNOWN_SCHEME,
            rel_path=PurePosixPath(record_field),
            content_ref=ContentRef(kind=ContentKind.ON_DISK, on_disk_path=None),
            flags=EntryFlags(outside_known_scheme=True),
        )
    category, rel = _categorize(
        absolute, record_field, dist_info_dir, scheme, root_is_purelib
    )

    algo, digest = parse_record_hash(hash_field)
    src_hash = (algo, digest) if digest else None
    src_size: Optional[int] = None
    if size_field:
        try:
            src_size = int(size_field)
        except ValueError:
            src_size = None

    flags = EntryFlags()
    flags.outside_known_scheme = category == Category.OUTSIDE_KNOWN_SCHEME
    flags.is_pyc = _is_pyc(rel)

    is_record_self = (
        category == Category.DIST_INFO
        and rel.as_posix() == "RECORD"
        and not digest
    )
    flags.is_record = is_record_self

    if category == Category.DIST_INFO:
        leaf = rel.as_posix()
        if leaf in _SIGNATURE_NAMES:
            flags.is_signature = True
        if leaf in _PROVENANCE_NAMES:
            flags.is_provenance = True

    content_ref = (
        ContentRef.regenerate_record(on_disk_path=absolute)
        if is_record_self
        else ContentRef.on_disk(absolute)
    )

    if category == Category.SCRIPTS and rel.name:
        stem = PurePosixPath(rel.name).stem
        # A scripts-category entry is a synthesized entry-point wrapper
        # when its filename stem matches a name in ``entry_points.txt``
        # under ``[console_scripts]`` or ``[gui_scripts]``.
        if stem in script_ep_names:
            flags.is_entrypoint_script = True
            content_ref = ContentRef.regenerate_script(
                stem, on_disk_path=absolute
            )

    return Entry(
        category=category,
        rel_path=rel,
        content_ref=content_ref,
        src_hash=src_hash,
        src_size=src_size,
        flags=flags,
    )


def _parse_record_csv(text: str) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    reader = csv.reader(io.StringIO(text))
    for raw_row in reader:
        if not raw_row:
            continue
        path = raw_row[0]
        hash_value = raw_row[1] if len(raw_row) > 1 else ""
        size_value = raw_row[2] if len(raw_row) > 2 else ""
        rows.append((path, hash_value, size_value))
    return rows


def _build_from_pep376(
    located: LocatedDistribution, scheme: InstallScheme
) -> Inventory:
    record_text = (located.dist_info_dir / "RECORD").read_text(encoding="utf-8")
    entry_points = _read_entry_points(located.dist_info_dir)
    script_ep_names = _script_entry_point_names(entry_points)
    root_is_purelib = distribution_root_is_purelib(
        located.dist_info_dir, scheme
    )
    entries = [
        _build_entry(
            path,
            hash_value,
            size_value,
            located.dist_info_dir,
            scheme,
            root_is_purelib,
            script_ep_names,
        )
        for path, hash_value, size_value in _parse_record_csv(record_text)
    ]
    return Inventory(
        dist_name=located.canonical_name,
        dist_version=located.version,
        dist_info_dir=located.dist_info_dir,
        source_scheme=scheme,
        root_is_purelib=root_is_purelib,
        entries=entries,
        entry_points=entry_points,
    )


def _build_from_installed_files(
    located: LocatedDistribution, scheme: InstallScheme
) -> Inventory:
    text = (located.dist_info_dir / "installed-files.txt").read_text(
        encoding="utf-8"
    )
    entry_points = _read_entry_points(located.dist_info_dir)
    script_ep_names = _script_entry_point_names(entry_points)
    root_is_purelib = distribution_root_is_purelib(
        located.dist_info_dir, scheme
    )
    entries: List[Entry] = []
    for line in text.splitlines():
        path = line.strip()
        if not path:
            continue
        entries.append(
            _build_entry(
                path,
                "",
                "",
                located.dist_info_dir,
                scheme,
                root_is_purelib,
                script_ep_names,
            )
        )
    return Inventory(
        dist_name=located.canonical_name,
        dist_version=located.version,
        dist_info_dir=located.dist_info_dir,
        source_scheme=scheme,
        root_is_purelib=root_is_purelib,
        entries=entries,
        entry_points=entry_points,
    )


def parse_record(
    located: LocatedDistribution, scheme: InstallScheme
) -> Inventory:
    """Build an inventory from the located distribution's manifest.

    Reads ``RECORD`` for PEP 376 wheel installs and
    ``installed-files.txt`` for the legacy setuptools layout. The
    latter has no recorded hashes, so reconciliation skips its
    hash-mismatch check for those entries.
    """
    if located.install_state == InstallState.PEP376_WHEEL:
        return _build_from_pep376(located, scheme)
    if located.install_state == InstallState.LEGACY_INSTALLED_FILES:
        return _build_from_installed_files(located, scheme)
    raise TranslocateRefused(
        IncompleteMetadata(
            dist_info=located.dist_info_dir,
            detail=f"unsupported install state: {located.install_state}",
        )
    )
