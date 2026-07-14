# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: MIT
# SPDX-PackageName: translocate

"""Project an inventory into a target interpreter's install scheme.

For each inventory entry, the destination is computed under the
target's ``sysconfig`` paths (queried by spawning the target Python).
The file is then linked or copied; entry-point script wrappers are
regenerated against the target interpreter rather than linked, since
their shebangs would otherwise point at the source interpreter. After
every file lands, a fresh ``RECORD`` is written that describes the
destinations, making the cloned distribution look like a first-class
install to ``pip`` and ``uv``.

Regenerating the target ``RECORD`` is what makes the clone look
like a native install: the manifest must describe destination paths
for ``pip uninstall`` and upgrade flows to operate on the clone.
"""

from __future__ import annotations

import csv
import io
import logging
import os
from concurrent.futures import Executor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from translocate import _target
from translocate._hashing import (
    format_record_hash,
    hash_bytes_sha256,
    hash_file_sha256,
)
from translocate.inventory import (
    Category,
    Entry,
    InstallScheme,
    Inventory,
)
from translocate.materialize.linking import (
    FallbackPolicy,
    LinkMode,
    LinkOutcome,
    materialize_file,
)
from translocate.scheme import (
    category_destination,
    scheme_from_target_query,
)
from translocate.scripts import EntryPoint, parse_target, render_script

logger = logging.getLogger(__name__)

_DROPPED_DIST_INFO_LEAVES = {
    "RECORD",
    "RECORD.jws",
    "RECORD.p7s",
    "direct_url.json",
    "INSTALLER",
}


@dataclass
class MaterializeResult:
    target_prefix: Path
    target_dist_info_dir: Path
    members_materialized: int
    members_skipped_residue: List[Path] = field(default_factory=list)
    members_skipped_outside_scheme: List[Path] = field(default_factory=list)
    fallback_count: int = 0
    hash_mismatches: List[Path] = field(default_factory=list)
    excluded_malformed_entry_points: List[str] = field(default_factory=list)


def _target_dist_info_dir(
    inventory: Inventory, target_scheme: InstallScheme
) -> Path:
    name_underscore = inventory.dist_name.replace("-", "_")
    dist_info_name = f"{name_underscore}-{inventory.dist_version}.dist-info"
    root = (
        target_scheme.purelib
        if inventory.root_is_purelib
        else target_scheme.platlib
    )
    return root / dist_info_name


def _entry_destination(
    entry: Entry,
    target_scheme: InstallScheme,
    target_dist_info: Path,
    root_is_purelib: bool,
) -> Optional[Path]:
    if entry.category == Category.DIST_INFO:
        leaf = entry.rel_path.as_posix()
        if leaf in _DROPPED_DIST_INFO_LEAVES:
            return None
        return target_dist_info / entry.rel_path.as_posix()
    base = category_destination(target_scheme, entry.category, root_is_purelib)
    if base is None:
        return None
    return base / entry.rel_path.as_posix()


@dataclass(slots=True)
class _Plan:
    entry: Entry
    dest: Path
    entry_point: Optional[EntryPoint] = None


def _resolve_entry_point(
    entry: Entry, inventory: Inventory, result: MaterializeResult
) -> Optional[EntryPoint]:
    """Resolve an entry-point script entry to its parsed target.

    Returns the parsed ``EntryPoint``, or ``None`` when the
    ``entry_points.txt`` target is malformed. A malformed target's
    name is recorded on ``result`` and logged. Raises ``RuntimeError``
    when the entry carries the entry-point flag but no matching name,
    an inconsistency ``record.py`` never produces.
    """
    name = entry.content_ref.entry_point_name or entry.rel_path.stem
    target_str = inventory.entry_points.get("console_scripts", {}).get(
        name
    ) or inventory.entry_points.get("gui_scripts", {}).get(name)
    if not target_str:
        raise RuntimeError(
            f"{name!r} carries the entry-point flag but is absent from "
            f"entry_points"
        )
    ep = parse_target(name, target_str)
    if ep is None:
        result.excluded_malformed_entry_points.append(name)
        logger.warning(
            "skipping entry-point %r: malformed target %r", name, target_str
        )
    return ep


def _plan_entries(
    inventory: Inventory,
    target_scheme: InstallScheme,
    target_dist_info: Path,
    result: MaterializeResult,
) -> List[_Plan]:
    plans: List[_Plan] = []
    for entry in inventory.entries:
        if entry.flags.is_record or entry.flags.is_signature:
            continue
        if entry.flags.is_provenance:
            continue
        if entry.flags.is_pyc:
            continue
        if entry.flags.is_unrecorded_residue:
            if entry.content_ref.on_disk_path is not None:
                result.members_skipped_residue.append(
                    entry.content_ref.on_disk_path
                )
            continue
        if entry.flags.outside_known_scheme:
            if entry.content_ref.on_disk_path is not None:
                result.members_skipped_outside_scheme.append(
                    entry.content_ref.on_disk_path
                )
            continue
        if entry.flags.missing_on_disk and not entry.flags.is_entrypoint_script:
            continue
        if entry.flags.hash_mismatch and entry.content_ref.on_disk_path:
            result.hash_mismatches.append(entry.content_ref.on_disk_path)
        dest = _entry_destination(
            entry, target_scheme, target_dist_info, inventory.root_is_purelib
        )
        if dest is None:
            continue
        if entry.flags.is_entrypoint_script:
            ep = _resolve_entry_point(entry, inventory, result)
            if ep is None:
                continue
            plans.append(_Plan(entry=entry, dest=dest, entry_point=ep))
            continue
        plans.append(_Plan(entry=entry, dest=dest))
    return plans


def _materialize_one(
    plan: _Plan,
    target_scheme: InstallScheme,
    link_mode: LinkMode,
    fallback: FallbackPolicy,
    dev_cache: Dict[Path, int],
) -> Tuple[_Plan, LinkOutcome, bytes]:
    """Place one entry at its destination.

    Returns ``(plan, outcome, embedded_bytes)``. ``embedded_bytes`` is
    non-empty for regenerated entry-point wrappers (so the RECORD
    rehash can use them without re-reading the destination); for
    linked/copied files it's ``b""`` and the rehash re-reads from
    disk.
    """
    entry = plan.entry
    dest = plan.dest
    if plan.entry_point is not None:
        body = render_script(plan.entry_point, target_scheme.executable)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        try:
            os.chmod(dest, 0o755)
        except OSError as e:
            logger.warning(
                "could not set execute bit on entry-point wrapper %s: %s",
                dest,
                e,
            )
        return plan, LinkOutcome(LinkMode.COPY, False), body

    source = entry.content_ref.on_disk_path
    if source is None:
        return plan, LinkOutcome(LinkMode.COPY, False), b""
    outcome = materialize_file(
        source=source,
        dest=dest,
        mode=link_mode,
        fallback=fallback,
        dev_cache=dev_cache,
    )
    return plan, outcome, b""


def _regenerate_record(
    target_dist_info: Path,
    materialized: List[Tuple[_Plan, LinkOutcome, bytes]],
    target_scheme: InstallScheme,
    root_is_purelib: bool,
    pool: Optional[Executor],
) -> None:
    """Write a fresh ``RECORD`` describing the actual destination paths.

    Paths are RECORD-style (POSIX, relative to the install's wheel
    root). The ``materialized`` list provides each file's destination
    and any inline bytes that bypassed disk; the hash pass uses the
    inline bytes when present and re-reads from disk otherwise.
    """
    target_root = (
        target_scheme.purelib if root_is_purelib else target_scheme.platlib
    )

    def _hash_for(plan_outcome: Tuple[_Plan, LinkOutcome, bytes]):
        plan, _, embedded = plan_outcome
        if embedded:
            digest, size = hash_bytes_sha256(embedded)
        else:
            digest, size = hash_file_sha256(plan.dest)
        try:
            rel = plan.dest.relative_to(target_root).as_posix()
        except ValueError:
            rel = os.path.relpath(plan.dest, target_root).replace(os.sep, "/")
        return rel, digest, size

    if pool is not None:
        rows = list(pool.map(_hash_for, materialized))
    else:
        rows = [_hash_for(item) for item in materialized]

    target_dist_info.mkdir(exist_ok=True)
    installer_path = target_dist_info / "INSTALLER"
    installer_bytes = b"translocate\n"
    installer_path.write_bytes(installer_bytes)
    installer_digest, installer_size = hash_bytes_sha256(installer_bytes)
    try:
        installer_rel = installer_path.relative_to(target_root).as_posix()
    except ValueError:
        installer_rel = os.path.relpath(installer_path, target_root).replace(
            os.sep, "/"
        )
    rows.append((installer_rel, installer_digest, installer_size))

    record_buf = io.StringIO()
    record_writer = csv.writer(record_buf, lineterminator="\n")
    for rel, digest, size in rows:
        record_writer.writerow(
            [rel, format_record_hash("sha256", digest), str(size)]
        )
    try:
        record_rel = (
            (target_dist_info / "RECORD").relative_to(target_root).as_posix()
        )
    except ValueError:
        record_rel = os.path.relpath(
            target_dist_info / "RECORD", target_root
        ).replace(os.sep, "/")
    record_writer.writerow([record_rel, "", ""])
    (target_dist_info / "RECORD").write_bytes(
        record_buf.getvalue().encode("utf-8")
    )


def materialize(
    inventory: Inventory,
    target_python: Path,
    link_mode: LinkMode = LinkMode.HARDLINK,
    fallback: FallbackPolicy = FallbackPolicy.COPY,
    pool: Optional[Executor] = None,
) -> Tuple[MaterializeResult, InstallScheme]:
    """Clone ``inventory``'s files into ``target_python``'s install scheme.

    Returns ``(result, target_scheme)``. The result reports counts and
    any entries that were skipped (residue, outside-scheme); the
    returned target scheme is the same one used to compute
    destinations, so callers can hand it to verification without
    re-querying the target Python.
    """
    target_query = _target.query_target_scheme(target_python)
    target_scheme = scheme_from_target_query(target_query)
    target_dist_info = _target_dist_info_dir(inventory, target_scheme)

    result = MaterializeResult(
        target_prefix=target_scheme.prefix,
        target_dist_info_dir=target_dist_info,
        members_materialized=0,
    )
    plans = _plan_entries(inventory, target_scheme, target_dist_info, result)

    dev_cache: Dict[Path, int] = {}

    def _one(p: _Plan):
        return _materialize_one(
            p, target_scheme, link_mode, fallback, dev_cache
        )

    if pool is not None:
        materialized = list(pool.map(_one, plans))
    else:
        materialized = [_one(p) for p in plans]
    for _, outcome, _bytes in materialized:
        if outcome.fell_back:
            result.fallback_count += 1
    result.members_materialized = len(materialized)

    _regenerate_record(
        target_dist_info,
        materialized,
        target_scheme,
        inventory.root_is_purelib,
        pool,
    )
    return result, target_scheme
