# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: MIT
# SPDX-PackageName: translocate

"""Verify a cloned prefix against its source inventory.

``verify_clone`` runs structural checks: the target's
``importlib.metadata`` must find the cloned distribution at the
expected version, and the regenerated ``RECORD`` must describe the
files actually on disk. The default check confirms every recorded path
exists at the expected size; ``verify_hashes`` additionally re-hashes
each file.

The optional smoke-import probe (``import <module>`` through the target
interpreter) is reported separately. An import failure after a clean
structural check usually indicates a runtime-environment gap (missing
system libraries or untranslocated deps) rather than a defect in the
clone, so it's advisory.
"""

from __future__ import annotations

import csv
import io
import logging
from concurrent.futures import Executor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from translocate import _target
from translocate._hashing import hash_file_sha256, parse_record_hash
from translocate.inventory import Category, Inventory
from translocate.locate import canonicalize

logger = logging.getLogger(__name__)


@dataclass
class VerifyResult:
    ok: bool
    checks_passed: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    smoke_import_ok: Optional[bool] = None
    smoke_import_error: Optional[str] = None


def verify_clone(
    inventory: Inventory,
    target_dist_info_dir: Path,
    target_python: Path,
    smoke_test_import: bool = False,
    smoke_import_module: Optional[str] = None,
    verify_hashes: bool = False,
    pool: Optional[Executor] = None,
) -> VerifyResult:
    """Confirm the cloned distribution is internally consistent at the target.

    Checks that the target's ``importlib.metadata`` finds the
    distribution at the expected version, and that the regenerated
    ``RECORD`` describes the files actually on disk. By default each
    recorded path is checked for existence and matching size;
    ``verify_hashes`` re-hashes every file instead. The optional
    smoke-import probe is run last and reported separately.
    """
    result = VerifyResult(ok=True)

    target_version = _target.query_target_version(
        target_python, inventory.dist_name
    )
    if target_version is None:
        result.ok = False
        result.failures.append(
            f"importlib.metadata.version({inventory.dist_name!r}) "
            f"raised PackageNotFoundError in target"
        )
    elif target_version != inventory.dist_version:
        result.ok = False
        result.failures.append(
            f"target reports version {target_version!r} but "
            f"clone provided {inventory.dist_version!r}"
        )
    else:
        result.checks_passed.append(
            f"target metadata version matches ({target_version})"
        )

    record_path = target_dist_info_dir / "RECORD"
    if not record_path.exists():
        result.ok = False
        result.failures.append(f"regenerated RECORD not found at {record_path}")
    else:
        failures = _verify_record_consistency(record_path, verify_hashes, pool)
        if failures:
            result.ok = False
            result.failures.extend(failures)
        else:
            result.checks_passed.append(
                "regenerated RECORD is internally consistent"
            )

    if smoke_test_import:
        result.smoke_import_ok, result.smoke_import_error = _smoke_import(
            target_python, smoke_import_module, inventory
        )
    return result


def _smoke_import(
    target_python: Path,
    smoke_import_module: Optional[str],
    inventory: Inventory,
) -> Tuple[Optional[bool], Optional[str]]:
    """Try to import the distribution through the target interpreter.

    Returns ``(ok, error)``: the first candidate module that imports
    wins; otherwise the first failure's message.
    """
    if smoke_import_module:
        candidates = [smoke_import_module]
    else:
        candidates = top_level_importable_names(inventory)
    if not candidates:
        return False, (
            "no top-level importable module could be derived from the "
            "inventory"
        )
    first_error = ""
    for name in candidates:
        ok, err = _target.smoke_import(target_python, name)
        if ok:
            return True, None
        logger.debug("smoke import candidate %r failed: %s", name, err)
        if not first_error:
            first_error = f"{name}: {err}"
    return False, first_error


def top_level_importable_names(inventory: Inventory) -> List[str]:
    """List the importable module names a distribution ships.

    Uses ``top_level.txt`` when the distribution provides one and
    falls back to scanning the inventory for top-level
    ``<name>/__init__.py`` package directories and bare ``<name>.py``
    modules at the purelib or platlib root. Underscore-prefixed names
    are dropped from both paths since they conventionally identify
    private helpers rather than the package's main import target.

    The list is what the smoke-import probe tries in order; the first
    name that imports successfully wins. Candidates whose canonical
    form matches the distribution name come first.
    """
    top_level = _read_top_level(inventory)
    if top_level:
        return _prioritize(top_level, inventory.dist_name)
    candidates: List[str] = []
    seen: set = set()
    for entry in inventory.entries:
        if entry.category not in (
            Category.ROOT_PURELIB,
            Category.ROOT_PLATLIB,
        ):
            continue
        if entry.flags.is_unrecorded_residue:
            continue
        parts = entry.rel_path.parts
        if not parts:
            continue
        if len(parts) >= 2 and parts[-1] == "__init__.py":
            name = parts[0]
        elif len(parts) == 1 and parts[0].endswith(".py"):
            name = parts[0][: -len(".py")]
        else:
            continue
        if name.startswith("_") or "-" in name:
            continue
        if not name.isidentifier():
            continue
        if name in seen:
            continue
        seen.add(name)
        candidates.append(name)
    return _prioritize(candidates, inventory.dist_name)


def _prioritize(candidates: List[str], dist_name: str) -> List[str]:
    """Put canonical-name matches first and preserve order otherwise."""
    target = canonicalize(dist_name)
    matches = [c for c in candidates if canonicalize(c) == target]
    others = [c for c in candidates if canonicalize(c) != target]
    return matches + others


def _read_top_level(inventory: Inventory) -> List[str]:
    path = inventory.dist_info_dir / "top_level.txt"
    if not path.exists():
        return []
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name or name.startswith("#") or name.startswith("_"):
            continue
        if name.isidentifier():
            out.append(name)
    return out


def _verify_record_consistency(
    record_path: Path, verify_hashes: bool, pool: Optional[Executor]
) -> List[str]:
    """Return inconsistencies between ``record_path`` and disk.

    An empty list means every recorded path exists at its destination
    with the recorded size (and, when ``verify_hashes`` is set, the
    recorded hash).
    """
    text = record_path.read_text(encoding="utf-8")
    # RECORD's paths are relative to the wheel root, and RECORD itself
    # lives one level down at ``.dist-info/RECORD``, so the wheel root
    # is two ``parent`` hops up.
    target_root = record_path.parent.parent
    rows: List[Tuple[str, str, str]] = []
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        rows.append(
            (
                row[0],
                row[1] if len(row) > 1 else "",
                row[2] if len(row) > 2 else "",
            )
        )

    def _check_row(row: Tuple[str, str, str]) -> Optional[str]:
        path_field, hash_field, size_field = row
        if not hash_field and path_field.endswith("RECORD"):
            # RECORD lists itself with empty hash and size; reading
            # the file is itself the existence proof.
            return None
        absolute = (target_root / path_field).resolve()
        if not absolute.exists():
            return f"missing file at {absolute}"
        if size_field:
            try:
                expected_size = int(size_field)
            except ValueError:
                expected_size = None
            if (
                expected_size is not None
                and absolute.stat().st_size != expected_size
            ):
                return f"size mismatch at {absolute}"
        if verify_hashes:
            _, digest = parse_record_hash(hash_field)
            if digest:
                observed, _ = hash_file_sha256(absolute)
                if observed != digest:
                    return f"hash mismatch at {absolute}"
        return None

    failures: List[str] = []
    if pool is not None:
        for failure in pool.map(_check_row, rows):
            if failure is not None:
                failures.append(failure)
    else:
        for row in rows:
            failure = _check_row(row)
            if failure is not None:
                failures.append(failure)
    return failures
