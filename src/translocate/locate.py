# SPDX-FileCopyrightText: 2026 CoreWeave
#
# SPDX-License-Identifier: MIT

"""Resolve a distribution name to a validated, translocatable handle.

Looks up the distribution's ``.dist-info`` directory through
``importlib.metadata``, classifies the install layout, and returns
either a ``LocatedDistribution`` or a typed ``Refusal`` describing why
the install cannot be enumerated faithfully.

Editable installs and legacy egg-info layouts are refused outright.
An editable RECORD describes the import shim, not the source tree it
points at. An egg-info ``SOURCES.txt`` lists build inputs, not
install outputs.
"""

from __future__ import annotations

import csv
import difflib
import io
import json
import re
from dataclasses import dataclass
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import List, Optional, Union

from translocate.errors import (
    DistributionNotFound,
    EditableInstall,
    EggInfoOnly,
    IncompleteMetadata,
    InvalidDistributionName,
    Refusal,
)

_NORMALIZE_RE = re.compile(r"[-_.]+")

# PEP 426 / PEP 503: a distribution name starts and ends with an
# alphanumeric, and contains only ASCII letters, digits, ``-``, ``_``,
# and ``.``. Single-character names like ``X`` are valid.
_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


def canonicalize(name: str) -> str:
    """PEP 503 name normalization."""
    return _NORMALIZE_RE.sub("-", name).lower()


def validate_name(name: str) -> Optional[InvalidDistributionName]:
    """Validate a distribution name against PEP 503 / PEP 426.

    Returns ``None`` when ``name`` is well-formed, or an
    ``InvalidDistributionName`` refusal naming the first observed
    rule violation.
    """
    if not name:
        return InvalidDistributionName(name=name, reason="name is empty")
    if name.strip() != name:
        return InvalidDistributionName(
            name=name, reason="name has leading or trailing whitespace"
        )
    for forbidden, hint in (
        ("=", "a version specifier; pass just the name"),
        ("<", "a version specifier; pass just the name"),
        (">", "a version specifier; pass just the name"),
        ("!", "a version specifier; pass just the name"),
        ("~", "a version specifier; pass just the name"),
        ("[", "an extras bracket; pass just the name"),
        (";", "an environment marker; pass just the name"),
        (" ", "whitespace inside a name"),
        ("/", "a path separator"),
        ("\\", "a path separator"),
    ):
        if forbidden in name:
            return InvalidDistributionName(
                name=name,
                reason=f"contains {forbidden!r}; that looks like {hint}",
            )
    if not _VALID_NAME_RE.match(name):
        return InvalidDistributionName(
            name=name,
            reason=(
                "must start and end with a letter or digit and contain "
                "only ASCII letters, digits, '.', '-', '_'"
            ),
        )
    return None


class InstallState(Enum):
    PEP376_WHEEL = "pep376-wheel"
    LEGACY_INSTALLED_FILES = "legacy-installed-files"


@dataclass(frozen=True)
class LocatedDistribution:
    name: str
    canonical_name: str
    version: str
    dist_info_dir: Path
    install_state: InstallState


def _all_installed_names() -> List[str]:
    names: List[str] = []
    for d in metadata.distributions():
        if d.metadata is None:
            continue
        n = d.metadata["Name"]
        if n:
            names.append(n)
    return names


def _dist_info_dir_of(dist: "metadata.Distribution") -> Optional[Path]:
    """Return the dist-info directory for ``dist``, or ``None``.

    Reads ``Distribution._path``, the attribute that
    ``importlib.metadata.PathDistribution`` exposes for filesystem-
    backed distributions.
    """
    raw = getattr(dist, "_path", None)
    if raw is not None:
        return Path(raw)
    return None


def _read_text(dist_info: Path, name: str) -> Optional[str]:
    p = dist_info / name
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return None


def _detect_editable(
    dist_info: Path, dist_name: str, dist_version: str
) -> Optional[str]:
    """Identify which editable-install form this dist uses, or ``None``.

    Three modern forms exist:

    - ``direct_url.json`` with ``dir_info.editable == true``, the pip
      and PEP 610 convention.
    - PEP 660 RECORD listing ``__editable___<name>_<ver>_finder.py`` or
      ``__editable__.<name>-<ver>.pth`` as one of its files.
    - Older setuptools: a ``.pth`` file in RECORD whose contents
      include the ``__editable__`` marker.

    Attribution goes through RECORD rather than name-matching against
    arbitrary ``.pth`` files in site-packages so namespace ``.pth``
    contributions like ``*-nspkg.pth`` don't trigger false positives.
    """
    direct_url = _read_text(dist_info, "direct_url.json")
    if direct_url:
        try:
            data = json.loads(direct_url)
            dir_info = data.get("dir_info")
            # A non-dict dir_info or malformed JSON is inconclusive; let
            # the RECORD-based checks below have a go before giving up.
            if isinstance(dir_info, dict) and dir_info.get("editable"):
                return "direct_url.json"
        except json.JSONDecodeError:
            pass

    record_text = _read_text(dist_info, "RECORD")
    if record_text is None:
        return None

    safe_name = re.sub(r"[-.]", "_", dist_name)
    pep660_markers = {
        f"__editable___{safe_name}_{dist_version}_finder.py",
        f"__editable__.{dist_name}-{dist_version}.pth",
    }
    pth_candidates: List[Path] = []
    for row in csv.reader(io.StringIO(record_text)):
        if not row:
            continue
        path_field = row[0]
        if not path_field:
            continue
        leaf = path_field.rsplit("/", 1)[-1]
        if leaf in pep660_markers:
            return "pep660-record"
        if path_field.endswith(".pth"):
            resolved = (dist_info.parent / path_field).resolve()
            pth_candidates.append(resolved)

    for pth in pth_candidates:
        try:
            contents = pth.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "__editable__" in contents:
            return "pth-editable"

    return None


def _classify_install_state(
    dist_info: Path,
) -> Union[InstallState, Refusal]:
    record = dist_info / "RECORD"
    if record.exists():
        return InstallState.PEP376_WHEEL
    if (dist_info / "installed-files.txt").exists():
        return InstallState.LEGACY_INSTALLED_FILES
    if (
        dist_info.name.endswith(".egg-info")
        or (dist_info / "SOURCES.txt").exists()
    ):
        return EggInfoOnly(location=dist_info)
    return IncompleteMetadata(
        dist_info=dist_info,
        detail="no RECORD, no installed-files.txt, no SOURCES.txt",
    )


def locate(name: str) -> Union[LocatedDistribution, Refusal]:
    """Resolve ``name`` to a translocatable handle or a typed refusal.

    Returns a ``Refusal`` (rather than raising) so callers can
    pattern-match on the kind of failure: invalid name, not installed,
    editable, egg-info-only, or metadata that can't be enumerated.
    """
    invalid = validate_name(name)
    if invalid is not None:
        return invalid
    canonical = canonicalize(name)
    try:
        dist = metadata.distribution(canonical)
    except metadata.PackageNotFoundError:
        all_names = _all_installed_names()
        candidates = difflib.get_close_matches(
            canonical, [canonicalize(n) for n in all_names], n=3
        )
        original_candidates = [
            n for n in all_names if canonicalize(n) in candidates
        ]
        return DistributionNotFound(name=name, candidates=original_candidates)

    dist_info = _dist_info_dir_of(dist)
    if dist_info is None or not dist_info.is_dir():
        return IncompleteMetadata(
            dist_info=dist_info,
            detail="distribution does not expose a directory location",
        )

    if dist_info.name.endswith(".egg-info"):
        return EggInfoOnly(location=dist_info)

    if dist.metadata is None or not dist.metadata.get("Name"):
        return IncompleteMetadata(
            dist_info=dist_info, detail="METADATA missing Name field"
        )
    canonical_name = dist.metadata["Name"]
    version = dist.version or ""
    if not version:
        return IncompleteMetadata(
            dist_info=dist_info, detail="METADATA missing Version"
        )

    editable_tag = _detect_editable(dist_info, canonical_name, version)
    if editable_tag is not None:
        return EditableInstall(dist_info=dist_info, detected_via=editable_tag)

    install_state = _classify_install_state(dist_info)
    if isinstance(install_state, Refusal):
        return install_state

    return LocatedDistribution(
        name=name,
        canonical_name=canonical_name,
        version=version,
        dist_info_dir=dist_info,
        install_state=install_state,
    )
