# SPDX-FileCopyrightText: 2026 CoreWeave
#
# SPDX-License-Identifier: MIT

"""Synthesize tiny ``.dist-info`` trees on disk for tests.

Each fixture writes a complete install layout under a caller-provided
root, with the standard scheme directories (``bin``, ``include``,
``lib/python3.X/site-packages``) and a dist-info containing METADATA,
WHEEL, RECORD, and any extras the test asks for.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class InstallLayout:
    prefix: Path
    purelib: Path
    platlib: Path
    scripts: Path
    headers: Path
    data: Path
    dist_info: Path


@dataclass
class FixtureSpec:
    name: str
    version: str
    purelib_files: Dict[str, bytes] = field(default_factory=dict)
    # Each entry maps a symlink's purelib-relative path to the
    # purelib-relative path of the ``purelib_files`` entry it points
    # at. The symlink and its referent each get a RECORD row.
    purelib_symlinks: Dict[str, str] = field(default_factory=dict)
    platlib_files: Dict[str, bytes] = field(default_factory=dict)
    scripts_files: Dict[str, bytes] = field(default_factory=dict)
    headers_files: Dict[str, bytes] = field(default_factory=dict)
    data_files: Dict[str, bytes] = field(default_factory=dict)
    dist_info_extras: Dict[str, bytes] = field(default_factory=dict)
    entry_points: Dict[str, Dict[str, str]] = field(default_factory=dict)
    metadata_extras: Dict[str, str] = field(default_factory=dict)
    requires_dist: List[str] = field(default_factory=list)
    root_is_purelib: bool = True
    record_entries_override: Optional[List[Tuple[str, str, str]]] = None


_PY_VERSION = f"python{sys.version_info.major}.{sys.version_info.minor}"


def standard_layout(root: Path, name: str, version: str) -> InstallLayout:
    purelib = root / "lib" / _PY_VERSION / "site-packages"
    platlib = purelib
    scripts = root / "bin"
    headers = root / "include" / _PY_VERSION
    data = root
    dist_info = purelib / f"{name.replace('-', '_')}-{version}.dist-info"
    for d in (purelib, scripts, headers, data, dist_info):
        d.mkdir(parents=True, exist_ok=True)
    return InstallLayout(
        prefix=root,
        purelib=purelib,
        platlib=platlib,
        scripts=scripts,
        headers=headers,
        data=data,
        dist_info=dist_info,
    )


def _b64(data: bytes) -> str:
    return (
        base64.urlsafe_b64encode(hashlib.sha256(data).digest())
        .rstrip(b"=")
        .decode("ascii")
    )


def _format_record_row(
    archive_path: str, data: bytes, with_hash: bool
) -> Tuple[str, str, str]:
    if not with_hash:
        return (archive_path, "", "")
    return (archive_path, f"sha256={_b64(data)}", str(len(data)))


def write_pep376_install(root: Path, spec: FixtureSpec) -> InstallLayout:
    """Materialize ``spec`` under ``root`` as a PEP 376 install."""
    layout = standard_layout(root, spec.name, spec.version)

    record_rows: List[Tuple[str, str, str]] = []

    def _write_payload(
        target_dir: Path, files: Dict[str, bytes], category_prefix: str
    ) -> None:
        for rel, content in files.items():
            dest = target_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
            archive_rel = _record_path_from(layout.dist_info, dest)
            record_rows.append(_format_record_row(archive_rel, content, True))

    _write_payload(layout.purelib, spec.purelib_files, "purelib")
    for link_rel, target_rel in spec.purelib_symlinks.items():
        link = layout.purelib / link_rel
        target = layout.purelib / target_rel
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(target, link.parent))
        record_rows.append(
            _format_record_row(
                _record_path_from(layout.dist_info, link),
                target.read_bytes(),
                True,
            )
        )
    if not spec.root_is_purelib or layout.purelib != layout.platlib:
        _write_payload(layout.platlib, spec.platlib_files, "platlib")
    _write_payload(layout.scripts, spec.scripts_files, "scripts")
    _write_payload(layout.headers, spec.headers_files, "headers")
    _write_payload(layout.data, spec.data_files, "data")

    metadata_lines = [
        "Metadata-Version: 2.1",
        f"Name: {spec.name}",
        f"Version: {spec.version}",
    ]
    for k, v in spec.metadata_extras.items():
        metadata_lines.append(f"{k}: {v}")
    for r in spec.requires_dist:
        metadata_lines.append(f"Requires-Dist: {r}")
    metadata_bytes = ("\n".join(metadata_lines) + "\n").encode("utf-8")
    (layout.dist_info / "METADATA").write_bytes(metadata_bytes)
    record_rows.append(
        _format_record_row(
            _record_path_from(layout.dist_info, layout.dist_info / "METADATA"),
            metadata_bytes,
            True,
        )
    )

    wheel_text = (
        "Wheel-Version: 1.0\n"
        "Generator: tests/fixtures\n"
        f"Root-Is-Purelib: {'true' if spec.root_is_purelib else 'false'}\n"
        "Tag: py3-none-any\n"
    ).encode("utf-8")
    (layout.dist_info / "WHEEL").write_bytes(wheel_text)
    record_rows.append(
        _format_record_row(
            _record_path_from(layout.dist_info, layout.dist_info / "WHEEL"),
            wheel_text,
            True,
        )
    )

    if spec.entry_points:
        ep_lines: List[str] = []
        for section, mapping in spec.entry_points.items():
            ep_lines.append(f"[{section}]")
            for name, target in mapping.items():
                ep_lines.append(f"{name} = {target}")
            ep_lines.append("")
        ep_bytes = "\n".join(ep_lines).encode("utf-8")
        (layout.dist_info / "entry_points.txt").write_bytes(ep_bytes)
        record_rows.append(
            _format_record_row(
                _record_path_from(
                    layout.dist_info, layout.dist_info / "entry_points.txt"
                ),
                ep_bytes,
                True,
            )
        )

    for name, content in spec.dist_info_extras.items():
        target = layout.dist_info / name
        target.write_bytes(content)
        record_rows.append(
            _format_record_row(
                _record_path_from(layout.dist_info, target),
                content,
                True,
            )
        )

    if spec.record_entries_override is not None:
        record_rows = spec.record_entries_override

    # RECORD self-row has empty hash and size.
    record_rows.append(
        (
            _record_path_from(layout.dist_info, layout.dist_info / "RECORD"),
            "",
            "",
        )
    )

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for row in record_rows:
        writer.writerow(row)
    (layout.dist_info / "RECORD").write_text(buf.getvalue(), encoding="utf-8")

    return layout


def _record_path_from(dist_info: Path, target: Path) -> str:
    rel = os.path.relpath(target, dist_info.parent)
    return rel.replace(os.sep, "/")
