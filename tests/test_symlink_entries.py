# SPDX-FileCopyrightText: 2026 CoreWeave
#
# SPDX-License-Identifier: MIT

"""Intra-distribution symlinks recorded as their own RECORD entries."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from tests.fixtures.build_fixtures import FixtureSpec, write_pep376_install
from translocate.emit.wheel import emit_wheel
from translocate.locate import InstallState, LocatedDistribution
from translocate.reconcile import reconcile
from translocate.record import parse_record
from translocate.scheme import resolve_source_scheme

# A top-level module plus a package that re-exports it through a
# symlink. Both paths are recorded; the symlink points at the
# top-level file.
_SPEC = FixtureSpec(
    name="reexport",
    version="1.0",
    purelib_files={
        "widget.py": b"VALUE = 1\n",
        "pkg/__init__.py": b"",
    },
    purelib_symlinks={"pkg/widget.py": "widget.py"},
)


def _inventory(tmp: Path):
    layout = write_pep376_install(tmp, _SPEC)
    located = LocatedDistribution(
        name=_SPEC.name,
        canonical_name=_SPEC.name,
        version=_SPEC.version,
        dist_info_dir=layout.dist_info,
        install_state=InstallState.PEP376_WHEEL,
    )
    scheme = resolve_source_scheme(
        raw_paths={
            "purelib": str(layout.purelib),
            "platlib": str(layout.platlib),
            "scripts": str(layout.scripts),
            "include": str(layout.headers),
            "data": str(layout.data),
        },
        soabi="x",
        ext_suffix=".so",
        executable=str(layout.prefix / "bin" / "python"),
        prefix=str(layout.prefix),
    )
    inv = parse_record(located, scheme)
    reconcile(inv)
    return inv


class SymlinkEntryTests(unittest.TestCase):
    def test_symlinked_entry_keeps_its_recorded_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inv = _inventory(Path(tmp))
            rels = {e.rel_path.as_posix() for e in inv.entries}
            self.assertIn("widget.py", rels)
            self.assertIn("pkg/widget.py", rels)

    def test_repackage_emits_both_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inv = _inventory(tmp_path)
            result = emit_wheel(inv, tmp_path)
            with zipfile.ZipFile(result.wheel_path) as zf:
                names = set(zf.namelist())
                self.assertIn("widget.py", names)
                self.assertIn("pkg/widget.py", names)
                self.assertEqual(zf.read("pkg/widget.py"), b"VALUE = 1\n")


if __name__ == "__main__":
    unittest.main()
