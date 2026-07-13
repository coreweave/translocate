# SPDX-FileCopyrightText: 2026 CoreWeave
#
# SPDX-License-Identifier: MIT

"""Reconcile the inventory against the live filesystem.

Each entry should pick up its on-disk hash and mode; missing files
and content that no longer matches the recorded hash should surface
as flags rather than crashes. The residue walk pairs
``__pycache__/*.pyc`` with recorded ``.py`` siblings (regenerable,
not surfaced) and reports everything else under the install's roots
as unrecorded residue.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.fixtures.build_fixtures import (
    FixtureSpec,
    write_pep376_install,
)
from translocate.locate import InstallState, LocatedDistribution
from translocate.reconcile import reconcile
from translocate.record import parse_record
from translocate.scheme import resolve_source_scheme


def _located(layout, name, version):
    return LocatedDistribution(
        name=name,
        canonical_name=name,
        version=version,
        dist_info_dir=layout.dist_info,
        install_state=InstallState.PEP376_WHEEL,
    )


def _scheme(layout):
    return resolve_source_scheme(
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


class ReconcileTests(unittest.TestCase):
    def test_hash_match_when_file_unmodified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = write_pep376_install(
                Path(tmp),
                FixtureSpec(
                    name="ok",
                    version="0.1.0",
                    purelib_files={"a.py": b"alpha"},
                ),
            )
            inv = parse_record(_located(layout, "ok", "0.1.0"), _scheme(layout))
            reconcile(inv)
            entry = next(
                e for e in inv.entries if e.rel_path.as_posix() == "a.py"
            )
            self.assertFalse(entry.flags.hash_mismatch)
            self.assertFalse(entry.flags.missing_on_disk)
            self.assertIsNotNone(entry.disk_hash)

    def test_missing_on_disk_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = write_pep376_install(
                Path(tmp),
                FixtureSpec(
                    name="gone",
                    version="0.1.0",
                    purelib_files={"a.py": b"alpha"},
                ),
            )
            (layout.purelib / "a.py").unlink()
            inv = parse_record(
                _located(layout, "gone", "0.1.0"), _scheme(layout)
            )
            reconcile(inv)
            entry = next(
                e for e in inv.entries if e.rel_path.as_posix() == "a.py"
            )
            self.assertTrue(entry.flags.missing_on_disk)

    def test_hash_mismatch_flagged_when_file_mutated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = write_pep376_install(
                Path(tmp),
                FixtureSpec(
                    name="mut",
                    version="0.1.0",
                    purelib_files={"a.py": b"alpha"},
                ),
            )
            (layout.purelib / "a.py").write_bytes(b"BETA")
            inv = parse_record(
                _located(layout, "mut", "0.1.0"), _scheme(layout)
            )
            reconcile(inv)
            entry = next(
                e for e in inv.entries if e.rel_path.as_posix() == "a.py"
            )
            self.assertTrue(entry.flags.hash_mismatch)

    def test_bytecode_paired_with_recorded_source_not_residue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = write_pep376_install(
                Path(tmp),
                FixtureSpec(
                    name="b",
                    version="0.1.0",
                    purelib_files={"mod.py": b"x"},
                ),
            )
            cache = layout.purelib / "__pycache__"
            cache.mkdir()
            (cache / "mod.cpython-310.pyc").write_bytes(b"<bytecode>")
            inv = parse_record(_located(layout, "b", "0.1.0"), _scheme(layout))
            reconcile(inv)
            residue = [e for e in inv.entries if e.flags.is_unrecorded_residue]
            self.assertEqual(residue, [])

    def test_unrecorded_residue_surfaced_inside_package_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = write_pep376_install(
                Path(tmp),
                FixtureSpec(
                    name="r",
                    version="0.1.0",
                    purelib_files={"r/__init__.py": b""},
                ),
            )
            (layout.purelib / "r" / "cache_file.dat").write_bytes(b"data")
            inv = parse_record(_located(layout, "r", "0.1.0"), _scheme(layout))
            reconcile(inv)
            residue = [e for e in inv.entries if e.flags.is_unrecorded_residue]
            self.assertEqual(len(residue), 1)
            self.assertEqual(
                residue[0].content_ref.on_disk_path,
                layout.purelib / "r" / "cache_file.dat",
            )

    def test_loose_file_in_shared_scheme_root_not_residue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = write_pep376_install(
                Path(tmp),
                FixtureSpec(
                    name="r",
                    version="0.1.0",
                    purelib_files={"r/__init__.py": b""},
                ),
            )
            # Foreign file from a different dist that shares the purelib
            # root. It must not show up as residue under r's inventory.
            (layout.purelib / "foreign.dat").write_bytes(b"data")
            inv = parse_record(_located(layout, "r", "0.1.0"), _scheme(layout))
            reconcile(inv)
            residue = [e for e in inv.entries if e.flags.is_unrecorded_residue]
            self.assertEqual(residue, [])


if __name__ == "__main__":
    unittest.main()
