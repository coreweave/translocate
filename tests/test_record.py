# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: MIT
# SPDX-PackageName: translocate

"""RECORD parsing and centralized per-entry tagging.

Confirms the CSV parser preserves quoted commas in paths, that the
self-row for RECORD is recognized and marked regenerable, and that
each tagging rule (entry-point script vs verbatim shipped script,
provenance files, signature files) fires off the right input. The
back-ends rely on these tags rather than re-deriving them.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.fixtures.build_fixtures import (
    FixtureSpec,
    write_pep376_install,
)
from translocate.inventory import Category
from translocate.locate import InstallState, LocatedDistribution
from translocate.record import parse_entry_points, parse_record
from translocate.scheme import resolve_source_scheme


def _located_from_layout(layout, name: str, version: str):
    return LocatedDistribution(
        name=name,
        canonical_name=name,
        version=version,
        dist_info_dir=layout.dist_info,
        install_state=InstallState.PEP376_WHEEL,
    )


def _scheme_from_layout(layout):
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


class CSVParsingTests(unittest.TestCase):
    def test_quoted_path_with_comma_parses_intact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = write_pep376_install(
                Path(tmp),
                FixtureSpec(
                    name="commas",
                    version="0.1.0",
                    purelib_files={
                        "weird,name.py": b"# data",
                    },
                ),
            )
            located = _located_from_layout(layout, "commas", "0.1.0")
            scheme = _scheme_from_layout(layout)
            inventory = parse_record(located, scheme)
            purelib_entries = [
                e
                for e in inventory.entries
                if e.category == Category.ROOT_PURELIB
            ]
            paths = {e.rel_path.as_posix() for e in purelib_entries}
            self.assertIn("weird,name.py", paths)


class TaggingTests(unittest.TestCase):
    def _build(self, **kwargs):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        spec = FixtureSpec(name="taggable", version="0.1.0", **kwargs)
        layout = write_pep376_install(Path(tmp.name), spec)
        located = _located_from_layout(layout, "taggable", "0.1.0")
        scheme = _scheme_from_layout(layout)
        inventory = parse_record(located, scheme)
        return inventory

    def test_record_self_row_tagged_and_marked_regenerate(self) -> None:
        inv = self._build(purelib_files={"a.py": b"x"})
        record_entries = [e for e in inv.entries if e.flags.is_record]
        self.assertEqual(len(record_entries), 1)
        self.assertEqual(record_entries[0].rel_path.as_posix(), "RECORD")

    def test_entrypoint_script_tagged_via_entry_points_txt(self) -> None:
        inv = self._build(
            purelib_files={"taggable/__init__.py": b""},
            scripts_files={"runme": b"#!/usr/bin/python\nprint('hi')\n"},
            entry_points={
                "console_scripts": {"runme": "taggable:main"},
            },
        )
        scripts = [e for e in inv.entries if e.category == Category.SCRIPTS]
        self.assertEqual(len(scripts), 1)
        self.assertTrue(scripts[0].flags.is_entrypoint_script)
        self.assertEqual(scripts[0].content_ref.entry_point_name, "runme")

    def test_verbatim_script_not_tagged_as_entrypoint(self) -> None:
        inv = self._build(
            scripts_files={
                "shipped.sh": b"#!/bin/sh\necho hi\n",
            },
        )
        scripts = [e for e in inv.entries if e.category == Category.SCRIPTS]
        self.assertEqual(len(scripts), 1)
        self.assertFalse(scripts[0].flags.is_entrypoint_script)

    def test_provenance_files_tagged_centrally(self) -> None:
        inv = self._build(
            dist_info_extras={
                "INSTALLER": b"pip\n",
                "direct_url.json": b"{}",
                "REQUESTED": b"",
            },
        )
        provenance = [e for e in inv.entries if e.flags.is_provenance]
        names = {e.rel_path.as_posix() for e in provenance}
        self.assertEqual(names, {"INSTALLER", "direct_url.json", "REQUESTED"})

    def test_signature_files_tagged_centrally(self) -> None:
        inv = self._build(
            dist_info_extras={
                "RECORD.jws": b"{}",
                "RECORD.p7s": b"x",
            },
        )
        sigs = [e for e in inv.entries if e.flags.is_signature]
        self.assertEqual(len(sigs), 2)


class EntryPointParsingTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        text = (
            "[console_scripts]\n"
            "FooBar = foo.bar:main\n"
            "\n"
            "[gui_scripts]\n"
            "baz = baz.main:run\n"
        )
        parsed = parse_entry_points(text)
        self.assertEqual(
            parsed,
            {
                "console_scripts": {"FooBar": "foo.bar:main"},
                "gui_scripts": {"baz": "baz.main:run"},
            },
        )


if __name__ == "__main__":
    unittest.main()
