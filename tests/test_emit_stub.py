# SPDX-FileCopyrightText: 2026 CoreWeave
#
# SPDX-License-Identifier: MIT

"""Metadata-only stub wheel emission.

Pins the stub contract. The archive holds exactly METADATA, WHEEL,
and RECORD. METADATA is byte-identical to the source, WHEEL is
regenerated with the source's tags, and RECORD rows hash-verify
against the archived members. A distribution with no METADATA is
refused with a typed error.
"""

from __future__ import annotations

import csv
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from tests.fixtures.build_fixtures import (
    FixtureSpec,
    write_pep376_install,
)
from translocate._hashing import hash_bytes_sha256, parse_record_hash
from translocate.emit.wheel import emit_stub_wheel, wheel_filename
from translocate.errors import IncompleteMetadata, TranslocateRefused


class StubWheelTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.layout = write_pep376_install(
            self.tmp,
            FixtureSpec(
                name="stubbable",
                version="1.2.3",
                purelib_files={"stubbable/__init__.py": b"payload = 1\n"},
            ),
        )
        self.out = self.tmp / "index"
        self.out.mkdir()

    def _emit(self):
        return emit_stub_wheel(
            "stubbable",
            "1.2.3",
            self.layout.dist_info,
            self.out,
            root_is_purelib=True,
        )

    def test_archive_holds_exactly_the_dist_info_triple(self) -> None:
        result = self._emit()
        with zipfile.ZipFile(result.wheel_path) as zf:
            names = sorted(zf.namelist())
        self.assertEqual(
            names,
            [
                "stubbable-1.2.3.dist-info/METADATA",
                "stubbable-1.2.3.dist-info/RECORD",
                "stubbable-1.2.3.dist-info/WHEEL",
            ],
        )
        self.assertEqual(result.members_emitted, 3)

    def test_payload_is_not_in_the_archive(self) -> None:
        result = self._emit()
        with zipfile.ZipFile(result.wheel_path) as zf:
            self.assertFalse(
                any("stubbable/__init__.py" in n for n in zf.namelist())
            )

    def test_metadata_carried_verbatim(self) -> None:
        source = (self.layout.dist_info / "METADATA").read_bytes()
        result = self._emit()
        with zipfile.ZipFile(result.wheel_path) as zf:
            archived = zf.read("stubbable-1.2.3.dist-info/METADATA")
        self.assertEqual(archived, source)

    def test_wheel_regenerated_with_inherited_tags(self) -> None:
        (self.layout.dist_info / "WHEEL").write_text(
            "Wheel-Version: 1.0\n"
            "Generator: someone-else\n"
            "Root-Is-Purelib: false\n"
            "Tag: cp310-cp310-linux_x86_64\n",
            encoding="utf-8",
        )
        result = emit_stub_wheel(
            "stubbable",
            "1.2.3",
            self.layout.dist_info,
            self.out,
            root_is_purelib=False,
        )
        self.assertEqual(
            result.wheel_path.name,
            "stubbable-1.2.3-cp310-cp310-linux_x86_64.whl",
        )
        with zipfile.ZipFile(result.wheel_path) as zf:
            body = zf.read("stubbable-1.2.3.dist-info/WHEEL").decode("utf-8")
        self.assertIn("Generator: translocate", body)
        self.assertIn("Tag: cp310-cp310-linux_x86_64", body)
        self.assertIn("Root-Is-Purelib: false", body)

    def test_record_rows_hash_verify(self) -> None:
        result = self._emit()
        with zipfile.ZipFile(result.wheel_path) as zf:
            record_text = zf.read("stubbable-1.2.3.dist-info/RECORD").decode(
                "utf-8"
            )
            for row in csv.reader(io.StringIO(record_text)):
                if not row or not row[1]:
                    continue
                _, expected = parse_record_hash(row[1])
                observed, size = hash_bytes_sha256(zf.read(row[0]))
                self.assertEqual(observed, expected, row[0])
                self.assertEqual(size, int(row[2]), row[0])

    def test_missing_wheel_falls_back_to_py3_none_any(self) -> None:
        (self.layout.dist_info / "WHEEL").unlink()
        result = self._emit()
        self.assertEqual(
            result.wheel_path.name, "stubbable-1.2.3-py3-none-any.whl"
        )

    def test_missing_metadata_refused(self) -> None:
        (self.layout.dist_info / "METADATA").unlink()
        with self.assertRaises(TranslocateRefused) as caught:
            self._emit()
        self.assertIsInstance(caught.exception.refusal, IncompleteMetadata)

    def test_wheel_filename_matches_emitted_stub(self) -> None:
        predicted = wheel_filename("stubbable", "1.2.3", self.layout.dist_info)
        result = self._emit()
        self.assertEqual(result.wheel_path.name, predicted)


if __name__ == "__main__":
    unittest.main()
