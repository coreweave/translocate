# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: MIT
# SPDX-PackageName: translocate

"""RECORD consistency checks for a cloned prefix."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from translocate._hashing import format_record_hash, hash_bytes_sha256
from translocate.materialize.verify import _verify_record_consistency


def _target_with_record(tmp: Path, on_disk: bytes, recorded: bytes) -> Path:
    """Lay out a target dist-info whose RECORD pins ``recorded`` for a
    file written with ``on_disk`` bytes. Returns the RECORD path.
    """
    dist_info = tmp / "pkg-1.0.dist-info"
    dist_info.mkdir(parents=True)
    (tmp / "pkg").mkdir()
    (tmp / "pkg" / "mod.py").write_bytes(on_disk)
    digest, size = hash_bytes_sha256(recorded)
    record = dist_info / "RECORD"
    with record.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            ("pkg/mod.py", format_record_hash("sha256", digest), str(size))
        )
        writer.writerow(("pkg-1.0.dist-info/RECORD", "", ""))
    return record


class RecordConsistencyTests(unittest.TestCase):
    def test_matching_file_passes_either_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            record = _target_with_record(Path(tmp), b"hello\n", b"hello\n")
            self.assertEqual(
                _verify_record_consistency(record, False, None), []
            )
            self.assertEqual(_verify_record_consistency(record, True, None), [])

    def test_size_mismatch_caught_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            record = _target_with_record(Path(tmp), b"hi\n", b"hello\n")
            failures = _verify_record_consistency(record, False, None)
            self.assertTrue(any("size mismatch" in f for f in failures))

    def test_same_size_corruption_needs_verify_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # "world\n" and "hello\n" are both six bytes, so only the
            # hash pass distinguishes the corrupted file.
            record = _target_with_record(Path(tmp), b"world\n", b"hello\n")
            self.assertEqual(
                _verify_record_consistency(record, False, None), []
            )
            failures = _verify_record_consistency(record, True, None)
            self.assertTrue(any("hash mismatch" in f for f in failures))


if __name__ == "__main__":
    unittest.main()
