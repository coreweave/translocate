# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: MIT
# SPDX-PackageName: translocate

"""sha256 helpers, including thread safety of file hashing."""

from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from translocate._hashing import CHUNK_SIZE, hash_file_sha256


class HashFileThreadSafetyTests(unittest.TestCase):
    def test_concurrent_hashing_matches_serial_per_file(self) -> None:
        # Several distinct files, each larger than one chunk, hashed
        # concurrently. A read buffer shared across workers would let
        # one file's bytes bleed into another's digest; each result
        # must match the file's own serially-computed hash.
        with tempfile.TemporaryDirectory() as tmp:
            expected = {}
            for i in range(8):
                path = Path(tmp) / f"blob{i}"
                path.write_bytes(bytes([i]) * (CHUNK_SIZE * 2 + i * 1234))
                expected[path] = hash_file_sha256(path)

            work = list(expected) * 10
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(
                    pool.map(lambda p: (p, hash_file_sha256(p)), work)
                )

        for path, digest in results:
            self.assertEqual(digest, expected[path])


if __name__ == "__main__":
    unittest.main()
