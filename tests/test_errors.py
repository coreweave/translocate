# SPDX-FileCopyrightText: 2026 CoreWeave
#
# SPDX-License-Identifier: MIT

"""Refusal message rendering."""

from __future__ import annotations

import unittest
from pathlib import Path

from translocate.errors import IncompleteMetadata


class IncompleteMetadataMessageTests(unittest.TestCase):
    def test_names_the_path_when_present(self) -> None:
        msg = str(
            IncompleteMetadata(
                dist_info=Path("/x/pkg-1.0.dist-info"), detail="boom"
            )
        )
        self.assertIn("/x/pkg-1.0.dist-info", msg)
        self.assertIn("boom", msg)

    def test_omits_location_when_path_absent(self) -> None:
        msg = str(
            IncompleteMetadata(
                dist_info=None, detail="does not expose a directory location"
            )
        )
        self.assertNotIn("None", msg)
        self.assertIn("does not expose a directory location", msg)


if __name__ == "__main__":
    unittest.main()
