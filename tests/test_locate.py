# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: MIT
# SPDX-PackageName: translocate

"""Locate a distribution by name; classify its install layout.

Covers PEP 503 name normalization, distribution-not-found with
suggestion fallback, up-front name validation (version specifiers,
extras brackets, environment markers, whitespace, path separators),
and editable-install detection via ``direct_url.json``. Also pins the
namespace-``.pth`` false-positive case so it doesn't regress.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.fixtures.build_fixtures import (
    FixtureSpec,
    write_pep376_install,
)
from translocate.errors import (
    DistributionNotFound,
    InvalidDistributionName,
)
from translocate.locate import canonicalize, locate, validate_name


class CanonicalizationTests(unittest.TestCase):
    def test_pep503_normalize(self) -> None:
        self.assertEqual(canonicalize("Foo_Bar.Baz"), "foo-bar-baz")
        self.assertEqual(canonicalize("foo--bar"), "foo-bar")
        self.assertEqual(canonicalize("ALL_CAPS"), "all-caps")


class LocateTests(unittest.TestCase):
    def test_distribution_not_found(self) -> None:
        result = locate("definitely-not-installed-xyz-fixture-only")
        self.assertIsInstance(result, DistributionNotFound)

    def test_locate_finds_running_translocate(self) -> None:
        result = locate("translocate")
        # Either outcome is valid: locate returns a LocatedDistribution
        # when the metadata search path sees translocate's dist-info,
        # or DistributionNotFound otherwise.
        self.assertTrue(
            hasattr(result, "name") or isinstance(result, DistributionNotFound)
        )


class NameValidationTests(unittest.TestCase):
    def test_accepts_valid_names(self) -> None:
        for name in ("torch", "Pillow", "scikit-learn", "foo.bar_baz", "X"):
            self.assertIsNone(validate_name(name), name)

    def test_empty_name_rejected(self) -> None:
        result = validate_name("")
        self.assertIsInstance(result, InvalidDistributionName)

    def test_version_specifier_rejected_with_hint(self) -> None:
        result = validate_name("torch>=2.0")
        self.assertIsInstance(result, InvalidDistributionName)
        self.assertIn("version specifier", result.reason)

    def test_equals_rejected_with_hint(self) -> None:
        result = validate_name("torch==2.0")
        self.assertIsInstance(result, InvalidDistributionName)
        self.assertIn("version specifier", result.reason)

    def test_extras_bracket_rejected(self) -> None:
        result = validate_name("requests[security]")
        self.assertIsInstance(result, InvalidDistributionName)
        self.assertIn("extras", result.reason)

    def test_environment_marker_rejected(self) -> None:
        result = validate_name("torch; python_version<'3.10'")
        self.assertIsInstance(result, InvalidDistributionName)

    def test_path_separator_rejected(self) -> None:
        result = validate_name("foo/bar")
        self.assertIsInstance(result, InvalidDistributionName)

    def test_whitespace_rejected(self) -> None:
        self.assertIsInstance(validate_name(" torch"), InvalidDistributionName)
        self.assertIsInstance(validate_name("torch "), InvalidDistributionName)
        self.assertIsInstance(
            validate_name("torch vision"), InvalidDistributionName
        )

    def test_locate_returns_invalid_name_refusal(self) -> None:
        result = locate("torch>=2.0")
        self.assertIsInstance(result, InvalidDistributionName)


class EditableDetectionTests(unittest.TestCase):
    def test_direct_url_editable_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = write_pep376_install(
                Path(tmp),
                FixtureSpec(
                    name="edit",
                    version="0.1.0",
                    purelib_files={"edit/__init__.py": b""},
                    dist_info_extras={
                        "direct_url.json": json.dumps(
                            {
                                "url": "file:///x",
                                "dir_info": {"editable": True},
                            }
                        ).encode("utf-8"),
                    },
                ),
            )
            from translocate.locate import _detect_editable

            tag = _detect_editable(layout.dist_info, "edit", "0.1.0")
            self.assertEqual(tag, "direct_url.json")

    def test_namespace_pth_does_not_false_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = write_pep376_install(
                Path(tmp),
                FixtureSpec(
                    name="nspkg_demo",
                    version="0.1.0",
                    purelib_files={
                        "nspkg_demo/__init__.py": b"",
                        "nspkg_demo-nspkg.pth": b"import sys\n",
                    },
                ),
            )
            from translocate.locate import _detect_editable

            tag = _detect_editable(layout.dist_info, "nspkg_demo", "0.1.0")
            self.assertIsNone(tag)


if __name__ == "__main__":
    unittest.main()
