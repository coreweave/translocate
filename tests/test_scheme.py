# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: MIT
# SPDX-PackageName: translocate

"""Scheme inversion, path classification, and root_is_purelib."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from translocate.inventory import Category, InstallScheme
from translocate.scheme import (
    classify,
    distribution_root_is_purelib,
    scheme_from_target_query,
)


def _scheme_for(prefix: Path) -> InstallScheme:
    site = prefix / "lib" / "python" / "site-packages"
    return InstallScheme(
        purelib=site,
        platlib=site,
        scripts=prefix / "bin",
        headers=prefix / "include" / "python",
        data=prefix,
        prefix=prefix,
        executable=prefix / "bin" / "python",
        soabi="cpython-310-x86_64-linux-gnu",
        ext_suffix=".cpython-310-x86_64-linux-gnu.so",
    )


class ClassifyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.prefix = Path(self.tmp.name).resolve()
        for d in (
            self.prefix / "lib" / "python" / "site-packages",
            self.prefix / "bin",
            self.prefix / "include" / "python",
        ):
            d.mkdir(parents=True, exist_ok=True)
        self.scheme = _scheme_for(self.prefix)

    def test_script_under_prefix_classifies_as_scripts_not_data(self) -> None:
        path = self.prefix / "bin" / "foo"
        path.write_text("x")
        category, rel = classify(path, self.scheme, True)
        self.assertEqual(category, Category.SCRIPTS)
        self.assertEqual(rel.as_posix(), "foo")

    def test_header_under_prefix_classifies_as_headers(self) -> None:
        path = self.prefix / "include" / "python" / "h.h"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
        category, rel = classify(path, self.scheme, True)
        self.assertEqual(category, Category.HEADERS)
        self.assertEqual(rel.as_posix(), "h.h")

    def test_purelib_file_classifies_as_root_purelib(self) -> None:
        path = self.prefix / "lib" / "python" / "site-packages" / "foo.py"
        path.write_text("x")
        category, rel = classify(path, self.scheme, True)
        self.assertEqual(category, Category.ROOT_PURELIB)
        self.assertEqual(rel.as_posix(), "foo.py")

    def test_arbitrary_data_file_under_prefix_classifies_as_data(self) -> None:
        path = self.prefix / "share" / "thing.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
        category, rel = classify(path, self.scheme, True)
        self.assertEqual(category, Category.DATA)
        self.assertEqual(rel.as_posix(), "share/thing.txt")

    def test_outside_prefix_classifies_as_outside_known_scheme(self) -> None:
        with tempfile.TemporaryDirectory() as foreign:
            foreign_path = Path(foreign) / "stray"
            foreign_path.write_text("x")
            category, _ = classify(foreign_path, self.scheme, True)
            self.assertEqual(category, Category.OUTSIDE_KNOWN_SCHEME)


def _split_scheme(root: Path) -> InstallScheme:
    return InstallScheme(
        purelib=root / "purelib",
        platlib=root / "platlib",
        scripts=root / "bin",
        headers=root / "include",
        data=root,
        prefix=root,
        executable=root / "bin" / "python",
        soabi="x",
        ext_suffix=".so",
    )


class DistributionRootTests(unittest.TestCase):
    def _dist_info(self, parent: Path, wheel: str = None) -> Path:
        dist_info = parent / "x-1.0.dist-info"
        dist_info.mkdir(parents=True, exist_ok=True)
        if wheel is not None:
            (dist_info / "WHEEL").write_text(wheel, encoding="utf-8")
        return dist_info

    def test_wheel_root_is_purelib_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scheme = _split_scheme(Path(tmp).resolve())
            dist_info = self._dist_info(
                scheme.platlib, "Root-Is-Purelib: true\n"
            )
            self.assertTrue(distribution_root_is_purelib(dist_info, scheme))

    def test_wheel_root_is_purelib_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scheme = _split_scheme(Path(tmp).resolve())
            dist_info = self._dist_info(
                scheme.purelib, "Root-Is-Purelib: false\n"
            )
            self.assertFalse(distribution_root_is_purelib(dist_info, scheme))

    def test_no_wheel_split_dirs_under_purelib(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scheme = _split_scheme(Path(tmp).resolve())
            dist_info = self._dist_info(scheme.purelib)
            self.assertTrue(distribution_root_is_purelib(dist_info, scheme))

    def test_no_wheel_split_dirs_under_platlib(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scheme = _split_scheme(Path(tmp).resolve())
            dist_info = self._dist_info(scheme.platlib)
            self.assertFalse(distribution_root_is_purelib(dist_info, scheme))

    def test_no_wheel_coincident_dirs_defaults_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scheme = _scheme_for(Path(tmp).resolve())
            dist_info = self._dist_info(scheme.purelib)
            self.assertFalse(distribution_root_is_purelib(dist_info, scheme))


class TargetQueryTests(unittest.TestCase):
    def _query(self, root: Path) -> dict:
        return {
            "purelib": str(root / "site-packages"),
            "platlib": str(root / "site-packages"),
            "scripts": str(root / "bin"),
            "headers": str(root / "include"),
            "data": str(root),
            "soabi": "x",
            "ext_suffix": ".so",
            "executable": str(root / "bin" / "python"),
            "prefix": str(root),
        }

    def test_null_required_field_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            query = self._query(Path(tmp).resolve())
            query["headers"] = None
            with self.assertRaises(TypeError):
                scheme_from_target_query(query)

    def test_non_string_field_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            query = self._query(Path(tmp).resolve())
            query["purelib"] = 123
            with self.assertRaises(TypeError):
                scheme_from_target_query(query)

    def test_well_formed_query_builds_scheme(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scheme = scheme_from_target_query(self._query(Path(tmp).resolve()))
            self.assertEqual(scheme.soabi, "x")
            self.assertEqual(scheme.ext_suffix, ".so")


if __name__ == "__main__":
    unittest.main()
