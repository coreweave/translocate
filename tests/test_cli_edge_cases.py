# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: MIT
# SPDX-PackageName: translocate

"""CLI-layer up-front guards and exit-code contract.

Refuses an invocation early when the source and target resolve to the
same prefix, when they share a category root (purelib/platlib/scripts)
without being the same interpreter, or when their SOABI / EXT_SUFFIX
disagree so compiled extensions wouldn't load. Also covers the
end-to-end CLI exit-code mapping for invalid distribution names,
missing distributions, and ``--out`` pointing at a non-directory.
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
import venv
import zipfile
from pathlib import Path, PurePosixPath

from tests.fixtures.build_fixtures import (
    FixtureSpec,
    write_pep376_install,
)
from translocate.cli import (
    EXIT_DIST_NOT_FOUND,
    EXIT_INVALID_INVOCATION,
    _check_abi,
    _check_target_distinct,
    _hash_mismatch_refusal,
    _is_platform_specific,
    main,
)
from translocate.errors import (
    ABIMismatch,
    CloneToSelf,
    HashMismatch,
    SourceTargetSharedRoot,
)
from translocate.inventory import (
    Category,
    ContentRef,
    Entry,
    EntryFlags,
    InstallScheme,
    Inventory,
)
from translocate.locate import InstallState, LocatedDistribution
from translocate.record import parse_record
from translocate.scheme import resolve_source_scheme


def _scheme(
    prefix: Path,
    purelib_suffix: str = "lib/python/site-packages",
    soabi: str = "cpython-310-x86_64-linux-gnu",
    ext_suffix: str = ".cpython-310-x86_64-linux-gnu.so",
) -> InstallScheme:
    return InstallScheme(
        purelib=prefix / purelib_suffix,
        platlib=prefix / purelib_suffix,
        scripts=prefix / "bin",
        headers=prefix / "include" / "python",
        data=prefix,
        prefix=prefix,
        executable=prefix / "bin" / "python",
        soabi=soabi,
        ext_suffix=ext_suffix,
    )


class CloneToSelfTests(unittest.TestCase):
    def test_same_prefix_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp).resolve()
            (p / "bin").mkdir(parents=True, exist_ok=True)
            (p / "lib" / "python" / "site-packages").mkdir(
                parents=True, exist_ok=True
            )
            (p / "include" / "python").mkdir(parents=True, exist_ok=True)
            scheme = _scheme(p)
            self.assertIsInstance(
                _check_target_distinct(scheme, scheme), CloneToSelf
            )

    def test_shared_purelib_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            src = tmp_path / "src"
            tgt = tmp_path / "tgt"
            for p in (src, tgt):
                (p / "bin").mkdir(parents=True, exist_ok=True)
                (p / "include" / "python").mkdir(parents=True, exist_ok=True)
            shared = tmp_path / "shared-site"
            shared.mkdir(parents=True, exist_ok=True)
            src_scheme = InstallScheme(
                purelib=shared,
                platlib=shared,
                scripts=src / "bin",
                headers=src / "include" / "python",
                data=src,
                prefix=src,
                executable=src / "bin" / "python",
                soabi="x",
                ext_suffix=".so",
            )
            tgt_scheme = InstallScheme(
                purelib=shared,
                platlib=shared,
                scripts=tgt / "bin",
                headers=tgt / "include" / "python",
                data=tgt,
                prefix=tgt,
                executable=tgt / "bin" / "python",
                soabi="x",
                ext_suffix=".so",
            )
            result = _check_target_distinct(src_scheme, tgt_scheme)
            self.assertIsInstance(result, SourceTargetSharedRoot)

    def test_distinct_prefixes_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            src = tmp_path / "src"
            tgt = tmp_path / "tgt"
            for p in (src, tgt):
                (p / "lib" / "python" / "site-packages").mkdir(
                    parents=True, exist_ok=True
                )
                (p / "bin").mkdir(parents=True, exist_ok=True)
                (p / "include" / "python").mkdir(parents=True, exist_ok=True)
            self.assertIsNone(
                _check_target_distinct(_scheme(src), _scheme(tgt))
            )


class CLIMultiNameTests(unittest.TestCase):
    def test_multiple_names_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "wheels"
            code = main(
                [
                    "repackage",
                    "torch[gpu]",
                    "numpy",
                    "--out",
                    str(out),
                ]
            )
            self.assertEqual(code, EXIT_INVALID_INVOCATION)

    def test_repackage_out_pointing_at_file_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "single.whl"
            out.write_text("not a directory")
            code = main(
                [
                    "repackage",
                    "definitely-not-real-zzz",
                    "--out",
                    str(out),
                ]
            )
            self.assertEqual(code, EXIT_INVALID_INVOCATION)


class CLINameErrorTests(unittest.TestCase):
    """End-to-end CLI behavior for invalid names and missing distributions."""

    def test_invalid_name_returns_exit_invalid_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp).resolve()
            (target / "bin").mkdir()
            code = main(
                [
                    "clone",
                    "torch>=2.0",
                    "--target",
                    str(target),
                ]
            )
            self.assertEqual(code, EXIT_INVALID_INVOCATION)

    def test_empty_name_returns_exit_invalid_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp).resolve()
            (target / "bin").mkdir()
            code = main(["clone", "", "--target", str(target)])
            self.assertEqual(code, EXIT_INVALID_INVOCATION)

    def test_missing_distribution_returns_exit_dist_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp).resolve()
            (target / "bin").mkdir()
            code = main(
                [
                    "clone",
                    "definitely-not-installed-fixture-name-zzz",
                    "--target",
                    str(target),
                ]
            )
            self.assertEqual(code, EXIT_DIST_NOT_FOUND)

    def test_repackage_invalid_name_returns_exit_invalid_invocation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "wheels"
            code = main(
                [
                    "repackage",
                    "torch[gpu]",
                    "--out",
                    str(out),
                ]
            )
            self.assertEqual(code, EXIT_INVALID_INVOCATION)


class ABICheckTests(unittest.TestCase):
    def test_matching_soabi_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            src = tmp_path / "src"
            tgt = tmp_path / "tgt"
            for p in (src, tgt):
                p.mkdir()
            self.assertIsNone(_check_abi(_scheme(src), _scheme(tgt)))

    def test_mismatched_soabi_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            src = tmp_path / "src"
            tgt = tmp_path / "tgt"
            for p in (src, tgt):
                p.mkdir()
            src_scheme = _scheme(src, soabi="cpython-310-x86_64-linux-gnu")
            tgt_scheme = _scheme(tgt, soabi="cpython-312-aarch64-linux-gnu")
            result = _check_abi(src_scheme, tgt_scheme)
            self.assertIsInstance(result, ABIMismatch)


class PlatformSpecificityTests(unittest.TestCase):
    """``_is_platform_specific`` decides whether the ABI gate applies."""

    def _inventory(self, tmp: Path, wheel_text: str | None):
        layout = write_pep376_install(
            tmp,
            FixtureSpec(name="probe", version="0.1.0"),
        )
        wheel_path = layout.dist_info / "WHEEL"
        if wheel_text is None:
            wheel_path.unlink()
        else:
            wheel_path.write_text(wheel_text, encoding="utf-8")
        located = LocatedDistribution(
            name="probe",
            canonical_name="probe",
            version="0.1.0",
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
        return parse_record(located, scheme)

    def test_pure_python_wheel_is_not_platform_specific(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inv = self._inventory(
                Path(tmp),
                "Wheel-Version: 1.0\nTag: py3-none-any\n",
            )
            self.assertFalse(_is_platform_specific(inv.dist_info_dir))

    def test_compiled_wheel_is_platform_specific(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inv = self._inventory(
                Path(tmp),
                "Wheel-Version: 1.0\nTag: cp310-cp310-linux_x86_64\n",
            )
            self.assertTrue(_is_platform_specific(inv.dist_info_dir))

    def test_universal_tag_set_is_pure_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inv = self._inventory(
                Path(tmp),
                "Wheel-Version: 1.0\nTag: py2-none-any\nTag: py3-none-any\n",
            )
            self.assertFalse(_is_platform_specific(inv.dist_info_dir))

    def test_missing_wheel_is_treated_as_platform_specific(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inv = self._inventory(Path(tmp), None)
            self.assertTrue(_is_platform_specific(inv.dist_info_dir))


class HashMismatchPolicyTests(unittest.TestCase):
    def _inventory(self, *, mismatched: bool) -> Inventory:
        entry = Entry(
            category=Category.ROOT_PURELIB,
            rel_path=PurePosixPath("pkg/mod.py"),
            content_ref=ContentRef.on_disk(Path("/src/pkg/mod.py")),
            flags=EntryFlags(hash_mismatch=mismatched),
        )
        return Inventory(
            dist_name="pkg",
            dist_version="1.0",
            dist_info_dir=Path("/src/pkg-1.0.dist-info"),
            source_scheme=_scheme(Path("/src")),
            root_is_purelib=True,
            entries=[entry],
        )

    def test_fail_mode_refuses_on_mismatch(self) -> None:
        refusal = _hash_mismatch_refusal(
            self._inventory(mismatched=True), "fail"
        )
        self.assertIsInstance(refusal, HashMismatch)

    def test_warn_mode_carries_mismatch(self) -> None:
        self.assertIsNone(
            _hash_mismatch_refusal(self._inventory(mismatched=True), "warn")
        )

    def test_fail_mode_passes_clean_inventory(self) -> None:
        self.assertIsNone(
            _hash_mismatch_refusal(self._inventory(mismatched=False), "fail")
        )


class _SyntheticDistTestCase(unittest.TestCase):
    """Base for CLI runs against a locatable synthetic install.

    Puts the fixture's purelib on ``sys.path`` so
    ``importlib.metadata`` resolves the distribution by name.
    """

    dist_name = "translocate-cli-fixture"

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.layout = write_pep376_install(
            self.tmp,
            FixtureSpec(
                name=self.dist_name,
                version="1.2.3",
                purelib_files={
                    "translocate_cli_fixture/__init__.py": b"x = 1\n"
                },
            ),
        )
        sys.path.insert(0, str(self.layout.purelib))
        self.addCleanup(sys.path.remove, str(self.layout.purelib))


class RepackageMetadataOnlyTests(_SyntheticDistTestCase):
    def test_stub_wheel_written(self) -> None:
        out = self.tmp / "index"
        code = main(
            [
                "repackage",
                self.dist_name,
                "--out",
                str(out),
                "--mkdir",
                "--metadata-only",
            ]
        )
        self.assertEqual(code, 0)
        wheels = list(out.glob("*.whl"))
        self.assertEqual(len(wheels), 1)
        self.assertLess(wheels[0].stat().st_size, 10_000)
        with zipfile.ZipFile(wheels[0]) as zf:
            self.assertEqual(len(zf.namelist()), 3)

    def test_uv_config_stdout(self) -> None:
        out = self.tmp / "index"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(
                [
                    "repackage",
                    self.dist_name,
                    "--out",
                    str(out),
                    "--mkdir",
                    "--metadata-only",
                    "--uv-config",
                    "-",
                ]
            )
        self.assertEqual(code, 0)
        text = stdout.getvalue()
        self.assertIn("[[tool.uv.index]]", text)
        self.assertIn(f'url = "file://{out.resolve()}"', text)
        self.assertIn(
            'translocate-cli-fixture = { index = "translocated" }', text
        )


class RepackageDryRunTests(_SyntheticDistTestCase):
    def test_writes_no_wheels_and_skips_mkdir(self) -> None:
        out = self.tmp / "index"
        code = main(
            [
                "repackage",
                self.dist_name,
                "--out",
                str(out),
                "--mkdir",
                "--dry-run",
            ]
        )
        self.assertEqual(code, 0)
        self.assertFalse(out.exists())

    def test_uv_config_file_written_under_dry_run(self) -> None:
        out = self.tmp / "index"
        config = self.tmp / "uv-snippet.toml"
        code = main(
            [
                "repackage",
                self.dist_name,
                "--out",
                str(out),
                "--dry-run",
                "--uv-config",
                str(config),
            ]
        )
        self.assertEqual(code, 0)
        self.assertFalse(out.exists())
        self.assertIn(
            'translocate-cli-fixture = { index = "translocated" }',
            config.read_text(encoding="utf-8"),
        )


class CloneDryRunTests(_SyntheticDistTestCase):
    def test_materializes_nothing(self) -> None:
        target = self.tmp / "venv"
        venv.create(target, with_pip=False)
        code = main(
            [
                "clone",
                self.dist_name,
                "--target",
                str(target),
                "--dry-run",
            ]
        )
        self.assertEqual(code, 0)
        site_dirs = list(target.glob("lib/python*/site-packages"))
        self.assertTrue(site_dirs)
        for site in site_dirs:
            self.assertFalse(
                any("fixture" in p.name for p in site.iterdir()),
                f"dry run materialized files under {site}",
            )


if __name__ == "__main__":
    unittest.main()
