# SPDX-FileCopyrightText: 2026 CoreWeave
#
# SPDX-License-Identifier: MIT

"""Serialize a reconciled inventory back into a wheel archive.

Pins the emitter's load-bearing contract: every emitted member has a
fresh ``sha256`` row in the regenerated RECORD, install-site
provenance is dropped, entry-point script wrappers are left out for
the target installer to regenerate, and verbatim shipped scripts are
carried with their interpreter shebang neutralized back to
``#!python``. Also covers PEP 425 tag compression in the output
filename.
"""

from __future__ import annotations

import csv
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import translocate
from tests.fixtures.build_fixtures import (
    FixtureSpec,
    write_pep376_install,
)
from translocate.emit.wheel import _compress_tags, emit_wheel
from translocate.locate import InstallState, LocatedDistribution
from translocate.reconcile import reconcile
from translocate.record import parse_record
from translocate.scheme import resolve_source_scheme


def _built(tmp: Path, spec: FixtureSpec):
    layout = write_pep376_install(tmp, spec)
    located = LocatedDistribution(
        name=spec.name,
        canonical_name=spec.name,
        version=spec.version,
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
    return layout, inv


class EmitWheelTests(unittest.TestCase):
    def test_emits_wheel_with_consistent_regenerated_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _, inv = _built(
                tmp_path,
                FixtureSpec(
                    name="emit_demo",
                    version="0.1.0",
                    purelib_files={"emit_demo/__init__.py": b"x = 1\n"},
                ),
            )
            out = tmp_path
            result = emit_wheel(inv, out)
            self.assertTrue(result.wheel_path.exists())
            with zipfile.ZipFile(result.wheel_path) as zf:
                names = zf.namelist()
                self.assertIn("emit_demo/__init__.py", names)
                record_member = next(
                    n for n in names if n.endswith(".dist-info/RECORD")
                )
                record_text = zf.read(record_member).decode("utf-8")
            rows = list(csv.reader(io.StringIO(record_text)))
            path_to_row = {r[0]: r for r in rows if r}
            self.assertIn("emit_demo/__init__.py", path_to_row)
            self.assertTrue(
                path_to_row["emit_demo/__init__.py"][1].startswith("sha256=")
            )

    def test_provenance_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _, inv = _built(
                tmp_path,
                FixtureSpec(
                    name="prov",
                    version="0.1.0",
                    purelib_files={"prov/__init__.py": b""},
                    dist_info_extras={
                        "INSTALLER": b"pip\n",
                        "REQUESTED": b"",
                        "direct_url.json": b"{}",
                    },
                ),
            )
            result = emit_wheel(inv, tmp_path)
            with zipfile.ZipFile(result.wheel_path) as zf:
                names = set(zf.namelist())
            self.assertFalse(
                any(n.endswith("/INSTALLER") for n in names),
                f"INSTALLER survived in {names}",
            )
            self.assertFalse(any(n.endswith("/REQUESTED") for n in names))
            self.assertFalse(any(n.endswith("/direct_url.json") for n in names))

    def test_entrypoint_scripts_excluded_from_wheel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _, inv = _built(
                tmp_path,
                FixtureSpec(
                    name="ep",
                    version="0.1.0",
                    purelib_files={"ep/__init__.py": b""},
                    scripts_files={
                        "runme": b"#!/usr/bin/python\nprint('hi')\n",
                    },
                    entry_points={
                        "console_scripts": {"runme": "ep:main"},
                    },
                ),
            )
            result = emit_wheel(inv, tmp_path)
            with zipfile.ZipFile(result.wheel_path) as zf:
                names = set(zf.namelist())
            self.assertFalse(
                any("scripts/runme" in n for n in names),
                f"entry-point script leaked into wheel: {names}",
            )
            self.assertEqual(result.excluded_entrypoint_scripts, ["runme"])

    def test_verbatim_script_carried_with_neutralized_shebang(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _, inv = _built(
                tmp_path,
                FixtureSpec(
                    name="vs",
                    version="0.1.0",
                    purelib_files={"vs/__init__.py": b""},
                    scripts_files={
                        "shipped.sh": b"#!/path/to/python\necho hi\n",
                    },
                ),
            )
            result = emit_wheel(inv, tmp_path)
            with zipfile.ZipFile(result.wheel_path) as zf:
                target = next(
                    n for n in zf.namelist() if n.endswith("/shipped.sh")
                )
                body = zf.read(target)
            self.assertTrue(body.startswith(b"#!python\n"))

    def test_malformed_wheel_tag_warns_and_keeps_valid(self) -> None:
        # The source WHEEL pairs a real tag triple with a junk line.
        # The junk should warn and drop, and the real tag should
        # survive into both the regenerated WHEEL and the output
        # filename.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            layout, inv = _built(
                tmp_path,
                FixtureSpec(
                    name="badtag",
                    version="0.1.0",
                    purelib_files={"badtag/__init__.py": b""},
                ),
            )
            (layout.dist_info / "WHEEL").write_bytes(
                b"Wheel-Version: 1.0\n"
                b"Generator: tests\n"
                b"Root-Is-Purelib: true\n"
                b"Tag: not a valid tag\n"
                b"Tag: cp310-cp310-manylinux_2_17_x86_64\n"
            )
            with self.assertLogs(
                "translocate.emit.wheel", level="WARNING"
            ) as logs:
                result = emit_wheel(inv, tmp_path)
            self.assertTrue(
                any("not a valid tag" in msg for msg in logs.output),
                f"no warning surfaced; logs were {logs.output}",
            )
            self.assertIn(
                "cp310-cp310-manylinux_2_17_x86_64",
                result.wheel_path.name,
            )
            with zipfile.ZipFile(result.wheel_path) as zf:
                wheel_text = next(
                    zf.read(n).decode("utf-8")
                    for n in zf.namelist()
                    if n.endswith("/WHEEL")
                )
            self.assertIn("Tag: cp310-cp310-manylinux_2_17_x86_64", wheel_text)
            self.assertNotIn("not a valid tag", wheel_text)

    def test_generator_reports_package_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _, inv = _built(
                tmp_path,
                FixtureSpec(
                    name="gen",
                    version="0.1.0",
                    purelib_files={"gen/__init__.py": b""},
                ),
            )
            result = emit_wheel(inv, tmp_path)
            with zipfile.ZipFile(result.wheel_path) as zf:
                wheel_text = next(
                    zf.read(n).decode("utf-8")
                    for n in zf.namelist()
                    if n.endswith("/WHEEL")
                )
            self.assertIn(
                f"Generator: translocate {translocate.__version__}",
                wheel_text,
            )

    def test_missing_out_dir_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _, inv = _built(
                tmp_path,
                FixtureSpec(
                    name="x",
                    version="0.1.0",
                    purelib_files={"x/__init__.py": b""},
                ),
            )
            with self.assertRaises(OSError):
                emit_wheel(inv, tmp_path / "does" / "not" / "exist")


class TagCompressionTests(unittest.TestCase):
    def test_single_tag_passes_through(self) -> None:
        self.assertEqual(_compress_tags(["py3-none-any"]), "py3-none-any")

    def test_platform_variants_compress(self) -> None:
        self.assertEqual(
            _compress_tags(
                [
                    "cp310-cp310-manylinux_2_17_x86_64",
                    "cp310-cp310-manylinux2014_x86_64",
                    "cp310-cp310-manylinux_2_28_x86_64",
                ]
            ),
            "cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64."
            "manylinux_2_28_x86_64",
        )

    def test_cartesian_python_and_abi_compress(self) -> None:
        self.assertEqual(
            _compress_tags(
                [
                    "py3-none-any",
                    "py2-none-any",
                ]
            ),
            "py3.py2-none-any",
        )

    def test_non_cartesian_falls_back_to_dot_join(self) -> None:
        # py and plat both vary, but not as a product: 2 tags, not 2*2=4.
        result = _compress_tags(
            ["cp310-cp310-linux_x86_64", "cp311-cp311-manylinux2014_x86_64"]
        )
        self.assertEqual(
            result,
            "cp310-cp310-linux_x86_64.cp311-cp311-manylinux2014_x86_64",
        )


class CompressionTests(unittest.TestCase):
    def _member_compress_types(self, tmp_path: Path, level: int) -> set:
        _, inv = _built(
            tmp_path,
            FixtureSpec(
                name="comp",
                version="0.1.0",
                purelib_files={"comp/__init__.py": b"x" * 4096},
            ),
        )
        result = emit_wheel(inv, tmp_path, compression_level=level)
        with zipfile.ZipFile(result.wheel_path) as zf:
            return {info.compress_type for info in zf.infolist()}

    def test_level_zero_stores_every_member(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            types = self._member_compress_types(Path(tmp), 0)
            self.assertEqual(types, {zipfile.ZIP_STORED})

    def test_default_level_deflates_every_member(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            types = self._member_compress_types(Path(tmp), 6)
            self.assertEqual(types, {zipfile.ZIP_DEFLATED})


if __name__ == "__main__":
    unittest.main()
