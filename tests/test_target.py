# SPDX-FileCopyrightText: 2026 CoreWeave
#
# SPDX-License-Identifier: MIT

"""Target-interpreter subprocess probes."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.fixtures.build_fixtures import FixtureSpec, write_pep376_install
from translocate._target import (
    _INSTALLED_NAMES_CODE,
    _METADATA_VERSION_CODE,
    query_target_installed_names,
    query_target_version,
)
from translocate.errors import TargetNotAPython, TranslocateRefused


def _fake_subprocess(returncode: int, stdout: str = "", stderr: str = ""):
    """Build a stand-in subprocess.run that yields a chosen result."""

    def runner(*args, **kwargs):
        return mock.MagicMock(
            returncode=returncode, stdout=stdout, stderr=stderr
        )

    return runner


def _run_probe_in_process(
    code: str, argv, purelib: Path, platlib: Path
) -> dict:
    """Run an embedded probe string in-process and decode its JSON output.

    The probe expects ``sys.argv`` and ``sysconfig.get_paths()`` from
    the target interpreter; we substitute synthetic values for both.
    """
    fake_paths = {
        "purelib": str(purelib),
        "platlib": str(platlib),
        "scripts": str(purelib),
        "include": str(purelib),
        "data": str(purelib),
    }
    buf = io.StringIO()
    with mock.patch("sysconfig.get_paths", return_value=fake_paths):
        with mock.patch.object(sys, "argv", list(argv)):
            with contextlib.redirect_stdout(buf):
                exec(code, {"__name__": "__main__"})
    return json.loads(buf.getvalue().strip())


class TargetProbeFailureTests(unittest.TestCase):
    def test_query_target_version_raises_on_nonzero_rc(self) -> None:
        with mock.patch(
            "translocate._target.subprocess.run",
            new=_fake_subprocess(1, stderr="boom"),
        ):
            with self.assertRaises(TranslocateRefused) as cm:
                query_target_version(Path("/fake/python"), "somepkg")
        self.assertIsInstance(cm.exception.refusal, TargetNotAPython)

    def test_query_target_installed_names_raises_on_nonzero_rc(self) -> None:
        with mock.patch(
            "translocate._target.subprocess.run",
            new=_fake_subprocess(1, stderr="boom"),
        ):
            with self.assertRaises(TranslocateRefused) as cm:
                query_target_installed_names(Path("/fake/python"))
        self.assertIsInstance(cm.exception.refusal, TargetNotAPython)


class TargetProbePathRestrictionTests(unittest.TestCase):
    # A venv created with ``--system-site-packages`` reaches inherited
    # packages through ``sys.path`` but does not own them. uv treats
    # such inherited packages as not-present; translocate matches that
    # policy by passing the venv's own purelib/platlib to
    # ``metadata.distributions(path=...)`` rather than walking the
    # full ``sys.path``.

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.layout = write_pep376_install(
            self.tmp / "owned",
            FixtureSpec(
                name="ownedpkg",
                version="9.9.9",
                purelib_files={"ownedpkg/__init__.py": b""},
            ),
        )
        self.elsewhere = self.tmp / "elsewhere"
        self.elsewhere.mkdir()
        sys.path.insert(0, str(self.layout.purelib))
        self.addCleanup(sys.path.remove, str(self.layout.purelib))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_version_probe_ignores_sys_path_only_dists(self) -> None:
        result = _run_probe_in_process(
            _METADATA_VERSION_CODE,
            ["_", "ownedpkg"],
            purelib=self.elsewhere,
            platlib=self.elsewhere,
        )
        self.assertEqual(result, {"ok": False})

    def test_version_probe_finds_dist_under_supplied_paths(self) -> None:
        result = _run_probe_in_process(
            _METADATA_VERSION_CODE,
            ["_", "ownedpkg"],
            purelib=self.layout.purelib,
            platlib=self.layout.purelib,
        )
        self.assertEqual(result, {"ok": True, "version": "9.9.9"})

    def test_installed_names_probe_ignores_sys_path_only_dists(self) -> None:
        names = _run_probe_in_process(
            _INSTALLED_NAMES_CODE,
            ["_"],
            purelib=self.elsewhere,
            platlib=self.elsewhere,
        )
        self.assertNotIn("ownedpkg", names)

    def test_installed_names_probe_finds_dist_under_supplied_paths(
        self,
    ) -> None:
        names = _run_probe_in_process(
            _INSTALLED_NAMES_CODE,
            ["_"],
            purelib=self.layout.purelib,
            platlib=self.layout.purelib,
        )
        self.assertIn("ownedpkg", names)


if __name__ == "__main__":
    unittest.main()
