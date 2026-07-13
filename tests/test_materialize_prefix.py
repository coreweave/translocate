# SPDX-FileCopyrightText: 2026 CoreWeave
#
# SPDX-License-Identifier: MIT

"""Project an inventory into a target interpreter's install scheme.

Covers the link-mode primitives (copy as the universal fallback,
symlink resolving through to the source) and the end-to-end flow
into a freshly-created venv: the target's ``importlib.metadata``
finds the cloned distribution at the right version, and a generated
console-script wrapper carries the target interpreter's shebang
rather than the source's.
"""

from __future__ import annotations

import errno
import os
import subprocess
import tempfile
import unittest
import venv
from pathlib import Path, PurePosixPath
from unittest import mock

from tests.fixtures.build_fixtures import (
    FixtureSpec,
    write_pep376_install,
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
from translocate.materialize.linking import (
    FallbackPolicy,
    LinkMode,
    materialize_file,
)
from translocate.materialize.prefix import (
    MaterializeResult,
    _materialize_one,
    _Plan,
    _plan_entries,
    materialize,
)
from translocate.reconcile import reconcile
from translocate.record import parse_record
from translocate.scheme import resolve_source_scheme
from translocate.scripts import EntryPoint


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


class LinkingTests(unittest.TestCase):
    def test_copy_mode_always_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.bin"
            src.write_bytes(b"payload")
            dest = Path(tmp) / "out" / "dest.bin"
            outcome = materialize_file(
                src, dest, LinkMode.COPY, FallbackPolicy.ERROR
            )
            self.assertEqual(outcome.mode_used, LinkMode.COPY)
            self.assertEqual(dest.read_bytes(), b"payload")

    def test_symlink_mode_resolves_to_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.bin"
            src.write_bytes(b"payload")
            dest = Path(tmp) / "out" / "dest.bin"
            materialize_file(src, dest, LinkMode.SYMLINK, FallbackPolicy.ERROR)
            self.assertTrue(dest.is_symlink())
            self.assertEqual(dest.read_bytes(), b"payload")

    def test_copy_mode_preserves_executable_bit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.sh"
            src.write_bytes(b"#!/bin/sh\n")
            src.chmod(0o755)
            dest = Path(tmp) / "out" / "dest.sh"
            materialize_file(src, dest, LinkMode.COPY, FallbackPolicy.ERROR)
            self.assertTrue(os.access(dest, os.X_OK))

    def test_copy_mode_preserves_non_executable_bit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.txt"
            src.write_bytes(b"data\n")
            src.chmod(0o644)
            dest = Path(tmp) / "out" / "dest.txt"
            materialize_file(src, dest, LinkMode.COPY, FallbackPolicy.ERROR)
            self.assertFalse(os.access(dest, os.X_OK))

    def test_hardlink_falls_back_to_copy_on_exdev(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.bin"
            src.write_bytes(b"payload")
            dest = Path(tmp) / "out" / "dest.bin"
            with mock.patch(
                "translocate.materialize.linking._hardlink",
                side_effect=OSError(errno.EXDEV, "cross-device link"),
            ):
                outcome = materialize_file(
                    src, dest, LinkMode.HARDLINK, FallbackPolicy.COPY
                )
            self.assertTrue(outcome.fell_back)
            self.assertEqual(outcome.mode_used, LinkMode.COPY)
            self.assertEqual(dest.read_bytes(), b"payload")
            self.assertFalse(dest.is_symlink())

    def test_hardlink_exdev_under_error_policy_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.bin"
            src.write_bytes(b"payload")
            dest = Path(tmp) / "out" / "dest.bin"
            with mock.patch(
                "translocate.materialize.linking._hardlink",
                side_effect=OSError(errno.EXDEV, "cross-device link"),
            ):
                with self.assertRaises(OSError):
                    materialize_file(
                        src, dest, LinkMode.HARDLINK, FallbackPolicy.ERROR
                    )

    def test_hardlink_follows_symlink_to_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "target.so"
            target.write_bytes(b"realbytes")
            link = Path(tmp) / "link.so"
            link.symlink_to(target)
            dest = Path(tmp) / "out" / "dest.so"
            materialize_file(
                link, dest, LinkMode.HARDLINK, FallbackPolicy.ERROR
            )
            self.assertFalse(dest.is_symlink())
            self.assertEqual(os.stat(dest).st_ino, os.stat(target).st_ino)

    def test_hardlink_follows_symlink_chain_to_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "target.so"
            target.write_bytes(b"realbytes")
            link1 = Path(tmp) / "link1.so"
            link1.symlink_to(target)
            link2 = Path(tmp) / "link2.so"
            link2.symlink_to(link1)
            dest = Path(tmp) / "out" / "dest.so"
            materialize_file(
                link2, dest, LinkMode.HARDLINK, FallbackPolicy.ERROR
            )
            self.assertFalse(dest.is_symlink())
            self.assertEqual(os.stat(dest).st_ino, os.stat(target).st_ino)


class MaterializeEndToEndTests(unittest.TestCase):
    def test_clone_into_real_venv_passes_structural_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _, inv = _built(
                tmp_path,
                FixtureSpec(
                    name="cloneme",
                    version="0.2.0",
                    purelib_files={
                        "cloneme/__init__.py": b"value = 42\n",
                    },
                ),
            )
            venv_dir = tmp_path / "venv"
            venv.create(venv_dir, with_pip=False)
            target_python = venv_dir / "bin" / "python"
            result, _ = materialize(
                inventory=inv,
                target_python=target_python,
                link_mode=LinkMode.COPY,
            )
            self.assertGreater(result.members_materialized, 0)
            self.assertTrue((result.target_dist_info_dir / "RECORD").exists())
            # The target interpreter must see the new dist.
            probe = subprocess.run(
                [
                    str(target_python),
                    "-c",
                    "from importlib import metadata; "
                    "print(metadata.version('cloneme'))",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(probe.returncode, 0, probe.stderr)
            self.assertEqual(probe.stdout.strip(), "0.2.0")

    def test_entrypoint_script_regenerated_against_target_interpreter(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _, inv = _built(
                tmp_path,
                FixtureSpec(
                    name="hasentry",
                    version="0.1.0",
                    purelib_files={
                        "hasentry/__init__.py": (
                            b"def main():\n" b"    print('called')\n"
                        ),
                    },
                    scripts_files={
                        "doit": (b"#!/source/python\nprint('source')\n"),
                    },
                    entry_points={
                        "console_scripts": {"doit": "hasentry:main"},
                    },
                ),
            )
            venv_dir = tmp_path / "venv"
            venv.create(venv_dir, with_pip=False)
            target_python = venv_dir / "bin" / "python"
            materialize(
                inventory=inv,
                target_python=target_python,
                link_mode=LinkMode.COPY,
            )
            generated = venv_dir / "bin" / "doit"
            self.assertTrue(generated.exists())
            shebang = generated.read_bytes().splitlines()[0]
            self.assertIn(b"python", shebang)
            self.assertNotIn(b"/source/python", shebang)


def _scheme(root: Path) -> InstallScheme:
    site = root / "site-packages"
    return InstallScheme(
        purelib=site,
        platlib=site,
        scripts=root / "bin",
        headers=root / "include",
        data=root,
        prefix=root,
        executable=root / "bin" / "python",
        soabi=None,
        ext_suffix=None,
    )


def _entry_point_entry() -> Entry:
    return Entry(
        category=Category.SCRIPTS,
        rel_path=PurePosixPath("tool"),
        content_ref=ContentRef.regenerate_script("tool"),
        flags=EntryFlags(is_entrypoint_script=True),
    )


def _entry_point_plan(dest: Path) -> _Plan:
    return _Plan(
        entry=_entry_point_entry(),
        dest=dest,
        entry_point=EntryPoint(name="tool", module="tool", attr="main"),
    )


def _entry_point_inventory(
    scheme: InstallScheme, entry_points: dict
) -> Inventory:
    return Inventory(
        dist_name="tool",
        dist_version="1.0",
        dist_info_dir=scheme.purelib / "tool-1.0.dist-info",
        source_scheme=scheme,
        root_is_purelib=True,
        entries=[_entry_point_entry()],
        entry_points=entry_points,
    )


def _fresh_result(scheme: InstallScheme) -> MaterializeResult:
    return MaterializeResult(
        target_prefix=scheme.prefix,
        target_dist_info_dir=scheme.purelib / "tool-1.0.dist-info",
        members_materialized=0,
    )


class MaterializeEntryPointTests(unittest.TestCase):
    def test_chmod_failure_on_wrapper_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scheme = _scheme(root)
            plan = _entry_point_plan(root / "bin" / "tool")
            with mock.patch(
                "translocate.materialize.prefix.os.chmod",
                side_effect=OSError("read-only"),
            ):
                with self.assertLogs(
                    "translocate.materialize.prefix", level="WARNING"
                ) as logs:
                    _materialize_one(
                        plan,
                        scheme,
                        LinkMode.COPY,
                        FallbackPolicy.COPY,
                        {},
                    )
            self.assertTrue(
                any("execute bit" in msg for msg in logs.output),
                f"no chmod warning surfaced; logs were {logs.output}",
            )

    def test_entry_point_flagged_but_absent_is_unreachable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scheme = _scheme(Path(tmp))
            inv = _entry_point_inventory(scheme, {})
            result = _fresh_result(scheme)
            with self.assertRaises(RuntimeError):
                _plan_entries(inv, scheme, inv.dist_info_dir, result)

    def test_malformed_entry_point_recorded_and_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scheme = _scheme(Path(tmp))
            inv = _entry_point_inventory(
                scheme, {"console_scripts": {"tool": "not a target"}}
            )
            result = _fresh_result(scheme)
            plans = _plan_entries(inv, scheme, inv.dist_info_dir, result)
            self.assertEqual(result.excluded_malformed_entry_points, ["tool"])
            self.assertFalse(
                any(p.entry.flags.is_entrypoint_script for p in plans)
            )


if __name__ == "__main__":
    unittest.main()
