# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: MIT
# SPDX-PackageName: translocate

"""Dependency-closure walk, name extraction, and deps-file emission."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.fixtures.build_fixtures import FixtureSpec, write_pep376_install
from translocate.closure import (
    Closure,
    LocatedMember,
    _extract_name,
    closure_for,
    write_deps_file,
)
from translocate.locate import InstallState, LocatedDistribution


class NameExtractionTests(unittest.TestCase):
    def test_extracts_bare_name(self) -> None:
        self.assertEqual(_extract_name("foo"), "foo")

    def test_extracts_through_version(self) -> None:
        self.assertEqual(_extract_name("requests >= 2.0"), "requests")

    def test_extracts_through_extras_and_marker(self) -> None:
        self.assertEqual(
            _extract_name("requests[security]>=2.0 ; python_version<'3.10'"),
            "requests",
        )

    def test_extracts_dotted_underscored(self) -> None:
        self.assertEqual(_extract_name("foo.bar_baz>=1.0"), "foo.bar_baz")


def _located(name: str, version: str) -> LocatedDistribution:
    return LocatedDistribution(
        name=name,
        canonical_name=name.lower(),
        version=version,
        dist_info_dir=Path("/tmp/none"),
        install_state=InstallState.PEP376_WHEEL,
    )


class DepsFileTests(unittest.TestCase):
    def test_writes_resolved_versions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "reqs.txt"
            plan = Closure(
                roots=[LocatedMember("root", "root", _located("root", "1.0"))],
                members=[
                    LocatedMember("depX", "depx", _located("depX", "2.3"))
                ],
            )
            write_deps_file(plan, target)
            lines = target.read_text(encoding="utf-8").strip().splitlines()
            self.assertIn("root==1.0", lines)
            self.assertIn("depX==2.3", lines)


class ClosureWalkTests(unittest.TestCase):
    def _make(self, root: Path, name: str, reqs=()) -> LocatedDistribution:
        layout = write_pep376_install(
            root / name,
            FixtureSpec(name=name, version="1.0", requires_dist=list(reqs)),
        )
        return LocatedDistribution(
            name=name,
            canonical_name=name,
            version="1.0",
            dist_info_dir=layout.dist_info,
            install_state=InstallState.PEP376_WHEEL,
        )

    def test_shared_transitive_dep_walked_once(self) -> None:
        # A dependency that two roots both pull in appears once in the
        # closure, not once per root.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = self._make(root, "shared")
            alpha = self._make(root, "alpha", reqs=["shared"])
            beta = self._make(root, "beta", reqs=["shared"])
            located_by_name = {
                "alpha": alpha,
                "beta": beta,
                "shared": shared,
            }
            index = {"alpha": "alpha", "beta": "beta", "shared": "shared"}
            with (
                mock.patch(
                    "translocate.closure._installed_index", return_value=index
                ),
                mock.patch(
                    "translocate.closure.locate",
                    side_effect=lambda n: located_by_name[n],
                ),
            ):
                plan = closure_for(
                    [("alpha", alpha), ("beta", beta)], depth=None
                )
            members = [m.canonical_name for m in plan.members]
            self.assertEqual(members.count("shared"), 1)
            self.assertEqual(
                [r.canonical_name for r in plan.roots], ["alpha", "beta"]
            )

    def test_edges_record_in_closure_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            leaf = self._make(root, "leaf")
            mid = self._make(root, "mid", reqs=["leaf"])
            top = self._make(root, "top", reqs=["mid", "leaf"])
            located_by_name = {"top": top, "mid": mid, "leaf": leaf}
            index = {"top": "top", "mid": "mid", "leaf": "leaf"}
            with (
                mock.patch(
                    "translocate.closure._installed_index", return_value=index
                ),
                mock.patch(
                    "translocate.closure.locate",
                    side_effect=lambda n: located_by_name[n],
                ),
            ):
                plan = closure_for([("top", top)], depth=None)
            self.assertEqual(set(plan.edges["top"]), {"mid", "leaf"})
            self.assertEqual(plan.edges["mid"], ("leaf",))
            self.assertEqual(plan.edges["leaf"], ())

    def test_depth_one_stops_at_direct_deps(self) -> None:
        # With a chain of root -> mid -> leaf, depth=1 keeps the direct
        # dependency and stops before its transitive one.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            leaf = self._make(root, "leaf")
            mid = self._make(root, "mid", reqs=["leaf"])
            alpha = self._make(root, "alpha", reqs=["mid"])
            located_by_name = {"alpha": alpha, "mid": mid, "leaf": leaf}
            index = {"alpha": "alpha", "mid": "mid", "leaf": "leaf"}
            with (
                mock.patch(
                    "translocate.closure._installed_index", return_value=index
                ),
                mock.patch(
                    "translocate.closure.locate",
                    side_effect=lambda n: located_by_name[n],
                ),
            ):
                plan = closure_for([("alpha", alpha)], depth=1)
            members = [m.canonical_name for m in plan.members]
            self.assertEqual(members, ["mid"])

    def test_duplicate_roots_dedup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            located = self._make(Path(tmp), "alpha")
            plan = closure_for(
                [("alpha", located), ("alpha", located)], depth=0
            )
            self.assertEqual(len(plan.roots), 1)


if __name__ == "__main__":
    unittest.main()
