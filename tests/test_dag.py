# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: MIT
# SPDX-PackageName: translocate

"""Dependency-DAG traversal: pruning and shared-dependency survival."""

from __future__ import annotations

import asyncio
import unittest

from translocate._dag import traverse


def _walk(edges, roots, failures):
    """Run ``traverse`` with a fake step that fails the named nodes.

    Returns ``(ran, results)`` where ``ran`` is the list of nodes the
    step touched and ``results`` maps each run node to its success
    bool.
    """
    nodes = set(edges) | {dep for deps in edges.values() for dep in deps}
    nodes.update(roots)
    ran = []

    def run_node(node):
        ran.append(node)
        return node not in failures

    results = asyncio.run(
        traverse(nodes, edges, roots, run_node, lambda ok: ok)
    )
    return ran, results


class TraverseTests(unittest.TestCase):
    def test_failed_root_prunes_its_subtree(self) -> None:
        edges = {"a": ("b",), "b": ("c",), "c": ()}
        ran, results = _walk(edges, ["a"], failures={"a"})
        self.assertEqual(ran, ["a"])
        self.assertNotIn("b", results)
        self.assertNotIn("c", results)

    def test_failed_intermediate_prunes_below_it(self) -> None:
        edges = {"a": ("b",), "b": ("c",), "c": ()}
        ran, results = _walk(edges, ["a"], failures={"b"})
        self.assertEqual(set(ran), {"a", "b"})
        self.assertNotIn("c", results)

    def test_shared_dep_survives_a_failed_parent(self) -> None:
        edges = {"a": ("s",), "b": ("s",), "s": ()}
        ran, results = _walk(edges, ["a", "b"], failures={"a"})
        self.assertEqual(ran.count("s"), 1)
        self.assertIn("s", results)

    def test_shared_dep_runs_once_when_both_parents_succeed(self) -> None:
        edges = {"a": ("s",), "b": ("s",), "s": ()}
        ran, _ = _walk(edges, ["a", "b"], failures=set())
        self.assertEqual(ran.count("s"), 1)

    def test_root_outside_node_set_raises(self) -> None:
        with self.assertRaises(ValueError):
            asyncio.run(
                traverse({"a"}, {}, ["b"], lambda n: True, lambda ok: ok)
            )


if __name__ == "__main__":
    unittest.main()
