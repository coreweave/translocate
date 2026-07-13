# SPDX-FileCopyrightText: 2026 CoreWeave
#
# SPDX-License-Identifier: MIT

"""Run a node-processing step over a dependency DAG, sources first.

``traverse`` starts at the roots and launches a node's children only
once that node's step succeeds, so a node is reached exactly when at
least one path to it from a root succeeded end to end. A node whose
every path runs through a failed ancestor is never run; a node shared
by a failed and a surviving ancestor still runs, once.

The per-node step is an opaque blocking callable run in a worker
thread; the event loop schedules the walk and a semaphore bounds how
many steps run at once. Nodes absent from the returned mapping were
pruned (no successful path reached them).
"""

from __future__ import annotations

import asyncio
from typing import Callable, Dict, Iterable, Mapping, Set, Tuple, TypeVar

T = TypeVar("T")


async def traverse(
    nodes: Set[str],
    edges: Mapping[str, Tuple[str, ...]],
    roots: Iterable[str],
    run_node: Callable[[str], T],
    is_success: Callable[[T], bool],
    *,
    concurrency: int = 8,
) -> Dict[str, T]:
    """Walk ``edges`` from ``roots``, running ``run_node`` per node.

    Only nodes in ``nodes`` are run; edges to anything outside that set
    are ignored. Returns each run node's result keyed by node; a node
    that was pruned is absent. ``run_node`` runs in a worker thread,
    with at most ``concurrency`` running at once.
    """
    sem = asyncio.Semaphore(concurrency)
    tasks: Dict[str, asyncio.Task] = {}
    results: Dict[str, T] = {}

    async def visit(node: str) -> None:
        async with sem:
            results[node] = await asyncio.to_thread(run_node, node)
        if not is_success(results[node]):
            return
        children = [dep for dep in edges.get(node, ()) if dep in nodes]
        for dep in children:
            if dep not in tasks:
                tasks[dep] = asyncio.create_task(visit(dep))
        if children:
            await asyncio.gather(*(tasks[dep] for dep in children))

    root_tasks = []
    for root in roots:
        if root not in nodes:
            raise ValueError(f"root {root!r} is not in the node set")
        if root not in tasks:
            tasks[root] = asyncio.create_task(visit(root))
            root_tasks.append(tasks[root])
    if root_tasks:
        await asyncio.gather(*root_tasks)
    return results
