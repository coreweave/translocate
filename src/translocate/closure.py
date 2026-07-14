# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: MIT
# SPDX-PackageName: translocate

"""Walk a distribution's actually-installed dependency closure.

Starting from pre-located roots, follow ``Requires-Dist`` breadth-first
through the source environment's already-installed set. Names absent
from that set are ignored: the source installer already evaluated
environment markers and extras when it placed those deps.

A dependency that resolves to a ``Refusal`` (editable, egg-info,
incomplete metadata) is set aside as untranslocatable rather than
followed further. It still appears in an emitted deps file, pinned to
its installed version.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from translocate.errors import Refusal
from translocate.locate import LocatedDistribution, canonicalize, locate

_NAME_RE = re.compile(r"^([A-Za-z0-9._-]+)")


def _extract_name(requires_dist: str) -> Optional[str]:
    m = _NAME_RE.match(requires_dist.strip())
    return m.group(1) if m else None


def _installed_index() -> Dict[str, str]:
    """Map canonical PEP 503 name -> original name as it appears in METADATA."""
    out: Dict[str, str] = {}
    for d in metadata.distributions():
        if d.metadata is None:
            continue
        n = d.metadata["Name"]
        if not n:
            continue
        out[canonicalize(n)] = n
    return out


@dataclass(frozen=True, slots=True)
class LocatedMember:
    name: str
    canonical_name: str
    located: LocatedDistribution


@dataclass(frozen=True, slots=True)
class RefusedMember:
    name: str
    canonical_name: str
    refusal: Refusal


ClosureMember = Union[LocatedMember, RefusedMember]


def _member_from_locate(
    name: str, result: Union[LocatedDistribution, Refusal]
) -> ClosureMember:
    canonical = canonicalize(name)
    if isinstance(result, Refusal):
        return RefusedMember(
            name=name, canonical_name=canonical, refusal=result
        )
    return LocatedMember(name=name, canonical_name=canonical, located=result)


@dataclass(slots=True)
class Closure:
    """The roots, their reachable deps, and the deps that were refused.

    ``roots`` and ``members`` are translocatable; ``skipped`` holds the
    untranslocatable deps. Each canonical name appears at most once
    across the three lists.
    """

    roots: List[LocatedMember] = field(default_factory=list)
    members: List[LocatedMember] = field(default_factory=list)
    skipped: List[RefusedMember] = field(default_factory=list)
    # Adjacency among cloneable nodes: a canonical name maps to the
    # canonical names of its direct dependencies that are themselves in
    # the closure. Complete only for a full (``depth=None``) walk.
    edges: Dict[str, Tuple[str, ...]] = field(default_factory=dict)


def _requires_of(located: LocatedDistribution) -> List[str]:
    """Parse ``Requires-Dist`` lines into a list of bare names."""
    metadata_path = located.dist_info_dir / "METADATA"
    if not metadata_path.exists():
        return []
    names: List[str] = []
    for raw_line in metadata_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        if not raw_line.lower().startswith("requires-dist:"):
            continue
        value = raw_line.split(":", 1)[1].strip()
        # PEP 508 dep specifier: ``name [extras] specifier ; marker``.
        # Only the bare name identifies the distribution.
        name = _extract_name(value)
        if name:
            names.append(name)
    return names


def closure_for(
    roots: List[Tuple[str, LocatedDistribution]],
    *,
    depth: Optional[int] = None,
) -> Closure:
    """Build a closure from pre-located roots.

    ``depth=None`` walks the full transitive closure; ``depth=1``
    returns the roots plus their direct ``Requires-Dist``; ``depth=0``
    returns just the roots. A dependency reachable from more than one
    root appears once.
    """
    index = _installed_index()
    visited: Set[str] = set()
    result_roots: List[LocatedMember] = []
    members: List[LocatedMember] = []
    skipped: List[RefusedMember] = []
    # canonical name -> canonical names of its direct ``Requires-Dist``,
    # captured as each node is walked so the edges can be assembled
    # once the full cloneable set is known.
    direct: Dict[str, List[str]] = {}

    current_layer: List[Tuple[str, LocatedDistribution]] = []
    for name, located in roots:
        canonical = canonicalize(name)
        if canonical in visited:
            continue
        visited.add(canonical)
        result_roots.append(
            LocatedMember(name=name, canonical_name=canonical, located=located)
        )
        current_layer.append((canonical, located))

    layer = 0
    while current_layer and (depth is None or layer < depth):
        next_layer: List[Tuple[str, LocatedDistribution]] = []
        for parent_canonical, parent in current_layer:
            reqs = [canonicalize(req) for req in _requires_of(parent)]
            direct[parent_canonical] = reqs
            for canonical in reqs:
                if canonical in visited:
                    continue
                original = index.get(canonical)
                if original is None:
                    continue
                visited.add(canonical)
                member = _member_from_locate(original, locate(original))
                if isinstance(member, RefusedMember):
                    skipped.append(member)
                else:
                    members.append(member)
                    next_layer.append((member.canonical_name, member.located))
        current_layer = next_layer
        layer += 1

    cloneable = {r.canonical_name for r in result_roots}
    cloneable.update(m.canonical_name for m in members)
    edges = {
        node: tuple(dep for dep in deps if dep in cloneable)
        for node, deps in direct.items()
    }
    for node in cloneable:
        edges.setdefault(node, ())

    return Closure(
        roots=result_roots, members=members, skipped=skipped, edges=edges
    )


def deps_file_lines(plan: Closure) -> List[str]:
    """Pinned ``name==version`` lines for every member of ``plan``.

    Lists every root, then every other reachable member, then every
    member that couldn't be translocated, with each canonical name
    appearing once. A member whose version can't be determined is
    omitted.
    """
    lines: List[str] = []
    seen: Set[str] = set()
    for member in [*plan.roots, *plan.members, *plan.skipped]:
        if member.canonical_name in seen:
            continue
        seen.add(member.canonical_name)
        version = _version_of(member)
        if version is None:
            continue
        lines.append(f"{member.name}=={version}")
    return lines


def write_deps_file(plan: Closure, output: Path) -> None:
    """Write the pinned ``name==version`` deps lines for ``plan``."""
    with output.open("w", encoding="utf-8") as f:
        f.write("\n".join(deps_file_lines(plan)))
        f.write("\n")


def _version_of(member: ClosureMember) -> Optional[str]:
    if isinstance(member, LocatedMember):
        return member.located.version
    try:
        return metadata.distribution(member.name).version
    except metadata.PackageNotFoundError:
        return None


def filter_already_at_target(
    plan: Closure, target_installed: List[str]
) -> Tuple[List[LocatedMember], List[LocatedMember]]:
    """Split planned members into (translocate, skip-already-present)."""
    target_canonical = {canonicalize(n) for n in target_installed}
    root_canonical = {r.canonical_name for r in plan.roots}
    translocate: List[LocatedMember] = []
    skip: List[LocatedMember] = []
    candidates = list(plan.roots) + [
        m for m in plan.members if m.canonical_name not in root_canonical
    ]
    for member in candidates:
        if member.canonical_name in target_canonical:
            skip.append(member)
        else:
            translocate.append(member)
    return translocate, skip
