# SPDX-FileCopyrightText: 2026 CoreWeave
#
# SPDX-License-Identifier: MIT

"""Typed refusals for cases the translocate pipeline declines to handle.

A ``Refusal`` is a plain frozen dataclass carrying just the data that
describes the problem. Callers (the CLI, the dep-closure walker) pick
the response with structural pattern-matching: invalid name → hard
error, untranslocatable dep → skip-and-warn, and so on.

``TranslocateRefused`` is the exception wrapper used when a function
deep in the stack needs to surface a refusal without a return-value
channel. The CLI's top level catches it and translates it to the
right exit code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True, slots=True)
class Refusal:
    def __str__(self) -> str:
        return _describe(self)


@dataclass(frozen=True, slots=True)
class DistributionNotFound(Refusal):
    name: str
    candidates: List[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class InvalidDistributionName(Refusal):
    name: str
    reason: str


@dataclass(frozen=True, slots=True)
class EditableInstall(Refusal):
    dist_info: Path
    detected_via: str


@dataclass(frozen=True, slots=True)
class EggInfoOnly(Refusal):
    location: Path


@dataclass(frozen=True, slots=True)
class IncompleteMetadata(Refusal):
    dist_info: Optional[Path]
    detail: str


@dataclass(frozen=True, slots=True)
class CloneToSelf(Refusal):
    source_prefix: Path
    target_prefix: Path


@dataclass(frozen=True, slots=True)
class SourceTargetSharedRoot(Refusal):
    category: str
    path: Path


@dataclass(frozen=True, slots=True)
class TargetNotAPython(Refusal):
    target: Path
    detail: str


@dataclass(frozen=True, slots=True)
class ABIMismatch(Refusal):
    source_soabi: Optional[str]
    target_soabi: Optional[str]
    source_ext_suffix: Optional[str]
    target_ext_suffix: Optional[str]


@dataclass(frozen=True, slots=True)
class AlreadyPresentAtTarget(Refusal):
    name: str
    source_version: str
    target_version: str


@dataclass(frozen=True, slots=True)
class HashMismatch(Refusal):
    dist_name: str
    paths: List[Path]


class TranslocateRefused(Exception):
    """Wraps a Refusal for raising across CLI boundaries."""

    def __init__(self, refusal: Refusal) -> None:
        self.refusal = refusal
        super().__init__(str(refusal))


def _describe(r: Refusal) -> str:
    match r:
        case DistributionNotFound(name=n, candidates=c):
            suffix = f" Did you mean: {', '.join(c)}?" if c else ""
            return (
                f"distribution {n!r} not found in source environment.{suffix}"
            )
        case InvalidDistributionName(name=n, reason=why):
            return f"invalid distribution name {n!r}: {why}"
        case EditableInstall(dist_info=p, detected_via=how):
            return (
                f"editable install at {p} (detected via {how}) cannot be "
                f"translocated. Its RECORD describes the import shim, "
                f"not the source tree."
            )
        case EggInfoOnly(location=p):
            return (
                f"legacy egg-info layout at {p} has no faithful manifest, "
                f"so this install cannot be enumerated."
            )
        case IncompleteMetadata(dist_info=p, detail=d):
            if p is None:
                return f"incomplete metadata: {d}"
            return f"incomplete metadata at {p}: {d}"
        case CloneToSelf(source_prefix=s, target_prefix=t):
            return (
                f"source and target resolve to the same prefix ({s}). "
                f"Refusing to clone an interpreter onto itself."
            )
        case SourceTargetSharedRoot(category=c, path=p):
            return (
                f"source and target share the {c} root at {p}. Refusing "
                f"to link or copy files onto themselves."
            )
        case TargetNotAPython(target=t, detail=d):
            return f"target {t} is not a Python install: {d}"
        case ABIMismatch(
            source_soabi=s_so,
            target_soabi=t_so,
            source_ext_suffix=s_ex,
            target_ext_suffix=t_ex,
        ):
            return (
                f"ABI mismatch between source and target: "
                f"SOABI {s_so!r} vs {t_so!r}, EXT_SUFFIX {s_ex!r} vs "
                f"{t_ex!r}. Compiled extensions will not load."
            )
        case AlreadyPresentAtTarget(
            name=n, source_version=sv, target_version=tv
        ):
            return (
                f"skipping {n}: already installed at the target "
                f"(source {sv}, target {tv})."
            )
        case HashMismatch(dist_name=n, paths=ps):
            shown = ", ".join(str(p) for p in ps[:3])
            if len(ps) > 3:
                shown += f", and {len(ps) - 3} more"
            return (
                f"{len(ps)} file(s) in {n} no longer match their "
                f"recorded hash: {shown}"
            )
        case _:
            return f"refused: {r!r}"
