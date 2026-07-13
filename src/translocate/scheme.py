# SPDX-FileCopyrightText: 2026 CoreWeave
#
# SPDX-License-Identifier: MIT

"""Resolve install schemes and classify absolute paths into wheel categories.

Each Python interpreter advertises five install-scheme directories
through ``sysconfig``: ``purelib``, ``platlib``, ``scripts``,
``headers``, and ``data``. Reading them from the interpreter that
owns the install handles distro-specific layouts (e.g. Debian's
``dist-packages`` split) without any hardcoded layout knowledge.

``classify`` maps an absolute file path back to ``(category,
category-relative path)`` by longest-prefix match, with specific
categories searched ahead of ``data``. The ``data`` root typically
equals the install prefix and so contains ``scripts`` and
``headers``; without the ordering rule, a script under the prefix
would be mis-classified as data.
"""

from __future__ import annotations

import logging
import os
import sys
import sysconfig
from pathlib import Path, PurePosixPath
from typing import Mapping, Optional, Tuple, cast

from translocate.inventory import Category, InstallScheme

logger = logging.getLogger(__name__)


def _real(path: os.PathLike) -> Path:
    return Path(os.path.realpath(path))


def resolve_source_scheme(
    raw_paths: Optional[Mapping[str, str]] = None,
    soabi: Optional[str] = None,
    ext_suffix: Optional[str] = None,
    executable: Optional[str] = None,
    prefix: Optional[str] = None,
) -> InstallScheme:
    """Build an ``InstallScheme`` from the running interpreter's sysconfig.

    ``raw_paths`` and the metadata-shaped optional arguments let tests
    and ``scheme_from_target_query`` substitute a synthetic scheme;
    everything left as ``None`` reads from the current interpreter.
    """
    paths = dict(raw_paths) if raw_paths is not None else sysconfig.get_paths()
    purelib = _real(paths["purelib"])
    platlib = _real(paths["platlib"])
    scripts = _real(paths["scripts"])
    headers_raw = paths.get("include") or paths.get("headers")
    if headers_raw is None:
        raise KeyError("sysconfig paths missing 'include'/'headers'")
    headers = _real(headers_raw)
    data = _real(paths["data"])
    return InstallScheme(
        purelib=purelib,
        platlib=platlib,
        scripts=scripts,
        headers=headers,
        data=data,
        prefix=Path(prefix) if prefix else Path(sys.prefix),
        executable=Path(executable) if executable else Path(sys.executable),
        soabi=soabi if soabi is not None else sysconfig.get_config_var("SOABI"),
        ext_suffix=(
            ext_suffix
            if ext_suffix is not None
            else sysconfig.get_config_var("EXT_SUFFIX")
        ),
    )


def _wheel_root_is_purelib(dist_info_dir: Path) -> Optional[bool]:
    """Read ``Root-Is-Purelib`` from a distribution's ``WHEEL``.

    Returns ``None`` when ``WHEEL`` is unreadable or carries no such
    line.
    """
    try:
        text = (dist_info_dir / "WHEEL").read_text(encoding="utf-8")
    except OSError as e:
        logger.debug("could not read WHEEL at %s: %s", dist_info_dir, e)
        return None
    for raw_line in text.splitlines():
        key, sep, value = raw_line.partition(":")
        if sep and key.strip() == "Root-Is-Purelib":
            return value.strip().lower() == "true"
    return None


def distribution_root_is_purelib(
    dist_info_dir: Path, scheme: InstallScheme
) -> bool:
    """Whether the distribution at ``dist_info_dir`` is rooted at purelib.

    Prefers the ``Root-Is-Purelib`` field of the distribution's
    ``WHEEL``. When that's absent, falls back to which of
    purelib/platlib the dist-info sits under, which tells them apart
    only when they differ. When the two coincide and ``WHEEL`` is
    silent, returns ``False`` (platlib-rooted) as the more restrictive
    default.
    """
    recorded = _wheel_root_is_purelib(dist_info_dir)
    if recorded is not None:
        logger.debug(
            "%s: root_is_purelib=%s from WHEEL", dist_info_dir, recorded
        )
        return recorded
    if scheme.purelib != scheme.platlib:
        located = _real(dist_info_dir.parent) == scheme.purelib
        logger.debug(
            "%s: root_is_purelib=%s from dist-info location",
            dist_info_dir,
            located,
        )
        return located
    logger.debug("%s: root_is_purelib=False by default", dist_info_dir)
    return False


def scheme_from_target_query(query: Mapping[str, object]) -> InstallScheme:
    """Build a ``InstallScheme`` from a remote interpreter's probe output.

    The ``query`` dict matches the JSON shape ``_target`` produces
    by spawning the target Python and reading its ``sysconfig`` back.
    """
    nullable = {"soabi", "ext_suffix"}
    for key in (
        "purelib",
        "platlib",
        "scripts",
        "headers",
        "data",
        "executable",
        "prefix",
        "soabi",
        "ext_suffix",
    ):
        value = query.get(key)
        if value is None:
            if key not in nullable:
                raise TypeError(
                    f"target scheme probe field {key!r} is required"
                )
        elif not isinstance(value, str):
            raise TypeError(
                f"target scheme probe field {key!r} must be a string, "
                f"got {value!r}"
            )
    return resolve_source_scheme(
        raw_paths={
            "purelib": cast(str, query["purelib"]),
            "platlib": cast(str, query["platlib"]),
            "scripts": cast(str, query["scripts"]),
            "include": cast(str, query["headers"]),
            "data": cast(str, query["data"]),
        },
        soabi=cast(Optional[str], query.get("soabi")),
        ext_suffix=cast(Optional[str], query.get("ext_suffix")),
        executable=cast(str, query["executable"]),
        prefix=cast(str, query["prefix"]),
    )


def _category_order(
    scheme: InstallScheme, root_is_purelib: bool
) -> Tuple[Tuple[Category, Path], ...]:
    purelib_pair = (Category.ROOT_PURELIB, scheme.purelib)
    platlib_pair = (Category.ROOT_PLATLIB, scheme.platlib)
    root_first = (
        (purelib_pair, platlib_pair)
        if root_is_purelib
        else (platlib_pair, purelib_pair)
    )
    return (
        (Category.SCRIPTS, scheme.scripts),
        (Category.HEADERS, scheme.headers),
        root_first[0],
        root_first[1],
        (Category.DATA, scheme.data),
    )


def classify(
    absolute_path: Path, scheme: InstallScheme, root_is_purelib: bool
) -> Tuple[Category, PurePosixPath]:
    """Map an absolute path to ``(category, category-relative path)``.

    Resolves symlinks in the ancestor directories before matching, so
    a scheme root reached through a symlinked parent still matches, but
    leaves the final component alone: a recorded file that is itself a
    symlink keeps its own location instead of collapsing onto its
    target. Picks the deepest scheme root that contains the path; when
    two roots tie on depth, the iteration order in ``_category_order``
    wins, which keeps the specific categories ahead of ``data``.
    """
    parent = Path(os.path.realpath(absolute_path.parent))
    real = parent / absolute_path.name
    best_len = -1
    best: Optional[Tuple[Category, PurePosixPath]] = None
    for category, root in _category_order(scheme, root_is_purelib):
        try:
            rel = real.relative_to(root)
        except ValueError:
            continue
        depth = len(root.parts)
        if depth > best_len:
            best_len = depth
            best = (category, PurePosixPath(rel.as_posix()))
    if best is None:
        return (
            Category.OUTSIDE_KNOWN_SCHEME,
            PurePosixPath(absolute_path.as_posix()),
        )
    return best


def category_destination(
    target_scheme: InstallScheme, category: Category, root_is_purelib: bool
) -> Optional[Path]:
    """Return the directory in ``target_scheme`` that owns ``category``,
    or ``None`` for ``OUTSIDE_KNOWN_SCHEME``.
    """
    if category == Category.ROOT_PURELIB:
        return target_scheme.purelib
    if category == Category.ROOT_PLATLIB:
        return target_scheme.platlib
    if category == Category.SCRIPTS:
        return target_scheme.scripts
    if category == Category.HEADERS:
        return target_scheme.headers
    if category == Category.DATA:
        return target_scheme.data
    if category == Category.DIST_INFO:
        if root_is_purelib:
            return target_scheme.purelib
        return target_scheme.platlib
    return None
