# SPDX-FileCopyrightText: 2026 CoreWeave
#
# SPDX-License-Identifier: MIT

"""Probe a target Python interpreter via subprocess.

The current process cannot directly introspect another interpreter's
``sysconfig``, installed distributions, or import behaviour. This
module spawns the target Python with small ``-c`` snippets and parses
the JSON it prints back, so every caller gets uniform escaping and
error handling.
"""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from translocate.errors import TargetNotAPython, TranslocateRefused


def _run(python: Path, code: str) -> Tuple[int, str, str]:
    # ``-I`` isolates the probe from PYTHONPATH/PYTHONHOME, the user
    # site directory, and the cwd, so it reports the target's intrinsic
    # config rather than a value the surrounding environment skewed.
    proc = subprocess.run(
        [python, "-I", "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


_SCHEME_CODE = textwrap.dedent("""
    import json, sys, sysconfig
    paths = sysconfig.get_paths()
    out = {
        "purelib": paths.get("purelib"),
        "platlib": paths.get("platlib"),
        "scripts": paths.get("scripts"),
        "headers": paths.get("include") or paths.get("headers"),
        "data": paths.get("data"),
        "prefix": sys.prefix,
        "executable": sys.executable,
        "soabi": sysconfig.get_config_var("SOABI"),
        "ext_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "version_info": list(sys.version_info[:3]),
    }
    print(json.dumps(out))
    """).strip()


def find_target_interpreter(prefix: Path) -> Path:
    """Locate ``python`` under a target prefix.

    Accepts the prefix dir itself, a venv layout (``<prefix>/bin/python``),
    or a path that already points at the interpreter executable.
    """
    if prefix.is_file():
        return prefix
    candidates = [
        prefix / "bin" / "python",
        prefix / "bin" / "python3",
        prefix / "Scripts" / "python.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise TranslocateRefused(
        TargetNotAPython(
            target=prefix,
            detail="no python executable found under bin/ or Scripts/",
        )
    )


def query_target_scheme(python: Path) -> Dict[str, object]:
    rc, out, err = _run(python, _SCHEME_CODE)
    if rc != 0:
        raise TranslocateRefused(
            TargetNotAPython(
                target=python,
                detail=f"sysconfig probe failed (rc={rc}): {err.strip()}",
            )
        )
    return json.loads(out)


_METADATA_VERSION_CODE = textwrap.dedent("""
    import json, sys, sysconfig
    from importlib import metadata
    name = sys.argv[1]
    paths = sysconfig.get_paths()
    search = list({paths["purelib"], paths["platlib"]})
    dists = list(metadata.distributions(name=name, path=search))
    if dists:
        print(json.dumps({"ok": True, "version": dists[0].version}))
    else:
        print(json.dumps({"ok": False}))
    """).strip()


def query_target_version(python: Path, name: str) -> Optional[str]:
    proc = subprocess.run(
        [python, "-I", "-c", _METADATA_VERSION_CODE, name],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise TranslocateRefused(
            TargetNotAPython(
                target=python,
                detail=(
                    f"metadata.version probe for {name!r} failed "
                    f"(rc={proc.returncode}): {proc.stderr.strip()}"
                ),
            )
        )
    data = json.loads(proc.stdout)
    return data.get("version") if data.get("ok") else None


_INSTALLED_NAMES_CODE = textwrap.dedent("""
    import json, sysconfig
    from importlib import metadata
    paths = sysconfig.get_paths()
    search = list({paths["purelib"], paths["platlib"]})
    names = sorted({d.metadata["Name"]
                    for d in metadata.distributions(path=search)
                    if d.metadata and d.metadata["Name"]})
    print(json.dumps(names))
    """).strip()


def query_target_installed_names(python: Path) -> List[str]:
    rc, out, err = _run(python, _INSTALLED_NAMES_CODE)
    if rc != 0:
        raise TranslocateRefused(
            TargetNotAPython(
                target=python,
                detail=(
                    f"installed-names probe failed (rc={rc}): " f"{err.strip()}"
                ),
            )
        )
    return json.loads(out)


_SMOKE_IMPORT_CODE = textwrap.dedent("""
    import sys, sysconfig
    paths = sysconfig.get_paths()
    # Resolve the import against the interpreter's own site directories
    # first, so a path a .pth or sitecustomize injected can't shadow
    # the copy that was just materialized there.
    for key in ("platlib", "purelib"):
        p = paths.get(key)
        if p:
            sys.path.insert(0, p)
    __import__(sys.argv[1])
    """).strip()


def smoke_import(python: Path, name: str) -> Tuple[bool, str]:
    proc = subprocess.run(
        [python, "-I", "-c", _SMOKE_IMPORT_CODE, name],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return True, ""
    return False, proc.stderr.strip()
