# SPDX-FileCopyrightText: 2026 CoreWeave
#
# SPDX-License-Identifier: MIT

"""Render and re-neutralize console_scripts / gui_scripts wrappers.

Entry-point wrappers ship with the source interpreter's absolute
path baked into their shebang. Both translocate paths regenerate
them: the materializer renders fresh wrappers against the target's
interpreter, and the wheel emitter drops the source wrappers so the
installer of the new wheel renders its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_TARGET_RE = re.compile(
    r"^(?P<module>\w+(?:\.\w+)*)\s*:\s*"
    r"(?P<attr>\w+(?:\.\w+)*)(?:\s*\[.*\])?\s*$"
)


@dataclass(frozen=True)
class EntryPoint:
    name: str
    module: str
    attr: str


def parse_target(name: str, target: str) -> Optional[EntryPoint]:
    m = _TARGET_RE.match(target.strip())
    if not m:
        return None
    return EntryPoint(name=name, module=m.group("module"), attr=m.group("attr"))


def render_script(ep: EntryPoint, interpreter: Path) -> bytes:
    """Render a POSIX text-launcher script for ``ep`` against ``interpreter``.

    The body matches the convention pip and the ``installer`` package
    both produce: import the entry-point attribute, then call it
    inside ``sys.exit`` so its return code becomes the process exit
    code.
    """
    interp = str(interpreter)
    body = (
        f"#!{interp}\n"
        f"# -*- coding: utf-8 -*-\n"
        f"import re\n"
        f"import sys\n"
        f"from {ep.module} import {ep.attr}\n"
        f"if __name__ == '__main__':\n"
        f"    sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0])\n"
        f"    sys.exit({ep.attr}())\n"
    )
    return body.encode("utf-8")


def neutralized_script(content: bytes) -> bytes:
    """Restore the placeholder ``#!python`` shebang on a script body.

    A wheel ships verbatim scripts (the ones with no matching entry
    point) with a literal ``#!python`` shebang; the installer rewrites
    it to an absolute interpreter path at install time. Repackaging
    needs the placeholder back so the rebuilt wheel installs into any
    target.

    Only installer-written shebangs are rewritten. An installer always
    writes an absolute interpreter path whose basename contains
    ``python``; an author-written shebang like ``#!/usr/bin/env python3``
    or any non-Python shebang is the script's own concern, and passes
    through unchanged.
    """
    if not content.startswith(b"#!"):
        return content
    newline = content.find(b"\n")
    if newline == -1:
        return content
    shebang_line = content[2:newline]
    tokens = shebang_line.split()
    if not tokens or b"python" not in tokens[0].lower():
        return content
    line_ending = (
        b"\r\n"
        if newline > 0 and content[newline - 1 : newline] == b"\r"
        else b"\n"
    )
    return b"#!python" + line_ending + content[newline + 1 :]
