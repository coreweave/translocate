# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: MIT
# SPDX-PackageName: translocate

"""Render the ``pyproject.toml`` snippet that configures a wheel
directory as a uv flat index.

The snippet declares one ``[[tool.uv.index]]`` entry and one
``[tool.uv.sources]`` line per package. The index is marked
``explicit = true``, so uv consults it only for the packages that
name it. Merged into a project's ``pyproject.toml``, the snippet
makes ``uv lock`` resolve those packages from the wheel directory at
their exact built versions.
"""

from __future__ import annotations

from typing import Sequence

_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def _toml_basic_string(value: str) -> str:
    """Quote ``value`` as a TOML basic string.

    Escapes the characters TOML basic strings cannot hold verbatim:
    backslash, double quote, and the C0 control characters.
    """
    out = ['"']
    for ch in value:
        if ch in _ESCAPES:
            out.append(_ESCAPES[ch])
        elif ord(ch) < 0x20 or ch == "\x7f":
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def render_uv_config(
    index_url: str,
    package_names: Sequence[str],
    index_name: str = "translocated",
) -> str:
    """Render the uv index-and-sources snippet as TOML text.

    ``index_url`` is where uv will read the flat index from,
    typically ``file://`` plus the absolute path of the wheel
    directory. Each name in ``package_names`` becomes a bare key
    under ``[tool.uv.sources]``, so pass PEP 503 canonical names.
    """
    lines = [
        "[[tool.uv.index]]",
        f"name = {_toml_basic_string(index_name)}",
        f"url = {_toml_basic_string(index_url)}",
        'format = "flat"',
        "explicit = true",
        "",
        "[tool.uv.sources]",
    ]
    index_ref = _toml_basic_string(index_name)
    for name in package_names:
        lines.append(f"{name} = {{ index = {index_ref} }}")
    return "\n".join(lines) + "\n"
