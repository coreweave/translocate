# SPDX-FileCopyrightText: 2026 CoreWeave
#
# SPDX-License-Identifier: MIT

"""Script shebang neutralization and entry-point target parsing."""

from __future__ import annotations

import unittest

from translocate.scripts import neutralized_script, parse_target


class NeutralizedScriptTests(unittest.TestCase):
    def test_neutralizes_absolute_python_shebang(self) -> None:
        body = b"#!/usr/bin/python\nimport sys\n"
        self.assertEqual(
            neutralized_script(body),
            b"#!python\nimport sys\n",
        )

    def test_preserves_crlf_line_endings(self) -> None:
        # Mixed-ending scripts must come back with the same line
        # ending on the rewritten shebang as the rest of the file
        # uses, or the script becomes ambiguously terminated.
        body = b"#!/usr/bin/python\r\nimport sys\r\n"
        self.assertEqual(
            neutralized_script(body),
            b"#!python\r\nimport sys\r\n",
        )

    def test_case_insensitive_python_match(self) -> None:
        body = b"#!/usr/local/bin/Python3\nimport sys\n"
        self.assertEqual(
            neutralized_script(body),
            b"#!python\nimport sys\n",
        )

    def test_leaves_env_python_alone(self) -> None:
        # ``#!/usr/bin/env python3`` is author-written; installers
        # write absolute interpreter paths. An author-written shebang
        # is intentional and the wheel should preserve it verbatim.
        body = b"#!/usr/bin/env python3\nimport sys\n"
        self.assertEqual(neutralized_script(body), body)

    def test_leaves_perl_shebang_alone(self) -> None:
        # ``.data/scripts/`` may carry non-Python scripts (perl,
        # bash, etc.) verbatim. We rewrite only shebangs the
        # source installer wrote for us.
        body = b"#!/usr/bin/perl\nprint 'hi';\n"
        self.assertEqual(neutralized_script(body), body)

    def test_leaves_bash_shebang_alone(self) -> None:
        body = b"#!/bin/bash\necho hi\n"
        self.assertEqual(neutralized_script(body), body)

    def test_no_shebang_passes_through(self) -> None:
        body = b"import sys\n"
        self.assertEqual(neutralized_script(body), body)


class ParseTargetTests(unittest.TestCase):
    def test_simple_module_attr(self) -> None:
        ep = parse_target("foo", "pkg:main")
        assert ep is not None
        self.assertEqual(ep.module, "pkg")
        self.assertEqual(ep.attr, "main")

    def test_dotted_module_and_attr(self) -> None:
        ep = parse_target("foo", "pkg.sub.module:cls.method")
        assert ep is not None
        self.assertEqual(ep.module, "pkg.sub.module")
        self.assertEqual(ep.attr, "cls.method")

    def test_rejects_double_dot(self) -> None:
        self.assertIsNone(parse_target("foo", "pkg..mod:main"))

    def test_rejects_leading_dot(self) -> None:
        self.assertIsNone(parse_target("foo", ".pkg:main"))

    def test_rejects_trailing_dot(self) -> None:
        self.assertIsNone(parse_target("foo", "pkg.:main"))


if __name__ == "__main__":
    unittest.main()
