# SPDX-FileCopyrightText: 2026 CoreWeave
#
# SPDX-License-Identifier: MIT

"""Rendering of the uv index-and-sources pyproject snippet.

Exact-text assertions pin correctness and run on every supported
Python version. On 3.11+, the output is also parsed back with
``tomllib``.
"""

from __future__ import annotations

import sys
import unittest

from translocate.uvconfig import render_uv_config


class RenderTests(unittest.TestCase):
    def test_exact_shape(self) -> None:
        text = render_uv_config("file:///idx", ["torch", "triton"])
        self.assertEqual(
            text,
            "[[tool.uv.index]]\n"
            'name = "translocated"\n'
            'url = "file:///idx"\n'
            'format = "flat"\n'
            "explicit = true\n"
            "\n"
            "[tool.uv.sources]\n"
            'torch = { index = "translocated" }\n'
            'triton = { index = "translocated" }\n',
        )

    def test_custom_index_name(self) -> None:
        text = render_uv_config("file:///idx", ["torch"], index_name="my-index")
        self.assertIn('name = "my-index"', text)
        self.assertIn('torch = { index = "my-index" }', text)

    def test_no_packages_still_renders_valid_tables(self) -> None:
        text = render_uv_config("file:///idx", [])
        self.assertIn("[[tool.uv.index]]", text)
        self.assertTrue(text.endswith("[tool.uv.sources]\n"))

    def test_quote_and_backslash_escaped(self) -> None:
        text = render_uv_config('file:///pa"th\\x', ["a"])
        self.assertIn('url = "file:///pa\\"th\\\\x"', text)

    def test_control_characters_escaped(self) -> None:
        text = render_uv_config("file:///a\nb\tc", ["a"])
        self.assertIn('url = "file:///a\\nb\\tc"', text)

    @unittest.skipUnless(
        sys.version_info >= (3, 11), "tomllib requires Python 3.11"
    )
    def test_round_trips_through_tomllib(self) -> None:
        import tomllib

        data = tomllib.loads(
            render_uv_config('file:///we"ird\\dir', ["torch", "triton"])
        )
        index = data["tool"]["uv"]["index"][0]
        self.assertEqual(index["name"], "translocated")
        self.assertEqual(index["url"], 'file:///we"ird\\dir')
        self.assertEqual(index["format"], "flat")
        self.assertTrue(index["explicit"])
        self.assertEqual(
            data["tool"]["uv"]["sources"],
            {
                "torch": {"index": "translocated"},
                "triton": {"index": "translocated"},
            },
        )


if __name__ == "__main__":
    unittest.main()
