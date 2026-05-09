from __future__ import annotations

import unittest

from exact.tools import PythonTool, PythonToolError


class PythonToolTests(unittest.TestCase):
    def test_prints_last_expression(self) -> None:
        result = PythonTool().execute("x = 2 + 3\nx")
        self.assertTrue(result.ok)
        self.assertEqual(result.stdout.strip(), "5")

    def test_rejects_dangerous_import(self) -> None:
        with self.assertRaises(PythonToolError):
            PythonTool().execute("import os\nprint(os.getcwd())")


if __name__ == "__main__":
    unittest.main()

