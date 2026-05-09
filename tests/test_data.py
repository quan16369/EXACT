from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from exact.data import normalize_records


class DataTests(unittest.TestCase):
    def test_logic_record_is_flattened_by_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "logic.jsonl"
            path.write_text(
                '{"id":"r1","premises-NL":["A."],"questions":["Q1?","Q2?"],"answers":["Yes","No"],"explanation":["E1","E2"]}\n',
                encoding="utf-8",
            )
            examples = normalize_records([path])
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].example_id, "r1_q0")
        self.assertEqual(examples[1].answer, "No")


if __name__ == "__main__":
    unittest.main()

