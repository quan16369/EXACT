from __future__ import annotations

import json
import unittest

from exact.schema import ExactExample
from exact.tokenization import WhitespaceTokenizer, iter_manifest_records


class TokenizationTests(unittest.TestCase):
    def test_completion_only_masks_prompt(self) -> None:
        tokenizer = WhitespaceTokenizer()
        example = ExactExample(
            example_id="p1",
            task="physics",
            question="Find current.",
            answer="2",
            explanation="Use Ohm law.",
            unit="A",
        )
        record = next(
            iter_manifest_records(
                [example],
                tokenizer,
                max_seq_len=256,
            )
        )
        mask = json.loads(record.mask_json)
        self.assertGreater(mask.count(0), 0)
        self.assertGreater(mask.count(1), 0)
        self.assertEqual(record.num_loss_tokens, mask.count(1))

    def test_logic_example_is_not_duplicated(self) -> None:
        tokenizer = WhitespaceTokenizer()
        example = ExactExample(
            example_id="l1",
            task="logic",
            question="Can it pass?",
            answer="No",
            explanation="A premise blocks it.",
            premises_nl=["If A then B.", "A."],
        )
        records = list(
            iter_manifest_records(
                [example],
                tokenizer,
                max_seq_len=256,
            )
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].segment, "exact_logic.jsonl")


if __name__ == "__main__":
    unittest.main()
