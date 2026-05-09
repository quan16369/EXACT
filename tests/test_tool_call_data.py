from __future__ import annotations

import unittest

from exact.schema import ExactExample
from exact.tool_call_data import make_tool_call_completion


class ToolCallDataTests(unittest.TestCase):
    def test_physics_tool_call_completion_contains_tool_and_final(self) -> None:
        example = ExactExample(
            example_id="physics_demo",
            task="physics",
            question="Calculate the energy stored in a capacitor when C = 0.1 F and U = 30 V.",
            answer="45",
            explanation="",
            unit="J",
        )
        completion = make_tool_call_completion(example)
        self.assertIsNotNone(completion)
        assert completion is not None
        self.assertIn('"tool":"python"', completion)
        self.assertIn("<|im_start|>tool name=python", completion)
        self.assertIn('"answer":"45"', completion)

    def test_can_fallback_to_gold_answer(self) -> None:
        example = ExactExample(
            example_id="physics_demo",
            task="physics",
            question="Write an algebraic expression representing distance after x hours.",
            answer="v*x",
            explanation="Distance equals speed times time.",
            unit="",
        )
        completion = make_tool_call_completion(example, fallback_to_gold_answer=True)
        self.assertIsNotNone(completion)
        assert completion is not None
        self.assertIn("Fallback calculation target", completion)
        self.assertIn('"answer":"v*x"', completion)


if __name__ == "__main__":
    unittest.main()
